// GraphInstance, Execution and CompiledGraph: a desc's graph driven one execution at a time, without GraphHost's frame
// ring. docs/job-graphs.md, "Graph instances".
#include "OpenRenderGraph/GraphInstance.h"

#include <BasicTelemetry/Tracy.h>
#include <limits>
#include <stdexcept>

#include "HostUploadExtension.h"
#include "OpenRenderGraph/Runtime.h"
#include "Render/MemoryIntrospectionBackend.h"
#include "Render/PassExecutionContext.h"
#include "Render/RenderGraph/GraphCompiler.h"
#include "Render/RenderGraph/PersistentGraph.h"
#include "Render/Runtime/IUploadService.h"
#include "Render/Runtime/ScopedActiveGraphServices.h"

namespace org {

namespace {
// Adds a desc's passes to the graph, in the order they were added.
class DescExtension final : public RenderGraph::IRenderGraphExtension {
public:
	explicit DescExtension(std::vector<RenderGraph::ExternalPassDesc> passes) : m_passes(std::move(passes)) {}
	void GatherStructuralPasses(RenderGraph&, std::vector<RenderGraph::ExternalPassDesc>& out) override {
		out.insert(out.end(), m_passes.begin(), m_passes.end());
	}
private:
	std::vector<RenderGraph::ExternalPassDesc> m_passes;
};
}

uint64_t CompiledGraph::StructuralRevision() const noexcept { return m_executable ? m_executable->structuralRevision : 0; }

size_t CompiledGraph::BatchCount() const noexcept {
	return m_executable && m_executable->graph ? m_executable->graph->batches.size() : 0;
}

Execution::Execution(Execution&& other) noexcept
	: m_instance(std::exchange(other.m_instance, nullptr)), m_index(other.m_index), m_slot(other.m_slot),
	m_hostData(other.m_hostData), m_submitted(std::exchange(other.m_submitted, true)) {}

Execution::~Execution() {
	if (!m_instance || m_submitted || m_instance->m_adopted) return;
	try { Submit(); } catch (...) {}
}

ExecutionHandle Execution::Submit() {
	if (!m_instance) throw std::logic_error("Execution::Submit on a moved-from execution");
	if (m_submitted) throw std::logic_error("Execution::Submit: already submitted");
	m_submitted = true;
	return m_instance->SubmitPrepared(*this, nullptr);
}

ExecutionHandle Execution::Submit(PassExecutionContext& context) {
	if (!m_instance) throw std::logic_error("Execution::Submit on a moved-from execution");
	if (m_submitted) throw std::logic_error("Execution::Submit: already submitted");
	m_submitted = true;
	return m_instance->SubmitPrepared(*this, &context);
}

std::unique_ptr<GraphInstance> GraphInstance::Create(std::shared_ptr<Runtime> runtime, GraphDesc desc) {
	BT_ZONE_SCOPE("ORG.GraphInstance.Create");
	if (!runtime) throw std::invalid_argument("GraphInstance::Create needs a runtime");
	const uint32_t frames = (std::max)(desc.m_framesInFlight, 1u);
	if (frames > runtime->Slots())
		throw std::invalid_argument("GraphInstance '" + desc.m_name + "': more frames in flight than the runtime has slots");
	std::unique_ptr<GraphInstance> instance(new GraphInstance());
	instance->m_runtime = runtime;
	instance->m_name = std::move(desc.m_name);
	instance->m_key = desc.m_key;
	instance->m_parameters = std::move(desc.m_parameters);
	instance->m_slots.resize(frames);
	// A graph's own uploads: one built while another graph is being built must not point the shared service at its
	// registry.
	instance->m_uploads = runtime::CreateGraphUploadService();
	instance->m_uploads->Initialize();
	auto graph = std::make_unique<RenderGraph>(runtime);
	graph->SetUploadService(instance->m_uploads);
	graph->SetDrivesFrameServices(false);
	graph->RegisterExtension(std::make_unique<detail::HostUploadExtension>(graph->GetUploadService()), "org.host.io");
	graph->RegisterExtension(std::make_unique<DescExtension>(std::move(desc.m_passes)), "org.graph-desc");
	runtime::ScopedActiveGraphServices services(graph->GetUploadService(), graph->GetDescriptorService());
	graph->PrepareExtensionsForBuild();
	graph->GetMemorySnapshotProvider().SetProvider(memory::CreateECSMemorySnapshotProvider());
	graph->CompileStructural();
	graph->Setup();
	instance->m_graph = std::move(graph);
	return instance;
}

std::unique_ptr<GraphInstance> GraphInstance::Adopt(std::shared_ptr<Runtime> runtime, std::unique_ptr<RenderGraph> graph) {
	if (!runtime || !graph) throw std::invalid_argument("GraphInstance::Adopt needs a runtime and a graph");
	std::unique_ptr<GraphInstance> instance(new GraphInstance());
	instance->m_runtime = std::move(runtime);
	instance->m_name = "adopted";
	instance->m_graph = std::move(graph);
	instance->m_adopted = true;
	return instance;
}

void GraphInstance::Rebuild(const std::function<void(RenderGraph&)>& declare) {
	BT_ZONE_SCOPE("ORG.GraphInstance.Rebuild");
	if (!m_adopted) throw std::logic_error("GraphInstance::Rebuild: only an adopted graph is rebuilt by its app");
	m_pending = false;
	m_graph->ResetForRebuild();
	if (declare) declare(*m_graph);
	m_graph->CompileStructural();
	m_graph->Setup();
}

Execution GraphInstance::Prepare(const UpdateExecutionContext& context) {
	BT_ZONE_SCOPE("ORG.GraphInstance.Prepare");
	if (!m_adopted) throw std::logic_error("GraphInstance::Prepare(context): only an adopted graph takes the app's context");
	m_graph->Update(context, m_runtime->Device());
	m_pending = true;
	Execution execution(*this, ++m_prepared, context.preparationSlot);
	execution.m_hostData = context.hostData;
	return execution;
}

GraphInstance::~GraphInstance() {
	if (m_adopted) {
		m_graph.reset();
		return;
	}
	if (m_graph) {
		m_graph->StopFrameProduction();
		// Its own work only: what each slot's last execution submitted on its queues. Its external client (the uploads)
		// waits for its own submissions when the graph destroys it.
		{
			BT_ZONE_SCOPE("ORG.GraphInstance.WaitForExecutions");
			for (const auto& slot : m_slots)
				if (slot.busy)
					for (const auto& [timeline, value] : slot.points) (void)timeline->HostWait(value);
		}
		m_graph->RetirePersistentExecutions();
		m_graph->ShutdownExtensions();
		m_graph->ShutdownTaskWorkers();
		m_graph.reset();
	}
	if (m_uploads) m_uploads->Cleanup();
	m_runtime->Retire();
}

void GraphInstance::Bind(std::span<const GraphBinding> bindings) {
	for (const auto& binding : bindings) {
		const auto found = m_parameters.find(binding.parameter);
		if (found == m_parameters.end())
			throw std::invalid_argument("GraphInstance '" + m_name + "' has no parameter '" + std::string(binding.parameter) + "'");
		if (!binding.resource)
			throw std::invalid_argument("GraphInstance '" + m_name + "': parameter '" + std::string(binding.parameter) + "' bound to null");
		found->second->SetResource(binding.resource);
	}
}

namespace {
bool Reached(const std::vector<std::pair<rhi::Timeline*, uint64_t>>& points) {
	for (const auto& [timeline, value] : points)
		if (timeline->GetCompletedValue() < value) return false;
	return true;
}
}

uint32_t GraphInstance::AcquireSlot() {
	Retire();
	uint32_t oldest = 0;
	uint64_t oldestSubmitted = (std::numeric_limits<uint64_t>::max)();
	for (uint32_t index = 0; index < m_slots.size(); ++index) {
		auto& slot = m_slots[index];
		if (!slot.busy || Reached(slot.points)) {
			slot.busy = false;
			return index;
		}
		if (slot.submitted < oldestSubmitted) { oldestSubmitted = slot.submitted; oldest = index; }
	}
	BT_ZONE_SCOPE("ORG.GraphInstance.WaitSlot");
	auto& slot = m_slots[oldest];
	for (const auto& [timeline, value] : slot.points) (void)timeline->HostWait(value);
	slot.busy = false;
	Retire();
	return oldest;
}

bool GraphInstance::CanPrepare() {
	Retire();
	for (const auto& slot : m_slots)
		if (!slot.busy || Reached(slot.points)) return true;
	return false;
}

Execution GraphInstance::Prepare(const IHostExecutionData* hostData) {
	BT_ZONE_SCOPE("ORG.GraphInstance.Prepare");
	if (m_adopted) throw std::logic_error("GraphInstance::Prepare: an adopted graph is prepared with the app's context");
	if (m_pending) throw std::logic_error("GraphInstance '" + m_name + "': an execution is prepared and not submitted");
	const auto slot = AcquireSlot();
	runtime::ScopedActiveGraphServices services(m_graph->GetUploadService(), m_graph->GetDescriptorService());
	// The slot's last execution is done on the GPU, so the upload pages it used can be recycled.
	m_uploads->ProcessDeferredReleases(static_cast<uint32_t>(slot));
	m_runtime->Retire();
	const auto index = ++m_prepared;
	UpdateExecutionContext update{};
	update.frameIndex = slot;
	update.preparationSlot = slot;
	update.frameFenceValue = index;
	update.hostData = hostData;
	m_graph->Update(update, m_runtime->Device());
	m_pending = true;
	Execution execution(*this, index, slot);
	execution.m_hostData = hostData;
	return execution;
}

ExecutionHandle GraphInstance::SubmitPrepared(Execution& execution, PassExecutionContext* appContext) {
	BT_ZONE_SCOPE("ORG.GraphInstance.Submit");
	struct PendingScope {
		bool& pending;
		~PendingScope() { pending = false; }
	} pendingScope{m_pending};
	PassExecutionContext fallback{};
	fallback.device = m_runtime->Device();
	fallback.frameIndex = execution.m_slot;
	fallback.frameFenceValue = execution.m_index;
	fallback.hostData = execution.m_hostData;
	auto& context = appContext ? *appContext : fallback;
	if (m_adopted) {
		// The app's frame slots, services and completion: nothing of the instance's to track.
		m_graph->Execute(context);
		m_last = m_graph->LastExecution();
		return m_last;
	}
	runtime::ScopedActiveGraphServices services(m_graph->GetUploadService(), m_graph->GetDescriptorService());
	m_graph->Execute(context);
	// Every batch signalled its queue's timeline: the value each queue has reached is the execution's completion there.
	auto& slot = m_slots[execution.m_slot];
	slot.points.clear();
	auto& queues = m_graph->GetQueueRegistry();
	for (size_t index = 0; index < queues.SlotCount(); ++index) {
		const auto queueSlot = static_cast<QueueSlotIndex>(static_cast<uint8_t>(index));
		if (const auto next = queues.GetCurrentFenceValue(queueSlot); next > 1) slot.points.emplace_back(&queues.GetFence(queueSlot), next - 1);
	}
	slot.submitted = ++m_submissions;
	slot.busy = true;
	m_last = m_graph->LastExecution();
	slot.execution = m_last;
	return m_last;
}

void GraphInstance::Retire() {
	if (m_graph) m_graph->RetirePersistentExecutions();
}

std::shared_ptr<const CompiledGraph> GraphInstance::Compiled() const {
	auto executable = m_graph ? m_graph->SelectedExecutable() : nullptr;
	if (!executable) return nullptr;
	return std::shared_ptr<const CompiledGraph>(new CompiledGraph(std::move(executable)));
}

ExecutionHandle GraphInstance::LastExecution() const { return m_last; }

} // namespace org
