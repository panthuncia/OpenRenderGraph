#include "OpenRenderGraph/PersistentGraphHost.h"

#include <algorithm>
#include <stdexcept>

#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Render/PassExecutionContext.h"
#include "Render/Runtime/IUploadService.h"
#include "Render/Runtime/RuntimeDevice.h"
#include "Render/Runtime/ScopedActiveGraphServices.h"
#include "Render/Runtime/ThreadPoolTaskService.h"

namespace org {

namespace {

constexpr const char* kUploadPassName = "org.host.uploads";

// The built-in upload pass: graph-ordered buffer/texture uploads requested with
// BUFFER_UPLOAD and friends. It records immediate-mode copies, so it runs in the
// persistent Pre segment ahead of the main executable.
class HostUploadExtension final : public RenderGraph::IRenderGraphExtension {
public:
	explicit HostUploadExtension(runtime::IUploadService* uploads) : m_uploads(uploads) {}
	void OnRegistryReset(ResourceRegistry* registry) override {
		if (m_uploads) m_uploads->SetUploadResolveContext({registry, 0});
	}
	void GatherStructuralPasses(RenderGraph&, std::vector<RenderGraph::ExternalPassDesc>& out) override {
		if (!m_uploads) return;
		if (auto pass = m_uploads->GetUploadPass())
			out.push_back(RenderGraph::ExternalPassDesc::Render(kUploadPassName, std::move(pass))
				.At(RenderGraph::ExternalInsertPoint::Begin(-1000)).CollectStatistics(false));
	}
private:
	runtime::IUploadService* m_uploads;
};

} // namespace

PersistentGraphHost::PersistentGraphHost(Desc desc) : m_desc(std::move(desc)) {
	if (!m_desc.device) throw std::invalid_argument("PersistentGraphHost requires a device");
	if (m_desc.backend == rhi::Backend::Null) throw std::invalid_argument("PersistentGraphHost requires a device backend");
	if (!m_desc.tasks) m_desc.tasks = std::make_shared<runtime::ThreadPoolTaskService>();
	runtime::InitializeRuntimeDevice(m_desc.device);
	if (m_desc.device.CreateTimeline(m_frameTimeline, 0, "ORG host frames") != rhi::Result::Ok)
		throw std::runtime_error("PersistentGraphHost could not create its frame timeline");
	m_slotFrameValues.assign((std::max)(m_desc.framesInFlight, 1u), 0);
}

PersistentGraphHost::~PersistentGraphHost() {
	DestroyGraph();
	runtime::ShutdownRuntimeDevice();
}

void PersistentGraphHost::AddExtension(std::string id, ExtensionFactory factory) {
	RemoveExtension(id);
	m_extensions.push_back({std::move(id), std::move(factory)});
	m_rebuildRequested = true;
}

void PersistentGraphHost::RemoveExtension(const std::string& id) {
	const auto erased = std::erase_if(m_extensions, [&](const Registered& entry) { return entry.id == id; });
	if (erased) m_rebuildRequested = true;
}

runtime::IUploadService* PersistentGraphHost::Uploads() noexcept {
	return m_graph ? m_graph->GetUploadService() : nullptr;
}

void PersistentGraphHost::DestroyGraph() {
	if (!m_graph) return;
	m_graph->StopFrameProduction();
	// Adopted devices wait on their own timelines only (never the whole device).
	(void)m_desc.device.WaitIdle();
	m_graph->ShutdownExtensions();
	m_graph->ShutdownTaskWorkers();
	m_graph.reset();
}

void PersistentGraphHost::Build() {
	DestroyGraph();
	auto graph = std::make_unique<RenderGraph>(m_desc.device, m_desc.backend);
	graph->SetTaskService(m_desc.tasks);
	graph->RegisterExtension(std::make_unique<HostUploadExtension>(graph->GetUploadService()), "org.host.io");
	for (const auto& entry : m_extensions)
		if (auto extension = entry.factory()) graph->RegisterExtension(std::move(extension), entry.id);
	runtime::ScopedActiveGraphServices services(graph->GetUploadService(), graph->GetDescriptorService());
	graph->PrepareExtensionsForBuild();
	graph->CompileStructural();
	graph->SetPersistentSegment(kUploadPassName, RenderGraph::PersistentSegmentKind::Pre);
	graph->SetPersistentExecutionEnabled(true);
	graph->SetExternalQueueBoundary(m_desc.queueBoundary);
	graph->Setup();
	m_graph = std::move(graph);
	m_rebuildRequested = false;
}

void PersistentGraphHost::ExecuteFrame(const IHostExecutionData* hostData, const FrameCallback& beforePrepare) {
	if (m_rebuildRequested || !m_graph) Build();
	runtime::ScopedActiveGraphServices services(m_graph->GetUploadService(), m_graph->GetDescriptorService());
	const auto slot = static_cast<uint32_t>(m_frameNumber % (std::max)(m_desc.framesInFlight, 1u));
	// The slot's previous frame must be done on the GPU before the upload pages it used are recycled
	// (and before this frame records uploads into the slot).
	if (m_slotFrameValues[slot]) {
		(void)m_frameTimeline->HostWait(m_slotFrameValues[slot]);
		// Nothing else completes a frame's statistics on this path: without it the pass timestamps were
		// written and resolved every frame and never read, and their pending resolves only grew.
		if (auto* stats = m_graph->GetStatisticsService()) {
			rhi::Queue queue = m_desc.device.GetQueue(rhi::QueueKind::Graphics);
			stats->OnFrameComplete(slot, queue);
			if (m_completedFrame) m_completedFrame(m_slotFrameValues[slot] - 1, *stats);
		}
	}
	if (auto* uploads = m_graph->GetUploadService()) uploads->ProcessDeferredReleases(static_cast<uint8_t>(slot));
	DescriptorHeapManager::GetInstance().ProcessDeferredReleases(static_cast<uint8_t>(slot));
	if (beforePrepare) beforePrepare(*m_graph);
	UpdateExecutionContext update{};
	update.frameIndex = slot;
	update.preparationSlot = slot;
	update.frameFenceValue = m_frameNumber + 1;
	update.hostData = hostData;
	m_graph->Update(update, m_desc.device);
	PassExecutionContext execute{};
	execute.device = m_desc.device;
	execute.frameIndex = slot;
	execute.frameFenceValue = m_frameNumber + 1;
	execute.hostData = hostData;
	m_graph->Execute(execute);
	// Execute submitted the frame; this signal follows it on the same queue.
	const uint64_t frameValue = m_frameNumber + 1;
	if (m_desc.device.GetQueue(rhi::QueueKind::Graphics).Signal({ m_frameTimeline->GetHandle(), frameValue }) == rhi::Result::Ok)
		m_slotFrameValues[slot] = frameValue;
	++m_frameNumber;
}

} // namespace org
