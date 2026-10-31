// GraphDesc, JobGraph and GraphCache: small graphs built once and executed with per-execution bindings, each on a
// GraphInstance of its own. docs/job-graphs.md, "Graph descs".
#include "OpenRenderGraph/GraphDesc.h"
#include "OpenRenderGraph/GraphInstance.h"

#include <BasicTelemetry/Tracy.h>
#include <stdexcept>

namespace org {

namespace {
// Marks a job graph busy while its lock is held.
struct BusyScope {
	std::atomic<bool>& busy;
	explicit BusyScope(std::atomic<bool>& flag) : busy(flag) { busy.store(true, std::memory_order_release); }
	~BusyScope() { busy.store(false, std::memory_order_release); }
};
}

std::shared_ptr<DynamicResource> GraphDesc::Parameter(std::string name, std::shared_ptr<Resource> initial) {
	if (!initial) throw std::invalid_argument("GraphDesc::Parameter: '" + name + "' needs an initial resource");
	if (m_parameters.contains(name)) throw std::invalid_argument("GraphDesc::Parameter: '" + name + "' added twice");
	auto wrapper = std::make_shared<DynamicResource>(std::move(initial));
	wrapper->SetName(m_name + "." + name);
	m_parameters.emplace(std::move(name), wrapper);
	return wrapper;
}

GraphDesc& GraphDesc::Add(RenderGraph::ExternalPassDesc pass) {
	m_passes.push_back(std::move(pass));
	return *this;
}

std::shared_ptr<JobGraph> JobGraph::Create(std::shared_ptr<Runtime> runtime, GraphDesc desc) {
	BT_ZONE_SCOPE("ORG.JobGraph.Create");
	std::shared_ptr<JobGraph> graph(new JobGraph());
	graph->m_instance = GraphInstance::Create(std::move(runtime), std::move(desc));
	return graph;
}

JobGraph::~JobGraph() = default;

ExecutionHandle JobGraph::Execute(std::span<const Binding> bindings, const IHostExecutionData* hostData) {
	BT_ZONE_SCOPE("ORG.JobGraph.Execute");
	std::scoped_lock lock(m_mutex);
	BusyScope busy(m_busy);
	m_instance->Bind(bindings);
	return m_instance->Prepare(hostData).Submit();
}

void JobGraph::Retire() {
	std::scoped_lock lock(m_mutex);
	BusyScope busy(m_busy);
	m_instance->Retire();
}

bool JobGraph::TryRetire() {
	if (m_busy.load(std::memory_order_acquire)) return false;
	std::unique_lock lock(m_mutex, std::try_to_lock);
	if (!lock) return false;
	BusyScope busy(m_busy);
	m_instance->Retire();
	return true;
}

const std::string& JobGraph::Name() const noexcept { return m_instance->Name(); }
uint64_t JobGraph::Key() const noexcept { return m_instance->Key(); }
RenderGraph* JobGraph::Graph() noexcept { return m_instance->Graph(); }

std::shared_ptr<JobGraph> GraphCache::GetOrCompile(GraphDesc desc) {
	const auto key = desc.Key();
	if (key) {
		std::scoped_lock lock(m_mutex);
		if (const auto found = m_graphs.find(key); found != m_graphs.end()) {
			++m_hits;
			return found->second;
		}
	}
	// Built outside the lock, so other keys are not held up. Two callers building one key keep the first.
	auto built = JobGraph::Create(m_runtime, std::move(desc));
	std::scoped_lock lock(m_mutex);
	++m_compiles;
	if (!key) return built;
	return m_graphs.try_emplace(key, std::move(built)).first->second;
}

void GraphCache::Clear() {
	std::scoped_lock lock(m_mutex);
	m_graphs.clear();
}

GraphCache::Stats GraphCache::GetStats() const {
	std::scoped_lock lock(m_mutex);
	return {m_hits, m_compiles, m_graphs.size()};
}

} // namespace org
