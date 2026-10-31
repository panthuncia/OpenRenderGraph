// GraphDesc, CompiledGraph and GraphCache: small graphs compiled once and executed with per-execution bindings.
// A compiled graph runs on a GraphHost of its own for now; docs/job-graphs.md, "Graph descs".
#include "OpenRenderGraph/GraphDesc.h"
#include "OpenRenderGraph/GraphHost.h"

#include <BasicTelemetry/Tracy.h>
#include <stdexcept>

namespace org {

namespace {
// Marks a compiled graph busy while its lock is held.
struct BusyScope {
	std::atomic<bool>& busy;
	explicit BusyScope(std::atomic<bool>& flag) : busy(flag) { busy.store(true, std::memory_order_release); }
	~BusyScope() { busy.store(false, std::memory_order_release); }
};

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

std::shared_ptr<CompiledGraph> CompiledGraph::Compile(std::shared_ptr<Runtime> runtime, GraphDesc desc) {
	BT_ZONE_SCOPE("ORG.CompiledGraph.Compile");
	if (!runtime) throw std::invalid_argument("CompiledGraph::Compile needs a runtime");
	std::shared_ptr<CompiledGraph> graph(new CompiledGraph());
	graph->m_name = std::move(desc.m_name);
	graph->m_key = desc.m_key;
	graph->m_parameters = std::move(desc.m_parameters);
	graph->m_host = std::make_unique<GraphHost>(GraphHost::Desc{.runtime = std::move(runtime), .framesInFlight = desc.m_framesInFlight});
	graph->m_host->AddExtension("org.graph-desc", [passes = std::move(desc.m_passes)] { return std::make_unique<DescExtension>(passes); });
	graph->m_host->BuildIfRequested();
	return graph;
}

CompiledGraph::~CompiledGraph() = default;

ExecutionHandle CompiledGraph::Execute(std::span<const Binding> bindings, const IHostExecutionData* hostData) {
	BT_ZONE_SCOPE("ORG.CompiledGraph.Execute");
	std::scoped_lock lock(m_mutex);
	BusyScope busy(m_busy);
	for (const auto& binding : bindings) {
		const auto found = m_parameters.find(binding.parameter);
		if (found == m_parameters.end())
			throw std::invalid_argument("CompiledGraph '" + m_name + "' has no parameter '" + std::string(binding.parameter) + "'");
		if (!binding.resource)
			throw std::invalid_argument("CompiledGraph '" + m_name + "': parameter '" + std::string(binding.parameter) + "' bound to null");
		found->second->SetResource(binding.resource);
	}
	m_host->ExecuteFrame(hostData);
	return m_host->Graph()->LastExecution();
}

void CompiledGraph::Retire() {
	std::scoped_lock lock(m_mutex);
	BusyScope busy(m_busy);
	if (auto* graph = m_host->Graph()) graph->RetirePersistentExecutions();
}

bool CompiledGraph::TryRetire() {
	if (m_busy.load(std::memory_order_acquire)) return false;
	std::unique_lock lock(m_mutex, std::try_to_lock);
	if (!lock) return false;
	BusyScope busy(m_busy);
	if (auto* graph = m_host->Graph()) graph->RetirePersistentExecutions();
	return true;
}

RenderGraph* CompiledGraph::Graph() noexcept { return m_host ? m_host->Graph() : nullptr; }

std::shared_ptr<CompiledGraph> GraphCache::GetOrCompile(GraphDesc desc) {
	const auto key = desc.Key();
	if (key) {
		std::scoped_lock lock(m_mutex);
		if (const auto found = m_graphs.find(key); found != m_graphs.end()) {
			++m_hits;
			return found->second;
		}
	}
	// Compiled outside the lock, so other keys are not held up. Two callers compiling one key keep the first.
	auto compiled = CompiledGraph::Compile(m_runtime, std::move(desc));
	std::scoped_lock lock(m_mutex);
	++m_compiles;
	if (!key) return compiled;
	return m_graphs.try_emplace(key, std::move(compiled)).first->second;
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
