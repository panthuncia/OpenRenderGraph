#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "OpenRenderGraph/ExecutionHandle.h"
#include "OpenRenderGraph/Runtime.h"
#include "Render/RenderGraph/RenderGraph.h"
#include "Resources/DynamicResource.h"

namespace org {

class GraphInstance;
struct IHostExecutionData;

// A resource bound to a desc's parameter, by name.
struct GraphBinding {
	std::string_view parameter;
	std::shared_ptr<Resource> resource;
};

// What a small graph is made of: its passes, in order, and its parameters. A parameter is a resource the graph is
// bound to anew for each execution. Passes declare the parameter's wrapper (Parameter) instead of a concrete resource.
//
// Descs with the same key are interchangeable: GraphCache gives a later desc the job graph built from an earlier one,
// with the earlier desc's pass objects. So passes must take everything that changes between executions from the
// parameters and the host data, not from their own fields. Key 0 means never cached.
class GraphDesc {
public:
	explicit GraphDesc(std::string name, uint64_t key = 0) : m_name(std::move(name)), m_key(key) {}

	// Adds a parameter. `initial` is what it is bound to until an execution binds something else. Passes declare the
	// returned wrapper. Binding a resource of another shape later makes a structural build.
	std::shared_ptr<DynamicResource> Parameter(std::string name, std::shared_ptr<Resource> initial);
	// Adds a pass. Passes without an insert point run in the order they were added.
	GraphDesc& Add(RenderGraph::ExternalPassDesc pass);
	// Executions that may run on the GPU at once (GraphInstance's slots). Preparing one more waits on the CPU for the
	// oldest. Must not exceed the runtime's slots.
	GraphDesc& SetFramesInFlight(uint32_t frames) { m_framesInFlight = frames; return *this; }

	const std::string& Name() const noexcept { return m_name; }
	uint64_t Key() const noexcept { return m_key; }

private:
	friend class GraphInstance;
	std::string m_name;
	uint64_t m_key = 0;
	uint32_t m_framesInFlight = 3;
	std::vector<RenderGraph::ExternalPassDesc> m_passes;
	std::map<std::string, std::shared_ptr<DynamicResource>, std::less<>> m_parameters;
};

// A desc built once (as a GraphInstance) and executed any number of times, each with its own parameter bindings.
// Rebinding a parameter to another resource of the same shape is a binding edit, so the schedule, barriers and alias
// plan are reused. Executions on one runtime are ordered by the runtime's sequencer, across graphs too: an execution
// that reads what another graph's earlier execution wrote waits for it on the GPU.
//
// Any thread may call it. Executions of one job graph run one at a time; use separate job graphs (or no cache key)
// to run the same desc in parallel.
class JobGraph {
public:
	using Binding = GraphBinding;

	// Builds the graph. The first execution also lowers its passes.
	static std::shared_ptr<JobGraph> Create(std::shared_ptr<Runtime> runtime, GraphDesc desc);
	~JobGraph();
	JobGraph(const JobGraph&) = delete;
	JobGraph& operator=(const JobGraph&) = delete;

	// Binds the named parameters (the others keep their last binding), prepares, records and submits one execution.
	// Throws std::invalid_argument for an unknown parameter or a null resource. hostData reaches the passes'
	// Prepare as PassPrepareContext::preparationData, for this execution only. Waits on the CPU only when the desc's
	// frames in flight are all still running.
	ExecutionHandle Execute(std::span<const Binding> bindings = {}, const IHostExecutionData* hostData = nullptr);
	// Notices executions the GPU finished, completing their handles and running their callbacks. Execute does this
	// too; call it when the graph may not execute again for a while.
	void Retire();
	// Retire, unless an execution or another retire is running: then nothing, and false. For pollers that must not
	// block (a StateGraph's completion pump).
	bool TryRetire();

	const std::string& Name() const noexcept;
	uint64_t Key() const noexcept;
	GraphInstance& Instance() noexcept { return *m_instance; }
	// The graph itself, for diagnostics (schedule snapshots, timings). Not for driving it.
	RenderGraph* Graph() noexcept;

private:
	JobGraph() = default;
	std::mutex m_mutex;
	// Set while m_mutex is held, so TryRetire from a callback on the holding thread never locks it again.
	std::atomic<bool> m_busy{false};
	std::unique_ptr<GraphInstance> m_instance;
};

// Job graphs by desc key, on one runtime. A desc whose key was built before gets that job graph, so its schedule,
// barrier plans and alias plan are reused. Any thread.
class GraphCache {
public:
	explicit GraphCache(std::shared_ptr<Runtime> runtime) : m_runtime(std::move(runtime)) {}
	// The cached job graph of desc.Key(), or desc built (and cached unless the key is 0).
	std::shared_ptr<JobGraph> GetOrCompile(GraphDesc desc);
	void Clear();
	struct Stats { uint64_t hits = 0, compiles = 0; size_t cached = 0; };
	Stats GetStats() const;

private:
	std::shared_ptr<Runtime> m_runtime;
	mutable std::mutex m_mutex;
	std::unordered_map<uint64_t, std::shared_ptr<JobGraph>> m_graphs;
	uint64_t m_hits = 0, m_compiles = 0;
};

} // namespace org
