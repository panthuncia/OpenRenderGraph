#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "OpenRenderGraph/ExecutionHandle.h"
#include "OpenRenderGraph/GraphDesc.h"

namespace rhi { class Timeline; }

namespace org {

namespace persistent { struct ExecutableGeneration; }
namespace runtime { class IUploadService; }
class RenderGraph;
class Runtime;
struct IHostExecutionData;
class GraphInstance;

// What an instance's executions run: the schedule of its selected publication (batches, barriers, alias placement).
// Immutable. A structural edit gives the instance a new one; binding edits keep it.
class CompiledGraph {
public:
	uint64_t StructuralRevision() const noexcept;
	size_t BatchCount() const noexcept;
	// Whether executions of both run the same batches and barriers.
	bool SameSchedule(const CompiledGraph& other) const noexcept { return m_executable == other.m_executable; }

private:
	friend class GraphInstance;
	explicit CompiledGraph(std::shared_ptr<const persistent::ExecutableGeneration> executable)
		: m_executable(std::move(executable)) {}
	std::shared_ptr<const persistent::ExecutableGeneration> m_executable;
};

// One execution of an instance: prepared (GraphInstance::Prepare), then submitted. The passes' invocations and the
// uploads were recorded when it was prepared, so an execution dropped without Submit is submitted by its destructor.
class Execution {
public:
	Execution(Execution&& other) noexcept;
	Execution& operator=(Execution&& other) = delete;
	Execution(const Execution&) = delete;
	Execution& operator=(const Execution&) = delete;
	~Execution();

	// Submits its uploads, then admits its main segment after the runtime's earlier executions, records and submits it.
	// Once; the instance's thread. Throws if it was submitted already.
	ExecutionHandle Submit();
	bool Submitted() const noexcept { return m_submitted; }
	// Its index among the instance's executions (PassPrepareContext::executionIndex).
	uint64_t Index() const noexcept { return m_index; }

private:
	friend class GraphInstance;
	Execution(GraphInstance& instance, uint64_t index, uint32_t slot) : m_instance(&instance), m_index(index), m_slot(slot) {}
	GraphInstance* m_instance = nullptr;
	uint64_t m_index = 0;
	uint32_t m_slot = 0;
	const IHostExecutionData* m_hostData = nullptr;  // Prepare's: must outlive the submission
	bool m_submitted = false;
};

// A graph built from a desc, executed by its owner thread one execution at a time. It owns the pass objects, the
// registry, the slot directory and the alias pools; each execution binds the parameters as they are, prepares against
// the selected publication, and is submitted after the runtime's earlier executions. Rebinding a parameter to a
// resource of the same shape is a binding edit; anything else is a structural edit that builds on a worker while
// executions keep the current schedule.
//
// It has framesInFlight preparation slots (GraphDesc::SetFramesInFlight). A slot is free when its last execution has
// completed. Prepare waits on the CPU only when none is free (CanPrepare tells). There is no frame ring: executions
// complete through their handles, which Retire (or the next Prepare) completes.
//
// It drives no frame services: no readback polling, statistics or captures. Its uploads go through a service of its own.
class GraphInstance {
public:
	// Builds the desc's graph on the runtime. The first Prepare lowers the passes and builds the first publication.
	// Throws std::invalid_argument without a runtime, or for more frames in flight than the runtime has slots.
	static std::unique_ptr<GraphInstance> Create(std::shared_ptr<Runtime> runtime, GraphDesc desc);
	// Waits for its submitted executions on its own queues (not the whole device), then releases the graph.
	~GraphInstance();
	GraphInstance(const GraphInstance&) = delete;
	GraphInstance& operator=(const GraphInstance&) = delete;

	// Binds the named parameters for the next preparation; the others keep their last binding. Throws
	// std::invalid_argument for an unknown parameter or a null resource.
	void Bind(std::span<const GraphBinding> bindings);
	// Prepares the next execution. Throws std::logic_error while another prepared execution is not submitted.
	// hostData reaches the passes' Prepare as PassPrepareContext::preparationData, for this execution only. It must
	// live until the execution is submitted.
	Execution Prepare(const IHostExecutionData* hostData = nullptr);
	// Whether Prepare would find a free slot without waiting. Retires first.
	bool CanPrepare();
	// Notices executions the GPU finished, completing their handles. Never waits.
	void Retire();

	// The schedule the next execution runs. Null before the first preparation.
	std::shared_ptr<const CompiledGraph> Compiled() const;
	// The latest execution submitted. Empty before the first.
	ExecutionHandle LastExecution() const;
	const std::string& Name() const noexcept { return m_name; }
	uint64_t Key() const noexcept { return m_key; }
	uint32_t FramesInFlight() const noexcept { return static_cast<uint32_t>(m_slots.size()); }
	// The graph itself, for diagnostics (schedule snapshots, timings). Not for driving it.
	RenderGraph* Graph() noexcept { return m_graph.get(); }

private:
	friend class Execution;
	GraphInstance() = default;
	ExecutionHandle SubmitPrepared(Execution& execution);
	// A slot whose last execution completed; waits for the oldest when there is none.
	uint32_t AcquireSlot();

	struct Slot {
		ExecutionHandle execution;
		// Each queue's value when it was submitted, which the GPU reaches when it completes.
		std::vector<std::pair<rhi::Timeline*, uint64_t>> points;
		uint64_t submitted = 0;  // the instance's submission count then: the lowest is the oldest
		bool busy = false;
	};
	std::shared_ptr<Runtime> m_runtime;
	std::string m_name;
	uint64_t m_key = 0;
	std::map<std::string, std::shared_ptr<DynamicResource>, std::less<>> m_parameters;
	std::shared_ptr<runtime::IUploadService> m_uploads;
	std::unique_ptr<RenderGraph> m_graph;
	std::vector<Slot> m_slots;
	uint64_t m_prepared = 0, m_submissions = 0;
	bool m_pending = false;  // an execution prepared and not submitted
	ExecutionHandle m_last;
};

} // namespace org
