#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <rhi.h>

namespace org::compiler {
class PreparedRhiExecutionBatch;
}

namespace org {

/**
 * @brief An external epoch's execution (RenderGraph::SetExternalPersistentEpochs, GraphHost::TakeExternalExecution).
 * It holds what the epoch's ticket would have submitted. The host executes it in a command buffer of its own (DXVK's,
 * inside an external-commands callback) and submits it with that buffer. It contains, in order:
 * - the lists recorded ahead of it: the uploads, as a secondary list,
 * - each batch, either as a secondary list recorded ahead, or as a late batch that Execute records. A late batch is a
 *   pass that is recorded at execution.
 *
 * The host's submission of that buffer waits for `waits` and signals `signals`. Among the signals is the graph's own
 * value for the execution on its graphics timeline. The graph's later work on that timeline cannot complete until the
 * host signals it. A host that cannot execute it signals it as unexecuted (GraphHost::SignalExternalExecutionUnexecuted).
 */
class ExternalEpochExecution {
public:
	struct Item {
		std::shared_ptr<const compiler::PreparedRhiExecutionBatch> packet;
		// The heaps its lists bind, which are the recording's defaults. They are bound in the executing list before each
		// execution of secondaries. A secondary's own heap binding does not hold on every driver while the executing list
		// has other heaps bound.
		rhi::DescriptorHeapHandle resourceHeap{}, samplerHeap{};
	};
	std::vector<rhi::CommandList> leading;
	rhi::DescriptorHeapHandle leadingResourceHeap{}, leadingSamplerHeap{};
	std::vector<Item> items;
	std::vector<rhi::TimelinePoint> waits;
	std::vector<rhi::TimelinePoint> signals;
	// Keeps alive what the execution's lists and recordings read until it completes: the ticket's and the uploads' data.
	std::vector<std::shared_ptr<const void>> keepAlive;

	/**
	 * @brief Records the execution into a_primary, outside any pass. Call it once, on the host's executing thread.
	 * Afterwards, everything a_primary had bound is undefined: pipeline, heaps, push data and dynamic state.
	 */
	void Execute(rhi::CommandList a_primary) const;
	bool Executed() const noexcept { return m_executed.load(std::memory_order_acquire); }

private:
	mutable std::atomic<bool> m_executed{ false };
};

}  // namespace org
