#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <rhi.h>

namespace org::experimental {
class PreparedRhiExecutionBatch;
}

namespace org {

/**
 * @brief An external epoch's execution (RenderGraph::SetExternalPersistentEpochs; PersistentGraphHost::TakeExternalExecution):
 * what its ticket's submission would have submitted, for the host to execute in a command buffer of its own (DXVK's, inside an
 * external-commands callback) and to submit with that buffer. In order:
 * - the lists recorded ahead of it (the uploads, a secondary);
 * - each batch: a secondary list recorded ahead, or a late batch (a pass recorded at execution) recorded by Execute.
 *
 * The host's submission of that buffer waits for `waits` and signals `signals`, among them the graph's own value for the
 * execution on its graphics timeline. The graph's later work on that timeline cannot complete until it does. A host that cannot
 * execute it signals it unexecuted (PersistentGraphHost::SignalExternalExecutionUnexecuted).
 */
class ExternalEpochExecution {
public:
	struct Item {
		std::shared_ptr<const experimental::PreparedRhiExecutionBatch> packet;
		// The heaps its lists bind (the recording's defaults): bound in the executing list before each execution of secondaries,
		// whose own binding does not hold on every driver while the executing list has other heaps bound.
		rhi::DescriptorHeapHandle resourceHeap{}, samplerHeap{};
	};
	std::vector<rhi::CommandList> leading;
	rhi::DescriptorHeapHandle leadingResourceHeap{}, leadingSamplerHeap{};
	std::vector<Item> items;
	std::vector<rhi::TimelinePoint> waits;
	std::vector<rhi::TimelinePoint> signals;
	// What the execution's lists and recordings read until it completes (the ticket's, the uploads').
	std::vector<std::shared_ptr<const void>> keepAlive;

	/**
	 * @brief Executing thread (the host's): records the execution into a_primary, outside any pass. Once. Everything a_primary had
	 * bound (pipeline, heaps, push data, dynamic state) is undefined afterwards.
	 */
	void Execute(rhi::CommandList a_primary) const;
	bool Executed() const noexcept { return m_executed.load(std::memory_order_acquire); }

private:
	mutable std::atomic<bool> m_executed{ false };
};

}  // namespace org
