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
 * It holds what the epoch's ticket would have submitted on the graphics queue. The host executes it in command buffers of
 * its own (DXVK's, inside external-commands callbacks) and submits it with them. The epoch's batches on other queues were
 * submitted by the graph already, on those queues.
 *
 * It is an ordered list of segments, which the host executes in order, each once. A segment contains, in order:
 * - the lists recorded ahead of it: the uploads (the first segment only) and its batches' handoff into the execution, as
 *   secondary lists,
 * - each of its batches, either as a secondary list recorded ahead, or as a late batch that Execute records. A late batch
 *   is a pass that is recorded at execution.
 *
 * The host's submission that holds a segment waits for the segment's `waits` and signals its `signals` (at its end at the
 * latest). Among the signals are the graph's own values for the segment's batches on its graphics timeline: the graph's
 * later work on that timeline cannot complete until the host signals them. A new segment starts where a batch waits for
 * a batch of the execution on another queue. A segment marked `split` must start a new host submission: one of its waits
 * is for a batch that waits for an earlier segment's signal, so a submission that held both would wait for itself. A host
 * that cannot execute a segment signals it as unexecuted (GraphHost::SignalExternalExecutionUnexecuted).
 *
 * With a submission thread (GraphHost::Desc::submissionThread) the host hands the execution out before its tickets are
 * finished. Its segments and exitSplit are complete only once WaitFinished returns.
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
	struct Segment {
		std::vector<rhi::CommandList> leading;
		rhi::DescriptorHeapHandle leadingResourceHeap{}, leadingSamplerHeap{};
		std::vector<Item> items;
		std::vector<rhi::TimelinePoint> waits;
		std::vector<rhi::TimelinePoint> signals;
		bool split = false;
	};
	std::vector<Segment> segments;
	// A batch on another queue waits for a segment: a wait for the execution's batches on other queues (its exit) must start a
	// host submission after the one that holds its last segment.
	bool exitSplit = false;
	// Keeps alive what the execution's lists and recordings read until it completes: the ticket's and the uploads' data.
	std::vector<std::shared_ptr<const void>> keepAlive;

	ExternalEpochExecution() = default;
	ExternalEpochExecution(const ExternalEpochExecution&) = delete;
	ExternalEpochExecution& operator=(const ExternalEpochExecution&) = delete;

	// The tickets that add their segments to it. Set by the host on the submitting thread, before it hands the execution out.
	uint32_t tickets = 0;

	/**
	 * @brief Any thread: blocks until every ticket of the execution was finished (each added its segments). Returns false when one
	 * failed: then the segments are only what the others added, and the host's submissions fail from then on.
	 */
	bool WaitFinished() const noexcept;
	// The host, once per ticket, after it added the ticket's segments (or failed to).
	void FinishTicket(bool a_ok) noexcept;

	/**
	 * @brief Records segment a_segment into a_primary, outside any pass. Call it once per segment, in order, on the host's
	 * executing thread. Afterwards, everything a_primary had bound is undefined: pipeline, heaps, push data and dynamic state.
	 */
	void Execute(size_t a_segment, rhi::CommandList a_primary) const;
	// How many segments were executed (Execute): the first ones, in order.
	size_t ExecutedSegments() const noexcept { return m_executed.load(std::memory_order_acquire); }
	bool Executed() const noexcept { return ExecutedSegments() == segments.size(); }

private:
	mutable std::atomic<size_t> m_executed{ 0 };
	std::atomic<uint32_t> m_finished{ 0 };
	std::atomic<bool> m_failed{ false };
};

}  // namespace org
