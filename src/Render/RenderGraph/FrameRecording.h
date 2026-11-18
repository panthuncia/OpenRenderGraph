#pragma once

#include "Render/RenderGraph/RhiExecution.h"
#include "Render/Runtime/ITaskService.h"
#include <functional>

namespace org::compiler {

struct FrameRecordingJob {
    uint32_t slot = 0;
    rhi::QueueKind queueKind = rhi::QueueKind::Graphics;
    rhi::Device device;
    rhi::Queue queue;
    std::shared_ptr<CommandListPool> pool;
    OwnedRecordingList recording;
    // A late batch of an external epoch (CompilePass::recordAtExecution). It has no command list. Its packet keeps the
    // recording for later, and the host's execution records it (PreparedRhiExecutionBatch::RecordDeferred).
    bool deferred = false;
    // When non-empty, the plan's timelines as this job's device sees them. A peer device waits on its imported copies.
    std::vector<PreparedTimelineBinding> timelines;
};

// Everything the recorder may read is frozen before dispatch. It has no graph,
// pass registry, host execution context, or submission ledger to consult.
struct PlannedFrame {
    PlannedFrame() = default;
    PlannedFrame(PlannedFrame&&) = default;
    PlannedFrame& operator=(PlannedFrame&&) = default;
    PlannedFrame(const PlannedFrame&) = delete;
    PlannedFrame& operator=(const PlannedFrame&) = delete;
    PlannedFrame(std::shared_ptr<const RenderFrameSnapshot> frame, std::vector<FrameRecordingJob> work,
        std::vector<PreparedTimelineBinding> bindings, std::vector<std::vector<ExecutionTimelinePoint>> waits)
        : snapshot(std::move(frame)), jobs(std::move(work)), timelines(std::move(bindings)),
          incomingWaits(std::move(waits)) {}
    std::shared_ptr<const RenderFrameSnapshot> snapshot;
    std::vector<FrameRecordingJob> jobs;
    std::vector<PreparedTimelineBinding> timelines;
    std::vector<std::vector<ExecutionTimelinePoint>> incomingWaits;
};

class RecordedFrame {
public:
    RecordedFrame(RecordedFrame&&) = default;
    RecordedFrame& operator=(RecordedFrame&&) = default;
    RecordedFrame(const RecordedFrame&) = delete;
    const auto& Snapshot() const noexcept { return m_snapshot; }
    std::shared_ptr<const GraphExecutionTimeline> Submit(ExecutionTimelineAdmission& admission) &&;
    // Admits the frame at submission time (ExecutionTimelineAdmission::SubmitPrepared). The caller passes:
    // - the waits the admission computed at submission, which replace the plan's waits,
    // - the lists that go ahead of each batch's own lists,
    // - the leases that keep those lists alive.
    std::shared_ptr<const GraphExecutionTimeline> Submit(ExecutionTimelineAdmission& admission,
        const std::vector<std::vector<ExecutionTimelinePoint>>& incomingWaits,
        std::span<const std::span<const rhi::CommandList>> leading, std::vector<std::shared_ptr<const void>> leases,
        std::span<const PreparedTimelineBinding> extraTimelines = {}) &&;
    // For async epochs. Hands the packets to their queues with the timelines the caller assigned, one per batch.
    // It touches no admission state. So it is safe on the submitting thread while the owner prepares other work.
    // The owner records the submission afterwards (ExecutionTimelineAdmission::RecordSubmitted).
    // Returns false when a packet failed. The frame is consumed either way.
    bool SubmitPackets(std::span<const ExecutionBatchTimeline> batches);
    const auto& Packets() const noexcept { return m_batches; }
    const auto& IncomingWaits() const noexcept { return m_incomingWaits; }
private:
    friend RecordedFrame RecordFrame(PlannedFrame, const std::shared_ptr<runtime::ITaskService>&, size_t);
    RecordedFrame() = default;
    std::shared_ptr<const RenderFrameSnapshot> m_snapshot;
    std::vector<std::vector<ExecutionTimelinePoint>> m_incomingWaits;
    std::vector<std::shared_ptr<const IPreparedExecutionBatch>> m_batches;
    bool m_submitted = false;
};

// Owns peer recording workers for one logical frame. No worker waits for
// another worker. To collect them, the submission thread first checks
// readiness without blocking, then joins the workers in FIFO order.
RecordedFrame RecordFrame(PlannedFrame plan,
    const std::shared_ptr<runtime::ITaskService>& tasks, size_t concurrency);

} // namespace org::compiler
