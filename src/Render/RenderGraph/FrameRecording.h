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
    // An external epoch's late batch (CompilePass::recordAtExecution): no list; its packet keeps the recording for the host's
    // execution (PreparedRhiExecutionBatch::RecordDeferred).
    bool deferred = false;
    // Non-empty: the plan's timelines as this job's device sees them (a peer device waits on its imported copies).
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
    // Async epochs: hands the packets to their queues with timelines the caller assigned (one per batch),
    // touching no admission state - safe on the submitting thread while the owner prepares other work. The
    // owner records the submission afterwards (ExecutionTimelineAdmission::RecordSubmitted). False when a
    // packet failed; the frame is consumed either way.
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
// another worker: collection is a non-blocking readiness probe followed by a
// FIFO join on the submission thread.
RecordedFrame RecordFrame(PlannedFrame plan,
    const std::shared_ptr<runtime::ITaskService>& tasks, size_t concurrency);

} // namespace org::compiler
