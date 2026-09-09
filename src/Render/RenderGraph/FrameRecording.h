#pragma once

#include "Render/RenderGraph/ExperimentalRhiExecution.h"
#include "Render/Runtime/ITaskService.h"
#include "FramePlanning.h"

namespace org::experimental {

struct FrameRecordingJob {
    uint32_t slot = 0;
    rhi::QueueKind queueKind = rhi::QueueKind::Graphics;
    rhi::Device device;
    rhi::Queue queue;
    std::shared_ptr<CommandListPool> pool;
    OwnedRecordingList recording;
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
        std::vector<PreparedTimelineBinding> bindings, std::vector<std::vector<ExecutionTimelinePoint>> waits,
        std::shared_ptr<const PlannedFrameState> state = {})
        : snapshot(std::move(frame)), jobs(std::move(work)), timelines(std::move(bindings)),
          incomingWaits(std::move(waits)), planning(std::move(state)) {}
    std::shared_ptr<const RenderFrameSnapshot> snapshot;
    std::vector<FrameRecordingJob> jobs;
    std::vector<PreparedTimelineBinding> timelines;
    std::vector<std::vector<ExecutionTimelinePoint>> incomingWaits;
    std::shared_ptr<const PlannedFrameState> planning;
};

class RecordedFrame {
public:
    RecordedFrame(RecordedFrame&&) = default;
    RecordedFrame& operator=(RecordedFrame&&) = default;
    RecordedFrame(const RecordedFrame&) = delete;
    const auto& Snapshot() const noexcept { return m_snapshot; }
    std::shared_ptr<const GraphExecutionTimeline> Submit(ExecutionTimelineAdmission& admission) &&;
private:
    friend RecordedFrame RecordFrame(PlannedFrame, const std::shared_ptr<runtime::ITaskService>&, size_t);
    RecordedFrame() = default;
    std::shared_ptr<const RenderFrameSnapshot> m_snapshot;
    std::vector<std::vector<ExecutionTimelinePoint>> m_incomingWaits;
    std::vector<std::shared_ptr<const IPreparedExecutionBatch>> m_batches;
    std::shared_ptr<const PlannedFrameState> m_planning;
    bool m_submitted = false;
};

RecordedFrame RecordFrame(PlannedFrame plan,
    const std::shared_ptr<runtime::ITaskService>& tasks, size_t concurrency);

} // namespace org::experimental
