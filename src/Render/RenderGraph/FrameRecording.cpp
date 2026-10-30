#include "FrameRecording.h"
#include "FrameTrace.h"

namespace org::experimental {

RecordedFrame RecordFrame(PlannedFrame plan,
    const std::shared_ptr<runtime::ITaskService>& tasks, size_t concurrency) {
    BT_ZONE_SCOPE("ORG.Frame.Record");
    if (!plan.snapshot || !plan.snapshot->layout || !plan.snapshot->layout->bundle
        || plan.jobs.empty() || !concurrency)
        throw std::invalid_argument("Incomplete planned frame");
    const auto& frame = plan.snapshot->layout->bundle->input->frameContext;
    AnnotateFrameTrace(frame);
    RecordedFrame result;
    result.m_snapshot = plan.snapshot;
    result.m_incomingWaits = std::move(plan.incomingWaits);
    result.m_batches.resize(plan.jobs.size());
    if (frame) frame->Advance(FrameStage::Planned, FrameStage::Recording);
    try {
        auto record = [&](size_t batch) {
            BT_ZONE_SCOPE("ORG.Frame.RecordBatch");
            AnnotateFrameTrace(frame);
            auto& job = plan.jobs[batch];
            auto& recording = job.recording;
            if (job.deferred) {
                result.m_batches[batch] = std::make_shared<const PreparedRhiExecutionBatch>(job.slot, job.queue,
                    std::move(recording), plan.timelines, plan.snapshot);
                return;
            }
            if (!job.pool || !recording.allocation->pair.list)
                throw std::invalid_argument("Frame recording has no prepared command list");
            std::vector<OwnedRecordingList> recordings;
            recordings.push_back(std::move(recording));
            result.m_batches[batch] = RecordPreparedRhiExecutionBatch(job.slot, job.queue,
                std::move(recordings), plan.timelines, plan.snapshot);
        };
        // ParallelForLimited joins all children, including when a child throws.
        if (tasks && concurrency > 1)
            tasks->ParallelForLimited("ORG.Frame.RecordBatches", plan.jobs.size(), concurrency, record);
        else
            for (size_t batch = 0; batch < plan.jobs.size(); ++batch) record(batch);
        if (frame) frame->Advance(FrameStage::Recording, FrameStage::Ready);
    } catch (...) {
        if (frame) frame->CancelAfterJoin();
        basic_telemetry::AddCounter("ORG.Frame.RecordingFailures");
        throw;
    }
    return result;
}

bool RecordedFrame::SubmitPackets(std::span<const ExecutionBatchTimeline> batches) {
    BT_ZONE_SCOPE("ORG.Frame.SubmitPackets");
    if (m_submitted || !m_snapshot || m_batches.empty() || batches.size() != m_batches.size())
        throw std::logic_error("Recorded frame already consumed or incomplete");
    m_submitted = true;
    for (size_t i = 0; i < m_batches.size(); ++i)
        if (!m_batches[i]->Submit(batches[i])) return false;
    return true;
}

std::shared_ptr<const GraphExecutionTimeline> RecordedFrame::Submit(ExecutionTimelineAdmission& admission) && {
    BT_ZONE_SCOPE("ORG.Frame.Submit");
    if (m_submitted || !m_snapshot || m_batches.empty())
        throw std::logic_error("Recorded frame already consumed or incomplete");
    // Set before touching the API: even an uncertain submission is single-use.
    m_submitted = true;
    AnnotateFrameTrace(m_snapshot->layout->bundle->input->frameContext);
    return admission.SubmitPrepared(m_snapshot->layout->bundle, m_incomingWaits, m_batches, {});
}

} // namespace org::experimental
