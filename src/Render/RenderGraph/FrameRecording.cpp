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
    result.m_planning = plan.planning;
    result.m_incomingWaits = std::move(plan.incomingWaits);
    result.m_batches.resize(plan.jobs.size());
    if (frame) frame->Advance(FrameStage::Planned, FrameStage::Recording);
    try {
        auto record = [&](size_t batch) {
            BT_ZONE_SCOPE("ORG.Frame.RecordBatch");
            AnnotateFrameTrace(frame);
            auto& job = plan.jobs[batch];
            auto& recording = job.recording;
            if (!job.pool) throw std::invalid_argument("Frame recording has no slot command pool");
            recording.allocation->pair = job.pool->Request();
            recording.allocation->pool = job.pool;
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

std::shared_ptr<const GraphExecutionTimeline> RecordedFrame::Submit(ExecutionTimelineAdmission& admission) && {
    BT_ZONE_SCOPE("ORG.Frame.Submit");
    if (m_submitted || !m_snapshot || m_batches.empty())
        throw std::logic_error("Recorded frame already consumed or incomplete");
    auto incoming = m_incomingWaits;
    if (m_planning) {
        auto planned = m_planning->ResolveWaits();
        if (planned.size() != incoming.size()) throw std::logic_error("Planned wait dimensions changed");
        for (size_t i = 0; i < incoming.size(); ++i)
            incoming[i].insert(incoming[i].end(), planned[i].begin(), planned[i].end());
    }
    // Set before touching the API: even an uncertain submission is single-use.
    m_submitted = true;
    AnnotateFrameTrace(m_snapshot->layout->bundle->input->frameContext);
    return admission.SubmitPrepared(m_snapshot->layout->bundle, incoming, m_batches);
}

} // namespace org::experimental
