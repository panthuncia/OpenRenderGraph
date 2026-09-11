#include "FrameRecording.h"
#include "FrameTrace.h"

namespace org::experimental {

struct DispatchedFrameRecording::State {
    State(RecordedFrame&& recorded, std::vector<FrameRecordingJob>&& recordingJobs,
        std::vector<PreparedTimelineBinding>&& timelineBindings,
        std::shared_ptr<FrameContext> frameOwner)
        : result(std::move(recorded)), jobs(std::move(recordingJobs)),
          timelines(std::move(timelineBindings)), frame(std::move(frameOwner)) {}

    RecordedFrame result;
    std::vector<FrameRecordingJob> jobs;
    std::vector<PreparedTimelineBinding> timelines;
    std::shared_ptr<FrameContext> frame;
    std::vector<std::future<void>> workers;
    std::atomic_size_t nextBatch{0};
    bool joined = false;
};

bool DispatchedFrameRecording::Ready() const {
    if (!m_state || m_state->workers.empty() || m_state->joined) return false;
    return std::ranges::all_of(m_state->workers, [](std::future<void>& worker) {
        return worker.valid()
            && worker.wait_for(std::chrono::seconds{0}) == std::future_status::ready;
    });
}

RecordedFrame DispatchedFrameRecording::Join() {
    if (!m_state || m_state->joined) throw std::logic_error("Dispatched recording already consumed");
    auto state = std::exchange(m_state, {});
    state->joined = true;
    std::exception_ptr failure;
    for (auto& worker : state->workers) {
        try { worker.get(); }
        catch (...) { if (!failure) failure = std::current_exception(); }
    }
    if (failure) {
        if (state->frame) state->frame->CancelAfterJoin();
        basic_telemetry::AddCounter("ORG.Frame.RecordingFailures");
        std::rethrow_exception(failure);
    }
    if (state->frame)
        state->frame->Advance(FrameStage::Recording, FrameStage::Ready);
    return std::move(state->result);
}

DispatchedFrameRecording DispatchFrameRecording(PlannedFrame plan,
    const std::shared_ptr<runtime::ITaskService>& tasks,
    std::shared_ptr<runtime::ITaskScope>& scope, size_t concurrency) {
    if (!tasks || !plan.snapshot || !plan.snapshot->layout || !plan.snapshot->layout->bundle
        || plan.jobs.empty() || !concurrency)
        throw std::invalid_argument("Incomplete dispatched frame recording");
    if (!scope) scope = tasks->CreateScope("ORG.Frame.Recording");
    if (!scope) throw std::runtime_error("Frame recording scope rejected");

    auto frame = plan.snapshot->layout->bundle->input->frameContext;
    AnnotateFrameTrace(frame);
    RecordedFrame recorded;
    recorded.m_snapshot = plan.snapshot;
    recorded.m_planning = plan.planning;
    recorded.m_incomingWaits = std::move(plan.incomingWaits);
    recorded.m_batches.resize(plan.jobs.size());
    if (frame) frame->Advance(FrameStage::Planned, FrameStage::Recording);

    auto state = std::make_shared<DispatchedFrameRecording::State>(
        std::move(recorded), std::move(plan.jobs), std::move(plan.timelines), frame);
    const auto workerCount = (std::min)(concurrency, state->jobs.size());
    state->workers.reserve(workerCount);
    for (size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex) {
        std::weak_ptr<DispatchedFrameRecording::State> weakState = state;
        auto worker = std::make_shared<std::packaged_task<void()>>([weakState] {
            auto state = weakState.lock();
            if (!state) throw std::runtime_error("Frame recording owner was cancelled");
            BT_ZONE_SCOPE("ORG.Frame.RecordWorker");
            while (true) {
                const auto batch = state->nextBatch.fetch_add(1, std::memory_order_relaxed);
                if (batch >= state->jobs.size()) return;
                BT_ZONE_SCOPE("ORG.Frame.RecordBatch");
                AnnotateFrameTrace(state->frame);
                auto& job = state->jobs[batch];
                if (!job.pool) throw std::invalid_argument("Frame recording has no slot command pool");
                job.recording.allocation->pair = job.pool->Request();
                job.recording.allocation->pool = job.pool;
                std::vector<OwnedRecordingList> recordings;
                recordings.push_back(std::move(job.recording));
                state->result.m_batches[batch] = RecordPreparedRhiExecutionBatch(
                    job.slot, job.queue, std::move(recordings), state->timelines,
                    state->result.m_snapshot);
            }
        });
        state->workers.push_back(worker->get_future());
        if (!tasks->Submit(scope, runtime::TaskPriority::FrameCritical,
                "ORG.Frame.RecordWorker", [worker] { (*worker)(); })) {
            worker.reset();
            break;
        }
        worker.reset();
    }
    DispatchedFrameRecording dispatch;
    dispatch.m_state = std::move(state);
    return dispatch;
}

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
