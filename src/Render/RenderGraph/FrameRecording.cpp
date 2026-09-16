#include "FrameRecording.h"
#include "FrameTrace.h"

namespace org::experimental {

PersistentRecordingLanes::PersistentRecordingLanes(size_t laneCount) {
    if (!laneCount) throw std::invalid_argument("Recording lane count must be nonzero");
    m_threads.reserve(laneCount);
    for (size_t lane = 0; lane < laneCount; ++lane)
        m_threads.emplace_back([this] { Run(); });
}

PersistentRecordingLanes::~PersistentRecordingLanes() {
    {
        std::lock_guard lock(m_mutex);
        m_stopping = true;
    }
    m_ready.notify_all();
    for (auto& thread : m_threads) if (thread.joinable()) thread.join();
}

void PersistentRecordingLanes::Submit(std::function<void()> work) {
    if (!work) throw std::invalid_argument("Empty recording-lane work");
    {
        std::lock_guard lock(m_mutex);
        if (m_stopping) throw std::runtime_error("Recording lanes are stopping");
        m_work.push_back(std::move(work));
    }
    m_ready.notify_one();
}

void PersistentRecordingLanes::Run() {
    while (true) {
        std::function<void()> work;
        {
            std::unique_lock lock(m_mutex);
            m_ready.wait(lock, [this] { return m_stopping || !m_work.empty(); });
            if (m_work.empty()) {
                if (m_stopping) return;
                continue;
            }
            work = std::move(m_work.front());
            m_work.pop_front();
        }
        work();
    }
}

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
    std::atomic_size_t nextBatch{0};
    std::atomic_size_t completedWorkers{0};
    size_t workerCount = 0;
    std::mutex completionMutex;
    std::condition_variable completion;
    std::mutex failureMutex;
    std::exception_ptr failure;
    bool joined = false;
};

bool DispatchedFrameRecording::Ready() const {
    return m_state && !m_state->joined && m_state->workerCount
        && m_state->completedWorkers.load(std::memory_order_acquire) == m_state->workerCount;
}

RecordedFrame DispatchedFrameRecording::Join() {
    if (!m_state || m_state->joined) throw std::logic_error("Dispatched recording already consumed");
    auto state = std::exchange(m_state, {});
    state->joined = true;
    {
        std::unique_lock lock(state->completionMutex);
        state->completion.wait(lock, [&] {
            return state->completedWorkers.load(std::memory_order_acquire) == state->workerCount;
        });
    }
    if (state->failure) {
        if (state->frame) state->frame->CancelAfterJoin();
        basic_telemetry::AddCounter("ORG.Frame.RecordingFailures");
        std::rethrow_exception(state->failure);
    }
    if (state->frame)
        state->frame->Advance(FrameStage::Recording, FrameStage::Ready);
    return std::move(state->result);
}

DispatchedFrameRecording DispatchFrameRecording(PlannedFrame plan,
    PersistentRecordingLanes& lanes, size_t concurrency) {
    if (!plan.snapshot || !plan.snapshot->layout || !plan.snapshot->layout->bundle
        || plan.jobs.empty() || !concurrency)
        throw std::invalid_argument("Incomplete dispatched frame recording");

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
    state->workerCount = (std::min)({concurrency, state->jobs.size(), lanes.LaneCount()});
    for (size_t workerIndex = 0; workerIndex < state->workerCount; ++workerIndex) {
        lanes.Submit([state] {
            BT_ZONE_SCOPE("ORG.Frame.RecordWorker");
            try {
                while (true) {
                    const auto batch = state->nextBatch.fetch_add(1, std::memory_order_relaxed);
                    if (batch >= state->jobs.size()) break;
                    BT_ZONE_SCOPE("ORG.Frame.RecordBatch");
                    AnnotateFrameTrace(state->frame);
                    auto& job = state->jobs[batch];
                    if (!job.pool || !job.recording.allocation->pair.list)
                        throw std::invalid_argument("Frame recording has no prepared command list");
                    std::vector<OwnedRecordingList> recordings;
                    recordings.push_back(std::move(job.recording));
                    state->result.m_batches[batch] = RecordPreparedRhiExecutionBatch(
                        job.slot, job.queue, std::move(recordings), state->timelines,
                        state->result.m_snapshot);
                }
            } catch (...) {
                std::lock_guard lock(state->failureMutex);
                if (!state->failure) state->failure = std::current_exception();
            }
            if (state->completedWorkers.fetch_add(1, std::memory_order_release) + 1 == state->workerCount)
                state->completion.notify_all();
        });
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
    std::vector<ExecutionTimelinePoint> reserved;
    if (m_planning) {
        reserved.reserve(m_planning->signals.size());
        for (const auto& signal : m_planning->signals) reserved.push_back(signal->symbolic);
        // Asynchronous planning uses symbolic predecessor tokens. Only the
        // synchronous planner reserves concrete admission signal values.
        if (std::ranges::any_of(reserved, [](auto point) { return point.value & (uint64_t{1} << 63); }))
            reserved.clear();
    }
    return admission.SubmitPrepared(m_snapshot->layout->bundle, incoming, m_batches, reserved);
}

} // namespace org::experimental
