#include "FramePlanning.h"
#include "FrameTrace.h"

namespace org::experimental {

ExecutionTimelinePoint FrameDependency::Resolve() const {
    if (!predecessor) return submitted;
    const auto value = predecessor->submittedValue.load(std::memory_order_acquire);
    if (!value) throw std::logic_error("Frame dependency has not been submitted");
    return {predecessor->symbolic.timeline, value};
}

std::vector<std::vector<ExecutionTimelinePoint>> PlannedFrameState::ResolveWaits() const {
    if (cancelled.load(std::memory_order_acquire)) throw std::logic_error("Cancelled frame plan");
    if (!submissionEligible.load(std::memory_order_acquire)) throw std::logic_error("Frame is not the FIFO submission head");
    std::vector<std::vector<ExecutionTimelinePoint>> result(dependencies.size());
    for (size_t i = 0; i < dependencies.size(); ++i)
        for (const auto& dependency : dependencies[i]) result[i].push_back(dependency.Resolve());
    return result;
}

std::shared_ptr<const PlannedFrameState> FramePlanningState::Plan(
    std::shared_ptr<const CompiledGraphBundle> bundle, std::shared_ptr<const PreparedFramePayload> payload,
    std::span<const ExecutionTimelinePoint> queues) {
    BT_ZONE_SCOPE("ORG.Frame.Plan");
    if (m_recovery || m_failedPlanning || m_pending.size() >= m_capacity) throw std::logic_error("Frame planner unavailable");
    if (!bundle || !bundle->input || !bundle->graph || !bundle->graph->structure || !payload || bundle->sequence != m_nextSequence)
        throw std::invalid_argument("Out-of-order frame planning");
    const auto generation = bundle->input->structure.generation;
    if (!generation || (m_generation && generation != m_generation)) throw std::invalid_argument("Frame planning generation mismatch");
    if (queues.size() != bundle->graph->structure->queues.size()) throw std::invalid_argument("Planning queue mismatch");
    for (auto queue : queues) if (!queue.timeline || queue.value >= PlannedBit)
        throw std::invalid_argument("Invalid submitted queue point for planning");
    AnnotateFrameTrace(bundle->input->frameContext);

    // Build immutable barriers and dependencies before updating the planned
    // tail. A failed tail update cancels the suffix instead of copying all
    // historical state on every successful frame.
    auto result = std::make_shared<PlannedFrameState>();
    if (!payload->initialStates) throw std::invalid_argument("Missing prepared backing states");
    const auto& initial = *payload->initialStates;
    auto& planningLedgers = m_capacity == 1 ? m_confirmed : m_tail;
    result->invalidated = planningLedgers.aliases.ApplyInitialStates(*bundle->graph, initial);
    auto layout = BuildExecutionLayout(bundle, *bundle->input);
    auto barriers = std::make_shared<const PreparedExecutionBarrierPlan>(planningLedgers.states.Prepare(
        *bundle->graph, initial, result->invalidated));
    result->snapshot = BuildRenderFrameSnapshot(payload->frameNumber, std::move(layout), payload->passes,
        payload->bindings, payload->initialStates, barriers, payload->leases,
        payload->externalWaitsByPreparedPass, payload->resources, payload->preparationSlot,
        payload->frameData);
    const auto& graph = *bundle->graph;
    std::vector<std::vector<ExecutionTimelinePoint>> incoming(graph.batches.size());
    planningLedgers.accesses.AppendIncomingWaits(graph, *result->snapshot->initialStates, queues, incoming);
    planningLedgers.aliases.AppendIncomingWaits(graph, *result->snapshot->initialStates, queues, incoming);
    result->dependencies.resize(incoming.size());
    for (size_t i = 0; i < incoming.size(); ++i) for (auto point : incoming[i]) {
        if (point.value & PlannedBit) {
            const auto found = m_symbols.find(point.value);
            if (found == m_symbols.end()) throw std::logic_error("Missing predecessor frame signal");
            result->dependencies[i].push_back({{}, found->second});
        } else result->dependencies[i].push_back({point, {}});
    }
    GraphExecutionTimeline symbolic;
    symbolic.bundle = bundle;
    symbolic.batches.resize(graph.batches.size());
    auto nextSymbol = m_nextSymbol;
    auto symbols = m_symbols;
    if (m_capacity == 1 && m_reservedQueues.empty())
        m_reservedQueues.assign(queues.begin(), queues.end());
    if (m_capacity == 1) {
        if (m_reservedQueues.size() != queues.size())
            throw std::invalid_argument("Queue reservation dimensions changed");
        for (size_t queue = 0; queue < queues.size(); ++queue) {
            if (m_reservedQueues[queue].timeline != queues[queue].timeline)
                throw std::invalid_argument("Queue reservation identity changed");
            m_reservedQueues[queue].value =
                (std::max)(m_reservedQueues[queue].value, queues[queue].value);
        }
    }
    for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
        auto signal = std::make_shared<PlannedBatchSignal>();
        signal->frameSequence = bundle->sequence;
        signal->batch = batch;
        if (m_capacity == 1) {
            auto& reserved = m_reservedQueues.at(graph.batches[batch].queue);
            if (reserved.value == UINT64_MAX) throw std::overflow_error("Queue reservation exhausted");
            signal->symbolic = {reserved.timeline, ++reserved.value};
        } else {
            if (nextSymbol == UINT64_MAX - 1) throw std::overflow_error("Planning token space exhausted");
            signal->symbolic = {queues[graph.batches[batch].queue].timeline, ++nextSymbol};
            symbols.emplace(signal->symbolic.value, signal);
        }
        symbolic.batches[batch].signal = signal->symbolic;
        result->signals.push_back(std::move(signal));
    }
    result->submissionEligible.store(m_pending.empty());
    m_pending.push_back(result);
    try {
        // A single admitted frame has no speculative successor. Its effects
        // are applied exactly once, to the confirmed ledger, after submission.
        if (m_capacity != 1) {
            m_tail.states.Invalidate(result->invalidated);
            for (const auto& batch : barriers->batches) m_tail.states.CommitBatch(batch);
            m_tail.accesses.Commit(graph, *result->snapshot->initialStates, symbolic);
            m_tail.aliases.Commit(graph, *result->snapshot->initialStates, symbolic);
        }
        if (bundle->input->frameContext) bundle->input->frameContext->Advance(FrameStage::Compiling, FrameStage::Planned);
    } catch (...) { m_failedPlanning = true; throw; }
    m_symbols = std::move(symbols);
    m_generation = generation;
    ++m_nextSequence;
    m_nextSymbol = nextSymbol;
    BT_PLOT("ORG.Frame.PlannedDepth", static_cast<int64_t>(m_pending.size()));
    BT_PLOT("ORG.Frame.PlannedSignals", static_cast<int64_t>(m_symbols.size()));
    return result;
}

void FramePlanningState::CheckHead(const std::shared_ptr<const PlannedFrameState>& frame) const {
    if (m_recovery || m_failedPlanning || m_pending.empty() || frame != m_pending.front())
        throw std::logic_error("Non-FIFO frame confirmation");
}

void FramePlanningState::CommitKnown(const PlannedFrameState& frame,
    const GraphExecutionTimeline& execution, uint32_t count) {
    const auto& snapshot = *frame.snapshot;
    const auto& graph = *snapshot.layout->bundle->graph;
    if (execution.bundle != snapshot.layout->bundle || execution.batches.size() != frame.signals.size()
        || count > frame.signals.size()) throw std::invalid_argument("Frame submission receipt mismatch");
    std::unordered_map<uint64_t, ExecutionTimelinePoint> resolutions;
    for (uint32_t i = 0; i < count; ++i) {
        const auto point = execution.batches[i].signal;
        if (!point.value || point.value >= PlannedBit || point.timeline != frame.signals[i]->symbolic.timeline)
            throw std::invalid_argument("Invalid confirmed frame signal");
        if (m_capacity == 1) {
            if (point != frame.signals[i]->symbolic)
                throw std::invalid_argument("Submission did not consume its reserved signal");
        } else resolutions.emplace(frame.signals[i]->symbolic.value, point);
    }
    auto invalidated = frame.invalidated;
    for (uint32_t i = 0; i < count; ++i) {
        const auto& batch = snapshot.barrierPlan->batches[i];
        std::erase_if(invalidated, [&](auto resource) {
            if (!std::ranges::any_of(batch.seeds, [&](const auto& seed) {
                return seed.resource.index == resource.index && seed.resource.generation == resource.generation;
            })) return false;
            m_confirmed.states.Invalidate(std::span(&resource, 1));
            return true;
        });
        m_confirmed.states.CommitBatch(batch);
    }
    m_confirmed.accesses.Commit(graph, *snapshot.initialStates, execution, count);
    m_confirmed.aliases.Commit(graph, *snapshot.initialStates, execution, count);
    if (m_capacity != 1) {
        m_tail.accesses.ResolvePlannedPoints(resolutions);
        m_tail.aliases.ResolvePlannedPoints(resolutions);
    }
    for (uint32_t i = 0; i < count; ++i) {
        frame.signals[i]->submittedValue.store(execution.batches[i].signal.value, std::memory_order_release);
        m_symbols.erase(frame.signals[i]->symbolic.value);
    }
}

void FramePlanningState::Confirm(const std::shared_ptr<const PlannedFrameState>& frame,
    const GraphExecutionTimeline& execution) {
    BT_ZONE_SCOPE("ORG.Frame.ConfirmSubmission");
    CheckHead(frame);
    AnnotateFrameTrace(frame->snapshot->layout->bundle->input->frameContext);
    try { CommitKnown(*frame, execution, static_cast<uint32_t>(frame->signals.size())); }
    catch (...) {
        m_recovery = true;
        for (const auto& pending : m_pending) pending->submissionEligible.store(false, std::memory_order_release);
        throw;
    }
    m_pending.pop_front();
    if (!m_pending.empty()) m_pending.front()->submissionEligible.store(true, std::memory_order_release);
}

void FramePlanningState::ConfirmFailure(const std::shared_ptr<const PlannedFrameState>& frame,
    const GraphExecutionTimeline& execution, uint32_t signaledBatches) {
    CheckHead(frame);
    m_recovery = true; // No further planning/submission until device recovery.
    for (const auto& pending : m_pending) pending->submissionEligible.store(false, std::memory_order_release);
    CommitKnown(*frame, execution, signaledBatches);
}

void FramePlanningState::CancelUnsubmittedSuffixAfterJoin() {
    BT_ZONE_SCOPE("ORG.Frame.CancelPlannedSuffix");
    if (m_recovery) throw std::logic_error("Potential GPU submission requires recovery");
    auto tail = m_confirmed;
    for (const auto& pending : m_pending) {
        const auto& frame = pending->snapshot->layout->bundle->input->frameContext;
        if (frame && frame->Stage() >= FrameStage::Submitted && frame->Stage() != FrameStage::Cancelled)
            throw std::logic_error("Cannot cancel submitted frame ownership");
    }
    if (!m_pending.empty()) m_nextSequence = m_pending.front()->snapshot->layout->bundle->sequence;
    for (const auto& pending : m_pending) {
        pending->cancelled.store(true, std::memory_order_release);
        const auto& frame = pending->snapshot->layout->bundle->input->frameContext;
        if (frame && frame->Stage() != FrameStage::Cancelled) frame->CancelAfterJoin();
    }
    basic_telemetry::AddCounter("ORG.Frame.CancelledPlans", static_cast<int64_t>(m_pending.size()));
    m_pending.clear(); m_symbols.clear();
    if (m_capacity != 1) m_tail = std::move(tail);
    m_failedPlanning = false;
}

SynchronousFramePlan SynchronousPlanningState::Plan(
    std::shared_ptr<const CompiledGraphBundle> bundle,
    std::shared_ptr<const PreparedFramePayload> payload,
    std::span<const ExecutionTimelinePoint> queues) {
    BT_ZONE_SCOPE("ORG.SyncExecution.Plan");
    if (!bundle || !bundle->input || !bundle->graph || !payload
        || queues.size() != bundle->graph->structure->queues.size())
        throw std::invalid_argument("Incomplete synchronous planning input");
    SynchronousFramePlan result;
    if (!payload->initialStates) throw std::invalid_argument("Missing prepared backing states");
    const auto& initial = *payload->initialStates;
    result.invalidated = m_aliases.ApplyInitialStates(*bundle->graph, initial);
    auto layout = BuildExecutionLayout(bundle, *bundle->input);
    auto barriers = std::make_shared<const PreparedExecutionBarrierPlan>(
        m_states.Prepare(*bundle->graph, initial, result.invalidated));
    result.snapshot = BuildRenderFrameSnapshot(payload->frameNumber, std::move(layout),
        payload->passes, payload->bindings, payload->initialStates, std::move(barriers),
        payload->leases, payload->externalWaitsByPreparedPass, payload->resources,
        payload->preparationSlot, payload->frameData);
    result.incomingWaits.resize(bundle->graph->batches.size());
    m_accesses.AppendIncomingWaits(*bundle->graph, *result.snapshot->initialStates,
        queues, result.incomingWaits);
    m_aliases.AppendIncomingWaits(*bundle->graph, *result.snapshot->initialStates,
        queues, result.incomingWaits);
    if (bundle->input->frameContext)
        bundle->input->frameContext->Advance(FrameStage::Compiling, FrameStage::Planned);
    return result;
}

void SynchronousPlanningState::Confirm(const SynchronousFramePlan& frame,
    const GraphExecutionTimeline& execution) {
    BT_ZONE_SCOPE("ORG.SyncExecution.Confirm");
    if (!frame.snapshot || !frame.snapshot->layout
        || execution.bundle != frame.snapshot->layout->bundle)
        throw std::invalid_argument("Synchronous submission receipt mismatch");
    const auto& graph = *frame.snapshot->layout->bundle->graph;
    auto invalidated = frame.invalidated;
    for (const auto& batch : frame.snapshot->barrierPlan->batches) {
        std::erase_if(invalidated, [&](auto resource) {
            if (!std::ranges::any_of(batch.seeds, [&](const auto& seed) {
                return seed.resource.index == resource.index
                    && seed.resource.generation == resource.generation;
            })) return false;
            m_states.Invalidate(std::span(&resource, 1));
            return true;
        });
        m_states.CommitBatch(batch);
    }
    m_accesses.Commit(graph, *frame.snapshot->initialStates, execution);
    m_aliases.Commit(graph, *frame.snapshot->initialStates, execution);
}

} // namespace org::experimental
