#include "FramePlanning.h"
#include "FrameTrace.h"

namespace org::experimental {

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
