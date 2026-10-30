#pragma once
#include "Render/RenderGraph/ExperimentalRhiExecution.h"

namespace org::experimental {

struct SynchronousFramePlan {
    std::shared_ptr<const RenderFrameSnapshot> snapshot;
    std::vector<std::vector<ExecutionTimelinePoint>> incomingWaits;
    std::vector<rhi::ResourceHandle> invalidated;
};

// Immediate admission policy: no pending tail, symbolic signal, FIFO token,
// cancellation state, or frame-lifecycle transition.
class SynchronousPlanningState {
public:
    SynchronousFramePlan Plan(std::shared_ptr<const CompiledGraphBundle>,
        std::shared_ptr<const PreparedFramePayload>,
        std::span<const ExecutionTimelinePoint> queues);
    void Confirm(const SynchronousFramePlan&, const GraphExecutionTimeline&);
private:
    BackingStateAdmissionLedger m_states;
    BackingAccessAdmissionLedger m_accesses;
    AliasAccessAdmissionLedger m_aliases;
};

} // namespace org::experimental
