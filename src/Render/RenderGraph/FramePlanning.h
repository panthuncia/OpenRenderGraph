#pragma once
#include "Render/RenderGraph/ExperimentalRhiExecution.h"
#include <deque>

namespace org::experimental {

// These tokens never reach an RHI timeline. The ordered submitter publishes
// the real signal after successful submission; waiting frames retain the token.
struct PlannedBatchSignal {
    uint64_t frameSequence = 0;
    uint32_t batch = 0;
    ExecutionTimelinePoint symbolic;
    std::atomic_uint64_t submittedValue{0};
};
struct FrameDependency {
    ExecutionTimelinePoint submitted;
    std::shared_ptr<const PlannedBatchSignal> predecessor;
    ExecutionTimelinePoint Resolve() const;
};
struct PlannedFrameState {
    std::shared_ptr<const RenderFrameSnapshot> snapshot;
    std::vector<std::vector<FrameDependency>> dependencies;
    std::vector<std::shared_ptr<PlannedBatchSignal>> signals;
    std::vector<rhi::ResourceHandle> invalidated;
    std::atomic_bool cancelled{false};
    std::atomic_bool submissionEligible{false};
    std::vector<std::vector<ExecutionTimelinePoint>> ResolveWaits() const;
};

// Serialized owner operations. Recording jobs only read returned snapshots;
// they never access these ledgers. Confirmation is FIFO, while recording may
// complete in any order. Cancellation requires the caller to join the suffix.
class FramePlanningState {
public:
    explicit FramePlanningState(size_t capacity) : m_capacity(capacity) {
        if (!capacity || capacity > 64) throw std::invalid_argument("Invalid planning capacity");
    }
    std::shared_ptr<const PlannedFrameState> Plan(std::shared_ptr<const CompiledGraphBundle> bundle,
        std::shared_ptr<const PreparedFramePayload> payload, std::span<const ExecutionTimelinePoint> queues);
    void Confirm(const std::shared_ptr<const PlannedFrameState>&, const GraphExecutionTimeline&);
    void ConfirmFailure(const std::shared_ptr<const PlannedFrameState>&, const GraphExecutionTimeline&, uint32_t signaledBatches);
    void CancelUnsubmittedSuffixAfterJoin();
    size_t Pending() const noexcept { return m_pending.size(); }
    size_t SymbolCount() const noexcept { return m_symbols.size(); }
    bool RecoveryRequired() const noexcept { return m_recovery; }
private:
    struct Ledgers {
        BackingStateAdmissionLedger states;
        BackingAccessAdmissionLedger accesses;
        AliasAccessAdmissionLedger aliases;
    };
    void CheckHead(const std::shared_ptr<const PlannedFrameState>&) const;
    void CommitKnown(const PlannedFrameState&, const GraphExecutionTimeline&, uint32_t count);
    static constexpr uint64_t PlannedBit = uint64_t{1} << 63;
    size_t m_capacity;
    uint64_t m_generation = 0, m_nextSequence = 1, m_nextSymbol = PlannedBit;
    bool m_recovery = false, m_failedPlanning = false;
    Ledgers m_confirmed, m_tail;
    std::deque<std::shared_ptr<PlannedFrameState>> m_pending;
    std::unordered_map<uint64_t, std::shared_ptr<PlannedBatchSignal>> m_symbols;
};

} // namespace org::experimental
