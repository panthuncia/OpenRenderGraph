#pragma once

#include "Render/RenderGraph/ExperimentalGraphCompiler.h"
#include "Resources/AliasingPlacement.h"
#include <rhi.h>
#include <span>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace org::experimental {

struct PreparedStateRegion {
    CompileRange range;
    CompileResourceState state;
};

// Owner-thread snapshot of the actual backing state at frame preparation.
// The concrete handle, rather than a graph resource ID, is the cross-frame
// identity: replacing a backing starts with its newly captured initial state.
struct PreparedBackingState {
    uint64_t graphResourceID = 0;
    std::string graphResourceKey;
    rhi::ResourceHandle resource{};
    CompileResourceShape shape{};
    std::shared_ptr<const std::vector<PreparedStateRegion>> regions;
    std::shared_ptr<const AliasHeapGeneration> aliasHeap;
    const AliasHeapGeneration* aliasHeapIdentity = nullptr;
    uint64_t aliasPoolID = 0;
    uint64_t aliasOffset = 0;
    uint64_t aliasSize = 0;
    // Immutable realization metadata used to encode legal backend barriers.
    // D3D12 upload/readback buffers remain in fixed states and must not receive
    // enhanced buffer transition barriers.
    rhi::HeapType heapType = rhi::HeapType::DeviceLocal;
    // Frame-only report from an external producer. Its complete incoming states
    // supersede the last graph submission, transactionally at batch commit.
    bool authoritativeIncoming = false;
};

struct PreparedBatchBarriers {
    struct BeforePass {
        std::vector<rhi::TextureBarrier> textures;
        std::vector<rhi::BufferBarrier> buffers;
    };
    // Indexed in the same order as CompiledGraph::batches[i].passes.
    std::vector<BeforePass> beforePass;
    std::vector<BeforePass> afterPass;
    // Batch-entry compatibility barriers. New symbolic transitions are always
    // assigned to beforePass; these remain for imported/backend policy work.
    std::vector<rhi::TextureBarrier> textures;
    std::vector<rhi::BufferBarrier> buffers;
    std::vector<PreparedStateRegion> committedStates;
    std::vector<rhi::ResourceHandle> committedResources;
    // Queue slot that performed each committed state update. Kept parallel to
    // committedStates/resources so the admission ledger can distinguish a
    // same-queue memory dependency from a timeline-ordered queue handoff.
    std::vector<uint32_t> committedQueues;
    std::vector<PreparedBackingState> seeds;
    std::vector<rhi::ResourceHandle> authoritativeSeeds;
};

struct PreparedExecutionBarrierPlan {
    std::vector<PreparedBatchBarriers> batches;
};

// Ordered admission state. Compile workers never access this object. Prepare is
// transactional; CommitBatch is called only for work known to have submitted.
class BackingStateAdmissionLedger {
public:
    // closeToHome: the execution is closed. Every resource it touches leaves it in its home state - the
    // backing's seeded region state, or the state its first access in the graph expects - so the next
    // execution's entry states are the same whatever ran before it (or whether anything did). Textures get
    // a layout transition after their last access; the ledger records home as their final state. Closed
    // executions provide their own memory visibility (a full barrier at entry, see FrameAdmission::closed),
    // so buffers need no exit barrier. Every resource must be used from a single queue.
    PreparedExecutionBarrierPlan Prepare(
        const CompiledGraph& graph, std::span<const PreparedBackingState> initial,
        std::span<const rhi::ResourceHandle> invalidated = {}, bool closeToHome = false) const;
    void CommitBatch(const PreparedBatchBarriers&);
    void Invalidate(std::span<const rhi::ResourceHandle> resources);
    size_t BackingCount() const noexcept { return m_states.size(); }
    void Reset() { m_states.clear(); }
private:
    struct StateGrid {
        CompileResourceShape shape{};
        std::vector<CompileResourceState> cells;
        std::vector<uint32_t> queues;
    };
    std::unordered_map<uint64_t, StateGrid> m_states;
};

// Ordered, submission-backed cross-frame hazard state. Unlike symbolic graph
// dependencies, these points refer only to work that was actually submitted.
// The ledger is keyed by concrete backing identity and subresource cell so
// unrelated resources/ranges and queues remain independent.
class BackingAccessAdmissionLedger {
public:
    void AppendIncomingWaits(const CompiledGraph& graph,
        std::span<const PreparedBackingState> resources,
        std::span<const ExecutionTimelinePoint> queues,
        std::vector<std::vector<ExecutionTimelinePoint>>& incoming) const;
    void Commit(const CompiledGraph& graph,
        std::span<const PreparedBackingState> resources,
        const GraphExecutionTimeline& execution, uint32_t batchCount = UINT32_MAX);
    void Reset() { m_accesses.clear(); }
    void ResolvePlannedPoints(const std::unordered_map<uint64_t, ExecutionTimelinePoint>& points);
    bool RetireCompleted(rhi::ResourceHandle resource, const std::map<uint64_t, uint64_t>& completed);
    size_t BackingCount() const noexcept { return m_accesses.size(); }
private:
    struct Cell {
        ExecutionTimelinePoint writer{};
        std::vector<ExecutionTimelinePoint> readers;
    };
    struct AccessGrid {
        CompileResourceShape shape{};
        std::vector<Cell> cells;
    };
    std::unordered_map<uint64_t, AccessGrid> m_accesses;
};

// Tracks submitted use of physical placed-resource ranges. Different graph
// resource handles that overlap the same heap generation conflict even for
// read/read access because they represent distinct alias occupants.
//
// Representation: one occupant per (heap, handle) with the placements it has
// committed and its latest signal per timeline. Committing assigns a monotonic
// sequence to the placement; a query range conflicts with every other
// occupant range that overlaps it and was committed after the query occupant
// last committed that same range. This is equivalent to the byte-interval
// ownership model (a later commit takes over the overlapping bytes) but needs
// no interval fragmentation, so steady-state placements cost O(occupants).
class AliasAccessAdmissionLedger {
public:
    std::vector<rhi::ResourceHandle> ApplyInitialStates(
        const CompiledGraph& graph,
        std::span<const PreparedBackingState> resources) const;
    void AppendIncomingWaits(const CompiledGraph& graph,
        std::span<const PreparedBackingState> resources,
        std::span<const ExecutionTimelinePoint> queues,
        std::vector<std::vector<ExecutionTimelinePoint>>& incoming) const;
    void Commit(const CompiledGraph& graph,
        std::span<const PreparedBackingState> resources,
        const GraphExecutionTimeline& execution, uint32_t batchCount = UINT32_MAX);
    void Reset() { m_heaps.clear(); m_heapOwners.clear(); }
    void ResolvePlannedPoints(const std::unordered_map<uint64_t, ExecutionTimelinePoint>& points);
    bool RetireCompleted(const AliasHeapGeneration* heap, const std::weak_ptr<const AliasHeapGeneration>& owner,
        const std::map<uint64_t, uint64_t>& completed);
    size_t HeapCount() const noexcept { return m_heaps.size(); }
private:
    struct OccupantRange {
        uint64_t begin = 0, end = 0;
        uint64_t sequence = 0; // Commit order of the latest submission using this placement.
    };
    struct Occupant {
        rhi::ResourceHandle handle{};
        std::vector<OccupantRange> ranges;             // Usually one; more after a re-placement.
        std::vector<ExecutionTimelinePoint> accesses;  // Latest signal per timeline.
    };
    struct Heap {
        std::vector<Occupant> occupants;
        std::unordered_map<uint64_t, uint32_t> occupantByHandle;
    };
    // Sequence of the query occupant's own commit at exactly this placement,
    // zero when it never committed there. Ranges committed later conflict.
    static uint64_t OwnSequence(const Heap& heap, const PreparedBackingState& resource);
    // Visits every other occupant with a range overlapping the query placement
    // that was committed after the query occupant's own commit there.
    template<class Fn> void ForEachConflict(const PreparedBackingState& resource, Fn&& fn) const;
    std::unordered_map<const AliasHeapGeneration*, Heap> m_heaps;
    std::unordered_map<const AliasHeapGeneration*, std::weak_ptr<const AliasHeapGeneration>> m_heapOwners;
    uint64_t m_sequence = 0;
};

} // namespace org::experimental
