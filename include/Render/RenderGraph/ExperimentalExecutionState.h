#pragma once

#include "Render/RenderGraph/ExperimentalGraphCompiler.h"
#include "Resources/AliasingPlacement.h"
#include <rhi.h>
#include <span>
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
    std::vector<PreparedStateRegion> regions;
    std::shared_ptr<const AliasHeapGeneration> aliasHeap;
    const AliasHeapGeneration* aliasHeapIdentity = nullptr;
    uint64_t aliasPoolID = 0;
    uint64_t aliasOffset = 0;
    uint64_t aliasSize = 0;
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
    std::vector<PreparedBackingState> seeds;
};

struct PreparedExecutionBarrierPlan {
    std::vector<PreparedBatchBarriers> batches;
};

// Ordered admission state. Compile workers never access this object. Prepare is
// transactional; CommitBatch is called only for work known to have submitted.
class BackingStateAdmissionLedger {
public:
    PreparedExecutionBarrierPlan Prepare(
        const CompiledGraph& graph, std::span<const PreparedBackingState> initial) const;
    void CommitBatch(const PreparedBatchBarriers&);
    void Invalidate(std::span<const rhi::ResourceHandle> resources);
    void Reset() { m_states.clear(); }
private:
    struct StateGrid {
        CompileResourceShape shape{};
        std::vector<CompileResourceState> cells;
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

// Tracks submitted use of physical placed-resource intervals. Different graph
// resource handles that overlap the same heap generation conflict even for
// read/read access because they represent distinct alias occupants.
class AliasAccessAdmissionLedger {
public:
    std::vector<rhi::ResourceHandle> ApplyInitialStates(
        const CompiledGraph& graph, std::vector<PreparedBackingState>& resources) const;
    void AppendIncomingWaits(const CompiledGraph& graph,
        std::span<const PreparedBackingState> resources,
        std::span<const ExecutionTimelinePoint> queues,
        std::vector<std::vector<ExecutionTimelinePoint>>& incoming) const;
    void Commit(const CompiledGraph& graph,
        std::span<const PreparedBackingState> resources,
        const GraphExecutionTimeline& execution, uint32_t batchCount = UINT32_MAX);
    void Reset() { m_intervals.clear(); }
private:
    struct SubmittedInterval {
        uint64_t begin = 0, end = 0;
        rhi::ResourceHandle occupant{};
        std::vector<ExecutionTimelinePoint> accesses;
    };
    std::unordered_map<const AliasHeapGeneration*, std::vector<SubmittedInterval>> m_intervals;
};

} // namespace org::experimental
