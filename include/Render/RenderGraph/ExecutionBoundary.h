#pragma once

#include "Render/RenderGraph/ExperimentalExecutionState.h"
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace org::experimental {

// An execution boundary is a description of accesses, not a barrier. The
// submission owner resolves it against work actually accepted by the queue.
// Keeping this separate from recorded commands permits asynchronous recording
// without guessing the predecessor of a closed execution.
struct BoundaryResourceAccess {
    rhi::ResourceHandle backing{};
    CompileRange subresources{};
    uint64_t offset = 0;
    uint64_t size = UINT64_MAX;
    uint32_t aspects = 0; // zero: every aspect of the realized image
    CompileResourceState state{};
    bool image = false;
};

struct ExecutionBoundaryManifest {
    uint32_t queueSlot = 0;
    // Preserve the ordered access stream. In particular, a read followed by a
    // write must not lose its WAR dependency against preceding external reads.
    std::vector<BoundaryResourceAccess> accesses;
    // Own immutable backing/descriptor versions, not mutable Resource objects.
    std::vector<std::shared_ptr<const void>> leases;
};

inline std::shared_ptr<const ExecutionBoundaryManifest> BuildExecutionBoundaryManifest(
    const CompiledGraph& graph, uint32_t batch,
    std::span<const PreparedBackingState> backings,
    std::vector<std::shared_ptr<const void>> leases = {}) {
    if (!graph.structure || backings.size() != graph.structure->resourceIDs.size()
        || graph.boundaryAccessesByBatch.size() != graph.batches.size())
        throw std::invalid_argument("Incomplete execution boundary metadata");
    auto result = std::make_shared<ExecutionBoundaryManifest>();
    result->queueSlot = graph.batches.at(batch).queue;
    result->leases = std::move(leases);
    const auto& accesses = graph.boundaryAccessesByBatch.at(batch);
    result->accesses.reserve(accesses.size());
    for (const auto& access : accesses) {
        if (access.resource >= backings.size())
            throw std::invalid_argument("Execution boundary references an unknown backing");
        const auto& backing = backings[access.resource];
        if (!backing.resource.valid() || !backing.shape.mips || !backing.shape.slices
            || !access.range.mips || !access.range.slices
            || access.range.mip >= backing.shape.mips
            || access.range.mips > backing.shape.mips - access.range.mip
            || access.range.slice >= backing.shape.slices
            || access.range.slices > backing.shape.slices - access.range.slice)
            throw std::invalid_argument("Invalid execution boundary resource range");
        BoundaryResourceAccess concrete;
        concrete.backing = backing.resource;
        concrete.subresources = access.range;
        concrete.state = access.state;
        concrete.image = backing.shape.hasLayout;
        if (!access.byteSize || (access.byteSize != UINT64_MAX && access.byteOffset > UINT64_MAX - access.byteSize)
            || (concrete.image && (access.byteOffset || access.byteSize != UINT64_MAX))
            || (!concrete.image && access.aspects))
            throw std::invalid_argument("Invalid execution boundary access extent");
        concrete.offset = access.byteOffset;
        concrete.size = access.byteSize;
        concrete.aspects = access.aspects;
        result->accesses.push_back(concrete);
    }
    return result;
}

} // namespace org::experimental
