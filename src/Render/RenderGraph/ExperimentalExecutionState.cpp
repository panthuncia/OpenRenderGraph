#include "Render/RenderGraph/ExperimentalExecutionState.h"
#include "Render/BufferBarrierHelpers.h"

#include <algorithm>
#include <stdexcept>
#include <BasicTelemetry/Tracy.h>

namespace org::experimental {
namespace {
uint64_t Key(rhi::ResourceHandle handle) {
    if (!handle.valid()) throw std::invalid_argument("Invalid prepared backing handle");
    return (uint64_t{handle.generation} << 32) | handle.index;
}
bool SameHandle(rhi::ResourceHandle lhs, rhi::ResourceHandle rhs) {
    return lhs.index == rhs.index && lhs.generation == rhs.generation;
}
size_t CellIndex(const CompileResourceShape& shape, uint32_t mip, uint32_t slice) {
    return size_t{slice} * shape.mips + mip;
}
void Visit(const CompileResourceShape& shape, CompileRange range, auto&& fn) {
    if (!range.mips || !range.slices || range.mip + range.mips > shape.mips
        || range.slice + range.slices > shape.slices)
        throw std::invalid_argument("Invalid prepared backing state range");
    for (uint32_t slice = range.slice; slice < range.slice + range.slices; ++slice)
        for (uint32_t mip = range.mip; mip < range.mip + range.mips; ++mip)
            fn(mip, slice);
}

std::span<const PreparedBackingState> ValidatePreparedBackings(
    const GraphCompileStructure& structure, std::span<const PreparedBackingState> resources) {
    if (resources.size() != structure.resourceIDs.size())
        throw std::invalid_argument("Incomplete prepared backing input");
    for (size_t i = 0; i < resources.size(); ++i) {
        const std::string_view expectedKey = structure.resourceKeys.empty()
            ? std::string_view{} : std::string_view{structure.resourceKeys[i]};
        if (resources[i].graphResourceID != structure.resourceIDs[i]
            || resources[i].graphResourceKey != expectedKey)
            throw std::invalid_argument("Prepared backings are not in compiler resource order");
    }
    return resources;
}
}

PreparedExecutionBarrierPlan BackingStateAdmissionLedger::Prepare(
    const CompiledGraph& graph, std::span<const PreparedBackingState> initial,
    std::span<const rhi::ResourceHandle> invalidated) const {
    BT_ZONE_SCOPE("ORG.Execution.ResolveBackingStates");
    if (!graph.structure || !graph.states.complete || initial.size() != graph.structure->resourceIDs.size())
        throw std::invalid_argument("Incomplete symbolic state admission input");
    const auto ordered = ValidatePreparedBackings(*graph.structure, initial);
    PreparedExecutionBarrierPlan result;
    result.batches.resize(graph.batches.size());
    if (graph.batchByPass.size() != graph.structure->passes.size()
        || graph.positionByPass.size() != graph.structure->passes.size()
        || graph.barrierCapacityByPass.size() != graph.structure->passes.size()
        || graph.stateDeltaCapacityByBatch.size() != graph.batches.size()
        || graph.stateSeedCapacityByBatch.size() != graph.batches.size())
        throw std::invalid_argument("Compiled graph has no execution placement index");
    for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
        result.batches[batch].beforePass.resize(graph.batches[batch].passes.size());
        result.batches[batch].afterPass.resize(graph.batches[batch].passes.size());
        for (uint32_t position = 0; position < graph.batches[batch].passes.size(); ++position) {
            const auto pass = graph.batches[batch].passes[position];
            if (pass >= graph.batchByPass.size() || graph.batchByPass[pass] != batch
                || graph.positionByPass[pass] != position)
                throw std::invalid_argument("Invalid symbolic pass placement during admission");
            const auto& capacity = graph.barrierCapacityByPass[pass];
            result.batches[batch].beforePass[position].textures.reserve(capacity.beforeTextures);
            result.batches[batch].beforePass[position].buffers.reserve(capacity.beforeBuffers);
            result.batches[batch].afterPass[position].textures.reserve(capacity.afterTextures);
        }
        result.batches[batch].committedResources.reserve(graph.stateDeltaCapacityByBatch[batch]);
        result.batches[batch].committedStates.reserve(graph.stateDeltaCapacityByBatch[batch]);
        result.batches[batch].committedQueues.reserve(graph.stateDeltaCapacityByBatch[batch]);
        result.batches[batch].seeds.reserve(graph.stateSeedCapacityByBatch[batch]);
    }
    const auto wasInvalidated = [&](rhi::ResourceHandle handle) {
        return std::ranges::any_of(invalidated, [&](auto other) { return SameHandle(handle, other); });
    };
    for (size_t resource = 0; resource < initial.size(); ++resource) {
        const auto& captured = ordered[resource];
        const auto key = Key(captured.resource);
        auto existing = m_states.find(key);
        if (existing != m_states.end() && !wasInvalidated(captured.resource)
            && existing->second.shape != captured.shape)
            throw std::invalid_argument("Backing shape changed without a new backing identity");
    }
    uint64_t entrySteps = 0, intraBatchSteps = 0, crossQueueSteps = 0;
    for (const auto& step : graph.states.steps) {
        if (step.resource >= ordered.size() || step.batch >= result.batches.size())
            throw std::invalid_argument("Invalid symbolic state step");
        const auto& captured = ordered[step.resource];
        if (step.pass >= graph.batchByPass.size() || graph.batchByPass[step.pass] != step.batch)
            throw std::invalid_argument("Symbolic state step has no consuming pass placement");
        auto& output = result.batches[step.batch];
        auto& beforePass = output.beforePass[graph.positionByPass[step.pass]];
        const bool crossQueue = step.previousBatch != UINT32_MAX
            && graph.batches[step.previousBatch].queue != graph.batches[step.batch].queue;
        // A buffer state imported at the beginning of an execution has no producer
        // batch in this plan. A copy list cannot name shader access as AccessBefore,
        // so use the queue-neutral boundary access after admission has supplied the
        // inter-execution timeline ordering. Texture layouts require an explicit
        // producer-side release and are intentionally not rewritten here.
        const bool entryCopyQueueAcquire = step.previousBatch == UINT32_MAX
            && graph.batches[step.batch].queue == 2;
        PreparedBatchBarriers::BeforePass* afterProducer = nullptr;
        if (crossQueue) {
            if (step.previousPass >= graph.batchByPass.size()
                || graph.batchByPass[step.previousPass] != step.previousBatch)
                throw std::invalid_argument("Cross-queue state step has no producer pass placement");
            afterProducer = &result.batches[step.previousBatch].afterPass[graph.positionByPass[step.previousPass]];
            ++crossQueueSteps;
        }
        ++entrySteps;
        intraBatchSteps += step.previousBatch == step.batch;
        const auto handle = captured.resource;
        const bool boundaryStep = step.previousBatch == UINT32_MAX;
        const bool invalidatedState = boundaryStep && wasInvalidated(handle);
        if (boundaryStep
            && (!m_states.contains(Key(handle)) || invalidatedState)
            && std::none_of(output.seeds.begin(), output.seeds.end(),
                [&](const auto& seed) { return Key(seed.resource) == Key(handle); }))
            output.seeds.push_back(captured);
        const auto stateKey = Key(handle);
        const auto existing = boundaryStep ? m_states.find(stateKey) : m_states.end();
        const bool useSubmittedState = boundaryStep
            && existing != m_states.end() && !invalidatedState;
        // Intra-frame state is compiler-invariant. Encode its complete range
        // once instead of replaying one identical barrier per mip/slice cell.
        if (!boundaryStep) {
            const auto consumerQueue = graph.batches[step.batch].queue;
            const auto previousQueue = graph.batches[step.previousBatch].queue;
            if (captured.shape.hasLayout) {
                rhi::TextureBarrier barrier{};
                barrier.texture = handle;
                barrier.range = {step.range.mip, step.range.mips,
                    step.range.slice, step.range.slices};
                barrier.beforeAccess = crossQueue ? rhi::ResourceAccessType::Common
                    : static_cast<rhi::ResourceAccessType>(step.before.access);
                barrier.afterAccess = static_cast<rhi::ResourceAccessType>(step.after.access);
                barrier.beforeLayout = crossQueue ? rhi::ResourceLayout::Common
                    : static_cast<rhi::ResourceLayout>(step.before.layout);
                barrier.afterLayout = static_cast<rhi::ResourceLayout>(step.after.layout);
                barrier.beforeSync = crossQueue ? rhi::ResourceSyncState::All
                    : static_cast<rhi::ResourceSyncState>(step.before.sync);
                barrier.afterSync = static_cast<rhi::ResourceSyncState>(step.after.sync);
                barrier.discard = barrier.beforeLayout == rhi::ResourceLayout::Undefined;
                if (crossQueue) {
                    auto release = barrier;
                    release.beforeAccess = static_cast<rhi::ResourceAccessType>(step.before.access);
                    release.afterAccess = rhi::ResourceAccessType::Common;
                    release.beforeLayout = static_cast<rhi::ResourceLayout>(step.before.layout);
                    release.afterLayout = rhi::ResourceLayout::Common;
                    release.beforeSync = static_cast<rhi::ResourceSyncState>(step.before.sync);
                    release.afterSync = rhi::ResourceSyncState::All;
                    release.discard = release.beforeLayout == rhi::ResourceLayout::Undefined;
                    afterProducer->textures.push_back(release);
                }
                beforePass.textures.push_back(barrier);
            } else {
                const bool fixedHeapState = captured.heapType == rhi::HeapType::Upload
                    || captured.heapType == rhi::HeapType::Readback;
                const auto beforeAccess = static_cast<rhi::ResourceAccessType>(step.before.access);
                const auto beforeSync = static_cast<rhi::ResourceSyncState>(step.before.sync);
                const auto afterAccess = static_cast<rhi::ResourceAccessType>(step.after.access);
                const auto afterSync = static_cast<rhi::ResourceSyncState>(step.after.sync);
                if (!fixedHeapState && previousQueue == consumerQueue
                    && beforeAccess != rhi::ResourceAccessType::None
                    && beforeAccess != rhi::ResourceAccessType::Common
                    && afterAccess != rhi::ResourceAccessType::None
                    && afterAccess != rhi::ResourceAccessType::Common
                    && NeedsWholeBufferBarrier(beforeAccess, afterAccess, beforeSync, afterSync))
                    beforePass.buffers.push_back(MakeWholeBufferBarrier(handle,
                        beforeAccess, afterAccess, beforeSync, afterSync));
            }
            output.committedResources.push_back(handle);
            output.committedStates.push_back({step.range, step.after});
            output.committedQueues.push_back(consumerQueue);
            continue;
        }
        std::optional<CompileResourceState> uniformBefore;
        uint32_t uniformPreviousQueue = UINT32_MAX;
        if (useSubmittedState) {
            bool uniform = true;
            Visit(captured.shape, step.range, [&](uint32_t mip, uint32_t slice) {
                const auto cell = CellIndex(captured.shape, mip, slice);
                if (!uniformBefore) {
                    uniformBefore = existing->second.cells[cell];
                    uniformPreviousQueue = existing->second.queues[cell];
                } else if (*uniformBefore != existing->second.cells[cell]
                    || uniformPreviousQueue != existing->second.queues[cell]) uniform = false;
            });
            if (!uniform) uniformBefore.reset();
        } else if (invalidatedState) {
            uniformBefore = {static_cast<uint64_t>(rhi::ResourceAccessType::None),
                static_cast<uint64_t>(captured.shape.hasLayout
                    ? rhi::ResourceLayout::Undefined : rhi::ResourceLayout::Common),
                static_cast<uint64_t>(rhi::ResourceSyncState::None), false};
        } else if (captured.regions) {
            const auto rangeEndMip = step.range.mip + step.range.mips;
            const auto rangeEndSlice = step.range.slice + step.range.slices;
            for (const auto& region : *captured.regions) {
                if (region.range.mip <= step.range.mip
                    && region.range.mip + region.range.mips >= rangeEndMip
                    && region.range.slice <= step.range.slice
                    && region.range.slice + region.range.slices >= rangeEndSlice) {
                    uniformBefore = region.state;
                    break;
                }
            }
        }
        if (uniformBefore) {
            const auto consumerQueue = graph.batches[step.batch].queue;
            if (captured.shape.hasLayout) {
                rhi::TextureBarrier barrier{};
                barrier.texture = handle;
                barrier.range = {step.range.mip, step.range.mips,
                    step.range.slice, step.range.slices};
                barrier.beforeAccess = static_cast<rhi::ResourceAccessType>(uniformBefore->access);
                barrier.afterAccess = static_cast<rhi::ResourceAccessType>(step.after.access);
                barrier.beforeLayout = static_cast<rhi::ResourceLayout>(uniformBefore->layout);
                barrier.afterLayout = static_cast<rhi::ResourceLayout>(step.after.layout);
                barrier.beforeSync = static_cast<rhi::ResourceSyncState>(uniformBefore->sync);
                barrier.afterSync = static_cast<rhi::ResourceSyncState>(step.after.sync);
                barrier.discard = barrier.beforeLayout == rhi::ResourceLayout::Undefined;
                beforePass.textures.push_back(barrier);
            } else {
                const auto beforeAccess = static_cast<rhi::ResourceAccessType>(uniformBefore->access);
                const auto beforeSync = static_cast<rhi::ResourceSyncState>(uniformBefore->sync);
                const auto afterAccess = static_cast<rhi::ResourceAccessType>(step.after.access);
                const auto afterSync = static_cast<rhi::ResourceSyncState>(step.after.sync);
                const bool fixedHeapState = captured.heapType == rhi::HeapType::Upload
                    || captured.heapType == rhi::HeapType::Readback;
                const bool queueHandoff = uniformPreviousQueue != UINT32_MAX
                    && uniformPreviousQueue != consumerQueue;
                const bool copyCompatible = consumerQueue != 2
                    || beforeSync == rhi::ResourceSyncState::None
                    || beforeSync == rhi::ResourceSyncState::Copy;
                if (!fixedHeapState && copyCompatible && !queueHandoff
                    && beforeAccess != rhi::ResourceAccessType::None
                    && beforeAccess != rhi::ResourceAccessType::Common
                    && afterAccess != rhi::ResourceAccessType::None
                    && afterAccess != rhi::ResourceAccessType::Common
                    && NeedsWholeBufferBarrier(beforeAccess, afterAccess, beforeSync, afterSync))
                    beforePass.buffers.push_back(MakeWholeBufferBarrier(handle,
                        beforeAccess, afterAccess, beforeSync, afterSync));
            }
            output.committedResources.push_back(handle);
            output.committedStates.push_back({step.range, step.after});
            output.committedQueues.push_back(consumerQueue);
            continue;
        }
        Visit(captured.shape, step.range, [&](uint32_t mip, uint32_t slice) {
            const auto cellIndex = CellIndex(captured.shape, mip, slice);
            CompileResourceState before = step.before;
            uint32_t previousQueue = step.previousBatch == UINT32_MAX ? UINT32_MAX
                : graph.batches[step.previousBatch].queue;
            if (useSubmittedState) {
                before = existing->second.cells[cellIndex];
                previousQueue = existing->second.queues[cellIndex];
            } else if (boundaryStep && invalidatedState) {
                before = {static_cast<uint64_t>(rhi::ResourceAccessType::None),
                    static_cast<uint64_t>(captured.shape.hasLayout
                        ? rhi::ResourceLayout::Undefined : rhi::ResourceLayout::Common),
                    static_cast<uint64_t>(rhi::ResourceSyncState::None), false};
            } else if (boundaryStep) {
                if (captured.regions) for (const auto& region : *captured.regions) {
                    if (mip >= region.range.mip && mip < region.range.mip + region.range.mips
                        && slice >= region.range.slice && slice < region.range.slice + region.range.slices) {
                        before = region.state;
                        break;
                    }
                }
            }
            const auto consumerQueue = graph.batches[step.batch].queue;
            const bool submittedQueueHandoff = previousQueue != UINT32_MAX
                && previousQueue != consumerQueue;
            if (captured.shape.hasLayout) {
                rhi::TextureBarrier barrier{};
                barrier.texture = handle;
                barrier.range = {mip, 1, slice, 1};
                barrier.beforeAccess = crossQueue ? rhi::ResourceAccessType::Common
                    : static_cast<rhi::ResourceAccessType>(before.access);
                barrier.afterAccess = static_cast<rhi::ResourceAccessType>(step.after.access);
                barrier.beforeLayout = crossQueue ? rhi::ResourceLayout::Common
                    : static_cast<rhi::ResourceLayout>(before.layout);
                barrier.afterLayout = static_cast<rhi::ResourceLayout>(step.after.layout);
                barrier.beforeSync = crossQueue ? rhi::ResourceSyncState::All
                    : static_cast<rhi::ResourceSyncState>(before.sync);
                barrier.afterSync = static_cast<rhi::ResourceSyncState>(step.after.sync);
                barrier.discard = barrier.beforeLayout == rhi::ResourceLayout::Undefined;
                if (crossQueue) {
                    auto release = barrier;
                    release.beforeAccess = static_cast<rhi::ResourceAccessType>(before.access);
                    release.afterAccess = rhi::ResourceAccessType::Common;
                    release.beforeLayout = static_cast<rhi::ResourceLayout>(before.layout);
                    release.afterLayout = rhi::ResourceLayout::Common;
                    release.beforeSync = static_cast<rhi::ResourceSyncState>(before.sync);
                    release.afterSync = rhi::ResourceSyncState::All;
                    release.discard = release.beforeLayout == rhi::ResourceLayout::Undefined;
                    afterProducer->textures.push_back(release);
                }
                beforePass.textures.push_back(barrier);
            } else {
                const auto heapType = captured.heapType;
                const bool fixedHeapState = heapType == rhi::HeapType::Upload
                    || heapType == rhi::HeapType::Readback;
                const auto beforeAccess = static_cast<rhi::ResourceAccessType>(before.access);
                const auto beforeSync = static_cast<rhi::ResourceSyncState>(before.sync);
                const auto afterAccess = static_cast<rhi::ResourceAccessType>(step.after.access);
                const auto afterSync = static_cast<rhi::ResourceSyncState>(step.after.sync);
                const bool copyQueueCompatibleBefore = graph.batches[step.batch].queue != 2
                    || beforeSync == rhi::ResourceSyncState::None
                    || beforeSync == rhi::ResourceSyncState::Copy;
                // Buffers have no layout or queue ownership state in D3D12. The
                // timeline edge orders a cross-queue producer, after which the
                // consumer can express the actual producer and consumer scopes in
                // one barrier. Splitting through COMMON manufactures NO_ACCESS
                // buffer barriers, which are invalid enhanced barriers. Vulkan's
                // queue-family ownership is represented separately by the explicit
                // execution timeline/import policy rather than this layout split.
                // A newly-created buffer has no prior GPU access or layout to
                // transition. D3D12 enhanced buffer barriers have no discard
                // operation, and a NONE/NO_ACCESS before scope is not a legal
                // synchronization barrier. The first access itself establishes
                // state; subsequent and cross-queue accesses use the common
                // helper shared with the synchronous recorder.
                if (!fixedHeapState
                    && copyQueueCompatibleBefore
                    && !submittedQueueHandoff
                    && beforeAccess != rhi::ResourceAccessType::None
                    && beforeAccess != rhi::ResourceAccessType::Common
                    && afterAccess != rhi::ResourceAccessType::None
                    && afterAccess != rhi::ResourceAccessType::Common
                    && NeedsWholeBufferBarrier(beforeAccess, afterAccess, beforeSync, afterSync))
                    beforePass.buffers.push_back(MakeWholeBufferBarrier(handle,
                        beforeAccess, afterAccess, beforeSync, afterSync));
            }
        });
        output.committedResources.push_back(handle);
        output.committedStates.push_back({step.range, step.after});
        output.committedQueues.push_back(graph.batches[step.batch].queue);
    }
    BT_PLOT("ORG.Execution.StateBarrierSteps", static_cast<int64_t>(entrySteps));
    BT_PLOT("ORG.Execution.IntraBatchStateBarrierSteps", static_cast<int64_t>(intraBatchSteps));
    BT_PLOT("ORG.Execution.CrossQueueStateBarrierSteps", static_cast<int64_t>(crossQueueSteps));
    if (intraBatchSteps)
        basic_telemetry::AddCounter("ORG.Execution.IntraBatchStateBarrierSteps",
            static_cast<int64_t>(intraBatchSteps));
    return result;
}

void BackingStateAdmissionLedger::CommitBatch(const PreparedBatchBarriers& batch) {
    if (batch.committedResources.size() != batch.committedStates.size()
        || batch.committedQueues.size() != batch.committedStates.size())
        throw std::invalid_argument("Invalid backing-state commit");
    for (const auto& seed : batch.seeds) {
        const auto key = Key(seed.resource);
        if (m_states.contains(key)) continue;
        StateGrid grid;
        grid.shape = seed.shape;
        grid.cells.resize(size_t{grid.shape.mips} * grid.shape.slices);
        grid.queues.assign(grid.cells.size(), UINT32_MAX);
        if (seed.regions) for (const auto& region : *seed.regions)
            Visit(grid.shape, region.range, [&](uint32_t mip, uint32_t slice) {
                grid.cells[CellIndex(grid.shape, mip, slice)] = region.state;
            });
        m_states.emplace(key, std::move(grid));
    }
    for (size_t i = 0; i < batch.committedStates.size(); ++i) {
        const auto key = Key(batch.committedResources[i]);
        auto found = m_states.find(key);
        if (found == m_states.end())
            throw std::logic_error("Backing state was not seeded before commit");
        const auto& update = batch.committedStates[i];
        Visit(found->second.shape, update.range, [&](uint32_t mip, uint32_t slice) {
            const auto cellIndex = CellIndex(found->second.shape, mip, slice);
            found->second.cells[cellIndex] = update.state;
            found->second.queues[cellIndex] = batch.committedQueues[i];
        });
    }
}

void BackingStateAdmissionLedger::Invalidate(std::span<const rhi::ResourceHandle> resources) {
    for (const auto resource : resources) if (resource.valid()) m_states.erase(Key(resource));
}

namespace {
template<class Fn>
void VisitPassAccesses(const CompiledGraph& graph, uint32_t passIndex, Fn&& fn) {
    const auto& pass = graph.structure->passes.at(passIndex);
    if (!pass.entryStates.empty()) {
        for (const auto& use : pass.entryStates) fn(use.resource, use.range, use.state.write);
        return;
    }
    // Dependency-only test inputs may omit state declarations. Treat those
    // accesses as whole-resource ranges without weakening hazard coverage.
    for (const auto& access : pass.accesses) {
        const auto shape = graph.structure->resourceShapes.at(access.resourceIndex);
        fn(access.resourceIndex, CompileRange{0, shape.mips, 0, shape.slices}, access.write);
    }
}

std::span<const PreparedBackingState> OrderBackings(
    const CompiledGraph& graph, std::span<const PreparedBackingState> resources) {
    if (!graph.structure)
        throw std::invalid_argument("Incomplete backing access input");
    if (resources.size() != graph.structure->resourceIDs.size())
        throw std::invalid_argument("Incomplete prepared backing input");
    // BackingStateAdmissionLedger::Prepare validates the compiler order once
    // for the frame before the access and alias ledgers consume the same span.
    return resources;
}

void AppendWait(std::vector<ExecutionTimelinePoint>& waits, ExecutionTimelinePoint point,
    uint64_t consumerTimeline) {
    if (!point.timeline || !point.value || point.timeline == consumerTimeline) return;
    auto found = std::find_if(waits.begin(), waits.end(),
        [&](const auto& wait) { return wait.timeline == point.timeline; });
    if (found == waits.end()) waits.push_back(point);
    else found->value = (std::max)(found->value, point.value);
}
}

void BackingAccessAdmissionLedger::AppendIncomingWaits(const CompiledGraph& graph,
    std::span<const PreparedBackingState> resources,
    std::span<const ExecutionTimelinePoint> queues,
    std::vector<std::vector<ExecutionTimelinePoint>>& incoming) const {
    BT_ZONE_SCOPE("ORG.Execution.ResolveCrossFrameHazards");
    if (incoming.size() != graph.batches.size() || queues.size() != graph.structure->queues.size())
        throw std::invalid_argument("Invalid cross-frame hazard dimensions");
    const auto ordered = OrderBackings(graph, resources);
    if (graph.incomingBoundaryAccessesByBatch.size() != graph.batches.size())
        throw std::invalid_argument("Compiled graph has no boundary access stream");
    uint64_t waitsAdded = 0;
    for (uint32_t batchIndex = 0; batchIndex < graph.batches.size(); ++batchIndex) {
        const auto& batch = graph.batches[batchIndex];
        const auto consumerTimeline = queues[batch.queue].timeline;
        const auto before = incoming[batchIndex].size();
        for (const auto& access : graph.incomingBoundaryAccessesByBatch[batchIndex]) {
            const auto resourceIndex = access.resource;
            const auto range = access.range;
            const bool write = access.state.write;
                const auto key = Key(ordered[resourceIndex].resource);
                const auto found = m_accesses.find(key);
                if (found == m_accesses.end()) continue;
                if (found->second.shape != ordered[resourceIndex].shape)
                    throw std::invalid_argument("Backing access shape changed without identity change");
                Visit(found->second.shape, range, [&](uint32_t mip, uint32_t slice) {
                    const auto& cell = found->second.cells[CellIndex(found->second.shape, mip, slice)];
                    AppendWait(incoming[batchIndex], cell.writer, consumerTimeline);
                    if (write) for (const auto reader : cell.readers)
                        AppendWait(incoming[batchIndex], reader, consumerTimeline);
                });
        }
        waitsAdded += incoming[batchIndex].size() - before;
    }
    BT_PLOT("ORG.Execution.CrossFrameHazardWaits", static_cast<int64_t>(waitsAdded));
    if (waitsAdded) basic_telemetry::AddCounter(
        "ORG.Execution.CrossFrameHazardWaits", static_cast<int64_t>(waitsAdded));
}

void BackingAccessAdmissionLedger::Commit(const CompiledGraph& graph,
    std::span<const PreparedBackingState> resources,
    const GraphExecutionTimeline& execution, uint32_t batchCount) {
    BT_ZONE_SCOPE("ORG.Execution.CommitCrossFrameHazards");
    if (execution.batches.size() != graph.batches.size())
        throw std::invalid_argument("Invalid submitted hazard dimensions");
    const auto ordered = OrderBackings(graph, resources);
    if (graph.boundaryAccessesByBatch.size() != graph.batches.size())
        throw std::invalid_argument("Compiled graph has no boundary access stream");
    batchCount = (std::min)(batchCount, static_cast<uint32_t>(graph.batches.size()));
    for (uint32_t batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
        const auto signal = execution.batches[batchIndex].signal;
        if (!signal.timeline || !signal.value) throw std::invalid_argument("Unsubmitted hazard point");
        for (const auto& access : graph.boundaryAccessesByBatch[batchIndex]) {
            const auto resourceIndex = access.resource;
            const auto range = access.range;
            const bool write = access.state.write;
                const auto& backing = ordered[resourceIndex];
                auto [found, inserted] = m_accesses.try_emplace(Key(backing.resource));
                auto& grid = found->second;
                if (inserted) {
                    grid.shape = backing.shape;
                    grid.cells.resize(size_t{grid.shape.mips} * grid.shape.slices);
                } else if (grid.shape != backing.shape) {
                    throw std::invalid_argument("Backing access shape changed without identity change");
                }
                Visit(grid.shape, range, [&](uint32_t mip, uint32_t slice) {
                    auto& cell = grid.cells[CellIndex(grid.shape, mip, slice)];
                    if (write) {
                        cell.writer = signal;
                        cell.readers.clear();
                    } else {
                        auto reader = std::find_if(cell.readers.begin(), cell.readers.end(),
                            [&](const auto& point) { return point.timeline == signal.timeline; });
                        if (reader == cell.readers.end()) cell.readers.push_back(signal);
                        else if (reader->value < signal.value) *reader = signal;
                    }
                });
        }
    }
}

namespace {
bool Overlaps(uint64_t lhsBegin, uint64_t lhsEnd, uint64_t rhsBegin, uint64_t rhsEnd) {
    return (std::max)(lhsBegin, rhsBegin) < (std::min)(lhsEnd, rhsEnd);
}
}

std::vector<rhi::ResourceHandle> AliasAccessAdmissionLedger::ApplyInitialStates(
    const CompiledGraph& graph,
    std::span<const PreparedBackingState> resources) const {
    BT_ZONE_SCOPE("ORG.Execution.ResolveAliasActivations");
    const auto ordered = OrderBackings(graph, resources);
    if (graph.aliasFinalResourcesByBatch.size() != graph.batches.size())
        throw std::invalid_argument("Compiled graph has no alias final-use stream");
    std::vector<rhi::ResourceHandle> activated;
    for (const auto& resource : ordered) {
        if (!resource.aliasHeapIdentity || !resource.aliasSize) continue;
        const auto found = m_intervals.find(resource.aliasHeapIdentity);
        if (found == m_intervals.end()) continue;
        const uint64_t end = resource.aliasOffset + resource.aliasSize;
        const bool replacesOccupant = std::ranges::any_of(found->second, [&](const auto& prior) {
            return !SameHandle(prior.occupant, resource.resource)
                && Overlaps(resource.aliasOffset, end, prior.begin, prior.end);
        });
        if (!replacesOccupant) continue;
        activated.push_back(resource.resource);
    }
    BT_PLOT("ORG.Execution.AliasActivations", static_cast<int64_t>(activated.size()));
    if (!activated.empty()) basic_telemetry::AddCounter(
        "ORG.Execution.AliasActivations", static_cast<int64_t>(activated.size()));
    return activated;
}

void AliasAccessAdmissionLedger::AppendIncomingWaits(const CompiledGraph& graph,
    std::span<const PreparedBackingState> resources,
    std::span<const ExecutionTimelinePoint> queues,
    std::vector<std::vector<ExecutionTimelinePoint>>& incoming) const {
    BT_ZONE_SCOPE("ORG.Execution.ResolveAliasHazards");
    if (incoming.size() != graph.batches.size() || queues.size() != graph.structure->queues.size())
        throw std::invalid_argument("Invalid alias hazard dimensions");
    const auto ordered = OrderBackings(graph, resources);
    if (graph.aliasFirstResourcesByBatch.size() != graph.batches.size())
        throw std::invalid_argument("Compiled graph has no alias entry stream");
    uint64_t waitsAdded = 0, overlapsFound = 0;
    for (uint32_t batchIndex = 0; batchIndex < graph.batches.size(); ++batchIndex) {
        const auto& batch = graph.batches[batchIndex];
        const auto consumerTimeline = queues[batch.queue].timeline;
        const auto before = incoming[batchIndex].size();
        for (const auto resourceIndex : graph.aliasFirstResourcesByBatch[batchIndex]) {
                const auto& resource = ordered[resourceIndex];
                if (!resource.aliasHeapIdentity || !resource.aliasSize) continue;
                const auto found = m_intervals.find(resource.aliasHeapIdentity);
                if (found == m_intervals.end()) continue;
                const uint64_t end = resource.aliasOffset + resource.aliasSize;
                for (const auto& prior : found->second) {
                    if (SameHandle(prior.occupant, resource.resource)
                        || !Overlaps(resource.aliasOffset, end, prior.begin, prior.end)) continue;
                    ++overlapsFound;
                    for (const auto point : prior.accesses)
                        AppendWait(incoming[batchIndex], point, consumerTimeline);
                }
        }
        waitsAdded += incoming[batchIndex].size() - before;
    }
    BT_PLOT("ORG.Execution.AliasHazardWaits", static_cast<int64_t>(waitsAdded));
    BT_PLOT("ORG.Execution.AliasOverlaps", static_cast<int64_t>(overlapsFound));
    if (waitsAdded) basic_telemetry::AddCounter(
        "ORG.Execution.AliasHazardWaits", static_cast<int64_t>(waitsAdded));
}

void AliasAccessAdmissionLedger::Commit(const CompiledGraph& graph,
    std::span<const PreparedBackingState> resources,
    const GraphExecutionTimeline& execution, uint32_t batchCount) {
    BT_ZONE_SCOPE("ORG.Execution.CommitAliasHazards");
    if (execution.batches.size() != graph.batches.size())
        throw std::invalid_argument("Invalid submitted alias dimensions");
    const auto ordered = OrderBackings(graph, resources);
    if (graph.aliasFinalResourcesByBatch.size() != graph.batches.size())
        throw std::invalid_argument("Compiled graph has no alias final-use stream");
    batchCount = (std::min)(batchCount, static_cast<uint32_t>(graph.batches.size()));
    for (uint32_t batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
        const auto signal = execution.batches[batchIndex].signal;
        if (!signal.timeline || !signal.value) throw std::invalid_argument("Unsubmitted alias point");
        for (const auto resourceIndex : graph.aliasFinalResourcesByBatch[batchIndex]) {
                const auto& resource = ordered[resourceIndex];
                if (!resource.aliasHeapIdentity || !resource.aliasSize) continue;
                auto& intervals = m_intervals[resource.aliasHeapIdentity];
                const uint64_t end = resource.aliasOffset + resource.aliasSize;
                std::vector<SubmittedInterval> retainedFragments;
                for (auto it = intervals.begin(); it != intervals.end();) {
                    if (SameHandle(it->occupant, resource.resource)
                        || !Overlaps(resource.aliasOffset, end, it->begin, it->end)) {
                        ++it;
                        continue;
                    }
                    if (it->begin < resource.aliasOffset) {
                        auto left = *it;
                        left.end = resource.aliasOffset;
                        retainedFragments.push_back(std::move(left));
                    }
                    if (end < it->end) {
                        auto right = *it;
                        right.begin = end;
                        retainedFragments.push_back(std::move(right));
                    }
                    it = intervals.erase(it);
                }
                intervals.insert(intervals.end(),
                    std::make_move_iterator(retainedFragments.begin()),
                    std::make_move_iterator(retainedFragments.end()));
                auto found = std::find_if(intervals.begin(), intervals.end(), [&](const auto& interval) {
                    return SameHandle(interval.occupant, resource.resource)
                        && interval.begin == resource.aliasOffset && interval.end == end;
                });
                if (found == intervals.end()) {
                    intervals.push_back({resource.aliasOffset, end, resource.resource, {signal}});
                    continue;
                }
                auto point = std::find_if(found->accesses.begin(), found->accesses.end(),
                    [&](const auto& existing) { return existing.timeline == signal.timeline; });
                if (point == found->accesses.end()) found->accesses.push_back(signal);
                else if (point->value < signal.value) *point = signal;
        }
    }
}

void BackingAccessAdmissionLedger::ResolvePlannedPoints(
    const std::unordered_map<uint64_t, ExecutionTimelinePoint>& points) {
    auto resolve = [&](ExecutionTimelinePoint& point) {
        if (const auto found = points.find(point.value); found != points.end()) point = found->second;
    };
    for (auto& [_, grid] : m_accesses) for (auto& cell : grid.cells) {
        resolve(cell.writer);
        for (auto& reader : cell.readers) resolve(reader);
    }
}

void AliasAccessAdmissionLedger::ResolvePlannedPoints(
    const std::unordered_map<uint64_t, ExecutionTimelinePoint>& points) {
    for (auto& [_, intervals] : m_intervals) for (auto& interval : intervals)
        for (auto& point : interval.accesses)
            if (const auto found = points.find(point.value); found != points.end()) point = found->second;
}

} // namespace org::experimental
