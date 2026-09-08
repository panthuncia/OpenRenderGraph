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
}

PreparedExecutionBarrierPlan BackingStateAdmissionLedger::Prepare(
    const CompiledGraph& graph, std::span<const PreparedBackingState> initial) const {
    BT_ZONE_SCOPE("ORG.AsyncExecution.ResolveBackingStates");
    if (!graph.structure || !graph.states.complete || initial.size() != graph.structure->resourceIDs.size())
        throw std::invalid_argument("Incomplete symbolic state admission input");
    std::vector<const PreparedBackingState*> ordered(initial.size());
    for (size_t resource = 0; resource < graph.structure->resourceIDs.size(); ++resource) {
        const auto expectedID = graph.structure->resourceIDs[resource];
        const std::string_view expectedKey = graph.structure->resourceKeys.empty()
            ? std::string_view{} : std::string_view{graph.structure->resourceKeys[resource]};
        for (const auto& captured : initial) {
            if (captured.graphResourceID == expectedID && captured.graphResourceKey == expectedKey) {
                if (ordered[resource]) throw std::invalid_argument("Duplicate prepared backing identity");
                ordered[resource] = &captured;
            }
        }
        if (!ordered[resource]) throw std::invalid_argument("Missing prepared backing identity");
    }
    PreparedExecutionBarrierPlan result;
    result.batches.resize(graph.batches.size());
    std::vector<uint32_t> batchByPass(graph.structure->passes.size(), UINT32_MAX);
    std::vector<uint32_t> positionByPass(graph.structure->passes.size(), UINT32_MAX);
    for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
        result.batches[batch].beforePass.resize(graph.batches[batch].passes.size());
        result.batches[batch].afterPass.resize(graph.batches[batch].passes.size());
        for (uint32_t position = 0; position < graph.batches[batch].passes.size(); ++position) {
            const auto pass = graph.batches[batch].passes[position];
            if (pass >= batchByPass.size() || batchByPass[pass] != UINT32_MAX)
                throw std::invalid_argument("Invalid symbolic pass placement during admission");
            batchByPass[pass] = batch;
            positionByPass[pass] = position;
        }
    }
    std::vector<StateGrid> projected(initial.size());
    for (size_t resource = 0; resource < initial.size(); ++resource) {
        const auto& captured = *ordered[resource];
        const auto key = Key(captured.resource);
        auto existing = m_states.find(key);
        if (existing != m_states.end()) projected[resource] = existing->second;
        else {
            auto& grid = projected[resource];
            grid.shape = captured.shape;
            grid.cells.resize(size_t{grid.shape.mips} * grid.shape.slices);
            grid.queues.assign(grid.cells.size(), UINT32_MAX);
            for (const auto& region : captured.regions)
                Visit(grid.shape, region.range, [&](uint32_t mip, uint32_t slice) {
                    grid.cells[CellIndex(grid.shape, mip, slice)] = region.state;
                });
        }
        if (projected[resource].shape != captured.shape)
            throw std::invalid_argument("Backing shape changed without a new backing identity");
        if (projected[resource].queues.size() != projected[resource].cells.size())
            projected[resource].queues.assign(projected[resource].cells.size(), UINT32_MAX);
    }
    uint64_t entrySteps = 0, intraBatchSteps = 0, crossQueueSteps = 0;
    for (const auto& step : graph.states.steps) {
        if (step.resource >= projected.size() || step.batch >= result.batches.size())
            throw std::invalid_argument("Invalid symbolic state step");
        auto& grid = projected[step.resource];
        if (step.pass >= batchByPass.size() || batchByPass[step.pass] != step.batch)
            throw std::invalid_argument("Symbolic state step has no consuming pass placement");
        auto& output = result.batches[step.batch];
        auto& beforePass = output.beforePass[positionByPass[step.pass]];
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
            if (step.previousPass >= batchByPass.size()
                || batchByPass[step.previousPass] != step.previousBatch)
                throw std::invalid_argument("Cross-queue state step has no producer pass placement");
            afterProducer = &result.batches[step.previousBatch].afterPass[positionByPass[step.previousPass]];
            ++crossQueueSteps;
        }
        ++entrySteps;
        intraBatchSteps += step.previousBatch == step.batch;
        const auto handle = ordered[step.resource]->resource;
        if (!m_states.contains(Key(handle)) && std::none_of(output.seeds.begin(), output.seeds.end(),
            [&](const auto& seed) { return Key(seed.resource) == Key(handle); }))
            output.seeds.push_back(*ordered[step.resource]);
        Visit(grid.shape, step.range, [&](uint32_t mip, uint32_t slice) {
            const auto cellIndex = CellIndex(grid.shape, mip, slice);
            auto& before = grid.cells[cellIndex];
            auto& previousQueue = grid.queues[cellIndex];
            const auto consumerQueue = graph.batches[step.batch].queue;
            const bool submittedQueueHandoff = previousQueue != UINT32_MAX
                && previousQueue != consumerQueue;
            if (grid.shape.hasLayout) {
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
                const auto heapType = ordered[step.resource]->heapType;
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
            before = step.after;
            previousQueue = consumerQueue;
        });
        output.committedResources.push_back(handle);
        output.committedStates.push_back({step.range, step.after});
        output.committedQueues.push_back(graph.batches[step.batch].queue);
    }
    BT_PLOT("ORG.AsyncExecution.StateBarrierSteps", static_cast<int64_t>(entrySteps));
    BT_PLOT("ORG.AsyncExecution.IntraBatchStateBarrierSteps", static_cast<int64_t>(intraBatchSteps));
    BT_PLOT("ORG.AsyncExecution.CrossQueueStateBarrierSteps", static_cast<int64_t>(crossQueueSteps));
    if (intraBatchSteps)
        basic_telemetry::AddCounter("ORG.AsyncExecution.IntraBatchStateBarrierSteps",
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
        for (const auto& region : seed.regions)
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

std::vector<const PreparedBackingState*> OrderBackings(
    const CompiledGraph& graph, std::span<const PreparedBackingState> resources) {
    if (!graph.structure || resources.size() != graph.structure->resourceIDs.size())
        throw std::invalid_argument("Incomplete backing access input");
    std::vector<const PreparedBackingState*> ordered(resources.size());
    for (size_t i = 0; i < ordered.size(); ++i) {
        const std::string_view expectedKey = graph.structure->resourceKeys.empty()
            ? std::string_view{} : std::string_view{graph.structure->resourceKeys[i]};
        for (const auto& resource : resources) {
            if (resource.graphResourceID == graph.structure->resourceIDs[i]
                && resource.graphResourceKey == expectedKey) {
                if (ordered[i]) throw std::invalid_argument("Duplicate backing access identity");
                ordered[i] = &resource;
            }
        }
        if (!ordered[i]) throw std::invalid_argument("Missing backing access identity");
    }
    return ordered;
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
    BT_ZONE_SCOPE("ORG.AsyncExecution.ResolveCrossFrameHazards");
    if (incoming.size() != graph.batches.size() || queues.size() != graph.structure->queues.size())
        throw std::invalid_argument("Invalid cross-frame hazard dimensions");
    const auto ordered = OrderBackings(graph, resources);
    uint64_t waitsAdded = 0;
    for (uint32_t batchIndex = 0; batchIndex < graph.batches.size(); ++batchIndex) {
        const auto& batch = graph.batches[batchIndex];
        const auto consumerTimeline = queues[batch.queue].timeline;
        const auto before = incoming[batchIndex].size();
        for (const auto pass : batch.passes) {
            VisitPassAccesses(graph, pass, [&](uint32_t resourceIndex, CompileRange range, bool write) {
                const auto key = Key(ordered.at(resourceIndex)->resource);
                const auto found = m_accesses.find(key);
                if (found == m_accesses.end()) return;
                if (found->second.shape != ordered[resourceIndex]->shape)
                    throw std::invalid_argument("Backing access shape changed without identity change");
                Visit(found->second.shape, range, [&](uint32_t mip, uint32_t slice) {
                    const auto& cell = found->second.cells[CellIndex(found->second.shape, mip, slice)];
                    AppendWait(incoming[batchIndex], cell.writer, consumerTimeline);
                    if (write) for (const auto reader : cell.readers)
                        AppendWait(incoming[batchIndex], reader, consumerTimeline);
                });
            });
        }
        waitsAdded += incoming[batchIndex].size() - before;
    }
    BT_PLOT("ORG.AsyncExecution.CrossFrameHazardWaits", static_cast<int64_t>(waitsAdded));
    if (waitsAdded) basic_telemetry::AddCounter(
        "ORG.AsyncExecution.CrossFrameHazardWaits", static_cast<int64_t>(waitsAdded));
}

void BackingAccessAdmissionLedger::Commit(const CompiledGraph& graph,
    std::span<const PreparedBackingState> resources,
    const GraphExecutionTimeline& execution, uint32_t batchCount) {
    BT_ZONE_SCOPE("ORG.AsyncExecution.CommitCrossFrameHazards");
    if (execution.batches.size() != graph.batches.size())
        throw std::invalid_argument("Invalid submitted hazard dimensions");
    const auto ordered = OrderBackings(graph, resources);
    batchCount = (std::min)(batchCount, static_cast<uint32_t>(graph.batches.size()));
    for (uint32_t batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
        const auto signal = execution.batches[batchIndex].signal;
        if (!signal.timeline || !signal.value) throw std::invalid_argument("Unsubmitted hazard point");
        for (const auto pass : graph.batches[batchIndex].passes) {
            VisitPassAccesses(graph, pass, [&](uint32_t resourceIndex, CompileRange range, bool write) {
                const auto& backing = *ordered.at(resourceIndex);
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
    const CompiledGraph& graph, std::vector<PreparedBackingState>& resources) const {
    BT_ZONE_SCOPE("ORG.AsyncExecution.ResolveAliasActivations");
    const auto ordered = OrderBackings(graph, resources);
    std::vector<rhi::ResourceHandle> activated;
    for (const auto* resource : ordered) {
        if (!resource->aliasHeapIdentity || !resource->aliasSize) continue;
        const auto found = m_intervals.find(resource->aliasHeapIdentity);
        if (found == m_intervals.end()) continue;
        const uint64_t end = resource->aliasOffset + resource->aliasSize;
        const bool replacesOccupant = std::ranges::any_of(found->second, [&](const auto& prior) {
            return !SameHandle(prior.occupant, resource->resource)
                && Overlaps(resource->aliasOffset, end, prior.begin, prior.end);
        });
        if (!replacesOccupant) continue;
        auto mutableResource = std::find_if(resources.begin(), resources.end(), [&](const auto& candidate) {
            return SameHandle(candidate.resource, resource->resource);
        });
        if (mutableResource == resources.end()) throw std::logic_error("Missing alias activation backing");
        mutableResource->regions.assign(1, PreparedStateRegion{
            {0, mutableResource->shape.mips, 0, mutableResource->shape.slices},
            {static_cast<uint64_t>(rhi::ResourceAccessType::None),
                static_cast<uint64_t>(mutableResource->shape.hasLayout
                    ? rhi::ResourceLayout::Undefined : rhi::ResourceLayout::Common),
                static_cast<uint64_t>(rhi::ResourceSyncState::None), false}});
        activated.push_back(resource->resource);
    }
    BT_PLOT("ORG.AsyncExecution.AliasActivations", static_cast<int64_t>(activated.size()));
    if (!activated.empty()) basic_telemetry::AddCounter(
        "ORG.AsyncExecution.AliasActivations", static_cast<int64_t>(activated.size()));
    return activated;
}

void AliasAccessAdmissionLedger::AppendIncomingWaits(const CompiledGraph& graph,
    std::span<const PreparedBackingState> resources,
    std::span<const ExecutionTimelinePoint> queues,
    std::vector<std::vector<ExecutionTimelinePoint>>& incoming) const {
    BT_ZONE_SCOPE("ORG.AsyncExecution.ResolveAliasHazards");
    if (incoming.size() != graph.batches.size() || queues.size() != graph.structure->queues.size())
        throw std::invalid_argument("Invalid alias hazard dimensions");
    const auto ordered = OrderBackings(graph, resources);
    uint64_t waitsAdded = 0, overlapsFound = 0;
    for (uint32_t batchIndex = 0; batchIndex < graph.batches.size(); ++batchIndex) {
        const auto& batch = graph.batches[batchIndex];
        const auto consumerTimeline = queues[batch.queue].timeline;
        const auto before = incoming[batchIndex].size();
        for (const auto pass : batch.passes) {
            VisitPassAccesses(graph, pass, [&](uint32_t resourceIndex, CompileRange, bool) {
                const auto& resource = *ordered.at(resourceIndex);
                if (!resource.aliasHeapIdentity || !resource.aliasSize) return;
                const auto found = m_intervals.find(resource.aliasHeapIdentity);
                if (found == m_intervals.end()) return;
                const uint64_t end = resource.aliasOffset + resource.aliasSize;
                for (const auto& prior : found->second) {
                    if (SameHandle(prior.occupant, resource.resource)
                        || !Overlaps(resource.aliasOffset, end, prior.begin, prior.end)) continue;
                    ++overlapsFound;
                    for (const auto point : prior.accesses)
                        AppendWait(incoming[batchIndex], point, consumerTimeline);
                }
            });
        }
        waitsAdded += incoming[batchIndex].size() - before;
    }
    BT_PLOT("ORG.AsyncExecution.AliasHazardWaits", static_cast<int64_t>(waitsAdded));
    BT_PLOT("ORG.AsyncExecution.AliasOverlaps", static_cast<int64_t>(overlapsFound));
    if (waitsAdded) basic_telemetry::AddCounter(
        "ORG.AsyncExecution.AliasHazardWaits", static_cast<int64_t>(waitsAdded));
}

void AliasAccessAdmissionLedger::Commit(const CompiledGraph& graph,
    std::span<const PreparedBackingState> resources,
    const GraphExecutionTimeline& execution, uint32_t batchCount) {
    BT_ZONE_SCOPE("ORG.AsyncExecution.CommitAliasHazards");
    if (execution.batches.size() != graph.batches.size())
        throw std::invalid_argument("Invalid submitted alias dimensions");
    const auto ordered = OrderBackings(graph, resources);
    batchCount = (std::min)(batchCount, static_cast<uint32_t>(graph.batches.size()));
    for (uint32_t batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
        const auto signal = execution.batches[batchIndex].signal;
        if (!signal.timeline || !signal.value) throw std::invalid_argument("Unsubmitted alias point");
        for (const auto pass : graph.batches[batchIndex].passes) {
            VisitPassAccesses(graph, pass, [&](uint32_t resourceIndex, CompileRange, bool) {
                const auto& resource = *ordered.at(resourceIndex);
                if (!resource.aliasHeapIdentity || !resource.aliasSize) return;
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
                    return;
                }
                auto point = std::find_if(found->accesses.begin(), found->accesses.end(),
                    [&](const auto& existing) { return existing.timeline == signal.timeline; });
                if (point == found->accesses.end()) found->accesses.push_back(signal);
                else if (point->value < signal.value) *point = signal;
            });
        }
    }
}

} // namespace org::experimental
