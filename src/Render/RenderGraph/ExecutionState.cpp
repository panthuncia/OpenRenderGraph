#include "Render/RenderGraph/ExecutionState.h"
#include "Render/BufferBarrierHelpers.h"
#include "OpenRenderGraph/Trace.h"
#include <rhi_helpers.h>

#include <algorithm>
#include <unordered_map>
#include <stdexcept>
#include <BasicTelemetry/Tracy.h>
#include <BasicTelemetry/Telemetry.h>

namespace {
// A texture barrier between identical states on one queue orders nothing: no layout change, the same access and
// scope, and no ownership transfer. Unordered access still orders writes (NeedsWholeBufferBarrier).
bool NeedsTextureBarrier(const rhi::TextureBarrier& barrier) noexcept {
    return barrier.queueOwnership != rhi::QueueOwnership::None || barrier.beforeLayout != barrier.afterLayout
        || org::NeedsWholeBufferBarrier(barrier.beforeAccess, barrier.afterAccess, barrier.beforeSync, barrier.afterSync);
}

// Schedule queue slots map to QueueKind the same way submission does
// (RenderGraph.cpp: 0 graphics, 1 compute, otherwise copy).
rhi::QueueKind OwnershipPeerForQueue(uint32_t queue) noexcept {
    return queue == 0 ? rhi::QueueKind::Graphics
        : queue == 1 ? rhi::QueueKind::Compute : rhi::QueueKind::Copy;
}
}

namespace org::compiler {

namespace {
// Decision-trace labels (OpenRenderGraph/Trace.h).
std::string TracePassName(const Schedule& graph, uint32_t pass) {
    if (pass == UINT32_MAX) return "<entry>";
    const auto& names = graph.structure->diagnosticPassNames;
    if (pass < names.size() && !names[pass].empty()) return names[pass];
    return "pass#" + std::to_string(pass);
}
std::string TraceResourceName(const PreparedBackingState& backing) {
    if (!backing.debugName.empty()) return backing.debugName;
    if (!backing.graphResourceKey.empty()) return backing.graphResourceKey;
    return "resource#" + std::to_string(backing.graphResourceID);
}
const char* TraceLayout(uint64_t layout) {
    return rhi::helpers::ResourceLayoutToString(static_cast<rhi::ResourceLayout>(layout));
}
const char* TraceLayout(rhi::ResourceLayout layout) { return rhi::helpers::ResourceLayoutToString(layout); }
const char* TraceOwnership(rhi::QueueOwnership ownership) {
    switch (ownership) {
    case rhi::QueueOwnership::Acquire: return "acquire";
    case rhi::QueueOwnership::Release: return "release";
    default: return "none";
    }
}
uint64_t TraceHandle(rhi::ResourceHandle handle) { return (uint64_t{handle.generation} << 32) | handle.index; }
}

namespace {
bool PointCompleted(ExecutionTimelinePoint point, const std::map<uint64_t, uint64_t>& completed) {
    if (!point.timeline && !point.value) return true;
    const auto found = completed.find(point.timeline);
    return found != completed.end() && found->second >= point.value;
}
}
bool BackingAccessAdmissionLedger::RetireCompleted(rhi::ResourceHandle resource,
    const std::map<uint64_t, uint64_t>& completed) {
    const auto key = (uint64_t{resource.generation} << 32) | resource.index;
    const auto* grid = m_accesses.Find(key);
    if (!grid) return true;
    for (const auto& cell : grid->cells) {
        if (!PointCompleted(PointOf(cell.writer), completed)) return false;
        for (const auto& reader : cell.readers) if (!PointCompleted(PointOf(reader), completed)) return false;
    }
    m_accesses.Erase(key);
    return true;
}
bool AliasAccessAdmissionLedger::RetireCompleted(const AliasHeapGeneration* heap,
    const std::weak_ptr<const AliasHeapGeneration>& owner, const std::map<uint64_t, uint64_t>& completed) {
    const auto found = m_heaps.find(heap);
    if (found == m_heaps.end()) return true;
    const auto recorded = m_heapOwners.find(heap);
    if (recorded == m_heapOwners.end() || !owner.expired()
        || owner.owner_before(recorded->second) || recorded->second.owner_before(owner)) return false;
    for (const auto& occupant : found->second.occupants)
        for (auto point : occupant.accesses) if (!PointCompleted(point, completed)) return false;
    m_heaps.erase(found);
    m_heapOwners.erase(heap);
    return true;
}
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

template<class Grid>
uint64_t Fingerprint(const Grid& grid) {
    uint64_t hash = 1469598103934665603ull;
    const auto mix = [&](uint64_t value) { hash = (hash ^ value) * 1099511628211ull; };
    for (const auto& cell : grid.cells) { mix(cell.access); mix(cell.layout); mix(cell.sync); mix(cell.write); }
    for (const auto queue : grid.queues) mix(queue);
    return hash;
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

void ValidatePreparedBackingOrder(const GraphCompileStructure& structure, std::span<const PreparedBackingState> backings) {
    (void)ValidatePreparedBackings(structure, backings);
}

PreparedExecutionBarrierPlan BackingStateAdmissionLedger::Prepare(
    const Schedule& graph, std::span<const PreparedBackingState> initial,
    std::span<const rhi::ResourceHandle> invalidated, bool closeToHome, bool entryOnly, bool validated, uint64_t backingSet,
    bool externalHome) const {
    BT_ZONE_SCOPE("ORG.Execution.ResolveBackingStates");
    if (!graph.structure || !graph.states.complete || initial.size() != graph.structure->resourceIDs.size())
        throw std::invalid_argument("Incomplete symbolic state admission input");
    const auto ordered = validated ? initial : ValidatePreparedBackings(*graph.structure, initial);
    // Each resource's grid, by its compiler index: found once for the backings' set.
    auto& resolved = m_states.Resolve(initial, backingSet);
    const auto gridOf = [&](uint32_t resource) -> const StateGrid* {
        if (resource >= resolved.size()) return nullptr;
        if (!resolved[resource] && ordered[resource].resource.valid()) resolved[resource] = const_cast<StateGrid*>(m_states.Find(Key(ordered[resource].resource)));
        return resolved[resource];
    };
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
        // The body's barriers: none in an entry-only plan (an aliased backing's entry, which goes there, is the body's too).
        for (uint32_t position = 0; !entryOnly && position < graph.batches[batch].passes.size(); ++position) {
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
    // Every backing against what the ledger holds: an entry-only plan's backings were checked by the plan its body came from.
    for (size_t resource = 0; !entryOnly && resource < initial.size(); ++resource) {
        const auto& captured = ordered[resource];
        if (!captured.resource.valid()) continue; // Reserved slot: no backing this frame.
        if (captured.authoritativeIncoming) {
            if (!captured.regions || captured.regions->empty())
                throw std::invalid_argument("Authoritative incoming state has no complete regions");
            const uint64_t area = uint64_t{captured.shape.mips} * captured.shape.slices;
            uint64_t covered = 0;
            for (size_t i = 0; i < captured.regions->size(); ++i) {
                const auto& range = captured.regions->at(i).range;
                if (!range.mips || !range.slices || range.mip >= captured.shape.mips || range.slice >= captured.shape.slices
                    || range.mips > captured.shape.mips-range.mip || range.slices > captured.shape.slices-range.slice)
                    throw std::invalid_argument("Authoritative incoming state range violates backing shape");
                for (size_t j = 0; j < i; ++j) {
                    const auto& previous = captured.regions->at(j).range;
                    if (range.mip < previous.mip+previous.mips && previous.mip < range.mip+range.mips
                        && range.slice < previous.slice+previous.slices && previous.slice < range.slice+range.slices)
                        throw std::invalid_argument("Authoritative incoming state regions overlap");
                }
                covered += uint64_t{range.mips} * range.slices;
            }
            if (covered != area) throw std::invalid_argument("Authoritative incoming state regions do not cover backing");
        }
        const auto* existing = gridOf(static_cast<uint32_t>(resource));
        if (existing && !wasInvalidated(captured.resource)
            && existing->shape != captured.shape)
            throw std::invalid_argument("Backing shape changed without a new backing identity");
    }
    uint64_t entrySteps = 0, intraBatchSteps = 0, crossQueueSteps = 0;
    std::vector<rhi::ResourceHandle> authoritativeSeeded;
    // Multi-device resources: a queue's barriers name its device's representation, and a handoff between devices is an
    // external ownership transfer rather than a queue-family one.
    const auto deviceOfQueue = [&](uint32_t queue) -> uint32_t {
        return queue < graph.structure->queues.size() ? graph.structure->queues[queue].backendInstance : 0u;
    };
    const auto handleOn = [&](const PreparedBackingState& captured, uint32_t queue) {
        const auto device = deviceOfQueue(queue);
        return device < captured.deviceHandles.size() && captured.deviceHandles[device].valid()
            ? captured.deviceHandles[device] : captured.resource;
    };
    const auto toExternal = [](auto& barrier, rhi::QueueOwnership direction) {
        using External = std::remove_cvref_t<decltype(barrier.externalOwnership)>;
        barrier.queueOwnership = rhi::QueueOwnership::None;
        barrier.externalOwnership = direction == rhi::QueueOwnership::Release ? External::Release : External::Acquire;
    };
    // A texture an earlier execution left on another queue of this graph's: released there (result.releases) from the state it left
    // in to Common, and acquired here from Common. Not for a closed execution (its textures stay on one queue), across devices
    // (the external ownership handoff covers it) or from a queue this graph does not have. The ledger names queues by slot, the same
    // native queue in every graph on the runtime.
    const auto handOff = [&](rhi::TextureBarrier& barrier, const PreparedBackingState& captured, uint32_t previousQueue,
        uint32_t consumerQueue, uint32_t batch) {
        const char* skipped = closeToHome ? "closed-execution"
            : barrier.discard ? "discarded"
            : previousQueue == UINT32_MAX ? "previous-queue-unknown"
            : previousQueue == consumerQueue ? "same-queue"
            : previousQueue >= graph.structure->queues.size() ? "previous-queue-not-in-graph"
            : deviceOfQueue(previousQueue) != deviceOfQueue(consumerQueue) ? "other-device" : nullptr;
        if (skipped) {
            ORG_TRACE(Handoff, "handoff-skipped").Resource(TraceResourceName(captured)).Add("reason", skipped)
                .Add("previousQueue", previousQueue).Add("consumerQueue", consumerQueue).Add("batch", batch)
                .Add("beforeLayout", TraceLayout(barrier.beforeLayout)).Add("afterLayout", TraceLayout(barrier.afterLayout));
            return false;
        }
        ORG_TRACE(Handoff, "handoff-planned").Resource(TraceResourceName(captured))
            .Add("fromQueue", previousQueue).Add("toQueue", consumerQueue).Add("consumerBatch", batch)
            .Add("releaseFromLayout", TraceLayout(barrier.beforeLayout)).Add("afterLayout", TraceLayout(barrier.afterLayout));
        auto release = barrier;
        release.texture = handleOn(captured, previousQueue);
        release.afterAccess = rhi::ResourceAccessType::Common;
        release.afterLayout = rhi::ResourceLayout::Common;
        release.afterSync = rhi::ResourceSyncState::All;
        release.queueOwnership = rhi::QueueOwnership::Release;
        release.ownershipPeer = OwnershipPeerForQueue(consumerQueue);
        // The acquire repeats the release (Vulkan pairs them exactly), in the entry batch; the caller's barrier then takes the
        // texture on from Common, in the batch after it (PreparedBatchBarriers::acquired).
        auto acquire = release;
        acquire.texture = barrier.texture;
        acquire.queueOwnership = rhi::QueueOwnership::Acquire;
        acquire.ownershipPeer = OwnershipPeerForQueue(previousQueue);
        result.batches[batch].textures.push_back(acquire);
        barrier.beforeAccess = rhi::ResourceAccessType::Common;
        barrier.beforeLayout = rhi::ResourceLayout::Common;
        barrier.beforeSync = rhi::ResourceSyncState::All;
        auto found = std::ranges::find(result.releases, previousQueue, &PreparedExecutionBarrierPlan::Release::queue);
        if (found == result.releases.end()) {
            result.releases.push_back({});
            result.releases.back().queue = previousQueue;
            found = result.releases.end() - 1;
        }
        found->textures.push_back(release);
        if (std::ranges::find(found->consumers, batch) == found->consumers.end()) found->consumers.push_back(batch);
        return true;
    };
    const size_t stepCount = entryOnly ? graph.states.entrySteps.size() : graph.states.steps.size();
    for (size_t stepIndex = 0; stepIndex < stepCount; ++stepIndex) {
        const auto& step = graph.states.steps[entryOnly ? graph.states.entrySteps[stepIndex] : stepIndex];
        if (step.resource >= ordered.size() || step.batch >= result.batches.size())
            throw std::invalid_argument("Invalid symbolic state step");
        const auto& captured = ordered[step.resource];
        if (step.pass >= graph.batchByPass.size() || graph.batchByPass[step.pass] != step.batch)
            throw std::invalid_argument("Symbolic state step has no consuming pass placement");
        if (!captured.resource.valid()) continue; // Reserved slot: no barriers, seeds or commits.
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
        const auto consumerHandle = handleOn(captured, graph.batches[step.batch].queue);
        const bool multiDevice = !captured.deviceHandles.empty();
        const bool crossDevice = crossQueue
            && deviceOfQueue(graph.batches[step.previousBatch].queue) != deviceOfQueue(graph.batches[step.batch].queue);
        const auto producerHandle = crossQueue ? handleOn(captured, graph.batches[step.previousBatch].queue) : handle;
        const bool boundaryStep = step.previousBatch == UINT32_MAX;
        const bool authoritativeState = boundaryStep && captured.authoritativeIncoming;
        // An aliased (transient) backing keeps no contents across executions, and another occupant may have overwritten its
        // memory since its last: it enters every execution from Undefined, which its body barrier (before its first pass) encodes
        // without the ledger. Every other backing enters from the state the ledger hands off, in its batch's entry barriers: the
        // body's own barriers then depend on the compiled graph alone (docs/job-graphs.md, "Phase 2 investigation").
        const bool aliasedBacking = captured.aliasSize != 0;
        const bool invalidatedState = boundaryStep && (aliasedBacking || wasInvalidated(handle)) && !authoritativeState;
        auto& boundaryTextures = aliasedBacking ? beforePass.textures : output.textures;
        auto& boundaryBuffers = aliasedBacking ? beforePass.buffers : output.buffers;
        if (boundaryStep
            && (!gridOf(step.resource) || invalidatedState || authoritativeState)
            && std::none_of(output.seeds.begin(), output.seeds.end(),
                [&](const auto& seed) { return Key(seed.resource) == Key(handle); }))
        {
            const bool alreadySeeded = authoritativeState && std::ranges::any_of(authoritativeSeeded,
                [&](auto resource) { return SameHandle(resource,handle); });
            if (!alreadySeeded) {
                output.seeds.push_back(captured);
                if (authoritativeState) {
                    authoritativeSeeded.push_back(handle);
                    output.authoritativeSeeds.push_back(handle);
                }
            }
        }
        const StateGrid* existing = boundaryStep ? gridOf(step.resource) : nullptr;
        const bool useSubmittedState = boundaryStep
            && existing && !invalidatedState && !authoritativeState;
        // A host-owned texture under external home states that enters from its seeded (home) state is on the graphics queue.
        const bool homeSeeded = boundaryStep && externalHome && !closeToHome && captured.hostOwned && captured.shape.hasLayout
            && !multiDevice && !useSubmittedState && !invalidatedState;
        if (trace::Enabled(trace::Category::Barriers)) {
            const auto consumerQueue = graph.batches[step.batch].queue;
            const char* path = !boundaryStep ? (crossQueue ? "body-cross-queue" : step.previousBatch == step.batch
                    ? "body-same-batch" : "body-same-queue")
                : authoritativeState ? "entry-authoritative"
                : invalidatedState ? (aliasedBacking ? "entry-aliased" : "entry-invalidated")
                : useSubmittedState ? "entry-ledger" : "entry-seeded";
            trace::Builder event(trace::Category::Barriers, "state-step");
            event.Resource(TraceResourceName(captured)).Add("handle", TraceHandle(handle)).Add("path", path)
                .Add("pass", TracePassName(graph, step.pass)).Add("batch", step.batch).Add("queue", consumerQueue)
                .Add("previousPass", TracePassName(graph, step.previousPass))
                .Add("previousBatch", step.previousBatch)
                .Add("previousQueue", step.previousBatch == UINT32_MAX ? UINT32_MAX : graph.batches[step.previousBatch].queue)
                .Add("mips", std::to_string(step.range.mip) + "+" + std::to_string(step.range.mips))
                .Add("slices", std::to_string(step.range.slice) + "+" + std::to_string(step.range.slices))
                .Add("compiledBeforeLayout", TraceLayout(step.before.layout)).Add("afterLayout", TraceLayout(step.after.layout))
                .Add("aliased", aliasedBacking).Add("closeToHome", closeToHome);
            if (useSubmittedState && !existing->cells.empty()) {
                const auto cell = CellIndex(existing->shape, step.range.mip, step.range.slice);
                event.Add("ledgerLayout", TraceLayout(existing->cells[cell].layout))
                    .Add("ledgerQueue", existing->queues[cell]);
            }
        }
        // Intra-frame state is compiler-invariant. Encode its complete range
        // once instead of replaying one identical barrier per mip/slice cell.
        if (!boundaryStep) {
            const auto consumerQueue = graph.batches[step.batch].queue;
            const auto previousQueue = graph.batches[step.previousBatch].queue;
            if (captured.shape.hasLayout) {
                rhi::TextureBarrier barrier{};
                barrier.texture = consumerHandle;
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
                    release.texture = producerHandle;
                    release.beforeAccess = static_cast<rhi::ResourceAccessType>(step.before.access);
                    release.afterAccess = rhi::ResourceAccessType::Common;
                    release.beforeLayout = static_cast<rhi::ResourceLayout>(step.before.layout);
                    release.afterLayout = rhi::ResourceLayout::Common;
                    release.beforeSync = static_cast<rhi::ResourceSyncState>(step.before.sync);
                    release.afterSync = rhi::ResourceSyncState::All;
                    release.discard = release.beforeLayout == rhi::ResourceLayout::Undefined;
                    release.queueOwnership = rhi::QueueOwnership::Release;
                    release.ownershipPeer = OwnershipPeerForQueue(consumerQueue);
                    barrier.queueOwnership = rhi::QueueOwnership::Acquire;
                    barrier.ownershipPeer = OwnershipPeerForQueue(previousQueue);
                    if (crossDevice) { toExternal(release, rhi::QueueOwnership::Release); toExternal(barrier, rhi::QueueOwnership::Acquire); }
                    afterProducer->textures.push_back(release);
                    // Vulkan pairs the halves of a queue-family transfer exactly: the acquire repeats the release (to Common), at the
                    // batch's entry, and the pass's own barrier takes the texture on from Common. A backend whose transfer collapses
                    // (D3D12; one family; a concurrent resource) drops the acquire. An aliased texture keeps the acquire in its pass:
                    // another occupant may use its memory earlier in the batch.
                    if (!crossDevice && !captured.aliasSize) {
                        auto acquire = release;
                        acquire.texture = consumerHandle;
                        acquire.queueOwnership = rhi::QueueOwnership::Acquire;
                        acquire.ownershipPeer = OwnershipPeerForQueue(previousQueue);
                        output.textures.push_back(acquire);
                        barrier.queueOwnership = rhi::QueueOwnership::None;
                    }
                }
                if (NeedsTextureBarrier(barrier)) beforePass.textures.push_back(barrier);
            } else {
                const bool fixedHeapState = captured.heapType == rhi::HeapType::Upload
                    || captured.heapType == rhi::HeapType::Readback;
                const auto beforeAccess = static_cast<rhi::ResourceAccessType>(step.before.access);
                const auto beforeSync = static_cast<rhi::ResourceSyncState>(step.before.sync);
                const auto afterAccess = static_cast<rhi::ResourceAccessType>(step.after.access);
                const auto afterSync = static_cast<rhi::ResourceSyncState>(step.after.sync);
                if (!fixedHeapState && crossQueue) {
                    // Queue-family ownership handoff for exclusive buffers. The RHI
                    // drops the pair on D3D12 and on Vulkan when families match or the
                    // buffer is QueueSharing::Concurrent.
                    auto release = MakeWholeBufferBarrier(producerHandle, beforeAccess,
                        rhi::ResourceAccessType::Common, beforeSync, rhi::ResourceSyncState::All);
                    release.queueOwnership = rhi::QueueOwnership::Release;
                    release.ownershipPeer = OwnershipPeerForQueue(consumerQueue);
                    auto acquire = MakeWholeBufferBarrier(consumerHandle, rhi::ResourceAccessType::Common,
                        afterAccess, rhi::ResourceSyncState::All, afterSync);
                    acquire.queueOwnership = rhi::QueueOwnership::Acquire;
                    acquire.ownershipPeer = OwnershipPeerForQueue(previousQueue);
                    if (crossDevice) { toExternal(release, rhi::QueueOwnership::Release); toExternal(acquire, rhi::QueueOwnership::Acquire); }
                    afterProducer->buffers.push_back(release);
                    beforePass.buffers.push_back(acquire);
                }
                if (!fixedHeapState && previousQueue == consumerQueue
                    && beforeAccess != rhi::ResourceAccessType::None
                    && beforeAccess != rhi::ResourceAccessType::Common
                    && afterAccess != rhi::ResourceAccessType::None
                    && afterAccess != rhi::ResourceAccessType::Common
                    && NeedsWholeBufferBarrier(beforeAccess, afterAccess, beforeSync, afterSync))
                    beforePass.buffers.push_back(MakeWholeBufferBarrier(consumerHandle,
                        beforeAccess, afterAccess, beforeSync, afterSync));
            }
            output.committedResources.push_back(handle);
            output.committedIndices.push_back(step.resource);
            output.committedStates.push_back({step.range, step.after});
            output.committedQueues.push_back(consumerQueue);
            continue;
        }
        // A multi-device resource enters every execution released to COMMON (see the end of Prepare): its first use acquires it.
        const bool externalAcquire = multiDevice && useSubmittedState;
        std::optional<CompileResourceState> uniformBefore;
        uint32_t uniformPreviousQueue = UINT32_MAX;
        if (useSubmittedState) {
            bool uniform = true;
            Visit(captured.shape, step.range, [&](uint32_t mip, uint32_t slice) {
                const auto cell = CellIndex(captured.shape, mip, slice);
                if (!uniformBefore) {
                    uniformBefore = existing->cells[cell];
                    uniformPreviousQueue = existing->queues[cell];
                } else if (*uniformBefore != existing->cells[cell]
                    || uniformPreviousQueue != existing->queues[cell]) uniform = false;
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
                barrier.texture = consumerHandle;
                barrier.range = {step.range.mip, step.range.mips,
                    step.range.slice, step.range.slices};
                if (externalAcquire) toExternal(barrier, rhi::QueueOwnership::Acquire);
                barrier.beforeAccess = static_cast<rhi::ResourceAccessType>(uniformBefore->access);
                barrier.afterAccess = static_cast<rhi::ResourceAccessType>(step.after.access);
                barrier.beforeLayout = static_cast<rhi::ResourceLayout>(uniformBefore->layout);
                barrier.afterLayout = static_cast<rhi::ResourceLayout>(step.after.layout);
                barrier.beforeSync = static_cast<rhi::ResourceSyncState>(uniformBefore->sync);
                barrier.afterSync = static_cast<rhi::ResourceSyncState>(step.after.sync);
                barrier.discard = barrier.beforeLayout == rhi::ResourceLayout::Undefined;
                const bool handedOff = !externalAcquire && (useSubmittedState || homeSeeded)
                    && handOff(barrier, captured, homeSeeded ? 0u : uniformPreviousQueue, consumerQueue, step.batch);
                if (NeedsTextureBarrier(barrier)) (handedOff ? output.acquired : boundaryTextures).push_back(barrier);
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
                if (externalAcquire) {
                    auto acquire = MakeWholeBufferBarrier(consumerHandle, rhi::ResourceAccessType::Common,
                        afterAccess, rhi::ResourceSyncState::All, afterSync);
                    toExternal(acquire, rhi::QueueOwnership::Acquire);
                    boundaryBuffers.push_back(acquire);
                } else if (!fixedHeapState && copyCompatible && !queueHandoff
                    && beforeAccess != rhi::ResourceAccessType::None
                    && beforeAccess != rhi::ResourceAccessType::Common
                    && afterAccess != rhi::ResourceAccessType::None
                    && afterAccess != rhi::ResourceAccessType::Common
                    && NeedsWholeBufferBarrier(beforeAccess, afterAccess, beforeSync, afterSync))
                    boundaryBuffers.push_back(MakeWholeBufferBarrier(consumerHandle,
                        beforeAccess, afterAccess, beforeSync, afterSync));
            }
            output.committedResources.push_back(handle);
            output.committedIndices.push_back(step.resource);
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
                before = existing->cells[cellIndex];
                previousQueue = existing->queues[cellIndex];
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
                if (homeSeeded) previousQueue = 0;
            }
            const auto consumerQueue = graph.batches[step.batch].queue;
            const bool submittedQueueHandoff = previousQueue != UINT32_MAX
                && previousQueue != consumerQueue;
            if (captured.shape.hasLayout) {
                bool handedOff = false;
                rhi::TextureBarrier barrier{};
                barrier.texture = consumerHandle;
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
                    release.texture = producerHandle;
                    release.beforeAccess = static_cast<rhi::ResourceAccessType>(before.access);
                    release.afterAccess = rhi::ResourceAccessType::Common;
                    release.beforeLayout = static_cast<rhi::ResourceLayout>(before.layout);
                    release.afterLayout = rhi::ResourceLayout::Common;
                    release.beforeSync = static_cast<rhi::ResourceSyncState>(before.sync);
                    release.afterSync = rhi::ResourceSyncState::All;
                    release.discard = release.beforeLayout == rhi::ResourceLayout::Undefined;
                    release.queueOwnership = rhi::QueueOwnership::Release;
                    release.ownershipPeer = OwnershipPeerForQueue(consumerQueue);
                    barrier.queueOwnership = rhi::QueueOwnership::Acquire;
                    barrier.ownershipPeer = OwnershipPeerForQueue(previousQueue);
                    if (crossDevice) { toExternal(release, rhi::QueueOwnership::Release); toExternal(barrier, rhi::QueueOwnership::Acquire); }
                    afterProducer->textures.push_back(release);
                } else if (externalAcquire) {
                    toExternal(barrier, rhi::QueueOwnership::Acquire);
                } else if (submittedQueueHandoff) {
                    handedOff = handOff(barrier, captured, previousQueue, consumerQueue, step.batch);
                    // An earlier execution left it on a queue whose release cannot be planned (handOff). Counted so these
                    // handoffs are visible; graph buffers are Concurrent.
                    if (!handedOff) basic_telemetry::AddCounter("ORG.Execution.CrossFrameTextureQueueHandoff");
                }
                if (NeedsTextureBarrier(barrier)) (handedOff ? output.acquired : boundaryTextures).push_back(barrier);
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
                if (!fixedHeapState && crossQueue) {
                    auto release = MakeWholeBufferBarrier(producerHandle, beforeAccess,
                        rhi::ResourceAccessType::Common, beforeSync, rhi::ResourceSyncState::All);
                    release.queueOwnership = rhi::QueueOwnership::Release;
                    release.ownershipPeer = OwnershipPeerForQueue(graph.batches[step.batch].queue);
                    auto acquire = MakeWholeBufferBarrier(consumerHandle, rhi::ResourceAccessType::Common,
                        afterAccess, rhi::ResourceSyncState::All, afterSync);
                    acquire.queueOwnership = rhi::QueueOwnership::Acquire;
                    acquire.ownershipPeer = OwnershipPeerForQueue(previousQueue);
                    if (crossDevice) { toExternal(release, rhi::QueueOwnership::Release); toExternal(acquire, rhi::QueueOwnership::Acquire); }
                    afterProducer->buffers.push_back(release);
                    boundaryBuffers.push_back(acquire);
                } else if (!fixedHeapState && externalAcquire) {
                    auto acquire = MakeWholeBufferBarrier(consumerHandle, rhi::ResourceAccessType::Common,
                        afterAccess, rhi::ResourceSyncState::All, afterSync);
                    toExternal(acquire, rhi::QueueOwnership::Acquire);
                    boundaryBuffers.push_back(acquire);
                }
                if (!fixedHeapState && !externalAcquire
                    && copyQueueCompatibleBefore
                    && !submittedQueueHandoff
                    && beforeAccess != rhi::ResourceAccessType::None
                    && beforeAccess != rhi::ResourceAccessType::Common
                    && afterAccess != rhi::ResourceAccessType::None
                    && afterAccess != rhi::ResourceAccessType::Common
                    && NeedsWholeBufferBarrier(beforeAccess, afterAccess, beforeSync, afterSync))
                    boundaryBuffers.push_back(MakeWholeBufferBarrier(consumerHandle,
                        beforeAccess, afterAccess, beforeSync, afterSync));
            }
        });
        output.committedResources.push_back(handle);
            output.committedIndices.push_back(step.resource);
        output.committedStates.push_back({step.range, step.after});
        output.committedQueues.push_back(graph.batches[step.batch].queue);
    }
    // An entry-only plan commits each cell's last state (its exit step) after the entries: the body's steps between them change
    // nothing the ledger keeps.
    if (entryOnly) for (const auto index : graph.states.exitSteps) {
        const auto& step = graph.states.steps[index];
        if (step.previousBatch == UINT32_MAX) continue;  // an entry step, committed above
        const auto& captured = ordered[step.resource];
        if (!captured.resource.valid()) continue;
        auto& output = result.batches[step.batch];
        output.committedResources.push_back(captured.resource);
        output.committedIndices.push_back(step.resource);
        output.committedStates.push_back({step.range, step.after});
        output.committedQueues.push_back(graph.batches[step.batch].queue);
    }
    {
        // A multi-device resource leaves every execution released to COMMON on the device of its last use, so the next one may
        // acquire it on any device (the timeline orders the two).
        // Per resource: the last step of each cell (a texture's mips and slices may leave in different states).
        std::unordered_map<uint32_t, std::vector<uint32_t>> lastSteps;
        for (uint32_t index = 0; index < graph.states.steps.size(); ++index) {
            const auto& step = graph.states.steps[index];
            const auto& captured = ordered[step.resource];
            if (captured.deviceHandles.empty() || !captured.resource.valid()) continue;
            auto& cells = lastSteps[step.resource];
            if (cells.empty()) cells.assign(size_t{captured.shape.mips} * captured.shape.slices, UINT32_MAX);
            Visit(captured.shape, step.range, [&](uint32_t mip, uint32_t slice) { cells[CellIndex(captured.shape, mip, slice)] = index; });
        }
        const CompileResourceState common{static_cast<uint64_t>(rhi::ResourceAccessType::Common),
            static_cast<uint64_t>(rhi::ResourceLayout::Common), static_cast<uint64_t>(rhi::ResourceSyncState::All), false};
        for (const auto& [resource, cells] : lastSteps) {
            const auto& captured = ordered[resource];
            for (uint32_t mip = 0; mip < captured.shape.mips; ++mip)
                for (uint32_t slice = 0; slice < captured.shape.slices; ++slice) {
                    const auto index = cells[CellIndex(captured.shape, mip, slice)];
                    if (index == UINT32_MAX) continue;
                    // A buffer (one cell) or each texture cell, released after its own last use.
                    const auto& last = graph.states.steps[index];
                    const auto queue = graph.batches[last.batch].queue;
                    auto& output = result.batches[last.batch];
                    auto& after = output.afterPass[graph.positionByPass[last.pass]];
                    if (captured.shape.hasLayout) {
                        rhi::TextureBarrier barrier{};
                        barrier.texture = handleOn(captured, queue);
                        barrier.range = {mip, 1, slice, 1};
                        barrier.beforeAccess = static_cast<rhi::ResourceAccessType>(last.after.access);
                        barrier.afterAccess = rhi::ResourceAccessType::Common;
                        barrier.beforeLayout = static_cast<rhi::ResourceLayout>(last.after.layout);
                        barrier.afterLayout = rhi::ResourceLayout::Common;
                        barrier.beforeSync = static_cast<rhi::ResourceSyncState>(last.after.sync);
                        barrier.afterSync = rhi::ResourceSyncState::All;
                        toExternal(barrier, rhi::QueueOwnership::Release);
                        after.textures.push_back(barrier);
                    } else {
                        auto barrier = MakeWholeBufferBarrier(handleOn(captured, queue),
                            static_cast<rhi::ResourceAccessType>(last.after.access), rhi::ResourceAccessType::Common,
                            static_cast<rhi::ResourceSyncState>(last.after.sync), rhi::ResourceSyncState::All);
                        toExternal(barrier, rhi::QueueOwnership::Release);
                        after.buffers.push_back(barrier);
                    }
                    output.committedResources.push_back(captured.resource);
                    output.committedIndices.push_back(resource);
                    output.committedStates.push_back({CompileRange{mip, 1, slice, 1}, common});
                    output.committedQueues.push_back(queue);
                }
        }
    }
    // A host-owned texture under external home states goes home as a closed execution's resources do, on the graphics queue.
    const auto goesHomeExternally = [&](const PreparedBackingState& captured) {
        return !closeToHome && externalHome && captured.hostOwned && captured.shape.hasLayout && captured.deviceHandles.empty();
    };
    if (closeToHome || externalHome) {
        // Per resource and cell: the first step (its entry state is home when the backing seeds nothing
        // for the cell) and the last (where the resource leaves the execution).
        //
        // Closed: a resource without a layout (a buffer) may be used from several queues: the execution's relative waits order
        // its uses inside it, and the host orders whole executions against each other on every queue
        // (ExternalQueueBoundary). It goes home on the queue of its last use. A texture's home layout would need a
        // queue-family ownership transfer back, so a texture stays on one queue.
        // External home: a host-owned texture may be used from any queue. A cell last used on another queue than graphics is
        // released there and acquired on graphics (homeAcquires). Its body barriers (the transition and the release) are the
        // full plan's; an entry-only plan has the commits and the acquires.
        struct Cells { std::vector<uint32_t> first, last; uint32_t queue = UINT32_MAX; };
        std::unordered_map<uint32_t, Cells> byResource;
        for (uint32_t index = 0; index < graph.states.steps.size(); ++index) {
            const auto& step = graph.states.steps[index];
            const auto& captured = ordered[step.resource];
            if (!captured.resource.valid()) continue;
            const bool external = goesHomeExternally(captured);
            if (!closeToHome && !external) continue;
            auto& cells = byResource[step.resource];
            if (cells.first.empty()) {
                const size_t count = size_t{captured.shape.mips} * captured.shape.slices;
                cells.first.assign(count, UINT32_MAX);
                cells.last.assign(count, UINT32_MAX);
            }
            const auto queue = graph.batches[step.batch].queue;
            if (!external && cells.queue != UINT32_MAX && cells.queue != queue && captured.shape.hasLayout)
                throw std::invalid_argument("A closed execution uses a texture from more than one queue");
            cells.queue = external ? 0u : queue;
            Visit(captured.shape, step.range, [&](uint32_t mip, uint32_t slice) {
                const auto cell = CellIndex(captured.shape, mip, slice);
                if (cells.first[cell] == UINT32_MAX) cells.first[cell] = index;
                cells.last[cell] = index;
            });
        }
        for (const auto& [resource, cells] : byResource) {
            const auto& captured = ordered[resource];
            const bool external = goesHomeExternally(captured);
            const bool fixedHeapState = captured.heapType == rhi::HeapType::Upload
                || captured.heapType == rhi::HeapType::Readback;
            for (uint32_t mip = 0; mip < captured.shape.mips; ++mip)
                for (uint32_t slice = 0; slice < captured.shape.slices; ++slice) {
                    const auto cell = CellIndex(captured.shape, mip, slice);
                    if (cells.last[cell] == UINT32_MAX) continue;
                    const auto& last = graph.states.steps[cells.last[cell]];
                    CompileResourceState home = graph.states.steps[cells.first[cell]].before;
                    if (captured.regions) for (const auto& region : *captured.regions)
                        if (mip >= region.range.mip && mip < region.range.mip + region.range.mips
                            && slice >= region.range.slice && slice < region.range.slice + region.range.slices) {
                            home = region.state;
                            break;
                        }
                    auto& output = result.batches[last.batch];
                    const auto lastQueue = graph.batches[last.batch].queue;
                    if (external && lastQueue != 0 && !fixedHeapState) {
                        // Released on its last use's queue to graphics, from where it was left to Common, and acquired on
                        // graphics, repeating it (Vulkan pairs them exactly; a backend whose transfer collapses drops the acquire),
                        // then taken to its home layout.
                        rhi::TextureBarrier barrier{};
                        barrier.texture = captured.resource;
                        barrier.range = {mip, 1, slice, 1};
                        barrier.beforeAccess = static_cast<rhi::ResourceAccessType>(last.after.access);
                        barrier.afterAccess = rhi::ResourceAccessType::Common;
                        barrier.beforeLayout = static_cast<rhi::ResourceLayout>(last.after.layout);
                        barrier.afterLayout = rhi::ResourceLayout::Common;
                        barrier.beforeSync = static_cast<rhi::ResourceSyncState>(last.after.sync);
                        barrier.afterSync = rhi::ResourceSyncState::All;
                        barrier.queueOwnership = rhi::QueueOwnership::Release;
                        barrier.ownershipPeer = rhi::QueueKind::Graphics;
                        if (!entryOnly) output.afterPass[graph.positionByPass[last.pass]].textures.push_back(barrier);
                        barrier.queueOwnership = rhi::QueueOwnership::Acquire;
                        barrier.ownershipPeer = OwnershipPeerForQueue(lastQueue);
                        result.homeAcquires.textures.push_back(barrier);
                        if (home.layout != static_cast<uint64_t>(rhi::ResourceLayout::Common)) {
                            rhi::TextureBarrier toHome{};
                            toHome.texture = captured.resource;
                            toHome.range = {mip, 1, slice, 1};
                            toHome.beforeAccess = rhi::ResourceAccessType::Common;
                            toHome.afterAccess = rhi::ResourceAccessType::Common;
                            toHome.beforeLayout = rhi::ResourceLayout::Common;
                            toHome.afterLayout = static_cast<rhi::ResourceLayout>(home.layout);
                            toHome.beforeSync = rhi::ResourceSyncState::All;
                            toHome.afterSync = rhi::ResourceSyncState::All;
                            result.homeAcquires.home.push_back(toHome);
                        }
                        if (std::ranges::find(result.homeAcquires.producers, last.batch) == result.homeAcquires.producers.end())
                            result.homeAcquires.producers.push_back(last.batch);
                    } else if (captured.shape.hasLayout && !fixedHeapState && home.layout != last.after.layout
                        && !(external && entryOnly)) {
                        rhi::TextureBarrier barrier{};
                        barrier.texture = captured.resource;
                        barrier.range = {mip, 1, slice, 1};
                        barrier.beforeAccess = static_cast<rhi::ResourceAccessType>(last.after.access);
                        barrier.afterAccess = rhi::ResourceAccessType::Common;
                        barrier.beforeLayout = static_cast<rhi::ResourceLayout>(last.after.layout);
                        barrier.afterLayout = static_cast<rhi::ResourceLayout>(home.layout);
                        barrier.beforeSync = static_cast<rhi::ResourceSyncState>(last.after.sync);
                        barrier.afterSync = rhi::ResourceSyncState::All;
                        output.afterPass[graph.positionByPass[last.pass]].textures.push_back(barrier);
                    }
                    output.committedResources.push_back(captured.resource);
                    output.committedIndices.push_back(resource);
                    output.committedStates.push_back({CompileRange{mip, 1, slice, 1}, home});
                    output.committedQueues.push_back(cells.queue);
                }
        }
    }
    // Within a batch only each cell's last commit counts (a receipt commits whole batches): a commit the batch's later ones cover
    // entirely is dropped. A cell left where the execution found it (a closed one's home, an entry-only plan's entry and then exit)
    // is then written once, and not at all when it is already there (CommitBatch).
    std::vector<std::vector<uint8_t>> covered(ordered.size());
    for (auto& output : result.batches) {
        const auto count = output.committedStates.size();
        if (count < 2) continue;
        std::vector<uint8_t> keep(count, 0);
        size_t kept = 0;
        for (auto& cells : covered) cells.clear();
        for (size_t i = count; i-- > 0;) {
            const auto resource = output.committedIndices[i];
            if (resource >= ordered.size()) {
                keep[i] = 1;
                ++kept;
                continue;
            }
            const auto& shape = ordered[resource].shape;
            auto& cells = covered[resource];
            if (cells.empty()) cells.assign(size_t{shape.mips} * shape.slices, 0);
            bool any = false;
            Visit(shape, output.committedStates[i].range, [&](uint32_t mip, uint32_t slice) {
                auto& cell = cells[CellIndex(shape, mip, slice)];
                any = any || !cell;
                cell = 1;
            });
            if (any) {
                keep[i] = 1;
                ++kept;
            }
        }
        if (kept == count) continue;
        size_t next = 0;
        for (size_t i = 0; i < count; ++i) {
            if (!keep[i]) continue;
            output.committedResources[next] = output.committedResources[i];
            output.committedIndices[next] = output.committedIndices[i];
            output.committedStates[next] = output.committedStates[i];
            output.committedQueues[next] = output.committedQueues[i];
            ++next;
        }
        output.committedResources.resize(next);
        output.committedIndices.resize(next);
        output.committedStates.resize(next);
        output.committedQueues.resize(next);
    }
    BT_PLOT("ORG.Execution.StateBarrierSteps", static_cast<int64_t>(entrySteps));
    BT_PLOT("ORG.Execution.IntraBatchStateBarrierSteps", static_cast<int64_t>(intraBatchSteps));
    BT_PLOT("ORG.Execution.CrossQueueStateBarrierSteps", static_cast<int64_t>(crossQueueSteps));
    if (intraBatchSteps)
        basic_telemetry::AddCounter("ORG.Execution.IntraBatchStateBarrierSteps",
            static_cast<int64_t>(intraBatchSteps));
    if (trace::Enabled(trace::Category::Barriers)) {
        // Every texture barrier the plan holds, where it is recorded.
        std::unordered_map<uint64_t, std::string> names;
        for (const auto& backing : ordered) {
            if (!backing.resource.valid()) continue;
            names.emplace(TraceHandle(backing.resource), TraceResourceName(backing));
            for (const auto device : backing.deviceHandles) names.emplace(TraceHandle(device), TraceResourceName(backing));
        }
        const auto emit = [&](const rhi::TextureBarrier& barrier, const char* list, uint32_t batch, uint32_t queue,
                              uint32_t pass) {
            const auto name = names.find(TraceHandle(barrier.texture));
            ORG_TRACE(Barriers, "barrier").Resource(name == names.end() ? std::string("<unknown>") : name->second)
                .Add("handle", TraceHandle(barrier.texture)).Add("list", list).Add("batch", batch).Add("queue", queue)
                .Add("pass", TracePassName(graph, pass))
                .Add("beforeLayout", TraceLayout(barrier.beforeLayout)).Add("afterLayout", TraceLayout(barrier.afterLayout))
                .Add("beforeAccess", static_cast<uint64_t>(barrier.beforeAccess))
                .Add("afterAccess", static_cast<uint64_t>(barrier.afterAccess))
                .Add("ownership", TraceOwnership(barrier.queueOwnership)).Add("discard", barrier.discard)
                .Add("mips", std::to_string(barrier.range.baseMip) + "+" + std::to_string(barrier.range.mipCount));
        };
        for (const auto& release : result.releases)
            for (const auto& barrier : release.textures) emit(barrier, "release-list", UINT32_MAX, release.queue, UINT32_MAX);
        for (uint32_t batch = 0; batch < result.batches.size(); ++batch) {
            const auto& out = result.batches[batch];
            const auto queue = graph.batches[batch].queue;
            for (const auto& barrier : out.textures) emit(barrier, "entry", batch, queue, UINT32_MAX);
            for (const auto& barrier : out.acquired) emit(barrier, "entry-acquired", batch, queue, UINT32_MAX);
            for (uint32_t position = 0; position < out.beforePass.size(); ++position) {
                const auto pass = graph.batches[batch].passes[position];
                for (const auto& barrier : out.beforePass[position].textures) emit(barrier, "before-pass", batch, queue, pass);
                for (const auto& barrier : out.afterPass[position].textures) emit(barrier, "after-pass", batch, queue, pass);
            }
        }
    }
    return result;
}

void BackingStateAdmissionLedger::CommitBatch(const PreparedBatchBarriers& batch) {
    if (batch.committedResources.size() != batch.committedStates.size()
        || batch.committedQueues.size() != batch.committedStates.size())
        throw std::invalid_argument("Invalid backing-state commit");
    for (const auto& seed : batch.seeds) {
        const auto key = Key(seed.resource);
        const bool authoritative = std::ranges::any_of(batch.authoritativeSeeds,
            [&](auto resource) { return SameHandle(resource,seed.resource); });
        if (m_states.Contains(key) && !authoritative) continue;
        auto& grid = m_states.Emplace(key);
        grid = StateGrid{};
        grid.shape = seed.shape;
        grid.cells.resize(size_t{grid.shape.mips} * grid.shape.slices);
        grid.queues.assign(grid.cells.size(), UINT32_MAX);
        if (seed.regions) for (const auto& region : *seed.regions)
            Visit(grid.shape, region.range, [&](uint32_t mip, uint32_t slice) {
                grid.cells[CellIndex(grid.shape, mip, slice)] = region.state;
            });
        grid.fingerprint = Fingerprint(grid);
    }
    for (size_t i = 0; i < batch.committedStates.size(); ++i) {
        const auto key = Key(batch.committedResources[i]);
        auto* grid = m_states.Find(key);
        if (!grid)
            throw std::logic_error("Backing state was not seeded before commit");
        const auto& update = batch.committedStates[i];
        Visit(grid->shape, update.range, [&](uint32_t mip, uint32_t slice) {
            const auto cellIndex = CellIndex(grid->shape, mip, slice);
            grid->cells[cellIndex] = update.state;
            grid->queues[cellIndex] = batch.committedQueues[i];
        });
        grid->fingerprint = Fingerprint(*grid);
        ORG_TRACE(Admission, "ledger-commit").Add("handle", TraceHandle(batch.committedResources[i]))
            .Add("layout", TraceLayout(update.state.layout)).Add("queue", batch.committedQueues[i])
            .Add("mips", std::to_string(update.range.mip) + "+" + std::to_string(update.range.mips));
    }
}

void BackingStateAdmissionLedger::Invalidate(std::span<const rhi::ResourceHandle> resources) {
    for (const auto resource : resources) if (resource.valid()) m_states.Erase(Key(resource));
}

void BackingStateAdmissionLedger::Capture(StatePrediction& prediction, const Schedule& graph,
    std::span<const PreparedBackingState> initial, uint64_t backingSet) const {
    BT_ZONE_SCOPE("ORG.Execution.CapturePrediction");
    auto& resolved = m_states.Resolve(initial, backingSet);
    std::vector<uint8_t> taken(initial.size());
    prediction.entries.clear();
    prediction.entries.reserve(graph.states.entrySteps.size());
    for (const auto index : graph.states.entrySteps) {
        const auto resource = graph.states.steps[index].resource;
        if (taken[resource] || !initial[resource].resource.valid()) continue;
        taken[resource] = 1;
        if (!resolved[resource]) resolved[resource] = const_cast<StateGrid*>(m_states.Find(Key(initial[resource].resource)));
        const auto* grid = resolved[resource];
        prediction.entries.push_back({resource, grid != nullptr, grid ? grid->fingerprint : 0});
    }
}

bool BackingStateAdmissionLedger::Predicts(const StatePrediction& prediction, std::span<const PreparedBackingState> initial,
    uint64_t backingSet) const {
    BT_ZONE_SCOPE("ORG.Execution.CheckPrediction");
    if (!prediction.plan) return false;
    auto& resolved = m_states.Resolve(initial, backingSet);
    for (const auto& entry : prediction.entries) {
        if (entry.resource >= resolved.size()) return false;
        auto*& grid = resolved[entry.resource];
        // None then: one now (the first commit seeds it) is a change; still none is not, and costs a lookup.
        if (!grid) grid = const_cast<StateGrid*>(m_states.Find(Key(initial[entry.resource].resource)));
        if (!grid) {
            if (entry.present) return false;
            continue;
        }
        if (!entry.present || grid->fingerprint != entry.fingerprint) return false;
    }
    return true;
}

void BackingStateAdmissionLedger::CommitBatch(const PreparedBatchBarriers& batch, std::span<const PreparedBackingState> backings,
    uint64_t backingSet) {
    BT_ZONE_SCOPE("ORG.Execution.CommitStates");
    if (batch.committedResources.size() != batch.committedStates.size()
        || batch.committedQueues.size() != batch.committedStates.size()
        || batch.committedIndices.size() != batch.committedStates.size())
        throw std::invalid_argument("Invalid backing-state commit");
    if (!batch.seeds.empty()) {
        // Seeding inserts grids (the first executions of a backing): every grid found by its key.
        CommitBatch(batch);
        return;
    }
    auto& resolved = m_states.Resolve(backings, backingSet);
    for (size_t i = 0; i < batch.committedStates.size(); ++i) {
        const auto index = batch.committedIndices[i];
        StateGrid* grid = index < resolved.size() ? resolved[index] : nullptr;
        if (!grid) {
            grid = m_states.Find(Key(batch.committedResources[i]));
            if (!grid) throw std::logic_error("Backing state was not seeded before commit");
            if (index < resolved.size()) resolved[index] = grid;
        }
        const auto& update = batch.committedStates[i];
        // A commit that leaves the cells as they were (an execution that returns what it touches to where it found it, a closed
        // one's home states) writes nothing, and the fingerprint stands.
        bool changed = false;
        Visit(grid->shape, update.range, [&](uint32_t mip, uint32_t slice) {
            const auto cellIndex = CellIndex(grid->shape, mip, slice);
            if (grid->cells[cellIndex] == update.state && grid->queues[cellIndex] == batch.committedQueues[i]) return;
            grid->cells[cellIndex] = update.state;
            grid->queues[cellIndex] = batch.committedQueues[i];
            changed = true;
        });
        if (changed) grid->fingerprint = Fingerprint(*grid);
        ORG_TRACE(Admission, "ledger-commit").Add("handle", TraceHandle(batch.committedResources[i]))
            .Add("layout", TraceLayout(update.state.layout)).Add("queue", batch.committedQueues[i])
            .Add("mips", std::to_string(update.range.mip) + "+" + std::to_string(update.range.mips));
    }
}

namespace {
template<class Fn>
void VisitPassAccesses(const Schedule& graph, uint32_t passIndex, Fn&& fn) {
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
    const Schedule& graph, std::span<const PreparedBackingState> resources) {
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

uint64_t BackingAccessAdmissionLedger::OpenStream(uint32_t batches) {
    const auto stream = m_nextStream++;
    auto& producers = m_streams[stream];
    producers.reserve(batches);
    for (uint32_t batch = 0; batch < batches; ++batch) {
        uint32_t id = 0;
        if (!m_freeProducers.empty()) {
            id = m_freeProducers.back();
            m_freeProducers.pop_back();
            m_producers[id] = {};
        } else {
            id = static_cast<uint32_t>(m_producers.size());
            m_producers.emplace_back();
        }
        producers.push_back(id);
    }
    return stream;
}

void BackingAccessAdmissionLedger::CloseStream(uint64_t stream) {
    const auto found = m_streams.find(stream);
    if (found == m_streams.end()) return;
    for (const auto id : found->second) m_producers[id].closed = true;
    m_closedProducers += found->second.size();
    m_streams.erase(found);
    if (m_closedProducers >= 1024) FixClosedProducers();
}

void BackingAccessAdmissionLedger::FixClosedProducers() {
    m_accesses.ForEach([&](uint64_t, AccessGrid& grid) {
        bool changed = false;
        const auto fix = [&](AccessRef& ref) {
            if (!ref.producer || !m_producers[ref.producer].closed) return;
            ref = {0, m_producers[ref.producer].signal};
            changed = true;
        };
        for (auto& cell : grid.cells) {
            fix(cell.writer);
            for (auto& reader : cell.readers) fix(reader);
            // Fixed points keep one reader per timeline, the latest.
            auto& readers = cell.readers;
            for (size_t i = 0; i < readers.size(); ++i) {
                if (readers[i].producer) continue;
                for (size_t j = readers.size(); j-- > i + 1;)
                    if (!readers[j].producer && readers[j].point.timeline == readers[i].point.timeline) {
                        readers[i].point.value = (std::max)(readers[i].point.value, readers[j].point.value);
                        readers.erase(readers.begin() + static_cast<std::ptrdiff_t>(j));
                    }
            }
        }
        if (changed) grid.fingerprint = Fingerprint(grid);
    });
    for (uint32_t id = 1; id < m_producers.size(); ++id)
        if (m_producers[id].closed) {
            m_producers[id] = {};
            m_freeProducers.push_back(id);
        }
    m_closedProducers = 0;
}

uint64_t BackingAccessAdmissionLedger::Fingerprint(const AccessGrid& grid) {
    uint64_t hash = 1469598103934665603ull;
    const auto mix = [&](uint64_t value) { hash = (hash ^ value) * 1099511628211ull; };
    const auto mixRef = [&](const AccessRef& ref) {
        if (ref.producer) mix(0x8000000000000000ull | ref.producer);
        else { mix(ref.point.timeline); mix(ref.point.value); }
    };
    for (const auto& cell : grid.cells) {
        mixRef(cell.writer);
        mix(cell.readers.size());
        for (const auto& reader : cell.readers) mixRef(reader);
    }
    return hash;
}

void BackingAccessAdmissionLedger::AppendIncomingWaits(const Schedule& graph,
    std::span<const PreparedBackingState> resources,
    std::span<const ExecutionTimelinePoint> queues,
    std::vector<std::vector<ExecutionTimelinePoint>>& incoming, uint64_t backingSet, AccessPrediction* capture) const {
    BT_ZONE_SCOPE("ORG.Execution.ResolveCrossFrameHazards");
    if (incoming.size() != graph.batches.size() || queues.size() != graph.structure->queues.size())
        throw std::invalid_argument("Invalid cross-frame hazard dimensions");
    const auto ordered = OrderBackings(graph, resources);
    if (graph.incomingBoundaryAccessesByBatch.size() != graph.batches.size())
        throw std::invalid_argument("Compiled graph has no boundary access stream");
    auto& resolved = m_accesses.Resolve(resources, backingSet);
    // A capture lists each resource once, and each batch's sources once; one with very many sources is not kept.
    std::vector<uint8_t> taken;
    bool keep = capture != nullptr;
    if (capture) {
        capture->valid = false;
        capture->entries.clear();
        capture->waits.assign(graph.batches.size(), {});
        taken.assign(ordered.size(), 0);
    }
    uint64_t waitsAdded = 0;
    for (uint32_t batchIndex = 0; batchIndex < graph.batches.size(); ++batchIndex) {
        const auto& batch = graph.batches[batchIndex];
        const auto consumerTimeline = queues[batch.queue].timeline;
        const auto before = incoming[batchIndex].size();
        const auto add = [&](const AccessRef& ref) {
            AppendWait(incoming[batchIndex], PointOf(ref), consumerTimeline);
            if (!keep || (!ref.producer && !ref.point.value)) return;
            auto& sources = capture->waits[batchIndex];
            if (std::ranges::find(sources, ref) != sources.end()) return;
            if (sources.size() >= 64) keep = false;
            else sources.push_back(ref);
        };
        for (const auto& access : graph.incomingBoundaryAccessesByBatch[batchIndex]) {
            const auto resourceIndex = access.resource;
            const bool write = access.state.write;
            if (!ordered[resourceIndex].resource.valid()) continue;
            auto*& grid = resolved[resourceIndex];
            if (!grid) grid = const_cast<AccessGrid*>(m_accesses.Find(Key(ordered[resourceIndex].resource)));
            if (capture && !taken[resourceIndex]) {
                taken[resourceIndex] = 1;
                capture->entries.push_back({resourceIndex, grid != nullptr, grid ? grid->fingerprint : 0});
            }
            if (!grid) continue;
            if (grid->shape != ordered[resourceIndex].shape)
                throw std::invalid_argument("Backing access shape changed without identity change");
            Visit(grid->shape, access.range, [&](uint32_t mip, uint32_t slice) {
                const auto& cell = grid->cells[CellIndex(grid->shape, mip, slice)];
                add(cell.writer);
                if (write) for (const auto& reader : cell.readers) add(reader);
            });
        }
        waitsAdded += incoming[batchIndex].size() - before;
    }
    if (capture) capture->valid = keep;
    BT_PLOT("ORG.Execution.CrossFrameHazardWaits", static_cast<int64_t>(waitsAdded));
    if (waitsAdded) basic_telemetry::AddCounter(
        "ORG.Execution.CrossFrameHazardWaits", static_cast<int64_t>(waitsAdded));
}

bool BackingAccessAdmissionLedger::PredictedWaits(const AccessPrediction& prediction, const Schedule& graph,
    std::span<const PreparedBackingState> resources,
    std::span<const ExecutionTimelinePoint> queues,
    std::vector<std::vector<ExecutionTimelinePoint>>& incoming, uint64_t backingSet) const {
    BT_ZONE_SCOPE("ORG.Execution.PredictedCrossFrameHazards");
    if (!prediction.valid || prediction.waits.size() != graph.batches.size() || incoming.size() != graph.batches.size()
        || queues.size() != graph.structure->queues.size())
        return false;
    auto& resolved = m_accesses.Resolve(resources, backingSet);
    for (const auto& entry : prediction.entries) {
        if (entry.resource >= resolved.size()) return false;
        auto*& grid = resolved[entry.resource];
        if (!grid) grid = const_cast<AccessGrid*>(m_accesses.Find(Key(resources[entry.resource].resource)));
        if (!grid) {
            if (entry.present) return false;
            continue;
        }
        if (!entry.present || grid->fingerprint != entry.fingerprint) return false;
    }
    uint64_t waitsAdded = 0;
    for (uint32_t batchIndex = 0; batchIndex < graph.batches.size(); ++batchIndex) {
        const auto consumerTimeline = queues[graph.batches[batchIndex].queue].timeline;
        const auto before = incoming[batchIndex].size();
        for (const auto& ref : prediction.waits[batchIndex]) AppendWait(incoming[batchIndex], PointOf(ref), consumerTimeline);
        waitsAdded += incoming[batchIndex].size() - before;
    }
    BT_PLOT("ORG.Execution.CrossFrameHazardWaits", static_cast<int64_t>(waitsAdded));
    if (waitsAdded) basic_telemetry::AddCounter(
        "ORG.Execution.CrossFrameHazardWaits", static_cast<int64_t>(waitsAdded));
    return true;
}

void BackingAccessAdmissionLedger::AppendBoundaryAccessWaits(const Schedule& graph,
    std::span<const PreparedBackingState> resources, std::span<const uint8_t> selected, ResourceWaits& out, uint64_t backingSet) const {
    BT_ZONE_SCOPE("ORG.Execution.ResolveBoundaryAccessWaits");
    const auto ordered = OrderBackings(graph, resources);
    if (out.size() != graph.batches.size() || graph.incomingBoundaryAccessesByBatch.size() != graph.batches.size()
        || selected.size() != ordered.size())
        throw std::invalid_argument("Invalid boundary access wait dimensions");
    auto& resolved = m_accesses.Resolve(resources, backingSet);
    for (uint32_t batchIndex = 0; batchIndex < graph.batches.size(); ++batchIndex) {
        auto& waits = out[batchIndex];
        for (const auto& access : graph.incomingBoundaryAccessesByBatch[batchIndex]) {
            const auto resourceIndex = access.resource;
            if (!selected[resourceIndex] || !ordered[resourceIndex].resource.valid()) continue;
            auto*& grid = resolved[resourceIndex];
            if (!grid) grid = const_cast<AccessGrid*>(m_accesses.Find(Key(ordered[resourceIndex].resource)));
            if (!grid || grid->shape != ordered[resourceIndex].shape) continue;
            // One entry per timeline among this access's own: another access of the resource may add the same timeline again, which
            // the reader merges (the latest value per timeline). Searching every earlier entry would cost the batch's accesses squared.
            const size_t first = waits.size();
            const auto append = [&](ExecutionTimelinePoint point) {
                if (!point.timeline || !point.value) return;
                for (size_t i = first; i < waits.size(); ++i)
                    if (waits[i].second.timeline == point.timeline) {
                        waits[i].second.value = (std::max)(waits[i].second.value, point.value);
                        return;
                    }
                waits.push_back({resourceIndex, point});
            };
            Visit(grid->shape, access.range, [&](uint32_t mip, uint32_t slice) {
                const auto& cell = grid->cells[CellIndex(grid->shape, mip, slice)];
                append(PointOf(cell.writer));
                for (const auto& reader : cell.readers) append(PointOf(reader));
            });
        }
    }
}

void BackingAccessAdmissionLedger::Commit(const Schedule& graph,
    std::span<const PreparedBackingState> resources,
    const GraphExecutionTimeline& execution, uint32_t batchCount, uint64_t backingSet, uint64_t stream) {
    BT_ZONE_SCOPE("ORG.Execution.CommitCrossFrameHazards");
    if (execution.batches.size() != graph.batches.size())
        throw std::invalid_argument("Invalid submitted hazard dimensions");
    const auto ordered = OrderBackings(graph, resources);
    if (graph.boundaryAccessesByBatch.size() != graph.batches.size())
        throw std::invalid_argument("Compiled graph has no boundary access stream");
    const std::vector<uint32_t>* producers = nullptr;
    // A stream closed since the admission (its executable admitted again with other backings first) commits fixed points.
    if (const auto found = stream ? m_streams.find(stream) : m_streams.end(); found != m_streams.end()) {
        if (found->second.size() != graph.batches.size()) throw std::invalid_argument("Commit to an access stream of other batches");
        producers = &found->second;
    }
    batchCount = (std::min)(batchCount, static_cast<uint32_t>(graph.batches.size()));
    auto& resolved = m_accesses.Resolve(resources, backingSet);
    for (uint32_t batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
        const auto signal = execution.batches[batchIndex].signal;
        if (!signal.timeline || !signal.value) throw std::invalid_argument("Unsubmitted hazard point");
        // A stream's batch: its producer's latest signal, which every cell the stream touched names (the same stream touches the
        // same cells every time, so none is left naming a later signal than its own).
        AccessRef ref{0, signal};
        if (producers) {
            ref = {(*producers)[batchIndex], {}};
            auto& producer = m_producers[ref.producer];
            if (producer.signal.timeline && producer.signal.timeline != signal.timeline)
                throw std::logic_error("A stream's batch signalled another timeline");
            producer.signal = signal;
        }
        for (const auto& access : graph.boundaryAccessesByBatch[batchIndex]) {
            const auto resourceIndex = access.resource;
            const bool write = access.state.write;
            const auto& backing = ordered[resourceIndex];
            if (!backing.resource.valid()) continue;
            bool inserted = false;
            auto*& entry = resolved[resourceIndex];
            if (!entry) entry = &m_accesses.Emplace(Key(backing.resource), &inserted);
            auto& grid = *entry;
            if (inserted) {
                grid.shape = backing.shape;
                grid.cells.resize(size_t{grid.shape.mips} * grid.shape.slices);
            } else if (grid.shape != backing.shape) {
                throw std::invalid_argument("Backing access shape changed without identity change");
            }
            // A cell is written only when who touched it changes, so a frame that repeats leaves it (and the grid's fingerprint) as it
            // was. Its readers are those since its writer last changed, one per producer (a stream's batch) and one per timeline
            // among fixed points (the latest): the same writer again keeps them. A read from before that write is complete when the
            // write is, so the next writer's wait on it is redundant, never wrong.
            bool changed = inserted;
            Visit(grid.shape, access.range, [&](uint32_t mip, uint32_t slice) {
                auto& cell = grid.cells[CellIndex(grid.shape, mip, slice)];
                if (write) {
                    if (cell.writer == ref) return;
                    cell.writer = ref;
                    cell.readers.clear();
                    changed = true;
                    return;
                }
                if (std::ranges::find(cell.readers, ref) != cell.readers.end()) return;
                if (ref.producer) {
                    cell.readers.push_back(ref);
                    changed = true;
                    return;
                }
                auto reader = std::ranges::find_if(cell.readers,
                    [&](const AccessRef& other) { return !other.producer && other.point.timeline == signal.timeline; });
                if (reader == cell.readers.end()) cell.readers.push_back(ref);
                else if (reader->point.value < signal.value) *reader = ref;
                else return;
                changed = true;
            });
            if (changed) grid.fingerprint = Fingerprint(grid);
        }
    }
}

namespace {
bool Overlaps(uint64_t lhsBegin, uint64_t lhsEnd, uint64_t rhsBegin, uint64_t rhsEnd) {
    return (std::max)(lhsBegin, rhsBegin) < (std::min)(lhsEnd, rhsEnd);
}
}

uint64_t AliasAccessAdmissionLedger::OwnSequence(const Heap& heap, const PreparedBackingState& resource) {
    const auto found = heap.occupantByHandle.find(Key(resource.resource));
    if (found == heap.occupantByHandle.end()) return 0;
    const uint64_t end = resource.aliasOffset + resource.aliasSize;
    for (const auto& range : heap.occupants[found->second].ranges)
        if (range.begin == resource.aliasOffset && range.end == end) return range.sequence;
    return 0;
}

template<class Fn>
void AliasAccessAdmissionLedger::ForEachConflict(const PreparedBackingState& resource, Fn&& fn) const {
    if (!resource.aliasHeapIdentity || !resource.aliasSize) return;
    const auto found = m_heaps.find(resource.aliasHeapIdentity);
    if (found == m_heaps.end()) return;
    const auto& heap = found->second;
    const uint64_t own = OwnSequence(heap, resource);
    const uint64_t end = resource.aliasOffset + resource.aliasSize;
    for (const auto& occupant : heap.occupants) {
        if (SameHandle(occupant.handle, resource.resource)) continue;
        for (const auto& range : occupant.ranges) {
            if (range.sequence <= own || !Overlaps(resource.aliasOffset, end, range.begin, range.end)) continue;
            fn(occupant);
            break;
        }
    }
}

std::vector<rhi::ResourceHandle> AliasAccessAdmissionLedger::ApplyInitialStates(
    const Schedule& graph,
    std::span<const PreparedBackingState> resources) const {
    BT_ZONE_SCOPE("ORG.Execution.ResolveAliasActivations");
    const auto ordered = OrderBackings(graph, resources);
    if (graph.aliasFinalResourcesByBatch.size() != graph.batches.size())
        throw std::invalid_argument("Compiled graph has no alias final-use stream");
    std::vector<rhi::ResourceHandle> activated;
    for (const auto& resource : ordered) {
        if (!resource.aliasHeapIdentity || !resource.aliasSize) continue;
        bool replacesOccupant = false;
        ForEachConflict(resource, [&](const Occupant&) { replacesOccupant = true; });
        if (!replacesOccupant) continue;
        activated.push_back(resource.resource);
    }
    BT_PLOT("ORG.Execution.AliasActivations", static_cast<int64_t>(activated.size()));
    if (!activated.empty()) basic_telemetry::AddCounter(
        "ORG.Execution.AliasActivations", static_cast<int64_t>(activated.size()));
    return activated;
}

void AliasAccessAdmissionLedger::AppendIncomingWaits(const Schedule& graph,
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
            ForEachConflict(resource, [&](const Occupant& occupant) {
                ++overlapsFound;
                for (const auto point : occupant.accesses)
                    AppendWait(incoming[batchIndex], point, consumerTimeline);
            });
        }
        waitsAdded += incoming[batchIndex].size() - before;
    }
    BT_PLOT("ORG.Execution.AliasHazardWaits", static_cast<int64_t>(waitsAdded));
    BT_PLOT("ORG.Execution.AliasOverlaps", static_cast<int64_t>(overlapsFound));
    if (waitsAdded) basic_telemetry::AddCounter(
        "ORG.Execution.AliasHazardWaits", static_cast<int64_t>(waitsAdded));
}

void AliasAccessAdmissionLedger::Commit(const Schedule& graph,
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
            if (resource.aliasHeap) m_heapOwners[resource.aliasHeapIdentity] = resource.aliasHeap;
            auto& heap = m_heaps[resource.aliasHeapIdentity];
            const auto key = Key(resource.resource);
            auto slot = heap.occupantByHandle.find(key);
            if (slot == heap.occupantByHandle.end()) {
                slot = heap.occupantByHandle.emplace(key, static_cast<uint32_t>(heap.occupants.size())).first;
                heap.occupants.push_back({resource.resource, {}, {}});
            }
            auto& occupant = heap.occupants[slot->second];
            const uint64_t end = resource.aliasOffset + resource.aliasSize;
            const auto sequence = ++m_sequence;
            auto range = std::find_if(occupant.ranges.begin(), occupant.ranges.end(), [&](const auto& existing) {
                return existing.begin == resource.aliasOffset && existing.end == end;
            });
            if (range == occupant.ranges.end()) occupant.ranges.push_back({resource.aliasOffset, end, sequence});
            else range->sequence = sequence;
            auto point = std::find_if(occupant.accesses.begin(), occupant.accesses.end(),
                [&](const auto& existing) { return existing.timeline == signal.timeline; });
            if (point == occupant.accesses.end()) occupant.accesses.push_back(signal);
            else if (point->value < signal.value) *point = signal;
        }
    }
}

void BackingAccessAdmissionLedger::ResolvePlannedPoints(
    const std::unordered_map<uint64_t, ExecutionTimelinePoint>& points) {
    auto resolve = [&](ExecutionTimelinePoint& point) {
        if (const auto found = points.find(point.value); found != points.end()) point = found->second;
    };
    for (auto& producer : m_producers) resolve(producer.signal);
    m_accesses.ForEach([&](uint64_t, AccessGrid& grid) {
        for (auto& cell : grid.cells) {
            if (!cell.writer.producer) resolve(cell.writer.point);
            for (auto& reader : cell.readers) if (!reader.producer) resolve(reader.point);
        }
        grid.fingerprint = Fingerprint(grid);
    });
}

void AliasAccessAdmissionLedger::ResolvePlannedPoints(
    const std::unordered_map<uint64_t, ExecutionTimelinePoint>& points) {
    for (auto& [_, heap] : m_heaps) for (auto& occupant : heap.occupants)
        for (auto& point : occupant.accesses)
            if (const auto found = points.find(point.value); found != points.end()) point = found->second;
}

} // namespace org::compiler
