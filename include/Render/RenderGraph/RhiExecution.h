#pragma once

#include <atomic>
#include "Render/RenderGraph/GraphCompiler.h"
#include "Render/RenderGraph/ExecutionState.h"
#include "Render/PreparedPass.h"
#include "Render/RenderGraph/PersistentGraph.h"
#include "Render/RenderGraph/SubmissionCensus.h"
#include "Render/BufferBarrierHelpers.h"
#include "Render/CommandListPool.h"
#include "Render/Runtime/IStatisticsService.h"
#include <algorithm>
#include <chrono>
#include <functional>
#include <span>
#include <mutex>
#include <set>
#include <string_view>
#include <rhi.h>
#include <rhi_colors.h>
#include <rhi_debug.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <BasicTelemetry/Tracy.h>

namespace org::compiler {

// Immutable realization selected with a compiled structural plan. Compilation
// never dereferences this object; recording and retirement retain it so backing
// replacement cannot invalidate a selected frame.
struct RealizedResourceBundle {
    std::vector<uint64_t> backingGenerations;
    std::vector<std::string> resourceKeys;
    // Frame-bound slots (swapchain images and equivalent dynamic imports) are
    // rebound from the admission frame. Other slots stay owned by the request.
    // Zero means fixed in the captured realization; non-zero is the
    // ExternalBindingKey resolved during ordered admission.
    std::vector<uint32_t> admissionBoundResources;
    std::shared_ptr<const FrozenExecutionBindings> bindings;
    std::shared_ptr<const std::vector<PreparedBackingState>> initialStates;
    std::vector<std::shared_ptr<const void>> leases;
};

// Immutable frame publication consumed by recording/admission. Fresh pass
// packets are deliberately separate from the reusable compiled layout.
struct RenderFrameSnapshot {
    RenderFrameSnapshot() = default;
    RenderFrameSnapshot(const RenderFrameSnapshot&) = delete;
    RenderFrameSnapshot& operator=(const RenderFrameSnapshot&) = delete;
    ~RenderFrameSnapshot() {
        // Recording workers and submitted batches retain this packet. Its
        // final owner disappears only after their CPU/GPU terminal conditions.
        // Submitted/completed effects are guarded by PreparedPass itself.
        if (publication) for (const auto& pass : passes) {
            try { pass.Abandon(AbandonReason::AdmissionFailed); }
            catch (...) { basic_telemetry::AddCounter("ORG.Persistent.AbandonLifecycleFailures"); }
        }
    }
    std::shared_ptr<const persistent::SelectedPublication> publication;
    std::vector<std::vector<ExecutionTimelinePoint>> incomingWaits;
    uint64_t frameNumber = 0;
    uint32_t preparationSlot = 0;
    std::shared_ptr<const GraphExecutionLayout> layout;
    std::vector<PreparedPass> passes;
    std::shared_ptr<const FrozenExecutionBindings> bindings;
    std::shared_ptr<const std::vector<PreparedBackingState>> initialStates;
    std::vector<std::vector<ExternalTimelinePoint>> externalWaitsByPreparedPass;
    std::shared_ptr<const PreparedExecutionBarrierPlan> barrierPlan;
    std::vector<std::shared_ptr<const void>> leases;
    std::shared_ptr<const RealizedResourceBundle> resources;
    std::shared_ptr<const IHostExecutionData> frameData;
};

// Async sealing is an explicit ownership operation. Synchronous preparation
// need not allocate this packet or construct a compile request/candidate.
inline std::shared_ptr<const RenderFrameSnapshot> SealPersistentFrame(
    uint64_t frameNumber, const persistent::FrameAdmission& admission,
    std::vector<PreparedPass> invocations) {
    const auto selected = admission.publication;
    // A plan (SynchronousAdmission::PreparePlan, a ticket's body) has no sequence: it is sealed the same way.
    if (!frameNumber || !admission.domain || !selected || !selected->executable
        || !selected->executable->executionLayout)
        throw std::invalid_argument("Incomplete persistent frame seal");
    const auto layout = selected->executable->executionLayout;
    const auto& barriers = admission.Barriers();
    if (invocations.size() != layout->placements.size()
        || barriers.batches.size() != selected->executable->graph->batches.size()
        || admission.incomingWaits.size() != barriers.batches.size())
        throw std::invalid_argument("Persistent frame seal dimensions mismatch");
    const auto& slots = selected->executable->resourceSlots;
    if (admission.backings.size() != slots.size())
        throw std::invalid_argument("Persistent frame seal backing count mismatch");
    for (size_t i = 0; i < slots.size(); ++i) {
        const auto& expected = selected->bindings.At(slots[i]).admission;
        const auto& actual = admission.backings[i];
        if (std::find(admission.rebound.begin(), admission.rebound.end(), static_cast<uint32_t>(i)) != admission.rebound.end()) {
            // Frame-supplied backing: the contract (shape/heap) is checked by
            // admission; the physical identity is deliberately not the publication's.
            if (!expected || !actual.resource.valid() || !actual.regions || actual.shape != expected->shape)
                throw std::invalid_argument("Persistent frame seal has an invalid rebound backing");
            continue;
        }
        if (!expected || actual.resource.index != expected->resource.index
            || actual.resource.generation != expected->resource.generation
            || actual.graphResourceID != expected->graphResourceID || actual.shape != expected->shape
            || !actual.regions || actual.aliasHeapIdentity != expected->aliasHeapIdentity
            || actual.aliasOffset != expected->aliasOffset || actual.aliasSize != expected->aliasSize
            || actual.aliasPoolID != expected->aliasPoolID || actual.heapType != expected->heapType
            || actual.aliasHeap.get() != expected->aliasHeap.get()
            || actual.aliasHeap.owner_before(expected->aliasHeap) || expected->aliasHeap.owner_before(actual.aliasHeap))
            throw std::invalid_argument("Persistent frame seal lost selected backing metadata");
    }
    for (size_t i = 0; i < invocations.size(); ++i) {
        const bool active = layout->placements[i].preparedPass != UINT32_MAX;
        if (active != bool(invocations[i]) || (active && (invocations[i].IsConsumed() || !invocations[i].IsWorkerSafe())))
            throw std::invalid_argument("Persistent frame seal requires fresh active invocations");
    }
    auto result = std::make_shared<RenderFrameSnapshot>();
    result->frameNumber = frameNumber;
    result->incomingWaits = admission.incomingWaits;
    result->layout = layout;
    result->passes = std::move(invocations);
    result->initialStates = admission.backings.Shared();
    // A shared plan (a prediction's, or a ticket body's: SynchronousAdmission::CachedBody) is immutable, so it is kept, not copied.
    result->barrierPlan = admission.sharedBarriers ? admission.sharedBarriers
        : std::make_shared<const PreparedExecutionBarrierPlan>(admission.barriers);
    // Arm lifecycle ownership only after every fallible allocation succeeds.
    // A pending packet still owns these handles if sealing fails partway through.
    result->publication = selected;
    return result;
}

// Owned asynchronous preparation before ordered admission. CPU tasks retain the
// packet until they join; final release abandons untransferred invocation effects.
// Seal is called only by the ordered owner, after preparation consumers join.
// The synchronous path continues to borrow tables directly and needs no packet.
class PendingPersistentFrame {
public:
    PendingPersistentFrame(uint64_t frameNumber,
        std::shared_ptr<const persistent::SelectedPublication> selected,
        std::vector<PreparedPass> invocations)
        : m_frameNumber(frameNumber), m_publication(std::move(selected)), m_passes(std::move(invocations)) {
        try {
            if (!m_frameNumber || !m_publication || !m_publication->executable
                || !m_publication->executable->executionLayout)
                throw std::invalid_argument("Incomplete pending persistent frame");
            const auto& placements = m_publication->executable->executionLayout->placements;
            if (placements.size() != m_passes.size())
                throw std::invalid_argument("Pending persistent invocation dimensions mismatch");
            for (size_t i = 0; i < placements.size(); ++i) {
                const bool active = placements[i].preparedPass != UINT32_MAX;
                if (active != bool(m_passes[i]) || (active && (m_passes[i].IsConsumed() || !m_passes[i].IsWorkerSafe())))
                    throw std::invalid_argument("Pending persistent frame requires fresh invocations");
            }
        } catch (...) { Abandon(); throw; }
    }
    PendingPersistentFrame(const PendingPersistentFrame&) = delete;
    PendingPersistentFrame& operator=(const PendingPersistentFrame&) = delete;
    ~PendingPersistentFrame() { Abandon(); }
    const std::shared_ptr<const persistent::SelectedPublication>& Publication() const noexcept { return m_publication; }
    std::shared_ptr<const RenderFrameSnapshot> Seal(const persistent::FrameAdmission& admission) {
        if (m_transferred) throw std::logic_error("Pending persistent frame already transferred");
        if (admission.publication != m_publication)
            throw std::invalid_argument("Pending persistent frame admission changed publication");
        // Copy the small handles so failed validation retains the pending effects.
        auto result = SealPersistentFrame(m_frameNumber,admission,m_passes);
        m_transferred = true;
        m_passes.clear();
        return result;
    }
private:
    void Abandon() noexcept {
        if (m_transferred) return;
        for (const auto& pass : m_passes) {
            try { pass.Abandon(AbandonReason::AdmissionFailed); }
            catch (...) { basic_telemetry::AddCounter("ORG.Persistent.AbandonLifecycleFailures"); }
        }
    }
    uint64_t m_frameNumber;
    const std::shared_ptr<const persistent::SelectedPublication> m_publication;
    std::vector<PreparedPass> m_passes;
    bool m_transferred = false;
};

struct PreparedFramePayload final : IFramePayloadLifecycle {
    uint64_t frameNumber = 0;
    uint32_t preparationSlot = 0;
    std::vector<PreparedPass> passes;
    std::shared_ptr<const FrozenExecutionBindings> bindings;
    std::shared_ptr<const std::vector<PreparedBackingState>> initialStates;
    std::vector<std::vector<ExternalTimelinePoint>> externalWaitsByPreparedPass;
    std::vector<std::shared_ptr<const void>> leases;
    std::shared_ptr<const RealizedResourceBundle> resources;
    std::shared_ptr<const IHostExecutionData> frameData;
    void Abandon(uint8_t reason) const noexcept override {
        if (abandoned.exchange(true)) return;
        for (const auto& pass : passes)
            pass.Abandon(static_cast<AbandonReason>(reason));
    }
private:
    mutable std::atomic_bool abandoned{false};
};

// Captured during Update alongside the structural request. Frame-varying typed
// pass data is prepared later at ordered admission, while immediate one-shot
// work is reserved here so a delayed/cancelled compile cannot silently consume
// it from a newer live pass container.
struct FramePreparationBasis {
    std::shared_ptr<const RealizedResourceBundle> resources;
    // Execution-time synchronization is deliberately excluded from the
    // structural compiler key and remapped to batches after graph selection.
    std::vector<std::vector<ExternalTimelinePoint>> externalWaitsByPreparedPass;
    // Indexed by the source frame-pass slot. Empty entries are ordinary typed
    // passes prepared fresh at admission.
    std::vector<PreparedPass> reservedImmediatePasses;
    std::vector<uint8_t> immediatePassSlots;
    std::shared_ptr<const IHostExecutionData> updateData;
};

struct PreparedBatchRecording {
    uint32_t queueSlot = 0;
    std::vector<PreparedPass> passes;
};

// Converts the immutable compiler layout into the exact recording batches.
// This is deliberately independent of command allocator creation so admission
// can reserve an execution slot before any backend object is mutated.
inline std::vector<PreparedBatchRecording> BuildPreparedBatchRecordings(
    const RenderFrameSnapshot& frame) {
    BT_ZONE_SCOPE("ORG.Execution.BuildRecordingPlan");
    if (!frame.layout || !frame.layout->bundle || !frame.layout->bundle->graph
        || frame.passes.size() != frame.layout->placements.size())
        throw std::invalid_argument("Incomplete async recording plan input");
    const auto& batches = frame.layout->bundle->graph->batches;
    const auto& structure = *frame.layout->bundle->graph->structure;
    std::vector<PreparedBatchRecording> result(batches.size());
    std::vector<uint8_t> assigned(frame.passes.size());
    for (uint32_t batch = 0; batch < batches.size(); ++batch) {
        result[batch].queueSlot = batches[batch].queue;
        for (const auto compilerPass : batches[batch].passes) {
            if (compilerPass >= structure.passes.size())
                throw std::invalid_argument("Invalid compiler pass in recording plan");
            const auto prepared = structure.passes[compilerPass].preparedPassIndex;
            if (prepared >= frame.layout->placements.size())
                throw std::invalid_argument("Invalid prepared pass in recording plan");
            const auto placement = frame.layout->placements[prepared];
            if (placement.preparedPass != prepared || placement.batch != batch
                || placement.queue != batches[batch].queue || assigned[prepared])
                throw std::invalid_argument("Invalid async pass placement");
            result[batch].passes.push_back(frame.passes[prepared]);
            assigned[prepared] = 1;
        }
    }
    for (uint32_t batch = 0; batch < result.size(); ++batch) {
        if (result[batch].passes.empty())
            throw std::invalid_argument("Compiled batch has no prepared recording packet");
    }
    return result;
}

inline std::shared_ptr<const PreparedFramePayload> BuildPreparedFramePayload(
    uint64_t frameNumber, std::vector<PreparedPass> passes,
    std::shared_ptr<const FrozenExecutionBindings> bindings,
    std::vector<PreparedBackingState> initialStates,
    std::vector<std::vector<ExternalTimelinePoint>> externalWaitsByPreparedPass = {},
    std::vector<std::shared_ptr<const void>> leases = {},
    uint32_t preparationSlot = 0) {
    if (!frameNumber || !bindings || passes.empty() || initialStates.empty())
        throw std::invalid_argument("Incomplete prepared frame payload");
    for (const auto& pass : passes)
        if (!pass) throw std::invalid_argument("Legacy pass prevents async frame preparation");
    auto result = std::make_shared<PreparedFramePayload>();
    result->frameNumber = frameNumber;
    result->preparationSlot = preparationSlot;
    result->passes = std::move(passes);
    result->bindings = std::move(bindings);
    result->initialStates = std::make_shared<const std::vector<PreparedBackingState>>(
        std::move(initialStates));
    result->externalWaitsByPreparedPass = std::move(externalWaitsByPreparedPass);
    result->leases = std::move(leases);
    TraceBindingHolder(result->bindings->PublicationRoot(), result, "PreparedFramePayload", frameNumber);
    return result;
}

inline std::shared_ptr<const PreparedFramePayload> BuildPreparedFramePayloadWithInitialStates(
    uint64_t frameNumber, std::vector<PreparedPass> passes,
    std::shared_ptr<const RealizedResourceBundle> resources,
    std::vector<PreparedBackingState> initialStates,
    std::vector<std::vector<ExternalTimelinePoint>> externalWaitsByPreparedPass = {},
    uint32_t preparationSlot = 0,
    std::shared_ptr<const IHostExecutionData> frameData = {}) {
    if (!resources || !frameNumber || !resources->bindings || passes.empty() || initialStates.empty())
        throw std::invalid_argument("Incomplete realized frame payload");
    for (const auto& pass : passes)
        if (!pass) throw std::invalid_argument("Legacy pass prevents async frame preparation");
    auto result = std::make_shared<PreparedFramePayload>();
    result->frameNumber = frameNumber;
    result->preparationSlot = preparationSlot;
    result->passes = std::move(passes);
    result->bindings = resources->bindings;
    result->initialStates = std::make_shared<const std::vector<PreparedBackingState>>(
        std::move(initialStates));
    result->externalWaitsByPreparedPass = std::move(externalWaitsByPreparedPass);
    result->leases = resources->leases;
    result->resources = std::move(resources);
    result->frameData = std::move(frameData);
    TraceBindingHolder(result->bindings->PublicationRoot(), result, "PreparedFramePayload", frameNumber);
    return result;
}

inline std::shared_ptr<const PreparedFramePayload> BuildPreparedFramePayload(
    uint64_t frameNumber, std::vector<PreparedPass> passes,
    std::shared_ptr<const RealizedResourceBundle> resources,
    std::vector<std::vector<ExternalTimelinePoint>> externalWaitsByPreparedPass = {},
    uint32_t preparationSlot = 0,
    std::shared_ptr<const IHostExecutionData> frameData = {}) {
    if (!resources || !frameNumber || !resources->bindings || passes.empty()
        || !resources->initialStates || resources->initialStates->empty())
        throw std::invalid_argument("Incomplete realized frame payload");
    for (const auto& pass : passes)
        if (!pass) throw std::invalid_argument("Legacy pass prevents async frame preparation");
    auto result = std::make_shared<PreparedFramePayload>();
    result->frameNumber = frameNumber;
    result->preparationSlot = preparationSlot;
    result->passes = std::move(passes);
    result->bindings = resources->bindings;
    result->initialStates = resources->initialStates;
    result->externalWaitsByPreparedPass = std::move(externalWaitsByPreparedPass);
    result->leases = resources->leases;
    result->resources = std::move(resources);
    result->frameData = std::move(frameData);
    TraceBindingHolder(result->bindings->PublicationRoot(), result, "PreparedFramePayload", frameNumber);
    return result;
}

inline std::shared_ptr<const RenderFrameSnapshot> BuildRenderFrameSnapshot(
    uint64_t frameNumber,
    std::shared_ptr<const GraphExecutionLayout> layout,
    std::vector<PreparedPass> passes,
    std::shared_ptr<const FrozenExecutionBindings> bindings,
    std::shared_ptr<const std::vector<PreparedBackingState>> initialStates,
    std::shared_ptr<const PreparedExecutionBarrierPlan> barrierPlan,
    std::vector<std::shared_ptr<const void>> leases = {},
    std::vector<std::vector<ExternalTimelinePoint>> externalWaitsByPreparedPass = {},
    std::shared_ptr<const RealizedResourceBundle> resources = {},
    uint32_t preparationSlot = 0,
    std::shared_ptr<const IHostExecutionData> frameData = {}) {
    BT_ZONE_SCOPE("ORG.Execution.BuildFrameSnapshot");
    if (!frameNumber || !layout || !layout->bundle || !bindings
        || passes.size() != layout->placements.size() || !initialStates
        || initialStates->empty() || !barrierPlan
        || (!externalWaitsByPreparedPass.empty() && externalWaitsByPreparedPass.size() != passes.size())
        || barrierPlan->batches.size() != layout->bundle->graph->batches.size())
        throw std::invalid_argument("Incomplete async render-frame snapshot");
    for (const auto& pass : passes)
        if (!pass) throw std::invalid_argument("Legacy pass prevents async frame publication");
    auto result = std::make_shared<RenderFrameSnapshot>();
    result->frameNumber = frameNumber;
    result->preparationSlot = preparationSlot;
    result->layout = std::move(layout);
    result->passes = std::move(passes);
    result->bindings = std::move(bindings);
    result->initialStates = std::move(initialStates);
    if (externalWaitsByPreparedPass.empty()) externalWaitsByPreparedPass.resize(result->passes.size());
    result->externalWaitsByPreparedPass = std::move(externalWaitsByPreparedPass);
    result->barrierPlan = std::move(barrierPlan);
    result->leases = std::move(leases);
    result->resources = std::move(resources);
    result->frameData = std::move(frameData);
    return result;
}

inline std::shared_ptr<const RenderFrameSnapshot> BuildRenderFrameSnapshot(
    std::shared_ptr<const GraphExecutionLayout> layout,
    std::shared_ptr<const PreparedFramePayload> payload,
    BackingStateAdmissionLedger& stateLedger) {
    if (!payload) throw std::invalid_argument("Missing prepared frame payload");
    if (!layout || !layout->bundle || !layout->bundle->graph)
        throw std::invalid_argument("Missing compiled layout for state admission");
    auto barrierPlan = std::make_shared<const PreparedExecutionBarrierPlan>(
        stateLedger.Prepare(*layout->bundle->graph, *payload->initialStates));
    return BuildRenderFrameSnapshot(payload->frameNumber, std::move(layout),
        payload->passes, payload->bindings, payload->initialStates,
        std::move(barrierPlan), payload->leases, payload->externalWaitsByPreparedPass,
        payload->resources, payload->preparationSlot, payload->frameData);
}

// Numeric IDs are admission identities, never casts of backend handles.
struct PreparedTimelineBinding {
    uint64_t identity;
    rhi::TimelineHandle handle;
};

// Preparation records barriers and pass commands before closing these lists.
// The lease must own their allocators, concrete backing/descriptor versions,
// queue/device and timelines until retirement. A shared Resource alone is NOT
// such a lease: mutable backing may be replaced while that object remains alive.
// This adapter is not yet used by legacy passes or the benchmark renderer.
struct OwnedRecordingList;

class PreparedRhiExecutionBatch final : public IPreparedExecutionBatch {
public:
    // replayable: every pass is replayable (PreparedPass::IsReplayable), and the lists were recorded for resubmission.
    // Such a packet has no submission effects. Each submission, after the previous one completed, submits the same lists.
    // Its lists go back to their pool when its last owner releases it, not at a completion.
    PreparedRhiExecutionBatch(uint32_t queueSlot, rhi::Queue queue, std::vector<rhi::CommandList> lists,
        std::vector<PreparedTimelineBinding> timelines, std::shared_ptr<const void> lease,
        std::vector<PreparedPass> submissionEffects = {}, std::function<void()> retireCommandLists = {}, bool replayable = false)
        : m_queueSlot(queueSlot), m_queue(queue), m_lists(std::move(lists)), m_timelines(std::move(timelines)),
          m_lease(std::move(lease)), m_submissionEffects(std::move(submissionEffects)),
          m_retireCommandLists(std::move(retireCommandLists)), m_replayable(replayable) {
        if (!m_queue || m_lists.empty() || !m_lease || m_lists.size() > UINT32_MAX)
            throw std::invalid_argument("Incomplete prepared RHI packet");
        if (m_replayable && !m_submissionEffects.empty())
            throw std::invalid_argument("A replayable packet has no submission effects");
        for (auto list : m_lists) if (!list) throw std::invalid_argument("Invalid command list");
        for (size_t i = 0; i < m_timelines.size(); ++i) {
            if (!m_timelines[i].identity) throw std::invalid_argument("Invalid timeline identity");
            for (size_t j = 0; j < i; ++j)
                if (m_timelines[i].identity == m_timelines[j].identity)
                    throw std::invalid_argument("Duplicate timeline identity");
        }
    }
    // Makes a late batch, whose passes are recorded at execution (CompilePass::recordAtExecution). It has no lists yet.
    // - Of an external epoch: its recording is kept for RecordDeferred, which the host's execution calls once, into the
    //   host's list. The graph never submits it.
    // - Inline (OwnedRecordingList::inlineAtExecution, RenderGraph::SetInlineAtExecution): Submit records it into a
    //   primary list of the recording's pool, then submits that list as it submits any other batch's.
    PreparedRhiExecutionBatch(uint32_t queueSlot, rhi::Queue queue, OwnedRecordingList deferred,
        std::vector<PreparedTimelineBinding> timelines, std::shared_ptr<const void> lease);
    ~PreparedRhiExecutionBatch() override { Abandon(); }
    // Its lists, for a host that executes them, such as an external epoch's secondaries. A deferred batch has none.
    std::span<rhi::CommandList> Lists() const noexcept { return {m_lists.data(), m_lists.size()}; }
    // A late batch: an external epoch's (its host records it) or an inline one (Submit records it).
    bool Deferred() const noexcept { return static_cast<bool>(m_deferred); }
    // A late batch that Submit records itself, into a primary list of its own (RenderGraph::SetInlineAtExecution).
    bool Inline() const noexcept { return m_inline; }
    // An inline late batch: its passes' answers to SkipsAtExecution for this run (SkipsAgain once recorded; 1 for a pass skipped
    // ahead), asked before its list is begun. Its submission asks them once and passes them on (ExecutionBatchTimeline::skips).
    void AskSkips(std::vector<uint8_t>& a_out) const;
    // The timeline an identity binds (a null handle: none).
    rhi::TimelineHandle ResolveTimeline(uint64_t identity) const noexcept {
        for (const auto& item : m_timelines) if (item.identity == identity) return item.handle;
        return {};
    }
    // For an external epoch. The host executes the packet in its own submission (ExternalEpochExecution). That submission
    // waits for and signals what the batch would have. This call:
    // - consumes the packet, as Submit does,
    // - commits its passes as submitted with a_signal. For a deferred batch, this happens once RecordDeferred records them.
    // - adds the passes' external signals to a_signals.
    // Returns false when the packet was already consumed, or when a pass's external signal is invalid.
    bool CommitExternal(uint64_t a_signal, std::vector<rhi::TimelinePoint>& a_signals) const noexcept {
        if (!m_replayable && m_consumed.exchange(true)) return false;
        for (const auto& pass : m_submissionEffects)
            for (const auto& external : pass.ExternalSignalsAfterCompletion()) {
                if (!external.timeline || !external.value) {
                    AbandonEffects();
                    return false;
                }
                a_signals.push_back({external.timeline.GetHandle(), external.value});
            }
        if (m_deferred) {
            m_deferredSignal = a_signal;
            return true;
        }
        return CommitEffectsSubmitted(a_signal);
    }
    // Records an external epoch's late batch into the host's list, outside any pass. Call it once, on the host's thread.
    void RecordDeferred(rhi::CommandList a_list) const;
    uint32_t QueueSlot() const noexcept override { return m_queueSlot; }
    bool Replayable() const noexcept override { return m_replayable; }
    // Whether a timeline identity resolves through this packet's bindings. Only such an identity can be a wait for
    // SubmitLeading.
    bool BindsTimeline(uint64_t identity) const noexcept {
        return std::any_of(m_timelines.begin(), m_timelines.end(), [&](const auto& item) { return item.identity == identity; });
    }
    // Submits `lists` on this packet's queue, ahead of the packet. It waits for `waits`, which must be identities the
    // packet binds, and signals nothing. This is for work a host orders before the batch on its queue, such as
    // RenderGraph's queue-scoped ticket uploads. The batch's own Submit follows in queue order, so the batch's signal
    // covers this work too. It consumes nothing. The packet is submitted as usual afterwards.
    SubmissionReceipt SubmitLeading(std::span<rhi::CommandList> lists, std::span<const ExecutionTimelinePoint> waits) const noexcept {
        if (lists.empty()) return {SubmissionState::Signaled};
        std::vector<rhi::TimelinePoint> points;
        points.reserve(waits.size());
        for (const auto wait : waits) {
            if (!wait.value) continue;
            const auto found = std::find_if(m_timelines.begin(), m_timelines.end(), [&](const auto& item) { return item.identity == wait.timeline; });
            if (found == m_timelines.end()) return {SubmissionState::NotSubmitted, SubmissionFailureStage::Validation};
            points.push_back({found->handle, wait.value});
        }
        const auto submitted = m_queue.Submit({lists.data(), static_cast<uint32_t>(lists.size())},
            {{points.data(), static_cast<uint32_t>(points.size())}, {}});
        if (submitted != rhi::Result::Ok)
            return {SubmissionState::SubmissionUncertain, SubmissionFailureStage::Submit, static_cast<uint32_t>(submitted)};
        return {SubmissionState::Signaled};
    }
    // What one queue submission of one or more batches holds. Gather adds a batch's part to it; the caller submits it on the
    // batches' queue and then calls CommitGathered for each batch, in order.
    struct GatheredSubmission {
        std::vector<rhi::TimelinePoint> waits;
        std::vector<rhi::CommandList> lists;
        std::vector<rhi::TimelinePoint> signals;
        void Clear() noexcept {
            waits.clear();
            lists.clear();
            signals.clear();
        }
    };
    // Adds this batch's waits, lists (its leading lists first) and signals to a_into, without submitting. An inline late batch
    // is recorded here. On failure the batch's effects are abandoned, as Submit's would be, and a_into is unchanged.
    SubmissionReceipt Gather(const ExecutionBatchTimeline& batch, GatheredSubmission& a_into) const noexcept {
        // An external epoch's late batch has no lists: only its host's execution records it.
        if (m_deferred && !m_inline) return {SubmissionState::NotSubmitted, SubmissionFailureStage::Validation};
        // The packet is consumed once, even after a failure, because uploads and readbacks cannot replay.
        // A replayable packet has neither.
        if (!m_replayable && m_consumed.exchange(true)) return {SubmissionState::NotSubmitted, SubmissionFailureStage::Replay};
        auto find = [&](uint64_t identity) -> const PreparedTimelineBinding* {
            for (const auto& item : m_timelines) if (item.identity == identity) return &item;
            for (const auto& item : batch.extraTimelines) if (item.identity == identity) return &item;
            return nullptr;
        };
        const auto* signal = find(batch.signal.timeline);
        if (!signal || !batch.signal.value || batch.signal.value == UINT64_MAX) {
            AbandonEffects();
            return {SubmissionState::NotSubmitted, SubmissionFailureStage::Validation};
        }
        for (auto wait : batch.waits) if (!find(wait.timeline)) {
            AbandonEffects();
            return {SubmissionState::NotSubmitted, SubmissionFailureStage::Validation};
        }
        // One queue submission holds the waits, the lists, the passes' external signals and the batch's
        // own signal. The external signals are fixed at preparation, so they can travel with the lists.
        // A host that orders submissions in its own stream (DXVK) then takes one entry per batch, instead
        // of one per wait and one per signal.
        std::vector<rhi::TimelinePoint> signals;
        for (const auto& pass : m_submissionEffects)
            for (const auto& external : pass.ExternalSignalsAfterCompletion()) {
                if (!external.timeline || !external.value) {
                    AbandonEffects();
                    return {SubmissionState::NotSubmitted, SubmissionFailureStage::Validation};
                }
                signals.push_back({external.timeline.GetHandle(), external.value});
            }
        signals.push_back({signal->handle, batch.signal.value});
        // An inline late batch is recorded now, after its run's commit, into a primary list of its own. Its barriers are the
        // admission's, as any batch's, and it is submitted where a batch recorded ahead would be.
        rhi::CommandList recorded{};
        if (m_inline) {
            recorded = RecordInline(batch.skips, batch.untouched);
            if (!recorded) {
                AbandonEffects();
                return {SubmissionState::NotSubmitted, SubmissionFailureStage::Lifecycle};
            }
        }
        for (auto wait : batch.waits) a_into.waits.push_back({find(wait.timeline)->handle, wait.value});
        a_into.lists.insert(a_into.lists.end(), batch.leading.begin(), batch.leading.end());
        if (m_inline) a_into.lists.push_back(recorded);
        else a_into.lists.insert(a_into.lists.end(), m_lists.begin(), m_lists.end());
        a_into.signals.insert(a_into.signals.end(), signals.begin(), signals.end());
        return {SubmissionState::Signaled};
    }
    // The queue this packet submits on.
    rhi::Queue SubmitQueue() const noexcept { return m_queue; }
    // After the submission that held this batch (Gather) was accepted: commits its passes as submitted with the batch's signal.
    SubmissionReceipt CommitGathered(const ExecutionBatchTimeline& batch) const noexcept {
        bool lifecycleFailed = false;
        for (const auto& pass : m_submissionEffects) {
            try { pass.CommitSubmitted({batch.signal.value}); }
            catch (...) {
                lifecycleFailed = true;
                basic_telemetry::AddCounter("ORG.Execution.SubmissionLifecycleFailures");
            }
        }
        // The signal is already queued; a failed lifecycle is still reported, conservatively, as unsignaled.
        if (lifecycleFailed)
            return {SubmissionState::SubmittedWithoutSignal, SubmissionFailureStage::Lifecycle};
        return {SubmissionState::Signaled};
    }
    SubmissionReceipt Submit(const ExecutionBatchTimeline& batch) const noexcept override {
        GatheredSubmission gathered;
        if (const auto receipt = Gather(batch, gathered); !receipt) return receipt;
        const auto submitted = m_queue.Submit({gathered.lists.data(), static_cast<uint32_t>(gathered.lists.size())},
            {{gathered.waits.data(), static_cast<uint32_t>(gathered.waits.size())},
             {gathered.signals.data(), static_cast<uint32_t>(gathered.signals.size())}});
        if (submitted != rhi::Result::Ok)
            return {SubmissionState::SubmissionUncertain, SubmissionFailureStage::Submit, static_cast<uint32_t>(submitted)};
        return CommitGathered(batch);
    }
    void Complete(uint64_t submission) const noexcept override {
        if (m_replayable) return;  // the lists are reused. They are released when the last owner releases the packet.
        // A deferred batch that its host never recorded, because its execution was signalled unexecuted.
        // Its passes did nothing.
        if (m_deferred && !m_deferredRecorded.load(std::memory_order_acquire)) {
            AbandonEffects();
            return;
        }
        for (const auto& pass : m_submissionEffects) {
            try { pass.CommitCompleted({submission}); }
            catch (...) { basic_telemetry::AddCounter("ORG.Execution.InvalidCompletionTransition"); }
        }
        // Admission has observed every queue completion for this execution.
        // Return command allocators immediately instead of waiting for the
        // retired execution's diagnostic/resource garbage to be destroyed.
        if (!m_commandListsRetired.exchange(true)) {
            if (m_retireCommandLists) {
                try { m_retireCommandLists(); }
                catch (...) { basic_telemetry::AddCounter("ORG.CommandListPool.RetirementFailures"); }
            }
            if (m_inline) RetireInlineList();
        }
    }
    void Abandon() const noexcept override {
        if (m_replayable || m_consumed.exchange(true)) return;
        AbandonEffects();
    }
private:
    // Records the late batch's barriers and passes into a_list (again, for a replayable packet). Commits nothing. Returns the
    // passes recorded (not skipped).
    size_t RecordDeferredCommands(rhi::CommandList a_list, std::span<const uint8_t> a_skips = {}, const UntouchedResources* a_untouched = nullptr) const;
    // An inline late batch: begins its own list (again, for a replayable packet), records it and closes it. A null list
    // when that failed.
    rhi::CommandList RecordInline(std::span<const uint8_t> a_skips, const UntouchedResources* a_untouched) const noexcept;
    // An inline late batch's list goes back to its pool, once its submission completed.
    void RetireInlineList() const noexcept;
    bool CommitEffectsSubmitted(uint64_t a_signal) const noexcept {
        bool lifecycleFailed = false;
        for (const auto& pass : m_submissionEffects) {
            try { pass.CommitSubmitted({a_signal}); }
            catch (...) {
                lifecycleFailed = true;
                basic_telemetry::AddCounter("ORG.Execution.SubmissionLifecycleFailures");
            }
        }
        return !lifecycleFailed;
    }
    void AbandonEffects() const noexcept {
        for (const auto& pass : m_submissionEffects) {
            try { pass.Abandon(AbandonReason::AdmissionFailed); }
            catch (...) { basic_telemetry::AddCounter("ORG.Execution.AbandonLifecycleFailures"); }
        }
    }
    uint32_t m_queueSlot;
    mutable rhi::Queue m_queue;
    mutable std::vector<rhi::CommandList> m_lists; // RHI Span uses non-const handles.
    std::vector<PreparedTimelineBinding> m_timelines;
    std::shared_ptr<const void> m_lease;
    std::vector<PreparedPass> m_submissionEffects;
    mutable std::function<void()> m_retireCommandLists;
    mutable std::atomic_bool m_consumed{false};
    mutable std::atomic_bool m_commandListsRetired{false};
    bool m_replayable = false;
    std::shared_ptr<OwnedRecordingList> m_deferred;
    bool m_inline = false;  // m_deferred->inlineAtExecution
    // An inline replayable batch whose passes are all keyed by their skip answers (OwnedRecordingList::keepBySkips). Its list
    // is recorded again only when the answers differ from those it was recorded with (m_keptSkips). Submitting thread only.
    bool m_keyedBySkips = false;
    mutable bool m_keptList = false;
    mutable std::vector<uint8_t> m_keptSkips;
    mutable uint64_t m_keptUntouched = 0;  // the untouched resources' signature the kept list was recorded with
    mutable std::atomic_bool m_deferredRecorded{false};
    mutable uint64_t m_deferredSignal = 0;  // set by CommitExternal. RecordDeferred commits its passes with it.
};

// One owned recording packet. The binding table transitively owns descriptor
// snapshots and backing versions; pass data owns immutable pipeline references.
// A recording allocation returns its queue-scoped pair only after frame
// ownership proves GPU completion. Member order retires command lists before
// their allocators and bindings.
struct FrameCommandAllocation {
    std::shared_ptr<CommandListPool> pool;
    CommandListPair pair;
    bool closed = false;
    // Call once the list is closed. It gives up the list's allocator claim (CommandListPool's threading rule).
    void Close() noexcept {
        closed = true;
        pair.EndRecording();
    }
    void RecycleAfterCompletion() noexcept {
        if (!pool) return;
        try { pool->RecycleForNextRequest(std::move(pair)); }
        catch (...) { pool->Discard(std::move(pair)); }
        pool.reset();
    }
    ~FrameCommandAllocation() {
        if (!pool) return;
        if (!closed) { pool->Discard(std::move(pair)); return; }
        RecycleAfterCompletion();
    }
};

// Framework service state for one recording job. Workers produce query ranges
// and CPU samples locally. The submission owner publishes them after success.
// A late batch is different. It belongs to an external epoch's pass that is recorded at execution (atCompletion), so
// it is recorded after the submission. Its publication is left to its frame slot's completion
// (IStatisticsService::PublishAtCompletion), once the execution that records it has run on the GPU. The publication
// is dropped when the execution did not record it, because the execution was signalled unexecuted.
struct OwnedRecordingStatistics : std::enable_shared_from_this<OwnedRecordingStatistics> {
    std::shared_ptr<runtime::IStatisticsService> service;
    unsigned frameIndex = 0;
    rhi::QueueKind queueKind = rhi::QueueKind::Graphics;
    bool gpuQueries = true;
    bool atCompletion = false;
    std::atomic<bool> recorded{false};  // a late batch's: set (release) by each recording, taken (acquire) by its publication
    std::vector<int> passIndices;
    std::vector<double> cpuMilliseconds;
    runtime::QueryRecordingContext queries;

    void Publish() {
        if (!atCompletion) return PublishRecorded();
        // Taken by this completion: a kept recording's late batch is recorded, and published, once per execution.
        service->PublishAtCompletion(frameIndex, [self = shared_from_this()] {
            if (self->recorded.exchange(false, std::memory_order_acq_rel)) self->PublishRecorded();
        });
    }

private:
    void PublishRecorded() {
        if (gpuQueries) service->MergePendingResolves(queueKind, frameIndex, queries);
        for (size_t i = 0; i < passIndices.size(); ++i)
            if (passIndices[i] >= 0)
                service->RecordCpuExecuteTime(static_cast<unsigned>(passIndices[i]), cpuMilliseconds.at(i));
    }
};

// A profiler's GPU range around each pass of a list (RenderGraph::SetGpuPassRangeCallbacks), named by the pass.
struct OwnedRecordingGpuRanges {
    std::function<void(rhi::CommandList, rhi::Queue, const char* queueName, const char* passName)> begin;
    std::function<void(rhi::CommandList, rhi::Queue)> end;
    const char* queueName = "Graphics";
};

// A range name for the barriers recorded around a pass ("cs.org.barrier.<where>.<pass>"), interned for the process.
// The pass ranges cover only a pass's own commands. So a profiler sees a list's barrier batches only through these.
inline const char* OwnedRecordingBarrierRangeName(std::string_view where, std::string_view pass) {
    static std::mutex mutex;
    static std::set<std::string, std::less<>> names;
    // Built in a buffer kept per thread and looked up by view: a name seen before allocates nothing.
    thread_local std::string name;
    name.assign("cs.org.barrier.");
    name.append(where);
    if (!pass.empty()) { name.push_back('.'); name.append(pass); }
    std::lock_guard lock(mutex);
    if (const auto found = names.find(std::string_view(name)); found != names.end()) return found->c_str();
    return names.insert(name).first->c_str();
}

struct OwnedRecordingList {
    OwnedRecordingList() = default;
    OwnedRecordingList(OwnedRecordingList&&) = default;
    OwnedRecordingList& operator=(OwnedRecordingList&&) = default;
    OwnedRecordingList(const OwnedRecordingList&) = delete;
    OwnedRecordingList& operator=(const OwnedRecordingList&) = delete;
    std::shared_ptr<const FrozenExecutionBindings> bindings;
    // Exact selected version, rooted through CPU recording and GPU retirement.
    // Persistent-only packets need no legacy frozen leaf table.
    std::shared_ptr<const persistent::SelectedPublication> publication;
    std::shared_ptr<const std::vector<ExternalDescriptorBindingValue>> externalBindings;
    // Keep invocation lifecycle ownership until CPU recording and batch
    // retirement finish, even if the caller drops its sealed frame handle.
    std::shared_ptr<const RenderFrameSnapshot> frame;
    // Admission-captured defaults for passes using directly-indexed root
    // signatures. Individual passes may rebind a compatible snapshot.
    rhi::DescriptorHeapHandle resourceDescriptorHeap{};
    rhi::DescriptorHeapHandle samplerDescriptorHeap{};
    std::vector<rhi::TextureBarrier> textureBarriers;
    std::vector<rhi::BufferBarrier> bufferBarriers;
    std::vector<rhi::TextureBarrier> acquiredBarriers;  // PreparedBatchBarriers::acquired
    std::vector<PreparedBatchBarriers::BeforePass> barriersBeforePass;
    std::vector<PreparedBatchBarriers::BeforePass> barriersAfterPass;
    std::vector<PreparedPass> passes;
    std::shared_ptr<OwnedRecordingStatistics> statistics;
    std::shared_ptr<const OwnedRecordingGpuRanges> gpuPassRanges;
    std::shared_ptr<FrameCommandAllocation> allocation = std::make_shared<FrameCommandAllocation>();
    // What the passes' commands name at record time (RecordingContext::Retain): released with the list's lease, after
    // its last submission retires.
    std::shared_ptr<RecordingRetention> retention = std::make_shared<RecordingRetention>();
    // RecordingContext::Run for the passes of this list.
    RunRef run{};
    // RenderGraph::ExternalQueueBoundary: full memory barrier before the first
    // pass / after the last pass of this command list.
    bool externalEntryBarrier = false;
    bool externalExitBarrier = false;
    // Set for an external epoch (RenderGraph::SetExternalPersistentEpochs). The list is recorded into a secondary list
    // from its pool, or at execution into the host's list (a late batch). The graph never submits it. It has no Tracy
    // GPU zones, because their state is queue-scoped.
    bool external = false;
    // A late batch of an epoch that is not external, under RenderGraph::SetInlineAtExecution. Its packet records it at
    // submission, on the submitting thread, into a primary list from allocation->pool, and submits that list in the
    // batch's place. It has no Tracy GPU zones: the submitting thread records it while the recording workers may record
    // other lists for the same queue.
    bool inlineAtExecution = false;
    // With inlineAtExecution, under RenderGraph::SetKeptGatedInline: a replayable packet whose passes are all keyed by their
    // skip answers (PreparedPass::KeyedBySkip) submits its list again, without recording, while the answers are unchanged.
    bool keepBySkips = false;
    // Record for resubmission, when every pass of the packet is replayable (PreparedRhiExecutionBatch's replayable
    // packets). The list then has no profiler GPU zones or ranges, because a resubmission would write their queries
    // again. Nothing a list records may name a resource its recording does not own. Its statistics queries, if any, are
    // its own (QueryRecordingContext::owned).
    bool replayable = false;
};

} // namespace org::compiler

// Diagnostic switches for timing runs: they leave out parts of every recording, unsafe for ordinary rendering.
namespace org::diagnostics {

// Diagnostics. While set, recordings skip the descriptor heap bind at the start of a list that holds only latched
// copies. Those copies read no descriptor. Use it only to time the cost of the bind.
inline std::atomic<bool>& DiagnosticSkipListHeapBinds() {
    static std::atomic<bool> skip{false};
    return skip;
}

// Diagnostics. While set, recordings leave out:
// - 1: their full external entry and exit barriers, including the closed execution's,
// - 2: every barrier batch.
// Unsafe. Use it only for timing runs whose lists access nothing.
inline std::atomic<int>& DiagnosticSkipListBarriers() {
    static std::atomic<int> skip{0};
    return skip;
}

// Diagnostics: lists recorded while set have no pass timestamp queries (nor their resets and resolve) and no Tracy GPU zones.
inline std::atomic<bool>& DiagnosticSkipPassQueries() {
    static std::atomic<bool> skip{false};
    return skip;
}

// Diagnostics: lists built while set keep the barriers of the passes their host turned off ahead where the plan put them
// (FoldSkippedPassBarriers is not applied). For A/B timing runs.
inline std::atomic<bool>& DiagnosticKeepSkippedPassBarriers() {
    static std::atomic<bool> keep{false};
    return keep;
}

// Barriers a late batch left out at execution: those of passes that skipped the run (PreparedPass::SkipsAtExecution),
// folded into the next pass that ran (FoldBarriersOfSkipped). A running count; readers take differences.
inline std::atomic<uint64_t>& FoldedAtExecutionBarriers() {
    static std::atomic<uint64_t> count{0};
    return count;
}

// Entry barriers (and releases) an execution's submission left out because no running pass touched their resource
// (compiler::UntouchedResources). A running count; readers take differences.
inline std::atomic<uint64_t>& UntouchedEntryBarriers() {
    static std::atomic<uint64_t> count{0};
    return count;
}

} // namespace org::diagnostics

namespace org::compiler {

// Folds the barriers of the passes skipped[i] marks into the next pass that runs, writing the result to outBefore and
// outAfter (one batch per pass, as before and after hold them).
// A run of skipped passes does no work, so the transitions it plans on one subresource compose into one: from the state
// before the run to the state after it. They go into the before batch of the next pass that runs, composed with that
// pass's own where they name the same subresource. A run at the end of the list goes into the last pass's after batch.
// A composed transition between equal states is dropped. Each backing ends in the state the plan gives it, so the
// ledger's commits stay true.
// False, with the outputs unspecified, when a skipped pass's barrier transfers ownership, or when two barriers of one batch
// would name overlapping but different ranges of a resource (barriers of one batch are unordered against each other): the
// caller keeps the barriers as they are (DropUntouchedBarriers still applies). removed gets the barriers the fold took out.
// untouched: resources no running pass of the execution touches. Their barriers are left out, not composed: they stay in
// the state the execution found them in, and the ledger keeps that state (UntouchedResources).
inline bool FoldBarriersOfSkipped(const std::vector<PreparedBatchBarriers::BeforePass>& before,
    const std::vector<PreparedBatchBarriers::BeforePass>& after, const std::vector<uint8_t>& skipped,
    std::vector<PreparedBatchBarriers::BeforePass>& outBefore, std::vector<PreparedBatchBarriers::BeforePass>& outAfter, size_t& removed,
    const UntouchedResources* untouched = nullptr) {
    removed = 0;
    if (untouched && untouched->Empty()) untouched = nullptr;
    const size_t count = skipped.size();
    if (!count || before.size() != count || after.size() != count) return false;
    const auto sameHandle = [](rhi::ResourceHandle a, rhi::ResourceHandle b) { return a.index == b.index && a.generation == b.generation; };
    const auto sameRange = [](const rhi::TextureSubresourceRange& a, const rhi::TextureSubresourceRange& b) {
        return a.baseMip == b.baseMip && a.mipCount == b.mipCount && a.baseLayer == b.baseLayer && a.layerCount == b.layerCount
            && a.basePlane == b.basePlane && a.planeCount == b.planeCount;
    };
    const auto overlaps = [](uint32_t a, uint32_t an, uint32_t b, uint32_t bn) {
        const uint64_t ae = an == ~0u ? ~0ull : uint64_t{a} + an, be = bn == ~0u ? ~0ull : uint64_t{b} + bn;
        return a < be && b < ae;
    };
    const auto rangesOverlap = [&](const rhi::TextureSubresourceRange& a, const rhi::TextureSubresourceRange& b) {
        return overlaps(a.baseMip, a.mipCount, b.baseMip, b.mipCount) && overlaps(a.baseLayer, a.layerCount, b.baseLayer, b.layerCount)
            && overlaps(a.basePlane, a.planeCount, b.basePlane, b.planeCount);
    };
    const auto transfers = [](const auto& barrier) {
        return barrier.queueOwnership != rhi::QueueOwnership::None
            || barrier.externalOwnership != std::remove_cvref_t<decltype(barrier)>::ExternalOwnership::None;
    };
    using Batch = PreparedBatchBarriers::BeforePass;
    // Folds a_barriers into a_into, in order. False when one cannot be folded.
    const auto fold = [&](Batch& into, const Batch& barriers) {
        for (const auto& barrier : barriers.textures) {
            if (untouched && untouched->Holds(barrier.texture)) continue;
            bool placed = false;
            for (auto& held : into.textures) {
                if (!sameHandle(held.texture, barrier.texture)) continue;
                if (sameRange(held.range, barrier.range) && !transfers(held) && !transfers(barrier)) {
                    held.afterSync = barrier.afterSync;
                    held.afterAccess = barrier.afterAccess;
                    held.afterLayout = barrier.afterLayout;
                    placed = true;
                    break;
                }
                if (rangesOverlap(held.range, barrier.range)) return false;
            }
            if (!placed) into.textures.push_back(barrier);
        }
        for (const auto& barrier : barriers.buffers) {
            if (untouched && untouched->Holds(barrier.buffer)) continue;
            bool placed = false;
            for (auto& held : into.buffers) {
                if (!sameHandle(held.buffer, barrier.buffer)) continue;
                if (held.offset == barrier.offset && held.size == barrier.size && !transfers(held) && !transfers(barrier)) {
                    held.afterSync = barrier.afterSync;
                    held.afterAccess = barrier.afterAccess;
                    placed = true;
                    break;
                }
                const uint64_t heldEnd = held.size == ~0ull ? ~0ull : held.offset + held.size;
                const uint64_t end = barrier.size == ~0ull ? ~0ull : barrier.offset + barrier.size;
                if (held.offset < end && barrier.offset < heldEnd) return false;
            }
            if (!placed) into.buffers.push_back(barrier);
        }
        return true;
    };
    // The outputs keep their batches' storage, and the pending run is a per-thread buffer: a caller that reuses its outputs
    // allocates nothing once they have grown.
    const auto clear = [](Batch& batch) {
        batch.textures.clear();
        batch.buffers.clear();
    };
    const auto copy = [&](Batch& into, const Batch& from) {
        into.textures.assign(from.textures.begin(), from.textures.end());
        into.buffers.assign(from.buffers.begin(), from.buffers.end());
        if (untouched) {
            std::erase_if(into.textures, [&](const rhi::TextureBarrier& b) { return untouched->Holds(b.texture); });
            std::erase_if(into.buffers, [&](const rhi::BufferBarrier& b) { return untouched->Holds(b.buffer); });
        }
    };
    outBefore.resize(count);
    outAfter.resize(count);
    for (size_t i = 0; i < count; ++i) {
        clear(outBefore[i]);
        clear(outAfter[i]);
    }
    thread_local Batch pending;
    clear(pending);
    for (size_t i = 0; i < count; ++i) {
        if (skipped[i]) {
            if (!fold(pending, before[i]) || !fold(pending, after[i])) return false;
            continue;
        }
        if (!fold(pending, before[i])) return false;
        copy(outBefore[i], pending);
        clear(pending);
        copy(outAfter[i], after[i]);
    }
    // The list ends in skipped passes: the last one's after batch holds the run.
    if (!pending.textures.empty() || !pending.buffers.empty()) copy(outAfter[count - 1], pending);
    // A composed transition between equal states orders nothing.
    const auto drop = [&](Batch& batch) {
        std::erase_if(batch.textures, [&](const rhi::TextureBarrier& b) {
            return !transfers(b) && b.beforeLayout == b.afterLayout && !org::NeedsWholeBufferBarrier(b.beforeAccess, b.afterAccess, b.beforeSync, b.afterSync);
        });
        std::erase_if(batch.buffers, [&](const rhi::BufferBarrier& b) {
            return !transfers(b) && !org::NeedsWholeBufferBarrier(b.beforeAccess, b.afterAccess, b.beforeSync, b.afterSync);
        });
    };
    size_t was = 0, now = 0;
    for (size_t i = 0; i < count; ++i) {
        was += before[i].textures.size() + before[i].buffers.size() + after[i].textures.size() + after[i].buffers.size();
        drop(outBefore[i]);
        drop(outAfter[i]);
        now += outBefore[i].textures.size() + outBefore[i].buffers.size() + outAfter[i].textures.size() + outAfter[i].buffers.size();
    }
    removed = was > now ? was - now : 0;
    return true;
}

// Leaves out every barrier on a resource no running pass touches (UntouchedResources). Returns the barriers removed.
inline size_t DropUntouchedBarriers(std::vector<PreparedBatchBarriers::BeforePass>& batches, const UntouchedResources& untouched) {
    if (untouched.Empty()) return 0;
    size_t removed = 0;
    for (auto& batch : batches) {
        removed += std::erase_if(batch.textures, [&](const rhi::TextureBarrier& b) { return untouched.Holds(b.texture); });
        removed += std::erase_if(batch.buffers, [&](const rhi::BufferBarrier& b) { return untouched.Holds(b.buffer); });
    }
    return removed;
}

// Folds the barriers of the passes its host turned off ahead (PreparedPass::SkippedAhead) into the next pass that runs
// (FoldBarriersOfSkipped). ahead: the resources every use of which in the execution was skipped ahead. Their barriers go
// (UntouchedResources), even where the rest cannot be folded. Returns the barriers removed.
inline size_t FoldSkippedPassBarriers(OwnedRecordingList& recording, const UntouchedResources* ahead = nullptr) {
    const size_t count = recording.passes.size();
    if (!count || recording.barriersBeforePass.size() != count || recording.barriersAfterPass.size() != count) return 0;
    if (ahead && ahead->Empty()) ahead = nullptr;
    if (std::none_of(recording.passes.begin(), recording.passes.end(), [](const PreparedPass& p) { return p.SkippedAhead(); })) return 0;
    std::vector<uint8_t> skipped(count);
    for (size_t i = 0; i < count; ++i) skipped[i] = recording.passes[i].SkippedAhead() ? 1 : 0;
    std::vector<PreparedBatchBarriers::BeforePass> before, after;
    size_t removed = 0;
    if (!FoldBarriersOfSkipped(recording.barriersBeforePass, recording.barriersAfterPass, skipped, before, after, removed, ahead)) {
        if (!ahead) return 0;
        return DropUntouchedBarriers(recording.barriersBeforePass, *ahead) + DropUntouchedBarriers(recording.barriersAfterPass, *ahead);
    }
    recording.barriersBeforePass = std::move(before);
    recording.barriersAfterPass = std::move(after);
    return removed;
}

// ahead: the resources every use of which in the execution is skipped ahead (FindUntouched over the passes' SkippedAhead). The
// list leaves out their barriers, and every execution of it must leave them unmoved (FrameAdmission::unmoved).
inline OwnedRecordingList BuildPersistentRecordingList(std::shared_ptr<const RenderFrameSnapshot> sealed, uint32_t batch,
    const UntouchedResources* ahead = nullptr) {
    if (!sealed) throw std::invalid_argument("Missing persistent frame seal");
    const auto& frame = *sealed;
    if (!frame.publication || !frame.barrierPlan
        || frame.layout != frame.publication->executable->executionLayout)
        throw std::invalid_argument("Recording list requires an exact persistent frame seal");
    const auto& graph = *frame.publication->executable->graph;
    const auto& compiled = graph.batches.at(batch);
    const auto& barriers = frame.barrierPlan->batches.at(batch);
    OwnedRecordingList result;
    result.frame = std::move(sealed);
    result.publication = frame.publication;
    result.textureBarriers = barriers.textures;
    result.bufferBarriers = barriers.buffers;
    result.acquiredBarriers = barriers.acquired;
    result.barriersBeforePass = barriers.beforePass;
    result.barriersAfterPass = barriers.afterPass;
    for (auto pass : compiled.passes) {
        const auto& invocation = frame.passes.at(graph.structure->passes.at(pass).preparedPassIndex);
        if (!invocation || invocation.IsConsumed() || !invocation.IsWorkerSafe())
            throw std::invalid_argument("Persistent recording list requires fresh worker-safe invocations");
        result.passes.push_back(invocation);
    }
    if (!diagnostics::DiagnosticKeepSkippedPassBarriers().load(std::memory_order_relaxed)) {
        const auto folded = FoldSkippedPassBarriers(result, ahead);
        if (folded) basic_telemetry::AddCounter("ORG.Execution.FoldedSkippedBarriers", static_cast<int64_t>(folded));
    }
    return result;
}

// The recording context of one owned recording, on `list`: its bindings (legacy, persistent, or both) and frame slot.
inline RecordingContext MakeOwnedRecordingContext(rhi::CommandList list, const OwnedRecordingList& recording) {
    auto context = recording.bindings
        ? RecordingContext(list, recording.bindings, recording.externalBindings)
        : RecordingContext::FromPersistentBindings(list, recording.publication);
    if (recording.bindings && recording.publication)
        context = context.WithPersistentBindings(recording.publication);
    context.SetRun(recording.run);
    context.SetRetention(recording.retention);
    context.SetDescriptorHeaps(recording.resourceDescriptorHeap, recording.samplerDescriptorHeap);
    return context;
}

// Records one owned recording's commands into the context's list:
// - its heaps,
// - its entry, per-pass and exit barriers,
// - its passes,
// - the resolve of its queries.
// It does not begin or end the list. RecordPreparedRhiExecutionBatch does that for its own lists. A late batch of an
// external epoch records into the host's list (PreparedRhiExecutionBatch::RecordDeferred), and the host ends it.
// a_again records a replayable recording's passes again (PreparedPass::RecordAgain). This is for a late batch of a
// kept recording. a_atExecution: a late batch, recorded after its run's commit, whose passes may skip the run
// (PreparedPass::SkipsAtExecution). Returns the passes recorded, not counting the skipped ones.
// a_skips: with a_atExecution, the passes' answers, asked by the submission already (empty: asked here). a_untouched: with
// a_atExecution, the resources no running pass of the execution touches; their barriers are left out (UntouchedResources).
inline size_t RecordOwnedRecordingCommands(RecordingContext& context, const OwnedRecordingList& recording, rhi::Queue queue, bool replayable,
    bool a_again = false, bool a_atExecution = false, std::span<const uint8_t> a_skips = {}, const UntouchedResources* a_untouched = nullptr) {
    struct TracyGpuZoneScope {
        rhi::CommandList& commands;
        bool open = false;
        TracyGpuZoneScope(rhi::CommandList& commandList, const rhi::Queue& queue, const char* name)
            : commands(commandList) {
            BT_ZONE_SCOPE("ORG.Execution.RecordOwned.TracyBegin");
            const auto result = commands.BeginTracyGpuZone(queue, name);
            open = result == rhi::Result::Ok;
            if (open) basic_telemetry::AddCounter("ORG.TracyGpuZones.BeginAccepted");
            else if (result != rhi::Result::Unsupported)
                basic_telemetry::AddCounter("ORG.TracyGpuZones.BeginFailures");
        }
        TracyGpuZoneScope(const TracyGpuZoneScope&) = delete;
        TracyGpuZoneScope& operator=(const TracyGpuZoneScope&) = delete;
        ~TracyGpuZoneScope() { Close(); }
        void Close() noexcept {
            if (!open) return;
            BT_ZONE_SCOPE("ORG.Execution.RecordOwned.TracyEnd");
            commands.EndTracyGpuZone();
            open = false;
        }
    };
    size_t recordedPasses = 0;
    auto* statistics = recording.statistics.get();
    if (statistics) statistics->cpuMilliseconds.resize(recording.passes.size());
    const bool skipQueries = diagnostics::DiagnosticSkipPassQueries().load(std::memory_order_relaxed);
    if (skipQueries && statistics) statistics->gpuQueries = false;
    // The diagnostic skips it only for lists of copies alone (the latched copies'), which read no descriptor.
    const bool copiesOnly = std::all_of(recording.passes.begin(), recording.passes.end(),
        [](const PreparedPass& p) { return p.DebugName().ends_with("latched-copies"); });
    if (recording.resourceDescriptorHeap.valid() && !(copiesOnly && diagnostics::DiagnosticSkipListHeapBinds().load(std::memory_order_relaxed))) {
        context.Commands().SetDescriptorHeaps(recording.resourceDescriptorHeap,
            recording.samplerDescriptorHeap.valid()
                ? std::optional<rhi::DescriptorHeapHandle>{recording.samplerDescriptorHeap}
                : std::nullopt);
    }
    const auto* barrierRanges = recording.gpuPassRanges.get();
    const auto firstPass = recording.passes.empty() ? std::string_view{} : std::string_view(recording.passes.front().DebugName());
    // A kept recording's own queries (OwnedTimestampQueries) time its barrier batches too, under the profiler's range names.
    auto* const ownedQueries = statistics && statistics->gpuQueries && statistics->queries.owned ? &statistics->queries : nullptr;
    // BasicRHI's command census (rhi::census) labels the list with its first pass.
    if (rhi::census::Enabled()) rhi::census::SetLabel(firstPass.empty() ? "<unnamed>" : std::string(firstPass).c_str());
    // List mode: one range for the whole list, from before its entry barriers to after its exit barrier (the resolve).
    if (ownedQueries && ownedQueries->owned->listOnly) {
        thread_local std::string listName;
        listName.assign("list: ").append(firstPass.empty() ? std::string_view("<unnamed>") : firstPass);
        // With BasicRHI's submission trace on, the name carries the list's handle, so a reader can match the range to its submission.
        if (rhi::submittrace::Enabled()) {
            const auto handle = context.Commands().GetHandle();
            listName.append(" #").append(std::to_string(uint64_t{handle.index} | (uint64_t{handle.generation} << 32)));
        }
        statistics->service->BeginListRangeQuery(listName, context.Commands(), *ownedQueries);
    }
    const int skipBarriers = diagnostics::DiagnosticSkipListBarriers().load(std::memory_order_relaxed);
    const auto recordBarriers = [&](const char* where, std::string_view pass, const rhi::BarrierBatch& barriers) {
        if (skipBarriers >= 2 || (skipBarriers == 1 && barriers.globals.size != 0))
            return;
        const char* name = barrierRanges || ownedQueries ? OwnedRecordingBarrierRangeName(where, pass) : nullptr;
        if (barrierRanges) barrierRanges->begin(context.Commands(), queue, barrierRanges->queueName, name);
        if (ownedQueries) statistics->service->BeginRangeQuery(name, context.Commands(), *ownedQueries);
        context.Commands().Barriers(barriers);
        if (ownedQueries) statistics->service->EndRangeQuery(context.Commands(), *ownedQueries);
        if (barrierRanges) barrierRanges->end(context.Commands(), queue);
    };
    if (recording.externalEntryBarrier) {
        auto full = rhi::FullMemoryBarrier();
        rhi::BarrierBatch barriers{};
        barriers.globals = {&full, 1u};
        recordBarriers("external-entry", firstPass, barriers);
    }
    if (!recording.textureBarriers.empty() || !recording.bufferBarriers.empty()) {
        rhi::BarrierBatch barriers{};
        barriers.textures = {recording.textureBarriers.data(), static_cast<uint32_t>(recording.textureBarriers.size())};
        barriers.buffers = {recording.bufferBarriers.data(), static_cast<uint32_t>(recording.bufferBarriers.size())};
        recordBarriers("entry", firstPass, barriers);
    }
    if (!recording.acquiredBarriers.empty()) {
        rhi::BarrierBatch barriers{};
        barriers.textures = {recording.acquiredBarriers.data(), static_cast<uint32_t>(recording.acquiredBarriers.size())};
        recordBarriers("entry", firstPass, barriers);
    }
    // A late batch asks each pass first whether it skips this run. The barriers of the passes that skip fold into the next
    // pass that runs, as a list built with passes skipped ahead does (FoldSkippedPassBarriers): a pass with nothing to do
    // costs no barrier batch of its own. Every backing still ends in the state the plan gives it.
    // Kept per thread, so a late batch allocates nothing here once they have grown. A recording recorded inside another one's pass
    // on the same thread (none does today) takes its own.
    struct FoldScratch {
        std::vector<uint8_t> skips;
        std::vector<PreparedBatchBarriers::BeforePass> before, after;
        bool busy = false;
    };
    thread_local FoldScratch kept;
    FoldScratch own;
    FoldScratch& scratch = kept.busy ? own : kept;
    scratch.busy = true;
    struct Release {
        FoldScratch& scratch;
        ~Release() { scratch.busy = false; }
    } release{scratch};
    auto& skips = scratch.skips;
    auto& foldedBefore = scratch.before;
    auto& foldedAfter = scratch.after;
    skips.clear();
    const auto* beforeBatches = &recording.barriersBeforePass;
    const auto* afterBatches = &recording.barriersAfterPass;
    if (a_atExecution) {
        bool late = false;
        if (a_skips.size() == recording.passes.size()) {
            skips.assign(a_skips.begin(), a_skips.end());
            for (size_t i = 0; i < recording.passes.size(); ++i) late = late || (skips[i] && !recording.passes[i].SkippedAhead());
        } else {
            skips.assign(recording.passes.size(), 0);
            for (size_t i = 0; i < recording.passes.size(); ++i) {
                const auto& pass = recording.passes[i];
                const bool ahead = pass.SkippedAhead();
                skips[i] = static_cast<uint8_t>(ahead || (a_again ? pass.SkipsAgain(context) : pass.SkipsAtExecution(context)));
                late = late || (skips[i] && !ahead);
            }
        }
        const bool keep = diagnostics::DiagnosticKeepSkippedPassBarriers().load(std::memory_order_relaxed);
        const bool untouched = a_untouched && !a_untouched->Empty();
        size_t removed = 0;
        if ((late || untouched) && !keep) {
            if (FoldBarriersOfSkipped(recording.barriersBeforePass, recording.barriersAfterPass, skips, foldedBefore, foldedAfter, removed,
                    a_untouched)) {
                beforeBatches = &foldedBefore;
                afterBatches = &foldedAfter;
            } else if (untouched) {
                // Not folded: the barriers stay where the plan put them, but those on the untouched resources go all the same.
                foldedBefore = recording.barriersBeforePass;
                foldedAfter = recording.barriersAfterPass;
                removed = DropUntouchedBarriers(foldedBefore, *a_untouched) + DropUntouchedBarriers(foldedAfter, *a_untouched);
                beforeBatches = &foldedBefore;
                afterBatches = &foldedAfter;
            }
            if (removed) diagnostics::FoldedAtExecutionBarriers().fetch_add(removed, std::memory_order_relaxed);
        }
    }
    const auto recordPassBarriers = [&](const std::vector<PreparedBatchBarriers::BeforePass>& all, size_t passIndex, const char* where,
        std::string_view pass) {
        if (all.empty()) return;
        const auto& batch = all[passIndex];
        if (batch.textures.empty() && batch.buffers.empty()) return;
        rhi::BarrierBatch barriers{};
        barriers.textures = {batch.textures.data(), static_cast<uint32_t>(batch.textures.size())};
        barriers.buffers = {batch.buffers.data(), static_cast<uint32_t>(batch.buffers.size())};
        recordBarriers(where, pass, barriers);
    };
    for (size_t passIndex = 0; passIndex < recording.passes.size(); ++passIndex) {
        // Recorded at execution, a pass with nothing to do this run (PreparedPass::SkipsAtExecution) records only its barriers:
        // the states the admission planned still change. No label, timestamps or GPU zone for it. A kept recording's pass
        // recorded again asks the same for its new run (SkipsAgain).
        // A pass its host turned off for this execution (PreparedPass::SkippedAhead) is skipped the same way, recorded ahead or not.
        const auto& pass = recording.passes[passIndex];
        if (a_atExecution ? skips[passIndex] != 0 : pass.SkippedAhead()) {
            BT_ZONE_SCOPE("ORG.Execution.RecordOwned.Skipped");
            const auto passLabel = pass.DebugName();
            BT_ZONE_TEXT(passLabel.data(), passLabel.size());
            recordPassBarriers(*beforeBatches, passIndex, "before", passLabel);
            if (!a_again) pass.RecordNothing();
            recordPassBarriers(*afterBatches, passIndex, "after", passLabel);
            continue;
        }
        BT_ZONE_SCOPE("ORG.Execution.RecordOwned.Pass");
        ++recordedPasses;
        // A capture tool's region for the pass (PIX / VK_EXT_debug_utils), opened before the pass's entry
        // barriers so that a wait on the previous pass's output shows up inside the pass that needed it.
        // The legacy recorder labels its passes; without this, persistent graphs recorded none at all.
        const auto passLabel = recording.passes[passIndex].DebugName();
        BT_ZONE_TEXT(passLabel.data(), passLabel.size());
        rhi::debug::Scope captureScope(context.Commands(), rhi::colors::Mint, passLabel.empty() ? "<unnamed>" : passLabel.data());
        if (!beforeBatches->empty()) {
            BT_ZONE_SCOPE("ORG.Execution.RecordOwned.BeforeBarriers");
            const auto& before = (*beforeBatches)[passIndex];
            if (!before.textures.empty() || !before.buffers.empty()) {
                rhi::BarrierBatch barriers{};
                barriers.textures = {before.textures.data(), static_cast<uint32_t>(before.textures.size())};
                barriers.buffers = {before.buffers.data(), static_cast<uint32_t>(before.buffers.size())};
                recordBarriers("before", passLabel, barriers);
            }
        }
        const int statisticsIndex = statistics ? statistics->passIndices[passIndex] : -1;
        const auto debugName = recording.passes[passIndex].DebugName();
        // Copy batches are recorded concurrently and already have explicit
        // CPU/statistics attribution. Injecting a Tracy timestamp pair and
        // ResolveQueryData for every tiny copy pass makes the profiler's
        // queue-wide query/readback state another concurrent writer. DX12
        // can report that earlier instrumentation error only when a later
        // command list closes, obscuring the actual pass. Keep GPU zones on
        // graphics/compute queues and leave copy lists instrumentation-free.
        const bool allowTracyGpuZone = !replayable && !skipQueries && !recording.external && !recording.inlineAtExecution && (!statistics ||
            statistics->queueKind != rhi::QueueKind::Copy);
        TracyGpuZoneScope tracyGpuZone(context.Commands(), queue,
            allowTracyGpuZone
                ? (debugName.empty() ? "<unnamed>" : debugName.data())
                : nullptr);
        if (statisticsIndex >= 0 && statistics->gpuQueries)
            statistics->service->BeginQuery(static_cast<unsigned>(statisticsIndex), statistics->frameIndex,
                queue, context.Commands(), statistics->queries);
        const auto cpuStart = statisticsIndex >= 0 ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        // A replayable list carries its ranges too. The callbacks push a range only while a capture is armed. A host
        // that keeps lists across frames records them again when a capture arms and when it ends. So no range outlives
        // its session.
        const auto* ranges = recording.gpuPassRanges.get();
        if (ranges) ranges->begin(context.Commands(), queue, ranges->queueName, debugName.empty() ? "<unnamed>" : debugName.data());
        {
            BT_ZONE_SCOPE("ORG.Execution.RecordOwned.Record");
            BT_ZONE_TEXT(passLabel.data(), passLabel.size());
            // The pass may time parts of itself (BeginRecordingSubRange) under its own range, with the recording's queries.
            // ...and profile them under the profiler's ranges.
            runtime::RecordingRanges subRanges{ ownedQueries ? statistics->service.get() : nullptr, ownedQueries, &context.Commands() };
            if (ranges) {
                subRanges.gpuRangeBegin = &ranges->begin;
                subRanges.gpuRangeEnd = &ranges->end;
                subRanges.queue = queue;
                subRanges.queueName = ranges->queueName;
                subRanges.passName = debugName.empty() ? "<unnamed>" : debugName.data();
            }
            runtime::RecordingRangeScope subRangeScope(ownedQueries || ranges ? &subRanges : nullptr);
            if (a_again) recording.passes[passIndex].RecordAgain(context);
            else recording.passes[passIndex].Record(context);
        }
        if (ranges) ranges->end(context.Commands(), queue);
        if (statisticsIndex >= 0) {
            statistics->cpuMilliseconds[passIndex] = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - cpuStart).count();
            if (statistics->gpuQueries)
                statistics->service->EndQuery(static_cast<unsigned>(statisticsIndex), statistics->frameIndex,
                    queue, context.Commands(), statistics->queries);
        }
        if (!afterBatches->empty()) {
            BT_ZONE_SCOPE("ORG.Execution.RecordOwned.AfterBarriers");
            const auto& after = (*afterBatches)[passIndex];
            if (!after.textures.empty() || !after.buffers.empty()) {
                rhi::BarrierBatch barriers{};
                barriers.textures = {after.textures.data(), static_cast<uint32_t>(after.textures.size())};
                barriers.buffers = {after.buffers.data(), static_cast<uint32_t>(after.buffers.size())};
                recordBarriers("after", passLabel, barriers);
            }
        }
        tracyGpuZone.Close();
    }
    if (recording.externalExitBarrier) {
        auto full = rhi::FullMemoryBarrier();
        rhi::BarrierBatch barriers{};
        barriers.globals = {&full, 1u};
        recordBarriers("external-exit", firstPass, barriers);
    }
    // After the exit barrier, so a kept recording's own queries time it too.
    if (statistics && statistics->gpuQueries)
        statistics->service->ResolveQueries(statistics->frameIndex, queue, context.Commands(), statistics->queries);
    return recordedPasses;
}

inline PreparedRhiExecutionBatch::PreparedRhiExecutionBatch(uint32_t queueSlot, rhi::Queue queue, OwnedRecordingList deferred,
    std::vector<PreparedTimelineBinding> timelines, std::shared_ptr<const void> lease)
    : m_queueSlot(queueSlot), m_queue(queue), m_timelines(std::move(timelines)), m_lease(std::move(lease)),
      m_deferred(std::make_shared<OwnedRecordingList>(std::move(deferred))) {
    if (!m_queue || !m_lease || m_deferred->passes.empty())
        throw std::invalid_argument("Incomplete deferred RHI packet");
    m_inline = m_deferred->inlineAtExecution;
    if (m_inline && (!m_deferred->allocation || !m_deferred->allocation->pool || m_deferred->allocation->pool->Secondary()))
        throw std::invalid_argument("An inline late batch needs a pool of primary command lists");
    // A kept recording's late batch, whose passes are replayable. Every execution that takes the recording records it.
    // There are no effects to commit.
    m_replayable = m_deferred->replayable
        && std::all_of(m_deferred->passes.begin(), m_deferred->passes.end(), [](const PreparedPass& pass) { return pass.IsReplayable(); });
    if (!m_replayable) m_submissionEffects = m_deferred->passes;
    m_keyedBySkips = m_inline && m_replayable && m_deferred->keepBySkips
        && std::all_of(m_deferred->passes.begin(), m_deferred->passes.end(), [](const PreparedPass& pass) { return pass.KeyedBySkip(); });
}

inline void PreparedRhiExecutionBatch::AskSkips(std::vector<uint8_t>& a_out) const {
    a_out.clear();
    if (!m_deferred) return;
    const auto context = RecordingContext::ForSkipQuestions(m_deferred->run);
    const bool again = m_deferredRecorded.load(std::memory_order_relaxed);
    a_out.reserve(m_deferred->passes.size());
    for (const auto& pass : m_deferred->passes)
        a_out.push_back(static_cast<uint8_t>(pass.SkippedAhead() || (again ? pass.SkipsAgain(context) : pass.SkipsAtExecution(context))));
}

inline size_t PreparedRhiExecutionBatch::RecordDeferredCommands(rhi::CommandList a_list, std::span<const uint8_t> a_skips,
    const UntouchedResources* a_untouched) const {
    const bool again = m_deferredRecorded.exchange(true);
    if (again && !m_replayable) throw std::logic_error("A deferred packet is recorded once, unless it is replayable");
    // A kept recording's own queries start over in each list that records it. The last execution's queries were read
    // at its completion.
    if (auto* statistics = m_deferred->statistics.get(); again && statistics && statistics->queries.owned) {
        auto& owned = *statistics->queries.owned;
        owned.entries.clear();
        owned.open.clear();
        owned.resolved = false;
        owned.overflow = 0;
    }
    auto context = MakeOwnedRecordingContext(a_list, *m_deferred);
    const size_t recorded = RecordOwnedRecordingCommands(context, *m_deferred, m_queue, false, again, true, a_skips, a_untouched);
    // Its queries are in a_list now. Its frame slot's completion publishes them (OwnedRecordingStatistics::atCompletion).
    if (m_deferred->statistics) m_deferred->statistics->recorded.store(true, std::memory_order_release);
    return recorded;
}

inline void PreparedRhiExecutionBatch::RecordDeferred(rhi::CommandList a_list) const {
    if (!m_deferred) throw std::logic_error("Not a deferred packet");
    if (m_inline) throw std::logic_error("An inline late batch is recorded by its own submission");
    RecordDeferredCommands(a_list);
    if (!m_replayable && !CommitEffectsSubmitted(m_deferredSignal)) throw std::logic_error("A deferred packet's passes could not be committed as submitted");
}

inline rhi::CommandList PreparedRhiExecutionBatch::RecordInline(std::span<const uint8_t> a_skips, const UntouchedResources* a_untouched) const noexcept {
    BT_ZONE_SCOPE("ORG.Execution.RecordInline");
    try {
        auto& allocation = *m_deferred->allocation;
        auto& census = diagnostics::Census();
        // A batch keyed by its passes' skip answers: the list recorded for the same answers is submitted again as it is. Its
        // last submission is done (see below), and nothing else in it depends on the run.
        // The answers are asked before the passes are recorded: SkipsAtExecution only before a pass's first recording,
        // SkipsAgain after it.
        // The answers the submission asked, or asked here once. A kept list also depends on the untouched resources, whose
        // barriers it leaves out.
        std::vector<uint8_t> skips(a_skips.begin(), a_skips.end());
        if (skips.size() != m_deferred->passes.size()) AskSkips(skips);
        const uint64_t untouched = a_untouched ? a_untouched->signature : 0;
        if (m_keyedBySkips && m_keptList && allocation.closed && allocation.pool && allocation.pair.list) {
            if (skips == m_keptSkips && untouched == m_keptUntouched) {
                census.Add(diagnostics::SubmissionCensus::List::InlineKept);
                return allocation.pair.list.Get();
            }
        }
        m_keptList = false;
        // A new list at the first submission. A replayable packet records again at each one, into the same list: its last
        // submission is done by then (a list is never pending twice; GraphHost submits a kept recording's copy only once
        // that copy's last run completed), so Begin resets it.
        allocation.pool->Begin(allocation.pair);
        allocation.closed = false;
        // Recorded before: the list's last submission is done (above), so what that recording retained is no longer named
        // by pending work. It goes now; the new recording retains its own.
        if (m_deferredRecorded.load(std::memory_order_relaxed) && m_deferred->retention) m_deferred->retention->Release();
        auto list = allocation.pair.list.Get();
        const size_t recorded = RecordDeferredCommands(list, skips, a_untouched);
        census.Add(diagnostics::SubmissionCensus::List::Inline);
        if (!recorded) census.Add(diagnostics::SubmissionCensus::List::InlineBarriersOnly);
        if (census.On()) {
            // Why this one was recorded: its first pass not keyed by its skip answer, else the answers changed.
            std::string_view why = !m_replayable ? std::string_view("not replayable") : !m_deferred->keepBySkips ? std::string_view("keeping off") : std::string_view();
            if (why.empty())
                for (const auto& pass : m_deferred->passes)
                    if (!pass.KeyedBySkip()) {
                        why = pass.DebugName();
                        break;
                    }
            census.NoteInlineRecorded(why.empty() ? std::string_view(m_keptSkips.empty() ? "keyed: first recording" : "keyed: answers changed") : why);
        }
        const auto closed = [&] {
            BT_ZONE_SCOPE("ORG.Execution.RecordInline.End");
            return list.EndChecked();
        }();
        if (closed != rhi::Result::Ok) {
            basic_telemetry::AddCounter("ORG.Execution.RecordingCloseFailures");
            return {};
        }
        allocation.Close();
        if (m_keyedBySkips) {
            m_keptSkips = std::move(skips);
            m_keptUntouched = untouched;
            m_keptList = true;
        }
        return list;
    } catch (...) {
        basic_telemetry::AddCounter("ORG.Execution.InlineRecordingFailures");
        return {};
    }
}

inline void PreparedRhiExecutionBatch::RetireInlineList() const noexcept {
    auto& allocation = *m_deferred->allocation;
    // A list left open by a failed recording is discarded with the packet instead.
    if (allocation.closed) allocation.RecycleAfterCompletion();
}

// Called only after admission freezes bindings and barriers. It may run on a
// recording worker; no declaration/Setup callbacks or live registries are used.
// The runtime owner must own the queue/device and supplied timeline objects.
inline std::shared_ptr<const PreparedRhiExecutionBatch> RecordPreparedRhiExecutionBatch(
    uint32_t queueSlot, rhi::Queue queue, std::vector<OwnedRecordingList> recordings,
    std::vector<PreparedTimelineBinding> timelines, std::shared_ptr<const void> runtimeOwner) {
    BT_ZONE_SCOPE("ORG.Execution.RecordOwnedBatch");
    if (!runtimeOwner || recordings.empty()) throw std::invalid_argument("Missing recording ownership");
    struct Ownership {
        std::shared_ptr<const void> runtime;
        std::vector<OwnedRecordingList> recordings;
    };
    // Validate all packets before consuming any pass. Empty legacy preparation
    // must be selected into a synchronous route by the preparation owner.
    {
    BT_ZONE_SCOPE("ORG.Execution.RecordOwned.Validate");
    for (const auto& recording : recordings) {
        if ((!recording.bindings && !recording.publication) || !recording.allocation || !recording.allocation->pair.list
            || recording.passes.empty())
            throw std::invalid_argument("Incomplete owned recording packet");
        if (!recording.allocation->pair.list.Get().SupportsCheckedEnd())
            throw std::invalid_argument("Backend lacks checked command list close");
        if (!recording.barriersBeforePass.empty()
            && recording.barriersBeforePass.size() != recording.passes.size())
            throw std::invalid_argument("Prepared pass/barrier count mismatch");
        if (!recording.barriersAfterPass.empty()
            && recording.barriersAfterPass.size() != recording.passes.size())
            throw std::invalid_argument("Prepared post-pass/barrier count mismatch");
        if (recording.statistics && (!recording.statistics->service
            || recording.statistics->passIndices.size() != recording.passes.size()))
            throw std::invalid_argument("Prepared pass/statistics count mismatch");
        for (const auto& pass : recording.passes)
            if (!pass || pass.IsConsumed()) throw std::invalid_argument("Missing or consumed pass in owned recording batch");
    }
    }
    bool replayable = true;
    for (const auto& recording : recordings) {
        replayable &= recording.replayable;
        for (const auto& pass : recording.passes) replayable &= pass.IsReplayable();
    }
    auto ownership = std::make_shared<Ownership>(Ownership{std::move(runtimeOwner), std::move(recordings)});
    // A pool's list is begun here, on the recording thread, unless its acquirer began it already. Begin may trade the list
    // for another idle one, so the list handles are read only after it.
    // This thread records the batch's lists one after another, so each pool's lists share one allocator claim: one
    // command pool for the batch instead of one per list.
    {
        std::vector<std::pair<CommandListPool*, std::vector<CommandListPair*>>> groups;
        for (const auto& recording : ownership->recordings) {
            auto& allocation = *recording.allocation;
            if (!allocation.pool || allocation.pair.Recording()) continue;
            auto group = std::find_if(groups.begin(), groups.end(), [&](const auto& g) { return g.first == allocation.pool.get(); });
            if (group == groups.end()) group = groups.insert(groups.end(), { allocation.pool.get(), {} });
            group->second.push_back(&allocation.pair);
        }
        for (auto& [pool, pairs] : groups) pool->BeginGroup(pairs);
    }
    std::vector<rhi::CommandList> lists;
    std::vector<PreparedPass> submissionEffects;
    std::vector<std::shared_ptr<FrameCommandAllocation>> allocations;
    lists.reserve(ownership->recordings.size());
    allocations.reserve(ownership->recordings.size());
    for (const auto& recording : ownership->recordings) {
        BT_ZONE_SCOPE("ORG.Execution.RecordOwned.PacketEntries");
        lists.push_back(recording.allocation->pair.list.Get());
        allocations.push_back(recording.allocation);
        if (!replayable) submissionEffects.insert(submissionEffects.end(), recording.passes.begin(), recording.passes.end());
    }
    auto packet = std::make_shared<const PreparedRhiExecutionBatch>(queueSlot, queue,
        std::move(lists), std::move(timelines), ownership, std::move(submissionEffects),
        [allocations = std::move(allocations)]() mutable {
            for (const auto& allocation : allocations) allocation->RecycleAfterCompletion();
            allocations.clear();
        }, replayable);
    for (const auto& recording : ownership->recordings) {
        BT_ZONE_SCOPE("ORG.Execution.RecordOwned.List");
        auto list = recording.allocation->pair.list.Get();
        // A list from a secondary pool (an external epoch's) is handed out recycled but not begun. Begin it here, for
        // work outside any pass.
        if (recording.allocation->pool && recording.allocation->pool->Secondary()) {
            rhi::SecondaryInheritance outside{};
            outside.insideRendering = false;
            outside.debugName = "ORG external epoch";
            // The host's command buffers run it, and a replayable one runs again in later ones. Without simultaneous use a
            // driver may chain a secondary into the buffer that executes it (AMD patches its end to jump back there): a later
            // execution or recording then breaks the host's commands that follow the earlier one.
            outside.simultaneousUse = true;
            if (list.BeginSecondary(outside) != rhi::Result::Ok)
                throw std::runtime_error("Owned secondary command list could not begin");
        }
        auto context = MakeOwnedRecordingContext(list, recording);
        RecordOwnedRecordingCommands(context, recording, queue, replayable);
        const auto closeResult = [&] {
            BT_ZONE_SCOPE("ORG.Execution.RecordOwned.EndChecked");
            return context.Commands().EndChecked();
        }();
        if (closeResult != rhi::Result::Ok) {
            basic_telemetry::AddCounter("ORG.Execution.RecordingCloseFailures");
            std::string names;
            for (const auto& pass : recording.passes) {
                if (!names.empty()) names += ", ";
                names += pass.DebugName();
            }
            size_t beforeTextures = 0, beforeBuffers = 0, afterTextures = 0, afterBuffers = 0;
            for (const auto& barriers : recording.barriersBeforePass) {
                beforeTextures += barriers.textures.size();
                beforeBuffers += barriers.buffers.size();
            }
            for (const auto& barriers : recording.barriersAfterPass) {
                afterTextures += barriers.textures.size();
                afterBuffers += barriers.buffers.size();
            }
            throw std::runtime_error("Owned command list close failed before submission: queueSlot="
                + std::to_string(queueSlot) + " entryTextures=" + std::to_string(recording.textureBarriers.size())
                + " entryBuffers=" + std::to_string(recording.bufferBarriers.size())
                + " beforeTextures=" + std::to_string(beforeTextures)
                + " beforeBuffers=" + std::to_string(beforeBuffers)
                + " afterTextures=" + std::to_string(afterTextures)
                + " afterBuffers=" + std::to_string(afterBuffers) + " passes=" + names);
        }
        recording.allocation->Close();
    }
    return packet;
}

} // namespace org::compiler
