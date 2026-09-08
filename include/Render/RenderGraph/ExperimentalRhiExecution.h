#pragma once

#include "Render/RenderGraph/ExperimentalGraphCompiler.h"
#include "Render/RenderGraph/ExperimentalExecutionState.h"
#include "Render/PreparedPass.h"
#include <rhi.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <BasicTelemetry/Tracy.h>

namespace org::experimental {

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
    std::vector<PreparedBackingState> initialStates;
    std::vector<std::shared_ptr<const void>> leases;
};

// Immutable frame publication consumed by recording/admission. Fresh pass
// packets are deliberately separate from the reusable compiled layout.
struct RenderFrameSnapshot {
    uint64_t frameNumber = 0;
    uint32_t preparationSlot = 0;
    std::shared_ptr<const GraphExecutionLayout> layout;
    std::vector<PreparedPass> passes;
    std::shared_ptr<const FrozenExecutionBindings> bindings;
    std::vector<PreparedBackingState> initialStates;
    std::vector<std::vector<ExternalTimelinePoint>> externalWaitsByPreparedPass;
    std::shared_ptr<const PreparedExecutionBarrierPlan> barrierPlan;
    std::vector<std::shared_ptr<const void>> leases;
    std::shared_ptr<const RealizedResourceBundle> resources;
};

struct PreparedFramePayload final : IFramePayloadLifecycle {
    uint64_t frameNumber = 0;
    uint32_t preparationSlot = 0;
    std::vector<PreparedPass> passes;
    std::shared_ptr<const FrozenExecutionBindings> bindings;
    std::vector<PreparedBackingState> initialStates;
    std::vector<std::vector<ExternalTimelinePoint>> externalWaitsByPreparedPass;
    std::vector<std::shared_ptr<const void>> leases;
    std::shared_ptr<const RealizedResourceBundle> resources;
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
    BT_ZONE_SCOPE("ORG.AsyncExecution.BuildRecordingPlan");
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
    result->initialStates = std::move(initialStates);
    result->externalWaitsByPreparedPass = std::move(externalWaitsByPreparedPass);
    result->leases = std::move(leases);
    return result;
}

inline std::shared_ptr<const PreparedFramePayload> BuildPreparedFramePayloadWithInitialStates(
    uint64_t frameNumber, std::vector<PreparedPass> passes,
    std::shared_ptr<const RealizedResourceBundle> resources,
    std::vector<PreparedBackingState> initialStates,
    std::vector<std::vector<ExternalTimelinePoint>> externalWaitsByPreparedPass = {},
    uint32_t preparationSlot = 0) {
    if (!resources || !frameNumber || !resources->bindings || passes.empty() || initialStates.empty())
        throw std::invalid_argument("Incomplete realized frame payload");
    for (const auto& pass : passes)
        if (!pass) throw std::invalid_argument("Legacy pass prevents async frame preparation");
    auto result = std::make_shared<PreparedFramePayload>();
    result->frameNumber = frameNumber;
    result->preparationSlot = preparationSlot;
    result->passes = std::move(passes);
    result->bindings = resources->bindings;
    result->initialStates = std::move(initialStates);
    result->externalWaitsByPreparedPass = std::move(externalWaitsByPreparedPass);
    result->leases = resources->leases;
    result->resources = std::move(resources);
    return result;
}

inline std::shared_ptr<const PreparedFramePayload> BuildPreparedFramePayload(
    uint64_t frameNumber, std::vector<PreparedPass> passes,
    std::shared_ptr<const RealizedResourceBundle> resources,
    std::vector<std::vector<ExternalTimelinePoint>> externalWaitsByPreparedPass = {},
    uint32_t preparationSlot = 0) {
    if (!resources || !frameNumber || !resources->bindings || passes.empty()
        || resources->initialStates.empty())
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
    return result;
}

inline std::shared_ptr<const RenderFrameSnapshot> BuildRenderFrameSnapshot(
    uint64_t frameNumber,
    std::shared_ptr<const GraphExecutionLayout> layout,
    std::vector<PreparedPass> passes,
    std::shared_ptr<const FrozenExecutionBindings> bindings,
    std::vector<PreparedBackingState> initialStates,
    std::shared_ptr<const PreparedExecutionBarrierPlan> barrierPlan,
    std::vector<std::shared_ptr<const void>> leases = {},
    std::vector<std::vector<ExternalTimelinePoint>> externalWaitsByPreparedPass = {},
    std::shared_ptr<const RealizedResourceBundle> resources = {},
    uint32_t preparationSlot = 0) {
    BT_ZONE_SCOPE("ORG.AsyncExecution.BuildFrameSnapshot");
    if (!frameNumber || !layout || !layout->bundle || !bindings
        || passes.size() != layout->placements.size() || initialStates.empty() || !barrierPlan
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
        stateLedger.Prepare(*layout->bundle->graph, payload->initialStates));
    return BuildRenderFrameSnapshot(payload->frameNumber, std::move(layout),
        payload->passes, payload->bindings, payload->initialStates,
        std::move(barrierPlan), payload->leases, payload->externalWaitsByPreparedPass,
        payload->resources, payload->preparationSlot);
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
class PreparedRhiExecutionBatch final : public IPreparedExecutionBatch {
public:
    PreparedRhiExecutionBatch(uint32_t queueSlot, rhi::Queue queue, std::vector<rhi::CommandList> lists,
        std::vector<PreparedTimelineBinding> timelines, std::shared_ptr<const void> lease,
        std::vector<PreparedPass> submissionEffects = {})
        : m_queueSlot(queueSlot), m_queue(queue), m_lists(std::move(lists)), m_timelines(std::move(timelines)),
          m_lease(std::move(lease)), m_submissionEffects(std::move(submissionEffects)) {
        if (!m_queue || m_lists.empty() || !m_lease || m_lists.size() > UINT32_MAX)
            throw std::invalid_argument("Incomplete prepared RHI packet");
        for (auto list : m_lists) if (!list) throw std::invalid_argument("Invalid command list");
        for (size_t i = 0; i < m_timelines.size(); ++i) {
            if (!m_timelines[i].identity) throw std::invalid_argument("Invalid timeline identity");
            for (size_t j = 0; j < i; ++j)
                if (m_timelines[i].identity == m_timelines[j].identity)
                    throw std::invalid_argument("Duplicate timeline identity");
        }
    }
    uint32_t QueueSlot() const noexcept override { return m_queueSlot; }
    SubmissionReceipt Submit(const ExecutionBatchTimeline& batch) const noexcept override {
        // Single-consumption even after failure: uploads/readbacks cannot replay.
        if (m_consumed.exchange(true)) return {SubmissionState::NotSubmitted, SubmissionFailureStage::Replay};
        auto find = [&](uint64_t identity) -> const PreparedTimelineBinding* {
            for (const auto& item : m_timelines) if (item.identity == identity) return &item;
            return nullptr;
        };
        const auto* signal = find(batch.signal.timeline);
        if (!signal || !batch.signal.value || batch.signal.value == UINT64_MAX) {
            Abandon();
            return {SubmissionState::NotSubmitted, SubmissionFailureStage::Validation};
        }
        for (auto wait : batch.waits) if (!find(wait.timeline)) {
            Abandon();
            return {SubmissionState::NotSubmitted, SubmissionFailureStage::Validation};
        }
        for (auto wait : batch.waits) {
            const auto result = m_queue.Wait({find(wait.timeline)->handle, wait.value});
            if (result != rhi::Result::Ok) {
                Abandon();
                return {SubmissionState::NotSubmitted, SubmissionFailureStage::Wait, static_cast<uint32_t>(result)};
            }
        }
        const auto submitted = m_queue.Submit({m_lists.data(), static_cast<uint32_t>(m_lists.size())}, {});
        if (submitted != rhi::Result::Ok)
            return {SubmissionState::SubmissionUncertain, SubmissionFailureStage::Submit, static_cast<uint32_t>(submitted)};
        for (const auto& pass : m_submissionEffects) pass.CommitSubmitted({batch.signal.value});
        for (const auto& pass : m_submissionEffects) {
            for (const auto& external : pass.ExternalSignalsAfterCompletion()) {
                if (!external.timeline || !external.value)
                    return {SubmissionState::SubmittedWithoutSignal, SubmissionFailureStage::Signal};
                const auto externalSignaled = m_queue.Signal({external.timeline.GetHandle(), external.value});
                if (externalSignaled != rhi::Result::Ok)
                    return {SubmissionState::SubmittedWithoutSignal, SubmissionFailureStage::Signal,
                        static_cast<uint32_t>(externalSignaled)};
            }
        }
        const auto signaled = m_queue.Signal({signal->handle, batch.signal.value});
        if (signaled != rhi::Result::Ok)
            return {SubmissionState::SubmittedWithoutSignal, SubmissionFailureStage::Signal, static_cast<uint32_t>(signaled)};
        return {SubmissionState::Signaled};
    }
    void Complete(uint64_t submission) const noexcept override {
        for (const auto& pass : m_submissionEffects) {
            try { pass.CommitCompleted({submission}); }
            catch (...) { basic_telemetry::AddCounter("ORG.AsyncExecution.InvalidCompletionTransition"); }
        }
    }
    void Abandon() const noexcept override {
        for (const auto& pass : m_submissionEffects) pass.Abandon(AbandonReason::AdmissionFailed);
    }
private:
    uint32_t m_queueSlot;
    mutable rhi::Queue m_queue;
    mutable std::vector<rhi::CommandList> m_lists; // RHI Span uses non-const handles.
    std::vector<PreparedTimelineBinding> m_timelines;
    std::shared_ptr<const void> m_lease;
    std::vector<PreparedPass> m_submissionEffects;
    mutable std::atomic_bool m_consumed{false};
};

// One owned recording packet. The binding table transitively owns descriptor
// snapshots and backing versions; pass data owns immutable pipeline references.
// Member order retires command lists before their allocators and bindings.
struct OwnedRecordingList {
    std::shared_ptr<const FrozenExecutionBindings> bindings;
    std::shared_ptr<const std::vector<ExternalDescriptorBindingValue>> externalBindings;
    // Admission-captured defaults for passes using directly-indexed root
    // signatures. Individual passes may rebind a compatible snapshot.
    rhi::DescriptorHeapHandle resourceDescriptorHeap{};
    rhi::DescriptorHeapHandle samplerDescriptorHeap{};
    std::vector<rhi::TextureBarrier> textureBarriers;
    std::vector<rhi::BufferBarrier> bufferBarriers;
    std::vector<PreparedBatchBarriers::BeforePass> barriersBeforePass;
    std::vector<PreparedBatchBarriers::BeforePass> barriersAfterPass;
    std::vector<PreparedPass> passes;
    rhi::CommandAllocatorPtr allocator;
    rhi::CommandListPtr commands;
};

// Called only after admission freezes bindings and barriers. It may run on a
// recording worker; no declaration/Setup callbacks or live registries are used.
// The runtime owner must own the queue/device and supplied timeline objects.
inline std::shared_ptr<const PreparedRhiExecutionBatch> RecordPreparedRhiExecutionBatch(
    uint32_t queueSlot, rhi::Queue queue, std::vector<OwnedRecordingList> recordings,
    std::vector<PreparedTimelineBinding> timelines, std::shared_ptr<const void> runtimeOwner) {
    BT_ZONE_SCOPE("ORG.AsyncExecution.RecordOwnedBatch");
    if (!runtimeOwner || recordings.empty()) throw std::invalid_argument("Missing recording ownership");
    struct Ownership {
        std::shared_ptr<const void> runtime;
        std::vector<OwnedRecordingList> recordings;
    };
    // Validate all packets before consuming any pass. Empty legacy preparation
    // must be selected into a synchronous route by the preparation owner.
    for (const auto& recording : recordings) {
        if (!recording.bindings || !recording.allocator || !recording.commands || recording.passes.empty())
            throw std::invalid_argument("Incomplete owned recording packet");
        if (!recording.commands.Get().SupportsCheckedEnd())
            throw std::invalid_argument("Backend lacks checked command list close");
        if (!recording.barriersBeforePass.empty()
            && recording.barriersBeforePass.size() != recording.passes.size())
            throw std::invalid_argument("Prepared pass/barrier count mismatch");
        if (!recording.barriersAfterPass.empty()
            && recording.barriersAfterPass.size() != recording.passes.size())
            throw std::invalid_argument("Prepared post-pass/barrier count mismatch");
        for (const auto& pass : recording.passes)
            if (!pass) throw std::invalid_argument("Legacy pass in owned recording batch");
    }
    auto ownership = std::make_shared<Ownership>(Ownership{std::move(runtimeOwner), std::move(recordings)});
    std::vector<rhi::CommandList> lists;
    std::vector<PreparedPass> submissionEffects;
    lists.reserve(ownership->recordings.size());
    for (const auto& recording : ownership->recordings) {
        lists.push_back(recording.commands.Get());
        submissionEffects.insert(submissionEffects.end(), recording.passes.begin(), recording.passes.end());
    }
    auto packet = std::make_shared<const PreparedRhiExecutionBatch>(queueSlot, queue,
        std::move(lists), std::move(timelines), ownership, std::move(submissionEffects));
    for (const auto& recording : ownership->recordings) {
        RecordingContext context(recording.commands.Get(), recording.bindings, recording.externalBindings);
        if (recording.resourceDescriptorHeap.valid()) {
            context.Commands().SetDescriptorHeaps(recording.resourceDescriptorHeap,
                recording.samplerDescriptorHeap.valid()
                    ? std::optional<rhi::DescriptorHeapHandle>{recording.samplerDescriptorHeap}
                    : std::nullopt);
        }
        if (!recording.textureBarriers.empty() || !recording.bufferBarriers.empty()) {
            rhi::BarrierBatch barriers{};
            barriers.textures = {recording.textureBarriers.data(), static_cast<uint32_t>(recording.textureBarriers.size())};
            barriers.buffers = {recording.bufferBarriers.data(), static_cast<uint32_t>(recording.bufferBarriers.size())};
            context.Commands().Barriers(barriers);
        }
        for (size_t passIndex = 0; passIndex < recording.passes.size(); ++passIndex) {
            if (!recording.barriersBeforePass.empty()) {
                const auto& before = recording.barriersBeforePass[passIndex];
                if (!before.textures.empty() || !before.buffers.empty()) {
                    rhi::BarrierBatch barriers{};
                    barriers.textures = {before.textures.data(), static_cast<uint32_t>(before.textures.size())};
                    barriers.buffers = {before.buffers.data(), static_cast<uint32_t>(before.buffers.size())};
                    context.Commands().Barriers(barriers);
                }
            }
            recording.passes[passIndex].Record(context);
            if (!recording.barriersAfterPass.empty()) {
                const auto& after = recording.barriersAfterPass[passIndex];
                if (!after.textures.empty() || !after.buffers.empty()) {
                    rhi::BarrierBatch barriers{};
                    barriers.textures = {after.textures.data(), static_cast<uint32_t>(after.textures.size())};
                    barriers.buffers = {after.buffers.data(), static_cast<uint32_t>(after.buffers.size())};
                    context.Commands().Barriers(barriers);
                }
            }
        }
        if (context.Commands().EndChecked() != rhi::Result::Ok) {
            basic_telemetry::AddCounter("ORG.AsyncExecution.RecordingCloseFailures");
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
    }
    return packet;
}

} // namespace org::experimental
