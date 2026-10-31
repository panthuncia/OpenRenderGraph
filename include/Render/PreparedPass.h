#pragma once

#include <atomic>
#include <concepts>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <unordered_map>
#include <vector>
#include <string>
#include <rhi.h>
#include "RenderPasses/Base/PassReturn.h"
#include "Render/ExternalBindings.h"
#include "Render/PipelineState.h"
#include "Render/BindlessResourceViews.h"
#include "Render/ResourceUseDeclaration.h"
#include "Render/PreparedInvocationArena.h"
#include "Render/PublicationBindingBundle.h"
#include "Render/RenderGraph/CompileTelemetry.h"
#include "Resources/ResourceIdentifier.h"

namespace org {
namespace persistent { class SelectedPublication; class BindingToken; class ViewToken; }

struct IHostExecutionData;
struct SubmissionContext { uint64_t submissionID = 0; };
struct CompletionContext { uint64_t submissionID = 0; };
enum class AbandonReason : uint8_t { Shutdown, GenerationInvalidated, PreparationFailed, AdmissionFailed };
struct PreparedResourceReference { uint32_t slot = 0; };

class PreparedLifecycleEffect {
public:
    virtual ~PreparedLifecycleEffect() = default;
    virtual void Submitted(SubmissionContext) const {}
    virtual void Completed(CompletionContext) const {}
    virtual void Abandoned(AbandonReason) const {}
    virtual std::span<const ExternalTimelinePoint> SignalsAfterCompletion() const { return {}; }
};

// Reusable lifecycle adapter for framework-owned reservations. The callbacks
// receive the retained state, so they cannot accidentally depend on a pass
// object. Pass helpers can expose a domain-specific Reserve(...) operation
// without adding lifecycle fields or hooks to author FrameData.
template<class State>
class PreparedOwnedLifecycle final : public PreparedLifecycleEffect {
public:
    using SubmittedFn = void (*)(State&, SubmissionContext);
    using CompletedFn = void (*)(State&, CompletionContext);
    using AbandonedFn = void (*)(State&, AbandonReason);

    PreparedOwnedLifecycle(std::shared_ptr<State> state, SubmittedFn submitted,
        CompletedFn completed = nullptr, AbandonedFn abandoned = nullptr)
        : m_state(std::move(state)), m_submitted(submitted),
          m_completed(completed), m_abandoned(abandoned) {
        if (!m_state) throw std::invalid_argument("Cannot reserve empty lifecycle state");
    }
    void Submitted(SubmissionContext context) const override {
        if (m_submitted) m_submitted(*m_state, context);
    }
    void Completed(CompletionContext context) const override {
        if (m_completed) m_completed(*m_state, context);
    }
    void Abandoned(AbandonReason reason) const override {
        if (m_abandoned) m_abandoned(*m_state, reason);
    }
private:
    std::shared_ptr<State> m_state;
    SubmittedFn m_submitted;
    CompletedFn m_completed;
    AbandonedFn m_abandoned;
};

// What a packet keeps alive and commits with its lifecycle: ownership of what its recording names
// (Retain) and framework reservations (Reserve).
class PreparedDependencySnapshot {
public:
    bool HasLifecycleEffects() const noexcept { return !m_effects.empty(); }
    PreparedDependencySnapshot(std::vector<std::shared_ptr<const void>> owners,
        std::vector<std::shared_ptr<const PreparedLifecycleEffect>> effects)
        : m_owners(std::move(owners)), m_effects(std::move(effects)) {
        for (const auto& effect : m_effects) {
            const auto signals = effect->SignalsAfterCompletion();
            m_signals.insert(m_signals.end(), signals.begin(), signals.end());
        }
    }
    const std::vector<ExternalTimelinePoint>& SignalsAfterCompletion() const { return m_signals; }
    bool empty() const noexcept { return m_owners.empty() && m_effects.empty(); }
    void Submitted(SubmissionContext context) const {
        for (const auto& effect : m_effects) effect->Submitted(context);
    }
    void Completed(CompletionContext context) const {
        for (const auto& effect : m_effects) effect->Completed(context);
    }
    void Abandoned(AbandonReason reason) const {
        for (const auto& effect : m_effects) effect->Abandoned(reason);
    }
private:
    std::vector<std::shared_ptr<const void>> m_owners;
    std::vector<std::shared_ptr<const PreparedLifecycleEffect>> m_effects;
    std::vector<ExternalTimelinePoint> m_signals;
};

class PreparedDependencyCollector {
public:
    template<class T>
    void Retain(std::shared_ptr<T> owner) {
        if (!owner) throw std::invalid_argument("Cannot retain an empty prepared owner");
        m_owners.push_back(std::move(owner));
    }
    template<class T>
    void RetainValue(T owner) {
        m_owners.push_back(std::make_shared<const T>(std::move(owner)));
    }
    void Reserve(std::shared_ptr<const PreparedLifecycleEffect> effect) {
        if (!effect) throw std::invalid_argument("Cannot retain an empty prepared lifecycle effect");
        m_effects.push_back(std::move(effect));
    }
    std::shared_ptr<const PreparedDependencySnapshot> Freeze() && {
        if (m_owners.empty() && m_effects.empty()) return {};
        return std::make_shared<const PreparedDependencySnapshot>(std::move(m_owners), std::move(m_effects));
    }
private:
    std::vector<std::shared_ptr<const void>> m_owners;
    std::vector<std::shared_ptr<const PreparedLifecycleEffect>> m_effects;
};

// Constructed by preparation from the pass's declared permissions. These are
// concrete version owners, not shared pointers to mutable resource wrappers.
class FrozenExecutionBindings {
public:
    enum class OwnershipPolicy : uint8_t { Owned, Borrowed };
    struct ResourceBinding {
        rhi::Resource resource;
        std::shared_ptr<const void> owner;
        std::shared_ptr<const BindlessResourceViews> views;
        std::shared_ptr<const void> descriptorOwner;
        // The frame's publication root owns this exact generation. Keep a
        // non-owning lookup here so recipe capture never searches the catalog.
        const ResourceBindingSnapshot* publicationBinding = nullptr;
    };
    struct DescriptorBinding { rhi::DescriptorSlot descriptor; std::shared_ptr<const void> owner; };
    FrozenExecutionBindings(std::vector<ResourceBinding> resources,
        std::vector<DescriptorBinding> descriptors = {},
        OwnershipPolicy policy = OwnershipPolicy::Owned,
        std::shared_ptr<const PublicationBindingBundle> publicationRoot = {})
        : m_resources(std::move(resources)), m_descriptors(std::move(descriptors)), m_publicationRoot(std::move(publicationRoot)) {
        for (const auto& binding : m_resources)
            if (!binding.resource.GetHandle().valid()
                || (binding.publicationBinding && (!m_publicationRoot
                    || binding.publicationBinding->resource.GetHandle().index != binding.resource.GetHandle().index
                    || binding.publicationBinding->resource.GetHandle().generation != binding.resource.GetHandle().generation
                    || binding.publicationBinding->views != binding.views))
                || (policy == OwnershipPolicy::Owned && !binding.owner && !m_publicationRoot))
                throw std::invalid_argument("Unowned recording resource");
        for (const auto& binding : m_descriptors)
            if (policy == OwnershipPolicy::Owned && !binding.owner)
                throw std::invalid_argument("Unowned recording descriptor");
    }
    // Persistent publications may leave slots unbound (reserved capacity,
    // late-bound imports). Entries with an invalid handle are permitted here and
    // reject resolution, so a pass touching an unbound slot fails at preparation.
    static std::shared_ptr<const FrozenExecutionBindings> Sparse(std::vector<ResourceBinding> resources) {
        auto result = std::make_shared<FrozenExecutionBindings>(std::vector<ResourceBinding>{});
        for (const auto& binding : resources)
            if (binding.resource.GetHandle().valid() && !binding.owner)
                throw std::invalid_argument("Unowned sparse recording resource");
        result->m_resources = std::move(resources);
        return result;
    }
    rhi::Resource Resolve(PreparedResourceReference ref) const {
        const auto& resource = m_resources.at(ref.slot).resource;
        if (!resource.GetHandle().valid()) throw std::logic_error("Recording resolved an unbound persistent slot");
        return resource;
    }
    // Reverse lookup by native handle.
    std::optional<PreparedResourceReference> FindByHandle(rhi::ResourceHandle handle) const {
        for (uint32_t slot = 0; slot < m_resources.size(); ++slot) {
            const auto candidate = m_resources[slot].resource.GetHandle();
            if (candidate.index == handle.index && candidate.generation == handle.generation) return PreparedResourceReference{slot};
        }
        return std::nullopt;
    }
    std::shared_ptr<const void> Owner(PreparedResourceReference ref) const {
        const auto& binding = m_resources.at(ref.slot);
        if (binding.publicationBinding)
            return binding.publicationBinding->recordingOwner
                ? binding.publicationBinding->recordingOwner : binding.publicationBinding->allocationOwner;
        if (!binding.owner && m_publicationRoot)
            if (const auto* version = m_publicationRoot->FindNative(binding.resource.GetHandle(), binding.views.get()))
                return (*version)->recordingOwner ? (*version)->recordingOwner : (*version)->allocationOwner;
        return binding.descriptorOwner ? binding.descriptorOwner : binding.owner ? binding.owner : m_publicationRoot;
    }
    const BindlessResourceViews& Views(PreparedResourceReference ref) const {
        const auto& views = m_resources.at(ref.slot).views;
        if (!views) throw std::out_of_range("Resource has no captured bindless views");
        return *views;
    }
    const std::vector<ResourceBinding>& Resources() const { return m_resources; }
    const std::vector<DescriptorBinding>& Descriptors() const { return m_descriptors; }
    const std::shared_ptr<const PublicationBindingBundle>& PublicationRoot() const noexcept { return m_publicationRoot; }
private:
    std::vector<ResourceBinding> m_resources;
    std::vector<DescriptorBinding> m_descriptors;
    std::shared_ptr<const PublicationBindingBundle> m_publicationRoot;
};

struct FramePreparationContext {
    std::shared_ptr<const ResourceUseLayout> resourceUses;
    // The declaration to prepare against: in a persistent graph, the one its selected executable was built from (a pending
    // structural build's relowering takes effect at its install). Unset, the pass's latest declaration.
    std::shared_ptr<const ResourceUseLayout> preparedResourceUses;
    std::shared_ptr<const void> preparedBindings;
    std::function<rhi::DescriptorSlot(uint32_t, uint32_t)> resolveDeclaredView;
    std::function<rhi::Resource(uint32_t)> resolveDeclaredResource;
    // The view of a resolver group's first bound member this frame (ResourceUseLayout::resolverViews ordinal); an
    // invalid slot when the group has no bound member.
    std::function<rhi::DescriptorSlot(uint32_t resolverGroup, BindlessViewRequest)> resolveResolverView;
    std::shared_ptr<PreparedInvocationArena> invocationArena;
    std::function<void(std::shared_ptr<const void>)> retireOwnership;
    rhi::Device device; // Host-owned device; valid through frame retirement.
    // The device the pass runs on: program and pipeline captures default to its representation.
    BackendInstanceId backendInstance = BackendInstanceId::Primary;
    uint32_t frameIndex = 0;
    // Stable CPU frame-data slot retained by this prepared request. The
    // swapchain image and execution slot are intentionally late-bound.
    uint32_t preparationSlot = 0;
    uint64_t frameNumber = 0;
    float deltaTime = 0;
    std::shared_ptr<const FrozenExecutionBindings> bindings;
    // Valid only for the duration of PrepareFrame. Implementations copy the
    // values they need into PreparedPass data; retaining this pointer is a
    // contract violation.
    const IHostExecutionData* preparationData = nullptr;
    // Current admission-frame data. preparationData remains the retained
    // Update snapshot expected by existing pass preparation implementations.
    const IHostExecutionData* admissionData = nullptr;
    // Preparation-only access. Providers must contain owned/value snapshots,
    // not legacy RenderContext wrappers with mutable manager pointers.
    std::shared_ptr<const IHostExecutionData> frameData;
    // Installed by TypedRenderGraphPass for this one packet: ownership the packet keeps (Retain) and
    // lifecycle reservations it commits (Reserve).
    std::shared_ptr<PreparedDependencyCollector> dependencyCollector;
    // Per-pass permission map assembled from that pass's declarations. This
    // lets preparation capture a stable binding slot without retaining a
    // Resource wrapper or knowing anything about graph indices.
    struct ResourceSlots : std::vector<std::pair<uint64_t, uint32_t>> {
        using std::vector<std::pair<uint64_t, uint32_t>>::vector;
        bool sorted = false;
        const_iterator Find(uint64_t id) const {
            if (!sorted) return std::find_if(begin(), end(),
                [id](const auto& entry) { return entry.first == id; });
            const auto found = std::lower_bound(begin(), end(), id,
                [](const auto& entry, uint64_t key) { return entry.first < key; });
            return found != end() && found->first == id ? found : end();
        }
    };
    std::shared_ptr<const ResourceSlots> resourceSlots;
    PreparedResourceReference CaptureResource(uint64_t globalResourceID) const {
        if (!resourceSlots) throw std::logic_error("Resource capture is unavailable outside owned preparation");
        const auto found = resourceSlots->Find(globalResourceID);
        if (found == resourceSlots->end()) {
            std::string message = "Pass attempted to capture undeclared resource "
                + std::to_string(globalResourceID) + "; declared IDs:";
            size_t reported = 0;
            for (const auto& [id, slot] : *resourceSlots) {
                (void)slot;
                if (reported++ == 8) {
                    message += " ...";
                    break;
                }
                message += " " + std::to_string(id);
            }
            throw std::invalid_argument(std::move(message));
        }
        return {found->second};
    }

    PreparedResourceReference CaptureResource(ResourceBindingToken binding) const {
        if (resourceSlots) {
            const auto registry = resourceSlots->Find(binding.registryResourceID);
            if (registry != resourceSlots->end()) {
                return {registry->second};
            }
            const auto global = resourceSlots->Find(binding.globalResourceID);
            if (global != resourceSlots->end()) {
                return {global->second};
            }
        }
        return CaptureResource(binding.registryResourceID);
    }

    const ResourceUseDeclaration& DeclaredUse(const DeclaredViewToken& token) const {
        if (!resourceUses || token.layout != resourceUses || token.use >= resourceUses->uses.size())
            throw std::invalid_argument("Foreign or stale declared view token");
        const auto& use = resourceUses->uses[token.use];
        if (token.view >= use.requiredViews.size()) throw std::invalid_argument("Invalid declared view ordinal");
        return use;
    }
    rhi::DescriptorSlot Resolve(const DeclaredViewToken& token) const {
        const auto& use = DeclaredUse(token);
        if (resolveDeclaredView) {
            (void)bindings->Views(DeclaredReference(token));
            return resolveDeclaredView(token.use, token.view);
        }
        if (!bindings) throw std::logic_error("Frozen resource bindings are unavailable during preparation");
        const auto request = use.requiredViews[token.view];
        try {
            return bindings->Views(CaptureResource(use.binding)).Resolve(request);
        } catch (const std::exception& error) {
            throw std::out_of_range(std::string(error.what()) + "; resource="
                + std::to_string(use.binding.globalResourceID) + "/" + std::to_string(use.binding.registryResourceID)
                + " kind=" + std::to_string(static_cast<uint32_t>(request.kind))
                + " variant=" + std::to_string(request.variant)
                + " mip=" + std::to_string(request.mip) + " slice=" + std::to_string(request.slice));
        }
    }
    PreparedResourceReference DeclaredReference(const DeclaredViewToken& token) const {
        const auto& use = DeclaredUse(token);
        if (!resolveDeclaredResource) return CaptureResource(use.binding);
        const auto resource = resolveDeclaredResource(token.use);
        const auto reference = bindings->FindByHandle(resource.GetHandle());
        if (!reference) throw std::invalid_argument("Declared view backing is absent from frozen pass bindings");
        return *reference;
    }
    // Every declared view of the pass, resolved once against this frame's bindings, in
    // ResourceUseLayout::FlatView order: what RecordingContext::Index reads. A view that cannot be resolved is
    // left invalid and fails when a recording asks for it.
    std::vector<rhi::DescriptorSlot> ResolveDeclaredViewTable() const {
        std::vector<rhi::DescriptorSlot> table;
        if (!resourceUses) return table;
        table.reserve(resourceUses->ViewCount());
        for (uint32_t use = 0; use < resourceUses->uses.size(); ++use)
            for (uint32_t view = 0; view < resourceUses->uses[use].requiredViews.size(); ++view) {
                rhi::DescriptorSlot slot{};
                try {
                    slot = resolveDeclaredView ? resolveDeclaredView(use, view) : Resolve(DeclaredViewToken{resourceUses, use, view});
                } catch (const std::exception&) { basic_telemetry::AddCounter("ORG.Execution.UnresolvedDeclaredViews"); }
                table.push_back(slot);
            }
        return table;
    }
    // The views of the pass's names declared through a resolver group, parallel to ResourceUseLayout::namedViews
    // (invalid for the other names, and for a group without a bound member).
    std::vector<rhi::DescriptorSlot> ResolveNamedGroupViews() const {
        std::vector<rhi::DescriptorSlot> views;
        if (!resourceUses || !resolveResolverView) return views;
        views.resize(resourceUses->namedViews.size());
        for (size_t i = 0; i < views.size(); ++i) {
            const auto& named = resourceUses->namedViews[i];
            if (named.resolverGroup == UINT32_MAX) continue;
            try { views[i] = resolveResolverView(named.resolverGroup, named.request); }
            catch (const std::exception&) { basic_telemetry::AddCounter("ORG.Execution.UnresolvedDeclaredViews"); }
        }
        return views;
    }

    const rhi::ResourceDesc& Describe(ResourceBindingToken binding) const {
        if (!bindings) throw std::logic_error("Frozen resource bindings are unavailable during preparation");
        return bindings->Views(CaptureResource(binding)).description;
    }
    const rhi::ResourceDesc& Describe(const DeclaredViewToken& token) const {
        if (!bindings) throw std::logic_error("Frozen resource bindings are unavailable during preparation");
        return bindings->Views(DeclaredReference(token)).description;
    }

    rhi::ClearValue ClearValue(ResourceBindingToken binding) const {
        if (!bindings) throw std::logic_error("Frozen resource bindings are unavailable during preparation");
        const auto& views = bindings->Views(CaptureResource(binding));
        if (!views.hasClear) throw std::out_of_range("Resource has no captured clear value");
        return views.clear;
    }


    rhi::Resource ResolveCapturedResource(PreparedResourceReference reference) const {
        if (!bindings) throw std::logic_error("Frozen resource bindings are unavailable during preparation");
        return bindings->Resolve(reference);
    }

    template<class T>
    void Retain(std::shared_ptr<T> owner) const {
        if (!dependencyCollector) throw std::logic_error("Ownership capture is unavailable outside typed preparation");
        dependencyCollector->Retain(std::move(owner));
    }

    template<class T>
    void RetainValue(T owner) const {
        if (!dependencyCollector) throw std::logic_error("Ownership capture is unavailable outside typed preparation");
        dependencyCollector->RetainValue(std::move(owner));
    }

    // Framework helpers use this for reservations whose commit/abandonment is
    // coupled to the packet lifecycle. Ordinary pass data does not carry or
    // forward these effects itself.
    void Reserve(std::shared_ptr<const PreparedLifecycleEffect> effect) const {
        if (!dependencyCollector) throw std::logic_error("Lifecycle reservation is unavailable outside typed preparation");
        dependencyCollector->Reserve(std::move(effect));
    }

    template<class State>
    void Reserve(std::shared_ptr<State> state,
        typename PreparedOwnedLifecycle<State>::SubmittedFn submitted,
        typename PreparedOwnedLifecycle<State>::CompletedFn completed = nullptr,
        typename PreparedOwnedLifecycle<State>::AbandonedFn abandoned = nullptr) const {
        Reserve(std::make_shared<const PreparedOwnedLifecycle<State>>(
            std::move(state), submitted, completed, abandoned));
    }

};

// Ownership of what one recording's commands name (the pipelines it bound, the command signatures and work
// graphs it used), kept until the recording's GPU work retires - including every resubmission of a kept
// recording. A late batch recorded again appends to the same retention.
class RecordingRetention {
public:
    void Retain(std::shared_ptr<const void> owner) {
        if (!owner) throw std::invalid_argument("Cannot retain an empty recording owner");
        std::lock_guard lock(m_mutex);
        for (const auto& existing : m_owners) if (existing == owner) return;
        m_owners.push_back(std::move(owner));
    }
private:
    std::mutex m_mutex;
    std::vector<std::shared_ptr<const void>> m_owners;
};

// A pass's declarations as its recording sees them: the declared-view layout and the views the frame
// resolved for it.
struct DeclaredRecordingScope {
    std::shared_ptr<const ResourceUseLayout> layout;
    std::vector<rhi::DescriptorSlot> views;  // ResourceUseLayout::FlatView order
    std::vector<rhi::DescriptorSlot> namedGroupViews;  // parallel to ResourceUseLayout::namedViews
    BackendInstanceId backendInstance = BackendInstanceId::Primary;
};

// RecordingContext::Bind*Pipeline: ResourceDescriptorIndex names (prefixes) that may be undeclared and read UINT32_MAX.
struct PipelineBindOptions { std::span<const std::string_view> absentPrefixes{}; };

class RecordingContext {
public:
    static RecordingContext FromPersistentBindings(rhi::CommandList commands,
        std::shared_ptr<const persistent::SelectedPublication> publication);
    RecordingContext WithPersistentBindings(std::shared_ptr<const persistent::SelectedPublication> publication) const;
    const persistent::SelectedPublication* PersistentPublication() const noexcept { return m_persistentPublication.get(); }
    // The frame slot of the execution being recorded (the host's frames-in-flight index). A recording
    // may depend on it - a LatchBlock region is per slot - but on nothing else about the frame.
    uint32_t FrameSlot() const noexcept { return m_frameSlot; }
    void SetFrameSlot(uint32_t slot) noexcept { m_frameSlot = slot; }
    rhi::Resource Resolve(persistent::BindingToken token) const;
    rhi::DescriptorSlot Resolve(persistent::ViewToken token) const;
    RecordingContext(rhi::CommandList commands, std::shared_ptr<const FrozenExecutionBindings> bindings,
        std::shared_ptr<const std::vector<ExternalDescriptorBindingValue>> externalBindings = {})
        : m_commands(commands), m_bindings(std::move(bindings)), m_externalBindings(std::move(externalBindings)) {
        if (!m_commands || !m_bindings) throw std::invalid_argument("Incomplete recording context");
    }
    rhi::CommandList& Commands() { return m_commands; }
    rhi::Resource Resolve(PreparedResourceReference ref) const {
        if (!m_bindings) throw std::logic_error("Legacy reference requires frozen recording bindings");
        return m_bindings->Resolve(ref);
    }
    // The frame's backing of a declared resource.
    rhi::Resource Resolve(ResourceBindingToken token) const;
    rhi::Resource Resolve(const DeclaredViewToken& token) const { return Resolve(token.Resource()); }
    // The frame's descriptor of a declared view, and its bindless index.
    rhi::DescriptorSlot View(const DeclaredViewToken& token) const;
    uint32_t Index(const DeclaredViewToken& token) const { return View(token).index; }
    // Any bindless view of a declared resource, for a pass that moves it between access kinds with barriers of its
    // own (declared for unordered access, read through its shader-resource view after an in-pass barrier).
    rhi::DescriptorSlot View(ResourceBindingToken resource, BindlessViewRequest request) const;
    uint32_t Index(ResourceBindingToken resource, BindlessViewRequest request) const { return View(resource, request).index; }
    rhi::DescriptorSlot Resolve(ExternalBindingKey key) const {
        if (!m_externalBindings) throw std::out_of_range("Missing external binding table");
        for (const auto& binding : *m_externalBindings)
            if (binding.key == key && binding.descriptor.heap.valid()) return binding.descriptor;
        throw std::out_of_range("Missing external descriptor binding");
    }

    // Keeps `owner` alive until this recording's GPU work retires.
    void Retain(std::shared_ptr<const void> owner);

    // Binds the pipeline's current payload (retained by the recording) and its layout, then fills its
    // ResourceDescriptorIndex slots from the pass's declarations by resource name. A slot whose name the pass did
    // not declare gets UINT32_MAX when it is optional, belongs to a feature domain the pass names nothing of, or
    // starts with one of `absentPrefixes`; otherwise binding throws.
    using BindOptions = PipelineBindOptions;
    void BindComputePipeline(const PipelineState& pipeline, BindOptions options = {});
    void BindGraphicsPipeline(const PipelineState& pipeline, BindOptions options = {});
    void BindComputePipeline(std::shared_ptr<const PipelineStatePayload> payload, BindOptions options = {});
    void BindGraphicsPipeline(std::shared_ptr<const PipelineStatePayload> payload, BindOptions options = {});
    // A pipeline made outside PipelineState: no descriptor slots.
    void BindComputePipeline(std::shared_ptr<const rhi::PipelinePtr> pipeline, rhi::PipelineLayoutHandle layout);
    void BindGraphicsPipeline(std::shared_ptr<const rhi::PipelinePtr> pipeline, rhi::PipelineLayoutHandle layout);
    // Root constants for the stages of the last bound pipeline.
    void PushConstants(uint32_t rootParameter, std::span<const uint32_t> values, uint32_t destinationOffset = 0);
    // The indices a pipeline's ResourceDescriptorIndex slots resolve to (what binding it pushes).
    std::vector<uint32_t> DescriptorSlotIndices(const PipelineStatePayload& payload, BindOptions options = {}) const;

    // Buffer barriers on declared resources, as one batch.
    void BufferBarriers(std::initializer_list<ResourceBindingToken> buffers,
        rhi::ResourceAccessType beforeAccess, rhi::ResourceAccessType afterAccess,
        rhi::ResourceSyncState beforeSync = rhi::ResourceSyncState::ComputeShading,
        rhi::ResourceSyncState afterSync = rhi::ResourceSyncState::ComputeShading);
    // A full unordered-access barrier between compute dispatches.
    void UavBarrier();
    void ExecuteIndirect(const std::shared_ptr<const rhi::CommandSignaturePtr>& signature,
        ResourceBindingToken arguments, uint64_t argumentOffset = 0, uint32_t maxCommandCount = 1,
        std::optional<ResourceBindingToken> count = std::nullopt, uint64_t countOffset = 0);
    void SetWorkGraph(const std::shared_ptr<const rhi::WorkGraphPtr>& workGraph, ResourceBindingToken backing,
        bool initializeBacking);

    // The framework creates a pass-local copy of the recording context. Its
    // permissions and concrete allocation table travel with the prepared frame;
    // declaration tokens never consult the current graph or a resource wrapper.
    RecordingContext WithDeclaredResources(
        std::shared_ptr<const FrozenExecutionBindings> bindings,
        std::shared_ptr<const FramePreparationContext::ResourceSlots> slots,
        std::shared_ptr<const DeclaredRecordingScope> scope = {}) const {
        if (!bindings || !slots) throw std::invalid_argument("Incomplete declared recording scope");
        auto result = *this;
        result.m_bindings = std::move(bindings);
        result.m_declaredResourceSlots = std::move(slots);
        result.m_declaredScope = std::move(scope);
        return result;
    }
    void SetRetention(std::shared_ptr<RecordingRetention> retention) noexcept { m_retention = std::move(retention); }
    // The descriptor heaps the list binds before its first pass (the admitted frame's). A pass that hands the list to
    // code binding heaps of its own (a vendor SDK, an overlay) calls RestoreDescriptorHeaps afterwards, so the passes
    // after it read the frame's heaps again.
    void SetDescriptorHeaps(rhi::DescriptorHeapHandle resources, rhi::DescriptorHeapHandle samplers) noexcept {
        m_resourceHeap = resources;
        m_samplerHeap = samplers;
    }
    void RestoreDescriptorHeaps() {
        if (!m_resourceHeap.valid()) return;
        m_commands.SetDescriptorHeaps(m_resourceHeap,
            m_samplerHeap.valid() ? std::optional<rhi::DescriptorHeapHandle>{m_samplerHeap} : std::nullopt);
    }
    void SetPreparedDependencies(std::shared_ptr<const PreparedDependencySnapshot> dependencies) {
        m_dependencies = std::move(dependencies);
    }
private:
    void BindPayload(std::shared_ptr<const PipelineStatePayload> payload, rhi::ShaderStage stages, BindOptions options);
    struct PersistentTag {};
    RecordingContext(rhi::CommandList commands, std::shared_ptr<const persistent::SelectedPublication> publication, PersistentTag);
    std::shared_ptr<const persistent::SelectedPublication> m_persistentPublication;
    uint32_t m_frameSlot = 0;
    rhi::CommandList m_commands;
    std::shared_ptr<const FrozenExecutionBindings> m_bindings;
    std::shared_ptr<const std::vector<ExternalDescriptorBindingValue>> m_externalBindings;
    std::shared_ptr<const PreparedDependencySnapshot> m_dependencies;
    std::shared_ptr<const FramePreparationContext::ResourceSlots> m_declaredResourceSlots;
    std::shared_ptr<const DeclaredRecordingScope> m_declaredScope;
    std::shared_ptr<RecordingRetention> m_retention;
    rhi::ShaderStage m_boundStages = rhi::ShaderStage::Compute;
    rhi::DescriptorHeapHandle m_resourceHeap{}, m_samplerHeap{};
};

// One owned packet per submitted frame. Copies share the single-consumption
// state; no mutable pass object is implicitly retained by the recording thunk.
class PreparedPass {
public:
    PreparedPass() = default; // Explicit unsupported/legacy preparation result.
    static PreparedPass NoOp() {
        return MakeOwned(uint8_t{}, +[](const uint8_t&, RecordingContext&) {});
    }
    static PreparedPass NoOpWithExternalSignals(std::vector<ExternalTimelinePoint> signals) {
        PreparedPass result;
        auto storage = std::make_shared<Storage<uint8_t>>(
            uint8_t{}, +[](const uint8_t&, RecordingContext&) {});
        storage->externalSignals = std::move(signals);
        storage->workerSafe = true;
        result.m_storage = std::move(storage);
        return result;
    }
    template<class Data>
    static PreparedPass Make(Data data, void (*record)(const Data&, RecordingContext&)) {
        if (!record) throw std::invalid_argument("Missing prepared recording function");
        PreparedPass result;
        result.m_storage = std::make_shared<Storage<Data>>(std::move(data), record);
        return result;
    }
    // Transitional entry point for immutable, transitively-owned packets that
    // already follow the typed recording contract but have not yet moved their
    // containing pass to TypedRenderGraphPass. Unlike Make(), this explicitly
    // opts into worker recording; pointer-retaining compatibility adapters must
    // continue to use Make().
    template<class Data>
    static PreparedPass MakeOwned(Data data, void (*record)(const Data&, RecordingContext&)) {
        if (!record) throw std::invalid_argument("Missing owned recording function");
        PreparedPass result;
        auto storage = std::make_shared<Storage<Data>>(std::move(data), record);
        storage->workerSafe = true;
        result.m_storage = std::move(storage);
        return result;
    }
    template<class Data>
    static PreparedPass MakeOwnedWithExternalSignals(Data data,
        void (*record)(const Data&, RecordingContext&),
        std::vector<ExternalTimelinePoint> signals) {
        if (!record) throw std::invalid_argument("Missing owned recording function");
        PreparedPass result;
        auto storage = std::make_shared<Storage<Data>>(std::move(data), record);
        storage->externalSignals = std::move(signals);
        storage->workerSafe = true;
        result.m_storage = std::move(storage);
        return result;
    }
    template<class Data>
    static PreparedPass MakeOwned(Data data,
        void (*record)(const Data&, RecordingContext&),
        void (*submitted)(const Data&)) {
        if (!record || !submitted) throw std::invalid_argument("Missing owned pass callback");
        PreparedPass result;
        auto storage = std::make_shared<Storage<Data>>(std::move(data), record, submitted);
        storage->workerSafe = true;
        result.m_storage = std::move(storage);
        return result;
    }
    template<class Data>
    static PreparedPass MakeOwned(Data data,
        void (*record)(const Data&, RecordingContext&),
        void (*submitted)(const Data&),
        std::vector<ExternalTimelinePoint> signals) {
        if (!record || !submitted) throw std::invalid_argument("Missing owned pass callback");
        PreparedPass result;
        auto storage = std::make_shared<Storage<Data>>(std::move(data), record, submitted);
        storage->externalSignals = std::move(signals);
        storage->workerSafe = true;
        result.m_storage = std::move(storage);
        return result;
    }
    // Canonical typed-pass entry point. Derived owns no runtime instance
    // dependency: recording and lifecycle hooks are static and operate only on
    // the immutable frame data captured by Prepare.
    template<class Derived, class Data>
    static PreparedPass FromTyped(Data data,
        std::shared_ptr<const PreparedDependencySnapshot> dependencies = {},
        std::shared_ptr<PreparedInvocationArena> arena = {}) {
        static_assert(requires(const Data& value, RecordingContext& context) {
            { Derived::Record(value, context) } -> std::same_as<void>;
        }, "Typed passes require static Record(const FrameData&, RecordingContext&)");
        PreparedPass result;
        auto storage = arena ? arena->MakeShared<TypedStorage<Derived, Data>>(std::move(data))
                             : std::make_shared<TypedStorage<Derived, Data>>(std::move(data));
        if (dependencies) storage->externalSignals = dependencies->SignalsAfterCompletion();
        storage->dependencies = std::move(dependencies);
        storage->workerSafe = true;
        result.m_storage = std::move(storage);
        return result;
    }
    template<class Data>
    static PreparedPass MakeWithExternalSignals(Data data,
        void (*record)(const Data&, RecordingContext&),
        std::vector<ExternalTimelinePoint> signals) {
        if (!record) throw std::invalid_argument("Missing prepared recording function");
        PreparedPass result;
        auto storage = std::make_shared<Storage<Data>>(std::move(data), record);
        storage->externalSignals = std::move(signals);
        result.m_storage = std::move(storage);
        return result;
    }
    template<class Data>
    static PreparedPass Make(Data data, void (*record)(const Data&, RecordingContext&),
        void (*submitted)(const Data&)) {
        if (!record || !submitted) throw std::invalid_argument("Missing prepared pass callback");
        PreparedPass result;
        result.m_storage = std::make_shared<Storage<Data>>(std::move(data), record, submitted);
        return result;
    }
    template<class Data>
    static PreparedPass Make(Data data, void (*record)(const Data&, RecordingContext&),
        void (*submitted)(const Data&), std::vector<ExternalTimelinePoint> signals) {
        if (!record || !submitted) throw std::invalid_argument("Missing prepared pass callback");
        PreparedPass result;
        auto storage = std::make_shared<Storage<Data>>(std::move(data), record, submitted);
        storage->externalSignals = std::move(signals);
        result.m_storage = std::move(storage);
        return result;
    }
    // The packet followed by commands of its own and lifecycle effects of its own, as one packet: a copy of a resource
    // the pass wrote, recorded right after it in the same executable (a readback capture), with no new pass to schedule.
    // The trailer records into the pass's list after the pass; the effects commit after the pass's own. Never replayable.
    static PreparedPass WithTrailer(PreparedPass inner, std::function<void(RecordingContext&)> trailer,
        std::vector<std::shared_ptr<const PreparedLifecycleEffect>> effects, std::shared_ptr<const void> keepAlive = {}) {
        if (!inner.m_storage || inner.IsConsumed()) throw std::invalid_argument("A trailer needs an unconsumed packet");
        if (!trailer) throw std::invalid_argument("Missing trailer recording");
        auto storage = std::make_shared<TrailerStorage>();
        storage->inner = inner.m_storage;
        storage->trailer = std::move(trailer);
        storage->effects = std::move(effects);
        storage->keepAlive = std::move(keepAlive);
        storage->dependencies = std::move(inner.m_storage->dependencies);
        storage->externalSignals = inner.m_storage->externalSignals;
        for (const auto& effect : storage->effects)
            for (const auto& signal : effect->SignalsAfterCompletion()) storage->externalSignals.push_back(signal);
        storage->debugName = inner.m_storage->debugName;
        storage->workerSafe = inner.m_storage->workerSafe;
        PreparedPass result;
        result.m_storage = std::move(storage);
        return result;
    }
    explicit operator bool() const { return static_cast<bool>(m_storage); }
    bool IsConsumed() const { return m_storage && m_storage->state.load(std::memory_order_acquire) != LifecycleState::Prepared; }
    bool IsWorkerSafe() const noexcept { return m_storage && m_storage->workerSafe; }
    // Its recording may be submitted again and again (a replayable packet, OwnedRecordingList::replayable): it has no
    // submission, completion or abandonment callback, no external signal and no dependency lifecycle effect, so a submission
    // has nothing to commit but the commands themselves. A replayed pass is recorded once and never transitions further;
    // its data lives as long as the recording.
    bool IsReplayable() const noexcept {
        return m_storage && !m_storage->HasCallbacks() && m_storage->externalSignals.empty()
            && (!m_storage->dependencies || !m_storage->dependencies->HasLifecycleEffects());
    }
    void SetDebugName(std::string name) const { if (m_storage) m_storage->debugName = std::move(name); }
    std::string_view DebugName() const { return m_storage ? std::string_view{m_storage->debugName} : std::string_view{}; }
    const std::vector<ExternalTimelinePoint>& ExternalSignalsAfterCompletion() const {
        static const std::vector<ExternalTimelinePoint> empty;
        return m_storage ? m_storage->externalSignals : empty;
    }
    void Record(RecordingContext& context) const {
        if (!m_storage) throw std::logic_error("Legacy pass has no owned recording packet");
        auto expected = LifecycleState::Prepared;
        if (!m_storage->state.compare_exchange_strong(expected, LifecycleState::Recorded))
            throw std::logic_error("Prepared pass already consumed");
        // A failed recording is not replayable: preparation must produce a new
        // packet and the work owner decides whether its commands may be retried.
        context.SetPreparedDependencies(m_storage->dependencies);
        try {
            m_storage->Record(context);
            context.SetPreparedDependencies({});
        }
        catch (...) {
            context.SetPreparedDependencies({});
            throw;
        }
    }
    // A replayable pass recorded again (after Record), into another list, with the same data: a late batch of a kept recording,
    // which each execution records into its host's list (PreparedRhiExecutionBatch::RecordDeferred). No transition: a replayable
    // pass has no lifecycle to advance.
    void RecordAgain(RecordingContext& context) const {
        if (!m_storage) throw std::logic_error("Legacy pass has no owned recording packet");
        if (!IsReplayable() || m_storage->state.load(std::memory_order_acquire) != LifecycleState::Recorded)
            throw std::logic_error("Only a recorded replayable pass is recorded again");
        context.SetPreparedDependencies(m_storage->dependencies);
        try {
            m_storage->Record(context);
            context.SetPreparedDependencies({});
        }
        catch (...) {
            context.SetPreparedDependencies({});
            throw;
        }
    }
    void CommitSubmitted(SubmissionContext context = {}) const {
        if (!m_storage) throw std::logic_error("Legacy pass has no submission effect");
        auto expected = LifecycleState::Recorded;
        if (!m_storage->state.compare_exchange_strong(expected, LifecycleState::Submitted))
            throw std::logic_error("Prepared pass submission transition is invalid");
        if (m_storage->dependencies) m_storage->dependencies->Submitted(context);
        m_storage->CommitSubmitted(context);
    }
    void CommitCompleted(CompletionContext context = {}) const {
        if (!m_storage) throw std::logic_error("Missing prepared pass completion state");
        auto expected = LifecycleState::Submitted;
        if (!m_storage->state.compare_exchange_strong(expected, LifecycleState::Completed))
            throw std::logic_error("Prepared pass completion transition is invalid");
        if (m_storage->dependencies) m_storage->dependencies->Completed(context);
        m_storage->CommitCompleted(context);
        m_storage->dependencies.reset();
    }
    bool Abandon(AbandonReason reason) const {
        if (!m_storage) return false;
        auto current = m_storage->state.load(std::memory_order_acquire);
        while (current == LifecycleState::Prepared || current == LifecycleState::Recorded) {
            if (m_storage->state.compare_exchange_weak(current, LifecycleState::Abandoned)) {
                if (m_storage->dependencies) m_storage->dependencies->Abandoned(reason);
                m_storage->Abandon(reason);
                m_storage->dependencies.reset();
                return true;
            }
        }
        return false;
    }
private:
    enum class LifecycleState : uint8_t { Prepared, Recorded, Submitted, Completed, Abandoned };
    struct StorageBase {
        virtual ~StorageBase() = default;
        virtual void Record(RecordingContext&) const = 0;
        virtual void CommitSubmitted(SubmissionContext) const = 0;
        virtual void CommitCompleted(CompletionContext) const {}
        virtual void Abandon(AbandonReason) const {}
        virtual bool HasCallbacks() const noexcept { return false; }
        mutable std::atomic<LifecycleState> state{LifecycleState::Prepared};
        mutable std::shared_ptr<const PreparedDependencySnapshot> dependencies;
        std::vector<ExternalTimelinePoint> externalSignals;
        mutable std::string debugName;
        bool workerSafe = false;
    };
    template<class Data> struct Storage final : StorageBase {
        Storage(Data value, void (*fn)(const Data&, RecordingContext&), void (*onSubmitted)(const Data&) = nullptr)
			: data(std::move(value)), record(fn), submitted(onSubmitted) {}
		void Record(RecordingContext& context) const override { record(*data, context); }
		void CommitSubmitted(SubmissionContext) const override { if (submitted) submitted(*data); }
		void CommitCompleted(CompletionContext) const override { data.reset(); }
		void Abandon(AbandonReason) const override { data.reset(); }
		bool HasCallbacks() const noexcept override { return submitted != nullptr; }
		mutable std::optional<Data> data;
        void (*const record)(const Data&, RecordingContext&);
        void (*const submitted)(const Data&);
    };
    template<class Derived, class Data> struct TypedStorage final : StorageBase {
        explicit TypedStorage(Data value) : data(std::move(value)) {}
		void Record(RecordingContext& context) const override { Derived::Record(*data, context); }
        void CommitSubmitted(SubmissionContext context) const override {
			if constexpr (requires { Derived::Submitted(*data, context); }) Derived::Submitted(*data, context);
        }
        void CommitCompleted(CompletionContext context) const override {
			if constexpr (requires { Derived::Completed(*data, context); }) Derived::Completed(*data, context);
			data.reset();
        }
        void Abandon(AbandonReason reason) const override {
			if constexpr (requires { Derived::Abandoned(*data, reason); }) Derived::Abandoned(*data, reason);
			data.reset();
        }
        bool HasCallbacks() const noexcept override {
            return requires(const Data& value, SubmissionContext submitted) { Derived::Submitted(value, submitted); }
                || requires(const Data& value, CompletionContext completed) { Derived::Completed(value, completed); }
                || requires(const Data& value, AbandonReason reason) { Derived::Abandoned(value, reason); };
        }
		mutable std::optional<Data> data;
    };
    // The inner packet's storage is driven directly: the outer packet owns the lifecycle state and the dependencies.
    struct TrailerStorage final : StorageBase {
        void Record(RecordingContext& context) const override {
            inner->Record(context);
            trailer(context);
        }
        void CommitSubmitted(SubmissionContext context) const override {
            inner->CommitSubmitted(context);
            for (const auto& effect : effects) effect->Submitted(context);
        }
        void CommitCompleted(CompletionContext context) const override {
            inner->CommitCompleted(context);
            for (const auto& effect : effects) effect->Completed(context);
            Release();
        }
        void Abandon(AbandonReason reason) const override {
            inner->Abandon(reason);
            for (const auto& effect : effects) effect->Abandoned(reason);
            Release();
        }
        bool HasCallbacks() const noexcept override { return true; }
        void Release() const { effects.clear(); keepAlive.reset(); }
        std::shared_ptr<const StorageBase> inner;
        std::function<void(RecordingContext&)> trailer;
        mutable std::vector<std::shared_ptr<const PreparedLifecycleEffect>> effects;
        mutable std::shared_ptr<const void> keepAlive;
    };
    std::shared_ptr<const StorageBase> m_storage;
};

} // namespace org
