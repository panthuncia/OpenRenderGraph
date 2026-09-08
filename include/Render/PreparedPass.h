#pragma once

#include <atomic>
#include <concepts>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <unordered_map>
#include <vector>
#include <string>
#include <rhi.h>
#include "RenderPasses/Base/PassReturn.h"
#include "Render/ExternalBindings.h"
#include "Render/PipelineState.h"

namespace org {

struct IHostExecutionData;
struct SubmissionContext { uint64_t submissionID = 0; };
struct CompletionContext { uint64_t submissionID = 0; };
enum class AbandonReason : uint8_t { Shutdown, GenerationInvalidated, PreparationFailed, AdmissionFailed };
struct PreparedResourceReference { uint32_t slot = 0; };
// Stable declaration-time token retained by the pass. It carries no resource
// ownership and is resolved to the request's frozen backing during Prepare.
struct ResourceBindingToken {
    uint64_t globalResourceID = 0;
    uint64_t registryResourceID = 0;
};
struct PreparedDescriptorReference { uint32_t slot = 0; };
struct PreparedProgramReference { uint32_t slot = 0; };
struct PreparedWorkGraphReference { uint32_t slot = 0; };
struct PreparedProgramBinding {
    PreparedProgramReference program;
    std::vector<unsigned int> descriptorIndices;
};
struct CapturedPipeline {
    rhi::PipelineHandle pipeline{};
    // Pipelines created outside PipelineState use the same dependency slot.
    // Recording only needs an immutable handle plus lifetime ownership; pass
    // authors should not need a separate frame-data lifecycle for custom PSOs.
    std::shared_ptr<const void> owner;
    explicit operator bool() const noexcept { return pipeline.valid() && static_cast<bool>(owner); }
};

class PreparedLifecycleEffect {
public:
    virtual ~PreparedLifecycleEffect() = default;
    virtual void Submitted(SubmissionContext) const {}
    virtual void Completed(CompletionContext) const {}
    virtual void Abandoned(AbandonReason) const {}
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

class PreparedDependencySnapshot {
public:
    struct DescriptorBinding {
        rhi::DescriptorSlot descriptor;
        std::shared_ptr<const void> owner;
    };
    PreparedDependencySnapshot(std::vector<CapturedPipeline> programs,
        std::vector<std::shared_ptr<const rhi::WorkGraphPtr>> workGraphs,
        std::vector<DescriptorBinding> descriptors,
        std::vector<std::shared_ptr<const void>> owners,
        std::vector<std::shared_ptr<const PreparedLifecycleEffect>> effects)
        : m_programs(std::move(programs)), m_workGraphs(std::move(workGraphs)),
          m_descriptors(std::move(descriptors)),
          m_owners(std::move(owners)),
          m_effects(std::move(effects)) {}
    const CapturedPipeline& Resolve(PreparedProgramReference ref) const {
        return m_programs.at(ref.slot);
    }
    rhi::WorkGraphHandle Resolve(PreparedWorkGraphReference ref) const {
        return m_workGraphs.at(ref.slot)->Get().GetHandle();
    }
    rhi::DescriptorSlot Resolve(PreparedDescriptorReference ref) const {
        return m_descriptors.at(ref.slot).descriptor;
    }
    size_t DescriptorCount() const noexcept { return m_descriptors.size(); }
    bool empty() const noexcept { return m_programs.empty() && m_workGraphs.empty() && m_descriptors.empty(); }
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
    std::vector<CapturedPipeline> m_programs;
    std::vector<std::shared_ptr<const rhi::WorkGraphPtr>> m_workGraphs;
    std::vector<DescriptorBinding> m_descriptors;
    std::vector<std::shared_ptr<const void>> m_owners;
    std::vector<std::shared_ptr<const PreparedLifecycleEffect>> m_effects;
};

class PreparedDependencyCollector {
public:
    PreparedProgramReference Capture(std::shared_ptr<const PipelineStatePayload> owner) {
        if (!owner || !owner->pso) throw std::invalid_argument("Pipeline has no immutable payload");
        const auto slot = static_cast<uint32_t>(m_programs.size());
        m_programs.push_back({owner->pso.Get().GetHandle(), std::move(owner)});
        return {slot};
    }
    PreparedProgramReference Capture(const PipelineState& pipeline,
        BackendInstanceId backend = BackendInstanceId::Primary) {
        return Capture(pipeline.GetPayload(backend));
    }
    PreparedProgramReference Capture(std::shared_ptr<const rhi::PipelinePtr> owner) {
        if (!owner || !*owner) throw std::invalid_argument("Cannot capture an empty pipeline");
        const auto slot = static_cast<uint32_t>(m_programs.size());
        m_programs.push_back({owner->Get().GetHandle(), std::move(owner)});
        return {slot};
    }
    PreparedWorkGraphReference CaptureWorkGraph(
        std::shared_ptr<const rhi::WorkGraphPtr> owner) {
        if (!owner || !*owner)
            throw std::invalid_argument("Cannot capture an empty work graph");
        const auto slot = static_cast<uint32_t>(m_workGraphs.size());
        m_workGraphs.push_back(std::move(owner));
        return {slot};
    }
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
    PreparedDescriptorReference CaptureDescriptor(rhi::DescriptorSlot descriptor,
        std::shared_ptr<const void> owner) {
        if (!descriptor.heap.valid() || !owner)
            throw std::invalid_argument("Cannot capture an unowned descriptor");
        const auto slot = static_cast<uint32_t>(m_descriptors.size());
        m_descriptors.push_back({descriptor, std::move(owner)});
        return {slot};
    }
    std::shared_ptr<const PreparedDependencySnapshot> Freeze() && {
        if (m_programs.empty() && m_descriptors.empty() && m_owners.empty() && m_effects.empty()) return {};
        return std::make_shared<const PreparedDependencySnapshot>(
            std::move(m_programs), std::move(m_workGraphs), std::move(m_descriptors),
            std::move(m_owners), std::move(m_effects));
    }
private:
    std::vector<CapturedPipeline> m_programs;
    std::vector<std::shared_ptr<const rhi::WorkGraphPtr>> m_workGraphs;
    std::vector<PreparedDependencySnapshot::DescriptorBinding> m_descriptors;
    std::vector<std::shared_ptr<const void>> m_owners;
    std::vector<std::shared_ptr<const PreparedLifecycleEffect>> m_effects;
};

// Constructed by preparation from the pass's declared permissions. These are
// concrete version owners, not shared pointers to mutable resource wrappers.
class FrozenExecutionBindings {
public:
    struct ResourceBinding { rhi::Resource resource; std::shared_ptr<const void> owner; };
    struct DescriptorBinding { rhi::DescriptorSlot descriptor; std::shared_ptr<const void> owner; };
    FrozenExecutionBindings(std::vector<ResourceBinding> resources,
        std::vector<DescriptorBinding> descriptors = {})
        : m_resources(std::move(resources)), m_descriptors(std::move(descriptors)) {
        for (const auto& binding : m_resources)
            if (!binding.resource.GetHandle().valid() || !binding.owner)
                throw std::invalid_argument("Unowned recording resource");
        for (const auto& binding : m_descriptors)
            if (!binding.owner) throw std::invalid_argument("Unowned recording descriptor");
    }
    rhi::Resource Resolve(PreparedResourceReference ref) const { return m_resources.at(ref.slot).resource; }
    std::shared_ptr<const void> Owner(PreparedResourceReference ref) const {
        return m_resources.at(ref.slot).owner;
    }
    rhi::DescriptorSlot Resolve(PreparedDescriptorReference ref) const { return m_descriptors.at(ref.slot).descriptor; }
    const std::vector<ResourceBinding>& Resources() const { return m_resources; }
    const std::vector<DescriptorBinding>& Descriptors() const { return m_descriptors; }
private:
    std::vector<ResourceBinding> m_resources;
    std::vector<DescriptorBinding> m_descriptors;
};

struct FramePreparationContext {
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
    // Installed automatically by TypedRenderGraphPass for this one packet.
    // Authors receive compact slot references; PreparedPass owns the captured
    // immutable program generations until terminal lifecycle state.
    std::shared_ptr<PreparedDependencyCollector> dependencyCollector;
    // Per-pass permission map assembled from that pass's declarations. This
    // lets preparation capture a stable binding slot without retaining a
    // Resource wrapper or knowing anything about graph indices.
    std::shared_ptr<const std::unordered_map<uint64_t, uint32_t>> resourceSlots;
    // Installed by TypedRenderGraphPass for the duration of Prepare. This
    // hides the legacy registry-view descriptor helper from pass authors and
    // keeps descriptor indices coherent with the captured pipeline payload.
    std::function<std::vector<unsigned int>(const PipelineResources&)>
        captureDescriptorIndices;

    CapturedPipeline CapturePipeline(const PipelineState& pipeline,
        BackendInstanceId backend = BackendInstanceId::Primary) const {
        auto owner = pipeline.GetPayload(backend);
        if (!owner || !owner->pso) throw std::invalid_argument("Pipeline has no immutable payload");
        return {owner->pso.Get().GetHandle(), std::move(owner)};
    }

    PreparedProgramReference CaptureProgram(const PipelineState& pipeline,
        BackendInstanceId backend = BackendInstanceId::Primary) const {
        if (!dependencyCollector) throw std::logic_error("Program capture is unavailable outside typed preparation");
        return dependencyCollector->Capture(pipeline, backend);
    }

    PreparedProgramReference CaptureProgram(
        std::shared_ptr<const rhi::PipelinePtr> pipeline) const {
        if (!dependencyCollector) throw std::logic_error("Program capture is unavailable outside typed preparation");
        return dependencyCollector->Capture(std::move(pipeline));
    }

    PreparedProgramBinding CaptureProgramBinding(
        std::shared_ptr<const rhi::PipelinePtr> pipeline,
        const PipelineResources& resources) const {
        if (!dependencyCollector || !captureDescriptorIndices)
            throw std::logic_error("Program binding capture is unavailable outside typed preparation");
        auto indices = captureDescriptorIndices(resources);
        return {dependencyCollector->Capture(std::move(pipeline)), std::move(indices)};
    }

    PreparedProgramBinding CaptureProgramBinding(const PipelineState& pipeline,
        BackendInstanceId backend = BackendInstanceId::Primary) const {
        if (!dependencyCollector || !captureDescriptorIndices)
            throw std::logic_error("Program binding capture is unavailable outside typed preparation");
        auto payload = pipeline.GetPayload(backend);
        if (!payload || !payload->pso)
            throw std::invalid_argument("Pipeline has no immutable payload");
        return CaptureProgramBinding(std::move(payload));
    }

    PreparedProgramBinding CaptureProgramBinding(
        std::shared_ptr<const PipelineStatePayload> payload) const {
        if (!dependencyCollector || !captureDescriptorIndices)
            throw std::logic_error("Program binding capture is unavailable outside typed preparation");
        if (!payload || !payload->pso)
            throw std::invalid_argument("Pipeline has no immutable payload");
        auto indices = captureDescriptorIndices(payload->pipelineResources);
        return {dependencyCollector->Capture(std::move(payload)), std::move(indices)};
    }

    PreparedWorkGraphReference CaptureWorkGraph(
        std::shared_ptr<const rhi::WorkGraphPtr> owner) const {
        if (!dependencyCollector)
            throw std::logic_error("Work-graph capture is unavailable outside typed preparation");
        return dependencyCollector->CaptureWorkGraph(std::move(owner));
    }

    PreparedResourceReference CaptureResource(uint64_t globalResourceID) const {
        if (!resourceSlots) throw std::logic_error("Resource capture is unavailable outside owned preparation");
        const auto found = resourceSlots->find(globalResourceID);
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
            if (const auto found = resourceSlots->find(binding.globalResourceID);
                found != resourceSlots->end()) return {found->second};
            if (const auto found = resourceSlots->find(binding.registryResourceID);
                found != resourceSlots->end()) return {found->second};
        }
        return CaptureResource(binding.globalResourceID);
    }

    PreparedDescriptorReference CaptureDescriptor(ResourceBindingToken binding,
        rhi::DescriptorSlot descriptor) const {
        if (!dependencyCollector || !bindings)
            throw std::logic_error("Descriptor capture is unavailable outside typed preparation");
        const auto resource = CaptureResource(binding);
        return dependencyCollector->CaptureDescriptor(descriptor, bindings->Owner(resource));
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

class RecordingContext {
public:
    RecordingContext(rhi::CommandList commands, std::shared_ptr<const FrozenExecutionBindings> bindings,
        std::shared_ptr<const std::vector<ExternalDescriptorBindingValue>> externalBindings = {})
        : m_commands(commands), m_bindings(std::move(bindings)), m_externalBindings(std::move(externalBindings)) {
        if (!m_commands || !m_bindings) throw std::invalid_argument("Incomplete recording context");
    }
    rhi::CommandList& Commands() { return m_commands; }
    rhi::Resource Resolve(PreparedResourceReference ref) const { return m_bindings->Resolve(ref); }
    rhi::DescriptorSlot Resolve(PreparedDescriptorReference ref) const {
        if (m_dependencies && ref.slot < m_dependencies->DescriptorCount())
            return m_dependencies->Resolve(ref);
        return m_bindings->Resolve(ref);
    }
    rhi::DescriptorSlot Resolve(ExternalBindingKey key) const {
        if (!m_externalBindings) throw std::out_of_range("Missing external binding table");
        for (const auto& binding : *m_externalBindings)
            if (binding.key == key && binding.descriptor.heap.valid()) return binding.descriptor;
        throw std::out_of_range("Missing external descriptor binding");
    }
    rhi::PipelineHandle Resolve(PreparedProgramReference ref) const {
        if (!m_dependencies) throw std::out_of_range("Missing prepared dependency snapshot");
        return m_dependencies->Resolve(ref).pipeline;
    }
    rhi::WorkGraphHandle Resolve(PreparedWorkGraphReference ref) const {
        if (!m_dependencies) throw std::out_of_range("Missing prepared dependency snapshot");
        return m_dependencies->Resolve(ref);
    }
    void SetPreparedDependencies(std::shared_ptr<const PreparedDependencySnapshot> dependencies) {
        m_dependencies = std::move(dependencies);
    }
private:
    rhi::CommandList m_commands;
    std::shared_ptr<const FrozenExecutionBindings> m_bindings;
    std::shared_ptr<const std::vector<ExternalDescriptorBindingValue>> m_externalBindings;
    std::shared_ptr<const PreparedDependencySnapshot> m_dependencies;
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
        std::shared_ptr<const PreparedDependencySnapshot> dependencies = {}) {
        static_assert(requires(const Data& value, RecordingContext& context) {
            { Derived::Record(value, context) } -> std::same_as<void>;
        }, "Typed passes require static Record(const FrameData&, RecordingContext&)");
        PreparedPass result;
        auto storage = std::make_shared<TypedStorage<Derived, Data>>(std::move(data));
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
    explicit operator bool() const { return static_cast<bool>(m_storage); }
    bool IsConsumed() const { return m_storage && m_storage->state.load(std::memory_order_acquire) != LifecycleState::Prepared; }
    bool IsWorkerSafe() const noexcept { return m_storage && m_storage->workerSafe; }
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
    }
    bool Abandon(AbandonReason reason) const {
        if (!m_storage) return false;
        auto current = m_storage->state.load(std::memory_order_acquire);
        while (current == LifecycleState::Prepared || current == LifecycleState::Recorded) {
            if (m_storage->state.compare_exchange_weak(current, LifecycleState::Abandoned)) {
                if (m_storage->dependencies) m_storage->dependencies->Abandoned(reason);
                m_storage->Abandon(reason);
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
        mutable std::atomic<LifecycleState> state{LifecycleState::Prepared};
        std::shared_ptr<const PreparedDependencySnapshot> dependencies;
        std::vector<ExternalTimelinePoint> externalSignals;
        mutable std::string debugName;
        bool workerSafe = false;
    };
    template<class Data> struct Storage final : StorageBase {
        Storage(Data value, void (*fn)(const Data&, RecordingContext&), void (*onSubmitted)(const Data&) = nullptr)
            : data(std::move(value)), record(fn), submitted(onSubmitted) {}
        void Record(RecordingContext& context) const override { record(data, context); }
        void CommitSubmitted(SubmissionContext) const override { if (submitted) submitted(data); }
        const Data data;
        void (*const record)(const Data&, RecordingContext&);
        void (*const submitted)(const Data&);
    };
    template<class Derived, class Data> struct TypedStorage final : StorageBase {
        explicit TypedStorage(Data value) : data(std::move(value)) {}
        void Record(RecordingContext& context) const override { Derived::Record(data, context); }
        void CommitSubmitted(SubmissionContext context) const override {
            if constexpr (requires { Derived::Submitted(data, context); }) Derived::Submitted(data, context);
        }
        void CommitCompleted(CompletionContext context) const override {
            if constexpr (requires { Derived::Completed(data, context); }) Derived::Completed(data, context);
        }
        void Abandon(AbandonReason reason) const override {
            if constexpr (requires { Derived::Abandoned(data, reason); }) Derived::Abandoned(data, reason);
        }
        const Data data;
    };
    std::shared_ptr<const StorageBase> m_storage;
};

} // namespace org
