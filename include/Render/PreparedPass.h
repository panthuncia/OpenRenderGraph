#pragma once

#include <atomic>
#include <concepts>
#include <memory>
#include <stdexcept>
#include <utility>
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
struct PreparedDescriptorReference { uint32_t slot = 0; };
struct CapturedPipeline {
    rhi::PipelineHandle pipeline{};
    std::shared_ptr<const PipelineStatePayload> owner;
    explicit operator bool() const noexcept { return pipeline.valid() && static_cast<bool>(owner); }
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
    rhi::DescriptorSlot Resolve(PreparedDescriptorReference ref) const { return m_descriptors.at(ref.slot).descriptor; }
    const std::vector<ResourceBinding>& Resources() const { return m_resources; }
    const std::vector<DescriptorBinding>& Descriptors() const { return m_descriptors; }
private:
    std::vector<ResourceBinding> m_resources;
    std::vector<DescriptorBinding> m_descriptors;
};

struct FramePreparationContext {
    uint32_t frameIndex = 0;
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

    CapturedPipeline CapturePipeline(const PipelineState& pipeline,
        BackendInstanceId backend = BackendInstanceId::Primary) const {
        auto owner = pipeline.GetPayload(backend);
        if (!owner || !owner->pso) throw std::invalid_argument("Pipeline has no immutable payload");
        return {owner->pso.Get().GetHandle(), std::move(owner)};
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
    rhi::DescriptorSlot Resolve(PreparedDescriptorReference ref) const { return m_bindings->Resolve(ref); }
    rhi::DescriptorSlot Resolve(ExternalBindingKey key) const {
        if (!m_externalBindings) throw std::out_of_range("Missing external binding table");
        for (const auto& binding : *m_externalBindings)
            if (binding.key == key && binding.descriptor.heap.valid()) return binding.descriptor;
        throw std::out_of_range("Missing external descriptor binding");
    }
private:
    rhi::CommandList m_commands;
    std::shared_ptr<const FrozenExecutionBindings> m_bindings;
    std::shared_ptr<const std::vector<ExternalDescriptorBindingValue>> m_externalBindings;
};

// One owned packet per submitted frame. Copies share the single-consumption
// state; no mutable pass object is implicitly retained by the recording thunk.
class PreparedPass {
public:
    PreparedPass() = default; // Explicit unsupported/legacy preparation result.
    static PreparedPass NoOp() {
        return Make(uint8_t{}, +[](const uint8_t&, RecordingContext&) {});
    }
    static PreparedPass NoOpWithExternalSignals(std::vector<ExternalTimelinePoint> signals) {
        PreparedPass result;
        auto storage = std::make_shared<Storage<uint8_t>>(
            uint8_t{}, +[](const uint8_t&, RecordingContext&) {});
        storage->externalSignals = std::move(signals);
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
    // Canonical typed-pass entry point. Derived owns no runtime instance
    // dependency: recording and lifecycle hooks are static and operate only on
    // the immutable frame data captured by Prepare.
    template<class Derived, class Data>
    static PreparedPass FromTyped(Data data) {
        static_assert(requires(const Data& value, RecordingContext& context) {
            { Derived::Record(value, context) } -> std::same_as<void>;
        }, "Typed passes require static Record(const FrameData&, RecordingContext&)");
        PreparedPass result;
        result.m_storage = std::make_shared<TypedStorage<Derived, Data>>(std::move(data));
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
        m_storage->Record(context);
    }
    void CommitSubmitted(SubmissionContext context = {}) const {
        if (!m_storage) throw std::logic_error("Legacy pass has no submission effect");
        auto expected = LifecycleState::Recorded;
        if (!m_storage->state.compare_exchange_strong(expected, LifecycleState::Submitted))
            throw std::logic_error("Prepared pass submission transition is invalid");
        m_storage->CommitSubmitted(context);
    }
    void CommitCompleted(CompletionContext context = {}) const {
        if (!m_storage) throw std::logic_error("Missing prepared pass completion state");
        auto expected = LifecycleState::Submitted;
        if (!m_storage->state.compare_exchange_strong(expected, LifecycleState::Completed))
            throw std::logic_error("Prepared pass completion transition is invalid");
        m_storage->CommitCompleted(context);
    }
    bool Abandon(AbandonReason reason) const {
        if (!m_storage) return false;
        auto current = m_storage->state.load(std::memory_order_acquire);
        while (current == LifecycleState::Prepared || current == LifecycleState::Recorded) {
            if (m_storage->state.compare_exchange_weak(current, LifecycleState::Abandoned)) {
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
        std::vector<ExternalTimelinePoint> externalSignals;
        mutable std::string debugName;
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
