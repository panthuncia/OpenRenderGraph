#pragma once

#include "RenderPasses/Base/RenderPass.h"

#include <concepts>
#include <type_traits>
#include <utility>

namespace org {

// The declaration implementation is currently the superset graphics builder;
// queue preference and declared operations determine the compiler queue class.
// Keeping this alias here gives pass authors one API while graph storage is
// collapsed from its historical graphics/compute/copy variants.
using PassBuilder = RenderPassBuilder;
using PassPrepareContext = FramePreparationContext;
using PassRecordContext = RecordingContext;

template<class Data>
concept OwnedPassFrameData = std::movable<Data>
    && !std::is_pointer_v<Data>
    && !std::is_reference_v<Data>;

// One author-facing pass contract for synchronous and asynchronous execution.
// Both modes call Prepare once and consume the resulting type-erased packet;
// neither compiler route calls a pass object.
struct EmptyPassFrameData {};

template<class Derived, OwnedPassFrameData FrameData = EmptyPassFrameData>
class TypedRenderGraphPass : public RenderPass {
public:
    using PreparedData = FrameData;

    bool UsesTypedPreparation() const noexcept final { return true; }

    PreparedPass PrepareFrame(FramePreparationContext& context) final {
        auto collector = std::make_shared<PreparedDependencyCollector>();
        auto typedContext = context;
        typedContext.dependencyCollector = collector;
        typedContext.captureDescriptorIndices = [this](const PipelineResources& resources) {
            return this->CaptureResourceDescriptorIndices(resources);
        };
        if constexpr (requires(Derived& pass, const PassPrepareContext& prepare) {
            { pass.Prepare(prepare) } -> std::same_as<FrameData>;
        }) {
            static_assert(requires(const FrameData& data, PassRecordContext& record) {
                { Derived::Record(data, record) } -> std::same_as<void>;
            }, "Prepared passes require static void Record(const FrameData&, PassRecordContext&)");
            auto data = static_cast<Derived*>(this)->Prepare(typedContext);
            return PreparedPass::FromTyped<Derived>(std::move(data), std::move(*collector).Freeze());
        } else {
            static_assert(std::same_as<FrameData, EmptyPassFrameData>,
                "Passes with prepared data require FrameData Prepare(const PassPrepareContext&)");
            static_assert(requires(PassRecordContext& record) {
                { Derived::Record(record) } -> std::same_as<void>;
            }, "Direct passes require static void Record(PassRecordContext&)");
            return PreparedPass::FromTyped<DirectRecorder>(EmptyPassFrameData{},
                std::move(*collector).Freeze());
        }
    }

    // Both Off and Async modes execute the owned packet produced above. Keep
    // the historical virtual solely as an ABI bridge and fail loudly if graph
    // execution ever attempts to re-enter a typed pass through it.
    PassReturn Execute(PassExecutionContext&) final {
        throw std::logic_error(
            "TypedRenderGraphPass reached the removed legacy Execute route");
    }

    void Setup() final {
        if constexpr (requires(Derived& pass) { pass.Initialize(); })
            static_cast<Derived*>(this)->Initialize();
    }
    void Cleanup() final {
        if constexpr (requires(Derived& pass) { pass.ShutdownPass(); })
            static_cast<Derived*>(this)->ShutdownPass();
    }

private:
    struct DirectRecorder {
        static void Record(const EmptyPassFrameData&, PassRecordContext& context) {
            Derived::Record(context);
        }
    };

protected:
    [[nodiscard]] PreparedProgramBinding CaptureProgramBinding(
        const PassPrepareContext& preparation,
        const PipelineState& pipeline,
        BackendInstanceId backend = BackendInstanceId::Primary) const {
        return preparation.CaptureProgramBinding(pipeline, backend);
    }

    void DeclareResourceUsages(RenderPassBuilder* builder) final {
        static_assert(requires(Derived& pass, PassBuilder& declaration) {
            { pass.Declare(declaration) } -> std::same_as<void>;
        }, "Typed passes require void Declare(PassBuilder&)");
        static_cast<Derived*>(this)->Declare(*builder);
    }
};

} // namespace org
