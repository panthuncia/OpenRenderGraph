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
template<class Derived, OwnedPassFrameData FrameData>
class TypedRenderGraphPass : public RenderPass {
public:
    using PreparedData = FrameData;

    PreparedPass PrepareFrame(FramePreparationContext& context) final {
        static_assert(requires(Derived& pass, const PassPrepareContext& prepare) {
            { pass.Prepare(prepare) } -> std::same_as<FrameData>;
        }, "Typed passes require FrameData Prepare(const PassPrepareContext&)");
        static_assert(requires(const FrameData& data, PassRecordContext& record) {
            { Derived::Record(data, record) } -> std::same_as<void>;
        }, "Typed passes require static void Record(const FrameData&, PassRecordContext&)");
        return PreparedPass::FromTyped<Derived>(
            static_cast<Derived*>(this)->Prepare(context));
    }

    // Synchronous mode consumes the identical prepared packet. Submission and
    // completion hooks are owned by the graph submission path; this method
    // performs command recording only.
    PassReturn Execute(PassExecutionContext& execution) final {
        auto bindings = std::make_shared<const FrozenExecutionBindings>(
            std::vector<FrozenExecutionBindings::ResourceBinding>{});
        FramePreparationContext preparation{
            .frameIndex = execution.frameIndex,
            .deltaTime = execution.deltaTime,
            .bindings = bindings,
            .preparationData = execution.hostData,
            .admissionData = execution.hostData,
        };
        auto packet = PrepareFrame(preparation);
        auto externalBindings = std::make_shared<const std::vector<ExternalDescriptorBindingValue>>(
            execution.externalDescriptorBindings);
        RecordingContext recording(execution.commandList, std::move(bindings), std::move(externalBindings));
        packet.Record(recording);
        PassReturn result;
        result.externalSignalsAfterCompletion = packet.ExternalSignalsAfterCompletion();
        return result;
    }

    void Setup() final {
        if constexpr (requires(Derived& pass) { pass.Initialize(); })
            static_cast<Derived*>(this)->Initialize();
    }
    void Cleanup() final {
        if constexpr (requires(Derived& pass) { pass.ShutdownPass(); })
            static_cast<Derived*>(this)->ShutdownPass();
    }

protected:
    void DeclareResourceUsages(RenderPassBuilder* builder) final {
        static_assert(requires(Derived& pass, PassBuilder& declaration) {
            { pass.Declare(declaration) } -> std::same_as<void>;
        }, "Typed passes require void Declare(PassBuilder&)");
        static_cast<Derived*>(this)->Declare(*builder);
    }
};

} // namespace org
