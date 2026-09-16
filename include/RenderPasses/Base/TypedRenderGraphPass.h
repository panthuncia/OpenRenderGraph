#pragma once

#include "RenderPasses/Base/RenderPass.h"

#include <concepts>
#include <type_traits>
#include <utility>
#include <optional>
#include <algorithm>
#include <BasicTelemetry/Telemetry.h>
#include <BasicTelemetry/Tracy.h>

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
struct LegacyPassBindings {};
struct NoPassRecordingRecipe {};

template<class Derived, OwnedPassFrameData FrameData = EmptyPassFrameData,
    class Bindings = LegacyPassBindings, class Recipe = NoPassRecordingRecipe>
class TypedRenderGraphPass : public RenderPass {
public:
    using PreparedData = FrameData;

    bool UsesTypedPreparation() const noexcept final { return true; }

    PreparedPass PrepareFrame(FramePreparationContext& context) final {
        if constexpr (!std::same_as<Recipe, NoPassRecordingRecipe>)
            return PrepareRecipeInvocation(context);
        else {
        auto collector = std::make_shared<PreparedDependencyCollector>(
            context.borrowedDependencies);
        auto typedContext = context;
        typedContext.dependencyCollector = collector;
        typedContext.captureDescriptorIndices = [this](const PipelineResources& resources) {
            return this->CaptureResourceDescriptorIndices(resources);
        };
        if constexpr (!std::same_as<Bindings, LegacyPassBindings>) {
            static_assert(std::copy_constructible<Bindings>, "Declared bindings must be values");
            if (!m_declaredBindings || !context.bindings || !context.resourceSlots)
                throw std::logic_error("Declared pass prepared without resolved declarations");
            // The permission table is built once for this owned frame and is
            // already immutable. FrozenExecutionBindings validated the complete
            // resource table at construction, and CaptureResource validates the
            // declarations actually used by this pass. Re-resolving every alias
            // of every declaration here made preparation scale with declaration
            // volume even when a pass captured only a handful of resources.
            auto slots = context.resourceSlots;
            auto data = [&]() -> FrameData {
                if constexpr (requires(const Derived& pass, const Bindings& bindings,
                    const PassPrepareContext& prepare) {
                    { pass.Prepare(bindings, prepare) } -> std::same_as<FrameData>;
                }) return static_cast<const Derived*>(this)->Prepare(*m_declaredBindings, typedContext);
                else {
                    static_assert(std::same_as<FrameData, EmptyPassFrameData>,
                        "Declared passes with data require const Prepare(bindings, context)");
                    return {};
                }
            }();
            return PreparedPass::FromTyped<DeclaredRecorder>(
                DeclaredFrame{*m_declaredBindings, std::move(data), context.bindings, std::move(slots)},
                std::move(*collector).Freeze(), context.invocationArena);
        } else if constexpr (requires(Derived& pass, const PassPrepareContext& prepare) {
            { pass.Prepare(prepare) } -> std::same_as<FrameData>;
        }) {
            static_assert(requires(const FrameData& data, PassRecordContext& record) {
                { Derived::Record(data, record) } -> std::same_as<void>;
            }, "Prepared passes require static void Record(const FrameData&, PassRecordContext&)");
            auto data = static_cast<Derived*>(this)->Prepare(typedContext);
            return PreparedPass::FromTyped<Derived>(std::move(data), std::move(*collector).Freeze(), context.invocationArena);
        } else {
            static_assert(std::same_as<FrameData, EmptyPassFrameData>,
                "Passes with prepared data require FrameData Prepare(const PassPrepareContext&)");
            static_assert(requires(PassRecordContext& record) {
                { Derived::Record(record) } -> std::same_as<void>;
            }, "Direct passes require static void Record(PassRecordContext&)");
            return PreparedPass::FromTyped<DirectRecorder>(EmptyPassFrameData{},
                std::move(*collector).Freeze(), context.invocationArena);
        }
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
    struct RecordingRecipe {
        Recipe data;
        std::shared_ptr<const PreparedDependencySnapshot> dependencies;
        std::vector<uint64_t> revision;
    };
    std::shared_ptr<const RecordingRecipe> m_recipe;
    struct RecipeInvocation {
        std::shared_ptr<const RecordingRecipe> recipe;
        FrameData data;
        std::shared_ptr<const FrozenExecutionBindings> resources;
        std::shared_ptr<const FramePreparationContext::ResourceSlots> slots;
    };
    struct RecipeRecorder {
        static void Record(const RecipeInvocation& frame, PassRecordContext& context) {
            auto scoped = context.WithDeclaredResources(frame.resources, frame.slots);
            scoped.SetPreparedDependencies(frame.recipe->dependencies);
            Derived::Record(frame.recipe->data, frame.data, scoped);
        }
    };
    PreparedPass PrepareRecipeInvocation(FramePreparationContext& context) {
        if (!context.bindings || !context.resourceSlots
            || (!std::same_as<Bindings, LegacyPassBindings> && !m_declaredBindings))
            throw std::logic_error("Recipe pass has no resolved declarations");
        // Alias IDs, not compiler enumeration, define the recipe's slot layout.
        auto slots = std::make_shared<FramePreparationContext::ResourceSlots>(*context.resourceSlots);
        std::sort(slots->begin(), slots->end());
        slots->erase(std::unique(slots->begin(), slots->end()), slots->end());
        std::unordered_map<uint32_t, uint32_t> localSlots;
        std::vector<uint32_t> map;
        std::vector<uint64_t> revision = static_cast<const Derived*>(this)->RecipeRevision(context);
        revision.insert(revision.begin(), revision.size());
        revision.push_back(slots->size());
        for (auto& [id, slot] : *slots) {
            const auto [found, inserted] = localSlots.emplace(slot, static_cast<uint32_t>(map.size()));
            if (inserted) {
                map.push_back(slot);
                const auto& binding = context.bindings->Resources().at(slot);
                const auto handle = binding.resource.GetHandle();
                revision.push_back((uint64_t{handle.generation} << 32) | handle.index);
                revision.push_back(reinterpret_cast<uintptr_t>(binding.views.get()));
            }
            slot = found->second;
            revision.push_back(id);
            revision.push_back(slot);
        }
        auto local = context;
        local.resourceSlots = slots;
        local.bindings = FrozenExecutionBindings::WithResourceMap(context.bindings, std::move(map));
        local.captureDescriptorIndices = [this](const PipelineResources& resources) {
            return this->CaptureResourceDescriptorIndices(resources);
        };
        if (!m_recipe || m_recipe->revision != revision) {
            BT_ZONE_SCOPE("ORG.Execution.BuildRecordingRecipe");
            local.borrowedDependencies = false; // Recipes survive publication rotation in both policies.
            local.dependencyCollector = std::make_shared<PreparedDependencyCollector>();
            auto data = [&] {
                if constexpr (std::same_as<Bindings, LegacyPassBindings>)
                    return static_cast<const Derived*>(this)->BuildRecipe(local);
                else return static_cast<const Derived*>(this)->BuildRecipe(*m_declaredBindings, local);
            }();
            // Retain exact recording generations, including embedded view
            // indices, without retaining publication semantic-consumer pins.
            for (const auto& [frameSlot, recipeSlot] : localSlots) {
                auto owner = local.bindings->Owner({recipeSlot});
                if (owner == context.bindings->PublicationRoot())
                    throw std::logic_error("Recording recipe requires exact ownership, not a frame publication root; slot="
                        + std::to_string(frameSlot));
                local.dependencyCollector->Retain(std::move(owner));
                local.dependencyCollector->Retain(context.bindings->Resources().at(frameSlot).views);
            }
            auto dependencies = std::move(*local.dependencyCollector).Freeze();
            if (dependencies && dependencies->HasLifecycleEffects())
                throw std::logic_error("Recording recipes cannot own frame lifecycle reservations");
            auto replacement = std::make_shared<const RecordingRecipe>(
                RecordingRecipe{std::move(data), std::move(dependencies), std::move(revision)});
            auto retired = std::exchange(m_recipe, std::move(replacement));
            if (retired && context.retireOwnership) context.retireOwnership(std::move(retired));
            basic_telemetry::AddCounter("ORG.Execution.RecordingRecipeBuilds");
        } else basic_telemetry::AddCounter("ORG.Execution.RecordingRecipeReuses");
        BT_ZONE_SCOPE("ORG.Execution.PrepareInvocation");
        local.borrowedDependencies = context.borrowedDependencies;
        local.dependencyCollector = std::make_shared<PreparedDependencyCollector>(context.borrowedDependencies);
        auto data = [&] {
            if constexpr (std::same_as<Bindings, LegacyPassBindings>)
                return static_cast<const Derived*>(this)->PrepareInvocation(m_recipe->data, local);
            else return static_cast<const Derived*>(this)->PrepareInvocation(m_recipe->data, *m_declaredBindings, local);
        }();
        return PreparedPass::FromTyped<RecipeRecorder>(
            RecipeInvocation{m_recipe, std::move(data), local.bindings, std::move(slots)},
            std::move(*local.dependencyCollector).Freeze(), context.invocationArena);
    }
    std::optional<Bindings> m_declaredBindings;
    struct DeclaredFrame {
        Bindings bindings;
        FrameData data;
        std::shared_ptr<const FrozenExecutionBindings> resources;
        std::shared_ptr<const FramePreparationContext::ResourceSlots> slots;
    };
    struct DeclaredRecorder {
        static void Record(const DeclaredFrame& frame, PassRecordContext& context) {
            auto scoped = context.WithDeclaredResources(frame.resources, frame.slots);
            if constexpr (std::same_as<FrameData, EmptyPassFrameData>)
                Derived::Record(frame.bindings, scoped);
            else
                Derived::Record(frame.bindings, frame.data, scoped);
        }
    };
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
        if constexpr (std::same_as<Bindings, LegacyPassBindings>) {
            static_assert(requires(Derived& pass, PassBuilder& declaration) {
                { pass.Declare(declaration) } -> std::same_as<void>;
            }, "Typed passes require void Declare(PassBuilder&)");
            static_cast<Derived*>(this)->Declare(*builder);
        } else {
            static_assert(requires(Derived& pass, PassBuilder& declaration) {
                { pass.Declare(declaration) } -> std::same_as<Bindings>;
            }, "Declared passes require Bindings Declare(PassBuilder&)");
            m_declaredBindings = static_cast<Derived*>(this)->Declare(*builder);
        }
    }
};

} // namespace org
