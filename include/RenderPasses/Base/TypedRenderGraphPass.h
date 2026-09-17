#pragma once
#include <typeinfo>
#include <spdlog/spdlog.h>
#include <atomic>
#include <span>
#include <unordered_map>
#include <string>

#include "RenderPasses/Base/RenderPass.h"

#include <concepts>
#include <type_traits>
#include <utility>
#include <optional>
#include <algorithm>
#include <string_view>
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
        PreparedDescriptorIndexCache descriptorIndices;
        typedContext.captureDescriptorIndices = [this, &descriptorIndices](const PipelineResources& resources) {
            return this->CaptureResourceDescriptorIndices(resources, &descriptorIndices);
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
        // Bindings the recipe embedded (alias id, FrozenExecutionBindings touch
        // flags) and registry descriptor indices it captured (hash, index).
        // Only these participate in the revision; a frame whose declared set
        // differs elsewhere reuses the recipe.
        std::vector<std::pair<uint64_t, uint8_t>> touched;
        std::vector<std::pair<size_t, uint32_t>> descriptorIndices;
        std::vector<std::string> descriptorNames; // parallel to descriptorIndices (diagnostics)
    };
    std::shared_ptr<const RecordingRecipe> m_recipe;
    // Pass-local slot numbering, stable for the pass's lifetime: recipe
    // references stay valid when the owner's permission table gains or loses
    // resources the recipe never embedded. The frame map is rebuilt only when
    // the owner hands over a different permission table object.
    std::unordered_map<uint64_t, uint32_t> m_localSlotById;
    std::shared_ptr<const FramePreparationContext::ResourceSlots> m_localSlots;
    std::shared_ptr<const FramePreparationContext::ResourceSlots> m_layoutSourceOwner;
    std::vector<uint32_t> m_frameMap;
    std::vector<uint64_t> m_recipeRevisionScratch;
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
    void LayoutRecipeSlots(const FramePreparationContext& context) {
        if (m_localSlots && m_layoutSourceOwner == context.resourceSlots) return;
        BT_ZONE_SCOPE("ORG.Execution.LayoutRecipeSlots");
        std::fill(m_frameMap.begin(), m_frameMap.end(), UINT32_MAX);
        auto local = std::make_shared<FramePreparationContext::ResourceSlots>();
        local->reserve(context.resourceSlots->size());
        for (const auto& [id, frameSlot] : *context.resourceSlots) {
            const auto entry = m_localSlotById.try_emplace(id, static_cast<uint32_t>(m_localSlotById.size())).first;
            if (entry->second >= m_frameMap.size()) m_frameMap.resize(entry->second + 1, UINT32_MAX);
            // A duplicate alias id keeps its first (lowest) frame slot, as Find does.
            if (m_frameMap[entry->second] != UINT32_MAX) continue;
            m_frameMap[entry->second] = frameSlot;
            local->emplace_back(id, entry->second);
        }
        std::sort(local->begin(), local->end());
        local->sorted = true;
        m_localSlots = std::move(local);
        m_layoutSourceOwner = context.resourceSlots;
    }
    // False when an embedded binding is no longer declared (forces a rebuild).
    bool ComputeRecipeRevision(const FramePreparationContext& context, std::span<const uint64_t> dependencyRevision,
        std::span<const std::pair<uint64_t, uint8_t>> touched, std::span<const std::pair<size_t, uint32_t>> descriptorIndices,
        std::vector<uint64_t>& revision) const {
        revision.assign(dependencyRevision.begin(), dependencyRevision.end());
        revision.insert(revision.begin(), revision.size());
        const auto& resources = context.bindings->Resources();
        for (const auto& [id, flags] : touched) {
            const auto found = context.resourceSlots->Find(id);
            if (found == context.resourceSlots->end()) return false;
            const auto& binding = resources.at(found->second);
            revision.push_back(id);
            if (flags & FrozenExecutionBindings::TouchHandle) {
                const auto handle = binding.resource.GetHandle();
                revision.push_back((uint64_t{handle.generation} << 32) | handle.index);
            }
            if (flags & FrozenExecutionBindings::TouchViews) revision.push_back(reinterpret_cast<uintptr_t>(binding.views.get()));
        }
        for (const auto& [hash, index] : descriptorIndices) {
            (void)index;
            revision.push_back(this->m_resourceDescriptorIndexHelper->GetResourceDescriptorIndex(hash, true));
        }
        return true;
    }
    PreparedPass PrepareRecipeInvocation(FramePreparationContext& context) {
        if (!context.bindings || !context.resourceSlots
            || (!std::same_as<Bindings, LegacyPassBindings> && !m_declaredBindings))
            throw std::logic_error("Recipe pass has no resolved declarations");
        LayoutRecipeSlots(context);
        auto& revision = m_recipeRevisionScratch;
        bool reusable = false;
        std::vector<uint64_t> dependencyRevision;
        {
            BT_ZONE_SCOPE("ORG.Execution.SelectRecipeBindings");
            dependencyRevision = static_cast<const Derived*>(this)->RecipeRevision(context);
            if (m_recipe) reusable = ComputeRecipeRevision(context, dependencyRevision, m_recipe->touched,
                m_recipe->descriptorIndices, revision) && revision == m_recipe->revision;
        }
        // Only captured registry descriptor indices changed: relocate them in a
        // copy of the recipe instead of rebuilding it, when the recipe type can.
        if constexpr (requires (Recipe& recipe, const DescriptorIndexRemap& remap) { RemapDescriptorIndices(recipe, remap); }) {
            if (!reusable && m_recipe && !m_recipe->descriptorIndices.empty()
                && revision.size() == m_recipe->revision.size()) {
                const auto prefix = revision.size() - m_recipe->descriptorIndices.size();
                if (std::equal(revision.begin(), revision.begin() + prefix, m_recipe->revision.begin())) {
                    BT_ZONE_SCOPE("ORG.Execution.RemapRecordingRecipe");
                    DescriptorIndexRemap remap;
                    bool relocatable = true;
                    auto indices = m_recipe->descriptorIndices;
                    for (size_t i = 0; i < indices.size() && relocatable; ++i) {
                        const auto fresh = static_cast<uint32_t>(revision[prefix + i]);
                        const auto previous = indices[i].second;
                        if (fresh == previous) continue;
                        // A now-missing optional binding, or two registrations that
                        // shared an index and diverged, cannot be relocated by value.
                        if (fresh == UINT32_MAX) { relocatable = false; break; }
                        if (const auto found = remap.find(previous); found != remap.end() && found->second != fresh) { relocatable = false; break; }
                        remap.emplace(previous, fresh);
                        indices[i].second = fresh;
                    }
                    if (relocatable) {
                        Recipe data = m_recipe->data;
                        RemapDescriptorIndices(data, remap);
                        auto replacement = std::make_shared<const RecordingRecipe>(RecordingRecipe{std::move(data), m_recipe->dependencies,
                            revision, m_recipe->touched, std::move(indices), m_recipe->descriptorNames});
                        auto retired = std::exchange(m_recipe, std::move(replacement));
                        if (retired && context.retireOwnership) context.retireOwnership(std::move(retired));
                        basic_telemetry::AddCounter("ORG.Execution.RecordingRecipeRemaps");
                        reusable = true;
                    }
                }
            }
        }
        auto local = context;
        local.resourceSlots = m_localSlots;
        local.bindings = FrozenExecutionBindings::WithResourceMap(context.bindings, m_frameMap);
        PreparedDescriptorIndexCache descriptorIndices;
        local.captureDescriptorIndices = [this, &descriptorIndices](const PipelineResources& resources) {
            return this->CaptureResourceDescriptorIndices(resources, &descriptorIndices);
        };
        local.captureDescriptorIndex = [this, &descriptorIndices](const ResourceIdentifier& binding, bool optional) {
            return descriptorIndices.Resolve(binding, optional, [this](const ResourceIdentifier& b, bool o) {
                return this->m_resourceDescriptorIndexHelper->GetResourceDescriptorIndex(b, o);
            });
        };
        if (!reusable) {
            BT_ZONE_SCOPE("ORG.Execution.BuildRecordingRecipe");
            const std::string_view reason = !m_recipe ? "bootstrap" :
                m_recipe->revision.size() < revision.front() + 1
                    || !std::equal(revision.begin(), revision.begin() + revision.front() + 1, m_recipe->revision.begin())
                        ? "pass dependencies" : "resource bindings";
            BT_ZONE_TEXT(reason.data(), reason.size());
            if (m_recipe && reason == "resource bindings") {
                // Diagnostic: name the first embedded binding that changed.
                static std::atomic<uint32_t> reported{0};
                if (reported.fetch_add(1) < 6) {
                    std::string changed;
                    size_t cursor = revision.front() + 1;
                    for (const auto& [id, flags] : m_recipe->touched) {
                        const size_t width = 1 + ((flags & FrozenExecutionBindings::TouchHandle) ? 1 : 0)
                            + ((flags & FrozenExecutionBindings::TouchViews) ? 1 : 0);
                        if (cursor + width > revision.size() || cursor + width > m_recipe->revision.size()) break;
                        if (!std::equal(revision.begin() + cursor, revision.begin() + cursor + width, m_recipe->revision.begin() + cursor)) {
                            const auto found = context.resourceSlots->Find(id);
                            const auto& binding = context.bindings->Resources().at(found->second);
                            changed += std::to_string(id) + "(" + (binding.views && binding.views->description.type != rhi::ResourceType::Unknown ? "" : "") + "flags=" + std::to_string(flags) + ") ";
                        }
                        cursor += width;
                    }
                    for (size_t i = 0; i < m_recipe->descriptorIndices.size(); ++i) {
                        const auto [hash, index] = m_recipe->descriptorIndices[i];
                        const auto fresh = this->m_resourceDescriptorIndexHelper->GetResourceDescriptorIndex(hash, true);
                        if (fresh != index) changed += "idx:" + (i < m_recipe->descriptorNames.size() ? m_recipe->descriptorNames[i] : std::string("?"))
                            + "(" + std::to_string(index) + "->" + std::to_string(fresh) + ") ";
                    }
                    basic_telemetry::AddCounter("ORG.Execution.RecipeRebuildDiagnostics");
                    spdlog::info("Recipe rebuild ({}): changed bindings: {}", typeid(Derived).name(), changed);
                }
            }
            local.borrowedDependencies = false; // Recipes survive publication rotation in both policies.
            local.dependencyCollector = std::make_shared<PreparedDependencyCollector>();
            std::vector<uint8_t> touchFlags(m_frameMap.size());
            local.bindings->TrackTouches(&touchFlags);
            auto data = [&] {
                BT_ZONE_SCOPE("ORG.Execution.BuildRecipeData");
                if constexpr (std::same_as<Bindings, LegacyPassBindings>)
                    return static_cast<const Derived*>(this)->BuildRecipe(local);
                else return static_cast<const Derived*>(this)->BuildRecipe(*m_declaredBindings, local);
            }();
            local.bindings->TrackTouches(nullptr);
            std::vector<std::pair<uint64_t, uint8_t>> touched;
            for (const auto& [id, localSlot] : *m_localSlots)
                if (touchFlags[localSlot]) touched.emplace_back(id, touchFlags[localSlot]);
            std::vector<std::pair<size_t, uint32_t>> capturedIndices(descriptorIndices.Indices().begin(), descriptorIndices.Indices().end());
            std::sort(capturedIndices.begin(), capturedIndices.end());
            std::vector<std::string> capturedNames;
            for (const auto& [hash, index] : capturedIndices) {
                const auto found = descriptorIndices.Names().find(hash);
                capturedNames.push_back(found != descriptorIndices.Names().end() ? found->second : std::string{});
            }
            // Retain exact recording generations of the embedded bindings,
            // including their view snapshots, without retaining publication
            // semantic-consumer pins.
            {
                BT_ZONE_SCOPE("ORG.Execution.RetainRecipeBindings");
                for (const auto& [id, flags] : touched) {
                    const auto frameSlot = context.resourceSlots->Find(id)->second;
                    auto owner = context.bindings->Owner({frameSlot});
                    if (owner == context.bindings->PublicationRoot())
                        throw std::logic_error("Recording recipe requires exact ownership, not a frame publication root; slot="
                            + std::to_string(frameSlot));
                    local.dependencyCollector->Retain(std::move(owner));
                    local.dependencyCollector->Retain(context.bindings->Resources().at(frameSlot).views);
                }
            }
            auto dependencies = std::move(*local.dependencyCollector).Freeze();
            if (dependencies && dependencies->HasLifecycleEffects())
                throw std::logic_error("Recording recipes cannot own frame lifecycle reservations");
            if (!ComputeRecipeRevision(context, dependencyRevision, touched, capturedIndices, revision))
                throw std::logic_error("Recording recipe embedded an undeclared binding");
            auto replacement = std::make_shared<const RecordingRecipe>(
                RecordingRecipe{std::move(data), std::move(dependencies), revision, std::move(touched), std::move(capturedIndices), std::move(capturedNames)});
            auto retired = std::exchange(m_recipe, std::move(replacement));
            if (retired && context.retireOwnership) context.retireOwnership(std::move(retired));
            basic_telemetry::AddCounter("ORG.Execution.RecordingRecipeBuilds");
            basic_telemetry::AddCounter(std::string("ORG.Execution.RecipeBuild.") + (reason == "resource bindings" ? "B." : reason == "pass dependencies" ? "D." : "S.") + typeid(Derived).name());
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
            RecipeInvocation{m_recipe, std::move(data), local.bindings, m_localSlots},
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
