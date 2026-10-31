#pragma once
#include <concepts>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "RenderPasses/Base/RenderPass.h"

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

struct EmptyPassFrameData {};
struct NoPassBindings {};

// One author-facing pass contract for synchronous and asynchronous execution:
//
//   Bindings Declare(PassBuilder&);                                  // tokens and pipelines; immutable until rebuild
//   void Update(const UpdateExecutionContext&);                      // optional: resizes and other structural work
//   FrameData Prepare(const Bindings&, const PassPrepareContext&);        // optional: this frame's values
//   static void Record(const Bindings&, const FrameData&, PassRecordContext&);
//   void Revision(std::vector<uint64_t>&) const;                     // optional: opts into kept-recording reuse
//
// Update and Prepare run on the preparation owner, one frame at a time, and may change the pass. Record runs
// later, possibly on a worker and possibly replayed. It reads only its arguments and resolves
// everything else - declared views and resources, pipelines - through the context, against the frame the
// packet was prepared for. A pass without declared bindings writes `void Declare(PassBuilder&)`,
// `FrameData Prepare(const PassPrepareContext&)` and `static void Record(const FrameData&, PassRecordContext&)`
// (or `static void Record(PassRecordContext&)` without frame data).
//
// Revision appends what Record's output depends on beyond its bindings and the frame's resources: settings,
// pass state, pipeline generations. While it stays equal, a kept recording of the pass may be submitted again
// instead of recording anew. A pass without it is never part of a kept recording.
template<class Derived, OwnedPassFrameData FrameData = EmptyPassFrameData, class Bindings = NoPassBindings>
class TypedRenderGraphPass : public RenderPass {
public:
    using PreparedData = FrameData;

    bool UsesTypedPreparation() const noexcept final { return true; }

    static constexpr bool kHasRevision = requires(const Derived& pass, std::vector<uint64_t>& out) {
        { pass.Revision(out) } -> std::same_as<void>;
    };

    bool RevisionHash(uint64_t& hash) const final {
        if constexpr (kHasRevision) {
            std::vector<uint64_t> revision;
            static_cast<const Derived*>(this)->Revision(revision);
            uint64_t value = 1469598103934665603ull;
            for (const auto word : revision) value = (value ^ word) * 1099511628211ull;
            hash = value ^ revision.size();
            return true;
        } else {
            (void)hash;
            return false;
        }
    }

    PreparedPass PrepareFrame(FramePreparationContext& context) final {
        context.resourceUses = context.preparedResourceUses ? context.preparedResourceUses : this->ResourceUses();
        auto collector = std::make_shared<PreparedDependencyCollector>();
        auto typedContext = context;
        typedContext.dependencyCollector = collector;
        std::shared_ptr<const DeclaredRecordingScope> scope;
        if (context.bindings && context.resourceSlots) {
            auto declared = std::make_shared<DeclaredRecordingScope>();
            declared->layout = context.resourceUses;
            declared->views = context.ResolveDeclaredViewTable();
            declared->namedGroupViews = context.ResolveNamedGroupViews();
            declared->backendInstance = context.backendInstance;
            scope = std::move(declared);
        }
        if constexpr (!std::same_as<Bindings, NoPassBindings>) {
            static_assert(std::copy_constructible<Bindings>, "Declared bindings must be values");
            const auto declaredBindings = context.preparedBindings
                ? std::static_pointer_cast<const Bindings>(context.preparedBindings) : m_declaredBindings;
            if (!declaredBindings || !scope) throw std::logic_error("Declared pass prepared without resolved declarations");
            FrameData data = [&]() -> FrameData {
                if constexpr (requires(Derived& pass, const Bindings& bindings, const PassPrepareContext& prepare) {
                    { pass.Prepare(bindings, prepare) } -> std::same_as<FrameData>;
                }) return static_cast<Derived*>(this)->Prepare(*declaredBindings, typedContext);
                else {
                    static_assert(std::same_as<FrameData, EmptyPassFrameData>,
                        "Declared passes with frame data require FrameData Prepare(const Bindings&, const PassPrepareContext&)");
                    return {};
                }
            }();
            return PreparedPass::FromTyped<DeclaredRecorder>(
                DeclaredFrame{*declaredBindings, std::move(data), context.bindings, context.resourceSlots, std::move(scope)},
                std::move(*collector).Freeze(), context.invocationArena);
        } else {
            FrameData data = [&]() -> FrameData {
                if constexpr (requires(Derived& pass, const PassPrepareContext& prepare) {
                    { pass.Prepare(prepare) } -> std::same_as<FrameData>;
                }) return static_cast<Derived*>(this)->Prepare(typedContext);
                else {
                    static_assert(std::same_as<FrameData, EmptyPassFrameData>,
                        "Passes with frame data require FrameData Prepare(const PassPrepareContext&)");
                    return {};
                }
            }();
            return PreparedPass::FromTyped<UndeclaredRecorder>(
                UndeclaredFrame{std::move(data), context.bindings, context.resourceSlots, std::move(scope)},
                std::move(*collector).Freeze(), context.invocationArena);
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
    std::shared_ptr<const Bindings> m_declaredBindings;

    // A recording failure names the pass in the log - Record runs on a worker, away from anything that could tell
    // which - and propagates unchanged.
    template<class Fn>
    static void Named(Fn&& record) {
        try { record(); }
        catch (const std::exception& error) {
            spdlog::error("{} failed to record: {}", typeid(Derived).name(), error.what());
            throw;
        }
    }
    static RecordingContext Scoped(RecordingContext& context,
        const std::shared_ptr<const FrozenExecutionBindings>& resources,
        const std::shared_ptr<const FramePreparationContext::ResourceSlots>& slots,
        const std::shared_ptr<const DeclaredRecordingScope>& scope) {
        return resources && slots ? context.WithDeclaredResources(resources, slots, scope) : context;
    }
    struct DeclaredFrame {
        Bindings bindings;
        FrameData data;
        std::shared_ptr<const FrozenExecutionBindings> resources;
        std::shared_ptr<const FramePreparationContext::ResourceSlots> slots;
        std::shared_ptr<const DeclaredRecordingScope> scope;
    };
    struct DeclaredRecorder {
        static void Record(const DeclaredFrame& frame, PassRecordContext& context) {
            auto scoped = Scoped(context, frame.resources, frame.slots, frame.scope);
            Named([&] {
                if constexpr (requires(PassRecordContext& record) { Derived::Record(frame.bindings, frame.data, record); })
                    Derived::Record(frame.bindings, frame.data, scoped);
                else {
                    static_assert(std::same_as<FrameData, EmptyPassFrameData>,
                        "Declared passes require static void Record(const Bindings&, const FrameData&, PassRecordContext&)");
                    Derived::Record(frame.bindings, scoped);
                }
            });
        }
    };
    struct UndeclaredFrame {
        FrameData data;
        std::shared_ptr<const FrozenExecutionBindings> resources;
        std::shared_ptr<const FramePreparationContext::ResourceSlots> slots;
        std::shared_ptr<const DeclaredRecordingScope> scope;
    };
    struct UndeclaredRecorder {
        static void Record(const UndeclaredFrame& frame, PassRecordContext& context) {
            auto scoped = Scoped(context, frame.resources, frame.slots, frame.scope);
            Named([&] {
                if constexpr (requires(PassRecordContext& record) { Derived::Record(frame.data, record); })
                    Derived::Record(frame.data, scoped);
                else {
                    static_assert(std::same_as<FrameData, EmptyPassFrameData>,
                        "Passes require static void Record(const FrameData&, PassRecordContext&)");
                    Derived::Record(scoped);
                }
            });
        }
    };

protected:
    void DeclareResourceUsages(RenderPassBuilder* builder) final {
        if constexpr (std::same_as<Bindings, NoPassBindings>) {
            static_assert(requires(Derived& pass, PassBuilder& declaration) {
                { pass.Declare(declaration) } -> std::same_as<void>;
            }, "Typed passes require void Declare(PassBuilder&)");
            static_cast<Derived*>(this)->Declare(*builder);
        } else {
            static_assert(requires(Derived& pass, PassBuilder& declaration) {
                { pass.Declare(declaration) } -> std::same_as<Bindings>;
            }, "Declared passes require Bindings Declare(PassBuilder&)");
            m_declaredBindings = std::make_shared<const Bindings>(static_cast<Derived*>(this)->Declare(*builder));
            this->SetDeclaredBindings(m_declaredBindings);
        }
        this->SetResourceUses(GetResourceUses(*builder));
    }
};

} // namespace org
