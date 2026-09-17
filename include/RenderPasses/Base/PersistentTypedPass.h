#pragma once

#include "Render/RenderGraph/PersistentGraph.h"
#include "Render/PreparedPass.h"
#include <concepts>
#include <memory>
#include <stdexcept>
#include <utility>

namespace org::persistent {

// Used only by explicit graph edits, never by frame preparation. Resource slots
// are producer-owned logical identities; this builder performs no resolver polling.
class PassDeclaration {
public:
    PassDeclaration(GraphEditTransaction& edit, PassId pass) : m_edit(edit), m_pass(pass) {}
    BindingToken Resource(ResourceSlotId slot, experimental::CompileResourceState state,
        experimental::CompileRange range = {}) {
        return m_edit.Declare(m_pass,slot,range,state);
    }
    BindingToken Dependency(ResourceSlotId slot, bool write) { return m_edit.DeclareDependency(m_pass,slot,write); }
    ViewToken View(BindingToken binding, BindlessViewRequest request = {}) { return m_edit.DeclareView(binding,request); }
    void Postcondition(BindingToken binding, experimental::CompileResourceState state,
        experimental::CompileRange range = {}) { m_edit.DeclarePostcondition(binding,range,state); }
    void Group(ResourceGroupId group, experimental::CompileResourceState state, uint32_t phase,
        std::optional<experimental::CompileRange> range = {}) {
        m_edit.DeclareGroupAccess(m_pass,group,state,phase,range);
    }
    PassId Id() const noexcept { return m_pass; }
private:
    GraphEditTransaction& m_edit;
    PassId m_pass;
};

// Declare and immutable program preparation happen once per explicit edit.
// Fresh invocation preparation and static recording use the production packet
// contract. No author/pass/manager pointer or generic recipe revision is retained.
template<class Derived>
class TypedPassExecutable {
public:
    using Bindings = typename Derived::Bindings;
    using ProgramInterface = typename Derived::ProgramInterface;
    using Invocation = typename Derived::Invocation;
    template<class BuildContext>
    static TypedPassExecutable Build(GraphEditTransaction& edit, experimental::CompilePass scheduling,
        Derived& author, const BuildContext& context) {
        try {
            const auto pass = edit.AddPass(std::move(scheduling));
            PassDeclaration declaration(edit,pass);
            auto bindings = author.Declare(declaration);
            auto program = author.BuildProgramInterface(bindings,context);
            auto definition = std::make_shared<const Definition>(Definition{std::move(bindings),std::move(program)});
            edit.SetPassRecordingInterface(pass,definition);
            return TypedPassExecutable(pass,std::move(definition));
        } catch (...) {
            edit.Abort();
            throw;
        }
    }
    // For program changes compatible with the existing declaration/binding
    // layout. Changes to accesses or binding layout require a structural edit.
    template<class BuildContext>
    TypedPassExecutable RebuildProgramInterface(GraphEditTransaction& edit,
        Derived& author, const BuildContext& context) const {
        try {
            auto program = author.BuildProgramInterface(m_definition->bindings,context);
            auto definition = std::make_shared<const Definition>(Definition{m_definition->bindings,std::move(program)});
            edit.SetPassRecordingInterface(m_pass,definition);
            return TypedPassExecutable(m_pass,std::move(definition));
        } catch (...) {
            edit.Abort();
            throw;
        }
    }
    template<class Context>
    PreparedPass PrepareInvocation(const SelectedPublication& selected, const Context& context,
        std::shared_ptr<PreparedInvocationArena> arena = {}) const {
        if (m_pass.index >= selected.logical->passSlots.size())
            throw std::logic_error("Persistent pass is absent from selected publication");
        const auto& pass = selected.logical->passSlots[m_pass.index];
        if (!pass.active || pass.generation != m_pass.generation || pass.recordingInterface != m_definition)
            throw std::logic_error("Persistent pass interface does not match selected publication");
        auto invocation = Derived::PrepareInvocation(m_definition->program,m_definition->bindings,context);
        return PreparedPass::FromTyped<Recorder>(Packet{m_definition,&selected,std::move(invocation)},{},std::move(arena));
    }
    PassId Id() const noexcept { return m_pass; }
private:
    struct Definition { Bindings bindings; ProgramInterface program; };
    struct Packet {
        std::shared_ptr<const Definition> definition;
        const SelectedPublication* selected; // Identity only; the frame recorder retains this root.
        Invocation invocation;
    };
    struct Recorder {
        static void Record(const Packet& packet, RecordingContext& context) {
            if (context.PersistentPublication() != packet.selected)
                throw std::logic_error("Persistent invocation recorded against a different publication");
            Derived::Record(packet.definition->program,packet.definition->bindings,packet.invocation,context);
        }
        static void Submitted(const Packet& packet, SubmissionContext context) {
            if constexpr (requires { Derived::Submitted(packet.invocation,context); })
                Derived::Submitted(packet.invocation,context);
        }
        static void Completed(const Packet& packet, CompletionContext context) {
            if constexpr (requires { Derived::Completed(packet.invocation,context); })
                Derived::Completed(packet.invocation,context);
        }
        static void Abandoned(const Packet& packet, AbandonReason reason) {
            if constexpr (requires { Derived::Abandoned(packet.invocation,reason); })
                Derived::Abandoned(packet.invocation,reason);
        }
    };
    TypedPassExecutable(PassId pass, std::shared_ptr<const Definition> definition)
        : m_pass(pass), m_definition(std::move(definition)) {}
    PassId m_pass;
    std::shared_ptr<const Definition> m_definition;
};

} // namespace org::persistent
