#include "RenderPasses/Base/TypedRenderGraphPass.h"
#include "RenderPasses/Base/PersistentTypedPass.h"
#include "Render/PublicationBindingBundle.h"
#include "Render/RenderGraph/ExperimentalGraphCompiler.h"
#include "Render/RenderGraph/PersistentGraph.h"
#include "Render/RenderGraph/ExperimentalRhiExecution.h"
#include <cstdio>
#include <cstdlib>

#define CHECK(...) do { if (!(__VA_ARGS__)) { std::fprintf(stderr, "Recipe check failed at %d: %s\n", __LINE__, #__VA_ARGS__); std::abort(); } } while (false)
using namespace org;

struct Effect final : PreparedLifecycleEffect {
    mutable int submitted = 0, completed = 0, abandoned = 0;
    void Submitted(SubmissionContext) const override { ++submitted; }
    void Completed(CompletionContext) const override { ++completed; }
    void Abandoned(AbandonReason) const override { ++abandoned; }
};
struct Recipe { PreparedResourceReference resource; };
struct PersistentProbe {
    struct Invocation {
        std::shared_ptr<const org::persistent::SelectedPublication> publication;
        org::persistent::BindingToken resource;
        org::persistent::ViewToken view;
        uint32_t expectedDescriptor = 0;
    };
    static void Record(const Invocation& invocation, RecordingContext& context) {
        auto scoped = context.WithPersistentBindings(invocation.publication);
        CHECK(scoped.Resolve(invocation.resource).GetHandle().index == 11);
        CHECK(scoped.Resolve(invocation.view).index == invocation.expectedDescriptor);
    }
};
struct PersistentAuthor {
    struct Bindings { persistent::BindingToken resource; persistent::ViewToken view; };
    struct ProgramInterface { uint32_t generation; };
    struct Invocation { uint32_t descriptor, scalar; std::shared_ptr<Effect> effect; };
    persistent::ResourceSlotId slot;
    int declarations = 0, builds = 0;
    Bindings Declare(persistent::PassDeclaration& declaration) {
        ++declarations;
        const auto resource = declaration.Resource(slot,{2,0,1,false});
        return {resource,declaration.View(resource)};
    }
    ProgramInterface BuildProgramInterface(const Bindings&, const uint32_t& generation) {
        ++builds;
        if (!generation) throw std::runtime_error("Persistent program preparation failed");
        return {generation};
    }
    static Invocation PrepareInvocation(const ProgramInterface&, const Bindings&, const Invocation& invocation) {
        return invocation;
    }
    static void Record(const ProgramInterface& program, const Bindings& bindings,
        const Invocation& invocation, RecordingContext& context) {
        CHECK(program.generation == 17 || program.generation == 18);
        CHECK(context.Resolve(bindings.resource).GetHandle().index == 11);
        CHECK(context.Resolve(bindings.view).index == invocation.descriptor);
        CHECK(invocation.scalar != 0);
    }
    static void Submitted(const Invocation& data, SubmissionContext context) { data.effect->Submitted(context); }
    static void Completed(const Invocation& data, CompletionContext context) { data.effect->Completed(context); }
    static void Abandoned(const Invocation& data, AbandonReason reason) { data.effect->Abandoned(reason); }
};
struct Probe final : TypedRenderGraphPass<Probe, uint64_t, LegacyPassBindings, Recipe> {
    mutable int builds = 0, invocations = 0;
    uint64_t revision = 1;
    bool failBuild = false;
    std::shared_ptr<Effect> effect = std::make_shared<Effect>();
    void Declare(PassBuilder&) {}
    std::vector<uint64_t> RecipeRevision(const PassPrepareContext&) const { return {revision}; }
    Recipe BuildRecipe(const PassPrepareContext& context) const {
        ++builds;
        if (failBuild) throw std::runtime_error("Recipe build probe failure");
        return {context.CaptureResource(101)};
    }
    uint64_t PrepareInvocation(const Recipe& recipe, const PassPrepareContext& context) const {
        ++invocations;
        CHECK(context.ResolveCapturedResource(recipe.resource).GetHandle().index == 11);
        context.Reserve(effect);
        return context.frameNumber;
    }
    static void Record(const Recipe& recipe, const uint64_t&, RecordingContext& context) {
        CHECK(context.Resolve(recipe.resource).GetHandle().index == 11);
    }
};

int main() {
    const ResourceIdentifier descriptor{"recipe-descriptor"}, missing{"missing-descriptor"};
    uint32_t descriptorVersion = 11, resolutions = 0;
    auto resolveDescriptor = [&](const ResourceIdentifier& id, bool optional) -> uint32_t {
        ++resolutions;
        if (id.hash == missing.hash) {
            if (optional) return UINT32_MAX;
            throw std::invalid_argument("missing mandatory descriptor");
        }
        return descriptorVersion;
    };
    PreparedDescriptorIndexCache capture;
    CHECK(capture.Resolve(descriptor, false, resolveDescriptor) == 11);
    CHECK(capture.Resolve(descriptor, true, resolveDescriptor) == 11 && resolutions == 1);
    descriptorVersion = 12;
    PreparedDescriptorIndexCache nextCapture;
    CHECK(nextCapture.Resolve(descriptor, false, resolveDescriptor) == 12 && resolutions == 2);
    CHECK(nextCapture.Resolve(missing, true, resolveDescriptor) == UINT32_MAX);
    bool mandatoryMissingRejected = false;
    try { nextCapture.Resolve(missing, false, resolveDescriptor); }
    catch (const std::invalid_argument&) { mandatoryMissingRejected = true; }
    CHECK(mandatoryMissingRejected && resolutions == 4);

    auto owner = std::make_shared<int>(7);
    auto views = std::make_shared<BindlessResourceViews>();
    const auto resource = rhi::Resource{rhi::ResourceHandle{11,1}};
    auto bindings = std::make_shared<const FrozenExecutionBindings>(
        std::vector<FrozenExecutionBindings::ResourceBinding>{{resource, owner, views},
            {rhi::Resource{rhi::ResourceHandle{12,1}}, owner, views}});
    auto arena = std::make_shared<PreparedInvocationArena>();
    FramePreparationContext context;
    context.invocationArena = arena;
    context.bindings = bindings;
    context.resourceSlots = std::make_shared<const FramePreparationContext::ResourceSlots>(
        FramePreparationContext::ResourceSlots{{101,0},{102,1},{101,0}});
    Probe probe;
    {
        Probe resizedProbe;
        auto resized = context;
        auto initialPacket = resizedProbe.PrepareFrame(resized);
        std::vector<FrozenExecutionBindings::ResourceBinding> expanded(256,bindings->Resources()[1]);
        expanded.back() = bindings->Resources()[0];
        resized.bindings = std::make_shared<const FrozenExecutionBindings>(std::move(expanded));
        resized.resourceSlots = std::make_shared<const FramePreparationContext::ResourceSlots>(
            FramePreparationContext::ResourceSlots{{101,255},{102,0}});
        auto large = resizedProbe.PrepareFrame(resized);
        resized = context;
        auto shrunk = resizedProbe.PrepareFrame(resized);
        CHECK(resizedProbe.builds == 1 && resizedProbe.invocations == 3);
        CHECK(initialPacket.Abandon(AbandonReason::Shutdown));
        CHECK(large.Abandon(AbandonReason::Shutdown));
        CHECK(shrunk.Abandon(AbandonReason::Shutdown));
    }
    auto first = probe.PrepareFrame(context);
    context.frameNumber = 2;
    auto second = probe.PrepareFrame(context);
    CHECK(probe.builds == 1 && probe.invocations == 2);

    // Same exact resources, different fresh compiler enumeration.
    context.bindings = std::make_shared<const FrozenExecutionBindings>(
        std::vector<FrozenExecutionBindings::ResourceBinding>{bindings->Resources()[1], bindings->Resources()[0]});
    context.resourceSlots = std::make_shared<const FramePreparationContext::ResourceSlots>(
        FramePreparationContext::ResourceSlots{{102,0},{101,1}});
    auto permuted = probe.PrepareFrame(context);
    CHECK(probe.builds == 1);
    rhi::CommandListVTable commandTable{};
    commandTable.abi_version = rhi::RHI_CL_ABI_MIN;
    rhi::CommandList commands{rhi::CommandListHandle{1,1}};
    commands.impl = &commandTable;
    commands.vt = &commandTable;
    RecordingContext recording(commands, bindings);
    permuted.Record(recording);
    permuted.CommitSubmitted();
    permuted.CommitCompleted();
    CHECK(probe.effect->submitted == 1 && probe.effect->completed == 1);
    CHECK(!permuted.Abandon(AbandonReason::Shutdown));
    CHECK(first.Abandon(AbandonReason::Shutdown));
    CHECK(!first.Abandon(AbandonReason::Shutdown));
    CHECK(second.Abandon(AbandonReason::Shutdown));
    CHECK(probe.effect->abandoned == 2);

    ++probe.revision;
    auto changed = probe.PrepareFrame(context);
    CHECK(probe.builds == 2);
    CHECK(changed.Abandon(AbandonReason::Shutdown));
    auto nextViews = std::make_shared<BindlessResourceViews>();
    context.bindings = std::make_shared<const FrozenExecutionBindings>(
        std::vector<FrozenExecutionBindings::ResourceBinding>{{rhi::Resource{rhi::ResourceHandle{12,1}},owner,views},
            {resource,owner,nextViews}});
    // The recipe embedded resource 101's handle, never its views: a view
    // snapshot replacement on an untouched binding reuses the recipe.
    auto descriptorChanged = probe.PrepareFrame(context);
    CHECK(probe.builds == 2);
    CHECK(descriptorChanged.Abandon(AbandonReason::Shutdown));
    probe.failBuild = true;
    ++probe.revision;
    bool buildFailed = false;
    try { probe.PrepareFrame(context); }
    catch (const std::runtime_error&) { buildFailed = true; }
    CHECK(buildFailed && probe.builds == 3);
    probe.failBuild = false;
    --probe.revision;
    auto preserved = probe.PrepareFrame(context);
    CHECK(probe.builds == 3 && preserved.Abandon(AbandonReason::Shutdown));

    // Packet allocator remains usable after its arena and slot owners leave.
    {
        PreparedInvocationArena measured;
        auto first = measured.MakeShared<int>(1);
        first.reset();
        const auto warm = measured.MemoryStatistics();
        CHECK(warm.allocations > 0 && warm.liveBytes > 0 && warm.peakBytes >= warm.liveBytes);
        for (int i = 0; i < 256; ++i) {
            auto invocation = measured.MakeShared<int>(i);
            CHECK(*invocation == i);
        }
        const auto reused = measured.MemoryStatistics();
        CHECK(reused.liveBytes == warm.liveBytes);
        CHECK(reused.allocatedBytes == warm.allocatedBytes);
        auto frameA = measured.MakeShared<int>(41);
        auto frameB = measured.MakeShared<int>(42);
        auto frameC = measured.MakeShared<int>(43);
        CHECK(frameA.get() != frameB.get() && frameB.get() != frameC.get() && frameA.get() != frameC.get());
        CHECK(!measured.ReleaseUnusedStorage());
        std::weak_ptr<int> weakFrame = frameA;
        frameA.reset(); frameB.reset();
        CHECK(*frameC == 43 && !measured.ReleaseUnusedStorage());
        frameC.reset();
        CHECK(!measured.ReleaseUnusedStorage());
        weakFrame.reset();
        const auto retained = measured.MemoryStatistics();
        CHECK(measured.ReleaseUnusedStorage());
        const auto released = measured.MemoryStatistics();
        CHECK(released.liveBytes < retained.liveBytes && released.peakBytes >= retained.liveBytes);
        auto nextFrame = measured.MakeShared<int>(44);
        CHECK(*nextFrame == 44 && !measured.ReleaseUnusedStorage());
    }
    auto held = arena->MakeShared<int>(42);
    context.invocationArena.reset(); arena.reset();
    CHECK(*held == 42); held.reset();

    auto bundleOwner = std::make_shared<int>(9);
    auto snapshot = std::make_shared<ResourceBindingSnapshot>();
    snapshot->resourceID = 101; snapshot->resource = resource; snapshot->allocationOwner = bundleOwner;
    snapshot->backingGeneration = 1;
    snapshot->views = views;
    std::weak_ptr<int> lifetime = bundleOwner;
    auto bundle = std::make_shared<const PublicationBindingBundle>(
        std::vector<PublicationBindingBundle::Snapshot>{snapshot});
    CHECK(bundle->Find(101) && !bundle->Find(999));
    auto directBindings = std::make_shared<const FrozenExecutionBindings>(
        std::vector<FrozenExecutionBindings::ResourceBinding>{{resource, {}, views, {}, snapshot.get()}},
        std::vector<FrozenExecutionBindings::DescriptorBinding>{},
        FrozenExecutionBindings::OwnershipPolicy::Owned, bundle);
    CHECK(directBindings->Owner({0}) == snapshot->allocationOwner);
    auto directMap = FrozenExecutionBindings::WithResourceMap(directBindings, {0});
    CHECK(directMap->Owner({0}) == snapshot->allocationOwner);
    bool staleDirectBindingRejected = false;
    try { FrozenExecutionBindings stale(
        {{rhi::Resource{rhi::ResourceHandle{11,2}}, {}, views, {}, snapshot.get()}}, {},
        FrozenExecutionBindings::OwnershipPolicy::Owned, bundle); }
    catch (const std::invalid_argument&) { staleDirectBindingRejected = true; }
    CHECK(staleDirectBindingRejected);
    directMap.reset(); directBindings.reset();
    bool duplicateRejected = false;
    try { PublicationBindingBundle duplicate({snapshot,snapshot}); }
    catch (const std::invalid_argument&) { duplicateRejected = true; }
    CHECK(duplicateRejected);
    auto replacement = std::make_shared<ResourceBindingSnapshot>(*snapshot);
    replacement->backingGeneration = 2;
    replacement->resource = rhi::Resource{rhi::ResourceHandle{11,2}};
    replacement->allocationOwner = std::make_shared<int>(10);
    replacement->views = nextViews;
    auto successor = std::make_shared<const PublicationBindingBundle>(
        std::vector<PublicationBindingBundle::Snapshot>{replacement});
    auto selected = bundle;
    bundle = successor;
    CHECK((*selected->Find(101))->backingGeneration == 1);
    CHECK((*bundle->Find(101))->backingGeneration == 2);
    CHECK((*selected->Find(101))->views == views && (*bundle->Find(101))->views == nextViews);
    bundleOwner.reset(); snapshot.reset(); CHECK(!lifetime.expired());
    selected.reset(); CHECK(lifetime.expired());

    // Stable recording ownership must not pin every semantic publication root.
    {
        auto semantic = std::make_shared<int>(1);
        std::weak_ptr<int> semanticLifetime = semantic;
        auto version = std::make_shared<ResourceBindingSnapshot>();
        version->resourceID = 201; version->resource = resource;
        version->views = views;
        version->allocationOwner = std::make_shared<int>(2);
        version->recordingOwner = version->allocationOwner;
        version->semanticConsumer = semantic;
        auto root = std::make_shared<const PublicationBindingBundle>(
            std::vector<PublicationBindingBundle::Snapshot>{version});
        auto frozen = std::make_shared<const FrozenExecutionBindings>(
            std::vector<FrozenExecutionBindings::ResourceBinding>{{resource,{},views}},
            std::vector<FrozenExecutionBindings::DescriptorBinding>{}, FrozenExecutionBindings::OwnershipPolicy::Owned, root);
        auto recordingOwner = frozen->Owner({0});
        CHECK(recordingOwner == version->allocationOwner);
        semantic.reset(); version.reset(); root.reset(); frozen.reset();
        CHECK(semanticLifetime.expired() && recordingOwner);
    }

    // Diagnostic observations must not retain publication or execution owners.
    {
        auto version = std::make_shared<ResourceBindingSnapshot>();
        version->resourceID = 301; version->resource = resource;
        version->allocationOwner = std::make_shared<int>(3);
        auto root = std::make_shared<const PublicationBindingBundle>(
            std::vector<PublicationBindingBundle::Snapshot>{version});
        auto payload = std::make_shared<int>(4);
        auto execution = std::make_shared<int>(5);
        std::weak_ptr<const PublicationBindingBundle> rootLifetime = root;
        std::weak_ptr<int> payloadLifetime = payload, executionLifetime = execution;
        TraceBindingHolder(root, payload, "TestPayload", 7);
        TraceBindingHolderFromOwner(payload.get(), execution, "TestExecution");
        root.reset(); payload.reset(); execution.reset(); version.reset();
        CHECK(rootLifetime.expired() && payloadLifetime.expired() && executionLifetime.expired());
    }

    experimental::GraphCompileInput input;
    input.structure.resourceIDs = {1};
    input.structure.passes = {{0,0,{{0,true}}},{1,0,{{0,false}}}};
    input.structure.placementEdges = {{0,1}};
    experimental::NormalizeCompileInput(input);
    experimental::CompileWorkspace workspace;
    std::atomic_bool cancelled{false};
    auto fresh = experimental::CompileGraph(std::make_shared<const experimental::GraphCompileInput>(input), workspace, cancelled);
    input.analyzedDependencies = workspace.AnalyzeDependencies(input.structure, cancelled);
    auto staged = experimental::CompileGraph(std::make_shared<const experimental::GraphCompileInput>(input), workspace, cancelled);
    CHECK(fresh->edges == staged->edges && fresh->schedulingEdges == staged->schedulingEdges);
    CHECK(fresh->topologicalOrder == staged->topologicalOrder);
    {
        persistent::GraphProgram program;
        auto edit = program.BeginEdit();
        auto snapshot = std::make_shared<ResourceBindingSnapshot>();
        snapshot->resource = resource; snapshot->backingGeneration = 1;
        snapshot->allocationOwner = owner; snapshot->descriptorOwner = owner;
        auto nativeViews = std::make_shared<BindlessResourceViews>();
        nativeViews->views.push_back({BindlessViewKind::ShaderResource,UINT32_MAX,0,0,
            rhi::DescriptorSlot(rhi::DescriptorHeapHandle{4,1},9)});
        snapshot->views = nativeViews;
        persistent::BindingVersion version;
        version.identity = (uint64_t{1} << 32) | 11; version.backingRevision = 1;
        version.owner = owner; version.recording = snapshot;
        const auto slot = edit.AddResource(version.shape,std::move(version));
        PersistentAuthor author{slot};
        auto executable = persistent::TypedPassExecutable<PersistentAuthor>::Build(edit,{},author,uint32_t{17});
        CHECK(author.declarations == 1 && author.builds == 1);
        const auto pass = edit.AddPass({});
        const auto resourceToken = edit.Declare(pass,slot,{}, {2,0,1,false});
        const auto viewToken = edit.DeclareView(resourceToken,{});
        auto old = edit.Build(workspace,cancelled);
        CHECK(program.Install(edit,old));
        {
            auto cancellation = std::make_shared<Effect>();
            auto packet = std::make_shared<experimental::RenderFrameSnapshot>();
            packet->publication = old;
            packet->passes.push_back(executable.PrepareInvocation(*old,
                PersistentAuthor::Invocation{9,99,cancellation},arena));
            auto cpuConsumer = packet;
            packet.reset();
            CHECK(cancellation->abandoned == 0);
            cpuConsumer.reset();
            CHECK(cancellation->abandoned == 1);
        }
        {
            auto cancellation = std::make_shared<Effect>();
            auto packet = std::make_shared<experimental::RenderFrameSnapshot>();
            packet->publication = old; packet->layout = old->executable->executionLayout;
            auto barriers = std::make_shared<experimental::PreparedExecutionBarrierPlan>();
            barriers->batches.resize(old->executable->graph->batches.size());
            packet->barrierPlan = barriers;
            packet->passes.push_back(executable.PrepareInvocation(*old,
                PersistentAuthor::Invocation{9,100,cancellation},arena));
            packet->passes.push_back(PreparedPass::FromTyped<PersistentProbe>(
                PersistentProbe::Invocation{old,resourceToken,viewToken,9}));
            CHECK(old->executable->graph->batches.size() == 1);
            auto valid = experimental::BuildPersistentRecordingList(packet,0);
            CHECK(valid.passes.size() == 2);
            CHECK(packet->passes[1].Abandon(AbandonReason::AdmissionFailed));
            bool rejected = false;
            try { experimental::BuildPersistentRecordingList(packet,0); }
            catch (const std::invalid_argument&) { rejected = true; }
            CHECK(rejected && !packet->passes[0].IsConsumed());
            packet.reset();
            CHECK(cancellation->abandoned == 0);
            valid = {};
            CHECK(cancellation->abandoned == 1);
        }
        auto typedEffect = std::make_shared<Effect>();
        auto heldTypedPacket = executable.PrepareInvocation(*old,PersistentAuthor::Invocation{9,1,typedEffect},arena);
        auto cancelledTypedPacket = executable.PrepareInvocation(*old,PersistentAuthor::Invocation{9,2,typedEffect},arena);
        CHECK(cancelledTypedPacket.Abandon(AbandonReason::AdmissionFailed));
        CHECK(!cancelledTypedPacket.Abandon(AbandonReason::AdmissionFailed));
        CHECK(typedEffect->abandoned == 1);
        auto heldPacket = PreparedPass::FromTyped<PersistentProbe>(
            PersistentProbe::Invocation{old,resourceToken,viewToken,9});
        auto rotation = program.BeginEdit();
        auto next = old->bindings.At(slot);
        auto replaced = std::make_shared<ResourceBindingSnapshot>(*next.recording);
        auto replacedViews = std::make_shared<BindlessResourceViews>(*replaced->views);
        replacedViews->views[0].descriptor.index = 14;
        replaced->views = replacedViews; next.recording = replaced;
        rotation.ReplaceBinding(slot,std::move(next));
        auto selected = rotation.Build(workspace,cancelled);
        CHECK(program.Install(rotation,selected));
        auto oldDirect = RecordingContext::FromPersistentBindings(commands,old);
        heldTypedPacket.Record(oldDirect);
        heldTypedPacket.CommitSubmitted({1}); heldTypedPacket.CommitCompleted({1});
        CHECK(typedEffect->submitted == 1 && typedEffect->completed == 1);
        auto typedPacket = executable.PrepareInvocation(*selected,PersistentAuthor::Invocation{14,3,typedEffect},arena);
        auto selectedDirect = RecordingContext::FromPersistentBindings(commands,selected);
        typedPacket.Record(selectedDirect);
        typedPacket.CommitSubmitted({2}); typedPacket.CommitCompleted({2});
        CHECK(author.declarations == 1 && author.builds == 1);
        CHECK(typedEffect->submitted == 2 && typedEffect->completed == 2);
        auto wrongPublication = executable.PrepareInvocation(*old,PersistentAuthor::Invocation{9,4,typedEffect},arena);
        bool wrongRootRejected = false;
        try { wrongPublication.Record(selectedDirect); } catch (const std::logic_error&) { wrongRootRejected = true; }
        CHECK(wrongRootRejected);
        heldPacket.Record(recording);
        auto freshPacket = PreparedPass::FromTyped<PersistentProbe>(
            PersistentProbe::Invocation{selected,resourceToken,viewToken,14});
        freshPacket.Record(recording);
        auto direct = RecordingContext::FromPersistentBindings(commands,selected);
        CHECK(direct.Resolve(resourceToken).GetHandle().index == 11 && direct.Resolve(viewToken).index == 14);
        bool legacyRejected = false;
        try { direct.Resolve(PreparedResourceReference{0}); } catch (const std::logic_error&) { legacyRejected = true; }
        CHECK(legacyRejected);
        bool consumedRejected = false;
        try { freshPacket.Record(direct); } catch (const std::logic_error&) { consumedRejected = true; }
        CHECK(consumedRejected);
        auto directPacket = PreparedPass::FromTyped<PersistentProbe>(
            PersistentProbe::Invocation{selected,resourceToken,viewToken,14});
        directPacket.Record(direct);
        bool unselectedRejected = false;
        try { recording.Resolve(resourceToken); } catch (const std::logic_error&) { unselectedRejected = true; }
        CHECK(unselectedRejected);
        CHECK(recording.Resolve(PreparedResourceReference{0}).GetHandle().index == 11);
        auto programEdit = program.BeginEdit();
        auto replacedExecutable = executable.RebuildProgramInterface(programEdit,author,uint32_t{18});
        auto changedProgram = programEdit.Build(workspace,cancelled);
        CHECK(changedProgram->executable == selected->executable);
        CHECK(program.Install(programEdit,changedProgram));
        CHECK(author.declarations == 1 && author.builds == 2);
        bool staleInterfaceRejected = false;
        try { executable.PrepareInvocation(*changedProgram,PersistentAuthor::Invocation{14,7,typedEffect},arena); }
        catch (const std::logic_error&) { staleInterfaceRejected = true; }
        CHECK(staleInterfaceRejected);
        auto newProgramPacket = replacedExecutable.PrepareInvocation(*changedProgram,PersistentAuthor::Invocation{14,8,typedEffect},arena);
        auto changedProgramContext = RecordingContext::FromPersistentBindings(commands,changedProgram);
        newProgramPacket.Record(changedProgramContext);
        auto failedBuild = program.BeginEdit();
        bool failedProgramRejected = false;
        try { persistent::TypedPassExecutable<PersistentAuthor>::Build(failedBuild,{},author,uint32_t{0}); }
        catch (const std::runtime_error&) { failedProgramRejected = true; }
        CHECK(failedProgramRejected);
        bool failedTransactionRejected = false;
        try { failedBuild.Build(workspace,cancelled); } catch (const std::invalid_argument&) { failedTransactionRejected = true; }
        CHECK(failedTransactionRejected && program.Select() == changedProgram);
        auto removal = program.BeginEdit();
        removal.RemovePass(executable.Id());
        auto removed = removal.Build(workspace,cancelled);
        CHECK(program.Install(removal,removed));
        bool retiredRejected = false;
        try { executable.PrepareInvocation(*removed,PersistentAuthor::Invocation{14,5,typedEffect},arena); }
        catch (const std::logic_error&) { retiredRejected = true; }
        CHECK(retiredRejected);
        CHECK(!removed->logical->passSlots[executable.Id().index].recordingInterface);
        auto stillHeld = executable.PrepareInvocation(*old,PersistentAuthor::Invocation{9,6,typedEffect},arena);
        stillHeld.Record(oldDirect);
    }
    std::puts("Recording recipe tests passed");
}
