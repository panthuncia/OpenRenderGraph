#include "RenderPasses/Base/TypedRenderGraphPass.h"
#include "TestSupport.h"
#include "Render/FeatureDomainRegistry.h"
#include "Render/PublicationBindingBundle.h"
#include "Render/ShaderAPI.h"
#include "Render/RenderGraph/GraphCompiler.h"
#include "Render/RenderGraph/PersistentGraph.h"
#include "Render/RenderGraph/RhiExecution.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace org;

namespace {

template<class Exception = std::exception, class F> bool Rejects(F&& fn) {
    try { fn(); } catch (const Exception&) { return true; } catch (...) { return false; }
    return false;
}

struct Effect final : PreparedLifecycleEffect {
    mutable int submitted = 0, completed = 0, abandoned = 0;
    void Submitted(SubmissionContext) const override { ++submitted; }
    void Completed(CompletionContext) const override { ++completed; }
    void Abandoned(AbandonReason) const override { ++abandoned; }
};

struct PersistentProbe {
    struct Invocation {
        std::shared_ptr<const org::persistent::SelectedPublication> publication;
        org::persistent::BindingToken resource;
        org::persistent::ViewToken view;
        uint32_t expectedDescriptor = 0;
    };
    static void Record(const Invocation& invocation, RecordingContext& context) {
        auto scoped = context.WithPersistentBindings(invocation.publication);
        ENSURE(scoped.Resolve(invocation.resource).GetHandle().index == 11);
        ENSURE(scoped.Resolve(invocation.view).index == invocation.expectedDescriptor);
    }
};

// A pass without declared bindings whose layout the test installs directly (SetResourceUses): its recording reads
// the frame's declared views through the context, never through values captured at preparation.
struct ViewFrame {
    std::vector<DeclaredViewToken> views;
    ResourceBindingToken resource;
    std::shared_ptr<std::vector<uint32_t>> recorded;
};
struct ViewProbe final : TypedRenderGraphPass<ViewProbe, ViewFrame> {
    std::shared_ptr<const ResourceUseLayout> layout;
    std::shared_ptr<std::vector<uint32_t>> recorded = std::make_shared<std::vector<uint32_t>>();
    std::shared_ptr<Effect> effect = std::make_shared<Effect>();
    int prepares = 0;
    void Declare(PassBuilder&) {}
    ViewFrame Prepare(const PassPrepareContext& context) {
        ++prepares;
        context.Reserve(effect);
        return {{{layout, 0, 0}, {layout, 0, 1}, {layout, 1, 0}}, layout->uses[1].binding, recorded};
    }
    static void Record(const ViewFrame& frame, PassRecordContext& context) {
        for (const auto& view : frame.views) frame.recorded->push_back(context.Index(view));
        frame.recorded->push_back(context.Resolve(frame.resource).GetHandle().index);
    }
};

struct RevisionProbe final : TypedRenderGraphPass<RevisionProbe> {
    uint64_t setting = 1;
    void Declare(PassBuilder&) {}
    void Revision(std::vector<uint64_t>& out) const { out.push_back(setting); }
    static void Record(PassRecordContext&) {}
};

struct DeclaredFixture {
    std::shared_ptr<ResourceUseLayout> layout = std::make_shared<ResourceUseLayout>();
    std::shared_ptr<int> owner = std::make_shared<int>(7);

    DeclaredFixture() {
        using A = rhi::ResourceAccessType;
        using V = BindlessViewKind;
        // Resource 41 is declared both as a shader resource and for unordered access under one name; resource 42
        // only as a shader resource. namedViews is what RenderPassBuilder::NameUseViews leaves for those
        // declarations: the unordered-access view wins the shared name.
        layout->uses.push_back({{}, {A::UnorderedAccess, rhi::ResourceLayout::Common, rhi::ResourceSyncState::All},
            {41, 41}, {{V::ShaderResource, UINT32_MAX, 0, 0}, {V::UnorderedAccess, UINT32_MAX, 0, 0}}});
        layout->uses.push_back({{}, {A::ShaderResource, rhi::ResourceLayout::Common, rhi::ResourceSyncState::All},
            {42, 42}, {{V::ShaderResource, UINT32_MAX, 0, 0}}});
        layout->namedViews.push_back({ResourceIdentifier{"Test::Color"}.hash, 0, 1, V::UnorderedAccess});
        layout->namedViews.push_back({ResourceIdentifier{"Test::Depth"}.hash, 1, 0, V::ShaderResource});
        layout->activeFeatureDomains.push_back(FeatureDomainIdentifier{"Test::FeatureDomains::Active"}.hash);
    }

    // The frame's bindings: descriptor `base` for 41's SRV, base+1 for its UAV, base+2 for 42's SRV.
    FramePreparationContext Frame(uint32_t base, bool depthViews = true) const {
        using V = BindlessViewKind;
        FramePreparationContext context;
        context.resourceUses = layout;
        auto colorViews = std::make_shared<BindlessResourceViews>();
        colorViews->views = {{V::ShaderResource, UINT32_MAX, 0, 0, {{3,1}, base}},
            {V::UnorderedAccess, UINT32_MAX, 0, 0, {{3,1}, base + 1}}};
        auto depthViewsTable = std::make_shared<BindlessResourceViews>();
        if (depthViews) depthViewsTable->views = {{V::ShaderResource, UINT32_MAX, 0, 0, {{3,1}, base + 2}}};
        context.bindings = std::make_shared<const FrozenExecutionBindings>(std::vector<FrozenExecutionBindings::ResourceBinding>{
            {rhi::Resource({10,1}), owner, colorViews}, {rhi::Resource({11,1}), owner, depthViewsTable}});
        context.resourceSlots = std::make_shared<const FramePreparationContext::ResourceSlots>(
            FramePreparationContext::ResourceSlots{{41,0},{42,1}});
        return context;
    }
    std::shared_ptr<const DeclaredRecordingScope> Scope(const FramePreparationContext& frame) const {
        auto scope = std::make_shared<DeclaredRecordingScope>();
        scope->layout = layout;
        scope->views = frame.ResolveDeclaredViewTable();
        return scope;
    }
};

std::shared_ptr<const PipelineStatePayload> Payload(std::vector<ResourceIdentifier> mandatory,
    std::vector<ResourceIdentifier> optional = {}) {
    return std::make_shared<const PipelineStatePayload>(rhi::PipelinePtr{}, 0,
        PipelineResources{std::move(mandatory), std::move(optional)}, 1, 0, 0, "TestPipeline");
}

} // namespace

int main() {
    rhi::CommandListVTable commandTable{};
    commandTable.abi_version = rhi::RHI_CL_ABI_MIN;
    rhi::CommandList commands{rhi::CommandListHandle{1,1}};
    commands.impl = &commandTable;
    commands.vt = &commandTable;

    // Declared view table and RecordingContext::View/Index.
    DeclaredFixture fixture;
    {
        const auto frame = fixture.Frame(7);
        const auto table = frame.ResolveDeclaredViewTable();
        ENSURE(table.size() == 3 && fixture.layout->ViewCount() == 3);
        ENSURE(table[0].index == 7 && table[1].index == 8 && table[2].index == 9);
        RecordingContext base(commands, frame.bindings);
        ENSURE(Rejects<std::logic_error>([&] { (void)base.View({fixture.layout, 0, 0}); }));
        ENSURE(Rejects<std::logic_error>([&] { (void)base.Resolve(ResourceBindingToken{41, 41}); }));
        auto scoped = base.WithDeclaredResources(frame.bindings, frame.resourceSlots, fixture.Scope(frame));
        ENSURE(scoped.Index({fixture.layout, 0, 0}) == 7 && scoped.Index({fixture.layout, 0, 1}) == 8);
        ENSURE(scoped.Index({fixture.layout, 1, 0}) == 9 && scoped.View({fixture.layout, 1, 0}).heap.valid());
        ENSURE(scoped.Resolve(ResourceBindingToken{42, 42}).GetHandle().index == 11);
        ENSURE(scoped.Resolve(DeclaredViewToken{fixture.layout, 0, 1}).GetHandle().index == 10);
        ENSURE(Rejects<std::invalid_argument>([&] { (void)scoped.Resolve(ResourceBindingToken{99, 99}); }));
        // A token of another declaration (a copied layout), or of a pass's previous declaration, is rejected.
        DeclaredViewToken foreign{std::make_shared<ResourceUseLayout>(*fixture.layout), 0, 0};
        ENSURE(Rejects<std::invalid_argument>([&] { (void)scoped.View(foreign); }));
        DeclaredFixture redeclared;
        auto staleScope = std::make_shared<DeclaredRecordingScope>();
        staleScope->layout = redeclared.layout;
        staleScope->views = table;
        auto stale = base.WithDeclaredResources(frame.bindings, frame.resourceSlots, staleScope);
        ENSURE(Rejects<std::invalid_argument>([&] { (void)stale.Index({fixture.layout, 0, 0}); }));
        // Ordinals outside the declaration.
        ENSURE(Rejects<std::invalid_argument>([&] { (void)scoped.View({fixture.layout, 0, 2}); }));
        ENSURE(Rejects<std::invalid_argument>([&] { (void)scoped.View({fixture.layout, 2, 0}); }));
        ENSURE(Rejects([&] { (void)base.WithDeclaredResources({}, frame.resourceSlots); }));
        // A declared view the frame could not resolve stays invalid in the table and fails when recorded.
        const auto partial = fixture.Frame(30, false);
        const auto partialTable = partial.ResolveDeclaredViewTable();
        ENSURE(partialTable.size() == 3 && partialTable[0].index == 30 && !partialTable[2].heap.valid());
        auto unresolved = base.WithDeclaredResources(partial.bindings, partial.resourceSlots, fixture.Scope(partial));
        ENSURE(unresolved.Index({fixture.layout, 0, 1}) == 31);
        ENSURE(Rejects<std::out_of_range>([&] { (void)unresolved.View({fixture.layout, 1, 0}); }));
        // A short table (built for fewer views than the layout declares) is the same failure, not an overrun.
        auto shortScope = std::make_shared<DeclaredRecordingScope>(*fixture.Scope(frame));
        shortScope->views.resize(1);
        auto truncated = base.WithDeclaredResources(frame.bindings, frame.resourceSlots, shortScope);
        ENSURE(Rejects<std::out_of_range>([&] { (void)truncated.View({fixture.layout, 1, 0}); }));
    }

    // A typed pass prepares per frame; Record resolves against the frame its packet was prepared for.
    {
        ViewProbe probe;
        probe.layout = fixture.layout;
        probe.SetResourceUses(fixture.layout);
        auto arena = std::make_shared<PreparedInvocationArena>();
        auto oldFrame = fixture.Frame(7), newFrame = fixture.Frame(19);
        oldFrame.invocationArena = newFrame.invocationArena = arena;
        auto oldPacket = probe.PrepareFrame(oldFrame);
        auto newPacket = probe.PrepareFrame(newFrame);
        auto abandoned = probe.PrepareFrame(newFrame);
        ENSURE(probe.prepares == 3 && oldPacket.IsWorkerSafe());
        // A packet that reserved an effect has a lifecycle to commit: never replayed.
        ENSURE(!oldPacket.IsReplayable());
        RecordingContext recording(commands, oldFrame.bindings);
        newPacket.Record(recording);
        oldPacket.Record(recording);
        ENSURE((*probe.recorded == std::vector<uint32_t>{19, 20, 21, 11, 7, 8, 9, 11}));
        ENSURE(Rejects<std::logic_error>([&] { oldPacket.Record(recording); }));
        oldPacket.CommitSubmitted(); oldPacket.CommitCompleted();
        newPacket.CommitSubmitted(); newPacket.CommitCompleted();
        ENSURE(!oldPacket.Abandon(AbandonReason::Shutdown));
        ENSURE(abandoned.Abandon(AbandonReason::Shutdown) && !abandoned.Abandon(AbandonReason::Shutdown));
        ENSURE(probe.effect->submitted == 2 && probe.effect->completed == 2 && probe.effect->abandoned == 1);

        // Revision opts a pass into kept-recording reuse; without it the pass cannot say and is never current.
        uint64_t hash = 0;
        ENSURE(!probe.RevisionHash(hash));
        RevisionProbe revision;
        uint64_t first = 0, same = 0, changed = 0;
        ENSURE(revision.RevisionHash(first) && revision.RevisionHash(same) && first == same);
        revision.setting = 2;
        ENSURE(revision.RevisionHash(changed) && changed != first);
        revision.setting = 1;
        ENSURE(revision.RevisionHash(same) && same == first);
    }

    // Pipeline ResourceDescriptorIndex slots are filled from the pass's declarations by name.
    {
        FeatureDomainRegistry::Get().RegisterResourceDomain(ResourceIdentifier{"Test::Inactive"},
            FeatureDomainIdentifier{"Test::FeatureDomains::Inactive"});
        FeatureDomainRegistry::Get().RegisterResourceDomain(ResourceIdentifier{"Test::Active"},
            FeatureDomainIdentifier{"Test::FeatureDomains::Active"});
        const auto frame = fixture.Frame(7);
        RecordingContext base(commands, frame.bindings);
        const auto scoped = base.WithDeclaredResources(frame.bindings, frame.resourceSlots, fixture.Scope(frame));
        // Declared shader resource; a name declared as SRV and UAV resolves to the UAV; optional undeclared.
        ENSURE((scoped.DescriptorSlotIndices(*Payload({"Test::Depth", "Test::Color"}, {"Test::Missing", "Test::Depth"}))
            == std::vector<uint32_t>{9, 8, UINT32_MAX, 9}));
        // A mandatory undeclared name throws, naming the pipeline and the resource.
        bool named = false;
        try { (void)scoped.DescriptorSlotIndices(*Payload({"Test::Depth", "Test::Missing"})); }
        catch (const std::invalid_argument& error) {
            const std::string message = error.what();
            named = message.find("TestPipeline") != std::string::npos && message.find("Test::Missing") != std::string::npos;
        }
        ENSURE(named);
        // absentPrefixes.
        const std::array<std::string_view, 1> prefixes{"Test::Absent"};
        ENSURE((scoped.DescriptorSlotIndices(*Payload({"Test::Absent::Mask", "Test::Depth"}), {prefixes})
            == std::vector<uint32_t>{UINT32_MAX, 9}));
        ENSURE(Rejects<std::invalid_argument>([&] { (void)scoped.DescriptorSlotIndices(*Payload({"Test::Absent::Mask"})); }));
        // A feature domain the pass names nothing of; an active domain's undeclared name still throws.
        ENSURE((scoped.DescriptorSlotIndices(*Payload({"Test::Inactive::Mask"})) == std::vector<uint32_t>{UINT32_MAX}));
        ENSURE(Rejects<std::invalid_argument>([&] { (void)scoped.DescriptorSlotIndices(*Payload({"Test::Active::Mask"})); }));
        // Without a declared scope nothing is declared: only the absent rules apply.
        ENSURE(Rejects<std::invalid_argument>([&] { (void)base.DescriptorSlotIndices(*Payload({"Test::Depth"})); }));
        ENSURE((base.DescriptorSlotIndices(*Payload({"Test::Inactive::Mask", "Test::Active::Mask"}, {"Test::Depth"}))
            == std::vector<uint32_t>{UINT32_MAX, UINT32_MAX, UINT32_MAX}));
        // A named view the frame could not resolve fails even for an optional slot: it was declared.
        const auto partial = fixture.Frame(30, false);
        const auto unresolved = base.WithDeclaredResources(partial.bindings, partial.resourceSlots, fixture.Scope(partial));
        ENSURE(Rejects<std::out_of_range>([&] { (void)unresolved.DescriptorSlotIndices(*Payload({}, {"Test::Depth"})); }));
        // More slots than the root constants hold.
        std::vector<ResourceIdentifier> many(shaderapi::kNumResourceDescriptorIndicesRootConstants + 1, ResourceIdentifier{"Test::Missing"});
        ENSURE(Rejects<std::out_of_range>([&] { (void)scoped.DescriptorSlotIndices(*Payload({}, many)); }));
        ENSURE(Rejects<std::invalid_argument>([&] {
            RecordingContext target(commands, frame.bindings);
            target.BindComputePipeline(std::shared_ptr<const PipelineStatePayload>{});
        }));
    }

    // RecordingRetention: what a recording names lives until the recording retires, once per owner.
    {
        auto bindings = fixture.Frame(7).bindings;
        auto retention = std::make_shared<RecordingRetention>();
        std::weak_ptr<RecordingRetention> retentionLifetime = retention;
        auto payload = Payload({"Test::Depth"});
        std::weak_ptr<const PipelineStatePayload> payloadLifetime = payload;
        {
            RecordingContext recording(commands, bindings);
            recording.SetRetention(retention);
            recording.Retain(payload);
            recording.Retain(payload);
            ENSURE(payload.use_count() == 2);
            // A pass-local copy retains into the same recording.
            const auto frame = fixture.Frame(7);
            auto scoped = recording.WithDeclaredResources(frame.bindings, frame.resourceSlots, fixture.Scope(frame));
            scoped.Retain(payload);
            auto other = std::make_shared<int>(3);
            std::weak_ptr<int> otherLifetime = other;
            scoped.Retain(other);
            other.reset();
            ENSURE(payload.use_count() == 2 && !otherLifetime.expired());
            ENSURE(Rejects<std::invalid_argument>([&] { scoped.Retain({}); }));
        }
        payload.reset();
        ENSURE(!payloadLifetime.expired());
        retention.reset();
        ENSURE(retentionLifetime.expired() && payloadLifetime.expired());

        // Without an installed retention a context keeps what it retains itself.
        auto loose = std::make_shared<int>(4);
        std::weak_ptr<int> looseLifetime = loose;
        {
            RecordingContext recording(commands, bindings);
            recording.Retain(loose);
            loose.reset();
            ENSURE(!looseLifetime.expired());
        }
        ENSURE(looseLifetime.expired());

        // An owned recording list installs its retention in the contexts made for it.
        compiler::OwnedRecordingList list;
        list.bindings = bindings;
        ENSURE(list.retention);
        auto listed = std::make_shared<int>(5);
        std::weak_ptr<int> listedLifetime = listed;
        {
            auto recording = compiler::MakeOwnedRecordingContext(commands, list);
            recording.Retain(listed);
            auto again = compiler::MakeOwnedRecordingContext(commands, list);
            again.Retain(listed);
        }
        ENSURE(listed.use_count() == 2);
        listed.reset();
        ENSURE(!listedLifetime.expired());
        list = {};
        ENSURE(listedLifetime.expired());
    }

    auto owner = std::make_shared<int>(7);
    auto views = std::make_shared<BindlessResourceViews>();
    const auto resource = rhi::Resource{rhi::ResourceHandle{11,1}};
    auto bindings = std::make_shared<const FrozenExecutionBindings>(
        std::vector<FrozenExecutionBindings::ResourceBinding>{{resource, owner, views},
            {rhi::Resource{rhi::ResourceHandle{12,1}}, owner, views}});
    RecordingContext recording(commands, bindings);
    auto arena = std::make_shared<PreparedInvocationArena>();

    // Packet allocator remains usable after its arena and slot owners leave.
    {
        PreparedInvocationArena measured;
        auto first = measured.MakeShared<int>(1);
        first.reset();
        const auto warm = measured.MemoryStatistics();
        ENSURE(warm.allocations > 0 && warm.liveBytes > 0 && warm.peakBytes >= warm.liveBytes);
        for (int i = 0; i < 256; ++i) {
            auto invocation = measured.MakeShared<int>(i);
            ENSURE(*invocation == i);
        }
        const auto reused = measured.MemoryStatistics();
        ENSURE(reused.liveBytes == warm.liveBytes);
        ENSURE(reused.allocatedBytes == warm.allocatedBytes);
        auto frameA = measured.MakeShared<int>(41);
        auto frameB = measured.MakeShared<int>(42);
        auto frameC = measured.MakeShared<int>(43);
        ENSURE(frameA.get() != frameB.get() && frameB.get() != frameC.get() && frameA.get() != frameC.get());
        ENSURE(!measured.ReleaseUnusedStorage());
        std::weak_ptr<int> weakFrame = frameA;
        frameA.reset(); frameB.reset();
        ENSURE(*frameC == 43 && !measured.ReleaseUnusedStorage());
        frameC.reset();
        ENSURE(!measured.ReleaseUnusedStorage());
        weakFrame.reset();
        const auto retained = measured.MemoryStatistics();
        ENSURE(measured.ReleaseUnusedStorage());
        const auto released = measured.MemoryStatistics();
        ENSURE(released.liveBytes < retained.liveBytes && released.peakBytes >= retained.liveBytes);
        auto nextFrame = measured.MakeShared<int>(44);
        ENSURE(*nextFrame == 44 && !measured.ReleaseUnusedStorage());
    }
    auto held = arena->MakeShared<int>(42);
    arena.reset();
    ENSURE(*held == 42); held.reset();

    auto bundleOwner = std::make_shared<int>(9);
    auto snapshot = std::make_shared<ResourceBindingSnapshot>();
    snapshot->resourceID = 101; snapshot->resource = resource; snapshot->allocationOwner = bundleOwner;
    snapshot->backingGeneration = 1;
    snapshot->views = views;
    std::weak_ptr<int> lifetime = bundleOwner;
    auto bundle = std::make_shared<const PublicationBindingBundle>(
        std::vector<PublicationBindingBundle::Snapshot>{snapshot});
    ENSURE(bundle->Find(101) && !bundle->Find(999));
    auto directBindings = std::make_shared<const FrozenExecutionBindings>(
        std::vector<FrozenExecutionBindings::ResourceBinding>{{resource, {}, views, {}, snapshot.get()}},
        std::vector<FrozenExecutionBindings::DescriptorBinding>{},
        FrozenExecutionBindings::OwnershipPolicy::Owned, bundle);
    ENSURE(directBindings->Owner({0}) == snapshot->allocationOwner);
    bool staleDirectBindingRejected = false;
    try { FrozenExecutionBindings stale(
        {{rhi::Resource{rhi::ResourceHandle{11,2}}, {}, views, {}, snapshot.get()}}, {},
        FrozenExecutionBindings::OwnershipPolicy::Owned, bundle); }
    catch (const std::invalid_argument&) { staleDirectBindingRejected = true; }
    ENSURE(staleDirectBindingRejected);
    directBindings.reset();
    bool duplicateRejected = false;
    try { PublicationBindingBundle duplicate({snapshot,snapshot}); }
    catch (const std::invalid_argument&) { duplicateRejected = true; }
    ENSURE(duplicateRejected);
    auto nextViews = std::make_shared<BindlessResourceViews>();
    auto replacement = std::make_shared<ResourceBindingSnapshot>(*snapshot);
    replacement->backingGeneration = 2;
    replacement->resource = rhi::Resource{rhi::ResourceHandle{11,2}};
    replacement->allocationOwner = std::make_shared<int>(10);
    replacement->views = nextViews;
    auto successor = std::make_shared<const PublicationBindingBundle>(
        std::vector<PublicationBindingBundle::Snapshot>{replacement});
    auto selectedBundle = bundle;
    bundle = successor;
    ENSURE((*selectedBundle->Find(101))->backingGeneration == 1);
    ENSURE((*bundle->Find(101))->backingGeneration == 2);
    ENSURE((*selectedBundle->Find(101))->views == views && (*bundle->Find(101))->views == nextViews);
    bundleOwner.reset(); snapshot.reset(); ENSURE(!lifetime.expired());
    selectedBundle.reset(); ENSURE(lifetime.expired());

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
        ENSURE(recordingOwner == version->allocationOwner);
        semantic.reset(); version.reset(); root.reset(); frozen.reset();
        ENSURE(semanticLifetime.expired() && recordingOwner);
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
        ENSURE(rootLifetime.expired() && payloadLifetime.expired() && executionLifetime.expired());
    }

    compiler::GraphCompileInput input;
    input.structure.resourceIDs = {1};
    input.structure.passes = {{0,0,{{0,true}}},{1,0,{{0,false}}}};
    input.structure.placementEdges = {{0,1}};
    compiler::NormalizeCompileInput(input);
    compiler::CompileWorkspace workspace;
    std::atomic_bool cancelled{false};
    auto fresh = compiler::CompileGraph(std::make_shared<const compiler::GraphCompileInput>(input), workspace, cancelled);
    input.analyzedDependencies = workspace.AnalyzeDependencies(input.structure, cancelled);
    auto staged = compiler::CompileGraph(std::make_shared<const compiler::GraphCompileInput>(input), workspace, cancelled);
    ENSURE(fresh->edges == staged->edges && fresh->schedulingEdges == staged->schedulingEdges);
    ENSURE(fresh->topologicalOrder == staged->topologicalOrder);

    // Persistent bindings: a packet records against the publication it was prepared for.
    {
        persistent::GraphProgram program;
        auto edit = program.BeginEdit();
        auto bindingSnapshot = std::make_shared<ResourceBindingSnapshot>();
        bindingSnapshot->resource = resource; bindingSnapshot->backingGeneration = 1;
        bindingSnapshot->allocationOwner = owner; bindingSnapshot->descriptorOwner = owner;
        auto nativeViews = std::make_shared<BindlessResourceViews>();
        nativeViews->views.push_back({BindlessViewKind::ShaderResource,UINT32_MAX,0,0,
            rhi::DescriptorSlot(rhi::DescriptorHeapHandle{4,1},9)});
        bindingSnapshot->views = nativeViews;
        persistent::BindingVersion version;
        version.identity = (uint64_t{1} << 32) | 11; version.backingRevision = 1;
        version.owner = owner; version.recording = bindingSnapshot;
        const auto slot = edit.AddResource(version.shape,std::move(version));
        const auto pass = edit.AddPass({});
        const auto resourceToken = edit.Declare(pass,slot,{}, {2,0,1,false});
        const auto viewToken = edit.DeclareView(resourceToken,{});
        auto old = edit.Build(workspace,cancelled);
        ENSURE(program.Install(edit,old));
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
        ENSURE(program.Install(rotation,selected));
        heldPacket.Record(recording);
        auto freshPacket = PreparedPass::FromTyped<PersistentProbe>(
            PersistentProbe::Invocation{selected,resourceToken,viewToken,14});
        freshPacket.Record(recording);
        auto oldDirect = RecordingContext::FromPersistentBindings(commands,old);
        ENSURE(oldDirect.Resolve(viewToken).index == 9);
        auto direct = RecordingContext::FromPersistentBindings(commands,selected);
        ENSURE(direct.Resolve(resourceToken).GetHandle().index == 11 && direct.Resolve(viewToken).index == 14);
        ENSURE(Rejects<std::logic_error>([&] { (void)direct.Resolve(PreparedResourceReference{0}); }));
        ENSURE(Rejects<std::logic_error>([&] { freshPacket.Record(direct); }));
        auto directPacket = PreparedPass::FromTyped<PersistentProbe>(
            PersistentProbe::Invocation{selected,resourceToken,viewToken,14});
        directPacket.Record(direct);
        ENSURE(Rejects<std::logic_error>([&] { (void)recording.Resolve(resourceToken); }));
        ENSURE(recording.Resolve(PreparedResourceReference{0}).GetHandle().index == 11);
    }
    std::puts("Recording tests passed");
}
