#include "RenderPasses/Base/TypedRenderGraphPass.h"
#include "Render/PublicationBindingBundle.h"
#include "Render/RenderGraph/ExperimentalGraphCompiler.h"
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
    auto descriptorChanged = probe.PrepareFrame(context);
    CHECK(probe.builds == 3);
    CHECK(descriptorChanged.Abandon(AbandonReason::Shutdown));
    probe.failBuild = true;
    ++probe.revision;
    bool buildFailed = false;
    try { probe.PrepareFrame(context); }
    catch (const std::runtime_error&) { buildFailed = true; }
    CHECK(buildFailed && probe.builds == 4);
    probe.failBuild = false;
    --probe.revision;
    auto preserved = probe.PrepareFrame(context);
    CHECK(probe.builds == 4 && preserved.Abandon(AbandonReason::Shutdown));

    // Packet allocator remains usable after its arena and slot owners leave.
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
    std::puts("Recording recipe tests passed");
}
