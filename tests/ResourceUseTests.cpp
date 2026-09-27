#include "Render/DeclaredTableLayout.h"
#include "Render/RenderGraph/PersistentGraph.h"
#include <cstdio>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Failure at %d: %s\n", __LINE__, #x); return 1; } } while (false)
template<class F> bool Rejects(F&& fn) { try { fn(); } catch (const std::exception&) { return true; } return false; }

int main() {
    using namespace org;
    using A = rhi::ResourceAccessType;
    using V = BindlessViewKind;
    ValidateResourceUse({A::ShaderResource, {{V::ShaderResource}}});
    ValidateResourceUse({A::UnorderedAccessClear, {{V::UnorderedAccess}, {V::NonShaderVisibleUnorderedAccess}}});
    ValidateResourceUse({A::CopySource, {}});
    ValidateResourceUse({A::IndirectArgument, {}});
    ValidateResourceUse({A::RenderTarget, {{V::RenderTarget}}});
    ValidateResourceUse({A::DepthRead, {{V::DepthStencil}}});
    CHECK(Rejects([] { ValidateResourceUse({A::ShaderResource, {{V::UnorderedAccess}}}); }));
    CHECK(Rejects([] { ValidateResourceUse({A::CopySource, {{V::ShaderResource}}}); }));
    auto layout = std::make_shared<ResourceUseLayout>();
    layout->uses.push_back({{}, {A::ShaderResource, rhi::ResourceLayout::Common, rhi::ResourceSyncState::All},
        {41, 41}, {{V::ShaderResource, UINT32_MAX, 0, 0}, {V::ShaderResource, 3, 1, 2}}});
    DeclaredViewToken first{layout, 0, 0}, variant{layout, 0, 1};
    auto owner = std::make_shared<int>(1);
    auto makeContext = [&](uint32_t index) {
        FramePreparationContext context;
        context.resourceUses = layout;
        auto views = std::make_shared<BindlessResourceViews>();
        views->views = {{V::ShaderResource, UINT32_MAX, 0, 0, {{3,1},index}}, {V::ShaderResource, 3, 1, 2, {{3,1},index+1}}};
        context.bindings = std::make_shared<FrozenExecutionBindings>(std::vector<FrozenExecutionBindings::ResourceBinding>{
            {rhi::Resource({10,1}), owner, views, owner, nullptr}});
        context.resourceSlots = std::make_shared<FramePreparationContext::ResourceSlots>(
            FramePreparationContext::ResourceSlots{{41,0}});
        context.dependencyCollector = std::make_shared<PreparedDependencyCollector>();
        return context;
    };
    auto old = makeContext(7), fresh = makeContext(19);
    old.ValidateDeclaredViews(); fresh.ValidateDeclaredViews();
    CHECK(old.Resolve(first).index == 7 && fresh.Resolve(first).index == 19);
    CHECK(old.Resolve(variant).index == 8 && fresh.Resolve(variant).index == 20);
    (void)old.Capture(first);
    // A resolver republishes under a new global ID without redeclaring the pass.
    auto rotated = fresh;
    rotated.resourceSlots = std::make_shared<FramePreparationContext::ResourceSlots>(
        FramePreparationContext::ResourceSlots{{99,0}});
    rotated.resolveDeclaredResource = [](uint32_t) { return rhi::Resource({10,1}); };
    rotated.resolveDeclaredView = [](uint32_t, uint32_t view) { return rhi::DescriptorSlot({3,1}, 19 + view); };
    rotated.ValidateDeclaredViews();
    CHECK(rotated.Resolve(first).index == 19 && rotated.Resolve(variant).index == 20);
    (void)rotated.Capture(first);

    auto foreign = first; foreign.layout = std::make_shared<ResourceUseLayout>(*layout);
    CHECK(Rejects([&] { old.Resolve(foreign); }));
    auto invalid = first; invalid.view = 2;
    CHECK(Rejects([&] { old.Resolve(invalid); }));
    auto stale = old; stale.resourceUses = foreign.layout;
    CHECK(Rejects([&] { stale.Resolve(first); }));
    struct Row { uint32_t value = 1, a = UINT32_MAX, b = UINT32_MAX; };
    DeclaredTableLayout<Row> table(2);
    table.Field(0, &Row::a).Bind(first);
    table.Field(1, &Row::b).Bind(first);
    table.Field(0, &Row::b).Bind(variant);
    CHECK(Rejects([&] { table.Field(2, &Row::a); }));
    CHECK(Rejects([&] { table.Field(0, &Row::a).Bind(first); }));
    CHECK(Rejects([&] { table.Field(1, &Row::a).Bind({}); }));
    std::vector<Row> rows(2);
    const auto oldRows = table.Resolve(old, rows), newRows = table.Resolve(fresh, rows);
    CHECK(oldRows[0].a == 7 && newRows[0].a == 19 && oldRows[0].b == 8);
    CHECK(oldRows[1].b == 7 && oldRows[1].a == UINT32_MAX);
    rows[0].value = 23;
    CHECK(table.Resolve(old, rows)[0].value == 23 && table.SameLayout(table));
    DeclaredTableLayout<Row> remapped(2);
    remapped.Field(1, &Row::a).Bind(first);
    CHECK(!table.SameLayout(remapped));
    CHECK(Rejects([&] { table.Resolve(old, std::span<const Row>{}); }));

    // Declare before materialization, then rotate owned descriptor snapshots.
    persistent::GraphProgram program;
    auto edit = program.BeginEdit();
    const auto group = edit.AddGroup({2,3,true}, {});
    const auto members = edit.ReserveGroupMembers(group, 2);
    const auto slot = members[0];
    const auto pass = edit.AddPass({});
    edit.DeclareGroupAccess(pass, group, {2,0,1,false}, 0);
    const auto binding = edit.BindGroupMember(pass, group, slot);
    const auto future = edit.BindGroupMember(pass, group, members[1]);
    edit.RequireView(future, {V::ShaderResource, 3, 1, 2});
    const auto view = edit.RequireView(binding, {V::ShaderResource, 3, 1, 2});
    auto snapshot = std::make_shared<ResourceBindingSnapshot>();
    snapshot->resource = rhi::Resource({10,1}); snapshot->backingGeneration = 1;
    snapshot->allocationOwner = owner; snapshot->descriptorOwner = owner;
    snapshot->views = old.bindings->Resources()[0].views;
    persistent::BindingVersion version;
    version.identity = (uint64_t{1} << 32) | 10; version.backingRevision = 1;
    version.shape = {2,3,true}; version.owner = owner; version.recording = snapshot;
    edit.BindReserved(slot, version);
    experimental::CompileWorkspace workspace;
    std::atomic_bool cancelled{false};
    const auto held = edit.Build(workspace, cancelled);
    CHECK(held && program.Install(edit, held));
    CHECK(held->ResolveView(view).index == 8);
    CHECK(held->logical->declarations.passes[pass.index].accesses.empty());
    auto join = program.BeginEdit();
    auto unavailable = version;
    auto missingViews = std::make_shared<ResourceBindingSnapshot>(*snapshot);
    missingViews->views = std::make_shared<BindlessResourceViews>();
    unavailable.recording = missingViews;
    CHECK(Rejects([&] { join.BindReserved(members[1], unavailable); }));
    auto rotation = program.BeginEdit();
    auto replacement = std::make_shared<ResourceBindingSnapshot>(*snapshot);
    replacement->views = fresh.bindings->Resources()[0].views;
    version.recording = replacement; ++version.descriptorRevision;
    rotation.ReplaceBinding(slot, version);
    const auto selected = rotation.Build(workspace, cancelled);
    CHECK(selected->ResolveView(view).index == 20 && held->ResolveView(view).index == 8);
    auto missing = program.BeginEdit();
    replacement = std::make_shared<ResourceBindingSnapshot>(*snapshot);
    replacement->views = std::make_shared<BindlessResourceViews>(); version.recording = replacement;
    CHECK(Rejects([&] { missing.ReplaceBinding(slot, version); }));
    auto unbound = program.BeginEdit();
    const auto placeholder = unbound.ReserveResource({1,1,false});
    const auto strict = unbound.DeclareDependency(pass, placeholder, false);
    CHECK(Rejects([&] { unbound.DeclareView(strict, {}); }));
    return 0;
}
