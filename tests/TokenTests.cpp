// Declaration layouts compared by content (SameLayoutContent).
//
// A pass that declares the same thing again gets a new layout object. The persistent layer treats it as the same
// declaration when the content is equal, so it builds nothing. These tests check what "equal" covers:
// - every part of a use (range, state including sync, views, resolver identity) and of the layout (resolver views,
//   named views, feature domains) makes layouts differ;
// - with resources ignored, layouts that only name other resources, use for use, are equal (a rebind).

#include "Render/ResourceUseDeclaration.h"
#include "TestSupport.h"

#include <cstdio>
#include <functional>
#include <memory>
#include <vector>

namespace {
using org::ResourceUseLayout;

ResourceUseLayout MakeLayout() {
    ResourceUseLayout layout;
    org::ResourceUseDeclaration read;
    read.state = {rhi::ResourceAccessType::ShaderResource, rhi::ResourceLayout::ShaderResource, rhi::ResourceSyncState::All};
    read.binding = {11, 12};
    read.requiredViews = {org::SrvView{}, org::SrvView{UINT32_MAX, 1, 0}};
    org::ResourceUseDeclaration write;
    write.state = {rhi::ResourceAccessType::UnorderedAccess, rhi::ResourceLayout::UnorderedAccess, rhi::ResourceSyncState::ComputeShading};
    write.binding = {21, 22};
    write.requiredViews = {org::UavView{}};
    layout.uses = {read, write};
    layout.resolverViews.push_back({std::make_shared<int>(1), {org::SrvView{}}});
    layout.namedViews.push_back({1234u, 0u, 0u, org::BindlessViewKind::ShaderResource});
    layout.activeFeatureDomains = {7u};
    return layout;
}

int TestEqual() {
    const auto a = MakeLayout();
    auto b = a;
    CHECK(org::SameLayoutContent(a, b, true));
    CHECK(org::SameLayoutContent(a, b, false));
    // An empty layout equals an empty layout.
    CHECK(org::SameLayoutContent(ResourceUseLayout{}, ResourceUseLayout{}, true));
    return 0;
}

// Each change must make the layouts differ, with or without resources.
int TestEachPartMatters() {
    const auto base = MakeLayout();
    const std::vector<std::pair<const char*, std::function<void(ResourceUseLayout&)>>> changes = {
        {"use count", [](auto& l) { l.uses.pop_back(); }},
        {"use order", [](auto& l) { std::swap(l.uses[0], l.uses[1]); }},
        {"range", [](auto& l) { l.uses[0].resource.range.mipLower = {org::BoundType::Exact, 1}; }},
        {"access", [](auto& l) { l.uses[0].state.access = rhi::ResourceAccessType::CopySource; }},
        {"layout", [](auto& l) { l.uses[0].state.layout = rhi::ResourceLayout::CopySource; }},
        {"sync", [](auto& l) { l.uses[0].state.sync = rhi::ResourceSyncState::Copy; }},
        {"view count", [](auto& l) { l.uses[0].requiredViews.pop_back(); }},
        {"view mip", [](auto& l) { l.uses[0].requiredViews[1].mip = 2; }},
        {"view kind", [](auto& l) { l.uses[1].requiredViews[0].kind = org::BindlessViewKind::ShaderResource; }},
        {"resolver identity", [](auto& l) { l.uses[0].resolverIdentity = std::make_shared<int>(2); }},
        {"resolver views identity", [](auto& l) { l.resolverViews[0].identity = std::make_shared<int>(3); }},
        {"resolver views", [](auto& l) { l.resolverViews[0].views.push_back(org::UavView{}); }},
        {"named view", [](auto& l) { l.namedViews[0].view = 1; }},
        {"named view hash", [](auto& l) { l.namedViews[0].nameHash = 99; }},
        {"named group", [](auto& l) { l.namedViews[0].resolverGroup = 0; }},
        {"feature domains", [](auto& l) { l.activeFeatureDomains.push_back(8); }},
    };
    for (const auto& [name, change] : changes) {
        auto changed = base;
        change(changed);
        if (org::SameLayoutContent(base, changed, true) || org::SameLayoutContent(base, changed, false)) {
            std::fprintf(stderr, "layout change not detected: %s\n", name);
            return __LINE__;
        }
    }
    return 0;
}

// Other resources of the same declaration: different with resources, equal without.
int TestResourcesOnly() {
    const auto base = MakeLayout();
    const std::vector<std::function<void(ResourceUseLayout&)>> changes = {
        [](auto& l) { l.uses[0].binding.globalResourceID = 99; },
        [](auto& l) { l.uses[1].binding.registryResourceID = 98; },
    };
    for (const auto& change : changes) {
        auto changed = base;
        change(changed);
        CHECK(!org::SameLayoutContent(base, changed, true));
        CHECK(org::SameLayoutContent(base, changed, false));
    }
    return 0;
}
} // namespace

int main() {
    if (const int line = TestEqual()) return line;
    if (const int line = TestEachPartMatters()) return line;
    if (const int line = TestResourcesOnly()) return line;
    std::puts("OpenRenderGraphTokenTests passed");
    return 0;
}
