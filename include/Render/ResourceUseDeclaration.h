#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>
#include "Render/BindlessResourceViews.h"
#include "Render/ResourceRequirements.h"

namespace org {

struct ResourceBindingToken {
    uint64_t globalResourceID = 0;
    uint64_t registryResourceID = 0;
};

struct SrvView {
    uint32_t variant = UINT32_MAX, mip = 0, slice = 0;
    operator BindlessViewRequest() const { return {BindlessViewKind::ShaderResource, variant, mip, slice}; }
};
struct UavView {
    uint32_t variant = UINT32_MAX, mip = 0, slice = 0;
    operator BindlessViewRequest() const { return {BindlessViewKind::UnorderedAccess, variant, mip, slice}; }
};
struct RtvView {
    uint32_t variant = UINT32_MAX, mip = 0, slice = 0;
    operator BindlessViewRequest() const { return {BindlessViewKind::RenderTarget, variant, mip, slice}; }
};
struct DsvView {
    uint32_t variant = UINT32_MAX, mip = 0, slice = 0;
    operator BindlessViewRequest() const { return {BindlessViewKind::DepthStencil, variant, mip, slice}; }
};

struct ResourceUseSpecification {
    rhi::ResourceAccessType access;
    std::vector<BindlessViewRequest> views;
    // An acceleration structure build's input (shader resource) or output or scratch (unordered access): synchronized
    // with build operations as well as shaders, which the access alone can't say.
    bool accelerationStructureBuildStage = false;
};

inline bool SameView(BindlessViewRequest a, BindlessViewRequest b) {
    return a.kind == b.kind && a.variant == b.variant && a.mip == b.mip && a.slice == b.slice;
}

inline void ValidateResourceUse(const ResourceUseSpecification& specification) {
    using A = rhi::ResourceAccessType;
    using V = BindlessViewKind;
    for (const auto view : specification.views) {
        const bool compatible =
            (view.kind == V::ShaderResource && specification.access == A::ShaderResource) ||
            (view.kind == V::UnorderedAccess && (specification.access == A::UnorderedAccess || specification.access == A::UnorderedAccessClear)) ||
            (view.kind == V::NonShaderVisibleUnorderedAccess && specification.access == A::UnorderedAccessClear) ||
            (view.kind == V::ConstantBuffer && specification.access == A::ConstantBuffer) ||
            (view.kind == V::RenderTarget && (specification.access == A::RenderTarget || specification.access == A::RenderTargetClear)) ||
            (view.kind == V::DepthStencil && (specification.access == A::DepthRead || specification.access == A::DepthReadWrite || specification.access == A::DepthStencilClear));
        if (!compatible) throw std::invalid_argument("Descriptor view is incompatible with declared access");
    }
}

struct ResourceUseDeclaration {
    ResourceHandleAndRange resource;
    ResourceState state;
    ResourceBindingToken binding;
    std::vector<BindlessViewRequest> requiredViews;
    std::shared_ptr<const void> resolverIdentity;
};

// One immutable layout per pass declaration. Tokens retain identity, never live descriptors.
struct ResourceUseLayout {
    struct ResolverViews { std::shared_ptr<const void> identity; std::vector<BindlessViewRequest> views; };
    std::vector<ResolverViews> resolverViews;
    std::vector<ResourceUseDeclaration> uses;
    // Shader-visible views declared by resource name, with builder.ShaderResource(Builtin::X) and similar calls.
    // At record time, a pipeline's ResourceDescriptorIndex(Builtin::X) slots are filled from these. A name declared
    // both ways resolves to its unordered-access view. A name's resolver may have had no member when the pass declared
    // it. Such a name refers to its resolver group instead (resolverViews[resolverGroup]). Each frame then resolves
    // `request` on the group's first bound member.
    struct NamedView {
        size_t nameHash = 0;
        uint32_t use = UINT32_MAX, view = UINT32_MAX;
        BindlessViewKind kind{};
        uint32_t resolverGroup = UINT32_MAX;
        BindlessViewRequest request{};
    };
    std::vector<NamedView> namedViews;
    // Hashes of the feature domains (FeatureDomainRegistry) that the declarations named a resource from.
    std::vector<size_t> activeFeatureDomains;
    const NamedView* FindNamedView(size_t nameHash) const noexcept {
        for (const auto& named : namedViews) if (named.nameHash == nameHash) return &named;
        return nullptr;
    }
    // Position of (use, view) in a pass's frozen view table: the views of every use, in declaration order.
    uint32_t ViewCount() const noexcept {
        uint32_t count = 0;
        for (const auto& use : uses) count += static_cast<uint32_t>(use.requiredViews.size());
        return count;
    }
    uint32_t FlatView(uint32_t use, uint32_t view) const {
        if (use >= uses.size() || view >= uses[use].requiredViews.size()) throw std::invalid_argument("Invalid declared view ordinal");
        uint32_t flat = 0;
        for (uint32_t i = 0; i < use; ++i) flat += static_cast<uint32_t>(uses[i].requiredViews.size());
        return flat + view;
    }
};

// True when two layouts declare the same uses, views, names and feature domains. With `resources` false, the
// resources the uses name may differ: the second layout swapped resources for others, use for use.
inline bool SameLayoutContent(const ResourceUseLayout& a, const ResourceUseLayout& b, bool resources) {
    const auto sameViews = [](const std::vector<BindlessViewRequest>& x, const std::vector<BindlessViewRequest>& y) {
        if (x.size() != y.size()) return false;
        for (size_t i = 0; i < x.size(); ++i) if (!SameView(x[i], y[i])) return false;
        return true;
    };
    if (a.uses.size() != b.uses.size() || a.resolverViews.size() != b.resolverViews.size()
        || a.namedViews.size() != b.namedViews.size() || a.activeFeatureDomains != b.activeFeatureDomains) return false;
    for (size_t i = 0; i < a.uses.size(); ++i) {
        const auto& x = a.uses[i];
        const auto& y = b.uses[i];
        if (!(x.resource.range == y.resource.range) || !(x.state == y.state) || x.state.sync != y.state.sync
            || x.resolverIdentity != y.resolverIdentity || !sameViews(x.requiredViews, y.requiredViews)) return false;
        if (resources && (!(x.resource.resource == y.resource.resource)
            || x.binding.globalResourceID != y.binding.globalResourceID
            || x.binding.registryResourceID != y.binding.registryResourceID)) return false;
    }
    for (size_t i = 0; i < a.resolverViews.size(); ++i)
        if (a.resolverViews[i].identity != b.resolverViews[i].identity
            || !sameViews(a.resolverViews[i].views, b.resolverViews[i].views)) return false;
    for (size_t i = 0; i < a.namedViews.size(); ++i) {
        const auto& x = a.namedViews[i];
        const auto& y = b.namedViews[i];
        if (x.nameHash != y.nameHash || x.use != y.use || x.view != y.view || x.kind != y.kind
            || x.resolverGroup != y.resolverGroup || !SameView(x.request, y.request)) return false;
    }
    return true;
}

struct DeclaredViewToken {
    std::shared_ptr<const ResourceUseLayout> layout;
    uint32_t use = UINT32_MAX, view = UINT32_MAX;
    ResourceBindingToken Resource() const {
        if (!layout || use >= layout->uses.size()) throw std::invalid_argument("Invalid declared view token");
        return layout->uses[use].binding;
    }
};
struct DeclaredResourceUse {
    // Groups expose all members; scalar consumers must select one explicitly.
    std::vector<ResourceBindingToken> resources;
    std::vector<DeclaredViewToken> views;
    DeclaredViewToken View() const {
        if (views.size() != 1) throw std::invalid_argument("Resource use does not have one view; select a view index explicitly");
        return views.front();
    }
    DeclaredViewToken View(size_t index) const { return views.at(index); }
    operator DeclaredViewToken() const { return View(); }
    ResourceBindingToken Resource() const {
        if (resources.size() != 1) throw std::invalid_argument("Resource use is not scalar");
        return resources.front();
    }
};

} // namespace org
