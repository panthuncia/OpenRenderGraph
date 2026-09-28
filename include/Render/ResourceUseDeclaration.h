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
    uint32_t memberOrdinal = 0;
};

// One immutable layout per pass declaration. Tokens retain identity, never live descriptors.
struct ResourceUseLayout {
    struct ResolverViews { std::shared_ptr<const void> identity; std::vector<BindlessViewRequest> views; };
    std::vector<ResolverViews> resolverViews;
    std::vector<ResourceUseDeclaration> uses;
};
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
