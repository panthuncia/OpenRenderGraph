#pragma once

#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <rhi.h>

namespace org {

enum class BindlessViewKind : uint8_t {
    ShaderResource, UnorderedAccess, NonShaderVisibleUnorderedAccess,
    RenderTarget, DepthStencil, ConstantBuffer
};

struct BindlessViewRequest {
    BindlessViewKind kind = BindlessViewKind::ShaderResource;
    uint32_t variant = UINT32_MAX; // Default view for the requested kind.
    uint32_t mip = 0;
    uint32_t slice = 0;
};

// Value snapshot copied from a published resource version. Slots are immutable
// for that version and their ownership is carried by FrozenExecutionBindings.
struct BindlessResourceViews {
    struct View {
        BindlessViewKind kind{};
        uint32_t variant = UINT32_MAX;
        uint32_t mip = 0;
        uint32_t slice = 0;
        rhi::DescriptorSlot descriptor{};
    };
    rhi::ResourceDesc description{};
    rhi::ClearValue clear{};
    bool hasClear = false;
    uint32_t defaultSrvVariant = UINT32_MAX;
    std::vector<View> views;

    rhi::DescriptorSlot Resolve(BindlessViewRequest request) const {
        if (request.kind == BindlessViewKind::ShaderResource && request.variant == UINT32_MAX)
            request.variant = defaultSrvVariant;
        for (const auto& view : views)
            if (view.kind == request.kind && view.variant == request.variant
                && view.mip == request.mip && view.slice == request.slice)
                return view.descriptor;
        throw std::out_of_range("Declared resource does not publish the requested bindless view; kind="
            + std::to_string(static_cast<uint32_t>(request.kind))
            + " variant=" + std::to_string(request.variant)
            + " mip=" + std::to_string(request.mip)
            + " slice=" + std::to_string(request.slice)
            + " available=" + std::to_string(views.size()));
    }
    uint32_t SliceCount(BindlessViewRequest request) const noexcept {
        if (request.kind == BindlessViewKind::ShaderResource && request.variant == UINT32_MAX)
            request.variant = defaultSrvVariant;
        uint32_t count = 0;
        for (const auto& view : views)
            if (view.kind == request.kind && view.variant == request.variant && view.mip == request.mip)
                count = (std::max)(count, view.slice + 1);
        return count;
    }
};

} // namespace org
