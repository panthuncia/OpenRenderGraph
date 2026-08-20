#pragma once

#include <memory>
#include <rhi.h>
#include <resource_states.h>

#include "Resources/GloballyIndexedResource.h"
#include "Resources/ResourceStateTracker.h"
#include "Resources/TextureDescription.h"

namespace org {

// Graph wrapper for an externally allocated texture. The legacy handle constructor remains
// non-owning; CreateShared retains an imported BasicRHI resource for generation-safe interop.
class ExternalTextureResource : public GloballyIndexedResource {
public:
    ExternalTextureResource(rhi::ResourceHandle handle, unsigned int width, unsigned int height,
        rhi::Format format = rhi::Format::Unknown)
        : m_handle(handle), m_width(width), m_height(height)
    {
        m_hasLayout = true;
        m_mipLevels = 1;
        m_arraySize = 1;
        m_description.format = format;
        m_description.imageDimensions.push_back({ width, height, 0, 0 });
        ResetToUndefined();
    }

    static std::shared_ptr<ExternalTextureResource> CreateShared(
        rhi::ResourcePtr resource, const TextureDescription& description,
        bool commonLayoutOnly = true);

    // Replace only the native backing of an existing graph resource. Returns
    // false when the new description would change the structural graph and
    // therefore requires a new graph generation.
    bool RefreshShared(rhi::ResourcePtr resource, const TextureDescription& description,
        bool commonLayoutOnly = true);
    bool IsStructurallyCompatible(const TextureDescription& description) const noexcept;

    rhi::Resource GetAPIResource() override {
        return m_resource ? m_resource.Get() : rhi::Resource(m_handle, true);
    }

    rhi::BarrierBatch GetEnhancedBarrierGroup(
        RangeSpec range,
        rhi::ResourceAccessType previousAccess,
        rhi::ResourceAccessType nextAccess,
        rhi::ResourceLayout previousLayout,
        rhi::ResourceLayout nextLayout,
        rhi::ResourceSyncState previousSync,
        rhi::ResourceSyncState nextSync) override;

    SymbolicTracker* GetStateTracker() override { return &m_stateTracker; }
    bool TryGetRHIResourceDesc(rhi::ResourceDesc& outDesc) const override;
    rhi::Format GetFormat() const { return m_description.format; }
    unsigned int GetWidth() const { return m_width; }
    unsigned int GetHeight() const { return m_height; }

    void SetHandle(rhi::ResourceHandle handle) { m_handle = handle; }
    rhi::ResourceHandle GetHandle() const { return m_handle; }
    bool HasHandle() const { return m_handle.valid(); }
    void SetDimensions(unsigned int width, unsigned int height) { m_width = width; m_height = height; }
    void SetRTVSlot(rhi::DescriptorSlot slot) { m_rtvSlot = slot; }
    bool HasRTVSlot() const { return m_rtvSlot.heap.valid(); }
    rhi::DescriptorSlot GetRTVSlot() const { return m_rtvSlot; }

    void ResetToUndefined();
    void ResetToCommon();
	void ResetState(ResourceState state);

private:
    rhi::ResourcePtr m_resource;
    rhi::ResourceHandle m_handle{};
    unsigned int m_width{};
    unsigned int m_height{};
    rhi::TextureBarrier m_barrier{};
    SymbolicTracker m_stateTracker;
    rhi::DescriptorSlot m_rtvSlot{};
    // D3D11 shared textures are exposed by D3D12 with
    // RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS. Enhanced barriers may change
    // their access and sync scopes, but their layout must remain COMMON.
    bool m_commonLayoutOnly{};
    TextureDescription m_description{};
};

} // namespace org
