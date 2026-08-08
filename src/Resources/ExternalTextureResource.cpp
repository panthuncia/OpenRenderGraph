#include "Resources/ExternalTextureResource.h"

#include "Utilities/ORGUtilities.h"
#include <rhi_interop_dx12.h>
#include <d3d12.h>

namespace org {

std::shared_ptr<ExternalTextureResource> ExternalTextureResource::CreateShared(
    rhi::ResourcePtr resource, const TextureDescription& description)
{
    auto result = std::shared_ptr<ExternalTextureResource>(new ExternalTextureResource(
        resource.Get().GetHandle(), description.imageDimensions[0].width, description.imageDimensions[0].height));
    result->m_resource = std::move(resource);
    result->m_mipLevels = static_cast<uint16_t>((std::max)(size_t{ 1 }, description.imageDimensions.size()));
    result->m_arraySize = static_cast<uint16_t>(description.isCubemap ? 6u * description.arraySize :
        (description.isArray ? description.arraySize : 1u));
    // CreateShared is the retained cross-API import path. Such resources have
    // the simultaneous-access contract even on proxy devices that do not
    // faithfully expose every native creation flag through GetDesc(). The
    // legacy constructor remains available for ordinary external textures.
    result->m_commonLayoutOnly = true;
    if (description.initialLayout == rhi::ResourceLayout::Common)
        result->ResetToCommon();
    return result;
}

rhi::BarrierBatch ExternalTextureResource::GetEnhancedBarrierGroup(
    RangeSpec range,
    rhi::ResourceAccessType previousAccess,
    rhi::ResourceAccessType nextAccess,
    rhi::ResourceLayout previousLayout,
    rhi::ResourceLayout nextLayout,
    rhi::ResourceSyncState previousSync,
    rhi::ResourceSyncState nextSync)
{
    const auto resolved = ResolveRangeSpec(range, m_mipLevels, m_arraySize);
    m_barrier.beforeAccess = previousAccess;
    m_barrier.afterAccess = nextAccess;
    m_barrier.beforeLayout = m_commonLayoutOnly ? rhi::ResourceLayout::Common : previousLayout;
    m_barrier.afterLayout = m_commonLayoutOnly ? rhi::ResourceLayout::Common : nextLayout;
    m_barrier.beforeSync = previousSync;
    m_barrier.afterSync = nextSync;
    m_barrier.discard = false;
    m_barrier.range = { resolved.firstMip, resolved.mipCount, resolved.firstSlice, resolved.sliceCount };
    m_barrier.texture = m_handle;
    rhi::BarrierBatch batch{};
    batch.textures = { &m_barrier };
    return batch;
}

void ExternalTextureResource::ResetToUndefined()
{
    RangeSpec wholeRange{};
    wholeRange.mipLower = { BoundType::All, 0 };
    wholeRange.mipUpper = { BoundType::All, 0 };
    wholeRange.sliceLower = { BoundType::All, 0 };
    wholeRange.sliceUpper = { BoundType::All, 0 };
    m_stateTracker = SymbolicTracker(wholeRange, ResourceState{
        rhi::ResourceAccessType::None, rhi::ResourceLayout::Undefined, rhi::ResourceSyncState::None });
}

void ExternalTextureResource::ResetToCommon()
{
    m_stateTracker = SymbolicTracker{};
}

} // namespace org
