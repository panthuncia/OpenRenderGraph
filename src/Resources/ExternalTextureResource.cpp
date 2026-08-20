#include "Resources/ExternalTextureResource.h"

#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Utilities/ORGUtilities.h"
#include <rhi_interop_dx12.h>
#include <d3d12.h>

namespace org {

bool ExternalTextureResource::TryGetRHIResourceDesc(rhi::ResourceDesc& outDesc) const
{
	if (m_description.imageDimensions.empty()) return false;
	outDesc = {};
	outDesc.type = rhi::ResourceType::Texture2D;
	outDesc.heapType = rhi::HeapType::DeviceLocal;
	outDesc.texture.format = m_description.format;
	outDesc.texture.width = m_width;
	outDesc.texture.height = m_height;
	outDesc.texture.depthOrLayers = m_arraySize;
	outDesc.texture.mipLevels = m_mipLevels;
	outDesc.texture.sampleCount = 1;
	return outDesc.texture.format != rhi::Format::Unknown;
}

namespace {
DescriptorHeapManager::ViewRequirements BuildExternalTextureViews(
	const TextureDescription& description, uint32_t mipLevels, uint32_t arraySize)
{
	DescriptorHeapManager::ViewRequirements requirements{};
	DescriptorHeapManager::ViewRequirements::TextureViews views{};
	views.mipLevels = mipLevels;
	views.isCubemap = description.isCubemap;
	views.isArray = description.isArray;
	views.arraySize = description.arraySize;
	views.totalArraySlices = arraySize;
	views.baseFormat = description.format;
	views.srvFormat = description.srvFormat;
	views.uavFormat = description.uavFormat;
	views.rtvFormat = description.rtvFormat;
	views.dsvFormat = description.dsvFormat;
	views.createSRV = description.hasSRV;
	views.createUAV = description.hasUAV;
	views.createNonShaderVisibleUAV = description.hasNonShaderVisibleUAV;
	views.createRTV = description.hasRTV;
	views.createDSV = description.hasDSV;
	views.createCubemapAsArraySRV = description.isCubemap && description.hasSRV;
	requirements.views = views;
	return requirements;
}
}

std::shared_ptr<ExternalTextureResource> ExternalTextureResource::CreateShared(
    rhi::ResourcePtr resource, const TextureDescription& description, bool commonLayoutOnly)
{
	if (!resource || description.imageDimensions.empty())
		return {};
    auto result = std::shared_ptr<ExternalTextureResource>(new ExternalTextureResource(
        resource.Get().GetHandle(), description.imageDimensions[0].width, description.imageDimensions[0].height));
    result->m_resource = std::move(resource);
	result->m_description = description;
    result->m_mipLevels = static_cast<uint16_t>((std::max)(size_t{ 1 }, description.imageDimensions.size()));
    result->m_arraySize = static_cast<uint16_t>(description.isCubemap ? 6u * description.arraySize :
        (description.isArray ? description.arraySize : 1u));
    // CreateShared is the retained cross-API import path. Such resources have
    // the simultaneous-access contract even on proxy devices that do not
    // faithfully expose every native creation flag through GetDesc(). The
    // legacy constructor remains available for ordinary external textures.
	result->m_commonLayoutOnly = commonLayoutOnly;
	auto views = BuildExternalTextureViews(description, result->m_mipLevels, result->m_arraySize);
	auto apiResource = result->m_resource.Get();
	DescriptorHeapManager::GetInstance().AssignDescriptorSlots(*result, apiResource, views);
    if (description.initialLayout == rhi::ResourceLayout::Common)
        result->ResetToCommon();
    return result;
}

bool ExternalTextureResource::IsStructurallyCompatible(const TextureDescription& description) const noexcept
{
	if (description.imageDimensions.empty() || m_description.imageDimensions.size() != description.imageDimensions.size())
		return false;
	if (m_description.format != description.format || m_description.channels != description.channels ||
		m_description.isCubemap != description.isCubemap || m_description.isArray != description.isArray ||
		m_description.arraySize != description.arraySize || m_description.hasRTV != description.hasRTV ||
		m_description.rtvFormat != description.rtvFormat || m_description.hasDSV != description.hasDSV ||
		m_description.dsvFormat != description.dsvFormat || m_description.hasUAV != description.hasUAV ||
		m_description.uavFormat != description.uavFormat || m_description.hasSRV != description.hasSRV ||
		m_description.srvFormat != description.srvFormat ||
		m_description.hasNonShaderVisibleUAV != description.hasNonShaderVisibleUAV)
		return false;
	for (size_t i = 0; i < description.imageDimensions.size(); ++i) {
		if (m_description.imageDimensions[i].width != description.imageDimensions[i].width ||
			m_description.imageDimensions[i].height != description.imageDimensions[i].height)
			return false;
	}
	return true;
}

bool ExternalTextureResource::RefreshShared(
	rhi::ResourcePtr resource, const TextureDescription& description, bool commonLayoutOnly)
{
	if (!resource || !IsStructurallyCompatible(description))
		return false;
	auto previous = std::move(m_resource);
	m_resource = std::move(resource);
	m_handle = m_resource.Get().GetHandle();
	m_commonLayoutOnly = commonLayoutOnly;
	m_description = description;
	if (description.initialLayout == rhi::ResourceLayout::Common)
		ResetToCommon();
	else
		ResetToUndefined();
	auto views = BuildExternalTextureViews(description, m_mipLevels, m_arraySize);
	auto apiResource = m_resource.Get();
	DescriptorHeapManager::GetInstance().UpdateDescriptorContents(*this, apiResource, views);
	DescriptorHeapManager::GetInstance().RetireNativeResource(std::move(previous));
	return true;
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
    thread_local rhi::TextureBarrier barrier{};
    barrier.beforeAccess = previousAccess;
    barrier.afterAccess = nextAccess;
    barrier.beforeLayout = m_commonLayoutOnly ? rhi::ResourceLayout::Common : previousLayout;
    barrier.afterLayout = m_commonLayoutOnly ? rhi::ResourceLayout::Common : nextLayout;
    barrier.beforeSync = previousSync;
    barrier.afterSync = nextSync;
    barrier.discard = false;
    barrier.range = { resolved.firstMip, resolved.mipCount, resolved.firstSlice, resolved.sliceCount };
    barrier.texture = m_handle;
    rhi::BarrierBatch batch{};
    batch.textures = { &barrier };
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

void ExternalTextureResource::ResetState(ResourceState state)
{
	RangeSpec wholeRange{};
	m_stateTracker = SymbolicTracker(wholeRange, state);
}

} // namespace org
