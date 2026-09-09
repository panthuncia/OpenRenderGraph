#include "Resources/PixelBuffer.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>

#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Resources/GPUBacking/GPUTextureBacking.h"
#include "Utilities/ORGUtilities.h"


namespace org {

namespace {
uint16_t ResolveTextureMipLevels(const TextureDescription& desc)
{
    if (desc.imageDimensions.empty()) {
        return 1;
    }

    const uint32_t totalArraySlices = desc.isCubemap
        ? 6u * desc.arraySize
        : (desc.isArray ? desc.arraySize : 1u);

    if (totalArraySlices > 0 &&
        desc.imageDimensions.size() > totalArraySlices &&
        (desc.imageDimensions.size() % totalArraySlices) == 0)
    {
        return static_cast<uint16_t>(desc.imageDimensions.size() / totalArraySlices);
    }

    if (desc.generateMipMaps) {
        uint32_t width = desc.imageDimensions[0].width;
        uint32_t height = desc.imageDimensions[0].height;
        if (desc.padInternalResolution) {
            width = (std::max)(1u, static_cast<uint32_t>(std::pow(2, std::ceil(std::log2(width)))));
            height = (std::max)(1u, static_cast<uint32_t>(std::pow(2, std::ceil(std::log2(height)))));
        }
        return org::util::CalculateMipLevels(width, height);
    }

    return 1;
}

DescriptorHeapManager::ViewRequirements::TextureViews BuildTextureViewRequirements(
    const TextureDescription& desc,
    uint32_t mipLevels,
    uint32_t totalArraySlices)
{
    DescriptorHeapManager::ViewRequirements::TextureViews texViews;
    texViews.mipLevels = mipLevels;
    texViews.isCubemap = desc.isCubemap;
    texViews.isArray = desc.isArray;
    texViews.arraySize = desc.arraySize;
    texViews.totalArraySlices = totalArraySlices;

    texViews.baseFormat = desc.format;
    texViews.srvFormat = desc.srvFormat;
    texViews.uavFormat = desc.uavFormat;
    texViews.rtvFormat = desc.rtvFormat;
    texViews.dsvFormat = desc.dsvFormat;

    texViews.createSRV = true;
    texViews.createUAV = desc.hasUAV;
    texViews.createNonShaderVisibleUAV = desc.hasNonShaderVisibleUAV;
    texViews.createRTV = desc.hasRTV;
    texViews.createDSV = desc.hasDSV;

    if (desc.hasUAV && rhi::helpers::IsSRGB(desc.format)) {
        if (texViews.srvFormat == rhi::Format::Unknown) {
            texViews.srvFormat = desc.format;
        }
        texViews.baseFormat = rhi::helpers::typlessFromSrgb(desc.format);
        texViews.uavFormat = rhi::helpers::stripSrgb(desc.format);
    }

    texViews.createCubemapAsArraySRV = desc.isCubemap;
    texViews.uavFirstMip = 0;
    return texViews;
}
}

PixelBuffer::PixelBuffer(const TextureDescription& desc, bool materialize)
{
    m_hasLayout = true;
    m_desc = desc;
    if (materialize) {
        Materialize();
    }
    if (desc.padInternalResolution) {
        m_internalWidth = (std::max)(1u, static_cast<unsigned int>(std::pow(2, std::ceil(std::log2(desc.imageDimensions[0].width)))));
        m_internalHeight = (std::max)(1u, static_cast<unsigned int>(std::pow(2, std::ceil(std::log2(desc.imageDimensions[0].height)))));
    }
    else {
        m_internalHeight = desc.imageDimensions[0].height;
        m_internalWidth = desc.imageDimensions[0].width;
    }

    m_clearValue.type = rhi::ClearValueType::Color;
    m_clearValue.format = desc.format;
    m_clearValue.depthStencil.depth = desc.depthClearValue;
    if (desc.hasDSV) {
        m_clearValue.type = rhi::ClearValueType::DepthStencil;
    }
    else {
        for (int i = 0; i < 4; i++) {
            m_clearValue.rgba[i] = desc.clearColor[i];
        }
    }
}

PixelBuffer::~PixelBuffer() = default;

rhi::Resource PixelBuffer::GetAPIResource() {
    if (auto attached = GetAttachedAPIRepresentation(BackendInstanceId::Primary)) return attached;
    std::scoped_lock lock(m_materializationMutex);
    EnsureMaterializedLocked("GetAPIResource");
    return m_backing->GetAPIResource();
}

rhi::BarrierBatch PixelBuffer::GetEnhancedBarrierGroup(
    RangeSpec range,
    rhi::ResourceAccessType prevAccessType,
    rhi::ResourceAccessType newAccessType,
    rhi::ResourceLayout prevLayout,
    rhi::ResourceLayout newLayout,
    rhi::ResourceSyncState prevSyncState,
    rhi::ResourceSyncState newSyncState)
{
	if (GetAttachedAPIRepresentation(BackendInstanceId::Primary)) {
		return Resource::GetEnhancedBarrierGroup(BackendInstanceId::Primary, range,
			prevAccessType, newAccessType, prevLayout, newLayout, prevSyncState, newSyncState);
	}
    std::scoped_lock lock(m_materializationMutex);
    EnsureMaterializedLocked("GetEnhancedBarrierGroup");
    return m_backing->GetEnhancedBarrierGroup(
        range, prevAccessType, newAccessType, prevLayout, newLayout, prevSyncState, newSyncState
    );
}

void PixelBuffer::OnSetName() {
    std::scoped_lock lock(m_materializationMutex);
    if (m_backing && m_backing->HasValidResource()) {
        m_backing->SetName(name.c_str());
    }
}

bool PixelBuffer::HasValidBackingResource() const {
	if (GetAttachedAPIRepresentation(BackendInstanceId::Primary)) return true;
    std::scoped_lock lock(m_materializationMutex);
    return m_backing && m_backing->HasValidResource();
}

void PixelBuffer::ApplyMetadataComponentBundle(const EntityComponentBundle& bundle) const {
    std::scoped_lock lock(m_materializationMutex);
    m_metadataBundles.emplace_back(bundle);
    if (!m_backing) {
        return;
    }
    m_backing->ApplyMetadataComponentBundle(bundle);
}

SymbolicTracker* PixelBuffer::GetStateTracker() {
    if (auto* attached = GetAttachedStateTracker(BackendInstanceId::Primary)) return attached;
    std::scoped_lock lock(m_materializationMutex);
    EnsureMaterializedLocked("GetStateTracker");
    return m_backing->GetStateTracker();
}

bool PixelBuffer::TryGetRHIResourceDesc(rhi::ResourceDesc& outDesc) const {
	if (m_desc.imageDimensions.empty() || m_desc.type != rhi::ResourceType::Texture2D || m_desc.sampleCount != 1 || m_desc.hasDSV) return false;
	const uint32_t arraySize = m_desc.isCubemap ? 6u * m_desc.arraySize : (m_desc.isArray ? m_desc.arraySize : 1u);
	uint32_t width = m_desc.imageDimensions[0].width;
	uint32_t height = m_desc.imageDimensions[0].height;
	if (m_desc.padInternalResolution) {
		width = (std::max)(1u, static_cast<uint32_t>(std::bit_ceil(width)));
		height = (std::max)(1u, static_cast<uint32_t>(std::bit_ceil(height)));
	}
	outDesc = {};
	outDesc.type = rhi::ResourceType::Texture2D;
	outDesc.heapType = rhi::HeapType::DeviceLocal;
	outDesc.texture.format = m_desc.format;
	outDesc.texture.width = width;
	outDesc.texture.height = height;
	outDesc.texture.depthOrLayers = static_cast<uint16_t>(arraySize);
	outDesc.texture.mipLevels = static_cast<uint16_t>(m_desc.generateMipMaps
		? org::util::CalculateMipLevels(width, height)
		: (m_desc.imageDimensions.size() > arraySize ? m_desc.imageDimensions.size() / arraySize : 1));
	outDesc.texture.sampleCount = 1;
	outDesc.texture.initialLayout = rhi::ResourceLayout::Undefined;
	if (m_desc.hasRTV) outDesc.resourceFlags |= rhi::ResourceFlags::RF_AllowRenderTarget;
	if (m_desc.hasUAV) outDesc.resourceFlags |= rhi::ResourceFlags::RF_AllowUnorderedAccess;
	return outDesc.texture.format != rhi::Format::Unknown;
}

void PixelBuffer::RefreshAPIRepresentationDescriptors(BackendInstanceId backendInstance) {
	// Multi-RHI materialization installs an explicit representation for every
	// participating device, including the primary device.  The original backing
	// may already have populated the primary descriptor arena, so it is essential
	// to overwrite those descriptors with views of the new shared representation.
	// Otherwise barriers/copies resolve the shared resource while shaders and
	// render passes continue to access the superseded primary-only backing.
	auto resource = Resource::GetAPIResource(backendInstance);
	if (!resource) return;
	// An attached representation replaces (and may be created before) the normal
	// GpuTextureBacking.  Do not inherit cached dimensions from that superseded
	// backing: the logical description is authoritative for every representation.
	const uint16_t resolvedMipLevels = ResolveTextureMipLevels(m_desc);
	const uint32_t resolvedArraySize = m_desc.isCubemap
		? 6u * m_desc.arraySize : (m_desc.isArray ? m_desc.arraySize : 1u);
	{
		std::scoped_lock lock(m_materializationMutex);
		m_mipLevels = resolvedMipLevels;
		m_arraySize = resolvedArraySize;
	}
	EnsureVirtualDescriptorSlotsAllocated();
	DescriptorHeapManager::ViewRequirements views;
	views.views = BuildTextureViewRequirements(m_desc, resolvedMipLevels, resolvedArraySize);
	DescriptorHeapManager::GetInstance().UpdateDescriptorContents(*this, resource, views, backendInstance);
}

void PixelBuffer::Materialize(const MaterializeOptions* options) {
	ClearAPIRepresentations();
    std::scoped_lock lock(m_materializationMutex);
    if (m_backing) {
        return;
    }

    EnsureVirtualDescriptorSlotsAllocatedLocked();

    auto newDesc = m_desc;
    if (m_desc.padInternalResolution) {
        for (auto& dim : newDesc.imageDimensions) {
            dim.width = (std::max)(1u, static_cast<unsigned int>(std::pow(2, std::ceil(std::log2(dim.width)))));
            dim.height = (std::max)(1u, static_cast<unsigned int>(std::pow(2, std::ceil(std::log2(dim.height)))));
        }
    }

    if (options && options->aliasPlacement.has_value()) {
        m_backing = GpuTextureBacking::CreateUnique(newDesc, GetGlobalResourceID(), options->aliasPlacement.value(), name.empty() ? nullptr : name.c_str());
    }
    else {
        m_backing = GpuTextureBacking::CreateUnique(newDesc, GetGlobalResourceID(), name.empty() ? nullptr : name.c_str());
    }

    m_mipLevels = m_backing->GetMipLevels();
    m_arraySize = m_backing->GetArraySize();

    auto& rm = DescriptorHeapManager::GetInstance();
    DescriptorHeapManager::ViewRequirements views;
    views.views = BuildTextureViewRequirements(m_desc, m_mipLevels, m_arraySize);
    auto res = m_backing->GetAPIResource();
    rm.UpdateDescriptorContents(*this, res, views);

    if (m_desc.aliasingPoolID.has_value()) {
        m_backing->ApplyMetadataComponentBundle(
            EntityComponentBundle().Set<MemoryStatisticsComponents::AliasingPool>({ m_desc.aliasingPoolID })
        );
    }

    for (const auto& bundle : m_metadataBundles) {
        m_backing->ApplyMetadataComponentBundle(bundle);
    }

    ++m_backingGeneration;
}

BackingAllocationSnapshot PixelBuffer::CaptureBackingAllocation() {
    std::scoped_lock lock(m_materializationMutex);
    if (!m_backing || GetAttachedAPIRepresentation(BackendInstanceId::Primary).IsValid()) return {};
    auto lease = m_backing->CaptureAllocationLease();
    if (!lease) return {};
    BackingAllocationSnapshot snapshot{
        GetGlobalResourceID(), m_backingGeneration, m_backing->GetAPIResource(), std::move(lease)};
    snapshot.aliasHeap = m_backing->GetAliasHeap();
    snapshot.aliasPoolID = m_backing->GetAliasPoolID();
    snapshot.aliasOffset = m_backing->GetAliasOffset();
    snapshot.aliasSize = m_backing->GetAliasSize();
    return snapshot;
}

void PixelBuffer::Dematerialize() {
	const bool hadAttachedPrimary = GetAttachedAPIRepresentation(BackendInstanceId::Primary).IsValid();
	ClearAPIRepresentations();
    std::scoped_lock lock(m_materializationMutex);
    if (!m_backing) {
		if (hadAttachedPrimary) {
			RotateDescriptorSlotsForPublication();
			++m_backingGeneration;
		}
        return;
    }

	RotateDescriptorSlotsForPublication();
    m_backing.reset();
    ++m_backingGeneration;
}

void PixelBuffer::EnsureVirtualDescriptorSlotsAllocated() {
    std::scoped_lock lock(m_materializationMutex);
    EnsureVirtualDescriptorSlotsAllocatedLocked();
}

void PixelBuffer::EnsureVirtualDescriptorSlotsAllocatedLocked() {
    if (HasAnyDescriptorSlots()) {
        return;
    }

    auto& rm = DescriptorHeapManager::GetInstance();
    const uint16_t mipLevels = ResolveTextureMipLevels(m_desc);
    const uint32_t arraySize = m_desc.isCubemap
        ? 6u * m_desc.arraySize
        : (m_desc.isArray ? m_desc.arraySize : 1u);

    DescriptorHeapManager::ViewRequirements views;
    views.views = BuildTextureViewRequirements(m_desc, mipLevels, arraySize);
    rm.ReserveDescriptorSlots(*this, views);
}

void PixelBuffer::EnsureMaterializedLocked(const char* operation) const {
    if (m_backing) {
        return;
    }
    throw std::runtime_error(std::string("PixelBuffer '") + name + "' is unmaterialized during " + operation);
}

void PixelBuffer::ApplyMetadataComponentBundle(const EntityComponentBundle& bundle) {
    std::scoped_lock lock(m_materializationMutex);
    m_metadataBundles.emplace_back(bundle);
    if (m_backing) {
        m_backing->ApplyMetadataComponentBundle(bundle);
    }
}


} // namespace org
