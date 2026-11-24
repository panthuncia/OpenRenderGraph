#include "Resources/PixelBuffer.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <spdlog/spdlog.h>

#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Resources/GPUBacking/GPUTextureBacking.h"
#include "Utilities/ORGUtilities.h"


namespace org {

namespace {

DescriptorHeapManager::ViewRequirements::TextureViews BuildTextureViewRequirements(
    const TextureDescription& desc,
    uint32_t mipLevels,
    uint32_t totalArraySlices)
{
    DescriptorHeapManager::ViewRequirements::TextureViews texViews;
    texViews.mipLevels = mipLevels;
    texViews.isCubemap = desc.isCubemap;
    texViews.isArray = desc.isArray;
    texViews.is3D = desc.type == rhi::ResourceType::Texture3D;
    texViews.depth = desc.depth;
    texViews.arraySize = desc.arraySize;
    texViews.totalArraySlices = totalArraySlices;
    texViews.sampleCount = (std::max)(1u, static_cast<uint32_t>(desc.sampleCount));

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

// A multisampled texture is a 2D texture or array with one mip and no unordered access. D3D12 and Vulkan allow no
// other kind without extra features.
void RequireValidMultisampling(const TextureDescription& desc, uint32_t mipLevels)
{
    if (desc.sampleCount <= 1) return;
    if (desc.type != rhi::ResourceType::Texture2D || desc.isCubemap)
        throw std::invalid_argument("A multisampled texture must be a 2D texture or 2D array");
    if (mipLevels != 1 || desc.generateMipMaps)
        throw std::invalid_argument("A multisampled texture must have one mip");
    if (desc.hasUAV || desc.hasNonShaderVisibleUAV)
        throw std::invalid_argument("A multisampled texture has no unordered-access views");
}
}

PixelBuffer::PixelBuffer(const TextureDescription& desc, bool materialize)
{
    m_hasLayout = true;
    m_desc = desc;
    // Graph declarations can come before the backing is allocated. So from
    // construction on, range validation uses the logical texture description.
    // A padded texture's mips are those of its padded size, as its backing's are.
    m_mipLevels = ResolveTextureMipLevels(BackingDescription());
    m_arraySize = desc.isCubemap ? 6u * desc.arraySize : (desc.isArray ? desc.arraySize : 1u);
    RequireValidMultisampling(desc, m_mipLevels);
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
	if (m_desc.imageDimensions.empty() || m_desc.type != rhi::ResourceType::Texture2D || m_desc.hasDSV) return false;
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
	// Use the same mip count as the backing. ResolveTextureMipLevels prefers authored levels over generateMipMaps.
	outDesc.texture.mipLevels = static_cast<uint16_t>(m_mipLevels);
	outDesc.texture.sampleCount = (std::max)(1u, static_cast<uint32_t>(m_desc.sampleCount));
	outDesc.texture.initialLayout = rhi::ResourceLayout::Undefined;
	outDesc.queueSharing = m_desc.queueSharing; // As GpuTextureBacking creates it: representations share queues the same way.
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
	// An attached representation replaces the normal GpuTextureBacking, and may
	// be created before it. Do not take cached dimensions from that replaced
	// backing. The logical description sets the dimensions of every representation.
	const uint16_t resolvedMipLevels = ResolveTextureMipLevels(BackingDescription());
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

    const auto newDesc = BackingDescription();

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

    ApplyRelocationClass(GetGlobalResourceID());
    ++m_backingGeneration;
}

TextureDescription PixelBuffer::BackingDescription() const {
    auto newDesc = m_desc;
    if (m_desc.padInternalResolution) {
        for (auto& dim : newDesc.imageDimensions) {
            dim.width = (std::max)(1u, static_cast<unsigned int>(std::pow(2, std::ceil(std::log2(dim.width)))));
            dim.height = (std::max)(1u, static_cast<unsigned int>(std::pow(2, std::ceil(std::log2(dim.height)))));
        }
    }
    return newDesc;
}

std::unique_ptr<GpuTextureBacking> PixelBuffer::MakeRelocationBacking() {
    std::scoped_lock lock(m_materializationMutex);
    if (!m_backing) return {};
    return GpuTextureBacking::CreateUnique(BackingDescription(), GetGlobalResourceID(), name.empty() ? nullptr : name.c_str());
}

bool PixelBuffer::AdoptRelocatedBacking(std::unique_ptr<GpuTextureBacking>& next, uint64_t fromGeneration, bool rewriteInPlace) {
    std::scoped_lock lock(m_materializationMutex);
    if (!m_backing || m_backingGeneration != fromGeneration || GetAttachedAPIRepresentation(BackendInstanceId::Primary).IsValid())
        return false;
    if (!rewriteInPlace) RotateDescriptorSlotsForPublication();
    if (next) std::swap(m_backing, next);
    // Fresh slots after a rotation, or the same slots when they are rewritten in place.
    EnsureVirtualDescriptorSlotsAllocatedLocked();
    DescriptorHeapManager::ViewRequirements views;
    views.views = BuildTextureViewRequirements(m_desc, m_mipLevels, m_arraySize);
    auto resource = m_backing->GetAPIResource();
    DescriptorHeapManager::GetInstance().UpdateDescriptorContents(*this, resource, views);
    if (next) {
        // A new backing object: it gets the resource's metadata and name, as Materialize gives them.
        for (const auto& bundle : m_metadataBundles) m_backing->ApplyMetadataComponentBundle(bundle);
        if (!name.empty()) m_backing->SetName(name.c_str());
    }
    ApplyRelocationClass(GetGlobalResourceID());
    ++m_backingGeneration;
    ForgetPublishedAllocation();
    return true;
}

rhi::Result PixelBuffer::MakeDefragmentationResource(const rhi::ma::Allocation& source, rhi::ma::Allocation& destination, rhi::ResourcePtr& out) {
    std::scoped_lock lock(m_materializationMutex);
    if (!m_backing || m_backing->GetAllocation() != &source || GetAttachedAPIRepresentation(BackendInstanceId::Primary).IsValid())
        return rhi::Result::InvalidCall;
    return m_backing->CreateResourceAt(destination, name.empty() ? nullptr : name.c_str(), out);
}

rhi::ma::Allocation* PixelBuffer::RelocationAllocation() noexcept {
    std::scoped_lock lock(m_materializationMutex);
    return m_backing ? m_backing->GetAllocation() : nullptr;
}

BackingAllocationSnapshot PixelBuffer::CaptureBackingAllocation(bool retain) {
    std::scoped_lock lock(m_materializationMutex);
    if (!m_backing || GetAttachedAPIRepresentation(BackendInstanceId::Primary).IsValid()) return {};
    auto lease = retain ? m_backing->CaptureAllocationLease() : std::shared_ptr<const void>{};
    if (retain && !lease) return {};
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
    ForgetPublishedAllocation();
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
    // The slots cover the backing's mips: a padded texture has those of its padded size.
    const uint16_t mipLevels = ResolveTextureMipLevels(BackingDescription());
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
