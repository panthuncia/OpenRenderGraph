#include "Resources/GPUBacking/GPUTextureBacking.h"
#include <string>
#include <stdexcept>

#include "Managers/Singletons/DeviceManager.h"
#include "Utilities/ORGUtilities.h"
#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Managers/Singletons/UploadManager.h"
#include "Managers/Singletons/DeletionManager.h"
#include "Resources/MemoryStatisticsComponents.h"
#include <rhi_interop.h>


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
		return org::util::CalculateMipLevels(desc.imageDimensions[0].width, desc.imageDimensions[0].height);
	}

	return 1;
}
}

GpuTextureBacking::GpuTextureBacking(CreateTag)
{

}

GpuTextureBacking::~GpuTextureBacking()
{
	UnregisterLiveAlloc();
	DeletionManager::GetInstance().MarkForDelete(std::move(m_textureHandle));
}

std::unique_ptr<GpuTextureBacking>
GpuTextureBacking::CreateUnique(const TextureDescription& desc,
	uint64_t owningResourceID,
	const char* name)
{
	auto pb = std::make_unique<GpuTextureBacking>(CreateTag{});
	pb->initialize(desc, owningResourceID, nullptr, name);
#if BUILD_TYPE == BUILD_DEBUG
	pb->m_creation = std::stacktrace::current();
#endif
	return std::move(pb);
}

std::unique_ptr<GpuTextureBacking>
GpuTextureBacking::CreateUnique(const TextureDescription& desc,
	uint64_t owningResourceID,
	const TextureAliasPlacement& placement,
	const char* name)
{
	auto pb = std::make_unique<GpuTextureBacking>(CreateTag{});
	pb->initialize(desc, owningResourceID, &placement, name);
#if BUILD_TYPE == BUILD_DEBUG
	pb->m_creation = std::stacktrace::current();
#endif
	return std::move(pb);
}

void GpuTextureBacking::initialize(const TextureDescription& desc,
	uint64_t owningResourceID,
	const char* name)
{
	initialize(desc, owningResourceID, nullptr, name);
}

void GpuTextureBacking::initialize(const TextureDescription& desc,
	uint64_t owningResourceID,
	const TextureAliasPlacement* placement,
	const char* name)
{
	m_desc = desc;
    if (placement && !placement->heap) throw std::invalid_argument("Missing alias heap owner");
	DescriptorHeapManager& rm = DescriptorHeapManager::GetInstance();

	// Honor explicit subresource chains for file-backed textures and caches.
	uint16_t mipLevels = ResolveTextureMipLevels(desc);

	// Determine the array size
	uint32_t arraySize = desc.arraySize;
	if (!desc.isArray && !desc.isCubemap) {
		arraySize = 1;
	}

	// Create the texture resource description
	auto width = desc.imageDimensions[0].width;
	auto height = desc.imageDimensions[0].height;
	if (desc.type == rhi::ResourceType::Texture1D) height = 1;
	if (desc.padInternalResolution) { // Pad the width and height to the next power of two
		width = std::max(1u, static_cast<unsigned int>(std::pow(2, std::ceil(std::log2(width)))));
		height = std::max(1u, static_cast<unsigned int>(std::pow(2, std::ceil(std::log2(height)))));
	}

	if (width > std::numeric_limits<uint32_t>().max() || height > std::numeric_limits<uint32_t>().max()) {
		spdlog::error("Texture dimensions above uint32_max not implemented");
	}

	// Handle clear values for RTV and DSV
	rhi::ClearValue* clearValue = nullptr;
	rhi::ClearValue depthClearValue = {};
	rhi::ClearValue colorClearValue = {};
	if (desc.hasDSV) {
		depthClearValue.type = rhi::ClearValueType::DepthStencil;
		depthClearValue.format = desc.dsvFormat == rhi::Format::Unknown ? desc.format : desc.dsvFormat;
		depthClearValue.depthStencil.depth = desc.depthClearValue;
		depthClearValue.depthStencil.stencil = 0;
		clearValue = &depthClearValue;
	}
	else if (desc.hasRTV) {
		colorClearValue.type = rhi::ClearValueType::Color;
		colorClearValue.format = desc.rtvFormat == rhi::Format::Unknown ? desc.format : desc.rtvFormat;
		colorClearValue.rgba[0] = desc.clearColor[0];
		colorClearValue.rgba[1] = desc.clearColor[1];
		colorClearValue.rgba[2] = desc.clearColor[2];
		colorClearValue.rgba[3] = desc.clearColor[3];
		clearValue = &colorClearValue;
	}

	rhi::ResourceDesc textureDesc{
		.type = desc.type,
		.texture = {
			.format = desc.format,
			.width = static_cast<uint32_t>(width),
			.height = static_cast<uint32_t>(height),
			.depthOrLayers = static_cast<uint16_t>(desc.type == rhi::ResourceType::Texture3D ?
				desc.depth : (desc.isCubemap ? 6 * arraySize : arraySize)),
			.mipLevels = mipLevels,
			.sampleCount = desc.sampleCount,
			.initialLayout = desc.initialLayout,
			.optimizedClear = clearValue
		}
	};
	if (desc.isCubemap) {
		textureDesc.resourceFlags |= rhi::ResourceFlags::RF_TextureCubeCompatible;
	}
	if (desc.hasRTV) {
		textureDesc.resourceFlags |= rhi::ResourceFlags::RF_AllowRenderTarget;
	}
	if (desc.hasDSV) {
		textureDesc.resourceFlags |= rhi::ResourceFlags::RF_AllowDepthStencil;
	}
	if (desc.hasUAV) {
		textureDesc.resourceFlags |= rhi::ResourceFlags::RF_AllowUnorderedAccess;
	}
	// Create the texture resource

	AllocationTrackDesc trackDesc(static_cast<int>(owningResourceID));
	EntityComponentBundle allocationBundle;
	if (name != nullptr) {
		allocationBundle.Set<MemoryStatisticsComponents::ResourceName>({ name });
	}

	auto device = DeviceManager::GetInstance().GetDevice();
	// Vulkan images are born in UNDEFINED (the only other legal initial layout is
	// PREINITIALIZED for linear host-written images).  ResourceLayout::Common is
	// a useful D3D12 creation state, but treating it as GENERAL in Vulkan's
	// symbolic tracker skips the first real layout transition and makes uploads
	// submit an image that validation still knows is UNDEFINED.
	rhi::VulkanDeviceInfo vulkanDeviceInfo{};
	const bool vulkanInitialUndefined = rhi::QueryNativeDevice(
		device,
		rhi::RHI_IID_VK_DEVICE,
		&vulkanDeviceInfo,
		sizeof(vulkanDeviceInfo));
	if (vulkanInitialUndefined) {
		textureDesc.texture.initialLayout = rhi::ResourceLayout::Undefined;
	}

	rhi::ResourceAllocationInfo allocInfo;
	device.GetResourceAllocationInfo(&textureDesc, 1, &allocInfo);
	if (placement) {
		m_aliasHeap = placement->heap;
		m_aliasPoolID = placement->poolID.value_or(0);
		m_aliasOffset = placement->offset;
		m_aliasSize = allocInfo.sizeInBytes;
	}

	allocationBundle
		.Set<MemoryStatisticsComponents::MemSizeBytes>({ allocInfo.sizeInBytes })
		.Set<MemoryStatisticsComponents::ResourceType>({ desc.type })
		.Set<MemoryStatisticsComponents::ResourceID>({ owningResourceID })
		.Set<MemoryStatisticsComponents::TextureShape>({
			desc.imageDimensions[0].width,
			desc.imageDimensions[0].height,
			ResolveTextureMipLevels(desc),
			desc.isCubemap ? 6u * desc.arraySize : (desc.isArray ? desc.arraySize : 1u),
			desc.format,
			placement != nullptr });
	if (desc.aliasingPoolID.has_value()) {
		allocationBundle.Set<MemoryStatisticsComponents::AliasingPool>({ desc.aliasingPoolID });
	}
	trackDesc.attach = allocationBundle;

	if (placement) {
		if (placement->poolID.has_value()) {
			allocationBundle.Set<MemoryStatisticsComponents::AliasingPool>({ placement->poolID });
		}
		trackDesc.attach = allocationBundle;

		const auto result = DeviceManager::GetInstance().CreateAliasingResourceTracked(
			placement->heap->Allocation(),
			placement->offset,
			textureDesc,
			0,
			nullptr,
			m_textureHandle,
			trackDesc);
		if (!rhi::IsOk(result)) {
			throw std::runtime_error("Failed to create aliased texture resource backing");
		}
		m_textureHandle.RetainLifetimeOwner(placement->heap);
	}
	else {

		rhi::ma::AllocationDesc allocationDesc;
		allocationDesc.heapType = rhi::HeapType::DeviceLocal;

		const auto result = DeviceManager::GetInstance().CreateResourceTracked(
			allocationDesc,
			textureDesc,
			0,
			nullptr,
			m_textureHandle,
			trackDesc);
		if (!rhi::IsOk(result)) {
			throw std::runtime_error("Failed to create committed texture resource backing");
		}

		//auto result = device.CreateCommittedResource(textureDesc, textureResource);
	}

	//m_placedResourceHeap = aliasTarget ? aliasTarget->GetPlacedResourceHeap() : rhi::HeapHandle();

	m_width = desc.imageDimensions[0].width;
	m_height = desc.imageDimensions[0].height;
	m_mipLevels = ResolveTextureMipLevels(desc);
	m_arraySize = desc.type == rhi::ResourceType::Texture3D ? 1u :
		(desc.isCubemap ? 6 * desc.arraySize : (desc.isArray ? desc.arraySize : 1));
	m_format = desc.format;
	RangeSpec wholeRange;
	wholeRange.mipLower = { BoundType::All, 0 };
	wholeRange.mipUpper = { BoundType::All, 0 };
	wholeRange.sliceLower = { BoundType::All, 0 };
	wholeRange.sliceUpper = { BoundType::All, 0 };
	const bool startsInCommon =
		desc.initialLayout == rhi::ResourceLayout::Common && !vulkanInitialUndefined;
	const rhi::ResourceLayout trackedInitialLayout = vulkanInitialUndefined
		? rhi::ResourceLayout::Undefined
		: desc.initialLayout;
	m_stateTracker = SymbolicTracker(
		wholeRange,
		ResourceState{
			startsInCommon ? rhi::ResourceAccessType::Common : rhi::ResourceAccessType::None,
			trackedInitialLayout,
			startsInCommon ? rhi::ResourceSyncState::All : rhi::ResourceSyncState::None });

	size_t subCount = m_mipLevels * m_arraySize;

	if (name && HasValidResource()) {
		m_textureHandle.GetResource().SetName(name);
	}

	RegisterLiveAlloc();
	UpdateLiveAllocName(name);

}

void GpuTextureBacking::SetName(const char* newName)
{
	if (!newName || !HasValidResource()) {
		return;
	}
	m_textureHandle.ApplyComponentBundle(EntityComponentBundle().Set<MemoryStatisticsComponents::ResourceName>({ newName }));
	m_textureHandle.GetResource().SetName(newName);
	UpdateLiveAllocName(newName);
}

bool GpuTextureBacking::HasValidResource() const
{
	if (!m_textureHandle) {
		return false;
	}

	auto& resource = const_cast<TrackedHandle&>(m_textureHandle).GetResource();
	if (!resource.IsValid()) {
		return false;
	}

	rhi::D3D12ResourceInfo resourceInfo{};
	if (rhi::QueryNativeResource(resource, rhi::RHI_IID_D3D12_RESOURCE, &resourceInfo, sizeof(resourceInfo)) &&
		resourceInfo.resource != nullptr) {
		return true;
	}

	rhi::VulkanResourceInfo vkResourceInfo{};
	return rhi::QueryNativeResource(resource, rhi::RHI_IID_VK_RESOURCE, &vkResourceInfo, sizeof(vkResourceInfo)) &&
		vkResourceInfo.resource != nullptr;
}

std::mutex& GpuTextureBacking::LiveAllocMutex() {
	static auto* mutex = new std::mutex();
	return *mutex;
}

std::unordered_map<const GpuTextureBacking*, GpuTextureBacking::LiveAllocInfo>& GpuTextureBacking::LiveAllocs() {
	static auto* liveAllocs = new std::unordered_map<const GpuTextureBacking*, LiveAllocInfo>();
	return *liveAllocs;
}

rhi::BarrierBatch GpuTextureBacking::GetEnhancedBarrierGroup(RangeSpec range, rhi::ResourceAccessType prevAccessType, rhi::ResourceAccessType newAccessType, rhi::ResourceLayout prevLayout, rhi::ResourceLayout newLayout, rhi::ResourceSyncState prevSyncState, rhi::ResourceSyncState newSyncState) {

	rhi::BarrierBatch batch = {};

	auto resolvedRange = ResolveRangeSpec(range, m_mipLevels, m_arraySize);

	thread_local rhi::TextureBarrier barrier{};
	barrier.afterAccess = newAccessType;
	barrier.beforeAccess = prevAccessType;
	barrier.afterLayout = newLayout;
	barrier.beforeLayout = prevLayout;
	barrier.afterSync = newSyncState;
	barrier.beforeSync = prevSyncState;
	barrier.discard = false;
	barrier.range = { resolvedRange.firstMip, resolvedRange.mipCount, resolvedRange.firstSlice, resolvedRange.sliceCount };
	barrier.texture = m_textureHandle.GetResource().GetHandle();

	batch.textures = { &barrier };

	return batch;
}

void GpuTextureBacking::RegisterLiveAlloc() {
	auto& liveAllocs = LiveAllocs();
	std::scoped_lock lock(LiveAllocMutex());
	LiveAllocInfo info{};
	liveAllocs[this] = info;
}

void GpuTextureBacking::UnregisterLiveAlloc() {
	auto& liveAllocs = LiveAllocs();
	std::scoped_lock lock(LiveAllocMutex());
	liveAllocs.erase(this);
}

void GpuTextureBacking::UpdateLiveAllocName(const char* name) {
	auto& liveAllocs = LiveAllocs();
	std::scoped_lock lock(LiveAllocMutex());
	auto it = liveAllocs.find(this);
	if (it != liveAllocs.end()) {
		it->second.name = name ? name : "";
	}
}

unsigned int GpuTextureBacking::DumpLiveTextures() {
	auto& liveAllocs = LiveAllocs();
	std::scoped_lock lock(LiveAllocMutex());
	for (const auto& [ptr, info] : liveAllocs) {
		spdlog::warn("Live texture still tracked: name='{}'", info.name);
	}
	return static_cast<unsigned int>(liveAllocs.size());
}


} // namespace org
