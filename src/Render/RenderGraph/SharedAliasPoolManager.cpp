#include "Render/RenderGraph/SharedAliasPoolManager.h"

#include <algorithm>

#include <rhi_interop_dx12.h>
#include <rhi_interop_vulkan.h>

namespace org {

rhi::Result SharedAliasPoolManager::EnsureD3D12VulkanPool(
	uint64_t poolId,
	uint64_t generation,
	uint64_t capacityBytes,
	uint64_t alignment,
	uint8_t resourceClass,
	DeviceRegistryEntry& d3d12,
	DeviceRegistryEntry& vulkan,
	std::vector<rhi::HeapPtr>& retiredHeaps) {
	if (resourceClass != 1 && resourceClass != 2) return rhi::Result::InvalidArgument;
	auto& pool = m_pools[poolId];
	if (pool.d3d12 && pool.vulkan && pool.generation == generation &&
		pool.capacityBytes >= capacityBytes && pool.resourceClass == resourceClass)
		return rhi::Result::Ok;

	rhi::HeapDesc description{};
	description.sizeBytes = capacityBytes;
	description.alignment = alignment;
	description.memory = rhi::HeapType::DeviceLocal;
	description.flags = rhi::HeapFlags::Shared | (resourceClass == 1
		? rhi::HeapFlags::AllowOnlyBuffers : rhi::HeapFlags::AllowOnlyRtDsTextures);
	description.debugName = "RenderGraph multi-RHI alias pool";

	rhi::HeapPtr newD3D12;
	rhi::HeapPtr newVulkan;
	auto result = d3d12.device.CreateHeap(description, newD3D12);
	rhi::ExternalHandle handle{};
	if (rhi::IsOk(result)) result = rhi::dx12::export_shared_heap(d3d12.device, newD3D12.Get(), handle);
	if (rhi::IsOk(result)) result = rhi::vulkan::import_d3d12_heap(vulkan.device, handle, description, newVulkan);
	if (rhi::Failed(result)) return result;

	if (pool.d3d12) retiredHeaps.push_back(std::move(pool.d3d12));
	if (pool.vulkan) retiredHeaps.push_back(std::move(pool.vulkan));
	pool.d3d12 = std::move(newD3D12);
	pool.vulkan = std::move(newVulkan);
	pool.capacityBytes = capacityBytes;
	pool.generation = generation;
	pool.resourceClass = resourceClass;
	return rhi::Result::Ok;
}

SharedAliasPoolManager::Pool* SharedAliasPoolManager::Find(uint64_t poolId) noexcept {
	const auto it = m_pools.find(poolId);
	return it == m_pools.end() ? nullptr : &it->second;
}

const SharedAliasPoolManager::Pool* SharedAliasPoolManager::Find(uint64_t poolId) const noexcept {
	const auto it = m_pools.find(poolId);
	return it == m_pools.end() ? nullptr : &it->second;
}

} // namespace org
