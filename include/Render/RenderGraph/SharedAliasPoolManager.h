#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <rhi.h>

#include "Render/DeviceRegistry.h"

namespace org {

class SharedAliasPoolManager {
public:
	struct Pool {
		rhi::HeapPtr d3d12;
		rhi::HeapPtr vulkan;
		uint64_t capacityBytes = 0;
		uint64_t generation = 0;
		uint8_t resourceClass = 0;
	};

	rhi::Result EnsureD3D12VulkanPool(
		uint64_t poolId,
		uint64_t generation,
		uint64_t capacityBytes,
		uint64_t alignment,
		uint8_t resourceClass,
		DeviceRegistryEntry& d3d12,
		DeviceRegistryEntry& vulkan,
		std::vector<rhi::HeapPtr>& retiredHeaps);

	Pool* Find(uint64_t poolId) noexcept;
	const Pool* Find(uint64_t poolId) const noexcept;
	void Clear() noexcept { m_pools.clear(); }
	void clear() noexcept { Clear(); } // container-style compatibility during subsystem extraction

private:
	std::unordered_map<uint64_t, Pool> m_pools;
};

} // namespace org
