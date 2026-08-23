#pragma once

#include <rhi.h>

#include "Render/DeviceRegistry.h"

namespace org {

struct InteropResourcePair {
	rhi::ResourcePtr canonical;
	rhi::ResourcePtr imported;
};

/// Allocates committed cross-API resources. Policy (including D3D12 as the
/// canonical Windows allocator) lives here rather than in graph compilation.
class InteropAllocator {
public:
	static rhi::Result CreateCommittedD3D12Vulkan(
		DeviceRegistryEntry& d3d12,
		DeviceRegistryEntry& vulkan,
		const rhi::ResourceDesc& description,
		InteropResourcePair& output);
};

} // namespace org
