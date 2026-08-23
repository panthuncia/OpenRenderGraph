#include "Render/RenderGraph/InteropAllocator.h"

#include <rhi_interop_dx12.h>
#include <rhi_interop_vulkan.h>

namespace org {

rhi::Result InteropAllocator::CreateCommittedD3D12Vulkan(
	DeviceRegistryEntry& d3d12,
	DeviceRegistryEntry& vulkan,
	const rhi::ResourceDesc& description,
	InteropResourcePair& output) {
	output = {};
	if (d3d12.backend != rhi::Backend::D3D12 || vulkan.backend != rhi::Backend::Vulkan)
		return rhi::Result::InvalidArgument;
	if (!d3d12.device || !vulkan.device)
		return rhi::Result::InvalidArgument;

	rhi::ResourceDesc sharedDescription = description;
	sharedDescription.heapFlags |= rhi::HeapFlags::Shared;
	if (sharedDescription.type == rhi::ResourceType::Texture2D &&
		!rhi::vulkan::query_d3d12_texture_support(vulkan.device, sharedDescription).supported)
		return rhi::Result::Unsupported;

	auto result = d3d12.device.CreateCommittedResource(sharedDescription, output.canonical);
	if (rhi::Failed(result)) return result;

	rhi::ExternalHandle handle{};
	result = rhi::dx12::export_shared_resource(d3d12.device, output.canonical.Get(), handle);
	if (rhi::Failed(result)) {
		output = {};
		return result;
	}
	result = sharedDescription.type == rhi::ResourceType::Buffer
		? rhi::vulkan::import_d3d12_buffer(vulkan.device, handle, sharedDescription, output.imported)
		: rhi::vulkan::import_d3d12_texture(vulkan.device, handle, sharedDescription, output.imported);
	if (rhi::Failed(result)) output = {};
	return result;
}

} // namespace org
