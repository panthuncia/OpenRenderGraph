#include <Render/DeviceRegistry.h>
#include <Render/RenderGraph/SharedAliasPoolManager.h>

#include <cassert>
#include <stdexcept>

int main() {
	rhi::DeviceVTable vtable{};
	int primaryStorage = 0;
	int secondStorage = 0;
	rhi::Device primary{ &primaryStorage, &vtable };
	rhi::Device second{ &secondStorage, &vtable };

	org::DeviceRegistry registry;
	const auto primaryId = registry.RegisterPrimary(rhi::Backend::D3D12, primary);
	assert(primaryId == registry.PrimaryId());
	assert(static_cast<uint8_t>(primaryId) == 0);

	// Device identity, not API identity, controls registration.
	const auto secondId = registry.Register(rhi::Backend::D3D12, second);
	assert(secondId != primaryId);
	assert(static_cast<uint8_t>(secondId) == 1);
	assert(registry.Register(rhi::Backend::D3D12, second) == secondId);
	assert(registry.Find(secondId)->device.impl == &secondStorage);
	assert(registry.FindFirst(rhi::Backend::D3D12)->id == primaryId);
	assert(registry.Find(org::DeviceInstanceId::Invalid) == nullptr);

	bool rejectedInvalid = false;
	try {
		registry.Register(rhi::Backend::Null, {});
	} catch (const std::invalid_argument&) {
		rejectedInvalid = true;
	}
	assert(rejectedInvalid);

	org::SharedAliasPoolManager pools;
	assert(pools.Find(17) == nullptr);
	std::vector<rhi::HeapPtr> retired;
	org::DeviceRegistryEntry d3dEntry{ primaryId, rhi::Backend::D3D12, primary };
	org::DeviceRegistryEntry vkEntry{ secondId, rhi::Backend::Vulkan, second };
	assert(pools.EnsureD3D12VulkanPool(17, 1, 65536, 65536, 0, d3dEntry, vkEntry, retired)
		== rhi::Result::InvalidArgument);
	assert(pools.Find(17) == nullptr);
	pools.Clear();
	return 0;
}
