#include <Render/DeviceRegistry.h>

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

	return 0;
}
