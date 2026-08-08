#include <Render/Runtime/RuntimeDevice.h>

#include "Managers/Singletons/DeviceManager.h"
#include "Managers/Singletons/ECSManager.h"
#include "Resources/TrackedAllocation.h"
#include "Resources/MemoryStatisticsComponents.h"
#include "Resources/ResourceIdentifier.h"

namespace org::runtime {

void InitializeRuntimeDevice(rhi::Device device)
{
	auto& world = ECSManager::GetInstance().GetWorld();
	world.component<MemoryStatisticsComponents::MemSizeBytes>();
	world.component<MemoryStatisticsComponents::ResourceType>();
	world.component<MemoryStatisticsComponents::ResourceID>();
	world.component<MemoryStatisticsComponents::ResourceName>();
	world.component<MemoryStatisticsComponents::AliasingPool>();
	world.component<MemoryStatisticsComponents::ResourceUsage>();
	world.component<MemoryStatisticsComponents::TextureShape>();
	world.component<ResourceIdentifier>();
	TrackedEntityToken::Hooks hooks{};
	hooks.createEntity = [](flecs::entity existing) {
		auto& world = ECSManager::GetInstance().GetWorld();
		flecs::entity entity = existing;
		if (!entity.is_alive()) entity = world.entity();
		return TrackedEntityToken(world, entity.id());
	};
	hooks.isRuntimeAlive = [] { return ECSManager::GetInstance().IsAlive(); };
	hooks.isMainThread = [] { return true; };
	hooks.enqueueAttachBundle = [](flecs::entity_t id, EntityComponentBundle bundle) {
		auto& world = ECSManager::GetInstance().GetWorld();
		flecs::entity entity{ world, id };
		if (entity.is_alive()) bundle.ApplyTo(entity);
	};
	hooks.destroyEntity = [](flecs::world& world, flecs::entity_t id) {
		flecs::entity entity{ world, id };
		if (entity.is_alive()) entity.destruct();
	};
	TrackedEntityToken::SetHooks(std::move(hooks));
	DeviceManager::GetInstance().Initialize(device);
}

void ShutdownRuntimeDevice() noexcept
{
	DeviceManager::GetInstance().Cleanup();
	TrackedEntityToken::ResetHooks();
}

} // namespace org::runtime
