// The resource-tracking world (ECSManager) has one owner at a time. Resources are created, named and destroyed on many
// threads at once. They post their changes, and a reader sees a consistent world.
// Before this design, every thread changed the flecs world directly. A creation on the render thread next to a
// destruction on a host's thread then wrote through a null component.
#include "Managers/Singletons/ECSManager.h"
#include "TestSupport.h"

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>


int main() {
	using namespace org;
	using Size = MemoryStatisticsComponents::MemSizeBytes;
	auto& ecs = ECSManager::GetInstance();
	ecs.InstallTrackingHooks();

	constexpr int kThreads = 8;
	constexpr int kPerThread = 4000;
	// Each thread keeps every fourth token alive to the end. It resets the rest:
	// - half at once, often before the entity exists,
	// - half after attaching a second bundle.
	std::vector<std::vector<TrackedEntityToken>> kept(kThreads);
	std::atomic<bool> producing{ true };
	std::atomic<int> snapshots{ 0 };
	std::thread reader([&] {
		while (producing.load()) {
			ecs.Access([&](flecs::world& world) {
				world.each([](flecs::entity, const Size&) {});
			});
			snapshots.fetch_add(1);
		}
	});
	std::vector<std::thread> producers;
	for (int t = 0; t < kThreads; ++t) {
		producers.emplace_back([&, t] {
			for (int i = 0; i < kPerThread; ++i) {
				auto token = TrackedEntityToken::CreateFromHooks();
				token.ApplyAttachBundle(EntityComponentBundle().Set<Size>({ static_cast<uint64_t>(t * kPerThread + i) }));
				if (i % 4 == 0) {
					kept[t].push_back(std::move(token));
				} else if (i % 2 == 0) {
					token.ApplyAttachBundle(EntityComponentBundle().Set<Size>({ 1 }));
					token.Reset();
				} else {
					token.Reset();
				}
			}
		});
	}
	for (auto& producer : producers) producer.join();
	producing.store(false);
	reader.join();
	CHECK(snapshots.load() > 0);

	// Every kept token has its entity, with the value it attached; nothing else is left.
	int live = 0;
	bool valuesKept = true;
	ecs.Access([&](flecs::world& world) {
		world.each([&](flecs::entity, const Size& size) {
			++live;
			if (size.size == 1) valuesKept = false;
		});
	});
	CHECK(live == kThreads * (kPerThread / 4));
	CHECK(valuesKept);
	for (auto& tokens : kept) {
		for (auto& token : tokens) {
			flecs::world* world = nullptr;
			flecs::entity_t id = 0;
			CHECK(token.TryGetResolved(world, id));
		}
	}

	// Reset from other threads than created them.
	std::vector<std::thread> resetters;
	for (int t = 0; t < kThreads; ++t)
		resetters.emplace_back([&, t] { kept[(t + 1) % kThreads].clear(); });
	for (auto& resetter : resetters) resetter.join();
	live = 0;
	ecs.Access([&](flecs::world& world) { world.each([&](flecs::entity, const Size&) { ++live; }); });
	CHECK(live == 0);

	TrackedEntityToken::ResetHooks();
	std::puts("TrackingWorldTests passed");
	return 0;
}
