#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <utility>

#include <flecs.h>

#include "Render/Runtime/MpscQueue.h"
#include "Resources/MemoryStatisticsComponents.h"
#include "Resources/ResourceIdentifier.h"
#include "Resources/TrackedAllocation.h"


namespace org {

/**
 * The resource-tracking world, which holds the memory statistics.
 *
 * It has one rule. flecs is not thread-safe, so the world has a single owner at a time. Nothing outside that ownership
 * touches it.
 *
 * Resources are created and destroyed on any thread. That includes the render thread, a graph host's thread, task
 * workers, and whichever thread releases a retired backing. None of them changes the world directly. Instead each one:
 * - posts a command (Post). A wait-free queue keeps the commands in order.
 * - then offers to drain the queue (Drain). It applies the commands only if no other thread is applying them already.
 *   It never waits.
 * A reader (Access) does wait for ownership. Then it drains the queue and reads. Readers are tools such as the memory
 * view. They are never part of a frame's work.
 *
 * This is why tracking tokens are created deferred (CreateTrackedToken). The entity is made when the creation command
 * is applied. Anything attached to the token before then is applied together with it. A token destroyed before that
 * never gets an entity.
 */
class ECSManager {
public:
	static ECSManager& GetInstance();

	/** Applies a_command to the world in posting order, on whichever thread drains next. Never waits. */
	void Post(std::function<void(flecs::world&)> a_command) {
		m_commands.Push(std::move(a_command));
		// Count the command after the push. Then a drainer that sees the count can also pop the command (see Drain).
		m_pending.fetch_add(1, std::memory_order_seq_cst);
		Drain();
	}

	/**
	 * Applies the posted commands if no other thread is applying them. Otherwise returns at once, and that other thread
	 * applies them. When the draining thread releases ownership, it checks the count again. Both sides use sequentially
	 * consistent operations. So a command posted while ownership was held is never stranded. Either its poster takes
	 * ownership after the release, or the drainer sees its count.
	 */
	void Drain() {
		while (m_pending.load(std::memory_order_seq_cst) > 0) {
			if (m_owned.exchange(true, std::memory_order_seq_cst))
				return;
			const bool progressed = ApplyPosted();
			m_owned.store(false, std::memory_order_seq_cst);
			// Nothing was popped, but commands are counted. A push that is still linking its node holds the counted
			// commands behind it. Returning now could strand them, because their posters found ownership taken. So look
			// again. The window is only the two instructions of that push.
			if (!progressed)
				std::this_thread::yield();
		}
	}

	/** Runs a_read with the world owned and every command posted before it applied. Waits for ownership. */
	template <class F>
	decltype(auto) Access(F&& a_read) {
		while (m_owned.exchange(true, std::memory_order_seq_cst))
			std::this_thread::yield();
		struct Release {
			ECSManager& self;
			~Release() {
				self.m_owned.store(false, std::memory_order_seq_cst);
				// Commands posted while the reader held ownership.
				self.Drain();
			}
		} release{ *this };
		(void)ApplyPosted();
		return std::forward<F>(a_read)(m_world);
	}

	/**
	 * Creates a tracking token whose entity is made by the creation command. The entity is a_existing's entity when that
	 * is alive, or a new one otherwise. Bundles attached before then are applied with it. A token reset before then never
	 * gets an entity.
	 */
	TrackedEntityToken CreateTrackedToken(flecs::entity_t a_existing = 0) {
		auto token = TrackedEntityToken::CreateDeferred();
		Post([state = token.deferredState, a_existing](flecs::world& a_world) {
			flecs::entity entity{ a_world, a_existing };
			if (!a_existing || !entity.is_alive())
				entity = a_world.entity();
			std::vector<std::function<void(flecs::entity)>> pending;
			bool destroyRequested = false;
			if (!TrackedEntityToken::ResolveDeferredState(state, a_world, entity.id(), pending, destroyRequested)) {
				entity.destruct();
				return;
			}
			for (auto& op : pending)
				op(entity);
		});
		return token;
	}

	/**
	 * Registers the tracking components and installs ORG's TrackedEntityToken hooks. InitializeRuntimeDevice calls it.
	 * Every hook posts a command, and none touches the world directly. Tokens are deferred. A resolved token's
	 * attachments go through enqueueAttachBundle, with isMainThread false for every thread. A host with a world of its
	 * own installs its own hooks after this.
	 */
	void InstallTrackingHooks() {
		Access([](flecs::world& world) {
			world.component<MemoryStatisticsComponents::MemSizeBytes>();
			world.component<MemoryStatisticsComponents::ResourceType>();
			world.component<MemoryStatisticsComponents::ResourceID>();
			world.component<MemoryStatisticsComponents::ResourceName>();
			world.component<MemoryStatisticsComponents::AliasingPool>();
			world.component<MemoryStatisticsComponents::ResourceOwner>();
			world.component<MemoryStatisticsComponents::PlacedInAliasPool>();
			world.component<MemoryStatisticsComponents::ResourceUsage>();
			world.component<MemoryStatisticsComponents::HeapKind>();
			world.component<MemoryStatisticsComponents::TextureShape>();
			world.component<ResourceIdentifier>();
		});
		TrackedEntityToken::Hooks hooks{};
		hooks.createEntity = [](flecs::entity existing) {
			return GetInstance().CreateTrackedToken(existing.id());
		};
		hooks.isRuntimeAlive = [] { return GetInstance().IsAlive(); };
		hooks.isMainThread = [] { return false; };
		hooks.enqueueAttachBundle = [](flecs::entity_t id, EntityComponentBundle bundle) {
			GetInstance().Post([id, bundle = std::move(bundle)](flecs::world& world) {
				flecs::entity entity{ world, id };
				if (entity.is_alive()) bundle.ApplyTo(entity);
			});
		};
		hooks.destroyEntity = [](flecs::world&, flecs::entity_t id) {
			GetInstance().Post([id](flecs::world& world) {
				flecs::entity entity{ world, id };
				if (entity.is_alive()) entity.destruct();
			});
		};
		TrackedEntityToken::SetHooks(std::move(hooks));
	}

	bool IsAlive() const {
		return true;
	}

private:
	ECSManager() = default;

	// Owner only.
	bool ApplyPosted() {
		bool progressed = false;
		std::function<void(flecs::world&)> command;
		while (m_commands.Pop(command)) {
			m_pending.fetch_sub(1, std::memory_order_seq_cst);
			progressed = true;
			command(m_world);
			command = nullptr;
		}
		return progressed;
	}

	flecs::world m_world;
	runtime::MpscQueue<std::function<void(flecs::world&)>> m_commands;
	// The number of commands posted and not yet applied. It is signed because a command can be popped between its push
	// and its count. The count then dips below zero for a moment, and the poster's increment brings it back.
	std::atomic<std::int64_t> m_pending{ 0 };
	std::atomic<bool> m_owned{ false };
};

inline ECSManager& ECSManager::GetInstance() {
	static ECSManager instance;
	return instance;
}


} // namespace org
