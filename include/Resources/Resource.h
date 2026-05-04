#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <mutex>

#include <resource_states.h>
#include <rhi.h>
#include <flecs.h>

#include "Resources/ResourceStateTracker.h"

class SymbolicTracker;

class Resource : public std::enable_shared_from_this<Resource> {
public:
	struct ECSEntityHandle {
		struct DeferredState {
			mutable std::mutex mutex;
			flecs::world* world = nullptr;
			flecs::entity_t id = 0;
			bool destroyRequested = false;
			bool destroyed = false;
		};

		flecs::world* world = nullptr;
		flecs::entity_t id = 0;
		std::shared_ptr<DeferredState> deferredState;

		static ECSEntityHandle CreateDeferred() {
			ECSEntityHandle handle;
			handle.deferredState = std::make_shared<DeferredState>();
			return handle;
		}

		explicit operator bool() const noexcept {
			if (world != nullptr && id != 0) {
				return true;
			}
			if (!deferredState) {
				return false;
			}

			std::scoped_lock lock(deferredState->mutex);
			return deferredState->world != nullptr && deferredState->id != 0 && !deferredState->destroyed;
		}

		void Disarm() noexcept {
			world = nullptr;
			id = 0;
			if (deferredState) {
				std::scoped_lock lock(deferredState->mutex);
				deferredState->world = nullptr;
				deferredState->id = 0;
				deferredState->destroyed = true;
			}
		}

		bool RequestDestroy() const noexcept {
			if (!deferredState) {
				return world != nullptr && id != 0;
			}

			std::scoped_lock lock(deferredState->mutex);
			deferredState->destroyRequested = true;
			return deferredState->world != nullptr && deferredState->id != 0 && !deferredState->destroyed;
		}

		bool Resolve(flecs::world& resolvedWorld, flecs::entity_t resolvedId) const noexcept {
			if (!deferredState) {
				return false;
			}

			std::scoped_lock lock(deferredState->mutex);
			if (deferredState->destroyRequested || deferredState->destroyed) {
				deferredState->world = nullptr;
				deferredState->id = 0;
				deferredState->destroyed = true;
				return false;
			}

			deferredState->world = &resolvedWorld;
			deferredState->id = resolvedId;
			return true;
		}

		bool TryGetResolved(flecs::world*& outWorld, flecs::entity_t& outId) const noexcept {
			if (world != nullptr && id != 0) {
				outWorld = world;
				outId = id;
				return true;
			}
			if (!deferredState) {
				return false;
			}

			std::scoped_lock lock(deferredState->mutex);
			if (deferredState->world == nullptr || deferredState->id == 0 || deferredState->destroyed) {
				return false;
			}

			outWorld = deferredState->world;
			outId = deferredState->id;
			return true;
		}

		void MarkDestroyed() const noexcept {
			if (!deferredState) {
				return;
			}

			std::scoped_lock lock(deferredState->mutex);
			deferredState->world = nullptr;
			deferredState->id = 0;
			deferredState->destroyed = true;
		}

		flecs::entity ToEntity() const {
			flecs::world* resolvedWorld = nullptr;
			flecs::entity_t resolvedId = 0;
			if (!TryGetResolved(resolvedWorld, resolvedId)) {
				return {};
			}
			return flecs::entity{ *resolvedWorld, resolvedId };
		}
	};

    struct ECSEntityHooks {
        std::function<ECSEntityHandle()> createEntity;
		std::function<void(const ECSEntityHandle&)> destroyEntity;
        std::function<bool()> isRuntimeAlive;
    };

    static void SetEntityHooks(ECSEntityHooks hooks) {
        s_ecsEntityHooks = std::move(hooks);
    }

    static void ResetEntityHooks() {
        s_ecsEntityHooks = {};
    }

    Resource() {
        m_globalResourceID = globalResourceCount.fetch_add(1, std::memory_order_relaxed);
		if (s_ecsEntityHooks.createEntity) {
			m_ecsEntity = s_ecsEntityHooks.createEntity();
		}
    }
	virtual ~Resource() {
		if (!m_ecsEntity) {
			return;
		}

		if (s_ecsEntityHooks.isRuntimeAlive && !s_ecsEntityHooks.isRuntimeAlive()) {
			m_ecsEntity.Disarm();
			return;
		}

		if (s_ecsEntityHooks.destroyEntity) {
			s_ecsEntityHooks.destroyEntity(m_ecsEntity);
			m_ecsEntity.Disarm();
			return;
		}

		// Hooks have been reset (shutdown) - the world pointer may be dangling.
		// Only attempt entity cleanup if hooks are still installed.
		m_ecsEntity.Disarm();
	}


    const std::string& GetName() const { return name; }
    virtual void SetName(const std::string& newName) { this->name = newName; OnSetName(); }
	virtual rhi::Resource GetAPIResource() = 0;
    virtual uint64_t GetGlobalResourceID() const { return m_globalResourceID; }
    virtual rhi::BarrierBatch GetEnhancedBarrierGroup(RangeSpec range, rhi::ResourceAccessType prevAccessType, rhi::ResourceAccessType newAccessType, rhi::ResourceLayout prevLayout, rhi::ResourceLayout newLayout, rhi::ResourceSyncState prevSyncState, rhi::ResourceSyncState newSyncState) = 0;
	bool HasLayout() const { return m_hasLayout; }
	void AddAliasedResource(Resource* resource) {
		m_aliasedResources.push_back(resource);
	}
	bool HasAliasedResources() const {
		return !m_aliasedResources.empty();
	}
	std::vector<Resource*> GetAliasedResources() const {
		return m_aliasedResources;
	}
	unsigned int GetMipLevels() const { return m_mipLevels; }
	unsigned int GetArraySize() const { return m_arraySize; }
	std::pair<unsigned int, unsigned int> GetSubresourceMipSlice(unsigned int subresourceIndex) const {
		unsigned int mip = subresourceIndex % m_mipLevels;
		unsigned int slice = subresourceIndex / m_mipLevels;
		return std::make_pair(mip, slice);
	}

	virtual SymbolicTracker* GetStateTracker() = 0;

	// Optional capability: buffer-like resources can expose a byte size for generic readback/copy operations.
	// This avoids relying on a specific concrete C++ type (e.g. Buffer vs DynamicBuffer).
	virtual bool TryGetBufferByteSize(uint64_t& outByteSize) const { (void)outByteSize; return false; }
	flecs::entity GetECSEntity() const {
		return m_ecsEntity.ToEntity();
	}

protected:
    virtual void OnSetName() {}

    std::string name;
	bool m_hasLayout = false; // Only textures have a layout
	std::vector<Resource*> m_aliasedResources; // Resources that are aliased with this resource

    unsigned int m_mipLevels = 1;
	unsigned int m_arraySize = 1;

private:
    bool m_uploadInProgress = false;
    inline static std::atomic<uint64_t> globalResourceCount;
	inline static ECSEntityHooks s_ecsEntityHooks{};
    uint64_t m_globalResourceID;
	ECSEntityHandle m_ecsEntity; // For access through ECS queries without dereferencing Flecs during teardown

    //friend class RenderGraph;
    friend class ResourceGroup;
    friend class ResourceManager;
    friend class DynamicResource;
    friend class DynamicGloballyIndexedResource;
    friend class DynamicBuffer;
    friend class UploadManager; // Kinda a hack, for deduplicating transition lists
};