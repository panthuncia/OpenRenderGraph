#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <mutex>
#include <unordered_map>

#include <resource_states.h>
#include <rhi.h>
#include <flecs.h>

#include "Resources/ResourceStateTracker.h"
#include "Render/QueueKind.h"


namespace org {

class SymbolicTracker;

class Resource : public std::enable_shared_from_this<Resource> {
public:
	enum class GraphOwnership : uint8_t {
		GraphManaged = 0,
		ExternalImmutableShaderResource,
	};
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
	struct APIRepresentation {
		rhi::ResourcePtr resource;
		SymbolicTracker tracker;
		uint64_t backingGeneration = 0;
	};
	using APIRepresentationPtr = std::shared_ptr<APIRepresentation>;
	struct PendingAPIRepresentation {
		BackendInstanceId device;
		rhi::ResourcePtr resource;
		ResourceState initialState{ rhi::ResourceAccessType::None, rhi::ResourceLayout::Undefined, rhi::ResourceSyncState::None };
	};
	APIRepresentationPtr AcquireAPIRepresentation(BackendInstanceId backendInstance) const {
		std::scoped_lock lock(m_representationMutex);
		const auto it = m_representations.find(static_cast<uint8_t>(backendInstance));
		return it != m_representations.end() ? it->second : APIRepresentationPtr{};
	}
	rhi::Resource GetAttachedAPIRepresentation(BackendInstanceId backendInstance) const {
		auto representation = AcquireAPIRepresentation(backendInstance);
		return representation && representation->resource ? representation->resource.Get() : rhi::Resource{};
	}
	virtual rhi::Resource GetAPIResource(BackendInstanceId backendInstance) {
		if (auto attached = GetAttachedAPIRepresentation(backendInstance)) return attached;
		return backendInstance == BackendInstanceId::Primary ? GetAPIResource() : rhi::Resource{};
	}
	bool AttachAPIRepresentation(BackendInstanceId backendInstance, rhi::ResourcePtr resource,
		ResourceState initialState = { rhi::ResourceAccessType::None, rhi::ResourceLayout::Undefined, rhi::ResourceSyncState::None }) {
		std::vector<PendingAPIRepresentation> pending;
		pending.push_back({ backendInstance, std::move(resource), initialState });
		return PublishAPIRepresentations(std::move(pending));
	}
	// Publishes a complete set of device-local backings as one generation. No
	// execution thread can observe only half of a cross-device resource pair.
	bool PublishAPIRepresentations(std::vector<PendingAPIRepresentation> pending) {
		if (pending.empty()) return true;
		std::unordered_map<uint8_t, APIRepresentationPtr> replacements;
		replacements.reserve(pending.size());
		const uint64_t generation = m_nextRepresentationGeneration.fetch_add(1, std::memory_order_relaxed);
		for (auto& item : pending) {
			if (!item.resource) return false;
			auto representation = std::make_shared<APIRepresentation>();
			representation->resource = std::move(item.resource);
			representation->tracker = SymbolicTracker(RangeSpec{}, item.initialState);
			representation->backingGeneration = generation;
			replacements[static_cast<uint8_t>(item.device)] = std::move(representation);
		}
		std::scoped_lock lock(m_representationMutex);
		for (auto& [device, representation] : replacements)
			m_representations[device] = std::move(representation);
		m_publishedRepresentationGeneration = generation;
		return true;
	}
	uint64_t GetAPIRepresentationGeneration() const noexcept { return m_publishedRepresentationGeneration.load(std::memory_order_acquire); }
	bool HasAPIRepresentation(BackendInstanceId backendInstance) {
		if (GetAttachedAPIRepresentation(backendInstance)) return true;
		// The implicit primary backing is intentionally not counted as an explicit
		// representation. This query is used to decide whether interop
		// materialization still has work to do and must be safe for unmaterialized
		// graph resources.
		return false;
	}
	std::vector<BackendInstanceId> GetRepresentationInstances() const {
		std::vector<BackendInstanceId> result;
		std::scoped_lock lock(m_representationMutex);
		for (const auto& [id, representation] : m_representations) {
			if (representation && representation->resource) result.push_back(static_cast<BackendInstanceId>(id));
		}
		if (result.empty()) result.push_back(BackendInstanceId::Primary); // legacy implicit primary backing
		return result;
	}
	void ClearAPIRepresentations() {
		std::scoped_lock lock(m_representationMutex);
		m_representations.clear();
	}
	std::vector<APIRepresentationPtr> TakeAPIRepresentations() {
		std::vector<APIRepresentationPtr> retired;
		std::scoped_lock lock(m_representationMutex);
		retired.reserve(m_representations.size());
		for (auto& [device, representation] : m_representations) {
			(void)device;
			if (representation) retired.push_back(std::move(representation));
		}
		m_representations.clear();
		return retired;
	}
    virtual uint64_t GetGlobalResourceID() const { return m_globalResourceID; }
	// Identity used by render-graph scheduling. Dynamic wrappers override this so
	// dependency identity remains stable when their backing resource changes.
	virtual uint64_t GetSchedulingResourceID() const { return GetGlobalResourceID(); }
	GraphOwnership GetGraphOwnership() const noexcept { return m_graphOwnership; }
	void SetGraphOwnership(GraphOwnership ownership) noexcept { m_graphOwnership = ownership; }
	bool IsRenderGraphManaged() const noexcept { return m_graphOwnership == GraphOwnership::GraphManaged; }
	virtual rhi::BarrierBatch GetEnhancedBarrierGroup(RangeSpec range, rhi::ResourceAccessType prevAccessType, rhi::ResourceAccessType newAccessType, rhi::ResourceLayout prevLayout, rhi::ResourceLayout newLayout, rhi::ResourceSyncState prevSyncState, rhi::ResourceSyncState newSyncState) = 0;
	virtual rhi::BarrierBatch GetEnhancedBarrierGroup(BackendInstanceId backendInstance, RangeSpec range,
		rhi::ResourceAccessType prevAccessType, rhi::ResourceAccessType newAccessType,
		rhi::ResourceLayout prevLayout, rhi::ResourceLayout newLayout,
		rhi::ResourceSyncState prevSyncState, rhi::ResourceSyncState newSyncState) {
		std::unique_lock lock(m_representationMutex);
		auto it = m_representations.find(static_cast<uint8_t>(backendInstance));
		if (it == m_representations.end()) {
			lock.unlock();
			return backendInstance == BackendInstanceId::Primary ? GetEnhancedBarrierGroup(
				range, prevAccessType, newAccessType, prevLayout, newLayout, prevSyncState, newSyncState) : rhi::BarrierBatch{};
		}
		auto& representation = *it->second;
		if (HasLayout()) {
			thread_local rhi::TextureBarrier textureBarrier{};
			const auto resolved = ResolveRangeSpec(range, m_mipLevels, m_arraySize);
			textureBarrier = {
				.texture = representation.resource->GetHandle(),
				.range = { resolved.firstMip, resolved.mipCount, resolved.firstSlice, resolved.sliceCount },
				.beforeSync = prevSyncState, .afterSync = newSyncState,
				.beforeAccess = prevAccessType, .afterAccess = newAccessType,
				.beforeLayout = prevLayout, .afterLayout = newLayout };
			return { .textures = { &textureBarrier, 1 } };
		}
		thread_local rhi::BufferBarrier bufferBarrier{};
		bufferBarrier = { .buffer = representation.resource->GetHandle(), .offset = 0, .size = ~0ull,
			.beforeSync = prevSyncState, .afterSync = newSyncState,
			.beforeAccess = prevAccessType, .afterAccess = newAccessType };
		return { .buffers = { &bufferBarrier, 1 } };
	}
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
	SymbolicTracker* GetAttachedStateTracker(BackendInstanceId backendInstance) {
		std::scoped_lock lock(m_representationMutex);
		const auto it = m_representations.find(static_cast<uint8_t>(backendInstance));
		return it != m_representations.end() && it->second ? &it->second->tracker : nullptr;
	}
	virtual SymbolicTracker* GetStateTracker(BackendInstanceId backendInstance) {
		if (auto* attached = GetAttachedStateTracker(backendInstance)) return attached;
		return backendInstance == BackendInstanceId::Primary ? GetStateTracker() : nullptr;
	}
	virtual bool TryGetRHIResourceDesc(rhi::ResourceDesc& outDesc) const { (void)outDesc; return false; }
	virtual void RefreshAPIRepresentationDescriptors(BackendInstanceId) {}

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
	mutable std::mutex m_representationMutex;
	std::unordered_map<uint8_t, APIRepresentationPtr> m_representations;
	std::atomic<uint64_t> m_nextRepresentationGeneration{ 1 };
	std::atomic<uint64_t> m_publishedRepresentationGeneration{ 0 };
    bool m_uploadInProgress = false;
    inline static std::atomic<uint64_t> globalResourceCount;
	inline static ECSEntityHooks s_ecsEntityHooks{};
    uint64_t m_globalResourceID;
	GraphOwnership m_graphOwnership = GraphOwnership::GraphManaged;
	ECSEntityHandle m_ecsEntity; // For access through ECS queries without dereferencing Flecs during teardown

    //friend class RenderGraph;
    friend class ResourceGroup;
    friend class ResourceManager;
    friend class DynamicResource;
    friend class DynamicGloballyIndexedResource;
    friend class DynamicBuffer;
    friend class UploadManager; // Kinda a hack, for deduplicating transition lists
};


} // namespace org

using org::Resource;
using namespace org;
