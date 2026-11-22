#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <optional>
#include <functional>

#include <rhi.h>
#include <rhi_allocator.h>
#include "Resources/MemoryGroup.h"
#include "Resources/TrackedAllocation.h"


namespace org {

class DeviceManager {
public:
	struct TrackingHooks {
		std::function<TrackedEntityToken(flecs::entity existing)> createTrackingToken;
	};

    static DeviceManager& GetInstance();

	static void SetTrackingHooks(TrackingHooks hooks) {
		s_trackingHooks = std::move(hooks);
	}

	static void ResetTrackingHooks() {
		s_trackingHooks = {};
	}

	void Initialize(rhi::Device device);
	void Cleanup();
	rhi::Device GetDevice() {
        return m_device.Get();
    }

	rhi::Queue& GetGraphicsQueue() {
		return m_graphicsQueue;
	}

	rhi::Queue& GetComputeQueue() {
		return m_computeQueue;
	}

	rhi::Queue& GetCopyQueue() {
		return m_copyQueue;
	}

	rhi::ma::Allocator* GetAllocator() {
		return m_allocator;
	}

	// Create a resource and track its allocation with an entity.
	rhi::Result CreateResourceTracked(
		const rhi::ma::AllocationDesc& allocDesc,
		const rhi::ResourceDesc& resourceDesc,
		UINT32 numCastableFormats,
		const rhi::Format* pCastableFormats,
		TrackedHandle& outAllocation,
		std::optional<AllocationTrackDesc> trackDesc = std::nullopt
	) const noexcept;

	rhi::Result CreateAliasingResourceTracked(
		rhi::ma::Allocation& allocation,
		UINT64 allocationLocalOffset,
		const rhi::ResourceDesc& resourceDesc,
		UINT32 numCastableFormats,
		const rhi::Format* pCastableFormats,
		TrackedHandle& out,
		std::optional<AllocationTrackDesc> trackDesc) const noexcept;

	// Create a resource the caller owns outright, with no tracking entity (org::memory::CreateAllocatedResource). Any thread.
	rhi::Result CreateResourceUntracked(
		const rhi::ma::AllocationDesc& allocDesc,
		const rhi::ResourceDesc& resourceDesc,
		rhi::ma::AllocationPtr& outAllocation) const noexcept;

	// Creates a resource in an allocation's memory, at its start, with no tracking entity. The caller hands it to the allocation
	// (Allocation::SetResource). The defragmenter makes the resource of a move's destination this way. Any thread.
	rhi::Result CreateResourceAt(
		rhi::ma::Allocation& allocation,
		const rhi::ResourceDesc& resourceDesc,
		UINT32 numCastableFormats,
		const rhi::Format* pCastableFormats,
		rhi::ResourcePtr& out) const noexcept;

	rhi::Result AllocateMemoryTracked(
		const rhi::ma::AllocationDesc& allocDesc,
		const rhi::ResourceAllocationInfo& allocationInfo,
		TrackedHandle& outAllocation,
		std::optional<AllocationTrackDesc> trackDesc = std::nullopt) const noexcept;

	// Over the OS budget, video memory demotes lower priorities to system memory first. A placed allocation takes the
	// priority of the heap it shares.
	void SetResidencyPriority(TrackedHandle& allocation, rhi::ResidencyPriority priority) const noexcept;

	// The block size of a memory group's pools. Default allocations use the allocator's own blocks.
	static constexpr uint64_t kPerFrameBlockBytes = 32ull << 20;

	/**
	 * @brief The block pool for a memory group and heap type, made on first use. Any thread.
	 * @return Null for MemoryGroup::Default (the allocator's own blocks), for a heap type the group has no pool for, before the
	 * allocator exists, or when the pool could not be made. The caller then allocates from the allocator's own blocks.
	 */
	rhi::ma::Pool* MemoryGroupPool(MemoryGroup group, rhi::HeapType heapType) noexcept;

	/**
	 * @brief Adds up the block statistics of a memory group's pools. Any thread.
	 * @return False for MemoryGroup::Default or when the group has no pool yet; out is zero then.
	 */
	bool MemoryGroupStatistics(MemoryGroup group, rhi::ma::Statistics& out) noexcept;

	/** @brief Like MemoryGroupStatistics, for one heap type only. */
	bool MemoryGroupStatistics(MemoryGroup group, rhi::HeapType heapType, rhi::ma::Statistics& out) noexcept;

private:
	// One pool per heap type (indexed by the rhi::HeapType value) for MemoryGroup::PerFrame.
	static constexpr size_t kPoolHeapTypes = 8;
	mutable std::mutex m_poolMutex;
	std::array<rhi::ma::Pool*, kPoolHeapTypes> m_perFramePools{};
	std::array<bool, kPoolHeapTypes> m_perFramePoolFailed{};
	void ReleasePools() noexcept;

    DeviceManager() = default;
	rhi::DevicePtr m_device;
	rhi::Queue m_graphicsQueue;
	rhi::Queue m_computeQueue;
	rhi::Queue m_copyQueue;
	rhi::ma::Allocator* m_allocator = nullptr;
	mutable std::mutex m_resourceCreationMutex;
	inline static TrackingHooks s_trackingHooks{};

};

inline DeviceManager& DeviceManager::GetInstance() {
	static DeviceManager instance;
	return instance;
}


} // namespace org
