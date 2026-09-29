#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

#include <rhi_allocator.h>
#include <rhi_helpers.h>

#include "Render/Runtime/OpenRenderGraphSettings.h"
#include "Resources/TrackedAllocation.h"


namespace org {

class DeletionManager {
public:
	struct Stats {
		uint64_t objectCount = 0;
		uint64_t allocationCount = 0;
		uint64_t trackedAllocationCount = 0;
	};

	// What one retirement step (Rotate) retired: safe to destroy, and destroyed wherever the owner drops it. Members are
	// destroyed in reverse order, so the objects go first, then the allocations, then the tracked allocations.
	struct Retired {
		std::vector<TrackedHandle> trackedAllocations;
		std::vector<rhi::ma::AllocationPtr> allocations;
		std::vector<rhi::helpers::AnyObjectPtr> objects;

		bool Empty() const noexcept { return objects.empty() && allocations.empty() && trackedAllocations.empty(); }
	};

	static DeletionManager& GetInstance();

	bool IsInitialized() const {
		std::scoped_lock lock(m_mutex);
		return IsInitializedUnlocked();
	}

	void Initialize() {
		std::scoped_lock lock(m_mutex);
		m_numFramesInFlight = org::runtime::GetOpenRenderGraphSettings().numFramesInFlight;
		const size_t retirementSlotCount = static_cast<size_t>(m_numFramesInFlight) + 1u;
		m_deletionQueue.resize(retirementSlotCount);
		m_allocationDeletionQueue.resize(retirementSlotCount);
		m_trackedAllocationDeletionQueue.resize(retirementSlotCount);
	}

	void MarkForDelete(rhi::helpers::AnyObjectPtr ptr) {
		std::scoped_lock lock(m_mutex);
		if (!IsInitializedUnlocked()) {
			return;
		}
		m_deletionQueue[0].push_back(std::move(ptr));
	}

	void MarkForDelete(rhi::ma::AllocationPtr ptr) {
		std::scoped_lock lock(m_mutex);
		if (!IsInitializedUnlocked()) {
			return;
		}
		m_allocationDeletionQueue[0].push_back(std::move(ptr));
	}

	void MarkForDelete(TrackedHandle&& alloc) {
		std::scoped_lock lock(m_mutex);
		if (!IsInitializedUnlocked()) {
			alloc.Reset();
			return;
		}
		m_trackedAllocationDeletionQueue[0].push_back(std::move(alloc));
	}

	// One retirement step: the oldest queue retires and every other ages by one. The retired objects are returned, not
	// destroyed: the destruction (driver frees, taking the device's memory locks) happens after this lock is released, on
	// whichever thread the caller chooses, so MarkForDelete never waits for it.
	Retired Rotate() {
		Retired retired;
		std::scoped_lock lock(m_mutex);
		if (!IsInitializedUnlocked()) {
			return retired;
		}
		retired.objects.swap(m_deletionQueue.back());
		for (int i = static_cast<int>(m_deletionQueue.size()) - 1; i >= 1; --i) {
			m_deletionQueue[i].swap(m_deletionQueue[i - 1]);
		}

		retired.allocations.swap(m_allocationDeletionQueue.back());
		for (int i = static_cast<int>(m_allocationDeletionQueue.size()) - 1; i >= 1; --i) {
			m_allocationDeletionQueue[i].swap(m_allocationDeletionQueue[i - 1]);
		}

		retired.trackedAllocations.swap(m_trackedAllocationDeletionQueue.back());
		for (int i = static_cast<int>(m_trackedAllocationDeletionQueue.size()) - 1; i >= 1; --i) {
			m_trackedAllocationDeletionQueue[i].swap(m_trackedAllocationDeletionQueue[i - 1]);
		}
		return retired;
	}

	// One retirement step, destroying what retires on the calling thread.
	void ProcessDeletions() {
		(void)Rotate();
	}

	Stats GetStats() const {
		std::scoped_lock lock(m_mutex);
		Stats stats{};
		for (const auto& queue : m_deletionQueue) {
			stats.objectCount += queue.size();
		}
		for (const auto& queue : m_allocationDeletionQueue) {
			stats.allocationCount += queue.size();
		}
		for (const auto& queue : m_trackedAllocationDeletionQueue) {
			stats.trackedAllocationCount += queue.size();
		}
		return stats;
	}

	void DrainAll() {
		std::scoped_lock lock(m_mutex);
		if (!IsInitializedUnlocked()) {
			return;
		}

		for (auto& queue : m_deletionQueue) {
			queue.clear();
		}
		for (auto& queue : m_allocationDeletionQueue) {
			queue.clear();
		}
		for (auto& queue : m_trackedAllocationDeletionQueue) {
			queue.clear();
		}
	}

	void Cleanup() {
		std::scoped_lock lock(m_mutex);
		m_deletionQueue.clear();
		m_allocationDeletionQueue.clear();
		m_trackedAllocationDeletionQueue.clear();
		m_numFramesInFlight = 0;
	}

private:
	uint8_t m_numFramesInFlight = 0;
	DeletionManager() = default;

	bool IsInitializedUnlocked() const noexcept {
		return m_numFramesInFlight != 0 &&
			!m_deletionQueue.empty() &&
			!m_allocationDeletionQueue.empty() &&
			!m_trackedAllocationDeletionQueue.empty();
	}

	mutable std::mutex m_mutex;
	std::vector<std::vector<rhi::helpers::AnyObjectPtr>> m_deletionQueue;
	std::vector<std::vector<rhi::ma::AllocationPtr>> m_allocationDeletionQueue;
	std::vector<std::vector<TrackedHandle>> m_trackedAllocationDeletionQueue;
};

inline DeletionManager& DeletionManager::GetInstance() {
	static DeletionManager instance;
	return instance;
}


} // namespace org
