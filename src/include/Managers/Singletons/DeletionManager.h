#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

#include <rhi_allocator.h>
#include <rhi_helpers.h>

#include "Render/Runtime/OpenRenderGraphSettings.h"
#include "Resources/TrackedAllocation.h"

class DeletionManager {
public:
	static DeletionManager& GetInstance();

	bool IsInitialized() const {
		std::scoped_lock lock(m_mutex);
		return IsInitializedUnlocked();
	}

	void Initialize() {
		std::scoped_lock lock(m_mutex);
		m_numFramesInFlight = rg::runtime::GetOpenRenderGraphSettings().numFramesInFlight;
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

	void ProcessDeletions() {
		std::scoped_lock lock(m_mutex);
		if (!IsInitializedUnlocked()) {
			return;
		}
		m_deletionQueue.back().clear();
		for (int i = static_cast<int>(m_deletionQueue.size()) - 1; i >= 1; --i) {
			m_deletionQueue[i].swap(m_deletionQueue[i - 1]);
		}

		m_allocationDeletionQueue.back().clear();
		for (int i = static_cast<int>(m_allocationDeletionQueue.size()) - 1; i >= 1; --i) {
			m_allocationDeletionQueue[i].swap(m_allocationDeletionQueue[i - 1]);
		}

		m_trackedAllocationDeletionQueue.back().clear();
		for (int i = static_cast<int>(m_trackedAllocationDeletionQueue.size()) - 1; i >= 1; --i) {
			m_trackedAllocationDeletionQueue[i].swap(m_trackedAllocationDeletionQueue[i - 1]);
		}
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
