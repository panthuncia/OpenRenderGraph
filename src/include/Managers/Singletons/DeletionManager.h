#pragma once

#include <cstdint>
#include <vector>

#include <rhi_allocator.h>
#include <rhi_helpers.h>

#include "Render/Runtime/OpenRenderGraphSettings.h"
#include "Resources/TrackedAllocation.h"

class DeletionManager {
public:
	static DeletionManager& GetInstance();

	bool IsInitialized() const noexcept {
		return m_numFramesInFlight != 0 &&
			!m_deletionQueue.empty() &&
			!m_allocationDeletionQueue.empty() &&
			!m_trackedAllocationDeletionQueue.empty();
	}

	void Initialize() {
		m_numFramesInFlight = rg::runtime::GetOpenRenderGraphSettings().numFramesInFlight;
		m_deletionQueue.resize(m_numFramesInFlight);
		m_allocationDeletionQueue.resize(m_numFramesInFlight);
		m_trackedAllocationDeletionQueue.resize(m_numFramesInFlight);
	}

	void MarkForDelete(rhi::helpers::AnyObjectPtr ptr) {
		if (!IsInitialized()) {
			return;
		}
		m_deletionQueue[0].push_back(std::move(ptr));
	}

	void MarkForDelete(rhi::ma::AllocationPtr ptr) {
		if (!IsInitialized()) {
			return;
		}
		m_allocationDeletionQueue[0].push_back(std::move(ptr));
	}

	void MarkForDelete(TrackedHandle&& alloc) {
		if (!IsInitialized()) {
			alloc.Reset();
			return;
		}
		m_trackedAllocationDeletionQueue[0].push_back(std::move(alloc));
	}

	void ProcessDeletions() {
		if (!IsInitialized()) {
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

	void Cleanup() {
		m_deletionQueue.clear();
		m_allocationDeletionQueue.clear();
		m_trackedAllocationDeletionQueue.clear();
		m_numFramesInFlight = 0;
	}

private:
	uint8_t m_numFramesInFlight = 0;
	DeletionManager() = default;

	std::vector<std::vector<rhi::helpers::AnyObjectPtr>> m_deletionQueue;
	std::vector<std::vector<rhi::ma::AllocationPtr>> m_allocationDeletionQueue;
	std::vector<std::vector<TrackedHandle>> m_trackedAllocationDeletionQueue;
};

inline DeletionManager& DeletionManager::GetInstance() {
	static DeletionManager instance;
	return instance;
}
