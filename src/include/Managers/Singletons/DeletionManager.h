#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <vector>

#include <rhi_allocator.h>
#include <rhi_helpers.h>

#include "Render/Runtime/SubmissionFrontier.h"
#include "Resources/TrackedAllocation.h"


namespace org {

// Defers the destruction of GPU objects by submission frontier. A marked object waits until every queue reaches the
// progress it had at the next Retire. That progress is a SubmissionFrontier. Any work that could still use the object
// was submitted before then. The runtime installs the frontier with SetFrontier. Without a frontier, nothing retires
// before DrainAll.
// A batch is sealed only while the frontier is settled: no submission is between taking its values and publishing them.
// A frontier taken inside that window would miss the values that submission is about to signal.
class DeletionManager {
public:
	struct Stats {
		uint64_t objectCount = 0;
		uint64_t allocationCount = 0;
		uint64_t trackedAllocationCount = 0;
	};

	// The objects that one Retire retired. They are safe to destroy, and they are destroyed wherever the owner drops this
	// struct. Members are destroyed in reverse order. So the objects go first, then the allocations, then the tracked
	// allocations.
	struct Retired {
		std::vector<TrackedHandle> trackedAllocations;
		std::vector<rhi::ma::AllocationPtr> allocations;
		std::vector<rhi::helpers::AnyObjectPtr> objects;

		bool Empty() const noexcept { return objects.empty() && allocations.empty() && trackedAllocations.empty(); }
	};

	static DeletionManager& GetInstance();

	bool IsInitialized() const {
		std::scoped_lock lock(m_mutex);
		return m_initialized;
	}

	void Initialize() {
		std::scoped_lock lock(m_mutex);
		m_initialized = true;
	}

	// Installs the runtime's frontier, which Retire seals batches with, and the check for when its points are reached.
	// settled says whether the frontier is complete now (no values handed out but not yet published). Without it, it
	// always is. All three are called under this manager's lock, so they must not mark anything.
	void SetFrontier(std::function<runtime::SubmissionFrontier()> frontier, runtime::PointReached reached,
		std::function<bool()> settled = {}) {
		std::scoped_lock lock(m_mutex);
		m_frontier = std::move(frontier);
		m_reached = std::move(reached);
		m_settled = std::move(settled);
	}

	void MarkForDelete(rhi::helpers::AnyObjectPtr ptr) {
		std::scoped_lock lock(m_mutex);
		if (!m_initialized) {
			return;
		}
		m_open.objects.push_back(std::move(ptr));
	}

	void MarkForDelete(rhi::ma::AllocationPtr ptr) {
		std::scoped_lock lock(m_mutex);
		if (!m_initialized) {
			return;
		}
		m_open.allocations.push_back(std::move(ptr));
	}

	void MarkForDelete(TrackedHandle&& alloc) {
		std::scoped_lock lock(m_mutex);
		if (!m_initialized) {
			alloc.Reset();
			return;
		}
		m_open.trackedAllocations.push_back(std::move(alloc));
	}

	// Takes every sealed batch whose frontier is reached. Then seals everything marked since the last call with the
	// current frontier.
	// The retired objects are returned, not destroyed. Destroying them frees driver objects and takes the device's memory
	// locks. That happens after this lock is released, on whichever thread the caller chooses. So MarkForDelete never
	// waits for it.
	Retired Retire() {
		Retired retired;
		std::scoped_lock lock(m_mutex);
		if (!m_initialized) {
			return retired;
		}
		// Frontiers only move forward, so batches retire in order. Stop at the first batch whose work is still running.
		while (!m_sealed.empty() && runtime::Reached(m_sealed.front().frontier, m_reached)) {
			auto& batch = m_sealed.front().contents;
			Append(retired.objects, batch.objects);
			Append(retired.allocations, batch.allocations);
			Append(retired.trackedAllocations, batch.trackedAllocations);
			m_sealed.pop_front();
		}
		// Not settled: the open batch waits for the next Retire, whose frontier covers that submission's values.
		if (m_frontier && !m_open.Empty() && (!m_settled || m_settled())) {
			m_sealed.push_back({ m_frontier(), std::move(m_open) });
			m_open = {};
		}
		return retired;
	}

	// Runs one Retire and destroys what it retired on the calling thread.
	void ProcessDeletions() {
		(void)Retire();
	}

	Stats GetStats() const {
		std::scoped_lock lock(m_mutex);
		Stats stats{};
		auto count = [&](const Retired& batch) {
			stats.objectCount += batch.objects.size();
			stats.allocationCount += batch.allocations.size();
			stats.trackedAllocationCount += batch.trackedAllocations.size();
		};
		count(m_open);
		for (const auto& batch : m_sealed) count(batch.contents);
		return stats;
	}

	// The device is idle: everything goes.
	void DrainAll() {
		std::scoped_lock lock(m_mutex);
		m_sealed.clear();
		m_open = {};
	}

	void Cleanup() {
		std::scoped_lock lock(m_mutex);
		m_sealed.clear();
		m_open = {};
		m_frontier = {};
		m_reached = {};
		m_settled = {};
		m_initialized = false;
	}

private:
	struct Sealed {
		runtime::SubmissionFrontier frontier;
		Retired contents;
	};

	DeletionManager() = default;

	template <class T>
	static void Append(std::vector<T>& to, std::vector<T>& from) {
		if (to.empty()) { to.swap(from); return; }
		for (auto& item : from) to.push_back(std::move(item));
		from.clear();
	}

	mutable std::mutex m_mutex;
	bool m_initialized = false;
	std::function<runtime::SubmissionFrontier()> m_frontier;
	runtime::PointReached m_reached;
	std::function<bool()> m_settled;
	Retired m_open;
	std::deque<Sealed> m_sealed;
};

inline DeletionManager& DeletionManager::GetInstance() {
	static DeletionManager instance;
	return instance;
}


} // namespace org
