#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <source_location>
#include <string>
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

	// A place in the code: where an object was marked, where its batch was sealed, where it was retired.
	struct Origin {
		const char* file = nullptr;
		uint32_t line = 0;
		static Origin Of(const std::source_location& where) noexcept { return {where.file_name(), where.line()}; }
	};
	// What a marked object was, for a free that turns out slow: its name (the backing's, or the allocation's), its size,
	// where it was marked, and the batch it was sealed in (an index into Retired::batches once retired).
	struct Label {
		std::string name;
		uint64_t bytes = 0;
		Origin marked;
		uint32_t batch = 0;
	};
	// A sealed batch: the frontier it waited for, and where and when it was sealed.
	struct Batch {
		runtime::SubmissionFrontier frontier;
		Origin sealedBy;
		std::chrono::steady_clock::time_point sealedAt{};
	};

	// The objects that one Retire retired. They are safe to destroy, and they are destroyed wherever the owner drops this
	// struct, or by DestroyTimed. The objects go first, then the allocations, then the tracked allocations. Each vector
	// has a label per entry.
	struct Retired {
		std::vector<TrackedHandle> trackedAllocations;
		std::vector<rhi::ma::AllocationPtr> allocations;
		std::vector<rhi::helpers::AnyObjectPtr> objects;
		std::vector<Label> trackedLabels, allocationLabels, objectLabels;
		std::vector<Batch> batches;
		Origin retiredBy;

		bool Empty() const noexcept { return objects.empty() && allocations.empty() && trackedAllocations.empty(); }

		// One free that took at least DestroyTimed's threshold.
		struct SlowFree {
			const char* kind = "";
			const Label* label = nullptr;
			const Batch* batch = nullptr;
			double ms = 0.0;
		};
		// Destroys everything in the order above, one entry at a time, and calls slow for each entry whose destruction took
		// at least slowMs. Only the slow ones cost more than a clock read.
		void DestroyTimed(double slowMs, const std::function<void(const SlowFree&)>& slow) {
			const auto timed = [&](auto& entries, const std::vector<Label>& labels, auto&& kindOf) {
				for (size_t i = 0; i < entries.size(); ++i) {
					const char* kind = kindOf(entries[i]);
					const auto begin = std::chrono::steady_clock::now();
					entries[i].Reset();
					const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
					if (ms < slowMs || !slow) continue;
					const Label* label = i < labels.size() ? &labels[i] : nullptr;
					const Batch* batch = label && label->batch < batches.size() ? &batches[label->batch] : nullptr;
					slow({kind, label, batch, ms});
				}
				entries.clear();
			};
			timed(objects, objectLabels, [](const rhi::helpers::AnyObjectPtr& object) { return KindOf(object); });
			timed(allocations, allocationLabels, [](const rhi::ma::AllocationPtr&) { return "allocation"; });
			timed(trackedAllocations, trackedLabels, [](const TrackedHandle&) { return "tracked allocation"; });
		}
	};

	// The kind of a type-erased object, for the log.
	static const char* KindOf(const rhi::helpers::AnyObjectPtr& object) noexcept {
		if (object.Is<rhi::ResourcePtr>()) return "resource";
		if (object.Is<rhi::HeapPtr>()) return "heap";
		if (object.Is<rhi::QueryPoolPtr>()) return "query pool";
		if (object.Is<rhi::PipelinePtr>()) return "pipeline";
		if (object.Is<rhi::PipelineLayoutPtr>()) return "pipeline layout";
		if (object.Is<rhi::DescriptorHeapPtr>()) return "descriptor heap";
		if (object.Is<rhi::CommandAllocatorPtr>()) return "command allocator";
		if (object.Is<rhi::CommandListPtr>()) return "command list";
		if (object.Is<rhi::CommandSignaturePtr>()) return "command signature";
		if (object.Is<rhi::TimelinePtr>()) return "timeline";
		return "object";
	}

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

	// Each MarkForDelete takes an optional name (the allocation's own is used without one) and records its caller.
	void MarkForDelete(rhi::helpers::AnyObjectPtr ptr, std::string name = {},
		const std::source_location& where = std::source_location::current()) {
		Label label{std::move(name), 0, Origin::Of(where)};
		std::scoped_lock lock(m_mutex);
		if (!m_initialized) {
			return;
		}
		m_open.objects.push_back(std::move(ptr));
		m_open.objectLabels.push_back(std::move(label));
	}

	void MarkForDelete(rhi::ma::AllocationPtr ptr, std::string name = {},
		const std::source_location& where = std::source_location::current()) {
		Label label{std::move(name), 0, Origin::Of(where)};
		if (auto* allocation = ptr.Get()) {
			label.bytes = allocation->GetSize();
			if (label.name.empty() && allocation->GetName()) label.name = allocation->GetName();
		}
		std::scoped_lock lock(m_mutex);
		if (!m_initialized) {
			return;
		}
		m_open.allocations.push_back(std::move(ptr));
		m_open.allocationLabels.push_back(std::move(label));
	}

	void MarkForDelete(TrackedHandle&& alloc, std::string name = {},
		const std::source_location& where = std::source_location::current()) {
		Label label{std::move(name), 0, Origin::Of(where)};
		if (auto* allocation = alloc.GetAllocation()) {
			label.bytes = allocation->GetSize();
			if (label.name.empty() && allocation->GetName()) label.name = allocation->GetName();
		}
		std::scoped_lock lock(m_mutex);
		if (!m_initialized) {
			alloc.Reset();
			return;
		}
		m_open.trackedAllocations.push_back(std::move(alloc));
		m_open.trackedLabels.push_back(std::move(label));
	}

	// Takes every sealed batch whose frontier is reached. Then seals everything marked since the last call with the
	// current frontier.
	// The retired objects are returned, not destroyed. Destroying them frees driver objects and takes the device's memory
	// locks. That happens after this lock is released, on whichever thread the caller chooses. So MarkForDelete never
	// waits for it.
	// where: the caller, named in the log of a slow free.
	Retired Retire(const std::source_location& where = std::source_location::current()) {
		Retired retired;
		retired.retiredBy = Origin::Of(where);
		std::scoped_lock lock(m_mutex);
		if (!m_initialized) {
			return retired;
		}
		// Frontiers only move forward, so batches retire in order. Stop at the first batch whose work is still running.
		while (!m_sealed.empty() && runtime::Reached(m_sealed.front().batch.frontier, m_reached)) {
			auto& sealed = m_sealed.front();
			const auto index = static_cast<uint32_t>(retired.batches.size());
			for (auto* labels : {&sealed.contents.objectLabels, &sealed.contents.allocationLabels, &sealed.contents.trackedLabels})
				for (auto& label : *labels) label.batch = index;
			retired.batches.push_back(std::move(sealed.batch));
			Append(retired.objects, sealed.contents.objects);
			Append(retired.allocations, sealed.contents.allocations);
			Append(retired.trackedAllocations, sealed.contents.trackedAllocations);
			Append(retired.objectLabels, sealed.contents.objectLabels);
			Append(retired.allocationLabels, sealed.contents.allocationLabels);
			Append(retired.trackedLabels, sealed.contents.trackedLabels);
			m_sealed.pop_front();
		}
		// Not settled: the open batch waits for the next Retire, whose frontier covers that submission's values.
		if (m_frontier && !m_open.Empty() && (!m_settled || m_settled())) {
			m_sealed.push_back({ Batch{ m_frontier(), Origin::Of(where), std::chrono::steady_clock::now() }, std::move(m_open) });
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
		Batch batch;
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
