#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "Render/RunRef.h"

namespace org {

// Every live RunSlots, so a host that gathers a run's values in a staging run (GraphHost::EpochStagingRun) can move them
// onto the run it submits, as LatchBlock::CopyStableRun does for latch regions. Thread-safe.
class RunSlotsRegistry {
public:
	struct Entry {
		virtual void MoveRun(const RunRef& from, const RunRef& to) const = 0;
	};

	static void MoveAll(const RunRef& from, const RunRef& to) {
		auto& registry = Get();
		std::scoped_lock lock(registry.m_mutex);
		for (const auto* entry : registry.m_entries) entry->MoveRun(from, to);
	}

	static void Add(const Entry* entry) {
		auto& registry = Get();
		std::scoped_lock lock(registry.m_mutex);
		registry.m_entries.push_back(entry);
	}

	static void Remove(const Entry* entry) {
		auto& registry = Get();
		std::scoped_lock lock(registry.m_mutex);
		std::erase(registry.m_entries, entry);
	}

private:
	static RunSlotsRegistry& Get() {
		// Never destroyed, since RunSlots in static objects remove themselves during static destruction.
		static auto* registry = new RunSlotsRegistry;
		return *registry;
	}

	std::mutex m_mutex;
	std::vector<const Entry*> m_entries;
};

// One value per run, stored by a client's commit and taken by the pass that records that run.
// A value is found only by its own run, so a later run of the same lease index takes nothing. Thread-safe.
template<class T>
class RunSlots final : public RunSlotsRegistry::Entry {
public:
	RunSlots() { RunSlotsRegistry::Add(this); }
	~RunSlots() { RunSlotsRegistry::Remove(this); }
	RunSlots(const RunSlots&) = delete;
	RunSlots& operator=(const RunSlots&) = delete;

	// Moves run `from`'s value, if it has one, onto run `to`.
	void MoveRun(const RunRef& from, const RunRef& to) const override {
		std::scoped_lock lock(m_mutex);
		if (from.lease >= m_entries.size() || !(m_entries[from.lease].run == from)) return;
		auto value = std::exchange(m_entries[from.lease].value, nullptr);
		if (m_entries.size() <= to.lease) m_entries.resize(static_cast<std::size_t>(to.lease) + 1);
		m_entries[to.lease] = {to, std::move(value)};
	}

	void Store(const RunRef& run, std::shared_ptr<const T> value) const {
		std::scoped_lock lock(m_mutex);
		if (m_entries.size() <= run.lease) m_entries.resize(static_cast<std::size_t>(run.lease) + 1);
		m_entries[run.lease] = {run, std::move(value)};
	}

	// The run's value, left in place for a pass that reads it more than once.
	std::shared_ptr<const T> Find(const RunRef& run) const {
		std::scoped_lock lock(m_mutex);
		if (run.lease >= m_entries.size() || !(m_entries[run.lease].run == run)) return nullptr;
		return m_entries[run.lease].value;
	}

	// The run's value, taken out.
	std::shared_ptr<const T> Take(const RunRef& run) const {
		std::scoped_lock lock(m_mutex);
		if (run.lease >= m_entries.size() || !(m_entries[run.lease].run == run)) return nullptr;
		return std::exchange(m_entries[run.lease].value, nullptr);
	}

	void Clear() const {
		std::scoped_lock lock(m_mutex);
		m_entries.clear();
	}

private:
	struct Slot {
		RunRef run{};
		std::shared_ptr<const T> value;
	};
	mutable std::mutex m_mutex;
	mutable std::vector<Slot> m_entries;  // by lease index
};

} // namespace org
