#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "Render/LatchBlock.h"
#include "Render/RunRef.h"

namespace org {

// A per-run on/off switch for a client's indirect dispatches and draws that keeps their recording.
//
// A pass records an indirect dispatch or draw whose arguments live in the gate's latch block (Latch, Offset), so the
// recording stays current whatever the gate says. A run's region starts zeroed, so every entry starts closed. The host
// closes every gate of a client at the start of each run, and the client's commit opens the entries that have work.
//
// Beside the latch words, the gate keeps a bit per entry and lease index: whether the run opened the entry with work.
// A pass recorded at execution reads it (IsOpen) to skip runs that left it closed.
//
// Open and Close run on the host's writing thread (inside the run's commit), IsOpen on the recording thread.
class RunGate {
public:
	// Three group counts and a pad, or four draw words.
	static constexpr uint32_t kEntryBytes = 16;

	// A closed gate with `entries` entries. Null, with the reason in `error` when it is given, when there are no entries
	// or no runtime with a frame arena (Runtime::Current). Gate names a type derived from RunGate with RunGate's
	// constructor, for a host that adds its own helpers.
	template<class Gate = RunGate>
	static std::shared_ptr<Gate> Create(std::string_view name, uint32_t entries, std::string* error = nullptr) {
		auto latch = CreateLatch(name, entries, error);
		if (!latch) return nullptr;
		return std::make_shared<Gate>(std::move(latch), entries);
	}

	RunGate(std::shared_ptr<LatchBlock> latch, uint32_t entries);
	~RunGate();
	RunGate(const RunGate&) = delete;
	RunGate& operator=(const RunGate&) = delete;

	uint32_t Entries() const noexcept { return m_entries; }

	// Closes every entry for the run: zeroes its latch words and its open bits.
	void Close(const RunRef& run) const;

	// Entry `entry` dispatches x * y * z groups in the run.
	void OpenDispatch(const RunRef& run, uint32_t entry, uint32_t x, uint32_t y = 1, uint32_t z = 1) const;

	// Entry `entry` draws `vertices` vertices `instances` times in the run. The default is a fullscreen triangle.
	void OpenDraw(const RunRef& run, uint32_t entry, uint32_t vertices = 3, uint32_t instances = 1) const;

	// Copies the gate's open bits of run `from` onto run `to`. The latch words travel with the run on their own.
	void CopyOpen(const RunRef& from, const RunRef& to) const;

	// CopyOpen for every live gate. A host that gathers a run's values in a staging run calls it when it takes the run.
	static void CopyOpenAll(const RunRef& from, const RunRef& to);

	// Clears every live gate's open bits for the run, not its latch words.
	static void ClearOpenAll(const RunRef& run);

	// Whether the run opened `entry` with work to do. Only for passes recorded at execution.
	// True when the gate cannot tell (an entry or lease index out of range), so a caller that skips on false never drops work.
	bool IsOpen(const RunRef& run, uint32_t entry) const;

	// The latch and offset an indirect dispatch or draw reads for `entry`.
	const LatchBlock& Latch() const noexcept { return *m_latch; }
	uint32_t Offset(uint32_t entry) const noexcept { return entry * kEntryBytes; }

private:
	// Null, with the reason in `error`, when the gate cannot be made.
	static std::shared_ptr<LatchBlock> CreateLatch(std::string_view name, uint32_t entries, std::string* error);

	// `work` says whether the words make the dispatch or draw do something.
	void Write(const RunRef& run, uint32_t entry, const std::array<uint32_t, 4>& words, bool work) const;

	std::shared_ptr<LatchBlock> m_latch;
	uint32_t m_entries = 0;
	// A bit per entry the run opened with work, one row per lease index.
	// Close clears a row at each run's start, and the run is over before the lease index is reused.
	uint32_t m_words = 0;
	std::unique_ptr<std::atomic<uint64_t>[]> m_open;
};

// Work that runs until one run with it submits, such as a first fill or clear, through a RunGate entry.
// The client's commit opens its entry while the work is pending, and the run's executed callback ends it. Keep one per
// registration, so a new registration runs the work again. The work must be safe to run twice. The client must stay active
// while Pending(), so its commit runs. One thread: the host's writing thread.
class FirstRun {
public:
	bool Pending() const noexcept { return m_pending; }

	// In the commit: opens `entry` of the run with x * y * z groups while the work is pending. A copy or clear takes one group.
	void Open(const RunGate& gate, const RunRef& run, uint32_t entry, uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) {
		m_opened = m_pending;
		if (m_pending) gate.OpenDispatch(run, entry, x, y, z);
	}

	// In the executed callback: the run submitted. True when its commit had opened the work, which is then done.
	bool Executed() {
		if (!std::exchange(m_opened, false)) return false;
		m_pending = false;
		return true;
	}

private:
	bool m_pending = true;  // no run whose commit opened the work has submitted yet
	bool m_opened = false;  // the last commit opened the work
};

} // namespace org
