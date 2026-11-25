#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <atomic>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "Render/RenderGraph/FramePlacement.h"

namespace org {

class FrameAliasArena;
class PersistentAliasArena;

/**
 * @brief GraphHost's frame-wide placement of transients.
 *
 * Each epoch graph's alias plan publishes a usage summary (Publish). This keeps the latest summary of each epoch. A worker
 * thread of its own computes a frame placement (FramePlacementVersion) from them once they have been quiet for a while, so
 * a burst of builds gives one computation. Each new layout is logged with what it would need against today's pools.
 *
 * With the placement enabled, a new layout becomes the current version (Current). Graphs take it at their next alias plan,
 * which runs at their first preparation after a build. A version is never changed once published: a new layout is a new
 * version, with blocks of its own.
 *
 * The first summary with candidates computes the first version at once, on the publishing thread, from the summaries so far,
 * so the plan that published it places from it (a graph publishes its summary before its first plan takes a version). The
 * later epochs' first plans fit their candidates into its free bytes and report what it leaves uncovered, which computes the
 * next version at once. The worker thread computes no version until every frame epoch (SetFrameEpochs) has a summary, so no
 * version is made from no summary at all.
 *
 * Once the versions have been quiet for Settings::trimAfter, a version whose heaps are larger than its need (growth headroom,
 * a class it no longer uses) is published again with exact heaps: the same layout under a new id, which graphs take in place.
 *
 * Any thread.
 */
class FramePlacement {
public:
	struct Settings {
		bool enabled = false;
		std::vector<uint32_t> repeatedEpochs;
		bool beamSearch = false;
		bool logLifetimes = false;
		std::chrono::milliseconds settle{250};
		// How the shared block memory sizes its heaps (FramePlacementMemory::Sizing).
		FramePlacementMemory::Sizing sizing;
		// The period of the memory breakdown (LogMemory) while logLifetimes is set. Zero: never.
		std::chrono::seconds memoryLogPeriod{10};
		// How long the summaries and versions stay quiet before heaps larger than their need are trimmed. Zero: never.
		std::chrono::milliseconds trimAfter{2000};
	};

	explicit FramePlacement(std::vector<uint32_t> epochOrder);
	~FramePlacement();
	FramePlacement(const FramePlacement&) = delete;
	FramePlacement& operator=(const FramePlacement&) = delete;

	void SetSettings(Settings settings);
	Settings GetSettings() const;
	/** @brief The frame alias arena, for its bytes in the diagnostic. */
	void SetFrameArena(std::weak_ptr<FrameAliasArena> arena);
	/**
	 * @brief The epochs the first version waits for: none is computed until each has a summary. Empty: the epoch order. While an
	 * epoch's graph is not built, no version is computed, so graphs place on their own.
	 */
	void SetFrameEpochs(std::vector<uint32_t> epochs);
	/** @brief An epoch graph's latest summary. Starts the worker on the first call. */
	void Publish(PlacementSummary summary);
	/** @brief The version graphs place from: null while disabled or before the first one. */
	std::shared_ptr<const FramePlacementVersion> Current() const;
	/** @brief Current()'s id, 0 for none. Lock-free, for each graph preparation. */
	uint64_t CurrentId() const noexcept { return m_currentId.load(std::memory_order_acquire); }
	/** @brief Settings::enabled. Lock-free, for each graph preparation. True before the first version too. */
	bool Enabled() const noexcept { return m_enabled.load(std::memory_order_acquire); }
	/** @brief The block memory's heaps (FramePlacementMemory::Heaps). */
	std::vector<FramePlacementMemory::HeapInfo> Heaps() const { return m_memory->Heaps(); }
	/** @brief The last layout computed, enabled or not. Null before the first one. */
	std::shared_ptr<const FramePlacementVersion> Latest() const;
	/** @brief The latest summary of each epoch, in ascending epoch order. */
	std::vector<std::shared_ptr<const PlacementSummary>> Summaries() const;
	/** @brief Logs the latest layout against today's pools. detailed: one line per resource too. */
	void Log(bool detailed) const;
	/** @brief Whether the memory breakdown is due (Settings::memoryLogPeriod while logLifetimes is set). Starts the next period. */
	bool MemoryLogDue();
	/**
	 * @brief Logs where alias memory goes now:
	 * - each frame placement heap, current or replaced and still alive: its class, bytes reserved against the bytes its latest
	 *   version packed, its generation, and what holds it (the resources still placed in it, by epoch, and its reference count),
	 * - the frame alias arena's blocks against the bytes reserved in them,
	 * - the shared arena's pools (pools may be null),
	 * - graphs, the line the host gives (which version each graph placed from).
	 */
	void LogMemory(PersistentAliasArena* pools, const std::string& graphs) const;

private:
	void Run();
	// Under m_mutex: whether every frame epoch has a summary.
	bool SummariesCompleteLocked() const;
	// Under m_mutex: the inputs of a computation from the latest summaries, which summaries receives.
	FramePlacementInputs InputsLocked(std::vector<std::shared_ptr<const PlacementSummary>>& summaries) const;
	// Under m_mutex: when the current version's heaps are due for a trim check, if they are.
	std::optional<std::chrono::steady_clock::time_point> TrimDueLocked() const;
	void LogVersion(const FramePlacementVersion& version, const std::vector<std::shared_ptr<const PlacementSummary>>& summaries,
		bool detailed) const;

	const std::vector<uint32_t> m_epochOrder;
	mutable std::mutex m_mutex;
	std::condition_variable m_wake;
	Settings m_settings;
	std::weak_ptr<FrameAliasArena> m_frameArena;
	std::map<uint32_t, std::shared_ptr<const PlacementSummary>> m_summaries;
	uint64_t m_changes = 0;    // summary or setting changes so far
	uint64_t m_computed = 0;   // the changes the last computation saw
	std::chrono::steady_clock::time_point m_lastChange{};
	std::chrono::steady_clock::time_point m_lastPublish{};  // when the current version was published
	std::vector<uint32_t> m_frameEpochs;                    // SetFrameEpochs
	uint64_t m_trimChecked = 0;                             // the version the last trim check ran for
	uint64_t m_nextId = 1;
	std::shared_ptr<const FramePlacementVersion> m_latest, m_current;
	// Block memory every version shares (FramePlacementMemory).
	const std::shared_ptr<FramePlacementMemory> m_memory = std::make_shared<FramePlacementMemory>();
	std::atomic<uint64_t> m_currentId{0};  // m_current's id
	std::atomic<bool> m_enabled{false};    // m_settings.enabled
	std::chrono::steady_clock::time_point m_nextMemoryLog{};  // MemoryLogDue, under m_mutex
	bool m_stopping = false;
	std::thread m_thread;
};

}  // namespace org
