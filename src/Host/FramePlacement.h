#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

#include "Render/RenderGraph/FramePlacement.h"

namespace org {

class FrameAliasArena;

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
	};

	explicit FramePlacement(std::vector<uint32_t> epochOrder);
	~FramePlacement();
	FramePlacement(const FramePlacement&) = delete;
	FramePlacement& operator=(const FramePlacement&) = delete;

	void SetSettings(Settings settings);
	Settings GetSettings() const;
	/** @brief The frame alias arena, for its bytes in the diagnostic. */
	void SetFrameArena(std::weak_ptr<FrameAliasArena> arena);
	/** @brief An epoch graph's latest summary. Starts the worker on the first call. */
	void Publish(PlacementSummary summary);
	/** @brief The version graphs place from: null while disabled or before the first one. */
	std::shared_ptr<const FramePlacementVersion> Current() const;
	/** @brief Current()'s id, 0 for none. Lock-free, for each graph preparation. */
	uint64_t CurrentId() const noexcept { return m_currentId.load(std::memory_order_acquire); }
	/** @brief Settings::enabled. Lock-free, for each graph preparation. True before the first version too. */
	bool Enabled() const noexcept { return m_enabled.load(std::memory_order_acquire); }
	/** @brief The last layout computed, enabled or not. Null before the first one. */
	std::shared_ptr<const FramePlacementVersion> Latest() const;
	/** @brief The latest summary of each epoch, in ascending epoch order. */
	std::vector<std::shared_ptr<const PlacementSummary>> Summaries() const;
	/** @brief Logs the latest layout against today's pools. detailed: one line per resource too. */
	void Log(bool detailed) const;

private:
	void Run();
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
	uint64_t m_nextId = 1;
	std::shared_ptr<const FramePlacementVersion> m_latest, m_current;
	// Block memory every version shares (FramePlacementMemory).
	const std::shared_ptr<FramePlacementMemory> m_memory = std::make_shared<FramePlacementMemory>();
	std::atomic<uint64_t> m_currentId{0};  // m_current's id
	std::atomic<bool> m_enabled{false};    // m_settings.enabled
	bool m_stopping = false;
	std::thread m_thread;
};

}  // namespace org
