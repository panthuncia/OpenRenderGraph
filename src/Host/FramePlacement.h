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
 * Staged sets (BeginStaging): graphs built off the owner thread place their transients before they are attached. Their summaries
 * go to the staging, and a version they need is computed into it, not published. The running graphs keep the current version
 * and their backings until the set attaches (CommitStaging), which publishes both at once.
 *
 * A frame runs one version (Freeze): once a graph that placed from a version has recorded in a frame, Current and CurrentId give
 * that version until the host thaws it at the frame's end. A version the worker publishes meanwhile (settled, trimmed) reaches
 * every graph at its next frame, so no graph moves a resource another graph of the frame wrote already. A version a plan of the
 * frame makes at once for its uncovered candidates becomes the frame's version at once: the older one may give that graph no
 * valid layout (a resource newly shared with an epoch that ran already).
 *
 * Any thread.
 */
class FramePlacement : public std::enable_shared_from_this<FramePlacement> {
public:
	/**
	 * @brief A staged set's view of the frame placement (BeginStaging). Its graphs' usage summaries go here, and a version their
	 * uncovered candidates need is computed here (pending) instead of published. CommitStaging publishes both; a staging closed
	 * without it (DropStaging, or its last reference gone) changes nothing. While one is open the worker publishes no version.
	 */
	class Staging {
	public:
		~Staging();
		Staging(const Staging&) = delete;
		Staging& operator=(const Staging&) = delete;
		/** @brief Committed or dropped: its graphs then take the host's current version, as attached graphs do. */
		bool Closed() const noexcept { return m_closed.load(std::memory_order_acquire); }
		/** @brief The version its graphs place from (the pending one, else the current version when it began). Lock-free. */
		uint64_t CurrentId() const noexcept { return m_currentId.load(std::memory_order_acquire); }
	private:
		friend class FramePlacement;
		Staging() = default;
		std::weak_ptr<FramePlacement> m_owner;
		std::vector<uint32_t> m_epochs;
		// Under the owner's mutex.
		std::shared_ptr<const FramePlacementVersion> m_base, m_pending;
		std::map<uint32_t, std::shared_ptr<const PlacementSummary>> m_summaries;
		std::atomic<uint64_t> m_currentId{0};
		std::atomic<bool> m_closed{false};
		// Under the owner's mutex: a graph of the set placed from the pending version (or left its placement to the attach while
		// it was current), so the set keeps it; or the set declined it (DeclineStaged) and places from its base from then on.
		bool m_pendingUsed = false, m_declined = false;
		// Under the owner's mutex: the set's epochs whose graphs placed at the build (NoteStagedPlacement). A later version of the
		// staging keeps their resources too.
		std::vector<uint32_t> m_placedEpochs;
		// Its versions keep the running graphs' resources (Settings::keepRunning, and some epoch with transients is not in the set).
		bool m_keep = false;
	};

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
		// A version made at once for a plan's uncovered candidates (a staged set's, or an attached graph's) keeps the resources
		// the running graphs bind where they are (FramePlacementVersion::Keeping), so those graphs take it without placing
		// again. Off, or when that cannot be done: the version is packed from scratch, and every graph places again from it.
		bool keepRunning = true;
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
	/** @brief The version graphs place from: null while disabled or before the first one. The frozen version while frozen. */
	std::shared_ptr<const FramePlacementVersion> Current() const;
	/** @brief Current()'s id, 0 for none. Lock-free, for each graph preparation. */
	uint64_t CurrentId() const noexcept {
		return m_frozen.load(std::memory_order_acquire) ? m_frozenId.load(std::memory_order_acquire) : m_currentId.load(std::memory_order_acquire);
	}
	/**
	 * @brief The frame's version: from now until Thaw, Current and CurrentId give it, whatever is published meanwhile (the owner
	 * thread calls it once a graph that placed from it has recorded in the frame, and thaws at the frame's end).
	 */
	void Freeze(std::shared_ptr<const FramePlacementVersion> version);
	void Thaw();
	/** @brief A staging for a set of graphs of these epochs, based on the current version (not the frozen one). */
	std::shared_ptr<Staging> BeginStaging(std::vector<uint32_t> epochs);
	/** @brief A staged graph's summary. Computes the staging's pending version at once when its plan needs one (as Publish). */
	void PublishStaged(Staging& staging, PlacementSummary summary);
	/** @brief The version the staging's graphs place from: its pending version, else its base. */
	std::shared_ptr<const FramePlacementVersion> StagedCurrent(const Staging& staging) const;
	/**
	 * @brief A staged graph cannot use the pending version (it would move resources the running graphs bind): the set places from
	 * its base instead, and computes no other version. Refused (false) once a graph of the set placed from the pending one. True
	 * again once declined, for the set's later graphs.
	 */
	bool DeclineStaged(Staging& staging);
	/**
	 * @brief A graph of the set placed from this version (or left its placement to the attach while it was the staging's). placed:
	 * its epoch's graph placed at the build, so a later version of the staging keeps its resources too.
	 */
	void NoteStagedPlacement(Staging& staging, uint64_t versionId, uint32_t epoch = UINT32_MAX, bool placed = false);
	/** @brief Allocates the block memory of the staging's pending version now, so the set's attach allocates none. Any thread. */
	void AcquireStagedBlocks(const Staging& staging);
	/**
	 * @brief At the set's attach (owner thread, between frames): its summaries replace the epochs' ones, and its pending version, if
	 * any, is published. Returns whether a version was published. Closes it.
	 */
	bool CommitStaging(Staging& staging);
	/** @brief Closes the staging without changing anything (its set was dropped). */
	void DropStaging(Staging& staging);
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
	// Under m_mutex: a staging closed (committed or dropped).
	void CloseStagingLocked(Staging& staging);
	// Under m_mutex: whether every frame epoch has a summary.
	bool SummariesCompleteLocked() const;
	// Under m_mutex: the inputs of a computation from the latest summaries, which summaries receives.
	FramePlacementInputs InputsLocked(std::vector<std::shared_ptr<const PlacementSummary>>& summaries) const;
	// Under m_mutex: when the current version's heaps are due for a trim check, if they are.
	std::optional<std::chrono::steady_clock::time_point> TrimDueLocked() const;
	// Any thread, no lock held: fresh, or a version that keeps every resource placed from a frame placement where it is
	// (FramePlacementVersion::Keeping) when the settings ask for it and base allows one. keptEpochs: the epochs whose graphs keep
	// running. what names the computation in the log.
	std::shared_ptr<FramePlacementVersion> KeepRunning(std::shared_ptr<FramePlacementVersion> fresh,
		const std::shared_ptr<const FramePlacementVersion>& base, std::vector<uint32_t> keptEpochs, bool keep, const char* what,
		const std::vector<uint32_t>* pinEpochs = nullptr) const;
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
	// Freeze: the frame's version, under m_mutex; m_frozenId and m_frozen for CurrentId.
	std::shared_ptr<const FramePlacementVersion> m_frozenVersion;
	std::atomic<uint64_t> m_frozenId{0};
	std::atomic<bool> m_frozen{false};
	size_t m_openStagings = 0;  // under m_mutex: the worker publishes nothing while one is open
	std::chrono::steady_clock::time_point m_nextMemoryLog{};  // MemoryLogDue, under m_mutex
	bool m_stopping = false;
	std::thread m_thread;
};

}  // namespace org
