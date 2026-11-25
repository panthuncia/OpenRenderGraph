#pragma once

#include <algorithm>
#include <array>
#include <map>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "Resources/AliasingPlacement.h"

namespace org {

/**
 * @brief One alias candidate of an epoch graph's alias plan, as the graph's last structural build saw it.
 *
 * A host uses these to place transients across the whole frame (FramePlacementVersion).
 */
struct PlacementUse {
	// Stable across builds of the epoch's graph: a hash of the resource's name, its memory description (size, alignment,
	// memory class, backends) and its ordinal among the graph's candidates with the same name and description, in first-use
	// order. A new build makes new resources, so their identities change, but their keys do not. Unique in the summary.
	uint64_t key = 0;
	// The resource's identity (Resource::GetGlobalResourceID plus one; 0 for none). Graphs of several epochs that use one
	// resource give the same identity, so it is matched across epochs by identity, never by name.
	uint64_t objectId = 0;
	std::string name;
	// Pass ranks within the epoch's schedule, inclusive.
	uint32_t firstRank = 0, lastRank = 0;
	// False: the first use in this epoch reads. In the frame's first epoch that uses it, that is a previous-frame read.
	bool firstUseWrites = false;
	// False when the pass of the first use may skip its writes (ExternalPassDesc::mayNotWrite).
	bool writeGuaranteed = false;
	// Its contents cross the runs of a repeated epoch (Resource::SetContentAcrossRuns). The host spans the whole slot of each
	// epoch that uses it, so it has one lifetime over every run. The host also spans the whole slot of a repeated epoch for a
	// resource several epochs use.
	bool loopCrossing = false;
	// The graph's planner would place it in an automatic pool on one backend, if its first use wrote. False for manual pools,
	// resources shared between devices, resources used on several backends, scores below the threshold and auto aliasing off.
	bool eligible = false;
	// Why it is not eligible, or the graph's own exclusion. Empty otherwise.
	std::string exclusion;
	uint64_t size = 0, alignment = 1;
	// 1 buffers, 2 render target and depth textures, 3 other textures. The planner's pools keep them apart.
	uint8_t memoryClass = 0;
	uint64_t backendMask = 0;
};

/** @brief A resource placed from a frame placement that is still at that placement (FramePlacedResidents). */
struct FramePlacedResident {
	const AliasHeapGeneration* heap = nullptr;  // the heap generation it lies in
	std::string name;
	uint64_t offset = 0, size = 0;
	uint32_t epoch = UINT32_MAX;  // the epoch whose graph placed it
};

/**
 * @brief Every resource that an epoch graph's alias plan placed from a frame placement and that is still there. Each one keeps
 * its heap generation alive. Any thread.
 */
std::vector<FramePlacedResident> FramePlacedResidents();

/** @brief Where an epoch graph's own transients go today. */
enum class PlacementMemory : uint8_t {
	SharedArena,   // the host's shared alias arena: every such epoch packs from offset 0 of the same pools
	PrivatePools,  // pools of the graph's own
	FrameArena,    // a range of the frame alias arena reserved for the epoch's frame position
};

/**
 * @brief What one epoch graph's structural build publishes about its alias candidates (RenderGraph::
 * SetPlacementSummaryObserver).
 */
struct PlacementSummary {
	uint32_t epoch = UINT32_MAX;
	// Passes in the epoch's schedule. Ranks run from 0 to rankCount - 1.
	uint32_t rankCount = 0;
	PlacementMemory memory = PlacementMemory::PrivatePools;
	// One per candidate, sorted by key. Each key appears once (Finish merges repeated keys and makes them ineligible).
	std::vector<PlacementUse> uses;
	// What this graph's own pools needed in its plan, by memory class (index 1 to 3), and in total.
	std::array<uint64_t, 4> pooledBytesByClass{};
	uint64_t pooledBytes = 0;
	// The bytes this graph placed from a frame placement (FramePlacementVersion) in its plan.
	uint64_t framePlacedBytes = 0;
	// Candidates the graph would pool that the version it placed from did not place: on its own path, or held back. While no
	// summary has any, the version still serves every graph, and the host publishes a new one only when it is smaller.
	uint32_t uncovered = 0;
	// A hash of the epoch, the ranks, the uses and uncovered. Two summaries with the same hash lead to the same frame placement.
	uint64_t hash = 0;

	/** @brief Sorts the uses, merges repeated keys and sets hash. Call it once the summary is filled in. */
	void Finish();
};

/** @brief What a frame placement is computed from. */
struct FramePlacementInputs {
	// The epochs in the order the frame runs them. Epochs with a summary and not in the order follow it, in ascending order.
	std::vector<uint32_t> epochOrder;
	// Epochs that may run more than once a frame. A resource that several epochs use spans the whole slot of each repeated
	// epoch at its ends.
	std::vector<uint32_t> repeatedEpochs;
	std::vector<std::shared_ptr<const PlacementSummary>> summaries;
	// Beam search instead of the greedy sweep line (alias::PackBeamSearch).
	bool beamSearch = false;
};

/**
 * @brief One placement of every alias candidate of a frame, across all its epoch graphs.
 *
 * The frame's timeline is the epochs in order, each a slot holding its ranks. A resource's lifetime is the union of its
 * uses across epochs. The resources are packed per memory class and backend set, one block each, so two resources share
 * bytes only when their lifetimes do not overlap in the frame.
 *
 * A version is immutable once computed, apart from its block memory. That is allocated when a graph first places into a
 * block (AcquireBlock), and freed when the version goes. Placements made in a block keep its memory alive through their
 * leases.
 *
 * Deterministic: the same inputs give the same layout and the same LayoutHash.
 */
/**
 * @brief Block memory shared by the versions of one host.
 *
 * A version's block takes the heap an earlier version used for the same memory class and backend set when it is big enough.
 * So a new version with the same layout, or a similar one, allocates nothing. Graphs that still place from the earlier version
 * share the heap with the new one for a frame or two. The alias ledger orders their uses of the same bytes, as it does for
 * any placed resource. A heap too small, or too big (more than shrinkFactor times what it needs with headroom), is replaced.
 * The replaced memory stays alive through the placements made in it. Heaps lists the replaced ones that are still alive.
 *
 * A heap is made at its need. Only a heap that grows gets headroom (the larger of its need and headroom times its old size), so a burst of versions that grow a little reuses it. Once
 * the versions settle, the host trims the heaps to the need (FramePlacement::Settings::trimAfter): a version whose blocks take
 * exact heaps (Acquire's exact).
 *
 * Any thread.
 */
class FramePlacementMemory {
public:
	/** @brief How heaps are sized. */
	struct Sizing {
		// A heap that grows takes the larger of its need and this many times its old capacity (as the planner's pools), so
		// small growth between versions reuses it. A heap's first size is its need.
		double headroom = 1.25;
		// A heap more than this many times its need with headroom is replaced by a smaller one.
		double shrinkFactor = 2.0;
	};
	/** @brief One heap, current or replaced and still alive (Heaps). */
	struct HeapInfo {
		uint8_t memoryClass = 0;
		uint64_t backendMask = 0;
		uint64_t capacity = 0, generation = 0;
		// The capacity the latest version asked for when it took the heap (its block's packed bytes). 0 for a replaced heap.
		uint64_t needed = 0;
		uint64_t neededBy = 0;  // that version's id
		bool current = false;
		// Owners of the heap generation now: the placements made in it, the bindings and recordings that hold them, and
		// FramePlacementMemory's own reference to a current heap.
		long references = 0;
		const AliasHeapGeneration* identity = nullptr;
	};

	FramePlacementMemory() = default;
	~FramePlacementMemory();
	FramePlacementMemory(const FramePlacementMemory&) = delete;
	FramePlacementMemory& operator=(const FramePlacementMemory&) = delete;
	/**
	 * @brief A heap of at least capacity bytes. poolID and versionID are for memory statistics and the log. exact: a heap larger
	 * than capacity (beyond TrimSlack) is replaced by one of capacity, and a new heap gets no headroom. Throws on failure.
	 */
	std::shared_ptr<const AliasHeapGeneration> Acquire(uint8_t memoryClass, uint64_t backendMask, uint64_t capacity, uint64_t alignment,
		uint64_t poolID, uint64_t versionID, bool exact = false);
	/**
	 * @brief Whether a version with these blocks would trim a heap: a current heap is larger than its blocks' need by more than
	 * TrimSlack, or no block of the version uses its class and backends.
	 */
	bool WouldTrim(const std::vector<std::pair<std::pair<uint8_t, uint64_t>, uint64_t>>& needs) const;
	/** @brief Drops the current heaps of the classes and backends no block in needs uses. Their placements keep them alive. */
	void ReleaseUnused(const std::vector<std::pair<std::pair<uint8_t, uint64_t>, uint64_t>>& needs);
	// What a heap may exceed its need by before a trim replaces it: the larger of 4 MB and 1/64 of the need.
	static uint64_t TrimSlack(uint64_t need) noexcept { return (std::max)(uint64_t{4} << 20, need / 64); }
	/** @brief The current heaps' capacities, in bytes. Replaced heaps still alive are not counted (Heaps lists them). */
	uint64_t Bytes() const;
	/** @brief The current heaps, then the replaced heaps that are still alive. Drops the records of replaced heaps that are gone. */
	std::vector<HeapInfo> Heaps() const;
	/** @brief Sets how later heaps are sized. Heaps already made stay until a version replaces them. */
	void SetSizing(Sizing sizing);

private:
	struct Heap {
		uint64_t capacity = 0, alignment = 1, generation = 0;
		uint64_t needed = 0, neededBy = 0;
		TrackedHandle allocation;
		std::shared_ptr<const AliasHeapGeneration> heap;
	};
	struct Replaced {
		uint8_t memoryClass = 0;
		uint64_t backendMask = 0, capacity = 0, generation = 0;
		std::weak_ptr<const AliasHeapGeneration> heap;
	};
	mutable std::mutex m_mutex;
	Sizing m_sizing;
	std::map<std::pair<uint8_t, uint64_t>, Heap> m_heaps;  // by memory class and backend mask
	mutable std::vector<Replaced> m_replaced;
};

class FramePlacementVersion {
public:
	/** @brief An epoch's slot in the frame's timeline. */
	struct Slot {
		uint32_t epoch = 0;
		uint32_t base = 0;       // the timeline position of the epoch's rank 0
		uint32_t rankCount = 0;
		bool repeated = false;
		PlacementMemory memory = PlacementMemory::PrivatePools;
	};
	/** @brief One memory block of the placement. */
	struct Block {
		uint64_t backendMask = 0;
		uint8_t memoryClass = 0;
		uint64_t capacity = 0, alignment = 1;
		uint64_t poolID = 0;  // for memory statistics and the alias ledger's records
		uint32_t members = 0;
	};
	/** @brief A resource's ranks in one epoch. */
	struct Use {
		uint32_t epoch = 0;
		uint32_t firstRank = 0, lastRank = 0;
		bool firstUseWrites = false;
		uint64_t key = 0;  // its key in that epoch's summary
	};
	/**
	 * @brief One resource and where it goes. Uses of several epochs are one entry when they name the same resource (its
	 * identity), never by name: two epochs' resources with the same key but different identities are two entries.
	 */
	struct Entry {
		uint64_t key = 0;  // its key in its first epoch's summary
		uint64_t objectId = 0;  // PlacementUse::objectId when the summaries were taken
		std::string name;
		uint64_t size = 0, alignment = 1;
		uint8_t memoryClass = 0;
		uint64_t backendMask = 0;
		std::vector<Use> uses;  // in frame order
		uint32_t frameFirst = 0, frameLast = 0;  // its lifetime on the frame's timeline, inclusive
		bool placed = false;
		uint32_t block = UINT32_MAX;
		uint64_t offset = 0;
		// Its contents cross runs (PlacementUse::loopCrossing): its lifetime spans the whole slot of every epoch that uses it.
		bool acrossRuns = false;
		// Why it is not placed. Empty when placed.
		std::string reason;
		bool MultiEpoch() const noexcept { return uses.size() > 1; }
		// The use in an epoch, or null.
		const Use* UseIn(uint32_t epoch) const noexcept;
	};

	/** @brief Computes the placement. id: the version's number, which names its blocks' pool ids. */
	static std::shared_ptr<FramePlacementVersion> Compute(const FramePlacementInputs& inputs, uint64_t id);

	FramePlacementVersion() = default;
	~FramePlacementVersion();
	FramePlacementVersion(const FramePlacementVersion&) = delete;
	FramePlacementVersion& operator=(const FramePlacementVersion&) = delete;

	uint64_t Id() const noexcept { return m_id; }
	const std::vector<Slot>& Slots() const noexcept { return m_slots; }
	const std::vector<Block>& Blocks() const noexcept { return m_blocks; }
	const std::vector<Entry>& Entries() const noexcept { return m_entries; }
	/** @brief The entry for an epoch's key (PlacementUse::key), or null. */
	const Entry* Find(uint64_t key, uint32_t epoch) const noexcept;
	/** @brief The entry for a resource identity (PlacementUse::objectId), or null. */
	const Entry* FindObject(uint64_t objectId) const noexcept;
	/** @brief Its blocks take heaps of exactly their need (FramePlacementMemory::Acquire's exact): a trim of the heaps. */
	bool ExactHeaps() const noexcept { return m_exactHeaps; }
	void SetExactHeaps(bool exact) noexcept { m_exactHeaps = exact; }
	/** @brief Each block's class and backends with its capacity, summed per class and backends (FramePlacementMemory::WouldTrim). */
	std::vector<std::pair<std::pair<uint8_t, uint64_t>, uint64_t>> BlockNeeds() const;
	/** @brief Whether an epoch has a slot. */
	bool HasEpoch(uint32_t epoch) const noexcept;
	/** @brief A hash of the slots, blocks and entries. It leaves the id out, so two versions with one layout match. */
	uint64_t LayoutHash() const noexcept { return m_layoutHash; }
	/** @brief The blocks' capacities, in total and by memory class. */
	uint64_t Bytes() const noexcept;
	uint64_t BytesOfClass(uint8_t memoryClass) const noexcept;
	/**
	 * @brief The block's memory, allocated by the first call. Any thread. Throws when the memory cannot be allocated, or for a
	 * block the version does not have.
	 */
	std::shared_ptr<const AliasHeapGeneration> AcquireBlock(uint32_t block) const;
	/** @brief Whether any block's memory is allocated now. */
	bool Allocated() const;
	/** @brief Takes block memory from memory (shared with the host's other versions) instead of allocating its own. Before any AcquireBlock. */
	void SetMemory(std::shared_ptr<FramePlacementMemory> memory) { m_memory = std::move(memory); }

private:
	uint64_t m_id = 0;
	uint64_t m_layoutHash = 0;
	std::vector<Slot> m_slots;
	std::vector<Block> m_blocks;
	std::vector<Entry> m_entries;
	std::map<std::pair<uint64_t, uint32_t>, uint32_t> m_entryByKey;  // by key and epoch
	std::unordered_map<uint64_t, uint32_t> m_entryByObject;
	bool m_exactHeaps = false;
	// Block memory, made by AcquireBlock. Without shared memory (SetMemory) the version owns its blocks' memory. With it, the
	// version only remembers the heaps its blocks took, weakly: the placements made in a heap keep it alive, and a version a
	// graph still holds keeps no heap alive by itself. A block whose heap went takes one again at its next AcquireBlock.
	mutable std::mutex m_memoryMutex;
	mutable std::vector<TrackedHandle> m_allocations;
	mutable std::vector<std::shared_ptr<const AliasHeapGeneration>> m_heaps;
	mutable std::vector<std::weak_ptr<const AliasHeapGeneration>> m_sharedHeaps;
	std::shared_ptr<FramePlacementMemory> m_memory;  // SetMemory
};

}  // namespace org
