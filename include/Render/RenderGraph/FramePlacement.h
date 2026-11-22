#pragma once

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
	// Stable across builds: a hash of the resource's name and its memory description (size, alignment, memory class,
	// backends). Two graphs that use one resource give it the same key.
	uint64_t key = 0;
	std::string name;
	// Pass ranks within the epoch's schedule, inclusive.
	uint32_t firstRank = 0, lastRank = 0;
	// False: the first use in this epoch reads. In the frame's first epoch that uses it, that is a previous-frame read.
	bool firstUseWrites = false;
	// False when the pass of the first use may skip its writes (ExternalPassDesc::mayNotWrite).
	bool writeGuaranteed = false;
	// Used in more than one run of a repeated epoch. Graphs cannot see runs, so they leave it false. The host spans the
	// whole slot of a repeated epoch for a resource several epochs use.
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
	// One per candidate, sorted by key. Each key appears once.
	std::vector<PlacementUse> uses;
	// What this graph's own pools needed in its plan, by memory class (index 1 to 3), and in total.
	std::array<uint64_t, 4> pooledBytesByClass{};
	uint64_t pooledBytes = 0;
	// The bytes this graph placed from a frame placement (FramePlacementVersion) in its plan.
	uint64_t framePlacedBytes = 0;
	// A hash of the epoch, the ranks and the uses. Two summaries with the same hash lead to the same frame placement.
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
 * any placed resource. A heap too small, or more than twice too big, is replaced. The replaced memory stays alive through the
 * placements made in it.
 *
 * Any thread.
 */
class FramePlacementMemory {
public:
	FramePlacementMemory() = default;
	~FramePlacementMemory();
	FramePlacementMemory(const FramePlacementMemory&) = delete;
	FramePlacementMemory& operator=(const FramePlacementMemory&) = delete;
	/** @brief A heap of at least capacity bytes. poolID and versionID are for memory statistics and the log. Throws on failure. */
	std::shared_ptr<const AliasHeapGeneration> Acquire(uint8_t memoryClass, uint64_t backendMask, uint64_t capacity, uint64_t alignment,
		uint64_t poolID, uint64_t versionID);
	/** @brief The heaps' capacities now, in bytes. */
	uint64_t Bytes() const;

private:
	struct Heap {
		uint64_t capacity = 0, alignment = 1, generation = 0;
		TrackedHandle allocation;
		std::shared_ptr<const AliasHeapGeneration> heap;
	};
	mutable std::mutex m_mutex;
	std::map<std::pair<uint8_t, uint64_t>, Heap> m_heaps;  // by memory class and backend mask
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
	};
	/** @brief One resource key and where it goes. */
	struct Entry {
		uint64_t key = 0;
		std::string name;
		uint64_t size = 0, alignment = 1;
		uint8_t memoryClass = 0;
		uint64_t backendMask = 0;
		std::vector<Use> uses;  // in frame order
		uint32_t frameFirst = 0, frameLast = 0;  // its lifetime on the frame's timeline, inclusive
		bool placed = false;
		uint32_t block = UINT32_MAX;
		uint64_t offset = 0;
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
	/** @brief The entry for a key, or null. */
	const Entry* Find(uint64_t key) const noexcept;
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
	/** @brief Whether any block's memory was allocated. */
	bool Allocated() const;
	/** @brief Takes block memory from memory (shared with the host's other versions) instead of allocating its own. Before any AcquireBlock. */
	void SetMemory(std::shared_ptr<FramePlacementMemory> memory) { m_memory = std::move(memory); }

private:
	uint64_t m_id = 0;
	uint64_t m_layoutHash = 0;
	std::vector<Slot> m_slots;
	std::vector<Block> m_blocks;
	std::vector<Entry> m_entries;
	std::unordered_map<uint64_t, uint32_t> m_entryByKey;
	// Block memory, made by AcquireBlock.
	mutable std::mutex m_memoryMutex;
	mutable std::vector<TrackedHandle> m_allocations;
	mutable std::vector<std::shared_ptr<const AliasHeapGeneration>> m_heaps;
	std::shared_ptr<FramePlacementMemory> m_memory;  // SetMemory
};

}  // namespace org
