#include "Render/RenderGraph/FramePlacement.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

#include <spdlog/spdlog.h>

#include "AliasPacking.h"
#include "Managers/Singletons/DeletionManager.h"
#include "Managers/Singletons/DeviceManager.h"
#include "Render/RenderGraph/Aliasing/RenderGraphAliasingSubsystem.h"
#include "Resources/MemoryStatisticsComponents.h"

namespace org {

namespace {
// Pool ids of frame placement blocks: apart from the planner's pools and the frame alias arena's blocks.
constexpr uint64_t kFramePlacementPoolBase = 0xF9A0000000000000ull;
constexpr uint64_t kBlockGranularity = 64ull * 1024ull;

struct Hasher {
	uint64_t h = 0xcbf29ce484222325ull;
	void Mix(uint64_t value) noexcept {
		h ^= value;
		h *= 0x100000001b3ull;
		h ^= h >> 29;
	}
};

uint64_t AlignUp(uint64_t value, uint64_t alignment) {
	return alignment <= 1 ? value : (value + alignment - 1) / alignment * alignment;
}
}  // namespace

void PlacementSummary::Finish() {
	std::sort(uses.begin(), uses.end(), [](const PlacementUse& a, const PlacementUse& b) {
		if (a.key != b.key) return a.key < b.key;
		return a.firstRank < b.firstRank;
	});
	// Two candidates of one graph with the same key cannot be told apart by the host. Their ranks merge, and neither is
	// eligible, so both stay on the graph's own path.
	std::vector<PlacementUse> merged;
	merged.reserve(uses.size());
	for (auto& use : uses) {
		if (!merged.empty() && merged.back().key == use.key) {
			auto& kept = merged.back();
			kept.firstRank = (std::min)(kept.firstRank, use.firstRank);
			kept.lastRank = (std::max)(kept.lastRank, use.lastRank);
			kept.eligible = false;
			kept.exclusion = "key not unique in the graph";
			continue;
		}
		merged.push_back(std::move(use));
	}
	uses = std::move(merged);
	Hasher hasher;
	hasher.Mix(epoch);
	hasher.Mix(rankCount);
	hasher.Mix(static_cast<uint64_t>(memory));
	hasher.Mix(uncovered);
	for (const auto& use : uses) {
		hasher.Mix(use.key);
		// A new resource under a key changes the summary, so the host computes again: the version matches uses of several
		// epochs by identity.
		hasher.Mix(use.objectId);
		hasher.Mix(use.firstRank);
		hasher.Mix(use.lastRank);
		hasher.Mix((use.firstUseWrites ? 1u : 0u) | (use.writeGuaranteed ? 2u : 0u) | (use.loopCrossing ? 4u : 0u) | (use.eligible ? 8u : 0u));
		hasher.Mix(use.size);
		hasher.Mix(use.alignment);
		hasher.Mix(use.memoryClass);
		hasher.Mix(use.backendMask);
	}
	hash = hasher.h;
}

const FramePlacementVersion::Use* FramePlacementVersion::Entry::UseIn(uint32_t epoch) const noexcept {
	for (const auto& use : uses)
		if (use.epoch == epoch) return &use;
	return nullptr;
}

std::shared_ptr<FramePlacementVersion> FramePlacementVersion::Compute(const FramePlacementInputs& inputs, uint64_t id) {
	auto version = std::make_shared<FramePlacementVersion>();
	version->m_id = id;

	// The slots: the epochs in frame order, then epochs not in the order. Each slot holds the epoch's ranks, with one free
	// position after it, so the last rank of one epoch and the first of the next never coincide.
	std::map<uint32_t, std::shared_ptr<const PlacementSummary>> byEpoch;
	for (const auto& summary : inputs.summaries)
		if (summary) byEpoch[summary->epoch] = summary;
	std::vector<uint32_t> order;
	for (const auto epoch : inputs.epochOrder)
		if (byEpoch.contains(epoch) && std::ranges::find(order, epoch) == order.end()) order.push_back(epoch);
	for (const auto& [epoch, summary] : byEpoch)
		if (std::ranges::find(order, epoch) == order.end()) order.push_back(epoch);
	uint32_t base = 0;
	std::unordered_map<uint32_t, size_t> slotOf;
	for (const auto epoch : order) {
		const auto& summary = *byEpoch.at(epoch);
		Slot slot;
		slot.epoch = epoch;
		slot.base = base;
		slot.rankCount = summary.rankCount;
		slot.repeated = std::ranges::find(inputs.repeatedEpochs, epoch) != inputs.repeatedEpochs.end();
		slot.memory = summary.memory;
		slotOf.emplace(epoch, version->m_slots.size());
		version->m_slots.push_back(slot);
		base += (std::max)(summary.rankCount, 1u) + 1;
	}

	// One entry per resource, its uses in frame order. Uses of several epochs are one resource when they have one identity
	// (PlacementUse::objectId): resources several epochs' graphs share are the same objects. A key is unique within an epoch
	// only, and two epochs' resources with one name are two resources. A use without an identity is a resource of its epoch.
	struct Pending {
		Entry entry;
		std::string ineligible;
		bool firstWriteGuaranteed = false;
		bool loopCrossing = false;
		size_t firstSlot = 0;
	};
	// (identity, key, epoch): (identity, 0, 0) for a use with an identity, (0, key, epoch) otherwise.
	std::map<std::tuple<uint64_t, uint64_t, uint32_t>, Pending> byKey;
	for (const auto epoch : order) {
		for (const auto& use : byEpoch.at(epoch)->uses) {
			const auto group = use.objectId ? std::tuple(use.objectId, uint64_t{0}, 0u) : std::tuple(uint64_t{0}, use.key, epoch);
			auto [found, inserted] = byKey.try_emplace(group);
			auto& pending = found->second;
			if (inserted) {
				pending.entry.key = use.key;
				pending.entry.objectId = use.objectId;
				pending.firstSlot = slotOf.at(epoch);
				pending.entry.name = use.name;
				pending.entry.size = use.size;
				pending.entry.alignment = (std::max<uint64_t>)(use.alignment, 1);
				pending.entry.memoryClass = use.memoryClass;
				pending.entry.backendMask = use.backendMask;
				pending.firstWriteGuaranteed = use.writeGuaranteed;
			}
			pending.entry.uses.push_back({epoch, use.firstRank, use.lastRank, use.firstUseWrites, use.key});
			pending.loopCrossing = pending.loopCrossing || use.loopCrossing;
			if (!use.eligible && pending.ineligible.empty())
				pending.ineligible = "epoch " + std::to_string(epoch) + ": " + (use.exclusion.empty() ? std::string("not eligible") : use.exclusion);
		}
	}

	// Lifetimes on the frame's timeline, and whether each entry may be placed.
	for (auto& [key, pending] : byKey) {
		auto& entry = pending.entry;
		const auto& first = entry.uses.front();
		const auto& last = entry.uses.back();
		const auto& firstSlot = version->m_slots[slotOf.at(first.epoch)];
		const auto& lastSlot = version->m_slots[slotOf.at(last.epoch)];
		entry.frameFirst = firstSlot.base + first.firstRank;
		entry.frameLast = lastSlot.base + last.lastRank;
		// A resource several epochs use, at an end that is a repeated epoch, crosses that epoch's runs. It spans the whole
		// slot there. A resource whose contents cross runs spans the whole slots at both ends, whether or not the host lists
		// the epoch as repeated: the mark says it repeats.
		entry.acrossRuns = pending.loopCrossing;
		const bool spansSlots = entry.MultiEpoch() || pending.loopCrossing;
		if (spansSlots && (firstSlot.repeated || pending.loopCrossing)) entry.frameFirst = firstSlot.base;
		if (spansSlots && (lastSlot.repeated || pending.loopCrossing))
			entry.frameLast = lastSlot.base + (std::max)(lastSlot.rankCount, 1u) - 1;
		if (!pending.ineligible.empty()) entry.reason = pending.ineligible;
		else if (!first.firstUseWrites)
			entry.reason = "previous-frame read: its first use in frame order (epoch " + std::to_string(first.epoch) + ") reads";
		else if (!pending.firstWriteGuaranteed)
			entry.reason = "its first writer in frame order (epoch " + std::to_string(first.epoch) + ") may not write";
		else entry.placed = true;
	}

	// The entries in an order that does not depend on identities: by their first epoch in frame order, then by key (unique in an
	// epoch). So a rebuild that makes new resources under the same keys gives the same layout.
	std::vector<Pending*> sorted;
	sorted.reserve(byKey.size());
	for (auto& [group, pending] : byKey) sorted.push_back(&pending);
	std::sort(sorted.begin(), sorted.end(), [](const Pending* a, const Pending* b) {
		return std::tuple(a->firstSlot, a->entry.key) < std::tuple(b->firstSlot, b->entry.key);
	});

	// Pack per memory class and backend set, as the planner's pools are kept apart.
	std::map<std::pair<uint64_t, uint8_t>, std::vector<Entry*>> groups;
	for (auto* pending : sorted) {
		if (pending->entry.placed) groups[{pending->entry.backendMask, pending->entry.memoryClass}].push_back(&pending->entry);
	}
	for (auto& [group, members] : groups) {
		std::sort(members.begin(), members.end(), [](const Entry* a, const Entry* b) {
			return std::tuple(a->frameFirst, ~a->size, a->frameLast, a->key) < std::tuple(b->frameFirst, ~b->size, b->frameLast, b->key);
		});
		std::vector<alias::PackingItem> items;
		items.reserve(members.size());
		for (const auto* entry : members) items.push_back({entry->size, entry->alignment, entry->frameFirst, entry->frameLast, entry->key});
		auto packed = inputs.beamSearch ? alias::PackBeamSearch(items) : alias::PackGreedySweepLine(items);
		if (!inputs.beamSearch && members.size() > 1) {
			// As the planner does: placing the longest-lived first among resources that start together can leave the free
			// memory in one range. The smaller result is kept.
			std::vector<size_t> sorted(items.size());
			for (size_t i = 0; i < sorted.size(); ++i) sorted[i] = i;
			std::stable_sort(sorted.begin(), sorted.end(), [&](size_t l, size_t r) {
				const auto& a = items[l];
				const auto& b = items[r];
				if (a.firstUse != b.firstUse) return a.firstUse < b.firstUse;
				if (a.lastUse != b.lastUse) return a.lastUse > b.lastUse;
				return a.sizeBytes > b.sizeBytes;
			});
			std::vector<alias::PackingItem> reordered(items.size());
			for (size_t i = 0; i < sorted.size(); ++i) reordered[i] = items[sorted[i]];
			const auto longestFirst = alias::PackGreedySweepLine(reordered);
			if (longestFirst.heapEnd < packed.heapEnd) {
				packed.heapEnd = longestFirst.heapEnd;
				packed.alignment = longestFirst.alignment;
				for (size_t i = 0; i < sorted.size(); ++i) packed.offsets[sorted[i]] = longestFirst.offsets[i];
			}
		}
		if (packed.heapEnd == 0) continue;
		Block block;
		block.backendMask = group.first;
		block.memoryClass = group.second;
		block.alignment = (std::max)(packed.alignment, kBlockGranularity);
		block.capacity = AlignUp(packed.heapEnd, kBlockGranularity);
		block.poolID = kFramePlacementPoolBase | ((id & 0xFFFFFFFFull) << 8) | (version->m_blocks.size() & 0xFFull);
		block.members = static_cast<uint32_t>(members.size());
		const auto blockIndex = static_cast<uint32_t>(version->m_blocks.size());
		version->m_blocks.push_back(block);
		for (size_t i = 0; i < members.size(); ++i) {
			members[i]->block = blockIndex;
			members[i]->offset = packed.offsets[i];
		}
	}
	// A pack that produced no block (only empty resources) leaves them unplaced.
	for (auto& [key, pending] : byKey)
		if (pending.entry.placed && pending.entry.block == UINT32_MAX) {
			pending.entry.placed = false;
			pending.entry.reason = "no bytes";
		}

	version->m_entries.reserve(sorted.size());
	for (auto* pending : sorted) version->m_entries.push_back(std::move(pending->entry));
	version->Finish();
	return version;
}

void FramePlacementVersion::Finish() {
	m_entryByKey.clear();
	m_entryByObject.clear();
	for (uint32_t index = 0; index < m_entries.size(); ++index) {
		const auto& entry = m_entries[index];
		for (const auto& use : entry.uses) m_entryByKey.emplace(std::pair(use.key, use.epoch), index);
		if (entry.objectId) m_entryByObject.emplace(entry.objectId, index);
	}
	m_allocations.resize(m_blocks.size());
	m_heaps.resize(m_blocks.size());
	m_sharedHeaps.resize(m_blocks.size());

	Hasher hasher;
	for (const auto& slot : m_slots) {
		hasher.Mix(slot.epoch);
		hasher.Mix(slot.base);
		hasher.Mix(slot.rankCount);
		hasher.Mix(slot.repeated ? 1 : 0);
	}
	for (const auto& block : m_blocks) {
		hasher.Mix(block.backendMask);
		hasher.Mix(block.memoryClass);
		hasher.Mix(block.capacity);
		hasher.Mix(block.alignment);
		hasher.Mix(block.lane);
	}
	for (const auto& entry : m_entries) {
		hasher.Mix(entry.key);
		// The identity only where it decides what the version does: a resource several epochs share. A rebuild that makes an
		// epoch's own resources again keeps the layout.
		if (entry.MultiEpoch()) hasher.Mix(entry.objectId);
		hasher.Mix((entry.placed ? 1 : 0) | (entry.acrossRuns ? 2 : 0));
		hasher.Mix(entry.block);
		hasher.Mix(entry.offset);
		hasher.Mix(entry.frameFirst);
		hasher.Mix(entry.frameLast);
		for (const auto& use : entry.uses) {
			hasher.Mix(use.epoch);
			hasher.Mix(use.firstRank);
			hasher.Mix(use.lastRank);
		}
	}
	m_layoutHash = hasher.h;
}

FramePlacementVersion::~FramePlacementVersion() {
	// The resources placed in a block keep its memory alive through their leases until they go.
	for (auto& allocation : m_allocations) {
		if (!allocation.GetAllocation()) continue;
		try {
			DeletionManager::GetInstance().MarkForDelete(std::move(allocation), "RenderGraph frame placement block");
		} catch (...) {
		}
	}
}

std::shared_ptr<FramePlacementVersion> FramePlacementVersion::Keeping(const FramePlacementVersion& fresh, const FramePlacementVersion& base,
	const std::vector<FramePlacementPin>& pins, const std::vector<uint32_t>& keptEpochs, uint64_t id, std::string* why,
	const std::vector<uint32_t>* pinEpochs) {
	const auto fail = [&](std::string reason) -> std::shared_ptr<FramePlacementVersion> {
		if (why) *why = std::move(reason);
		return nullptr;
	};
	auto version = std::make_shared<FramePlacementVersion>();
	version->m_id = id;
	version->m_keptFrom = base.Id();
	version->m_slots = fresh.m_slots;
	version->m_entries = fresh.m_entries;
	// The blocks start as base's, on base's heaps. A block whose heap no placement holds any more takes one again when acquired.
	// A heap may be larger than its block (headroom), and its free bytes past the block are free here too.
	version->m_blocks = base.m_blocks;
	const auto baseBlocks = static_cast<uint32_t>(base.m_blocks.size());
	std::vector<std::shared_ptr<const AliasHeapGeneration>> heaps(baseBlocks);
	std::unordered_map<const AliasHeapGeneration*, uint64_t> heapCapacity;
	if (base.m_memory)
		for (const auto& heap : base.m_memory->Heaps()) heapCapacity[heap.identity] = heap.capacity;
	for (uint32_t b = 0; b < baseBlocks; ++b) {
		auto& block = version->m_blocks[b];
		block.members = 0;
		heaps[b] = base.PeekBlock(b);
		if (!heaps[b]) continue;
		if (const auto found = heapCapacity.find(heaps[b].get()); found != heapCapacity.end())
			block.capacity = (std::max)(block.capacity, found->second / kBlockGranularity * kBlockGranularity);
	}
	std::unordered_map<uint64_t, const FramePlacementPin*> pinOf;
	for (const auto& pin : pins)
		if (pin.objectId) pinOf.emplace(pin.objectId, &pin);
	const auto inKeptEpoch = [&](const Entry& entry) {
		return std::ranges::any_of(entry.uses, [&](const Use& use) { return std::ranges::find(keptEpochs, use.epoch) != keptEpochs.end(); });
	};
	// Per block: the byte ranges and frame spans of what lies in it, and the entry of each (for the reason of a failure).
	std::vector<std::vector<alias::SpannedRange>> occupied(baseBlocks);
	std::vector<std::vector<uint32_t>> occupants(baseBlocks);
	std::vector<uint32_t> free;
	struct Pinned {
		uint32_t entry = 0, block = 0;
		uint64_t offset = 0;
	};
	std::vector<Pinned> pinned;
	for (uint32_t i = 0; i < version->m_entries.size(); ++i) {
		auto& entry = version->m_entries[i];
		const bool placed = entry.placed;
		entry.block = UINT32_MAX;
		entry.offset = 0;
		auto pin = entry.objectId ? pinOf.find(entry.objectId) : pinOf.end();
		if (pin != pinOf.end() && pinEpochs
			&& std::ranges::none_of(entry.uses, [&](const Use& use) { return std::ranges::find(*pinEpochs, use.epoch) != pinEpochs->end(); }))
			pin = pinOf.end();
		if (pin != pinOf.end()) {
			// A running graph binds it where it is: it stays there, with its new lifetime.
			uint32_t block = UINT32_MAX;
			for (uint32_t b = 0; b < baseBlocks && block == UINT32_MAX; ++b)
				if (heaps[b] && heaps[b].get() == pin->second->heap) block = b;
			if (block == UINT32_MAX) return fail("'" + entry.name + "' lies in a heap no block of version " + std::to_string(base.Id()) + " has");
			auto& target = version->m_blocks[block];
			if (target.memoryClass != entry.memoryClass || target.backendMask != entry.backendMask || pin->second->size != entry.size)
				return fail("'" + entry.name + "' changed its memory description");
			if (!placed) {
				// The new layout does not place it (its first use in frame order now reads, say). The graphs keep it where it is, as
				// for any resource a version does not cover that several epochs use, so its bytes are held through the whole frame.
				occupied[block].push_back({pin->second->offset, pin->second->offset + entry.size, 0, UINT32_MAX - 1, 0});
				occupants[block].push_back(i);
				target.capacity = (std::max)(target.capacity, AlignUp(pin->second->offset + entry.size, kBlockGranularity));
				++target.members;
				continue;
			}
			pinned.push_back({i, block, pin->second->offset});
			continue;
		}
		if (!placed) continue;
		if (inKeptEpoch(entry)) {
			// A graph that keeps running uses it but has it elsewhere (its own pools, or where another epoch placed it). It stays
			// there: placing it here would move it under that graph.
			entry.placed = false;
			entry.reason = "kept where a running graph has it";
			continue;
		}
		free.push_back(i);
	}
	// Two pinned resources that share bytes may now be live together: a resource newly shared with a later epoch now lives
	// until that epoch, and the earlier epoch's next resource in its bytes does not wait for it. One of them must move, and the
	// graph that binds it places again. Resources several epochs share are kept first (moving them moves them under every
	// epoch's graph), then the larger.
	std::stable_sort(pinned.begin(), pinned.end(), [&](const Pinned& l, const Pinned& r) {
		const auto& a = version->m_entries[l.entry];
		const auto& b = version->m_entries[r.entry];
		return std::tuple(!a.MultiEpoch(), ~a.size) < std::tuple(!b.MultiEpoch(), ~b.size);
	});
	for (const auto& pin : pinned) {
		auto& entry = version->m_entries[pin.entry];
		const alias::SpannedRange range{pin.offset, pin.offset + entry.size, entry.frameFirst, entry.frameLast, 0};
		uint32_t conflictWith = UINT32_MAX;
		for (size_t o = 0; o < occupied[pin.block].size() && conflictWith == UINT32_MAX; ++o) {
			const auto& other = occupied[pin.block][o];
			if (alias::SpansConflict(range, other) && range.begin < other.end && other.begin < range.end) conflictWith = occupants[pin.block][o];
		}
		if (conflictWith != UINT32_MAX) {
			if (version->m_keptMoved.size() < 4) {
				version->m_keptMoved.push_back(entry.name);
				version->m_keptMovedWith.push_back(version->m_entries[conflictWith].name);
			}
			++version->m_keptMovedCount;
			free.push_back(pin.entry);
			continue;
		}
		auto& target = version->m_blocks[pin.block];
		entry.block = pin.block;
		entry.offset = pin.offset;
		occupied[pin.block].push_back(range);
		occupants[pin.block].push_back(pin.entry);
		target.capacity = (std::max)(target.capacity, AlignUp(range.end, kBlockGranularity));
		++target.members;
	}
	// The rest in the order Compute packs them: by first use, the larger first.
	std::stable_sort(free.begin(), free.end(), [&](uint32_t l, uint32_t r) {
		const auto& a = version->m_entries[l];
		const auto& b = version->m_entries[r];
		return std::tuple(a.frameFirst, ~a.size, a.frameLast, a.key) < std::tuple(b.frameFirst, ~b.size, b.frameLast, b.key);
	});
	for (const auto i : free) {
		auto& entry = version->m_entries[i];
		const auto alignment = (std::max<uint64_t>)(entry.alignment, 1);
		uint32_t block = UINT32_MAX;
		uint64_t offset = UINT64_MAX;
		for (uint32_t b = 0; b < version->m_blocks.size() && block == UINT32_MAX; ++b) {
			const auto& candidate = version->m_blocks[b];
			if (candidate.memoryClass != entry.memoryClass || candidate.backendMask != entry.backendMask) continue;
			// base's blocks keep their heaps, so they do not grow. A block added here has no heap yet and takes its size at the end.
			const uint64_t capacity = b < baseBlocks ? candidate.capacity : UINT64_MAX / 2;
			offset = alias::PlaceSpanned(occupied[b], capacity, entry.size, alignment, entry.frameFirst, entry.frameLast, 0);
			if (offset != UINT64_MAX) block = b;
		}
		if (block == UINT32_MAX) {
			// A block of a new lane for its class and backends: no heap a pin lies in grows.
			Block added;
			added.backendMask = entry.backendMask;
			added.memoryClass = entry.memoryClass;
			added.alignment = kBlockGranularity;
			bool groupKnown = false;
			uint32_t lane = 0;
			for (const auto& existing : version->m_blocks)
				if (existing.memoryClass == entry.memoryClass && existing.backendMask == entry.backendMask) {
					groupKnown = true;
					lane = (std::max)(lane, existing.lane + 1);
				}
			added.lane = groupKnown ? lane : 0;
			added.poolID = kFramePlacementPoolBase | ((id & 0xFFFFFFFFull) << 8) | (version->m_blocks.size() & 0xFFull);
			block = static_cast<uint32_t>(version->m_blocks.size());
			version->m_blocks.push_back(added);
			occupied.emplace_back();
			occupants.emplace_back();
			offset = 0;
		}
		auto& target = version->m_blocks[block];
		if (block >= baseBlocks) {
			target.capacity = (std::max)(target.capacity, AlignUp(offset + entry.size, kBlockGranularity));
			target.alignment = (std::max)(target.alignment, alignment);
		}
		entry.block = block;
		entry.offset = offset;
		occupied[block].push_back({offset, offset + entry.size, entry.frameFirst, entry.frameLast, 0});
		occupants[block].push_back(i);
		++target.members;
	}
	// Blocks nothing lies in any more go, so their heaps are not kept for nothing.
	std::vector<uint32_t> remap(version->m_blocks.size(), UINT32_MAX);
	std::vector<Block> blocks;
	std::vector<std::shared_ptr<const AliasHeapGeneration>> keptHeaps;
	for (uint32_t b = 0; b < version->m_blocks.size(); ++b) {
		if (!version->m_blocks[b].members) continue;
		remap[b] = static_cast<uint32_t>(blocks.size());
		blocks.push_back(version->m_blocks[b]);
		keptHeaps.push_back(b < baseBlocks ? heaps[b] : nullptr);
	}
	version->m_blocks = std::move(blocks);
	for (auto& entry : version->m_entries)
		if (entry.placed) entry.block = remap.at(entry.block);
	version->Finish();
	for (uint32_t b = 0; b < version->m_blocks.size(); ++b) version->m_sharedHeaps[b] = keptHeaps[b];
	return version;
}

std::shared_ptr<FramePlacementVersion> FramePlacementVersion::Relabel(const FramePlacementVersion& from, uint64_t id) {
	auto version = std::make_shared<FramePlacementVersion>();
	version->m_id = id;
	version->m_keptFrom = from.m_keptFrom;
	version->m_slots = from.m_slots;
	version->m_blocks = from.m_blocks;
	version->m_entries = from.m_entries;
	version->Finish();
	return version;
}

const FramePlacementVersion::Entry* FramePlacementVersion::Find(uint64_t key, uint32_t epoch) const noexcept {
	const auto found = m_entryByKey.find(std::pair(key, epoch));
	return found == m_entryByKey.end() ? nullptr : &m_entries[found->second];
}

const FramePlacementVersion::Entry* FramePlacementVersion::FindObject(uint64_t objectId) const noexcept {
	if (!objectId) return nullptr;
	const auto found = m_entryByObject.find(objectId);
	return found == m_entryByObject.end() ? nullptr : &m_entries[found->second];
}

std::vector<FramePlacementBlockNeed> FramePlacementVersion::BlockNeeds() const {
	std::map<std::tuple<uint8_t, uint64_t, uint32_t>, uint64_t> needs;
	for (const auto& block : m_blocks) needs[{block.memoryClass, block.backendMask, block.lane}] += block.capacity;
	std::vector<FramePlacementBlockNeed> out;
	for (const auto& [key, capacity] : needs) out.push_back({std::get<0>(key), std::get<1>(key), std::get<2>(key), capacity});
	return out;
}

bool FramePlacementVersion::HasEpoch(uint32_t epoch) const noexcept {
	return std::ranges::any_of(m_slots, [&](const Slot& slot) { return slot.epoch == epoch; });
}

uint64_t FramePlacementVersion::Bytes() const noexcept {
	uint64_t bytes = 0;
	for (const auto& block : m_blocks) bytes += block.capacity;
	return bytes;
}

uint64_t FramePlacementVersion::BytesOfClass(uint8_t memoryClass) const noexcept {
	uint64_t bytes = 0;
	for (const auto& block : m_blocks)
		if (block.memoryClass == memoryClass) bytes += block.capacity;
	return bytes;
}

std::shared_ptr<const AliasHeapGeneration> FramePlacementVersion::AcquireBlock(uint32_t blockIndex) const {
	if (blockIndex >= m_blocks.size()) throw std::out_of_range("A frame placement has no such block");
	std::lock_guard lock(m_memoryMutex);
	const auto& block = m_blocks[blockIndex];
	if (m_memory) {
		// The heap the block took before, while a placement still holds it. So every graph that places from this version at the
		// same time places in the same heap.
		if (auto heap = m_sharedHeaps[blockIndex].lock()) return heap;
		auto heap = m_memory->Acquire(block.memoryClass, block.backendMask, block.capacity, block.alignment, block.poolID, m_id, m_exactHeaps,
			block.lane);
		m_sharedHeaps[blockIndex] = heap;
		return heap;
	}
	if (m_heaps[blockIndex]) return m_heaps[blockIndex];
	rhi::ma::AllocationDesc allocDesc{};
	allocDesc.heapType = rhi::HeapType::DeviceLocal;
	allocDesc.flags = rhi::ma::AllocationFlagCanAlias;
	allocDesc.extraHeapFlags = alias::AliasClassHeapFlags(block.memoryClass);
	rhi::ResourceAllocationInfo allocInfo{};
	allocInfo.alignment = block.alignment;
	allocInfo.sizeInBytes = block.capacity;
	AllocationTrackDesc trackDesc(0);
	trackDesc.attach
		.Set<MemoryStatisticsComponents::ResourceName>({"RenderGraph Frame Placement"})
		.Set<MemoryStatisticsComponents::ResourceType>({rhi::ResourceType::Unknown})
		.Set<MemoryStatisticsComponents::ResourceUsage>({"RenderGraph alias pools"})
		.Set<MemoryStatisticsComponents::AliasingPool>({block.poolID});
	TrackedHandle allocation;
	if (!rhi::IsOk(DeviceManager::GetInstance().AllocateMemoryTracked(allocDesc, allocInfo, allocation, trackDesc)))
		throw std::runtime_error("Failed to allocate frame placement memory");
	// Frame targets and per-frame working memory: over the OS budget, the costliest to leave in system memory.
	DeviceManager::GetInstance().SetResidencyPriority(allocation, rhi::ResidencyPriorityHigh);
	m_heaps[blockIndex] = AliasHeapGeneration::Capture(allocation, 1);
	m_allocations[blockIndex] = std::move(allocation);
	spdlog::info("Frame placement {}: block {:#x} allocated: class={} capacity={} resources={}", m_id, block.poolID, block.memoryClass,
		block.capacity, block.members);
	return m_heaps[blockIndex];
}

std::shared_ptr<const AliasHeapGeneration> FramePlacementVersion::PeekBlock(uint32_t blockIndex) const {
	if (blockIndex >= m_blocks.size()) return nullptr;
	std::lock_guard lock(m_memoryMutex);
	const auto& block = m_blocks[blockIndex];
	if (m_memory) {
		if (auto heap = m_sharedHeaps[blockIndex].lock()) return heap;
		return m_memory->Peek(block.memoryClass, block.backendMask, block.capacity, block.alignment, m_exactHeaps, block.lane);
	}
	return m_heaps[blockIndex];
}

std::shared_ptr<const AliasHeapGeneration> FramePlacementMemory::Peek(uint8_t memoryClass, uint64_t backendMask, uint64_t capacity,
	uint64_t alignment, bool exact, uint32_t lane) const {
	std::lock_guard lock(m_mutex);
	const auto found = m_heaps.find({memoryClass, backendMask, lane});
	if (found == m_heaps.end() || !found->second.heap) return nullptr;
	const auto& cached = found->second;
	// As Acquire: the cached heap when it fits and is not too big for the need.
	const double headroom = (std::max)(m_sizing.headroom, 1.0);
	const double shrinkFactor = (std::max)(m_sizing.shrinkFactor, 1.0);
	const bool fits = cached.capacity >= capacity && cached.alignment >= alignment;
	const bool tooBig = exact ? cached.capacity > AlignUp(capacity, kBlockGranularity) + TrimSlack(capacity)
							  : static_cast<double>(cached.capacity) > shrinkFactor * static_cast<double>(capacity) * headroom;
	return fits && !tooBig ? cached.heap : nullptr;
}

FramePlacementMemory::~FramePlacementMemory() {
	for (auto& [key, heap] : m_heaps) {
		if (!heap.allocation.GetAllocation()) continue;
		try {
			DeletionManager::GetInstance().MarkForDelete(std::move(heap.allocation), "RenderGraph frame placement block");
		} catch (...) {
		}
	}
}

std::shared_ptr<const AliasHeapGeneration> FramePlacementMemory::Acquire(uint8_t memoryClass, uint64_t backendMask, uint64_t capacity,
	uint64_t alignment, uint64_t poolID, uint64_t versionID, bool exact, uint32_t lane) {
	std::lock_guard lock(m_mutex);
	auto& cached = m_heaps[{memoryClass, backendMask, lane}];
	// Room to grow a little, so small changes between versions reuse the heap.
	const double headroom = (std::max)(m_sizing.headroom, 1.0);
	const double shrinkFactor = (std::max)(m_sizing.shrinkFactor, 1.0);
	const bool fits = cached.heap && cached.capacity >= capacity && cached.alignment >= alignment;
	const bool tooBig = cached.heap
		&& (exact ? cached.capacity > AlignUp(capacity, kBlockGranularity) + TrimSlack(capacity)
				  : static_cast<double>(cached.capacity) > shrinkFactor * static_cast<double>(capacity) * headroom);
	if (fits && !tooBig) {
		cached.needed = capacity;
		cached.neededBy = versionID;
		return cached.heap;
	}
	// A first heap and a trim take the need. A heap that grows takes the larger of the need and headroom times its old capacity,
	// as the planner's pools do, so a burst of small growths reuses it; the trim once the versions settle gives the rest back.
	const bool grow = cached.heap && !fits;
	const double grown = grow && !exact ? static_cast<double>(cached.capacity) * headroom : 0.0;
	const auto size = static_cast<uint64_t>((std::max)(static_cast<double>(capacity), grown) + 65535.0) / 65536ull * 65536ull;
	rhi::ma::AllocationDesc allocDesc{};
	allocDesc.heapType = rhi::HeapType::DeviceLocal;
	allocDesc.flags = rhi::ma::AllocationFlagCanAlias;
	allocDesc.extraHeapFlags = alias::AliasClassHeapFlags(memoryClass);
	rhi::ResourceAllocationInfo allocInfo{};
	allocInfo.alignment = (std::max)(alignment, cached.alignment);
	allocInfo.sizeInBytes = size;
	AllocationTrackDesc trackDesc(0);
	trackDesc.attach
		.Set<MemoryStatisticsComponents::ResourceName>({"RenderGraph Frame Placement"})
		.Set<MemoryStatisticsComponents::ResourceType>({rhi::ResourceType::Unknown})
		.Set<MemoryStatisticsComponents::ResourceUsage>({"RenderGraph alias pools"})
		.Set<MemoryStatisticsComponents::AliasingPool>({poolID});
	TrackedHandle allocation;
	const auto allocationBegin = std::chrono::steady_clock::now();
	if (!rhi::IsOk(DeviceManager::GetInstance().AllocateMemoryTracked(allocDesc, allocInfo, allocation, trackDesc)))
		throw std::runtime_error("Failed to allocate frame placement memory");
	// Frame targets and per-frame working memory: over the OS budget, the costliest to leave in system memory.
	DeviceManager::GetInstance().SetResidencyPriority(allocation, rhi::ResidencyPriorityHigh);
	const double allocationMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - allocationBegin).count();
	// The placements in the replaced heap keep it alive through their leases until they go. Heaps lists it while it lives.
	if (cached.heap) m_replaced.push_back({memoryClass, backendMask, cached.capacity, cached.generation, lane, cached.heap});
	if (cached.allocation.GetAllocation())
		DeletionManager::GetInstance().MarkForDelete(std::move(cached.allocation), "RenderGraph frame placement block (replaced)");
	cached.allocation = std::move(allocation);
	cached.capacity = size;
	cached.alignment = allocInfo.alignment;
	cached.needed = capacity;
	cached.neededBy = versionID;
	++cached.generation;
	cached.heap = AliasHeapGeneration::Capture(cached.allocation, cached.generation);
	spdlog::info("Frame placement {}: heap for class {}{} {}: capacity={} (needed {}) generation={} in {:.2f} ms", versionID, memoryClass,
		lane ? " lane " + std::to_string(lane) : std::string(), fits ? (exact ? "trimmed" : "shrank") : cached.generation == 1 ? "allocated" : "grew",
		size, capacity, cached.generation, allocationMs);
	return cached.heap;
}

namespace {
bool NeedMatches(const FramePlacementBlockNeed& need, const std::tuple<uint8_t, uint64_t, uint32_t>& key) {
	return need.memoryClass == std::get<0>(key) && need.backendMask == std::get<1>(key) && need.lane == std::get<2>(key);
}
}  // namespace

bool FramePlacementMemory::WouldTrim(const std::vector<FramePlacementBlockNeed>& needs) const {
	std::lock_guard lock(m_mutex);
	for (const auto& [key, heap] : m_heaps) {
		if (!heap.heap) continue;
		const auto found = std::ranges::find_if(needs, [&](const auto& need) { return NeedMatches(need, key); });
		if (found == needs.end()) return true;
		// A heap smaller than the need grows when a graph places into it: nothing to trim.
		if (heap.capacity > AlignUp(found->capacity, kBlockGranularity) + TrimSlack(found->capacity)) return true;
	}
	return false;
}

void FramePlacementMemory::ReleaseUnused(const std::vector<FramePlacementBlockNeed>& needs) {
	std::lock_guard lock(m_mutex);
	for (auto it = m_heaps.begin(); it != m_heaps.end();) {
		const auto& key = it->first;
		if (std::ranges::any_of(needs, [&](const auto& need) { return NeedMatches(need, key); })) {
			++it;
			continue;
		}
		auto& heap = it->second;
		if (heap.heap) {
			spdlog::info("Frame placement: heap for class {} lane {} released: capacity={} generation={} (no block uses it)", std::get<0>(key),
				std::get<2>(key), heap.capacity, heap.generation);
			m_replaced.push_back({std::get<0>(key), std::get<1>(key), heap.capacity, heap.generation, std::get<2>(key), heap.heap});
		}
		if (heap.allocation.GetAllocation())
			DeletionManager::GetInstance().MarkForDelete(std::move(heap.allocation), "RenderGraph frame placement block (unused)");
		it = m_heaps.erase(it);
	}
}

uint64_t FramePlacementMemory::Bytes() const {
	std::lock_guard lock(m_mutex);
	uint64_t bytes = 0;
	for (const auto& [key, heap] : m_heaps) bytes += heap.capacity;
	return bytes;
}

std::vector<FramePlacementMemory::HeapInfo> FramePlacementMemory::Heaps() const {
	std::lock_guard lock(m_mutex);
	std::vector<HeapInfo> out;
	for (const auto& [key, heap] : m_heaps) {
		if (!heap.heap) continue;
		HeapInfo info;
		info.memoryClass = std::get<0>(key);
		info.backendMask = std::get<1>(key);
		info.lane = std::get<2>(key);
		info.capacity = heap.capacity;
		info.generation = heap.generation;
		info.needed = heap.needed;
		info.neededBy = heap.neededBy;
		info.current = true;
		info.references = heap.heap.use_count();
		info.identity = heap.heap.get();
		out.push_back(info);
	}
	std::erase_if(m_replaced, [](const Replaced& replaced) { return replaced.heap.expired(); });
	for (const auto& replaced : m_replaced) {
		const auto heap = replaced.heap.lock();
		if (!heap) continue;
		HeapInfo info;
		info.memoryClass = replaced.memoryClass;
		info.backendMask = replaced.backendMask;
		info.lane = replaced.lane;
		info.capacity = replaced.capacity;
		info.generation = replaced.generation;
		// Less the reference taken here.
		info.references = heap.use_count() - 1;
		info.identity = heap.get();
		out.push_back(info);
	}
	return out;
}

void FramePlacementMemory::SetSizing(Sizing sizing) {
	std::lock_guard lock(m_mutex);
	m_sizing = sizing;
}

bool FramePlacementVersion::Allocated() const {
	std::lock_guard lock(m_memoryMutex);
	return std::ranges::any_of(m_heaps, [](const auto& heap) { return static_cast<bool>(heap); })
		|| std::ranges::any_of(m_sharedHeaps, [](const auto& heap) { return !heap.expired(); });
}

}  // namespace org
