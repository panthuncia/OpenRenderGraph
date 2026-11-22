#include "Render/RenderGraph/FramePlacement.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

#include <spdlog/spdlog.h>

#include "AliasPacking.h"
#include "Managers/Singletons/DeletionManager.h"
#include "Managers/Singletons/DeviceManager.h"
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
	for (const auto& use : uses) {
		hasher.Mix(use.key);
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

	// One entry per key, its uses in frame order. A key is the resource's name and memory description, so a key seen in two
	// epochs is the same resource, or two resources that are never live together.
	struct Pending {
		Entry entry;
		std::string ineligible;
		bool firstWriteGuaranteed = false;
		bool loopCrossing = false;
	};
	std::map<uint64_t, Pending> byKey;
	for (const auto epoch : order) {
		for (const auto& use : byEpoch.at(epoch)->uses) {
			auto [found, inserted] = byKey.try_emplace(use.key);
			auto& pending = found->second;
			if (inserted) {
				pending.entry.key = use.key;
				pending.entry.name = use.name;
				pending.entry.size = use.size;
				pending.entry.alignment = (std::max<uint64_t>)(use.alignment, 1);
				pending.entry.memoryClass = use.memoryClass;
				pending.entry.backendMask = use.backendMask;
				pending.firstWriteGuaranteed = use.writeGuaranteed;
			}
			pending.entry.uses.push_back({epoch, use.firstRank, use.lastRank, use.firstUseWrites});
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

	// Pack per memory class and backend set, as the planner's pools are kept apart.
	std::map<std::pair<uint64_t, uint8_t>, std::vector<Entry*>> groups;
	for (auto& [key, pending] : byKey) {
		if (pending.entry.placed) groups[{pending.entry.backendMask, pending.entry.memoryClass}].push_back(&pending.entry);
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

	version->m_entries.reserve(byKey.size());
	for (auto& [key, pending] : byKey) {
		version->m_entryByKey.emplace(key, static_cast<uint32_t>(version->m_entries.size()));
		version->m_entries.push_back(std::move(pending.entry));
	}
	version->m_allocations.resize(version->m_blocks.size());
	version->m_heaps.resize(version->m_blocks.size());

	Hasher hasher;
	for (const auto& slot : version->m_slots) {
		hasher.Mix(slot.epoch);
		hasher.Mix(slot.base);
		hasher.Mix(slot.rankCount);
		hasher.Mix(slot.repeated ? 1 : 0);
	}
	for (const auto& block : version->m_blocks) {
		hasher.Mix(block.backendMask);
		hasher.Mix(block.memoryClass);
		hasher.Mix(block.capacity);
		hasher.Mix(block.alignment);
	}
	for (const auto& entry : version->m_entries) {
		hasher.Mix(entry.key);
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
	version->m_layoutHash = hasher.h;
	return version;
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

const FramePlacementVersion::Entry* FramePlacementVersion::Find(uint64_t key) const noexcept {
	const auto found = m_entryByKey.find(key);
	return found == m_entryByKey.end() ? nullptr : &m_entries[found->second];
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
	if (m_heaps[blockIndex]) return m_heaps[blockIndex];
	const auto& block = m_blocks[blockIndex];
	if (m_memory) {
		m_heaps[blockIndex] = m_memory->Acquire(block.memoryClass, block.backendMask, block.capacity, block.alignment, block.poolID, m_id);
		return m_heaps[blockIndex];
	}
	rhi::ma::AllocationDesc allocDesc{};
	allocDesc.heapType = rhi::HeapType::DeviceLocal;
	allocDesc.flags = rhi::ma::AllocationFlagCanAlias;
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
	uint64_t alignment, uint64_t poolID, uint64_t versionID) {
	std::lock_guard lock(m_mutex);
	auto& cached = m_heaps[{memoryClass, backendMask}];
	// Room to grow a little, so small changes between versions reuse the heap.
	constexpr double kHeadroom = 1.25;
	const bool fits = cached.heap && cached.capacity >= capacity && cached.alignment >= alignment;
	const bool tooBig = cached.heap && cached.capacity > 2 * static_cast<uint64_t>(static_cast<double>(capacity) * kHeadroom);
	if (fits && !tooBig) return cached.heap;
	const auto size = static_cast<uint64_t>(static_cast<double>(capacity) * kHeadroom + 65535.0) / 65536ull * 65536ull;
	rhi::ma::AllocationDesc allocDesc{};
	allocDesc.heapType = rhi::HeapType::DeviceLocal;
	allocDesc.flags = rhi::ma::AllocationFlagCanAlias;
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
	if (!rhi::IsOk(DeviceManager::GetInstance().AllocateMemoryTracked(allocDesc, allocInfo, allocation, trackDesc)))
		throw std::runtime_error("Failed to allocate frame placement memory");
	// Frame targets and per-frame working memory: over the OS budget, the costliest to leave in system memory.
	DeviceManager::GetInstance().SetResidencyPriority(allocation, rhi::ResidencyPriorityHigh);
	// The placements in the replaced heap keep it alive through their leases until they go.
	if (cached.allocation.GetAllocation())
		DeletionManager::GetInstance().MarkForDelete(std::move(cached.allocation), "RenderGraph frame placement block (replaced)");
	cached.allocation = std::move(allocation);
	cached.capacity = size;
	cached.alignment = allocInfo.alignment;
	++cached.generation;
	cached.heap = AliasHeapGeneration::Capture(cached.allocation, cached.generation);
	spdlog::info("Frame placement {}: heap for class {} {}: capacity={} (needed {}) generation={}", versionID, memoryClass,
		fits ? "shrank" : cached.generation == 1 ? "allocated" : "grew", size, capacity, cached.generation);
	return cached.heap;
}

uint64_t FramePlacementMemory::Bytes() const {
	std::lock_guard lock(m_mutex);
	uint64_t bytes = 0;
	for (const auto& [key, heap] : m_heaps) bytes += heap.capacity;
	return bytes;
}

bool FramePlacementVersion::Allocated() const {
	std::lock_guard lock(m_memoryMutex);
	return std::ranges::any_of(m_heaps, [](const auto& heap) { return static_cast<bool>(heap); });
}

}  // namespace org
