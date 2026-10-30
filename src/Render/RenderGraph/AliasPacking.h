#pragma once
// Packing one alias pool: byte offsets for resources whose lifetimes (inclusive pass ranges) may share memory when they
// do not overlap. Pure; the persistent alias planner calls it per pool.
#include <cstdint>
#include <span>
#include <vector>

namespace org::alias {

struct PackingItem {
	uint64_t sizeBytes = 0, alignment = 1;
	uint32_t firstUse = 0, lastUse = 0;
	uint64_t tieBreak = 0; // Stable order between otherwise equal items (a resource ID).
};

struct PackingResult {
	std::vector<uint64_t> offsets; // One per item, in item order.
	uint64_t heapEnd = 0, alignment = 1;
	bool truncated = false; // The search was cut short; the result is the best found, not proven minimal.
};

// Sweep in item order (callers sort by first use), placing each item in the smallest free range that fits.
PackingResult PackGreedySweepLine(std::span<const PackingItem> items);

// Beam search over placements (heaviest size x lifetime first), seeded and bounded by the greedy result: never larger.
PackingResult PackBeamSearch(std::span<const PackingItem> items);

} // namespace org::alias
