#pragma once
// Packs one alias pool. It assigns byte offsets so that resources whose lifetimes do not overlap can share memory.
// A lifetime is an inclusive range of passes. The packing is pure. The persistent alias planner calls it once per pool.
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

// Sweeps the items in order. Callers sort them by first use. Each item goes in the smallest free range that fits it.
PackingResult PackGreedySweepLine(std::span<const PackingItem> items);

// Beam search over placements, heaviest items first by size times lifetime. The greedy result seeds and bounds the
// search, so the result is never larger than the greedy one.
PackingResult PackBeamSearch(std::span<const PackingItem> items);

} // namespace org::alias
