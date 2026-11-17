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

// A range of a memory block that a frame alias arena gave out (FrameAliasArena). Its span is a range of frame positions
// (the stages of a frame, in order), inclusive. Two ranges may share bytes only when their spans do not overlap.
// A lane other than 0 marks ranges that never run in the same frame: the builds of one epoch. Ranges of the same lane may
// share bytes whatever their spans.
struct SpannedRange {
	uint64_t begin = 0, end = 0;
	uint32_t first = 0, last = 0;
	uint32_t lane = 0;
};

// Whether two ranges may not share bytes: their spans overlap and they are not in the same lane.
bool SpansConflict(const SpannedRange& a, const SpannedRange& b);

// The lowest aligned offset in a block of capacity bytes where a range of size bytes with this span and lane overlaps no
// conflicting range. occupied: the block's ranges, in any order. Returns UINT64_MAX when nothing fits.
uint64_t PlaceSpanned(std::span<const SpannedRange> occupied, uint64_t capacity, uint64_t size, uint64_t alignment,
	uint32_t first, uint32_t last, uint32_t lane);

} // namespace org::alias
