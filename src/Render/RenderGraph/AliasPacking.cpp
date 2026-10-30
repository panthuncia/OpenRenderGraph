#include "AliasPacking.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <unordered_set>

namespace org::alias {
namespace {

uint64_t AlignUp(uint64_t value, uint64_t alignment) {
	return alignment <= 1 ? value : (value + alignment - 1) / alignment * alignment;
}

bool LifetimesOverlap(const PackingItem& a, const PackingItem& b) {
	return !(a.lastUse < b.firstUse || b.lastUse < a.firstUse);
}

} // namespace

PackingResult PackGreedySweepLine(std::span<const PackingItem> items) {
	struct Active { uint32_t lastUse; uint64_t start, end; bool operator>(const Active& o) const { return lastUse > o.lastUse; } };
	struct Free { uint64_t start, end; };
	std::priority_queue<Active, std::vector<Active>, std::greater<Active>> active;
	std::vector<Free> freeRanges;
	auto release = [&](Free range) {
		if (range.end <= range.start) return;
		auto it = std::lower_bound(freeRanges.begin(), freeRanges.end(), range.start, [](const Free& f, uint64_t s) { return f.start < s; });
		size_t pos = static_cast<size_t>(it - freeRanges.begin());
		freeRanges.insert(it, range);
		if (pos > 0 && freeRanges[pos - 1].end >= freeRanges[pos].start) {
			freeRanges[pos - 1].end = (std::max)(freeRanges[pos - 1].end, freeRanges[pos].end);
			freeRanges.erase(freeRanges.begin() + static_cast<std::ptrdiff_t>(pos)); --pos;
		}
		while (pos + 1 < freeRanges.size() && freeRanges[pos].end >= freeRanges[pos + 1].start) {
			freeRanges[pos].end = (std::max)(freeRanges[pos].end, freeRanges[pos + 1].end);
			freeRanges.erase(freeRanges.begin() + static_cast<std::ptrdiff_t>(pos + 1));
		}
	};
	PackingResult result;
	result.offsets.resize(items.size());
	for (size_t i = 0; i < items.size(); ++i) {
		const auto& item = items[i];
		while (!active.empty() && active.top().lastUse < item.firstUse) { release({active.top().start, active.top().end}); active.pop(); }
		bool found = false; size_t bestRange = 0; uint64_t bestStart = 0, bestSlack = UINT64_MAX;
		for (size_t r = 0; r < freeRanges.size(); ++r) {
			const auto start = AlignUp(freeRanges[r].start, item.alignment);
			const auto end = start + item.sizeBytes;
			if (start < freeRanges[r].start || end > freeRanges[r].end) continue;
			const auto slack = freeRanges[r].end - end;
			if (!found || slack < bestSlack || (slack == bestSlack && start < bestStart)) { found = true; bestRange = r; bestStart = start; bestSlack = slack; }
		}
		uint64_t start;
		if (found) {
			const auto selected = freeRanges[bestRange];
			start = bestStart;
			const auto end = start + item.sizeBytes;
			if (selected.start < start && end < selected.end) {
				freeRanges[bestRange].end = start;
				freeRanges.insert(freeRanges.begin() + static_cast<std::ptrdiff_t>(bestRange + 1), Free{end, selected.end});
			} else if (selected.start < start) freeRanges[bestRange].end = start;
			else if (end < selected.end) freeRanges[bestRange].start = end;
			else freeRanges.erase(freeRanges.begin() + static_cast<std::ptrdiff_t>(bestRange));
		} else {
			start = AlignUp(result.heapEnd, item.alignment);
		}
		result.heapEnd = (std::max)(result.heapEnd, start + item.sizeBytes);
		result.alignment = (std::max)(result.alignment, item.alignment);
		active.push({item.lastUse, start, start + item.sizeBytes});
		result.offsets[i] = start;
	}
	return result;
}

PackingResult PackBeamSearch(std::span<const PackingItem> items) {
	constexpr size_t kBeamWidth = 24;
	constexpr size_t kStartsPerState = 8;
	struct Placed { size_t item; uint64_t start, end; };
	struct State {
		std::vector<Placed> placed;
		std::vector<uint8_t> placedMask;
		uint64_t heapSize = 0;
		double score = 0.0;
	};
	auto best = PackGreedySweepLine(items);
	// Heaviest (size x lifetime) first: those constrain the layout most.
	std::vector<size_t> order(items.size());
	for (size_t i = 0; i < order.size(); ++i) order[i] = i;
	std::sort(order.begin(), order.end(), [&](size_t l, size_t r) {
		const auto& a = items[l]; const auto& b = items[r];
		const uint64_t aWeight = a.sizeBytes * (uint64_t{a.lastUse} - a.firstUse + 1), bWeight = b.sizeBytes * (uint64_t{b.lastUse} - b.firstUse + 1);
		if (aWeight != bWeight) return aWeight > bWeight;
		if (a.sizeBytes != b.sizeBytes) return a.sizeBytes > b.sizeBytes;
		if (a.firstUse != b.firstUse) return a.firstUse < b.firstUse;
		return a.tieBreak < b.tieBreak;
	});
	auto finish = [&](const State& state) {
		if (state.placed.size() != items.size() || state.heapSize >= best.heapEnd) return;
		best.heapEnd = state.heapSize;
		for (const auto& placed : state.placed) best.offsets[placed.item] = placed.start;
	};
	std::vector<State> beam(1);
	beam.front().placedMask.assign(items.size(), 0);
	for (size_t depth = 0; depth < items.size() && !beam.empty(); ++depth) {
		std::vector<State> next;
		next.reserve(beam.size() * kStartsPerState);
		for (const auto& state : beam) {
			const auto pending = std::ranges::find_if(order, [&](size_t i) { return !state.placedMask[i]; });
			if (pending == order.end()) { finish(state); continue; }
			const auto index = *pending;
			const auto& item = items[index];
			// Candidate starts: the heap start and the end of every placed item whose lifetime overlaps this one.
			std::vector<uint64_t> starts{0};
			for (const auto& placed : state.placed)
				if (LifetimesOverlap(item, items[placed.item])) starts.push_back(placed.end);
			std::vector<std::pair<uint64_t, uint64_t>> feasible; // (start, resulting heap size)
			std::unordered_set<uint64_t> seen;
			for (const auto raw : starts) {
				const auto start = AlignUp(raw, item.alignment);
				if (!seen.insert(start).second) continue;
				const auto end = start + item.sizeBytes;
				const bool conflicts = std::ranges::any_of(state.placed, [&](const Placed& placed) {
					return LifetimesOverlap(item, items[placed.item]) && (std::max)(start, placed.start) < (std::min)(end, placed.end);
				});
				if (conflicts) continue;
				const auto heap = (std::max)(state.heapSize, end);
				if (heap < best.heapEnd) feasible.emplace_back(start, heap);
			}
			if (feasible.empty()) {
				const auto start = AlignUp(state.heapSize, item.alignment);
				if (start + item.sizeBytes < best.heapEnd) feasible.emplace_back(start, start + item.sizeBytes);
			}
			std::sort(feasible.begin(), feasible.end(), [](const auto& a, const auto& b) { return a.second != b.second ? a.second < b.second : a.first < b.first; });
			if (feasible.size() > kStartsPerState) { feasible.resize(kStartsPerState); best.truncated = true; }
			for (const auto& [start, heap] : feasible) {
				State grown = state;
				grown.placedMask[index] = 1;
				grown.heapSize = heap;
				grown.placed.push_back({index, start, start + item.sizeBytes});
				double waste = 0.0;
				for (const auto& placed : grown.placed) waste += static_cast<double>(placed.end - placed.start);
				grown.score = static_cast<double>(heap) + 0.000001 * waste;
				next.push_back(std::move(grown));
			}
		}
		if (next.empty()) break;
		std::sort(next.begin(), next.end(), [](const State& a, const State& b) { return a.score != b.score ? a.score < b.score : a.heapSize < b.heapSize; });
		if (next.size() > kBeamWidth) { next.resize(kBeamWidth); best.truncated = true; }
		beam = std::move(next);
	}
	for (const auto& state : beam) finish(state);
	return best;
}

} // namespace org::alias
