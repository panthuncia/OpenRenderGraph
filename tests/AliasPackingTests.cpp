#include "../src/Render/RenderGraph/AliasPacking.h"
#include "TestSupport.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>


using namespace org::alias;

namespace {
// Every pair of items whose lifetimes overlap occupies disjoint, aligned bytes inside the heap.
void CheckValid(const std::vector<PackingItem>& items, const PackingResult& result) {
	ENSURE(result.offsets.size() == items.size());
	for (size_t i = 0; i < items.size(); ++i) {
		ENSURE(result.offsets[i] % items[i].alignment == 0);
		ENSURE(result.offsets[i] + items[i].sizeBytes <= result.heapEnd);
		for (size_t j = i + 1; j < items.size(); ++j) {
			const bool lifetimes = !(items[i].lastUse < items[j].firstUse || items[j].lastUse < items[i].firstUse);
			const bool bytes = result.offsets[i] < result.offsets[j] + items[j].sizeBytes && result.offsets[j] < result.offsets[i] + items[i].sizeBytes;
			ENSURE(!(lifetimes && bytes));
		}
	}
}
} // namespace

int main() {
	// Disjoint lifetimes share one range; overlapping ones stack.
	{
		const std::vector<PackingItem> items{{256, 64, 0, 1, 1}, {256, 64, 2, 3, 2}, {128, 64, 1, 2, 3}};
		const auto greedy = PackGreedySweepLine(items);
		CheckValid(items, greedy);
		ENSURE(greedy.heapEnd == 384);
	}
	// Random pools: both strategies are valid, and the beam search is never worse than the greedy sweep it starts from.
	std::mt19937 rng(7);
	for (int run = 0; run < 200; ++run) {
		std::vector<PackingItem> items(1 + rng() % 24);
		for (size_t i = 0; i < items.size(); ++i) {
			const uint32_t first = rng() % 40;
			items[i] = {uint64_t{64} * (1 + rng() % 64), uint64_t{64} << (rng() % 3), first, first + rng() % 12, i};
		}
		std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
			return a.firstUse != b.firstUse ? a.firstUse < b.firstUse : a.sizeBytes > b.sizeBytes;
		});
		const auto greedy = PackGreedySweepLine(items);
		const auto beam = PackBeamSearch(items);
		CheckValid(items, greedy);
		CheckValid(items, beam);
		ENSURE(beam.heapEnd <= greedy.heapEnd);
	}
	// Frame arena ranges: spans that do not overlap share bytes; overlapping spans stack; one lane shares whatever its span.
	{
		const std::vector<SpannedRange> occupied{{0, 400, 3, 9, 0}, {0, 200, 10, 10, 7}};
		ENSURE(PlaceSpanned(occupied, 1024, 300, 64, 11, 12, 0) == 0);      // after both spans
		ENSURE(PlaceSpanned(occupied, 1024, 100, 64, 6, 6, 0) == 448);      // inside the first span: above it, aligned
		ENSURE(PlaceSpanned(occupied, 1024, 300, 64, 10, 10, 7) == 0);      // the same lane as the second range
		ENSURE(PlaceSpanned(occupied, 1024, 300, 64, 10, 10, 8) == 256);    // another lane at the second range's position
		ENSURE(PlaceSpanned(occupied, 512, 300, 64, 5, 10, 0) == UINT64_MAX); // conflicts with both, and 400 + 300 > 512
	}
	// Random blocks: every placement keeps conflicting ranges apart.
	for (int run = 0; run < 200; ++run) {
		std::vector<SpannedRange> placed;
		for (int i = 0; i < 16; ++i) {
			const uint32_t first = rng() % 12;
			const uint32_t last = first + rng() % 4;
			const uint32_t lane = rng() % 3;
			const uint64_t size = uint64_t{64} * (1 + rng() % 32);
			const auto offset = PlaceSpanned(placed, 1u << 20, size, 256, first, last, lane);
			ENSURE(offset != UINT64_MAX && offset % 256 == 0);
			const SpannedRange range{offset, offset + size, first, last, lane};
			for (const auto& other : placed)
				ENSURE(!(SpansConflict(range, other) && range.begin < other.end && other.begin < range.end));
			placed.push_back(range);
		}
	}
	// Widening in place: a range may take a wider span when no range that shares its bytes conflicts with it then.
	{
		const std::vector<SpannedRange> block{{0, 400, 4, 6, 0}, {0, 300, 11, 11, 0}, {0, 100, 10, 10, 7}, {400, 600, 5, 5, 0}};
		ENSURE(WidenBlocker(block, 0, 3, 9) == SIZE_MAX);   // the others' bytes there are at 10, 11, or elsewhere
		ENSURE(WidenBlocker(block, 0, 3, 10) == 2);         // the post stage's range at 10 shares its bytes
		ENSURE(WidenBlocker(block, 1, 0, 11) == 0);         // the first range holds those bytes at 4 to 6
		ENSURE(WidenBlocker(block, 3, 0, 11) == SIZE_MAX);  // bytes nobody else holds
		ENSURE(WidenBlocker(block, 0, 3, 10, 7) == SIZE_MAX);  // the range at 10 is an earlier build of the epoch that asks
	}
	// Derived spans: a group spans its users' positions, first to last. It starts empty and only grows.
	{
		FrameSpans spans(11);
		const auto gbuffer = spans.AddGroup("G-buffer", {101, 102});
		const auto scratch = spans.AddGroup("scratch", {201});
		ENSURE(spans.SpanOf(gbuffer).Empty());
		const uint64_t preMain[] = {101, 999};
		auto grown = spans.NoteUses(4, preMain);
		ENSURE(grown.size() == 1 && grown[0].group == gbuffer && grown[0].from.Empty() && grown[0].to == (FrameSpans::Span{4, 4}));
		const uint64_t deferred[] = {102};
		grown = spans.NoteUses(6, deferred);
		ENSURE(grown.size() == 1 && grown[0].to == (FrameSpans::Span{4, 6}));
		ENSURE(spans.NoteUses(5, deferred).empty());  // inside the span: nothing grows
		const uint64_t depth[] = {101, 102};
		grown = spans.NoteUses(3, depth);
		ENSURE(grown.size() == 1 && grown[0].to == (FrameSpans::Span{3, 6}));  // one growth for the group, however many members
		const uint64_t ui[] = {201};
		grown = spans.NoteUses(11, ui);
		ENSURE(grown.size() == 1 && grown[0].group == scratch && spans.SpanOf(scratch) == (FrameSpans::Span{11, 11}));
		// A graph with no position spans the whole frame.
		const uint64_t unpositioned[] = {201};
		grown = spans.NoteUses(FrameSpans::kNoPosition, unpositioned);
		ENSURE(grown.size() == 1 && spans.SpanOf(scratch) == (FrameSpans::Span{0, 11}));
		// A caller that cannot place the new span sets the old one back.
		spans.SetSpan(scratch, grown[0].from);
		ENSURE(spans.SpanOf(scratch) == (FrameSpans::Span{11, 11}));
		ENSURE(spans.GroupOf(102) == gbuffer && spans.GroupOf(999) == 0);
		// The textures made again under the same label start where the last ones reached.
		spans.RemoveGroup(gbuffer);
		ENSURE(spans.GroupOf(101) == 0);
		const auto again = spans.AddGroup("G-buffer", {301});
		ENSURE(spans.SpanOf(again) == (FrameSpans::Span{3, 6}));
		ENSURE(spans.Groups().size() == 2);
	}
	std::puts("Alias packing tests passed.");
}
