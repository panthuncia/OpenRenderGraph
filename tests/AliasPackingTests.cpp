#include "../src/Render/RenderGraph/AliasPacking.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>

#define CHECK(...) do { if (!(__VA_ARGS__)) { std::fprintf(stderr, "Check failed at %d: %s\n", __LINE__, #__VA_ARGS__); std::abort(); } } while (false)

using namespace org::alias;

namespace {
// Every pair of items whose lifetimes overlap occupies disjoint, aligned bytes inside the heap.
void CheckValid(const std::vector<PackingItem>& items, const PackingResult& result) {
	CHECK(result.offsets.size() == items.size());
	for (size_t i = 0; i < items.size(); ++i) {
		CHECK(result.offsets[i] % items[i].alignment == 0);
		CHECK(result.offsets[i] + items[i].sizeBytes <= result.heapEnd);
		for (size_t j = i + 1; j < items.size(); ++j) {
			const bool lifetimes = !(items[i].lastUse < items[j].firstUse || items[j].lastUse < items[i].firstUse);
			const bool bytes = result.offsets[i] < result.offsets[j] + items[j].sizeBytes && result.offsets[j] < result.offsets[i] + items[i].sizeBytes;
			CHECK(!(lifetimes && bytes));
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
		CHECK(greedy.heapEnd == 384);
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
		CHECK(beam.heapEnd <= greedy.heapEnd);
	}
	std::puts("Alias packing tests passed.");
}
