#pragma once
// Packs one alias pool. It assigns byte offsets so that resources whose lifetimes do not overlap can share memory.
// A lifetime is an inclusive range of passes. The packing is pure. The persistent alias planner calls it once per pool.
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
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

// Whether range `index` of a block may take the span first..last in place: no other range of the block that shares its
// bytes conflicts with it then. Returns the index of the first range in the way, or SIZE_MAX when none is.
// replacedLane (not 0): ranges of that lane do not block. They are the earlier builds of the epoch whose new build asks for
// the wider span; the new build replaces them and places its own ranges after this.
size_t WidenBlocker(std::span<const SpannedRange> ranges, size_t index, uint32_t first, uint32_t last, uint32_t replacedLane = 0);

// The spans of a frame's shared transients, derived from the graphs that declare them (GraphHost's frame transients).
//
// A frame transient is a resource several graphs of a frame use, placed once in a frame alias arena. Each graph has a frame
// position (its stage's place in the frame). A group's span is the hull of the positions of every graph that declared one of
// its members: its first user to its last. It starts empty (no user yet: it shares bytes with anything) and only grows, so
// a placement never has to move. A graph with no position spans the whole frame.
//
// A group placed again under the same label (a resize makes the textures again) starts with the span its predecessor
// reached: the users are the same graphs.
//
// Pure, not thread-safe: the caller holds a lock.
class FrameSpans {
public:
	static constexpr uint32_t kNoPosition = UINT32_MAX;
	struct Span {
		uint32_t first = UINT32_MAX, last = 0;
		bool Empty() const noexcept { return first > last; }
		bool operator==(const Span&) const = default;
	};
	// The hull of a span and one position.
	static Span Hull(Span span, uint32_t position) noexcept;

	// lastPosition: the frame's last position, the span of a graph with none.
	explicit FrameSpans(uint32_t lastPosition = 0) : m_lastPosition(lastPosition) {}
	void SetLastPosition(uint32_t lastPosition) noexcept { m_lastPosition = lastPosition; }

	// A group of resources (by id) placed together. Its span starts at the one its label reached last (Remembered).
	uint64_t AddGroup(std::string label, std::vector<uint64_t> members);
	void RemoveGroup(uint64_t group);

	struct Growth {
		uint64_t group = 0;
		Span from, to;
	};
	// The resources one graph declares, at its position (kNoPosition: none). Returns the groups whose span grew, with the old
	// and the new span; the new one is already set. A caller that cannot place a group for its new span sets the old one back.
	std::vector<Growth> NoteUses(uint32_t position, std::span<const uint64_t> resources);
	void SetSpan(uint64_t group, Span span);

	Span SpanOf(uint64_t group) const;
	const std::string& LabelOf(uint64_t group) const;
	// The group a resource is a member of; 0 for none.
	uint64_t GroupOf(uint64_t resource) const;
	// The span the last group with this label reached (empty for a new label).
	Span Remembered(const std::string& label) const;
	std::vector<uint64_t> Groups() const;

private:
	struct Group {
		std::string label;
		std::vector<uint64_t> members;
		Span span;
	};
	uint32_t m_lastPosition = 0;
	uint64_t m_next = 1;
	std::unordered_map<uint64_t, Group> m_groups;
	std::unordered_map<uint64_t, uint64_t> m_groupOf;    // member resource id -> group
	std::unordered_map<std::string, Span> m_remembered;  // label -> the span its last group reached
};

} // namespace org::alias
