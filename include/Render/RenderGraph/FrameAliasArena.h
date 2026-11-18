#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "Resources/AliasingPlacement.h"

namespace org {

class PixelBuffer;

/**
 * @brief A frame's positions, first to last, inclusive: the stages of a frame in their fixed order. A resource or a graph's
 * transients that live within a span may share memory with any other whose span does not overlap.
 */
struct AliasSpan {
	uint32_t first = 0, last = 0;
	bool operator==(const AliasSpan&) const = default;
	// No position at all (first after last): a range nothing uses yet. It shares bytes with anything.
	static constexpr AliasSpan None() noexcept { return {UINT32_MAX, 0}; }
	bool Empty() const noexcept { return first > last; }
};

/**
 * @brief One arena of alias memory for a whole frame.
 *
 * The application runs its frame as an ordered list of stages, one graph each. Each user of the arena reserves a range of
 * bytes for a span of those stages:
 * - a graph reserves one range per alias pool for its own transients (RenderGraph::SetFrameAliasRegion). Its planner packs
 *   them inside the range, as it packs a pool.
 * - the host reserves a range for resources that several stages use (GraphHost::AddFrameTransients, ReserveTextures). Their
 *   placement never moves. Their span is derived from the graphs that use them and grows in place (Widen).
 *
 * Ranges whose spans overlap get disjoint bytes. Ranges whose spans do not overlap share bytes. So the frame's memory is set by
 * the stage where the most bytes are live, not by the sum of each stage's own peak.
 *
 * The arena does not order the GPU work. The runtime's alias ledger does that for any placed resource: a resource's first
 * use after another one used the same bytes enters from Undefined, and waits for that use on every queue. So the spans only
 * keep content apart. A resource's content survives within its span because nothing with an overlapping span shares its
 * bytes.
 *
 * Memory comes in blocks. A reservation goes in the first block where its range fits, else in a new block of its size. A
 * block is freed when its last reservation goes. Reservations never move, so a block never grows.
 *
 * Any thread.
 */
class FrameAliasArena {
public:
	/** @brief A reserved range. The range is given back when the last reference goes. */
	class Reservation {
	public:
		~Reservation();
		Reservation(const Reservation&) = delete;
		Reservation& operator=(const Reservation&) = delete;
		const std::shared_ptr<const AliasHeapGeneration>& Heap() const noexcept { return m_heap; }
		uint64_t Offset() const noexcept { return m_offset; }
		uint64_t Size() const noexcept { return m_size; }
		// The block's pool id, for memory statistics and the alias ledger's records.
		uint64_t PoolID() const noexcept { return m_poolID; }
	private:
		friend class FrameAliasArena;
		Reservation() = default;
		std::weak_ptr<FrameAliasArena> m_arena;
		std::shared_ptr<const AliasHeapGeneration> m_heap;
		uint64_t m_offset = 0, m_size = 0, m_poolID = 0, m_id = 0;
	};

	static std::shared_ptr<FrameAliasArena> Create();

	/**
	 * @brief Reserves size bytes for a span. resourceClass keeps memory kinds apart as the planner's pools do (1 buffers, 2 render
	 * target and depth textures, 3 other textures). lane: ranges of one lane (not 0) never run in the same frame, so they may
	 * share bytes whatever their spans; a graph's builds use their epoch's lane. name: for the log. Throws when the memory
	 * cannot be allocated.
	 */
	std::shared_ptr<Reservation> Reserve(uint8_t resourceClass, AliasSpan span, uint32_t lane, uint64_t size, uint64_t alignment,
		std::string name);

	/**
	 * @brief Places unmaterialized textures for a span: one reservation per memory class, the textures one after another inside
	 * it, each materialized at its place. The textures' placements last as long as the returned reservations. Throws on failure.
	 * An empty span (AliasSpan::None) places them where nothing is reserved yet, or over ranges they may later grow beside
	 * (Widen).
	 */
	std::vector<std::shared_ptr<Reservation>> ReserveTextures(AliasSpan span, std::span<const std::shared_ptr<PixelBuffer>> textures,
		std::string name);

	/**
	 * @brief Gives a reservation a wider span in place: its bytes stay. Possible when no other range that shares its bytes has
	 * a span that conflicts with the new one. False otherwise, with the range in the way named in blocker; the span stays.
	 * replacedLane (not 0): the lane of the epoch whose new build asks; its earlier builds' ranges do not block (the new build
	 * replaces them and places its own ranges after this).
	 */
	bool Widen(Reservation& reservation, AliasSpan span, std::string* blocker = nullptr, uint32_t replacedLane = 0);

	struct RangeInfo {
		std::string name;
		uint64_t begin = 0, end = 0;
		AliasSpan span;
		uint32_t lane = 0;
	};
	struct BlockInfo {
		uint64_t poolID = 0, capacity = 0;
		uint8_t resourceClass = 0;
		std::vector<RangeInfo> ranges;
	};
	/** @brief The blocks and their ranges now. */
	std::vector<BlockInfo> Blocks() const;
	/** @brief The blocks' memory now, in bytes. */
	uint64_t Bytes() const;

	FrameAliasArena() = default;
	FrameAliasArena(const FrameAliasArena&) = delete;
	FrameAliasArena& operator=(const FrameAliasArena&) = delete;

private:
	struct Impl;
	void Release(uint64_t id) noexcept;
	std::shared_ptr<Impl> m_impl;
	std::weak_ptr<FrameAliasArena> m_self;
};

}  // namespace org
