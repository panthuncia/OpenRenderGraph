#pragma once

// Latched copies. A host writes per-run values into a latch block (LatchBlock). A copy pass at the start of the work then
// copies them from the run's region to the buffers where the other passes read them. So the recorded commands depend only
// on where each value goes, not on the values.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <rhi.h>

#include "Render/LatchBlock.h"
#include "Render/RunRef.h"
#include "Resources/Buffers/Buffer.h"
#include "Resources/Buffers/VersionedBuffer.h"

namespace org {

// One copy from a run's latch region, at latchOffset, to a target buffer at dstOffset.
struct LatchedCopy {
	// The target's key (LatchedTarget::Key): the buffer, or a versioned buffer's key.
	const void* target = nullptr;
	uint32_t latchOffset = 0, bytes = 0;
	uint64_t dstOffset = 0;
	bool operator==(const LatchedCopy&) const = default;
};

// The copies out of one latch block. A fixed list uses a dedicated block, whose one region every run reads.
struct LatchedList {
	std::shared_ptr<const LatchBlock> latch;
	std::vector<LatchedCopy> copies;
	bool fixed = false;
	bool operator==(const LatchedList& other) const {
		return latch == other.latch && copies == other.copies && fixed == other.fixed;
	}
};

// A buffer latched copies write: a plain buffer, or a versioned buffer, whose current version is the target.
struct LatchedTarget {
	std::shared_ptr<Buffer> buffer;
	std::shared_ptr<VersionedBuffer> versioned;
	// The key LatchedCopy::target names this target by.
	const void* Key() const { return versioned ? versioned->Key() : buffer.get(); }
};

// What one latched copies pass copies for a run.
// The pass copies `lists` first, then each group's lists in order. A pass may cover the copies of several owners this way.
struct LatchedSource {
	struct Group {
		std::shared_ptr<const void> owner;  // keeps the group's lists alive
		std::span<const LatchedList> lists;
	};
	std::shared_ptr<const void> owner;  // keeps `lists` alive
	std::span<const LatchedList> lists;
	std::vector<Group> groups;
	// False leaves out the fixed lists.
	bool copyFixed = true;
};

// Tracks which parts of each run's latch region already hold bytes that only change with a key. Use it from one thread.
// A stable run's region is pinned to its lease index, so a host need rewrite those bytes only when the key changes.
struct LatchRegionsHeld {
	struct Slot {
		RunRef run{};                             // the run whose region the parts were written to
		std::weak_ptr<const void> latch, source;  // owner identities, so a reused address is not mistaken for the old one
		uint64_t extra = 0;                       // anything else the bytes depend on
		std::vector<uint8_t> parts;               // by the caller's part index: written since the key was set
	};
	// The record for `run`. It is cleared when it belongs to another run or another key.
	Slot& Of(const RunRef& run, const std::shared_ptr<const void>& latch, const std::shared_ptr<const void>& source, uint64_t extra) {
		if (run.lease >= slots.size()) slots.resize(std::size_t(run.lease) + 1);
		auto& slot = slots[run.lease];
		if (!(slot.run == run) || !Same(slot.latch, latch) || !Same(slot.source, source) || slot.extra != extra) {
			slot.run = run;
			slot.latch = latch;
			slot.source = source;
			slot.extra = extra;
			slot.parts.clear();
		}
		return slot;
	}
	// Marks part `part` as held. Returns whether it already was.
	static bool Hold(Slot& slot, std::size_t part) {
		if (part >= slot.parts.size()) slot.parts.resize(part + 1, 0);
		return std::exchange(slot.parts[part], uint8_t{1}) != 0;
	}
	static bool Same(const std::weak_ptr<const void>& held, const std::shared_ptr<const void>& now) {
		return now && !held.owner_before(now) && !now.owner_before(held);
	}

	std::vector<Slot> slots;  // by lease index
};

// Sets a shader's latch words for `run`: the region's view index and its offset plus `offset`. Both depend on the run, so
// set them when the pass records. Constants needs the fields latchIndex and latchOffset.
template<class Constants>
void SetLatchWords(Constants& constants, const LatchBlock& latch, const RunRef& run, uint64_t offset = 0) {
	const auto region = latch.At(run);
	constants.latchIndex = region.srvIndex;
	constants.latchOffset = static_cast<uint32_t>(region.offset + offset);
}

// Joins buffer copies into as few commands as possible. A copy that starts where the pending one ends, with the same
// source and target, extends it. Any other copy first hands the pending one to `emit`.
struct LatchedCopyRuns {
	struct Copy {
		rhi::ResourceHandle target, latch;
		uint64_t dst = 0, src = 0, bytes = 0;
	};
	template<class Emit> void Add(const Copy& copy, Emit&& emit) {
		constexpr rhi::HandleEqual<rhi::ResourceHandle> same{};
		if (pending.bytes && same(pending.target, copy.target) && same(pending.latch, copy.latch) &&
			pending.dst + pending.bytes == copy.dst && pending.src + pending.bytes == copy.src) {
			pending.bytes += copy.bytes;
			return;
		}
		Flush(emit);
		pending = copy;
	}
	template<class Emit> void Flush(Emit&& emit) {
		if (!pending.bytes) return;
		emit(static_cast<const Copy&>(pending));
		pending.bytes = 0;
	}
	Copy pending;
};

// Records the copies `source` names for `run`: its own lists, then each group's.
// targetOf(key) returns the target buffer for a copy's target key. A copy whose target resolves to an invalid handle is
// skipped. The latch regions are allocated for the run if they do not exist yet (LatchBlock::At).
template<class TargetOf>
void RecordLatchedCopies(rhi::CommandList& commands, const LatchedSource& source, const RunRef& run, const TargetOf& targetOf) {
	LatchedCopyRuns runs;
	const auto emit = [&](const LatchedCopyRuns::Copy& copy) {
		commands.CopyBufferRegion(copy.target, copy.dst, copy.latch, copy.src, copy.bytes);
	};
	const auto copyLists = [&](std::span<const LatchedList> lists) {
		for (const auto& list : lists) {
			if (!list.latch || (list.fixed && !source.copyFixed)) continue;
			const auto at = list.latch->At(run);
			const auto latch = at.buffer->GetAPIResource().GetHandle();
			for (const auto& copy : list.copies) {
				const rhi::ResourceHandle target = targetOf(copy.target);
				if (!target.valid()) continue;
				runs.Add({target, latch, copy.dstOffset, at.offset + copy.latchOffset, copy.bytes}, emit);
			}
		}
	};
	copyLists(source.lists);
	for (const auto& group : source.groups) copyLists(group.lists);
	runs.Flush(emit);
}

} // namespace org
