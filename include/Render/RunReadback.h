#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <rhi.h>

#include "Render/RunRef.h"

namespace org {

class LatchBlock;
class RenderPass;
class Resource;

// ORG's one readback service: small copies from the GPU to the host that a kept recording can replay.
//
// A stage recorded ahead submits its recording again and again (a revision-driven epoch). So a readback's commands may
// depend only on the run, never on a request made this frame. Each run copies into a region of a host buffer that the run
// names, and the host reads a region only once the run that last copied there is done. Nothing waits.
// Make one per reader and keep it across builds.
//
// Three schemes share the regions:
// - Parts (Create): one region per lease index, with fixed parts. Each copy also copies a stamp from the client's latch,
//   so a run the client did not commit is never read as one it did. A region is read when the lease's next run commits.
// - Lists (CreateList): one growable, count-prefixed part per stable run, for an event list a GPU pass appends to. The
//   client writes the stamp into the list's header. A region is read when its stable run commits again.
// - Regions (CreateRegions): one fixed region per stable run, which the client's passes copy into and the client reads
//   when it knows the region's last copy is done. The client's own stamp, copied with the data, says which run it is.
//
// Errors go to the host's error handler, given at creation. When it returns, the call that failed does nothing more: a
// Create returns null, Commit returns 0 (never a stamp), and a copy or read is left out.
class RunReadback {
public:
	// Called with what went wrong. It may be called with the readback's lock held, so it must not call the readback.
	using ErrorHandler = std::function<void(std::string_view)>;

	// The latch bytes a run's stamp takes. The client writes it where Bind said, with the value Commit returns.
	struct Stamp {
		uint32_t value = 0;
		uint32_t pad[3]{};
	};

	static constexpr uint32_t kNoRegion = UINT32_MAX;

	// One part per copy, parts[i] bytes each.
	static std::shared_ptr<RunReadback> Create(std::string_view name, std::vector<uint32_t> parts, ErrorHandler onError);

	// At every build, before the passes: the client's latch, and the offset of the Stamp the client writes there.
	void Bind(std::shared_ptr<const LatchBlock> latch, uint32_t stampOffset);

	// A pass that copies the first bytes of buffer source into part part. Empty, with why, when it cannot be made.
	std::function<std::shared_ptr<RenderPass>()> BufferPass(uint32_t part, std::shared_ptr<Resource> source, std::string& why) const;

	// A pass that copies mip 0 of a small 2D texture into part part. Rows are 16 bytes a texel, aligned to 256. Empty, with
	// why, when it cannot.
	std::function<std::shared_ptr<RenderPass>()> TexturePass(uint32_t part, std::shared_ptr<Resource> source, uint32_t width,
		uint32_t height, std::string& why) const;

	// In the client's commit, before Take. Reads the lease's last run and stamps this run. Returns this run's stamp, which the
	// client writes into run's latch as a Stamp. Stamps grow by one each commit. 0 when the run's lease is past the regions.
	uint32_t Commit(const RunRef& run);

	// In the client's executed callback: the run last committed was submitted.
	void Executed();

	// The newest copy of part part read since the last Take, and its run's stamp. Empty when none arrived.
	std::vector<std::byte> Take(uint32_t part, uint32_t* stamp = nullptr);

	// A host that gathers a run's values in a staging run (GraphHost::EpochStagingRun) moves what every part readback
	// committed there onto the run it submits, as LatchBlock::CopyStableRun moves the latches. to's last run is done.
	static void MoveStagingRun(const RunRef& from, const RunRef& to);

	// A list readback's part: a header whose first word is the item count, then the items.
	struct List {
		uint32_t headerBytes = 16;
		uint32_t stampOffset = 4;  // the header word the client writes the run's stamp into, at its commit
		uint32_t itemBytes = 0;
		uint32_t regions = 16;     // enough for every stable run that copies the list
	};
	// What CommitList read for its region's last run. words is null when that run's copy is not in any set.
	struct ListRead {
		uint32_t region = 0, stamp = 0;
		const uint32_t* words = nullptr;   // the header, then the items. Mapped only during the call.
		uint32_t count = 0, capacity = 0;  // count may pass capacity: the items past it were left out
	};
	// A list commit's region, kNoRegion for a run with none (a leased run, or every region taken), and its stamp.
	struct ListCommit {
		uint32_t region = kNoRegion, stamp = 0;
	};
	// A list readback's counts. The first three are since the last take.
	struct ListStats {
		uint64_t overflowed = 0, missed = 0, setsAdded = 0;
		uint32_t sets = 0, largest = 0;  // buffer sets, and the largest capacity
	};
	// The buffers for one list capacity. Kept until the readback goes, since a kept recording may name any of them.
	struct ListSet;

	// A list readback with a buffer set for items.
	static std::shared_ptr<RunReadback> CreateList(std::string_view name, const List& list, uint32_t items, ErrorHandler onError);
	// Adds a buffer set for a list of items, if none has it. Any thread, one at a time.
	void Grow(uint32_t items);
	// The set for a list of items, or null. A pass binds the list only with a set, so its copy misses nothing. Any thread.
	const ListSet* SetFor(uint32_t items) const;
	// In a pass's recording: copies list source, its header and every item of set's capacity, into run's region.
	void RecordList(rhi::CommandList& commands, const RunRef& run, rhi::ResourceHandle source, const ListSet& set) const;
	// In the client's commit. Reads the region's last committed run, whose lease has retired, and calls read once for it.
	// Then stamps this run. The client writes the returned stamp at the list's stampOffset with a zero count. Nothing waits.
	ListCommit CommitList(const RunRef& run, const std::function<void(const ListRead&)>& read);
	// The largest count a read saw past its list. The list grows to it, and what was left out is appended again. Any thread.
	uint32_t Wanted() const;
	// A list's or a region readback's region count.
	uint32_t Regions() const;
	// A list readback's counts; take zeroes the first three. Any thread.
	ListStats Stats(bool take);

	// A region readback: regions regions of bytes bytes each, one per stable run.
	static std::shared_ptr<RunReadback> CreateRegions(std::string_view name, uint32_t bytes, uint32_t regions, ErrorHandler onError);
	// run's region, given on first use. kNoRegion for a leased run, whose lease changes every run, or when every region is
	// taken. Any thread.
	uint32_t RegionOf(const RunRef& run) const;
	// In a pass's recording: copies bytes bytes of source at sourceOffset to offset in region. An error past the region.
	void RecordCopy(rhi::CommandList& commands, uint32_t region, uint64_t offset, rhi::ResourceHandle source, uint64_t sourceOffset,
		uint64_t bytes) const;
	// Maps region for read. The caller knows the region's last copy is done and no later copy runs meanwhile.
	void ReadRegion(uint32_t region, const std::function<void(const std::byte*)>& read) const;

	struct State;
	explicit RunReadback(std::shared_ptr<State> state) : m_state(std::move(state)) {}

private:
	std::shared_ptr<State> m_state;
};

} // namespace org
