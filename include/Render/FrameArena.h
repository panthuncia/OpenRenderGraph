#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "Render/RunRef.h"
#include "Render/Runtime/SubmissionFrontier.h"

namespace org {

class Buffer;

// The memory every LatchBlock region lives in: one arena per runtime (Runtime::Arena), host-visible and persistently
// mapped. Shaders read it as a ByteAddressBuffer, and it is also valid as an indirect-argument buffer and a copy source.
//
// The arena is made of chunks. A chunk never moves and has one shader resource view, so a region's {view, offset} stays
// valid for its whole life. The arena grows by a chunk when the runs in flight need more, and frees chunks that stayed
// empty for a while (Trim).
//
// Two kinds of regions:
// - A leased run's region (RunRef::Leased) is bump-allocated from the current chunk. It is freed when the host retires
//   the run's lease (ReleaseRun), after the GPU is done with the run. So the arena holds what the runs in flight wrote.
// - A pinned region is kept until its owner frees it. Published copies and runs outside a lease use these
//   (RunRef::Stable), because their recordings are submitted again. Pinned regions live in chunks of their own, so they
//   never hold a run's chunk. A freed pinned region is reused only once the GPU reached the point it was freed at.
// - An oversize region (AcquireOversize) has a chunk of its own, of kChunkBytes or more: a client's staging data, such as
//   vertices or texture rows, which may be larger than a chunk. It starts at the chunk's first byte. It is freed as a
//   pinned region is, when its last holder lets it go, and its chunk is then reused for the next one that fits.
//
// Every region but an oversize one is zeroed when it is allocated. Any thread.
class FrameArena : public std::enable_shared_from_this<FrameArena> {
public:
	static constexpr uint32_t kAlignment = 256;
	static constexpr uint32_t kChunkBytes = 256u * 1024u;

	struct Region {
		Buffer* buffer = nullptr;  // the chunk, alive while the region is
		uint32_t srvIndex = 0;     // the chunk's ByteAddressBuffer view
		uint64_t offset = 0;       // the region's first byte in the chunk
		std::byte* mapped = nullptr;
		uint32_t size = 0;
		explicit operator bool() const noexcept { return mapped != nullptr; }
	};

	struct Stats {
		uint64_t chunks = 0, chunkBytes = 0;    // every chunk, for runs and pinned
		uint64_t pinnedChunks = 0;
		uint64_t oversizeChunks = 0;
		uint64_t liveBytes = 0, peakLiveBytes = 0;  // bytes in live regions now, and the most since the last TakeStats
		uint64_t pinnedBytes = 0;               // of liveBytes, in pinned regions
		uint64_t liveRegions = 0;
		uint64_t runs = 0;                      // leased runs holding regions now
		uint64_t trimmed = 0;                   // chunks freed since the last TakeStats
	};

	FrameArena();
	~FrameArena();
	FrameArena(const FrameArena&) = delete;
	FrameArena& operator=(const FrameArena&) = delete;

	// How a freed pinned region waits for the GPU: the frontier at the free, and how to test it. Without one, a freed
	// region is reused at once (a runtime shutting down). A region freed while an assignment scope is open takes its
	// frontier at the first later retire after every scope open at the free has closed (runtime::AssignmentGate).
	void SetFrontier(std::function<runtime::SubmissionFrontier()> frontier, runtime::PointReached reached,
		runtime::AssignmentGate gate = {});

	// A zeroed region of `bytes` (rounded up to kAlignment) for the run. A leased run's is freed with its lease
	// (ReleaseRun). A stable run's is pinned: the caller frees it with FreePinned.
	Region Allocate(const RunRef& run, uint32_t bytes);
	Region AllocatePinned(uint32_t bytes);
	// The region is reused once the GPU reached what was submitted until now.
	void FreePinned(const Region& region);
	// An oversize region of at least `bytes`, in a chunk of its own: the smallest empty oversize chunk that holds it, else a
	// new one of kChunkBytes or the next power of two. Its offset is 0, so the chunk's view reads it from its start. It is not
	// zeroed. The last holder frees it, through FreePinned, so the holder must be what the GPU work that reads it retains
	// (RecordingContext::Retain). Throws when it cannot be made. The arena must be owned by a shared_ptr.
	std::shared_ptr<const Region> AcquireOversize(uint64_t bytes);
	// The host retired the run's lease: its regions are free.
	void ReleaseRun(uint32_t lease, uint32_t generation);

	// Frees chunks that stayed empty for more than idleCalls calls, keeping one spare. The host calls it once a frame.
	void Trim(uint32_t idleCalls = 240);

	Stats TakeStats();

private:
	struct Chunk;
	struct RunRegions {
		uint32_t generation = 0;
		std::vector<std::pair<Chunk*, uint32_t>> regions;  // the chunk and the region's size
	};
	struct PendingFree {
		Chunk* chunk = nullptr;
		uint32_t size = 0;
		runtime::SubmissionFrontier frontier;
		bool open = false;  // freed while an assignment scope was open: frontier is not taken yet
		uint64_t mark = 0;  // the gate's mark at the free, for an open one
	};

	enum class Kind : uint8_t { Run, Pinned, Oversize };
	Chunk* MakeChunkLocked(uint32_t bytes, Kind kind);
	Region TakeLocked(bool pinned, uint32_t bytes, Chunk** taken = nullptr);
	void FreeLocked(Chunk& chunk, uint32_t size);
	void ReleaseRunLocked(RunRegions& run);
	void RetirePendingLocked(bool all);

	mutable std::mutex m_mutex;
	std::vector<std::unique_ptr<Chunk>> m_chunks;
	Chunk* m_current = nullptr;        // where leased runs bump-allocate
	Chunk* m_currentPinned = nullptr;  // where pinned regions bump-allocate
	std::vector<RunRegions> m_runs;    // by lease index
	std::vector<PendingFree> m_pending;
	std::function<runtime::SubmissionFrontier()> m_frontier;
	runtime::PointReached m_reached;
	runtime::AssignmentGate m_gate;
	Stats m_stats{};
};

} // namespace org
