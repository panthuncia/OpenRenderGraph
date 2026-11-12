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
//
// Every region is zeroed when it is allocated. Any thread.
class FrameArena {
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
	// region is reused at once (a runtime shutting down).
	void SetFrontier(std::function<runtime::SubmissionFrontier()> frontier, runtime::PointReached reached);

	// A zeroed region of `bytes` (rounded up to kAlignment) for the run. A leased run's is freed with its lease
	// (ReleaseRun). A stable run's is pinned: the caller frees it with FreePinned.
	Region Allocate(const RunRef& run, uint32_t bytes);
	Region AllocatePinned(uint32_t bytes);
	// The region is reused once the GPU reached what was submitted until now.
	void FreePinned(const Region& region);
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
	};

	Chunk* MakeChunkLocked(uint32_t bytes, bool pinned);
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
	Stats m_stats{};
};

} // namespace org
