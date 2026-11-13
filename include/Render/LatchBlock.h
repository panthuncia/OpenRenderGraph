#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "Render/FrameArena.h"
#include "Render/RunRef.h"

namespace org {

class Buffer;

// Late latching. Some per-execution values would otherwise be baked into a pass's recording: matrices, counts,
// dispatch sizes, frame stamps. Instead, they live in a small host-visible region per run. The pass records only where
// the region is. That location depends on the run, not on the values. So the pass's invocation and its recorded commands
// do not depend on the values, and they can be prepared before the values are known. The host writes the values into the
// run's region just before submission.
//
// A block is a view into the runtime's frame arena (FrameArena, Runtime::Arena). It has no memory of its own:
// - A leased run's region is allocated the first time anything asks for it (At, Offset, Write). The writer (the epoch's
//   beforeSubmit, through GraphHost::CurrentRun) and the recorder (through RecordingContext::Run) name the same run, so
//   whichever comes first allocates it and the other finds it. It is freed when the host retires the run's lease.
// - A stable run's region (a published copy's, or an index outside a lease: RunRef::Stable) is pinned to the index for
//   the block's life, because its recording is submitted again and reads the same place each time.
// A region is zeroed when it is allocated. A run that reads a block without writing it reads zeros.
//
// Contract:
//  - The host writes a run's region only from the run's start (GraphHost::CurrentRun inside its beforeSubmit or
//    ExecuteFrame's beforePrepare, or an external run) until the run is submitted.
//  - Readers see the host writes made before the submission. The memory is host-coherent, and queue submission makes
//    the writes available.
//  - The graph's state tracking ignores the block. It is not a graph resource, and it needs no declaration, barriers
//    or transitions.
//  - Shaders read a region as a ByteAddressBuffer through its view (Region::srvIndex) at its offset. A region is also
//    valid as an indirect-argument buffer and as a copy source (Region::buffer).
//  - A run reference is valid until its lease retires. Never ask for the region of a run whose lease retired.
class LatchBlock {
public:
	using Region = FrameArena::Region;

	// bytesPerRun is rounded up to 256, so every region is aligned for any use. Any thread: the block makes nothing until a
	// region is asked for. Needs a runtime (Runtime::Current).
	LatchBlock(std::string name, uint32_t bytesPerRun);
	static std::shared_ptr<LatchBlock> Create(std::string name, uint32_t bytesPerRun);
	~LatchBlock();
	LatchBlock(const LatchBlock&) = delete;
	LatchBlock& operator=(const LatchBlock&) = delete;

	const std::string& Name() const noexcept { return m_name; }
	uint32_t Stride() const noexcept { return m_stride; }

	// The run's region, allocated on first use.
	Region At(const RunRef& run) const;
	uint64_t Offset(const RunRef& run) const { return At(run).offset; }
	uint32_t SrvIndex(const RunRef& run) const { return At(run).srvIndex; }
	std::span<std::byte> Bytes(const RunRef& run) const {
		const auto region = At(run);
		return {region.mapped, region.size};
	}
	// Copies bytes to the run's region at offset; throws if they do not fit.
	void Write(const RunRef& run, uint32_t offset, std::span<const std::byte> bytes) const;
	template<class T> void WriteValue(const RunRef& run, uint32_t offset, const T& value) const {
		Write(run, offset, std::as_bytes(std::span<const T, 1>(&value, 1)));
	}

	// A block with a buffer of its own and one region, which every run names: a client's own staging buffer, which the client keeps
	// from reuse while the GPU reads it (RecordingContext::Retain). Its region is zeroed when made.
	static std::shared_ptr<LatchBlock> CreateDedicated(std::string name, uint32_t bytes);
	// A block from a registration that names how many regions it has (ORGModuleServices' LatchRegistration): 1 makes a dedicated
	// block, whose one region every run reads; any other count a frame arena block, whose regions follow the runs.
	static std::shared_ptr<LatchBlock> Create(std::string name, uint32_t bytes, uint32_t regions) {
		return regions == 1 ? CreateDedicated(std::move(name), bytes) : Create(std::move(name), bytes);
	}
	bool Dedicated() const noexcept { return static_cast<bool>(m_buffer); }
	// A dedicated block's buffer, its view and its bytes (null, 0 and empty for an arena block).
	const std::shared_ptr<Buffer>& Resource() const noexcept { return m_buffer; }
	uint32_t SrvIndex() const noexcept { return m_srvIndex; }
	std::span<std::byte> Bytes() const noexcept { return {m_mapped, m_mapped ? m_stride : 0u}; }

	struct DedicatedTag {};
	LatchBlock(DedicatedTag, std::string name, uint32_t bytes);

private:
	struct RunEntry {
		uint32_t generation = 0;
		Region region;
	};

	std::string m_name;
	uint32_t m_stride = 0;
	std::shared_ptr<FrameArena> m_arena;
	mutable std::mutex m_mutex;
	mutable std::vector<RunEntry> m_runs;    // by lease index: the leased run that last asked
	mutable std::vector<Region> m_stable;    // by lease index: pinned for the block's life
	// A dedicated block's buffer.
	uint32_t m_srvIndex = 0;
	std::shared_ptr<Buffer> m_buffer;
	std::byte* m_mapped = nullptr;
};

} // namespace org
