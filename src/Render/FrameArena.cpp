#include "Render/FrameArena.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <stdexcept>
#include <string>

#include "Managers/Singletons/DeviceManager.h"
#include "Resources/Buffers/Buffer.h"

#include <rhi_feature_info.h>
#include <spdlog/spdlog.h>

namespace org {

namespace {
// Device-local memory that the host can write, when the device has it (ReBAR, or D3D12's GPU upload heap). Then every
// dispatch that reads a region reads video memory instead of host memory over the bus. Otherwise host memory.
rhi::HeapType ArenaHeap() {
	static const rhi::HeapType heap = [] {
		ResourceAllocationFeatureInfo allocation{};
		const auto device = DeviceManager::GetInstance().GetDevice();
		const bool deviceLocal = device.QueryFeatureInfo(&allocation.header) == rhi::Result::Ok && allocation.gpuUploadHeapSupported;
		spdlog::info("[ORG] frame arena in {} memory", deviceLocal ? "device-local host-visible" : "host");
		return deviceLocal ? rhi::HeapType::GPUUpload : rhi::HeapType::Upload;
	}();
	return heap;
}

uint32_t AlignUp(uint32_t bytes) noexcept {
	return (bytes + FrameArena::kAlignment - 1) & ~(FrameArena::kAlignment - 1);
}
} // namespace

struct FrameArena::Chunk {
	std::shared_ptr<Buffer> buffer;
	std::byte* mapped = nullptr;
	uint32_t srvIndex = 0;
	uint32_t capacity = 0;
	uint32_t used = 0;   // the bump cursor
	uint32_t live = 0;   // live regions in it
	uint32_t idle = 0;   // Trim calls it has been empty for
	bool pinned = false;    // pinned regions, or an oversize one
	bool oversize = false;  // one oversize region, at offset 0

	~Chunk() {
		if (buffer && mapped) buffer->GetAPIResource().Unmap(0, capacity);
	}
};

FrameArena::FrameArena() = default;

FrameArena::~FrameArena() {
	std::scoped_lock lock(m_mutex);
	m_pending.clear();
	m_runs.clear();
	m_current = m_currentPinned = nullptr;
	m_chunks.clear();
}

void FrameArena::SetFrontier(std::function<runtime::SubmissionFrontier()> frontier, runtime::PointReached reached,
	std::function<bool()> settled) {
	std::scoped_lock lock(m_mutex);
	m_frontier = std::move(frontier);
	m_reached = std::move(reached);
	m_settled = std::move(settled);
	if (!m_frontier) RetirePendingLocked(true);
}

FrameArena::Chunk* FrameArena::MakeChunkLocked(uint32_t bytes, Kind kind) {
	auto chunk = std::make_unique<Chunk>();
	const bool pinned = kind != Kind::Run;
	chunk->oversize = kind == Kind::Oversize;
	chunk->capacity = chunk->oversize ? std::bit_ceil((std::max)(kChunkBytes, AlignUp(bytes))) : (std::max)(kChunkBytes, AlignUp(bytes));
	chunk->pinned = pinned;
	// No ECS entity: the arena is not a graph resource, and it may grow on any thread. Its last reference may drop on any
	// thread, and its buffer then goes through ORG's deletion path like any other.
	Resource::ScopedECSRegistrationSuppression suppressECS;
	// 32-bit words, read as a ByteAddressBuffer. The host writes it and never reads it back, because device-local
	// host-visible memory is write-combined.
	chunk->buffer = Buffer::CreateUnmaterializedStructuredBuffer(chunk->capacity / 4, 4, false, false, false, ArenaHeap());
	const char* name = chunk->oversize ? "ORG frame arena (oversize) " : pinned ? "ORG frame arena (pinned) " : "ORG frame arena ";
	chunk->buffer->SetName(std::string(name) + std::to_string(m_chunks.size()));
	chunk->buffer->Materialize();
	void* mapped = nullptr;
	chunk->buffer->GetAPIResource().Map(&mapped, 0, chunk->capacity);
	if (!mapped) throw std::runtime_error("A frame arena chunk could not be mapped");
	chunk->mapped = static_cast<std::byte*>(mapped);
	chunk->srvIndex = chunk->buffer->GetSRVInfo(0).slot.index;
	m_stats.chunks += 1;
	m_stats.chunkBytes += chunk->capacity;
	if (chunk->oversize) m_stats.oversizeChunks += 1;
	else if (pinned) m_stats.pinnedChunks += 1;
	m_chunks.push_back(std::move(chunk));
	return m_chunks.back().get();
}

FrameArena::Region FrameArena::TakeLocked(bool pinned, uint32_t bytes, Chunk** taken) {
	RetirePendingLocked(false);
	auto*& current = pinned ? m_currentPinned : m_current;
	const auto fits = [&](const Chunk* chunk) { return chunk && chunk->used + bytes <= chunk->capacity; };
	if (!fits(current)) {
		// An empty chunk of the same kind, the smallest that holds the region. Else a new one.
		Chunk* best = nullptr;
		for (const auto& chunk : m_chunks)
			if (chunk->pinned == pinned && !chunk->oversize && chunk->live == 0 && chunk->capacity >= bytes && (!best || chunk->capacity < best->capacity))
				best = chunk.get();
		if (best) best->used = 0;
		current = best ? best : MakeChunkLocked(bytes, pinned ? Kind::Pinned : Kind::Run);
	}
	Chunk& chunk = *current;
	Region region;
	region.buffer = chunk.buffer.get();
	region.srvIndex = chunk.srvIndex;
	region.offset = chunk.used;
	region.mapped = chunk.mapped + chunk.used;
	region.size = bytes;
	chunk.used += bytes;
	chunk.live += 1;
	chunk.idle = 0;
	std::memset(region.mapped, 0, bytes);
	m_stats.liveBytes += bytes;
	m_stats.liveRegions += 1;
	if (pinned) m_stats.pinnedBytes += bytes;
	m_stats.peakLiveBytes = (std::max)(m_stats.peakLiveBytes, m_stats.liveBytes);
	if (taken) *taken = &chunk;
	return region;
}

void FrameArena::FreeLocked(Chunk& chunk, uint32_t size) {
	chunk.live -= 1;
	m_stats.liveBytes -= size;
	m_stats.liveRegions -= 1;
	if (chunk.pinned) m_stats.pinnedBytes -= size;
	// An empty chunk starts over, even the current one: nothing in it is read any more.
	if (chunk.live == 0) chunk.used = 0;
}

FrameArena::Region FrameArena::Allocate(const RunRef& run, uint32_t bytes) {
	if (!bytes) throw std::invalid_argument("A frame arena region needs a size");
	bytes = AlignUp(bytes);
	std::scoped_lock lock(m_mutex);
	if (run.Stable()) return TakeLocked(true, bytes);
	if (m_runs.size() <= run.lease) m_runs.resize(size_t{run.lease} + 1);
	auto& regions = m_runs[run.lease];
	// What an older run of the index left is its: that run retired, or it would not have been opened again.
	if (regions.generation != run.generation) {
		ReleaseRunLocked(regions);
		regions.generation = run.generation;
	}
	const bool first = regions.regions.empty();
	Chunk* chunk = nullptr;
	auto region = TakeLocked(false, bytes, &chunk);
	regions.regions.emplace_back(chunk, bytes);
	if (first) m_stats.runs += 1;
	return region;
}

FrameArena::Region FrameArena::AllocatePinned(uint32_t bytes) {
	if (!bytes) throw std::invalid_argument("A frame arena region needs a size");
	bytes = AlignUp(bytes);
	std::scoped_lock lock(m_mutex);
	return TakeLocked(true, bytes);
}

void FrameArena::FreePinned(const Region& region) {
	if (!region) return;
	std::scoped_lock lock(m_mutex);
	Chunk* chunk = nullptr;
	for (const auto& candidate : m_chunks)
		if (candidate->buffer.get() == region.buffer) chunk = candidate.get();
	if (!chunk) return;
	if (!m_frontier) {
		FreeLocked(*chunk, region.size);
		return;
	}
	// A submission still queued on a host's submission thread may read the region. Its values are not handed out yet, so the
	// frontier now would miss it: the free waits for a frontier taken once nothing is queued.
	if (m_settled && !m_settled()) {
		m_pending.push_back({chunk, region.size, {}, true});
		return;
	}
	m_pending.push_back({chunk, region.size, m_frontier()});
}

std::shared_ptr<const FrameArena::Region> FrameArena::AcquireOversize(uint64_t bytes) {
	if (!bytes || bytes > (1ull << 31)) throw std::invalid_argument("An oversize frame arena region needs 1 byte to 2 GiB, not " + std::to_string(bytes));
	const auto size = AlignUp(static_cast<uint32_t>(bytes));
	Region region;
	{
		std::scoped_lock lock(m_mutex);
		RetirePendingLocked(false);
		Chunk* best = nullptr;
		for (const auto& chunk : m_chunks)
			if (chunk->oversize && chunk->live == 0 && chunk->capacity >= size && (!best || chunk->capacity < best->capacity)) best = chunk.get();
		Chunk& chunk = best ? *best : *MakeChunkLocked(size, Kind::Oversize);
		region.buffer = chunk.buffer.get();
		region.srvIndex = chunk.srvIndex;
		region.offset = 0;
		region.mapped = chunk.mapped;
		region.size = size;
		chunk.used = size;
		chunk.live = 1;
		chunk.idle = 0;
		m_stats.liveBytes += size;
		m_stats.liveRegions += 1;
		m_stats.pinnedBytes += size;
		m_stats.peakLiveBytes = (std::max)(m_stats.peakLiveBytes, m_stats.liveBytes);
	}
	// The holder keeps the arena, so a region let go after the runtime's last reference still comes back.
	auto* held = new Region(region);
	return std::shared_ptr<const Region>(held, [arena = shared_from_this()](const Region* freed) {
		arena->FreePinned(*freed);
		delete freed;
	});
}

void FrameArena::ReleaseRunLocked(RunRegions& run) {
	if (run.regions.empty()) return;
	for (auto& [chunk, size] : run.regions) FreeLocked(*chunk, size);
	run.regions.clear();
	m_stats.runs -= 1;
}

void FrameArena::ReleaseRun(uint32_t lease, uint32_t generation) {
	std::scoped_lock lock(m_mutex);
	if (lease >= m_runs.size() || m_runs[lease].generation != generation) return;
	ReleaseRunLocked(m_runs[lease]);
}

void FrameArena::RetirePendingLocked(bool all) {
	const bool settled = !m_settled || m_settled();
	std::erase_if(m_pending, [&](PendingFree& pending) {
		if (!all && pending.open) {
			if (!settled || !m_frontier) return false;
			pending.frontier = m_frontier();
			pending.open = false;
		}
		if (!all && !runtime::Reached(pending.frontier, m_reached)) return false;
		FreeLocked(*pending.chunk, pending.size);
		return true;
	});
}

void FrameArena::Trim(uint32_t idleCalls) {
	std::scoped_lock lock(m_mutex);
	RetirePendingLocked(false);
	// The current chunks stay, and so does the first other empty chunk of each kind: the next runs take it at once.
	bool spareKept[3] = {false, false, false};
	std::erase_if(m_chunks, [&](std::unique_ptr<Chunk>& chunk) {
		if (chunk->live != 0 || chunk.get() == m_current || chunk.get() == m_currentPinned) return false;
		if (auto& spare = spareKept[chunk->oversize ? 2 : chunk->pinned ? 1 : 0]; !spare) {
			spare = true;
			return false;
		}
		if (++chunk->idle <= idleCalls) return false;
		m_stats.chunks -= 1;
		m_stats.chunkBytes -= chunk->capacity;
		if (chunk->oversize) m_stats.oversizeChunks -= 1;
		else if (chunk->pinned) m_stats.pinnedChunks -= 1;
		m_stats.trimmed += 1;
		return true;
	});
}

FrameArena::Stats FrameArena::TakeStats() {
	std::scoped_lock lock(m_mutex);
	auto stats = m_stats;
	m_stats.peakLiveBytes = m_stats.liveBytes;
	m_stats.trimmed = 0;
	return stats;
}

} // namespace org
