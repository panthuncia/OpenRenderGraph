#include "Render/LatchBlock.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "OpenRenderGraph/Runtime.h"
#include "Managers/Singletons/DeviceManager.h"
#include "Resources/Buffers/Buffer.h"

#include <rhi_feature_info.h>

namespace org {

namespace {
// Every arena block alive, so a stable run's regions can be cleared in all of them (ZeroStableRun).
std::mutex& RegistryMutex() {
	static std::mutex mutex;
	return mutex;
}
std::vector<const LatchBlock*>& Registry() {
	static std::vector<const LatchBlock*> blocks;
	return blocks;
}
// By stable lease index: the arena blocks with a pinned region there, so a stable run's regions are found without walking every
// block. A leaf lock: taken inside a block's own mutex (At) and inside RegistryMutex, never around either.
std::mutex& StableMutex() {
	static std::mutex mutex;
	return mutex;
}
std::vector<std::vector<const LatchBlock*>>& StableBlocks() {
	static std::vector<std::vector<const LatchBlock*>> blocks;
	return blocks;
}
void NoteStable(const LatchBlock* block, uint32_t lease) {
	std::scoped_lock lock(StableMutex());
	auto& lists = StableBlocks();
	if (lists.size() <= lease) lists.resize(size_t{lease} + 1);
	lists[lease].push_back(block);
}
// The stable lease indices whose regions are host memory (MarkHostOnlyRun). Read in At under a block's mutex: a leaf lock.
std::mutex& HostOnlyMutex() {
	static std::mutex mutex;
	return mutex;
}
std::vector<bool>& HostOnly() {
	static std::vector<bool> leases;
	return leases;
}
bool IsHostOnly(uint32_t lease) {
	std::scoped_lock lock(HostOnlyMutex());
	return lease < HostOnly().size() && HostOnly()[lease];
}
std::vector<const LatchBlock*> StableAt(uint32_t lease) {
	std::scoped_lock lock(StableMutex());
	const auto& lists = StableBlocks();
	return lease < lists.size() ? lists[lease] : std::vector<const LatchBlock*>{};
}
// Appends lease's blocks to `out`, which keeps its capacity across calls.
void AppendStableAt(uint32_t lease, std::vector<const LatchBlock*>& out) {
	std::scoped_lock lock(StableMutex());
	const auto& lists = StableBlocks();
	if (lease < lists.size()) out.insert(out.end(), lists[lease].begin(), lists[lease].end());
}
}

void LatchBlock::ZeroStableRun(uint32_t lease) {
	std::scoped_lock registry(RegistryMutex());
	// A host-only run's region is cleared only as far as anything was written to it (m_stableHigh). Any other run's whole region: its
	// recordings may write through the region's pointer.
	const bool hostOnly = IsHostOnly(lease);
	for (const auto* block : StableAt(lease)) {
		std::scoped_lock lock(block->m_mutex);
		if (lease >= block->m_stable.size() || !block->m_stable[lease]) continue;
		auto& high = block->m_stableHigh[lease];
		const uint32_t bytes = hostOnly ? high : block->m_stable[lease].size;
		if (bytes) std::memset(block->m_stable[lease].mapped, 0, bytes);
		high = 0;
	}
}

void LatchBlock::MarkHostOnlyRun(uint32_t lease) {
	std::scoped_lock lock(HostOnlyMutex());
	if (HostOnly().size() <= lease) HostOnly().resize(size_t{lease} + 1);
	HostOnly()[lease] = true;
}

void LatchBlock::CopyStableRun(uint32_t from, uint32_t to) {
	if (from == to) return;
	std::scoped_lock registry(RegistryMutex());
	// A block with a region at `to` and none at `from` is zeroed; one with a region at `from` is copied. Only the bytes that can differ
	// move: the source's written part (all of it unless `from` is host-only), and the target's part a write or an earlier copy left
	// non-zero. A block written at neither costs nothing, so a stage's clients that staged nothing this frame cost nothing.
	const bool hostOnly = IsHostOnly(from);
	// Reused: a stage's commit copies a run every frame. Callers hold RegistryMutex, so one thread uses it at a time.
	static std::vector<const LatchBlock*> blocks;
	blocks.clear();
	AppendStableAt(to, blocks);
	AppendStableAt(from, blocks);
	std::sort(blocks.begin(), blocks.end());
	blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
	for (const auto* block : blocks) {
		std::scoped_lock lock(block->m_mutex);
		const bool hasSource = from < block->m_stable.size() && block->m_stable[from];
		const uint32_t written = !hasSource ? 0u : hostOnly ? block->m_stableHigh[from] : block->m_stable[from].size;
		const bool hasTarget = to < block->m_stable.size() && block->m_stable[to];
		const uint32_t left = hasTarget ? block->m_stableHigh[to] : 0u;
		if (!written && !left) continue;
		const Region target = block->StableRegion(to);
		const uint32_t copied = (std::min)(written, target.size);
		if (copied) std::memcpy(target.mapped, block->m_stable[from].mapped, copied);
		if (left > copied) std::memset(target.mapped + copied, 0, (std::min)(left, target.size) - copied);
		block->m_stableHigh[to] = copied;
	}
}

LatchBlock::LatchBlock(std::string name, uint32_t bytesPerRun)
	: m_name(std::move(name)), m_stride((bytesPerRun + FrameArena::kAlignment - 1) & ~(FrameArena::kAlignment - 1)) {
	if (!bytesPerRun) throw std::invalid_argument("Latch block '" + m_name + "' needs a size");
	auto* runtime = Runtime::Current();
	if (!runtime || !runtime->Arena()) throw std::logic_error("Latch block '" + m_name + "' needs a runtime and its frame arena");
	m_arena = runtime->Arena();
	std::scoped_lock registry(RegistryMutex());
	Registry().push_back(this);
}

std::shared_ptr<LatchBlock> LatchBlock::Create(std::string name, uint32_t bytesPerRun) {
	return std::make_shared<LatchBlock>(std::move(name), bytesPerRun);
}

LatchBlock::LatchBlock(DedicatedTag, std::string name, uint32_t bytes)
	: m_name(std::move(name)), m_stride((bytes + FrameArena::kAlignment - 1) & ~(FrameArena::kAlignment - 1)) {
	if (!bytes) throw std::invalid_argument("Latch block '" + m_name + "' needs a size");
	// No ECS entity: the block is not a graph resource, and a producer's thread may make it. Its last reference may drop on any
	// thread, and its buffer then goes through ORG's deletion path like any other.
	Resource::ScopedECSRegistrationSuppression suppressECS;
	// Device-local memory the host can write when the device has it (ReBAR, D3D12's GPU upload heap), else host memory.
	ResourceAllocationFeatureInfo allocation{};
	const bool deviceLocal = DeviceManager::GetInstance().GetDevice().QueryFeatureInfo(&allocation.header) == rhi::Result::Ok && allocation.gpuUploadHeapSupported;
	m_buffer = Buffer::CreateUnmaterializedStructuredBuffer(m_stride / 4, 4, false, false, false, deviceLocal ? rhi::HeapType::GPUUpload : rhi::HeapType::Upload);
	m_buffer->SetName(m_name);
	m_buffer->Materialize();
	void* mapped = nullptr;
	m_buffer->GetAPIResource().Map(&mapped, 0, m_stride);
	if (!mapped) throw std::runtime_error("Latch block '" + m_name + "' could not be mapped");
	m_mapped = static_cast<std::byte*>(mapped);
	std::memset(m_mapped, 0, m_stride);
	m_srvIndex = m_buffer->GetSRVInfo(0).slot.index;
}

std::shared_ptr<LatchBlock> LatchBlock::CreateDedicated(std::string name, uint32_t bytes) {
	return std::make_shared<LatchBlock>(DedicatedTag{}, std::move(name), bytes);
}

LatchBlock::~LatchBlock() {
	if (!m_buffer) {
		std::scoped_lock registry(RegistryMutex());
		std::erase(Registry(), this);
		std::scoped_lock stable(StableMutex());
		for (auto& list : StableBlocks()) std::erase(list, this);
	}
	if (m_buffer && m_mapped) m_buffer->GetAPIResource().Unmap(0, m_stride);
	// A leased run's region belongs to its lease, which frees it when the run retires. The pinned ones are the block's: they
	// are reused once the GPU is past what was submitted so far.
	for (const auto& region : m_stable) {
		if (!region) continue;
		if (region.buffer) m_arena->FreePinned(region);
		else delete[] region.mapped;
	}
}

LatchBlock::Region LatchBlock::StableRegion(uint32_t lease) const {
	if (m_stable.size() <= lease) {
		m_stable.resize(size_t{lease} + 1);
		m_stableHigh.resize(size_t{lease} + 1);
	}
	auto& region = m_stable[lease];
	if (!region) {
		if (IsHostOnly(lease)) {
			// Plain zeroed host memory, freed with the block (no buffer: the destructor knows it by that).
			region = Region{nullptr, 0, 0, new std::byte[m_stride]{}, m_stride};
		} else {
			region = m_arena->AllocatePinned(m_stride);
		}
		m_stableHigh[lease] = 0;
		NoteStable(this, lease);
	}
	return region;
}

LatchBlock::Region LatchBlock::At(const RunRef& run) const {
	// A dedicated block has one region, which every run names.
	if (m_buffer) return Region{m_buffer.get(), m_srvIndex, 0, m_mapped, m_stride};
	std::scoped_lock lock(m_mutex);
	if (run.Stable()) {
		const auto region = StableRegion(run.lease);
		// A host-only run's region handed out: whoever asked may write any of it.
		if (IsHostOnly(run.lease)) m_stableHigh[run.lease] = region.size;
		return region;
	}
	if (m_runs.size() <= run.lease) m_runs.resize(size_t{run.lease} + 1);
	auto& entry = m_runs[run.lease];
	if (entry.generation != run.generation || !entry.region) {
		entry.region = m_arena->Allocate(run, m_stride);
		entry.generation = run.generation;
	}
	return entry.region;
}

void LatchBlock::Zero(const RunRef& run) const {
	if (m_buffer || !run.Stable() || !IsHostOnly(run.lease)) {
		const auto region = At(run);
		std::memset(region.mapped, 0, region.size);
		return;
	}
	std::scoped_lock lock(m_mutex);
	const auto region = StableRegion(run.lease);
	auto& high = m_stableHigh[run.lease];
	if (high) std::memset(region.mapped, 0, (std::min)(high, region.size));
	high = 0;
}

void LatchBlock::Write(const RunRef& run, uint32_t offset, std::span<const std::byte> bytes) const {
	if (uint64_t(offset) + bytes.size() > m_stride)
		throw std::out_of_range("Latch block '" + m_name + "' write past its region");
	if (m_buffer || !run.Stable()) {
		const auto region = At(run);
		std::memcpy(region.mapped + offset, bytes.data(), bytes.size());
		return;
	}
	std::scoped_lock lock(m_mutex);
	const auto region = StableRegion(run.lease);
	std::memcpy(region.mapped + offset, bytes.data(), bytes.size());
	auto& high = m_stableHigh[run.lease];
	high = (std::max)(high, offset + static_cast<uint32_t>(bytes.size()));
}

} // namespace org
