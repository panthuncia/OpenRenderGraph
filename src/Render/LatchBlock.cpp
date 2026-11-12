#include "Render/LatchBlock.h"

#include <cstring>
#include <stdexcept>

#include "OpenRenderGraph/Runtime.h"
#include "Managers/Singletons/DeviceManager.h"
#include "Resources/Buffers/Buffer.h"

#include <rhi_feature_info.h>

namespace org {

LatchBlock::LatchBlock(std::string name, uint32_t bytesPerRun)
	: m_name(std::move(name)), m_stride((bytesPerRun + FrameArena::kAlignment - 1) & ~(FrameArena::kAlignment - 1)) {
	if (!bytesPerRun) throw std::invalid_argument("Latch block '" + m_name + "' needs a size");
	auto* runtime = Runtime::Current();
	if (!runtime || !runtime->Arena()) throw std::logic_error("Latch block '" + m_name + "' needs a runtime and its frame arena");
	m_arena = runtime->Arena();
}

std::shared_ptr<LatchBlock> LatchBlock::Create(std::string name, uint32_t bytesPerRun) {
	return std::make_shared<LatchBlock>(std::move(name), bytesPerRun);
}

LatchBlock::LatchBlock(std::string name, uint32_t bytesPerSlot, uint32_t slots)
	: m_name(std::move(name)), m_stride((bytesPerSlot + FrameArena::kAlignment - 1) & ~(FrameArena::kAlignment - 1)), m_slots(slots) {
	if (!bytesPerSlot || !slots) throw std::invalid_argument("Latch block '" + m_name + "' needs a size and slots");
	const uint64_t bytes = uint64_t(m_stride) * m_slots;
	Resource::ScopedECSRegistrationSuppression suppressECS;
	ResourceAllocationFeatureInfo allocation{};
	const bool deviceLocal = DeviceManager::GetInstance().GetDevice().QueryFeatureInfo(&allocation.header) == rhi::Result::Ok && allocation.gpuUploadHeapSupported;
	m_buffer = Buffer::CreateUnmaterializedStructuredBuffer(static_cast<uint32_t>(bytes / 4), 4, false, false, false,
		deviceLocal ? rhi::HeapType::GPUUpload : rhi::HeapType::Upload);
	m_buffer->SetName(m_name);
	m_buffer->Materialize();
	void* mapped = nullptr;
	m_buffer->GetAPIResource().Map(&mapped, 0, bytes);
	if (!mapped) throw std::runtime_error("Latch block '" + m_name + "' could not be mapped");
	m_mapped = static_cast<std::byte*>(mapped);
	std::memset(m_mapped, 0, bytes);
	m_srvIndex = m_buffer->GetSRVInfo(0).slot.index;
}

std::shared_ptr<LatchBlock> LatchBlock::Create(std::string name, uint32_t bytesPerSlot, uint32_t slots) {
	return std::make_shared<LatchBlock>(std::move(name), bytesPerSlot, slots);
}

void LatchBlock::Write(uint32_t slot, uint32_t offset, std::span<const std::byte> bytes) const {
	if (uint64_t(offset) + bytes.size() > m_stride)
		throw std::out_of_range("Latch block '" + m_name + "' write past its slot");
	std::memcpy(Slot(slot).data() + offset, bytes.data(), bytes.size());
}

LatchBlock::~LatchBlock() {
	if (m_buffer && m_mapped) m_buffer->GetAPIResource().Unmap(0, uint64_t(m_stride) * m_slots);
	// A leased run's region belongs to its lease, which frees it when the run retires. The pinned ones are the block's: they
	// are reused once the GPU is past what was submitted so far.
	for (const auto& region : m_stable)
		if (region) m_arena->FreePinned(region);
}

LatchBlock::Region LatchBlock::At(const RunRef& run) const {
	if (m_buffer) return Region{m_buffer.get(), m_srvIndex, Offset(run.lease), m_mapped + Offset(run.lease), m_stride};
	std::scoped_lock lock(m_mutex);
	if (run.Stable()) {
		if (m_stable.size() <= run.lease) m_stable.resize(size_t{run.lease} + 1);
		auto& region = m_stable[run.lease];
		if (!region) region = m_arena->AllocatePinned(m_stride);
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

void LatchBlock::Write(const RunRef& run, uint32_t offset, std::span<const std::byte> bytes) const {
	if (uint64_t(offset) + bytes.size() > m_stride)
		throw std::out_of_range("Latch block '" + m_name + "' write past its region");
	const auto region = At(run);
	std::memcpy(region.mapped + offset, bytes.data(), bytes.size());
}

} // namespace org
