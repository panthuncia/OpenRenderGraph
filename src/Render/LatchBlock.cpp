#include "Render/LatchBlock.h"

#include <cstring>
#include <stdexcept>

#include "Managers/Singletons/DeviceManager.h"
#include "Resources/Buffers/Buffer.h"

#include <rhi_feature_info.h>
#include <spdlog/spdlog.h>

namespace org {

namespace {
// Device-local where the host can write it (ReBAR, D3D12's GPU upload heap): every dispatch reading its region reads video memory,
// not the host's over the bus. Else the host's memory.
rhi::HeapType LatchHeap() {
    static const rhi::HeapType heap = [] {
        ResourceAllocationFeatureInfo allocation{};
        const auto device = DeviceManager::GetInstance().GetDevice();
        const bool deviceLocal = device.QueryFeatureInfo(&allocation.header) == rhi::Result::Ok && allocation.gpuUploadHeapSupported;
        spdlog::info("[ORG] latch blocks in {} memory", deviceLocal ? "device-local host-visible" : "host");
        return deviceLocal ? rhi::HeapType::GPUUpload : rhi::HeapType::Upload;
    }();
    return heap;
}
}

LatchBlock::LatchBlock(std::string name, uint32_t bytesPerSlot, uint32_t slots)
    : m_name(std::move(name)), m_stride((bytesPerSlot + 255u) & ~255u), m_slots(slots) {
    if (!bytesPerSlot || !slots) throw std::invalid_argument("Latch block '" + m_name + "' needs a size and slots");
    // Words, read as a ByteAddressBuffer; host-visible and coherent, written by the host every execution (never read back: device-local
    // host-visible memory is write-combined).
    const uint64_t bytes = uint64_t(m_stride) * m_slots;
    m_buffer = Buffer::CreateUnmaterializedStructuredBuffer(static_cast<uint32_t>(bytes / 4), 4, false, false, false, LatchHeap());
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
    Resource::ScopedECSRegistrationSuppression suppressECS;
    return std::make_shared<LatchBlock>(std::move(name), bytesPerSlot, slots);
}

LatchBlock::~LatchBlock() {
    if (m_buffer && m_mapped) m_buffer->GetAPIResource().Unmap(0, uint64_t(m_stride) * m_slots);
}

void LatchBlock::Write(uint32_t slot, uint32_t offset, std::span<const std::byte> bytes) const {
    if (uint64_t(offset) + bytes.size() > m_stride)
        throw std::out_of_range("Latch block '" + m_name + "' write past its slot");
    std::memcpy(m_mapped + Offset(slot) + offset, bytes.data(), bytes.size());
}

} // namespace org
