#include "Render/LatchBlock.h"

#include <cstring>
#include <stdexcept>

#include "Resources/Buffers/Buffer.h"

namespace org {

LatchBlock::LatchBlock(std::string name, uint32_t bytesPerSlot, uint32_t slots)
    : m_name(std::move(name)), m_stride((bytesPerSlot + 255u) & ~255u), m_slots(slots) {
    if (!bytesPerSlot || !slots) throw std::invalid_argument("Latch block '" + m_name + "' needs a size and slots");
    // Words, read as a ByteAddressBuffer; host-visible and coherent, written by the host every execution.
    const uint64_t bytes = uint64_t(m_stride) * m_slots;
    m_buffer = Buffer::CreateUnmaterializedStructuredBuffer(static_cast<uint32_t>(bytes / 4), 4, false, false, false, rhi::HeapType::Upload);
    m_buffer->SetName(m_name);
    m_buffer->Materialize();
    void* mapped = nullptr;
    m_buffer->GetAPIResource().Map(&mapped, 0, bytes);
    if (!mapped) throw std::runtime_error("Latch block '" + m_name + "' could not be mapped");
    m_mapped = static_cast<std::byte*>(mapped);
    std::memset(m_mapped, 0, bytes);
    m_srvIndex = m_buffer->GetSRVInfo(0).slot.index;
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
