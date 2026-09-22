#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace org {

class Buffer;

// Late latching: the per-execution values a pass would otherwise bake into its recording (matrices,
// counts, dispatch sizes, frame stamps) live in a small host-visible block with one region per frame
// slot, and the pass records only where its region is. That location is a function of the slot, not of
// the values, so the pass's invocation and its recorded commands stay the same from one execution to the
// next - which is what lets them be reused, or prepared ahead of the values - while the host writes the
// values into the slot just before submission.
//
// Contract:
//  - The host writes slot s only while no submitted work that reads slot s is incomplete. With a
//    PersistentGraphHost that holds from the slot wait of the execution using s (inside ExecuteFrame, so
//    from beforePrepare on) until that execution is submitted; its slot is CurrentFrameSlot().
//  - Readers see host writes made before the submission (host-coherent memory; queue submission makes
//    them available). Nothing about the block goes through the graph's state tracking: it is not a graph
//    resource and needs no declaration, barriers or transitions.
//  - Shaders read it as a ByteAddressBuffer through SrvIndex(), at Offset(slot); it is also valid as an
//    indirect-argument buffer (Resource()) and as a copy source.
class LatchBlock {
public:
    // bytesPerSlot is rounded up to 256 so every region is aligned for any use.
    LatchBlock(std::string name, uint32_t bytesPerSlot, uint32_t slots);
    ~LatchBlock();
    LatchBlock(const LatchBlock&) = delete;
    LatchBlock& operator=(const LatchBlock&) = delete;

    uint32_t Stride() const noexcept { return m_stride; }
    uint32_t Slots() const noexcept { return m_slots; }
    uint64_t Offset(uint32_t slot) const noexcept { return uint64_t(slot % m_slots) * m_stride; }
    uint32_t SrvIndex() const noexcept { return m_srvIndex; }
    const std::shared_ptr<Buffer>& Resource() const noexcept { return m_buffer; }

    // The slot's region, persistently mapped.
    std::span<std::byte> Slot(uint32_t slot) const noexcept {
        return {m_mapped + Offset(slot), m_stride};
    }
    // Copies bytes to the slot's region at offset; throws if they do not fit.
    void Write(uint32_t slot, uint32_t offset, std::span<const std::byte> bytes) const;
    template<class T> void WriteValue(uint32_t slot, uint32_t offset, const T& value) const {
        Write(slot, offset, std::as_bytes(std::span<const T, 1>(&value, 1)));
    }

private:
    std::string m_name;
    uint32_t m_stride = 0;
    uint32_t m_slots = 0;
    uint32_t m_srvIndex = 0;
    std::shared_ptr<Buffer> m_buffer;
    std::byte* m_mapped = nullptr;
};

} // namespace org
