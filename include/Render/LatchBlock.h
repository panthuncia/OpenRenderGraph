#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace org {

class Buffer;

// Late latching. Some per-execution values would otherwise be baked into a pass's recording: matrices, counts,
// dispatch sizes, frame stamps. Instead, they live in a small host-visible block with one region per frame slot.
// The pass records only where its region is. That location depends on the slot, not on the values. So the pass's
// invocation and its recorded commands stay the same from one execution to the next. This is what lets them be
// reused, or prepared before the values are known. The host writes the values into the slot just before submission.
//
// Contract:
//  - The host writes slot s only while no submitted work that reads slot s is incomplete. With a GraphHost, this
//    holds from the slot wait of the execution that uses s until that execution is submitted. The slot wait happens
//    inside ExecuteFrame, so the window starts at beforePrepare. The execution's slot is CurrentFrameSlot().
//  - Readers see the host writes made before the submission. The memory is host-coherent, and queue submission makes
//    the writes available.
//  - The graph's state tracking ignores the block. It is not a graph resource, and it needs no declaration, barriers
//    or transitions.
//  - Shaders read it as a ByteAddressBuffer through SrvIndex(), at Offset(slot). It is also valid as an
//    indirect-argument buffer (Resource()) and as a copy source.
class LatchBlock {
public:
    // bytesPerSlot is rounded up to 256, so every region is aligned for any use. The buffer is registered with the
    // host's world, like any resource made on the owner's thread. Use Create to make a block on any thread.
    LatchBlock(std::string name, uint32_t bytesPerSlot, uint32_t slots);
    // Callable from any thread. Makes a block without an ECS entity (Resource::ScopedECSRegistrationSuppression), the
    // same way a versioned buffer's pending version is made (VersionedBuffer::MakeStructured). The block is not a graph
    // resource, since nothing declares it. So nothing in the host's world needs it, and a producer's thread may not touch
    // that world. Its last reference may drop on any thread. Its buffer then goes through ORG's deletion path like any
    // other.
    static std::shared_ptr<LatchBlock> Create(std::string name, uint32_t bytesPerSlot, uint32_t slots);
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
