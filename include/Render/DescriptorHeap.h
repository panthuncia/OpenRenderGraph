#pragma once

#ifdef _WIN32
// Consumers use Microsoft::WRL::ComPtr through this header.
#include <wrl/client.h>
#endif
#include <rhi.h>
#include <queue>
#include <vector>
#include <mutex>
#include <memory>


namespace org {

class DescriptorHeap : public std::enable_shared_from_this<DescriptorHeap> {
public:
    DescriptorHeap(rhi::Device& device, rhi::DescriptorHeapType type, uint32_t numDescriptors, bool shaderVisible = false, std::string name = "Descriptor Heap");
    ~DescriptorHeap();

    // Non-copyable and non-movable
    DescriptorHeap(const DescriptorHeap&) = delete;
    DescriptorHeap& operator=(const DescriptorHeap&) = delete;

    rhi::DescriptorHeap GetHeap();

    UINT AllocateDescriptor();
    void ReleaseDescriptor(UINT index, std::shared_ptr<const void> owner = {});
    // CPU frames retain slots before they have submission fences. Retirement
    // requests reclamation; a retained slot cannot return to the free list.
    std::shared_ptr<const void> CaptureDescriptorLease(UINT index);

private:
    rhi::DescriptorHeapPtr m_heap;
    UINT m_descriptorSize;
    UINT m_numDescriptorsAllocated;
    uint32_t m_totalSize;
    std::queue<UINT> m_freeIndices;
    struct SlotState {
        bool allocated = false;
        bool awaitingLease = false;
        uint64_t leaseGeneration = 0;
        std::weak_ptr<const void> lease;
        std::shared_ptr<const void> retiredOwner;
    };
    struct Lease;
    void ReleaseLease(UINT index, uint64_t generation);
    std::vector<SlotState> m_slots;
    rhi::DescriptorHeapType m_type;
    bool m_shaderVisible;
    std::mutex m_allocationMutex;
};


} // namespace org
