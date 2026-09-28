#include "Render/DescriptorHeap.h"
#include <mutex>
#include <algorithm>


namespace org {

DescriptorHeap::DescriptorHeap(rhi::Device& device, rhi::DescriptorHeapType type, uint32_t numDescriptors, bool shaderVisible, std::string name)
    : m_type(type), m_shaderVisible(shaderVisible), m_numDescriptorsAllocated(0) {

	rhi::DescriptorHeapDesc heapDesc = {.type = type, .capacity = numDescriptors, .shaderVisible = shaderVisible, .debugName = name.c_str()};
    auto result = device.CreateDescriptorHeap(heapDesc, m_heap);
    m_heap->SetName(name.c_str());

    m_descriptorSize = device.GetDescriptorHandleIncrementSize(type);
    m_totalSize = numDescriptors;
}

DescriptorHeap::~DescriptorHeap() {
}

rhi::DescriptorHeap DescriptorHeap::GetHeap() {
    return m_heap.Get();
}

UINT DescriptorHeap::AllocateDescriptor() {
    std::lock_guard lock(m_allocationMutex);
    UINT index;
    if (!m_freeIndices.empty()) {
        index = m_freeIndices.front();
        m_freeIndices.pop();
    } else if (m_numDescriptorsAllocated < m_totalSize) {
        index = m_numDescriptorsAllocated++;
        m_slots.emplace_back();
    } else {
        throw std::runtime_error("Out of descriptor heap space (including retained frame slots)");
    }
    m_slots[index].allocated = true;
    return index;
}

struct DescriptorHeap::Lease {
    std::shared_ptr<DescriptorHeap> heap;
    UINT index;
    uint64_t generation;
    ~Lease() { heap->ReleaseLease(index, generation); }
};

std::shared_ptr<const void> DescriptorHeap::CaptureDescriptorLease(UINT index) {
    std::lock_guard lock(m_allocationMutex);
    if (index >= m_slots.size() || !m_slots[index].allocated)
        throw std::logic_error("Cannot capture an unallocated or retired descriptor slot");
    auto& slot = m_slots[index];
    if (auto existing = slot.lease.lock()) return existing;
    // A distinct control block per slot; aliasing the heap's control block
    // would make the lease live for the entire heap lifetime.
    auto lease = std::make_shared<Lease>(shared_from_this(), index, ++slot.leaseGeneration);
    slot.lease = lease;
    return lease;
}

void DescriptorHeap::ReleaseDescriptor(UINT index, std::shared_ptr<const void> owner) {
    std::lock_guard lock(m_allocationMutex);
    if (index >= m_slots.size() || !m_slots[index].allocated)
        throw std::logic_error("Descriptor slot released more than once or out of range");
    auto& slot = m_slots[index];
    slot.allocated = false;
    if (slot.lease.expired()) m_freeIndices.push(index);
    else {
        slot.awaitingLease = true;
        slot.retiredOwner = std::move(owner);
    }
}

void DescriptorHeap::ReleaseLease(UINT index, uint64_t generation) {
    std::shared_ptr<const void> owner;
    {
        std::lock_guard lock(m_allocationMutex);
        auto& slot = m_slots[index];
        // A final-reference destructor can wait on this mutex while a new
        // capture creates another lease. The old callback cannot reclaim it.
        if (slot.leaseGeneration != generation || !slot.awaitingLease) return;
        slot.awaitingLease = false;
        owner = std::move(slot.retiredOwner);
        m_freeIndices.push(index);
    }
    // Releasing an imported owner may retire other resources or descriptors.
    // Do that outside the heap lock.
}


} // namespace org
