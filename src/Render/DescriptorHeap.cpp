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
    // Reclaim CPU-retained retirements only when another free slot is needed.
    // No GPU polling or second retirement queue is introduced here: callers
    // request ReleaseDescriptor only after their fence requirements are met.
    if (m_freeIndices.empty()) {
        std::erase_if(m_retainedRetiredIndices, [&](UINT index) {
            if (!m_slots[index].lease.expired()) return false;
            m_freeIndices.push(index);
            return true;
        });
    }
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

std::shared_ptr<const void> DescriptorHeap::CaptureDescriptorLease(UINT index) {
    std::lock_guard lock(m_allocationMutex);
    if (index >= m_slots.size() || !m_slots[index].allocated)
        throw std::logic_error("Cannot capture an unallocated or retired descriptor slot");
    auto& slot = m_slots[index];
    if (auto existing = slot.lease.lock()) return existing;
    // A distinct control block per slot; aliasing the heap's control block
    // would make the lease live for the entire heap lifetime.
    auto lease = std::make_shared<std::shared_ptr<DescriptorHeap>>(shared_from_this());
    slot.lease = lease;
    return lease;
}

void DescriptorHeap::ReleaseDescriptor(UINT index) {
    std::lock_guard lock(m_allocationMutex);
    if (index >= m_slots.size() || !m_slots[index].allocated)
        throw std::logic_error("Descriptor slot released more than once or out of range");
    auto& slot = m_slots[index];
    slot.allocated = false;
    if (slot.lease.expired()) m_freeIndices.push(index);
    else m_retainedRetiredIndices.push_back(index);
}



} // namespace org
