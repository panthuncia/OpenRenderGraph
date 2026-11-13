// org::memory: ORG's device allocator for resources a host owns itself, and the allocator's VRAM report.
#include "OpenRenderGraph/DeviceMemory.h"

#include <utility>

#include "Managers/Singletons/DeviceManager.h"

namespace org::memory {

AllocatedResource::AllocatedResource(rhi::ma::AllocationPtr allocation) noexcept
    : m_allocation(std::move(allocation))
{
}

rhi::Resource& AllocatedResource::Resource() noexcept
{
    return m_allocation->GetResource();
}

rhi::ma::AllocationPlacement AllocatedResource::Placement() const noexcept
{
    return m_allocation->GetPlacement();
}

uint64_t AllocatedResource::Size() const noexcept
{
    return m_allocation->GetSize();
}

rhi::Result CreateAllocatedResource(const rhi::ResourceDesc& resourceDesc, const rhi::ma::AllocationDesc& allocationDesc,
    std::shared_ptr<AllocatedResource>& out) noexcept
{
    out.reset();
    rhi::ma::AllocationPtr allocation;
    const auto result = DeviceManager::GetInstance().CreateResourceUntracked(allocationDesc, resourceDesc, allocation);
    if (result != rhi::Result::Ok) {
        return result;
    }
    if (!allocation) {
        return rhi::Result::Failed;
    }
    try {
        out = std::make_shared<AllocatedResource>(std::move(allocation));
    } catch (...) {
        return rhi::Result::OutOfMemory;
    }
    return rhi::Result::Ok;
}

std::vector<rhi::ma::OwnerMemoryReport> OwnerReport()
{
    std::vector<rhi::ma::OwnerMemoryReport> rows;
    auto* allocator = DeviceManager::GetInstance().GetAllocator();
    if (!allocator) {
        return rows;
    }
    // Owners can be added between the count and the copy: grow until the copy fits.
    uint32_t capacity = 32;
    for (;;) {
        rows.resize(capacity);
        const uint32_t count = allocator->GetOwnerReport(rows.data(), capacity);
        if (count <= capacity) {
            rows.resize(count);
            return rows;
        }
        capacity = count + 8;
    }
}

bool AllocatorBudget(rhi::ma::Budget& local, rhi::ma::Budget& nonLocal) noexcept
{
    auto* allocator = DeviceManager::GetInstance().GetAllocator();
    if (!allocator) {
        return false;
    }
    allocator->GetBudget(&local, &nonLocal);
    return true;
}

uint64_t MemoryCapacity(rhi::MemorySegmentGroup segmentGroup) noexcept
{
    auto* allocator = DeviceManager::GetInstance().GetAllocator();
    return allocator ? allocator->GetMemoryCapacity(segmentGroup) : 0;
}

} // namespace org::memory
