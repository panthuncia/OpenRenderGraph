#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <rhi.h>
#include <rhi_allocator.h>

// ORG's device allocator for a host: resources the host owns itself, and the allocator's VRAM report.
namespace org::memory {

// One resource made through ORG's allocator outside any graph resource: an image the host hands to another API, or a staging
// buffer a worker fills. The memory and the resource go with the last reference, so the host keeps it until the GPU is done.
class AllocatedResource {
public:
    explicit AllocatedResource(rhi::ma::AllocationPtr allocation) noexcept;
    AllocatedResource(const AllocatedResource&) = delete;
    AllocatedResource& operator=(const AllocatedResource&) = delete;

    // The resource. Valid while this object lives.
    rhi::Resource& Resource() noexcept;
    // Where the memory landed: video or system memory, and whether a spillable allocation spilled.
    rhi::ma::AllocationPlacement Placement() const noexcept;
    // The allocation's size in bytes.
    uint64_t Size() const noexcept;

private:
    rhi::ma::AllocationPtr m_allocation;
};

// Any thread. Creates a resource through ORG's allocator. allocationDesc gives the heap, the flags (AllocationFlagSpillable) and
// the owner label for the report. InvalidCall before the allocator exists (the device is not initialized yet).
rhi::Result CreateAllocatedResource(const rhi::ResourceDesc& resourceDesc, const rhi::ma::AllocationDesc& allocationDesc,
    std::shared_ptr<AllocatedResource>& out) noexcept;

// Any thread. One row per owner label of ORG's allocator (rhi::ma::Allocator::GetOwnerReport). Empty without an allocator.
// The rows' owner strings stay valid while the allocator lives.
std::vector<rhi::ma::OwnerMemoryReport> OwnerReport();

// Any thread. The allocator's figures for video (local) and system (nonLocal) memory. False without an allocator.
bool AllocatorBudget(rhi::ma::Budget& local, rhi::ma::Budget& nonLocal) noexcept;

// Any thread. The size of the memory a segment group's heaps hold (the card's VRAM for Local). 0 without an allocator.
uint64_t MemoryCapacity(rhi::MemorySegmentGroup segmentGroup) noexcept;

} // namespace org::memory
