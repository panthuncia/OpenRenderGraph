#pragma once

#include <memory>

#include <rhi_allocator.h>

#include "Resources/RelocationRecord.h"

// The one place ORG frees allocator memory it owns: tracked allocations (TrackedHandle), resources a host owns
// (memory::AllocatedResource) and old ranges the defragmenter kept (relocation::Defragmenter).
//
// A defragmentation pass names allocations by raw pointer between BeginPass and EndPass. The allocator does not lock them in that
// time, so a free on another thread would leave the pass with a freed allocation. While a ReleaseHold is alive, frees wait in a
// list and run when the last hold ends.
namespace org::relocation {

// Any thread. Frees the allocation, then its owner record. While a hold is alive, both wait until the last hold ends.
void ReleaseAllocation(rhi::ma::AllocationPtr allocation, std::unique_ptr<RelocationRecord> record = {}) noexcept;

// Holds frees off while it lives. Its constructor waits for frees that already started to finish, so no free runs while a hold is
// alive. Holds may nest. Do not take one inside a free.
class ReleaseHold {
public:
    ReleaseHold();
    ~ReleaseHold();
    ReleaseHold(const ReleaseHold&) = delete;
    ReleaseHold& operator=(const ReleaseHold&) = delete;
};

// Any thread: the frees waiting for the holds to end, and their bytes.
struct ParkedReleases {
    uint64_t count = 0;
    uint64_t bytes = 0;
};
ParkedReleases GetParkedReleases() noexcept;

} // namespace org::relocation
