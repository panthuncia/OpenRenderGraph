#pragma once

#include <optional>
#include <rhi_allocator.h>
#include "Resources/TrackedAllocation.h"


namespace org {

// A lifetime reference to an immutable allocation generation. It grants no
// exclusive ownership of bytes; admission still orders overlapping GPU access.
class AliasHeapGeneration {
public:
    static std::shared_ptr<const AliasHeapGeneration> Capture(TrackedHandle& allocation, uint64_t generation) {
        if (!allocation.GetAllocation()) throw std::invalid_argument("Missing alias heap allocation");
        auto owner = allocation.CaptureAllocationLease();
        // Obtain the pointer AFTER capture has moved ownership to stable storage.
        return std::shared_ptr<const AliasHeapGeneration>(new AliasHeapGeneration(
            std::move(owner), allocation.GetAllocation(), generation));
    }
    rhi::ma::Allocation& Allocation() const { return *m_allocation; }
    uint64_t Generation() const { return m_generation; }
private:
    AliasHeapGeneration(std::shared_ptr<const TrackedHandle> owner, rhi::ma::Allocation* allocation, uint64_t generation)
        : m_owner(std::move(owner)), m_allocation(allocation), m_generation(generation) {}
    std::shared_ptr<const TrackedHandle> m_owner;
    rhi::ma::Allocation* m_allocation;
    uint64_t m_generation;
};

struct TextureAliasPlacement {
	std::shared_ptr<const AliasHeapGeneration> heap;
	uint64_t offset = 0;
	std::optional<uint64_t> poolID;
};

struct BufferAliasPlacement {
	std::shared_ptr<const AliasHeapGeneration> heap;
	uint64_t offset = 0;
	std::optional<uint64_t> poolID;
};


} // namespace org
