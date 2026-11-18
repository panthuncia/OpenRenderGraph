#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <rhi.h>
#include <rhi_allocator.h>
#include "Resources/TrackedAllocation.h"


namespace org {

// The memory of an alias pool shared between a D3D12 and a Vulkan device: a D3D12 heap created shared, and its import on the
// Vulkan device. A resource placed at one offset of both is one allocation seen by both devices.
struct SharedAliasHeaps {
    rhi::HeapPtr d3d12, vulkan;
    // Backend instance of each device (BackendInstanceId values).
    uint8_t d3d12Device = 0, vulkanDevice = 0;
};

// A lifetime reference to an immutable allocation generation. It grants no
// exclusive ownership of bytes; admission still orders overlapping GPU access.
class AliasHeapGeneration {
public:
    static std::shared_ptr<const AliasHeapGeneration> Capture(TrackedHandle& allocation, uint64_t generation) {
        if (!allocation.GetAllocation()) throw std::invalid_argument("Missing alias heap allocation");
        auto owner = allocation.CaptureAllocationLease();
        return std::shared_ptr<const AliasHeapGeneration>(new AliasHeapGeneration(
            std::move(owner), allocation.GetAllocation(), nullptr, generation));
    }
    static std::shared_ptr<const AliasHeapGeneration> CaptureShared(std::shared_ptr<const SharedAliasHeaps> heaps, uint64_t generation) {
        if (!heaps || !heaps->d3d12 || !heaps->vulkan) throw std::invalid_argument("Missing shared alias heaps");
        const auto* shared = heaps.get();
        return std::shared_ptr<const AliasHeapGeneration>(new AliasHeapGeneration(std::move(heaps), nullptr, shared, generation));
    }
    // The primary device's allocation. A shared generation has none: it places through Shared().
    rhi::ma::Allocation& Allocation() const {
        if (!m_allocation) throw std::logic_error("A shared alias heap generation has no primary-device allocation");
        return *m_allocation;
    }
    const SharedAliasHeaps* Shared() const noexcept { return m_shared; }
    uint64_t Generation() const { return m_generation; }
private:
    AliasHeapGeneration(std::shared_ptr<const void> owner, rhi::ma::Allocation* allocation, const SharedAliasHeaps* shared,
        uint64_t generation)
        : m_owner(std::move(owner)), m_allocation(allocation), m_shared(shared), m_generation(generation) {}
    std::shared_ptr<const void> m_owner;
    rhi::ma::Allocation* m_allocation;
    const SharedAliasHeaps* m_shared;
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
