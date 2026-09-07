#pragma once

#include <cstdint>
#include <memory>
#include <rhi.h>


// Graph-facing contract for resources whose API handle is backed by a concrete GPU allocation.
// RenderGraph should depend on this capability instead of specific resource subclasses.
namespace org {

class AliasHeapGeneration;

// Allocation ownership only: never implies descriptor-slot or alias-range
// ownership. Unsupported/imported/aliased backings return an empty snapshot.
struct BackingAllocationSnapshot {
    uint64_t resourceID = 0, generation = 0;
    rhi::Resource resource;
    std::shared_ptr<const void> lease;
    uint64_t bufferByteSize = 0; // Zero for textures/unknown size.
    // Populated only for placed resources. This is lifetime and physical
    // identity metadata for ordered admission; it does not reserve the range.
    std::shared_ptr<const AliasHeapGeneration> aliasHeap;
    uint64_t aliasPoolID = 0;
    uint64_t aliasOffset = 0;
    uint64_t aliasSize = 0;
    explicit operator bool() const noexcept { return resource && lease; }
};

class BackedResource {
public:
    virtual ~BackedResource() = default;

    virtual bool IsMaterialized() const = 0;
    virtual uint64_t GetBackingGeneration() const = 0;
    virtual void EnsureVirtualDescriptorSlotsAllocated() = 0;
    // Capture and backing mutation are serialized by the preparation owner.
    virtual BackingAllocationSnapshot CaptureBackingAllocation() { return {}; }
};


} // namespace org
