#pragma once

#include <cstdint>

// Graph-facing contract for resources whose API handle is backed by a concrete GPU allocation.
// RenderGraph should depend on this capability instead of specific resource subclasses.
class BackedResource {
public:
    virtual ~BackedResource() = default;

    virtual bool IsMaterialized() const = 0;
    virtual uint64_t GetBackingGeneration() const = 0;
    virtual void EnsureVirtualDescriptorSlotsAllocated() = 0;
};
