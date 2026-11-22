#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <rhi.h>

#include "Resources/BackedResource.h"
#include "Resources/Resource.h"


namespace org {

class GpuBufferBacking;

class ExternalBackingResource final : public Resource, public BackedResource {
public:
    static std::shared_ptr<ExternalBackingResource> CreateShared(std::unique_ptr<GpuBufferBacking> backing);

    rhi::Resource GetAPIResource() override;

    rhi::BarrierBatch GetEnhancedBarrierGroup(RangeSpec r,
        rhi::ResourceAccessType prevA, rhi::ResourceAccessType newA,
        rhi::ResourceLayout prevL, rhi::ResourceLayout newL,
        rhi::ResourceSyncState prevS, rhi::ResourceSyncState newS) override;

    SymbolicTracker* GetStateTracker() override;

    bool TryGetBufferByteSize(uint64_t& outByteSize) const override;

    bool IsMaterialized() const override;

    uint64_t GetBackingGeneration() const override;
    BackingAllocationSnapshot CaptureBackingAllocation(bool retain = true) override;
    std::recursive_mutex* BackingMutex() noexcept override;

    void EnsureVirtualDescriptorSlotsAllocated() override;

    // Relocation (relocation::Service). Switches to a new backing of the same size, as PixelBuffer::AdoptRelocatedBacking
    // does. The generation moves, so persistent executables see a new RotationKey. On success `next` holds the old backing
    // (or stays null when it was null). False when nothing changed.
    bool AdoptRelocatedBacking(std::unique_ptr<GpuBufferBacking>& next, uint64_t fromGeneration);
    // Gives up the backing, for a resource used only to carry a new backing through the relocation batch. The resource has
    // no backing afterwards.
    std::unique_ptr<GpuBufferBacking> ReleaseBacking();

    ~ExternalBackingResource() override;

protected:
    rhi::ma::Allocation* RelocationAllocation() noexcept override;

private:
    struct Impl;

    ExternalBackingResource(std::unique_ptr<GpuBufferBacking> backing);

    std::unique_ptr<Impl> m_impl;
};


} // namespace org
