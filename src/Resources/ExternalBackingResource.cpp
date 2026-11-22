#include "Resources/ExternalBackingResource.h"

#include <mutex>

#include "Resources/GPUBacking/GpuBufferBacking.h"


namespace org {

struct ExternalBackingResource::Impl {
    explicit Impl(std::unique_ptr<GpuBufferBacking> inBacking)
        : backing(std::move(inBacking)) {
    }

    std::unique_ptr<GpuBufferBacking> backing;
    // Moves with each relocation, so the RotationKey of the resource changes with its backing.
    uint64_t generation = 1;
    std::recursive_mutex mutex;
};

std::shared_ptr<ExternalBackingResource> ExternalBackingResource::CreateShared(std::unique_ptr<GpuBufferBacking> backing)
{
    return std::shared_ptr<ExternalBackingResource>(
        new ExternalBackingResource(std::move(backing))
    );
}

rhi::Resource ExternalBackingResource::GetAPIResource() {
    return m_impl->backing->GetAPIResource();
}

rhi::BarrierBatch ExternalBackingResource::GetEnhancedBarrierGroup(RangeSpec r,
    rhi::ResourceAccessType prevA, rhi::ResourceAccessType newA,
    rhi::ResourceLayout prevL, rhi::ResourceLayout newL,
    rhi::ResourceSyncState prevS, rhi::ResourceSyncState newS)
{
    return m_impl->backing->GetEnhancedBarrierGroup(r, prevA, newA, prevL, newL, prevS, newS);
}

SymbolicTracker* ExternalBackingResource::GetStateTracker() {
    return m_impl->backing->GetStateTracker();
}

bool ExternalBackingResource::TryGetBufferByteSize(uint64_t& outByteSize) const {
    if (!m_impl || !m_impl->backing) {
        return false;
    }

    outByteSize = static_cast<uint64_t>(m_impl->backing->GetSize());
    return true;
}

bool ExternalBackingResource::IsMaterialized() const {
    return m_impl && m_impl->backing != nullptr;
}

uint64_t ExternalBackingResource::GetBackingGeneration() const {
    return IsMaterialized() ? m_impl->generation : 0u;
}

std::recursive_mutex* ExternalBackingResource::BackingMutex() noexcept {
    return &m_impl->mutex;
}

bool ExternalBackingResource::AdoptRelocatedBacking(std::unique_ptr<GpuBufferBacking>& next, uint64_t fromGeneration) {
    std::scoped_lock lock(m_impl->mutex);
    if (!m_impl->backing || m_impl->generation != fromGeneration) return false;
    if (next && next->GetSize() != m_impl->backing->GetSize()) return false;
    if (next) std::swap(m_impl->backing, next);
    ApplyRelocationClass(GetGlobalResourceID());
    ++m_impl->generation;
    ForgetPublishedAllocation();
    return true;
}

std::unique_ptr<GpuBufferBacking> ExternalBackingResource::ReleaseBacking() {
    std::scoped_lock lock(m_impl->mutex);
    ++m_impl->generation;
    ForgetPublishedAllocation();
    return std::move(m_impl->backing);
}

rhi::ma::Allocation* ExternalBackingResource::RelocationAllocation() noexcept {
    std::scoped_lock lock(m_impl->mutex);
    return m_impl->backing ? m_impl->backing->GetAllocation() : nullptr;
}

void ExternalBackingResource::EnsureVirtualDescriptorSlotsAllocated() {
}

BackingAllocationSnapshot ExternalBackingResource::CaptureBackingAllocation(bool retain) {
    std::scoped_lock lock(m_impl->mutex);
    if (!IsMaterialized() || GetAttachedAPIRepresentation(BackendInstanceId::Primary).IsValid()) return {};
    auto lease = retain ? m_impl->backing->CaptureAllocationLease() : std::shared_ptr<const void>{};
    if (retain && !lease) return {};
    return {GetGlobalResourceID(), GetBackingGeneration(), GetAPIResource(), std::move(lease), m_impl->backing->GetSize()};
}

ExternalBackingResource::ExternalBackingResource(std::unique_ptr<GpuBufferBacking> backing)
    : m_impl(std::make_unique<Impl>(std::move(backing)))
{
    m_hasLayout = false;
    m_mipLevels = 1;
    m_arraySize = 1;
}

ExternalBackingResource::~ExternalBackingResource() = default;


} // namespace org
