#include "Render/OwnedDescriptorBinding.h"

#include <atomic>
#include <stdexcept>
#include "Render/DescriptorHeap.h"
#include "Render/PublicationBindingBundle.h"
#include "Render/Runtime/ResourceCleanupQueue.h"

namespace org {
namespace { std::atomic<uint64_t> nextBinding{1}; }
struct OwnedDescriptorBinding::State {
    // Destruction order: descriptor allocation, resource backing, heap, device.
    std::shared_ptr<const void> deviceOwner;
    std::shared_ptr<DescriptorHeap> heap;
    std::shared_ptr<const void> backingOwner;
    std::shared_ptr<const void> leaseOwner;
    rhi::DescriptorSlot slot{};
    DescriptorBindingIdentity identity{};
    bool allocated = false;
    ~State() { if (allocated) heap->ReleaseDescriptor(slot.index, std::move(leaseOwner)); }
};
uint32_t OwnedDescriptorBinding::Index() const noexcept { return m_state ? m_state->slot.index : UINT32_MAX; }
rhi::DescriptorSlot OwnedDescriptorBinding::Slot() const noexcept { return m_state ? m_state->slot : rhi::DescriptorSlot{}; }
DescriptorBindingIdentity OwnedDescriptorBinding::Identity() const noexcept { return m_state ? m_state->identity : DescriptorBindingIdentity{}; }

OwnedDescriptorBinding OwnedDescriptorBinding::CaptureResourceView(
    std::shared_ptr<const ResourceBindingSnapshot> snapshot, const BindlessViewRequest& view,
    uint64_t deviceGeneration, std::shared_ptr<runtime::ResourceCleanupQueue> cleanup) {
    if (!snapshot || !snapshot->allocationOwner || !snapshot->descriptorOwner ||
        !snapshot->views || !snapshot->resource.GetHandle().valid() || !deviceGeneration || !cleanup)
        throw std::invalid_argument("Captured binding requires an owned immutable resource view");
    const auto slot = snapshot->views->Resolve(view);
    if (!slot.heap.valid() || slot.index == UINT32_MAX)
        throw std::invalid_argument("Captured binding has no descriptor slot");
    auto state = cleanup->Make<State>();
    state->slot = slot;
    state->identity = {deviceGeneration, (uint64_t{slot.heap.generation} << 32) | slot.heap.index,
        snapshot->backingGeneration, nextBinding.fetch_add(1, std::memory_order_relaxed)};
    // The snapshot owns both the allocation and descriptor lease; unlike the
    // factory path this state must never call ReleaseDescriptor itself.
    state->backingOwner = std::move(snapshot);
    OwnedDescriptorBinding result;
    result.m_state = std::move(state);
    return result;
}

OwnedDescriptorBinding OwnedDescriptorBinding::CaptureSampler(uint32_t index, uint64_t deviceGeneration,
    std::shared_ptr<DescriptorHeap> heap, std::shared_ptr<runtime::ResourceCleanupQueue> cleanup) {
    if (!deviceGeneration || !heap || !cleanup || index == UINT32_MAX)
        throw std::invalid_argument("Captured sampler requires heap and cleanup ownership");
    auto state = cleanup->Make<State>();
    state->heap = std::move(heap);
    state->slot = {state->heap->GetHeap().GetHandle(), index};
    state->leaseOwner = state->heap->CaptureDescriptorLease(index);
    const auto handle = state->slot.heap;
    const auto binding = nextBinding.fetch_add(1, std::memory_order_relaxed);
    state->identity = {deviceGeneration, (uint64_t{handle.generation} << 32) | handle.index, binding, binding};
    OwnedDescriptorBinding result;
    result.m_state = std::move(state);
    return result;
}
OwnedDescriptorBinding OwnedDescriptorBinding::CreateShaderResourceView(rhi::Device device,
    std::shared_ptr<const void> deviceOwner, uint64_t deviceGeneration,
    std::shared_ptr<DescriptorHeap> heap, std::shared_ptr<runtime::ResourceCleanupQueue> cleanup,
    rhi::Resource resource, std::shared_ptr<const void> backingOwner, const rhi::SrvDesc& description) {
    if (!device || !deviceOwner || !deviceGeneration || !heap || !cleanup ||
        (resource.GetHandle().valid() && !backingOwner))
        throw std::invalid_argument("Owned SRV requires device, heap, cleanup and exact backing ownership");
    auto state = cleanup->Make<State>();
    state->deviceOwner = std::move(deviceOwner);
    state->heap = std::move(heap);
    state->backingOwner = std::move(backingOwner);
    state->leaseOwner = cleanup->Make<std::pair<std::shared_ptr<const void>, std::shared_ptr<const void>>>(
        state->deviceOwner, state->backingOwner);
    state->slot = {state->heap->GetHeap().GetHandle(), state->heap->AllocateDescriptor()};
    state->allocated = true;
    const auto handle = state->slot.heap;
    const auto binding = nextBinding.fetch_add(1, std::memory_order_relaxed);
    state->identity = {deviceGeneration, (uint64_t{handle.generation} << 32) | handle.index, binding, binding};
    if (device.CreateShaderResourceView(state->slot, resource.GetHandle(), description) != rhi::Result::Ok) return {};
    OwnedDescriptorBinding result;
    result.m_state = std::move(state);
    return result;
}
OwnedDescriptorBinding OwnedDescriptorBinding::CreateSampler(rhi::Device device,
    std::shared_ptr<const void> deviceOwner, uint64_t deviceGeneration,
    std::shared_ptr<DescriptorHeap> heap, std::shared_ptr<runtime::ResourceCleanupQueue> cleanup,
    const rhi::SamplerDesc& description) {
    if (!device || !deviceOwner || !deviceGeneration || !heap || !cleanup)
        throw std::invalid_argument("Owned sampler requires device, heap and cleanup ownership");
    auto state = cleanup->Make<State>();
    state->deviceOwner = std::move(deviceOwner);
    state->heap = std::move(heap);
    state->leaseOwner = cleanup->Make<std::shared_ptr<const void>>(state->deviceOwner);
    state->slot = {state->heap->GetHeap().GetHandle(), state->heap->AllocateDescriptor()};
    state->allocated = true;
    const auto handle = state->slot.heap;
    const auto binding = nextBinding.fetch_add(1, std::memory_order_relaxed);
    state->identity = {deviceGeneration, (uint64_t{handle.generation} << 32) | handle.index, binding, binding};
    if (device.CreateSampler(state->slot, description) != rhi::Result::Ok) return {};
    OwnedDescriptorBinding result;
    result.m_state = std::move(state);
    return result;
}
} // namespace org
