#pragma once

#include <cstdint>
#include <memory>
#include <rhi.h>

namespace org {
class DescriptorHeap;
struct ResourceBindingSnapshot;
struct BindlessViewRequest;
namespace runtime { class ResourceCleanupQueue; }

struct DescriptorBindingIdentity {
    uint64_t device = 0;
    uint64_t heap = 0;
    uint64_t allocation = 0;
    uint64_t binding = 0;
    bool operator==(const DescriptorBindingIdentity&) const = default;
};

// Copyable immutable CPU ownership. Index() is only for serialization into a
// binding table whose version retains this object, never for unowned caching.
class OwnedDescriptorBinding {
public:
    OwnedDescriptorBinding() = default;
    explicit operator bool() const noexcept { return static_cast<bool>(m_state); }
    uint32_t Index() const noexcept;
    rhi::DescriptorSlot Slot() const noexcept;
    DescriptorBindingIdentity Identity() const noexcept;
    std::shared_ptr<const void> Owner() const noexcept { return m_state; }

    // Captures an already-created immutable ORG view. Does not allocate,
    // rewrite, or take over retirement of its descriptor slot.
    static OwnedDescriptorBinding CaptureResourceView(
        std::shared_ptr<const ResourceBindingSnapshot> snapshot, const BindlessViewRequest& view,
        uint64_t deviceGeneration, std::shared_ptr<runtime::ResourceCleanupQueue> cleanup);
    static OwnedDescriptorBinding CaptureSampler(uint32_t index, uint64_t deviceGeneration,
        std::shared_ptr<DescriptorHeap> heap, std::shared_ptr<runtime::ResourceCleanupQueue> cleanup);

    static OwnedDescriptorBinding CreateShaderResourceView(rhi::Device device,
        std::shared_ptr<const void> deviceOwner, uint64_t deviceGeneration,
        std::shared_ptr<DescriptorHeap> heap, std::shared_ptr<runtime::ResourceCleanupQueue> cleanup,
        rhi::Resource resource, std::shared_ptr<const void> backingOwner, const rhi::SrvDesc& description);
    static OwnedDescriptorBinding CreateSampler(rhi::Device device,
        std::shared_ptr<const void> deviceOwner, uint64_t deviceGeneration,
        std::shared_ptr<DescriptorHeap> heap, std::shared_ptr<runtime::ResourceCleanupQueue> cleanup,
        const rhi::SamplerDesc& description);
private:
    struct State;
    std::shared_ptr<const State> m_state;
};
} // namespace org
