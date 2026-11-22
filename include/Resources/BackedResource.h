#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <rhi.h>

#include "Resources/RelocationRecord.h"


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
    // Borrowed synchronous callers need the immutable allocation metadata but
    // must not manufacture an ownership lease. Async publication requests the
    // default retained form.
    virtual BackingAllocationSnapshot CaptureBackingAllocation(bool /*retain*/ = true) { return {}; }
    // The lock that (de)materialization holds. A publication capture holds it across the allocation and the descriptor
    // views, so both come from one backing: another graph of the arena can re-place this resource on its own thread.
    virtual std::recursive_mutex* BackingMutex() noexcept { return nullptr; }
    // Resource-version publication. A stable backing generation returns the
    // same immutable token; this is allocation ownership, not graph caching.
    std::shared_ptr<const BackingAllocationSnapshot> CapturePublishedBackingAllocation() {
        const auto generation = GetBackingGeneration();
        std::scoped_lock lock(m_publicationMutex);
        if (m_publishedAllocation && m_publishedGeneration == generation)
            return m_publishedAllocation;
        auto snapshot = CaptureBackingAllocation(true);
        if (!snapshot) {
            m_publishedAllocation.reset();
            m_publishedGeneration = generation;
            return {};
        }
        m_publishedAllocation =
            std::make_shared<const BackingAllocationSnapshot>(std::move(snapshot));
        m_publishedGeneration = generation;
        return m_publishedAllocation;
    }
    // The relocation class of this resource's backings (relocation::Class), from now on and for every later backing.
    // An owner marks a resource ReadOnly once nothing writes it any more (a sealed resource or version). Unset, each
    // backing keeps the class DeviceManager gave it. Owner thread of the resource.
    void SetRelocationClass(relocation::Class value);
    std::optional<relocation::Class> GetRelocationClass() const noexcept {
        const auto value = m_relocationClass.load(std::memory_order_acquire);
        if (value == kNoRelocationClass) return std::nullopt;
        return static_cast<relocation::Class>(value);
    }
    // The owner record of the current backing's allocation. Null without a backing, or for a placed backing.
    relocation::RelocationRecord* CurrentRelocationRecord() noexcept;
    // Points the current backing's record at this resource (RelocationRecord::Owner), so the defragmenter can find it. A factory
    // calls it once the resource is owned by a shared_ptr, because a backing made in the constructor has no owner to point at.
    void NoteRelocationOwner();
protected:
    // The current backing's allocation. Null when there is none, or when the backing is placed in another allocation.
    // Called with BackingMutex held, when there is one.
    virtual rhi::ma::Allocation* RelocationAllocation() noexcept { return nullptr; }
    // Gives the current backing's record the class set with SetRelocationClass, if one was set, and the resource's ID.
    // Each owner calls it after it makes or adopts a backing, with BackingMutex held.
    void ApplyRelocationClass(uint64_t resourceID) noexcept;
    // A dematerialized resource lets go of its cached publication snapshot. The snapshot holds a lease on the old allocation, so
    // keeping it would keep the memory of a backing given back until the next publication, which a resource no graph uses never
    // gets. Publications already captured keep their own reference.
    void ForgetPublishedAllocation() {
        std::scoped_lock lock(m_publicationMutex);
        m_publishedAllocation.reset();
        m_publishedGeneration = UINT64_MAX;
    }
private:
    static constexpr uint8_t kNoRelocationClass = 0xFF;
    std::atomic<uint8_t> m_relocationClass{kNoRelocationClass};
    std::mutex m_publicationMutex;
    uint64_t m_publishedGeneration = UINT64_MAX;
    std::shared_ptr<const BackingAllocationSnapshot> m_publishedAllocation;
};


} // namespace org
