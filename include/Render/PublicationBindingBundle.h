#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>
#include <iosfwd>
#include <rhi.h>
#include "Render/BindlessResourceViews.h"

namespace org {
class Resource;
class AliasHeapGeneration;

// Exact recording version. Incoming states and timeline waits are deliberately
// absent: those belong to an execution, not a publication.
struct ResourceBindingSnapshot {
    uint64_t resourceID = 0, backingGeneration = 0;
    rhi::Resource resource;
    rhi::ResourceDesc description{};
    std::shared_ptr<const BindlessResourceViews> views;
    std::shared_ptr<const void> allocationOwner, descriptorOwner;
    std::shared_ptr<const void> recordingOwner;
    std::shared_ptr<const void> semanticConsumer;
    std::shared_ptr<const AliasHeapGeneration> aliasHeap;
    uint64_t aliasPoolID = 0, aliasOffset = 0, aliasSize = 0;
};

class PublicationBindingBundle {
public:
    using Snapshot = std::shared_ptr<const ResourceBindingSnapshot>;
    PublicationBindingBundle(std::vector<Snapshot> bindings,
        std::vector<std::shared_ptr<const void>> semanticOwners = {},
        std::vector<std::shared_ptr<const PublicationBindingBundle>> parents = {});
    const Snapshot* Find(uint64_t resourceID) const noexcept;
    const Snapshot* FindNative(rhi::ResourceHandle resource, const BindlessResourceViews* views) const noexcept;
    std::span<const Snapshot> Bindings() const noexcept { return m_bindings; }
    // Capture only on the producer/preparation owner, before making the
    // immutable bundle visible. Workers consume snapshots, never Resource.
    static Snapshot Capture(Resource& resource);
private:
    std::vector<Snapshot> m_bindings;
    std::vector<std::shared_ptr<const void>> m_semanticOwners;
    std::unordered_map<uint64_t, size_t> m_indices;
    std::unordered_multimap<uint64_t, size_t> m_nativeIndices;
    std::vector<std::shared_ptr<const PublicationBindingBundle>> m_parents;
};
// Explicit diagnostics only. Weak observations never become lifetime owners.
bool BindingLifetimeTraceEnabled();
void TraceBindingHolder(const std::shared_ptr<const PublicationBindingBundle>&,
    std::shared_ptr<const void> holder, const char* role, uint64_t frame = 0);
void TraceBindingHolderFromOwner(const void* source, std::shared_ptr<const void> holder, const char* role);
void DumpBindingHolders(uint64_t resourceID, std::ostream&);
}
