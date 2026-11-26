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
class BindingTableVersion;
class GloballyIndexedResource;
struct BackingAllocationSnapshot;

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
    std::shared_ptr<const BindingTableVersion> bindingTable;
    std::shared_ptr<const AliasHeapGeneration> aliasHeap;
    uint64_t aliasPoolID = 0, aliasOffset = 0, aliasSize = 0;
    // The resource is GraphOwnership::ExternalImmutableShaderResource: safe to read through an undeclared descriptor.
    bool externallyImmutable = false;
    // For a resource shared across devices: its representation on each device instance, by index. The primary device's
    // representation is `resource`. Empty for a resource that only the primary device uses.
    std::vector<rhi::Resource> representations;
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
    // What Capture will return for resource once it adopted a prepared backing (BackedResource::AdoptPreparedBacking): allocation
    // is that backing's, slots holds the descriptor slots written for it. Any thread: it reads only what does not change with
    // the backing.
    static Snapshot CapturePrepared(Resource& resource, std::shared_ptr<const BackingAllocationSnapshot> allocation,
        const GloballyIndexedResource& slots);
private:
    std::vector<Snapshot> m_bindings;
    std::vector<std::shared_ptr<const void>> m_semanticOwners;
    std::unordered_map<uint64_t, size_t> m_indices;
    std::unordered_multimap<uint64_t, size_t> m_nativeIndices;
    std::vector<std::shared_ptr<const PublicationBindingBundle>> m_parents;
};
}
