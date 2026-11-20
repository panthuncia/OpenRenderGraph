#include "Render/PublicationBindingBundle.h"
#include "Resources/BackedResource.h"
#include "Resources/GloballyIndexedResource.h"
#include "Resources/Resource.h"
#include "Resources/DynamicResource.h"
#include <BasicTelemetry/Tracy.h>
#include <stdexcept>
#include <cstdlib>
#include <mutex>
#include <ostream>
#include <algorithm>
#include <tuple>

namespace org {
PublicationBindingBundle::PublicationBindingBundle(std::vector<Snapshot> bindings,
    std::vector<std::shared_ptr<const void>> semanticOwners,
    std::vector<std::shared_ptr<const PublicationBindingBundle>> parents)
    : m_bindings(std::move(bindings)), m_semanticOwners(std::move(semanticOwners)), m_parents(std::move(parents)) {
    for (size_t i = 0; i < m_bindings.size(); ++i) {
        const auto& binding = m_bindings[i];
        if (!binding || !binding->resource.GetHandle().valid() || !binding->allocationOwner)
            throw std::invalid_argument("Publication has an unowned resource binding");
        if (!m_indices.emplace(binding->resourceID, i).second)
            throw std::invalid_argument("Publication has duplicate resource bindings");
        const auto handle = binding->resource.GetHandle();
        m_nativeIndices.emplace((uint64_t{handle.generation} << 32) | handle.index, i);
    }
}
const PublicationBindingBundle::Snapshot* PublicationBindingBundle::Find(uint64_t id) const noexcept {
    const auto found = m_indices.find(id);
    if (found != m_indices.end()) return &m_bindings[found->second];
    for (const auto& parent : m_parents) if (parent)
        if (const auto* binding = parent->Find(id)) return binding;
    return nullptr;
}
const PublicationBindingBundle::Snapshot* PublicationBindingBundle::FindNative(rhi::ResourceHandle resource,
    const BindlessResourceViews* views) const noexcept {
    const auto [begin, end] = m_nativeIndices.equal_range((uint64_t{resource.generation} << 32) | resource.index);
    for (auto found = begin; found != end; ++found)
        if (m_bindings[found->second]->views.get() == views) return &m_bindings[found->second];
    for (const auto& parent : m_parents) if (parent)
        if (const auto* binding = parent->FindNative(resource, views)) return binding;
    return nullptr;
}
PublicationBindingBundle::Snapshot PublicationBindingBundle::Capture(Resource& resource) {
    if (auto* dynamic = dynamic_cast<DynamicResource*>(&resource)) {
        auto concrete = dynamic->GetResource();
        return concrete ? Capture(*concrete) : Snapshot{};
    }
    auto result = std::make_shared<ResourceBindingSnapshot>();
    std::unique_lock<std::recursive_mutex> backingLock;
    result->resourceID = resource.GetGlobalResourceID();
    result->externallyImmutable = resource.GetGraphOwnership() == Resource::GraphOwnership::ExternalImmutableShaderResource;
    resource.TryGetRHIResourceDesc(result->description);
    if (auto set = resource.AcquireAPIRepresentationSet();
        set && !set->devices.empty() && set->devices.front() && set->devices.front()->resource) {
        // Explicit per-device representations (multi-device) replace the backing. The primary device's representation is the
        // resource. The published set owns every device's representation and is the allocation's identity: slots of one
        // resource captured at different times must name the same owner. Representations placed in a shared alias pool are
        // aliased like a placed backing: they enter every execution undefined, and the alias ledger orders their range.
        const auto& devices = set->devices;
        const auto& primary = devices.front();
        result->representations.resize(devices.size());
        for (size_t index = 0; index < devices.size(); ++index)
            if (devices[index] && devices[index]->resource) result->representations[index] = devices[index]->resource.Get();
        result->resource = primary->resource.Get();
        result->backingGeneration = primary->backingGeneration;
        result->aliasHeap = set->placement.heap;
        result->aliasPoolID = set->placement.poolID;
        result->aliasOffset = set->placement.offset;
        result->aliasSize = set->placement.size;
        result->allocationOwner = std::move(set);
    } else if (auto* backed = dynamic_cast<BackedResource*>(&resource)) {
        // Held until the views are captured: a re-placement between the two would pair this allocation with no views, or
        // with the next backing's.
        if (auto* mutex = backed->BackingMutex()) backingLock = std::unique_lock(*mutex);
        auto allocation = backed->CapturePublishedBackingAllocation();
        if (!allocation) return {};
        result->resource = allocation->resource;
        result->backingGeneration = allocation->generation;
        result->aliasHeap = allocation->aliasHeap;
        result->aliasPoolID = allocation->aliasPoolID;
        result->aliasOffset = allocation->aliasOffset;
        result->aliasSize = allocation->aliasSize;
        result->allocationOwner = std::move(allocation);
    } else {
        result->resource = resource.GetAPIResource();
        result->allocationOwner = resource.weak_from_this().lock();
        const auto handle = result->resource.GetHandle();
        result->backingGeneration = (uint64_t{handle.generation} << 32) | handle.index;
    }
    if (!result->resource.GetHandle().valid() || !result->allocationOwner) return {};
    result->semanticConsumer = resource.CaptureSemanticConsumerLease();
    if (auto* indexed = dynamic_cast<GloballyIndexedResource*>(&resource)) {
        result->views = indexed->CaptureBindlessViews();
        result->descriptorOwner = indexed->CaptureDescriptorOwnership();
    }
    result->bindingTable = resource.GetBindingTableVersion();
    result->recordingOwner = std::make_shared<const std::tuple<std::shared_ptr<const void>, std::shared_ptr<const void>,
        std::shared_ptr<const BindingTableVersion>>>(result->allocationOwner, result->descriptorOwner, result->bindingTable);
    return result;
}
}
