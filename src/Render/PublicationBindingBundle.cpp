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

namespace org {
namespace {
struct BindingHolderObservation {
    std::weak_ptr<const PublicationBindingBundle> bundle;
    std::weak_ptr<const void> holder;
    const void* identity;
    const char* role;
    uint64_t frame;
};
std::mutex g_bindingHolderMutex;
std::vector<BindingHolderObservation> g_bindingHolders;
std::vector<std::weak_ptr<const ResourceBindingSnapshot>> g_producerSnapshots;
}
bool BindingLifetimeTraceEnabled() {
    static const bool enabled = [] {
        const auto* value = std::getenv("ORG_RESOURCE_LIFETIME_TRACE");
        return value && value[0] == '1';
    }();
    return enabled;
}
void TraceBindingHolder(const std::shared_ptr<const PublicationBindingBundle>& bundle,
    std::shared_ptr<const void> holder, const char* role, uint64_t frame) {
    if (!BindingLifetimeTraceEnabled() || !bundle || !holder) return;
    std::lock_guard lock(g_bindingHolderMutex);
    std::erase_if(g_bindingHolders, [](const auto& entry) { return entry.holder.expired(); });
    for (auto& entry : g_bindingHolders) if (entry.identity == holder.get()) {
        entry = {bundle, holder, holder.get(), role, frame};
        return;
    }
    g_bindingHolders.push_back({bundle, holder, holder.get(), role, frame});
}
void TraceBindingHolderFromOwner(const void* source, std::shared_ptr<const void> holder, const char* role) {
    if (!BindingLifetimeTraceEnabled() || !holder) return;
    std::shared_ptr<const PublicationBindingBundle> bundle;
    uint64_t frame = 0;
    {
        std::lock_guard lock(g_bindingHolderMutex);
        for (const auto& entry : g_bindingHolders) if (entry.identity == source && !entry.holder.expired()) {
            bundle = entry.bundle.lock(); frame = entry.frame; break;
        }
    }
    TraceBindingHolder(bundle, std::move(holder), role, frame);
}
void DumpBindingHolders(uint64_t resourceID, std::ostream& out) {
    std::vector<BindingHolderObservation> observations;
    std::vector<std::weak_ptr<const ResourceBindingSnapshot>> snapshots;
    { std::lock_guard lock(g_bindingHolderMutex); observations = g_bindingHolders; snapshots = g_producerSnapshots; }
    for (const auto& weak : snapshots) {
        const auto snapshot = weak.lock();
        if (!snapshot || snapshot->resourceID != resourceID) continue;
        const auto handle = snapshot->resource.GetHandle();
        out << "snapshot," << resourceID << ',' << snapshot.get() << ',' << snapshot->backingGeneration
            << ',' << handle.index << ',' << handle.generation << ',' << snapshot->views.get()
            << ',' << weak.use_count() - 1 << ',' << snapshot->semanticConsumer.get()
            << ',' << snapshot->semanticConsumer.use_count() << '\n';
    }
    for (const auto& entry : observations) {
        const auto holder = entry.holder.lock();
        const auto bundle = entry.bundle.lock();
        if (!holder || !bundle) continue;
        const auto* binding = bundle->Find(resourceID);
        if (!binding) continue;
        const auto& snapshot = **binding;
        out << "holder," << resourceID << ',' << snapshot.backingGeneration << ','
            << entry.role << ',' << entry.frame << ',' << holder.get() << ',' << bundle.get()
            << ',' << entry.holder.use_count() - (holder.get() == bundle.get() ? 2 : 1) << ',' << entry.bundle.use_count() - (holder.get() == bundle.get() ? 2 : 1)
            << ',' << binding->use_count() << ',' << snapshot.semanticConsumer.use_count() << '\n';
    }
}
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
    if (auto* dynamic = dynamic_cast<DynamicGloballyIndexedResource*>(&resource)) {
        auto concrete = dynamic->GetResource();
        return concrete ? Capture(*concrete) : Snapshot{};
    }
    auto result = std::make_shared<ResourceBindingSnapshot>();
    result->resourceID = resource.GetGlobalResourceID();
    resource.TryGetRHIResourceDesc(result->description);
    if (auto* backed = dynamic_cast<BackedResource*>(&resource)) {
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
    result->recordingOwner = std::make_shared<const std::pair<std::shared_ptr<const void>, std::shared_ptr<const void>>>(
        result->allocationOwner, result->descriptorOwner);
    if (BindingLifetimeTraceEnabled()) {
        std::lock_guard lock(g_bindingHolderMutex);
        std::erase_if(g_producerSnapshots, [](const auto& entry) { return entry.expired(); });
        g_producerSnapshots.push_back(result);
    }
    return result;
}
}
