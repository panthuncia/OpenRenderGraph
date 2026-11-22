#include "Resources/Relocation.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <unordered_set>

#include <spdlog/spdlog.h>
#include <BasicTelemetry/Tracy.h>

#include "Managers/Singletons/DescriptorHeapManager.h"
#include "OpenRenderGraph/ExternalClient.h"
#include "Resources/Buffers/DynamicBufferBase.h"
#include "Resources/ExternalBackingResource.h"
#include "Resources/ExternalBufferResource.h"
#include "Resources/ExternalTextureResource.h"
#include "Resources/GPUBacking/GPUTextureBacking.h"
#include "Resources/GPUBacking/GpuBufferBacking.h"
#include "Resources/PixelBuffer.h"
#include "Resources/Resource.h"
#include "Resources/ResourceStateTracker.h"

namespace org::relocation {

Relocation::Relocation() = default;
Relocation::~Relocation() = default;
Relocation::Relocation(Relocation&&) noexcept = default;
Relocation& Relocation::operator=(Relocation&&) noexcept = default;

const char* SourceName(Source value) noexcept {
    switch (value) {
    case Source::Dxvk: return "Dxvk";
    case Source::Defragmenter: return "Defragmenter";
    case Source::Owner: return "Owner";
    case Source::Stress: return "Stress";
    default: return "Unknown";
    }
}

bool MayMove(const RelocationRecord* record, bool gpuIdle, IgnoreReason* why) noexcept {
    IgnoreReason reason = IgnoreReason::Count;
    if (!record) reason = IgnoreReason::NoRecord;
    else if (record->GetClass() == Class::Pinned) reason = IgnoreReason::Pinned;
    else if (record->GetClass() == Class::Written && !gpuIdle) reason = IgnoreReason::WrittenNotIdle;
    if (reason == IgnoreReason::Count) return true;
    if (why) *why = reason;
    return false;
}

// ---- journal ----

void RelocationJournal::Append(uint64_t resourceID, uint64_t toGeneration, Source reason) {
    std::scoped_lock lock(m_mutex);
    const auto sequence = m_head.load(std::memory_order_relaxed);
    m_entries.push_back({sequence, resourceID, toGeneration, reason});
    while (m_entries.size() > m_capacity) m_entries.pop_front();
    m_head.store(sequence + 1, std::memory_order_release);
}

bool RelocationJournal::Read(uint64_t& cursor, std::vector<JournalEntry>& out) const {
    std::scoped_lock lock(m_mutex);
    const auto head = m_head.load(std::memory_order_relaxed);
    const uint64_t oldest = m_entries.empty() ? head : m_entries.front().sequence;
    const bool complete = cursor >= oldest;
    for (const auto& entry : m_entries)
        if (entry.sequence >= cursor) out.push_back(entry);
    cursor = head;
    return complete;
}

// ---- service ----

namespace {

// How many relocations one Poll call may return.
constexpr size_t kPollBatch = 64;
// How many moved-from generations are kept per resource. A recording older than that many moves of one resource is
// treated as current. It still holds the backing it bound, so it stays safe to submit.
constexpr size_t kMovedFromKept = 16;

uint64_t HandleKey(rhi::Resource resource) noexcept {
    const auto handle = resource.GetHandle();
    return (uint64_t{handle.generation} << 32) | handle.index;
}

// What a relocation leaves behind: the old backing, the copy target and the source's own retired object. They go together
// once the GPU reached the frontier taken after the relocation batch.
struct RetiredParts {
    std::unique_ptr<GpuTextureBacking> texture;
    std::unique_ptr<GpuBufferBacking> buffer;
    rhi::ResourcePtr next;
    std::shared_ptr<Resource> copyTarget;
    std::shared_ptr<const void> retired;
    uint64_t bytes = 0;
    bool counted = false;  // a committed move's old backing, counted as awaiting retirement
    ~RetiredParts() {
        if (counted) detail::NoteRetired(bytes);
    }
};

void Retire(Relocation& relocation, bool committed) {
    auto parts = std::make_shared<RetiredParts>();
    parts->texture = std::move(relocation.nextTexture);
    parts->buffer = std::move(relocation.nextBuffer);
    parts->next = std::move(relocation.next);
    parts->copyTarget = std::move(relocation.copyTarget);
    parts->retired = std::move(relocation.retired);
    parts->bytes = relocation.sizeBytes;
    parts->counted = committed;
    if (committed) detail::NoteAwaitingRetirement(parts->bytes);
    DescriptorHeapManager::GetInstance().RetireRelocated(std::move(parts));
}

// Switches the resource to its new backing. `from` is the generation it moved away from. False with why when it did not
// switch.
bool Adopt(Relocation& relocation, bool inPlace, uint64_t& from, IgnoreReason& why) {
    Resource* resource = relocation.resource.get();
    if (auto* texture = dynamic_cast<PixelBuffer*>(resource)) {
        if (relocation.nextBuffer || relocation.next) { why = IgnoreReason::Unsupported; return false; }
        from = relocation.fromGeneration ? relocation.fromGeneration : texture->GetBackingGeneration();
        if (!texture->AdoptRelocatedBacking(relocation.nextTexture, from, inPlace)) { why = IgnoreReason::Stale; return false; }
        relocation.toGeneration = texture->GetBackingGeneration();
        return true;
    }
    if (auto* buffer = dynamic_cast<BufferBase*>(resource)) {
        if (relocation.nextTexture || relocation.next) { why = IgnoreReason::Unsupported; return false; }
        from = relocation.fromGeneration ? relocation.fromGeneration : buffer->GetBackingGeneration();
        if (!buffer->AdoptRelocatedBacking(relocation.nextBuffer, from, inPlace)) { why = IgnoreReason::Stale; return false; }
        relocation.toGeneration = buffer->GetBackingGeneration();
        return true;
    }
    if (auto* backing = dynamic_cast<ExternalBackingResource*>(resource)) {
        if (relocation.nextTexture || relocation.next) { why = IgnoreReason::Unsupported; return false; }
        from = relocation.fromGeneration ? relocation.fromGeneration : backing->GetBackingGeneration();
        if (!backing->AdoptRelocatedBacking(relocation.nextBuffer, from)) { why = IgnoreReason::Stale; return false; }
        relocation.toGeneration = backing->GetBackingGeneration();
        return true;
    }
    // Imports. Their snapshots name the backing by its handle (PublicationBindingBundle::Capture), so the generations here
    // are handle keys. RefreshShared rotates the descriptors and retires the old native resource itself.
    if (auto* imported = dynamic_cast<ExternalTextureResource*>(resource)) {
        if (!relocation.next) { why = IgnoreReason::Unsupported; return false; }
        from = HandleKey(imported->GetAPIResource());
        if (relocation.fromGeneration && relocation.fromGeneration != from) { why = IgnoreReason::Stale; return false; }
        const TextureDescription description = imported->GetDescription();
        if (!imported->RefreshShared(std::move(relocation.next), description, imported->CommonLayoutOnly())) {
            why = IgnoreReason::Failed;
            return false;
        }
        relocation.toGeneration = HandleKey(imported->GetAPIResource());
        return true;
    }
    if (auto* imported = dynamic_cast<ExternalBufferResource*>(resource)) {
        if (!relocation.next) { why = IgnoreReason::Unsupported; return false; }
        from = HandleKey(imported->GetAPIResource());
        if (relocation.fromGeneration && relocation.fromGeneration != from) { why = IgnoreReason::Stale; return false; }
        const auto views = imported->GetViews();
        if (!imported->RefreshShared(std::move(relocation.next), imported->GetByteSize(), views)) {
            why = IgnoreReason::Failed;
            return false;
        }
        relocation.toGeneration = HandleKey(imported->GetAPIResource());
        return true;
    }
    why = IgnoreReason::Unsupported;
    return false;
}

// The resource a relocation batch copies into: a view of a new backing's native resource with the moving resource's shape.
// It does not own the native resource. It keeps the new backing's allocation lease, so the native resource outlives every
// use of this wrapper. Its state starts as a new resource's (Undefined for a texture, Common for a buffer), which is where
// the ledgers seed a backing they do not know yet.
class CopyTarget final : public Resource {
public:
    CopyTarget(rhi::Resource resource, Resource& like, std::shared_ptr<const void> lease)
        : m_resource(resource), m_lease(std::move(lease)) {
        m_hasLayout = like.HasLayout();
        m_mipLevels = (std::max)(1u, like.GetMipLevels());
        m_arraySize = (std::max)(1u, like.GetArraySize());
        m_hasDescription = like.TryGetRHIResourceDesc(m_description);
        // The description's pointers name the moving resource's storage. A copy needs neither.
        m_description.castableFormats = {};
        m_description.debugName = nullptr;
        if (!like.TryGetBufferByteSize(m_bufferBytes)) m_bufferBytes = 0;
        const ResourceState initial = m_hasLayout
            ? ResourceState{rhi::ResourceAccessType::None, rhi::ResourceLayout::Undefined, rhi::ResourceSyncState::None}
            : ResourceState{rhi::ResourceAccessType::Common, rhi::ResourceLayout::Common, rhi::ResourceSyncState::All};
        m_tracker = SymbolicTracker(RangeSpec{}, initial);
        name = like.GetName() + " (relocation target)";
    }

    rhi::Resource GetAPIResource() override { return m_resource; }
    rhi::BarrierBatch GetEnhancedBarrierGroup(RangeSpec range, rhi::ResourceAccessType previousAccess, rhi::ResourceAccessType nextAccess,
        rhi::ResourceLayout previousLayout, rhi::ResourceLayout nextLayout, rhi::ResourceSyncState previousSync,
        rhi::ResourceSyncState nextSync) override {
        rhi::BarrierBatch batch{};
        if (!m_hasLayout) {
            m_bufferBarrier = rhi::BufferBarrier{
                .buffer = m_resource.GetHandle(),
                .offset = 0,
                .size = UINT64_MAX,
                .beforeSync = previousSync,
                .afterSync = nextSync,
                .beforeAccess = previousAccess,
                .afterAccess = nextAccess};
            batch.buffers = {&m_bufferBarrier};
            return batch;
        }
        const auto resolved = ResolveRangeSpec(range, m_mipLevels, m_arraySize);
        m_textureBarrier = {};
        m_textureBarrier.beforeAccess = previousAccess;
        m_textureBarrier.afterAccess = nextAccess;
        m_textureBarrier.beforeLayout = previousLayout;
        m_textureBarrier.afterLayout = nextLayout;
        m_textureBarrier.beforeSync = previousSync;
        m_textureBarrier.afterSync = nextSync;
        m_textureBarrier.range = {resolved.firstMip, resolved.mipCount, resolved.firstSlice, resolved.sliceCount};
        m_textureBarrier.texture = m_resource.GetHandle();
        batch.textures = {&m_textureBarrier};
        return batch;
    }
    SymbolicTracker* GetStateTracker() override { return &m_tracker; }
    bool TryGetRHIResourceDesc(rhi::ResourceDesc& outDesc) const override {
        if (!m_hasDescription) return false;
        outDesc = m_description;
        return true;
    }
    bool TryGetBufferByteSize(uint64_t& outByteSize) const override {
        if (m_hasLayout) return false;
        outByteSize = m_bufferBytes;
        return true;
    }

private:
    rhi::Resource m_resource;
    std::shared_ptr<const void> m_lease;
    SymbolicTracker m_tracker;
    rhi::ResourceDesc m_description{};
    bool m_hasDescription = false;
    uint64_t m_bufferBytes = 0;
    rhi::TextureBarrier m_textureBarrier{};
    rhi::BufferBarrier m_bufferBarrier{};
};

} // namespace

std::unique_ptr<Relocation> MakeGraphRelocation(const std::shared_ptr<Resource>& resource, Source source, bool gpuIdle, IgnoreReason* why) {
    const auto fail = [&](IgnoreReason reason) -> std::unique_ptr<Relocation> {
        if (why) *why = reason;
        return nullptr;
    };
    if (!resource) return fail(IgnoreReason::Expired);
    auto* backed = dynamic_cast<BackedResource*>(resource.get());
    if (!backed) return fail(IgnoreReason::Unsupported);
    // An externally immutable resource gets no barriers from the ledgers, so the copy could not move it to CopySource.
    if (resource->GetGraphOwnership() == Resource::GraphOwnership::ExternalImmutableShaderResource
        || resource->GetAttachedAPIRepresentation(BackendInstanceId::Primary).IsValid())
        return fail(IgnoreReason::Unsupported);
    IgnoreReason reason = IgnoreReason::Failed;
    // Placed backings have no record of their own, so they stop here as NoRecord.
    if (!MayMove(backed->CurrentRelocationRecord(), gpuIdle, &reason)) return fail(reason);
    auto relocation = std::make_unique<Relocation>();
    relocation->resource = resource;
    relocation->resourceID = resource->GetGlobalResourceID();
    relocation->fromGeneration = backed->GetBackingGeneration();
    relocation->source = source;
    try {
        if (auto* texture = dynamic_cast<PixelBuffer*>(resource.get())) {
            const auto& description = texture->GetDescription();
            // A copy cannot carry a multisampled texture. On D3D12 it copies one plane, so a depth-stencil texture would
            // lose its stencil.
            if (description.sampleCount > 1 || description.hasDSV) return fail(IgnoreReason::Unsupported);
            auto next = texture->MakeRelocationBacking();
            if (!next) return fail(IgnoreReason::NoRecord);
            relocation->copyTarget = std::make_shared<CopyTarget>(next->GetAPIResource(), *resource, next->CaptureAllocationLease());
            relocation->nextTexture = std::move(next);
        } else if (auto* buffer = dynamic_cast<BufferBase*>(resource.get())) {
            auto next = buffer->MakeRelocationBacking();
            if (!next) return fail(IgnoreReason::NoRecord);
            relocation->copyTarget = std::make_shared<CopyTarget>(next->GetAPIResource(), *resource, next->CaptureAllocationLease());
            relocation->nextBuffer = std::move(next);
        } else {
            return fail(IgnoreReason::Unsupported);
        }
    } catch (const std::exception& error) {
        spdlog::warn("ORG relocation: making a new backing for '{}' failed: {}", resource->GetName(), error.what());
        return fail(IgnoreReason::Failed);
    }
    if (auto* record = backed->CurrentRelocationRecord()) relocation->sizeBytes = record->SizeBytes();
    // The backing changed while the new one was made: the copy would read the wrong backing.
    if (backed->GetBackingGeneration() != relocation->fromGeneration) return fail(IgnoreReason::Stale);
    return relocation;
}

// Resources marked ReadOnly, for the stress switch. Taken in turn, so each one moves now and then.
struct Service::Candidates {
    std::mutex mutex;
    std::vector<std::weak_ptr<Resource>> resources;
    std::unordered_set<uint64_t> ids;
    size_t next = 0;
};

Service::Service() : m_candidates(std::make_unique<Candidates>()) {
    // A debug switch, read once from the environment like ORG's other ones. SetStress overrides it.
    if (const auto* value = std::getenv("ORG_RELOCATION_STRESS")) {
        const auto parsed = (std::min)(std::strtoul(value, nullptr, 10), 4096ul);
        m_stress.store(static_cast<uint32_t>(parsed), std::memory_order_release);
        if (parsed) spdlog::info("ORG relocation: ORG_RELOCATION_STRESS moves {} ReadOnly backings at each commit", parsed);
    }
}

Service::~Service() = default;

void Service::SetStress(uint32_t perCommit) noexcept {
    m_stress.store(perCommit, std::memory_order_release);
}

void Service::NoteReadOnly(const std::shared_ptr<Resource>& resource) {
    if (!resource) return;
    std::scoped_lock lock(m_candidates->mutex);
    if (!m_candidates->ids.insert(resource->GetGlobalResourceID()).second) return;
    m_candidates->resources.push_back(resource);
}

void Service::MakeStressRelocations(uint32_t count, std::vector<Relocation>& out) {
    BT_ZONE_SCOPE("ORG.Relocation.Stress");
    std::vector<std::shared_ptr<Resource>> picked;
    {
        std::scoped_lock lock(m_candidates->mutex);
        auto& candidates = *m_candidates;
        // One pass over the list at most, from where the last commit stopped. Gone resources leave the list.
        const size_t size = candidates.resources.size();
        for (size_t seen = 0; seen < size && picked.size() < count && !candidates.resources.empty(); ++seen) {
            if (candidates.next >= candidates.resources.size()) candidates.next = 0;
            auto resource = candidates.resources[candidates.next].lock();
            if (!resource) {
                candidates.resources.erase(candidates.resources.begin() + static_cast<std::ptrdiff_t>(candidates.next));
                continue;
            }
            ++candidates.next;
            auto* backed = dynamic_cast<BackedResource*>(resource.get());
            if (!backed || backed->GetRelocationClass() != Class::ReadOnly) continue;
            picked.push_back(std::move(resource));
        }
        // The ID set only stops duplicates. It is rebuilt once gone resources have left the list.
        if (candidates.ids.size() > candidates.resources.size()) {
            candidates.ids.clear();
            for (const auto& weak : candidates.resources)
                if (const auto resource = weak.lock()) candidates.ids.insert(resource->GetGlobalResourceID());
        }
    }
    for (const auto& resource : picked) {
        IgnoreReason why = IgnoreReason::Failed;
        auto relocation = MakeGraphRelocation(resource, Source::Stress, false, &why);
        if (!relocation) {
            detail::NoteIgnored(why);
            continue;
        }
        out.push_back(std::move(*relocation));
    }
}

Service& Service::Get() {
    // Never destroyed: retirements may run during static destruction.
    static auto* service = new Service();
    return *service;
}

void Service::AddSource(std::shared_ptr<IRelocationSource> source) {
    if (!source) return;
    std::scoped_lock lock(m_sourcesMutex);
    m_sources.push_back(std::move(source));
}

void Service::RemoveSource(const IRelocationSource* source) {
    std::scoped_lock lock(m_sourcesMutex);
    std::erase_if(m_sources, [&](const auto& entry) { return entry.get() == source; });
}

bool Service::MovedFrom(uint64_t resourceID, uint64_t generation) const {
    std::shared_lock lock(m_movedMutex);
    const auto found = m_movedFrom.find(resourceID);
    return found != m_movedFrom.end() && std::ranges::find(found->second, generation) != found->second.end();
}

void Service::NoteMovedFrom(uint64_t resourceID, uint64_t generation) {
    std::unique_lock lock(m_movedMutex);
    auto& generations = m_movedFrom[resourceID];
    generations.push_back(generation);
    if (generations.size() > kMovedFromKept) generations.erase(generations.begin());
}

bool Service::SubmitCopies(const CommitContext& context, std::span<Relocation*> copies) {
    if (copies.empty()) return true;
    if (!context.client) {
        spdlog::error("ORG relocation: {} moves need a copy, and the commit has no external client to submit it", copies.size());
        return false;
    }
    BT_ZONE_SCOPE("ORG.Relocation.SubmitCopies");
    // The external client works out the barriers from the ledgers:
    // - each old backing goes from the state its last user left it in to CopySource,
    // - each new backing has no ledger entry yet, so it enters from its creation state to CopyDest.
    // It records the new states, so the next graph that uses a moved resource starts from CopyDest and waits for this batch.
    const ResourceState copySource{rhi::ResourceAccessType::CopySource, rhi::ResourceLayout::CopySource, rhi::ResourceSyncState::Copy};
    const ResourceState copyDest{rhi::ResourceAccessType::CopyDest, rhi::ResourceLayout::CopyDest, rhi::ResourceSyncState::Copy};
    struct Copy {
        rhi::ResourceHandle from, to;
        uint64_t bytes = 0;  // a buffer's size; 0 for a texture
        uint32_t mips = 1, slices = 1;
    };
    std::vector<ExternalClient::Access> accesses;
    std::vector<Copy> plan;
    accesses.reserve(copies.size() * 2);
    plan.reserve(copies.size());
    for (auto* relocation : copies) {
        auto& resource = *relocation->resource;
        auto& target = *relocation->copyTarget;
        Copy copy;
        copy.from = resource.GetAPIResource().GetHandle();
        copy.to = target.GetAPIResource().GetHandle();
        if (!resource.TryGetBufferByteSize(copy.bytes)) copy.bytes = 0;
        copy.mips = (std::max)(1u, resource.GetMipLevels());
        copy.slices = (std::max)(1u, resource.GetArraySize());
        plan.push_back(copy);
        accesses.push_back({&resource, RangeSpec{}, copySource});
        accesses.push_back({&target, RangeSpec{}, copyDest});
    }
    try {
        // Not cached: the publication names the old backings, which must go once the GPU is done with this batch.
        context.client->Submit(context.copySlot, accesses, [&](rhi::CommandList& list) {
            for (const auto& copy : plan) {
                if (copy.bytes) {
                    list.CopyBufferRegion(copy.to, 0, copy.from, 0, copy.bytes);
                    continue;
                }
                // A zero extent copies the whole subresource, a 3D texture's every depth slice included.
                for (uint32_t slice = 0; slice < copy.slices; ++slice)
                    for (uint32_t mip = 0; mip < copy.mips; ++mip)
                        list.CopyTextureRegion(rhi::TextureCopyRegion{copy.to, mip, slice}, rhi::TextureCopyRegion{copy.from, mip, slice});
            }
        }, {}, {}, {}, false);
    } catch (const std::exception& error) {
        spdlog::error("ORG relocation: the relocation batch of {} copies failed: {}", copies.size(), error.what());
        return false;
    }
    basic_telemetry::AddCounter("ORG.Relocation.Copies", static_cast<int64_t>(copies.size()));
    return true;
}

size_t Service::Commit(const CommitContext& context) {
    BT_ZONE_SCOPE("ORG.Relocation.Commit");
    std::vector<std::shared_ptr<IRelocationSource>> sources;
    {
        std::scoped_lock lock(m_sourcesMutex);
        sources = m_sources;
    }
    std::vector<Relocation> pending;
    // Per source that returned any, the range of `pending` it wrote, for BeforeAdopt.
    struct Polled {
        IRelocationSource* source = nullptr;
        size_t first = 0, last = 0;
    };
    std::vector<Polled> polled;
    for (const auto& source : sources) {
        const size_t first = pending.size();
        for (;;) {
            const size_t base = pending.size();
            pending.resize(base + kPollBatch);
            const size_t written = (std::min)(source->Poll(std::span<Relocation>(pending.data() + base, kPollBatch)), kPollBatch);
            pending.resize(base + written);
            if (written < kPollBatch) break;
        }
        if (pending.size() > first) polled.push_back({source.get(), first, pending.size()});
    }
    if (const auto stress = Stress()) MakeStressRelocations(stress, pending);
    if (pending.empty()) return 0;

    // Moves whose resource is gone, or whose backing changed since they were made, are skipped.
    std::vector<uint8_t> live(pending.size(), 1);
    const auto ignore = [&](size_t index, IgnoreReason reason) {
        live[index] = 0;
        detail::NoteIgnored(reason);
    };
    for (size_t i = 0; i < pending.size(); ++i) {
        auto& relocation = pending[i];
        if (!relocation.resource) {
            ignore(i, IgnoreReason::Expired);
            continue;
        }
        if (!relocation.resourceID) relocation.resourceID = relocation.resource->GetGlobalResourceID();
        auto* backed = dynamic_cast<BackedResource*>(relocation.resource.get());
        if (!relocation.sizeBytes && backed)
            if (auto* record = backed->CurrentRelocationRecord()) relocation.sizeBytes = record->SizeBytes();
        if (relocation.fromGeneration && backed && backed->GetBackingGeneration() != relocation.fromGeneration)
            ignore(i, IgnoreReason::Stale);
    }

    // 1. The relocation batch: one submission on the graphics queue, before the frame's first graph submission.
    std::vector<Relocation*> copies;
    std::vector<size_t> copyIndices;
    for (size_t i = 0; i < pending.size(); ++i)
        if (live[i] && pending[i].copyTarget) {
            copies.push_back(&pending[i]);
            copyIndices.push_back(i);
        }
    if (!copies.empty() && !SubmitCopies(context, copies))
        for (const auto index : copyIndices) ignore(index, IgnoreReason::Failed);

    // 2. Data another queue or API copied: every queue the graphs use waits for it before their next work.
    std::vector<rhi::TimelinePoint> waits;
    for (size_t i = 0; i < pending.size(); ++i)
        if (live[i] && pending[i].ready.timeline.IsValid() && pending[i].ready.value)
            waits.push_back({pending[i].ready.timeline.GetHandle(), pending[i].ready.value});
    if (!waits.empty())
        for (auto queue : context.queues) {
            if (!queue) continue;
            rhi::SubmitDesc submit{};
            submit.waits = {waits.data(), static_cast<uint32_t>(waits.size())};
            if (const auto result = queue.Submit({}, submit); result != rhi::Result::Ok)
                spdlog::error("ORG relocation: a queue's wait for {} moved resources failed: {}", waits.size(), rhi::ResultName(result));
        }

    // 3. Sources finish their part before the switch (a defragmenter's EndPass).
    for (const auto& entry : polled)
        entry.source->BeforeAdopt(std::span<Relocation>(pending.data() + entry.first, entry.last - entry.first));

    // 4. The switch, and the retirement of what each move leaves behind. The retirement is queued after the relocation
    // batch was submitted, so its frontier covers the copies.
    size_t committed = 0;
    for (size_t i = 0; i < pending.size(); ++i) {
        auto& relocation = pending[i];
        if (!live[i]) {
            // A new backing a failed batch may name: it goes behind the frontier too.
            Retire(relocation, false);
            continue;
        }
        uint64_t from = 0;
        IgnoreReason why = IgnoreReason::Failed;
        bool adopted = false;
        try {
            adopted = Adopt(relocation, context.gpuIdle, from, why);
        } catch (const std::exception& error) {
            spdlog::error("ORG relocation: moving '{}' failed: {}", relocation.resource->GetName(), error.what());
        }
        if (!adopted) {
            ignore(i, why);
            Retire(relocation, false);
            continue;
        }
        NoteMovedFrom(relocation.resourceID, from);
        m_journal.Append(relocation.resourceID, relocation.toGeneration, relocation.source);
        detail::NoteMoved(relocation.sizeBytes);
        Retire(relocation, true);
        ++committed;
    }
    detail::NoteJournalHead(m_journal.Head());
    basic_telemetry::AddCounter("ORG.Relocation.Moved", static_cast<int64_t>(committed));
    return committed;
}

} // namespace org::relocation
