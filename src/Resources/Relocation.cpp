#include "Resources/Relocation.h"

#include <algorithm>
#include <stdexcept>

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

} // namespace

Service::Service() = default;

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
