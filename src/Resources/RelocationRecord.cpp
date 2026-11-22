#include "Resources/RelocationRecord.h"

#include <algorithm>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <rhi_allocator.h>

#include "Resources/AllocationRelease.h"
#include "Resources/BackedResource.h"
#include "Resources/Relocation.h"
#include "Resources/Resource.h"

namespace org::relocation {

namespace {

struct Counters {
    std::atomic<uint64_t> moved{0}, movedBytes{0};
    std::array<std::atomic<uint64_t>, static_cast<size_t>(IgnoreReason::Count)> ignored{};
    std::atomic<uint64_t> retired{0}, retiredBytes{0};
    std::atomic<uint64_t> awaiting{0}, awaitingBytes{0};
    std::array<std::atomic<uint64_t>, static_cast<size_t>(Class::Count)> records{};
    std::array<std::atomic<uint64_t>, static_cast<size_t>(Class::Count)> recordBytes{};
    std::atomic<uint64_t> journalHead{0};
    std::atomic<uint64_t> defragCampaigns{0}, defragPasses{0}, defragMoves{0}, defragMovedBytes{0}, defragIgnored{0};
    std::atomic<uint64_t> defragBlocksFreed{0}, defragMicroseconds{0};
    std::atomic<int64_t> defragLingering{0}, defragLingeringBytes{0};
};

// Never destroyed: records can outlive static destruction order.
Counters& TheCounters() noexcept {
    static auto* counters = new Counters();
    return *counters;
}

size_t IndexOf(Class value) noexcept {
    const auto index = static_cast<size_t>(value);
    return index < static_cast<size_t>(Class::Count) ? index : static_cast<size_t>(Class::Pinned);
}

} // namespace

const char* ClassName(Class value) noexcept {
    switch (value) {
    case Class::ReadOnly: return "ReadOnly";
    case Class::Written: return "Written";
    case Class::Pinned: return "Pinned";
    default: return "Unknown";
    }
}

const char* IgnoreReasonName(IgnoreReason value) noexcept {
    switch (value) {
    case IgnoreReason::NoRecord: return "NoRecord";
    case IgnoreReason::Pinned: return "Pinned";
    case IgnoreReason::WrittenNotIdle: return "WrittenNotIdle";
    case IgnoreReason::Expired: return "Expired";
    case IgnoreReason::Stale: return "Stale";
    case IgnoreReason::Unsupported: return "Unsupported";
    case IgnoreReason::Failed: return "Failed";
    default: return "Unknown";
    }
}

RelocationRecord::RelocationRecord(Class initial, uint64_t resourceID, uint64_t sizeBytes) noexcept
    : m_class(initial), m_resourceID(resourceID), m_sizeBytes(sizeBytes) {
    auto& counters = TheCounters();
    counters.records[IndexOf(initial)].fetch_add(1, std::memory_order_relaxed);
    counters.recordBytes[IndexOf(initial)].fetch_add(sizeBytes, std::memory_order_relaxed);
}

RelocationRecord::~RelocationRecord() {
    auto& counters = TheCounters();
    const auto index = IndexOf(GetClass());
    counters.records[index].fetch_sub(1, std::memory_order_relaxed);
    counters.recordBytes[index].fetch_sub(m_sizeBytes, std::memory_order_relaxed);
}

void RelocationRecord::SetClass(Class value) noexcept {
    const auto previous = m_class.exchange(value, std::memory_order_acq_rel);
    if (previous == value) return;
    auto& counters = TheCounters();
    counters.records[IndexOf(previous)].fetch_sub(1, std::memory_order_relaxed);
    counters.recordBytes[IndexOf(previous)].fetch_sub(m_sizeBytes, std::memory_order_relaxed);
    counters.records[IndexOf(value)].fetch_add(1, std::memory_order_relaxed);
    counters.recordBytes[IndexOf(value)].fetch_add(m_sizeBytes, std::memory_order_relaxed);
}

std::shared_ptr<Resource> RelocationRecord::Owner() const {
    std::scoped_lock lock(m_ownerMutex);
    return m_owner.lock();
}

void RelocationRecord::SetOwner(std::weak_ptr<Resource> owner) {
    std::scoped_lock lock(m_ownerMutex);
    m_owner = std::move(owner);
}

RelocationRecord* RecordOf(const rhi::ma::Allocation* allocation) noexcept {
    return allocation ? static_cast<RelocationRecord*>(allocation->GetPrivateData()) : nullptr;
}

Stats GetStats() noexcept {
    auto& counters = TheCounters();
    Stats stats;
    stats.moved = counters.moved.load(std::memory_order_relaxed);
    stats.movedBytes = counters.movedBytes.load(std::memory_order_relaxed);
    for (size_t i = 0; i < stats.ignored.size(); ++i) stats.ignored[i] = counters.ignored[i].load(std::memory_order_relaxed);
    stats.retired = counters.retired.load(std::memory_order_relaxed);
    stats.retiredBytes = counters.retiredBytes.load(std::memory_order_relaxed);
    stats.awaitingRetirement = counters.awaiting.load(std::memory_order_relaxed);
    stats.awaitingRetirementBytes = counters.awaitingBytes.load(std::memory_order_relaxed);
    for (size_t i = 0; i < stats.records.size(); ++i) {
        stats.records[i] = counters.records[i].load(std::memory_order_relaxed);
        stats.recordBytes[i] = counters.recordBytes[i].load(std::memory_order_relaxed);
    }
    stats.journalHead = counters.journalHead.load(std::memory_order_relaxed);
    stats.defragCampaigns = counters.defragCampaigns.load(std::memory_order_relaxed);
    stats.defragPasses = counters.defragPasses.load(std::memory_order_relaxed);
    stats.defragMoves = counters.defragMoves.load(std::memory_order_relaxed);
    stats.defragMovedBytes = counters.defragMovedBytes.load(std::memory_order_relaxed);
    stats.defragIgnored = counters.defragIgnored.load(std::memory_order_relaxed);
    stats.defragBlocksFreed = counters.defragBlocksFreed.load(std::memory_order_relaxed);
    stats.defragMicroseconds = counters.defragMicroseconds.load(std::memory_order_relaxed);
    stats.defragLingering = static_cast<uint64_t>((std::max)(int64_t{0}, counters.defragLingering.load(std::memory_order_relaxed)));
    stats.defragLingeringBytes = static_cast<uint64_t>((std::max)(int64_t{0}, counters.defragLingeringBytes.load(std::memory_order_relaxed)));
    return stats;
}

namespace detail {

void NoteMoved(uint64_t bytes) noexcept {
    auto& counters = TheCounters();
    counters.moved.fetch_add(1, std::memory_order_relaxed);
    counters.movedBytes.fetch_add(bytes, std::memory_order_relaxed);
}

void NoteIgnored(IgnoreReason reason) noexcept {
    const auto index = static_cast<size_t>(reason);
    if (index < static_cast<size_t>(IgnoreReason::Count))
        TheCounters().ignored[index].fetch_add(1, std::memory_order_relaxed);
}

void NoteAwaitingRetirement(uint64_t bytes) noexcept {
    auto& counters = TheCounters();
    counters.awaiting.fetch_add(1, std::memory_order_relaxed);
    counters.awaitingBytes.fetch_add(bytes, std::memory_order_relaxed);
}

void NoteRetired(uint64_t bytes) noexcept {
    auto& counters = TheCounters();
    counters.awaiting.fetch_sub(1, std::memory_order_relaxed);
    counters.awaitingBytes.fetch_sub(bytes, std::memory_order_relaxed);
    counters.retired.fetch_add(1, std::memory_order_relaxed);
    counters.retiredBytes.fetch_add(bytes, std::memory_order_relaxed);
}

void NoteJournalHead(uint64_t head) noexcept {
    TheCounters().journalHead.store(head, std::memory_order_relaxed);
}

void NoteDefragCampaign() noexcept {
    TheCounters().defragCampaigns.fetch_add(1, std::memory_order_relaxed);
}

void NoteDefragPass(uint64_t moves, uint64_t bytes, uint64_t ignored, uint64_t blocksFreed, uint64_t microseconds) noexcept {
    auto& counters = TheCounters();
    counters.defragPasses.fetch_add(1, std::memory_order_relaxed);
    counters.defragMoves.fetch_add(moves, std::memory_order_relaxed);
    counters.defragMovedBytes.fetch_add(bytes, std::memory_order_relaxed);
    counters.defragIgnored.fetch_add(ignored, std::memory_order_relaxed);
    counters.defragBlocksFreed.fetch_add(blocksFreed, std::memory_order_relaxed);
    counters.defragMicroseconds.fetch_add(microseconds, std::memory_order_relaxed);
}

void NoteDefragLingering(int64_t count, int64_t bytes) noexcept {
    auto& counters = TheCounters();
    counters.defragLingering.fetch_add(count, std::memory_order_relaxed);
    counters.defragLingeringBytes.fetch_add(bytes, std::memory_order_relaxed);
}

} // namespace detail

// ---- allocation release ----

namespace {

struct ReleaseGate {
    // Live holds.
    std::atomic<uint32_t> holds{0};
    // Frees that checked the holds and may be freeing now. A new hold waits for them.
    std::atomic<uint32_t> releasing{0};
    std::mutex parkedMutex;
    std::vector<std::pair<rhi::ma::AllocationPtr, std::unique_ptr<RelocationRecord>>> parked;
    uint64_t parkedBytes = 0;
};

// Never destroyed: allocations can be freed during static destruction.
ReleaseGate& Gate() noexcept {
    static auto* gate = new ReleaseGate();
    return *gate;
}

} // namespace

void ReleaseAllocation(rhi::ma::AllocationPtr allocation, std::unique_ptr<RelocationRecord> record) noexcept {
    auto& gate = Gate();
    // The two counters work like a two-flag lock. A free announces itself, then checks the holds. A hold counts itself, then
    // waits until no free is announced. So either the free sees the hold and parks, or the hold waits for the free to finish.
    gate.releasing.fetch_add(1, std::memory_order_seq_cst);
    if (gate.holds.load(std::memory_order_seq_cst) == 0) {
        // The allocation goes before its record, so the allocation never points at a freed record.
        allocation.Reset();
        record.reset();
        gate.releasing.fetch_sub(1, std::memory_order_seq_cst);
        return;
    }
    gate.releasing.fetch_sub(1, std::memory_order_seq_cst);
    try {
        std::scoped_lock lock(gate.parkedMutex);
        const uint64_t bytes = allocation ? allocation->GetSize() : 0;
        gate.parked.emplace_back(std::move(allocation), std::move(record));
        gate.parkedBytes += bytes;
    } catch (...) {
        // Out of memory for the list. Freeing now is the only choice left.
        allocation.Reset();
        record.reset();
    }
}

ReleaseHold::ReleaseHold() {
    auto& gate = Gate();
    gate.holds.fetch_add(1, std::memory_order_seq_cst);
    while (gate.releasing.load(std::memory_order_seq_cst) != 0) std::this_thread::yield();
}

ReleaseHold::~ReleaseHold() {
    auto& gate = Gate();
    std::vector<std::pair<rhi::ma::AllocationPtr, std::unique_ptr<RelocationRecord>>> parked;
    {
        // Under the list's lock, so a free that parks now lands in the list taken here or sees no hold.
        std::scoped_lock lock(gate.parkedMutex);
        if (gate.holds.fetch_sub(1, std::memory_order_seq_cst) != 1) return;
        parked.swap(gate.parked);
        gate.parkedBytes = 0;
    }
    for (auto& [allocation, record] : parked) {
        allocation.Reset();
        record.reset();
    }
}

ParkedReleases GetParkedReleases() noexcept {
    auto& gate = Gate();
    std::scoped_lock lock(gate.parkedMutex);
    return {gate.parked.size(), gate.parkedBytes};
}

} // namespace org::relocation

namespace org {

void BackedResource::SetRelocationClass(relocation::Class value) {
    m_relocationClass.store(static_cast<uint8_t>(value), std::memory_order_release);
    std::unique_lock<std::recursive_mutex> lock;
    if (auto* mutex = BackingMutex()) lock = std::unique_lock(*mutex);
    if (auto* record = relocation::RecordOf(RelocationAllocation())) record->SetClass(value);
    if (lock) lock.unlock();
    // A ReadOnly resource is a candidate for the stress switch. A resource not owned by a shared_ptr yet (still in its
    // constructor) is not registered.
    if (value == relocation::Class::ReadOnly)
        if (auto* resource = dynamic_cast<Resource*>(this))
            if (auto owned = resource->weak_from_this().lock()) relocation::Service::Get().NoteReadOnly(owned);
}

relocation::RelocationRecord* BackedResource::CurrentRelocationRecord() noexcept {
    std::unique_lock<std::recursive_mutex> lock;
    if (auto* mutex = BackingMutex()) lock = std::unique_lock(*mutex);
    return relocation::RecordOf(RelocationAllocation());
}

void BackedResource::ApplyRelocationClass(uint64_t resourceID) noexcept {
    auto* record = relocation::RecordOf(RelocationAllocation());
    if (!record) return;
    record->SetResourceID(resourceID);
    if (const auto value = GetRelocationClass()) record->SetClass(*value);
    // The weak pointer is empty while the resource is still in its constructor. Its factory calls NoteRelocationOwner later.
    if (auto* resource = dynamic_cast<Resource*>(this)) {
        auto owner = resource->weak_from_this();
        if (!owner.expired()) record->SetOwner(std::move(owner));
    }
}

void BackedResource::NoteRelocationOwner() {
    std::unique_lock<std::recursive_mutex> lock;
    if (auto* mutex = BackingMutex()) lock = std::unique_lock(*mutex);
    auto* record = relocation::RecordOf(RelocationAllocation());
    if (!record) return;
    if (auto* resource = dynamic_cast<Resource*>(this)) record->SetOwner(resource->weak_from_this());
}

} // namespace org
