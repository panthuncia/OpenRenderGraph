#include "Resources/RelocationRecord.h"

#include <mutex>

#include <rhi_allocator.h>

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

} // namespace detail

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
}

} // namespace org
