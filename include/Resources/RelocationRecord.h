#pragma once

#include <array>
#include <atomic>
#include <cstdint>

namespace rhi::ma {
class Allocation;
}

// Owner records for relocation (docs: org-relocation.md, "Movability").
// Each tracked allocation carries a RelocationRecord. Allocation::SetPrivateData points at it, so a defragmenter that only
// sees allocations can find out whether a move is allowed. The record lives as long as the allocation's TrackedHandle.
namespace org::relocation {

// When a backing may move to new memory.
enum class Class : uint8_t {
    ReadOnly,  // Any time. Nothing writes it any more (a sealed resource or version).
    Written,   // Only while the GPU is idle. Passes write it.
    Pinned,    // Never. Mapped heaps, alias heaps, shared resources, handed-out addresses.
    Count
};

// Why a relocation was not committed.
enum class IgnoreReason : uint8_t {
    NoRecord,        // The backing has no owner record, so nothing says it may move.
    Pinned,          // Its class is Pinned.
    WrittenNotIdle,  // Its class is Written and the GPU was not idle.
    Expired,         // The resource is gone.
    Stale,           // Its backing changed between the move's preparation and its commit.
    Unsupported,     // A kind of resource the follow path cannot move (multisampled, depth-stencil, exempt, ...).
    Failed,          // Making the new backing or the copy failed.
    Count
};

const char* ClassName(Class value) noexcept;
const char* IgnoreReasonName(IgnoreReason value) noexcept;

// The record an allocation's private data points at.
// The class may change while the allocation lives, so it is atomic. The live counts by class follow the change.
class RelocationRecord {
public:
    RelocationRecord(Class initial, uint64_t resourceID, uint64_t sizeBytes) noexcept;
    ~RelocationRecord();
    RelocationRecord(const RelocationRecord&) = delete;
    RelocationRecord& operator=(const RelocationRecord&) = delete;

    Class GetClass() const noexcept { return m_class.load(std::memory_order_acquire); }
    void SetClass(Class value) noexcept;
    // The global resource ID of the owning resource (Resource::GetGlobalResourceID). 0 when it has none.
    uint64_t ResourceID() const noexcept { return m_resourceID.load(std::memory_order_acquire); }
    void SetResourceID(uint64_t id) noexcept { m_resourceID.store(id, std::memory_order_release); }
    uint64_t SizeBytes() const noexcept { return m_sizeBytes; }

private:
    std::atomic<Class> m_class;
    std::atomic<uint64_t> m_resourceID;
    uint64_t m_sizeBytes = 0;
};

// The record of an allocation. Null when the allocation has none (untracked allocations, placed resources).
RelocationRecord* RecordOf(const rhi::ma::Allocation* allocation) noexcept;

// Counters for the application's VRAM report. All of them count since startup, except the live ones.
struct Stats {
    uint64_t moved = 0;       // relocations committed
    uint64_t movedBytes = 0;  // their sizes
    std::array<uint64_t, static_cast<size_t>(IgnoreReason::Count)> ignored{};  // relocations not committed, by reason
    uint64_t retired = 0;       // old backings freed after their move
    uint64_t retiredBytes = 0;  // their sizes
    uint64_t awaitingRetirement = 0;       // live: old backings that moved and are not freed yet
    uint64_t awaitingRetirementBytes = 0;  // live: their sizes
    std::array<uint64_t, static_cast<size_t>(Class::Count)> records{};      // live: owner records by class
    std::array<uint64_t, static_cast<size_t>(Class::Count)> recordBytes{};  // live: their allocation sizes
    uint64_t journalHead = 0;  // entries written to the relocation journal
};
// Any thread.
Stats GetStats() noexcept;

namespace detail {
// Counter updates for the relocation service. Any thread.
void NoteMoved(uint64_t bytes) noexcept;
void NoteIgnored(IgnoreReason reason) noexcept;
void NoteAwaitingRetirement(uint64_t bytes) noexcept;
void NoteRetired(uint64_t bytes) noexcept;
void NoteJournalHead(uint64_t head) noexcept;
}

} // namespace org::relocation
