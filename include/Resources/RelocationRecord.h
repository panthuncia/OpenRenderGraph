#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

namespace rhi::ma {
class Allocation;
}

namespace org {
class Resource;
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
    // The resource whose backing this allocation is. The defragmenter, which only sees allocations, moves the resource through it.
    // Set by the owner (BackedResource::ApplyRelocationClass, BackedResource::NoteRelocationOwner). Null when no resource owns the
    // allocation, or when the resource is gone.
    std::shared_ptr<Resource> Owner() const;
    void SetOwner(std::weak_ptr<Resource> owner);

private:
    std::atomic<Class> m_class;
    std::atomic<uint64_t> m_resourceID;
    uint64_t m_sizeBytes = 0;
    mutable std::mutex m_ownerMutex;
    std::weak_ptr<Resource> m_owner;
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
    // The defragmenter (relocation::Defragmenter).
    uint64_t defragCampaigns = 0;      // campaigns started
    uint64_t defragPasses = 0;         // passes run
    uint64_t defragMoves = 0;          // moves handed to the commit
    uint64_t defragMovedBytes = 0;     // their sizes
    uint64_t defragIgnored = 0;        // moves the allocator proposed that the defragmenter left in place
    uint64_t defragBlocksFreed = 0;    // blocks fewer at the start of a pass than at the start of the one before
    uint64_t defragMicroseconds = 0;   // time in passes, without the waits for the GPU
    uint64_t defragLingering = 0;      // live: old ranges kept until no recording names them
    uint64_t defragLingeringBytes = 0; // live: their sizes
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
void NoteDefragCampaign() noexcept;
// One pass: the moves handed to the commit, their bytes, the proposed moves left in place, the blocks freed since the last pass
// and the pass's time.
void NoteDefragPass(uint64_t moves, uint64_t bytes, uint64_t ignored, uint64_t blocksFreed, uint64_t microseconds) noexcept;
void NoteDefragLingering(int64_t count, int64_t bytes) noexcept;
}

} // namespace org::relocation
