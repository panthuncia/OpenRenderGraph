#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <vector>

#include <rhi.h>

#include "Render/Runtime/SubmissionFrontier.h"
#include "Resources/RelocationRecord.h"

namespace org {
class BufferBase;
class Resource;
class ExternalClient;
class GpuTextureBacking;
class GpuBufferBacking;
}

// Resource relocation: a resource keeps its identity (Resource*, global ID, versioned slot) while its backing moves to new
// memory. See docs/development/plans/org-relocation.md in the application.
//
// Relocations come from sources (IRelocationSource). Service::Commit takes them at the frame boundary, on the GraphHost owner
// thread, and for each one:
// 1. copies the old backing into the new one, when the relocation asks for it (the relocation batch),
// 2. swaps the backing under BackedResource::BackingMutex, which bumps the backing generation,
// 3. updates the descriptors: new slots in steady state, the same slots rewritten when the GPU is idle,
// 4. retires the old backing behind the descriptor manager's frontier (DescriptorHeapManager::RetireRelocated),
// 5. appends an entry to the journal, which clients that cache bindless indices read once a frame.
namespace org::relocation {

// Who produced a relocation.
enum class Source : uint8_t {
    Dxvk,          // DXVK moved one of its resources and already copied the data.
    Defragmenter,  // ORG's defragmenter. The Allocation already names the new resource when the backing is adopted.
    Owner,         // An owner that rebuilt its resource itself (streaming windows).
    Stress,        // The ORG_RELOCATION_STRESS debug switch.
    Count
};
const char* SourceName(Source value) noexcept;

// One move of one resource's backing.
//
// What each source fills in:
// - DXVK: resource (an ExternalTextureResource or ExternalBufferResource), next (the new storage, imported), ready (DXVK's
//   copy), retired (whose deleter gives DXVK its storage back).
// - Defragmenter: resource, fromGeneration, copyTarget (a wrapper of the new resource at the destination), fromBuffer for a
//   buffer, and in BeforeAdopt retired (the retained allocation). nextTexture and nextBuffer stay null: after EndPass the
//   backing's Allocation names the new resource.
// - Graph resources made again (MakeGraphRelocation, the stress switch): resource, fromGeneration, nextTexture or nextBuffer,
//   copyTarget.
struct Relocation {
    Relocation();
    ~Relocation();
    Relocation(Relocation&&) noexcept;
    Relocation& operator=(Relocation&&) noexcept;

    // Stable identity: Resource::GetGlobalResourceID. Filled from `resource` when 0.
    uint64_t resourceID = 0;
    // The resource that moves.
    std::shared_ptr<Resource> resource;
    // The backing generation the relocation was made for (BackedResource::GetBackingGeneration). Commit skips it when the
    // backing changed since. 0: not checked (imports, whose sources do not know it).
    uint64_t fromGeneration = 0;
    // The generation after the move. Set by Commit.
    uint64_t toGeneration = 0;
    // Imports: the new native resource.
    rhi::ResourcePtr next;
    // Graph resources: the new backing object, made with the same description. Null when the backing's Allocation already
    // names the new resource.
    std::unique_ptr<GpuTextureBacking> nextTexture;
    std::unique_ptr<GpuBufferBacking> nextBuffer;
    // When set, Commit copies the resource's current backing into this resource in the relocation batch, before the switch.
    // It names the new storage.
    std::shared_ptr<Resource> copyTarget;
    // Keeps the old backing (resource and memory) alive until the GPU is done with it.
    std::shared_ptr<const void> retired;
    // The GPU point after which the new storage holds the data, when another queue or API copied it. Every queue the
    // graphs use waits for it before their next work. Invalid when nothing needs waiting for.
    runtime::SubmittedPoint ready{};
    // The size of the old backing, for the counters. Commit fills it from the owner record when 0.
    uint64_t sizeBytes = 0;
    // Buffers: the native buffer the move leaves. Needed when the backing's Allocation already names the new resource at the
    // switch (the defragmenter), so the switch can hand the buffer's view ref pair over and read the old device address. Commit
    // takes the current native buffer when it is not set.
    rhi::ResourceHandle fromBuffer{};
    // Buffers: the device addresses before and after the move, set by Commit when it adopts the move. 0 when the buffer has
    // none, and for textures.
    uint64_t oldAddress = 0;
    uint64_t newAddress = 0;
    Source source = Source::Owner;
    // Set by Commit when it will not adopt the relocation (its resource is gone, its backing changed, its copy failed). A source
    // reads it in BeforeAdopt, and may set it there itself to call the move off.
    bool skipped = false;
};

struct CommitContext;

// A producer of relocations. Registered with Service::AddSource.
class IRelocationSource {
public:
    virtual ~IRelocationSource() = default;
    // Owner thread, once per commit, repeated while it fills the whole span: the relocations ready now, written into out
    // from its start. Returns how many it wrote. A source decides what may move (MayMove); Commit only skips moves whose
    // resource is gone, whose backing changed, or whose kind it cannot move.
    virtual size_t Poll(std::span<Relocation> out) = 0;
    // Owner thread, once per commit, before the first Poll. A source that must know whether the GPU is idle (the defragmenter
    // moves Written backings only then) reads it here.
    virtual void BeginCommit(const CommitContext& context) { (void)context; }
    // Owner thread, once per commit in which Poll returned any: after the relocation batch was submitted and before the
    // backings switch. A defragmenter ends its pass here (EndPass), so each Allocation names its new resource by then.
    virtual void BeforeAdopt(std::span<Relocation> polled) { (void)polled; }
    // Owner thread, once per commit in which Poll returned any: after the backings switched. Each relocation's toGeneration is set
    // when it was adopted, and skipped when it was not. A defragmenter lets go of the backing locks it held since Poll here.
    virtual void AfterAdopt(std::span<Relocation> polled) { (void)polled; }
};

// Whether a backing with this record may move now. When it may not, why says why.
bool MayMove(const RelocationRecord* record, bool gpuIdle, IgnoreReason* why = nullptr) noexcept;

// The rules a buffer adds to MayMove. A PerFrame buffer (MemoryGroup::PerFrame) never moves (Pinned). A buffer with a buffer view
// ref pair (rhi::Device::HasBufferViewRef) moves only while the GPU is idle (WrittenNotIdle otherwise), because the move rewrites
// the pair in place. Owner thread.
bool BufferMayMove(BufferBase& buffer, bool gpuIdle, IgnoreReason* why = nullptr);

// The journal of committed relocations: what moved, and to which backing generation. Clients that cache a resource's bindless
// index or raw handle read it once a frame with a cursor of their own and refresh what moved.
struct JournalEntry {
    uint64_t sequence = 0;
    uint64_t resourceID = 0;
    uint64_t toGeneration = 0;
    Source reason = Source::Owner;
    // Buffers: the device address before and after the move, so a client that wrote the old address into GPU data can rebase
    // it (new = old + (newAddress - oldAddress) for an address inside the buffer). 0 when the buffer has none, and for textures.
    uint64_t oldAddress = 0;
    uint64_t newAddress = 0;
};

class RelocationJournal {
public:
    explicit RelocationJournal(size_t capacity = 4096) : m_capacity(capacity ? capacity : 1) {}
    // Owner thread (Service::Commit).
    void Append(uint64_t resourceID, uint64_t toGeneration, Source reason, uint64_t oldAddress = 0, uint64_t newAddress = 0);
    // Any thread. The sequence the next entry gets. A new client starts its cursor here.
    uint64_t Head() const noexcept { return m_head.load(std::memory_order_acquire); }
    // Any thread. Appends the entries from `cursor` on to `out`, and moves `cursor` to Head(). Returns false when entries
    // after `cursor` were dropped (the journal keeps its last `capacity` entries). The client then re-reads everything it
    // caches; `out` still holds the entries that are kept.
    bool Read(uint64_t& cursor, std::vector<JournalEntry>& out) const;

private:
    mutable std::mutex m_mutex;
    std::deque<JournalEntry> m_entries;
    size_t m_capacity;
    std::atomic<uint64_t> m_head{0};
};

// What a commit needs from the host. GraphHost::CommitRelocations fills it from its graphs.
struct CommitContext {
    // Submits the relocation batch (the copies). Required when any relocation has a copyTarget.
    ExternalClient* client = nullptr;
    // The client's graphics queue slot.
    uint32_t copySlot = 0;
    // Every queue the graphs run on. Each waits for the relocations' ready points.
    std::vector<rhi::Queue> queues;
    // The GPU is idle: descriptors are rewritten in place (bindless indices stay), and Written backings may move.
    bool gpuIdle = false;
};

class Service {
public:
    static Service& Get();

    // Any thread. A source is polled at every commit until it is removed.
    void AddSource(std::shared_ptr<IRelocationSource> source);
    void RemoveSource(const IRelocationSource* source);

    // Owner thread, at the frame boundary before any preparation (GraphHost::SealSubmitted calls it through
    // GraphHost::CommitRelocations). Returns how many relocations were committed. Never runs inside a preparation.
    size_t Commit(const CommitContext& context);

    // Any thread.
    const RelocationJournal& Journal() const noexcept { return m_journal; }
    // Any thread: whether the resource's backing of this generation was moved away. A recording that bound that backing is
    // stale and records again (RenderGraph::PersistentRecordingBackingsCurrent).
    bool MovedFrom(uint64_t resourceID, uint64_t generation) const;

    // The ORG_RELOCATION_STRESS debug switch: each commit makes this many ReadOnly graph backings again at a new
    // allocation, copies them in the relocation batch and commits the moves, as DXVK's stress relocations do. The environment
    // variable is read once; SetStress overrides it. 0 is off. Any thread.
    void SetStress(uint32_t perCommit) noexcept;
    uint32_t Stress() const noexcept { return m_stress.load(std::memory_order_acquire); }

    // A resource was marked ReadOnly (BackedResource::SetRelocationClass). The stress switch picks from these. Any thread.
    void NoteReadOnly(const std::shared_ptr<Resource>& resource);

    ~Service();

private:
    Service();
    struct Candidates;
    // Appends up to `count` relocations of resources marked ReadOnly, taken in turn (the stress switch).
    void MakeStressRelocations(uint32_t count, std::vector<Relocation>& out);
    // Records and submits the relocation batch: one copy per relocation that has a copyTarget. Returns false when the
    // submission failed; the relocations are then ignored.
    bool SubmitCopies(const CommitContext& context, std::span<Relocation*> copies);
    void NoteMovedFrom(uint64_t resourceID, uint64_t generation);

    std::mutex m_sourcesMutex;
    std::vector<std::shared_ptr<IRelocationSource>> m_sources;
    RelocationJournal m_journal;
    mutable std::shared_mutex m_movedMutex;
    // Per resource, the generations its backing moved away from (the last few).
    std::unordered_map<uint64_t, std::vector<uint64_t>> m_movedFrom;
    std::atomic<uint32_t> m_stress{0};
    std::unique_ptr<Candidates> m_candidates;
};

inline Service& GetService() { return Service::Get(); }

// Makes a relocation of a graph resource that has a backing of its own (a PixelBuffer, or a BufferBase): a new backing of the
// same description, and a copy of the current backing into it in the relocation batch. Null when the resource may not move
// now (MayMove with gpuIdle) or cannot move this way: no backing of its own, placed in an alias heap, multisampled,
// depth-stencil, externally immutable, or represented on other devices. why says which.
// Owner thread. The caller hands the result to a commit (from a source's Poll).
std::unique_ptr<Relocation> MakeGraphRelocation(const std::shared_ptr<Resource>& resource, Source source, bool gpuIdle = false,
    IgnoreReason* why = nullptr);

namespace detail {
// The resource a relocation batch copies into (Relocation::copyTarget): a view of a native resource with the shape of `like`. It
// does not own the native resource. lease keeps whatever owns it alive while the wrapper lives. Its state starts as a new
// resource's: Undefined for a texture, Common for a buffer.
std::shared_ptr<Resource> MakeCopyTarget(rhi::Resource resource, Resource& like, std::shared_ptr<const void> lease);
}

} // namespace org::relocation
