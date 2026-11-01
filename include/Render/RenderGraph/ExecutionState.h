#pragma once

#include "Render/RenderGraph/GraphCompiler.h"
#include "Resources/AliasingPlacement.h"
#include <rhi.h>
#include <span>
#include <deque>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace org::compiler {

struct PreparedStateRegion {
    CompileRange range;
    CompileResourceState state;
};

// Owner-thread snapshot of the actual backing state at frame preparation.
// The concrete handle, rather than a graph resource ID, is the cross-frame
// identity: replacing a backing starts with its newly captured initial state.
struct PreparedBackingState {
    uint64_t graphResourceID = 0;
    std::string graphResourceKey;
    rhi::ResourceHandle resource{};
    CompileResourceShape shape{};
    std::shared_ptr<const std::vector<PreparedStateRegion>> regions;
    std::shared_ptr<const AliasHeapGeneration> aliasHeap;
    const AliasHeapGeneration* aliasHeapIdentity = nullptr;
    uint64_t aliasPoolID = 0;
    uint64_t aliasOffset = 0;
    uint64_t aliasSize = 0;
    // Immutable realization metadata used to encode legal backend barriers.
    // D3D12 upload/readback buffers remain in fixed states and must not receive
    // enhanced buffer transition barriers.
    rhi::HeapType heapType = rhi::HeapType::DeviceLocal;
    // Frame-only report from an external producer. Its complete incoming states
    // supersede the last graph submission, transactionally at batch commit.
    bool authoritativeIncoming = false;
    // The host owns it (Resource::HostOwned): with external home states (Prepare's externalHome), each execution leaves it in
    // its home state on the graphics queue.
    bool hostOwned = false;
    // A resource shared across devices: its handle on each device instance (by index; [0] is resource). Barriers on a queue use
    // its device's handle, and a use on another device than the last is an external ownership handoff. Empty: one device.
    std::vector<rhi::ResourceHandle> deviceHandles;
    // The resource's name, captured while the decision trace (OpenRenderGraph/Trace.h) is on: diagnostics only.
    std::string debugName;
};

struct PreparedBatchBarriers {
    struct BeforePass {
        std::vector<rhi::TextureBarrier> textures;
        std::vector<rhi::BufferBarrier> buffers;
    };
    // Indexed in the same order as Schedule::batches[i].passes.
    std::vector<BeforePass> beforePass;
    std::vector<BeforePass> afterPass;
    // Batch-entry barriers: the handoff from the state an earlier execution left each backing in (the ledger's) to its first use
    // here, for every backing but aliased ones. beforePass and afterPass depend on the compiled graph alone.
    std::vector<rhi::TextureBarrier> textures;
    std::vector<rhi::BufferBarrier> buffers;
    // Entry barriers that take a texture on from Common after its queue-family acquire (in textures): a barrier batch of their own,
    // after the entry's, since barriers in one batch are unordered against each other.
    std::vector<rhi::TextureBarrier> acquired;
    std::vector<PreparedStateRegion> committedStates;
    std::vector<rhi::ResourceHandle> committedResources;
    // The compiler resource index of each committed backing (parallel to committedResources): where an indexed commit finds its
    // ledger entry (BackingStateAdmissionLedger::CommitBatch with backings).
    std::vector<uint32_t> committedIndices;
    // Queue slot that performed each committed state update. Kept parallel to
    // committedStates/resources so the admission ledger can distinguish a
    // same-queue memory dependency from a timeline-ordered queue handoff.
    std::vector<uint32_t> committedQueues;
    std::vector<PreparedBackingState> seeds;
    std::vector<rhi::ResourceHandle> authoritativeSeeds;
};

struct PreparedExecutionBarrierPlan {
    std::vector<PreparedBatchBarriers> batches;
    // Textures an earlier execution left on another queue (an open admission's): released on that queue, by a list submitted there
    // ahead of the execution, and acquired from Common by the entry barriers of the consumers, which wait for the release.
    struct Release {
        uint32_t queue = 0;
        std::vector<rhi::TextureBarrier> textures;
        std::vector<uint32_t> consumers;
    };
    std::vector<Release> releases;
    // Host-owned textures whose last use is on another queue than the graphics queue (external home states): released there after
    // that use to Common, and acquired on the graphics queue by these barriers (textures), then taken to their home layouts (home,
    // a barrier batch of its own), in a list submitted after the producers (batches). The host uses them only after it.
    struct HomeAcquires {
        std::vector<rhi::TextureBarrier> textures;
        std::vector<rhi::TextureBarrier> home;
        std::vector<uint32_t> producers;
    } homeAcquires;
};

// An entry-only plan (BackingStateAdmissionLedger::Prepare's entryOnly) and what the ledger held of each resource its entry steps
// name when it was made: per resource, whether it had an entry and its fingerprint (its cells and queues). While the ledger holds
// the same (BackingStateAdmissionLedger::Predicts), Prepare would make this plan again: a frame repeats, so an execution mostly
// finds what the same executable found the frame before.
struct StatePrediction {
    std::shared_ptr<const PreparedExecutionBarrierPlan> plan;
    struct Entry {
        uint32_t resource = 0;  // compiler resource index
        bool present = false;
        uint64_t fingerprint = 0;
    };
    std::vector<Entry> entries;
};

// An earlier access the access ledger holds: a producer's (a stream's batch, BackingAccessAdmissionLedger::OpenStream), which is
// that batch's latest signal, or a fixed point.
struct AccessRef {
    uint32_t producer = 0;  // 0: `point`
    ExecutionTimelinePoint point{};
    bool operator==(const AccessRef& other) const { return producer == other.producer && (producer || point == other.point); }
};

// The same for the access ledger (BackingAccessAdmissionLedger::AppendIncomingWaits): what it held of each resource the batches use
// first (its writers and readers, as references), and the earlier accesses each batch's waits came from. While the ledger holds the
// same references, the waits are those accesses' points now.
struct AccessPrediction {
    bool valid = false;
    std::vector<StatePrediction::Entry> entries;
    std::vector<std::vector<AccessRef>> waits;  // per batch, each once
};

namespace detail {
// Ledger entries by backing key, in storage whose addresses hold until the entry is erased. A backing set's entries are found by
// key once (Resolve, by the set's id: AdmissionBackingSet::Id) and from then on by compiler resource index, so an admission's
// per-resource work needs no lookup. Not thread-safe: the ledgers are used under the sequencer's lock.
template<class T>
class BackingTable {
public:
    T* Find(uint64_t key) noexcept {
        const auto found = m_index.find(key);
        return found == m_index.end() ? nullptr : &m_entries[found->second];
    }
    const T* Find(uint64_t key) const noexcept {
        const auto found = m_index.find(key);
        return found == m_index.end() ? nullptr : &m_entries[found->second];
    }
    bool Contains(uint64_t key) const noexcept { return m_index.contains(key); }
    T& Emplace(uint64_t key, bool* inserted = nullptr) {
        if (const auto found = m_index.find(key); found != m_index.end()) {
            if (inserted) *inserted = false;
            return m_entries[found->second];
        }
        uint32_t slot = 0;
        if (!m_free.empty()) {
            slot = m_free.back();
            m_free.pop_back();
            m_entries[slot] = T{};
        } else {
            slot = static_cast<uint32_t>(m_entries.size());
            m_entries.emplace_back();
        }
        m_index.emplace(key, slot);
        if (inserted) *inserted = true;
        return m_entries[slot];
    }
    bool Erase(uint64_t key) {
        const auto found = m_index.find(key);
        if (found == m_index.end()) return false;
        m_entries[found->second] = T{};
        m_free.push_back(found->second);
        m_index.erase(found);
        ++m_generation;
        return true;
    }
    void Clear() {
        m_index.clear();
        m_entries.clear();
        m_free.clear();
        m_resolved.clear();
        ++m_generation;
    }
    size_t Size() const noexcept { return m_index.size(); }
    template<class Fn> void ForEach(Fn&& fn) { for (auto& [key, slot] : m_index) fn(key, m_entries[slot]); }
    // Per resource of the backing set, its entry (null: none when resolved; the caller finds it by key then, and keeps what it
    // finds here). backingSet 0: no set to remember, resolved for this call.
    std::vector<T*>& Resolve(std::span<const PreparedBackingState> backings, uint64_t backingSet) const {
        auto& resolved = backingSet ? m_resolved[backingSet] : m_scratch;
        if (!backingSet || resolved.generation != m_generation || resolved.entries.size() != backings.size()) {
            if (m_resolved.size() > 512) {
                m_resolved.clear();
                return Resolve(backings, backingSet);
            }
            resolved.generation = m_generation;
            resolved.entries.assign(backings.size(), nullptr);
            for (size_t i = 0; i < backings.size(); ++i)
                if (backings[i].resource.valid())
                    resolved.entries[i] = const_cast<T*>(Find((uint64_t{backings[i].resource.generation} << 32) | backings[i].resource.index));
        }
        return resolved.entries;
    }
private:
    std::unordered_map<uint64_t, uint32_t> m_index;
    mutable std::deque<T> m_entries;  // stable addresses
    std::vector<uint32_t> m_free;
    uint64_t m_generation = 1;  // moves when an entry goes (Erase, Clear): an address resolved before may be another's now
    struct Resolved {
        uint64_t generation = 0;
        std::vector<T*> entries;
    };
    mutable std::unordered_map<uint64_t, Resolved> m_resolved;
    mutable Resolved m_scratch;
};
}

// Throws unless the backings are the structure's resources, in its order (what every ledger's Prepare expects).
void ValidatePreparedBackingOrder(const GraphCompileStructure& structure, std::span<const PreparedBackingState> backings);

// Ordered admission state. Compile workers never access this object. Prepare is
// transactional; CommitBatch is called only for work known to have submitted.
class BackingStateAdmissionLedger {
public:
    // closeToHome: the execution is closed. Every resource it touches leaves it in its home state - the
    // backing's seeded region state, or the state its first access in the graph expects - so the next
    // execution's entry states are the same whatever ran before it (or whether anything did). Textures get
    // a layout transition after their last access; the ledger records home as their final state. Closed
    // executions provide their own memory visibility (a full barrier at entry, see FrameAdmission::closed),
    // so buffers need no exit barrier. Every resource must be used from a single queue.
    // entryOnly: for an execution whose body is recorded already (from an earlier plan of the same executable and backings): the
    // entry barriers, releases, seeds and committed states alone, from the plan's entry and exit steps (SymbolicStatePlan); the
    // body's barriers (beforePass, afterPass) are left out. validated: the backings were checked against the graph's resource order
    // (ValidatePreparedBackingOrder) when they were made.
    // backingSet: the backings' set (AdmissionBackingSet::Id), whose ledger entries are found once; 0 for none.
    // externalHome: external home states, an interop option. Every host-owned texture (PreparedBackingState::hostOwned) the
    // execution touches leaves it in its home state (as closeToHome's) on the graphics queue (slot 0), so the host finds it where
    // it left it. A last use on another queue releases it to the graphics queue (homeAcquires). Other resources are left as the
    // execution leaves them, for the ledger to hand off.
    PreparedExecutionBarrierPlan Prepare(
        const Schedule& graph, std::span<const PreparedBackingState> initial,
        std::span<const rhi::ResourceHandle> invalidated = {}, bool closeToHome = false, bool entryOnly = false,
        bool validated = false, uint64_t backingSet = 0, bool externalHome = false) const;
    void CommitBatch(const PreparedBatchBarriers&);
    // The same, each commit's entry found by its compiler resource index (PreparedBatchBarriers::committedIndices) in the backings'
    // set (backingSet as for Prepare).
    void CommitBatch(const PreparedBatchBarriers& batch, std::span<const PreparedBackingState> backings, uint64_t backingSet);
    // The prediction's entries taken anew: what the ledger holds now of each resource the graph's entry steps name.
    void Capture(StatePrediction& prediction, const Schedule& graph, std::span<const PreparedBackingState> initial,
        uint64_t backingSet = 0) const;
    // Whether the ledger still holds what the prediction's entries took (its plan stands).
    bool Predicts(const StatePrediction& prediction, std::span<const PreparedBackingState> initial, uint64_t backingSet = 0) const;
    void Invalidate(std::span<const rhi::ResourceHandle> resources);
    size_t BackingCount() const noexcept { return m_states.Size(); }
    void Reset() { m_states.Clear(); }
private:
    struct StateGrid {
        CompileResourceShape shape{};
        std::vector<CompileResourceState> cells;
        std::vector<uint32_t> queues;
        uint64_t fingerprint = 0;  // of cells and queues, kept by every change (StatePrediction)
    };
    detail::BackingTable<StateGrid> m_states;
};

// Ordered, submission-backed cross-frame hazard state. Unlike symbolic graph
// dependencies, these points refer only to work that was actually submitted.
// The ledger is keyed by concrete backing identity and subresource cell so
// unrelated resources/ranges and queues remain independent.
class BackingAccessAdmissionLedger {
public:
    // A stream: the executions of one executable with one backing set (the same accesses to the same backings each time). Each of
    // its batches is a producer: the cells it touches name the producer instead of a point, and its commit sets the producer's
    // latest signal once, so a cell changes only when who touched it last changes. A closed stream's producers keep their last
    // signals (fixed points from then on). Returns the stream's id (never 0).
    uint64_t OpenStream(uint32_t batches);
    void CloseStream(uint64_t stream);
    // backingSet: as for BackingStateAdmissionLedger::Prepare. capture: filled with what the waits came from, for PredictedWaits.
    void AppendIncomingWaits(const Schedule& graph,
        std::span<const PreparedBackingState> resources,
        std::span<const ExecutionTimelinePoint> queues,
        std::vector<std::vector<ExecutionTimelinePoint>>& incoming, uint64_t backingSet = 0,
        AccessPrediction* capture = nullptr) const;
    // AppendIncomingWaits from a capture, when the ledger still holds what it was captured from: the same waits, at the points
    // their producers have reached since. False (incoming untouched) when it does not.
    bool PredictedWaits(const AccessPrediction& prediction, const Schedule& graph,
        std::span<const PreparedBackingState> resources,
        std::span<const ExecutionTimelinePoint> queues,
        std::vector<std::vector<ExecutionTimelinePoint>>& incoming, uint64_t backingSet) const;
    // Per batch, (compiler resource index, wait).
    using ResourceWaits = std::vector<std::vector<std::pair<uint32_t, ExecutionTimelinePoint>>>;
    // For each boundary access (incomingBoundaryAccessesByBatch) to a selected resource (selected: one flag per compiler
    // resource), every earlier access the ledger holds - its writer and all its readers, on every queue, the consumer's own
    // included: what a write into the resource on any queue ahead of the execution must wait for (RenderGraph's queue-scoped
    // ticket uploads), where AppendIncomingWaits gives a read only the writer, and nothing on its own queue. out: one list per batch.
    void AppendBoundaryAccessWaits(const Schedule& graph,
        std::span<const PreparedBackingState> resources,
        std::span<const uint8_t> selected,
        ResourceWaits& out, uint64_t backingSet = 0) const;
    // stream: the execution's (OpenStream), or 0 for none: its cells then hold its signals as fixed points.
    void Commit(const Schedule& graph,
        std::span<const PreparedBackingState> resources,
        const GraphExecutionTimeline& execution, uint32_t batchCount = UINT32_MAX, uint64_t backingSet = 0, uint64_t stream = 0);
    void Reset() { m_accesses.Clear(); }
    void ResolvePlannedPoints(const std::unordered_map<uint64_t, ExecutionTimelinePoint>& points);
    bool RetireCompleted(rhi::ResourceHandle resource, const std::map<uint64_t, uint64_t>& completed);
    size_t BackingCount() const noexcept { return m_accesses.Size(); }
private:
    struct Cell {
        AccessRef writer{};
        std::vector<AccessRef> readers;  // at most one per timeline
    };
    struct AccessGrid {
        CompileResourceShape shape{};
        std::vector<Cell> cells;
        uint64_t fingerprint = 0;  // of the cells' references, kept by every change (AccessPrediction)
    };
    struct Producer {
        ExecutionTimelinePoint signal{};  // the latest; its timeline is set by the first commit
        bool closed = false;
    };
    ExecutionTimelinePoint PointOf(const AccessRef& ref) const noexcept {
        return ref.producer ? m_producers[ref.producer].signal : ref.point;
    }
    uint64_t TimelineOf(const AccessRef& ref) const noexcept { return PointOf(ref).timeline; }
    static uint64_t Fingerprint(const AccessGrid& grid);
    // Closed producers' references become their points, and the producers are reused (after enough have closed).
    void FixClosedProducers();
    detail::BackingTable<AccessGrid> m_accesses;
    std::vector<Producer> m_producers{1};  // [0] unused: AccessRef::producer 0 is a point
    std::vector<uint32_t> m_freeProducers;
    size_t m_closedProducers = 0;
    std::unordered_map<uint64_t, std::vector<uint32_t>> m_streams;  // id -> its batches' producers
    uint64_t m_nextStream = 1;
};

// Tracks submitted use of physical placed-resource ranges. Different graph
// resource handles that overlap the same heap generation conflict even for
// read/read access because they represent distinct alias occupants.
//
// Representation: one occupant per (heap, handle) with the placements it has
// committed and its latest signal per timeline. Committing assigns a monotonic
// sequence to the placement; a query range conflicts with every other
// occupant range that overlaps it and was committed after the query occupant
// last committed that same range. This is equivalent to the byte-interval
// ownership model (a later commit takes over the overlapping bytes) but needs
// no interval fragmentation, so steady-state placements cost O(occupants).
class AliasAccessAdmissionLedger {
public:
    std::vector<rhi::ResourceHandle> ApplyInitialStates(
        const Schedule& graph,
        std::span<const PreparedBackingState> resources) const;
    void AppendIncomingWaits(const Schedule& graph,
        std::span<const PreparedBackingState> resources,
        std::span<const ExecutionTimelinePoint> queues,
        std::vector<std::vector<ExecutionTimelinePoint>>& incoming) const;
    void Commit(const Schedule& graph,
        std::span<const PreparedBackingState> resources,
        const GraphExecutionTimeline& execution, uint32_t batchCount = UINT32_MAX);
    void Reset() { m_heaps.clear(); m_heapOwners.clear(); }
    void ResolvePlannedPoints(const std::unordered_map<uint64_t, ExecutionTimelinePoint>& points);
    bool RetireCompleted(const AliasHeapGeneration* heap, const std::weak_ptr<const AliasHeapGeneration>& owner,
        const std::map<uint64_t, uint64_t>& completed);
    size_t HeapCount() const noexcept { return m_heaps.size(); }
private:
    struct OccupantRange {
        uint64_t begin = 0, end = 0;
        uint64_t sequence = 0; // Commit order of the latest submission using this placement.
    };
    struct Occupant {
        rhi::ResourceHandle handle{};
        std::vector<OccupantRange> ranges;             // Usually one; more after a re-placement.
        std::vector<ExecutionTimelinePoint> accesses;  // Latest signal per timeline.
    };
    struct Heap {
        std::vector<Occupant> occupants;
        std::unordered_map<uint64_t, uint32_t> occupantByHandle;
    };
    // Sequence of the query occupant's own commit at exactly this placement,
    // zero when it never committed there. Ranges committed later conflict.
    static uint64_t OwnSequence(const Heap& heap, const PreparedBackingState& resource);
    // Visits every other occupant with a range overlapping the query placement
    // that was committed after the query occupant's own commit there.
    template<class Fn> void ForEachConflict(const PreparedBackingState& resource, Fn&& fn) const;
    std::unordered_map<const AliasHeapGeneration*, Heap> m_heaps;
    std::unordered_map<const AliasHeapGeneration*, std::weak_ptr<const AliasHeapGeneration>> m_heapOwners;
    uint64_t m_sequence = 0;
};

} // namespace org::compiler
