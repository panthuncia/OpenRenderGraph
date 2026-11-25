#pragma once

#include "Render/RenderGraph/GraphCompiler.h"
#include "Resources/AliasingPlacement.h"
#include <rhi.h>
#include <algorithm>
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
    // An aliased backing whose contents carry over into this graph's executions from an earlier execution in the same frame:
    // - a resource of a whole-frame placement in an epoch after its first one,
    // - a resource that keeps its contents across the runs of a repeated epoch (Resource::SetContentAcrossRuns).
    // No other occupant of its bytes is used in this graph before it. So it enters from the state the ledger holds, in its batch's
    // entry barriers, like a backing of its own. It enters from Undefined only when another occupant took its bytes since its
    // last use (an alias activation), or when the ledger has no state for it. A plain aliased backing enters every execution
    // from Undefined, in its body barrier.
    bool aliasContinues = false;
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
    // Externally immutable (Resource::GraphOwnership::ExternalImmutableShaderResource): already in its shader-read state, and
    // never written while the graph reads it. The graph plans no barriers, seeds or commits for it, and keeps no ledger entry.
    bool exempt = false;
    // A resource shared across devices: its handle on each device instance (by index; [0] is resource). Barriers on a queue use
    // its device's handle, and a use on another device than the last is an external ownership handoff. Empty: one device.
    std::vector<rhi::ResourceHandle> deviceHandles;
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
    // ledger entry (BackingStateAdmissionLedger::CommitBatches).
    std::vector<uint32_t> committedIndices;
    // Queue slot that performed each committed state update. Kept parallel to
    // committedStates/resources so the admission ledger can distinguish a
    // same-queue memory dependency from a timeline-ordered queue handoff.
    std::vector<uint32_t> committedQueues;
    // Per commit (parallel to committedStates): the batch from which on later batches of the execution commit all its cells again,
    // or UINT32_MAX. When that batch commits too (CommitBatches), the ledger's cells end in the later states, so this commit is
    // left out: a resource used in two batches is not taken through its earlier state every frame.
    std::vector<uint32_t> committedSupersededAt;
    std::vector<PreparedBackingState> seeds;
    std::vector<rhi::ResourceHandle> authoritativeSeeds;
    // The state ledger's last commit of this batch (BackingStateAdmissionLedger::CommitBatches): the ledger, its change count
    // after the commit, and what the commit was made with. A grid that has not changed since is as that commit left it, so the
    // same commit again writes nothing there and skips it. Touched only under the ledger's lock.
    struct LastCommit {
        const void* ledger = nullptr;
        uint64_t changes = 0;
        uint64_t backingSet = 0;
        uint64_t key = 0;  // of the batches committed with it and the unmoved resources
    };
    mutable LastCommit lastCommit;
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
    // The aliased backings the plan was made with entering from Undefined (AliasAccessAdmissionLedger::Resolve). A graph
    // whose transients share memory activates the same ones at every execution: the plan stands for an admission that activates
    // the same backings.
    std::vector<rhi::ResourceHandle> activations;
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

// The same for the access ledger (BackingAccessAdmissionLedger::AppendIncomingWaits): which resources the batches use first, and
// the earlier accesses each batch's waits came from. While none of those resources' grids changed since, the waits are those
// accesses' points now. The ledger's grids only grow (BackingAccessAdmissionLedger), so a capture that no longer holds never holds
// again: an admission keeps one.
struct AccessPrediction {
    struct Entry {
        uint32_t resource = 0;
        bool present = false;
        // A batch writes the resource, so its readers are waited for too: any change to the grid counts. Otherwise only its
        // writers can give waits, and only a change to them counts.
        bool readers = false;
    };
    bool valid = false;
    uint64_t generation = 0;  // the ledger's change count the capture was taken or last checked at
    std::vector<Entry> entries;
    // Per batch: its queue's timeline, and the accesses its waits come from, each once, without those on that timeline (queue
    // order covers them; a reference's timeline is fixed once it is in a cell).
    std::vector<uint64_t> consumers;
    std::vector<std::vector<AccessRef>> waits;
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
    // Moves when an entry goes (Erase, Clear).
    uint64_t Generation() const noexcept { return m_generation; }
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

// The resources no running pass of an execution touches: every pass that uses one skips this execution (skipped ahead, or
// skipped at execution). Such a resource stays as the execution found it. Its barriers (entry, body, releases) are left out,
// and the ledger keeps the state it had (BackingStateAdmissionLedger::CommitBatches' unmoved).
struct UntouchedResources {
    std::vector<uint32_t> indices;  // compiler resource indices, ascending
    std::vector<uint64_t> keys;     // their backings' handles on every device, (generation << 32) | index, ascending
    uint64_t signature = 0;         // of the indices; 0 when there are none
    bool Empty() const noexcept { return indices.empty(); }
    bool Holds(rhi::ResourceHandle handle) const noexcept {
        if (keys.empty()) return false;
        const uint64_t key = (uint64_t{handle.generation} << 32) | handle.index;
        return std::binary_search(keys.begin(), keys.end(), key);
    }
    bool HoldsIndex(uint32_t index) const noexcept { return std::binary_search(indices.begin(), indices.end(), index); }
};

// The resources no running pass of an execution touches. skips[batch][position]: the pass at that position of the batch
// skips this execution. late[batch]: the batch is recorded at execution; empty: every batch is. A pass of a batch recorded
// ahead counts as running for every resource but those in `ahead`: the resources whose every use was skipped ahead, which
// the lists recorded ahead left out. Reserved and exempt backings are never in the result.
UntouchedResources FindUntouched(const Schedule& graph, std::span<const PreparedBackingState> backings,
    std::span<const std::vector<uint8_t>> skips, std::span<const uint8_t> late = {}, const UntouchedResources* ahead = nullptr);

// Ordered admission state. Compile workers never access this object. Prepare is
// transactional; CommitBatch is called only for work known to have submitted.
class BackingStateAdmissionLedger {
public:
    // entryOnly: for an execution whose body is recorded already (from an earlier plan of the same executable and backings): the
    // entry barriers, releases, seeds and committed states alone, from the plan's entry and exit steps (SymbolicStatePlan); the
    // body's barriers (beforePass, afterPass) are left out. validated: the backings were checked against the graph's resource order
    // (ValidatePreparedBackingOrder) when they were made.
    // backingSet: the backings' set (AdmissionBackingSet::Id), whose ledger entries are found once; 0 for none.
    // externalHome: external home states, an interop option. Every host-owned texture (PreparedBackingState::hostOwned) the
    // execution touches leaves it in its home state (the backing's seeded region state, or the state its first access in the graph
    // expects) on the graphics queue (slot 0), so the host finds it where
    // it left it. A last use on another queue releases it to the graphics queue (homeAcquires). Other resources are left as the
    // execution leaves them, for the ledger to hand off.
    PreparedExecutionBarrierPlan Prepare(
        const Schedule& graph, std::span<const PreparedBackingState> initial,
        std::span<const rhi::ResourceHandle> invalidated = {}, bool entryOnly = false,
        bool validated = false, uint64_t backingSet = 0, bool externalHome = false) const;
    void CommitBatch(const PreparedBatchBarriers&);
    // The same for an execution's submitted batches, in order, each commit's entry found by its compiler resource index
    // (PreparedBatchBarriers::committedIndices) in the backings' set (backingSet as for Prepare). unmoved: compiler resource indices
    // (ascending) no running pass touched (UntouchedResources). Their commits are left out, so the ledger keeps the state the
    // execution found them in.
    void CommitBatches(std::span<const PreparedBatchBarriers> batches, std::span<const PreparedBackingState> backings,
        uint64_t backingSet, std::span<const uint32_t> unmoved = {});
    // A batch's seeds: the grids of backings the ledger does not hold yet, and of those whose state the batch makes again.
    void SeedBatch(const PreparedBatchBarriers& batch);
    // Whether the ledger holds a state for the backing.
    bool Holds(rhi::ResourceHandle resource) const;
    // Seeds a backing the ledger does not hold yet: every cell in `state`, last committed on `queue`. For a texture's first
    // contents, written outside the graphs before its first execution.
    void SeedUniform(rhi::ResourceHandle resource, const CompileResourceShape& shape, const CompileResourceState& state, uint32_t queue);
    // The prediction's entries taken anew: what the ledger holds now of each resource the graph's entry steps name.
    void Capture(StatePrediction& prediction, const Schedule& graph, std::span<const PreparedBackingState> initial,
        uint64_t backingSet = 0) const;
    // Whether the ledger still holds what the prediction's entries took (its plan stands).
    bool Predicts(const StatePrediction& prediction, std::span<const PreparedBackingState> initial, uint64_t backingSet = 0) const;
    void Invalidate(std::span<const rhi::ResourceHandle> resources);
    size_t BackingCount() const noexcept { return m_states.Size(); }
private:
    struct StateGrid {
        CompileResourceShape shape{};
        std::vector<CompileResourceState> cells;
        std::vector<uint32_t> queues;
        uint64_t fingerprint = 0;  // of cells and queues, kept by every change (StatePrediction)
        uint64_t changedIn = 0;    // m_changes at its last change (made, or a cell written)
    };
    // Commits one region's state: writes the cells it changes and updates the fingerprint for them. True when it changed one.
    static bool CommitCells(StateGrid& grid, const PreparedStateRegion& update, uint32_t queue);
    detail::BackingTable<StateGrid> m_states;
    uint64_t m_changes = 1;  // changes to the grids so far: a commit that wrote cells, a grid made or gone
};

// Ordered, submission-backed cross-frame hazard state. Unlike symbolic graph
// dependencies, these points refer only to work that was actually submitted.
// The ledger is keyed by concrete backing identity and subresource cell so
// unrelated resources/ranges and queues remain independent.
//
// A cell keeps every earlier access that wrote it and every one that read it: each producer once (a stream's batch, at its
// latest signal), and fixed points as the latest per timeline. Nothing is taken out again, so once every producer that touches
// a cell has touched it, the cell no longer changes: a frame that repeats changes nothing, and its commits only move their
// producers' signals. A read waits for the cell's writers, a write for its writers and readers. Compared with keeping only the
// latest writer and the reads since, the extra waits are on accesses the latest writer already waited for (an older write or
// read: write-after-write, write-after-read), or on the same timeline (dropped), so they are redundant, never wrong, and order
// nothing more on the GPU. A producer's latest signal is always one where it touched the cell: a stream touches the same cells
// every time.
class BackingAccessAdmissionLedger {
public:
    // A stream: the executions of one executable with one backing set (the same accesses to the same backings each time). Each of
    // its batches is a producer: the cells it touches name the producer instead of a point, and its commit sets the producer's
    // latest signal once. A stream commit that repeats the stream's last walk (same graph, backings and batches, no grid gone since)
    // only moves the signals. A closed stream's producers keep their last signals (fixed points from then on). Returns the
    // stream's id (never 0).
    uint64_t OpenStream(uint32_t batches);
    void CloseStream(uint64_t stream);
    // backingSet: as for BackingStateAdmissionLedger::Prepare. capture: filled with what the waits came from, for PredictedWaits.
    void AppendIncomingWaits(const Schedule& graph,
        std::span<const PreparedBackingState> resources,
        std::span<const ExecutionTimelinePoint> queues,
        std::vector<std::vector<ExecutionTimelinePoint>>& incoming, uint64_t backingSet = 0,
        AccessPrediction* capture = nullptr) const;
    // AppendIncomingWaits from a capture, when none of the grids it depends on changed since it was taken: the same waits, at the
    // points their producers have reached since. False (incoming untouched) when one did. A capture that holds is marked as
    // checked at the ledger's current change count, so the next check while nothing changes is one comparison.
    bool PredictedWaits(AccessPrediction& prediction, const Schedule& graph,
        std::span<const PreparedBackingState> resources,
        std::span<const ExecutionTimelinePoint> queues,
        std::vector<std::vector<ExecutionTimelinePoint>>& incoming, uint64_t backingSet) const;
    // Per batch, (compiler resource index, wait).
    using ResourceWaits = std::vector<std::vector<std::pair<uint32_t, ExecutionTimelinePoint>>>;
    // For each boundary access (incomingBoundaryAccessesByBatch) to a selected resource (selected: one flag per compiler
    // resource), every earlier access the ledger holds - its writers and readers, on every queue, the consumer's own
    // included: what a write into the resource on any queue ahead of the execution must wait for (RenderGraph's queue-scoped
    // ticket uploads), where AppendIncomingWaits gives a read only the writers, and nothing on its own queue. out: one list per batch.
    void AppendBoundaryAccessWaits(const Schedule& graph,
        std::span<const PreparedBackingState> resources,
        std::span<const uint8_t> selected,
        ResourceWaits& out, uint64_t backingSet = 0) const;
    // stream: the execution's (OpenStream), or 0 for none: its cells then hold its signals as fixed points.
    void Commit(const Schedule& graph,
        std::span<const PreparedBackingState> resources,
        const GraphExecutionTimeline& execution, uint32_t batchCount = UINT32_MAX, uint64_t backingSet = 0, uint64_t stream = 0);
    bool RetireCompleted(rhi::ResourceHandle resource, const std::map<uint64_t, uint64_t>& completed);
    size_t BackingCount() const noexcept { return m_accesses.Size(); }
    // Changes to the grids so far (a grid made, grown or gone, closed producers fixed). Steady frames leave it alone.
    uint64_t Changes() const noexcept { return m_changes; }
private:
    struct Cell {
        std::vector<AccessRef> writers;
        std::vector<AccessRef> readers;
    };
    struct AccessGrid {
        CompileResourceShape shape{};
        std::vector<Cell> cells;
        // m_changes at the last change to any of its cells, and to any of their writers (AccessPrediction).
        uint64_t changedIn = 0;
        uint64_t writersChangedIn = 0;
    };
    struct Producer {
        ExecutionTimelinePoint signal{};  // the latest; its timeline is set by the first commit
        bool closed = false;
    };
    ExecutionTimelinePoint PointOf(const AccessRef& ref) const noexcept {
        return ref.producer ? m_producers[ref.producer].signal : ref.point;
    }
    // Closed producers' references become their points, and the producers are reused (after enough have closed).
    void FixClosedProducers();
    // In one cell's writers or readers: closed producers' references become their points, and fixed points keep the latest per
    // timeline. True when it changed them.
    bool FoldClosed(std::vector<AccessRef>& refs) const;
    detail::BackingTable<AccessGrid> m_accesses;
    std::vector<Producer> m_producers{1};  // [0] unused: AccessRef::producer 0 is a point
    std::vector<uint32_t> m_freeProducers;
    size_t m_closedProducers = 0;
    // An open stream, and what its last walk put in the grids: every cell its first `batchCount` batches touch on `graph` with
    // `backingSet` names its producers. That holds until a grid goes (the table's generation moves): nothing else takes an open
    // producer out of a cell.
    struct Stream {
        std::vector<uint32_t> producers;  // per batch
        const Schedule* graph = nullptr;
        uint64_t backingSet = 0;
        uint32_t batchCount = 0;
        uint64_t tableGeneration = 0;
    };
    std::unordered_map<uint64_t, Stream> m_streams;
    uint64_t m_nextStream = 1;
    uint64_t m_changes = 1;
    // AppendIncomingWaits lists each producer once per batch: the batch's mark per producer.
    mutable std::vector<uint64_t> m_producerMarks;
    mutable uint64_t m_mark = 0;
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
    // Appends each batch's waits for the other occupants of the bytes its first uses take over, and returns the
    // aliased backings that replace an occupant (they enter from Undefined).
    std::vector<rhi::ResourceHandle> Resolve(const Schedule& graph,
        std::span<const PreparedBackingState> resources,
        std::span<const ExecutionTimelinePoint> queues,
        std::vector<std::vector<ExecutionTimelinePoint>>& incoming) const;
    void Commit(const Schedule& graph,
        std::span<const PreparedBackingState> resources,
        const GraphExecutionTimeline& execution, uint32_t batchCount = UINT32_MAX);
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
