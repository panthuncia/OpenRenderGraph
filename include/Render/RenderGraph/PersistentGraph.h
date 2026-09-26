#pragma once

#include "Render/RenderGraph/ExperimentalGraphCompiler.h"
#include "Render/RenderGraph/ExperimentalExecutionState.h"
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>
#include <utility>
#include "Render/PublicationBindingBundle.h"

namespace org::persistent {

// Epochs: a host that submits the graph at several points of its own frame (a
// renderer embedded in another engine) tags passes with the epoch they belong to.
// The program compiles once over the whole frame, epochs in their declared order,
// so scheduling and alias placement see every lifetime of the frame; each epoch
// also gets an executable of its own passes, which is what runs at that epoch.
// Untagged passes (AllEpochs) belong to every epoch, which with no tags at all is
// the whole graph in every execution.
inline constexpr uint32_t AllEpochs = UINT32_MAX;

struct ResourceSlotId {
    uint32_t index = UINT32_MAX, generation = 0;
    bool operator==(const ResourceSlotId&) const = default;
};
struct PassId {
    uint32_t index = UINT32_MAX, generation = 0;
    bool operator==(const PassId&) const = default;
};
struct ResourceGroupId {
    uint32_t index = UINT32_MAX, generation = 0;
    bool operator==(const ResourceGroupId&) const = default;
};
class BindingToken {
    friend class GraphEditTransaction;
    friend class SelectedPublication;
    uint64_t domain = 0, layoutRevision = 0;
    uint64_t declarationId = 0;
    PassId pass;
    uint32_t ordinal = UINT32_MAX;
    BindingToken(uint64_t d, uint64_t revision, PassId p, uint32_t entry, uint64_t id)
        : domain(d), layoutRevision(revision), declarationId(id), pass(p), ordinal(entry) {}
public:
    BindingToken() = default;
};
class ViewToken {
    friend class GraphEditTransaction;
    friend class SelectedPublication;
    BindingToken binding;
    uint32_t ordinal = UINT32_MAX;
    uint64_t declarationId = 0;
    ViewToken(BindingToken token, uint32_t entry, uint64_t id) : binding(token), ordinal(entry), declarationId(id) {}
public:
    ViewToken() = default;
};
struct GroupAccess {
    PassId pass;
    experimental::CompileResourceState state;
    uint32_t phase = 0;
    std::optional<experimental::CompileRange> range;
};
struct ResourceGroup {
    experimental::CompileResourceShape memberShape;
    uint64_t membershipRevision = 1;
    std::vector<ResourceSlotId> members;
    // Reverse subscriptions are authored once, never reconstructed by frames.
    std::vector<GroupAccess> subscribers;
    uint32_t generation = 1;
    bool active = true;
    rhi::ResourceType memberType = rhi::ResourceType::Unknown;
};
// Pointer-free recording contract. Initial states, debug names and allocation
// optimization hints are not binding compatibility and remain publication data.
struct NativeBindingContract {
    rhi::ResourceType type = rhi::ResourceType::Unknown;
    rhi::HeapType heapType = rhi::HeapType::DeviceLocal;
    rhi::HeapFlags heapFlags = rhi::HeapFlags::None;
    rhi::ResourceFlags requiredFlags{};
    uint64_t minimumBufferBytes = 0, maximumBufferBytes = 0;
    rhi::Format format = rhi::Format::Unknown;
    uint32_t width = 0, height = 0, depthOrLayers = 0, mips = 0, samples = 0;
    std::vector<rhi::Format> castableFormats;
    static NativeBindingContract Capture(const rhi::ResourceDesc& description);
    bool Matches(const rhi::ResourceDesc& description, experimental::CompileResourceShape shape) const;
};
struct LogicalGraph {
    struct PassSlot {
        struct BindingEntry {
            struct ViewRequirement { BindlessViewRequest request; uint64_t declarationId = 0; };
            ResourceSlotId resource; uint64_t declarationId = 0;
            std::vector<ViewRequirement> requiredViews;
        };
        uint32_t generation = 1; bool active = true;
        uint64_t layoutRevision = 1;
        std::vector<BindingEntry> bindingSlots;
        std::shared_ptr<const void> recordingInterface;
        uint32_t epoch = AllEpochs;
    };
    // Execution order of the epochs within one host frame. Epochs not listed
    // follow the listed ones in ascending order.
    std::vector<uint32_t> epochOrder;
    uint64_t domain = 0;
    experimental::GraphCompileStructure declarations;
    std::vector<ResourceGroup> groups;
    std::vector<PassSlot> passSlots;
    std::vector<uint8_t> resourceActive;
    std::vector<std::optional<NativeBindingContract>> nativeContracts;
    std::vector<std::vector<std::pair<uint32_t,uint32_t>>> bindingSubscribers;
    // Group index per slot (NoGroup / MultipleGroups), so binding edits check a
    // member's group class without scanning every group's member list.
    static constexpr uint32_t NoGroup = UINT32_MAX, MultipleGroups = UINT32_MAX - 1;
    std::vector<uint32_t> groupBySlot;
};

// Numeric physical identity plus exact ownership. No native handles are used by
// the logical compiler. State and timeline admission remain execution-specific.
struct BindingVersion {
    static BindingVersion FromSnapshot(std::shared_ptr<const ResourceBindingSnapshot> snapshot,
        const experimental::PreparedBackingState& initial, uint64_t descriptorRevision = 0,
        uint64_t contentRevision = 0);
    uint64_t identity = 0, backingRevision = 0, descriptorRevision = 0;
    uint64_t contentRevision = 0, waitRevision = 0;
    experimental::CompileResourceShape shape;
    std::shared_ptr<const void> owner;
    std::shared_ptr<const experimental::PreparedBackingState> admission;
    bool active = true;
    // A reserved slot is active and compiled but currently has no backing.
    // Admission and recording skip it; binding one later is not structural.
    bool bound = true;
    uint32_t slotGeneration = 1;
    std::shared_ptr<const ResourceBindingSnapshot> recording;
    using DescriptorTable = std::map<uint64_t,std::vector<rhi::DescriptorSlot>>;
    std::shared_ptr<const DescriptorTable> preparedViews;
};
class BindingTable {
public:
    static constexpr uint32_t PageSize = 32;
    using Page = std::array<BindingVersion, PageSize>;
    const BindingVersion& At(ResourceSlotId slot) const;
    ResourceSlotId CurrentSlot(uint32_t index) const;
    // Null for retired slots; reserved (unbound) slots are returned.
    const BindingVersion* TryAt(uint32_t index) const noexcept;
    uint32_t Size() const noexcept { return m_size; }
private:
    friend class GraphEditTransaction;
    std::vector<std::shared_ptr<const Page>> m_pages;
    static constexpr uint32_t IdentityBuckets = 64;
    using IdentityBucket = std::map<uint64_t,std::vector<uint32_t>>;
    std::array<std::shared_ptr<const IdentityBucket>,IdentityBuckets> m_identities;
    uint32_t m_size = 0;
};

struct ExecutableGeneration {
    uint64_t structuralRevision = 0;
    std::shared_ptr<const experimental::CompiledGraph> graph;
    // Compiler enumeration is private to this generation. Admission resolves
    // stable slots through this immutable map, never assumes table density.
    std::vector<ResourceSlotId> resourceSlots;
    // Retained once per structural generation for the production submission
    // adapter. No frame creates or reconstructs this immutable compiler input.
    std::shared_ptr<const experimental::GraphCompileInput> compileInput;
    // Production recording/submission mapping, prepared once with the executable.
    // Retired logical pass slots remain empty; active slots keep stable indices.
    std::shared_ptr<const experimental::GraphExecutionLayout> executionLayout;
    std::vector<uint32_t> resourceIndexBySlot;
    std::vector<std::vector<uint32_t>> incomingBatchesByResource;
    std::vector<uint32_t> incomingStateBatchByResource;
    // Per epoch (in execution order), the executable of that epoch's passes plus
    // the untagged ones, compiled over the same bindings and alias placements as
    // this whole-frame executable. Empty when no pass is tagged.
    std::vector<std::pair<uint32_t, std::shared_ptr<const ExecutableGeneration>>> epochs;
};
class SelectedPublication {
public:
    // The publication an epoch executes: these bindings, the epoch's executable.
    // Null when the program has no epochs, or none of its passes is in this one
    // (the caller then executes this publication, or nothing). Cached.
    std::shared_ptr<const SelectedPublication> ForEpoch(uint32_t epoch) const;
    bool HasEpochs() const noexcept { return executable && !executable->epochs.empty(); }
    const BindingVersion& Resolve(BindingToken token) const;
    rhi::Resource ResolveNative(BindingToken token) const;
    rhi::DescriptorSlot ResolveView(BindingToken token, BindlessViewRequest view) const;
    rhi::DescriptorSlot ResolveView(ViewToken token) const;
    const uint64_t revision;
    const std::shared_ptr<const ExecutableGeneration> executable;
    const BindingTable bindings;
    const std::shared_ptr<const LogicalGraph> logical;
    const std::weak_ptr<const SelectedPublication> source;
private:
    friend class GraphProgram;
    friend class GraphEditTransaction;
    SelectedPublication(uint64_t value, std::shared_ptr<const ExecutableGeneration> generation,
        BindingTable table, std::shared_ptr<const LogicalGraph> program,
        std::weak_ptr<const SelectedPublication> base = {})
        : revision(value), executable(std::move(generation)), bindings(std::move(table)), logical(std::move(program)), source(std::move(base)) {}
    mutable std::mutex m_epochMutex;
    mutable std::vector<std::pair<uint32_t, std::shared_ptr<const SelectedPublication>>> m_epochViews;
};

// Build may run on a worker. Installation is an O(1) revision check. A rejected
// or failed build never changes the selected publication.
class GraphEditTransaction {
public:
    explicit GraphEditTransaction(std::shared_ptr<const SelectedPublication> base);
    GraphEditTransaction(const GraphEditTransaction&) = delete;
    GraphEditTransaction& operator=(const GraphEditTransaction&) = delete;
    GraphEditTransaction(GraphEditTransaction&&) noexcept = default;
    GraphEditTransaction& operator=(GraphEditTransaction&&) noexcept = default;
    ResourceSlotId AddResource(experimental::CompileResourceShape shape, BindingVersion binding);
    // Reserved capacity: the slot participates in compilation with a placeholder
    // backing so a later BindReserved/Unbind is a binding-only edit. The optional
    // contract is checked when a real binding arrives.
    ResourceSlotId ReserveResource(experimental::CompileResourceShape shape,
        std::optional<NativeBindingContract> contract = {});
    void BindReserved(ResourceSlotId slot, BindingVersion binding);
    // Slot indices this transaction currently binds to one physical identity.
    // Every such slot must carry the same admission state when the edit builds.
    std::vector<uint32_t> SlotsSharingIdentity(uint64_t identity) const;
    void Unbind(ResourceSlotId slot);
    void RemoveResource(ResourceSlotId slot);
    void SetNativeBindingContract(ResourceSlotId slot, NativeBindingContract contract);
    // Passes are ordered by authored order (hazard direction, tie-breaks). By
    // default a pass is placed at index * kAuthoredOrderStride, leaving room to
    // insert passes between existing ones with an explicit order.
    static constexpr uint32_t kAuthoredOrderStride = 1024;
    PassId AddPass(experimental::CompilePass pass, std::optional<uint32_t> authoredOrder = std::nullopt);
    uint32_t AuthoredOrder(PassId pass) const;
    void ReplacePass(PassId pass, experimental::CompilePass declaration);
    void RemovePass(PassId pass);
    void SetPassRecordingInterface(PassId pass, std::shared_ptr<const void> recordingInterface);
    // Structural: the epoch the pass executes in (AllEpochs: every one).
    void SetPassEpoch(PassId pass, uint32_t epoch);
    // Structural: the order the host executes its epochs in within a frame.
    void SetEpochOrder(std::vector<uint32_t> order);
    BindingToken Declare(PassId pass, ResourceSlotId resource,
        experimental::CompileRange range, experimental::CompileResourceState state,
        uint64_t byteOffset = 0, uint64_t byteSize = UINT64_MAX, uint32_t aspects = 0);
    BindingToken DeclareDependency(PassId pass, ResourceSlotId resource, bool write);
    void DeclarePostcondition(BindingToken binding, experimental::CompileRange range,
        experimental::CompileResourceState state);
    ViewToken DeclareView(BindingToken binding, BindlessViewRequest view);
    void AddOrdering(PassId before, PassId after);
    void AddPlacementOrdering(PassId before, PassId after);
    // Drops every placement ordering (structural). Alias planning re-derives
    // them from the schedule of the new executable.
    void ClearPlacementOrderings();
    void ReplaceResourceContract(ResourceSlotId slot, experimental::CompileResourceShape shape, BindingVersion binding);
    void SetQueues(std::vector<experimental::CompileQueue> queues);
    void ReplaceBinding(ResourceSlotId slot, BindingVersion binding);
    ResourceSlotId AddSnapshot(std::shared_ptr<const ResourceBindingSnapshot> snapshot,
        const experimental::PreparedBackingState& initial, uint64_t descriptorRevision = 0, uint64_t contentRevision = 0);
    void ReplaceSnapshot(ResourceSlotId slot, std::shared_ptr<const ResourceBindingSnapshot> snapshot,
        const experimental::PreparedBackingState& initial, uint64_t descriptorRevision = 0, uint64_t contentRevision = 0);
    ResourceGroupId AddGroup(experimental::CompileResourceShape memberShape, std::vector<ResourceSlotId> members);
    void ReplaceGroupMembers(ResourceGroupId group, std::vector<ResourceSlotId> members);
    // Appends `count` reserved member slots (structural). Binding them later is
    // a binding-only edit; this is how streaming membership stays non-structural.
    std::vector<ResourceSlotId> ReserveGroupMembers(ResourceGroupId group, uint32_t count);
    void SetGroupResourceClass(ResourceGroupId group, rhi::ResourceType type);
    void DeclareGroupAccess(PassId pass, ResourceGroupId group,
        experimental::CompileResourceState state, uint32_t phase,
        std::optional<experimental::CompileRange> range = {});
    void RemoveGroupAccess(PassId pass, ResourceGroupId group);
    void RemoveGroup(ResourceGroupId group);
    static void UnindexGroupMember(LogicalGraph& logical, ResourceSlotId slot, uint32_t groupIndex);
    // validateAliasOrder=false builds a schedule-only publication for alias
    // planning: existing placements may be unordered against new users. Such a
    // publication must not be installed; plan, re-edit and build again.
    std::shared_ptr<const SelectedPublication> Build(experimental::CompileWorkspace& workspace,
        const std::atomic_bool& cancelled, bool validateAliasOrder = true);
    uint64_t BaseRevision() const noexcept;
    // Producer-side preparation failure makes the entire edit unselectable.
    void Abort() noexcept { m_failed = true; }
private:
    experimental::GraphCompileStructure& EditStructure();
    LogicalGraph& EditLogical(bool structural = true);
    void ValidatePass(PassId pass) const;
    void ClearPassBindings(PassId pass);
    std::shared_ptr<const SelectedPublication> m_base;
    std::optional<LogicalGraph> m_logical;
    BindingTable m_bindings;
    std::map<uint32_t, std::shared_ptr<BindingTable::Page>> m_changedPages;
    std::map<uint32_t, std::shared_ptr<BindingTable::IdentityBucket>> m_changedIdentities;
    bool m_failed = false;
    bool m_structuralChanged = false;
};

class GraphProgram {
public:
    explicit GraphProgram(std::function<void(std::shared_ptr<const void>)> retireOwnership = {});
    GraphEditTransaction BeginEdit() const;
    std::shared_ptr<const SelectedPublication> Select() const;
    bool Install(const GraphEditTransaction& transaction,
        std::shared_ptr<const SelectedPublication> ready);
    bool Install(std::shared_ptr<const SelectedPublication> ready);
    // Installs a publication built from an older base after the caller has
    // replayed every edit installed since onto it. The publication is
    // renumbered as the successor of the current selection.
    bool InstallRebased(std::shared_ptr<const SelectedPublication> ready);
private:
    mutable std::mutex m_mutex;
    std::shared_ptr<const SelectedPublication> m_selected;
    std::function<void(std::shared_ptr<const void>)> m_retireOwnership;
};

// Adapter into the existing production state/hazard ledgers. Recording payloads
// remain caller-owned; this view roots the exact selected publication.
struct FrameAdmission {
    uint64_t sequence = 0, domain = 0;
    std::shared_ptr<const SelectedPublication> publication;
    std::vector<experimental::PreparedBackingState> backings;
    experimental::PreparedExecutionBarrierPlan barriers;
    std::vector<std::vector<experimental::ExecutionTimelinePoint>> incomingWaits;
    std::vector<uint64_t> queueTimelines;
    struct IncomingEffect {
        rhi::ResourceHandle resource;
        uint64_t revision;
        experimental::ExecutionTimelinePoint completion;
        uint32_t firstBatch;
    };
    std::vector<IncomingEffect> incomingEffects;
    // Compiler resource indices whose backing was supplied by the frame
    // (swapchain images and similar late-bound imports), not the publication.
    std::vector<uint32_t> rebound;
    // A closed execution (SynchronousAdmission::SetClosedExecutions): its recording must begin with a full
    // memory barrier on each queue, which is what makes the previous execution's writes visible to it.
    bool closed = false;
    // The barrier plan came from the admission's cache rather than the ledgers.
    bool cachedPlan = false;
};
// Frame-supplied backing for a slot the publication deliberately leaves
// unbound. The executable and its state contract stay selected; only the
// physical identity is late-bound, so this never supersedes a pending edit.
struct FrameRebinding {
    ResourceSlotId slot;
    experimental::PreparedBackingState backing;
    std::shared_ptr<const ResourceBindingSnapshot> recording;
};

// Producer completion requirements are invocation data, not graph structure or
// binding compatibility. Stable consumers resolve through the selected layout.
struct FrameProducerWait {
    PassId consumer;
    experimental::ExecutionTimelinePoint completion;
};
struct FrameIncomingState {
    // New external submission since the last committed graph state. Report
    // complete backend-legal incoming states after any required producer release.
    // Revisions increase per physical backing. Commit consumes a revision at its
    // first state-use batch; repeated consumed reports preserve newer graph states.
    ResourceSlotId resource;
    rhi::ResourceHandle backing;
    std::shared_ptr<const std::vector<experimental::PreparedStateRegion>> regions;
    experimental::ExecutionTimelinePoint producerCompletion;
    uint64_t revision = 1;
};
class SynchronousAdmission {
public:
    // Capture once when a producer replaces/removes a binding. These weak
    // tickets cannot keep retired publications or native allocations alive.
    class RetirementTicket {
        friend class SynchronousAdmission;
        rhi::ResourceHandle resource{};
        std::weak_ptr<const void> owner, allocation;
        std::weak_ptr<const AliasHeapGeneration> heap;
        const AliasHeapGeneration* heapIdentity = nullptr;
    };
    static RetirementTicket CaptureRetirement(const BindingVersion& binding);
    // Owner-thread maintenance, outside preparation. False means a CPU owner
    // or submitted GPU consumer still exists; retry after retirement advances.
    bool RetireBackingMetadata(const RetirementTicket& ticket);
    bool RetireAliasMetadata(const RetirementTicket& ticket);
    size_t StateBackingCount() const noexcept { return m_states.BackingCount(); }
    size_t HazardBackingCount() const noexcept { return m_accesses.BackingCount(); }
    size_t AliasHeapCount() const noexcept { return m_aliases.HeapCount(); }
    size_t IncomingRevisionCount() const noexcept { return m_incomingRevisions.size(); }
    explicit SynchronousAdmission(size_t frameCapacity = 3,
        std::function<void(std::shared_ptr<const void>)> retireOwnership = {});
    // Closed executions: every resource an execution touches leaves it in its home state
    // (BackingStateAdmissionLedger::Prepare closeToHome) and every execution starts with a full memory
    // barrier (FrameAdmission::closed). An execution's barrier plan then depends only on its executable and
    // backings, not on what ran before it; the admission caches it per executable, and allows several
    // admissions to be prepared before the earlier ones commit, and to commit or be abandoned in any order
    // (a commit's signals must still rise on each timeline: they commit in submission order).
    // Requires every resource of an execution to be used from one queue. Set before the first Prepare.
    void SetClosedExecutions(bool closed);
    bool ClosedExecutions() const noexcept { return m_closed; }
    size_t PendingAdmissions() const noexcept { return m_pending.size(); }
    uint64_t PlanCacheHits() const noexcept { return m_planCacheHits; }
    uint64_t PlanCacheMisses() const noexcept { return m_planCacheMisses; }
    FrameAdmission Prepare(std::shared_ptr<const SelectedPublication> publication,
        std::span<const experimental::ExecutionTimelinePoint> queues,
        std::span<const FrameProducerWait> producerWaits = {},
        std::span<const FrameIncomingState> incomingStates = {},
        std::span<const FrameRebinding> rebindings = {});
    void Commit(const FrameAdmission& frame, const experimental::GraphExecutionTimeline& receipt,
        uint32_t signaledBatches = UINT32_MAX);
    void Abandon(const FrameAdmission& frame);
    // Work submitted outside a graph receipt on a graph queue (presentation
    // tails). Completion observations are clamped against this.
    void ExtendSubmitted(experimental::ExecutionTimelinePoint point);
    uint64_t Submitted(uint64_t timeline) const noexcept;
    size_t RetireCompleted(std::span<const experimental::ExecutionTimelinePoint> completed);
    size_t RetainedFrames() const noexcept { return m_retained.size(); }
private:
    experimental::BackingStateAdmissionLedger m_states;
    experimental::BackingAccessAdmissionLedger m_accesses;
    experimental::AliasAccessAdmissionLedger m_aliases;
    uint64_t m_nextSequence = 1;
    // Admissions prepared and not yet committed or abandoned, in sequence order (at most one unless closed).
    std::vector<uint64_t> m_pending;
    bool m_closed = false;
    struct CachedPlan {
        std::weak_ptr<const ExecutableGeneration> executable;
        std::vector<rhi::ResourceHandle> resources;
        std::vector<const void*> regions;
        experimental::PreparedExecutionBarrierPlan barriers;
    };
    std::vector<CachedPlan> m_planCache;
    uint64_t m_planCacheHits = 0, m_planCacheMisses = 0;
    uint64_t m_domain = 0;
    std::map<uint64_t,uint64_t> m_submitted, m_completed;
    struct IncomingRevision { uint64_t revision = 0; experimental::ExecutionTimelinePoint completion; };
    std::map<uint64_t,IncomingRevision> m_incomingRevisions;
    struct RetainedFrame {
        std::shared_ptr<const SelectedPublication> publication;
        std::vector<experimental::ExecutionTimelinePoint> completions;
    };
    std::vector<RetainedFrame> m_retained;
    size_t m_capacity;
    std::function<void(std::shared_ptr<const void>)> m_retireOwnership;
};

enum class PublicationState { Pending, Ready, Selected, Failed, Superseded };
// Async publication building is optional and entirely outside synchronous
// frame instantiation. The owner calls Pump; workers never select publications.
class PublicationCoordinator {
public:
    using StructuralEdit = std::function<void(GraphEditTransaction&)>;
    PublicationCoordinator(GraphProgram& program, std::shared_ptr<runtime::ITaskService> tasks);
    ~PublicationCoordinator();
    uint64_t Submit(GraphEditTransaction transaction);
    // Replayable structural edit. Inline binding edits installed while this
    // builds do not lose it: Pump re-applies the closure on the new base.
    uint64_t Submit(StructuralEdit edit);
    PublicationState State() const;
    PublicationState Pump();
    std::string Failure() const;
    uint64_t Rebases() const noexcept { return m_rebases; }
private:
    struct Job;
    void Start(const std::shared_ptr<Job>& job);
    GraphProgram& m_program;
    std::shared_ptr<runtime::ITaskService> m_tasks;
    std::shared_ptr<runtime::ITaskScope> m_scope;
    std::shared_ptr<Job> m_latest;
    uint64_t m_nextSequence = 1;
    uint64_t m_rebases = 0;
};

} // namespace org::persistent
