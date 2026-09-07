#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "Render/Runtime/ITaskService.h"
#include "Render/RenderGraph/CompilerAlgorithms.h"

namespace org::experimental {

// This extraction produces dependencies and a symbolic queue schedule. It is deliberately
// not executable: realization, alias planning and ordered admission must be
// extracted before a candidate can replace a live execution graph.
enum class CompiledGraphStage : uint8_t { Dependencies, SymbolicSchedule };

// Numeric, owned metadata only. Unlike the legacy ResourceState comparison,
// synchronization scopes participate in equality and structural cache keys.
struct CompileResourceShape {
    // {0,0,false} is a dependency-only identity, with no stateful resource.
    // A state declaration referencing it is unsupported, never treated as 1x1.
    uint32_t mips = 1, slices = 1;
    bool hasLayout = false;
    bool operator==(const CompileResourceShape&) const = default;
};
struct CompileRange {
    uint32_t mip = 0, mips = 1, slice = 0, slices = 1;
    bool operator==(const CompileRange&) const = default;
};
struct CompileResourceState {
    uint64_t access = 0, layout = 0, sync = 0;
    bool write = false;
    bool operator==(const CompileResourceState&) const = default;
};
struct CompileStateUse {
    uint32_t resource = 0;
    CompileRange range;
    CompileResourceState state;
    bool operator==(const CompileStateUse&) const = default;
};

struct CompileAccess {
    uint32_t resourceIndex = 0;
    bool write = false;
    bool operator==(const CompileAccess&) const = default;
};
struct CompilePass {
    uint32_t originalOrder = 0;
    uint32_t backend = 0;
    std::vector<CompileAccess> accesses;
    std::vector<uint32_t> compatibleQueueSlots{0};
    uint32_t preferredQueueSlot = 0;
    std::vector<CompileStateUse> entryStates;
    // Declared callback postconditions, not compiler-emitted barriers.
    std::vector<CompileStateUse> exitStates;
    // Stable index into the prepared frame-pass array. Execution must not infer
    // this association from compiler-local order or a mutable pass container.
    uint32_t preparedPassIndex = UINT32_MAX;
    // Immediate/single-consumption work and pass-local external waits retain a
    // batch boundary. This is authored preparation metadata, not worker access
    // to the mutable pass object.
    bool forceBatchIsolation = false;
    bool operator==(const CompilePass&) const = default;
};
struct CompileQueue {
    uint32_t backendInstance = 0;
    bool active = true;
    bool operator==(const CompileQueue&) const = default;
};
struct GraphCompileStructure {
    uint64_t generation = 0;
    uint64_t registryGeneration = 0;
    std::vector<uint64_t> resourceIDs;
    // Empty for persistent concrete identities. Named frame-transient slots
    // use a semantic key so a new backing/global ID does not force recompilation.
    // The string is retained for full collision-safe equality.
    std::vector<std::string> resourceKeys;
    std::vector<CompilePass> passes;
    std::vector<std::pair<uint32_t, uint32_t>> explicitEdges;
    // Additional constraints from owned alias-placement preparation. These do
    // not substitute for independently deriving the resource dependency DAG.
    std::vector<std::pair<uint32_t, uint32_t>> placementEdges;
    std::vector<CompileQueue> queues{{}};
    // Empty means dependency/schedule-only capture. Otherwise one per ID.
    std::vector<CompileResourceShape> resourceShapes;
    bool operator==(const GraphCompileStructure&) const = default;
};
using DependencyEdges = std::vector<std::pair<uint32_t, uint32_t>>;

class IFramePayloadLifecycle {
public:
    virtual ~IFramePayloadLifecycle() = default;
    virtual void Abandon(uint8_t reason) const noexcept = 0;
};

struct GraphCompileInput {
    GraphCompileStructure structure;
    // Realization identity, ordered with structure.resourceIDs. It is excluded
    // from symbolic compilation equality: same layout/new backing reuses a plan.
    std::vector<uint64_t> backingGenerations;
    // Retains the exact publication/backings that preparation resolved. No
    // worker dereferences an opaque lease or calls a resource/pass interface.
    std::vector<std::shared_ptr<const void>> leases;
    // Owned frame recording payload paired with this request. The pure
    // compiler never dereferences it; realization/admission interprets it.
    std::shared_ptr<const void> executionPayload;
    std::shared_ptr<const IFramePayloadLifecycle> executionLifecycle;
    std::shared_ptr<const DependencyEdges> expectedEdges;
    std::shared_ptr<const DependencyEdges> expectedSchedulingEdges;
};

// Ordered passes recorded on one queue. Batch indices and waits are relative:
// no queue timeline is touched here. The initial policy may still isolate
// passes, but the representation is shared-scheduler ready and does not encode
// that temporary restriction.
struct SymbolicBatch {
    std::vector<uint32_t> passes;
    uint32_t queue = 0;
    SymbolicBatch() = default;
    SymbolicBatch(uint32_t pass, uint32_t queueSlot) : passes{pass}, queue(queueSlot) {}
    bool operator==(const SymbolicBatch&) const = default;
};
struct RelativeQueueWait {
    uint32_t consumerBatch = 0, producerBatch = 0;
    bool operator==(const RelativeQueueWait&) const = default;
};

struct SymbolicStateStep {
    uint32_t resource = 0, batch = 0;
    // UINT32_MAX means resolve the incoming state/ownership at admission.
    uint32_t previousBatch = UINT32_MAX;
    CompileRange range;
    CompileResourceState before, after;
    // Consuming compiler pass. Required to place barriers between passes that
    // share a command-list batch rather than hoisting them to batch entry.
    uint32_t pass = UINT32_MAX;
    // Producer compiler pass for cross-queue release placement. UINT32_MAX
    // denotes the execution-boundary state supplied by admission.
    uint32_t previousPass = UINT32_MAX;
    bool operator==(const SymbolicStateStep&) const = default;
};
struct SymbolicFinalState {
    uint32_t resource = 0, batch = 0;
    CompileRange range;
    CompileResourceState state;
    bool operator==(const SymbolicFinalState&) const = default;
};
struct SymbolicStatePlan {
    bool complete = false;
    std::string fallbackReason;
    // Conservative state dependencies, not backend barriers. Read/read steps
    // may later be elided only after queue/API-specific validation.
    std::vector<SymbolicStateStep> steps;
    std::vector<SymbolicFinalState> finalStates;
};

struct CompiledGraph {
    CompiledGraphStage stage = CompiledGraphStage::Dependencies;
    // Structural reuse must not pin the first content publication indefinitely.
    // Publication leases belong to bundles/execution instances, not this plan.
    std::shared_ptr<const GraphCompileStructure> structure;
    DependencyEdges edges;
    DependencyEdges schedulingEdges;
    std::vector<uint32_t> topologicalOrder;
    std::vector<uint32_t> criticality;
    std::vector<SymbolicBatch> batches;
    std::vector<RelativeQueueWait> relativeWaits;
    SymbolicStatePlan states;
    bool scheduleValidated = false;
    std::string scheduleValidationError;
    std::string stateValidationError;
};

// Normalizes enumeration order only; concrete identities and pass order remain
// significant. Invalid or duplicate global IDs are rejected, never conflated.
void NormalizeCompileInput(GraphCompileInput&);
// Independent schedule legality check, including same-queue order and actual
// wait reachability. Does not require the live scheduler's exact batch choices.
std::string ValidateSymbolicSchedule(const GraphCompileInput&, const CompiledGraph&);
std::string ValidateSymbolicStates(const GraphCompileInput&, const CompiledGraph&);

class CompileWorkspace {
public:
    std::shared_ptr<const CompiledGraph> Compile(
        std::shared_ptr<const GraphCompileInput> input, const std::atomic_bool& cancelled);
private:
    std::vector<compiler::DependencySequence<uint32_t>> m_resources;
    std::vector<std::vector<uint32_t>> m_successors;
    std::vector<uint32_t> m_indegrees;
    std::vector<uint32_t> m_ready;
};

// Single owned-input compiler entry point used by both workers and synchronous
// validation tools. Keep alternate compiler routes out of call sites.
std::shared_ptr<const CompiledGraph> CompileGraph(
    std::shared_ptr<const GraphCompileInput> input,
    CompileWorkspace& workspace,
    const std::atomic_bool& cancelled);

struct CompiledGraphBundle {
    uint64_t sequence = 0;
    std::shared_ptr<const CompiledGraph> graph;
    // May be a newer coalesced publication with the same complete structure.
    std::shared_ptr<const GraphCompileInput> input;
};

// A compile result is paired permanently with the exact owned frame request
// that produced it. CompiledGraph may be shared by structurally identical
// requests, but input/executionPayload is never replaced or skipped.
struct ExecutableCompiledFrame {
    uint64_t sequence = 0;
    std::shared_ptr<const CompiledGraphBundle> bundle;
};

// Admission-side compatibility check. This is intentionally independent of
// coordinator publication order: a completed candidate is selectable only for
// the exact freshly prepared structure, with an unambiguous prepared-pass map.
bool IsExecutionCompatible(const CompiledGraphBundle&, const GraphCompileInput&) noexcept;

struct ExecutionPassPlacement {
    uint32_t preparedPass = 0;
    uint32_t batch = 0;
    uint32_t queue = 0;
    bool operator==(const ExecutionPassPlacement&) const = default;
};
struct GraphExecutionLayout {
    std::shared_ptr<const CompiledGraphBundle> bundle;
    // Indexed by prepared pass, so recording never consults compiler scratch.
    std::vector<ExecutionPassPlacement> placements;
};
std::shared_ptr<const GraphExecutionLayout> BuildExecutionLayout(
    std::shared_ptr<const CompiledGraphBundle>, const GraphCompileInput& prepared);


struct ExecutionTimelinePoint {
    uint64_t timeline = 0, value = 0;
    bool operator==(const ExecutionTimelinePoint&) const = default;
};
struct ExecutionBatchTimeline {
    ExecutionTimelinePoint signal;
    std::vector<ExecutionTimelinePoint> waits;
};
class IPreparedExecutionBatch;
// Timeline portion only, not an executable command packet. The admission owner
// must supply exclusive queue timelines and cross-frame/resource waits after
// backing/ownership resolution. Workers never construct absolute fence values.
struct GraphExecutionTimeline {
    uint64_t submission = 0;
    std::shared_ptr<const CompiledGraphBundle> bundle;
    std::vector<ExecutionBatchTimeline> batches;
    std::vector<std::shared_ptr<const void>> executionLeases;
    // Kept separately from opaque leases so lifecycle completion can be
    // delivered when all queue signals for this execution are observed.
    std::vector<std::shared_ptr<const IPreparedExecutionBatch>> preparedBatches;
};
// Prepared packets contain no live pass callbacks. Implementations own closed
// command lists, allocators, backing/descriptor versions and timeline leases.
enum class SubmissionState { NotSubmitted, SubmissionUncertain, SubmittedWithoutSignal, Signaled };
enum class SubmissionFailureStage { None, Validation, Wait, Submit, Signal, Replay };
struct SubmissionReceipt {
    SubmissionState state = SubmissionState::NotSubmitted;
    SubmissionFailureStage failureStage = SubmissionFailureStage::None;
    uint32_t backendResult = 0;
    explicit operator bool() const noexcept { return state == SubmissionState::Signaled; }
};
struct FailedExecutionBatch {
    uint64_t submission = 0;
    uint32_t batch = 0;
    SubmissionReceipt receipt;
};
class IPreparedExecutionBatch {
public:
    virtual ~IPreparedExecutionBatch() = default;
    virtual uint32_t QueueSlot() const noexcept = 0;
    virtual SubmissionReceipt Submit(const ExecutionBatchTimeline&) const noexcept = 0;
    virtual void Complete(uint64_t submission) const noexcept = 0;
    virtual void Abandon() const noexcept = 0;
};
class ExecutionTimelineAdmission {
public:
    explicit ExecutionTimelineAdmission(std::vector<ExecutionTimelinePoint> queues,
        size_t maximumInFlight = 3);
    ExecutionTimelineAdmission(const ExecutionTimelineAdmission&) = delete;
    ExecutionTimelineAdmission& operator=(const ExecutionTimelineAdmission&) = delete;
    std::shared_ptr<const GraphExecutionTimeline> Prepare(
        std::shared_ptr<const CompiledGraphBundle>,
        const std::vector<std::vector<ExecutionTimelinePoint>>& incomingWaits,
        std::vector<std::shared_ptr<const void>> executionLeases = {},
        std::vector<std::shared_ptr<const IPreparedExecutionBatch>> preparedBatches = {});
    std::shared_ptr<const GraphExecutionTimeline> SubmitPrepared(
        std::shared_ptr<const CompiledGraphBundle>,
        const std::vector<std::vector<ExecutionTimelinePoint>>& incomingWaits,
        const std::vector<std::shared_ptr<const IPreparedExecutionBatch>>& packets);
    // Call only after the backend successfully submits this batch's signal.
    void CommitBatch(uint64_t submission, uint32_t batch);
    // A partial failure preserves committed values and permanently closes this
    // admission owner until device/error recovery constructs a fresh owner.
    void Fail(uint64_t submission, SubmissionReceipt receipt = {});
    // Called by the ordered owner with observed GPU completion values, in the
    // same queue order as construction. Releases only fully completed bundles.
    size_t RetireCompleted(std::span<const ExecutionTimelinePoint> completed);
    size_t InFlight() const { return m_retained.size() + (m_pending ? 1 : 0); }
    std::span<const ExecutionTimelinePoint> Submitted() const { return m_submitted; }
    bool Failed() const { return m_failed; }
    const std::optional<FailedExecutionBatch>& Failure() const { return m_failure; }
    std::shared_ptr<const GraphExecutionTimeline> PendingExecution() const { return m_pending; }
private:
    std::vector<ExecutionTimelinePoint> m_reserved, m_submitted;
    std::shared_ptr<const GraphExecutionTimeline> m_pending;
    std::vector<std::shared_ptr<const GraphExecutionTimeline>> m_retained;
    std::vector<ExecutionTimelinePoint> m_completed;
    size_t m_maximumInFlight;
    uint64_t m_sequence = 0;
    uint32_t m_nextBatch = 0;
    bool m_failed = false;
    std::optional<FailedExecutionBatch> m_failure;
};

struct CompileCoordinatorStatistics {
    uint64_t requested = 0, coalesced = 0, started = 0, completed = 0;
    uint64_t cancelled = 0, failed = 0, oracleComparisons = 0, oracleFailures = 0;
    uint64_t rejected = 0, selectedSequence = 0;
    uint64_t scheduleComparisons = 0, scheduleFailures = 0;
    uint64_t stateComparisons = 0, stateFailures = 0, stateFallbacks = 0;
    uint64_t membershipChanges = 0, passChanges = 0, constraintChanges = 0;
    uint64_t queueChanges = 0, realizationChanges = 0;
    uint64_t completedCacheHits = 0;
    size_t active = 0, pending = 0, peakActive = 0, peakRunning = 0;
    size_t retainedBytes = 0;
    std::string lastError;
    std::string lastStateFallback;
};

// Request/Pump/Reset/Shutdown belong to a single preparation owner. Workers
// share only owned immutable inputs and an atomic completion mailbox per job.
// Completion order never changes the order of publication.
class GraphCompileCoordinator {
public:
    struct RequestReceipt {
        uint64_t sequence = 0;
        std::shared_ptr<const GraphCompileInput> input;
    };
    explicit GraphCompileCoordinator(std::shared_ptr<runtime::ITaskService> tasks,
        size_t concurrency = 2);
    ~GraphCompileCoordinator();
    GraphCompileCoordinator(const GraphCompileCoordinator&) = delete;
    GraphCompileCoordinator& operator=(const GraphCompileCoordinator&) = delete;
    uint64_t Request(GraphCompileInput input);
    // Bootstrap may compile inline, but invokes the same CompileGraph entry
    // point and workspace type as worker jobs. No alternate compiler exists.
    RequestReceipt RequestOwned(GraphCompileInput input, bool compileInline = false);
    void Pump();
	void WaitForLatest();
	void WaitForSequence(uint64_t sequence);
    // Waits for, and consumes, exactly sequence. Later completed requests stay
    // in the reorder buffer. Compilation failure for sequence is reported by
    // exception rather than leaving an unfillable queue hole.
    std::shared_ptr<const CompiledGraphBundle> WaitAndPop(uint64_t sequence);
    void SetConcurrency(size_t concurrency);
    void Reset(uint64_t generation);
    void Shutdown();
    std::shared_ptr<const CompiledGraphBundle> Latest() const { return m_latest; }
    // Returns the oldest completed request newer than the admitted sequence.
    // Ready results remain ordered even when worker completion is reversed.
    std::shared_ptr<const CompiledGraphBundle> AcquireNextReadyAfter(uint64_t sequence);
    CompileCoordinatorStatistics Statistics() const;
private:
    struct RequestState {
        uint64_t sequence;
        std::shared_ptr<const GraphCompileInput> input;
        std::chrono::steady_clock::time_point queued;
    };
    struct Job;
    struct RunningStatistics {
        std::atomic_size_t running{0}, peak{0};
    };
    void StartPending();
    void Accept(const RequestState&, std::shared_ptr<const CompiledGraph>);
    std::shared_ptr<runtime::ITaskService> m_tasks;
    std::shared_ptr<runtime::ITaskScope> m_scope;
    std::shared_ptr<RunningStatistics> m_running = std::make_shared<RunningStatistics>();
    size_t m_concurrency;
    bool m_stopped = false;
    uint64_t m_generation = 0, m_sequence = 0;
    std::vector<std::shared_ptr<Job>> m_jobs;
    std::deque<RequestState> m_pending;
    std::map<uint64_t, std::shared_ptr<const CompiledGraphBundle>> m_ready;
    std::map<uint64_t, std::string> m_failures;
    std::shared_ptr<const CompiledGraphBundle> m_latest;
    // Structural plans only: never retain an old publication lease in this
    // bounded generation cache. Frame-slot rotations can revisit older keys.
    std::vector<std::shared_ptr<const CompiledGraph>> m_completedPlans;
    static constexpr size_t kMaximumCompletedPlans = 8;
    std::weak_ptr<const GraphCompileInput> m_previousRequest;
    CompileCoordinatorStatistics m_stats;
};

} // namespace org::experimental
