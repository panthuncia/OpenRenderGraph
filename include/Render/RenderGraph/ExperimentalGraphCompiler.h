#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "Render/Runtime/ITaskService.h"

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

struct GraphCompileInput {
    GraphCompileStructure structure;
    // Retains the exact publication/backings that preparation resolved. No
    // worker dereferences an opaque lease or calls a resource/pass interface.
    std::vector<std::shared_ptr<const void>> leases;
    std::shared_ptr<const DependencyEdges> expectedEdges;
    std::shared_ptr<const DependencyEdges> expectedSchedulingEdges;
};

// One isolated pass per batch is the conservative first symbolic scheduler.
// Batch indices and waits are relative: no queue timeline is touched here.
struct SymbolicBatch {
    uint32_t pass = 0, queue = 0;
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
    struct ResourceSequence {
        uint32_t writer = UINT32_MAX;
        std::vector<uint32_t> readers;
        uint32_t lastAccess = UINT32_MAX;
        uint32_t backend = 0;
    };
    std::vector<ResourceSequence> m_resources;
    std::vector<std::vector<uint32_t>> m_successors;
    std::vector<uint32_t> m_indegrees;
    std::vector<uint32_t> m_ready;
};

struct CompiledGraphBundle {
    uint64_t sequence = 0;
    std::shared_ptr<const CompiledGraph> graph;
    // May be a newer coalesced publication with the same complete structure.
    std::shared_ptr<const GraphCompileInput> input;
};

struct CompileCoordinatorStatistics {
    uint64_t requested = 0, coalesced = 0, started = 0, completed = 0;
    uint64_t cancelled = 0, failed = 0, oracleComparisons = 0, oracleFailures = 0;
    uint64_t rejected = 0, selectedSequence = 0;
    uint64_t scheduleComparisons = 0, scheduleFailures = 0;
    uint64_t stateComparisons = 0, stateFailures = 0, stateFallbacks = 0;
    uint64_t membershipChanges = 0, passChanges = 0, constraintChanges = 0;
    uint64_t queueChanges = 0;
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
    explicit GraphCompileCoordinator(std::shared_ptr<runtime::ITaskService> tasks,
        size_t concurrency = 2);
    ~GraphCompileCoordinator();
    GraphCompileCoordinator(const GraphCompileCoordinator&) = delete;
    GraphCompileCoordinator& operator=(const GraphCompileCoordinator&) = delete;
    uint64_t Request(GraphCompileInput input);
    void Pump();
    void SetConcurrency(size_t concurrency);
    void Reset(uint64_t generation);
    void Shutdown();
    std::shared_ptr<const CompiledGraphBundle> Latest() const { return m_latest; }
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
    std::unique_ptr<RequestState> m_pending;
    std::shared_ptr<const CompiledGraphBundle> m_latest;
    // Structural plans only: never retain an old publication lease in this
    // bounded generation cache. Frame-slot rotations can revisit older keys.
    std::vector<std::shared_ptr<const CompiledGraph>> m_completedPlans;
    static constexpr size_t kMaximumCompletedPlans = 8;
    std::weak_ptr<const GraphCompileInput> m_previousRequest;
    CompileCoordinatorStatistics m_stats;
};

} // namespace org::experimental
