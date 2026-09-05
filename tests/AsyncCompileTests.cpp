#include "Render/RenderGraph/ExperimentalGraphCompiler.h"
#include <algorithm>
#include <barrier>
#include <cstdio>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>

using namespace org::experimental;
using namespace org::runtime;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Check failed at %d: %s\n", __LINE__, #x); std::abort(); } } while (false)

// The test scheduler permits deterministic reversed completion without sleeps.
struct ManualTasks : ITaskService {
    struct Scope : ITaskScope {
        ManualTasks* tasks;
        explicit Scope(ManualTasks* t) : tasks(t) {}
        void Cancel() noexcept override {}
        void Wait() override { tasks->RunAll(); }
        void CancelAndWait() override { Wait(); }
    };
    std::vector<std::function<void()>> queue;
    bool reject = false;
    void Run(size_t i) { auto task = std::move(queue.at(i)); if (task) task(); }
    void RunAll() { for (size_t i = 0; i < queue.size(); ++i) Run(i); }
    void ParallelFor(std::string_view, size_t n, std::function<void(size_t)> fn) override {
        for (size_t i = 0; i < n; ++i) fn(i);
    }
    void ParallelForLimited(std::string_view name, size_t n, size_t, std::function<void(size_t)> fn) override {
        ParallelFor(name, n, std::move(fn));
    }
    std::shared_ptr<ITaskScope> CreateScope(std::string_view) override { return std::make_shared<Scope>(this); }
    bool Submit(const std::shared_ptr<ITaskScope>&, TaskPriority, std::string_view, std::function<void()>&& fn) override {
        if (reject) return false;
        queue.push_back(std::move(fn)); return true;
    }
    bool ScheduleAfter(const std::shared_ptr<ITaskScope>& scope, std::chrono::steady_clock::duration,
        TaskPriority priority, std::string_view name, std::function<void()>&& fn) override {
        return Submit(scope, priority, name, std::move(fn));
    }
};

GraphCompileInput Input(uint64_t generation = 1, uint64_t resource = 1) {
    GraphCompileInput input;
    input.structure.generation = generation;
    input.structure.registryGeneration = generation;
    input.structure.resourceIDs = {resource};
    input.structure.passes = {{0, 0, {{0, false}}}, {1, 0, {{0, true}}}, {2, 0, {{0, false}}}};
    input.expectedEdges = std::make_shared<const DependencyEdges>(DependencyEdges{{0, 1}, {1, 2}});
    return input;
}

// Independent quadratic oracle: a hazard between two accesses is required
// unless an intervening writer already carries that hazard transitively.
DependencyEdges Oracle(const GraphCompileStructure& s) {
    DependencyEdges edges;
    for (uint32_t resource = 0; resource < s.resourceIDs.size(); ++resource) {
        std::vector<std::pair<uint32_t, bool>> uses;
        for (uint32_t pass = 0; pass < s.passes.size(); ++pass)
            for (auto a : s.passes[pass].accesses) if (a.resourceIndex == resource) uses.emplace_back(pass, a.write);
        for (size_t i = 0; i < uses.size(); ++i) {
            if (i && s.passes[uses[i-1].first].backend != s.passes[uses[i].first].backend)
                edges.emplace_back(uses[i-1].first, uses[i].first);
            for (size_t j = i + 1; j < uses.size(); ++j) {
                if (uses[i].second || uses[j].second) edges.emplace_back(uses[i].first, uses[j].first);
                if (uses[j].second) break;
            }
        }
    }
    for (auto e : s.explicitEdges) if (e.first != e.second) edges.push_back(e);
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    return edges;
}

int main() {
    std::atomic_bool cancelled{false};
    CompileWorkspace workspace;
    auto graph = workspace.Compile(std::make_shared<const GraphCompileInput>(Input()), cancelled);
    CHECK(graph->edges == *Input().expectedEdges);
    CHECK((graph->topologicalOrder == std::vector<uint32_t>{0,1,2}));
    CHECK((graph->criticality == std::vector<uint32_t>{2,1,0}));
    CHECK(graph->stage == CompiledGraphStage::SymbolicSchedule);
    CHECK(ValidateSymbolicSchedule(Input(), *graph).empty());
    cancelled = true;
    CHECK(!workspace.Compile(std::make_shared<const GraphCompileInput>(Input()), cancelled));
    cancelled = false;
    auto cyclic = Input(); cyclic.structure.explicitEdges = {{2, 0}};
    bool threw = false;
    try { workspace.Compile(std::make_shared<const GraphCompileInput>(cyclic), cancelled); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    // Capture normalization changes enumeration, not semantic resource identity.
    auto enumerated = Input();
    enumerated.structure.resourceIDs = {20, 10};
    enumerated.structure.passes = {{0, 0, {{0, false}, {1, false}, {0, true}}}};
    enumerated.structure.explicitEdges = {{0, 0}};
    NormalizeCompileInput(enumerated);
    CHECK((enumerated.structure.resourceIDs == std::vector<uint64_t>{10, 20}));
    CHECK((enumerated.structure.passes[0].accesses == std::vector<CompileAccess>{{0, false}, {1, true}}));
    CHECK(enumerated.structure.explicitEdges.empty());
    auto zeroBased = Input(1, 0);
    NormalizeCompileInput(zeroBased);
    CHECK(zeroBased.structure.resourceIDs[0] == 0);
    auto zeroGraph = workspace.Compile(std::make_shared<const GraphCompileInput>(zeroBased), cancelled);
    CHECK(ValidateSymbolicSchedule(zeroBased, *zeroGraph).empty());
    auto invalidIDs = enumerated; invalidIDs.structure.resourceIDs = {10, 10};
    threw = false;
    try { NormalizeCompileInput(invalidIDs); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    // Resource-less cross-queue dependency and read/read ownership must survive
    // queue wait coalescing. Mutation tests exercise an independent validator.
    auto multiQueue = Input();
    multiQueue.structure.queues = {{0, true}, {0, true}, {1, false}};
    multiQueue.structure.passes[0].compatibleQueueSlots = {0};
    multiQueue.structure.passes[1].compatibleQueueSlots = {1};
    multiQueue.structure.passes[2].compatibleQueueSlots = {0};
    multiQueue.structure.placementEdges = {{0, 2}};
    multiQueue.expectedSchedulingEdges = std::make_shared<const DependencyEdges>(DependencyEdges{{0,1},{1,2},{0,2}});
    auto scheduled = workspace.Compile(std::make_shared<const GraphCompileInput>(multiQueue), cancelled);
    CHECK(scheduled->relativeWaits.size() == 2);
    CHECK(ValidateSymbolicSchedule(multiQueue, *scheduled).empty());
    auto corrupted = *scheduled; corrupted.relativeWaits.clear();
    CHECK(!ValidateSymbolicSchedule(multiQueue, corrupted).empty());
    corrupted = *scheduled; corrupted.batches[1].queue = 2;
    CHECK(!ValidateSymbolicSchedule(multiQueue, corrupted).empty());
    corrupted = *scheduled; corrupted.batches[1].pass = corrupted.batches[0].pass;
    CHECK(!ValidateSymbolicSchedule(multiQueue, corrupted).empty());
    corrupted = *scheduled; corrupted.relativeWaits.push_back({0, 1});
    CHECK(!ValidateSymbolicSchedule(multiQueue, corrupted).empty());
    multiQueue.structure.passes[1].compatibleQueueSlots = {2};
    threw = false;
    try { workspace.Compile(std::make_shared<const GraphCompileInput>(multiQueue), cancelled); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    auto aliasCycle = Input(); aliasCycle.structure.placementEdges = {{2, 0}};
    threw = false;
    try { workspace.Compile(std::make_shared<const GraphCompileInput>(aliasCycle), cancelled); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    std::mt19937 rng(0x517a);
    for (int iteration = 0; iteration < 100; ++iteration) {
        auto input = Input(); input.structure.resourceIDs.resize(20); input.structure.passes.clear();
        std::iota(input.structure.resourceIDs.begin(), input.structure.resourceIDs.end(), 1);
        input.structure.queues = {{0, true}, {1, true}, {0, true}};
        for (uint32_t p = 0; p < 30; ++p) {
            CompilePass pass{p, rng() % 2, {}};
            for (uint32_t r = 0; r < 20; ++r) if (rng() % 3 == 0) pass.accesses.push_back({r, rng() % 4 == 0});
            pass.compatibleQueueSlots = {rng() % 3};
            input.structure.passes.push_back(std::move(pass));
        }
        input.structure.explicitEdges = {{0, 29}, {0, 29}, {2, 2}};
        auto output = workspace.Compile(std::make_shared<const GraphCompileInput>(input), cancelled);
        CHECK(output->edges == Oracle(input.structure));
        CHECK(ValidateSymbolicSchedule(input, *output).empty());
    }

    auto tasks = std::make_shared<ManualTasks>();
    // Owned state planning resolves neither first-use state nor absolute queue
    // values. Partial ranges and callback postconditions survive compilation.
    auto stateInput = Input();
    stateInput.structure.resourceShapes = {{4, 2, true}};
    const CompileResourceState readState{1, 2, 4, false}, writeState{8, 16, 32, true};
    stateInput.structure.passes[0].entryStates = {{0, {0, 4, 0, 2}, readState}};
    stateInput.structure.passes[1].entryStates = {{0, {1, 2, 1, 1}, writeState}};
    stateInput.structure.passes[1].exitStates = {{0, {1, 1, 1, 1}, readState}};
    stateInput.structure.passes[2].entryStates = {{0, {0, 4, 0, 2}, readState}};
    auto stateGraph = workspace.Compile(std::make_shared<const GraphCompileInput>(stateInput), cancelled);
    CHECK(stateGraph->states.complete && stateGraph->stateValidationError.empty());
    CHECK(stateGraph->states.steps.front().previousBatch == UINT32_MAX);
    CHECK(ValidateSymbolicStates(stateInput, *stateGraph).empty());
    auto dependencyOnly = stateInput;
    dependencyOnly.structure.resourceIDs.push_back(2);
    dependencyOnly.structure.resourceShapes.push_back({0, 0, false});
    dependencyOnly.structure.passes[0].accesses.push_back({1, false});
    auto dependencyOnlyGraph = workspace.Compile(std::make_shared<const GraphCompileInput>(dependencyOnly), cancelled);
    CHECK(dependencyOnlyGraph->states.complete && dependencyOnlyGraph->stateValidationError.empty());
    dependencyOnly.structure.passes[0].entryStates.push_back({1, {}, readState});
    CHECK(!workspace.Compile(std::make_shared<const GraphCompileInput>(dependencyOnly), cancelled)->states.complete);
    auto shapeRemap = stateInput;
    shapeRemap.structure.resourceIDs = {20, 10};
    shapeRemap.structure.resourceShapes.push_back({0, 0, false});
    NormalizeCompileInput(shapeRemap);
    CHECK(shapeRemap.structure.resourceShapes[1].mips == 4 && shapeRemap.structure.passes[0].entryStates[0].resource == 1);
    CHECK(workspace.Compile(std::make_shared<const GraphCompileInput>(shapeRemap), cancelled)->states.complete);
    auto missingHazard = stateInput;
    missingHazard.structure.passes[0].accesses.clear();
    CHECK(!workspace.Compile(std::make_shared<const GraphCompileInput>(missingHazard), cancelled)->states.complete);
    auto readOnlyStates = stateInput;
    readOnlyStates.structure.queues = {{0, true}, {0, true}};
    readOnlyStates.expectedEdges = std::make_shared<const DependencyEdges>();
    for (auto& pass : readOnlyStates.structure.passes) {
        pass.accesses.clear(); pass.exitStates.clear();
        pass.entryStates = {{0, {0, 4, 0, 2}, readState}};
    }
    readOnlyStates.structure.passes[1].compatibleQueueSlots = {1};
    auto readOnlyGraph = workspace.Compile(std::make_shared<const GraphCompileInput>(readOnlyStates), cancelled);
    CHECK(readOnlyGraph->edges.empty() && readOnlyGraph->relativeWaits.size() == 2);
    CHECK(readOnlyGraph->states.complete && readOnlyGraph->stateValidationError.empty());
    CHECK(ValidateSymbolicSchedule(readOnlyStates, *readOnlyGraph).empty());
    auto brokenState = *stateGraph;
    brokenState.states.steps.erase(brokenState.states.steps.begin());
    CHECK(!ValidateSymbolicStates(stateInput, brokenState).empty());
    brokenState = *stateGraph; brokenState.states.steps.back().before.sync ^= 1;
    CHECK(!ValidateSymbolicStates(stateInput, brokenState).empty());
    brokenState = *stateGraph; brokenState.states.finalStates.pop_back();
    CHECK(!ValidateSymbolicStates(stateInput, brokenState).empty());
    brokenState = *stateGraph; brokenState.states.finalStates.push_back(brokenState.states.finalStates.front());
    CHECK(!ValidateSymbolicStates(stateInput, brokenState).empty());
    auto conflictInput = stateInput;
    conflictInput.structure.passes[0].entryStates.push_back({0, {1, 1, 0, 1}, writeState});
    auto conflictGraph = workspace.Compile(std::make_shared<const GraphCompileInput>(conflictInput), cancelled);
    CHECK(!conflictGraph->states.complete && !conflictGraph->states.fallbackReason.empty());
    CHECK(conflictGraph->states.steps.empty() && conflictGraph->states.finalStates.empty());
    auto layoutOnly = stateInput;
    auto duplicateEntry = layoutOnly.structure.passes[0].entryStates[0];
    duplicateEntry.state.layout ^= 32;
    layoutOnly.structure.passes[0].entryStates.push_back(duplicateEntry);
    CHECK(!workspace.Compile(std::make_shared<const GraphCompileInput>(layoutOnly), cancelled)->states.complete);
    layoutOnly.structure.resourceShapes[0].hasLayout = false;
    auto bufferLayoutPlan = workspace.Compile(std::make_shared<const GraphCompileInput>(layoutOnly), cancelled);
    CHECK(bufferLayoutPlan->states.complete && bufferLayoutPlan->stateValidationError.empty());
    CHECK(bufferLayoutPlan->states.steps.front().after.layout == 0);
    conflictInput = stateInput;
    conflictInput.structure.passes[1].exitStates[0].range = {0, 4, 0, 2};
    CHECK(!workspace.Compile(std::make_shared<const GraphCompileInput>(conflictInput), cancelled)->states.complete);
    conflictInput = stateInput; conflictInput.structure.passes[0].entryStates[0].range.mips = UINT32_MAX;
    CHECK(!workspace.Compile(std::make_shared<const GraphCompileInput>(conflictInput), cancelled)->states.complete);
    for (unsigned iteration = 0; iteration < 100; ++iteration) {
        auto randomStates = stateInput;
        randomStates.structure.passes.clear();
        randomStates.expectedEdges.reset();
        for (uint32_t p = 0; p < 20; ++p) {
            CompilePass pass{p, 0, {{0, true}}};
            const uint32_t mip = rng() % 4, slice = rng() % 2;
            const CompileRange range{mip, 1 + rng() % (4 - mip), slice, 1 + rng() % (2 - slice)};
            pass.entryStates = {{0, range, rng() % 2 ? readState : writeState}};
            if (rng() % 2) pass.exitStates = {{0, range, readState}};
            randomStates.structure.passes.push_back(std::move(pass));
        }
        auto plan = workspace.Compile(std::make_shared<const GraphCompileInput>(randomStates), cancelled);
        CHECK(plan->states.complete && plan->stateValidationError.empty());
    }

    auto stateTasks = std::make_shared<ManualTasks>();
    GraphCompileCoordinator stateCoordinator(stateTasks);
    stateCoordinator.Request(stateInput); stateTasks->RunAll(); stateCoordinator.Pump();
    auto ownedPlan = stateCoordinator.Latest()->graph;
    CHECK(stateCoordinator.Statistics().stateComparisons == 1 && stateCoordinator.Statistics().stateFailures == 0);
    // Sync-only changes are structural for state plans (unlike content/waits).
    stateInput.structure.passes[0].entryStates[0].state.sync ^= 128;
    stateCoordinator.Request(stateInput); stateTasks->RunAll(); stateCoordinator.Pump();
    CHECK(stateCoordinator.Statistics().started == 2);
    ++stateInput.structure.registryGeneration;
    stateCoordinator.Request(stateInput); stateTasks->RunAll(); stateCoordinator.Pump();
    CHECK(stateCoordinator.Statistics().started == 3);
    stateCoordinator.Shutdown();
    CHECK(ownedPlan->states.complete); // No pointers into the workspace/coordinator.

    GraphCompileCoordinator coordinator(tasks, 2);
    const auto first = coordinator.Request(Input(1, 10));
    const auto second = coordinator.Request(Input(1, 20));
    CHECK(coordinator.Statistics().active == 2);
    tasks->Run(1); coordinator.Pump();
    CHECK(coordinator.Latest()->sequence == second);
    tasks->Run(0); coordinator.Pump();
    CHECK(coordinator.Latest()->sequence == second && second > first);
    auto lease = std::make_shared<int>(42); std::weak_ptr<int> weakLease = lease;
    auto coalesced = Input(1, 20); coalesced.leases.push_back(lease); lease.reset();
    const auto third = coordinator.Request(std::move(coalesced));
    CHECK(coordinator.Latest()->sequence == third && !weakLease.expired());
    CHECK(coordinator.Statistics().started == 2);
    CHECK(coordinator.Statistics().scheduleFailures == 0);
    const auto retainedGraph = coordinator.Latest()->graph;
    coordinator.Request(Input(1, 20));
    CHECK(weakLease.expired()); // Reusing a plan must not retain old content.
    coordinator.Reset(2);
    CHECK(weakLease.expired() && !coordinator.Latest());
    CHECK(coordinator.Request(Input(1)) == 0);

    coordinator.Request(Input(2, 30));
    coordinator.Request(Input(2, 40));
    coordinator.Request(Input(2, 50));
    const auto latest = coordinator.Request(Input(2, 60));
    CHECK(coordinator.Statistics().active == 2 && coordinator.Statistics().pending == 1);
    tasks->RunAll(); coordinator.Pump(); tasks->RunAll(); coordinator.Pump();
    CHECK(coordinator.Latest()->sequence == latest);
    CHECK(coordinator.Statistics().oracleFailures == 0);

    auto badOracle = Input(2, 70); badOracle.expectedEdges = std::make_shared<const DependencyEdges>();
    coordinator.Request(std::move(badOracle)); tasks->RunAll(); coordinator.Pump();
    CHECK(coordinator.Statistics().oracleFailures == 1);
    CHECK(coordinator.Latest()->sequence == latest);
    coordinator.Request(Input(2, 80));
    coordinator.Reset(3); tasks->RunAll(); coordinator.Pump();
    CHECK(!coordinator.Latest());
    tasks->reject = true;
    coordinator.Request(Input(3));
    CHECK(coordinator.Statistics().active == 0 && coordinator.Statistics().rejected >= 1);
    coordinator.Shutdown(); CHECK(coordinator.Request(Input(3)) == 0);

    // Actual overlap on separate workers with independently owned inputs. The
    // shared 1,000-resource/60-pass case exercises read fanout and replacements.
    auto concurrentTasks = std::make_shared<ManualTasks>();
    GraphCompileCoordinator concurrent(concurrentTasks, 2);
    auto large = Input(); large.structure.resourceIDs.resize(1000);
    large.structure.resourceShapes.resize(1000);
    std::iota(large.structure.resourceIDs.begin(), large.structure.resourceIDs.end(), 1);
    large.structure.passes.clear();
    for (uint32_t p = 0; p < 60; ++p) {
        CompilePass pass{p, 0, {}};
        for (uint32_t r = 0; r < 1000; ++r) pass.accesses.push_back({r, p == 0 || p == 59});
        for (uint32_t r = 0; r < 1000; ++r) pass.entryStates.push_back({r, {}, p == 0 || p == 59 ? writeState : readState});
        large.structure.passes.push_back(std::move(pass));
    }
    large.expectedEdges = std::make_shared<const DependencyEdges>(Oracle(large.structure));
    concurrent.Request(large);
    for (uint32_t r = 0; r < 10; ++r) large.structure.resourceIDs[r] += 1000;
    const auto newer = concurrent.Request(std::move(large));
    std::barrier gate(3);
    std::jthread a([&]{ gate.arrive_and_wait(); concurrentTasks->Run(0); });
    std::jthread b([&]{ gate.arrive_and_wait(); concurrentTasks->Run(1); });
    gate.arrive_and_wait(); a.join(); b.join(); concurrent.Pump();
    CHECK(concurrent.Latest()->sequence == newer);
    CHECK(concurrent.Statistics().completed == 2);
    CHECK(concurrent.Statistics().peakRunning == 2);
    CHECK(concurrent.Statistics().oracleFailures == 0);
    CHECK(concurrent.Statistics().scheduleComparisons == 2);
    CHECK(concurrent.Statistics().scheduleFailures == 0);
    CHECK(concurrent.Statistics().stateComparisons == 2 && concurrent.Statistics().stateFailures == 0);

    // Equivalent global-ID enumerations must coalesce while preserving the
    // newest publication. No resource, pass, or queue semantic change is hidden.
    auto canonicalTasks = std::make_shared<ManualTasks>();
    GraphCompileCoordinator canonical(canonicalTasks, 2);
    auto original = Input(); original.structure.resourceIDs = {10, 20};
    original.structure.passes[0].accesses = {{0, false}, {1, false}};
    original.expectedEdges.reset();
    canonical.Request(original); canonicalTasks->RunAll(); canonical.Pump();
    auto reordered = original; reordered.structure.resourceIDs = {20, 10};
    for (auto& pass : reordered.structure.passes) {
        for (auto& access : pass.accesses) access.resourceIndex = 1 - access.resourceIndex;
        std::reverse(pass.accesses.begin(), pass.accesses.end());
    }
    canonical.Request(reordered);
    CHECK(canonical.Statistics().started == 1 && canonical.Statistics().coalesced == 1);
    CHECK(canonical.Statistics().membershipChanges == 0 && canonical.Statistics().passChanges == 0);
    reordered.structure.resourceIDs[0] = 30;
    canonical.Request(reordered);
    CHECK(canonical.Statistics().membershipChanges == 1);

    auto rotationTasks = std::make_shared<ManualTasks>();
    GraphCompileCoordinator rotation(rotationTasks, 2);
    auto publication = std::make_shared<int>(73);
    std::weak_ptr<int> oldPublication = publication;
    auto firstSlot = Input(1, 100); firstSlot.leases.push_back(publication);
    publication.reset();
    rotation.Request(std::move(firstSlot)); rotationTasks->RunAll(); rotation.Pump();
    auto firstPlan = rotation.Latest()->graph;
    std::weak_ptr<const CompiledGraph> evictedPlan = firstPlan;
    for (uint64_t id : {200, 300}) {
        rotation.Request(Input(1, id)); rotationTasks->RunAll(); rotation.Pump();
    }
    CHECK(oldPublication.expired()); // Cache owns structure, not old content.
    const auto revisitedSequence = rotation.Request(Input(1, 100));
    CHECK(rotation.Latest()->sequence == revisitedSequence && rotation.Latest()->graph == firstPlan);
    CHECK(rotation.Statistics().completedCacheHits == 1 && rotation.Statistics().started == 3);
    firstPlan.reset();
    for (uint64_t id = 400; id < 1300; id += 100) {
        rotation.Request(Input(1, id)); rotationTasks->RunAll(); rotation.Pump();
    }
    CHECK(evictedPlan.expired()); // Eight completed structures at most.
    rotation.Reset(2);
    const auto beforeResetBuilds = rotation.Statistics().started;
    rotation.Request(Input(2, 1200)); rotationTasks->RunAll(); rotation.Pump();
    CHECK(rotation.Statistics().started == beforeResetBuilds + 1);
    const auto goodSequence = rotation.Latest()->sequence;
    auto invalidJob = Input(2, 1400); invalidJob.structure.placementEdges = {{2, 0}};
    rotation.Request(invalidJob); rotationTasks->RunAll(); rotation.Pump();
    CHECK(rotation.Statistics().failed == 1 && rotation.Latest()->sequence == goodSequence);
    // Teardown with two outstanding closures and a replaceable pending input.
    // The scheduler outlives the coordinator: no closure may keep a dangling
    // owner or retain publication leases after shutdown drains its scope.
    for (unsigned iteration = 0; iteration < 32; ++iteration) {
        auto teardownTasks = std::make_shared<ManualTasks>();
        std::vector<std::weak_ptr<int>> publications;
        {
            GraphCompileCoordinator teardown(teardownTasks, 2);
            for (uint64_t resource = 1; resource <= 4; ++resource) {
                auto input = Input(1, resource);
                auto publicationLease = std::make_shared<int>(static_cast<int>(resource));
                publications.push_back(publicationLease);
                input.leases.push_back(std::move(publicationLease));
                teardown.Request(std::move(input));
            }
            CHECK(teardown.Statistics().active == 2 && teardown.Statistics().pending == 1);
            CHECK(publications[2].expired()); // Replaced pending request.
            if (iteration % 2 == 0) {
                teardown.Reset(2);
                CHECK(publications[3].expired());
                teardown.Request(Input(2, 5));
            }
            if (iteration % 3 == 0) {
                teardown.Shutdown();
                teardown.Shutdown(); // Explicit shutdown followed by destruction.
                CHECK(teardown.Statistics().active == 0 && teardown.Statistics().pending == 0);
                CHECK(teardown.Request(Input(2, 6)) == 0);
            }
        }
        for (const auto& publicationLease : publications) CHECK(publicationLease.expired());
        for (const auto& queued : teardownTasks->queue) CHECK(!queued);
        teardownTasks->RunAll(); // No late callbacks after owner destruction.
    }
    std::puts("Async compiler ownership, oracle, ordering, cancellation and overlap tests passed.");
}
