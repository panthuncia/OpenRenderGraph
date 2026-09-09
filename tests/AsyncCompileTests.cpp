#include "Render/RenderGraph/ExperimentalGraphCompiler.h"
#include "Render/RenderGraph/ExperimentalExecutionState.h"
#include "Render/RenderGraph/RenderGraphCompileProfile.h"
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
#define CHECK(...) do { if (!(__VA_ARGS__)) { std::fprintf(stderr, "Check failed at %d: %s\n", __LINE__, #__VA_ARGS__); std::abort(); } } while (false)

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

struct PayloadLifecycleProbe final : IFramePayloadLifecycle {
    mutable std::atomic_uint32_t abandonCount{0};
    mutable std::atomic_uint8_t reason{UINT8_MAX};
    void Abandon(uint8_t value) const noexcept override {
        reason.store(value);
        abandonCount.fetch_add(1);
    }
};

GraphCompileInput Input(uint64_t generation = 1, uint64_t resource = 1) {
    GraphCompileInput input;
    input.structure.generation = generation;
    input.structure.registryGeneration = generation;
    input.structure.resourceIDs = {resource};
    input.structure.passes = {{0, 0, {{0, false}}}, {1, 0, {{0, true}}}, {2, 0, {{0, false}}}};
    for (uint32_t i = 0; i < input.structure.passes.size(); ++i) {
        input.structure.passes[i].preparedPassIndex = i;
        input.structure.passes[i].forceBatchIsolation = true;
    }
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

void RunFramePlanningTests();
int main() {
    RunFramePlanningTests();
    // Export after all step objects and their dynamically supplied names have
    // died. Telemetry retains callsites, so stack-local definitions are unsafe.
    {
        basic_telemetry::Session session({.mode = basic_telemetry::CaptureMode::Trace});
        for (unsigned i = 0; i < 32; ++i) {
            std::string name = "compile-step-lifetime-" + std::to_string(i);
            org::profile::ScopedCompileProfileStep step(name.c_str());
        }
        const auto snapshot = session.Snapshot();
        CHECK(snapshot.events.size() == 32);
        for (unsigned i = 0; i < 32; ++i) {
            const auto name = "compile-step-lifetime-" + std::to_string(i);
            CHECK(std::ranges::any_of(snapshot.scopeDefinitions,
                [&](const auto& definition) { return definition.name == name; }));
        }
    }
    // Cross-frame hazards are derived from concrete backing/subresource access
    // and actual submitted signals, never from compile-request order.
    {
        BackingAccessAdmissionLedger ledger;
        rhi::ResourceHandle handle{};
        handle.index = 41; handle.generation = 7;
        PreparedBackingState backing{
            .graphResourceID = 99,
            .resource = handle,
            .shape = {2, 1, false},
        };
        auto makeGraph = [&](bool write, uint32_t queue, uint32_t mip) {
            CompiledGraph graph;
            auto structure = std::make_shared<GraphCompileStructure>();
            structure->resourceIDs = {99};
            structure->resourceShapes = {{2, 1, false}};
            structure->queues = {{0, true}, {0, true}};
            CompilePass pass;
            pass.preparedPassIndex = 0;
            pass.entryStates.push_back({0, {mip, 1, 0, 1}, {1, 1, 1, write}});
            structure->passes.push_back(std::move(pass));
            graph.structure = std::move(structure);
            graph.batches.push_back({0, queue});
            return graph;
        };
        const std::array queues{ExecutionTimelinePoint{11, 0}, ExecutionTimelinePoint{22, 0}};
        auto writeMip0 = makeGraph(true, 0, 0);
        GraphExecutionTimeline first;
        first.batches = {{{11, 5}, {}}};
        ledger.Commit(writeMip0, std::span{&backing, size_t{1}}, first);

        auto readMip0 = makeGraph(false, 1, 0);
        std::vector<std::vector<ExecutionTimelinePoint>> waits(1);
        ledger.AppendIncomingWaits(readMip0, std::span{&backing, size_t{1}}, queues, waits);
        CHECK(waits[0] == std::vector<ExecutionTimelinePoint>{{11, 5}});
        GraphExecutionTimeline second;
        second.batches = {{{22, 7}, {}}};
        ledger.Commit(readMip0, std::span{&backing, size_t{1}}, second);

        auto writeMip0Again = makeGraph(true, 0, 0);
        waits.assign(1, {});
        ledger.AppendIncomingWaits(writeMip0Again, std::span{&backing, size_t{1}}, queues, waits);
        CHECK(waits[0] == std::vector<ExecutionTimelinePoint>{{22, 7}});

        // A disjoint mip has no dependency on mip zero.
        auto readMip1 = makeGraph(false, 1, 1);
        waits.assign(1, {});
        ledger.AppendIncomingWaits(readMip1, std::span{&backing, size_t{1}}, queues, waits);
        CHECK(waits[0].empty());
    }

    // Physical aliases are ordered by the realized heap interval even when
    // their logical resource handles differ. Disjoint ranges remain parallel.
    {
        AliasAccessAdmissionLedger ledger;
        auto makeGraph = [](uint64_t resourceID, uint32_t queue) {
            CompiledGraph graph;
            auto structure = std::make_shared<GraphCompileStructure>();
            structure->resourceIDs = {resourceID};
            structure->resourceShapes = {{1, 1, false}};
            structure->queues = {{0, true}, {0, true}};
            CompilePass pass;
            pass.preparedPassIndex = 0;
            pass.entryStates.push_back({0, {0, 1, 0, 1}, {1, 0, 1, false}});
            structure->passes.push_back(std::move(pass));
            graph.structure = std::move(structure);
            graph.batches.push_back({0, queue});
            return graph;
        };
        const auto* heap = reinterpret_cast<const org::AliasHeapGeneration*>(uintptr_t{1});
        rhi::ResourceHandle firstHandle{}, secondHandle{}, disjointHandle{};
        firstHandle.index = 61; firstHandle.generation = 1;
        secondHandle.index = 62; secondHandle.generation = 1;
        disjointHandle.index = 63; disjointHandle.generation = 1;
        PreparedBackingState first{.graphResourceID = 101, .resource = firstHandle,
            .shape = {1, 1, false}, .aliasHeapIdentity = heap,
            .aliasOffset = 0, .aliasSize = 256};
        PreparedBackingState second{.graphResourceID = 102, .resource = secondHandle,
            .shape = {1, 1, false}, .aliasHeapIdentity = heap,
            .aliasOffset = 128, .aliasSize = 256};
        PreparedBackingState disjoint{.graphResourceID = 103, .resource = disjointHandle,
            .shape = {1, 1, false}, .aliasHeapIdentity = heap,
            .aliasOffset = 512, .aliasSize = 128};
        const std::array queues{ExecutionTimelinePoint{11, 0}, ExecutionTimelinePoint{22, 0}};
        auto firstGraph = makeGraph(101, 0);
        GraphExecutionTimeline submitted;
        submitted.batches = {{{11, 9}, {}}};
        ledger.Commit(firstGraph, std::span{&first, size_t{1}}, submitted);

        auto secondGraph = makeGraph(102, 1);
        std::vector<std::vector<ExecutionTimelinePoint>> waits(1);
        ledger.AppendIncomingWaits(secondGraph, std::span{&second, size_t{1}}, queues, waits);
        CHECK(waits[0] == std::vector<ExecutionTimelinePoint>{{11, 9}});
        GraphExecutionTimeline secondSubmitted;
        secondSubmitted.batches = {{{22, 12}, {}}};
        ledger.Commit(secondGraph, std::span{&second, size_t{1}}, secondSubmitted);

        std::vector firstAgain{first};
        auto activations = ledger.ApplyInitialStates(firstGraph, firstAgain);
        CHECK(activations.size() == 1);
        CHECK(firstAgain[0].regions.size() == 1);
        CHECK(firstAgain[0].regions[0].state.access
            == static_cast<uint64_t>(rhi::ResourceAccessType::None));

        auto disjointGraph = makeGraph(103, 1);
        waits.assign(1, {});
        ledger.AppendIncomingWaits(disjointGraph, std::span{&disjoint, size_t{1}}, queues, waits);
        CHECK(waits[0].empty());
        std::vector disjointCopy{disjoint};
        CHECK(ledger.ApplyInitialStates(disjointGraph, disjointCopy).empty());
    }

    // Independent linear-ready selection and recursive longest-path oracle.
    // Exercise both the live size_t view and owned uint32_t view of shared code.
    {
        std::mt19937 random(0x43718);
        for (int trial = 0; trial < 100; ++trial) {
            constexpr uint32_t count = 20;
            std::vector<std::vector<uint32_t>> successors(count);
            std::vector<uint32_t> degrees(count), authoredOrder(count);
            for (uint32_t i = 0; i < count; ++i) {
                authoredOrder[i] = random() % 7; // Intentional order-key ties.
                for (uint32_t j = i + 1; j < count; ++j)
                    if (random() % 4 == 0) { successors[i].push_back(j); ++degrees[j]; }
            }
            auto remaining = degrees;
            std::vector<bool> visited(count);
            std::vector<uint32_t> expected;
            for (uint32_t step = 0; step < count; ++step) {
                uint32_t best = UINT32_MAX;
                for (uint32_t i = 0; i < count; ++i)
                    if (!visited[i] && !remaining[i] && (best == UINT32_MAX
                        || authoredOrder[i] < authoredOrder[best]
                        || (authoredOrder[i] == authoredOrder[best] && i < best))) best = i;
                CHECK(best != UINT32_MAX);
                visited[best] = true; expected.push_back(best);
                for (auto next : successors[best]) --remaining[next];
            }
            std::vector<uint32_t> longest(count, UINT32_MAX);
            std::function<uint32_t(uint32_t)> oracleLength = [&](uint32_t i) {
                if (longest[i] != UINT32_MAX) return longest[i];
                uint32_t best = 0;
                for (auto next : successors[i]) best = (std::max)(best, 1 + oracleLength(next));
                return longest[i] = best;
            };
            for (uint32_t i = 0; i < count; ++i) oracleLength(i);
            auto check = [&]<class Index>() {
                std::vector<uint32_t> scratch, criticality(count, 999);
                std::vector<Index> ready, order;
                auto run = [&](bool cancel) {
                    return org::compiler::BuildTopologicalOrder<Index>(count,
                        [&](Index i) { return authoredOrder[i]; },
                        [&](Index i) -> const auto& { return successors[i]; },
                        [&](Index i) { return degrees[i]; }, [=] { return cancel; }, scratch, ready, order);
                };
                CHECK(run(true) == org::compiler::TopologyResult::Cancelled);
                CHECK(run(false) == org::compiler::TopologyResult::Complete);
                CHECK(std::equal(order.begin(), order.end(), expected.begin(), expected.end()));
                org::compiler::ComputeCriticality(order,
                    [&](Index i) -> const auto& { return successors[i]; },
                    [&](Index i) { return criticality[i]; },
                    [&](Index i, uint32_t value) { criticality[i] = value; });
                CHECK(criticality == longest);
            };
            check.operator()<uint32_t>();
            check.operator()<size_t>();
        }

        // Queue choice is an exact shared kernel, not a copied worker policy.
        // Existing compatible work wins over preference, then preference wins
        // over the remaining compatible queues.
        const std::vector<uint32_t> ready{2, 0, 1};
        const std::vector<std::vector<uint32_t>> queues{{0, 1}, {1}, {2, 0}};
        const std::vector<uint32_t> preferred{1, 1, 2};
        auto select = [&](auto hasWork, auto fits) {
            return org::compiler::SelectFirstFitCandidate<uint32_t>(ready,
                [&](uint32_t node) -> std::span<const uint32_t> { return queues[node]; },
                [&](uint32_t node) { return preferred[node]; }, hasWork, fits);
        };
        auto selected = select([](uint32_t queue) { return queue == 0; },
            [](uint32_t node, uint32_t queue) { return node != 2 || queue != 0; });
        CHECK(selected && *selected == (std::pair<size_t, uint32_t>{0, 2}));
        selected = select([](uint32_t) { return false; },
            [](uint32_t node, uint32_t queue) { return node == 2 && queue == 2; });
        CHECK(selected && *selected == (std::pair<size_t, uint32_t>{0, 2}));
        selected = select([](uint32_t) { return false; },
            [](uint32_t, uint32_t) { return false; });
        CHECK(!selected);
    }
    std::atomic_bool cancelled{false};
    CompileWorkspace workspace;
    auto graph = CompileGraph(std::make_shared<const GraphCompileInput>(Input()), workspace, cancelled);
    CHECK(graph->edges == *Input().expectedEdges);
    CHECK((graph->topologicalOrder == std::vector<uint32_t>{0,1,2}));
    CHECK((graph->criticality == std::vector<uint32_t>{2,1,0}));
    CHECK(graph->stage == CompiledGraphStage::SymbolicSchedule);
    CHECK(ValidateSymbolicSchedule(Input(), *graph).empty());
    {
        auto packedInput = Input();
        packedInput.structure.resourceIDs = {1, 2};
        packedInput.structure.passes = {{0, 0, {{0, false}}}, {1, 0, {{1, false}}}};
        for (uint32_t i = 0; i < packedInput.structure.passes.size(); ++i)
            packedInput.structure.passes[i].preparedPassIndex = i;
        packedInput.expectedEdges.reset();
        auto packed = workspace.Compile(std::make_shared<const GraphCompileInput>(packedInput), cancelled);
        CHECK(packed->batches.size() == 1);
        CHECK(packed->batches[0].passes == (std::vector<uint32_t>{0, 1}));
        CHECK(ValidateSymbolicSchedule(packedInput, *packed).empty());
        auto owned = std::make_shared<const GraphCompileInput>(packedInput);
        auto bundle = std::make_shared<const CompiledGraphBundle>(CompiledGraphBundle{1, packed, owned});
        auto layout = BuildExecutionLayout(bundle, *owned);
        CHECK(layout->placements == (std::vector<ExecutionPassPlacement>{{0,0,0},{1,0,0}}));
    }
    {
        auto prepared = std::make_shared<const GraphCompileInput>(Input());
        CompiledGraphBundle compatible{1, graph, prepared};
        CHECK(IsExecutionCompatible(compatible, *prepared));
        auto compatibleOwner = std::make_shared<const CompiledGraphBundle>(compatible);
        auto layout = BuildExecutionLayout(compatibleOwner, *prepared);
        CHECK(layout->bundle == compatibleOwner);
        CHECK(layout->placements == (std::vector<ExecutionPassPlacement>{
            {0,0,0}, {1,1,0}, {2,2,0}}));
        auto wrongGeneration = Input(2);
        CHECK(!IsExecutionCompatible(compatible, wrongGeneration));
        auto duplicateMap = Input();
        duplicateMap.structure.passes[2].preparedPassIndex = 1;
        auto duplicateGraph = CompileGraph(
            std::make_shared<const GraphCompileInput>(duplicateMap), workspace, cancelled);
        CHECK(!IsExecutionCompatible(CompiledGraphBundle{2, duplicateGraph,
            std::make_shared<const GraphCompileInput>(duplicateMap)}, duplicateMap));
        bool rejectedLayout = false;
        try { BuildExecutionLayout(compatibleOwner, wrongGeneration); }
        catch (const std::invalid_argument&) { rejectedLayout = true; }
        CHECK(rejectedLayout);
    }
    // Admission reserves distinct values for each execution, but publishes only
    // batches the backend actually submitted. No compile request consumes them.
    {
        auto input = Input();
        input.structure.queues = {{0, true}, {0, true}};
        input.structure.passes[1].compatibleQueueSlots = {1};
        input.structure.passes[1].preferredQueueSlot = 1;
        auto owned = std::make_shared<const GraphCompileInput>(input);
        auto compiled = workspace.Compile(owned, cancelled);
        auto bundle = std::make_shared<const CompiledGraphBundle>(CompiledGraphBundle{1, compiled, owned});
        ExecutionTimelineAdmission admission({{10, 100}, {20, 200}});
        std::vector<std::vector<ExecutionTimelinePoint>> waits(3);
        waits[0] = {{30, 2}, {30, 7}, {30, 1}, {10, 100}};
        auto first = admission.Prepare(bundle, waits);
        CHECK(first->batches[0].signal == (ExecutionTimelinePoint{10, 101}));
        CHECK(first->batches[1].signal == (ExecutionTimelinePoint{20, 201}));
        CHECK(first->batches[2].signal == (ExecutionTimelinePoint{10, 102}));
        CHECK(first->batches[0].waits == (std::vector<ExecutionTimelinePoint>{{10,100},{30,7}}));
        CHECK(first->batches[1].waits == (std::vector<ExecutionTimelinePoint>{{10,101}}));
        auto rejects = [](auto fn) { try { fn(); } catch (const std::exception&) { return true; } return false; };
        CHECK(rejects([&] { admission.Prepare(bundle, waits); }));
        CHECK(rejects([&] { admission.CommitBatch(first->submission, 1); }));
        CHECK(admission.Submitted()[0].value == 100);
        for (uint32_t i = 0; i < 3; ++i) admission.CommitBatch(first->submission, i);
        CHECK(admission.Submitted()[0].value == 102);
        CHECK(rejects([&] { admission.CommitBatch(first->submission, 2); }));
        waits[0] = {{10, 103}};
        CHECK(rejects([&] { admission.Prepare(bundle, waits); }));
        waits[0] = {{10, 102}};
        auto second = admission.Prepare(bundle, waits);
        CHECK(second->submission == first->submission + 1);
        CHECK(second->batches[0].signal.value == 103);
        CHECK(first->batches[0].signal.value == 101);
        admission.CommitBatch(second->submission, 0);
        admission.Fail(second->submission);
        CHECK(admission.Failed());
        CHECK(admission.Submitted()[0].value == 103);
        CHECK(admission.Submitted()[1].value == 201);
        CHECK(rejects([&] { admission.CommitBatch(second->submission, 1); }));
        CHECK(rejects([&] { admission.Prepare(bundle, waits); }));
        CHECK(rejects([] { ExecutionTimelineAdmission bad({{10,0},{10,0}}); }));
        ExecutionTimelineAdmission overflow({{10,UINT64_MAX},{20,0}});
        CHECK(rejects([&] { overflow.Prepare(bundle, std::vector<std::vector<ExecutionTimelinePoint>>(3)); }));
        CHECK(overflow.Submitted()[0].value == UINT64_MAX);
        ExecutionTimelineAdmission bounded({{10,0},{20,0}}, 1);
        auto packet = bounded.Prepare(bundle, std::vector<std::vector<ExecutionTimelinePoint>>(3));
        std::weak_ptr<const GraphExecutionTimeline> lifetime = packet;
        for (uint32_t i = 0; i < 3; ++i) bounded.CommitBatch(packet->submission, i);
        packet.reset();
        CHECK(!lifetime.expired());
        CHECK(bounded.InFlight() == 1);
        CHECK(rejects([&] { bounded.Prepare(bundle, std::vector<std::vector<ExecutionTimelinePoint>>(3)); }));
        std::vector<ExecutionTimelinePoint> completed{{10,1},{20,1}};
        CHECK(bounded.RetireCompleted(completed) == 0);
        CHECK(!lifetime.expired());
        completed[0].value = 2;
        CHECK(bounded.RetireCompleted(completed) == 1);
        CHECK(lifetime.expired());
        CHECK(bounded.InFlight() == 0);
        completed[0].value = 1;
        CHECK(rejects([&] { bounded.RetireCompleted(completed); }));
        completed[0].value = 3;
        CHECK(rejects([&] { bounded.RetireCompleted(completed); }));
        auto incompatible = std::make_shared<CompiledGraphBundle>(*bundle);
        incompatible->input = std::make_shared<const GraphCompileInput>(Input(2));
        ExecutionTimelineAdmission clean({{10,0},{20,0}});
        CHECK(rejects([&] { clean.Prepare(incompatible, waits); }));
        struct Packet : IPreparedExecutionBatch {
            uint32_t queueSlot = 0;
            uint32_t QueueSlot() const noexcept override { return queueSlot; }
            bool succeeds = true;
            mutable size_t calls = 0, completed = 0, abandoned = 0;
            SubmissionReceipt Submit(const ExecutionBatchTimeline&) const noexcept override {
                ++calls; return {succeeds ? SubmissionState::Signaled : SubmissionState::SubmittedWithoutSignal};
            }
            void Complete(uint64_t) const noexcept override { ++completed; }
            void Abandon() const noexcept override { ++abandoned; }
        };
        std::vector<std::shared_ptr<const IPreparedExecutionBatch>> packets;
        for (int i = 0; i < 3; ++i) {
            auto packet = std::make_shared<Packet>(); packet->queueSlot = i == 1 ? 1 : 0;
            packets.push_back(packet);
        }
        ExecutionTimelineAdmission submission({{10,0},{20,0}});
        auto submitted = submission.SubmitPrepared(bundle,
            std::vector<std::vector<ExecutionTimelinePoint>>(3), packets);
        CHECK(submission.Submitted()[0].value == 2);
        CHECK(submission.Submitted()[1].value == 1);
        std::weak_ptr<const IPreparedExecutionBatch> packetLease = packets[0];
        packets.clear(); submitted.reset();
        CHECK(!packetLease.expired());
        CHECK(submission.RetireCompleted(std::vector<ExecutionTimelinePoint>{{10,2},{20,1}}) == 1);
        CHECK(packetLease.expired());
        auto firstPacket = std::make_shared<Packet>();
        auto failedPacket = std::make_shared<Packet>(); failedPacket->succeeds = false;
        failedPacket->queueSlot = 1;
        auto lastPacket = std::make_shared<Packet>();
        org::FrameSlotPool failureSlots(1);
        auto failureFrame = std::make_shared<org::FrameContext>(1, 1, failureSlots.TryAcquire(0));
        for (auto stage : {org::FrameStage::Preparing, org::FrameStage::Compiling,
            org::FrameStage::Planned, org::FrameStage::Recording})
            failureFrame->Advance(stage, static_cast<org::FrameStage>(static_cast<unsigned>(stage) + 1));
        auto failureInput = std::make_shared<GraphCompileInput>(*bundle->input);
        failureInput->frameContext = failureFrame;
        auto failureBundle = std::make_shared<CompiledGraphBundle>(*bundle);
        failureBundle->input = failureInput;
        CHECK(rejects([&] { submission.SubmitPrepared(failureBundle,
            std::vector<std::vector<ExecutionTimelinePoint>>(3), {firstPacket, failedPacket, lastPacket}); }));
        CHECK(submission.Failed());
        CHECK(submission.Failure()->batch == 1);
        CHECK(submission.Failure()->receipt.state == SubmissionState::SubmittedWithoutSignal);
        CHECK(firstPacket->calls == 1 && failedPacket->calls == 1 && lastPacket->calls == 0);
        CHECK(submission.Submitted()[0].value == 3 && submission.Submitted()[1].value == 1);
        CHECK(failureFrame->Stage() == org::FrameStage::Recovery);
        CHECK(rejects([&] { failureFrame->CancelAfterJoin(); }));
        failureFrame.reset(); failureInput.reset(); failureBundle.reset();
        CHECK(!failureSlots.TryAcquire(0));
    }
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
    auto semanticSlots = enumerated;
    semanticSlots.structure.resourceIDs = {42, 42}; // Deliberate hash collision.
    semanticSlots.structure.resourceKeys = {"frame:b", "frame:a"};
    semanticSlots.structure.passes[0].accesses = {{0, false}, {1, true}};
    NormalizeCompileInput(semanticSlots);
    CHECK((semanticSlots.structure.resourceKeys == std::vector<std::string>{"frame:a", "frame:b"}));
    CHECK((semanticSlots.structure.passes[0].accesses == std::vector<CompileAccess>{{0, true}, {1, false}}));
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
    corrupted = *scheduled; corrupted.batches[1].passes = corrupted.batches[0].passes;
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
    auto packedStateInput = stateInput;
    for (auto& pass : packedStateInput.structure.passes) pass.forceBatchIsolation = false;
    auto packedStateGraph = workspace.Compile(
        std::make_shared<const GraphCompileInput>(packedStateInput), cancelled);
    CHECK(packedStateGraph->batches.size() == 1);
    CHECK(packedStateGraph->states.complete && packedStateGraph->stateValidationError.empty());
    CHECK(ValidateSymbolicStates(packedStateInput, *packedStateGraph).empty());
    auto crossQueueStateInput = stateInput;
    crossQueueStateInput.structure.queues = {{0, true}, {0, true}};
    crossQueueStateInput.structure.passes[0].compatibleQueueSlots = {0};
    crossQueueStateInput.structure.passes[1].compatibleQueueSlots = {1};
    crossQueueStateInput.structure.passes[2].compatibleQueueSlots = {1};
    auto crossQueueStateGraph = workspace.Compile(
        std::make_shared<const GraphCompileInput>(crossQueueStateInput), cancelled);
    auto crossQueueStep = std::find_if(crossQueueStateGraph->states.steps.begin(),
        crossQueueStateGraph->states.steps.end(), [](const auto& step) {
            return step.previousBatch != UINT32_MAX && step.previousBatch != step.batch;
        });
    CHECK(crossQueueStep != crossQueueStateGraph->states.steps.end());
    CHECK(crossQueueStep->previousPass != UINT32_MAX);
    rhi::ResourceHandle crossQueueHandle{};
    crossQueueHandle.index = 51; crossQueueHandle.generation = 9;
    PreparedBackingState crossQueueBacking{
        .graphResourceID = 1,
        .resource = crossQueueHandle,
        .shape = {4, 2, true},
        .regions = {{{0, 4, 0, 2}, {}}},
    };
    BackingStateAdmissionLedger stateLedger;
    auto crossQueueBarriers = stateLedger.Prepare(*crossQueueStateGraph,
        std::span{&crossQueueBacking, size_t{1}});
    const auto producerPosition = std::find(
        crossQueueStateGraph->batches[crossQueueStep->previousBatch].passes.begin(),
        crossQueueStateGraph->batches[crossQueueStep->previousBatch].passes.end(),
        crossQueueStep->previousPass)
        - crossQueueStateGraph->batches[crossQueueStep->previousBatch].passes.begin();
    const auto consumerPosition = std::find(
        crossQueueStateGraph->batches[crossQueueStep->batch].passes.begin(),
        crossQueueStateGraph->batches[crossQueueStep->batch].passes.end(),
        crossQueueStep->pass)
        - crossQueueStateGraph->batches[crossQueueStep->batch].passes.begin();
    CHECK(!crossQueueBarriers.batches[crossQueueStep->previousBatch]
        .afterPass[producerPosition].textures.empty());
    CHECK(!crossQueueBarriers.batches[crossQueueStep->batch]
        .beforePass[consumerPosition].textures.empty());
    CHECK(crossQueueBarriers.batches[crossQueueStep->previousBatch]
        .afterPass[producerPosition].textures.front().afterLayout == rhi::ResourceLayout::Common);
    CHECK(crossQueueBarriers.batches[crossQueueStep->batch]
        .beforePass[consumerPosition].textures.front().beforeLayout == rhi::ResourceLayout::Common);
    CHECK(crossQueueBarriers.batches[crossQueueStep->previousBatch]
        .afterPass[producerPosition].textures.front().afterSync == rhi::ResourceSyncState::All);
    CHECK(crossQueueBarriers.batches[crossQueueStep->batch]
        .beforePass[consumerPosition].textures.front().beforeSync == rhi::ResourceSyncState::All);
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
    auto ownedPlan = stateCoordinator.PeekReady(1)->graph;
    CHECK(stateCoordinator.Statistics().stateComparisons == 1 && stateCoordinator.Statistics().stateFailures == 0);
    // Sync-only changes are structural for state plans (unlike content/waits).
    stateInput.structure.passes[0].entryStates[0].state.sync ^= 128;
    stateCoordinator.Request(stateInput); stateTasks->RunAll(); stateCoordinator.Pump();
    CHECK(stateCoordinator.Statistics().started == 2);
    ++stateInput.structure.registryGeneration;
    stateCoordinator.Request(stateInput); stateTasks->RunAll(); stateCoordinator.Pump();
    CHECK(stateCoordinator.Statistics().started == 3);
    stateInput.backingGenerations = {1};
    auto realizationLease = std::make_shared<int>(1);
    stateInput.leases = {realizationLease};
    const auto realizationSequence = stateCoordinator.Request(stateInput);
    CHECK(stateCoordinator.Statistics().started == 3);
    CHECK(stateCoordinator.PeekReady(realizationSequence)->sequence == realizationSequence);
    CHECK(stateCoordinator.PeekReady(realizationSequence)->input->backingGenerations == std::vector<uint64_t>{1});
    ++stateInput.backingGenerations[0];
    auto newerRealizationLease = std::make_shared<int>(2);
    std::weak_ptr<int> oldRealization = realizationLease;
    stateInput.leases = {newerRealizationLease}; realizationLease.reset();
    const auto replacementSequence = stateCoordinator.Request(stateInput);
    CHECK(stateCoordinator.Statistics().started == 3);
    CHECK(stateCoordinator.PeekReady(replacementSequence)->sequence == replacementSequence);
    CHECK(stateCoordinator.PeekReady(replacementSequence)->input->backingGenerations == std::vector<uint64_t>{2});
    CHECK(!oldRealization.expired()); // Every queued frame retains its own realization.
    for (uint64_t sequence = 1; sequence <= realizationSequence; ++sequence) {
        auto queuedFrame = stateCoordinator.WaitAndPop(sequence);
        CHECK(queuedFrame && queuedFrame->sequence == sequence);
    }
    CHECK(oldRealization.expired());
    CHECK(stateCoordinator.WaitAndPop(replacementSequence)->sequence == replacementSequence);
    CHECK(stateCoordinator.Statistics().realizationChanges == 2);
    stateCoordinator.Shutdown();
    CHECK(ownedPlan->states.complete); // No pointers into the workspace/coordinator.

    // Synchronous bootstrap and worker compilation call the same CompileGraph
    // implementation. Inline bootstrap must not enqueue a duplicate job.
    {
        auto inlineTasks = std::make_shared<ManualTasks>();
        GraphCompileCoordinator inlineCoordinator(inlineTasks, 2);
        const auto receipt = inlineCoordinator.RequestOwned(Input(1, 9), true);
        CHECK(receipt.sequence == 1);
        CHECK(inlineCoordinator.PeekReady(receipt.sequence));
        CHECK(inlineTasks->queue.empty());
        CHECK(inlineCoordinator.Statistics().started == 1);
        CHECK(inlineCoordinator.Statistics().completed == 1);
    }

    GraphCompileCoordinator coordinator(tasks, 2);
    auto firstReceipt = coordinator.RequestOwned(Input(1, 10));
    const auto first = firstReceipt.sequence;
    CHECK(first && firstReceipt.input && firstReceipt.input->structure.resourceIDs == std::vector<uint64_t>{10});
    const auto second = coordinator.Request(Input(1, 20));
    CHECK(coordinator.Statistics().active == 2);
    tasks->Run(1); coordinator.Pump();
    CHECK(coordinator.PeekReady(second));
    CHECK(!coordinator.PeekReady(first)); // Sequence one is still compiling.
    tasks->Run(0); coordinator.Pump();
    CHECK(coordinator.PeekReady(first) && coordinator.PeekReady(second) && second > first);
    CHECK(coordinator.WaitAndPop(first)->sequence == first);
    CHECK(coordinator.WaitAndPop(second)->sequence == second);
    auto lease = std::make_shared<int>(42); std::weak_ptr<int> weakLease = lease;
    auto coalesced = Input(1, 20); coalesced.leases.push_back(lease); lease.reset();
    const auto third = coordinator.Request(std::move(coalesced));
    CHECK(coordinator.PeekReady(third) && !weakLease.expired());
    CHECK(coordinator.Statistics().started == 2);
    CHECK(coordinator.Statistics().scheduleFailures == 0);
    const auto retainedGraph = coordinator.PeekReady(third)->graph;
    const auto fourth = coordinator.Request(Input(1, 20));
    CHECK(!weakLease.expired()); // Queued frame retains its own content.
    CHECK(coordinator.WaitAndPop(third)->sequence == third);
    CHECK(weakLease.expired());
    CHECK(coordinator.WaitAndPop(fourth)->sequence == fourth);
    coordinator.Reset(2);
    CHECK(weakLease.expired() && !coordinator.PeekReady(third));
    CHECK(coordinator.Request(Input(1)) == 0);

    coordinator.Request(Input(2, 30));
    coordinator.Request(Input(2, 40));
    coordinator.Request(Input(2, 50));
    const auto latest = coordinator.Request(Input(2, 60));
    CHECK(coordinator.Statistics().active == 2 && coordinator.Statistics().pending == 2);
    tasks->RunAll(); coordinator.Pump(); tasks->RunAll(); coordinator.Pump();
    CHECK(coordinator.PeekReady(latest));
    CHECK(coordinator.Statistics().oracleFailures == 0);

    auto badOracle = Input(2, 70); badOracle.expectedEdges = std::make_shared<const DependencyEdges>();
    coordinator.Request(std::move(badOracle)); tasks->RunAll(); coordinator.Pump();
    CHECK(coordinator.Statistics().oracleFailures == 1);
    CHECK(coordinator.PeekReady(latest));
    coordinator.Request(Input(2, 80));
    auto cancelledPayload = std::make_shared<PayloadLifecycleProbe>();
    auto cancelledInput = Input(2, 81);
    cancelledInput.executionLifecycle = cancelledPayload;
    coordinator.Request(std::move(cancelledInput));
    coordinator.Reset(3); tasks->RunAll(); coordinator.Pump();
    CHECK(cancelledPayload->abandonCount == 1 && cancelledPayload->reason == 1);
    CHECK(!coordinator.PeekReady(1));
    tasks->reject = true;
    const auto inlineFallback = coordinator.Request(Input(3));
    coordinator.Pump();
    CHECK(coordinator.WaitAndPop(inlineFallback)->sequence == inlineFallback);
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
    CHECK(concurrent.PeekReady(newer));
    CHECK(concurrent.Statistics().completed == 2);
    CHECK(concurrent.Statistics().peakRunning == 2);
    CHECK(concurrent.Statistics().oracleFailures == 0);
    CHECK(concurrent.Statistics().scheduleComparisons == 2);
    CHECK(concurrent.Statistics().scheduleFailures == 0);
    CHECK(concurrent.Statistics().stateComparisons == 2 && concurrent.Statistics().stateFailures == 0);

    // A backing-only publication arriving while the structural job is active
    // must not start another compile. The completed plan is paired with the
    // newest coherent realization metadata and lease.
    auto realizationTasks = std::make_shared<ManualTasks>();
    GraphCompileCoordinator realizationRace(realizationTasks, 2);
    auto backingOne = std::make_shared<int>(1), backingTwo = std::make_shared<int>(2);
    auto payloadOne = std::make_shared<int>(11), payloadTwo = std::make_shared<int>(22);
    std::weak_ptr<int> payloadOneWeak = payloadOne, payloadTwoWeak = payloadTwo;
    std::weak_ptr<int> backingOneWeak = backingOne, backingTwoWeak = backingTwo;
    auto backingInput = Input(); backingInput.backingGenerations = {1};
    backingInput.leases = {backingOne}; backingOne.reset();
    backingInput.executionPayload = payloadOne; payloadOne.reset();
    realizationRace.Request(backingInput);
    backingInput.backingGenerations[0] = 2;
    backingInput.leases = {backingTwo}; backingTwo.reset();
    backingInput.executionPayload = payloadTwo; payloadTwo.reset();
    const auto latestRealization = realizationRace.Request(backingInput);
    backingInput.leases.clear();
    backingInput.executionPayload.reset();
    CHECK(realizationRace.Statistics().started == 1 && realizationRace.Statistics().coalesced == 1);
    CHECK(!backingOneWeak.expired() && !backingTwoWeak.expired());
    realizationTasks->RunAll(); realizationRace.Pump();
    CHECK(realizationRace.PeekReady(latestRealization));
    CHECK(realizationRace.PeekReady(latestRealization)->input->backingGenerations == std::vector<uint64_t>{2});
    CHECK(!backingOneWeak.expired() && !backingTwoWeak.expired());
    CHECK(!payloadOneWeak.expired() && !payloadTwoWeak.expired());
    CHECK(realizationRace.WaitAndPop(1)->sequence == 1);
    CHECK(backingOneWeak.expired() && payloadOneWeak.expired());
    CHECK(realizationRace.WaitAndPop(latestRealization)->sequence == latestRealization);
    realizationRace.Shutdown();
    CHECK(backingTwoWeak.expired());
    CHECK(payloadTwoWeak.expired());

    // Equivalent global-ID enumerations must coalesce while preserving the
    // newest publication. No resource, pass, or queue semantic change is hidden.
    auto canonicalTasks = std::make_shared<ManualTasks>();
    GraphCompileCoordinator canonical(canonicalTasks, 2);
    auto original = Input(); original.structure.resourceIDs = {10, 20};
    original.backingGenerations = {100, 200};
    original.structure.passes[0].accesses = {{0, false}, {1, false}};
    original.expectedEdges.reset();
    canonical.Request(original); canonicalTasks->RunAll(); canonical.Pump();
    auto reordered = original; reordered.structure.resourceIDs = {20, 10};
    for (auto& pass : reordered.structure.passes) {
        for (auto& access : pass.accesses) access.resourceIndex = 1 - access.resourceIndex;
        std::reverse(pass.accesses.begin(), pass.accesses.end());
    }
    std::reverse(reordered.backingGenerations.begin(), reordered.backingGenerations.end());
    canonical.Request(reordered);
    CHECK(canonical.Statistics().started == 1 && canonical.Statistics().coalesced == 1);
    CHECK(canonical.Statistics().membershipChanges == 0 && canonical.Statistics().passChanges == 0);
    CHECK(canonical.PeekReady(2)->input->backingGenerations == std::vector<uint64_t>({100,200}));
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
    auto firstPlan = rotation.PeekReady(1)->graph;
    std::weak_ptr<const CompiledGraph> evictedPlan = firstPlan;
    for (uint64_t id : {200, 300}) {
        rotation.Request(Input(1, id)); rotationTasks->RunAll(); rotation.Pump();
    }
    CHECK(!oldPublication.expired());
    CHECK(rotation.WaitAndPop(1)->sequence == 1);
    CHECK(oldPublication.expired()); // Consumed frame releases content; plan cache does not retain it.
    CHECK(rotation.WaitAndPop(2)->sequence == 2);
    CHECK(rotation.WaitAndPop(3)->sequence == 3);
    const auto revisitedSequence = rotation.Request(Input(1, 100));
    CHECK(rotation.PeekReady(revisitedSequence)->graph == firstPlan);
    CHECK(rotation.Statistics().completedCacheHits == 1 && rotation.Statistics().started == 3);
    CHECK(rotation.WaitAndPop(revisitedSequence)->sequence == revisitedSequence);
    firstPlan.reset();
    for (uint64_t id = 400; id < 1300; id += 100) {
        const auto sequence = rotation.Request(Input(1, id)); rotationTasks->RunAll(); rotation.Pump();
        CHECK(rotation.WaitAndPop(sequence)->sequence == sequence);
    }
    CHECK(evictedPlan.expired()); // Eight completed structures at most.
    rotation.Reset(2);
    const auto beforeResetBuilds = rotation.Statistics().started;
    rotation.Request(Input(2, 1200)); rotationTasks->RunAll(); rotation.Pump();
    CHECK(rotation.Statistics().started == beforeResetBuilds + 1);
    const auto goodSequence = rotation.Statistics().highestReadySequence;
    auto invalidJob = Input(2, 1400); invalidJob.structure.placementEdges = {{2, 0}};
    const auto invalidSequence = rotation.Request(invalidJob); rotationTasks->RunAll(); rotation.Pump();
    CHECK(rotation.Statistics().failed == 1 && rotation.Statistics().highestReadySequence == goodSequence);
    bool sawOrderedFailure = false;
    try { (void)rotation.WaitAndPop(invalidSequence); }
    catch (const std::runtime_error&) { sawOrderedFailure = true; }
    CHECK(sawOrderedFailure);
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
            CHECK(teardown.Statistics().active == 2 && teardown.Statistics().pending == 2);
            CHECK(!publications[2].expired() && !publications[3].expired());
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
