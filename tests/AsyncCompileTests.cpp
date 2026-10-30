#include "Render/RenderGraph/ExperimentalGraphCompiler.h"
#include "Render/BindingTable.h"
#include "Render/RenderGraph/ExperimentalExecutionState.h"
#include <BasicTelemetry/Telemetry.h>
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

void DiagnosticPassAttribution() {
    auto input = Input();
    input.structure.resourceIDs = {20, 10};
    input.structure.resourceShapes = {{1, 1, false}, {2, 1, true}};
    input.structure.diagnosticResourceNames = {"frameBuffer", "unusedTexture"};
    input.structure.diagnosticPassNames = {"reader", "writer", "consumer"};
    for (auto& pass : input.structure.passes)
        pass.entryStates.push_back({0, {0, 1, 0, 1}, {0, 0, 0, pass.accesses.front().write}});
    NormalizeCompileInput(input);
    CHECK(input.structure.diagnosticResourceNames == std::vector<std::string>{"unusedTexture", "frameBuffer"});
    std::atomic_bool cancelled{false};
    CompileWorkspace workspace;
    basic_telemetry::Session session({ .mode = basic_telemetry::CaptureMode::Trace,
        .traceRootScopes = {"ORG.FreshCompile.DependencyCompile"} });
    const auto graph = workspace.Compile(std::make_shared<const GraphCompileInput>(input), cancelled);
    CHECK(graph && graph->states.complete);
    const auto snapshot = session.Snapshot();
    for (const auto stage : {"ORG.FreshCompile.Dependencies.Pass", "ORG.FreshCompile.Schedule.Pass",
        "ORG.FreshCompile.States.Pass", "ORG.FreshCompile.Boundary.Pass",
        "ORG.FreshCompile.IncomingBoundary.Pass"}) {
        const auto definition = std::ranges::find_if(snapshot.scopeDefinitions,
            [&](const auto& d) { return d.name == stage; });
        CHECK(definition != snapshot.scopeDefinitions.end());
        for (const auto& label : input.structure.diagnosticPassNames)
            CHECK(std::ranges::count_if(snapshot.events, [&](const auto& e) {
                return e.scopeId == definition->id && e.text == label;
            }) == 1);
    }
    input.structure.diagnosticPassNames.clear();
    input.structure.diagnosticResourceNames.clear();
    const auto reference = workspace.Compile(std::make_shared<const GraphCompileInput>(input), cancelled);
    CHECK(reference && reference->edges == graph->edges && reference->batches == graph->batches);
    CHECK(reference->states.steps == graph->states.steps);
}

int main() {
    DiagnosticPassAttribution();
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
            auto structure = std::make_shared<GraphCompileStructure>();
            structure->resourceIDs = {99};
            structure->resourceShapes = {{2, 1, false}};
            structure->queues = {{0, true}, {0, true}};
            CompilePass pass;
            pass.preparedPassIndex = 0;
            pass.compatibleQueueSlots = {queue};
            pass.preferredQueueSlot = queue;
            pass.accesses = {{0, write}};
            pass.entryStates.push_back({0, {mip, 1, 0, 1}, {1, 1, 1, write}});
            structure->passes.push_back(std::move(pass));
            GraphCompileInput input;
            input.structure = std::move(*structure);
            CompileWorkspace workspace;
            std::atomic_bool cancelled{false};
            auto graph = CompileGraph(std::make_shared<const GraphCompileInput>(std::move(input)), workspace, cancelled);
            CHECK(graph && graph->stateValidationError.empty());
            return *graph;
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
            auto structure = std::make_shared<GraphCompileStructure>();
            structure->resourceIDs = {resourceID};
            structure->resourceShapes = {{1, 1, false}};
            structure->queues = {{0, true}, {0, true}};
            CompilePass pass;
            pass.preparedPassIndex = 0;
            pass.compatibleQueueSlots = {queue};
            pass.preferredQueueSlot = queue;
            pass.accesses = {{0, false}};
            pass.entryStates.push_back({0, {0, 1, 0, 1}, {1, 0, 1, false}});
            structure->passes.push_back(std::move(pass));
            GraphCompileInput input;
            input.structure = std::move(*structure);
            CompileWorkspace workspace;
            std::atomic_bool cancelled{false};
            auto graph = CompileGraph(std::make_shared<const GraphCompileInput>(std::move(input)), workspace, cancelled);
            CHECK(graph && graph->stateValidationError.empty());
            return *graph;
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
        CHECK(activations[0].index == first.resource.index && activations[0].generation == first.resource.generation);
        // Activation is returned to state admission; this ledger does not
        // rewrite the immutable selected backing snapshot.
        CHECK(!firstAgain[0].regions);

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
        auto owned = std::make_shared<const GraphCompileInput>(packedInput);
        auto packed = workspace.Compile(owned, cancelled);
        CHECK(packed->batches.size() == 1);
        CHECK(packed->batches[0].passes == (std::vector<uint32_t>{0, 1}));
        CHECK(ValidateSymbolicSchedule(packedInput, *packed).empty());
        auto bundle = std::make_shared<const CompiledGraphBundle>(CompiledGraphBundle{1, packed, owned});
        auto layout = BuildExecutionLayout(bundle, *owned);
        CHECK(layout->placements == (std::vector<ExecutionPassPlacement>{{0,0,0},{1,0,0}}));
    }
    {
        auto prepared = std::make_shared<const GraphCompileInput>(Input());
        auto compatibleGraph = CompileGraph(prepared, workspace, cancelled);
        CompiledGraphBundle compatible{1, compatibleGraph, prepared};
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
        auto cleanup = org::runtime::ResourceCleanupQueue::Create();
        auto sourceOwner = cleanup->Make<int>(42);
        std::weak_ptr<int> sourceWeak = sourceOwner;
        auto reusable = org::ExecutionResourceLease::Create(cleanup, {sourceOwner});
        sourceOwner.reset();
        auto packet = bounded.Prepare(bundle, std::vector<std::vector<ExecutionTimelinePoint>>(3), {reusable.Owner()});
        std::weak_ptr<const GraphExecutionTimeline> lifetime = packet;
        for (uint32_t i = 0; i < 3; ++i) bounded.CommitBatch(packet->submission, i);
        packet.reset();
        CHECK(!lifetime.expired());
        CHECK(bounded.InFlight() == 1);
        CHECK(rejects([&] { bounded.Prepare(bundle, std::vector<std::vector<ExecutionTimelinePoint>>(3)); }));
        std::vector<ExecutionTimelinePoint> completed{{10,1},{20,1}};
        CHECK(bounded.RetireCompleted(completed) == 0);
        CHECK(!lifetime.expired());
        cleanup->Drain();
        CHECK(!sourceWeak.expired());
        completed[0].value = 2;
        CHECK(bounded.RetireCompleted(completed) == 1);
        bounded.TakeRetiredGarbage().clear();
        CHECK(lifetime.expired());
        CHECK(bounded.InFlight() == 0);
        cleanup->Drain();
        CHECK(!sourceWeak.expired()); // GPU completion cannot revoke a replayable CPU packet.
        reusable = {};
        cleanup->Drain();
        CHECK(sourceWeak.expired());
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
        submission.TakeRetiredGarbage().clear();
        CHECK(packetLease.expired());
        auto firstPacket = std::make_shared<Packet>();
        auto failedPacket = std::make_shared<Packet>(); failedPacket->succeeds = false;
        failedPacket->queueSlot = 1;
        auto lastPacket = std::make_shared<Packet>();
        auto failureInput = std::make_shared<GraphCompileInput>(*bundle->input);
        auto failureGraph = workspace.Compile(failureInput, cancelled);
        auto failureBundle = std::make_shared<CompiledGraphBundle>(CompiledGraphBundle{
            bundle->sequence, std::move(failureGraph), failureInput});
        CHECK(rejects([&] { submission.SubmitPrepared(failureBundle,
            std::vector<std::vector<ExecutionTimelinePoint>>(3), {firstPacket, failedPacket, lastPacket}); }));
        CHECK(submission.Failed());
        CHECK(submission.Failure()->batch == 1);
        CHECK(submission.Failure()->receipt.state == SubmissionState::SubmittedWithoutSignal);
        CHECK(firstPacket->calls == 1 && failedPacket->calls == 1 && lastPacket->calls == 0);
        CHECK(submission.Submitted()[0].value == 3 && submission.Submitted()[1].value == 1);
        failureInput.reset(); failureBundle.reset();
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
        .regions = std::make_shared<const std::vector<PreparedStateRegion>>(
            std::initializer_list<PreparedStateRegion>{{{0, 4, 0, 2}, {}}}),
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
    // Queue-family ownership travels with the pair: the producer (slot 0,
    // graphics) releases to the consumer's family (slot 1, compute) and the
    // consumer acquires from graphics.
    CHECK(crossQueueBarriers.batches[crossQueueStep->previousBatch]
        .afterPass[producerPosition].textures.front().queueOwnership == rhi::QueueOwnership::Release);
    CHECK(crossQueueBarriers.batches[crossQueueStep->previousBatch]
        .afterPass[producerPosition].textures.front().ownershipPeer == rhi::QueueKind::Compute);
    CHECK(crossQueueBarriers.batches[crossQueueStep->batch]
        .beforePass[consumerPosition].textures.front().queueOwnership == rhi::QueueOwnership::Acquire);
    CHECK(crossQueueBarriers.batches[crossQueueStep->batch]
        .beforePass[consumerPosition].textures.front().ownershipPeer == rhi::QueueKind::Graphics);
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
    std::puts("Fresh compiler ownership, oracle, ordering and state tests passed.");
}
