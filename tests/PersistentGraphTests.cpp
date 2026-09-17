#include "Render/RenderGraph/PersistentGraph.h"
#include "Render/RenderGraph/GraphReplay.h"
#include "Render/RenderGraph/ExperimentalRhiExecution.h"
#include "GraphReplayRunner.h"
#include <cstdio>
#include <stdexcept>
#include <sstream>
#include <random>
#include <fstream>
#include <algorithm>

using namespace org::persistent;
using namespace org::experimental;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Failure at %d: %s\n", __LINE__, #x); return 1; } } while (false)
template<class F> bool Rejects(F&& fn) { try { fn(); } catch (const std::exception&) { return true; } return false; }
BindingVersion Binding(uint64_t id) {
    BindingVersion b;
    b.identity = id; b.backingRevision = id; b.shape = {1,1,false};
    b.owner = std::make_shared<const uint64_t>(id);
    return b;
}
struct ManualTasks final : org::runtime::ITaskService {
    struct Scope final : org::runtime::ITaskScope {
        ManualTasks& owner;
        explicit Scope(ManualTasks& value) : owner(value) {}
        void Cancel() noexcept override {}
        void Wait() override { owner.RunAll(); }
        void CancelAndWait() override { Wait(); }
    };
    bool reject = false;
    std::vector<std::function<void()>> jobs;
    void Run(size_t index) { auto fn = std::move(jobs.at(index)); if (fn) fn(); }
    void RunAll() { for (size_t i = 0; i < jobs.size(); ++i) Run(i); }
    void ParallelFor(std::string_view,size_t count,std::function<void(size_t)> fn) override {
        for (size_t i = 0; i < count; ++i) fn(i);
    }
    void ParallelForLimited(std::string_view name,size_t count,size_t,std::function<void(size_t)> fn) override {
        ParallelFor(name,count,std::move(fn));
    }
    std::shared_ptr<org::runtime::ITaskScope> CreateScope(std::string_view) override { return std::make_shared<Scope>(*this); }
    bool Submit(const std::shared_ptr<org::runtime::ITaskScope>&,org::runtime::TaskPriority,
        std::string_view,std::function<void()>&& fn) override {
        if (reject) return false;
        jobs.push_back(std::move(fn)); return true;
    }
    bool ScheduleAfter(const std::shared_ptr<org::runtime::ITaskScope>& scope,std::chrono::steady_clock::duration,
        org::runtime::TaskPriority priority,std::string_view name,std::function<void()>&& fn) override {
        return Submit(scope,priority,name,std::move(fn));
    }
};

// Independent small reference check: each conflicting authored access must
// precede its later user in execution order. No compiler helpers are called.
bool CheckConflicts(const CompiledGraph& graph) {
    if (!graph.structure) return false;
    std::vector<uint32_t> position(graph.structure->passes.size(), UINT32_MAX);
    uint32_t n = 0;
    for (const auto& batch : graph.batches)
        for (auto pass : batch.passes) {
            if (pass >= position.size() || position[pass] != UINT32_MAX) return false;
            position[pass] = n++;
        }
    if (n != position.size()) return false;
    const auto count = position.size();
    std::vector<uint8_t> happensBefore(count * count);
    std::vector<uint8_t> authored(count * count);
    for (const auto* edges : {&graph.structure->explicitEdges,&graph.structure->placementEdges})
        for (auto [from,to] : *edges) {
            if (from >= count || to >= count) return false;
            authored[from * count + to] = 1;
        }
    for (size_t k = 0; k < count; ++k)
        for (size_t i = 0; i < count; ++i)
            for (size_t j = 0; j < count; ++j)
                if (authored[i * count + k] && authored[k * count + j]) authored[i * count + j] = 1;
    std::vector<uint32_t> lastOnQueue(graph.structure->queues.size(),UINT32_MAX);
    for (const auto& batch : graph.batches) {
        if (batch.queue >= lastOnQueue.size()) return false;
        const auto& queue = graph.structure->queues[batch.queue];
        if (!queue.active) return false;
        auto& previous = lastOnQueue[batch.queue];
        for (auto pass : batch.passes) {
            const auto& declaration = graph.structure->passes[pass];
            // Pass backend affinity and queue backend-instance identity are
            // different domains; authored compatibility is the numeric contract.
            if (std::find(declaration.compatibleQueueSlots.begin(),declaration.compatibleQueueSlots.end(),batch.queue)
                    == declaration.compatibleQueueSlots.end()) return false;
            if (previous != UINT32_MAX) happensBefore[previous * count + pass] = 1;
            previous = pass;
        }
    }
    for (const auto& wait : graph.relativeWaits) {
        if (wait.producerBatch >= graph.batches.size() || wait.consumerBatch >= graph.batches.size()) return false;
        const auto& producer = graph.batches[wait.producerBatch];
        const auto& consumer = graph.batches[wait.consumerBatch];
        if (producer.passes.empty() || consumer.passes.empty()) return false;
        happensBefore[producer.passes.back() * count + consumer.passes.front()] = 1;
    }
    for (size_t k = 0; k < count; ++k)
        for (size_t i = 0; i < count; ++i)
            for (size_t j = 0; j < count; ++j)
                if (happensBefore[i * count + k] && happensBefore[k * count + j]) happensBefore[i * count + j] = 1;
    for (auto [from,to] : graph.structure->explicitEdges)
        if (from >= count || to >= count || !happensBefore[from * count + to]) return false;
    for (auto [from,to] : graph.structure->placementEdges)
        if (from >= count || to >= count || !happensBefore[from * count + to]) return false;
    for (uint32_t a = 0; a < position.size(); ++a)
        for (uint32_t b = a + 1; b < position.size(); ++b)
            for (const auto& x : graph.structure->passes[a].accesses)
                for (const auto& y : graph.structure->passes[b].accesses)
                    if (x.resourceIndex == y.resourceIndex && (x.write || y.write)
                        && !happensBefore[a * count + b]
                        && !(authored[b * count + a] && happensBefore[b * count + a])) return false;
    return true;
}
// Test-only full-cell simulation, deliberately independent of production range
// partitioning, dense placement maps and symbolic-state validation helpers.
bool CheckStates(const CompiledGraph& graph) {
    if (!graph.structure || !graph.states.complete) return false;
    const auto& structure = *graph.structure;
    if (structure.resourceShapes.size() != structure.resourceIDs.size()) return false;
    struct Cell {
        CompileResourceState state;
        uint32_t pass = UINT32_MAX, batch = UINT32_MAX;
        bool terminal = false;
    };
    std::vector<std::vector<Cell>> cells;
    for (const auto shape : structure.resourceShapes) {
        if (uint64_t{shape.mips} * shape.slices > 2'000'000) return false;
        cells.emplace_back(size_t{shape.mips} * shape.slices);
    }
    auto visit = [&](uint32_t resource, CompileRange range, auto&& fn) {
        if (resource >= cells.size()) return false;
        const auto shape = structure.resourceShapes[resource];
        if (!range.mips || !range.slices || range.mip >= shape.mips || range.slice >= shape.slices
            || range.mips > shape.mips - range.mip || range.slices > shape.slices - range.slice) return false;
        for (uint32_t slice = range.slice; slice < range.slice + range.slices; ++slice)
            for (uint32_t mip = range.mip; mip < range.mip + range.mips; ++mip)
                if (!fn(cells[resource][size_t{slice} * shape.mips + mip])) return false;
        return true;
    };
    auto normalized = [&](uint32_t resource, CompileResourceState state) {
        if (!structure.resourceShapes[resource].hasLayout) state.layout = 0;
        return state;
    };
    std::vector<bool> consumed(graph.states.steps.size());
    for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
        for (const auto passIndex : graph.batches[batch].passes) {
            if (passIndex >= structure.passes.size()) return false;
            for (size_t i = 0; i < graph.states.steps.size(); ++i) {
                const auto& step = graph.states.steps[i];
                if (step.pass != passIndex || step.batch != batch) continue;
                if (!visit(step.resource, step.range, [&](Cell& cell) {
                    if (cell.pass != step.previousPass || cell.batch != step.previousBatch
                        || cell.state != step.before || cell.pass == passIndex) return false;
                    cell = {step.after,passIndex,batch,false};
                    return true;
                })) return false;
                consumed[i] = true;
            }
            const auto& pass = structure.passes[passIndex];
            for (const auto& use : pass.entryStates) {
                if (!visit(use.resource,use.range,[&](Cell& cell) {
                    return cell.pass == passIndex && cell.batch == batch
                        && cell.state == normalized(use.resource,use.state);
                })) return false;
            }
            for (const auto& use : pass.exitStates) {
                if (!visit(use.resource,use.range,[&](Cell& cell) {
                    if (cell.pass != passIndex || cell.batch != batch) return false;
                    cell.state = normalized(use.resource,use.state);
                    return true;
                })) return false;
            }
        }
    }
    if (std::find(consumed.begin(),consumed.end(),false) != consumed.end()) return false;
    for (const auto& final : graph.states.finalStates) {
        if (!visit(final.resource,final.range,[&](Cell& cell) {
            if (cell.terminal || cell.batch != final.batch || cell.state != final.state) return false;
            cell.terminal = true;
            return true;
        })) return false;
    }
    for (const auto& resource : cells)
        for (const auto& cell : resource)
            if ((cell.pass != UINT32_MAX) != cell.terminal) return false;
    return true;
}
int main() {
    {
        GraphProgram program; CompileWorkspace workspace; std::atomic_bool cancelled{false};
        auto edit = program.BeginEdit(); edit.SetQueues({{0,true},{0,true}});
        std::array<ResourceSlotId,2> slots;
        for (uint32_t i = 0; i < 2; ++i) {
            auto binding = Binding(80+i); binding.identity = (uint64_t{1} << 32) | (80+i);
            auto state = std::make_shared<PreparedBackingState>();
            state->graphResourceID = i+1; state->resource = {80+i,1}; state->shape = binding.shape;
            state->regions = std::make_shared<const std::vector<PreparedStateRegion>>(); binding.admission = state;
            slots[i] = edit.AddResource(binding.shape,binding);
        }
        CompilePass queue0; queue0.compatibleQueueSlots = {0}; queue0.preferredQueueSlot = 0;
        CompilePass queue1; queue1.compatibleQueueSlots = {1}; queue1.preferredQueueSlot = 1;
        const auto first = edit.AddPass(queue0), second = edit.AddPass(queue1);
        edit.Declare(first,slots[0],{}, {32,0,1,true}); edit.Declare(second,slots[1],{}, {32,0,1,true});
        edit.DeclareDependency(first,slots[1],false);
        edit.AddOrdering(first,second);
        auto selected = edit.Build(workspace,cancelled); CHECK(program.Install(selected));
        SynchronousAdmission admission;
        const std::array<ExecutionTimelinePoint,2> queues{{{12,0},{13,0}}};
        FrameIncomingState report{slots[1],{81,1},std::make_shared<const std::vector<PreparedStateRegion>>(
            std::initializer_list<PreparedStateRegion>{{{}, {32,0,1,true}}}),{90,5},1};
        auto partial = admission.Prepare(selected,queues,{},std::span{&report,1});
        CHECK(partial.incomingEffects.size() == 1 && partial.incomingEffects[0].firstBatch == 1);
        GraphExecutionTimeline receipt; receipt.batches.resize(2); receipt.batches[0].signal = {12,1};
        admission.Commit(partial,receipt,1);
        CHECK(admission.IncomingRevisionCount() == 0);
        auto retry = admission.Prepare(selected,queues,{},std::span{&report,1});
        CHECK(retry.incomingEffects.size() == 1 && retry.backings[1].authoritativeIncoming);
        admission.Abandon(retry);
    }
    {
        GraphProgram program; CompileWorkspace workspace; std::atomic_bool cancelled{false};
        auto edit = program.BeginEdit();
        auto binding = Binding(71); binding.identity = (uint64_t{1} << 32) | 71; binding.shape = {2,1,true};
        PreparedBackingState backing; backing.graphResourceID = 1; backing.resource = {71,1}; backing.shape = binding.shape;
        const CompileResourceState oldState{static_cast<uint64_t>(rhi::ResourceAccessType::CopySource),
            static_cast<uint64_t>(rhi::ResourceLayout::CopySource),static_cast<uint64_t>(rhi::ResourceSyncState::Copy),false};
        const CompileResourceState externalState{static_cast<uint64_t>(rhi::ResourceAccessType::CopyDest),
            static_cast<uint64_t>(rhi::ResourceLayout::CopyDest),static_cast<uint64_t>(rhi::ResourceSyncState::Copy),true};
        const CompileResourceState consumerState{static_cast<uint64_t>(rhi::ResourceAccessType::UnorderedAccess),
            static_cast<uint64_t>(rhi::ResourceLayout::UnorderedAccess),static_cast<uint64_t>(rhi::ResourceSyncState::ComputeShading),true};
        backing.regions = std::make_shared<const std::vector<PreparedStateRegion>>(
            std::initializer_list<PreparedStateRegion>{{{0,2,0,1},oldState}});
        binding.admission = std::make_shared<const PreparedBackingState>(backing);
        const auto slot = edit.AddResource(binding.shape,binding);
        const auto first = edit.AddPass({}); edit.Declare(first,slot,{0,1,0,1},consumerState);
        auto selected = edit.Build(workspace,cancelled); CHECK(program.Install(selected));
        BackingStateAdmissionLedger ledger;
        const auto firstPlan = ledger.Prepare(*selected->executable->graph,std::span{&backing,1});
        for (const auto& batch : firstPlan.batches) ledger.CommitBatch(batch);
        auto reported = backing; reported.authoritativeIncoming = true;
        reported.regions = std::make_shared<const std::vector<PreparedStateRegion>>(
            std::initializer_list<PreparedStateRegion>{{{0,2,0,1},externalState}});
        SynchronousAdmission persistent;
        const std::array<ExecutionTimelinePoint,1> queues{{{12,0}}};
        FrameIncomingState report{slot,backing.resource,reported.regions,{90,5}};
        auto frame = persistent.Prepare(selected,queues,{},std::span{&report,1});
        CHECK(frame.publication == selected && selected->bindings.At(slot).admission->regions == backing.regions);
        CHECK(frame.backings[0].authoritativeIncoming && frame.backings[0].regions == reported.regions);
        CHECK(frame.incomingWaits[0] == std::vector<ExecutionTimelinePoint>{ExecutionTimelinePoint(90,5)});
        CHECK(frame.barriers.batches[0].beforePass[0].textures[0].beforeLayout == rhi::ResourceLayout::CopyDest);
        persistent.Abandon(frame);
        for (uint32_t failure = 0; failure < 4; ++failure) {
            auto invalid = report;
            if (failure == 0) invalid.backing.index = 72;
            if (failure == 1) ++invalid.resource.generation;
            if (failure == 2) invalid.producerCompletion.value = 0;
            if (failure == 3) invalid.regions.reset();
            CHECK(Rejects([&] { persistent.Prepare(selected,queues,{},std::span{&invalid,1}); }));
        }
        const std::array<FrameIncomingState,2> duplicate{{report,report}};
        CHECK(Rejects([&] { persistent.Prepare(selected,queues,{},duplicate); }));
        auto normal = persistent.Prepare(selected,queues);
        CHECK(!normal.backings[0].authoritativeIncoming && normal.backings[0].regions == backing.regions);
        persistent.Abandon(normal);
        auto invalidPublication = program.BeginEdit(); auto invalidBinding = binding;
        invalidBinding.admission = std::make_shared<const PreparedBackingState>(reported);
        CHECK(Rejects([&] { invalidPublication.ReplaceBinding(slot,invalidBinding); }));
        const auto incoming = ledger.Prepare(*selected->executable->graph,std::span{&reported,1});
        CHECK(incoming.batches[0].beforePass[0].textures[0].beforeLayout == rhi::ResourceLayout::CopyDest);
        CHECK(incoming.batches[0].authoritativeSeeds.size() == 1);
        const auto abandoned = ledger.Prepare(*selected->executable->graph,std::span{&backing,1});
        CHECK(abandoned.batches[0].beforePass[0].textures[0].beforeLayout == rhi::ResourceLayout::UnorderedAccess);
        for (uint32_t failure = 0; failure < 4; ++failure) {
            auto invalid = reported;
            auto regions = std::make_shared<std::vector<PreparedStateRegion>>(*reported.regions);
            if (failure == 0) regions->clear();
            if (failure == 1) regions->front().range.mips = 1;
            if (failure == 2) regions->push_back(regions->front());
            if (failure == 3) regions->front().range.mip = 1;
            invalid.regions = regions;
            CHECK(Rejects([&] { ledger.Prepare(*selected->executable->graph,std::span{&invalid,1}); }));
        }
        for (const auto& batch : incoming.batches) ledger.CommitBatch(batch);
        auto addConsumer = program.BeginEdit();
        addConsumer.SetQueues({{0,true},{0,true}});
        CompilePass secondQueue; secondQueue.compatibleQueueSlots = {1}; secondQueue.preferredQueueSlot = 1;
        const auto second = addConsumer.AddPass(secondQueue); addConsumer.Declare(second,slot,{1,1,0,1},consumerState);
        auto expanded = addConsumer.Build(workspace,cancelled); CHECK(program.Install(expanded));
        const auto later = ledger.Prepare(*expanded->executable->graph,std::span{&backing,1});
        const auto& graph = *expanded->executable->graph;
        CHECK(later.batches.at(graph.batchByPass.at(1)).beforePass.at(graph.positionByPass.at(1)).textures[0].beforeLayout
            == rhi::ResourceLayout::CopyDest);
        CHECK(graph.batches.size() == 2);
        const auto bothQueues = ledger.Prepare(graph,std::span{&reported,1});
        size_t authoritativeSeedCount = 0;
        for (const auto& batch : bothQueues.batches) authoritativeSeedCount += batch.authoritativeSeeds.size();
        CHECK(authoritativeSeedCount == 1);
        for (const auto& batch : bothQueues.batches) ledger.CommitBatch(batch);
        const auto committed = ledger.Prepare(graph,std::span{&backing,1});
        CHECK(committed.batches.at(graph.batchByPass.at(0)).beforePass.at(graph.positionByPass.at(0)).textures[0].beforeLayout
            == rhi::ResourceLayout::UnorderedAccess);
        const std::array<ExecutionTimelinePoint,2> twoQueues{{{12,0},{13,0}}};
        auto acrossQueues = persistent.Prepare(expanded,twoQueues,{},std::span{&report,1});
        CHECK(acrossQueues.incomingWaits.size() == 2 && acrossQueues.incomingWaits[0].size() == 1
            && acrossQueues.incomingWaits[1].size() == 1);
        persistent.Abandon(acrossQueues);
        auto consume = persistent.Prepare(expanded,twoQueues,{},std::span{&report,1});
        GraphExecutionTimeline receipt; receipt.batches.resize(graph.batches.size());
        for (size_t i = 0; i < graph.batches.size(); ++i) {
            receipt.batches[i].signal = {twoQueues[graph.batches[i].queue].timeline,1};
            receipt.batches[i].waits = consume.incomingWaits[i];
        }
        persistent.Commit(consume,receipt);
        CHECK(persistent.IncomingRevisionCount() == 1);
        auto repeated = persistent.Prepare(expanded,twoQueues,{},std::span{&report,1});
        CHECK(repeated.incomingEffects.empty() && !repeated.backings[0].authoritativeIncoming);
        CHECK(repeated.barriers.batches.at(graph.batchByPass.at(0)).beforePass.at(graph.positionByPass.at(0)).textures[0].beforeLayout
            == rhi::ResourceLayout::UnorderedAccess);
        persistent.Abandon(repeated);
        auto inconsistent = report; ++inconsistent.producerCompletion.value;
        CHECK(Rejects([&] { persistent.Prepare(expanded,twoQueues,{},std::span{&inconsistent,1}); }));
        inconsistent = report; ++inconsistent.revision;
        CHECK(Rejects([&] { persistent.Prepare(expanded,twoQueues,{},std::span{&inconsistent,1}); }));
        ++report.revision; ++report.producerCompletion.value;
        auto retry = persistent.Prepare(expanded,twoQueues,{},std::span{&report,1});
        CHECK(retry.incomingEffects.size() == 1 && retry.backings[0].authoritativeIncoming);
        persistent.Commit(retry,GraphExecutionTimeline{.batches = std::vector<ExecutionBatchTimeline>(graph.batches.size())},0);
        auto unsubmitted = persistent.Prepare(expanded,twoQueues,{},std::span{&report,1});
        CHECK(unsubmitted.incomingEffects.size() == 1); persistent.Abandon(unsubmitted);
        auto consumedPrefix = persistent.Prepare(expanded,twoQueues,{},std::span{&report,1});
        receipt.batches[0].signal.value = 2; receipt.batches[1].signal = {};
        persistent.Commit(consumedPrefix,receipt,1);
        auto afterPrefix = persistent.Prepare(expanded,twoQueues,{},std::span{&report,1});
        CHECK(afterPrefix.incomingEffects.empty()); persistent.Abandon(afterPrefix);
        auto regressed = report; --regressed.revision;
        CHECK(Rejects([&] { persistent.Prepare(expanded,twoQueues,{},std::span{&regressed,1}); }));
    }
    {
        GraphProgram program; CompileWorkspace workspace; std::atomic_bool cancelled{false};
        auto edit = program.BeginEdit(); edit.SetQueues({{0,true},{0,true}});
        const auto removed = edit.AddPass({});
        auto binding = Binding(71);
        binding.identity = (uint64_t{1} << 32) | 71;
        auto state = std::make_shared<PreparedBackingState>();
        state->resource = {71,1}; state->graphResourceID = 1; state->shape = binding.shape;
        state->regions = std::make_shared<const std::vector<PreparedStateRegion>>(); binding.admission = state;
        const auto slot = edit.AddResource(binding.shape,binding);
        CompilePass write; write.compatibleQueueSlots = {0}; write.preferredQueueSlot = 0;
        CompilePass read; read.compatibleQueueSlots = {1}; read.preferredQueueSlot = 1;
        const auto writer = edit.AddPass(write), reader = edit.AddPass(read);
        edit.RemovePass(removed);
        edit.Declare(writer,slot,{}, {32,0,1,true}); edit.Declare(reader,slot,{}, {256,0,1,false});
        auto selected = edit.Build(workspace,cancelled); CHECK(program.Install(selected));
        SynchronousAdmission admission;
        const std::array<ExecutionTimelinePoint,2> queues{{{12,0},{13,0}}};
        const std::array<FrameProducerWait,4> waits{{{reader,{90,4}},{reader,{90,9}},
            {reader,{91,3}},{writer,{90,5}}}};
        auto frame = admission.Prepare(selected,queues,waits);
        const auto writerBatch = selected->executable->executionLayout->placements.at(writer.index).batch;
        const auto readerBatch = selected->executable->executionLayout->placements.at(reader.index).batch;
        CHECK(writerBatch != readerBatch);
        CHECK(frame.incomingWaits.at(writerBatch).size() == 1 && frame.incomingWaits.at(writerBatch)[0] == ExecutionTimelinePoint(90,5));
        CHECK(frame.incomingWaits.at(readerBatch).size() == 2);
        CHECK(frame.incomingWaits.at(readerBatch)[0] == ExecutionTimelinePoint(90,9));
        CHECK(frame.incomingWaits.at(readerBatch)[1] == ExecutionTimelinePoint(91,3));
        ExecutionTimelineAdmission timeline(std::vector<ExecutionTimelinePoint>(queues.begin(),queues.end()));
        const auto submission = timeline.Prepare(selected->executable->executionLayout->bundle,frame.incomingWaits,{selected});
        const auto& submittedWaits = submission->batches.at(readerBatch).waits;
        CHECK(std::ranges::find(submittedWaits,ExecutionTimelinePoint(90,9)) != submittedWaits.end());
        CHECK(std::ranges::find(submittedWaits,ExecutionTimelinePoint(91,3)) != submittedWaits.end());
        CHECK(std::ranges::find(submittedWaits,submission->batches.at(writerBatch).signal) != submittedWaits.end());
        timeline.Fail(submission->submission);
        CHECK(program.Select() == selected); admission.Abandon(frame);
        for (const auto invalid : {FrameProducerWait{reader,{0,1}},FrameProducerWait{reader,{90,0}},
            FrameProducerWait{{reader.index,reader.generation+1},{90,1}},
            FrameProducerWait{{UINT32_MAX,1},{90,1}},FrameProducerWait{removed,{90,1}},FrameProducerWait{reader,{12,1}}}) {
            CHECK(Rejects([&] { admission.Prepare(selected,queues,std::span{&invalid,1}); }));
        }
        auto noWaits = admission.Prepare(selected,queues);
        CHECK(std::ranges::all_of(noWaits.incomingWaits,[](const auto& batch) { return batch.empty(); }));
        admission.Abandon(noWaits);
        const std::array<ExecutionTimelinePoint,2> advanced{{{12,8},{13,8}}};
        const std::array<FrameProducerWait,2> ordered{{{reader,{13,7}},{reader,{12,6}}}};
        auto queueOrdered = admission.Prepare(selected,advanced,ordered);
        CHECK(queueOrdered.incomingWaits.at(readerBatch).size() == 1);
        CHECK(queueOrdered.incomingWaits.at(readerBatch)[0] == ExecutionTimelinePoint(12,6));
        admission.Abandon(queueOrdered);
    }
    {
        GraphProgram program; CompileWorkspace workspace; std::atomic_bool cancelled{false};
        auto initial = program.BeginEdit(); initial.SetQueues({{0,true},{0,true}});
        auto version = Binding(300);
        version.identity = (uint64_t{1} << 32) | 71;
        auto state = std::make_shared<PreparedBackingState>();
        state->resource = {71,1}; state->graphResourceID = 1; state->shape = version.shape;
        state->regions = std::make_shared<const std::vector<PreparedStateRegion>>();
        version.admission = state;
        const auto a = initial.AddResource(version.shape,version);
        auto second = version; auto secondState = std::make_shared<PreparedBackingState>(*state);
        secondState->graphResourceID = 2; second.admission = secondState;
        const auto b = initial.AddResource(second.shape,second);
        const auto writer = initial.AddPass({});
        CompilePass readPass; readPass.compatibleQueueSlots = {1}; readPass.preferredQueueSlot = 1;
        const auto reader = initial.AddPass(readPass);
        const auto writeToken = initial.Declare(writer,a,{}, {32,0,1,true});
        const auto readToken = initial.Declare(reader,b,{}, {256,0,1,false});
        auto selected = initial.Build(workspace,cancelled); CHECK(program.Install(selected));
        CHECK(selected->logical->declarations.resourceIDs.size() == 2);
        CHECK(selected->executable->graph->structure->resourceIDs.size() == 1);
        CHECK(selected->executable->resourceSlots.size() == 1 && selected->executable->resourceSlots[0] == a);
        CHECK(CheckStates(*selected->executable->graph));
        CHECK(selected->executable->graph->batches.size() == 2);
        CHECK(selected->Resolve(writeToken).identity == selected->Resolve(readToken).identity);
        SynchronousAdmission admission;
        const std::array<ExecutionTimelinePoint,2> queues{{{12,0},{13,0}}};
        auto held = admission.Prepare(selected,queues);
        CHECK(held.backings.size() == 1 && held.backings[0].resource.index == 71);
        SynchronousAdmission sharedIncoming;
        FrameIncomingState sharedReport{b,state->resource,std::make_shared<const std::vector<PreparedStateRegion>>(
            std::initializer_list<PreparedStateRegion>{{{}, {static_cast<uint64_t>(rhi::ResourceAccessType::CopyDest),0,
                static_cast<uint64_t>(rhi::ResourceSyncState::Copy),true}}}),{90,7}};
        auto sharedFrame = sharedIncoming.Prepare(selected,queues,{},std::span{&sharedReport,1});
        CHECK(sharedFrame.backings.size() == 1 && sharedFrame.backings[0].authoritativeIncoming);
        CHECK(sharedFrame.incomingWaits.size() == 2 && sharedFrame.incomingWaits[0][0] == ExecutionTimelinePoint(90,7)
            && sharedFrame.incomingWaits[1][0] == ExecutionTimelinePoint(90,7));
        sharedIncoming.Abandon(sharedFrame);
        auto otherSharedReport = sharedReport; otherSharedReport.resource = a;
        const std::array<FrameIncomingState,2> repeatedPhysical{{sharedReport,otherSharedReport}};
        CHECK(Rejects([&] { sharedIncoming.Prepare(selected,queues,{},repeatedPhysical); }));
        auto descriptors = program.BeginEdit(); auto descriptorOnly = second; ++descriptorOnly.descriptorRevision;
        descriptors.ReplaceBinding(b,descriptorOnly);
        auto descriptorReady = descriptors.Build(workspace,cancelled); CHECK(program.Install(descriptorReady));
        CHECK(descriptorReady->executable == selected->executable);
        auto incompatible = program.BeginEdit(); auto badContent = descriptorOnly; ++badContent.contentRevision;
        incompatible.ReplaceBinding(b,badContent);
        CHECK(Rejects([&] { incompatible.Build(workspace,cancelled); }));
        CHECK(program.Select() == descriptorReady);
        auto rotate = program.BeginEdit();
        auto rotatedA = version; rotatedA.identity = (uint64_t{1} << 32) | 72; ++rotatedA.backingRevision;
        auto rotatedState = std::make_shared<PreparedBackingState>(*state); rotatedState->resource = {72,1};
        rotatedA.admission = rotatedState; rotatedA.owner = std::make_shared<const uint32_t>(72);
        auto rotatedB = rotatedA; auto rotatedSecondState = std::make_shared<PreparedBackingState>(*rotatedState);
        rotatedSecondState->graphResourceID = 2; rotatedB.admission = rotatedSecondState;
        rotate.ReplaceBinding(a,rotatedA); rotate.ReplaceBinding(b,rotatedB);
        auto rotated = rotate.Build(workspace,cancelled); CHECK(program.Install(rotated));
        CHECK(rotated->executable == selected->executable);
        CHECK(rotated->Resolve(writeToken).identity == rotated->Resolve(readToken).identity);
        CHECK(selected->Resolve(readToken).identity != rotated->Resolve(readToken).identity);
        CHECK(held.backings[0].resource.index == 71);
        admission.Abandon(held);
        auto split = program.BeginEdit(); split.ReplaceBinding(b,second);
        auto splitReady = split.Build(workspace,cancelled); CHECK(program.Install(splitReady));
        CHECK(splitReady->executable != rotated->executable && splitReady->executable->resourceSlots.size() == 2);
        CHECK(CheckStates(*splitReady->executable->graph));
        auto merge = program.BeginEdit(); merge.ReplaceBinding(b,rotatedB);
        auto merged = merge.Build(workspace,cancelled); CHECK(program.Install(merged));
        CHECK(merged->executable->resourceSlots.size() == 1 && CheckStates(*merged->executable->graph));
        auto grouped = program.BeginEdit();
        const auto producerGroup = grouped.AddGroup(version.shape,{a});
        const auto consumerGroup = grouped.AddGroup(version.shape,{b});
        grouped.DeclareGroupAccess(writer,producerGroup,{32,0,1,true},0);
        grouped.DeclareGroupAccess(reader,consumerGroup,{256,0,1,false},0);
        CHECK(Rejects([&] { grouped.Build(workspace,cancelled); }));
        grouped.RemoveGroupAccess(reader,consumerGroup);
        grouped.DeclareGroupAccess(reader,consumerGroup,{256,0,1,false},1);
        CHECK(CheckStates(*grouped.Build(workspace,cancelled)->executable->graph));
        auto repeated = program.BeginEdit(); repeated.Declare(writer,b,{}, {32,0,1,true});
        CHECK(CheckStates(*repeated.Build(workspace,cancelled)->executable->graph));
        auto conflicting = program.BeginEdit(); conflicting.Declare(writer,b,{}, {256,0,1,false});
        CHECK(Rejects([&] { conflicting.Build(workspace,cancelled); }));
        auto incoming = program.BeginEdit(); auto differentState = rotatedB;
        auto changedInitial = std::make_shared<PreparedBackingState>(*rotatedSecondState);
        changedInitial->regions = std::make_shared<const std::vector<PreparedStateRegion>>(
            std::vector<PreparedStateRegion>{{{}, {32,0,1,true}}});
        differentState.admission = changedInitial; incoming.ReplaceBinding(b,differentState);
        CHECK(Rejects([&] { incoming.Build(workspace,cancelled); }));
        auto retired = program.BeginEdit(); retired.RemovePass(writer); retired.RemoveResource(a);
        auto remaining = retired.Build(workspace,cancelled); CHECK(program.Install(remaining));
        CHECK(remaining->executable->resourceSlots.size() == 1 && remaining->executable->resourceSlots[0] == b);
        auto next = admission.Prepare(remaining,queues);
        CHECK(next.backings.size() == 1 && next.backings[0].graphResourceID == 2);
        admission.Abandon(next);
    }
    {
        GraphReplaySequence sequence;
        sequence.initial.resourceIDs = {10}; sequence.initial.resourceShapes = {{1,1,false}};
        sequence.initial.queues = {{0,true},{0,true}};
        CompilePass writer, reader;
        writer.entryStates = {{0,{}, {static_cast<uint64_t>(rhi::ResourceAccessType::UnorderedAccess),0,1,true}}};
        writer.accesses = {{0,true}};
        reader.entryStates = {{0,{}, {static_cast<uint64_t>(rhi::ResourceAccessType::ShaderResource),0,1,false}}};
        reader.accesses = {{0,false}}; reader.compatibleQueueSlots = {1}; reader.preferredQueueSlot = 1;
        sequence.initial.passes = {writer,reader};
        sequence.bindingEdits = {{7,2,99,3,4,5,0,1,{1,1,false}}};
        sequence.submissions = {{6,{{12,17},{13,19}},{{14,23}},2}};
        sequence.submissions[0].producerWaits = {{1,1,{91,7}}};
        sequence.submissions[0].incomingStates = {{0,1,1,1,{90,5},
            {{{}, {static_cast<uint64_t>(rhi::ResourceAccessType::CopyDest),0,
                static_cast<uint64_t>(rhi::ResourceSyncState::Copy),true}}}}};
        sequence.completions = {{8,12,17},{9,13,19},{10,14,23}};
        std::stringstream stream; WriteGraphReplaySequence(stream,sequence);
        auto decoded = ReadGraphReplaySequence(stream);
        CHECK(decoded.initial.resourceIDs == sequence.initial.resourceIDs);
        CHECK(decoded.bindingEdits.size() == 1 && decoded.bindingEdits[0].allocationIdentity == 99);
        CHECK(decoded.bindingEdits[0].descriptorRevision == 4 && decoded.bindingEdits[0].slotGeneration == 1);
        CHECK(decoded.completions.size() == 3 && decoded.completions[1].timeline == 13 && decoded.completions[1].value == 19);
        CHECK(decoded.submissions.size() == 1 && decoded.submissions[0].signaledBatches == 2);
        CHECK(decoded.submissions[0].tailCompletions.size() == 1 && decoded.submissions[0].tailCompletions[0].value == 23);
        CHECK(decoded.submissions[0].producerWaits.size() == 1 && decoded.submissions[0].producerWaits[0].completion.value == 7);
        CHECK(decoded.submissions[0].incomingStates.size() == 1 && decoded.submissions[0].incomingStates[0].revision == 1
            && decoded.submissions[0].incomingStates[0].regions[0].state.write);
        const auto replayed = org::test::RunGraphReplaySequence(decoded);
        CHECK(replayed.submissions == 1 && replayed.bindingChanges == 1 && replayed.retiredFrames == 1);
        CHECK(replayed.peakRetainedFrames == 1 && replayed.admissionMs.size() == 1);
        CHECK(replayed.submissionWaits == 4); // Three incoming waits plus the reader's cross-queue dependency.
        auto missingSignal = decoded; missingSignal.submissions[0].batchSignals[0].value = 0;
        CHECK(Rejects([&] { org::test::RunGraphReplaySequence(missingSignal); }));
        auto wrongTimeline = decoded; wrongTimeline.submissions[0].batchSignals[1].timeline = 12;
        CHECK(Rejects([&] { org::test::RunGraphReplaySequence(wrongTimeline); }));
        auto wrongBacking = decoded; wrongBacking.submissions[0].incomingStates[0].allocationIdentity = 99;
        CHECK(Rejects([&] { org::test::RunGraphReplaySequence(wrongBacking); }));
        auto missingRegions = decoded; missingRegions.submissions[0].incomingStates[0].regions.clear();
        CHECK(Rejects([&] { org::test::RunGraphReplaySequence(missingRegions); }));
        auto staleWait = decoded; staleWait.submissions[0].producerWaits[0].generation = 2;
        CHECK(Rejects([&] { org::test::RunGraphReplaySequence(staleWait); }));
        auto invalidReport = decoded; invalidReport.submissions[0].incomingStates[0].revision = 0;
        std::stringstream invalidReportStream; WriteGraphReplaySequence(invalidReportStream,invalidReport);
        CHECK(Rejects([&] { ReadGraphReplaySequence(invalidReportStream); }));
        auto missingTail = decoded; missingTail.completions.pop_back();
        CHECK(Rejects([&] { org::test::RunGraphReplaySequence(missingTail); }));
        auto impossibleCompletion = decoded; impossibleCompletion.completions[0].value = 18;
        CHECK(Rejects([&] { org::test::RunGraphReplaySequence(impossibleCompletion); }));
        auto wrongRevision = decoded; wrongRevision.bindingEdits[0].publicationRevision = 3;
        CHECK(Rejects([&] { org::test::RunGraphReplaySequence(wrongRevision); }));
        auto unsorted = decoded; std::swap(unsorted.completions[0],unsorted.completions[1]);
        CHECK(Rejects([&] { org::test::RunGraphReplaySequence(unsorted); }));
        auto staleSlot = decoded; staleSlot.bindingEdits[0].slotGeneration = 2;
        CHECK(Rejects([&] { org::test::RunGraphReplaySequence(staleSlot); }));
        auto batchedEdits = decoded;
        batchedEdits.initial.resourceIDs.push_back(11); batchedEdits.initial.resourceShapes.push_back({1,1,false});
        auto secondEdit = batchedEdits.bindingEdits[0]; secondEdit.slot = 1; secondEdit.allocationIdentity = 100;
        batchedEdits.bindingEdits.push_back(secondEdit);
        CHECK(org::test::RunGraphReplaySequence(batchedEdits).bindingChanges == 2);
        std::ifstream checkedSequence(std::string(ORG_REPLAY_FIXTURE_DIR)+"/binding_rotation_tail.orgsequence");
        const auto checked = ReadGraphReplaySequence(checkedSequence);
        const auto checkedResult = org::test::RunGraphReplaySequence(checked);
        CHECK(checkedResult.submissions == 2 && checkedResult.retiredFrames == 2 && checkedResult.bindingChanges == 1);
        CHECK(checkedResult.digest == org::test::RunGraphReplaySequence(checked).digest);
        std::ifstream stateSequence(std::string(ORG_REPLAY_FIXTURE_DIR)+"/incoming_state_rotation.orgsequence");
        const auto stateResult = org::test::RunGraphReplaySequence(ReadGraphReplaySequence(stateSequence));
        CHECK(stateResult.submissions == 2 && stateResult.incomingReports == 2 && stateResult.incomingRevisions == 2
            && stateResult.producerWaits == 1 && stateResult.retiredFrames == 2);
        CHECK(stateResult.retiredBackings == 1 && stateResult.finalStateBackings == 1);
        GraphReplaySequence churn; churn.initial = decoded.initial;
        for (uint64_t i = 0; i < 128; ++i) {
            const auto allocation = i ? i+100 : 1;
            if (i) churn.bindingEdits.push_back({i,i+1,allocation,i+1,i+1,i+1,0,1,{1,1,false}});
            ReplaySubmission submission{i,{{12,i+1},{13,i+1}},{},2};
            submission.incomingStates = decoded.submissions[0].incomingStates;
            submission.incomingStates[0].allocationIdentity = allocation;
            submission.incomingStates[0].completion.value = i+1;
            churn.submissions.push_back(std::move(submission));
            churn.completions.push_back({i+1,12,i+1}); churn.completions.push_back({i+1,13,i+1});
        }
        const auto churnResult = org::test::RunGraphReplaySequence(churn);
        CHECK(churnResult.retiredBackings == 127 && churnResult.retiredFrames == 128
            && churnResult.peakStateBackings == 1 && churnResult.finalStateBackings == 1);
        std::stringstream legacy;
        legacy << "ORG_GRAPH_SEQUENCE 1\n"; WriteGraphReplay(legacy,sequence.initial); legacy << "0\n0\n";
        CHECK(ReadGraphReplaySequence(legacy).submissions.empty());
        auto invalidSubmission = sequence; invalidSubmission.submissions[0].signaledBatches = 3;
        std::stringstream invalidStream; WriteGraphReplaySequence(invalidStream,invalidSubmission);
        CHECK(Rejects([&] { ReadGraphReplaySequence(invalidStream); }));
        std::stringstream unsupported("ORG_GRAPH_SEQUENCE 4\n");
        CHECK(Rejects([&] { ReadGraphReplaySequence(unsupported); }));
        std::stringstream truncated(stream.str().substr(0,stream.str().size()-4));
        CHECK(Rejects([&] { ReadGraphReplaySequence(truncated); }));
        // Fixture-local allocation IDs map to synthetic native identities only
        // in this GPU-free adapter; the file contains no RHI handle encoding.
        GraphProgram program; CompileWorkspace workspace; std::atomic_bool cancelled{false};
        auto bootstrap = program.BeginEdit(); bootstrap.SetQueues(decoded.initial.queues);
        auto binding = Binding(10); binding.identity = (uint64_t{1} << 32) | 101;
        auto state = std::make_shared<PreparedBackingState>();
        state->resource = {101,1}; state->graphResourceID = 1; state->shape = binding.shape;
        state->regions = std::make_shared<const std::vector<PreparedStateRegion>>();
        binding.admission = state;
        const auto slot = bootstrap.AddResource(binding.shape,binding);
        for (auto pass : decoded.initial.passes) bootstrap.AddPass(std::move(pass));
        auto old = bootstrap.Build(workspace,cancelled); CHECK(program.Install(old));
        SynchronousAdmission admission;
        const std::array<ExecutionTimelinePoint,2> queues{{{12,0},{13,0}}};
        auto held = admission.Prepare(old,queues);
        const auto& event = decoded.bindingEdits[0];
        auto replacement = binding;
        replacement.identity = (uint64_t{1} << 32) | 102;
        replacement.owner = std::make_shared<const uint64_t>(event.allocationIdentity);
        replacement.backingRevision = event.backingRevision; replacement.descriptorRevision = event.descriptorRevision;
        replacement.contentRevision = event.contentRevision; replacement.shape = event.shape;
        auto changedState = std::make_shared<PreparedBackingState>(*state); changedState->resource = {102,1};
        replacement.admission = changedState;
        auto edit = program.BeginEdit(); edit.ReplaceBinding({event.slot,event.slotGeneration},replacement);
        auto selected = edit.Build(workspace,cancelled); CHECK(program.Install(selected));
        CHECK(selected->revision == event.publicationRevision && selected->executable == old->executable);
        CHECK(held.backings[0].resource.index == 101 && selected->bindings.At(slot).admission->resource.index == 102);
        GraphExecutionTimeline receipt; receipt.batches.resize(2);
        const auto& submitted = decoded.submissions[0];
        for (size_t i = 0; i < submitted.batchSignals.size(); ++i) receipt.batches[i].signal = submitted.batchSignals[i];
        receipt.tailCompletions = submitted.tailCompletions;
        admission.Commit(held,receipt,submitted.signaledBatches);
        for (size_t i = 0; i < decoded.completions.size(); ++i) {
            const auto& complete = decoded.completions[i];
            const std::array points{ExecutionTimelinePoint{complete.timeline,complete.value}};
            CHECK(admission.RetireCompleted(points) == (i == 2 ? 1 : 0));
            CHECK(admission.RetainedFrames() == (i == 2 ? 0 : 1));
        }
        CHECK(admission.RetainedFrames() == 0);
        // An unsubmitted suffix must not enter either the completion ledger or
        // the resource hazard ledger. Its zero signal is valid replay data.
        auto partialSequence = sequence;
        partialSequence.submissions = {{11,{{12,20},{13,0}},{},1}};
        partialSequence.completions = {{12,12,20}};
        std::stringstream partialStream; WriteGraphReplaySequence(partialStream,partialSequence);
        const auto partialDecoded = ReadGraphReplaySequence(partialStream);
        auto partial = admission.Prepare(selected,queues);
        const auto& prefix = partialDecoded.submissions[0];
        GraphExecutionTimeline partialReceipt; partialReceipt.batches.resize(2);
        for (size_t i = 0; i < prefix.batchSignals.size(); ++i) partialReceipt.batches[i].signal = prefix.batchSignals[i];
        admission.Commit(partial,partialReceipt,prefix.signaledBatches);
        // Move the next writer to the other timeline: this requires an
        // external wait rather than relying on same-queue submission order.
        const std::array<ExecutionTimelinePoint,2> swappedQueues{{{13,0},{12,0}}};
        auto afterPrefix = admission.Prepare(selected,swappedQueues);
        bool waitsForSubmittedWriter = false;
        for (const auto& waits : afterPrefix.incomingWaits) for (const auto point : waits) {
            CHECK(point.timeline != 13);
            CHECK(point.value != 0);
            waitsForSubmittedWriter |= point.timeline == 12 && point.value == 20;
        }
        CHECK(waitsForSubmittedWriter);
        admission.Abandon(afterPrefix);
        const std::array unsubmitted{ExecutionTimelinePoint{13,20}};
        CHECK(Rejects([&] { admission.RetireCompleted(unsubmitted); }));
        CHECK(admission.RetainedFrames() == 1);
        const auto& completedPrefix = partialDecoded.completions[0];
        const std::array prefixDone{ExecutionTimelinePoint{completedPrefix.timeline,completedPrefix.value}};
        CHECK(admission.RetireCompleted(prefixDone) == 1);
        CHECK(admission.RetainedFrames() == 0);
    }
    {
        GraphProgram program; CompileWorkspace workspace; std::atomic_bool cancelled{false};
        auto snapshot = std::make_shared<org::ResourceBindingSnapshot>();
        snapshot->resource = rhi::Resource(rhi::ResourceHandle{81,1}); snapshot->backingGeneration = 1;
        snapshot->allocationOwner = std::make_shared<const uint32_t>(1);
        snapshot->description.type = rhi::ResourceType::AccelerationStructure;
        snapshot->description.buffer.sizeBytes = 4096;
        PreparedBackingState initial;
        initial.graphResourceID = 1; initial.resource = {81,1}; initial.shape = {1,1,false};
        initial.regions = std::make_shared<const std::vector<PreparedStateRegion>>();
        const auto contract = NativeBindingContract::Capture(snapshot->description);
        CHECK(contract.Matches(snapshot->description,initial.shape));
        CHECK(contract.Matches(snapshot->description,{0,0,false}));
        auto buffer = snapshot->description; buffer.type = rhi::ResourceType::Buffer;
        CHECK(!contract.Matches(buffer,initial.shape));
        auto edit = program.BeginEdit(); const auto slot = edit.AddSnapshot(snapshot,initial);
        const auto group = edit.AddGroup(initial.shape,{slot});
        edit.SetGroupResourceClass(group,rhi::ResourceType::AccelerationStructure);
        const auto pass = edit.AddPass({});
        edit.DeclareGroupAccess(pass,group,
            {static_cast<uint64_t>(rhi::ResourceAccessType::RaytracingAccelerationStructureRead),0,
                static_cast<uint64_t>(rhi::ResourceSyncState::All),false},0);
        auto ready = edit.Build(workspace,cancelled); CHECK(program.Install(ready));
        CHECK(CheckStates(*ready->executable->graph));
        auto invalid = program.BeginEdit(); auto other = std::make_shared<org::ResourceBindingSnapshot>(*snapshot);
        other->resource = rhi::Resource(rhi::ResourceHandle{82,1}); other->description = buffer;
        auto otherInitial = initial; otherInitial.graphResourceID = 2; otherInitial.resource = {82,1};
        const auto ordinary = invalid.AddSnapshot(other,otherInitial);
        CHECK(Rejects([&] { invalid.ReplaceGroupMembers(group,{slot,ordinary}); }));
        CHECK(Rejects([&] { invalid.Build(workspace,cancelled); }));
        CHECK(program.Select() == ready);
    }
    {
        GraphProgram program; CompileWorkspace workspace; std::atomic_bool cancelled{false};
        auto edit = program.BeginEdit(); auto binding = Binding(900);
        binding.shape = {3,2,true};
        const auto slot = edit.AddResource(binding.shape,binding);
        const auto group = edit.AddGroup(binding.shape,{slot});
        const auto first = edit.AddPass({}), second = edit.AddPass({});
        CompileResourceState write{static_cast<uint64_t>(rhi::ResourceAccessType::UnorderedAccess),
            static_cast<uint64_t>(rhi::ResourceLayout::UnorderedAccess),
            static_cast<uint64_t>(rhi::ResourceSyncState::ComputeShading),true};
        edit.DeclareGroupAccess(first,group,write,0,CompileRange{0,1,0,1});
        edit.DeclareGroupAccess(first,group,write,0,CompileRange{1,1,0,1});
        edit.DeclareGroupAccess(second,group,write,0,CompileRange{2,1,1,1});
        auto ready = edit.Build(workspace,cancelled);
        CHECK(program.Install(ready));
        CHECK(ready->executable->graph->structure->passes[0].entryStates[0].range == (CompileRange{0,1,0,1}));
        CHECK(ready->executable->graph->structure->passes[0].entryStates[1].range == (CompileRange{1,1,0,1}));
        CHECK(ready->executable->graph->structure->passes[1].entryStates[0].range == (CompileRange{2,1,1,1}));
        CHECK(CheckStates(*ready->executable->graph));
        auto invalid = program.BeginEdit(); const auto third = invalid.AddPass({});
        CHECK(Rejects([&] { invalid.DeclareGroupAccess(third,group,write,0,CompileRange{3,1,0,1}); }));
        CHECK(Rejects([&] { invalid.Build(workspace,cancelled); }));
        auto overlap = program.BeginEdit(); const auto overlapping = overlap.AddPass({});
        overlap.DeclareGroupAccess(overlapping,group,write,0,CompileRange{0,1,0,1});
        CHECK(Rejects([&] { overlap.Build(workspace,cancelled); }));
        CHECK(program.Select() == ready);
        auto duplicate = program.BeginEdit();
        CHECK(Rejects([&] { duplicate.DeclareGroupAccess(first,group,write,0,CompileRange{0,2,0,1}); }));
        CHECK(Rejects([&] { duplicate.Build(workspace,cancelled); }));
        auto removal = program.BeginEdit(); removal.RemoveGroupAccess(first,group);
        auto removed = removal.Build(workspace,cancelled);
        CHECK(removed->executable->graph->structure->passes[0].entryStates.empty());
        CHECK(removed->logical->groups[group.index].subscribers.size() == 1);
        auto replacement = program.BeginEdit(); replacement.ReplacePass(first,{});
        auto replaced = replacement.Build(workspace,cancelled);
        CHECK(replaced->logical->groups[group.index].subscribers.size() == 1);
        CHECK(replaced->executable->graph->structure->passes[0].entryStates.empty());
        CHECK(ready->logical->groups[group.index].subscribers.size() == 3);
        auto retirement = program.BeginEdit(); retirement.RemovePass(first);
        const auto reused = retirement.AddPass({});
        CHECK(reused.index == first.index && reused.generation != first.generation);
        auto reusedPublication = retirement.Build(workspace,cancelled);
        CHECK(reusedPublication->logical->groups[group.index].subscribers.size() == 1);
        CHECK(reusedPublication->executable->graph->structure->passes[0].entryStates.empty());
    }
    {
        auto snapshot = std::make_shared<org::ResourceBindingSnapshot>();
        snapshot->resource = rhi::Resource(rhi::ResourceHandle{71,2});
        snapshot->backingGeneration = 7;
        snapshot->allocationOwner = std::make_shared<const uint64_t>(9);
        snapshot->description.type = rhi::ResourceType::Buffer;
        snapshot->description.buffer.sizeBytes = 128;
        PreparedBackingState initial;
        initial.resource = {71,2}; initial.graphResourceID = 1; initial.shape = {1,1,false};
        initial.regions = std::make_shared<const std::vector<PreparedStateRegion>>();
        auto binding = BindingVersion::FromSnapshot(snapshot,initial,11,13);
        CHECK(binding.identity == ((uint64_t{2} << 32) | 71));
        CHECK(binding.backingRevision == 7 && binding.descriptorRevision == 11 && binding.contentRevision == 13);
        CHECK(binding.owner == snapshot && binding.recording == snapshot);
        auto wrong = initial; wrong.resource.generation = 3;
        CHECK(Rejects([&] { BindingVersion::FromSnapshot(snapshot,wrong); }));
        wrong = initial; wrong.shape = {2,1,false};
        CHECK(Rejects([&] { BindingVersion::FromSnapshot(snapshot,wrong); }));
        wrong = initial; wrong.heapType = rhi::HeapType::Readback;
        CHECK(Rejects([&] { BindingVersion::FromSnapshot(snapshot,wrong); }));
        wrong = initial; wrong.aliasOffset = 128;
        CHECK(Rejects([&] { BindingVersion::FromSnapshot(snapshot,wrong); }));
        wrong = initial; wrong.regions.reset();
        CHECK(Rejects([&] { BindingVersion::FromSnapshot(snapshot,wrong); }));
        {
            // Numeric-only fixture; heap objects are never dereferenced here.
            auto storage = std::make_shared<const uint64_t>(1);
            const auto address = reinterpret_cast<const org::AliasHeapGeneration*>(storage.get());
            auto heap = std::shared_ptr<const org::AliasHeapGeneration>(storage,address);
            auto foreignStorage = std::make_shared<const uint64_t>(2);
            auto foreignHeap = std::shared_ptr<const org::AliasHeapGeneration>(foreignStorage,address);
            auto placed = std::make_shared<org::ResourceBindingSnapshot>(*snapshot);
            placed->aliasHeap = heap; placed->aliasSize = 128;
            auto placement = initial; placement.aliasHeap = heap;
            placement.aliasHeapIdentity = address; placement.aliasSize = 128;
            CHECK(BindingVersion::FromSnapshot(placed,placement).recording == placed);
            auto invalid = BindingVersion::FromSnapshot(placed,placement);
            placement.aliasHeap = foreignHeap;
            CHECK(Rejects([&] { BindingVersion::FromSnapshot(placed,placement); }));
            // Direct binding users receive the same ownership check.
            GraphProgram program; auto edit = program.BeginEdit();
            invalid.admission = std::make_shared<const PreparedBackingState>(placement);
            CHECK(Rejects([&] { edit.AddResource(invalid.shape,invalid); }));
        }
        std::weak_ptr<const org::ResourceBindingSnapshot> retained = snapshot;
        {
            GraphProgram program;
            CompileWorkspace workspace; std::atomic_bool cancelled{false};
            auto bootstrap = program.BeginEdit();
            const auto slot = bootstrap.AddSnapshot(snapshot,initial,11,13);
            const auto group = bootstrap.AddGroup(initial.shape,{slot});
            bootstrap.SetGroupResourceClass(group,rhi::ResourceType::Buffer);
            auto selected = bootstrap.Build(workspace,cancelled);
            CHECK(program.Install(selected));
            CHECK(selected->logical->groups[group.index].memberType == rhi::ResourceType::Buffer);
            auto incompatibleClass = program.BeginEdit();
            CHECK(Rejects([&] { incompatibleClass.SetGroupResourceClass(group,rhi::ResourceType::AccelerationStructure); }));
            CHECK(Rejects([&] { incompatibleClass.Build(workspace,cancelled); }));
            auto incompatibleMember = program.BeginEdit();
            const auto untyped = incompatibleMember.AddResource(initial.shape,Binding(99));
            CHECK(Rejects([&] { incompatibleMember.ReplaceGroupMembers(group,{untyped}); }));
            CHECK(Rejects([&] { incompatibleMember.Build(workspace,cancelled); }));
            CHECK(program.Select() == selected);
            auto failed = program.BeginEdit();
            failed.ReplaceSnapshot(slot,snapshot,initial,12,14);
            // A valid earlier entry must not become visible after a later
            // producer snapshot fails before ReplaceBinding is entered.
            CHECK(Rejects([&] { failed.ReplaceSnapshot(slot,snapshot,wrong); }));
            CHECK(Rejects([&] { failed.Build(workspace,cancelled); }));
            CHECK(program.Select() == selected);
            auto failedBootstrap = program.BeginEdit();
            CHECK(Rejects([&] { failedBootstrap.AddSnapshot(snapshot,wrong); }));
            CHECK(Rejects([&] { failedBootstrap.Build(workspace,cancelled); }));
            CHECK(program.Select() == selected);
        }
        snapshot.reset(); CHECK(!retained.expired());
        binding = {}; CHECK(retained.expired());
    }
    {
        auto semanticOwner = std::make_shared<const uint64_t>(99);
        std::weak_ptr<const void> weak = semanticOwner;
        auto capturedInput = std::make_shared<GraphCompileInput>();
        capturedInput->leases.push_back(semanticOwner);
        auto alias = std::shared_ptr<const GraphCompileStructure>(capturedInput,&capturedInput->structure);
        ObserveGraphReplay(alias);
        semanticOwner.reset(); capturedInput.reset(); alias.reset();
        CHECK(weak.expired());
    }
    GraphProgram program;
    CompileWorkspace workspace; std::atomic_bool cancelled{false};
    auto edit = program.BeginEdit();
    auto slot = edit.AddResource({1,1,false}, Binding(1));
    CompilePass producer; producer.accesses = {{slot.index,true}};
    producer.entryStates = {{slot.index,{}, {1,0,1,true}}};
    auto first = edit.AddPass(producer);
    CompilePass consumer; consumer.accesses = {{slot.index,false}};
    consumer.entryStates = {{slot.index,{}, {2,0,1,false}}};
    auto second = edit.AddPass(consumer);
    auto ready = edit.Build(workspace, cancelled);
    CHECK(program.Install(edit, ready));
    CHECK(ready->executable->executionLayout);
    CHECK(ready->executable->executionLayout->bundle->graph == ready->executable->graph);
    CHECK(ready->executable->executionLayout->bundle->input == ready->executable->compileInput);
    CHECK(ready->executable->compileInput);
    CHECK(ready->executable->graph->structure.get() == &ready->executable->compileInput->structure);
    CHECK(CheckConflicts(*ready->executable->graph));
    CHECK(CheckStates(*ready->executable->graph));
    auto missingTransition = *ready->executable->graph;
    missingTransition.states.steps.erase(missingTransition.states.steps.begin());
    CHECK(!CheckStates(missingTransition));
    auto wrongBefore = *ready->executable->graph;
    wrongBefore.states.steps.back().before.access ^= 16;
    CHECK(!CheckStates(wrongBefore));
    auto missingTerminal = *ready->executable->graph;
    missingTerminal.states.finalStates.clear();
    CHECK(!CheckStates(missingTerminal));
    std::stringstream replay;
    WriteGraphReplay(replay, *ready->executable->graph->structure);
    auto restored = ReadGraphReplay(replay);
    CHECK(restored.resourceIDs == ready->executable->graph->structure->resourceIDs);
    CHECK(restored.passes == ready->executable->graph->structure->passes);
    std::stringstream truncated("ORG_GRAPH_REPLAY 1\n650\n");
    CHECK(Rejects([&] { ReadGraphReplay(truncated); }));
    auto corrupted = *ready->executable->graph;
    std::reverse(corrupted.batches.front().passes.begin(), corrupted.batches.front().passes.end());
    CHECK(!CheckConflicts(corrupted));
    auto oldFrame = program.Select();
    auto replacement = program.BeginEdit();
    replacement.ReplaceBinding(slot, Binding(2));
    auto newReady = replacement.Build(workspace, cancelled);
    CHECK(newReady->executable == oldFrame->executable);
    // Editing a transaction after Build cannot mutate the sealed publication.
    replacement.ReplaceBinding(slot, Binding(3));
    CHECK(newReady->bindings.At(slot).identity == 2);
    CHECK(program.Install(replacement, newReady));
    CHECK(oldFrame->bindings.At(slot).identity == 1);
    CHECK(program.Select()->bindings.At(slot).identity == 2);
    CHECK(!program.Install(replacement, newReady));
    auto stale = program.BeginEdit(); auto winning = program.BeginEdit();
    winning.ReplaceBinding(slot, Binding(4));
    CHECK(program.Install(winning, winning.Build(workspace, cancelled)));
    CHECK(!program.Install(stale, stale.Build(workspace, cancelled)));
    auto invalid = program.BeginEdit();
    CHECK(Rejects([&] { invalid.ReplaceBinding({slot.index,2}, Binding(5)); }));
    auto wrongShape = Binding(5); wrongShape.shape.mips = 2;
    CHECK(Rejects([&] { invalid.ReplaceBinding(slot, wrongShape); }));
    invalid.AddOrdering(second, first);
    CHECK(Rejects([&] { invalid.Build(workspace, cancelled); }));
    CHECK(program.Select()->bindings.At(slot).identity == 4);
    cancelled = true;
    CHECK(!program.BeginEdit().Build(workspace, cancelled));
    cancelled = false;
    {
        auto tasks = std::make_shared<ManualTasks>();
        PublicationCoordinator coordinator(program,tasks);
        auto old = program.Select();
        auto superseded = program.BeginEdit(); superseded.ReplaceBinding(slot,Binding(77));
        auto newest = program.BeginEdit(); newest.ReplaceBinding(slot,Binding(88));
        coordinator.Submit(std::move(superseded)); coordinator.Submit(std::move(newest));
        CHECK(coordinator.Pump() == PublicationState::Pending);
        CHECK(program.Select() == old);
        tasks->Run(1);
        CHECK(coordinator.State() == PublicationState::Ready);
        CHECK(program.Select() == old);
        CHECK(coordinator.Pump() == PublicationState::Selected);
        tasks->Run(0);
        CHECK(program.Select()->bindings.At(slot).identity == 88);
        CHECK(old->bindings.At(slot).identity == 4);
        CHECK(program.Select()->executable == old->executable);
        auto failure = program.BeginEdit(); failure.AddOrdering(second,first);
        coordinator.Submit(std::move(failure)); tasks->RunAll();
        CHECK(coordinator.Pump() == PublicationState::Failed);
        CHECK(!coordinator.Failure().empty());
        CHECK(program.Select()->bindings.At(slot).identity == 88);
        tasks->reject = true;
        coordinator.Submit(program.BeginEdit());
        CHECK(coordinator.Pump() == PublicationState::Failed);
        tasks->reject = false;
        coordinator.Submit(program.BeginEdit()); tasks->RunAll();
        auto intervening = program.BeginEdit(); intervening.ReplaceBinding(slot,Binding(99));
        CHECK(program.Install(intervening,intervening.Build(workspace,cancelled)));
        CHECK(coordinator.Pump() == PublicationState::Superseded);
    }
    {
        GraphProgram admittedProgram;
        auto transaction = admittedProgram.BeginEdit();
        transaction.SetQueues({{0,true},{0,true}});
        auto binding = Binding(1);
        auto state = std::make_shared<PreparedBackingState>();
        state->graphResourceID = 1; state->resource = {1,1}; state->shape = binding.shape;
        state->regions = std::make_shared<const std::vector<PreparedStateRegion>>(
            std::initializer_list<PreparedStateRegion>{{{}, {static_cast<uint64_t>(rhi::ResourceAccessType::Common),0,
                static_cast<uint64_t>(rhi::ResourceSyncState::All),false}}});
        binding.identity = (uint64_t{1} << 32) | 1;
        binding.admission = state;
        transaction.AddResource(binding.shape,binding);
        CompilePass writePass;
        writePass.accesses = {{0,true}};
        writePass.entryStates = {{0,{}, {static_cast<uint64_t>(rhi::ResourceAccessType::UnorderedAccess),0,
            static_cast<uint64_t>(rhi::ResourceSyncState::ComputeShading),true}}};
        transaction.AddPass(writePass);
        CompilePass readPass;
        readPass.accesses = {{0,false}}; readPass.compatibleQueueSlots = {1}; readPass.preferredQueueSlot = 1;
        readPass.entryStates = {{0,{}, {static_cast<uint64_t>(rhi::ResourceAccessType::ShaderResource),0,
            static_cast<uint64_t>(rhi::ResourceSyncState::ComputeShading),false}}};
        transaction.AddPass(readPass);
        auto selected = transaction.Build(workspace,cancelled);
        CHECK(admittedProgram.Install(transaction,selected));
        CHECK(CheckConflicts(*selected->executable->graph));
        auto noWaits = *selected->executable->graph;
        noWaits.relativeWaits.clear();
        CHECK(!CheckConflicts(noWaits));
        auto wrongQueue = *selected->executable->graph;
        wrongQueue.batches[1].queue = 0;
        CHECK(!CheckConflicts(wrongQueue));
        SynchronousAdmission admission;
        const std::array<ExecutionTimelinePoint,2> queues{{{1,0},{2,0}}};
        auto frame = admission.Prepare(selected,queues);
        CHECK(Rejects([&] { admission.Prepare(selected,queues); }));
        GraphExecutionTimeline receipt;
        receipt.batches.resize(2);
        receipt.batches[0].signal = {1,5}; receipt.batches[1].signal = {2,7};
        receipt.tailCompletions = {{3,9}};
        admission.Commit(frame,receipt);
        BackingAccessAdmissionLedger retirementHazards;
        retirementHazards.Commit(*selected->executable->graph,frame.backings,receipt);
        CHECK(!retirementHazards.RetireCompleted(state->resource,{{1,5}}));
        CHECK(retirementHazards.BackingCount() == 1);
        CHECK(retirementHazards.RetireCompleted(state->resource,{{1,5},{2,7}}));
        CHECK(retirementHazards.BackingCount() == 0);
        // Pointer-only alias fixture: no heap allocation is dereferenced by
        // the numeric ledger. Distinct control blocks model address reuse.
        auto heapOwner = std::make_shared<const uint64_t>(1);
        auto heap = std::shared_ptr<const org::AliasHeapGeneration>(heapOwner,
            reinterpret_cast<const org::AliasHeapGeneration*>(heapOwner.get()));
        const auto heapIdentity = heap.get();
        const std::weak_ptr<const org::AliasHeapGeneration> heapWeak = heap;
        auto aliasBackings = frame.backings;
        aliasBackings[0].aliasHeap = heap;
        aliasBackings[0].aliasHeapIdentity = heapIdentity;
        aliasBackings[0].aliasSize = 4096;
        auto aliasGraph = *selected->executable->graph;
        aliasGraph.aliasFinalResourcesByBatch = {{0},{0}};
        AliasAccessAdmissionLedger retirementAliases;
        retirementAliases.Commit(aliasGraph,aliasBackings,receipt);
        CHECK(!retirementAliases.RetireCompleted(heapIdentity,heapWeak,{{1,5},{2,7}}));
        aliasBackings.clear(); heap.reset(); heapOwner.reset();
        CHECK(!retirementAliases.RetireCompleted(heapIdentity,heapWeak,{{1,5}}));
        auto unrelated = std::make_shared<const uint64_t>(2);
        std::weak_ptr<const org::AliasHeapGeneration> wrongOwner =
            std::shared_ptr<const org::AliasHeapGeneration>(unrelated,heapIdentity);
        unrelated.reset();
        CHECK(!retirementAliases.RetireCompleted(heapIdentity,wrongOwner,{{1,5},{2,7}}));
        CHECK(retirementAliases.RetireCompleted(heapIdentity,heapWeak,{{1,5},{2,7}}));
        CHECK(retirementAliases.HeapCount() == 0);
        CHECK(Rejects([&] { admission.Commit(frame,receipt); }));
        CHECK(admission.RetainedFrames() == 1);
        const std::array<ExecutionTimelinePoint,2> batchDone{{{1,5},{2,7}}};
        CHECK(admission.RetireCompleted(batchDone) == 0);
        const std::array<ExecutionTimelinePoint,1> tailDone{{{3,9}}};
        CHECK(admission.RetireCompleted(tailDone) == 1);
        const std::array<ExecutionTimelinePoint,1> impossibleDone{{{1,6}}};
        CHECK(Rejects([&] { admission.RetireCompleted(impossibleDone); }));
        auto next = admission.Prepare(selected,queues);
        CHECK(!next.incomingWaits[0].empty());
        CHECK(next.incomingWaits[0][0].timeline == 2);
        CHECK(next.incomingWaits[0][0].value == 7);
        admission.Abandon(next);
        CHECK(Rejects([&] { admission.Abandon(next); }));
        auto partial = admission.Prepare(selected,queues);
        receipt.tailCompletions.clear(); receipt.batches[0].signal = {1,10}; receipt.batches[1].signal = {};
        admission.Commit(partial,receipt,1);
        const std::array<ExecutionTimelinePoint,1> partialDone{{{1,10}}};
        CHECK(admission.RetireCompleted(partialDone) == 1);
        // Sustained replacement must not preserve historical state grids.
        frame = {}; next = {}; partial = {};
        for (uint32_t generation = 2; generation <= 129; ++generation) {
            const auto old = SynchronousAdmission::CaptureRetirement(binding);
            auto replacement = binding;
            replacement.owner = std::make_shared<const uint32_t>(generation);
            replacement.identity = (uint64_t{generation} << 32) | 1;
            replacement.backingRevision = generation;
            auto replacementState = std::make_shared<PreparedBackingState>(*state);
            replacementState->resource = {1,generation};
            replacement.admission = replacementState;
            {
                auto rotation = admittedProgram.BeginEdit();
                rotation.ReplaceBinding({0,1},replacement);
                CHECK(admittedProgram.Install(rotation,rotation.Build(workspace,cancelled)));
            }
            selected = admittedProgram.Select();
            binding = replacement;
            transaction = admittedProgram.BeginEdit();
            CHECK(admission.RetireBackingMetadata(old));
            FrameIncomingState incoming{{0,1},replacementState->resource,
                std::make_shared<const std::vector<PreparedStateRegion>>(
                    std::initializer_list<PreparedStateRegion>{{{}, {static_cast<uint64_t>(rhi::ResourceAccessType::CopyDest),0,
                        static_cast<uint64_t>(rhi::ResourceSyncState::Copy),true}}}),{90,generation},1};
            auto fresh = admission.Prepare(selected,queues,{},std::span{&incoming,1});
            receipt.batches[0].signal = {1,10 + generation};
            receipt.batches[1].signal = {2,7 + generation};
            admission.Commit(fresh,receipt);
            const std::array<ExecutionTimelinePoint,2> rotationDone{{
                receipt.batches[0].signal,receipt.batches[1].signal}};
            CHECK(admission.RetireCompleted(rotationDone) == 1);
            CHECK(admission.StateBackingCount() == 1);
            CHECK(admission.HazardBackingCount() == 1);
            CHECK(admission.IncomingRevisionCount() == 1);
        }
        // Hold one fresh CPU packet beyond completion to test delayed cleanup.
        frame = admission.Prepare(selected,queues);
        admission.Abandon(frame);
        const auto retirement = SynchronousAdmission::CaptureRetirement(binding);
        CHECK(admission.StateBackingCount() == 1);
        CHECK(admission.HazardBackingCount() == 1);
        CHECK(!admission.RetireBackingMetadata(retirement));
        {
            auto removal = admittedProgram.BeginEdit();
            removal.RemovePass({0,1});
            removal.RemovePass({1,1});
            removal.RemoveResource({0,1});
            CHECK(admittedProgram.Install(removal, removal.Build(workspace,cancelled)));
        }
        // A held CPU frame protects state even after all queue/tail completion.
        selected.reset(); binding.owner.reset();
        CHECK(!admission.RetireBackingMetadata(retirement));
        frame = {}; next = {}; partial = {};
        transaction = admittedProgram.BeginEdit();
        CHECK(admission.RetireBackingMetadata(retirement));
        CHECK(admission.StateBackingCount() == 0);
        CHECK(admission.HazardBackingCount() == 0);
        CHECK(admission.IncomingRevisionCount() == 0);
        CHECK(admission.RetireBackingMetadata(retirement));
    }
    // Fixed seed and independent checker provide reproducible multi-queue cases.
    {
        GraphProgram views;
        auto edit = views.BeginEdit();
        auto binding = Binding(1);
        binding.identity = (uint64_t{1} << 32) | 10;
        binding.backingRevision = 9;
        auto snapshot = std::make_shared<org::ResourceBindingSnapshot>();
        snapshot->resource = rhi::Resource(rhi::ResourceHandle{10,1});
        snapshot->backingGeneration = 9;
        snapshot->allocationOwner = binding.owner;
        snapshot->descriptorOwner = std::make_shared<const uint32_t>(7);
        auto publishedViews = std::make_shared<org::BindlessResourceViews>();
        publishedViews->views.push_back({org::BindlessViewKind::ShaderResource,UINT32_MAX,0,0,
            rhi::DescriptorSlot(rhi::DescriptorHeapHandle{3,1},7)});
        snapshot->views = publishedViews;
        binding.recording = snapshot;
        auto resource = edit.AddResource(binding.shape,std::move(binding));
        auto pass = edit.AddPass({});
        auto token = edit.Declare(pass,resource,{}, {2,0,1,false});
        const auto descriptorToken = edit.DeclareView(token,{});
        auto held = edit.Build(workspace,cancelled);
        CHECK(views.Install(edit,held));
        CHECK(held->ResolveView(token,{}).index == 7);
        CHECK(held->ResolveView(descriptorToken).index == 7);
        auto shared = views.BeginEdit();
        const auto otherPass = shared.AddPass({});
        const auto otherBinding = shared.Declare(otherPass,resource,{}, {2,0,1,false});
        const auto otherView = shared.DeclareView(otherBinding,{});
        auto sharedReady = shared.Build(workspace,cancelled);
        CHECK(sharedReady->bindings.At(resource).preparedViews->size() == 2);
        GraphEditTransaction removeOne(sharedReady);
        removeOne.RemovePass(pass);
        auto remaining = removeOne.Build(workspace,cancelled);
        CHECK(remaining->bindings.At(resource).preparedViews->size() == 1);
        CHECK(remaining->ResolveView(otherView).index == 7);
        CHECK(sharedReady->ResolveView(descriptorToken).index == 7);
        GraphEditTransaction removeLast(remaining);
        removeLast.RemovePass(otherPass);
        auto noConsumers = removeLast.Build(workspace,cancelled);
        CHECK(!noConsumers->bindings.At(resource).preparedViews);
        auto rotation = views.BeginEdit();
        auto next = held->bindings.At(resource);
        auto replacement = std::make_shared<org::ResourceBindingSnapshot>(*next.recording);
        auto replacementViews = std::make_shared<org::BindlessResourceViews>(*replacement->views);
        replacementViews->views[0].descriptor.index = 11;
        replacement->views = replacementViews;
        replacement->descriptorOwner = std::make_shared<const uint32_t>(11);
        next.recording = replacement; ++next.descriptorRevision;
        rotation.ReplaceBinding(resource,std::move(next));
        auto selected = rotation.Build(workspace,cancelled);
        CHECK(selected->executable == held->executable);
        CHECK(selected->ResolveView(token,{}).index == 11 && held->ResolveView(token,{}).index == 7);
        CHECK(selected->ResolveView(descriptorToken).index == 11 && held->ResolveView(descriptorToken).index == 7);
        CHECK(Rejects([&] { held->ResolveView(ViewToken{}); }));
        auto badBounds = views.BeginEdit();
        CHECK(Rejects([&] { badBounds.DeclareView(token,{org::BindlessViewKind::ShaderResource,UINT32_MAX,1,0}); }));
        auto badHeap = views.BeginEdit();
        auto badHeapBinding = held->bindings.At(resource);
        auto badHeapSnapshot = std::make_shared<org::ResourceBindingSnapshot>(*badHeapBinding.recording);
        auto badHeapViews = std::make_shared<org::BindlessResourceViews>(*badHeapSnapshot->views);
        badHeapViews->views[0].descriptor.heap = {};
        badHeapSnapshot->views = badHeapViews; badHeapBinding.recording = badHeapSnapshot;
        CHECK(Rejects([&] { badHeap.ReplaceBinding(resource,std::move(badHeapBinding)); }));
        auto competingA = views.BeginEdit();
        const auto pendingView = competingA.DeclareView(token,{});
        auto readyA = competingA.Build(workspace,cancelled);
        auto competingB = views.BeginEdit();
        const auto alternateView = competingB.DeclareView(token,{});
        auto readyB = competingB.Build(workspace,cancelled);
        CHECK(readyA->ResolveView(pendingView).index == 7 && readyB->ResolveView(alternateView).index == 7);
        CHECK(Rejects([&] { readyB->ResolveView(pendingView); }));
        CHECK(Rejects([&] { selected->ResolveView(token,{org::BindlessViewKind::UnorderedAccess}); }));
        auto unowned = views.BeginEdit();
        auto invalid = held->bindings.At(resource);
        auto invalidSnapshot = std::make_shared<org::ResourceBindingSnapshot>(*invalid.recording);
        invalidSnapshot->descriptorOwner.reset(); invalid.recording = invalidSnapshot;
        CHECK(Rejects([&] { unowned.ReplaceBinding(resource,std::move(invalid)); }));
        auto missing = views.BeginEdit();
        auto missingBinding = held->bindings.At(resource);
        auto missingSnapshot = std::make_shared<org::ResourceBindingSnapshot>(*missingBinding.recording);
        missingSnapshot->views = std::make_shared<const org::BindlessResourceViews>();
        missingBinding.recording = missingSnapshot;
        CHECK(Rejects([&] { missing.ReplaceBinding(resource,missingBinding); }));
        CHECK(views.Select() == held);
        auto layoutChange = views.BeginEdit();
        layoutChange.ReplacePass(pass,{});
        layoutChange.ReplaceBinding(resource,missingBinding);
        auto changed = layoutChange.Build(workspace,cancelled);
        CHECK(changed->logical->bindingSubscribers[resource.index].empty());
        CHECK(!changed->bindings.At(resource).preparedViews);
        CHECK(held->bindings.At(resource).preparedViews);
        CHECK(Rejects([&] { changed->ResolveView(token,{}); }));
        CHECK(Rejects([&] { changed->ResolveView(descriptorToken); }));
        auto retirement = views.BeginEdit();
        retirement.RemovePass(pass);
        auto retired = retirement.Build(workspace,cancelled);
        CHECK(!retired->bindings.At(resource).preparedViews);
        CHECK(held->ResolveView(descriptorToken).index == 7);
    }
    {
        GraphProgram postconditions;
        auto edit = postconditions.BeginEdit();
        auto binding = Binding(1101); binding.shape = {2,2,true};
        const auto resource = edit.AddResource(binding.shape,std::move(binding));
        const auto producer = edit.AddPass({});
        const auto token = edit.Declare(producer,resource,{0,2,0,2},{2,3,1,false});
        edit.DeclarePostcondition(token,{1,1,1,1},{1,4,2,true});
        const auto consumer = edit.AddPass({});
        edit.Declare(consumer,resource,{1,1,1,1},{2,5,1,false});
        auto ready = edit.Build(workspace,cancelled);
        CHECK(postconditions.Install(edit,ready));
        CHECK(CheckConflicts(*ready->executable->graph) && CheckStates(*ready->executable->graph));
        const auto& steps = ready->executable->graph->states.steps;
        CHECK(std::any_of(steps.begin(),steps.end(),[&](const auto& step) {
            return step.pass == consumer.index && step.before == CompileResourceState{1,4,2,true};
        }));
        auto replaced = postconditions.BeginEdit();
        replaced.ReplacePass(producer,{});
        CHECK(Rejects([&] { replaced.DeclarePostcondition(token,{}, {2,3,1,false}); }));
        GraphProgram foreign;
        auto foreignEdit = foreign.BeginEdit();
        CHECK(Rejects([&] { foreignEdit.DeclarePostcondition(token,{}, {2,3,1,false}); }));
        auto missingEntry = postconditions.BeginEdit();
        const auto callback = missingEntry.AddPass({});
        const auto dependency = missingEntry.DeclareDependency(callback,resource,false);
        missingEntry.DeclarePostcondition(dependency,{}, {2,3,1,false});
        CHECK(Rejects([&] { missingEntry.Build(workspace,cancelled); }));
        CHECK(postconditions.Select() == ready);
    }
    {
        GraphProgram typed;
        auto bootstrap = typed.BeginEdit();
        const auto a = bootstrap.AddResource({1,1,false},Binding(1001));
        const auto b = bootstrap.AddResource({1,1,false},Binding(1002));
        const auto pass = bootstrap.AddPass({});
        const auto token = bootstrap.Declare(pass,a,{}, {2,0,1,false});
        CHECK(Rejects([&] { typed.Select()->Resolve(token); }));
        auto held = bootstrap.Build(workspace,cancelled);
        CHECK(typed.Install(bootstrap,held));
        CHECK(held->Resolve(token).identity == 1001);
        CHECK(Rejects([&] { held->Resolve({}); }));
        auto backing = typed.BeginEdit();
        backing.ReplaceBinding(a,Binding(1003));
        auto rotated = backing.Build(workspace,cancelled);
        CHECK(typed.Install(backing,rotated));
        CHECK(rotated->executable == held->executable);
        CHECK(rotated->Resolve(token).identity == 1003 && held->Resolve(token).identity == 1001);
        GraphProgram foreign;
        CHECK(Rejects([&] { foreign.Select()->Resolve(token); }));
        auto firstCandidate = typed.BeginEdit();
        firstCandidate.ReplacePass(pass,{});
        const auto rejectedToken = firstCandidate.Declare(pass,a,{}, {2,0,1,false});
        auto firstReady = firstCandidate.Build(workspace,cancelled);
        auto secondCandidate = typed.BeginEdit();
        secondCandidate.ReplacePass(pass,{});
        const auto selectedToken = secondCandidate.Declare(pass,b,{}, {2,0,1,false});
        auto secondReady = secondCandidate.Build(workspace,cancelled);
        CHECK(typed.Install(secondCandidate,secondReady));
        CHECK(!typed.Install(firstCandidate,firstReady));
        CHECK(secondReady->Resolve(selectedToken).identity == 1002);
        CHECK(Rejects([&] { secondReady->Resolve(rejectedToken); }));
        CHECK(Rejects([&] { secondReady->Resolve(token); }));
        CHECK(firstReady->Resolve(rejectedToken).identity == 1003);
        auto invalidRange = typed.BeginEdit();
        CHECK(Rejects([&] { invalidRange.Declare(pass,b,{0,2,0,1},{2,0,1,false}); }));
        CHECK(Rejects([&] { invalidRange.Build(workspace,cancelled); }));
        auto removal = typed.BeginEdit();
        removal.RemovePass(pass); removal.RemoveResource(b);
        auto retiredReady = removal.Build(workspace,cancelled);
        CHECK(retiredReady->logical->bindingSubscribers[b.index].empty());
        const auto recycled = removal.AddResource({1,1,false},Binding(1004));
        const auto recycledPass = removal.AddPass({});
        const auto newToken = removal.Declare(recycledPass,recycled,{}, {2,0,1,false});
        auto fresh = removal.Build(workspace,cancelled);
        CHECK(fresh->Resolve(newToken).identity == 1004);
        CHECK(Rejects([&] { fresh->Resolve(selectedToken); }));
        CHECK(secondReady->Resolve(selectedToken).identity == 1002);
        CHECK(CheckConflicts(*fresh->executable->graph) && CheckStates(*fresh->executable->graph));
        auto dependency = typed.BeginEdit();
        auto dependencyBinding = Binding(1005);
        dependencyBinding.shape = {0,0,false};
        const auto dependencySlot = dependency.AddResource(dependencyBinding.shape,std::move(dependencyBinding));
        const auto dependencyPass = dependency.AddPass({});
        const auto dependencyToken = dependency.DeclareDependency(dependencyPass,dependencySlot,false);
        auto dependencyReady = dependency.Build(workspace,cancelled);
        CHECK(dependencyReady->Resolve(dependencyToken).identity == 1005);
        CHECK(CheckStates(*dependencyReady->executable->graph));
    }
    {
        GraphProgram resources;
        auto bootstrap = resources.BeginEdit();
        ResourceSlotId slots[2];
        for (uint32_t i = 0; i < 2; ++i) {
            auto binding = Binding(i+1);
            auto state = std::make_shared<PreparedBackingState>();
            state->graphResourceID = i+1; state->resource = {i+1,1};
            binding.identity = (uint64_t{1} << 32) | (i+1);
            state->regions = std::make_shared<const std::vector<PreparedStateRegion>>(
                std::initializer_list<PreparedStateRegion>{{{}, {static_cast<uint64_t>(rhi::ResourceAccessType::Common),0,
                    static_cast<uint64_t>(rhi::ResourceSyncState::All),false}}});
            binding.admission = state;
            slots[i] = bootstrap.AddResource(binding.shape,std::move(binding));
        }
        const auto group = bootstrap.AddGroup({1,1,false},{slots[0]});
        CompilePass pass;
        pass.accesses = {{slots[1].index,true}};
        pass.entryStates = {{slots[1].index,{}, {static_cast<uint64_t>(rhi::ResourceAccessType::UnorderedAccess),0,
            static_cast<uint64_t>(rhi::ResourceSyncState::ComputeShading),true}}};
        bootstrap.AddPass(pass);
        auto held = bootstrap.Build(workspace,cancelled);
        CHECK(resources.Install(bootstrap,held));
        auto referenced = resources.BeginEdit();
        CHECK(Rejects([&] { referenced.RemoveResource(slots[1]); }));
        auto member = resources.BeginEdit();
        CHECK(Rejects([&] { member.RemoveResource(slots[0]); }));
        auto removal = resources.BeginEdit();
        removal.ReplaceGroupMembers(group,{});
        removal.RemoveResource(slots[0]);
        auto removed = removal.Build(workspace,cancelled);
        CHECK(resources.Install(removal,removed));
        CHECK(Rejects([&] { removed->bindings.At(slots[0]); }));
        CHECK(held->bindings.At(slots[0]).owner);
        CHECK(removed->executable->resourceSlots.size() == 1);
        CHECK(removed->executable->resourceSlots[0] == slots[1]);
        CHECK(removed->executable->graph->structure->passes[0].accesses[0].resourceIndex == 0);
        CHECK(CheckConflicts(*removed->executable->graph) && CheckStates(*removed->executable->graph));
        SynchronousAdmission admission;
        const std::array<ExecutionTimelinePoint,1> queues{{{1,0}}};
        auto frame = admission.Prepare(removed,queues);
        CHECK(frame.backings.size() == 1 && frame.backings[0].graphResourceID == 2);
        admission.Abandon(frame);
        auto invalid = resources.BeginEdit();
        pass.accesses = {{slots[0].index,true}};
        pass.entryStates[0].resource = slots[0].index;
        invalid.AddPass(pass);
        CHECK(Rejects([&] { invalid.Build(workspace,cancelled); }));
        CHECK(resources.Select() == removed);
        auto reuse = resources.BeginEdit();
        auto replacementBinding = held->bindings.At(slots[0]);
        auto replacementState = std::make_shared<PreparedBackingState>(*replacementBinding.admission);
        replacementState->resource.generation = 2;
        replacementBinding.identity = (uint64_t{2} << 32) | 1;
        replacementBinding.admission = replacementState;
        replacementBinding.owner = std::make_shared<const uint64_t>(replacementBinding.identity);
        const auto recycled = reuse.AddResource(replacementBinding.shape,std::move(replacementBinding));
        CHECK(recycled.index == slots[0].index && recycled.generation == slots[0].generation + 1);
        auto reused = reuse.Build(workspace,cancelled);
        CHECK(resources.Install(reuse,reused));
        CHECK(reused->bindings.Size() == 2);
        CHECK(Rejects([&] { reused->bindings.At(slots[0]); }));
        CHECK(held->bindings.At(slots[0]).admission->resource.generation == 1);
        frame = admission.Prepare(reused,queues);
        CHECK(frame.backings[0].resource.generation == 2);
        admission.Abandon(frame);
        auto staleReplacement = resources.BeginEdit();
        CHECK(Rejects([&] { staleReplacement.ReplaceBinding(slots[0],Binding(999)); }));
        auto staleMembership = resources.BeginEdit();
        CHECK(Rejects([&] { staleMembership.ReplaceGroupMembers(group,{slots[0]}); }));
        auto current = recycled;
        for (uint32_t i = 0; i < 128; ++i) {
            auto churn = resources.BeginEdit();
            churn.RemoveResource(current);
            const auto next = churn.AddResource({1,1,false},Binding(2000+i));
            CHECK(next.index == current.index && next.generation == current.generation+1);
            auto ready = churn.Build(workspace,cancelled);
            CHECK(resources.Install(churn,ready));
            CHECK(ready->bindings.Size() == 2);
            CHECK(CheckStates(*ready->executable->graph));
            current = next;
        }
    }
    {
        GraphProgram retiredPasses;
        auto bootstrap = retiredPasses.BeginEdit();
        const auto slot = bootstrap.AddResource({1,1,false},Binding(901));
        const auto group = bootstrap.AddGroup({1,1,false},{slot});
        const auto producer = bootstrap.AddPass({});
        const auto middle = bootstrap.AddPass({});
        const auto terminal = bootstrap.AddPass({});
        bootstrap.DeclareGroupAccess(producer,group,{1,0,1,true},0);
        bootstrap.DeclareGroupAccess(middle,group,{2,0,1,false},1);
        bootstrap.DeclareGroupAccess(terminal,group,{2,0,1,false},2);
        bootstrap.AddOrdering(producer,middle);
        bootstrap.AddPlacementOrdering(middle,terminal);
        auto held = bootstrap.Build(workspace,cancelled);
        CHECK(retiredPasses.Install(bootstrap,held));
        auto removal = retiredPasses.BeginEdit();
        removal.RemovePass(middle);
        auto removed = removal.Build(workspace,cancelled);
        CHECK(retiredPasses.Install(removal,removed));
        CHECK(removed->executable->graph->structure->passes.size() == 2);
        CHECK(removed->executable->graph->structure->passes[1].preparedPassIndex == terminal.index);
        CHECK(removed->executable->executionLayout->placements.size() == 3);
        CHECK(removed->executable->executionLayout->placements[middle.index].preparedPass == UINT32_MAX);
        CHECK(removed->executable->executionLayout->placements[terminal.index].preparedPass == terminal.index);
        CHECK(removed->logical->groups[0].subscribers.size() == 2);
        CHECK(removed->logical->declarations.explicitEdges.empty());
        CHECK(removed->logical->declarations.placementEdges.empty());
        CHECK(held->logical->passSlots[middle.index].active);
        CHECK(held->executable->graph->structure->passes.size() == 3);
        CHECK(CheckConflicts(*removed->executable->graph) && CheckStates(*removed->executable->graph));
        auto reuse = retiredPasses.BeginEdit();
        const auto replacement = reuse.AddPass({});
        CHECK(replacement.index == middle.index && replacement.generation == middle.generation + 1);
        reuse.DeclareGroupAccess(replacement,group,{2,0,1,false},1);
        auto replaced = reuse.Build(workspace,cancelled);
        CHECK(retiredPasses.Install(reuse,replaced));
        CHECK(CheckConflicts(*replaced->executable->graph) && CheckStates(*replaced->executable->graph));
        auto stale = retiredPasses.BeginEdit();
        CHECK(Rejects([&] { stale.AddOrdering(producer,middle); }));
        CHECK(Rejects([&] { stale.Build(workspace,cancelled); }));
        auto selfCycle = retiredPasses.BeginEdit();
        CHECK(Rejects([&] { selfCycle.AddOrdering(producer,producer); }));
        auto selfPlacement = retiredPasses.BeginEdit();
        CHECK(Rejects([&] { selfPlacement.AddPlacementOrdering(producer,producer); }));
        auto all = retiredPasses.BeginEdit();
        all.RemovePass(producer); all.RemovePass(replacement); all.RemovePass(terminal);
        auto empty = all.Build(workspace,cancelled);
        CHECK(empty->executable->graph->batches.empty());
        CHECK(CheckStates(*empty->executable->graph));
    }
    {
        GraphProgram grouped;
        auto transaction = grouped.BeginEdit();
        transaction.SetQueues({{0,true},{0,true}});
        const auto a = transaction.AddResource({1,1,false},Binding(501));
        const auto b = transaction.AddResource({1,1,false},Binding(502));
        const auto c = transaction.AddResource({1,1,false},Binding(503));
        CompilePass producerPass, consumerPass;
        consumerPass.compatibleQueueSlots = {1}; consumerPass.preferredQueueSlot = 1;
        const auto producerId = transaction.AddPass(producerPass);
        const auto consumerId = transaction.AddPass(consumerPass);
        const auto group = transaction.AddGroup({1,1,false},{a,b});
        transaction.DeclareGroupAccess(producerId,group,{1,0,1,true},0);
        transaction.DeclareGroupAccess(consumerId,group,{2,0,1,false},1);
        auto old = transaction.Build(workspace,cancelled);
        CHECK(grouped.Install(transaction,old));
        CHECK(old->logical->declarations.passes[0].accesses.empty());
        CHECK(old->executable->graph->structure->passes[0].accesses.size() == 2);
        CHECK(CheckConflicts(*old->executable->graph) && CheckStates(*old->executable->graph));
        auto backing = grouped.BeginEdit();
        backing.ReplaceBinding(a,Binding(601));
        auto compatible = backing.Build(workspace,cancelled);
        CHECK(compatible->logical == old->logical && compatible->executable == old->executable);
        CHECK(grouped.Install(backing,compatible));
        auto membership = grouped.BeginEdit();
        membership.ReplaceGroupMembers(group,{c});
        auto changed = membership.Build(workspace,cancelled);
        CHECK(changed->logical->groups[0].membershipRevision == old->logical->groups[0].membershipRevision + 1);
        CHECK(changed->executable->graph->structure->passes[0].accesses.size() == 1);
        CHECK(changed->executable->graph->structure->passes[0].accesses[0].resourceIndex == c.index);
        CHECK(old->logical->groups[0].members.size() == 2);
        CHECK(CheckConflicts(*changed->executable->graph) && CheckStates(*changed->executable->graph));
        CHECK(grouped.Install(membership,changed));
        auto unchanged = grouped.BeginEdit();
        unchanged.ReplaceGroupMembers(group,{c});
        auto unchangedReady = unchanged.Build(workspace,cancelled);
        CHECK(unchangedReady->logical == changed->logical && unchangedReady->executable == changed->executable);
        auto unsubscribe = grouped.BeginEdit();
        unsubscribe.RemoveGroupAccess(consumerId,group);
        auto producerOnly = unsubscribe.Build(workspace,cancelled);
        CHECK(producerOnly->executable->graph->structure->passes[1].accesses.empty());
        CHECK(changed->logical->groups[0].subscribers.size() == 2);
        CHECK(CheckStates(*producerOnly->executable->graph));
        auto incompatible = grouped.BeginEdit();
        auto shapeChanged = Binding(701);
        shapeChanged.shape = {2,1,false};
        incompatible.ReplaceResourceContract(c,shapeChanged.shape,shapeChanged);
        CHECK(Rejects([&] { incompatible.Build(workspace,cancelled); }));
        auto duplicate = grouped.BeginEdit();
        CHECK(Rejects([&] { duplicate.ReplaceGroupMembers(group,{a,a}); }));
        CHECK(Rejects([&] { duplicate.Build(workspace,cancelled); }));
        auto stale = grouped.BeginEdit();
        CHECK(Rejects([&] { stale.ReplaceGroupMembers({group.index,2},{a}); }));
        auto overlap = grouped.BeginEdit();
        const auto other = overlap.AddGroup({1,1,false},{c});
        overlap.DeclareGroupAccess(consumerId,other,{1,0,1,true},0);
        CHECK(Rejects([&] { overlap.Build(workspace,cancelled); }));
        CHECK(grouped.Select() == changed);
        auto orderedOverlap = grouped.BeginEdit();
        const auto orderedGroup = orderedOverlap.AddGroup({1,1,false},{c});
        orderedOverlap.DeclareGroupAccess(consumerId,orderedGroup,{2,0,1,false},0);
        orderedOverlap.AddOrdering(producerId,consumerId);
        auto orderedReady = orderedOverlap.Build(workspace,cancelled);
        CHECK(CheckConflicts(*orderedReady->executable->graph) && CheckStates(*orderedReady->executable->graph));
        auto empty = grouped.BeginEdit();
        empty.ReplaceGroupMembers(group,{});
        auto emptyReady = empty.Build(workspace,cancelled);
        CHECK(emptyReady->executable->graph->structure->passes[0].accesses.empty());
        CHECK(CheckStates(*emptyReady->executable->graph));
        auto removal = grouped.BeginEdit();
        removal.RemoveGroup(group);
        auto removed = removal.Build(workspace,cancelled);
        CHECK(grouped.Install(removal,removed));
        CHECK(removed->executable->graph->structure->passes[0].accesses.empty());
        auto reuse = grouped.BeginEdit();
        const auto recycled = reuse.AddGroup({1,1,false},{a});
        CHECK(recycled.index == group.index && recycled.generation == group.generation + 1);
        reuse.DeclareGroupAccess(producerId,recycled,{1,0,1,true},0);
        auto reused = reuse.Build(workspace,cancelled);
        CHECK(CheckStates(*reused->executable->graph));
        CHECK(old->logical->groups[group.index].active);
        auto staleGroup = grouped.BeginEdit();
        CHECK(Rejects([&] { staleGroup.RemoveGroup(group); }));
    }
    {
        GraphProgram reversedRegistration;
        auto transaction = reversedRegistration.BeginEdit();
        auto slot = transaction.AddResource({1,1,false},Binding(801));
        auto consumerId = transaction.AddPass({});
        auto producerId = transaction.AddPass({});
        auto group = transaction.AddGroup({1,1,false},{slot});
        transaction.DeclareGroupAccess(consumerId,group,{2,0,1,false},1);
        transaction.DeclareGroupAccess(producerId,group,{1,0,1,true},0);
        auto ready = transaction.Build(workspace,cancelled);
        CHECK(ready->executable->graph->batches[0].passes[0] == producerId.index);
        CHECK(CheckConflicts(*ready->executable->graph) && CheckStates(*ready->executable->graph));
        const std::array<uint32_t,2> invalidOrder{{0,0}};
        CHECK(Rejects([&] { workspace.AnalyzeDependencies(ready->logical->declarations,cancelled,invalidOrder); }));
    }
    std::mt19937 rng(0x53415250);
    for (uint32_t iteration = 0; iteration < 100; ++iteration) {
        GraphProgram randomProgram;
        auto transaction = randomProgram.BeginEdit();
        transaction.SetQueues({{0,true},{0,true}});
        for (uint32_t i = 0; i < 8; ++i) transaction.AddResource({1,1,false}, Binding(i+1));
        for (uint32_t i = 0; i < 16; ++i) {
            CompilePass pass;
            const uint32_t resource = rng()%8;
            const bool write = (rng()%3) == 0;
            pass.accesses = {{resource,write}};
            pass.preferredQueueSlot = rng()%2;
            pass.compatibleQueueSlots = {pass.preferredQueueSlot};
            pass.entryStates = {{resource,{}, {write ? 1ull : 2ull,0,1,write}}};
            transaction.AddPass(std::move(pass));
        }
        auto randomReady = transaction.Build(workspace,cancelled);
        CHECK(CheckConflicts(*randomReady->executable->graph));
        CHECK(CheckStates(*randomReady->executable->graph));
    }
    {
        std::ifstream captured(std::string(ORG_REPLAY_FIXTURE_DIR) + "/radius20_graph_snapshot.orggraph");
        GraphCompileInput input;
        input.structure = ReadGraphReplay(captured);
        NormalizeCompileInput(input);
        auto owned = std::make_shared<const GraphCompileInput>(std::move(input));
        auto graph = CompileGraph(owned,workspace,cancelled);
        CHECK(graph && graph->states.complete);
        CHECK(CheckConflicts(*graph));
        CHECK(CheckStates(*graph));
        CHECK(ValidateSymbolicSchedule(*owned,*graph).empty());
        CHECK(ValidateSymbolicStates(*owned,*graph).empty());
    }
    {
        GraphProgram native;
        auto edit = native.BeginEdit();
        auto binding = Binding(1);
        binding.identity = (uint64_t{1} << 32) | 71;
        auto snapshot = std::make_shared<org::ResourceBindingSnapshot>();
        snapshot->resource = rhi::Resource(rhi::ResourceHandle{71,1});
        snapshot->backingGeneration = 1; snapshot->allocationOwner = binding.owner;
        snapshot->description.type = rhi::ResourceType::Buffer;
        snapshot->description.buffer.sizeBytes = 128;
        snapshot->description.resourceFlags = rhi::ResourceFlags::RF_AllowUnorderedAccess;
        binding.recording = snapshot;
        const auto slot = edit.AddResource(binding.shape,binding);
        const auto pass = edit.AddPass({});
        const auto token = edit.Declare(pass,slot,{}, {2,0,1,false});
        auto held = edit.Build(workspace,cancelled);
        CHECK(native.Install(held));
        CHECK(held->logical->nativeContracts[slot.index]->minimumBufferBytes == 128);
        uint32_t nextNativeHandle = 72;
        auto changed = [&](auto mutate) {
            const auto handle = nextNativeHandle++;
            auto version = held->bindings.At(slot);
            auto recording = std::make_shared<org::ResourceBindingSnapshot>(*version.recording);
            recording->resource = rhi::Resource(rhi::ResourceHandle{handle,1});
            mutate(recording->description);
            version.identity = (uint64_t{1} << 32) | handle; version.recording = recording;
            return version;
        };
        for (auto bad : {
            changed([](auto& desc) { desc.buffer.sizeBytes = 64; }),
            changed([](auto& desc) { desc.buffer.sizeBytes = 256; }),
            changed([](auto& desc) { desc.type = rhi::ResourceType::Unknown; }),
            changed([](auto& desc) { desc.heapType = rhi::HeapType::Upload; }),
            changed([](auto& desc) { desc.resourceFlags = {}; })}) {
            auto rejected = native.BeginEdit();
            CHECK(Rejects([&] { rejected.ReplaceBinding(slot,bad); }));
            CHECK(native.Select() == held);
        }
        auto compatible = native.BeginEdit();
        compatible.ReplaceBinding(slot,changed([](auto&) {}));
        auto sameSize = compatible.Build(workspace,cancelled);
        CHECK(sameSize->executable == held->executable);
        CHECK(native.Install(sameSize));
        auto contract = *held->logical->nativeContracts[slot.index];
        contract.maximumBufferBytes = 512;
        auto widen = native.BeginEdit();
        widen.SetNativeBindingContract(slot,contract);
        auto widened = widen.Build(workspace,cancelled);
        CHECK(native.Install(widened));
        auto growth = native.BeginEdit();
        growth.ReplaceBinding(slot,changed([](auto& desc) { desc.buffer.sizeBytes = 256; }));
        auto grown = growth.Build(workspace,cancelled);
        CHECK(grown->executable == widened->executable && native.Install(grown));
        CHECK(grown->Resolve(token).recording->description.buffer.sizeBytes == 256);
        CHECK(held->Resolve(token).recording->description.buffer.sizeBytes == 128);
        auto impossibleResize = native.BeginEdit();
        auto samePhysical = grown->bindings.At(slot);
        auto impossibleSnapshot = std::make_shared<org::ResourceBindingSnapshot>(*samePhysical.recording);
        impossibleSnapshot->description.buffer.sizeBytes = 300;
        samePhysical.recording = impossibleSnapshot;
        CHECK(Rejects([&] { impossibleResize.ReplaceBinding(slot,samePhysical); }));
        auto tooLarge = native.BeginEdit();
        CHECK(Rejects([&] { tooLarge.ReplaceBinding(slot,changed([](auto& desc) { desc.buffer.sizeBytes = 1024; })); }));
        auto invalid = native.BeginEdit(); contract.minimumBufferBytes = 1024;
        CHECK(Rejects([&] { invalid.SetNativeBindingContract(slot,contract); }));
        rhi::ResourceDesc texture{};
        texture.type = rhi::ResourceType::Texture2D;
        texture.texture = {rhi::Format::R32_Float,64,32,3,2,1};
        const rhi::Format formats[]{rhi::Format::R32_Float,rhi::Format::R32_UInt};
        texture.castableFormats = formats;
        auto textureContract = NativeBindingContract::Capture(texture);
        CHECK(textureContract.Matches(texture,{2,3,true}));
        texture.texture.initialLayout = rhi::ResourceLayout::Common;
        CHECK(textureContract.Matches(texture,{2,3,true}));
        texture.texture.width = 128;
        CHECK(!textureContract.Matches(texture,{2,3,true}));
        texture.texture.width = 64; texture.texture.format = rhi::Format::R32_UInt;
        CHECK(!textureContract.Matches(texture,{2,3,true}));
        texture.texture.format = rhi::Format::R32_Float; texture.texture.sampleCount = 4;
        CHECK(!textureContract.Matches(texture,{2,3,true}));
        texture.texture.sampleCount = 1; texture.castableFormats = {};
        CHECK(!textureContract.Matches(texture,{2,3,true}));
        texture.castableFormats = formats;
        CHECK(!textureContract.Matches(texture,{1,3,true}));
        CHECK(!textureContract.Matches(texture,{2,1,true}));
        CHECK(!textureContract.Matches(texture,{2,3,false}));
        CHECK(Rejects([&] { NativeBindingContract::Capture(rhi::ResourceDesc{}); }));
        // Retiring/reusing a stable slot must not inherit the old native contract.
        auto retire = native.BeginEdit(); retire.RemovePass(pass); retire.RemoveResource(slot);
        auto retired = retire.Build(workspace,cancelled); CHECK(native.Install(retired));
        CHECK(retired->executable->graph->states.complete);
        CHECK(ValidateSymbolicSchedule(*retired->executable->compileInput,*retired->executable->graph).empty());
        CHECK(ValidateSymbolicStates(*retired->executable->compileInput,*retired->executable->graph).empty());
        auto corruptedEmpty = *retired->executable->graph;
        corruptedEmpty.relativeWaits.push_back({0,0});
        CHECK(!ValidateSymbolicSchedule(*retired->executable->compileInput,corruptedEmpty).empty());
        auto reuse = native.BeginEdit(); const auto recycled = reuse.AddResource({1,1,false},Binding(99));
        auto reused = reuse.Build(workspace,cancelled);
        CHECK(recycled.index == slot.index && recycled.generation != slot.generation);
        CHECK(!reused->logical->nativeContracts[slot.index]);
    }
    {
        // Native description/view validation occurs on publication edits, before
        // declarations or frame recording can observe an incompatible version.
        auto publish = [&](rhi::ResourceDesc desc, org::BindlessViewKind kind, bool mismatch = false) {
            GraphProgram graph;
            auto edit = graph.BeginEdit();
            auto binding = Binding(1);
            binding.identity = (uint64_t{1} << 32) | 91;
            binding.shape = desc.type == rhi::ResourceType::Buffer ? CompileResourceShape{1,1,false} : CompileResourceShape{1,1,true};
            auto snapshot = std::make_shared<org::ResourceBindingSnapshot>();
            snapshot->resource = rhi::Resource(rhi::ResourceHandle{91,1});
            snapshot->backingGeneration = 1; snapshot->allocationOwner = binding.owner; snapshot->descriptorOwner = binding.owner;
            snapshot->description = desc;
            auto views = std::make_shared<org::BindlessResourceViews>();
            views->description = desc;
            if (mismatch) views->description.buffer.sizeBytes *= 2;
            views->views.push_back({kind,UINT32_MAX,0,0,rhi::DescriptorSlot(rhi::DescriptorHeapHandle{3,1},7)});
            snapshot->views = views; binding.recording = snapshot;
            const auto slot = edit.AddResource(binding.shape,std::move(binding));
            const auto pass = edit.AddPass({});
            auto token = edit.Declare(pass,slot,{}, {2,0,1,false});
            edit.DeclareView(token,{kind});
            auto ready = edit.Build(workspace,cancelled);
            if (!graph.Install(ready)) throw std::logic_error("Failed native view test installation");
        };
        rhi::ResourceDesc buffer{};
        buffer.type = rhi::ResourceType::Buffer; buffer.buffer.sizeBytes = 4096;
        publish(buffer,org::BindlessViewKind::ConstantBuffer);
        publish(buffer,org::BindlessViewKind::ShaderResource);
        CHECK(Rejects([&] { publish(buffer,org::BindlessViewKind::UnorderedAccess); }));
        CHECK(Rejects([&] { publish(buffer,org::BindlessViewKind::NonShaderVisibleUnorderedAccess); }));
        CHECK(Rejects([&] { publish(buffer,org::BindlessViewKind::RenderTarget); }));
        CHECK(Rejects([&] { publish(buffer,org::BindlessViewKind::DepthStencil); }));
        CHECK(Rejects([&] { publish(buffer,org::BindlessViewKind::ShaderResource,true); }));
        buffer.resourceFlags = rhi::ResourceFlags::RF_AllowUnorderedAccess;
        publish(buffer,org::BindlessViewKind::UnorderedAccess);
        publish(buffer,org::BindlessViewKind::NonShaderVisibleUnorderedAccess);
        buffer.resourceFlags = rhi::ResourceFlags::RF_DenyShaderResource;
        CHECK(Rejects([&] { publish(buffer,org::BindlessViewKind::ShaderResource); }));
        rhi::ResourceDesc texture{};
        texture.type = rhi::ResourceType::Texture2D; texture.texture = {rhi::Format::R32_Float,16,16,1,1,1};
        publish(texture,org::BindlessViewKind::ShaderResource);
        CHECK(Rejects([&] { publish(texture,org::BindlessViewKind::ConstantBuffer); }));
        texture.resourceFlags = rhi::ResourceFlags::RF_AllowRenderTarget;
        publish(texture,org::BindlessViewKind::RenderTarget);
        texture.resourceFlags = rhi::ResourceFlags::RF_AllowDepthStencil;
        publish(texture,org::BindlessViewKind::DepthStencil);
        texture.resourceFlags = rhi::ResourceFlags::RF_AllowUnorderedAccess;
        publish(texture,org::BindlessViewKind::UnorderedAccess);
        texture.texture.sampleCount = 4;
        CHECK(Rejects([&] { publish(texture,org::BindlessViewKind::UnorderedAccess); }));
        buffer.resourceFlags = {};
        CHECK(Rejects([&] { publish(buffer,static_cast<org::BindlessViewKind>(255)); }));
    }
    {
        // Reserved capacity: membership within capacity is a binding edit, unbound
        // slots produce no admission work, and a reserved slot can be rebound per
        // frame without touching the publication.
        GraphProgram program; CompileWorkspace workspace; std::atomic_bool cancelled{false};
        auto edit = program.BeginEdit();
        const auto group = edit.AddGroup({1,1,false},{});
        const auto reserved = edit.ReserveGroupMembers(group,4);
        CHECK(reserved.size() == 4);
        const auto swapchain = edit.ReserveResource({1,1,false});
        const auto consumer = edit.AddPass({});
        edit.DeclareGroupAccess(consumer,group,{2,0,1,false},0);
        const auto present = edit.AddPass({});
        edit.Declare(present,swapchain,{},{static_cast<uint64_t>(rhi::ResourceAccessType::RenderTarget),0,1,true});
        auto makeSnapshot = [](uint32_t index) {
            auto snapshot = std::make_shared<org::ResourceBindingSnapshot>();
            snapshot->resource = rhi::Resource(rhi::ResourceHandle{index,1}); snapshot->backingGeneration = 1;
            snapshot->allocationOwner = std::make_shared<const uint32_t>(index);
            return snapshot;
        };
        auto makeBinding = [&](uint32_t index, ResourceSlotId slot) {
            PreparedBackingState initial;
            initial.graphResourceID = uint64_t{slot.index} + 1; initial.resource = {index,1}; initial.shape = {1,1,false};
            initial.regions = std::make_shared<const std::vector<PreparedStateRegion>>();
            return BindingVersion::FromSnapshot(makeSnapshot(index),initial);
        };
        edit.BindReserved(reserved[0],makeBinding(100,reserved[0]));
        auto base = edit.Build(workspace,cancelled); CHECK(program.Install(edit,base));
        CHECK(base->executable->resourceSlots.size() == 5);
        CHECK(CheckConflicts(*base->executable->graph) && CheckStates(*base->executable->graph));
        auto membership = program.BeginEdit();
        membership.BindReserved(reserved[1],makeBinding(101,reserved[1]));
        membership.BindReserved(reserved[2],makeBinding(102,reserved[2]));
        membership.Unbind(reserved[0]);
        auto rotated = membership.Build(workspace,cancelled); CHECK(program.Install(membership,rotated));
        CHECK(rotated->executable == base->executable);
        CHECK(!rotated->bindings.At(reserved[0]).bound && rotated->bindings.At(reserved[1]).bound);
        CHECK(base->bindings.At(reserved[0]).bound);
        CHECK(Rejects([&] { auto stale = program.BeginEdit(); stale.Unbind(reserved[0]); }));
        CHECK(Rejects([&] { auto stale = program.BeginEdit(); stale.BindReserved(reserved[1],makeBinding(103,reserved[1])); }));
        SynchronousAdmission admission;
        const std::array<ExecutionTimelinePoint,1> queues{{{1,0}}};
        FrameRebinding rebinding;
        rebinding.slot = swapchain;
        rebinding.backing.graphResourceID = uint64_t{swapchain.index} + 1;
        rebinding.backing.resource = {200,1}; rebinding.backing.shape = {1,1,false};
        rebinding.backing.regions = std::make_shared<const std::vector<PreparedStateRegion>>();
        rebinding.recording = makeSnapshot(200);
        auto frame = admission.Prepare(rotated,queues,{},{},std::span{&rebinding,1});
        CHECK(frame.rebound.size() == 1);
        const auto swapchainIndex = rotated->executable->resourceIndexBySlot.at(swapchain.index);
        CHECK(frame.backings[swapchainIndex].resource.index == 200);
        size_t seeded = 0;
        for (const auto& batch : frame.barriers.batches) seeded += batch.seeds.size();
        CHECK(seeded == 3); // Two bound members plus the rebound swapchain; unbound slots are skipped.
        GraphExecutionTimeline receipt; receipt.batches.resize(frame.barriers.batches.size());
        for (size_t i = 0; i < receipt.batches.size(); ++i) receipt.batches[i].signal = {1,i+1};
        std::vector<org::PreparedPass> invocations(rotated->logical->passSlots.size());
        invocations[consumer.index] = org::PreparedPass::NoOp();
        invocations[present.index] = org::PreparedPass::NoOp();
        auto sealed = org::experimental::SealPersistentFrame(1,frame,std::move(invocations));
        CHECK(sealed && sealed->initialStates->at(swapchainIndex).resource.index == 200);
        admission.Commit(frame,receipt);
        CHECK(admission.StateBackingCount() == 3);
        auto bad = rebinding; bad.slot = reserved[1];
        CHECK(Rejects([&] { admission.Prepare(rotated,queues,{},{},std::span{&bad,1}); }));
        admission.ExtendSubmitted({1,50});
        CHECK(admission.Submitted(1) == 50);
        auto grow = program.BeginEdit();
        const auto more = grow.ReserveGroupMembers(group,4);
        CHECK(more.size() == 4);
        auto grown = grow.Build(workspace,cancelled); CHECK(program.Install(grow,grown));
        CHECK(grown->executable != rotated->executable && grown->executable->resourceSlots.size() == 9);
        CHECK(CheckConflicts(*grown->executable->graph) && CheckStates(*grown->executable->graph));
    }
    {
        // Schedule-only builds skip alias-order validation so a planner can
        // derive lifetimes from the schedule; a validated build still rejects
        // overlapping placements whose users are unordered, and clearing the
        // placement orderings re-exposes that.
        GraphProgram program; CompileWorkspace workspace; std::atomic_bool cancelled{false};
        auto heapOwner = std::make_shared<const uint64_t>(1);
        auto heap = std::shared_ptr<const org::AliasHeapGeneration>(heapOwner,
            reinterpret_cast<const org::AliasHeapGeneration*>(heapOwner.get()));
        auto placed = [&](uint64_t index, uint64_t offset) {
            auto binding = Binding(index); binding.identity = (uint64_t{1} << 32) | index;
            PreparedBackingState backing; backing.graphResourceID = index; backing.resource = {static_cast<uint32_t>(index),1};
            backing.shape = binding.shape;
            backing.regions = std::make_shared<const std::vector<PreparedStateRegion>>(
                std::initializer_list<PreparedStateRegion>{{{0,1,0,1},{0,0,0,false}}});
            backing.aliasHeap = heap; backing.aliasHeapIdentity = heap.get();
            backing.aliasOffset = offset; backing.aliasSize = 256;
            binding.admission = std::make_shared<const PreparedBackingState>(backing);
            return binding;
        };
        auto edit = program.BeginEdit();
        edit.SetQueues({{0,true},{0,true}});
        const auto x = edit.AddResource({1,1,false},placed(1,0));
        const auto y = edit.AddResource({1,1,false},placed(2,128));
        const CompileResourceState write{static_cast<uint64_t>(rhi::ResourceAccessType::UnorderedAccess),
            static_cast<uint64_t>(rhi::ResourceLayout::UnorderedAccess),static_cast<uint64_t>(rhi::ResourceSyncState::ComputeShading),true};
        CompilePass onSecondQueue; onSecondQueue.compatibleQueueSlots = {1}; onSecondQueue.preferredQueueSlot = 1;
        const auto a = edit.AddPass({}); edit.Declare(a,x,{0,1,0,1},write);
        const auto b = edit.AddPass(onSecondQueue); edit.Declare(b,y,{0,1,0,1},write);
        CHECK(Rejects([&] { edit.Build(workspace,cancelled); }));
        auto scheduled = edit.Build(workspace,cancelled,false);
        CHECK(scheduled && scheduled->executable->graph->batches.size() == 2);
        edit.AddPlacementOrdering(a,b);
        auto ordered = edit.Build(workspace,cancelled);
        CHECK(ordered && program.Install(edit,ordered));
        CHECK(ordered->logical->declarations.placementEdges.size() == 1);
        auto cleared = program.BeginEdit();
        cleared.ClearPlacementOrderings();
        CHECK(Rejects([&] { cleared.Build(workspace,cancelled); }));
        auto reordered = program.BeginEdit();
        reordered.ClearPlacementOrderings();
        reordered.AddPlacementOrdering(b,a);
        auto flipped = reordered.Build(workspace,cancelled);
        CHECK(flipped && program.Install(reordered,flipped));
        CHECK(flipped->logical->declarations.placementEdges.size() == 1 && flipped->logical->declarations.placementEdges[0].first == b.index);
        CHECK(CheckConflicts(*flipped->executable->graph) && CheckStates(*flipped->executable->graph));
    }
    std::puts("Persistent graph tests passed");
}
