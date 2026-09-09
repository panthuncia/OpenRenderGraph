#include "../src/Render/RenderGraph/FramePlanning.h"
#include "../src/Render/RenderGraph/FrameWorker.h"
#include <cstdio>
#include <cstdlib>

using namespace org::experimental;
#define CHECK(...) do { if (!(__VA_ARGS__)) { std::fprintf(stderr, "Planning check failed at %d: %s\n", __LINE__, #__VA_ARGS__); std::abort(); } } while (false)

namespace {
template<class F> bool Rejects(F&& f) { try { f(); } catch (const std::exception&) { return true; } return false; }
struct InputFrame {
    std::shared_ptr<const CompiledGraphBundle> bundle;
    std::shared_ptr<const PreparedFramePayload> payload;
};
InputFrame MakeFrame(uint64_t sequence, uint32_t queue, bool write) {
    GraphCompileInput input;
    input.structure.generation = input.structure.registryGeneration = 1;
    input.structure.resourceIDs = {1};
    input.structure.resourceShapes = {{1,1,false}};
    input.structure.queues = {{0,true},{0,true}};
    CompilePass pass;
    pass.preparedPassIndex = 0;
    pass.compatibleQueueSlots = {queue}; pass.preferredQueueSlot = queue;
    pass.accesses = {{0,write}};
    pass.entryStates = {{0,{}, {static_cast<uint64_t>(write ? rhi::ResourceAccessType::UnorderedAccess
        : rhi::ResourceAccessType::ShaderResource),0,static_cast<uint64_t>(rhi::ResourceSyncState::ComputeShading),write}}};
    input.structure.passes = {pass};
    NormalizeCompileInput(input);
    auto owned = std::make_shared<const GraphCompileInput>(std::move(input));
    CompileWorkspace workspace; std::atomic_bool cancelled{false};
    auto graph = CompileGraph(owned, workspace, cancelled);
    auto bundle = std::make_shared<const CompiledGraphBundle>(CompiledGraphBundle{sequence, graph, owned});
    auto bindings = std::make_shared<const org::FrozenExecutionBindings>(std::vector<org::FrozenExecutionBindings::ResourceBinding>{});
    PreparedBackingState backing;
    backing.graphResourceID = 1; backing.resource = {1,1}; backing.shape = {1,1,false};
    backing.regions = {{{}, {static_cast<uint64_t>(rhi::ResourceAccessType::Common),0,
        static_cast<uint64_t>(rhi::ResourceSyncState::All),false}}};
    auto payload = BuildPreparedFramePayload(sequence,
        {org::PreparedPass::Make(0,+[](const int&,org::RecordingContext&) {})}, bindings, {backing});
    return {bundle, payload};
}
GraphExecutionTimeline Receipt(const InputFrame& input, uint64_t timeline, uint64_t value) {
    GraphExecutionTimeline result; result.bundle = input.bundle;
    result.batches.resize(1); result.batches[0].signal = {timeline,value};
    return result;
}
struct BoundaryTasks : org::runtime::ITaskService {
    enum Mode { Worker, Inline, Drop, Reject } mode = Worker;
    struct Scope : org::runtime::ITaskScope {
        std::vector<std::jthread> threads;
        void Cancel() noexcept override {}
        void Wait() override { threads.clear(); }
        void CancelAndWait() override { Wait(); }
    };
    void ParallelFor(std::string_view,size_t n,std::function<void(size_t)> fn) override {
        for (size_t i = 0; i < n; ++i) fn(i);
    }
    void ParallelForLimited(std::string_view name,size_t n,size_t,std::function<void(size_t)> fn) override {
        ParallelFor(name,n,std::move(fn));
    }
    std::shared_ptr<org::runtime::ITaskScope> CreateScope(std::string_view) override { return std::make_shared<Scope>(); }
    bool Submit(const std::shared_ptr<org::runtime::ITaskScope>& scope,org::runtime::TaskPriority,
        std::string_view,std::function<void()>&& fn) override {
        if (mode == Reject) return false;
        if (mode == Inline) fn();
        if (mode == Worker) std::static_pointer_cast<Scope>(scope)->threads.emplace_back(std::move(fn));
        return true;
    }
    bool ScheduleAfter(const std::shared_ptr<org::runtime::ITaskScope>&,
        std::chrono::steady_clock::duration,org::runtime::TaskPriority,std::string_view,std::function<void()>&&) override { return false; }
};
}

void RunFramePlanningTests() {
    // No silent inline fallback or helping wait can put Async callbacks back
    // on the host. Dropped jobs break their promise instead of hanging shutdown.
    auto tasks = std::make_shared<BoundaryTasks>();
    std::shared_ptr<org::runtime::ITaskScope> scope;
    const auto host = std::this_thread::get_id();
    CHECK(RunFrameWorker(tasks,scope,true,[] { return std::this_thread::get_id(); }) != host);
    scope->Wait();
    CHECK(RunFrameWorker(tasks,scope,false,[] { return std::this_thread::get_id(); }) == host);
    tasks->mode = BoundaryTasks::Inline;
    CHECK(Rejects([&] { RunFrameWorker(tasks,scope,true,[] { return 1; }); }));
    tasks->mode = BoundaryTasks::Drop;
    CHECK(Rejects([&] { RunFrameWorker(tasks,scope,true,[] { return 1; }); }));
    tasks->mode = BoundaryTasks::Reject;
    CHECK(Rejects([&] { RunFrameWorker(tasks,scope,true,[] { return 1; }); }));
    const std::vector<ExecutionTimelinePoint> queues{{10,0},{20,0}};
    // RAW followed by WAR: future waits identify a predecessor batch, not a
    // guessed absolute fence. Real values deliberately contain unrelated gaps.
    FramePlanningState planner(3);
    auto a = MakeFrame(1,0,true), b = MakeFrame(2,1,false), c = MakeFrame(3,0,true);
    CHECK(Rejects([&] { planner.Plan(b.bundle,b.payload,queues); }));
    auto pa = planner.Plan(a.bundle,a.payload,queues);
    auto pb = planner.Plan(b.bundle,b.payload,queues);
    auto pc = planner.Plan(c.bundle,c.payload,queues);
    CHECK(planner.Pending() == 3 && planner.SymbolCount() == 3);
    CHECK(pb->dependencies[0].size() == 1 && pb->dependencies[0][0].predecessor->frameSequence == 1);
    CHECK(pc->dependencies[0].size() == 1 && pc->dependencies[0][0].predecessor->frameSequence == 2);
    CHECK(Rejects([&] { pb->ResolveWaits(); })); // FIFO even if recording B finishes first.
    CHECK(Rejects([&] { planner.Confirm(pb, Receipt(b,20,91)); }));
    planner.Confirm(pa,Receipt(a,10,37));
    CHECK(pb->ResolveWaits()[0] == std::vector<ExecutionTimelinePoint>{{10,37}});
    CHECK(planner.SymbolCount() == 2);
    planner.Confirm(pb,Receipt(b,20,91));
    CHECK(pc->ResolveWaits()[0] == std::vector<ExecutionTimelinePoint>{{20,91}});
    planner.Confirm(pc,Receipt(c,10,103));
    CHECK(planner.Pending() == 0 && planner.SymbolCount() == 0);

    // Same-queue order also enforces FIFO when no explicit queue wait exists.
    FramePlanningState sameQueue(2);
    auto same = MakeFrame(2,0,false);
    auto p1 = sameQueue.Plan(a.bundle,a.payload,queues);
    auto p2 = sameQueue.Plan(same.bundle,same.payload,queues);
    CHECK(p2->dependencies[0].empty() && Rejects([&] { p2->ResolveWaits(); }));
    sameQueue.Confirm(p1,Receipt(a,10,7));
    CHECK(p2->ResolveWaits()[0].empty());

    // Cancel all unsubmitted successors and rebuild from confirmed state.
    // Discarded read/write states and their symbolic fence values cannot leak.
    FramePlanningState cancel(3);
    auto pca = cancel.Plan(a.bundle,a.payload,queues);
    cancel.Confirm(pca,Receipt(a,10,43));
    auto pcb = cancel.Plan(b.bundle,b.payload,queues);
    auto pcc = cancel.Plan(c.bundle,c.payload,queues);
    cancel.CancelUnsubmittedSuffixAfterJoin();
    CHECK(pcb->cancelled && pcc->cancelled && cancel.SymbolCount() == 0);
    CHECK(Rejects([&] { pcb->ResolveWaits(); }));
    auto replacement = cancel.Plan(b.bundle,b.payload,queues);
    CHECK(replacement->ResolveWaits()[0] == std::vector<ExecutionTimelinePoint>{{10,43}});
    CHECK(!replacement->dependencies[0][0].predecessor);
    cancel.Confirm(replacement,Receipt(b,20,71));
    auto next = MakeFrame(3,0,true);
    auto pn = cancel.Plan(next.bundle,next.payload,queues);
    CHECK(pn->ResolveWaits()[0] == std::vector<ExecutionTimelinePoint>{{20,71}});

    FramePlanningState stateCancel(2);
    auto committed = stateCancel.Plan(a.bundle,a.payload,queues);
    stateCancel.Confirm(committed,Receipt(a,10,17));
    stateCancel.Plan(same.bundle,same.payload,queues);
    stateCancel.CancelUnsubmittedSuffixAfterJoin();
    auto rebuilt = stateCancel.Plan(same.bundle,same.payload,queues);
    const auto& before = rebuilt->snapshot->barrierPlan->batches[0].beforePass[0].buffers;
    CHECK(!before.empty() && before[0].beforeAccess == rhi::ResourceAccessType::UnorderedAccess);
    CHECK(before[0].afterAccess == rhi::ResourceAccessType::ShaderResource);

    // Even read/read access conflicts for distinct physical alias occupants.
    // Cancelling B must restore A as the confirmed occupant of the interval.
    auto aliasPayload = [](const InputFrame& frame, uint32_t handle, uint64_t offset) {
        auto backing = frame.payload->initialStates;
        backing[0].resource = {handle,1};
        backing[0].aliasHeapIdentity = reinterpret_cast<const org::AliasHeapGeneration*>(uintptr_t{1});
        backing[0].aliasOffset = offset; backing[0].aliasSize = 256;
        return BuildPreparedFramePayload(frame.payload->frameNumber,frame.payload->passes,
            frame.payload->bindings,std::move(backing));
    };
    FramePlanningState aliases(2);
    auto aliasA = MakeFrame(1,0,false), aliasB = MakeFrame(2,1,false), aliasAgain = MakeFrame(2,0,false);
    auto paliasA = aliases.Plan(aliasA.bundle,aliasPayload(aliasA,11,0),queues);
    auto paliasB = aliases.Plan(aliasB.bundle,aliasPayload(aliasB,12,128),queues);
    CHECK(paliasB->invalidated.size() == 1 && paliasB->dependencies[0].size() == 1);
    CHECK(paliasB->dependencies[0][0].predecessor->frameSequence == 1);
    aliases.Confirm(paliasA,Receipt(aliasA,10,29));
    aliases.CancelUnsubmittedSuffixAfterJoin();
    auto paliasAgain = aliases.Plan(aliasAgain.bundle,aliasPayload(aliasAgain,11,0),queues);
    CHECK(paliasAgain->invalidated.empty() && paliasAgain->dependencies[0].empty());

    // Reusing an old alias occupant must seed its newly activated state even
    // though its handle still exists in the historical backing-state ledger.
    FramePlanningState reactivate(2);
    auto ra = reactivate.Plan(aliasA.bundle,aliasPayload(aliasA,11,0),queues);
    reactivate.Confirm(ra,Receipt(aliasA,10,31));
    auto rb = reactivate.Plan(aliasB.bundle,aliasPayload(aliasB,12,128),queues);
    reactivate.Confirm(rb,Receipt(aliasB,20,32));
    auto aliasThird = MakeFrame(3,0,false);
    auto rc = reactivate.Plan(aliasThird.bundle,aliasPayload(aliasThird,11,0),queues);
    CHECK(rc->invalidated.size() == 1 && !rc->snapshot->barrierPlan->batches[0].seeds.empty());
    reactivate.Confirm(rc,Receipt(aliasThird,10,33));

    // WAW across queues and uncertain submission stay recovery-owned.
    FramePlanningState failure(2);
    auto writer = MakeFrame(2,1,true);
    auto fa = failure.Plan(a.bundle,a.payload,queues);
    auto fb = failure.Plan(writer.bundle,writer.payload,queues);
    CHECK(fb->dependencies[0][0].predecessor->frameSequence == 1);
    failure.ConfirmFailure(fa,Receipt(a,10,55),0);
    CHECK(failure.RecoveryRequired());
    CHECK(fa->signals[0]->submittedValue == 0);
    CHECK(Rejects([&] { failure.CancelUnsubmittedSuffixAfterJoin(); }));
    CHECK(Rejects([&] { failure.Plan(c.bundle,c.payload,queues); }));
    CHECK(Rejects([&] { fb->ResolveWaits(); }));
    std::puts("Ordered frame planning, symbolic waits, suffix cancellation and recovery tests passed.");
}
