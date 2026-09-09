#include "../src/Render/RenderGraph/FrameRecording.h"
#include <array>
#include <future>
#include <thread>

#define CHECK(x) do { if (!(x)) return __LINE__; } while (false)
using namespace org::experimental;

namespace {
struct RecordingProbe {
    std::array<std::promise<void>,2> entered, release;
    std::array<std::shared_future<void>,2> gates{release[0].get_future(),release[1].get_future()};
    std::atomic_uint active{0}, peak{0};
    std::atomic_bool recordedOnHost{false};
    std::thread::id host = std::this_thread::get_id();
    bool failFirst = false;
};
struct ProbeData { std::shared_ptr<RecordingProbe> probe; uint32_t index; };
void RecordProbe(const ProbeData& data, org::RecordingContext&) {
    auto& probe = *data.probe;
    if (std::this_thread::get_id() == probe.host) probe.recordedOnHost = true;
    const auto active = probe.active.fetch_add(1) + 1;
    auto peak = probe.peak.load();
    while (peak < active && !probe.peak.compare_exchange_weak(peak,active)) {}
    probe.entered[data.index].set_value();
    probe.gates[data.index].wait();
    probe.active.fetch_sub(1);
    if (!data.index && probe.failFirst) throw std::runtime_error("Injected frame recording failure");
}
struct Runtime {
    rhi::DevicePtr device;
    rhi::TimelinePtr timeline;
};
struct Work {
    std::shared_ptr<const PlannedFrameState> state;
    PlannedFrame recording;
};
Work MakeWork(FramePlanningState& planner, org::FrameSlotPool& slots, uint64_t sequence,
    uint32_t slot, const std::shared_ptr<Runtime>& runtime, const std::shared_ptr<RecordingProbe>& probe) {
    GraphCompileInput input;
    input.structure.generation = input.structure.registryGeneration = 1;
    input.structure.resourceIDs = {1}; input.structure.resourceShapes = {{1,1,false}};
    CompileResourceState common{static_cast<uint64_t>(rhi::ResourceAccessType::Common),0,
        static_cast<uint64_t>(rhi::ResourceSyncState::All),false};
    CompilePass pass; pass.preparedPassIndex = 0; pass.accesses = {{0,false}}; pass.entryStates = {{0,{},common}};
    input.structure.passes = {pass};
    input.frameContext = std::make_shared<org::FrameContext>(sequence,1,slots.TryAcquire(slot));
    input.frameContext->Retain(runtime);
    input.frameContext->Advance(org::FrameStage::Preparing,org::FrameStage::Compiling);
    NormalizeCompileInput(input);
    auto owned = std::make_shared<const GraphCompileInput>(std::move(input));
    CompileWorkspace workspace; std::atomic_bool cancelled{false};
    auto graph = CompileGraph(owned,workspace,cancelled);
    auto bundle = std::make_shared<const CompiledGraphBundle>(CompiledGraphBundle{sequence,graph,owned});
    PreparedBackingState backing; backing.graphResourceID = 1; backing.resource = {1,1};
    backing.shape = {1,1,false}; backing.regions = {{{},common}};
    auto bindings = std::make_shared<const org::FrozenExecutionBindings>(std::vector<org::FrozenExecutionBindings::ResourceBinding>{});
    auto packet = org::PreparedPass::Make(ProbeData{probe,slot},RecordProbe);
    auto payload = BuildPreparedFramePayload(sequence,{packet},bindings,{backing},{},{runtime},slot);
    const std::array queues{ExecutionTimelinePoint{1,0}};
    auto state = planner.Plan(bundle,payload,queues);
    const auto& barriers = state->snapshot->barrierPlan->batches[0];
    // The synthetic COMMON read is only a graph declaration. These no-op GPU
    // lists must never reference the synthetic handle in an encoded barrier.
    if (!barriers.buffers.empty() || !barriers.textures.empty()
        || !barriers.beforePass[0].buffers.empty() || !barriers.beforePass[0].textures.empty())
        throw std::logic_error("No-op test unexpectedly needs a native backing");
    auto device = runtime->device.Get();
    FrameRecordingJob job; job.device = device; job.queue = device.GetQueue(rhi::QueueKind::Graphics);
    job.pool = std::make_shared<org::CommandListPool>(device,rhi::QueueKind::Graphics);
    job.recording.bindings = bindings; job.recording.passes = {packet};
    PlannedFrame plan; plan.snapshot = state->snapshot; plan.planning = state;
    plan.timelines = {{1,runtime->timeline.Get().GetHandle()}}; plan.incomingWaits.resize(1);
    plan.jobs.push_back(std::move(job));
    return {state,std::move(plan)};
}
}

int TestDelayedFrameRecording(const rhi::DeviceCreateInfo& create) {
    auto runtime = std::make_shared<Runtime>();
    CHECK(rhi::CreateD3D12Device(create,runtime->device) == rhi::Result::Ok);
    CHECK(runtime->device.Get().CreateTimeline(runtime->timeline,0,"Frame recording FIFO test") == rhi::Result::Ok);
    FramePlanningState planner(2); org::FrameSlotPool slots(2);
    ExecutionTimelineAdmission admission({{1,0}},2);
    for (unsigned failureRun = 0; failureRun != 2; ++failureRun) {
        auto probe = std::make_shared<RecordingProbe>(); probe->failFirst = failureRun != 0;
        auto entered0 = probe->entered[0].get_future(), entered1 = probe->entered[1].get_future();
        auto first = MakeWork(planner,slots,1 + failureRun * 2,0,runtime,probe);
        auto second = MakeWork(planner,slots,2 + failureRun * 2,1,runtime,probe);
        auto record = [](PlannedFrame plan) { return RecordFrame(std::move(plan),{},1); };
        auto worker0 = std::async(std::launch::async,record,std::move(first.recording));
        auto worker1 = std::async(std::launch::async,record,std::move(second.recording));
        CHECK(entered0.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
        CHECK(entered1.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
        CHECK(probe->peak == 2 && !probe->recordedOnHost);
        CHECK(!slots.TryAcquire(0) && !slots.TryAcquire(1));
        probe->release[1].set_value();
        std::optional<RecordedFrame> ready1(worker1.get());
        bool rejected = false;
        try { std::move(*ready1).Submit(admission); } catch (const std::logic_error&) { rejected = true; }
        CHECK(rejected && admission.Submitted()[0].value == failureRun * 2);
        probe->release[0].set_value();
        if (failureRun) {
            bool failed = false;
            try { worker0.get(); } catch (const std::runtime_error&) { failed = true; }
            CHECK(failed && probe->active == 0);
            planner.CancelUnsubmittedSuffixAfterJoin();
            CHECK(first.state->cancelled && second.state->cancelled);
            rejected = false;
            try { std::move(*ready1).Submit(admission); } catch (const std::logic_error&) { rejected = true; }
            CHECK(rejected && admission.Submitted()[0].value == 2 && planner.SymbolCount() == 0);
        } else {
            std::optional<RecordedFrame> ready0(worker0.get());
            auto receipt0 = std::move(*ready0).Submit(admission);
            planner.Confirm(first.state,*receipt0);
            auto receipt1 = std::move(*ready1).Submit(admission);
            planner.Confirm(second.state,*receipt1);
            CHECK(receipt0->submission == 1 && receipt1->submission == 2);
            CHECK(runtime->timeline.Get().HostWait(2,10000) == rhi::Result::Ok);
            ready0.reset(); ready1.reset(); receipt0.reset(); receipt1.reset();
            CHECK(admission.RetireCompleted(std::array{ExecutionTimelinePoint{1,2}}) == 2);
        }
        ready1.reset(); first.state.reset(); second.state.reset();
        CHECK(slots.Active() == 0);
    }
    return 0;
}
