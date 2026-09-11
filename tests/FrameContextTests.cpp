#include "Render/RenderGraph/FrameContext.h"

#include <array>
#include <barrier>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define CHECK(...) do { if (!(__VA_ARGS__)) { std::fprintf(stderr, "Check failed at %d: %s\n", __LINE__, #__VA_ARGS__); std::abort(); } } while (false)

using namespace org;

int main() {
    FrameSlotPool slots(2);
    auto first = std::make_shared<FrameContext>(1, 1, slots.TryAcquire(0));
    runtime::OpenRenderGraphSettings acceptedSettings;
    acceptedSettings.useAsyncCompute = false;
    acceptedSettings.queueSchedulingWidthScale = 7.5f;
    acceptedSettings.autoAliasPoolRetireIdleFrames = 91;
    auto second = std::make_shared<FrameContext>(
        2, 1, slots.TryAcquire(1), true, 4, acceptedSettings);
    CHECK(slots.Active() == 2);
    CHECK(slots.Capacity() == 2);
    CHECK(!first->AsynchronousScheduling() && first->CompileConcurrency() == 2);
    CHECK(second->AsynchronousScheduling() && second->CompileConcurrency() == 4);
    acceptedSettings.useAsyncCompute = true;
    acceptedSettings.queueSchedulingWidthScale = 1.0f;
    acceptedSettings.autoAliasPoolRetireIdleFrames = 1;
    CHECK(!second->AcceptedSettings().useAsyncCompute);
    CHECK(second->AcceptedSettings().queueSchedulingWidthScale == 7.5f);
    CHECK(second->AcceptedSettings().autoAliasPoolRetireIdleFrames == 91);
    CHECK(!slots.TryAcquire(0)); // No GPU fence exists yet; CPU work owns the slot.
    auto oldBacking = std::make_shared<int>(42);
    auto oldPipeline = std::make_shared<int>(17);
    std::weak_ptr<int> backing = oldBacking, pipeline = oldPipeline;
    first->Retain(oldBacking);
    first->Retain(oldPipeline);
    oldBacking = std::make_shared<int>(43);
    oldPipeline = std::make_shared<int>(18);
    CHECK(!backing.expired() && !pipeline.expired());

    for (auto stage : {FrameStage::Preparing, FrameStage::Compiling, FrameStage::Planned, FrameStage::Recording})
        first->Advance(stage, static_cast<FrameStage>(static_cast<unsigned>(stage) + 1));
    CompletionSet completion;
    completion.Include({1, 7});
    completion.Include({2, 11});
    completion.Include({1, 5});
    CHECK(completion.Points().size() == 2);
    first->MarkSubmitted(std::move(completion));
    CHECK(!first->Retire(std::array{FrameCompletionPoint{1, 7}, FrameCompletionPoint{2, 10}}));
    CHECK(!first->Retire(std::array{FrameCompletionPoint{1, 7}, FrameCompletionPoint{2, UINT64_MAX}}));
    CHECK(!first->Retire(std::array{FrameCompletionPoint{3, 999}}));
    CHECK(!backing.expired() && !pipeline.expired());
    bool rejected = false;
    try { first->CancelAfterJoin(); } catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected);
    CHECK(first->Retire(std::array{FrameCompletionPoint{1, 7}, FrameCompletionPoint{2, 11}}));
    first.reset();
    CHECK(backing.expired() && pipeline.expired());
    CHECK(slots.TryAcquire(0));

    // Cancellation does not release a slot while an outstanding job owns it.
    std::barrier rendezvous(2);
    std::thread worker([held = second, &rendezvous] {
        rendezvous.arrive_and_wait();
        rendezvous.arrive_and_wait();
    });
    rendezvous.arrive_and_wait();
    second.reset();
    CHECK(!slots.TryAcquire(1));
    rendezvous.arrive_and_wait();
    worker.join();
    CHECK(slots.Active() == 0);
    auto cancelled = std::make_shared<FrameContext>(3, 2, slots.TryAcquire(1));
    cancelled->CancelAfterJoin();
    CHECK(cancelled->Stage() == FrameStage::Cancelled);
    cancelled.reset();
    CHECK(slots.TryAcquire(1));
    auto uncertain = std::make_shared<FrameContext>(4, 2, slots.TryAcquire(0));
    for (auto stage : {FrameStage::Preparing, FrameStage::Compiling, FrameStage::Planned, FrameStage::Recording})
        uncertain->Advance(stage, static_cast<FrameStage>(static_cast<unsigned>(stage) + 1));
    uncertain->EnterRecovery();
    rejected = false;
    try { uncertain->CancelAfterJoin(); } catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected && !slots.TryAcquire(0));
    CHECK(!uncertain->Retire(std::array{FrameCompletionPoint{1, UINT64_MAX - 1}}));
    std::puts("Frame slot and multi-queue ownership tests passed.");
}
