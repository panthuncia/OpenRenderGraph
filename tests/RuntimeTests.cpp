#include "OpenRenderGraph/Runtime.h"
#include "Managers/Singletons/DeletionManager.h"
#include "Managers/Singletons/StatisticsManager.h"
#include "TestSupport.h"

#include <rhi_helpers.h>

#include <atomic>
#include <cstdio>
#include <memory>

// Tests retirement by submission frontier. An object released while earlier GPU work is still pending outlives that
// work. It is destroyed at the first Retire after the work completes. This holds whichever queue the work and the
// release are on.
int TestRuntimeRetirement(rhi::Device device) {
    auto* runtime = org::Runtime::Current();
    CHECK(runtime);
    auto& deletions = org::DeletionManager::GetInstance();
    // What earlier checks left: wait for everything submitted, then retire it.
    for (auto point : runtime->Frontier()) CHECK(point.timeline.HostWait(point.value) == rhi::Result::Ok);
    runtime->Retire();
    runtime->Retire();
    const auto before = deletions.GetStats().objectCount;

    // Graphics work held back by a timeline only the copy queue signals.
    rhi::TimelinePtr gate;
    CHECK(device.CreateTimeline(gate, 0, "test.runtime.gate") == rhi::Result::Ok);
    rhi::CommandAllocatorPtr allocator;
    rhi::CommandListPtr list;
    CHECK(device.CreateCommandAllocator(rhi::QueueKind::Graphics, allocator) == rhi::Result::Ok);
    CHECK(device.CreateCommandList(rhi::QueueKind::Graphics, allocator.Get(), list) == rhi::Result::Ok);
    list->End();
    rhi::CommandList commands = list.Get();
    const rhi::TimelinePoint wait{gate->GetHandle(), 1};
    rhi::SubmitDesc submit{};
    submit.waits = {&wait, 1};
    auto graphics = device.GetQueue(rhi::QueueKind::Graphics);
    CHECK(graphics.Submit({&commands, 1}, submit) == rhi::Result::Ok);
    auto progress = graphics.Progress();
    CHECK(progress.timeline && progress.submitted > 0);

    // Released after that submission: it must wait for it.
    rhi::TimelinePtr released;
    CHECK(device.CreateTimeline(released, 0, "test.runtime.released") == rhi::Result::Ok);
    deletions.MarkForDelete(rhi::helpers::AnyObjectPtr(std::move(released)));
    runtime->Retire();  // seals it with the frontier, which includes the held-back work
    runtime->Retire();
    CHECK(deletions.GetStats().objectCount == before + 1);

    CHECK(device.GetQueue(rhi::QueueKind::Copy).Signal({gate->GetHandle(), 1}) == rhi::Result::Ok);
    CHECK(progress.timeline.HostWait(progress.submitted) == rhi::Result::Ok);
    runtime->Retire();
    CHECK(deletions.GetStats().objectCount == before);

    list.Reset();
    allocator.Reset();
    return 0;
}

namespace {
// Waits for everything the frontier holds now, then retires it, so a test starts from an empty deletion queue.
int DrainRetirement(org::Runtime& runtime) {
    for (auto point : runtime.Frontier()) CHECK(point.timeline.HostWait(point.value) == rhi::Result::Ok);
    runtime.Retire();
    runtime.Retire();
    return 0;
}
}

// Tests the frontier of a timeline whose values are given out before the work that signals them reaches an ORG queue.
// A graph host's in-stream epochs work so: the host's own stream signals them, at its own submission. An object released
// after such a value was given out waits for it.
int TestAssignedTimelineRetirement(rhi::Device device) {
    auto* runtime = org::Runtime::Current();
    CHECK(runtime);
    auto& deletions = org::DeletionManager::GetInstance();
    if (const auto failure = DrainRetirement(*runtime)) return failure;
    const auto before = deletions.GetStats().objectCount;

    auto fence = std::make_shared<rhi::TimelinePtr>();
    CHECK(device.CreateTimeline(*fence, 0, "test.runtime.assigned") == rhi::Result::Ok);
    auto next = std::make_shared<std::atomic<uint64_t>>(1);
    runtime->RegisterAssignedTimeline(fence, next);
    // Values 1 and 2 are given out to work nothing has submitted yet.
    next->store(3, std::memory_order_release);

    rhi::TimelinePtr released;
    CHECK(device.CreateTimeline(released, 0, "test.runtime.assigned.released") == rhi::Result::Ok);
    deletions.MarkForDelete(rhi::helpers::AnyObjectPtr(std::move(released)));
    runtime->Retire();  // seals it with the frontier, which includes value 2
    runtime->Retire();
    CHECK(deletions.GetStats().objectCount == before + 1);

    // The host's stream submits and signals value 2.
    CHECK(device.GetQueue(rhi::QueueKind::Copy).Signal({(*fence)->GetHandle(), 2}) == rhi::Result::Ok);
    CHECK((*fence)->HostWait(2) == rhi::Result::Ok);
    runtime->Retire();
    CHECK(deletions.GetStats().objectCount == before);

    runtime->UnregisterAssignedTimeline(fence);
    runtime->NoteTimelineDestroyed(fence->Get());
    fence->Reset();
    return 0;
}

// Tests that the statistics' shared query pool and readback buffers outlive every list recorded with them. When passes
// register, the pool is replaced. A list recorded before that may not have reached the GPU yet. This happens with a ticket
// prepared ahead, or with an in-stream epoch's list in a host's stream. So the old objects go to the deletion queue only
// when the last recording context that holds them is released. A list whose pool was replaced while it recorded writes
// no more queries: a resolve of queries the new pool never wrote would wait on the GPU forever.
int TestStatisticsPoolRetirement(rhi::Device device) {
    auto* runtime = org::Runtime::Current();
    CHECK(runtime);
    auto& deletions = org::DeletionManager::GetInstance();
    auto& statistics = org::StatisticsManager::GetInstance();
    if (const auto failure = DrainRetirement(*runtime)) return failure;

    rhi::CommandAllocatorPtr allocator;
    rhi::CommandListPtr list;
    CHECK(device.CreateCommandAllocator(rhi::QueueKind::Graphics, allocator) == rhi::Result::Ok);
    CHECK(device.CreateCommandList(rhi::QueueKind::Graphics, allocator.Get(), list) == rhi::Result::Ok);
    rhi::CommandList commands = list.Get();
    auto queue = device.GetQueue(rhi::QueueKind::Graphics);

    const unsigned first = statistics.RegisterPass("test.statistics.retire.first");
    statistics.SetupQueryHeap();
    auto context = std::make_unique<org::QueryRecordingContext>();
    statistics.BeginQuery(first, 0, queue, commands, *context);
    if (!context->sharedQueries) {
        // Pass statistics are off in this configuration: nothing records into the shared pool.
        std::printf("TestStatisticsPoolRetirement: pass statistics are off; skipped\n");
        list->End();
        return 0;
    }
    const auto recorded = context->recordedIndices.size();
    CHECK(recorded == 1);

    // A new pass replaces the pool while the list above still names the old one.
    const auto before = deletions.GetStats().objectCount;
    (void)statistics.RegisterPass("test.statistics.retire.second");
    statistics.SetupQueryHeap();
    CHECK(deletions.GetStats().objectCount == before);

    // The list's queries stay in the old pool: nothing more is written or resolved.
    statistics.EndQuery(first, 0, queue, commands, *context);
    statistics.ResolveQueries(0, queue, commands, *context);
    CHECK(context->recordedIndices.size() == recorded);
    CHECK(context->pendingRanges.empty());
    list->End();

    // The recording is released (its submissions completed): the old objects go to the deletion queue.
    context.reset();
    CHECK(deletions.GetStats().objectCount > before);
    if (const auto failure = DrainRetirement(*runtime)) return failure;

    list.Reset();
    allocator.Reset();
    return 0;
}
