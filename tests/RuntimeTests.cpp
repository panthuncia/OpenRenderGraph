#include "OpenRenderGraph/Runtime.h"
#include "Managers/Singletons/DeletionManager.h"
#include "TestSupport.h"

#include <rhi_helpers.h>

// Retirement by submission frontier: an object released while earlier GPU work is still pending outlives that work, and goes
// at the first Retire after it completes - whichever queue the work and the release are on.
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
