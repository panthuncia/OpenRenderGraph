#include "Managers/Singletons/DescriptorHeapManager.h"
#include <Render/RenderGraph/RenderGraph.h>
#include <Render/PassExecutionContext.h>
#include <memory>
#include <cstdio>
#include <atomic>
#include <cstring>
#include <thread>
#include <future>
#include <chrono>
#include "Render/BindingTable.h"

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Retirement check failed at %d: %s\n", __LINE__, #x); return __LINE__; } } while (false)

namespace {
struct RetirementChain {
    std::shared_ptr<RetirementChain> next;
    std::shared_ptr<int> destroyed;
    std::shared_ptr<int> payload;
    ~RetirementChain() {
        ++*destroyed;
        if (next) org::DescriptorHeapManager::GetInstance().RetireExecutionLease(std::move(next));
    }
};
}

int TestBindingOwnership(rhi::Device device);

int TestCompletionWake(rhi::Device device) {
    std::unique_ptr<rhi::CompletionWait> waiter;
    CHECK(device.CreateCompletionWait(waiter) == rhi::Result::Ok && waiter);
    // Wake before arming must not be lost, including when there are no GPU points.
    for (int iteration = 0; iteration < 100; ++iteration) {
        const auto observed = waiter->WakeVersion();
        auto future = std::async(std::launch::async, [&] { return waiter->Wait({}, observed); });
        CHECK(waiter->Notify() == rhi::Result::Ok);
        if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) std::abort();
        CHECK(future.get() == rhi::Result::Ok);
    }
    rhi::TimelinePtr first, second;
    CHECK(device.CreateTimeline(first, 0, "Completion wait first") == rhi::Result::Ok);
    CHECK(device.CreateTimeline(second, 0, "Completion wait second") == rhi::Result::Ok);
    const rhi::TimelinePoint points[]{{first->GetHandle(), 1}, {second->GetHandle(), 1}};
    const auto observed = waiter->WakeVersion();
    auto future = std::async(std::launch::async, [&] { return waiter->Wait({points, 2}, observed); });
    auto queue = device.GetQueue(rhi::QueueKind::Graphics);
    CHECK(queue.Signal(points[1]) == rhi::Result::Ok);
    if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) std::abort();
    CHECK(future.get() == rhi::Result::Ok);
    CHECK(first->GetCompletedValue() == 0); // Wait-any, not wait-all.
    CHECK(queue.Signal(points[0]) == rhi::Result::Ok);
    CHECK(first->HostWait(1, 5000) == rhi::Result::Ok);
    waiter.reset(); // Wait object precedes timelines in destruction order.
    return 0;
}


int TestFrameRetirement(rhi::Device device, rhi::Backend backend) {
    CHECK(TestCompletionWake(device) == 0);
    CHECK(TestBindingOwnership(device) == 0);
    {
        org::RenderGraph graph(device, backend);
        graph.StopFrameProduction();
        graph.StopFrameProduction();
        org::UpdateExecutionContext update{};
        org::PassExecutionContext execute{};
        bool rejectedUpdate = false, rejectedExecute = false;
        try { graph.Update(update, device); } catch (const std::logic_error&) { rejectedUpdate = true; }
        try { graph.Execute(execute); } catch (const std::logic_error&) { rejectedExecute = true; }
        CHECK(rejectedUpdate && rejectedExecute);

        auto& retirement = org::DescriptorHeapManager::GetInstance();
        retirement.Initialize();
        rhi::TimelinePtr descriptorDone;
        CHECK(device.CreateTimeline(descriptorDone, 0, "Owned descriptor retirement") == rhi::Result::Ok);
        const auto ownedSlot = retirement.AllocateDescriptorSlot(rhi::DescriptorHeapType::CbvSrvUav, true);
        auto cpuLease = retirement.GetCBVSRVUAVHeap()->CaptureDescriptorLease(ownedSlot.index);
        auto ownedView = std::make_shared<int>(17);
        std::weak_ptr<int> ownedViewWeak = ownedView;
        retirement.PublishQueueFenceSnapshot({{descriptorDone.Get(), 1}});
        retirement.RetireDescriptorSlotWithOwner(ownedSlot, std::move(ownedView));
        retirement.ProcessDeferredReleases(0);
        CHECK(!ownedViewWeak.expired());
        auto descriptorQueue = device.GetQueue(rhi::QueueKind::Graphics);
        CHECK(descriptorQueue.Signal({descriptorDone.Get().GetHandle(), 1}) == rhi::Result::Ok);
        CHECK(descriptorDone.Get().HostWait(1, 10000) == rhi::Result::Ok);
        retirement.ProcessDeferredReleases(0);
        CHECK(!ownedViewWeak.expired()); // GPU completion alone cannot release a CPU-retained owner.
        cpuLease.reset();
        CHECK(ownedViewWeak.expired());
        CHECK(device.WaitIdle() == rhi::Result::Ok);
        retirement.DrainDeferredReleasesAfterDeviceIdle();
    }
    auto& retirement = org::DescriptorHeapManager::GetInstance();
    CHECK(device.WaitIdle() == rhi::Result::Ok);
    retirement.DrainDeferredReleasesAfterDeviceIdle();
    rhi::TimelinePtr required, unrelated;
    CHECK(device.CreateTimeline(required, 0, "Required frame retirement") == rhi::Result::Ok);
    CHECK(device.CreateTimeline(unrelated, 0, "Unrelated frame retirement") == rhi::Result::Ok);
    auto owner = std::make_shared<int>(42);
    std::weak_ptr<int> weak = owner;
    retirement.RetireExecutionLease(std::move(owner), {{required.Get(), 7}});
    auto queue = device.GetQueue(rhi::QueueKind::Graphics);
    CHECK(queue.Signal({unrelated.Get().GetHandle(), 99}) == rhi::Result::Ok);
    CHECK(unrelated.Get().HostWait(99, 10000) == rhi::Result::Ok);
    retirement.ProcessDeferredReleases(0);
    CHECK(!weak.expired() && retirement.GetDeferredReleaseStats().blockedReleaseCount == 1);
    CHECK(queue.Signal({required.Get().GetHandle(), 7}) == rhi::Result::Ok);
    CHECK(required.Get().HostWait(7, 10000) == rhi::Result::Ok);
    retirement.ProcessDeferredReleases(0);
    CHECK(weak.expired() && retirement.GetDeferredReleaseStats().releaseCount == 0);

    {
        auto heap = std::make_shared<org::DescriptorHeap>(device,
            rhi::DescriptorHeapType::CbvSrvUav, 1, true, "Final lease callback");
        const auto index = heap->AllocateDescriptor();
        // A lease can disappear while the slot stays allocated, then be
        // reacquired by a successor. The later lease must prevent recycling.
        auto lease = heap->CaptureDescriptorLease(index);
        lease.reset();
        lease = heap->CaptureDescriptorLease(index);
        auto otherConsumer = heap->CaptureDescriptorLease(index);
        auto backing = std::make_shared<int>(123);
        std::weak_ptr<int> backingWeak = backing;
        heap->ReleaseDescriptor(index, std::move(backing));
        lease.reset();
        CHECK(!backingWeak.expired());
        bool full = false;
        try { (void)heap->AllocateDescriptor(); } catch (const std::runtime_error&) { full = true; }
        CHECK(full);
        otherConsumer.reset();
        CHECK(backingWeak.expired()); // Callback ran before any allocation or polling.
        CHECK(heap->AllocateDescriptor() == index);
        heap->ReleaseDescriptor(index);
    }

    {
        auto heap = std::make_shared<org::DescriptorHeap>(device,
            rhi::DescriptorHeapType::CbvSrvUav, 1, true, "Retained bindless slot");
        const auto slot = heap->AllocateDescriptor();
        auto queuedFrame = heap->CaptureDescriptorLease(slot);
        retirement.RetireDescriptorSlots({{heap, slot}});
        retirement.ProcessDeferredReleases(0);
        auto unavailable = [&] {
            try { (void)heap->AllocateDescriptor(); }
            catch (const std::runtime_error&) { return true; }
            return false;
        };
        CHECK(unavailable()); // No frame submission fence exists yet.
        retirement.RetireExecutionLease(std::move(queuedFrame), {{required.Get(), 9}});
        CHECK(queue.Signal({unrelated.Get().GetHandle(), 101}) == rhi::Result::Ok);
        CHECK(unrelated.Get().HostWait(101, 10000) == rhi::Result::Ok);
        retirement.ProcessDeferredReleases(0);
        CHECK(unavailable());
        CHECK(queue.Signal({required.Get().GetHandle(), 9}) == rhi::Result::Ok);
        CHECK(required.Get().HostWait(9, 10000) == rhi::Result::Ok);
        retirement.ProcessDeferredReleases(0);
        CHECK(heap->AllocateDescriptor() == slot);
        heap->ReleaseDescriptor(slot);
    }

    // Owners may release further owners, which in turn retire descriptors or
    // backings. Cleanup must drain all waves while its arenas are still alive.
    auto destroyed = std::make_shared<int>(0);
    auto payload = std::make_shared<int>(9);
    weak = payload;
    std::shared_ptr<RetirementChain> chain;
    for (int i = 0; i < 5; ++i) {
        auto link = std::make_shared<RetirementChain>();
        link->next = std::move(chain); link->destroyed = destroyed; link->payload = payload;
        chain = std::move(link);
    }
    payload.reset();
    retirement.RetireExecutionLease(std::move(chain));
    CHECK(device.WaitIdle() == rhi::Result::Ok);
    retirement.Cleanup();
    CHECK(*destroyed == 5 && weak.expired());
    CHECK(retirement.GetDeferredReleaseStats().releaseCount == 0);
    retirement.Cleanup(); // Shutdown is idempotent.
    CHECK(retirement.GetDeferredReleaseStats().releaseCount == 0);
    return 0;
}
