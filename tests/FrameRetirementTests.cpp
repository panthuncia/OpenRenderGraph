#include "Managers/Singletons/DescriptorHeapManager.h"
#include <Render/RenderGraph/RenderGraph.h>
#include <Render/PassExecutionContext.h>
#include <memory>
#include <cstdio>

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

int TestFrameRetirement(rhi::Device device) {
    {
        org::RenderGraph graph(device, rhi::Backend::D3D12);
        graph.StopFrameProduction();
        graph.StopFrameProduction();
        org::UpdateExecutionContext update{};
        org::PassExecutionContext execute{};
        bool rejectedUpdate = false, rejectedExecute = false;
        try { graph.Update(update, device); } catch (const std::logic_error&) { rejectedUpdate = true; }
        try { graph.Execute(execute); } catch (const std::logic_error&) { rejectedExecute = true; }
        CHECK(rejectedUpdate && rejectedExecute);
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
