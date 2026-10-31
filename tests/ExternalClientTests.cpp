#include "OpenRenderGraph/ExternalClient.h"
#include "OpenRenderGraph/Runtime.h"
#include "Render/RenderGraph/GraphCompiler.h"
#include "Resources/Buffers/Buffer.h"
#include "TestSupport.h"

#include <cstring>

// Two external client submissions on different queues, ordered through the runtime's ledgers:
// 1. a copy into a buffer on the graphics queue,
// 2. a copy out of that buffer on the copy queue. It must wait for the first copy, then reads back its data.
int TestExternalClient(rhi::Device device) {
    auto* runtime = org::Runtime::Current();
    CHECK(runtime);
    constexpr uint64_t kBytes = 256;
    auto source = org::Buffer::CreateShared(rhi::HeapType::Upload, kBytes);
    auto target = org::Buffer::CreateShared(rhi::HeapType::DeviceLocal, kBytes);
    auto readback = org::Buffer::CreateShared(rhi::HeapType::Readback, kBytes);
    {
        auto allocation = source->CaptureBackingAllocation();
        CHECK(allocation);
        void* mapped = nullptr;
        allocation.resource.Map(&mapped, 0, kBytes);
        CHECK(mapped);
        std::memset(mapped, 0x5a, kBytes);
        allocation.resource.Unmap(0, kBytes);
    }
    org::ExternalClient client({.runtime = runtime->shared_from_this(),
        .queues = {{device.GetQueue(rhi::QueueKind::Graphics), rhi::QueueKind::Graphics},
                   {device.GetQueue(rhi::QueueKind::Compute), rhi::QueueKind::Compute},
                   {device.GetQueue(rhi::QueueKind::Copy), rhi::QueueKind::Copy}},
        .name = "test.external"});
    const org::ResourceState copyDest{rhi::ResourceAccessType::CopyDest, rhi::ResourceLayout::CopyDest, rhi::ResourceSyncState::Copy};
    const org::ResourceState copySource{rhi::ResourceAccessType::CopySource, rhi::ResourceLayout::CopySource, rhi::ResourceSyncState::Copy};

    const std::array<org::ExternalClient::Access, 1> upload{{{target.get(), {}, copyDest}}};
    const auto uploaded = client.Submit(0, upload, [&](rhi::CommandList& list) {
        list.CopyBufferRegion(target->GetAPIResource().GetHandle(), 0, source->GetAPIResource().GetHandle(), 0, kBytes);
    });
    CHECK(uploaded.value == 1);
    const std::array<org::ExternalClient::Access, 2> download{{{target.get(), {}, copySource}, {readback.get(), {}, copyDest}}};
    const auto downloaded = client.Submit(2, download, [&](rhi::CommandList& list) {
        list.CopyBufferRegion(readback->GetAPIResource().GetHandle(), 0, target->GetAPIResource().GetHandle(), 0, kBytes);
    });
    CHECK(downloaded.value == 1);
    CHECK(rhi::Timeline(downloaded.timeline).HostWait(downloaded.value) == rhi::Result::Ok);
    {
        void* mapped = nullptr;
        readback->GetAPIResource().Map(&mapped, 0, kBytes);
        CHECK(mapped);
        const auto* bytes = static_cast<const uint8_t*>(mapped);
        bool same = true;
        for (uint64_t i = 0; i < kBytes; ++i) same = same && bytes[i] == 0x5a;
        readback->GetAPIResource().Unmap(0, 0);
        CHECK(same);
    }
    // The same upload again: it reuses the earlier publication (or at least its schedule), so nothing compiles.
    const auto missesBefore = runtime->Schedules()->GetStats().misses;
    const auto again = client.Submit(0, upload, [&](rhi::CommandList& list) {
        list.CopyBufferRegion(target->GetAPIResource().GetHandle(), 0, source->GetAPIResource().GetHandle(), 0, kBytes);
    });
    CHECK(again.value == 2);
    CHECK(runtime->Schedules()->GetStats().misses == missesBefore);
    // The same resource listed twice, ahead of another one: the repeated access shares one slot.
    const std::array<org::ExternalClient::Access, 3> repeated{{{target.get(), {}, copyDest}, {target.get(), {}, copyDest},
        {readback.get(), {}, copyDest}}};
    const auto both = client.Submit(0, repeated, [&](rhi::CommandList& list) {
        list.CopyBufferRegion(target->GetAPIResource().GetHandle(), 0, source->GetAPIResource().GetHandle(), 0, kBytes);
        list.CopyBufferRegion(readback->GetAPIResource().GetHandle(), 0, source->GetAPIResource().GetHandle(), 0, kBytes);
    });
    CHECK(both.value == 3);
    client.Retire();
    return 0;
}
