// CopyQueueUploadService: worker-native uploads on the copy queue.
//
// Runs on WARP by default (--hardware selects a hardware adapter). Checks:
//  - segmented buffer uploads land byte-exact when read back on the graphics
//    queue after a wait on the ticket's timeline;
//  - tickets move Queued -> Submitted -> Completed, and cancellation before
//    the submitter claims a copy never submits it;
//  - staging pages are reused across rounds instead of created per upload,
//    and oversize uploads take a dedicated page that is not recycled;
//  - a texture zero fill overwrites every mip, in bands where a mip is larger
//    than the zero source, leaves the texture in Common and says so in its
//    state tracker, and refuses an exclusive texture.

#include <Render/Runtime/CopyQueueUploadService.h>
#include "TestSupport.h"
#include <Render/Runtime/RuntimeDevice.h>
#include <Resources/ExternalBufferResource.h>
#include <Resources/ExternalTextureResource.h>
#include <Resources/ResourceStateTracker.h>
#include <rhi.h>
#include <rhi_helpers.h>
#include <rhi_interop_dx12.h>

#include <wrl/client.h>
#include <dxgi1_6.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>


namespace {

struct Harness {
    rhi::DevicePtr device;
    rhi::Queue graphics;
    rhi::Queue copy;
    org::runtime::CopyQueueUploadService service;
};

std::shared_ptr<org::ExternalBufferResource> MakeDeviceBuffer(rhi::Device& device, uint64_t bytes, const char* name) {
    auto desc = rhi::helpers::ResourceDesc::Buffer(bytes, rhi::HeapType::DeviceLocal, {}, name);
    desc.queueSharing = rhi::QueueSharing::Concurrent;
    rhi::ResourcePtr resource;
    if (rhi::Failed(device.CreateCommittedResource(desc, resource)) || !resource) return {};
    org::ExternalBufferResource::ViewRequirements views{};
    return org::ExternalBufferResource::CreateShared(std::move(resource), bytes, views);
}

int ReadBack(Harness& h, const std::shared_ptr<org::ExternalBufferResource>& source, uint64_t bytes,
    const org::TrackedUploadTicket& ticket, std::vector<uint8_t>& out) {
    rhi::ResourcePtr readback;
    CHECK(!rhi::Failed(h.device->CreateCommittedResource(
        rhi::helpers::ResourceDesc::Buffer(bytes, rhi::HeapType::Readback, {}, "CopyQueueUploadReadback"), readback)));
    rhi::CommandAllocatorPtr allocator;
    rhi::CommandListPtr list;
    CHECK(!rhi::Failed(h.device->CreateCommandAllocator(rhi::QueueKind::Graphics, allocator)));
    CHECK(!rhi::Failed(h.device->CreateCommandList(rhi::QueueKind::Graphics, allocator.Get(), list)));
    rhi::BufferBarrier toSource{
        .buffer = source->GetAPIResource().GetHandle(),
        .beforeSync = rhi::ResourceSyncState::All,
        .afterSync = rhi::ResourceSyncState::Copy,
        .beforeAccess = rhi::ResourceAccessType::Common,
        .afterAccess = rhi::ResourceAccessType::CopySource,
    };
    list->Barriers({ .buffers = { &toSource, 1 } });
    list->CopyBufferRegion(readback->GetHandle(), 0, source->GetAPIResource().GetHandle(), 0, bytes);
    list->End();
    const rhi::CommandList lists[] = { list.Get() };
    // The consumer waits on the producer's timeline point, exactly as the
    // graph does through the published selection's GpuSubmissionSet.
    const auto owner = std::static_pointer_cast<const rhi::TimelinePtr>(ticket.timelineOwner);
    CHECK(owner && *owner);
    const rhi::TimelinePoint wait{ (*owner)->GetHandle(), ticket.timelineValue };
    CHECK(!rhi::Failed(h.graphics.Submit(lists, { .waits = { &wait, 1 } })));
    CHECK(!rhi::Failed(h.device->WaitIdle()));
    void* mapped = nullptr;
    readback->Map(&mapped, 0, bytes);
    CHECK(mapped);
    out.assign(static_cast<uint8_t*>(mapped), static_cast<uint8_t*>(mapped) + bytes);
    readback->Unmap(0, 0);
    return 0;
}

std::vector<uint8_t> Pattern(size_t bytes, uint32_t seed) {
    std::vector<uint8_t> data(bytes);
    uint32_t state = seed * 2654435761u + 12345u;
    for (size_t i = 0; i < bytes; ++i) {
        state = state * 1664525u + 1013904223u;
        data[i] = static_cast<uint8_t>(state >> 24);
    }
    return data;
}

bool WaitTicket(const std::shared_ptr<org::TrackedUploadTicket>& ticket, org::TrackedUploadTicketState state, int ms = 5000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (state == org::TrackedUploadTicketState::Completed) (void)ticket->Complete();
        const auto current = ticket->state.load(std::memory_order_acquire);
        if (current == state) return true;
        if (current == org::TrackedUploadTicketState::Cancelled) return state == org::TrackedUploadTicketState::Cancelled;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return false;
}

int TestSegmentedUpload(Harness& h) {
    constexpr uint64_t kBytes = 300 * 1024;
    auto destination = MakeDeviceBuffer(h.device.Get(), kBytes, "CopyQueueUploadTarget");
    CHECK(destination);
    const auto expected = Pattern(kBytes, 7);
    const org::StreamingUploadSegment segments[] = {
        { expected.data(), 100 * 1024 }, { expected.data() + 100 * 1024, 150 * 1024 }, { expected.data() + 250 * 1024, 50 * 1024 } };
    auto ticket = h.service.QueueBufferUpload(segments, kBytes,
        org::WorkerOwnedDestination{ destination, org::WorkerOwnedDestination::Ownership::PooledBackingLease }, 0);
    CHECK(ticket);
    CHECK(WaitTicket(ticket, org::TrackedUploadTicketState::Submitted));
    std::vector<uint8_t> actual;
    if (const auto failure = ReadBack(h, destination, kBytes, *ticket, actual)) return failure;
    CHECK(actual == expected);
    CHECK(WaitTicket(ticket, org::TrackedUploadTicketState::Completed));
    CHECK(h.service.WaitIdle(5000));
    return 0;
}

int TestOffsetAndOversize(Harness& h) {
    // A 40 MB upload exceeds the 16 MB page and takes a dedicated page; a
    // smallData upload at an offset lands in the shared page.
    constexpr uint64_t kBig = uint64_t{ 40 } << 20;
    auto destination = MakeDeviceBuffer(h.device.Get(), kBig + 4096, "CopyQueueUploadBig");
    CHECK(destination);
    const auto bigData = Pattern(static_cast<size_t>(kBig), 3);
    const auto smallData = Pattern(4096, 5);
    const org::StreamingUploadSegment bigSegment[] = { { bigData.data(), bigData.size() } };
    const org::StreamingUploadSegment smallSegment[] = { { smallData.data(), smallData.size() } };
    const auto before = h.service.GetStats();
    auto bigTicket = h.service.QueueBufferUpload(bigSegment, kBig,
        org::WorkerOwnedDestination{ destination, org::WorkerOwnedDestination::Ownership::PooledBackingLease }, 0);
    auto smallTicket = h.service.QueueBufferUpload(smallSegment, 4096,
        org::WorkerOwnedDestination{ destination, org::WorkerOwnedDestination::Ownership::PooledBackingLease }, kBig);
    CHECK(bigTicket && smallTicket);
    CHECK(WaitTicket(smallTicket, org::TrackedUploadTicketState::Submitted));
    CHECK(WaitTicket(bigTicket, org::TrackedUploadTicketState::Submitted));
    std::vector<uint8_t> actual;
    const auto& lastTicket = smallTicket->timelineValue >= bigTicket->timelineValue ? *smallTicket : *bigTicket;
    if (const auto failure = ReadBack(h, destination, kBig + 4096, lastTicket, actual)) return failure;
    CHECK(std::memcmp(actual.data(), bigData.data(), bigData.size()) == 0);
    CHECK(std::memcmp(actual.data() + kBig, smallData.data(), smallData.size()) == 0);
    CHECK(h.service.WaitIdle(10000));
    const auto after = h.service.GetStats();
    CHECK(after.pagesCreated >= before.pagesCreated + 1); // the dedicated page
    return 0;
}

int TestPageReuseAndCancel(Harness& h) {
    constexpr uint64_t kBytes = 64 * 1024;
    auto destination = MakeDeviceBuffer(h.device.Get(), kBytes * 64, "CopyQueueUploadMany");
    CHECK(destination);
    const auto data = Pattern(kBytes, 11);
    const org::StreamingUploadSegment segment[] = { { data.data(), data.size() } };
    const auto before = h.service.GetStats();
    // 64 uploads of 64 KB = 4 MB: they must share pages, not create one each.
    for (int round = 0; round < 3; ++round) {
        std::vector<std::shared_ptr<org::TrackedUploadTicket>> tickets;
        for (uint64_t i = 0; i < 64; ++i) {
            tickets.push_back(h.service.QueueBufferUpload(segment, kBytes,
                org::WorkerOwnedDestination{ destination, org::WorkerOwnedDestination::Ownership::PooledBackingLease }, i * kBytes));
            CHECK(tickets.back());
        }
        for (auto& ticket : tickets) CHECK(WaitTicket(ticket, org::TrackedUploadTicketState::Completed));
        CHECK(h.service.WaitIdle(5000));
    }
    const auto after = h.service.GetStats();
    CHECK(after.pagesCreated - before.pagesCreated <= 2);
    CHECK(after.copiesSubmitted - before.copiesSubmitted == 3 * 64);

    // Cancel before the submitter claims it: never submitted.
    auto cancelled = h.service.QueueBufferUpload(segment, kBytes,
        org::WorkerOwnedDestination{ destination, org::WorkerOwnedDestination::Ownership::PooledBackingLease }, 0);
    CHECK(cancelled);
    const bool cancelledEarly = cancelled->Cancel();
    CHECK(h.service.WaitIdle(5000));
    const auto final = h.service.GetStats();
    if (cancelledEarly) {
        CHECK(final.copiesCancelled >= after.copiesCancelled + 1);
    }
    CHECK(cancelled->state.load() == org::TrackedUploadTicketState::Cancelled ||
        cancelled->state.load() == org::TrackedUploadTicketState::Completed);
    return 0;
}

// Tests signal entries (QueueSignal). It checks that:
// - a consumer submitted before the copies waits on the external timeline's value. The wait comes before the signal.
// - the consumer reads every copy the producer queued ahead of the signal.
// - a signal with nothing to copy goes out in a batch of its own.
int TestSignalEntries(Harness& h) {
    constexpr uint64_t kBytes = 96 * 1024;
    auto destination = MakeDeviceBuffer(h.device.Get(), kBytes, "CopyQueueUploadSignalled");
    CHECK(destination);
    auto timeline = std::make_shared<rhi::TimelinePtr>();
    CHECK(!rhi::Failed(h.device->CreateTimeline(*timeline, 0, "CopyQueueUploadFrameValues")) && *timeline);
    rhi::ResourcePtr readback;
    CHECK(!rhi::Failed(h.device->CreateCommittedResource(
        rhi::helpers::ResourceDesc::Buffer(kBytes, rhi::HeapType::Readback, {}, "CopyQueueUploadSignalReadback"), readback)));
    rhi::CommandAllocatorPtr allocator;
    rhi::CommandListPtr list;
    CHECK(!rhi::Failed(h.device->CreateCommandAllocator(rhi::QueueKind::Graphics, allocator)));
    CHECK(!rhi::Failed(h.device->CreateCommandList(rhi::QueueKind::Graphics, allocator.Get(), list)));
    rhi::BufferBarrier toSource{
        .buffer = destination->GetAPIResource().GetHandle(),
        .beforeSync = rhi::ResourceSyncState::All,
        .afterSync = rhi::ResourceSyncState::Copy,
        .beforeAccess = rhi::ResourceAccessType::Common,
        .afterAccess = rhi::ResourceAccessType::CopySource,
    };
    list->Barriers({ .buffers = { &toSource, 1 } });
    list->CopyBufferRegion(readback->GetHandle(), 0, destination->GetAPIResource().GetHandle(), 0, kBytes);
    list->End();
    const rhi::CommandList lists[] = { list.Get() };
    const rhi::TimelinePoint wait{ (*timeline)->GetHandle(), 3 };
    CHECK(!rhi::Failed(h.graphics.Submit(lists, { .waits = { &wait, 1 } })));
    const auto expected = Pattern(kBytes, 17);
    // Three regions from three uploads, then the signal: the consumer sees all of them.
    for (uint64_t part = 0; part < 3; ++part) {
        const org::StreamingUploadSegment segment[] = { { expected.data() + part * (kBytes / 3), kBytes / 3 } };
        CHECK(h.service.QueueBufferUpload(segment, kBytes / 3,
            org::WorkerOwnedDestination{ destination, org::WorkerOwnedDestination::Ownership::PooledBackingLease }, part * (kBytes / 3)));
    }
    CHECK(h.service.QueueSignal(timeline, 3));
    CHECK(!rhi::Failed(h.device->WaitIdle()));
    CHECK((*timeline)->GetCompletedValue() >= 3);
    void* mapped = nullptr;
    readback->Map(&mapped, 0, kBytes);
    CHECK(mapped);
    CHECK(std::memcmp(mapped, expected.data(), kBytes) == 0);
    readback->Unmap(0, 0);
    // Nothing to copy: the signal alone.
    CHECK(h.service.QueueSignal(timeline, 4));
    CHECK((*timeline)->HostWait(4, 5000) == rhi::Result::Ok);
    // Invalid arguments.
    CHECK(!h.service.QueueSignal(timeline, 0));
    CHECK(!h.service.QueueSignal({}, 5));
    CHECK(h.service.WaitIdle(5000));
    return 0;
}

std::shared_ptr<org::ExternalTextureResource> MakeTexture(rhi::Device& device, uint32_t width, uint32_t height, uint16_t mips,
    rhi::QueueSharing sharing, const char* name) {
    auto desc = rhi::helpers::ResourceDesc::Tex2D(rhi::Format::R32G32B32A32_Float, rhi::HeapType::DeviceLocal, width, height, mips, 1, 1,
        rhi::ResourceLayout::Undefined, nullptr, {}, name);
    desc.queueSharing = sharing;
    rhi::ResourcePtr resource;
    if (rhi::Failed(device.CreateCommittedResource(desc, resource)) || !resource) return {};
    org::TextureDescription description{};
    description.format = rhi::Format::R32G32B32A32_Float;
    description.queueSharing = sharing;
    for (uint32_t mip = 0; mip < mips; ++mip) description.imageDimensions.push_back({ (std::max)(1u, width >> mip), (std::max)(1u, height >> mip), 0, 0 });
    return org::ExternalTextureResource::CreateShared(std::move(resource), description);
}

// Graphics queue: copies every mip of `texture` (from `layout`) through `list` into a readback buffer, after `wait`.
int ReadTexture(Harness& h, org::Resource& texture, uint32_t mips, rhi::ResourceLayout layout, uint8_t fill,
    const org::TrackedUploadTicket* wait, uint64_t& nonZero) {
    const auto handle = texture.GetAPIResource().GetHandle();
    std::vector<rhi::CopyableFootprint> footprints(mips);
    rhi::FootprintRangeDesc range{};
    range.texture = handle;
    range.mipCount = mips;
    const auto info = h.device->GetCopyableFootprints(range, footprints.data(), mips);
    CHECK(info.count == mips);
    rhi::ResourcePtr staging;
    CHECK(!rhi::Failed(h.device->CreateCommittedResource(rhi::helpers::ResourceDesc::Buffer(info.totalBytes,
        fill ? rhi::HeapType::Upload : rhi::HeapType::Readback, {}, "CopyQueueTextureStaging"), staging)));
    void* mapped = nullptr;
    if (fill) {
        staging->Map(&mapped, 0, info.totalBytes);
        CHECK(mapped);
        std::memset(mapped, fill, static_cast<size_t>(info.totalBytes));
        staging->Unmap(0, info.totalBytes);
    }
    rhi::CommandAllocatorPtr allocator;
    rhi::CommandListPtr list;
    CHECK(!rhi::Failed(h.device->CreateCommandAllocator(rhi::QueueKind::Graphics, allocator)));
    CHECK(!rhi::Failed(h.device->CreateCommandList(rhi::QueueKind::Graphics, allocator.Get(), list)));
    const auto access = fill ? rhi::ResourceAccessType::CopyDest : rhi::ResourceAccessType::CopySource;
    rhi::TextureBarrier barrier{};
    barrier.texture = handle;
    barrier.range = { 0, mips, 0, 1 };
    barrier.beforeSync = rhi::ResourceSyncState::All;
    barrier.afterSync = rhi::ResourceSyncState::Copy;
    barrier.beforeAccess = layout == rhi::ResourceLayout::Undefined ? rhi::ResourceAccessType::None : rhi::ResourceAccessType::Common;
    barrier.afterAccess = access;
    barrier.beforeLayout = layout;
    barrier.afterLayout = fill ? rhi::ResourceLayout::CopyDest : rhi::ResourceLayout::CopySource;
    barrier.discard = layout == rhi::ResourceLayout::Undefined;
    list->Barriers({ .textures = { &barrier, 1 } });
    for (uint32_t mip = 0; mip < mips; ++mip) {
        rhi::BufferTextureCopyFootprint copy{};
        copy.texture = handle;
        copy.buffer = staging->GetHandle();
        copy.mip = mip;
        copy.footprint = footprints[mip];
        if (fill) list->CopyBufferToTexture(copy);
        else list->CopyTextureToBuffer(copy);
    }
    // Back to Common, where a fresh texture's zero fill and the next read expect it.
    barrier.beforeSync = rhi::ResourceSyncState::Copy;
    barrier.afterSync = rhi::ResourceSyncState::All;
    barrier.beforeAccess = access;
    barrier.afterAccess = rhi::ResourceAccessType::Common;
    barrier.beforeLayout = barrier.afterLayout;
    barrier.afterLayout = rhi::ResourceLayout::Common;
    barrier.discard = false;
    list->Barriers({ .textures = { &barrier, 1 } });
    list->End();
    const rhi::CommandList lists[] = { list.Get() };
    rhi::TimelinePoint point{};
    if (wait) {
        const auto owner = std::static_pointer_cast<const rhi::TimelinePtr>(wait->timelineOwner);
        CHECK(owner && *owner);
        point = { (*owner)->GetHandle(), wait->timelineValue };
    }
    CHECK(!rhi::Failed(h.graphics.Submit(lists, { .waits = { &point, wait ? 1u : 0u } })));
    CHECK(!rhi::Failed(h.device->WaitIdle()));
    nonZero = 0;
    if (fill) return 0;
    staging->Map(&mapped, 0, info.totalBytes);
    CHECK(mapped);
    for (const auto& footprint : footprints)
        for (uint32_t row = 0; row < footprint.height; ++row) {
            const auto* bytes = static_cast<const uint8_t*>(mapped) + footprint.offset + uint64_t{ row } * footprint.rowPitch;
            for (uint32_t i = 0; i < footprint.width * 16u; ++i) nonZero += bytes[i] != 0;
        }
    staging->Unmap(0, 0);
    return 0;
}

int TestTextureZeroFill(Harness& h) {
    // Mip 0's rows (1024 texels of 16 bytes) times its height is past the 4 MB zero source, so it is copied in bands.
    constexpr uint32_t kWidth = 1024, kHeight = 1100;
    constexpr uint16_t kMips = 3;
    auto texture = MakeTexture(h.device.Get(), kWidth, kHeight, kMips, rhi::QueueSharing::Concurrent, "CopyQueueZeroFillTarget");
    CHECK(texture);
    uint64_t nonZero = 0;
    if (const auto failure = ReadTexture(h, *texture, kMips, rhi::ResourceLayout::Undefined, 0x5A, nullptr, nonZero)) return failure;
    if (const auto failure = ReadTexture(h, *texture, kMips, rhi::ResourceLayout::Common, 0, nullptr, nonZero)) return failure;
    CHECK(nonZero != 0);
    const auto before = h.service.GetStats();
    auto ticket = h.service.QueueTextureZeroFill(
        org::WorkerOwnedDestination{ texture, org::WorkerOwnedDestination::Ownership::FreshTexture });
    CHECK(ticket);
    for (const auto& segment : texture->GetStateTracker()->GetSegments())
        CHECK(segment.state.layout == rhi::ResourceLayout::Common && segment.state.access == rhi::ResourceAccessType::Common);
    CHECK(WaitTicket(ticket, org::TrackedUploadTicketState::Completed));
    if (const auto failure = ReadTexture(h, *texture, kMips, rhi::ResourceLayout::Common, 0, ticket.get(), nonZero)) return failure;
    CHECK(nonZero == 0);
    CHECK(h.service.WaitIdle(5000));
    const auto after = h.service.GetStats();
    // Mip 0 in 5 bands of 256 rows, mip 1 in 2, mip 2 whole. Nothing staged: no page was made.
    CHECK(after.copiesSubmitted - before.copiesSubmitted == 8);
    CHECK(after.pagesCreated == before.pagesCreated);
    // An exclusive texture would need a queue ownership transfer: refused.
    auto exclusive = MakeTexture(h.device.Get(), 64, 64, 1, rhi::QueueSharing::Exclusive, "CopyQueueZeroFillExclusive");
    CHECK(exclusive);
    CHECK(!h.service.QueueTextureZeroFill(
        org::WorkerOwnedDestination{ exclusive, org::WorkerOwnedDestination::Ownership::FreshTexture }));
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    bool hardware = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--hardware") == 0) hardware = true;
        else if (std::strcmp(argv[i], "--warp") != 0) return 2;
    }
    Microsoft::WRL::ComPtr<IDXGIFactory6> factory; Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
    CHECK(SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))));
    if (hardware) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> candidate;
        for (UINT index = 0; SUCCEEDED(factory->EnumAdapterByGpuPreference(index,
            DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate))); ++index) {
            DXGI_ADAPTER_DESC1 description{};
            CHECK(SUCCEEDED(candidate->GetDesc1(&description)));
            if (!(description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) { adapter = candidate; break; }
            candidate.Reset();
        }
        if (!adapter) { std::puts("SKIP: no hardware adapter available"); return 77; }
    } else {
        CHECK(SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))));
    }
    rhi::DeviceCreateInfo create{}; create.backend = rhi::Backend::D3D12; create.nativeAdapter = adapter.Get(); create.framesInFlight = 2;
    Harness h;
    CHECK(!rhi::Failed(rhi::CreateD3D12Device(create, h.device)) && h.device);
    org::runtime::InitializeRuntimeDevice(h.device.Get());
    h.graphics = h.device->GetQueue(rhi::QueueKind::Graphics);
    h.copy = h.device->GetQueue(rhi::QueueKind::Copy);
    CHECK(h.graphics && h.copy);
    h.service.Initialize(h.device.Get(), h.copy);
    CHECK(h.service.Initialized());

    if (const auto failure = TestSegmentedUpload(h)) return failure;
    if (const auto failure = TestOffsetAndOversize(h)) return failure;
    if (const auto failure = TestPageReuseAndCancel(h)) return failure;
    if (const auto failure = TestSignalEntries(h)) return failure;
    if (const auto failure = TestTextureZeroFill(h)) return failure;

    h.service.Cleanup();
    CHECK(!h.service.Initialized());
    std::puts("CopyQueueUploadService tests passed.");
    return 0;
}
