// CopyQueueUploadService: worker-native uploads on the copy queue.
//
// Runs on WARP by default (--hardware selects a hardware adapter). Checks:
//  - segmented buffer uploads land byte-exact when read back on the graphics
//    queue after a wait on the ticket's timeline;
//  - tickets move Queued -> Submitted -> Completed, and cancellation before
//    the submitter claims a copy never submits it;
//  - staging pages are reused across rounds instead of created per upload,
//    and oversize uploads take a dedicated page that is not recycled.

#include <Render/Runtime/CopyQueueUploadService.h>
#include <Render/Runtime/RuntimeDevice.h>
#include <Resources/ExternalBufferResource.h>
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

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #x); return __LINE__; } } while (false)

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

    h.service.Cleanup();
    CHECK(!h.service.Initialized());
    std::puts("CopyQueueUploadService tests passed.");
    return 0;
}
