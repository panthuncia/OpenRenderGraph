#include <RenderPasses/ReadbackCopyCapturePass.h>
#include <cstring>
#include <cstdio>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Readback reservation check failed at %d: %s\n", __LINE__, #x); return __LINE__; } } while (false)
namespace {
class CaptureService final : public org::runtime::IReadbackService {
public:
    std::vector<org::ReadbackCaptureRequest> requests;
    void Initialize(rhi::Timeline, rhi::Timeline) override {}
    void RequestReadbackCapture(const std::string&, org::Resource*, const org::RangeSpec&,
        org::ReadbackCaptureCallback, org::QueueKind) override {}
    std::vector<org::runtime::ReadbackCaptureInfo> ConsumeCaptureRequests() override { return {}; }
    std::shared_ptr<org::Resource> AcquireReadbackBuffer(uint64_t size, const char*) override {
        return org::Buffer::CreateShared(rhi::HeapType::Readback, size);
    }
    org::runtime::ReadbackCaptureToken EnqueueCapture(org::ReadbackCaptureRequest&& request) override {
        requests.push_back(std::move(request)); return {requests.size()};
    }
    void FinalizeCapture(org::runtime::ReadbackCaptureToken, org::QueueKind,
        std::shared_ptr<rhi::TimelinePtr>, uint64_t) override {}
    uint64_t GetNextReadbackFenceValue(org::QueueKind) override { return 1; }
    rhi::Timeline GetReadbackFence(org::QueueKind) const override { return {}; }
    void ProcessReadbackRequests() override {}
    void Cleanup() override {}
};
template<org::QueueKind Kind>
int TestQueue(rhi::Device device) {
    auto service = std::make_shared<CaptureService>();
    auto source = org::Buffer::CreateShared(rhi::HeapType::Upload, 256);
    auto sourceAllocation = source->CaptureBackingAllocation();
    CHECK(sourceAllocation);
    void* mapped = nullptr;
    sourceAllocation.resource.Map(&mapped, 0, 256);
    CHECK(mapped);
    std::memset(mapped, 0x5a, 256);
    sourceAllocation.resource.Unmap(0, 256);
    auto bindings = std::make_shared<org::FrozenExecutionBindings>(
        std::vector<org::FrozenExecutionBindings::ResourceBinding>{{sourceAllocation.resource, sourceAllocation.lease}});
    auto slots = std::make_shared<std::unordered_map<uint64_t, uint32_t>>();
    slots->emplace(source->GetGlobalResourceID(), 0);
    org::FramePreparationContext preparation{};
    preparation.device = device; preparation.bindings = bindings; preparation.resourceSlots = slots;
    org::PreparedPass cancelled, submitted;
    {
        org::BasicReadbackCapturePass<Kind> pass({}, source, {}, service);
        CHECK(pass.CompileKey() != 0);
        cancelled = pass.PrepareFrame(preparation);
        submitted = pass.PrepareFrame(preparation);
    }
    CHECK(service->requests.empty());
    CHECK(cancelled.Abandon(org::AbandonReason::GenerationInvalidated));
    CHECK(!cancelled.Abandon(org::AbandonReason::Shutdown));
    cancelled = {};
    CHECK(service->requests.empty());
    std::weak_ptr<CaptureService> serviceLifetime = service;
    service.reset();
    CHECK(!serviceLifetime.expired());
    const auto queueKind = Kind == org::QueueKind::Copy ? rhi::QueueKind::Copy : rhi::QueueKind::Graphics;
    rhi::CommandAllocatorPtr allocator;
    rhi::CommandListPtr commands;
    CHECK(device.CreateCommandAllocator(queueKind, allocator) == rhi::Result::Ok);
    CHECK(device.CreateCommandList(queueKind, allocator.Get(), commands) == rhi::Result::Ok);
    org::RecordingContext recording(commands.Get(), bindings);
    submitted.Record(recording);
    CHECK(commands->EndChecked() == rhi::Result::Ok);
    auto list = commands.Get();
    auto queue = device.GetQueue(queueKind);
    CHECK(queue.Submit({&list, 1}) == rhi::Result::Ok);
    for (const auto& signal : submitted.ExternalSignalsAfterCompletion())
        CHECK(queue.Signal({signal.timeline.GetHandle(), signal.value}) == rhi::Result::Ok);
    submitted.CommitSubmitted();
    auto retainedService = serviceLifetime.lock();
    CHECK(retainedService && retainedService->requests.size() == 1);
    CHECK(device.WaitIdle() == rhi::Result::Ok);
    submitted.CommitCompleted();
    const auto& request = retainedService->requests.front();
    CHECK(request.signalFenceOwner && request.fenceValue == 1 && request.signalQueueKind == Kind);
    request.readbackBuffer->GetAPIResource().Map(&mapped, 0, 256);
    CHECK(mapped);
    for (size_t i = 0; i < 256; ++i) CHECK(static_cast<unsigned char*>(mapped)[i] == 0x5a);
    request.readbackBuffer->GetAPIResource().Unmap(0, 0);
    submitted = {};
    retainedService.reset();
    CHECK(serviceLifetime.expired());
    return 0;
}
}
int TestReadbackCaptures(rhi::Device device) {
    if (auto failure = TestQueue<org::QueueKind::Graphics>(device)) return failure;
    return TestQueue<org::QueueKind::Copy>(device);
}
