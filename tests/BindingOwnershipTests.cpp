#include "Render/BindingTable.h"
#include "Render/DescriptorHeap.h"
#include "Render/PublicationBindingBundle.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Binding ownership check failed at %d: %s\n", __LINE__, #x); return __LINE__; } } while (false)

int TestBindingOwnership(rhi::Device device) {
    auto cleanup = org::runtime::ResourceCleanupQueue::Create();
    {
        // A captured ORG view must use the original physical slot and retain
        // its exact backing even after the original resource owner disappears.
        auto backing = cleanup->Make<rhi::ResourcePtr>();
        rhi::ResourceDesc description{};
        description.type = rhi::ResourceType::Buffer;
        description.buffer.sizeBytes = 256;
        CHECK(device.CreateCommittedResource(description, *backing) == rhi::Result::Ok);
        auto viewsHeap = std::make_shared<org::DescriptorHeap>(device,
            rhi::DescriptorHeapType::CbvSrvUav, 1, true, "Captured binding test");
        rhi::SrvDesc view{};
        view.dimension = rhi::SrvDim::Buffer;
        view.buffer.numElements = 64;
        auto original = org::OwnedDescriptorBinding::CreateShaderResourceView(device,
            cleanup->Make<int>(1), 1, viewsHeap, cleanup, backing->Get(), backing, view);
        CHECK(original);
        const auto index = original.Index();
        std::weak_ptr<rhi::ResourcePtr> weakBacking = backing;
        auto snapshot = std::make_shared<org::ResourceBindingSnapshot>();
        snapshot->resourceID = 73;
        snapshot->backingGeneration = 11;
        snapshot->resource = backing->Get();
        snapshot->allocationOwner = backing;
        snapshot->descriptorOwner = viewsHeap->CaptureDescriptorLease(index);
        auto views = std::make_shared<org::BindlessResourceViews>();
        views->defaultSrvVariant = 0;
        views->views.push_back({org::BindlessViewKind::ShaderResource, 0, 0, 0, original.Slot()});
        snapshot->views = views;
        auto captured = org::OwnedDescriptorBinding::CaptureResourceView(snapshot, {}, 1, cleanup);
        CHECK(captured.Index() == index && captured.Identity().allocation == 11);
        auto bundle = cleanup->Make<org::PublicationBindingBundle>(
            std::vector<org::PublicationBindingBundle::Snapshot>{snapshot});
        auto prepared = org::ExecutionResourceLease::FromBundle(bundle);
        auto patch = org::ExecutionResourceLease::Create(cleanup, {captured.Owner()});
        auto combined = org::ExecutionResourceLease::Combine(cleanup, prepared, patch);
        CHECK(combined.Bundle()->Find(73) && *combined.Bundle()->Find(73) == snapshot);
        original = {};
        backing.reset();
        snapshot.reset();
        captured = {};
        bundle.reset();
        prepared = {};
        patch = {};
        cleanup->Drain();
        CHECK(!weakBacking.expired());
        bool exhausted = false;
        try { (void)viewsHeap->AllocateDescriptor(); } catch (const std::runtime_error&) { exhausted = true; }
        CHECK(exhausted);
        combined = {};
        cleanup->Drain();
        CHECK(weakBacking.expired());
        CHECK(viewsHeap->AllocateDescriptor() == index);
        viewsHeap->ReleaseDescriptor(index);
    }
    const auto caller = std::this_thread::get_id();
    std::atomic<int> destroyed{0};
    std::atomic<bool> wrongThread{false};
    struct DestructionProbe {
        std::atomic<int>* destroyed;
        std::atomic<bool>* wrongThread;
        std::thread::id caller;
        ~DestructionProbe() {
            if (std::this_thread::get_id() == caller) wrongThread->store(true);
            destroyed->fetch_add(1);
        }
    };
    auto deviceOwner = cleanup->Make<DestructionProbe>(&destroyed, &wrongThread, caller);
    auto heap = std::make_shared<org::DescriptorHeap>(device, rhi::DescriptorHeapType::Sampler, 2, true, "Owned binding test");
    auto binding = org::OwnedDescriptorBinding::CreateSampler(device, deviceOwner, 1, heap, cleanup, rhi::SamplerDesc{});
    CHECK(binding);
    auto capturedSampler = org::OwnedDescriptorBinding::CaptureSampler(binding.Index(), 1, heap, cleanup);
    CHECK(capturedSampler.Index() == binding.Index());
    capturedSampler = {};
    cleanup->Drain();
    const auto firstIndex = binding.Index();
    const auto firstIdentity = binding.Identity();
    std::weak_ptr<const void> weakBinding = binding.Owner();
    org::BindingRecordBuilder recordBuilder(cleanup, sizeof(uint32_t));
    recordBuilder.WriteBinding(0, binding);
    auto record = recordBuilder.Seal();
    uint32_t encoded = UINT32_MAX;
    std::memcpy(&encoded, record->Bytes().data(), sizeof(encoded));
    CHECK(encoded == binding.Index());
    org::BindingTableBuilder builder(cleanup);
    builder.Set(0, record);
    builder.Set(64, record);
    auto first = builder.Seal();
    org::BindingTableBuilder unchanged(cleanup, first);
    CHECK(unchanged.Seal() == first);
    org::BindingRecordBuilder clearedRecord(cleanup, *record);
    clearedRecord.WriteBinding(0, {});
    auto cleared = clearedRecord.Seal();
    std::memcpy(&encoded, cleared->Bytes().data(), sizeof(encoded));
    CHECK(encoded == UINT32_MAX);
    org::BindingTableBuilder successor(cleanup, first);
    successor.Set(0, cleared);
    auto second = successor.Seal();
    CHECK(second->BaseVersion() == first->Version());
    CHECK(second->PageAt(0) != first->PageAt(0) && second->PageAt(1) == first->PageAt(1));
    CHECK(first->Get(0) == record && second->Get(0) == cleared);
    auto prepared = org::ExecutionResourceLease::Create(cleanup, {first});
    CHECK(prepared.Bundle());
    CHECK(org::ExecutionResourceLease::FromBundle(prepared.Bundle()).Owner() == prepared.Owner());
    auto patches = org::ExecutionResourceLease::Create(cleanup, {second});
    auto combined = org::ExecutionResourceLease::Combine(cleanup, prepared, patches);
    org::BindingTableBuilder removal(cleanup, second);
    removal.Clear(64);
    auto final = removal.Seal();
    CHECK(final->PageCount() == 1 && !final->Get(64));
    binding = {};
    record.reset();
    first.reset();
    second.reset();
    prepared = {};
    patches = {};
    cleanup->Drain();
    CHECK(!weakBinding.expired()); // A prepared/replayable execution still owns its exact version.
    combined = {};
    cleanup->Drain();
    CHECK(weakBinding.expired()); // Successor does not retain its predecessor's obsolete pages.
    auto reused = org::OwnedDescriptorBinding::CreateSampler(device, deviceOwner, 1, heap, cleanup, rhi::SamplerDesc{});
    CHECK(reused.Index() == firstIndex && reused.Identity() != firstIdentity);
    reused = {};
    cleanup->Drain();

    // Failure after allocating a slot must roll it back on cleanup. Use the
    // real heap with a failing create function, without sending invalid Vulkan.
    auto failFunctions = *device.vt;
    failFunctions.createSampler = [](rhi::Device*, rhi::DescriptorSlot, const rhi::SamplerDesc&) noexcept { return rhi::Result::Failed; };
    auto failingDevice = device;
    failingDevice.vt = &failFunctions;
    CHECK(!org::OwnedDescriptorBinding::CreateSampler(failingDevice, deviceOwner, 1, heap, cleanup, rhi::SamplerDesc{}));
    cleanup->Drain();
    auto afterFailure = org::OwnedDescriptorBinding::CreateSampler(device, deviceOwner, 1, heap, cleanup, rhi::SamplerDesc{});
    CHECK(afterFailure && afterFailure.Index() == firstIndex);
    afterFailure = {};
    deviceOwner.reset();
    cleanup->Drain();
    CHECK(destroyed.load() == 1 && !wrongThread.load());
    final.reset();
    cleared.reset();
    cleanup->Drain();
    heap.reset();
    return 0;
}
