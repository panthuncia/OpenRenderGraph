// Versioned resources without a graph: the adoption observer.
//
// The observer sees every Publish and Adopt (and Grow*, which adopt), and no PublishLate, whose versions are late-bound by
// design. A host counts adoptions outside its frame boundary with it.

#include "OpenRenderGraph/Runtime.h"
#include "Resources/Buffers/Buffer.h"
#include "Resources/Buffers/VersionedBuffer.h"
#include "Resources/PixelBuffer.h"
#include "Resources/VersionedTexture.h"
#include "TestSupport.h"

#include <rhi.h>
#include <rhi_interop_dx12.h>
#include <directx/d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>

namespace {
constexpr uint32_t kCount = 64;
constexpr uint64_t kBytes = kCount * sizeof(uint32_t);

std::atomic<uint32_t> g_observedAdoptions{0};
std::atomic<uint64_t> g_observedNumber{0};

void CountAdoption(const org::VersionedResource&, const org::ResourceVersion& version) noexcept {
    g_observedAdoptions.fetch_add(1);
    g_observedNumber.store(version.number);
}

int TestAdoptionObserver() {
    const auto makeBuffer = [] { return org::Buffer::CreateShared(rhi::HeapType::Upload, kBytes); };
    org::TextureDescription description{};
    description.imageDimensions = {{kCount, 1, 0, 0}};
    description.format = rhi::Format::R32_UInt;
    description.channels = 1;
    org::VersionedResource::SetAdoptionObserver(&CountAdoption);
    // Create publishes its first version.
    auto buffer = org::VersionedBuffer::Create(makeBuffer());
    CHECK(g_observedAdoptions.load() == 1);
    // A pending version is not adopted until its owner adopts it.
    const auto pending = buffer->Make(makeBuffer());
    CHECK(g_observedAdoptions.load() == 1);
    buffer->Adopt(pending);
    CHECK(g_observedAdoptions.load() == 2);
    CHECK(g_observedNumber.load() == pending->number);
    buffer->GrowBytes(kBytes * 2);
    CHECK(g_observedAdoptions.load() == 3);
    auto texture = org::VersionedTexture::Create(org::PixelBuffer::CreateShared(description));
    const auto next = texture->Publish(org::PixelBuffer::CreateShared(description));
    CHECK(g_observedAdoptions.load() == 5);
    CHECK(g_observedNumber.load() == next->number);
    // Late-bound versions are never reported.
    auto late = org::VersionedBuffer::CreateLateBound(makeBuffer());
    CHECK(late->PublishLate(makeBuffer()) != nullptr);
    CHECK(g_observedAdoptions.load() == 5);
    // Without an observer nothing is reported.
    org::VersionedResource::SetAdoptionObserver(nullptr);
    buffer->Publish(makeBuffer());
    CHECK(g_observedAdoptions.load() == 5);
    return 0;
}
} // namespace

int main() {
    Microsoft::WRL::ComPtr<IDXGIFactory6> factory;
    Microsoft::WRL::ComPtr<IDXGIAdapter> warp;
    CHECK(SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))));
    CHECK(SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))));
    rhi::DeviceCreateInfo create{};
    create.backend = rhi::Backend::D3D12;
    create.nativeAdapter = warp.Get();
    create.framesInFlight = 2;
    rhi::DevicePtr device;
    CHECK(!rhi::Failed(rhi::CreateD3D12Device(create, device)) && device);
    auto runtime = org::Runtime::Create({.device = device.Get(), .backend = rhi::Backend::D3D12});
    if (const int line = TestAdoptionObserver()) return line;
    std::puts("PASS: VersionedResourceTests");
    return 0;
}
