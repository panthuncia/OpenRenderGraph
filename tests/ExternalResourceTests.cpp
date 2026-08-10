#include <Resources/ExternalBufferResource.h>
#include <Resources/ExternalTextureResource.h>
#include <Render/Runtime/RuntimeDevice.h>
#include <rhi_interop_dx12.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#define CHECK(x) do { if (!(x)) return __LINE__; } while (false)

namespace {
Microsoft::WRL::ComPtr<ID3D12Resource> MakeResource(ID3D12Device* device, bool texture, uint64_t size) {
	D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
	D3D12_RESOURCE_DESC desc{}; desc.Dimension = texture ? D3D12_RESOURCE_DIMENSION_TEXTURE2D : D3D12_RESOURCE_DIMENSION_BUFFER;
	desc.Width = texture ? 64 : size; desc.Height = texture ? 64 : 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
	desc.Format = texture ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_UNKNOWN; desc.SampleDesc.Count = 1;
	desc.Layout = texture ? D3D12_TEXTURE_LAYOUT_UNKNOWN : D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	Microsoft::WRL::ComPtr<ID3D12Resource> result;
	device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&result));
	return result;
}
}

int main() {
	Microsoft::WRL::ComPtr<IDXGIFactory6> factory; Microsoft::WRL::ComPtr<IDXGIAdapter> warp;
	CHECK(SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))));
	CHECK(SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))));
	rhi::DeviceCreateInfo create{}; create.backend = rhi::Backend::D3D12; create.nativeAdapter = warp.Get(); create.framesInFlight = 2;
	rhi::DevicePtr device; CHECK(!rhi::Failed(rhi::CreateD3D12Device(create, device)) && device);
	auto* nativeDevice = rhi::dx12::get_device(device.Get()); CHECK(nativeDevice);
	org::runtime::InitializeRuntimeDevice(device.Get());

	auto bufferA = MakeResource(nativeDevice, false, 4096), bufferB = MakeResource(nativeDevice, false, 4096);
	rhi::ResourcePtr importedA, importedB;
	CHECK(!rhi::Failed(rhi::dx12::import_resource(device.Get(), bufferA.Get(), importedA)));
	CHECK(!rhi::Failed(rhi::dx12::import_resource(device.Get(), bufferB.Get(), importedB)));
	org::ExternalBufferResource::ViewRequirements bufferViews{};
	auto buffer = org::ExternalBufferResource::CreateShared(std::move(importedA), 4096, bufferViews);
	CHECK(buffer && buffer->RefreshShared(std::move(importedB), 4096, bufferViews));
	rhi::ResourcePtr wrongSize; CHECK(!rhi::Failed(rhi::dx12::import_resource(device.Get(), bufferA.Get(), wrongSize)));
	CHECK(!buffer->RefreshShared(std::move(wrongSize), 8192, bufferViews));

	org::TextureDescription description{}; description.imageDimensions = { { 64, 64 } };
	description.channels = 4; description.format = rhi::Format::R8G8B8A8_UNorm; description.hasSRV = true;
	description.initialLayout = rhi::ResourceLayout::Common;
	auto textureA = MakeResource(nativeDevice, true, 0), textureB = MakeResource(nativeDevice, true, 0);
	rhi::ResourcePtr textureImportA, textureImportB;
	CHECK(!rhi::Failed(rhi::dx12::import_resource(device.Get(), textureA.Get(), textureImportA)));
	CHECK(!rhi::Failed(rhi::dx12::import_resource(device.Get(), textureB.Get(), textureImportB)));
	auto texture = org::ExternalTextureResource::CreateShared(std::move(textureImportA), description, false);
	CHECK(texture && texture->RefreshShared(std::move(textureImportB), description, false));
	auto changed = description; changed.imageDimensions[0].width = 32;
	rhi::ResourcePtr incompatible; CHECK(!rhi::Failed(rhi::dx12::import_resource(device.Get(), textureA.Get(), incompatible)));
	CHECK(!texture->RefreshShared(std::move(incompatible), changed, false));
	buffer.reset(); texture.reset();
	org::runtime::ShutdownRuntimeDevice();
	return 0;
}
