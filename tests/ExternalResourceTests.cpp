#include <Resources/ExternalBufferResource.h>
#include <Resources/ExternalTextureResource.h>
#include <Render/Runtime/RuntimeDevice.h>
#include <Render/PassBuilders.h>
#include <rhi_interop_dx12.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#define CHECK(x) do { if (!(x)) return __LINE__; } while (false)

namespace {
struct EmptyResolver final : org::ClonableResolver<EmptyResolver> {
	struct State {
		uint64_t setRevision = 1, contentRevision = 1;
		std::shared_ptr<const org::ResolverResourceList> resources = std::make_shared<const org::ResolverResourceList>();
	};
	std::shared_ptr<State> identity = std::make_shared<State>();
	std::vector<std::shared_ptr<org::Resource>> Resolve() const override { return *identity->resources; }
	std::shared_ptr<const org::ResolverDeclarationState> CaptureDeclarationState() const override {
		auto state = std::make_shared<org::ResolverDeclarationState>();
		state->tracked = true;
		state->dependencyIdentity = identity;
		state->resourceSetIdentity = {identity->setRevision, 0};
		state->contentRevision = identity->contentRevision;
		state->resources = identity->resources;
		return state;
	}
};
struct DeclarationTestPass final : org::ComputePass {
	void Setup() override {}
	void Cleanup() override {}
};
struct RefreshTestPass final : org::ComputePass {
	std::shared_ptr<EmptyResolver> resolver;
	std::shared_ptr<EmptyResolver> secondResolver;
	bool symbolic;
	int declarations = 0, setups = 0;
	RefreshTestPass(std::shared_ptr<EmptyResolver> r, bool s, std::shared_ptr<EmptyResolver> second = {})
		: resolver(std::move(r)), secondResolver(std::move(second)), symbolic(s) {}
	void Setup() override { ++setups; }
	void Cleanup() override {}
	org::ResourceRegistryView* View() const { return m_resourceRegistryView.get(); }
	void DeclareResourceUsages(org::ComputePassBuilder* builder) override {
		++declarations;
		if (symbolic) builder->WithShaderResource(org::ResourceIdentifier("test.empty-resolver"));
		else builder->WithShaderResource(*resolver);
		if (secondResolver) builder->WithShaderResource(*secondResolver);
	}
};

int TestEmptyResolverRefresh(rhi::Device device, const std::shared_ptr<org::Resource>& resource) {
	org::RenderGraph graph(device, rhi::Backend::D3D12);
	auto resolver = std::make_shared<EmptyResolver>();
	graph.RegisterResolver(org::ResourceIdentifier("test.empty-resolver"), resolver);
	graph.BuildComputePass<RefreshTestPass>("Direct", resolver, false);
	graph.BuildComputePass<RefreshTestPass>("Symbolic", resolver, true);
	graph.CompileStructural();
	graph.Setup();
	auto direct = std::dynamic_pointer_cast<RefreshTestPass>(graph.GetComputePassByName("Direct"));
	auto symbolic = std::dynamic_pointer_cast<RefreshTestPass>(graph.GetComputePassByName("Symbolic"));
	CHECK(direct && symbolic && direct->declarations == 1 && symbolic->declarations == 1);
	auto* directView = direct->View();
	auto* symbolicView = symbolic->View();
	const auto setupCount = direct->setups;
	for (unsigned step = 0; step != 3; ++step) {
		resolver->identity->resources = std::make_shared<const org::ResolverResourceList>(
			step == 1 ? org::ResolverResourceList{} : org::ResolverResourceList{resource});
		++resolver->identity->setRevision;
		org::UpdateExecutionContext context{};
		context.frameIndex = step % 2;
		context.frameFenceValue = step + 1;
		graph.Update(context, device);
		CHECK(direct->declarations == 1 && symbolic->declarations == 1);
		CHECK(direct->setups == setupCount + step + 1);
		CHECK(direct->View() == directView && symbolic->View() == symbolicView);
		const org::ResourceIdentifier numericID(std::to_string(resource->GetGlobalResourceID()));
		if (step != 1) {
			CHECK(directView->RequestHandle(numericID).GetGlobalResourceID() == resource->GetGlobalResourceID());
			CHECK(symbolicView->RequestHandle(numericID).GetGlobalResourceID() == resource->GetGlobalResourceID());
		} else {
			bool denied = false;
			try { directView->RequestHandle(numericID); } catch (const std::runtime_error&) { denied = true; }
			CHECK(denied);
		}
	}
	++resolver->identity->contentRevision;
	graph.Update(org::UpdateExecutionContext{}, device);
	CHECK(direct->declarations == 1 && direct->setups == setupCount + 3);
	return 0;
}

int TestMixedResolverCollision(rhi::Device device, const std::shared_ptr<org::Resource>& resource) {
	org::RenderGraph graph(device, rhi::Backend::D3D12);
	auto initiallyEmpty = std::make_shared<EmptyResolver>();
	auto populated = std::make_shared<EmptyResolver>();
	populated->identity->resources = std::make_shared<const org::ResolverResourceList>(org::ResolverResourceList{resource});
	graph.BuildComputePass<RefreshTestPass>("Mixed", initiallyEmpty, false, populated);
	graph.CompileStructural();
	graph.Setup();
	auto pass = std::dynamic_pointer_cast<RefreshTestPass>(graph.GetComputePassByName("Mixed"));
	CHECK(pass && pass->declarations == 1);
	initiallyEmpty->identity->resources = populated->identity->resources;
	++initiallyEmpty->identity->setRevision;
	graph.Update(org::UpdateExecutionContext{}, device);
	// A new set overlapping an unchanged resolver must use the full merge path,
	// not silently duplicate accesses after discarding unchanged membership IDs.
	CHECK(pass->declarations == 2);
	return 0;
}

int TestEmptyResolverDeclarations(rhi::Device device) {
	org::RenderGraph graph(device, rhi::Backend::D3D12);
	auto resolver = std::make_shared<EmptyResolver>();
	const org::ResourceIdentifier identifier("test.empty-resolver");
	graph.RegisterResolver(identifier, resolver);
	auto& direct = graph.BuildComputePass<DeclarationTestPass>("Direct");
	direct.WithShaderResource(*resolver);
	auto& symbolic = graph.BuildComputePass<DeclarationTestPass>("Symbolic");
	symbolic.WithShaderResource(identifier);
	auto directStates = direct.TakeResolverSnapshots();
	auto symbolicStates = symbolic.TakeResolverSnapshots();
	CHECK(directStates.size() == 1 && symbolicStates.size() == 1);
	const auto& a = directStates.front();
	const auto& b = symbolicStates.front();
	CHECK(a.resourceIDs.empty() && b.resourceIDs.empty());
	CHECK(a.dependencyIdentity == b.dependencyIdentity);
	CHECK(a.resolver.get() != resolver.get());
	CHECK(!a.hasUnclassifiedDeclaration && !b.hasUnclassifiedDeclaration);
	CHECK(a.declaredRequirementTemplates.size() == 1 && b.declaredRequirementTemplates.size() == 1);
	CHECK(a.declaredRequirementTemplates[0].range == b.declaredRequirementTemplates[0].range);
	CHECK(a.declaredRequirementTemplates[0].state == b.declaredRequirementTemplates[0].state);
	CHECK(a.declaredRequirementTemplates[0].state.sync == b.declaredRequirementTemplates[0].state.sync);
	// Multiple authored uses must survive an empty initial capture, too.
	auto& multiple = graph.BuildComputePass<DeclarationTestPass>("Multiple");
	multiple.WithShaderResource(*resolver).WithUnorderedAccess(*resolver).WithShaderResource(*resolver);
	auto multiStates = multiple.TakeResolverSnapshots();
	CHECK(multiStates.size() == 1);
	CHECK(multiStates[0].declaredRequirementTemplates.size() == 2);
	return 0;
}

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
	if (const auto failure = TestEmptyResolverDeclarations(device.Get())) return failure;

	auto bufferA = MakeResource(nativeDevice, false, 4096), bufferB = MakeResource(nativeDevice, false, 4096);
	rhi::ResourcePtr importedA, importedB;
	CHECK(!rhi::Failed(rhi::dx12::import_resource(device.Get(), bufferA.Get(), importedA)));
	CHECK(!rhi::Failed(rhi::dx12::import_resource(device.Get(), bufferB.Get(), importedB)));
	org::ExternalBufferResource::ViewRequirements bufferViews{};
	auto buffer = org::ExternalBufferResource::CreateShared(std::move(importedA), 4096, bufferViews);
	CHECK(buffer && buffer->RefreshShared(std::move(importedB), 4096, bufferViews));
	rhi::ResourcePtr wrongSize; CHECK(!rhi::Failed(rhi::dx12::import_resource(device.Get(), bufferA.Get(), wrongSize)));
	CHECK(!buffer->RefreshShared(std::move(wrongSize), 8192, bufferViews));
	{
		org::ResourceRegistry registry;
		rhi::ResourcePtr temporaryImport;
		CHECK(!rhi::Failed(rhi::dx12::import_resource(device.Get(), bufferA.Get(), temporaryImport)));
		auto temporary = org::ExternalBufferResource::CreateShared(std::move(temporaryImport), 4096, bufferViews);
		const org::ResourceIdentifier expiredID(std::to_string(temporary->GetGlobalResourceID()));
		registry.RegisterAnonymousWeak(temporary);
		CHECK(registry.MakeHandle(expiredID).GetGeneration() != 0);
		temporary.reset();
		CHECK(registry.MakeHandle(expiredID).GetGeneration() == 0);
		registry.ReclaimExpiredAnonymous();
		registry.RegisterAnonymous(buffer); // reuses the expired slot
		CHECK(registry.MakeHandle(expiredID).GetGeneration() == 0);
		const org::ResourceIdentifier bufferID(std::to_string(buffer->GetGlobalResourceID()));
		CHECK(registry.MakeHandle(bufferID).GetGlobalResourceID() == buffer->GetGlobalResourceID());
		registry = {};
		CHECK(registry.MakeHandle(bufferID).GetGeneration() == 0);
	}
	if (const auto failure = TestEmptyResolverRefresh(device.Get(), buffer)) return failure;
	if (const auto failure = TestMixedResolverCollision(device.Get(), buffer)) return failure;

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
