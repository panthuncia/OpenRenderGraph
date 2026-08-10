#include <OpenRenderGraph/ContributorGraphHost.h>
#include <OpenRenderGraph/ExtensionRegistry.h>
#include "ContributorExecutionContext.h"

#include <Render/RenderGraph/RenderGraph.h>
#include <Render/Runtime/RuntimeDevice.h>
#include <Render/Runtime/OpenRenderGraphSettings.h>
#include <Render/Runtime/IUploadService.h>
#include <Render/Runtime/UploadServiceAccess.h>
#include <Render/Runtime/DescriptorServiceAccess.h>
#include <Render/MemoryIntrospectionBackend.h>
#include <RenderPasses/Base/ComputePass.h>
#include <RenderPasses/Base/RenderPass.h>
#include <RenderPasses/Base/CopyPass.h>
#include <Interfaces/IDynamicDeclaredResources.h>
#include <Resources/Buffers/Buffer.h>
#include <Resources/DynamicResource.h>
#include <Resources/GloballyIndexedResource.h>
#include <Resources/PixelBuffer.h>
#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace org::contributor {

namespace
{
	class ActiveGraphServices final
	{
	public:
		ActiveGraphServices(org::runtime::IUploadService* uploads,
			org::runtime::IDescriptorService* descriptors) noexcept
			: previousUploads_(org::runtime::GetActiveUploadService()),
			  previousDescriptors_(org::runtime::GetActiveDescriptorService())
		{
			org::runtime::SetActiveUploadService(uploads);
			org::runtime::SetActiveDescriptorService(descriptors);
		}
		~ActiveGraphServices()
		{
			org::runtime::SetActiveDescriptorService(previousDescriptors_);
			org::runtime::SetActiveUploadService(previousUploads_);
		}
	private:
		org::runtime::IUploadService* previousUploads_{};
		org::runtime::IDescriptorService* previousDescriptors_{};
	};

	uint16_t FormatChannels(ORGFormat format) noexcept
	{
		switch (format) {
		case ORG_RG_FORMAT_R32G32B32_TYPELESS: case ORG_RG_FORMAT_R32G32B32_FLOAT:
		case ORG_RG_FORMAT_R32G32B32_UINT: case ORG_RG_FORMAT_R32G32B32_SINT:
		case ORG_RG_FORMAT_R11G11B10_FLOAT: return 3;
		case ORG_RG_FORMAT_R32G32_TYPELESS: case ORG_RG_FORMAT_R32G32_FLOAT:
		case ORG_RG_FORMAT_R32G32_UINT: case ORG_RG_FORMAT_R32G32_SINT:
		case ORG_RG_FORMAT_R16G16_TYPELESS: case ORG_RG_FORMAT_R16G16_FLOAT:
		case ORG_RG_FORMAT_R16G16_UNORM: case ORG_RG_FORMAT_R16G16_UINT:
		case ORG_RG_FORMAT_R16G16_SNORM: case ORG_RG_FORMAT_R16G16_SINT:
		case ORG_RG_FORMAT_R8G8_TYPELESS: case ORG_RG_FORMAT_R8G8_UNORM:
		case ORG_RG_FORMAT_R8G8_UINT: case ORG_RG_FORMAT_R8G8_SNORM: case ORG_RG_FORMAT_R8G8_SINT:
		case ORG_RG_FORMAT_BC5_TYPELESS: case ORG_RG_FORMAT_BC5_UNORM: case ORG_RG_FORMAT_BC5_SNORM: return 2;
		case ORG_RG_FORMAT_R32_TYPELESS: case ORG_RG_FORMAT_D32_FLOAT: case ORG_RG_FORMAT_R32_FLOAT:
		case ORG_RG_FORMAT_R32_UINT: case ORG_RG_FORMAT_R32_SINT:
		case ORG_RG_FORMAT_R16_TYPELESS: case ORG_RG_FORMAT_R16_FLOAT: case ORG_RG_FORMAT_R16_UNORM:
		case ORG_RG_FORMAT_R16_UINT: case ORG_RG_FORMAT_R16_SNORM: case ORG_RG_FORMAT_R16_SINT:
		case ORG_RG_FORMAT_R8_TYPELESS: case ORG_RG_FORMAT_R8_UNORM: case ORG_RG_FORMAT_R8_UINT:
		case ORG_RG_FORMAT_R8_SNORM: case ORG_RG_FORMAT_R8_SINT:
		case ORG_RG_FORMAT_BC4_TYPELESS: case ORG_RG_FORMAT_BC4_UNORM: case ORG_RG_FORMAT_BC4_SNORM: return 1;
		default: return 4;
		}
	}

	struct ExecutionHost
	{
		struct PersistentResource { ORGResourceDesc description{}; std::shared_ptr<org::Resource> resource; };
		org::runtime::IUploadService* graphUploads{};
		std::unordered_map<uint64_t, std::shared_ptr<org::Resource>> resources;
		std::unordered_map<uint64_t, org::ResourceIdentifier> resourceIdentifiers;
		std::unordered_map<std::string, PersistentResource> persistentResources;
	};
	bool StructurallyCompatible(const ORGResourceDesc& left, const ORGResourceDesc& right) noexcept
	{
		return left.dimension == right.dimension && left.heapClass == right.heapClass &&
			left.format == right.format && left.byteSize == right.byteSize &&
			left.structureByteStride == right.structureByteStride && left.width == right.width &&
			left.height == right.height && left.depthOrArraySize == right.depthOrArraySize &&
			left.mipLevels == right.mipLevels && left.sampleCount == right.sampleCount &&
			left.allowedUsages == right.allowedUsages && left.allowAlias == right.allowAlias &&
			left.aliasingPool == right.aliasingPool;
	}
	constexpr org::ExternalTimelineBinding kD3D11ReadyBinding = 1;

	struct FrontendState
	{
		rhi::Timeline readyTimeline{};
		uint64_t readyValue{};
		rhi::Timeline completeTimeline{};
		uint64_t completeValue{};
		ContributorGraphHost::FrameInfo frame{};
		std::vector<ContributorRegistry::Pass> genericPasses;
		std::vector<ContributorRegistry::Resource> genericResources;
		std::vector<std::string> anchors;
		ExecutionHost* host{};
		rhi::Device device{};
		org::runtime::IDescriptorService* descriptorService{};
	};

	org::ResourcePtrAndRange GenericResourceRange(const ContributorRegistry::Access& access,
		const ExecutionHost& host)
	{
		const auto resource = host.resources.find(access.resource);
		if (resource == host.resources.end() || !resource->second)
			throw std::runtime_error("Missing ORG resource for generic pass access");
		org::RangeSpec range{};
		range.mipLower = { org::BoundType::From, access.range.firstMip };
		range.mipUpper = access.range.mipCount == UINT32_MAX ? org::Bound{ org::BoundType::All, 0 } :
			org::Bound{ org::BoundType::UpTo, access.range.firstMip + access.range.mipCount - 1 };
		range.sliceLower = { org::BoundType::From, access.range.firstArraySlice };
		range.sliceUpper = access.range.arraySize == UINT32_MAX ? org::Bound{ org::BoundType::All, 0 } :
			org::Bound{ org::BoundType::UpTo, access.range.firstArraySlice + access.range.arraySize - 1 };
		return { resource->second, range };
	}

	void DeclareGeneric(org::RenderPassBuilder* builder, const ContributorRegistry::Pass& pass, const FrontendState& state)
	{
		for (const auto& domain : pass.featureDomains) builder->WithActiveFeatureDomain(domain);
		for (const auto& access : pass.accesses) {
			auto resource = GenericResourceRange(access, *state.host);
			switch (access.kind) {
			case ORG_RG_ACCESS_SHADER_RESOURCE: builder->WithShaderResource(resource); break;
			case ORG_RG_ACCESS_CONSTANT_BUFFER: builder->WithConstantBuffer(resource); break;
			case ORG_RG_ACCESS_UNORDERED_ACCESS: builder->WithUnorderedAccess(resource); break;
			case ORG_RG_ACCESS_UNORDERED_ACCESS_CLEAR: builder->WithUnorderedAccessClear(resource); break;
			case ORG_RG_ACCESS_RENDER_TARGET: builder->WithRenderTarget(resource); break;
			case ORG_RG_ACCESS_RENDER_TARGET_CLEAR: builder->WithRenderTargetClear(resource); break;
			case ORG_RG_ACCESS_DEPTH_READ: builder->WithDepthRead(resource); break;
			case ORG_RG_ACCESS_DEPTH_READ_WRITE: builder->WithDepthReadWrite(resource); break;
			case ORG_RG_ACCESS_DEPTH_STENCIL_CLEAR: builder->WithDepthStencilClear(resource); break;
			case ORG_RG_ACCESS_COPY_SOURCE: builder->WithCopySource(resource); break;
			case ORG_RG_ACCESS_COPY_DESTINATION: builder->WithCopyDest(resource); break;
			case ORG_RG_ACCESS_INDIRECT_ARGUMENT: builder->WithIndirectArguments(resource); break;
			case ORG_RG_ACCESS_INDEX_BUFFER: builder->WithIndexBuffer(resource); break;
			case ORG_RG_ACCESS_LEGACY_INTEROP: builder->WithLegacyInterop(resource); break;
			default: throw std::runtime_error("Unsupported render-pass resource access");
			}
		}
	}

	void DeclareGeneric(org::ComputePassBuilder* builder, const ContributorRegistry::Pass& pass, const FrontendState& state)
	{
		for (const auto& domain : pass.featureDomains) builder->WithActiveFeatureDomain(domain);
		for (const auto& access : pass.accesses) {
			auto resource = GenericResourceRange(access, *state.host);
			switch (access.kind) {
			case ORG_RG_ACCESS_SHADER_RESOURCE: builder->WithShaderResource(resource); break;
			case ORG_RG_ACCESS_CONSTANT_BUFFER: builder->WithConstantBuffer(resource); break;
			case ORG_RG_ACCESS_UNORDERED_ACCESS: builder->WithUnorderedAccess(resource); break;
			case ORG_RG_ACCESS_UNORDERED_ACCESS_CLEAR: builder->WithUnorderedAccessClear(resource); break;
			case ORG_RG_ACCESS_INDIRECT_ARGUMENT: builder->WithIndirectArguments(resource); break;
			case ORG_RG_ACCESS_LEGACY_INTEROP: builder->WithLegacyInterop(resource); break;
			default: throw std::runtime_error("Unsupported compute-pass resource access");
			}
		}
	}

	void DeclareGeneric(org::CopyPassBuilder* builder, const ContributorRegistry::Pass& pass, const FrontendState& state)
	{
		for (const auto& access : pass.accesses) {
			auto resource = GenericResourceRange(access, *state.host);
			if (access.kind == ORG_RG_ACCESS_COPY_SOURCE) builder->WithCopySource(resource);
			else if (access.kind == ORG_RG_ACCESS_COPY_DESTINATION) builder->WithCopyDest(resource);
			else throw std::runtime_error("Unsupported copy-pass resource access");
		}
	}

	ORGFrameInfo GenericFrameInfo(const FrontendState& state)
	{
		return { sizeof(ORGFrameInfo), ORG_RENDER_GRAPH_API_CURRENT, state.frame.frameIndex,
			state.frame.generation, state.frame.completionValue, state.frame.frameSlot,
			state.frame.framesInFlight, state.frame.width, state.frame.height,
			state.frame.outputWidth, state.frame.outputHeight };
	}

	ORGStatus QueueGraphBufferUpload(void* user, ORGBinding binding, uint64_t offset,
		const void* data, uint64_t size)
	{
		auto* host = static_cast<ExecutionContextHost*>(user);
		if (!host || !host->graphUploads || !data || !size || size > SIZE_MAX) return ORG_RG_E_INVALID_ARGUMENT;
		const auto found = host->graphResourcesByBinding.find(binding);
		if (found == host->graphResourcesByBinding.end() || !found->second.resource) return ORG_RG_E_STALE_HANDLE;
		if (offset > found->second.byteSize || size > found->second.byteSize - offset)
			return ORG_RG_E_INVALID_ARGUMENT;
		try {
			host->graphUploads->UploadData(data, static_cast<size_t>(size),
				org::runtime::UploadTarget::FromShared(found->second.resource),
				static_cast<size_t>(found->second.baseOffset + offset)
#if BUILD_TYPE == BUILD_TYPE_DEBUG
				, __FILE__, __LINE__
#endif
			);
			return ORG_RG_OK;
		} catch (...) { return ORG_RG_E_INTERNAL; }
	}

	org::GloballyIndexedResource* ResolveGloballyIndexed(org::Resource* resource)
	{
		if (auto* dynamic = dynamic_cast<org::DynamicGloballyIndexedResource*>(resource))
			return dynamic->GetResource().get();
		return dynamic_cast<org::GloballyIndexedResource*>(resource);
	}

	void ResolveGenericDescriptor(const std::shared_ptr<FrontendState>& state,
		const ContributorRegistry::Access& access, org::Resource* resource, ORGBindingInfo& info,
		ExecutionContextHost& host, const std::unordered_map<ORGBinding, rhi::DescriptorSlot>& shaderSlots)
	{
		if (access.viewKind == ORG_RG_VIEW_NONE) return;
		if (const auto found = shaderSlots.find(access.binding); found != shaderSlots.end())
			info.descriptorIndex = found->second.index;
		info.shaderVisible = info.descriptorIndex != UINT32_MAX;
	}

	void PopulateGenericHost(const std::shared_ptr<FrontendState>& state, const ContributorRegistry::Pass& pass,
		ExecutionContextHost& host,
		const std::unordered_map<ORGBinding, rhi::DescriptorSlot>& shaderSlots)
	{
		host.graphUploads = state->host->graphUploads;
		host.uploadUser = &host;
		host.queueBufferUpload = &QueueGraphBufferUpload;
		for (const auto& access : pass.accesses) {
			const auto resource = state->host->resources.find(access.resource);
			if (resource == state->host->resources.end() || !resource->second)
				throw std::runtime_error("Generic binding has no ORG resource");
			const auto definition = std::ranges::find_if(state->genericResources,
				[&](const ContributorRegistry::Resource& value) { return value.handle == access.resource; });
			if (definition == state->genericResources.end())
				throw std::runtime_error("Generic binding has no resource definition");
			host.graphResourcesByBinding.emplace(access.binding,
				ExecutionContextHost::BoundResource{ resource->second, 0, definition->desc.byteSize });
			ORGBindingInfo info{ sizeof(ORGBindingInfo), ORG_RENDER_GRAPH_API_CURRENT };
			info.binding = access.binding; info.access = access.kind; info.viewKind = access.viewKind;
			info.dimension = definition->desc.dimension; info.resourceFormat = definition->desc.format;
			info.viewFormat = access.viewFormat == ORG_RG_FORMAT_UNKNOWN ? definition->desc.format : access.viewFormat;
			info.width = definition->desc.width; info.height = definition->desc.height;
			info.depthOrArraySize = definition->desc.depthOrArraySize; info.mipLevels = definition->desc.mipLevels;
			info.byteSize = definition->desc.byteSize; info.descriptorIndex = UINT32_MAX;
			ResolveGenericDescriptor(state, access, resource->second.get(), info, host, shaderSlots);
			host.bindingInfo.emplace(access.binding, info);
			if (access.counterBinding) {
				const uint64_t counterOffset = (definition->desc.byteSize + 4095u) & ~uint64_t{ 4095u };
				host.graphResourcesByBinding.emplace(access.counterBinding,
					ExecutionContextHost::BoundResource{ resource->second, counterOffset, 4 });
				ORGBindingInfo counter{ sizeof(counter), ORG_RENDER_GRAPH_API_CURRENT };
				counter.binding = access.counterBinding; counter.access = ORG_RG_ACCESS_UNORDERED_ACCESS;
				counter.viewKind = ORG_RG_VIEW_UNORDERED_ACCESS; counter.dimension = ORG_RG_RESOURCE_BUFFER;
				counter.resourceFormat = counter.viewFormat = ORG_RG_FORMAT_R32_UINT;
				counter.byteSize = 4; counter.descriptorIndex = UINT32_MAX;
				if (const auto slot = shaderSlots.find(access.counterBinding); slot != shaderSlots.end()) {
					counter.descriptorIndex = slot->second.index; counter.shaderVisible = 1;
				}
				host.bindingInfo.emplace(access.counterBinding, counter);
			}
		}
	}

	template <class PassBase>
	class GenericProxyPass : public PassBase
	{
	public:
		GenericProxyPass(std::shared_ptr<FrontendState> state, size_t index) : state_(std::move(state)), index_(index) {}
		~GenericProxyPass() override { ReleaseDescriptors(); }
		void Setup() override
		{
			const auto& pass = state_->genericPasses.at(index_);
			for (const auto& access : pass.accesses) CreateDescriptor(access);
			if (pass.prepare) Invoke(pass.prepare, "preparation");
			prepared_ = true;
		}
		void Update(const org::UpdateExecutionContext&) override
		{
			const auto& pass = state_->genericPasses.at(index_);
			if (pass.update) Invoke(pass.update, "update");
		}
		void Cleanup() override
		{
			if (cleaned_) return;
			cleaned_ = true;
			const auto& pass = state_->genericPasses.at(index_);
			if (prepared_ && pass.cleanup) try { pass.cleanup(pass.userData, state_->frame.generation); } catch (...) {}
			ReleaseDescriptors();
		}
		org::PassReturn ExecuteProxy(org::PassExecutionContext& context)
		{
			Invoke(state_->genericPasses.at(index_).execute, "execution");
			return {};
		}
	protected:
		std::shared_ptr<FrontendState> state_;
		size_t index_{};
	private:
		template <class Callback>
		void Invoke(Callback callback, const char* phase)
		{
			const auto& pass = state_->genericPasses.at(index_);
			ExecutionContextHost host{};
			PopulateGenericHost(state_, pass, host, shaderSlots_);
			auto frame = GenericFrameInfo(*state_);
			ORGExecutionContext execution{ sizeof(execution), ORG_RENDER_GRAPH_API_CURRENT,
				ORG_RG_BACKEND_NONE, 0, &frame, &host };
			ORGStatus status = ORG_RG_E_CALLBACK_FAILED;
			try { status = callback(pass.userData, &execution); } catch (...) {}
			if (status != ORG_RG_OK)
				throw std::runtime_error("Generic pass " + std::string(phase) + " failed: " + pass.id);
		}
		bool prepared_{};
		bool cleaned_{};
		std::unordered_map<ORGBinding, rhi::DescriptorSlot> shaderSlots_;
		std::unordered_map<ORGBinding, rhi::DescriptorSlot> cpuSlots_;
		std::vector<rhi::DescriptorSlot> leasedSlots_;
		void ReleaseDescriptors() noexcept {
			if (state_ && state_->descriptorService) {
				for (const auto& slot : leasedSlots_) try { state_->descriptorService->RetireDescriptorSlot(slot); } catch (...) {}
			}
			shaderSlots_.clear(); cpuSlots_.clear(); leasedSlots_.clear();
		}

		void CreateDescriptor(const ContributorRegistry::Access& access)
		{
			if (access.viewKind == ORG_RG_VIEW_NONE) return;
			if (!state_->descriptorService) throw std::runtime_error("ORG descriptor service unavailable");
			const auto resourceIt = state_->host->resources.find(access.resource);
			const auto definitionIt = std::ranges::find_if(state_->genericResources,
				[&](const ContributorRegistry::Resource& value) { return value.handle == access.resource; });
			if (resourceIt == state_->host->resources.end() || !resourceIt->second || definitionIt == state_->genericResources.end())
				throw std::runtime_error("Descriptor binding has no resolved resource");
			auto apiResource = resourceIt->second->GetAPIResource();
			auto format = access.viewFormat == ORG_RG_FORMAT_UNKNOWN ?
				static_cast<rhi::Format>(definitionIt->desc.format) : static_cast<rhi::Format>(access.viewFormat);
			if ((access.viewFlags & ORG_RG_VIEW_FLAG_RAW_BUFFER) != 0) format = rhi::Format::R32_Typeless;
			const uint32_t mipCount = access.range.mipCount == UINT32_MAX ?
				definitionIt->desc.mipLevels - access.range.firstMip : access.range.mipCount;
			const uint32_t arraySize = access.range.arraySize == UINT32_MAX ?
				definitionIt->desc.depthOrArraySize - access.range.firstArraySlice : access.range.arraySize;
			auto allocate = [&](rhi::DescriptorHeapType type, bool shaderVisible) {
				auto slot = state_->descriptorService->AllocateDescriptorSlot(type, shaderVisible);
				leasedSlots_.push_back(slot);
				return slot;
			};
			if (access.viewKind == ORG_RG_VIEW_CONSTANT_BUFFER) {
				auto slot = allocate(rhi::DescriptorHeapType::CbvSrvUav, true);
				rhi::CbvDesc desc{ access.firstElement, access.elementCount == UINT32_MAX ?
					static_cast<uint32_t>(definitionIt->desc.byteSize - access.firstElement) : access.elementCount };
				if (rhi::Failed(state_->device.CreateConstantBufferView(slot, apiResource.GetHandle(), desc)))
					throw std::runtime_error("Failed to create exact ORG CBV");
				shaderSlots_.emplace(access.binding, slot); return;
			}
			if (access.viewKind == ORG_RG_VIEW_SHADER_RESOURCE) {
				auto slot = allocate(rhi::DescriptorHeapType::CbvSrvUav, true); rhi::SrvDesc desc{}; desc.formatOverride = format;
				if (definitionIt->desc.dimension == ORG_RG_RESOURCE_BUFFER) {
					const uint64_t stride = (access.viewFlags & ORG_RG_VIEW_FLAG_RAW_BUFFER) ? 4u :
						access.structureByteStride ? access.structureByteStride : 1u;
					const uint64_t total = definitionIt->desc.byteSize / stride;
					const uint32_t count = access.elementCount == UINT32_MAX ?
						static_cast<uint32_t>(total - access.firstElement) : access.elementCount;
					desc.dimension = rhi::SrvDim::Buffer; desc.buffer.kind = (access.viewFlags & ORG_RG_VIEW_FLAG_RAW_BUFFER) ?
						rhi::BufferViewKind::Raw : access.structureByteStride ? rhi::BufferViewKind::Structured : rhi::BufferViewKind::Typed;
					desc.buffer.firstElement = access.firstElement; desc.buffer.numElements = count;
					desc.buffer.structureByteStride = access.structureByteStride;
				} else if (access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_CUBE) {
					desc.dimension = rhi::SrvDim::TextureCube; desc.cube = { access.range.firstMip, mipCount, 0.0f };
				} else if (access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_CUBE_ARRAY) {
					desc.dimension = rhi::SrvDim::TextureCubeArray; desc.cubeArray = { access.range.firstMip, mipCount,
						access.range.firstArraySlice, arraySize / 6, 0.0f };
				} else if (access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_1D) {
					desc.dimension = rhi::SrvDim::Texture1D; desc.tex1D = { access.range.firstMip, mipCount, 0.0f };
				} else if (access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_1D_ARRAY) {
					desc.dimension = rhi::SrvDim::Texture1DArray; desc.tex1DArray = { access.range.firstMip, mipCount,
						access.range.firstArraySlice, arraySize, 0.0f };
				} else if (access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_2D_MS) {
					desc.dimension = rhi::SrvDim::Texture2DMS;
				} else if (access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_2D_MS_ARRAY) {
					desc.dimension = rhi::SrvDim::Texture2DMSArray; desc.tex2DMSArray = {
						access.range.firstArraySlice, arraySize };
				} else if (access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_3D) {
					desc.dimension = rhi::SrvDim::Texture3D; desc.tex3D = { access.range.firstMip, mipCount, 0.0f };
				} else if (arraySize > 1 || access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_2D_ARRAY) {
					desc.dimension = rhi::SrvDim::Texture2DArray; desc.tex2DArray = { access.range.firstMip, mipCount,
						access.range.firstArraySlice, arraySize, 0, 0.0f };
				} else { desc.dimension = rhi::SrvDim::Texture2D; desc.tex2D = { access.range.firstMip, mipCount, 0, 0.0f }; }
				if (rhi::Failed(state_->device.CreateShaderResourceView(slot, apiResource.GetHandle(), desc)))
					throw std::runtime_error("Failed to create exact ORG SRV");
				shaderSlots_.emplace(access.binding, slot); return;
			}
			if (access.viewKind == ORG_RG_VIEW_UNORDERED_ACCESS) {
				auto slot = allocate(rhi::DescriptorHeapType::CbvSrvUav, true); rhi::UavDesc desc{}; desc.formatOverride = format;
				if (definitionIt->desc.dimension == ORG_RG_RESOURCE_BUFFER) {
					const uint64_t stride = (access.viewFlags & ORG_RG_VIEW_FLAG_RAW_BUFFER) ? 4u :
						access.structureByteStride ? access.structureByteStride : 1u;
					const uint64_t total = definitionIt->desc.byteSize / stride;
					const uint32_t count = access.elementCount == UINT32_MAX ?
						static_cast<uint32_t>(total - access.firstElement) : access.elementCount;
					desc.dimension = rhi::UavDim::Buffer; desc.buffer.kind = (access.viewFlags & ORG_RG_VIEW_FLAG_RAW_BUFFER) ?
						rhi::BufferViewKind::Raw : access.structureByteStride ? rhi::BufferViewKind::Structured : rhi::BufferViewKind::Typed;
					desc.buffer.firstElement = access.firstElement; desc.buffer.numElements = count;
					desc.buffer.structureByteStride = access.structureByteStride;
					if (access.counterBinding)
						desc.buffer.counterOffsetInBytes = (definitionIt->desc.byteSize + 4095u) & ~uint64_t{ 4095u };
				} else if (access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_1D) {
					desc.dimension = rhi::UavDim::Texture1D; desc.texture1D = { access.range.firstMip };
				} else if (access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_1D_ARRAY) {
					desc.dimension = rhi::UavDim::Texture1DArray; desc.texture1DArray = { access.range.firstMip,
						access.range.firstArraySlice, arraySize };
				} else if (access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_3D) {
					desc.dimension = rhi::UavDim::Texture3D; desc.texture3D = { access.range.firstMip,
						access.range.firstArraySlice, arraySize };
				} else if (arraySize > 1 || access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_2D_ARRAY) {
					desc.dimension = rhi::UavDim::Texture2DArray; desc.texture2DArray = { access.range.firstMip,
						access.range.firstArraySlice, arraySize, 0 };
				} else { desc.dimension = rhi::UavDim::Texture2D; desc.texture2D = { access.range.firstMip, 0 }; }
				if (rhi::Failed(state_->device.CreateUnorderedAccessView(slot, apiResource.GetHandle(), desc)))
					throw std::runtime_error("Failed to create exact ORG UAV");
				shaderSlots_.emplace(access.binding, slot);
				if (access.counterBinding) {
					auto counterSlot = allocate(rhi::DescriptorHeapType::CbvSrvUav, true);
					rhi::UavDesc counter{}; counter.dimension = rhi::UavDim::Buffer;
					counter.formatOverride = rhi::Format::R32_Typeless;
					counter.buffer.kind = rhi::BufferViewKind::Raw;
					counter.buffer.firstElement = ((definitionIt->desc.byteSize + 4095u) & ~uint64_t{ 4095u }) / 4u;
					counter.buffer.numElements = 1;
					if (rhi::Failed(state_->device.CreateUnorderedAccessView(counterSlot, apiResource.GetHandle(), counter)))
						throw std::runtime_error("Failed to create embedded ORG counter UAV");
					shaderSlots_.emplace(access.counterBinding, counterSlot);
				}
				if (access.kind == ORG_RG_ACCESS_UNORDERED_ACCESS_CLEAR) {
					auto cpu = allocate(rhi::DescriptorHeapType::CbvSrvUav, false);
					if (rhi::Failed(state_->device.CreateUnorderedAccessView(cpu, apiResource.GetHandle(), desc)))
						throw std::runtime_error("Failed to create exact CPU ORG UAV");
					cpuSlots_.emplace(access.binding, cpu);
				}
				return;
			}
			if (access.viewKind == ORG_RG_VIEW_RENDER_TARGET) {
				auto slot = allocate(rhi::DescriptorHeapType::RTV, false); rhi::RtvDesc desc{}; desc.formatOverride = format;
				switch (access.viewDimension) {
				case ORG_RG_VIEW_DIMENSION_TEXTURE_1D: desc.dimension = rhi::RtvDim::Texture1D; break;
				case ORG_RG_VIEW_DIMENSION_TEXTURE_1D_ARRAY: desc.dimension = rhi::RtvDim::Texture1DArray; break;
				case ORG_RG_VIEW_DIMENSION_TEXTURE_2D_MS: desc.dimension = rhi::RtvDim::Texture2DMS; break;
				case ORG_RG_VIEW_DIMENSION_TEXTURE_2D_MS_ARRAY: desc.dimension = rhi::RtvDim::Texture2DMSArray; break;
				case ORG_RG_VIEW_DIMENSION_TEXTURE_3D: desc.dimension = rhi::RtvDim::Texture3D; break;
				case ORG_RG_VIEW_DIMENSION_TEXTURE_2D_ARRAY: desc.dimension = rhi::RtvDim::Texture2DArray; break;
				default: desc.dimension = rhi::RtvDim::Texture2D; break;
				}
				desc.range = { access.range.firstMip, 1, access.range.firstArraySlice, arraySize };
				if (rhi::Failed(state_->device.CreateRenderTargetView(slot, apiResource.GetHandle(), desc)))
					throw std::runtime_error("Failed to create exact ORG RTV");
				cpuSlots_.emplace(access.binding, slot); return;
			}
			auto slot = allocate(rhi::DescriptorHeapType::DSV, false); rhi::DsvDesc desc{}; desc.formatOverride = format;
			switch (access.viewDimension) {
			case ORG_RG_VIEW_DIMENSION_TEXTURE_1D: desc.dimension = rhi::DsvDim::Texture1D; break;
			case ORG_RG_VIEW_DIMENSION_TEXTURE_1D_ARRAY: desc.dimension = rhi::DsvDim::Texture1DArray; break;
			case ORG_RG_VIEW_DIMENSION_TEXTURE_2D_MS: desc.dimension = rhi::DsvDim::Texture2DMS; break;
			case ORG_RG_VIEW_DIMENSION_TEXTURE_2D_MS_ARRAY: desc.dimension = rhi::DsvDim::Texture2DMSArray; break;
			case ORG_RG_VIEW_DIMENSION_TEXTURE_2D_ARRAY: desc.dimension = rhi::DsvDim::Texture2DArray; break;
			default: desc.dimension = rhi::DsvDim::Texture2D; break;
			}
			desc.range = { access.range.firstMip, 1, access.range.firstArraySlice, arraySize };
			desc.readOnlyDepth = (access.viewFlags & ORG_RG_VIEW_FLAG_READ_ONLY_DEPTH) != 0;
			desc.readOnlyStencil = (access.viewFlags & ORG_RG_VIEW_FLAG_READ_ONLY_STENCIL) != 0;
			if (rhi::Failed(state_->device.CreateDepthStencilView(slot, apiResource.GetHandle(), desc)))
				throw std::runtime_error("Failed to create exact ORG DSV");
			cpuSlots_.emplace(access.binding, slot);
		}
	};

	class GenericRenderPass final : public GenericProxyPass<org::RenderPass>
	{
	public:
		using GenericProxyPass::GenericProxyPass;
		void DeclareResourceUsages(org::RenderPassBuilder* builder) override { DeclareGeneric(builder, state_->genericPasses.at(index_), *state_); }
		org::PassReturn Execute(org::PassExecutionContext& context) override { return ExecuteProxy(context); }
	};

	class GenericComputePass final : public GenericProxyPass<org::ComputePass>
	{
	public:
		using GenericProxyPass::GenericProxyPass;
		void DeclareResourceUsages(org::ComputePassBuilder* builder) override { DeclareGeneric(builder, state_->genericPasses.at(index_), *state_); }
		org::PassReturn Execute(org::PassExecutionContext& context) override { return ExecuteProxy(context); }
	};

	class GenericCopyPass final : public GenericProxyPass<org::CopyPass>
	{
	public:
		using GenericProxyPass::GenericProxyPass;
		void DeclareResourceUsages(org::CopyPassBuilder* builder) override { DeclareGeneric(builder, state_->genericPasses.at(index_), *state_); }
		org::PassReturn Execute(org::PassExecutionContext& context) override { return ExecuteProxy(context); }
	};

	class AnchorPass final : public org::ComputePass
	{
	public:
		void Setup() override {}
		void Cleanup() override {}
		void DeclareResourceUsages(org::ComputePassBuilder*) override {}
		org::PassReturn Execute(org::PassExecutionContext&) override { return {}; }
	};

	class BoundaryPass final : public org::ComputePass, public org::IDynamicDeclaredResources
	{
	public:
		BoundaryPass(std::shared_ptr<FrontendState> state, bool begin) : state(std::move(state)), begin(begin) {}
		void Setup() override {}
		void Cleanup() override {}
		bool DeclaredResourcesChanged() const override { return true; }
		bool RequiresPassRebindAfterDeclarationRefresh() const noexcept override { return false; }
		void DeclareResourceUsages(org::ComputePassBuilder* builder) override
		{
			(void)builder;
		}
		org::PassReturn Execute(org::PassExecutionContext&) override
		{
			org::PassReturn result{};
			if (!begin) result.externalSignalsAfterCompletion.push_back({ state->completeTimeline, state->completeValue });
			return result;
		}
	private:
		std::shared_ptr<FrontendState> state;
		bool begin{};
	};

	class FrontendExtension final : public org::RenderGraph::IRenderGraphExtension
	{
	public:
		explicit FrontendExtension(std::vector<std::string> anchors) : state(std::make_shared<FrontendState>()) {
			state->anchors = std::move(anchors);
		}
		void SetFrame(FrontendState value) {
			state->readyTimeline = value.readyTimeline; state->readyValue = value.readyValue;
			state->completeTimeline = value.completeTimeline; state->completeValue = value.completeValue;
			state->frame = value.frame;
			state->host = value.host;
		}
		void SetGenericPasses(std::vector<ContributorRegistry::Pass> value) { state->genericPasses = std::move(value); }
		void SetGenericResources(std::vector<ContributorRegistry::Resource> value) { state->genericResources = std::move(value); }
		void SetDescriptorContext(rhi::Device device, org::runtime::IDescriptorService& descriptors) {
			state->device = device;
			state->descriptorService = &descriptors;
		}
		std::shared_ptr<FrontendState> GetState() const { return state; }
		void GatherStructuralPasses(org::RenderGraph&, std::vector<org::RenderGraph::ExternalPassDesc>& out) override {
			auto begin = org::RenderGraph::ExternalPassDesc::Compute("org.contributor.begin", std::make_shared<BoundaryPass>(state, true))
				.PreferQueue(org::QueueKind::Graphics).CollectStatistics(false);
			auto end = org::RenderGraph::ExternalPassDesc::Compute("org.contributor.end", std::make_shared<BoundaryPass>(state, false))
				.PreferQueue(org::QueueKind::Graphics).CollectStatistics(false);
			out.push_back(std::move(begin));
			std::string previous = "org.contributor.begin";
			for (const auto& anchor : state->anchors) {
				auto anchorDesc = org::RenderGraph::ExternalPassDesc::Compute(anchor, std::make_shared<AnchorPass>())
					.At(org::RenderGraph::ExternalInsertPoint::After(previous)).PreferQueue(org::QueueKind::Graphics).CollectStatistics(false);
				out.push_back(std::move(anchorDesc));
				previous = anchor;
			}
			std::string previousSerialized;
			for (size_t i = 0; i < state->genericPasses.size(); ++i) {
				const auto& item = state->genericPasses[i];
				auto point = org::RenderGraph::ExternalInsertPoint::After("org.contributor.begin");
				point.keepExtensionOrder = false;
				point.priority = item.priority;
				for (const auto& dependency : item.after) point.AlsoAfter(dependency);
				for (const auto& dependency : item.before) point.AlsoBefore(dependency);
				if ((item.flags & ORG_RG_PASS_PARALLEL_RECORDING_SAFE) == 0 && !previousSerialized.empty())
					point.AlsoAfter(previousSerialized);
				org::RenderGraph::ExternalPassDesc desc{};
				switch (item.kind) {
				case ORG_RG_PASS_RENDER: desc = org::RenderGraph::ExternalPassDesc::Render(
					item.id, std::make_shared<GenericRenderPass>(state, i)); break;
				case ORG_RG_PASS_COMPUTE: desc = org::RenderGraph::ExternalPassDesc::Compute(
					item.id, std::make_shared<GenericComputePass>(state, i)); break;
				case ORG_RG_PASS_COPY: desc = org::RenderGraph::ExternalPassDesc::Copy(
					item.id, std::make_shared<GenericCopyPass>(state, i)); break;
				default: throw std::runtime_error("Unknown generic pass kind");
				}
				desc.At(std::move(point));
				switch (item.queue) {
				case ORG_RG_QUEUE_AUTOMATIC: desc.AutomaticQueueAssignment(); break;
				case ORG_RG_QUEUE_FORCE_GRAPHICS: desc.PreferQueue(org::QueueKind::Graphics); break;
				case ORG_RG_QUEUE_FORCE_COMPUTE: desc.PreferQueue(org::QueueKind::Compute); break;
				case ORG_RG_QUEUE_FORCE_COPY: desc.PreferQueue(org::QueueKind::Copy); break;
				default: throw std::runtime_error("Unknown generic queue assignment");
				}
				desc.CollectStatistics((item.flags & ORG_RG_PASS_DISABLE_STATISTICS) == 0)
					.GeometryPass((item.flags & ORG_RG_PASS_GEOMETRY) != 0);
				if (!item.technique.empty()) desc.Technique(item.technique);
				out.push_back(std::move(desc));
				if ((item.flags & ORG_RG_PASS_PARALLEL_RECORDING_SAFE) == 0) previousSerialized = item.id;
			}
			auto endPoint = org::RenderGraph::ExternalInsertPoint::After(
				state->anchors.empty() ? "org.contributor.begin" : state->anchors.back());
			endPoint.keepExtensionOrder = false;
			for (const auto& item : state->genericPasses) endPoint.AlsoAfter(item.id);
			end.At(std::move(endPoint));
			out.push_back(std::move(end));
		}
	private:
		std::shared_ptr<FrontendState> state;
	};

	class InternalIOExtension final : public org::RenderGraph::IRenderGraphExtension
	{
	public:
		explicit InternalIOExtension(org::runtime::IUploadService* uploads) : uploads_(uploads) {}
		void OnRegistryReset(org::ResourceRegistry* registry) override {
			if (uploads_) uploads_->SetUploadResolveContext({ registry, 0 });
		}
		void GatherStructuralPasses(org::RenderGraph&, std::vector<org::RenderGraph::ExternalPassDesc>& out) override {
			if (uploads_) if (auto pass = uploads_->GetUploadPass())
				out.push_back(org::RenderGraph::ExternalPassDesc::Render("org.runtime.uploads", std::move(pass))
					.At(org::RenderGraph::ExternalInsertPoint::Begin(-1000)).CollectStatistics(false));
		}
	private:
		org::runtime::IUploadService* uploads_{};
	};
}

class ContributorGraphHost::Impl
{
public:
	struct GraphGeneration
	{
		struct NativeLifecycle { ExtensionRegistry::GenerationCallback activated; ExtensionRegistry::GenerationCallback retired; };
		std::unique_ptr<org::RenderGraph> graph;
		FrontendExtension* frontendExtension{};
		ExecutionHost host{};
		std::vector<NativeLifecycle> nativeLifecycle;
		uint64_t generation{};
		bool activationDelivered{};
		uint64_t lastCompletion{};
		~GraphGeneration() {
			if (graph) { graph->ShutdownExtensions(); graph.reset(); }
			if (activationDelivered) for (auto& lifecycle : nativeLifecycle)
				if (lifecycle.retired) try { lifecycle.retired(generation); } catch (...) {}
		}
		void Activate(uint64_t value) {
			if (activationDelivered) return;
			generation = value; activationDelivered = true;
			for (auto& lifecycle : nativeLifecycle)
				if (lifecycle.activated) try { lifecycle.activated(generation); } catch (...) {}
		}
	};

		explicit Impl(rhi::Device runtimeDevice, ExtensionRegistry* registry, std::vector<std::string> hostAnchors)
			: extensions(registry), anchors(std::move(hostAnchors)), device(runtimeDevice)
	{
		auto settings = org::runtime::GetOpenRenderGraphSettings();
		settings.collectPassStatistics = true;
		settings.collectPipelineStatistics = false;
		settings.renderGraphBatchTraceEnabled = false;
		org::runtime::SetOpenRenderGraphSettings(settings);
		org::runtime::InitializeRuntimeDevice(device);
		// The host has no useful graph until the registry publishes its first
		// complete candidate.  Creating an empty bootstrap generation here made
		// its resources overlap the first real generation and then retired them
		// at the first completion boundary.  Start without an active generation;
		// SetStructuralDefinition publishes the first fully prepared graph.
	}

	~Impl()
	{
		org::runtime::SetActiveDescriptorService(nullptr);
		org::runtime::SetActiveUploadService(nullptr);
		active.reset();
		retired.clear();
		org::runtime::ShutdownRuntimeDevice();
	}

	std::unique_ptr<GraphGeneration> Compile(const ContributorRegistry::Candidate& candidate)
	{
		auto generation = std::make_unique<GraphGeneration>();
		generation->graph = std::make_unique<org::RenderGraph>(device);
		generation->host.graphUploads = generation->graph->GetUploadService();
		generation->graph->RegisterExtension(
			std::make_unique<InternalIOExtension>(generation->host.graphUploads), "org.runtime.io");
		for (const auto& definition : candidate.resources) {
			std::shared_ptr<org::Resource> resource;
			if (definition.desc.lifetime == ORG_RG_RESOURCE_PERSISTENT && active) {
				const auto previous = active->host.persistentResources.find(definition.id);
				if (previous != active->host.persistentResources.end() &&
					StructurallyCompatible(previous->second.description, definition.desc))
					resource = previous->second.resource;
			}
			if (!resource && definition.desc.dimension == ORG_RG_RESOURCE_BUFFER) {
				const bool unordered = (definition.desc.allowedUsages & ORG_RG_USAGE_UNORDERED_ACCESS) != 0;
				const bool hasCounter = std::ranges::any_of(candidate.passes, [&](const ContributorRegistry::Pass& pass) {
					return std::ranges::any_of(pass.accesses, [&](const ContributorRegistry::Access& access) {
						return access.resource == definition.handle && access.counterBinding != 0;
					});
				});
					const auto heap = definition.desc.heapClass == ORG_RG_HEAP_UPLOAD ? rhi::HeapType::Upload :
						definition.desc.heapClass == ORG_RG_HEAP_READBACK ? rhi::HeapType::Readback : rhi::HeapType::DeviceLocal;
					std::shared_ptr<org::Buffer> buffer;
					if (definition.desc.structureByteStride) {
						const auto count = definition.desc.byteSize / definition.desc.structureByteStride;
						if (count > UINT32_MAX) throw std::runtime_error("Structured buffer element count exceeds ORG limits");
						buffer = org::Buffer::CreateUnmaterializedStructuredBuffer(static_cast<uint32_t>(count),
							definition.desc.structureByteStride, unordered, hasCounter,
							(definition.desc.allowedUsages & ORG_RG_USAGE_UNORDERED_ACCESS) != 0, heap);
					} else {
						buffer = org::Buffer::CreateSharedUnmaterialized(heap, definition.desc.byteSize, unordered);
						org::BufferBase::DescriptorRequirements requirements{};
						const bool shaderRead = (definition.desc.allowedUsages & ORG_RG_USAGE_SHADER_RESOURCE) != 0;
						const bool constant = (definition.desc.allowedUsages & ORG_RG_USAGE_CONSTANT_BUFFER) != 0;
						requirements.createSRV = shaderRead; requirements.createCBV = constant;
						requirements.createUAV = unordered;
						requirements.createNonShaderVisibleUAV = unordered;
						if (shaderRead) requirements.srvDesc = { .dimension = rhi::SrvDim::Buffer,
							.formatOverride = rhi::Format::R32_Typeless,
							.buffer = { .kind = rhi::BufferViewKind::Raw, .firstElement = 0,
								.numElements = static_cast<uint32_t>(definition.desc.byteSize / 4) } };
						if (unordered) requirements.uavDesc = { .dimension = rhi::UavDim::Buffer,
							.formatOverride = rhi::Format::R32_Typeless,
							.buffer = { .kind = rhi::BufferViewKind::Raw, .firstElement = 0,
								.numElements = static_cast<uint32_t>(definition.desc.byteSize / 4) } };
						if (constant) requirements.cbvDesc.byteSize = static_cast<uint32_t>(definition.desc.byteSize);
						if (shaderRead || unordered || constant) buffer->SetDescriptorRequirements(requirements);
					}
					buffer->SetName(definition.id);
					buffer->SetAllowAlias(definition.desc.allowAlias != 0);
					if (definition.desc.aliasingPool) buffer->SetAliasingPool(definition.desc.aliasingPool);
				resource = std::move(buffer);
			} else if (!resource && (definition.desc.dimension == ORG_RG_RESOURCE_TEXTURE_1D ||
				definition.desc.dimension == ORG_RG_RESOURCE_TEXTURE_2D ||
				definition.desc.dimension == ORG_RG_RESOURCE_TEXTURE_3D)) {
				org::TextureDescription texture{};
				texture.imageDimensions.resize(definition.desc.mipLevels);
				uint32_t width = definition.desc.width, height = definition.desc.height;
				for (auto& dimensions : texture.imageDimensions) {
					dimensions.width = width; dimensions.height = height;
					width = (std::max)(1u, width >> 1); height = (std::max)(1u, height >> 1);
				}
				texture.format = static_cast<rhi::Format>(definition.desc.format);
				texture.type = definition.desc.dimension == ORG_RG_RESOURCE_TEXTURE_1D ? rhi::ResourceType::Texture1D :
					definition.desc.dimension == ORG_RG_RESOURCE_TEXTURE_3D ? rhi::ResourceType::Texture3D : rhi::ResourceType::Texture2D;
				texture.channels = FormatChannels(definition.desc.format);
				texture.depth = texture.type == rhi::ResourceType::Texture3D ? definition.desc.depthOrArraySize : 1u;
				texture.isCubemap = std::ranges::any_of(candidate.passes, [&](const ContributorRegistry::Pass& pass) {
					return std::ranges::any_of(pass.accesses, [&](const ContributorRegistry::Access& access) {
						return access.resource == definition.handle &&
							(access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_CUBE ||
							 access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_CUBE_ARRAY);
					});
				});
				texture.arraySize = texture.type == rhi::ResourceType::Texture3D ? 1u :
					(texture.isCubemap ? definition.desc.depthOrArraySize / 6u : definition.desc.depthOrArraySize);
				texture.isArray = texture.type != rhi::ResourceType::Texture3D &&
					(texture.isCubemap ? texture.arraySize > 1 : texture.arraySize > 1);
				texture.sampleCount = static_cast<uint8_t>(definition.desc.sampleCount);
				texture.hasSRV = (definition.desc.allowedUsages & ORG_RG_USAGE_SHADER_RESOURCE) != 0;
				texture.hasUAV = (definition.desc.allowedUsages & ORG_RG_USAGE_UNORDERED_ACCESS) != 0;
				texture.hasRTV = (definition.desc.allowedUsages & ORG_RG_USAGE_RENDER_TARGET) != 0;
				texture.hasDSV = (definition.desc.allowedUsages & (ORG_RG_USAGE_DEPTH_READ | ORG_RG_USAGE_DEPTH_WRITE)) != 0;
				texture.allowAlias = definition.desc.allowAlias != 0;
				if (definition.desc.aliasingPool) texture.aliasingPoolID = definition.desc.aliasingPool;
				resource = org::PixelBuffer::CreateShared(texture);
			} else if (!resource) throw std::runtime_error("Unsupported generic ORG resource dimension");
			const org::ResourceIdentifier identifier{ definition.id };
			generation->graph->RegisterResource(identifier, resource);
			generation->host.resources.emplace(definition.handle, std::move(resource));
			generation->host.resourceIdentifiers.emplace(definition.handle, identifier);
			if (definition.desc.lifetime == ORG_RG_RESOURCE_PERSISTENT)
				generation->host.persistentResources.emplace(definition.id,
					ExecutionHost::PersistentResource{ definition.desc, generation->host.resources.at(definition.handle) });
		}
		std::unordered_set<std::string> availableResources;
		for (const auto& definition : candidate.resources)
			availableResources.insert(definition.id);
		auto nativeExtensions = extensions ? extensions->BuildCandidate(availableResources) :
			std::vector<ExtensionRegistry::Installed>{};
		for (auto& installed : nativeExtensions) {
			generation->nativeLifecycle.push_back({ std::move(installed.activated), std::move(installed.retired) });
			generation->graph->RegisterExtension(std::move(installed.extension), installed.id);
		}
		auto extension = std::make_unique<FrontendExtension>(anchors);
		generation->frontendExtension = extension.get();
		extension->SetFrame(FrontendState{ .frame = { .generation = candidate.generation }, .host = &generation->host });
		extension->SetGenericPasses(candidate.passes);
		extension->SetGenericResources(candidate.resources);
		extension->SetDescriptorContext(device, *generation->graph->GetDescriptorService());
		generation->graph->RegisterExtension(std::move(extension), "org.contributor.frontend");
		ActiveGraphServices services(generation->graph->GetUploadService(),
			generation->graph->GetDescriptorService());
		// Native extensions register their symbolic providers here. ORG keeps this
		// step explicit so applications can finish installing every extension
		// before any provider resolves a cross-extension resource.
		generation->graph->PrepareExtensionsForBuild();
		generation->graph->GetMemorySnapshotProvider().SetProvider(org::memory::CreateECSMemorySnapshotProvider());
		generation->graph->CompileStructural();
		// Structural compilation gathers the extension passes. Setup must follow
		// it so those passes receive registry views, descriptor helpers, and their
		// exactly-once Setup callback before the candidate can activate.
		generation->graph->Setup();
		return generation;
	}

	std::unique_ptr<GraphGeneration> active;
	std::deque<std::pair<uint64_t, std::unique_ptr<GraphGeneration>>> retired;
	ExtensionRegistry* extensions{};
	std::vector<std::string> anchors;
	rhi::Device device{};
};

ContributorGraphHost::ContributorGraphHost(std::unique_ptr<Impl> implementation) : impl(std::move(implementation)) {}
ContributorGraphHost::~ContributorGraphHost() = default;

std::unique_ptr<ContributorGraphHost> ContributorGraphHost::Create(
	rhi::Device device, ExtensionRegistry* extensions, std::vector<std::string> anchors)
{
	return std::unique_ptr<ContributorGraphHost>(new ContributorGraphHost(
		std::make_unique<Impl>(device, extensions, std::move(anchors))));
}

void ContributorGraphHost::SetStructuralDefinition(const ContributorRegistry::Candidate& genericCandidate)
{
	auto candidate = impl->Compile(genericCandidate);
	if (impl->active) impl->retired.emplace_back(impl->active->lastCompletion, std::move(impl->active));
	impl->active = std::move(candidate);
}

void ContributorGraphHost::Execute(
	uint32_t frameIndex,
	uint64_t frameFenceValue,
	rhi::Timeline readyTimeline,
	uint64_t readyValue,
	rhi::Timeline completeTimeline,
	uint64_t completeValue,
	const FrameInfo& frame)
{
	FrontendState state{}; state.readyTimeline = readyTimeline; state.readyValue = readyValue;
	state.completeTimeline = completeTimeline; state.completeValue = completeValue; state.frame = frame;
	state.host = &impl->active->host;
	impl->active->Activate(frame.generation);
	impl->active->frontendExtension->SetFrame(std::move(state));
	ActiveGraphServices services(impl->active->graph->GetUploadService(),
		impl->active->graph->GetDescriptorService());
	org::UpdateExecutionContext update{};
	update.frameIndex = frameIndex;
	update.frameFenceValue = frameFenceValue;
	impl->active->graph->Update(update, impl->device);
	org::PassExecutionContext execute{};
	execute.device = impl->device;
	execute.frameIndex = frameIndex;
	execute.frameFenceValue = frameFenceValue;
	execute.externalTimelineBindings.push_back({ kD3D11ReadyBinding, { readyTimeline, readyValue } });
	impl->active->graph->Execute(execute);
	impl->active->lastCompletion = completeValue;
}

void ContributorGraphHost::Retire(uint64_t completedValue) noexcept {
	while (!impl->retired.empty() && impl->retired.front().first <= completedValue) impl->retired.pop_front();
}

} // namespace org::contributor
