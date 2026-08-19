#include "Render/PassExecutionContext.h"
#include "Resources/Resource.h"
#include "Resources/GloballyIndexedResource.h"
#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Render/PipelineState.h"

namespace org {

rhi::Resource PassExecutionContext::Resolve(Resource& resource) const {
	return resource.GetAPIResource(backendInstance);
}

rhi::Resource PassExecutionContext::Resolve(const std::shared_ptr<Resource>& resource) const {
	return resource ? resource->GetAPIResource(backendInstance) : rhi::Resource{};
}

SymbolicTracker* PassExecutionContext::ResolveState(Resource& resource) const {
	return resource.GetStateTracker(backendInstance);
}

rhi::DescriptorHeap PassExecutionContext::GetResourceDescriptorHeap() const {
	return DescriptorHeapManager::GetInstance().GetSRVDescriptorHeap(backendInstance);
}

rhi::DescriptorHeap PassExecutionContext::GetSamplerDescriptorHeap() const {
	return DescriptorHeapManager::GetInstance().GetSamplerDescriptorHeap(backendInstance);
}

rhi::DescriptorSlot PassExecutionContext::ResolveRTV(const GloballyIndexedResource& resource, uint32_t mip, uint32_t slice) const {
	if (!resource.HasRTV()) return {};
	return DescriptorHeapManager::GetInstance().ResolveDescriptorSlot(
		backendInstance, rhi::DescriptorHeapType::RTV, false, resource.GetRTVInfo(mip, slice).slot.index);
}

rhi::DescriptorSlot PassExecutionContext::ResolveDSV(const GloballyIndexedResource& resource, uint32_t mip, uint32_t slice) const {
	if (!resource.HasDSV()) return {};
	return DescriptorHeapManager::GetInstance().ResolveDescriptorSlot(
		backendInstance, rhi::DescriptorHeapType::DSV, false, resource.GetDSVInfo(mip, slice).slot.index);
}

rhi::DescriptorSlot PassExecutionContext::ResolveNonShaderVisibleUAV(const GloballyIndexedResource& resource, uint32_t mip, uint32_t slice) const {
	if (!resource.HasUAVNonShaderVisible()) return {};
	return DescriptorHeapManager::GetInstance().ResolveDescriptorSlot(
		backendInstance, rhi::DescriptorHeapType::CbvSrvUav, false,
		resource.GetUAVNonShaderVisibleInfo(mip, slice).slot.index);
}

const rhi::Pipeline& PassExecutionContext::ResolvePipeline(const PipelineState& pipeline) const {
	return pipeline.GetAPIPipelineState(backendInstance);
}

} // namespace org
