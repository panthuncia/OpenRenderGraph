#include "RenderPasses/Base/RenderPass.h"

#include "Render/Runtime/IDescriptorService.h"
#include "Render/Runtime/IUploadService.h"
#include "Render/Runtime/DescriptorServiceAccess.h"

namespace org {

RenderGraphPass::~RenderGraphPass() = default;

runtime::IUploadService& RenderGraphPass::UploadService() const {
    auto service = m_uploadService.lock();
    if (!service) throw std::runtime_error("Pass upload service generation is unavailable");
    return *service;
}

runtime::IDescriptorService& RenderGraphPass::DescriptorService() const {
    auto service = m_descriptorService.lock();
    if (service) return *service;
    // Pass construction can precede graph service binding during startup;
    // retain the weak binding contract while allowing that initialization
    // window to resolve through the graph's active generation.
    if (auto* active = runtime::GetActiveDescriptorService()) return *active;
    throw std::runtime_error("Pass descriptor service generation is unavailable");
}

void RenderGraphPass::UploadBufferData(const void* data, size_t size,
    runtime::UploadTarget target, size_t offset, std::source_location source) const {
    UploadService().UploadData(data, size, std::move(target), offset,
        source.file_name(), static_cast<int>(source.line()));
}

RenderPass::~RenderPass() = default;

void RenderPass::SetResourceRegistryView(
	std::shared_ptr<ResourceRegistryView> resourceRegistryView,
	std::unordered_set<FeatureDomainIdentifier, FeatureDomainIdentifier::Hasher> activeFeatureDomains) {
	m_resourceRegistryView = resourceRegistryView;
	m_resourceDescriptorIndexHelper = std::make_unique<ResourceDescriptorIndexHelper>(resourceRegistryView, std::move(activeFeatureDomains));
}

void RenderPass::SetResourceRegistryView(
	std::shared_ptr<ResourceRegistryView> resourceRegistryView,
	const std::unordered_set<FeatureDomainIdentifier, FeatureDomainIdentifier::Hasher>& activeFeatureDomains,
	const std::vector<AutoDescriptorRegistration>& autoDescriptorShaderResources,
	const std::vector<AutoDescriptorRegistration>& autoDescriptorConstantBuffers,
	const std::vector<AutoDescriptorRegistration>& autoDescriptorUnorderedAccessViews) {
	SetResourceRegistryView(std::move(resourceRegistryView), activeFeatureDomains);
	for (const auto& registration : autoDescriptorShaderResources) {
		m_resourceDescriptorIndexHelper->RegisterDescriptor(registration);
	}
	for (const auto& registration : autoDescriptorConstantBuffers) {
		m_resourceDescriptorIndexHelper->RegisterDescriptor(registration);
	}
	for (const auto& registration : autoDescriptorUnorderedAccessViews) {
		m_resourceDescriptorIndexHelper->RegisterDescriptor(registration);
	}
}

void RenderPass::ConfigureResourceRegistryView(
	std::shared_ptr<ResourceRegistryView> view,
	const PassParameters& parameters) {
	SetResourceRegistryView(std::move(view), parameters.activeFeatureDomains,
		parameters.autoDescriptorShaderResources,
		parameters.autoDescriptorConstantBuffers,
		parameters.autoDescriptorUnorderedAccessViews);
}

unsigned int RenderPass::ResolveResourceDescriptorIndex(const ResourceIdentifier& binding, bool optional) const {
	return m_resourceDescriptorIndexHelper->GetResourceDescriptorIndex(binding, optional);
}

void RenderPass::BindResourceDescriptorIndices(rhi::CommandList& commandList, const PipelineResources& resources) {
	unsigned int indices[org::shaderapi::kNumResourceDescriptorIndicesRootConstants] = {};
	int i = 0;
	for (auto& binding : resources.mandatoryResourceDescriptorSlots) {
		indices[i] = ResolveResourceDescriptorIndex(binding, false);
		i++;
	}
	for (auto& binding : resources.optionalResourceDescriptorSlots) {
		indices[i] = ResolveResourceDescriptorIndex(binding, true);
		i++;
	}
	if (i > 0) {
		commandList.PushConstants(rhi::ShaderStage::All, 0, org::shaderapi::kResourceDescriptorIndicesRootParameter, 0, i, indices);
	}
}

std::vector<unsigned int> RenderPass::CaptureResourceDescriptorIndices(const PipelineResources& resources,
	PreparedDescriptorIndexCache* cache) const {
	auto resolve = [this](const ResourceIdentifier& binding, bool optional) {
		return ResolveResourceDescriptorIndex(binding, optional);
	};
	std::vector<unsigned int> indices;
	indices.reserve(resources.mandatoryResourceDescriptorSlots.size()
		+ resources.optionalResourceDescriptorSlots.size());
	for (const auto& binding : resources.mandatoryResourceDescriptorSlots)
		indices.push_back(cache ? cache->Resolve(binding, false, resolve) : resolve(binding, false));
	for (const auto& binding : resources.optionalResourceDescriptorSlots)
		indices.push_back(cache ? cache->Resolve(binding, true, resolve) : resolve(binding, true));
	return indices;
}

} // namespace org
