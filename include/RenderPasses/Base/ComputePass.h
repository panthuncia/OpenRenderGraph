#pragma once
#include "Render/PreparedPass.h"
#include "RenderPasses/Base/RenderPass.h"

#include <vector>
#include <unordered_set>
#include <rhi.h>

#include "Resources/Resource.h"
#include "Render/ResourceRequirements.h"
#include "RenderPasses/Base/PassReturn.h"
#include "Resources/ResourceStateTracker.h"
#include "Resources/ResourceIdentifier.h"
#include "Render/ResourceRegistry.h"
#include "ResourceDescriptorIndexHelper.h"
#include "Render/PipelineState.h"
#include "Interfaces/IResourceProvider.h"
#include "Render/PassInputs.h"
#include "Render/PassExecutionContext.h"
#include "Render/FeatureDomainRegistry.h"
#include "Render/ShaderAPI.h"
#include "Render/QueueKind.h"


namespace org {

using ComputePassParameters = PassParameters;

class ComputePassBuilder;

class ComputePass : public RenderGraphPass {
public:
	virtual ~ComputePass() = default;

	void SetResourceRegistryView(
		std::shared_ptr<ResourceRegistryView> resourceRegistryView,
		std::unordered_set<FeatureDomainIdentifier, FeatureDomainIdentifier::Hasher> activeFeatureDomains = {}) {
		m_resourceRegistryView = resourceRegistryView;
		m_resourceDescriptorIndexHelper = std::make_unique<ResourceDescriptorIndexHelper>(resourceRegistryView, std::move(activeFeatureDomains));
	}

	void SetResourceRegistryView(
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

    void ConfigureResourceRegistryView(
        std::shared_ptr<ResourceRegistryView> view,
        const PassParameters& parameters) override {
        SetResourceRegistryView(std::move(view), parameters.activeFeatureDomains,
            parameters.autoDescriptorShaderResources,
            parameters.autoDescriptorConstantBuffers,
            parameters.autoDescriptorUnorderedAccessViews);
    }

	virtual void Setup() = 0;
    // Preparation owner only. Empty means explicit synchronous legacy fallback.
    virtual PreparedPass PrepareFrame(FramePreparationContext&) { return {}; }

	virtual void Update(const UpdateExecutionContext& context) {};
	virtual PassReturn Execute(PassExecutionContext& context) { return {}; };
	virtual void Cleanup() = 0;

	void Invalidate() override { invalidated = true; }
	bool IsInvalidated() const override { return invalidated; }

protected:
	bool invalidated = true;
	virtual void DeclareResourceUsages(ComputePassBuilder* builder) {};

	template<class CommandSink>
	void BindResourceDescriptorIndices(CommandSink& commandList, const PipelineResources& resources) {
		unsigned int indices[org::shaderapi::kNumResourceDescriptorIndicesRootConstants] = {};
		int i = 0;
		for (auto& binding : resources.mandatoryResourceDescriptorSlots) {
			indices[i] = m_resourceDescriptorIndexHelper->GetResourceDescriptorIndex(binding, false);
			i++;
		}
		for (auto& binding : resources.optionalResourceDescriptorSlots) {
			indices[i] = m_resourceDescriptorIndexHelper->GetResourceDescriptorIndex(binding, true);
			i++;
		}
		if (i > 0) {
			commandList.PushConstants(rhi::ShaderStage::Compute, 0, org::shaderapi::kResourceDescriptorIndicesRootParameter, 0, i, indices);
		}
	}

	std::vector<unsigned int> CaptureResourceDescriptorIndices(const PipelineResources& resources) const {
		std::vector<unsigned int> indices;
		indices.reserve(resources.mandatoryResourceDescriptorSlots.size()
			+ resources.optionalResourceDescriptorSlots.size());
		for (const auto& binding : resources.mandatoryResourceDescriptorSlots)
			indices.push_back(m_resourceDescriptorIndexHelper->GetResourceDescriptorIndex(binding, false));
		for (const auto& binding : resources.optionalResourceDescriptorSlots)
			indices.push_back(m_resourceDescriptorIndexHelper->GetResourceDescriptorIndex(binding, true));
		return indices;
	}

	void RegisterSRV(SRVViewType type, ResourceIdentifier id, unsigned int mip = 0, unsigned int slice = 0) {
		m_resourceDescriptorIndexHelper->RegisterSRV(type, id, mip, slice);
	}
	void RegisterSRV(ResourceIdentifier id, unsigned int mip = 0, unsigned int slice = 0) {
		m_resourceDescriptorIndexHelper->RegisterSRV(id, mip, slice);
	}
	void RegisterUAV(ResourceIdentifier id, unsigned int mip = 0, unsigned int slice = 0) {
		m_resourceDescriptorIndexHelper->RegisterUAV(id, mip, slice);
	}
	void RegisterUAV(UAVViewType type, ResourceIdentifier id, unsigned int mip = 0, unsigned int slice = 0) {
		m_resourceDescriptorIndexHelper->RegisterUAV(type, id, mip, slice);
	}
	void RegisterCBV(ResourceIdentifier id) {
		m_resourceDescriptorIndexHelper->RegisterCBV(id);
	}

	virtual std::shared_ptr<Resource> ProvideResource(ResourceIdentifier const& key) { return nullptr; }
	virtual std::vector<ResourceIdentifier> GetSupportedKeys() { return {}; }

	std::unique_ptr<ResourceDescriptorIndexHelper> m_resourceDescriptorIndexHelper;
	std::shared_ptr<ResourceRegistryView> m_resourceRegistryView;
	friend class ComputePassBuilder;
	friend class RenderGraph;
};


} // namespace org
