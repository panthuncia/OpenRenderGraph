#pragma once

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
#include "interfaces/IResourceProvider.h"
#include "Render/PassInputs.h"
#include "Render/PassExecutionContext.h"
#include "Render/ShaderAPI.h"
#include "Render/QueueKind.h"

struct ComputePassParameters {
	std::vector<ResourceHandleAndRange> shaderResources;
	std::vector<ResourceHandleAndRange> constantBuffers;
	std::vector<ResourceHandleAndRange> unorderedAccessViews;
	std::vector<ResourceHandleAndRange> indirectArgumentBuffers;
	std::vector<ResourceHandleAndRange> legacyInteropResources;
	std::vector<std::pair<ResourceHandleAndRange, ResourceState>> internalTransitions;

	std::unordered_set<ResourceIdentifier, ResourceIdentifier::Hasher> identifierSet;
	std::vector<AutoDescriptorRegistration> autoDescriptorShaderResources;
	std::vector<AutoDescriptorRegistration> autoDescriptorConstantBuffers;
	std::vector<AutoDescriptorRegistration> autoDescriptorUnorderedAccessViews;
	std::vector<ResourceRequirement> staticResourceRequirements; // Static resource requirements for the pass
	std::vector<ResourceRequirement> frameResourceRequirements; // Immediate-mode requirements recorded for this frame
	mutable std::vector<ResourceRequirement> mergedFrameResourceRequirements; // Lazily built static + immediate requirements when a contiguous view is needed
	mutable bool mergedFrameRequirementsDirty = false;
	QueueKind preferredQueueKind = QueueKind::Compute;
	QueueAssignmentPolicy queueAssignmentPolicy = QueueAssignmentPolicy::Automatic;
	std::optional<QueueSlotIndex> pinnedQueueSlot; // Target a specific queue slot instead of using preferredQueueKind
};

class ComputePassBuilder;

class ComputePass : public IResourceProvider, public RenderGraphPassBase {
public:
	virtual ~ComputePass() = default;

	void SetResourceRegistryView(std::shared_ptr<ResourceRegistryView> resourceRegistryView) {
		m_resourceRegistryView = resourceRegistryView;
		m_resourceDescriptorIndexHelper = std::make_unique<ResourceDescriptorIndexHelper>(resourceRegistryView);
	}

	void SetResourceRegistryView(
		std::shared_ptr<ResourceRegistryView> resourceRegistryView,
		const std::vector<AutoDescriptorRegistration>& autoDescriptorShaderResources,
		const std::vector<AutoDescriptorRegistration>& autoDescriptorConstantBuffers,
		const std::vector<AutoDescriptorRegistration>& autoDescriptorUnorderedAccessViews) {
		SetResourceRegistryView(std::move(resourceRegistryView));
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

	virtual void Setup() = 0;

	virtual void Update(const UpdateExecutionContext& context) {};
	virtual void RecordImmediateCommands(ImmediateExecutionContext& context) {};
	virtual PassReturn Execute(PassExecutionContext& context) { return {}; };
	virtual void Cleanup() = 0;

	void Invalidate() { invalidated = true; }
	bool IsInvalidated() const { return invalidated; }

protected:
	bool invalidated = true;
	virtual void DeclareResourceUsages(ComputePassBuilder* builder) {};

	void BindResourceDescriptorIndices(rhi::CommandList& commandList, const PipelineResources& resources) {
		unsigned int indices[rg::shaderapi::kNumResourceDescriptorIndicesRootConstants] = {};
		int i = 0;
		for (auto& binding : resources.mandatoryResourceDescriptorSlots) {
			indices[i] = m_resourceDescriptorIndexHelper->GetResourceDescriptorIndex(binding.hash, false, &binding.name);
			i++;
		}
		for (auto& binding : resources.optionalResourceDescriptorSlots) {
			indices[i] = m_resourceDescriptorIndexHelper->GetResourceDescriptorIndex(binding.hash, true, &binding.name);
			i++;
		}
		if (i > 0) {
			commandList.PushConstants(rhi::ShaderStage::Compute, 0, rg::shaderapi::kResourceDescriptorIndicesRootParameter, 0, i, indices);
		}
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