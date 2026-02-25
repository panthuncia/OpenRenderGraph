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

struct CopyPassParameters {
	std::vector<ResourceHandleAndRange> copyTargets;
	std::vector<ResourceHandleAndRange> copySources;
	std::vector<std::pair<ResourceHandleAndRange, ResourceState>> internalTransitions;

	std::unordered_set<ResourceIdentifier, ResourceIdentifier::Hasher> identifierSet;
	std::vector<ResourceRequirement> staticResourceRequirements;
	std::vector<ResourceRequirement> frameResourceRequirements;
	CopyQueueSelection queueSelection = CopyQueueSelection::Copy;
};

class CopyPassBuilder;

class CopyPass : public IResourceProvider, public RenderGraphPassBase {
public:
	virtual ~CopyPass() = default;

	void SetResourceRegistryView(std::shared_ptr<ResourceRegistryView> resourceRegistryView) {
		m_resourceRegistryView = resourceRegistryView;
		m_resourceDescriptorIndexHelper = std::make_unique<ResourceDescriptorIndexHelper>(resourceRegistryView);
	}

	virtual void Setup() = 0;

	virtual void Update(const UpdateExecutionContext& context) {};
	virtual void ExecuteImmediate(ImmediateExecutionContext& context) {};
	virtual PassReturn Execute(PassExecutionContext& context) { return {}; };
	virtual void Cleanup() = 0;

	void Invalidate() { invalidated = true; }
	bool IsInvalidated() const { return invalidated; }

protected:
	bool invalidated = true;
	virtual void DeclareResourceUsages(CopyPassBuilder* builder) {};

	virtual std::shared_ptr<Resource> ProvideResource(ResourceIdentifier const& key) { return nullptr; }
	virtual std::vector<ResourceIdentifier> GetSupportedKeys() { return {}; }

	std::unique_ptr<ResourceDescriptorIndexHelper> m_resourceDescriptorIndexHelper;
	std::shared_ptr<ResourceRegistryView> m_resourceRegistryView;
	friend class CopyPassBuilder;
	friend class RenderGraph;
};
