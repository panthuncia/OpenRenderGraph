#pragma once
#include "Render/PreparedPass.h"

#include <vector>
#include <unordered_set>
#include <rhi.h>

#include "Resources/Resource.h"
#include "Render/ResourceRequirements.h"
#include "RenderPasses/Base/PassReturn.h"
#include "Resources/ResourceStateTracker.h"
#include "Resources/ResourceIdentifier.h"
#include "Render/ResourceRegistry.h"
#include "Render/PipelineState.h"
#include "Interfaces/IResourceProvider.h"
#include "Render/PassInputs.h"
#include "Render/PassExecutionContext.h"
#include "Render/FeatureDomainRegistry.h"
#include "Render/ShaderAPI.h"
#include "Render/QueueKind.h"
#include "RenderPasses/Base/RenderGraphPass.h"


namespace org {

struct PassParameters {
	std::vector<ResourceHandleAndRange> shaderResources;
    std::vector<ResourceHandleAndRange> renderTargets;
	std::vector<ResourceHandleAndRange> renderTargetClearResources;
    std::vector<ResourceHandleAndRange> depthReadResources;
	std::vector<ResourceHandleAndRange> depthReadWriteResources;
	std::vector<ResourceHandleAndRange> depthStencilClearResources;
	std::vector<ResourceHandleAndRange> constantBuffers;
	std::vector<ResourceHandleAndRange> unorderedAccessViews;
	std::vector<ResourceHandleAndRange> unorderedAccessClearViews;
	std::vector<ResourceHandleAndRange> copyTargets;
	std::vector<ResourceHandleAndRange> copySources;
	std::vector<ResourceHandleAndRange> indirectArgumentBuffers;
	std::vector<ResourceHandleAndRange> indexBuffers;
	std::vector<ResourceHandleAndRange> vertexBuffers;
	std::vector<ResourceHandleAndRange> presentResources;
	std::vector<ResourceHandleAndRange> legacyInteropResources;
	std::vector<std::pair<ResourceHandleAndRange, ResourceState>> internalTransitions;
	std::vector<ExternalTimelinePoint> externalWaitsBeforeTransitions;
	std::vector<ExternalTimelineBinding> externalWaitBindingsBeforeTransitions;

	std::unordered_set<ResourceIdentifier, ResourceIdentifier::Hasher> identifierSet;
	std::unordered_set<FeatureDomainIdentifier, FeatureDomainIdentifier::Hasher> activeFeatureDomains;
	std::vector<ResourceRequirement> staticResourceRequirements; // Static resource requirements for the pass
	std::vector<std::shared_ptr<const ResolverRequirementBlock>> resolverRequirementBlocks;
	std::vector<ResourceRequirement> frameResourceRequirements; // Immediate-mode requirements recorded for this frame
	mutable std::vector<ResourceRequirement> mergedFrameResourceRequirements; // Lazily built static + immediate requirements when a contiguous view is needed
	mutable bool mergedFrameRequirementsDirty = false;
	bool isGeometryPass = false;
	QueueKind preferredQueueKind = QueueKind::Graphics;
	QueueAssignmentPolicy queueAssignmentPolicy = QueueAssignmentPolicy::ForcePreferred;
	std::optional<QueueSlotIndex> pinnedQueueSlot; // Target a specific queue slot instead of using preferredQueueKind
	BackendAffinity backendAffinity{};
};
using RenderPassParameters = PassParameters;

class RenderPassBuilder;

class RenderPass : public RenderGraphPass {
public:
    ~RenderPass() override;

	// Declare and Setup may look resources up by name; recording never does.
	void SetResourceRegistryView(std::shared_ptr<ResourceRegistryView> resourceRegistryView) {
		m_resourceRegistryView = std::move(resourceRegistryView);
	}

    void ConfigureResourceRegistryView(
        std::shared_ptr<ResourceRegistryView> view,
        const PassParameters& parameters) override;

	void Setup() override = 0;
    // Preparation owner only. Empty means explicit synchronous legacy fallback.
    PreparedPass PrepareFrame(FramePreparationContext&) override { return {}; }

	void Update(const UpdateExecutionContext& /*context*/) override {};
	PassReturn Execute(PassExecutionContext& /*context*/) override { return {}; };
    void Cleanup() override = 0;

	void Invalidate() override { invalidated = true; }
	bool IsInvalidated() const override { return invalidated; }
	bool SupportsUnifiedDeclaration() const override { return true; }
	void DeclareUnified(RenderPassBuilder& builder) override { DeclareResourceUsages(&builder); }

protected:
	bool invalidated = true;
	virtual void DeclareResourceUsages(RenderPassBuilder* /*builder*/) {};

	std::shared_ptr<Resource> ProvideResource(ResourceIdentifier const& /*key*/) override { return nullptr; }
	std::vector<ResourceIdentifier> GetSupportedKeys() override { return {}; }

	std::shared_ptr<ResourceRegistryView> m_resourceRegistryView;
	friend class RenderPassBuilder;
	friend class RenderGraph;
};


} // namespace org
