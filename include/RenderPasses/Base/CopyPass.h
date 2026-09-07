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
#include "Render/ShaderAPI.h"
#include "Render/QueueKind.h"


namespace org {

using CopyPassParameters = PassParameters;

class CopyPassBuilder;

class CopyPass : public RenderGraphPass {
public:
	virtual ~CopyPass() = default;

	void SetResourceRegistryView(std::shared_ptr<ResourceRegistryView> resourceRegistryView) {
		m_resourceRegistryView = resourceRegistryView;
		m_resourceDescriptorIndexHelper = std::make_unique<ResourceDescriptorIndexHelper>(resourceRegistryView);
	}

    void ConfigureResourceRegistryView(
        std::shared_ptr<ResourceRegistryView> view,
        const PassParameters&) override {
        SetResourceRegistryView(std::move(view));
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
	virtual void DeclareResourceUsages(CopyPassBuilder* builder) {};

	virtual std::shared_ptr<Resource> ProvideResource(ResourceIdentifier const& key) { return nullptr; }
	virtual std::vector<ResourceIdentifier> GetSupportedKeys() { return {}; }

	std::unique_ptr<ResourceDescriptorIndexHelper> m_resourceDescriptorIndexHelper;
	std::shared_ptr<ResourceRegistryView> m_resourceRegistryView;
	friend class CopyPassBuilder;
	friend class RenderGraph;
};


} // namespace org
