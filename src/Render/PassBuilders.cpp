#include "Render/RenderGraph/RenderGraph.h"
#include "OpenRenderGraph/Runtime.h"
#include "Render/PipelineLibrary.h"


namespace org {

std::shared_ptr<const ResourceUseLayout> GetResourceUses(const RenderPassBuilder& builder) { return builder.ResourceUses(); }

std::vector<ResourceHandleAndRange>
expandToRanges(ResourceIdentifierAndRange const & rir, RenderGraph* graph)
{
    auto resPtr = graph->RequestResourceHandle(rir.identifier);

    // Wrap the resolved resource and rir.range into a ResourceAndRange.
    ResourceHandleAndRange actualRAR(resPtr);
    actualRAR.range    = rir.range;
    return { actualRAR };
}


ResourceRegistry::RegistryHandle BuilderGraphAccess::RequestResourceHandle(RenderGraph* graph, Resource* resource, bool allowFailure) {
    return graph->RequestResourceHandle(resource, allowFailure);
}

std::shared_ptr<IResourceResolver> BuilderGraphAccess::RequestResolver(RenderGraph* graph, ResourceIdentifier const& id, bool allowFailure) {
    return graph->RequestResolver(id, allowFailure);
}

std::shared_ptr<const ResolverDeclarationState> BuilderGraphAccess::CaptureResolverDeclarationState(RenderGraph* graph,
    const IResourceResolver& resolver) {
    return graph->CaptureResolverDeclarationState(resolver);
}

std::shared_ptr<const std::vector<ResourceHandleAndRange>> BuilderGraphAccess::RequestResolverResourceHandles(RenderGraph* graph,
    const ResolverDeclarationState& state) {
    return graph->RequestResolverResourceHandles(state);
}

void BuilderGraphAccess::RegisterResolvedResourceAlias(RenderGraph* graph, ResourceIdentifier const& id, std::shared_ptr<Resource> resource) {
    graph->RegisterResolvedResourceAlias(id, std::move(resource));
}

void BuilderGraphAccess::AddResource(RenderGraph* graph, std::shared_ptr<Resource> resource) {
    graph->AddResource(std::move(resource));
}

Resource* BuilderGraphAccess::Resolve(RenderGraph* graph, ResourceRegistry::RegistryHandle handle) {
    return graph->_registry.Resolve(handle);
}

std::shared_ptr<PipelineLibraryEntry> BuilderGraphAccess::AcquirePipeline(RenderGraph* graph, PipelineDesc desc) {
    return graph->GetRuntime().Pipelines()->Acquire(std::move(desc));
}

PipelineState RenderPassBuilder::Pipeline(PipelineDesc desc) {
    auto entry = BuilderGraphAccess::AcquirePipeline(graph, std::move(desc));
    auto state = entry->State();
    if (std::ranges::find(params.declaredPipelines, entry) == params.declaredPipelines.end())
        params.declaredPipelines.push_back(std::move(entry));
    return state;
}

std::shared_ptr<CommandSignatureLibraryEntry> BuilderGraphAccess::AcquireCommandSignature(RenderGraph* graph, CommandSignatureDesc desc) {
    return graph->GetRuntime().Pipelines()->AcquireCommandSignature(std::move(desc));
}

CommandSignatureState RenderPassBuilder::CommandSignature(CommandSignatureDesc desc) {
    auto entry = BuilderGraphAccess::AcquireCommandSignature(graph, std::move(desc));
    auto state = entry->State();
    if (std::ranges::find(params.declaredCommandSignatures, entry) == params.declaredCommandSignatures.end())
        params.declaredCommandSignatures.push_back(std::move(entry));
    return state;
}

void BuilderGraphAccess::AddRenderPass(RenderGraph* graph, std::shared_ptr<RenderPass> pass, RenderPassParameters& resources, std::string name,
    std::vector<ResolverSnapshot> resolverSnapshots) {
    graph->AddRenderPass(std::move(pass), resources, std::move(name), std::move(resolverSnapshots));
}

} // namespace org
