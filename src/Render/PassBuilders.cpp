#include "Render/PassBuilders.h"


namespace org {

std::shared_ptr<const ResourceUseLayout> GetResourceUses(const RenderPassBuilder& builder) { return builder.ResourceUses(); }

std::vector<ResourceHandleAndRange>
expandToRanges(ResourceIdentifierAndRange const & rir, RenderGraph* graph)
{
    auto resPtr = graph->RequestResourceHandle(rir.identifier);

    // Now wrap that actual resource + rir.range into a ResourceAndRange:
    ResourceHandleAndRange actualRAR(resPtr);
    actualRAR.range    = rir.range;
    return { actualRAR };
}


} // namespace org
