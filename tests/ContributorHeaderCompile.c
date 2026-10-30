#include <OpenRenderGraph/ContributorAPI.h>
#include <OpenRenderGraph/ShaderCompilerService.h>
#include <stddef.h>

_Static_assert(sizeof(ORGRegistrationHandle) == 8, "registration handles must be 64-bit");
_Static_assert(sizeof(ORGBinding) == 4, "bindings must be 32-bit");
_Static_assert(offsetof(ORGResourceDesc, structSize) == 0, "resource ABI header must be first");
_Static_assert(offsetof(ORGPassDesc, structSize) == 0, "pass ABI header must be first");
_Static_assert(offsetof(ORGContributorDesc, structSize) == 0, "contributor ABI header must be first");
_Static_assert(offsetof(ORGPreparedPass, structSize) == 0, "packet ABI header must be first");
_Static_assert(offsetof(ORGPrepareContext, structSize) == 0, "prepare context ABI header must be first");
_Static_assert(offsetof(ORGRecordContext, structSize) == 0, "record context ABI header must be first");
_Static_assert(offsetof(ORGExportDescriptor, structSize) == 0, "export ABI header must be first");
_Static_assert(offsetof(ORGEpochDescriptor, structSize) == 0, "epoch ABI header must be first");
_Static_assert(offsetof(ORGPipelineService, structSize) == 0, "pipeline service ABI header must be first");

int main(void)
{
	ORGRenderGraphAPI graph = { sizeof(graph), ORG_RENDER_GRAPH_API_CURRENT };
	ORGShaderCompilerServiceAPI shaders = { sizeof(shaders), ORG_SHADER_COMPILER_SERVICE_VERSION_1 };
	ORGPipelineService pipelines = { sizeof(pipelines), ORG_PIPELINE_SERVICE_VERSION_1 };
	return graph.apiVersion == 2 && shaders.apiVersion == 1 && pipelines.apiVersion == 1 ? 0 : 1;
}
