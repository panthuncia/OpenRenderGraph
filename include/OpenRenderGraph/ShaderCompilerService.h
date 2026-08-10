#ifndef OPEN_RENDER_GRAPH_SHADER_COMPILER_SERVICE_H
#define OPEN_RENDER_GRAPH_SHADER_COMPILER_SERVICE_H

#include <OpenRenderGraph/ContributorAPI.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ORG_SHADER_COMPILER_SERVICE_ID "org.shader-compiler"
#define ORG_SHADER_COMPILER_SERVICE_VERSION_1 1u

typedef uint64_t ORGShaderHandle;
typedef enum ORGShaderRequestState {
	ORG_SHADER_REQUEST_PENDING = 0,
	ORG_SHADER_REQUEST_READY = 1,
	ORG_SHADER_REQUEST_FAILED = 2
} ORGShaderRequestState;

typedef struct ORGShaderRequest {
	uint32_t structSize, apiVersion;
	const char* sourceName;
	const void* source;
	uint64_t sourceSize;
	const char* entryPoint;
	const char* target;
} ORGShaderRequest;

typedef struct ORGShaderCompilerServiceAPI {
	uint32_t structSize, apiVersion;
	void* context;
	ORGStatus(ORG_RG_CALL* Request)(void*, const ORGShaderRequest*, ORGShaderHandle*);
	ORGStatus(ORG_RG_CALL* GetStatus)(void*, ORGShaderHandle, uint32_t*, const void**, uint64_t*);
	ORGStatus(ORG_RG_CALL* Release)(void*, ORGShaderHandle);
} ORGShaderCompilerServiceAPI;

#ifdef __cplusplus
}
#endif
#endif
