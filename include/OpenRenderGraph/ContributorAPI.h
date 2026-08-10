#ifndef OPEN_RENDER_GRAPH_CONTRIBUTOR_API_H
#define OPEN_RENDER_GRAPH_CONTRIBUTOR_API_H

#include <stdint.h>

#if defined(_WIN32)
#define ORG_RG_CALL __cdecl
#else
#define ORG_RG_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ORG_RENDER_GRAPH_API_VERSION_1 1u
#define ORG_RENDER_GRAPH_API_CURRENT ORG_RENDER_GRAPH_API_VERSION_1
#define ORG_SHADER_COMPILER_SERVICE_ID "org.shader-compiler"

typedef uint64_t ORGRegistrationHandle;
typedef uint64_t ORGBuildHandle;
typedef uint64_t ORGGenerationHandle;
typedef uint32_t ORGBinding;

typedef enum ORGStatus {
	ORG_RG_OK = 0,
	ORG_RG_E_INVALID_ARGUMENT = 1,
	ORG_RG_E_UNSUPPORTED_VERSION = 2,
	ORG_RG_E_RUNTIME_UNAVAILABLE = 3,
	ORG_RG_E_STALE_HANDLE = 4,
	ORG_RG_E_DUPLICATE_ID = 5,
	ORG_RG_E_RESERVED_ID = 6,
	ORG_RG_E_MISSING_DEPENDENCY = 7,
	ORG_RG_E_CYCLE = 8,
	ORG_RG_E_INCOMPATIBLE_RESOURCE = 9,
	ORG_RG_E_WRONG_THREAD = 10,
	ORG_RG_E_CALLBACK_FAILED = 11,
	ORG_RG_E_CLOSED = 12,
	ORG_RG_E_NOT_READY = 13,
	ORG_RG_E_UNSUPPORTED_CAPABILITY = 14,
	ORG_RG_E_DUPLICATE_BINDING = 15,
	ORG_RG_E_INTERNAL = 0x7fffffff
} ORGStatus;

typedef enum ORGBackend { ORG_RG_BACKEND_NONE, ORG_RG_BACKEND_D3D12, ORG_RG_BACKEND_VULKAN } ORGBackend;
typedef enum ORGContributorKind { ORG_RG_CONTRIBUTOR_REQUIRED, ORG_RG_CONTRIBUTOR_OPTIONAL, ORG_RG_CONTRIBUTOR_DIAGNOSTIC } ORGContributorKind;
typedef enum ORGRegistrationState { ORG_RG_REGISTRATION_ACTIVE, ORG_RG_REGISTRATION_UNREGISTERING, ORG_RG_REGISTRATION_RETIRED } ORGRegistrationState;
typedef enum ORGPassKind { ORG_RG_PASS_RENDER, ORG_RG_PASS_COMPUTE, ORG_RG_PASS_COPY } ORGPassKind;
typedef enum ORGQueueAssignment { ORG_RG_QUEUE_AUTOMATIC, ORG_RG_QUEUE_FORCE_GRAPHICS, ORG_RG_QUEUE_FORCE_COMPUTE, ORG_RG_QUEUE_FORCE_COPY } ORGQueueAssignment;
typedef enum ORGResourceLifetime { ORG_RG_RESOURCE_TRANSIENT, ORG_RG_RESOURCE_PERSISTENT } ORGResourceLifetime;
typedef enum ORGResourceDimension { ORG_RG_RESOURCE_BUFFER, ORG_RG_RESOURCE_TEXTURE_1D, ORG_RG_RESOURCE_TEXTURE_2D, ORG_RG_RESOURCE_TEXTURE_3D } ORGResourceDimension;
typedef enum ORGHeapClass { ORG_RG_HEAP_DEVICE_LOCAL, ORG_RG_HEAP_UPLOAD, ORG_RG_HEAP_READBACK } ORGHeapClass;
typedef enum ORGSizingMode { ORG_RG_SIZE_ABSOLUTE, ORG_RG_SIZE_RENDER_RELATIVE, ORG_RG_SIZE_OUTPUT_RELATIVE } ORGSizingMode;

/* Stable backend-neutral BasicRHI format ordinals. */
typedef enum ORGFormat {
	ORG_RG_FORMAT_UNKNOWN = 0,
	ORG_RG_FORMAT_R32G32B32A32_TYPELESS, ORG_RG_FORMAT_R32G32B32A32_FLOAT, ORG_RG_FORMAT_R32G32B32A32_UINT, ORG_RG_FORMAT_R32G32B32A32_SINT,
	ORG_RG_FORMAT_R32G32B32_TYPELESS, ORG_RG_FORMAT_R32G32B32_FLOAT, ORG_RG_FORMAT_R32G32B32_UINT, ORG_RG_FORMAT_R32G32B32_SINT,
	ORG_RG_FORMAT_R16G16B16A16_TYPELESS, ORG_RG_FORMAT_R16G16B16A16_FLOAT, ORG_RG_FORMAT_R16G16B16A16_UNORM, ORG_RG_FORMAT_R16G16B16A16_UINT, ORG_RG_FORMAT_R16G16B16A16_SNORM, ORG_RG_FORMAT_R16G16B16A16_SINT,
	ORG_RG_FORMAT_R32G32_TYPELESS, ORG_RG_FORMAT_R32G32_FLOAT, ORG_RG_FORMAT_R32G32_UINT, ORG_RG_FORMAT_R32G32_SINT,
	ORG_RG_FORMAT_R10G10B10A2_TYPELESS, ORG_RG_FORMAT_R10G10B10A2_UNORM, ORG_RG_FORMAT_R10G10B10A2_UINT,
	ORG_RG_FORMAT_R11G11B10_FLOAT,
	ORG_RG_FORMAT_R8G8B8A8_TYPELESS, ORG_RG_FORMAT_R8G8B8A8_UNORM, ORG_RG_FORMAT_R8G8B8A8_UNORM_SRGB, ORG_RG_FORMAT_R8G8B8A8_UINT, ORG_RG_FORMAT_R8G8B8A8_SNORM, ORG_RG_FORMAT_R8G8B8A8_SINT,
	ORG_RG_FORMAT_R16G16_TYPELESS, ORG_RG_FORMAT_R16G16_FLOAT, ORG_RG_FORMAT_R16G16_UNORM, ORG_RG_FORMAT_R16G16_UINT, ORG_RG_FORMAT_R16G16_SNORM, ORG_RG_FORMAT_R16G16_SINT,
	ORG_RG_FORMAT_R32_TYPELESS, ORG_RG_FORMAT_D32_FLOAT, ORG_RG_FORMAT_R32_FLOAT, ORG_RG_FORMAT_R32_UINT, ORG_RG_FORMAT_R32_SINT,
	ORG_RG_FORMAT_R8G8_TYPELESS, ORG_RG_FORMAT_R8G8_UNORM, ORG_RG_FORMAT_R8G8_UINT, ORG_RG_FORMAT_R8G8_SNORM, ORG_RG_FORMAT_R8G8_SINT,
	ORG_RG_FORMAT_R16_TYPELESS, ORG_RG_FORMAT_R16_FLOAT, ORG_RG_FORMAT_R16_UNORM, ORG_RG_FORMAT_R16_UINT, ORG_RG_FORMAT_R16_SNORM, ORG_RG_FORMAT_R16_SINT,
	ORG_RG_FORMAT_R8_TYPELESS, ORG_RG_FORMAT_R8_UNORM, ORG_RG_FORMAT_R8_UINT, ORG_RG_FORMAT_R8_SNORM, ORG_RG_FORMAT_R8_SINT,
	ORG_RG_FORMAT_BC1_TYPELESS, ORG_RG_FORMAT_BC1_UNORM, ORG_RG_FORMAT_BC1_UNORM_SRGB,
	ORG_RG_FORMAT_BC2_TYPELESS, ORG_RG_FORMAT_BC2_UNORM, ORG_RG_FORMAT_BC2_UNORM_SRGB,
	ORG_RG_FORMAT_BC3_TYPELESS, ORG_RG_FORMAT_BC3_UNORM, ORG_RG_FORMAT_BC3_UNORM_SRGB,
	ORG_RG_FORMAT_BC4_TYPELESS, ORG_RG_FORMAT_BC4_UNORM, ORG_RG_FORMAT_BC4_SNORM,
	ORG_RG_FORMAT_BC5_TYPELESS, ORG_RG_FORMAT_BC5_UNORM, ORG_RG_FORMAT_BC5_SNORM,
	ORG_RG_FORMAT_B8G8R8A8_TYPELESS, ORG_RG_FORMAT_B8G8R8A8_UNORM, ORG_RG_FORMAT_B8G8R8A8_UNORM_SRGB,
	ORG_RG_FORMAT_BC6H_TYPELESS, ORG_RG_FORMAT_BC6H_UF16, ORG_RG_FORMAT_BC6H_SF16,
	ORG_RG_FORMAT_BC7_TYPELESS, ORG_RG_FORMAT_BC7_UNORM, ORG_RG_FORMAT_BC7_UNORM_SRGB,
	ORG_RG_FORMAT_COUNT
} ORGFormat;

typedef enum ORGResourceUsageBits {
	ORG_RG_USAGE_NONE = 0, ORG_RG_USAGE_SHADER_RESOURCE = 1u << 0, ORG_RG_USAGE_CONSTANT_BUFFER = 1u << 1,
	ORG_RG_USAGE_UNORDERED_ACCESS = 1u << 2, ORG_RG_USAGE_RENDER_TARGET = 1u << 3,
	ORG_RG_USAGE_DEPTH_READ = 1u << 4, ORG_RG_USAGE_DEPTH_WRITE = 1u << 5,
	ORG_RG_USAGE_COPY_SOURCE = 1u << 6, ORG_RG_USAGE_COPY_DESTINATION = 1u << 7,
	ORG_RG_USAGE_INDIRECT_ARGUMENT = 1u << 8, ORG_RG_USAGE_INDEX_BUFFER = 1u << 9,
	ORG_RG_USAGE_LEGACY_INTEROP = 1u << 10
} ORGResourceUsageBits;

typedef enum ORGAccessKind {
	ORG_RG_ACCESS_NONE, ORG_RG_ACCESS_SHADER_RESOURCE, ORG_RG_ACCESS_CONSTANT_BUFFER,
	ORG_RG_ACCESS_UNORDERED_ACCESS, ORG_RG_ACCESS_UNORDERED_ACCESS_CLEAR,
	ORG_RG_ACCESS_RENDER_TARGET, ORG_RG_ACCESS_RENDER_TARGET_CLEAR, ORG_RG_ACCESS_DEPTH_READ,
	ORG_RG_ACCESS_DEPTH_READ_WRITE, ORG_RG_ACCESS_DEPTH_STENCIL_CLEAR, ORG_RG_ACCESS_COPY_SOURCE,
	ORG_RG_ACCESS_COPY_DESTINATION, ORG_RG_ACCESS_INDIRECT_ARGUMENT, ORG_RG_ACCESS_INDEX_BUFFER,
	ORG_RG_ACCESS_LEGACY_INTEROP
} ORGAccessKind;
typedef enum ORGViewKind { ORG_RG_VIEW_NONE, ORG_RG_VIEW_SHADER_RESOURCE, ORG_RG_VIEW_CONSTANT_BUFFER, ORG_RG_VIEW_UNORDERED_ACCESS, ORG_RG_VIEW_RENDER_TARGET, ORG_RG_VIEW_DEPTH_STENCIL } ORGViewKind;
typedef enum ORGViewDimension {
	ORG_RG_VIEW_DIMENSION_DEFAULT, ORG_RG_VIEW_DIMENSION_BUFFER,
	ORG_RG_VIEW_DIMENSION_TEXTURE_1D, ORG_RG_VIEW_DIMENSION_TEXTURE_1D_ARRAY,
	ORG_RG_VIEW_DIMENSION_TEXTURE_2D, ORG_RG_VIEW_DIMENSION_TEXTURE_2D_ARRAY,
	ORG_RG_VIEW_DIMENSION_TEXTURE_2D_MS, ORG_RG_VIEW_DIMENSION_TEXTURE_2D_MS_ARRAY,
	ORG_RG_VIEW_DIMENSION_TEXTURE_3D, ORG_RG_VIEW_DIMENSION_TEXTURE_CUBE,
	ORG_RG_VIEW_DIMENSION_TEXTURE_CUBE_ARRAY
} ORGViewDimension;
typedef enum ORGViewFlags { ORG_RG_VIEW_FLAG_NONE = 0, ORG_RG_VIEW_FLAG_RAW_BUFFER = 1u << 0, ORG_RG_VIEW_FLAG_READ_ONLY_DEPTH = 1u << 1, ORG_RG_VIEW_FLAG_READ_ONLY_STENCIL = 1u << 2 } ORGViewFlags;
typedef enum ORGPassFlags { ORG_RG_PASS_NONE = 0, ORG_RG_PASS_PARALLEL_RECORDING_SAFE = 1u << 0, ORG_RG_PASS_DISABLE_STATISTICS = 1u << 1, ORG_RG_PASS_GEOMETRY = 1u << 2 } ORGPassFlags;
typedef enum ORGCapabilityBits {
	ORG_RG_CAP_RENDER_PASSES = 1ull << 0, ORG_RG_CAP_COMPUTE_PASSES = 1ull << 1,
	ORG_RG_CAP_COPY_PASSES = 1ull << 2, ORG_RG_CAP_ASYNC_COMPUTE = 1ull << 3,
	ORG_RG_CAP_COPY_QUEUE = 1ull << 4, ORG_RG_CAP_MANAGED_RESOURCES = 1ull << 5,
	ORG_RG_CAP_RESOURCE_ALIASING = 1ull << 6, ORG_RG_CAP_SCHEDULED_UPLOADS = 1ull << 7,
	ORG_RG_CAP_SERVICE_QUERY = 1ull << 8
} ORGCapabilityBits;

typedef struct ORGSubresourceRange { uint32_t firstMip, mipCount, firstArraySlice, arraySize; } ORGSubresourceRange;
typedef struct ORGResourceDesc {
	uint32_t structSize, apiVersion; const char* id; ORGResourceLifetime lifetime;
	ORGResourceDimension dimension; ORGHeapClass heapClass; ORGSizingMode sizing; ORGFormat format;
	uint64_t byteSize; uint32_t structureByteStride, width, height, depthOrArraySize, mipLevels, sampleCount;
	float widthScale, heightScale; uint32_t allowedUsages, allowAlias; uint64_t aliasingPool;
} ORGResourceDesc;
typedef struct ORGResourceAccessDesc {
	uint32_t structSize, apiVersion; const char* resourceId; ORGBinding binding; ORGAccessKind access;
	ORGSubresourceRange range; ORGViewKind viewKind; ORGViewDimension viewDimension; ORGFormat viewFormat;
	uint32_t viewFlags; uint64_t firstElement; uint32_t elementCount, structureByteStride; ORGBinding counterBinding;
} ORGResourceAccessDesc;
typedef struct ORGBindingInfo {
	uint32_t structSize, apiVersion; ORGBinding binding; ORGAccessKind access; ORGViewKind viewKind;
	ORGResourceDimension dimension; ORGFormat resourceFormat, viewFormat; uint32_t width, height,
	depthOrArraySize, mipLevels; uint64_t byteSize; uint32_t descriptorIndex, shaderVisible;
} ORGBindingInfo;
typedef struct ORGFrameInfo {
	uint32_t structSize, apiVersion; uint64_t frameIndex; ORGGenerationHandle generation;
	uint64_t completionValue; uint32_t frameSlot, framesInFlight, renderWidth, renderHeight, outputWidth, outputHeight;
} ORGFrameInfo;
typedef struct ORGExecutionContext { uint32_t structSize, apiVersion; ORGBackend backend; uint32_t reserved; const ORGFrameInfo* frame; const void* hostContext; } ORGExecutionContext;

typedef ORGStatus(ORG_RG_CALL* ORGExecuteCallback)(void*, const ORGExecutionContext*);
typedef ORGStatus(ORG_RG_CALL* ORGPrepareCallback)(void*, const ORGExecutionContext*);
typedef ORGStatus(ORG_RG_CALL* ORGUpdateCallback)(void*, const ORGExecutionContext*);
typedef void(ORG_RG_CALL* ORGCleanupCallback)(void*, ORGGenerationHandle);
typedef struct ORGPassDesc {
	uint32_t structSize, apiVersion; const char* id; ORGPassKind kind; ORGQueueAssignment queue;
	uint32_t flags; int32_t priority; const char* techniquePath; const char* const* featureDomains;
	uint32_t featureDomainCount; const char* const* after; uint32_t afterCount; const char* const* before;
	uint32_t beforeCount; const ORGResourceAccessDesc* accesses; uint32_t accessCount;
	ORGPrepareCallback prepare; ORGUpdateCallback update; ORGExecuteCallback execute; ORGCleanupCallback cleanup;
} ORGPassDesc;

typedef ORGStatus(ORG_RG_CALL* ORGBuildCallback)(void*, ORGBuildHandle);
typedef void(ORG_RG_CALL* ORGGenerationCallback)(void*, ORGGenerationHandle);
typedef void(ORG_RG_CALL* ORGDeviceLostCallback)(void*, uint32_t);
typedef void(ORG_RG_CALL* ORGShutdownCallback)(void*);
typedef struct ORGContributorDesc {
	uint32_t structSize, apiVersion; const char* id; ORGContributorKind kind; void* userData;
	ORGBuildCallback build; ORGGenerationCallback generationActivated, generationRetired;
	ORGDeviceLostCallback deviceLost; ORGShutdownCallback shutdown;
} ORGContributorDesc;

typedef struct ORGAnchorDescriptor { uint32_t structSize, apiVersion; const char* id; const char* displayName; } ORGAnchorDescriptor;
typedef struct ORGHostDescriptor {
	uint32_t structSize, apiVersion; const char* id; const char* displayName; const char* version;
	const ORGAnchorDescriptor* anchors; uint32_t anchorCount;
} ORGHostDescriptor;
typedef struct ORGHostInfo { uint32_t structSize, apiVersion; const char* id; const char* displayName; const char* version; } ORGHostInfo;
typedef struct ORGRuntimeInfo {
	uint32_t structSize, apiVersion, available; ORGBackend backend; uint64_t capabilities;
	ORGGenerationHandle activeGeneration; uint64_t lastSubmittedCompletion, completedFence;
	uint32_t framesInFlight, reserved;
} ORGRuntimeInfo;
typedef struct ORGDiagnostic {
	uint32_t structSize, apiVersion; ORGStatus status; uint32_t phase; uint64_t sequence;
	ORGGenerationHandle generation; ORGRegistrationHandle contributor; char message[512];
} ORGDiagnostic;
typedef struct ORGBufferUpload {
	uint32_t structSize, apiVersion; ORGBinding destination; uint64_t destinationOffset;
	const void* data; uint64_t dataSize;
} ORGBufferUpload;

typedef struct ORGRenderGraphAPI {
	uint32_t structSize, apiVersion;
	ORGStatus(ORG_RG_CALL* GetRuntimeInfo)(ORGRuntimeInfo*);
	ORGStatus(ORG_RG_CALL* GetHostInfo)(ORGHostInfo*);
	ORGStatus(ORG_RG_CALL* GetAnchorCount)(uint32_t*);
	ORGStatus(ORG_RG_CALL* GetAnchor)(uint32_t, ORGAnchorDescriptor*);
	ORGStatus(ORG_RG_CALL* RegisterContributor)(const ORGContributorDesc*, ORGRegistrationHandle*);
	ORGStatus(ORG_RG_CALL* BeginUnregister)(ORGRegistrationHandle);
	ORGStatus(ORG_RG_CALL* GetRegistrationState)(ORGRegistrationHandle, ORGRegistrationState*);
	ORGStatus(ORG_RG_CALL* DeclareResource)(ORGBuildHandle, const ORGResourceDesc*);
	ORGStatus(ORG_RG_CALL* DeclarePass)(ORGBuildHandle, const ORGPassDesc*);
	ORGStatus(ORG_RG_CALL* RequestRebuild)(ORGRegistrationHandle);
	ORGStatus(ORG_RG_CALL* GetDiagnostic)(ORGRegistrationHandle, ORGDiagnostic*);
	ORGStatus(ORG_RG_CALL* GetBindingInfo)(const ORGExecutionContext*, ORGBinding, ORGBindingInfo*);
	ORGStatus(ORG_RG_CALL* QueueBufferUpload)(const ORGExecutionContext*, const ORGBufferUpload*);
	ORGStatus(ORG_RG_CALL* QueryService)(const char*, uint32_t, void*, uint32_t);
	const char*(ORG_RG_CALL* StatusString)(ORGStatus);
} ORGRenderGraphAPI;

/* Implemented by the hosting DLL and normally discovered with GetProcAddress/dlsym. */
#if defined(_WIN32) && defined(ORG_RENDER_GRAPH_HOST_EXPORTS)
#define ORG_RG_HOST_API __declspec(dllexport)
#elif defined(_WIN32)
#define ORG_RG_HOST_API __declspec(dllimport)
#else
#define ORG_RG_HOST_API
#endif
ORGStatus ORG_RG_HOST_API ORG_RG_CALL ORG_GetRenderGraphAPI(uint32_t version, ORGRenderGraphAPI* out);

#ifdef __cplusplus
}
#endif
#undef ORG_RG_HOST_API
#endif
