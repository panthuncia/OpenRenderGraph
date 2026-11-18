#ifndef OPEN_RENDER_GRAPH_CONTRIBUTOR_API_H
#define OPEN_RENDER_GRAPH_CONTRIBUTOR_API_H

/* The contributor ABI. A module outside the host (a DLL) uses it to add resources and passes to the host's render graph.
 *
 * A contributor registers once (RegisterContributor). Whenever the host builds its graph, it calls the contributor's
 * build callback, which declares resources and passes with DeclareResource and DeclarePass. Each build makes a new
 * generation of the declarations. The host activates a generation when its graph is installed. It retires the
 * generation once no frame can still use it.
 *
 * Every frame, each declared pass is prepared on the host's preparation thread. `prepare` returns an owned packet
 * (ORGPreparedPass). It builds the packet from the frame, the pass's bindings (resolved views and resources) and the
 * host's per-frame snapshot. Then the packet goes through these steps:
 * 1. It is recorded later on a recording thread, possibly after the next frame was prepared. Recording goes through a
 *    borrowed BasicRHI recorder that is valid only for that call.
 * 2. It is submitted.
 * 3. It is completed or abandoned.
 * 4. It is destroyed.
 *
 * Every structure starts with structSize and apiVersion. The host rejects a structure that is smaller than this
 * header's version of it, or that has another version. Identifiers are namespaced:
 * - a contributor's id has at least two segments ("sarp.producer"),
 * - its resources and passes are named in its namespace, which is the id's first segment ("sarp.surface.motion"),
 * - "org." is reserved. */

#include <stddef.h>
#include <stdint.h>

#include <OpenRenderGraph/ORGTypes.h>
#include <rhi_c.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ORG_RENDER_GRAPH_API_VERSION_2 2u
#define ORG_RENDER_GRAPH_API_CURRENT ORG_RENDER_GRAPH_API_VERSION_2
#define ORG_SHADER_COMPILER_SERVICE_ID "org.shader-compiler"
/* The host's device. Query it with QueryService(ORG_PIPELINE_SERVICE_ID, ORG_PIPELINE_SERVICE_VERSION_1,
 * ORGPipelineService*, sizeof). */
#define ORG_PIPELINE_SERVICE_ID "org.pipelines"
#define ORG_PIPELINE_SERVICE_VERSION_1 1u

typedef uint64_t ORGRegistrationHandle;
typedef uint64_t ORGBuildHandle;
typedef uint64_t ORGGenerationHandle;
/* A pass's name for one of its accesses (ORGResourceAccessDesc::binding), unique within the pass. */
typedef uint32_t ORGBinding;

typedef enum ORGContributorKind { ORG_RG_CONTRIBUTOR_REQUIRED, ORG_RG_CONTRIBUTOR_OPTIONAL, ORG_RG_CONTRIBUTOR_DIAGNOSTIC } ORGContributorKind;
typedef enum ORGRegistrationState { ORG_RG_REGISTRATION_ACTIVE, ORG_RG_REGISTRATION_UNREGISTERING, ORG_RG_REGISTRATION_RETIRED } ORGRegistrationState;
typedef enum ORGPassKind { ORG_RG_PASS_RENDER, ORG_RG_PASS_COMPUTE, ORG_RG_PASS_COPY } ORGPassKind;
typedef enum ORGQueueAssignment { ORG_RG_QUEUE_AUTOMATIC, ORG_RG_QUEUE_FORCE_GRAPHICS, ORG_RG_QUEUE_FORCE_COMPUTE, ORG_RG_QUEUE_FORCE_COPY } ORGQueueAssignment;
typedef enum ORGResourceLifetime { ORG_RG_RESOURCE_TRANSIENT, ORG_RG_RESOURCE_PERSISTENT } ORGResourceLifetime;
typedef enum ORGResourceDimension { ORG_RG_RESOURCE_BUFFER, ORG_RG_RESOURCE_TEXTURE_1D, ORG_RG_RESOURCE_TEXTURE_2D, ORG_RG_RESOURCE_TEXTURE_3D } ORGResourceDimension;
typedef enum ORGHeapClass { ORG_RG_HEAP_DEVICE_LOCAL, ORG_RG_HEAP_UPLOAD, ORG_RG_HEAP_READBACK } ORGHeapClass;
typedef enum ORGSizingMode { ORG_RG_SIZE_ABSOLUTE, ORG_RG_SIZE_RENDER_RELATIVE, ORG_RG_SIZE_OUTPUT_RELATIVE } ORGSizingMode;

typedef enum ORGResourceUsageBits {
	ORG_RG_USAGE_NONE = 0, ORG_RG_USAGE_SHADER_RESOURCE = 1u << 0, ORG_RG_USAGE_CONSTANT_BUFFER = 1u << 1,
	ORG_RG_USAGE_UNORDERED_ACCESS = 1u << 2, ORG_RG_USAGE_RENDER_TARGET = 1u << 3,
	ORG_RG_USAGE_DEPTH_READ = 1u << 4, ORG_RG_USAGE_DEPTH_WRITE = 1u << 5,
	ORG_RG_USAGE_COPY_SOURCE = 1u << 6, ORG_RG_USAGE_COPY_DESTINATION = 1u << 7,
	ORG_RG_USAGE_INDIRECT_ARGUMENT = 1u << 8, ORG_RG_USAGE_INDEX_BUFFER = 1u << 9
} ORGResourceUsageBits;

typedef enum ORGAccessKind {
	ORG_RG_ACCESS_NONE, ORG_RG_ACCESS_SHADER_RESOURCE, ORG_RG_ACCESS_CONSTANT_BUFFER,
	ORG_RG_ACCESS_UNORDERED_ACCESS, ORG_RG_ACCESS_UNORDERED_ACCESS_CLEAR,
	ORG_RG_ACCESS_RENDER_TARGET, ORG_RG_ACCESS_RENDER_TARGET_CLEAR, ORG_RG_ACCESS_DEPTH_READ,
	ORG_RG_ACCESS_DEPTH_READ_WRITE, ORG_RG_ACCESS_DEPTH_STENCIL_CLEAR, ORG_RG_ACCESS_COPY_SOURCE,
	ORG_RG_ACCESS_COPY_DESTINATION, ORG_RG_ACCESS_INDIRECT_ARGUMENT, ORG_RG_ACCESS_INDEX_BUFFER
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
typedef enum ORGPassFlags { ORG_RG_PASS_NONE = 0, ORG_RG_PASS_DISABLE_STATISTICS = 1u << 1, ORG_RG_PASS_GEOMETRY = 1u << 2 } ORGPassFlags;
typedef enum ORGExportFlags { ORG_RG_EXPORT_READABLE = 1u << 0, ORG_RG_EXPORT_WRITABLE = 1u << 1 } ORGExportFlags;
typedef enum ORGAbandonReason {
	ORG_RG_ABANDON_SHUTDOWN, ORG_RG_ABANDON_GENERATION_INVALIDATED, ORG_RG_ABANDON_PREPARATION_FAILED, ORG_RG_ABANDON_ADMISSION_FAILED
} ORGAbandonReason;
typedef enum ORGCapabilityBits {
	ORG_RG_CAP_RENDER_PASSES = 1ull << 0, ORG_RG_CAP_COMPUTE_PASSES = 1ull << 1,
	ORG_RG_CAP_COPY_PASSES = 1ull << 2, ORG_RG_CAP_ASYNC_COMPUTE = 1ull << 3,
	ORG_RG_CAP_COPY_QUEUE = 1ull << 4, ORG_RG_CAP_MANAGED_RESOURCES = 1ull << 5,
	ORG_RG_CAP_RESOURCE_ALIASING = 1ull << 6, ORG_RG_CAP_SCHEDULED_UPLOADS = 1ull << 7,
	ORG_RG_CAP_SERVICE_QUERY = 1ull << 8, ORG_RG_CAP_PIPELINES = 1ull << 9
} ORGCapabilityBits;

/* A mipCount or arraySize of UINT32_MAX covers the rest of the resource. */
typedef struct ORGSubresourceRange { uint32_t firstMip, mipCount, firstArraySlice, arraySize; } ORGSubresourceRange;

/* A resource the contributor owns.
 * - A transient resource may share memory with other resources whose lifetimes do not overlap (allowAlias).
 * - A persistent resource keeps its contents across frames and builds while its description is unchanged.
 * A render- or output-relative texture is widthScale x heightScale of the reference extent. A render- or
 * output-relative buffer holds byteSize bytes per pixel of that extent. For a structured buffer, this must be a whole
 * number of structures. */
typedef struct ORGResourceDesc {
	uint32_t structSize, apiVersion; const char* id; ORGResourceLifetime lifetime;
	ORGResourceDimension dimension; ORGHeapClass heapClass; ORGSizingMode sizing; ORGFormat format;
	uint64_t byteSize; uint32_t structureByteStride, width, height, depthOrArraySize, mipLevels, sampleCount;
	float widthScale, heightScale; uint32_t allowedUsages, allowAlias; uint64_t aliasingPool;
} ORGResourceDesc;

/* One access of a pass. resourceId names a contributor resource or a host export.
 * A buffer's views cover the whole buffer (firstElement 0, elementCount UINT32_MAX):
 * - a buffer without a structure stride gets raw views (ORG_RG_VIEW_FLAG_RAW_BUFFER),
 * - a buffer with a structure stride gets structured views,
 * - constant buffer views also cover the whole buffer.
 * The host clears the *_CLEAR accesses to clearValue before the pass records:
 * - render targets and unordered access take RGBA, as integers for integer formats and buffers,
 * - depth takes depth in [0] and stencil in [1]. */
typedef struct ORGResourceAccessDesc {
	uint32_t structSize, apiVersion; const char* resourceId; ORGBinding binding; ORGAccessKind access;
	ORGSubresourceRange range; ORGViewKind viewKind; ORGViewDimension viewDimension; ORGFormat viewFormat;
	uint32_t viewFlags; uint64_t firstElement; uint32_t elementCount, structureByteStride; ORGBinding counterBinding;
	float clearValue[4];
} ORGResourceAccessDesc;

/* An access as it resolved for one frame.
 * - descriptorIndex is the bindless index of the access's view in the host's resource heap. It is set for shader
 *   resource, constant buffer and unordered access views, and is UINT32_MAX otherwise.
 * - resource is the resource itself, for copies, indirect arguments and index buffers.
 * The host binds render-target and depth accesses itself. A render pass's record callback runs inside a render pass
 * over them. */
typedef struct ORGBindingInfo {
	uint32_t structSize, apiVersion; ORGBinding binding; ORGAccessKind access; ORGViewKind viewKind;
	ORGResourceDimension dimension; ORGFormat resourceFormat, viewFormat; uint32_t width, height,
	depthOrArraySize, mipLevels; uint64_t byteSize; uint32_t descriptorIndex, shaderVisible;
	basicrhi_c_resource resource;
} ORGBindingInfo;

/* The frame being prepared. epoch is the host epoch it executes in, as the index GetEpoch takes. It is UINT32_MAX for
 * a whole frame. */
typedef struct ORGFrameInfo {
	uint32_t structSize, apiVersion; uint64_t frameIndex; ORGGenerationHandle generation;
	uint32_t frameSlot, framesInFlight, renderWidth, renderHeight, outputWidth, outputHeight, epoch, reserved;
} ORGFrameInfo;

/* The host's per-frame data, such as view matrices and timing. The host defines its structure and names it by typeId.
 * Valid during prepare. */
typedef struct ORGHostSnapshot { const char* typeId; const void* data; uint64_t size; } ORGHostSnapshot;

typedef struct ORGPrepareContext {
	uint32_t structSize, apiVersion; ORGBackend backend; uint32_t bindingCount;
	const ORGFrameInfo* frame; const ORGBindingInfo* bindings; ORGHostSnapshot snapshot;
	/* The host's resource and sampler descriptor heaps, for set_descriptor_heaps. */
	basicrhi_c_descriptor_heap resourceHeap, samplerHeap;
} ORGPrepareContext;

/* The recording of one prepared packet. The recorder is valid only during the record callback. */
typedef struct ORGRecordContext {
	uint32_t structSize, apiVersion; uint32_t bindingCount, reserved;
	const ORGFrameInfo* frame; const ORGBindingInfo* bindings; basicrhi_c_borrowed_recorder_v1 recorder;
} ORGRecordContext;

/* An owned packet for one frame. The contributor owns `data` until destroy runs. record is required.
 * submitted, completed and abandoned are optional. A pass with invocationRevision must not set them.
 * Callbacks may run on any thread. */
typedef struct ORGPreparedPass {
	uint32_t structSize, apiVersion; void* data;
	void(ORG_RG_CALL* record)(const void* data, const ORGRecordContext* context);
	void(ORG_RG_CALL* submitted)(const void* data, uint64_t submission);
	void(ORG_RG_CALL* completed)(const void* data, uint64_t submission);
	void(ORG_RG_CALL* abandoned)(const void* data, ORGAbandonReason reason);
	void(ORG_RG_CALL* destroy)(void* data);
} ORGPreparedPass;

typedef ORGStatus(ORG_RG_CALL* ORGPrepareCallback)(void* userData, const ORGPrepareContext* context, ORGPreparedPass* out);
/* A hash of everything prepare reads except the bindings, such as settings, pipelines and contributor state.
 * With it, the host may:
 * - reuse the previous packet while the revision and the bindings are unchanged,
 * - check a packet prepared ahead against the pass's current state.
 * It must not depend on the frame. */
typedef ORGStatus(ORG_RG_CALL* ORGRevisionCallback)(void* userData, uint64_t* revision);

/* after and before list host anchors (GetAnchor) or the ids of other passes.
 * epoch is the id of the host epoch the pass runs in (GetEpoch). NULL means every epoch. */
typedef struct ORGPassDesc {
	uint32_t structSize, apiVersion; const char* id; ORGPassKind kind; ORGQueueAssignment queue;
	uint32_t flags; int32_t priority; const char* techniquePath; const char* const* featureDomains;
	uint32_t featureDomainCount; const char* const* after; uint32_t afterCount; const char* const* before;
	uint32_t beforeCount; const ORGResourceAccessDesc* accesses; uint32_t accessCount; const char* epoch;
	ORGPrepareCallback prepare; ORGRevisionCallback invocationRevision;
} ORGPassDesc;

typedef ORGStatus(ORG_RG_CALL* ORGBuildCallback)(void* userData, ORGBuildHandle build);
typedef void(ORG_RG_CALL* ORGGenerationCallback)(void* userData, ORGGenerationHandle generation);
typedef void(ORG_RG_CALL* ORGDeviceLostCallback)(void* userData, uint32_t reason);
typedef void(ORG_RG_CALL* ORGShutdownCallback)(void* userData);
/* generationRetired may run on any thread. By then, all of the generation's packets are destroyed. */
typedef struct ORGContributorDesc {
	uint32_t structSize, apiVersion; const char* id; ORGContributorKind kind; void* userData;
	ORGBuildCallback build; ORGGenerationCallback generationActivated, generationRetired;
	ORGDeviceLostCallback deviceLost; ORGShutdownCallback shutdown;
} ORGContributorDesc;

/* A point in the host's frame a pass can be ordered against. */
typedef struct ORGAnchorDescriptor { uint32_t structSize, apiVersion; const char* id; const char* displayName; } ORGAnchorDescriptor;
/* One of the points of its frame at which the host executes the graph. */
typedef struct ORGEpochDescriptor { uint32_t structSize, apiVersion; const char* id; const char* displayName; } ORGEpochDescriptor;
/* A host resource contributors may access: its shape, the usages it allows, and whether it may be read or written. */
typedef struct ORGExportDescriptor {
	uint32_t structSize, apiVersion; const char* id; const char* displayName; ORGResourceDimension dimension; ORGFormat format;
	uint32_t width, height, depthOrArraySize, mipLevels; uint64_t byteSize; uint32_t structureByteStride, allowedUsages, flags, reserved;
} ORGExportDescriptor;
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
/* Written into the binding's resource ahead of the pass, in the frame being prepared. */
typedef struct ORGBufferUpload {
	uint32_t structSize, apiVersion; ORGBinding destination; uint64_t destinationOffset;
	const void* data; uint64_t dataSize;
} ORGBufferUpload;
/* Flags for basicrhi_c_pipeline_desc_v1::raster_flags, used through the pipeline service. Without depth flags, depth is
 * off. Without culling flags, both faces draw. */
typedef enum ORGRasterFlags {
	ORG_RG_RASTER_DEPTH_TEST = 1u << 0, ORG_RG_RASTER_DEPTH_WRITE = 1u << 1, ORG_RG_RASTER_DEPTH_GREATER = 1u << 2,
	ORG_RG_RASTER_CULL_BACK = 1u << 3, ORG_RG_RASTER_CULL_FRONT = 1u << 4, ORG_RG_RASTER_ALPHA_BLEND = 1u << 5,
	ORG_RG_RASTER_FRONT_CCW = 1u << 6
} ORGRasterFlags;
/* The host's device and its bindless pipeline layouts.
 * `device` supports only create_pipeline and create_command_signature. Resources come from DeclareResource instead.
 * The other entries return BASICRHI_C_UNSUPPORTED.
 * primitive_topology is an ordinal of BasicRHI's PrimitiveTopology. Command-signature argument kinds are ordinals of
 * its IndirectArgKind. */
typedef struct ORGPipelineService {
	uint32_t structSize, apiVersion; ORGBackend backend; uint32_t reserved;
	const basicrhi_c_device_api_v1* device; basicrhi_c_pipeline_layout graphicsLayout, computeLayout;
} ORGPipelineService;

typedef struct ORGRenderGraphAPI {
	uint32_t structSize, apiVersion;
	ORGStatus(ORG_RG_CALL* GetRuntimeInfo)(ORGRuntimeInfo*);
	ORGStatus(ORG_RG_CALL* GetHostInfo)(ORGHostInfo*);
	ORGStatus(ORG_RG_CALL* GetAnchorCount)(uint32_t*);
	ORGStatus(ORG_RG_CALL* GetAnchor)(uint32_t, ORGAnchorDescriptor*);
	ORGStatus(ORG_RG_CALL* GetEpochCount)(uint32_t*);
	ORGStatus(ORG_RG_CALL* GetEpoch)(uint32_t, ORGEpochDescriptor*);
	ORGStatus(ORG_RG_CALL* GetExportCount)(uint32_t*);
	ORGStatus(ORG_RG_CALL* GetExport)(uint32_t, ORGExportDescriptor*);
	ORGStatus(ORG_RG_CALL* RegisterContributor)(const ORGContributorDesc*, ORGRegistrationHandle*);
	ORGStatus(ORG_RG_CALL* BeginUnregister)(ORGRegistrationHandle);
	ORGStatus(ORG_RG_CALL* GetRegistrationState)(ORGRegistrationHandle, ORGRegistrationState*);
	ORGStatus(ORG_RG_CALL* DeclareResource)(ORGBuildHandle, const ORGResourceDesc*);
	ORGStatus(ORG_RG_CALL* DeclarePass)(ORGBuildHandle, const ORGPassDesc*);
	ORGStatus(ORG_RG_CALL* RequestRebuild)(ORGRegistrationHandle);
	ORGStatus(ORG_RG_CALL* GetDiagnostic)(ORGRegistrationHandle, ORGDiagnostic*);
	/* During prepare only. */
	ORGStatus(ORG_RG_CALL* QueueBufferUpload)(const ORGPrepareContext*, const ORGBufferUpload*);
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
