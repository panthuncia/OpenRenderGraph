#ifndef OPEN_RENDER_GRAPH_CONTRIBUTOR_API_H
#define OPEN_RENDER_GRAPH_CONTRIBUTOR_API_H

#include <stddef.h>
#include <stdint.h>
#include <rhi_c.h>

#if defined(_WIN32)
#define ORG_C_CALL __cdecl
#else
#define ORG_C_CALL
#endif

#define ORG_CONTRIBUTOR_API_VERSION 2u

typedef enum org_c_pass_kind {
    ORG_C_PASS_RASTER = 0,
    ORG_C_PASS_COMPUTE = 1,
    ORG_C_PASS_COPY = 2
} org_c_pass_kind;

typedef enum org_c_binding_kind {
    ORG_C_BINDING_SRV = 0,
    ORG_C_BINDING_UAV = 1,
    ORG_C_BINDING_RTV = 2,
    ORG_C_BINDING_DSV = 3,
    ORG_C_BINDING_INDIRECT = 4,
    ORG_C_BINDING_COPY_SOURCE = 5,
    ORG_C_BINDING_COPY_DESTINATION = 6
} org_c_binding_kind;

typedef struct org_c_string_view {
    const char* data;
    size_t size_bytes;
} org_c_string_view;

typedef enum org_c_result {
    ORG_C_OK = 0,
    ORG_C_INVALID_ARGUMENT = 1,
    ORG_C_INCOMPATIBLE_VERSION = 2,
    ORG_C_INVALID_STATE = 3,
    ORG_C_UNSUPPORTED = 4,
    ORG_C_HOST_ERROR = 5
} org_c_result;

typedef enum org_c_resource_kind {
    ORG_C_RESOURCE_TEXTURE_2D = 0,
    ORG_C_RESOURCE_BUFFER = 1
} org_c_resource_kind;

typedef enum org_c_resource_lifetime {
    ORG_C_RESOURCE_TRANSIENT = 0,
    ORG_C_RESOURCE_CONTRIBUTOR_PERSISTENT = 1,
    ORG_C_RESOURCE_IMPORTED = 2
} org_c_resource_lifetime;

typedef enum org_c_format {
    ORG_C_FORMAT_UNKNOWN = 0,
    ORG_C_FORMAT_R8G8B8A8_UNORM = 1,
    ORG_C_FORMAT_R16G16_FLOAT = 2,
    ORG_C_FORMAT_R16G16B16A16_FLOAT = 3,
    ORG_C_FORMAT_R32_FLOAT = 4,
    ORG_C_FORMAT_R32_UINT = 5,
    ORG_C_FORMAT_R32G32_UINT = 6,
    ORG_C_FORMAT_D32_FLOAT = 7
} org_c_format;

typedef enum org_c_resource_usage {
    ORG_C_RESOURCE_USAGE_NONE = 0,
    ORG_C_RESOURCE_USAGE_SRV = 1u << 0,
    ORG_C_RESOURCE_USAGE_UAV = 1u << 1,
    ORG_C_RESOURCE_USAGE_RTV = 1u << 2,
    ORG_C_RESOURCE_USAGE_DSV = 1u << 3,
    ORG_C_RESOURCE_USAGE_INDIRECT = 1u << 4,
    ORG_C_RESOURCE_USAGE_COPY_SOURCE = 1u << 5,
    ORG_C_RESOURCE_USAGE_COPY_DESTINATION = 1u << 6
} org_c_resource_usage;

typedef struct org_c_resource_desc {
    uint32_t structure_size;
    uint32_t kind;
    org_c_string_view symbolic_resource;
    org_c_string_view debug_name;
    uint32_t lifetime;
    uint32_t usage_flags;
    uint32_t format;
    uint32_t element_stride;
    uint64_t element_count;
    uint32_t element_count_per_render_pixel;
    uint32_t reserved0;
    uint32_t width;
    uint32_t height;
    uint32_t width_scale_numerator;
    uint32_t width_scale_denominator;
    uint32_t height_scale_numerator;
    uint32_t height_scale_denominator;
    uint32_t array_size;
    uint32_t mip_levels;
    uint32_t clear_value[4];
    basicrhi_c_resource imported_resource;
} org_c_resource_desc;

typedef struct org_c_frame_context {
    uint32_t structure_size;
    uint32_t api_version;
    uint64_t frame_index;
    uint64_t completion_value;
    float delta_seconds;
    uint32_t render_width;
    uint32_t render_height;
    uint32_t display_width;
    uint32_t display_height;
    float viewport[4];
    float jitter_pixels[2];
    uint32_t reversed_depth;
    uint32_t clip_depth_zero_to_one;
    float view[16];
    float projection_jittered[16];
    float projection_unjittered[16];
    float previous_view[16];
    float previous_projection_jittered[16];
    float previous_projection_unjittered[16];
} org_c_frame_context;

typedef struct org_c_host_api {
    uint32_t structure_size;
    uint32_t api_version;
    void* context;
    const basicrhi_c_device_api_v1* device;
    uint64_t (ORG_C_CALL *completed_value)(void* context);
    org_c_result (ORG_C_CALL *request_graph_rebuild)(void* context, uint32_t reason);
    org_c_result (ORG_C_CALL *schedule_upload)(void* context, basicrhi_c_resource destination,
        uint64_t destination_offset, const void* data, size_t size_bytes, uint64_t* completion_value);
    void (ORG_C_CALL *retire_resource)(void* context, basicrhi_c_resource resource, uint64_t completion_value);
    void (ORG_C_CALL *log)(void* context, uint32_t severity, const char* message, size_t message_bytes);
} org_c_host_api;

typedef struct org_c_binding {
    uint32_t structure_size;
    uint32_t kind;
    org_c_string_view symbolic_resource;
    basicrhi_c_resource resource;
    basicrhi_c_view view;
    basicrhi_c_descriptor_heap descriptor_heap;
    uint32_t descriptor_slot;
    uint32_t reserved;
} org_c_binding;

typedef struct org_c_execute_context {
    uint32_t structure_size;
    uint32_t api_version;
    uint64_t frame_index;
    uint64_t completion_value;
    basicrhi_c_borrowed_recorder_v1 recorder;
    const org_c_binding* bindings;
    size_t binding_count;
    const void* host_data;
} org_c_execute_context;

typedef struct org_c_pass {
    uint32_t structure_size;
    uint32_t kind;
    org_c_string_view name;
    org_c_string_view technique_path;
    const org_c_binding* declared_bindings;
    size_t declared_binding_count;
    const org_c_string_view* after_passes;
    size_t after_pass_count;
    const org_c_string_view* before_passes;
    size_t before_pass_count;
    void* user_data;
    void (ORG_C_CALL *execute)(void* user_data, const org_c_execute_context* context);
} org_c_pass;

typedef struct org_c_contributor_api {
    uint32_t structure_size;
    uint32_t api_version;
    void* contributor;
    org_c_result (ORG_C_CALL *initialize)(void* contributor, const org_c_host_api* host);
    void (ORG_C_CALL *shutdown)(void* contributor);
    size_t (ORG_C_CALL *describe_resources)(void* contributor, org_c_resource_desc* resources, size_t capacity);
    size_t (ORG_C_CALL *describe_passes)(void* contributor, org_c_pass* passes, size_t capacity);
    org_c_result (ORG_C_CALL *prepare_frame)(void* contributor, const org_c_frame_context* frame);
    void (ORG_C_CALL *on_graph_registered)(void* contributor, uint64_t registration);
    void (ORG_C_CALL *on_graph_rebuilt)(void* contributor, uint64_t graph_revision);
    void (ORG_C_CALL *on_graph_unregistered)(void* contributor);
    void (ORG_C_CALL *on_device_lost)(void* contributor);
    org_c_result (ORG_C_CALL *on_device_restored)(void* contributor, const org_c_host_api* host);
    org_c_result (ORG_C_CALL *query_capabilities)(void* contributor, void* output, size_t output_size);
    org_c_result (ORG_C_CALL *query_diagnostics)(void* contributor, void* output, size_t output_size);
} org_c_contributor_api;

#endif
