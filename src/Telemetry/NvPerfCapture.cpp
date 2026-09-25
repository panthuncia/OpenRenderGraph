// volk (through BasicRHI's Vulkan interop) must precede every Vulkan header in this translation unit.
#include <rhi_interop_vulkan.h>
#if BASICRHI_ENABLE_D3D12
#include <rhi_interop_dx12.h>
#endif

#include "Telemetry/NvPerfCapture.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if ORG_ENABLE_NVPERF
#include <nvperf_host.h>
#include <nvperf_target.h>
#include <nvperf_vulkan_host.h>
#include <nvperf_vulkan_target.h>
#if BASICRHI_ENABLE_D3D12
#include <nvperf_d3d12_host.h>
#include <nvperf_d3d12_target.h>
#endif
#include <Windows.h>
#endif

namespace org::telemetry::nvperf {

QueueTarget MakeQueueTarget(rhi::Backend backend, rhi::Device device, rhi::Queue queue, std::string name)
{
    QueueTarget target{};
    target.backend = backend;
    target.name = std::move(name);
    switch (backend) {
#if BASICRHI_ENABLE_D3D12
    case rhi::Backend::D3D12:
        target.device = rhi::dx12::get_device(device);
        target.queue = rhi::dx12::get_queue(queue);
        break;
#endif
#if BASICRHI_HAS_VULKAN_HEADERS
    case rhi::Backend::Vulkan:
        target.instance = rhi::vulkan::get_instance(device);
        target.physicalDevice = rhi::vulkan::get_physical_device(device);
        target.device = rhi::vulkan::get_device(device);
        target.queue = rhi::vulkan::get_queue(queue);
        target.getInstanceProcAddr = reinterpret_cast<void*>(vkGetInstanceProcAddr);
        target.getDeviceProcAddr = reinterpret_cast<void*>(rhi::vulkan::get_device_proc_addr());
        break;
#endif
    default:
        break;
    }
    return target;
}

#if ORG_ENABLE_NVPERF
namespace {

const char* StatusName(NVPA_Status status)
{
    const char* statusName = nullptr;
    const char* comment = nullptr;
    NVPW_NVPAStatusToString(status, &statusName, &comment);
    return statusName ? statusName : "NVPA_STATUS_UNKNOWN";
}

bool Failed(NVPA_Status status, const char* operation)
{
    if (status == NVPA_STATUS_SUCCESS)
        return false;
    spdlog::warn("NVPerf: {} failed with {}", operation, StatusName(status));
    return true;
}

std::string CsvEscape(std::string_view text)
{
    if (text.find_first_of("\",\r\n") == std::string_view::npos)
        return std::string(text);
    std::string escaped = "\"";
    for (char ch : text) {
        if (ch == '"')
            escaped.push_back('"');
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

struct SelectedMetric {
    MetricRequest spec;
    NVPW_MetricEvalRequest request{};
};

// What a default capture measures: time, throughput of the main units, the instruction mix and the warp stall reasons.
std::vector<MetricRequest> DefaultMetrics()
{
    const auto counter = [](const char* name) { return MetricRequest{ {}, name, {}, {}, NVPW_METRIC_TYPE_COUNTER, NVPW_ROLLUP_OP_SUM, NVPW_SUBMETRIC_NONE, false }; };
    const auto throughput = [](const char* name) { return MetricRequest{ {}, name, {}, {}, NVPW_METRIC_TYPE_THROUGHPUT, NVPW_ROLLUP_OP_AVG, NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_ELAPSED, false }; };
    const auto ratio = [](const char* name) { return MetricRequest{ {}, name, {}, {}, NVPW_METRIC_TYPE_RATIO, NVPW_ROLLUP_OP_AVG, NVPW_SUBMETRIC_RATIO, false }; };
    return {
        counter("gpu__time_duration"),
        throughput("sm__throughput"),
        throughput("l1tex__throughput"),
        throughput("lts__throughput"),
        throughput("dram__throughput"),
        counter("smsp__inst_executed"),
        ratio("smsp__warps_active"),
        ratio("smsp__warps_eligible"),
        ratio("smsp__warp_issue_stalled_long_scoreboard_per_warp_active"),
        ratio("smsp__warp_issue_stalled_short_scoreboard_per_warp_active"),
        ratio("smsp__warp_issue_stalled_imc_miss_per_warp_active"),
        ratio("smsp__warp_issue_stalled_tex_throttle_per_warp_active"),
        ratio("smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active"),
        ratio("smsp__warp_issue_stalled_wait_per_warp_active"),
        ratio("smsp__warp_issue_stalled_no_instruction_per_warp_active"),
        ratio("smsp__warp_issue_stalled_mio_throttle_per_warp_active"),
        ratio("smsp__warp_issue_stalled_lg_throttle_per_warp_active"),
        ratio("tpc__average_registers_per_thread_shader_ps"),
        counter("tpc__warps_launched_shader_ps"),
    };
}

// Per-backend profiler calls. Everything else in a capture is the same on both.
struct Backend {
    rhi::Backend api = rhi::Backend::Null;

    static bool Supported(rhi::Backend backend)
    {
#if BASICRHI_ENABLE_D3D12
        if (backend == rhi::Backend::D3D12)
            return true;
#endif
        return backend == rhi::Backend::Vulkan;
    }

    bool DeviceIndex(const QueueTarget& target, size_t& index) const
    {
        if (api == rhi::Backend::Vulkan) {
            NVPW_VK_LoadDriver_Params load{ NVPW_VK_LoadDriver_Params_STRUCT_SIZE };
            load.instance = static_cast<VkInstance>(target.instance);
            if (Failed(NVPW_VK_LoadDriver(&load), "NVPW_VK_LoadDriver"))
                return false;
            NVPW_VK_Device_GetDeviceIndex_Params params{ NVPW_VK_Device_GetDeviceIndex_Params_STRUCT_SIZE };
            params.instance = static_cast<VkInstance>(target.instance);
            params.physicalDevice = static_cast<VkPhysicalDevice>(target.physicalDevice);
            params.device = static_cast<VkDevice>(target.device);
            params.pfnGetInstanceProcAddr = target.getInstanceProcAddr;
            params.pfnGetDeviceProcAddr = target.getDeviceProcAddr;
            if (Failed(NVPW_VK_Device_GetDeviceIndex(&params), "NVPW_VK_Device_GetDeviceIndex"))
                return false;
            index = params.deviceIndex;
            return true;
        }
#if BASICRHI_ENABLE_D3D12
        NVPW_D3D12_LoadDriver_Params load{ NVPW_D3D12_LoadDriver_Params_STRUCT_SIZE };
        if (Failed(NVPW_D3D12_LoadDriver(&load), "NVPW_D3D12_LoadDriver"))
            return false;
        NVPW_D3D12_Device_GetDeviceIndex_Params params{ NVPW_D3D12_Device_GetDeviceIndex_Params_STRUCT_SIZE };
        params.pDevice = static_cast<ID3D12Device*>(target.device);
        if (Failed(NVPW_D3D12_Device_GetDeviceIndex(&params), "NVPW_D3D12_Device_GetDeviceIndex"))
            return false;
        index = params.deviceIndex;
        return true;
#else
        return false;
#endif
    }

    bool IsGpuSupported(size_t deviceIndex) const
    {
        if (api == rhi::Backend::Vulkan) {
            NVPW_VK_Profiler_IsGpuSupported_Params params{ NVPW_VK_Profiler_IsGpuSupported_Params_STRUCT_SIZE };
            params.deviceIndex = deviceIndex;
            return !Failed(NVPW_VK_Profiler_IsGpuSupported(&params), "NVPW_VK_Profiler_IsGpuSupported") && params.isSupported;
        }
#if BASICRHI_ENABLE_D3D12
        NVPW_D3D12_Profiler_IsGpuSupported_Params params{ NVPW_D3D12_Profiler_IsGpuSupported_Params_STRUCT_SIZE };
        params.deviceIndex = deviceIndex;
        return !Failed(NVPW_D3D12_Profiler_IsGpuSupported(&params), "NVPW_D3D12_Profiler_IsGpuSupported") && params.isSupported;
#else
        return false;
#endif
    }

    std::vector<uint8_t> CounterAvailability(const QueueTarget& target) const
    {
        std::vector<uint8_t> image;
        if (api == rhi::Backend::Vulkan) {
            NVPW_VK_Profiler_Queue_GetCounterAvailability_Params params{ NVPW_VK_Profiler_Queue_GetCounterAvailability_Params_STRUCT_SIZE };
            params.instance = static_cast<VkInstance>(target.instance);
            params.physicalDevice = static_cast<VkPhysicalDevice>(target.physicalDevice);
            params.device = static_cast<VkDevice>(target.device);
            params.queue = static_cast<VkQueue>(target.queue);
            params.pfnGetInstanceProcAddr = target.getInstanceProcAddr;
            params.pfnGetDeviceProcAddr = target.getDeviceProcAddr;
            if (Failed(NVPW_VK_Profiler_Queue_GetCounterAvailability(&params), "NVPW_VK_Profiler_Queue_GetCounterAvailability(size)"))
                return {};
            image.resize(params.counterAvailabilityImageSize);
            params.pCounterAvailabilityImage = image.data();
            if (Failed(NVPW_VK_Profiler_Queue_GetCounterAvailability(&params), "NVPW_VK_Profiler_Queue_GetCounterAvailability"))
                return {};
            image.resize(params.counterAvailabilityImageSize);
            return image;
        }
#if BASICRHI_ENABLE_D3D12
        NVPW_D3D12_Profiler_Queue_GetCounterAvailability_Params params{ NVPW_D3D12_Profiler_Queue_GetCounterAvailability_Params_STRUCT_SIZE };
        params.pCommandQueue = static_cast<ID3D12CommandQueue*>(target.queue);
        if (Failed(NVPW_D3D12_Profiler_Queue_GetCounterAvailability(&params), "NVPW_D3D12_Profiler_Queue_GetCounterAvailability(size)"))
            return {};
        image.resize(params.counterAvailabilityImageSize);
        params.pCounterAvailabilityImage = image.data();
        if (Failed(NVPW_D3D12_Profiler_Queue_GetCounterAvailability(&params), "NVPW_D3D12_Profiler_Queue_GetCounterAvailability"))
            return {};
        image.resize(params.counterAvailabilityImageSize);
#endif
        return image;
    }

    // A metrics evaluator for the chip, or (with a counter data image) for the device that produced the image.
    NVPW_MetricsEvaluator* CreateEvaluator(std::vector<uint8_t>& scratch, const char* chipName, const std::vector<uint8_t>* image) const
    {
        if (api == rhi::Backend::Vulkan) {
            NVPW_VK_MetricsEvaluator_CalculateScratchBufferSize_Params size{ NVPW_VK_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE };
            size.pChipName = chipName;
            if (Failed(NVPW_VK_MetricsEvaluator_CalculateScratchBufferSize(&size), "NVPW_VK_MetricsEvaluator_CalculateScratchBufferSize"))
                return nullptr;
            scratch.assign(size.scratchBufferSize, 0);
            NVPW_VK_MetricsEvaluator_Initialize_Params init{ NVPW_VK_MetricsEvaluator_Initialize_Params_STRUCT_SIZE };
            init.pScratchBuffer = scratch.data();
            init.scratchBufferSize = scratch.size();
            init.pChipName = image ? nullptr : chipName;
            init.pCounterDataImage = image ? image->data() : nullptr;
            init.counterDataImageSize = image ? image->size() : 0;
            if (Failed(NVPW_VK_MetricsEvaluator_Initialize(&init), "NVPW_VK_MetricsEvaluator_Initialize"))
                return nullptr;
            return init.pMetricsEvaluator;
        }
#if BASICRHI_ENABLE_D3D12
        NVPW_D3D12_MetricsEvaluator_CalculateScratchBufferSize_Params size{ NVPW_D3D12_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE };
        size.pChipName = chipName;
        if (Failed(NVPW_D3D12_MetricsEvaluator_CalculateScratchBufferSize(&size), "NVPW_D3D12_MetricsEvaluator_CalculateScratchBufferSize"))
            return nullptr;
        scratch.assign(size.scratchBufferSize, 0);
        NVPW_D3D12_MetricsEvaluator_Initialize_Params init{ NVPW_D3D12_MetricsEvaluator_Initialize_Params_STRUCT_SIZE };
        init.pScratchBuffer = scratch.data();
        init.scratchBufferSize = scratch.size();
        init.pChipName = image ? nullptr : chipName;
        init.pCounterDataImage = image ? image->data() : nullptr;
        init.counterDataImageSize = image ? image->size() : 0;
        if (Failed(NVPW_D3D12_MetricsEvaluator_Initialize(&init), "NVPW_D3D12_MetricsEvaluator_Initialize"))
            return nullptr;
        return init.pMetricsEvaluator;
#else
        (void)scratch;
        (void)chipName;
        (void)image;
        return nullptr;
#endif
    }

    NVPW_RawCounterConfig* CreateRawCounterConfig(const char* chipName) const
    {
        if (api == rhi::Backend::Vulkan) {
            NVPW_VK_RawCounterConfig_Create_Params create{ NVPW_VK_RawCounterConfig_Create_Params_STRUCT_SIZE };
            create.pChipName = chipName;
            create.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
            return Failed(NVPW_VK_RawCounterConfig_Create(&create), "NVPW_VK_RawCounterConfig_Create") ? nullptr : create.pRawCounterConfig;
        }
#if BASICRHI_ENABLE_D3D12
        NVPW_D3D12_RawCounterConfig_Create_Params create{ NVPW_D3D12_RawCounterConfig_Create_Params_STRUCT_SIZE };
        create.pChipName = chipName;
        create.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
        return Failed(NVPW_D3D12_RawCounterConfig_Create(&create), "NVPW_D3D12_RawCounterConfig_Create") ? nullptr : create.pRawCounterConfig;
#else
        (void)chipName;
        return nullptr;
#endif
    }

    size_t TraceBufferSize(size_t maxRangesPerPass) const
    {
        if (api == rhi::Backend::Vulkan) {
            NVPW_VK_Profiler_CalcTraceBufferSize_Params params{ NVPW_VK_Profiler_CalcTraceBufferSize_Params_STRUCT_SIZE };
            params.maxRangesPerPass = maxRangesPerPass;
            params.avgRangeNameLength = 96;
            return Failed(NVPW_VK_Profiler_CalcTraceBufferSize(&params), "NVPW_VK_Profiler_CalcTraceBufferSize") ? 0 : params.traceBufferSize;
        }
#if BASICRHI_ENABLE_D3D12
        NVPW_D3D12_Profiler_CalcTraceBufferSize_Params params{ NVPW_D3D12_Profiler_CalcTraceBufferSize_Params_STRUCT_SIZE };
        params.maxRangesPerPass = maxRangesPerPass;
        params.avgRangeNameLength = 96;
        return Failed(NVPW_D3D12_Profiler_CalcTraceBufferSize(&params), "NVPW_D3D12_Profiler_CalcTraceBufferSize") ? 0 : params.traceBufferSize;
#else
        (void)maxRangesPerPass;
        return 0;
#endif
    }

    // The counter data image and its scratch buffer, sized for maxRanges ranges.
    bool InitializeCounterData(const std::vector<uint8_t>& prefix, size_t maxRanges, std::vector<uint8_t>& image, std::vector<uint8_t>& scratch) const
    {
        if (api == rhi::Backend::Vulkan) {
            NVPW_VK_Profiler_CounterDataImageOptions options{ NVPW_VK_Profiler_CounterDataImageOptions_STRUCT_SIZE };
            options.pCounterDataPrefix = prefix.data();
            options.counterDataPrefixSize = prefix.size();
            options.maxNumRanges = static_cast<uint32_t>(maxRanges);
            options.maxNumRangeTreeNodes = static_cast<uint32_t>(maxRanges);
            options.maxRangeNameLength = 128;
            NVPW_VK_Profiler_CounterDataImage_CalculateSize_Params size{ NVPW_VK_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE };
            size.counterDataImageOptionsSize = NVPW_VK_Profiler_CounterDataImageOptions_STRUCT_SIZE;
            size.pOptions = &options;
            if (Failed(NVPW_VK_Profiler_CounterDataImage_CalculateSize(&size), "NVPW_VK_Profiler_CounterDataImage_CalculateSize"))
                return false;
            image.assign(size.counterDataImageSize, 0);
            NVPW_VK_Profiler_CounterDataImage_Initialize_Params init{ NVPW_VK_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE };
            init.counterDataImageOptionsSize = NVPW_VK_Profiler_CounterDataImageOptions_STRUCT_SIZE;
            init.pOptions = &options;
            init.counterDataImageSize = image.size();
            init.pCounterDataImage = image.data();
            if (Failed(NVPW_VK_Profiler_CounterDataImage_Initialize(&init), "NVPW_VK_Profiler_CounterDataImage_Initialize"))
                return false;
            NVPW_VK_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchSize{ NVPW_VK_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE };
            scratchSize.counterDataImageSize = image.size();
            scratchSize.pCounterDataImage = image.data();
            if (Failed(NVPW_VK_Profiler_CounterDataImage_CalculateScratchBufferSize(&scratchSize), "NVPW_VK_Profiler_CounterDataImage_CalculateScratchBufferSize"))
                return false;
            scratch.assign(scratchSize.counterDataScratchBufferSize, 0);
            NVPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params scratchInit{ NVPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE };
            scratchInit.counterDataImageSize = image.size();
            scratchInit.pCounterDataImage = image.data();
            scratchInit.counterDataScratchBufferSize = scratch.size();
            scratchInit.pCounterDataScratchBuffer = scratch.data();
            return !Failed(NVPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer(&scratchInit), "NVPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer");
        }
#if BASICRHI_ENABLE_D3D12
        NVPW_D3D12_Profiler_CounterDataImageOptions options{ NVPW_D3D12_Profiler_CounterDataImageOptions_STRUCT_SIZE };
        options.pCounterDataPrefix = prefix.data();
        options.counterDataPrefixSize = prefix.size();
        options.maxNumRanges = static_cast<uint32_t>(maxRanges);
        options.maxNumRangeTreeNodes = static_cast<uint32_t>(maxRanges);
        options.maxRangeNameLength = 128;
        NVPW_D3D12_Profiler_CounterDataImage_CalculateSize_Params size{ NVPW_D3D12_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE };
        size.counterDataImageOptionsSize = NVPW_D3D12_Profiler_CounterDataImageOptions_STRUCT_SIZE;
        size.pOptions = &options;
        if (Failed(NVPW_D3D12_Profiler_CounterDataImage_CalculateSize(&size), "NVPW_D3D12_Profiler_CounterDataImage_CalculateSize"))
            return false;
        image.assign(size.counterDataImageSize, 0);
        NVPW_D3D12_Profiler_CounterDataImage_Initialize_Params init{ NVPW_D3D12_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE };
        init.counterDataImageOptionsSize = NVPW_D3D12_Profiler_CounterDataImageOptions_STRUCT_SIZE;
        init.pOptions = &options;
        init.counterDataImageSize = image.size();
        init.pCounterDataImage = image.data();
        if (Failed(NVPW_D3D12_Profiler_CounterDataImage_Initialize(&init), "NVPW_D3D12_Profiler_CounterDataImage_Initialize"))
            return false;
        NVPW_D3D12_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchSize{ NVPW_D3D12_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE };
        scratchSize.counterDataImageSize = image.size();
        scratchSize.pCounterDataImage = image.data();
        if (Failed(NVPW_D3D12_Profiler_CounterDataImage_CalculateScratchBufferSize(&scratchSize), "NVPW_D3D12_Profiler_CounterDataImage_CalculateScratchBufferSize"))
            return false;
        scratch.assign(scratchSize.counterDataScratchBufferSize, 0);
        NVPW_D3D12_Profiler_CounterDataImage_InitializeScratchBuffer_Params scratchInit{ NVPW_D3D12_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE };
        scratchInit.counterDataImageSize = image.size();
        scratchInit.pCounterDataImage = image.data();
        scratchInit.counterDataScratchBufferSize = scratch.size();
        scratchInit.pCounterDataScratchBuffer = scratch.data();
        return !Failed(NVPW_D3D12_Profiler_CounterDataImage_InitializeScratchBuffer(&scratchInit), "NVPW_D3D12_Profiler_CounterDataImage_InitializeScratchBuffer");
#else
        (void)prefix;
        (void)maxRanges;
        (void)image;
        (void)scratch;
        return false;
#endif
    }

    bool BeginSession(const QueueTarget& target, size_t traceBuffers, size_t traceBufferSize, size_t maxRangesPerPass) const
    {
        if (api == rhi::Backend::Vulkan) {
            NVPW_VK_Profiler_Queue_BeginSession_Params params{ NVPW_VK_Profiler_Queue_BeginSession_Params_STRUCT_SIZE };
            params.device = static_cast<VkDevice>(target.device);
            params.queue = static_cast<VkQueue>(target.queue);
            params.numTraceBuffers = traceBuffers;
            params.traceBufferSize = traceBufferSize;
            params.maxRangesPerPass = maxRangesPerPass;
            params.instance = static_cast<VkInstance>(target.instance);
            params.physicalDevice = static_cast<VkPhysicalDevice>(target.physicalDevice);
            params.pfnGetInstanceProcAddr = target.getInstanceProcAddr;
            params.pfnGetDeviceProcAddr = target.getDeviceProcAddr;
            return !Failed(NVPW_VK_Profiler_Queue_BeginSession(&params), "NVPW_VK_Profiler_Queue_BeginSession");
        }
#if BASICRHI_ENABLE_D3D12
        NVPW_D3D12_Profiler_Queue_BeginSession_Params params{ NVPW_D3D12_Profiler_Queue_BeginSession_Params_STRUCT_SIZE };
        params.pCommandQueue = static_cast<ID3D12CommandQueue*>(target.queue);
        params.numTraceBuffers = traceBuffers;
        params.traceBufferSize = traceBufferSize;
        params.maxRangesPerPass = maxRangesPerPass;
        return !Failed(NVPW_D3D12_Profiler_Queue_BeginSession(&params), "NVPW_D3D12_Profiler_Queue_BeginSession");
#else
        return false;
#endif
    }

    bool EndSession(void* queue, uint32_t timeoutMs, bool& timedOut) const
    {
        if (api == rhi::Backend::Vulkan) {
            NVPW_VK_Profiler_Queue_EndSession_Params params{ NVPW_VK_Profiler_Queue_EndSession_Params_STRUCT_SIZE };
            params.queue = static_cast<VkQueue>(queue);
            params.timeout = timeoutMs;
            const bool ok = !Failed(NVPW_VK_Profiler_Queue_EndSession(&params), "NVPW_VK_Profiler_Queue_EndSession");
            timedOut = params.timeoutExpired;
            return ok;
        }
#if BASICRHI_ENABLE_D3D12
        NVPW_D3D12_Profiler_Queue_EndSession_Params params{ NVPW_D3D12_Profiler_Queue_EndSession_Params_STRUCT_SIZE };
        params.pCommandQueue = static_cast<ID3D12CommandQueue*>(queue);
        params.timeout = timeoutMs;
        const bool ok = !Failed(NVPW_D3D12_Profiler_Queue_EndSession(&params), "NVPW_D3D12_Profiler_Queue_EndSession");
        timedOut = params.timeoutExpired;
        return ok;
#else
        (void)queue;
        (void)timeoutMs;
        timedOut = false;
        return false;
#endif
    }

    bool SetConfig(void* queue, const std::vector<uint8_t>& config, size_t passIndex, uint16_t targetNestingLevel) const
    {
        if (api == rhi::Backend::Vulkan) {
            NVPW_VK_Profiler_Queue_SetConfig_Params params{ NVPW_VK_Profiler_Queue_SetConfig_Params_STRUCT_SIZE };
            params.queue = static_cast<VkQueue>(queue);
            params.pConfig = config.data();
            params.configSize = config.size();
            params.minNestingLevel = 1;
            params.numNestingLevels = 1;
            params.passIndex = passIndex;
            params.targetNestingLevel = targetNestingLevel;
            return !Failed(NVPW_VK_Profiler_Queue_SetConfig(&params), "NVPW_VK_Profiler_Queue_SetConfig");
        }
#if BASICRHI_ENABLE_D3D12
        NVPW_D3D12_Profiler_Queue_SetConfig_Params params{ NVPW_D3D12_Profiler_Queue_SetConfig_Params_STRUCT_SIZE };
        params.pCommandQueue = static_cast<ID3D12CommandQueue*>(queue);
        params.pConfig = config.data();
        params.configSize = config.size();
        params.minNestingLevel = 1;
        params.numNestingLevels = 1;
        params.passIndex = passIndex;
        params.targetNestingLevel = targetNestingLevel;
        return !Failed(NVPW_D3D12_Profiler_Queue_SetConfig(&params), "NVPW_D3D12_Profiler_Queue_SetConfig");
#else
        (void)queue;
        (void)config;
        (void)passIndex;
        (void)targetNestingLevel;
        return false;
#endif
    }

    bool BeginPass(void* queue) const
    {
        if (api == rhi::Backend::Vulkan) {
            NVPW_VK_Profiler_Queue_BeginPass_Params params{ NVPW_VK_Profiler_Queue_BeginPass_Params_STRUCT_SIZE };
            params.queue = static_cast<VkQueue>(queue);
            return !Failed(NVPW_VK_Profiler_Queue_BeginPass(&params), "NVPW_VK_Profiler_Queue_BeginPass");
        }
#if BASICRHI_ENABLE_D3D12
        NVPW_D3D12_Profiler_Queue_BeginPass_Params params{ NVPW_D3D12_Profiler_Queue_BeginPass_Params_STRUCT_SIZE };
        params.pCommandQueue = static_cast<ID3D12CommandQueue*>(queue);
        return !Failed(NVPW_D3D12_Profiler_Queue_BeginPass(&params), "NVPW_D3D12_Profiler_Queue_BeginPass");
#else
        (void)queue;
        return false;
#endif
    }

    bool EndPass(void* queue, size_t& nextPassIndex, uint16_t& targetNestingLevel, bool& allPassesSubmitted) const
    {
        if (api == rhi::Backend::Vulkan) {
            NVPW_VK_Profiler_Queue_EndPass_Params params{ NVPW_VK_Profiler_Queue_EndPass_Params_STRUCT_SIZE };
            params.queue = static_cast<VkQueue>(queue);
            if (Failed(NVPW_VK_Profiler_Queue_EndPass(&params), "NVPW_VK_Profiler_Queue_EndPass"))
                return false;
            nextPassIndex = params.passIndex;
            targetNestingLevel = params.targetNestingLevel;
            allPassesSubmitted = params.allPassesSubmitted != 0;
            return true;
        }
#if BASICRHI_ENABLE_D3D12
        NVPW_D3D12_Profiler_Queue_EndPass_Params params{ NVPW_D3D12_Profiler_Queue_EndPass_Params_STRUCT_SIZE };
        params.pCommandQueue = static_cast<ID3D12CommandQueue*>(queue);
        if (Failed(NVPW_D3D12_Profiler_Queue_EndPass(&params), "NVPW_D3D12_Profiler_Queue_EndPass"))
            return false;
        nextPassIndex = params.passIndex;
        targetNestingLevel = params.targetNestingLevel;
        allPassesSubmitted = params.allPassesSubmitted != 0;
        return true;
#else
        (void)queue;
        (void)nextPassIndex;
        (void)targetNestingLevel;
        (void)allPassesSubmitted;
        return false;
#endif
    }

    struct Decoded {
        bool ok = false, onePass = false, allPasses = false;
        size_t rangesDropped = 0, bytesDropped = 0, passIndex = 0;
    };
    Decoded Decode(void* queue, std::vector<uint8_t>& image, std::vector<uint8_t>& scratch) const
    {
        Decoded out{};
        if (api == rhi::Backend::Vulkan) {
            NVPW_VK_Profiler_Queue_DecodeCounters_Params params{ NVPW_VK_Profiler_Queue_DecodeCounters_Params_STRUCT_SIZE };
            params.queue = static_cast<VkQueue>(queue);
            params.counterDataImageSize = image.size();
            params.pCounterDataImage = image.data();
            params.counterDataScratchBufferSize = scratch.size();
            params.pCounterDataScratchBuffer = scratch.data();
            out.ok = !Failed(NVPW_VK_Profiler_Queue_DecodeCounters(&params), "NVPW_VK_Profiler_Queue_DecodeCounters");
            out.onePass = params.onePassCollected;
            out.allPasses = params.allPassesCollected;
            out.rangesDropped = params.numRangesDropped;
            out.bytesDropped = params.numTraceBytesDropped;
            out.passIndex = params.passIndexDecoded;
            return out;
        }
#if BASICRHI_ENABLE_D3D12
        NVPW_D3D12_Profiler_Queue_DecodeCounters_Params params{ NVPW_D3D12_Profiler_Queue_DecodeCounters_Params_STRUCT_SIZE };
        params.pCommandQueue = static_cast<ID3D12CommandQueue*>(queue);
        params.counterDataImageSize = image.size();
        params.pCounterDataImage = image.data();
        params.counterDataScratchBufferSize = scratch.size();
        params.pCounterDataScratchBuffer = scratch.data();
        out.ok = !Failed(NVPW_D3D12_Profiler_Queue_DecodeCounters(&params), "NVPW_D3D12_Profiler_Queue_DecodeCounters");
        out.onePass = params.onePassCollected;
        out.allPasses = params.allPassesCollected;
        out.rangesDropped = params.numRangesDropped;
        out.bytesDropped = params.numTraceBytesDropped;
        out.passIndex = params.passIndexDecoded;
#else
        (void)queue;
        (void)image;
        (void)scratch;
#endif
        return out;
    }

    void ServicePending(void* queue, uint32_t operations, uint32_t timeoutMs) const
    {
        if (api == rhi::Backend::Vulkan) {
            NVPW_VK_Queue_ServicePendingGpuOperations_Params params{ NVPW_VK_Queue_ServicePendingGpuOperations_Params_STRUCT_SIZE };
            params.queue = static_cast<VkQueue>(queue);
            params.numOperations = operations;
            params.timeout = timeoutMs;
            (void)NVPW_VK_Queue_ServicePendingGpuOperations(&params);
            return;
        }
#if BASICRHI_ENABLE_D3D12
        NVPW_D3D12_Queue_ServicePendingGpuOperations_Params params{ NVPW_D3D12_Queue_ServicePendingGpuOperations_Params_STRUCT_SIZE };
        params.pCommandQueue = static_cast<ID3D12CommandQueue*>(queue);
        params.numOperations = operations;
        params.timeout = timeoutMs;
        (void)NVPW_D3D12_Queue_ServicePendingGpuOperations(&params);
#else
        (void)queue;
        (void)operations;
        (void)timeoutMs;
#endif
    }

    bool PushRange(void* commandBuffer, const char* name) const
    {
        if (api == rhi::Backend::Vulkan) {
            NVPW_VK_Profiler_CommandBuffer_PushRange_Params params{ NVPW_VK_Profiler_CommandBuffer_PushRange_Params_STRUCT_SIZE };
            params.commandBuffer = static_cast<VkCommandBuffer>(commandBuffer);
            params.pRangeName = name;
            params.rangeNameLength = std::strlen(name);
            return !Failed(NVPW_VK_Profiler_CommandBuffer_PushRange(&params), "NVPW_VK_Profiler_CommandBuffer_PushRange");
        }
#if BASICRHI_ENABLE_D3D12
        NVPW_D3D12_Profiler_CommandList_PushRange_Params params{ NVPW_D3D12_Profiler_CommandList_PushRange_Params_STRUCT_SIZE };
        params.pCommandList = static_cast<ID3D12GraphicsCommandList*>(commandBuffer);
        params.pRangeName = name;
        params.rangeNameLength = std::strlen(name);
        return !Failed(NVPW_D3D12_Profiler_CommandList_PushRange(&params), "NVPW_D3D12_Profiler_CommandList_PushRange");
#else
        (void)commandBuffer;
        (void)name;
        return false;
#endif
    }

    bool PopRange(void* commandBuffer) const
    {
        if (api == rhi::Backend::Vulkan) {
            NVPW_VK_Profiler_CommandBuffer_PopRange_Params params{ NVPW_VK_Profiler_CommandBuffer_PopRange_Params_STRUCT_SIZE };
            params.commandBuffer = static_cast<VkCommandBuffer>(commandBuffer);
            return !Failed(NVPW_VK_Profiler_CommandBuffer_PopRange(&params), "NVPW_VK_Profiler_CommandBuffer_PopRange");
        }
#if BASICRHI_ENABLE_D3D12
        NVPW_D3D12_Profiler_CommandList_PopRange_Params params{ NVPW_D3D12_Profiler_CommandList_PopRange_Params_STRUCT_SIZE };
        params.pCommandList = static_cast<ID3D12GraphicsCommandList*>(commandBuffer);
        return !Failed(NVPW_D3D12_Profiler_CommandList_PopRange(&params), "NVPW_D3D12_Profiler_CommandList_PopRange");
#else
        (void)commandBuffer;
        return false;
#endif
    }
};

struct QueueCapture {
    QueueTarget target;
    std::vector<uint8_t> counterDataImage;
    std::vector<uint8_t> counterDataScratch;
    // The session's pending GPU operations (the queue waits on them between passes), serviced until EndSession returns
    // it: nothing the host's threads wait on may service them, since those waits are what they block.
    std::thread service;
    bool sessionActive = false;
    bool passActive = false;
    bool allPassesSubmitted = false;
    bool decoded = false;
    size_t nextPassIndex = 0;
    uint16_t targetNestingLevel = 1;
};

struct Profiler {
    std::mutex mutex;
    QueueControl queueControl = QueueControl::Inline;
    Backend backend{};
    bool configured = false;
    bool armed = false;
    bool initialized = false;
    bool failed = false;
    bool finished = false;
    uint64_t captureStartFrame = 0;
    uint64_t captureEndFrame = 0;
    uint64_t sampleId = 0;
    std::string chipName;
    std::string controllerQueueName = "Graphics";
    std::filesystem::path csvPath;
    std::vector<MetricRequest> requestedMetrics;
    std::vector<MetricRequest> unsupportedMetrics;
    std::vector<PassFilter> passFilters;
    std::vector<SelectedMetric> metrics;
    std::vector<uint8_t> configImage;
    std::vector<uint8_t> counterDataPrefix;
    size_t configPassCount = 0;
    size_t traceBufferSize = 0;
    size_t maxRangesPerPass = 512;
    size_t traceBufferCount = 8;
    uint32_t syncTimeoutMs = 10000;
    uint64_t framesWithoutRange = 0;
    uint64_t rangesPushed = 0;
    uint64_t droppedRanges = 0;
    uint64_t droppedTraceBytes = 0;
    std::string error;
    std::optional<CaptureResult> result;
    std::unordered_map<void*, QueueCapture> queues;
    std::unordered_map<void*, uint32_t> openRanges;  // per command buffer
};

Profiler& Get()
{
    static Profiler profiler;
    return profiler;
}

HMODULE g_library = nullptr;
bool g_libraryChecked = false;

bool InitializeLibrary()
{
    static bool initialized = false;
    static bool available = false;
    if (initialized)
        return available;
    initialized = true;
    // A delay-loading host must have loaded the library by path first (LoadLibraryFrom): an unresolved delay-loaded
    // import would raise rather than fail.
    if (!g_library && !::GetModuleHandleW(L"nvperf_grfx_host.dll") && !::LoadLibraryW(L"nvperf_grfx_host.dll")) {
        spdlog::warn("NVPerf: nvperf_grfx_host.dll is not available");
        return false;
    }
    NVPW_InitializeHost_Params host{ NVPW_InitializeHost_Params_STRUCT_SIZE };
    if (Failed(NVPW_InitializeHost(&host), "NVPW_InitializeHost"))
        return false;
    NVPW_InitializeTarget_Params target{ NVPW_InitializeTarget_Params_STRUCT_SIZE };
    if (Failed(NVPW_InitializeTarget(&target), "NVPW_InitializeTarget"))
        return false;
    available = true;
    spdlog::info("NVPerf: host and target libraries initialized");
    return true;
}

std::optional<size_t> FindMetricIndex(NVPW_MetricsEvaluator* evaluator, const MetricRequest& spec)
{
    NVPW_MetricsEvaluator_GetMetricNames_Params names{ NVPW_MetricsEvaluator_GetMetricNames_Params_STRUCT_SIZE };
    names.pMetricsEvaluator = evaluator;
    names.metricType = spec.metricType;
    if (Failed(NVPW_MetricsEvaluator_GetMetricNames(&names), "NVPW_MetricsEvaluator_GetMetricNames"))
        return std::nullopt;
    for (size_t i = 0; i < names.numMetrics; ++i) {
        const char* metricName = names.pMetricNames + names.pMetricNameBeginIndices[i];
        if (metricName && spec.name == metricName)
            return i;
    }
    return std::nullopt;
}

void DestroyEvaluator(NVPW_MetricsEvaluator* evaluator)
{
    NVPW_MetricsEvaluator_Destroy_Params destroy{ NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE };
    destroy.pMetricsEvaluator = evaluator;
    (void)NVPW_MetricsEvaluator_Destroy(&destroy);
}

void DestroyRawConfig(NVPW_RawCounterConfig* config)
{
    NVPW_RawCounterConfig_Destroy_Params destroy{ NVPW_RawCounterConfig_Destroy_Params_STRUCT_SIZE };
    destroy.pRawCounterConfig = config;
    (void)NVPW_RawCounterConfig_Destroy(&destroy);
}

// The metrics the chip has, the raw counters behind them, and the configuration and counter data prefix that collect
// them.
bool BuildConfig(Profiler& profiler, const std::vector<uint8_t>& availability)
{
    profiler.metrics.clear();
    profiler.unsupportedMetrics.clear();
    profiler.configImage.clear();
    profiler.counterDataPrefix.clear();
    const char* chip = profiler.chipName.c_str();

    std::vector<uint8_t> scratch;
    NVPW_MetricsEvaluator* evaluator = profiler.backend.CreateEvaluator(scratch, chip, nullptr);
    if (!evaluator)
        return false;
    std::unordered_set<std::string> rawCounterNames;
    const auto requested = profiler.requestedMetrics.empty() ? DefaultMetrics() : profiler.requestedMetrics;
    for (const MetricRequest& spec : requested) {
        const auto index = FindMetricIndex(evaluator, spec);
        if (!index) {
            spdlog::warn("NVPerf: metric '{}' is unavailable on chip '{}'", spec.name, profiler.chipName);
            profiler.unsupportedMetrics.push_back(spec);
            continue;
        }
        SelectedMetric metric{};
        metric.spec = spec;
        metric.request.metricIndex = *index;
        metric.request.metricType = spec.metricType;
        metric.request.rollupOp = spec.rollupOp;
        metric.request.submetric = spec.submetric;
        NVPW_MetricsEvaluator_GetMetricRawDependencies_Params deps{ NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE };
        deps.pMetricsEvaluator = evaluator;
        deps.pMetricEvalRequests = &metric.request;
        deps.numMetricEvalRequests = 1;
        deps.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        deps.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
        if (Failed(NVPW_MetricsEvaluator_GetMetricRawDependencies(&deps), "NVPW_MetricsEvaluator_GetMetricRawDependencies(count)"))
            continue;
        std::vector<const char*> dependencies(deps.numRawDependencies);
        deps.ppRawDependencies = dependencies.data();
        deps.numRawDependencies = dependencies.size();
        if (Failed(NVPW_MetricsEvaluator_GetMetricRawDependencies(&deps), "NVPW_MetricsEvaluator_GetMetricRawDependencies"))
            continue;
        for (const char* dependency : dependencies)
            if (dependency)
                rawCounterNames.insert(dependency);
        profiler.metrics.push_back(metric);
    }
    DestroyEvaluator(evaluator);

    std::vector<NVPW_RawCounterRequest> rawCounters;
    rawCounters.reserve(rawCounterNames.size());
    for (const std::string& name : rawCounterNames) {
        NVPW_RawCounterRequest request{};
        request.pRawCounterName = name.c_str();
        request.domain = NVPW_RAW_COUNTER_DOMAIN_INVALID;
        request.keepInstances = false;
        rawCounters.push_back(request);
    }
    if (profiler.metrics.empty() || rawCounters.empty()) {
        spdlog::warn("NVPerf: no requested metric could be configured");
        return false;
    }

    NVPW_RawCounterConfig* config = profiler.backend.CreateRawCounterConfig(chip);
    if (!config)
        return false;
    if (!availability.empty()) {
        NVPW_RawCounterConfig_SetCounterAvailability_Params set{ NVPW_RawCounterConfig_SetCounterAvailability_Params_STRUCT_SIZE };
        set.pRawCounterConfig = config;
        set.pCounterAvailabilityImage = availability.data();
        (void)NVPW_RawCounterConfig_SetCounterAvailability(&set);
    }
    NVPW_RawCounterConfig_GetAllAvailableRawCounterDomains_Params domainCount{ NVPW_RawCounterConfig_GetAllAvailableRawCounterDomains_Params_STRUCT_SIZE };
    domainCount.pRawCounterConfig = config;
    if (Failed(NVPW_RawCounterConfig_GetAllAvailableRawCounterDomains(&domainCount), "NVPW_RawCounterConfig_GetAllAvailableRawCounterDomains(count)")) {
        DestroyRawConfig(config);
        return false;
    }
    std::vector<NVPW_RawCounterDomain> domains(domainCount.numAvailableDomains);
    NVPW_RawCounterConfig_GetAllAvailableRawCounterDomains_Params domainValues{ NVPW_RawCounterConfig_GetAllAvailableRawCounterDomains_Params_STRUCT_SIZE };
    domainValues.pRawCounterConfig = config;
    domainValues.numAvailableDomains = domains.size();
    domainValues.pAvailableDomains = domains.data();
    if (Failed(NVPW_RawCounterConfig_GetAllAvailableRawCounterDomains(&domainValues), "NVPW_RawCounterConfig_GetAllAvailableRawCounterDomains")) {
        DestroyRawConfig(config);
        return false;
    }
    domains.resize(domainValues.numAvailableDomains);
    NVPW_RawCounterConfig_BeginPassGroup_Params begin{ NVPW_RawCounterConfig_BeginPassGroup_Params_STRUCT_SIZE };
    begin.pRawCounterConfig = config;
    begin.numDomains = domains.size();
    begin.pDomains = domains.data();
    NVPW_RawCounterConfig_AddRawCounters_Params add{ NVPW_RawCounterConfig_AddRawCounters_Params_STRUCT_SIZE };
    add.pRawCounterConfig = config;
    add.rawCounterRequestStructSize = NVPW_RAW_COUNTER_REQUEST_STRUCT_SIZE;
    add.numRawCounterRequests = rawCounters.size();
    add.pRawCounterRequests = rawCounters.data();
    NVPW_RawCounterConfig_EndPassGroup_Params end{ NVPW_RawCounterConfig_EndPassGroup_Params_STRUCT_SIZE };
    end.pRawCounterConfig = config;
    end.numDomains = domains.size();
    end.pDomains = domains.data();
    NVPW_RawCounterConfig_GenerateConfigImage_Params generate{ NVPW_RawCounterConfig_GenerateConfigImage_Params_STRUCT_SIZE };
    generate.pRawCounterConfig = config;
    if (Failed(NVPW_RawCounterConfig_BeginPassGroup(&begin), "NVPW_RawCounterConfig_BeginPassGroup") ||
        Failed(NVPW_RawCounterConfig_AddRawCounters(&add), "NVPW_RawCounterConfig_AddRawCounters") ||
        Failed(NVPW_RawCounterConfig_EndPassGroup(&end), "NVPW_RawCounterConfig_EndPassGroup") ||
        Failed(NVPW_RawCounterConfig_GenerateConfigImage(&generate), "NVPW_RawCounterConfig_GenerateConfigImage")) {
        DestroyRawConfig(config);
        return false;
    }
    NVPW_RawCounterConfig_GetConfigImage_Params image{ NVPW_RawCounterConfig_GetConfigImage_Params_STRUCT_SIZE };
    image.pRawCounterConfig = config;
    if (Failed(NVPW_RawCounterConfig_GetConfigImage(&image), "NVPW_RawCounterConfig_GetConfigImage(size)")) {
        DestroyRawConfig(config);
        return false;
    }
    profiler.configImage.resize(image.bytesCopied);
    image.bytesAllocated = profiler.configImage.size();
    image.pBuffer = profiler.configImage.data();
    if (Failed(NVPW_RawCounterConfig_GetConfigImage(&image), "NVPW_RawCounterConfig_GetConfigImage")) {
        DestroyRawConfig(config);
        return false;
    }
    NVPW_RawCounterConfig_GetNumPasses_Params passes{ NVPW_RawCounterConfig_GetNumPasses_Params_STRUCT_SIZE };
    passes.pRawCounterConfig = config;
    if (!Failed(NVPW_RawCounterConfig_GetNumPasses(&passes), "NVPW_RawCounterConfig_GetNumPasses"))
        profiler.configPassCount = passes.numPasses;
    DestroyRawConfig(config);

    NVPW_CounterDataBuilder_Create_Params builderCreate{ NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE };
    builderCreate.pChipName = chip;
    if (Failed(NVPW_CounterDataBuilder_Create(&builderCreate), "NVPW_CounterDataBuilder_Create"))
        return false;
    NVPA_CounterDataBuilder* builder = builderCreate.pCounterDataBuilder;
    const auto destroyBuilder = [&] {
        NVPW_CounterDataBuilder_Destroy_Params destroy{ NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE };
        destroy.pCounterDataBuilder = builder;
        (void)NVPW_CounterDataBuilder_Destroy(&destroy);
    };
    NVPW_CounterDataBuilder_AddRawCounters_Params builderAdd{ NVPW_CounterDataBuilder_AddRawCounters_Params_STRUCT_SIZE };
    builderAdd.pCounterDataBuilder = builder;
    builderAdd.rawCounterRequestStructSize = NVPW_RAW_COUNTER_REQUEST_STRUCT_SIZE;
    builderAdd.numRawCounterRequests = rawCounters.size();
    builderAdd.pRawCounterRequests = rawCounters.data();
    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params prefix{ NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE };
    prefix.pCounterDataBuilder = builder;
    if (Failed(NVPW_CounterDataBuilder_AddRawCounters(&builderAdd), "NVPW_CounterDataBuilder_AddRawCounters") ||
        Failed(NVPW_CounterDataBuilder_GetCounterDataPrefix(&prefix), "NVPW_CounterDataBuilder_GetCounterDataPrefix(size)")) {
        destroyBuilder();
        return false;
    }
    profiler.counterDataPrefix.resize(prefix.bytesCopied);
    prefix.bytesAllocated = profiler.counterDataPrefix.size();
    prefix.pBuffer = profiler.counterDataPrefix.data();
    const bool prefixOk = !Failed(NVPW_CounterDataBuilder_GetCounterDataPrefix(&prefix), "NVPW_CounterDataBuilder_GetCounterDataPrefix");
    destroyBuilder();
    if (!prefixOk)
        return false;

    profiler.traceBufferSize = profiler.backend.TraceBufferSize(profiler.maxRangesPerPass);
    if (!profiler.traceBufferSize)
        return false;
    spdlog::info("NVPerf: capture configured chip='{}' metrics={} rawCounters={} replayPasses={} maxRangesPerPass={}", profiler.chipName,
        profiler.metrics.size(), rawCounters.size(), profiler.configPassCount, profiler.maxRangesPerPass);
    return true;
}

// The first time an armed capture reaches a queue: the chip, the configuration. Queue calls.
bool Prepare(Profiler& profiler, const QueueTarget& target, uint64_t frameNumber)
{
    if (!profiler.armed || profiler.failed || profiler.finished)
        return false;
    if (profiler.initialized)
        return true;
    if (!Backend::Supported(target.backend)) {
        profiler.failed = true;
        profiler.error = "NVPerf capture: unsupported backend";
        return false;
    }
    if (!InitializeLibrary()) {
        profiler.failed = true;
        profiler.error = "NVPerf is unavailable";
        return false;
    }
    profiler.backend.api = target.backend;
    size_t deviceIndex = 0;
    if (!profiler.backend.DeviceIndex(target, deviceIndex)) {
        profiler.failed = true;
        profiler.error = "NVPerf could not identify the device";
        return false;
    }
    NVPW_Device_GetNames_Params names{ NVPW_Device_GetNames_Params_STRUCT_SIZE };
    names.deviceIndex = deviceIndex;
    if (Failed(NVPW_Device_GetNames(&names), "NVPW_Device_GetNames") || !names.pChipName) {
        profiler.failed = true;
        profiler.error = "NVPerf could not name the chip";
        return false;
    }
    profiler.chipName = names.pChipName;
    const auto availability = profiler.backend.CounterAvailability(target);
    if (!BuildConfig(profiler, availability)) {
        profiler.failed = true;
        profiler.error = "failed to build the NVPerf metric configuration";
        return false;
    }
    for (const auto& unsupported : profiler.unsupportedMetrics) {
        if (unsupported.required) {
            profiler.failed = true;
            profiler.error = "required NVPerf metric is unavailable: " + unsupported.name;
            return false;
        }
    }
    profiler.captureStartFrame = frameNumber;
    profiler.traceBufferCount = std::max({ profiler.traceBufferCount, profiler.configPassCount + 2, size_t{ 5 } });
    profiler.initialized = true;
    spdlog::info("NVPerf: capture armed at frame {} on queue '{}' traceBuffers={}", frameNumber, target.name, profiler.traceBufferCount);
    return true;
}

QueueCapture& CaptureOf(Profiler& profiler, const QueueTarget& target)
{
    auto& capture = profiler.queues[target.queue];
    if (!capture.target.queue)
        capture.target = target;
    return capture;
}

// Ends the queue's session, its open pass first, and retires the service thread EndSession releases. Queue calls.
bool CloseSession(Profiler& profiler, QueueCapture& capture)
{
    if (capture.passActive) {
        size_t nextPassIndex = 0;
        uint16_t targetNestingLevel = 1;
        bool allPassesSubmitted = false;
        (void)profiler.backend.EndPass(capture.target.queue, nextPassIndex, targetNestingLevel, allPassesSubmitted);
        capture.passActive = false;
    }
    bool ended = true;
    if (capture.sessionActive) {
        bool timedOut = false;
        ended = profiler.backend.EndSession(capture.target.queue, profiler.syncTimeoutMs, timedOut) && !timedOut;
        capture.sessionActive = false;
    }
    if (capture.service.joinable()) {
        if (ended)
            capture.service.join();
        else
            capture.service.detach();  // it holds only copies; it returns when the profiler lets go of the queue
    }
    return ended;
}

// Every finished or failed capture closed its sessions; a thread still running here belongs to a session the host
// abandoned, and it holds only copies.
void ForgetQueues(Profiler& profiler)
{
    for (auto& [queue, capture] : profiler.queues)
        if (capture.service.joinable())
            capture.service.detach();
    profiler.queues.clear();
}

// A failure mid-capture: the session must still end, or the queue keeps waiting on operations no one services.
void Fail(Profiler& profiler, QueueCapture& capture, const char* error)
{
    profiler.failed = true;
    profiler.error = error;
    (void)CloseSession(profiler, capture);
}

// The session, then the pass, on the queue. Queue calls.
bool EnsurePass(Profiler& profiler, QueueCapture& capture)
{
    if (capture.allPassesSubmitted || capture.decoded)
        return false;
    if (!capture.sessionActive) {
        if (!profiler.backend.InitializeCounterData(profiler.counterDataPrefix, profiler.maxRangesPerPass, capture.counterDataImage, capture.counterDataScratch) ||
            !profiler.backend.BeginSession(capture.target, profiler.traceBufferCount, profiler.traceBufferSize, profiler.maxRangesPerPass)) {
            profiler.failed = true;
            profiler.error = "NVPerf could not begin a session";
            return false;
        }
        capture.sessionActive = true;
        capture.service = std::thread([backend = profiler.backend, queue = capture.target.queue] { backend.ServicePending(queue, 0, 0xFFFFFFFFu); });
        spdlog::info("NVPerf: session began on queue '{}'", capture.target.name);
    }
    if (capture.passActive)
        return true;
    const bool configured = profiler.backend.SetConfig(capture.target.queue, profiler.configImage, capture.nextPassIndex, capture.targetNestingLevel);
    if (!configured || !profiler.backend.BeginPass(capture.target.queue)) {
        Fail(profiler, capture, "NVPerf could not begin a pass");
        return false;
    }
    capture.passActive = true;
    return true;
}

std::string RangeName(const std::vector<uint8_t>& image, size_t rangeIndex)
{
    NVPW_Profiler_CounterData_GetRangeDescriptions_Params desc{ NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE };
    desc.pCounterDataImage = image.data();
    desc.rangeIndex = rangeIndex;
    if (Failed(NVPW_Profiler_CounterData_GetRangeDescriptions(&desc), "NVPW_Profiler_CounterData_GetRangeDescriptions(count)"))
        return {};
    std::vector<const char*> descriptions(desc.numDescriptions);
    desc.ppDescriptions = descriptions.data();
    desc.numDescriptions = descriptions.size();
    if (Failed(NVPW_Profiler_CounterData_GetRangeDescriptions(&desc), "NVPW_Profiler_CounterData_GetRangeDescriptions") || descriptions.empty() ||
        !descriptions.back())
        return {};
    return descriptions.back();
}

bool EvaluateRanges(Profiler& profiler, const QueueCapture& capture)
{
    NVPW_CounterData_GetNumRanges_Params numRanges{ NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE };
    numRanges.pCounterDataImage = capture.counterDataImage.data();
    if (Failed(NVPW_CounterData_GetNumRanges(&numRanges), "NVPW_CounterData_GetNumRanges"))
        return false;
    if (!numRanges.numRanges) {
        spdlog::warn("NVPerf: no ranges were collected on queue '{}'", capture.target.name);
        return true;
    }
    std::vector<uint8_t> scratch;
    NVPW_MetricsEvaluator* evaluator = profiler.backend.CreateEvaluator(scratch, profiler.chipName.c_str(), &capture.counterDataImage);
    if (!evaluator)
        return false;

    std::ofstream csv;
    if (!profiler.csvPath.empty()) {
        std::error_code ec;
        if (profiler.csvPath.has_parent_path())
            std::filesystem::create_directories(profiler.csvPath.parent_path(), ec);
        const bool header = !std::filesystem::exists(profiler.csvPath, ec) || std::filesystem::file_size(profiler.csvPath, ec) == 0;
        csv.open(profiler.csvPath, std::ios::app);
        if (csv && header) {
            csv << "sample,start_frame,queue,range_index,range,occurrence";
            for (const auto& metric : profiler.metrics)
                csv << ',' << CsvEscape(metric.spec.outputName.empty() ? metric.spec.name : metric.spec.outputName);
            csv << '\n';
        }
    }

    std::vector<NVPW_MetricEvalRequest> requests;
    for (const auto& metric : profiler.metrics)
        requests.push_back(metric.request);
    std::vector<double> values(requests.size());
    std::unordered_map<std::string, uint32_t> occurrences;
    for (size_t rangeIndex = 0; rangeIndex < numRanges.numRanges; ++rangeIndex) {
        std::fill(values.begin(), values.end(), 0.0);
        NVPW_MetricsEvaluator_EvaluateToGpuValues_Params eval{ NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE };
        eval.pMetricsEvaluator = evaluator;
        eval.pMetricEvalRequests = requests.data();
        eval.numMetricEvalRequests = requests.size();
        eval.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        eval.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
        eval.pCounterDataImage = capture.counterDataImage.data();
        eval.counterDataImageSize = capture.counterDataImage.size();
        eval.rangeIndex = rangeIndex;
        eval.pMetricValues = values.data();
        if (Failed(NVPW_MetricsEvaluator_EvaluateToGpuValues(&eval), "NVPW_MetricsEvaluator_EvaluateToGpuValues"))
            continue;
        std::string name = RangeName(capture.counterDataImage, rangeIndex);
        if (name.empty())
            name = "<unnamed>";
        RangeResult range;
        range.queue = capture.target.name;
        range.passName = name;
        range.occurrence = occurrences[name]++;
        range.rangeIndex = static_cast<uint32_t>(rangeIndex);
        range.values = values;
        if (csv) {
            csv << profiler.sampleId << ',' << profiler.captureStartFrame << ',' << CsvEscape(range.queue) << ',' << rangeIndex << ',' << CsvEscape(name) << ','
                << range.occurrence;
            for (double value : values)
                csv << ',' << value;
            csv << '\n';
        }
        if (!profiler.result)
            profiler.result.emplace();
        profiler.result->ranges.push_back(std::move(range));
    }
    DestroyEvaluator(evaluator);
    spdlog::info("NVPerf: {} ranges decoded on queue '{}'", numRanges.numRanges, capture.target.name);
    return true;
}

bool DecodeCapture(Profiler& profiler, QueueCapture& capture)
{
    for (size_t iteration = 0;; ++iteration) {
        const auto decoded = profiler.backend.Decode(capture.target.queue, capture.counterDataImage, capture.counterDataScratch);
        if (!decoded.ok)
            return false;
        if (decoded.rangesDropped || decoded.bytesDropped) {
            profiler.droppedRanges += decoded.rangesDropped;
            profiler.droppedTraceBytes += decoded.bytesDropped;
            spdlog::warn("NVPerf: decode on queue '{}' dropped {} ranges, {} trace bytes", capture.target.name, decoded.rangesDropped, decoded.bytesDropped);
        }
        if (decoded.allPasses)
            break;
        if (!decoded.onePass)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));  // the GPU is still on the last pass
        if (iteration > 2048) {
            spdlog::warn("NVPerf: decode on queue '{}' exceeded its iteration guard", capture.target.name);
            return false;
        }
    }
    if (!EvaluateRanges(profiler, capture))
        return false;
    capture.decoded = true;
    return true;
}

void FinishCapture(Profiler& profiler, uint64_t frameNumber, bool decoded)
{
    profiler.finished = decoded;
    if (!decoded)
        return;
    profiler.captureEndFrame = frameNumber;
    if (!profiler.result)
        profiler.result.emplace();
    auto& result = *profiler.result;
    result.sampleId = profiler.sampleId;
    result.startFrame = profiler.captureStartFrame;
    result.endFrame = frameNumber;
    result.scheduledPasses = profiler.configPassCount;
    result.droppedRanges = profiler.droppedRanges;
    result.droppedTraceBytes = profiler.droppedTraceBytes;
    result.success = !profiler.droppedRanges && !profiler.droppedTraceBytes;
    result.error = result.success ? std::string{} : "NVPerf dropped counter data";
    result.chipName = profiler.chipName;
    result.unsupportedMetrics = profiler.unsupportedMetrics;
    result.metrics.clear();
    for (const auto& metric : profiler.metrics)
        result.metrics.push_back(metric.spec);
    spdlog::info("NVPerf: capture complete, frames {}-{}", profiler.captureStartFrame, frameNumber);
}

bool NameMatches(const std::string& filter, std::string_view name)
{
    if (!filter.empty() && filter.back() == '*')
        return name.substr(0, filter.size() - 1) == std::string_view(filter).substr(0, filter.size() - 1);
    return filter == name;
}

bool Selected(const Profiler& profiler, std::string_view queueName, std::string_view name)
{
    if (profiler.passFilters.empty())
        return true;
    return std::ranges::any_of(profiler.passFilters, [&](const PassFilter& filter) {
        return NameMatches(filter.name, name) && (filter.queue.empty() || filter.queue == queueName);
    });
}

}  // namespace
#endif

void SetQueueControl(QueueControl control)
{
#if ORG_ENABLE_NVPERF
    auto& profiler = Get();
    std::lock_guard lock(profiler.mutex);
    profiler.queueControl = control;
#else
    (void)control;
#endif
}

bool LoadLibraryFrom(const std::filesystem::path& path)
{
#if ORG_ENABLE_NVPERF
    if (!g_library)
        g_library = ::LoadLibraryW(path.c_str());
    if (!g_library)
        spdlog::warn("NVPerf: could not load {}", path.string());
    return g_library != nullptr;
#else
    (void)path;
    return false;
#endif
}

bool Available()
{
#if ORG_ENABLE_NVPERF
    return InitializeLibrary();
#else
    return false;
#endif
}

void LogStartupProbe(const QueueTarget& queue)
{
#if ORG_ENABLE_NVPERF
    if (!InitializeLibrary() || !Backend::Supported(queue.backend))
        return;
    Backend backend{ queue.backend };
    size_t index = 0;
    if (!backend.DeviceIndex(queue, index))
        return;
    NVPW_Device_GetNames_Params names{ NVPW_Device_GetNames_Params_STRUCT_SIZE };
    names.deviceIndex = index;
    const bool named = !Failed(NVPW_Device_GetNames(&names), "NVPW_Device_GetNames");
    spdlog::info("NVPerf: device {} '{}' chip '{}', range profiler supported: {}", index, named && names.pDeviceName ? names.pDeviceName : "?",
        named && names.pChipName ? names.pChipName : "?", backend.IsGpuSupported(index));
#else
    (void)queue;
#endif
}

bool ConfigureCapture(const CaptureConfiguration& configuration, std::string& error)
{
#if ORG_ENABLE_NVPERF
    auto& profiler = Get();
    std::lock_guard lock(profiler.mutex);
    if (profiler.armed && !profiler.finished && !profiler.failed) {
        error = "NVPerf capture is active";
        return false;
    }
    for (const auto& metric : configuration.metrics) {
        if (metric.name.empty()) {
            error = "NVPerf metric name cannot be empty";
            return false;
        }
    }
    profiler.configured = true;
    profiler.armed = false;
    profiler.initialized = false;
    profiler.failed = false;
    profiler.finished = false;
    profiler.result.reset();
    profiler.requestedMetrics = configuration.metrics;  // empty: DefaultMetrics
    profiler.passFilters = configuration.passes;
    profiler.controllerQueueName = configuration.controllerQueue.empty() ? "Graphics" : configuration.controllerQueue;
    profiler.syncTimeoutMs = configuration.syncTimeoutMs;
    return true;
#else
    (void)configuration;
    error = "built without NVPerf (ORG_ENABLE_NVPERF)";
    return false;
#endif
}

bool ArmCapture(uint64_t sampleId, std::string& error)
{
#if ORG_ENABLE_NVPERF
    auto& profiler = Get();
    std::lock_guard lock(profiler.mutex);
    if (!profiler.configured) {
        error = "NVPerf capture has not been configured";
        return false;
    }
    if (profiler.armed && !profiler.finished && !profiler.failed) {
        error = "NVPerf capture is already armed";
        return false;
    }
    ForgetQueues(profiler);
    profiler.openRanges.clear();
    profiler.armed = true;
    profiler.initialized = false;
    profiler.failed = false;
    profiler.finished = false;
    profiler.sampleId = sampleId;
    profiler.captureStartFrame = profiler.captureEndFrame = 0;
    profiler.framesWithoutRange = profiler.rangesPushed = 0;
    profiler.droppedRanges = profiler.droppedTraceBytes = 0;
    profiler.error.clear();
    profiler.result.emplace();
    profiler.result->sampleId = sampleId;
    return true;
#else
    (void)sampleId;
    error = "built without NVPerf (ORG_ENABLE_NVPERF)";
    return false;
#endif
}

bool CaptureConfigured()
{
#if ORG_ENABLE_NVPERF
    auto& profiler = Get();
    std::lock_guard lock(profiler.mutex);
    return profiler.configured;
#else
    return false;
#endif
}

bool CaptureArmed()
{
#if ORG_ENABLE_NVPERF
    auto& profiler = Get();
    std::lock_guard lock(profiler.mutex);
    return profiler.armed && !profiler.finished && !profiler.failed;
#else
    return false;
#endif
}

bool CaptureComplete()
{
#if ORG_ENABLE_NVPERF
    auto& profiler = Get();
    std::lock_guard lock(profiler.mutex);
    return profiler.armed && (profiler.finished || profiler.failed);
#else
    return false;
#endif
}

size_t ScheduledPassCount()
{
#if ORG_ENABLE_NVPERF
    auto& profiler = Get();
    std::lock_guard lock(profiler.mutex);
    return profiler.configPassCount;
#else
    return 0;
#endif
}

std::optional<CaptureResult> TakeCaptureResult()
{
#if ORG_ENABLE_NVPERF
    auto& profiler = Get();
    std::lock_guard lock(profiler.mutex);
    if (!profiler.armed || (!profiler.finished && !profiler.failed))
        return std::nullopt;
    if (!profiler.result)
        profiler.result.emplace();
    auto& result = *profiler.result;
    result.sampleId = profiler.sampleId;
    result.startFrame = profiler.captureStartFrame;
    result.endFrame = profiler.captureEndFrame;
    result.scheduledPasses = profiler.configPassCount;
    result.chipName = profiler.chipName;
    result.droppedRanges = profiler.droppedRanges;
    result.droppedTraceBytes = profiler.droppedTraceBytes;
    if (profiler.failed) {
        result.success = false;
        result.error = profiler.error.empty() ? "NVPerf capture failed" : profiler.error;
        result.unsupportedMetrics = profiler.unsupportedMetrics;
    }
    auto taken = std::move(profiler.result);
    profiler.result.reset();
    profiler.armed = false;
    return taken;
#else
    return std::nullopt;
#endif
}

void ResetCaptureConfiguration()
{
#if ORG_ENABLE_NVPERF
    auto& profiler = Get();
    std::lock_guard lock(profiler.mutex);
    ForgetQueues(profiler);
    profiler.openRanges.clear();
    profiler.requestedMetrics.clear();
    profiler.passFilters.clear();
    profiler.configured = profiler.armed = profiler.initialized = profiler.failed = profiler.finished = false;
    profiler.result.reset();
#endif
}

void SetCsvPath(std::filesystem::path path)
{
#if ORG_ENABLE_NVPERF
    auto& profiler = Get();
    std::lock_guard lock(profiler.mutex);
    profiler.csvPath = std::move(path);
#else
    (void)path;
#endif
}

bool CaptureActive()
{
#if ORG_ENABLE_NVPERF
    auto& profiler = Get();
    std::lock_guard lock(profiler.mutex);
    if (!profiler.initialized || profiler.failed || profiler.finished)
        return false;
    return std::ranges::any_of(profiler.queues, [](const auto& entry) { return entry.second.sessionActive || entry.second.passActive; });
#else
    return false;
#endif
}

void BeginFrameCapture(const QueueTarget& queue, uint64_t frameNumber)
{
#if ORG_ENABLE_NVPERF
    auto& profiler = Get();
    std::lock_guard lock(profiler.mutex);
    if (!Prepare(profiler, queue, frameNumber) || queue.name != profiler.controllerQueueName)
        return;
    (void)EnsurePass(profiler, CaptureOf(profiler, queue));
#else
    (void)queue;
    (void)frameNumber;
#endif
}

void EndFrameCapture(const QueueTarget& queue, uint64_t frameNumber)
{
#if ORG_ENABLE_NVPERF
    auto& profiler = Get();
    std::lock_guard lock(profiler.mutex);
    if (!profiler.initialized || profiler.failed || profiler.finished)
        return;
    const auto found = profiler.queues.find(queue.queue);
    if (found == profiler.queues.end())
        return;
    auto& capture = found->second;
    if (capture.passActive) {
        if (!profiler.rangesPushed && ++profiler.framesWithoutRange >= 8) {
            Fail(profiler, capture, "no selected NVPerf range was recorded within 8 frames");
            return;
        }
        profiler.rangesPushed = 0;
        bool allSubmitted = false;
        if (!profiler.backend.EndPass(capture.target.queue, capture.nextPassIndex, capture.targetNestingLevel, allSubmitted)) {
            capture.passActive = false;
            Fail(profiler, capture, "NVPerf could not end a pass");
            return;
        }
        capture.allPassesSubmitted = allSubmitted;
        capture.passActive = false;
    }
    if (!capture.sessionActive || !capture.allPassesSubmitted)
        return;
    const bool decoded = DecodeCapture(profiler, capture);
    if (!CloseSession(profiler, capture)) {
        profiler.failed = true;
        profiler.error = "NVPerf could not end the session";
        return;
    }
    FinishCapture(profiler, frameNumber, decoded);
    if (!decoded) {
        profiler.failed = true;
        profiler.error = "NVPerf could not decode the capture";
    }
#else
    (void)queue;
    (void)frameNumber;
#endif
}

bool RangeSelected(const char* queueName, const char* rangeName)
{
#if ORG_ENABLE_NVPERF
    auto& profiler = Get();
    std::lock_guard lock(profiler.mutex);
    return profiler.armed && !profiler.failed && !profiler.finished && Selected(profiler, queueName ? queueName : "", rangeName ? rangeName : "");
#else
    (void)queueName;
    (void)rangeName;
    return false;
#endif
}

bool PushRange(const QueueTarget& queue, void* commandBuffer, const char* queueName, const char* rangeName)
{
#if ORG_ENABLE_NVPERF
    auto& profiler = Get();
    std::lock_guard lock(profiler.mutex);
    if (!profiler.armed || profiler.failed || profiler.finished || !commandBuffer)
        return false;
    const std::string_view resolvedQueue = queueName && queueName[0] ? queueName : "Unknown";
    const char* name = rangeName && rangeName[0] ? rangeName : "<unnamed>";
    if (resolvedQueue != profiler.controllerQueueName || !Selected(profiler, resolvedQueue, name))
        return false;
    // Inline: the range begins the frame's pass. Boundaries: the range records whenever a capture is armed, since
    // its command buffer may be recorded frames ahead of the pass it executes in (a host's prepared epochs); one
    // executed outside a pass is not counted.
    if (profiler.queueControl == QueueControl::Inline && (!profiler.initialized || !EnsurePass(profiler, CaptureOf(profiler, queue))))
        return false;
    if (Backend{ queue.backend }.PushRange(commandBuffer, name)) {
        ++profiler.openRanges[commandBuffer];
        ++profiler.rangesPushed;
        return true;
    }
    return false;
#else
    (void)queue;
    (void)commandBuffer;
    (void)queueName;
    (void)rangeName;
    return false;
#endif
}

bool PopRange(rhi::Backend backend, void* commandBuffer)
{
#if ORG_ENABLE_NVPERF
    auto& profiler = Get();
    std::lock_guard lock(profiler.mutex);
    const auto open = profiler.openRanges.find(commandBuffer);
    if (open == profiler.openRanges.end() || !open->second)
        return false;
    if (!Backend{ backend }.PopRange(commandBuffer))
        return false;
    if (--open->second == 0)
        profiler.openRanges.erase(open);
    return true;
#else
    (void)backend;
    (void)commandBuffer;
    return false;
#endif
}

void BeginPassRange(rhi::Backend backend, rhi::Device device, rhi::CommandList commandList, rhi::Queue queue, const char* queueName, const char* passName)
{
    void* native = nullptr;
    switch (backend) {
#if BASICRHI_ENABLE_D3D12
    case rhi::Backend::D3D12:
        native = rhi::dx12::get_cmd_list(commandList);
        break;
#endif
#if BASICRHI_HAS_VULKAN_HEADERS
    case rhi::Backend::Vulkan:
        native = rhi::vulkan::get_cmd_list(commandList);
        break;
#endif
    default:
        return;
    }
    const auto target = MakeQueueTarget(backend, device, queue, queueName ? queueName : "Graphics");
    PushRange(target, native, queueName, passName);
}

void EndPassRange(rhi::Backend backend, rhi::CommandList commandList, rhi::Queue queue)
{
    (void)queue;
    switch (backend) {
#if BASICRHI_ENABLE_D3D12
    case rhi::Backend::D3D12:
        PopRange(backend, rhi::dx12::get_cmd_list(commandList));
        break;
#endif
#if BASICRHI_HAS_VULKAN_HEADERS
    case rhi::Backend::Vulkan:
        PopRange(backend, rhi::vulkan::get_cmd_list(commandList));
        break;
#endif
    default:
        break;
    }
}

}  // namespace org::telemetry::nvperf
