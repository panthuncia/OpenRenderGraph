#pragma once

#include <BasicTelemetry/GpuCapture.h>

#include <rhi.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

/**
 * @brief GPU hardware counters per named range through NVIDIA's Nsight Perf SDK (NVPW's range profiler), on D3D12 or
 * Vulkan.
 *
 * A capture is configured (the metrics, the ranges to measure), armed, and then collected over as many frames as the
 * metrics need replay passes: each frame is one profiler pass, bracketed on the queue by BeginFrameCapture and
 * EndFrameCapture, and the selected ranges in it are pushed and popped in the command buffers around the work they
 * name. Once every pass is submitted the counters are decoded into a CaptureResult.
 *
 * Queue-level calls (sessions, passes, decoding) submit to the queue, so a host makes them only where it may use the
 * queue. A host that owns its queue can let a range start the frame's pass itself (QueueControl::Inline); a host that
 * shares another's queue (a D3D11 translation layer's) makes them only at its frame boundaries, on the thread that
 * submits, and ranges then only record (QueueControl::Boundaries).
 *
 * Built only with ORG_ENABLE_NVPERF; otherwise every call reports that it is unavailable.
 */
namespace org::telemetry::nvperf
{
using basic_telemetry::gpu_capture::CaptureConfiguration;
using basic_telemetry::gpu_capture::CaptureResult;
using basic_telemetry::gpu_capture::MetricRequest;
using basic_telemetry::gpu_capture::PassFilter;
using basic_telemetry::gpu_capture::RangeResult;

// The queue a capture profiles, as native handles (Vulkan: the loader's or the next layer's entry points).
struct QueueTarget
{
    rhi::Backend backend = rhi::Backend::Null;
    void* device = nullptr;          // ID3D12Device* / VkDevice
    void* queue = nullptr;           // ID3D12CommandQueue* / VkQueue
    void* instance = nullptr;        // VkInstance
    void* physicalDevice = nullptr;  // VkPhysicalDevice
    void* getInstanceProcAddr = nullptr;
    void* getDeviceProcAddr = nullptr;
    std::string name = "Graphics";
};

// A host that runs on BasicRHI.
QueueTarget MakeQueueTarget(rhi::Backend backend, rhi::Device device, rhi::Queue queue, std::string name);

enum class QueueControl
{
    Inline,      // a range may begin the frame's pass (the host owns the queue)
    Boundaries,  // only BeginFrameCapture and EndFrameCapture make queue calls
};
void SetQueueControl(QueueControl control);

// Loads nvperf_grfx_host.dll from a_path before the first profiler call, for hosts that delay-load it (a plugin must
// not fail to load when the file is absent). False when it cannot be loaded; every call is then unavailable.
bool LoadLibraryFrom(const std::filesystem::path& a_path);
bool Available();

// Logs the chip and whether the profiler supports it on the queue's device.
void LogStartupProbe(const QueueTarget& queue);

bool ConfigureCapture(const CaptureConfiguration& configuration, std::string& error);
bool ArmCapture(std::uint64_t sampleId, std::string& error);
bool CaptureConfigured();
bool CaptureArmed();
bool CaptureComplete();
std::size_t ScheduledPassCount();
std::optional<CaptureResult> TakeCaptureResult();
void ResetCaptureConfiguration();
// Also writes every decoded range as a CSV row (empty: none).
void SetCsvPath(std::filesystem::path path);

// A session's pending GPU operations (the queue waits on them between passes) are serviced by its own thread, so a
// host needs no servicing in its frame waits.
bool CaptureActive();

// Frame boundaries: where the host may make queue calls, before and after the frame's submissions.
void BeginFrameCapture(const QueueTarget& queue, std::uint64_t frameNumber);
void EndFrameCapture(const QueueTarget& queue, std::uint64_t frameNumber);

// Whether a range of that name on that queue is one the capture measures.
bool RangeSelected(const char* queueName, const char* rangeName);

// A range in a native command buffer (ID3D12GraphicsCommandList* / VkCommandBuffer) that will execute on the queue,
// when it is selected. It records while a capture is armed, whenever the command buffer executes: only the ranges that
// execute inside a pass are counted (under QueueControl::Inline, a range begins the frame's pass). True when it was
// pushed: the caller pops it in the same command buffer.
bool PushRange(const QueueTarget& queue, void* commandBuffer, const char* queueName, const char* rangeName);
// False when no range is open in that command buffer (e.g. it was opened in another one).
bool PopRange(rhi::Backend backend, void* commandBuffer);

// A render graph's per-pass hooks (PassExecutionContext::beginGpuPassRange / endGpuPassRange).
void BeginPassRange(rhi::Backend backend, rhi::Device device, rhi::CommandList commandList, rhi::Queue queue, const char* queueName, const char* passName);
void EndPassRange(rhi::Backend backend, rhi::CommandList commandList, rhi::Queue queue);
}  // namespace org::telemetry::nvperf
