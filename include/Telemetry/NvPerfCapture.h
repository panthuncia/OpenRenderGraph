#pragma once

#include <BasicTelemetry/GpuCapture.h>

#include <rhi.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

/**
 * @brief GPU hardware counters per named range, through NVIDIA's Nsight Perf SDK (the NVPW range profiler), on D3D12
 * or Vulkan.
 *
 * A capture goes through these steps:
 * 1. It is configured with the metrics and the ranges to measure.
 * 2. It is armed.
 * 3. It is collected over as many frames as the metrics need replay passes. Each frame is one profiler pass,
 *    bracketed on the queue by BeginFrameCapture and EndFrameCapture. The selected ranges in the frame are pushed and
 *    popped in the command buffers around the work they name.
 * 4. Once every pass is submitted, the counters are decoded into a CaptureResult.
 *
 * Queue-level calls (sessions, passes, decoding) submit to the queue. So a host makes them only where it may use the
 * queue.
 * - A host that owns its queue can let a range start the frame's pass itself (QueueControl::Inline).
 * - A host that shares another's queue, such as a D3D11 translation layer's, makes queue calls only at its frame
 *   boundaries, on the thread that submits. Ranges then only record (QueueControl::Boundaries).
 *
 * Built only with ORG_ENABLE_NVPERF. Otherwise every call reports that it is unavailable.
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
// The number of ranges pushed (recorded into command buffers) since ArmCapture. Some hosts record their command
// buffers ahead of the frame they execute in, and record them again at arming. Such a host begins the first pass
// only once this is nonzero.
std::uint64_t RangesRecordedSinceArm();
bool RangeSelected(const char* queueName, const char* rangeName);

// Pushes a range in a native command buffer (ID3D12GraphicsCommandList* / VkCommandBuffer) that will execute on the
// queue, if the range is selected. It records while a capture is armed, whenever the command buffer executes. Only
// the ranges that execute inside a pass are counted. Under QueueControl::Inline, a range begins the frame's pass.
// Returns true when the range was pushed. The caller then pops it in the same command buffer.
bool PushRange(const QueueTarget& queue, void* commandBuffer, const char* queueName, const char* rangeName);
// False when no range is open in that command buffer (e.g. it was opened in another one).
bool PopRange(rhi::Backend backend, void* commandBuffer);

// A render graph's per-pass hooks (PassExecutionContext::beginGpuPassRange / endGpuPassRange).
void BeginPassRange(rhi::Backend backend, rhi::Device device, rhi::CommandList commandList, rhi::Queue queue, const char* queueName, const char* passName);
void EndPassRange(rhi::Backend backend, rhi::CommandList commandList, rhi::Queue queue);
}  // namespace org::telemetry::nvperf
