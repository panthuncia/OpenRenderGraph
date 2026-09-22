#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace org::runtime {

struct PassStats {
    double gpuTimeEma = 0.0;
    double gpuTimeMs = 0.0;
    uint64_t gpuSampleSerial = 0;
    // The raw timestamp pair of the last sample, in GPU ticks (IStatisticsService::GetGpuTicksToMilliseconds).
    // A pass's begin is written after its entry barriers and can precede the previous pass's end, so
    // per-pass durations overlap; the pair lets a consumer compute exclusive time along the queue.
    uint64_t gpuBeginTick = 0;
    uint64_t gpuEndTick = 0;
    double cpuUpdateTimeEma = 0.0;
    double cpuExecuteTimeEma = 0.0;
    static constexpr double alpha = 0.1;

    double GetCpuTimeEma() const noexcept {
        return cpuUpdateTimeEma + cpuExecuteTimeEma;
    }
};

struct MeshPipelineStats {
    double invocationsEma = 0.0;
    double primitivesEma = 0.0;
};

struct MemoryBudgetStats {
    uint64_t usageBytes = 0;
    uint64_t budgetBytes = 0;
    uint64_t sampleFrameSerial = 0;
    bool valid = false;
};

/// Per-task context for thread-safe query recording.
/// Each parallel recording task owns one of these; the main thread
/// merges them after all tasks complete.
struct QueryRecordingContext {
    std::vector<uint32_t> recordedIndices;
    std::vector<std::pair<uint32_t, uint32_t>> pendingRanges;
};

}
