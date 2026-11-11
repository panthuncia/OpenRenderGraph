#pragma once

#include <cstdint>
#include <memory>
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
    // ORG's allocator, split into device-local (local) and host (nonLocal) segments. For each segment: the memory in its
    // blocks, and the part of that memory that allocations use. Block bytes beyond allocation bytes are its slack.
    uint64_t localBlockBytes = 0;
    uint64_t localAllocationBytes = 0;
    uint32_t localBlockCount = 0;
    uint32_t localAllocationCount = 0;
    uint64_t nonLocalBlockBytes = 0;
    uint64_t nonLocalAllocationBytes = 0;
};

struct OwnedTimestampQueries;

/// Per-task context for thread-safe query recording.
/// Each parallel recording task owns one of these; the main thread
/// merges them after all tasks complete.
struct QueryRecordingContext {
    std::vector<uint32_t> recordedIndices;
    std::vector<std::pair<uint32_t, uint32_t>> pendingRanges;
    // For a recording kept for resubmission: its own queries (IStatisticsService::CreateOwnedTimestamps). They replace
    // the frame-indexed shared queries, which the service recreates as passes register. They are kept across merges,
    // so every submission of the recording publishes them again.
    std::shared_ptr<OwnedTimestampQueries> owned;
    // The set of shared query pools and readback buffers this context's list writes: the one current at its first
    // query. The service replaces the set when passes register, and destroys a replaced one only after every context
    // holding it is released. A recording holds its context until its last submission completes, so a list in flight
    // never names a destroyed pool. A list whose set was replaced while it recorded writes and resolves no more queries.
    std::shared_ptr<const void> sharedQueries;
};

}
