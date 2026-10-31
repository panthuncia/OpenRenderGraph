#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <rhi.h>

#include "Render/Runtime/StatisticsTypes.h"

namespace org::runtime {

enum class TimestampRangeKind : uint8_t {
    Pass,
    Barrier,  // a barrier batch recorded around a pass (OwnedRecordingBarrierRangeName's "cs.org.barrier.<where>.<pass>")
    Sub,      // a range a pass recorded inside itself (BeginRecordingSubRange), nested in its pass's
};

// The timestamp queries of one recording that is kept for resubmission (CreateOwnedTimestamps).
// A kept recording replays the same commands at every submission. So it cannot use the frame-indexed slots of the
// shared pool. That pool is also recreated as passes register, while recordings can outlive it. Instead, a kept
// recording owns its pool and readback buffer. Its list does the following:
// - resets the queries,
// - writes a begin/end pair per range,
// - resolves them into the buffer.
// A recording belongs to one frame slot. Each submission on that slot starts only after the previous one completed.
// So when the slot completes, the buffer holds the last submission's values. OnFrameComplete reads them then.
// The queries are recorded once, on the recording's thread. They are read after submissions, by the completion.
struct OwnedTimestampQueries {
    struct Entry {
        std::string name;
        int passIndex = -1;  // index into GetPassNames(), or -1 for a barrier batch
        TimestampRangeKind kind = TimestampRangeKind::Pass;
        uint32_t occurrence = 0;  // number of ranges with this name before it in the list
    };
    rhi::QueryPoolPtr pool;
    rhi::ResourcePtr readback;
    rhi::QueueKind queueKind = rhi::QueueKind::Graphics;
    double ticksToMilliseconds = 0.0;  // the queue's timestamp calibration
    uint64_t stride = 0;               // bytes per query result
    uint32_t capacity = 0;             // ranges; each takes queries 2i (begin) and 2i + 1 (end)
    std::vector<Entry> entries;        // the ranges recorded, in recording order
    std::vector<int> open;             // recording: the ranges begun and not yet ended, innermost last
    bool resolved = false;             // recording: the resolve is recorded, and nothing more can be
    uint32_t overflow = 0;             // ranges past capacity, not timed

    OwnedTimestampQueries() = default;
    OwnedTimestampQueries(const OwnedTimestampQueries&) = delete;
    OwnedTimestampQueries& operator=(const OwnedTimestampQueries&) = delete;
    ~OwnedTimestampQueries();  // the pool and buffer go to deferred deletion
};

// One timestamp pair read back by OnFrameComplete. It is one of:
// - a pass's pair, from the shared pool or from a kept recording's own queries,
// - a barrier batch's pair, from kept recordings only.
// Pairs come from every queue. A consumer that computes time along a queue must keep the queues apart.
struct CompletedTimestamp {
    std::string_view name;  // valid until the next OnFrameComplete or RegisterPass
    int passIndex = -1;     // index into GetPassNames(), or -1
    uint32_t occurrence = 0;
    rhi::QueueKind queueKind = rhi::QueueKind::Graphics;
    TimestampRangeKind kind = TimestampRangeKind::Pass;
    bool replayed = false;  // from a kept recording's own queries
    uint64_t beginTick = 0, endTick = 0;  // the queue's ticks (comparable within one queue)
    double milliseconds = 0.0;            // end - begin at the queue's calibration
};

class IStatisticsService {
public:
    virtual ~IStatisticsService() = default;

    virtual void Initialize() = 0;
    virtual void BeginFrame() = 0;
    virtual void ClearAll() = 0;

    virtual unsigned RegisterPass(const std::string& passName, bool isGeometryPass, std::string_view techniquePath = {}) = 0;
    virtual void RegisterQueue(rhi::QueueKind queueKind) = 0;
    virtual void SetupQueryHeap() = 0;

    virtual void BeginQuery(unsigned passIndex, unsigned frameIndex, rhi::Queue& queue, rhi::CommandList& cmdList) = 0;
    virtual void EndQuery(unsigned passIndex, unsigned frameIndex, rhi::Queue& queue, rhi::CommandList& cmdList) = 0;
    virtual void ResolveQueries(unsigned frameIndex, rhi::Queue& queue, rhi::CommandList& cmdList) = 0;
    virtual void OnFrameComplete(unsigned frameIndex, rhi::Queue& queue) = 0;
    virtual void RecordCpuUpdateTime(unsigned passIndex, double milliseconds) = 0;
    virtual void RecordCpuExecuteTime(unsigned passIndex, double milliseconds) = 0;

    // Thread-safe overloads that write to a per-task context instead of shared state.
    virtual void BeginQuery(unsigned passIndex, unsigned frameIndex, rhi::Queue& queue, rhi::CommandList& cmdList, QueryRecordingContext& ctx) = 0;
    virtual void EndQuery(unsigned passIndex, unsigned frameIndex, rhi::Queue& queue, rhi::CommandList& cmdList, QueryRecordingContext& ctx) = 0;
    virtual void ResolveQueries(unsigned frameIndex, rhi::Queue& queue, rhi::CommandList& cmdList, QueryRecordingContext& ctx) = 0;
    virtual void MergePendingResolves(rhi::QueueKind queueKind, unsigned frameIndex, QueryRecordingContext& ctx) = 0;

    virtual const std::vector<std::string>& GetPassNames() const = 0;
    virtual const std::vector<std::string>& GetPassTechniquePaths() const = 0;
    virtual const std::vector<PassStats>& GetPassStats() const = 0;
    virtual const std::vector<MeshPipelineStats>& GetMeshStats() const = 0;
    virtual MemoryBudgetStats GetMemoryBudgetStats() const = 0;
    virtual const std::vector<bool>& GetIsGeometryPassVector() const = 0;
    virtual const std::vector<unsigned>& GetVisiblePassIndices(uint64_t maxStaleFrames) const = 0;
    // Passes read back by the latest OnFrameComplete carry this value as their gpuSampleSerial.
    virtual uint64_t GetFrameSerial() const = 0;
    virtual double GetGpuTicksToMilliseconds() const = 0;

    // Creates the queries of a recording that is kept for resubmission, for this many ranges on that queue
    // (QueryRecordingContext::owned). Then:
    // - the context's BeginQuery, EndQuery and ResolveQueries use them,
    // - BeginRangeQuery and EndRangeQuery time the recording's barrier batches,
    // - each MergePendingResolves (a submission) makes the frame slot's OnFrameComplete read them.
    // Returns null when they are not collected.
    virtual std::shared_ptr<OwnedTimestampQueries> CreateOwnedTimestamps(rhi::QueueKind queueKind, uint32_t ranges) {
        (void)queueKind; (void)ranges;
        return nullptr;
    }
    // A named range (a barrier batch) in a context with owned queries; no-op without.
    virtual void BeginRangeQuery(std::string_view name, rhi::CommandList& cmdList, QueryRecordingContext& ctx) {
        (void)name; (void)cmdList; (void)ctx;
    }
    virtual void EndRangeQuery(rhi::CommandList& cmdList, QueryRecordingContext& ctx) { (void)cmdList; (void)ctx; }
    // A named range that a pass records inside itself, nested in the pass's range, in a context with owned queries.
    // Returns false when the range is not timed: there are no owned queries, or they are past capacity.
    // End only a range that returned true.
    virtual bool BeginSubRangeQuery(std::string_view name, rhi::CommandList& cmdList, QueryRecordingContext& ctx) {
        (void)name; (void)cmdList; (void)ctx;
        return false;
    }
    // Queues a publication for the frame slot's next OnFrameComplete. It runs before that call reads the slot's queries,
    // on the thread that calls both. This is for MergePendingResolves and the CPU samples of a recording made after its
    // submission, such as a late batch of an external epoch.
    virtual void PublishAtCompletion(unsigned frameIndex, std::function<void()> publish) { (void)frameIndex; (void)publish; }
    // Every timestamp pair the latest OnFrameComplete read, in reading order.
    virtual const std::vector<CompletedTimestamp>& GetCompletedTimestamps() const {
        static const std::vector<CompletedTimestamp> none;
        return none;
    }
};

std::shared_ptr<IStatisticsService> CreateDefaultStatisticsService();

// The queries of the kept recording that a pass is being recorded into. The persistent graph's recorder sets them
// around the pass's Record (RecordingRangeScope), on the recording thread, while timestamps are collected for it.
// A pass times parts of itself through BeginRecordingSubRange and EndRecordingSubRange, for example a draw pass's
// views or draw groups. The pass needs no access of its own to the recording's statistics.
struct RecordingRanges {
    IStatisticsService* service = nullptr;
    QueryRecordingContext* context = nullptr;
    rhi::CommandList* commands = nullptr;
};
inline thread_local RecordingRanges* t_recordingRanges = nullptr;

class RecordingRangeScope {
public:
    explicit RecordingRangeScope(RecordingRanges* ranges) : m_previous(std::exchange(t_recordingRanges, ranges)) {}
    ~RecordingRangeScope() { t_recordingRanges = m_previous; }
    RecordingRangeScope(const RecordingRangeScope&) = delete;
    RecordingRangeScope& operator=(const RecordingRangeScope&) = delete;

private:
    RecordingRanges* m_previous;
};

// Whether the pass being recorded on this thread can time sub-ranges.
inline bool RecordingSubRangesActive() noexcept { return t_recordingRanges != nullptr; }
// Begins a sub-range of the pass being recorded on this thread, in its own command list. Returns false when the
// range is not timed. End only a range that returned true.
inline bool BeginRecordingSubRange(std::string_view name) {
    auto* ranges = t_recordingRanges;
    return ranges && ranges->service->BeginSubRangeQuery(name, *ranges->commands, *ranges->context);
}
inline void EndRecordingSubRange() {
    if (auto* ranges = t_recordingRanges)
        ranges->service->EndRangeQuery(*ranges->commands, *ranges->context);
}

}
