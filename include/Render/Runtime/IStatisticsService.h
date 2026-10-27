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

// The timestamp queries of one recording kept for resubmission (CreateOwnedTimestamps). A kept recording replays the same
// commands every submission, so it cannot take the frame-indexed slots of the shared pool (which is also recreated as
// passes register, under recordings that outlive it): it owns its pool and readback buffer instead. Its list resets them,
// writes a begin/end pair per range and resolves them into the buffer; a recording belongs to one frame slot, whose
// submissions follow each other's completion, so the buffer holds the last submission's values when that slot completes
// (OnFrameComplete reads it then). Recorded once, on the recording's thread; read after submissions, by the completion.
struct OwnedTimestampQueries {
    struct Entry {
        std::string name;
        int passIndex = -1;  // GetPassNames()'s, or -1 (a barrier batch)
        TimestampRangeKind kind = TimestampRangeKind::Pass;
        uint32_t occurrence = 0;  // ranges of this name before it in the list
    };
    rhi::QueryPoolPtr pool;
    rhi::ResourcePtr readback;
    rhi::QueueKind queueKind = rhi::QueueKind::Graphics;
    double ticksToMilliseconds = 0.0;  // the queue's timestamp calibration
    uint64_t stride = 0;               // bytes per query result
    uint32_t capacity = 0;             // ranges; each takes queries 2i (begin) and 2i + 1 (end)
    std::vector<Entry> entries;        // the ranges recorded, in recording order
    std::vector<int> open;             // recording: the ranges begun and not yet ended, innermost last
    bool resolved = false;             // recording: the resolve is recorded; nothing more is
    uint32_t overflow = 0;             // ranges past capacity, not timed

    OwnedTimestampQueries() = default;
    OwnedTimestampQueries(const OwnedTimestampQueries&) = delete;
    OwnedTimestampQueries& operator=(const OwnedTimestampQueries&) = delete;
    ~OwnedTimestampQueries();  // the pool and buffer go to deferred deletion
};

// One timestamp pair read back by OnFrameComplete: a pass's (from the shared pool or a kept recording's own queries), or a
// barrier batch's (kept recordings only). Every queue's: a consumer computing time along a queue keeps them apart.
struct CompletedTimestamp {
    std::string_view name;  // valid until the next OnFrameComplete or RegisterPass
    int passIndex = -1;     // GetPassNames()'s, or -1
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

    // Queries of a recording kept for resubmission, for this many ranges on that queue (QueryRecordingContext::owned): the
    // context's BeginQuery/EndQuery/ResolveQueries then use them, BeginRangeQuery/EndRangeQuery time its barrier batches,
    // and each MergePendingResolves (a submission) has the frame slot's OnFrameComplete read them. Null: not collected.
    virtual std::shared_ptr<OwnedTimestampQueries> CreateOwnedTimestamps(rhi::QueueKind queueKind, uint32_t ranges) {
        (void)queueKind; (void)ranges;
        return nullptr;
    }
    // A named range (a barrier batch) in a context with owned queries; no-op without.
    virtual void BeginRangeQuery(std::string_view name, rhi::CommandList& cmdList, QueryRecordingContext& ctx) {
        (void)name; (void)cmdList; (void)ctx;
    }
    virtual void EndRangeQuery(rhi::CommandList& cmdList, QueryRecordingContext& ctx) { (void)cmdList; (void)ctx; }
    // A named range a pass records inside itself (nested in the pass's), in a context with owned queries. False: not timed
    // (no owned queries, or past their capacity); only a true one is ended.
    virtual bool BeginSubRangeQuery(std::string_view name, rhi::CommandList& cmdList, QueryRecordingContext& ctx) {
        (void)name; (void)cmdList; (void)ctx;
        return false;
    }
    // A publication (MergePendingResolves and the CPU samples of a recording made after its submission: a late batch of an external
    // epoch) run by the frame slot's next OnFrameComplete, before it reads the slot's queries, on the thread that calls both.
    virtual void PublishAtCompletion(unsigned frameIndex, std::function<void()> publish) { (void)frameIndex; (void)publish; }
    // Every timestamp pair the latest OnFrameComplete read, in reading order.
    virtual const std::vector<CompletedTimestamp>& GetCompletedTimestamps() const {
        static const std::vector<CompletedTimestamp> none;
        return none;
    }
};

std::shared_ptr<IStatisticsService> CreateDefaultStatisticsService();

// The queries of the kept recording a pass is being recorded into: set around the pass's Record by the persistent graph's
// recorder (RecordingRangeScope), on the recording thread, while timestamps are collected for it. A pass times parts of itself
// through BeginRecordingSubRange / EndRecordingSubRange (e.g. a draw pass's views or draw groups), with no access to the
// recording's statistics of its own.
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
// A sub-range of the pass being recorded on this thread, into its own command list. False: not timed; end only a true one.
inline bool BeginRecordingSubRange(std::string_view name) {
    auto* ranges = t_recordingRanges;
    return ranges && ranges->service->BeginSubRangeQuery(name, *ranges->commands, *ranges->context);
}
inline void EndRecordingSubRange() {
    if (auto* ranges = t_recordingRanges)
        ranges->service->EndRangeQuery(*ranges->commands, *ranges->context);
}

}
