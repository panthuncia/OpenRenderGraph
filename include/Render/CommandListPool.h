#pragma once

#include <cstddef>
#include <deque>
#include <mutex>
#include <vector>
#include <cstdint>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <rhi.h>
#include "Render/Runtime/ITaskService.h"


namespace org {

struct CommandListPair {
    rhi::CommandAllocatorPtr allocator;
    rhi::CommandListPtr list;
};

class CommandListPool {
public:
    struct Diagnostics {
        size_t lastRequestedCount = 0;
        size_t availableCount = 0;
        size_t checkedOutCount = 0;
        size_t inFlightCount = 0;
        size_t preparedDeficit = 0;
        size_t backgroundResetPendingCount = 0;
        size_t totalOwnedCount = 0;
        size_t readyTargetCount = 0;
        size_t capacityTargetCount = 0;
        uint64_t acquisitionWaitCount = 0;
        uint64_t acquisitionWaitDurationNs = 0;
        uint64_t inlineCreateFallbackCount = 0;
        uint64_t startupStallCount = 0;
        uint64_t growthStallCount = 0;
        uint64_t backgroundCreatedCount = 0;
        uint64_t backgroundResetCount = 0;
        uint64_t acquiredCount = 0;
        uint64_t recycleQueuedCount = 0;
    };

    CommandListPool(rhi::Device& device, rhi::QueueKind type);
    ~CommandListPool();
    void ShutdownBackgroundReset();

    // Acquire a command allocator / list pair ready for recording.
    CommandListPair Request();
    // Acquire a complete recording generation, then asynchronously replenish
    // the ready reserve for the next frame. Only startup, demand growth, or an
    // unavailable preparation worker may create or wait on the calling thread.
    std::vector<CommandListPair> AcquireBatch(size_t count, uint64_t completedFenceValue);

    // Publish known future demand without waiting. Compilation/planning may use
    // this as soon as a recording layout is available.
    void PublishDemand(size_t requiredCount, uint64_t completedFenceValue);

    // Recycle a pair after execution. If fenceValue is 0 the pair is queued for
    // background reset immediately. Otherwise it will be queued once
    // RecycleCompleted is called with a sufficiently large fence value.
    void Recycle(CommandListPair&& pair, uint64_t fenceValue);

    // Frame ownership already proves GPU completion. Reset on the next
    // recording worker, without scheduling work from the retirement thread.
    void RecycleForNextRequest(CommandListPair&& pair);
    void Discard(CommandListPair&& pair) noexcept;

    // Queue any completed command lists for background reset.
    void RecycleCompleted(uint64_t completedFenceValue);

    Diagnostics GetDiagnostics() const {
        std::lock_guard lock(m_mutex);
        return m_diagnostics;
    }

private:
    void PreparePairForReuse(CommandListPair& pair);
    CommandListPair CreateReadyPair();
    void BackgroundResetDrain();
    bool ScheduleBackgroundReset();
    void UpdateDiagnosticsCountsLocked();

    rhi::Device m_device;
    rhi::QueueKind m_type;
    std::atomic<uint64_t> m_nextDebugNameId{ 1 };
    Diagnostics m_diagnostics{};

    mutable std::mutex m_mutex;
    std::condition_variable m_availableReady;
    std::shared_ptr<org::runtime::ITaskService> m_taskService;
    std::shared_ptr<org::runtime::ITaskScope> m_taskScope;
    std::atomic<bool> m_backgroundResetScheduled{false};
    bool m_stopBackgroundReset = false;
    size_t m_backgroundResetActiveCount = 0;
    size_t m_backgroundCreateActiveCount = 0;
    size_t m_checkedOutCount = 0;
    size_t m_highWaterRequestedCount = 0;
    size_t m_readyTargetCount = 0;
    size_t m_capacityTargetCount = 0;
    std::exception_ptr m_backgroundFailure;

    std::string m_telemetryPrefix;

    std::vector<CommandListPair> m_available;
    std::deque<std::pair<uint64_t, CommandListPair>> m_inFlight;
    std::vector<CommandListPair> m_pendingBackgroundReset;
};


} // namespace org
