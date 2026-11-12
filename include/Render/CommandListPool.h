#pragma once

#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <cstdint>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <rhi.h>
#include "Render/Runtime/ITaskService.h"


namespace org {

// Command lists share command allocators. Each shared allocator is a shard. On Vulkan a shard is one VkCommandPool.
// Every pool costs driver memory as soon as one of its buffers is recorded (about 2 MB of VRAM each on AMD). Sharing
// keeps the number of pools near the number of lists that record at the same time, instead of the number of lists.
//
// The threading rule. Vulkan requires external synchronization of a pool for every reset, begin, recorded command, end
// and free of any buffer in it. So:
// - A thread touches a list's commands only while the list's pair holds the claim on the list's shard. A shard has one
//   claim.
// - The claim is taken when the list is begun (CommandListPool::Begin, or Request, which begins). The claim travels with
//   the pair. A list begun on one thread may be recorded on another, when the hand-over orders the two.
// - The claim is released when the list is closed (CommandListPair::EndRecording). Recycling, discarding or destroying
//   the pair also releases it.
// - Taking a claim never waits. Begin takes the list's own shard when it is free. Otherwise it takes another free shard,
//   or makes a new one. So the shard count follows the peak number of open lists.
// - A list is freed only by a thread that holds its shard's claim. A list dropped without it goes to its shard's
//   orphans. The next claimant of the shard frees it.
// - A shared allocator is never reset as a whole. Each list is reset on its own when it is begun. So a kept list, or one
//   the GPU still runs, stays valid while other lists of its shard are reset. Trim runs only under a claim.
// - Submitting a closed list, or executing a closed secondary from another list, needs no claim.
//
// D3D12 allocators cannot share lists (rhi::CommandAllocator::SupportsSharedLists). There each list keeps a shard of its
// own, and Begin resets the whole allocator, as before.
class CommandAllocatorShard {
public:
    CommandAllocatorShard(rhi::CommandAllocatorPtr allocator, bool shared) noexcept
        : m_allocator(std::move(allocator)), m_shared(shared) {}
    ~CommandAllocatorShard();
    CommandAllocatorShard(const CommandAllocatorShard&) = delete;
    CommandAllocatorShard& operator=(const CommandAllocatorShard&) = delete;

    const rhi::CommandAllocator& Allocator() const noexcept { return m_allocator.Get(); }
    // Only the claimant uses the mutable allocator (reset, trim).
    rhi::CommandAllocator& Allocator() noexcept { return m_allocator.Get(); }
    // True when many lists share this allocator. False on D3D12, where each list has its own.
    bool Shared() const noexcept { return m_shared; }

    bool TryClaim() noexcept {
        bool expected = false;
        return m_claimed.compare_exchange_strong(expected, true, std::memory_order_acquire, std::memory_order_relaxed);
    }
    void Release() noexcept { m_claimed.store(false, std::memory_order_release); }
    bool Claimed() const noexcept { return m_claimed.load(std::memory_order_acquire); }

    // Frees the list now if the shard is free. Otherwise the shard's next claimant frees it.
    void Orphan(rhi::CommandListPtr&& list) noexcept;
    // Frees the orphaned lists. Only the claimant calls this.
    void DestroyOrphans() noexcept;
    bool HasOrphans() const noexcept { return m_orphanCount.load(std::memory_order_acquire) != 0; }

    // Lists alive in this shard, orphans included. A shard with none can be dropped.
    std::atomic<uint32_t> lists{0};

private:
    rhi::CommandAllocatorPtr m_allocator;
    bool m_shared = false;
    std::atomic<bool> m_claimed{false};
    std::mutex m_orphanMutex;
    std::vector<rhi::CommandListPtr> m_orphans;
    std::atomic<size_t> m_orphanCount{0};
};

// A command list and the allocator it records from. A pool's pair names the list's shard. A pair made outside a pool
// owns its allocator instead.
struct CommandListPair {
    rhi::CommandAllocatorPtr allocator;  // only for a pair made outside a pool
    rhi::CommandListPtr list;
    std::shared_ptr<CommandAllocatorShard> shard;  // only for a pool's pair

    CommandListPair() = default;
    CommandListPair(CommandListPair&& other) noexcept;
    CommandListPair& operator=(CommandListPair&& other) noexcept;
    CommandListPair(const CommandListPair&) = delete;
    CommandListPair& operator=(const CommandListPair&) = delete;
    ~CommandListPair() { ReleaseList(); }

    // True between the pool's Begin and EndRecording. The pair then holds its shard's claim.
    bool Recording() const noexcept { return m_claimed; }
    // Call after the list is closed. Gives up the shard's claim, so other lists of the shard can be begun.
    void EndRecording() noexcept {
        if (m_claimed && shard) shard->Release();
        m_claimed = false;
    }

private:
    friend class CommandListPool;
    void ReleaseList() noexcept;
    bool m_claimed = false;
};

class CommandListPool {
public:
    struct Diagnostics {
        size_t lastRequestedCount = 0;
        size_t availableCount = 0;
        size_t checkedOutCount = 0;
        size_t inFlightCount = 0;
        size_t preparedDeficit = 0;
        size_t backgroundResetPendingCount = 0;  // always 0. Lists are reset when they are begun.
        size_t totalOwnedCount = 0;
        size_t readyTargetCount = 0;
        size_t capacityTargetCount = 0;
        size_t shardCount = 0;
        uint64_t acquisitionWaitCount = 0;
        uint64_t acquisitionWaitDurationNs = 0;
        uint64_t inlineCreateFallbackCount = 0;
        uint64_t startupStallCount = 0;
        uint64_t growthStallCount = 0;
        uint64_t backgroundCreatedCount = 0;
        uint64_t backgroundResetCount = 0;  // idle lists the background worker freed
        uint64_t acquiredCount = 0;
        uint64_t recycleQueuedCount = 0;
    };

    // secondary: the pool makes secondary lists (Device::CreateSecondaryCommandList). Begin resets them but does not begin
    // them. The recorder begins each one with CommandList::BeginSecondary. External epochs use these
    // (ExternalEpochExecution).
    CommandListPool(rhi::Device& device, rhi::QueueKind type, bool secondary = false);
    ~CommandListPool();
    void ShutdownBackgroundReset();

    // A pair begun on the calling thread. It holds its shard's claim (see the threading rule above). Call
    // CommandListPair::EndRecording after closing the list.
    CommandListPair Request();
    // Pairs for one recording generation, not begun. The recorder calls Begin on each one, then EndRecording after
    // closing it. The background worker then tops up the idle lists for the next generation. Lists are made on the
    // calling thread only at startup, when demand grows, or when no worker is available.
    std::vector<CommandListPair> AcquireBatch(size_t count, uint64_t completedFenceValue);

    // Resets the pair's list, begins it and takes its shard's claim. A secondary list is reset but not begun. The GPU
    // must be done with the list. A pair that already holds its claim is reset again under it. An empty pair gets a new
    // list. The pair's list may be swapped for another idle list of the pool, so read pair.list only after this call.
    void Begin(CommandListPair& pair);

    // Publish known future demand without waiting. Compilation/planning may use
    // this as soon as a recording layout is available.
    void PublishDemand(size_t requiredCount, uint64_t completedFenceValue);

    // Recycle a pair after execution. If fenceValue is 0 the pair is idle at once. Otherwise it becomes idle once
    // RecycleCompleted is called with a sufficiently large fence value.
    void Recycle(CommandListPair&& pair, uint64_t fenceValue);

    // The frame's ownership already proves that the GPU is done with the pair. It is idle at once. Its list is reset
    // when it is next begun. No work is scheduled from the retirement thread.
    void RecycleForNextRequest(CommandListPair&& pair);
    // Drops a pair whose list is in an unknown state. The list is freed under its shard's claim.
    void Discard(CommandListPair&& pair) noexcept;

    // Make any completed command lists idle.
    void RecycleCompleted(uint64_t completedFenceValue);

    bool Secondary() const noexcept { return m_secondary; }

    Diagnostics GetDiagnostics() const {
        std::lock_guard lock(m_mutex);
        return m_diagnostics;
    }

private:
    // Takes the claim on a free shard, or makes a new claimed shard. Never waits.
    std::shared_ptr<CommandAllocatorShard> ClaimAnyShard();
    std::shared_ptr<CommandAllocatorShard> CreateShard();
    // Makes a list in a shard the caller has claimed. A primary list is closed, so Begin can reset it.
    rhi::CommandListPtr CreateList(CommandAllocatorShard& shard);
    // Resets the pair's list under the claim the pair holds.
    void ResetClaimed(CommandListPair& pair);
    // Makes idle lists, for the background worker or the inline fallback.
    std::vector<CommandListPair> CreateIdlePairs(size_t count);
    // Frees idle lists over the targets, frees orphans and trims the shards. Then drops shards with no lists.
    void ShedIdle();
    bool SheddableLocked() const noexcept;
    void BackgroundResetDrain();
    bool ScheduleBackgroundReset();
    void UpdateDiagnosticsCountsLocked();
    size_t TotalOwnedLocked() const noexcept;

    rhi::Device m_device;
    rhi::QueueKind m_type;
    bool m_secondary = false;
    std::atomic<uint64_t> m_nextDebugNameId{ 1 };
    Diagnostics m_diagnostics{};

    mutable std::mutex m_mutex;
    std::condition_variable m_availableReady;
    std::shared_ptr<org::runtime::ITaskService> m_taskService;
    std::shared_ptr<org::runtime::ITaskScope> m_taskScope;
    std::atomic<bool> m_backgroundResetScheduled{false};
    bool m_stopBackgroundReset = false;
    size_t m_backgroundCreateActiveCount = 0;
    size_t m_checkedOutCount = 0;
    size_t m_highWaterRequestedCount = 0;
    size_t m_readyTargetCount = 0;
    size_t m_capacityTargetCount = 0;
    // Whether the device's allocators share lists. Unknown (-1) until the first shard is made.
    int m_sharedLists = -1;
    std::exception_ptr m_backgroundFailure;

    std::string m_telemetryPrefix;

    // The shards, guarded by m_mutex. Declared before the lists so that the lists are destroyed first.
    std::vector<std::shared_ptr<CommandAllocatorShard>> m_shards;
    // Idle lists. The GPU is done with them. They are reset when begun.
    std::vector<CommandListPair> m_available;
    std::deque<std::pair<uint64_t, CommandListPair>> m_inFlight;
};


} // namespace org
