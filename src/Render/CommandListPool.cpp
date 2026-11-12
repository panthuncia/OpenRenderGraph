#include "Render/CommandListPool.h"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <optional>
#include <string>
#include <stdexcept>
#include <utility>
#include <spdlog/spdlog.h>
#include <BasicTelemetry/Tracy.h>
#include "Render/Runtime/TaskServiceAccess.h"


namespace org {

namespace {
    // Idle lists kept beyond the high water of one generation's demand. The rest of the capacity grows on demand, to
    // cover the lists still in flight.
    constexpr size_t kWarmSlackCommandLists = 2;

    const char* QueueKindDebugName(rhi::QueueKind type) noexcept {
        switch (type) {
        case rhi::QueueKind::Graphics: return "Graphics";
        case rhi::QueueKind::Compute: return "Compute";
        case rhi::QueueKind::Copy: return "Copy";
        default: return "Unknown";
        }
    }
}

// ---------------------------------------------------------------------------
// Shards and pairs

CommandAllocatorShard::~CommandAllocatorShard() {
    // Nothing else holds the shard now. Free its lists before its allocator.
    m_orphans.clear();
}

void CommandAllocatorShard::Orphan(rhi::CommandListPtr&& list) noexcept {
    if (!list) return;
    if (TryClaim()) {
        list.Reset();
        lists.fetch_sub(1, std::memory_order_relaxed);
        DestroyOrphans();
        Release();
        return;
    }
    try {
        std::lock_guard lock(m_orphanMutex);
        m_orphans.push_back(std::move(list));
        m_orphanCount.store(m_orphans.size(), std::memory_order_release);
    } catch (...) {
        // Out of memory. Freeing the list here would race its shard's claimant, so leak it instead.
        (void)list.Release();
        lists.fetch_sub(1, std::memory_order_relaxed);
    }
}

void CommandAllocatorShard::DestroyOrphans() noexcept {
    std::vector<rhi::CommandListPtr> dying;
    {
        std::lock_guard lock(m_orphanMutex);
        dying.swap(m_orphans);
        m_orphanCount.store(0, std::memory_order_release);
    }
    lists.fetch_sub(static_cast<uint32_t>(dying.size()), std::memory_order_relaxed);
    dying.clear();
}

CommandListPair::CommandListPair(CommandListPair&& other) noexcept
    : allocator(std::move(other.allocator)), list(std::move(other.list)), shard(std::move(other.shard)),
      m_claimed(std::exchange(other.m_claimed, false)) {
}

CommandListPair& CommandListPair::operator=(CommandListPair&& other) noexcept {
    if (this == &other) return *this;
    ReleaseList();
    allocator = std::move(other.allocator);
    list = std::move(other.list);
    shard = std::move(other.shard);
    m_claimed = std::exchange(other.m_claimed, false);
    return *this;
}

void CommandListPair::ReleaseList() noexcept {
    EndRecording();
    if (shard) {
        // The list is freed under its shard's claim (now, or by the shard's next claimant).
        shard->Orphan(std::move(list));
        shard.reset();
    }
    // A pair made outside a pool frees its list before its allocator.
    list.Reset();
    allocator.Reset();
}

// ---------------------------------------------------------------------------
// Pool

CommandListPool::CommandListPool(rhi::Device& device, rhi::QueueKind type, bool secondary)
    : m_device(device), m_type(type), m_secondary(secondary),
      m_telemetryPrefix(std::string("ORG.CommandListPool.") + QueueKindDebugName(type) + (secondary ? ".Secondary." : ".")) {
}

CommandListPool::~CommandListPool() {
    ShutdownBackgroundReset();
}

void CommandListPool::ShutdownBackgroundReset() {
    std::shared_ptr<org::runtime::ITaskScope> scope;
    {
        std::lock_guard lock(m_mutex);
        m_stopBackgroundReset = true;
        scope = std::move(m_taskScope);
    }
    m_availableReady.notify_all();
    if (scope) scope->CancelAndWait();
    m_backgroundResetScheduled.store(false, std::memory_order_release);
    std::lock_guard lock(m_mutex);
    m_taskService.reset();
}

size_t CommandListPool::TotalOwnedLocked() const noexcept {
    return m_available.size() + m_checkedOutCount + m_inFlight.size() + m_backgroundCreateActiveCount;
}

void CommandListPool::UpdateDiagnosticsCountsLocked() {
    m_diagnostics.availableCount = m_available.size();
    m_diagnostics.checkedOutCount = m_checkedOutCount;
    m_diagnostics.inFlightCount = m_inFlight.size();
    m_diagnostics.backgroundResetPendingCount = 0;
    m_diagnostics.totalOwnedCount = TotalOwnedLocked();
    m_diagnostics.readyTargetCount = m_readyTargetCount;
    m_diagnostics.capacityTargetCount = m_capacityTargetCount;
    m_diagnostics.shardCount = m_shards.size();
    const size_t readyOrPreparing = m_available.size() + m_backgroundCreateActiveCount;
    m_diagnostics.preparedDeficit = m_readyTargetCount > readyOrPreparing
        ? m_readyTargetCount - readyOrPreparing : 0;
}

std::shared_ptr<CommandAllocatorShard> CommandListPool::CreateShard() {
    BT_ZONE_SCOPE("CommandListPool::CreateShard");
    if (!m_device) throw std::runtime_error("CommandListPool lost its RHI device");
    rhi::CommandAllocatorPtr allocator;
    const auto result = m_device.CreateCommandAllocator(m_type, allocator);
    if (!rhi::IsOk(result) || !allocator) {
        throw std::runtime_error(std::string("Failed to create ORG command allocator: ") + rhi::ResultName(result));
    }
    const bool shared = allocator->SupportsSharedLists();
    auto shard = std::make_shared<CommandAllocatorShard>(std::move(allocator), shared);
    shard->TryClaim();  // a new shard is free, so this always succeeds
    size_t count = 0;
    {
        std::lock_guard lock(m_mutex);
        m_shards.push_back(shard);
        m_sharedLists = shared ? 1 : 0;
        count = m_shards.size();
        UpdateDiagnosticsCountsLocked();
    }
    basic_telemetry::AddCounter(m_telemetryPrefix + "ShardsCreated");
    basic_telemetry::SetGauge(m_telemetryPrefix + "Shards", static_cast<int64_t>(count));
    return shard;
}

std::shared_ptr<CommandAllocatorShard> CommandListPool::ClaimAnyShard() {
    {
        std::lock_guard lock(m_mutex);
        // An allocator that cannot share lists gets one list only, so every new list needs a new shard.
        if (m_sharedLists == 1) {
            for (const auto& shard : m_shards)
                if (shard->TryClaim()) return shard;
        }
    }
    return CreateShard();
}

rhi::CommandListPtr CommandListPool::CreateList(CommandAllocatorShard& shard) {
    BT_ZONE_SCOPE("CommandListPool::CreateList");
    const auto createBegin = std::chrono::steady_clock::now();
    rhi::CommandListPtr list;
    const auto result = m_secondary ? m_device.CreateSecondaryCommandList(m_type, shard.Allocator(), list)
                                    : m_device.CreateCommandList(m_type, shard.Allocator(), list);
    if (!rhi::IsOk(result) || !list) {
        throw std::runtime_error(std::string("Failed to create ORG ") + (m_secondary ? "secondary " : "") + "command list: " + rhi::ResultName(result));
    }
    shard.lists.fetch_add(1, std::memory_order_relaxed);
    const uint64_t nameId = m_nextDebugNameId.fetch_add(1, std::memory_order_relaxed);
    std::string debugName = std::string("ORG ") + QueueKindDebugName(m_type) + (m_secondary ? " Secondary" : "") + " CommandList #" + std::to_string(nameId);
    list->SetName(debugName.c_str());
    // A primary list is created in the recording state. A secondary list is not. It starts recording in BeginSecondary.
    if (!m_secondary) list->End();
    const auto duration = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - createBegin).count());
    basic_telemetry::AddCounter(m_telemetryPrefix + "Created");
    basic_telemetry::Record(m_telemetryPrefix + "CreateDurationNs", duration);
    return list;
}

void CommandListPool::ResetClaimed(CommandListPair& pair) {
    BT_ZONE_SCOPE("CommandListPool::ResetClaimed");
    auto& shard = *pair.shard;
    if (shard.HasOrphans()) shard.DestroyOrphans();
    // A shard of one list resets its whole allocator. A shared one resets only this list (CommandList::Recycle with the
    // list's own allocator).
    if (!shard.Shared()) shard.Allocator().Recycle();
    pair.list->Recycle(shard.Allocator());
}

void CommandListPool::Begin(CommandListPair& pair) {
    BT_ZONE_SCOPE("CommandListPool::Begin");
    if (pair.allocator) throw std::logic_error("A command list with its own allocator is not a pool's");
    if (pair.m_claimed) {
        ResetClaimed(pair);
        return;
    }
    // The list's own shard, when nobody holds it.
    if (pair.list && pair.shard && pair.shard->TryClaim()) {
        pair.m_claimed = true;
        ResetClaimed(pair);
        return;
    }
    // Otherwise trade the list for an idle one of a free shard. The traded list is idle too: the GPU is done with it.
    {
        std::lock_guard lock(m_mutex);
        for (auto it = m_available.rbegin(); it != m_available.rend(); ++it) {
            if (!it->list || !it->shard || !it->shard->TryClaim()) continue;
            std::swap(pair.list, it->list);
            std::swap(pair.shard, it->shard);
            if (!it->list) m_available.erase(std::next(it).base());
            pair.m_claimed = true;
            UpdateDiagnosticsCountsLocked();
            break;
        }
    }
    if (pair.m_claimed) {
        ResetClaimed(pair);
        return;
    }
    // No free shard has an idle list. Make a list in a free or new shard. The old list goes back to the idle ones.
    auto shard = ClaimAnyShard();
    rhi::CommandListPtr list;
    try {
        list = CreateList(*shard);
    } catch (...) {
        shard->Release();
        throw;
    }
    if (pair.list) {
        CommandListPair old;
        old.list = std::move(pair.list);
        old.shard = std::move(pair.shard);
        std::lock_guard lock(m_mutex);
        m_available.emplace_back(std::move(old));
        UpdateDiagnosticsCountsLocked();
    }
    pair.list = std::move(list);
    pair.shard = std::move(shard);
    pair.m_claimed = true;
    basic_telemetry::AddCounter(m_telemetryPrefix + "BeginCreated");
    ResetClaimed(pair);
}

std::vector<CommandListPair> CommandListPool::CreateIdlePairs(size_t count) {
    BT_ZONE_SCOPE("CommandListPool::CreateIdlePairs");
    std::vector<CommandListPair> created;
    created.reserve(count);
    std::shared_ptr<CommandAllocatorShard> shard;
    try {
        for (size_t i = 0; i < count; ++i) {
            // One claim serves every list made in a shared shard. A shard of one list is released after its list.
            if (!shard) shard = ClaimAnyShard();
            CommandListPair pair;
            pair.list = CreateList(*shard);
            pair.shard = shard;
            created.emplace_back(std::move(pair));
            if (!shard->Shared()) {
                shard->Release();
                shard.reset();
            }
        }
    } catch (...) {
        if (shard) shard->Release();
        throw;
    }
    if (shard) shard->Release();
    return created;
}

CommandListPair CommandListPool::Request() {
    auto batch = AcquireBatch(1, 0);
    CommandListPair pair = std::move(batch.front());
    try {
        Begin(pair);
    } catch (...) {
        RecycleForNextRequest(std::move(pair));
        throw;
    }
    return pair;
}

void CommandListPool::PublishDemand(size_t requiredCount, uint64_t completedFenceValue) {
    BT_ZONE_SCOPE("CommandListPool::PublishDemand");
    RecycleCompleted(completedFenceValue);
    Diagnostics diagnostics;
    {
        std::lock_guard lock(m_mutex);
        m_diagnostics.lastRequestedCount = requiredCount;
        m_highWaterRequestedCount = (std::max)(m_highWaterRequestedCount, requiredCount);
        m_readyTargetCount = m_highWaterRequestedCount + kWarmSlackCommandLists;
        // Lists still in flight raise the capacity when an acquisition runs short (AcquireBatch).
        m_capacityTargetCount = (std::max)(m_capacityTargetCount, m_readyTargetCount);
        UpdateDiagnosticsCountsLocked();
        diagnostics = m_diagnostics;
    }
    basic_telemetry::SetGauge(m_telemetryPrefix + "Ready", static_cast<int64_t>(diagnostics.availableCount));
    basic_telemetry::SetGauge(m_telemetryPrefix + "ReadyOrPreparing", static_cast<int64_t>(diagnostics.availableCount));
    basic_telemetry::SetGauge(m_telemetryPrefix + "InFlight", static_cast<int64_t>(diagnostics.inFlightCount));
    basic_telemetry::SetGauge(m_telemetryPrefix + "CheckedOut", static_cast<int64_t>(diagnostics.checkedOutCount));
    basic_telemetry::SetGauge(m_telemetryPrefix + "TotalOwned", static_cast<int64_t>(diagnostics.totalOwnedCount));
    basic_telemetry::SetGauge(m_telemetryPrefix + "ReadyTarget", static_cast<int64_t>(diagnostics.readyTargetCount));
    basic_telemetry::SetGauge(m_telemetryPrefix + "CapacityTarget", static_cast<int64_t>(diagnostics.capacityTargetCount));
    basic_telemetry::SetGauge(m_telemetryPrefix + "ReadyDeficit", static_cast<int64_t>(diagnostics.preparedDeficit));
    basic_telemetry::SetGauge(m_telemetryPrefix + "Shards", static_cast<int64_t>(diagnostics.shardCount));
    ScheduleBackgroundReset();
}

std::vector<CommandListPair> CommandListPool::AcquireBatch(size_t count, uint64_t completedFenceValue) {
    BT_ZONE_SCOPE("CommandListPool::AcquireBatch");
    if (!count) return {};
    const auto acquireBegin = std::chrono::steady_clock::now();
    const size_t previousHighWater = [&] {
        std::lock_guard lock(m_mutex);
        return m_highWaterRequestedCount;
    }();
    PublishDemand(count, completedFenceValue);

    bool startupStall = false;
    bool growthStall = false;
    {
        std::lock_guard lock(m_mutex);
        if (m_available.size() < count) {
            if (previousHighWater == 0) {
                ++m_diagnostics.startupStallCount;
                startupStall = true;
            }
            else if (count > previousHighWater) {
                ++m_diagnostics.growthStallCount;
                growthStall = true;
            }
        }
    }
    if (startupStall) basic_telemetry::AddCounter(m_telemetryPrefix + "StartupStalls");
    if (growthStall) basic_telemetry::AddCounter(m_telemetryPrefix + "GrowthStalls");

    bool waited = false;
    auto waitBegin = std::chrono::steady_clock::now();
    // Held from the check that enough pairs are available until they are taken. Another thread could take them otherwise.
    std::unique_lock lock(m_mutex, std::defer_lock);
    for (;;) {
        if (!lock.owns_lock()) lock.lock();
        if (m_backgroundFailure) std::rethrow_exception(m_backgroundFailure);
        if (m_available.size() >= count) break;
        if (m_stopBackgroundReset)
            throw std::runtime_error("Command-list pool stopped before satisfying batch demand");
        if (m_backgroundResetScheduled.load(std::memory_order_acquire)) {
            waited = true;
            m_availableReady.wait(lock, [this, count] {
                return m_available.size() >= count || m_stopBackgroundReset
                    || m_backgroundFailure
                    || !m_backgroundResetScheduled.load(std::memory_order_acquire);
            });
            continue;
        }

        const size_t shortage = count - m_available.size();
        // The capacity target covers what one generation asks for. The lists still in flight are not idle, so grow the
        // capacity by the shortage. Grow asynchronously before falling back to driver work on this thread.
        m_capacityTargetCount = (std::max)(m_capacityTargetCount, TotalOwnedLocked() + shortage);
        UpdateDiagnosticsCountsLocked();
        lock.unlock();
        if (ScheduleBackgroundReset()) {
            waited = true;
            continue;
        }
        lock.lock();
        m_backgroundCreateActiveCount += shortage;
        ++m_diagnostics.inlineCreateFallbackCount;
        UpdateDiagnosticsCountsLocked();
        lock.unlock();
        basic_telemetry::AddCounter(m_telemetryPrefix + "InlineCreateFallbacks");

        std::vector<CommandListPair> created;
        try {
            created = CreateIdlePairs(shortage);
        } catch (...) {
            lock.lock();
            m_backgroundCreateActiveCount -= shortage;
            UpdateDiagnosticsCountsLocked();
            throw;
        }
        lock.lock();
        m_backgroundCreateActiveCount -= shortage;
        for (auto& pair : created) m_available.emplace_back(std::move(pair));
        UpdateDiagnosticsCountsLocked();
    }

    if (waited) {
        const auto duration = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - waitBegin).count());
        ++m_diagnostics.acquisitionWaitCount;
        m_diagnostics.acquisitionWaitDurationNs += duration;
        basic_telemetry::Record("ORG.CommandListPool.AcquisitionWait.DurationNs", duration);
        basic_telemetry::AddCounter(m_telemetryPrefix + "AcquisitionWaits");
        basic_telemetry::Record(m_telemetryPrefix + "AcquisitionWaitDurationNs", duration);
    }
    std::vector<CommandListPair> result;
    result.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        result.emplace_back(std::move(m_available.back()));
        m_available.pop_back();
    }
    m_checkedOutCount += count;
    m_diagnostics.acquiredCount += count;
    basic_telemetry::AddCounter(m_telemetryPrefix + "Acquired", static_cast<int64_t>(count));
    UpdateDiagnosticsCountsLocked();
    lock.unlock();
    basic_telemetry::Record(m_telemetryPrefix + "AcquireDurationNs",
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - acquireBegin).count()));
    ScheduleBackgroundReset();
    return result;
}

void CommandListPool::RecycleForNextRequest(CommandListPair&& pair) {
    // The recorder is done with the list, so its claim can go even if it was never ended.
    pair.EndRecording();
    {
        std::lock_guard lock(m_mutex);
        if (m_checkedOutCount != 0) --m_checkedOutCount;
        if (pair.list && pair.shard) {
            m_available.emplace_back(std::move(pair));
            ++m_diagnostics.recycleQueuedCount;
        }
        UpdateDiagnosticsCountsLocked();
    }
    m_availableReady.notify_all();
}

void CommandListPool::Discard(CommandListPair&& pair) noexcept {
    {
        std::lock_guard lock(m_mutex);
        if (m_checkedOutCount != 0) --m_checkedOutCount;
        UpdateDiagnosticsCountsLocked();
    }
    // Releases the claim, then frees the list under its shard's claim.
    CommandListPair dropped = std::move(pair);
}

void CommandListPool::Recycle(CommandListPair&& pair, uint64_t fenceValue) {
    BT_ZONE_SCOPE("CommandListPool::Recycle");
    if (fenceValue == 0) {
        RecycleForNextRequest(std::move(pair));
        return;
    }
    pair.EndRecording();
    std::lock_guard lock(m_mutex);
    if (m_checkedOutCount != 0) --m_checkedOutCount;
    if (pair.list && pair.shard) m_inFlight.emplace_back(fenceValue, std::move(pair));
    UpdateDiagnosticsCountsLocked();
}

void CommandListPool::RecycleCompleted(uint64_t completedFenceValue) {
    BT_ZONE_SCOPE("CommandListPool::RecycleCompleted");
    size_t movedCount = 0;
    {
        std::lock_guard lock(m_mutex);
        while (!m_inFlight.empty() && m_inFlight.front().first <= completedFenceValue) {
            m_available.emplace_back(std::move(m_inFlight.front().second));
            m_inFlight.pop_front();
            ++movedCount;
        }
        m_diagnostics.recycleQueuedCount += movedCount;
        UpdateDiagnosticsCountsLocked();
    }

    if (movedCount > 0) {
        basic_telemetry::AddCounter(m_telemetryPrefix + "FenceRecycled", static_cast<int64_t>(movedCount));
        m_availableReady.notify_all();
    }
}

bool CommandListPool::SheddableLocked() const noexcept {
    return m_available.size() > m_readyTargetCount && TotalOwnedLocked() > m_capacityTargetCount;
}

bool CommandListPool::ScheduleBackgroundReset() {
    std::shared_ptr<org::runtime::ITaskService> service;
    std::shared_ptr<org::runtime::ITaskScope> scope;
    org::runtime::TaskPriority priority = org::runtime::TaskPriority::Background;
    {
        std::lock_guard lock(m_mutex);
        if (m_stopBackgroundReset) return false;
        if (!m_taskService) m_taskService = org::runtime::GetDefaultTaskService();
        if (!m_taskService) return false;
        if (!m_taskScope) {
            m_taskScope = m_taskService->CreateScope(
                std::string("ORG.CommandListReset.") + QueueKindDebugName(m_type));
        }
        const bool grow = TotalOwnedLocked() < m_capacityTargetCount;
        if (!grow && !SheddableLocked()) return false;
        const size_t readyOrPreparing = m_available.size() + m_backgroundCreateActiveCount;
        if (grow && readyOrPreparing < m_readyTargetCount)
            priority = org::runtime::TaskPriority::FrameCritical;
        else if (grow) priority = org::runtime::TaskPriority::Streaming;
        service = m_taskService;
        scope = m_taskScope;
    }
    if (!service || !scope) return false;
    bool expected = false;
    if (!m_backgroundResetScheduled.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return true;
    if (!service->Submit(scope, priority,
            "ORG.CommandListReset", [this] { BackgroundResetDrain(); })) {
        m_backgroundResetScheduled.store(false, std::memory_order_release);
        spdlog::error("ORG command-list reset submission was rejected for {} queue",
            QueueKindDebugName(m_type));
        m_availableReady.notify_all();
        return false;
    }
    return true;
}

void CommandListPool::ShedIdle() {
    BT_ZONE_SCOPE("CommandListPool::ShedIdle");
    // The coldest idle lists are at the front: acquisitions take from the back.
    std::vector<CommandListPair> shed;
    std::vector<std::shared_ptr<CommandAllocatorShard>> shards;
    {
        std::lock_guard lock(m_mutex);
        if (SheddableLocked()) {
            const size_t excess = (std::min)(m_available.size() - m_readyTargetCount,
                TotalOwnedLocked() - m_capacityTargetCount);
            shed.reserve(excess);
            for (size_t i = 0; i < excess; ++i) shed.emplace_back(std::move(m_available[i]));
            m_available.erase(m_available.begin(), m_available.begin() + static_cast<std::ptrdiff_t>(excess));
            m_diagnostics.backgroundResetCount += excess;
            UpdateDiagnosticsCountsLocked();
        }
        shards = m_shards;
    }
    const bool freedLists = !shed.empty();
    // Each list is freed now if its shard is free. Otherwise it is an orphan its shard's next claimant frees.
    shed.clear();
    // Free the orphans and give the freed memory back, in every shard that nobody holds.
    for (const auto& shard : shards) {
        if (!shard->TryClaim()) continue;
        const bool hadOrphans = shard->HasOrphans();
        shard->DestroyOrphans();
        if (freedLists || hadOrphans) shard->Allocator().Trim();
        shard->Release();
    }
    shards.clear();
    // Drop the shards with no lists, but keep one: a new pool costs the driver more than an empty one.
    std::vector<std::shared_ptr<CommandAllocatorShard>> dropped;
    {
        std::lock_guard lock(m_mutex);
        for (auto it = m_shards.begin(); it != m_shards.end() && m_shards.size() > 1;) {
            auto& shard = *it;
            // Claimed here so that no thread takes it between the check and the erase. The pool's lock covers every
            // claim of a shard with no lists (ClaimAnyShard).
            if (shard->lists.load(std::memory_order_relaxed) == 0 && shard->TryClaim()) {
                dropped.push_back(std::move(*it));
                it = m_shards.erase(it);
            } else {
                ++it;
            }
        }
        UpdateDiagnosticsCountsLocked();
    }
    for (auto& shard : dropped) shard->Release();
    if (!dropped.empty())
        basic_telemetry::AddCounter(m_telemetryPrefix + "ShardsDropped", static_cast<int64_t>(dropped.size()));
}

void CommandListPool::BackgroundResetDrain() {
    // Making a list is normally tens of microseconds. Amortize the scheduler hand-off while keeping the typical slice
    // below 2 ms.
    constexpr size_t kPairsPerDrain = 16;
    size_t createWarmCount = 0;
    {
        std::lock_guard lock(m_mutex);
        const size_t totalOwned = TotalOwnedLocked();
        createWarmCount = totalOwned < m_capacityTargetCount
            ? (std::min)(kPairsPerDrain, m_capacityTargetCount - totalOwned) : 0;
        m_backgroundCreateActiveCount += createWarmCount;
        UpdateDiagnosticsCountsLocked();
    }

    std::vector<CommandListPair> warmPairs;
    try {
        if (createWarmCount) {
            BT_ZONE_SCOPE("CommandListPool::BackgroundResetDrain::CreateWarmPair");
            warmPairs = CreateIdlePairs(createWarmCount);
        } else {
            ShedIdle();
        }
    } catch (...) {
        {
            std::lock_guard lock(m_mutex);
            m_backgroundCreateActiveCount -= createWarmCount;
            m_backgroundFailure = std::current_exception();
            m_backgroundResetScheduled.store(false, std::memory_order_release);
            UpdateDiagnosticsCountsLocked();
        }
        m_availableReady.notify_all();
        spdlog::error("ORG command-list background preparation failed for {} queue",
            QueueKindDebugName(m_type));
        return;
    }

    bool hasMore = false;
    {
        std::lock_guard lock(m_mutex);
        m_available.reserve(m_available.size() + warmPairs.size());
        for (auto& pair : warmPairs) m_available.emplace_back(std::move(pair));
        m_backgroundCreateActiveCount -= createWarmCount;
        m_diagnostics.backgroundCreatedCount += warmPairs.size();
        if (!warmPairs.empty())
            basic_telemetry::AddCounter(m_telemetryPrefix + "BackgroundCreated", static_cast<int64_t>(warmPairs.size()));
        m_backgroundResetScheduled.store(false, std::memory_order_release);
        hasMore = !m_stopBackgroundReset && TotalOwnedLocked() < m_capacityTargetCount;
        UpdateDiagnosticsCountsLocked();
    }
    m_availableReady.notify_all();
    if (hasMore) ScheduleBackgroundReset();
}


} // namespace org
