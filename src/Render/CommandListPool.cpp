#include "Render/CommandListPool.h"

#include <algorithm>
#include <chrono>
#include <optional>
#include <string>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include <BasicTelemetry/Tracy.h>
#include "Render/Runtime/TaskServiceAccess.h"


namespace org {

namespace {
    constexpr size_t kPreparedGenerations = 4;
    constexpr size_t kWarmSlackCommandLists = 4;

    const char* QueueKindDebugName(rhi::QueueKind type) noexcept {
        switch (type) {
        case rhi::QueueKind::Graphics: return "Graphics";
        case rhi::QueueKind::Compute: return "Compute";
        case rhi::QueueKind::Copy: return "Copy";
        default: return "Unknown";
        }
    }
}

CommandListPool::CommandListPool(rhi::Device& device, rhi::QueueKind type)
    : m_device(device), m_type(type),
      m_telemetryPrefix(std::string("ORG.CommandListPool.") + QueueKindDebugName(type) + ".") {
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

void CommandListPool::PreparePairForReuse(CommandListPair& pair) {
    BT_ZONE_SCOPE("CommandListPool::PreparePairForReuse");
    {
        BT_ZONE_SCOPE("CommandListPool::PreparePairForReuse::AllocatorReset");
        pair.allocator->Recycle();
    }
    {
        BT_ZONE_SCOPE("CommandListPool::PreparePairForReuse::CommandListReset");
        pair.list->Recycle(pair.allocator.Get());
    }
}

void CommandListPool::UpdateDiagnosticsCountsLocked() {
    m_diagnostics.availableCount = m_available.size();
    m_diagnostics.checkedOutCount = m_checkedOutCount;
    m_diagnostics.inFlightCount = m_inFlight.size();
    m_diagnostics.backgroundResetPendingCount = m_pendingBackgroundReset.size() + m_backgroundResetActiveCount;
    m_diagnostics.totalOwnedCount = m_available.size() + m_checkedOutCount + m_inFlight.size() + m_pendingBackgroundReset.size() + m_backgroundResetActiveCount + m_backgroundCreateActiveCount;
    m_diagnostics.readyTargetCount = m_readyTargetCount;
    m_diagnostics.capacityTargetCount = m_capacityTargetCount;
    const size_t readyOrPreparing = m_available.size() + m_pendingBackgroundReset.size()
        + m_backgroundResetActiveCount + m_backgroundCreateActiveCount;
    m_diagnostics.preparedDeficit = m_readyTargetCount > readyOrPreparing
        ? m_readyTargetCount - readyOrPreparing : 0;

}

CommandListPair CommandListPool::CreateReadyPair() {
    BT_ZONE_SCOPE("CommandListPool::CreateReadyPair");
	const auto createBegin = std::chrono::steady_clock::now();
	if (!m_device) {
		throw std::runtime_error("CommandListPool lost its RHI device");
	}
    CommandListPair pair;
	auto result = m_device.CreateCommandAllocator(m_type, pair.allocator);
	if (!rhi::IsOk(result) || !pair.allocator) {
		throw std::runtime_error(std::string("Failed to create ORG command allocator: ") + rhi::ResultName(result));
	}
	result = m_device.CreateCommandList(m_type, pair.allocator.Get(), pair.list);
	if (!rhi::IsOk(result) || !pair.list) {
		throw std::runtime_error(std::string("Failed to create ORG command list: ") + rhi::ResultName(result));
	}

    const uint64_t nameId = m_nextDebugNameId.fetch_add(1, std::memory_order_relaxed);
    std::string debugName = std::string("ORG ") + QueueKindDebugName(m_type) + " CommandList #" + std::to_string(nameId);
    pair.list->SetName(debugName.c_str());

    pair.list->End();
    PreparePairForReuse(pair);
    const auto duration = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - createBegin).count());
    basic_telemetry::AddCounter(m_telemetryPrefix + "Created");
    basic_telemetry::Record(m_telemetryPrefix + "CreateDurationNs", duration);
    return pair;
}

CommandListPair CommandListPool::Request() {
    auto batch = AcquireBatch(1, 0);
    return std::move(batch.front());
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
        m_capacityTargetCount = (std::max)(m_capacityTargetCount,
            m_highWaterRequestedCount * kPreparedGenerations + kWarmSlackCommandLists);
        UpdateDiagnosticsCountsLocked();
        diagnostics = m_diagnostics;
    }
    basic_telemetry::SetGauge(m_telemetryPrefix + "Ready", static_cast<int64_t>(diagnostics.availableCount));
    basic_telemetry::SetGauge(m_telemetryPrefix + "ReadyOrPreparing",
        static_cast<int64_t>(diagnostics.availableCount + diagnostics.backgroundResetPendingCount));
    basic_telemetry::SetGauge(m_telemetryPrefix + "InFlight", static_cast<int64_t>(diagnostics.inFlightCount));
    basic_telemetry::SetGauge(m_telemetryPrefix + "CheckedOut", static_cast<int64_t>(diagnostics.checkedOutCount));
    basic_telemetry::SetGauge(m_telemetryPrefix + "TotalOwned", static_cast<int64_t>(diagnostics.totalOwnedCount));
    basic_telemetry::SetGauge(m_telemetryPrefix + "ReadyTarget", static_cast<int64_t>(diagnostics.readyTargetCount));
    basic_telemetry::SetGauge(m_telemetryPrefix + "CapacityTarget", static_cast<int64_t>(diagnostics.capacityTargetCount));
    basic_telemetry::SetGauge(m_telemetryPrefix + "ReadyDeficit", static_cast<int64_t>(diagnostics.preparedDeficit));
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
    for (;;) {
        std::unique_lock lock(m_mutex);
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
        const size_t totalOwned = m_available.size() + m_checkedOutCount + m_inFlight.size()
            + m_pendingBackgroundReset.size() + m_backgroundResetActiveCount
            + m_backgroundCreateActiveCount;
        // The normal capacity target covers GPU latency plus one ready
        // generation. If it is exhausted, grow asynchronously before falling
        // back to driver work on this thread.
        m_capacityTargetCount = (std::max)(m_capacityTargetCount, totalOwned + shortage);
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
        created.reserve(shortage);
        try {
            for (size_t i = 0; i < shortage; ++i) created.emplace_back(CreateReadyPair());
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

    std::unique_lock lock(m_mutex, std::defer_lock);
    lock.lock();
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
    {
        std::lock_guard lock(m_mutex);
        m_pendingBackgroundReset.emplace_back(std::move(pair));
        if (m_checkedOutCount != 0) --m_checkedOutCount;
        UpdateDiagnosticsCountsLocked();
    }
    ScheduleBackgroundReset();
}

void CommandListPool::Discard(CommandListPair&& pair) noexcept {
    std::lock_guard lock(m_mutex);
    if (m_checkedOutCount != 0) --m_checkedOutCount;
    UpdateDiagnosticsCountsLocked();
}

void CommandListPool::Recycle(CommandListPair&& pair, uint64_t fenceValue) {
    BT_ZONE_SCOPE("CommandListPool::Recycle");
    bool notifyBackgroundReset = false;
    if (fenceValue == 0) {
        {
            std::lock_guard lock(m_mutex);
            if (m_checkedOutCount != 0) --m_checkedOutCount;
            m_pendingBackgroundReset.emplace_back(std::move(pair));
            ++m_diagnostics.recycleQueuedCount;
            UpdateDiagnosticsCountsLocked();
        }
        notifyBackgroundReset = true;
    }
    else {
        std::lock_guard lock(m_mutex);
        if (m_checkedOutCount != 0) --m_checkedOutCount;
        m_inFlight.emplace_back(fenceValue, std::move(pair));
        UpdateDiagnosticsCountsLocked();
    }

    if (notifyBackgroundReset) {
        ScheduleBackgroundReset();
    }
}

void CommandListPool::RecycleCompleted(uint64_t completedFenceValue) {
    BT_ZONE_SCOPE("CommandListPool::RecycleCompleted");
    size_t movedCount = 0;
    {
        std::lock_guard lock(m_mutex);
        while (!m_inFlight.empty() && m_inFlight.front().first <= completedFenceValue) {
            m_pendingBackgroundReset.emplace_back(std::move(m_inFlight.front().second));
            m_inFlight.pop_front();
            ++movedCount;
        }
        m_diagnostics.recycleQueuedCount += movedCount;
        UpdateDiagnosticsCountsLocked();
    }

    if (movedCount > 0) {
        basic_telemetry::AddCounter(m_telemetryPrefix + "FenceRecycled", static_cast<int64_t>(movedCount));
        ScheduleBackgroundReset();
    }
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
        const size_t totalOwned = m_available.size() + m_checkedOutCount + m_inFlight.size()
            + m_pendingBackgroundReset.size() + m_backgroundResetActiveCount
            + m_backgroundCreateActiveCount;
        if (m_pendingBackgroundReset.empty() && totalOwned >= m_capacityTargetCount) return false;
        const size_t readyOrPreparing = m_available.size() + m_pendingBackgroundReset.size()
            + m_backgroundResetActiveCount + m_backgroundCreateActiveCount;
        if (readyOrPreparing < m_readyTargetCount)
            priority = org::runtime::TaskPriority::FrameCritical;
        else if (!m_pendingBackgroundReset.empty()) priority = org::runtime::TaskPriority::Streaming;
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

void CommandListPool::BackgroundResetDrain() {
    // Allocator/list recycling is normally tens of microseconds. Amortize the
    // scheduler hand-off while keeping the typical slice below 2 ms. A single
    // driver recycle can occasionally exceed that budget by itself, but that
    // stall remains on a background worker rather than the render thread.
    constexpr size_t kPairsPerDrain = 16;
    std::vector<CommandListPair> local;
    size_t createWarmCount = 0;
    {
        std::lock_guard lock(m_mutex);
        const size_t count = (std::min)(kPairsPerDrain, m_pendingBackgroundReset.size());
        local.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            local.emplace_back(std::move(m_pendingBackgroundReset.back()));
            m_pendingBackgroundReset.pop_back();
        }
        m_backgroundResetActiveCount += local.size();
        const size_t totalOwned = m_available.size() + m_checkedOutCount + m_inFlight.size()
            + m_pendingBackgroundReset.size() + m_backgroundResetActiveCount
            + m_backgroundCreateActiveCount;
        createWarmCount = local.empty() && totalOwned < m_capacityTargetCount
            ? (std::min)(kPairsPerDrain, m_capacityTargetCount - totalOwned) : 0;
        m_backgroundCreateActiveCount += createWarmCount;
        UpdateDiagnosticsCountsLocked();
    }

    std::vector<CommandListPair> warmPairs;
    try {
        {
            BT_ZONE_SCOPE("CommandListPool::BackgroundResetDrain::ResetPairs");
            for (auto& pair : local) PreparePairForReuse(pair);
        }
        if (createWarmCount) {
            BT_ZONE_SCOPE("CommandListPool::BackgroundResetDrain::CreateWarmPair");
            warmPairs.reserve(createWarmCount);
            for (size_t i = 0; i < createWarmCount; ++i)
                warmPairs.emplace_back(CreateReadyPair());
        }
    } catch (...) {
        {
            std::lock_guard lock(m_mutex);
            m_backgroundCreateActiveCount -= createWarmCount;
            m_backgroundResetActiveCount -= local.size();
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
        m_available.reserve(m_available.size() + local.size());
        for (auto& pair : local) {
            m_available.emplace_back(std::move(pair));
        }
        for (auto& pair : warmPairs) m_available.emplace_back(std::move(pair));
        m_backgroundCreateActiveCount -= warmPairs.size();
        m_backgroundResetActiveCount -= local.size();
        m_diagnostics.backgroundResetCount += local.size();
        m_diagnostics.backgroundCreatedCount += warmPairs.size();
        if (!local.empty())
            basic_telemetry::AddCounter(m_telemetryPrefix + "Reset", static_cast<int64_t>(local.size()));
        if (!warmPairs.empty())
            basic_telemetry::AddCounter(m_telemetryPrefix + "BackgroundCreated", static_cast<int64_t>(warmPairs.size()));
        m_backgroundResetScheduled.store(false, std::memory_order_release);
        const size_t totalOwned = m_available.size() + m_checkedOutCount + m_inFlight.size()
            + m_pendingBackgroundReset.size() + m_backgroundResetActiveCount
            + m_backgroundCreateActiveCount;
        hasMore = !m_stopBackgroundReset &&
            (!m_pendingBackgroundReset.empty() || totalOwned < m_capacityTargetCount);
        UpdateDiagnosticsCountsLocked();
    }
    m_availableReady.notify_all();
    if (hasMore) ScheduleBackgroundReset();
}


} // namespace org
