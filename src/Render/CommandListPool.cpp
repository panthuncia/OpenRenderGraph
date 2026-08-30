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
    constexpr size_t kWarmFramesInFlight = 4;
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
    : m_device(device), m_type(type) {
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
    m_diagnostics.warmTargetCount = m_warmTargetCount;
}

CommandListPair CommandListPool::CreateReadyPair() {
    BT_ZONE_SCOPE("CommandListPool::CreateReadyPair");
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
    return pair;
}

CommandListPair CommandListPool::Request() {
    BT_ZONE_SCOPE("CommandListPool::Request");
    {
        std::lock_guard lock(m_mutex);
        if (!m_available.empty()) {
            CommandListPair pair = std::move(m_available.back());
            m_available.pop_back();
            ++m_checkedOutCount;
            ++m_diagnostics.reusedThisFrame;
            UpdateDiagnosticsCountsLocked();
            return pair;
        }
    }

    CommandListPair pair = CreateReadyPair();
    {
        std::lock_guard lock(m_mutex);
        ++m_checkedOutCount;
        ++m_diagnostics.createdThisFrame;
        UpdateDiagnosticsCountsLocked();
    }
    return pair;
}

void CommandListPool::PrepareForRequests(size_t requiredCount, uint64_t completedFenceValue) {
    BT_ZONE_SCOPE("CommandListPool::PrepareForRequests");
    const auto prepareBegin = std::chrono::steady_clock::now();
    {
        std::lock_guard lock(m_mutex);
        m_diagnostics.lastRequestedCount = requiredCount;
        m_diagnostics.createdThisFrame = 0;
        m_diagnostics.reusedThisFrame = 0;
        m_diagnostics.preparedDeficit = 0;
        m_diagnostics.enqueuedForBackgroundResetThisFrame = 0;
        m_diagnostics.backgroundResetCompletedThisFrame = 0;
        UpdateDiagnosticsCountsLocked();
    }

    RecycleCompleted(completedFenceValue);

    size_t deficit = 0;
    size_t availableBeforeWarm = 0;
    size_t totalOwnedBeforeWarm = 0;
    size_t warmTargetCount = 0;
    {
        std::lock_guard lock(m_mutex);
        m_highWaterRequestedCount = std::max(m_highWaterRequestedCount, requiredCount);
        warmTargetCount = (m_highWaterRequestedCount * kWarmFramesInFlight) + kWarmSlackCommandLists;
        m_warmTargetCount = warmTargetCount;
        availableBeforeWarm = m_available.size();
        totalOwnedBeforeWarm = m_available.size()
            + m_checkedOutCount
            + m_inFlight.size()
            + m_pendingBackgroundReset.size()
            + m_backgroundResetActiveCount
            + m_backgroundCreateActiveCount;
        const size_t requiredAvailableDeficit = requiredCount > m_available.size()
            ? requiredCount - m_available.size()
            : 0;
        // Warm ownership is reached naturally across frames-in-flight. Do not
        // synchronously allocate speculative warm capacity on the render
        // thread when a graph grows during camera movement.
        deficit = requiredAvailableDeficit;
        if (deficit > 0) {
            m_diagnostics.preparedDeficit = deficit;
            m_available.reserve(m_available.size() + deficit);
        }
    }

    if (deficit > 0) {
        std::vector<CommandListPair> created;
        created.reserve(deficit);
        for (size_t i = 0; i < deficit; ++i) {
            created.emplace_back(CreateReadyPair());
        }

        size_t inFlightCount = 0;
        {
            std::lock_guard lock(m_mutex);
            m_available.reserve(std::max(m_available.size() + created.size(), requiredCount));
            for (auto& pair : created) {
                m_available.emplace_back(std::move(pair));
                ++m_diagnostics.createdThisFrame;
            }
            UpdateDiagnosticsCountsLocked();
            inFlightCount = m_diagnostics.inFlightCount;
        }

        spdlog::debug(
            "CommandListPool::PrepareForRequests queue={} required={} availableBeforeWarm={} totalOwnedBeforeWarm={} warmTarget={} created={} inFlight={}",
            QueueKindDebugName(m_type),
            requiredCount,
            availableBeforeWarm,
            totalOwnedBeforeWarm,
            warmTargetCount,
            deficit,
            inFlightCount);
    }

    BT_PLOT("ORG.CommandListPool.Prepare.Required", static_cast<int64_t>(requiredCount));
    BT_PLOT("ORG.CommandListPool.Prepare.Created", static_cast<int64_t>(deficit));
    BT_PLOT("ORG.CommandListPool.Prepare.AvailableBeforeWarm", static_cast<int64_t>(availableBeforeWarm));
    BT_PLOT("ORG.CommandListPool.Prepare.TotalOwnedBeforeWarm", static_cast<int64_t>(totalOwnedBeforeWarm));
    BT_PLOT("ORG.CommandListPool.Prepare.WarmTarget", static_cast<int64_t>(warmTargetCount));
    BT_PLOT("ORG.CommandListPool.Prepare.ResetInline", int64_t{ 0 });
    basic_telemetry::Record(
        "ORG.CommandListPool.Prepare.DurationNs",
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - prepareBegin).count()));

    {
        std::lock_guard lock(m_mutex);
        UpdateDiagnosticsCountsLocked();
    }
    ScheduleBackgroundReset();
}

void CommandListPool::Recycle(CommandListPair&& pair, uint64_t fenceValue) {
    BT_ZONE_SCOPE("CommandListPool::Recycle");
    bool notifyBackgroundReset = false;
    if (fenceValue == 0) {
        {
            std::lock_guard lock(m_mutex);
            if (m_checkedOutCount != 0) --m_checkedOutCount;
            m_pendingBackgroundReset.emplace_back(std::move(pair));
            ++m_diagnostics.enqueuedForBackgroundResetThisFrame;
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
        m_diagnostics.enqueuedForBackgroundResetThisFrame += movedCount;
        UpdateDiagnosticsCountsLocked();
    }

    if (movedCount > 0) {
        ScheduleBackgroundReset();
    }
}

void CommandListPool::ScheduleBackgroundReset() {
    std::shared_ptr<org::runtime::ITaskService> service;
    std::shared_ptr<org::runtime::ITaskScope> scope;
    org::runtime::TaskPriority priority = org::runtime::TaskPriority::Background;
    {
        std::lock_guard lock(m_mutex);
        if (m_stopBackgroundReset) return;
        if (!m_taskService) m_taskService = org::runtime::GetDefaultTaskService();
        if (!m_taskService) return;
        if (!m_taskScope) {
            m_taskScope = m_taskService->CreateScope(
                std::string("ORG.CommandListReset.") + QueueKindDebugName(m_type));
        }
        const size_t totalOwned = m_available.size()
            + m_checkedOutCount
            + m_inFlight.size()
            + m_pendingBackgroundReset.size()
            + m_backgroundResetActiveCount
            + m_backgroundCreateActiveCount;
        if (m_pendingBackgroundReset.empty() && totalOwned >= m_warmTargetCount) return;
        // Completed command lists are needed by a future frame and must make
        // progress even while the background lane is saturated by streaming.
        // Speculative warm creation remains background priority.
        if (!m_pendingBackgroundReset.empty()) {
            priority = org::runtime::TaskPriority::Streaming;
        }
        service = m_taskService;
        scope = m_taskScope;
    }
    bool expected = false;
    if (!m_backgroundResetScheduled.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return;
    if (!service->Submit(scope, priority,
            "ORG.CommandListReset", [this] { BackgroundResetDrain(); })) {
        m_backgroundResetScheduled.store(false, std::memory_order_release);
        spdlog::error("ORG command-list reset submission was rejected for {} queue", QueueKindDebugName(m_type));
    }
}

void CommandListPool::BackgroundResetDrain() {
    // Allocator/list recycling is normally tens of microseconds. Amortize the
    // scheduler hand-off while keeping the typical slice below 2 ms. A single
    // driver recycle can occasionally exceed that budget by itself, but that
    // stall remains on a background worker rather than the render thread.
    constexpr size_t kPairsPerDrain = 16;
    std::vector<CommandListPair> local;
    bool createWarmPair = false;
    {
        std::lock_guard lock(m_mutex);
        const size_t count = (std::min)(kPairsPerDrain, m_pendingBackgroundReset.size());
        local.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            local.emplace_back(std::move(m_pendingBackgroundReset.back()));
            m_pendingBackgroundReset.pop_back();
        }
        m_backgroundResetActiveCount += local.size();
        const size_t totalOwned = m_available.size()
            + m_checkedOutCount
            + m_inFlight.size()
            + m_pendingBackgroundReset.size()
            + m_backgroundResetActiveCount
            + m_backgroundCreateActiveCount;
        createWarmPair = local.empty() && totalOwned < m_warmTargetCount;
        if (createWarmPair) {
            ++m_backgroundCreateActiveCount;
        }
        UpdateDiagnosticsCountsLocked();
    }

    {
        BT_ZONE_SCOPE("CommandListPool::BackgroundResetDrain::ResetPairs");
        for (auto& pair : local) {
            PreparePairForReuse(pair);
        }
    }

    std::optional<CommandListPair> warmPair;
    if (createWarmPair) {
        BT_ZONE_SCOPE("CommandListPool::BackgroundResetDrain::CreateWarmPair");
        warmPair.emplace(CreateReadyPair());
    }

    bool hasMore = false;
    {
        std::lock_guard lock(m_mutex);
        m_available.reserve(m_available.size() + local.size());
        for (auto& pair : local) {
            m_available.emplace_back(std::move(pair));
        }
        if (warmPair) {
            m_available.emplace_back(std::move(*warmPair));
            --m_backgroundCreateActiveCount;
        }
        m_backgroundResetActiveCount -= local.size();
        m_diagnostics.backgroundResetCompletedThisFrame += local.size();
        m_backgroundResetScheduled.store(false, std::memory_order_release);
        const size_t totalOwned = m_available.size()
            + m_checkedOutCount
            + m_inFlight.size()
            + m_pendingBackgroundReset.size()
            + m_backgroundResetActiveCount
            + m_backgroundCreateActiveCount;
        hasMore = !m_stopBackgroundReset &&
            (!m_pendingBackgroundReset.empty() || totalOwned < m_warmTargetCount);
        UpdateDiagnosticsCountsLocked();
    }
    if (hasMore) ScheduleBackgroundReset();
}


} // namespace org
