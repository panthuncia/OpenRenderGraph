#include "Render/CommandListPool.h"

#include <algorithm>
#include <string>
#include <spdlog/spdlog.h>
#include <tracy/Tracy.hpp>

namespace {
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
    m_backgroundResetThread = std::thread([this] { BackgroundResetMain(); });
}

CommandListPool::~CommandListPool() {
    {
        std::lock_guard lock(m_mutex);
        m_stopBackgroundReset = true;
    }
    m_backgroundResetCv.notify_one();
    if (m_backgroundResetThread.joinable()) {
        m_backgroundResetThread.join();
    }
}

void CommandListPool::PreparePairForReuse(CommandListPair& pair) {
    ZoneScopedN("CommandListPool::PreparePairForReuse");
    pair.allocator->Recycle();
    pair.list->Recycle(pair.allocator.Get());
}

void CommandListPool::UpdateDiagnosticsCountsLocked() {
    m_diagnostics.availableCount = m_available.size();
    m_diagnostics.inFlightCount = m_inFlight.size();
    m_diagnostics.backgroundResetPendingCount = m_pendingBackgroundReset.size() + m_backgroundResetActiveCount;
}

CommandListPair CommandListPool::CreateReadyPair() {
    ZoneScopedN("CommandListPool::CreateReadyPair");
    CommandListPair pair;
	auto result = m_device.CreateCommandAllocator(m_type, pair.allocator);
	result = m_device.CreateCommandList(m_type, pair.allocator.Get(), pair.list);
    (void)result;

    const uint64_t nameId = m_nextDebugNameId.fetch_add(1, std::memory_order_relaxed);
    std::string debugName = std::string("ORG ") + QueueKindDebugName(m_type) + " CommandList #" + std::to_string(nameId);
    pair.list->SetName(debugName.c_str());

    pair.list->End();
    PreparePairForReuse(pair);
    return pair;
}

CommandListPair CommandListPool::Request() {
    ZoneScopedN("CommandListPool::Request");
    {
        std::lock_guard lock(m_mutex);
        if (!m_available.empty()) {
            CommandListPair pair = std::move(m_available.back());
            m_available.pop_back();
            ++m_diagnostics.reusedThisFrame;
            UpdateDiagnosticsCountsLocked();
            return pair;
        }
    }

    CommandListPair pair = CreateReadyPair();
    {
        std::lock_guard lock(m_mutex);
        ++m_diagnostics.createdThisFrame;
        UpdateDiagnosticsCountsLocked();
    }
    return pair;
}

void CommandListPool::PrepareForRequests(size_t requiredCount, uint64_t completedFenceValue) {
    ZoneScopedN("CommandListPool::PrepareForRequests");
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
    {
        std::lock_guard lock(m_mutex);
        if (m_available.size() < requiredCount) {
            deficit = requiredCount - m_available.size();
            m_diagnostics.preparedDeficit = deficit;
            m_available.reserve(requiredCount);
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
            "CommandListPool::PrepareForRequests queue={} required={} availableBeforeWarm={} created={} inFlight={}",
            QueueKindDebugName(m_type),
            requiredCount,
            requiredCount - deficit,
            deficit,
            inFlightCount);
    }

    {
        std::lock_guard lock(m_mutex);
        UpdateDiagnosticsCountsLocked();
    }
}

void CommandListPool::Recycle(CommandListPair&& pair, uint64_t fenceValue) {
    ZoneScopedN("CommandListPool::Recycle");
    bool notifyBackgroundReset = false;
    if (fenceValue == 0) {
        {
            std::lock_guard lock(m_mutex);
            m_pendingBackgroundReset.emplace_back(std::move(pair));
            ++m_diagnostics.enqueuedForBackgroundResetThisFrame;
            UpdateDiagnosticsCountsLocked();
        }
        notifyBackgroundReset = true;
    }
    else {
        std::lock_guard lock(m_mutex);
        m_inFlight.emplace_back(fenceValue, std::move(pair));
        UpdateDiagnosticsCountsLocked();
    }

    if (notifyBackgroundReset) {
        m_backgroundResetCv.notify_one();
    }
}

void CommandListPool::RecycleCompleted(uint64_t completedFenceValue) {
    ZoneScopedN("CommandListPool::RecycleCompleted");
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
        m_backgroundResetCv.notify_one();
    }
}

void CommandListPool::BackgroundResetMain() {
    tracy::SetThreadName("ORG CommandListPool Reset");
    std::vector<CommandListPair> local;

    for (;;) {
        {
            std::unique_lock lock(m_mutex);
            m_backgroundResetCv.wait(lock, [this] {
                return m_stopBackgroundReset || !m_pendingBackgroundReset.empty();
            });

            if (m_stopBackgroundReset && m_pendingBackgroundReset.empty()) {
                break;
            }

            local.clear();
            local.swap(m_pendingBackgroundReset);
            m_backgroundResetActiveCount += local.size();
            UpdateDiagnosticsCountsLocked();
        }

        {
            ZoneScopedN("CommandListPool::BackgroundResetMain::ResetPairs");
            for (auto& pair : local) {
                PreparePairForReuse(pair);
            }
        }

        {
            std::lock_guard lock(m_mutex);
            m_available.reserve(m_available.size() + local.size());
            for (auto& pair : local) {
                m_available.emplace_back(std::move(pair));
            }
            m_backgroundResetActiveCount -= local.size();
            m_diagnostics.backgroundResetCompletedThisFrame += local.size();
            UpdateDiagnosticsCountsLocked();
        }
    }
}
