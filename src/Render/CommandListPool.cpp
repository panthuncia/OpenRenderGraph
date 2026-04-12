#include "Render/CommandListPool.h"

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
}

void CommandListPool::PreparePairForReuse(CommandListPair& pair) {
    pair.allocator->Recycle();
    pair.list->Recycle(pair.allocator.Get());
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
    if (!m_available.empty()) {
        CommandListPair pair = std::move(m_available.back());
        m_available.pop_back();
        ++m_diagnostics.reusedThisFrame;
        m_diagnostics.availableCount = m_available.size();
        m_diagnostics.inFlightCount = m_inFlight.size();
        return pair;
    }

    ++m_diagnostics.createdThisFrame;
    CommandListPair pair = CreateReadyPair();
    m_diagnostics.availableCount = m_available.size();
    m_diagnostics.inFlightCount = m_inFlight.size();
    return pair;
}

void CommandListPool::PrepareForRequests(size_t requiredCount, uint64_t completedFenceValue) {
    ZoneScopedN("CommandListPool::PrepareForRequests");
    m_diagnostics.lastRequestedCount = requiredCount;
    m_diagnostics.createdThisFrame = 0;
    m_diagnostics.reusedThisFrame = 0;
    m_diagnostics.preparedDeficit = 0;

    RecycleCompleted(completedFenceValue);

    if (m_available.size() < requiredCount) {
        const size_t deficit = requiredCount - m_available.size();
        m_diagnostics.preparedDeficit = deficit;
        m_available.reserve(requiredCount);
        for (size_t i = 0; i < deficit; ++i) {
            m_available.emplace_back(CreateReadyPair());
            ++m_diagnostics.createdThisFrame;
        }

        spdlog::debug(
            "CommandListPool::PrepareForRequests queue={} required={} availableBeforeWarm={} created={} inFlight={}",
            QueueKindDebugName(m_type),
            requiredCount,
            requiredCount - deficit,
            deficit,
            m_inFlight.size());
    }

    m_diagnostics.availableCount = m_available.size();
    m_diagnostics.inFlightCount = m_inFlight.size();
}

void CommandListPool::Recycle(CommandListPair&& pair, uint64_t fenceValue) {
    ZoneScopedN("CommandListPool::Recycle");
    if (fenceValue == 0) {
        PreparePairForReuse(pair);
        m_available.emplace_back(std::move(pair));
    }
    else {
        m_inFlight.emplace_back(fenceValue, std::move(pair));
    }

    m_diagnostics.availableCount = m_available.size();
    m_diagnostics.inFlightCount = m_inFlight.size();
}

void CommandListPool::RecycleCompleted(uint64_t completedFenceValue) {
    ZoneScopedN("CommandListPool::RecycleCompleted");
    while (!m_inFlight.empty() && m_inFlight.front().first <= completedFenceValue) {
        auto pair = std::move(m_inFlight.front().second);
        m_inFlight.pop_front();
        PreparePairForReuse(pair);
        m_available.push_back(std::move(pair));
    }

    m_diagnostics.availableCount = m_available.size();
    m_diagnostics.inFlightCount = m_inFlight.size();
}
