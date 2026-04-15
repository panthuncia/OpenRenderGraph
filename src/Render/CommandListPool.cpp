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
    spdlog::info(
        "CommandListPool::PreparePairForReuse queue={} allocator={} list={} begin",
        QueueKindDebugName(m_type),
        static_cast<bool>(pair.allocator),
        static_cast<bool>(pair.list));
    pair.allocator->Recycle();
    spdlog::info("CommandListPool::PreparePairForReuse queue={} allocator recycle complete", QueueKindDebugName(m_type));
    pair.list->Recycle(pair.allocator.Get());
    spdlog::info("CommandListPool::PreparePairForReuse queue={} list recycle complete", QueueKindDebugName(m_type));
}

CommandListPair CommandListPool::CreateReadyPair() {
    ZoneScopedN("CommandListPool::CreateReadyPair");
    spdlog::info("CommandListPool::CreateReadyPair queue={} begin", QueueKindDebugName(m_type));
    CommandListPair pair;
	spdlog::info("CommandListPool::CreateReadyPair queue={} before CreateCommandAllocator", QueueKindDebugName(m_type));
	auto result = m_device.CreateCommandAllocator(m_type, pair.allocator);
	spdlog::info(
	    "CommandListPool::CreateReadyPair queue={} after CreateCommandAllocator result={} allocatorValid={}",
	    QueueKindDebugName(m_type),
	    static_cast<int>(result),
	    static_cast<bool>(pair.allocator));
	spdlog::info("CommandListPool::CreateReadyPair queue={} before CreateCommandList", QueueKindDebugName(m_type));
	result = m_device.CreateCommandList(m_type, pair.allocator.Get(), pair.list);
	spdlog::info(
	    "CommandListPool::CreateReadyPair queue={} after CreateCommandList result={} listValid={}",
	    QueueKindDebugName(m_type),
	    static_cast<int>(result),
	    static_cast<bool>(pair.list));
    (void)result;

    const uint64_t nameId = m_nextDebugNameId.fetch_add(1, std::memory_order_relaxed);
    std::string debugName = std::string("ORG ") + QueueKindDebugName(m_type) + " CommandList #" + std::to_string(nameId);
    pair.list->SetName(debugName.c_str());
    spdlog::info("CommandListPool::CreateReadyPair queue={} set debug name '{}'", QueueKindDebugName(m_type), debugName);

    spdlog::info("CommandListPool::CreateReadyPair queue={} before initial End", QueueKindDebugName(m_type));
    pair.list->End();
    spdlog::info("CommandListPool::CreateReadyPair queue={} after initial End", QueueKindDebugName(m_type));
    PreparePairForReuse(pair);
    spdlog::info("CommandListPool::CreateReadyPair queue={} complete", QueueKindDebugName(m_type));
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
    spdlog::info(
        "CommandListPool::PrepareForRequests queue={} begin required={} completedFence={} available={} inFlight={}",
        QueueKindDebugName(m_type),
        requiredCount,
        completedFenceValue,
        m_available.size(),
        m_inFlight.size());
    m_diagnostics.lastRequestedCount = requiredCount;
    m_diagnostics.createdThisFrame = 0;
    m_diagnostics.reusedThisFrame = 0;
    m_diagnostics.preparedDeficit = 0;

    spdlog::info("CommandListPool::PrepareForRequests queue={} before RecycleCompleted", QueueKindDebugName(m_type));
    RecycleCompleted(completedFenceValue);
    spdlog::info(
        "CommandListPool::PrepareForRequests queue={} after RecycleCompleted available={} inFlight={}",
        QueueKindDebugName(m_type),
        m_available.size(),
        m_inFlight.size());

    if (m_available.size() < requiredCount) {
        const size_t deficit = requiredCount - m_available.size();
        m_diagnostics.preparedDeficit = deficit;
        m_available.reserve(requiredCount);
        for (size_t i = 0; i < deficit; ++i) {
            spdlog::info(
                "CommandListPool::PrepareForRequests queue={} creating ready pair {} of {}",
                QueueKindDebugName(m_type),
                i + 1,
                deficit);
            m_available.emplace_back(CreateReadyPair());
            ++m_diagnostics.createdThisFrame;
            spdlog::info(
                "CommandListPool::PrepareForRequests queue={} created ready pair {} of {} available={} inFlight={}",
                QueueKindDebugName(m_type),
                i + 1,
                deficit,
                m_available.size(),
                m_inFlight.size());
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
    spdlog::info(
        "CommandListPool::PrepareForRequests queue={} complete available={} inFlight={} createdThisFrame={} reusedThisFrame={}",
        QueueKindDebugName(m_type),
        m_available.size(),
        m_inFlight.size(),
        m_diagnostics.createdThisFrame,
        m_diagnostics.reusedThisFrame);
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
    spdlog::info(
        "CommandListPool::RecycleCompleted queue={} begin completedFence={} inFlight={} available={}",
        QueueKindDebugName(m_type),
        completedFenceValue,
        m_inFlight.size(),
        m_available.size());
    while (!m_inFlight.empty() && m_inFlight.front().first <= completedFenceValue) {
        auto pair = std::move(m_inFlight.front().second);
        m_inFlight.pop_front();
        PreparePairForReuse(pair);
        m_available.push_back(std::move(pair));
    }

    m_diagnostics.availableCount = m_available.size();
    m_diagnostics.inFlightCount = m_inFlight.size();
    spdlog::info(
        "CommandListPool::RecycleCompleted queue={} complete inFlight={} available={}",
        QueueKindDebugName(m_type),
        m_inFlight.size(),
        m_available.size());
}
