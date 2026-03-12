#include "Render/CommandListPool.h"

#include <string>

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

CommandListPair CommandListPool::Request() {
    auto assignDebugName = [&](CommandListPair& pair) {
        const uint64_t nameId = m_nextDebugNameId.fetch_add(1, std::memory_order_relaxed);
        std::string debugName = std::string("ORG ") + QueueKindDebugName(m_type) + " CommandList #" + std::to_string(nameId);
        pair.list->SetName(debugName.c_str());
    };

    if (!m_available.empty()) {
        CommandListPair pair = std::move(m_available.back());
        m_available.pop_back();
        pair.list->Recycle(pair.allocator.Get());
        assignDebugName(pair);
        return pair;
    }

    CommandListPair pair;
	auto result = m_device.CreateCommandAllocator(m_type, pair.allocator);
	result = m_device.CreateCommandList(m_type, pair.allocator.Get(), pair.list);
    pair.list->End();
    pair.allocator->Recycle();
    pair.list->Recycle(pair.allocator.Get());
    assignDebugName(pair);
    return pair;
}

void CommandListPool::Recycle(CommandListPair&& pair, uint64_t fenceValue) {
    if (fenceValue == 0) {
        pair.allocator->Recycle();
        m_available.emplace_back(std::move(pair));
    }
    else {
        m_inFlight.emplace_back(fenceValue, std::move(pair));
    }
}

void CommandListPool::RecycleCompleted(uint64_t completedFenceValue) {
    while (!m_inFlight.empty() && m_inFlight.front().first <= completedFenceValue) {
        auto pair = std::move(m_inFlight.front().second);
        m_inFlight.pop_front();
        pair.allocator->Recycle();
        m_available.push_back(std::move(pair));
    }
}
