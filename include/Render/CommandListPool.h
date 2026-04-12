#pragma once

#include <cstddef>
#include <deque>
#include <vector>
#include <cstdint>
#include <atomic>
#include <rhi.h>

struct CommandListPair {
    rhi::CommandAllocatorPtr allocator;
    rhi::CommandListPtr list;
};

class CommandListPool {
public:
    struct Diagnostics {
        size_t lastRequestedCount = 0;
        size_t availableCount = 0;
        size_t inFlightCount = 0;
        size_t createdThisFrame = 0;
        size_t reusedThisFrame = 0;
        size_t preparedDeficit = 0;
    };

    CommandListPool(rhi::Device& device, rhi::QueueKind type);

    // Acquire a command allocator / list pair ready for recording
    CommandListPair Request();

    // Reclaim completed pairs and ensure at least requiredCount are available
    // for immediate Request() calls.
    void PrepareForRequests(size_t requiredCount, uint64_t completedFenceValue);

    // Recycle a pair after execution. If fenceValue is 0 the pair becomes
    // immediately available. Otherwise it will be returned to the available
    // pool once RecycleCompleted is called with a sufficiently large fence value.
    void Recycle(CommandListPair&& pair, uint64_t fenceValue);

    // Return any completed command lists to the available pool.
    void RecycleCompleted(uint64_t completedFenceValue);

    const Diagnostics& GetDiagnostics() const noexcept { return m_diagnostics; }

private:
    void PreparePairForReuse(CommandListPair& pair);
    CommandListPair CreateReadyPair();

    rhi::Device m_device;
    rhi::QueueKind m_type;
    std::atomic<uint64_t> m_nextDebugNameId{ 1 };
    Diagnostics m_diagnostics{};

    std::vector<CommandListPair> m_available;
    std::deque<std::pair<uint64_t, CommandListPair>> m_inFlight;
};
