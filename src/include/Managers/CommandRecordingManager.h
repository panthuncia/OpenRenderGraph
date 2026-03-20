#pragma once
#include <array>
#include <cstdint>
#include <rhi.h>
#include "Render/CommandListPool.h"
#include "Render/QueueKind.h"

struct Signal {
    bool     enable = false;
    uint64_t value = 0; // if enable and 0, manager will pick next monotonic
};

class CommandRecordingManager {
public:
    struct Init {
        rhi::Queue* graphicsQ = nullptr;
        rhi::Timeline* graphicsF = nullptr;
        CommandListPool* graphicsPool = nullptr; // pool created with DIRECT

        rhi::Queue* computeQ = nullptr;
        rhi::Timeline* computeF = nullptr;
        CommandListPool* computePool = nullptr; // pool created with COMPUTE

        rhi::Queue* copyQ = nullptr;
        rhi::Timeline* copyF = nullptr;
        CommandListPool* copyPool = nullptr; // pool created with COPY
    };

    explicit CommandRecordingManager(const Init& init);

    // Get an open list for 'qk'. Creates one if needed, bound to 'frameEpoch'.
    rhi::CommandList EnsureOpen(QueueKind qk, uint32_t frameEpoch);

    // Close + Execute current list if dirty; optionally Signal. Returns the signaled value (or 0).
    uint64_t Flush(QueueKind qk, Signal sig = {});

    // Recycle allocators whose fences have completed (once per frame).
    void EndFrame();

    rhi::Timeline* Fence(QueueKind qk) const;
    rhi::Queue* Queue(QueueKind qk) const;

    // Last value signaled by this CRM on the given logical queue.
    uint64_t LastSignaledValue(QueueKind qk) const {
        return m_lastSignaledValue[static_cast<size_t>(qk)];
    }

    // Raise the tracked last-signaled value for a queue so that subsequent
    // auto-generated signals (e.g. cleanup Flush) start above this floor.
    // Does NOT issue a GPU signal; only adjusts the CRM's bookkeeping.
    void EnsureMinSignaledValue(QueueKind qk, uint64_t minValue) {
        const size_t idx = static_cast<size_t>(qk);
        if (minValue > m_lastSignaledValue[idx])
            m_lastSignaledValue[idx] = minValue;
    }

    void ShutdownThreadLocal();

private:
    struct QueueBinding {
        rhi::Queue* queue = nullptr;
        rhi::Timeline* fence = nullptr;
        CommandListPool* pool = nullptr;
        rhi::QueueKind listType = rhi::QueueKind::Graphics;
        bool valid() const { return queue && fence && pool; }
    };

    // One per process; alias compute -> graphics in AliasToGraphics mode dynamically
    std::array<QueueBinding, static_cast<size_t>(QueueKind::Count)> m_bind{};

    // Last value signaled by this manager per logical queue.
    std::array<uint64_t, static_cast<size_t>(QueueKind::Count)> m_lastSignaledValue{};

    //Per-thread recording state
    struct PerQueueCtx {
        rhi::CommandAllocatorPtr   alloc;
        rhi::CommandListPtr list;
        uint32_t epoch = ~0u;
        bool dirty = false;
        void reset_soft() { list.Reset(); alloc.Reset(); dirty = false; epoch = ~0u; }
    };

    struct ThreadState {
        std::array<PerQueueCtx, static_cast<size_t>(QueueKind::Count)> ctxs{};
        uint32_t cachedEpoch = ~0u; // to force rebind at new frame
    };

    static thread_local ThreadState s_tls;

};
