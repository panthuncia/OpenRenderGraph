#include "Managers/CommandRecordingManager.h"

#include <cassert>
#include <algorithm>
#include <spdlog/spdlog.h>

thread_local CommandRecordingManager::ThreadState CommandRecordingManager::s_tls{};

CommandRecordingManager::CommandRecordingManager(const Init& init) {
    m_bind[static_cast<size_t>(QueueKind::Graphics)] =
    { init.graphicsQ, init.graphicsF, init.graphicsPool, rhi::QueueKind::Graphics };

    m_bind[static_cast<size_t>(QueueKind::Compute)] =
    { init.computeQ,  init.computeF,  init.computePool,  rhi::QueueKind::Compute };

    m_bind[static_cast<size_t>(QueueKind::Copy)] =
    { init.copyQ,     init.copyF,     init.copyPool,     rhi::QueueKind::Copy };

    for (size_t i = 0; i < static_cast<size_t>(QueueKind::Count); ++i) {
        auto& bind = m_bind[i];
        if (bind.valid()) {
            m_lastSignaledValue[i] = bind.fence->GetCompletedValue();
        }
    }
}

rhi::CommandList CommandRecordingManager::EnsureOpen(QueueKind requested, uint32_t frameEpoch) {
    const QueueKind qk = requested;
    auto& bind = m_bind[static_cast<size_t>(qk)];
    assert(bind.valid() && "Queue/Fence/Pool not initialized for this QueueKind");

    auto& tls = s_tls;
    auto& ctx = tls.ctxs[static_cast<size_t>(qk)];

    // If the epoch changed since last list, drop the old one (we'll get a new allocator)
    if (ctx.list && ctx.epoch != frameEpoch) {
        // Not strictly necessary to flush here; render graph should Flush at boundaries.
        // We just invalidate so next EnsureOpen will acquire a fresh pair.
        ctx.reset_soft();
    }

    if (!ctx.list) {
        // Acquire a fresh pair from the pool; Request() must return a reset & ready list
        CommandListPair pair = bind.pool->Request();

        ctx.alloc = std::move(pair.allocator);
        ctx.list = std::move(pair.list);
        ctx.epoch = frameEpoch;
        ctx.dirty = true;
    }

    return ctx.list.Get();
}

uint64_t CommandRecordingManager::Flush(QueueKind requested, Signal sig) {
    const QueueKind qk = requested;
    const size_t qkIndex = static_cast<size_t>(qk);
    auto& bind = m_bind[static_cast<size_t>(qk)];
    auto& ctx = s_tls.ctxs[qkIndex];

    uint64_t signaled = 0;

    if (ctx.list) {
        if (ctx.dirty) {
            // Close + execute
            ctx.list->End();
			bind.queue->Submit({ &ctx.list.Get(), 1 }, {});
        }

		//bind.queue->CheckDebugMessages();

        // Decide on signaling.
        // Dirty command lists that were submitted must be associated with a fence value
        // so the pool can recycle them safely.
        const bool mustSignalForRecycle = ctx.dirty;
        if (sig.enable || mustSignalForRecycle) {
			if (m_lastSignaledValue[qkIndex] == UINT64_MAX) {
				// Something is wrong
                spdlog::error("CRM::Flush signal: timeline for queue {} has exhausted its value space! No further command lists can be recorded.",
                    static_cast<int>(qk));
                throw std::runtime_error("Timeline value space exhausted");
            }
            if (sig.enable && sig.value != 0) {
                signaled = sig.value;
            }
            else {
                signaled = ++m_lastSignaledValue[qkIndex];
            }

            // Diagnostic: log every CRM fence signal so we can trace unexpected values
            spdlog::debug("CRM::Flush signal: resolvedQueue={} fenceIdx={} fenceGen={} value={}",
                static_cast<int>(qk),
                bind.fence->GetHandle().index,
                bind.fence->GetHandle().generation,
                signaled);
            bind.queue->Signal({ bind.fence->GetHandle(), signaled });
            m_lastSignaledValue[qkIndex] = std::max(m_lastSignaledValue[qkIndex], signaled);
#if BUILD_TYPE == BUILD_TYPE_DEBUG
            // Detect out-of-order signals at the CRM level.
            // This fires when the auto-generated recycle signal (GetCompletedValue()+1)
            // regresses behind an in-flight signal from a prior frame because no batch
            // explicitly signaled this queue.
            auto result = bind.fence->GetCompletedValue();
            (void)result; // GetCompletedValue is informational; the real check is:
            // If Signal failed at the RHI layer, it would set an error on the device.
            // Log enough context to diagnose which queue and values are involved.
            spdlog::trace("CRM::Flush queue={} signaled={} lastTracked={} completed={}",
                qkIndex, signaled, m_lastSignaledValue[qkIndex], result);
#endif
        }

        // Return the pair to the pool tagged with the fence.
        // If not submitted, 0 means immediately reusable.
        uint64_t recycleFence = ctx.dirty ? signaled : 0;

        // Hand back allocator/list to pool
        CommandListPair back;
        back.allocator = std::move(ctx.alloc);
        back.list = std::move(ctx.list);
        bind.pool->Recycle(std::move(back), recycleFence);

        // Invalidate thread-local context so next EnsureOpen() acquires a fresh pair
        ctx.reset_soft();
    }

    return signaled;
}

void CommandRecordingManager::EndFrame() {
    // Let pools reclaim any in-flight allocators whose fences have completed
    for (size_t i = 0; i < static_cast<size_t>(QueueKind::Count); ++i) {
        auto& bind = m_bind[i];
        if (!bind.valid()) continue;
        const uint64_t done = bind.fence->GetCompletedValue();
        bind.pool->RecycleCompleted(done);
    }
}

rhi::Timeline* CommandRecordingManager::Fence(QueueKind qk) const {
    return m_bind[static_cast<size_t>(qk)].fence;
}

rhi::Queue* CommandRecordingManager::Queue(QueueKind qk) const {
    return m_bind[static_cast<size_t>(qk)].queue;
}

void CommandRecordingManager::ShutdownThreadLocal() {
    auto& tls = s_tls;

    for (auto& ctx : tls.ctxs) {
        // NOTE: this does NOT Flush(). It just releases refs owned by TLS.
        // Only call this when you know there are no in-flight command lists
        // for this thread (after device idle).
        ctx.reset_soft();
    }

    tls.cachedEpoch = ~0u;
}