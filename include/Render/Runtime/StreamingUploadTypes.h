#pragma once

#include <vector>
#include <memory>
#include <cstddef>
#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>


namespace org {

struct StreamingUploadSegment {
	const void* data = nullptr;
	std::size_t size = 0;
};

class Resource;

enum class TrackedUploadTicketState : uint8_t { Queued, Claimed, Submitted, Completed, Cancelled };

struct TrackedUploadTicket {
    std::atomic<TrackedUploadTicketState> state{ TrackedUploadTicketState::Queued };
    mutable std::mutex timelineMutex;
    std::shared_ptr<const void> timelineOwner;
    uint64_t timelineValue = 0;
    std::function<bool(uint64_t)> isTimelineComplete;
    std::vector<std::function<void()>> changeCallbacks;

    void SetChangeCallback(std::function<void()> callback) {
        std::lock_guard lock(timelineMutex);
        if (callback) changeCallbacks.push_back(std::move(callback));
    }

    void NotifyChanged() const noexcept {
        std::vector<std::function<void()>> callbacks;
        {
            std::lock_guard lock(timelineMutex);
            callbacks = changeCallbacks;
        }
        for (const auto& callback : callbacks) {
            try {
                callback();
            } catch (...) {
                // Completion notification is advisory; polling remains the
                // recovery path. Never let a consumer callback terminate the
                // upload executor thread.
            }
        }
    }

    bool Cancel() noexcept {
        auto current = state.load(std::memory_order_acquire);
        while (current == TrackedUploadTicketState::Queued || current == TrackedUploadTicketState::Claimed) {
            if (state.compare_exchange_weak(current, TrackedUploadTicketState::Cancelled,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                NotifyChanged();
                return true;
            }
        }
        return current == TrackedUploadTicketState::Cancelled;
    }

    [[nodiscard]] bool Complete() {
        auto current = state.load(std::memory_order_acquire);
        if (current == TrackedUploadTicketState::Completed) return true;
        if (current != TrackedUploadTicketState::Submitted) return false;
        {
            std::lock_guard lock(timelineMutex);
            if (!isTimelineComplete || !isTimelineComplete(timelineValue)) return false;
        }
        auto expected = TrackedUploadTicketState::Submitted;
        if (state.compare_exchange_strong(expected, TrackedUploadTicketState::Completed,
                std::memory_order_acq_rel, std::memory_order_acquire)) NotifyChanged();
        return state.load(std::memory_order_acquire) == TrackedUploadTicketState::Completed;
    }
};

/// Descriptor for a single copy-queue upload consumed by a graph copy pass
/// (CLod::AsyncUpload snapshots).
struct StreamingUploadDescriptor {
    std::shared_ptr<Resource> srcUploadBuffer;   // Upload-heap page
    size_t srcOffset = 0;
    std::shared_ptr<Resource> dstResource;       // GPU-local target
    size_t dstOffset = 0;
    size_t size = 0;
    std::shared_ptr<TrackedUploadTicket> ticket;
};

// Worker-side upload destination. The producer attests that, for the whole
// write window, no queue can access any byte of the resource: it is either a
// pooled versioned backing leased from VersionedGpuBufferBackingPool::Acquire
// (pool-only owned or retired past the frame ring) or a texture that no queue
// has used yet. Live, published resources never go through this path; they are
// written by the render thread's frame upload instance, ordered by the graph.
// The resource must be created with rhi::QueueSharing::Concurrent so the copy
// queue write needs no queue-family ownership transfer on any backend.
struct WorkerOwnedDestination {
    enum class Ownership : uint8_t { PooledBackingLease, FreshTexture };
    std::shared_ptr<Resource> resource;
    Ownership ownership = Ownership::PooledBackingLease;
};


} // namespace org
