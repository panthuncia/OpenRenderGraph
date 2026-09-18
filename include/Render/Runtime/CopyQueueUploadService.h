#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <rhi.h>

#include "Render/Runtime/MpscQueue.h"
#include "Render/Runtime/StreamingUploadTypes.h"

namespace org { class Resource; }

namespace org::runtime {

// Copy-queue upload service shared by every worker thread. Producers stage
// their bytes into a persistently mapped upload page (one active page per
// producer thread, bump allocated, no lock) and push a copy descriptor onto a
// lock-free MPSC queue. One submitter thread batches descriptors into copy
// lists, submits them on the copy queue with its own timeline, keeps several
// batches in flight, and completes tickets by polling the timeline. Pages
// recycle when every region they carried has completed on the GPU. The same
// barriers are recorded on every backend (Common -> CopyDest -> Common);
// cross-family correctness comes from QueueSharing::Concurrent destinations.
class CopyQueueUploadService {
public:
    struct Config {
        size_t pageBytes = size_t{ 16 } << 20;
        size_t maxInFlightBatches = 8;
        size_t maxBatchBytes = size_t{ 16 } << 20;
        size_t maxBatchCopies = 256;
        size_t maxFreePages = 8;
        std::string debugName = "CopyQueueUploads";
    };

    struct Stats {
        uint64_t pagesCreated = 0;
        uint64_t pagesRecycled = 0;
        uint64_t pagesDropped = 0;
        uint64_t batchesSubmitted = 0;
        uint64_t copiesSubmitted = 0;
        uint64_t bytesSubmitted = 0;
        uint64_t copiesCancelled = 0;
        uint64_t queuedCopies = 0;
        uint64_t inFlightBatches = 0;
        uint64_t freePages = 0;
        uint64_t closedPages = 0;
    };

    CopyQueueUploadService();
    ~CopyQueueUploadService();
    CopyQueueUploadService(const CopyQueueUploadService&) = delete;
    CopyQueueUploadService& operator=(const CopyQueueUploadService&) = delete;

    // The queue must be a copy queue of `device`. Idempotent.
    void Initialize(rhi::Device device, rhi::Queue copyQueue, Config config = {});
    bool Initialized() const noexcept { return m_initialized.load(std::memory_order_acquire); }

    // Copies `totalSize` bytes gathered from `segments` to `destination` at
    // `dstOffset`. Returns a ticket that reaches Submitted when the copy list is
    // on the queue and Completed when the timeline passes it; null when the
    // service is not initialized or the arguments are invalid.
    std::shared_ptr<TrackedUploadTicket> QueueBufferUpload(
        std::span<const StreamingUploadSegment> segments, size_t totalSize,
        org::WorkerOwnedDestination destination, size_t dstOffset);

    // Blocks until nothing is queued or in flight (tests and shutdown).
    bool WaitIdle(uint32_t timeoutMs = UINT32_MAX);

    Stats GetStats() const;

    // Stops the submitter, cancels queued tickets, waits for in-flight batches
    // and releases every page. Idempotent.
    void Cleanup();

private:
    struct StagingPage {
        rhi::ResourcePtr buffer;
        size_t capacity = 0;
        size_t tail = 0; // owner (producer) thread only while open
        bool dedicated = false;
        uint64_t index = 0;
        std::atomic<uint32_t> pendingRegions{ 0 };
        std::atomic<bool> closed{ false };
    };

    struct PendingCopy {
        std::shared_ptr<StagingPage> page;
        size_t srcOffset = 0;
        std::shared_ptr<Resource> destination;
        size_t dstOffset = 0;
        size_t size = 0;
        std::shared_ptr<TrackedUploadTicket> ticket;
        std::chrono::steady_clock::time_point queued{};
    };

    struct CommandPair {
        rhi::CommandAllocatorPtr allocator;
        rhi::CommandListPtr list;
    };

    struct Batch {
        CommandPair pair;
        std::vector<PendingCopy> copies;
        uint64_t timelineValue = 0;
        size_t bytes = 0;
    };

    struct ThreadSlot {
        uint64_t epoch = 0;
        std::shared_ptr<StagingPage> page;
    };


    std::shared_ptr<StagingPage> CreatePage(size_t capacity, bool dedicated);
    std::shared_ptr<StagingPage> AcquirePage();
    void ClosePage(std::shared_ptr<StagingPage> page);
    bool StageChunk(std::span<const StreamingUploadSegment> segments, size_t& segmentIndex, size_t& segmentOffset,
        size_t chunkBytes, std::shared_ptr<StagingPage>& outPage, size_t& outOffset);
    void SubmitterMain(std::stop_token stopToken);
    bool CollectBatch(Batch& batch);
    bool RecordAndSubmit(Batch& batch);
    void CompleteFinishedBatches(uint64_t completedValue);
    void FinishBatch(Batch& batch, bool cancelled);
    void RetirePages();
    CommandPair AcquireCommandPair();
    void RecycleCommandPair(CommandPair&& pair);
    ThreadSlot& Slot();

    rhi::Device m_device{};
    rhi::Queue m_copyQueue{};
    Config m_config{};
    std::atomic<bool> m_initialized{ false };
    std::atomic<uint64_t> m_epoch{ 1 };

    std::shared_ptr<rhi::TimelinePtr> m_timeline;
    std::atomic<uint64_t> m_nextTimelineValue{ 0 };
    std::atomic<uint64_t> m_lastSubmittedValue{ 0 };

    // Lock-free MPSC intake: producers push, the submitter pops.
    MpscQueue<PendingCopy> m_intake;

    // Page lists are touched once per page (16 MB), never per upload.
    mutable std::mutex m_pageMutex;
    std::vector<std::shared_ptr<StagingPage>> m_freePages;
    std::vector<std::shared_ptr<StagingPage>> m_closedPages;
    std::vector<std::weak_ptr<StagingPage>> m_allPages;
    std::atomic<uint64_t> m_nextPageIndex{ 0 };

    std::jthread m_submitter;
    std::mutex m_wakeMutex;
    std::condition_variable_any m_wake;
    std::condition_variable_any m_idle;
    std::deque<Batch> m_inFlight;        // submitter-owned
    std::vector<CommandPair> m_freePairs; // submitter-owned
    std::atomic<uint64_t> m_inFlightCount{ 0 };

    mutable std::mutex m_statsMutex;
    Stats m_stats{};
};

} // namespace org::runtime
