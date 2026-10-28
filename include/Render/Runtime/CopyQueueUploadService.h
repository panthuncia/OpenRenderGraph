#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
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
// barriers are recorded on every backend (Common -> CopyDest -> Common, and
// Common -> CopySource -> Common for buffer copy sources); cross-family
// correctness comes from QueueSharing::Concurrent destinations and sources.
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
    void Initialize(rhi::Device device, rhi::Queue copyQueue, Config config);
    // Not a `Config config = {}` default: GCC rejects a nested class's default member
    // initializers in a default argument of the enclosing class.
    void Initialize(rhi::Device device, rhi::Queue copyQueue) { Initialize(device, copyQueue, Config{}); }
    bool Initialized() const noexcept { return m_initialized.load(std::memory_order_acquire); }

    // Copies `totalSize` bytes gathered from `segments` to `destination` at
    // `dstOffset`. Returns a ticket that reaches Submitted when the copy list is
    // on the queue and Completed when the timeline passes it; null when the
    // service is not initialized or the arguments are invalid.
    std::shared_ptr<TrackedUploadTicket> QueueBufferUpload(
        std::span<const StreamingUploadSegment> segments, size_t totalSize,
        org::WorkerOwnedDestination destination, size_t dstOffset);

    // Copies every range to `destination` at its own dstOffset, as one entry per staging page: the page is mapped once for
    // the ranges it takes, one intake push and one wake for the call, one copy region recorded per range (a range larger than
    // a page is split as QueueBufferUpload splits). The bytes are staged before it returns. One ticket for the batch, on the
    // last entry (batches complete in order). Zero-sized ranges are skipped; null when the service is not initialized, the
    // arguments are invalid or nothing is to be copied.
    std::shared_ptr<TrackedUploadTicket> QueueBufferUploadRanges(
        std::span<const StreamingUploadRange> ranges, org::WorkerOwnedDestination destination);

    // Copies every range from `source` to `destination` on the GPU, as one entry with no staging page: one copy region per
    // range, the source held until the batch carrying it retires. It is ordered after every entry the calling thread queued
    // before it: the batch ends before a copy whose source an earlier entry of the batch writes (or whose destination an
    // earlier entry reads), since one batch has no barriers between its copies. The source must not be written while the copy
    // is queued and, like the destination, be rhi::QueueSharing::Concurrent. One ticket for the ranges. Zero-sized ranges are
    // skipped; with none left the ticket rides a copy-less entry and completes with its batch, in order. Null when the service
    // is not initialized or the arguments are invalid (the source is the destination).
    std::shared_ptr<TrackedUploadTicket> QueueBufferCopyRanges(std::shared_ptr<Resource> source,
        std::span<const StreamingCopyRange> ranges, org::WorkerOwnedDestination destination);

    // Signals `timeline` to `value` from the copy queue once every copy the calling thread queued before it is done (the
    // intake is FIFO per producer and batches go to the one queue in order; the batch carrying it signals it with its own
    // timeline). Another queue can wait on that value for those copies - before it is signalled, too (wait-before-signal), so
    // a producer that announced one must always queue it: with nothing to copy it goes out in a batch of its own. Its ticket is
    // Submitted once the signal is on the copy queue (Cancelled if its batch failed): a waiter submitted to a queue whose own
    // progress the copy queue's submission depends on (a host lock held across a blocking present) waits for that first. Null
    // when the service is not initialized or the arguments are invalid.
    std::shared_ptr<TrackedUploadTicket> QueueSignal(std::shared_ptr<rhi::TimelinePtr> timeline, uint64_t value);
    // Any thread: the batch holding the uploads the calling thread queues next waits, on the GPU, for each timeline to reach its
    // value (IUploadService::QueueStreamingWait). False without the queue.
    bool QueueWait(std::span<const std::pair<std::shared_ptr<rhi::TimelinePtr>, uint64_t>> points);

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

    // One copy region of a ranges entry (QueueBufferUploadRanges): page bytes at srcOffset to the destination at dstOffset.
    // Of a buffer copy entry (QueueBufferCopyRanges): source bytes at srcOffset.
    struct CopyRegion {
        size_t srcOffset = 0;
        size_t dstOffset = 0;
        size_t size = 0;
    };

    struct PendingCopy {
        std::shared_ptr<StagingPage> page;
        size_t srcOffset = 0;
        std::shared_ptr<Resource> destination;
        size_t dstOffset = 0;
        size_t size = 0; // a ranges entry: the sum of its regions
        // A ranges entry: its regions on `page` (srcOffset/dstOffset unused); empty: the one copy above.
        std::vector<CopyRegion> regions;
        // A buffer copy entry (QueueBufferCopyRanges): its regions read this, not a page (`page` is null).
        std::shared_ptr<Resource> source;
        std::shared_ptr<TrackedUploadTicket> ticket;
        std::chrono::steady_clock::time_point queued{};
        // A signal entry (QueueSignal): no copy, the batch signals this.
        std::shared_ptr<rhi::TimelinePtr> signalTimeline;
        uint64_t signalValue = 0;
        // A wait entry (QueueWait): no copy, the batch waits for these.
        std::vector<std::pair<std::shared_ptr<rhi::TimelinePtr>, uint64_t>> waits;
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
    // Submitter-owned: the entry that ended the last batch on a read/write hazard, first in the next one. The flag mirrors it
    // for WaitIdle.
    std::optional<PendingCopy> m_carried;
    std::atomic<bool> m_hasCarried{ false };

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
