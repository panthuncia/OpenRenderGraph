#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include <functional>
#include <stdexcept>

#include <rhi.h>
#include <rhi_helpers.h>

#include "Render/ResourceRegistry.h"
#include "Render/Runtime/UploadTypes.h"
#include "Render/Runtime/StreamingUploadTypes.h"

namespace org {
class ResourceRegistry;
class RenderPass;
class Resource;
}

namespace org::runtime {

class StagedUploadBatch;

class IUploadService {
public:
    virtual ~IUploadService() = default;

    virtual void Initialize() = 0;
    // Marks the calling thread as the owner of the frame (in-graph) upload path.
    virtual void SetOwnerThread() {}
    virtual void SetUploadResolveContext(UploadResolveContext context) = 0;
    virtual std::shared_ptr<RenderPass> GetUploadPass() const = 0;

#if BUILD_TYPE == BUILD_TYPE_DEBUG
    virtual void UploadData(const void* data, size_t size, UploadTarget resourceToUpdate, size_t dataBufferOffset, const char* file, int line) = 0;
    virtual void UploadTextureSubresources(
        UploadTarget target,
        rhi::Format fmt,
        uint32_t baseWidth,
        uint32_t baseHeight,
        uint32_t depthOrLayers,
        uint32_t mipLevels,
        uint32_t arraySize,
        const rhi::helpers::SubresourceData* srcSubresources,
        uint32_t srcCount,
        const char* file,
        int line) = 0;
#else
    virtual void UploadData(const void* data, size_t size, UploadTarget resourceToUpdate, size_t dataBufferOffset) = 0;
    virtual void UploadTextureSubresources(
        UploadTarget target,
        rhi::Format fmt,
        uint32_t baseWidth,
        uint32_t baseHeight,
        uint32_t depthOrLayers,
        uint32_t mipLevels,
        uint32_t arraySize,
        const rhi::helpers::SubresourceData* srcSubresources,
        uint32_t srcCount) = 0;
#endif

    // Texture upload whose source bytes are owned by the caller through
    // `keepAlive`. On the owner (render) thread this is UploadTextureSubresources;
    // from any other thread the request is posted to the owner's mailbox and
    // recorded by the next frame's upload pass, so producers never touch the
    // frame upload queue. Live textures published to consumers are graph-ordered
    // this way; only never-used, concurrent textures may use the worker service.
#if BUILD_TYPE == BUILD_TYPE_DEBUG
    virtual void PostTextureSubresources(UploadTarget target, rhi::Format fmt, uint32_t baseWidth, uint32_t baseHeight,
        uint32_t depthOrLayers, uint32_t mipLevels, uint32_t arraySize,
        std::shared_ptr<const std::vector<rhi::helpers::SubresourceData>> subresources,
        std::shared_ptr<const void> keepAlive, const char* file, int line) {
        (void)keepAlive;
        if (!subresources) return;
        UploadTextureSubresources(std::move(target), fmt, baseWidth, baseHeight, depthOrLayers, mipLevels, arraySize,
            subresources->data(), static_cast<uint32_t>(subresources->size()), file, line);
    }
#else
    virtual void PostTextureSubresources(UploadTarget target, rhi::Format fmt, uint32_t baseWidth, uint32_t baseHeight,
        uint32_t depthOrLayers, uint32_t mipLevels, uint32_t arraySize,
        std::shared_ptr<const std::vector<rhi::helpers::SubresourceData>> subresources,
        std::shared_ptr<const void> keepAlive) {
        (void)keepAlive;
        if (!subresources) return;
        UploadTextureSubresources(std::move(target), fmt, baseWidth, baseHeight, depthOrLayers, mipLevels, arraySize,
            subresources->data(), static_cast<uint32_t>(subresources->size()));
    }
#endif

    // Upload several regions of one target as a single queue entry: one telemetry
    // capture, one upload-heap allocation, one map/unmap. Regions must be sorted
    // by dstOffset and must not overlap. The default forwards region by region.
#if BUILD_TYPE == BUILD_TYPE_DEBUG
    virtual void UploadDataBatch(UploadTarget resourceToUpdate, std::span<const UploadRegion> regions, const char* file, int line) {
        for (const auto& region : regions) UploadData(region.data, region.size, resourceToUpdate, region.dstOffset, file, line);
    }
#else
    virtual void UploadDataBatch(UploadTarget resourceToUpdate, std::span<const UploadRegion> regions) {
        for (const auto& region : regions) UploadData(region.data, region.size, resourceToUpdate, region.dstOffset);
    }
#endif

    virtual void QueueResourceCopy(const std::shared_ptr<Resource>& destination, const std::shared_ptr<Resource>& source, size_t size) = 0;

    // Owner thread: queues a batch a producer staged (StagedUploadBatch) as this frame's uploads, in order after
    // the uploads queued before it, without copying its bytes. The service holds the batch until the GPU is
    // done with the frame slot that records it.
    virtual void SubmitStagedUploads(std::shared_ptr<StagedUploadBatch> batch) {
        (void)batch;
        throw std::logic_error("This upload service does not take staged uploads");
    }
    // Owner thread. Direct: staged batches wait for RecordStagedUploads instead of joining the upload pass's
    // queue - for a host that records the frame's uploads itself (RenderGraph::RecordPendingUploads), where
    // every entry is then one copy with no further bookkeeping. Switching it off queues what is waiting.
    virtual void SetStagedUploadsRecordedDirectly(bool direct) { (void)direct; }
    // Owner thread: records the waiting staged batches' copies into `list` (whose frame slot is frameIndex:
    // the batches live until it retires), in submission order: a copy that overlaps one recorded before it in
    // these batches waits for it. afterWork: the copies follow work they must wait for (copies earlier in the
    // list, or the previous execution's): one full barrier ahead of the first, recorded only when there is a
    // copy, so an epoch with no batch pays nothing. Returns the number of copies.
    virtual size_t RecordStagedUploads(rhi::CommandList& list, uint8_t frameIndex, bool afterWork) {
        (void)list; (void)frameIndex; (void)afterWork;
        return 0;
    }
    // Owner thread, direct mode: queues a batch its producer already recorded (StagedUploadBatch::Record), whose list
    // the host submits as it is, ahead of the copies it records itself - so only when that is the batch's place: nothing
    // staged, queued or posted before it is waiting, and no backing was released since it was recorded. False leaves the
    // batch with the caller, which submits it as staged (SubmitStagedUploads) instead.
    virtual bool SubmitRecordedUploads(std::shared_ptr<StagedUploadBatch> batch) {
        (void)batch;
        return false;
    }
    // Owner thread: the accepted recorded batches' lists, in order, for frame slot frameIndex (the batches live until it
    // retires). Call before RecordStagedUploads: a batch whose recording a backing release has since outdated goes back
    // to be recorded there, first, and none is returned.
    virtual size_t TakeRecordedUploads(std::vector<rhi::CommandList>& out, uint8_t frameIndex) {
        (void)out; (void)frameIndex;
        return 0;
    }
    // Where the uploads go, for a host that splits them between queues (RenderGraph::RecordPendingUploads' queue-scoped uploads).
    // TakeRecordedUploads, also appending to `destinations` every buffer backing the taken lists write. False when that is not
    // known (the default, unless nothing was taken): the caller then treats the lists as writing anything.
    virtual bool TakeRecordedUploadsWithDestinations(std::vector<rhi::CommandList>& out, uint8_t frameIndex,
        std::vector<rhi::ResourceHandle>& destinations) {
        (void)destinations;
        const size_t before = out.size();
        TakeRecordedUploads(out, frameIndex);
        return out.size() == before;
    }
    // Owner thread, direct mode: appends the destination backing of every copy RecordStagedUploads would record now, one per copy
    // in recording order. False when one is not known (the default): the caller then records them with RecordStagedUploads.
    virtual bool StagedUploadDestinations(std::vector<rhi::ResourceHandle>& out) const {
        (void)out;
        return false;
    }
    // RecordStagedUploads, with each copy recorded into the list `route` names for its destination backing (null: `list`) instead:
    // the copies of one destination all go to the same list, in order. Each list that takes a copy starts with afterWork's barrier.
    // Returns the copies recorded in all; routed counts those that went elsewhere than `list`. Only a service whose
    // StagedUploadDestinations answers routes (the default records everything into `list`).
    virtual size_t RecordStagedUploadsRouted(rhi::CommandList& list, uint8_t frameIndex, bool afterWork,
        const std::function<rhi::CommandList*(rhi::ResourceHandle)>& route, size_t& routed) {
        (void)route;
        routed = 0;
        return RecordStagedUploads(list, frameIndex, afterWork);
    }
    virtual void ProcessDeferredReleases(uint8_t frameIndex) = 0;
    // Owner thread: ProcessDeferredReleases, except that the GPU objects retiring at this step are handed back instead of
    // destroyed here. They are already safe to destroy; they go when the returned owner is dropped, on the thread that
    // drops it - for a host whose render thread must not pay for driver frees.
    virtual std::shared_ptr<void> ProcessDeferredReleasesRetiringElsewhere(uint8_t frameIndex) {
        ProcessDeferredReleases(frameIndex);
        return {};
    }

    // ── Worker upload path (copy queue, CopyQueueUploadService) ──────
    // Never touches the frame upload instance: bytes are staged into the
    // worker service's pages and copied on the copy queue; completion is
    // observed only through the ticket. See WorkerOwnedDestination for the
    // ownership rule the destination must satisfy.
    virtual std::shared_ptr<TrackedUploadTicket> QueueTrackedStreamingUploadSegments(
        std::span<const StreamingUploadSegment> segments, size_t totalSize,
        WorkerOwnedDestination destination, size_t dstOffset = 0) = 0;
    // The same path, many ranges of one destination in one call (CopyQueueUploadService::QueueBufferUploadRanges): one
    // ticket for the batch, completed after every range. The bytes are staged before it returns. The default forwards range
    // by range and returns the last range's ticket (or the first that failed: null or cancelled).
    virtual std::shared_ptr<TrackedUploadTicket> QueueTrackedStreamingUploadRanges(
        std::span<const StreamingUploadRange> ranges, WorkerOwnedDestination destination) {
        std::shared_ptr<TrackedUploadTicket> ticket;
        for (const auto& range : ranges) {
            if (range.size == 0) continue;
            const StreamingUploadSegment segment{ range.data, range.size };
            ticket = QueueTrackedStreamingUploadSegments({ &segment, 1 }, segment.size, destination, range.dstOffset);
            if (!ticket || ticket->state.load(std::memory_order_acquire) == TrackedUploadTicketState::Cancelled) return ticket;
        }
        return ticket;
    }
    // Whether that path has a queue of its own (the dedicated uploader of graph work). Without one its copies go to the graph's
    // copy queue, which a host may alias to its graphics queue: a producer that relies on running beside the frame checks this.
    virtual bool HasDedicatedStreamingQueue() const { return false; }
    // Any thread: signals `timeline` to `value` on that path's queue once every upload the calling thread queued on it before
    // is done (CopyQueueUploadService::QueueSignal). Another queue may wait on the value before it is signalled, so a producer
    // that announced one always queues it. False when the path has no queue of its own to signal from.
    virtual bool QueueStreamingSignal(std::shared_ptr<rhi::TimelinePtr> timeline, uint64_t value) {
        (void)timeline; (void)value;
        return false;
    }

    virtual void Cleanup() = 0;
};

std::shared_ptr<IUploadService> CreateDefaultUploadService();

}
