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
    // the batches live until it retires), in submission order: a copy that overlaps one recorded before it -
    // in these batches, or among the list's copies before them when afterCopies - waits for it. Returns the
    // number of copies; the caller orders them against everything else.
    virtual size_t RecordStagedUploads(rhi::CommandList& list, uint8_t frameIndex, bool afterCopies) {
        (void)list; (void)frameIndex; (void)afterCopies;
        return 0;
    }
    virtual void ProcessDeferredReleases(uint8_t frameIndex) = 0;

    // ── Worker upload path (copy queue, CopyQueueUploadService) ──────
    // Never touches the frame upload instance: bytes are staged into the
    // worker service's pages and copied on the copy queue; completion is
    // observed only through the ticket. See WorkerOwnedDestination for the
    // ownership rule the destination must satisfy.
    virtual std::shared_ptr<TrackedUploadTicket> QueueTrackedStreamingUploadSegments(
        std::span<const StreamingUploadSegment> segments, size_t totalSize,
        WorkerOwnedDestination destination, size_t dstOffset = 0) = 0;

    virtual void Cleanup() = 0;
};

std::shared_ptr<IUploadService> CreateDefaultUploadService();

}
