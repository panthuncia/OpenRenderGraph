#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include <functional>

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

class IUploadService {
public:
    virtual ~IUploadService() = default;

    virtual void Initialize() = 0;
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
    virtual void ProcessDeferredReleases(uint8_t frameIndex) = 0;

    // ── Streaming upload (copy-queue path) ──────────────────────────
    virtual void QueueStreamingUpload(const void* data, size_t size,
                                      std::shared_ptr<Resource> destination,
                                      size_t dstOffset = 0) = 0;
    virtual std::shared_ptr<TrackedUploadTicket> QueueTrackedStreamingUpload(
        const void* data, size_t size, std::shared_ptr<Resource> destination,
        size_t dstOffset = 0) = 0;
    virtual std::shared_ptr<TrackedUploadTicket> QueueTrackedStreamingUploadSegments(
        std::span<const StreamingUploadSegment> segments, size_t totalSize,
        std::shared_ptr<Resource> destination, size_t dstOffset = 0) = 0;
    virtual void ResetStreamingPagePool() = 0;

    virtual void Cleanup() = 0;
};

std::shared_ptr<IUploadService> CreateDefaultUploadService();

}
