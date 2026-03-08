#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <rhi.h>
#include <rhi_helpers.h>

#include "Render/ResourceRegistry.h"
#include "Render/Runtime/UploadTypes.h"
#include "Render/Runtime/StreamingUploadTypes.h"

class ResourceRegistry;
class RenderPass;
class Resource;

namespace rg::runtime {

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

    virtual void QueueResourceCopy(const std::shared_ptr<Resource>& destination, const std::shared_ptr<Resource>& source, size_t size) = 0;
    virtual void ProcessDeferredReleases(uint8_t frameIndex) = 0;

    // ── Streaming upload (copy-queue path) ──────────────────────────
    virtual void QueueStreamingUpload(const void* data, size_t size,
                                      std::shared_ptr<Resource> destination,
                                      size_t dstOffset = 0) = 0;
    virtual std::vector<StreamingUploadDescriptor> ConsumeStreamingUploads() = 0;
    virtual void ResetStreamingPagePool() = 0;

    virtual void Cleanup() = 0;
};

std::shared_ptr<IUploadService> CreateDefaultUploadService();

}
