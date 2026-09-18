#include "Render/Runtime/IUploadService.h"

#include "Managers/Singletons/DeletionManager.h"
#include "Managers/Singletons/UploadManager.h"

namespace org::runtime {

namespace {
class DefaultUploadService final : public IUploadService {
public:
    void Initialize() override {
        UploadManager::GetInstance().Initialize();
    }

    void SetOwnerThread() override {
        UploadManager::GetInstance().SetOwnerThread();
    }

    void SetUploadResolveContext(UploadResolveContext context) override {
        UploadManager::GetInstance().SetUploadResolveContext(context);
    }

    std::shared_ptr<RenderPass> GetUploadPass() const override {
        return UploadManager::GetInstance().GetUploadPass();
    }

#if BUILD_TYPE == BUILD_TYPE_DEBUG
    void UploadData(const void* data, size_t size, UploadTarget resourceToUpdate, size_t dataBufferOffset, const char* file, int line) override {
        UploadManager::GetInstance().UploadData(data, size, std::move(resourceToUpdate), dataBufferOffset, file, line);
    }

    void UploadDataBatch(UploadTarget resourceToUpdate, std::span<const UploadRegion> regions, const char* file, int line) override {
        UploadManager::GetInstance().UploadDataBatch(std::move(resourceToUpdate), regions, file, line);
    }

    void PostTextureSubresources(UploadTarget target, rhi::Format fmt, uint32_t baseWidth, uint32_t baseHeight,
        uint32_t depthOrLayers, uint32_t mipLevels, uint32_t arraySize,
        std::shared_ptr<const std::vector<rhi::helpers::SubresourceData>> subresources,
        std::shared_ptr<const void> keepAlive, const char* file, int line) override {
        UploadManager::GetInstance().PostTextureSubresources(std::move(target), fmt, baseWidth, baseHeight,
            depthOrLayers, mipLevels, arraySize, std::move(subresources), std::move(keepAlive), file, line);
    }

    void UploadTextureSubresources(
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
        int line) override {
        UploadManager::GetInstance().UploadTextureSubresources(
            std::move(target),
            fmt,
            baseWidth,
            baseHeight,
            depthOrLayers,
            mipLevels,
            arraySize,
            srcSubresources,
            srcCount,
            file,
            line);
    }
#else
    void UploadData(const void* data, size_t size, UploadTarget resourceToUpdate, size_t dataBufferOffset) override {
        UploadManager::GetInstance().UploadData(data, size, std::move(resourceToUpdate), dataBufferOffset);
    }

    void UploadDataBatch(UploadTarget resourceToUpdate, std::span<const UploadRegion> regions) override {
        UploadManager::GetInstance().UploadDataBatch(std::move(resourceToUpdate), regions);
    }

    void PostTextureSubresources(UploadTarget target, rhi::Format fmt, uint32_t baseWidth, uint32_t baseHeight,
        uint32_t depthOrLayers, uint32_t mipLevels, uint32_t arraySize,
        std::shared_ptr<const std::vector<rhi::helpers::SubresourceData>> subresources,
        std::shared_ptr<const void> keepAlive) override {
        UploadManager::GetInstance().PostTextureSubresources(std::move(target), fmt, baseWidth, baseHeight,
            depthOrLayers, mipLevels, arraySize, std::move(subresources), std::move(keepAlive));
    }

    void UploadTextureSubresources(
        UploadTarget target,
        rhi::Format fmt,
        uint32_t baseWidth,
        uint32_t baseHeight,
        uint32_t depthOrLayers,
        uint32_t mipLevels,
        uint32_t arraySize,
        const rhi::helpers::SubresourceData* srcSubresources,
        uint32_t srcCount) override {
        UploadManager::GetInstance().UploadTextureSubresources(
            std::move(target),
            fmt,
            baseWidth,
            baseHeight,
            depthOrLayers,
            mipLevels,
            arraySize,
            srcSubresources,
            srcCount);
    }
#endif

    void QueueResourceCopy(const std::shared_ptr<Resource>& destination, const std::shared_ptr<Resource>& source, size_t size) override {
        UploadManager::GetInstance().QueueResourceCopy(destination, source, size);
    }

    void ProcessDeferredReleases(uint8_t frameIndex) override {
        UploadManager::GetInstance().ProcessDeferredReleases(frameIndex);
        DeletionManager::GetInstance().ProcessDeletions();
    }

    std::shared_ptr<TrackedUploadTicket> QueueTrackedStreamingUploadSegments(
        std::span<const StreamingUploadSegment> segments, size_t totalSize,
        WorkerOwnedDestination destination, size_t dstOffset) override {
        return UploadManager::GetInstance().QueueTrackedStreamingUploadSegments(
            segments, totalSize, std::move(destination), dstOffset);
    }

    void Cleanup() override {
        UploadManager::GetInstance().Cleanup();
    }
};
}

std::shared_ptr<IUploadService> CreateDefaultUploadService() {
    return std::make_shared<DefaultUploadService>();
}

}
