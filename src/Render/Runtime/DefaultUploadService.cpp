#include "Render/Runtime/IUploadService.h"

#include "Managers/Singletons/UploadManager.h"

namespace org::runtime {

namespace {
class DefaultUploadService final : public IUploadService {
public:
    DefaultUploadService() = default;
    // A graph's own (CreateGraphUploadService).
    explicit DefaultUploadService(std::unique_ptr<UploadManager> owned) : m_owned(std::move(owned)) {}

    void Initialize() override {
        Manager().Initialize();
    }

    void SetOwnerThread() override {
        Manager().SetOwnerThread();
    }

    void SetUploadResolveContext(UploadResolveContext context) override {
        Manager().SetUploadResolveContext(context);
    }

    std::shared_ptr<RenderPass> GetUploadPass() const override {
        return Manager().GetUploadPass();
    }

    void UploadData(const void* data, size_t size, UploadTarget resourceToUpdate, size_t dataBufferOffset, const char* file, int line) override {
        Manager().UploadData(data, size, std::move(resourceToUpdate), dataBufferOffset, file, line);
    }

    void UploadDataBatch(UploadTarget resourceToUpdate, std::span<const UploadRegion> regions, const char* file, int line) override {
        Manager().UploadDataBatch(std::move(resourceToUpdate), regions, file, line);
    }

    void PostTextureSubresources(UploadTarget target, rhi::Format fmt, uint32_t baseWidth, uint32_t baseHeight,
        uint32_t depthOrLayers, uint32_t mipLevels, uint32_t arraySize,
        std::shared_ptr<const std::vector<rhi::helpers::SubresourceData>> subresources,
        std::shared_ptr<const void> keepAlive, const char* file, int line) override {
        Manager().PostTextureSubresources(std::move(target), fmt, baseWidth, baseHeight,
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
        Manager().UploadTextureSubresources(
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

    void QueueResourceCopy(const std::shared_ptr<Resource>& destination, const std::shared_ptr<Resource>& source, size_t size) override {
        Manager().QueueResourceCopy(destination, source, size);
    }

    void SubmitStagedUploads(std::shared_ptr<StagedUploadBatch> batch) override {
        Manager().SubmitStagedUploads(std::move(batch));
    }

    void SetStagedUploadsRecordedDirectly(bool direct) override {
        Manager().SetStagedUploadsRecordedDirectly(direct);
    }

    size_t RecordStagedUploads(rhi::CommandList& list, uint8_t frameIndex, bool afterWork) override {
        return Manager().RecordStagedUploads(list, frameIndex, afterWork);
    }

    bool SubmitRecordedUploads(std::shared_ptr<StagedUploadBatch> batch) override {
        return Manager().SubmitRecordedUploads(std::move(batch));
    }

    size_t TakeRecordedUploads(std::vector<rhi::CommandList>& out, uint8_t frameIndex) override {
        return Manager().TakeRecordedUploads(out, frameIndex);
    }

    bool TakeRecordedUploadsWithDestinations(std::vector<rhi::CommandList>& out, uint8_t frameIndex,
        std::vector<rhi::ResourceHandle>& destinations) override {
        return Manager().TakeRecordedUploadsWithDestinations(out, frameIndex, destinations);
    }

    bool StagedUploadDestinations(std::vector<rhi::ResourceHandle>& out) const override {
        return Manager().StagedUploadDestinations(out);
    }

    size_t RecordStagedUploadsRouted(rhi::CommandList& list, uint8_t frameIndex, bool afterWork,
        const std::function<rhi::CommandList*(rhi::ResourceHandle)>& route, size_t& routed) override {
        return Manager().RecordStagedUploadsRouted(list, frameIndex, afterWork, route, routed);
    }

    void ProcessDeferredReleases(uint8_t frameIndex) override {
        Manager().ProcessDeferredReleases(frameIndex);
    }

    std::shared_ptr<TrackedUploadTicket> QueueTrackedStreamingUploadSegments(
        std::span<const StreamingUploadSegment> segments, size_t totalSize,
        WorkerOwnedDestination destination, size_t dstOffset) override {
        return Manager().QueueTrackedStreamingUploadSegments(
            segments, totalSize, std::move(destination), dstOffset);
    }

    std::shared_ptr<TrackedUploadTicket> QueueTrackedStreamingUploadRanges(
        std::span<const StreamingUploadRange> ranges, WorkerOwnedDestination destination) override {
        return Manager().QueueTrackedStreamingUploadRanges(ranges, std::move(destination));
    }

    std::shared_ptr<TrackedUploadTicket> QueueTrackedStreamingCopyRanges(std::shared_ptr<Resource> source,
        std::span<const StreamingCopyRange> ranges, WorkerOwnedDestination destination) override {
        return Manager().QueueTrackedStreamingCopyRanges(
            std::move(source), ranges, std::move(destination));
    }

    bool HasDedicatedStreamingQueue() const override {
        return Manager().HasDedicatedStreamingQueue();
    }

    bool QueueStreamingWait(std::span<const std::pair<std::shared_ptr<rhi::TimelinePtr>, uint64_t>> points) override {
        return Manager().QueueStreamingWait(points);
    }
    std::shared_ptr<TrackedUploadTicket> QueueStreamingSignal(std::shared_ptr<rhi::TimelinePtr> timeline, uint64_t value) override {
        return Manager().QueueStreamingSignal(std::move(timeline), value);
    }

    void Cleanup() override {
        Manager().Cleanup();
    }

private:
    UploadManager& Manager() const { return m_owned ? *m_owned : UploadManager::GetInstance(); }
    std::unique_ptr<UploadManager> m_owned;
};
}

std::shared_ptr<IUploadService> CreateDefaultUploadService() {
    return std::make_shared<DefaultUploadService>();
}

std::shared_ptr<IUploadService> CreateGraphUploadService() {
    return std::make_shared<DefaultUploadService>(std::make_unique<UploadManager>(UploadManager::GetInstance()));
}

}
