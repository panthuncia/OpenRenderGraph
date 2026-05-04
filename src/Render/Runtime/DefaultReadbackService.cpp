#include "Render/Runtime/IReadbackService.h"

#include "Managers/Singletons/ReadbackManager.h"

namespace rg::runtime {

namespace {
class DefaultReadbackService final : public IReadbackService {
public:
    void Initialize(rhi::Timeline graphicsReadbackFence, rhi::Timeline copyReadbackFence) override {
        ReadbackManager::GetInstance().Initialize(graphicsReadbackFence, copyReadbackFence);
    }

    void RequestReadbackCapture(const std::string& passName, Resource* resource, const RangeSpec& range, ReadbackCaptureCallback callback, QueueKind preferredQueueKind = QueueKind::Graphics) override {
        ReadbackManager::GetInstance().RequestReadbackCapture(passName, resource, range, std::move(callback), preferredQueueKind);
    }

    std::vector<rg::runtime::ReadbackCaptureInfo> ConsumeCaptureRequests() override {
        auto captures = ReadbackManager::GetInstance().ConsumeCaptureRequests();
        std::vector<rg::runtime::ReadbackCaptureInfo> out;
        out.reserve(captures.size());
        for (auto& capture : captures) {
            out.push_back({
                .passName = std::move(capture.passName),
                .resource = std::move(capture.resource),
                .resourceId = capture.resourceId,
                .range = capture.range,
                .callback = std::move(capture.callback),
                .preferredQueueKind = capture.preferredQueueKind
                });
        }
        return out;
    }

    rg::runtime::ReadbackCaptureToken EnqueueCapture(ReadbackCaptureRequest&& request) override {
        auto token = ReadbackManager::GetInstance().EnqueueCapture(std::move(request));
        return { token.id };
    }

    void FinalizeCapture(rg::runtime::ReadbackCaptureToken token, QueueKind queueKind, std::shared_ptr<rhi::TimelinePtr> signalFenceOwner, uint64_t fenceValue) override {
        ReadbackManager::GetInstance().FinalizeCapture({ token.id }, queueKind, std::move(signalFenceOwner), fenceValue);
    }

    uint64_t GetNextReadbackFenceValue(QueueKind queueKind) override {
        return ReadbackManager::GetInstance().GetNextReadbackFenceValue(queueKind);
    }

    rhi::Timeline GetReadbackFence(QueueKind queueKind) const override {
        return ReadbackManager::GetInstance().GetReadbackFence(queueKind);
    }

    void ProcessReadbackRequests() override {
        ReadbackManager::GetInstance().ProcessReadbackRequests();
    }

    void Cleanup() override {
        ReadbackManager::GetInstance().Cleanup();
    }
};
}

std::shared_ptr<IReadbackService> CreateDefaultReadbackService() {
    return std::make_shared<DefaultReadbackService>();
}

}
