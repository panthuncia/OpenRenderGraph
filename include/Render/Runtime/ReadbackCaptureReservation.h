#pragma once
#include "Render/PreparedPass.h"
#include "Render/Runtime/IReadbackService.h"

namespace org::runtime {
// Unsubmitted captures never enter the readback service's pending queue.
// The reservation retains its own timeline and request through submission.
class ReadbackCaptureReservation final : public PreparedLifecycleEffect {
public:
    ReadbackCaptureReservation(rhi::Device device, std::shared_ptr<IReadbackService> service,
        ReadbackCaptureRequest request, QueueKind queue)
        : m_service(std::move(service)), m_request(std::move(request)) {
        if (!m_service) throw std::invalid_argument("Readback capture requires a service owner");
        auto fence = std::make_shared<rhi::TimelinePtr>();
        if (rhi::Failed(device.CreateTimeline(*fence, 0, "ReadbackCapture.Completion")))
            throw std::runtime_error("Failed to reserve readback completion timeline");
        m_fence = std::move(fence);
        m_request.signalFenceOwner = m_fence;
        m_request.signalQueueKind = queue;
        m_request.fenceValue = 1;
        m_signal = {m_request.signalFenceOwner->Get(), 1};
    }
    std::span<const ExternalTimelinePoint> SignalsAfterCompletion() const override {
        return {&m_signal, 1};
    }
    void Submitted(SubmissionContext) const override {
        if (!m_resolved.exchange(true)) m_service->EnqueueCapture(std::move(m_request));
    }
    void Abandoned(AbandonReason) const override { m_resolved.exchange(true); }
private:
    std::shared_ptr<IReadbackService> m_service;
    mutable ReadbackCaptureRequest m_request;
    std::shared_ptr<rhi::TimelinePtr> m_fence;
    ExternalTimelinePoint m_signal{};
    mutable std::atomic<bool> m_resolved{false};
};
} // namespace org::runtime

