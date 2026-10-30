#pragma once

#include "RenderPasses/Base/TypedRenderGraphPass.h"
#include "Render/PassBuilders.h"
#include "Render/Runtime/IReadbackService.h"
#include "Render/Runtime/ReadbackCapturePlan.h"
#include "Resources/Buffers/Buffer.h"
#include "Resources/PixelBuffer.h"

namespace org {
struct ReadbackCaptureInputs {
    ResourceHandleAndRange target;
    RG_DEFINE_PASS_INPUTS(ReadbackCaptureInputs, &ReadbackCaptureInputs::target);
};

struct ReadbackCaptureFrameData {
    PreparedResourceReference source;
    runtime::ReadbackCapturePlan plan;
};

// Queue capability is the only difference between graphics and copy capture.
template<QueueKind Queue>
class BasicReadbackCapturePass final
    : public TypedRenderGraphPass<BasicReadbackCapturePass<Queue>, ReadbackCaptureFrameData> {
public:
    BasicReadbackCapturePass(ReadbackCaptureInputs inputs, std::shared_ptr<Resource> source,
        ReadbackCaptureCallback callback, std::shared_ptr<runtime::IReadbackService> service,
        std::string debugName = {})
        : m_source(std::move(source)),
          m_callback(std::move(callback)), m_service(std::move(service)), m_name(std::move(debugName)) { this->SetInputs(std::move(inputs)); }

    void Declare(PassBuilder& builder) {
        const auto& inputs = this->template Inputs<ReadbackCaptureInputs>();
        builder.CopySource(inputs.target);
        builder.PreferQueue(Queue);
    }

    ReadbackCaptureFrameData Prepare(const PassPrepareContext& preparation) {
        const auto& inputs = this->template Inputs<ReadbackCaptureInputs>();
        ReadbackCaptureFrameData data{};
        if (!m_source || !m_service) return data;
        data.source = preparation.CaptureResource(m_source->GetGlobalResourceID());
        data.plan = runtime::PlanReadbackCapture(preparation.device, m_service, *m_source,
            preparation.ResolveCapturedResource(data.source), inputs.target.range, m_callback, Queue, m_name);
        if (!data.plan) return {};
        if (data.plan.destinationLease) preparation.Retain(data.plan.destinationLease);
        preparation.Reserve(data.plan.reservation);
        data.plan.destinationLease.reset();
        data.plan.reservation.reset();
        return data;
    }

    static void Record(const ReadbackCaptureFrameData& data, PassRecordContext& recording) {
        if (!data.plan) return;
        runtime::RecordReadbackCapture(data.plan, recording.Commands(), recording.Resolve(data.source).GetHandle());
    }

private:
    std::shared_ptr<Resource> m_source;
    ReadbackCaptureCallback m_callback;
    std::shared_ptr<runtime::IReadbackService> m_service;
    std::string m_name;
};
using ReadbackCapturePass = BasicReadbackCapturePass<QueueKind::Graphics>;
} // namespace org
