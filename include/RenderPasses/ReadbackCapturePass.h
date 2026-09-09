#pragma once

#include "RenderPasses/Base/TypedRenderGraphPass.h"
#include "Render/PassBuilders.h"
#include "Render/Runtime/ReadbackCaptureReservation.h"
#include "Resources/Buffers/Buffer.h"
#include "Resources/PixelBuffer.h"

namespace org {
struct ReadbackCaptureInputs {
    ResourceHandleAndRange target;
    RG_DEFINE_PASS_INPUTS(ReadbackCaptureInputs, &ReadbackCaptureInputs::target);
};

struct ReadbackCaptureFrameData {
    PreparedResourceReference source;
    rhi::ResourceHandle destination{};
    uint64_t bytes = 0;
    uint32_t firstMip = 0, firstSlice = 0, mipCount = 0, sliceCount = 0;
    std::vector<rhi::CopyableFootprint> footprints;
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
        builder.WithCopySource(inputs.target).PreferQueue(Queue);
    }
    ReadbackCaptureFrameData Prepare(const PassPrepareContext& preparation) {
        const auto& inputs = this->template Inputs<ReadbackCaptureInputs>();
        ReadbackCaptureFrameData data{};
        if (!m_source || !m_service) return data;
        data.source = preparation.CaptureResource(m_source->GetGlobalResourceID());
        ReadbackCaptureRequest request{};
        request.desc.range = inputs.target.range;
        request.desc.resourceId = m_source->GetGlobalResourceID();
        request.callback = m_callback;
        if (m_source->HasLayout()) {
            rhi::ResourceDesc desc{};
            if (!m_source->TryGetRHIResourceDesc(desc))
                throw std::runtime_error("Readback texture does not expose an RHI description");
            const auto handle = inputs.target.resource;
            const auto range = ResolveRangeSpec(inputs.target.range,
                handle.GetNumMipLevels(), handle.GetArraySize());
            if (range.isEmpty()) return data;
            data.firstMip = range.firstMip; data.firstSlice = range.firstSlice;
            data.mipCount = range.mipCount; data.sliceCount = range.sliceCount;
            data.footprints.resize(range.mipCount * range.sliceCount);
            rhi::FootprintRangeDesc footprint{};
            footprint.texture = preparation.ResolveCapturedResource(data.source).GetHandle();
            footprint.firstMip = range.firstMip; footprint.mipCount = range.mipCount;
            footprint.firstArraySlice = range.firstSlice; footprint.arraySize = range.sliceCount;
            footprint.planeCount = 1;
            auto device = preparation.device;
            auto info = device.GetCopyableFootprints(footprint, data.footprints.data(),
                static_cast<uint32_t>(data.footprints.size()));
            data.bytes = info.totalBytes;
            request.desc.kind = ReadbackResourceKind::Texture;
            request.layouts = data.footprints;
            request.format = desc.texture.format;
            request.width = desc.texture.width;
            request.height = desc.texture.height;
            request.depth = 1;
        } else {
            if (!m_source->TryGetBufferByteSize(data.bytes) || !data.bytes)
                throw std::runtime_error("Readback buffer does not expose its byte size");
            request.desc.kind = ReadbackResourceKind::Buffer;
        }
        auto destination = m_service->AcquireReadbackBuffer(data.bytes,
            m_name.empty() ? "ReadbackCaptureBuffer" : m_name.c_str());
        if (!destination) return {};
        auto* backing = dynamic_cast<BackedResource*>(destination.get());
        auto allocation = backing ? backing->CaptureBackingAllocation() : BackingAllocationSnapshot{};
        if (!allocation) throw std::runtime_error("Readback destination lacks concrete allocation ownership");
        data.destination = allocation.resource.GetHandle();
        preparation.Retain(allocation.lease);
        request.readbackBuffer = std::move(destination);
        request.totalSize = data.bytes;
        preparation.Reserve(std::make_shared<runtime::ReadbackCaptureReservation>(
            preparation.device, m_service, std::move(request), Queue));
        return data;
    }
    static void Record(const ReadbackCaptureFrameData& data, PassRecordContext& recording) {
        if (!data.bytes) return;
        const auto source = recording.Resolve(data.source).GetHandle();
        if (data.footprints.empty()) {
            recording.Commands().CopyBufferRegion(data.destination, 0, source, 0, data.bytes);
            return;
        }
        for (uint32_t slice = 0; slice < data.sliceCount; ++slice)
            for (uint32_t mip = 0; mip < data.mipCount; ++mip) {
                rhi::BufferTextureCopyFootprint region{};
                region.texture = source; region.buffer = data.destination;
                region.mip = data.firstMip + mip; region.arraySlice = data.firstSlice + slice;
                region.footprint = data.footprints[slice * data.mipCount + mip];
                recording.Commands().CopyTextureToBuffer(region);
            }
    }
private:
    std::shared_ptr<Resource> m_source;
    ReadbackCaptureCallback m_callback;
    std::shared_ptr<runtime::IReadbackService> m_service;
    std::string m_name;
};
using ReadbackCapturePass = BasicReadbackCapturePass<QueueKind::Graphics>;
} // namespace org
