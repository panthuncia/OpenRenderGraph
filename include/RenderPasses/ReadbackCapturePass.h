#pragma once

#include <boost/container_hash/hash.hpp>
#include <BasicTelemetry/Tracy.h>

#include "RenderPasses/Base/RenderPass.h"
#include "Render/Runtime/IReadbackService.h"
#include "Render/ResourceRequirements.h"
#include "Resources/Buffers/Buffer.h"
#include "Resources/PixelBuffer.h"
#include "Resources/ResourceStateTracker.h"


namespace org {

struct ReadbackCaptureInputs {
    ResourceHandleAndRange target;

    RG_DEFINE_PASS_INPUTS(ReadbackCaptureInputs, &ReadbackCaptureInputs::target);
};

class ReadbackCapturePass final : public RenderPass, public IHasImmediateModeCommands {
public:
    ReadbackCapturePass(
        ReadbackCaptureInputs inputs,
        std::shared_ptr<Resource> sourceResource,
        ReadbackCaptureCallback callback,
        org::runtime::IReadbackService* readbackService,
        std::string debugCaptureName = {})
        : m_sourceResource(std::move(sourceResource)),
        m_callback(std::move(callback)),
        m_readbackService(readbackService),
        m_debugCaptureName(std::move(debugCaptureName)) {
        SetInputs(inputs);
    }

    void DeclareResourceUsages(RenderPassBuilder* builder) override {
        const auto& inputs = Inputs<ReadbackCaptureInputs>();
        builder->WithCopySource(inputs.target);
    }

    void Setup() override {
    }

    void RecordImmediateCommands(ImmediateExecutionContext& context) override {
        BT_ZONE_SCOPE("ReadbackCapturePass::RecordImmediateCommands");
        if (!m_debugCaptureName.empty()) {
            BT_ZONE_TEXT(m_debugCaptureName.c_str(), m_debugCaptureName.size());
        }

        const auto& inputs = Inputs<ReadbackCaptureInputs>();
        auto* resource = m_sourceResource.get();
        if (!resource) {
            return;
        }

        ReadbackCaptureRequest request{};
        request.desc.range = inputs.target.range;
        request.desc.resourceId = resource->GetGlobalResourceID();

        if (resource->HasLayout()) {
            auto* texture = dynamic_cast<PixelBuffer*>(resource);
            rhi::ResourceDesc textureDesc{};
            if (!resource->TryGetRHIResourceDesc(textureDesc))
                throw std::runtime_error("ReadbackCapturePass: texture resource does not expose an RHI description.");

            const auto handle = inputs.target.resource;
            const SubresourceRange sr = ResolveRangeSpec(inputs.target.range, handle.GetNumMipLevels(), handle.GetArraySize());
            if (sr.isEmpty()) {
                return;
            }

            std::vector<rhi::CopyableFootprint> footprints(sr.mipCount * sr.sliceCount);
            rhi::FootprintRangeDesc fr{};
            // Readback extension passes currently default to the primary device.
            // The immediate list resolves the same primary representation for the copy.
            fr.texture = resource->GetAPIResource().GetHandle();
            fr.firstMip = sr.firstMip;
            fr.mipCount = sr.mipCount;
            fr.firstArraySlice = sr.firstSlice;
            fr.arraySize = sr.sliceCount;
            fr.firstPlane = 0;
            fr.planeCount = 1;
            fr.baseOffset = 0;

            auto info = context.device.GetCopyableFootprints(fr, footprints.data(), static_cast<uint32_t>(footprints.size()));

            auto readbackBuffer = m_readbackService
                ? m_readbackService->AcquireReadbackBuffer(info.totalBytes, "ReadbackCaptureBuffer")
                : std::static_pointer_cast<Resource>(Buffer::CreateShared(rhi::HeapType::Readback, info.totalBytes));
            if (!readbackBuffer) {
                return;
            }
            BT_PLOT("Readback.CaptureRequestedBytes", static_cast<int64_t>(info.totalBytes));

            for (uint32_t slice = 0; slice < sr.sliceCount; ++slice) {
                for (uint32_t mip = 0; mip < sr.mipCount; ++mip) {
                    const uint32_t subresourceIndex = (slice * sr.mipCount) + mip;
                    const auto& fp = footprints[subresourceIndex];

                    context.list.CopyTextureToBuffer(
                        m_sourceResource,
                        sr.firstMip + mip,
                        sr.firstSlice + slice,
                        readbackBuffer,
                        fp,
                        0,
                        0,
                        0);
                }
            }

            request.desc.kind = ReadbackResourceKind::Texture;
            request.readbackBuffer = readbackBuffer;
            request.layouts = std::move(footprints);
            request.totalSize = info.totalBytes;
            request.format = textureDesc.texture.format;
            request.width = textureDesc.texture.width;
            request.height = textureDesc.texture.height;
            request.depth = 1;
        }
        else {
            uint64_t byteSize = 0;
            if (!resource->TryGetBufferByteSize(byteSize) || byteSize == 0) {
                throw std::runtime_error("ReadbackCapturePass: resource is not a texture (has no layout) and does not expose a buffer byte size for readback.");
            }
            auto readbackBuffer = m_readbackService
                ? m_readbackService->AcquireReadbackBuffer(byteSize, "ReadbackCaptureBuffer")
                : std::static_pointer_cast<Resource>(Buffer::CreateShared(rhi::HeapType::Readback, byteSize));
            if (!readbackBuffer) {
                return;
            }
            BT_PLOT("Readback.CaptureRequestedBytes", static_cast<int64_t>(byteSize));

            context.list.CopyBufferRegion(readbackBuffer, 0, m_sourceResource, 0, byteSize);

            request.desc.kind = ReadbackResourceKind::Buffer;
            request.readbackBuffer = readbackBuffer;
            request.totalSize = byteSize;
        }

        request.callback = m_callback;
        if (!m_readbackService) {
            return;
        }

        m_pendingToken = m_readbackService->EnqueueCapture(std::move(request));
        m_hasPendingToken = true;
    }

    PassReturn Execute(PassExecutionContext& context) override {
        if (!m_hasPendingToken) {
            return {};
        }

        if (!m_readbackService) {
            m_hasPendingToken = false;
            return {};
        }

        const rhi::Timeline signalFence = m_readbackService->GetReadbackFence(QueueKind::Graphics);
        if (!signalFence.IsValid()) {
            m_hasPendingToken = false;
            return {};
        }

        const uint64_t fenceValue = m_readbackService->GetNextReadbackFenceValue(QueueKind::Graphics);
        m_readbackService->FinalizeCapture(m_pendingToken, QueueKind::Graphics, nullptr, fenceValue);
        m_hasPendingToken = false;
        return { signalFence, fenceValue };
    }

    std::optional<OwnedImmediateSubmissionEffect> TakeOwnedImmediateSubmissionEffect() override {
        if (!m_hasPendingToken || !m_readbackService) return std::nullopt;
        const rhi::Timeline signalFence = m_readbackService->GetReadbackFence(QueueKind::Graphics);
        if (!signalFence.IsValid()) return std::nullopt;
        const uint64_t fenceValue = m_readbackService->GetNextReadbackFenceValue(QueueKind::Graphics);
        const auto token = m_pendingToken;
        auto* service = m_readbackService;
        m_hasPendingToken = false;
        return OwnedImmediateSubmissionEffect{
            .completionSignals = {{signalFence, fenceValue}},
            .commit = [service, token, fenceValue]() {
                service->FinalizeCapture(token, QueueKind::Graphics, nullptr, fenceValue);
            },
        };
    }

    void Cleanup() override {
    }

private:
    std::shared_ptr<Resource> m_sourceResource;
    ReadbackCaptureCallback m_callback;
    org::runtime::ReadbackCaptureToken m_pendingToken{};
    org::runtime::IReadbackService* m_readbackService = nullptr; // non-owning
    std::string m_debugCaptureName;
    bool m_hasPendingToken = false;
};


} // namespace org
