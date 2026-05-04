#pragma once

#include <boost/container_hash/hash.hpp>

#include "RenderPasses/Base/CopyPass.h"
#include "Render/Runtime/IReadbackService.h"
#include "Render/ResourceRequirements.h"
#include "Resources/Buffers/Buffer.h"
#include "Resources/PixelBuffer.h"
#include "Resources/ResourceStateTracker.h"

struct ReadbackCopyCaptureInputs {
    ResourceHandleAndRange target;

    RG_DEFINE_PASS_INPUTS(ReadbackCopyCaptureInputs, &ReadbackCopyCaptureInputs::target);
};

/// A CopyPass variant of ReadbackCapturePass that runs on the copy queue.
/// This allows readback copies to overlap with graphics/compute work.
class ReadbackCopyCapturePass final : public CopyPass, public IHasImmediateModeCommands {
public:
    ReadbackCopyCapturePass(
        ReadbackCopyCaptureInputs inputs,
        ReadbackCaptureCallback callback,
        rg::runtime::IReadbackService* readbackService)
        : m_callback(std::move(callback)),
        m_readbackService(readbackService) {
        SetInputs(inputs);
    }

    void DeclareResourceUsages(CopyPassBuilder* builder) override {
        const auto& inputs = Inputs<ReadbackCopyCaptureInputs>();
        builder->WithCopySource(inputs.target);
        builder->PreferQueue(QueueKind::Copy);
    }

    void Setup() override {
    }

    void RecordImmediateCommands(ImmediateExecutionContext& context) override {
        const auto& inputs = Inputs<ReadbackCopyCaptureInputs>();
        auto* resource = m_resourceRegistryView->Resolve<Resource>(inputs.target.resource);
        if (!resource) {
            return;
        }

        ReadbackCaptureRequest request{};
        request.desc.range = inputs.target.range;
        request.desc.resourceId = resource->GetGlobalResourceID();

        if (resource->HasLayout()) {
            auto* texture = dynamic_cast<PixelBuffer*>(resource);
            if (!texture) {
                throw std::runtime_error("ReadbackCopyCapturePass: texture resource type mismatch.");
            }

            const auto handle = inputs.target.resource;
            const SubresourceRange sr = ResolveRangeSpec(inputs.target.range, handle.GetNumMipLevels(), handle.GetArraySize());
            if (sr.isEmpty()) {
                return;
            }

            std::vector<rhi::CopyableFootprint> footprints(sr.mipCount * sr.sliceCount);
            rhi::FootprintRangeDesc fr{};
            fr.texture = texture->GetAPIResource().GetHandle();
            fr.firstMip = sr.firstMip;
            fr.mipCount = sr.mipCount;
            fr.firstArraySlice = sr.firstSlice;
            fr.arraySize = sr.sliceCount;
            fr.firstPlane = 0;
            fr.planeCount = 1;
            fr.baseOffset = 0;

            auto info = context.device.GetCopyableFootprints(fr, footprints.data(), static_cast<uint32_t>(footprints.size()));

            auto readbackBuffer = Buffer::CreateShared(rhi::HeapType::Readback, info.totalBytes);
            readbackBuffer->SetName("ReadbackCopyCaptureBuffer");

            for (uint32_t slice = 0; slice < sr.sliceCount; ++slice) {
                for (uint32_t mip = 0; mip < sr.mipCount; ++mip) {
                    const uint32_t subresourceIndex = (slice * sr.mipCount) + mip;
                    const auto& fp = footprints[subresourceIndex];

                    context.list.CopyTextureToBuffer(
                        texture,
                        sr.firstMip + mip,
                        sr.firstSlice + slice,
                        readbackBuffer.get(),
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
            request.format = texture->GetFormat();
            request.width = texture->GetWidth();
            request.height = texture->GetHeight();
            request.depth = 1;
        }
        else {
            uint64_t byteSize = 0;
            if (!resource->TryGetBufferByteSize(byteSize) || byteSize == 0) {
                throw std::runtime_error("ReadbackCopyCapturePass: resource is not a texture and does not expose a buffer byte size for readback.");
            }
            auto readbackBuffer = Buffer::CreateShared(rhi::HeapType::Readback, byteSize);
            readbackBuffer->SetName("ReadbackCopyCaptureBuffer");

            context.list.CopyBufferRegion(readbackBuffer.get(), 0, resource, 0, byteSize);

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

        const uint64_t fenceValue = m_readbackService->GetNextReadbackFenceValue();
        m_readbackService->FinalizeCapture(m_pendingToken, fenceValue);
        m_hasPendingToken = false;
        return { m_readbackService->GetReadbackFence(), fenceValue };
    }

    void Cleanup() override {
    }

private:
    ReadbackCaptureCallback m_callback;
    rg::runtime::ReadbackCaptureToken m_pendingToken{};
    rg::runtime::IReadbackService* m_readbackService = nullptr; // non-owning
    bool m_hasPendingToken = false;
};
