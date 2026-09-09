#pragma once

#include <vector>
#include <memory>
#include <cstddef>
#include <cstdint>

#include "RenderPasses/Base/TypedRenderGraphPass.h"
#include "Render/PassBuilders.h"
#include "Render/ResourceRequirements.h"
#include "Render/Runtime/StreamingUploadTypes.h"
#include "Interfaces/IResourceResolver.h"
#include "Resources/Resource.h"


/// Inputs for the StreamingUploadPass. Contains all pending uploads to process.
namespace org {

struct StreamingUploadInputs {
    std::vector<StreamingUploadDescriptor> uploads;
    /// Optional resolver for additional resources that are copy destinations
    /// (e.g. page-pool slab buffers). If non-null, all resolved resources are
    /// declared as copy-dest so the render graph can schedule transitions.
    std::unique_ptr<IResourceResolver> poolResolver;
};

inline org::Hash64 HashValue(const StreamingUploadInputs& i) {
    // Ephemeral per-frame pass; hash by upload count for differentiation
    return static_cast<org::Hash64>(i.uploads.size());
}

inline bool operator==(const StreamingUploadInputs& a, const StreamingUploadInputs& b) {
    return a.uploads.size() == b.uploads.size(); // identity by reference; ephemeral
}

struct StreamingUploadFrameData {
    struct Copy {
        PreparedResourceReference source, destination;
        uint64_t sourceOffset, destinationOffset, size;
    };
    std::vector<Copy> copies;
};

class StreamingUploadPass final : public TypedRenderGraphPass<StreamingUploadPass, StreamingUploadFrameData> {
public:
    explicit StreamingUploadPass(StreamingUploadInputs inputs) { SetInputs(std::move(inputs)); }
    void Declare(PassBuilder& builder) {
        const auto& inputs = Inputs<StreamingUploadInputs>();
        for (const auto& upload : inputs.uploads) {
            if (!upload.dstResource || !upload.srcUploadBuffer || !upload.size) continue;
            builder.WithCopyDest(upload.dstResource).WithCopySource(upload.srcUploadBuffer);
        }
        if (inputs.poolResolver) builder.WithCopyDest(*inputs.poolResolver);
        builder.PreferQueue(QueueKind::Copy);
    }
    StreamingUploadFrameData Prepare(const PassPrepareContext& preparation) {
        StreamingUploadFrameData data;
        for (const auto& upload : Inputs<StreamingUploadInputs>().uploads) {
            if (!upload.dstResource || !upload.srcUploadBuffer || !upload.size) continue;
            data.copies.push_back({preparation.CaptureResource(upload.srcUploadBuffer->GetGlobalResourceID()),
                preparation.CaptureResource(upload.dstResource->GetGlobalResourceID()),
                upload.srcOffset, upload.dstOffset, upload.size});
        }
        return data;
    }
    static void Record(const StreamingUploadFrameData& data, PassRecordContext& recording) {
        for (const auto& copy : data.copies)
            recording.Commands().CopyBufferRegion(recording.Resolve(copy.destination).GetHandle(),
                copy.destinationOffset, recording.Resolve(copy.source).GetHandle(), copy.sourceOffset, copy.size);
    }
};
} // namespace org
