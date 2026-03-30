#pragma once

#include <vector>
#include <memory>
#include <cstddef>
#include <cstdint>

#include "RenderPasses/Base/CopyPass.h"
#include "Render/ResourceRequirements.h"
#include "Render/Runtime/StreamingUploadTypes.h"
#include "Interfaces/IResourceResolver.h"
#include "Resources/Resource.h"

/// Inputs for the StreamingUploadPass. Contains all pending uploads to process.
struct StreamingUploadInputs {
    std::vector<StreamingUploadDescriptor> uploads;
    /// Optional resolver for additional resources that are copy destinations
    /// (e.g. page-pool slab buffers). If non-null, all resolved resources are
    /// declared as copy-dest so the render graph can schedule transitions.
    std::unique_ptr<IResourceResolver> poolResolver;
};

inline rg::Hash64 HashValue(const StreamingUploadInputs& i) {
    // Ephemeral per-frame pass; hash by upload count for differentiation
    return static_cast<rg::Hash64>(i.uploads.size());
}

inline bool operator==(const StreamingUploadInputs& a, const StreamingUploadInputs& b) {
    return a.uploads.size() == b.uploads.size(); // identity by reference; ephemeral
}

/// A CopyPass that runs on the copy queue and performs streaming buffer uploads.
/// Created per-frame by the streaming extension when there are pending uploads.
class StreamingUploadPass final : public CopyPass {
public:
    explicit StreamingUploadPass(StreamingUploadInputs inputs) {
        SetInputs(std::move(inputs));
    }

    void DeclareResourceUsages(CopyPassBuilder* builder) override {
        const auto& inputs = Inputs<StreamingUploadInputs>();
        for (const auto& upload : inputs.uploads) {
            if (upload.dstResource) {
                builder->WithCopyDest(upload.dstResource);
            }
            // Upload-heap sources don't need to be declared — they are
            // ephemeral resources pinned via the shared_ptr overload of
            // CopyBufferRegion in ExecuteImmediate.
        }
        // Declare all pool slab buffers as copy destinations so the render
        // graph knows about them and can schedule transitions correctly,
        // even though individual uploads already declare their specific
        // destination resource.
        if (inputs.poolResolver) {
            builder->WithCopyDest(*inputs.poolResolver);
        }
        builder->PreferQueue(QueueKind::Copy);
    }

    void Setup() override {}

    void ExecuteImmediate(ImmediateExecutionContext& context) override {
        const auto& inputs = Inputs<StreamingUploadInputs>();
        for (const auto& upload : inputs.uploads) {
            if (!upload.dstResource || !upload.srcUploadBuffer || upload.size == 0) {
                continue;
            }
            context.list.CopyBufferRegion(
                upload.dstResource.get(), upload.dstOffset,
                upload.srcUploadBuffer, upload.srcOffset,
                upload.size);
        }
    }

    PassReturn Execute(PassExecutionContext& context) override {
        return {};
    }

    void Cleanup() override {}
};
