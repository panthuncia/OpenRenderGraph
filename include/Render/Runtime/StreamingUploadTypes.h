#pragma once

#include <vector>
#include <memory>
#include <cstddef>

class Resource;

/// Descriptor for a single streaming upload operation.
/// Captured by the UploadManager's streaming path and consumed each frame
/// by the StreamingUploadPass.
struct StreamingUploadDescriptor {
    std::shared_ptr<Resource> srcUploadBuffer;   // Upload-heap page
    size_t srcOffset = 0;
    std::shared_ptr<Resource> dstResource;       // GPU-local target
    size_t dstOffset = 0;
    size_t size = 0;
};
