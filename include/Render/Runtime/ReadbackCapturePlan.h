#pragma once

#include <memory>
#include <string>
#include <vector>

#include <rhi.h>

#include "Render/PreparedPass.h"
#include "Render/QueueKind.h"
#include "Resources/ReadbackRequest.h"

namespace org {
class Resource;
}

namespace org::runtime {

class IReadbackService;

// One capture's copy into a readback buffer, planned against the source's bound backing: what both the capture pass
// (ReadbackCapturePass) and a capture appended to a pass's own recording (RenderGraph's persistent route) record.
struct ReadbackCapturePlan {
    rhi::ResourceHandle destination{};
    uint64_t bytes = 0;
    uint32_t firstMip = 0, firstSlice = 0, mipCount = 0, sliceCount = 0;
    std::vector<rhi::CopyableFootprint> footprints;  // [slice][mip]; empty for a buffer
    std::shared_ptr<const void> destinationLease;    // the readback buffer's allocation
    // Hands the request to the service when the copy is submitted; drops it if the packet is abandoned.
    std::shared_ptr<const PreparedLifecycleEffect> reservation;

    explicit operator bool() const noexcept { return bytes != 0; }
};

// Plans a capture of source (bound this frame as boundSource) over range. An empty plan when the range is empty or the
// service has no readback buffer to give; throws when the source cannot describe itself.
ReadbackCapturePlan PlanReadbackCapture(rhi::Device device, const std::shared_ptr<IReadbackService>& service, Resource& source,
    rhi::Resource boundSource, const RangeSpec& range, ReadbackCaptureCallback callback, QueueKind queue, const std::string& debugName);

// Records the plan's copies from source.
void RecordReadbackCapture(const ReadbackCapturePlan& plan, rhi::CommandList& commands, rhi::ResourceHandle source);

}  // namespace org::runtime
