#include "Render/Runtime/ReadbackCapturePlan.h"

#include <stdexcept>

#include "Render/Runtime/ReadbackCaptureReservation.h"
#include "Resources/BackedResource.h"
#include "Resources/Resource.h"
#include "Resources/ResourceStateTracker.h"

namespace org::runtime {

ReadbackCapturePlan PlanReadbackCapture(rhi::Device device, const std::shared_ptr<IReadbackService>& service, Resource& source,
    rhi::Resource boundSource, const RangeSpec& range, ReadbackCaptureCallback callback, QueueKind queue, const std::string& debugName) {
    ReadbackCapturePlan plan;
    if (!service) return plan;
    ReadbackCaptureRequest request{};
    request.desc.range = range;
    request.desc.resourceId = source.GetGlobalResourceID();
    request.callback = std::move(callback);
    if (source.HasLayout()) {
        rhi::ResourceDesc desc{};
        if (!source.TryGetRHIResourceDesc(desc)) throw std::runtime_error("Readback texture does not expose an RHI description");
        const auto resolved = ResolveRangeSpec(range, source.GetMipLevels(), source.GetArraySize());
        if (resolved.isEmpty()) return plan;
        plan.firstMip = resolved.firstMip; plan.firstSlice = resolved.firstSlice;
        plan.mipCount = resolved.mipCount; plan.sliceCount = resolved.sliceCount;
        plan.footprints.resize(resolved.mipCount * resolved.sliceCount);
        rhi::FootprintRangeDesc footprint{};
        footprint.texture = boundSource.GetHandle();
        footprint.firstMip = resolved.firstMip; footprint.mipCount = resolved.mipCount;
        footprint.firstArraySlice = resolved.firstSlice; footprint.arraySize = resolved.sliceCount;
        footprint.planeCount = 1;
        const auto info = device.GetCopyableFootprints(footprint, plan.footprints.data(), static_cast<uint32_t>(plan.footprints.size()));
        plan.bytes = info.totalBytes;
        request.desc.kind = ReadbackResourceKind::Texture;
        request.layouts = plan.footprints;
        request.format = desc.texture.format;
        request.width = desc.texture.width;
        request.height = desc.texture.height;
        request.depth = 1;
    } else {
        if (!source.TryGetBufferByteSize(plan.bytes) || !plan.bytes) throw std::runtime_error("Readback buffer does not expose its byte size");
        request.desc.kind = ReadbackResourceKind::Buffer;
    }
    auto destination = service->AcquireReadbackBuffer(plan.bytes, debugName.empty() ? "ReadbackCaptureBuffer" : debugName.c_str());
    if (!destination) return {};
    auto* backing = dynamic_cast<BackedResource*>(destination.get());
    auto allocation = backing ? backing->CaptureBackingAllocation() : BackingAllocationSnapshot{};
    if (!allocation) throw std::runtime_error("Readback destination lacks concrete allocation ownership");
    plan.destination = allocation.resource.GetHandle();
    plan.destinationLease = allocation.lease;
    request.readbackBuffer = std::move(destination);
    request.totalSize = plan.bytes;
    plan.reservation = std::make_shared<ReadbackCaptureReservation>(device, service, std::move(request), queue);
    return plan;
}

void RecordReadbackCapture(const ReadbackCapturePlan& plan, rhi::CommandList& commands, rhi::ResourceHandle source) {
    if (!plan.bytes) return;
    if (plan.footprints.empty()) {
        commands.CopyBufferRegion(plan.destination, 0, source, 0, plan.bytes);
        return;
    }
    for (uint32_t slice = 0; slice < plan.sliceCount; ++slice)
        for (uint32_t mip = 0; mip < plan.mipCount; ++mip) {
            rhi::BufferTextureCopyFootprint region{};
            region.texture = source; region.buffer = plan.destination;
            region.mip = plan.firstMip + mip; region.arraySlice = plan.firstSlice + slice;
            region.footprint = plan.footprints[slice * plan.mipCount + mip];
            commands.CopyTextureToBuffer(region);
        }
}

}  // namespace org::runtime
