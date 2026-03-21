#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <cstdint>

#include <imgui.h>
#include <rhi.h>

#include "Render/RenderGraph/RenderGraph.h"
#include "Resources/ReadbackRequest.h"

// Return true if the pass uses 'resourceId'.
// passKind: 0 = RenderPassAndResources, 1 = ComputePassAndResources, 2 = CopyPassAndResources
using RGPassUsesResourceFn = std::function<bool(const void* passAndResources, uint64_t resourceId, int passKind)>;
using RGResourceNameByIdFn = std::function<std::string(uint64_t resourceId)>;
using RGResourcePtrByIdFn = std::function<Resource*(uint64_t resourceId)>;
using RGRequestReadbackCaptureFn = std::function<void(const std::string&, Resource*, const RangeSpec&, ReadbackCaptureCallback)>;

struct RGInspectorOptions {
    // Horizontal placement within a batch (x spans [batch, batch+1])
    float blockLeftTransitions = 0.05f; // transitions start offset
    float blockWidthTransitions = 0.20f;
    float blockGap = 0.05f;
    float blockWidthPasses = 0.60f; // passes span
    float blockWidthBatchEnd = 0.20f; // width of batch-end transitions
    float rowHeight = 1.0f;  // ImPlot units
    float laneSpacing = 1.2f;  // extra space between lanes

    // ImGui descriptor heap callbacks for texture preview in MemoryViewWidget.
    std::function<uint32_t()> imguiAllocDescriptor;
    std::function<void(uint32_t)> imguiFreeDescriptor;
    std::function<ImTextureID(uint32_t)> imguiGpuHandle;
    rhi::DescriptorHeapHandle imguiHeapHandle{};
};

namespace RGInspector {
    void Show(const std::vector<RenderGraph::PassBatch>& batches,
        const QueueRegistry& registry,
        RGPassUsesResourceFn passUses = nullptr,
        RGResourceNameByIdFn resourceNameById = nullptr,
        RGResourcePtrByIdFn resourcePtrById = nullptr,
        RGRequestReadbackCaptureFn requestReadbackCapture = nullptr,
        const RGInspectorOptions& opts = {});
}
