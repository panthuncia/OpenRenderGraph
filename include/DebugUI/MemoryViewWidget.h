#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <imgui.h>
#include <rhi.h>

#include "Resources/ReadbackRequest.h"
#include "Resources/ResourceStateTracker.h"

class Resource;
struct LayoutNode;

namespace ui {

    using MemoryViewRequestCaptureFn = std::function<void(const std::string&, Resource*, const RangeSpec&, ReadbackCaptureCallback)>;

    // Callback to allocate/free descriptors from the ImGui shader-visible heap.
    // Returns a descriptor index.
    using ImGuiDescriptorAllocFn = std::function<uint32_t()>;
    using ImGuiDescriptorFreeFn = std::function<void(uint32_t)>;
    using ImGuiGpuHandleFn = std::function<ImTextureID(uint32_t)>;

    class MemoryViewWidget {
    public:
        // Schedules a readback capture and opens the window.
        void Open(const std::string& passName, Resource* resource, const RangeSpec& range, MemoryViewRequestCaptureFn requestCapture);

        // Draws the window if open.
        void Draw(bool* pOpen);

        // Set callbacks that let the widget allocate ImGui descriptors (from BasicRenderer's Menu).
        void SetImGuiDescriptorCallbacks(
            ImGuiDescriptorAllocFn alloc,
            ImGuiDescriptorFreeFn free,
            ImGuiGpuHandleFn gpuHandle,
            rhi::DescriptorHeapHandle heapHandle)
        {
            imguiAllocDesc_ = std::move(alloc);
            imguiFreeDesc_ = std::move(free);
            imguiGpuHandle_ = std::move(gpuHandle);
            imguiHeapHandle_ = heapHandle;
        }

    private:
        void DrawBufferView(const ReadbackCaptureResult& r);
        void DrawTextureView(const ReadbackCaptureResult& r);
        void SaveCurrentResourceLayoutState();
        void LoadResourceLayoutState(uint64_t resourceId);
        void ReleasePreviewTexture();

        struct PendingRequest {
            std::string passName;
            Resource* resource = nullptr;
            RangeSpec range{};
            uint64_t resourceId = 0;
            std::string resourceName;
        };

        std::mutex mutex_;
        std::optional<PendingRequest> pending_;
        std::optional<ReadbackCaptureResult> result_;
        bool waiting_ = false;
        std::string status_;

        // UI state — buffer view
        int bytesPerRow_ = 16;
        std::array<char, 16 * 1024> structInputBuf_{};
        std::string reflectionDiagnostics_;
        size_t reflectedRootSizeBytes_ = 0;
        size_t reflectedRootStrideBytes_ = 0;
        bool reflectionValid_ = false;
        std::shared_ptr<LayoutNode> reflectedRoot_;

        int goToElementInput_ = 0;
        int scrollToElement_ = -1;

        uint64_t goToByteOffsetInput_ = 0;
        uint64_t scrollToByteOffset_ = UINT64_MAX;
        uint64_t highlightedByteOffset_ = UINT64_MAX;

        uint64_t currentResourceId_ = 0;

        struct ResourceLayoutState {
            std::string structInput;
            std::string diagnostics;
            size_t rootSizeBytes = 0;
            size_t rootStrideBytes = 0;
            bool reflectionValid = false;
            std::shared_ptr<LayoutNode> reflectedRoot;
            int goToElementInput = 0;
        };

        std::unordered_map<uint64_t, ResourceLayoutState> perResourceLayoutState_;

        // UI state - texture view
        int selectedMip_ = 0;
        int selectedSlice_ = 0;
        bool previewDirty_ = true;
        ImTextureID previewTextureId_ = 0;
        uint32_t previewDescriptorIndex_ = UINT32_MAX;
        uint32_t previewWidth_ = 0;
        uint32_t previewHeight_ = 0;
        rhi::ResourcePtr previewTexture_;
        rhi::ResourcePtr previewUploadBuffer_;

        // ImGui descriptor callbacks (set by host app)
        ImGuiDescriptorAllocFn imguiAllocDesc_;
        ImGuiDescriptorFreeFn imguiFreeDesc_;
        ImGuiGpuHandleFn imguiGpuHandle_;
        rhi::DescriptorHeapHandle imguiHeapHandle_{};
    };

} // namespace ui
