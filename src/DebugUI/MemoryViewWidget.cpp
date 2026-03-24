#include "DebugUI/MemoryViewWidget.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <iomanip>
#include <string>

#include "Resources/Resource.h"
#include "structFormatHelper.h"
#include "TextureDecoder.h"
#include "Managers/Singletons/DeviceManager.h"

#include <rhi_helpers.h>
#include <spdlog/spdlog.h>

namespace ui {

    namespace {

        static bool HasNonWhitespace(const char* s)
        {
            if (!s) {
                return false;
            }
            while (*s) {
                if (!std::isspace(static_cast<unsigned char>(*s))) {
                    return true;
                }
                ++s;
            }
            return false;
        }

        static float HalfToFloat(uint16_t h)
        {
            return ui::HalfToFloat(h);
        }

        static std::string FormatScalar(
            NumericKind kind,
            uint32_t bits,
            const std::byte* ptr,
            size_t remaining)
        {
            auto need = [&](size_t n) { return remaining >= n; };

            switch (kind) {
            case NumericKind::SInt:
                if (bits == 8 && need(1)) { int8_t v; std::memcpy(&v, ptr, 1); return std::to_string(v); }
                if (bits == 16 && need(2)) { int16_t v; std::memcpy(&v, ptr, 2); return std::to_string(v); }
                if (bits == 32 && need(4)) { int32_t v; std::memcpy(&v, ptr, 4); return std::to_string(v); }
                if (bits == 64 && need(8)) { int64_t v; std::memcpy(&v, ptr, 8); return std::to_string(v); }
                break;
            case NumericKind::UInt:
                if (bits == 8 && need(1)) { uint8_t v; std::memcpy(&v, ptr, 1); return std::to_string(v); }
                if (bits == 16 && need(2)) { uint16_t v; std::memcpy(&v, ptr, 2); return std::to_string(v); }
                if (bits == 32 && need(4)) { uint32_t v; std::memcpy(&v, ptr, 4); return std::to_string(v); }
                if (bits == 64 && need(8)) { uint64_t v; std::memcpy(&v, ptr, 8); return std::to_string(v); }
                break;
            case NumericKind::Float:
                if (bits == 16 && need(2)) {
                    uint16_t h; std::memcpy(&h, ptr, 2);
                    std::ostringstream os; os << HalfToFloat(h);
                    return os.str();
                }
                if (bits == 32 && need(4)) { float v; std::memcpy(&v, ptr, 4); std::ostringstream os; os << v; return os.str(); }
                if (bits == 64 && need(8)) { double v; std::memcpy(&v, ptr, 8); std::ostringstream os; os << v; return os.str(); }
                break;
            case NumericKind::Bool:
                if (need(4)) { uint32_t v; std::memcpy(&v, ptr, 4); return v ? "true" : "false"; }
                if (need(1)) { uint8_t v; std::memcpy(&v, ptr, 1); return v ? "true" : "false"; }
                break;
            default:
                break;
            }

            return "<unsupported>";
        }

        static std::string FormatLaneValue(
            const LayoutNode& node,
            const std::vector<std::byte>& data,
            size_t absoluteOffset,
            uint32_t lane)
        {
            if (absoluteOffset >= data.size()) {
                return "<out-of-bounds>";
            }

            const uint32_t bits = node.numeric.bits;
            const uint32_t lanes = std::max(1u, node.numeric.lanes);

            // Use reflected byte size first (important for bool, which often reports bits=1 but occupies 4 bytes in HLSL layouts).
            size_t elemBytes = 0;
            if (node.sizeBytes > 0 && lanes > 0 && (node.sizeBytes % lanes) == 0) {
                elemBytes = node.sizeBytes / lanes;
            }
            if (elemBytes == 0 && bits > 0) {
                elemBytes = (bits + 7) / 8; // round up
            }
            if (node.numeric.kind == NumericKind::None || elemBytes == 0) {
                return "<non-numeric>";
            }

            const size_t laneOffset = absoluteOffset + static_cast<size_t>(lane) * elemBytes;
            if (laneOffset >= data.size()) {
                return "<oob>";
            }

            const size_t remaining = data.size() - laneOffset;
            return FormatScalar(node.numeric.kind, bits, data.data() + laneOffset, remaining);
        }

        static void DrawLeafValue(const LayoutNode& node, const std::vector<std::byte>& data, size_t absoluteOffset)
        {
            const uint32_t rows = std::max(1u, node.numeric.rows);
            const uint32_t cols = std::max(1u, node.numeric.cols);
            const uint32_t lanes = std::max(1u, node.numeric.lanes);

            if (rows > 1 && cols > 1) {
                // Matrix layout
                ImGui::Indent();
                ImGui::Indent();
                for (uint32_t r = 0; r < rows; ++r) {
                    std::ostringstream row;
                    row << "[";
                    for (uint32_t c = 0; c < cols; ++c) {
                        if (c > 0) {
                            row << ", ";
                        }
                        const uint32_t lane = r * cols + c;
                        row << FormatLaneValue(node, data, absoluteOffset, lane);
                    }
                    row << "]";
                    ImGui::TextUnformatted(row.str().c_str());
                }
                ImGui::Unindent();
                ImGui::Unindent();
                return;
            }

            std::ostringstream os;
            os << "[";
            for (uint32_t lane = 0; lane < lanes; ++lane) {
                if (lane > 0) {
                    os << ", ";
                }
                os << FormatLaneValue(node, data, absoluteOffset, lane);
            }
            os << "]";
            ImGui::TextUnformatted(os.str().c_str());
        }

        static void DrawLayoutNode(const LayoutNode& node, const std::vector<std::byte>& data, size_t elementBaseOffset)
        {
            const size_t abs = elementBaseOffset + node.offsetBytes;
            const bool hasChildren = !node.children.empty();
            const bool isMatrix = (std::max(1u, node.numeric.rows) > 1 && std::max(1u, node.numeric.cols) > 1);

            std::ostringstream label;
            label << node.name;
            if (!node.typeName.empty()) {
                label << " : " << node.typeName;
            }

            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth;
            if (!hasChildren) {
                flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
            }

            const bool open = ImGui::TreeNodeEx(label.str().c_str(), flags);

            if (!hasChildren) {
                if (node.numeric.kind != NumericKind::None) {
                    if (!isMatrix) {
                        ImGui::SameLine();
                    }
                    DrawLeafValue(node, data, abs);
                }
                else {
                    ImGui::SameLine();
                    ImGui::TextDisabled("<non-numeric>");
                }
                return;
            }

            if (open) {
                for (auto const& c : node.children) {
                    ImGui::PushID(c.path.c_str());
                    DrawLayoutNode(c, data, elementBaseOffset);
                    ImGui::PopID();
                }
                ImGui::TreePop();
            }
        }

    } // namespace

    static inline char ToPrintableAscii(std::byte b) {
        unsigned char c = static_cast<unsigned char>(b);
        if (c >= 32 && c <= 126) {
            return static_cast<char>(c);
        }
        return '.';
    }

    void MemoryViewWidget::SaveCurrentResourceLayoutState() {
        if (currentResourceId_ == 0) {
            return;
        }

        ResourceLayoutState s{};
        s.structInput = std::string(structInputBuf_.data());
        s.diagnostics = reflectionDiagnostics_;
        s.rootSizeBytes = reflectedRootSizeBytes_;
        s.rootStrideBytes = reflectedRootStrideBytes_;
        s.reflectionValid = reflectionValid_;
        s.reflectedRoot = reflectedRoot_;
        s.goToElementInput = goToElementInput_;

        perResourceLayoutState_[currentResourceId_] = std::move(s);
    }

    void MemoryViewWidget::LoadResourceLayoutState(uint64_t resourceId) {
        std::fill(structInputBuf_.begin(), structInputBuf_.end(), 0);
        reflectionDiagnostics_.clear();
        reflectedRootSizeBytes_ = 0;
        reflectedRootStrideBytes_ = 0;
        reflectionValid_ = false;
        reflectedRoot_.reset();
        goToElementInput_ = 0;
        scrollToElement_ = -1;

        auto it = perResourceLayoutState_.find(resourceId);
        if (it == perResourceLayoutState_.end()) {
            return;
        }

        auto const& s = it->second;
        if (!s.structInput.empty()) {
			strncpy_s(structInputBuf_.data(), structInputBuf_.size(), s.structInput.c_str(), s.structInput.size());
        	structInputBuf_.back() = 0;
        }

        reflectionDiagnostics_ = s.diagnostics;
        reflectedRootSizeBytes_ = s.rootSizeBytes;
        reflectedRootStrideBytes_ = s.rootStrideBytes;
        reflectionValid_ = s.reflectionValid;
        reflectedRoot_ = s.reflectedRoot;
        goToElementInput_ = s.goToElementInput;
    }

    void MemoryViewWidget::Open(const std::string& passName, Resource* resource, const RangeSpec& range, MemoryViewRequestCaptureFn requestCapture) {
        std::scoped_lock lock(mutex_);

        result_.reset();
        waiting_ = false;
        status_.clear();

        if (!resource || passName.empty()) {
            status_ = "Missing pass or resource selection.";
            return;
        }

        PendingRequest req{};
        req.passName = passName;
        req.resource = resource;
        req.range = range;
        req.resourceId = resource->GetGlobalResourceID();
        req.resourceName = resource->GetName();
        pending_ = req;

        if (currentResourceId_ != 0 && currentResourceId_ != req.resourceId) {
            SaveCurrentResourceLayoutState();
            ReleasePreviewTexture();
            selectedMip_ = 0;
            selectedSlice_ = 0;
            previewDirty_ = true;
        }
        currentResourceId_ = req.resourceId;
        LoadResourceLayoutState(currentResourceId_);

        waiting_ = true;
        status_ = "Scheduling readback...";

        if (!requestCapture) {
            waiting_ = false;
            status_ = "Readback service unavailable.";
            return;
        }

        // Schedule capture. Callback may run later when GPU completes + ProcessReadbackRequests() copies data.
        requestCapture(
            passName,
            resource,
            range,
            [this](ReadbackCaptureResult&& r) {
                std::scoped_lock cbLock(mutex_);
                result_ = std::move(r);
                waiting_ = false;
                previewDirty_ = true;
                status_ = "Readback complete.";
            });
    }

    void MemoryViewWidget::Draw(bool* pOpen) {
        if (!pOpen || !*pOpen) {
            return;
        }

        if (!ImGui::Begin("Memory View", pOpen)) {
            ImGui::End();
            return;
        }

        PendingRequest pendingCopy{};
        bool havePending = false;
        std::optional<ReadbackCaptureResult> resultCopy;
        bool waiting = false;
        std::string status;

        {
            std::scoped_lock lock(mutex_);
            if (pending_) {
                pendingCopy = *pending_;
                havePending = true;
            }
            resultCopy = result_;
            waiting = waiting_;
            status = status_;
        }

        if (havePending) {
            ImGui::Text("Pass: %s", pendingCopy.passName.c_str());
            ImGui::Text("Resource: %s [%llu]", pendingCopy.resourceName.empty() ? "(unnamed)" : pendingCopy.resourceName.c_str(),
                static_cast<unsigned long long>(pendingCopy.resourceId));
        }
        else {
            ImGui::TextUnformatted("No capture requested.");
        }

        if (!status.empty()) {
            if (waiting) {
                ImGui::TextDisabled("%s", status.c_str());
            }
            else {
                ImGui::TextUnformatted(status.c_str());
            }
        }

        ImGui::Separator();

        if (!resultCopy.has_value()) {
            ImGui::TextUnformatted(waiting ? "Waiting for GPU readback..." : "No data yet.");
            ImGui::End();
            return;
        }

        const ReadbackCaptureResult& r = *resultCopy;

        if (ImGui::BeginTabBar("##MemoryViewTabs")) {
            if (ImGui::BeginTabItem("Buffer")) {
                if (r.desc.kind == ReadbackResourceKind::Buffer) {
                    DrawBufferView(r);
                }
                else {
                    ImGui::TextUnformatted("The captured resource is a texture.");
                }
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Texture")) {
                if (r.desc.kind == ReadbackResourceKind::Texture) {
                    DrawTextureView(r);
                }
                else {
                    ImGui::TextUnformatted("The captured resource is a buffer.");
                }
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }

        ImGui::End();
    }

    void MemoryViewWidget::ReleasePreviewTexture() {
        if (previewDescriptorIndex_ != UINT32_MAX && imguiFreeDesc_) {
            imguiFreeDesc_(previewDescriptorIndex_);
        }
        previewDescriptorIndex_ = UINT32_MAX;
        previewTextureId_ = 0;
        previewTexture_.Reset();
        previewUploadBuffer_.Reset();
        previewWidth_ = 0;
        previewHeight_ = 0;
    }

    void MemoryViewWidget::DrawTextureView(const ReadbackCaptureResult& r) {
        ImGui::Text("Format: %d", static_cast<int>(r.format));
        ImGui::Text("Dimensions: %ux%u (depth %u)", r.width, r.height, r.depth);
        ImGui::Text("Footprints: %zu", r.layouts.size());
        ImGui::Text("Data size: %llu bytes", static_cast<unsigned long long>(r.data.size()));

        if (r.layouts.empty() || r.data.empty()) {
            ImGui::TextDisabled("No texture data.");
            return;
        }

        // Determine mip/slice counts from footprints.
        // ReadbackCapturePass stores footprints as [slice][mip], so:
        //   subresourceIndex = slice * mipCount + mip
        // We need to figure out mipCount. Use the first footprint width progression.
        uint32_t mipCount = 1;
        uint32_t sliceCount = 1;
        {
            // Heuristic: count how many consecutive footprints have shrinking width
            // starting from index 0. That gives us mipCount for slice 0.
            for (uint32_t i = 1; i < static_cast<uint32_t>(r.layouts.size()); ++i) {
                if (r.layouts[i].width <= r.layouts[i - 1].width) {
                    mipCount = i + 1;
                } else {
                    break;
                }
            }
            if (mipCount > 0) {
                sliceCount = static_cast<uint32_t>(r.layouts.size()) / mipCount;
            }
            if (sliceCount == 0) sliceCount = 1;
            if (mipCount == 0) mipCount = 1;
        }

        ImGui::Separator();
        bool changed = false;
        if (mipCount > 1) {
            int maxMip = static_cast<int>(mipCount) - 1;
            changed |= ImGui::SliderInt("Mip Level", &selectedMip_, 0, maxMip);
            selectedMip_ = std::clamp(selectedMip_, 0, maxMip);
        }
        if (sliceCount > 1) {
            int maxSlice = static_cast<int>(sliceCount) - 1;
            changed |= ImGui::SliderInt("Array Slice", &selectedSlice_, 0, maxSlice);
            selectedSlice_ = std::clamp(selectedSlice_, 0, maxSlice);
        }
        if (changed) {
            previewDirty_ = true;
        }

        const uint32_t subresourceIndex = static_cast<uint32_t>(selectedSlice_) * mipCount + static_cast<uint32_t>(selectedMip_);
        if (subresourceIndex >= static_cast<uint32_t>(r.layouts.size())) {
            ImGui::TextDisabled("Subresource index out of range.");
            return;
        }

        const auto& fp = r.layouts[subresourceIndex];
        ImGui::Text("Subresource %u: %ux%u", subresourceIndex, fp.width, fp.height);
        if (r.format == rhi::Format::R32_UInt) {
            ImGui::TextDisabled("Preview uses stable false color for head-pointer indices; empty sentinel pixels are black.");
        }
        else if (r.format == rhi::Format::R32G32_UInt) {
            ImGui::TextDisabled("Preview uses stable false color for packed uint data; cleared pixels are black.");
        }
        else if (r.format == rhi::Format::R32G32B32_Float) {
            ImGui::TextDisabled("Preview normalizes float3 channels across this subresource for readability.");
        }
        else if (r.format == rhi::Format::R32G32B32A32_Float || r.format == rhi::Format::R32G32B32A32_Typeless) {
            ImGui::TextDisabled("Preview normalizes float4 channels across this subresource for readability; typeless is treated as float.");
        }

        if (previewDirty_) {
            previewDirty_ = false;

            // Decode to RGBA8
            auto rgba8 = DecodeSubresourceToRGBA8(r.data.data(), r.data.size(), fp, r.format);
            if (rgba8.empty()) {
                ReleasePreviewTexture();
                ImGui::TextDisabled("Format not supported for preview (%d).", static_cast<int>(r.format));
                return;
            }

            // Check if we have descriptor callbacks
            if (!imguiAllocDesc_ || !imguiFreeDesc_ || !imguiGpuHandle_) {
                ImGui::TextDisabled("ImGui descriptor callbacks not set.");
                return;
            }

            ReleasePreviewTexture();

            auto device = DeviceManager::GetInstance().GetDevice();

            // Create a staging (GPU-local) texture
            rhi::ResourceDesc texDesc{};
            texDesc.type = rhi::ResourceType::Texture2D;
            texDesc.heapType = rhi::HeapType::DeviceLocal;
            texDesc.debugName = "MemViewPreview";
            texDesc.texture.format = rhi::Format::R8G8B8A8_UNorm;
            texDesc.texture.width = fp.width;
            texDesc.texture.height = fp.height;
            texDesc.texture.depthOrLayers = 1;
            texDesc.texture.mipLevels = 1;
            texDesc.texture.sampleCount = 1;

            auto createResult = device.CreateCommittedResource(texDesc, previewTexture_);
            if (rhi::Failed(createResult) || !previewTexture_) {
                spdlog::error("MemoryViewWidget: failed to create preview texture.");
                return;
            }

            // Upload RGBA8 data via rhi::helpers
            rhi::CommandAllocatorPtr cmdAlloc;
            rhi::CommandListPtr cmdList;
            rhi::TimelinePtr fence;
            device.CreateCommandAllocator(rhi::QueueKind::Graphics, cmdAlloc);
            device.CreateCommandList(rhi::QueueKind::Graphics, cmdAlloc.Get(), cmdList);
            device.CreateTimeline(fence, 0, "MemViewUploadFence");

            rhi::helpers::SubresourceData subData{};
            subData.pData = rgba8.data();
            subData.rowPitch = fp.width * 4;
            subData.slicePitch = fp.width * fp.height * 4;

            previewUploadBuffer_ = rhi::helpers::UpdateTextureSubresources(
                device,
                cmdList.Get(),
                previewTexture_.Get(),
                rhi::Format::R8G8B8A8_UNorm,
                fp.width, fp.height, 1,
                1, 1,
                rhi::Span<const rhi::helpers::SubresourceData>(&subData, 1));

            cmdList->End();

            auto& clRef = cmdList.Get();
            auto& queue = DeviceManager::GetInstance().GetGraphicsQueue();
            queue.Submit(rhi::Span<rhi::CommandList>(&clRef, 1));
            queue.Signal({ fence->GetHandle(), 1 });
            fence->HostWait(1);

            // Create SRV in ImGui's descriptor heap
            previewDescriptorIndex_ = imguiAllocDesc_();

            rhi::SrvDesc srvDesc{};
            srvDesc.dimension = rhi::SrvDim::Texture2D;
            srvDesc.formatOverride = rhi::Format::R8G8B8A8_UNorm;
            srvDesc.tex2D.mipLevels = 1;
            srvDesc.tex2D.mostDetailedMip = 0;

            rhi::DescriptorSlot slot{ imguiHeapHandle_, previewDescriptorIndex_ };
            device.CreateShaderResourceView(slot, previewTexture_->GetHandle(), srvDesc);

            previewTextureId_ = imguiGpuHandle_(previewDescriptorIndex_);
            previewWidth_ = fp.width;
            previewHeight_ = fp.height;
        }

        if (previewTextureId_ != 0 && previewWidth_ > 0 && previewHeight_ > 0) {
            // Scale to fit available region while maintaining aspect ratio
            ImVec2 avail = ImGui::GetContentRegionAvail();
            float aspect = static_cast<float>(previewWidth_) / static_cast<float>(previewHeight_);
            float displayW = avail.x;
            float displayH = displayW / aspect;
            if (displayH > avail.y) {
                displayH = avail.y;
                displayW = displayH * aspect;
            }
            if (displayW < 1.0f) displayW = 1.0f;
            if (displayH < 1.0f) displayH = 1.0f;

            ImGui::Image(previewTextureId_, ImVec2(displayW, displayH));
        } else {
            ImGui::TextDisabled("No preview available.");
        }
    }

    void MemoryViewWidget::DrawBufferView(const ReadbackCaptureResult& r) {
        const size_t sizeBytes = r.data.size();
        ImGui::Text("Size: %llu bytes", static_cast<unsigned long long>(sizeBytes));

        ImGui::Separator();
        ImGui::TextUnformatted("Struct Layout (Slang)");
        ImGui::InputTextMultiline("##StructInput", structInputBuf_.data(), structInputBuf_.size(), ImVec2(-1.0f, 150.0f));

        const bool hasStructInput = HasNonWhitespace(structInputBuf_.data());

        if (ImGui::Button("Reflect Struct")) {
            reflectionDiagnostics_.clear();
            reflectionValid_ = false;
            reflectedRoot_.reset();
            reflectedRootSizeBytes_ = 0;
            reflectedRootStrideBytes_ = 0;
            goToElementInput_ = 0;
            scrollToElement_ = -1;

            if (!hasStructInput) {
                reflectionDiagnostics_ = "Struct input is empty.";
            }
            else {
                LayoutNode root{};
                std::string diagnostics;
                auto result = ReflectStructLayoutWithSlang(root, std::string(structInputBuf_.data()), &diagnostics);

                reflectionDiagnostics_ = diagnostics;

                if (SLANG_SUCCEEDED(result)) {
                    reflectedRootSizeBytes_ = root.sizeBytes;
                    reflectedRootStrideBytes_ = (root.strideBytes != 0) ? root.strideBytes : root.sizeBytes;
                    reflectedRoot_ = std::make_shared<LayoutNode>(std::move(root));
                    reflectionValid_ = (reflectedRoot_ != nullptr) && reflectedRootStrideBytes_ != 0;

                    if (!reflectionValid_ && reflectionDiagnostics_.empty()) {
                        reflectionDiagnostics_ = "Reflection succeeded but no usable root layout was produced.";
                    }
                }
                else if (reflectionDiagnostics_.empty()) {
                    reflectionDiagnostics_ = "Reflection failed.";
                }
            }

            SaveCurrentResourceLayoutState();
        }

        if (!reflectionDiagnostics_.empty()) {
            ImGui::TextWrapped("%s", reflectionDiagnostics_.c_str());
        }

        if (hasStructInput) {
            if (!reflectionValid_) {
                ImGui::TextDisabled("Reflect a valid struct to view typed values.");
            }
            else {
            const size_t stride = std::max<size_t>(1, reflectedRootStrideBytes_);
            const size_t count = sizeBytes / stride;

                ImGui::Text("Reflected stride: %llu bytes, element count: %llu",
                static_cast<unsigned long long>(stride),
                static_cast<unsigned long long>(count));

                ImGui::SetNextItemWidth(140.0f);
                ImGui::InputInt("Go To Index", &goToElementInput_, 0, 0);
                const bool enterPressed = ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false);
                const bool goIndexOnEnter = (ImGui::IsItemFocused() && enterPressed) || ImGui::IsItemDeactivatedAfterEdit();
                ImGui::SameLine();
                if (goIndexOnEnter || ImGui::Button("Go")) {
                    if (count > 0) {
                        goToElementInput_ = std::clamp(goToElementInput_, 0, static_cast<int>(count) - 1);
                        scrollToElement_ = goToElementInput_;
                    }
                    else {
                        goToElementInput_ = 0;
                        scrollToElement_ = -1;
                    }
                }

                if (count == 0) {
                    ImGui::TextDisabled("No complete elements in current buffer for reflected stride.");
                }
                else {
                    ImGui::BeginChild("##TypedElements", ImVec2(0, ImGui::GetContentRegionAvail().y), true);

                    int pendingScrollToElement = scrollToElement_;
                    bool scrollApplied = false;

                    auto drawElementRow = [&](int idx) {
                        ImGui::PushID(idx);

                        const size_t elementBase = static_cast<size_t>(idx) * stride;
                        char hdr[96];
                        std::snprintf(hdr, sizeof(hdr), "Element %d (base=0x%llX)", idx, static_cast<unsigned long long>(elementBase));

                        bool open = ImGui::TreeNodeEx(hdr, ImGuiTreeNodeFlags_SpanAvailWidth);

                        if (pendingScrollToElement == idx && !scrollApplied) {
                            ImGui::SetScrollHereY(0.25f);
                            scrollApplied = true;
                        }

                        if (open) {
                            if (reflectedRoot_->children.empty()) {
                                DrawLayoutNode(*reflectedRoot_, r.data, elementBase);
                            }
                            else {
                                for (auto const& c : reflectedRoot_->children) {
                                    ImGui::PushID(c.path.c_str());
                                    DrawLayoutNode(c, r.data, elementBase);
                                    ImGui::PopID();
                                }
                            }
                            ImGui::TreePop();
                        }

                        ImGui::PopID();
                    };

                    bool hasExpandedTopLevelElement = false;
                    if (pendingScrollToElement < 0) {
                        ImGuiStorage* storage = ImGui::GetStateStorage();
                        for (int idx = 0; idx < static_cast<int>(count); ++idx) {
                            ImGui::PushID(idx);
                            char hdr[96];
                            const size_t elementBase = static_cast<size_t>(idx) * stride;
                            std::snprintf(hdr, sizeof(hdr), "Element %d (base=0x%llX)", idx, static_cast<unsigned long long>(elementBase));
                            const ImGuiID id = ImGui::GetID(hdr);
                            ImGui::PopID();

                            if (storage && storage->GetBool(id, false)) {
                                hasExpandedTopLevelElement = true;
                                break;
                            }
                        }
                    }

                    if (hasExpandedTopLevelElement || pendingScrollToElement >= 0) {
                        for (int idx = 0; idx < static_cast<int>(count); ++idx) {
                            drawElementRow(idx);
                        }
                    }
                    else {
                        ImGuiListClipper clipper;
                        clipper.Begin(static_cast<int>(count));
                        while (clipper.Step()) {
                            for (int idx = clipper.DisplayStart; idx < clipper.DisplayEnd; ++idx) {
                                drawElementRow(idx);
                            }
                        }
                    }

                    if (pendingScrollToElement >= 0 && scrollApplied) {
                        scrollToElement_ = -1;
                    }

                    ImGui::EndChild();
                }
            }
        }

        if (!hasStructInput) {
            ImGui::Separator();
            ImGui::TextUnformatted("Raw Hex");

            int bpr = bytesPerRow_;
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::InputInt("Bytes/row", &bpr)) {
                bpr = std::clamp(bpr, 4, 64);
                bytesPerRow_ = bpr;
            }

            ImGui::SameLine();
            ImGui::SetNextItemWidth(170.0f);
            ImGui::InputScalar("Go To Byte", ImGuiDataType_U64, &goToByteOffsetInput_);
            const bool enterPressed = ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false);
            const bool goByteOnEnter = (ImGui::IsItemFocused() && enterPressed) || ImGui::IsItemDeactivatedAfterEdit();
            ImGui::SameLine();
            if (goByteOnEnter || ImGui::Button("Go##RawHex")) {
                if (sizeBytes == 0) {
                    goToByteOffsetInput_ = 0;
                    scrollToByteOffset_ = UINT64_MAX;
                    highlightedByteOffset_ = UINT64_MAX;
                }
                else {
                    const uint64_t maxOffset = static_cast<uint64_t>(sizeBytes - 1);
                    goToByteOffsetInput_ = std::min(goToByteOffsetInput_, maxOffset);
                    scrollToByteOffset_ = goToByteOffsetInput_;
                    highlightedByteOffset_ = goToByteOffsetInput_;
                }
            }

            if (sizeBytes == 0) {
                ImGui::TextUnformatted("(empty)");
                return;
            }

            const int bytesPerRow = std::clamp(bytesPerRow_, 4, 64);
            const int rowCount = static_cast<int>((sizeBytes + static_cast<size_t>(bytesPerRow) - 1) / static_cast<size_t>(bytesPerRow));

            if (highlightedByteOffset_ != UINT64_MAX && highlightedByteOffset_ >= static_cast<uint64_t>(sizeBytes)) {
                highlightedByteOffset_ = UINT64_MAX;
            }

            ImGui::BeginChild("##HexView", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);

            if (scrollToByteOffset_ != UINT64_MAX) {
                const uint64_t targetRow = scrollToByteOffset_ / static_cast<uint64_t>(bytesPerRow);
                const float rowHeight = ImGui::GetTextLineHeightWithSpacing();
                const float targetY = static_cast<float>(targetRow) * rowHeight;
                const float viewH = ImGui::GetWindowHeight();
                const float scrollY = std::max(0.0f, targetY - viewH * 0.25f);
                ImGui::SetScrollY(scrollY);
                scrollToByteOffset_ = UINT64_MAX;
            }

            const ImVec4 highlightedByteColor = ImVec4(0.60f, 0.80f, 1.00f, 1.00f); // light blue

            ImGuiListClipper clip;
            clip.Begin(rowCount);
            while (clip.Step()) {
                for (int row = clip.DisplayStart; row < clip.DisplayEnd; ++row) {
                    const size_t base = static_cast<size_t>(row) * static_cast<size_t>(bytesPerRow);

                    char prefix[32];
                    std::snprintf(prefix, sizeof(prefix), "%08llX  ", static_cast<unsigned long long>(base));
                    ImGui::TextUnformatted(prefix);
                    ImGui::SameLine(0.0f, 0.0f);

                    // Hex bytes with per-byte highlight
                    for (int i = 0; i < bytesPerRow; ++i) {
                        const size_t idx = base + static_cast<size_t>(i);
                        char token[8];
                        if (idx < sizeBytes) {
                            const unsigned v = static_cast<unsigned>(std::to_integer<unsigned char>(r.data[idx]));
                            std::snprintf(token, sizeof(token), "%02X ", v);
                            if (highlightedByteOffset_ != UINT64_MAX && idx == static_cast<size_t>(highlightedByteOffset_)) {
                                ImGui::TextColored(highlightedByteColor, "%s", token);
                            } else {
                                ImGui::TextUnformatted(token);
                            }
                        }
                        else {
                            std::snprintf(token, sizeof(token), "   ");
                            ImGui::TextUnformatted(token);
                        }
                        ImGui::SameLine(0.0f, 0.0f);
                    }

                    // ASCII
                    char ascii[96];
                    int n = 0;
                    n += std::snprintf(ascii + n, sizeof(ascii) - n, " |");
                    for (int i = 0; i < bytesPerRow; ++i) {
                        const size_t idx = base + static_cast<size_t>(i);
                        if (idx < sizeBytes) {
                            ascii[n++] = ToPrintableAscii(r.data[idx]);
                        }
                        else {
                            ascii[n++] = ' ';
                        }
                    }
                    ascii[n++] = '|';
                    ascii[n++] = 0;

                    ImGui::TextUnformatted(ascii);
                }
            }

            ImGui::EndChild();
        }
    }

} // namespace ui
