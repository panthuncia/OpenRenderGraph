#include "DebugUI/MemoryIntrospectionWidget.h"

#include <cmath>
#include <unordered_map>


namespace org {

namespace ui {

    static bool RadioButtonView(const char* label, int* v, int buttonValue) {
        ImGui::SameLine();
        return ImGui::RadioButton(label, v, buttonValue);
    }

    void MemoryIntrospectionWidget::PushFrameSample(double timeSeconds, uint64_t totalBytes) {
        const double y = static_cast<double>(totalBytes);

        const size_t N = rtSeries_.x.size();

        // Target spacing so N samples can cover maxSeconds.
        double minDt = 0.0;
        if (rt_.maxSeconds > 0) {
            minDt = static_cast<double>(rt_.maxSeconds) / static_cast<double>(std::max<size_t>(1, N - 1));
        }

        // First sample
        if (rtSeries_.count == 0) {
            rtSeries_.push(timeSeconds, y);
            rtLastCommittedTime_ = timeSeconds;
            return;
        }

        // If we're sampling too fast, overwrite the last point instead of pushing a new one.
        if (minDt > 0.0 && rtLastCommittedTime_ >= 0.0 && (timeSeconds - rtLastCommittedTime_) < minDt) {
            const size_t lastIdx = (rtSeries_.head + N - 1) % N; // last written
            rtSeries_.x[lastIdx] = timeSeconds; // keep it "current" visually
            rtSeries_.y[lastIdx] = y;
            return;
        }

        // Commit a new sample
        rtSeries_.push(timeSeconds, y);
        rtLastCommittedTime_ = timeSeconds;
    }



    void MemoryIntrospectionWidget::Draw(bool* pOpen,
        const MemorySnapshot* snapshot,
        const FrameGraphSnapshot* frameGraph) {
        if (pOpen && !*pOpen) {
            return;
        }

        if (!ImGui::Begin("Memory Introspection", pOpen)) {
            ImGui::End();
            return;
        }

        DrawToolbar();
        ImGui::Separator();

        // Dummy data only for a host that passes none (a preview of the widget); a host's empty snapshot or missing frame graph
        // is shown as such, never as made-up rows.
        if (snapshot && snapshot->categories.empty() && snapshot->resources.empty() && snapshot->heaps.empty()
            && snapshot->reconciliation.empty()) {
            ImGui::TextUnformatted("No tracked allocations.");
            ImGui::End();
            return;
        }
        if (snapshot && !frameGraph && view_ == ViewMode::BatchTimeline) {
            ImGui::TextWrapped("The host provides no frame-graph batches for this graph.");
            ImGui::End();
            return;
        }

        MemorySnapshot dummyMem;
        const MemorySnapshot* ms = snapshot;
        if (ms == nullptr) {
            dummyMem = MakeDummySnapshot();
            ms = &dummyMem;
        }
        MemorySnapshot localMem = *ms;
        if (localMem.totalBytes == 0) {
            localMem.totalBytes = ComputeTotalBytes(localMem);
        }

        FrameGraphSnapshot dummyFG;
        const FrameGraphSnapshot* fg = frameGraph;
        if (fg == nullptr || fg->batches.empty()) {
            dummyFG = MakeDummyFrameGraph();
            fg = &dummyFG;
        }

        if (snapshot) {
            PushGroupSample(localMem);
        }

        switch (view_) {
        case ViewMode::Pie:
            if (groupBy_ != GroupBy::Category && !localMem.resources.empty()) {
                MemorySnapshot grouped;
                grouped.categories = GroupRows(localMem, groupBy_);
                grouped.totalBytes = CountedBytes(localMem);
                DrawPieView(grouped);
            }
            else {
                DrawPieView(localMem);
            }
            break;
        case ViewMode::List:     DrawListView(localMem); break;
        case ViewMode::Heaps:    DrawHeapsView(localMem); break;
        case ViewMode::Reconciliation: DrawReconciliationView(localMem); break;
        case ViewMode::Timeline:
            timelineMode_ = TimelineMode::RealTime;
            if (rt_.stacked && snapshot && !localMem.resources.empty()) {
                DrawStackedTimeline();
            }
            else {
                DrawTimelineView(*fg, localMem);
            }
            break;
        case ViewMode::BatchTimeline:
            timelineMode_ = TimelineMode::FrameGraph;
            DrawTimelineView(*fg, localMem);
            break;
        }

        ImGui::End();
    }

    void MemoryIntrospectionWidget::DrawToolbar() {
        int v = static_cast<int>(view_);

        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("View:");
        RadioButtonView("Pie", &v, static_cast<int>(ViewMode::Pie));
        RadioButtonView("List", &v, static_cast<int>(ViewMode::List));
        RadioButtonView("Timeline", &v, static_cast<int>(ViewMode::Timeline));
        RadioButtonView("Batch Timeline", &v, static_cast<int>(ViewMode::BatchTimeline));
        RadioButtonView("Heaps", &v, static_cast<int>(ViewMode::Heaps));
        RadioButtonView("Reconciliation", &v, static_cast<int>(ViewMode::Reconciliation));
        view_ = static_cast<ViewMode>(v);

        if (view_ == ViewMode::Pie || view_ == ViewMode::Timeline) {
            static const char* groupNames[] = { "Category", "Owner", "Source", "Heap", "Type" };
            int g = static_cast<int>(groupBy_);
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted("Group by:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImGui::GetFontSize() * 9.0f);
            if (ImGui::Combo("##mem_group_by", &g, groupNames, IM_ARRAYSIZE(groupNames))) {
                groupBy_ = static_cast<GroupBy>(g);
            }
            if (view_ == ViewMode::Timeline) {
                ImGui::SameLine();
                ImGui::Checkbox("Stacked", &rt_.stacked);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(ImGui::GetFontSize() * 6.0f);
                ImGui::SliderInt("Window (s)", &rt_.maxSeconds, 5, 120);
            }
        }
    }

    // ----------------- Grouping -----------------

    uint64_t MemoryIntrospectionWidget::CountedBytes(const MemorySnapshot& s) {
        uint64_t sum = 0;
        for (const auto& r : s.resources) {
            if (!(r.flags & MemoryRowFlag_NotCounted)) {
                sum += r.bytes;
            }
        }
        return sum;
    }

    std::vector<MemoryCategorySlice> MemoryIntrospectionWidget::GroupRows(const MemorySnapshot& s, GroupBy a_groupBy) {
        if (a_groupBy == GroupBy::Category) {
            return s.categories;
        }

        std::unordered_map<std::string, uint64_t> bytes;
        for (const auto& r : s.resources) {
            if (r.flags & MemoryRowFlag_NotCounted) {
                continue;
            }
            const std::string* key = nullptr;
            switch (a_groupBy) {
            case GroupBy::Owner:  key = &r.owner; break;
            case GroupBy::Source: key = &r.source; break;
            case GroupBy::Heap:   key = &r.heap; break;
            default:              key = &r.type; break;
            }
            bytes[key->empty() ? std::string("Unspecified") : *key] += r.bytes;
        }

        std::vector<MemoryCategorySlice> out;
        out.reserve(bytes.size());
        for (auto& [label, b] : bytes) {
            out.push_back({ label, b });
        }
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.bytes > b.bytes; });
        return out;
    }

    void MemoryIntrospectionWidget::PushGroupSample(const MemorySnapshot& s) {
        if (s.resources.empty() && s.categories.empty()) {
            return;
        }
        if (groupSamplesBy_ != groupBy_) {
            groupSamples_.clear();
            groupSamplesBy_ = groupBy_;
        }

        const double now = ImGui::GetTime();
        const double window = static_cast<double>(std::max(1, rt_.maxSeconds));
        const double minDt = window / 300.0;

        GroupSample sample;
        sample.time = now;
        // The top level of each group ("CS/Feature" -> "CS") keeps the stack readable.
        for (auto& slice : GroupRows(s, groupBy_)) {
            std::string label = slice.label;
            if (const auto slash = label.find('/'); slash != std::string::npos && groupBy_ != GroupBy::Category) {
                label.resize(slash);
            }
            auto it = std::find_if(sample.groups.begin(), sample.groups.end(), [&](const auto& g) { return g.label == label; });
            if (it == sample.groups.end()) {
                sample.groups.push_back({ std::move(label), slice.bytes });
            }
            else {
                it->bytes += slice.bytes;
            }
        }
        for (const auto& heap : s.heaps) {
            if (heap.deviceLocal) {
                sample.usageBytes += heap.usageBytes;
                sample.budgetBytes += heap.budgetBytes;
            }
        }

        if (!groupSamples_.empty() && now - groupSamples_.back().time < minDt) {
            groupSamples_.back() = std::move(sample);
        }
        else {
            groupSamples_.push_back(std::move(sample));
        }

        const auto stale = std::find_if(groupSamples_.begin(), groupSamples_.end(),
            [&](const GroupSample& g) { return g.time >= now - window; });
        groupSamples_.erase(groupSamples_.begin(), stale);
    }

    // ----------------- Dummy data -----------------

    uint64_t MemoryIntrospectionWidget::ComputeTotalBytes(const MemorySnapshot& s) {
        if (!s.categories.empty()) {
            uint64_t sum = 0;
            for (auto& c : s.categories) {
                sum += c.bytes;
            }
            return sum;
        }
        if (!s.resources.empty()) {
            return CountedBytes(s);
        }
        return 0;
    }

    MemorySnapshot MemoryIntrospectionWidget::MakeDummySnapshot() {
        MemorySnapshot s;
        s.categories = {
            {"Textures/Material",        640ull * 1024 * 1024},
            {"Textures/Fullscreen RTs",  512ull * 1024 * 1024},
            {"Buffers/Vertex+Index",     160ull * 1024 * 1024},
            {"Buffers/Other",             60ull * 1024 * 1024},
            {"AccelStructs/BLAS+TLAS",   180ull * 1024 * 1024},
            {"Other/Misc",                90ull * 1024 * 1024},
        };
        s.resources = {
            {"gbuffer_albedo",     "Texture2D", 128ull * 1024 * 1024, 1001},
            {"gbuffer_normals",    "Texture2D", 128ull * 1024 * 1024, 1002},
            {"gbuffer_depth",      "Texture2D",  64ull * 1024 * 1024, 1003},
            {"material_atlas_0",   "Texture2D", 256ull * 1024 * 1024, 2001},
            {"material_atlas_1",   "Texture2D", 384ull * 1024 * 1024, 2002},
            {"meshlets",           "Buffer",     96ull * 1024 * 1024, 3002},
        };
        s.totalBytes = ComputeTotalBytes(s);
        return s;
    }

    FrameGraphSnapshot MemoryIntrospectionWidget::MakeDummyFrameGraph() {
        FrameGraphSnapshot fg;
        fg.batches = {
            {"Setup",       220ull * 1024 * 1024, 220ull * 1024 * 1024, 220ull * 1024 * 1024, 0,                      false, {"Culling", "Depth Prepass"}},
            {"GBuffer",     780ull * 1024 * 1024, 720ull * 1024 * 1024, 780ull * 1024 * 1024,  60ull * 1024 * 1024, true,  {"VB Resolve", "GBuffer Resolve"}},
            {"Lighting",    940ull * 1024 * 1024, 860ull * 1024 * 1024, 940ull * 1024 * 1024,  80ull * 1024 * 1024, true,  {"Tiled Light", "SSAO", "Shadows"}},
            {"Reflections", 610ull * 1024 * 1024, 540ull * 1024 * 1024, 610ull * 1024 * 1024,  70ull * 1024 * 1024, false, {"SSR", "Denoise"}},
            {"Post",        520ull * 1024 * 1024, 470ull * 1024 * 1024, 520ull * 1024 * 1024,  50ull * 1024 * 1024, false, {"Bloom", "Tonemap", "TAA"}},
            {"UI/Compose",  180ull * 1024 * 1024, 180ull * 1024 * 1024, 180ull * 1024 * 1024, 0,                      false, {"UI", "Composite"}},
        };
        return fg;
    }

    namespace {

        static inline void TrimInPlace(std::string& s) {
            auto is_ws = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
            while (!s.empty() && is_ws(static_cast<unsigned char>(s.front()))) {
                s.erase(s.begin());
            }
            while (!s.empty() && is_ws(static_cast<unsigned char>(s.back()))) {
                s.pop_back();
            }
        }

        static inline void SplitMajorSub(std::string label, std::string& outMajor, std::string& outSub) {
            // Accept "Major/Sub" or "Major:Sub". If no delimiter -> Major=label, Sub=""
            TrimInPlace(label);
            size_t pos = label.find('/');
            if (pos == std::string::npos) {
                pos = label.find(':');
            }

            if (pos == std::string::npos) {
                outMajor = label;
                outSub.clear();
                return;
            }
            outMajor = label.substr(0, pos);
            outSub = label.substr(pos + 1);
            TrimInPlace(outMajor);
            TrimInPlace(outSub);
        }

        static inline double WrapAngle(double a) {
            const double twoPi = 6.2831853071795864769;
            while (a < 0.0) {
                a += twoPi;
            }
            while (a >= twoPi) {
                a -= twoPi;
            }
            return a;
        }

        static inline bool AngleInRange(double a, double a0, double a1) {
            // a, a0, a1 in [0,2pi). Handles wrap.
            if (a0 <= a1) {
                return a >= a0 && a <= a1;
            }
            // wrapped
            return a >= a0 || a <= a1;
        }

        static void DrawDonutSlice(ImDrawList* dl, ImVec2 c, float r0, float r1,
            float a0, float a1, ImU32 fill, ImU32 border, float borderThickness) {
            if (a1 <= a0 || r1 <= 0.0f) {
                return;
            }
            if (r0 < 0.0f) {
                r0 = 0.0f;
            }
            if (r0 > r1) {
                std::swap(r0, r1);
            }

            const float arc = a1 - a0;

            // Segment count heuristic: more segments for larger radius/arc.
            int seg = static_cast<int>(std::ceil(static_cast<double>(arc) * static_cast<double>(r1) / 10.0));
            if (seg < 8) {
                seg = 8;
            }
            if (seg > 160) {
                seg = 160;
            }

            const int vtxCount = (seg + 1) * 2;
            const int idxCount = seg * 6;

            dl->PrimReserve(idxCount, vtxCount);
            const int baseVtx = dl->_VtxCurrentIdx;

            // Write vertices (outer, inner) pairs.
            for (int i = 0; i <= seg; ++i) {
                const float t = static_cast<float>(i) / static_cast<float>(seg);
                const float a = a0 + arc * t;
                const float cs = std::cos(a);
                const float sn = std::sin(a);

                const ImVec2 uv = ImGui::GetFontTexUvWhitePixel();

                dl->PrimWriteVtx(ImVec2(c.x + cs * r1, c.y + sn * r1), uv, fill);
                dl->PrimWriteVtx(ImVec2(c.x + cs * r0, c.y + sn * r0), uv, fill);
            }

            // Indices: two triangles per segment (quad strip), sharing vertices.
            for (int i = 0; i < seg; ++i) {
                const ImDrawIdx o0 = static_cast<ImDrawIdx>(baseVtx + i * 2 + 0);
                const ImDrawIdx i0 = static_cast<ImDrawIdx>(baseVtx + i * 2 + 1);
                const ImDrawIdx o1 = static_cast<ImDrawIdx>(baseVtx + (i + 1) * 2 + 0);
                const ImDrawIdx i1 = static_cast<ImDrawIdx>(baseVtx + (i + 1) * 2 + 1);

                dl->PrimWriteIdx(o0); dl->PrimWriteIdx(o1); dl->PrimWriteIdx(i1);
                dl->PrimWriteIdx(o0); dl->PrimWriteIdx(i1); dl->PrimWriteIdx(i0);
            }

            if (borderThickness > 0.0f) {
                // Outer arc
                dl->PathClear();
                dl->PathArcTo(c, r1, a0, a1, seg);
                dl->PathStroke(border, false, borderThickness);

                // Inner arc (only if donut, not a pie slice)
                if (r0 > 0.0f) {
                    dl->PathClear();
                    dl->PathArcTo(c, r0, a0, a1, seg);
                    dl->PathStroke(border, false, borderThickness);
                }

                // Radial edges
                const ImVec2 p0o(c.x + std::cos(a0) * r1, c.y + std::sin(a0) * r1);
                const ImVec2 p0i(c.x + std::cos(a0) * r0, c.y + std::sin(a0) * r0);
                const ImVec2 p1o(c.x + std::cos(a1) * r1, c.y + std::sin(a1) * r1);
                const ImVec2 p1i(c.x + std::cos(a1) * r0, c.y + std::sin(a1) * r0);
                dl->AddLine(p0i, p0o, border, borderThickness);
                dl->AddLine(p1i, p1o, border, borderThickness);
            }
        }

        static ImU32 Shade(ImU32 c, float factor /*0..1 darken*/) {
            const ImU32 a = (c >> IM_COL32_A_SHIFT) & 0xFF;
            ImU32 r = (c >> IM_COL32_R_SHIFT) & 0xFF;
            ImU32 g = (c >> IM_COL32_G_SHIFT) & 0xFF;
            ImU32 b = (c >> IM_COL32_B_SHIFT) & 0xFF;
            r = static_cast<ImU32>(r * factor);
            g = static_cast<ImU32>(g * factor);
            b = static_cast<ImU32>(b * factor);
            return IM_COL32(r, g, b, a);
        }
        struct CatColorLUT {
            std::unordered_map<std::string, ImU32> byLabel;
            ImU32 fallback = IM_COL32(200, 200, 200, 200);

            ImU32 get(const std::string& label) const {
                auto it = byLabel.find(label);
                return (it != byLabel.end()) ? it->second : fallback;
            }
        };

        static CatColorLUT BuildCategoryColorLUT(const ui::MemorySnapshot& s) {
            // Same palette as DrawPieView
            static const ImU32 majorCols[] = {
                IM_COL32(90, 170, 255, 230),
                IM_COL32(255, 140,  90, 230),
                IM_COL32(120, 220, 140, 230),
                IM_COL32(220, 140, 255, 230),
                IM_COL32(255, 220, 120, 230),
                IM_COL32(140, 220, 220, 230),
                IM_COL32(200, 200, 200, 230),
            };
            const int paletteN = static_cast<int>(sizeof(majorCols) / sizeof(majorCols[0]));

            struct SubAgg { std::string sub; uint64_t bytes = 0; };
            struct MajorAgg {
                std::string maj;
                uint64_t bytes = 0;
                std::unordered_map<std::string, uint64_t> subBytes; // sub -> bytes
            };

            std::vector<MajorAgg> majors;

            auto findMajor = [&](const std::string& m) -> MajorAgg* {
                for (auto& x : majors) {
                    if (x.maj == m) {
                        return &x;
                    }
                }
                majors.push_back(MajorAgg{ m });
                return &majors.back();
                };

            for (auto& c : s.categories) {
                if (c.bytes == 0) {
                    continue;
                }
                std::string maj, sub;
                SplitMajorSub(c.label, maj, sub);
                if (maj.empty()) {
                    maj = "Other";
                }
                if (sub.empty()) {
                    sub = maj;
                }

                auto* M = findMajor(maj);
                M->bytes += c.bytes;
                M->subBytes[sub] += c.bytes;
            }

            std::sort(majors.begin(), majors.end(),
                [](const MajorAgg& a, const MajorAgg& b) { return a.bytes > b.bytes; });

            CatColorLUT lut;
            lut.byLabel.reserve(s.categories.size() + 16);

            for (int mi = 0; mi < static_cast<int>(majors.size()); ++mi) {
                const ImU32 majorColor = majorCols[mi % paletteN];

                std::vector<std::pair<std::string, uint64_t>> subs;
                subs.reserve(majors[mi].subBytes.size());
                for (auto& kv : majors[mi].subBytes) {
                    subs.push_back(kv);
                }
                std::sort(subs.begin(), subs.end(),
                    [](auto& a, auto& b) { return a.second > b.second; });

                for (int si = 0; si < static_cast<int>(subs.size()); ++si) {
                    const float shade = 0.95f - 0.08f * static_cast<float>(si);
                    const ImU32 subCol = Shade(majorColor, std::max(0.45f, shade));
                    lut.byLabel[majors[mi].maj + "/" + subs[si].first] = subCol;
                }
            }

            return lut;
        }
    } // anonymous

    void ui::MemoryIntrospectionWidget::DrawPieView(const MemorySnapshot& s) {
        const uint64_t totalBytes = (s.totalBytes != 0) ? s.totalBytes : ComputeTotalBytes(s);
        ImGui::Text("Total: %s", FormatBytes(totalBytes).c_str());

        if (s.categories.empty()) {
            ImGui::TextUnformatted("No category data.");
            return;
        }

        // Build Major -> Sub slices from "Major/Sub" labels
        struct Sub { std::string label; uint64_t bytes; };
        struct Major { std::string label; std::vector<Sub> subs; uint64_t bytes = 0; };

        std::vector<Major> majors;
        majors.reserve(8);

        auto findMajor = [&](const std::string& m) -> Major* {
            for (auto& x : majors) {
                if (x.label == m) {
                    return &x;
                }
            }
            majors.push_back(Major{ m });
            return &majors.back();
            };

        for (auto& c : s.categories) {
            if (c.bytes == 0) {
                continue;
            }

            std::string maj, sub;
            SplitMajorSub(c.label, maj, sub);
            if (maj.empty()) {
                maj = "Other";
            }

            Major* M = findMajor(maj);
            if (sub.empty()) {
                // If no subcategory provided, treat as a single sub with same label as major.
                sub = maj;
            }
            M->subs.push_back(Sub{ sub, c.bytes });
            M->bytes += c.bytes;
        }

        if (majors.empty()) {
            ImGui::TextUnformatted("No non-zero category slices.");
            return;
        }

        // sort majors by size (largest first)
        std::sort(majors.begin(), majors.end(), [](const Major& a, const Major& b) { return a.bytes > b.bytes; });

        // Adjustable plot height
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        float plotH = pie_.autoHeight ? std::min(std::max(avail.y, 200.0f), 480.0f)
            : std::min(pie_.heightPx, std::max(avail.y, 120.0f));

        if (!ImPlot::BeginPlot("##MemoryPieHier", ImVec2(-1.0f, plotH),
            ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
            return;
        }

        // plot canvas setup
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
        ImPlot::SetupAxesLimits(-1, 1, -1, 1, ImGuiCond_Always);

        ImDrawList* dl = ImPlot::GetPlotDrawList();
        const ImVec2 plotPos = ImPlot::GetPlotPos();
        const ImVec2 plotSize = ImPlot::GetPlotSize();
        const ImVec2 center = ImVec2(plotPos.x + plotSize.x * 0.5f, plotPos.y + plotSize.y * 0.5f);

        const float outerR = pie_.radius * 0.5f * std::min(plotSize.x, plotSize.y);
        const float innerR = outerR * pie_.innerRatio;

        // A small fixed palette for *majors*; subs are shaded variants
        static const ImU32 majorCols[] = {
            IM_COL32(90, 170, 255, 230),
            IM_COL32(255, 140,  90, 230),
            IM_COL32(120, 220, 140, 230),
            IM_COL32(220, 140, 255, 230),
            IM_COL32(255, 220, 120, 230),
            IM_COL32(140, 220, 220, 230),
            IM_COL32(200, 200, 200, 230),
        };
        const int paletteN = static_cast<int>(sizeof(majorCols) / sizeof(majorCols[0]));

        // Hover tracking
        int hoveredMajor = -1;
        int hoveredSub = -1;

        // Mouse polar coords
        const ImVec2 mp = ImGui::GetMousePos();
        const float dx = mp.x - center.x;
        const float dy = mp.y - center.y;
        const float mr = std::sqrt(dx * dx + dy * dy);
        double ma = WrapAngle(std::atan2(static_cast<double>(dy), static_cast<double>(dx))); // [0,2pi)

        // Start angle at 12 o'clock (top), going clockwise
        double aCursor = -1.5707963267948966; // -pi/2
        const double twoPi = 6.2831853071795864769;

        // Stroke colors
        const ImU32 border = IM_COL32(0, 0, 0, 255);
        const ImU32 sepCol = IM_COL32(0, 0, 0, 255);

        for (int mi = 0; mi < static_cast<int>(majors.size()); ++mi) {
            Major& M = majors[mi];
            const double majFrac = (totalBytes == 0) ? 0.0 : static_cast<double>(M.bytes) / static_cast<double>(totalBytes);
            const double a0 = aCursor;
            const double a1 = aCursor + majFrac * twoPi;
            const ImU32 majorColor = majorCols[mi % paletteN];

            // Inner ring: major slice
            DrawDonutSlice(dl, center, 0.0f, innerR, static_cast<float>(a0), static_cast<float>(a1),
                majorColor, border, 1.0f);

            // Hover test for major (inner ring)
            if (mr <= innerR) {
                const double wa0 = WrapAngle(a0), wa1 = WrapAngle(a1);
                if (AngleInRange(ma, wa0, wa1)) {
                    hoveredMajor = mi;
                }
            }

            // Outer ring: subs inside major
            // group tiny subs into "Other"
            std::sort(M.subs.begin(), M.subs.end(), [](const Sub& a, const Sub& b) { return a.bytes > b.bytes; });

            std::vector<Sub> subs;
            subs.reserve(M.subs.size() + 1);
            uint64_t other = 0;

            for (auto& ss : M.subs) {
                const double pctOfTotal = (totalBytes == 0) ? 0.0 : (100.0 * static_cast<double>(ss.bytes) / static_cast<double>(totalBytes));
                if (pctOfTotal < static_cast<double>(pie_.minSlicePct)) {
                    other += ss.bytes;
                }
                else {
                    subs.push_back(ss);
                }
            }
            if (other > 0) {
                subs.push_back(Sub{ "Other", other });
            }

            double subCursor = a0;
            for (int si = 0; si < static_cast<int>(subs.size()); ++si) {
                const Sub& S = subs[si];
                const double subFrac = (M.bytes == 0) ? 0.0 : static_cast<double>(S.bytes) / static_cast<double>(M.bytes);
                const double sa0 = subCursor;
                const double sa1 = subCursor + subFrac * (a1 - a0);

                // shade subs progressively so they're distinguishable inside the major
                const float shade = 0.95f - 0.08f * static_cast<float>(si);
                const ImU32 subCol = Shade(majorColor, std::max(0.45f, shade));

                DrawDonutSlice(dl, center, innerR, outerR, static_cast<float>(sa0), static_cast<float>(sa1),
                    subCol, border, 1.0f);

                // Hover test for subs (outer ring)
                if (mr >= innerR && mr <= outerR) {
                    const double wsa0 = WrapAngle(sa0), wsa1 = WrapAngle(sa1);
                    if (AngleInRange(ma, wsa0, wsa1)) {
                        hoveredMajor = mi;
                        hoveredSub = si;
                    }
                }

                subCursor = sa1;
            }

            // Major separators: radial line at start angle
            if (pie_.showMajorSeparators) {
                const float t = pie_.majorSeparatorThickness;
                const float ca = std::cos(static_cast<float>(a0));
                const float sa = std::sin(static_cast<float>(a0));
                ImVec2 p0 = center;
                ImVec2 p1 = ImVec2(center.x + outerR * ca, center.y + outerR * sa);
                dl->AddLine(p0, p1, sepCol, t);
            }

            aCursor = a1;
        }

        // draw final separator at end
        if (pie_.showMajorSeparators) {
            const float t = pie_.majorSeparatorThickness;
            const float ca = std::cos(static_cast<float>(aCursor));
            const float sa = std::sin(static_cast<float>(aCursor));
            dl->AddLine(center, ImVec2(center.x + outerR * ca, center.y + outerR * sa), sepCol, t);
        }

        // Tooltip
        if ((hoveredMajor >= 0) && ImPlot::IsPlotHovered()) {
            const auto& M = majors[hoveredMajor];

            std::string subLabel;
            uint64_t subBytes = 0;

            if (hoveredSub >= 0) {
                // rebuild the same grouped subs list to match indices (small cost, ok for now)
                std::vector<Sub> subs = M.subs;
                std::sort(subs.begin(), subs.end(), [](const Sub& a, const Sub& b) { return a.bytes > b.bytes; });

                std::vector<Sub> grouped;
                uint64_t other = 0;
                for (auto& ss : subs) {
                    const double pctOfTotal = (totalBytes == 0) ? 0.0 : (100.0 * static_cast<double>(ss.bytes) / static_cast<double>(totalBytes));
                    if (pctOfTotal < static_cast<double>(pie_.minSlicePct)) {
                        other += ss.bytes;
                    }
                    else {
                        grouped.push_back(ss);
                    }
                }
                if (other > 0) {
                    grouped.push_back(Sub{ "Other", other });
                }

                if (hoveredSub < static_cast<int>(grouped.size())) {
                    subLabel = grouped[hoveredSub].label;
                    subBytes = grouped[hoveredSub].bytes;
                }
            }

            ImGui::BeginTooltip();
            ImGui::Text("%s", M.label.c_str());
            ImGui::Text("Major: %s", FormatBytes(M.bytes).c_str());
            if (!subLabel.empty()) {
                ImGui::Separator();
                ImGui::Text("Sub: %s", subLabel.c_str());
                ImGui::Text("Sub: %s", FormatBytes(subBytes).c_str());
                if (totalBytes != 0) {
                    ImGui::Text("Sub %% of total: %.2f%%", 100.0 * static_cast<double>(subBytes) / static_cast<double>(totalBytes));
                }
            }
            else if (totalBytes != 0) {
                ImGui::Text("Major %% of total: %.2f%%", 100.0 * static_cast<double>(M.bytes) / static_cast<double>(totalBytes));
            }
            ImGui::EndTooltip();
        }

        ImPlot::EndPlot();

        // Legend
        ImGui::Separator();
        for (int mi = 0; mi < static_cast<int>(majors.size()); ++mi) {
            const auto& M = majors[mi];
            const double pct = (totalBytes == 0) ? 0.0 : (100.0 * static_cast<double>(M.bytes) / static_cast<double>(totalBytes));
            ImGui::BulletText("%s: %s%s",
                M.label.c_str(),
                FormatBytes(M.bytes).c_str(),
                (std::string("  (") + std::to_string(pct).substr(0, 4) + "%)").c_str());
        }
    }

    namespace {
        static std::string FlagsText(uint32_t flags) {
            std::string out;
            auto add = [&](uint32_t bit, const char* text) {
                if (flags & bit) {
                    if (!out.empty()) {
                        out += ", ";
                    }
                    out += text;
                }
            };
            add(MemoryRowFlag_Aliased, "aliased");
            add(MemoryRowFlag_NotCounted, "not counted");
            add(MemoryRowFlag_Imported, "imported");
            add(MemoryRowFlag_Internal, "internal");
            add(MemoryRowFlag_Unattributed, "unattributed");
            return out;
        }

        enum ListColumn : int { ColName, ColOwner, ColType, ColSource, ColHeap, ColSize, ColPct, ColFlags, ColUid, ColCallsite, ColCount };
    }

    void MemoryIntrospectionWidget::DrawListView(const MemorySnapshot& s) {
        if (s.resources.empty()) {
            ImGui::TextUnformatted("No per-resource data.");
            return;
        }

        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 18.0f);
        list_.filter.Draw("Filter (name, owner, type, source, call site; '-' excludes)");

        std::vector<const MemoryResourceRow*> rows;
        rows.reserve(s.resources.size());
        uint64_t filteredBytes = 0;
        for (auto& r : s.resources) {
            if (list_.filter.IsActive()) {
                if (!list_.filter.PassFilter(r.name.c_str()) && !list_.filter.PassFilter(r.type.c_str())
                    && !list_.filter.PassFilter(r.owner.c_str()) && !list_.filter.PassFilter(r.source.c_str())
                    && !list_.filter.PassFilter(r.callsite.c_str())) {
                    continue;
                }
            }
            rows.push_back(&r);
            if (!(r.flags & MemoryRowFlag_NotCounted)) {
                filteredBytes += r.bytes;
            }
        }

        const uint64_t total = (s.totalBytes != 0) ? s.totalBytes : ComputeTotalBytes(s);
        ImGui::Text("Resources: %d   Shown: %s of %s", static_cast<int>(rows.size()), FormatBytes(filteredBytes).c_str(),
            FormatBytes(total).c_str());

        ImGuiTableFlags flags =
            ImGuiTableFlags_BordersV | ImGuiTableFlags_BordersOuterH | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable
            | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Sortable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Reorderable;

        const float tableH = ImGui::GetContentRegionAvail().y;
        if (ImGui::BeginTable("##mem_list", ColCount, flags, ImVec2(-1.0f, tableH))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch, 0.40f, ColName);
            ImGui::TableSetupColumn("Owner", ImGuiTableColumnFlags_WidthStretch, 0.25f, ColOwner);
            ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthStretch, 0.20f, ColType);
            ImGui::TableSetupColumn("Source", ImGuiTableColumnFlags_WidthFixed, 90.0f, ColSource);
            ImGui::TableSetupColumn("Heap", ImGuiTableColumnFlags_WidthFixed, 60.0f, ColHeap);
            ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_DefaultSort
                | ImGuiTableColumnFlags_PreferSortDescending, 100.0f, ColSize);
            ImGui::TableSetupColumn("%", ImGuiTableColumnFlags_WidthFixed, 55.0f, ColPct);
            ImGui::TableSetupColumn("Flags", ImGuiTableColumnFlags_WidthFixed, 110.0f, ColFlags);
            ImGui::TableSetupColumn("UID", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_DefaultHide, 110.0f, ColUid);
            ImGui::TableSetupColumn("Call site", ImGuiTableColumnFlags_WidthStretch, 0.30f, ColCallsite);
            ImGui::TableHeadersRow();

            // Percent sorts as size does; flags by their bits.
            if (const ImGuiTableSortSpecs* specs = ImGui::TableGetSortSpecs(); specs && specs->SpecsCount > 0) {
                const ImGuiTableColumnSortSpecs spec = specs->Specs[0];
                const bool descending = spec.SortDirection == ImGuiSortDirection_Descending;
                auto key = [&](const MemoryResourceRow* r) -> const std::string& {
                    switch (spec.ColumnUserID) {
                    case ColOwner:    return r->owner;
                    case ColType:     return r->type;
                    case ColSource:   return r->source;
                    case ColHeap:     return r->heap;
                    case ColCallsite: return r->callsite;
                    default:          return r->name;
                    }
                };
                std::stable_sort(rows.begin(), rows.end(), [&](const MemoryResourceRow* a, const MemoryResourceRow* b) {
                    switch (spec.ColumnUserID) {
                    case ColSize:
                    case ColPct:   return descending ? a->bytes > b->bytes : a->bytes < b->bytes;
                    case ColUid:   return descending ? a->uid > b->uid : a->uid < b->uid;
                    case ColFlags: return descending ? a->flags > b->flags : a->flags < b->flags;
                    default:       return descending ? key(a) > key(b) : key(a) < key(b);
                    }
                });
            }

            const int pageSize = std::max(1, list_.pageSize);
            const int count = std::min(static_cast<int>(rows.size()), pageSize);

            ImGuiListClipper clipper;
            clipper.Begin(count);
            while (clipper.Step()) {
                for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                    const MemoryResourceRow& r = *rows[i];
                    const bool counted = !(r.flags & MemoryRowFlag_NotCounted);
                    const double pct = (total == 0 || !counted) ? 0.0 : (100.0 * static_cast<double>(r.bytes) / static_cast<double>(total));

                    ImGui::TableNextRow();
                    if (!counted) {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                    }

                    ImGui::TableSetColumnIndex(ColName); ImGui::TextUnformatted(r.name.c_str());
                    ImGui::TableSetColumnIndex(ColOwner); ImGui::TextUnformatted(r.owner.c_str());
                    ImGui::TableSetColumnIndex(ColType); ImGui::TextUnformatted(r.type.c_str());
                    ImGui::TableSetColumnIndex(ColSource); ImGui::TextUnformatted(r.source.c_str());
                    ImGui::TableSetColumnIndex(ColHeap); ImGui::TextUnformatted(r.heap.c_str());
                    ImGui::TableSetColumnIndex(ColSize); ImGui::TextUnformatted(FormatBytes(r.bytes).c_str());
                    ImGui::TableSetColumnIndex(ColPct); ImGui::Text("%.2f", pct);
                    ImGui::TableSetColumnIndex(ColFlags); ImGui::TextUnformatted(FlagsText(r.flags).c_str());
                    ImGui::TableSetColumnIndex(ColUid); ImGui::Text("%llu", static_cast<unsigned long long>(r.uid));
                    ImGui::TableSetColumnIndex(ColCallsite); ImGui::TextUnformatted(r.callsite.c_str());

                    if (!counted) {
                        ImGui::PopStyleColor();
                    }
                }
            }

            if (static_cast<int>(rows.size()) > pageSize) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(ColName);
                ImGui::TextDisabled("Showing the first %d of %d rows; narrow the filter to see the rest.", pageSize,
                    static_cast<int>(rows.size()));
            }

            ImGui::EndTable();
        }
    }

    // ----------------- Heaps and reconciliation -----------------

    void MemoryIntrospectionWidget::DrawHeapsView(const MemorySnapshot& s) {
        if (s.heaps.empty()) {
            ImGui::TextUnformatted("The host reports no heaps.");
            return;
        }

        const ImU32 attributedCol = IM_COL32(90, 170, 255, 230);
        const ImU32 residualCol = IM_COL32(255, 200, 90, 230);
        const ImU32 budgetCol = IM_COL32(255, 90, 90, 255);

        for (const auto& heap : s.heaps) {
            ImGui::PushID(heap.label.c_str());
            const uint64_t residual = heap.usageBytes > heap.attributedBytes ? heap.usageBytes - heap.attributedBytes : 0;
            ImGui::Text("%s%s", heap.label.c_str(), heap.deviceLocal ? "" : " (host)");
            ImGui::SameLine();
            ImGui::TextDisabled("usage %s, budget %s, attributed %s, unattributed %s", FormatBytes(heap.usageBytes).c_str(),
                FormatBytes(heap.budgetBytes).c_str(), FormatBytes(heap.attributedBytes).c_str(), FormatBytes(residual).c_str());

            // One bar on the scale of max(usage, budget): attributed, then the rest of usage, and the budget as a mark.
            const double scale = static_cast<double>(std::max<uint64_t>({ heap.usageBytes, heap.budgetBytes, heap.attributedBytes, 1 }));
            const float width = std::max(1.0f, ImGui::GetContentRegionAvail().x);
            const float height = ImGui::GetFrameHeight();
            const ImVec2 p0 = ImGui::GetCursorScreenPos();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(p0, ImVec2(p0.x + width, p0.y + height), ImGui::GetColorU32(ImGuiCol_FrameBg), 3.0f);
            const float xAttributed = p0.x + width * static_cast<float>(static_cast<double>(std::min(heap.attributedBytes, heap.usageBytes)) / scale);
            const float xUsage = p0.x + width * static_cast<float>(static_cast<double>(heap.usageBytes) / scale);
            dl->AddRectFilled(p0, ImVec2(xAttributed, p0.y + height), attributedCol, 3.0f);
            if (xUsage > xAttributed) {
                dl->AddRectFilled(ImVec2(xAttributed, p0.y), ImVec2(xUsage, p0.y + height), residualCol);
            }
            if (heap.budgetBytes) {
                const float xBudget = p0.x + width * static_cast<float>(static_cast<double>(heap.budgetBytes) / scale);
                dl->AddLine(ImVec2(xBudget, p0.y - 2.0f), ImVec2(xBudget, p0.y + height + 2.0f), budgetCol, 2.0f);
            }
            ImGui::Dummy(ImVec2(width, height));
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("Attributed: %s", FormatBytes(heap.attributedBytes).c_str());
                ImGui::Text("Unattributed: %s", FormatBytes(residual).c_str());
                if (heap.budgetBytes) {
                    ImGui::Text("Usage: %.1f%% of budget", 100.0 * static_cast<double>(heap.usageBytes) / static_cast<double>(heap.budgetBytes));
                }
                ImGui::EndTooltip();
            }
            ImGui::Spacing();
            ImGui::PopID();
        }

        ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(attributedCol), "Attributed");
        ImGui::SameLine();
        ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(residualCol), "Unattributed");
        ImGui::SameLine();
        ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(budgetCol), "| Budget");
    }

    void MemoryIntrospectionWidget::DrawReconciliationView(const MemorySnapshot& s) {
        if (s.reconciliation.empty()) {
            ImGui::TextUnformatted("The host provides no reconciliation.");
            return;
        }

        auto kindColor = [](MemoryReconciliationKind kind) {
            switch (kind) {
            case MemoryReconciliationKind::Total:    return ImGui::GetStyleColorVec4(ImGuiCol_Text);
            case MemoryReconciliationKind::Residual: return ImVec4(1.0f, 0.78f, 0.35f, 1.0f);
            case MemoryReconciliationKind::Opaque:   return ImVec4(1.0f, 0.45f, 0.45f, 1.0f);
            default:                                 return ImVec4(0.55f, 0.8f, 1.0f, 1.0f);
            }
        };

        const ImGuiTableFlags flags = ImGuiTableFlags_BordersV | ImGuiTableFlags_BordersOuterH | ImGuiTableFlags_RowBg
            | ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY;
        if (ImGui::BeginTable("##mem_reconciliation", 4, flags, ImVec2(-1.0f, ImGui::GetContentRegionAvail().y))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Breakdown", ImGuiTableColumnFlags_WidthStretch, 0.5f);
            ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 100.0f);
            ImGui::TableSetupColumn("% of parent", ImGuiTableColumnFlags_WidthFixed, 160.0f);
            ImGui::TableSetupColumn("Note", ImGuiTableColumnFlags_WidthStretch, 0.5f);
            ImGui::TableHeadersRow();

            // A row's parent is the nearest row above it one depth shallower.
            std::vector<uint64_t> parentBytes;
            for (const auto& row : s.reconciliation) {
                const size_t depth = static_cast<size_t>(std::max(0, row.depth));
                parentBytes.resize(depth + 1);
                parentBytes[depth] = row.bytes;
                const uint64_t parent = depth > 0 ? parentBytes[depth - 1] : 0;
                const float indent = static_cast<float>(depth) * ImGui::GetStyle().IndentSpacing;

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                if (indent > 0.0f) {
                    ImGui::Indent(indent);
                }
                ImGui::TextColored(kindColor(row.kind), "%s", row.label.c_str());
                if (indent > 0.0f) {
                    ImGui::Unindent(indent);
                }
                ImGui::TableSetColumnIndex(1);
                ImGui::TextUnformatted(FormatBytes(row.bytes).c_str());
                ImGui::TableSetColumnIndex(2);
                if (parent) {
                    const float fraction = static_cast<float>(static_cast<double>(row.bytes) / static_cast<double>(parent));
                    char overlay[32];
                    std::snprintf(overlay, sizeof(overlay), "%.1f%%", 100.0 * fraction);
                    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, kindColor(row.kind));
                    ImGui::ProgressBar(std::min(fraction, 1.0f), ImVec2(-1.0f, 0.0f), overlay);
                    ImGui::PopStyleColor();
                }
                ImGui::TableSetColumnIndex(3);
                ImGui::TextWrapped("%s", row.note.c_str());
            }
            ImGui::EndTable();
        }
    }

    // ----------------- Timeline view: Real-time + Frame-graph -----------------

    void MemoryIntrospectionWidget::DrawTimelineView(const FrameGraphSnapshot& fg, const MemorySnapshot& ms) {
        if (timelineMode_ == TimelineMode::RealTime) {
            DrawRealTimeTimeline();
        }
        else {
            DrawFrameGraphTimeline(fg, ms);
        }
    }

    enum class ByteUnit { B, KiB, MiB, GiB };

    static inline const char* UnitLabel(ByteUnit u) {
        switch (u) {
        case ByteUnit::B:   return "B";
        case ByteUnit::KiB: return "KiB";
        case ByteUnit::MiB: return "MiB";
        case ByteUnit::GiB: return "GiB";
        default:            return "B";
        }
    }

    static inline double UnitDiv(ByteUnit u) {
        switch (u) {
        case ByteUnit::B:   return 1.0;
        case ByteUnit::KiB: return 1024.0;
        case ByteUnit::MiB: return 1024.0 * 1024.0;
        case ByteUnit::GiB: return 1024.0 * 1024.0 * 1024.0;
        default:            return 1.0;
        }
    }

    static inline ByteUnit ChooseUnitForRange(double maxBytes) {
        const double KiB = 1024.0;
        const double MiB = 1024.0 * 1024.0;
        const double GiB = 1024.0 * 1024.0 * 1024.0;
        if (maxBytes >= GiB) {
            return ByteUnit::GiB;
        }
        if (maxBytes >= MiB) {
            return ByteUnit::MiB;
        }
        if (maxBytes >= KiB) {
            return ByteUnit::KiB;
        }
        return ByteUnit::B;
    }

    void MemoryIntrospectionWidget::DrawRealTimeTimeline() {
        rtSeries_.ordered(tmpX_, tmpY_);
        if (tmpX_.size() < 2) {
            ImGui::TextUnformatted("No timeline samples yet. Call PushFrameSample(timeSeconds, totalBytes) each frame.");
            return;
        }

        // Visible X window
        double tMax = tmpX_.back();
        double tMin = tmpX_.front();
        if (rt_.maxSeconds > 0) {
            tMin = tMax - static_cast<double>(rt_.maxSeconds);
        }

        auto it = std::lower_bound(tmpX_.begin(), tmpX_.end(), tMin);
        size_t start = (it == tmpX_.begin()) ? 0 : static_cast<size_t>(it - tmpX_.begin() - 1);
        const size_t n = tmpX_.size() - start;

        const double* xPtr = tmpX_.data() + start;
        const double* yPtrBytes = tmpY_.data() + start;

        // Compute min/max bytes in visible window for unit + Y padding
        double minBytes = yPtrBytes[0];
        double maxBytes = yPtrBytes[0];
        for (size_t i = 0; i < n; ++i) {
            minBytes = std::min(minBytes, yPtrBytes[i]);
            maxBytes = std::max(maxBytes, yPtrBytes[i]);
        }
        if (maxBytes <= 0.0) {
            maxBytes = 1.0;
        }

        const ByteUnit unit = ChooseUnitForRange(maxBytes);
        const double div = UnitDiv(unit);

        // Scale visible Y into chosen unit
        tmpYScaled_.resize(n);
        for (size_t i = 0; i < n; ++i) {
            tmpYScaled_[i] = yPtrBytes[i] / div;
        }

        // Y limits with padding (prevents line clipping when flat at extremes)
        double yMin = minBytes / div;
        double yMax = maxBytes / div;
        double yRange = yMax - yMin;

        double pad = 0.0;
        if (yRange < 1e-9) {
            // Flat line: give it a visible band
            pad = std::max(0.1, yMax * 0.05);
            yMin -= pad;
            yMax += pad;
        }
        else {
            pad = std::max(0.1, yRange * 0.05);
            yMin -= pad;
            yMax += pad;
        }
        if (yMin < 0.0) {
            yMin = 0.0; // memory can't be negative
        }

        char yLabel[32];
        std::snprintf(yLabel, sizeof(yLabel), "Total (%s)", UnitLabel(unit));

        if (ImPlot::BeginPlot("##MemoryTimelineRT", ImVec2(-1.0f, -1.0f), ImPlotFlags_NoTitle)) {
            // Don't use AutoFit here; set explicit limits with padding.
            ImPlot::SetupAxes("Time (s)", yLabel, ImPlotAxisFlags_None, ImPlotAxisFlags_None);

            // X window always enforced when sliding
            if (rt_.maxSeconds > 0) {
                ImPlot::SetupAxisLimits(ImAxis_X1, tMin, tMax, ImGuiCond_Always);
            }

            // Y auto-scale with padding
        	ImPlot::SetupAxisLimits(ImAxis_Y1, yMin, yMax, ImGuiCond_Always);

            ImPlot::PlotLine("Total", xPtr, tmpYScaled_.data(), static_cast<int>(n));

            ImPlot::EndPlot();
        }
    }


    // ----------------- Stacked real-time timeline -----------------

    void MemoryIntrospectionWidget::DrawStackedTimeline() {
        if (groupSamples_.size() < 2) {
            ImGui::TextUnformatted("Collecting samples...");
            return;
        }

        // The groups largest in the latest sample get a band each; the rest are summed into "Other".
        std::vector<std::string> bands;
        {
            auto latest = groupSamples_.back().groups;
            std::sort(latest.begin(), latest.end(), [](const auto& a, const auto& b) { return a.bytes > b.bytes; });
            for (const auto& g : latest) {
                if (static_cast<int>(bands.size()) >= std::max(1, rt_.maxGroups)) {
                    break;
                }
                bands.push_back(g.label);
            }
        }
        const size_t bandCount = bands.size() + 1;  // + "Other"
        const size_t n = groupSamples_.size();

        std::vector<double> xs(n);
        std::vector<std::vector<double>> cumulative(bandCount + 1, std::vector<double>(n, 0.0));
        std::vector<double> usage(n), budget(n);
        double maxBytes = 1.0;
        bool hasHeaps = false;
        for (size_t i = 0; i < n; ++i) {
            const auto& sample = groupSamples_[i];
            xs[i] = sample.time - groupSamples_.back().time;
            std::vector<double> perBand(bandCount, 0.0);
            for (const auto& g : sample.groups) {
                const auto it = std::find(bands.begin(), bands.end(), g.label);
                perBand[it == bands.end() ? bands.size() : static_cast<size_t>(it - bands.begin())] += static_cast<double>(g.bytes);
            }
            for (size_t b = 0; b < bandCount; ++b) {
                cumulative[b + 1][i] = cumulative[b][i] + perBand[b];
            }
            usage[i] = static_cast<double>(sample.usageBytes);
            budget[i] = static_cast<double>(sample.budgetBytes);
            hasHeaps = hasHeaps || sample.usageBytes != 0;
            maxBytes = std::max({ maxBytes, cumulative[bandCount][i], usage[i], budget[i] });
        }

        const ByteUnit unit = ChooseUnitForRange(maxBytes);
        const double div = UnitDiv(unit);
        for (auto& series : cumulative) {
            for (auto& v : series) {
                v /= div;
            }
        }
        for (size_t i = 0; i < n; ++i) {
            usage[i] /= div;
            budget[i] /= div;
        }

        char yLabel[32];
        std::snprintf(yLabel, sizeof(yLabel), "Bytes (%s)", UnitLabel(unit));
        if (ImPlot::BeginPlot("##MemoryTimelineStacked", ImVec2(-1.0f, -1.0f), ImPlotFlags_NoTitle)) {
            ImPlot::SetupAxes("Time (s)", yLabel, ImPlotAxisFlags_None, ImPlotAxisFlags_None);
            ImPlot::SetupAxisLimits(ImAxis_X1, -static_cast<double>(rt_.maxSeconds), 0.0, ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, maxBytes / div * 1.08, ImGuiCond_Always);

            const int count = static_cast<int>(n);
            for (size_t b = 0; b < bandCount; ++b) {
                const char* label = b < bands.size() ? bands[b].c_str() : "Other";
                ImPlot::PlotShaded(label, xs.data(), cumulative[b].data(), cumulative[b + 1].data(), count);
            }
            if (hasHeaps) {
                ImPlot::PlotLine("Heap usage", xs.data(), usage.data(), count);
                ImPlot::PlotLine("Budget", xs.data(), budget.data(), count);
            }
            ImPlot::EndPlot();
        }
    }

    // ----------------- Frame-graph timeline rendering -----------------

    std::vector<MemoryIntrospectionWidget::BatchLayout>
        MemoryIntrospectionWidget::BuildBatchLayouts(const FrameGraphSnapshot& fg,
            const FrameGraphTimelineSettings& opts,
            double* outTotalW) {
        const double wT = opts.blockWidthTransitions;
        const double wP = opts.blockWidthPasses;
        const double wE = opts.blockWidthBatchEnd;
        const double g = opts.blockGap;
        const double l = opts.blockLeftTransitions;
        const double rGutter = 0.02;

        std::vector<BatchLayout> out;
        out.resize(fg.batches.size());

        double cursor = 0.0;
        for (int i = 0; i < static_cast<int>(fg.batches.size()); ++i) {
            const auto& b = fg.batches[i];
            BatchLayout bl{};
            bl.baseX = cursor;

            bl.t0 = bl.baseX + l;
            bl.t1 = bl.t0 + wT;

            bl.p0 = bl.t1 + g;
            bl.p1 = bl.p0 + wP;

            bl.hasEnd = b.hasEndTransitions;
            if (bl.hasEnd) {
                bl.e0 = bl.p1 + g;
                bl.e1 = bl.e0 + wE;
                bl.width = (bl.e1 - bl.baseX) + rGutter;
            }
            else {
                bl.width = (bl.p1 - bl.baseX) + rGutter;
            }

            out[i] = bl;
            cursor += bl.width;
        }

        if (outTotalW) {
            *outTotalW = out.empty() ? 1.0 : (out.back().baseX + out.back().width);
        }
        return out;
    }

    void MemoryIntrospectionWidget::DrawBlock(ImDrawList* dl,
        const ImPlotPoint& minP, const ImPlotPoint& maxP,
        ImU32 fill, ImU32 border, float rad) {
        ImVec2 a = ImPlot::PlotToPixels(minP);
        ImVec2 b = ImPlot::PlotToPixels(maxP);
        if (a.x > b.x) {
            std::swap(a.x, b.x);
        }
        if (a.y > b.y) {
            std::swap(a.y, b.y);
        }
        dl->AddRectFilled(a, b, fill, rad);
        dl->AddRect(a, b, border, rad);
    }

    bool MemoryIntrospectionWidget::IsMouseOver(const ImPlotPoint& minP, const ImPlotPoint& maxP) {
        ImVec2 mp = ImGui::GetMousePos();
        ImVec2 a = ImPlot::PlotToPixels(minP);
        ImVec2 b = ImPlot::PlotToPixels(maxP);
        if (a.x > b.x) {
            std::swap(a.x, b.x);
        }
        if (a.y > b.y) {
            std::swap(a.y, b.y);
        }
        return mp.x >= a.x && mp.x <= b.x && mp.y >= a.y && mp.y <= b.y;
    }

    void MemoryIntrospectionWidget::DrawFrameGraphTimeline(const FrameGraphSnapshot& fg, const MemorySnapshot& ms) {
        if (fg.batches.empty()) {
            ImGui::TextUnformatted("No frame-graph batches.");
            return;
        }

        double totalW = 1.0;
        auto layouts = BuildBatchLayouts(fg, fg_, &totalW);

        // Precompute max footprint/live total for axis limits
        uint64_t maxBytes = 0;
        for (auto& b : fg.batches) {
            maxBytes = std::max(maxBytes, b.footprintBytes);
            maxBytes = std::max(maxBytes, b.peakLiveBytes);
            maxBytes = std::max(maxBytes, b.peakNaiveLiveBytes);
        }
        const double maxY = BytesToMiB(maxBytes);
        const double yPad = (maxY > 0.0) ? (maxY * 0.15) : 1.0;

        ImGui::TextUnformatted("Stacked bars: touched footprint    Line: peak live total");

        const ImVec2 avail = ImGui::GetContentRegionAvail();
        const float topH = std::max(120.0f, avail.y - 4.0f);

        if (ImPlot::BeginPlot("##BatchFootprints", ImVec2(-1.0f, topH), ImPlotFlags_CanvasOnly)) {
            ImPlot::SetupAxes(nullptr, nullptr,
                ImPlotAxisFlags_NoDecorations,
                ImPlotAxisFlags_NoDecorations);

            ImPlot::SetupAxisLimits(ImAxis_X1, -0.1, totalW + 0.1, ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, maxY + yPad, ImGuiCond_Always);

            ImDrawList* dl = ImPlot::GetPlotDrawList();

            // Vertical separators at batch boundaries (helps alignment with timeline below)
            for (int i = 0; i <= static_cast<int>(layouts.size()); ++i) {
                double x = (i == static_cast<int>(layouts.size())) ? totalW : layouts[i].baseX;
                ImVec2 p0 = ImPlot::PlotToPixels(ImPlotPoint(x, 0.0));
                ImVec2 p1 = ImPlot::PlotToPixels(ImPlotPoint(x, maxY + yPad));
                dl->AddLine(p0, p1, IM_COL32(180, 180, 180, 64), (i % 5 == 0) ? 2.0f : 1.0f);
            }

            // Build color mapping from the same categories used by the pie.
            const CatColorLUT colors = BuildCategoryColorLUT(ms);
            const ImU32 liveLineColor = IM_COL32(32, 32, 32, 255);
            const ImU32 livePointColor = IM_COL32(255, 230, 120, 255);

            int hoveredBatch = -1;
            std::string hoveredCat;
            uint64_t hoveredCatBytes = 0;
            std::vector<ImVec2> livePoints;
            livePoints.reserve(fg.batches.size());

            for (int i = 0; i < static_cast<int>(fg.batches.size()); ++i) {
                const auto& b = fg.batches[i];
                const auto& L = layouts[i];

                // Hover/click region = whole column
                const bool overColumn = IsMouseOver(ImPlotPoint(L.baseX, 0.0),
                    ImPlotPoint(L.baseX + L.width, maxY + yPad));
                if (overColumn && hoveredBatch < 0) {
                    hoveredBatch = i;
                }
                if (overColumn && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    selectedBatch_ = i;
                }

                livePoints.push_back(ImPlot::PlotToPixels(ImPlotPoint(L.baseX + L.width * 0.5, BytesToMiB(b.peakLiveBytes))));

                // Stack segments bottom-up
                double y0 = 0.0;

                // Optional: keep a deterministic order by sorting per batch (largest on top/bottom, your call)
                std::vector<MemoryCategorySlice> slices = b.categories;
                std::sort(slices.begin(), slices.end(),
                    [](const auto& a, const auto& b) { return a.bytes > b.bytes; });

                for (const auto& s : slices) {
                    if (s.bytes == 0) {
                        continue;
                    }

                    const double segH = BytesToMiB(s.bytes);
                    const double y1 = y0 + segH;

                    ImPlotPoint segMin(L.baseX, y0);
                    ImPlotPoint segMax(L.baseX + L.width, y1);

                    const bool overSeg = IsMouseOver(segMin, segMax);
                    if (overSeg) {
                        hoveredBatch = i;
                        hoveredCat = s.label;
                        hoveredCatBytes = s.bytes;
                    }

                    // Slight highlight on hover or selection
                    const bool highlight = (selectedBatch_ == i) || overSeg;
                    ImU32 fill = colors.get(s.label);
                    if (highlight) {
                        fill = IM_COL32((fill >> 0) & 0xFF, (fill >> 8) & 0xFF, (fill >> 16) & 0xFF, 255);
                    }

                    DrawBlock(dl, segMin, segMax, fill, IM_COL32(0, 0, 0, 255), 0.0f);
                    y0 = y1;
                }

                // Outline the full bar (selection cue)
                const bool selected = (selectedBatch_ == i);
                if (selected) {
                    DrawBlock(dl,
                        ImPlotPoint(L.baseX, 0.0),
                        ImPlotPoint(L.baseX + L.width, BytesToMiB(b.footprintBytes)),
                        IM_COL32(0, 0, 0, 0), IM_COL32(0, 0, 0, 255), 2.0f);
                }

                // assert/visualize mismatch if y0 != y?
            }

            if (livePoints.size() >= 2) {
                dl->AddPolyline(livePoints.data(), static_cast<int>(livePoints.size()), liveLineColor, 0, 2.0f);
            }
            for (int i = 0; i < static_cast<int>(livePoints.size()); ++i) {
                const float radius = (hoveredBatch == i || selectedBatch_ == i) ? 5.0f : 4.0f;
                dl->AddCircleFilled(livePoints[i], radius, livePointColor);
                dl->AddCircle(livePoints[i], radius, liveLineColor, 0, 1.5f);
            }

			// Tooltip
            if (ImPlot::IsPlotHovered() && hoveredBatch >= 0) {
                const auto& b = fg.batches[hoveredBatch];
                ImGui::BeginTooltip();
                ImGui::Text("Batch %d: %s", hoveredBatch, b.label.c_str());
                ImGui::Separator();
                ImGui::Text("Touched footprint: %s", FormatBytes(b.footprintBytes).c_str());
                ImGui::Text("Peak live total: %s", FormatBytes(b.peakLiveBytes).c_str());
                ImGui::Text("Peak naive live: %s", FormatBytes(b.peakNaiveLiveBytes).c_str());
                if (b.aliasSavingsBytes > 0) {
                    ImGui::Text("Aliasing savings: %s", FormatBytes(b.aliasSavingsBytes).c_str());
                }

                if (!hoveredCat.empty()) {
                    ImGui::Separator();
                    ImGui::Text("Category: %s", hoveredCat.c_str());
                    ImGui::Text("Size: %s", FormatBytes(hoveredCatBytes).c_str());
                }

                // Full breakdown list (like pie legend/tooltip)
                ImGui::Separator();
                ImGui::TextUnformatted("Touched breakdown:");
                // show top N to avoid massive tooltip spam
                int shown = 0;
                for (auto& s : b.categories) {
                    if (s.bytes == 0) {
                        continue;
                    }
                    if (shown++ >= 12) {
                        ImGui::TextDisabled("...");
                        break;
                    }
                    ImGui::BulletText("%s: %s", s.label.c_str(), FormatBytes(s.bytes).c_str());
                }
                ImGui::EndTooltip();
            }

            ImPlot::EndPlot();
        }

    }

} // namespace ui


} // namespace org
