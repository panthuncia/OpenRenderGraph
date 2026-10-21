#include "DebugUI/ScheduleInspector.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

#include "imgui.h"

namespace org::debugui {

namespace {

using Trace = ScheduleTrace;

constexpr float kGutter = 96.0f;       // lane labels
constexpr float kTop = 18.0f;          // batch index row
constexpr float kLaneHeight = 62.0f;
constexpr float kLaneGap = 8.0f;
constexpr float kBatchPad = 5.0f;
constexpr float kBatchHeader = 14.0f;  // the batch's own label inside its box
constexpr float kBatchGap = 30.0f;     // room for the wait arrows between batch columns
constexpr float kPassGap = 2.0f;
constexpr float kWaitHandle = 5.0f;    // the hover target on a wait arrow
constexpr size_t kTooltipLines = 40;   // per list in a tooltip; the selection panel lists everything

ImU32 QueueColor(uint32_t queue) {
    static constexpr ImU32 colors[] = {
        IM_COL32(70, 130, 220, 255),   // graphics
        IM_COL32(90, 190, 110, 255),   // compute
        IM_COL32(210, 190, 90, 255),   // copy
        IM_COL32(180, 130, 200, 255),
    };
    return colors[queue % (sizeof(colors) / sizeof(colors[0]))];
}
ImU32 Scaled(ImU32 color, float factor, int alpha) {
    const auto channel = [&](int shift) {
        const float value = static_cast<float>((color >> shift) & 0xFF) * factor;
        return static_cast<int>(std::clamp(value, 0.0f, 255.0f));
    };
    return IM_COL32(channel(IM_COL32_R_SHIFT), channel(IM_COL32_G_SHIFT), channel(IM_COL32_B_SHIFT), alpha);
}
constexpr ImU32 kColBatch = IM_COL32(45, 45, 52, 255);
constexpr ImU32 kColBatchSelected = IM_COL32(70, 70, 90, 255);
constexpr ImU32 kColBorder = IM_COL32(0, 0, 0, 255);
constexpr ImU32 kColSelected = IM_COL32(255, 255, 255, 255);
constexpr ImU32 kColMatch = IM_COL32(255, 90, 60, 255);
constexpr ImU32 kColWait = IM_COL32(255, 200, 64, 220);
constexpr ImU32 kColWaitDim = IM_COL32(255, 200, 64, 70);
constexpr ImU32 kColWaitHot = IM_COL32(255, 240, 160, 255);
constexpr ImU32 kColText = IM_COL32(235, 235, 235, 255);
constexpr ImU32 kColTextDim = IM_COL32(150, 150, 150, 255);
constexpr ImVec4 kMatchText{ 1.0f, 0.45f, 0.3f, 1.0f };
constexpr ImVec4 kWarnText{ 1.0f, 0.75f, 0.3f, 1.0f };

const char* KindName(Trace::WaitCause::Kind kind) {
    switch (kind) {
    case Trace::WaitCause::Kind::Resource: return "resource";
    case Trace::WaitCause::Kind::Explicit: return "explicit";
    case Trace::WaitCause::Kind::Placement: return "placement";
    case Trace::WaitCause::Kind::Unknown: break;
    }
    return "unknown";
}
const char* HazardName(Trace::WaitCause::Hazard hazard) {
    switch (hazard) {
    case Trace::WaitCause::Hazard::ReadAfterWrite: return "RAW";
    case Trace::WaitCause::Hazard::WriteAfterRead: return "WAR";
    case Trace::WaitCause::Hazard::WriteAfterWrite: return "WAW";
    case Trace::WaitCause::Hazard::None: break;
    }
    return "-";
}
const char* SourceName(ScheduleSubmissionWaits::Point::Source source) {
    switch (source) {
    case ScheduleSubmissionWaits::Point::Source::Incoming: return "incoming";
    case ScheduleSubmissionWaits::Point::Source::Uploads: return "uploads";
    case ScheduleSubmissionWaits::Point::Source::FrameValues: return "frame values";
    }
    return "?";
}

unsigned long long U64(uint64_t value) { return static_cast<unsigned long long>(value); }

bool ContainsNoCase(std::string_view haystack, std::string_view needle) {
    if (needle.empty()) {
        return true;
    }
    const auto it = std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end(), [](char a, char b) {
        return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
    });
    return it != haystack.end();
}

// Names that stay safe on a trace whose indices are out of range.
struct Names {
    const Trace& trace;
    std::string Queue(uint32_t queue) const {
        if (queue < trace.queueNames.size() && !trace.queueNames[queue].empty()) {
            return trace.queueNames[queue];
        }
        return "queue " + std::to_string(queue);
    }
    const char* Pass(uint32_t pass) const {
        if (pass < trace.passes.size()) {
            return trace.passes[pass].name.c_str();
        }
        return pass == UINT32_MAX ? "(entry)" : "(invalid pass)";
    }
    const char* Resource(uint32_t resource) const {
        if (resource < trace.resources.size()) {
            return trace.resources[resource].name.c_str();
        }
        return resource == UINT32_MAX ? "-" : "(invalid resource)";
    }
    uint32_t BatchQueue(uint32_t batch) const {
        return batch < trace.batches.size() ? trace.batches[batch].queue : 0;
    }
};

// The resources whose names contain the filter (empty: no filter).
struct Filter {
    bool active = false;
    std::vector<char> resources;
    bool Resource(uint32_t resource) const {
        return active && resource < resources.size() && resources[resource] != 0;
    }
    bool Pass(const Trace::Pass& pass) const {
        if (!active) {
            return false;
        }
        for (const auto& access : pass.accesses) {
            if (Resource(access.resource)) {
                return true;
            }
        }
        return false;
    }
};

void TextRange(const Trace::StateChange& change) {
    if (change.mipCount == 0 && change.sliceCount == 0) {
        ImGui::TextDisabled("    range: whole resource");
        return;
    }
    ImGui::TextDisabled("    range: mips %u+%u, slices %u+%u", change.firstMip, change.mipCount, change.firstSlice, change.sliceCount);
}

void DrawPassDetails(const Trace& trace, const Filter& filter, uint32_t passIndex, size_t limit) {
    const Names names{ trace };
    if (passIndex >= trace.passes.size()) {
        ImGui::TextDisabled("(no such pass)");
        return;
    }
    const auto& pass = trace.passes[passIndex];
    ImGui::TextUnformatted(pass.name.c_str());
    ImGui::Text("pass %u: batch %u, position %u, prepared pass %u", passIndex, pass.batch, pass.position, pass.preparedPassIndex);
    const std::string queue = names.Queue(pass.queue);
    const std::string preferred = names.Queue(pass.preferredQueue);
    ImGui::Text("queue: %s (preferred: %s)%s", queue.c_str(), preferred.c_str(), pass.forceBatchIsolation ? ", forced batch isolation" : "");

    ImGui::SeparatorText("Accesses");
    if (pass.accesses.empty()) {
        ImGui::TextDisabled("none");
    }
    for (size_t i = 0; i < pass.accesses.size(); ++i) {
        if (i == limit) {
            ImGui::TextDisabled("... %zu more", pass.accesses.size() - limit);
            break;
        }
        const auto& access = pass.accesses[i];
        const char* name = names.Resource(access.resource);
        if (filter.Resource(access.resource)) {
            ImGui::TextColored(kMatchText, "%s  %s", access.write ? "W" : "R", name);
        } else {
            ImGui::Text("%s  %s", access.write ? "W" : "R", name);
        }
    }

    ImGui::SeparatorText("State changes before it");
    if (pass.stateChanges.empty()) {
        ImGui::TextDisabled("none");
    }
    for (size_t i = 0; i < pass.stateChanges.size(); ++i) {
        if (i == limit) {
            ImGui::TextDisabled("... %zu more", pass.stateChanges.size() - limit);
            break;
        }
        const auto& change = pass.stateChanges[i];
        const bool texture = change.resource < trace.resources.size() && trace.resources[change.resource].hasLayout;
        const char* name = names.Resource(change.resource);
        if (filter.Resource(change.resource)) {
            ImGui::TextColored(kMatchText, "%s  (%s -> %s)", name, change.beforeWrite ? "W" : "R", change.afterWrite ? "W" : "R");
        } else {
            ImGui::Text("%s  (%s -> %s)", name, change.beforeWrite ? "W" : "R", change.afterWrite ? "W" : "R");
        }
        ImGui::Text("    access 0x%llx -> 0x%llx", U64(change.beforeAccess), U64(change.afterAccess));
        if (texture) {
            ImGui::Text("    layout 0x%llx -> 0x%llx", U64(change.beforeLayout), U64(change.afterLayout));
        }
        ImGui::Text("    sync   0x%llx -> 0x%llx", U64(change.beforeSync), U64(change.afterSync));
        TextRange(change);
        if (change.previousPass == UINT32_MAX) {
            ImGui::TextDisabled("    previous: the execution's entry state");
        } else {
            ImGui::TextDisabled("    previous: %s (batch %u)", names.Pass(change.previousPass), change.previousBatch);
        }
    }
}

void DrawWaitCauses(const Trace& trace, const Filter& filter, const Trace::Wait& wait, size_t limit) {
    const Names names{ trace };
    const std::string consumer = names.Queue(names.BatchQueue(wait.consumerBatch));
    const std::string producer = names.Queue(names.BatchQueue(wait.producerBatch));
    ImGui::Text("batch %u (%s) waits for batch %u (%s)", wait.consumerBatch, consumer.c_str(), wait.producerBatch, producer.c_str());
    ImGui::SeparatorText("Causes");
    if (wait.causes.empty()) {
        ImGui::TextDisabled("none recorded");
    }
    for (size_t i = 0; i < wait.causes.size(); ++i) {
        if (i == limit) {
            ImGui::TextDisabled("... %zu more", wait.causes.size() - limit);
            break;
        }
        const auto& cause = wait.causes[i];
        const bool match = filter.Resource(cause.resource);
        if (cause.kind == Trace::WaitCause::Kind::Resource) {
            const ImVec4 color = match ? kMatchText : ImGui::GetStyleColorVec4(ImGuiCol_Text);
            ImGui::TextColored(color, "%s %s: %s -> %s on %s", KindName(cause.kind), HazardName(cause.hazard), names.Pass(cause.fromPass),
                names.Pass(cause.toPass), names.Resource(cause.resource));
        } else {
            ImGui::Text("%s: %s -> %s", KindName(cause.kind), names.Pass(cause.fromPass), names.Pass(cause.toPass));
        }
    }
}

void DrawBatchDetails(const Trace& trace, const ScheduleSubmissionWaits* submission, uint32_t batchIndex, size_t limit) {
    const Names names{ trace };
    if (batchIndex >= trace.batches.size()) {
        ImGui::TextDisabled("(no such batch)");
        return;
    }
    const auto& batch = trace.batches[batchIndex];
    const std::string queue = names.Queue(batch.queue);
    ImGui::Text("batch %u on %s: %zu passes", batchIndex, queue.c_str(), batch.passes.size());

    ImGui::SeparatorText("Waits inside the execution");
    bool any = false;
    for (const auto& wait : trace.waits) {
        if (wait.consumerBatch == batchIndex) {
            const std::string producer = names.Queue(names.BatchQueue(wait.producerBatch));
            ImGui::Text("waits for batch %u (%s): %zu causes", wait.producerBatch, producer.c_str(), wait.causes.size());
            any = true;
        } else if (wait.producerBatch == batchIndex) {
            const std::string consumer = names.Queue(names.BatchQueue(wait.consumerBatch));
            ImGui::Text("signals batch %u (%s): %zu causes", wait.consumerBatch, consumer.c_str(), wait.causes.size());
            any = true;
        }
    }
    if (!any) {
        ImGui::TextDisabled("none");
    }

    ImGui::SeparatorText("At submission (last frame)");
    if (!submission) {
        ImGui::TextDisabled("no submission recorded for '%s'", trace.label.c_str());
    } else if (batchIndex >= submission->batches.size()) {
        ImGui::TextDisabled("frame %llu: no entry for this batch", U64(submission->frame));
    } else {
        const auto& submitted = submission->batches[batchIndex];
        ImGui::Text("frame %llu; external entry barrier: %s, exit barrier: %s", U64(submission->frame), submitted.externalEntryBarrier ? "yes" : "no",
            submitted.externalExitBarrier ? "yes" : "no");
        if (submitted.points.empty()) {
            ImGui::TextDisabled("no timeline waits");
        }
        for (size_t i = 0; i < submitted.points.size(); ++i) {
            if (i == limit) {
                ImGui::TextDisabled("... %zu more", submitted.points.size() - limit);
                break;
            }
            const auto& point = submitted.points[i];
            ImGui::Text("%s: timeline 0x%llx value %llu", SourceName(point.source), U64(point.timeline), U64(point.value));
        }
        if (submitted.queueUploadCopies) {
            ImGui::Text("uploads on this queue just ahead: %llu copies, %zu waits", U64(submitted.queueUploadCopies),
                submitted.queueUploadWaits.size());
            for (size_t i = 0; i < submitted.queueUploadWaits.size() && i < limit; ++i) {
                const auto& point = submitted.queueUploadWaits[i];
                ImGui::Text("  %s: timeline 0x%llx value %llu", SourceName(point.source), U64(point.timeline), U64(point.value));
            }
        }
    }

    ImGui::SeparatorText("Passes");
    for (size_t i = 0; i < batch.passes.size(); ++i) {
        if (i == limit) {
            ImGui::TextDisabled("... %zu more", batch.passes.size() - limit);
            break;
        }
        ImGui::Text("%zu: %s", i, names.Pass(batch.passes[i]));
    }
}

ImVec2 Bezier(ImVec2 p0, ImVec2 c0, ImVec2 c1, ImVec2 p1, float t) {
    const float u = 1.0f - t;
    const float w0 = u * u * u, w1 = 3.0f * u * u * t, w2 = 3.0f * u * t * t, w3 = t * t * t;
    return ImVec2(w0 * p0.x + w1 * c0.x + w2 * c1.x + w3 * p1.x, w0 * p0.y + w1 * c0.y + w2 * c1.y + w3 * p1.y);
}

// A wait arrow, producer (left) to consumer (right); returns its midpoint (the hover handle).
ImVec2 DrawArrow(ImDrawList* drawList, ImVec2 from, ImVec2 to, ImU32 color, float thickness) {
    if (to.x <= from.x) {
        to.x = from.x + 1.0f;
    }
    const float handle = 0.30f * std::fabs(to.y - from.y) + 12.0f;
    const ImVec2 c0(from.x + handle, from.y);
    const ImVec2 c1(to.x - handle, to.y);
    drawList->AddBezierCubic(from, c0, c1, to, color, thickness);
    // The head along the end tangent (c1 -> to points right by construction).
    float tx = to.x - c1.x, ty = to.y - c1.y;
    const float length = std::sqrt(tx * tx + ty * ty);
    if (length < 1e-4f) {
        tx = 1.0f;
        ty = 0.0f;
    } else {
        tx /= length;
        ty /= length;
    }
    const float headLength = 9.0f, headWidth = 4.5f;
    const ImVec2 b(to.x - tx * headLength - ty * headWidth, to.y - ty * headLength + tx * headWidth);
    const ImVec2 c(to.x - tx * headLength + ty * headWidth, to.y - ty * headLength - tx * headWidth);
    drawList->AddTriangleFilled(to, b, c, color);
    return Bezier(from, c0, c1, to, 0.5f);
}

bool Inside(ImVec2 point, ImVec2 lo, ImVec2 hi) {
    return point.x >= lo.x && point.x < hi.x && point.y >= lo.y && point.y < hi.y;
}

struct Hover {
    enum class Kind : uint8_t { None, Pass, Batch, Wait } kind = Kind::None;
    uint32_t index = UINT32_MAX;
};

// The lanes: one per queue slot, the batches in batch order along x, the waits as arrows. Returns what the mouse is over.
Hover DrawLanes(const Trace& trace, const Filter& filter, ScheduleInspectorState& state, float height) {
    const Names names{ trace };
    uint32_t queueCount = static_cast<uint32_t>(trace.queueNames.size());
    for (const auto& batch : trace.batches) {
        queueCount = std::max(queueCount, batch.queue + 1);
    }
    queueCount = std::max(queueCount, 1u);

    const float passWidth = std::max(state.passWidth, 24.0f);
    std::vector<float> batchX(trace.batches.size()), batchW(trace.batches.size());
    float x = kGutter;
    for (size_t b = 0; b < trace.batches.size(); ++b) {
        const float passes = static_cast<float>(std::max<size_t>(trace.batches[b].passes.size(), 1));
        batchX[b] = x;
        batchW[b] = 2.0f * kBatchPad + passes * passWidth + (passes - 1.0f) * kPassGap;
        x += batchW[b] + kBatchGap;
    }
    const float contentW = x;
    const float contentH = kTop + static_cast<float>(queueCount) * (kLaneHeight + kLaneGap);

    Hover hover;
    if (!ImGui::BeginChild("##scheduleLanes", ImVec2(0.0f, height), ImGuiChildFlags_Borders, ImGuiWindowFlags_HorizontalScrollbar)) {
        ImGui::EndChild();
        return hover;
    }
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 clipMin = drawList->GetClipRectMin();
    const ImVec2 clipMax = drawList->GetClipRectMax();
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const bool canHover = ImGui::IsWindowHovered() && Inside(mouse, clipMin, clipMax);  // not over the scrollbars

    const auto laneTop = [&](uint32_t queue) { return origin.y + kTop + static_cast<float>(queue) * (kLaneHeight + kLaneGap); };
    const auto visible = [&](float left, float right) { return right >= clipMin.x && left <= clipMax.x; };

    // Lane backgrounds and labels.
    for (uint32_t queue = 0; queue < queueCount; ++queue) {
        const float top = laneTop(queue);
        drawList->AddRectFilled(ImVec2(std::min(origin.x, clipMin.x), top), ImVec2(std::max(origin.x + contentW, clipMax.x), top + kLaneHeight),
            Scaled(QueueColor(queue), 0.18f, 255));
        const std::string label = names.Queue(queue);
        drawList->AddText(ImVec2(origin.x + 6.0f, top + kLaneHeight * 0.5f - ImGui::GetTextLineHeight() * 0.5f), kColText, label.c_str());
    }

    // Batches and their passes.
    const bool anySelection = state.selectedPass < trace.passes.size();
    for (size_t b = 0; b < trace.batches.size(); ++b) {
        const auto& batch = trace.batches[b];
        const float left = origin.x + batchX[b];
        const float right = left + batchW[b];
        if (!visible(left, right)) {
            continue;
        }
        const uint32_t batchIndex = static_cast<uint32_t>(b);
        const float top = laneTop(batch.queue);
        const ImVec2 boxMin(left, top + 2.0f), boxMax(right, top + kLaneHeight - 2.0f);
        drawList->AddRectFilled(boxMin, boxMax, state.selectedBatch == batchIndex ? kColBatchSelected : kColBatch, 4.0f);
        drawList->AddRect(boxMin, boxMax, state.selectedBatch == batchIndex ? kColSelected : kColBorder, 4.0f);
        char header[32];
        std::snprintf(header, sizeof(header), "B%zu", b);
        drawList->AddText(ImVec2(left, origin.y + 1.0f), kColTextDim, header);
        char count[32];
        std::snprintf(count, sizeof(count), "%zu passes", batch.passes.size());
        drawList->PushClipRect(boxMin, boxMax, true);
        drawList->AddText(ImVec2(left + kBatchPad, top + 2.0f), kColTextDim, count);
        drawList->PopClipRect();
        if (canHover && Inside(mouse, boxMin, boxMax)) {
            hover = Hover{ Hover::Kind::Batch, batchIndex };
        }

        for (size_t i = 0; i < batch.passes.size(); ++i) {
            const uint32_t passIndex = batch.passes[i];
            const float passLeft = left + kBatchPad + static_cast<float>(i) * (passWidth + kPassGap);
            const ImVec2 passMin(passLeft, top + 2.0f + kBatchHeader), passMax(passLeft + passWidth, top + kLaneHeight - 2.0f - kBatchPad);
            if (!visible(passMin.x, passMax.x)) {
                continue;
            }
            const bool valid = passIndex < trace.passes.size();
            const bool match = valid && filter.Pass(trace.passes[passIndex]);
            const bool selected = state.selectedPass == passIndex;
            const float shade = filter.active && !match ? 0.45f : 1.0f;
            ImU32 fill = Scaled(QueueColor(batch.queue), shade, 230);
            if (match) {
                fill = kColMatch;
            }
            drawList->AddRectFilled(passMin, passMax, fill, 3.0f);
            drawList->AddRect(passMin, passMax, selected ? kColSelected : kColBorder, 3.0f, 0, selected ? 2.0f : 1.0f);
            drawList->PushClipRect(passMin, ImVec2(passMax.x - 2.0f, passMax.y), true);
            drawList->AddText(ImVec2(passMin.x + 3.0f, passMin.y + 2.0f), filter.active && !match ? kColTextDim : kColText, names.Pass(passIndex));
            drawList->PopClipRect();
            if (canHover && Inside(mouse, passMin, passMax)) {
                hover = Hover{ Hover::Kind::Pass, passIndex };
            }
        }
    }

    // Waits: producer batch's right edge to consumer batch's left edge, lane centre to lane centre.
    for (size_t w = 0; w < trace.waits.size(); ++w) {
        const auto& wait = trace.waits[w];
        if (wait.consumerBatch >= trace.batches.size() || wait.producerBatch >= trace.batches.size()) {
            continue;
        }
        const ImVec2 from(origin.x + batchX[wait.producerBatch] + batchW[wait.producerBatch],
            laneTop(trace.batches[wait.producerBatch].queue) + kLaneHeight * 0.5f);
        const ImVec2 to(origin.x + batchX[wait.consumerBatch], laneTop(trace.batches[wait.consumerBatch].queue) + kLaneHeight * 0.5f);
        if (!visible(std::min(from.x, to.x), std::max(from.x, to.x))) {
            continue;
        }
        bool related = !anySelection && state.selectedBatch >= trace.batches.size() && !filter.active;
        for (const auto& cause : wait.causes) {
            if (cause.fromPass == state.selectedPass || cause.toPass == state.selectedPass || filter.Resource(cause.resource)) {
                related = true;
            }
        }
        if (state.selectedBatch == wait.consumerBatch || state.selectedBatch == wait.producerBatch) {
            related = true;
        }
        const ImVec2 handleCenter = DrawArrow(drawList, from, to, related ? kColWait : kColWaitDim, related ? 2.0f : 1.0f);
        const float dx = mouse.x - handleCenter.x, dy = mouse.y - handleCenter.y;
        const bool hot = canHover && dx * dx + dy * dy <= (kWaitHandle + 2.0f) * (kWaitHandle + 2.0f);
        drawList->AddCircleFilled(handleCenter, kWaitHandle, hot ? kColWaitHot : (related ? kColWait : kColWaitDim));
        if (hot) {
            hover = Hover{ Hover::Kind::Wait, static_cast<uint32_t>(w) };
        }
    }

    ImGui::Dummy(ImVec2(contentW, contentH));
    if (canHover && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        state.selectedPass = UINT32_MAX;
        state.selectedBatch = UINT32_MAX;
        if (hover.kind == Hover::Kind::Pass) {
            state.selectedPass = hover.index;
        } else if (hover.kind == Hover::Kind::Batch) {
            state.selectedBatch = hover.index;
        } else if (hover.kind == Hover::Kind::Wait) {
            state.selectedBatch = trace.waits[hover.index].consumerBatch;
        }
    }
    ImGui::EndChild();
    return hover;
}

const ScheduleSubmissionWaits* FindSubmission(const PersistentScheduleSnapshot& snapshot, const std::string& label) {
    for (const auto& submission : snapshot.lastSubmission) {
        if (submission.label == label) {
            return &submission;
        }
    }
    return nullptr;
}

void DrawCausesTable(const Trace& trace, const Filter& filter, ScheduleInspectorState& state) {
    const Names names{ trace };
    struct Row {
        uint32_t wait = 0;
        uint32_t cause = UINT32_MAX;  // UINT32_MAX: a wait without causes
    };
    std::vector<Row> rows;
    for (size_t w = 0; w < trace.waits.size(); ++w) {
        const auto& wait = trace.waits[w];
        if (wait.causes.empty() && !(filter.active && state.filterCausesTable)) {
            rows.push_back(Row{ static_cast<uint32_t>(w), UINT32_MAX });
        }
        for (size_t c = 0; c < wait.causes.size(); ++c) {
            if (filter.active && state.filterCausesTable && !filter.Resource(wait.causes[c].resource)) {
                continue;
            }
            rows.push_back(Row{ static_cast<uint32_t>(w), static_cast<uint32_t>(c) });
        }
    }
    ImGui::Text("%zu waits, %zu rows", trace.waits.size(), rows.size());
    constexpr ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_Resizable |
        ImGuiTableFlags_SizingStretchProp;
    if (!ImGui::BeginTable("##scheduleCauses", 7, flags, ImVec2(0.0f, 0.0f))) {
        return;
    }
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Consumer", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("Producer", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthStretch, 0.7f);
    ImGui::TableSetupColumn("From pass", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupColumn("To pass", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupColumn("Resource", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupColumn("Hazard", ImGuiTableColumnFlags_WidthStretch, 0.5f);
    ImGui::TableHeadersRow();
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(rows.size()));
    while (clipper.Step()) {
        for (int r = clipper.DisplayStart; r < clipper.DisplayEnd; ++r) {
            const Row& row = rows[static_cast<size_t>(r)];
            const auto& wait = trace.waits[row.wait];
            const Trace::WaitCause* cause = row.cause == UINT32_MAX ? nullptr : &wait.causes[row.cause];
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::PushID(r);
            const std::string consumer = "B" + std::to_string(wait.consumerBatch) + " " + names.Queue(names.BatchQueue(wait.consumerBatch));
            const bool selected = cause ? state.selectedPass == cause->toPass : state.selectedBatch == wait.consumerBatch;
            if (ImGui::Selectable(consumer.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns)) {
                state.selectedPass = cause ? cause->toPass : UINT32_MAX;
                state.selectedBatch = cause ? UINT32_MAX : wait.consumerBatch;
            }
            ImGui::PopID();
            ImGui::TableSetColumnIndex(1);
            const std::string producer = "B" + std::to_string(wait.producerBatch) + " " + names.Queue(names.BatchQueue(wait.producerBatch));
            ImGui::TextUnformatted(producer.c_str());
            if (!cause) {
                ImGui::TableSetColumnIndex(2);
                ImGui::TextDisabled("(no causes)");
                continue;
            }
            ImGui::TableSetColumnIndex(2);
            ImGui::TextUnformatted(KindName(cause->kind));
            ImGui::TableSetColumnIndex(3);
            ImGui::TextUnformatted(names.Pass(cause->fromPass));
            ImGui::TableSetColumnIndex(4);
            ImGui::TextUnformatted(names.Pass(cause->toPass));
            ImGui::TableSetColumnIndex(5);
            if (filter.Resource(cause->resource)) {
                ImGui::TextColored(kMatchText, "%s", names.Resource(cause->resource));
            } else {
                ImGui::TextUnformatted(names.Resource(cause->resource));
            }
            ImGui::TableSetColumnIndex(6);
            ImGui::TextUnformatted(HazardName(cause->hazard));
        }
    }
    ImGui::EndTable();
}

}  // namespace

void ShowScheduleInspector(const PersistentScheduleSnapshot& snapshot, ScheduleInspectorState& state) {
    // The selected executable, by label (indices move as epochs come and go).
    const Trace* trace = nullptr;
    for (const auto& executable : snapshot.executables) {
        if (executable && executable->label == state.selectedLabel) {
            trace = executable.get();
            break;
        }
    }
    if (!trace) {
        for (const auto& executable : snapshot.executables) {
            if (executable) {
                trace = executable.get();
                break;
            }
        }
    }
    if (!trace) {
        ImGui::TextDisabled("No persistent executable traced yet.");
        return;
    }

    const auto comboLabel = [](const Trace& t) {
        return t.label + " (rev " + std::to_string(t.structuralRevision) + ", " + std::to_string(t.batches.size()) + " batches, " +
            std::to_string(t.passes.size()) + " passes)";
    };
    ImGui::SetNextItemWidth(320.0f);
    const std::string current = comboLabel(*trace);
    if (ImGui::BeginCombo("Executable", current.c_str())) {
        for (size_t i = 0; i < snapshot.executables.size(); ++i) {
            const auto& executable = snapshot.executables[i];
            if (!executable) {
                continue;
            }
            ImGui::PushID(static_cast<int>(i));
            const std::string label = comboLabel(*executable);
            if (ImGui::Selectable(label.c_str(), executable.get() == trace)) {
                trace = executable.get();
            }
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    if (trace->label != state.selectedLabel || trace->structuralRevision != state.selectedRevision) {
        state.selectedLabel = trace->label;
        state.selectedRevision = trace->structuralRevision;
        state.selectedPass = UINT32_MAX;
        state.selectedBatch = UINT32_MAX;
    }

    ImGui::SameLine();
    if (ImGui::Button("Copy as text")) {
        ImGui::SetClipboardText(ScheduleTraceToText(*trace).c_str());
    }
    ImGui::SameLine();
    if (ImGui::Button("Copy all")) {
        std::string text;
        for (const auto& executable : snapshot.executables) {
            if (executable) {
                text += ScheduleTraceToText(*executable);
                text += '\n';
            }
        }
        ImGui::SetClipboardText(text.c_str());
    }

    ImGui::SetNextItemWidth(240.0f);
    ImGui::InputTextWithHint("##resourceFilter", "resource filter", state.resourceFilter, sizeof(state.resourceFilter));
    ImGui::SameLine();
    ImGui::Checkbox("Filter the causes table", &state.filterCausesTable);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(140.0f);
    ImGui::SliderFloat("Pass width", &state.passWidth, 40.0f, 300.0f, "%.0f px");

    Filter filter;
    const std::string_view needle(state.resourceFilter);
    if (!needle.empty()) {
        filter.active = true;
        filter.resources.resize(trace->resources.size());
        for (size_t r = 0; r < trace->resources.size(); ++r) {
            filter.resources[r] = ContainsNoCase(trace->resources[r].name, needle) ? 1 : 0;
        }
    }

    ImGui::Text("%s: structural revision %llu, %zu batches, %zu passes, %zu resources, %zu waits", trace->label.c_str(),
        U64(trace->structuralRevision), trace->batches.size(), trace->passes.size(), trace->resources.size(), trace->waits.size());
    if (!trace->stateFallbackReason.empty()) {
        ImGui::TextColored(kWarnText, "State plan incomplete: %s", trace->stateFallbackReason.c_str());
    }
    const ScheduleSubmissionWaits* submission = FindSubmission(snapshot, trace->label);

    uint32_t queueCount = static_cast<uint32_t>(trace->queueNames.size());
    for (const auto& batch : trace->batches) {
        queueCount = std::max(queueCount, batch.queue + 1);
    }
    const float wanted = kTop + static_cast<float>(std::max(queueCount, 1u)) * (kLaneHeight + kLaneGap) + ImGui::GetStyle().ScrollbarSize +
        2.0f * ImGui::GetStyle().WindowPadding.y;
    const float lanesHeight = std::clamp(wanted, 100.0f, std::max(100.0f, ImGui::GetContentRegionAvail().y * 0.6f));
    const Hover hover = DrawLanes(*trace, filter, state, lanesHeight);

    if (hover.kind != Hover::Kind::None && ImGui::BeginTooltip()) {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 40.0f);
        if (hover.kind == Hover::Kind::Pass) {
            DrawPassDetails(*trace, filter, hover.index, kTooltipLines);
        } else if (hover.kind == Hover::Kind::Batch) {
            DrawBatchDetails(*trace, submission, hover.index, kTooltipLines);
        } else {
            DrawWaitCauses(*trace, filter, trace->waits[hover.index], kTooltipLines);
        }
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }

    if (ImGui::BeginTabBar("##scheduleTabs")) {
        if (ImGui::BeginTabItem("Selection")) {
            if (ImGui::BeginChild("##scheduleSelection", ImVec2(0.0f, 0.0f))) {
                if (state.selectedPass < trace->passes.size()) {
                    DrawPassDetails(*trace, filter, state.selectedPass, SIZE_MAX);
                } else if (state.selectedBatch < trace->batches.size()) {
                    DrawBatchDetails(*trace, submission, state.selectedBatch, SIZE_MAX);
                    for (const auto& wait : trace->waits) {
                        if (wait.consumerBatch == state.selectedBatch) {
                            ImGui::Spacing();
                            DrawWaitCauses(*trace, filter, wait, SIZE_MAX);
                        }
                    }
                } else {
                    ImGui::TextDisabled("Click a pass, a batch or a wait's handle in the lanes, or a row of the causes table.");
                }
            }
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Waits and causes")) {
            DrawCausesTable(*trace, filter, state);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

}  // namespace org::debugui
