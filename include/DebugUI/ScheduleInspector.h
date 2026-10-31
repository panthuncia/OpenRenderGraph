#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "Render/RenderGraph/ScheduleTrace.h"

namespace org::debugui {

// What the schedule inspector keeps between frames. One per window.
struct ScheduleInspectorState {
    std::string selectedLabel;                // label of the selected executable; empty selects the first one
    uint64_t selectedRevision = 0;            // selected executable's revision; a change resets the pass and batch selection
    uint32_t selectedPass = UINT32_MAX;       // index into the selected executable's passes
    uint32_t selectedBatch = UINT32_MAX;      // index into the selected executable's batches
    char resourceFilter[128] = {};            // highlights the passes that use a resource whose name contains it (any case)
    bool filterCausesTable = true;            // the causes table lists only the causes on a matching resource
    float passWidth = 110.0f;                 // a pass box's width in the lanes, pixels
};

// Called when a capture is requested for a resource, as a pass left it. It receives the selected pass's name and the
// resource's scheduling identity and name (see ScheduleTrace::Resource and RenderGraph::GetScheduledResource).
// The host requests the readback through IReadbackService and shows it with MemoryViewWidget.
// The inspector itself needs no service.
using ScheduleCaptureFn = std::function<void(const std::string& pass, uint64_t resourceId, const std::string& resourceName)>;

// Draws the persistent schedule (ScheduleTrace.h) into the current ImGui window. It shows:
// - a selector of executables: the whole frame and each epoch,
// - the batches per queue lane, with their passes and the waits between them (hover for details),
// - the selected pass,
// - a table of every wait and its causes.
// Needs only Dear ImGui (OPENRENDERGRAPH_ENABLE_SCHEDULE_UI). When `capture` is given, each resource of the selected
// pass offers a capture of its contents right after the pass.
void ShowScheduleInspector(const PersistentScheduleSnapshot& snapshot, ScheduleInspectorState& state, const ScheduleCaptureFn& capture = {});

}  // namespace org::debugui
