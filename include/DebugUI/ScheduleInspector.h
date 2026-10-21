#pragma once

#include <cstdint>
#include <string>

#include "Render/RenderGraph/ScheduleTrace.h"

namespace org::debugui {

// What the schedule inspector keeps between frames. One per window.
struct ScheduleInspectorState {
    std::string selectedLabel;                // the selected executable's label; empty: the first one
    uint64_t selectedRevision = 0;            // the selection's executable revision (a new one drops the pass/batch selection)
    uint32_t selectedPass = UINT32_MAX;       // into the selected executable's passes
    uint32_t selectedBatch = UINT32_MAX;      // into its batches
    char resourceFilter[128] = {};            // highlights the passes that use a resource whose name contains it (any case)
    bool filterCausesTable = true;            // the causes table lists only the causes on a matching resource
    float passWidth = 110.0f;                 // a pass box's width in the lanes, pixels
};

// Draws the persistent schedule (ScheduleTrace.h) into the current ImGui window: a selector of executables (the whole frame and
// each epoch), the batches per queue lane with their passes and the waits between them (hover for the details), the selected
// pass, and a table of every wait and its causes. Needs only Dear ImGui (OPENRENDERGRAPH_ENABLE_SCHEDULE_UI).
void ShowScheduleInspector(const PersistentScheduleSnapshot& snapshot, ScheduleInspectorState& state);

}  // namespace org::debugui
