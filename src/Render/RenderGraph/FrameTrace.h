#pragma once
#include "Render/RenderGraph/FrameContext.h"
#include <BasicTelemetry/Telemetry.h>
#include <cstdio>

namespace org {
// Trace events already carry their thread ID and start/end timestamps. Attach
// logical frame identity explicitly; the thread's host frame may be different.
inline void AnnotateFrameTrace(const std::shared_ptr<FrameContext>& frame) {
    if (!frame) return;
    char identity[160];
    const int length = std::snprintf(identity, sizeof(identity),
        "{\"logical_frame\":%llu,\"generation\":%llu,\"slot\":%u}",
        static_cast<unsigned long long>(frame->Number()),
        static_cast<unsigned long long>(frame->Generation()), frame->Slot());
    basic_telemetry::SetCurrentScopeValue(frame->Number());
    if (length > 0 && static_cast<size_t>(length) < sizeof(identity))
        basic_telemetry::AnnotateCurrentScope(std::string_view(identity, length));
}
}
