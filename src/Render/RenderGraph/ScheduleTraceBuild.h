#pragma once

#include "Render/RenderGraph/ScheduleTrace.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace org {

namespace persistent { struct ExecutableGeneration; }

// What a trace names its passes, resources and queues by. The compiler keeps no names: the persistent execution supplies them.
struct ScheduleTraceNames {
    std::function<std::string(uint32_t preparedPassIndex)> pass;  // a logical pass slot's name; empty: unnamed
    std::function<std::string(uint32_t slotIndex)> resource;      // a resource slot's name; empty: unnamed
    std::vector<std::string> queues;                              // by queue slot
};

// The executable's schedule, explained (ScheduleTrace). Wait causes are reconstructed as the compiler derives its scheduling edges
// (ExperimentalGraphCompiler.cpp, CompileWorkspace::Compile): it keeps no provenance.
std::shared_ptr<const ScheduleTrace> BuildScheduleTrace(const persistent::ExecutableGeneration& executable, const ScheduleTraceNames& names,
    std::string label, uint32_t epoch);

}  // namespace org
