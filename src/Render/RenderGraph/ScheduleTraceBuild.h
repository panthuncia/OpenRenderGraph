#pragma once

#include "Render/RenderGraph/ScheduleTrace.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace org {

namespace persistent { struct ExecutableGeneration; }

// The names a trace uses for passes, resources and queues. The compiler keeps no names, so the persistent execution
// supplies them.
struct ScheduleTraceNames {
    std::function<std::string(uint32_t preparedPassIndex)> pass;  // a logical pass slot's name, empty when unnamed
    std::function<std::string(uint32_t slotIndex)> resource;      // a resource slot's name, empty when unnamed
    std::vector<std::string> queues;                              // by queue slot
};

// Builds the executable's schedule with explanations (ScheduleTrace). The compiler does not record why it adds a
// wait. So the wait causes are reconstructed the same way the compiler derives its scheduling edges
// (CompileWorkspace::Compile in GraphCompiler.cpp).
std::shared_ptr<const ScheduleTrace> BuildScheduleTrace(const persistent::ExecutableGeneration& executable, const ScheduleTraceNames& names,
    std::string label, uint32_t epoch);

}  // namespace org
