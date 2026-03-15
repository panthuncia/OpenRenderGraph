#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>

namespace rg::runtime {

class ITaskService {
public:
    virtual ~ITaskService() = default;

    virtual void ParallelFor(std::string_view taskName, size_t itemCount, std::function<void(size_t)> func) = 0;

    // Optional telemetry hook — default is a no-op.
    virtual void ReportTaskTelemetry(std::string_view /*name*/, uint64_t /*durationMicros*/) {}
};

}
