#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <chrono>
#include <string_view>

namespace org::runtime {

enum class TaskPriority : uint8_t { FrameCritical, Streaming, Background };

class ITaskScope {
public:
    virtual ~ITaskScope() = default;
    virtual void Cancel() noexcept = 0;
    virtual void Wait() = 0;
    virtual void CancelAndWait() = 0;
};

class ITaskService {
public:
    virtual ~ITaskService() = default;

    virtual void ParallelFor(std::string_view taskName, size_t itemCount, std::function<void(size_t)> func) = 0;

    virtual std::shared_ptr<ITaskScope> CreateScope(std::string_view name) = 0;
    virtual bool Submit(
        const std::shared_ptr<ITaskScope>& scope,
        TaskPriority priority,
        std::string_view taskName,
        std::function<void()>&& func) = 0;
    virtual bool ScheduleAfter(
        const std::shared_ptr<ITaskScope>& scope,
        std::chrono::steady_clock::duration delay,
        TaskPriority priority,
        std::string_view taskName,
        std::function<void()>&& func) = 0;

    // Optional telemetry hook — default is a no-op.
    virtual void ReportTaskTelemetry(std::string_view /*name*/, uint64_t /*durationMicros*/) {}
};

}
