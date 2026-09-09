#pragma once
#include "Render/Runtime/ITaskService.h"
#include <BasicTelemetry/Tracy.h>
#include <future>
#include <thread>

namespace org::experimental {
enum class FrameWorkerStage { Planning, Recording };
// Waiting uses a future rather than a helping scheduler wait: the host thread
// must not execute a recording callback while waiting for worker readiness.
template<FrameWorkerStage stage = FrameWorkerStage::Recording, class Function>
auto RunFrameWorker(const std::shared_ptr<runtime::ITaskService>& tasks,
    std::shared_ptr<runtime::ITaskScope>& scope, bool asynchronous, Function&& function) {
    if (!asynchronous) return function();
    if (!tasks) throw std::runtime_error("Async frame execution requires a task service");
    if (!scope) scope = tasks->CreateScope("ORG.Frame.Worker");
    if (!scope) throw std::runtime_error("Frame worker scope rejected");
    using Result = decltype(function());
    const auto owner = std::this_thread::get_id();
    auto work = std::make_shared<std::packaged_task<Result()>>(
        [owner, function = std::forward<Function>(function)]() mutable {
            if (std::this_thread::get_id() == owner)
                throw std::logic_error("Async frame work ran on the submission thread");
            return function();
        });
    auto ready = work->get_future();
    if (!tasks->Submit(scope, runtime::TaskPriority::FrameCritical, "ORG.Frame.Worker", [work] { (*work)(); }))
        throw std::runtime_error("Frame worker task rejected");
    work.reset(); // A dropped/cancelled task must break the promise, not hang its waiter.
    if constexpr (stage == FrameWorkerStage::Planning) {
        BT_ZONE_SCOPE("ORG.Frame.WaitForPlanning");
        return ready.get();
    } else {
        BT_ZONE_SCOPE("ORG.Frame.WaitForRecording");
        return ready.get();
    }
}
}
