#pragma once

#include <memory>
#include <mutex>

#include "Render/Runtime/ITaskService.h"

namespace org::runtime {

inline std::mutex& DefaultTaskServiceMutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::shared_ptr<ITaskService>& DefaultTaskServiceSlot() {
    static std::shared_ptr<ITaskService> service;
    return service;
}

inline void SetDefaultTaskService(std::shared_ptr<ITaskService> service) {
    std::lock_guard lock(DefaultTaskServiceMutex());
    DefaultTaskServiceSlot() = std::move(service);
}

inline std::shared_ptr<ITaskService> GetDefaultTaskService() {
    std::lock_guard lock(DefaultTaskServiceMutex());
    return DefaultTaskServiceSlot();
}

} // namespace org::runtime
