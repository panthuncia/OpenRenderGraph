#pragma once

#include "Render/Runtime/DescriptorServiceAccess.h"
#include "Render/Runtime/UploadServiceAccess.h"

namespace org::runtime {

// Compatibility bridge for code that still reaches graph services through
// the thread-local accessors.  Async stage owners install the exact services
// belonging to their graph and restore the worker's prior state on exit.
class ScopedActiveGraphServices final {
public:
    ScopedActiveGraphServices(IUploadService* uploads,
        IDescriptorService* descriptors) noexcept
        : m_previousUploads(GetActiveUploadService()),
          m_previousDescriptors(GetActiveDescriptorService()) {
        SetActiveUploadService(uploads);
        SetActiveDescriptorService(descriptors);
    }

    ~ScopedActiveGraphServices() {
        SetActiveDescriptorService(m_previousDescriptors);
        SetActiveUploadService(m_previousUploads);
    }

    ScopedActiveGraphServices(const ScopedActiveGraphServices&) = delete;
    ScopedActiveGraphServices& operator=(const ScopedActiveGraphServices&) = delete;

private:
    IUploadService* m_previousUploads = nullptr;
    IDescriptorService* m_previousDescriptors = nullptr;
};

} // namespace org::runtime
