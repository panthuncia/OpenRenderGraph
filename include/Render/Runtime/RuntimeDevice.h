#pragma once

#include <rhi.h>

namespace org::runtime {

// Initializes the process-wide ORG runtime services against a host-owned RHI
// device. The host retains ownership of the device and must keep it alive until
// ShutdownRuntimeDevice returns.
void InitializeRuntimeDevice(rhi::Device device);
void ShutdownRuntimeDevice() noexcept;

} // namespace org::runtime
