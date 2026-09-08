#pragma once

#include <cstdint>
#include <memory>
#include <rhi.h>

namespace org {

enum class ExternalBindingKey : uint32_t {
    SwapchainColor = 1,
    SwapchainDepth = 2,
};

struct ExternalDescriptorBindingValue {
    ExternalBindingKey key{};
    rhi::DescriptorSlot descriptor{};
};

// Concrete admission value for a resource declared through an external key.
// The owner is retained by the executable frame through recording and GPU
// retirement; compiler inputs contain only the key and fixed resource shape.
struct ExternalResourceBindingValue {
    ExternalBindingKey key{};
    rhi::Resource resource{};
    std::shared_ptr<const void> owner;
};

} // namespace org
