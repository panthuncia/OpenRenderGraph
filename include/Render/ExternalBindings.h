#pragma once

#include <cstdint>
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

} // namespace org
