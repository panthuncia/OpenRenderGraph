#pragma once

#include <cstdint>

namespace org::shaderapi {
	inline constexpr uint32_t kResourceDescriptorIndicesRootParameter = 5;
	inline constexpr uint32_t kNumResourceDescriptorIndicesRootConstants = 64;

	inline constexpr uint32_t kIndirectCommandSignatureRootParameter = 6;
	inline constexpr uint32_t kNumIndirectCommandSignatureRootConstants = 5;
}
