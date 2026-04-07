#pragma once

#include <cstdint>

namespace rg::shaderapi {
	inline constexpr uint32_t kResourceDescriptorIndicesRootParameter = 5;
	inline constexpr uint32_t kNumResourceDescriptorIndicesRootConstants = 31;

	inline constexpr uint32_t kIndirectCommandSignatureRootParameter = 6;
	inline constexpr uint32_t kNumIndirectCommandSignatureRootConstants = 4;
}
