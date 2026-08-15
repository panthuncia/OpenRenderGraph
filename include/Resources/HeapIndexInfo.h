#pragma once
#include <rhi.h>


namespace org {

struct ShaderVisibleIndexInfo {
	rhi::DescriptorSlot slot;
};

struct NonShaderVisibleIndexInfo {
	rhi::DescriptorSlot slot;
};


} // namespace org

using org::NonShaderVisibleIndexInfo;
using org::ShaderVisibleIndexInfo;
