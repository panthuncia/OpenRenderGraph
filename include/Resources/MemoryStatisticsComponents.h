#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "rhi.h"

namespace MemoryStatisticsComponents
{
	struct MemSizeBytes {
		uint64_t size;
	};

	struct ResourceType {
		rhi::ResourceType type;
	};

	struct ResourceID {
		uint64_t id;
	};

	struct ResourceName {
		std::string name;
	};

	struct AliasingPool {
		std::optional<uint64_t> poolID;
	};

	struct ResourceUsage {
		std::string usage;
	};

	struct TextureShape {
		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t mipLevels = 0;
		uint32_t arraySize = 0;
		rhi::Format format = rhi::Format::Unknown;
		bool aliased = false;
	};
}
