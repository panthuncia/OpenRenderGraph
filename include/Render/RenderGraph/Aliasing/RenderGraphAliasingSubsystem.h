#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "Render/ResourceRegistry.h"
#include "Resources/TrackedAllocation.h"
#include "Resources/AliasingPlacement.h"

namespace org {
enum class AutoAliasMode : uint8_t {
	Off = 0,
	Conservative = 1,
	Balanced = 2,
	Aggressive = 3
};

enum class AutoAliasPackingStrategy : uint8_t {
	GreedySweepLine = 0,
	BranchAndBound = 1,
};
class RenderGraph;
struct TextureDescription;
}

namespace org::alias {

struct AutoAliasReasonCount {
	std::string reason;
	size_t count = 0;
};

struct AutoAliasExcludedResourceDebug {
	uint64_t resourceID = 0;
	std::string resourceName;
	uint64_t sizeBytes = 0;
	std::string reason;
};

struct AutoAliasPoolRangeDebug {
	uint64_t resourceID = 0;
	std::string resourceName;
	uint64_t startByte = 0;
	uint64_t endByte = 0;
	uint64_t sizeBytes = 0;
	size_t firstUse = 0;
	size_t lastUse = 0;
	bool overlapsByteRange = false;
};

struct AutoAliasPoolDebug {
	uint64_t poolID = 0;
	uint64_t requiredBytes = 0;
	uint64_t reservedBytes = 0;
	std::vector<AutoAliasPoolRangeDebug> ranges;
};

struct AutoAliasPlannerStats {
	size_t candidatesSeen = 0;
	size_t manuallyAssigned = 0;
	size_t autoAssigned = 0;
	size_t excluded = 0;
	uint64_t candidateBytes = 0;
	uint64_t autoAssignedBytes = 0;
	uint64_t pooledIndependentBytes = 0;
	uint64_t pooledActualBytes = 0;
	uint64_t pooledSavedBytes = 0;
	size_t planCacheHits = 0;
	size_t planCacheMisses = 0;
	const char* primaryPlanCacheMissReason = nullptr;
};

struct AutoAliasDebugSnapshot {
	AutoAliasMode mode{};
	AutoAliasPackingStrategy packingStrategy{};
	size_t candidatesSeen = 0;
	size_t manuallyAssigned = 0;
	size_t autoAssigned = 0;
	size_t excluded = 0;
	uint64_t candidateBytes = 0;
	uint64_t autoAssignedBytes = 0;
	uint64_t pooledIndependentBytes = 0;
	uint64_t pooledActualBytes = 0;
	uint64_t pooledSavedBytes = 0;
	size_t planCacheHits = 0;
	size_t planCacheMisses = 0;
	std::string primaryPlanCacheMissReason;
	std::vector<AutoAliasReasonCount> exclusionReasons;
	std::vector<AutoAliasExcludedResourceDebug> excludedResources;
	std::vector<AutoAliasPoolDebug> poolDebug;
};

// Resource descriptions of alias-placed textures and buffers (the persistent alias planner sizes candidates with them).
rhi::ResourceDesc AliasTextureResourceDesc(const TextureDescription& desc);
rhi::ResourceDesc AliasBufferResourceDesc(uint64_t sizeBytes, bool unorderedAccess, rhi::HeapType heapType);

} // namespace org::alias
