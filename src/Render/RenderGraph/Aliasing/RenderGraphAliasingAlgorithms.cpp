#include "Render/RenderGraph/RenderGraph.h"
#include "../RenderGraphCompilerState.h"
#include "Render/FrameGraphSnapshot.h"


#include <algorithm>
#include <boost/functional/hash.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <BasicTelemetry/Tracy.h>
#include <tuple>

#include <rhi_helpers.h>

#include "Managers/Singletons/DeviceManager.h"
#include "Managers/Singletons/DeletionManager.h"
#include "Resources/GPUBacking/GPUTextureBacking.h"
#include "Resources/PixelBuffer.h"
#include "Resources/Buffers/DynamicBufferBase.h"
#include "Resources/MemoryStatisticsComponents.h"


namespace org {

RenderGraph::AutoAliasDebugSnapshot RenderGraph::GetAutoAliasDebugSnapshot() const {
	// Reports the latest persistent alias plan, made by PlanPersistentAliasPlacement.
	AutoAliasDebugSnapshot out{};
	out.mode = autoAliasModeLastFrame;
	out.packingStrategy = autoAliasPackingStrategyLastFrame;
	const auto& stats = autoAliasPlannerStats;
	out.candidatesSeen = stats.candidatesSeen;
	out.manuallyAssigned = stats.manuallyAssigned;
	out.autoAssigned = stats.autoAssigned;
	out.excluded = stats.excluded;
	out.candidateBytes = stats.candidateBytes;
	out.autoAssignedBytes = stats.autoAssignedBytes;
	out.pooledIndependentBytes = stats.pooledIndependentBytes;
	out.pooledActualBytes = stats.pooledActualBytes;
	out.pooledSavedBytes = stats.pooledSavedBytes;
	out.planCacheHits = stats.planCacheHits;
	out.planCacheMisses = stats.planCacheMisses;
	out.primaryPlanCacheMissReason = stats.primaryPlanCacheMissReason ? stats.primaryPlanCacheMissReason : std::string{};
	out.exclusionReasons = autoAliasExclusionReasonSummary;
	out.excludedResources = autoAliasExcludedResources;
	out.poolDebug = autoAliasPoolDebug;
	return out;
}

namespace {
	constexpr size_t kInvalidPassIndex = std::numeric_limits<size_t>::max();
	void MarkDebugOverlappingRanges(std::vector<RenderGraph::AutoAliasPoolRangeDebug>& ranges) {
		for (size_t i = 0; i < ranges.size(); ++i) {
			for (size_t j = i + 1; j < ranges.size(); ++j) {
				auto& a = ranges[i];
				auto& b = ranges[j];
				const uint64_t overlapStart = std::max(a.startByte, b.startByte);
				const uint64_t overlapEnd = std::min(a.endByte, b.endByte);
				if (overlapStart < overlapEnd) {
					a.overlapsByteRange = true;
					b.overlapsByteRange = true;
				}
			}
		}
	}

	struct FrameGraphMemoryInfo {
		uint64_t bytes = 0;
		std::string category;
	};

	struct BatchPassSpan {
		size_t firstPassIndex = kInvalidPassIndex;
		size_t lastPassIndex = kInvalidPassIndex;

		bool IsValid() const {
			return firstPassIndex != kInvalidPassIndex &&
				lastPassIndex != kInvalidPassIndex &&
				firstPassIndex <= lastPassIndex;
		}
	};

	struct LiveIntervalRecord {
		uint64_t resourceID = 0;
		uint64_t bytes = 0;
		size_t firstPassIndex = kInvalidPassIndex;
		size_t lastPassIndex = kInvalidPassIndex;
		uint64_t poolID = 0;
		uint64_t startByte = 0;
		uint64_t endByte = 0;
		bool pooled = false;
	};

	const char* FrameGraphMajorCategory(rhi::ResourceType type) {
		using RT = rhi::ResourceType;
		switch (type) {
		case RT::Buffer:                return "Buffers";
		case RT::Texture1D:             return "Textures";
		case RT::Texture2D:             return "Textures";
		case RT::Texture3D:             return "Textures";
		case RT::AccelerationStructure: return "AccelStructs";
		default:                        return "Other";
		}
	}

	std::unordered_map<uint64_t, FrameGraphMemoryInfo> BuildFrameGraphMemoryIndex(
		const std::vector<org::memory::ResourceMemoryRecord>& memoryRecords) {
		std::unordered_map<uint64_t, FrameGraphMemoryInfo> out;
		out.reserve(memoryRecords.size());

		for (const auto& record : memoryRecords) {
			if (record.resourceID == 0) {
				continue;
			}

			const char* major = FrameGraphMajorCategory(record.resourceType);
			const char* usage = !record.usage.empty() ? record.usage.c_str() : "Unspecified";

			out[record.resourceID] = FrameGraphMemoryInfo{
				.bytes = record.bytes,
				.category = std::string(major) + "/" + usage,
			};
		}

		return out;
	}

	uint64_t ComputeUnionBytes(std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
		if (ranges.empty()) {
			return 0;
		}

		std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
			if (a.first != b.first) {
				return a.first < b.first;
			}
			return a.second < b.second;
		});

		uint64_t total = 0;
		uint64_t currentStart = ranges.front().first;
		uint64_t currentEnd = ranges.front().second;

		for (size_t i = 1; i < ranges.size(); ++i) {
			const auto& range = ranges[i];
			if (range.second <= range.first) {
				continue;
			}

			if (range.first > currentEnd) {
				total += currentEnd - currentStart;
				currentStart = range.first;
				currentEnd = range.second;
				continue;
			}

			currentEnd = (std::max)(currentEnd, range.second);
		}

		total += currentEnd - currentStart;
		return total;
	}
}

void RenderGraph::BuildMemoryIntrospectionFrameGraphSnapshot(
	ui::FrameGraphSnapshot& out,
	const std::vector<org::memory::ResourceMemoryRecord>& memoryRecords) const {
	out.batches.clear();
	if (!m_compilerState || !m_compilerState->persistent) return;
	const auto memIndex = BuildFrameGraphMemoryIndex(memoryRecords);
	std::unordered_map<uint64_t, std::pair<uint64_t, std::string>> memory;
	memory.reserve(memIndex.size());
	for (const auto& [id, info] : memIndex) memory.emplace(id, std::pair{info.bytes, info.category});
	BuildPersistentFrameGraphSnapshot(out, memory);
}

namespace {
	rhi::ResourceDesc BuildAliasBufferResourceDesc(uint64_t sizeBytes, bool unorderedAccess, rhi::HeapType heapType) {
		auto desc = rhi::helpers::ResourceDesc::Buffer(sizeBytes);
		desc.heapType = heapType;
		if (unorderedAccess) {
			desc.resourceFlags |= rhi::ResourceFlags::RF_AllowUnorderedAccess;
		}
		return desc;
	}
}


rhi::ResourceDesc org::alias::AliasTextureResourceDesc(const TextureDescription& desc) {
	// The creation description, so a placement covers the whole allocation (every mip, slice and depth).
	return TextureResourceDesc(desc);
}
rhi::ResourceDesc org::alias::AliasBufferResourceDesc(uint64_t sizeBytes, bool unorderedAccess, rhi::HeapType heapType) {
	return BuildAliasBufferResourceDesc(sizeBytes, unorderedAccess, heapType);
}

} // namespace org
