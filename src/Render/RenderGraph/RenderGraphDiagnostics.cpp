// RenderGraph's diagnostics: the VRAM usage dump.
#include "RenderGraphInternal.h"

namespace org {

using namespace graph_internal;

void RenderGraph::WriteVramUsageDebugDump(uint8_t frameIndex) const
{
	try {
		struct DumpCategoryRow {
			std::string label;
			uint64_t bytes = 0;
			size_t resourceCount = 0;
		};

		auto majorCategory = [](rhi::ResourceType type) -> const char* {
			switch (type) {
			case rhi::ResourceType::Buffer:
				return "Buffers";
			case rhi::ResourceType::Texture1D:
			case rhi::ResourceType::Texture2D:
			case rhi::ResourceType::Texture3D:
				return "Textures";
			case rhi::ResourceType::AccelerationStructure:
				return "AccelStructs";
			default:
				return "Other";
			}
		};

		auto resourceTypeLabel = [](rhi::ResourceType type) -> const char* {
			switch (type) {
			case rhi::ResourceType::Buffer:
				return "Buffer";
			case rhi::ResourceType::Texture1D:
				return "Texture1D";
			case rhi::ResourceType::Texture2D:
				return "Texture2D";
			case rhi::ResourceType::Texture3D:
				return "Texture3D";
			case rhi::ResourceType::AccelerationStructure:
				return "AccelerationStructure";
			default:
				return "Unknown";
			}
		};

		auto formatPct = [](uint64_t bytes, uint64_t totalBytes) {
			return totalBytes == 0
				? 0.0
				: (100.0 * static_cast<double>(bytes) / static_cast<double>(totalBytes));
		};

		std::vector<org::memory::ResourceMemoryRecord> memoryRecords;
		m_memorySnapshotProvider.BuildSnapshot(memoryRecords);

		std::unordered_map<std::string, DumpCategoryRow> categoriesByLabel;
		categoriesByLabel.reserve(memoryRecords.size() * 2 + 8);
		uint64_t totalBytes = 0;

		for (const auto& record : memoryRecords) {
			// Placements in an aliasing pool are in the pool's own record.
			if (!record.CountsTowardTotal()) continue;
			totalBytes += record.bytes;
			const char* usage = record.usage.empty() ? "Unspecified" : record.usage.c_str();
			std::string categoryLabel = std::string(majorCategory(record.resourceType)) + "/" + usage;
			auto& category = categoriesByLabel[categoryLabel];
			category.label = std::move(categoryLabel);
			category.bytes += record.bytes;
			category.resourceCount += 1;
		}

		std::vector<DumpCategoryRow> categories;
		categories.reserve(categoriesByLabel.size());
		for (auto& [label, row] : categoriesByLabel) {
			(void)label;
			categories.push_back(std::move(row));
		}
		std::sort(categories.begin(), categories.end(), [](const DumpCategoryRow& a, const DumpCategoryRow& b) {
			if (a.bytes != b.bytes) {
				return a.bytes > b.bytes;
			}
			return a.label < b.label;
		});

		std::vector<const org::memory::ResourceMemoryRecord*> resources;
		resources.reserve(memoryRecords.size());
		for (const auto& record : memoryRecords) {
			resources.push_back(&record);
		}
		std::sort(resources.begin(), resources.end(), [](const auto* a, const auto* b) {
			if (a->bytes != b->bytes) {
				return a->bytes > b->bytes;
			}
			if (a->resourceName != b->resourceName) {
				return a->resourceName < b->resourceName;
			}
			return a->resourceID < b->resourceID;
		});

		const AutoAliasDebugSnapshot aliasSnapshot = GetAutoAliasDebugSnapshot();

		auto modeLabel = [&](AutoAliasMode mode) -> const char* {
			switch (mode) {
			case AutoAliasMode::Off: return "Off";
			case AutoAliasMode::Conservative: return "Conservative";
			case AutoAliasMode::Balanced: return "Balanced";
			case AutoAliasMode::Aggressive: return "Aggressive";
			default: return "Unknown";
			}
		};

		auto packingStrategyLabel = [&](AutoAliasPackingStrategy strategy) -> const char* {
			switch (strategy) {
			case AutoAliasPackingStrategy::GreedySweepLine: return "Greedy Sweep-Line";
			case AutoAliasPackingStrategy::BranchAndBound: return "Beam Search (Near-Optimal)";
			default: return "Unknown";
			}
		};

		std::ostringstream dump;
		dump << "RenderGraph VRAM Usage Dump\n";
		dump << "frame_index=" << static_cast<unsigned int>(frameIndex) << "\n";
		dump << "resource_count=" << memoryRecords.size()
			 << " total_bytes=" << totalBytes
			 << " category_count=" << categories.size()
			 << " alias_pool_count=" << aliasSnapshot.poolDebug.size() << "\n\n";

		dump << "[Categories]\n";
		if (categories.empty()) {
			dump << "<none>\n";
		}
		else {
			for (const auto& category : categories) {
				dump << category.label
					 << " bytes=" << category.bytes
					 << " pct_total=" << formatPct(category.bytes, totalBytes)
					 << " resources=" << category.resourceCount
					 << "\n";
			}
		}

		dump << "\n[Resources]\n";
		if (resources.empty()) {
			dump << "<none>\n";
		}
		else {
			for (const auto* record : resources) {
				const std::string categoryLabel = std::string(majorCategory(record->resourceType)) + "/" +
					(record->usage.empty() ? "Unspecified" : record->usage);
				dump << "id=" << record->resourceID
					 << " bytes=" << record->bytes
					 << " pct_total=" << formatPct(record->bytes, totalBytes)
					 << " category=\"" << categoryLabel << "\""
					 << " type=" << resourceTypeLabel(record->resourceType);
				if (!record->resourceName.empty()) {
					dump << " name=\"" << record->resourceName << "\"";
				}
				if (!record->identifier.empty()) {
					dump << " identifier=\"" << record->identifier << "\"";
				}
				dump << "\n";
			}
		}

		dump << "\n[Aliasing]\n";
		dump << "mode=" << modeLabel(aliasSnapshot.mode)
			 << " packing_strategy=" << packingStrategyLabel(aliasSnapshot.packingStrategy)
			 << " candidates_seen=" << aliasSnapshot.candidatesSeen
			 << " manual=" << aliasSnapshot.manuallyAssigned
			 << " auto=" << aliasSnapshot.autoAssigned
			 << " excluded=" << aliasSnapshot.excluded
			 << " candidate_bytes=" << aliasSnapshot.candidateBytes
			 << " auto_assigned_bytes=" << aliasSnapshot.autoAssignedBytes
			 << " pooled_independent_bytes=" << aliasSnapshot.pooledIndependentBytes
			 << " pooled_actual_bytes=" << aliasSnapshot.pooledActualBytes
			 << " pooled_saved_bytes=" << aliasSnapshot.pooledSavedBytes
			 << " plan_cache_hits=" << aliasSnapshot.planCacheHits
			 << " plan_cache_misses=" << aliasSnapshot.planCacheMisses;
		if (!aliasSnapshot.primaryPlanCacheMissReason.empty()) {
			dump << " primary_plan_cache_miss_reason=\"" << aliasSnapshot.primaryPlanCacheMissReason << "\"";
		}
		dump
			 << "\n";

		if (!aliasSnapshot.exclusionReasons.empty()) {
			dump << "  exclusion_reasons:\n";
			for (const auto& reason : aliasSnapshot.exclusionReasons) {
				dump << "    - reason=\"" << reason.reason << "\" count=" << reason.count << "\n";
			}
		}

		dump << "\n[AliasPools]\n";
		if (aliasSnapshot.poolDebug.empty()) {
			dump << "<none>\n";
		}
		else {
			for (const auto& pool : aliasSnapshot.poolDebug) {
				dump << "pool=" << pool.poolID
					 << " required_bytes=" << pool.requiredBytes
					 << " reserved_bytes=" << pool.reservedBytes
					 << " resource_count=" << pool.ranges.size()
					 << "\n";

				std::vector<const AutoAliasPoolRangeDebug*> ranges;
				ranges.reserve(pool.ranges.size());
				for (const auto& range : pool.ranges) {
					ranges.push_back(&range);
				}
				std::sort(ranges.begin(), ranges.end(), [](const auto* a, const auto* b) {
					if (a->startByte != b->startByte) {
						return a->startByte < b->startByte;
					}
					return a->resourceID < b->resourceID;
				});

				for (const auto* range : ranges) {
					dump << "  - id=" << range->resourceID
						 << " name=\"" << range->resourceName << "\""
						 << " bytes=[" << range->startByte << ", " << range->endByte << ")"
						 << " size=" << range->sizeBytes
						 << " firstUse=" << range->firstUse
						 << " lastUse=" << range->lastUse
						 << " overlaps_byte_range=" << (range->overlapsByteRange ? "true" : "false")
						 << "\n";
				}
			}
		}

		namespace fs = std::filesystem;
		std::error_code fsError;
		fs::path dumpDir = fs::current_path(fsError);
		if (fsError) {
			dumpDir.clear();
		}
		dumpDir /= "rendergraph_dumps";
		fs::create_directories(dumpDir, fsError);

		const fs::path dumpPath = dumpDir / "rendergraph_vram_usage_latest.txt";
		std::ofstream outFile(dumpPath, std::ios::out | std::ios::trunc);
		if (!outFile.is_open()) {
			spdlog::warn("Failed to open render graph VRAM usage dump '{}'", dumpPath.string());
			return;
		}
		outFile << dump.str();
		outFile.close();

		static bool announcedDumpPath = false;
		if (!announcedDumpPath) {
			announcedDumpPath = true;
			spdlog::info("Render graph VRAM usage dump will be written to '{}'", dumpPath.string());
		}
	}
	catch (const std::exception& ex) {
		spdlog::warn("Failed to write render graph VRAM usage dump: {}", ex.what());
	}
}

} // namespace org
