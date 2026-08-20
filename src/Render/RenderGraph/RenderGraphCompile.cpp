#include "Render/RenderGraph/RenderGraph.h"
#include "RenderGraphCompilerState.h"

#include <span>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <BasicTelemetry/Tracy.h>

#include "Interfaces/IDynamicDeclaredResources.h"
#include "Resources/DynamicResource.h"
#include "Resources/BackedResource.h"
#include "Resources/ExternalTextureResource.h"


namespace org {

namespace {
	constexpr uint64_t kFrameDAGResourceIndexEmptyKey = std::numeric_limits<uint64_t>::max();

	uint64_t MixFrameDAGResourceID(uint64_t value) noexcept {
		value ^= value >> 33;
		value *= 0xff51afd7ed558ccdull;
		value ^= value >> 33;
		value *= 0xc4ceb9fe1a85ec53ull;
		value ^= value >> 33;
		return value;
	}

	constexpr size_t QueueIndex(QueueKind queue) noexcept {
		return static_cast<size_t>(queue);
	}

	Resource* UnwrapDynamicResource(Resource* resource) noexcept {
		auto* current = resource;
		while (auto* dynamicResource = dynamic_cast<DynamicResource*>(current)) {
			auto backing = dynamicResource->GetResource();
			current = backing.get();
			if (!current) {
				break;
			}
		}
		return current;
	}

	BackedResource* TryGetBackedResource(Resource* resource) noexcept {
		return dynamic_cast<BackedResource*>(UnwrapDynamicResource(resource));
	}

	bool HasLiveCompileResourceBacking(Resource* resource) {
		if (!resource) {
			return false;
		}

		if (auto* backedResource = TryGetBackedResource(resource)) {
			return backedResource->IsMaterialized();
		}
		return true;
	}

	uint64_t HashCombine64(uint64_t seed, uint64_t value) noexcept {
		seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
		return seed;
	}

	uint64_t HashString64(std::string_view value) noexcept {
		uint64_t hash = 0xcbf29ce484222325ull;
		for (char c : value) {
			hash ^= static_cast<uint8_t>(c);
			hash *= 0x100000001b3ull;
		}
		return hash;
	}

	uint64_t HashResolverSnapshots(std::span<const ResolverSnapshot> resolverSnapshots) noexcept {
		uint64_t hash = resolverSnapshots.size();
		for (const auto& snapshot : resolverSnapshots) {
			hash = HashCombine64(hash, snapshot.version);
		}
		return hash;
	}

	struct CachedHandleValidationInfo {
		bool containsEphemeralOrAnonymousHandles = false;
		bool requiresStaleHandleValidation = false;
		std::vector<RenderGraph::RetainedDeclarationCache::AnonymousSlotValidationEntry> staticRequirementAnonymousEntries;
		std::vector<RenderGraph::RetainedDeclarationCache::AnonymousSlotValidationEntry> internalTransitionAnonymousEntries;
	};

	template<class PassResourceData>
	CachedHandleValidationInfo AnalyzeCachedHandleValidation(
		const ResourceRegistry& registry,
		const PassResourceData& resources)
	{
		CachedHandleValidationInfo info{};
		auto inspectHandle = [&](const ResourceRegistry::RegistryHandle& handle, uint32_t index, bool internalTransition) {
			if (handle.IsEphemeral()) {
				info.containsEphemeralOrAnonymousHandles = true;
				return;
			}
			if (registry.IsAnonymous(handle)) {
				info.containsEphemeralOrAnonymousHandles = true;
				info.requiresStaleHandleValidation = true;
				RenderGraph::RetainedDeclarationCache::AnonymousSlotValidationEntry entry{
					handle.GetKey().idx,
					index
				};
				if (internalTransition) {
					info.internalTransitionAnonymousEntries.push_back(entry);
				} else {
					info.staticRequirementAnonymousEntries.push_back(entry);
				}
			}
		};

		for (uint32_t i = 0; i < resources.staticResourceRequirements.size(); ++i) {
			inspectHandle(resources.staticResourceRequirements[i].resourceHandleAndRange.resource, i, false);
		}
		for (uint32_t i = 0; i < resources.internalTransitions.size(); ++i) {
			inspectHandle(resources.internalTransitions[i].first.resource, i, true);
		}

		auto dedupeBySlot = [](auto& entries) {
			std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
				if (lhs.anonymousSlot != rhs.anonymousSlot) {
					return lhs.anonymousSlot < rhs.anonymousSlot;
				}
				return lhs.handleIndex < rhs.handleIndex;
			});
			entries.erase(
				std::unique(entries.begin(), entries.end(),
					[](const auto& lhs, const auto& rhs) {
						return lhs.anonymousSlot == rhs.anonymousSlot && lhs.handleIndex == rhs.handleIndex;
					}),
				entries.end());
		};
		dedupeBySlot(info.staticRequirementAnonymousEntries);
		dedupeBySlot(info.internalTransitionAnonymousEntries);
		return info;
	}

	uint64_t HashBoundForDeclaration(uint64_t seed, const Bound& bound) noexcept {
		seed = HashCombine64(seed, static_cast<uint64_t>(bound.type));
		seed = HashCombine64(seed, bound.value);
		return seed;
	}

	uint64_t HashRangeForDeclaration(uint64_t seed, const RangeSpec& range) noexcept {
		seed = HashBoundForDeclaration(seed, range.mipLower);
		seed = HashBoundForDeclaration(seed, range.mipUpper);
		seed = HashBoundForDeclaration(seed, range.sliceLower);
		seed = HashBoundForDeclaration(seed, range.sliceUpper);
		return seed;
	}

	uint64_t HashStateForDeclaration(uint64_t seed, const ResourceState& state) noexcept {
		seed = HashCombine64(seed, static_cast<uint64_t>(state.access));
		seed = HashCombine64(seed, static_cast<uint64_t>(state.layout));
		seed = HashCombine64(seed, static_cast<uint64_t>(state.sync));
		return seed;
	}

	template<class PassResourceData>
	uint64_t HashPassDeclaration(const PassResourceData& resources, std::span<const ResolverSnapshot> resolverSnapshots) {
		std::vector<uint64_t> entries;
		entries.reserve(resources.staticResourceRequirements.size() + resources.internalTransitions.size());

		for (const auto& req : resources.staticResourceRequirements) {
			uint64_t entry = 0x7265717569726501ull;
			entry = HashCombine64(entry, req.resourceHandleAndRange.resource.GetGlobalResourceID());
			entry = HashRangeForDeclaration(entry, req.resourceHandleAndRange.range);
			entry = HashStateForDeclaration(entry, req.state);
			entries.push_back(entry);
		}

		for (const auto& transition : resources.internalTransitions) {
			uint64_t entry = 0x7472616e73697401ull;
			entry = HashCombine64(entry, transition.first.resource.GetGlobalResourceID());
			entry = HashRangeForDeclaration(entry, transition.first.range);
			entry = HashStateForDeclaration(entry, transition.second);
			entries.push_back(entry);
		}

		for (const auto& wait : resources.externalWaitsBeforeTransitions) {
			uint64_t entry = 0x6578747761697401ull;
			entry = HashCombine64(entry, wait.timeline.GetHandle().index);
			entry = HashCombine64(entry, wait.timeline.GetHandle().generation);
			entry = HashCombine64(entry, wait.value);
			entries.push_back(entry);
		}
		for (const auto binding : resources.externalWaitBindingsBeforeTransitions) {
			entries.push_back(HashCombine64(0x65787462696e6401ull, binding));
		}

		std::sort(entries.begin(), entries.end());
		uint64_t hash = 0xd1ec1a6a710f0001ull;
		hash = HashCombine64(hash, entries.size());
		for (const uint64_t entry : entries) {
			hash = HashCombine64(hash, entry);
		}
		hash = HashCombine64(hash, HashResolverSnapshots(resolverSnapshots));
		return hash;
	}

	template<class PassAndResources>
	uint64_t BuildStaticPassAccessCacheKey(RenderGraph::PassType type, std::string_view name, const PassAndResources& passAndResources) {
		uint64_t key = 0xa11ce55acce55001ull;
		key = HashCombine64(key, static_cast<uint64_t>(type));
		key = HashCombine64(key, HashString64(name));
		key = HashCombine64(key, reinterpret_cast<uintptr_t>(passAndResources.pass.get()));
		key = HashCombine64(key, static_cast<uint64_t>(passAndResources.run));
		key = HashCombine64(key, static_cast<uint64_t>(passAndResources.resources.preferredQueueKind));
		key = HashCombine64(key, static_cast<uint64_t>(passAndResources.resources.queueAssignmentPolicy));
		key = HashCombine64(key, passAndResources.resources.pinnedQueueSlot
			? static_cast<uint64_t>(static_cast<uint8_t>(*passAndResources.resources.pinnedQueueSlot)) + 1ull
			: 0ull);
		key = HashCombine64(key, 0x57a71c5a77cacc01ull);
		key = HashCombine64(key, passAndResources.declarationCache.declarationGeneration);
		key = HashCombine64(key, passAndResources.declarationCache.declarationFingerprint);
		return key;
	}

	template<class PassAndResources>
	uint64_t BuildRetainedPassAccessCacheKey(
		const ResourceRegistry& registry,
		RenderGraph::PassType type,
		std::string_view name,
		const PassAndResources& passAndResources)
	{
		auto hashHandleAndRange = [&](uint64_t seed, const ResourceHandleAndRange& handleAndRange) {
			seed = HashCombine64(seed, handleAndRange.resource.GetGlobalResourceID());
			Resource* resource = handleAndRange.resource.IsEphemeral()
				? handleAndRange.resource.GetEphemeralPtr()
				: const_cast<Resource*>(registry.Resolve(handleAndRange.resource));
			if (auto* dynamicResource = dynamic_cast<DynamicResource*>(resource)) {
				seed = HashCombine64(seed, dynamicResource->GetDynamicWrapperGlobalResourceID());
				if (auto backing = dynamicResource->GetResource()) {
					seed = HashCombine64(seed, backing->GetGlobalResourceID());
				}
			}
			return HashRangeForDeclaration(seed, handleAndRange.range);
		};

		uint64_t key = 0xa11ce55acce55001ull;
		key = HashCombine64(key, static_cast<uint64_t>(type));
		key = HashCombine64(key, HashString64(name));
		key = HashCombine64(key, reinterpret_cast<uintptr_t>(passAndResources.pass.get()));
		key = HashCombine64(key, static_cast<uint64_t>(passAndResources.run));
		key = HashCombine64(key, static_cast<uint64_t>(passAndResources.resources.preferredQueueKind));
		key = HashCombine64(key, static_cast<uint64_t>(passAndResources.resources.queueAssignmentPolicy));
		key = HashCombine64(key, passAndResources.resources.pinnedQueueSlot
			? static_cast<uint64_t>(static_cast<uint8_t>(*passAndResources.resources.pinnedQueueSlot)) + 1ull
			: 0ull);
		key = HashCombine64(key, passAndResources.declarationCache.declarationFingerprint);

		uint64_t requirementsHash = 0xf12e5e71f12e5e71ull;
		auto reqs = GetFrameRequirementsSpan(passAndResources.resources);
		requirementsHash = HashCombine64(requirementsHash, reqs.size());
		for (const auto& req : reqs) {
			uint64_t entry = 0x7265717569726501ull;
			entry = hashHandleAndRange(entry, req.resourceHandleAndRange);
			entry = HashStateForDeclaration(entry, req.state);
			requirementsHash = HashCombine64(requirementsHash, entry);
		}
		requirementsHash = HashCombine64(requirementsHash, passAndResources.resources.internalTransitions.size());
		for (const auto& transition : passAndResources.resources.internalTransitions) {
			uint64_t entry = 0x7472616e73697401ull;
			entry = hashHandleAndRange(entry, transition.first);
			entry = HashStateForDeclaration(entry, transition.second);
			requirementsHash = HashCombine64(requirementsHash, entry);
		}
		return HashCombine64(key, requirementsHash);
	}

	template<class PassAndResources>
	void UpdateRetainedDeclarationCacheImpl(
		const ResourceRegistry& registry,
		RenderGraph::PassType type,
		std::string_view name,
		PassAndResources& passAndResources)
	{
		auto* dynamicInterface = dynamic_cast<IDynamicDeclaredResources*>(passAndResources.pass.get());
		const CachedHandleValidationInfo handleValidation = AnalyzeCachedHandleValidation(registry, passAndResources.resources);
		auto& declarationCache = passAndResources.declarationCache;
		declarationCache.hasDynamicDeclaredResources = dynamicInterface != nullptr;
		declarationCache.dynamicInterface = dynamicInterface;
		declarationCache.containsEphemeralOrAnonymousHandles = handleValidation.containsEphemeralOrAnonymousHandles;
		declarationCache.requiresStaleHandleValidation = handleValidation.requiresStaleHandleValidation;
		declarationCache.staleHandleValidationStaticRequirementAnonymousEntries = std::move(handleValidation.staticRequirementAnonymousEntries);
		declarationCache.staleHandleValidationInternalTransitionAnonymousEntries = std::move(handleValidation.internalTransitionAnonymousEntries);
		declarationCache.resolverSnapshotHash = HashResolverSnapshots(passAndResources.resolverSnapshots);
		declarationCache.declarationFingerprint = HashPassDeclaration(passAndResources.resources, passAndResources.resolverSnapshots);
		++declarationCache.declarationGeneration;
		const bool fullyStaticDeclaration = declarationCache.dynamicInterface == nullptr
			&& passAndResources.resolverSnapshots.empty()
			&& !declarationCache.requiresStaleHandleValidation;
		declarationCache.staticAccessCacheKey = fullyStaticDeclaration
			? BuildStaticPassAccessCacheKey(type, name, passAndResources)
			: 0;
		declarationCache.retainedAccessCacheKey = passAndResources.resources.frameResourceRequirements.empty()
			? BuildRetainedPassAccessCacheKey(registry, type, name, passAndResources)
			: 0;
	}

	constexpr QueueKind DefaultPreferredQueueKind(RenderGraph::PassType type) noexcept {
		switch (type) {
		case RenderGraph::PassType::Render:
			return QueueKind::Graphics;
		case RenderGraph::PassType::Compute:
			return QueueKind::Compute;
		case RenderGraph::PassType::Copy:
			return QueueKind::Copy;
		default:
			return QueueKind::Graphics;
		}
	}

	constexpr QueueAssignmentPolicy DefaultQueueAssignmentPolicy(RenderGraph::PassType type) noexcept {
		switch (type) {
		case RenderGraph::PassType::Compute:
			return QueueAssignmentPolicy::Automatic;
		case RenderGraph::PassType::Render:
		case RenderGraph::PassType::Copy:
		case RenderGraph::PassType::Unknown:
		default:
			return QueueAssignmentPolicy::ForcePreferred;
		}
	}

	constexpr bool IsPreferredQueueKindCompatible(RenderGraph::PassType type, QueueKind kind) noexcept {
		switch (type) {
		case RenderGraph::PassType::Render:
			return IsQueueKindSupportedByRenderPass(kind);
		case RenderGraph::PassType::Compute:
			return IsQueueKindSupportedByComputePass(kind);
		case RenderGraph::PassType::Copy:
			return IsQueueKindSupportedByCopyPass(kind);
		default:
			return false;
		}
	}

	bool StatesExactlyEqual(const ResourceState& lhs, const ResourceState& rhs) {
		return lhs.access == rhs.access
			&& lhs.layout == rhs.layout
			&& lhs.sync == rhs.sync;
	}

	bool IsWholeResourceRange(const RangeSpec& range, ResourceRegistry::RegistryHandle resource) {
		const uint32_t totalMips = resource.GetNumMipLevels();
		const uint32_t totalSlices = resource.GetArraySize();
		if (totalMips == 0 || totalSlices == 0) {
			return false;
		}

		const SubresourceRange resolved = ResolveRangeSpec(range, totalMips, totalSlices);
		return !resolved.isEmpty()
			&& resolved.firstMip == 0
			&& resolved.mipCount == totalMips
			&& resolved.firstSlice == 0
			&& resolved.sliceCount == totalSlices;
	}

	bool TryGetWholeResourceTrackerState(const SymbolicTracker& tracker, ResourceState& outState) {
		const auto& segments = tracker.GetSegments();
		if (segments.size() != 1) {
			return false;
		}

		const auto& segment = segments.front();
		if (segment.rangeSpec.mipLower.type != BoundType::All
			|| segment.rangeSpec.mipUpper.type != BoundType::All
			|| segment.rangeSpec.sliceLower.type != BoundType::All
			|| segment.rangeSpec.sliceUpper.type != BoundType::All) {
			return false;
		}

		outState = segment.state;
		return true;
	}

	SymbolicTracker SeedCompileTrackerFromLiveResource(Resource* resource) {
		if (auto* texture = dynamic_cast<PixelBuffer*>(resource); texture && !texture->IsMaterialized()) {
			RangeSpec wholeRange;
			wholeRange.mipLower = { BoundType::All, 0 };
			wholeRange.mipUpper = { BoundType::All, 0 };
			wholeRange.sliceLower = { BoundType::All, 0 };
			wholeRange.sliceUpper = { BoundType::All, 0 };
			return SymbolicTracker(
				wholeRange,
				ResourceState{
					rhi::ResourceAccessType::None,
					rhi::ResourceLayout::Undefined,
					rhi::ResourceSyncState::None });
		}

		SymbolicTracker seed{};
		if (HasLiveCompileResourceBacking(resource)) {
			if (auto* tracker = resource->GetStateTracker()) {
				seed = *tracker;
			}
		}
		return seed;
	}

	ResourceState NormalizeStateForQueue(QueueKind queue, ResourceState state) {
		if (queue == QueueKind::Copy) {
			const auto copyAccess = state.access & (rhi::ResourceAccessType::CopySource | rhi::ResourceAccessType::CopyDest);
			if (copyAccess != rhi::ResourceAccessType(0)) {
				state.layout = rhi::ResourceLayout::Common;
				state.sync = rhi::ResourceSyncState::Copy;
			}
		}

		return state;
	}

	const char* TransitionPlacementModeToString(org::runtime::TransitionPlacementMode mode) noexcept {
		switch (mode) {
		case org::runtime::TransitionPlacementMode::InlineEarlyPlacement: return "InlineEarlyPlacement";
		case org::runtime::TransitionPlacementMode::CanonicalThenOptimize: return "CanonicalThenOptimize";
		default: return "Unknown";
		}
	}

	std::string FormatRangeSpec(const RangeSpec& range) {
		std::ostringstream oss;
		oss << "mip=[" << range.mipLower.ToString() << ".." << range.mipUpper.ToString()
			<< "] slice=[" << range.sliceLower.ToString() << ".." << range.sliceUpper.ToString() << "]";
		return oss.str();
	}

	ResourceRegistry::RegistryHandle ResolveByIdThunk(void* user, ResourceIdentifier const& id, bool allowFailure) {
		return static_cast<RenderGraph*>(user)->RequestResourceHandle(id, allowFailure);
	}

	ResourceRegistry::RegistryHandle ResolveByPtrThunk(void* user, Resource* ptr, bool allowFailure) {
		return static_cast<RenderGraph*>(user)->RequestResourceHandle(ptr, allowFailure);
	}

	bool Overlap(SubresourceRange a, SubresourceRange b) {
		auto aMipEnd = a.firstMip + a.mipCount;
		auto bMipEnd = b.firstMip + b.mipCount;
		auto aSliceEnd = a.firstSlice + a.sliceCount;
		auto bSliceEnd = b.firstSlice + b.sliceCount;
		return (a.firstMip < bMipEnd && b.firstMip < aMipEnd) &&
			(a.firstSlice < bSliceEnd && b.firstSlice < aSliceEnd);
	}

	bool RequirementsConflict(
		std::span<const ResourceRequirement> retained,
		std::span<const ResourceRequirement> immediate)
	{
		if (retained.empty() || immediate.empty()) {
			return false;
		}

		std::unordered_map<uint64_t, std::vector<const ResourceRequirement*>> immediateByID;
		immediateByID.reserve(immediate.size());
		for (auto const& immediateRequirement : immediate) {
			immediateByID[immediateRequirement.resourceHandleAndRange.resource.GetGlobalResourceID()].push_back(&immediateRequirement);
		}

		for (auto const& retainedRequirement : retained) {
			auto resource = retainedRequirement.resourceHandleAndRange.resource;
			const uint64_t resourceID = resource.GetGlobalResourceID();
			auto it = immediateByID.find(resourceID);
			if (it == immediateByID.end()) {
				continue;
			}

			auto retainedRange = ResolveRangeSpec(
				retainedRequirement.resourceHandleAndRange.range,
				resource.GetNumMipLevels(),
				resource.GetArraySize());
			if (retainedRange.isEmpty()) {
				continue;
			}

			for (auto const* immediateRequirement : it->second) {
				auto immediateRange = ResolveRangeSpec(
					immediateRequirement->resourceHandleAndRange.range,
					resource.GetNumMipLevels(),
					resource.GetArraySize());
				if (immediateRange.isEmpty()) {
					continue;
				}

				if (Overlap(retainedRange, immediateRange) && !(retainedRequirement.state == immediateRequirement->state)) {
					return true;
				}
			}
		}
		return false;
	}
}

void RenderGraph::UpdateRetainedDeclarationCache(PassType type, std::string_view name, RenderPassAndResources& passAndResources) {
	UpdateRetainedDeclarationCacheImpl(_registry, type, name, passAndResources);
}

void RenderGraph::UpdateRetainedDeclarationCache(PassType type, std::string_view name, ComputePassAndResources& passAndResources) {
	UpdateRetainedDeclarationCacheImpl(_registry, type, name, passAndResources);
}

void RenderGraph::UpdateRetainedDeclarationCache(PassType type, std::string_view name, CopyPassAndResources& passAndResources) {
	UpdateRetainedDeclarationCacheImpl(_registry, type, name, passAndResources);
}

void RenderGraph::RebuildFramePassAccessSummaries() {
	BT_ZONE_SCOPE("RenderGraph::RebuildFramePassAccessSummaries");
	BT_ZONE_VALUE(m_framePasses.size());
	auto& compiler = *m_compilerState;
	{
		BT_ZONE_SCOPE("RGPassAccess::Initialize");
		m_framePassAccessSummaries.resize(m_framePasses.size());
	}

	auto resolveHandleResource = [&](const ResourceRegistry::RegistryHandle& handle) -> Resource* {
		return handle.IsEphemeral()
			? handle.GetEphemeralPtr()
			: _registry.Resolve(handle);
	};

	auto schedulingResourceIDForHandle = [&](const ResourceRegistry::RegistryHandle& handle) {
		Resource* resource = handle.IsEphemeral()
			? handle.GetEphemeralPtr()
			: _registry.Resolve(handle);
		return resource ? resource->GetSchedulingResourceID() : handle.GetGlobalResourceID();
	};

	auto appendHandleResourceIDs = [&](std::vector<uint64_t>& out, const ResourceRegistry::RegistryHandle& handle) {
		out.push_back(handle.GetGlobalResourceID());
		Resource* resource = handle.IsEphemeral()
			? handle.GetEphemeralPtr()
			: _registry.Resolve(handle);
		if (auto* dynamicResource = dynamic_cast<DynamicResource*>(resource)) {
			out.push_back(dynamicResource->GetDynamicWrapperGlobalResourceID());
			out.push_back(dynamicResource->GetGlobalResourceID());
			if (auto backing = dynamicResource->GetResource()) {
				out.push_back(backing->GetGlobalResourceID());
			}
		}
	};

	auto appendHandleResourceIDsResolved = [&](std::vector<uint64_t>& out, const ResourceRegistry::RegistryHandle& handle, Resource* resource) {
		const uint64_t handleID = handle.GetGlobalResourceID();
		out.push_back(handleID);
		if (resource) {
			const uint64_t stableID = resource->GetSchedulingResourceID();
			if (stableID != handleID) {
				out.push_back(stableID);
			}
			const uint64_t currentID = resource->GetGlobalResourceID();
			if (currentID != handleID && currentID != stableID) {
				out.push_back(currentID);
			}
			return stableID;
		}
		return handleID;
	};

	{
		BT_ZONE_SCOPE("RGPassAccess::DenseFullRebuild");
		compiler.resourceIDs.clear();
		compiler.densePassAccessKeys.resize(m_framePasses.size(), 0);

		{
			BT_ZONE_SCOPE("RGPassAccess::InitializeDenseResourceIndex");
			size_t hashCapacity = m_frameDAGResourceIndexHashKeys.size();
			if (hashCapacity == 0) {
				hashCapacity = 1024;
				m_frameDAGResourceIndexHashKeys.resize(hashCapacity);
				m_frameDAGResourceIndexHashValues.resize(hashCapacity);
			}
			std::fill(
				m_frameDAGResourceIndexHashKeys.begin(),
				m_frameDAGResourceIndexHashKeys.end(),
				kFrameDAGResourceIndexEmptyKey);
			m_frameDAGResourceIDsByIndex.clear();
			m_frameDAGResourcePtrByIndex.clear();
			m_frameDAGUnmaterializedResourceIndices.clear();
			compiler.resourcesWritten.clear();
		}

		auto growDenseResourceIndex = [&]() {
			const size_t newCapacity = m_frameDAGResourceIndexHashKeys.size() * 2;
			m_frameDAGResourceIndexHashKeys.assign(newCapacity, kFrameDAGResourceIndexEmptyKey);
			m_frameDAGResourceIndexHashValues.resize(newCapacity);
			const size_t hashMask = newCapacity - 1;
			for (uint32_t resourceIndex = 0;
				resourceIndex < static_cast<uint32_t>(m_frameDAGResourceIDsByIndex.size());
				++resourceIndex) {
				const uint64_t resourceID = m_frameDAGResourceIDsByIndex[resourceIndex];
				size_t hashSlot = static_cast<size_t>(MixFrameDAGResourceID(resourceID)) & hashMask;
				while (m_frameDAGResourceIndexHashKeys[hashSlot] != kFrameDAGResourceIndexEmptyKey) {
					hashSlot = (hashSlot + 1) & hashMask;
				}
				m_frameDAGResourceIndexHashKeys[hashSlot] = resourceID;
				m_frameDAGResourceIndexHashValues[hashSlot] = resourceIndex;
			}
		};

		auto insertDenseResourceID = [&](uint64_t resourceID) -> uint32_t {
			if (resourceID == kFrameDAGResourceIndexEmptyKey) {
				return UINT32_MAX;
			}
			if ((m_frameDAGResourceIDsByIndex.size() + 1) * 2
				>= m_frameDAGResourceIndexHashKeys.size()) {
				growDenseResourceIndex();
			}

			const size_t hashMask = m_frameDAGResourceIndexHashKeys.size() - 1;
			size_t hashSlot = static_cast<size_t>(MixFrameDAGResourceID(resourceID)) & hashMask;
			for (;;) {
				const uint64_t key = m_frameDAGResourceIndexHashKeys[hashSlot];
				if (key == resourceID) {
					return m_frameDAGResourceIndexHashValues[hashSlot];
				}
				if (key == kFrameDAGResourceIndexEmptyKey) {
					const uint32_t resourceIndex = static_cast<uint32_t>(m_frameDAGResourceIDsByIndex.size());
					m_frameDAGResourceIndexHashKeys[hashSlot] = resourceID;
					m_frameDAGResourceIndexHashValues[hashSlot] = resourceIndex;
					m_frameDAGResourceIDsByIndex.push_back(resourceID);
					m_frameDAGResourcePtrByIndex.push_back(nullptr);
					compiler.resourcesWritten.push_back(0);
					return resourceIndex;
				}
				hashSlot = (hashSlot + 1) & hashMask;
			}
		};

		auto registerDenseHandleResource = [&](const ResourceRegistry::RegistryHandle& handle, Resource* resource) {
			const uint64_t handleID = handle.GetGlobalResourceID();
			const uint32_t handleIndex = insertDenseResourceID(handleID);
			if (!resource) {
				return std::pair<uint64_t, uint32_t>{ handleID, handleIndex };
			}

			const uint64_t stableID = resource->GetSchedulingResourceID();
			const uint32_t stableIndex = stableID == handleID
				? handleIndex
				: insertDenseResourceID(stableID);
			const uint64_t currentID = resource->GetGlobalResourceID();
			if (currentID != handleID && currentID != stableID) {
				insertDenseResourceID(currentID);
			}
			return std::pair<uint64_t, uint32_t>{ stableID, stableIndex };
		};

		auto captureDenseResourcePtr = [&](uint32_t dagResourceIndex, Resource* resolvedResource) {
			if (dagResourceIndex == UINT32_MAX
				|| dagResourceIndex >= m_frameDAGResourcePtrByIndex.size()
				|| m_frameDAGResourcePtrByIndex[dagResourceIndex] != nullptr) {
				return;
			}
			m_frameDAGResourcePtrByIndex[dagResourceIndex] = resolvedResource;
			if (auto* backedResource = TryGetBackedResource(resolvedResource);
				backedResource && !backedResource->IsMaterialized()) {
				m_frameDAGUnmaterializedResourceIndices.push_back(dagResourceIndex);
			}
		};

		{
			BT_ZONE_SCOPE("RGPassAccess::BuildDenseSummaries");
			for (size_t passIndex = 0; passIndex < m_framePasses.size(); ++passIndex) {
				const auto& pass = m_framePasses[passIndex];
				auto& summary = m_framePassAccessSummaries[passIndex];
				uint64_t accessKey = 0;
				if (passIndex >= m_framePassIsFrameExtension.size()
					|| m_framePassIsFrameExtension[passIndex] == 0) {
					accessKey = std::visit([](const auto& passAndResources) -> uint64_t {
						using T = std::decay_t<decltype(passAndResources)>;
						if constexpr (std::is_same_v<T, std::monostate>) {
							return 0;
						}
						else {
							if (passAndResources.run != PassRunMask::Retained
								|| !passAndResources.resources.frameResourceRequirements.empty()) {
								return 0;
							}
							const uint64_t declarationKey = passAndResources.declarationCache.staticAccessCacheKey != 0
								? passAndResources.declarationCache.staticAccessCacheKey
								: passAndResources.declarationCache.retainedAccessCacheKey;
							return declarationKey == 0
								? 0
								: HashCombine64(declarationKey, reinterpret_cast<uintptr_t>(passAndResources.pass.get()));
						}
					}, pass.pass);
				}
				const bool reuseSummary = accessKey != 0
					&& compiler.densePassAccessKeys[passIndex] == accessKey
					&& summary.type == pass.type;
				compiler.densePassAccessKeys[passIndex] = accessKey;

				if (!reuseSummary) {
					summary.requirementSummaries.clear();
					summary.internalTransitionSummaries.clear();
					summary.touchedResourceIDs.clear();
					summary.uavResourceIDs.clear();
					summary.dagAccesses.clear();
					summary.type = pass.type;
					summary.preferredQueueKind = DefaultPreferredQueueKind(pass.type);
					summary.queueAssignmentPolicy = DefaultQueueAssignmentPolicy(pass.type);
					summary.pinnedQueueSlot.reset();
				}

				PassView view = GetPassView(pass);
				if (!reuseSummary && summary.requirementSummaries.capacity() < view.reqs.size()) {
					summary.requirementSummaries.reserve(view.reqs.size());
				}
				const size_t internalTransitionCount = view.internalTransitions ? view.internalTransitions->size() : 0;
				if (!reuseSummary && summary.internalTransitionSummaries.capacity() < internalTransitionCount) {
					summary.internalTransitionSummaries.reserve(internalTransitionCount);
				}
				if (!reuseSummary && pass.type == PassType::Render) {
					const auto& passResources = std::get<RenderPassAndResources>(pass.pass).resources;
					summary.preferredQueueKind = passResources.preferredQueueKind;
					summary.queueAssignmentPolicy = passResources.queueAssignmentPolicy;
					summary.pinnedQueueSlot = passResources.pinnedQueueSlot;
					summary.backendAffinity = passResources.backendAffinity;
				}
				else if (!reuseSummary && pass.type == PassType::Compute) {
					const auto& passResources = std::get<ComputePassAndResources>(pass.pass).resources;
					summary.preferredQueueKind = passResources.preferredQueueKind;
					summary.queueAssignmentPolicy = passResources.queueAssignmentPolicy;
					summary.pinnedQueueSlot = passResources.pinnedQueueSlot;
					summary.backendAffinity = passResources.backendAffinity;
				}
				else if (!reuseSummary && pass.type == PassType::Copy) {
					const auto& passResources = std::get<CopyPassAndResources>(pass.pass).resources;
					summary.preferredQueueKind = passResources.preferredQueueKind;
					summary.queueAssignmentPolicy = passResources.queueAssignmentPolicy;
					summary.pinnedQueueSlot = passResources.pinnedQueueSlot;
					summary.backendAffinity = passResources.backendAffinity;
				}

				if (reuseSummary) {
					for (auto& req : summary.requirementSummaries) {
						const auto [resourceID, dagResourceIndex] =
							registerDenseHandleResource(req.resource, req.resolvedResource);
						req.resourceID = resourceID;
						req.dagResourceIndex = dagResourceIndex;
						captureDenseResourcePtr(dagResourceIndex, req.resolvedResource);
						if (req.isWrite && dagResourceIndex < compiler.resourcesWritten.size()) {
							compiler.resourcesWritten[dagResourceIndex] = 1;
						}
					}
					for (auto& transition : summary.internalTransitionSummaries) {
						const auto [resourceID, dagResourceIndex] =
							registerDenseHandleResource(transition.resource, transition.resolvedResource);
						transition.resourceID = resourceID;
						transition.dagResourceIndex = dagResourceIndex;
						captureDenseResourcePtr(dagResourceIndex, transition.resolvedResource);
						if (dagResourceIndex < compiler.resourcesWritten.size()) {
							compiler.resourcesWritten[dagResourceIndex] = 1;
						}
					}
					continue;
				}

				for (const auto& req : view.reqs) {
					const auto resource = req.resourceHandleAndRange.resource;
					Resource* resolvedResource = resolveHandleResource(resource);
					const auto [resourceID, dagResourceIndex] =
						registerDenseHandleResource(resource, resolvedResource);
					captureDenseResourcePtr(dagResourceIndex, resolvedResource);
					const bool isWrite = AccessTypeIsWriteType(req.state.access);
					if (isWrite && dagResourceIndex < compiler.resourcesWritten.size()) {
						compiler.resourcesWritten[dagResourceIndex] = 1;
					}
					summary.requirementSummaries.push_back(FramePassRequirementStaticSummary{
						.resource = resource,
						.resolvedResource = resolvedResource,
						.resourceID = resourceID,
						.dagResourceIndex = dagResourceIndex,
						.range = req.resourceHandleAndRange.range,
						.state = req.state,
						.isUAV = IsUAVState(req.state),
						.isWrite = isWrite,
					});
				}

				if (view.internalTransitions) {
					for (const auto& transition : *view.internalTransitions) {
						const auto resource = transition.first.resource;
						Resource* resolvedResource = resolveHandleResource(resource);
						const auto [resourceID, dagResourceIndex] =
							registerDenseHandleResource(resource, resolvedResource);
						captureDenseResourcePtr(dagResourceIndex, resolvedResource);
						if (dagResourceIndex < compiler.resourcesWritten.size()) {
							compiler.resourcesWritten[dagResourceIndex] = 1;
						}
						summary.internalTransitionSummaries.push_back(FramePassInternalTransitionStaticSummary{
							.resource = resource,
							.resolvedResource = resolvedResource,
							.resourceID = resourceID,
							.dagResourceIndex = dagResourceIndex,
						});
					}
				}
			}
		}

		m_frameDAGResourceCount = m_frameDAGResourceIDsByIndex.size();

		{
			BT_ZONE_SCOPE("RGPassAccess::FinalizeDensePassAccessLists");
			const auto ensureEpochScratch = [&]() {
				if (compiler.accessEpochs.size() < m_frameDAGResourceCount) {
					compiler.accessEpochs.resize(m_frameDAGResourceCount, 0);
					compiler.accessWriteEpochs.resize(m_frameDAGResourceCount, 0);
					compiler.accessUavEpochs.resize(m_frameDAGResourceCount, 0);
					compiler.accessDagEpochs.resize(m_frameDAGResourceCount, 0);
				}
				if (compiler.accessEpoch == std::numeric_limits<uint32_t>::max()) {
					std::fill(compiler.accessEpochs.begin(), compiler.accessEpochs.end(), 0);
					std::fill(compiler.accessWriteEpochs.begin(), compiler.accessWriteEpochs.end(), 0);
					std::fill(compiler.accessUavEpochs.begin(), compiler.accessUavEpochs.end(), 0);
					std::fill(compiler.accessDagEpochs.begin(), compiler.accessDagEpochs.end(), 0);
					compiler.accessEpoch = 1;
				}
			};
			ensureEpochScratch();

			for (auto& summary : m_framePassAccessSummaries) {
				summary.touchedResourceIDs.clear();
				summary.uavResourceIDs.clear();
				summary.dagAccesses.clear();
				compiler.accessOrder.clear();

				const uint32_t epoch = compiler.accessEpoch++;
				auto mark = [&](uint32_t dagResourceIndex, AccessKind accessKind, bool isUav) {
					if (dagResourceIndex == UINT32_MAX || dagResourceIndex >= m_frameDAGResourceIDsByIndex.size()) {
						return;
					}

					if (compiler.accessEpochs[dagResourceIndex] != epoch) {
						compiler.accessEpochs[dagResourceIndex] = epoch;
						compiler.accessOrder.push_back(dagResourceIndex);
					}
					if (isUav) {
						compiler.accessUavEpochs[dagResourceIndex] = epoch;
					}
					if (accessKind == AccessKind::Write) {
						compiler.accessWriteEpochs[dagResourceIndex] = epoch;
						compiler.accessDagEpochs[dagResourceIndex] = epoch;
					}
					else if (dagResourceIndex < compiler.resourcesWritten.size()
						&& compiler.resourcesWritten[dagResourceIndex] != 0) {
						compiler.accessDagEpochs[dagResourceIndex] = epoch;
					}
				};

				for (const auto& req : summary.requirementSummaries) {
					mark(req.dagResourceIndex, req.isWrite ? AccessKind::Write : AccessKind::Read, req.isUAV);
				}
				for (const auto& transition : summary.internalTransitionSummaries) {
					mark(transition.dagResourceIndex, AccessKind::Write, false);
				}

				if (summary.touchedResourceIDs.capacity() < compiler.accessOrder.size()) {
					summary.touchedResourceIDs.reserve(compiler.accessOrder.size());
				}
				if (summary.dagAccesses.capacity() < compiler.accessOrder.size()) {
					summary.dagAccesses.reserve(compiler.accessOrder.size());
				}

				for (uint32_t dagResourceIndex : compiler.accessOrder) {
					const uint64_t resourceID = m_frameDAGResourceIDsByIndex[dagResourceIndex];
					summary.touchedResourceIDs.push_back(resourceID);
					if (compiler.accessUavEpochs[dagResourceIndex] == epoch) {
						summary.uavResourceIDs.push_back(resourceID);
					}
					if (compiler.accessDagEpochs[dagResourceIndex] == epoch) {
						summary.dagAccesses.push_back(NodeAccess{
							.resourceIndex = dagResourceIndex,
							.kind = compiler.accessWriteEpochs[dagResourceIndex] == epoch ? AccessKind::Write : AccessKind::Read,
						});
					}
				}
			}
		}

		return;
	}

	auto hashHandleAndRange = [&](uint64_t seed, const ResourceHandleAndRange& handleAndRange) {
		seed = HashCombine64(seed, handleAndRange.resource.GetGlobalResourceID());
		Resource* resource = handleAndRange.resource.IsEphemeral()
			? handleAndRange.resource.GetEphemeralPtr()
			: _registry.Resolve(handleAndRange.resource);
		if (auto* dynamicResource = dynamic_cast<DynamicResource*>(resource)) {
			seed = HashCombine64(seed, dynamicResource->GetDynamicWrapperGlobalResourceID());
			if (auto backing = dynamicResource->GetResource()) {
				seed = HashCombine64(seed, backing->GetGlobalResourceID());
			}
		}
		return HashRangeForDeclaration(seed, handleAndRange.range);
	};

	auto retainedDeclarationFullyStatic = [](const auto& passAndResources) {
		const auto& cache = passAndResources.declarationCache;
		return cache.dynamicInterface == nullptr
			&& passAndResources.resolverSnapshots.empty()
			&& !cache.requiresStaleHandleValidation;
	};

	auto buildPassAccessKey = [&](const AnyPassAndResources& pass) {
		uint64_t key = 0xa11ce55acce55001ull;
		key = HashCombine64(key, static_cast<uint64_t>(pass.type));
		key = HashCombine64(key, HashString64(pass.name));

		std::visit([&](auto const& passAndResources) {
			using T = std::decay_t<decltype(passAndResources)>;
			if constexpr (!std::is_same_v<T, std::monostate>) {
				key = HashCombine64(key, reinterpret_cast<uintptr_t>(passAndResources.pass.get()));
				key = HashCombine64(key, static_cast<uint64_t>(passAndResources.run));
				key = HashCombine64(key, static_cast<uint64_t>(passAndResources.resources.preferredQueueKind));
				key = HashCombine64(key, static_cast<uint64_t>(passAndResources.resources.queueAssignmentPolicy));
				key = HashCombine64(key, passAndResources.resources.pinnedQueueSlot
					? static_cast<uint64_t>(static_cast<uint8_t>(*passAndResources.resources.pinnedQueueSlot)) + 1ull
					: 0ull);
				if (retainedDeclarationFullyStatic(passAndResources)) {
					key = HashCombine64(key, 0x57a71c5a77cacc01ull);
					key = HashCombine64(key, passAndResources.declarationCache.declarationGeneration);
					key = HashCombine64(key, passAndResources.declarationCache.declarationFingerprint);
				}
				else {
					key = HashCombine64(key, passAndResources.declarationCache.declarationFingerprint);

					uint64_t requirementsHash = 0xf12e5e71f12e5e71ull;
					auto reqs = GetFrameRequirementsSpan(passAndResources.resources);
					requirementsHash = HashCombine64(requirementsHash, reqs.size());
					for (const auto& req : reqs) {
						uint64_t entry = 0x7265717569726501ull;
						entry = hashHandleAndRange(entry, req.resourceHandleAndRange);
						entry = HashStateForDeclaration(entry, req.state);
						requirementsHash = HashCombine64(requirementsHash, entry);
					}
					requirementsHash = HashCombine64(requirementsHash, passAndResources.resources.internalTransitions.size());
					for (const auto& transition : passAndResources.resources.internalTransitions) {
						uint64_t entry = 0x7472616e73697401ull;
						entry = hashHandleAndRange(entry, transition.first);
						entry = HashStateForDeclaration(entry, transition.second);
						requirementsHash = HashCombine64(requirementsHash, entry);
					}
					key = HashCombine64(key, requirementsHash);
				}
			}
		}, pass.pass);

		return key;
	};

	auto tryGetPrecomputedStaticPassAccessKey = [](const AnyPassAndResources& pass) -> uint64_t {
		return std::visit(
			[](auto const& passAndResources) -> uint64_t {
				using T = std::decay_t<decltype(passAndResources)>;
				if constexpr (std::is_same_v<T, std::monostate>) {
					return 0;
				}
				else {
					return passAndResources.declarationCache.staticAccessCacheKey;
				}
			},
			pass.pass);
	};
	auto tryGetPrecomputedRetainedPassAccessKey = [](const AnyPassAndResources& pass) -> uint64_t {
		return std::visit(
			[](auto const& passAndResources) -> uint64_t {
				using T = std::decay_t<decltype(passAndResources)>;
				if constexpr (std::is_same_v<T, std::monostate>) {
					return 0;
				}
				else {
					return passAndResources.declarationCache.retainedAccessCacheKey;
				}
			},
			pass.pass);
	};

	struct PassAccessWorkItem {
		uint64_t cacheKey = 0;
		bool cacheable = false;
		bool cacheHit = false;
		const CachedFramePassAccessSummary* cachedSummary = nullptr;
		FramePassStaticAccessSummary summary;
		std::vector<uint64_t> usedResourceIDs;
	};

	auto passAccessSummaryCacheable = [&](size_t passIndex, const AnyPassAndResources& pass) {
		(void)passIndex;
		(void)pass;
		return false;
#if 0
		if (passIndex < m_framePassIsFrameExtension.size() && m_framePassIsFrameExtension[passIndex] != 0) {
			return false;
		}

		bool cacheable = true;
		std::visit([&](auto const& passAndResources) {
			using T = std::decay_t<decltype(passAndResources)>;
			if constexpr (!std::is_same_v<T, std::monostate>) {
				cacheable = passAndResources.resources.frameResourceRequirements.empty()
					&& passAndResources.run == PassRunMask::Retained;
			}
		}, pass.pass);
		return cacheable;
#endif
	};

	std::vector<PassAccessWorkItem> workItems(m_framePasses.size());
	size_t estimatedUsedResourceIDCount = 0;
	{
		BT_ZONE_SCOPE("RGPassAccess::BuildKeysAndLoadCacheHits");
		uint64_t cacheablePassCount = 0;
		uint64_t staticPrecomputedKeyCount = 0;
		uint64_t dynamicKeyBuildCount = 0;
		uint64_t cacheHitCount = 0;
		uint64_t cacheMissCount = 0;
		uint64_t nonCacheablePassCount = 0;
		uint64_t retainedPrecomputedKeyCount = 0;
		uint64_t estimatedCacheHitResourceIDs = 0;
		uint64_t estimatedCacheHitRequirements = 0;
		uint64_t estimatedCacheHitTransitions = 0;
		for (size_t passIndex = 0; passIndex < m_framePasses.size(); ++passIndex) {
			auto& workItem = workItems[passIndex];
			{
				BT_ZONE_SCOPE("RGPassAccess::CheckCacheable");
				workItem.cacheable = passAccessSummaryCacheable(passIndex, m_framePasses[passIndex]);
			}
			if (workItem.cacheable) {
				++cacheablePassCount;
				{
					BT_ZONE_SCOPE("RGPassAccess::TryPrecomputedStaticKey");
					workItem.cacheKey = tryGetPrecomputedStaticPassAccessKey(m_framePasses[passIndex]);
				}
				if (workItem.cacheKey == 0) {
					BT_ZONE_SCOPE("RGPassAccess::TryPrecomputedRetainedKey");
					workItem.cacheKey = tryGetPrecomputedRetainedPassAccessKey(m_framePasses[passIndex]);
					if (workItem.cacheKey != 0) {
						++retainedPrecomputedKeyCount;
					}
				}
				else {
					++staticPrecomputedKeyCount;
				}
				if (workItem.cacheKey == 0) {
					++dynamicKeyBuildCount;
					BT_ZONE_SCOPE("RGPassAccess::BuildDynamicAccessKey");
					workItem.cacheKey = buildPassAccessKey(m_framePasses[passIndex]);
				}
			}
			else {
				++nonCacheablePassCount;
				workItem.cacheKey = 0;
			}
			if (workItem.cacheable) {
				BT_ZONE_SCOPE("RGPassAccess::LookupAccessSummaryCache");
				auto cacheIt = m_framePassAccessSummaryCache.find(workItem.cacheKey);
				if (cacheIt != m_framePassAccessSummaryCache.end()) {
					workItem.cacheHit = true;
					workItem.cachedSummary = &cacheIt->second;
					estimatedUsedResourceIDCount += cacheIt->second.usedResourceIDs.size();
					estimatedCacheHitResourceIDs += cacheIt->second.usedResourceIDs.size();
					estimatedCacheHitRequirements += cacheIt->second.summary.requirementSummaries.size();
					estimatedCacheHitTransitions += cacheIt->second.summary.internalTransitionSummaries.size();
					++cacheHitCount;
				}
				else {
					++cacheMissCount;
				}
			}
		}
		BT_ZONE_VALUE(cacheablePassCount);
		BT_PLOT("RGPassAccess.CacheablePasses", static_cast<int64_t>(cacheablePassCount));
		BT_PLOT("RGPassAccess.NonCacheablePasses", static_cast<int64_t>(nonCacheablePassCount));
		BT_PLOT("RGPassAccess.StaticPrecomputedKeys", static_cast<int64_t>(staticPrecomputedKeyCount));
		BT_PLOT("RGPassAccess.RetainedPrecomputedKeys", static_cast<int64_t>(retainedPrecomputedKeyCount));
		BT_PLOT("RGPassAccess.DynamicKeyBuilds", static_cast<int64_t>(dynamicKeyBuildCount));
		BT_PLOT("RGPassAccess.CacheHits", static_cast<int64_t>(cacheHitCount));
		BT_PLOT("RGPassAccess.CacheMisses", static_cast<int64_t>(cacheMissCount));
		BT_PLOT("RGPassAccess.CacheHitResourceIDs", static_cast<int64_t>(estimatedCacheHitResourceIDs));
		BT_PLOT("RGPassAccess.CacheHitRequirements", static_cast<int64_t>(estimatedCacheHitRequirements));
		BT_PLOT("RGPassAccess.CacheHitTransitions", static_cast<int64_t>(estimatedCacheHitTransitions));
	}

	auto buildPassSummary = [&](size_t passIndex) {
		auto& workItem = workItems[passIndex];
		if (workItem.cacheHit) {
			return;
		}
		BT_ZONE_SCOPE("RGPassAccess::BuildCacheMissSummary");
		BT_ZONE_VALUE(passIndex);

		const auto& pass = m_framePasses[passIndex];
		auto& summary = workItem.summary;
		summary = FramePassStaticAccessSummary{};
		summary.type = pass.type;
		summary.preferredQueueKind = DefaultPreferredQueueKind(pass.type);
		summary.queueAssignmentPolicy = DefaultQueueAssignmentPolicy(pass.type);

		PassView view = GetPassView(pass);
		if (!view.reqs.empty()) {
			summary.requirementSummaries.reserve(view.reqs.size());
		}
		if (view.internalTransitions) {
			summary.internalTransitionSummaries.reserve(view.internalTransitions->size());
		}

		if (pass.type == PassType::Render) {
			const auto& passResources = std::get<RenderPassAndResources>(pass.pass).resources;
			summary.preferredQueueKind = passResources.preferredQueueKind;
			summary.queueAssignmentPolicy = passResources.queueAssignmentPolicy;
			summary.pinnedQueueSlot = passResources.pinnedQueueSlot;
			summary.backendAffinity = passResources.backendAffinity;
		}
		else if (pass.type == PassType::Compute) {
			const auto& passResources = std::get<ComputePassAndResources>(pass.pass).resources;
			summary.preferredQueueKind = passResources.preferredQueueKind;
			summary.queueAssignmentPolicy = passResources.queueAssignmentPolicy;
			summary.pinnedQueueSlot = passResources.pinnedQueueSlot;
			summary.backendAffinity = passResources.backendAffinity;
		}
		else if (pass.type == PassType::Copy) {
			const auto& passResources = std::get<CopyPassAndResources>(pass.pass).resources;
			summary.preferredQueueKind = passResources.preferredQueueKind;
			summary.queueAssignmentPolicy = passResources.queueAssignmentPolicy;
			summary.pinnedQueueSlot = passResources.pinnedQueueSlot;
			summary.backendAffinity = passResources.backendAffinity;
		}

		if (!view.reqs.empty()) {
			workItem.usedResourceIDs.reserve((view.reqs.size() + (view.internalTransitions ? view.internalTransitions->size() : 0)) * 4);
			for (const auto& req : view.reqs) {
				const auto resource = req.resourceHandleAndRange.resource;
				const uint64_t resourceID = schedulingResourceIDForHandle(resource);
				const bool isWrite = AccessTypeIsWriteType(req.state.access);
				const bool isUAV = IsUAVState(req.state);
				appendHandleResourceIDs(workItem.usedResourceIDs, resource);

				summary.requirementSummaries.push_back(FramePassRequirementStaticSummary{
					.resource = resource,
					.resourceID = resourceID,
					.range = req.resourceHandleAndRange.range,
					.state = req.state,
					.isUAV = isUAV,
					.isWrite = isWrite,
				});
			}
		}

		if (view.internalTransitions) {
			for (const auto& transition : *view.internalTransitions) {
				const auto resource = transition.first.resource;
				const uint64_t resourceID = schedulingResourceIDForHandle(resource);
				appendHandleResourceIDs(workItem.usedResourceIDs, resource);

				summary.internalTransitionSummaries.push_back(FramePassInternalTransitionStaticSummary{
					.resource = resource,
					.resourceID = resourceID,
				});
			}
		}

		std::sort(workItem.usedResourceIDs.begin(), workItem.usedResourceIDs.end());
		workItem.usedResourceIDs.erase(
			std::unique(workItem.usedResourceIDs.begin(), workItem.usedResourceIDs.end()),
			workItem.usedResourceIDs.end());
	};

	{
		BT_ZONE_SCOPE("RGPassAccess::ParallelBuildCacheMisses");
		ParallelForOptional("RGPrecompilePassAccess", workItems.size(), buildPassSummary);
	}

	{
		BT_ZONE_SCOPE("RGPassAccess::PublishSummariesAndCacheMisses");
		for (size_t passIndex = 0; passIndex < workItems.size(); ++passIndex) {
			auto& workItem = workItems[passIndex];
			if (workItem.cacheable && !workItem.cacheHit) {
				m_framePassAccessSummaryCache[workItem.cacheKey] = CachedFramePassAccessSummary{
					.key = workItem.cacheKey,
					.summary = workItem.summary,
					.usedResourceIDs = workItem.usedResourceIDs,
				};
				estimatedUsedResourceIDCount += workItem.usedResourceIDs.size();
			}
			m_framePassAccessSummaries[passIndex] = workItem.cacheHit && workItem.cachedSummary
				? workItem.cachedSummary->summary
				: std::move(workItem.summary);
		}
	}

	{
		BT_ZONE_SCOPE("RGPassAccess::MergeUsedResourceIDs");
		std::vector<uint64_t> flattenedResourceIDs;
		flattenedResourceIDs.reserve(estimatedUsedResourceIDCount);
		for (const auto& workItem : workItems) {
			const auto& usedResourceIDs = workItem.cacheHit && workItem.cachedSummary
				? workItem.cachedSummary->usedResourceIDs
				: workItem.usedResourceIDs;
			flattenedResourceIDs.insert(flattenedResourceIDs.end(), usedResourceIDs.begin(), usedResourceIDs.end());
		}
		std::sort(flattenedResourceIDs.begin(), flattenedResourceIDs.end());
		flattenedResourceIDs.erase(
			std::unique(flattenedResourceIDs.begin(), flattenedResourceIDs.end()),
			flattenedResourceIDs.end());

		{
			BT_ZONE_SCOPE("RGPassAccess::BuildDAGResourceIndex");
			m_frameDAGResourceIDsByIndex = std::move(flattenedResourceIDs);
			m_frameDAGResourceCount = m_frameDAGResourceIDsByIndex.size();
			m_frameDAGResourcePtrByIndex.assign(m_frameDAGResourceCount, nullptr);
			m_frameDAGUnmaterializedResourceIndices.clear();
			if (m_frameDAGUnmaterializedResourceIndices.capacity() < m_frameDAGResourceCount) {
				m_frameDAGUnmaterializedResourceIndices.reserve(m_frameDAGResourceCount);
			}
		}
	}

	{
		BT_ZONE_SCOPE("RGPassAccess::AssignDenseDAGIndicesAndPtrs");
		auto findDAGResourceIndex = [&](uint64_t resourceID) {
			const auto it = std::lower_bound(
				m_frameDAGResourceIDsByIndex.begin(),
				m_frameDAGResourceIDsByIndex.end(),
				resourceID);
			return it != m_frameDAGResourceIDsByIndex.end() && *it == resourceID
				? static_cast<uint32_t>(it - m_frameDAGResourceIDsByIndex.begin())
				: UINT32_MAX;
		};
		auto captureDenseResourcePtr = [&](uint32_t dagResourceIndex, const ResourceRegistry::RegistryHandle& resource) {
			if (dagResourceIndex == UINT32_MAX
				|| dagResourceIndex >= m_frameDAGResourcePtrByIndex.size()
				|| m_frameDAGResourcePtrByIndex[dagResourceIndex] != nullptr) {
				return;
			}
			Resource* resolvedResource = resource.IsEphemeral()
				? resource.GetEphemeralPtr()
				: _registry.Resolve(resource);
			m_frameDAGResourcePtrByIndex[dagResourceIndex] = resolvedResource;
			if (auto* backedResource = TryGetBackedResource(resolvedResource);
				backedResource && !backedResource->IsMaterialized()) {
				m_frameDAGUnmaterializedResourceIndices.push_back(dagResourceIndex);
			}
		};
		for (auto& summary : m_framePassAccessSummaries) {
			for (auto& req : summary.requirementSummaries) {
				req.dagResourceIndex = findDAGResourceIndex(req.resourceID);
				captureDenseResourcePtr(req.dagResourceIndex, req.resource);
			}
			for (auto& transition : summary.internalTransitionSummaries) {
				transition.dagResourceIndex = findDAGResourceIndex(transition.resourceID);
				captureDenseResourcePtr(transition.dagResourceIndex, transition.resource);
			}
		}
	}

	std::vector<uint8_t> resourcesWrittenThisFrame(m_frameDAGResourceCount, 0);
	{
		BT_ZONE_SCOPE("RGPassAccess::MarkWrittenDAGResources");
		for (const auto& summary : m_framePassAccessSummaries) {
			for (const auto& req : summary.requirementSummaries) {
				if (!req.isWrite) {
					continue;
				}
				if (req.dagResourceIndex < resourcesWrittenThisFrame.size()) {
					resourcesWrittenThisFrame[req.dagResourceIndex] = 1;
				}
			}
			for (const auto& transition : summary.internalTransitionSummaries) {
				if (transition.dagResourceIndex < resourcesWrittenThisFrame.size()) {
					resourcesWrittenThisFrame[transition.dagResourceIndex] = 1;
				}
			}
		}
	}

	{
		BT_ZONE_SCOPE("RGPassAccess::FinalizePerPassAccessLists");
		ParallelForOptional("RGFinalizePassAccessLists", m_framePassAccessSummaries.size(), [&](size_t passIndex) {
			auto& summary = m_framePassAccessSummaries[passIndex];
			summary.touchedResourceIDs.clear();
			summary.uavResourceIDs.clear();
			summary.dagAccesses.clear();

			struct AccessRecord {
				uint32_t dagResourceIndex = 0;
				uint64_t resourceID = 0;
				bool touched = false;
				bool uav = false;
				bool contributesToDag = false;
				bool write = false;
			};

			std::vector<AccessRecord> records;
			records.reserve(summary.requirementSummaries.size() + summary.internalTransitionSummaries.size());

			auto mark = [&](uint32_t dagResourceIndex, uint64_t resourceID, AccessKind accessKind, bool isUav) {
				if (dagResourceIndex == UINT32_MAX) {
					return;
				}

				records.push_back(AccessRecord{
					.dagResourceIndex = dagResourceIndex,
					.resourceID = resourceID,
					.touched = true,
					.uav = isUav,
					.contributesToDag = accessKind == AccessKind::Write
						|| (dagResourceIndex < resourcesWrittenThisFrame.size() && resourcesWrittenThisFrame[dagResourceIndex] != 0),
					.write = accessKind == AccessKind::Write,
				});
			};

			summary.touchedResourceIDs.reserve(summary.requirementSummaries.size() + summary.internalTransitionSummaries.size());
			summary.uavResourceIDs.reserve(summary.requirementSummaries.size());
			summary.dagAccesses.reserve(summary.requirementSummaries.size() + summary.internalTransitionSummaries.size());

			for (const auto& req : summary.requirementSummaries) {
				mark(
					req.dagResourceIndex,
					req.resourceID,
					req.isWrite ? AccessKind::Write : AccessKind::Read,
					req.isUAV);
			}

			for (const auto& transition : summary.internalTransitionSummaries) {
				mark(transition.dagResourceIndex, transition.resourceID, AccessKind::Write, false);
			}

			std::sort(records.begin(), records.end(), [](const AccessRecord& lhs, const AccessRecord& rhs) {
				if (lhs.dagResourceIndex != rhs.dagResourceIndex) {
					return lhs.dagResourceIndex < rhs.dagResourceIndex;
				}
				return lhs.resourceID < rhs.resourceID;
			});

			for (size_t begin = 0; begin < records.size();) {
				size_t end = begin + 1;
				while (end < records.size() && records[end].dagResourceIndex == records[begin].dagResourceIndex) {
					++end;
				}

				bool hasUav = false;
				bool contributesToDag = false;
				bool hasWrite = false;
				for (size_t i = begin; i < end; ++i) {
					hasUav = hasUav || records[i].uav;
					contributesToDag = contributesToDag || records[i].contributesToDag;
					hasWrite = hasWrite || records[i].write;
				}

				summary.touchedResourceIDs.push_back(records[begin].resourceID);
				if (hasUav) {
					summary.uavResourceIDs.push_back(records[begin].resourceID);
				}
				if (contributesToDag) {
					summary.dagAccesses.push_back(NodeAccess{
						.resourceIndex = records[begin].dagResourceIndex,
						.kind = hasWrite ? AccessKind::Write : AccessKind::Read,
					});
				}

				begin = end;
			}
		});
	}
}

void RenderGraph::RebuildSchedulingEquivalentIDCache(std::span<const uint64_t> resourceIDs) {
	BT_ZONE_SCOPE("RenderGraph::RebuildSchedulingEquivalentIDCache");
	{
		BT_ZONE_SCOPE("RenderGraph::RebuildSchedulingEquivalentIDCache::Reset");
		m_schedulingEquivalentIDsCache.clear();
		m_schedulingEquivalentIDFlat.clear();
		if (m_schedulingEquivalentIDRangeByResourceIndex.size() != m_frameSchedulingResourceCount) {
			m_schedulingEquivalentIDRangeByResourceIndex.resize(m_frameSchedulingResourceCount);
		}
		for (size_t i = 0; i < m_frameSchedulingResourceCount; ++i) {
			m_schedulingEquivalentIDRangeByResourceIndex[i] = SchedulingEquivalentIDRange{};
		}
	}

	auto& placedResources = m_compilerState->schedulingPlacedResources;
	{
		BT_ZONE_SCOPE("RenderGraph::RebuildSchedulingEquivalentIDCache::CountAndReserve");
		placedResources.clear();
		placedResources.reserve(m_frameSchedulingResourceIndexEntries.size());
		for (const auto& [resourceID, resourceIndex] : m_frameSchedulingResourceIndexEntries) {
			if (resourceIndex >= m_hasSchedulingPlacementByResourceIndex.size()
				|| !m_hasSchedulingPlacementByResourceIndex[resourceIndex]) {
				continue;
			}
			const auto& placement = m_schedulingPlacementRangeByResourceIndex[resourceIndex];
			placedResources.push_back(CompilerState::SchedulingPlacedResource{
				.poolID = placement.poolID,
				.resourceID = resourceID,
				.startByte = placement.startByte,
				.endByte = placement.endByte,
			});
		}
		if (placedResources.empty()) {
			return;
		}
		std::sort(placedResources.begin(), placedResources.end(), [](const auto& lhs, const auto& rhs) {
			return lhs.poolID != rhs.poolID ? lhs.poolID < rhs.poolID : lhs.resourceID < rhs.resourceID;
		});
		m_schedulingEquivalentIDFlat.reserve(placedResources.size());
	}

	auto buildEquivalentIDsInto = [&](size_t resourceIndex, uint64_t resourceID) {
		const auto* placement = TryGetSchedulingPlacementRangeByResourceIndex(resourceIndex);
		if (!placement) {
			return;
		}

		const uint32_t offset = static_cast<uint32_t>(m_schedulingEquivalentIDFlat.size());
		const auto poolBegin = std::lower_bound(
			placedResources.begin(),
			placedResources.end(),
			placement->poolID,
			[](const auto& candidate, uint64_t poolID) { return candidate.poolID < poolID; });
		const auto poolEnd = std::upper_bound(
			poolBegin,
			placedResources.end(),
			placement->poolID,
			[](uint64_t poolID, const auto& candidate) { return poolID < candidate.poolID; });
		for (auto candidate = poolBegin; candidate != poolEnd; ++candidate) {
			const uint64_t overlapStart = (std::max)(placement->startByte, candidate->startByte);
			const uint64_t overlapEnd = (std::min)(placement->endByte, candidate->endByte);
			if (overlapStart < overlapEnd) {
				if (m_schedulingEquivalentIDFlat.size() == offset
					|| m_schedulingEquivalentIDFlat.back() != candidate->resourceID) {
					m_schedulingEquivalentIDFlat.push_back(candidate->resourceID);
				}
			}
		}

		if (m_schedulingEquivalentIDFlat.size() == offset) {
			m_schedulingEquivalentIDFlat.push_back(resourceID);
		}
		m_schedulingEquivalentIDRangeByResourceIndex[resourceIndex] = SchedulingEquivalentIDRange{
			.offset = offset,
			.count = static_cast<uint32_t>(m_schedulingEquivalentIDFlat.size() - offset),
		};
	};

	{
		BT_ZONE_SCOPE("RenderGraph::RebuildSchedulingEquivalentIDCache::BuildRanges");
		for (uint64_t resourceID : resourceIDs) {
			auto resourceIndex = TryGetFrameSchedulingResourceIndex(resourceID);
			if (!resourceIndex.has_value()
				|| *resourceIndex >= m_schedulingEquivalentIDRangeByResourceIndex.size()
				|| !TryGetSchedulingPlacementRangeByResourceIndex(*resourceIndex)) {
				continue;
			}
			buildEquivalentIDsInto(*resourceIndex, resourceID);
		}
	}
}

std::span<const uint64_t> RenderGraph::GetSchedulingEquivalentIDsCached(uint64_t resourceID) {
	auto resourceIndex = TryGetFrameSchedulingResourceIndex(resourceID);
	if (resourceIndex.has_value() && *resourceIndex < m_schedulingEquivalentIDRangeByResourceIndex.size()) {
		const auto range = m_schedulingEquivalentIDRangeByResourceIndex[*resourceIndex];
		if (range.count != 0 && static_cast<size_t>(range.offset) + range.count <= m_schedulingEquivalentIDFlat.size()) {
			return std::span<const uint64_t>(
				m_schedulingEquivalentIDFlat.data() + range.offset,
				range.count);
		}
	}

	if (!TryGetSchedulingPlacementRange(resourceID)) {
		thread_local std::array<uint64_t, 1> identityEquivalentIDs;
		identityEquivalentIDs[0] = resourceID;
		return std::span<const uint64_t>(identityEquivalentIDs.data(), identityEquivalentIDs.size());
	}

	thread_local std::vector<uint64_t> fallbackEquivalentIDs;
	if (resourceIndex.has_value()) {
		if (m_schedulingEquivalentIDRangeByResourceIndex.size() <= *resourceIndex) {
			m_schedulingEquivalentIDRangeByResourceIndex.resize(*resourceIndex + 1);
		}
		fallbackEquivalentIDs = BuildSchedulingEquivalentIDs(resourceID);
		const uint32_t offset = static_cast<uint32_t>(m_schedulingEquivalentIDFlat.size());
		m_schedulingEquivalentIDFlat.insert(
			m_schedulingEquivalentIDFlat.end(),
			fallbackEquivalentIDs.begin(),
			fallbackEquivalentIDs.end());
		m_schedulingEquivalentIDRangeByResourceIndex[*resourceIndex] = SchedulingEquivalentIDRange{
			.offset = offset,
			.count = static_cast<uint32_t>(fallbackEquivalentIDs.size()),
		};
		return std::span<const uint64_t>(
			m_schedulingEquivalentIDFlat.data() + offset,
			fallbackEquivalentIDs.size());
	}

	fallbackEquivalentIDs = BuildSchedulingEquivalentIDs(resourceID);
	return std::span<const uint64_t>(fallbackEquivalentIDs.data(), fallbackEquivalentIDs.size());
}

void RenderGraph::CompileFrame(rhi::Device device, uint8_t frameIndex, const IHostExecutionData* hostData) {
	BT_ZONE_SCOPE("RenderGraph::CompileFrame");
	BeginCompileProfileFrame(frameIndex);
	auto endCompileProfileFrame = [this](RenderGraph* graph) {
		if (graph) {
			graph->EndCompileProfileFrame();
		}
	};
	std::unique_ptr<RenderGraph, decltype(endCompileProfileFrame)> compileProfileFrameGuard(this, endCompileProfileFrame);
	std::optional<org::profile::ScopedCompileProfileStep> activeCompileProfileStep;
	const bool traceLifecycle = std::getenv("SARP_RG_COMPILE_TRACE") != nullptr;
	auto traceCompileStep = [&](const char* step) {
		activeCompileProfileStep.reset();
		if (traceLifecycle) {
			spdlog::info("RG frame {} compile step: {}", frameIndex, step);
		}
		if (m_compileProfileFrame
			&& std::string_view(step) != "begin"
			&& std::string_view(step) != "complete") {
			activeCompileProfileStep.emplace(step);
		}
	};
	traceCompileStep("begin");

	{
		traceCompileStep("ResetCompileFrameScratch");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::ResetCompileFrameScratch");
		m_schedulingEquivalentIDsCache.clear();
	}
	{
		traceCompileStep("ResetCompileSchedulingScratch");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::ResetCompileSchedulingScratch");
		m_schedulingDecisionTrace.clear();
		m_transitionPlacementCandidates.clear();
		m_transitionPlacementStats = {};
	}
	{
		traceCompileStep("ResetCompileCounters");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::ResetCompileCounters");
		m_frameDeclarationRefreshRequestedCount = 0;
		m_frameDeclarationRefreshEquivalentCount = 0;
	}
	{
		traceCompileStep("ResetCompileAliasingScratch");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::ResetCompileAliasingScratch");
		autoAliasPlannerStats = {};
		autoAliasPreviousMode = autoAliasModeLastFrame;
	}

	// Gather frame work before retained declarations and immediate command recording.
	// Frame hooks are allowed to publish work to retained structural passes (notably
	// streaming uploads).  Gathering here lets those passes consume that work in their
	// normal structural slot instead of requiring a duplicate late-upload frame pass.
	auto& frameExt = m_compilerState->frameExtensions;
	frameExt.clear();
	if (frameExt.capacity() < 16) {
		frameExt.reserve(16);
	}
	{
		traceCompileStep("GatherFrameExtensions");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::GatherFrameExtensions");
		for (auto& ext : m_extensions) {
			if (!ext) continue;
			ext->GatherFramePasses(*this, frameExt);
		}
	}
	BT_PLOT("ORG.FrameExtensions.EmittedPasses", static_cast<int64_t>(frameExt.size()));

	traceCompileStep("RefreshRetainedDeclarations");
	{
		traceCompileStep("RefreshRetainedDeclarations::AnonymousSlotChanges");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::RefreshRetainedDeclarations::AnonymousSlotChanges");
		_registry.TakeAnonymousSlotChanges(m_anonymousSlotChangesThisFrame);
		BT_PLOT("ORG.RefreshRetained.AnonymousSlotChanges", static_cast<int64_t>(m_anonymousSlotChangesThisFrame.size()));
	}
	using AnonymousSlotValidationEntry = RenderGraph::RetainedDeclarationCache::AnonymousSlotValidationEntry;
	size_t resolverSnapshotCheckCount = 0;
	size_t dynamicDeclarationCheckCount = 0;
	size_t dynamicDeclarationChangedCount = 0;
	auto needsRefresh = [&](auto& p) -> bool {
		// Check if any stored resolver's content version has changed
		{
			resolverSnapshotCheckCount += p.resolverSnapshots.size();
			for (const auto& snap : p.resolverSnapshots) {
				uint64_t cv = snap.resolver->GetContentVersion();
				if (cv != 0 && cv != snap.version) {
					return true;
				}
			}
		}

		if (p.declarationCache.requiresStaleHandleValidation) {
			if (!m_anonymousSlotChangesThisFrame.empty()) {
				const auto& staticEntries = p.declarationCache.staleHandleValidationStaticRequirementAnonymousEntries;
				const auto& transitionEntries = p.declarationCache.staleHandleValidationInternalTransitionAnonymousEntries;
				if (!staticEntries.empty()) {
					const bool iterateSmallBySlot = m_anonymousSlotChangesThisFrame.size() < staticEntries.size();
					if (iterateSmallBySlot) {
						for (const uint32_t anonymousSlot : m_anonymousSlotChangesThisFrame) {
							auto it = std::lower_bound(staticEntries.begin(), staticEntries.end(), anonymousSlot,
								[](const AnonymousSlotValidationEntry& entry, uint32_t slot) { return entry.anonymousSlot < slot; });
							if (it == staticEntries.end() || it->anonymousSlot != anonymousSlot) {
								continue;
							}
							if (it->handleIndex >= p.resources.staticResourceRequirements.size()) {
								continue;
							}
							const auto& resourceHandle = p.resources.staticResourceRequirements[it->handleIndex].resourceHandleAndRange.resource;
							if (resourceHandle.IsEphemeral() || _registry.IsValid(resourceHandle)) {
								continue;
							}
							spdlog::warn(
								"RG frame {} forcing retained declaration refresh for pass '{}' due to stale cached {} handle: resourceId={} registryHandleInfo='{}'",
								frameIndex,
								p.name,
								"requirement",
								resourceHandle.GetGlobalResourceID(),
								_registry.DescribeHandle(resourceHandle));
							return true;
						}
					}
					else {
						size_t i = 0;
						size_t j = 0;
						while (i < staticEntries.size() && j < m_anonymousSlotChangesThisFrame.size()) {
							const uint32_t staticSlot = staticEntries[i].anonymousSlot;
							const uint32_t changedSlot = m_anonymousSlotChangesThisFrame[j];
							if (staticSlot == changedSlot) {
								if (staticEntries[i].handleIndex < p.resources.staticResourceRequirements.size()) {
									const auto& resourceHandle = p.resources.staticResourceRequirements[staticEntries[i].handleIndex].resourceHandleAndRange.resource;
									if (!resourceHandle.IsEphemeral() && !_registry.IsValid(resourceHandle)) {
										spdlog::warn(
											"RG frame {} forcing retained declaration refresh for pass '{}' due to stale cached {} handle: resourceId={} registryHandleInfo='{}'",
											frameIndex,
											p.name,
											"requirement",
											resourceHandle.GetGlobalResourceID(),
											_registry.DescribeHandle(resourceHandle));
										return true;
									}
								}
								++i;
								++j;
							}
							else if (staticSlot < changedSlot) {
								++i;
							}
							else {
								++j;
							}
						}
					}
				}

				if (!transitionEntries.empty()) {
					const bool iterateSmallBySlot = m_anonymousSlotChangesThisFrame.size() < transitionEntries.size();
					if (iterateSmallBySlot) {
						for (const uint32_t anonymousSlot : m_anonymousSlotChangesThisFrame) {
							auto it = std::lower_bound(transitionEntries.begin(), transitionEntries.end(), anonymousSlot,
								[](const AnonymousSlotValidationEntry& entry, uint32_t slot) { return entry.anonymousSlot < slot; });
							if (it == transitionEntries.end() || it->anonymousSlot != anonymousSlot) {
								continue;
							}
							if (it->handleIndex >= p.resources.internalTransitions.size()) {
								continue;
							}
							const auto& resourceHandle = p.resources.internalTransitions[it->handleIndex].first.resource;
							if (resourceHandle.IsEphemeral() || _registry.IsValid(resourceHandle)) {
								continue;
							}
							spdlog::warn(
								"RG frame {} forcing retained declaration refresh for pass '{}' due to stale cached {} handle: resourceId={} registryHandleInfo='{}'",
								frameIndex,
								p.name,
								"internal-transition",
								resourceHandle.GetGlobalResourceID(),
								_registry.DescribeHandle(resourceHandle));
							return true;
						}
					}
					else {
						size_t i = 0;
						size_t j = 0;
						while (i < transitionEntries.size() && j < m_anonymousSlotChangesThisFrame.size()) {
							const uint32_t transitionSlot = transitionEntries[i].anonymousSlot;
							const uint32_t changedSlot = m_anonymousSlotChangesThisFrame[j];
							if (transitionSlot == changedSlot) {
								if (transitionEntries[i].handleIndex < p.resources.internalTransitions.size()) {
									const auto& resourceHandle = p.resources.internalTransitions[transitionEntries[i].handleIndex].first.resource;
									if (!resourceHandle.IsEphemeral() && !_registry.IsValid(resourceHandle)) {
										spdlog::warn(
											"RG frame {} forcing retained declaration refresh for pass '{}' due to stale cached {} handle: resourceId={} registryHandleInfo='{}'",
											frameIndex,
											p.name,
											"internal-transition",
											resourceHandle.GetGlobalResourceID(),
											_registry.DescribeHandle(resourceHandle));
										return true;
									}
								}
								++i;
								++j;
							}
							else if (transitionSlot < changedSlot) {
								++i;
							}
							else {
								++j;
							}
						}
					}
				}
			}
		}

		if (!p.declarationCache.dynamicInterface) {
			// if pass doesn't opt-in, assume no change
			return false;
		}

		{
			BT_ZONE_SCOPE("RenderGraph::CompileFrame::RefreshRetainedDeclarations::NeedsRefresh::DeclaredResourcesChanged");
			if (!p.name.empty()) {
				BT_ZONE_TEXT(p.name.data(), p.name.size());
			}
			++dynamicDeclarationCheckCount;
			const bool changed = p.declarationCache.dynamicInterface->DeclaredResourcesChanged();
			dynamicDeclarationChangedCount += changed ? 1ull : 0ull;
			return changed &&
				!p.declarationCache.dynamicInterface->DeclarationsProvidedByImmediateCommands();
		}
		};

	auto& frameExtensionPassNames = m_compilerState->frameExtensionPassNames;
	frameExtensionPassNames.clear();
	auto& refreshNeededMasterIndices = m_compilerState->refreshNeededMasterIndices;
	refreshNeededMasterIndices.clear();
	if (refreshNeededMasterIndices.capacity() < m_retainedDeclarationRefreshCandidateMasterIndices.size()) {
		refreshNeededMasterIndices.reserve(m_retainedDeclarationRefreshCandidateMasterIndices.size());
	}
	size_t refreshCandidateCount = 0;
	size_t refreshNeededCount = 0;
	size_t renderRefreshCount = 0;
	size_t computeRefreshCount = 0;
	size_t copyRefreshCount = 0;
	size_t changedRefreshCount = 0;
	size_t equivalentRefreshCount = 0;
	{
		traceCompileStep("RefreshRetainedDeclarationChecks");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::RefreshRetainedDeclarations::CheckCandidates");
		BT_PLOT("ORG.RefreshRetained.Candidates", static_cast<int64_t>(m_retainedDeclarationRefreshCandidateMasterIndices.size()));
		for (size_t candidateIndex : m_retainedDeclarationRefreshCandidateMasterIndices) {
			++refreshCandidateCount;
			if (candidateIndex >= m_masterPassList.size()) {
				continue;
			}
			auto& pr = m_masterPassList[candidateIndex];
			if (pr.type == PassType::Compute) {
				auto& p = std::get<ComputePassAndResources>(pr.pass);
				if (needsRefresh(p)) {
					++refreshNeededCount;
					refreshNeededMasterIndices.push_back(candidateIndex);
				}
			}
			else if (pr.type == PassType::Render) {
				auto& p = std::get<RenderPassAndResources>(pr.pass);
				if (needsRefresh(p)) {
					++refreshNeededCount;
					refreshNeededMasterIndices.push_back(candidateIndex);
				}
			}
			else if (pr.type == PassType::Copy) {
				auto& p = std::get<CopyPassAndResources>(pr.pass);
				if (needsRefresh(p)) {
					++refreshNeededCount;
					refreshNeededMasterIndices.push_back(candidateIndex);
				}
			}
		}
		BT_PLOT("ORG.RefreshRetained.CandidatesChecked", static_cast<int64_t>(refreshCandidateCount));
		BT_PLOT("ORG.RefreshRetained.RefreshNeeded", static_cast<int64_t>(refreshNeededCount));
		BT_PLOT("ORG.RefreshRetained.ResolverSnapshots", static_cast<int64_t>(resolverSnapshotCheckCount));
		BT_PLOT("ORG.RefreshRetained.DynamicDeclarationChecks", static_cast<int64_t>(dynamicDeclarationCheckCount));
		BT_PLOT("ORG.RefreshRetained.DynamicDeclaredResourcesChanged", static_cast<int64_t>(dynamicDeclarationChangedCount));
	}
	{
		traceCompileStep("RefreshRetainedDeclarationApply");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::RefreshRetainedDeclarations::Apply");
		for (size_t candidateIndex : refreshNeededMasterIndices) {
			if (candidateIndex >= m_masterPassList.size()) {
				continue;
			}
			auto& pr = m_masterPassList[candidateIndex];
			if (pr.type == PassType::Compute) {
				auto& p = std::get<ComputePassAndResources>(pr.pass);
				BT_ZONE_SCOPE("RenderGraph::CompileFrame::RefreshRetainedDeclarations::RefreshComputePass");
				if (!p.name.empty()) {
					BT_ZONE_TEXT(p.name.data(), p.name.size());
				}
				++computeRefreshCount;
				++m_frameDeclarationRefreshRequestedCount;
				const bool declarationActuallyChanged = RefreshRetainedDeclarationsForFrame(p, frameIndex);
				if (!declarationActuallyChanged) {
					++m_frameDeclarationRefreshEquivalentCount;
					++equivalentRefreshCount;
				}
				else {
					++changedRefreshCount;
				}
				if (declarationActuallyChanged && !p.name.empty()) {
				}
			}
			else if (pr.type == PassType::Render) {
				auto& p = std::get<RenderPassAndResources>(pr.pass);
				BT_ZONE_SCOPE("RenderGraph::CompileFrame::RefreshRetainedDeclarations::RefreshRenderPass");
				if (!p.name.empty()) {
					BT_ZONE_TEXT(p.name.data(), p.name.size());
				}
				++renderRefreshCount;
				++m_frameDeclarationRefreshRequestedCount;
				const bool declarationActuallyChanged = RefreshRetainedDeclarationsForFrame(p, frameIndex);
				if (!declarationActuallyChanged) {
					++m_frameDeclarationRefreshEquivalentCount;
					++equivalentRefreshCount;
				}
				else {
					++changedRefreshCount;
				}
				if (declarationActuallyChanged && !p.name.empty()) {
				}
			}
			else if (pr.type == PassType::Copy) {
				auto& p = std::get<CopyPassAndResources>(pr.pass);
				BT_ZONE_SCOPE("RenderGraph::CompileFrame::RefreshRetainedDeclarations::RefreshCopyPass");
				if (!p.name.empty()) {
					BT_ZONE_TEXT(p.name.data(), p.name.size());
				}
					++copyRefreshCount;
					++m_frameDeclarationRefreshRequestedCount;
					const bool declarationActuallyChanged = RefreshRetainedDeclarationsForFrame(p, frameIndex);
					if (!declarationActuallyChanged) {
						++m_frameDeclarationRefreshEquivalentCount;
						++equivalentRefreshCount;
					}
					else {
						++changedRefreshCount;
					}
					if (declarationActuallyChanged && !p.name.empty()) {
					}
			}
		}
		BT_PLOT("ORG.RefreshRetained.RefreshRender", static_cast<int64_t>(renderRefreshCount));
		BT_PLOT("ORG.RefreshRetained.RefreshCompute", static_cast<int64_t>(computeRefreshCount));
		BT_PLOT("ORG.RefreshRetained.RefreshCopy", static_cast<int64_t>(copyRefreshCount));
		BT_PLOT("ORG.RefreshRetained.RefreshChanged", static_cast<int64_t>(changedRefreshCount));
		BT_PLOT("ORG.RefreshRetained.RefreshEquivalent", static_cast<int64_t>(equivalentRefreshCount));
	}
	{
		traceCompileStep("RefreshRetainedDeclarationPrune");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::RefreshRetainedDeclarations::PruneCandidateList");
		m_retainedDeclarationRefreshCandidateMasterIndices.erase(
			std::remove_if(
				m_retainedDeclarationRefreshCandidateMasterIndices.begin(),
				m_retainedDeclarationRefreshCandidateMasterIndices.end(),
				[&](size_t candidateIndex) {
					return candidateIndex >= m_masterPassList.size()
						|| !RetainedDeclarationMayNeedRefresh(m_masterPassList[candidateIndex]);
				}),
			m_retainedDeclarationRefreshCandidateMasterIndices.end());
	}

	{
		traceCompileStep("InitFramePassState");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::InitFramePassState");
		if (!batches.empty()) {
			m_reusablePassBatches.clear();
			m_reusablePassBatches.swap(batches);
		}
		batches.push_back(AcquireReusablePassBatch(m_queueRegistry.SlotCount())); // Dummy batch 0 for pre-first-pass transitions
		if (m_baseFramePassRefs.capacity() < m_masterPassList.size()) {
			m_baseFramePassRefs.reserve(m_masterPassList.size());
		}
		if (m_frameGeneratedPasses.capacity() < m_masterPassList.size()) {
			m_frameGeneratedPasses.reserve(m_masterPassList.size());
		}
		m_baseFramePassRefs.clear();
		m_frameGeneratedPasses.clear();
		m_frameExtensionPasses.clear();
		m_framePassIsFrameExtension.clear();
	}

	ImmediateExecutionContext renderImmediateContext{ device,
		{org::imm::ImmediatePassKind::Render,
		m_immediateDispatch,
		&ResolveByIdThunk,
		&ResolveByPtrThunk,
		this},
		frameIndex,
		hostData };
	ImmediateExecutionContext computeImmediateContext{ device,
		{org::imm::ImmediatePassKind::Compute,
		m_immediateDispatch,
		&ResolveByIdThunk,
		&ResolveByPtrThunk,
		this},
		frameIndex,
		hostData };
	ImmediateExecutionContext copyImmediateContext{ device,
		{org::imm::ImmediatePassKind::Copy,
		m_immediateDispatch,
		&ResolveByIdThunk,
		&ResolveByPtrThunk,
		this},
		frameIndex,
		hostData };
	if (m_compilerState->immediateModePassPointers.size() < m_masterPassList.size()) {
		m_compilerState->immediateModePassPointers.resize(m_masterPassList.size(), nullptr);
		m_compilerState->immediateModeInterfaces.resize(m_masterPassList.size(), nullptr);
	}
	auto getImmediateModeCommands = [this](size_t masterPassIndex, auto* pass) -> IHasImmediateModeCommands* {
		const void* passIdentity = pass;
		if (m_compilerState->immediateModePassPointers[masterPassIndex] != passIdentity) {
			m_compilerState->immediateModePassPointers[masterPassIndex] = passIdentity;
			m_compilerState->immediateModeInterfaces[masterPassIndex] =
				dynamic_cast<IHasImmediateModeCommands*>(pass);
		}
		return m_compilerState->immediateModeInterfaces[masterPassIndex];
	};
	auto prepareImmediateContext = [&](ImmediateExecutionContext& context, auto& pass) -> ImmediateExecutionContext& {
		context.frameIndex = frameIndex;
		context.hostData = hostData;
		org::imm::FrameData recycled{
			std::move(pass.immediateBytecode),
			std::move(pass.resources.frameResourceRequirements),
			{}
		};
		pass.immediateKeepAlive.reset();
		pass.resources.mergedFrameRequirementsDirty = true;
		context.list.Reset(std::move(recycled));
		return context;
	};
	auto prepareFreshImmediateContext = [&](ImmediateExecutionContext& context) -> ImmediateExecutionContext& {
		context.frameIndex = frameIndex;
		context.hostData = hostData;
		context.list.Reset();
		return context;
	};

	traceCompileStep("BuildFramePassList");
	{
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::BuildFramePassList");
		auto appendBaseFramePassRef = [&](AnyPassAndResources& pass) {
			m_baseFramePassRefs.push_back(std::addressof(pass));
		};
		auto appendBaseFramePassMove = [&](AnyPassAndResources&& pass) {
			m_frameGeneratedPasses.emplace_back(std::move(pass));
			m_baseFramePassRefs.push_back(std::addressof(m_frameGeneratedPasses.back()));
		};

		// Record immediate-mode commands + access for each pass and fold into per-frame requirements
		for (size_t masterPassIndex = 0; masterPassIndex < m_masterPassList.size(); ++masterPassIndex) {
		auto& pr = m_masterPassList[masterPassIndex];

		if (pr.type == PassType::Compute) {
			auto& p = std::get<ComputePassAndResources>(pr.pass);
			auto* immediateModeCommands = getImmediateModeCommands(masterPassIndex, p.pass.get());
			if (!immediateModeCommands) {
				p.run = PassRunMask::Retained;
				appendBaseFramePassRef(pr);
				continue;
			}

			// Re-record into the storage retained by this pass from the previous frame.
			auto& c = prepareImmediateContext(computeImmediateContext, p);

			// Record immediate-mode commands
			{
				BT_ZONE_SCOPE("RenderGraph::CompileFrame::RecordImmediateCommands");
				if (!p.name.empty()) {
					BT_ZONE_TEXT(p.name.data(), p.name.size());
				}
				if (traceLifecycle) {
					spdlog::debug("RG frame {} compute pass '{}' RecordImmediateCommands begin", frameIndex, p.name);
				}
				immediateModeCommands->RecordImmediateCommands(c);
				if (traceLifecycle) {
					spdlog::debug("RG frame {} compute pass '{}' RecordImmediateCommands complete", frameIndex, p.name);
				}
			}

			const bool hasRecordedWork = c.list.HasRecordedWork();
			auto immediateFrameData = c.list.Finalize();
			if (!hasRecordedWork) {
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				SetImmediateFrameRequirements(p.resources, std::move(immediateFrameData.requirements));
				p.run = PassRunMask::Retained;
				appendBaseFramePassRef(pr);
				continue;
			}
			// If there is a conflict between retained and immediate requirements, split the pass
			bool conflict = RequirementsConflict(
				p.resources.staticResourceRequirements,
				immediateFrameData.requirements);
			if (conflict) {
				// Create new PassAndResources for the immediate requirements
				ComputePassAndResources immediatePassAndResources;
				immediatePassAndResources.pass = p.pass;
				SetImmediateFrameRequirements(immediatePassAndResources.resources, std::move(immediateFrameData.requirements));
				immediatePassAndResources.resources.preferredQueueKind = p.resources.preferredQueueKind;
				immediatePassAndResources.resources.pinnedQueueSlot = p.resources.pinnedQueueSlot;
				immediatePassAndResources.resources.backendAffinity = p.resources.backendAffinity;
				immediatePassAndResources.immediateBytecode = std::move(immediateFrameData.bytecode);
				immediatePassAndResources.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				immediatePassAndResources.run = PassRunMask::Immediate;
				AnyPassAndResources immediateAnyPassAndResources;
				immediateAnyPassAndResources.type = PassType::Compute;
				immediateAnyPassAndResources.pass = immediatePassAndResources;
				appendBaseFramePassMove(std::move(immediateAnyPassAndResources));
				p.run = PassRunMask::Retained;
				appendBaseFramePassRef(pr); // Retained pass
			}
			else {
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				SetImmediateFrameRequirements(p.resources, std::move(immediateFrameData.requirements));
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
				appendBaseFramePassRef(pr);
			}
		}
		else if (pr.type == PassType::Render) {
			auto& p = std::get<RenderPassAndResources>(pr.pass);
			auto* immediateModeCommands = getImmediateModeCommands(masterPassIndex, p.pass.get());
			if (!immediateModeCommands) {
				p.run = PassRunMask::Retained;
				appendBaseFramePassRef(pr);
				continue;
			}

			auto& c = prepareImmediateContext(renderImmediateContext, p);
			{
				BT_ZONE_SCOPE("RenderGraph::CompileFrame::RecordImmediateCommands");
				if (!p.name.empty()) {
					BT_ZONE_TEXT(p.name.data(), p.name.size());
				}
				if (traceLifecycle) {
					spdlog::debug("RG frame {} render pass '{}' RecordImmediateCommands begin", frameIndex, p.name);
				}
				immediateModeCommands->RecordImmediateCommands(c);
				if (traceLifecycle) {
					spdlog::debug("RG frame {} render pass '{}' RecordImmediateCommands complete", frameIndex, p.name);
				}
			}
			const bool hasRecordedWork = c.list.HasRecordedWork();
			auto immediateFrameData = c.list.Finalize();
			if (!hasRecordedWork) {
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				SetImmediateFrameRequirements(p.resources, std::move(immediateFrameData.requirements));
				p.run = PassRunMask::Retained;
				appendBaseFramePassRef(pr);
				continue;
			}

			bool conflict = RequirementsConflict(
				p.resources.staticResourceRequirements,
				immediateFrameData.requirements);

			if (conflict) {
				// Create new PassAndResources for the immediate requirements
				RenderPassAndResources immediatePassAndResources;
				immediatePassAndResources.pass = p.pass;
				SetImmediateFrameRequirements(immediatePassAndResources.resources, std::move(immediateFrameData.requirements));
				immediatePassAndResources.resources.preferredQueueKind = p.resources.preferredQueueKind;
				immediatePassAndResources.resources.pinnedQueueSlot = p.resources.pinnedQueueSlot;
				immediatePassAndResources.resources.backendAffinity = p.resources.backendAffinity;
				immediatePassAndResources.immediateBytecode = std::move(immediateFrameData.bytecode);
				immediatePassAndResources.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				immediatePassAndResources.run = PassRunMask::Immediate;
				AnyPassAndResources immediateAnyPassAndResources;
				immediateAnyPassAndResources.type = PassType::Render;
				immediateAnyPassAndResources.pass = immediatePassAndResources;
				appendBaseFramePassMove(std::move(immediateAnyPassAndResources));
				p.run = PassRunMask::Retained;
				appendBaseFramePassRef(pr); // Retained pass
			}
			else {
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				SetImmediateFrameRequirements(p.resources, std::move(immediateFrameData.requirements));
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
				appendBaseFramePassRef(pr);
			}
		}
		else if (pr.type == PassType::Copy) {
			auto& p = std::get<CopyPassAndResources>(pr.pass);
			auto* immediateModeCommands = getImmediateModeCommands(masterPassIndex, p.pass.get());
			if (!immediateModeCommands) {
				p.run = PassRunMask::Retained;
				appendBaseFramePassRef(pr);
				continue;
			}

			auto& c = prepareImmediateContext(copyImmediateContext, p);

			{
				BT_ZONE_SCOPE("RenderGraph::CompileFrame::RecordImmediateCommands");
				if (!p.name.empty()) {
					BT_ZONE_TEXT(p.name.data(), p.name.size());
				}
				if (traceLifecycle) {
					spdlog::debug("RG frame {} copy pass '{}' RecordImmediateCommands begin", frameIndex, p.name);
				}
				immediateModeCommands->RecordImmediateCommands(c);
				if (traceLifecycle) {
					spdlog::debug("RG frame {} copy pass '{}' RecordImmediateCommands complete", frameIndex, p.name);
				}
			}
			const bool hasRecordedWork = c.list.HasRecordedWork();
			auto immediateFrameData = c.list.Finalize();
			if (!hasRecordedWork) {
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				SetImmediateFrameRequirements(p.resources, std::move(immediateFrameData.requirements));
				p.run = PassRunMask::Retained;
				appendBaseFramePassRef(pr);
				continue;
			}

			bool conflict = RequirementsConflict(
				p.resources.staticResourceRequirements,
				immediateFrameData.requirements);

			if (conflict) {
				CopyPassAndResources immediatePassAndResources;
				immediatePassAndResources.pass = p.pass;
				SetImmediateFrameRequirements(immediatePassAndResources.resources, std::move(immediateFrameData.requirements));
				immediatePassAndResources.resources.preferredQueueKind = p.resources.preferredQueueKind;
				immediatePassAndResources.resources.pinnedQueueSlot = p.resources.pinnedQueueSlot;
				immediatePassAndResources.resources.backendAffinity = p.resources.backendAffinity;
				immediatePassAndResources.immediateBytecode = std::move(immediateFrameData.bytecode);
				immediatePassAndResources.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				immediatePassAndResources.run = PassRunMask::Immediate;
				AnyPassAndResources immediateAnyPassAndResources;
				immediateAnyPassAndResources.type = PassType::Copy;
				immediateAnyPassAndResources.pass = immediatePassAndResources;
				appendBaseFramePassMove(std::move(immediateAnyPassAndResources));
				p.run = PassRunMask::Retained;
				appendBaseFramePassRef(pr);
			}
			else {
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				SetImmediateFrameRequirements(p.resources, std::move(immediateFrameData.requirements));
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
				appendBaseFramePassRef(pr);
			}
		}
		}
	}

	// Materialize the genuinely ephemeral passes gathered before structural refresh.
	// These are injected into the per-frame pass list (not m_masterPassList) so they do not accumulate.
	// explicit After(anchor) edges (anchorName -> injectedName)
	auto& explicitAfterByName = m_compilerState->frameExplicitAfterByName;
	{
		traceCompileStep("CopyStructuralExplicitEdges");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::CopyStructuralExplicitEdges");
		explicitAfterByName.assign(m_structuralExplicitAfterByName.begin(), m_structuralExplicitAfterByName.end());
		explicitAfterByName.reserve(m_structuralExplicitAfterByName.size() + frameExt.size());
	}

	if (!frameExt.empty()) {
		traceCompileStep("IntegrateFrameExtensions");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::IntegrateFrameExtensions");
		auto recordImmediateCommands = [&](AnyPassAndResources& pr) {
			if (pr.type == PassType::Compute) {
				auto& p = std::get<ComputePassAndResources>(pr.pass);
				auto* immediateModeCommands = dynamic_cast<IHasImmediateModeCommands*>(p.pass.get());
				if (!immediateModeCommands) {
					p.run = PassRunMask::Retained;
					return;
				}
				p.immediateBytecode.clear();
				p.immediateKeepAlive.reset();
				ClearImmediateFrameRequirements(p.resources);

				auto& c = prepareFreshImmediateContext(computeImmediateContext);

				{
					BT_ZONE_SCOPE("RenderGraph::CompileFrame::RecordImmediateCommands");
					if (!p.name.empty()) {
						BT_ZONE_TEXT(p.name.data(), p.name.size());
					}
					if (traceLifecycle) {
						spdlog::debug("RG frame {} compute pass '{}' RecordImmediateCommands begin", frameIndex, p.name);
					}
					immediateModeCommands->RecordImmediateCommands(c);
					if (traceLifecycle) {
						spdlog::debug("RG frame {} compute pass '{}' RecordImmediateCommands complete", frameIndex, p.name);
					}
				}
				if (!c.list.HasRecordedWork()) {
					p.run = PassRunMask::Retained;
					return;
				}
				auto immediateFrameData = c.list.Finalize();
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				SetImmediateFrameRequirements(p.resources, std::move(immediateFrameData.requirements));
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
			}
			else if (pr.type == PassType::Copy) {
				auto& p = std::get<CopyPassAndResources>(pr.pass);
				auto* immediateModeCommands = dynamic_cast<IHasImmediateModeCommands*>(p.pass.get());
				if (!immediateModeCommands) {
					p.run = PassRunMask::Retained;
					return;
				}
				p.immediateBytecode.clear();
				p.immediateKeepAlive.reset();
				ClearImmediateFrameRequirements(p.resources);

				auto& c = prepareFreshImmediateContext(copyImmediateContext);

				{
					BT_ZONE_SCOPE("RenderGraph::CompileFrame::RecordImmediateCommands");
					if (!p.name.empty()) {
						BT_ZONE_TEXT(p.name.data(), p.name.size());
					}
					if (traceLifecycle) {
						spdlog::debug("RG frame {} copy pass '{}' RecordImmediateCommands begin", frameIndex, p.name);
					}
					immediateModeCommands->RecordImmediateCommands(c);
					if (traceLifecycle) {
						spdlog::debug("RG frame {} copy pass '{}' RecordImmediateCommands complete", frameIndex, p.name);
					}
				}
				if (!c.list.HasRecordedWork()) {
					p.run = PassRunMask::Retained;
					return;
				}
				auto immediateFrameData = c.list.Finalize();
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				SetImmediateFrameRequirements(p.resources, std::move(immediateFrameData.requirements));
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
			}
			else {
				auto& p = std::get<RenderPassAndResources>(pr.pass);
				auto* immediateModeCommands = dynamic_cast<IHasImmediateModeCommands*>(p.pass.get());
				if (!immediateModeCommands) {
					p.run = PassRunMask::Retained;
					return;
				}
				p.immediateBytecode.clear();
				p.immediateKeepAlive.reset();
				ClearImmediateFrameRequirements(p.resources);

				auto& c = prepareFreshImmediateContext(renderImmediateContext);

				{
					BT_ZONE_SCOPE("RenderGraph::CompileFrame::RecordImmediateCommands");
					if (!p.name.empty()) {
						BT_ZONE_TEXT(p.name.data(), p.name.size());
					}
					if (traceLifecycle) {
						spdlog::debug("RG frame {} render pass '{}' RecordImmediateCommands begin", frameIndex, p.name);
					}
					immediateModeCommands->RecordImmediateCommands(c);
					if (traceLifecycle) {
						spdlog::debug("RG frame {} render pass '{}' RecordImmediateCommands complete", frameIndex, p.name);
					}
				}
				if (!c.list.HasRecordedWork()) {
					p.run = PassRunMask::Retained;
					return;
				}
				auto immediateFrameData = c.list.Finalize();
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				SetImmediateFrameRequirements(p.resources, std::move(immediateFrameData.requirements));
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
			}
		};

		const auto& baseFramePasses = m_baseFramePassRefs;

		const size_t invalidInsertIndex = (std::numeric_limits<size_t>::max)();
		auto& pendingFrameInserts = m_compilerState->pendingFrameInserts;
		pendingFrameInserts.clear();
		pendingFrameInserts.reserve(frameExt.size());
		auto& slotHeadByIndex = m_compilerState->frameInsertSlotHeads;
		auto& slotTailByIndex = m_compilerState->frameInsertSlotTails;
		slotHeadByIndex.assign(baseFramePasses.size() + 1, invalidInsertIndex);
		slotTailByIndex.assign(baseFramePasses.size() + 1, invalidInsertIndex);
		auto& pendingInsertIndexByName = m_compilerState->pendingInsertIndexByName;
		pendingInsertIndexByName.clear();
		pendingInsertIndexByName.reserve(frameExt.size());
		auto& pendingInsertTailByAnchorName = m_compilerState->pendingInsertTailByAnchorName;
		pendingInsertTailByAnchorName.clear();
		pendingInsertTailByAnchorName.reserve(frameExt.size());
		if (m_frameExtensionPasses.capacity() < frameExt.size()) {
			m_frameExtensionPasses.reserve(frameExt.size());
		}

		auto appendPendingToSlot = [&](size_t pendingIndex, size_t slotIndex) {
			auto& pending = pendingFrameInserts[pendingIndex];
			pending.slotIndex = slotIndex;
			pending.nextInsertIndex = invalidInsertIndex;
			if (slotHeadByIndex[slotIndex] == invalidInsertIndex) {
				slotHeadByIndex[slotIndex] = pendingIndex;
			}
			else {
				pendingFrameInserts[slotTailByIndex[slotIndex]].nextInsertIndex = pendingIndex;
			}
			slotTailByIndex[slotIndex] = pendingIndex;
		};

		auto insertPendingAfter = [&](size_t pendingIndex, size_t previousPendingIndex) {
			auto& pending = pendingFrameInserts[pendingIndex];
			auto& previous = pendingFrameInserts[previousPendingIndex];
			pending.slotIndex = previous.slotIndex;
			pending.nextInsertIndex = previous.nextInsertIndex;
			previous.nextInsertIndex = pendingIndex;
			if (slotTailByIndex[pending.slotIndex] == previousPendingIndex) {
				slotTailByIndex[pending.slotIndex] = pendingIndex;
			}
		};

		{
			traceCompileStep("MaterializeFrameExtensions");
			BT_ZONE_SCOPE("RenderGraph::CompileFrame::MaterializeFrameExtensions");
			for (auto& d : frameExt) {
				if (d.type == PassType::Unknown) continue;
				if (std::holds_alternative<std::monostate>(d.pass)) continue;
				if (d.name.empty()) {
					spdlog::warn("Frame extension emitted a pass with empty name; skipping.");
					continue;
				}

				AnyPassAndResources any = MaterializeExternalPass(d, true, false);
				recordImmediateCommands(any);
				const std::string insertedPassName = any.name;
				if (!insertedPassName.empty()) {
					frameExtensionPassNames.insert(insertedPassName);
				}
				m_frameExtensionPasses.emplace_back(std::move(any));
				const size_t pendingIndex = pendingFrameInserts.size();
				pendingFrameInserts.push_back(CompilerState::PendingFrameInsert{ .pass = std::addressof(m_frameExtensionPasses.back()) });

				std::string_view anchorName;
				bool insertedRelativeToAnchor = false;

				if (d.where.has_value()) {
					for (auto const& a : d.where->after) {
						auto tailIt = pendingInsertTailByAnchorName.find(a);
						if (tailIt != pendingInsertTailByAnchorName.end()) {
							anchorName = a;
							insertPendingAfter(pendingIndex, tailIt->second);
							insertedRelativeToAnchor = true;
							break;
						}

						auto pendingAnchorIt = pendingInsertIndexByName.find(a);
						if (pendingAnchorIt != pendingInsertIndexByName.end()) {
							anchorName = a;
							insertPendingAfter(pendingIndex, pendingAnchorIt->second);
							insertedRelativeToAnchor = true;
							break;
						}

						auto baseAnchorIt = std::find_if(baseFramePasses.begin(), baseFramePasses.end(), [&](const AnyPassAndResources* basePass) {
							return basePass && basePass->name == a;
						});
						if (baseAnchorIt != baseFramePasses.end()) {
							anchorName = a;
							appendPendingToSlot(
								pendingIndex,
								static_cast<size_t>(baseAnchorIt - baseFramePasses.begin()) + 1);
							insertedRelativeToAnchor = true;
							break;
						}
					}
				}

				if (!insertedRelativeToAnchor) {
					appendPendingToSlot(pendingIndex, baseFramePasses.size());
				}

				if (!anchorName.empty()) {
					explicitAfterByName.emplace_back(std::string(anchorName), insertedPassName);
					pendingInsertTailByAnchorName[std::string(anchorName)] = pendingIndex;
				}
				if (d.where.has_value()) {
					for (auto const& b : d.where->before) {
						if (!insertedPassName.empty()) {
							explicitAfterByName.push_back({ insertedPassName, b });
						}
					}
				}

				if (!insertedPassName.empty()) {
					pendingInsertIndexByName[insertedPassName] = pendingIndex;
				}
			}
		}

		{
			traceCompileStep("AssembleFramePassList");
			BT_ZONE_SCOPE("RenderGraph::CompileFrame::AssembleFramePassList");
			if (m_framePasses.capacity() < baseFramePasses.size() + pendingFrameInserts.size()) {
				m_framePasses.reserve(baseFramePasses.size() + pendingFrameInserts.size());
			}
			m_framePasses.clear();
			auto appendSlot = [&](size_t slotIndex) {
				for (size_t pendingIndex = slotHeadByIndex[slotIndex]; pendingIndex != invalidInsertIndex; pendingIndex = pendingFrameInserts[pendingIndex].nextInsertIndex) {
					m_framePasses.push_back(pendingFrameInserts[pendingIndex].pass);
				}
			};

			for (size_t i = 0; i < baseFramePasses.size(); ++i) {
				appendSlot(i);
				if (baseFramePasses[i]) {
					m_framePasses.push_back(baseFramePasses[i]);
				}
			}
			appendSlot(baseFramePasses.size());
		}
	}
	else {
		traceCompileStep("AssembleFramePassList");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::AssembleFramePassList");
		if (m_framePasses.capacity() < m_baseFramePassRefs.size()) {
			m_framePasses.reserve(m_baseFramePassRefs.size());
		}
		m_framePasses.clear();
		for (AnyPassAndResources* pass : m_baseFramePassRefs) {
			if (!pass) {
				continue;
			}
			m_framePasses.push_back(pass);
		}
	}

	{
		traceCompileStep("ClassifyFramePasses");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::ClassifyFramePasses");
		m_framePassIsFrameExtension.assign(m_framePasses.size(), 0);
		for (size_t passIndex = 0; passIndex < m_framePasses.size(); ++passIndex) {
			const auto& passName = m_framePasses[passIndex].name;
			if (!passName.empty()) {
				m_framePassIsFrameExtension[passIndex] = frameExtensionPassNames.contains(passName) ? 1 : 0;
			}
		}
	}

	// Register/refresh pass statistics indices for this frame's concrete pass list.
	// This supports transient passes and per-frame retained/immediate splits.
	if (m_statisticsService) {
		traceCompileStep("RegisterStatistics");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::RegisterStatistics");
		for (size_t i = 0; i < m_framePasses.size(); ++i) {
			auto& any = m_framePasses[i];
			if (any.type == PassType::Render) {
				auto& p = std::get<RenderPassAndResources>(any.pass);
				if (!p.collectStatistics) {
					p.statisticsIndex = -1;
					continue;
				}
				if (p.name.empty()) {
					p.name = "RenderPass#" + std::to_string(i);
				}
				any.name = p.name;
				if (p.statisticsIndex < 0) {
					p.statisticsIndex = static_cast<int>(m_statisticsService->RegisterPass(p.name, p.resources.isGeometryPass, p.techniquePath));
				}
			}
			else if (any.type == PassType::Compute) {
				auto& p = std::get<ComputePassAndResources>(any.pass);
				if (!p.collectStatistics) {
					p.statisticsIndex = -1;
					continue;
				}
				if (p.name.empty()) {
					p.name = "ComputePass#" + std::to_string(i);
				}
				any.name = p.name;
				if (p.statisticsIndex < 0) {
					p.statisticsIndex = static_cast<int>(m_statisticsService->RegisterPass(p.name, false, p.techniquePath));
				}
			}
			else if (any.type == PassType::Copy) {
				auto& p = std::get<CopyPassAndResources>(any.pass);
				if (!p.collectStatistics) {
					p.statisticsIndex = -1;
					continue;
				}
				if (p.name.empty()) {
					p.name = "CopyPass#" + std::to_string(i);
				}
				any.name = p.name;
				if (p.statisticsIndex < 0) {
					p.statisticsIndex = static_cast<int>(m_statisticsService->RegisterPass(p.name, false, p.techniquePath));
				}
			}
		}

		m_statisticsService->SetupQueryHeap();
	}

	// Convert explicit After(anchorName)->(passName) constraints into node-index edges.
	auto& explicitEdges = m_compilerState->explicitEdges;
	{
		traceCompileStep("BuildExplicitEdges");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::BuildExplicitEdges");
		explicitEdges.clear();
		if (explicitEdges.capacity() < explicitAfterByName.size()) {
			explicitEdges.reserve(explicitAfterByName.size());
		}
		if (!explicitAfterByName.empty()) {
			auto& nameToIndex = m_compilerState->explicitPassNameIndices;
			nameToIndex.clear();
			if (nameToIndex.capacity() < m_framePasses.size()) {
				nameToIndex.reserve(m_framePasses.size());
			}
			for (size_t i = 0; i < m_framePasses.size(); ++i) {
				if (!m_framePasses[i].name.empty()) {
					nameToIndex.emplace_back(m_framePasses[i].name, i);
				}
			}
			std::sort(nameToIndex.begin(), nameToIndex.end(), [](const auto& lhs, const auto& rhs) {
				return lhs.first < rhs.first;
			});
			auto findPassIndex = [&](std::string_view name) -> std::optional<size_t> {
				auto it = std::lower_bound(nameToIndex.begin(), nameToIndex.end(), name,
					[](const auto& entry, std::string_view value) { return entry.first < value; });
				if (it == nameToIndex.end() || it->first != name) {
					return std::nullopt;
				}
				return it->second;
			};
			for (auto const& e : explicitAfterByName) {
				auto anchorIndex = findPassIndex(e.first);
				auto passIndex = findPassIndex(e.second);
				if (!anchorIndex || !passIndex) {
					spdlog::warn("Explicit After edge dropped (anchor='{}', pass='{}'): name not found in frame pass list.", e.first, e.second);
					continue;
				}
				explicitEdges.emplace_back(*anchorIndex, *passIndex);
			}
		}
	}

	auto& nodes = m_compilerState->nodes;
	{
		traceCompileStep("RebuildFramePassAccessSummaries");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::RebuildFramePassAccessSummaries");
		RebuildFramePassAccessSummaries();
	}
	std::span<const uint64_t> usedResourceIDs(m_frameDAGResourceIDsByIndex.data(), m_frameDAGResourceIDsByIndex.size());
	{
		traceCompileStep("ApplyIdleDematerializationPolicy");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::ApplyIdleDematerializationPolicy");
		ApplyIdleDematerializationPolicy(usedResourceIDs);
	}
	{
		traceCompileStep("RebuildSchedulingEquivalentIDCacheBeforeAliasing");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::RebuildSchedulingEquivalentIDCacheBeforeAliasing");
		RebuildSchedulingEquivalentIDCache(usedResourceIDs);
	}
	{
		traceCompileStep("RebuildFrameSchedulingResourceIndex");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::RebuildFrameSchedulingResourceIndex");
		RebuildFrameSchedulingResourceIndex(usedResourceIDs);
	}
	{
		traceCompileStep("BuildNodes");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::BuildNodes");
		BuildNodes(*this, nodes);
	}
	{
		traceCompileStep("BuildDependencyGraph");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::BuildDependencyGraph");
		if (!BuildDependencyGraph(nodes, explicitEdges)) {
			auto passNameForNode = [&](size_t nodeIndex) -> std::string_view {
				if (nodeIndex >= nodes.size()) {
					return "<invalid-node>";
				}
				const size_t passIndex = nodes[nodeIndex].passIndex;
				if (passIndex >= m_framePasses.size()) {
					return "<invalid-pass>";
				}
				const auto& name = m_framePasses[passIndex].name;
				return name.empty() ? "<unnamed-pass>" : std::string_view(name);
			};

			std::vector<uint8_t> color(nodes.size(), 0u);
			std::vector<size_t> stack;
			std::vector<size_t> cycle;
			auto dfs = [&](auto&& self, size_t nodeIndex) -> bool {
				color[nodeIndex] = 1u;
				stack.push_back(nodeIndex);
				for (size_t succ : nodes[nodeIndex].out) {
					if (succ >= nodes.size()) {
						continue;
					}
					if (color[succ] == 0u) {
						if (self(self, succ)) {
							return true;
						}
						continue;
					}
					if (color[succ] == 1u) {
						auto it = std::find(stack.begin(), stack.end(), succ);
						if (it != stack.end()) {
							cycle.assign(it, stack.end());
							cycle.push_back(succ);
							return true;
						}
					}
				}
				stack.pop_back();
				color[nodeIndex] = 2u;
				return false;
			};

			for (size_t nodeIndex = 0; nodeIndex < nodes.size() && cycle.empty(); ++nodeIndex) {
				if (color[nodeIndex] == 0u && dfs(dfs, nodeIndex)) {
					break;
				}
			}

			if (!cycle.empty()) {
				std::ostringstream oss;
				for (size_t i = 0; i < cycle.size(); ++i) {
					if (i != 0u) {
						oss << " -> ";
					}
					oss << "[" << cycle[i] << ":" << passNameForNode(cycle[i]) << "]";
				}
				spdlog::error("Render graph dependency cycle path: {}", oss.str());

				for (size_t nodeIndex : cycle) {
					if (nodeIndex >= nodes.size()) {
						continue;
					}
					std::ostringstream pred;
					for (size_t i = 0; i < nodes[nodeIndex].in.size(); ++i) {
						if (i != 0u) {
							pred << ", ";
						}
						pred << nodes[nodeIndex].in[i] << ":" << passNameForNode(nodes[nodeIndex].in[i]);
					}
					std::ostringstream succ;
					for (size_t i = 0; i < nodes[nodeIndex].out.size(); ++i) {
						if (i != 0u) {
							succ << ", ";
						}
						succ << nodes[nodeIndex].out[i] << ":" << passNameForNode(nodes[nodeIndex].out[i]);
					}
					spdlog::error(
						"Render graph cycle node {} pass='{}' passIndex={} originalOrder={} indegree={} queueSlot={} preferredQueue={} preds=[{}] succs=[{}]",
						nodeIndex,
						passNameForNode(nodeIndex),
						nodes[nodeIndex].passIndex,
						nodes[nodeIndex].originalOrder,
						nodes[nodeIndex].indegree,
						nodes[nodeIndex].queueSlot,
						static_cast<uint32_t>(nodes[nodeIndex].preferredQueueKind),
						pred.str(),
						succ.str());
				}
			}
			else {
				spdlog::error("Render graph dependency cycle reconstruction failed; dumping nodes with nonzero indegree.");
				for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
					if (nodes[nodeIndex].indegree == 0u) {
						continue;
					}
					spdlog::error(
						"Render graph blocked node {} pass='{}' passIndex={} originalOrder={} indegree={}",
						nodeIndex,
						passNameForNode(nodeIndex),
						nodes[nodeIndex].passIndex,
						nodes[nodeIndex].originalOrder,
						nodes[nodeIndex].indegree);
				}
			}
			// Cycle detected
			spdlog::error("Render graph contains a dependency cycle! Render graph compilation failed.");
			throw std::runtime_error("Render graph contains a dependency cycle");
		}
	}

	const AutoAliasMode autoAliasMode = m_getAutoAliasMode ? m_getAutoAliasMode() : AutoAliasMode::Off;
	auto hasManualAliasPoolThisFrame = [&]() {
		for (uint64_t resourceID : usedResourceIDs) {
			Resource* resource = nullptr;
			if (auto it = resourcesByID.find(resourceID); it != resourcesByID.end() && it->second) {
				resource = it->second.get();
			}
			else if (auto it = m_transientFrameResourcesByID.find(resourceID); it != m_transientFrameResourcesByID.end() && it->second) {
				resource = it->second.get();
			}
			resource = UnwrapDynamicResource(resource);
			if (auto* texture = dynamic_cast<PixelBuffer*>(resource)) {
				if (texture->GetDescription().allowAlias && texture->GetDescription().aliasingPoolID.has_value()) {
					return true;
				}
			}
			else if (auto* buffer = dynamic_cast<BufferBase*>(resource)) {
				if (buffer->IsAliasingAllowed() && buffer->GetAliasingPoolHint().has_value()) {
					return true;
				}
			}
		}
		return false;
	};
	const bool needsAliasCompile = autoAliasMode != AutoAliasMode::Off || hasManualAliasPoolThisFrame();
	if (needsAliasCompile) {
		std::vector<org::alias::AliasSchedulingNode> aliasNodes;
		aliasNodes.reserve(nodes.size());
		for (const auto& node : nodes) {
			aliasNodes.push_back(org::alias::AliasSchedulingNode{
				.passIndex = node.passIndex,
				.originalOrder = node.originalOrder,
				.topoRank = node.topoRank,
				.indegree = node.indegree,
				.criticality = node.criticality,
				.backendInstance = static_cast<uint8_t>(m_queueRegistry.GetBackendInstance(
					static_cast<QueueSlotIndex>(static_cast<uint8_t>(node.queueSlot)))),
				.out = node.out,
			});
		}

		org::alias::FrameAliasAnalysis* aliasAnalysis = nullptr;
		{
			traceCompileStep("BuildAliasFrameAnalysis");
			BT_ZONE_SCOPE("RenderGraph::CompileFrame::BuildAliasFrameAnalysis");
			aliasAnalysis = &m_aliasingSubsystem.BuildAliasFrameAnalysis(*this, aliasNodes);
			m_frameBackendUseMaskByResourceID.clear();
			for (const auto& info : aliasAnalysis->infoByResourceIndex) {
				if (info.resourceID) m_frameBackendUseMaskByResourceID[info.resourceID] = info.backendUseMask;
			}
			for (auto& info : aliasAnalysis->infoByResourceIndex) {
				if (std::popcount(info.backendUseMask) < 2) continue;
				auto resource = GetResourceByID(info.resourceID);
				rhi::ResourceDesc desc{};
				if (!resource || !resource->TryGetRHIResourceDesc(desc)) continue;
				for (const auto& backendDevice : m_backendDevices) {
					if ((info.backendUseMask & (uint64_t{1} << static_cast<uint8_t>(backendDevice.id))) == 0) continue;
					rhi::ResourceAllocationInfo requirements{};
					backendDevice.device.GetResourceAllocationInfo(&desc, 1, &requirements);
					info.sizeBytes = (std::max)(info.sizeBytes, requirements.sizeInBytes);
					info.alignment = (std::max)(info.alignment, requirements.alignment);
				}
			}
		}
		{
			traceCompileStep("AutoAssignAliasingPoolsFromAnalysis");
			BT_ZONE_SCOPE("RenderGraph::CompileFrame::AutoAssignAliasingPoolsFromAnalysis");
			m_aliasingSubsystem.AutoAssignAliasingPoolsFromAnalysis(*this, *aliasAnalysis);
			for (auto& info : aliasAnalysis->infoByResourceIndex) {
				if (!info.hasFinalPool) continue;
				boost::hash_combine(info.finalPoolID, info.backendUseMask);
				boost::hash_combine(info.finalPoolID, info.resourceClass);
			}
		}
		{
			traceCompileStep("BuildAliasPlanFromAnalysis");
			BT_ZONE_SCOPE("RenderGraph::CompileFrame::BuildAliasPlanFromAnalysis");
			m_aliasingSubsystem.BuildAliasPlanFromAnalysis(*this, *aliasAnalysis);
		}
		MaterializeMultiBackendRepresentations();
		{
			traceCompileStep("AddCurrentFrameAliasSchedulingEdges");
			BT_ZONE_SCOPE("RenderGraph::CompileFrame::AddCurrentFrameAliasSchedulingEdges");
			if (!AddCurrentFrameAliasSchedulingEdges(nodes)) {
				spdlog::error("Render graph alias scheduling introduced a dependency cycle! Render graph compilation failed.");
				throw std::runtime_error("Render graph alias scheduling introduced a dependency cycle");
			}
		}
	}
	else {
		traceCompileStep("SkipAliasCompile");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::SkipAliasCompile");
		aliasMaterializeOptionsByID.clear();
		m_aliasMaterializeOptionsByResourceIndex.clear();
		m_aliasMaterializeResourceIDs.clear();
		aliasActivationPending.clear();
		aliasPlacementPoolByID.clear();
		aliasPlacementRangesByID.clear();
		schedulingPlacementRangesByID.clear();
		autoAliasPoolByID.clear();
		autoAliasExclusionReasonByID.clear();
		autoAliasExclusionReasonSummary.clear();
		autoAliasExcludedResources.clear();
		m_aliasPlacementRangeByResourceIndex.assign(m_frameSchedulingResourceCount, org::alias::AliasPlacementRange{});
		m_hasAliasPlacementByResourceIndex.assign(m_frameSchedulingResourceCount, 0);
		m_schedulingPlacementRangeByResourceIndex.assign(m_frameSchedulingResourceCount, org::alias::AliasPlacementRange{});
		m_hasSchedulingPlacementByResourceIndex.assign(m_frameSchedulingResourceCount, 0);
		m_aliasActivationPendingByResourceIndex.assign(m_frameSchedulingResourceCount, org::alias::AliasActivationReason::None);
	}
	{
		traceCompileStep("RebuildSchedulingEquivalentIDCache");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::RebuildSchedulingEquivalentIDCache");
		RebuildSchedulingEquivalentIDCache(usedResourceIDs);
	}
	{
		traceCompileStep("RebuildEquivalentResourceIndicesByResourceIndex");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::RebuildEquivalentResourceIndicesByResourceIndex");
		RebuildEquivalentResourceIndicesByResourceIndex();
		ResetFrameQueueBatchHistoryTables();
	}
	{
		traceCompileStep("RebuildFramePassSchedulingSummaries");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::RebuildFramePassSchedulingSummaries");
		RebuildFramePassSchedulingSummaries();
	}

	{
		traceCompileStep("PlanActiveQueueSlots");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::PlanActiveQueueSlots");
		PlanActiveQueueSlots(*this, m_framePasses, nodes);
	}
	{
		traceCompileStep("AssignQueueSlots");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::AssignQueueSlots");
		m_assignedQueueSlotsByFramePass.resize(m_framePasses.size());
		for (auto& node : nodes) {
			size_t defaultSlot = node.queueSlot;
			for (size_t compatibleSlot : node.compatibleQueueSlots) {
				if (compatibleSlot < m_activeQueueSlotsThisFrame.size() && m_activeQueueSlotsThisFrame[compatibleSlot]) {
					defaultSlot = compatibleSlot;
					break;
				}
			}
			node.assignedQueueSlot = defaultSlot;
			if (node.passIndex < m_assignedQueueSlotsByFramePass.size()) {
				m_assignedQueueSlotsByFramePass[node.passIndex] = defaultSlot;
			}
		}
	}
	{
		traceCompileStep("RebuildFrameResourceAccessSummaries");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::RebuildFrameResourceAccessSummaries");
		RebuildFrameResourceAccessSummaries(nodes);
	}
	{
		traceCompileStep("MaterializeUnmaterializedResources");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::MaterializeUnmaterializedResources");
		MaterializeUnmaterializedResources(usedResourceIDs);
	}
	{
		traceCompileStep("RebuildFrameCompileResources");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::RebuildFrameCompileResources");
		RebuildFrameCompileResources();
	}
	{
		traceCompileStep("SnapshotCompiledResourceGenerations");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::SnapshotCompiledResourceGenerations");
		SnapshotCompiledResourceGenerations(usedResourceIDs);
	}

	{
		traceCompileStep("AutoScheduleAndBuildBatches");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::AutoScheduleAndBuildBatches");
		AutoScheduleAndBuildBatches(*this, m_framePasses, nodes);
	}
	{
		traceCompileStep("MaterializeMultiBackendRepresentations");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::MaterializeMultiBackendRepresentations");
		MaterializeMultiBackendRepresentations();
		spdlog::info("RenderGraph frame compile: multi-RHI representation materialization complete");
	}
	{
		traceCompileStep("PlanMultiBackendOwnershipTransfers");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::PlanMultiBackendOwnershipTransfers");
		PlanMultiBackendOwnershipTransfers();
		spdlog::info("RenderGraph frame compile: multi-RHI ownership planning complete");
	}
	{
		traceCompileStep("ApplyAliasQueueSynchronization");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::ApplyAliasQueueSynchronization");
		m_aliasingSubsystem.ApplyAliasQueueSynchronization(*this);
	}
	{
		traceCompileStep("CaptureCompileTrackersForExecution");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::CaptureCompileTrackersForExecution");
		CaptureCompileTrackersForExecution(usedResourceIDs);
	}

	{
		traceCompileStep("PlanCrossFrameQueueWaits");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::PlanCrossFrameQueueWaits");
		for (auto& row : m_hasPendingFrameStartQueueWait) {
			std::fill(row.begin(), row.end(), false);
		}
		for (auto& row : m_pendingFrameStartQueueWaitFenceValue) {
			std::fill(row.begin(), row.end(), UINT64(0));
		}
		uint32_t overlapTriggeredWaitCount = 0;
		uint64_t overlapSampleCurrentResourceId = 0;
		uint64_t overlapSamplePreviousResourceId = 0;
		bool loggedGraphicsWaitOnCopySample = false;
		if (m_crossFrameFirstUseResourceEpochs.size() != m_frameSchedulingResourceCount) {
			m_crossFrameFirstUseResourceEpochs.assign(m_frameSchedulingResourceCount, 0);
			m_crossFrameFirstUseResourceEpoch = 1;
		}
		const uint32_t firstUseEpoch = m_crossFrameFirstUseResourceEpoch++;
		if (m_crossFrameFirstUseResourceEpoch == 0) {
			std::fill(m_crossFrameFirstUseResourceEpochs.begin(), m_crossFrameFirstUseResourceEpochs.end(), 0);
			m_crossFrameFirstUseResourceEpoch = 2;
		}

		auto resourceDebugName = [&](uint64_t resourceID) -> std::string {
			if (auto it = resourcesByID.find(resourceID); it != resourcesByID.end() && it->second) {
				return it->second->GetName();
			}
			if (auto it = m_transientFrameResourcesByID.find(resourceID); it != m_transientFrameResourcesByID.end() && it->second) {
				return it->second->GetName();
			}
			return {};
		};

		auto markCrossFrameWait = [&](size_t dstSlot, size_t srcSlot, uint64_t fenceValue) {
			if (dstSlot == srcSlot) {
				return;
			}
			auto& enabled = m_hasPendingFrameStartQueueWait[dstSlot][srcSlot];
			auto& maxFence = m_pendingFrameStartQueueWaitFenceValue[dstSlot][srcSlot];
			enabled = true;
			maxFence = std::max(maxFence, fenceValue);
		};

		const auto shouldProcessFirstUse = [&](size_t resourceIndex) {
			if (resourceIndex >= m_crossFrameFirstUseResourceEpochs.size()) {
				return false;
			}
			if (resourceIndex >= m_equivalentResourceIndicesByResourceIndex.size()) {
				// Fallback to the non-aliased behavior if cached-equivalent indices are not available yet.
				if (m_crossFrameFirstUseResourceEpochs[resourceIndex] == firstUseEpoch) {
					return false;
				}
				m_crossFrameFirstUseResourceEpochs[resourceIndex] = firstUseEpoch;
				return true;
			}

			const bool hasAliasPlacement = resourceIndex < m_hasAliasPlacementByResourceIndex.size()
				&& m_hasAliasPlacementByResourceIndex[resourceIndex] != 0;
			if (!hasAliasPlacement) {
				if (m_crossFrameFirstUseResourceEpochs[resourceIndex] == firstUseEpoch) {
					return false;
				}
				m_crossFrameFirstUseResourceEpochs[resourceIndex] = firstUseEpoch;
				return true;
			}

			bool hasNewEquivalent = m_crossFrameFirstUseResourceEpochs[resourceIndex] != firstUseEpoch;
			if (!hasNewEquivalent) {
				for (size_t equivalentResourceIndex : m_equivalentResourceIndicesByResourceIndex[resourceIndex]) {
					if (equivalentResourceIndex < m_crossFrameFirstUseResourceEpochs.size()
						&& m_crossFrameFirstUseResourceEpochs[equivalentResourceIndex] != firstUseEpoch) {
						hasNewEquivalent = true;
						break;
					}
				}
			}
			if (!hasNewEquivalent) {
				return false;
			}

			m_crossFrameFirstUseResourceEpochs[resourceIndex] = firstUseEpoch;
			for (size_t equivalentResourceIndex : m_equivalentResourceIndicesByResourceIndex[resourceIndex]) {
				if (equivalentResourceIndex < m_crossFrameFirstUseResourceEpochs.size()) {
					m_crossFrameFirstUseResourceEpochs[equivalentResourceIndex] = firstUseEpoch;
				}
			}
			return true;
		};

		auto accumulateCrossFrameWaitForHandle = [&](
			size_t passQueueSlot,
			size_t resourceIndex,
			std::string_view passName,
			uint64_t originalResourceID,
			bool waitForPriorAccesses) {
			if (resourceIndex >= m_frameSchedulingResourceIDByIndex.size()) {
				return;
			}

			const auto processCandidateResource = [&](size_t candidateResourceIndex, bool hasAliasPlacement) {
				if (candidateResourceIndex >= m_frameSchedulingResourceIDByIndex.size()) {
					return;
				}

				const uint64_t rid = m_frameSchedulingResourceIDByIndex[candidateResourceIndex];
				if (waitForPriorAccesses) {
					if (auto accessIt = m_lastAccessByResourceAcrossFrames.find(rid);
						accessIt != m_lastAccessByResourceAcrossFrames.end()) {
						for (const auto& priorAccess : accessIt->second) {
							markCrossFrameWait(
								passQueueSlot,
								priorAccess.queueSlot,
								priorAccess.fenceValue);
							if (passQueueSlot != priorAccess.queueSlot
								&& m_getRenderGraphBatchTraceEnabled
								&& m_getRenderGraphBatchTraceEnabled()) {
								spdlog::info(
									"RG cross-frame write-after-read wait: pass='{}' resource={} name='{}' dstSlot={} srcSlot={} fence={}",
									passName,
									rid,
									resourceDebugName(rid),
									passQueueSlot,
									priorAccess.queueSlot,
									priorAccess.fenceValue);
							}
						}
					}
				}
				auto it = m_lastProducerByResourceAcrossFrames.find(rid);
				if (it != m_lastProducerByResourceAcrossFrames.end()) {
					markCrossFrameWait(passQueueSlot, it->second.queueSlot, it->second.fenceValue);
					if (!loggedGraphicsWaitOnCopySample && passQueueSlot == 0 && it->second.queueSlot == 2) {
						loggedGraphicsWaitOnCopySample = true;
						spdlog::warn(
							"RG PlanCrossFrameQueueWaits graphics<-copy sample: source=last_producer pass='{}' originalResource={} matchedResource={} resourceName='{}' producerFence={} producerPublishSerial={} anonymous={}",
							passName,
							originalResourceID,
							rid,
							resourceDebugName(rid),
							it->second.fenceValue,
							it->second.publishSerial,
							it->second.anonymous);
					}
				}
				if (!hasAliasPlacement) {
					return;
				}
				if (candidateResourceIndex >= m_aliasPlacementRangeByResourceIndex.size()) {
					return;
				}

				const auto& placement = m_aliasPlacementRangeByResourceIndex[candidateResourceIndex];
				auto itPoolState = persistentAliasPools.find(placement.poolID);
				if (itPoolState == persistentAliasPools.end()) {
					return;
				}

				auto itPrevPool = m_lastAliasPlacementProducersByPoolAcrossFrames.find(placement.poolID);
				if (itPrevPool == m_lastAliasPlacementProducersByPoolAcrossFrames.end()) {
					return;
				}

				const uint64_t curStart = placement.startByte;
				const uint64_t curEnd = placement.endByte;
				const uint64_t curPoolGeneration = itPoolState->second.generation;

				for (const auto& prevPlacementProducer : itPrevPool->second) {
					if (prevPlacementProducer.poolGeneration != curPoolGeneration) {
						continue;
					}

					const uint64_t overlapStart = std::max(curStart, prevPlacementProducer.startByte);
					const uint64_t overlapEnd = std::min(curEnd, prevPlacementProducer.endByte);
					if (overlapStart >= overlapEnd) {
						continue;
					}

					markCrossFrameWait(passQueueSlot, prevPlacementProducer.producer.queueSlot, prevPlacementProducer.producer.fenceValue);
					if (!loggedGraphicsWaitOnCopySample && passQueueSlot == 0 && prevPlacementProducer.producer.queueSlot == 2) {
						loggedGraphicsWaitOnCopySample = true;
						spdlog::warn(
							"RG PlanCrossFrameQueueWaits graphics<-copy sample: source=alias_overlap pass='{}' currentResource={} currentName='{}' previousResource={} previousName='{}' pool={} overlap=[{}, {}) producerFence={} producerPublishSerial={} anonymous={}",
							passName,
							rid,
							resourceDebugName(rid),
							prevPlacementProducer.resourceID,
							resourceDebugName(prevPlacementProducer.resourceID),
							placement.poolID,
							overlapStart,
							overlapEnd,
							prevPlacementProducer.producer.fenceValue,
							prevPlacementProducer.producer.publishSerial,
							prevPlacementProducer.producer.anonymous);
					}
					overlapTriggeredWaitCount++;
					if (overlapSampleCurrentResourceId == 0) {
						overlapSampleCurrentResourceId = rid;
						overlapSamplePreviousResourceId = prevPlacementProducer.resourceID;
					}
				}
			};

			const bool hasAliasPlacement = resourceIndex < m_hasAliasPlacementByResourceIndex.size()
				&& m_hasAliasPlacementByResourceIndex[resourceIndex] != 0;
			processCandidateResource(resourceIndex, hasAliasPlacement);
			if (resourceIndex < m_equivalentResourceIndicesByResourceIndex.size()) {
				const auto& equivalentResourceIndices = m_equivalentResourceIndicesByResourceIndex[resourceIndex];
				for (size_t equivalentResourceIndex : equivalentResourceIndices) {
					const bool candidateHasAliasPlacement = equivalentResourceIndex < m_hasAliasPlacementByResourceIndex.size()
						&& m_hasAliasPlacementByResourceIndex[equivalentResourceIndex] != 0;
					processCandidateResource(equivalentResourceIndex, candidateHasAliasPlacement);
				}
			}
		};

		auto& firstWriteResourceEpochs = m_compilerState->crossFrameFirstWriteResourceEpochs;
		if (firstWriteResourceEpochs.size() != m_frameSchedulingResourceCount) {
			firstWriteResourceEpochs.assign(m_frameSchedulingResourceCount, 0);
			m_compilerState->crossFrameFirstWriteResourceEpoch = 1;
		}
		const uint32_t firstWriteEpoch = m_compilerState->crossFrameFirstWriteResourceEpoch++;
		if (m_compilerState->crossFrameFirstWriteResourceEpoch == 0) {
			std::fill(firstWriteResourceEpochs.begin(), firstWriteResourceEpochs.end(), 0);
			m_compilerState->crossFrameFirstWriteResourceEpoch = 2;
		}
		const auto shouldProcessFirstWrite = [&](size_t resourceIndex) {
			if (resourceIndex >= firstWriteResourceEpochs.size()
				|| firstWriteResourceEpochs[resourceIndex] == firstWriteEpoch) {
				return false;
			}
			firstWriteResourceEpochs[resourceIndex] = firstWriteEpoch;
			if (resourceIndex < m_equivalentResourceIndicesByResourceIndex.size()) {
				for (size_t equivalentResourceIndex : m_equivalentResourceIndicesByResourceIndex[resourceIndex]) {
					if (equivalentResourceIndex < firstWriteResourceEpochs.size()) {
						firstWriteResourceEpochs[equivalentResourceIndex] = firstWriteEpoch;
					}
				}
			}
			return true;
		};

		const size_t queueCount = m_queueRegistry.SlotCount();
		const size_t decisionBucketCount = batches.size() * queueCount;
		auto& decisionCounts = m_compilerState->crossFrameDecisionCounts;
		auto& decisionOffsets = m_compilerState->crossFrameDecisionOffsets;
		auto& decisionWriteOffsets = m_compilerState->crossFrameDecisionWriteOffsets;
		auto& orderedDecisionIndices = m_compilerState->crossFrameOrderedDecisionIndices;
		decisionCounts.assign(decisionBucketCount, 0);
		decisionOffsets.resize(decisionBucketCount + 1);
		decisionWriteOffsets.resize(decisionBucketCount);
		orderedDecisionIndices.resize(m_schedulingDecisionTrace.size());

		for (const auto& decision : m_schedulingDecisionTrace) {
			if (decision.batchIndex < batches.size() && decision.assignedQueueSlot < queueCount) {
				++decisionCounts[decision.batchIndex * queueCount + decision.assignedQueueSlot];
			}
		}
		decisionOffsets[0] = 0;
		for (size_t bucketIndex = 0; bucketIndex < decisionBucketCount; ++bucketIndex) {
			decisionOffsets[bucketIndex + 1] = decisionOffsets[bucketIndex] + decisionCounts[bucketIndex];
			decisionWriteOffsets[bucketIndex] = decisionOffsets[bucketIndex];
		}
		for (uint32_t decisionIndex = 0; decisionIndex < m_schedulingDecisionTrace.size(); ++decisionIndex) {
			const auto& decision = m_schedulingDecisionTrace[decisionIndex];
			if (decision.batchIndex >= batches.size() || decision.assignedQueueSlot >= queueCount) {
				continue;
			}
			const size_t bucketIndex = decision.batchIndex * queueCount + decision.assignedQueueSlot;
			orderedDecisionIndices[decisionWriteOffsets[bucketIndex]++] = decisionIndex;
		}

		for (size_t bucketIndex = 0; bucketIndex < decisionBucketCount; ++bucketIndex) {
			const size_t queueIndex = bucketIndex % queueCount;
			for (uint32_t orderedIndex = decisionOffsets[bucketIndex];
				orderedIndex < decisionOffsets[bucketIndex + 1];
				++orderedIndex) {
				const auto& decision = m_schedulingDecisionTrace[orderedDecisionIndices[orderedIndex]];
				if (decision.passIndex >= m_framePasses.size()
					|| decision.passIndex >= m_framePassSchedulingSummaries.size()) {
					continue;
				}

				const auto& framePass = m_framePasses[decision.passIndex];
				const auto& passSummary = m_framePassSchedulingSummaries[decision.passIndex];
				const std::string_view passName = framePass.name;
				for (const auto& req : passSummary.requirements) {
					if (req.resource.IsEphemeral()) {
						continue;
					}
					if (shouldProcessFirstUse(req.resourceIndex)) {
						accumulateCrossFrameWaitForHandle(
							queueIndex,
							req.resourceIndex,
							passName,
							req.resourceID,
							false);
					}
					if (AccessTypeIsWriteType(req.state.access)
						&& shouldProcessFirstWrite(req.resourceIndex)) {
						accumulateCrossFrameWaitForHandle(
							queueIndex,
							req.resourceIndex,
							passName,
							req.resourceID,
							true);
					}
				}

				std::visit([&](const auto& passEntry) {
					using PassEntry = std::decay_t<decltype(passEntry)>;
					if constexpr (!std::is_same_v<PassEntry, std::monostate>) {
						const size_t transitionCount = (std::min)(
							passEntry.resources.internalTransitions.size(),
							passSummary.internalTransitions.size());
						for (size_t transitionIndex = 0; transitionIndex < transitionCount; ++transitionIndex) {
							const auto& handle =
								passEntry.resources.internalTransitions[transitionIndex].first.resource;
							if (handle.IsEphemeral()) {
								continue;
							}
							const auto& denseTransition = passSummary.internalTransitions[transitionIndex];
							if (shouldProcessFirstUse(denseTransition.resourceIndex)) {
								accumulateCrossFrameWaitForHandle(
									queueIndex,
									denseTransition.resourceIndex,
									passName,
									denseTransition.resourceID,
									false);
							}
						}
					}
				}, framePass.pass);
			}
		}

		//if (overlapTriggeredWaitCount > 0) {
		//	spdlog::info(
		//		"RG cross-frame overlap waits: hits={} sampleCurrentResourceId={} samplePreviousResourceId={}",
		//		overlapTriggeredWaitCount,
		//		overlapSampleCurrentResourceId,
		//		overlapSamplePreviousResourceId);
		//}
	}

	// Insert transitions to loop resources back to their initial states
	//ComputeResourceLoops();

	{
		traceCompileStep("DeduplicateQueueWaits");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::DeduplicateQueueWaits");
		// Cut out repeat waits on the same fence per destination/source queue pair.
		// Fence values are only comparable within a single source queue timeline.
		const size_t slotCount = m_queueRegistry.SlotCount();
		std::vector<std::vector<uint64_t>> lastWaitFenceByDstSrcQueue(
			slotCount,
			std::vector<uint64_t>(slotCount, 0));
		for (auto& batch : batches) {
			const size_t batchQueueCount = batch.QueueCount();
			for (size_t dstIndex = 0; dstIndex < batchQueueCount; ++dstIndex) {
				for (auto phase : { BatchWaitPhase::BeforeTransitions, BatchWaitPhase::BeforeExecution }) {
					for (size_t srcIndex = 0; srcIndex < batchQueueCount; ++srcIndex) {
						if (!batch.HasQueueWait(phase, dstIndex, srcIndex)) {
							continue;
						}

						const auto waitFence = batch.GetQueueWaitFenceValue(phase, dstIndex, srcIndex);
						auto& lastWaitFence = lastWaitFenceByDstSrcQueue[dstIndex][srcIndex];
						if (waitFence <= lastWaitFence) {
							batch.ClearQueueWait(phase, dstIndex, srcIndex);
						}
						else {
							lastWaitFence = waitFence;
						}
					}
				}
			}
		}
	}

	{
		traceCompileStep("PruneUnusedQueueSignals");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::PruneUnusedQueueSignals");
		const size_t slotCount = m_queueRegistry.SlotCount();

		struct SignalOwner {
			UINT64 fenceValue = 0;
			size_t batchIndex = 0;
			BatchSignalPhase phase = BatchSignalPhase::AfterTransitions;
		};
		std::vector<std::vector<SignalOwner>> signalOwnersByQueue(slotCount);
		for (auto& owners : signalOwnersByQueue) {
			owners.reserve(batches.size() * PassBatch::kSignalPhaseCount / (std::max)(slotCount, size_t{ 1 }));
		}
		for (size_t batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
			auto& batch = batches[batchIndex];
			for (size_t queueIndex = 0; queueIndex < batch.QueueCount(); ++queueIndex) {
				for (size_t signalPhaseIndex = 0; signalPhaseIndex < PassBatch::kSignalPhaseCount; ++signalPhaseIndex) {
					const auto signalPhase = static_cast<BatchSignalPhase>(signalPhaseIndex);
					if (!batch.HasQueueSignal(signalPhase, queueIndex)) {
						continue;
					}

					signalOwnersByQueue[queueIndex].push_back(SignalOwner{
						.fenceValue = batch.GetQueueSignalFenceValue(signalPhase, queueIndex),
						.batchIndex = batchIndex,
						.phase = signalPhase,
					});
				}
			}
		}
		for (auto& owners : signalOwnersByQueue) {
			std::sort(owners.begin(), owners.end(), [](const SignalOwner& lhs, const SignalOwner& rhs) {
				return lhs.fenceValue < rhs.fenceValue;
			});
		}

		const auto requiredSignalIndex = [slotCount](size_t batchIndex, size_t phaseIndex, size_t queueIndex) {
			return (batchIndex * PassBatch::kSignalPhaseCount + phaseIndex) * slotCount + queueIndex;
		};
		std::vector<uint8_t> requiredSignals(
			batches.size() * PassBatch::kSignalPhaseCount * slotCount,
			0);

		for (size_t batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
			auto& batch = batches[batchIndex];
			for (size_t waitPhaseIndex = 0; waitPhaseIndex < PassBatch::kWaitPhaseCount; ++waitPhaseIndex) {
				const auto waitPhase = static_cast<BatchWaitPhase>(waitPhaseIndex);
				for (size_t dstQueueIndex = 0; dstQueueIndex < batch.QueueCount(); ++dstQueueIndex) {
					for (size_t srcQueueIndex = 0; srcQueueIndex < batch.QueueCount(); ++srcQueueIndex) {
						if (dstQueueIndex == srcQueueIndex || !batch.HasQueueWait(waitPhase, dstQueueIndex, srcQueueIndex)) {
							continue;
						}

						const UINT64 waitFenceValue = batch.GetQueueWaitFenceValue(waitPhase, dstQueueIndex, srcQueueIndex);
						const auto& owners = signalOwnersByQueue[srcQueueIndex];
						auto itSignalOwner = std::lower_bound(
							owners.begin(), owners.end(), waitFenceValue,
							[](const SignalOwner& owner, UINT64 value) { return owner.fenceValue < value; });
						if (itSignalOwner == owners.end() || itSignalOwner->fenceValue != waitFenceValue) {
							continue;
						}

						requiredSignals[requiredSignalIndex(
							itSignalOwner->batchIndex,
							static_cast<size_t>(itSignalOwner->phase),
							srcQueueIndex)] = 1;
					}
				}
			}
		}

		for (size_t batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
			auto& batch = batches[batchIndex];
			for (size_t queueIndex = 0; queueIndex < batch.QueueCount(); ++queueIndex) {
				for (size_t signalPhaseIndex = 0; signalPhaseIndex < PassBatch::kSignalPhaseCount; ++signalPhaseIndex) {
					const auto signalPhase = static_cast<BatchSignalPhase>(signalPhaseIndex);
					if (!batch.HasQueueSignal(signalPhase, queueIndex)) {
						continue;
					}

					if (!requiredSignals[requiredSignalIndex(batchIndex, signalPhaseIndex, queueIndex)]) {
						batch.ClearQueueSignal(signalPhase, queueIndex);
					}
				}
			}
		}
	}

	if (m_getRenderGraphLightweightCompileSummaryEnabled && m_getRenderGraphLightweightCompileSummaryEnabled()) {
		spdlog::info(
			"RG lightweight compile summary frame={} path=full_compile total_passes={} batches={}",
			static_cast<unsigned int>(frameIndex),
			static_cast<uint64_t>(nodes.size()),
			static_cast<uint64_t>(batches.size() > 0 ? batches.size() - 1 : 0));
	}
	if (m_getRenderGraphCompileDumpEnabled && m_getRenderGraphCompileDumpEnabled()) {
		traceCompileStep("WriteCompiledGraphDebugDump");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::WriteCompiledGraphDebugDump");
		WriteCompiledGraphDebugDump(frameIndex, nodes);
	}
	if (m_getRenderGraphVramDumpEnabled && m_getRenderGraphVramDumpEnabled()) {
		traceCompileStep("WriteVramUsageDebugDump");
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::WriteVramUsageDebugDump");
		WriteVramUsageDebugDump(frameIndex);
	}
	RecordCompileProfileCounters(nodes, usedResourceIDs);
	traceCompileStep("complete");

#if BUILD_TYPE == BUILD_TYPE_DEBUG
	// Sanity checks:
	// 1. No conflicting resource transitions in a batch
	const size_t slotCount = m_queueRegistry.SlotCount();

	auto queueName = [](QueueKind queue) -> const char* {
		switch (queue) {
		case QueueKind::Graphics: return "Graphics";
		case QueueKind::Compute: return "Compute";
		case QueueKind::Copy: return "Copy";
		default: return "Unknown";
		}
		};

	auto phaseName = [](BatchTransitionPhase phase) -> const char* {
		switch (phase) {
		case BatchTransitionPhase::BeforePasses: return "BeforePasses";
		case BatchTransitionPhase::AfterPasses: return "AfterPasses";
		default: return "Unknown";
		}
		};

	auto dumpTransitionsForBatchPhase = [&](size_t batchIndex, const PassBatch& batch, BatchTransitionPhase phase) {
		for (size_t queueIndex = 0; queueIndex < batch.QueueCount(); ++queueIndex) {
			const auto& transitions = batch.Transitions(queueIndex, phase);
			spdlog::error(
				"RG transition dump: batch={} phase={} queue={} count={}",
				batchIndex,
				phaseName(phase),
				queueIndex < static_cast<size_t>(QueueKind::Count) ? queueName(static_cast<QueueKind>(queueIndex)) : "Dynamic",
				transitions.size());

			for (size_t ti = 0; ti < transitions.size(); ++ti) {
				const auto& transition = transitions[ti];
				if (!transition.pResource) {
					spdlog::error(
						"  [{}] resource=<null>",
						ti);
					continue;
				}

				auto prevAccess = rhi::helpers::ResourceAccessMaskToString(transition.prevAccessType);
				auto newAccess = rhi::helpers::ResourceAccessMaskToString(transition.newAccessType);
				std::string mipLower = transition.range.mipLower.ToString();
				std::string mipUpper = transition.range.mipUpper.ToString();
				std::string sliceLower = transition.range.sliceLower.ToString();
				std::string sliceUpper = transition.range.sliceUpper.ToString();

				spdlog::error(
					"  [{}] resource='{}' id={} mip=[{}..{}] slice=[{}..{}] discard={} layout:{}->{} access:{}->{} sync:{}->{}",
					ti,
					transition.pResource->GetName(),
					transition.pResource->GetGlobalResourceID(),
					mipLower,
					mipUpper,
					sliceLower,
					sliceUpper,
					transition.discard,
					rhi::helpers::ResourceLayoutToString(transition.prevLayout),
					rhi::helpers::ResourceLayoutToString(transition.newLayout),
					prevAccess,
					newAccess,
					rhi::helpers::ResourceSyncToString(transition.prevSyncState),
					rhi::helpers::ResourceSyncToString(transition.newSyncState));
			}
		}
		};

	// Validate per transition phase.
	// NOTE: transitions for the same resource can be valid across phases
	// (e.g. BeforePasses transitions into RT state, then AfterPasses transitions back for consumers).
	for (size_t bi = 0; bi < batches.size(); bi++) {
		auto& batch = batches[bi];
		for (size_t phaseIndex = 0; phaseIndex < static_cast<size_t>(BatchTransitionPhase::Count); ++phaseIndex) {
			std::vector<ResourceTransition> phaseTransitions;
			const auto phase = static_cast<BatchTransitionPhase>(phaseIndex);
			for (size_t queueIndex = 0; queueIndex < batch.QueueCount(); ++queueIndex) {
				const auto& transitions = batch.Transitions(queueIndex, phase);
				phaseTransitions.insert(phaseTransitions.end(), transitions.begin(), transitions.end());
			}
			

			// Validate this phase
			TransitionConflict out;
			if (bool ok = ValidateNoConflictingTransitions(phaseTransitions, &out); !ok) {
				const uint32_t conflictMip = static_cast<uint32_t>(out.mip);
				const uint32_t conflictSlice = static_cast<uint32_t>(out.slice);
				spdlog::error(
					"Render graph has conflicting resource transitions in batch {} phase {} ({}) (resource='{}' mip={} slice={})",
					bi,
					phaseIndex,
					phaseName(phase),
					out.resource ? out.resource->GetName() : std::string("<null>"),
					conflictMip,
					conflictSlice);
				dumpTransitionsForBatchPhase(bi, batch, phase);
				throw std::runtime_error("Render graph has conflicting resource transitions!");
			}
		}
	}

	// No out-of-order fence signals on any queue.
	// For each queue, enabled signals must be strictly monotonically increasing
	// in execution order: within a batch (AfterTransitions < AfterExecution < AfterCompletion)
	// and across batches (last signal of batch N < first signal of batch N+1).
	{
		auto signalPhaseName = [](BatchSignalPhase phase) -> const char* {
			switch (phase) {
			case BatchSignalPhase::AfterTransitions: return "AfterTransitions";
			case BatchSignalPhase::AfterExecution:   return "AfterExecution";
			case BatchSignalPhase::AfterCompletion:  return "AfterCompletion";
			default: return "Unknown";
			}
		};

		std::vector<UINT64> lastSignalValue(slotCount, 0);
		std::vector<int> lastSignalBatch(slotCount, -1);
		std::vector<BatchSignalPhase> lastSignalPhase(slotCount);

		for (size_t bi = 0; bi < batches.size(); ++bi) {
			const auto& batch = batches[bi];
			for (size_t qi = 0; qi < batch.QueueCount(); ++qi) {
				for (size_t sp = 0; sp < PassBatch::kSignalPhaseCount; ++sp) {
					const auto phase = static_cast<BatchSignalPhase>(sp);
					if (!batch.HasQueueSignal(phase, qi)) continue;

					UINT64 fenceVal = batch.GetQueueSignalFenceValue(phase, qi);
					if (fenceVal == 0) {
						spdlog::error(
							"Zero-value fence signal on queue {}: "
							"batch {} phase {} has signal enabled with fence value 0. "
							"This will produce an invalid timeline signal at execution time.",
							qi, bi, signalPhaseName(phase));
						throw std::runtime_error("Render graph has zero-value fence signal!");
					}
					if (lastSignalBatch[qi] >= 0 && fenceVal <= lastSignalValue[qi]) {
						spdlog::error(
							"Out-of-order fence signal on queue {}: "
							"batch {} phase {} signals fence={}, but batch {} phase {} already signaled fence={}",
							qi,
							bi, signalPhaseName(phase), fenceVal,
							lastSignalBatch[qi], signalPhaseName(lastSignalPhase[qi]), lastSignalValue[qi]);
						throw std::runtime_error("Render graph has out-of-order fence signals!");
					}
					lastSignalValue[qi] = fenceVal;
					lastSignalBatch[qi] = static_cast<int>(bi);
					lastSignalPhase[qi] = phase;
				}
			}
		}
	}

	// No wait references a fence value that no signal on that source queue will produce.
	{
		// Collect the set of all signaled fence values per queue.
		std::vector<std::unordered_set<UINT64>> signaledValues(slotCount);
		for (const auto& batch : batches) {
			for (size_t qi = 0; qi < batch.QueueCount(); ++qi) {
				for (size_t sp = 0; sp < PassBatch::kSignalPhaseCount; ++sp) {
					if (batch.HasQueueSignal(static_cast<BatchSignalPhase>(sp), qi))
						signaledValues[qi].insert(batch.GetQueueSignalFenceValue(static_cast<BatchSignalPhase>(sp), qi));
				}
			}
		}

		auto waitPhaseName = [](BatchWaitPhase phase) -> const char* {
			switch (phase) {
			case BatchWaitPhase::BeforeTransitions: return "BeforeTransitions";
			case BatchWaitPhase::BeforeExecution:   return "BeforeExecution";
			case BatchWaitPhase::BeforeAfterPasses:  return "BeforeAfterPasses";
			default: return "Unknown";
			}
		};

		for (size_t bi = 0; bi < batches.size(); ++bi) {
			const auto& batch = batches[bi];
			for (size_t wp = 0; wp < PassBatch::kWaitPhaseCount; ++wp) {
				const auto waitPhase = static_cast<BatchWaitPhase>(wp);
				for (size_t dst = 0; dst < batch.QueueCount(); ++dst) {
					for (size_t src = 0; src < batch.QueueCount(); ++src) {
						if (dst == src) continue;
						if (!batch.HasQueueWait(waitPhase, dst, src)) continue;

						UINT64 waitVal = batch.GetQueueWaitFenceValue(waitPhase, dst, src);
						if (signaledValues[src].find(waitVal) == signaledValues[src].end()) {
							spdlog::error(
								"Dangling fence wait: batch {} queue {} waits on queue {} fence={} "
								"at phase {}, but no signal for that value exists",
								bi, dst, src,
								waitVal, waitPhaseName(waitPhase));
							throw std::runtime_error("Render graph has a wait on an unsignaled fence value!");
						}
					}
				}
			}
		}
	}

	// Deadlock detection: check for circular wait-for-signal dependencies within
	// each batch. Within a batch, queues execute concurrently with this timeline:
	//   [BeforeTransitions waits] -> transitions -> AfterTransitions signal
	//   [BeforeExecution waits]   -> passes      -> AfterExecution signal
	//   [BeforeAfterPasses waits] -> post-trans  -> AfterCompletion signal
	// A deadlock occurs when queue A waits for a signal that queue B can only
	// produce after B is blocked waiting for A (or transitively through others).
	//
	// Algorithm: fixed-point iteration over per-queue "progress" levels.
	// progress[q] = maximum signal ordinal q can reach (3 = healthy).
	//   0 = stuck before transitions (no signals produced)
	//   1 = AfterTransitions produced, stuck before execution
	//   2 = AfterExecution produced, stuck before after-passes
	//   3 = AfterCompletion produced (no deadlock)
	{
		constexpr int kWaitBlockLevel[] = { 0, 1, 2 };   // BeforeTransitions, BeforeExecution, BeforeAfterPasses
		constexpr int kSignalRequired[] = { 1, 2, 3 };   // AfterTransitions, AfterExecution, AfterCompletion

		auto signalPhaseName = [](int sp) -> const char* {
			switch (sp) {
			case 0: return "AfterTransitions";
			case 1: return "AfterExecution";
			case 2: return "AfterCompletion";
			default: return "Unknown";
			}
		};

		auto waitPhaseName = [](int wp) -> const char* {
			switch (wp) {
			case 0: return "BeforeTransitions";
			case 1: return "BeforeExecution";
			case 2: return "BeforeAfterPasses";
			default: return "Unknown";
			}
		};

		for (size_t bi = 0; bi < batches.size(); ++bi) {
			const auto& batch = batches[bi];
			const size_t qc = batch.QueueCount();

			std::vector<int> progress(qc, 3);

			bool changed = true;
			while (changed) {
				changed = false;
				for (size_t dst = 0; dst < qc; ++dst) {
					for (size_t wp = 0; wp < PassBatch::kWaitPhaseCount; ++wp) {
						int blockLevel = kWaitBlockLevel[wp];
						if (progress[dst] <= blockLevel) continue; // already stuck here or earlier

						for (size_t src = 0; src < qc; ++src) {
							if (dst == src) continue;
							if (!batch.queueWaitEnabled[wp][dst][src]) continue;

							UINT64 fv = batch.queueWaitFenceValue[wp][dst][src];

							// Determine which signal phase of src this wait targets.
							int requiredSrcProgress = -1;
							for (size_t sp = 0; sp < PassBatch::kSignalPhaseCount; ++sp) {
								if (fv == batch.queueSignalFenceValue[sp][src]) {
									requiredSrcProgress = kSignalRequired[sp];
									break;
								}
							}
							if (requiredSrcProgress < 0) continue; // cross-batch wait, always safe

							if (progress[src] < requiredSrcProgress) {
								progress[dst] = std::min(progress[dst], blockLevel);
								changed = true;
							}
						}
					}
				}
			}

			for (size_t q = 0; q < qc; ++q) {
				if (progress[q] >= 3) continue;

				auto kind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(q)));
				auto inst = m_queueRegistry.GetInstance(static_cast<QueueSlotIndex>(static_cast<uint8_t>(q)));

				spdlog::error("DEADLOCK DETECTED: batch {} queue slot {} ({}:{}) stuck at progress {} "
					"(0=before-transitions, 1=after-transitions, 2=after-execution, 3=healthy)",
					bi, q, queueName(kind), inst, progress[q]);

				// Log the same-batch waits contributing to the cycle
				for (size_t wp = 0; wp < PassBatch::kWaitPhaseCount; ++wp) {
					for (size_t src = 0; src < qc; ++src) {
						if (q == src || !batch.queueWaitEnabled[wp][q][src]) continue;
						UINT64 fv = batch.queueWaitFenceValue[wp][q][src];
						for (size_t sp = 0; sp < PassBatch::kSignalPhaseCount; ++sp) {
							if (fv == batch.queueSignalFenceValue[sp][src]) {
								spdlog::error("  slot {} waits at {} for slot {} {} (fence={}), src progress={}",
									q, waitPhaseName(static_cast<int>(wp)),
									src, signalPhaseName(static_cast<int>(sp)),
									fv, progress[src]);
							}
						}
					}
				}

				throw std::runtime_error("Render graph has a GPU queue deadlock!");
			}
		}
	}

#endif
}


} // namespace org
