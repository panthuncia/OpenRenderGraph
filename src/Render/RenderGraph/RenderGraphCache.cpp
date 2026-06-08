#include "Render/RenderGraph/RenderGraph.h"

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
#include <tracy/Tracy.hpp>

#include "Interfaces/IDynamicDeclaredResources.h"
#include "Resources/DynamicResource.h"
#include "Resources/BackedResource.h"
#include "Resources/ExternalTextureResource.h"

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
	};

	template<class PassResourceData>
	CachedHandleValidationInfo AnalyzeCachedHandleValidation(
		const ResourceRegistry& registry,
		const PassResourceData& resources)
	{
		CachedHandleValidationInfo info{};
		auto inspectHandle = [&](const ResourceRegistry::RegistryHandle& handle) {
			if (handle.IsEphemeral()) {
				info.containsEphemeralOrAnonymousHandles = true;
				return;
			}
			if (registry.IsAnonymous(handle)) {
				info.containsEphemeralOrAnonymousHandles = true;
				info.requiresStaleHandleValidation = true;
			}
		};

		for (const auto& req : resources.staticResourceRequirements) {
			inspectHandle(req.resourceHandleAndRange.resource);
		}
		for (const auto& transition : resources.internalTransitions) {
			inspectHandle(transition.first.resource);
		}
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

	const char* RenderGraphRegionModeToString(rg::runtime::RenderGraphRegionMode mode) noexcept {
		switch (mode) {
		case rg::runtime::RenderGraphRegionMode::Disabled: return "Disabled";
		case rg::runtime::RenderGraphRegionMode::ExtractOnly: return "ExtractOnly";
		case rg::runtime::RenderGraphRegionMode::ValidateOnly: return "ValidateOnly";
		case rg::runtime::RenderGraphRegionMode::ShadowReplay: return "ShadowReplay";
		case rg::runtime::RenderGraphRegionMode::ReplayAuthoritative: return "ReplayAuthoritative";
		default: return "Unknown";
		}
	}

	const char* TransitionPlacementModeToString(rg::runtime::TransitionPlacementMode mode) noexcept {
		switch (mode) {
		case rg::runtime::TransitionPlacementMode::InlineEarlyPlacement: return "InlineEarlyPlacement";
		case rg::runtime::TransitionPlacementMode::CanonicalThenOptimize: return "CanonicalThenOptimize";
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
	ZoneScopedN("RenderGraph::RebuildFramePassAccessSummaries");
	ZoneValue(m_framePasses.size());
	{
		ZoneScopedN("RGPassAccess::Initialize");
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
		if (auto* dynamicResource = dynamic_cast<DynamicResource*>(resource)) {
			return dynamicResource->GetDynamicWrapperGlobalResourceID();
		}
		return handle.GetGlobalResourceID();
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
		if (auto* dynamicResource = dynamic_cast<DynamicResource*>(resource)) {
			const uint64_t stableID = dynamicResource->GetDynamicWrapperGlobalResourceID();
			out.push_back(stableID);
			out.push_back(dynamicResource->GetGlobalResourceID());
			if (auto backing = dynamicResource->GetResource()) {
				out.push_back(backing->GetGlobalResourceID());
			}
			return stableID;
		}
		return handleID;
	};

	{
		ZoneScopedN("RGPassAccess::DenseFullRebuild");
		m_compileScratchResourceIDs.clear();

		{
			ZoneScopedN("RGPassAccess::BuildDenseSummaries");
			ParallelForOptional("RGPassAccessBuildDenseSummaries", m_framePasses.size(), [&](size_t passIndex) {
				const auto& pass = m_framePasses[passIndex];
				auto& summary = m_framePassAccessSummaries[passIndex];
				summary.requirementSummaries.clear();
				summary.internalTransitionSummaries.clear();
				summary.touchedResourceIDs.clear();
				summary.uavResourceIDs.clear();
				summary.dagAccesses.clear();
				summary.type = pass.type;
				summary.preferredQueueKind = DefaultPreferredQueueKind(pass.type);
				summary.queueAssignmentPolicy = DefaultQueueAssignmentPolicy(pass.type);
				summary.pinnedQueueSlot.reset();

				PassView view = GetPassView(pass);
				if (summary.requirementSummaries.capacity() < view.reqs.size()) {
					summary.requirementSummaries.reserve(view.reqs.size());
				}
				const size_t internalTransitionCount = view.internalTransitions ? view.internalTransitions->size() : 0;
				if (summary.internalTransitionSummaries.capacity() < internalTransitionCount) {
					summary.internalTransitionSummaries.reserve(internalTransitionCount);
				}
				const size_t estimatedResourceIDCount = (view.reqs.size() + internalTransitionCount) * 4;
				if (summary.touchedResourceIDs.capacity() < estimatedResourceIDCount) {
					summary.touchedResourceIDs.reserve(estimatedResourceIDCount);
				}

				if (pass.type == PassType::Render) {
					const auto& passResources = std::get<RenderPassAndResources>(pass.pass).resources;
					summary.preferredQueueKind = passResources.preferredQueueKind;
					summary.queueAssignmentPolicy = passResources.queueAssignmentPolicy;
					summary.pinnedQueueSlot = passResources.pinnedQueueSlot;
				}
				else if (pass.type == PassType::Compute) {
					const auto& passResources = std::get<ComputePassAndResources>(pass.pass).resources;
					summary.preferredQueueKind = passResources.preferredQueueKind;
					summary.queueAssignmentPolicy = passResources.queueAssignmentPolicy;
					summary.pinnedQueueSlot = passResources.pinnedQueueSlot;
				}
				else if (pass.type == PassType::Copy) {
					const auto& passResources = std::get<CopyPassAndResources>(pass.pass).resources;
					summary.preferredQueueKind = passResources.preferredQueueKind;
					summary.queueAssignmentPolicy = passResources.queueAssignmentPolicy;
					summary.pinnedQueueSlot = passResources.pinnedQueueSlot;
				}

				for (const auto& req : view.reqs) {
					const auto resource = req.resourceHandleAndRange.resource;
					Resource* resolvedResource = resolveHandleResource(resource);
					const uint64_t resourceID = appendHandleResourceIDsResolved(summary.touchedResourceIDs, resource, resolvedResource);
					summary.requirementSummaries.push_back(FramePassRequirementStaticSummary{
						.resource = resource,
						.resolvedResource = resolvedResource,
						.resourceID = resourceID,
						.dagResourceIndex = UINT32_MAX,
						.range = req.resourceHandleAndRange.range,
						.state = req.state,
						.isUAV = IsUAVState(req.state),
						.isWrite = AccessTypeIsWriteType(req.state.access),
					});
				}

				if (view.internalTransitions) {
					for (const auto& transition : *view.internalTransitions) {
						const auto resource = transition.first.resource;
						Resource* resolvedResource = resolveHandleResource(resource);
						const uint64_t resourceID = appendHandleResourceIDsResolved(summary.touchedResourceIDs, resource, resolvedResource);
						summary.internalTransitionSummaries.push_back(FramePassInternalTransitionStaticSummary{
							.resource = resource,
							.resolvedResource = resolvedResource,
							.resourceID = resourceID,
							.dagResourceIndex = UINT32_MAX,
						});
					}
				}
			});
		}

		{
			ZoneScopedN("RGPassAccess::BuildDenseResourceIndex");
			size_t resourceIDCount = 0;
			{
				ZoneScopedN("RGPassAccess::BuildDenseResourceIndex::CountIDs");
				for (const auto& summary : m_framePassAccessSummaries) {
					resourceIDCount += summary.touchedResourceIDs.size();
				}
			}
			if (m_frameDAGResourceIDsByIndex.capacity() < resourceIDCount) {
				m_frameDAGResourceIDsByIndex.reserve(resourceIDCount);
			}
			{
				ZoneScopedN("RGPassAccess::BuildDenseResourceIndex::BuildFlatUniqueIndex");
				size_t hashCapacity = 1;
				while (hashCapacity < resourceIDCount * 2) {
					hashCapacity <<= 1;
				}
				if (m_frameDAGResourceIndexHashKeys.size() != hashCapacity) {
					m_frameDAGResourceIndexHashKeys.assign(hashCapacity, kFrameDAGResourceIndexEmptyKey);
					m_frameDAGResourceIndexHashValues.resize(hashCapacity);
				}
				else {
					std::fill(
						m_frameDAGResourceIndexHashKeys.begin(),
						m_frameDAGResourceIndexHashKeys.end(),
						kFrameDAGResourceIndexEmptyKey);
				}

				m_frameDAGResourceIDsByIndex.clear();
				const size_t hashMask = hashCapacity - 1;
				for (const auto& summary : m_framePassAccessSummaries) {
					for (uint64_t resourceID : summary.touchedResourceIDs) {
						size_t hashSlot = static_cast<size_t>(MixFrameDAGResourceID(resourceID)) & hashMask;
						for (;;) {
							const uint64_t key = m_frameDAGResourceIndexHashKeys[hashSlot];
							if (key == resourceID) {
								break;
							}
							if (key == kFrameDAGResourceIndexEmptyKey) {
								const uint32_t resourceIndex = static_cast<uint32_t>(m_frameDAGResourceIDsByIndex.size());
								m_frameDAGResourceIndexHashKeys[hashSlot] = resourceID;
								m_frameDAGResourceIndexHashValues[hashSlot] = resourceIndex;
								m_frameDAGResourceIDsByIndex.push_back(resourceID);
								break;
							}
							hashSlot = (hashSlot + 1) & hashMask;
						}
					}
				}
			}
			m_frameDAGResourceCount = m_frameDAGResourceIDsByIndex.size();
			{
				ZoneScopedN("RGPassAccess::BuildDenseResourceIndex::ResetResourcePtrs");
				m_frameDAGResourcePtrByIndex.assign(m_frameDAGResourceCount, nullptr);
				m_frameDAGUnmaterializedResourceIndices.clear();
				if (m_frameDAGUnmaterializedResourceIndices.capacity() < m_frameDAGResourceCount) {
					m_frameDAGUnmaterializedResourceIndices.reserve(m_frameDAGResourceCount);
				}
			}
		}

		auto findDenseDAGResourceIndex = [&](uint64_t resourceID) -> uint32_t {
			if (m_frameDAGResourceIndexHashKeys.empty() || resourceID == kFrameDAGResourceIndexEmptyKey) {
				return UINT32_MAX;
			}

			const size_t hashMask = m_frameDAGResourceIndexHashKeys.size() - 1;
			size_t hashSlot = static_cast<size_t>(MixFrameDAGResourceID(resourceID)) & hashMask;
			for (;;) {
				const uint64_t key = m_frameDAGResourceIndexHashKeys[hashSlot];
				if (key == resourceID) {
					return m_frameDAGResourceIndexHashValues[hashSlot];
				}
				if (key == kFrameDAGResourceIndexEmptyKey) {
					return UINT32_MAX;
				}
				hashSlot = (hashSlot + 1) & hashMask;
			}
		};

		{
			ZoneScopedN("RGPassAccess::AssignDenseDAGIndices");
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
			for (auto& summary : m_framePassAccessSummaries) {
				for (auto& req : summary.requirementSummaries) {
					req.dagResourceIndex = findDenseDAGResourceIndex(req.resourceID);
					captureDenseResourcePtr(req.dagResourceIndex, req.resolvedResource);
				}
				for (auto& transition : summary.internalTransitionSummaries) {
					transition.dagResourceIndex = findDenseDAGResourceIndex(transition.resourceID);
					captureDenseResourcePtr(transition.dagResourceIndex, transition.resolvedResource);
				}
			}
		}

		{
			ZoneScopedN("RGPassAccess::MarkWrittenDAGResourcesDense");
			m_compileScratchResourcesWritten.assign(m_frameDAGResourceCount, uint8_t{ 0 });
			for (const auto& summary : m_framePassAccessSummaries) {
				for (const auto& req : summary.requirementSummaries) {
					if (!req.isWrite) {
						continue;
					}
					if (req.dagResourceIndex != UINT32_MAX && req.dagResourceIndex < m_compileScratchResourcesWritten.size()) {
						m_compileScratchResourcesWritten[req.dagResourceIndex] = 1;
					}
				}
				for (const auto& transition : summary.internalTransitionSummaries) {
					if (transition.dagResourceIndex != UINT32_MAX && transition.dagResourceIndex < m_compileScratchResourcesWritten.size()) {
						m_compileScratchResourcesWritten[transition.dagResourceIndex] = 1;
					}
				}
			}
		}

		{
			ZoneScopedN("RGPassAccess::FinalizeDensePassAccessLists");
			const auto ensureEpochScratch = [&]() {
				if (m_compileScratchAccessEpochs.size() < m_frameDAGResourceCount) {
					m_compileScratchAccessEpochs.resize(m_frameDAGResourceCount, 0);
					m_compileScratchAccessWriteEpochs.resize(m_frameDAGResourceCount, 0);
					m_compileScratchAccessUavEpochs.resize(m_frameDAGResourceCount, 0);
					m_compileScratchAccessDagEpochs.resize(m_frameDAGResourceCount, 0);
				}
				if (m_compileScratchAccessEpoch == std::numeric_limits<uint32_t>::max()) {
					std::fill(m_compileScratchAccessEpochs.begin(), m_compileScratchAccessEpochs.end(), 0);
					std::fill(m_compileScratchAccessWriteEpochs.begin(), m_compileScratchAccessWriteEpochs.end(), 0);
					std::fill(m_compileScratchAccessUavEpochs.begin(), m_compileScratchAccessUavEpochs.end(), 0);
					std::fill(m_compileScratchAccessDagEpochs.begin(), m_compileScratchAccessDagEpochs.end(), 0);
					m_compileScratchAccessEpoch = 1;
				}
			};
			ensureEpochScratch();

			for (auto& summary : m_framePassAccessSummaries) {
				summary.touchedResourceIDs.clear();
				summary.uavResourceIDs.clear();
				summary.dagAccesses.clear();
				m_compileScratchAccessOrder.clear();

				const uint32_t epoch = m_compileScratchAccessEpoch++;
				auto mark = [&](uint32_t dagResourceIndex, AccessKind accessKind, bool isUav) {
					if (dagResourceIndex == UINT32_MAX || dagResourceIndex >= m_frameDAGResourceIDsByIndex.size()) {
						return;
					}

					if (m_compileScratchAccessEpochs[dagResourceIndex] != epoch) {
						m_compileScratchAccessEpochs[dagResourceIndex] = epoch;
						m_compileScratchAccessOrder.push_back(dagResourceIndex);
					}
					if (isUav) {
						m_compileScratchAccessUavEpochs[dagResourceIndex] = epoch;
					}
					if (accessKind == AccessKind::Write) {
						m_compileScratchAccessWriteEpochs[dagResourceIndex] = epoch;
						m_compileScratchAccessDagEpochs[dagResourceIndex] = epoch;
					}
					else if (dagResourceIndex < m_compileScratchResourcesWritten.size()
						&& m_compileScratchResourcesWritten[dagResourceIndex] != 0) {
						m_compileScratchAccessDagEpochs[dagResourceIndex] = epoch;
					}
				};

				for (const auto& req : summary.requirementSummaries) {
					mark(req.dagResourceIndex, req.isWrite ? AccessKind::Write : AccessKind::Read, req.isUAV);
				}
				for (const auto& transition : summary.internalTransitionSummaries) {
					mark(transition.dagResourceIndex, AccessKind::Write, false);
				}

				if (summary.touchedResourceIDs.capacity() < m_compileScratchAccessOrder.size()) {
					summary.touchedResourceIDs.reserve(m_compileScratchAccessOrder.size());
				}
				if (summary.dagAccesses.capacity() < m_compileScratchAccessOrder.size()) {
					summary.dagAccesses.reserve(m_compileScratchAccessOrder.size());
				}

				for (uint32_t dagResourceIndex : m_compileScratchAccessOrder) {
					const uint64_t resourceID = m_frameDAGResourceIDsByIndex[dagResourceIndex];
					summary.touchedResourceIDs.push_back(resourceID);
					if (m_compileScratchAccessUavEpochs[dagResourceIndex] == epoch) {
						summary.uavResourceIDs.push_back(resourceID);
					}
					if (m_compileScratchAccessDagEpochs[dagResourceIndex] == epoch) {
						summary.dagAccesses.push_back(NodeAccess{
							.resourceIndex = dagResourceIndex,
							.kind = m_compileScratchAccessWriteEpochs[dagResourceIndex] == epoch ? AccessKind::Write : AccessKind::Read,
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
		ZoneScopedN("RGPassAccess::BuildKeysAndLoadCacheHits");
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
				ZoneScopedN("RGPassAccess::CheckCacheable");
				workItem.cacheable = passAccessSummaryCacheable(passIndex, m_framePasses[passIndex]);
			}
			if (workItem.cacheable) {
				++cacheablePassCount;
				{
					ZoneScopedN("RGPassAccess::TryPrecomputedStaticKey");
					workItem.cacheKey = tryGetPrecomputedStaticPassAccessKey(m_framePasses[passIndex]);
				}
				if (workItem.cacheKey == 0) {
					ZoneScopedN("RGPassAccess::TryPrecomputedRetainedKey");
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
					ZoneScopedN("RGPassAccess::BuildDynamicAccessKey");
					workItem.cacheKey = buildPassAccessKey(m_framePasses[passIndex]);
				}
			}
			else {
				++nonCacheablePassCount;
				workItem.cacheKey = 0;
			}
			if (workItem.cacheable) {
				ZoneScopedN("RGPassAccess::LookupAccessSummaryCache");
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
		ZoneValue(cacheablePassCount);
		TracyPlot("RGPassAccess.CacheablePasses", static_cast<int64_t>(cacheablePassCount));
		TracyPlot("RGPassAccess.NonCacheablePasses", static_cast<int64_t>(nonCacheablePassCount));
		TracyPlot("RGPassAccess.StaticPrecomputedKeys", static_cast<int64_t>(staticPrecomputedKeyCount));
		TracyPlot("RGPassAccess.RetainedPrecomputedKeys", static_cast<int64_t>(retainedPrecomputedKeyCount));
		TracyPlot("RGPassAccess.DynamicKeyBuilds", static_cast<int64_t>(dynamicKeyBuildCount));
		TracyPlot("RGPassAccess.CacheHits", static_cast<int64_t>(cacheHitCount));
		TracyPlot("RGPassAccess.CacheMisses", static_cast<int64_t>(cacheMissCount));
		TracyPlot("RGPassAccess.CacheHitResourceIDs", static_cast<int64_t>(estimatedCacheHitResourceIDs));
		TracyPlot("RGPassAccess.CacheHitRequirements", static_cast<int64_t>(estimatedCacheHitRequirements));
		TracyPlot("RGPassAccess.CacheHitTransitions", static_cast<int64_t>(estimatedCacheHitTransitions));
	}

	auto buildPassSummary = [&](size_t passIndex) {
		auto& workItem = workItems[passIndex];
		if (workItem.cacheHit) {
			return;
		}
		ZoneScopedN("RGPassAccess::BuildCacheMissSummary");
		ZoneValue(passIndex);

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
		}
		else if (pass.type == PassType::Compute) {
			const auto& passResources = std::get<ComputePassAndResources>(pass.pass).resources;
			summary.preferredQueueKind = passResources.preferredQueueKind;
			summary.queueAssignmentPolicy = passResources.queueAssignmentPolicy;
			summary.pinnedQueueSlot = passResources.pinnedQueueSlot;
		}
		else if (pass.type == PassType::Copy) {
			const auto& passResources = std::get<CopyPassAndResources>(pass.pass).resources;
			summary.preferredQueueKind = passResources.preferredQueueKind;
			summary.queueAssignmentPolicy = passResources.queueAssignmentPolicy;
			summary.pinnedQueueSlot = passResources.pinnedQueueSlot;
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
		ZoneScopedN("RGPassAccess::ParallelBuildCacheMisses");
		ParallelForOptional("RGPrecompilePassAccess", workItems.size(), buildPassSummary);
	}

	{
		ZoneScopedN("RGPassAccess::PublishSummariesAndCacheMisses");
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
		ZoneScopedN("RGPassAccess::MergeUsedResourceIDs");
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
			ZoneScopedN("RGPassAccess::BuildDAGResourceIndex");
			m_frameDAGResourceIndexByID.clear();
			m_frameDAGResourceIDsByIndex = std::move(flattenedResourceIDs);
			m_frameDAGResourceIndexByID.reserve(m_frameDAGResourceIDsByIndex.size());
			for (size_t resourceIndex = 0; resourceIndex < m_frameDAGResourceIDsByIndex.size(); ++resourceIndex) {
				m_frameDAGResourceIndexByID.emplace(m_frameDAGResourceIDsByIndex[resourceIndex], resourceIndex);
			}
			m_frameDAGResourceCount = m_frameDAGResourceIDsByIndex.size();
			m_frameDAGResourcePtrByIndex.assign(m_frameDAGResourceCount, nullptr);
			m_frameDAGUnmaterializedResourceIndices.clear();
			if (m_frameDAGUnmaterializedResourceIndices.capacity() < m_frameDAGResourceCount) {
				m_frameDAGUnmaterializedResourceIndices.reserve(m_frameDAGResourceCount);
			}
		}
	}

	{
		ZoneScopedN("RGPassAccess::AssignDenseDAGIndicesAndPtrs");
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
				auto dagResourceIt = m_frameDAGResourceIndexByID.find(req.resourceID);
				req.dagResourceIndex = dagResourceIt != m_frameDAGResourceIndexByID.end()
					? static_cast<uint32_t>(dagResourceIt->second)
					: UINT32_MAX;
				captureDenseResourcePtr(req.dagResourceIndex, req.resource);
			}
			for (auto& transition : summary.internalTransitionSummaries) {
				auto dagResourceIt = m_frameDAGResourceIndexByID.find(transition.resourceID);
				transition.dagResourceIndex = dagResourceIt != m_frameDAGResourceIndexByID.end()
					? static_cast<uint32_t>(dagResourceIt->second)
					: UINT32_MAX;
				captureDenseResourcePtr(transition.dagResourceIndex, transition.resource);
			}
		}
	}

	std::vector<uint8_t> resourcesWrittenThisFrame(m_frameDAGResourceCount, 0);
	{
		ZoneScopedN("RGPassAccess::MarkWrittenDAGResources");
		for (const auto& summary : m_framePassAccessSummaries) {
			for (const auto& req : summary.requirementSummaries) {
				if (!req.isWrite) {
					continue;
				}
				auto dagResourceIt = m_frameDAGResourceIndexByID.find(req.resourceID);
				if (dagResourceIt != m_frameDAGResourceIndexByID.end() && dagResourceIt->second < resourcesWrittenThisFrame.size()) {
					resourcesWrittenThisFrame[dagResourceIt->second] = 1;
				}
			}
			for (const auto& transition : summary.internalTransitionSummaries) {
				auto dagResourceIt = m_frameDAGResourceIndexByID.find(transition.resourceID);
				if (dagResourceIt != m_frameDAGResourceIndexByID.end() && dagResourceIt->second < resourcesWrittenThisFrame.size()) {
					resourcesWrittenThisFrame[dagResourceIt->second] = 1;
				}
			}
		}
	}

	{
		ZoneScopedN("RGPassAccess::FinalizePerPassAccessLists");
		const auto& dagResourceIndexByID = m_frameDAGResourceIndexByID;
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

			auto mark = [&](uint64_t resourceID, AccessKind accessKind, bool isUav) {
				auto dagResourceIt = dagResourceIndexByID.find(resourceID);
				if (dagResourceIt == dagResourceIndexByID.end()) {
					return;
				}

				const uint32_t dagResourceIndex = static_cast<uint32_t>(dagResourceIt->second);
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
					req.resourceID,
					req.isWrite ? AccessKind::Write : AccessKind::Read,
					req.isUAV);
			}

			for (const auto& transition : summary.internalTransitionSummaries) {
				mark(transition.resourceID, AccessKind::Write, false);
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
	ZoneScopedN("RenderGraph::RebuildSchedulingEquivalentIDCache");
	m_schedulingEquivalentIDsCache.clear();
	m_schedulingEquivalentIDFlat.clear();
	if (m_schedulingEquivalentIDRangeByResourceIndex.size() != m_frameSchedulingResourceCount) {
		m_schedulingEquivalentIDRangeByResourceIndex.resize(m_frameSchedulingResourceCount);
	}
	for (size_t i = 0; i < m_frameSchedulingResourceCount; ++i) {
		m_schedulingEquivalentIDRangeByResourceIndex[i] = SchedulingEquivalentIDRange{};
	}

	size_t schedulingPlacementCount = 0;
	for (uint8_t hasPlacement : m_hasSchedulingPlacementByResourceIndex) {
		schedulingPlacementCount += hasPlacement != 0 ? 1ull : 0ull;
	}
	if (schedulingPlacementCount == 0) {
		return;
	}
	const size_t targetFlatCapacity = schedulingPlacementCount * schedulingPlacementCount;
	if (m_schedulingEquivalentIDFlat.capacity() < targetFlatCapacity) {
		m_schedulingEquivalentIDFlat.reserve(targetFlatCapacity);
	}

	auto buildEquivalentIDsInto = [&](size_t resourceIndex, uint64_t resourceID) {
		const auto* placement = TryGetSchedulingPlacementRangeByResourceIndex(resourceIndex);
		if (!placement) {
			return;
		}

		const uint32_t offset = static_cast<uint32_t>(m_schedulingEquivalentIDFlat.size());
		for (const auto& [candidateID, candidateIndex] : m_frameSchedulingResourceIndexEntries) {
			const auto* otherPlacement = TryGetSchedulingPlacementRangeByResourceIndex(candidateIndex);
			if (!otherPlacement || otherPlacement->poolID != placement->poolID) {
				continue;
			}

			const uint64_t overlapStart = (std::max)(placement->startByte, otherPlacement->startByte);
			const uint64_t overlapEnd = (std::min)(placement->endByte, otherPlacement->endByte);
			if (overlapStart < overlapEnd) {
				m_schedulingEquivalentIDFlat.push_back(candidateID);
			}
		}

		if (m_schedulingEquivalentIDFlat.size() == offset) {
			m_schedulingEquivalentIDFlat.push_back(resourceID);
		}
		auto beginIt = m_schedulingEquivalentIDFlat.begin() + static_cast<std::ptrdiff_t>(offset);
		auto endIt = m_schedulingEquivalentIDFlat.end();
		std::sort(beginIt, endIt);
		const auto uniqueEnd = std::unique(beginIt, endIt);
		m_schedulingEquivalentIDFlat.erase(uniqueEnd, endIt);
		m_schedulingEquivalentIDRangeByResourceIndex[resourceIndex] = SchedulingEquivalentIDRange{
			.offset = offset,
			.count = static_cast<uint32_t>(m_schedulingEquivalentIDFlat.size() - offset),
		};
	};

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

void RenderGraph::ExtractScheduleRegionsFromAuthoritativeCompile(
	const std::vector<Node>& nodes,
	const RenderGraph::FramePassList& framePasses,
	const std::vector<PassBatch>& compiledBatches,
	std::vector<ScheduledRegion>& outRegions,
	RegionCacheStats& outStats,
	std::vector<std::string>& outCandidateDiagnostics) const
{
	const TraceScanRange fullTraceRange{
		.firstTraceIndex = 0,
		.lastTraceIndex = m_schedulingDecisionTrace.empty()
			? 0u
			: static_cast<uint32_t>(m_schedulingDecisionTrace.size() - 1),
	};
	ExtractScheduleRegionsFromAuthoritativeCompile(
		nodes,
		framePasses,
		compiledBatches,
		m_schedulingDecisionTrace.empty() ? std::span<const TraceScanRange>{} : std::span<const TraceScanRange>(&fullTraceRange, 1),
		outRegions,
		outStats,
		outCandidateDiagnostics);
}

void RenderGraph::ExtractScheduleRegionsFromAuthoritativeCompile(
	const std::vector<Node>& nodes,
	const RenderGraph::FramePassList& framePasses,
	const std::vector<PassBatch>& compiledBatches,
	std::span<const TraceScanRange> traceRanges,
	std::vector<ScheduledRegion>& outRegions,
	RegionCacheStats& outStats,
	std::vector<std::string>& outCandidateDiagnostics) const
{
	ZoneScopedN("RenderGraph::ExtractScheduleRegionsFromAuthoritativeCompile");
	outRegions.clear();
	outStats = {};
	outCandidateDiagnostics.clear();

	const uint32_t minPassCount = m_getRenderGraphRegionMinPassCount
		? std::max(1u, m_getRenderGraphRegionMinPassCount())
		: 4u;
	const uint32_t requestedMaxPassCount = m_getRenderGraphRegionMaxPassCount
		? m_getRenderGraphRegionMaxPassCount()
		: 0u;
	const uint32_t maxPassCount = requestedMaxPassCount == 0u
		? 0u
		: std::max(minPassCount, requestedMaxPassCount);

	std::unordered_map<const void*, size_t> passIndexByPointer;
	passIndexByPointer.reserve(framePasses.size());
	for (size_t passIndex = 0; passIndex < framePasses.size(); ++passIndex) {
		std::visit(
			[&](const auto& passEntry) {
				using T = std::decay_t<decltype(passEntry)>;
				if constexpr (!std::is_same_v<T, std::monostate>) {
					passIndexByPointer.emplace(static_cast<const void*>(&passEntry), passIndex);
				}
			},
			framePasses[passIndex].pass);
	}

	auto reject = [&](RegionRejectReason reason) {
		++outStats.rejectedRegionCount;
		++outStats.rejectedByReason[static_cast<size_t>(reason)];
	};

	auto rejectReasonName = [](RegionRejectReason reason) noexcept -> const char* {
		switch (reason) {
		case RegionRejectReason::QueueSlotChange: return "queue_change";
		case RegionRejectReason::PassCountBelowThreshold: return "pass_count_below_threshold";
		case RegionRejectReason::ImmediateWork: return "immediate_work";
		case RegionRejectReason::FrameExtensionPass: return "frame_extension_pass";
		case RegionRejectReason::DeclarationRefreshedThisFrame: return "declaration_refreshed";
		case RegionRejectReason::InteriorIncomingEdge: return "interior_incoming_edge";
		case RegionRejectReason::InteriorOutgoingEdge: return "interior_outgoing_edge";
		case RegionRejectReason::AliasActivation: return "alias_activation";
		case RegionRejectReason::AliasPlacementInstability: return "alias_placement_instability";
		case RegionRejectReason::CrossQueueSync: return "cross_queue_sync";
		case RegionRejectReason::GraphicsFallbackTransition: return "graphics_fallback_transition";
		case RegionRejectReason::UnsupportedSubresourceState: return "unsupported_subresource_state";
		case RegionRejectReason::BatchHazardBoundary: return "batch_hazard_boundary";
		case RegionRejectReason::Count: return "accepted";
		default: return "unknown";
		}
	};

	auto shortPassName = [&](uint32_t passIndex) {
		if (passIndex >= framePasses.size()) {
			return std::string("<invalid>");
		}
		std::string name = framePasses[passIndex].name.empty()
			? std::string("<unnamed>")
			: framePasses[passIndex].name;
		constexpr size_t kMaxPassNameChars = 48;
		if (name.size() > kMaxPassNameChars) {
			name.resize(kMaxPassNameChars - 3);
			name += "...";
		}
		return name;
	};

	auto appendCandidateDiagnostic = [&](
		size_t candidateIndex,
		size_t startTrace,
		size_t endTrace,
		uint16_t queueSlot,
		uint32_t firstPassIndex,
		uint32_t lastPassIndex,
		uint32_t firstBatchIndex,
		uint32_t lastBatchIndex,
		uint32_t requirementCount,
		uint32_t isNewBatchNeededChecks,
		uint32_t boundaryInputEdges,
		uint32_t boundaryOutputEdges,
		uint32_t crossQueueBoundaryInputEdges,
		uint32_t crossQueueBoundaryOutputEdges,
		uint32_t boundarySyncCount,
		uint32_t sameBatchPrefixPasses,
		uint32_t sameBatchSuffixPasses,
		uint32_t sameBatchInterleavedPasses,
		uint32_t crossQueueBoundaryPasses,
		uint32_t crossQueueTransitions,
		RegionRejectReason reason,
		const std::string& detail) {
		std::ostringstream line;
		line << "#" << candidateIndex
			<< " traces=" << startTrace << "-" << (endTrace > startTrace ? endTrace - 1 : startTrace)
			<< " queue=" << queueSlot
			<< " passes=" << (endTrace - startTrace)
			<< " batches=";
		if (firstBatchIndex == std::numeric_limits<uint32_t>::max()) {
			line << "<none>";
		}
		else {
			line << firstBatchIndex << "-" << lastBatchIndex;
		}
		line << " requirements=" << requirementCount
			<< " is_new_batch_checks=" << isNewBatchNeededChecks
			<< " boundary_inputs=" << boundaryInputEdges
			<< " boundary_outputs=" << boundaryOutputEdges
			<< " cross_queue_boundary_inputs=" << crossQueueBoundaryInputEdges
			<< " cross_queue_boundary_outputs=" << crossQueueBoundaryOutputEdges
			<< " boundary_syncs=" << boundarySyncCount
			<< " same_batch_prefix_passes=" << sameBatchPrefixPasses
			<< " same_batch_suffix_passes=" << sameBatchSuffixPasses
			<< " same_batch_interleaved_passes=" << sameBatchInterleavedPasses
			<< " cross_queue_boundary_passes=" << crossQueueBoundaryPasses
			<< " cross_queue_transitions=" << crossQueueTransitions
			<< " first=\"" << shortPassName(firstPassIndex) << "\""
			<< " last=\"" << shortPassName(lastPassIndex) << "\""
			<< " result=" << rejectReasonName(reason);
		if (!detail.empty()) {
			line << " detail=" << detail;
		}
		outCandidateDiagnostics.push_back(line.str());
	};

	auto passHasImmediateWork = [](const AnyPassAndResources& any) {
		return std::visit(
			[](const auto& passEntry) -> bool {
				using T = std::decay_t<decltype(passEntry)>;
				if constexpr (std::is_same_v<T, std::monostate>) {
					return true;
				}
				else {
					return passEntry.run != PassRunMask::Retained || !passEntry.immediateBytecode.empty();
				}
			},
			any.pass);
	};
	auto replayTemplateResourceID = [](Resource* resource) {
		struct Result {
			uint64_t logicalID = 0;
			uint64_t backingID = 0;
			bool dynamic = false;
		};
		if (!resource) {
			return Result{};
		}
		if (auto* dynamicResource = dynamic_cast<DynamicResource*>(resource)) {
			return Result{
				.logicalID = dynamicResource->GetDynamicWrapperGlobalResourceID(),
				.backingID = dynamicResource->GetGlobalResourceID(),
				.dynamic = true,
			};
		}
		return Result{
			.logicalID = resource->GetGlobalResourceID(),
			.backingID = resource->GetGlobalResourceID(),
			.dynamic = false,
		};
	};

	auto countPassRequirements = [&](size_t passIndex) -> uint32_t {
		if (passIndex >= m_framePassSchedulingSummaries.size()) {
			return 0;
		}
		return static_cast<uint32_t>(m_framePassSchedulingSummaries[passIndex].requirements.size());
	};

	auto passHasAliasActivation = [&](size_t passIndex) -> bool {
		if (passIndex >= m_framePassSchedulingSummaries.size()) {
			return false;
		}

		for (const auto& requirement : m_framePassSchedulingSummaries[passIndex].requirements) {
			if (requirement.resourceIndex < m_frameResourceAccessSummaries.size()
				&& m_frameResourceAccessSummaries[requirement.resourceIndex].hasAliasActivation) {
				return true;
			}
		}
		return false;
	};

	auto passForcesCandidateSplit = [&](size_t passIndex, RegionRejectReason& outReason) -> bool {
		if (passIndex >= framePasses.size()) {
			outReason = RegionRejectReason::ImmediateWork;
			return true;
		}
		if (passHasImmediateWork(framePasses[passIndex])) {
			outReason = RegionRejectReason::ImmediateWork;
			return true;
		}
		if (passIndex < m_framePassIsFrameExtension.size()
			&& m_framePassIsFrameExtension[passIndex] != 0) {
			outReason = RegionRejectReason::FrameExtensionPass;
			return true;
		}
		return false;
	};

	auto countBatchCrossQueueSync = [](const PassBatch& batch) {
		uint32_t count = 0;
		for (size_t dst = 0; dst < batch.QueueCount(); ++dst) {
			for (size_t src = 0; src < batch.QueueCount(); ++src) {
				if (dst == src) {
					continue;
				}
				for (size_t phaseIndex = 0; phaseIndex < PassBatch::kWaitPhaseCount; ++phaseIndex) {
					if (batch.HasQueueWait(static_cast<BatchWaitPhase>(phaseIndex), dst, src)) {
						++count;
					}
				}
			}
		}
		for (size_t queueIndex = 0; queueIndex < batch.QueueCount(); ++queueIndex) {
			for (size_t phaseIndex = 0; phaseIndex < PassBatch::kSignalPhaseCount; ++phaseIndex) {
				if (batch.HasQueueSignal(static_cast<BatchSignalPhase>(phaseIndex), queueIndex)) {
					++count;
				}
			}
		}
		return count;
	};

	auto countBatchTransitions = [](const PassBatch& batch) {
		uint32_t count = 0;
		for (size_t queueIndex = 0; queueIndex < batch.QueueCount(); ++queueIndex) {
			for (size_t phaseIndex = 0; phaseIndex < static_cast<size_t>(BatchTransitionPhase::Count); ++phaseIndex) {
				count += static_cast<uint32_t>(batch.Transitions(queueIndex, static_cast<BatchTransitionPhase>(phaseIndex)).size());
			}
		}
		return count;
	};

	std::vector<size_t> traceIndexByNodeIndex(nodes.size(), std::numeric_limits<size_t>::max());
	std::vector<size_t> traceIndexByPassIndex(framePasses.size(), std::numeric_limits<size_t>::max());
	for (size_t traceIndex = 0; traceIndex < m_schedulingDecisionTrace.size(); ++traceIndex) {
		const auto& trace = m_schedulingDecisionTrace[traceIndex];
		if (trace.nodeIndex < traceIndexByNodeIndex.size()) {
			traceIndexByNodeIndex[trace.nodeIndex] = traceIndex;
		}
		if (trace.passIndex < traceIndexByPassIndex.size()) {
			traceIndexByPassIndex[trace.passIndex] = traceIndex;
		}
	}

	for (const TraceScanRange& scanRange : traceRanges) {
		if (m_schedulingDecisionTrace.empty() || scanRange.firstTraceIndex >= m_schedulingDecisionTrace.size()) {
			continue;
		}
		const size_t rangeBegin = scanRange.firstTraceIndex;
		const size_t rangeEnd = std::min<size_t>(
			static_cast<size_t>(scanRange.lastTraceIndex) + 1,
			m_schedulingDecisionTrace.size());
		size_t start = rangeBegin;
		while (start < rangeEnd) {
		const uint16_t queueSlot = m_schedulingDecisionTrace[start].assignedQueueSlot;
		RegionRejectReason startSplitReason = RegionRejectReason::Count;
		const bool startForcesSplit = passForcesCandidateSplit(m_schedulingDecisionTrace[start].passIndex, startSplitReason);
		size_t end = start + 1;
		if (!startForcesSplit) {
			while (end < rangeEnd
				&& (maxPassCount == 0u || end - start < maxPassCount)
				&& m_schedulingDecisionTrace[end].assignedQueueSlot == queueSlot) {
				RegionRejectReason nextSplitReason = RegionRejectReason::Count;
				if (passForcesCandidateSplit(m_schedulingDecisionTrace[end].passIndex, nextSplitReason)) {
					break;
				}
				++end;
			}
		}

		++outStats.candidateRegionCount;
		const size_t candidateIndex = outStats.candidateRegionCount;
		RegionRejectReason rejectReason = RegionRejectReason::Count;
		std::string rejectDetail;
		if (startForcesSplit) {
			rejectReason = startSplitReason;
			rejectDetail = "pass=\"" + shortPassName(m_schedulingDecisionTrace[start].passIndex) + "\" split_before_region";
		}
		else if (end - start < minPassCount) {
			rejectReason = RegionRejectReason::PassCountBelowThreshold;
			std::ostringstream detail;
			detail << "candidate_passes=" << (end - start) << " min_passes=" << minPassCount;
			rejectDetail = detail.str();
		}

		std::unordered_set<size_t> regionNodeIndices;
		std::unordered_set<size_t> regionPassIndices;
		regionNodeIndices.reserve(end - start);
		regionPassIndices.reserve(end - start);

		uint32_t requirementCount = 0;
		uint32_t isNewBatchNeededChecks = 0;
		uint32_t firstPassIndex = std::numeric_limits<uint32_t>::max();
		uint32_t lastPassIndex = 0;
		uint32_t firstBatchIndex = std::numeric_limits<uint32_t>::max();
		uint32_t lastBatchIndex = 0;
		uint32_t boundaryInputEdges = 0;
		uint32_t boundaryOutputEdges = 0;
		uint32_t crossQueueBoundaryInputEdges = 0;
		uint32_t crossQueueBoundaryOutputEdges = 0;
		uint32_t boundarySyncCount = 0;
		uint32_t sameBatchPrefixPasses = 0;
		uint32_t sameBatchSuffixPasses = 0;
		uint32_t sameBatchInterleavedPasses = 0;
		uint32_t crossQueueBoundaryPasses = 0;
		uint32_t crossQueueTransitions = 0;

		for (size_t traceIndex = start; traceIndex < end; ++traceIndex) {
			const auto& trace = m_schedulingDecisionTrace[traceIndex];
			regionNodeIndices.insert(trace.nodeIndex);
			regionPassIndices.insert(trace.passIndex);
			requirementCount += countPassRequirements(trace.passIndex);
			isNewBatchNeededChecks += trace.isNewBatchNeededChecks;
			firstPassIndex = std::min(firstPassIndex, trace.passIndex);
			lastPassIndex = std::max(lastPassIndex, trace.passIndex);
			firstBatchIndex = std::min(firstBatchIndex, trace.batchIndex);
			lastBatchIndex = std::max(lastBatchIndex, trace.batchIndex);

			if (rejectReason == RegionRejectReason::Count && trace.passIndex < framePasses.size()) {
				if (passHasImmediateWork(framePasses[trace.passIndex])) {
					rejectReason = RegionRejectReason::ImmediateWork;
					rejectDetail = "pass=\"" + shortPassName(trace.passIndex) + "\"";
				}
				else if (trace.passIndex < m_framePassIsFrameExtension.size()
					&& m_framePassIsFrameExtension[trace.passIndex] != 0) {
					rejectReason = RegionRejectReason::FrameExtensionPass;
					rejectDetail = "pass=\"" + shortPassName(trace.passIndex) + "\"";
				}
				else if (trace.passIndex < m_framePassDeclarationRefreshedThisFrame.size()
					&& m_framePassDeclarationRefreshedThisFrame[trace.passIndex] != 0) {
					rejectReason = RegionRejectReason::DeclarationRefreshedThisFrame;
					rejectDetail = "pass=\"" + shortPassName(trace.passIndex) + "\"";
				}
			}
		}

		if (rejectReason == RegionRejectReason::Count) {
			for (size_t traceIndex = start; traceIndex < end; ++traceIndex) {
				const auto& trace = m_schedulingDecisionTrace[traceIndex];
				if (trace.nodeIndex >= nodes.size()) {
					rejectReason = RegionRejectReason::InteriorIncomingEdge;
					rejectDetail = "trace=" + std::to_string(traceIndex) + " node_index_out_of_range";
					break;
				}

				const auto& node = nodes[trace.nodeIndex];
				for (size_t pred : node.in) {
					if (!regionNodeIndices.contains(pred)) {
						if (pred >= traceIndexByNodeIndex.size()
							|| traceIndexByNodeIndex[pred] == std::numeric_limits<size_t>::max()) {
							rejectReason = RegionRejectReason::InteriorIncomingEdge;
							std::ostringstream detail;
							detail << "pass=\"" << shortPassName(trace.passIndex) << "\""
								<< " node=" << trace.nodeIndex
								<< " outside_pred=" << pred
								<< " missing_trace";
							rejectDetail = detail.str();
							break;
						}
						const size_t predTraceIndex = traceIndexByNodeIndex[pred];
						const auto& predTrace = m_schedulingDecisionTrace[predTraceIndex];
						if (predTraceIndex >= start) {
							rejectReason = RegionRejectReason::InteriorIncomingEdge;
							std::ostringstream detail;
							detail << "pass=\"" << shortPassName(trace.passIndex) << "\""
								<< " node=" << trace.nodeIndex
								<< " outside_pred=" << pred
								<< " pred_trace=" << predTraceIndex
								<< " pred_queue=" << predTrace.assignedQueueSlot
								<< " expected_before=" << start
								<< " region_queue=" << queueSlot;
							rejectDetail = detail.str();
							break;
						}
						++boundaryInputEdges;
						if (predTrace.assignedQueueSlot != queueSlot) {
							++crossQueueBoundaryInputEdges;
						}
					}
				}
				if (rejectReason != RegionRejectReason::Count) {
					break;
				}
				for (size_t succ : node.out) {
					if (!regionNodeIndices.contains(succ)) {
						if (succ >= traceIndexByNodeIndex.size()
							|| traceIndexByNodeIndex[succ] == std::numeric_limits<size_t>::max()) {
							rejectReason = RegionRejectReason::InteriorOutgoingEdge;
							std::ostringstream detail;
							detail << "pass=\"" << shortPassName(trace.passIndex) << "\""
								<< " node=" << trace.nodeIndex
								<< " outside_succ=" << succ
								<< " missing_trace";
							rejectDetail = detail.str();
							break;
						}
						const size_t succTraceIndex = traceIndexByNodeIndex[succ];
						const auto& succTrace = m_schedulingDecisionTrace[succTraceIndex];
						if (succTraceIndex < end) {
							rejectReason = RegionRejectReason::InteriorOutgoingEdge;
							std::ostringstream detail;
							detail << "pass=\"" << shortPassName(trace.passIndex) << "\""
								<< " node=" << trace.nodeIndex
								<< " outside_succ=" << succ
								<< " succ_trace=" << succTraceIndex
								<< " succ_queue=" << succTrace.assignedQueueSlot
								<< " expected_after_or_at=" << end
								<< " region_queue=" << queueSlot;
							rejectDetail = detail.str();
							break;
						}
						++boundaryOutputEdges;
						if (succTrace.assignedQueueSlot != queueSlot) {
							++crossQueueBoundaryOutputEdges;
						}
					}
				}
				if (rejectReason != RegionRejectReason::Count) {
					break;
				}
			}
		}

		uint32_t transitionCount = 0;
		if (rejectReason == RegionRejectReason::Count) {
			for (uint32_t batchIndex = firstBatchIndex; batchIndex <= lastBatchIndex; ++batchIndex) {
				if (batchIndex >= compiledBatches.size()) {
					rejectReason = RegionRejectReason::BatchHazardBoundary;
					rejectDetail = "batch=" + std::to_string(batchIndex) + " out_of_range";
					break;
				}

				const auto& batch = compiledBatches[batchIndex];
				boundarySyncCount += countBatchCrossQueueSync(batch);

				for (size_t qi = 0; qi < batch.QueueCount(); ++qi) {
					for (size_t phaseIndex = 0; phaseIndex < static_cast<size_t>(BatchTransitionPhase::Count); ++phaseIndex) {
						if (qi != queueSlot && batch.HasTransitions(qi, static_cast<BatchTransitionPhase>(phaseIndex))) {
							crossQueueTransitions += static_cast<uint32_t>(
								batch.Transitions(qi, static_cast<BatchTransitionPhase>(phaseIndex)).size());
						}
					}
				}

				for (size_t qi = 0; qi < batch.QueueCount(); ++qi) {
					for (const auto& queuedPass : batch.Passes(qi)) {
					bool belongsToRegion = false;
					size_t queuedPassTraceIndex = std::numeric_limits<size_t>::max();
					size_t queuedPassIndex = std::numeric_limits<size_t>::max();
					std::visit(
						[&](const auto* passEntry) {
							auto it = passIndexByPointer.find(static_cast<const void*>(passEntry));
							if (it == passIndexByPointer.end()) {
								return;
							}
							queuedPassIndex = it->second;
							belongsToRegion = regionPassIndices.contains(it->second);
							if (it->second < traceIndexByPassIndex.size()) {
								queuedPassTraceIndex = traceIndexByPassIndex[it->second];
							}
						},
						queuedPass);
					if (!belongsToRegion) {
						if (queuedPassTraceIndex == std::numeric_limits<size_t>::max()) {
							rejectReason = RegionRejectReason::BatchHazardBoundary;
							std::ostringstream detail;
							detail << "batch=" << batchIndex
								<< " queue=" << qi
								<< " contains_untraced_pass_outside_candidate";
							rejectDetail = detail.str();
							break;
						}
						if (queuedPassTraceIndex < start) {
							++sameBatchPrefixPasses;
							if (qi != queueSlot) {
								++crossQueueBoundaryPasses;
							}
							continue;
						}
						if (queuedPassTraceIndex >= end) {
							++sameBatchSuffixPasses;
							if (qi != queueSlot) {
								++crossQueueBoundaryPasses;
							}
							continue;
						}
						++sameBatchInterleavedPasses;
						rejectReason = RegionRejectReason::BatchHazardBoundary;
						std::ostringstream detail;
						detail << "batch=" << batchIndex
							<< " queue=" << qi
							<< " contains_interleaved_pass_outside_candidate"
							<< " pass=\"" << shortPassName(static_cast<uint32_t>(queuedPassIndex)) << "\""
							<< " pass_trace=" << queuedPassTraceIndex
							<< " candidate_traces=" << start << "-" << (end > start ? end - 1 : start);
						rejectDetail = detail.str();
						break;
					}
				}
				if (rejectReason != RegionRejectReason::Count) {
					break;
				}
				}
				if (rejectReason != RegionRejectReason::Count) {
					break;
				}
				transitionCount += countBatchTransitions(batch);
			}
		}

		if (rejectReason != RegionRejectReason::Count) {
			appendCandidateDiagnostic(
				candidateIndex,
				start,
				end,
				queueSlot,
				firstPassIndex,
				lastPassIndex,
				firstBatchIndex,
				lastBatchIndex,
				requirementCount,
				isNewBatchNeededChecks,
				boundaryInputEdges,
				boundaryOutputEdges,
				crossQueueBoundaryInputEdges,
				crossQueueBoundaryOutputEdges,
				boundarySyncCount,
				sameBatchPrefixPasses,
				sameBatchSuffixPasses,
				sameBatchInterleavedPasses,
				crossQueueBoundaryPasses,
				crossQueueTransitions,
				rejectReason,
				rejectDetail);
			reject(rejectReason);
			start = end;
			continue;
		}

		ScheduledRegion region{};
		region.firstTraceIndex = static_cast<uint32_t>(start);
		region.lastTraceIndex = static_cast<uint32_t>(end - 1);
		region.firstPassIndex = firstPassIndex;
		region.lastPassIndex = lastPassIndex;
		region.firstBatchIndex = firstBatchIndex;
		region.lastBatchIndex = lastBatchIndex;
		region.queueSlot = queueSlot;
		region.passCount = static_cast<uint32_t>(end - start);
		region.requirementCount = requirementCount;
		region.batchCount = lastBatchIndex >= firstBatchIndex ? (lastBatchIndex - firstBatchIndex + 1) : 0;
		region.transitionCount = transitionCount;
		region.boundaryInputEdgeCount = boundaryInputEdges;
		region.boundaryOutputEdgeCount = boundaryOutputEdges;
		region.crossQueueBoundaryInputEdgeCount = crossQueueBoundaryInputEdges;
		region.crossQueueBoundaryOutputEdgeCount = crossQueueBoundaryOutputEdges;
		region.boundarySyncCount = boundarySyncCount;
		region.sameBatchPrefixPassCount = sameBatchPrefixPasses;
		region.sameBatchSuffixPassCount = sameBatchSuffixPasses;
		region.sameBatchInterleavedPassCount = sameBatchInterleavedPasses;
		region.crossQueueBoundaryPassCount = crossQueueBoundaryPasses;
		region.crossQueueTransitionCount = crossQueueTransitions;
		outRegions.push_back(region);

		++outStats.acceptedRegionCount;
		outStats.coveredPassCount += region.passCount;
		outStats.coveredRequirementCount += region.requirementCount;
		outStats.coveredBatchCount += region.batchCount;
		outStats.coveredTransitionCount += region.transitionCount;
		outStats.largestRegionPassCount = std::max<uint64_t>(outStats.largestRegionPassCount, region.passCount);
		outStats.largestRegionRequirementCount = std::max<uint64_t>(outStats.largestRegionRequirementCount, region.requirementCount);
		outStats.estimatedSavedAddTransitionCalls += region.requirementCount;
		outStats.estimatedSavedIsNewBatchNeededCalls += isNewBatchNeededChecks;
		outStats.boundaryInputEdgeCount += region.boundaryInputEdgeCount;
		outStats.boundaryOutputEdgeCount += region.boundaryOutputEdgeCount;
		outStats.crossQueueBoundaryInputEdgeCount += region.crossQueueBoundaryInputEdgeCount;
		outStats.crossQueueBoundaryOutputEdgeCount += region.crossQueueBoundaryOutputEdgeCount;
		outStats.boundarySyncCount += region.boundarySyncCount;
		outStats.sameBatchPrefixPassCount += region.sameBatchPrefixPassCount;
		outStats.sameBatchSuffixPassCount += region.sameBatchSuffixPassCount;
		outStats.sameBatchInterleavedPassCount += region.sameBatchInterleavedPassCount;
		outStats.crossQueueBoundaryPassCount += region.crossQueueBoundaryPassCount;
		outStats.crossQueueTransitionCount += region.crossQueueTransitionCount;
		appendCandidateDiagnostic(
			candidateIndex,
			start,
			end,
			queueSlot,
			firstPassIndex,
			lastPassIndex,
			firstBatchIndex,
			lastBatchIndex,
			requirementCount,
			isNewBatchNeededChecks,
			boundaryInputEdges,
			boundaryOutputEdges,
			crossQueueBoundaryInputEdges,
			crossQueueBoundaryOutputEdges,
			boundarySyncCount,
			sameBatchPrefixPasses,
			sameBatchSuffixPasses,
			sameBatchInterleavedPasses,
			crossQueueBoundaryPasses,
			crossQueueTransitions,
			RegionRejectReason::Count,
			{});
		start = end;
		}
	}
}

void RenderGraph::ExtractReplaySegmentsFromAuthoritativeCompile(
	const std::vector<Node>& nodes,
	const RenderGraph::FramePassList& framePasses,
	const std::vector<PassBatch>& compiledBatches,
	std::span<const ScheduledRegion> regions,
	std::vector<CachedReplaySegment>& outSegments) const
{
	ZoneScopedN("RenderGraph::ExtractReplaySegmentsFromAuthoritativeCompile");
	outSegments.clear();
	outSegments.reserve(regions.size());

	auto hashState = [](uint64_t seed, const ResourceState& state) {
		seed = HashCombine64(seed, static_cast<uint64_t>(state.access));
		seed = HashCombine64(seed, static_cast<uint64_t>(state.layout));
		seed = HashCombine64(seed, static_cast<uint64_t>(state.sync));
		return seed;
	};
	auto hashBound = [](uint64_t seed, const Bound& bound) {
		seed = HashCombine64(seed, static_cast<uint64_t>(bound.type));
		seed = HashCombine64(seed, bound.value);
		return seed;
	};
	auto hashRange = [&](uint64_t seed, const RangeSpec& range) {
		seed = hashBound(seed, range.mipLower);
		seed = hashBound(seed, range.mipUpper);
		seed = hashBound(seed, range.sliceLower);
		seed = hashBound(seed, range.sliceUpper);
		return seed;
	};
	auto passDeclarationFingerprint = [](const AnyPassAndResources& any) -> uint64_t {
		return std::visit(
			[](const auto& passEntry) -> uint64_t {
				using T = std::decay_t<decltype(passEntry)>;
				if constexpr (std::is_same_v<T, std::monostate>) {
					return 0;
				}
				else {
					return passEntry.declarationCache.declarationFingerprint;
				}
			},
			any.pass);
	};
	auto passHasImmediateWork = [](const AnyPassAndResources& any) {
		return std::visit(
			[](const auto& passEntry) -> bool {
				using T = std::decay_t<decltype(passEntry)>;
				if constexpr (std::is_same_v<T, std::monostate>) {
					return true;
				}
				else {
					return passEntry.run != PassRunMask::Retained || !passEntry.immediateBytecode.empty();
				}
			},
			any.pass);
	};
	auto passHasAliasActivation = [&](size_t passIndex) -> bool {
		if (passIndex >= m_framePassSchedulingSummaries.size()) {
			return false;
		}
		for (const auto& requirement : m_framePassSchedulingSummaries[passIndex].requirements) {
			if (requirement.resourceIndex < m_frameResourceAccessSummaries.size()
				&& m_frameResourceAccessSummaries[requirement.resourceIndex].hasAliasActivation) {
				return true;
			}
		}
		return false;
	};
	auto replayTemplateResourceID = [](Resource* resource) {
		struct Result {
			uint64_t logicalID = 0;
			uint64_t backingID = 0;
			bool dynamic = false;
		};
		if (!resource) {
			return Result{};
		}
		if (auto* dynamicResource = dynamic_cast<DynamicResource*>(resource)) {
			return Result{
				.logicalID = dynamicResource->GetDynamicWrapperGlobalResourceID(),
				.backingID = dynamicResource->GetGlobalResourceID(),
				.dynamic = true,
			};
		}
		return Result{
			.logicalID = resource->GetGlobalResourceID(),
			.backingID = resource->GetGlobalResourceID(),
			.dynamic = false,
		};
	};

	std::unordered_map<const void*, size_t> passIndexByPointer;
	passIndexByPointer.reserve(framePasses.size());
	for (size_t passIndex = 0; passIndex < framePasses.size(); ++passIndex) {
		std::visit(
			[&](const auto& passEntry) {
				using T = std::decay_t<decltype(passEntry)>;
				if constexpr (!std::is_same_v<T, std::monostate>) {
					passIndexByPointer.emplace(static_cast<const void*>(&passEntry), passIndex);
				}
			},
			framePasses[passIndex].pass);
	}

	std::unordered_map<uint32_t, size_t> transitionOwnerRegionByBatch;
	transitionOwnerRegionByBatch.reserve(regions.size() * 2);
	for (size_t regionIndex = 0; regionIndex < regions.size(); ++regionIndex) {
		const auto& region = regions[regionIndex];
		if (region.firstBatchIndex == std::numeric_limits<uint32_t>::max()) {
			continue;
		}
		for (uint32_t batchIndex = region.firstBatchIndex;
			batchIndex <= region.lastBatchIndex && batchIndex < compiledBatches.size();
			++batchIndex) {
			transitionOwnerRegionByBatch.emplace(batchIndex, regionIndex);
			if (batchIndex == std::numeric_limits<uint32_t>::max()) {
				break;
			}
		}
	}

	std::vector<size_t> traceIndexByNodeIndex(nodes.size(), std::numeric_limits<size_t>::max());
	for (size_t traceIndex = 0; traceIndex < m_schedulingDecisionTrace.size(); ++traceIndex) {
		const auto& trace = m_schedulingDecisionTrace[traceIndex];
		if (trace.nodeIndex < traceIndexByNodeIndex.size()) {
			traceIndexByNodeIndex[trace.nodeIndex] = traceIndex;
		}
	}

	auto rangeSpecIsAll = [](const RangeSpec& range) {
		return range.mipLower.type == BoundType::All
			&& range.mipUpper.type == BoundType::All
			&& range.sliceLower.type == BoundType::All
			&& range.sliceUpper.type == BoundType::All;
	};
	auto addInputRequirement = [&](std::vector<ReplaySegmentInputRequirement>& inputs, ReplaySegmentInputRequirement input) {
		auto it = std::find_if(inputs.begin(), inputs.end(), [&](const ReplaySegmentInputRequirement& existing) {
			if (existing.resourceID != input.resourceID) {
				return false;
			}
			const bool exactRange =
				existing.range.mipLower == input.range.mipLower
				&& existing.range.mipUpper == input.range.mipUpper
				&& existing.range.sliceLower == input.range.sliceLower
				&& existing.range.sliceUpper == input.range.sliceUpper;
			if (exactRange) {
				return true;
			}
			// Input requirements are segment-entry expectations. If an earlier
			// whole-resource expectation exists, later narrower transition
			// before-states are internal evolution, not extra boundary inputs.
			return existing.wholeResource || rangeSpecIsAll(existing.range);
		});
		if (it == inputs.end()) {
			inputs.push_back(input);
		}
	};
	struct ReplaySegmentOutputAccumulator {
		ResourceRegistry::RegistryHandle handle{};
		Resource* resource = nullptr;
		SymbolicTracker tracker{};
		bool initialized = false;
	};
	std::vector<ResourceTransition> outputAccumulatorScratch;
	auto addQueueUsage = [](std::vector<ReplaySegmentQueueUsageSummary>& summaries, ReplaySegmentQueueUsageSummary usage) {
		auto it = std::find_if(summaries.begin(), summaries.end(), [&](const ReplaySegmentQueueUsageSummary& existing) {
			return existing.resourceID == usage.resourceID && existing.queueSlot == usage.queueSlot;
		});
		if (it == summaries.end()) {
			summaries.push_back(usage);
			return;
		}
		it->firstLocalBatch = std::min(it->firstLocalBatch, usage.firstLocalBatch);
		it->lastLocalBatch = std::max(it->lastLocalBatch, usage.lastLocalBatch);
		it->read = it->read || usage.read;
		it->write = it->write || usage.write;
		it->transition = it->transition || usage.transition;
		it->producer = it->producer || usage.producer;
	};

	for (size_t regionIndex = 0; regionIndex < regions.size(); ++regionIndex) {
		const ScheduledRegion& region = regions[regionIndex];
		CachedReplaySegment segment{};
		segment.schedule = region;
		segment.identity.passCount = region.passCount;
		segment.identity.passSequenceHash = 0x7365677061737301ull;
		segment.identity.structuralPositionHash = 0x736567706f730001ull;
		segment.fingerprint.declarationHash = 0x7365676465636c01ull;
		segment.fingerprint.accessHash = 0x7365676163630001ull;
		segment.fingerprint.queueHash = 0x7365677175650001ull;
		segment.fingerprint.aliasHash = 0x736567616c690001ull;
		segment.fingerprint.boundaryHash = 0x736567626f750001ull;
		segment.fingerprint.templateShapeHash = 0x736567746d700001ull;
		segment.templateStats.passOrderHash = 0x736567706f726401ull;
		segment.templateStats.transitionShapeHash = 0x7365677473687001ull;
		segment.templateStats.transitionStateHash = 0x7365677473746101ull;
		segment.templateStats.syncShapeHash = 0x73656773796e6301ull;
		std::unordered_map<uint64_t, ReplaySegmentOutputAccumulator> outputAccumulators;

		segment.identity.structuralPositionHash = HashCombine64(segment.identity.structuralPositionHash, region.firstTraceIndex);
		segment.identity.structuralPositionHash = HashCombine64(segment.identity.structuralPositionHash, region.lastTraceIndex);
		segment.identity.structuralPositionHash = HashCombine64(segment.identity.structuralPositionHash, region.queueSlot);
		segment.identity.structuralPositionHash = HashCombine64(segment.identity.structuralPositionHash, region.passCount);
		segment.fingerprint.boundaryHash = HashCombine64(segment.fingerprint.boundaryHash, region.boundaryInputEdgeCount);
		segment.fingerprint.boundaryHash = HashCombine64(segment.fingerprint.boundaryHash, region.boundaryOutputEdgeCount);
		segment.fingerprint.boundaryHash = HashCombine64(segment.fingerprint.boundaryHash, region.crossQueueBoundaryInputEdgeCount);
		segment.fingerprint.boundaryHash = HashCombine64(segment.fingerprint.boundaryHash, region.crossQueueBoundaryOutputEdgeCount);
		segment.fingerprint.boundaryHash = HashCombine64(segment.fingerprint.boundaryHash, region.boundarySyncCount);
		segment.fingerprint.boundaryHash = HashCombine64(segment.fingerprint.boundaryHash, region.sameBatchPrefixPassCount);
		segment.fingerprint.boundaryHash = HashCombine64(segment.fingerprint.boundaryHash, region.sameBatchSuffixPassCount);
		segment.fingerprint.boundaryHash = HashCombine64(segment.fingerprint.boundaryHash, region.crossQueueTransitionCount);

		std::unordered_set<size_t> regionNodeIndices;
		std::unordered_set<size_t> regionPassIndices;
		regionNodeIndices.reserve(region.passCount);
		regionPassIndices.reserve(region.passCount);
		outputAccumulators.reserve(region.passCount * 2);

		auto addOutputState = [&](uint64_t resourceID, const RangeSpec& range, const ResourceState& finalState, ResourceRegistry::RegistryHandle handle, Resource* resource) {
			if (resource == nullptr && handle.IsEphemeral()) {
				resource = handle.GetEphemeralPtr();
			}
			auto& accumulator = outputAccumulators[resourceID];
			if (!accumulator.initialized) {
				accumulator.handle = handle;
				accumulator.resource = resource;
				accumulator.tracker = SymbolicTracker(range, finalState);
				accumulator.initialized = true;
				return;
			}
			if (accumulator.resource == nullptr) {
				accumulator.resource = resource;
			}
			if (accumulator.handle.IsEphemeral() && !handle.IsEphemeral()) {
				accumulator.handle = handle;
			}
			outputAccumulatorScratch.clear();
			accumulator.tracker.Apply(range, accumulator.resource, finalState, outputAccumulatorScratch);
		};
		auto materializeOutputStates = [&]() {
			segment.contract.outputStates.clear();
			for (const auto& [resourceID, accumulator] : outputAccumulators) {
				if (!accumulator.initialized) {
					continue;
				}
				for (const auto& exitSegment : accumulator.tracker.GetSegments()) {
					const bool wholeResource = IsWholeResourceRange(exitSegment.rangeSpec, accumulator.handle);
					segment.contract.outputStates.push_back(ReplaySegmentOutputState{
						.resourceID = resourceID,
						.range = exitSegment.rangeSpec,
						.finalState = exitSegment.state,
						.wholeResource = wholeResource,
						.validFastState = wholeResource,
					});
				}
			}
			auto boundKey = [](const Bound& bound) {
				return std::pair<uint32_t, uint32_t>{ static_cast<uint32_t>(bound.type), bound.value };
			};
			std::sort(segment.contract.outputStates.begin(), segment.contract.outputStates.end(), [&](const auto& lhs, const auto& rhs) {
				return std::tuple{
					lhs.resourceID,
					boundKey(lhs.range.sliceLower),
					boundKey(lhs.range.sliceUpper),
					boundKey(lhs.range.mipLower),
					boundKey(lhs.range.mipUpper),
				} < std::tuple{
					rhs.resourceID,
					boundKey(rhs.range.sliceLower),
					boundKey(rhs.range.sliceUpper),
					boundKey(rhs.range.mipLower),
					boundKey(rhs.range.mipUpper),
				};
			});
		};

		for (uint32_t traceIndex = region.firstTraceIndex; traceIndex <= region.lastTraceIndex && traceIndex < m_schedulingDecisionTrace.size(); ++traceIndex) {
			const auto& trace = m_schedulingDecisionTrace[traceIndex];
			regionNodeIndices.insert(trace.nodeIndex);
			regionPassIndices.insert(trace.passIndex);
		}

		for (uint32_t traceIndex = region.firstTraceIndex; traceIndex <= region.lastTraceIndex && traceIndex < m_schedulingDecisionTrace.size(); ++traceIndex) {
			const auto& trace = m_schedulingDecisionTrace[traceIndex];
			if (trace.passIndex >= framePasses.size()) {
				continue;
			}
			const auto& pass = framePasses[trace.passIndex];
			segment.identity.passSequenceHash = HashCombine64(segment.identity.passSequenceHash, HashString64(pass.name));
			segment.identity.passSequenceHash = HashCombine64(segment.identity.passSequenceHash, static_cast<uint64_t>(pass.type));
			segment.fingerprint.declarationHash = HashCombine64(segment.fingerprint.declarationHash, passDeclarationFingerprint(pass));
			segment.fingerprint.queueHash = HashCombine64(segment.fingerprint.queueHash, trace.assignedQueueSlot);

			if (trace.passIndex < m_framePassSchedulingSummaries.size()) {
				const auto& passSummary = m_framePassSchedulingSummaries[trace.passIndex];
				for (const auto& requirement : passSummary.requirements) {
					segment.fingerprint.accessHash = HashCombine64(segment.fingerprint.accessHash, requirement.resourceID);
					segment.fingerprint.accessHash = hashRange(segment.fingerprint.accessHash, requirement.range);
					segment.fingerprint.accessHash = hashState(segment.fingerprint.accessHash, requirement.state);

					const bool isWrite = AccessTypeIsWriteType(requirement.state.access);
					addQueueUsage(segment.contract.queueUsage, ReplaySegmentQueueUsageSummary{
						.resourceID = requirement.resourceID,
						.queueSlot = static_cast<uint16_t>(trace.assignedQueueSlot),
						.firstLocalBatch = trace.batchIndex - region.firstBatchIndex,
						.lastLocalBatch = trace.batchIndex - region.firstBatchIndex,
						.read = !isWrite,
						.write = isWrite,
						.transition = false,
						.producer = isWrite,
					});

					if (const auto* aliasRange = TryGetSchedulingPlacementRangeByResourceIndex(requirement.resourceIndex)) {
						segment.fingerprint.aliasHash = HashCombine64(segment.fingerprint.aliasHash, requirement.resourceID);
						segment.fingerprint.aliasHash = HashCombine64(segment.fingerprint.aliasHash, aliasRange->poolID);
						segment.fingerprint.aliasHash = HashCombine64(segment.fingerprint.aliasHash, aliasRange->startByte);
						segment.fingerprint.aliasHash = HashCombine64(segment.fingerprint.aliasHash, aliasRange->endByte);
						segment.fingerprint.aliasHash = HashCombine64(segment.fingerprint.aliasHash, aliasRange->dedicatedBacking ? 1ull : 0ull);
					}
				}
				for (const auto& transition : passSummary.internalTransitions) {
					segment.fingerprint.accessHash = HashCombine64(segment.fingerprint.accessHash, 0x7472616e736974ull);
					segment.fingerprint.accessHash = HashCombine64(segment.fingerprint.accessHash, transition.resourceID);
					addQueueUsage(segment.contract.queueUsage, ReplaySegmentQueueUsageSummary{
						.resourceID = transition.resourceID,
						.queueSlot = static_cast<uint16_t>(trace.assignedQueueSlot),
						.firstLocalBatch = trace.batchIndex - region.firstBatchIndex,
						.lastLocalBatch = trace.batchIndex - region.firstBatchIndex,
						.read = false,
						.write = true,
						.transition = true,
						.producer = true,
					});
				}
			}
		}

		for (uint32_t traceIndex = region.firstTraceIndex; traceIndex <= region.lastTraceIndex && traceIndex < m_schedulingDecisionTrace.size(); ++traceIndex) {
			const auto& trace = m_schedulingDecisionTrace[traceIndex];
			if (trace.nodeIndex >= nodes.size()) {
				continue;
			}
			const auto& node = nodes[trace.nodeIndex];
			auto addBoundaryEdge = [&](size_t outsideNode, bool incoming) {
				if (outsideNode >= traceIndexByNodeIndex.size()) {
					return;
				}
				const size_t outsideTraceIndex = traceIndexByNodeIndex[outsideNode];
				if (outsideTraceIndex == std::numeric_limits<size_t>::max()
					|| outsideTraceIndex >= m_schedulingDecisionTrace.size()) {
					return;
				}
				const auto& outsideTrace = m_schedulingDecisionTrace[outsideTraceIndex];
				const bool crossQueue = outsideTrace.assignedQueueSlot != trace.assignedQueueSlot;
				segment.contract.boundaryEdges.push_back(ReplaySegmentBoundaryEdge{
					.insideNode = trace.nodeIndex,
					.outsideNode = static_cast<uint32_t>(outsideNode),
					.insideTraceIndex = traceIndex,
					.outsideTraceIndex = static_cast<uint32_t>(outsideTraceIndex),
					.insideQueueSlot = trace.assignedQueueSlot,
					.outsideQueueSlot = outsideTrace.assignedQueueSlot,
					.incoming = incoming,
					.crossQueue = crossQueue,
				});
				segment.fingerprint.boundaryHash = HashCombine64(segment.fingerprint.boundaryHash, trace.nodeIndex);
				segment.fingerprint.boundaryHash = HashCombine64(segment.fingerprint.boundaryHash, outsideNode);
				segment.fingerprint.boundaryHash = HashCombine64(segment.fingerprint.boundaryHash, incoming ? 1ull : 0ull);
				segment.fingerprint.boundaryHash = HashCombine64(segment.fingerprint.boundaryHash, crossQueue ? 1ull : 0ull);
			};
			for (size_t pred : node.in) {
				if (!regionNodeIndices.contains(pred)) {
					addBoundaryEdge(pred, true);
				}
			}
			for (size_t succ : node.out) {
				if (!regionNodeIndices.contains(succ)) {
					addBoundaryEdge(succ, false);
				}
			}
		}

		for (uint32_t batchIndex = region.firstBatchIndex; batchIndex <= region.lastBatchIndex && batchIndex < compiledBatches.size(); ++batchIndex) {
			const auto& batch = compiledBatches[batchIndex];
			ReplaySegmentBatchTemplate batchTemplate{};
			batchTemplate.localBatchIndex = batchIndex - region.firstBatchIndex;
			batchTemplate.originalBatchIndexAtExtraction = batchIndex;
			batchTemplate.partialBatch = false;
			batchTemplate.allResources = batch.allResources;
			batchTemplate.internallyTransitionedResources = batch.internallyTransitionedResources;

			for (size_t queueIndex = 0; queueIndex < batch.QueueCount(); ++queueIndex) {
				for (const auto& queuedPass : batch.Passes(queueIndex)) {
					std::visit(
						[&](const auto* passEntry) {
							auto it = passIndexByPointer.find(static_cast<const void*>(passEntry));
							if (it == passIndexByPointer.end() || !regionPassIndices.contains(it->second)) {
								batchTemplate.partialBatch = true;
							}
						},
						queuedPass);
				}
			}

			const auto ownerIt = transitionOwnerRegionByBatch.find(batchIndex);
			const bool ownsBatchTransitions = ownerIt == transitionOwnerRegionByBatch.end()
				|| ownerIt->second == regionIndex
				|| !batchTemplate.partialBatch;

			for (size_t queueIndex = 0; queueIndex < batch.QueueCount(); ++queueIndex) {
				auto addExpectedTransitionInput = [&](const ResourceTransition& transition) {
					if (!transition.pResource) {
						return;
					}
					const auto transitionResourceID = replayTemplateResourceID(transition.pResource);
					ResourceRegistry::RegistryHandle handle{};
					if (auto cachedHandle = _registry.GetHandleFor(transition.pResource)) {
						handle = *cachedHandle;
					}
					else {
						handle = ResourceRegistry::RegistryHandle::MakeEphemeral(transition.pResource);
					}
					auto resourceIndex = TryGetFrameSchedulingResourceIndex(transitionResourceID.logicalID);
					if (!resourceIndex && transitionResourceID.backingID != 0) {
						resourceIndex = TryGetFrameSchedulingResourceIndex(transitionResourceID.backingID);
					}
					if (!resourceIndex) {
						return;
					}
					addInputRequirement(segment.contract.inputRequirements, ReplaySegmentInputRequirement{
						.resource = handle,
						.resourceID = transitionResourceID.logicalID,
						.resourceIndexAtExtraction = *resourceIndex,
						.range = transition.range,
						.requiredState = ResourceState{ transition.prevAccessType, transition.prevLayout, transition.prevSyncState },
						.queueSlot = static_cast<uint16_t>(queueIndex),
						.wholeResource = IsWholeResourceRange(transition.range, handle),
						.aliasActivation = *resourceIndex < m_frameResourceAccessSummaries.size()
							&& m_frameResourceAccessSummaries[*resourceIndex].hasAliasActivation,
						.transitionBeforeState = true,
						.transitionDiscard = transition.discard,
						.readOnlyUniformWeakRequirement = *resourceIndex < m_frameResourceAccessSummaries.size()
							&& m_frameResourceAccessSummaries[*resourceIndex].readOnlyUniform,
					});
				};

				if (ownsBatchTransitions) {
					for (const auto& transition : batch.Transitions(queueIndex, BatchTransitionPhase::BeforePasses)) {
						addExpectedTransitionInput(transition);
					}
				}

				for (const auto& queuedPass : batch.Passes(queueIndex)) {
					std::visit(
						[&](const auto* passEntry) {
							auto it = passIndexByPointer.find(static_cast<const void*>(passEntry));
							if (it == passIndexByPointer.end() || !regionPassIndices.contains(it->second)) {
								return;
							}
							batchTemplate.queuedPasses.push_back(ReplaySegmentQueuedPassTemplate{
								.localPassOrdinal = static_cast<uint32_t>(batchTemplate.queuedPasses.size()),
								.originalFramePassIndexAtExtraction = static_cast<uint32_t>(it->second),
								.passNameHash = HashString64(framePasses[it->second].name),
								.queueSlot = static_cast<uint16_t>(queueIndex),
								.type = framePasses[it->second].type,
							});
							++segment.templateStats.queuedPassCount;
							segment.templateStats.passOrderHash = HashCombine64(segment.templateStats.passOrderHash, HashString64(framePasses[it->second].name));
							segment.templateStats.passOrderHash = HashCombine64(segment.templateStats.passOrderHash, queueIndex);
							segment.templateStats.passOrderHash = HashCombine64(segment.templateStats.passOrderHash, static_cast<uint64_t>(framePasses[it->second].type));

							if (it->second < m_framePassSchedulingSummaries.size()) {
								const auto& passSummary = m_framePassSchedulingSummaries[it->second];
								const QueueKind queueKind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(queueIndex));
								for (const auto& requirement : passSummary.requirements) {
									const ResourceState normalizedState = NormalizeStateForQueue(queueKind, requirement.state);
									addInputRequirement(segment.contract.inputRequirements, ReplaySegmentInputRequirement{
										.resource = requirement.resource,
										.resourceID = requirement.resourceID,
										.resourceIndexAtExtraction = requirement.resourceIndex,
										.range = requirement.range,
										.requiredState = normalizedState,
										.queueSlot = static_cast<uint16_t>(queueIndex),
										.wholeResource = IsWholeResourceRange(requirement.range, requirement.resource),
										.aliasActivation = requirement.resourceIndex < m_frameResourceAccessSummaries.size()
											&& m_frameResourceAccessSummaries[requirement.resourceIndex].hasAliasActivation,
										.transitionBeforeState = false,
										.transitionDiscard = false,
										.readOnlyUniformWeakRequirement = requirement.resourceIndex < m_frameResourceAccessSummaries.size()
											&& m_frameResourceAccessSummaries[requirement.resourceIndex].readOnlyUniform,
									});
									Resource* requirementResource = requirement.resource.IsEphemeral()
										? requirement.resource.GetEphemeralPtr()
										: const_cast<Resource*>(_registry.Resolve(requirement.resource));
									addOutputState(requirement.resourceID, requirement.range, normalizedState, requirement.resource, requirementResource);
								}
								const auto& denseTransitions = passSummary.internalTransitions;
								if (passEntry->resources.internalTransitions.size() == denseTransitions.size()) {
									for (size_t transitionIndex = 0; transitionIndex < denseTransitions.size(); ++transitionIndex) {
										const auto& exit = passEntry->resources.internalTransitions[transitionIndex];
										const auto& denseTransition = denseTransitions[transitionIndex];
										Resource* transitionResource = exit.first.resource.IsEphemeral()
											? exit.first.resource.GetEphemeralPtr()
											: const_cast<Resource*>(_registry.Resolve(exit.first.resource));
										addOutputState(denseTransition.resourceID, exit.first.range, exit.second, exit.first.resource, transitionResource);
									}
								}
							}
						},
						queuedPass);
				}

				for (size_t phaseIndex = 0; phaseIndex < static_cast<size_t>(BatchTransitionPhase::Count); ++phaseIndex) {
					const auto phase = static_cast<BatchTransitionPhase>(phaseIndex);
					if (!ownsBatchTransitions) {
						continue;
					}
					for (const auto& transition : batch.Transitions(queueIndex, phase)) {
						if (phase == BatchTransitionPhase::AfterPasses) {
							addExpectedTransitionInput(transition);
						}
						const auto transitionResourceID = replayTemplateResourceID(transition.pResource);
						if (transition.pResource != nullptr) {
							ResourceRegistry::RegistryHandle transitionHandle{};
							if (auto cachedHandle = _registry.GetHandleFor(transition.pResource)) {
								transitionHandle = *cachedHandle;
							}
							else {
								transitionHandle = ResourceRegistry::RegistryHandle::MakeEphemeral(transition.pResource);
							}
							addOutputState(
								transitionResourceID.logicalID,
								transition.range,
								ResourceState{ transition.newAccessType, transition.newLayout, transition.newSyncState },
								transitionHandle,
								transition.pResource);
						}
						batchTemplate.transitions.push_back(ReplaySegmentTransitionTemplate{
							.resourceID = transitionResourceID.logicalID,
							.backingResourceID = transitionResourceID.backingID,
							.range = transition.range,
							.before = ResourceState{ transition.prevAccessType, transition.prevLayout, transition.prevSyncState },
							.after = ResourceState{ transition.newAccessType, transition.newLayout, transition.newSyncState },
							.discard = transition.discard,
							.queueSlot = static_cast<uint16_t>(queueIndex),
							.phase = phase,
							.dynamicResource = transitionResourceID.dynamic,
						});
						++segment.templateStats.transitionCount;
						segment.fingerprint.templateShapeHash = HashCombine64(segment.fingerprint.templateShapeHash, transitionResourceID.logicalID);
						segment.fingerprint.templateShapeHash = HashCombine64(segment.fingerprint.templateShapeHash, transitionResourceID.dynamic ? 1ull : 0ull);
						segment.fingerprint.templateShapeHash = hashRange(segment.fingerprint.templateShapeHash, transition.range);
						segment.fingerprint.templateShapeHash = hashState(segment.fingerprint.templateShapeHash, ResourceState{ transition.newAccessType, transition.newLayout, transition.newSyncState });
						segment.fingerprint.templateShapeHash = HashCombine64(segment.fingerprint.templateShapeHash, queueIndex);
						segment.fingerprint.templateShapeHash = HashCombine64(segment.fingerprint.templateShapeHash, phaseIndex);
						segment.templateStats.transitionShapeHash = HashCombine64(segment.templateStats.transitionShapeHash, transitionResourceID.logicalID);
						segment.templateStats.transitionShapeHash = HashCombine64(segment.templateStats.transitionShapeHash, transitionResourceID.dynamic ? 1ull : 0ull);
						segment.templateStats.transitionShapeHash = hashRange(segment.templateStats.transitionShapeHash, transition.range);
						segment.templateStats.transitionShapeHash = hashState(segment.templateStats.transitionShapeHash, ResourceState{ transition.newAccessType, transition.newLayout, transition.newSyncState });
						segment.templateStats.transitionShapeHash = HashCombine64(segment.templateStats.transitionShapeHash, queueIndex);
						segment.templateStats.transitionShapeHash = HashCombine64(segment.templateStats.transitionShapeHash, phaseIndex);
						segment.templateStats.transitionShapeHash = HashCombine64(segment.templateStats.transitionShapeHash, transition.discard ? 1ull : 0ull);
						segment.templateStats.transitionStateHash = hashState(segment.templateStats.transitionStateHash, ResourceState{ transition.prevAccessType, transition.prevLayout, transition.prevSyncState });
						segment.templateStats.transitionStateHash = hashState(segment.templateStats.transitionStateHash, ResourceState{ transition.newAccessType, transition.newLayout, transition.newSyncState });
					}
				}

				auto queueHasTemplateWork = [&](size_t candidateQueueIndex) {
					for (const auto& queuedPass : batchTemplate.queuedPasses) {
						if (queuedPass.queueSlot == candidateQueueIndex) {
							return true;
						}
					}
					for (const auto& transitionTemplate : batchTemplate.transitions) {
						if (transitionTemplate.queueSlot == candidateQueueIndex) {
							return true;
						}
					}
					return false;
				};
				const bool queueOwnsTemplateWork = queueHasTemplateWork(queueIndex);

				for (size_t waitPhaseIndex = 0; waitPhaseIndex < PassBatch::kWaitPhaseCount; ++waitPhaseIndex) {
					const auto waitPhase = static_cast<BatchWaitPhase>(waitPhaseIndex);
					for (size_t src = 0; src < batch.QueueCount(); ++src) {
						if (!batch.HasQueueWait(waitPhase, queueIndex, src)) {
							continue;
						}
						if (!queueOwnsTemplateWork) {
							continue;
						}
						batchTemplate.waits.push_back(ReplaySegmentWaitTemplate{
							.dstQueue = static_cast<uint16_t>(queueIndex),
							.srcQueue = static_cast<uint16_t>(src),
							.phase = waitPhase,
						});
						++segment.templateStats.waitCount;
						segment.templateStats.syncShapeHash = HashCombine64(segment.templateStats.syncShapeHash, queueIndex);
						segment.templateStats.syncShapeHash = HashCombine64(segment.templateStats.syncShapeHash, src);
						segment.templateStats.syncShapeHash = HashCombine64(segment.templateStats.syncShapeHash, waitPhaseIndex);
						segment.contract.boundarySyncs.push_back(ReplaySegmentBoundarySync{
							.dstQueue = static_cast<uint16_t>(queueIndex),
							.srcQueue = static_cast<uint16_t>(src),
							.waitPhase = waitPhase,
							.signalPhase = BatchSignalPhase::AfterCompletion,
							.batchIndex = batchIndex,
							.internalOnly = false,
						});
					}
				}
				for (size_t signalPhaseIndex = 0; signalPhaseIndex < PassBatch::kSignalPhaseCount; ++signalPhaseIndex) {
					const auto signalPhase = static_cast<BatchSignalPhase>(signalPhaseIndex);
					if (!batch.HasQueueSignal(signalPhase, queueIndex)) {
						continue;
					}
					if (!queueOwnsTemplateWork) {
						continue;
					}
					batchTemplate.signals.push_back(ReplaySegmentSignalTemplate{
						.queueSlot = static_cast<uint16_t>(queueIndex),
						.phase = signalPhase,
					});
					++segment.templateStats.signalCount;
					segment.templateStats.syncShapeHash = HashCombine64(segment.templateStats.syncShapeHash, queueIndex);
					segment.templateStats.syncShapeHash = HashCombine64(segment.templateStats.syncShapeHash, signalPhaseIndex + 1024);
				}
			}

			segment.fingerprint.templateShapeHash = HashCombine64(segment.fingerprint.templateShapeHash, batchTemplate.queuedPasses.size());
			segment.fingerprint.templateShapeHash = HashCombine64(segment.fingerprint.templateShapeHash, batchTemplate.transitions.size());
			segment.fingerprint.templateShapeHash = HashCombine64(segment.fingerprint.templateShapeHash, batchTemplate.waits.size());
			segment.fingerprint.templateShapeHash = HashCombine64(segment.fingerprint.templateShapeHash, batchTemplate.signals.size());
			++segment.templateStats.batchCount;
			segment.templateStats.partialBatchCount += batchTemplate.partialBatch ? 1u : 0u;
			segment.fingerprint.templateShapeHash = HashCombine64(segment.fingerprint.templateShapeHash, batchTemplate.partialBatch ? 1ull : 0ull);
			segment.fingerprint.templateShapeHash = HashCombine64(segment.fingerprint.templateShapeHash, segment.templateStats.passOrderHash);
			segment.fingerprint.templateShapeHash = HashCombine64(segment.fingerprint.templateShapeHash, segment.templateStats.transitionShapeHash);
			segment.fingerprint.templateShapeHash = HashCombine64(segment.fingerprint.templateShapeHash, segment.templateStats.syncShapeHash);
			segment.batchTemplates.push_back(std::move(batchTemplate));
		}

		materializeOutputStates();

		segment.tier1Eligible = region.sameBatchInterleavedPassCount == 0;
		for (uint32_t traceIndex = region.firstTraceIndex; traceIndex <= region.lastTraceIndex && traceIndex < m_schedulingDecisionTrace.size(); ++traceIndex) {
			const auto& trace = m_schedulingDecisionTrace[traceIndex];
			if (trace.passIndex >= framePasses.size()) {
				segment.tier1Eligible = false;
				break;
			}
			if (passHasImmediateWork(framePasses[trace.passIndex])) {
				segment.tier1Eligible = false;
				break;
			}
			if (trace.passIndex < m_framePassIsFrameExtension.size() && m_framePassIsFrameExtension[trace.passIndex]) {
				segment.tier1Eligible = false;
				break;
			}
		}

		outSegments.push_back(std::move(segment));
	}
}

RenderGraph::CachedReplaySegment RenderGraph::RemapCachedReplaySegmentToCurrentFrame(
	const CachedReplaySegment& segment,
	const std::vector<PassBatch>& compiledBatches) const
{
	ZoneScopedN("RenderGraph::RemapCachedReplaySegmentToCurrentFrame");
	CachedReplaySegment remapped = segment;
	ScheduledRegion& region = remapped.schedule;

	uint32_t firstTraceIndex = std::numeric_limits<uint32_t>::max();
	uint32_t lastTraceIndex = 0;
	uint32_t firstPassIndex = std::numeric_limits<uint32_t>::max();
	uint32_t lastPassIndex = 0;
	uint32_t firstBatchIndex = std::numeric_limits<uint32_t>::max();
	uint32_t lastBatchIndex = 0;
	uint32_t passCount = 0;
	uint32_t requirementCount = 0;

	std::unordered_map<uint32_t, const SchedulingDecisionTrace*> traceByPassIndex;
	traceByPassIndex.reserve(m_schedulingDecisionTrace.size());
	for (const auto& trace : m_schedulingDecisionTrace) {
		traceByPassIndex.emplace(trace.passIndex, &trace);
	}

	for (const auto& batchTemplate : remapped.batchTemplates) {
		for (const auto& queuedPass : batchTemplate.queuedPasses) {
			auto traceIt = traceByPassIndex.find(queuedPass.originalFramePassIndexAtExtraction);
			if (traceIt == traceByPassIndex.end()) {
				continue;
			}
			const SchedulingDecisionTrace& trace = *traceIt->second;
			const uint32_t traceIndex = static_cast<uint32_t>(traceIt->second - m_schedulingDecisionTrace.data());
			firstTraceIndex = std::min(firstTraceIndex, traceIndex);
			lastTraceIndex = std::max(lastTraceIndex, traceIndex);
			firstPassIndex = std::min(firstPassIndex, trace.passIndex);
			lastPassIndex = std::max(lastPassIndex, trace.passIndex);
			firstBatchIndex = std::min(firstBatchIndex, trace.batchIndex);
			lastBatchIndex = std::max(lastBatchIndex, trace.batchIndex);
			if (trace.passIndex < m_framePassSchedulingSummaries.size()) {
				requirementCount += static_cast<uint32_t>(m_framePassSchedulingSummaries[trace.passIndex].requirements.size());
			}
			++passCount;
		}
	}

	if (passCount == 0 || firstTraceIndex == std::numeric_limits<uint32_t>::max()) {
		return remapped;
	}

	auto countBatchTransitions = [](const PassBatch& batch) {
		uint32_t count = 0;
		for (size_t queueIndex = 0; queueIndex < batch.QueueCount(); ++queueIndex) {
			for (size_t phaseIndex = 0; phaseIndex < static_cast<size_t>(BatchTransitionPhase::Count); ++phaseIndex) {
				count += static_cast<uint32_t>(batch.Transitions(queueIndex, static_cast<BatchTransitionPhase>(phaseIndex)).size());
			}
		}
		return count;
	};

	uint32_t transitionCount = 0;
	if (firstBatchIndex != std::numeric_limits<uint32_t>::max()) {
		for (uint32_t batchIndex = firstBatchIndex; batchIndex <= lastBatchIndex && batchIndex < compiledBatches.size(); ++batchIndex) {
			transitionCount += countBatchTransitions(compiledBatches[batchIndex]);
		}
	}

	region.firstTraceIndex = firstTraceIndex;
	region.lastTraceIndex = lastTraceIndex;
	region.firstPassIndex = firstPassIndex;
	region.lastPassIndex = lastPassIndex;
	region.firstBatchIndex = firstBatchIndex;
	region.lastBatchIndex = lastBatchIndex;
	region.queueSlot = m_schedulingDecisionTrace[firstTraceIndex].assignedQueueSlot;
	region.passCount = passCount;
	region.requirementCount = requirementCount;
	region.batchCount = firstBatchIndex != std::numeric_limits<uint32_t>::max() && lastBatchIndex >= firstBatchIndex
		? (lastBatchIndex - firstBatchIndex + 1)
		: 0;
	region.transitionCount = transitionCount;

	remapped.identity.passCount = passCount;
	remapped.identity.structuralPositionHash = 0x736567706f730001ull;
	remapped.identity.structuralPositionHash = HashCombine64(remapped.identity.structuralPositionHash, region.firstTraceIndex);
	remapped.identity.structuralPositionHash = HashCombine64(remapped.identity.structuralPositionHash, region.lastTraceIndex);
	remapped.identity.structuralPositionHash = HashCombine64(remapped.identity.structuralPositionHash, region.queueSlot);
	remapped.identity.structuralPositionHash = HashCombine64(remapped.identity.structuralPositionHash, region.passCount);

	return remapped;
}

RenderGraph::ReplaySegmentValidationStats RenderGraph::ValidateCachedSegmentsAgainstCurrentFrame(
	std::span<const CachedReplaySegment> previousSegments,
	std::span<const CachedReplaySegment> currentSegments) const
{
	ReplaySegmentValidationStats stats{};
	stats.previousSegmentCount = previousSegments.size();
	stats.currentSegmentCount = currentSegments.size();

	std::unordered_map<uint64_t, const CachedReplaySegment*> currentByPassSequence;
	currentByPassSequence.reserve(currentSegments.size());
	for (const auto& segment : currentSegments) {
		currentByPassSequence.emplace(segment.identity.passSequenceHash, &segment);
	}

	auto sameRange = [](const RangeSpec& lhs, const RangeSpec& rhs) {
		return lhs.mipLower == rhs.mipLower
			&& lhs.mipUpper == rhs.mipUpper
			&& lhs.sliceLower == rhs.sliceLower
			&& lhs.sliceUpper == rhs.sliceUpper;
	};
	auto appendState = [](std::ostringstream& oss, const ResourceState& state) {
		oss << "{access=" << static_cast<uint64_t>(state.access)
			<< ",layout=" << static_cast<uint64_t>(state.layout)
			<< ",sync=" << static_cast<uint64_t>(state.sync)
			<< "}";
	};
	auto buildTransitionShapeDiff = [&](const CachedReplaySegment& previous, const CachedReplaySegment& current) {
		std::ostringstream firstDiff;
		const size_t batchCount = std::min(previous.batchTemplates.size(), current.batchTemplates.size());
		for (size_t batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
			const auto& previousBatch = previous.batchTemplates[batchIndex];
			const auto& currentBatch = current.batchTemplates[batchIndex];
			const size_t transitionCount = std::min(previousBatch.transitions.size(), currentBatch.transitions.size());
			for (size_t transitionIndex = 0; transitionIndex < transitionCount; ++transitionIndex) {
				const auto& previousTransition = previousBatch.transitions[transitionIndex];
				const auto& currentTransition = currentBatch.transitions[transitionIndex];
				bool differs = false;
				std::ostringstream fields;
				if (previousTransition.resourceID != currentTransition.resourceID) {
					++stats.transitionShapeResourceDiffs;
					differs = true;
					fields << " resource=" << previousTransition.resourceID << "->" << currentTransition.resourceID;
					if (previousTransition.dynamicResource || currentTransition.dynamicResource) {
						fields << " dynamic=" << previousTransition.dynamicResource << "->" << currentTransition.dynamicResource
							<< " backing=" << previousTransition.backingResourceID << "->" << currentTransition.backingResourceID;
					}
				}
				else if ((previousTransition.dynamicResource || currentTransition.dynamicResource)
					&& previousTransition.backingResourceID != currentTransition.backingResourceID) {
					fields << " dynamic_backing_ignored=" << previousTransition.backingResourceID
						<< "->" << currentTransition.backingResourceID;
				}
				if (!sameRange(previousTransition.range, currentTransition.range)) {
					++stats.transitionShapeRangeDiffs;
					differs = true;
					fields << " range=\"" << FormatRangeSpec(previousTransition.range)
						<< "\"->\"" << FormatRangeSpec(currentTransition.range) << "\"";
				}
				if (!StatesExactlyEqual(previousTransition.after, currentTransition.after)) {
					++stats.transitionShapeAfterStateDiffs;
					differs = true;
					fields << " after=";
					appendState(fields, previousTransition.after);
					fields << "->";
					appendState(fields, currentTransition.after);
				}
				if (previousTransition.queueSlot != currentTransition.queueSlot) {
					++stats.transitionShapeQueueDiffs;
					differs = true;
					fields << " queue=" << previousTransition.queueSlot << "->" << currentTransition.queueSlot;
				}
				if (previousTransition.phase != currentTransition.phase) {
					++stats.transitionShapePhaseDiffs;
					differs = true;
					fields << " phase=" << static_cast<uint32_t>(previousTransition.phase)
						<< "->" << static_cast<uint32_t>(currentTransition.phase);
				}
				if (previousTransition.discard != currentTransition.discard) {
					++stats.transitionShapeDiscardDiffs;
					differs = true;
					fields << " discard=" << previousTransition.discard << "->" << currentTransition.discard;
				}
				if (differs && firstDiff.str().empty()) {
					firstDiff << "local_batch=" << batchIndex
						<< " original_batch=" << previousBatch.originalBatchIndexAtExtraction
						<< " transition=" << transitionIndex
						<< fields.str();
				}
			}
			if (previousBatch.transitions.size() != currentBatch.transitions.size() && firstDiff.str().empty()) {
				firstDiff << "local_batch=" << batchIndex
					<< " original_batch=" << previousBatch.originalBatchIndexAtExtraction
					<< " transition_count=" << previousBatch.transitions.size()
					<< "->" << currentBatch.transitions.size();
			}
		}
		if (previous.batchTemplates.size() != current.batchTemplates.size() && firstDiff.str().empty()) {
			firstDiff << "batch_template_count=" << previous.batchTemplates.size()
				<< "->" << current.batchTemplates.size();
		}
		return firstDiff.str();
	};

	for (const auto& previous : previousSegments) {
		auto it = currentByPassSequence.find(previous.identity.passSequenceHash);
		if (it == currentByPassSequence.end()) {
			++stats.misses;
			++stats.missesByReason[static_cast<size_t>(ReplaySegmentInvalidationReason::PassSequenceChanged)];
			continue;
		}
		const auto& current = *it->second;
		const bool relaxAliasPlacement = m_renderGraphSettingsService
			? m_renderGraphSettingsService->GetRenderGraphReplayRelaxAliasPlacement()
			: true;
		ReplaySegmentInvalidationReason reason = ReplaySegmentInvalidationReason::None;
		std::string detail;
		if (previous.fingerprint.declarationHash != current.fingerprint.declarationHash) {
			reason = ReplaySegmentInvalidationReason::DeclarationChanged;
			detail = "declaration_hash";
		}
		else if (previous.fingerprint.accessHash != current.fingerprint.accessHash) {
			reason = ReplaySegmentInvalidationReason::AccessChanged;
			detail = "access_hash";
		}
		else if (previous.fingerprint.queueHash != current.fingerprint.queueHash) {
			reason = ReplaySegmentInvalidationReason::QueueAssignmentChanged;
			detail = "queue_hash";
		}
		else if (!relaxAliasPlacement && previous.fingerprint.aliasHash != current.fingerprint.aliasHash) {
			reason = ReplaySegmentInvalidationReason::AliasPlacementChanged;
			detail = "alias_hash";
		}
		else if (previous.fingerprint.boundaryHash != current.fingerprint.boundaryHash) {
			reason = ReplaySegmentInvalidationReason::BoundaryChanged;
			detail = "boundary_hash";
		}
		else if (previous.fingerprint.templateShapeHash != current.fingerprint.templateShapeHash) {
			reason = ReplaySegmentInvalidationReason::TemplateShapeChanged;
			std::ostringstream oss;
			oss << "template_shape"
				<< " batches=" << previous.templateStats.batchCount << "->" << current.templateStats.batchCount
				<< " partial=" << previous.templateStats.partialBatchCount << "->" << current.templateStats.partialBatchCount
				<< " queued_passes=" << previous.templateStats.queuedPassCount << "->" << current.templateStats.queuedPassCount
				<< " transitions=" << previous.templateStats.transitionCount << "->" << current.templateStats.transitionCount
				<< " waits=" << previous.templateStats.waitCount << "->" << current.templateStats.waitCount
				<< " signals=" << previous.templateStats.signalCount << "->" << current.templateStats.signalCount
				<< " pass_order=0x" << std::hex << previous.templateStats.passOrderHash << "->0x" << current.templateStats.passOrderHash
				<< " transition_shape=0x" << previous.templateStats.transitionShapeHash << "->0x" << current.templateStats.transitionShapeHash
				<< " sync_shape=0x" << previous.templateStats.syncShapeHash << "->0x" << current.templateStats.syncShapeHash;
			detail = oss.str();
			if (previous.templateStats.transitionShapeHash != current.templateStats.transitionShapeHash) {
				std::string transitionDiff = buildTransitionShapeDiff(previous, current);
				if (stats.firstTransitionShapeDiffDetail.empty()) {
					stats.firstTransitionShapeDiffDetail = transitionDiff.empty() ? "transition_shape_hash_changed_but_ordered_template_diff_not_found" : transitionDiff;
				}
				if (!transitionDiff.empty()) {
					detail += " first_transition_diff=\"" + transitionDiff + "\"";
				}
			}
		}
		else if (previous.templateStats.transitionStateHash != current.templateStats.transitionStateHash) {
			++stats.templateStateDivergencesAllowed;
		}

		if (reason == ReplaySegmentInvalidationReason::None) {
			++stats.hits;
		}
		else {
			++stats.misses;
			++stats.missesByReason[static_cast<size_t>(reason)];
			if (stats.firstMissDetail.empty()) {
				std::ostringstream oss;
				oss << "pass_sequence=0x" << std::hex << previous.identity.passSequenceHash
					<< " structural=0x" << previous.identity.structuralPositionHash
					<< " reason=" << detail;
				stats.firstMissDetail = oss.str();
			}
		}
	}

	return stats;
}

RenderGraph::ReplaySegmentCacheKey RenderGraph::BuildReplaySegmentCacheKey(const CachedReplaySegment& segment) const
{
	ZoneScopedN("RenderGraph::BuildReplaySegmentCacheKey");
	return ReplaySegmentCacheKey{
		.passSequenceHash = segment.identity.passSequenceHash,
		.passCount = segment.identity.passCount,
	};
}

RenderGraph::ReplaySegmentVariantKey RenderGraph::BuildReplaySegmentVariantKey(const CachedReplaySegment& segment) const
{
	ZoneScopedN("RenderGraph::BuildReplaySegmentVariantKey");
	uint64_t hardBoundaryHash = 0x736567626f756e64ull;
	hardBoundaryHash = HashCombine64(hardBoundaryHash, segment.schedule.boundaryInputEdgeCount);
	hardBoundaryHash = HashCombine64(hardBoundaryHash, segment.schedule.boundaryOutputEdgeCount);
	hardBoundaryHash = HashCombine64(hardBoundaryHash, segment.schedule.crossQueueBoundaryInputEdgeCount);
	hardBoundaryHash = HashCombine64(hardBoundaryHash, segment.schedule.crossQueueBoundaryOutputEdgeCount);
	hardBoundaryHash = HashCombine64(hardBoundaryHash, segment.schedule.sameBatchPrefixPassCount);
	hardBoundaryHash = HashCombine64(hardBoundaryHash, segment.schedule.sameBatchSuffixPassCount);
	hardBoundaryHash = HashCombine64(hardBoundaryHash, segment.schedule.sameBatchInterleavedPassCount);
	hardBoundaryHash = HashCombine64(hardBoundaryHash, segment.schedule.crossQueueBoundaryPassCount);
	hardBoundaryHash = HashCombine64(hardBoundaryHash, segment.schedule.crossQueueTransitionCount);
	for (const auto& edge : segment.contract.boundaryEdges) {
		hardBoundaryHash = HashCombine64(hardBoundaryHash, edge.insideNode);
		hardBoundaryHash = HashCombine64(hardBoundaryHash, edge.outsideNode);
		hardBoundaryHash = HashCombine64(hardBoundaryHash, edge.insideQueueSlot);
		hardBoundaryHash = HashCombine64(hardBoundaryHash, edge.outsideQueueSlot);
		hardBoundaryHash = HashCombine64(hardBoundaryHash, edge.incoming ? 1ull : 0ull);
		hardBoundaryHash = HashCombine64(hardBoundaryHash, edge.crossQueue ? 1ull : 0ull);
	}

	uint64_t hardTemplateHash = 0x736567746d706864ull;
	hardTemplateHash = HashCombine64(hardTemplateHash, segment.templateStats.batchCount);
	hardTemplateHash = HashCombine64(hardTemplateHash, segment.templateStats.partialBatchCount);
	hardTemplateHash = HashCombine64(hardTemplateHash, segment.templateStats.queuedPassCount);
	hardTemplateHash = HashCombine64(hardTemplateHash, segment.templateStats.transitionCount);
	hardTemplateHash = HashCombine64(hardTemplateHash, segment.templateStats.passOrderHash);
	auto hashBound = [](uint64_t seed, const Bound& bound) {
		seed = HashCombine64(seed, static_cast<uint64_t>(bound.type));
		seed = HashCombine64(seed, bound.value);
		return seed;
	};
	auto hashRange = [&](uint64_t seed, const RangeSpec& range) {
		seed = hashBound(seed, range.mipLower);
		seed = hashBound(seed, range.mipUpper);
		seed = hashBound(seed, range.sliceLower);
		seed = hashBound(seed, range.sliceUpper);
		return seed;
	};
	auto hashState = [](uint64_t seed, const ResourceState& state) {
		seed = HashCombine64(seed, static_cast<uint64_t>(state.access));
		seed = HashCombine64(seed, static_cast<uint64_t>(state.layout));
		seed = HashCombine64(seed, static_cast<uint64_t>(state.sync));
		return seed;
	};
	for (const auto& batchTemplate : segment.batchTemplates) {
		hardTemplateHash = HashCombine64(hardTemplateHash, batchTemplate.partialBatch ? 1ull : 0ull);
		hardTemplateHash = HashCombine64(hardTemplateHash, batchTemplate.queuedPasses.size());
		hardTemplateHash = HashCombine64(hardTemplateHash, batchTemplate.transitions.size());
		for (const auto& queuedPass : batchTemplate.queuedPasses) {
			hardTemplateHash = HashCombine64(hardTemplateHash, queuedPass.passNameHash);
			hardTemplateHash = HashCombine64(hardTemplateHash, queuedPass.queueSlot);
			hardTemplateHash = HashCombine64(hardTemplateHash, static_cast<uint64_t>(queuedPass.type));
		}
		for (const auto& transition : batchTemplate.transitions) {
			// Stable graph resources are part of the replay contract. Dynamic wrappers
			// may legitimately rotate backing resources, so keep those relaxed.
			if (!transition.dynamicResource) {
				hardTemplateHash = HashCombine64(hardTemplateHash, transition.resourceID);
				hardTemplateHash = HashCombine64(hardTemplateHash, transition.backingResourceID);
			}
			hardTemplateHash = hashRange(hardTemplateHash, transition.range);
			hardTemplateHash = hashState(hardTemplateHash, transition.before);
			hardTemplateHash = hashState(hardTemplateHash, transition.after);
			hardTemplateHash = HashCombine64(hardTemplateHash, transition.queueSlot);
			hardTemplateHash = HashCombine64(hardTemplateHash, static_cast<uint64_t>(transition.phase));
			hardTemplateHash = HashCombine64(hardTemplateHash, transition.discard ? 1ull : 0ull);
			hardTemplateHash = HashCombine64(hardTemplateHash, transition.dynamicResource ? 1ull : 0ull);
		}
	}

	return ReplaySegmentVariantKey{
		.declarationHash = segment.fingerprint.declarationHash,
		.accessHash = segment.fingerprint.accessHash,
		.queueHash = segment.fingerprint.queueHash,
		// Kept for diagnostics, but replay lookup intentionally does not require equality here.
		// Alias pool placement can churn while the boundary and transition template remain replayable.
		.aliasHash = segment.fingerprint.aliasHash,
		.hardBoundaryHash = hardBoundaryHash,
		.hardTemplateHash = hardTemplateHash,
	};
}

bool RenderGraph::ReplaySegmentBoundaryEdgesMatch(const CachedReplaySegment& cached, const CachedReplaySegment& current) const
{
	if (cached.contract.boundaryEdges.size() != current.contract.boundaryEdges.size()) {
		return false;
	}
	for (size_t edgeIndex = 0; edgeIndex < cached.contract.boundaryEdges.size(); ++edgeIndex) {
		const auto& lhs = cached.contract.boundaryEdges[edgeIndex];
		const auto& rhs = current.contract.boundaryEdges[edgeIndex];
		if (lhs.insideNode != rhs.insideNode
			|| lhs.outsideNode != rhs.outsideNode
			|| lhs.insideQueueSlot != rhs.insideQueueSlot
			|| lhs.outsideQueueSlot != rhs.outsideQueueSlot
			|| lhs.incoming != rhs.incoming
			|| lhs.crossQueue != rhs.crossQueue) {
			return false;
		}
	}
	return cached.schedule.boundaryInputEdgeCount == current.schedule.boundaryInputEdgeCount
		&& cached.schedule.boundaryOutputEdgeCount == current.schedule.boundaryOutputEdgeCount
		&& cached.schedule.crossQueueBoundaryInputEdgeCount == current.schedule.crossQueueBoundaryInputEdgeCount
		&& cached.schedule.crossQueueBoundaryOutputEdgeCount == current.schedule.crossQueueBoundaryOutputEdgeCount
		&& cached.schedule.sameBatchPrefixPassCount == current.schedule.sameBatchPrefixPassCount
		&& cached.schedule.sameBatchSuffixPassCount == current.schedule.sameBatchSuffixPassCount
		&& cached.schedule.sameBatchInterleavedPassCount == current.schedule.sameBatchInterleavedPassCount
		&& cached.schedule.crossQueueBoundaryPassCount == current.schedule.crossQueueBoundaryPassCount
		&& cached.schedule.crossQueueTransitionCount == current.schedule.crossQueueTransitionCount;
}

bool RenderGraph::ReplaySegmentHardTemplateMatches(const CachedReplaySegment& cached, const CachedReplaySegment& current) const
{
	if (cached.templateStats.batchCount != current.templateStats.batchCount
		|| cached.templateStats.partialBatchCount != current.templateStats.partialBatchCount
		|| cached.templateStats.queuedPassCount != current.templateStats.queuedPassCount
		|| cached.templateStats.transitionCount != current.templateStats.transitionCount
		|| cached.templateStats.passOrderHash != current.templateStats.passOrderHash
		|| cached.batchTemplates.size() != current.batchTemplates.size()) {
		return false;
	}
	for (size_t batchIndex = 0; batchIndex < cached.batchTemplates.size(); ++batchIndex) {
		const auto& lhsBatch = cached.batchTemplates[batchIndex];
		const auto& rhsBatch = current.batchTemplates[batchIndex];
		if (lhsBatch.partialBatch != rhsBatch.partialBatch
			|| lhsBatch.queuedPasses.size() != rhsBatch.queuedPasses.size()
			|| lhsBatch.transitions.size() != rhsBatch.transitions.size()) {
			return false;
		}
		for (size_t passIndex = 0; passIndex < lhsBatch.queuedPasses.size(); ++passIndex) {
			const auto& lhs = lhsBatch.queuedPasses[passIndex];
			const auto& rhs = rhsBatch.queuedPasses[passIndex];
			if (lhs.passNameHash != rhs.passNameHash
				|| lhs.queueSlot != rhs.queueSlot
				|| lhs.type != rhs.type) {
				return false;
			}
		}
		for (size_t transitionIndex = 0; transitionIndex < lhsBatch.transitions.size(); ++transitionIndex) {
			const auto& lhs = lhsBatch.transitions[transitionIndex];
			const auto& rhs = rhsBatch.transitions[transitionIndex];
			if ((!lhs.dynamicResource || !rhs.dynamicResource)
				&& (lhs.resourceID != rhs.resourceID || lhs.backingResourceID != rhs.backingResourceID)) {
				return false;
			}
			if (lhs.range.mipLower != rhs.range.mipLower
				|| lhs.range.mipUpper != rhs.range.mipUpper
				|| lhs.range.sliceLower != rhs.range.sliceLower
				|| lhs.range.sliceUpper != rhs.range.sliceUpper
				|| lhs.before != rhs.before
				|| lhs.after != rhs.after
				|| lhs.queueSlot != rhs.queueSlot
				|| lhs.phase != rhs.phase
				|| lhs.discard != rhs.discard
				|| lhs.dynamicResource != rhs.dynamicResource) {
				return false;
			}
		}
	}
	return true;
}

bool RenderGraph::ReplaySegmentSyncShapeDiverged(const CachedReplaySegment& cached, const CachedReplaySegment& current) const
{
	return cached.contract.boundarySyncs.size() != current.contract.boundarySyncs.size()
		|| cached.templateStats.waitCount != current.templateStats.waitCount
		|| cached.templateStats.signalCount != current.templateStats.signalCount
		|| cached.templateStats.syncShapeHash != current.templateStats.syncShapeHash;
}

bool RenderGraph::ReplaySegmentHardReplayMatches(const CachedReplaySegment& cached, const CachedReplaySegment& current) const
{
	ZoneScopedN("RenderGraph::ReplaySegmentHardReplayMatches");
	const bool relaxAliasPlacement = m_renderGraphSettingsService
		? m_renderGraphSettingsService->GetRenderGraphReplayRelaxAliasPlacement()
		: true;
	return cached.fingerprint.declarationHash == current.fingerprint.declarationHash
		&& cached.fingerprint.accessHash == current.fingerprint.accessHash
		&& cached.fingerprint.queueHash == current.fingerprint.queueHash
		&& (relaxAliasPlacement || cached.fingerprint.aliasHash == current.fingerprint.aliasHash)
		&& ReplaySegmentBoundaryEdgesMatch(cached, current)
		&& ReplaySegmentHardTemplateMatches(cached, current);
}

std::string RenderGraph::FormatReplaySegmentBoundaryDiff(const CachedReplaySegment& cached, const CachedReplaySegment& current) const
{
	auto boundaryCounts = [](const CachedReplaySegment& segment) {
		struct Counts {
			uint64_t incoming = 0;
			uint64_t outgoing = 0;
			uint64_t crossQueueIncoming = 0;
			uint64_t crossQueueOutgoing = 0;
		};
		Counts counts{};
		for (const auto& edge : segment.contract.boundaryEdges) {
			if (edge.incoming) {
				++counts.incoming;
				counts.crossQueueIncoming += edge.crossQueue ? 1ull : 0ull;
			}
			else {
				++counts.outgoing;
				counts.crossQueueOutgoing += edge.crossQueue ? 1ull : 0ull;
			}
		}
		return counts;
	};

	std::ostringstream oss;
	const auto cachedCounts = boundaryCounts(cached);
	const auto currentCounts = boundaryCounts(current);
	oss << "edges=" << cached.contract.boundaryEdges.size() << "->" << current.contract.boundaryEdges.size()
		<< " incoming=" << cachedCounts.incoming << "->" << currentCounts.incoming
		<< " outgoing=" << cachedCounts.outgoing << "->" << currentCounts.outgoing
		<< " cross_in=" << cachedCounts.crossQueueIncoming << "->" << currentCounts.crossQueueIncoming
		<< " cross_out=" << cachedCounts.crossQueueOutgoing << "->" << currentCounts.crossQueueOutgoing
		<< " syncs=" << cached.contract.boundarySyncs.size() << "->" << current.contract.boundarySyncs.size()
		<< " region_counts={in:" << cached.schedule.boundaryInputEdgeCount << "->" << current.schedule.boundaryInputEdgeCount
		<< ",out:" << cached.schedule.boundaryOutputEdgeCount << "->" << current.schedule.boundaryOutputEdgeCount
		<< ",cross_in:" << cached.schedule.crossQueueBoundaryInputEdgeCount << "->" << current.schedule.crossQueueBoundaryInputEdgeCount
		<< ",cross_out:" << cached.schedule.crossQueueBoundaryOutputEdgeCount << "->" << current.schedule.crossQueueBoundaryOutputEdgeCount
		<< ",prefix:" << cached.schedule.sameBatchPrefixPassCount << "->" << current.schedule.sameBatchPrefixPassCount
		<< ",suffix:" << cached.schedule.sameBatchSuffixPassCount << "->" << current.schedule.sameBatchSuffixPassCount
		<< ",cross_trans:" << cached.schedule.crossQueueTransitionCount << "->" << current.schedule.crossQueueTransitionCount
		<< "}";
	const size_t edgeCount = std::min(cached.contract.boundaryEdges.size(), current.contract.boundaryEdges.size());
	for (size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex) {
		const auto& lhs = cached.contract.boundaryEdges[edgeIndex];
		const auto& rhs = current.contract.boundaryEdges[edgeIndex];
		if (lhs.insideNode == rhs.insideNode
			&& lhs.outsideNode == rhs.outsideNode
			&& lhs.insideTraceIndex == rhs.insideTraceIndex
			&& lhs.outsideTraceIndex == rhs.outsideTraceIndex
			&& lhs.insideQueueSlot == rhs.insideQueueSlot
			&& lhs.outsideQueueSlot == rhs.outsideQueueSlot
			&& lhs.incoming == rhs.incoming
			&& lhs.crossQueue == rhs.crossQueue) {
			continue;
		}
		oss << " first_edge_diff=index=" << edgeIndex
			<< " inside_node=" << lhs.insideNode << "->" << rhs.insideNode
			<< " outside_node=" << lhs.outsideNode << "->" << rhs.outsideNode
			<< " inside_trace=" << lhs.insideTraceIndex << "->" << rhs.insideTraceIndex
			<< " outside_trace=" << lhs.outsideTraceIndex << "->" << rhs.outsideTraceIndex
			<< " inside_queue=" << lhs.insideQueueSlot << "->" << rhs.insideQueueSlot
			<< " outside_queue=" << lhs.outsideQueueSlot << "->" << rhs.outsideQueueSlot
			<< " incoming=" << lhs.incoming << "->" << rhs.incoming
			<< " cross_queue=" << lhs.crossQueue << "->" << rhs.crossQueue;
		break;
	}
	return oss.str();
}

std::string RenderGraph::FormatReplaySegmentTemplateDiff(const CachedReplaySegment& cached, const CachedReplaySegment& current) const
{
	std::ostringstream oss;
	oss << "batches=" << cached.templateStats.batchCount << "->" << current.templateStats.batchCount
		<< " partial=" << cached.templateStats.partialBatchCount << "->" << current.templateStats.partialBatchCount
		<< " queued_passes=" << cached.templateStats.queuedPassCount << "->" << current.templateStats.queuedPassCount
		<< " transitions=" << cached.templateStats.transitionCount << "->" << current.templateStats.transitionCount
		<< " waits=" << cached.templateStats.waitCount << "->" << current.templateStats.waitCount
		<< " signals=" << cached.templateStats.signalCount << "->" << current.templateStats.signalCount
		<< " pass_order=0x" << std::hex << cached.templateStats.passOrderHash << "->0x" << current.templateStats.passOrderHash
		<< " transition_shape=0x" << cached.templateStats.transitionShapeHash << "->0x" << current.templateStats.transitionShapeHash
		<< " sync_shape=0x" << cached.templateStats.syncShapeHash << "->0x" << current.templateStats.syncShapeHash << std::dec;
	const size_t batchCount = std::min(cached.batchTemplates.size(), current.batchTemplates.size());
	for (size_t batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
		const auto& lhsBatch = cached.batchTemplates[batchIndex];
		const auto& rhsBatch = current.batchTemplates[batchIndex];
		if (lhsBatch.partialBatch != rhsBatch.partialBatch
			|| lhsBatch.queuedPasses.size() != rhsBatch.queuedPasses.size()
			|| lhsBatch.transitions.size() != rhsBatch.transitions.size()
			|| lhsBatch.waits.size() != rhsBatch.waits.size()
			|| lhsBatch.signals.size() != rhsBatch.signals.size()) {
			oss << " first_batch_shape_diff=local_batch=" << batchIndex
				<< " original=" << lhsBatch.originalBatchIndexAtExtraction << "->" << rhsBatch.originalBatchIndexAtExtraction
				<< " partial=" << lhsBatch.partialBatch << "->" << rhsBatch.partialBatch
				<< " queued=" << lhsBatch.queuedPasses.size() << "->" << rhsBatch.queuedPasses.size()
				<< " transitions=" << lhsBatch.transitions.size() << "->" << rhsBatch.transitions.size()
				<< " waits=" << lhsBatch.waits.size() << "->" << rhsBatch.waits.size()
				<< " signals=" << lhsBatch.signals.size() << "->" << rhsBatch.signals.size();
			return oss.str();
		}
		for (size_t transitionIndex = 0; transitionIndex < lhsBatch.transitions.size(); ++transitionIndex) {
			const auto& lhs = lhsBatch.transitions[transitionIndex];
			const auto& rhs = rhsBatch.transitions[transitionIndex];
			if (lhs.resourceID == rhs.resourceID
				&& lhs.range.mipLower == rhs.range.mipLower
				&& lhs.range.mipUpper == rhs.range.mipUpper
				&& lhs.range.sliceLower == rhs.range.sliceLower
				&& lhs.range.sliceUpper == rhs.range.sliceUpper
				&& lhs.after == rhs.after
				&& lhs.queueSlot == rhs.queueSlot
				&& lhs.phase == rhs.phase
				&& lhs.discard == rhs.discard
				&& lhs.dynamicResource == rhs.dynamicResource) {
				continue;
			}
			oss << " first_transition_diff=local_batch=" << batchIndex
				<< " transition=" << transitionIndex
				<< " resource=" << lhs.resourceID << "->" << rhs.resourceID
				<< " dynamic=" << lhs.dynamicResource << "->" << rhs.dynamicResource
				<< " backing=" << lhs.backingResourceID << "->" << rhs.backingResourceID
				<< " queue=" << lhs.queueSlot << "->" << rhs.queueSlot
				<< " phase=" << static_cast<uint32_t>(lhs.phase) << "->" << static_cast<uint32_t>(rhs.phase)
				<< " discard=" << lhs.discard << "->" << rhs.discard
				<< " range=\"" << FormatRangeSpec(lhs.range) << "\"->\"" << FormatRangeSpec(rhs.range) << "\"";
			return oss.str();
		}
	}
	if (cached.batchTemplates.size() != current.batchTemplates.size()) {
		oss << " batch_template_count=" << cached.batchTemplates.size() << "->" << current.batchTemplates.size();
	}
	return oss.str();
}

RenderGraph::ReplaySegmentLookupResult RenderGraph::LookupCachedReplaySegmentVariant(
	const CachedReplaySegment& currentSegment,
	uint64_t frameIndex)
{
	ZoneScopedN("RenderGraph::LookupCachedReplaySegmentVariant");
	ReplaySegmentLookupResult result{};
	const ReplaySegmentCacheKey key = BuildReplaySegmentCacheKey(currentSegment);
	const ReplaySegmentVariantKey currentVariantKey = BuildReplaySegmentVariantKey(currentSegment);
	ReplaySegmentCacheEntry* entry = nullptr;
	for (auto& candidateEntry : m_regionCache.replaySegmentEntries) {
		if (candidateEntry.key.passSequenceHash == key.passSequenceHash
			&& candidateEntry.key.passCount == key.passCount) {
			entry = &candidateEntry;
			break;
		}
	}
	if (entry == nullptr) {
		std::ostringstream oss;
		oss << "missing_cache_entry pass_sequence=0x" << std::hex << key.passSequenceHash
			<< " pass_count=" << std::dec << key.passCount;
		result.missReason = oss.str();
		return result;
	}

	CachedReplaySegmentVariant* bestVariant = nullptr;
	const bool relaxAliasPlacement = m_renderGraphSettingsService
		? m_renderGraphSettingsService->GetRenderGraphReplayRelaxAliasPlacement()
		: true;
	for (auto& variant : entry->variants) {
		if (variant.variantKey.declarationHash != currentVariantKey.declarationHash
			|| variant.variantKey.accessHash != currentVariantKey.accessHash
			|| variant.variantKey.queueHash != currentVariantKey.queueHash
			|| (!relaxAliasPlacement && variant.variantKey.aliasHash != currentVariantKey.aliasHash)
			|| variant.variantKey.hardBoundaryHash != currentVariantKey.hardBoundaryHash
			|| variant.variantKey.hardTemplateHash != currentVariantKey.hardTemplateHash
			|| !ReplaySegmentHardReplayMatches(variant.segment, currentSegment)) {
			continue;
		}
		if (bestVariant == nullptr
			|| variant.lastSeenFrame > bestVariant->lastSeenFrame
			|| (variant.lastSeenFrame == bestVariant->lastSeenFrame && variant.hitCount > bestVariant->hitCount)) {
			bestVariant = &variant;
		}
	}

	if (bestVariant != nullptr) {
		const uint64_t variantAgeFrames = frameIndex >= bestVariant->lastSeenFrame ? frameIndex - bestVariant->lastSeenFrame : 0;
		++bestVariant->hitCount;
		++bestVariant->seenCount;
		bestVariant->lastSeenFrame = frameIndex;
		result.variant = bestVariant;
		result.syncShapeDiverged = ReplaySegmentSyncShapeDiverged(bestVariant->segment, currentSegment);
		result.variantAgeFrames = variantAgeFrames;
		return result;
	}

	const CachedReplaySegmentVariant* newestVariant = nullptr;
	for (const auto& variant : entry->variants) {
		if (newestVariant == nullptr || variant.lastSeenFrame > newestVariant->lastSeenFrame) {
			newestVariant = &variant;
		}
	}
	if (newestVariant != nullptr) {
		std::ostringstream oss;
		oss << "no_matching_variant pass_sequence=0x" << std::hex << key.passSequenceHash
			<< " declaration=0x" << newestVariant->variantKey.declarationHash << "->0x" << currentVariantKey.declarationHash
			<< " access=0x" << newestVariant->variantKey.accessHash << "->0x" << currentVariantKey.accessHash
			<< " queue=0x" << newestVariant->variantKey.queueHash << "->0x" << currentVariantKey.queueHash
			<< " alias=0x" << newestVariant->variantKey.aliasHash << "->0x" << currentVariantKey.aliasHash
			<< " boundary=0x" << newestVariant->variantKey.hardBoundaryHash << "->0x" << currentVariantKey.hardBoundaryHash
			<< " template=0x" << newestVariant->variantKey.hardTemplateHash << "->0x" << currentVariantKey.hardTemplateHash;
		result.missReason = oss.str();
		result.boundaryDiff = FormatReplaySegmentBoundaryDiff(newestVariant->segment, currentSegment);
		result.templateDiff = FormatReplaySegmentTemplateDiff(newestVariant->segment, currentSegment);
	}
	return result;
}

RenderGraph::ReplaySegmentCacheUpdateStats RenderGraph::SummarizeReplaySegmentVariantCache() const
{
	ZoneScopedN("RenderGraph::SummarizeReplaySegmentVariantCache");
	ReplaySegmentCacheUpdateStats stats{};
	stats.entries = m_regionCache.replaySegmentEntries.size();
	for (const auto& entry : m_regionCache.replaySegmentEntries) {
		stats.variants += entry.variants.size();
	}
	return stats;
}

void RenderGraph::EvictOldReplaySegmentVariants(uint64_t frameIndex, ReplaySegmentCacheUpdateStats& stats)
{
	ZoneScopedN("RenderGraph::EvictOldReplaySegmentVariants");
	const uint64_t maxVariantAgeFrames = m_renderGraphSettingsService
		? static_cast<uint64_t>(m_renderGraphSettingsService->GetRenderGraphReplaySegmentCacheMaxAgeFrames())
		: 0ull;
	const size_t maxVariantsPerKey = m_renderGraphSettingsService
		? static_cast<size_t>(m_renderGraphSettingsService->GetRenderGraphReplaySegmentCacheMaxVariantsPerKey())
		: 32ull;
	const size_t maxCacheEntries = m_renderGraphSettingsService
		? static_cast<size_t>(m_renderGraphSettingsService->GetRenderGraphReplaySegmentCacheMaxEntries())
		: 256ull;
	const size_t maxTotalVariants = m_renderGraphSettingsService
		? static_cast<size_t>(m_renderGraphSettingsService->GetRenderGraphReplaySegmentCacheMaxVariants())
		: 128ull;
	const size_t effectiveMaxVariantsPerKey = (std::max)(size_t{ 1 }, maxVariantsPerKey);
	const size_t effectiveMaxCacheEntries = (std::max)(size_t{ 1 }, maxCacheEntries);
	const size_t effectiveMaxTotalVariants = (std::max)(size_t{ 1 }, maxTotalVariants);

	auto variantUseScore = [](const CachedReplaySegmentVariant& variant) {
		return variant.hitCount + variant.seenCount;
	};
	auto newestSeen = [](const ReplaySegmentCacheEntry& entry) {
		uint64_t newest = 0;
		for (const auto& variant : entry.variants) {
			newest = (std::max)(newest, variant.lastSeenFrame);
		}
		return newest;
	};
	auto totalVariantCount = [&]() {
		size_t total = 0;
		for (const auto& entry : m_regionCache.replaySegmentEntries) {
			total += entry.variants.size();
		}
		return total;
	};
	auto eraseEmptyEntries = [&]() {
		m_regionCache.replaySegmentEntries.erase(
			std::remove_if(m_regionCache.replaySegmentEntries.begin(), m_regionCache.replaySegmentEntries.end(), [](const ReplaySegmentCacheEntry& entry) {
				return entry.variants.empty();
			}),
			m_regionCache.replaySegmentEntries.end());
	};
	auto evictOldestVariant = [&]() -> bool {
		size_t bestEntryIndex = std::numeric_limits<size_t>::max();
		size_t bestVariantIndex = std::numeric_limits<size_t>::max();
		for (size_t entryIndex = 0; entryIndex < m_regionCache.replaySegmentEntries.size(); ++entryIndex) {
			const auto& entry = m_regionCache.replaySegmentEntries[entryIndex];
			for (size_t variantIndex = 0; variantIndex < entry.variants.size(); ++variantIndex) {
				const auto& candidate = entry.variants[variantIndex];
				if (bestEntryIndex == std::numeric_limits<size_t>::max()) {
					bestEntryIndex = entryIndex;
					bestVariantIndex = variantIndex;
					continue;
				}
				const auto& best = m_regionCache.replaySegmentEntries[bestEntryIndex].variants[bestVariantIndex];
				if (candidate.lastSeenFrame != best.lastSeenFrame) {
					if (candidate.lastSeenFrame < best.lastSeenFrame) {
						bestEntryIndex = entryIndex;
						bestVariantIndex = variantIndex;
					}
					continue;
				}
				if (variantUseScore(candidate) < variantUseScore(best)) {
					bestEntryIndex = entryIndex;
					bestVariantIndex = variantIndex;
				}
			}
		}
		if (bestEntryIndex == std::numeric_limits<size_t>::max()) {
			return false;
		}
		auto& variants = m_regionCache.replaySegmentEntries[bestEntryIndex].variants;
		variants.erase(variants.begin() + static_cast<std::ptrdiff_t>(bestVariantIndex));
		++stats.evicted;
		return true;
	};

	for (auto& entry : m_regionCache.replaySegmentEntries) {
		if (maxVariantAgeFrames > 0) {
			const auto oldSize = entry.variants.size();
			entry.variants.erase(
				std::remove_if(entry.variants.begin(), entry.variants.end(), [&](const CachedReplaySegmentVariant& variant) {
					return frameIndex > variant.lastSeenFrame
						&& frameIndex - variant.lastSeenFrame > maxVariantAgeFrames;
				}),
				entry.variants.end());
			stats.evicted += oldSize - entry.variants.size();
		}
		if (entry.variants.size() > effectiveMaxVariantsPerKey) {
			std::sort(entry.variants.begin(), entry.variants.end(), [](const auto& lhs, const auto& rhs) {
				if (lhs.lastSeenFrame != rhs.lastSeenFrame) {
					return lhs.lastSeenFrame > rhs.lastSeenFrame;
				}
				return lhs.hitCount + lhs.seenCount > rhs.hitCount + rhs.seenCount;
			});
			stats.evicted += entry.variants.size() - effectiveMaxVariantsPerKey;
			entry.variants.resize(effectiveMaxVariantsPerKey);
		}
	}

	eraseEmptyEntries();

	while (totalVariantCount() > effectiveMaxTotalVariants && evictOldestVariant()) {
		eraseEmptyEntries();
	}

	if (m_regionCache.replaySegmentEntries.size() > effectiveMaxCacheEntries) {
		std::sort(m_regionCache.replaySegmentEntries.begin(), m_regionCache.replaySegmentEntries.end(), [&](const auto& lhs, const auto& rhs) {
			const uint64_t lhsNewestSeen = newestSeen(lhs);
			const uint64_t rhsNewestSeen = newestSeen(rhs);
			if (lhsNewestSeen != rhsNewestSeen) {
				return lhsNewestSeen > rhsNewestSeen;
			}
			return lhs.variants.size() > rhs.variants.size();
		});
		stats.evicted += m_regionCache.replaySegmentEntries.size() - effectiveMaxCacheEntries;
		m_regionCache.replaySegmentEntries.resize(effectiveMaxCacheEntries);
	}
}

RenderGraph::ReplaySegmentCacheUpdateStats RenderGraph::InsertOrRefreshReplaySegmentVariants(
	std::span<const CachedReplaySegment> currentSegments,
	uint64_t frameIndex)
{
	ZoneScopedN("RenderGraph::InsertOrRefreshReplaySegmentVariants");
	ReplaySegmentCacheUpdateStats stats{};
	for (const auto& segment : currentSegments) {
		if (!segment.tier1Eligible || segment.batchTemplates.empty()) {
			continue;
		}
		const ReplaySegmentCacheKey key = BuildReplaySegmentCacheKey(segment);
		const ReplaySegmentVariantKey variantKey = BuildReplaySegmentVariantKey(segment);
		const bool relaxAliasPlacement = m_renderGraphSettingsService
			? m_renderGraphSettingsService->GetRenderGraphReplayRelaxAliasPlacement()
			: true;
		auto entryIt = std::find_if(m_regionCache.replaySegmentEntries.begin(), m_regionCache.replaySegmentEntries.end(), [&](const ReplaySegmentCacheEntry& entry) {
			return entry.key.passSequenceHash == key.passSequenceHash && entry.key.passCount == key.passCount;
		});
		if (entryIt == m_regionCache.replaySegmentEntries.end()) {
			entryIt = m_regionCache.replaySegmentEntries.insert(
				m_regionCache.replaySegmentEntries.end(),
				ReplaySegmentCacheEntry{ .key = key });
		}

		auto variantIt = std::find_if(entryIt->variants.begin(), entryIt->variants.end(), [&](const CachedReplaySegmentVariant& variant) {
			return variant.variantKey.declarationHash == variantKey.declarationHash
				&& variant.variantKey.accessHash == variantKey.accessHash
				&& variant.variantKey.queueHash == variantKey.queueHash
				&& (relaxAliasPlacement || variant.variantKey.aliasHash == variantKey.aliasHash)
				&& variant.variantKey.hardBoundaryHash == variantKey.hardBoundaryHash
				&& variant.variantKey.hardTemplateHash == variantKey.hardTemplateHash
				&& ReplaySegmentHardReplayMatches(variant.segment, segment);
		});
		if (variantIt == entryIt->variants.end()) {
			entryIt->variants.push_back(CachedReplaySegmentVariant{
				.segment = segment,
				.variantKey = variantKey,
				.firstSeenFrame = frameIndex,
				.lastSeenFrame = frameIndex,
				.hitCount = 0,
				.seenCount = 1,
			});
			++stats.inserted;
		}
		else {
			variantIt->segment = segment;
			variantIt->variantKey = variantKey;
			variantIt->lastSeenFrame = frameIndex;
			++variantIt->seenCount;
			++stats.refreshed;
		}
	}

	EvictOldReplaySegmentVariants(frameIndex, stats);
	const ReplaySegmentCacheUpdateStats summary = SummarizeReplaySegmentVariantCache();
	stats.entries = summary.entries;
	stats.variants = summary.variants;
	return stats;
}

RenderGraph::ReplaySegmentVerificationReport RenderGraph::VerifyAuthoritativeScheduleSemantics(
	const std::vector<Node>& nodes,
	const RenderGraph::FramePassList& framePasses,
	const std::vector<PassBatch>& compiledBatches) const
{
	ZoneScopedN("RenderGraph::VerifyAuthoritativeScheduleSemantics");
	ReplaySegmentVerificationReport report{};
	std::vector<size_t> traceIndexByNodeIndex(nodes.size(), std::numeric_limits<size_t>::max());
	for (size_t traceIndex = 0; traceIndex < m_schedulingDecisionTrace.size(); ++traceIndex) {
		const auto& trace = m_schedulingDecisionTrace[traceIndex];
		if (trace.nodeIndex >= nodes.size() || trace.passIndex >= framePasses.size()) {
			report.valid = false;
			++report.failures;
			if (report.firstFailure.empty()) {
				report.firstFailure = "trace entry references invalid node/pass";
			}
			continue;
		}
		traceIndexByNodeIndex[trace.nodeIndex] = traceIndex;
		++report.checkedPasses;
		if (trace.passIndex < m_framePassSchedulingSummaries.size()) {
			report.checkedRequirements += m_framePassSchedulingSummaries[trace.passIndex].requirements.size();
		}
	}

	for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
		const size_t nodeTrace = nodeIndex < traceIndexByNodeIndex.size() ? traceIndexByNodeIndex[nodeIndex] : std::numeric_limits<size_t>::max();
		if (nodeTrace == std::numeric_limits<size_t>::max()) {
			report.valid = false;
			++report.failures;
			if (report.firstFailure.empty()) {
				report.firstFailure = "node missing from scheduling trace";
			}
			continue;
		}
		for (size_t pred : nodes[nodeIndex].in) {
			++report.checkedEdges;
			if (pred >= traceIndexByNodeIndex.size()
				|| traceIndexByNodeIndex[pred] == std::numeric_limits<size_t>::max()
				|| traceIndexByNodeIndex[pred] >= nodeTrace) {
				report.valid = false;
				++report.failures;
				if (report.firstFailure.empty()) {
					std::ostringstream oss;
					oss << "DAG order violation node=" << nodeIndex << " pred=" << pred;
					report.firstFailure = oss.str();
				}
			}
		}
	}

	std::unordered_map<const void*, size_t> passIndexByPointer;
	passIndexByPointer.reserve(framePasses.size());
	for (size_t passIndex = 0; passIndex < framePasses.size(); ++passIndex) {
		std::visit(
			[&](const auto& passEntry) {
				using T = std::decay_t<decltype(passEntry)>;
				if constexpr (!std::is_same_v<T, std::monostate>) {
					passIndexByPointer.emplace(static_cast<const void*>(&passEntry), passIndex);
				}
			},
			framePasses[passIndex].pass);
	}

	std::unordered_map<uint64_t, SymbolicTracker> verificationTrackers;
	std::vector<ResourceTransition> generatedTransitions;
	std::vector<ResourceTransition> ignoredTransitionsForVerification;
	auto transitionMatches = [&](const ResourceTransition& generated, const ResourceTransition& emitted) {
		return generated.pResource == emitted.pResource
			&& generated.discard == emitted.discard
			&& generated.range.mipLower == emitted.range.mipLower
			&& generated.range.mipUpper == emitted.range.mipUpper
			&& generated.range.sliceLower == emitted.range.sliceLower
			&& generated.range.sliceUpper == emitted.range.sliceUpper
			&& generated.prevAccessType == emitted.prevAccessType
			&& generated.newAccessType == emitted.newAccessType
			&& generated.prevLayout == emitted.prevLayout
			&& generated.newLayout == emitted.newLayout
			&& generated.prevSyncState == emitted.prevSyncState
			&& generated.newSyncState == emitted.newSyncState;
	};
	auto getVerificationTracker = [&](Resource* resource) -> SymbolicTracker* {
		if (!resource) {
			return nullptr;
		}
		const uint64_t resourceID = resource->GetGlobalResourceID();
		auto [it, inserted] = verificationTrackers.try_emplace(resourceID);
		if (inserted) {
			it->second = SeedCompileTrackerFromLiveResource(resource);
		}
		return &it->second;
	};
	auto failVerification = [&](std::string failure) {
		report.valid = false;
		++report.failures;
		if (report.firstFailure.empty()) {
			report.firstFailure = std::move(failure);
		}
	};
	auto verifyTransitionList = [&](const std::vector<ResourceTransition>& transitions, size_t batchIndex, size_t queueIndex, BatchTransitionPhase phase) {
		for (const auto& transition : transitions) {
			auto* tracker = getVerificationTracker(transition.pResource);
			if (!tracker) {
				std::ostringstream oss;
				oss << "transition has null resource batch=" << batchIndex << " queue=" << queueIndex;
				failVerification(oss.str());
				continue;
			}
			if (transition.discard) {
				tracker->Apply(
					transition.range,
					transition.pResource,
					ResourceState{ transition.newAccessType, transition.newLayout, transition.newSyncState },
					ignoredTransitionsForVerification);
				ignoredTransitionsForVerification.clear();
				continue;
			}
			SymbolicTracker probeTracker = *tracker;
			generatedTransitions.clear();
			probeTracker.Apply(
				transition.range,
				transition.pResource,
				ResourceState{ transition.newAccessType, transition.newLayout, transition.newSyncState },
				generatedTransitions);
			if (generatedTransitions.size() != 1 || !transitionMatches(generatedTransitions.front(), transition)) {
				std::ostringstream oss;
				oss << "transition before-state mismatch batch=" << batchIndex
					<< " queue=" << queueIndex
					<< " phase=" << (phase == BatchTransitionPhase::BeforePasses ? "BeforePasses" : "AfterPasses")
					<< " resource=" << transition.pResource->GetGlobalResourceID()
					<< " name=\"" << transition.pResource->GetName() << "\""
					<< " range=" << FormatRangeSpec(transition.range)
					<< " emitted=" << static_cast<uint32_t>(transition.prevLayout)
					<< "->" << static_cast<uint32_t>(transition.newLayout)
					<< " generated_count=" << generatedTransitions.size();
				if (!generatedTransitions.empty()) {
					const auto& generated = generatedTransitions.front();
					oss << " generated_first=" << static_cast<uint32_t>(generated.prevLayout)
						<< "->" << static_cast<uint32_t>(generated.newLayout)
						<< " generated_range=" << FormatRangeSpec(generated.range)
						<< " generated_discard=" << (generated.discard ? 1 : 0);
				}
				failVerification(oss.str());
				continue;
			}
			*tracker = std::move(probeTracker);
		}
	};
	auto verifyPassRequirements = [&](const auto* passEntry, size_t passIndex, size_t batchIndex, size_t queueIndex) {
		if (!passEntry || passIndex >= m_framePassSchedulingSummaries.size()) {
			return;
		}
		const QueueKind queueKind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(queueIndex));
		for (const auto& requirement : m_framePassSchedulingSummaries[passIndex].requirements) {
			Resource* resource = requirement.resource.IsEphemeral()
				? requirement.resource.GetEphemeralPtr()
				: const_cast<Resource*>(_registry.Resolve(requirement.resource));
			if (!resource) {
				continue;
			}
			auto* tracker = getVerificationTracker(resource);
			if (!tracker) {
				continue;
			}
			const ResourceState requiredState = NormalizeStateForQueue(queueKind, requirement.state);
			if (tracker->WouldModify(requirement.range, requiredState)) {
				std::ostringstream oss;
				oss << "pass requirement unsatisfied batch=" << batchIndex
					<< " queue=" << queueIndex
					<< " pass=\"" << framePasses[passIndex].name << "\""
					<< " resource=" << requirement.resourceID
					<< " name=\"" << resource->GetName() << "\""
					<< " range=" << FormatRangeSpec(requirement.range)
					<< " required_layout=" << static_cast<uint32_t>(requiredState.layout)
					<< " required_access=" << static_cast<uint32_t>(requiredState.access);
				failVerification(oss.str());
			}
			else {
				tracker->Apply(requirement.range, resource, requiredState, ignoredTransitionsForVerification);
				ignoredTransitionsForVerification.clear();
			}
		}

		const auto& denseTransitions = m_framePassSchedulingSummaries[passIndex].internalTransitions;
		if (passEntry->resources.internalTransitions.size() == denseTransitions.size()) {
			for (size_t transitionIndex = 0; transitionIndex < denseTransitions.size(); ++transitionIndex) {
				const auto& exit = passEntry->resources.internalTransitions[transitionIndex];
				const auto& denseTransition = denseTransitions[transitionIndex];
				Resource* resource = exit.first.resource.IsEphemeral()
					? exit.first.resource.GetEphemeralPtr()
					: const_cast<Resource*>(_registry.Resolve(exit.first.resource));
				if (!resource) {
					continue;
				}
				auto* tracker = getVerificationTracker(resource);
				if (!tracker) {
					continue;
				}
				tracker->Apply(exit.first.range, resource, exit.second, ignoredTransitionsForVerification);
				ignoredTransitionsForVerification.clear();
				(void)denseTransition;
			}
		}
	};

	for (size_t batchIndex = 0; batchIndex < compiledBatches.size(); ++batchIndex) {
		const auto& batch = compiledBatches[batchIndex];
		for (size_t dst = 0; dst < batch.QueueCount(); ++dst) {
			for (size_t src = 0; src < batch.QueueCount(); ++src) {
				if (dst == src) {
					continue;
				}
				for (size_t waitPhaseIndex = 0; waitPhaseIndex < PassBatch::kWaitPhaseCount; ++waitPhaseIndex) {
					const auto waitPhase = static_cast<BatchWaitPhase>(waitPhaseIndex);
					if (!batch.HasQueueWait(waitPhase, dst, src)) {
						continue;
					}
					++report.checkedQueueSyncs;
					if (batch.GetQueueWaitFenceValue(waitPhase, dst, src) == 0) {
						report.valid = false;
						++report.failures;
						if (report.firstFailure.empty()) {
							std::ostringstream oss;
							oss << "queue wait has zero fence batch=" << batchIndex << " dst=" << dst << " src=" << src;
							report.firstFailure = oss.str();
						}
					}
				}
			}
		}
		for (size_t queueIndex = 0; queueIndex < batch.QueueCount(); ++queueIndex) {
			verifyTransitionList(batch.Transitions(queueIndex, BatchTransitionPhase::BeforePasses), batchIndex, queueIndex, BatchTransitionPhase::BeforePasses);
			for (const auto& passVariant : batch.Passes(queueIndex)) {
				std::visit(
					[&](const auto* passEntry) {
						if (!passEntry) {
							return;
						}
						auto passIt = passIndexByPointer.find(static_cast<const void*>(passEntry));
						if (passIt == passIndexByPointer.end()) {
							failVerification("batch pass pointer not found in current frame pass list");
							return;
						}
						verifyPassRequirements(passEntry, passIt->second, batchIndex, queueIndex);
					},
					passVariant);
			}
			verifyTransitionList(batch.Transitions(queueIndex, BatchTransitionPhase::AfterPasses), batchIndex, queueIndex, BatchTransitionPhase::AfterPasses);
		}
	}

	return report;
}

RenderGraph::ReplaySegmentVerificationReport RenderGraph::BuildShadowReplayScheduleFromCachedSegments(
	std::span<const CachedReplaySegment> previousSegments,
	std::span<const CachedReplaySegment> currentSegments) const
{
	ZoneScopedN("RenderGraph::BuildShadowReplayScheduleFromCachedSegments");
	ReplaySegmentVerificationReport report{};
	const ReplaySegmentValidationStats validation = ValidateCachedSegmentsAgainstCurrentFrame(previousSegments, currentSegments);
	std::unordered_map<uint64_t, const CachedReplaySegment*> previousByPassSequence;
	previousByPassSequence.reserve(previousSegments.size());
	for (const auto& segment : previousSegments) {
		previousByPassSequence.emplace(segment.identity.passSequenceHash, &segment);
	}

	std::vector<uint8_t> passCovered(m_framePasses.size(), 0);
	const bool relaxAliasPlacement = m_renderGraphSettingsService
		? m_renderGraphSettingsService->GetRenderGraphReplayRelaxAliasPlacement()
		: true;
	for (const auto& current : currentSegments) {
		auto previousIt = previousByPassSequence.find(current.identity.passSequenceHash);
		if (previousIt == previousByPassSequence.end()) {
			continue;
		}
		const auto& previous = *previousIt->second;
		if (previous.fingerprint.declarationHash != current.fingerprint.declarationHash
			|| previous.fingerprint.accessHash != current.fingerprint.accessHash
			|| previous.fingerprint.queueHash != current.fingerprint.queueHash
			|| (!relaxAliasPlacement && previous.fingerprint.aliasHash != current.fingerprint.aliasHash)
			|| previous.fingerprint.boundaryHash != current.fingerprint.boundaryHash
			|| previous.fingerprint.templateShapeHash != current.fingerprint.templateShapeHash) {
			continue;
		}
		++report.matchedSegments;
		report.replayedPasses += current.schedule.passCount;
		report.insertedInputTransitions += current.contract.inputRequirements.size();
		report.extraInputTransitionsAllowed += previous.templateStats.transitionStateHash != current.templateStats.transitionStateHash ? 1ull : 0ull;
		for (uint32_t traceIndex = current.schedule.firstTraceIndex; traceIndex <= current.schedule.lastTraceIndex && traceIndex < m_schedulingDecisionTrace.size(); ++traceIndex) {
			const auto passIndex = m_schedulingDecisionTrace[traceIndex].passIndex;
			if (passIndex < passCovered.size()) {
				passCovered[passIndex] = 1;
			}
		}
	}

	for (uint8_t covered : passCovered) {
		if (covered == 0) {
			++report.dynamicGapPasses;
		}
	}

	report.checkedPasses = report.replayedPasses + report.dynamicGapPasses;
	report.checkedEdges = validation.currentSegmentCount;
	report.checkedRequirements = 0;
	for (const auto& segment : currentSegments) {
		report.checkedRequirements += segment.contract.inputRequirements.size() + segment.contract.outputStates.size();
	}
	report.checkedQueueSyncs = 0;
	for (const auto& segment : currentSegments) {
		report.checkedQueueSyncs += segment.contract.boundarySyncs.size();
	}
	report.valid = validation.previousSegmentCount == 0 || report.matchedSegments > 0 || currentSegments.empty();
	if (!report.valid) {
		report.failures = validation.misses;
		report.firstFailure = "no cached replay segment matched the current frame";
	}
	return report;
}

RenderGraph::ReplaySegmentVerificationReport RenderGraph::VerifyReplayScheduleSemanticCorrectness(
	std::span<const CachedReplaySegment> replaySegments) const
{
	ZoneScopedN("RenderGraph::VerifyReplayScheduleSemanticCorrectness");
	ReplaySegmentVerificationReport report{};
	for (const auto& segment : replaySegments) {
		report.checkedPasses += segment.schedule.passCount;
		report.checkedEdges += segment.contract.boundaryEdges.size();
		report.checkedRequirements += segment.contract.inputRequirements.size() + segment.contract.outputStates.size();
		report.checkedQueueSyncs += segment.contract.boundarySyncs.size();
		if (segment.contract.inputRequirements.empty() && segment.schedule.requirementCount != 0) {
			report.valid = false;
			++report.failures;
			if (report.firstFailure.empty()) {
				report.firstFailure = "segment has requirements but no input contract";
			}
		}
		if (segment.batchTemplates.empty()) {
			report.valid = false;
			++report.failures;
			if (report.firstFailure.empty()) {
				report.firstFailure = "segment has no batch templates";
			}
		}
	}
	return report;
}

RenderGraph::ReplaySegmentVerificationReport RenderGraph::ReplayCurrentFrameSegmentsAsAuthoritative(
	std::span<const CachedReplaySegment> replaySegments,
	std::span<const SchedulingDecisionTrace> authoritativeTrace,
	RenderGraph::FramePassList& framePasses,
	std::vector<Node>& nodes)
{
	ZoneScopedN("RenderGraph::ReplayCurrentFrameSegmentsAsAuthoritative");
	ReplaySegmentVerificationReport report{};
	const size_t queueCount = m_queueRegistry.SlotCount();
	(void)authoritativeTrace;

	auto openNewBatch = [&]() -> PassBatch {
		PassBatch batch(queueCount);
		batch.passBatchTrackersByResourceIndex.assign(m_frameSchedulingResourceCount, nullptr);
		for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
			batch.SetQueueSignalFenceValue(BatchSignalPhase::AfterTransitions, queueIndex, GetNextQueueFenceValue(queueIndex));
			batch.SetQueueSignalFenceValue(BatchSignalPhase::AfterExecution, queueIndex, GetNextQueueFenceValue(queueIndex));
			batch.SetQueueSignalFenceValue(BatchSignalPhase::AfterCompletion, queueIndex, GetNextQueueFenceValue(queueIndex));
		}
		return batch;
	};

	std::unordered_map<uint32_t, uint32_t> nodeIndexByPassIndex;
	nodeIndexByPassIndex.reserve(nodes.size());
	constexpr uint32_t kInvalidReplayNodeIndex = std::numeric_limits<uint32_t>::max();
	std::vector<uint32_t> nodeIndexByPassIndexDense(framePasses.size(), kInvalidReplayNodeIndex);
	for (uint32_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
		const auto passIndex = static_cast<uint32_t>(nodes[nodeIndex].passIndex);
		nodeIndexByPassIndex.emplace(passIndex, nodeIndex);
		if (passIndex < nodeIndexByPassIndexDense.size()) {
			nodeIndexByPassIndexDense[passIndex] = nodeIndex;
		}
	}
	auto tryGetReplayNodeIndex = [&](uint32_t passIndex) -> std::optional<uint32_t> {
		if (passIndex < nodeIndexByPassIndexDense.size()) {
			const uint32_t nodeIndex = nodeIndexByPassIndexDense[passIndex];
			if (nodeIndex != kInvalidReplayNodeIndex) {
				return nodeIndex;
			}
		}
		auto nodeIt = nodeIndexByPassIndex.find(passIndex);
		if (nodeIt == nodeIndexByPassIndex.end()) {
			return std::nullopt;
		}
		return nodeIt->second;
	};

	FrameEpochSet scratchTransitioned;
	scratchTransitioned.Initialize(m_frameSchedulingResourceCount);
	FrameEpochSet scratchFallback;
	scratchFallback.Initialize(m_frameSchedulingResourceCount);
	std::vector<ResourceTransition> scratchTransitions;
	std::vector<ResourceTransition> ignoredTransitions;
	std::vector<uint64_t> latestSignalFenceByQueue(queueCount, 0);
	std::vector<std::pair<uint16_t, uint16_t>> pendingSegmentInputWaits;

	struct TemplateResourceLookupKey {
		uint64_t resourceID = 0;
		uint64_t backingResourceID = 0;
		bool dynamicResource = false;

		bool operator==(const TemplateResourceLookupKey& other) const noexcept {
			return resourceID == other.resourceID
				&& backingResourceID == other.backingResourceID
				&& dynamicResource == other.dynamicResource;
		}
	};
	struct TemplateResourceLookupKeyHash {
		size_t operator()(const TemplateResourceLookupKey& key) const noexcept {
			size_t h = std::hash<uint64_t>{}(key.resourceID);
			h ^= std::hash<uint64_t>{}(key.backingResourceID) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
			h ^= std::hash<bool>{}(key.dynamicResource) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
			return h;
		}
	};
	std::unordered_map<TemplateResourceLookupKey, Resource*, TemplateResourceLookupKeyHash> templateResourceCache;
	std::unordered_map<TemplateResourceLookupKey, std::optional<size_t>, TemplateResourceLookupKeyHash> templateResourceIndexCache;
	templateResourceCache.reserve(256);
	templateResourceIndexCache.reserve(256);

	auto setFirstRecomputeReason = [&](std::string reason) {
		if (report.firstRecomputeReason.empty()) {
			report.firstRecomputeReason = std::move(reason);
		}
	};

	batches.clear();
	batches.emplace_back(queueCount);
	m_schedulingDecisionTrace.clear();
	m_transitionPlacementCandidates.clear();
	m_transitionPlacementStats = {};
	ResetFrameQueueBatchHistoryTables();
	RebuildFrameCompileResources();

	auto resolveTemplateResource = [&](uint64_t resourceID, uint64_t backingResourceID, bool dynamicResource) -> Resource* {
		const TemplateResourceLookupKey key{ resourceID, backingResourceID, dynamicResource };
		if (auto cacheIt = templateResourceCache.find(key); cacheIt != templateResourceCache.end()) {
			return cacheIt->second;
		}
		Resource* resolved = nullptr;
		if (dynamicResource && resourceID != 0) {
			if (auto resource = GetResourceByID(resourceID)) {
				resolved = resource.get();
			}
		}
		if (resolved == nullptr && backingResourceID != 0) {
			if (auto resource = GetResourceByID(backingResourceID)) {
				resolved = resource.get();
			}
		}
		if (resolved == nullptr && resourceID != 0) {
			if (auto resource = GetResourceByID(resourceID)) {
				resolved = resource.get();
			}
		}
		templateResourceCache.emplace(key, resolved);
		return resolved;
	};

	auto resolveTemplateResourceIndex = [&](uint64_t resourceID, uint64_t backingResourceID, bool dynamicResource) -> std::optional<size_t> {
		const TemplateResourceLookupKey key{ resourceID, backingResourceID, dynamicResource };
		if (auto cacheIt = templateResourceIndexCache.find(key); cacheIt != templateResourceIndexCache.end()) {
			return cacheIt->second;
		}
		std::optional<size_t> resolvedIndex;
		if (dynamicResource && resourceID != 0) {
			if (auto resourceIndex = TryGetFrameSchedulingResourceIndex(resourceID)) {
				resolvedIndex = resourceIndex;
			}
			if (!resolvedIndex) {
				auto* resource = resolveTemplateResource(resourceID, backingResourceID, true);
				if (resource != nullptr) {
					if (auto resourceIndex = TryGetFrameSchedulingResourceIndex(resource->GetGlobalResourceID())) {
						resolvedIndex = resourceIndex;
					}
				}
			}
		}
		if (!resolvedIndex) {
			if (auto resourceIndex = TryGetFrameSchedulingResourceIndex(resourceID)) {
				resolvedIndex = resourceIndex;
			}
		}
		if (!resolvedIndex && backingResourceID != 0) {
			resolvedIndex = TryGetFrameSchedulingResourceIndex(backingResourceID);
		}
		templateResourceIndexCache.emplace(key, resolvedIndex);
		return resolvedIndex;
	};

	auto isExplicitWholeResourceRange = [](const RangeSpec& range) {
		return range.mipLower.type == BoundType::All
			&& range.mipUpper.type == BoundType::All
			&& range.sliceLower.type == BoundType::All
			&& range.sliceUpper.type == BoundType::All;
	};

	auto applyTrackerState = [&](size_t resourceIndex, uint64_t resourceID, Resource* resource, const RangeSpec& range, ResourceState state) {
		auto& compileResourceState = GetOrCreateFrameCompileResourceState(resourceIndex, resource, resourceID);
		resource = compileResourceState.resource ? compileResourceState.resource : resource;
		ignoredTransitions.clear();
		compileResourceState.tracker->Apply(range, resource, state, ignoredTransitions);
		if (resource && isExplicitWholeResourceRange(range)) {
			compileResourceState.fastState.valid = true;
			compileResourceState.fastState.wholeResourceOnly = true;
			compileResourceState.fastState.state = state;
		}
		else {
			compileResourceState.fastState.valid = false;
			compileResourceState.fastState.wholeResourceOnly = false;
		}
		return &*compileResourceState.tracker;
	};

	auto appendPassPointer = [&](PassBatch& batch, uint16_t queueSlot, AnyPassAndResources& passAndResources) -> bool {
		bool appended = false;
		std::visit(
			[&](auto& pass) {
				using PassTypeT = std::decay_t<decltype(pass)>;
				if constexpr (!std::is_same_v<PassTypeT, std::monostate>) {
					batch.Passes(queueSlot).emplace_back(&pass);
					for (const auto& wait : pass.resources.externalWaitsBeforeTransitions) {
						batch.AddExternalWaitBeforeTransitions(queueSlot, wait);
					}
					appended = true;
				}
			},
			passAndResources.pass);
		return appended;
	};

	auto recomputeTemplateBatch = [&](const ReplaySegmentBatchTemplate& batchTemplate, std::string_view reason) -> bool {
		ZoneScopedN("RenderGraph::Replay::FallbackRecomputeBatch");
		++report.recomputedBatches;
		setFirstRecomputeReason(std::string(reason));
		report.valid = false;
		++report.failures;
		if (report.firstFailure.empty()) {
			report.firstFailure = std::string(reason);
		}
		for (const auto& queuedPass : batchTemplate.queuedPasses) {
			const uint32_t passIndex = queuedPass.originalFramePassIndexAtExtraction;
			if (passIndex < m_framePassSchedulingSummaries.size()) {
				report.recomputedTransitionCalls += m_framePassSchedulingSummaries[passIndex].requirements.size();
			}
		}
		return false;
	};

	auto applySegmentOutputStates = [&](const CachedReplaySegment& segment) -> bool {
		ZoneScopedN("RenderGraph::Replay::ApplyOutputStates");
		for (const auto& output : segment.contract.outputStates) {
			auto resourceIndex = TryGetFrameSchedulingResourceIndex(output.resourceID);
			if (!resourceIndex) {
				continue;
			}
			Resource* resource = resolveTemplateResource(output.resourceID, 0, false);
			auto* tracker = applyTrackerState(*resourceIndex, output.resourceID, resource, output.range, output.finalState);
			(void)tracker;
			auto& compileResourceState = m_frameCompileResources[*resourceIndex];
			compileResourceState.fastState.valid = output.validFastState;
			compileResourceState.fastState.wholeResourceOnly = output.wholeResource && output.validFastState;
			if (output.validFastState) {
				compileResourceState.fastState.state = output.finalState;
			}
		}
		return true;
	};

	auto ensureSegmentInputs = [&](const CachedReplaySegment& segment) -> bool {
		ZoneScopedN("RenderGraph::Replay::EnsureSegmentInputs");
		pendingSegmentInputWaits.clear();
		scratchTransitioned.Clear();
		scratchFallback.Clear();
		if (segment.contract.inputRequirements.empty()) {
			return true;
		}

		PassBatch inputBatch = openNewBatch();
		const unsigned int inputBatchIndex = static_cast<unsigned int>(batches.size());
		std::vector<uint8_t> consumerQueues(queueCount, 0);
		std::unordered_set<uint64_t> internallyTransitionedResourceIDs;
		for (const auto& batchTemplate : segment.batchTemplates) {
			for (const auto& transition : batchTemplate.transitions) {
				internallyTransitionedResourceIDs.insert(transition.resourceID);
			}
		}

		auto hasInputBatchTransitions = [&]() {
			for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
				for (size_t phaseIndex = 0; phaseIndex < static_cast<size_t>(BatchTransitionPhase::Count); ++phaseIndex) {
					if (inputBatch.HasTransitions(queueIndex, static_cast<BatchTransitionPhase>(phaseIndex))) {
						return true;
					}
				}
			}
			return false;
		};

		for (const auto& input : segment.contract.inputRequirements) {
			// Transition segment-entry states only. Per-pass requirements for
			// resources with later internal transitions can describe states that
			// are not valid at the segment boundary.
			if (!input.transitionBeforeState
				&& internallyTransitionedResourceIDs.find(input.resourceID) != internallyTransitionedResourceIDs.end()) {
				continue;
			}
			if (input.queueSlot >= queueCount) {
				report.valid = false;
				++report.failures;
				if (report.firstFailure.empty()) {
					report.firstFailure = "segment input queue out of range";
				}
				return false;
			}
			consumerQueues[input.queueSlot] = 1;

			const auto resourceIndex = TryGetFrameSchedulingResourceIndex(input.resourceID);
			if (!resourceIndex) {
				report.valid = false;
				++report.failures;
				if (report.firstFailure.empty()) {
					std::ostringstream oss;
					oss << "segment input resource missing id=" << input.resourceID;
					report.firstFailure = oss.str();
				}
				return false;
			}

			Resource* resource = input.resource.IsEphemeral()
				? input.resource.GetEphemeralPtr()
				: _registry.Resolve(input.resource);
			if (resource == nullptr) {
				resource = resolveTemplateResource(input.resourceID, 0, false);
			}

			const QueueKind queueKind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(input.queueSlot));
			const ResourceState requiredState = NormalizeStateForQueue(queueKind, input.requiredState);
			const bool requiredStateIsNoAccess =
				requiredState.access == rhi::ResourceAccessType::None
				&& requiredState.layout == rhi::ResourceLayout::Undefined
				&& requiredState.sync == rhi::ResourceSyncState::None;
			auto& compileResourceState = GetOrCreateFrameCompileResourceState(*resourceIndex, resource, input.resourceID);
			if (requiredStateIsNoAccess) {
				// NONE/UNDEFINED/NONE is a template before-state marker, not a
				// usable segment boundary state. Emitting a ReplaySegmentInput
				// barrier to it would produce SyncAfter=None, after which D3D12
				// forbids later access in the same ExecuteCommandLists scope.
				inputBatch.passBatchTrackersByResourceIndex[*resourceIndex] = &*compileResourceState.tracker;
				continue;
			}
			if (!compileResourceState.tracker->WouldModify(input.range, requiredState)) {
				inputBatch.passBatchTrackersByResourceIndex[*resourceIndex] = &*compileResourceState.tracker;
				continue;
			}
			const bool aliasActivationPendingForInput =
				*resourceIndex < m_aliasActivationPendingByResourceIndex.size()
				&& m_aliasActivationPendingByResourceIndex[*resourceIndex] != 0;
			if (aliasActivationPendingForInput) {
				const bool inputCanActivateAlias = AccessTypeIsWriteType(requiredState.access)
					|| requiredState.access == rhi::ResourceAccessType::Common;
				if (input.transitionBeforeState && input.transitionDiscard) {
					// A cached discard transition is the segment's alias activation point.
					// Do not synthesize a ReplaySegmentInput transition to its before-state;
					// that turns an internal first-write into an invalid boundary read/use.
					inputBatch.passBatchTrackersByResourceIndex[*resourceIndex] = &*compileResourceState.tracker;
					continue;
				}
				if (!inputCanActivateAlias) {
					report.valid = false;
					++report.failures;
					if (report.firstFailure.empty()) {
						std::ostringstream oss;
						oss << "segment_input_alias_read_without_activation resource_id=" << input.resourceID
							<< " transition_before=" << (input.transitionBeforeState ? 1 : 0)
							<< " discard=" << (input.transitionDiscard ? 1 : 0);
						report.firstFailure = oss.str();
					}
					return false;
				}
			}

			DenseRequirementSummary requirement{
				.resource = input.resource,
				.resourceID = input.resourceID,
				.resourceIndex = *resourceIndex,
				.range = input.range,
				.state = input.requiredState,
				.isUAV = IsUAVState(input.requiredState),
				.isWholeResource = IsWholeResourceRange(input.range, input.resource),
				.equivalentResourceIndices = nullptr,
			};
			AddTransition(
				inputBatchIndex,
				inputBatch,
				input.queueSlot,
				"ReplaySegmentInput",
				requirement,
				scratchTransitioned,
				scratchFallback,
				scratchTransitions);
		}

		if (!hasInputBatchTransitions()) {
			return true;
		}

		std::vector<uint8_t> transitionQueues(queueCount, 0);
		for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
			for (size_t phaseIndex = 0; phaseIndex < static_cast<size_t>(BatchTransitionPhase::Count); ++phaseIndex) {
				if (inputBatch.HasTransitions(queueIndex, static_cast<BatchTransitionPhase>(phaseIndex))) {
					transitionQueues[queueIndex] = 1;
				}
			}
		}

		for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
			if (!transitionQueues[queueIndex]) {
				continue;
			}
			inputBatch.MarkQueueSignal(BatchSignalPhase::AfterTransitions, queueIndex);
			latestSignalFenceByQueue[queueIndex] = inputBatch.GetQueueSignalFenceValue(BatchSignalPhase::AfterTransitions, queueIndex);
			for (size_t dstQueue = 0; dstQueue < queueCount; ++dstQueue) {
				if (consumerQueues[dstQueue] && dstQueue != queueIndex) {
					pendingSegmentInputWaits.emplace_back(static_cast<uint16_t>(dstQueue), static_cast<uint16_t>(queueIndex));
				}
			}
		}

		for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
			for (size_t phaseIndex = 0; phaseIndex < static_cast<size_t>(BatchTransitionPhase::Count); ++phaseIndex) {
				report.insertedInputTransitions += inputBatch.Transitions(queueIndex, static_cast<BatchTransitionPhase>(phaseIndex)).size();
			}
		}

		batches.push_back(std::move(inputBatch));
		return true;
	};

	auto commitSegment = [&](const CachedReplaySegment& segment) -> bool {
		ZoneScopedN("RenderGraph::Replay::CommitCachedSegment");
		++report.matchedSegments;
		if (!ensureSegmentInputs(segment)) {
			return false;
		}

		const unsigned int replayBatchBaseIndex = static_cast<unsigned int>(batches.size());
		std::vector<unsigned int> committedBatchIndexByLocal(segment.batchTemplates.size(), 0);

		for (const auto& batchTemplate : segment.batchTemplates) {
			ZoneScopedN("RenderGraph::Replay::MaterializeCachedBatch");
			ZoneValue(batchTemplate.queuedPasses.size());
			if (batchTemplate.queuedPasses.empty()) {
				continue;
			}

			for (const auto& wait : batchTemplate.waits) {
				if (wait.dstQueue >= queueCount || wait.srcQueue >= queueCount) {
					return recomputeTemplateBatch(batchTemplate, "template_wait_queue_out_of_range");
				}
				if (latestSignalFenceByQueue[wait.srcQueue] == 0) {
					return recomputeTemplateBatch(batchTemplate, "template_wait_missing_source_signal");
				}
			}
			for (const auto& signal : batchTemplate.signals) {
				if (signal.queueSlot >= queueCount) {
					return recomputeTemplateBatch(batchTemplate, "template_signal_queue_out_of_range");
				}
			}

			PassBatch batch = openNewBatch();
			const unsigned int batchIndex = static_cast<unsigned int>(batches.size());
			if (batchTemplate.localBatchIndex < committedBatchIndexByLocal.size()) {
				committedBatchIndexByLocal[batchTemplate.localBatchIndex] = batchIndex;
			}

			batch.allResources = batchTemplate.allResources;
			batch.internallyTransitionedResources = batchTemplate.internallyTransitionedResources;

			for (const auto& [dstQueue, srcQueue] : pendingSegmentInputWaits) {
				if (dstQueue < queueCount && srcQueue < queueCount && latestSignalFenceByQueue[srcQueue] != 0) {
					batch.AddQueueWait(BatchWaitPhase::BeforeTransitions, dstQueue, srcQueue, latestSignalFenceByQueue[srcQueue]);
				}
			}
			pendingSegmentInputWaits.clear();

			for (const auto& wait : batchTemplate.waits) {
				const uint64_t sourceFence = latestSignalFenceByQueue[wait.srcQueue];
				batch.AddQueueWait(wait.phase, wait.dstQueue, wait.srcQueue, sourceFence);
			}

			for (const auto& transitionTemplate : batchTemplate.transitions) {
				if (transitionTemplate.queueSlot >= queueCount) {
					return recomputeTemplateBatch(batchTemplate, "template_transition_queue_out_of_range");
				}
				Resource* resource = resolveTemplateResource(transitionTemplate.resourceID, transitionTemplate.backingResourceID, transitionTemplate.dynamicResource);
				if (resource == nullptr) {
					return recomputeTemplateBatch(batchTemplate, "template_transition_resource_missing");
				}
				auto resourceIndex = resolveTemplateResourceIndex(transitionTemplate.resourceID, transitionTemplate.backingResourceID, transitionTemplate.dynamicResource);
				if (!resourceIndex) {
					return recomputeTemplateBatch(batchTemplate, "template_transition_resource_index_missing");
				}
				auto& compileResourceState = GetOrCreateFrameCompileResourceState(
					*resourceIndex,
					resource,
					resource ? resource->GetGlobalResourceID() : transitionTemplate.resourceID);
				if (transitionTemplate.discard) {
					ResourceState emittedBeforeState = transitionTemplate.before;
					ResourceState liveBeforeState{};
					if (TryGetWholeResourceTrackerState(*compileResourceState.tracker, liveBeforeState)) {
						emittedBeforeState = liveBeforeState;
					}
					ResourceTransition transition(
						resource,
						transitionTemplate.range,
						emittedBeforeState.access,
						transitionTemplate.after.access,
						emittedBeforeState.layout,
						transitionTemplate.after.layout,
						emittedBeforeState.sync,
						transitionTemplate.after.sync,
						true);
					batch.Transitions(transitionTemplate.queueSlot, transitionTemplate.phase).push_back(transition);
					ignoredTransitions.clear();
					compileResourceState.tracker->Apply(transitionTemplate.range, resource, transitionTemplate.after, ignoredTransitions);
				}
				else {
					scratchTransitions.clear();
					compileResourceState.tracker->Apply(transitionTemplate.range, resource, transitionTemplate.after, scratchTransitions);
					auto& transitions = batch.Transitions(transitionTemplate.queueSlot, transitionTemplate.phase);
					transitions.insert(transitions.end(), scratchTransitions.begin(), scratchTransitions.end());
				}
				RecordFrameQueueTransitionBatch(transitionTemplate.queueSlot, *resourceIndex, batchIndex);
				const bool transitionCanActivateAlias = AccessTypeIsWriteType(transitionTemplate.after.access)
					|| transitionTemplate.after.access == rhi::ResourceAccessType::Common;
				if (transitionTemplate.discard
					&& transitionCanActivateAlias
					&& *resourceIndex < m_aliasActivationPendingByResourceIndex.size()) {
					m_aliasActivationPendingByResourceIndex[*resourceIndex] = 0;
					aliasActivationPending.erase(resource->GetGlobalResourceID());
				}
			}
			report.replayedInternalTransitions += batchTemplate.transitions.size();

			bool batchHasPasses = false;
			for (const auto& queuedPass : batchTemplate.queuedPasses) {
				const uint32_t passIndex = queuedPass.originalFramePassIndexAtExtraction;
				auto nodeIndex = tryGetReplayNodeIndex(passIndex);
				if (!nodeIndex) {
					return recomputeTemplateBatch(batchTemplate, "template_pass_missing_current_node");
				}
				if (passIndex >= framePasses.size() || queuedPass.queueSlot >= queueCount) {
					return recomputeTemplateBatch(batchTemplate, "template_pass_out_of_range");
				}

				auto& node = nodes[*nodeIndex];
				node.assignedQueueSlot = queuedPass.queueSlot;
				if (node.passIndex < m_assignedQueueSlotsByFramePass.size()) {
					m_assignedQueueSlotsByFramePass[node.passIndex] = queuedPass.queueSlot;
				}
				m_schedulingDecisionTrace.push_back(SchedulingDecisionTrace{
					.nodeIndex = *nodeIndex,
					.passIndex = passIndex,
					.batchIndex = batchIndex,
					.assignedQueueSlot = queuedPass.queueSlot,
					.closedBatchBefore = !batchHasPasses,
					.readySetSize = 0,
					.candidateChecks = 0,
					.isNewBatchNeededChecks = 0,
					.fallbackCommit = false,
				});
				if (!appendPassPointer(batch, queuedPass.queueSlot, framePasses[passIndex])) {
					return recomputeTemplateBatch(batchTemplate, "template_pass_empty_variant");
				}
				if (passIndex < m_framePassSchedulingSummaries.size()) {
					report.addTransitionCallsSaved += m_framePassSchedulingSummaries[passIndex].requirements.size();
				}
				batchHasPasses = true;
				++report.checkedPasses;
				++report.replayedPasses;
			}

			for (const auto& signal : batchTemplate.signals) {
				batch.MarkQueueSignal(signal.phase, signal.queueSlot);
				latestSignalFenceByQueue[signal.queueSlot] = batch.GetQueueSignalFenceValue(signal.phase, signal.queueSlot);
			}

			if (batchHasPasses) {
				batches.push_back(std::move(batch));
				++report.templateReplayedBatches;
			}
		}

		auto committedBatchIndexForLocal = [&](uint32_t localBatchIndex) -> unsigned int {
			if (localBatchIndex < committedBatchIndexByLocal.size() && committedBatchIndexByLocal[localBatchIndex] != 0) {
				return committedBatchIndexByLocal[localBatchIndex];
			}
			return replayBatchBaseIndex + localBatchIndex;
		};

		{
			ZoneScopedN("RenderGraph::Replay::ApplySegmentQueueUsage");
			for (const auto& usage : segment.contract.queueUsage) {
				if (usage.queueSlot >= queueCount) {
					continue;
				}
				auto resourceIndex = resolveTemplateResourceIndex(usage.resourceID, 0, false);
				if (!resourceIndex) {
					continue;
				}
				const unsigned int batchIndex = committedBatchIndexForLocal(usage.lastLocalBatch);
				if (usage.read || usage.write) {
					RecordFrameQueueUsageBatch(usage.queueSlot, *resourceIndex, batchIndex);
				}
				if (usage.transition) {
					RecordFrameQueueTransitionBatch(usage.queueSlot, *resourceIndex, batchIndex);
				}
				if (usage.producer) {
					SetFrameQueueHistoryValue(m_frameQueueLastProducerBatch, usage.queueSlot, *resourceIndex, batchIndex);
					if (usage.queueSlot < m_compiledLastProducerBatchByResourceByQueue.size()) {
						m_compiledLastProducerBatchByResourceByQueue[usage.queueSlot][usage.resourceID] = batchIndex;
					}
					if (*resourceIndex < m_aliasActivationPendingByResourceIndex.size()) {
						m_aliasActivationPendingByResourceIndex[*resourceIndex] = 0;
					}
					aliasActivationPending.erase(usage.resourceID);
				}
			}
		}

		if (!applySegmentOutputStates(segment)) {
			return false;
		}
		return true;
	};

	std::unordered_map<uint32_t, const CachedReplaySegment*> segmentByFirstNode;
	segmentByFirstNode.reserve(replaySegments.size());
	std::unordered_map<const CachedReplaySegment*, size_t> replaySegmentIndexByPointer;
	replaySegmentIndexByPointer.reserve(replaySegments.size());
	std::vector<std::vector<uint32_t>> segmentNodesByFirstNode(nodes.size());
	std::unordered_map<uint32_t, uint32_t> segmentFirstNodeByNode;
	segmentFirstNodeByNode.reserve(nodes.size());
	for (size_t replaySegmentIndex = 0; replaySegmentIndex < replaySegments.size(); ++replaySegmentIndex) {
		const auto& segment = replaySegments[replaySegmentIndex];
		replaySegmentIndexByPointer.emplace(&segment, replaySegmentIndex);
		if (!segment.tier1Eligible || segment.batchTemplates.empty()) {
			continue;
		}
		std::vector<uint32_t> segmentNodes;
		segmentNodes.reserve(segment.schedule.passCount);
		uint32_t firstNodeIndex = std::numeric_limits<uint32_t>::max();
		bool validSegmentNodeSet = true;
		for (const auto& batchTemplate : segment.batchTemplates) {
			for (const auto& queuedPass : batchTemplate.queuedPasses) {
				auto nodeIndex = tryGetReplayNodeIndex(queuedPass.originalFramePassIndexAtExtraction);
				if (!nodeIndex) {
					validSegmentNodeSet = false;
					break;
				}
				segmentNodes.push_back(*nodeIndex);
				if (firstNodeIndex == std::numeric_limits<uint32_t>::max()
					|| nodes[*nodeIndex].originalOrder < nodes[firstNodeIndex].originalOrder) {
					firstNodeIndex = *nodeIndex;
				}
			}
			if (!validSegmentNodeSet) {
				break;
			}
		}
		if (!validSegmentNodeSet || firstNodeIndex >= nodes.size() || segmentNodes.empty()) {
			continue;
		}
		std::sort(segmentNodes.begin(), segmentNodes.end());
		segmentNodes.erase(std::unique(segmentNodes.begin(), segmentNodes.end()), segmentNodes.end());
		if (segmentNodes.size() != segment.schedule.passCount) {
			continue;
		}
		segmentByFirstNode.emplace(firstNodeIndex, &segment);
		for (uint32_t nodeIndex : segmentNodes) {
			segmentFirstNodeByNode.emplace(nodeIndex, firstNodeIndex);
		}
		segmentNodesByFirstNode[firstNodeIndex] = std::move(segmentNodes);
	}
	std::vector<uint32_t> segmentCandidateFirstNodes;
	segmentCandidateFirstNodes.reserve(segmentByFirstNode.size());
	for (const auto& [firstNodeIndex, segment] : segmentByFirstNode) {
		(void)segment;
		segmentCandidateFirstNodes.push_back(firstNodeIndex);
	}
	std::sort(segmentCandidateFirstNodes.begin(), segmentCandidateFirstNodes.end(), [&](uint32_t lhs, uint32_t rhs) {
		return nodes[lhs].originalOrder < nodes[rhs].originalOrder;
	});
	std::unordered_set<uint32_t> replayRejectedSegmentFirstNodes;

	auto updateLatestSignalsFromBatch = [&](const PassBatch& batch) {
		for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
			for (size_t signalPhaseIndex = 0; signalPhaseIndex < PassBatch::kSignalPhaseCount; ++signalPhaseIndex) {
				const auto signalPhase = static_cast<BatchSignalPhase>(signalPhaseIndex);
				if (batch.HasQueueSignal(signalPhase, queueIndex)) {
					latestSignalFenceByQueue[queueIndex] = batch.GetQueueSignalFenceValue(signalPhase, queueIndex);
				}
			}
		}
	};

	auto applyPendingSegmentInputWaits = [&](PassBatch& batch) {
		for (const auto& [dstQueue, srcQueue] : pendingSegmentInputWaits) {
			if (dstQueue < queueCount && srcQueue < queueCount && latestSignalFenceByQueue[srcQueue] != 0) {
				batch.AddQueueWait(BatchWaitPhase::BeforeTransitions, dstQueue, srcQueue, latestSignalFenceByQueue[srcQueue]);
			}
		}
		pendingSegmentInputWaits.clear();
	};

	std::vector<uint32_t> indeg(nodes.size());
	for (size_t i = 0; i < nodes.size(); ++i) {
		indeg[i] = nodes[i].indegree;
	}

	std::vector<size_t> ready;
	ready.reserve(nodes.size());
	for (size_t i = 0; i < nodes.size(); ++i) {
		if (indeg[i] == 0) {
			ready.push_back(i);
		}
	}

	PassBatch currentBatch = openNewBatch();
	unsigned int currentBatchIndex = 1;
	const size_t gfxSlot = QueueIndex(QueueKind::Graphics);
	BatchBuildState batchBuildState;
	batchBuildState.Initialize(nodes.size(), queueCount, m_frameSchedulingResourceCount);

	auto currentBatchHasPasses = [&]() {
		for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
			if (!currentBatch.Passes(queueIndex).empty()) {
				return true;
			}
		}
		return false;
	};

	auto closeCurrentBatch = [&]() {
		ZoneScopedN("RenderGraph::Replay::CloseDynamicBatch");
		if (!currentBatchHasPasses()) {
			return;
		}
		updateLatestSignalsFromBatch(currentBatch);
		batches.push_back(std::move(currentBatch));
		currentBatch = openNewBatch();
		batchBuildState.ResetForNewBatch();
		currentBatchIndex = static_cast<unsigned int>(batches.size());
	};

	auto passHasImmediateWork = [](const AnyPassAndResources& any) {
		return std::visit(
			[](const auto& passEntry) -> bool {
				using T = std::decay_t<decltype(passEntry)>;
				if constexpr (std::is_same_v<T, std::monostate>) {
					return true;
				}
				else {
					return passEntry.run != PassRunMask::Retained || !passEntry.immediateBytecode.empty();
				}
			},
			any.pass);
	};

	auto passForcesBatchIsolation = [&](size_t passIndex) {
		return passIndex >= framePasses.size() || passHasImmediateWork(framePasses[passIndex]);
	};

	auto updateBatchMembershipForCommittedPass = [&](const Node& committedNode) {
		const auto& passSummary = m_framePassSchedulingSummaries[committedNode.passIndex];
		for (const auto& transition : passSummary.internalTransitions) {
			batchBuildState.MarkInternalTransition(transition.resourceIndex);
		}
		const size_t passQueueSlot = committedNode.assignedQueueSlot.value_or(committedNode.queueSlot);
		for (size_t resourceIndex : passSummary.uavResourceIndices) {
			for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
				if (queueIndex != passQueueSlot) {
					batchBuildState.MarkOtherQueueUAV(queueIndex, resourceIndex);
				}
			}
		}
	};

	std::vector<uint8_t> scheduled(nodes.size(), 0);
	size_t remaining = nodes.size();
	bool closedBatchBeforeNextCommit = false;
	const double autoGraphicsBias = m_getQueueSchedulingAutoGraphicsBias ? static_cast<double>(m_getQueueSchedulingAutoGraphicsBias()) : 2.5;
	const double asyncOverlapBonus = m_getQueueSchedulingAsyncOverlapBonus ? static_cast<double>(m_getQueueSchedulingAsyncOverlapBonus()) : 3.0;
	const double crossQueueHandoffPenalty = m_getQueueSchedulingCrossQueueHandoffPenalty ? static_cast<double>(m_getQueueSchedulingCrossQueueHandoffPenalty()) : 2.0;

	auto isAliasActivationReadBlocked = [&](size_t passIndex, std::string* outReason) {
		if (passIndex >= m_framePassSchedulingSummaries.size()) {
			return false;
		}
		const auto& passSummary = m_framePassSchedulingSummaries[passIndex];
		for (const auto& requirement : passSummary.requirements) {
			if (requirement.resourceIndex >= m_aliasActivationPendingByResourceIndex.size()
				|| m_aliasActivationPendingByResourceIndex[requirement.resourceIndex] == 0) {
				continue;
			}
			const bool firstUseCanActivate = AccessTypeIsWriteType(requirement.state.access)
				|| requirement.state.access == rhi::ResourceAccessType::Common;
			if (firstUseCanActivate) {
				continue;
			}
			const auto* placement = TryGetAliasPlacementRange(requirement.resourceID);
			if (placement == nullptr
				|| placement->firstUsePassIndex == std::numeric_limits<size_t>::max()
				|| placement->firstUsePassIndex == passIndex) {
				if (outReason && outReason->empty()) {
					std::ostringstream oss;
					oss << "alias_activation_read_without_first_write pass=\"" << framePasses[passIndex].name
						<< "\" resource_id=" << requirement.resourceID;
					*outReason = oss.str();
				}
				return true;
			}
			auto firstUseNodeIndex = tryGetReplayNodeIndex(static_cast<uint32_t>(placement->firstUsePassIndex));
			if (!firstUseNodeIndex
				|| *firstUseNodeIndex >= scheduled.size()
				|| !scheduled[*firstUseNodeIndex]) {
				if (outReason && outReason->empty()) {
					std::ostringstream oss;
					oss << "alias_activation_read_before_first_write read_pass=\"" << framePasses[passIndex].name
						<< "\" first_write_pass=\"" << (placement->firstUsePassIndex < framePasses.size() ? framePasses[placement->firstUsePassIndex].name : std::string("<unknown>"))
						<< "\" resource_id=" << requirement.resourceID;
					*outReason = oss.str();
				}
				return true;
			}
		}
		return false;
	};

	auto eraseReadyNode = [&](size_t nodeIndex) {
		ready.erase(std::remove(ready.begin(), ready.end(), nodeIndex), ready.end());
	};

	auto releaseSuccessors = [&](size_t nodeIndex) {
		for (size_t succ : nodes[nodeIndex].out) {
			if (succ >= indeg.size() || scheduled[succ]) {
				continue;
			}
			if (indeg[succ] > 0) {
				--indeg[succ];
			}
			if (indeg[succ] == 0) {
				ready.push_back(succ);
			}
		}
	};

	std::string lastSegmentReadinessFailure;
	auto segmentReplayPreflight = [&](const CachedReplaySegment& segment, std::string* outReason) {
		auto setReason = [&](std::string reason) {
			if (outReason && outReason->empty()) {
				*outReason = std::move(reason);
			}
		};
		std::unordered_set<uint64_t> internallyTransitionedResourceIDs;
		for (const auto& batchTemplate : segment.batchTemplates) {
			for (const auto& transition : batchTemplate.transitions) {
				internallyTransitionedResourceIDs.insert(transition.resourceID);
			}
		}
		for (const auto& input : segment.contract.inputRequirements) {
			if (input.queueSlot >= queueCount) {
				setReason("segment_input_queue_out_of_range");
				return false;
			}
			const auto resourceIndex = resolveTemplateResourceIndex(input.resourceID, 0, false);
			if (!resourceIndex || *resourceIndex >= m_frameCompileResources.size()) {
				setReason("segment_input_resource_index_missing");
				return false;
			}
			Resource* resource = input.resource.IsEphemeral()
				? input.resource.GetEphemeralPtr()
				: _registry.Resolve(input.resource);
			if (resource == nullptr) {
				resource = resolveTemplateResource(input.resourceID, 0, false);
			}
			if (resource == nullptr) {
				setReason("segment_input_resource_missing");
				return false;
			}
			const auto& compileResourceState = m_frameCompileResources[*resourceIndex];
			if (!compileResourceState.trackerInitialized || !compileResourceState.tracker.has_value()) {
				setReason("segment_input_tracker_uninitialized");
				return false;
			}
			if (input.transitionDiscard) {
				continue;
			}
			if (input.transitionBeforeState) {
				continue;
			}
			const bool validateAsBoundaryInput =
				internallyTransitionedResourceIDs.find(input.resourceID) == internallyTransitionedResourceIDs.end();
			if (!validateAsBoundaryInput) {
				continue;
			}
			const QueueKind queueKind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(input.queueSlot));
			const ResourceState requiredState = NormalizeStateForQueue(queueKind, input.requiredState);
			if (compileResourceState.tracker->WouldModify(input.range, requiredState)) {
				continue;
			}
		}
		for (const auto& batchTemplate : segment.batchTemplates) {
			for (const auto& signal : batchTemplate.signals) {
				if (signal.queueSlot >= queueCount) {
					setReason("template_signal_queue_out_of_range");
					return false;
				}
			}
			for (const auto& queuedPass : batchTemplate.queuedPasses) {
				if (!tryGetReplayNodeIndex(queuedPass.originalFramePassIndexAtExtraction)) {
					setReason("template_pass_missing_current_node");
					return false;
				}
				if (queuedPass.originalFramePassIndexAtExtraction >= framePasses.size() || queuedPass.queueSlot >= queueCount) {
					setReason("template_pass_out_of_range");
					return false;
				}
			}
			for (const auto& transition : batchTemplate.transitions) {
				if (transition.queueSlot >= queueCount) {
					setReason("template_transition_queue_out_of_range");
					return false;
				}
				if (resolveTemplateResource(transition.resourceID, transition.backingResourceID, transition.dynamicResource) == nullptr) {
					setReason("template_transition_resource_missing");
					return false;
				}
				if (!resolveTemplateResourceIndex(transition.resourceID, transition.backingResourceID, transition.dynamicResource)) {
					setReason("template_transition_resource_index_missing");
					return false;
				}
			}
		}
		return true;
	};
	auto segmentTemplateWaitSourcesAvailable = [&](const CachedReplaySegment& segment, std::string* outReason) {
		auto setReason = [&](std::string reason) {
			if (outReason && outReason->empty()) {
				*outReason = std::move(reason);
			}
		};
		std::vector<uint8_t> sourceSignalAvailable(queueCount, 0);
		for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
			sourceSignalAvailable[queueIndex] = latestSignalFenceByQueue[queueIndex] != 0 ? 1 : 0;
		}
		for (size_t batchIndex = 0; batchIndex < segment.batchTemplates.size(); ++batchIndex) {
			const auto& batchTemplate = segment.batchTemplates[batchIndex];
			for (const auto& wait : batchTemplate.waits) {
				if (wait.dstQueue >= queueCount || wait.srcQueue >= queueCount) {
					setReason("template_wait_queue_out_of_range");
					return false;
				}
				if (!sourceSignalAvailable[wait.srcQueue]) {
					std::ostringstream oss;
					oss << "template_wait_missing_source_signal_dependency"
						<< " local_batch=" << batchIndex
						<< " original_batch=" << batchTemplate.originalBatchIndexAtExtraction
						<< " dst_queue=" << wait.dstQueue
						<< " src_queue=" << wait.srcQueue
						<< " phase=" << static_cast<uint32_t>(wait.phase);
					setReason(oss.str());
					return false;
				}
			}
			for (const auto& signal : batchTemplate.signals) {
				if (signal.queueSlot >= queueCount) {
					setReason("template_signal_queue_out_of_range");
					return false;
				}
				sourceSignalAvailable[signal.queueSlot] = 1;
			}
		}
		return true;
	};
	auto markSegmentBoundaryWaitSources = [&](const CachedReplaySegment& segment) {
		std::unordered_set<size_t> resourceIndices;
		resourceIndices.reserve(segment.contract.inputRequirements.size() + segment.contract.queueUsage.size());
		for (const auto& input : segment.contract.inputRequirements) {
			if (auto resourceIndex = resolveTemplateResourceIndex(input.resourceID, 0, false)) {
				resourceIndices.insert(*resourceIndex);
			}
		}
		for (const auto& usage : segment.contract.queueUsage) {
			if (auto resourceIndex = resolveTemplateResourceIndex(usage.resourceID, 0, false)) {
				resourceIndices.insert(*resourceIndex);
			}
		}

		auto markSourceSignal = [&](size_t srcQueue, unsigned int sourceBatchIndex) {
			if (srcQueue >= queueCount || sourceBatchIndex == 0) {
				return;
			}
			if (sourceBatchIndex >= batches.size()) {
				return;
			}
			batches[sourceBatchIndex].MarkQueueSignal(BatchSignalPhase::AfterCompletion, srcQueue);
			latestSignalFenceByQueue[srcQueue] = std::max(
				latestSignalFenceByQueue[srcQueue],
				batches[sourceBatchIndex].GetQueueSignalFenceValue(BatchSignalPhase::AfterCompletion, srcQueue));
		};

		for (const auto& batchTemplate : segment.batchTemplates) {
			for (const auto& wait : batchTemplate.waits) {
				if (wait.srcQueue >= queueCount) {
					continue;
				}
				unsigned int latestSourceBatch = 0;
				for (size_t resourceIndex : resourceIndices) {
					latestSourceBatch = std::max(
						latestSourceBatch,
						GetFrameQueueHistoryValue(m_frameQueueLastTransitionBatch, wait.srcQueue, resourceIndex));
					latestSourceBatch = std::max(
						latestSourceBatch,
						GetFrameQueueHistoryValue(m_frameQueueLastProducerBatch, wait.srcQueue, resourceIndex));
					latestSourceBatch = std::max(
						latestSourceBatch,
						GetFrameQueueHistoryValue(m_frameQueueLastUsageBatch, wait.srcQueue, resourceIndex));
				}
				markSourceSignal(wait.srcQueue, latestSourceBatch);
			}
		}
	};
	auto segmentHasPriorActivationWrite = [&](const CachedReplaySegment& segment, uint64_t resourceID, size_t beforePassIndex) {
		for (const auto& batchTemplate : segment.batchTemplates) {
			for (const auto& queuedPass : batchTemplate.queuedPasses) {
				const uint32_t passIndex = queuedPass.originalFramePassIndexAtExtraction;
				if (passIndex == beforePassIndex) {
					return false;
				}
				if (passIndex >= m_framePassSchedulingSummaries.size()) {
					continue;
				}
				for (const auto& requirement : m_framePassSchedulingSummaries[passIndex].requirements) {
					if (requirement.resourceID != resourceID) {
						continue;
					}
					if (AccessTypeIsWriteType(requirement.state.access)
						|| requirement.state.access == rhi::ResourceAccessType::Common) {
						return true;
					}
				}
			}
		}
		return false;
	};

	auto segmentReady = [&](const CachedReplaySegment& segment, uint32_t firstNodeIndex, const std::vector<uint32_t>& segmentNodes, std::string* outReason) {
		auto setReason = [&](std::string reason) {
			if (outReason && outReason->empty()) {
				*outReason = std::move(reason);
			}
		};
		if (firstNodeIndex >= nodes.size() || scheduled[firstNodeIndex]) {
			std::ostringstream oss;
			oss << "segment_first_node_unavailable first_node=" << firstNodeIndex;
			setReason(oss.str());
			return false;
		}
		std::unordered_set<uint32_t> segmentNodeSet;
		segmentNodeSet.reserve(segmentNodes.size());
		for (uint32_t nodeIndex : segmentNodes) {
			if (nodeIndex >= nodes.size() || scheduled[nodeIndex]) {
				std::ostringstream oss;
				oss << "segment_node_unavailable node=" << nodeIndex;
				if (nodeIndex < nodes.size()) {
					oss << " pass=\"" << framePasses[nodes[nodeIndex].passIndex].name << "\"";
				}
				setReason(oss.str());
				return false;
			}
			segmentNodeSet.insert(nodeIndex);
		}
		for (uint32_t nodeIndex : segmentNodes) {
			const size_t passIndex = nodes[nodeIndex].passIndex;
			if (passIndex >= m_framePassSchedulingSummaries.size()) {
				continue;
			}
			const auto& passSummary = m_framePassSchedulingSummaries[passIndex];
			for (const auto& requirement : passSummary.requirements) {
				if (requirement.resourceIndex >= m_aliasActivationPendingByResourceIndex.size()
					|| m_aliasActivationPendingByResourceIndex[requirement.resourceIndex] == 0) {
					continue;
				}
				const bool firstUseCanActivate = AccessTypeIsWriteType(requirement.state.access)
					|| requirement.state.access == rhi::ResourceAccessType::Common;
				if (firstUseCanActivate) {
					continue;
				}
				if (segmentHasPriorActivationWrite(segment, requirement.resourceID, passIndex)) {
					continue;
				}
				const auto* placement = TryGetAliasPlacementRange(requirement.resourceID);
				if (placement == nullptr
					|| placement->firstUsePassIndex == std::numeric_limits<size_t>::max()
					|| placement->firstUsePassIndex == passIndex) {
					std::ostringstream oss;
					oss << "segment_alias_read_without_first_write pass=\"" << framePasses[passIndex].name
						<< "\" resource_id=" << requirement.resourceID;
					setReason(oss.str());
					return false;
				}
				auto firstUseNodeIndex = tryGetReplayNodeIndex(static_cast<uint32_t>(placement->firstUsePassIndex));
				if (!firstUseNodeIndex) {
					std::ostringstream oss;
					oss << "segment_alias_first_write_node_missing read_pass=\"" << framePasses[passIndex].name
						<< "\" first_write_pass=\"" << (placement->firstUsePassIndex < framePasses.size() ? framePasses[placement->firstUsePassIndex].name : std::string("<unknown>"))
						<< "\" resource_id=" << requirement.resourceID;
					setReason(oss.str());
					return false;
				}
				if (*firstUseNodeIndex >= scheduled.size()) {
					std::ostringstream oss;
					oss << "segment_alias_first_write_node_out_of_range node=" << *firstUseNodeIndex;
					setReason(oss.str());
					return false;
				}
				if (!scheduled[*firstUseNodeIndex] && segmentNodeSet.find(*firstUseNodeIndex) == segmentNodeSet.end()) {
					std::ostringstream oss;
					oss << "segment_alias_read_before_first_write read_pass=\"" << framePasses[passIndex].name
						<< "\" first_write_pass=\"" << (placement->firstUsePassIndex < framePasses.size() ? framePasses[placement->firstUsePassIndex].name : std::string("<unknown>"))
						<< "\" resource_id=" << requirement.resourceID;
					setReason(oss.str());
					return false;
				}
			}
		}
		for (uint32_t nodeIndex : segmentNodes) {
			for (size_t pred : nodes[nodeIndex].in) {
				const uint32_t predNode = static_cast<uint32_t>(pred);
				if (segmentNodeSet.find(predNode) != segmentNodeSet.end()) {
					continue;
				}
				if (pred >= scheduled.size() || !scheduled[pred]) {
					std::ostringstream oss;
					oss << "segment_external_pred_unscheduled node=" << nodeIndex
						<< " pass=\"" << framePasses[nodes[nodeIndex].passIndex].name << "\""
						<< " pred=" << pred;
					if (pred < nodes.size()) {
						oss << " pred_pass=\"" << framePasses[nodes[pred].passIndex].name << "\"";
					}
					setReason(oss.str());
					return false;
				}
			}
		}
		return true;
	};

	auto tryReplayReadySegment = [&]() -> bool {
		std::string firstReadinessFailureThisAttempt;
		for (uint32_t firstNodeIndex : segmentCandidateFirstNodes) {
			if (replayRejectedSegmentFirstNodes.contains(firstNodeIndex)) {
				continue;
			}
			auto segmentIt = segmentByFirstNode.find(firstNodeIndex);
			if (segmentIt == segmentByFirstNode.end()) {
				continue;
			}
			const auto& segmentNodes = segmentNodesByFirstNode[firstNodeIndex];
			bool segmentAlreadyTouched = false;
			for (uint32_t nodeIndex : segmentNodes) {
				if (nodeIndex < scheduled.size() && scheduled[nodeIndex]) {
					segmentAlreadyTouched = true;
					break;
				}
			}
			if (segmentAlreadyTouched) {
				continue;
			}
			std::string readinessFailure;
			if (!segmentReady(*segmentIt->second, firstNodeIndex, segmentNodes, &readinessFailure)) {
				if (firstReadinessFailureThisAttempt.empty()) {
					firstReadinessFailureThisAttempt = std::move(readinessFailure);
				}
				continue;
			}
			std::string preflightFailure;
			if (!segmentReplayPreflight(*segmentIt->second, &preflightFailure)) {
				replayRejectedSegmentFirstNodes.insert(firstNodeIndex);
				setFirstRecomputeReason(preflightFailure.empty() ? "segment_replay_preflight_failed" : preflightFailure);
				if (firstReadinessFailureThisAttempt.empty()) {
					firstReadinessFailureThisAttempt = report.firstRecomputeReason;
				}
				continue;
			}
			lastSegmentReadinessFailure.clear();

			closeCurrentBatch();
			markSegmentBoundaryWaitSources(*segmentIt->second);
			std::string waitDependencyFailure;
			if (!segmentTemplateWaitSourcesAvailable(*segmentIt->second, &waitDependencyFailure)) {
				replayRejectedSegmentFirstNodes.insert(firstNodeIndex);
				if (firstReadinessFailureThisAttempt.empty()) {
					firstReadinessFailureThisAttempt = waitDependencyFailure.empty()
						? "segment_template_wait_dependency_unavailable"
						: waitDependencyFailure;
				}
				continue;
			}
			if (!commitSegment(*segmentIt->second)) {
				return false;
			}
			if (auto replaySegmentIndexIt = replaySegmentIndexByPointer.find(segmentIt->second);
				replaySegmentIndexIt != replaySegmentIndexByPointer.end()) {
				report.reusedReplaySegmentIndices.push_back(replaySegmentIndexIt->second);
			}
			batchBuildState.ResetForNewBatch();
			currentBatchIndex = static_cast<unsigned int>(batches.size());
			closedBatchBeforeNextCommit = true;

			for (uint32_t nodeIndex : segmentNodes) {
				scheduled[nodeIndex] = 1;
				eraseReadyNode(nodeIndex);
			}
			for (uint32_t nodeIndex : segmentNodes) {
				for (size_t succ : nodes[nodeIndex].out) {
					if (succ >= nodes.size() || scheduled[succ]) {
						continue;
					}
					if (std::binary_search(segmentNodes.begin(), segmentNodes.end(), static_cast<uint32_t>(succ))) {
						continue;
					}
					if (indeg[succ] > 0) {
						--indeg[succ];
					}
					if (indeg[succ] == 0) {
						ready.push_back(succ);
					}
				}
			}
			remaining -= segmentNodes.size();
			return true;
		}
		if (!firstReadinessFailureThisAttempt.empty()) {
			lastSegmentReadinessFailure = std::move(firstReadinessFailureThisAttempt);
		}
		return false;
	};

	auto nodeBelongsToIntactReplaySegment = [&](size_t nodeIndex) {
		auto ownerIt = segmentFirstNodeByNode.find(static_cast<uint32_t>(nodeIndex));
		if (ownerIt == segmentFirstNodeByNode.end()) {
			return false;
		}
		if (replayRejectedSegmentFirstNodes.contains(ownerIt->second)) {
			return false;
		}
		const auto& segmentNodes = segmentNodesByFirstNode[ownerIt->second];
		if (segmentNodes.empty()) {
			return false;
		}
		for (uint32_t segmentNode : segmentNodes) {
			if (segmentNode < scheduled.size() && scheduled[segmentNode]) {
				return false;
			}
		}
		return true;
	};

	while (remaining > 0) {
		if (tryReplayReadySegment()) {
			continue;
		}
		if (!report.valid) {
			return report;
		}

		int bestIdxInReady = -1;
		size_t bestQueueSlot = 0;
		double bestScore = -1e300;
		uint32_t candidateChecks = 0;
		uint32_t isNewBatchNeededChecks = 0;
		const uint32_t readySetSizeBeforeEvaluate = static_cast<uint32_t>(ready.size());
		std::string firstAliasReadBlockReason;
		{
			ZoneScopedN("RenderGraph::Replay::EvaluateDynamicCandidates");
			std::vector<uint8_t> batchHasQueue(queueCount);
			for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
				batchHasQueue[queueIndex] = currentBatch.HasPasses(queueIndex);
			}

			size_t readyGraphicsCapableCount = 0;
			for (size_t ni : ready) {
				if ((nodes[ni].compatibleQueueKindMask & static_cast<uint8_t>(1u << QueueIndex(QueueKind::Graphics))) != 0) {
					++readyGraphicsCapableCount;
				}
			}
			const bool batchHasGraphicsWork = gfxSlot < batchHasQueue.size() && batchHasQueue[gfxSlot] != 0;

			for (int ri = 0; ri < static_cast<int>(ready.size()); ++ri) {
				size_t ni = ready[ri];
				auto& n = nodes[ni];
				const auto& passSummary = m_framePassSchedulingSummaries[n.passIndex];
				if (nodeBelongsToIntactReplaySegment(ni)) {
					continue;
				}
				if (isAliasActivationReadBlocked(n.passIndex, &firstAliasReadBlockReason)) {
					continue;
				}
				for (size_t nodeQueueSlot : n.compatibleQueueSlots) {
					++candidateChecks;
					if (nodeQueueSlot >= queueCount) {
						continue;
					}
					if (nodeQueueSlot >= m_activeQueueSlotsThisFrame.size() || !m_activeQueueSlotsThisFrame[nodeQueueSlot]) {
						continue;
					}

					bool hasCrossQueuePredInBatch = false;
					for (size_t pred : n.in) {
						if (!batchBuildState.ContainsNode(pred)) {
							continue;
						}
						const size_t predQueueSlot = nodes[pred].assignedQueueSlot.value_or(nodes[pred].queueSlot);
						if (predQueueSlot != nodeQueueSlot) {
							hasCrossQueuePredInBatch = true;
							break;
						}
					}
					if (hasCrossQueuePredInBatch) {
						continue;
					}

					++isNewBatchNeededChecks;
					if (IsNewBatchNeeded(
						passSummary,
						currentBatch.passBatchTrackersByResourceIndex,
						batchBuildState,
						framePasses[n.passIndex].name,
						currentBatchIndex,
						nodeQueueSlot)) {
						continue;
					}

					int reuse = 0;
					int fresh = 0;
					for (size_t resourceIndex : passSummary.touchedResourceIndices) {
						if (batchBuildState.ContainsResource(resourceIndex)) {
							++reuse;
						}
						else {
							++fresh;
						}
					}
					double score = 3.0 * reuse - 1.0 * fresh;
					if (!batchHasQueue[nodeQueueSlot]) {
						score += 2.0;
					}
					score -= 0.25 * double(currentBatch.Passes(nodeQueueSlot).size());
					score += 0.05 * double(n.criticality);

					if (framePasses[n.passIndex].type == PassType::Compute
						&& n.queueAssignmentPolicy == QueueAssignmentPolicy::Automatic) {
						const QueueKind candidateKind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(nodeQueueSlot)));
						const uint8_t candidateKindMask = static_cast<uint8_t>(1u << QueueIndex(candidateKind));
						size_t predecessorCrossQueueCount = 0;
						for (size_t pred : n.in) {
							const size_t predSlot = nodes[pred].assignedQueueSlot.value_or(nodes[pred].queueSlot);
							const QueueKind predKind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(predSlot)));
							if (predKind != candidateKind) {
								++predecessorCrossQueueCount;
							}
						}
						size_t successorCrossQueueCount = 0;
						for (size_t succ : n.out) {
							if ((nodes[succ].compatibleQueueKindMask & candidateKindMask) == 0) {
								++successorCrossQueueCount;
							}
						}
						score -= crossQueueHandoffPenalty * double(predecessorCrossQueueCount + successorCrossQueueCount);
						if (candidateKind == QueueKind::Graphics) {
							score += autoGraphicsBias;
						}
						else if (candidateKind == QueueKind::Compute) {
							const bool candidateCanAlsoRunOnGraphics = (n.compatibleQueueKindMask & static_cast<uint8_t>(1u << QueueIndex(QueueKind::Graphics))) != 0;
							const size_t otherReadyGraphicsCandidates = readyGraphicsCapableCount > 0
								? readyGraphicsCapableCount - (candidateCanAlsoRunOnGraphics ? 1u : 0u)
								: 0u;
							score += (batchHasGraphicsWork || otherReadyGraphicsCandidates > 0) ? asyncOverlapBonus : -asyncOverlapBonus;
						}
					}

					score += 1e-6 * double(nodes.size() - n.originalOrder);
					if (score > bestScore) {
						bestScore = score;
						bestIdxInReady = ri;
						bestQueueSlot = nodeQueueSlot;
					}
				}
			}
		}

		if (bestIdxInReady < 0) {
			if (currentBatchHasPasses()) {
				closeCurrentBatch();
				closedBatchBeforeNextCommit = true;
				continue;
			}
			if (ready.empty()) {
				report.valid = false;
				++report.failures;
				if (report.firstFailure.empty()) {
					report.firstFailure = "mixed replay scheduler has no ready nodes";
				}
				return report;
			}
			if (!firstAliasReadBlockReason.empty()) {
				report.valid = false;
				++report.failures;
				if (report.firstFailure.empty()) {
					report.firstFailure = firstAliasReadBlockReason;
				}
				return report;
			}
			for (int ri = 0; ri < static_cast<int>(ready.size()); ++ri) {
				if (!nodeBelongsToIntactReplaySegment(ready[ri])) {
					bestIdxInReady = ri;
					break;
				}
			}
			if (bestIdxInReady < 0) {
				report.valid = false;
				++report.failures;
				if (report.firstFailure.empty()) {
					report.firstFailure = lastSegmentReadinessFailure.empty()
						? "mixed replay scheduler has only protected segment nodes ready"
						: lastSegmentReadinessFailure;
				}
				return report;
			}
			auto& fallbackNode = nodes[ready[bestIdxInReady]];
			bestQueueSlot = fallbackNode.queueSlot;
			for (size_t compatibleSlot : fallbackNode.compatibleQueueSlots) {
				if (compatibleSlot < m_activeQueueSlotsThisFrame.size() && m_activeQueueSlotsThisFrame[compatibleSlot]) {
					bestQueueSlot = compatibleSlot;
					break;
				}
			}
		}

		const size_t chosenNodeIndex = ready[bestIdxInReady];
		auto& chosen = nodes[chosenNodeIndex];
		{
			ZoneScopedN("RenderGraph::Replay::CommitDynamicPass");
			if (!framePasses[chosen.passIndex].name.empty()) {
				ZoneText(framePasses[chosen.passIndex].name.data(), framePasses[chosen.passIndex].name.size());
			}
			const bool isolateBatch = passForcesBatchIsolation(chosen.passIndex);
			if (isolateBatch) {
				closeCurrentBatch();
				closedBatchBeforeNextCommit = true;
			}
			applyPendingSegmentInputWaits(currentBatch);
			chosen.assignedQueueSlot = bestQueueSlot;
			if (chosen.passIndex < m_assignedQueueSlotsByFramePass.size()) {
				m_assignedQueueSlotsByFramePass[chosen.passIndex] = bestQueueSlot;
			}
			m_schedulingDecisionTrace.push_back(SchedulingDecisionTrace{
				.nodeIndex = static_cast<uint32_t>(chosenNodeIndex),
				.passIndex = static_cast<uint32_t>(chosen.passIndex),
				.batchIndex = currentBatchIndex,
				.assignedQueueSlot = static_cast<uint16_t>(bestQueueSlot),
				.closedBatchBefore = closedBatchBeforeNextCommit,
				.readySetSize = readySetSizeBeforeEvaluate,
				.candidateChecks = candidateChecks,
				.isNewBatchNeededChecks = isNewBatchNeededChecks,
				.fallbackCommit = true,
			});
			closedBatchBeforeNextCommit = false;
			CommitPassToBatch(
				*this,
				framePasses[chosen.passIndex],
				chosen,
				currentBatchIndex,
				currentBatch,
				batchBuildState,
				scratchTransitioned,
				scratchFallback,
				scratchTransitions);
			updateBatchMembershipForCommittedPass(chosen);
			if (isolateBatch) {
				closeCurrentBatch();
				closedBatchBeforeNextCommit = true;
			}
		}

		batchBuildState.MarkNode(chosenNodeIndex);
		scheduled[chosenNodeIndex] = 1;
		ready[bestIdxInReady] = ready.back();
		ready.pop_back();
		releaseSuccessors(chosenNodeIndex);
		--remaining;
		++report.checkedPasses;
		++report.dynamicGapPasses;

		if (m_getHeavyDebug && m_getHeavyDebug()) {
			closeCurrentBatch();
			closedBatchBeforeNextCommit = true;
		}
	}

	closeCurrentBatch();

	CoalesceQueueWaitsAndSignals(batches);

	std::vector<std::unordered_map<uint64_t, unsigned int>> crossFrameProducer(queueCount);
	for (unsigned int batchIndex = 1; batchIndex < static_cast<unsigned int>(batches.size()); ++batchIndex) {
		auto& batch = batches[batchIndex];
		for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
			for (auto& passVariant : batch.Passes(queueIndex)) {
				std::visit([&](const auto* passEntry) {
					ForEachFrameRequirement(passEntry->resources, [&](const auto& req) {
						if (AccessTypeIsWriteType(req.state.access)) {
							crossFrameProducer[queueIndex][req.resourceHandleAndRange.resource.GetGlobalResourceID()] = batchIndex;
						}
					});
				}, passVariant);
			}
		}
	}
	m_compiledLastProducerBatchByResourceByQueue = std::move(crossFrameProducer);

	report.checkedRequirements = 0;
	for (const auto& trace : m_schedulingDecisionTrace) {
		if (trace.passIndex < m_framePassSchedulingSummaries.size()) {
			report.checkedRequirements += m_framePassSchedulingSummaries[trace.passIndex].requirements.size();
		}
	}
	report.checkedQueueSyncs = 0;
	for (const auto& batch : batches) {
		for (size_t dst = 0; dst < batch.QueueCount(); ++dst) {
			for (size_t src = 0; src < batch.QueueCount(); ++src) {
				if (dst == src) {
					continue;
				}
				for (size_t phaseIndex = 0; phaseIndex < PassBatch::kWaitPhaseCount; ++phaseIndex) {
					if (batch.HasQueueWait(static_cast<BatchWaitPhase>(phaseIndex), dst, src)) {
						++report.checkedQueueSyncs;
					}
				}
			}
		}
	}
	report.rejectedReplaySegments = replayRejectedSegmentFirstNodes.size();
	report.dynamicTraceRanges.clear();
	for (uint32_t traceIndex = 0; traceIndex < m_schedulingDecisionTrace.size();) {
		if (!m_schedulingDecisionTrace[traceIndex].fallbackCommit) {
			++traceIndex;
			continue;
		}
		const uint32_t firstTraceIndex = traceIndex;
		while (traceIndex < m_schedulingDecisionTrace.size()
			&& m_schedulingDecisionTrace[traceIndex].fallbackCommit) {
			++traceIndex;
		}
		report.dynamicTraceRanges.push_back(TraceScanRange{
			.firstTraceIndex = firstTraceIndex,
			.lastTraceIndex = traceIndex - 1,
		});
	}
	return report;
}

void RenderGraph::CompileFrame(rhi::Device device, uint8_t frameIndex, const IHostExecutionData* hostData) {
	ZoneScopedN("RenderGraph::CompileFrame");
	BeginCompileProfileFrame(frameIndex);
	auto endCompileProfileFrame = [this](RenderGraph* graph) {
		if (graph) {
			graph->EndCompileProfileFrame();
		}
	};
	std::unique_ptr<RenderGraph, decltype(endCompileProfileFrame)> compileProfileFrameGuard(this, endCompileProfileFrame);
	std::optional<rg::profile::ScopedCompileProfileStep> activeCompileProfileStep;
	const bool traceLifecycle = m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled();
	auto traceCompileStep = [&](const char* step) {
		activeCompileProfileStep.reset();
		if (traceLifecycle) {
			spdlog::info("RG frame {} compile step: {}", frameIndex, step);
		}
		if (m_compileProfileFrameActive
			&& std::string_view(step) != "begin"
			&& std::string_view(step) != "complete") {
			activeCompileProfileStep.emplace(*this, step);
		}
	};
	traceCompileStep("begin");

	{
		traceCompileStep("ResetCompileFrameScratch");
		ZoneScopedN("RenderGraph::CompileFrame::ResetCompileFrameScratch");
		m_schedulingEquivalentIDsCache.clear();
	}
	{
		traceCompileStep("ResetCompileRegionScratch");
		ZoneScopedN("RenderGraph::CompileFrame::ResetCompileRegionScratch");
		m_lastRegionStats = {};
		m_lastExtractedRegions.clear();
		m_lastRegionCandidateDiagnostics.clear();
	}
	{
		traceCompileStep("ResetCompileSchedulingScratch");
		ZoneScopedN("RenderGraph::CompileFrame::ResetCompileSchedulingScratch");
		m_schedulingDecisionTrace.clear();
		m_transitionPlacementCandidates.clear();
		m_transitionPlacementStats = {};
	}
	{
		traceCompileStep("ResetCompileCounters");
		ZoneScopedN("RenderGraph::CompileFrame::ResetCompileCounters");
		m_frameDeclarationRefreshRequestedCount = 0;
		m_frameDeclarationRefreshEquivalentCount = 0;
	}
	{
		traceCompileStep("ResetCompileAliasingScratch");
		ZoneScopedN("RenderGraph::CompileFrame::ResetCompileAliasingScratch");
		autoAliasPlannerStats = {};
		autoAliasPreviousMode = autoAliasModeLastFrame;
	}

	traceCompileStep("RefreshRetainedDeclarations");
	auto needsRefresh = [&](auto& p) -> bool {
		ZoneScopedN("RenderGraph::CompileFrame::RefreshRetainedDeclarations::NeedsRefresh");
		if (!p.name.empty()) {
			ZoneText(p.name.data(), p.name.size());
		}
		// Check if any stored resolver's content version has changed
		{
			ZoneScopedN("RenderGraph::CompileFrame::RefreshRetainedDeclarations::NeedsRefresh::ResolverSnapshots");
			TracyPlot("ORG.RefreshRetained.ResolverSnapshots", static_cast<int64_t>(p.resolverSnapshots.size()));
			for (const auto& snap : p.resolverSnapshots) {
				uint64_t cv = snap.resolver->GetContentVersion();
				if (cv != 0 && cv != snap.version) {
					ZoneScopedN("RenderGraph::CompileFrame::RefreshRetainedDeclarations::NeedsRefresh::ResolverChanged");
					return true;
				}
			}
		}

		auto hasInvalidCachedHandle = [&](const auto& resourceHandleAndRange, const char* sourceKind) -> bool {
			const auto& resource = resourceHandleAndRange.resource;
			if (resource.IsEphemeral() || _registry.IsValid(resource)) {
				return false;
			}

			spdlog::warn(
				"RG frame {} forcing retained declaration refresh for pass '{}' due to stale cached {} handle: resourceId={} registryHandleInfo='{}'",
				frameIndex,
				p.name,
				sourceKind,
				resource.GetGlobalResourceID(),
				_registry.DescribeHandle(resource));
			return true;
		};

		if (p.declarationCache.requiresStaleHandleValidation) {
			ZoneScopedN("RenderGraph::CompileFrame::RefreshRetainedDeclarations::NeedsRefresh::StaleHandleValidation");
				for (const auto& req : p.resources.staticResourceRequirements) {
					if (hasInvalidCachedHandle(req.resourceHandleAndRange, "requirement")) {
						return true;
					}
				}

				for (const auto& transition : p.resources.internalTransitions) {
					if (hasInvalidCachedHandle(transition.first, "internal-transition")) {
						return true;
					}
				}
		}

		if (!p.declarationCache.dynamicInterface) {
			// if pass doesn't opt-in, assume no change
			return false;
		}

		{
			ZoneScopedN("RenderGraph::CompileFrame::RefreshRetainedDeclarations::NeedsRefresh::DeclaredResourcesChanged");
			const std::string dynamicZoneName = p.name.empty()
				? std::string("RG::DeclaredResourcesChanged::<unnamed>")
				: std::string("RG::DeclaredResourcesChanged::") + p.name;
			ZoneNamedN(dynamicDeclaredResourcesChangedZone, "RenderGraph::DeclaredResourcesChanged", true);
			ZoneNameV(dynamicDeclaredResourcesChangedZone, dynamicZoneName.c_str(), dynamicZoneName.size());
			if (!p.name.empty()) {
				ZoneTextV(dynamicDeclaredResourcesChangedZone, p.name.data(), p.name.size());
			}
			const bool changed = p.declarationCache.dynamicInterface->DeclaredResourcesChanged();
			TracyPlot("ORG.RefreshRetained.DynamicDeclaredResourcesChanged", static_cast<int64_t>(changed ? 1 : 0));
			return changed;
		}
		};

	std::unordered_set<std::string> declarationRefreshedPassNames;
	std::unordered_set<std::string> frameExtensionPassNames;
	m_compileScratchRefreshNeededMasterIndices.clear();
	if (m_compileScratchRefreshNeededMasterIndices.capacity() < m_retainedDeclarationRefreshCandidateMasterIndices.size()) {
		m_compileScratchRefreshNeededMasterIndices.reserve(m_retainedDeclarationRefreshCandidateMasterIndices.size());
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
		ZoneScopedN("RenderGraph::CompileFrame::RefreshRetainedDeclarations::CheckCandidates");
		TracyPlot("ORG.RefreshRetained.Candidates", static_cast<int64_t>(m_retainedDeclarationRefreshCandidateMasterIndices.size()));
		for (size_t candidateIndex : m_retainedDeclarationRefreshCandidateMasterIndices) {
			ZoneScopedN("RenderGraph::CompileFrame::RefreshRetainedDeclarations::Candidate");
			ZoneValue(static_cast<uint64_t>(candidateIndex));
			++refreshCandidateCount;
			if (candidateIndex >= m_masterPassList.size()) {
				continue;
			}
			auto& pr = m_masterPassList[candidateIndex];
			if (pr.type == PassType::Compute) {
				auto& p = std::get<ComputePassAndResources>(pr.pass);
				if (!p.name.empty()) {
					ZoneText(p.name.data(), p.name.size());
				}
				if (needsRefresh(p)) {
					++refreshNeededCount;
					m_compileScratchRefreshNeededMasterIndices.push_back(candidateIndex);
				}
			}
			else if (pr.type == PassType::Render) {
				auto& p = std::get<RenderPassAndResources>(pr.pass);
				if (!p.name.empty()) {
					ZoneText(p.name.data(), p.name.size());
				}
				if (needsRefresh(p)) {
					++refreshNeededCount;
					m_compileScratchRefreshNeededMasterIndices.push_back(candidateIndex);
				}
			}
			else if (pr.type == PassType::Copy) {
				auto& p = std::get<CopyPassAndResources>(pr.pass);
				if (!p.name.empty()) {
					ZoneText(p.name.data(), p.name.size());
				}
				if (needsRefresh(p)) {
					++refreshNeededCount;
					m_compileScratchRefreshNeededMasterIndices.push_back(candidateIndex);
				}
			}
		}
		TracyPlot("ORG.RefreshRetained.CandidatesChecked", static_cast<int64_t>(refreshCandidateCount));
		TracyPlot("ORG.RefreshRetained.RefreshNeeded", static_cast<int64_t>(refreshNeededCount));
	}
	{
		traceCompileStep("RefreshRetainedDeclarationApply");
		ZoneScopedN("RenderGraph::CompileFrame::RefreshRetainedDeclarations::Apply");
		for (size_t candidateIndex : m_compileScratchRefreshNeededMasterIndices) {
			if (candidateIndex >= m_masterPassList.size()) {
				continue;
			}
			auto& pr = m_masterPassList[candidateIndex];
			if (pr.type == PassType::Compute) {
				auto& p = std::get<ComputePassAndResources>(pr.pass);
				ZoneScopedN("RenderGraph::CompileFrame::RefreshRetainedDeclarations::RefreshComputePass");
				if (!p.name.empty()) {
					ZoneText(p.name.data(), p.name.size());
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
					declarationRefreshedPassNames.insert(p.name);
				}
			}
			else if (pr.type == PassType::Render) {
				auto& p = std::get<RenderPassAndResources>(pr.pass);
				ZoneScopedN("RenderGraph::CompileFrame::RefreshRetainedDeclarations::RefreshRenderPass");
				if (!p.name.empty()) {
					ZoneText(p.name.data(), p.name.size());
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
					declarationRefreshedPassNames.insert(p.name);
				}
			}
			else if (pr.type == PassType::Copy) {
				auto& p = std::get<CopyPassAndResources>(pr.pass);
				ZoneScopedN("RenderGraph::CompileFrame::RefreshRetainedDeclarations::RefreshCopyPass");
				if (!p.name.empty()) {
					ZoneText(p.name.data(), p.name.size());
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
						declarationRefreshedPassNames.insert(p.name);
					}
			}
		}
		TracyPlot("ORG.RefreshRetained.RefreshRender", static_cast<int64_t>(renderRefreshCount));
		TracyPlot("ORG.RefreshRetained.RefreshCompute", static_cast<int64_t>(computeRefreshCount));
		TracyPlot("ORG.RefreshRetained.RefreshCopy", static_cast<int64_t>(copyRefreshCount));
		TracyPlot("ORG.RefreshRetained.RefreshChanged", static_cast<int64_t>(changedRefreshCount));
		TracyPlot("ORG.RefreshRetained.RefreshEquivalent", static_cast<int64_t>(equivalentRefreshCount));
	}
	{
		traceCompileStep("RefreshRetainedDeclarationPrune");
		ZoneScopedN("RenderGraph::CompileFrame::RefreshRetainedDeclarations::PruneCandidateList");
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
		ZoneScopedN("RenderGraph::CompileFrame::InitFramePassState");
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
		m_framePassDeclarationRefreshedThisFrame.clear();
	}

	ImmediateExecutionContext renderImmediateContext{ device,
		{rg::imm::ImmediatePassKind::Render,
		m_immediateDispatch,
		&ResolveByIdThunk,
		&ResolveByPtrThunk,
		this},
		frameIndex,
		hostData };
	ImmediateExecutionContext computeImmediateContext{ device,
		{rg::imm::ImmediatePassKind::Compute,
		m_immediateDispatch,
		&ResolveByIdThunk,
		&ResolveByPtrThunk,
		this},
		frameIndex,
		hostData };
	ImmediateExecutionContext copyImmediateContext{ device,
		{rg::imm::ImmediatePassKind::Copy,
		m_immediateDispatch,
		&ResolveByIdThunk,
		&ResolveByPtrThunk,
		this},
		frameIndex,
		hostData };
	auto getImmediateModeCommands = [](auto* pass) -> IHasImmediateModeCommands* {
		return dynamic_cast<IHasImmediateModeCommands*>(pass);
	};
	auto prepareImmediateContext = [&](ImmediateExecutionContext& context) -> ImmediateExecutionContext& {
		context.frameIndex = frameIndex;
		context.hostData = hostData;
		context.list.Reset();
		return context;
	};

	traceCompileStep("BuildFramePassList");
	{
		ZoneScopedN("RenderGraph::CompileFrame::BuildFramePassList");
		auto appendBaseFramePassRef = [&](AnyPassAndResources& pass) {
			m_baseFramePassRefs.push_back(std::addressof(pass));
		};
		auto appendBaseFramePassMove = [&](AnyPassAndResources&& pass) {
			m_frameGeneratedPasses.emplace_back(std::move(pass));
			m_baseFramePassRefs.push_back(std::addressof(m_frameGeneratedPasses.back()));
		};

		// Record immediate-mode commands + access for each pass and fold into per-frame requirements
		for (auto& pr : m_masterPassList) {

		if (pr.type == PassType::Compute) {
			auto& p = std::get<ComputePassAndResources>(pr.pass);
			auto* immediateModeCommands = getImmediateModeCommands(p.pass.get());
			if (!immediateModeCommands) {
				p.run = PassRunMask::Retained;
				appendBaseFramePassRef(pr);
				continue;
			}

			// reset per-frame only for passes that actually record immediate work
			p.immediateBytecode.clear();
			p.immediateKeepAlive.reset();
			ClearImmediateFrameRequirements(p.resources);

			auto& c = prepareImmediateContext(computeImmediateContext);

			// Record immediate-mode commands
			{
				ZoneScopedN("RenderGraph::CompileFrame::RecordImmediateCommands");
				if (!p.name.empty()) {
					ZoneText(p.name.data(), p.name.size());
				}
				if (traceLifecycle) {
					spdlog::info("RG frame {} compute pass '{}' RecordImmediateCommands begin", frameIndex, p.name);
				}
				immediateModeCommands->RecordImmediateCommands(c);
				if (traceLifecycle) {
					spdlog::info("RG frame {} compute pass '{}' RecordImmediateCommands complete", frameIndex, p.name);
				}
			}

			if (!c.list.HasRecordedWork()) {
				p.run = PassRunMask::Retained;
				appendBaseFramePassRef(pr);
				continue;
			}

			auto immediateFrameData = c.list.Finalize();
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
			auto* immediateModeCommands = getImmediateModeCommands(p.pass.get());
			if (!immediateModeCommands) {
				p.run = PassRunMask::Retained;
				appendBaseFramePassRef(pr);
				continue;
			}

			p.immediateBytecode.clear();
			p.immediateKeepAlive.reset();
			ClearImmediateFrameRequirements(p.resources);

			auto& c = prepareImmediateContext(renderImmediateContext);
			{
				ZoneScopedN("RenderGraph::CompileFrame::RecordImmediateCommands");
				if (!p.name.empty()) {
					ZoneText(p.name.data(), p.name.size());
				}
				if (traceLifecycle) {
					spdlog::info("RG frame {} render pass '{}' RecordImmediateCommands begin", frameIndex, p.name);
				}
				immediateModeCommands->RecordImmediateCommands(c);
				if (traceLifecycle) {
					spdlog::info("RG frame {} render pass '{}' RecordImmediateCommands complete", frameIndex, p.name);
				}
			}
			if (!c.list.HasRecordedWork()) {
				p.run = PassRunMask::Retained;
				appendBaseFramePassRef(pr);
				continue;
			}
			auto immediateFrameData = c.list.Finalize();

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
			auto* immediateModeCommands = getImmediateModeCommands(p.pass.get());
			if (!immediateModeCommands) {
				p.run = PassRunMask::Retained;
				appendBaseFramePassRef(pr);
				continue;
			}

			p.immediateBytecode.clear();
			p.immediateKeepAlive.reset();
			ClearImmediateFrameRequirements(p.resources);

			auto& c = prepareImmediateContext(copyImmediateContext);

			{
				ZoneScopedN("RenderGraph::CompileFrame::RecordImmediateCommands");
				if (!p.name.empty()) {
					ZoneText(p.name.data(), p.name.size());
				}
				if (traceLifecycle) {
					spdlog::info("RG frame {} copy pass '{}' RecordImmediateCommands begin", frameIndex, p.name);
				}
				immediateModeCommands->RecordImmediateCommands(c);
				if (traceLifecycle) {
					spdlog::info("RG frame {} copy pass '{}' RecordImmediateCommands complete", frameIndex, p.name);
				}
			}
			if (!c.list.HasRecordedWork()) {
				p.run = PassRunMask::Retained;
				appendBaseFramePassRef(pr);
				continue;
			}
			auto immediateFrameData = c.list.Finalize();

			bool conflict = RequirementsConflict(
				p.resources.staticResourceRequirements,
				immediateFrameData.requirements);

			if (conflict) {
				CopyPassAndResources immediatePassAndResources;
				immediatePassAndResources.pass = p.pass;
				SetImmediateFrameRequirements(immediatePassAndResources.resources, std::move(immediateFrameData.requirements));
				immediatePassAndResources.resources.preferredQueueKind = p.resources.preferredQueueKind;
				immediatePassAndResources.resources.pinnedQueueSlot = p.resources.pinnedQueueSlot;
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

	// Per-frame extension passes (ephemeral)
	// These are injected into the per-frame pass list (not m_masterPassList) so they do not accumulate.
	std::vector<ExternalPassDesc> frameExt;
	frameExt.reserve(16);
	{
		traceCompileStep("GatherFrameExtensions");
		ZoneScopedN("RenderGraph::CompileFrame::GatherFrameExtensions");
		for (auto& ext : m_extensions) {
			if (!ext) continue;
			ext->GatherFramePasses(*this, frameExt);
		}
	}

	// explicit After(anchor) edges (anchorName -> injectedName)
	std::vector<std::pair<std::string, std::string>> explicitAfterByName;
	{
		traceCompileStep("CopyStructuralExplicitEdges");
		ZoneScopedN("RenderGraph::CompileFrame::CopyStructuralExplicitEdges");
		explicitAfterByName = m_structuralExplicitAfterByName;
		explicitAfterByName.reserve(m_structuralExplicitAfterByName.size() + frameExt.size());
	}

	if (!frameExt.empty()) {
		traceCompileStep("IntegrateFrameExtensions");
		ZoneScopedN("RenderGraph::CompileFrame::IntegrateFrameExtensions");
		auto recordImmediateCommands = [&](AnyPassAndResources& pr) {
			if (pr.type == PassType::Compute) {
				auto& p = std::get<ComputePassAndResources>(pr.pass);
				auto* immediateModeCommands = getImmediateModeCommands(p.pass.get());
				if (!immediateModeCommands) {
					p.run = PassRunMask::Retained;
					return;
				}
				p.immediateBytecode.clear();
				p.immediateKeepAlive.reset();
				ClearImmediateFrameRequirements(p.resources);

				auto& c = prepareImmediateContext(computeImmediateContext);

				{
					ZoneScopedN("RenderGraph::CompileFrame::RecordImmediateCommands");
					if (!p.name.empty()) {
						ZoneText(p.name.data(), p.name.size());
					}
					if (traceLifecycle) {
						spdlog::info("RG frame {} compute pass '{}' RecordImmediateCommands begin", frameIndex, p.name);
					}
					immediateModeCommands->RecordImmediateCommands(c);
					if (traceLifecycle) {
						spdlog::info("RG frame {} compute pass '{}' RecordImmediateCommands complete", frameIndex, p.name);
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
				auto* immediateModeCommands = getImmediateModeCommands(p.pass.get());
				if (!immediateModeCommands) {
					p.run = PassRunMask::Retained;
					return;
				}
				p.immediateBytecode.clear();
				p.immediateKeepAlive.reset();
				ClearImmediateFrameRequirements(p.resources);

				auto& c = prepareImmediateContext(copyImmediateContext);

				{
					ZoneScopedN("RenderGraph::CompileFrame::RecordImmediateCommands");
					if (!p.name.empty()) {
						ZoneText(p.name.data(), p.name.size());
					}
					if (traceLifecycle) {
						spdlog::info("RG frame {} copy pass '{}' RecordImmediateCommands begin", frameIndex, p.name);
					}
					immediateModeCommands->RecordImmediateCommands(c);
					if (traceLifecycle) {
						spdlog::info("RG frame {} copy pass '{}' RecordImmediateCommands complete", frameIndex, p.name);
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
				auto* immediateModeCommands = getImmediateModeCommands(p.pass.get());
				if (!immediateModeCommands) {
					p.run = PassRunMask::Retained;
					return;
				}
				p.immediateBytecode.clear();
				p.immediateKeepAlive.reset();
				ClearImmediateFrameRequirements(p.resources);

				auto& c = prepareImmediateContext(renderImmediateContext);

				{
					ZoneScopedN("RenderGraph::CompileFrame::RecordImmediateCommands");
					if (!p.name.empty()) {
						ZoneText(p.name.data(), p.name.size());
					}
					if (traceLifecycle) {
						spdlog::info("RG frame {} render pass '{}' RecordImmediateCommands begin", frameIndex, p.name);
					}
					immediateModeCommands->RecordImmediateCommands(c);
					if (traceLifecycle) {
						spdlog::info("RG frame {} render pass '{}' RecordImmediateCommands complete", frameIndex, p.name);
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

		std::unordered_map<std::string_view, size_t> baseFramePassIndexByName;
		{
			traceCompileStep("BuildBaseFramePassIndex");
			ZoneScopedN("RenderGraph::CompileFrame::BuildBaseFramePassIndex");
			baseFramePassIndexByName.reserve(baseFramePasses.size() + frameExt.size());
			for (size_t i = 0; i < baseFramePasses.size(); ++i) {
				const AnyPassAndResources* basePass = baseFramePasses[i];
				if (basePass && !basePass->name.empty()) {
					baseFramePassIndexByName[basePass->name] = i;
				}
			}
		}

		struct PendingFrameInsert {
			AnyPassAndResources* pass = nullptr;
			size_t slotIndex = 0;
			size_t nextInsertIndex = (std::numeric_limits<size_t>::max)();
		};

		const size_t invalidInsertIndex = (std::numeric_limits<size_t>::max)();
		std::vector<PendingFrameInsert> pendingFrameInserts;
		pendingFrameInserts.reserve(frameExt.size());
		std::vector<size_t> slotHeadByIndex(baseFramePasses.size() + 1, invalidInsertIndex);
		std::vector<size_t> slotTailByIndex(baseFramePasses.size() + 1, invalidInsertIndex);
		std::unordered_map<std::string_view, size_t> pendingInsertIndexByName;
		pendingInsertIndexByName.reserve(frameExt.size());
		std::unordered_map<std::string_view, size_t> pendingInsertTailByAnchorName;
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
			ZoneScopedN("RenderGraph::CompileFrame::MaterializeFrameExtensions");
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
				pendingFrameInserts.push_back(PendingFrameInsert{ .pass = std::addressof(m_frameExtensionPasses.back()) });

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

						auto baseAnchorIt = baseFramePassIndexByName.find(a);
						if (baseAnchorIt != baseFramePassIndexByName.end()) {
							anchorName = a;
							appendPendingToSlot(pendingIndex, baseAnchorIt->second + 1);
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
					pendingInsertTailByAnchorName[anchorName] = pendingIndex;
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
			ZoneScopedN("RenderGraph::CompileFrame::AssembleFramePassList");
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
		ZoneScopedN("RenderGraph::CompileFrame::AssembleFramePassList");
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
		ZoneScopedN("RenderGraph::CompileFrame::ClassifyFramePasses");
		m_framePassIsFrameExtension.assign(m_framePasses.size(), 0);
		m_framePassDeclarationRefreshedThisFrame.assign(m_framePasses.size(), 0);
		for (size_t passIndex = 0; passIndex < m_framePasses.size(); ++passIndex) {
			const auto& passName = m_framePasses[passIndex].name;
			if (!passName.empty()) {
				m_framePassIsFrameExtension[passIndex] = frameExtensionPassNames.contains(passName) ? 1 : 0;
				m_framePassDeclarationRefreshedThisFrame[passIndex] = declarationRefreshedPassNames.contains(passName) ? 1 : 0;
			}
		}
	}

	// Register/refresh pass statistics indices for this frame's concrete pass list.
	// This supports transient passes and per-frame retained/immediate splits.
	if (m_statisticsService) {
		traceCompileStep("RegisterStatistics");
		ZoneScopedN("RenderGraph::CompileFrame::RegisterStatistics");
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
				p.statisticsIndex = static_cast<int>(m_statisticsService->RegisterPass(p.name, p.resources.isGeometryPass, p.techniquePath));
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
				p.statisticsIndex = static_cast<int>(m_statisticsService->RegisterPass(p.name, false, p.techniquePath));
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
				p.statisticsIndex = static_cast<int>(m_statisticsService->RegisterPass(p.name, false, p.techniquePath));
			}
		}

		m_statisticsService->SetupQueryHeap();
	}

	// Convert explicit After(anchorName)->(passName) constraints into node-index edges.
	std::vector<std::pair<size_t, size_t>> explicitEdges;
	{
		traceCompileStep("BuildExplicitEdges");
		ZoneScopedN("RenderGraph::CompileFrame::BuildExplicitEdges");
		explicitEdges.reserve(explicitAfterByName.size());
		if (!explicitAfterByName.empty()) {
			std::unordered_map<std::string_view, size_t> nameToIndex;
			nameToIndex.reserve(m_framePasses.size());
			for (size_t i = 0; i < m_framePasses.size(); ++i) {
				if (!m_framePasses[i].name.empty()) {
					nameToIndex[m_framePasses[i].name] = i;
				}
			}
			for (auto const& e : explicitAfterByName) {
				auto itA = nameToIndex.find(std::string_view(e.first));
				auto itB = nameToIndex.find(std::string_view(e.second));
				if (itA == nameToIndex.end() || itB == nameToIndex.end()) {
					spdlog::warn("Explicit After edge dropped (anchor='{}', pass='{}'): name not found in frame pass list.", e.first, e.second);
					continue;
				}
				explicitEdges.push_back({ itA->second, itB->second });
			}
		}
	}

	std::vector<Node> nodes;
	{
		traceCompileStep("RebuildFramePassAccessSummaries");
		ZoneScopedN("RenderGraph::CompileFrame::RebuildFramePassAccessSummaries");
		RebuildFramePassAccessSummaries();
	}
	std::span<const uint64_t> usedResourceIDs(m_frameDAGResourceIDsByIndex.data(), m_frameDAGResourceIDsByIndex.size());
	{
		traceCompileStep("ApplyIdleDematerializationPolicy");
		ZoneScopedN("RenderGraph::CompileFrame::ApplyIdleDematerializationPolicy");
		ApplyIdleDematerializationPolicy(usedResourceIDs);
	}
	{
		traceCompileStep("RebuildSchedulingEquivalentIDCacheBeforeAliasing");
		ZoneScopedN("RenderGraph::CompileFrame::RebuildSchedulingEquivalentIDCacheBeforeAliasing");
		RebuildSchedulingEquivalentIDCache(usedResourceIDs);
	}
	{
		traceCompileStep("RebuildFrameSchedulingResourceIndex");
		ZoneScopedN("RenderGraph::CompileFrame::RebuildFrameSchedulingResourceIndex");
		RebuildFrameSchedulingResourceIndex(usedResourceIDs);
	}
	{
		traceCompileStep("BuildNodes");
		ZoneScopedN("RenderGraph::CompileFrame::BuildNodes");
		nodes = BuildNodes(*this);
	}
	{
		traceCompileStep("BuildDependencyGraph");
		ZoneScopedN("RenderGraph::CompileFrame::BuildDependencyGraph");
		if (!BuildDependencyGraph(nodes, explicitEdges)) {
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
		std::vector<rg::alias::AliasSchedulingNode> aliasNodes;
		aliasNodes.reserve(nodes.size());
		for (const auto& node : nodes) {
			aliasNodes.push_back(rg::alias::AliasSchedulingNode{
				.passIndex = node.passIndex,
				.originalOrder = node.originalOrder,
				.topoRank = node.topoRank,
				.indegree = node.indegree,
				.criticality = node.criticality,
				.out = node.out,
			});
		}

		rg::alias::FrameAliasAnalysis* aliasAnalysis = nullptr;
		{
			traceCompileStep("BuildAliasFrameAnalysis");
			ZoneScopedN("RenderGraph::CompileFrame::BuildAliasFrameAnalysis");
			aliasAnalysis = &m_aliasingSubsystem.BuildAliasFrameAnalysis(*this, aliasNodes);
		}
		{
			traceCompileStep("AutoAssignAliasingPoolsFromAnalysis");
			ZoneScopedN("RenderGraph::CompileFrame::AutoAssignAliasingPoolsFromAnalysis");
			m_aliasingSubsystem.AutoAssignAliasingPoolsFromAnalysis(*this, *aliasAnalysis);
		}
		{
			traceCompileStep("BuildAliasPlanFromAnalysis");
			ZoneScopedN("RenderGraph::CompileFrame::BuildAliasPlanFromAnalysis");
			m_aliasingSubsystem.BuildAliasPlanFromAnalysis(*this, *aliasAnalysis);
		}
		{
			traceCompileStep("AddCurrentFrameAliasSchedulingEdges");
			ZoneScopedN("RenderGraph::CompileFrame::AddCurrentFrameAliasSchedulingEdges");
			if (!AddCurrentFrameAliasSchedulingEdges(nodes)) {
				spdlog::error("Render graph alias scheduling introduced a dependency cycle! Render graph compilation failed.");
				throw std::runtime_error("Render graph alias scheduling introduced a dependency cycle");
			}
		}
	}
	else {
		traceCompileStep("SkipAliasCompile");
		ZoneScopedN("RenderGraph::CompileFrame::SkipAliasCompile");
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
		m_aliasPlacementRangeByResourceIndex.assign(m_frameSchedulingResourceCount, rg::alias::AliasPlacementRange{});
		m_hasAliasPlacementByResourceIndex.assign(m_frameSchedulingResourceCount, 0);
		m_schedulingPlacementRangeByResourceIndex.assign(m_frameSchedulingResourceCount, rg::alias::AliasPlacementRange{});
		m_hasSchedulingPlacementByResourceIndex.assign(m_frameSchedulingResourceCount, 0);
		m_aliasActivationPendingByResourceIndex.assign(m_frameSchedulingResourceCount, 0);
	}
	{
		traceCompileStep("RebuildSchedulingEquivalentIDCache");
		ZoneScopedN("RenderGraph::CompileFrame::RebuildSchedulingEquivalentIDCache");
		RebuildSchedulingEquivalentIDCache(usedResourceIDs);
	}
	{
		traceCompileStep("RebuildEquivalentResourceIndicesByResourceIndex");
		ZoneScopedN("RenderGraph::CompileFrame::RebuildEquivalentResourceIndicesByResourceIndex");
		RebuildEquivalentResourceIndicesByResourceIndex();
		ResetFrameQueueBatchHistoryTables();
	}
	{
		traceCompileStep("RebuildFramePassSchedulingSummaries");
		ZoneScopedN("RenderGraph::CompileFrame::RebuildFramePassSchedulingSummaries");
		RebuildFramePassSchedulingSummaries();
	}

	{
		traceCompileStep("PlanActiveQueueSlots");
		ZoneScopedN("RenderGraph::CompileFrame::PlanActiveQueueSlots");
		PlanActiveQueueSlots(*this, m_framePasses, nodes);
	}
	{
		traceCompileStep("AssignQueueSlots");
		ZoneScopedN("RenderGraph::CompileFrame::AssignQueueSlots");
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
		ZoneScopedN("RenderGraph::CompileFrame::RebuildFrameResourceAccessSummaries");
		RebuildFrameResourceAccessSummaries(nodes);
	}
	{
		traceCompileStep("MaterializeUnmaterializedResources");
		ZoneScopedN("RenderGraph::CompileFrame::MaterializeUnmaterializedResources");
		MaterializeUnmaterializedResources(usedResourceIDs);
	}
	{
		traceCompileStep("RebuildFrameCompileResources");
		ZoneScopedN("RenderGraph::CompileFrame::RebuildFrameCompileResources");
		RebuildFrameCompileResources();
	}
	{
		traceCompileStep("SnapshotCompiledResourceGenerations");
		ZoneScopedN("RenderGraph::CompileFrame::SnapshotCompiledResourceGenerations");
		SnapshotCompiledResourceGenerations(usedResourceIDs);
	}

	const std::unordered_set<uint64_t> aliasActivationPendingBeforeAuthoritativeCompile = aliasActivationPending;
	const std::vector<uint8_t> aliasActivationPendingByResourceIndexBeforeAuthoritativeCompile = m_aliasActivationPendingByResourceIndex;
	const auto regionMode = m_getRenderGraphRegionMode
		? m_getRenderGraphRegionMode()
		: rg::runtime::RenderGraphRegionMode::Disabled;
	const bool regionDiagnosticsEnabled = m_getRenderGraphRegionDiagnosticsEnabled && m_getRenderGraphRegionDiagnosticsEnabled();
	const bool wantsAuthoritativeReplay = false;
	(void)regionMode;

	m_lastAuthoritativeReplayAttempted = false;
	m_lastAuthoritativeReplaySucceeded = false;
	m_lastAuthoritativeReplaySegments = 0;
	m_lastAuthoritativeReplayPasses = 0;
	m_lastAuthoritativeReplayDynamicGapPasses = 0;
	m_lastAuthoritativeReplayFailure.clear();
	m_lastAuthoritativeReplayRecomputeReason.clear();

	bool fastReplaySucceeded = false;
	uint64_t lightweightReplayCacheSegments = m_regionCache.replaySegments.size();
	uint64_t lightweightReplayCacheEntries = m_regionCache.replaySegmentEntries.size();
	uint64_t lightweightReplaySelectedSegments = 0;
	uint64_t lightweightReplaySelectionSkippedIneligible = 0;
	uint64_t lightweightReplaySelectionSkippedTraceRange = 0;
	uint64_t lightweightReplaySelectionSkippedPassHash = 0;
	uint64_t lightweightReplaySelectionSkippedLookupMiss = 0;
	std::string lightweightReplayFirstMiss;
	if (wantsAuthoritativeReplay
		&& !regionDiagnosticsEnabled
		&& !m_regionCache.replaySegments.empty()
		&& !m_regionCache.replaySegmentEntries.empty()) {
		traceCompileStep("ReplayCurrentFrameSegmentsFromCache");
		ZoneScopedN("RenderGraph::CompileFrame::ReplayCurrentFrameSegmentsFromCache");

		auto buildLightweightTrace = [&]() {
			std::vector<SchedulingDecisionTrace> trace;
			trace.reserve(nodes.size());
			std::vector<uint32_t> pendingPredecessors(nodes.size(), 0);
			std::vector<uint8_t> scheduled(nodes.size(), 0);
			std::vector<uint32_t> ready;
			ready.reserve(nodes.size());
			for (uint32_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
				pendingPredecessors[nodeIndex] = static_cast<uint32_t>(nodes[nodeIndex].in.size());
				if (pendingPredecessors[nodeIndex] == 0) {
					ready.push_back(nodeIndex);
				}
			}

			auto orderReady = [&]() {
				std::sort(ready.begin(), ready.end(), [&](uint32_t lhs, uint32_t rhs) {
					if (nodes[lhs].originalOrder != nodes[rhs].originalOrder) {
						return nodes[lhs].originalOrder > nodes[rhs].originalOrder;
					}
					return lhs > rhs;
				});
			};
			orderReady();

			while (!ready.empty()) {
				const uint32_t nodeIndex = ready.back();
				ready.pop_back();
				if (nodeIndex >= nodes.size() || scheduled[nodeIndex]) {
					continue;
				}
				scheduled[nodeIndex] = 1;
				auto& node = nodes[nodeIndex];
				const uint16_t assignedQueueSlot = static_cast<uint16_t>(node.assignedQueueSlot.value_or(node.queueSlot));
				trace.push_back(SchedulingDecisionTrace{
					.nodeIndex = nodeIndex,
					.passIndex = static_cast<uint32_t>(node.passIndex),
					.batchIndex = static_cast<uint32_t>(trace.size() + 1),
					.assignedQueueSlot = assignedQueueSlot,
					.closedBatchBefore = true,
					.readySetSize = static_cast<uint32_t>(ready.size() + 1),
					.candidateChecks = 0,
					.isNewBatchNeededChecks = 0,
					.fallbackCommit = false,
				});
				for (size_t succ : node.out) {
					if (succ >= pendingPredecessors.size()) {
						continue;
					}
					if (pendingPredecessors[succ] > 0) {
						--pendingPredecessors[succ];
					}
					if (pendingPredecessors[succ] == 0) {
						ready.push_back(static_cast<uint32_t>(succ));
					}
				}
				orderReady();
			}
			return trace;
		};

		auto hashSegmentTemplatePassSequence = [&](const CachedReplaySegment& segment) {
			uint64_t hash = 0x7365677061737301ull;
			uint32_t passCount = 0;
			for (const auto& batchTemplate : segment.batchTemplates) {
				for (const auto& queuedPass : batchTemplate.queuedPasses) {
					const uint32_t passIndex = queuedPass.originalFramePassIndexAtExtraction;
					if (passIndex >= m_framePasses.size()) {
						return 0ull;
					}
					const auto& pass = m_framePasses[passIndex];
					hash = HashCombine64(hash, HashString64(pass.name));
					hash = HashCombine64(hash, static_cast<uint64_t>(pass.type));
					++passCount;
				}
			}
			if (passCount != segment.identity.passCount) {
				return 0ull;
			}
			return hash;
		};

		const uint64_t replayCacheFrameSerial = ++m_regionCache.replaySegmentFrameSerial;
		const std::vector<SchedulingDecisionTrace> replayTrace = buildLightweightTrace();
		std::vector<CachedReplaySegment> selectedReplaySegments;
		selectedReplaySegments.reserve(m_regionCache.replaySegments.size());
		for (const auto& segment : m_regionCache.replaySegments) {
			if (!segment.tier1Eligible || segment.batchTemplates.empty()) {
				++lightweightReplaySelectionSkippedIneligible;
				continue;
			}
			if (hashSegmentTemplatePassSequence(segment) != segment.identity.passSequenceHash) {
				++lightweightReplaySelectionSkippedPassHash;
				if (lightweightReplayFirstMiss.empty()) {
					std::ostringstream oss;
					oss << "pass_sequence_hash_mismatch pass_sequence=0x" << std::hex << segment.identity.passSequenceHash;
					lightweightReplayFirstMiss = oss.str();
				}
				continue;
			}
			ReplaySegmentLookupResult lookup = LookupCachedReplaySegmentVariant(segment, replayCacheFrameSerial);
			if (lookup.variant == nullptr) {
				++lightweightReplaySelectionSkippedLookupMiss;
				if (lightweightReplayFirstMiss.empty()) {
					lightweightReplayFirstMiss = lookup.missReason.empty()
						? "cache_lookup_miss"
						: lookup.missReason;
				}
				continue;
			}
			const bool relaxAliasPlacement = m_renderGraphSettingsService
				? m_renderGraphSettingsService->GetRenderGraphReplayRelaxAliasPlacement()
				: true;
			if (!relaxAliasPlacement && lookup.variant->segment.fingerprint.aliasHash != segment.fingerprint.aliasHash) {
				++lightweightReplaySelectionSkippedLookupMiss;
				if (lightweightReplayFirstMiss.empty()) {
					std::ostringstream oss;
					oss << "fast_replay_alias_hash_mismatch pass_sequence=0x" << std::hex << segment.identity.passSequenceHash
						<< " alias=0x" << lookup.variant->segment.fingerprint.aliasHash
						<< "->0x" << segment.fingerprint.aliasHash;
					lightweightReplayFirstMiss = oss.str();
				}
				continue;
			}
			selectedReplaySegments.push_back(lookup.variant->segment);
		}
		lightweightReplaySelectedSegments = selectedReplaySegments.size();
		if (selectedReplaySegments.empty() && m_lastAuthoritativeReplayFailure.empty()) {
			m_lastAuthoritativeReplayFailure = lightweightReplayFirstMiss.empty()
				? "no_replay_segments_selected"
				: lightweightReplayFirstMiss;
		}

		if (!selectedReplaySegments.empty()) {
			m_lastAuthoritativeReplayAttempted = true;
			aliasActivationPending = aliasActivationPendingBeforeAuthoritativeCompile;
			m_aliasActivationPendingByResourceIndex = aliasActivationPendingByResourceIndexBeforeAuthoritativeCompile;
			ReplaySegmentVerificationReport replayReport = ReplayCurrentFrameSegmentsAsAuthoritative(
				selectedReplaySegments,
				replayTrace,
				m_framePasses,
				nodes);
			m_lastAuthoritativeReplaySegments = replayReport.matchedSegments;
			m_lastAuthoritativeReplayPasses = replayReport.replayedPasses;
			m_lastAuthoritativeReplayDynamicGapPasses = replayReport.dynamicGapPasses;
			m_lastAuthoritativeReplayRecomputeReason = replayReport.firstRecomputeReason;
			fastReplaySucceeded = replayReport.valid && replayReport.matchedSegments != 0;
			m_lastAuthoritativeReplaySucceeded = fastReplaySucceeded;
			if (!fastReplaySucceeded) {
				m_lastAuthoritativeReplayFailure = replayReport.firstFailure.empty()
					? "fast_replay_failed"
					: replayReport.firstFailure;
				batches.clear();
				batches.emplace_back(m_queueRegistry.SlotCount());
				m_schedulingDecisionTrace.clear();
				m_transitionPlacementCandidates.clear();
				m_transitionPlacementStats = {};
				ResetFrameQueueBatchHistoryTables();
				RebuildFrameCompileResources();
				aliasActivationPending = aliasActivationPendingBeforeAuthoritativeCompile;
				m_aliasActivationPendingByResourceIndex = aliasActivationPendingByResourceIndexBeforeAuthoritativeCompile;
			}
			else {
				auto addRegionStats = [](RegionCacheStats& stats, const ScheduledRegion& region) {
					++stats.candidateRegionCount;
					++stats.acceptedRegionCount;
					stats.coveredPassCount += region.passCount;
					stats.coveredRequirementCount += region.requirementCount;
					stats.coveredBatchCount += region.batchCount;
					stats.coveredTransitionCount += region.transitionCount;
					stats.largestRegionPassCount = std::max<uint64_t>(
						stats.largestRegionPassCount,
						region.passCount);
					stats.largestRegionRequirementCount = std::max<uint64_t>(
						stats.largestRegionRequirementCount,
						region.requirementCount);
					stats.estimatedSavedAddTransitionCalls += region.requirementCount;
					stats.boundaryInputEdgeCount += region.boundaryInputEdgeCount;
					stats.boundaryOutputEdgeCount += region.boundaryOutputEdgeCount;
					stats.crossQueueBoundaryInputEdgeCount += region.crossQueueBoundaryInputEdgeCount;
					stats.crossQueueBoundaryOutputEdgeCount += region.crossQueueBoundaryOutputEdgeCount;
					stats.boundarySyncCount += region.boundarySyncCount;
					stats.sameBatchPrefixPassCount += region.sameBatchPrefixPassCount;
					stats.sameBatchSuffixPassCount += region.sameBatchSuffixPassCount;
					stats.sameBatchInterleavedPassCount += region.sameBatchInterleavedPassCount;
					stats.crossQueueBoundaryPassCount += region.crossQueueBoundaryPassCount;
					stats.crossQueueTransitionCount += region.crossQueueTransitionCount;
				};

				std::vector<CachedReplaySegment> reusedReplaySegments;
				reusedReplaySegments.reserve(replayReport.reusedReplaySegmentIndices.size());
				for (size_t replaySegmentIndex : replayReport.reusedReplaySegmentIndices) {
					if (replaySegmentIndex >= selectedReplaySegments.size()) {
						continue;
					}
					reusedReplaySegments.push_back(RemapCachedReplaySegmentToCurrentFrame(
						selectedReplaySegments[replaySegmentIndex],
						batches));
				}

				std::vector<ScheduledRegion> replayRegions;
				RegionCacheStats replayRegionStats;
				std::vector<std::string> replayCandidateDiagnostics;
				std::vector<CachedReplaySegment> currentReplaySegments;
				std::vector<CachedReplaySegment> newlyExtractedReplaySegments;
				replayRegions.reserve(reusedReplaySegments.size() + replayReport.dynamicTraceRanges.size());
				currentReplaySegments.reserve(reusedReplaySegments.size() + replayReport.dynamicTraceRanges.size());

				{
					ZoneScopedN("RenderGraph::Replay::HarvestDynamicReplayGaps");
					std::vector<ScheduledRegion> dynamicReplayRegions;
					RegionCacheStats dynamicReplayRegionStats;
					if (!replayReport.dynamicTraceRanges.empty()) {
						ExtractScheduleRegionsFromAuthoritativeCompile(
							nodes,
							m_framePasses,
							batches,
							replayReport.dynamicTraceRanges,
							dynamicReplayRegions,
							dynamicReplayRegionStats,
							replayCandidateDiagnostics);
						ExtractReplaySegmentsFromAuthoritativeCompile(
							nodes,
							m_framePasses,
							batches,
							dynamicReplayRegions,
							newlyExtractedReplaySegments);
					}
					replayRegionStats = dynamicReplayRegionStats;
					replayRegions = std::move(dynamicReplayRegions);
				}

				for (const auto& segment : reusedReplaySegments) {
					replayRegions.push_back(segment.schedule);
					currentReplaySegments.push_back(segment);
					addRegionStats(replayRegionStats, segment.schedule);
				}
				currentReplaySegments.insert(
					currentReplaySegments.end(),
					newlyExtractedReplaySegments.begin(),
					newlyExtractedReplaySegments.end());

				std::sort(replayRegions.begin(), replayRegions.end(), [](const ScheduledRegion& lhs, const ScheduledRegion& rhs) {
					return lhs.firstTraceIndex < rhs.firstTraceIndex;
				});
				std::sort(currentReplaySegments.begin(), currentReplaySegments.end(), [](const CachedReplaySegment& lhs, const CachedReplaySegment& rhs) {
					return lhs.schedule.firstTraceIndex < rhs.schedule.firstTraceIndex;
				});

				m_lastExtractedRegions = replayRegions;
				m_lastRegionStats = replayRegionStats;
				m_lastRegionCandidateDiagnostics = replayCandidateDiagnostics;
				m_regionCache.regions.clear();
				m_regionCache.regions.reserve(replayRegions.size());
				for (const auto& region : replayRegions) {
					m_regionCache.regions.push_back(CachedScheduleRegion{ .schedule = region });
				}
				m_regionCache.stats = replayRegionStats;
				m_regionCache.replaySegments = currentReplaySegments;
				if (!newlyExtractedReplaySegments.empty()) {
					InsertOrRefreshReplaySegmentVariants(newlyExtractedReplaySegments, replayCacheFrameSerial);
				}
				ReplaySegmentCacheUpdateStats replayCacheMaintenanceStats{};
				EvictOldReplaySegmentVariants(replayCacheFrameSerial, replayCacheMaintenanceStats);
			}
		}
	}

	if (!fastReplaySucceeded)
	{
		traceCompileStep("AutoScheduleAndBuildBatches");
		ZoneScopedN("RenderGraph::CompileFrame::AutoScheduleAndBuildBatches");
		AutoScheduleAndBuildBatches(*this, m_framePasses, nodes);
	}
	if (!fastReplaySucceeded)
	{
		if (wantsAuthoritativeReplay) {
			traceCompileStep("ReplayCurrentFrameSegmentsAsAuthoritative");
			ZoneScopedN("RenderGraph::CompileFrame::ReplayCurrentFrameSegmentsAsAuthoritative");

			const uint64_t replayCacheFrameSerial = ++m_regionCache.replaySegmentFrameSerial;
			const std::vector<SchedulingDecisionTrace> authoritativeTrace = m_schedulingDecisionTrace;
			const std::vector<CachedReplaySegment> previousReplaySegments = m_regionCache.replaySegments;
			std::vector<ScheduledRegion> replayRegions;
			RegionCacheStats replayRegionStats;
			std::vector<std::string> replayCandidateDiagnostics;
			std::vector<CachedReplaySegment> currentReplaySegments;
			{
				ZoneScopedN("RenderGraph::Replay::ExtractCurrentSegments");
				ExtractScheduleRegionsFromAuthoritativeCompile(
					nodes,
					m_framePasses,
					batches,
					replayRegions,
					replayRegionStats,
					replayCandidateDiagnostics);
				ExtractReplaySegmentsFromAuthoritativeCompile(
					nodes,
					m_framePasses,
					batches,
					replayRegions,
					currentReplaySegments);
			}
			m_lastExtractedRegions = replayRegions;
			m_lastRegionStats = replayRegionStats;
			m_lastRegionCandidateDiagnostics = replayCandidateDiagnostics;

			ReplaySegmentValidationStats validation{};
			if (regionDiagnosticsEnabled) {
				ZoneScopedN("RenderGraph::Replay::ValidatePreviousSegments");
				validation = ValidateCachedSegmentsAgainstCurrentFrame(
					previousReplaySegments,
					currentReplaySegments);
			}
			m_regionCache.regions.clear();
			m_regionCache.regions.reserve(replayRegions.size());
			for (const auto& region : replayRegions) {
				m_regionCache.regions.push_back(CachedScheduleRegion{ .schedule = region });
			}
			m_regionCache.stats = replayRegionStats;
			m_regionCache.replaySegments = currentReplaySegments;

			if (!regionDiagnosticsEnabled) {
				InsertOrRefreshReplaySegmentVariants(currentReplaySegments, replayCacheFrameSerial);
			}
			else {
			std::unordered_map<uint64_t, const CachedReplaySegment*> previousByPassSequence;
			previousByPassSequence.reserve(previousReplaySegments.size());
			for (const auto& segment : previousReplaySegments) {
				previousByPassSequence.emplace(segment.identity.passSequenceHash, &segment);
			}

			uint64_t replaySelectedSegments = 0;
			uint64_t replaySelectedPasses = 0;
			uint64_t replaySelectionHits = 0;
			uint64_t replaySelectionMisses = 0;
			uint64_t replayAllowedSyncShapeDivergences = 0;
			uint64_t replayOlderVariantHits = 0;
			uint64_t replayOldestVariantHitAge = 0;
			std::string firstReplaySelectionMiss;
			std::string firstReplayOlderVariantHit;
			std::string firstReplaySelectionBoundaryDiff;
			std::string firstReplaySelectionTemplateDiff;
			auto boundaryEdgesReplayMatch = [](const CachedReplaySegment& previous, const CachedReplaySegment& current) {
				if (previous.contract.boundaryEdges.size() != current.contract.boundaryEdges.size()) {
					return false;
				}
				for (size_t edgeIndex = 0; edgeIndex < previous.contract.boundaryEdges.size(); ++edgeIndex) {
					const auto& lhs = previous.contract.boundaryEdges[edgeIndex];
					const auto& rhs = current.contract.boundaryEdges[edgeIndex];
					if (lhs.insideNode != rhs.insideNode
						|| lhs.outsideNode != rhs.outsideNode
						|| lhs.insideQueueSlot != rhs.insideQueueSlot
						|| lhs.outsideQueueSlot != rhs.outsideQueueSlot
						|| lhs.incoming != rhs.incoming
						|| lhs.crossQueue != rhs.crossQueue) {
						return false;
					}
				}
				return previous.schedule.boundaryInputEdgeCount == current.schedule.boundaryInputEdgeCount
					&& previous.schedule.boundaryOutputEdgeCount == current.schedule.boundaryOutputEdgeCount
					&& previous.schedule.crossQueueBoundaryInputEdgeCount == current.schedule.crossQueueBoundaryInputEdgeCount
					&& previous.schedule.crossQueueBoundaryOutputEdgeCount == current.schedule.crossQueueBoundaryOutputEdgeCount
					&& previous.schedule.sameBatchPrefixPassCount == current.schedule.sameBatchPrefixPassCount
					&& previous.schedule.sameBatchSuffixPassCount == current.schedule.sameBatchSuffixPassCount
					&& previous.schedule.sameBatchInterleavedPassCount == current.schedule.sameBatchInterleavedPassCount
					&& previous.schedule.crossQueueBoundaryPassCount == current.schedule.crossQueueBoundaryPassCount
					&& previous.schedule.crossQueueTransitionCount == current.schedule.crossQueueTransitionCount;
			};
			auto hardTemplateReplayMatch = [](const CachedReplaySegment& previous, const CachedReplaySegment& current) {
				if (previous.templateStats.batchCount != current.templateStats.batchCount
					|| previous.templateStats.partialBatchCount != current.templateStats.partialBatchCount
					|| previous.templateStats.queuedPassCount != current.templateStats.queuedPassCount
					|| previous.templateStats.transitionCount != current.templateStats.transitionCount
					|| previous.templateStats.passOrderHash != current.templateStats.passOrderHash
					|| previous.batchTemplates.size() != current.batchTemplates.size()) {
					return false;
				}
				for (size_t batchIndex = 0; batchIndex < previous.batchTemplates.size(); ++batchIndex) {
					const auto& lhsBatch = previous.batchTemplates[batchIndex];
					const auto& rhsBatch = current.batchTemplates[batchIndex];
					if (lhsBatch.partialBatch != rhsBatch.partialBatch
						|| lhsBatch.queuedPasses.size() != rhsBatch.queuedPasses.size()
						|| lhsBatch.transitions.size() != rhsBatch.transitions.size()) {
						return false;
					}
					for (size_t passIndex = 0; passIndex < lhsBatch.queuedPasses.size(); ++passIndex) {
						const auto& lhs = lhsBatch.queuedPasses[passIndex];
						const auto& rhs = rhsBatch.queuedPasses[passIndex];
						if (lhs.passNameHash != rhs.passNameHash
							|| lhs.queueSlot != rhs.queueSlot
							|| lhs.type != rhs.type) {
							return false;
						}
					}
					for (size_t transitionIndex = 0; transitionIndex < lhsBatch.transitions.size(); ++transitionIndex) {
						const auto& lhs = lhsBatch.transitions[transitionIndex];
						const auto& rhs = rhsBatch.transitions[transitionIndex];
						if (lhs.range.mipLower != rhs.range.mipLower
							|| lhs.range.mipUpper != rhs.range.mipUpper
							|| lhs.range.sliceLower != rhs.range.sliceLower
							|| lhs.range.sliceUpper != rhs.range.sliceUpper
							|| lhs.before != rhs.before
							|| lhs.after != rhs.after
							|| lhs.queueSlot != rhs.queueSlot
							|| lhs.phase != rhs.phase
							|| lhs.discard != rhs.discard
							|| lhs.dynamicResource != rhs.dynamicResource) {
							return false;
						}
					}
				}
				return true;
			};
			auto syncShapeDiverged = [](const CachedReplaySegment& previous, const CachedReplaySegment& current) {
				return previous.contract.boundarySyncs.size() != current.contract.boundarySyncs.size()
					|| previous.templateStats.waitCount != current.templateStats.waitCount
					|| previous.templateStats.signalCount != current.templateStats.signalCount
					|| previous.templateStats.syncShapeHash != current.templateStats.syncShapeHash;
			};
			auto segmentReplayMatch = [&](const CachedReplaySegment& previous, const CachedReplaySegment& current) {
				const bool relaxAliasPlacement = m_renderGraphSettingsService
					? m_renderGraphSettingsService->GetRenderGraphReplayRelaxAliasPlacement()
					: true;
				return previous.fingerprint.declarationHash == current.fingerprint.declarationHash
					&& previous.fingerprint.accessHash == current.fingerprint.accessHash
					&& previous.fingerprint.queueHash == current.fingerprint.queueHash
					&& (relaxAliasPlacement || previous.fingerprint.aliasHash == current.fingerprint.aliasHash)
					&& boundaryEdgesReplayMatch(previous, current)
					&& hardTemplateReplayMatch(previous, current);
			};
			auto boundaryCounts = [](const CachedReplaySegment& segment) {
				struct Counts {
					uint64_t incoming = 0;
					uint64_t outgoing = 0;
					uint64_t crossQueueIncoming = 0;
					uint64_t crossQueueOutgoing = 0;
				};
				Counts counts{};
				for (const auto& edge : segment.contract.boundaryEdges) {
					if (edge.incoming) {
						++counts.incoming;
						counts.crossQueueIncoming += edge.crossQueue ? 1ull : 0ull;
					}
					else {
						++counts.outgoing;
						counts.crossQueueOutgoing += edge.crossQueue ? 1ull : 0ull;
					}
				}
				return counts;
			};
			auto formatBoundaryDiff = [&](const CachedReplaySegment& previous, const CachedReplaySegment& current) {
				std::ostringstream oss;
				const auto previousCounts = boundaryCounts(previous);
				const auto currentCounts = boundaryCounts(current);
				oss << "edges=" << previous.contract.boundaryEdges.size() << "->" << current.contract.boundaryEdges.size()
					<< " incoming=" << previousCounts.incoming << "->" << currentCounts.incoming
					<< " outgoing=" << previousCounts.outgoing << "->" << currentCounts.outgoing
					<< " cross_in=" << previousCounts.crossQueueIncoming << "->" << currentCounts.crossQueueIncoming
					<< " cross_out=" << previousCounts.crossQueueOutgoing << "->" << currentCounts.crossQueueOutgoing
					<< " syncs=" << previous.contract.boundarySyncs.size() << "->" << current.contract.boundarySyncs.size()
					<< " region_counts={in:" << previous.schedule.boundaryInputEdgeCount << "->" << current.schedule.boundaryInputEdgeCount
					<< ",out:" << previous.schedule.boundaryOutputEdgeCount << "->" << current.schedule.boundaryOutputEdgeCount
					<< ",cross_in:" << previous.schedule.crossQueueBoundaryInputEdgeCount << "->" << current.schedule.crossQueueBoundaryInputEdgeCount
					<< ",cross_out:" << previous.schedule.crossQueueBoundaryOutputEdgeCount << "->" << current.schedule.crossQueueBoundaryOutputEdgeCount
					<< ",prefix:" << previous.schedule.sameBatchPrefixPassCount << "->" << current.schedule.sameBatchPrefixPassCount
					<< ",suffix:" << previous.schedule.sameBatchSuffixPassCount << "->" << current.schedule.sameBatchSuffixPassCount
					<< ",cross_trans:" << previous.schedule.crossQueueTransitionCount << "->" << current.schedule.crossQueueTransitionCount
					<< "}";
				const size_t edgeCount = std::min(previous.contract.boundaryEdges.size(), current.contract.boundaryEdges.size());
				for (size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex) {
					const auto& lhs = previous.contract.boundaryEdges[edgeIndex];
					const auto& rhs = current.contract.boundaryEdges[edgeIndex];
					if (lhs.insideNode == rhs.insideNode
						&& lhs.outsideNode == rhs.outsideNode
						&& lhs.insideTraceIndex == rhs.insideTraceIndex
						&& lhs.outsideTraceIndex == rhs.outsideTraceIndex
						&& lhs.insideQueueSlot == rhs.insideQueueSlot
						&& lhs.outsideQueueSlot == rhs.outsideQueueSlot
						&& lhs.incoming == rhs.incoming
						&& lhs.crossQueue == rhs.crossQueue) {
						continue;
					}
					oss << " first_edge_diff=index=" << edgeIndex
						<< " inside_node=" << lhs.insideNode << "->" << rhs.insideNode
						<< " outside_node=" << lhs.outsideNode << "->" << rhs.outsideNode
						<< " inside_trace=" << lhs.insideTraceIndex << "->" << rhs.insideTraceIndex
						<< " outside_trace=" << lhs.outsideTraceIndex << "->" << rhs.outsideTraceIndex
						<< " inside_queue=" << lhs.insideQueueSlot << "->" << rhs.insideQueueSlot
						<< " outside_queue=" << lhs.outsideQueueSlot << "->" << rhs.outsideQueueSlot
						<< " incoming=" << lhs.incoming << "->" << rhs.incoming
						<< " cross_queue=" << lhs.crossQueue << "->" << rhs.crossQueue;
					break;
				}
				if (previous.contract.boundarySyncs.size() != current.contract.boundarySyncs.size()) {
					return oss.str();
				}
				for (size_t syncIndex = 0; syncIndex < previous.contract.boundarySyncs.size(); ++syncIndex) {
					const auto& lhs = previous.contract.boundarySyncs[syncIndex];
					const auto& rhs = current.contract.boundarySyncs[syncIndex];
					if (lhs.dstQueue == rhs.dstQueue
						&& lhs.srcQueue == rhs.srcQueue
						&& lhs.waitPhase == rhs.waitPhase
						&& lhs.signalPhase == rhs.signalPhase
						&& lhs.internalOnly == rhs.internalOnly) {
						continue;
					}
					oss << " first_sync_diff=index=" << syncIndex
						<< " dst=" << lhs.dstQueue << "->" << rhs.dstQueue
						<< " src=" << lhs.srcQueue << "->" << rhs.srcQueue
						<< " wait_phase=" << static_cast<uint32_t>(lhs.waitPhase) << "->" << static_cast<uint32_t>(rhs.waitPhase)
						<< " signal_phase=" << static_cast<uint32_t>(lhs.signalPhase) << "->" << static_cast<uint32_t>(rhs.signalPhase)
						<< " internal=" << lhs.internalOnly << "->" << rhs.internalOnly;
					break;
				}
				return oss.str();
			};
			auto formatTemplateDiff = [](const CachedReplaySegment& previous, const CachedReplaySegment& current) {
				std::ostringstream oss;
				oss << "batches=" << previous.templateStats.batchCount << "->" << current.templateStats.batchCount
					<< " partial=" << previous.templateStats.partialBatchCount << "->" << current.templateStats.partialBatchCount
					<< " queued_passes=" << previous.templateStats.queuedPassCount << "->" << current.templateStats.queuedPassCount
					<< " transitions=" << previous.templateStats.transitionCount << "->" << current.templateStats.transitionCount
					<< " waits=" << previous.templateStats.waitCount << "->" << current.templateStats.waitCount
					<< " signals=" << previous.templateStats.signalCount << "->" << current.templateStats.signalCount
					<< " pass_order=0x" << std::hex << previous.templateStats.passOrderHash << "->0x" << current.templateStats.passOrderHash
					<< " transition_shape=0x" << previous.templateStats.transitionShapeHash << "->0x" << current.templateStats.transitionShapeHash
					<< " sync_shape=0x" << previous.templateStats.syncShapeHash << "->0x" << current.templateStats.syncShapeHash << std::dec;
				const size_t batchCount = std::min(previous.batchTemplates.size(), current.batchTemplates.size());
				for (size_t batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
					const auto& lhsBatch = previous.batchTemplates[batchIndex];
					const auto& rhsBatch = current.batchTemplates[batchIndex];
					if (lhsBatch.partialBatch != rhsBatch.partialBatch
						|| lhsBatch.queuedPasses.size() != rhsBatch.queuedPasses.size()
						|| lhsBatch.transitions.size() != rhsBatch.transitions.size()
						|| lhsBatch.waits.size() != rhsBatch.waits.size()
						|| lhsBatch.signals.size() != rhsBatch.signals.size()) {
						oss << " first_batch_shape_diff=local_batch=" << batchIndex
							<< " original=" << lhsBatch.originalBatchIndexAtExtraction << "->" << rhsBatch.originalBatchIndexAtExtraction
							<< " partial=" << lhsBatch.partialBatch << "->" << rhsBatch.partialBatch
							<< " queued=" << lhsBatch.queuedPasses.size() << "->" << rhsBatch.queuedPasses.size()
							<< " transitions=" << lhsBatch.transitions.size() << "->" << rhsBatch.transitions.size()
							<< " waits=" << lhsBatch.waits.size() << "->" << rhsBatch.waits.size()
							<< " signals=" << lhsBatch.signals.size() << "->" << rhsBatch.signals.size();
						return oss.str();
					}
					for (size_t passIndex = 0; passIndex < lhsBatch.queuedPasses.size(); ++passIndex) {
						const auto& lhs = lhsBatch.queuedPasses[passIndex];
						const auto& rhs = rhsBatch.queuedPasses[passIndex];
						if (lhs.passNameHash == rhs.passNameHash
							&& lhs.queueSlot == rhs.queueSlot
							&& lhs.type == rhs.type) {
							continue;
						}
						oss << " first_pass_diff=local_batch=" << batchIndex
							<< " pass=" << passIndex
							<< " frame_pass=" << lhs.originalFramePassIndexAtExtraction << "->" << rhs.originalFramePassIndexAtExtraction
							<< " name_hash=0x" << std::hex << lhs.passNameHash << "->0x" << rhs.passNameHash << std::dec
							<< " queue=" << lhs.queueSlot << "->" << rhs.queueSlot
							<< " type=" << static_cast<uint32_t>(lhs.type) << "->" << static_cast<uint32_t>(rhs.type);
						return oss.str();
					}
					for (size_t transitionIndex = 0; transitionIndex < lhsBatch.transitions.size(); ++transitionIndex) {
						const auto& lhs = lhsBatch.transitions[transitionIndex];
						const auto& rhs = rhsBatch.transitions[transitionIndex];
						if (lhs.resourceID == rhs.resourceID
							&& lhs.range.mipLower == rhs.range.mipLower
							&& lhs.range.mipUpper == rhs.range.mipUpper
							&& lhs.range.sliceLower == rhs.range.sliceLower
							&& lhs.range.sliceUpper == rhs.range.sliceUpper
							&& lhs.after == rhs.after
							&& lhs.queueSlot == rhs.queueSlot
							&& lhs.phase == rhs.phase
							&& lhs.discard == rhs.discard
							&& lhs.dynamicResource == rhs.dynamicResource) {
							continue;
						}
						oss << " first_transition_diff=local_batch=" << batchIndex
							<< " transition=" << transitionIndex
							<< " resource=" << lhs.resourceID << "->" << rhs.resourceID
							<< " dynamic=" << lhs.dynamicResource << "->" << rhs.dynamicResource
							<< " backing=" << lhs.backingResourceID << "->" << rhs.backingResourceID
							<< " queue=" << lhs.queueSlot << "->" << rhs.queueSlot
							<< " phase=" << static_cast<uint32_t>(lhs.phase) << "->" << static_cast<uint32_t>(rhs.phase)
							<< " discard=" << lhs.discard << "->" << rhs.discard
							<< " range=\"" << FormatRangeSpec(lhs.range) << "\"->\"" << FormatRangeSpec(rhs.range) << "\"";
						return oss.str();
					}
					for (size_t waitIndex = 0; waitIndex < lhsBatch.waits.size(); ++waitIndex) {
						const auto& lhs = lhsBatch.waits[waitIndex];
						const auto& rhs = rhsBatch.waits[waitIndex];
						if (lhs.dstQueue == rhs.dstQueue && lhs.srcQueue == rhs.srcQueue && lhs.phase == rhs.phase) {
							continue;
						}
						oss << " first_wait_diff=local_batch=" << batchIndex
							<< " wait=" << waitIndex
							<< " dst=" << lhs.dstQueue << "->" << rhs.dstQueue
							<< " src=" << lhs.srcQueue << "->" << rhs.srcQueue
							<< " phase=" << static_cast<uint32_t>(lhs.phase) << "->" << static_cast<uint32_t>(rhs.phase);
						return oss.str();
					}
					for (size_t signalIndex = 0; signalIndex < lhsBatch.signals.size(); ++signalIndex) {
						const auto& lhs = lhsBatch.signals[signalIndex];
						const auto& rhs = rhsBatch.signals[signalIndex];
						if (lhs.queueSlot == rhs.queueSlot && lhs.phase == rhs.phase) {
							continue;
						}
						oss << " first_signal_diff=local_batch=" << batchIndex
							<< " signal=" << signalIndex
							<< " queue=" << lhs.queueSlot << "->" << rhs.queueSlot
							<< " phase=" << static_cast<uint32_t>(lhs.phase) << "->" << static_cast<uint32_t>(rhs.phase);
						return oss.str();
					}
				}
				if (previous.batchTemplates.size() != current.batchTemplates.size()) {
					oss << " batch_template_count=" << previous.batchTemplates.size() << "->" << current.batchTemplates.size();
				}
				return oss.str();
			};
			std::vector<CachedReplaySegment> selectedReplaySegments;
			selectedReplaySegments.reserve(currentReplaySegments.size());
			{
				ZoneScopedN("RenderGraph::Replay::SelectCachedSegments");
				for (const auto& currentSegment : currentReplaySegments) {
					if (!currentSegment.tier1Eligible) {
						continue;
					}
					++replaySelectedSegments;
					replaySelectedPasses += currentSegment.schedule.passCount;
					ReplaySegmentLookupResult lookup = LookupCachedReplaySegmentVariant(currentSegment, replayCacheFrameSerial);
					if (lookup.variant == nullptr) {
						++replaySelectionMisses;
						if (firstReplaySelectionMiss.empty()) {
							firstReplaySelectionMiss = lookup.missReason;
							firstReplaySelectionBoundaryDiff = lookup.boundaryDiff;
							firstReplaySelectionTemplateDiff = lookup.templateDiff;
						}
						continue;
					}
					if (lookup.syncShapeDiverged) {
						++replayAllowedSyncShapeDivergences;
						if (firstReplaySelectionBoundaryDiff.empty()) {
							firstReplaySelectionBoundaryDiff = FormatReplaySegmentBoundaryDiff(lookup.variant->segment, currentSegment);
						}
						if (firstReplaySelectionTemplateDiff.empty()) {
							firstReplaySelectionTemplateDiff = FormatReplaySegmentTemplateDiff(lookup.variant->segment, currentSegment);
						}
					}
					if (lookup.variantAgeFrames > 1) {
						++replayOlderVariantHits;
						replayOldestVariantHitAge = std::max(replayOldestVariantHitAge, lookup.variantAgeFrames);
						if (firstReplayOlderVariantHit.empty()) {
							std::ostringstream oss;
							oss << "pass_sequence=0x" << std::hex << currentSegment.identity.passSequenceHash
								<< " age=" << std::dec << lookup.variantAgeFrames
								<< " seen=" << lookup.variant->seenCount
								<< " hits=" << lookup.variant->hitCount;
							firstReplayOlderVariantHit = oss.str();
						}
					}
					// The cache hit proves this current-frame segment is replay-legal; use the current
					// authoritative template so resource pointers/IDs are fresh while avoiding internal
					// transition recomputation in the replay schedule.
					selectedReplaySegments.push_back(currentSegment);
					++replaySelectionHits;
				}
			}
			ReplaySegmentCacheUpdateStats cacheUpdateStats = InsertOrRefreshReplaySegmentVariants(
				currentReplaySegments,
				replayCacheFrameSerial);
			cacheUpdateStats.lookupHits = replaySelectionHits;
			cacheUpdateStats.lookupMisses = replaySelectionMisses;
			cacheUpdateStats.olderVariantHits = replayOlderVariantHits;
			cacheUpdateStats.oldestHitAge = replayOldestVariantHitAge;
			cacheUpdateStats.firstMiss = firstReplaySelectionMiss;
			cacheUpdateStats.firstOlderHit = firstReplayOlderVariantHit;
			const bool haveStableCachedSegments = replaySelectionHits != 0 && !selectedReplaySegments.empty();

			if (regionDiagnosticsEnabled) {
				spdlog::info(
					"RG compile validation R8 pre_replay_gate frame={} status={}\n"
					"  previous_segments={} current_segments={} validation_hits={} validation_misses={}\n"
					"  selected_segments={} selected_passes={} selected_hits={} selected_misses={} partial_replay={}\n"
					"  stale_previous_misses_allowed={} allowed_sync_shape_divergences={} older_variant_hits={} oldest_hit_age={} first_validation_miss=\"{}\"\n"
					"  first_selection_miss=\"{}\"\n"
					"  first_older_variant_hit=\"{}\"\n"
					"  first_selection_boundary_diff=\"{}\"\n"
					"  first_selection_template_diff=\"{}\"",
					static_cast<unsigned int>(frameIndex),
					haveStableCachedSegments ? "ok" : "blocked",
					validation.previousSegmentCount,
					validation.currentSegmentCount,
					validation.hits,
					validation.misses,
					replaySelectedSegments,
					replaySelectedPasses,
					replaySelectionHits,
					replaySelectionMisses,
					replaySelectionMisses != 0 ? 1 : 0,
					validation.misses > replaySelectionMisses ? validation.misses - replaySelectionMisses : 0,
					replayAllowedSyncShapeDivergences,
					replayOlderVariantHits,
					replayOldestVariantHitAge,
					validation.firstMissDetail,
					firstReplaySelectionMiss,
					firstReplayOlderVariantHit,
					firstReplaySelectionBoundaryDiff,
					firstReplaySelectionTemplateDiff);

				spdlog::info(
					"RG compile validation R8 segment_cache frame={} status=observed\n"
					"  entries={} variants={} inserted={} refreshed={} evicted={} max_entries={} max_variants={} max_variants_per_key={} max_age_frames={}\n"
					"  lookup_hits={} lookup_misses={} older_variant_hits={} oldest_hit_age={}\n"
					"  first_miss=\"{}\"\n"
					"  first_older_hit=\"{}\"",
					static_cast<unsigned int>(frameIndex),
					cacheUpdateStats.entries,
					cacheUpdateStats.variants,
					cacheUpdateStats.inserted,
					cacheUpdateStats.refreshed,
					cacheUpdateStats.evicted,
					m_renderGraphSettingsService ? m_renderGraphSettingsService->GetRenderGraphReplaySegmentCacheMaxEntries() : 256u,
					m_renderGraphSettingsService ? m_renderGraphSettingsService->GetRenderGraphReplaySegmentCacheMaxVariants() : 128u,
					m_renderGraphSettingsService ? m_renderGraphSettingsService->GetRenderGraphReplaySegmentCacheMaxVariantsPerKey() : 32u,
					m_renderGraphSettingsService ? m_renderGraphSettingsService->GetRenderGraphReplaySegmentCacheMaxAgeFrames() : 0u,
					cacheUpdateStats.lookupHits,
					cacheUpdateStats.lookupMisses,
					cacheUpdateStats.olderVariantHits,
					cacheUpdateStats.oldestHitAge,
					cacheUpdateStats.firstMiss,
					cacheUpdateStats.firstOlderHit);
			}

			if (haveStableCachedSegments) {
				m_lastAuthoritativeReplayAttempted = true;
				aliasActivationPending = aliasActivationPendingBeforeAuthoritativeCompile;
				m_aliasActivationPendingByResourceIndex = aliasActivationPendingByResourceIndexBeforeAuthoritativeCompile;
				ReplaySegmentVerificationReport replayReport = ReplayCurrentFrameSegmentsAsAuthoritative(
					selectedReplaySegments,
					authoritativeTrace,
					m_framePasses,
					nodes);
				ReplaySegmentVerificationReport semanticReport = replayReport.valid
					&& regionDiagnosticsEnabled
					? VerifyAuthoritativeScheduleSemantics(nodes, m_framePasses, batches)
					: replayReport;
				m_lastAuthoritativeReplaySegments = replayReport.matchedSegments;
				m_lastAuthoritativeReplayPasses = replayReport.replayedPasses;
				m_lastAuthoritativeReplayDynamicGapPasses = replayReport.dynamicGapPasses;
				m_lastAuthoritativeReplayRecomputeReason = replayReport.firstRecomputeReason;
				m_lastAuthoritativeReplaySucceeded = replayReport.valid && semanticReport.valid;
				if (!m_lastAuthoritativeReplaySucceeded) {
					m_lastAuthoritativeReplayFailure = !replayReport.firstFailure.empty()
						? replayReport.firstFailure
						: semanticReport.firstFailure;

					batches.clear();
					batches.emplace_back(m_queueRegistry.SlotCount());
					m_schedulingDecisionTrace.clear();
					m_transitionPlacementCandidates.clear();
					m_transitionPlacementStats = {};
					ResetFrameQueueBatchHistoryTables();
					RebuildFrameCompileResources();
					aliasActivationPending = aliasActivationPendingBeforeAuthoritativeCompile;
					m_aliasActivationPendingByResourceIndex = aliasActivationPendingByResourceIndexBeforeAuthoritativeCompile;
					AutoScheduleAndBuildBatches(*this, m_framePasses, nodes);
				}
				if (regionDiagnosticsEnabled) {
					spdlog::info(
						"RG compile validation R8 partial_replay frame={} status={}\n"
						"  replayed_segments={} replayed_passes={} dynamic_gap_passes={} compiled_gap_segments={} compiled_gap_passes={}\n"
						"  transition_replay: add_transition_calls_saved={} template_batches={} repaired_batches={} recomputed_batches={} replayed_transitions={} recomputed_transition_calls={}\n"
						"  inserted_input_transitions={} verifier_status={} first_failure=\"{}\" first_recompute_reason=\"{}\" top_transition_noise=\"{}\"",
						static_cast<unsigned int>(frameIndex),
						m_lastAuthoritativeReplaySucceeded ? "ok" : "fallback_full_compile",
						replayReport.matchedSegments,
						replayReport.replayedPasses,
						replayReport.dynamicGapPasses,
						replayReport.dynamicGapPasses,
						replayReport.dynamicGapPasses,
						replayReport.addTransitionCallsSaved,
						replayReport.templateReplayedBatches,
						replayReport.repairedBatches,
						replayReport.recomputedBatches,
						replayReport.replayedInternalTransitions,
						replayReport.recomputedTransitionCalls,
						replayReport.insertedInputTransitions,
						semanticReport.valid ? "ok" : "failed",
						m_lastAuthoritativeReplayFailure,
						replayReport.firstRecomputeReason,
						replayReport.topTransitionNoise);
				}
			}
			else {
				if (m_regionCache.replaySegmentEntries.empty()) {
					m_lastAuthoritativeReplayFailure = "no_cached_segment_variants";
				}
				else if (replaySelectedSegments == 0) {
					m_lastAuthoritativeReplayFailure = "no_tier1_segments_selected";
				}
				else if (!firstReplaySelectionMiss.empty()) {
					m_lastAuthoritativeReplayFailure = firstReplaySelectionMiss;
				}
				else {
					m_lastAuthoritativeReplayFailure = "selected_cached_segments_not_fully_valid";
				}
			}
			}
		}
	}
	{
		traceCompileStep("ApplyAliasQueueSynchronization");
		ZoneScopedN("RenderGraph::CompileFrame::ApplyAliasQueueSynchronization");
		m_aliasingSubsystem.ApplyAliasQueueSynchronization(*this);
	}
	{
		traceCompileStep("CaptureCompileTrackersForExecution");
		ZoneScopedN("RenderGraph::CompileFrame::CaptureCompileTrackersForExecution");
		CaptureCompileTrackersForExecution(usedResourceIDs);
	}

	{
		traceCompileStep("PlanCrossFrameQueueWaits");
		ZoneScopedN("RenderGraph::CompileFrame::PlanCrossFrameQueueWaits");
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

		auto shouldProcessFirstUse = [&](const ResourceRegistry::RegistryHandle& handle) {
			if (handle.IsEphemeral()) {
				return false;
			}

			const uint64_t id = handle.GetGlobalResourceID();
			const auto& equivalentIDs = GetSchedulingEquivalentIDsCached(id);
			bool hasNewEquivalent = false;
			for (uint64_t rid : equivalentIDs) {
				const auto resourceIndex = TryGetFrameSchedulingResourceIndex(rid);
				if (resourceIndex.has_value()
					&& *resourceIndex < m_crossFrameFirstUseResourceEpochs.size()
					&& m_crossFrameFirstUseResourceEpochs[*resourceIndex] != firstUseEpoch) {
					hasNewEquivalent = true;
					break;
				}
			}
			if (!hasNewEquivalent) {
				return false;
			}
			for (uint64_t rid : equivalentIDs) {
				const auto resourceIndex = TryGetFrameSchedulingResourceIndex(rid);
				if (resourceIndex.has_value() && *resourceIndex < m_crossFrameFirstUseResourceEpochs.size()) {
					m_crossFrameFirstUseResourceEpochs[*resourceIndex] = firstUseEpoch;
				}
			}
			return true;
		};

		auto accumulateCrossFrameWaitForHandle = [&](size_t passQueueSlot, const ResourceRegistry::RegistryHandle& handle, std::string_view passName) {
			if (handle.IsEphemeral()) {
				return;
			}

			const uint64_t id = handle.GetGlobalResourceID();
			for (uint64_t rid : GetSchedulingEquivalentIDsCached(id)) {
				auto it = m_lastProducerByResourceAcrossFrames.find(rid);
				if (it != m_lastProducerByResourceAcrossFrames.end()) {
					markCrossFrameWait(passQueueSlot, it->second.queueSlot, it->second.fenceValue);
					if (!loggedGraphicsWaitOnCopySample && passQueueSlot == 0 && it->second.queueSlot == 2) {
						loggedGraphicsWaitOnCopySample = true;
						spdlog::warn(
							"RG PlanCrossFrameQueueWaits graphics<-copy sample: source=last_producer pass='{}' originalResource={} matchedResource={} resourceName='{}' producerFence={} producerPublishSerial={} anonymous={}",
							passName,
							id,
							rid,
							resourceDebugName(rid),
							it->second.fenceValue,
							it->second.publishSerial,
							it->second.anonymous);
					}
				}

				const auto* placement = TryGetAliasPlacementRange(rid);
				if (!placement) {
					continue;
				}

				auto itPoolState = persistentAliasPools.find(placement->poolID);
				if (itPoolState == persistentAliasPools.end()) {
					continue;
				}

				auto itPrevPool = m_lastAliasPlacementProducersByPoolAcrossFrames.find(placement->poolID);
				if (itPrevPool == m_lastAliasPlacementProducersByPoolAcrossFrames.end()) {
					continue;
				}

				const uint64_t curStart = placement->startByte;
				const uint64_t curEnd = placement->endByte;
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
							placement->poolID,
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
			}
		};

		for (size_t batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
			const auto& batch = batches[batchIndex];
			for (size_t queueIndex = 0; queueIndex < batch.QueueCount(); ++queueIndex) {
				for (const auto& passVariant : batch.Passes(queueIndex)) {
					std::visit([&](const auto* passEntry) {
						if (!passEntry) {
							return;
						}
						const std::string_view passName = passEntry->name;
						ForEachFrameRequirement(passEntry->resources, [&](const auto& req) {
							if (shouldProcessFirstUse(req.resourceHandleAndRange.resource)) {
								accumulateCrossFrameWaitForHandle(queueIndex, req.resourceHandleAndRange.resource, passName);
							}
						});
						for (auto const& tr : passEntry->resources.internalTransitions) {
							if (shouldProcessFirstUse(tr.first.resource)) {
								accumulateCrossFrameWaitForHandle(queueIndex, tr.first.resource, passName);
							}
						}
					}, passVariant);
				}
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
		ZoneScopedN("RenderGraph::CompileFrame::DeduplicateQueueWaits");
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
		ZoneScopedN("RenderGraph::CompileFrame::PruneUnusedQueueSignals");
		const size_t slotCount = m_queueRegistry.SlotCount();

		std::vector<std::unordered_map<UINT64, std::pair<size_t, BatchSignalPhase>>> signalOwnerByQueue(slotCount);
		for (size_t batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
			auto& batch = batches[batchIndex];
			for (size_t queueIndex = 0; queueIndex < batch.QueueCount(); ++queueIndex) {
				for (size_t signalPhaseIndex = 0; signalPhaseIndex < PassBatch::kSignalPhaseCount; ++signalPhaseIndex) {
					const auto signalPhase = static_cast<BatchSignalPhase>(signalPhaseIndex);
					if (!batch.HasQueueSignal(signalPhase, queueIndex)) {
						continue;
					}

					signalOwnerByQueue[queueIndex].emplace(
						batch.GetQueueSignalFenceValue(signalPhase, queueIndex),
						std::make_pair(batchIndex, signalPhase));
				}
			}
		}

		std::vector<std::array<std::vector<uint8_t>, PassBatch::kSignalPhaseCount>> requiredSignals(batches.size());
		for (auto& requiredSignalsByPhase : requiredSignals) {
			for (auto& requiredSignalsByQueue : requiredSignalsByPhase) {
				requiredSignalsByQueue.assign(slotCount, 0);
			}
		}

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
						auto itSignalOwner = signalOwnerByQueue[srcQueueIndex].find(waitFenceValue);
						if (itSignalOwner == signalOwnerByQueue[srcQueueIndex].end()) {
							continue;
						}

						const auto [signalBatchIndex, signalPhase] = itSignalOwner->second;
						requiredSignals[signalBatchIndex][static_cast<size_t>(signalPhase)][srcQueueIndex] = 1;
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

					if (!requiredSignals[batchIndex][signalPhaseIndex][queueIndex]) {
						batch.ClearQueueSignal(signalPhase, queueIndex);
					}
				}
			}
		}
	}

	{
		traceCompileStep("RegionCompileSummary");
		ZoneScopedN("RenderGraph::CompileFrame::RegionCompileSummary");
		LogRegionCompileSummary(frameIndex, nodes);
	}

	if (m_getRenderGraphLightweightCompileSummaryEnabled && m_getRenderGraphLightweightCompileSummaryEnabled()) {
		const bool replayUsed = m_lastAuthoritativeReplaySucceeded && m_lastAuthoritativeReplaySegments != 0;
		const uint64_t cachedSegmentsUsed = replayUsed ? m_lastAuthoritativeReplaySegments : 0;
		const uint64_t cachedPassesSkipped = replayUsed ? m_lastAuthoritativeReplayPasses : 0;
		const uint64_t uncachedCommitPasses = replayUsed
			? m_lastAuthoritativeReplayDynamicGapPasses
			: static_cast<uint64_t>(nodes.size());
		const char* executionPath = replayUsed ? "cached_replay" : "full_compile";
		spdlog::info(
			"RG lightweight compile summary frame={} path={} attempted={} cached_segments={} cached_passes_skipped={} uncached_commit_passes={} total_passes={} batches={}\n"
			"  cache_snapshot: segments={} entries={} selected={} skipped=[ineligible={},trace_range={},pass_hash={},lookup_miss={}] first_failure=\"{}\"",
			static_cast<unsigned int>(frameIndex),
			executionPath,
			m_lastAuthoritativeReplayAttempted ? 1 : 0,
			cachedSegmentsUsed,
			cachedPassesSkipped,
			uncachedCommitPasses,
			static_cast<uint64_t>(nodes.size()),
			static_cast<uint64_t>(batches.size() > 0 ? batches.size() - 1 : 0),
			lightweightReplayCacheSegments,
			lightweightReplayCacheEntries,
			lightweightReplaySelectedSegments,
			lightweightReplaySelectionSkippedIneligible,
			lightweightReplaySelectionSkippedTraceRange,
			lightweightReplaySelectionSkippedPassHash,
			lightweightReplaySelectionSkippedLookupMiss,
			m_lastAuthoritativeReplayFailure);
	}

	if (m_getRenderGraphCompileDumpEnabled && m_getRenderGraphCompileDumpEnabled()) {
		traceCompileStep("WriteCompiledGraphDebugDump");
		ZoneScopedN("RenderGraph::CompileFrame::WriteCompiledGraphDebugDump");
		WriteCompiledGraphDebugDump(frameIndex, nodes);
	}
	if (m_getRenderGraphVramDumpEnabled && m_getRenderGraphVramDumpEnabled()) {
		traceCompileStep("WriteVramUsageDebugDump");
		ZoneScopedN("RenderGraph::CompileFrame::WriteVramUsageDebugDump");
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

std::vector<RGCacheOverlayRange> RenderGraph::BuildReplayCacheOverlayRanges() const
{
	ZoneScopedN("RenderGraph::BuildReplayCacheOverlayRanges");
	std::vector<RGCacheOverlayRange> ranges;
	if (batches.size() <= 1) {
		return ranges;
	}

	if (!m_lastAuthoritativeReplaySucceeded) {
		ranges.push_back(RGCacheOverlayRange{
			.firstBatch = 1,
			.lastBatch = static_cast<uint32_t>(batches.size() - 1),
			.cached = false,
		});
		return ranges;
	}

	enum class BatchCacheState : uint8_t {
		Unknown,
		Cached,
		Uncached,
	};

	std::vector<BatchCacheState> batchStates(batches.size(), BatchCacheState::Unknown);
	for (const auto& trace : m_schedulingDecisionTrace) {
		if (trace.batchIndex == 0 || trace.batchIndex >= batchStates.size()) {
			continue;
		}
		const BatchCacheState state = trace.fallbackCommit
			? BatchCacheState::Uncached
			: BatchCacheState::Cached;
		if (batchStates[trace.batchIndex] == BatchCacheState::Unknown
			|| batchStates[trace.batchIndex] == BatchCacheState::Cached) {
			batchStates[trace.batchIndex] = state;
		}
	}

	for (uint32_t batchIndex = 1; batchIndex < static_cast<uint32_t>(batchStates.size());) {
		BatchCacheState state = batchStates[batchIndex];
		if (state == BatchCacheState::Unknown) {
			state = BatchCacheState::Uncached;
		}

		const uint32_t firstBatch = batchIndex;
		while (batchIndex + 1 < batchStates.size()) {
			BatchCacheState nextState = batchStates[batchIndex + 1];
			if (nextState == BatchCacheState::Unknown) {
				nextState = BatchCacheState::Uncached;
			}
			if (nextState != state) {
				break;
			}
			++batchIndex;
		}

		ranges.push_back(RGCacheOverlayRange{
			.firstBatch = firstBatch,
			.lastBatch = batchIndex,
			.cached = state == BatchCacheState::Cached,
		});
		++batchIndex;
	}

	return ranges;
}

RenderGraph::ReplayAuthoritativeReadinessReport RenderGraph::CheckReplayAuthoritativeReadiness(
	const ReplaySegmentValidationStats& validation,
	const ReplaySegmentVerificationReport& semanticVerification,
	const ReplaySegmentVerificationReport& replayMetadataVerification,
	const ReplaySegmentVerificationReport& shadowReplayVerification) const
{
	ZoneScopedN("RenderGraph::CheckReplayAuthoritativeReadiness");
	ReplayAuthoritativeReadinessReport report{};
	report.matchedSegments = m_lastAuthoritativeReplaySucceeded ? m_lastAuthoritativeReplaySegments : shadowReplayVerification.matchedSegments;
	report.replayablePasses = m_lastAuthoritativeReplaySucceeded ? m_lastAuthoritativeReplayPasses : shadowReplayVerification.replayedPasses;
	report.dynamicGapPasses = m_lastAuthoritativeReplaySucceeded ? m_lastAuthoritativeReplayDynamicGapPasses : shadowReplayVerification.dynamicGapPasses;
	report.insertedInputTransitions = shadowReplayVerification.insertedInputTransitions;

	auto addBlocker = [&](std::string_view blocker) {
		if (!report.blockerSummary.empty()) {
			report.blockerSummary += ",";
		}
		report.blockerSummary += blocker;
		++report.blockers;
	};

	if (!semanticVerification.valid) {
		addBlocker("authoritative_semantic_verifier_failed");
	}
	if (!replayMetadataVerification.valid) {
		addBlocker("replay_metadata_verifier_failed");
	}
	if (!m_lastAuthoritativeReplaySucceeded && !shadowReplayVerification.valid) {
		addBlocker("shadow_replay_verifier_failed");
	}
	if (!m_lastAuthoritativeReplaySucceeded && validation.previousSegmentCount != 0 && validation.misses != 0) {
		addBlocker("cached_segment_misses");
	}
	if (!m_lastAuthoritativeReplaySucceeded && shadowReplayVerification.replayedPasses == 0) {
		addBlocker("no_replayable_passes");
	}

	if (!m_lastAuthoritativeReplaySucceeded) {
		addBlocker("authoritative_replay_not_committed");
		if (!m_lastAuthoritativeReplayFailure.empty()) {
			addBlocker(m_lastAuthoritativeReplayFailure);
		}
	}

	report.ready = report.blockers == 0;
	if (report.blockerSummary.empty()) {
		report.blockerSummary = "none";
	}
	return report;
}

void RenderGraph::LogRegionCompileSummary(uint8_t frameIndex, const std::vector<Node>& nodes) {
	const auto regionMode = m_getRenderGraphRegionMode
		? m_getRenderGraphRegionMode()
		: rg::runtime::RenderGraphRegionMode::Disabled;
	const auto transitionPlacementMode = m_getTransitionPlacementMode
		? m_getTransitionPlacementMode()
		: rg::runtime::TransitionPlacementMode::InlineEarlyPlacement;
	const bool diagnosticsEnabled = m_getRenderGraphRegionDiagnosticsEnabled && m_getRenderGraphRegionDiagnosticsEnabled();
	if (!diagnosticsEnabled) {
		return;
	}

	std::string traceSummary;
	const bool traceValid = ValidateSchedulingDecisionTrace(nodes, m_framePasses, batches, traceSummary);
	spdlog::info(
		"RG compile validation M1.1 frame={} status={}\n  {}",
		static_cast<unsigned int>(frameIndex),
		traceValid ? "ok" : "failed",
		traceSummary);

	auto hashState = [](uint64_t seed, const ResourceState& state) {
		seed = HashCombine64(seed, static_cast<uint64_t>(state.access));
		seed = HashCombine64(seed, static_cast<uint64_t>(state.layout));
		seed = HashCombine64(seed, static_cast<uint64_t>(state.sync));
		return seed;
	};
	auto hashBound = [](uint64_t seed, const auto& bound) {
		seed = HashCombine64(seed, static_cast<uint64_t>(bound.type));
		seed = HashCombine64(seed, static_cast<uint64_t>(bound.value));
		return seed;
	};
	auto hashRange = [&](uint64_t seed, const RangeSpec& range) {
		seed = hashBound(seed, range.mipLower);
		seed = hashBound(seed, range.mipUpper);
		seed = hashBound(seed, range.sliceLower);
		seed = hashBound(seed, range.sliceUpper);
		return seed;
	};
	uint64_t finalTrackerFingerprint = 0xcbf29ce484222325ull;
	size_t initializedTrackerCount = 0;
	size_t trackerSegmentCount = 0;
	for (const auto& compileResource : m_frameCompileResources) {
		finalTrackerFingerprint = HashCombine64(finalTrackerFingerprint, compileResource.resourceID);
		finalTrackerFingerprint = HashCombine64(finalTrackerFingerprint, compileResource.trackerInitialized ? 1ull : 0ull);
		if (!compileResource.trackerInitialized || !compileResource.tracker.has_value()) {
			continue;
		}
		++initializedTrackerCount;
		const auto& segments = compileResource.tracker->GetSegments();
		trackerSegmentCount += segments.size();
		finalTrackerFingerprint = HashCombine64(finalTrackerFingerprint, segments.size());
		for (const auto& segment : segments) {
			finalTrackerFingerprint = hashRange(finalTrackerFingerprint, segment.rangeSpec);
			finalTrackerFingerprint = hashState(finalTrackerFingerprint, segment.state);
		}
	}
	uint64_t emittedTransitionFingerprint = 0x84222325cbf29ce4ull;
	size_t emittedTransitionCount = 0;
	for (const auto& batch : batches) {
		for (size_t queueIndex = 0; queueIndex < batch.QueueCount(); ++queueIndex) {
			for (size_t phaseIndex = 0; phaseIndex < static_cast<size_t>(BatchTransitionPhase::Count); ++phaseIndex) {
				const auto phase = static_cast<BatchTransitionPhase>(phaseIndex);
				for (const auto& transition : batch.Transitions(queueIndex, phase)) {
					++emittedTransitionCount;
					emittedTransitionFingerprint = HashCombine64(emittedTransitionFingerprint, queueIndex);
					emittedTransitionFingerprint = HashCombine64(emittedTransitionFingerprint, phaseIndex);
					emittedTransitionFingerprint = HashCombine64(emittedTransitionFingerprint, transition.pResource ? transition.pResource->GetGlobalResourceID() : 0ull);
					emittedTransitionFingerprint = hashRange(emittedTransitionFingerprint, transition.range);
					emittedTransitionFingerprint = HashCombine64(emittedTransitionFingerprint, static_cast<uint64_t>(transition.prevAccessType));
					emittedTransitionFingerprint = HashCombine64(emittedTransitionFingerprint, static_cast<uint64_t>(transition.newAccessType));
					emittedTransitionFingerprint = HashCombine64(emittedTransitionFingerprint, static_cast<uint64_t>(transition.prevLayout));
					emittedTransitionFingerprint = HashCombine64(emittedTransitionFingerprint, static_cast<uint64_t>(transition.newLayout));
					emittedTransitionFingerprint = HashCombine64(emittedTransitionFingerprint, static_cast<uint64_t>(transition.prevSyncState));
					emittedTransitionFingerprint = HashCombine64(emittedTransitionFingerprint, static_cast<uint64_t>(transition.newSyncState));
					emittedTransitionFingerprint = HashCombine64(emittedTransitionFingerprint, transition.discard ? 1ull : 0ull);
				}
			}
		}
	}

	spdlog::info(
		"RG compile validation M2 frame={} status=observed\n"
		"  transition_mode={}\n"
		"  candidates={} emitted={} old_inline_eligible={} inline_early_placed={} consumer_before_pass={}\n"
		"  graphics_fallback={} alias_activation={} cross_queue_coordination_blocked={}\n"
		"  placement_candidates_recorded={}\n"
		"  final_tracker_fingerprint=0x{:016x} initialized_trackers={} tracker_segments={}\n"
		"  emitted_transition_fingerprint=0x{:016x} emitted_transitions_in_batches={}",
		static_cast<unsigned int>(frameIndex),
		TransitionPlacementModeToString(transitionPlacementMode),
		m_transitionPlacementStats.candidateCount,
		m_transitionPlacementStats.emittedTransitionCount,
		m_transitionPlacementStats.oldInlineEarlyEligibleCount,
		m_transitionPlacementStats.inlineEarlyPlacedCount,
		m_transitionPlacementStats.canonicalBeforePassCount,
		m_transitionPlacementStats.graphicsFallbackCount,
		m_transitionPlacementStats.aliasActivationCount,
		m_transitionPlacementStats.crossQueueCoordinationBlockedCount,
		m_transitionPlacementCandidates.size(),
		finalTrackerFingerprint,
		initializedTrackerCount,
		trackerSegmentCount,
		emittedTransitionFingerprint,
		emittedTransitionCount);

	spdlog::info(
		"RG compile validation M2.1 frame={} status=active\n"
		"  queue_wait_cleanup=CoalesceQueueWaitsAndSignals\n"
		"  call_site=AutoScheduleAndBuildBatches",
		static_cast<unsigned int>(frameIndex));

	if (static_cast<uint8_t>(regionMode) >= static_cast<uint8_t>(rg::runtime::RenderGraphRegionMode::ExtractOnly)) {
		const std::vector<CachedReplaySegment> previousReplaySegments = m_regionCache.replaySegments;
		const bool preserveAuthoritativeReplayCache =
			static_cast<uint8_t>(regionMode) >= static_cast<uint8_t>(rg::runtime::RenderGraphRegionMode::ReplayAuthoritative);
		if (!preserveAuthoritativeReplayCache || !m_lastAuthoritativeReplayAttempted) {
			ExtractScheduleRegionsFromAuthoritativeCompile(
				nodes,
				m_framePasses,
				batches,
				m_lastExtractedRegions,
				m_lastRegionStats,
				m_lastRegionCandidateDiagnostics);
			m_regionCache.regions.clear();
			m_regionCache.regions.reserve(m_lastExtractedRegions.size());
			for (const auto& region : m_lastExtractedRegions) {
				m_regionCache.regions.push_back(CachedScheduleRegion{ .schedule = region });
			}
			m_regionCache.stats = m_lastRegionStats;
		}
		std::vector<CachedReplaySegment> diagnosticReplaySegments;
		if (preserveAuthoritativeReplayCache && m_lastAuthoritativeReplayAttempted) {
			diagnosticReplaySegments = m_regionCache.replaySegments;
		}
		else {
			ExtractReplaySegmentsFromAuthoritativeCompile(
				nodes,
				m_framePasses,
				batches,
				m_lastExtractedRegions,
				diagnosticReplaySegments);
			m_regionCache.replaySegments = diagnosticReplaySegments;
		}
		const auto& replaySegmentsForDiagnostics = preserveAuthoritativeReplayCache
			? diagnosticReplaySegments
			: m_regionCache.replaySegments;
		const ReplaySegmentValidationStats segmentValidation = ValidateCachedSegmentsAgainstCurrentFrame(
			previousReplaySegments,
			replaySegmentsForDiagnostics);
		const ReplaySegmentVerificationReport semanticVerification = VerifyAuthoritativeScheduleSemantics(
			nodes,
			m_framePasses,
			batches);
		const ReplaySegmentVerificationReport replayMetadataVerification = VerifyReplayScheduleSemanticCorrectness(
			replaySegmentsForDiagnostics);
		ReplaySegmentVerificationReport shadowReplayVerification{};
		if (static_cast<uint8_t>(regionMode) >= static_cast<uint8_t>(rg::runtime::RenderGraphRegionMode::ShadowReplay)) {
			shadowReplayVerification = BuildShadowReplayScheduleFromCachedSegments(
				previousReplaySegments,
				replaySegmentsForDiagnostics);
		}
		const ReplayAuthoritativeReadinessReport authoritativeReplayReadiness =
			static_cast<uint8_t>(regionMode) >= static_cast<uint8_t>(rg::runtime::RenderGraphRegionMode::ShadowReplay)
				? CheckReplayAuthoritativeReadiness(
					segmentValidation,
					semanticVerification,
					replayMetadataVerification,
					shadowReplayVerification)
				: ReplayAuthoritativeReadinessReport{};

		auto rejectReasonName = [](RegionRejectReason reason) noexcept -> const char* {
			switch (reason) {
			case RegionRejectReason::QueueSlotChange: return "queue_change";
			case RegionRejectReason::PassCountBelowThreshold: return "pass_count_below_threshold";
			case RegionRejectReason::ImmediateWork: return "immediate_work";
			case RegionRejectReason::FrameExtensionPass: return "frame_extension_pass";
			case RegionRejectReason::DeclarationRefreshedThisFrame: return "declaration_refreshed";
			case RegionRejectReason::InteriorIncomingEdge: return "interior_incoming_edge";
			case RegionRejectReason::InteriorOutgoingEdge: return "interior_outgoing_edge";
			case RegionRejectReason::AliasActivation: return "alias_activation";
			case RegionRejectReason::AliasPlacementInstability: return "alias_placement_instability";
			case RegionRejectReason::CrossQueueSync: return "cross_queue_sync";
			case RegionRejectReason::GraphicsFallbackTransition: return "graphics_fallback_transition";
			case RegionRejectReason::UnsupportedSubresourceState: return "unsupported_subresource_state";
			case RegionRejectReason::BatchHazardBoundary: return "batch_hazard_boundary";
			default: return "unknown";
			}
		};
		auto invalidationReasonName = [](ReplaySegmentInvalidationReason reason) noexcept -> const char* {
			switch (reason) {
			case ReplaySegmentInvalidationReason::None: return "none";
			case ReplaySegmentInvalidationReason::PassSequenceChanged: return "pass_sequence_changed";
			case ReplaySegmentInvalidationReason::DeclarationChanged: return "declaration_changed";
			case ReplaySegmentInvalidationReason::AccessChanged: return "access_changed";
			case ReplaySegmentInvalidationReason::QueueAssignmentChanged: return "queue_assignment_changed";
			case ReplaySegmentInvalidationReason::AliasPlacementChanged: return "alias_placement_changed";
			case ReplaySegmentInvalidationReason::BoundaryChanged: return "boundary_changed";
			case ReplaySegmentInvalidationReason::TemplateShapeChanged: return "template_shape_changed";
			case ReplaySegmentInvalidationReason::TemplateStateChanged: return "template_state_changed";
			case ReplaySegmentInvalidationReason::ImmediateWorkInserted: return "immediate_work_inserted";
			case ReplaySegmentInvalidationReason::FrameExtensionInserted: return "frame_extension_inserted";
			case ReplaySegmentInvalidationReason::UnsupportedAliasActivation: return "unsupported_alias_activation";
			case ReplaySegmentInvalidationReason::BatchInterleavingChanged: return "batch_interleaving_changed";
			default: return "unknown";
			}
		};

		std::ostringstream rejectSummary;
		bool firstReject = true;
		for (size_t reasonIndex = 0; reasonIndex < static_cast<size_t>(RegionRejectReason::Count); ++reasonIndex) {
			const uint64_t count = m_lastRegionStats.rejectedByReason[reasonIndex];
			if (count == 0) {
				continue;
			}
			if (!firstReject) {
				rejectSummary << ",";
			}
			firstReject = false;
			rejectSummary << rejectReasonName(static_cast<RegionRejectReason>(reasonIndex)) << "=" << count;
		}
		if (firstReject) {
			rejectSummary << "none";
		}
		std::ostringstream invalidationSummary;
		bool firstInvalidation = true;
		for (size_t reasonIndex = 1; reasonIndex < static_cast<size_t>(ReplaySegmentInvalidationReason::Count); ++reasonIndex) {
			const uint64_t count = segmentValidation.missesByReason[reasonIndex];
			if (count == 0) {
				continue;
			}
			if (!firstInvalidation) {
				invalidationSummary << ",";
			}
			firstInvalidation = false;
			invalidationSummary << invalidationReasonName(static_cast<ReplaySegmentInvalidationReason>(reasonIndex)) << "=" << count;
		}
		if (firstInvalidation) {
			invalidationSummary << "none";
		}

		uint64_t segmentInputCount = 0;
		uint64_t segmentOutputCount = 0;
		uint64_t segmentBoundaryEdgeCount = 0;
		uint64_t segmentBoundarySyncCount = 0;
		uint64_t segmentQueueUsageCount = 0;
		uint64_t segmentBatchTemplateCount = 0;
		uint64_t segmentQueuedPassTemplateCount = 0;
		uint64_t segmentTransitionTemplateCount = 0;
		uint64_t segmentWaitTemplateCount = 0;
		uint64_t segmentSignalTemplateCount = 0;
		uint64_t tier1EligibleCount = 0;
		uint64_t segmentFingerprint = 0x7265706c61790001ull;
		const bool relaxAliasPlacementForDiagnostics = m_renderGraphSettingsService
			? m_renderGraphSettingsService->GetRenderGraphReplayRelaxAliasPlacement()
			: true;
		for (const auto& segment : replaySegmentsForDiagnostics) {
			segmentInputCount += segment.contract.inputRequirements.size();
			segmentOutputCount += segment.contract.outputStates.size();
			segmentBoundaryEdgeCount += segment.contract.boundaryEdges.size();
			segmentBoundarySyncCount += segment.contract.boundarySyncs.size();
			segmentQueueUsageCount += segment.contract.queueUsage.size();
			segmentBatchTemplateCount += segment.batchTemplates.size();
			tier1EligibleCount += segment.tier1Eligible ? 1ull : 0ull;
			segmentFingerprint = HashCombine64(segmentFingerprint, segment.identity.passSequenceHash);
			segmentFingerprint = HashCombine64(segmentFingerprint, segment.identity.structuralPositionHash);
			segmentFingerprint = HashCombine64(segmentFingerprint, segment.fingerprint.declarationHash);
			segmentFingerprint = HashCombine64(segmentFingerprint, segment.fingerprint.accessHash);
			segmentFingerprint = HashCombine64(segmentFingerprint, segment.fingerprint.queueHash);
			if (!relaxAliasPlacementForDiagnostics) {
				segmentFingerprint = HashCombine64(segmentFingerprint, segment.fingerprint.aliasHash);
			}
			segmentFingerprint = HashCombine64(segmentFingerprint, segment.fingerprint.boundaryHash);
			segmentFingerprint = HashCombine64(segmentFingerprint, segment.fingerprint.templateShapeHash);
			for (const auto& batchTemplate : segment.batchTemplates) {
				segmentQueuedPassTemplateCount += batchTemplate.queuedPasses.size();
				segmentTransitionTemplateCount += batchTemplate.transitions.size();
				segmentWaitTemplateCount += batchTemplate.waits.size();
				segmentSignalTemplateCount += batchTemplate.signals.size();
			}
		}

		spdlog::info(
			"RG compile validation M1 frame={} status=observed\n"
			"  region_mode={} transition_mode={} segment_extraction=enabled\n"
			"  candidates={} accepted={} rejected={} rejects=[{}]\n"
			"  declarations: refresh_requested={} equivalent_refreshes={} changed_refreshes={}\n"
			"  boundaries: inputs={} outputs={} cross_queue_inputs={} cross_queue_outputs={} syncs={}\n"
			"  partial_batches: prefix_passes={} suffix_passes={} interleaved_passes={} cross_queue_boundary_passes={} cross_queue_transitions={}\n"
			"  coverage: passes={}/{} requirements={} batches={}/{} transitions={}\n"
			"  largest: passes={} requirements={}\n"
			"  estimated_saved: AddTransition={} IsNewBatchNeeded={}",
			static_cast<unsigned int>(frameIndex),
			RenderGraphRegionModeToString(regionMode),
			TransitionPlacementModeToString(transitionPlacementMode),
			m_lastRegionStats.candidateRegionCount,
			m_lastRegionStats.acceptedRegionCount,
			m_lastRegionStats.rejectedRegionCount,
			rejectSummary.str(),
			m_frameDeclarationRefreshRequestedCount,
			m_frameDeclarationRefreshEquivalentCount,
			m_frameDeclarationRefreshRequestedCount - m_frameDeclarationRefreshEquivalentCount,
			m_lastRegionStats.boundaryInputEdgeCount,
			m_lastRegionStats.boundaryOutputEdgeCount,
			m_lastRegionStats.crossQueueBoundaryInputEdgeCount,
			m_lastRegionStats.crossQueueBoundaryOutputEdgeCount,
			m_lastRegionStats.boundarySyncCount,
			m_lastRegionStats.sameBatchPrefixPassCount,
			m_lastRegionStats.sameBatchSuffixPassCount,
			m_lastRegionStats.sameBatchInterleavedPassCount,
			m_lastRegionStats.crossQueueBoundaryPassCount,
			m_lastRegionStats.crossQueueTransitionCount,
			m_lastRegionStats.coveredPassCount,
			m_framePasses.size(),
			m_lastRegionStats.coveredRequirementCount,
			m_lastRegionStats.coveredBatchCount,
			batches.size(),
			m_lastRegionStats.coveredTransitionCount,
			m_lastRegionStats.largestRegionPassCount,
			m_lastRegionStats.largestRegionRequirementCount,
			m_lastRegionStats.estimatedSavedAddTransitionCalls,
			m_lastRegionStats.estimatedSavedIsNewBatchNeededCalls);

		spdlog::info(
			"RG compile validation R1 segment_metadata frame={} status=observed\n"
			"  segments={} tier1_eligible={} fingerprint=0x{:016x}\n"
			"  metadata: boundary_edges={} boundary_syncs={} queue_usage={}\n"
			"  contracts: inputs={} outputs={}\n"
			"  templates: batches={} queued_passes={} transitions={} waits={} signals={}",
			static_cast<unsigned int>(frameIndex),
			replaySegmentsForDiagnostics.size(),
			tier1EligibleCount,
			segmentFingerprint,
			segmentBoundaryEdgeCount,
			segmentBoundarySyncCount,
			segmentQueueUsageCount,
			segmentInputCount,
			segmentOutputCount,
			segmentBatchTemplateCount,
			segmentQueuedPassTemplateCount,
			segmentTransitionTemplateCount,
			segmentWaitTemplateCount,
			segmentSignalTemplateCount);

		if (static_cast<uint8_t>(regionMode) >= static_cast<uint8_t>(rg::runtime::RenderGraphRegionMode::ValidateOnly)) {
			spdlog::info(
				"RG compile validation R2 segment_validation frame={} status=observed\n"
				"  previous_segments={} current_segments={} hits={} misses={} allowed_template_state_divergences={} invalidations=[{}]\n"
				"  transition_shape_diffs: resource={} range={} after_state={} queue={} phase={} discard={}\n"
				"  first_miss=\"{}\"\n"
				"  first_transition_shape_diff=\"{}\"",
				static_cast<unsigned int>(frameIndex),
				segmentValidation.previousSegmentCount,
				segmentValidation.currentSegmentCount,
				segmentValidation.hits,
				segmentValidation.misses,
				segmentValidation.templateStateDivergencesAllowed,
				invalidationSummary.str(),
				segmentValidation.transitionShapeResourceDiffs,
				segmentValidation.transitionShapeRangeDiffs,
				segmentValidation.transitionShapeAfterStateDiffs,
				segmentValidation.transitionShapeQueueDiffs,
				segmentValidation.transitionShapePhaseDiffs,
				segmentValidation.transitionShapeDiscardDiffs,
				segmentValidation.firstMissDetail,
				segmentValidation.firstTransitionShapeDiffDetail);
		}

		spdlog::info(
			"RG compile validation R3 segment_contracts frame={} status={}\n"
			"  checked_passes={} checked_edges={} checked_requirements={} checked_syncs={} failures={}\n"
			"  replay_metadata_status={} replay_metadata_failures={} first_failure=\"{}\"",
			static_cast<unsigned int>(frameIndex),
			semanticVerification.valid ? "ok" : "failed",
			semanticVerification.checkedPasses,
			semanticVerification.checkedEdges,
			semanticVerification.checkedRequirements,
			semanticVerification.checkedQueueSyncs,
			semanticVerification.failures,
			replayMetadataVerification.valid ? "ok" : "failed",
			replayMetadataVerification.failures,
			!semanticVerification.firstFailure.empty()
				? semanticVerification.firstFailure
				: replayMetadataVerification.firstFailure);

		spdlog::info(
			"RG compile validation R4 segment_templates frame={} status=observed\n"
			"  symbolic_batches={} partial_segment_batches={} queued_passes={} transitions={} waits={} signals={}\n"
			"  concrete_fences_cached=0 global_batch_indices_cached=0 raw_pass_pointers_cached=0",
			static_cast<unsigned int>(frameIndex),
			segmentBatchTemplateCount,
			m_lastRegionStats.sameBatchPrefixPassCount + m_lastRegionStats.sameBatchSuffixPassCount + m_lastRegionStats.crossQueueBoundaryPassCount,
			segmentQueuedPassTemplateCount,
			segmentTransitionTemplateCount,
			segmentWaitTemplateCount,
			segmentSignalTemplateCount);

		if (static_cast<uint8_t>(regionMode) >= static_cast<uint8_t>(rg::runtime::RenderGraphRegionMode::ShadowReplay)) {
			spdlog::info(
				"RG compile validation R6 shadow_replay frame={} status={}\n"
				"  matched_cached_segments={} current_segments={} replayed_passes={} dynamic_gap_passes={}\n"
				"  checked_contract_entries={} checked_syncs={} inserted_input_transitions={} allowed_extra_input_transition_segments={}\n"
				"  failures={} first_failure=\"{}\"\n"
				"  authoritative_execution=unchanged",
				static_cast<unsigned int>(frameIndex),
				shadowReplayVerification.valid ? "ok" : "failed",
				shadowReplayVerification.matchedSegments,
				replaySegmentsForDiagnostics.size(),
				shadowReplayVerification.replayedPasses,
				shadowReplayVerification.dynamicGapPasses,
				shadowReplayVerification.checkedRequirements,
				shadowReplayVerification.checkedQueueSyncs,
				shadowReplayVerification.insertedInputTransitions,
				shadowReplayVerification.extraInputTransitionsAllowed,
				shadowReplayVerification.failures,
				shadowReplayVerification.firstFailure);

			const bool replayAuthoritativeRequested =
				static_cast<uint8_t>(regionMode) >= static_cast<uint8_t>(rg::runtime::RenderGraphRegionMode::ReplayAuthoritative);
			spdlog::info(
				"RG compile validation R8 authoritative_replay_readiness frame={} status={}\n"
				"  requested={} attempted={} committed={} execution_path={} matched_cached_segments={} replayable_passes={} dynamic_gap_passes={}\n"
				"  inserted_input_transitions={} blockers={} blocker_summary=[{}] first_failure=\"{}\"",
				static_cast<unsigned int>(frameIndex),
				authoritativeReplayReadiness.ready ? "ready" : "blocked",
				replayAuthoritativeRequested ? 1 : 0,
				m_lastAuthoritativeReplayAttempted ? 1 : 0,
				m_lastAuthoritativeReplaySucceeded ? 1 : 0,
				authoritativeReplayReadiness.ready && replayAuthoritativeRequested ? "replay_authoritative" : "full_compile",
				m_lastAuthoritativeReplayAttempted ? m_lastAuthoritativeReplaySegments : authoritativeReplayReadiness.matchedSegments,
				m_lastAuthoritativeReplayAttempted ? m_lastAuthoritativeReplayPasses : authoritativeReplayReadiness.replayablePasses,
				m_lastAuthoritativeReplayAttempted ? m_lastAuthoritativeReplayDynamicGapPasses : authoritativeReplayReadiness.dynamicGapPasses,
				authoritativeReplayReadiness.insertedInputTransitions,
				authoritativeReplayReadiness.blockers,
				authoritativeReplayReadiness.blockerSummary,
				m_lastAuthoritativeReplayFailure);
		}

		if (!m_lastRegionCandidateDiagnostics.empty()) {
			std::ostringstream candidateSummary;
			for (const auto& line : m_lastRegionCandidateDiagnostics) {
				candidateSummary << "\n  " << line;
			}
			spdlog::info(
				"RG compile validation M1 candidates frame={} shown={}/{}{}",
				static_cast<unsigned int>(frameIndex),
				m_lastRegionCandidateDiagnostics.size(),
				m_lastRegionStats.candidateRegionCount,
				candidateSummary.str());
		}
	}
}

