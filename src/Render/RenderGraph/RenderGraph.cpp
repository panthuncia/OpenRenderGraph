#include "Render/RenderGraph/RenderGraph.h"

#include <span>
#include <algorithm>
#include <cmath>
#include <map>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <tracy/Tracy.hpp>
#include <rhi_helpers.h>
#include <rhi_debug.h>
#include <random>

#include "Render/PassExecutionContext.h"
#include "Utilities/ORGUtilities.h"
#include "Managers/Singletons/DeviceManager.h"
#include "Managers/Singletons/DeletionManager.h"
#include "Managers/Singletons/UploadManager.h"
#include "Managers/Singletons/StatisticsManager.h"
#include "Render/PassBuilders.h"
#include "Resources/ResourceGroup.h"
#include "Managers/CommandRecordingManager.h"
#include "Interfaces/IHasMemoryMetadata.h"
#include "Interfaces/IDynamicDeclaredResources.h"
#include "Resources/DynamicResource.h"
#include "Resources/ExternalTextureResource.h"
#include "Resources/PixelBuffer.h"
#include "Resources/MemoryStatisticsComponents.h"

namespace {
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

	rhi::DescriptorSlot ResolveRTVSlot(Resource* resource, uint32_t mip, uint32_t slice) noexcept {
		if (!resource) {
			spdlog::error("RG RTV resolve: resource pointer is null");
			return {};
		}

		Resource* originalResource = resource;
		resource = UnwrapDynamicResource(resource);
		if (!resource) {
			spdlog::error(
				"RG RTV resolve: dynamic resource '{}' id={} unwrapped to null backing",
				originalResource->GetName(),
				originalResource->GetGlobalResourceID());
			return {};
		}

		if (auto* gir = dynamic_cast<GloballyIndexedResource*>(resource)) {
			if (!gir->HasRTV()) {
				spdlog::error(
					"RG RTV resolve: resource '{}' id={} has no RTV descriptors",
					resource->GetName(),
					resource->GetGlobalResourceID());
				return {};
			}
			return gir->GetRTVInfo(mip, slice).slot;
		}

		if (auto* externalTexture = dynamic_cast<ExternalTextureResource*>(resource)) {
			if (!externalTexture->HasHandle() || !externalTexture->HasRTVSlot()) {
				const auto handle = externalTexture->GetHandle();
				const auto rtvSlot = externalTexture->GetRTVSlot();
				spdlog::error(
					"RG RTV resolve: external texture '{}' id={} invalid backbuffer binding. handle=({}, {}) rtv=({}, {})",
					resource->GetName(),
					resource->GetGlobalResourceID(),
					handle.index,
					handle.generation,
					rtvSlot.heap.index,
					rtvSlot.index);
			}
			return externalTexture->GetRTVSlot();
		}

		spdlog::error(
			"RG RTV resolve: resource '{}' id={} type does not expose RTV descriptors",
			resource->GetName(),
			resource->GetGlobalResourceID());

		return {};
	}

	constexpr size_t QueueIndex(QueueKind queue) noexcept {
		return static_cast<size_t>(queue);
	}

	bool StatesExactlyEqual(const ResourceState& lhs, const ResourceState& rhs);
	bool IsWholeResourceRange(const RangeSpec& range, ResourceRegistry::RegistryHandle resource);
	bool TryGetWholeResourceTrackerState(const SymbolicTracker& tracker, ResourceState& outState);

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
	void UpdateRetainedDeclarationCache(const ResourceRegistry& registry, PassAndResources& passAndResources) {
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

	QueueKind ResolveExternalPreferredQueueKind(const RenderGraph::ExternalPassDesc& desc) {
		const QueueKind preferredQueueKind = desc.preferredQueueKind.value_or(DefaultPreferredQueueKind(desc.type));
		if (!IsPreferredQueueKindCompatible(desc.type, preferredQueueKind)) {
			throw std::runtime_error("External pass '" + desc.name + "' requested an incompatible queue kind");
		}
		return preferredQueueKind;
	}

	QueueAssignmentPolicy ResolveExternalQueueAssignmentPolicy(const RenderGraph::ExternalPassDesc& desc) {
		if (desc.queueAssignmentPolicy.has_value()) {
			return *desc.queueAssignmentPolicy;
		}

		if (desc.preferredQueueKind.has_value()) {
			return QueueAssignmentPolicy::ForcePreferred;
		}

		return DefaultQueueAssignmentPolicy(desc.type);
	}

	// Insert into a sorted vector, maintaining sorted order. No-op if already present.
	inline void SortedInsert(std::vector<uint64_t>& v, uint64_t val) {
		auto it = std::lower_bound(v.begin(), v.end(), val);
		if (it == v.end() || *it != val) v.insert(it, val);
	}

	bool QueueSupportsSyncState(QueueKind queue, rhi::ResourceSyncState state) { // TODO: Is this actually meaningful?
		switch (queue) {
		case QueueKind::Graphics:
			return true;
		case QueueKind::Compute:
			return !ResourceSyncStateIsNotComputeSyncState(state);
		case QueueKind::Copy:
			return ResourceSyncStateHasOnly(
				state,
				rhi::ResourceSyncState::None
				| rhi::ResourceSyncState::All
				| rhi::ResourceSyncState::Copy
				| rhi::ResourceSyncState::Resolve
				| rhi::ResourceSyncState::SyncSplit);
		default:
			return false;
		}
	}

	bool QueueSupportsAccessType(QueueKind queue, rhi::ResourceAccessType access) {
		if (queue == QueueKind::Graphics) {
			return true;
		}

		if (queue == QueueKind::Compute) {
			const auto unsupported = rhi::ResourceAccessType::RenderTarget |
				rhi::ResourceAccessType::RenderTargetClear |
				rhi::ResourceAccessType::DepthRead |
				rhi::ResourceAccessType::DepthReadWrite |
				rhi::ResourceAccessType::DepthStencilClear;
			return (access & unsupported) == 0;
		}

		if (queue == QueueKind::Copy) {
			const auto supported = rhi::ResourceAccessType::None |
				rhi::ResourceAccessType::Common |
				rhi::ResourceAccessType::CopySource |
				rhi::ResourceAccessType::CopyDest;
			return (access & ~supported) == 0;
		}

		return false;
	}

	ResourceState NormalizeStateForQueue(QueueKind queue, ResourceState state) {
		if (queue == QueueKind::Copy) {
			const auto copyAccess = state.access & (rhi::ResourceAccessType::CopySource | rhi::ResourceAccessType::CopyDest);
			if (copyAccess != rhi::ResourceAccessType(0)) {
				// D3D12 enhanced barriers require copy-queue texture usage to stay in COMMON layout.
				// Preserve copy-class access and sync, but normalize layout away from COPY_SOURCE/COPY_DEST.
				state.layout = rhi::ResourceLayout::Common;
				state.sync = rhi::ResourceSyncState::Copy;
			}
		}

		return state;
	}

	bool QueueSupportsLayout(QueueKind queue, rhi::ResourceLayout layout) {
		if (queue == QueueKind::Graphics) {
			return true;
		}

		if (queue == QueueKind::Compute) {
			return layout != rhi::ResourceLayout::RenderTarget &&
				layout != rhi::ResourceLayout::RenderTargetClear &&
				layout != rhi::ResourceLayout::DepthReadWrite &&
				layout != rhi::ResourceLayout::DepthStencilClear &&
				layout != rhi::ResourceLayout::DepthRead;
		}

		if (queue == QueueKind::Copy) {
			return layout == rhi::ResourceLayout::Common;
		}

		return false;
	}

	bool QueueSupportsTransition(QueueKind queue, const ResourceTransition& transition) {
		if (!QueueSupportsSyncState(queue, transition.prevSyncState)) {
			return false;
		}
		if (!QueueSupportsSyncState(queue, transition.newSyncState)) {
			return false;
		}
		if (!QueueSupportsAccessType(queue, transition.prevAccessType)) {
			return false;
		}
		if (!QueueSupportsAccessType(queue, transition.newAccessType)) {
			return false;
		}
		if (transition.pResource->HasLayout()) {
			if (!QueueSupportsLayout(queue, transition.prevLayout)) {
				return false;
			}
			if (!QueueSupportsLayout(queue, transition.newLayout)) {
				return false;
			}
		}
		return true;
	}

	const char* QueueKindToString(QueueKind queue) noexcept {
		switch (queue) {
		case QueueKind::Graphics: return "Graphics";
		case QueueKind::Compute: return "Compute";
		case QueueKind::Copy: return "Copy";
		default: return "Unknown";
		}
	}

	const char* PassTypeToString(RenderGraph::PassType type) noexcept {
		switch (type) {
		case RenderGraph::PassType::Render: return "Render";
		case RenderGraph::PassType::Compute: return "Compute";
		case RenderGraph::PassType::Copy: return "Copy";
		default: return "Unknown";
		}
	}

	const char* AutoAliasModeToString(AutoAliasMode mode) noexcept {
		switch (mode) {
		case AutoAliasMode::Off: return "Off";
		case AutoAliasMode::Conservative: return "Conservative";
		case AutoAliasMode::Balanced: return "Balanced";
		case AutoAliasMode::Aggressive: return "Aggressive";
		default: return "Unknown";
		}
	}

	const char* BatchWaitPhaseToString(RenderGraph::BatchWaitPhase phase) noexcept {
		switch (phase) {
		case RenderGraph::BatchWaitPhase::BeforeTransitions: return "BeforeTransitions";
		case RenderGraph::BatchWaitPhase::BeforeExecution: return "BeforeExecution";
		case RenderGraph::BatchWaitPhase::BeforeAfterPasses: return "BeforeAfterPasses";
		default: return "Unknown";
		}
	}

	const char* BatchSignalPhaseToString(RenderGraph::BatchSignalPhase phase) noexcept {
		switch (phase) {
		case RenderGraph::BatchSignalPhase::AfterTransitions: return "AfterTransitions";
		case RenderGraph::BatchSignalPhase::AfterExecution: return "AfterExecution";
		case RenderGraph::BatchSignalPhase::AfterCompletion: return "AfterCompletion";
		default: return "Unknown";
		}
	}

	const char* BatchTransitionPhaseToString(RenderGraph::BatchTransitionPhase phase) noexcept {
		switch (phase) {
		case RenderGraph::BatchTransitionPhase::BeforePasses: return "BeforePasses";
		case RenderGraph::BatchTransitionPhase::AfterPasses: return "AfterPasses";
		default: return "Unknown";
		}
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

	std::string PassRunMaskToString(PassRunMask mask) {
		switch (mask) {
		case PassRunMask::None: return "None";
		case PassRunMask::Immediate: return "Immediate";
		case PassRunMask::Retained: return "Retained";
		case PassRunMask::Both: return "Both";
		default: return std::to_string(static_cast<unsigned int>(to_u8(mask)));
		}
	}

	std::string FormatRangeSpec(const RangeSpec& range) {
		std::ostringstream oss;
		oss << "mip=[" << range.mipLower.ToString() << ".." << range.mipUpper.ToString()
			<< "] slice=[" << range.sliceLower.ToString() << ".." << range.sliceUpper.ToString() << "]";
		return oss.str();
	}
}


RenderGraph::AnyPassAndResources RenderGraph::MaterializeExternalPass(
	const ExternalPassDesc& d,
	bool callSetup,
	bool materializeReferencedResources)
{
	const bool traceLifecycle = m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled();
	const bool traceStructuralMaterialize = materializeReferencedResources && !callSetup;
	const bool logStructuralMaterialize = traceStructuralMaterialize && traceLifecycle;
	AnyPassAndResources any;
	any.type = d.type;
	any.name = d.name;

	if (logStructuralMaterialize) {
		spdlog::info(
			"RG structural materialize begin pass='{}' type={} preferredQueue={} pinnedQueue={} registerName={}",
			d.name,
			PassTypeToString(d.type),
			QueueKindToString(ResolveExternalPreferredQueueKind(d)),
			d.pinnedQueueSlot.has_value() ? std::to_string(static_cast<unsigned int>(static_cast<uint8_t>(*d.pinnedQueueSlot))) : std::string("none"),
			d.registerName);
	}

	if (d.type == PassType::Render) {
		auto rp = std::get<std::shared_ptr<RenderPass>>(d.pass);
		RenderPassAndResources par;
		par.pass = std::move(rp);
		par.name = d.name;
		par.techniquePath = d.techniquePath;
		par.collectStatistics = d.collectStatistics;
		{
			RenderPassBuilder b(this, d.name);
			b.pass = par.pass;
			b.built_ = true;
			b.params = {};
			b.params.isGeometryPass = d.isGeometryPass;
			b._declaredIds.clear();
			if (traceLifecycle) {
				spdlog::info("RG materialize external render pass '{}' declare begin", d.name);
			}
			EnsureProviderRegistered(par.pass.get());
			par.pass->DeclareResourceUsages(&b);
			if (logStructuralMaterialize) {
				spdlog::info(
					"RG structural materialize render pass='{}' declare complete requirements={} transitions={} identifiers={}",
					d.name,
					b.GatherResourceRequirements().size(),
					b.params.internalTransitions.size(),
					b.DeclaredResourceIds().size());
			}
			par.resources.staticResourceRequirements = b.GatherResourceRequirements();
			par.resources.internalTransitions = b.params.internalTransitions;
			par.resources.identifierSet = b.DeclaredResourceIds();
			par.resources.autoDescriptorShaderResources = b.params.autoDescriptorShaderResources;
			par.resources.autoDescriptorConstantBuffers = b.params.autoDescriptorConstantBuffers;
			par.resources.autoDescriptorUnorderedAccessViews = b.params.autoDescriptorUnorderedAccessViews;
			par.resources.activeFeatureDomains = b.params.activeFeatureDomains;
			par.resources.isGeometryPass = b.params.isGeometryPass;
			par.resources.preferredQueueKind = ResolveExternalPreferredQueueKind(d);
			par.resources.queueAssignmentPolicy = ResolveExternalQueueAssignmentPolicy(d);
			par.resources.pinnedQueueSlot = d.pinnedQueueSlot;
			if (materializeReferencedResources) {
				if (traceLifecycle) {
					spdlog::info("RG materialize external render pass '{}' materialize referenced resources begin", d.name);
				}
				if (logStructuralMaterialize) {
					spdlog::info("RG structural materialize render pass='{}' referenced resources begin", d.name);
				}
				MaterializeReferencedResources(par.resources.staticResourceRequirements, par.resources.internalTransitions, d.name);
				if (logStructuralMaterialize) {
					spdlog::info("RG structural materialize render pass='{}' referenced resources complete", d.name);
				}
			}
			par.retainedAnonymousKeepAlive = CaptureRetainedAnonymousKeepAlive(
				par.resources.staticResourceRequirements,
				par.resources.internalTransitions);
			UpdateRetainedDeclarationCache(_registry, par);
		}

		if (callSetup) {
			par.pass->SetResourceRegistryView(
				std::make_unique<ResourceRegistryView>(_registry, par.resources.identifierSet),
				par.resources.activeFeatureDomains,
				par.resources.autoDescriptorShaderResources,
				par.resources.autoDescriptorConstantBuffers,
				par.resources.autoDescriptorUnorderedAccessViews);
			if (traceLifecycle) {
				spdlog::info("RG materialize external render pass '{}' setup begin", d.name);
			}
			par.pass->Setup();
			if (traceLifecycle) {
				spdlog::info("RG materialize external render pass '{}' setup complete", d.name);
			}
		}

		any.pass = std::move(par);
	}
	else if (d.type == PassType::Compute) {
		auto cp = std::get<std::shared_ptr<ComputePass>>(d.pass);
		ComputePassAndResources par;
		par.pass = std::move(cp);
		par.name = d.name;
		par.techniquePath = d.techniquePath;
		par.collectStatistics = d.collectStatistics;
		{
			ComputePassBuilder b(this, d.name);
			b.pass = par.pass;
			b.built_ = true;
			b.params = {};
			b._declaredIds.clear();
			if (traceLifecycle) {
				spdlog::info("RG materialize external compute pass '{}' declare begin", d.name);
			}
			EnsureProviderRegistered(par.pass.get());
			par.pass->DeclareResourceUsages(&b);
			if (logStructuralMaterialize) {
				spdlog::info(
					"RG structural materialize compute pass='{}' declare complete requirements={} transitions={} identifiers={}",
					d.name,
					b.GatherResourceRequirements().size(),
					b.params.internalTransitions.size(),
					b.DeclaredResourceIds().size());
			}
			par.resources.staticResourceRequirements = b.GatherResourceRequirements();
			par.resources.internalTransitions = b.params.internalTransitions;
			par.resources.identifierSet = b.DeclaredResourceIds();
			par.resources.autoDescriptorShaderResources = b.params.autoDescriptorShaderResources;
			par.resources.autoDescriptorConstantBuffers = b.params.autoDescriptorConstantBuffers;
			par.resources.autoDescriptorUnorderedAccessViews = b.params.autoDescriptorUnorderedAccessViews;
			par.resources.activeFeatureDomains = b.params.activeFeatureDomains;
			par.resources.preferredQueueKind = ResolveExternalPreferredQueueKind(d);
			par.resources.queueAssignmentPolicy = ResolveExternalQueueAssignmentPolicy(d);
			par.resources.pinnedQueueSlot = d.pinnedQueueSlot;
			if (materializeReferencedResources) {
				if (traceLifecycle) {
					spdlog::info("RG materialize external compute pass '{}' materialize referenced resources begin", d.name);
				}
				if (logStructuralMaterialize) {
					spdlog::info("RG structural materialize compute pass='{}' referenced resources begin", d.name);
				}
				MaterializeReferencedResources(par.resources.staticResourceRequirements, par.resources.internalTransitions, d.name);
				if (logStructuralMaterialize) {
					spdlog::info("RG structural materialize compute pass='{}' referenced resources complete", d.name);
				}
			}
			par.retainedAnonymousKeepAlive = CaptureRetainedAnonymousKeepAlive(
				par.resources.staticResourceRequirements,
				par.resources.internalTransitions);
			UpdateRetainedDeclarationCache(_registry, par);
		}

		if (callSetup) {
			par.pass->SetResourceRegistryView(
				std::make_unique<ResourceRegistryView>(_registry, par.resources.identifierSet),
				par.resources.activeFeatureDomains,
				par.resources.autoDescriptorShaderResources,
				par.resources.autoDescriptorConstantBuffers,
				par.resources.autoDescriptorUnorderedAccessViews);
			if (traceLifecycle) {
				spdlog::info("RG materialize external compute pass '{}' setup begin", d.name);
			}
			par.pass->Setup();
			if (traceLifecycle) {
				spdlog::info("RG materialize external compute pass '{}' setup complete", d.name);
			}
		}

		any.pass = std::move(par);
	}
	else if (d.type == PassType::Copy) {
		auto cp = std::get<std::shared_ptr<CopyPass>>(d.pass);
		CopyPassAndResources par;
		par.pass = std::move(cp);
		par.name = d.name;
		par.techniquePath = d.techniquePath;
		par.collectStatistics = d.collectStatistics;
		{
			CopyPassBuilder b(this, d.name);
			b.pass = par.pass;
			b.built_ = true;
			b.params = {};
			b._declaredIds.clear();
			if (traceLifecycle) {
				spdlog::info("RG materialize external copy pass '{}' declare begin", d.name);
			}
			EnsureProviderRegistered(par.pass.get());
			par.pass->DeclareResourceUsages(&b);
			if (logStructuralMaterialize) {
				spdlog::info(
					"RG structural materialize copy pass='{}' declare complete requirements={} transitions={} identifiers={}",
					d.name,
					b.GatherResourceRequirements().size(),
					b.params.internalTransitions.size(),
					b.DeclaredResourceIds().size());
			}
			par.resources.staticResourceRequirements = b.GatherResourceRequirements();
			par.resources.internalTransitions = b.params.internalTransitions;
			par.resources.identifierSet = b.DeclaredResourceIds();
			par.resources.preferredQueueKind = ResolveExternalPreferredQueueKind(d);
			par.resources.queueAssignmentPolicy = ResolveExternalQueueAssignmentPolicy(d);
			par.resources.pinnedQueueSlot = d.pinnedQueueSlot;
			if (materializeReferencedResources) {
				if (traceLifecycle) {
					spdlog::info("RG materialize external copy pass '{}' materialize referenced resources begin", d.name);
				}
				if (logStructuralMaterialize) {
					spdlog::info("RG structural materialize copy pass='{}' referenced resources begin", d.name);
				}
				MaterializeReferencedResources(par.resources.staticResourceRequirements, par.resources.internalTransitions, d.name);
				if (logStructuralMaterialize) {
					spdlog::info("RG structural materialize copy pass='{}' referenced resources complete", d.name);
				}
			}
			par.retainedAnonymousKeepAlive = CaptureRetainedAnonymousKeepAlive(
				par.resources.staticResourceRequirements,
				par.resources.internalTransitions);
			UpdateRetainedDeclarationCache(_registry, par);
		}

		if (callSetup) {
			par.pass->SetResourceRegistryView(
				std::make_unique<ResourceRegistryView>(_registry, par.resources.identifierSet));
			if (traceLifecycle) {
				spdlog::info("RG materialize external copy pass '{}' setup begin", d.name);
			}
			par.pass->Setup();
			if (traceLifecycle) {
				spdlog::info("RG materialize external copy pass '{}' setup complete", d.name);
			}
		}

		any.pass = std::move(par);
	}

	if (logStructuralMaterialize) {
		spdlog::info("RG structural materialize complete pass='{}' type={}", d.name, PassTypeToString(d.type));
	}
	if (traceStructuralMaterialize) {
		if (m_structuralMaterializeCheckpointCallback) {
			m_structuralMaterializeCheckpointCallback(d.name);
		}
	}

	return any;
}

void RenderGraph::RegisterExternalPassName(const ExternalPassDesc& d, AnyPassAndResources& any)
{
	if (!d.registerName) {
		return;
	}

	if (d.type == PassType::Render) {
		auto& rp = std::get<RenderPassAndResources>(any.pass);
		if (!d.name.empty()) {
			renderPassesByName[d.name] = rp.pass;
		}
	}
	else if (d.type == PassType::Compute) {
		auto& cp = std::get<ComputePassAndResources>(any.pass);
		if (!d.name.empty()) {
			computePassesByName[d.name] = cp.pass;
		}
	}
}

void RenderGraph::WriteCompiledGraphDebugDump(uint8_t frameIndex, const std::vector<Node>& nodes) const
{
	try {
		auto resourceNameForHandle = [this](const ResourceRegistry::RegistryHandle& handle) -> std::string {
			if (auto* resource = _registry.Resolve(handle)) {
				return resource->GetName();
			}
			return {};
		};

		auto resourceLabelForHandle = [&](const ResourceRegistry::RegistryHandle& handle) -> std::string {
			std::ostringstream oss;
			oss << "id=" << handle.GetGlobalResourceID();
			const std::string resourceName = resourceNameForHandle(handle);
			if (!resourceName.empty()) {
				oss << " name=\"" << resourceName << "\"";
			}
			if (handle.IsEphemeral()) {
				oss << " handle=ephemeral";
			}
			return oss.str();
		};

		auto queueSlotLabel = [this](size_t queueSlot) -> std::string {
			std::ostringstream oss;
			oss << queueSlot << ":" << QueueKindToString(m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(queueSlot))));
			return oss.str();
		};

		auto nodeName = [this, &nodes](size_t nodeIndex) -> std::string {
			if (nodeIndex >= nodes.size()) {
				return "<invalid-node>";
			}
			const auto& node = nodes[nodeIndex];
			if (node.passIndex < m_framePasses.size() && !m_framePasses[node.passIndex].name.empty()) {
				return m_framePasses[node.passIndex].name;
			}
			return "PassIndex#" + std::to_string(node.passIndex);
		};

		auto passNameForIndex = [this](uint32_t passIndex) -> std::string {
			if (passIndex < m_framePasses.size() && !m_framePasses[passIndex].name.empty()) {
				return m_framePasses[passIndex].name;
			}
			return "PassIndex#" + std::to_string(passIndex);
		};

		auto resourceLabelForID = [&](uint64_t resourceID) -> std::string {
			std::ostringstream oss;
			oss << "id=" << resourceID;
			auto resourceIt = resourcesByID.find(resourceID);
			if (resourceIt != resourcesByID.end() && resourceIt->second && !resourceIt->second->GetName().empty()) {
				oss << " name=\"" << resourceIt->second->GetName() << "\"";
			}
			return oss.str();
		};

		std::ostringstream dump;

		auto appendState = [&](const char* prefix, const ResourceState& state) {
			dump << prefix
				 << "access=" << rhi::helpers::ResourceAccessMaskToString(state.access)
				 << " layout=" << rhi::helpers::ResourceLayoutToString(state.layout)
				 << " sync=" << rhi::helpers::ResourceSyncToString(state.sync);
		};

		auto appendPassEntry = [&](size_t passIndex, PassType passType, const auto& passEntry) {
			const auto& resources = passEntry.resources;
			const size_t assignedQueueSlot = passIndex < m_assignedQueueSlotsByFramePass.size()
				? m_assignedQueueSlotsByFramePass[passIndex]
				: (resources.pinnedQueueSlot.has_value()
					? static_cast<size_t>(static_cast<uint8_t>(*resources.pinnedQueueSlot))
					: QueueIndex(resources.preferredQueueKind));

			dump << "[" << passIndex << "] "
				 << PassTypeToString(passType)
				 << " name=\"" << passEntry.name << "\""
				 << " run=" << PassRunMaskToString(passEntry.run)
				 << " preferred_queue=" << QueueKindToString(resources.preferredQueueKind)
				 << " assigned_queue=" << queueSlotLabel(assignedQueueSlot);
			if (resources.pinnedQueueSlot.has_value()) {
				dump << " pinned_queue=" << static_cast<unsigned int>(static_cast<uint8_t>(*resources.pinnedQueueSlot));
			}
			if constexpr (requires { resources.isGeometryPass; }) {
				dump << " geometry_pass=" << (resources.isGeometryPass ? "true" : "false");
			}
			dump << " declared_requirements=" << GetFrameRequirementCount(resources)
				 << " internal_transitions=" << resources.internalTransitions.size()
				 << "\n";

			const auto frameRequirements = GetFrameRequirementsSpan(resources);
			if (!frameRequirements.empty()) {
				dump << "  requirements:\n";
				for (const auto& req : frameRequirements) {
					dump << "    - " << resourceLabelForHandle(req.resourceHandleAndRange.resource)
						 << " range=" << FormatRangeSpec(req.resourceHandleAndRange.range)
						 << " access=" << rhi::helpers::ResourceAccessMaskToString(req.state.access)
						 << " layout=" << rhi::helpers::ResourceLayoutToString(req.state.layout)
						 << " sync=" << rhi::helpers::ResourceSyncToString(req.state.sync)
						 << "\n";
				}
			}

			if (!resources.internalTransitions.empty()) {
				dump << "  internal_transitions:\n";
				for (const auto& internalTransition : resources.internalTransitions) {
					dump << "    - " << resourceLabelForHandle(internalTransition.first.resource)
						 << " range=" << FormatRangeSpec(internalTransition.first.range)
						 << " -> access=" << rhi::helpers::ResourceAccessMaskToString(internalTransition.second.access)
						 << " layout=" << rhi::helpers::ResourceLayoutToString(internalTransition.second.layout)
						 << " sync=" << rhi::helpers::ResourceSyncToString(internalTransition.second.sync)
						 << "\n";
				}
			}
		};

		dump << "RenderGraph Compiled State\n";
		dump << "frame_index=" << static_cast<unsigned int>(frameIndex) << "\n";
		dump << "pass_count=" << m_framePasses.size()
			 << " node_count=" << nodes.size()
			 << " batch_count=" << batches.size()
			 << " queue_slot_count=" << m_queueRegistry.SlotCount() << "\n";
		dump << "active_queue_slots=[";
		bool firstActive = true;
		for (size_t queueIndex = 0; queueIndex < m_activeQueueSlotsThisFrame.size(); ++queueIndex) {
			if (!m_activeQueueSlotsThisFrame[queueIndex]) {
				continue;
			}
			if (!firstActive) {
				dump << ", ";
			}
			firstActive = false;
			dump << queueSlotLabel(queueIndex);
		}
		dump << "]\n\n";

		const bool regionDiagnosticsEnabled = m_renderGraphSettingsService
			? m_renderGraphSettingsService->GetRenderGraphRegionDiagnosticsEnabled()
			: false;
		const bool relaxAliasPlacement = m_renderGraphSettingsService
			? m_renderGraphSettingsService->GetRenderGraphReplayRelaxAliasPlacement()
			: true;

		dump << "[ReplayDiagnostics]\n";
		dump << "attempted=" << (m_lastAuthoritativeReplayAttempted ? "true" : "false")
			 << " succeeded=" << (m_lastAuthoritativeReplaySucceeded ? "true" : "false")
			 << " segments=" << m_lastAuthoritativeReplaySegments
			 << " replayed_passes=" << m_lastAuthoritativeReplayPasses
			 << " dynamic_gap_passes=" << m_lastAuthoritativeReplayDynamicGapPasses
			 << " region_diagnostics_enabled=" << (regionDiagnosticsEnabled ? "true" : "false")
			 << " relax_alias_placement=" << (relaxAliasPlacement ? "true" : "false")
			 << " failure=\"" << m_lastAuthoritativeReplayFailure << "\""
			 << " recompute_reason=\"" << m_lastAuthoritativeReplayRecomputeReason << "\"\n";
		dump << "cached_replay_segments=" << m_regionCache.replaySegments.size() << "\n";
		for (size_t segmentIndex = 0; segmentIndex < m_regionCache.replaySegments.size(); ++segmentIndex) {
			const auto& segment = m_regionCache.replaySegments[segmentIndex];
			dump << "  segment[" << segmentIndex << "]"
				 << " tier1=" << (segment.tier1Eligible ? "true" : "false")
				 << " traces=" << segment.schedule.firstTraceIndex << "-" << segment.schedule.lastTraceIndex
				 << " passes=" << segment.schedule.passCount
				 << " batches=" << segment.schedule.batchCount
				 << " requirements=" << segment.schedule.requirementCount
				 << " first=\"" << passNameForIndex(segment.schedule.firstPassIndex) << "\""
				 << " last=\"" << passNameForIndex(segment.schedule.lastPassIndex) << "\""
				 << " identity={pass_sequence=0x" << std::hex << segment.identity.passSequenceHash
				 << " structural=0x" << segment.identity.structuralPositionHash << std::dec
				 << " pass_count=" << segment.identity.passCount << "}"
				 << " fingerprints={decl=0x" << std::hex << segment.fingerprint.declarationHash
				 << " access=0x" << segment.fingerprint.accessHash
				 << " queue=0x" << segment.fingerprint.queueHash
				 << " alias=0x" << segment.fingerprint.aliasHash
				 << " boundary=0x" << segment.fingerprint.boundaryHash
				 << " template=0x" << segment.fingerprint.templateShapeHash << std::dec << "}\n";
			dump << "    contract inputs=" << segment.contract.inputRequirements.size()
				 << " outputs=" << segment.contract.outputStates.size()
				 << " boundary_edges=" << segment.contract.boundaryEdges.size()
				 << " boundary_syncs=" << segment.contract.boundarySyncs.size() << "\n";
			for (size_t inputIndex = 0; inputIndex < segment.contract.inputRequirements.size(); ++inputIndex) {
				const auto& input = segment.contract.inputRequirements[inputIndex];
				dump << "      input[" << inputIndex << "] " << resourceLabelForID(input.resourceID)
					 << " queue=" << queueSlotLabel(input.queueSlot)
					 << " range=" << FormatRangeSpec(input.range)
					 << " whole=" << (input.wholeResource ? "true" : "false")
					 << " alias_activation=" << (input.aliasActivation ? "true" : "false")
					 << " transition_before=" << (input.transitionBeforeState ? "true" : "false")
					 << " transition_discard=" << (input.transitionDiscard ? "true" : "false")
					 << " weak_read=" << (input.readOnlyUniformWeakRequirement ? "true" : "false")
					 << " ";
				appendState("required_", input.requiredState);
				dump << "\n";
			}
			for (size_t batchTemplateIndex = 0; batchTemplateIndex < segment.batchTemplates.size(); ++batchTemplateIndex) {
				const auto& batchTemplate = segment.batchTemplates[batchTemplateIndex];
				dump << "      template_batch[" << batchTemplateIndex << "]"
					 << " local=" << batchTemplate.localBatchIndex
					 << " original=" << batchTemplate.originalBatchIndexAtExtraction
					 << " partial=" << (batchTemplate.partialBatch ? "true" : "false")
					 << " queued_passes=" << batchTemplate.queuedPasses.size()
					 << " transitions=" << batchTemplate.transitions.size()
					 << " waits=" << batchTemplate.waits.size()
					 << " signals=" << batchTemplate.signals.size() << "\n";
				if (!batchTemplate.queuedPasses.empty()) {
					dump << "        passes=[";
					for (size_t queuedIndex = 0; queuedIndex < batchTemplate.queuedPasses.size(); ++queuedIndex) {
						if (queuedIndex != 0) {
							dump << ", ";
						}
						const auto& queuedPass = batchTemplate.queuedPasses[queuedIndex];
						dump << passNameForIndex(queuedPass.originalFramePassIndexAtExtraction)
							 << "@" << queueSlotLabel(queuedPass.queueSlot);
					}
					dump << "]\n";
				}
				for (size_t transitionIndex = 0; transitionIndex < batchTemplate.transitions.size(); ++transitionIndex) {
					const auto& transition = batchTemplate.transitions[transitionIndex];
					dump << "        transition[" << transitionIndex << "] "
						 << resourceLabelForID(transition.resourceID)
						 << " backing_id=" << transition.backingResourceID
						 << " dynamic=" << (transition.dynamicResource ? "true" : "false")
						 << " queue=" << queueSlotLabel(transition.queueSlot)
						 << " phase=" << BatchTransitionPhaseToString(transition.phase)
						 << " discard=" << (transition.discard ? "true" : "false")
						 << " range=" << FormatRangeSpec(transition.range)
						 << " ";
					appendState("before_", transition.before);
					dump << " ";
					appendState("after_", transition.after);
					dump << "\n";
				}
				for (const auto& wait : batchTemplate.waits) {
					dump << "        wait phase=" << BatchWaitPhaseToString(wait.phase)
						 << " dst=" << queueSlotLabel(wait.dstQueue)
						 << " src=" << queueSlotLabel(wait.srcQueue) << "\n";
				}
				for (const auto& signal : batchTemplate.signals) {
					dump << "        signal phase=" << BatchSignalPhaseToString(signal.phase)
						 << " queue=" << queueSlotLabel(signal.queueSlot) << "\n";
				}
			}
		}
		dump << "\n";

		dump << "[CrossFrameQueueWaits]\n";
		bool wroteFrameStartWait = false;
		for (size_t dstIndex = 0; dstIndex < m_hasPendingFrameStartQueueWait.size(); ++dstIndex) {
			for (size_t srcIndex = 0; srcIndex < m_hasPendingFrameStartQueueWait[dstIndex].size(); ++srcIndex) {
				if (dstIndex == srcIndex || !m_hasPendingFrameStartQueueWait[dstIndex][srcIndex]) {
					continue;
				}
				wroteFrameStartWait = true;
				dump << "  wait dst=" << queueSlotLabel(dstIndex)
					 << " src=" << queueSlotLabel(srcIndex)
					 << " fence=" << m_pendingFrameStartQueueWaitFenceValue[dstIndex][srcIndex]
					 << "\n";
			}
		}
		if (!wroteFrameStartWait) {
			dump << "  none\n";
		}
		dump << "  last_producers=" << m_lastProducerByResourceAcrossFrames.size()
			 << " alias_pools=" << m_lastAliasPlacementProducersByPoolAcrossFrames.size()
			 << "\n";
		for (size_t queueIndex = 0; queueIndex < m_compiledLastProducerBatchByResourceByQueue.size(); ++queueIndex) {
			const auto& producers = m_compiledLastProducerBatchByResourceByQueue[queueIndex];
			if (producers.empty()) {
				continue;
			}
			dump << "  current_frame_producers queue=" << queueSlotLabel(queueIndex)
				 << " resources=" << producers.size()
				 << "\n";
		}
		dump << "\n";

		dump << "[FramePasses]\n";
		for (size_t passIndex = 0; passIndex < m_framePasses.size(); ++passIndex) {
			const auto& any = m_framePasses[passIndex];
			switch (any.type) {
			case PassType::Render:
				appendPassEntry(passIndex, any.type, std::get<RenderPassAndResources>(any.pass));
				break;
			case PassType::Compute:
				appendPassEntry(passIndex, any.type, std::get<ComputePassAndResources>(any.pass));
				break;
			case PassType::Copy:
				appendPassEntry(passIndex, any.type, std::get<CopyPassAndResources>(any.pass));
				break;
			default:
				dump << "[" << passIndex << "] Unknown name=\"" << any.name << "\" <unmaterialized>\n";
				break;
			}
		}

		dump << "\n[DependencyNodes]\n";
		for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
			const auto& node = nodes[nodeIndex];
			dump << "[" << nodeIndex << "]"
				 << " pass_index=" << node.passIndex
				 << " name=\"" << nodeName(nodeIndex) << "\""
				 << " original_order=" << node.originalOrder
				 << " criticality=" << node.criticality
				 << " default_queue=" << queueSlotLabel(node.queueSlot)
				 << " assigned_queue=" << (node.assignedQueueSlot.has_value() ? queueSlotLabel(*node.assignedQueueSlot) : std::string("<unset>"))
				 << " indegree=" << node.indegree
				 << "\n";

			if (!node.compatibleQueueSlots.empty()) {
				dump << "  compatible_queues=[";
				for (size_t i = 0; i < node.compatibleQueueSlots.size(); ++i) {
					if (i != 0) {
						dump << ", ";
					}
					dump << queueSlotLabel(node.compatibleQueueSlots[i]);
				}
				dump << "]\n";
			}

			if (!node.in.empty()) {
				dump << "  in=[";
				for (size_t i = 0; i < node.in.size(); ++i) {
					if (i != 0) {
						dump << ", ";
					}
					dump << nodeName(node.in[i]);
				}
				dump << "]\n";
			}

			if (!node.out.empty()) {
				dump << "  out=[";
				for (size_t i = 0; i < node.out.size(); ++i) {
					if (i != 0) {
						dump << ", ";
					}
					dump << nodeName(node.out[i]);
				}
				dump << "]\n";
			}

			const auto* dagAccesses = node.passIndex < m_framePassAccessSummaries.size()
				? &m_framePassAccessSummaries[node.passIndex].dagAccesses
				: nullptr;
			if (dagAccesses && !dagAccesses->empty()) {
				dump << "  access_by_id=[";
				for (size_t i = 0; i < dagAccesses->size(); ++i) {
					if (i != 0) {
						dump << ", ";
					}
					const auto& access = (*dagAccesses)[i];
					const uint64_t resourceID = access.resourceIndex < m_frameDAGResourceIDsByIndex.size()
						? m_frameDAGResourceIDsByIndex[access.resourceIndex]
						: 0;
					dump << resourceID;
					auto resourceIt = resourcesByID.find(resourceID);
					if (resourceIt != resourcesByID.end() && resourceIt->second && !resourceIt->second->GetName().empty()) {
						dump << ":\"" << resourceIt->second->GetName() << "\"";
					}
					dump << ":" << (access.kind == AccessKind::Read ? "Read" : "Write");
				}
				dump << "]\n";
			}
		}

		dump << "\n[Batches]\n";
		for (size_t batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
			const auto& batch = batches[batchIndex];
			dump << "[" << batchIndex << "] all_resources=" << batch.allResources.size()
				 << " internally_transitioned_resources=" << batch.internallyTransitionedResources.size()
				 << "\n";

			for (size_t queueIndex = 0; queueIndex < batch.QueueCount(); ++queueIndex) {
				const bool hasPasses = batch.HasPasses(queueIndex);
				const bool hasBeforeTransitions = batch.HasTransitions(queueIndex, BatchTransitionPhase::BeforePasses);
				const bool hasAfterTransitions = batch.HasTransitions(queueIndex, BatchTransitionPhase::AfterPasses);
				bool hasWaits = false;
				bool hasSignals = false;
				for (size_t sourceQueueIndex = 0; sourceQueueIndex < batch.QueueCount(); ++sourceQueueIndex) {
					for (size_t waitPhaseIndex = 0; waitPhaseIndex < PassBatch::kWaitPhaseCount; ++waitPhaseIndex) {
						if (batch.HasQueueWait(static_cast<BatchWaitPhase>(waitPhaseIndex), queueIndex, sourceQueueIndex)) {
							hasWaits = true;
						}
					}
				}
				for (size_t signalPhaseIndex = 0; signalPhaseIndex < PassBatch::kSignalPhaseCount; ++signalPhaseIndex) {
					if (batch.HasQueueSignal(static_cast<BatchSignalPhase>(signalPhaseIndex), queueIndex)) {
						hasSignals = true;
					}
				}

				if (!hasPasses && !hasBeforeTransitions && !hasAfterTransitions && !hasWaits && !hasSignals) {
					continue;
				}

				dump << "  queue[" << queueIndex << "]=" << queueSlotLabel(queueIndex) << "\n";

				for (size_t waitPhaseIndex = 0; waitPhaseIndex < PassBatch::kWaitPhaseCount; ++waitPhaseIndex) {
					const auto waitPhase = static_cast<BatchWaitPhase>(waitPhaseIndex);
					for (size_t sourceQueueIndex = 0; sourceQueueIndex < batch.QueueCount(); ++sourceQueueIndex) {
						if (!batch.HasQueueWait(waitPhase, queueIndex, sourceQueueIndex)) {
							continue;
						}
						dump << "    wait phase=" << BatchWaitPhaseToString(waitPhase)
							 << " src=" << queueSlotLabel(sourceQueueIndex)
							 << " fence=" << batch.GetQueueWaitFenceValue(waitPhase, queueIndex, sourceQueueIndex)
							 << "\n";
					}
				}

				for (size_t transitionPhaseIndex = 0; transitionPhaseIndex < static_cast<size_t>(BatchTransitionPhase::Count); ++transitionPhaseIndex) {
					const auto transitionPhase = static_cast<BatchTransitionPhase>(transitionPhaseIndex);
					const auto& transitions = batch.Transitions(queueIndex, transitionPhase);
					if (transitions.empty()) {
						continue;
					}
					dump << "    transitions " << BatchTransitionPhaseToString(transitionPhase) << ":\n";
					for (const auto& transition : transitions) {
						dump << "      - id=" << (transition.pResource ? transition.pResource->GetGlobalResourceID() : 0ull);
						if (transition.pResource && !transition.pResource->GetName().empty()) {
							dump << " name=\"" << transition.pResource->GetName() << "\"";
						}
						dump << " range=" << FormatRangeSpec(transition.range)
							 << " discard=" << (transition.discard ? "true" : "false")
							 << " layout=" << rhi::helpers::ResourceLayoutToString(transition.prevLayout)
							 << "->" << rhi::helpers::ResourceLayoutToString(transition.newLayout)
							 << " access=" << rhi::helpers::ResourceAccessMaskToString(transition.prevAccessType)
							 << "->" << rhi::helpers::ResourceAccessMaskToString(transition.newAccessType)
							 << " sync=" << rhi::helpers::ResourceSyncToString(transition.prevSyncState)
							 << "->" << rhi::helpers::ResourceSyncToString(transition.newSyncState)
							 << "\n";
					}
				}

				if (hasPasses) {
					dump << "    passes:\n";
					for (const auto& queuedPass : batch.Passes(queueIndex)) {
						std::visit([&](const auto* passEntry) {
							using TQueued = std::decay_t<decltype(passEntry)>;
							const PassType queuedPassType =
								std::is_same_v<TQueued, RenderPassAndResources*> ? PassType::Render :
								(std::is_same_v<TQueued, ComputePassAndResources*> ? PassType::Compute : PassType::Copy);
							dump << "      - " << passEntry->name
								 << " (" << PassTypeToString(queuedPassType)
								 << ", run=" << PassRunMaskToString(passEntry->run) << ")\n";
						}, queuedPass);
					}
				}

				for (size_t signalPhaseIndex = 0; signalPhaseIndex < PassBatch::kSignalPhaseCount; ++signalPhaseIndex) {
					const auto signalPhase = static_cast<BatchSignalPhase>(signalPhaseIndex);
					if (!batch.HasQueueSignal(signalPhase, queueIndex)) {
						continue;
					}
					dump << "    signal phase=" << BatchSignalPhaseToString(signalPhase)
						 << " fence=" << batch.GetQueueSignalFenceValue(signalPhase, queueIndex)
						 << "\n";
				}
			}
		}

		if (!aliasPlacementRangesByID.empty()) {
			dump << "\n[AliasPlacementRanges]\n";

			// Group placements by pool for readability
			std::map<uint64_t, std::vector<std::pair<uint64_t, const rg::alias::AliasPlacementRange*>>> byPool;
			for (const auto& [resourceID, placement] : aliasPlacementRangesByID) {
				byPool[placement.poolID].emplace_back(resourceID, &placement);
			}

			for (auto& [poolID, entries] : byPool) {
				// Sort by startByte within each pool
				std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
					return a.second->startByte < b.second->startByte;
				});

				auto poolIt = persistentAliasPools.find(poolID);
				dump << "pool=" << poolID;
				if (poolIt != persistentAliasPools.end()) {
					dump << " capacity=" << poolIt->second.capacityBytes
						 << " generation=" << poolIt->second.generation;
				}
				dump << " resource_count=" << entries.size() << "\n";

				for (const auto& [resourceID, placement] : entries) {
					dump << "  id=" << resourceID;
					auto resourceIt = resourcesByID.find(resourceID);
					if (resourceIt != resourcesByID.end() && resourceIt->second && !resourceIt->second->GetName().empty()) {
						dump << " name=\"" << resourceIt->second->GetName() << "\"";
					}
					dump << " bytes=[" << placement->startByte << ", " << placement->endByte << ")"
						 << " size=" << (placement->endByte - placement->startByte)
						 << " firstUse=" << placement->firstUse
						 << " lastUse=" << placement->lastUse
						 << " firstUsePass=" << placement->firstUsePassIndex
						 << " lastUsePass=" << placement->lastUsePassIndex;
					if (placement->firstUsePassIndex < m_framePasses.size()) {
						dump << " firstPassName=\"" << m_framePasses[placement->firstUsePassIndex].name << "\"";
					}
					if (placement->lastUsePassIndex < m_framePasses.size()) {
						dump << " lastPassName=\"" << m_framePasses[placement->lastUsePassIndex].name << "\"";
					}
					dump << "\n";
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

		std::random_device rd;

		std::mt19937 gen(rd());

		std::uniform_int_distribution<> distr(1, 1);

		
		std::string nameStr = "rendergraph_compiled_state_" + std::to_string(distr(gen));

		const fs::path dumpPath = dumpDir / nameStr;
		std::ofstream outFile(dumpPath, std::ios::out | std::ios::trunc);
		if (!outFile.is_open()) {
			spdlog::warn("Failed to open render graph debug dump '{}'", dumpPath.string());
			return;
		}
		outFile << dump.str();
		outFile.close();

		static bool announcedDumpPath = false;
		if (!announcedDumpPath) {
			announcedDumpPath = true;
			spdlog::info("Render graph compiled-state dump will be written to '{}'", dumpPath.string());
		}
	}
	catch (const std::exception& ex) {
		spdlog::warn("Failed to write render graph compiled-state dump: {}", ex.what());
	}
}

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

		std::vector<rg::memory::ResourceMemoryRecord> memoryRecords;
		m_memorySnapshotProvider.BuildSnapshot(memoryRecords);

		std::unordered_map<std::string, DumpCategoryRow> categoriesByLabel;
		categoriesByLabel.reserve(memoryRecords.size() * 2 + 8);
		uint64_t totalBytes = 0;

		for (const auto& record : memoryRecords) {
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

		std::vector<const rg::memory::ResourceMemoryRecord*> resources;
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

RenderGraph::PassView RenderGraph::GetPassView(const AnyPassAndResources& pr) {
	PassView v{};
	if (pr.type == PassType::Compute) {
		const auto& p = std::get<ComputePassAndResources>(pr.pass);
		v.reqs = GetFrameRequirementsSpan(p.resources);
		v.internalTransitions = &p.resources.internalTransitions;
	}
	else if (pr.type == PassType::Render) {
		const auto& p = std::get<RenderPassAndResources>(pr.pass);
		v.reqs = GetFrameRequirementsSpan(p.resources);
		v.internalTransitions = &p.resources.internalTransitions;
	}
	else if (pr.type == PassType::Copy) {
		const auto& p = std::get<CopyPassAndResources>(pr.pass);
		v.reqs = GetFrameRequirementsSpan(p.resources);
		v.internalTransitions = &p.resources.internalTransitions;
	}
	return v;
}

void RenderGraph::RebuildFramePassAccessSummaries() {
	ZoneScopedN("RenderGraph::RebuildFramePassAccessSummaries");
	m_framePassAccessSummaries.clear();
	m_framePassAccessSummaries.resize(m_framePasses.size());

	std::vector<uint8_t> resourcesWrittenThisFrame(m_frameDAGResourceCount, 0);
	auto schedulingResourceIDForHandle = [&](const ResourceRegistry::RegistryHandle& handle) {
		Resource* resource = handle.IsEphemeral()
			? handle.GetEphemeralPtr()
			: _registry.Resolve(handle);
		if (auto* dynamicResource = dynamic_cast<DynamicResource*>(resource)) {
			return dynamicResource->GetDynamicWrapperGlobalResourceID();
		}
		return handle.GetGlobalResourceID();
	};

	for (size_t passIndex = 0; passIndex < m_framePasses.size(); ++passIndex) {
		const auto& pass = m_framePasses[passIndex];
		auto& summary = m_framePassAccessSummaries[passIndex];
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
			for (const auto& req : view.reqs) {
				const auto resource = req.resourceHandleAndRange.resource;
				const uint64_t resourceID = schedulingResourceIDForHandle(resource);
				const bool isWrite = AccessTypeIsWriteType(req.state.access);
				const bool isUAV = IsUAVState(req.state);

				summary.requirementSummaries.push_back(FramePassRequirementStaticSummary{
					.resource = resource,
					.resourceID = resourceID,
					.range = req.resourceHandleAndRange.range,
					.state = req.state,
					.isUAV = isUAV,
					.isWrite = isWrite,
				});

				if (!isWrite) {
					continue;
				}
				auto dagResourceIt = m_frameDAGResourceIndexByID.find(resourceID);
				if (dagResourceIt != m_frameDAGResourceIndexByID.end() && dagResourceIt->second < resourcesWrittenThisFrame.size()) {
					resourcesWrittenThisFrame[dagResourceIt->second] = 1;
				}
			}
		}

		if (view.internalTransitions) {
			for (const auto& transition : *view.internalTransitions) {
				const auto resource = transition.first.resource;
				const uint64_t resourceID = schedulingResourceIDForHandle(resource);

				summary.internalTransitionSummaries.push_back(FramePassInternalTransitionStaticSummary{
					.resource = resource,
					.resourceID = resourceID,
				});

				auto dagResourceIt = m_frameDAGResourceIndexByID.find(resourceID);
				if (dagResourceIt != m_frameDAGResourceIndexByID.end() && dagResourceIt->second < resourcesWrittenThisFrame.size()) {
					resourcesWrittenThisFrame[dagResourceIt->second] = 1;
				}
			}
		}
	}

	std::vector<uint32_t> touchedSeen(m_frameDAGResourceCount, 0);
	std::vector<uint32_t> uavSeen(m_frameDAGResourceCount, 0);
	std::vector<uint32_t> accessSeen(m_frameDAGResourceCount, 0);
	std::vector<uint32_t> accessEntryIndexByResource(m_frameDAGResourceCount, 0);
	uint32_t passGeneration = 0;
	auto advancePassGeneration = [&]() {
		++passGeneration;
		if (passGeneration == 0) {
			std::fill(touchedSeen.begin(), touchedSeen.end(), 0);
			std::fill(uavSeen.begin(), uavSeen.end(), 0);
			std::fill(accessSeen.begin(), accessSeen.end(), 0);
			passGeneration = 1;
		}
	};

	for (auto& summary : m_framePassAccessSummaries) {
		advancePassGeneration();
		auto mark = [&](uint64_t resourceID, AccessKind accessKind, bool isUav) {
			auto dagResourceIt = m_frameDAGResourceIndexByID.find(resourceID);
			if (dagResourceIt == m_frameDAGResourceIndexByID.end()) {
				return;
			}

			const uint32_t dagResourceIndex = static_cast<uint32_t>(dagResourceIt->second);
			if (touchedSeen[dagResourceIndex] != passGeneration) {
				touchedSeen[dagResourceIndex] = passGeneration;
				summary.touchedResourceIDs.push_back(resourceID);
			}
			if (isUav && uavSeen[dagResourceIndex] != passGeneration) {
				uavSeen[dagResourceIndex] = passGeneration;
				summary.uavResourceIDs.push_back(resourceID);
			}

			if (accessKind == AccessKind::Read
				&& (dagResourceIndex >= resourcesWrittenThisFrame.size() || !resourcesWrittenThisFrame[dagResourceIndex])) {
				return;
			}

			if (accessSeen[dagResourceIndex] != passGeneration) {
				accessSeen[dagResourceIndex] = passGeneration;
				accessEntryIndexByResource[dagResourceIndex] = static_cast<uint32_t>(summary.dagAccesses.size());
				summary.dagAccesses.push_back({ dagResourceIndex, accessKind });
			}
			else if (accessKind == AccessKind::Write) {
				summary.dagAccesses[accessEntryIndexByResource[dagResourceIndex]].kind = AccessKind::Write;
			}
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
	}
}

std::vector<RenderGraph::Node> RenderGraph::BuildNodes(RenderGraph& rg) {
	ZoneScopedN("RenderGraph::BuildNodes");

	std::vector<Node> nodes;
	nodes.resize(rg.m_framePassAccessSummaries.size());
	const size_t slotCount = rg.m_queueRegistry.SlotCount();
	constexpr size_t passTypeCount = static_cast<size_t>(PassType::Copy) + 1;
	struct QueueCompatibilityCache {
		std::array<std::vector<size_t>, static_cast<size_t>(QueueKind::Count)> autoAssignableByKind;
		std::array<std::vector<size_t>, passTypeCount> automaticByPassType;
		std::array<std::vector<size_t>, static_cast<size_t>(QueueKind::Count)> fallbackByPreferredKind;
	};
	QueueCompatibilityCache queueCache{};
	auto passTypeIndex = [](PassType type) {
		return static_cast<size_t>(type);
	};
	for (size_t slotIndex = 0; slotIndex < slotCount; ++slotIndex) {
		const auto queueSlotIndex = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slotIndex));
		const QueueKind kind = rg.m_queueRegistry.GetKind(queueSlotIndex);
		if (!rg.m_queueRegistry.IsAutoAssignable(queueSlotIndex)) {
			continue;
		}

		queueCache.autoAssignableByKind[QueueIndex(kind)].push_back(slotIndex);
		for (PassType type : { PassType::Render, PassType::Compute, PassType::Copy }) {
			if (IsPreferredQueueKindCompatible(type, kind)) {
				queueCache.automaticByPassType[passTypeIndex(type)].push_back(slotIndex);
			}
		}
	}
	for (size_t kindIndex = 0; kindIndex < static_cast<size_t>(QueueKind::Count); ++kindIndex) {
		auto& fallbackSlots = queueCache.fallbackByPreferredKind[kindIndex];
		fallbackSlots = queueCache.autoAssignableByKind[kindIndex];
		if (fallbackSlots.empty()) {
			fallbackSlots.push_back(kindIndex);
		}
	}

	auto resolveCompatibleQueueSlotsForPass = [&queueCache, &passTypeIndex](const FramePassStaticAccessSummary& passAccess) -> std::vector<size_t> {
		if (passAccess.pinnedQueueSlot) {
			return std::vector<size_t>{ static_cast<size_t>(static_cast<uint8_t>(*passAccess.pinnedQueueSlot)) };
		}

		if (passAccess.queueAssignmentPolicy == QueueAssignmentPolicy::Automatic) {
			auto slots = queueCache.automaticByPassType[passTypeIndex(passAccess.type)];
				if (!slots.empty()) {
					return slots;
				}
		}

		return queueCache.fallbackByPreferredKind[QueueIndex(passAccess.preferredQueueKind)];
	};

	for (size_t i = 0; i < rg.m_framePassAccessSummaries.size(); ++i) {
		const auto& passAccess = rg.m_framePassAccessSummaries[i];
		Node n{};
		n.passIndex = i;
		n.compatibleQueueSlots = resolveCompatibleQueueSlotsForPass(passAccess);
		for (size_t slot : n.compatibleQueueSlots) {
			if (slot >= rg.m_queueRegistry.SlotCount()) {
				continue;
			}
			const QueueKind kind = rg.m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot)));
			n.compatibleQueueKindMask |= static_cast<uint8_t>(1u << QueueIndex(kind));
		}
		n.preferredQueueKind = passAccess.preferredQueueKind;
		n.queueAssignmentPolicy = passAccess.queueAssignmentPolicy;
		n.queueSlot = QueueIndex(n.preferredQueueKind);
		for (size_t slot : n.compatibleQueueSlots) {
			if (slot < rg.m_queueRegistry.SlotCount()
				&& rg.m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot))) == n.preferredQueueKind) {
				n.queueSlot = slot;
				break;
			}
		}
		if (n.compatibleQueueSlots.empty()) {
			n.compatibleQueueSlots.push_back(n.queueSlot);
		}
		n.assignedQueueSlot = n.queueSlot;
		n.originalOrder = static_cast<uint32_t>(i);
		n.touchedIDs = passAccess.touchedResourceIDs;
		n.uavIDs = passAccess.uavResourceIDs;

		nodes[i] = std::move(n);
	}

	return nodes;
}

std::vector<uint8_t> RenderGraph::PlanActiveQueueSlots(
	RenderGraph& rg,
	const std::vector<AnyPassAndResources>& passes,
	const std::vector<Node>& nodes)
{
	ZoneScopedN("RenderGraph::PlanActiveQueueSlots");
	const size_t slotCount = rg.m_queueRegistry.SlotCount();
	std::vector<uint8_t> activeSlots(slotCount, 0);
	if (slotCount == 0) {
		return activeSlots;
	}

	auto passHasExplicitQueuePin = [](const AnyPassAndResources& pr) {
		return std::visit([](auto const& passEntry) -> bool {
			using T = std::decay_t<decltype(passEntry)>;
			if constexpr (std::is_same_v<T, std::monostate>) {
				return false;
			}
			else {
				return passEntry.resources.pinnedQueueSlot.has_value();
			}
		}, pr.pass);
	};

	std::vector<uint32_t> indeg(nodes.size());
	std::vector<uint32_t> level(nodes.size(), 0);
	std::vector<size_t> ready;
	ready.reserve(nodes.size());
	for (size_t i = 0; i < nodes.size(); ++i) {
		indeg[i] = nodes[i].indegree;
		if (indeg[i] == 0) {
			ready.push_back(i);
		}
	}
	for (size_t head = 0; head < ready.size(); ++head) {
		const size_t u = ready[head];
		for (size_t v : nodes[u].out) {
			level[v] = (std::max)(level[v], level[u] + 1);
			if (--indeg[v] == 0) {
				ready.push_back(v);
			}
		}
	}

	std::array<std::vector<size_t>, static_cast<size_t>(QueueKind::Count)> slotsByKind;
	std::array<std::vector<size_t>, static_cast<size_t>(QueueKind::Count)> autoAssignableSlotsByKind;
	for (size_t slotIndex = 0; slotIndex < slotCount; ++slotIndex) {
		const auto queueSlotIndex = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slotIndex));
		const QueueKind kind = rg.m_queueRegistry.GetKind(queueSlotIndex);
		slotsByKind[QueueIndex(kind)].push_back(slotIndex);
		if (rg.m_queueRegistry.IsAutoAssignable(queueSlotIndex)) {
			autoAssignableSlotsByKind[QueueIndex(kind)].push_back(slotIndex);
		}
	}

	const bool allowAsyncCompute = rg.m_getUseAsyncCompute ? rg.m_getUseAsyncCompute() : true;
	const bool enableQueueSchedulingLogging = rg.m_getQueueSchedulingEnableLogging ? rg.m_getQueueSchedulingEnableLogging() : false;
	const double widthScale = rg.m_getQueueSchedulingWidthScale ? static_cast<double>(rg.m_getQueueSchedulingWidthScale()) : 1.0;
	const double penaltyBias = rg.m_getQueueSchedulingPenaltyBias ? static_cast<double>(rg.m_getQueueSchedulingPenaltyBias()) : 0.0;
	const double minPenalty = rg.m_getQueueSchedulingMinPenalty ? static_cast<double>(rg.m_getQueueSchedulingMinPenalty()) : 1.0;
	const double resourcePressureWeight = rg.m_getQueueSchedulingResourcePressureWeight ? static_cast<double>(rg.m_getQueueSchedulingResourcePressureWeight()) : 1.0;
	const double uavPressureWeight = rg.m_getQueueSchedulingUavPressureWeight ? static_cast<double>(rg.m_getQueueSchedulingUavPressureWeight()) : 0.5;
	auto queueKindName = [](QueueKind kind) -> const char* {
		switch (kind) {
		case QueueKind::Graphics: return "graphics";
		case QueueKind::Compute:  return "compute";
		case QueueKind::Copy:     return "copy";
		default:                 return "unknown";
		}
	};

	for (size_t kindIndex = 0; kindIndex < static_cast<size_t>(QueueKind::Count); ++kindIndex) {
		const QueueKind kind = static_cast<QueueKind>(kindIndex);
		auto& slots = slotsByKind[kindIndex];
		auto& autoAssignableSlots = autoAssignableSlotsByKind[kindIndex];
		if (slots.empty()) {
			continue;
		}

		std::unordered_set<size_t> pinnedSlots;
		std::unordered_set<uint64_t> uniqueTouchedIDs;
		std::unordered_map<uint32_t, size_t> widthByLevel;
		size_t compatibleNodeCount = 0;
		size_t totalTouched = 0;
		size_t totalUAV = 0;

		for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
			const auto& node = nodes[nodeIndex];
			const bool compatibleWithKind = (node.compatibleQueueKindMask & static_cast<uint8_t>(1u << kindIndex)) != 0;
			if (!compatibleWithKind) {
				continue;
			}

			++compatibleNodeCount;
			++widthByLevel[level[nodeIndex]];
			totalTouched += node.touchedIDs.size();
			totalUAV += node.uavIDs.size();
			uniqueTouchedIDs.insert(node.touchedIDs.begin(), node.touchedIDs.end());

			if (node.passIndex < passes.size() && passHasExplicitQueuePin(passes[node.passIndex])) {
				const size_t pinnedSlot = node.compatibleQueueSlots.empty() ? node.queueSlot : node.compatibleQueueSlots.front();
				if (pinnedSlot < slotCount
					&& rg.m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(pinnedSlot))) == kind) {
					pinnedSlots.insert(pinnedSlot);
				}
			}
		}

		if (kind == QueueKind::Graphics) {
			activeSlots[slots.front()] = 1;
			for (size_t pinnedSlot : pinnedSlots) {
				activeSlots[pinnedSlot] = 1;
			}
			continue;
		}

		if (compatibleNodeCount == 0 && pinnedSlots.empty()) {
			continue;
		}

		size_t maxLevelWidth = 0;
		for (const auto& [_, width] : widthByLevel) {
			maxLevelWidth = (std::max)(maxLevelWidth, width);
		}

		double resourcePressure = 1.0;
		if (!uniqueTouchedIDs.empty()) {
			resourcePressure = static_cast<double>(totalTouched) / static_cast<double>(uniqueTouchedIDs.size());
		}
		const double uavPressure = totalTouched == 0 ? 0.0 : static_cast<double>(totalUAV) / static_cast<double>(totalTouched);
		const double weightedPressure = penaltyBias + resourcePressureWeight * resourcePressure + uavPressureWeight * uavPressure;
		const double parallelismPenalty = (std::max)(minPenalty, weightedPressure);
		const double widthToPenaltyRatio = parallelismPenalty > 0.0
			? ((widthScale * static_cast<double>(maxLevelWidth)) / parallelismPenalty)
			: 0.0;
		size_t targetCount = compatibleNodeCount > 0 ? 1u : 0u;
		if (maxLevelWidth > 0) {
			targetCount = static_cast<size_t>(std::ceil(widthToPenaltyRatio));
			targetCount = (std::max)(size_t(1), targetCount);
		}
		if (kind == QueueKind::Compute && !allowAsyncCompute) {
			targetCount = (std::min)(targetCount, size_t(1));
		}
		targetCount = (std::min)(targetCount, autoAssignableSlots.size());
		targetCount = (std::max)(targetCount, pinnedSlots.size());

		for (size_t pinnedSlot : pinnedSlots) {
			activeSlots[pinnedSlot] = 1;
		}
		if (compatibleNodeCount > 0) {
			if (!autoAssignableSlots.empty()) {
				activeSlots[autoAssignableSlots.front()] = 1;
			}
		}

		size_t activeCount = 0;
		for (size_t slot : autoAssignableSlots) {
			activeCount += activeSlots[slot] ? 1u : 0u;
		}

		for (size_t slot : autoAssignableSlots) {
			if (activeCount >= targetCount) {
				break;
			}
			if (activeSlots[slot]) {
				continue;
			}
			activeSlots[slot] = 1;
			++activeCount;
		}

	}

	return activeSlots;
}

bool RenderGraph::AddEdgeDedup(
	size_t from, size_t to,
	std::vector<Node>& nodes,
	std::unordered_set<uint64_t>& edgeSet)
{
	if (from == to) return false;
	uint64_t key = (uint64_t(from) << 32) | uint64_t(to);
	if (!edgeSet.insert(key).second) return false;

	nodes[from].out.push_back(to);
	nodes[to].in.push_back(from);
	nodes[to].indegree++;
	return true;
}

bool RenderGraph::BuildDependencyGraph(
	std::vector<Node>& nodes)
{
	return BuildDependencyGraph(nodes, {});
}

bool RenderGraph::BuildDependencyGraph(
	std::vector<Node>& nodes,
	std::span<const std::pair<size_t, size_t>> explicitEdges)
{
	ZoneScopedN("RenderGraph::BuildDependencyGraph");
	std::vector<SeqState> seq(m_frameDAGResourceCount);

	std::unordered_set<uint64_t> edgeSet;
	edgeSet.reserve(nodes.size() * 8);

	// build deps in ORIGINAL order
	for (size_t i = 0; i < nodes.size(); ++i) {
		auto& node = nodes[i];
		const auto* dagAccesses = node.passIndex < m_framePassAccessSummaries.size()
			? &m_framePassAccessSummaries[node.passIndex].dagAccesses
			: nullptr;
		if (!dagAccesses) {
			continue;
		}

		for (const auto& access : *dagAccesses) {
			if (access.resourceIndex >= seq.size()) {
				continue;
			}

			auto& s = seq[access.resourceIndex];

			if (access.kind == AccessKind::Read) {
				if (s.lastWriter) AddEdgeDedup(*s.lastWriter, i, nodes, edgeSet);
				s.readsSinceWrite.push_back(i);
			}
			else { // Write
				if (s.lastWriter) AddEdgeDedup(*s.lastWriter, i, nodes, edgeSet);
				for (size_t r : s.readsSinceWrite)
					AddEdgeDedup(r, i, nodes, edgeSet);
				s.readsSinceWrite.clear();
				s.lastWriter = i;
			}
		}
	}

	// Apply explicit edges (e.g. "After(passName)")
	for (auto const& e : explicitEdges) {
		if (e.first >= nodes.size() || e.second >= nodes.size()) continue;
		AddEdgeDedup(e.first, e.second, nodes, edgeSet);
	}

	return FinalizeDependencyGraph(nodes);
}

bool RenderGraph::FinalizeDependencyGraph(std::vector<Node>& nodes)
{
	// topo + criticality (longest path)
	std::vector<uint32_t> indeg(nodes.size());
	for (size_t i = 0; i < nodes.size(); ++i) indeg[i] = nodes[i].indegree;

	auto originalOrderLess = [&](size_t lhs, size_t rhs) {
		if (nodes[lhs].originalOrder != nodes[rhs].originalOrder) {
			return nodes[lhs].originalOrder > nodes[rhs].originalOrder;
		}
		return lhs > rhs;
	};

	std::priority_queue<size_t, std::vector<size_t>, decltype(originalOrderLess)> ready(originalOrderLess);
	for (size_t i = 0; i < nodes.size(); ++i) {
		if (indeg[i] == 0) {
			ready.push(i);
		}
	}

	std::vector<size_t> topo;
	topo.reserve(nodes.size());

	while (!ready.empty()) {
		size_t u = ready.top();
		ready.pop();
		topo.push_back(u);
		for (size_t v : nodes[u].out) {
			if (--indeg[v] == 0) {
				ready.push(v);
			}
		}
	}

	if (topo.size() != nodes.size()) {
		// cycle: invalid graph
		return false;
	}

	for (size_t rank = 0; rank < topo.size(); ++rank) {
		nodes[topo[rank]].topoRank = rank;
	}

	// reverse topo DP
	for (auto& node : nodes) {
		node.criticality = 0;
	}
	for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
		size_t u = *it;
		uint32_t best = 0;
		for (size_t v : nodes[u].out)
			best = std::max(best, uint32_t(1 + nodes[v].criticality));
		nodes[u].criticality = best;
	}

	return true;
}

bool RenderGraph::AddCurrentFrameAliasSchedulingEdges(std::vector<Node>& nodes)
{
	ZoneScopedN("RenderGraph::AddCurrentFrameAliasSchedulingEdges");
	auto rangesOverlap = [](const rg::alias::AliasPlacementRange& lhs, const rg::alias::AliasPlacementRange& rhs) {
		const uint64_t overlapStart = (std::max)(lhs.startByte, rhs.startByte);
		const uint64_t overlapEnd = (std::min)(lhs.endByte, rhs.endByte);
		return overlapStart < overlapEnd;
	};

	auto resourceDebugName = [&](uint64_t resourceID) {
		auto it = resourcesByID.find(resourceID);
		if (it == resourcesByID.end() || !it->second || it->second->GetName().empty()) {
			return std::string("<unnamed>");
		}
		return it->second->GetName();
	};

	std::unordered_map<size_t, size_t> nodeIndexByPassIndex;
	nodeIndexByPassIndex.reserve(nodes.size());
	for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
		nodeIndexByPassIndex[nodes[nodeIndex].passIndex] = nodeIndex;
	}

	size_t existingEdgeCount = 0;
	for (const auto& node : nodes) {
		existingEdgeCount += node.out.size();
	}

	std::vector<uint64_t> resourceIDs;
	resourceIDs.reserve(aliasPlacementPoolByID.size());
	for (const auto& [resourceID, resourceIndex] : m_frameSchedulingResourceIndexByID) {
		if (TryGetAliasPlacementRangeByResourceIndex(resourceIndex)) {
			resourceIDs.push_back(resourceID);
		}
	}
	if (resourceIDs.empty()) {
		return true;
	}
	std::sort(resourceIDs.begin(), resourceIDs.end());

	std::unordered_set<uint64_t> edgeSet;
	edgeSet.reserve(existingEdgeCount + resourceIDs.size() * 4);
	for (size_t from = 0; from < nodes.size(); ++from) {
		for (size_t to : nodes[from].out) {
			edgeSet.insert((uint64_t(from) << 32) | uint64_t(to));
		}
	}

	for (size_t i = 0; i < resourceIDs.size(); ++i) {
		const uint64_t lhsResourceID = resourceIDs[i];
		const auto* lhs = TryGetAliasPlacementRange(lhsResourceID);
		if (!lhs) {
			continue;
		}
		if (lhs->firstUsePassIndex == std::numeric_limits<size_t>::max() ||
			lhs->lastUsePassIndex == std::numeric_limits<size_t>::max()) {
			continue;
		}

		for (size_t j = i + 1; j < resourceIDs.size(); ++j) {
			const uint64_t rhsResourceID = resourceIDs[j];
			const auto* rhs = TryGetAliasPlacementRange(rhsResourceID);
			if (!rhs) {
				continue;
			}
			if (lhs->poolID != rhs->poolID || !rangesOverlap(*lhs, *rhs)) {
				continue;
			}
			if (rhs->firstUsePassIndex == std::numeric_limits<size_t>::max() ||
				rhs->lastUsePassIndex == std::numeric_limits<size_t>::max()) {
				continue;
			}

			size_t fromPassIndex = std::numeric_limits<size_t>::max();
			size_t toPassIndex = std::numeric_limits<size_t>::max();
			if (lhs->lastUse < rhs->firstUse) {
				fromPassIndex = lhs->lastUsePassIndex;
				toPassIndex = rhs->firstUsePassIndex;
			}
			else if (rhs->lastUse < lhs->firstUse) {
				fromPassIndex = rhs->lastUsePassIndex;
				toPassIndex = lhs->firstUsePassIndex;
			}
			else {
				throw std::runtime_error(
					"Alias plan produced overlapping lifetimes for overlapping placements: resource " +
					std::to_string(lhsResourceID) + " ('" + resourceDebugName(lhsResourceID) + "') [" +
					std::to_string((*lhs).startByte) + ", " + std::to_string((*lhs).endByte) + ") firstUse=" +
					std::to_string(static_cast<uint64_t>((*lhs).firstUse)) + " lastUse=" +
					std::to_string(static_cast<uint64_t>((*lhs).lastUse)) + " and resource " +
					std::to_string(rhsResourceID) + " ('" + resourceDebugName(rhsResourceID) + "') [" +
					std::to_string((*rhs).startByte) + ", " + std::to_string((*rhs).endByte) + ") firstUse=" +
					std::to_string(static_cast<uint64_t>((*rhs).firstUse)) + " lastUse=" +
					std::to_string(static_cast<uint64_t>((*rhs).lastUse)));
			}

			if (fromPassIndex == toPassIndex) {
				continue;
			}

			auto fromNodeIt = nodeIndexByPassIndex.find(fromPassIndex);
			auto toNodeIt = nodeIndexByPassIndex.find(toPassIndex);
			if (fromNodeIt == nodeIndexByPassIndex.end() || toNodeIt == nodeIndexByPassIndex.end()) {
				continue;
			}

			AddEdgeDedup(fromNodeIt->second, toNodeIt->second, nodes, edgeSet);
		}
	}

	return FinalizeDependencyGraph(nodes);
}

void RenderGraph::CommitPassToBatch(
	RenderGraph& rg,
	AnyPassAndResources& pr,
	const Node& node,

	unsigned int currentBatchIndex,
	PassBatch& currentBatch,
	std::unordered_set<uint64_t>& scratchTransitioned,
	std::unordered_set<size_t>& scratchFallback,
	std::vector<ResourceTransition>& scratchTransitions)
{
	ZoneScopedN("RenderGraph::CommitPassToBatch");
	if (!pr.name.empty()) {
		ZoneText(pr.name.data(), pr.name.size());
	}
	const size_t passQueueSlot = node.assignedQueueSlot.value_or(node.queueSlot);
	const size_t queueCount = currentBatch.QueueCount();
	const size_t gfxSlot = QueueIndex(QueueKind::Graphics);
	const auto& passSummary = rg.m_framePassSchedulingSummaries[node.passIndex];
	scratchTransitioned.clear();
	auto& resourcesTransitionedThisPass = scratchTransitioned;

	scratchFallback.clear();
	auto& fallbackResourceIndices = scratchFallback;
	rg.ProcessResourceRequirements(
		passQueueSlot,
		passSummary.requirements,
		pr.name,
		currentBatchIndex,
		currentBatch,
		resourcesTransitionedThisPass,
		fallbackResourceIndices,
		scratchTransitions);

	// For fallback transitions delegated to the graphics queue in this batch's
	// BeforePasses, update graphics transition tracking and wait on prior producers.
	auto handleFallbackTransitions = [&]() {
		if (fallbackResourceIndices.empty()) {
			return;
		}

		for (size_t resourceIndex : fallbackResourceIndices) {
			rg.RecordFrameQueueTransitionBatch(gfxSlot, resourceIndex, currentBatchIndex);
		}

		for (size_t qi = 0; qi < queueCount; ++qi) {
			if (qi == gfxSlot) {
				continue;
			}

			int latestBatch = -1;
			for (size_t resourceIndex : fallbackResourceIndices) {
				latestBatch = std::max(latestBatch, static_cast<int>(GetFrameQueueHistoryValue(m_frameQueueLastTransitionBatch, qi, resourceIndex)));
				latestBatch = std::max(latestBatch, static_cast<int>(GetFrameQueueHistoryValue(m_frameQueueLastUsageBatch, qi, resourceIndex)));
			}

			if (latestBatch > 0 && static_cast<unsigned int>(latestBatch) != currentBatchIndex) {
				rg.batches[latestBatch].MarkQueueSignal(BatchSignalPhase::AfterCompletion, qi);
				currentBatch.AddQueueWait(
					BatchWaitPhase::BeforeTransitions,
					gfxSlot,
					qi,
					rg.batches[latestBatch].GetQueueSignalFenceValue(BatchSignalPhase::AfterCompletion, qi));
			}
		}
	};

	auto applyInternalTransitions = [&](const auto& pass) {
		const auto& denseTransitions = passSummary.internalTransitions;
		if (pass.resources.internalTransitions.size() != denseTransitions.size()) {
			throw std::runtime_error("Frame pass summary internal transition count mismatch");
		}

		for (size_t transitionIndex = 0; transitionIndex < denseTransitions.size(); ++transitionIndex) {
			const auto& exit = pass.resources.internalTransitions[transitionIndex];
			const auto& denseTransition = denseTransitions[transitionIndex];
			std::vector<ResourceTransition> ignoredTransitions;
			auto* pRes = exit.first.resource.IsEphemeral() ? exit.first.resource.GetEphemeralPtr() : _registry.Resolve(exit.first.resource);
			auto& compileResourceState = GetOrCreateFrameCompileResourceState(
				denseTransition.resourceIndex,
				pRes,
				denseTransition.resourceID);
			pRes = compileResourceState.resource ? compileResourceState.resource : pRes;
			compileResourceState.tracker.Apply(exit.first.range, pRes, exit.second, ignoredTransitions);
			if (IsWholeResourceRange(exit.first.range, exit.first.resource)) {
				compileResourceState.fastState.valid = true;
				compileResourceState.fastState.wholeResourceOnly = true;
				compileResourceState.fastState.state = exit.second;
			}
			else {
				compileResourceState.fastState.valid = false;
				compileResourceState.fastState.wholeResourceOnly = false;
			}
			SortedInsert(currentBatch.internallyTransitionedResources, denseTransition.resourceID);
		}
	};

	auto recordRequirementHistory = [&]() {
		for (const auto& requirement : passSummary.requirements) {
			SortedInsert(currentBatch.allResources, requirement.resourceID);
			rg.RecordFrameQueueUsageBatch(passQueueSlot, requirement.resourceIndex, currentBatchIndex);
			if (AccessTypeIsWriteType(requirement.state.access)) {
				SetFrameQueueHistoryValue(m_frameQueueLastProducerBatch, passQueueSlot, requirement.resourceIndex, currentBatchIndex);
			}
		}
	};

	handleFallbackTransitions();

	std::visit(
		[&](auto& pass) {
			using PassType = std::decay_t<decltype(pass)>;
			if constexpr (std::is_same_v<PassType, std::monostate>) {
				throw std::runtime_error("Unexpected empty pass variant in RenderGraph::CommitPassToBatch");
			}
			else {
				currentBatch.Passes(passQueueSlot).emplace_back(&pass);
				for (const auto& wait : pass.resources.externalWaitsBeforeTransitions) {
					currentBatch.AddExternalWaitBeforeTransitions(passQueueSlot, wait);
				}
				applyInternalTransitions(pass);
				recordRequirementHistory();

				for (size_t qi = 0; qi < queueCount; ++qi) {
					if (qi == passQueueSlot) {
						continue;
					}

					rg.applySynchronization(
						passQueueSlot,
						qi,
						currentBatch,
						currentBatchIndex,
						pass,
						resourcesTransitionedThisPass);
				}
			}
		},
		pr.pass);
}

void RenderGraph::AutoScheduleAndBuildBatches(
	RenderGraph& rg,
	std::vector<AnyPassAndResources>& passes,
	std::vector<Node>& nodes)
{
	ZoneScopedN("RenderGraph::AutoScheduleAndBuildBatches");
	// Working indegrees
	std::vector<uint32_t> indeg(nodes.size());
	for (size_t i = 0; i < nodes.size(); ++i) indeg[i] = nodes[i].indegree;

	std::vector<size_t> ready;
	ready.reserve(nodes.size());
	for (size_t i = 0; i < nodes.size(); ++i)
		if (indeg[i] == 0) ready.push_back(i);

	auto openNewBatch = [&]() -> PassBatch {
		const size_t queueCount = rg.m_queueRegistry.SlotCount();
		PassBatch b(queueCount);
		b.passBatchTrackersByResourceIndex.assign(rg.m_frameSchedulingResourceCount, nullptr);
		for (size_t qi = 0; qi < queueCount; ++qi) {
			b.SetQueueSignalFenceValue(RenderGraph::BatchSignalPhase::AfterTransitions, qi, rg.GetNextQueueFenceValue(qi));
			b.SetQueueSignalFenceValue(RenderGraph::BatchSignalPhase::AfterExecution, qi, rg.GetNextQueueFenceValue(qi));
			b.SetQueueSignalFenceValue(RenderGraph::BatchSignalPhase::AfterCompletion, qi, rg.GetNextQueueFenceValue(qi));
		}
		return b;
		};

	PassBatch currentBatch = openNewBatch();
	unsigned int currentBatchIndex = 1; // Start at batch 1- batch 0 is reserved for inserting transitions before first batch

	const size_t queueCount = rg.m_queueRegistry.SlotCount();
	const size_t gfxSlot = QueueIndex(QueueKind::Graphics);
	BatchBuildState batchBuildState;
	batchBuildState.Initialize(nodes.size(), queueCount, rg.m_frameSchedulingResourceCount);

	// Scratch sets reused across CommitPassToBatch calls to avoid per-call allocation
	std::unordered_set<uint64_t> scratchTransitioned;
	std::unordered_set<size_t> scratchFallback;
	std::vector<ResourceTransition> scratchTransitions;
	const double autoGraphicsBias = rg.m_getQueueSchedulingAutoGraphicsBias ? static_cast<double>(rg.m_getQueueSchedulingAutoGraphicsBias()) : 2.5;
	const double asyncOverlapBonus = rg.m_getQueueSchedulingAsyncOverlapBonus ? static_cast<double>(rg.m_getQueueSchedulingAsyncOverlapBonus()) : 3.0;
	const double crossQueueHandoffPenalty = rg.m_getQueueSchedulingCrossQueueHandoffPenalty ? static_cast<double>(rg.m_getQueueSchedulingCrossQueueHandoffPenalty()) : 2.0;
	auto queueKindName = [&rg](size_t queueIndex) -> const char* {
		if (queueIndex >= rg.m_queueRegistry.SlotCount()) {
			return "unknown";
		}
		switch (rg.m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(queueIndex)))) {
		case QueueKind::Graphics: return "graphics";
		case QueueKind::Compute:  return "compute";
		case QueueKind::Copy:     return "copy";
		default:                 return "unknown";
		}
	};

	auto closeBatch = [&]() {
		ZoneScopedN("RenderGraph::AutoScheduleAndBuildBatches::CloseBatch");
		bool hasAnyQueuedPasses = false;
		for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
			hasAnyQueuedPasses = hasAnyQueuedPasses || !currentBatch.Passes(queueIndex).empty();
		}
		if (!hasAnyQueuedPasses) {
			return;
		}
		rg.batches.push_back(std::move(currentBatch));
		currentBatch = openNewBatch();
		batchBuildState.ResetForNewBatch();
		++currentBatchIndex;
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
		return passIndex >= passes.size() || passHasImmediateWork(passes[passIndex]);
	};

	auto updateBatchMembershipForCommittedPass = [&](const Node& committedNode) {
		const auto& passSummary = rg.m_framePassSchedulingSummaries[committedNode.passIndex];
		for (size_t resourceIndex : passSummary.requiredResourceIndices) {
			batchBuildState.MarkResource(resourceIndex);
		}
		for (const auto& transition : passSummary.internalTransitions) {
			batchBuildState.MarkInternalTransition(transition.resourceIndex);
		}
		const size_t passQueueSlot = committedNode.assignedQueueSlot.value_or(committedNode.queueSlot);
		for (size_t resourceIndex : passSummary.uavResourceIndices) {
			for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
				if (queueIndex == passQueueSlot) {
					continue;
				}
				batchBuildState.MarkOtherQueueUAV(queueIndex, resourceIndex);
			}
		}
	};

	size_t remaining = nodes.size();
	bool closedBatchBeforeNextCommit = false;

	while (remaining > 0) {
		// Collect "fits" and pick best by heuristic
		int bestIdxInReady = -1;
		size_t bestQueueSlot = 0;
		double bestScore = -1e300;
		uint32_t candidateChecks = 0;
		uint32_t isNewBatchNeededChecks = 0;
		const uint32_t readySetSizeBeforeEvaluate = static_cast<uint32_t>(ready.size());
		{
			ZoneScopedN("RenderGraph::AutoScheduleAndBuildBatches::EvaluateCandidates");

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

		for (int ri = 0; ri < (int)ready.size(); ++ri) {
			size_t ni = ready[ri];

			auto& n = nodes[ni];
			const auto& passSummary = rg.m_framePassSchedulingSummaries[n.passIndex];

			for (size_t nodeQueueSlot : n.compatibleQueueSlots) {
				++candidateChecks;
				if (nodeQueueSlot >= queueCount) {
					continue;
				}
				if (nodeQueueSlot >= rg.m_activeQueueSlotsThisFrame.size() || !rg.m_activeQueueSlotsThisFrame[nodeQueueSlot]) {
					continue;
				}

				// Extra constraint: disallow cross-queue deps within the same batch.
				// A node can only join the current batch on a slot if every in-batch
				// predecessor is already assigned to that same slot.
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
				if (rg.IsNewBatchNeeded(
					passSummary,
					currentBatch.passBatchTrackersByResourceIndex,
					batchBuildState,
					passes[n.passIndex].name,
					currentBatchIndex,
					nodeQueueSlot))
				{
					continue;
				}

				// Score: pack by reusing resources already in batch, and encourage overlap
				int reuse = 0, fresh = 0;
				for (size_t resourceIndex : passSummary.touchedResourceIndices) {
					if (batchBuildState.ContainsResource(resourceIndex)) ++reuse;
					else ++fresh;
				}

				double score = 3.0 * reuse - 1.0 * fresh;

				// Encourage having more queues represented when legal.
				if (!batchHasQueue[nodeQueueSlot]) score += 2.0;
				// Encourage spreading compatible work across less-populated queues.
				score -= 0.25 * double(currentBatch.Passes(nodeQueueSlot).size());

				// Tie-break
				score += 0.05 * double(n.criticality);

				if (passes[n.passIndex].type == PassType::Compute
					&& n.queueAssignmentPolicy == QueueAssignmentPolicy::Automatic) {
					const QueueKind candidateKind = rg.m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(nodeQueueSlot)));
					const uint8_t candidateKindMask = static_cast<uint8_t>(1u << QueueIndex(candidateKind));
					size_t predecessorCrossQueueCount = 0;
					for (size_t pred : n.in) {
						const size_t predSlot = nodes[pred].assignedQueueSlot.value_or(nodes[pred].queueSlot);
						const QueueKind predKind = rg.m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(predSlot)));
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
						if (batchHasGraphicsWork || otherReadyGraphicsCandidates > 0) {
							score += asyncOverlapBonus;
						}
						else {
							score -= asyncOverlapBonus;
						}
					}
				}

				// Deterministic tie-break: prefer earlier original order slightly
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
			// Nothing ready fits: must end batch
			bool hasAnyQueuedPasses = false;
			for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
				hasAnyQueuedPasses = hasAnyQueuedPasses || !currentBatch.Passes(queueIndex).empty();
			}
			if (hasAnyQueuedPasses) {
				closeBatch();
				closedBatchBeforeNextCommit = true;
				continue;
			}
			else {
				ZoneScopedN("RenderGraph::AutoScheduleAndBuildBatches::CommitFallbackPass");
				// Should be rare; fall back by forcing one ready pass in.
				// If this happens, IsNewBatchNeeded is likely too strict on empty batch.
				size_t ni = ready.front();
				auto& n = nodes[ni];
				if (!passes[n.passIndex].name.empty()) {
					ZoneText(passes[n.passIndex].name.data(), passes[n.passIndex].name.size());
				}
				size_t fallbackSlot = n.queueSlot;
				for (size_t compatibleSlot : n.compatibleQueueSlots) {
					if (compatibleSlot < rg.m_activeQueueSlotsThisFrame.size() && rg.m_activeQueueSlotsThisFrame[compatibleSlot]) {
						fallbackSlot = compatibleSlot;
						break;
					}
				}
				n.assignedQueueSlot = fallbackSlot;
				if (n.passIndex < rg.m_assignedQueueSlotsByFramePass.size()) {
					rg.m_assignedQueueSlotsByFramePass[n.passIndex] = *n.assignedQueueSlot;
				}
				const bool isolateBatch = passForcesBatchIsolation(n.passIndex);
				if (isolateBatch) {
					closeBatch();
					closedBatchBeforeNextCommit = true;
				}
				rg.m_schedulingDecisionTrace.push_back(SchedulingDecisionTrace{
					.nodeIndex = static_cast<uint32_t>(ni),
					.passIndex = static_cast<uint32_t>(n.passIndex),
					.batchIndex = currentBatchIndex,
					.assignedQueueSlot = static_cast<uint16_t>(fallbackSlot),
					.closedBatchBefore = closedBatchBeforeNextCommit,
					.readySetSize = readySetSizeBeforeEvaluate,
					.candidateChecks = candidateChecks,
					.isNewBatchNeededChecks = isNewBatchNeededChecks,
					.fallbackCommit = true,
				});
				closedBatchBeforeNextCommit = false;
				CommitPassToBatch(
					rg, passes[n.passIndex], n,
					currentBatchIndex, currentBatch,
					scratchTransitioned,
					scratchFallback,
					scratchTransitions);
				updateBatchMembershipForCommittedPass(n);

				batchBuildState.MarkNode(ni);
				if (isolateBatch) {
					closeBatch();
					closedBatchBeforeNextCommit = true;
				}

				// Pop from ready
				ready[0] = ready.back();
				ready.pop_back();

				for (size_t v : nodes[ni].out) {
					if (--indeg[v] == 0) ready.push_back(v);
				}
				--remaining;
				if (rg.m_getHeavyDebug && rg.m_getHeavyDebug()) {
					closeBatch();
				}
				continue;
			}
		}

		// Commit chosen pass
		size_t chosenNodeIndex = ready[bestIdxInReady];
		auto& chosen = nodes[chosenNodeIndex];
		{
			ZoneScopedN("RenderGraph::AutoScheduleAndBuildBatches::CommitSelectedPass");
			if (!passes[chosen.passIndex].name.empty()) {
				ZoneText(passes[chosen.passIndex].name.data(), passes[chosen.passIndex].name.size());
			}
			chosen.assignedQueueSlot = bestQueueSlot;
			if (chosen.passIndex < rg.m_assignedQueueSlotsByFramePass.size()) {
				rg.m_assignedQueueSlotsByFramePass[chosen.passIndex] = bestQueueSlot;
			}
			const bool isolateBatch = passForcesBatchIsolation(chosen.passIndex);
			if (isolateBatch) {
				closeBatch();
				closedBatchBeforeNextCommit = true;
			}
			rg.m_schedulingDecisionTrace.push_back(SchedulingDecisionTrace{
				.nodeIndex = static_cast<uint32_t>(chosenNodeIndex),
				.passIndex = static_cast<uint32_t>(chosen.passIndex),
				.batchIndex = currentBatchIndex,
				.assignedQueueSlot = static_cast<uint16_t>(bestQueueSlot),
				.closedBatchBefore = closedBatchBeforeNextCommit,
				.readySetSize = readySetSizeBeforeEvaluate,
				.candidateChecks = candidateChecks,
				.isNewBatchNeededChecks = isNewBatchNeededChecks,
				.fallbackCommit = false,
			});
			closedBatchBeforeNextCommit = false;
			CommitPassToBatch(
				rg, passes[chosen.passIndex], chosen,
				currentBatchIndex, currentBatch,
				scratchTransitioned,
				scratchFallback,
				scratchTransitions);
			updateBatchMembershipForCommittedPass(chosen);
			if (isolateBatch) {
				closeBatch();
				closedBatchBeforeNextCommit = true;
			}
		}

		batchBuildState.MarkNode(chosenNodeIndex);

		// Remove from ready
		ready[bestIdxInReady] = ready.back();
		ready.pop_back();

		// Release successors
		for (size_t v : chosen.out) {
			if (--indeg[v] == 0) ready.push_back(v);
		}

		--remaining;

		// Heavy-debug: isolate every pass into its own batch.
		if (rg.m_getHeavyDebug && rg.m_getHeavyDebug()) {
			closeBatch();
		}
	}

	// Final batch
	bool hasAnyQueuedPasses = false;
	for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
		hasAnyQueuedPasses = hasAnyQueuedPasses || !currentBatch.Passes(queueIndex).empty();
	}
	if (hasAnyQueuedPasses) {
		rg.batches.push_back(std::move(currentBatch));
	}

	{
		ZoneScopedN("RenderGraph::AutoScheduleAndBuildBatches::CoalesceQueueWaits");
		rg.CoalesceQueueWaitsAndSignals(rg.batches);
	}

	// Build cross-frame producer tracking from the committed batch schedule.
	// Cross-frame tracking needs to know which queue
	// wrote each resource and in which batch so the next frame can insert
	// frame-start waits.
	{
		ZoneScopedN("RenderGraph::AutoScheduleAndBuildBatches::BuildCrossFrameProducerTracking");
		std::vector<std::unordered_map<uint64_t, unsigned int>> crossFrameProducer(queueCount);
		for (unsigned int bi = 1; bi < static_cast<unsigned int>(rg.batches.size()); ++bi) {
			auto& batch = rg.batches[bi];
			for (size_t qi = 0; qi < queueCount; ++qi) {
				for (auto& passVariant : batch.Passes(qi)) {
					std::visit([&](const auto* passEntry) {
						ForEachFrameRequirement(passEntry->resources, [&](const auto& req) {
							if (AccessTypeIsWriteType(req.state.access)) {
								crossFrameProducer[qi][req.resourceHandleAndRange.resource.GetGlobalResourceID()] = bi;
							}
						});
					}, passVariant);
				}
			}
		}
		rg.m_compiledLastProducerBatchByResourceByQueue = std::move(crossFrameProducer);
	}

	if (rg.m_getRenderGraphBatchTraceEnabled && rg.m_getRenderGraphBatchTraceEnabled()) {
		rg.LogAddTransitionDebugSummary();
	}
}


// Factory for the transition lambda
void RenderGraph::LogAddTransitionDebugSummary() const
{
	if (m_addTransitionDebugStatsByResource.empty()) {
		return;
	}

	size_t totalCalls = 0;
	size_t totalNoOpCalls = 0;
	size_t totalEmittedTransitions = 0;
	size_t totalEarlyPlacedTransitions = 0;
	size_t totalBeforePassTransitions = 0;
	size_t totalGraphicsFallbackTransitions = 0;
	size_t totalAliasActivationTransitions = 0;

	std::vector<std::pair<uint64_t, const AddTransitionDebugStats*>> ranked;
	ranked.reserve(m_addTransitionDebugStatsByResource.size());

	for (const auto& [resourceID, stats] : m_addTransitionDebugStatsByResource) {
		totalCalls += stats.callCount;
		totalNoOpCalls += stats.noOpCallCount;
		totalEmittedTransitions += stats.emittedTransitionCount;
		totalEarlyPlacedTransitions += stats.earlyPlacedTransitionCount;
		totalBeforePassTransitions += stats.beforePassTransitionCount;
		totalGraphicsFallbackTransitions += stats.graphicsFallbackTransitionCount;
		totalAliasActivationTransitions += stats.aliasActivationTransitionCount;
		ranked.emplace_back(resourceID, &stats);
	}

	std::sort(
		ranked.begin(),
		ranked.end(),
		[](const auto& lhs, const auto& rhs) {
			if (lhs.second->callCount != rhs.second->callCount) {
				return lhs.second->callCount > rhs.second->callCount;
			}
			if (lhs.second->emittedTransitionCount != rhs.second->emittedTransitionCount) {
				return lhs.second->emittedTransitionCount > rhs.second->emittedTransitionCount;
			}
			return lhs.first < rhs.first;
		});

	spdlog::info(
		"RG AddTransition summary: resources={} calls={} noOpCalls={} emittedTransitions={} earlyPlacedTransitions={} beforePassTransitions={} graphicsFallbackTransitions={} aliasActivationTransitions={}",
		ranked.size(),
		totalCalls,
		totalNoOpCalls,
		totalEmittedTransitions,
		totalEarlyPlacedTransitions,
		totalBeforePassTransitions,
		totalGraphicsFallbackTransitions,
		totalAliasActivationTransitions);

	constexpr size_t kMaxLoggedResources = 12;
	const size_t resourcesToLog = std::min(kMaxLoggedResources, ranked.size());
	for (size_t index = 0; index < resourcesToLog; ++index) {
		const uint64_t resourceID = ranked[index].first;
		const AddTransitionDebugStats& stats = *ranked[index].second;
		const std::string_view resourceName = stats.resourceName.empty() ? std::string_view("<unknown>") : std::string_view(stats.resourceName);
		spdlog::info(
			"RG AddTransition top[{}]: resource='{}' id={} calls={} noOpCalls={} emittedTransitions={} earlyPlacedTransitions={} beforePassTransitions={} graphicsFallbackTransitions={} aliasActivationTransitions={}",
			index,
			resourceName,
			resourceID,
			stats.callCount,
			stats.noOpCallCount,
			stats.emittedTransitionCount,
			stats.earlyPlacedTransitionCount,
			stats.beforePassTransitionCount,
			stats.graphicsFallbackTransitionCount,
			stats.aliasActivationTransitionCount);
	}
}

void RenderGraph::CoalesceQueueWaitsAndSignals(std::vector<PassBatch>& batchesToCoalesce) const
{
	// Coalesce redundant waits while preserving same-batch fence phase ordering.
	for (auto& batch : batchesToCoalesce) {
		const size_t batchQueueCount = batch.QueueCount();
		for (size_t dst = 0; dst < batchQueueCount; ++dst) {
			for (size_t src = 0; src < batchQueueCount; ++src) {
				if (dst == src) continue;

				int enabledCount = 0;
				for (size_t phase = 0; phase < PassBatch::kWaitPhaseCount; ++phase) {
					if (batch.queueWaitEnabled[phase][dst][src]) ++enabledCount;
				}
				if (enabledCount <= 1) continue;

				auto isSameBatchFence = [&](UINT64 f) -> bool {
					for (size_t sp = 0; sp < PassBatch::kSignalPhaseCount; ++sp) {
						if (f == batch.queueSignalFenceValue[sp][src]) return true;
					}
					return false;
				};

				for (size_t i = 0; i < PassBatch::kWaitPhaseCount; ++i) {
					if (!batch.queueWaitEnabled[i][dst][src]) continue;
					UINT64 fi = batch.queueWaitFenceValue[i][dst][src];
					for (size_t j = i + 1; j < PassBatch::kWaitPhaseCount; ++j) {
						if (!batch.queueWaitEnabled[j][dst][src]) continue;
						if (batch.queueWaitFenceValue[j][dst][src] <= fi) {
							batch.queueWaitEnabled[j][dst][src] = false;
							batch.queueWaitFenceValue[j][dst][src] = 0;
						}
					}
				}

				int earliest = -1;
				for (size_t phase = 0; phase < PassBatch::kWaitPhaseCount; ++phase) {
					if (batch.queueWaitEnabled[phase][dst][src]) {
						earliest = static_cast<int>(phase);
						break;
					}
				}
				if (earliest < 0) continue;

				for (size_t j = static_cast<size_t>(earliest) + 1; j < PassBatch::kWaitPhaseCount; ++j) {
					if (!batch.queueWaitEnabled[j][dst][src]) continue;
					UINT64 fj = batch.queueWaitFenceValue[j][dst][src];
					if (isSameBatchFence(fj)) continue;

					if (fj > batch.queueWaitFenceValue[earliest][dst][src]) {
						batch.queueWaitFenceValue[earliest][dst][src] = fj;
					}
					batch.queueWaitEnabled[j][dst][src] = false;
					batch.queueWaitFenceValue[j][dst][src] = 0;
				}

				for (size_t i = 0; i < PassBatch::kWaitPhaseCount; ++i) {
					if (!batch.queueWaitEnabled[i][dst][src]) continue;
					UINT64 fi = batch.queueWaitFenceValue[i][dst][src];
					for (size_t j = i + 1; j < PassBatch::kWaitPhaseCount; ++j) {
						if (!batch.queueWaitEnabled[j][dst][src]) continue;
						if (batch.queueWaitFenceValue[j][dst][src] <= fi) {
							batch.queueWaitEnabled[j][dst][src] = false;
							batch.queueWaitFenceValue[j][dst][src] = 0;
						}
					}
				}
			}
		}
	}
}

void RenderGraph::AssignQueueSignalFenceValuesInSubmissionOrder(std::vector<PassBatch>& batchesToAssign)
{
	ZoneScopedN("RenderGraph::AssignQueueSignalFenceValuesInSubmissionOrder");
	const size_t slotCount = m_queueRegistry.SlotCount();
	std::vector<std::unordered_map<UINT64, UINT64>> remappedFenceValuesByQueue(slotCount);

	for (auto& batch : batchesToAssign) {
		const size_t queueCount = std::min(batch.QueueCount(), slotCount);
		for (size_t qi = 0; qi < queueCount; ++qi) {
			for (size_t phaseIndex = 0; phaseIndex < PassBatch::kSignalPhaseCount; ++phaseIndex) {
				const auto phase = static_cast<BatchSignalPhase>(phaseIndex);
				if (!batch.HasQueueSignal(phase, qi)) {
					continue;
				}

				const UINT64 previousFenceValue = batch.GetQueueSignalFenceValue(phase, qi);
				const UINT64 submissionOrderFenceValue = GetNextQueueFenceValue(qi);
				batch.SetQueueSignalFenceValue(phase, qi, submissionOrderFenceValue);
				if (previousFenceValue != 0) {
					remappedFenceValuesByQueue[qi][previousFenceValue] = submissionOrderFenceValue;
				}
			}
		}
	}

	for (auto& batch : batchesToAssign) {
		const size_t queueCount = std::min(batch.QueueCount(), slotCount);
		for (size_t waitPhaseIndex = 0; waitPhaseIndex < PassBatch::kWaitPhaseCount; ++waitPhaseIndex) {
			for (size_t dst = 0; dst < queueCount; ++dst) {
				for (size_t src = 0; src < queueCount; ++src) {
					if (!batch.queueWaitEnabled[waitPhaseIndex][dst][src]) {
						continue;
					}
					const UINT64 previousFenceValue = batch.queueWaitFenceValue[waitPhaseIndex][dst][src];
					auto remapIt = remappedFenceValuesByQueue[src].find(previousFenceValue);
					if (remapIt != remappedFenceValuesByQueue[src].end()) {
						batch.queueWaitFenceValue[waitPhaseIndex][dst][src] = remapIt->second;
					}
				}
			}
		}
	}
}

void RenderGraph::AddTransition(
	unsigned int batchIndex,
	PassBatch& currentBatch,
	size_t passQueueSlot,
	std::string_view passName,
	const DenseRequirementSummary& requirement,
	std::unordered_set<uint64_t>& outTransitionedResourceIDs,
	std::unordered_set<size_t>& outFallbackResourceIndices,
	std::vector<ResourceTransition>& scratchTransitions)
{
	ZoneScopedN("RenderGraph::AddTransition");
	const QueueKind passQueue = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(passQueueSlot));
	const ResourceState requiredState = NormalizeStateForQueue(passQueue, requirement.state);

	if (TryAddTransitionFastNoOp(batchIndex, currentBatch, passQueueSlot, requirement, requiredState)) {
		return;
	}

	AddTransitionSlowPath(
		batchIndex,
		currentBatch,
		passQueueSlot,
		passName,
		requirement,
		requiredState,
		outTransitionedResourceIDs,
		outFallbackResourceIndices,
		scratchTransitions);
}

bool RenderGraph::TryAddTransitionFastNoOp(
	unsigned int batchIndex,
	PassBatch& currentBatch,
	size_t passQueueSlot,
	const DenseRequirementSummary& requirement,
	ResourceState requiredState)
{
	(void)batchIndex;
	(void)passQueueSlot;

	if (requirement.resourceIndex >= m_frameCompileResources.size()) {
		return false;
	}
	if (requirement.resourceIndex >= m_aliasActivationPendingByResourceIndex.size()) {
		return false;
	}
	if (m_aliasActivationPendingByResourceIndex[requirement.resourceIndex] != 0) {
		return false;
	}

	auto& entry = m_frameCompileResources[requirement.resourceIndex];
	if (!entry.fastState.valid || !entry.fastState.wholeResourceOnly) {
		return false;
	}
	if (!IsWholeResourceRange(requirement.range, requirement.resource)) {
		return false;
	}
	if (!StatesExactlyEqual(entry.fastState.state, requiredState)) {
		return false;
	}

	currentBatch.passBatchTrackersByResourceIndex[requirement.resourceIndex] = &entry.tracker;
	return true;
}

void RenderGraph::AddTransitionSlowPath(
	unsigned int batchIndex,
	PassBatch& currentBatch,
	size_t passQueueSlot,
	std::string_view passName,
	const DenseRequirementSummary& requirement,
	ResourceState requiredState,
	std::unordered_set<uint64_t>& outTransitionedResourceIDs,
	std::unordered_set<size_t>& outFallbackResourceIndices,
	std::vector<ResourceTransition>& scratchTransitions)
{
	ZoneScopedN("RenderGraph::AddTransitionSlowPath");

	auto resource = requirement.resource;
	const QueueKind passQueue = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(passQueueSlot));

	// If this triggers, you're probably queueing an operation on an external/ephemeral resource, and then discarding it before the graph can use it.
	if (!resource.IsEphemeral() && !_registry.IsValid(resource)) {
		auto uploadQueueMatch = [&]() -> std::string {
			if (passName != "Builtin::Uploads") {
				return {};
			}

			return UploadManager::GetInstance().DescribeQueuedTargetByGlobalResourceId(resource.GetGlobalResourceID());
		}();
		auto resourceName = [&]() -> std::string {
			auto resourceIt = resourcesByID.find(resource.GetGlobalResourceID());
			if (resourceIt != resourcesByID.end() && resourceIt->second) {
				const auto& name = resourceIt->second->GetName();
				if (!name.empty()) {
					return name;
				}
			}
			return std::string("<unknown>");
		}();
		const std::string registryHandleInfo = _registry.DescribeHandle(resource);
		spdlog::error(
			"Invalid resource handle in RenderGraph::AddTransition: pass='{}' resourceId={} keyIdx={} generation={} epoch={} resourceName='{}' uploadQueueMatch='{}' registryHandleInfo='{}' range={} access={} layout={} sync={}",
			passName,
			resource.GetGlobalResourceID(),
			resource.GetKey().idx,
			resource.GetGeneration(),
			resource.GetEpoch(),
			resourceName,
			uploadQueueMatch.empty() ? std::string("<none>") : uploadQueueMatch,
			registryHandleInfo,
			FormatRangeSpec(requirement.range),
			static_cast<uint32_t>(requiredState.access),
			static_cast<uint32_t>(requiredState.layout),
			static_cast<uint32_t>(requiredState.sync));
		throw (std::runtime_error("Invalid resource handle in RenderGraph::AddTransition"));
	}
	scratchTransitions.clear();
	auto& transitions = scratchTransitions;
	auto* pRes = resource.IsEphemeral() ? resource.GetEphemeralPtr() : _registry.Resolve(resource); // TODO: Can we get rid of pRes in transitions?
	auto& compileResourceState = GetOrCreateFrameCompileResourceState(requirement.resourceIndex, pRes, requirement.resourceID);
	pRes = compileResourceState.resource ? compileResourceState.resource : pRes;
	auto& fastState = compileResourceState.fastState;
	if (pRes && !pRes->GetName().empty()) {
		ZoneText(pRes->GetName().data(), pRes->GetName().size());
	}
	AddTransitionDebugStats* debugStats = nullptr;
	if (m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled()) {
		auto [it, inserted] = m_addTransitionDebugStatsByResource.try_emplace(requirement.resourceID);
		(void)inserted;
		debugStats = &it->second;
		if (debugStats->resourceName.empty() || debugStats->resourceName == "<unknown>") {
			if (pRes && !pRes->GetName().empty()) {
				debugStats->resourceName = pRes->GetName();
			}
			else if (auto resourceIt = resourcesByID.find(requirement.resourceID);
				resourceIt != resourcesByID.end() && resourceIt->second && !resourceIt->second->GetName().empty()) {
				debugStats->resourceName = resourceIt->second->GetName();
			}
			else if (resource.IsEphemeral()) {
				debugStats->resourceName = "<ephemeral>";
			}
			else {
				debugStats->resourceName = "<unknown>";
			}
		}
		++debugStats->callCount;
	}
	auto& compileTracker = compileResourceState.tracker;
	const bool isWholeResourceRequirement = IsWholeResourceRange(requirement.range, resource);

	bool isAliasActivation = false;
	if (requirement.resourceIndex < m_aliasActivationPendingByResourceIndex.size() && m_aliasActivationPendingByResourceIndex[requirement.resourceIndex] != 0) {
		isAliasActivation = true;
		const bool firstUseIsWrite = AccessTypeIsWriteType(requirement.state.access);
		const bool firstUseIsCommon = requirement.state.access == rhi::ResourceAccessType::Common;
		// Common counts as write for alias activation, as this is generally used to indicate that the resource will be
		// transitioned internally by an external system that still uses legacy barriers. Don't abuse this.
		if (firstUseIsWrite || firstUseIsCommon) { 
			const uint64_t id = requirement.resourceID;
			auto itSig = aliasPlacementSignatureByID.find(id);
			ResourceState activationBeforeState{
				rhi::ResourceAccessType::None,
				rhi::ResourceLayout::Undefined,
				rhi::ResourceSyncState::None };
			ResourceState trackedBeforeState{};
			if (TryGetWholeResourceTrackerState(compileTracker, trackedBeforeState)) {
				activationBeforeState = trackedBeforeState;
			}
			//spdlog::info(
			//	"RG alias activate: id={} name='{}' signature={} accessAfter={} layoutAfter={} syncAfter={} discard=1",
			//	id,
			//	pRes ? pRes->GetName() : std::string("<null>"),
			//	itSig != aliasPlacementSignatureByID.end() ? itSig->second : 0ull,
			//	static_cast<uint32_t>(r.state.access),
			//	static_cast<uint32_t>(r.state.layout),
			//	static_cast<uint32_t>(r.state.sync));
			transitions.emplace_back(
				pRes,
				requirement.range,
				activationBeforeState.access,
				requiredState.access,
				activationBeforeState.layout,
				requiredState.layout,
				activationBeforeState.sync,
				requiredState.sync,
				true);
		}
		else {
			const auto* placement = TryGetAliasPlacementRange(requirement.resourceID);
			spdlog::error(
				"RG alias activation rejected read first use: pass=\"{}\" resource_id={} resource_name=\"{}\" first_use_pass={} first_use_pass_name=\"{}\"",
				passName,
				requirement.resourceID,
				pRes ? pRes->GetName() : std::string("<null>"),
				placement ? static_cast<uint64_t>(placement->firstUsePassIndex) : UINT64_MAX,
				placement && placement->firstUsePassIndex < m_framePasses.size() ? m_framePasses[placement->firstUsePassIndex].name : std::string("<unknown>"));
			throw std::runtime_error("Alias activation requires first use to be a write when explicit initialization is disabled");
		}
		std::vector<ResourceTransition> ignored;
		compileTracker.Apply(requirement.range, pRes, requiredState, ignored);
		aliasActivationPending.erase(requirement.resourceID);
		m_aliasActivationPendingByResourceIndex[requirement.resourceIndex] = 0;
	}
	else {
		compileTracker.Apply(requirement.range, pRes, requiredState, transitions);
	}

	if (isWholeResourceRequirement) {
		fastState.valid = true;
		fastState.wholeResourceOnly = true;
		fastState.state = requiredState;
	}
	else {
		fastState.valid = false;
		fastState.wholeResourceOnly = false;
	}

	if (debugStats) {
		debugStats->emittedTransitionCount += transitions.size();
		if (isAliasActivation) {
			debugStats->aliasActivationTransitionCount += transitions.size();
		}
	}

	if (!transitions.empty()) {
		outTransitionedResourceIDs.insert(requirement.resourceID);
	}

	currentBatch.passBatchTrackersByResourceIndex[requirement.resourceIndex] = &compileTracker; // We will need to check subsequent passes against this

	if (transitions.empty()) {
		if (debugStats) {
			++debugStats->noOpCallCount;
		}
		return;
	}

	bool needsGraphicsQueueForTransitions = false;
	for (auto& transition : transitions) {
		if (!QueueSupportsTransition(passQueue, transition)) {
			needsGraphicsQueueForTransitions = true;
			break;
		}
	}

	const size_t gfxSlot = QueueIndex(QueueKind::Graphics);
	const size_t transitionSlot = (passQueue != QueueKind::Graphics && needsGraphicsQueueForTransitions)
		? gfxSlot : passQueueSlot;

	unsigned int lastUseBatch = 0;
	uint64_t lastUseQueueMask = 0;
	bool requiresCrossQueuePlacementCoordination = false;
	if (!isAliasActivation) {
		const bool canUseEventSummary = !m_frameResourceEventSummaries.empty();
		if (canUseEventSummary) {
			std::tie(lastUseBatch, lastUseQueueMask) = GetFrameResourceLastEventBeforeBatch(requirement.resourceIndex, batchIndex);
			requiresCrossQueuePlacementCoordination =
				lastUseQueueMask != 0 &&
				(lastUseQueueMask & ~(uint64_t{ 1 } << transitionSlot)) != 0;
		}
		else {
			int scannedLastUseBatch = -1;
			for (size_t qi = 0; qi < m_queueRegistry.SlotCount(); ++qi) {
				const unsigned int usageBatch = GetFrameQueueHistoryValue(m_frameQueueLastUsageBatch, qi, requirement.resourceIndex);
				if (usageBatch < batchIndex) {
					scannedLastUseBatch = std::max(scannedLastUseBatch, static_cast<int>(usageBatch));
				}
				const unsigned int transitionBatch = GetFrameQueueHistoryValue(m_frameQueueLastTransitionBatch, qi, requirement.resourceIndex);
				if (transitionBatch < batchIndex) {
					scannedLastUseBatch = std::max(scannedLastUseBatch, static_cast<int>(transitionBatch));
				}
			}
			if (scannedLastUseBatch > 0) {
				lastUseBatch = static_cast<unsigned int>(scannedLastUseBatch);
				for (size_t qi = 0; qi < m_queueRegistry.SlotCount(); ++qi) {
					if (qi == transitionSlot) {
						continue;
					}

					const bool usedInTargetBatch =
						GetFrameQueueHistoryValue(m_frameQueueLastUsageBatch, qi, requirement.resourceIndex) == lastUseBatch
						|| GetFrameQueueHistoryValue(m_frameQueueLastTransitionBatch, qi, requirement.resourceIndex) == lastUseBatch;
					if (usedInTargetBatch) {
						requiresCrossQueuePlacementCoordination = true;
						break;
					}
				}
			}
		}
	}

	m_transitionPlacementStats.candidateCount += transitions.size();
	m_transitionPlacementStats.emittedTransitionCount += transitions.size();
	if (isAliasActivation) {
		m_transitionPlacementStats.aliasActivationCount += transitions.size();
	}
	if (needsGraphicsQueueForTransitions) {
		m_transitionPlacementStats.graphicsFallbackCount += transitions.size();
	}
	if (lastUseBatch > 0 && !requiresCrossQueuePlacementCoordination && !isAliasActivation) {
		m_transitionPlacementStats.oldInlineEarlyEligibleCount += transitions.size();
	}
	else if (requiresCrossQueuePlacementCoordination) {
		m_transitionPlacementStats.crossQueueCoordinationBlockedCount += transitions.size();
	}
	for (const auto& transition : transitions) {
		m_transitionPlacementCandidates.push_back(TransitionPlacementCandidate{
			.transitionId = static_cast<uint32_t>(m_transitionPlacementCandidates.size()),
			.consumerBatch = batchIndex,
			.consumerQueueSlot = static_cast<uint16_t>(passQueueSlot),
			.transitionQueueSlot = static_cast<uint16_t>(transitionSlot),
			.resourceIndex = requirement.resourceIndex,
			.resourceID = resource.GetGlobalResourceID(),
			.lastUseBatch = lastUseBatch,
			.lastUseQueueMask = lastUseQueueMask,
			.isAliasActivation = isAliasActivation,
			.needsGraphicsFallback = needsGraphicsQueueForTransitions,
			.requiresCrossQueuePlacementCoordination = requiresCrossQueuePlacementCoordination,
			.transition = transition,
		});
	}

	const auto transitionPlacementMode = m_getTransitionPlacementMode
		? m_getTransitionPlacementMode()
		: rg::runtime::TransitionPlacementMode::InlineEarlyPlacement;
	if (transitionPlacementMode == rg::runtime::TransitionPlacementMode::CanonicalThenOptimize) {
		if (passQueue != QueueKind::Graphics && needsGraphicsQueueForTransitions) {
			for (auto& transition : transitions) {
				currentBatch.Transitions(gfxSlot, BatchTransitionPhase::BeforePasses).push_back(transition);
			}
			if (debugStats) {
				debugStats->beforePassTransitionCount += transitions.size();
				debugStats->graphicsFallbackTransitionCount += transitions.size();
			}
			outFallbackResourceIndices.insert(requirement.resourceIndex);
		}
		else {
			for (auto& transition : transitions) {
				currentBatch.Transitions(passQueueSlot, BatchTransitionPhase::BeforePasses).push_back(transition);
			}
			if (debugStats) {
				debugStats->beforePassTransitionCount += transitions.size();
			}
		}
		m_transitionPlacementStats.canonicalBeforePassCount += transitions.size();
		return;
	}

	// Try early placement: move transitions to AfterPasses of the batch where the resource was last used.
	// This reduces GPU idle time by allowing transitions to overlap with unrelated work on other queues.
	// Skip alias activations - those must stay in the consuming batch (discard semantics at first use).
	if (!isAliasActivation) {
		if (lastUseBatch > 0 && !requiresCrossQueuePlacementCoordination) { // > 0 to skip batch 0 (placeholder with no fence values)
			PassBatch& targetBatch = batches[lastUseBatch];

			for (auto& transition : transitions) {
				targetBatch.Transitions(transitionSlot, BatchTransitionPhase::AfterPasses).push_back(transition);
			}
			if (debugStats) {
				debugStats->earlyPlacedTransitionCount += transitions.size();
			}
			m_transitionPlacementStats.inlineEarlyPlacedCount += transitions.size();

			// Signal AfterCompletion on the transition queue so downstream consumers can wait on it
			targetBatch.MarkQueueSignal(BatchSignalPhase::AfterCompletion, transitionSlot);

			// Update tracking: the transition is now in the earlier batch
			RecordFrameQueueTransitionBatch(transitionSlot, requirement.resourceIndex, lastUseBatch);

			// Do NOT add to outFallbackResourceIDs- applySynchronization will handle
			// cross-queue waits based on the updated tracking maps.
			return;
		}
	}

	// Fallback: place in current batch's BeforePasses (existing behavior for first use or alias activations)
	if (passQueue != QueueKind::Graphics && needsGraphicsQueueForTransitions) {
		// The consuming pass's queue can't support these transitions, so delegate
		// them to the graphics queue within the *current* batch's BeforePasses phase.
		// CommitPassToBatch will set up:
		//   1. BeforeTransitions waits on Graphics for any prior non-graphics producers
		//   2. AfterTransitions signal on Graphics so the consuming queue can wait
		for (auto& transition : transitions) {
			currentBatch.Transitions(gfxSlot, BatchTransitionPhase::BeforePasses).push_back(transition);
		}
		if (debugStats) {
			debugStats->beforePassTransitionCount += transitions.size();
			debugStats->graphicsFallbackTransitionCount += transitions.size();
		}
		outFallbackResourceIndices.insert(requirement.resourceIndex);
	}
	else {
		for (auto& transition : transitions) {
			currentBatch.Transitions(passQueueSlot, BatchTransitionPhase::BeforePasses).push_back(transition);
		}
		if (debugStats) {
			debugStats->beforePassTransitionCount += transitions.size();
		}
	}
	m_transitionPlacementStats.canonicalBeforePassCount += transitions.size();
}

void RenderGraph::ProcessResourceRequirements(
	size_t passQueueSlot,
	const std::vector<DenseRequirementSummary>& resourceRequirements,
	std::string_view passName,
	unsigned int batchIndex,
	PassBatch& currentBatch, std::unordered_set<uint64_t>& outTransitionedResourceIDs,
	std::unordered_set<size_t>& outFallbackResourceIndices,
	std::vector<ResourceTransition>& scratchTransitions) {
	ZoneScopedN("RenderGraph::ProcessResourceRequirements");
	const bool enableReadOnlyUniformTransitionElision =
		m_getReadOnlyUniformTransitionElisionEnabled && m_getReadOnlyUniformTransitionElisionEnabled();

	for (const auto& resourceRequirement : resourceRequirements) {
		const bool isReadOnlyUniform =
			enableReadOnlyUniformTransitionElision
			&&
			resourceRequirement.resourceIndex < m_frameResourceAccessSummaries.size()
			&& m_frameResourceAccessSummaries[resourceRequirement.resourceIndex].readOnlyUniform;

		if (isReadOnlyUniform) {
			Resource* resource = resourceRequirement.resource.IsEphemeral()
				? resourceRequirement.resource.GetEphemeralPtr()
				: _registry.Resolve(resourceRequirement.resource);
			auto& compileResourceState = GetOrCreateFrameCompileResourceState(
				resourceRequirement.resourceIndex,
				resource,
				resourceRequirement.resourceID);
			if (!compileResourceState.readOnlyUniformTransitionChecked) {
				AddTransition(batchIndex, currentBatch, passQueueSlot, passName, resourceRequirement, outTransitionedResourceIDs, outFallbackResourceIndices, scratchTransitions);
				compileResourceState.readOnlyUniformTransitionChecked = true;
			}
			else {
				currentBatch.passBatchTrackersByResourceIndex[resourceRequirement.resourceIndex] = &compileResourceState.tracker;
			}
		}
		else {
			AddTransition(batchIndex, currentBatch, passQueueSlot, passName, resourceRequirement, outTransitionedResourceIDs, outFallbackResourceIndices, scratchTransitions);
		}

		if (AccessTypeIsWriteType(resourceRequirement.state.access)) {
			RecordFrameQueueTransitionBatch(passQueueSlot, resourceRequirement.resourceIndex, batchIndex);
		}
	}
}

bool ResolveFirstMipSlice(ResourceRegistry::RegistryHandle r, RangeSpec range, uint32_t& outMip, uint32_t& outSlice) noexcept
{
	const uint32_t totalMips = r.GetNumMipLevels();
	const uint32_t totalSlices = r.GetArraySize();
	if (totalMips == 0 || totalSlices == 0) return false;

	SubresourceRange sr = ResolveRangeSpec(range, totalMips, totalSlices);
	if (sr.isEmpty()) return false;

	outMip = sr.firstMip;
	outSlice = sr.firstSlice;
	return true;
}

RenderGraph::RenderGraph(rhi::Device device) {
	DeviceManager::GetInstance().Initialize(device);

	auto MakeDefaultImmediateDispatch = [&]() noexcept -> rg::imm::ImmediateDispatch
		{
			rg::imm::ImmediateDispatch d{};
			d.user = this;

			d.GetResourceHandle = [](RenderGraph* user, ResourceRegistry::RegistryHandle r) noexcept -> rhi::ResourceHandle {
				Resource* ptr;
				if (r.IsEphemeral()) {
					ptr = r.GetEphemeralPtr();
				}
				else {
					ptr = user->_registry.Resolve(r);
				}
				return ptr ? ptr->GetAPIResource().GetHandle() : rhi::ResourceHandle{};
				};

			d.GetRTV = +[](RenderGraph* user, ResourceRegistry::RegistryHandle r, RangeSpec range) noexcept -> rhi::DescriptorSlot {
				Resource* resource = r.IsEphemeral() ? r.GetEphemeralPtr() : user->_registry.Resolve(r);

				uint32_t mip = 0, slice = 0;
				if (!ResolveFirstMipSlice(r, range, mip, slice)) return {};

				return ResolveRTVSlot(resource, mip, slice);
				};

			d.GetDSV = +[](RenderGraph* user, ResourceRegistry::RegistryHandle r, RangeSpec range) noexcept -> rhi::DescriptorSlot {
				Resource* resource = r.IsEphemeral() ? r.GetEphemeralPtr() : user->_registry.Resolve(r);
				auto* gir = dynamic_cast<GloballyIndexedResource*>(resource);
				if (!gir || !gir->HasDSV()) return {};

				uint32_t mip = 0, slice = 0;
				if (!ResolveFirstMipSlice(r, range, mip, slice)) return {};

				return gir->GetDSVInfo(mip, slice).slot;
				};

			d.GetUavClearInfo = +[](RenderGraph* user, ResourceRegistry::RegistryHandle r, RangeSpec range, rhi::UavClearInfo& out) noexcept -> bool {
				Resource* resource = r.IsEphemeral() ? r.GetEphemeralPtr() : user->_registry.Resolve(r);
				auto* gir = dynamic_cast<GloballyIndexedResource*>(resource);

				// DX12 path requires both a shader-visible and CPU-visible UAV descriptor.
				if (!gir || !gir->HasUAVShaderVisible() || !gir->HasUAVNonShaderVisible()) return false;

				uint32_t mip = 0, slice = 0;
				if (!ResolveFirstMipSlice(r, range, mip, slice)) return false;

				out.shaderVisible = gir->GetUAVShaderVisibleInfo(mip, slice).slot;
				out.cpuVisible = gir->GetUAVNonShaderVisibleInfo(mip, slice).slot;

				out.resource = gir->GetAPIResource();

				return true;
				};

			return d;
		};

	m_immediateDispatch = MakeDefaultImmediateDispatch();
	if (!m_statisticsService) {
		m_statisticsService = rg::runtime::CreateDefaultStatisticsService();
	}
	if (!m_uploadService) {
		m_uploadService = rg::runtime::CreateDefaultUploadService();
	}
	if (!m_readbackService) {
		m_readbackService = rg::runtime::CreateDefaultReadbackService();
	}
	if (!m_descriptorService) {
		m_descriptorService = rg::runtime::CreateDefaultDescriptorService();
	}
	if (!m_renderGraphSettingsService) {
		m_renderGraphSettingsService = rg::runtime::CreateDefaultRenderGraphSettingsService();
	}
}

RenderGraph::~RenderGraph() {
	if (m_pCommandRecordingManager) {
		m_pCommandRecordingManager->ShutdownThreadLocal(); // Clears thread-local storage
	}
	ShutdownOwnedState();
}

void RenderGraph::ShutdownRuntime() {
	StatisticsManager::GetInstance().ClearAll();
	DeletionManager::GetInstance().DrainAll();
	DeletionManager::GetInstance().Cleanup();
	DeviceManager::GetInstance().Cleanup();
}

void RenderGraph::ShutdownOwnedState() {
	batches.clear();
	initialTransitions.clear();
	trackers.clear();
	m_frameCompileResources.clear();
	m_addTransitionDebugStatsByResource.clear();
	m_masterPassList.clear();
	m_framePasses.clear();
	m_framePassIsFrameExtension.clear();
	m_framePassDeclarationRefreshedThisFrame.clear();
	m_assignedQueueSlotsByFramePass.clear();
	m_activeQueueSlotsThisFrame.clear();
	renderPassesByName.clear();
	computePassesByName.clear();
	resourcesByID.clear();
	resourcesByName.clear();
	m_transientFrameResourcesByID.clear();
	m_dynamicResourcesByStableID.clear();
	m_transientFrameResourcesByName.clear();
	resourceBackingGenerationByID.clear();
	resourceIdleFrameCounts.clear();
	compiledResourceGenerationByID.clear();
	aliasMaterializeOptionsByID.clear();
	aliasPlacementSignatureByID.clear();
	aliasPlacementRangesByID.clear();
	schedulingPlacementRangesByID.clear();
	m_schedulingEquivalentIDsCache.clear();
	m_regionCache = {};
	m_lastRegionStats = {};
	m_lastExtractedRegions.clear();
	m_schedulingDecisionTrace.clear();
	m_transitionPlacementCandidates.clear();
	m_transitionPlacementStats = {};
	ClearFrameSchedulingResourceIndex();
	ClearFramePassSchedulingSummaries();
	aliasPlacementPoolByID.clear();
	aliasActivationPending.clear();
	persistentAliasPools.clear();
	autoAliasPoolByID.clear();
	m_lastProducerByResourceAcrossFrames.clear();
	m_lastAliasPlacementProducersByPoolAcrossFrames.clear();
	for (auto& producerMap : m_compiledLastProducerBatchByResourceByQueue) {
		producerMap.clear();
	}
	for (auto& row : m_hasPendingFrameStartQueueWait) {
		std::fill(row.begin(), row.end(), false);
	}
	for (auto& row : m_pendingFrameStartQueueWaitFenceValue) {
		std::fill(row.begin(), row.end(), UINT64(0));
	}

	m_passBuilderOrder.clear();
	m_passNamesSeenThisReset.clear();
	m_passBuildersByName.clear();
	m_extensions.clear();
    m_extensionRegistrationIds.clear();
	_providerMap.clear();
	_providers.clear();
	_resolverMap.clear();
	_registry = ResourceRegistry();

	m_pCommandRecordingManager.reset();
	m_queueRegistry.Clear();

	initialTransitionCommandAllocator.Reset();
	m_initialTransitionFence.Reset();
	m_frameStartSyncFence.Reset();
	m_readbackFence.Reset();
	m_copyReadbackFence.Reset();

	m_statisticsService.reset();
	m_uploadService.reset();
	m_readbackService.reset();
	m_descriptorService.reset();
	m_renderGraphSettingsService.reset();
}
namespace {
bool HasLiveCompileResourceBacking(Resource* resource) {
	if (!resource) {
		return false;
	}

	if (auto* texture = dynamic_cast<PixelBuffer*>(resource)) {
		return texture->IsMaterialized();
	}
	if (auto* buffer = dynamic_cast<Buffer*>(resource)) {
		return buffer->IsMaterialized();
	}
	return true;
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
}

RenderGraph::FrameCompileResourceState& RenderGraph::GetOrCreateFrameCompileResourceState(size_t resourceIndex, Resource* resource, uint64_t resourceID) {
	if (resourceIndex >= m_frameCompileResources.size()) {
		throw std::runtime_error("Frame compile resource index out of range");
	}

	auto& entry = m_frameCompileResources[resourceIndex];
	entry.resourceID = resourceID;
	if (!entry.resource) {
		if (resource) {
			entry.resource = resource;
		}
		else if (auto shared = GetResourceByID(resourceID)) {
			entry.resource = shared.get();
		}
	}
	if (!entry.trackerInitialized) {
		entry.tracker = SeedCompileTrackerFromLiveResource(entry.resource);
		entry.trackerInitialized = true;
		entry.readOnlyUniformTransitionChecked = false;
		entry.fastState.valid = TryGetWholeResourceTrackerState(entry.tracker, entry.fastState.state);
		entry.fastState.wholeResourceOnly = entry.fastState.valid;
	}
	return entry;
}

void RenderGraph::RebuildFrameCompileResources() {
	ZoneScopedN("RenderGraph::RebuildFrameCompileResources");
	m_frameCompileResources.clear();
	m_frameCompileResources.resize(m_frameSchedulingResourceCount);

	std::vector<uint64_t> preferredDynamicStableIDByIndex(m_frameSchedulingResourceCount, 0);
	for (const auto& [stableID, resource] : m_dynamicResourcesByStableID) {
		if (!resource) {
			continue;
		}
		auto resourceIndex = TryGetFrameSchedulingResourceIndex(stableID);
		if (resourceIndex.has_value() && *resourceIndex < preferredDynamicStableIDByIndex.size()) {
			preferredDynamicStableIDByIndex[*resourceIndex] = stableID;
		}
	}

	for (const auto& [resourceID, resourceIndex] : m_frameSchedulingResourceIndexByID) {
		if (resourceIndex >= m_frameCompileResources.size()) {
			continue;
		}
		if (resourceIndex < preferredDynamicStableIDByIndex.size()
			&& preferredDynamicStableIDByIndex[resourceIndex] != 0
			&& preferredDynamicStableIDByIndex[resourceIndex] != resourceID) {
			continue;
		}

		auto& entry = m_frameCompileResources[resourceIndex];
		entry.resourceID = resourceID;
		if (auto resource = GetResourceByID(resourceID)) {
			entry.resource = resource.get();
		}
		entry.tracker = SeedCompileTrackerFromLiveResource(entry.resource);
		entry.trackerInitialized = true;
		entry.readOnlyUniformTransitionChecked = false;
		entry.fastState.valid = TryGetWholeResourceTrackerState(entry.tracker, entry.fastState.state);
		entry.fastState.wholeResourceOnly = entry.fastState.valid;
		if (resourceIndex < m_frameResourceAccessSummaries.size()) {
			const auto& accessSummary = m_frameResourceAccessSummaries[resourceIndex];
			entry.readOnlyUniformTransitionChecked =
				accessSummary.readOnlyUniform
				&& entry.fastState.valid
				&& entry.fastState.wholeResourceOnly
				&& entry.resource
				&& HasLiveCompileResourceBacking(entry.resource)
				&& StatesExactlyEqual(entry.fastState.state, accessSummary.uniformState);
		}
	}
}

void RenderGraph::CaptureCompileTrackersForExecution(const std::unordered_set<uint64_t>& resourceIDs) {
	ZoneScopedN("RenderGraph::CaptureCompileTrackersForExecution");
	trackers.clear();
	trackers.reserve(resourceIDs.size());

	auto captureTracker = [&](uint64_t resourceID) {
		auto resourceIndex = TryGetFrameSchedulingResourceIndex(resourceID);
		if (!resourceIndex.has_value()) {
			return;
		}

		auto& compileResourceState = GetOrCreateFrameCompileResourceState(*resourceIndex, nullptr, resourceID);
		Resource* resource = compileResourceState.resource;
		if (!resource || !compileResourceState.trackerInitialized || !HasLiveCompileResourceBacking(resource)) {
			return;
		}

		if (auto* tracker = resource->GetStateTracker()) {
			trackers[resourceID] = tracker;
		}
	};

	for (uint64_t resourceID : resourceIDs) {
		captureTracker(resourceID);
	}
}

void RenderGraph::PublishCompiledTrackerStates() {
	for (const auto& [resourceID, liveTracker] : trackers) {
		if (!liveTracker) {
			continue;
		}

		auto resourceIndex = TryGetFrameSchedulingResourceIndex(resourceID);
		if (!resourceIndex.has_value() || *resourceIndex >= m_frameCompileResources.size()) {
			continue;
		}

		const auto& compileResourceState = m_frameCompileResources[*resourceIndex];
		if (!compileResourceState.trackerInitialized) {
			continue;
		}

		liveTracker->CopyFrom(compileResourceState.tracker);
	}
}

void RenderGraph::RebuildSchedulingEquivalentIDCache(const std::unordered_set<uint64_t>& resourceIDs) {
	ZoneScopedN("RenderGraph::RebuildSchedulingEquivalentIDCache");
	m_schedulingEquivalentIDsCache.clear();
	m_schedulingEquivalentIDsCache.reserve(resourceIDs.size());

	for (uint64_t resourceID : resourceIDs) {
		m_schedulingEquivalentIDsCache.emplace(resourceID, BuildSchedulingEquivalentIDs(resourceID));
	}
}

const std::vector<uint64_t>& RenderGraph::GetSchedulingEquivalentIDsCached(uint64_t resourceID) {
	auto it = m_schedulingEquivalentIDsCache.find(resourceID);
	if (it != m_schedulingEquivalentIDsCache.end()) {
		return it->second;
	}

	auto [insertedIt, inserted] = m_schedulingEquivalentIDsCache.emplace(resourceID, BuildSchedulingEquivalentIDs(resourceID));
	(void)inserted;
	return insertedIt->second;
}

void RenderGraph::ClearFrameSchedulingResourceIndex() {
	m_frameSchedulingResourceIndexByID.clear();
	m_frameSchedulingResourceCount = 0;
	m_frameCompileResources.clear();
	m_aliasPlacementRangeByResourceIndex.clear();
	m_hasAliasPlacementByResourceIndex.clear();
	m_schedulingPlacementRangeByResourceIndex.clear();
	m_hasSchedulingPlacementByResourceIndex.clear();
	m_aliasActivationPendingByResourceIndex.clear();
	m_frameQueueLastUsageBatch.clear();
	m_frameQueueLastProducerBatch.clear();
	m_frameQueueLastTransitionBatch.clear();
	m_frameResourceEventSummaries.clear();
}

void RenderGraph::ClearFramePassSchedulingSummaries() {
	m_framePassSchedulingSummaries.clear();
	m_frameResourceAccessSummaries.clear();
}

void RenderGraph::ResetFrameQueueBatchHistoryTables() {
	const size_t entryCount = m_queueRegistry.SlotCount() * m_frameSchedulingResourceCount;
	m_frameQueueLastUsageBatch.assign(entryCount, 0);
	m_frameQueueLastProducerBatch.assign(entryCount, 0);
	m_frameQueueLastTransitionBatch.assign(entryCount, 0);
	if (m_queueRegistry.SlotCount() <= 64) {
		m_frameResourceEventSummaries.assign(m_frameSchedulingResourceCount, FrameResourceEventSummary{});
	}
	else {
		m_frameResourceEventSummaries.clear();
	}
}

void RenderGraph::RebuildFrameSchedulingResourceIndex(const std::unordered_set<uint64_t>& resourceIDs) {
	ZoneScopedN("RenderGraph::RebuildFrameSchedulingResourceIndex");
	m_frameSchedulingResourceIndexByID.clear();
	m_frameSchedulingResourceIndexByID.reserve(resourceIDs.size() * 2);
	m_frameSchedulingResourceCount = 0;

	auto registerResourceID = [&](uint64_t resourceID) {
		auto [_, inserted] = m_frameSchedulingResourceIndexByID.emplace(resourceID, m_frameSchedulingResourceCount);
		if (inserted) {
			++m_frameSchedulingResourceCount;
		}
	};

	for (uint64_t resourceID : resourceIDs) {
		registerResourceID(resourceID);
		for (uint64_t equivalentID : GetSchedulingEquivalentIDsCached(resourceID)) {
			registerResourceID(equivalentID);
		}
	}

	for (const auto& [stableID, resource] : m_dynamicResourcesByStableID) {
		if (!resource) {
			continue;
		}
		const uint64_t backingID = resource->GetGlobalResourceID();
		auto stableIt = m_frameSchedulingResourceIndexByID.find(stableID);
		auto backingIt = m_frameSchedulingResourceIndexByID.find(backingID);
		if (stableIt != m_frameSchedulingResourceIndexByID.end()) {
			m_frameSchedulingResourceIndexByID[backingID] = stableIt->second;
		}
		else if (backingIt != m_frameSchedulingResourceIndexByID.end()) {
			m_frameSchedulingResourceIndexByID[stableID] = backingIt->second;
		}
	}

	RebuildEquivalentResourceIndicesByResourceIndex();

	m_aliasActivationPendingByResourceIndex.assign(m_frameSchedulingResourceCount, 0);
	for (uint64_t resourceID : aliasActivationPending) {
		auto resourceIndex = TryGetFrameSchedulingResourceIndex(resourceID);
		if (resourceIndex.has_value()) {
			m_aliasActivationPendingByResourceIndex[*resourceIndex] = 1;
		}
	}

	ResetFrameQueueBatchHistoryTables();
}

void RenderGraph::RebuildEquivalentResourceIndicesByResourceIndex() {
	ZoneScopedN("RenderGraph::RebuildEquivalentResourceIndicesByResourceIndex");
	m_equivalentResourceIndicesByResourceIndex.clear();
	m_equivalentResourceIndicesByResourceIndex.resize(m_frameSchedulingResourceCount);

	for (const auto& [resourceID, resourceIndex] : m_frameSchedulingResourceIndexByID) {
		if (resourceIndex >= m_equivalentResourceIndicesByResourceIndex.size()) {
			continue;
		}

		auto& equivalentIndices = m_equivalentResourceIndicesByResourceIndex[resourceIndex];
		for (uint64_t equivalentID : GetSchedulingEquivalentIDsCached(resourceID)) {
			if (equivalentID == resourceID) {
				continue;
			}

			auto equivalentIt = m_frameSchedulingResourceIndexByID.find(equivalentID);
			if (equivalentIt != m_frameSchedulingResourceIndexByID.end()) {
				equivalentIndices.push_back(equivalentIt->second);
			}
		}

		std::sort(equivalentIndices.begin(), equivalentIndices.end());
		equivalentIndices.erase(std::unique(equivalentIndices.begin(), equivalentIndices.end()), equivalentIndices.end());
	}
}

void RenderGraph::RebuildFramePassSchedulingSummaries() {
	ZoneScopedN("RenderGraph::RebuildFramePassSchedulingSummaries");
	m_framePassSchedulingSummaries.clear();
	m_framePassSchedulingSummaries.resize(m_framePassAccessSummaries.size());
	m_frameResourceAccessSummaries.assign(m_frameSchedulingResourceCount, FrameResourceAccessSummary{});

	for (size_t passIndex = 0; passIndex < m_framePassAccessSummaries.size(); ++passIndex) {
		auto& summary = m_framePassSchedulingSummaries[passIndex];
		const auto& passAccess = m_framePassAccessSummaries[passIndex];
		summary.requirements.reserve(passAccess.requirementSummaries.size());
		summary.internalTransitions.reserve(passAccess.internalTransitionSummaries.size());
		summary.requiredResourceIndices.reserve(passAccess.requirementSummaries.size());
		summary.touchedResourceIndices.reserve(passAccess.requirementSummaries.size() + passAccess.internalTransitionSummaries.size());
		summary.uavResourceIndices.reserve(passAccess.requirementSummaries.size());

		for (const auto& req : passAccess.requirementSummaries) {
			auto resourceIndex = TryGetFrameSchedulingResourceIndex(req.resourceID);
			if (!resourceIndex.has_value()) {
				continue;
			}

			auto& accessSummary = m_frameResourceAccessSummaries[*resourceIndex];
			accessSummary.hasWrite = accessSummary.hasWrite || req.isWrite;
			accessSummary.hasUAV = accessSummary.hasUAV || req.isUAV;
			accessSummary.hasAliasActivation = accessSummary.hasAliasActivation
				|| (*resourceIndex < m_aliasActivationPendingByResourceIndex.size() && m_aliasActivationPendingByResourceIndex[*resourceIndex] != 0);
			accessSummary.hasNonWholeResourceRange = accessSummary.hasNonWholeResourceRange
				|| !IsWholeResourceRange(req.range, req.resource);

			DenseRequirementSummary denseRequirement{};
			denseRequirement.resource = req.resource;
			denseRequirement.resourceID = req.resourceID;
			denseRequirement.resourceIndex = *resourceIndex;
			denseRequirement.range = req.range;
			denseRequirement.state = req.state;
			denseRequirement.isUAV = req.isUAV;
			if (*resourceIndex < m_equivalentResourceIndicesByResourceIndex.size()) {
				denseRequirement.equivalentResourceIndices = &m_equivalentResourceIndicesByResourceIndex[*resourceIndex];
			}
			summary.requirements.push_back(std::move(denseRequirement));
			summary.requiredResourceIndices.push_back(*resourceIndex);
			summary.touchedResourceIndices.push_back(*resourceIndex);
			if (req.isUAV) {
				summary.uavResourceIndices.push_back(*resourceIndex);
			}
		}

		for (const auto& transition : passAccess.internalTransitionSummaries) {
			auto resourceIndex = TryGetFrameSchedulingResourceIndex(transition.resourceID);
			if (!resourceIndex.has_value()) {
				continue;
			}

			m_frameResourceAccessSummaries[*resourceIndex].hasInternalTransition = true;

			DenseEquivalentResourceSummary denseTransition{};
			denseTransition.resourceID = transition.resourceID;
			denseTransition.resourceIndex = *resourceIndex;
			if (*resourceIndex < m_equivalentResourceIndicesByResourceIndex.size()) {
				denseTransition.equivalentResourceIndices = &m_equivalentResourceIndicesByResourceIndex[*resourceIndex];
			}
			summary.internalTransitions.push_back(std::move(denseTransition));
			summary.touchedResourceIndices.push_back(*resourceIndex);
		}

		std::sort(summary.requiredResourceIndices.begin(), summary.requiredResourceIndices.end());
		summary.requiredResourceIndices.erase(
			std::unique(summary.requiredResourceIndices.begin(), summary.requiredResourceIndices.end()),
			summary.requiredResourceIndices.end());

		std::sort(summary.touchedResourceIndices.begin(), summary.touchedResourceIndices.end());
		summary.touchedResourceIndices.erase(
			std::unique(summary.touchedResourceIndices.begin(), summary.touchedResourceIndices.end()),
			summary.touchedResourceIndices.end());

		std::sort(summary.uavResourceIndices.begin(), summary.uavResourceIndices.end());
		summary.uavResourceIndices.erase(
			std::unique(summary.uavResourceIndices.begin(), summary.uavResourceIndices.end()),
			summary.uavResourceIndices.end());
	}
}

void RenderGraph::RebuildFrameResourceAccessSummaries(const std::vector<Node>& nodes) {
	ZoneScopedN("RenderGraph::RebuildFrameResourceAccessSummaries");
	if (m_frameResourceAccessSummaries.size() != m_frameSchedulingResourceCount) {
		m_frameResourceAccessSummaries.assign(m_frameSchedulingResourceCount, FrameResourceAccessSummary{});
	}

	for (auto& accessSummary : m_frameResourceAccessSummaries) {
		accessSummary.hasMultipleRequiredStates = false;
		accessSummary.readOnlyUniform = false;
		accessSummary.uniformStateInitialized = false;
	}

	for (size_t passIndex = 0; passIndex < m_framePassSchedulingSummaries.size() && passIndex < nodes.size(); ++passIndex) {
		const auto& passSummary = m_framePassSchedulingSummaries[passIndex];
		const auto& node = nodes[passIndex];

		std::vector<size_t> activeCompatibleSlots;
		activeCompatibleSlots.reserve(node.compatibleQueueSlots.size() + 1);
		for (size_t queueSlot : node.compatibleQueueSlots) {
			if (queueSlot < m_activeQueueSlotsThisFrame.size() && m_activeQueueSlotsThisFrame[queueSlot]) {
				activeCompatibleSlots.push_back(queueSlot);
			}
		}
		if (activeCompatibleSlots.empty()) {
			activeCompatibleSlots.push_back(node.queueSlot);
		}
		std::sort(activeCompatibleSlots.begin(), activeCompatibleSlots.end());
		activeCompatibleSlots.erase(std::unique(activeCompatibleSlots.begin(), activeCompatibleSlots.end()), activeCompatibleSlots.end());

		for (const auto& requirement : passSummary.requirements) {
			if (requirement.resourceIndex >= m_frameResourceAccessSummaries.size()) {
				continue;
			}

			auto& accessSummary = m_frameResourceAccessSummaries[requirement.resourceIndex];
			for (size_t queueSlot : activeCompatibleSlots) {
				const QueueKind queueKind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(queueSlot)));
				const ResourceState normalizedState = NormalizeStateForQueue(queueKind, requirement.state);
				if (!accessSummary.uniformStateInitialized) {
					accessSummary.uniformState = normalizedState;
					accessSummary.uniformStateInitialized = true;
				}
				else if (!StatesExactlyEqual(accessSummary.uniformState, normalizedState)) {
					accessSummary.hasMultipleRequiredStates = true;
				}
			}
		}
	}

	for (auto& accessSummary : m_frameResourceAccessSummaries) {
		accessSummary.readOnlyUniform =
			!accessSummary.hasWrite &&
			!accessSummary.hasUAV &&
			!accessSummary.hasInternalTransition &&
			!accessSummary.hasAliasActivation &&
			!accessSummary.hasNonWholeResourceRange &&
			!accessSummary.hasMultipleRequiredStates &&
			accessSummary.uniformStateInitialized;
	}
}

bool RenderGraph::ValidateSchedulingDecisionTrace(
	const std::vector<Node>& nodes,
	const std::vector<AnyPassAndResources>& framePasses,
	const std::vector<PassBatch>& compiledBatches,
	std::string& outSummary) const
{
	size_t invalidEntries = 0;
	size_t missingBatchMembership = 0;
	size_t duplicatePasses = 0;
	std::vector<uint8_t> seen(framePasses.size(), 0);

	auto passPointerForIndex = [&](size_t passIndex) -> const void* {
		if (passIndex >= framePasses.size()) {
			return nullptr;
		}

		return std::visit(
			[](const auto& passEntry) -> const void* {
				using T = std::decay_t<decltype(passEntry)>;
				if constexpr (std::is_same_v<T, std::monostate>) {
					return nullptr;
				}
				else {
					return static_cast<const void*>(&passEntry);
				}
			},
			framePasses[passIndex].pass);
	};

	for (const auto& trace : m_schedulingDecisionTrace) {
		if (trace.nodeIndex >= nodes.size()
			|| trace.passIndex >= framePasses.size()
			|| trace.batchIndex >= compiledBatches.size()
			|| trace.assignedQueueSlot >= compiledBatches[trace.batchIndex].QueueCount()) {
			++invalidEntries;
			continue;
		}

		if (seen[trace.passIndex] != 0) {
			++duplicatePasses;
		}
		seen[trace.passIndex] = 1;

		const void* expectedPassPointer = passPointerForIndex(trace.passIndex);
		bool foundInBatch = false;
		for (const auto& queuedPass : compiledBatches[trace.batchIndex].Passes(trace.assignedQueueSlot)) {
			std::visit(
				[&](const auto* passEntry) {
					if (static_cast<const void*>(passEntry) == expectedPassPointer) {
						foundInBatch = true;
					}
				},
				queuedPass);
			if (foundInBatch) {
				break;
			}
		}

		if (!foundInBatch) {
			++missingBatchMembership;
		}
	}

	size_t missingPasses = 0;
	for (uint8_t value : seen) {
		if (value == 0) {
			++missingPasses;
		}
	}

	std::ostringstream summary;
	summary << "trace_entries=" << m_schedulingDecisionTrace.size()
		<< " passes=" << framePasses.size()
		<< " invalid=" << invalidEntries
		<< " duplicate_passes=" << duplicatePasses
		<< " missing_passes=" << missingPasses
		<< " missing_batch_membership=" << missingBatchMembership;
	outSummary = summary.str();

	return m_schedulingDecisionTrace.size() == framePasses.size()
		&& invalidEntries == 0
		&& duplicatePasses == 0
		&& missingPasses == 0
		&& missingBatchMembership == 0;
}

void RenderGraph::ExtractScheduleRegionsFromAuthoritativeCompile(
	const std::vector<Node>& nodes,
	const std::vector<AnyPassAndResources>& framePasses,
	const std::vector<PassBatch>& compiledBatches,
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

	size_t start = 0;
	while (start < m_schedulingDecisionTrace.size()) {
		const uint16_t queueSlot = m_schedulingDecisionTrace[start].assignedQueueSlot;
		RegionRejectReason startSplitReason = RegionRejectReason::Count;
		const bool startForcesSplit = passForcesCandidateSplit(m_schedulingDecisionTrace[start].passIndex, startSplitReason);
		size_t end = start + 1;
		if (!startForcesSplit) {
			while (end < m_schedulingDecisionTrace.size()
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

void RenderGraph::ExtractReplaySegmentsFromAuthoritativeCompile(
	const std::vector<Node>& nodes,
	const std::vector<AnyPassAndResources>& framePasses,
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
		++bestVariant->hitCount;
		result.variant = bestVariant;
		result.syncShapeDiverged = ReplaySegmentSyncShapeDiverged(bestVariant->segment, currentSegment);
		result.variantAgeFrames = frameIndex >= bestVariant->lastSeenFrame ? frameIndex - bestVariant->lastSeenFrame : 0;
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
	const std::vector<AnyPassAndResources>& framePasses,
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
	std::vector<AnyPassAndResources>& framePasses,
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

	std::unordered_set<uint64_t> scratchTransitioned;
	std::unordered_set<size_t> scratchFallback;
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
		compileResourceState.tracker.Apply(range, resource, state, ignoredTransitions);
		if (resource && isExplicitWholeResourceRange(range)) {
			compileResourceState.fastState.valid = true;
			compileResourceState.fastState.wholeResourceOnly = true;
			compileResourceState.fastState.state = state;
		}
		else {
			compileResourceState.fastState.valid = false;
			compileResourceState.fastState.wholeResourceOnly = false;
		}
		return &compileResourceState.tracker;
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
		scratchTransitioned.clear();
		scratchFallback.clear();
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
				inputBatch.passBatchTrackersByResourceIndex[*resourceIndex] = &compileResourceState.tracker;
				continue;
			}
			if (!compileResourceState.tracker.WouldModify(input.range, requiredState)) {
				inputBatch.passBatchTrackersByResourceIndex[*resourceIndex] = &compileResourceState.tracker;
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
					inputBatch.passBatchTrackersByResourceIndex[*resourceIndex] = &compileResourceState.tracker;
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
					if (TryGetWholeResourceTrackerState(compileResourceState.tracker, liveBeforeState)) {
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
					compileResourceState.tracker.Apply(transitionTemplate.range, resource, transitionTemplate.after, ignoredTransitions);
				}
				else {
					scratchTransitions.clear();
					compileResourceState.tracker.Apply(transitionTemplate.range, resource, transitionTemplate.after, scratchTransitions);
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
	std::vector<std::vector<uint32_t>> segmentNodesByFirstNode(nodes.size());
	std::unordered_map<uint32_t, uint32_t> segmentFirstNodeByNode;
	segmentFirstNodeByNode.reserve(nodes.size());
	for (const auto& segment : replaySegments) {
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
		for (size_t resourceIndex : passSummary.requiredResourceIndices) {
			batchBuildState.MarkResource(resourceIndex);
		}
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
			if (!compileResourceState.trackerInitialized) {
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
			if (compileResourceState.tracker.WouldModify(input.range, requiredState)) {
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
	return report;
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
		if (!compileResource.trackerInitialized) {
			continue;
		}
		++initializedTrackerCount;
		const auto& segments = compileResource.tracker.GetSegments();
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

std::optional<size_t> RenderGraph::TryGetFrameSchedulingResourceIndex(uint64_t resourceID) const {
	auto it = m_frameSchedulingResourceIndexByID.find(resourceID);
	if (it == m_frameSchedulingResourceIndexByID.end()) {
		return std::nullopt;
	}
	return it->second;
}

const rg::alias::AliasPlacementRange* RenderGraph::TryGetAliasPlacementRangeByResourceIndex(size_t resourceIndex) const {
	if (resourceIndex >= m_hasAliasPlacementByResourceIndex.size() || resourceIndex >= m_aliasPlacementRangeByResourceIndex.size()) {
		return nullptr;
	}
	if (m_hasAliasPlacementByResourceIndex[resourceIndex] == 0) {
		return nullptr;
	}
	return &m_aliasPlacementRangeByResourceIndex[resourceIndex];
}

const rg::alias::AliasPlacementRange* RenderGraph::TryGetAliasPlacementRange(uint64_t resourceID) const {
	auto resourceIndex = TryGetFrameSchedulingResourceIndex(resourceID);
	if (!resourceIndex.has_value()) {
		return nullptr;
	}
	return TryGetAliasPlacementRangeByResourceIndex(*resourceIndex);
}

const rg::alias::AliasPlacementRange* RenderGraph::TryGetSchedulingPlacementRangeByResourceIndex(size_t resourceIndex) const {
	if (resourceIndex >= m_hasSchedulingPlacementByResourceIndex.size() || resourceIndex >= m_schedulingPlacementRangeByResourceIndex.size()) {
		return nullptr;
	}
	if (m_hasSchedulingPlacementByResourceIndex[resourceIndex] == 0) {
		return nullptr;
	}
	return &m_schedulingPlacementRangeByResourceIndex[resourceIndex];
}

const rg::alias::AliasPlacementRange* RenderGraph::TryGetSchedulingPlacementRange(uint64_t resourceID) const {
	auto resourceIndex = TryGetFrameSchedulingResourceIndex(resourceID);
	if (!resourceIndex.has_value()) {
		return nullptr;
	}
	return TryGetSchedulingPlacementRangeByResourceIndex(*resourceIndex);
}

std::vector<uint64_t> RenderGraph::BuildSchedulingEquivalentIDs(uint64_t resourceID) const {
	const auto* placement = TryGetSchedulingPlacementRange(resourceID);
	if (!placement) {
		return { resourceID };
	}

	std::vector<uint64_t> out;
	out.reserve(8);
	for (const auto& [candidateID, candidateIndex] : m_frameSchedulingResourceIndexByID) {
		const auto* otherPlacement = TryGetSchedulingPlacementRangeByResourceIndex(candidateIndex);
		if (!otherPlacement || otherPlacement->poolID != placement->poolID) {
			continue;
		}

		const uint64_t overlapStart = (std::max)(placement->startByte, otherPlacement->startByte);
		const uint64_t overlapEnd = (std::min)(placement->endByte, otherPlacement->endByte);
		if (overlapStart < overlapEnd) {
			out.push_back(candidateID);
		}
	}

	if (out.empty()) {
		out.push_back(resourceID);
	}

	return out;
}

size_t RenderGraph::FrameQueueBatchHistoryOffset(size_t queueSlot, size_t resourceIndex) const {
	return queueSlot * m_frameSchedulingResourceCount + resourceIndex;
}

unsigned int RenderGraph::GetFrameQueueHistoryValue(const std::vector<unsigned int>& history, size_t queueSlot, size_t resourceIndex) const {
	if (m_frameSchedulingResourceCount == 0) {
		return 0;
	}
	const size_t offset = FrameQueueBatchHistoryOffset(queueSlot, resourceIndex);
	if (offset >= history.size()) {
		return 0;
	}
	return history[offset];
}

void RenderGraph::SetFrameQueueHistoryValue(std::vector<unsigned int>& history, size_t queueSlot, size_t resourceIndex, unsigned int batchIndex) {
	if (m_frameSchedulingResourceCount == 0) {
		return;
	}
	const size_t offset = FrameQueueBatchHistoryOffset(queueSlot, resourceIndex);
	if (offset >= history.size()) {
		return;
	}
	history[offset] = batchIndex;
}

void RenderGraph::RecordFrameResourceEvent(size_t queueSlot, size_t resourceIndex, unsigned int batchIndex) {
	if (batchIndex == 0 || queueSlot >= 64 || resourceIndex >= m_frameResourceEventSummaries.size()) {
		return;
	}

	auto& summary = m_frameResourceEventSummaries[resourceIndex];
	const uint64_t queueMask = uint64_t{ 1 } << queueSlot;
	if (batchIndex == summary.latestBatch) {
		summary.latestQueueMask |= queueMask;
		return;
	}
	if (batchIndex > summary.latestBatch) {
		summary.previousBatch = summary.latestBatch;
		summary.previousQueueMask = summary.latestQueueMask;
		summary.latestBatch = batchIndex;
		summary.latestQueueMask = queueMask;
		return;
	}
	if (batchIndex == summary.previousBatch) {
		summary.previousQueueMask |= queueMask;
		return;
	}
	if (batchIndex > summary.previousBatch) {
		summary.previousBatch = batchIndex;
		summary.previousQueueMask = queueMask;
	}
}

void RenderGraph::RecordFrameQueueUsageBatch(size_t queueSlot, size_t resourceIndex, unsigned int batchIndex) {
	SetFrameQueueHistoryValue(m_frameQueueLastUsageBatch, queueSlot, resourceIndex, batchIndex);
	RecordFrameResourceEvent(queueSlot, resourceIndex, batchIndex);
}

void RenderGraph::RecordFrameQueueTransitionBatch(size_t queueSlot, size_t resourceIndex, unsigned int batchIndex) {
	SetFrameQueueHistoryValue(m_frameQueueLastTransitionBatch, queueSlot, resourceIndex, batchIndex);
	RecordFrameResourceEvent(queueSlot, resourceIndex, batchIndex);
}

std::pair<unsigned int, uint64_t> RenderGraph::GetFrameResourceLastEventBeforeBatch(size_t resourceIndex, unsigned int batchIndex) const {
	if (resourceIndex >= m_frameResourceEventSummaries.size() || batchIndex == 0) {
		return { 0u, 0u };
	}

	const auto& summary = m_frameResourceEventSummaries[resourceIndex];
	if (summary.latestBatch > 0 && summary.latestBatch < batchIndex) {
		return { summary.latestBatch, summary.latestQueueMask };
	}
	if (summary.previousBatch > 0 && summary.previousBatch < batchIndex) {
		return { summary.previousBatch, summary.previousQueueMask };
	}
	return { 0u, 0u };
}

void RenderGraph::MaterializeReferencedResources(
	const std::vector<ResourceRequirement>& resourceRequirements,
	const std::vector<std::pair<ResourceHandleAndRange, ResourceState>>& internalTransitions,
	std::string_view debugPassName)
{
	auto materializeIfNeeded = [&](const ResourceRegistry::RegistryHandle& handle) {
		if (handle.IsEphemeral()) {
			return;
		}

		auto resource = _registry.Resolve(handle);
		if (!resource) {
			return;
		}

		auto texture = dynamic_cast<PixelBuffer*>(resource);
		if (texture) {
			if (texture->IsMaterialized()) {
				return;
			}

			if (texture->GetDescription().allowAlias) {
				// Alias placement is frame-dependent and produced later in CompileFrame.
				// Defer materialization until RenderGraph::MaterializeUnmaterializedResources,
				// after BuildAliasPlanAfterDag has produced placement options.
				return;
			}

			texture->Materialize();
			resourceBackingGenerationByID[handle.GetGlobalResourceID()] = texture->GetBackingGeneration();
			if (m_structuralMaterializeResourceCheckpointCallback && !debugPassName.empty()) {
				m_structuralMaterializeResourceCheckpointCallback(debugPassName, texture->GetName());
			}
			return;
		}

		auto buffer = dynamic_cast<Buffer*>(resource);
		if (buffer) {
			if (buffer->IsMaterialized()) {
				return;
			}

			if (buffer->IsAliasingAllowed()) {
				return;
			}

			buffer->Materialize();
			resourceBackingGenerationByID[handle.GetGlobalResourceID()] = buffer->GetBackingGeneration();
			if (m_structuralMaterializeResourceCheckpointCallback && !debugPassName.empty()) {
				m_structuralMaterializeResourceCheckpointCallback(debugPassName, buffer->GetName());
			}
		}
	};

	for (auto const& req : resourceRequirements) {
		materializeIfNeeded(req.resourceHandleAndRange.resource);
	}

	for (auto const& transition : internalTransitions) {
		materializeIfNeeded(transition.first.resource);
	}
}

std::vector<std::shared_ptr<Resource>> RenderGraph::CaptureRetainedAnonymousKeepAlive(
	const std::vector<ResourceRequirement>& resourceRequirements,
	const std::vector<std::pair<ResourceHandleAndRange, ResourceState>>& internalTransitions) const
{
	std::vector<std::shared_ptr<Resource>> keepAlive;
	std::unordered_set<uint64_t> seenResourceIDs;
	seenResourceIDs.reserve(resourceRequirements.size() + internalTransitions.size());

	auto maybeCapture = [&](const ResourceRegistry::RegistryHandle& handle) {
		if (handle.IsEphemeral() || !_registry.IsAnonymous(handle)) {
			return;
		}

		const Resource* resource = _registry.Resolve(handle);
		if (!resource) {
			return;
		}

		auto shared = std::const_pointer_cast<Resource>(resource->weak_from_this().lock());
		if (!shared) {
			return;
		}

		if (!seenResourceIDs.insert(shared->GetGlobalResourceID()).second) {
			return;
		}

		keepAlive.push_back(std::move(shared));
	};

	for (const auto& req : resourceRequirements) {
		maybeCapture(req.resourceHandleAndRange.resource);
	}

	for (const auto& transition : internalTransitions) {
		maybeCapture(transition.first.resource);
	}

	return keepAlive;
}

void RenderGraph::CollectFrameResourceIDs(std::unordered_set<uint64_t>& used) const {
	ZoneScopedN("RenderGraph::CollectFrameResourceIDs");
	used.clear();
	used.reserve(m_framePasses.size() * 4);

	auto insertHandleResourceIDs = [&](const ResourceRegistry::RegistryHandle& handle) {
		used.insert(handle.GetGlobalResourceID());
		Resource* resource = handle.IsEphemeral()
			? handle.GetEphemeralPtr()
			: const_cast<Resource*>(_registry.Resolve(handle));
		if (auto* dynamicResource = dynamic_cast<DynamicResource*>(resource)) {
			used.insert(dynamicResource->GetDynamicWrapperGlobalResourceID());
			used.insert(dynamicResource->GetGlobalResourceID());
			if (auto backing = dynamicResource->GetResource()) {
				used.insert(backing->GetGlobalResourceID());
			}
		}
	};

	for (auto const& pr : m_framePasses) {
		std::visit([&](auto const& passAndResources) {
			using T = std::decay_t<decltype(passAndResources)>;
			if constexpr (!std::is_same_v<T, std::monostate>) {
				ForEachFrameRequirement(passAndResources.resources, [&](const auto& req) {
					insertHandleResourceIDs(req.resourceHandleAndRange.resource);
				});
				for (auto const& t : passAndResources.resources.internalTransitions) {
					insertHandleResourceIDs(t.first.resource);
				}
			}
		}, pr.pass);
	}
}

void RenderGraph::ApplyIdleDematerializationPolicy(const std::unordered_set<uint64_t>& usedResourceIDs) {
	ZoneScopedN("RenderGraph::ApplyIdleDematerializationPolicy");
	for (auto& [id, resource] : resourcesByID) {
		if (!resource) {
			continue;
		}

		auto texture = std::dynamic_pointer_cast<PixelBuffer>(resource);
		if (!texture || !texture->IsIdleDematerializationEnabled()) {
			continue;
		}

		if (usedResourceIDs.find(id) != usedResourceIDs.end()) {
			resourceIdleFrameCounts[id] = 0;
			continue;
		}

		auto& idleFrames = resourceIdleFrameCounts[id];
		idleFrames++;

		if (texture->IsMaterialized() && idleFrames >= texture->GetIdleDematerializationThreshold()) {
			texture->Dematerialize();
			resourceBackingGenerationByID[id] = texture->GetBackingGeneration();
		}
	}
}

void RenderGraph::SnapshotCompiledResourceGenerations(const std::unordered_set<uint64_t>& usedResourceIDs) {
	compiledResourceGenerationByID.clear();
	compiledResourceGenerationByID.reserve(usedResourceIDs.size());

	for (uint64_t id : usedResourceIDs) {
		auto it = resourcesByID.find(id);
		if (it == resourcesByID.end() || !it->second) {
			continue;
		}

		auto texture = std::dynamic_pointer_cast<PixelBuffer>(it->second);
		if (texture) {
			compiledResourceGenerationByID[id] = texture->GetBackingGeneration();
			continue;
		}

		auto buffer = std::dynamic_pointer_cast<Buffer>(it->second);
		if (buffer) {
			compiledResourceGenerationByID[id] = buffer->GetBackingGeneration();
		}
	}
}

void RenderGraph::ValidateCompiledResourceGenerations() const {
	for (auto const& [id, compiledGeneration] : compiledResourceGenerationByID) {
		auto it = resourcesByID.find(id);
		if (it == resourcesByID.end() || !it->second) {
			continue;
		}

		auto texture = std::dynamic_pointer_cast<PixelBuffer>(it->second);
		if (texture) {
			const uint64_t currentGeneration = texture->GetBackingGeneration();
			if (currentGeneration != compiledGeneration) {
				throw std::runtime_error("Resource backing generation changed after compile and before execute. Resource ID: " + std::to_string(id));
			}
			continue;
		}

		auto buffer = std::dynamic_pointer_cast<Buffer>(it->second);
		if (buffer) {
			const uint64_t currentGeneration = buffer->GetBackingGeneration();
			if (currentGeneration != compiledGeneration) {
				throw std::runtime_error("Resource backing generation changed after compile and before execute. Resource ID: " + std::to_string(id));
			}
		}
	}
}


void RenderGraph::RegisterExtension(std::unique_ptr<IRenderGraphExtension> ext, std::optional<std::string_view> id) {
	if (!ext) return;

    auto* extPtr = ext.get();
	const auto& incomingType = typeid(*extPtr);
    const std::string registrationId = id.has_value()
        ? std::string(*id)
        : std::string(incomingType.name());

	for (const auto& existingId : m_extensionRegistrationIds) {
		if (existingId == registrationId) {
			spdlog::error("Duplicate RenderGraph extension registration: {}", registrationId);
			throw std::runtime_error("Duplicate RenderGraph extension registration");
		}
	}

	// Let the extension see the current registry immediately.
	ext->OnRegistryReset(&_registry);
	m_extensions.push_back(std::move(ext));
    try {
        m_extensionRegistrationIds.push_back(registrationId);
    }
    catch (...) {
        m_extensions.pop_back();
        throw;
    }
}

void RenderGraph::PrepareExtensionsForBuild() {
	for (auto& ext : m_extensions) {
		if (ext) {
			ext->PrepareForBuild(*this);
		}
	}
}

void RenderGraph::ResetForRebuild()
{
	if (m_pCommandRecordingManager) {
		m_pCommandRecordingManager->ShutdownThreadLocal();
		m_pCommandRecordingManager.reset();
	}

	++m_regionCache.structuralGeneration;
	m_lastAuthoritativeReplayAttempted = false;
	m_lastAuthoritativeReplaySucceeded = false;
	m_lastAuthoritativeReplaySegments = 0;
	m_lastAuthoritativeReplayPasses = 0;
	m_lastAuthoritativeReplayDynamicGapPasses = 0;
	m_lastAuthoritativeReplayFailure.clear();
	m_lastAuthoritativeReplayRecomputeReason.clear();

	// Clear any existing compile state
	m_masterPassList.clear();
	m_framePasses.clear();
	trackers.clear();
	ResetCompileFrameState();
	ResetStructuralBuildState();

	// Clear resources
	resourcesByID.clear();
	resourcesByName.clear();
	m_transientFrameResourcesByID.clear();
	m_dynamicResourcesByStableID.clear();
	m_transientFrameResourcesByName.clear();
	resourceBackingGenerationByID.clear();
	resourceIdleFrameCounts.clear();
	m_aliasingSubsystem.ResetPersistentState(*this);
	m_lastProducerByResourceAcrossFrames.clear();
	m_lastAliasPlacementProducersByPoolAcrossFrames.clear();
	m_compiledLastProducerBatchByResourceByQueue.clear();
	m_hasPendingFrameStartQueueWait.clear();
	m_pendingFrameStartQueueWaitFenceValue.clear();
	m_queueRegistry.Clear();
	renderPassesByName.clear();
	computePassesByName.clear();

	// Clear providers
	_providerMap.clear();
	_providers.clear();
	_resolverMap.clear();
	_registry = ResourceRegistry();

	// Notify extensions that the registry was replaced
	for (auto& ext : m_extensions) {
		if (ext) ext->OnRegistryReset(&_registry);
	}
}

void RenderGraph::ResetCompileFrameState() {
	batches.clear();
	compiledResourceGenerationByID.clear();
	m_transientFrameResourcesByID.clear();
	m_transientFrameResourcesByName.clear();
	m_frameCompileResources.clear();
	m_addTransitionDebugStatsByResource.clear();
	m_schedulingEquivalentIDsCache.clear();
	ClearFrameSchedulingResourceIndex();
	ClearFramePassSchedulingSummaries();
	m_assignedQueueSlotsByFramePass.clear();
	m_activeQueueSlotsThisFrame.clear();
	m_executionSchedule.Reset();
	for (auto& producerMap : m_compiledLastProducerBatchByResourceByQueue) {
		producerMap.clear();
	}
	for (auto& row : m_hasPendingFrameStartQueueWait) {
		std::fill(row.begin(), row.end(), false);
	}
	for (auto& row : m_pendingFrameStartQueueWaitFenceValue) {
		std::fill(row.begin(), row.end(), UINT64(0));
	}
}

void RenderGraph::ResetStructuralBuildState() {
	// Full rebuilds must drop cached pass instances before clearing resources.
	// Builders reuse pass objects across frames, and many passes capture
	// resource-owning shared_ptrs through constructor arguments.
	for (auto& [name, builder] : m_passBuildersByName) {
		(void)name;
		if (builder) {
			builder->Reset();
		}
	}

	m_passBuilderOrder.clear();
	m_passNamesSeenThisReset.clear();
}

void RenderGraph::ResetForFrame() {
	ZoneScopedN("RenderGraph::ResetForFrame");
	_registry.ReclaimExpiredAnonymous();
	m_aliasingSubsystem.ResetPerFrameState(*this);
	ResetCompileFrameState();
}

void RenderGraph::CompileStructural() {
	// Register resource providers from pass builders

	std::vector<unsigned int> empty;
	// Go backwards to build skip list
	for (int i = static_cast<int>(m_passBuilderOrder.size()) - 1; i >= 0; i--) {
		auto ptr = m_passBuilderOrder[i];
		auto prov = ptr->ResourceProvider();
		if (!prov) {
			empty.push_back(i); // This pass was not built
			continue;
		}
		EnsureProviderRegistered(prov);
	}
	unsigned int i = 0;
	for (auto ptr : m_passBuilderOrder) {
		if (!empty.empty() && empty.back() == i) {
			empty.pop_back();
			continue;
		}
		ptr->Finalize();
		i++;
	}

	batches.clear();

	struct Pending {
		AnyPassAndResources pr;

		std::string anchorKey;     // internal unique key used for anchoring/emission

		ExternalInsertPoint where; // concrete
		bool chained = false;      // true if where was synthesized from "natural ordering"
		int chainOrder = 0;        // preserves extension-local order among chained followers

		int priority = 0;
		size_t order = 0;          // global stable order across all gathered externals
	};

	auto makeAny = [&](ExternalPassDesc const& d) -> AnyPassAndResources {
		return MaterializeExternalPass(d, false, true);
	};

	// Sentinels (must not collide with real pass names)
	static constexpr const char* kBeginKey = "__rg_begin__";
	static constexpr const char* kAfterBaseKey = "__rg_after_base__";
	static constexpr const char* kEndKey = "__rg_end__"; // legacy alias for end-of-base anchor
	static constexpr const char* kFirstBaseKey = "__rg_first_base__"; // optional helper token

	struct ExtItem {
		AnyPassAndResources pr;
		std::string key;                 // unique key for anchoring
		ExternalInsertPoint where;        // constraints
		int priority = 0;
		size_t order = 0;                // global stable order
		int extIndex = 0;                // which extension emitted it
		int extLocalOrder = 0;           // order within that extension
	};

	struct MergeNode {
		std::string key;
		bool hasPass = false;            // sentinel nodes have no pass payload
		AnyPassAndResources pass{};      // valid iff hasPass
		int priority = 0;
		size_t order = 0;
		std::vector<size_t> out;
		uint32_t indeg = 0;
	};

	// Keep base passes
	auto base = std::move(m_masterPassList);
	m_masterPassList.clear();
	m_structuralExplicitAfterByName.clear();

	// Gather extension passes into ExtItem list
	std::vector<ExtItem> extItems;
	extItems.reserve(64);

	size_t globalOrder = 0;

	// helper: stable synthetic key for unnamed passes
	auto makeSyntheticKey = [](size_t n) -> std::string {
		return "__rg_ext_" + std::to_string(n);
		};

	for (int ei = 0; ei < (int)m_extensions.size(); ++ei) {
		auto& ext = m_extensions[ei];
		if (!ext) continue;

		std::vector<ExternalPassDesc> local;
		local.reserve(16);
		if (m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled()) {
			spdlog::info("RG gather structural extension {} begin", ei);
		}
		ext->GatherStructuralPasses(*this, local);
		if (m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled()) {
			spdlog::info("RG gather structural extension {} complete localPassCount={}", ei, local.size());
		}

		std::optional<std::string> prevKey; // for extension-local chaining
		int localOrder = 0;

		for (auto& d : local) {
			if (d.type == PassType::Unknown) continue;
			if (std::holds_alternative<std::monostate>(d.pass)) continue;

			ExtItem it;
			if (m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled()) {
				spdlog::info("RG gather structural extension {} materialize begin name='{}' type={} localOrder={}", ei, d.name, static_cast<int>(d.type), localOrder);
			}
			it.pr = MaterializeExternalPass(d, false, true);
			if (m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled()) {
				spdlog::info("RG gather structural extension {} materialize complete name='{}'", ei, d.name);
			}
			RegisterExternalPassName(d, it.pr);
			if (m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled()) {
				spdlog::info("RG gather structural extension {} register external pass name complete name='{}'", ei, d.name);
			}

			it.order = globalOrder++;
			it.key = !d.name.empty() ? d.name : makeSyntheticKey(it.order);

			if (d.where.has_value()) {
				it.where = *d.where;
			}
			else {
				it.where = ExternalInsertPoint{};
				it.where.keepExtensionOrder = true;

				// Only the *first* unconstrained pass gets anchored after base.
				// Followers will be ordered by chain edges.
				if (!prevKey.has_value()) {
					it.where.after.push_back(kAfterBaseKey);
				}
			}

			it.priority = it.where.priority;
			it.extIndex = ei;
			it.extLocalOrder = localOrder++;

			// Store prevKey for later chaining edges (we apply chaining even if explicit where)
			// (We don't add edges here because we haven't built node indices yet.)
			// We'll reconstruct chaining using extIndex/extLocalOrder + keepExtensionOrder below.
			extItems.push_back(std::move(it));
			prevKey = extItems.back().key;
		}
	}

	// Build nodes list: sentinels + base + externals
	std::vector<MergeNode> nodes;
	nodes.reserve(2 + base.size() + extItems.size());

	auto addNode = [&](std::string key, bool hasPass, AnyPassAndResources&& pass, int prio, size_t ord) -> size_t {
		MergeNode n;
		n.key = std::move(key);
		n.hasPass = hasPass;
		if (hasPass) n.pass = std::move(pass);
		n.priority = prio;
		n.order = ord;
		nodes.push_back(std::move(n));
		return nodes.size() - 1;
		};

	const size_t beginIdx = addNode(std::string(kBeginKey), false, AnyPassAndResources{}, INT_MIN, 0);
	const size_t afterBaseIdx = addNode(std::string(kAfterBaseKey), false, AnyPassAndResources{}, INT_MAX, 1);

	// Map key->node index (detect collisions early)
	std::unordered_map<std::string, size_t> keyToIdx;
	keyToIdx.reserve(2 + base.size() + extItems.size());
	keyToIdx.emplace(nodes[beginIdx].key, beginIdx);
	keyToIdx.emplace(nodes[afterBaseIdx].key, afterBaseIdx);
	keyToIdx.emplace(kEndKey, afterBaseIdx);

	// Add base pass nodes (preserve current base order deterministically)
	std::vector<size_t> baseIdx;
	baseIdx.reserve(base.size());

	for (size_t bi = 0; bi < base.size(); ++bi) {
		AnyPassAndResources bp = std::move(base[bi]);

		// Anchor keys for base: use name if present, else synthetic (not anchorable by user)
		std::string key = !bp.name.empty() ? bp.name : ("__rg_base_" + std::to_string(bi));

		if (keyToIdx.contains(key)) {
			throw std::runtime_error("Pass name/key collision during structural merge: " + key);
		}

		size_t idx = addNode(key, true, std::move(bp), /*prio=*/0, /*ord=*/1000 + bi);
		keyToIdx.emplace(nodes[idx].key, idx);
		baseIdx.push_back(idx);
	}

	// Add external pass nodes
	std::vector<size_t> extIdx;
	extIdx.reserve(extItems.size());

	for (size_t i = 0; i < extItems.size(); ++i) {
		auto& e = extItems[i];

		if (keyToIdx.contains(e.key)) {
			spdlog::error("External pass name/key collision during structural merge: {}", e.key);
			throw std::runtime_error("External pass name/key collision during structural merge: " + e.key);
		}

		size_t idx = addNode(e.key, true, std::move(e.pr), e.priority, /*ord=*/2000 + e.order);
		keyToIdx.emplace(nodes[idx].key, idx);
		extIdx.push_back(idx);
	}

	// Edge helper (dedup)
	std::unordered_set<uint64_t> edgeSet;
	edgeSet.reserve(nodes.size() * 8);

	auto addEdge = [&](size_t from, size_t to) {
		if (from == to) return;
		uint64_t k = (uint64_t(from) << 32) | uint64_t(to);
		if (!edgeSet.insert(k).second) return;
		nodes[from].out.push_back(to);
		nodes[to].indeg++;
		};

	// Base order edges: BEGIN -> base0 -> base1 -> ... -> lastBase -> AFTER_BASE
	if (!baseIdx.empty()) {
		addEdge(beginIdx, baseIdx.front());
		for (size_t i = 0; i + 1 < baseIdx.size(); ++i) addEdge(baseIdx[i], baseIdx[i + 1]);
		addEdge(baseIdx.back(), afterBaseIdx);
	}
	else {
		addEdge(beginIdx, afterBaseIdx);
	}

	// Apply external constraints + extension chaining
	// Build per-extension ordering list (by extLocalOrder)
	std::unordered_map<int, std::vector<std::pair<int, size_t>>> extOrder; // extIndex -> [(localOrder, nodeIdx)]
	extOrder.reserve(m_extensions.size());

	for (size_t i = 0; i < extItems.size(); ++i) {
		extOrder[extItems[i].extIndex].push_back({ extItems[i].extLocalOrder, extIdx[i] });
	}

	for (auto& [ei, v] : extOrder) {
		std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first < b.first; });
	}

	// Now attach constraints and chain edges
	for (size_t i = 0; i < extItems.size(); ++i) {
		auto& e = extItems[i];
		const size_t passNode = extIdx[i];

		// Helper: resolve special token for "first base"
		auto resolveAnchor = [&](std::string const& anchor) -> std::optional<size_t> {
			if (anchor == kFirstBaseKey) {
				if (!baseIdx.empty()) return baseIdx.front();
				return afterBaseIdx;
			}
			auto it = keyToIdx.find(anchor);
			if (it == keyToIdx.end()) return std::nullopt;
			return it->second;
			};

		bool anyConstraint = false;

		// after[] : anchor -> pass
		for (auto const& a : e.where.after) {
			auto idxOpt = resolveAnchor(a);
			if (!idxOpt) {
				if (!a.starts_with("CLodShadow::")) {
					spdlog::warn("External pass '{}' requested After('{}') but anchor not found; ignoring.", e.key, a);
				}
				continue;
			}
			addEdge(*idxOpt, passNode);
			if (a != kBeginKey && a != kAfterBaseKey && a != kEndKey && a != kFirstBaseKey) {
				m_structuralExplicitAfterByName.push_back({ a, e.key });
			}
			anyConstraint = true;
		}

		// before[] : pass -> anchor
		for (auto const& b : e.where.before) {
			auto idxOpt = resolveAnchor(b);
			if (!idxOpt) {
				spdlog::warn("External pass '{}' requested Before('{}') but anchor not found; ignoring.", e.key, b);
				continue;
			}
			addEdge(passNode, *idxOpt);
			if (b != kBeginKey && b != kAfterBaseKey && b != kEndKey && b != kFirstBaseKey) {
				m_structuralExplicitAfterByName.push_back({ e.key, b });
			}
			anyConstraint = true;
		}
	}

	// Extension chaining edges: prev -> next (if keepExtensionOrder on the *next* pass)
	for (auto& [ei, v] : extOrder) {
		for (size_t j = 1; j < v.size(); ++j) {
			// Find the extItems entry for this node to check keepExtensionOrder
			// (We can check by key because keys are unique.)
			const size_t prevNode = v[j - 1].second;
			const size_t nextNode = v[j].second;

			const std::string& nextKey = nodes[nextNode].key;

			// locate corresponding ext item (small N; linear is fine)
			bool keep = true;
			for (auto& e : extItems) {
				if (e.key == nextKey) { keep = e.where.keepExtensionOrder; break; }
			}
			if (keep) addEdge(prevNode, nextNode);
		}
	}

	// Topological sort (stable by priority then order)
	std::vector<uint32_t> indeg(nodes.size());
	for (size_t n = 0; n < nodes.size(); ++n) indeg[n] = nodes[n].indeg;

	std::vector<size_t> ready;
	ready.reserve(nodes.size());
	for (size_t n = 0; n < nodes.size(); ++n) if (indeg[n] == 0) ready.push_back(n);

	auto better = [&](size_t a, size_t b) {
		if (nodes[a].priority != nodes[b].priority) return nodes[a].priority < nodes[b].priority;
		return nodes[a].order < nodes[b].order;
		};

	std::vector<size_t> topo;
	topo.reserve(nodes.size());

	while (!ready.empty()) {
		auto it = std::min_element(ready.begin(), ready.end(), [&](size_t a, size_t b) { return better(a, b); });
		size_t u = *it;
		ready.erase(it);

		topo.push_back(u);
		for (size_t vtx : nodes[u].out) {
			if (--indeg[vtx] == 0) ready.push_back(vtx);
		}
	}

	if (topo.size() != nodes.size()) {
		spdlog::error("Structural merge has a cycle (extension anchors/chains impossible).");
		throw std::runtime_error("RenderGraph structural merge cycle");
	}

	// Emit final m_masterPassList in topo order (skip sentinels)
	m_masterPassList.clear();
	m_masterPassList.reserve(baseIdx.size() + extIdx.size());

	for (size_t u : topo) {
		if (!nodes[u].hasPass) {
			continue;
		}
		m_masterPassList.push_back(std::move(nodes[u].pass));
	}
}


static ResourceRegistry::RegistryHandle ResolveByIdThunk(void* user, ResourceIdentifier const& id, bool allowFailure) {
	return static_cast<RenderGraph*>(user)->RequestResourceHandle(id, allowFailure);
}

static ResourceRegistry::RegistryHandle ResolveByPtrThunk(void* user, Resource* ptr, bool allowFailure) {
	return static_cast<RenderGraph*>(user)->RequestResourceHandle(ptr, allowFailure);
}

static bool Overlap(SubresourceRange a, SubresourceRange b) {
	auto aMipEnd = a.firstMip + a.mipCount;
	auto bMipEnd = b.firstMip + b.mipCount;
	auto aSlEnd = a.firstSlice + a.sliceCount;
	auto bSlEnd = b.firstSlice + b.sliceCount;
	return (a.firstMip < bMipEnd && b.firstMip < aMipEnd) &&
		(a.firstSlice < bSlEnd && b.firstSlice < aSlEnd);
}

static bool RequirementsConflict(
	std::span<const ResourceRequirement> retained,
	std::span<const ResourceRequirement> immediate)
{
	if (retained.empty() || immediate.empty()) return false;

	// Group immediate requirements by resource ID for O(N+M) lookup
	std::unordered_map<uint64_t, std::vector<const ResourceRequirement*>> immediateByID;
	immediateByID.reserve(immediate.size());
	for (auto const& ib : immediate) {
		immediateByID[ib.resourceHandleAndRange.resource.GetGlobalResourceID()].push_back(&ib);
	}

	for (auto const& ra : retained) {
		auto res = ra.resourceHandleAndRange.resource;
		uint64_t rid = res.GetGlobalResourceID();
		auto it = immediateByID.find(rid);
		if (it == immediateByID.end()) continue;

		auto a = ResolveRangeSpec(ra.resourceHandleAndRange.range, res.GetNumMipLevels(), res.GetArraySize());
		if (a.isEmpty()) continue;

		for (auto const* ib : it->second) {
			auto b = ResolveRangeSpec(ib->resourceHandleAndRange.range, res.GetNumMipLevels(), res.GetArraySize());
			if (b.isEmpty()) continue;

			if (Overlap(a, b) && !(ra.state == ib->state)) {
				return true;
			}
		}
	}
	return false;
}


bool RenderGraph::RefreshRetainedDeclarationsForFrame(RenderPassAndResources& p, uint8_t frameIndex)
{
	ZoneScopedN("RenderGraph::RefreshRetainedDeclarationsForFrame(Render)");
	const bool traceLifecycle = m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled();
	const uint64_t previousDeclarationFingerprint = p.declarationCache.declarationFingerprint;
	if (!p.name.empty()) {
		ZoneText(p.name.data(), p.name.size());
	}
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh render pass '{}' declare begin", frameIndex, p.name);
	}
	RenderPassBuilder b(this, p.name);

	// Make it look like a normal builder enough for any pass code that queries ResourceProvider()
	b.pass = p.pass;
	b.built_ = true;

	// Clear any previous declarations
	b.params = {};
	b._declaredIds.clear();

	// Let the pass declare based on current per-frame state (queued mip jobs etc.)
	EnsureProviderRegistered(p.pass.get());
	p.pass->DeclareResourceUsages(&b);
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh render pass '{}' declare complete requirements={} transitions={}", frameIndex, p.name, b.GatherResourceRequirements().size(), b.params.internalTransitions.size());
	}
	const auto& refreshedRequirements = b.GatherResourceRequirements();

	// Update the frame view used by scheduling
	p.resources.staticResourceRequirements = refreshedRequirements;
	p.resources.mergedFrameRequirementsDirty = true;

	// Internal transitions also affect scheduling
	p.resources.internalTransitions = b.params.internalTransitions;

	p.resources.identifierSet = b.DeclaredResourceIds();
	p.resources.autoDescriptorShaderResources = b.params.autoDescriptorShaderResources;
	p.resources.autoDescriptorConstantBuffers = b.params.autoDescriptorConstantBuffers;
	p.resources.autoDescriptorUnorderedAccessViews = b.params.autoDescriptorUnorderedAccessViews;
	p.resources.activeFeatureDomains = b.params.activeFeatureDomains;
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh render pass '{}' materialize referenced resources begin", frameIndex, p.name);
	}
	MaterializeReferencedResources(p.resources.staticResourceRequirements, p.resources.internalTransitions);

	// Transfer resolver snapshots for auto-invalidation tracking
	p.resolverSnapshots = b.TakeResolverSnapshots();
	p.retainedAnonymousKeepAlive = CaptureRetainedAnonymousKeepAlive(
		p.resources.staticResourceRequirements,
		p.resources.internalTransitions);
	UpdateRetainedDeclarationCache(_registry, p);

	// Ensure the pass's view matches the refreshed identifier set
	p.pass->SetResourceRegistryView(
		std::make_unique<ResourceRegistryView>(_registry, p.resources.identifierSet),
		p.resources.activeFeatureDomains,
		p.resources.autoDescriptorShaderResources,
		p.resources.autoDescriptorConstantBuffers,
		p.resources.autoDescriptorUnorderedAccessViews
	);
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh render pass '{}' setup begin", frameIndex, p.name);
	}
	p.pass->Setup();
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh render pass '{}' setup complete", frameIndex, p.name);
	}
	return previousDeclarationFingerprint != p.declarationCache.declarationFingerprint;
}

bool RenderGraph::RefreshRetainedDeclarationsForFrame(ComputePassAndResources& p, uint8_t frameIndex)
{
	ZoneScopedN("RenderGraph::RefreshRetainedDeclarationsForFrame(Compute)");
	const bool traceLifecycle = m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled();
	const uint64_t previousDeclarationFingerprint = p.declarationCache.declarationFingerprint;
	if (!p.name.empty()) {
		ZoneText(p.name.data(), p.name.size());
	}
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh compute pass '{}' declare begin", frameIndex, p.name);
	}
	ComputePassBuilder b(this, p.name);
	b.pass = p.pass;
	b.built_ = true;

	b.params = {};
	b._declaredIds.clear();

	EnsureProviderRegistered(p.pass.get());
	p.pass->DeclareResourceUsages(&b);
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh compute pass '{}' declare complete requirements={} transitions={}", frameIndex, p.name, b.GatherResourceRequirements().size(), b.params.internalTransitions.size());
	}
	const auto& refreshedRequirements = b.GatherResourceRequirements();

	p.resources.staticResourceRequirements = refreshedRequirements;
	p.resources.mergedFrameRequirementsDirty = true;
	p.resources.internalTransitions = b.params.internalTransitions;
	p.resources.identifierSet = b.DeclaredResourceIds();
	p.resources.autoDescriptorShaderResources = b.params.autoDescriptorShaderResources;
	p.resources.autoDescriptorConstantBuffers = b.params.autoDescriptorConstantBuffers;
	p.resources.autoDescriptorUnorderedAccessViews = b.params.autoDescriptorUnorderedAccessViews;
	p.resources.activeFeatureDomains = b.params.activeFeatureDomains;
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh compute pass '{}' materialize referenced resources begin", frameIndex, p.name);
	}
	MaterializeReferencedResources(p.resources.staticResourceRequirements, p.resources.internalTransitions);

	// Transfer resolver snapshots for auto-invalidation tracking
	p.resolverSnapshots = b.TakeResolverSnapshots();
	p.retainedAnonymousKeepAlive = CaptureRetainedAnonymousKeepAlive(
		p.resources.staticResourceRequirements,
		p.resources.internalTransitions);
	UpdateRetainedDeclarationCache(_registry, p);

	p.pass->SetResourceRegistryView(
		std::make_unique<ResourceRegistryView>(_registry, p.resources.identifierSet),
		p.resources.activeFeatureDomains,
		p.resources.autoDescriptorShaderResources,
		p.resources.autoDescriptorConstantBuffers,
		p.resources.autoDescriptorUnorderedAccessViews
	);
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh compute pass '{}' setup begin", frameIndex, p.name);
	}

	p.pass->Setup();
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh compute pass '{}' setup complete", frameIndex, p.name);
	}
	return previousDeclarationFingerprint != p.declarationCache.declarationFingerprint;
}

bool RenderGraph::RefreshRetainedDeclarationsForFrame(CopyPassAndResources& p, uint8_t frameIndex)
{
	ZoneScopedN("RenderGraph::RefreshRetainedDeclarationsForFrame(Copy)");
	const bool traceLifecycle = m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled();
	const uint64_t previousDeclarationFingerprint = p.declarationCache.declarationFingerprint;
	if (!p.name.empty()) {
		ZoneText(p.name.data(), p.name.size());
	}
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh copy pass '{}' declare begin", frameIndex, p.name);
	}
	CopyPassBuilder b(this, p.name);
	b.pass = p.pass;
	b.built_ = true;

	b.params = {};
	b._declaredIds.clear();

	EnsureProviderRegistered(p.pass.get());
	p.pass->DeclareResourceUsages(&b);
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh copy pass '{}' declare complete requirements={} transitions={}", frameIndex, p.name, b.GatherResourceRequirements().size(), b.params.internalTransitions.size());
	}
	const auto& refreshedRequirements = b.GatherResourceRequirements();

	p.resources.staticResourceRequirements = refreshedRequirements;
	p.resources.mergedFrameRequirementsDirty = true;
	p.resources.internalTransitions = b.params.internalTransitions;
	p.resources.identifierSet = b.DeclaredResourceIds();
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh copy pass '{}' materialize referenced resources begin", frameIndex, p.name);
	}
	MaterializeReferencedResources(p.resources.staticResourceRequirements, p.resources.internalTransitions);

	// Transfer resolver snapshots for auto-invalidation tracking
	p.resolverSnapshots = b.TakeResolverSnapshots();
	p.retainedAnonymousKeepAlive = CaptureRetainedAnonymousKeepAlive(
		p.resources.staticResourceRequirements,
		p.resources.internalTransitions);
	UpdateRetainedDeclarationCache(_registry, p);

	p.pass->SetResourceRegistryView(
		std::make_unique<ResourceRegistryView>(_registry, p.resources.identifierSet)
	);
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh copy pass '{}' setup begin", frameIndex, p.name);
	}

	p.pass->Setup();
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh copy pass '{}' setup complete", frameIndex, p.name);
	}
	return previousDeclarationFingerprint != p.declarationCache.declarationFingerprint;
}

void RenderGraph::CompileFrame(rhi::Device device, uint8_t frameIndex, const IHostExecutionData* hostData) {
	ZoneScopedN("RenderGraph::CompileFrame");
	const bool traceLifecycle = m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled();
	auto traceCompileStep = [&](const char* step) {
		if (traceLifecycle) {
			spdlog::info("RG frame {} compile step: {}", frameIndex, step);
		}
	};
	traceCompileStep("begin");

	{
		traceCompileStep("ResetCompileState");
		ZoneScopedN("RenderGraph::CompileFrame::ResetCompileState");
		m_frameCompileResources.clear();
		m_schedulingEquivalentIDsCache.clear();
		m_lastRegionStats = {};
		m_lastExtractedRegions.clear();
		m_lastRegionCandidateDiagnostics.clear();
		m_schedulingDecisionTrace.clear();
		m_transitionPlacementCandidates.clear();
		m_transitionPlacementStats = {};
		m_frameDeclarationRefreshRequestedCount = 0;
		m_frameDeclarationRefreshEquivalentCount = 0;
		m_aliasingSubsystem.ResetPerFrameState(*this);
	}

	auto needsRefresh = [&](auto& p) -> bool {
		// Check if any stored resolver's content version has changed
		for (const auto& snap : p.resolverSnapshots) {
			uint64_t cv = snap.resolver->GetContentVersion();
			if (cv != 0 && cv != snap.version) {
				return true;
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

		return p.declarationCache.dynamicInterface->DeclaredResourcesChanged();
		};

	std::unordered_set<std::string> declarationRefreshedPassNames;
	std::unordered_set<std::string> frameExtensionPassNames;
	{
		ZoneScopedN("RenderGraph::CompileFrame::RefreshRetainedDeclarations");
		// First, refresh all retained declarations for this frame
		for (auto& pr : m_masterPassList) {
			if (pr.type == PassType::Compute) {
				auto& p = std::get<ComputePassAndResources>(pr.pass);
				if (needsRefresh(p)) {
					++m_frameDeclarationRefreshRequestedCount;
					const bool declarationActuallyChanged = RefreshRetainedDeclarationsForFrame(p, frameIndex);
					if (!declarationActuallyChanged) {
						++m_frameDeclarationRefreshEquivalentCount;
					}
					if (declarationActuallyChanged && !p.name.empty()) {
						declarationRefreshedPassNames.insert(p.name);
					}
				}
			}
			else if (pr.type == PassType::Render) {
				auto& p = std::get<RenderPassAndResources>(pr.pass);
				if (needsRefresh(p)) {
					++m_frameDeclarationRefreshRequestedCount;
					const bool declarationActuallyChanged = RefreshRetainedDeclarationsForFrame(p, frameIndex);
					if (!declarationActuallyChanged) {
						++m_frameDeclarationRefreshEquivalentCount;
					}
					if (declarationActuallyChanged && !p.name.empty()) {
						declarationRefreshedPassNames.insert(p.name);
					}
				}
			}
			else if (pr.type == PassType::Copy) {
				auto& p = std::get<CopyPassAndResources>(pr.pass);
				if (needsRefresh(p)) {
					++m_frameDeclarationRefreshRequestedCount;
					const bool declarationActuallyChanged = RefreshRetainedDeclarationsForFrame(p, frameIndex);
					if (!declarationActuallyChanged) {
						++m_frameDeclarationRefreshEquivalentCount;
					}
					if (declarationActuallyChanged && !p.name.empty()) {
						declarationRefreshedPassNames.insert(p.name);
					}
				}
			}
		}
	}

	{
		ZoneScopedN("RenderGraph::CompileFrame::InitFramePassState");
		batches.clear();
		batches.emplace_back(m_queueRegistry.SlotCount()); // Dummy batch 0 for pre-first-pass transitions
		m_framePasses.clear(); // Combined retained + immediate-mode passes for this frame
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

	// Record immediate-mode commands + access for each pass and fold into per-frame requirements
	for (auto& pr : m_masterPassList) {

		if (pr.type == PassType::Compute) {
			auto& p = std::get<ComputePassAndResources>(pr.pass);
			auto* immediateModeCommands = getImmediateModeCommands(p.pass.get());
			if (!immediateModeCommands) {
				p.run = PassRunMask::Retained;
				m_framePasses.push_back(pr);
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
				m_framePasses.push_back(pr);
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
				m_framePasses.push_back(immediateAnyPassAndResources);
				p.run = PassRunMask::Retained;
				m_framePasses.push_back(pr); // Retained pass
			}
			else {
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				SetImmediateFrameRequirements(p.resources, std::move(immediateFrameData.requirements));
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
				m_framePasses.push_back(pr);
			}
		}
		else if (pr.type == PassType::Render) {
			auto& p = std::get<RenderPassAndResources>(pr.pass);
			auto* immediateModeCommands = getImmediateModeCommands(p.pass.get());
			if (!immediateModeCommands) {
				p.run = PassRunMask::Retained;
				m_framePasses.push_back(pr);
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
				m_framePasses.push_back(pr);
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
				m_framePasses.push_back(immediateAnyPassAndResources);
				p.run = PassRunMask::Retained;
				m_framePasses.push_back(pr); // Retained pass
			}
			else {
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				SetImmediateFrameRequirements(p.resources, std::move(immediateFrameData.requirements));
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
				m_framePasses.push_back(pr);
			}
		}
		else if (pr.type == PassType::Copy) {
			auto& p = std::get<CopyPassAndResources>(pr.pass);
			auto* immediateModeCommands = getImmediateModeCommands(p.pass.get());
			if (!immediateModeCommands) {
				p.run = PassRunMask::Retained;
				m_framePasses.push_back(pr);
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
				m_framePasses.push_back(pr);
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
				m_framePasses.push_back(immediateAnyPassAndResources);
				p.run = PassRunMask::Retained;
				m_framePasses.push_back(pr);
			}
			else {
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				SetImmediateFrameRequirements(p.resources, std::move(immediateFrameData.requirements));
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
				m_framePasses.push_back(pr);
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
	std::vector<std::pair<std::string, std::string>> explicitAfterByName = m_structuralExplicitAfterByName;
	explicitAfterByName.reserve(m_structuralExplicitAfterByName.size() + frameExt.size());

	if (!frameExt.empty()) {
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

		std::vector<AnyPassAndResources> baseFramePasses = std::move(m_framePasses);
		m_framePasses.clear();

		std::unordered_map<std::string, size_t> baseFramePassIndexByName;
		baseFramePassIndexByName.reserve(baseFramePasses.size() + frameExt.size());
		for (size_t i = 0; i < baseFramePasses.size(); ++i) {
			if (!baseFramePasses[i].name.empty()) {
				baseFramePassIndexByName[baseFramePasses[i].name] = i;
			}
		}

		struct PendingFrameInsert {
			AnyPassAndResources pass;
			size_t slotIndex = 0;
			size_t nextInsertIndex = (std::numeric_limits<size_t>::max)();
		};

		const size_t invalidInsertIndex = (std::numeric_limits<size_t>::max)();
		std::vector<PendingFrameInsert> pendingFrameInserts;
		pendingFrameInserts.reserve(frameExt.size());
		std::vector<size_t> slotHeadByIndex(baseFramePasses.size() + 1, invalidInsertIndex);
		std::vector<size_t> slotTailByIndex(baseFramePasses.size() + 1, invalidInsertIndex);
		std::unordered_map<std::string, size_t> pendingInsertIndexByName;
		pendingInsertIndexByName.reserve(frameExt.size());
		std::unordered_map<std::string, size_t> pendingInsertTailByAnchorName;
		pendingInsertTailByAnchorName.reserve(frameExt.size());

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
			const size_t pendingIndex = pendingFrameInserts.size();
			pendingFrameInserts.push_back(PendingFrameInsert{ .pass = std::move(any) });

			std::string anchorName;
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
				explicitAfterByName.push_back({ anchorName, insertedPassName });
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

		m_framePasses.reserve(baseFramePasses.size() + pendingFrameInserts.size());
		auto appendSlot = [&](size_t slotIndex) {
			for (size_t pendingIndex = slotHeadByIndex[slotIndex]; pendingIndex != invalidInsertIndex; pendingIndex = pendingFrameInserts[pendingIndex].nextInsertIndex) {
				m_framePasses.push_back(std::move(pendingFrameInserts[pendingIndex].pass));
			}
		};

		for (size_t i = 0; i < baseFramePasses.size(); ++i) {
			appendSlot(i);
			m_framePasses.push_back(std::move(baseFramePasses[i]));
		}
		appendSlot(baseFramePasses.size());
	}

	m_framePassIsFrameExtension.assign(m_framePasses.size(), 0);
	m_framePassDeclarationRefreshedThisFrame.assign(m_framePasses.size(), 0);
	for (size_t passIndex = 0; passIndex < m_framePasses.size(); ++passIndex) {
		const auto& passName = m_framePasses[passIndex].name;
		if (!passName.empty()) {
			m_framePassIsFrameExtension[passIndex] = frameExtensionPassNames.contains(passName) ? 1 : 0;
			m_framePassDeclarationRefreshedThisFrame[passIndex] = declarationRefreshedPassNames.contains(passName) ? 1 : 0;
		}
	}

	// Register/refresh pass statistics indices for this frame's concrete pass list.
	// This supports transient passes and per-frame retained/immediate splits.
	if (m_statisticsService) {
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

	std::unordered_set<uint64_t> usedResourceIDs;
	{
		traceCompileStep("CollectFrameResourceIDs");
		ZoneScopedN("RenderGraph::CompileFrame::CollectFrameResourceIDs");
		CollectFrameResourceIDs(usedResourceIDs);
	}
	{
		traceCompileStep("ApplyIdleDematerializationPolicy");
		ZoneScopedN("RenderGraph::CompileFrame::ApplyIdleDematerializationPolicy");
		ApplyIdleDematerializationPolicy(usedResourceIDs);
	}

	// Convert explicit After(anchorName)->(passName) constraints into node-index edges.
	std::vector<std::pair<size_t, size_t>> explicitEdges;
	explicitEdges.reserve(explicitAfterByName.size());
	if (!explicitAfterByName.empty()) {
		std::unordered_map<std::string, size_t> nameToIndex;
		nameToIndex.reserve(m_framePasses.size());
		for (size_t i = 0; i < m_framePasses.size(); ++i) {
			if (!m_framePasses[i].name.empty()) {
				nameToIndex[m_framePasses[i].name] = i;
			}
		}
		for (auto const& e : explicitAfterByName) {
			auto itA = nameToIndex.find(e.first);
			auto itB = nameToIndex.find(e.second);
			if (itA == nameToIndex.end() || itB == nameToIndex.end()) {
				spdlog::warn("Explicit After edge dropped (anchor='{}', pass='{}'): name not found in frame pass list.", e.first, e.second);
				continue;
			}
			explicitEdges.push_back({ itA->second, itB->second });
		}
	}

	std::vector<Node> nodes;
	m_frameDAGResourceIndexByID.clear();
	m_frameDAGResourceIDsByIndex.clear();
	m_frameDAGResourceIndexByID.reserve(usedResourceIDs.size());
	m_frameDAGResourceIDsByIndex.reserve(usedResourceIDs.size());
	for (uint64_t resourceID : usedResourceIDs) {
		m_frameDAGResourceIndexByID.emplace(resourceID, m_frameDAGResourceIDsByIndex.size());
		m_frameDAGResourceIDsByIndex.push_back(resourceID);
	}
	m_frameDAGResourceCount = m_frameDAGResourceIDsByIndex.size();
	{
		traceCompileStep("RebuildFramePassAccessSummaries");
		ZoneScopedN("RenderGraph::CompileFrame::RebuildFramePassAccessSummaries");
		RebuildFramePassAccessSummaries();
	}
	{
		traceCompileStep("RebuildFrameSchedulingResourceIndexForAliasing");
		ZoneScopedN("RenderGraph::CompileFrame::RebuildFrameSchedulingResourceIndexForAliasing");
		RebuildFrameSchedulingResourceIndex(usedResourceIDs);
	}
	{
		traceCompileStep("RebuildFramePassSchedulingSummariesForAliasing");
		ZoneScopedN("RenderGraph::CompileFrame::RebuildFramePassSchedulingSummariesForAliasing");
		RebuildFramePassSchedulingSummaries();
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

	rg::alias::FrameAliasAnalysis aliasAnalysis;
	{
		traceCompileStep("BuildAliasFrameAnalysis");
		ZoneScopedN("RenderGraph::CompileFrame::BuildAliasFrameAnalysis");
		aliasAnalysis = m_aliasingSubsystem.BuildAliasFrameAnalysis(*this, aliasNodes);
	}
	{
		traceCompileStep("AutoAssignAliasingPoolsFromAnalysis");
		ZoneScopedN("RenderGraph::CompileFrame::AutoAssignAliasingPoolsFromAnalysis");
		m_aliasingSubsystem.AutoAssignAliasingPoolsFromAnalysis(*this, aliasAnalysis);
	}
	{
		traceCompileStep("BuildAliasPlanFromAnalysis");
		ZoneScopedN("RenderGraph::CompileFrame::BuildAliasPlanFromAnalysis");
		m_aliasingSubsystem.BuildAliasPlanFromAnalysis(*this, aliasAnalysis);
	}
	{
		traceCompileStep("AddCurrentFrameAliasSchedulingEdges");
		ZoneScopedN("RenderGraph::CompileFrame::AddCurrentFrameAliasSchedulingEdges");
		if (!AddCurrentFrameAliasSchedulingEdges(nodes)) {
			spdlog::error("Render graph alias scheduling introduced a dependency cycle! Render graph compilation failed.");
			throw std::runtime_error("Render graph alias scheduling introduced a dependency cycle");
		}
	}
	{
		traceCompileStep("RebuildSchedulingEquivalentIDCache");
		ZoneScopedN("RenderGraph::CompileFrame::RebuildSchedulingEquivalentIDCache");
		RebuildSchedulingEquivalentIDCache(usedResourceIDs);
	}
	{
		traceCompileStep("RebuildFrameSchedulingResourceIndex");
		ZoneScopedN("RenderGraph::CompileFrame::RebuildFrameSchedulingResourceIndex");
		RebuildFrameSchedulingResourceIndex(usedResourceIDs);
	}
	{
		traceCompileStep("RebuildFramePassSchedulingSummaries");
		ZoneScopedN("RenderGraph::CompileFrame::RebuildFramePassSchedulingSummaries");
		RebuildFramePassSchedulingSummaries();
	}

	{
		traceCompileStep("PlanActiveQueueSlots");
		ZoneScopedN("RenderGraph::CompileFrame::PlanActiveQueueSlots");
		m_activeQueueSlotsThisFrame = PlanActiveQueueSlots(*this, m_framePasses, nodes);
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
		MaterializeUnmaterializedResources(&usedResourceIDs);
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
	const bool wantsAuthoritativeReplay =
		static_cast<uint8_t>(regionMode) >= static_cast<uint8_t>(rg::runtime::RenderGraphRegionMode::ReplayAuthoritative);

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
				std::vector<ScheduledRegion> replayRegions;
				RegionCacheStats replayRegionStats;
				std::vector<std::string> replayCandidateDiagnostics;
				std::vector<CachedReplaySegment> currentReplaySegments;
				{
					ZoneScopedN("RenderGraph::Replay::HarvestSuccessfulReplaySegments");
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
				m_regionCache.regions.clear();
				m_regionCache.regions.reserve(replayRegions.size());
				for (const auto& region : replayRegions) {
					m_regionCache.regions.push_back(CachedScheduleRegion{ .schedule = region });
				}
				m_regionCache.stats = replayRegionStats;
				m_regionCache.replaySegments = currentReplaySegments;
				InsertOrRefreshReplaySegmentVariants(currentReplaySegments, replayCacheFrameSerial);
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
		if (static_cast<uint8_t>(regionMode) >= static_cast<uint8_t>(rg::runtime::RenderGraphRegionMode::ReplayAuthoritative)) {
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

		auto markCrossFrameWait = [&](size_t dstSlot, size_t srcSlot, uint64_t fenceValue) {
			if (dstSlot == srcSlot) {
				return;
			}
			auto& enabled = m_hasPendingFrameStartQueueWait[dstSlot][srcSlot];
			auto& maxFence = m_pendingFrameStartQueueWaitFenceValue[dstSlot][srcSlot];
			enabled = true;
			maxFence = std::max(maxFence, fenceValue);
		};

		auto accumulateCrossFrameWaitForHandle = [&](size_t passQueueSlot, const ResourceRegistry::RegistryHandle& handle) {
			if (handle.IsEphemeral()) {
				return;
			}

			const uint64_t id = handle.GetGlobalResourceID();
			for (uint64_t rid : GetSchedulingEquivalentIDsCached(id)) {
				auto it = m_lastProducerByResourceAcrossFrames.find(rid);
				if (it != m_lastProducerByResourceAcrossFrames.end()) {
					markCrossFrameWait(passQueueSlot, it->second.queueSlot, it->second.fenceValue);
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
					overlapTriggeredWaitCount++;
					if (overlapSampleCurrentResourceId == 0) {
						overlapSampleCurrentResourceId = rid;
						overlapSamplePreviousResourceId = prevPlacementProducer.resourceID;
					}
				}
			}
		};

		for (const auto& pr : m_framePasses) {
			const size_t passIndex = static_cast<size_t>(&pr - m_framePasses.data());
			std::visit([&](auto const& passAndResources) {
				using T = std::decay_t<decltype(passAndResources)>;
				if constexpr (!std::is_same_v<T, std::monostate>) {
					const size_t fallbackQueueSlot = passAndResources.resources.pinnedQueueSlot
						? static_cast<size_t>(static_cast<uint8_t>(*passAndResources.resources.pinnedQueueSlot))
						: QueueIndex(passAndResources.resources.preferredQueueKind);
					const size_t passQueueSlot = passIndex < m_assignedQueueSlotsByFramePass.size()
						? m_assignedQueueSlotsByFramePass[passIndex]
						: fallbackQueueSlot;
					ForEachFrameRequirement(passAndResources.resources, [&](const auto& req) {
						accumulateCrossFrameWaitForHandle(passQueueSlot, req.resourceHandleAndRange.resource);
					});
					for (auto const& tr : passAndResources.resources.internalTransitions) {
						accumulateCrossFrameWaitForHandle(passQueueSlot, tr.first.resource);
					}
				}
			}, pr.pass);
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

std::tuple<int, int, int> RenderGraph::GetBatchesToWaitOn(
	const ComputePassAndResources& pass,
	size_t sourceQueueSlot,
	std::unordered_set<uint64_t> const& resourcesTransitionedThisPass)
{
	ZoneScopedN("RenderGraph::GetBatchesToWaitOn(Compute)");
	if (!pass.name.empty()) {
		ZoneText(pass.name.data(), pass.name.size());
	}
	int latestTransition = -1, latestProducer = -1, latestUsage = -1;

	auto processResource = [&](ResourceRegistry::RegistryHandle const& res) {
		uint64_t id = res.GetGlobalResourceID();
		for (auto rid : GetSchedulingEquivalentIDsCached(id)) {
			auto resourceIndex = TryGetFrameSchedulingResourceIndex(rid);
			if (!resourceIndex.has_value()) {
				continue;
			}
			latestTransition = std::max(latestTransition, (int)GetFrameQueueHistoryValue(m_frameQueueLastTransitionBatch, sourceQueueSlot, *resourceIndex));
			latestProducer = std::max(latestProducer, (int)GetFrameQueueHistoryValue(m_frameQueueLastProducerBatch, sourceQueueSlot, *resourceIndex));
		}
		};

	ForEachFrameRequirement(pass.resources, [&](const auto& req) {
		processResource(req.resourceHandleAndRange.resource);
	});

	for (auto& transitionID : resourcesTransitionedThisPass) { // We only need to wait on the latest usage for resources that will be transitioned in this batch
		for (auto rid : GetSchedulingEquivalentIDsCached(transitionID)) {
			auto resourceIndex = TryGetFrameSchedulingResourceIndex(rid);
			if (!resourceIndex.has_value()) {
				continue;
			}
			latestUsage = std::max(latestUsage, (int)GetFrameQueueHistoryValue(m_frameQueueLastUsageBatch, sourceQueueSlot, *resourceIndex));
		}
	}

	return { latestTransition, latestProducer, latestUsage };
}

std::tuple<int, int, int> RenderGraph::GetBatchesToWaitOn(
	const RenderPassAndResources& pass,
	size_t sourceQueueSlot,
	std::unordered_set<uint64_t> const& resourcesTransitionedThisPass)
{
	ZoneScopedN("RenderGraph::GetBatchesToWaitOn(Render)");
	if (!pass.name.empty()) {
		ZoneText(pass.name.data(), pass.name.size());
	}
	int latestTransition = -1, latestProducer = -1, latestUsage = -1;

	auto processResource = [&](ResourceRegistry::RegistryHandle const& res) {
		uint64_t id = res.GetGlobalResourceID();
		for (auto rid : GetSchedulingEquivalentIDsCached(id)) {
			auto resourceIndex = TryGetFrameSchedulingResourceIndex(rid);
			if (!resourceIndex.has_value()) {
				continue;
			}
			latestTransition = std::max(latestTransition, (int)GetFrameQueueHistoryValue(m_frameQueueLastTransitionBatch, sourceQueueSlot, *resourceIndex));
			latestProducer = std::max(latestProducer, (int)GetFrameQueueHistoryValue(m_frameQueueLastProducerBatch, sourceQueueSlot, *resourceIndex));
		}
		};

	ForEachFrameRequirement(pass.resources, [&](const auto& req) {
		processResource(req.resourceHandleAndRange.resource);
	});

	for (auto& transitionID : resourcesTransitionedThisPass) { // We only need to wait on the latest usage for resources that will be transitioned in this batch
		for (auto rid : GetSchedulingEquivalentIDsCached(transitionID)) {
			auto resourceIndex = TryGetFrameSchedulingResourceIndex(rid);
			if (!resourceIndex.has_value()) {
				continue;
			}
			latestUsage = std::max(latestUsage, (int)GetFrameQueueHistoryValue(m_frameQueueLastUsageBatch, sourceQueueSlot, *resourceIndex));
		}
	}

	return { latestTransition, latestProducer, latestUsage };
}

std::tuple<int, int, int> RenderGraph::GetBatchesToWaitOn(
	const CopyPassAndResources& pass,
	size_t sourceQueueSlot,
	std::unordered_set<uint64_t> const& resourcesTransitionedThisPass)
{
	ZoneScopedN("RenderGraph::GetBatchesToWaitOn(Copy)");
	if (!pass.name.empty()) {
		ZoneText(pass.name.data(), pass.name.size());
	}
	int latestTransition = -1, latestProducer = -1, latestUsage = -1;

	auto processResource = [&](ResourceRegistry::RegistryHandle const& res) {
		uint64_t id = res.GetGlobalResourceID();
		for (auto rid : GetSchedulingEquivalentIDsCached(id)) {
			auto resourceIndex = TryGetFrameSchedulingResourceIndex(rid);
			if (!resourceIndex.has_value()) {
				continue;
			}
			latestTransition = std::max(latestTransition, (int)GetFrameQueueHistoryValue(m_frameQueueLastTransitionBatch, sourceQueueSlot, *resourceIndex));
			latestProducer = std::max(latestProducer, (int)GetFrameQueueHistoryValue(m_frameQueueLastProducerBatch, sourceQueueSlot, *resourceIndex));
		}
		};

	ForEachFrameRequirement(pass.resources, [&](const auto& req) {
		processResource(req.resourceHandleAndRange.resource);
	});

	for (auto& transitionID : resourcesTransitionedThisPass) {
		for (auto rid : GetSchedulingEquivalentIDsCached(transitionID)) {
			auto resourceIndex = TryGetFrameSchedulingResourceIndex(rid);
			if (!resourceIndex.has_value()) {
				continue;
			}
			latestUsage = std::max(latestUsage, (int)GetFrameQueueHistoryValue(m_frameQueueLastUsageBatch, sourceQueueSlot, *resourceIndex));
		}
	}

	return { latestTransition, latestProducer, latestUsage };
}

void RenderGraph::MaterializeUnmaterializedResources(const std::unordered_set<uint64_t>* onlyResourceIDs) {
	ZoneScopedN("RenderGraph::MaterializeUnmaterializedResources");
	// Returns the backing generation if the resource was materialized (or already materialized), or nullopt if skipped.
	auto materializeOne = [&](uint64_t id, Resource* resource) -> std::optional<uint64_t> {
		if (onlyResourceIDs && onlyResourceIDs->find(id) == onlyResourceIDs->end()) {
			return std::nullopt;
		}
		if (!resource) {
			return std::nullopt;
		}

		auto texture = dynamic_cast<PixelBuffer*>(resource);
		if (texture) {
			if (!texture->IsMaterialized()) {
				auto itAlias = aliasMaterializeOptionsByID.find(id);
				if (itAlias != aliasMaterializeOptionsByID.end()) {
					if (std::holds_alternative<PixelBuffer::MaterializeOptions>(itAlias->second)) {
						auto& options = std::get<PixelBuffer::MaterializeOptions>(itAlias->second);
						if (options.aliasPlacement.has_value()) {
							const auto& ap = options.aliasPlacement.value();
							spdlog::info(
								"RG alias materialize: id={} name='{}' pool={} offset={}",
								id,
								resource->GetName(),
								ap.poolID.has_value() ? ap.poolID.value() : 0ull,
								ap.offset);
						}
						texture->Materialize(&options);
					}
				}
				else {
					if (texture->GetDescription().allowAlias) {
						const bool hasManualAliasPool = texture->GetDescription().aliasingPoolID.has_value();
						const bool hasAutoAliasPool = autoAliasPoolByID.find(id) != autoAliasPoolByID.end();
						const bool aliasPlacementRequiredThisFrame = hasManualAliasPool || hasAutoAliasPool;

						if (onlyResourceIDs && aliasPlacementRequiredThisFrame) {
							throw std::runtime_error(
								"Aliasing placement missing for used aliased resource during frame materialization. Resource ID: " + std::to_string(id));
						}

						if (onlyResourceIDs && !aliasPlacementRequiredThisFrame) {
							spdlog::debug(
								"RG alias fallback materialize: id={} name='{}' allowAlias=1 but no pool assignment this frame; materializing standalone",
								id,
								resource->GetName());
						}

						// Setup-time eager materialization happens before alias planning.
						// Defer aliased resources until compile-time placement is available.
						if (!onlyResourceIDs) {
							return std::nullopt;
						}
					}
					texture->Materialize();
				}
			}

			return texture->GetBackingGeneration();
		}

		auto buffer = dynamic_cast<Buffer*>(resource);
		if (!buffer) {
			return std::nullopt;
		}

		if (!buffer->IsMaterialized()) {
			auto itAlias = aliasMaterializeOptionsByID.find(id);
			if (itAlias != aliasMaterializeOptionsByID.end()) {
				if (std::holds_alternative<BufferBase::MaterializeOptions>(itAlias->second)) {
					auto& options = std::get<BufferBase::MaterializeOptions>(itAlias->second);
					if (options.aliasPlacement.has_value()) {
						const auto& ap = options.aliasPlacement.value();
						spdlog::info(
							"RG alias materialize (buffer): id={} name='{}' pool={} offset={}",
							id,
							resource->GetName(),
							ap.poolID.has_value() ? ap.poolID.value() : 0ull,
							ap.offset);
					}
					buffer->Materialize(&options);
				}
			}
			else {
				if (buffer->IsAliasingAllowed()) {
					const bool hasManualAliasPool = buffer->GetAliasingPoolHint().has_value();
					const bool hasAutoAliasPool = autoAliasPoolByID.find(id) != autoAliasPoolByID.end();
					const bool aliasPlacementRequiredThisFrame = hasManualAliasPool || hasAutoAliasPool;

					if (onlyResourceIDs && aliasPlacementRequiredThisFrame) {
						throw std::runtime_error(
							"Aliasing placement missing for used aliased buffer during frame materialization. Resource ID: " + std::to_string(id));
					}

					if (onlyResourceIDs && !aliasPlacementRequiredThisFrame) {
						spdlog::debug(
							"RG alias fallback materialize (buffer): id={} name='{}' allowAlias=1 but no pool assignment this frame; materializing standalone",
							id,
							resource->GetName());
					}

					if (!onlyResourceIDs) {
						return std::nullopt;
					}
				}
				buffer->Materialize();
			}
		}

		return buffer->GetBackingGeneration();
	};

	// Collect all unique {id, resource*} items to materialize
	std::vector<std::pair<uint64_t, Resource*>> items;
	items.reserve(resourcesByID.size() + m_transientFrameResourcesByID.size());
	std::unordered_set<uint64_t> seen;

	for (auto& [id, resource] : resourcesByID) {
		if (seen.insert(id).second && resource) {
			items.emplace_back(id, resource.get());
		}
	}

	for (auto& [id, resource] : m_transientFrameResourcesByID) {
		if (resourcesByID.contains(id)) continue;
		if (seen.insert(id).second && resource) {
			items.emplace_back(id, resource.get());
		}
	}

	auto collectFromHandle = [&](const ResourceRegistry::RegistryHandle& handle) {
		const uint64_t id = handle.GetGlobalResourceID();
		if (!seen.insert(id).second) return;
		Resource* resource = handle.IsEphemeral() ? handle.GetEphemeralPtr() : _registry.Resolve(handle);
		if (resource) {
			TrackTransientFrameResource(resource);
			items.emplace_back(id, resource);
		}
	};

	for (const auto& pr : m_framePasses) {
		if (pr.type == PassType::Compute) {
			auto const& p = std::get<ComputePassAndResources>(pr.pass);
			ForEachFrameRequirement(p.resources, [&](const auto& req) {
				collectFromHandle(req.resourceHandleAndRange.resource);
			});
			for (auto const& t : p.resources.internalTransitions) {
				collectFromHandle(t.first.resource);
			}
		}
		else if (pr.type == PassType::Render) {
			auto const& p = std::get<RenderPassAndResources>(pr.pass);
			ForEachFrameRequirement(p.resources, [&](const auto& req) {
				collectFromHandle(req.resourceHandleAndRange.resource);
			});
			for (auto const& t : p.resources.internalTransitions) {
				collectFromHandle(t.first.resource);
			}
		}
		else if (pr.type == PassType::Copy) {
			auto const& p = std::get<CopyPassAndResources>(pr.pass);
			ForEachFrameRequirement(p.resources, [&](const auto& req) {
				collectFromHandle(req.resourceHandleAndRange.resource);
			});
			for (auto const& t : p.resources.internalTransitions) {
				collectFromHandle(t.first.resource);
			}
		}
	}

	// Parallel materialize phase
	struct GenerationResult {
		uint64_t id = 0;
		uint64_t generation = 0;
		bool valid = false;
	};
	std::vector<GenerationResult> genResults(items.size());

	ParallelForOptional("Materialize", items.size(), [&](size_t i) {
		auto [id, resource] = items[i];
		auto gen = materializeOne(id, resource);
		if (gen.has_value()) {
			genResults[i] = { id, gen.value(), true };
		}
	}, true); // Disable or now, resource creation creates flecs entities in renderer

	// Merge generation results
	for (auto& r : genResults) {
		if (r.valid) {
			resourceBackingGenerationByID[r.id] = r.generation;
		}
	}
}

void RenderGraph::ResizeQueueParallelVectors() {
	const size_t qc = m_queueRegistry.SlotCount();
	m_compiledLastProducerBatchByResourceByQueue.resize(qc);
	m_hasPendingFrameStartQueueWait.assign(qc, std::vector<uint8_t>(qc, 0));
	m_pendingFrameStartQueueWaitFenceValue.assign(qc, std::vector<UINT64>(qc, 0));
}

void RenderGraph::EnsureMinimumAutomaticSchedulingQueues() {
	auto autoAssignableCountForKind = [this](QueueKind kind) {
		uint8_t count = 0;
		for (size_t i = 0; i < m_queueRegistry.SlotCount(); ++i) {
			const auto slotIndex = static_cast<QueueSlotIndex>(static_cast<uint8_t>(i));
			if (m_queueRegistry.GetKind(slotIndex) == kind && m_queueRegistry.IsAutoAssignable(slotIndex)) {
				++count;
			}
		}
		return count;
	};

	auto queueNamePrefix = [](QueueKind kind) -> const char* {
		switch (kind) {
		case QueueKind::Graphics: return "AutoGraphics";
		case QueueKind::Compute: return "AutoCompute";
		case QueueKind::Copy: return "AutoCopy";
		default: return "AutoQueue";
		}
	};

	for (size_t kindIndex = 0; kindIndex < static_cast<size_t>(QueueKind::Count); ++kindIndex) {
		const QueueKind kind = static_cast<QueueKind>(kindIndex);
		const uint8_t minimumQueueCount = m_minAutomaticSchedulingQueuesByKind[kindIndex];
		uint8_t currentAutoQueueCount = autoAssignableCountForKind(kind);
		while (currentAutoQueueCount < minimumQueueCount) {
			std::string queueName = std::string(queueNamePrefix(kind)) + std::to_string(currentAutoQueueCount);
			const uint8_t previousAutoQueueCount = currentAutoQueueCount;
			CreateQueue(kind, queueName.c_str(), QueueAutoAssignmentPolicy::AllowAutomaticScheduling);
			currentAutoQueueCount = autoAssignableCountForKind(kind);
			if (currentAutoQueueCount <= previousAutoQueueCount) {
				spdlog::warn(
					"RenderGraph: requested {} automatic queues for kind {}, but the RHI did not provide a distinct native queue; using {} queue(s).",
					minimumQueueCount,
					static_cast<int>(kind),
					currentAutoQueueCount);
				break;
			}
		}
	}

	ResizeQueueParallelVectors();
}

void RenderGraph::Setup() {
	DeletionManager::GetInstance().Initialize();

	// Setup the statistics manager
	if (m_statisticsService) {
		m_statisticsService->ClearAll();
	}
	auto& manager = DeviceManager::GetInstance();
	if (m_statisticsService) {
		m_statisticsService->RegisterQueue(manager.GetGraphicsQueue().GetKind());
		m_statisticsService->RegisterQueue(manager.GetComputeQueue().GetKind());
		m_statisticsService->SetupQueryHeap();
	}

	auto device = DeviceManager::GetInstance().GetDevice();

	auto result = device.CreateTimeline(m_readbackFence);
	result = device.CreateTimeline(m_copyReadbackFence);
	result = device.CreateTimeline(m_frameStartSyncFence);

	if (m_readbackService) {
		m_readbackService->Initialize(m_readbackFence.Get(), m_copyReadbackFence.Get());
	}

	// Populate the queue registry with the 3 primary queues.
	// Each Register() call creates a CommandListPool and Timeline for the slot.
	{
		auto& gfxQ = DeviceManager::GetInstance().GetGraphicsQueue();
		auto& compQ = DeviceManager::GetInstance().GetComputeQueue();
		auto& copyQ = DeviceManager::GetInstance().GetCopyQueue();
		m_queueRegistry.Register({ QueueKind::Graphics, 0 }, gfxQ, device);
		m_queueRegistry.Register({ QueueKind::Compute, 0 }, compQ, device);
		m_queueRegistry.Register({ QueueKind::Copy,    0 }, copyQ, device);
	}
	EnsureMinimumAutomaticSchedulingQueues();

	// Size queue-parallel member vectors to match the registry.
	// Done after both primary queue registration and extension initialization
	// since extensions may create additional queues.
	ResizeQueueParallelVectors();

	// Notify extensions that the render graph is set up.
	// Extensions may create additional queues or allocate resources here.
	for (auto& ext : m_extensions) {
		if (ext) ext->Initialize(*this);
	}

	// Re-size in case extensions added queues.
	ResizeQueueParallelVectors();

	m_getUseAsyncCompute = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetUseAsyncCompute() : false;
	};
	m_getHeavyDebug = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetHeavyDebug() : false;
	};
	m_getRenderGraphCompileDumpEnabled = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetRenderGraphCompileDumpEnabled() : false;
	};
	m_getRenderGraphVramDumpEnabled = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetRenderGraphVramDumpEnabled() : false;
	};
	m_getRenderGraphBatchTraceEnabled = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetRenderGraphBatchTraceEnabled() : false;
	};
	m_getRenderGraphLightweightCompileSummaryEnabled = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetRenderGraphLightweightCompileSummaryEnabled() : false;
	};
	m_getReadOnlyUniformTransitionElisionEnabled = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetReadOnlyUniformTransitionElisionEnabled() : false;
	};

	m_getAutoAliasMode = [this]() {
		const auto mode = m_renderGraphSettingsService
			? m_renderGraphSettingsService->GetAutoAliasMode()
			: static_cast<uint8_t>(AutoAliasMode::Off);
		return static_cast<AutoAliasMode>(mode);
	};
	m_getAutoAliasPackingStrategy = [this]() {
		const auto strategy = m_renderGraphSettingsService
			? m_renderGraphSettingsService->GetAutoAliasPackingStrategy()
			: static_cast<uint8_t>(AutoAliasPackingStrategy::GreedySweepLine);
		return static_cast<AutoAliasPackingStrategy>(strategy);
	};
	m_getAutoAliasEnableLogging = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetAutoAliasEnableLogging() : false;
	};
	m_getAutoAliasLogExclusionReasons = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetAutoAliasLogExclusionReasons() : false;
	};
	m_getAutoAliasBuildDebugData = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetAutoAliasBuildDebugData() : false;
	};
	m_getQueueSchedulingEnableLogging = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetQueueSchedulingEnableLogging() : false;
	};
	m_getQueueSchedulingWidthScale = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetQueueSchedulingWidthScale() : 1.0f;
	};
	m_getQueueSchedulingPenaltyBias = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetQueueSchedulingPenaltyBias() : 0.0f;
	};
	m_getQueueSchedulingMinPenalty = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetQueueSchedulingMinPenalty() : 1.0f;
	};
	m_getQueueSchedulingResourcePressureWeight = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetQueueSchedulingResourcePressureWeight() : 1.0f;
	};
	m_getQueueSchedulingUavPressureWeight = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetQueueSchedulingUavPressureWeight() : 0.5f;
	};
	m_getQueueSchedulingAutoGraphicsBias = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetQueueSchedulingAutoGraphicsBias() : 2.5f;
	};
	m_getQueueSchedulingAsyncOverlapBonus = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetQueueSchedulingAsyncOverlapBonus() : 3.0f;
	};
	m_getQueueSchedulingCrossQueueHandoffPenalty = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetQueueSchedulingCrossQueueHandoffPenalty() : 2.0f;
	};
	m_getRenderGraphRegionMode = [this]() {
		return m_renderGraphSettingsService
			? m_renderGraphSettingsService->GetRenderGraphRegionMode()
			: rg::runtime::RenderGraphRegionMode::Disabled;
	};
	m_getTransitionPlacementMode = [this]() {
		return m_renderGraphSettingsService
			? m_renderGraphSettingsService->GetTransitionPlacementMode()
			: rg::runtime::TransitionPlacementMode::InlineEarlyPlacement;
	};
	m_getRenderGraphRegionMinPassCount = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetRenderGraphRegionMinPassCount() : 4u;
	};
	m_getRenderGraphRegionMaxPassCount = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetRenderGraphRegionMaxPassCount() : 0u;
	};
	m_getRenderGraphRegionDiagnosticsEnabled = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetRenderGraphRegionDiagnosticsEnabled() : false;
	};
	m_getRenderGraphRegionShadowStrictBatchMatch = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetRenderGraphRegionShadowStrictBatchMatch() : false;
	};
	m_getAutoAliasPoolRetireIdleFrames = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetAutoAliasPoolRetireIdleFrames() : 120u;
	};
	m_getAutoAliasPoolGrowthHeadroom = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetAutoAliasPoolGrowthHeadroom() : 1.5f;
	};
	MaterializeUnmaterializedResources();
	// Pass setup is intentionally serial. Many passes touch shared ECS/flecs world state
	// and singleton managers during Setup(), and iterating flecs queries from our task
	// worker threads can corrupt flecs iterator stack state.
	ParallelForOptional("PassSetup", m_masterPassList.size(), [this](size_t i) {
		const bool traceLifecycle = m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled();
		auto& pass = m_masterPassList[i];
		switch (pass.type) {
		case PassType::Render: {
			auto& renderPass = std::get<RenderPassAndResources>(pass.pass);
			if (traceLifecycle) {
				spdlog::info("RG setup render pass '{}' begin", renderPass.name);
			}
			renderPass.pass->SetResourceRegistryView(
				std::make_unique<ResourceRegistryView>(_registry, renderPass.resources.identifierSet),
				renderPass.resources.activeFeatureDomains,
				renderPass.resources.autoDescriptorShaderResources,
				renderPass.resources.autoDescriptorConstantBuffers,
				renderPass.resources.autoDescriptorUnorderedAccessViews);
			renderPass.pass->Setup();
			if (traceLifecycle) {
				spdlog::info("RG setup render pass '{}' complete", renderPass.name);
			}
			break;
		}
		case PassType::Compute: {
			auto& computePass = std::get<ComputePassAndResources>(pass.pass);
			if (traceLifecycle) {
				spdlog::info("RG setup compute pass '{}' begin", computePass.name);
			}
			computePass.pass->SetResourceRegistryView(
				std::make_unique<ResourceRegistryView>(_registry, computePass.resources.identifierSet),
				computePass.resources.activeFeatureDomains,
				computePass.resources.autoDescriptorShaderResources,
				computePass.resources.autoDescriptorConstantBuffers,
				computePass.resources.autoDescriptorUnorderedAccessViews);
			computePass.pass->Setup();
			if (traceLifecycle) {
				spdlog::info("RG setup compute pass '{}' complete", computePass.name);
			}
			break;
		}
		case PassType::Copy: {
			auto& copyPass = std::get<CopyPassAndResources>(pass.pass);
			if (traceLifecycle) {
				spdlog::info("RG setup copy pass '{}' begin", copyPass.name);
			}
			copyPass.pass->SetResourceRegistryView(std::make_unique<ResourceRegistryView>(_registry, copyPass.resources.identifierSet));
			copyPass.pass->Setup();
			if (traceLifecycle) {
				spdlog::info("RG setup copy pass '{}' complete", copyPass.name);
			}
			break;
		}
		}
	}, true);
}

void RenderGraph::AddRenderPass(std::shared_ptr<RenderPass> pass, RenderPassParameters& resources, std::string name, std::vector<ResolverSnapshot> resolverSnapshots) {
	RenderPassAndResources passAndResources;
	passAndResources.pass = pass;
	passAndResources.resources = resources;
	passAndResources.name = name;
	passAndResources.techniquePath = GetTechniquePathForPassName(name);
	passAndResources.retainedAnonymousKeepAlive = CaptureRetainedAnonymousKeepAlive(
		passAndResources.resources.staticResourceRequirements,
		passAndResources.resources.internalTransitions);
	passAndResources.resolverSnapshots = std::move(resolverSnapshots);
	UpdateRetainedDeclarationCache(_registry, passAndResources);
	AnyPassAndResources passAndResourcesAny;
	passAndResourcesAny.type = PassType::Render;
	passAndResourcesAny.pass = std::move(passAndResources);
	passAndResourcesAny.name = name;
	m_masterPassList.push_back(std::move(passAndResourcesAny));
	if (name != "") {
		renderPassesByName[name] = pass;
	}
}

void RenderGraph::AddComputePass(std::shared_ptr<ComputePass> pass, ComputePassParameters& resources, std::string name, std::vector<ResolverSnapshot> resolverSnapshots) {
	ComputePassAndResources passAndResources;
	passAndResources.pass = pass;
	passAndResources.resources = resources;
	passAndResources.name = name;
	passAndResources.techniquePath = GetTechniquePathForPassName(name);
	passAndResources.retainedAnonymousKeepAlive = CaptureRetainedAnonymousKeepAlive(
		passAndResources.resources.staticResourceRequirements,
		passAndResources.resources.internalTransitions);
	passAndResources.resolverSnapshots = std::move(resolverSnapshots);
	UpdateRetainedDeclarationCache(_registry, passAndResources);
	AnyPassAndResources passAndResourcesAny;
	passAndResourcesAny.type = PassType::Compute;
	passAndResourcesAny.pass = std::move(passAndResources);
	passAndResourcesAny.name = name;
	m_masterPassList.push_back(std::move(passAndResourcesAny));
	if (name != "") {
		computePassesByName[name] = pass;
	}
}

void RenderGraph::AddCopyPass(std::shared_ptr<CopyPass> pass, CopyPassParameters& resources, std::string name, std::vector<ResolverSnapshot> resolverSnapshots) {
	CopyPassAndResources passAndResources;
	passAndResources.pass = pass;
	passAndResources.resources = resources;
	passAndResources.name = name;
	passAndResources.techniquePath = GetTechniquePathForPassName(name);
	passAndResources.retainedAnonymousKeepAlive = CaptureRetainedAnonymousKeepAlive(
		passAndResources.resources.staticResourceRequirements,
		passAndResources.resources.internalTransitions);
	passAndResources.resolverSnapshots = std::move(resolverSnapshots);
	UpdateRetainedDeclarationCache(_registry, passAndResources);
	AnyPassAndResources passAndResourcesAny;
	passAndResourcesAny.type = PassType::Copy;
	passAndResourcesAny.pass = std::move(passAndResources);
	passAndResourcesAny.name = name;
	m_masterPassList.push_back(std::move(passAndResourcesAny));
}

void RenderGraph::SetPassTechnique(std::string passName, std::string techniquePath) {
	if (passName.empty()) {
		return;
	}

	if (techniquePath.empty()) {
		m_passTechniquePathsByName.erase(passName);
		return;
	}

	m_passTechniquePathsByName[std::move(passName)] = std::move(techniquePath);
}

std::string RenderGraph::GetTechniquePathForPassName(std::string_view passName) const {
	if (passName.empty()) {
		return {};
	}

	auto it = m_passTechniquePathsByName.find(std::string(passName));
	if (it == m_passTechniquePathsByName.end()) {
		return {};
	}

	return it->second;
}

void RenderGraph::AddResource(std::shared_ptr<Resource> resource, bool transition) {
	const uint64_t resourceID = resource->GetGlobalResourceID();
	if (auto dynamicResource = std::dynamic_pointer_cast<DynamicResource>(resource)) {
		m_dynamicResourcesByStableID[dynamicResource->GetDynamicWrapperGlobalResourceID()] = resource;
	}
	if (resourcesByID.contains(resourceID)) {
		return; // Resource already added
	}
	auto& name = resource->GetName();
	const bool traceLifecycle = m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled();

#ifdef _DEBUG
	//if (name == L"") {
	//	throw std::runtime_error("Resource name cannot be empty");
	//}
	//else if (resourcesByName.find(name) != resourcesByName.end()) {
	//	throw std::runtime_error("Resource with name " + ws2s(name) + " already exists");
	//}
#endif

	resourcesByName[name] = resource;
	resourcesByID[resourceID] = resource;
	if (traceLifecycle) {
		const char* resourceType = "Resource";
		if (std::dynamic_pointer_cast<PixelBuffer>(resource)) {
			resourceType = "PixelBuffer";
		}
		else if (std::dynamic_pointer_cast<Buffer>(resource)) {
			resourceType = "Buffer";
		}
		spdlog::info(
			"RenderGraph::AddResource type={} name='{}' id={} transition={}",
			resourceType,
			name,
			resourceID,
			transition);
	}

	if (auto texture = std::dynamic_pointer_cast<PixelBuffer>(resource)) {
		texture->EnsureVirtualDescriptorSlotsAllocated();
	}
	if (auto buffer = std::dynamic_pointer_cast<Buffer>(resource)) {
		buffer->EnsureVirtualDescriptorSlotsAllocated();
	}
}

void RenderGraph::TrackTransientFrameResource(const std::shared_ptr<Resource>& resource) {
	if (!resource) {
		return;
	}

	const uint64_t resourceID = resource->GetGlobalResourceID();
	if (auto dynamicResource = std::dynamic_pointer_cast<DynamicResource>(resource)) {
		m_dynamicResourcesByStableID[dynamicResource->GetDynamicWrapperGlobalResourceID()] = resource;
	}
	if (resourcesByID.contains(resourceID)) {
		return;
	}

	m_transientFrameResourcesByID[resourceID] = resource;
	const auto& resourceName = resource->GetName();
	if (!resourceName.empty()) {
		m_transientFrameResourcesByName[resourceName] = resource;
	}
}

void RenderGraph::TrackTransientFrameResource(Resource* resource) {
	if (!resource) {
		return;
	}

	if (auto shared = resource->weak_from_this().lock()) {
		TrackTransientFrameResource(shared);
	}
}

std::shared_ptr<Resource> RenderGraph::GetResourceByName(const std::string& name) {
	auto transientIt = m_transientFrameResourcesByName.find(name);
	if (transientIt != m_transientFrameResourcesByName.end()) {
		return transientIt->second;
	}
	auto it = resourcesByName.find(name);
	if (it != resourcesByName.end()) {
		return it->second;
	}
	return nullptr;
}

std::shared_ptr<Resource> RenderGraph::GetResourceByID(const uint64_t id) {
	auto transientIt = m_transientFrameResourcesByID.find(id);
	if (transientIt != m_transientFrameResourcesByID.end()) {
		return transientIt->second;
	}
	auto dynamicIt = m_dynamicResourcesByStableID.find(id);
	if (dynamicIt != m_dynamicResourcesByStableID.end()) {
		return dynamicIt->second;
	}
	auto it = resourcesByID.find(id);
	if (it != resourcesByID.end()) {
		return it->second;
	}
	return nullptr;
}
std::shared_ptr<RenderPass> RenderGraph::GetRenderPassByName(const std::string& name) {
	if (renderPassesByName.find(name) != renderPassesByName.end()) {
		return renderPassesByName[name];
	}
	else {
		return nullptr;
	}
}

std::shared_ptr<ComputePass> RenderGraph::GetComputePassByName(const std::string& name) {
	if (computePassesByName.find(name) != computePassesByName.end()) {
		return computePassesByName[name];
	}
	else {
		return nullptr;
	}
}

void RenderGraph::Update(const UpdateExecutionContext& context, rhi::Device device) {
	ZoneScopedN("RenderGraph::Update");
	const bool traceLifecycle = m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled();
	{
		ZoneScopedN("RenderGraph::Update::ResetForFrame");
		ResetForFrame();
	}

	// Poll readback completions early so that passes (e.g. CLod streaming)
	// can react to GPU-produced data from the previous frame immediately,
	// rather than waiting until post-Present.
	if (m_readbackService) {
		ZoneScopedN("RenderGraph::Update::ProcessReadbacks");
		m_readbackService->ProcessReadbackRequests();
	}

	if (m_statisticsService) {
		ZoneScopedN("RenderGraph::Update::BeginStatisticsFrame");
		m_statisticsService->BeginFrame();
	}

	auto toMilliseconds = [](auto duration) {
		return std::chrono::duration<double, std::milli>(duration).count();
	};

	{
		ZoneScopedN("RenderGraph::Update::PassUpdates");
		for (auto& pr : m_masterPassList) {	
			// Resolve into type and update
			std::visit([&](auto& obj) {
				using T = std::decay_t<decltype(obj)>;
				if constexpr (std::is_same_v<T, std::monostate>) {
					// no-op
				}
				else {
					ZoneScopedN("RenderGraph::Update::PassUpdate");
					if (!obj.name.empty()) {
						ZoneText(obj.name.data(), obj.name.size());
					}
					if (traceLifecycle) {
						spdlog::info("RG frame {} pass update '{}' begin", context.frameIndex, obj.name);
					}

					if (!obj.collectStatistics) {
						obj.statisticsIndex = -1;
					}
					else if (m_statisticsService && obj.statisticsIndex < 0) {
						if constexpr (std::is_same_v<T, RenderPassAndResources>) {
							obj.statisticsIndex = static_cast<int>(m_statisticsService->RegisterPass(obj.name, obj.resources.isGeometryPass, obj.techniquePath));
						}
						else {
							obj.statisticsIndex = static_cast<int>(m_statisticsService->RegisterPass(obj.name, false, obj.techniquePath));
						}
					}

					const auto start = std::chrono::steady_clock::now();
					obj.pass->Update(context);
					if (traceLifecycle) {
						spdlog::info("RG frame {} pass update '{}' complete", context.frameIndex, obj.name);
					}
					if (m_statisticsService && obj.statisticsIndex >= 0) {
						m_statisticsService->RecordCpuUpdateTime(
							static_cast<unsigned>(obj.statisticsIndex),
							toMilliseconds(std::chrono::steady_clock::now() - start));
					}
				}
				}, pr.pass);
		}
	}

	{
		ZoneScopedN("RenderGraph::Update::CompileFrame");
		CompileFrame(device, context.frameIndex, context.hostData);
	}
}

#define IFDEBUG(x) 

namespace {
	bool ResolveNonEmptyTransitionRange(const ResourceTransition& transition, SubresourceRange& outRange)
	{
		if (!transition.pResource || !transition.pResource->HasLayout()) {
			return false;
		}

		outRange = ResolveRangeSpec(
			transition.range,
			transition.pResource->GetMipLevels(),
			transition.pResource->GetArraySize());
		return !outRange.isEmpty();
	}

	// ExecuteTransitions: applies state-tracker bookkeeping AND records
	// barriers.  Used only in the non-parallel fallback path.
	void ExecuteTransitions(std::vector<ResourceTransition>& transitions,
		CommandRecordingManager* crm,
		QueueKind queueKind,
		rhi::CommandList& commandList) {
		rhi::helpers::OwnedBarrierBatch batch;
		for (auto& transition : transitions) {
			SubresourceRange resolvedRange{};
			if (transition.pResource->HasLayout() && !ResolveNonEmptyTransitionRange(transition, resolvedRange)) {
				continue;
			}

			std::vector<ResourceTransition> dummy;
			transition.pResource->GetStateTracker()->Apply(
				transition.range, transition.pResource,
				{ transition.newAccessType, transition.newLayout, transition.newSyncState }, dummy);
			auto bg = transition.pResource->GetEnhancedBarrierGroup(
				transition.range, transition.prevAccessType, transition.newAccessType,
				transition.prevLayout, transition.newLayout,
				transition.prevSyncState, transition.newSyncState);
			const size_t textureStart = batch.textures.size();
			const size_t bufferStart = batch.buffers.size();
			batch.Append(bg);
			if (transition.discard) {
				for (size_t i = textureStart; i < batch.textures.size(); ++i) {
					batch.textures[i].discard = true;
				}
				for (size_t i = bufferStart; i < batch.buffers.size(); ++i) {
					batch.buffers[i].discard = true;
				}
			}
		}
		if (!batch.Empty()) {
			commandList.Barriers(batch.View());
		}
	}

	// RecordTransitionBarriers: records barrier commands into a CL
	void RecordTransitionBarriers(std::vector<ResourceTransition>& transitions,
		rhi::CommandList& commandList) {
		if (transitions.empty()) return;

		rhi::helpers::OwnedBarrierBatch batch;
		batch.textures.reserve(transitions.size());
		batch.buffers.reserve(transitions.size());

		for (auto& t : transitions) {
			if (t.pResource->HasLayout()) {
				// Texture barrier
				SubresourceRange resolvedRange{};
				if (!ResolveNonEmptyTransitionRange(t, resolvedRange)) {
					continue;
				}

				rhi::TextureBarrier tb{};
				tb.beforeAccess = t.prevAccessType;
				tb.afterAccess  = t.newAccessType;
				tb.beforeLayout = t.prevLayout;
				tb.afterLayout  = t.newLayout;
				tb.beforeSync   = t.prevSyncState;
				tb.afterSync    = t.newSyncState;
				tb.discard      = t.discard;
				tb.range = { resolvedRange.firstMip, resolvedRange.mipCount,
				             resolvedRange.firstSlice, resolvedRange.sliceCount };
				tb.texture = t.pResource->GetAPIResource().GetHandle();
				batch.textures.push_back(tb);
			} else {
				// Buffer barrier
				rhi::BufferBarrier bb{};
				bb.buffer       = t.pResource->GetAPIResource().GetHandle();
				bb.offset       = 0;
				bb.size         = UINT64_MAX;
				bb.beforeSync   = t.prevSyncState;
				bb.afterSync    = t.newSyncState;
				bb.beforeAccess = t.prevAccessType;
				bb.afterAccess  = t.newAccessType;
				bb.discard      = t.discard;
				batch.buffers.push_back(bb);
			}
		}

		if (!batch.Empty()) {
			commandList.Barriers(batch.View());
		}
	}

	// Signal external fences on the queue. Must be called AFTER the command list
	struct ExternalFenceSignalKey {
		uint32_t index = 0;
		uint32_t generation = 0;
		uint64_t value = 0;

		bool operator==(const ExternalFenceSignalKey& other) const noexcept {
			return index == other.index && generation == other.generation && value == other.value;
		}
	};

	struct ExternalFenceSignalKeyHash {
		size_t operator()(const ExternalFenceSignalKey& key) const noexcept {
			size_t seed = static_cast<size_t>(key.index);
			seed ^= static_cast<size_t>(key.generation) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
			seed ^= static_cast<size_t>(key.value) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
			seed ^= static_cast<size_t>(key.value >> 32) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
			return seed;
		}
	};

	struct ExternalFenceSignalOrigin {
		QueueKind queueKind = QueueKind::Graphics;
		size_t queueSlot = 0;
		size_t batchIndex = 0;
		std::string passName;
	};

	uint64_t PackTimelineSignalKey(rhi::TimelineHandle handle) noexcept {
		return (static_cast<uint64_t>(handle.index) << 32) | static_cast<uint64_t>(handle.generation);
	}

	void LogQueuedExternalFence(
		unsigned frameIndex,
		QueueKind queueKind,
		size_t queueSlot,
		size_t batchIndex,
		std::string_view passName,
		std::unordered_map<ExternalFenceSignalKey, ExternalFenceSignalOrigin, ExternalFenceSignalKeyHash>& queuedSignals,
		const PassReturn& passReturn)
	{
		if (!passReturn.fence.has_value() && passReturn.externalSignalsAfterCompletion.empty()) {
			return;
		}
		auto logSignal = [&](rhi::TimelineHandle handle, uint64_t value) {
			ExternalFenceSignalKey key{
				.index = handle.index,
				.generation = handle.generation,
				.value = value,
			};
			auto [it, inserted] = queuedSignals.emplace(
				key,
				ExternalFenceSignalOrigin{
					.queueKind = queueKind,
					.queueSlot = queueSlot,
					.batchIndex = batchIndex,
					.passName = std::string(passName),
				});
			if (!inserted) {
				spdlog::error(
					"RenderGraph: duplicate external fence queue detected in frame {} for timeline(idx={}, gen={}) value={}. Previous pass='{}' queue={} slot={} batch={}; current pass='{}' queue={} slot={} batch={}",
					frameIndex,
					handle.index,
					handle.generation,
					value,
					it->second.passName,
					QueueKindToString(it->second.queueKind),
					it->second.queueSlot,
					it->second.batchIndex,
					passName,
					QueueKindToString(queueKind),
					queueSlot,
					batchIndex);
			}
			spdlog::info(
				"RenderGraph: frame {} queued external fence from pass '{}' on queue {} slot {} batch {} timeline(idx={}, gen={}) value={}",
				frameIndex,
				passName,
				QueueKindToString(queueKind),
				queueSlot,
				batchIndex,
				handle.index,
				handle.generation,
				value);
		};

		if (passReturn.fence.has_value()) {
			logSignal(passReturn.fence->GetHandle(), passReturn.fenceValue);
		}
		for (const auto& signal : passReturn.externalSignalsAfterCompletion) {
			if (signal.timeline.IsValid()) {
				logSignal(signal.timeline.GetHandle(), signal.value);
			}
		}
	}

	// containing the pass work has been flushed (submitted) so the signals fire
	// after the GPU work they depend on.
	void SignalExternalFences(
		rhi::Queue& queue,
		QueueKind queueKind,
		rhi::Timeline* slotFence,
		size_t queueSlot,
		size_t batchIndex,
		unsigned frameIndex,
		std::span<const uint64_t> queueSlotFenceTimelineKeys,
		std::unordered_map<ExternalFenceSignalKey, ExternalFenceSignalOrigin, ExternalFenceSignalKeyHash>& seenSignals,
		std::unordered_map<uint64_t, uint64_t>& lastExternalSignalValueByTimeline,
		std::vector<PassReturn>& externalFences) {
		ZoneScopedN("RenderGraph::SignalExternalFences");
		if (externalFences.empty()) return;
		for (auto& fr : externalFences) {
			if (!fr.externalSignalsAfterCompletion.empty()) {
				for (const auto& signal : fr.externalSignalsAfterCompletion) {
					if (!signal.timeline.IsValid()) {
						spdlog::warn("Pass returned an invalid external signal timeline. Skipping signal.");
						continue;
					}
					PassReturn singleSignal{};
					singleSignal.fence = signal.timeline;
					singleSignal.fenceValue = signal.value;
					std::vector<PassReturn> nested;
					nested.push_back(std::move(singleSignal));
					SignalExternalFences(
						queue,
						queueKind,
						slotFence,
						queueSlot,
						batchIndex,
						frameIndex,
						queueSlotFenceTimelineKeys,
						seenSignals,
						lastExternalSignalValueByTimeline,
						nested);
				}
			}
			if (!fr.fence.has_value()) {
				continue;
			}
            else {
				if (fr.fenceValue == 0) {
					auto h = fr.fence.value().GetHandle();
					spdlog::error(
						"SignalExternalFences: pass returned fence (index={}, gen={}) "
						"with value 0 - this will violate monotonic signal ordering. "
						"Skipping signal.",
						h.index, h.generation);
					continue;
				}
				if (fr.fenceValue == UINT64_MAX) {
					auto h = fr.fence.value().GetHandle();
					spdlog::error(
						"SignalExternalFences: pass returned fence (index={}, gen={}) with terminal value UINT64_MAX. Skipping signal.",
						h.index, h.generation);
					continue;
				}
				if (slotFence) {
					auto a = fr.fence.value().GetHandle();
					auto b = slotFence->GetHandle();
					if (a.index == b.index && a.generation == b.generation) {
						spdlog::error(
							"SignalExternalFences: external signal aliases queue slot fence timeline "
							"(idx={}, gen={}) value={} frame={} queue={} slot={} batch={}. "
							"Skipping; queue-slot completion signals already publish this timeline.",
							a.index,
							a.generation,
							fr.fenceValue,
							frameIndex,
							QueueKindToString(queueKind),
							queueSlot,
							batchIndex);
						continue;
					}
				}
				auto handle = fr.fence.value().GetHandle();
				const uint64_t timelineKey = PackTimelineSignalKey(handle);
				bool aliasesQueueSlotFence = false;
				for (size_t aliasedSlot = 0; aliasedSlot < queueSlotFenceTimelineKeys.size(); ++aliasedSlot) {
					if (queueSlotFenceTimelineKeys[aliasedSlot] != timelineKey) {
						continue;
					}
					spdlog::error(
						"SignalExternalFences: external signal aliases queue slot fence timeline "
						"(idx={}, gen={}) value={} frame={} queue={} slot={} batch={} aliasedSlot={}. "
						"Skipping; queue-slot completion signals already publish this timeline.",
						handle.index,
						handle.generation,
						fr.fenceValue,
						frameIndex,
						QueueKindToString(queueKind),
						queueSlot,
						batchIndex,
						aliasedSlot);
					aliasesQueueSlotFence = true;
					break;
				}
				if (aliasesQueueSlotFence) {
					continue;
				}
				if (fr.fence.value().GetCompletedValue() >= fr.fenceValue) {
					spdlog::debug(
						"SignalExternalFences: skipping already-completed external signal frame={} queue={} slot={} batch={} timeline(idx={}, gen={}) value={}",
						frameIndex,
						static_cast<int>(queueKind),
						queueSlot,
						batchIndex,
						handle.index,
						handle.generation,
						fr.fenceValue);
					continue;
				}
				auto [lastExternalIt, insertedExternalLast] = lastExternalSignalValueByTimeline.try_emplace(
					timelineKey,
					0);
				if (fr.fenceValue <= lastExternalIt->second) {
					spdlog::warn(
						"SignalExternalFences: skipping stale external signal frame={} queue={} slot={} batch={} timeline(idx={}, gen={}) value={} lastSignaledByRenderGraph={}",
						frameIndex,
						QueueKindToString(queueKind),
						queueSlot,
						batchIndex,
						handle.index,
						handle.generation,
						fr.fenceValue,
						lastExternalIt->second);
					continue;
				}
				ExternalFenceSignalKey key{
					.index = handle.index,
					.generation = handle.generation,
					.value = fr.fenceValue,
				};
				auto [it, inserted] = seenSignals.emplace(
					key,
					ExternalFenceSignalOrigin{
						.queueKind = queueKind,
						.queueSlot = queueSlot,
						.batchIndex = batchIndex,
						.passName = std::string{},
					});
				if (!inserted) {
					spdlog::error(
						"SignalExternalFences: duplicate external signal detected in frame {} for timeline(idx={}, gen={}) value={}. Previous signal queue={} slot={} batch={}; current queue={} slot={} batch={}",
						frameIndex,
						handle.index,
						handle.generation,
						fr.fenceValue,
						QueueKindToString(it->second.queueKind),
						it->second.queueSlot,
						it->second.batchIndex,
						QueueKindToString(queueKind),
						queueSlot,
						batchIndex);
					continue;
				}
				spdlog::debug(
					"SignalExternalFences: frame={} queue={} slot={} batch={} signaling timeline(idx={}, gen={}) value={}",
					frameIndex,
					static_cast<int>(queueKind),
					queueSlot,
					batchIndex,
					handle.index,
					handle.generation,
					fr.fenceValue);
				const rhi::Result signalResult = queue.Signal({ fr.fence.value().GetHandle(), fr.fenceValue });
				if (signalResult != rhi::Result::Ok) {
					spdlog::warn(
						"SignalExternalFences: skipped external signal after queue rejected it frame={} queue={} slot={} batch={} timeline(idx={}, gen={}) value={} result={} completed={} renderGraphHighWater={}",
						frameIndex,
						QueueKindToString(queueKind),
						queueSlot,
						batchIndex,
						handle.index,
						handle.generation,
						fr.fenceValue,
						rhi::ResultName(signalResult),
						fr.fence.value().GetCompletedValue(),
						lastExternalIt->second);
					continue;
				}
				lastExternalIt->second = fr.fenceValue;
			}
		}
		externalFences.clear();
	}

	// ExecuteQueueBatch: unified per-queue-per-batch execution using pre-allocated
	// command lists from the execution schedule.
	void SignalQueueFenceOrThrow(
		rhi::Queue& queue,
		rhi::Timeline& timeline,
		UINT64 value,
		QueueKind queueKind,
		size_t queueSlot,
		size_t batchIndex,
		std::string_view phase,
		unsigned frameIndex)
	{
		if (value == 0 || value == UINT64_MAX) {
			std::ostringstream oss;
			oss << "RenderGraph: frame " << frameIndex
				<< " rejected invalid queue signal value for " << QueueKindToString(queueKind)
				<< " slot " << queueSlot
				<< " batch " << batchIndex
				<< " phase " << phase
				<< " fence(idx=" << timeline.GetHandle().index
				<< ", gen=" << timeline.GetHandle().generation
				<< ") value=" << value
				<< " completed=" << timeline.GetCompletedValue();
			spdlog::error(oss.str());
			throw std::runtime_error(oss.str());
		}
		const rhi::Result signalResult = queue.Signal({ timeline.GetHandle(), value });
		if (signalResult == rhi::Result::Ok) {
			return;
		}

		std::ostringstream oss;
		oss << "RenderGraph: frame " << frameIndex
			<< " queue signal failed for " << QueueKindToString(queueKind)
			<< " slot " << queueSlot
			<< " batch " << batchIndex
			<< " phase " << phase
			<< " fence(idx=" << timeline.GetHandle().index
			<< ", gen=" << timeline.GetHandle().generation
			<< ") value=" << value
			<< " completed=" << timeline.GetCompletedValue()
			<< " result=" << static_cast<uint32_t>(signalResult);
		spdlog::error(oss.str());
		throw std::runtime_error(oss.str());
	}

	void WaitExternalFencesBeforeTransitions(
		rhi::Queue queue,
		const RenderGraph::PassBatch& batch,
		size_t queueSlot,
		size_t batchIndex,
		unsigned frameIndex)
	{
		const auto& waits = batch.ExternalWaitsBeforeTransitions(queueSlot);
		for (const auto& wait : waits) {
			if (!wait.timeline.IsValid() || wait.value == 0 || wait.value == UINT64_MAX) {
				spdlog::error(
					"RenderGraph: frame {} rejected invalid external wait on queue slot {} batch {} value={}",
					frameIndex,
					queueSlot,
					batchIndex,
					wait.value);
				throw std::runtime_error("RenderGraph external wait was invalid");
			}
			const rhi::Result waitResult = queue.Wait({ wait.timeline.GetHandle(), wait.value });
			if (waitResult != rhi::Result::Ok) {
				throw std::runtime_error(fmt::format(
					"RenderGraph external wait failed: queueSlot={} batch={} value={} result={}",
					queueSlot,
					batchIndex,
					wait.value,
					rhi::ResultName(waitResult)));
			}
		}
	}

	struct ExecuteQueueBatchArgs {
		QueueBatchSchedule& sched;
		RenderGraph::PassBatch& batch;
		size_t batchIndex;
		QueueKind queue;
		size_t queueSlot;
		rhi::Queue& rhiQueue;
		rhi::Timeline& fenceTimeline;
		CommandListPool& pool;
		UINT64 fenceOffset;             // always 0 currently
		PassExecutionContext& context;
		rg::runtime::IStatisticsService* statisticsService;
		std::vector<PassReturn>& outExternalFences;
		std::unordered_map<ExternalFenceSignalKey, ExternalFenceSignalOrigin, ExternalFenceSignalKeyHash>& queuedExternalFenceOrigins;
		UINT64& lastSignaledOnTimeline;
		bool batchTraceEnabled;
	};

	void ExecuteQueueBatch(
		ExecuteQueueBatchArgs& args,
		auto&& WaitOnSlot)
	{
		ZoneScopedN("RenderGraph::ExecuteQueueBatch");
		auto& sched    = args.sched;
		auto& batch    = args.batch;
		auto  queue    = args.queue;
		auto  qi       = args.queueSlot;
		auto& rhiQueue = args.rhiQueue;
		auto& pool     = args.pool;
		auto  fenceOffset = args.fenceOffset;

		uint8_t clIndex = 0; // index into preallocatedCLs

		// Waits: BeforeTransitions
		WaitExternalFencesBeforeTransitions(
			rhiQueue,
			batch,
			qi,
			args.batchIndex,
			static_cast<unsigned>(args.context.frameIndex));
		for (size_t srcIndex = 0; srcIndex < batch.QueueCount(); ++srcIndex) {
			if (!batch.HasQueueWait(RenderGraph::BatchWaitPhase::BeforeTransitions, qi, srcIndex))
				continue;
			UINT64 val = fenceOffset + batch.GetQueueWaitFenceValue(
				RenderGraph::BatchWaitPhase::BeforeTransitions, qi, srcIndex);
			WaitOnSlot(qi, srcIndex, val);
		}

		// Open first CL and record pre-transitions
		auto& cl0 = sched.preallocatedCLs[clIndex];
		rhi::CommandList commandList = cl0.list.Get();

		auto& preTransitions = batch.Transitions(qi, RenderGraph::BatchTransitionPhase::BeforePasses);
		ExecuteTransitions(preTransitions, /*crm=*/nullptr, queue, commandList);

		// Waits: BeforeExecution
		for (size_t srcIndex = 0; srcIndex < batch.QueueCount(); ++srcIndex) {
			if (!batch.HasQueueWait(RenderGraph::BatchWaitPhase::BeforeExecution, qi, srcIndex))
				continue;
			UINT64 val = fenceOffset + batch.GetQueueWaitFenceValue(
				RenderGraph::BatchWaitPhase::BeforeExecution, qi, srcIndex);
			WaitOnSlot(qi, srcIndex, val);
		}

		// Split after transitions if needed
		if (sched.splitAfterTransitions) {
			UINT64 signalValue = fenceOffset + batch.GetQueueSignalFenceValue(
				RenderGraph::BatchSignalPhase::AfterTransitions, qi);
			commandList.End();
			rhiQueue.Submit({ &commandList, 1 }, {});
			SignalQueueFenceOrThrow(
				rhiQueue,
				args.fenceTimeline,
				signalValue,
				queue,
				qi,
				args.batchIndex,
				"AfterTransitions",
				static_cast<unsigned>(args.context.frameIndex));
			args.lastSignaledOnTimeline = std::max(args.lastSignaledOnTimeline, signalValue);
			pool.Recycle(std::move(cl0), signalValue);

			++clIndex;
			commandList = sched.preallocatedCLs[clIndex].list.Get();
		}

		// Record all passes into the current CL
		args.context.commandList = commandList;

		auto executeOne = [&](auto& pr) {
			if (!pr.pass->IsInvalidated())
				return;
			const std::string_view passName = pr.name.empty() ? std::string_view("<unnamed>") : std::string_view(pr.name);
			const char* techniquePath = pr.techniquePath.empty() ? nullptr : pr.techniquePath.c_str();
			try {
				ZoneScopedN("RenderGraph::ExecuteQueueBatch::PassExecute");
				ZoneText(passName.data(), passName.size());
					if (args.batchTraceEnabled) {
						spdlog::info(
							"RenderGraph: frame {} queue {} slot {} batch {} begin pass {}",
							static_cast<unsigned>(args.context.frameIndex),
							QueueKindToString(queue),
							qi,
							args.batchIndex,
							passName);
					}
				rhi::debug::Scope scope(commandList, rhi::colors::Mint, std::string(passName).c_str());
				args.context.currentPassName = passName.data();
				args.context.currentTechniquePath = techniquePath;
				(void)rhi::debug::SetInstrumentationContext(commandList, args.context.currentPassName, args.context.currentTechniquePath);
				const bool hasStatistics = args.statisticsService && pr.statisticsIndex >= 0;
				const auto cpuStart = std::chrono::steady_clock::now();
				if (hasStatistics)
					args.statisticsService->BeginQuery(pr.statisticsIndex, args.context.frameIndex, rhiQueue, commandList);
				if ((pr.run & PassRunMask::Immediate) != PassRunMask::None)
					rg::imm::Replay(pr.immediateBytecode, commandList, *args.context.immediateDispatch);
				pr.immediateKeepAlive.reset();
				if ((pr.run & PassRunMask::Retained) != PassRunMask::None) {
					auto passReturn = pr.pass->Execute(args.context);
					// In batch trace, do some fence debug
					if (args.batchTraceEnabled && (passReturn.fence || !passReturn.externalSignalsAfterCompletion.empty())) {
						LogQueuedExternalFence(
							static_cast<unsigned>(args.context.frameIndex),
							queue,
							qi,
							args.batchIndex,
							passName,
							args.queuedExternalFenceOrigins,
							passReturn);
					}
					if (passReturn.fence || !passReturn.externalSignalsAfterCompletion.empty()) {
						args.outExternalFences.push_back(passReturn);
					}
				}
				if (hasStatistics)
					args.statisticsService->EndQuery(pr.statisticsIndex, args.context.frameIndex, rhiQueue, commandList);
				if (hasStatistics) {
					args.statisticsService->RecordCpuExecuteTime(
						static_cast<unsigned>(pr.statisticsIndex),
						std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuStart).count());
				}
				(void)rhi::debug::SetInstrumentationContext(commandList, nullptr, nullptr);
				args.context.currentPassName = nullptr;
				args.context.currentTechniquePath = nullptr;
					if (args.batchTraceEnabled) {
						spdlog::info(
							"RenderGraph: frame {} queue {} slot {} batch {} end pass {}",
							static_cast<unsigned>(args.context.frameIndex),
							QueueKindToString(queue),
							qi,
							args.batchIndex,
							passName);
					}
			}
			catch (const std::exception& ex) {
				(void)rhi::debug::SetInstrumentationContext(commandList, nullptr, nullptr);
				args.context.currentPassName = nullptr;
				args.context.currentTechniquePath = nullptr;
				std::ostringstream oss;
				oss << "RenderGraph::ExecuteQueueBatch failed while executing pass '"
					<< passName
					<< "' on queue " << QueueKindToString(queue)
					<< " (slot " << qi << ", batch " << args.batchIndex << "): " << ex.what();
				spdlog::error(oss.str());
				throw std::runtime_error(oss.str());
			}
			catch (...) {
				std::ostringstream oss;
				oss << "RenderGraph::ExecuteQueueBatch failed while executing pass '"
					<< passName
					<< "' on queue " << QueueKindToString(queue)
					<< " (slot " << qi << ", batch " << args.batchIndex << ") with a non-standard exception";
				spdlog::error(oss.str());
				throw std::runtime_error(oss.str());
			}
		};

		for (auto& passVariant : batch.Passes(qi)) {
			std::visit([&](auto* passEntry) { executeOne(*passEntry); }, passVariant);
		}
		if (args.statisticsService)
			args.statisticsService->ResolveQueries(args.context.frameIndex, rhiQueue, commandList);

		// Split after execution if needed
		if (sched.splitAfterExecution) {
			UINT64 signalValue = fenceOffset + batch.GetQueueSignalFenceValue(
				RenderGraph::BatchSignalPhase::AfterExecution, qi);
			commandList.End();
			rhiQueue.Submit({ &commandList, 1 }, {});
			SignalQueueFenceOrThrow(
				rhiQueue,
				args.fenceTimeline,
				signalValue,
				queue,
				qi,
				args.batchIndex,
				"AfterExecution",
				static_cast<unsigned>(args.context.frameIndex));
			args.lastSignaledOnTimeline = std::max(args.lastSignaledOnTimeline, signalValue);
			pool.Recycle(std::move(sched.preallocatedCLs[clIndex]), signalValue);

			++clIndex;
			commandList = sched.preallocatedCLs[clIndex].list.Get();
			args.context.commandList = commandList;
		}

		// Waits: BeforeAfterPasses
		for (size_t srcIndex = 0; srcIndex < batch.QueueCount(); ++srcIndex) {
			if (!batch.HasQueueWait(RenderGraph::BatchWaitPhase::BeforeAfterPasses, qi, srcIndex))
				continue;
			UINT64 val = fenceOffset + batch.GetQueueWaitFenceValue(
				RenderGraph::BatchWaitPhase::BeforeAfterPasses, qi, srcIndex);
			WaitOnSlot(qi, srcIndex, val);
		}

		// Record post-transitions
		auto& postTransitions = batch.Transitions(qi, RenderGraph::BatchTransitionPhase::AfterPasses);
		if (!postTransitions.empty())
			ExecuteTransitions(postTransitions, /*crm=*/nullptr, queue, commandList);

		// Final submit + recycle signal. Active queues always submit a final CL,
		// so always use the batch's reserved AfterCompletion fence value.
		{
			commandList.End();
			rhiQueue.Submit({ &commandList, 1 }, {});

			UINT64 recycleFence = fenceOffset + batch.GetQueueSignalFenceValue(
				RenderGraph::BatchSignalPhase::AfterCompletion, qi);
			if (recycleFence == 0) {
				spdlog::error("ExecuteQueueBatch: recycleFence is 0 for batch {} slot {} "
					"(fenceOffset={}, fenceValue={}). "
					"Falling back to monotonic signal.",
					args.batchIndex, qi, fenceOffset,
					batch.GetQueueSignalFenceValue(
						RenderGraph::BatchSignalPhase::AfterCompletion, qi));
				recycleFence = ++args.lastSignaledOnTimeline;
			}
			SignalQueueFenceOrThrow(
				rhiQueue,
				args.fenceTimeline,
				recycleFence,
				queue,
				qi,
				args.batchIndex,
				"AfterCompletion",
				static_cast<unsigned>(args.context.frameIndex));
			args.lastSignaledOnTimeline = std::max(args.lastSignaledOnTimeline, recycleFence);
			pool.Recycle(std::move(sched.preallocatedCLs[clIndex]), recycleFence);
		}
	}

	// RecordQueueBatch: records barrier + pass commands into pre-allocated
	// CLs and calls End() on each.  Does NOT Submit, Signal, Wait, or
	// Recycle.  Safe to call from a worker thread.
	struct RecordQueueBatchArgs {
		QueueBatchSchedule& sched;
		RenderGraph::PassBatch& batch;
		size_t batchIndex;
		QueueKind queue;
		size_t queueSlot;
		rhi::Queue& rhiQueue;           // needed for statistics Begin/EndQuery
		PassExecutionContext context;    // COPY: each task gets its own
		rg::runtime::IStatisticsService* statisticsService;
		std::unordered_map<ExternalFenceSignalKey, ExternalFenceSignalOrigin, ExternalFenceSignalKeyHash>& queuedExternalFenceOrigins;
		bool batchTraceEnabled;
	};

	void RecordQueueBatch(RecordQueueBatchArgs& args) {
		ZoneScopedN("RenderGraph::RecordQueueBatch");
		auto& sched = args.sched;
		auto& batch = args.batch;
		auto  queue = args.queue;
		auto  qi    = args.queueSlot;
		if (args.batchTraceEnabled) {
			spdlog::info(
				"RenderGraph: frame {} record batch {} queue {} slot {} begin preTransitions={} passes={} postTransitions={} numCLs={} splitAfterTransitions={} splitAfterExecution={}",
				static_cast<unsigned>(args.context.frameIndex),
				args.batchIndex,
				QueueKindToString(queue),
				qi,
				batch.Transitions(qi, RenderGraph::BatchTransitionPhase::BeforePasses).size(),
				batch.Passes(qi).size(),
				batch.Transitions(qi, RenderGraph::BatchTransitionPhase::AfterPasses).size(),
				sched.numCLs,
				sched.splitAfterTransitions,
				sched.splitAfterExecution);
		}

		uint8_t clIndex = 0;
		rhi::CommandList commandList = sched.preallocatedCLs[clIndex].list.Get();

		// Record pre-transitions
		auto& preTransitions = batch.Transitions(qi, RenderGraph::BatchTransitionPhase::BeforePasses);
		RecordTransitionBarriers(preTransitions, commandList);

		// Split after transitions?
		if (sched.splitAfterTransitions) {
			if (args.batchTraceEnabled) {
				spdlog::info(
					"RenderGraph: frame {} record batch {} queue {} slot {} ending transition CL {}",
					static_cast<unsigned>(args.context.frameIndex),
					args.batchIndex,
					QueueKindToString(queue),
					qi,
					clIndex);
			}
			commandList.End();
			++clIndex;
			commandList = sched.preallocatedCLs[clIndex].list.Get();
		}

		// Record all passes.
		args.context.commandList = commandList;

		auto executeOne = [&](auto& pr) {
			if (!pr.pass->IsInvalidated())
				return;
			const std::string_view passName = pr.name.empty() ? std::string_view("<unnamed>") : std::string_view(pr.name);
			const char* techniquePath = pr.techniquePath.empty() ? nullptr : pr.techniquePath.c_str();
			try {
				ZoneScopedN("RenderGraph::RecordQueueBatch::PassRecord");
				ZoneText(passName.data(), passName.size());
				if (args.batchTraceEnabled) {
					spdlog::info(
						"RenderGraph: frame {} batch {} queue {} slot {} begin pass {}",
						static_cast<unsigned>(args.context.frameIndex),
						args.batchIndex,
						QueueKindToString(queue),
						qi,
						passName);
				}
				rhi::debug::Scope scope(commandList, rhi::colors::Mint, std::string(passName).c_str());
				args.context.currentPassName = passName.data();
				args.context.currentTechniquePath = techniquePath;
				(void)rhi::debug::SetInstrumentationContext(commandList, args.context.currentPassName, args.context.currentTechniquePath);
				const bool hasStatistics = args.statisticsService && pr.statisticsIndex >= 0;
				const auto cpuStart = std::chrono::steady_clock::now();
				if (hasStatistics)
					args.statisticsService->BeginQuery(pr.statisticsIndex, args.context.frameIndex, args.rhiQueue, commandList, sched.queryRecordingContext);
				if ((pr.run & PassRunMask::Immediate) != PassRunMask::None)
					rg::imm::Replay(pr.immediateBytecode, commandList, *args.context.immediateDispatch);
				pr.immediateKeepAlive.reset();
				if ((pr.run & PassRunMask::Retained) != PassRunMask::None) {
					auto passReturn = pr.pass->Execute(args.context);
					if (passReturn.fence || !passReturn.externalSignalsAfterCompletion.empty()) {
						if (args.batchTraceEnabled) {
							LogQueuedExternalFence(
								static_cast<unsigned>(args.context.frameIndex),
								queue,
								qi,
								args.batchIndex,
								passName,
								args.queuedExternalFenceOrigins,
								passReturn);
						}
						sched.externalFences.push_back(passReturn);
					}
				}
				if (hasStatistics)
					args.statisticsService->EndQuery(pr.statisticsIndex, args.context.frameIndex, args.rhiQueue, commandList, sched.queryRecordingContext);
				if (hasStatistics) {
					args.statisticsService->RecordCpuExecuteTime(
						static_cast<unsigned>(pr.statisticsIndex),
						std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuStart).count());
				}
				(void)rhi::debug::SetInstrumentationContext(commandList, nullptr, nullptr);
				args.context.currentPassName = nullptr;
				args.context.currentTechniquePath = nullptr;
				if (args.batchTraceEnabled) {
					spdlog::info(
						"RenderGraph: frame {} batch {} queue {} slot {} end pass {}",
						static_cast<unsigned>(args.context.frameIndex),
						args.batchIndex,
						QueueKindToString(queue),
						qi,
						passName);
				}
			}
			catch (const std::exception& ex) {
				(void)rhi::debug::SetInstrumentationContext(commandList, nullptr, nullptr);
				args.context.currentPassName = nullptr;
				args.context.currentTechniquePath = nullptr;
				std::ostringstream oss;
				oss << "RenderGraph::RecordQueueBatch failed while recording pass '"
					<< passName
					<< "' on queue " << QueueKindToString(queue)
					<< " (slot " << qi << "): " << ex.what();
				spdlog::error(oss.str());
				throw std::runtime_error(oss.str());
			}
			catch (...) {
				std::ostringstream oss;
				oss << "RenderGraph::RecordQueueBatch failed while recording pass '"
					<< passName
					<< "' on queue " << QueueKindToString(queue)
					<< " (slot " << qi << ") with a non-standard exception";
				spdlog::error(oss.str());
				throw std::runtime_error(oss.str());
			}
		};

		for (auto& passVariant : batch.Passes(qi)) {
			std::visit([&](auto* passEntry) { executeOne(*passEntry); }, passVariant);
		}
		if (args.statisticsService)
			args.statisticsService->ResolveQueries(args.context.frameIndex, args.rhiQueue, commandList, sched.queryRecordingContext);

		// Split after execution?
		if (sched.splitAfterExecution) {
			if (args.batchTraceEnabled) {
				spdlog::info(
					"RenderGraph: frame {} record batch {} queue {} slot {} ending execution CL {}",
					static_cast<unsigned>(args.context.frameIndex),
					args.batchIndex,
					QueueKindToString(queue),
					qi,
					clIndex);
			}
			commandList.End();
			++clIndex;
			commandList = sched.preallocatedCLs[clIndex].list.Get();
		}

		// Record post-transitions (barriers only).
		auto& postTransitions = batch.Transitions(qi, RenderGraph::BatchTransitionPhase::AfterPasses);
		if (!postTransitions.empty())
			RecordTransitionBarriers(postTransitions, commandList);

		// End the last CL.
		if (args.batchTraceEnabled) {
			spdlog::info(
				"RenderGraph: frame {} record batch {} queue {} slot {} ending final CL {}",
				static_cast<unsigned>(args.context.frameIndex),
				args.batchIndex,
				QueueKindToString(queue),
				qi,
				clIndex);
		}
		commandList.End();
		if (args.batchTraceEnabled) {
			spdlog::info(
				"RenderGraph: frame {} record batch {} queue {} slot {} complete",
				static_cast<unsigned>(args.context.frameIndex),
				args.batchIndex,
				QueueKindToString(queue),
				qi);
		}
	}

	// -----------------------------------------------------------------------
	// SubmitQueueBatch: submits pre-recorded CLs with the correct wait /
	// signal / recycle ordering.  Must be called on the main thread.
	// CLs must already have had End() called (by RecordQueueBatch).
	// -----------------------------------------------------------------------
	struct SubmitQueueBatchArgs {
		QueueBatchSchedule& sched;
		RenderGraph::PassBatch& batch;
		size_t batchIndex;
		QueueKind queue;
		size_t queueSlot;
		rhi::Queue& rhiQueue;
		rhi::Timeline& fenceTimeline;
		CommandListPool& pool;
		UINT64 fenceOffset;
		UINT64& lastSignaledOnTimeline;
		unsigned frameIndex;
		bool batchTraceEnabled;
	};

	void SubmitQueueBatch(
		SubmitQueueBatchArgs& args,
		auto&& WaitOnSlot)
	{
		ZoneScopedN("RenderGraph::SubmitQueueBatch");
		ZoneText(QueueKindToString(args.queue), std::strlen(QueueKindToString(args.queue)));
		auto& sched      = args.sched;
		auto& batch      = args.batch;
		auto  queue      = args.queue;
		auto  qi         = args.queueSlot;
		auto& rhiQueue   = args.rhiQueue;
		auto  fenceOffset = args.fenceOffset;
		auto waitPhaseName = [](RenderGraph::BatchWaitPhase phase) -> const char* {
			switch (phase) {
			case RenderGraph::BatchWaitPhase::BeforeTransitions: return "BeforeTransitions";
			case RenderGraph::BatchWaitPhase::BeforeExecution: return "BeforeExecution";
			case RenderGraph::BatchWaitPhase::BeforeAfterPasses: return "BeforeAfterPasses";
			default: return "Unknown";
			}
		};
		auto signalPhaseName = [](RenderGraph::BatchSignalPhase phase) -> const char* {
			switch (phase) {
			case RenderGraph::BatchSignalPhase::AfterTransitions: return "AfterTransitions";
			case RenderGraph::BatchSignalPhase::AfterExecution: return "AfterExecution";
			case RenderGraph::BatchSignalPhase::AfterCompletion: return "AfterCompletion";
			default: return "Unknown";
			}
		};
		if (args.batchTraceEnabled) {
			spdlog::info(
				"RenderGraph: frame {} submit batch {} queue {} slot {} begin numCLs={} splitAfterTransitions={} splitAfterExecution={}",
				args.frameIndex,
				args.batchIndex,
				QueueKindToString(queue),
				qi,
				sched.numCLs,
				sched.splitAfterTransitions,
				sched.splitAfterExecution);
		}

		uint8_t clIndex = 0;

		// Waits: BeforeTransitions + BeforeExecution
		{
			ZoneScopedN("RenderGraph::SubmitQueueBatch::WaitBeforeTransitions");
			for (size_t srcIndex = 0; srcIndex < batch.QueueCount(); ++srcIndex) {
				if (batch.HasQueueWait(RenderGraph::BatchWaitPhase::BeforeTransitions, qi, srcIndex)) {
					UINT64 val = fenceOffset + batch.GetQueueWaitFenceValue(
						RenderGraph::BatchWaitPhase::BeforeTransitions, qi, srcIndex);
					if (args.batchTraceEnabled) {
						spdlog::info(
							"RenderGraph: frame {} submit batch {} queue {} slot {} begin wait phase={} srcSlot={} fence={} srcCompleted={}"
							,args.frameIndex,
							args.batchIndex,
							QueueKindToString(queue),
							qi,
							waitPhaseName(RenderGraph::BatchWaitPhase::BeforeTransitions),
							srcIndex,
							val,
							args.fenceTimeline.GetCompletedValue());
					}
					WaitOnSlot(qi, srcIndex, val);
					if (args.batchTraceEnabled) {
						spdlog::info(
							"RenderGraph: frame {} submit batch {} queue {} slot {} end wait phase={} srcSlot={} fence={}"
							,args.frameIndex,
							args.batchIndex,
							QueueKindToString(queue),
							qi,
							waitPhaseName(RenderGraph::BatchWaitPhase::BeforeTransitions),
							srcIndex,
							val);
					}
				}
			}
		}
		{
			ZoneScopedN("RenderGraph::SubmitQueueBatch::WaitBeforeExecution");
			for (size_t srcIndex = 0; srcIndex < batch.QueueCount(); ++srcIndex) {
				if (batch.HasQueueWait(RenderGraph::BatchWaitPhase::BeforeExecution, qi, srcIndex)) {
					UINT64 val = fenceOffset + batch.GetQueueWaitFenceValue(
						RenderGraph::BatchWaitPhase::BeforeExecution, qi, srcIndex);
					if (args.batchTraceEnabled) {
						spdlog::info(
							"RenderGraph: frame {} submit batch {} queue {} slot {} begin wait phase={} srcSlot={} fence={} srcCompleted={}"
							,args.frameIndex,
							args.batchIndex,
							QueueKindToString(queue),
							qi,
							waitPhaseName(RenderGraph::BatchWaitPhase::BeforeExecution),
							srcIndex,
							val,
							args.fenceTimeline.GetCompletedValue());
					}
					WaitOnSlot(qi, srcIndex, val);
					if (args.batchTraceEnabled) {
						spdlog::info(
							"RenderGraph: frame {} submit batch {} queue {} slot {} end wait phase={} srcSlot={} fence={}"
							,args.frameIndex,
							args.batchIndex,
							QueueKindToString(queue),
							qi,
							waitPhaseName(RenderGraph::BatchWaitPhase::BeforeExecution),
							srcIndex,
							val);
					}
				}
			}
		}

		// Submit + signal for the transitions CL if it was split out.
		if (sched.splitAfterTransitions) {
			ZoneScopedN("RenderGraph::SubmitQueueBatch::SubmitAfterTransitions");
			rhi::CommandList cl = sched.preallocatedCLs[clIndex].list.Get();
			if (args.batchTraceEnabled) {
				spdlog::info(
					"RenderGraph: frame {} submit batch {} queue {} slot {} begin submit phase={} clIndex={}",
					args.frameIndex,
					args.batchIndex,
					QueueKindToString(queue),
					qi,
					signalPhaseName(RenderGraph::BatchSignalPhase::AfterTransitions),
					clIndex);
			}
			rhiQueue.Submit({ &cl, 1 }, {});
			if (args.batchTraceEnabled) {
				spdlog::info(
					"RenderGraph: frame {} submit batch {} queue {} slot {} end submit phase={} clIndex={}",
					args.frameIndex,
					args.batchIndex,
					QueueKindToString(queue),
					qi,
					signalPhaseName(RenderGraph::BatchSignalPhase::AfterTransitions),
					clIndex);
			}
			UINT64 signalValue = fenceOffset + batch.GetQueueSignalFenceValue(
				RenderGraph::BatchSignalPhase::AfterTransitions, qi);
			if (args.batchTraceEnabled) {
				spdlog::info(
					"RenderGraph: frame {} submit batch {} queue {} slot {} begin signal phase={} fence={}",
					args.frameIndex,
					args.batchIndex,
					QueueKindToString(queue),
					qi,
					signalPhaseName(RenderGraph::BatchSignalPhase::AfterTransitions),
					signalValue);
			}
			SignalQueueFenceOrThrow(
				rhiQueue,
				args.fenceTimeline,
				signalValue,
				queue,
				qi,
				args.batchIndex,
				signalPhaseName(RenderGraph::BatchSignalPhase::AfterTransitions),
				args.frameIndex);
			if (args.batchTraceEnabled) {
				spdlog::info(
					"RenderGraph: frame {} submit batch {} queue {} slot {} end signal phase={} fence={}",
					args.frameIndex,
					args.batchIndex,
					QueueKindToString(queue),
					qi,
					signalPhaseName(RenderGraph::BatchSignalPhase::AfterTransitions),
					signalValue);
			}
			args.lastSignaledOnTimeline = std::max(args.lastSignaledOnTimeline, signalValue);
			args.pool.Recycle(std::move(sched.preallocatedCLs[clIndex]), signalValue);
			++clIndex;
		}

		// Submit + signal for the passes CL if it was split out.
		if (sched.splitAfterExecution) {
			ZoneScopedN("RenderGraph::SubmitQueueBatch::SubmitAfterExecution");
			rhi::CommandList cl = sched.preallocatedCLs[clIndex].list.Get();
			if (args.batchTraceEnabled) {
				spdlog::info(
					"RenderGraph: frame {} submit batch {} queue {} slot {} begin submit phase={} clIndex={}",
					args.frameIndex,
					args.batchIndex,
					QueueKindToString(queue),
					qi,
					signalPhaseName(RenderGraph::BatchSignalPhase::AfterExecution),
					clIndex);
			}
			rhiQueue.Submit({ &cl, 1 }, {});
			if (args.batchTraceEnabled) {
				spdlog::info(
					"RenderGraph: frame {} submit batch {} queue {} slot {} end submit phase={} clIndex={}",
					args.frameIndex,
					args.batchIndex,
					QueueKindToString(queue),
					qi,
					signalPhaseName(RenderGraph::BatchSignalPhase::AfterExecution),
					clIndex);
			}
			UINT64 signalValue = fenceOffset + batch.GetQueueSignalFenceValue(
				RenderGraph::BatchSignalPhase::AfterExecution, qi);
			if (args.batchTraceEnabled) {
				spdlog::info(
					"RenderGraph: frame {} submit batch {} queue {} slot {} begin signal phase={} fence={}",
					args.frameIndex,
					args.batchIndex,
					QueueKindToString(queue),
					qi,
					signalPhaseName(RenderGraph::BatchSignalPhase::AfterExecution),
					signalValue);
			}
			SignalQueueFenceOrThrow(
				rhiQueue,
				args.fenceTimeline,
				signalValue,
				queue,
				qi,
				args.batchIndex,
				signalPhaseName(RenderGraph::BatchSignalPhase::AfterExecution),
				args.frameIndex);
			if (args.batchTraceEnabled) {
				spdlog::info(
					"RenderGraph: frame {} submit batch {} queue {} slot {} end signal phase={} fence={}",
					args.frameIndex,
					args.batchIndex,
					QueueKindToString(queue),
					qi,
					signalPhaseName(RenderGraph::BatchSignalPhase::AfterExecution),
					signalValue);
			}
			args.lastSignaledOnTimeline = std::max(args.lastSignaledOnTimeline, signalValue);
			args.pool.Recycle(std::move(sched.preallocatedCLs[clIndex]), signalValue);
			++clIndex;
		}

		// Waits: BeforeAfterPasses
		{
			ZoneScopedN("RenderGraph::SubmitQueueBatch::WaitBeforeAfterPasses");
			for (size_t srcIndex = 0; srcIndex < batch.QueueCount(); ++srcIndex) {
				if (batch.HasQueueWait(RenderGraph::BatchWaitPhase::BeforeAfterPasses, qi, srcIndex)) {
					UINT64 val = fenceOffset + batch.GetQueueWaitFenceValue(
						RenderGraph::BatchWaitPhase::BeforeAfterPasses, qi, srcIndex);
					if (args.batchTraceEnabled) {
						spdlog::info(
							"RenderGraph: frame {} submit batch {} queue {} slot {} begin wait phase={} srcSlot={} fence={} srcCompleted={}",
							args.frameIndex,
							args.batchIndex,
							QueueKindToString(queue),
							qi,
							waitPhaseName(RenderGraph::BatchWaitPhase::BeforeAfterPasses),
							srcIndex,
							val,
							args.fenceTimeline.GetCompletedValue());
					}
					WaitOnSlot(qi, srcIndex, val);
					if (args.batchTraceEnabled) {
						spdlog::info(
							"RenderGraph: frame {} submit batch {} queue {} slot {} end wait phase={} srcSlot={} fence={}",
							args.frameIndex,
							args.batchIndex,
							QueueKindToString(queue),
							qi,
							waitPhaseName(RenderGraph::BatchWaitPhase::BeforeAfterPasses),
							srcIndex,
							val);
					}
				}
			}
		}

		// Submit the final CL and signal for recycle. Active queues always submit
		// a final CL, so always use the batch's reserved AfterCompletion fence value.
		{
			ZoneScopedN("RenderGraph::SubmitQueueBatch::SubmitAfterCompletion");
			rhi::CommandList cl = sched.preallocatedCLs[clIndex].list.Get();
			if (args.batchTraceEnabled) {
				spdlog::info(
					"RenderGraph: frame {} submit batch {} queue {} slot {} begin submit phase={} clIndex={}",
					args.frameIndex,
					args.batchIndex,
					QueueKindToString(queue),
					qi,
					signalPhaseName(RenderGraph::BatchSignalPhase::AfterCompletion),
					clIndex);
			}
			rhiQueue.Submit({ &cl, 1 }, {});
			if (args.batchTraceEnabled) {
				spdlog::info(
					"RenderGraph: frame {} submit batch {} queue {} slot {} end submit phase={} clIndex={}",
					args.frameIndex,
					args.batchIndex,
					QueueKindToString(queue),
					qi,
					signalPhaseName(RenderGraph::BatchSignalPhase::AfterCompletion),
					clIndex);
			}

			UINT64 recycleFence = fenceOffset + batch.GetQueueSignalFenceValue(
				RenderGraph::BatchSignalPhase::AfterCompletion, qi);
			if (recycleFence == 0) {
				spdlog::error(
					"SubmitQueueBatch: recycleFence is 0 (fenceOffset={}, fenceValue={}). "
					"Falling back to monotonic signal.",
					fenceOffset,
					batch.GetQueueSignalFenceValue(
						RenderGraph::BatchSignalPhase::AfterCompletion, qi));
				recycleFence = ++args.lastSignaledOnTimeline;
			}
			if (args.batchTraceEnabled) {
				spdlog::info(
					"RenderGraph: frame {} submit batch {} queue {} slot {} begin signal phase={} fence={}",
					args.frameIndex,
					args.batchIndex,
					QueueKindToString(queue),
					qi,
					signalPhaseName(RenderGraph::BatchSignalPhase::AfterCompletion),
					recycleFence);
			}
			SignalQueueFenceOrThrow(
				rhiQueue,
				args.fenceTimeline,
				recycleFence,
				queue,
				qi,
				args.batchIndex,
				signalPhaseName(RenderGraph::BatchSignalPhase::AfterCompletion),
				args.frameIndex);
			if (args.batchTraceEnabled) {
				spdlog::info(
					"RenderGraph: frame {} submit batch {} queue {} slot {} end signal phase={} fence={}",
					args.frameIndex,
					args.batchIndex,
					QueueKindToString(queue),
					qi,
					signalPhaseName(RenderGraph::BatchSignalPhase::AfterCompletion),
					recycleFence);
			}
			args.lastSignaledOnTimeline = std::max(args.lastSignaledOnTimeline, recycleFence);
			args.pool.Recycle(std::move(sched.preallocatedCLs[clIndex]), recycleFence);
		}
		if (args.batchTraceEnabled) {
			spdlog::info(
				"RenderGraph: frame {} submit batch {} queue {} slot {} complete",
				args.frameIndex,
				args.batchIndex,
				QueueKindToString(queue),
				qi);
		}
	}

} // namespace

void RenderGraph::BuildExecutionSchedule() {
	ZoneScopedN("RenderGraph::BuildExecutionSchedule");
	auto& schedule = m_executionSchedule;
	const size_t qc = m_queueRegistry.SlotCount();
	schedule.batches.clear();
	schedule.batches.reserve(batches.size());
	for (size_t i = 0; i < batches.size(); ++i) {
		schedule.batches.emplace_back(qc);
	}

	for (size_t bi = 0; bi < batches.size(); ++bi) {
		auto& batch = batches[bi];
		auto& batchSched = schedule.batches[bi];

		for (size_t qi = 0; qi < batchSched.queues.size(); ++qi) {
			auto& qs = batchSched.queues[qi];

			bool hasPre  = batch.HasTransitions(qi, BatchTransitionPhase::BeforePasses);
			bool hasPost = batch.HasTransitions(qi, BatchTransitionPhase::AfterPasses);
			bool hasPasses = batch.HasPasses(qi);
			qs.active = hasPre || hasPasses || hasPost;

			if (!qs.active) {
				qs.numCLs = 0;
				continue;
			}

			// Batch 0 is a dummy sentinel with all-zero fence values.
			// It should never be active. If it is, something upstream
			// incorrectly added transitions or passes to it.
			if (bi == 0) {
				spdlog::error(
					"BuildExecutionSchedule: batch 0 (sentinel) is unexpectedly "
					"active on queue {} (hasPre={}, hasPasses={}, hasPost={}). "
					"This will produce a zero-value fence signal. Forcing inactive.",
					qi, hasPre, hasPasses, hasPost);
				qs.active = false;
				qs.numCLs = 0;
				continue;
			}

			qs.splitAfterTransitions =
				batch.HasQueueSignal(BatchSignalPhase::AfterTransitions, qi);
			qs.splitAfterExecution =
				batch.HasQueueSignal(BatchSignalPhase::AfterExecution, qi);
			qs.signalAfterCompletion =
				batch.HasQueueSignal(BatchSignalPhase::AfterCompletion, qi);

			qs.numCLs = 1
				+ static_cast<uint8_t>(qs.splitAfterTransitions)
				+ static_cast<uint8_t>(qs.splitAfterExecution);
		}
	}
}

void RenderGraph::Execute(PassExecutionContext& context) {
	ZoneScopedN("RenderGraph::Execute");
	m_lastPresentDependency.reset();
	{
		ZoneScopedN("RenderGraph::Execute::ValidateCompiledResourceGenerations");
		ValidateCompiledResourceGenerations();
	}
	context.immediateDispatch = &m_immediateDispatch;

	const bool heavyDebug = m_getHeavyDebug ? m_getHeavyDebug() : false;
	const bool batchTraceEnabled = m_getRenderGraphBatchTraceEnabled ? m_getRenderGraphBatchTraceEnabled() : false;
	auto& manager = DeviceManager::GetInstance();
	const size_t slotCount = m_queueRegistry.SlotCount();
	if (batchTraceEnabled) {
		spdlog::info(
			"RenderGraph::Execute begin frame={} batches={} slotCount={} heavyDebug={}",
			static_cast<unsigned>(context.frameIndex),
			batches.size(),
			slotCount,
			heavyDebug);
	}

	// Create CRM from the primary queue slots in the registry.
	CommandRecordingManager::Init init{
		.graphicsQ = &manager.GetGraphicsQueue(),
		.graphicsF = &m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(0)),
		.graphicsPool = m_queueRegistry.GetPool(static_cast<QueueSlotIndex>(0)),

		.computeQ = &manager.GetComputeQueue(),
		.computeF = &m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(1)),
		.computePool = m_queueRegistry.GetPool(static_cast<QueueSlotIndex>(1)),

		.copyQ = &manager.GetCopyQueue(),
		.copyF = &m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(2)),
		.copyPool = m_queueRegistry.GetPool(static_cast<QueueSlotIndex>(2)),
	};

	{
		ZoneScopedN("RenderGraph::Execute::CreateCommandRecordingManager");
		m_pCommandRecordingManager = std::make_unique<CommandRecordingManager>(init);
	}
	if (batchTraceEnabled) {
		spdlog::info("RenderGraph::Execute frame={} created CommandRecordingManager", static_cast<unsigned>(context.frameIndex));
	}
	auto crm = m_pCommandRecordingManager.get();

	// Registry-based queue/fence/pool resolution
	auto SlotQueue = [&](size_t qi) -> rhi::Queue {
		return m_queueRegistry.GetQueue(static_cast<QueueSlotIndex>(qi));
	};
	auto SlotFence = [&](size_t qi) -> rhi::Timeline& {
		return m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(qi));
	};
	auto SlotPool = [&](size_t qi) -> CommandListPool* {
		return m_queueRegistry.GetPool(static_cast<QueueSlotIndex>(qi));
	};

	auto WaitOnSlot = [&](size_t dstSlot, size_t srcSlot, UINT64 absoluteFenceValue) {
		if (dstSlot == srcSlot) return;
		if (absoluteFenceValue == 0 || absoluteFenceValue == UINT64_MAX) {
			throw std::runtime_error(fmt::format(
				"WaitOnSlot rejected invalid fence value: dstSlot={} srcSlot={} value={}",
				dstSlot,
				srcSlot,
				absoluteFenceValue));
		}
		const UINT64 completedFenceValue = SlotFence(srcSlot).GetCompletedValue();
		if (completedFenceValue == UINT64_MAX) {
			throw std::runtime_error(fmt::format(
				"WaitOnSlot detected poisoned queue timeline before wait: dstSlot={} srcSlot={} requestedFence={} completed=UINT64_MAX",
				dstSlot,
				srcSlot,
				absoluteFenceValue));
		}
		auto dstQ = SlotQueue(dstSlot);
		const rhi::Result waitResult = dstQ.Wait({ SlotFence(srcSlot).GetHandle(), absoluteFenceValue });
		if (waitResult != rhi::Result::Ok) {
			throw std::runtime_error(fmt::format(
				"WaitOnSlot failed: dstSlot={} srcSlot={} fence={} completed={} result={}",
				dstSlot,
				srcSlot,
				absoluteFenceValue,
				completedFenceValue,
				rhi::ResultName(waitResult)));
		}
	};

	auto batchExecutesOnQueue = [](const PassBatch& batch, size_t queueIndex) {
		return batch.HasTransitions(queueIndex, BatchTransitionPhase::BeforePasses)
			|| batch.HasPasses(queueIndex)
			|| batch.HasTransitions(queueIndex, BatchTransitionPhase::AfterPasses);
	};

	std::vector<unsigned int> lastCrossFrameSignalBatchByQueue(slotCount, 0);
	for (unsigned int batchIndex = 1; batchIndex < static_cast<unsigned int>(batches.size()); ++batchIndex) {
		auto& batch = batches[batchIndex];
		for (size_t queueIndex = 0; queueIndex < slotCount; ++queueIndex) {
			if (batchExecutesOnQueue(batch, queueIndex)) {
				lastCrossFrameSignalBatchByQueue[queueIndex] = batchIndex;
			}
		}
	}

	{
		ZoneScopedN("RenderGraph::Execute::ApplyFrameStartWaits");
		// Frame-start waits from previous frame's last-producer tracking.
		for (size_t dstIndex = 0; dstIndex < slotCount; ++dstIndex) {
			for (size_t srcIndex = 0; srcIndex < slotCount; ++srcIndex) {
				if (dstIndex == srcIndex) continue;
				if (dstIndex >= m_hasPendingFrameStartQueueWait.size() ||
					srcIndex >= m_hasPendingFrameStartQueueWait[dstIndex].size()) continue;
				if (!m_hasPendingFrameStartQueueWait[dstIndex][srcIndex]) continue;
				WaitOnSlot(dstIndex, srcIndex,
					m_pendingFrameStartQueueWaitFenceValue[dstIndex][srcIndex]);
			}
		}
	}
	if (batchTraceEnabled) {
		spdlog::info("RenderGraph::Execute frame={} completed frame-start waits", static_cast<unsigned>(context.frameIndex));
	}

	{
		ZoneScopedN("RenderGraph::Execute::MarkCompletionSignals");
		// Cross-frame waits only need a monotonic signal that is guaranteed to fire
		// after the queue's final work for the frame. Marking every producer batch
		// forces extra submissions in the parallel path.
		for (size_t queueIndex = 0; queueIndex < slotCount; ++queueIndex) {
			if (queueIndex >= m_compiledLastProducerBatchByResourceByQueue.size()) continue;
			if (m_compiledLastProducerBatchByResourceByQueue[queueIndex].empty()) continue;

			const unsigned int signalBatch = lastCrossFrameSignalBatchByQueue[queueIndex];
			if (signalBatch > 0 && signalBatch < batches.size()) {
				batches[signalBatch].MarkQueueSignal(BatchSignalPhase::AfterCompletion, queueIndex);
			}
			else {
				spdlog::warn(
					"RenderGraph::Execute frame={} queue slot {} has {} cross-frame producers but no active batch to signal.",
					static_cast<unsigned>(context.frameIndex),
					queueIndex,
					m_compiledLastProducerBatchByResourceByQueue[queueIndex].size());
			}
		}
	}
	if (batchTraceEnabled) {
		spdlog::info("RenderGraph::Execute frame={} marked completion signals", static_cast<unsigned>(context.frameIndex));
	}

	{
		ZoneScopedN("RenderGraph::Execute::AssignQueueSignalFenceValues");
		AssignQueueSignalFenceValuesInSubmissionOrder(batches);
	}
	if (batchTraceEnabled) {
		spdlog::info("RenderGraph::Execute frame={} assigned queue signal fences in submission order", static_cast<unsigned>(context.frameIndex));
	}

	auto& nextLastProducerByResourceAcrossFrames = m_lastProducerByResourceAcrossFrames;
	auto& nextLastAliasPlacementProducersByPoolAcrossFrames = m_lastAliasPlacementProducersByPoolAcrossFrames;

	auto removeResourceFromLastAliasPlacementCache = [&](uint64_t resourceID) {
		for (auto& [poolID, producers] : nextLastAliasPlacementProducersByPoolAcrossFrames) {
			(void)poolID;
			producers.erase(
				std::remove_if(
					producers.begin(),
					producers.end(),
					[&](const LastAliasPlacementProducerAcrossFrames& p) {
						return p.resourceID == resourceID;
					}),
				producers.end());
		}
	};

	auto publishAliasPlacementProducer = [&](uint64_t resourceID, LastProducerAcrossFrames producer) {
		removeResourceFromLastAliasPlacementCache(resourceID);

		const auto* placement = TryGetAliasPlacementRange(resourceID);
		if (!placement) {
			return;
		}

		auto itPoolState = persistentAliasPools.find(placement->poolID);
		if (itPoolState == persistentAliasPools.end()) {
			return;
		}

		nextLastAliasPlacementProducersByPoolAcrossFrames[placement->poolID].push_back(
			LastAliasPlacementProducerAcrossFrames{
				.resourceID = resourceID,
				.poolID = placement->poolID,
				.poolGeneration = itPoolState->second.generation,
				.startByte = placement->startByte,
				.endByte = placement->endByte,
				.producer = producer,
			});
	};

	auto* statisticsService = m_statisticsService.get();

	// Build the execution schedule and pre-allocate command lists.
	{
		ZoneScopedN("RenderGraph::Execute::BuildExecutionSchedule");
		BuildExecutionSchedule();
	}
	if (batchTraceEnabled) {
		spdlog::info("RenderGraph::Execute frame={} built execution schedule", static_cast<unsigned>(context.frameIndex));
	}

#if BUILD_TYPE == BUILD_TYPE_DEBUG
	// Post-schedule validation: detect signals that will never fire
	// A signal is "live" only when the queue is active in that batch.
	// A wait that references a dead signal will deadlock the GPU.
	{
		ZoneScopedN("RenderGraph::Execute::DebugScheduleValidation");
		// 1. Collect the set of fence values that will actually be signaled.
		std::vector<std::unordered_set<UINT64>> liveSignalValues(slotCount);
		// Also track the highest value each queue will signal this frame so
		// we can verify frame-start waits from the previous frame.
		std::vector<UINT64> highestLiveSignal(slotCount, 0);

		for (size_t bi = 0; bi < batches.size(); ++bi) {
			auto& batch = batches[bi];
			auto& batchSched = m_executionSchedule.batches[bi];

			for (size_t qi = 0; qi < std::min(batchSched.queues.size(), slotCount); ++qi) {
				auto& qs = batchSched.queues[qi];
				if (!qs.active) {
					// Check: does this inactive queue have signals marked?
					for (size_t sp = 0; sp < PassBatch::kSignalPhaseCount; ++sp) {
						if (batch.HasQueueSignal(static_cast<BatchSignalPhase>(sp), qi)) {
							spdlog::error(
								"SIGNAL ON INACTIVE QUEUE: batch {} slot {} has {} signal "
								"enabled (fence={}) but queue is inactive (no passes/transitions). "
								"This signal will never fire on the GPU!",
								bi, qi,
								sp == 0 ? "AfterTransitions" : sp == 1 ? "AfterExecution" : "AfterCompletion",
								batch.GetQueueSignalFenceValue(static_cast<BatchSignalPhase>(sp), qi));
						}
					}
					continue;
				}

				// Active queue: record which signals will actually fire.
				if (qs.splitAfterTransitions) {
					UINT64 v = batch.GetQueueSignalFenceValue(BatchSignalPhase::AfterTransitions, qi);
					liveSignalValues[qi].insert(v);
					highestLiveSignal[qi] = std::max(highestLiveSignal[qi], v);
				}
				if (qs.splitAfterExecution) {
					UINT64 v = batch.GetQueueSignalFenceValue(BatchSignalPhase::AfterExecution, qi);
					liveSignalValues[qi].insert(v);
					highestLiveSignal[qi] = std::max(highestLiveSignal[qi], v);
				}
				if (qs.signalAfterCompletion) {
					UINT64 v = batch.GetQueueSignalFenceValue(BatchSignalPhase::AfterCompletion, qi);
					liveSignalValues[qi].insert(v);
					highestLiveSignal[qi] = std::max(highestLiveSignal[qi], v);
				}
			}
		}

		// 2. Validate within-frame waits: every wait on an active queue must
		//    reference a live signal or a value already completed (cross-frame).
		bool foundDeadWait = false;
		for (size_t bi = 0; bi < batches.size(); ++bi) {
			auto& batch = batches[bi];
			auto& batchSched = m_executionSchedule.batches[bi];

			for (size_t qi = 0; qi < std::min(batchSched.queues.size(), slotCount); ++qi) {
				if (!batchSched.queues[qi].active) continue;

				for (size_t wp = 0; wp < PassBatch::kWaitPhaseCount; ++wp) {
					auto waitPhase = static_cast<BatchWaitPhase>(wp);
					for (size_t src = 0; src < slotCount; ++src) {
						if (qi == src) continue;
						if (!batch.HasQueueWait(waitPhase, qi, src)) continue;

						UINT64 fv = batch.GetQueueWaitFenceValue(waitPhase, qi, src);

						// Check if this is a live signal from the current frame.
						bool isLive = liveSignalValues[src].count(fv) > 0;

						// Check if already completed from a previous frame.
						UINT64 completedValue = SlotFence(src).GetCompletedValue();
						bool isAlreadyCompleted = (completedValue >= fv);

						if (!isLive && !isAlreadyCompleted) {
							spdlog::error(
								"DEADLOCK: batch {} active slot {} waits at phase {} "
								"on slot {} fence={}, but that value is not a live signal "
								"(slot {} may be inactive) and not already completed "
								"(completed={}). GPU WILL HANG.",
								bi, qi,
								wp == 0 ? "BeforeTransitions" : wp == 1 ? "BeforeExecution" : "BeforeAfterPasses",
								src, fv, src, completedValue);
							foundDeadWait = true;
						}
					}
				}
			}
		}

		// 3. Validate frame-start waits: these reference previous-frame values.
		for (size_t dst = 0; dst < slotCount; ++dst) {
			if (dst >= m_hasPendingFrameStartQueueWait.size()) continue;
			for (size_t src = 0; src < slotCount; ++src) {
				if (dst == src) continue;
				if (src >= m_hasPendingFrameStartQueueWait[dst].size()) continue;
				if (!m_hasPendingFrameStartQueueWait[dst][src]) continue;

				UINT64 fv = m_pendingFrameStartQueueWaitFenceValue[dst][src];
				UINT64 completedValue = SlotFence(src).GetCompletedValue();

				if (completedValue < fv) {
					spdlog::warn(
						"Frame-start wait: slot {} waiting on slot {} fence={}, "
						"currently completed={}. Delta={}. "
						"This wait will block until the previous frame's queue "
						"signals this value.",
						dst, src, fv, completedValue, fv - completedValue);
				}
			}
		}

		// 4. Log cross-frame producer summary for diagnostics.
		for (size_t qi = 0; qi < slotCount; ++qi) {
			if (qi >= m_compiledLastProducerBatchByResourceByQueue.size()) continue;
			size_t count = m_compiledLastProducerBatchByResourceByQueue[qi].size();
			if (count > 0) {
				spdlog::debug(
					"Cross-frame producer tracking: slot {} has {} resources tracked",
					qi, count);
			}
		}

		if (foundDeadWait) {
			// Dump the full active/inactive map for debugging.
			for (size_t bi = 0; bi < batches.size(); ++bi) {
				auto& batchSched = m_executionSchedule.batches[bi];
				std::string activeStr;
				for (size_t qi = 0; qi < std::min(batchSched.queues.size(), slotCount); ++qi) {
					if (!activeStr.empty()) activeStr += ", ";
					activeStr += "slot" + std::to_string(qi) + "=" +
						(batchSched.queues[qi].active ? "ACTIVE" : "inactive");
				}
				spdlog::error("  batch {} queue activity: [{}]", bi, activeStr);
			}
			__debugbreak();
		}
	}
#endif

	{
		ZoneScopedN("RenderGraph::Execute::PreallocateCommandLists");
		std::vector<size_t> requiredCLsByQueue(slotCount, 0);
		for (size_t bi = 0; bi < m_executionSchedule.batches.size(); ++bi) {
			auto& batchSched = m_executionSchedule.batches[bi];
			for (size_t qi = 0; qi < batchSched.queues.size(); ++qi) {
				requiredCLsByQueue[qi] += batchSched.queues[qi].numCLs;
			}
		}

		for (size_t qi = 0; qi < slotCount; ++qi) {
			auto* pool = SlotPool(qi);
			if (!pool) {
				continue;
			}
			if (batchTraceEnabled) {
				spdlog::info(
					"RenderGraph::Execute frame={} preparing CL pool slot {} required={} completedFence={}",
					static_cast<unsigned>(context.frameIndex),
					qi,
					requiredCLsByQueue[qi],
					SlotFence(qi).GetCompletedValue());
			}

			pool->PrepareForRequests(requiredCLsByQueue[qi], SlotFence(qi).GetCompletedValue());
			if (batchTraceEnabled) {
				spdlog::info(
					"RenderGraph::Execute frame={} prepared CL pool slot {}",
					static_cast<unsigned>(context.frameIndex),
					qi);
			}
		}

		// Pre-allocate all CLs from the main thread using warmed registry pools.
		for (size_t bi = 0; bi < m_executionSchedule.batches.size(); ++bi) {
			auto& batchSched = m_executionSchedule.batches[bi];
			for (size_t qi = 0; qi < batchSched.queues.size(); ++qi) {
				auto& qs = batchSched.queues[qi];
				for (uint8_t ci = 0; ci < qs.numCLs; ++ci) {
					if (batchTraceEnabled) {
						spdlog::info(
							"RenderGraph::Execute frame={} requesting CL batch={} slot={} clIndex={} of {}",
							static_cast<unsigned>(context.frameIndex),
							bi,
							qi,
							ci,
							qs.numCLs);
					}
					qs.preallocatedCLs[ci] = SlotPool(qi)->Request();
					if (batchTraceEnabled) {
						spdlog::info(
							"RenderGraph::Execute frame={} acquired CL batch={} slot={} clIndex={} of {}",
							static_cast<unsigned>(context.frameIndex),
							bi,
							qi,
							ci,
							qs.numCLs);
					}
				}
			}
		}
	}
	if (batchTraceEnabled) {
		spdlog::info("RenderGraph::Execute frame={} preallocated command lists", static_cast<unsigned>(context.frameIndex));
	}

	// Per-slot signal tracking for monotonic recycle signals.
	std::vector<UINT64> lastSignaledPerSlot(slotCount);
	std::vector<uint64_t> queueSlotFenceTimelineKeys(slotCount);
	std::unordered_map<ExternalFenceSignalKey, ExternalFenceSignalOrigin, ExternalFenceSignalKeyHash> seenExternalFenceSignalsThisFrame;
	seenExternalFenceSignalsThisFrame.reserve(32);
	std::unordered_map<ExternalFenceSignalKey, ExternalFenceSignalOrigin, ExternalFenceSignalKeyHash> queuedExternalFenceOriginsThisFrame;
	queuedExternalFenceOriginsThisFrame.reserve(32);
	for (size_t qi = 0; qi < slotCount; ++qi) {
		const auto slotIndex = static_cast<QueueSlotIndex>(static_cast<uint8_t>(qi));
		queueSlotFenceTimelineKeys[qi] = PackTimelineSignalKey(SlotFence(qi).GetHandle());
		const UINT64 completedFenceValue = SlotFence(qi).GetCompletedValue();
		UINT64 nextFenceValue = m_queueRegistry.GetCurrentFenceValue(slotIndex);
		if (completedFenceValue != UINT64_MAX) {
			const UINT64 minimumNextFenceValue = completedFenceValue + 1;
			if (nextFenceValue < minimumNextFenceValue) {
				spdlog::warn(
					"RenderGraph::Execute frame={} slot={} queue={} nextFenceValue={} lagged completedFenceValue={}; raising next fence to {}",
					static_cast<unsigned>(context.frameIndex),
					qi,
					QueueKindToString(m_queueRegistry.GetKind(slotIndex)),
					nextFenceValue,
					completedFenceValue,
					minimumNextFenceValue);
				m_queueRegistry.EnsureNextFenceValueAtLeast(slotIndex, minimumNextFenceValue);
				nextFenceValue = m_queueRegistry.GetCurrentFenceValue(slotIndex);
			}
		}
		lastSignaledPerSlot[qi] = nextFenceValue > 0 ? nextFenceValue - 1 : 0;
	}

	// Execution, two paths: heavyDebug (serial) or normal (parallel).
	if (heavyDebug) {
		ZoneScopedN("RenderGraph::Execute::HeavyDebugPath");
		// Serial path: record + submit + drain per batch.
		unsigned int batchIndex = 0;
		for (size_t bi = 0; bi < batches.size(); ++bi) {
			auto& batch = batches[bi];
			auto& batchSched = m_executionSchedule.batches[bi];

			// Execute each queue slot in order.
			std::vector<std::vector<PassReturn>> slotExternalFences(slotCount);
			for (size_t qi = 0; qi < slotCount; ++qi) {
				auto& qs = batchSched.queues[qi];
				if (!qs.active) continue;
				auto rhiQ = SlotQueue(qi);
				ExecuteQueueBatchArgs args{
					.sched = qs,
					.batch = batch,
					.batchIndex = bi,
					.queue = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(qi)),
					.queueSlot = qi,
					.rhiQueue = rhiQ,
					.fenceTimeline = SlotFence(qi),
					.pool = *SlotPool(qi),
					.fenceOffset = 0,
					.context = context,
					.statisticsService = statisticsService,
					.outExternalFences = slotExternalFences[qi],
					.queuedExternalFenceOrigins = queuedExternalFenceOriginsThisFrame,
					.lastSignaledOnTimeline = lastSignaledPerSlot[qi],
					.batchTraceEnabled = batchTraceEnabled,
				};
				ExecuteQueueBatch(args, WaitOnSlot);
			}

			// Signal external fences AFTER all CLs in this batch are submitted.
			for (size_t qi = 0; qi < slotCount; ++qi) {
				if (!slotExternalFences[qi].empty()) {
					auto rhiQ = SlotQueue(qi);
					SignalExternalFences(
						rhiQ,
						m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(qi)),
						&SlotFence(qi),
						qi,
						bi,
						static_cast<unsigned>(context.frameIndex),
						queueSlotFenceTimelineKeys,
						seenExternalFenceSignalsThisFrame,
						m_lastExternalSignalValueByTimeline,
						slotExternalFences[qi]);
				}
			}

			// Drain all queues after every batch.
			for (size_t qi = 0; qi < slotCount; ++qi) {
				auto& qs = batchSched.queues[qi];
				if (!qs.active) continue;
				UINT64 highestSignal = 0;
				for (size_t sp = 0; sp < PassBatch::kSignalPhaseCount; ++sp) {
					if (batch.HasQueueSignal(static_cast<BatchSignalPhase>(sp), qi)) {
						UINT64 v = batch.GetQueueSignalFenceValue(static_cast<BatchSignalPhase>(sp), qi);
						if (v > highestSignal) highestSignal = v;
					}
				}
				highestSignal = std::max(
					highestSignal,
					batch.GetQueueSignalFenceValue(BatchSignalPhase::AfterCompletion, qi));
				if (highestSignal == 0) continue;
				auto result = SlotFence(qi).HostWait(highestSignal);
				DeviceManager::GetInstance().GetDevice().CheckDebugMessages();
				if (rhi::Failed(result)) {
					std::string passNames;
					auto collectNames = [&](size_t q) {
						for (auto& pv : batch.Passes(q)) {
							std::visit([&](auto* pr) {
								if (!passNames.empty()) passNames += ", ";
								passNames += pr->name;
							}, pv);
						}
					};
					for (size_t q = 0; q < slotCount; ++q) collectNames(q);
					spdlog::error(
						"[HeavyDebug] GPU fault after batch {} on queue slot {}. "
						"Passes in batch: [{}]",
						batchIndex, qi, passNames);
				}
			}
			++batchIndex;
		}
	} else {
		ZoneScopedN("RenderGraph::Execute::ParallelPath");
		if (batchTraceEnabled) {
			spdlog::info("RenderGraph::Execute frame={} entering parallel path", static_cast<unsigned>(context.frameIndex));
		}
		// Parallel recording path

		// Clear per-frame recording state from any previous frame.
		{
			ZoneScopedN("RenderGraph::Execute::ParallelPath::ResetRecordingState");
			for (size_t bi = 0; bi < batches.size(); ++bi) {
				auto& batchSched = m_executionSchedule.batches[bi];
				for (size_t qi = 0; qi < batchSched.queues.size(); ++qi) {
					auto& qs = batchSched.queues[qi];
					qs.externalFences.clear();
					qs.queryRecordingContext.recordedIndices.clear();
					qs.queryRecordingContext.pendingRanges.clear();
				}
			}
		}

		// Build flat task list: one entry per active (batch, queue) pair.
		struct RecordTask {
			size_t batchIndex;
			size_t queueIndex;
		};
		std::vector<RecordTask> tasks;
		{
			ZoneScopedN("RenderGraph::Execute::ParallelPath::BuildRecordTasks");
			tasks.reserve(batches.size() * slotCount);
			for (size_t bi = 0; bi < batches.size(); ++bi) {
				auto& batchSched = m_executionSchedule.batches[bi];
				for (size_t qi = 0; qi < batchSched.queues.size(); ++qi) {
					if (batchSched.queues[qi].active)
						tasks.push_back({bi, qi});
				}
			}
		}

		{
			ZoneScopedN("RenderGraph::Execute::ParallelPath::RecordAllBatches");
			if (batchTraceEnabled) {
				spdlog::info(
					"RenderGraph::Execute frame={} record-all-batches begin taskCount={} (serialBypass={})",
					static_cast<unsigned>(context.frameIndex),
					tasks.size(),
					true);
			}
			ParallelForOptional("RecordAllBatches", tasks.size(), [&](size_t taskIdx) {
				auto& task = tasks[taskIdx];
				auto& qs = m_executionSchedule.batches[task.batchIndex].queues[task.queueIndex];
				auto rhiQ = SlotQueue(task.queueIndex);
				const QueueKind queueKind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(task.queueIndex));
				try {
					RecordQueueBatchArgs args{
						.sched = qs,
						.batch = batches[task.batchIndex],
						.batchIndex = task.batchIndex,
						.queue = queueKind,
						.queueSlot = task.queueIndex,
						.rhiQueue = rhiQ,
						.context = context,
						.statisticsService = statisticsService,
						.queuedExternalFenceOrigins = queuedExternalFenceOriginsThisFrame,
						.batchTraceEnabled = batchTraceEnabled,
					};
					RecordQueueBatch(args);
				}
				catch (const std::exception& ex) {
					std::string passNames;
					for (auto& passVariant : batches[task.batchIndex].Passes(task.queueIndex)) {
						std::visit([&](auto* passEntry) {
							if (!passNames.empty()) {
								passNames += ", ";
							}
							passNames += passEntry->name.empty() ? std::string("<unnamed>") : passEntry->name;
						}, passVariant);
					}

					std::ostringstream oss;
					oss << "RenderGraph::Execute parallel recording failed for batch " << task.batchIndex
						<< ", queue " << QueueKindToString(queueKind)
						<< " (slot " << task.queueIndex << ")";
					if (!passNames.empty()) {
						oss << " with passes [" << passNames << "]";
					}
					oss << ": " << ex.what();
					throw std::runtime_error(oss.str());
				}
				}, false); // Toggle to true to force serial recording for easier debugging and validation
			if (batchTraceEnabled) {
				spdlog::info("RenderGraph::Execute frame={} record-all-batches complete", static_cast<unsigned>(context.frameIndex));
			}
		}

		// Merge per-task statistics contexts.
		if (statisticsService) {
			ZoneScopedN("RenderGraph::Execute::ParallelPath::MergePendingResolves");
			for (size_t bi = 0; bi < batches.size(); ++bi) {
				auto& batchSched = m_executionSchedule.batches[bi];
				for (size_t qi = 0; qi < batchSched.queues.size(); ++qi) {
					auto& qs = batchSched.queues[qi];
					if (!qs.active) continue;
					if (qs.queryRecordingContext.pendingRanges.empty()) continue;
					const auto kind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(qi));
					statisticsService->MergePendingResolves(
						static_cast<rhi::QueueKind>(kind), context.frameIndex,
						qs.queryRecordingContext);
				}
			}
		}

		// Sequential submission on the main thread.
		{
			ZoneScopedN("RenderGraph::Execute::ParallelPath::SubmitAllBatches");
			if (batchTraceEnabled) {
				spdlog::info("RenderGraph::Execute frame={} submit-all-batches begin", static_cast<unsigned>(context.frameIndex));
			}

			struct PendingQueueSubmission {
				std::vector<rhi::CommandList> pendingCommandLists;
				std::vector<CommandListPair> pendingPairs;
				std::vector<CommandListPair> submittedPairsAwaitingRecycle;

				bool HasPendingCommandLists() const {
					return !pendingCommandLists.empty();
				}

				bool HasOutstandingWork() const {
					return !pendingPairs.empty() || !submittedPairsAwaitingRecycle.empty();
				}
			};

			std::vector<PendingQueueSubmission> pendingSubmissions(slotCount);

			auto batchHasWaitsForQueue = [&](const PassBatch& batch, size_t queueIndex) {
				if (!batch.ExternalWaitsBeforeTransitions(queueIndex).empty()) {
					return true;
				}
				for (size_t waitPhaseIndex = 0; waitPhaseIndex < PassBatch::kWaitPhaseCount; ++waitPhaseIndex) {
					const auto waitPhase = static_cast<BatchWaitPhase>(waitPhaseIndex);
					for (size_t srcIndex = 0; srcIndex < batch.QueueCount(); ++srcIndex) {
						if (batch.HasQueueWait(waitPhase, queueIndex, srcIndex)) {
							return true;
						}
					}
				}
				return false;
			};

			auto submitPendingWithoutSignal = [&](size_t queueIndex, size_t batchIndex, const char* reason) {
				ZoneScopedN("RenderGraph::Execute::ParallelPath::SubmitPendingWithoutSignal");
				ZoneText(reason, std::strlen(reason));
				auto& pending = pendingSubmissions[queueIndex];
				if (!pending.HasPendingCommandLists()) {
					return;
				}

				auto rhiQueue = SlotQueue(queueIndex);
				const QueueKind queueKind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(queueIndex));
				ZoneText(QueueKindToString(queueKind), std::strlen(QueueKindToString(queueKind)));
				if (batchTraceEnabled) {
					spdlog::info(
						"RenderGraph::Execute frame={} submit pending queue {} slot {} batch {} reason={} clCount={}",
						static_cast<unsigned>(context.frameIndex),
						QueueKindToString(queueKind),
						queueIndex,
						batchIndex,
						reason,
						pending.pendingCommandLists.size());
				}

				rhiQueue.Submit({ pending.pendingCommandLists.data(), static_cast<uint32_t>(pending.pendingCommandLists.size()) }, {});
				for (auto& pair : pending.pendingPairs) {
					pending.submittedPairsAwaitingRecycle.push_back(std::move(pair));
				}
				pending.pendingPairs.clear();
				pending.pendingCommandLists.clear();
			};

			auto signalAndRecycleQueue = [&](size_t queueIndex, size_t batchIndex, UINT64 signalValue, const char* reason) {
				ZoneScopedN("RenderGraph::Execute::ParallelPath::SignalAndRecycleQueue");
				ZoneText(reason, std::strlen(reason));
				auto& pending = pendingSubmissions[queueIndex];
				if (!pending.HasOutstandingWork()) {
					return;
				}

				if (pending.HasPendingCommandLists()) {
					submitPendingWithoutSignal(queueIndex, batchIndex, reason);
				}

				auto rhiQueue = SlotQueue(queueIndex);
				auto& fenceTimeline = SlotFence(queueIndex);
				auto* pool = SlotPool(queueIndex);
				const QueueKind queueKind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(queueIndex));
				ZoneText(QueueKindToString(queueKind), std::strlen(QueueKindToString(queueKind)));

				if (signalValue == 0) {
					spdlog::error(
						"RenderGraph::Execute frame={} queue {} slot {} batch {} encountered zero signal for reason={} and is falling back to a monotonic recycle signal.",
						static_cast<unsigned>(context.frameIndex),
						QueueKindToString(queueKind),
						queueIndex,
						batchIndex,
						reason);
					if (lastSignaledPerSlot[queueIndex] == UINT64_MAX) {
						throw std::runtime_error(fmt::format(
							"RenderGraph::Execute cannot allocate fallback signal for slot {} batch {} reason {} because lastSignaled is UINT64_MAX",
							queueIndex,
							batchIndex,
							reason));
					}
					signalValue = lastSignaledPerSlot[queueIndex] + 1;
				}

				if (signalValue == UINT64_MAX) {
					throw std::runtime_error(fmt::format(
						"RenderGraph::Execute rejected terminal queue signal value: slot={} batch={} reason={}",
						queueIndex,
						batchIndex,
						reason));
				}

				if (batchTraceEnabled) {
					spdlog::info(
						"RenderGraph::Execute frame={} signal queue {} slot {} batch {} reason={} fence={} pendingPairs={}",
						static_cast<unsigned>(context.frameIndex),
						QueueKindToString(queueKind),
						queueIndex,
						batchIndex,
						reason,
						signalValue,
						pending.submittedPairsAwaitingRecycle.size());
				}

				const rhi::Result signalResult = rhiQueue.Signal({ fenceTimeline.GetHandle(), signalValue });
				if (signalResult != rhi::Result::Ok) {
					throw std::runtime_error(fmt::format(
						"RenderGraph::Execute queue signal failed: slot={} batch={} reason={} value={} result={}",
						queueIndex,
						batchIndex,
						reason,
						signalValue,
						rhi::ResultName(signalResult)));
				}
				lastSignaledPerSlot[queueIndex] = std::max(lastSignaledPerSlot[queueIndex], signalValue);
				auto [slotSignalIt, insertedSlotSignal] = m_lastExternalSignalValueByTimeline.try_emplace(
					PackTimelineSignalKey(fenceTimeline.GetHandle()),
					0);
				slotSignalIt->second = std::max(slotSignalIt->second, signalValue);

				if (pool) {
					ZoneScopedN("RenderGraph::Execute::ParallelPath::SignalAndRecycleQueue::RecyclePairs");
					for (auto& pair : pending.submittedPairsAwaitingRecycle) {
						pool->Recycle(std::move(pair), signalValue);
					}
				}
				pending.submittedPairsAwaitingRecycle.clear();
			};

			auto flushExternalFencesForQueue = [&](size_t queueIndex, size_t batchIndex, std::vector<PassReturn>& externalFences) {
				ZoneScopedN("RenderGraph::Execute::ParallelPath::FlushExternalFencesForQueue");
				if (externalFences.empty()) {
					return;
				}

				submitPendingWithoutSignal(queueIndex, batchIndex, "ExternalFences");
				auto rhiQueue = SlotQueue(queueIndex);
				SignalExternalFences(
					rhiQueue,
					m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(queueIndex)),
					&SlotFence(queueIndex),
					queueIndex,
					batchIndex,
					static_cast<unsigned>(context.frameIndex),
					queueSlotFenceTimelineKeys,
					seenExternalFenceSignalsThisFrame,
					m_lastExternalSignalValueByTimeline,
					externalFences);
			};

			auto queueRecordedCommandList = [&](size_t queueIndex, CommandListPair&& pair) {
				ZoneScopedN("RenderGraph::Execute::ParallelPath::QueueRecordedCommandList");
				auto& pending = pendingSubmissions[queueIndex];
				pending.pendingCommandLists.push_back(pair.list.Get());
				pending.pendingPairs.push_back(std::move(pair));
			};

			auto applyBatchWaitPhase = [&](const PassBatch& batch, size_t batchIndex, size_t queueIndex, BatchWaitPhase waitPhase) {
				const char* waitPhaseLabel = "Unknown";
				switch (waitPhase) {
				case BatchWaitPhase::BeforeTransitions:
					waitPhaseLabel = "BeforeTransitions";
					break;
				case BatchWaitPhase::BeforeExecution:
					waitPhaseLabel = "BeforeExecution";
					break;
				case BatchWaitPhase::BeforeAfterPasses:
					waitPhaseLabel = "BeforeAfterPasses";
					break;
				default:
					break;
				}
				ZoneScopedN("RenderGraph::Execute::ParallelPath::ApplyBatchWaitPhase");
				ZoneText(waitPhaseLabel, std::strlen(waitPhaseLabel));
				if (waitPhase == BatchWaitPhase::BeforeTransitions) {
					WaitExternalFencesBeforeTransitions(
						SlotQueue(queueIndex),
						batch,
						queueIndex,
						batchIndex,
						static_cast<unsigned>(context.frameIndex));
				}
				for (size_t srcIndex = 0; srcIndex < batch.QueueCount(); ++srcIndex) {
					if (!batch.HasQueueWait(waitPhase, queueIndex, srcIndex)) {
						continue;
					}
					WaitOnSlot(queueIndex, srcIndex, batch.GetQueueWaitFenceValue(waitPhase, queueIndex, srcIndex));
				}
			};

			for (size_t bi = 0; bi < batches.size(); ++bi) {
				ZoneScopedN("RenderGraph::Execute::ParallelPath::SubmitBatch");
				auto& batch = batches[bi];
				auto& batchSched = m_executionSchedule.batches[bi];

				for (size_t qi = 0; qi < batchSched.queues.size(); ++qi) {
					ZoneScopedN("RenderGraph::Execute::ParallelPath::SubmitQueue");
					auto& qs = batchSched.queues[qi];
					if (!qs.active) continue;
					const QueueKind queueKind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(qi));
					ZoneText(QueueKindToString(queueKind), std::strlen(QueueKindToString(queueKind)));

					if (batchHasWaitsForQueue(batch, qi)) {
						submitPendingWithoutSignal(qi, bi, "BeforeQueueWaits");
					}

					uint8_t clIndex = 0;
					{
						ZoneScopedN("RenderGraph::Execute::ParallelPath::SubmitBatch::ApplyBeforeTransitionWaits");
						applyBatchWaitPhase(batch, bi, qi, BatchWaitPhase::BeforeTransitions);
					}
					if (qs.splitAfterTransitions) {
						queueRecordedCommandList(qi, std::move(qs.preallocatedCLs[clIndex]));
						signalAndRecycleQueue(
							qi,
							bi,
							batch.GetQueueSignalFenceValue(BatchSignalPhase::AfterTransitions, qi),
							"AfterTransitions");
						++clIndex;
					}

					applyBatchWaitPhase(batch, bi, qi, BatchWaitPhase::BeforeExecution);

					if (qs.splitAfterExecution) {
						ZoneScopedN("RenderGraph::Execute::ParallelPath::SubmitBatch::ApplyAfterExecutionWaits");
						queueRecordedCommandList(qi, std::move(qs.preallocatedCLs[clIndex]));
						signalAndRecycleQueue(
							qi,
							bi,
							batch.GetQueueSignalFenceValue(BatchSignalPhase::AfterExecution, qi),
							"AfterExecution");
						++clIndex;
					}

					{
						ZoneScopedN("RenderGraph::Execute::ParallelPath::SubmitBatch::ApplyBeforeAfterPassesWaits");
						applyBatchWaitPhase(batch, bi, qi, BatchWaitPhase::BeforeAfterPasses);
						queueRecordedCommandList(qi, std::move(qs.preallocatedCLs[clIndex]));
					}
					if (qs.signalAfterCompletion) {
						ZoneScopedN("RenderGraph::Execute::ParallelPath::SubmitBatch::SignalAfterCompletion");
						signalAndRecycleQueue(
							qi,
							bi,
							batch.GetQueueSignalFenceValue(BatchSignalPhase::AfterCompletion, qi),
							"AfterCompletion");
					}

					{
						ZoneScopedN("RenderGraph::Execute::ParallelPath::SubmitBatch::FlushExternalFences");
						flushExternalFencesForQueue(qi, bi, qs.externalFences);
					}
				}
			}

			for (size_t qi = 0; qi < slotCount; ++qi) {
				ZoneScopedN("RenderGraph::Execute::ParallelPath::EndOfFrameRecycleQueue");
				auto& pending = pendingSubmissions[qi];
				if (!pending.HasOutstandingWork()) {
					continue;
				}
				const QueueKind queueKind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(qi));
				ZoneText(QueueKindToString(queueKind), std::strlen(QueueKindToString(queueKind)));
				if (lastSignaledPerSlot[qi] >= UINT64_MAX - 1) {
					throw std::runtime_error(fmt::format(
						"RenderGraph::Execute cannot allocate end-of-frame recycle signal for slot {} because lastSignaled={}",
						qi,
						lastSignaledPerSlot[qi]));
				}
				signalAndRecycleQueue(qi, batches.size(), lastSignaledPerSlot[qi] + 1, "EndOfFrameRecycle");
			}
			if (batchTraceEnabled) {
				spdlog::info("RenderGraph::Execute frame={} submit-all-batches complete", static_cast<unsigned>(context.frameIndex));
			}
		}
	}

	auto passDeclaresPresent = [](const PassBatch::QueuedPass& queuedPass) -> bool {
		return std::visit([](auto* passAndResources) -> bool {
			using PassPtr = std::decay_t<decltype(passAndResources)>;
			if constexpr (std::is_same_v<PassPtr, RenderPassAndResources*>) {
				return passAndResources && !passAndResources->resources.presentResources.empty();
			} else {
				return false;
			}
		}, queuedPass);
	};

	for (size_t bi = 0; bi < batches.size(); ++bi) {
		const auto& batch = batches[bi];
		for (size_t qi = 0; qi < batch.QueueCount(); ++qi) {
			const auto& queuedPasses = batch.queuePasses[qi];
			if (std::any_of(queuedPasses.begin(), queuedPasses.end(), passDeclaresPresent)) {
				const auto queueSlot = static_cast<QueueSlotIndex>(static_cast<uint8_t>(qi));
				m_lastPresentDependency = PresentDependency{
					.queue = SlotQueue(qi),
					.wait = { SlotFence(qi).GetHandle(), lastSignaledPerSlot[qi] },
					.queueSlot = queueSlot,
					.batchIndex = bi,
					.valid = lastSignaledPerSlot[qi] != 0,
				};
			}
		}
	}
	if (batchTraceEnabled && m_lastPresentDependency) {
		const size_t queueIndex = static_cast<size_t>(static_cast<uint8_t>(m_lastPresentDependency->queueSlot));
		spdlog::info(
			"RenderGraph::Execute frame={} present dependency batch={} queue={} slot={} fence={}",
			static_cast<unsigned>(context.frameIndex),
			m_lastPresentDependency->batchIndex,
			QueueKindToString(m_queueRegistry.GetKind(m_lastPresentDependency->queueSlot)),
			queueIndex,
			m_lastPresentDependency->wait.value);
	}

	{
		ZoneScopedN("RenderGraph::Execute::UpdateCrossFrameProducerTracking");
		const uint64_t publishSerial = ++m_crossFrameProducerPublishSerial;
		auto isAnonymousTrackedResource = [&](uint64_t resourceID) {
			auto itTransient = m_transientFrameResourcesByID.find(resourceID);
			if (itTransient != m_transientFrameResourcesByID.end() && itTransient->second) {
				return _registry.IsAnonymous(itTransient->second.get());
			}

			auto itResource = resourcesByID.find(resourceID);
			if (itResource != resourcesByID.end() && itResource->second) {
				return _registry.IsAnonymous(itResource->second.get());
			}

			return false;
		};

		// Update across-frame producer tracking (no aliasing remapping).
		// Publish the end-of-frame signal for each queue that produced cross-frame
		// resources. Timeline signals are monotonic, so waiting on a resource's
		// producer can safely wait for the queue's final signal from the frame.
		for (size_t queueIndex = 0; queueIndex < slotCount; ++queueIndex) {
			if (queueIndex >= m_compiledLastProducerBatchByResourceByQueue.size()) continue;
			if (m_compiledLastProducerBatchByResourceByQueue[queueIndex].empty()) continue;

			const unsigned int signalBatch = lastCrossFrameSignalBatchByQueue[queueIndex];
			if (signalBatch == 0 || signalBatch >= batches.size()) {
				spdlog::warn(
					"Cross-frame producer skip: slot {} has {} tracked resources but no active batch signal.",
					queueIndex,
					m_compiledLastProducerBatchByResourceByQueue[queueIndex].size());
				continue;
			}

			const uint64_t fenceValue =
				batches[signalBatch].GetQueueSignalFenceValue(BatchSignalPhase::AfterCompletion, queueIndex);

			// Skip if this fence value was never actually signaled on the GPU.
			if (fenceValue > lastSignaledPerSlot[queueIndex]) {
				spdlog::warn(
					"Cross-frame producer skip: slot {} signal batch {} fenceValue={} > lastSignaled={}",
					queueIndex,
					signalBatch,
					fenceValue,
					lastSignaledPerSlot[queueIndex]);
				continue;
			}

			for (const auto& [resourceID, producerBatch] : m_compiledLastProducerBatchByResourceByQueue[queueIndex]) {
				if (producerBatch == 0 || producerBatch >= batches.size()) continue;

				LastProducerAcrossFrames producer{
					.queueSlot = queueIndex,
					.fenceValue = fenceValue,
					.publishSerial = publishSerial,
					.anonymous = isAnonymousTrackedResource(resourceID),
				};
				nextLastProducerByResourceAcrossFrames[resourceID] = producer;
				publishAliasPlacementProducer(resourceID, producer);
			}
		}
	}

	{
		ZoneScopedN("RenderGraph::Execute::PruneCrossFrameProducerTracking");
		auto isLiveResourceID = [&](uint64_t resourceID) {
			auto itResource = resourcesByID.find(resourceID);
			if (itResource != resourcesByID.end() && itResource->second) {
				return true;
			}

			auto itTransient = m_transientFrameResourcesByID.find(resourceID);
			return itTransient != m_transientFrameResourcesByID.end() && itTransient->second;
		};

		std::unordered_set<uint64_t> liveResourceIDs;
		liveResourceIDs.reserve(resourcesByID.size() + m_transientFrameResourcesByID.size());

		for (const auto& [resourceID, resource] : resourcesByID) {
			if (resource) {
				liveResourceIDs.insert(resourceID);
			}
		}
		for (const auto& [resourceID, resource] : m_transientFrameResourcesByID) {
			if (resource) {
				liveResourceIDs.insert(resourceID);
			}
		}

		std::erase_if(
			nextLastProducerByResourceAcrossFrames,
			[&](const auto& entry) {
				if (entry.second.anonymous) {
					return entry.second.publishSerial != m_crossFrameProducerPublishSerial;
				}
				return !liveResourceIDs.contains(entry.first) && !isLiveResourceID(entry.first);
			});

		for (auto itPool = nextLastAliasPlacementProducersByPoolAcrossFrames.begin();
			 itPool != nextLastAliasPlacementProducersByPoolAcrossFrames.end();) {
			auto& producers = itPool->second;
			std::erase_if(
				producers,
				[&](const LastAliasPlacementProducerAcrossFrames& producer) {
					if (producer.producer.anonymous) {
						return producer.producer.publishSerial != m_crossFrameProducerPublishSerial;
					}
					return !liveResourceIDs.contains(producer.resourceID) && !isLiveResourceID(producer.resourceID);
				});

			if (producers.empty()) {
				itPool = nextLastAliasPlacementProducersByPoolAcrossFrames.erase(itPool);
			}
			else {
				++itPool;
			}
		}
	}

	// Sync CRM signal tracking with values we signaled directly.
	// Capped at primary queue count: CRM only tracks the 3 primary queues (Graphics/Compute/Copy).
	for (size_t qi = 0; qi < std::min(slotCount, static_cast<size_t>(QueueKind::Count)); ++qi) {
		UINT64 val = lastSignaledPerSlot[qi];
		if (val > 0) {
			crm->EnsureMinSignaledValue(static_cast<QueueKind>(qi), val);
		}
	}

	{
		ZoneScopedN("RenderGraph::Execute::FinalizeCommandRecording");
		if (batchTraceEnabled) {
			spdlog::info("RenderGraph::Execute frame={} finalizing CRM", static_cast<unsigned>(context.frameIndex));
		}
		crm->Flush(QueueKind::Graphics, { false, 0 });
		crm->Flush(QueueKind::Compute, { false, 0 });
		crm->Flush(QueueKind::Copy, { false, 0 });
		PublishCompiledTrackerStates();
		crm->EndFrame();
	}

	{
		ZoneScopedN("RenderGraph::Execute::RecycleCompletedCommandLists");
		// Recycle completed command lists on registry pools.
		// CRM::EndFrame() only recycles the old member pools; the registry owns
		// separate pools that receive Recycle() calls during execution above.
		for (size_t qi = 0; qi < slotCount; ++qi) {
			auto* pool = SlotPool(qi);
			if (pool) {
				uint64_t completed = SlotFence(qi).GetCompletedValue();
				pool->RecycleCompleted(completed);
			}
		}
	}
	if (batchTraceEnabled) {
		spdlog::info("RenderGraph::Execute end frame={}", static_cast<unsigned>(context.frameIndex));
	}
}

bool RenderGraph::IsNewBatchNeeded(
	const FramePassSchedulingSummary& passSummary,
	const std::vector<SymbolicTracker*>& passBatchTrackersByResourceIndex,
	const BatchBuildState& batchBuildState,
	std::string_view candidatePassName,
	unsigned int currentBatchIndex,
	size_t candidateQueueSlot)
{
	ZoneScopedN("RenderGraph::IsNewBatchNeeded");
	if (!candidatePassName.empty()) {
		ZoneText(candidatePassName.data(), candidatePassName.size());
	}
	auto overlapsAliasedResourceInBatch = [&](const auto& summaryEntry) {
		if (!summaryEntry.equivalentResourceIndices) {
			return false;
		}
		for (size_t equivalentResourceIndex : *summaryEntry.equivalentResourceIndices) {
			if (batchBuildState.ContainsResource(equivalentResourceIndex)) {
				return true;
			}
		}
		return false;
	};

	auto overlapsAliasedTransitionInBatch = [&](const auto& summaryEntry) {
		if (!summaryEntry.equivalentResourceIndices) {
			return false;
		}
		for (size_t equivalentResourceIndex : *summaryEntry.equivalentResourceIndices) {
			if (batchBuildState.ContainsInternalTransition(equivalentResourceIndex)) {
				return true;
			}
		}
		return false;
	};

	// For each internally modified resource
	for (const auto& transition : passSummary.internalTransitions) {
		// If this resource is used in the current batch, we need a new one
		if (batchBuildState.ContainsResource(transition.resourceIndex)) {
			return true;
		}
		if (overlapsAliasedResourceInBatch(transition)) {
			return true;
		}
	}

	// For each subresource requirement in this pass:
	for (const auto& requirement : passSummary.requirements) {
		const uint64_t id = requirement.resourceID;

		// Alias activations are emitted in BeforePasses of the consuming batch.
		// Only reject same-batch merging when that activation would clobber an
		// aliased-equivalent resource that is already live in the batch.
		if (requirement.resourceIndex < m_aliasActivationPendingByResourceIndex.size()
			&& m_aliasActivationPendingByResourceIndex[requirement.resourceIndex] != 0
			&& (overlapsAliasedResourceInBatch(requirement) || overlapsAliasedTransitionInBatch(requirement))) {
			return true;
		}

		// If this resource is internally modified in the current batch, we need a new one
		if (batchBuildState.ContainsInternalTransition(requirement.resourceIndex)) {
			return true;
		}
		if (overlapsAliasedResourceInBatch(requirement) || overlapsAliasedTransitionInBatch(requirement)) {
			return true;
		}

		ResourceState wantState{ requirement.state.access, requirement.state.layout, requirement.state.sync };

		// Changing state?
		SymbolicTracker* tracker = nullptr;
		if (requirement.resourceIndex < passBatchTrackersByResourceIndex.size()) {
			tracker = passBatchTrackersByResourceIndex[requirement.resourceIndex];
		}
		if (tracker && tracker->WouldModify(requirement.range, wantState)) {
			return true;
		}
		// first-use in this batch never forces a split.

		// Reusing the same UAV in later passes of the same batch requires a UAV
		// barrier even when the logical state remains UnorderedAccess. The batch
		// model only inserts state transitions at batch boundaries, so keep each
		// same-resource UAV use in its own batch.
		if (requirement.isUAV && batchBuildState.ContainsResource(requirement.resourceIndex)) {
			return true;
		}

		// Cross-queue UAV hazard?
		if (requirement.isUAV && batchBuildState.ContainsOtherQueueUAV(candidateQueueSlot, requirement.resourceIndex)) {
			return true;
		}
	}
	return false;
}

void RenderGraph::RegisterProvider(IResourceProvider* prov) {
	EnsureProviderRegistered(prov);
}

void RenderGraph::EnsureProviderRegistered(IResourceProvider* prov) {
	if (!prov) {
		return;
	}

	auto keys = prov->GetSupportedKeys();
	for (const auto& key : keys) {
		auto existing = _providerMap.find(key);
		if (existing != _providerMap.end()) {
			if (existing->second == prov) {
				continue;
			}
			std::string name = key.ToString();
			throw std::runtime_error("Resource provider already registered for key: " + name);
		}
		_providerMap[key] = prov;
	}
	if (std::find(_providers.begin(), _providers.end(), prov) == _providers.end()) {
		_providers.push_back(prov);
	}

	for (const auto& key : prov->GetSupportedKeys()) {
		if (_registry.GetHandleFor(key).has_value()) {
			continue;
		}

		auto resource = prov->ProvideResource(key);
		if (resource) {
			RegisterResource(key, resource, prov);
		}
		else {
			spdlog::warn("Provider returned null for advertised key: {}", key.ToString());
		}
	}

	// Register resolvers from this provider
	for (const auto& key : prov->GetSupportedResolverKeys()) {
		if (_resolverMap.contains(key)) {
			continue;
		}

		if (const auto resolver = prov->ProvideResolver(key); resolver) {
			RegisterResolver(key, resolver);
		}
		else {
			spdlog::warn("Provider returned null resolver for advertised key: {}", key.ToString());
		}
	}
}

void RenderGraph::RegisterResolver(ResourceIdentifier id, const std::shared_ptr<IResourceResolver>& resolver) {
	if (_resolverMap.contains(id)) {
		throw std::runtime_error("Resolver already registered for key: " + id.ToString());
	}
	// Resolve it and register its resources
	for (const auto& resource : resolver->Resolve()) {
		if (resource) {
			resourcesByID[resource->GetGlobalResourceID()] = resource;
			if (auto dynamicResource = std::dynamic_pointer_cast<DynamicResource>(resource)) {
				m_dynamicResourcesByStableID[dynamicResource->GetDynamicWrapperGlobalResourceID()] = resource;
			}
			// Anonymous registration
			_registry.RegisterAnonymous(resource);
		}
	}
	_resolverMap[id] = resolver;
	_registry.RegisterResolver(id, resolver);
}

std::shared_ptr<IResourceResolver> RenderGraph::RequestResolver(ResourceIdentifier const& rid, bool allowFailure) {
	if (auto it = _resolverMap.find(rid); it != _resolverMap.end()) {
		return it->second;
	}

	if (allowFailure) return nullptr;
	throw std::runtime_error("No resolver registered for key: " + rid.ToString());
}

void RenderGraph::RegisterResource(ResourceIdentifier id, std::shared_ptr<Resource> resource,
	IResourceProvider* provider) {
	const bool traceLifecycle = m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled();
	if (traceLifecycle) {
		spdlog::info(
			"RenderGraph::RegisterResource key='{}' id={} name='{}' provider={}",
			id.ToString(),
			resource ? resource->GetGlobalResourceID() : 0ull,
			resource ? resource->GetName() : std::string{},
			static_cast<const void*>(provider));
	}

	auto key = _registry.RegisterOrUpdate(id, resource);
	AddResource(resource);
	if (provider) {
		_providerMap[id] = provider;
	}

	// If resource can be cast to IHasMemoryMetadata, tag it with this ResouceIdentifier
	if (const auto hasMemoryMetadata = std::dynamic_pointer_cast<IHasMemoryMetadata>(resource); hasMemoryMetadata) {
		hasMemoryMetadata->ApplyMetadataComponentBundle(EntityComponentBundle().Set<ResourceIdentifier>(id));
	}
}

std::shared_ptr<Resource> RenderGraph::RequestResourcePtr(ResourceIdentifier const& rid, bool allowFailure) {
	// If it's already in our registry, return it
	auto cached = _registry.RequestShared(rid);
	if (cached) {
		return cached;
	}

	// We don't have it in our registry, check if we have a provider for it
	auto providerIt = _providerMap.find(rid);
	if (providerIt != _providerMap.end()) {
		// If we have a provider for this key, use it to provide the resource
		auto provider = providerIt->second;
		if (provider) {
			auto resource = provider->ProvideResource(rid);
			if (resource) {
				// Register the resource in our registry
				_registry.RegisterOrUpdate(rid, resource);
				AddResource(resource);
				return resource;
			}
			else {
				throw std::runtime_error("Provider returned null for key: " + rid.ToString());
			}
		}
	}

	// No provider registered for this key
	if (allowFailure) {
		// If we are allowed to fail, return nullptr
		return nullptr;
	}
	throw std::runtime_error("No resource provider registered for key: " + rid.ToString());
}

ResourceRegistry::RegistryHandle RenderGraph::RequestResourceHandle(ResourceIdentifier const& rid, bool allowFailure) {
	// If it's already in our registry, return it
	auto cached = _registry.GetHandleFor(rid);
	if (cached.has_value()) {
		return cached.value();
	}

	// We don't have it in our registry, check if we have a provider for it
	auto providerIt = _providerMap.find(rid);
	if (providerIt != _providerMap.end()) {
		// If we have a provider for this key, use it to provide the resource
		auto provider = providerIt->second;
		if (provider) {
			auto resource = provider->ProvideResource(rid);
			if (resource) {
				// Register the resource in our registry
				_registry.RegisterOrUpdate(rid, resource);
				AddResource(resource);
				return _registry.GetHandleFor(rid).value();
			}
			else {
				throw std::runtime_error("Provider returned null for key: " + rid.ToString());
			}
		}
	}

	// No provider registered for this key
	if (allowFailure) {
		// If we are allowed to fail, return nullptr
		return {};
	}
	throw std::runtime_error("No resource provider registered for key: " + rid.ToString());
}

ResourceRegistry::RegistryHandle RenderGraph::RequestResourceHandle(Resource* const& pResource, bool allowFailure) {
	if (!pResource) {
		if (allowFailure) {
			return {};
		}
		throw std::runtime_error("Null resource pointer passed to RequestResourceHandle(Resource*)");
	}

	auto pinTransientForFrame = [&]() {
		const uint64_t resourceID = pResource->GetGlobalResourceID();
		if (resourcesByID.contains(resourceID) || m_transientFrameResourcesByID.contains(resourceID)) {
			return;
		}

		TrackTransientFrameResource(pResource);
	};

	// If it's already in our registry, return it
	auto cached = _registry.GetHandleFor(pResource);
	if (cached.has_value()) {
		if (_registry.IsValid(cached.value())) {
			pinTransientForFrame();
			return cached.value();
		}

		//spdlog::warn(
		//	"Stale cached registry handle for resource '{}' (id={}) detected; reminting anonymous handle.",
		//	pResource ? pResource->GetName() : std::string("<null>"),
		//	pResource ? pResource->GetGlobalResourceID() : 0ull);

		// Fall through and remint a fresh handle for this live resource pointer.
		// This can happen if a resource was replaced but an old reverse-map entry remained.
		const auto reminted = _registry.RegisterAnonymousWeak(pResource->weak_from_this());
		pinTransientForFrame();
		return reminted;
	}

	if (allowFailure) {
		return {};
	}

	// Register anonymous resource
	const auto handle = _registry.RegisterAnonymousWeak(pResource->weak_from_this());
	pinTransientForFrame();

	return handle;
}


ComputePassBuilder& RenderGraph::GetOrCreateComputePassBuilder(std::string const& name) {
	if (auto it = m_passBuildersByName.find(name); it != m_passBuildersByName.end()) {
		if (m_passNamesSeenThisReset.contains(name)) {
			throw std::runtime_error("Pass names must be unique.");
		}
		if (it->second->Kind() != PassBuilderKind::Compute) {
			throw std::runtime_error("Pass builder name collision (render/compute/copy): " + name);
		}
		m_passBuilderOrder.push_back(it->second.get());
		return static_cast<ComputePassBuilder&>(*(it->second));
	}
	m_passNamesSeenThisReset.insert(name);
	auto ptr = std::unique_ptr<ComputePassBuilder>(new ComputePassBuilder(this, name));
	m_passBuilderOrder.push_back(ptr.get());
	m_passBuildersByName.emplace(name, std::move(ptr));
	return static_cast<ComputePassBuilder&>(*(m_passBuildersByName[name]));
}
RenderPassBuilder& RenderGraph::GetOrCreateRenderPassBuilder(std::string const& name) {
	if (auto it = m_passBuildersByName.find(name); it != m_passBuildersByName.end()) {
		if (m_passNamesSeenThisReset.contains(name)) {
			throw std::runtime_error("Pass names must be unique.");
		}
		if (it->second->Kind() != PassBuilderKind::Render) {
			throw std::runtime_error("Pass builder name collision (render/compute/copy): " + name);
		}
		m_passBuilderOrder.push_back(it->second.get());
		return static_cast<RenderPassBuilder&>(*(it->second));
	}
	m_passNamesSeenThisReset.insert(name);
	auto ptr = std::unique_ptr<RenderPassBuilder>(new RenderPassBuilder(this, name));
	m_passBuilderOrder.push_back(ptr.get());
	m_passBuildersByName.emplace(name, std::move(ptr));
	return static_cast<RenderPassBuilder&>(*(m_passBuildersByName[name]));
}

CopyPassBuilder& RenderGraph::GetOrCreateCopyPassBuilder(std::string const& name) {
	if (auto it = m_passBuildersByName.find(name); it != m_passBuildersByName.end()) {
		if (m_passNamesSeenThisReset.contains(name)) {
			throw std::runtime_error("Pass names must be unique.");
		}
		if (it->second->Kind() != PassBuilderKind::Copy) {
			throw std::runtime_error("Pass builder name collision (render/compute/copy): " + name);
		}
		m_passBuilderOrder.push_back(it->second.get());
		return static_cast<CopyPassBuilder&>(*(it->second));
	}
	m_passNamesSeenThisReset.insert(name);
	auto ptr = std::unique_ptr<CopyPassBuilder>(new CopyPassBuilder(this, name));
	m_passBuilderOrder.push_back(ptr.get());
	m_passBuildersByName.emplace(name, std::move(ptr));
	return static_cast<CopyPassBuilder&>(*(m_passBuildersByName[name]));
}

//void RenderGraph::RegisterPassBuilder(RenderPassBuilder&& builder) {
//	m_passBuildersByName[builder.passName] = std::move(builder);
//}
//void RenderGraph::RegisterPassBuilder(ComputePassBuilder&& builder) {
//	m_passBuildersByName[builder.passName] = std::move(builder);
//}

QueueSlotIndex RenderGraph::CreateQueue(QueueKind kind, const char* name, QueueAutoAssignmentPolicy autoAssignmentPolicy) {
	auto device = DeviceManager::GetInstance().GetDevice();
	rhi::Queue queue;
	auto result = device.CreateQueue(static_cast<rhi::QueueKind>(kind), name ? name : "UserQueue", queue);
	if (result != rhi::Result::Ok) {
		throw std::runtime_error(fmt::format(
			"Failed to create queue '{}' for kind {}: {}",
			name ? name : "UserQueue",
			static_cast<int>(kind),
			rhi::ResultName(result)));
	}

	// Determine instance number: count existing slots of this kind.
	uint8_t instance = 0;
	for (size_t i = 0; i < m_queueRegistry.SlotCount(); ++i) {
		if (m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(i))) == kind)
			++instance;
	}
	return m_queueRegistry.Register({ kind, instance }, queue, device, autoAssignmentPolicy, true);
}

void RenderGraph::SetMinimumAutomaticSchedulingQueues(QueueKind kind, uint8_t count) {
	const size_t kindIndex = static_cast<size_t>(kind);
	const uint8_t clampedCount = kind == QueueKind::Graphics ? (std::max)(uint8_t(1), count) : (std::max)(uint8_t(1), count);
	m_minAutomaticSchedulingQueuesByKind[kindIndex] = clampedCount;

	if (m_queueRegistry.SlotCount() > 0) {
		EnsureMinimumAutomaticSchedulingQueues();
	}
}
