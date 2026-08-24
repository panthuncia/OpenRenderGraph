#include "Render/RenderGraph/RenderGraph.h"
#include "RenderGraphCompilerState.h"
#include "Render/RenderGraph/InteropAllocator.h"

#include <span>
#include <algorithm>
#include <cmath>
#include <map>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <BasicTelemetry/Tracy.h>
#include <rhi_helpers.h>
#include <rhi_debug.h>
#include <rhi_interop_dx12.h>
#include <rhi_interop_vulkan.h>
#include <random>

#include "Render/PassExecutionContext.h"
#include "Utilities/ORGUtilities.h"
#include "Managers/Singletons/DeviceManager.h"
#include "Managers/Singletons/DeletionManager.h"
#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Managers/Singletons/UploadManager.h"
#include "Managers/Singletons/StatisticsManager.h"
#include "Render/PassBuilders.h"
#include "Resources/ResourceGroup.h"
#include "Managers/CommandRecordingManager.h"
#include "Interfaces/IHasMemoryMetadata.h"
#include "Interfaces/IDynamicDeclaredResources.h"
#include "Resources/DynamicResource.h"
#include "Resources/ExternalTextureResource.h"
#include "Resources/BackedResource.h"
#include "Resources/PixelBuffer.h"
#include "Resources/Buffers/DynamicBufferBase.h"
#include "Resources/MemoryStatisticsComponents.h"


namespace org {

namespace {
size_t PassRecordingConcurrency()
{
	static const size_t value = [] {
		const char* configured = std::getenv("ORG_PASS_RECORD_CONCURRENCY");
		if (!configured || !*configured) return (std::numeric_limits<size_t>::max)();
		char* end = nullptr;
		const auto parsed = std::strtoull(configured, &end, 10);
		return end != configured && *end == '\0' && parsed > 0
			? static_cast<size_t>(parsed)
			: (std::numeric_limits<size_t>::max)();
	}();
	return value;
}
}

namespace {
	constexpr uint64_t kFrameDAGResourceIndexEmptyKey = std::numeric_limits<uint64_t>::max();

	bool SarpClodImportDebugLoggingEnabled()
	{
		static const bool enabled = [] {
			char* value = nullptr;
			size_t length = 0;
			if (_dupenv_s(&value, &length, "SARP_DEBUG_CLOD_IMPORT") != 0 || value == nullptr) {
				return false;
			}
			const bool result = length > 1 && value[0] != '0';
			std::free(value);
			return result;
		}();
		return enabled;
	}

	uint64_t MixFrameDAGResourceID(uint64_t value) noexcept {
		value ^= value >> 33;
		value *= 0xff51afd7ed558ccdull;
		value ^= value >> 33;
		value *= 0xc4ceb9fe1a85ec53ull;
		value ^= value >> 33;
		return value;
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

	const char* CommandListPhaseName(const QueueBatchSchedule& schedule, uint8_t commandListIndex) noexcept {
		if (!schedule.splitAfterTransitions && !schedule.splitAfterExecution) {
			return "WholeBatch";
		}

		uint8_t index = 0;
		if (schedule.splitAfterTransitions) {
			if (commandListIndex == index) {
				return "BeforePasses";
			}
			++index;
		}

		if (schedule.splitAfterExecution) {
			if (commandListIndex == index) {
				return "Passes";
			}
			++index;
		}

		return "AfterPasses";
	}

	const char* DebugQueueKindName(QueueKind queue) noexcept {
		switch (queue) {
		case QueueKind::Graphics: return "Graphics";
		case QueueKind::Compute: return "Compute";
		case QueueKind::Copy: return "Copy";
		default: return "Unknown";
		}
	}

	std::string MakeRenderGraphCommandListName(
		unsigned frameIndex,
		size_t batchIndex,
		size_t queueSlot,
		QueueKind queue,
		const QueueBatchSchedule& schedule,
		uint8_t commandListIndex)
	{
		std::ostringstream oss;
		oss << "ORG frame=" << frameIndex
			<< " batch=" << batchIndex
			<< " queue=" << DebugQueueKindName(queue)
			<< " slot=" << queueSlot
			<< " cl=" << static_cast<unsigned>(commandListIndex)
			<< "/" << static_cast<unsigned>(schedule.numCLs)
			<< " phase=" << CommandListPhaseName(schedule, commandListIndex);
		return oss.str();
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

	const char* TransitionPlacementModeToString(org::runtime::TransitionPlacementMode mode) noexcept {
		switch (mode) {
		case org::runtime::TransitionPlacementMode::InlineEarlyPlacement: return "InlineEarlyPlacement";
		case org::runtime::TransitionPlacementMode::CanonicalThenOptimize: return "CanonicalThenOptimize";
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
			par.resources.externalWaitsBeforeTransitions = b.params.externalWaitsBeforeTransitions;
			par.resources.externalWaitBindingsBeforeTransitions = b.params.externalWaitBindingsBeforeTransitions;
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
			UpdateRetainedDeclarationCache(PassType::Render, par.name, par);
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
			par.resources.externalWaitsBeforeTransitions = b.params.externalWaitsBeforeTransitions;
			par.resources.externalWaitBindingsBeforeTransitions = b.params.externalWaitBindingsBeforeTransitions;
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
			UpdateRetainedDeclarationCache(PassType::Compute, par.name, par);
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
			par.resources.externalWaitsBeforeTransitions = b.params.externalWaitsBeforeTransitions;
			par.resources.externalWaitBindingsBeforeTransitions = b.params.externalWaitBindingsBeforeTransitions;
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
			UpdateRetainedDeclarationCache(PassType::Copy, par.name, par);
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
			 << " last_accesses=" << m_lastAccessByResourceAcrossFrames.size()
			 << " alias_pools=" << m_lastAliasPlacementProducersByPoolAcrossFrames.size()
			 << "\n";
		for (size_t queueIndex = 0; queueIndex < m_compilerState->compiledLastProducerBatchByResourceByQueue.size(); ++queueIndex) {
			const auto& producers = m_compilerState->compiledLastProducerBatchByResourceByQueue[queueIndex];
			if (producers.empty()) {
				continue;
			}
			dump << "  current_frame_producers queue=" << queueSlotLabel(queueIndex)
				 << " resources=" << producers.size()
				 << "\n";
		}
		for (size_t queueIndex = 0; queueIndex < m_compilerState->compiledLastAccessBatchByResourceByQueue.size(); ++queueIndex) {
			const auto& accesses = m_compilerState->compiledLastAccessBatchByResourceByQueue[queueIndex];
			if (!accesses.empty()) {
				dump << "  current_frame_accesses queue=" << queueSlotLabel(queueIndex)
					 << " resources=" << accesses.size()
					 << "\n";
			}
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
			std::map<uint64_t, std::vector<std::pair<uint64_t, const org::alias::AliasPlacementRange*>>> byPool;
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
						 << " overlaps=" << (placement->overlapsByteRange ? 1 : 0)
						 << " activationReason=" << static_cast<uint32_t>(placement->activationReasonBits)
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

		std::uniform_int_distribution<> distr(2, 2);

		
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

		std::vector<org::memory::ResourceMemoryRecord> memoryRecords;
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

void RenderGraph::BuildNodes(RenderGraph& rg, std::vector<Node>& nodes) {
	BT_ZONE_SCOPE("RenderGraph::BuildNodes");

	nodes.resize(rg.m_framePassAccessSummaries.size());
	const size_t slotCount = rg.m_queueRegistry.SlotCount();
	constexpr size_t passTypeCount = static_cast<size_t>(PassType::Copy) + 1;
	constexpr size_t maxQueueSlotCount = static_cast<size_t>(std::numeric_limits<uint8_t>::max()) + 1;
	struct QueueCompatibilityCache {
		std::array<std::array<size_t, maxQueueSlotCount>, static_cast<size_t>(QueueKind::Count)> autoAssignableByKind{};
		std::array<size_t, static_cast<size_t>(QueueKind::Count)> autoAssignableByKindCount{};
		std::array<std::array<size_t, maxQueueSlotCount>, passTypeCount> automaticByPassType{};
		std::array<size_t, passTypeCount> automaticByPassTypeCount{};
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

		auto& kindCount = queueCache.autoAssignableByKindCount[QueueIndex(kind)];
		queueCache.autoAssignableByKind[QueueIndex(kind)][kindCount++] = slotIndex;
		for (PassType type : { PassType::Render, PassType::Compute, PassType::Copy }) {
			if (IsPreferredQueueKindCompatible(type, kind)) {
				auto& typeCount = queueCache.automaticByPassTypeCount[passTypeIndex(type)];
				queueCache.automaticByPassType[passTypeIndex(type)][typeCount++] = slotIndex;
			}
		}
	}

	auto resolveCompatibleQueueSlotsForPass = [&queueCache, &passTypeIndex, &rg](
		const FramePassStaticAccessSummary& passAccess,
		std::vector<size_t>& compatibleSlots) {
		compatibleSlots.clear();
		if (passAccess.pinnedQueueSlot) {
			compatibleSlots.push_back(static_cast<size_t>(static_cast<uint8_t>(*passAccess.pinnedQueueSlot)));
			return;
		}

		if (passAccess.queueAssignmentPolicy == QueueAssignmentPolicy::Automatic) {
			const size_t typeIndex = passTypeIndex(passAccess.type);
			const size_t count = queueCache.automaticByPassTypeCount[typeIndex];
			if (count != 0) {
				compatibleSlots.assign(
					queueCache.automaticByPassType[typeIndex].begin(),
					queueCache.automaticByPassType[typeIndex].begin() + count);
			}
		}

		if (compatibleSlots.empty()) {
			const size_t kindIndex = QueueIndex(passAccess.preferredQueueKind);
			const size_t count = queueCache.autoAssignableByKindCount[kindIndex];
			if (count != 0) {
				compatibleSlots.assign(
					queueCache.autoAssignableByKind[kindIndex].begin(),
					queueCache.autoAssignableByKind[kindIndex].begin() + count);
			}
			else {
				compatibleSlots.push_back(kindIndex);
			}
		}

		const auto unfiltered = compatibleSlots;
		auto filter = [&](auto&& reject) {
			compatibleSlots.erase(std::remove_if(compatibleSlots.begin(), compatibleSlots.end(), [&](size_t slot) {
				if (slot >= rg.m_queueRegistry.SlotCount()) return true;
				return reject(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot)));
			}), compatibleSlots.end());
		};
		if (passAccess.backendAffinity.strength == BackendAffinityStrength::Primary) {
			filter([&](QueueSlotIndex slot) { return rg.m_queueRegistry.GetBackendInstance(slot) != rg.m_backendDevices.PrimaryId(); });
		}
		else {
			if (passAccess.backendAffinity.device) {
				filter([&](QueueSlotIndex slot) { return rg.m_queueRegistry.GetBackendInstance(slot) != *passAccess.backendAffinity.device; });
			} else {
				filter([&](QueueSlotIndex slot) { return rg.m_queueRegistry.GetBackend(slot) != passAccess.backendAffinity.backend; });
			}
			if (compatibleSlots.empty() && passAccess.backendAffinity.strength == BackendAffinityStrength::Preferred) {
				compatibleSlots = unfiltered;
				filter([&](QueueSlotIndex slot) { return rg.m_queueRegistry.GetBackendInstance(slot) != rg.m_backendDevices.PrimaryId(); });
			}
		}
	};

	for (size_t i = 0; i < rg.m_framePassAccessSummaries.size(); ++i) {
		const auto& passAccess = rg.m_framePassAccessSummaries[i];
		auto& n = nodes[i];
		n.out.clear();
		n.in.clear();
		n.compatibleQueueKindMask = 0;
		n.indegree = 0;
		n.criticality = 0;
		n.topoRank = 0;
		n.passIndex = i;
		resolveCompatibleQueueSlotsForPass(passAccess, n.compatibleQueueSlots);
		if (n.compatibleQueueSlots.empty() && passAccess.backendAffinity.strength == BackendAffinityStrength::Required) {
			throw std::runtime_error("RenderGraph required backend is unavailable for pass index " + std::to_string(i));
		}
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
		n.touchedIDs = &passAccess.touchedResourceIDs;
		n.uavIDs = &passAccess.uavResourceIDs;

	}
}

void RenderGraph::PlanActiveQueueSlots(
	RenderGraph& rg,
	const RenderGraph::FramePassList& passes,
	const std::vector<Node>& nodes)
{
	BT_ZONE_SCOPE("RenderGraph::PlanActiveQueueSlots");
	const size_t slotCount = rg.m_queueRegistry.SlotCount();
	auto& activeSlots = rg.m_activeQueueSlotsThisFrame;
	if (activeSlots.size() != slotCount) {
		activeSlots.assign(slotCount, 0);
	}
	else {
		std::fill(activeSlots.begin(), activeSlots.end(), 0);
	}
	if (slotCount == 0) {
		return;
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

	auto& indeg = rg.m_planActiveIndeg;
	if (indeg.size() != nodes.size()) {
		indeg.resize(nodes.size());
	}
	auto& level = rg.m_planActiveLevel;
	if (level.size() != nodes.size()) {
		level.assign(nodes.size(), 0);
	}
	else {
		std::fill(level.begin(), level.end(), 0);
	}
	auto& ready = rg.m_planActiveReady;
	ready.clear();
	if (ready.capacity() < nodes.size()) {
		ready.reserve(nodes.size());
	}
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

	constexpr size_t kMaxQueueSlots = 64;
	struct LocalSlotList {
		std::array<size_t, kMaxQueueSlots> values{};
		size_t count = 0;

		void Push(size_t value) {
			if (count < values.size()) {
				values[count++] = value;
			}
		}

		bool Empty() const { return count == 0; }
		size_t Front() const { return values[0]; }
		std::span<const size_t> Span() const { return std::span<const size_t>(values.data(), count); }
	};

	std::array<LocalSlotList, static_cast<size_t>(QueueKind::Count)> slotsByKind;
	std::array<LocalSlotList, static_cast<size_t>(QueueKind::Count)> autoAssignableSlotsByKind;
	for (size_t slotIndex = 0; slotIndex < slotCount; ++slotIndex) {
		const auto queueSlotIndex = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slotIndex));
		const QueueKind kind = rg.m_queueRegistry.GetKind(queueSlotIndex);
		slotsByKind[QueueIndex(kind)].Push(slotIndex);
		if (rg.m_queueRegistry.IsAutoAssignable(queueSlotIndex)) {
			autoAssignableSlotsByKind[QueueIndex(kind)].Push(slotIndex);
		}
	}

	const bool allowAsyncCompute = rg.m_getUseAsyncCompute ? rg.m_getUseAsyncCompute() : true;
	const double widthScale = rg.m_getQueueSchedulingWidthScale ? static_cast<double>(rg.m_getQueueSchedulingWidthScale()) : 1.0;
	const double penaltyBias = rg.m_getQueueSchedulingPenaltyBias ? static_cast<double>(rg.m_getQueueSchedulingPenaltyBias()) : 0.0;
	const double minPenalty = rg.m_getQueueSchedulingMinPenalty ? static_cast<double>(rg.m_getQueueSchedulingMinPenalty()) : 1.0;
	const double resourcePressureWeight = rg.m_getQueueSchedulingResourcePressureWeight ? static_cast<double>(rg.m_getQueueSchedulingResourcePressureWeight()) : 1.0;
	const double uavPressureWeight = rg.m_getQueueSchedulingUavPressureWeight ? static_cast<double>(rg.m_getQueueSchedulingUavPressureWeight()) : 0.5;

	if (rg.m_planActiveTouchedResourceEpochs.size() != rg.m_frameSchedulingResourceCount) {
		rg.m_planActiveTouchedResourceEpochs.assign(rg.m_frameSchedulingResourceCount, 0);
		rg.m_planActiveTouchedResourceEpoch = 1;
	}
	auto& widthByLevel = rg.m_planActiveWidthByLevel;
	if (widthByLevel.size() < nodes.size()) {
		widthByLevel.resize(nodes.size(), 0);
	}

	for (size_t kindIndex = 0; kindIndex < static_cast<size_t>(QueueKind::Count); ++kindIndex) {
		const QueueKind kind = static_cast<QueueKind>(kindIndex);
		auto& slots = slotsByKind[kindIndex];
		auto& autoAssignableSlots = autoAssignableSlotsByKind[kindIndex];
		if (slots.Empty()) {
			continue;
		}

		std::array<size_t, kMaxQueueSlots> pinnedSlots{};
		size_t pinnedSlotCount = 0;
		size_t maxLevelWidth = 0;
		size_t compatibleNodeCount = 0;
		size_t totalTouched = 0;
		size_t totalUAV = 0;
		size_t uniqueTouchedCount = 0;
		const uint32_t resourceEpoch = rg.m_planActiveTouchedResourceEpoch++;
		if (rg.m_planActiveTouchedResourceEpoch == 0) {
			std::fill(rg.m_planActiveTouchedResourceEpochs.begin(), rg.m_planActiveTouchedResourceEpochs.end(), 0);
			rg.m_planActiveTouchedResourceEpoch = 2;
		}

		auto pinSlot = [&](size_t pinnedSlot) {
			for (size_t index = 0; index < pinnedSlotCount; ++index) {
				if (pinnedSlots[index] == pinnedSlot) {
					return;
				}
			}
			if (pinnedSlotCount < pinnedSlots.size()) {
				pinnedSlots[pinnedSlotCount++] = pinnedSlot;
			}
		};

		for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
			const auto& node = nodes[nodeIndex];
			const bool compatibleWithKind = (node.compatibleQueueKindMask & static_cast<uint8_t>(1u << kindIndex)) != 0;
			if (!compatibleWithKind) {
				continue;
			}

			++compatibleNodeCount;
			const uint32_t nodeLevel = level[nodeIndex];
			if (nodeLevel < widthByLevel.size()) {
				const size_t width = ++widthByLevel[nodeLevel];
				maxLevelWidth = (std::max)(maxLevelWidth, width);
			}
			if (node.passIndex < rg.m_framePassSchedulingSummaries.size()) {
				const auto& passSummary = rg.m_framePassSchedulingSummaries[node.passIndex];
				totalTouched += passSummary.touchedResourceIndices.size();
				totalUAV += passSummary.uavResourceIndices.size();
				for (size_t resourceIndex : passSummary.touchedResourceIndices) {
					if (resourceIndex >= rg.m_planActiveTouchedResourceEpochs.size()) {
						continue;
					}
					if (rg.m_planActiveTouchedResourceEpochs[resourceIndex] == resourceEpoch) {
						continue;
					}
					rg.m_planActiveTouchedResourceEpochs[resourceIndex] = resourceEpoch;
					++uniqueTouchedCount;
				}
			}

			if (node.passIndex < passes.size() && passHasExplicitQueuePin(passes[node.passIndex])) {
				const size_t pinnedSlot = node.compatibleQueueSlots.empty() ? node.queueSlot : node.compatibleQueueSlots.front();
				if (pinnedSlot < slotCount
					&& rg.m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(pinnedSlot))) == kind) {
					pinSlot(pinnedSlot);
				}
			}
		}

		for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
			if ((nodes[nodeIndex].compatibleQueueKindMask & static_cast<uint8_t>(1u << kindIndex)) != 0) {
				const uint32_t nodeLevel = level[nodeIndex];
				if (nodeLevel < widthByLevel.size()) {
					widthByLevel[nodeLevel] = 0;
				}
			}
		}

		if (kind == QueueKind::Graphics) {
			activeSlots[slots.Front()] = 1;
			for (size_t index = 0; index < pinnedSlotCount; ++index) {
				const size_t pinnedSlot = pinnedSlots[index];
				activeSlots[pinnedSlot] = 1;
			}
			continue;
		}

		if (compatibleNodeCount == 0 && pinnedSlotCount == 0) {
			continue;
		}

		double resourcePressure = 1.0;
		if (uniqueTouchedCount > 0) {
			resourcePressure = static_cast<double>(totalTouched) / static_cast<double>(uniqueTouchedCount);
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
		targetCount = (std::min)(targetCount, autoAssignableSlots.count);
		targetCount = (std::max)(targetCount, pinnedSlotCount);

		for (size_t index = 0; index < pinnedSlotCount; ++index) {
			const size_t pinnedSlot = pinnedSlots[index];
			activeSlots[pinnedSlot] = 1;
		}
		if (compatibleNodeCount > 0) {
			if (!autoAssignableSlots.Empty()) {
				activeSlots[autoAssignableSlots.Front()] = 1;
			}
		}

		size_t activeCount = 0;
		for (size_t slot : autoAssignableSlots.Span()) {
			activeCount += activeSlots[slot] ? 1u : 0u;
		}

		for (size_t slot : autoAssignableSlots.Span()) {
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
	BT_ZONE_SCOPE("RenderGraph::BuildDependencyGraph");
	auto& seq = m_compilerState->dependencySeqStates;
	seq.resize(m_frameDAGResourceCount);
	for (auto& state : seq) {
		state.lastWriter.reset();
		state.readsSinceWrite.clear();
	}

	auto& edgeKeys = m_compilerState->dependencyEdgeKeys;
	edgeKeys.clear();
	edgeKeys.reserve(nodes.size() * 8);
	auto addEdgeCandidate = [&](size_t from, size_t to) {
		if (from == to) {
			return;
		}
		edgeKeys.push_back((uint64_t(from) << 32) | uint64_t(to));
	};

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
				if (s.lastWriter) addEdgeCandidate(*s.lastWriter, i);
				s.readsSinceWrite.push_back(i);
			}
			else { // Write
				if (s.lastWriter) addEdgeCandidate(*s.lastWriter, i);
				for (size_t r : s.readsSinceWrite)
					addEdgeCandidate(r, i);
				s.readsSinceWrite.clear();
				s.lastWriter = i;
			}
		}
	}

	// Imported memory has exclusive API ownership. Add an ordering edge even for
	// read/read uses when declaration order crosses an API boundary.
	std::vector<std::optional<size_t>> lastBackendAccess(m_frameDAGResourceCount);
	std::vector<rhi::Backend> lastBackend(m_frameDAGResourceCount, rhi::Backend::Null);
	const rhi::Backend primaryBackend = m_backendDevices.empty() ? rhi::Backend::Null : m_backendDevices.front().backend;
	for (size_t i = 0; i < nodes.size(); ++i) {
		const size_t passIndex = nodes[i].passIndex;
		if (passIndex >= m_framePassAccessSummaries.size()) continue;
		const auto& summary = m_framePassAccessSummaries[passIndex];
		const rhi::Backend backend = summary.backendAffinity.strength == BackendAffinityStrength::Primary
			? primaryBackend : summary.backendAffinity.backend;
		for (const auto& access : summary.dagAccesses) {
			if (access.resourceIndex >= lastBackendAccess.size()) continue;
			if (lastBackendAccess[access.resourceIndex] && lastBackend[access.resourceIndex] != backend) {
				addEdgeCandidate(*lastBackendAccess[access.resourceIndex], i);
			}
			lastBackendAccess[access.resourceIndex] = i;
			lastBackend[access.resourceIndex] = backend;
		}
	}

	// Apply explicit edges (e.g. "After(passName)")
	for (auto const& e : explicitEdges) {
		if (e.first >= nodes.size() || e.second >= nodes.size()) continue;
		addEdgeCandidate(e.first, e.second);
	}

	std::sort(edgeKeys.begin(), edgeKeys.end());
	edgeKeys.erase(std::unique(edgeKeys.begin(), edgeKeys.end()), edgeKeys.end());
	for (uint64_t key : edgeKeys) {
		const size_t from = static_cast<size_t>(key >> 32);
		const size_t to = static_cast<size_t>(key & 0xffffffffull);
		nodes[from].out.push_back(to);
		nodes[to].in.push_back(from);
		nodes[to].indegree++;
	}

	return FinalizeDependencyGraph(nodes);
}

bool RenderGraph::FinalizeDependencyGraph(std::vector<Node>& nodes)
{
	// topo + criticality (longest path)
	auto& indeg = m_compilerState->dependencyIndegrees;
	indeg.resize(nodes.size());
	for (size_t i = 0; i < nodes.size(); ++i) indeg[i] = nodes[i].indegree;

	auto originalOrderLess = [&](size_t lhs, size_t rhs) {
		if (nodes[lhs].originalOrder != nodes[rhs].originalOrder) {
			return nodes[lhs].originalOrder > nodes[rhs].originalOrder;
		}
		return lhs > rhs;
	};

	auto& ready = m_compilerState->dependencyReadyHeap;
	ready.clear();
	if (ready.capacity() < nodes.size()) {
		ready.reserve(nodes.size());
	}
	for (size_t i = 0; i < nodes.size(); ++i) {
		if (indeg[i] == 0) {
			ready.push_back(i);
			std::push_heap(ready.begin(), ready.end(), originalOrderLess);
		}
	}

	auto& topo = m_compilerState->dependencyTopoOrder;
	topo.clear();
	if (topo.capacity() < nodes.size()) {
		topo.reserve(nodes.size());
	}

	while (!ready.empty()) {
		std::pop_heap(ready.begin(), ready.end(), originalOrderLess);
		size_t u = ready.back();
		ready.pop_back();
		topo.push_back(u);
		for (size_t v : nodes[u].out) {
			if (--indeg[v] == 0) {
				ready.push_back(v);
				std::push_heap(ready.begin(), ready.end(), originalOrderLess);
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
	BT_ZONE_SCOPE("RenderGraph::AddCurrentFrameAliasSchedulingEdges");
	auto rangesOverlap = [](const org::alias::AliasPlacementRange& lhs, const org::alias::AliasPlacementRange& rhs) {
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

	auto& nodeIndexByPassIndex = m_aliasSchedulingNodeIndexByPassIndex;
	if (nodeIndexByPassIndex.size() != m_framePasses.size()) {
		nodeIndexByPassIndex.assign(m_framePasses.size(), SIZE_MAX);
	}
	else {
		std::fill(nodeIndexByPassIndex.begin(), nodeIndexByPassIndex.end(), SIZE_MAX);
	}
	for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
		const size_t passIndex = nodes[nodeIndex].passIndex;
		if (passIndex < nodeIndexByPassIndex.size()) {
			nodeIndexByPassIndex[passIndex] = nodeIndex;
		}
	}

	size_t existingEdgeCount = 0;
	for (const auto& node : nodes) {
		existingEdgeCount += node.out.size();
	}

	auto& resourceIDs = m_aliasSchedulingResourceIDs;
	resourceIDs.clear();
	if (resourceIDs.capacity() < aliasPlacementPoolByID.size()) {
		resourceIDs.reserve(aliasPlacementPoolByID.size());
	}
	for (const auto& [resourceID, resourceIndex] : m_frameSchedulingResourceIndexEntries) {
		if (TryGetAliasPlacementRangeByResourceIndex(resourceIndex)) {
			resourceIDs.push_back(resourceID);
		}
	}
	if (resourceIDs.empty()) {
		return true;
	}
	std::sort(resourceIDs.begin(), resourceIDs.end());

	auto& existingEdgeKeys = m_aliasSchedulingExistingEdgeKeys;
	existingEdgeKeys.clear();
	if (existingEdgeKeys.capacity() < existingEdgeCount) {
		existingEdgeKeys.reserve(existingEdgeCount);
	}
	for (size_t from = 0; from < nodes.size(); ++from) {
		for (size_t to : nodes[from].out) {
			existingEdgeKeys.push_back((uint64_t(from) << 32) | uint64_t(to));
		}
	}
	std::sort(existingEdgeKeys.begin(), existingEdgeKeys.end());
	existingEdgeKeys.erase(std::unique(existingEdgeKeys.begin(), existingEdgeKeys.end()), existingEdgeKeys.end());

	auto& proposedEdgeKeys = m_aliasSchedulingProposedEdgeKeys;
	proposedEdgeKeys.clear();
	const size_t maxAliasPairs = resourceIDs.size() > 1
		? (resourceIDs.size() * (resourceIDs.size() - 1)) / 2
		: 0;
	if (proposedEdgeKeys.capacity() < maxAliasPairs) {
		proposedEdgeKeys.reserve(maxAliasPairs);
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

			if (fromPassIndex >= nodeIndexByPassIndex.size() || toPassIndex >= nodeIndexByPassIndex.size()) {
				continue;
			}
			const size_t fromNodeIndex = nodeIndexByPassIndex[fromPassIndex];
			const size_t toNodeIndex = nodeIndexByPassIndex[toPassIndex];
			if (fromNodeIndex == SIZE_MAX || toNodeIndex == SIZE_MAX) {
				continue;
			}

			proposedEdgeKeys.push_back((uint64_t(fromNodeIndex) << 32) | uint64_t(toNodeIndex));
		}
	}

	if (!proposedEdgeKeys.empty()) {
		std::sort(proposedEdgeKeys.begin(), proposedEdgeKeys.end());
		proposedEdgeKeys.erase(std::unique(proposedEdgeKeys.begin(), proposedEdgeKeys.end()), proposedEdgeKeys.end());
		for (uint64_t key : proposedEdgeKeys) {
			if (std::binary_search(existingEdgeKeys.begin(), existingEdgeKeys.end(), key)) {
				continue;
			}
			const size_t from = static_cast<size_t>(key >> 32);
			const size_t to = static_cast<size_t>(key & 0xffffffffull);
			if (from >= nodes.size() || to >= nodes.size()) {
				continue;
			}
			nodes[from].out.push_back(to);
			nodes[to].in.push_back(from);
			nodes[to].indegree++;
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
	RenderGraph::BatchBuildState& batchBuildState,
	RenderGraph::FrameEpochSet& scratchTransitioned,
	RenderGraph::FrameEpochSet& scratchFallback,
	std::vector<ResourceTransition>& scratchTransitions)
{
	const size_t passQueueSlot = node.assignedQueueSlot.value_or(node.queueSlot);
	const size_t queueCount = currentBatch.QueueCount();
	if (passQueueSlot >= queueCount) {
		spdlog::error("RG invalid queue slot while committing pass index={} name='{}': slot={} queueCount={} preferred={} assigned={}",
			node.passIndex, pr.name, passQueueSlot, queueCount, node.queueSlot,
			node.assignedQueueSlot ? std::to_string(*node.assignedQueueSlot) : std::string("none"));
		throw std::runtime_error("RenderGraph assigned a pass to an unregistered queue slot");
	}
	if (node.passIndex >= rg.m_framePassSchedulingSummaries.size()) {
		spdlog::error("RG invalid scheduling summary index {} for pass '{}' (summaryCount={})",
			node.passIndex, pr.name, rg.m_framePassSchedulingSummaries.size());
		throw std::runtime_error("RenderGraph node references a missing pass scheduling summary");
	}
	const size_t gfxSlot = QueueIndex(QueueKind::Graphics);
	const auto& passSummary = rg.m_framePassSchedulingSummaries[node.passIndex];
	if (rg.m_getRenderGraphBatchTraceEnabled && rg.m_getRenderGraphBatchTraceEnabled()) {
		const size_t concreteRequirementCount = std::visit([](const auto& pass) -> size_t {
			using Pass = std::decay_t<decltype(pass)>;
			if constexpr (std::is_same_v<Pass, std::monostate>) return 0;
			else return GetFrameRequirementCount(pass.resources);
		}, pr.pass);
		spdlog::info("RG commit pass index={} name='{}' type={} variant={} queueSlot={}/{} summaryRequirements={} concreteRequirements={}",
			node.passIndex, pr.name, static_cast<int>(pr.type), pr.pass.index(), passQueueSlot, queueCount,
			passSummary.requirements.size(), concreteRequirementCount);
		if (passSummary.requirements.size() != concreteRequirementCount) {
			throw std::runtime_error("RenderGraph pass scheduling summary does not match concrete requirements");
		}
	}
	scratchTransitioned.Clear();
	scratchTransitioned.Reserve(passSummary.requirements.size());
	auto& resourcesTransitionedThisPass = scratchTransitioned;

	scratchFallback.Clear();
	scratchFallback.Reserve(passSummary.requirements.size());
	auto& fallbackResourceIndices = scratchFallback;
	scratchTransitions.reserve(1);
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
		if (fallbackResourceIndices.Empty()) {
			return;
		}

		for (size_t resourceIndex : fallbackResourceIndices.Values()) {
			rg.RecordFrameQueueTransitionBatch(gfxSlot, resourceIndex, currentBatchIndex);
		}

		for (size_t qi = 0; qi < queueCount; ++qi) {
			if (qi == gfxSlot) {
				continue;
			}

			int latestBatch = -1;
			for (size_t resourceIndex : fallbackResourceIndices.Values()) {
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

	auto& ignoredInternalTransitions = rg.m_compilerState->ignoredInternalTransitions;
	auto applyInternalTransitions = [&](const auto& pass) {
		const auto& denseTransitions = passSummary.internalTransitions;
		if (pass.resources.internalTransitions.size() != denseTransitions.size()) {
			throw std::runtime_error("Frame pass summary internal transition count mismatch");
		}

		for (size_t transitionIndex = 0; transitionIndex < denseTransitions.size(); ++transitionIndex) {
			const auto& exit = pass.resources.internalTransitions[transitionIndex];
			const auto& denseTransition = denseTransitions[transitionIndex];
			ignoredInternalTransitions.clear();
			auto* pRes = exit.first.resource.IsEphemeral() ? exit.first.resource.GetEphemeralPtr() : _registry.Resolve(exit.first.resource);
			auto& compileResourceState = GetOrCreateFrameCompileResourceState(
				denseTransition.resourceIndex,
				pRes,
				denseTransition.resourceID);
			pRes = compileResourceState.resource ? compileResourceState.resource : pRes;
			compileResourceState.tracker->Apply(exit.first.range, pRes, exit.second, ignoredInternalTransitions);
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
		for (size_t resourceIndex : passSummary.requiredResourceIndices) {
			if (!batchBuildState.ContainsResource(resourceIndex)) {
				currentBatch.allResources.push_back(rg.m_frameSchedulingResourceIDByIndex[resourceIndex]);
				batchBuildState.MarkResource(resourceIndex);
			}
			rg.RecordFrameQueueUsageBatch(passQueueSlot, resourceIndex, currentBatchIndex);
		}
		for (size_t resourceIndex : passSummary.writtenResourceIndices) {
			SetFrameQueueHistoryValue(m_frameQueueLastProducerBatch, passQueueSlot, resourceIndex, currentBatchIndex);
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
				for (const auto binding : pass.resources.externalWaitBindingsBeforeTransitions) {
					currentBatch.AddExternalWaitBindingBeforeTransitions(passQueueSlot, binding);
				}
				applyInternalTransitions(pass);
				recordRequirementHistory();
				const bool hasQueueCrossing = (queueCount > 1);
				if (hasQueueCrossing) {
					rg.GetBatchesToWaitOn(
						pass.name,
						0,
						passSummary,
						resourcesTransitionedThisPass);
				}

				for (size_t qi = 0; qi < queueCount; ++qi) {
					if (qi == passQueueSlot) {
						continue;
					}

					rg.ApplySynchronizationImpl(
						passQueueSlot,
						qi,
						currentBatch,
						currentBatchIndex,
						pass.name,
						qi < queueCount ? rg.m_waitCacheLatestTransitionByQueue[qi] : -1,
						qi < queueCount ? rg.m_waitCacheLatestProducerByQueue[qi] : -1,
						qi < queueCount ? rg.m_waitCacheLatestUsageByQueue[qi] : -1);
				}
			}
		},
		pr.pass);
}

void RenderGraph::AutoScheduleAndBuildBatches(
	RenderGraph& rg,
	RenderGraph::FramePassList& passes,
	std::vector<Node>& nodes)
{
	BT_ZONE_SCOPE("RenderGraph::AutoScheduleAndBuildBatches");
	uint64_t totalCandidateChecks = 0;
	uint64_t totalNewBatchChecks = 0;
	uint64_t maximumReadySetSize = 0;
	rg.m_compilerState->readOnlyUniformTransitionElisionEnabled =
		rg.m_getReadOnlyUniformTransitionElisionEnabled
		&& rg.m_getReadOnlyUniformTransitionElisionEnabled();
	// Working indegrees
	auto& indeg = rg.m_autoScheduleIndeg;
	if (indeg.size() != nodes.size()) {
		indeg.resize(nodes.size());
	}
	for (size_t i = 0; i < nodes.size(); ++i) indeg[i] = nodes[i].indegree;

	auto& ready = rg.m_autoScheduleReady;
	ready.clear();
	if (ready.capacity() < nodes.size()) {
		ready.reserve(nodes.size());
	}
	for (size_t i = 0; i < nodes.size(); ++i)
		if (indeg[i] == 0) ready.push_back(i);

	auto openNewBatch = [&]() -> PassBatch {
		const size_t queueCount = rg.m_queueRegistry.SlotCount();
		PassBatch b = rg.AcquireReusablePassBatch(queueCount);
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
	const auto selectionPolicy = rg.m_getQueueSchedulingSelectionPolicy
		? rg.m_getQueueSchedulingSelectionPolicy()
		: org::runtime::QueueSchedulingSelectionPolicy::FirstFit;
	const bool useScoredScheduling = selectionPolicy == org::runtime::QueueSchedulingSelectionPolicy::Scored;
	BatchBuildState& batchBuildState = rg.m_autoScheduleBatchBuildState;
	batchBuildState.Initialize(nodes.size(), queueCount, rg.m_frameSchedulingResourceCount);

	// Scratch sets reused across CommitPassToBatch calls to avoid per-call allocation
	FrameEpochSet& scratchTransitioned = rg.m_autoScheduleScratchTransitioned;
	scratchTransitioned.Initialize(rg.m_frameSchedulingResourceCount);
	FrameEpochSet& scratchFallback = rg.m_autoScheduleScratchFallback;
	scratchFallback.Initialize(rg.m_frameSchedulingResourceCount);
	std::vector<ResourceTransition>& scratchTransitions = rg.m_autoScheduleScratchTransitions;
	scratchTransitions.clear();
	scratchTransitions.reserve(16);
	rg.m_schedulingDecisionTrace.reserve(nodes.size());
	const double autoGraphicsBias = rg.m_getQueueSchedulingAutoGraphicsBias ? static_cast<double>(rg.m_getQueueSchedulingAutoGraphicsBias()) : 2.5;
	const double asyncOverlapBonus = rg.m_getQueueSchedulingAsyncOverlapBonus ? static_cast<double>(rg.m_getQueueSchedulingAsyncOverlapBonus()) : 3.0;
	const double crossQueueHandoffPenalty = rg.m_getQueueSchedulingCrossQueueHandoffPenalty ? static_cast<double>(rg.m_getQueueSchedulingCrossQueueHandoffPenalty()) : 2.0;

	auto closeBatch = [&]() {
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

	// Cross-API ownership barriers are pass-local, while ordinary transitions
	// are aggregated per batch.  Keep every user of a resource that may cross
	// APIs in its own batch so an acquire/release cannot consume or suppress a
	// neighbouring pass's per-subresource transition.
	std::unordered_set<uint64_t> crossBackendResourceIDs;
	if (rg.m_backendDevices.size() > 1) {
		for (const auto& summary : rg.m_framePassAccessSummaries) {
			if (summary.backendAffinity.strength == BackendAffinityStrength::Primary) continue;
			crossBackendResourceIDs.insert(summary.touchedResourceIDs.begin(), summary.touchedResourceIDs.end());
		}
	}

	auto passForcesBatchIsolation = [&](size_t passIndex) {
		if (passIndex >= passes.size() || passHasImmediateWork(passes[passIndex])) return true;
		if (passIndex < rg.m_framePassAccessSummaries.size() && !crossBackendResourceIDs.empty()) {
			const auto& touched = rg.m_framePassAccessSummaries[passIndex].touchedResourceIDs;
			if (std::ranges::any_of(touched, [&](uint64_t id) { return crossBackendResourceIDs.contains(id); })) {
				return true;
			}
		}
		return std::visit([](const auto& entry) {
			using T = std::decay_t<decltype(entry)>;
			if constexpr (std::is_same_v<T, std::monostate>) return false;
			else return !entry.resources.externalWaitBindingsBeforeTransitions.empty();
		}, passes[passIndex].pass);
	};

	auto updateBatchMembershipForCommittedPass = [&](const Node& committedNode) {
		const auto& passSummary = rg.m_framePassSchedulingSummaries[committedNode.passIndex];
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

	auto currentBatchHasPasses = [&]() {
		for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
			if (!currentBatch.Passes(queueIndex).empty()) {
				return true;
			}
		}
		return false;
	};

	auto queueSlotIsActiveAndCompatible = [&](const Node& node, size_t queueSlot) {
		if (queueSlot >= queueCount) {
			return false;
		}
		if (queueSlot >= rg.m_activeQueueSlotsThisFrame.size() || !rg.m_activeQueueSlotsThisFrame[queueSlot]) {
			return false;
		}
		for (size_t compatibleSlot : node.compatibleQueueSlots) {
			if (compatibleSlot == queueSlot) {
				return true;
			}
		}
		return false;
	};

	size_t remaining = nodes.size();
	bool closedBatchBeforeNextCommit = false;

	while (remaining > 0) {
		uint32_t candidateChecks = 0;
		uint32_t isNewBatchNeededChecks = 0;
		const uint32_t readySetSizeBeforeEvaluate = static_cast<uint32_t>(ready.size());
		size_t bestIdxInReady = SIZE_MAX;
		size_t bestQueueSlot = 0;
		{

			auto candidateFits = [&](size_t readyIndex, size_t nodeQueueSlot) {
				const size_t ni = ready[readyIndex];
				auto& n = nodes[ni];
				++candidateChecks;
				if (nodeQueueSlot >= queueCount) {
					return false;
				}
				if (nodeQueueSlot >= rg.m_activeQueueSlotsThisFrame.size() || !rg.m_activeQueueSlotsThisFrame[nodeQueueSlot]) {
					return false;
				}
				if (!queueSlotIsActiveAndCompatible(n, nodeQueueSlot)) {
					return false;
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
					return false;
				}

				++isNewBatchNeededChecks;
				const auto& passSummary = rg.m_framePassSchedulingSummaries[n.passIndex];
				if (rg.IsNewBatchNeeded(
					passSummary,
					currentBatch,
					batchBuildState,
					passes[n.passIndex].name,
					currentBatchIndex,
					nodeQueueSlot))
				{
					return false;
				}

				return true;
			};

			if (useScoredScheduling) {
				double bestScore = -1e300;
				std::array<uint8_t, 64> batchHasQueue{};
				for (size_t queueIndex = 0; queueIndex < queueCount && queueIndex < batchHasQueue.size(); ++queueIndex) {
					batchHasQueue[queueIndex] = currentBatch.HasPasses(queueIndex);
				}

				size_t readyGraphicsCapableCount = 0;
				for (size_t nodeIndex : ready) {
					if ((nodes[nodeIndex].compatibleQueueKindMask & static_cast<uint8_t>(1u << QueueIndex(QueueKind::Graphics))) != 0) {
						++readyGraphicsCapableCount;
					}
				}

				const bool batchHasGraphicsWork =
					gfxSlot < queueCount
					&& gfxSlot < batchHasQueue.size()
					&& batchHasQueue[gfxSlot] != 0;

				for (size_t readyIndex = 0; readyIndex < ready.size(); ++readyIndex) {
					const size_t nodeIndex = ready[readyIndex];
					Node& node = nodes[nodeIndex];
					const auto& passSummary = rg.m_framePassSchedulingSummaries[node.passIndex];

					for (size_t nodeQueueSlot : node.compatibleQueueSlots) {
						if (!candidateFits(readyIndex, nodeQueueSlot)) {
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

						double score = 3.0 * static_cast<double>(reuse) - static_cast<double>(fresh);
						if (nodeQueueSlot < batchHasQueue.size() && !batchHasQueue[nodeQueueSlot]) {
							score += 2.0;
						}
						score -= 0.25 * static_cast<double>(currentBatch.Passes(nodeQueueSlot).size());
						score += 0.05 * static_cast<double>(node.criticality);

						if (passes[node.passIndex].type == PassType::Compute
							&& node.queueAssignmentPolicy == QueueAssignmentPolicy::Automatic) {
							const QueueKind candidateKind = rg.m_queueRegistry.GetKind(
								static_cast<QueueSlotIndex>(static_cast<uint8_t>(nodeQueueSlot)));
							const uint8_t candidateKindMask = static_cast<uint8_t>(1u << QueueIndex(candidateKind));

							size_t predecessorCrossQueueCount = 0;
							for (size_t pred : node.in) {
								const size_t predSlot = nodes[pred].assignedQueueSlot.value_or(nodes[pred].queueSlot);
								const QueueKind predKind = rg.m_queueRegistry.GetKind(
									static_cast<QueueSlotIndex>(static_cast<uint8_t>(predSlot)));
								if (predKind != candidateKind) {
									++predecessorCrossQueueCount;
								}
							}

							size_t successorCrossQueueCount = 0;
							for (size_t succ : node.out) {
								if ((nodes[succ].compatibleQueueKindMask & candidateKindMask) == 0) {
									++successorCrossQueueCount;
								}
							}

							score -= crossQueueHandoffPenalty * static_cast<double>(predecessorCrossQueueCount + successorCrossQueueCount);

							if (candidateKind == QueueKind::Graphics) {
								score += autoGraphicsBias;
							}
							else if (candidateKind == QueueKind::Compute) {
								const bool candidateCanAlsoRunOnGraphics =
									(node.compatibleQueueKindMask & static_cast<uint8_t>(1u << QueueIndex(QueueKind::Graphics))) != 0;
								const size_t otherReadyGraphicsCandidates = readyGraphicsCapableCount > 0
									? readyGraphicsCapableCount - (candidateCanAlsoRunOnGraphics ? 1u : 0u)
									: 0u;
								score += (batchHasGraphicsWork || otherReadyGraphicsCandidates > 0)
									? asyncOverlapBonus
									: -asyncOverlapBonus;
							}
						}

						score += 1e-6 * static_cast<double>(nodes.size() - node.originalOrder);

						if (score > bestScore) {
							bestScore = score;
							bestIdxInReady = readyIndex;
							bestQueueSlot = nodeQueueSlot;
						}
					}
				}
			}
			else {
				auto trySelect = [&](size_t readyIndex, size_t queueSlot) {
					if (!candidateFits(readyIndex, queueSlot)) {
						return false;
					}
					bestIdxInReady = readyIndex;
					bestQueueSlot = queueSlot;
					return true;
				};

				for (size_t readyIndex = 0; readyIndex < ready.size() && bestIdxInReady == SIZE_MAX; ++readyIndex) {
					Node& node = nodes[ready[readyIndex]];

					for (size_t slot : node.compatibleQueueSlots) {
						if (slot < queueCount && currentBatch.HasPasses(slot) && trySelect(readyIndex, slot)) {
							break;
						}
					}
					if (bestIdxInReady != SIZE_MAX) {
						break;
					}
					if (trySelect(readyIndex, node.queueSlot)) {
						break;
					}
					for (size_t slot : node.compatibleQueueSlots) {
						if (slot != node.queueSlot && trySelect(readyIndex, slot)) {
							break;
						}
					}
				}
			}
		}
		totalCandidateChecks += candidateChecks;
		totalNewBatchChecks += isNewBatchNeededChecks;
		maximumReadySetSize = std::max(maximumReadySetSize, static_cast<uint64_t>(readySetSizeBeforeEvaluate));

		auto commitReadyIndex = [&](size_t readyIndex, size_t queueSlot, bool fallbackCommit) {
			const size_t nodeIndex = ready[readyIndex];
			auto& node = nodes[nodeIndex];
			{
				node.assignedQueueSlot = queueSlot;
				if (node.passIndex < rg.m_assignedQueueSlotsByFramePass.size()) {
					rg.m_assignedQueueSlotsByFramePass[node.passIndex] = queueSlot;
				}
				const bool isolateBatch = passForcesBatchIsolation(node.passIndex);
				if (isolateBatch) {
					closeBatch();
					closedBatchBeforeNextCommit = true;
				}
				rg.m_schedulingDecisionTrace.push_back(SchedulingDecisionTrace{
					.nodeIndex = static_cast<uint32_t>(nodeIndex),
					.passIndex = static_cast<uint32_t>(node.passIndex),
					.batchIndex = currentBatchIndex,
					.assignedQueueSlot = static_cast<uint16_t>(queueSlot),
					.closedBatchBefore = closedBatchBeforeNextCommit,
					.readySetSize = readySetSizeBeforeEvaluate,
					.candidateChecks = candidateChecks,
					.isNewBatchNeededChecks = isNewBatchNeededChecks,
					.fallbackCommit = fallbackCommit,
				});
				closedBatchBeforeNextCommit = false;
				CommitPassToBatch(
					rg, passes[node.passIndex], node,
					currentBatchIndex, currentBatch,
					batchBuildState,
					scratchTransitioned,
					scratchFallback,
					scratchTransitions);
				updateBatchMembershipForCommittedPass(node);
				if (isolateBatch) {
					closeBatch();
					closedBatchBeforeNextCommit = true;
				}
			}

			batchBuildState.MarkNode(nodeIndex);

			ready[readyIndex] = ready.back();
			ready.pop_back();

			for (size_t v : node.out) {
				if (--indeg[v] == 0) ready.push_back(v);
			}

			--remaining;

			if (rg.m_getHeavyDebug && rg.m_getHeavyDebug()) {
				closeBatch();
			}
		};

		if (bestIdxInReady == SIZE_MAX) {
			// Nothing ready fits: must end batch
			if (currentBatchHasPasses()) {
				closeBatch();
				closedBatchBeforeNextCommit = true;
				continue;
			}
			else {
				BT_ZONE_SCOPE("RenderGraph::CompileFrame::AutoScheduleAndBuildBatches::CommitFallbackPass");
				// Should be rare; fall back by forcing one ready pass in.
				// If this happens, IsNewBatchNeeded is likely too strict on empty batch.
				size_t fallbackReadyIndex = 0;
				size_t ni = ready[fallbackReadyIndex];
				auto& n = nodes[ni];
				size_t fallbackSlot = n.queueSlot;
				for (size_t compatibleSlot : n.compatibleQueueSlots) {
					if (compatibleSlot < rg.m_activeQueueSlotsThisFrame.size() && rg.m_activeQueueSlotsThisFrame[compatibleSlot]) {
						fallbackSlot = compatibleSlot;
						break;
					}
				}
				commitReadyIndex(fallbackReadyIndex, fallbackSlot, true);
				continue;
			}
		}

		commitReadyIndex(bestIdxInReady, bestQueueSlot, false);
	}
	BT_PLOT("ORG.AutoSchedule.CandidateChecks", static_cast<int64_t>(totalCandidateChecks));
	BT_PLOT("ORG.AutoSchedule.NewBatchChecks", static_cast<int64_t>(totalNewBatchChecks));
	BT_PLOT("ORG.AutoSchedule.MaximumReadySetSize", static_cast<int64_t>(maximumReadySetSize));

	// Final batch
	bool hasAnyQueuedPasses = false;
	for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
		hasAnyQueuedPasses = hasAnyQueuedPasses || !currentBatch.Passes(queueIndex).empty();
	}
	if (hasAnyQueuedPasses) {
		rg.batches.push_back(std::move(currentBatch));
	}

	// The dependency DAG is authoritative for execution ordering, including
	// explicit After/Before constraints that do not mention a resource.  The
	// resource synchronization path in CommitPassToBatch cannot see those
	// resource-less edges, so materialize every cross-queue DAG edge as a queue
	// signal/wait here, once final batch and queue assignments are known.
	// Without this, submission may legally coalesce work from one queue across
	// intervening batches and execute A,C,B for an explicit A->B->C chain.
	{
		std::vector<const SchedulingDecisionTrace*> placementByNode(nodes.size(), nullptr);
		for (const auto& decision : rg.m_schedulingDecisionTrace) {
			if (decision.nodeIndex < placementByNode.size()) {
				placementByNode[decision.nodeIndex] = &decision;
			}
		}

		for (size_t sourceNode = 0; sourceNode < nodes.size(); ++sourceNode) {
			const auto* source = placementByNode[sourceNode];
			if (!source || source->batchIndex >= rg.batches.size()) {
				continue;
			}
			for (size_t destinationNode : nodes[sourceNode].out) {
				if (destinationNode >= placementByNode.size()) {
					continue;
				}
				const auto* destination = placementByNode[destinationNode];
				if (!destination || destination->batchIndex >= rg.batches.size()
					|| source->assignedQueueSlot == destination->assignedQueueSlot) {
					continue;
				}

				auto& sourceBatch = rg.batches[source->batchIndex];
				auto& destinationBatch = rg.batches[destination->batchIndex];
				sourceBatch.MarkQueueSignal(BatchSignalPhase::AfterCompletion, source->assignedQueueSlot);
				destinationBatch.AddQueueWait(
					BatchWaitPhase::BeforeExecution,
					destination->assignedQueueSlot,
					source->assignedQueueSlot,
					sourceBatch.GetQueueSignalFenceValue(
						BatchSignalPhase::AfterCompletion,
						source->assignedQueueSlot));
			}
		}
	}

	{
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::AutoScheduleAndBuildBatches::CoalesceQueueWaits");
		rg.CoalesceQueueWaitsAndSignals(rg.batches);
	}

	// Build cross-frame producer and access tracking from the committed batch
	// schedule. Producers protect next-frame reads. All accesses are also
	// retained because a next-frame write must wait for prior readers.
	{
		BT_ZONE_SCOPE("RenderGraph::CompileFrame::AutoScheduleAndBuildBatches::BuildCrossFrameProducerTracking");
		auto& producersByQueue =
			rg.m_compilerState->compiledLastProducerBatchByResourceByQueue;
		auto& accessesByQueue =
			rg.m_compilerState->compiledLastAccessBatchByResourceByQueue;
		producersByQueue.resize(queueCount);
		accessesByQueue.resize(queueCount);
		for (auto& producers : producersByQueue) {
			producers.clear();
		}
		for (auto& accesses : accessesByQueue) {
			accesses.clear();
		}
		const size_t denseEntryCount = queueCount * rg.m_frameSchedulingResourceCount;
		auto& denseProducers =
			rg.m_compilerState->denseCompiledProducerBatchByQueueResource;
		auto& denseAccesses =
			rg.m_compilerState->denseCompiledAccessBatchByQueueResource;
		denseProducers.assign(denseEntryCount, {});
		denseAccesses.assign(denseEntryCount, {});
		for (const auto& decision : rg.m_schedulingDecisionTrace) {
			const size_t queueSlot = decision.assignedQueueSlot;
			if (queueSlot >= queueCount
				|| decision.passIndex >= rg.m_framePassSchedulingSummaries.size()) {
				continue;
			}
			const auto& passSummary = rg.m_framePassSchedulingSummaries[decision.passIndex];
			for (const auto& req : passSummary.requirements) {
				if (req.resourceIndex >= rg.m_frameSchedulingResourceCount) {
					continue;
				}
				const size_t denseIndex =
					queueSlot * rg.m_frameSchedulingResourceCount + req.resourceIndex;
				denseAccesses[denseIndex] = {
					.resourceID = req.resource.GetGlobalResourceID(),
					.batchIndex = decision.batchIndex,
					.anonymous = rg._registry.IsAnonymous(req.resource),
				};
				if (AccessTypeIsWriteType(req.state.access)) {
					denseProducers[denseIndex] = denseAccesses[denseIndex];
				}
			}
		}
		for (size_t queueSlot = 0; queueSlot < queueCount; ++queueSlot) {
			auto& producers = producersByQueue[queueSlot];
			auto& accesses = accessesByQueue[queueSlot];
			for (size_t resourceIndex = 0;
				resourceIndex < rg.m_frameSchedulingResourceCount;
				++resourceIndex) {
				const size_t denseIndex =
					queueSlot * rg.m_frameSchedulingResourceCount + resourceIndex;
				if (denseAccesses[denseIndex].batchIndex != 0) {
					accesses.push_back(denseAccesses[denseIndex]);
				}
				if (denseProducers[denseIndex].batchIndex != 0) {
					producers.push_back(denseProducers[denseIndex]);
				}
			}
		}
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
	BT_ZONE_SCOPE("RenderGraph::AssignQueueSignalFenceValuesInSubmissionOrder");
	const size_t slotCount = m_queueRegistry.SlotCount();
	std::vector<std::unordered_map<UINT64, UINT64>> remappedFenceValuesByQueue(slotCount);

	for (auto& batch : batchesToAssign) {
		const size_t queueCount = std::min(batch.QueueCount(), slotCount);
		for (size_t qi = 0; qi < queueCount; ++qi) {
			for (size_t phaseIndex = 0; phaseIndex < PassBatch::kSignalPhaseCount; ++phaseIndex) {
				const auto phase = static_cast<BatchSignalPhase>(phaseIndex);
				// Every active batch signals its reserved completion fence for
				// command-list recycling, even if it is not a graph dependency.
				// Reassign that value too, otherwise a newly assigned dependency
				// signal can jump ahead of a stale completion value.
				const bool activeCompletion =
					phase == BatchSignalPhase::AfterCompletion
					&& (batch.HasTransitions(qi, BatchTransitionPhase::BeforePasses)
						|| batch.HasPasses(qi)
						|| batch.HasTransitions(qi, BatchTransitionPhase::AfterPasses));
				if (!batch.HasQueueSignal(phase, qi) && !activeCompletion) {
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
	FrameEpochSet& outTransitionedResourceIndices,
	FrameEpochSet& outFallbackResourceIndices,
	std::vector<ResourceTransition>& scratchTransitions)
{
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
		outTransitionedResourceIndices,
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
	if (m_aliasActivationPendingByResourceIndex[requirement.resourceIndex] != org::alias::AliasActivationReason::None) {
		return false;
	}

	auto& entry = m_frameCompileResources[requirement.resourceIndex];
	if (entry.fastState.valid
		&& entry.fastState.wholeResourceOnly
		&& requirement.isWholeResource
		&& StatesExactlyEqual(entry.fastState.state, requiredState)) {
		return true;
	}

	if (!entry.trackerInitialized || !entry.tracker.has_value()) {
		return false;
	}

	const bool wouldModify = requirement.isWholeResource
		? entry.tracker->WouldModifyWholeResourceFast(requiredState)
		: entry.tracker->WouldModify(requirement.range, requiredState);
	if (wouldModify) {
		return false;
	}

	currentBatch.SetPassBatchTracker(
		requirement.resourceIndex,
		m_frameSchedulingResourceCount,
		&*entry.tracker);
	return true;
}

bool RenderGraph::TryAddTransitionTrackedNoOp(
	PassBatch& currentBatch,
	const DenseRequirementSummary& requirement,
	ResourceState requiredState)
{
	if (requirement.resourceIndex >= m_frameCompileResources.size()) {
		return false;
	}
	if (requirement.resourceIndex >= m_aliasActivationPendingByResourceIndex.size()) {
		return false;
	}
	if (m_aliasActivationPendingByResourceIndex[requirement.resourceIndex] != org::alias::AliasActivationReason::None) {
		return false;
	}

	auto& entry = m_frameCompileResources[requirement.resourceIndex];
	if (!entry.trackerInitialized || !entry.tracker.has_value()) {
		return false;
	}

	const bool wouldModify = requirement.isWholeResource
		? entry.tracker->WouldModifyWholeResourceFast(requiredState)
		: entry.tracker->WouldModify(requirement.range, requiredState);
	if (wouldModify) {
		return false;
	}

	currentBatch.SetPassBatchTracker(
		requirement.resourceIndex,
		m_frameSchedulingResourceCount,
		&*entry.tracker);
	return true;
}

void RenderGraph::AddTransitionSlowPath(
	unsigned int batchIndex,
	PassBatch& currentBatch,
	size_t passQueueSlot,
	std::string_view passName,
	const DenseRequirementSummary& requirement,
	ResourceState requiredState,
	RenderGraph::FrameEpochSet& outTransitionedResourceIndices,
	RenderGraph::FrameEpochSet& outFallbackResourceIndices,
	std::vector<ResourceTransition>& scratchTransitions)
{
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
	auto& compileTracker = *compileResourceState.tracker;
	const bool isWholeResourceRequirement = requirement.isWholeResource;

	bool isAliasActivation = false;
	if (requirement.resourceIndex < m_aliasActivationPendingByResourceIndex.size() && m_aliasActivationPendingByResourceIndex[requirement.resourceIndex] != org::alias::AliasActivationReason::None) {
		isAliasActivation = true;
		const auto activationReason = m_aliasActivationPendingByResourceIndex[requirement.resourceIndex];
		const bool firstUseIsWrite = AccessTypeIsWriteType(requirement.state.access);
		const bool firstUseIsCommon = requirement.state.access == rhi::ResourceAccessType::Common;
		// Common counts as write for alias activation, as this is generally used to indicate that the resource will be
		// transitioned internally by an external system that still uses legacy barriers. Don't abuse this.
		if (firstUseIsWrite || firstUseIsCommon) { 
			const bool isTexture = pRes && pRes->HasLayout();
			const RangeSpec wholeResourceRange{};
			const ResourceState activationBeforeState{
				rhi::ResourceAccessType::None,
				isTexture ? rhi::ResourceLayout::Undefined : rhi::ResourceLayout::Common,
				rhi::ResourceSyncState::None };
			transitions.emplace_back(
				pRes,
				wholeResourceRange,
				activationBeforeState.access,
				requiredState.access,
				activationBeforeState.layout,
				requiredState.layout,
				activationBeforeState.sync,
				requiredState.sync,
				isTexture);
			if (m_getRenderGraphCompileDumpEnabled && m_getRenderGraphCompileDumpEnabled()) {
				spdlog::debug(
					"RG alias activation: resource='{}' reason={} wholeResource=1 discard={} beforeLayout={} afterLayout={}",
					pRes ? pRes->GetName() : std::string("<null>"),
					static_cast<uint32_t>(activationReason),
					isTexture ? 1 : 0,
					static_cast<uint32_t>(activationBeforeState.layout),
					static_cast<uint32_t>(requiredState.layout));
			}
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
		compileTracker.Apply(RangeSpec{}, pRes, requiredState, ignored);
		aliasActivationPending.erase(requirement.resourceID);
		m_aliasActivationPendingByResourceIndex[requirement.resourceIndex] = org::alias::AliasActivationReason::None;
	}
	else {
		if (isWholeResourceRequirement && fastState.wholeResourceOnly) {
			if (!compileTracker.ApplyWholeResourceFast(requirement.range, pRes, requiredState, transitions)) {
				compileTracker.Apply(requirement.range, pRes, requiredState, transitions);
			}
		}
		else {
			compileTracker.Apply(requirement.range, pRes, requiredState, transitions);
		}
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
		outTransitionedResourceIndices.Insert(requirement.resourceIndex);
	}

	currentBatch.SetPassBatchTracker(
		requirement.resourceIndex,
		m_frameSchedulingResourceCount,
		&compileTracker); // We will need to check subsequent passes against this

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
	const auto transitionPlacementMode = m_getTransitionPlacementMode
		? m_getTransitionPlacementMode()
		: org::runtime::TransitionPlacementMode::InlineEarlyPlacement;
	if (transitionPlacementMode == org::runtime::TransitionPlacementMode::CanonicalThenOptimize) {
		if (passQueue != QueueKind::Graphics && needsGraphicsQueueForTransitions) {
			for (auto& transition : transitions) {
				currentBatch.Transitions(gfxSlot, BatchTransitionPhase::BeforePasses).push_back(transition);
			}
			if (debugStats) {
				debugStats->beforePassTransitionCount += transitions.size();
				debugStats->graphicsFallbackTransitionCount += transitions.size();
			}
			outFallbackResourceIndices.Insert(requirement.resourceIndex);
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
		outFallbackResourceIndices.Insert(requirement.resourceIndex);
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
	PassBatch& currentBatch,
	FrameEpochSet& outTransitionedResourceIndices,
	FrameEpochSet& outFallbackResourceIndices,
	std::vector<ResourceTransition>& scratchTransitions) {
	const bool enableReadOnlyUniformTransitionElision =
		m_compilerState->readOnlyUniformTransitionElisionEnabled;

	for (const auto& resourceRequirement : resourceRequirements) {
		const bool isReadOnlyUniform =
			enableReadOnlyUniformTransitionElision
			&&
			resourceRequirement.resourceIndex < m_frameResourceAccessSummaries.size()
			&& m_frameResourceAccessSummaries[resourceRequirement.resourceIndex].readOnlyUniform;

		if (isReadOnlyUniform) {
			FrameCompileResourceState* compileResourceState =
				resourceRequirement.resourceIndex < m_frameCompileResources.size()
				? &m_frameCompileResources[resourceRequirement.resourceIndex]
				: nullptr;
			if (!compileResourceState || !compileResourceState->readOnlyUniformTransitionChecked) {
				AddTransition(batchIndex, currentBatch, passQueueSlot, passName, resourceRequirement, outTransitionedResourceIndices, outFallbackResourceIndices, scratchTransitions);
				if (compileResourceState) {
					compileResourceState->readOnlyUniformTransitionChecked = true;
				}
			}
			else {
				currentBatch.SetPassBatchTracker(
					resourceRequirement.resourceIndex,
					m_frameSchedulingResourceCount,
					compileResourceState->trackerInitialized && compileResourceState->tracker.has_value()
						? &*compileResourceState->tracker
						: nullptr);
			}
		}
		else {
			AddTransition(batchIndex, currentBatch, passQueueSlot, passName, resourceRequirement, outTransitionedResourceIndices, outFallbackResourceIndices, scratchTransitions);
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

RenderGraph::RenderGraph(rhi::Device device, rhi::Backend primaryBackend)
	: m_compilerState(std::make_unique<CompilerState>()) {
	DeviceManager::GetInstance().Initialize(device);
	m_backendDevices.RegisterPrimary(primaryBackend, device);

	auto MakeDefaultImmediateDispatch = [&]() noexcept -> org::imm::ImmediateDispatch
		{
			org::imm::ImmediateDispatch d{};
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
		m_statisticsService = org::runtime::CreateDefaultStatisticsService();
	}
	if (!m_uploadService) {
		m_uploadService = org::runtime::CreateDefaultUploadService();
	}
	if (!m_readbackService) {
		m_readbackService = org::runtime::CreateDefaultReadbackService();
	}
	if (!m_descriptorService) {
		m_descriptorService = org::runtime::CreateDefaultDescriptorService();
	}
	if (!m_renderGraphSettingsService) {
		m_renderGraphSettingsService = org::runtime::CreateDefaultRenderGraphSettingsService();
	}
}

DeviceInstanceId RenderGraph::RegisterBackendDevice(rhi::Backend backend, rhi::Device device) {
	if (!device || backend == rhi::Backend::Null) {
		throw std::invalid_argument("RegisterBackendDevice requires a valid backend device");
	}
	const auto id = m_backendDevices.Register(backend, device);
	DescriptorHeapManager::GetInstance().RegisterBackend(id, device);
	return id;
}

RenderGraph::~RenderGraph() {
	if (m_pCommandRecordingManager) {
		m_pCommandRecordingManager->ShutdownThreadLocal(); // Clears thread-local storage
	}
	ShutdownOwnedState();
}

void RenderGraph::ShutdownTaskWorkers() {
	m_queueRegistry.ShutdownTaskWorkers();
	m_taskService.reset();
	org::runtime::SetDefaultTaskService({});
}

void RenderGraph::ShutdownRuntime() {
	StatisticsManager::GetInstance().ClearAll();
	DeletionManager::GetInstance().DrainAll();
	DeletionManager::GetInstance().Cleanup();
	DeviceManager::GetInstance().Cleanup();
}

void RenderGraph::ShutdownOwnedState() {
	ShutdownTaskWorkers();
	batches.clear();
	m_reusablePassBatches.clear();
	initialTransitions.clear();
	trackers.clear();
	m_frameCompileResources.clear();
	m_addTransitionDebugStatsByResource.clear();
	m_masterPassList.clear();
	m_compilerState->immediateModePassPointers.clear();
	m_compilerState->immediateModeInterfaces.clear();
	m_compilerState->densePassAccessKeys.clear();
	m_compilerState->frameExtensions.clear();
	m_compilerState->frameExtensionPassNames.clear();
	m_compilerState->frameExplicitAfterByName.clear();
	m_compilerState->pendingFrameInserts.clear();
	m_compilerState->frameInsertSlotHeads.clear();
	m_compilerState->frameInsertSlotTails.clear();
	m_compilerState->pendingInsertIndexByName.clear();
	m_compilerState->pendingInsertTailByAnchorName.clear();
	m_retainedDeclarationRefreshCandidateMasterIndices.clear();
	m_framePasses.clear();
	m_framePassIsFrameExtension.clear();
	m_framePassAccessSummaryCache.clear();
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
	compiledResourceGenerations.clear();
	aliasMaterializeOptionsByID.clear();
	m_aliasMaterializeOptionsByResourceIndex.clear();
	m_aliasMaterializeResourceIDs.clear();
	aliasPlacementSignatureByID.clear();
	aliasPlacementRangesByID.clear();
	schedulingPlacementRangesByID.clear();
	m_schedulingEquivalentIDsCache.clear();
	m_schedulingEquivalentIDFlat.clear();
	m_schedulingEquivalentIDRangeByResourceIndex.clear();
	m_aliasStaticInfoCacheByResourceID.clear();
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
	m_lastAccessByResourceAcrossFrames.clear();
	m_lastAliasPlacementProducersByPoolAcrossFrames.clear();
	for (auto& producerMap : m_compilerState->compiledLastProducerBatchByResourceByQueue) {
		producerMap.clear();
	}
	for (auto& accessMap : m_compilerState->compiledLastAccessBatchByResourceByQueue) {
		accessMap.clear();
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
	// Queue shutdown is the final completion point. Release placed resources
	// before their imported/canonical heap pairs, and before queue/device state.
	m_retiredInteropGenerations.clear();
	m_sharedAliasPools.clear();
	m_sharedAliasResourcePoolGeneration.clear();
	m_sharedAliasResourcePoolID.clear();

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

void RenderGraph::SetCompileProfileEnabled(bool enabled) noexcept {
	m_compileProfileEnabled = enabled;
	if (!enabled) {
		m_compileProfileFrame.reset();
	}
}

void RenderGraph::BeginCompileProfileFrame(uint8_t frameIndex) {
	if (!m_compileProfileEnabled || !basic_telemetry::Enabled()) {
		m_compileProfileFrame.reset();
		return;
	}

	m_compileProfileFrame.emplace("OpenRenderGraph.CompileFrame");
	m_compileProfileFrame->SetDimension("frame_index", frameIndex);
	m_compileProfileFrameStartedAtNs = basic_telemetry::NowNs();
	m_lastMaterializeCandidateCount = 0;
}

void RenderGraph::EndCompileProfileFrame() {
	if (!m_compileProfileFrame) {
		return;
	}

	BT_PLOT(
		"ORG.CompileProfile.TotalNs",
		static_cast<int64_t>(basic_telemetry::NowNs() - m_compileProfileFrameStartedAtNs));
	m_compileProfileFrame.reset();
}

void RenderGraph::RecordCompileProfileCounters(const std::vector<Node>& nodes, std::span<const uint64_t> usedResourceIDs) {
	if (!m_compileProfileFrame) {
		return;
	}

	uint64_t requirementCount = 0;
	for (const auto& summary : m_framePassSchedulingSummaries) {
		requirementCount += summary.requirements.size();
	}

	uint64_t dagEdgeCount = 0;
	for (const auto& node : nodes) {
		dagEdgeCount += node.out.size();
	}

	uint64_t transitionCount = 0;
	uint64_t queueWaitCount = 0;
	uint64_t queueSignalCount = 0;
	for (const auto& batch : batches) {
		const size_t queueCount = batch.QueueCount();
		for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
			for (size_t phaseIndex = 0; phaseIndex < PassBatch::kTransitionPhaseCount; ++phaseIndex) {
				transitionCount += batch.queueTransitions[phaseIndex][queueIndex].size();
			}
			for (size_t phaseIndex = 0; phaseIndex < PassBatch::kSignalPhaseCount; ++phaseIndex) {
				if (batch.queueSignalEnabled[phaseIndex][queueIndex]) {
					++queueSignalCount;
				}
			}
			for (size_t waitPhaseIndex = 0; waitPhaseIndex < PassBatch::kWaitPhaseCount; ++waitPhaseIndex) {
				for (size_t srcIndex = 0; srcIndex < queueCount; ++srcIndex) {
					if (batch.queueWaitEnabled[waitPhaseIndex][queueIndex][srcIndex]) {
						++queueWaitCount;
					}
				}
			}
		}
	}

	const uint64_t passCount = m_framePasses.size();
	const uint64_t resourceCount = m_frameSchedulingResourceCount != 0 ? m_frameSchedulingResourceCount : usedResourceIDs.size();
	const uint64_t batchCount = batches.size();
	const uint64_t aliasPlacementCount = static_cast<uint64_t>(std::count_if(
		m_hasAliasPlacementByResourceIndex.begin(),
		m_hasAliasPlacementByResourceIndex.end(),
		[](uint8_t hasPlacement) { return hasPlacement != 0; }));
	m_compileProfileFrame->SetDimension("passes", static_cast<int64_t>(passCount));
	m_compileProfileFrame->SetDimension("resources", static_cast<int64_t>(resourceCount));
	m_compileProfileFrame->SetDimension("requirements", static_cast<int64_t>(requirementCount));
	m_compileProfileFrame->SetDimension("dag_edges", static_cast<int64_t>(dagEdgeCount));
	m_compileProfileFrame->SetDimension("batches", static_cast<int64_t>(batchCount));
	m_compileProfileFrame->SetDimension("transitions", static_cast<int64_t>(transitionCount));
	m_compileProfileFrame->SetDimension("queue_waits", static_cast<int64_t>(queueWaitCount));
	m_compileProfileFrame->SetDimension("queue_signals", static_cast<int64_t>(queueSignalCount));
	m_compileProfileFrame->SetDimension("alias_placements", static_cast<int64_t>(aliasPlacementCount));
	m_compileProfileFrame->SetDimension("materialization_candidates", static_cast<int64_t>(m_lastMaterializeCandidateCount));
	BT_PLOT("ORG.CompileProfile.PassCount", static_cast<int64_t>(passCount));
	BT_PLOT("ORG.CompileProfile.ResourceCount", static_cast<int64_t>(resourceCount));
	BT_PLOT("ORG.CompileProfile.RequirementCount", static_cast<int64_t>(requirementCount));
	BT_PLOT("ORG.CompileProfile.DagEdgeCount", static_cast<int64_t>(dagEdgeCount));
	BT_PLOT("ORG.CompileProfile.BatchCount", static_cast<int64_t>(batchCount));
	BT_PLOT("ORG.CompileProfile.TransitionCount", static_cast<int64_t>(transitionCount));
	BT_PLOT("ORG.CompileProfile.QueueWaitCount", static_cast<int64_t>(queueWaitCount));
	BT_PLOT("ORG.CompileProfile.QueueSignalCount", static_cast<int64_t>(queueSignalCount));
	BT_PLOT("ORG.CompileProfile.AliasPlacementCount", static_cast<int64_t>(aliasPlacementCount));
	BT_PLOT("ORG.CompileProfile.MaterializationCandidateCount", static_cast<int64_t>(m_lastMaterializeCandidateCount));
}
namespace {
bool HasLiveCompileResourceBacking(Resource* resource) {
	if (!resource) {
		return false;
	}

	if (auto* backedResource = TryGetBackedResource(resource)) {
		return backedResource->IsMaterialized();
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
		if (!entry.tracker.has_value()) {
			entry.tracker.emplace();
		}
		bool copiedLiveTracker = false;
		if (entry.resource && HasLiveCompileResourceBacking(entry.resource)) {
			if (auto* liveTracker = entry.resource->GetStateTracker()) {
				entry.tracker->CopyFrom(*liveTracker);
				copiedLiveTracker = true;
			}
		}
		if (!copiedLiveTracker) {
			RangeSpec wholeRange;
			wholeRange.mipLower = { BoundType::All, 0 };
			wholeRange.mipUpper = { BoundType::All, 0 };
			wholeRange.sliceLower = { BoundType::All, 0 };
			wholeRange.sliceUpper = { BoundType::All, 0 };
			entry.tracker->Reset(
				wholeRange,
				ResourceState{
					rhi::ResourceAccessType::None,
					rhi::ResourceLayout::Undefined,
					rhi::ResourceSyncState::None });
		}
		entry.trackerInitialized = true;
		entry.readOnlyUniformTransitionChecked = false;
		entry.fastState.valid = TryGetWholeResourceTrackerState(*entry.tracker, entry.fastState.state);
		entry.fastState.wholeResourceOnly = entry.fastState.valid;
	}
	return entry;
}

void RenderGraph::RebuildFrameCompileResources() {
	BT_ZONE_SCOPE("RenderGraph::RebuildFrameCompileResources");
	if (m_frameCompileResources.size() < m_frameSchedulingResourceCount) {
		m_frameCompileResources.resize(m_frameSchedulingResourceCount);
	}
	else if (m_frameCompileResources.size() > m_frameSchedulingResourceCount) {
		m_frameCompileResources.resize(m_frameSchedulingResourceCount);
	}
	for (auto& entry : m_frameCompileResources) {
		entry.resourceID = 0;
		entry.resource = nullptr;
		entry.trackerInitialized = false;
		entry.readOnlyUniformTransitionChecked = false;
		entry.fastState = {};
	}

	auto& preferredDynamicStableIDByIndex = m_compilerState->preferredDynamicStableIDByIndex;
	if (preferredDynamicStableIDByIndex.size() != m_frameSchedulingResourceCount) {
		preferredDynamicStableIDByIndex.assign(m_frameSchedulingResourceCount, 0);
	}
	else {
		std::fill(preferredDynamicStableIDByIndex.begin(), preferredDynamicStableIDByIndex.end(), 0);
	}
	auto& trackerBackingGenerationByIndex =
		m_compilerState->compileTrackerBackingGenerationByIndex;
	auto& trackerPublishableByIndex =
		m_compilerState->compileTrackerPublishableByIndex;
	trackerBackingGenerationByIndex.assign(m_frameSchedulingResourceCount, 0);
	trackerPublishableByIndex.assign(m_frameSchedulingResourceCount, 0);
	for (const auto& [stableID, resource] : m_dynamicResourcesByStableID) {
		if (!resource) {
			continue;
		}
		auto resourceIndex = TryGetFrameSchedulingResourceIndex(stableID);
		if (resourceIndex.has_value() && *resourceIndex < preferredDynamicStableIDByIndex.size()) {
			preferredDynamicStableIDByIndex[*resourceIndex] = stableID;
		}
	}

	for (const auto& [resourceID, resourceIndex] : m_frameSchedulingResourceIndexEntries) {
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
		entry.readOnlyUniformTransitionChecked = false;
		BackedResource* backedResource = entry.resource ? TryGetBackedResource(entry.resource) : nullptr;
		const bool hasLiveBacking =
			entry.resource && (!backedResource || backedResource->IsMaterialized());
		if (hasLiveBacking) {
			if (auto* liveTracker = entry.resource->GetStateTracker()) {
				trackerPublishableByIndex[resourceIndex] = 1;
				trackerBackingGenerationByIndex[resourceIndex] =
					backedResource ? backedResource->GetBackingGeneration() : 0;
				entry.fastState.valid = TryGetWholeResourceTrackerState(*liveTracker, entry.fastState.state);
				entry.fastState.wholeResourceOnly = entry.fastState.valid;
			}
		}
		else {
			const ResourceState initialState{
				rhi::ResourceAccessType::None,
				rhi::ResourceLayout::Undefined,
				rhi::ResourceSyncState::None,
			};
			entry.fastState.valid = true;
			entry.fastState.wholeResourceOnly = true;
			entry.fastState.state = initialState;
		}
		if (resourceIndex < m_frameResourceAccessSummaries.size()) {
			const auto& accessSummary = m_frameResourceAccessSummaries[resourceIndex];
			entry.readOnlyUniformTransitionChecked =
				accessSummary.readOnlyUniform
				&& entry.fastState.valid
				&& entry.fastState.wholeResourceOnly
				&& entry.resource
				&& hasLiveBacking
				&& StatesExactlyEqual(entry.fastState.state, accessSummary.uniformState);
		}
	}
}

void RenderGraph::CaptureCompileTrackersForExecution(std::span<const uint64_t> resourceIDs) {
	BT_ZONE_SCOPE("RenderGraph::CaptureCompileTrackersForExecution");
	trackers.clear();
	if (trackers.capacity() < resourceIDs.size()) {
		trackers.reserve(resourceIDs.size());
	}

	const auto& schedulingIndexByDagIndex =
		m_compilerState->schedulingResourceIndexByDagResourceIndex;
	const auto& trackerBackingGenerationByIndex =
		m_compilerState->compileTrackerBackingGenerationByIndex;
	const auto& trackerPublishableByIndex =
		m_compilerState->compileTrackerPublishableByIndex;
	for (size_t dagResourceIndex = 0; dagResourceIndex < resourceIDs.size(); ++dagResourceIndex) {
		if (dagResourceIndex >= schedulingIndexByDagIndex.size()) {
			break;
		}
		const size_t resourceIndex = schedulingIndexByDagIndex[dagResourceIndex];
		if (resourceIndex >= m_frameCompileResources.size()
			|| resourceIndex >= trackerPublishableByIndex.size()
			|| !trackerPublishableByIndex[resourceIndex]) {
			continue;
		}
		const auto& compileResourceState = m_frameCompileResources[resourceIndex];
		if (!compileResourceState.trackerInitialized || !compileResourceState.tracker.has_value()) {
			continue;
		}
		trackers.push_back(CapturedTrackerResource{
			.resourceID = resourceIDs[dagResourceIndex],
			.backingGeneration = trackerBackingGenerationByIndex[resourceIndex],
		});
	}
}

void RenderGraph::PublishCompiledTrackerStates() {
	for (const auto& captured : trackers) {
		const uint64_t resourceID = captured.resourceID;
		auto resourceIndex = TryGetFrameSchedulingResourceIndex(resourceID);
		if (!resourceIndex.has_value() || *resourceIndex >= m_frameCompileResources.size()) {
			continue;
		}

		auto& compileResourceState = m_frameCompileResources[*resourceIndex];
		if (!compileResourceState.trackerInitialized) {
			continue;
		}

		// Compilation stores a non-owning pointer for hot-path access, but scene
		// publication may replace the registered resource before execution ends.
		// Reacquire ownership before dereferencing it and do not publish a state
		// compiled for a different resource object that reused the same ID.
		auto liveResource = GetResourceByID(resourceID);
		Resource* resource = liveResource.get();
		if (!resource
			|| resource != compileResourceState.resource
			|| !HasLiveCompileResourceBacking(resource)) {
			continue;
		}

		uint64_t currentBackingGeneration = 0u;
		if (auto* backedResource = TryGetBackedResource(resource)) {
			currentBackingGeneration = backedResource->GetBackingGeneration();
		}
		if (captured.backingGeneration != 0u
			&& currentBackingGeneration != 0u
			&& captured.backingGeneration != currentBackingGeneration) {
			spdlog::warn(
				"RenderGraph: resource '{}' id={} backing changed during execution; publishing compiled state to current tracker capturedGeneration={} currentGeneration={}",
				resource->GetName(),
				resourceID,
				captured.backingGeneration,
				currentBackingGeneration);
		}

		auto* liveTracker = resource->GetStateTracker();
		if (!liveTracker) {
			continue;
		}
		// Multi-representation state is predicted by the ownership planner for
		// each device independently. Publishing the legacy single logical tracker
		// here would erase the COMMON release state (or the destination-local
		// acquired state) and corrupt the next frame's barrier before-state.
		if (resource->GetRepresentationInstances().size() > 1) {
			continue;
		}
		if (!compileResourceState.tracker.has_value()) {
			continue;
		}
		liveTracker->CopyFrom(*compileResourceState.tracker);
	}
}

void RenderGraph::ClearFrameSchedulingResourceIndex() {
	m_frameSchedulingResourceIndexByID.clear();
	m_frameSchedulingResourceIndexEntries.clear();
	m_frameSchedulingResourceIndexLookup.clear();
	m_frameSchedulingResourceCount = 0;
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
	// Retain vector capacities across full recompiles. These summaries are
	// rebuilt for the active frame before use, and several passes have very
	// large requirement lists.
}

RenderGraph::PassBatch RenderGraph::AcquireReusablePassBatch(size_t queueCount) {
	if (m_reusablePassBatches.empty()) {
		return PassBatch(queueCount);
	}
	PassBatch batch = std::move(m_reusablePassBatches.back());
	m_reusablePassBatches.pop_back();
	batch.Reset(queueCount);
	return batch;
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

void RenderGraph::RebuildFrameSchedulingResourceIndex(std::span<const uint64_t> resourceIDs) {
	BT_ZONE_SCOPE("RenderGraph::RebuildFrameSchedulingResourceIndex");
	m_frameSchedulingResourceIndexByID.clear();
	m_frameSchedulingResourceIndexEntries.clear();
	m_frameSchedulingResourceCount = 0;

	std::vector<uint64_t> sortedResourceIDs(resourceIDs.begin(), resourceIDs.end());
	std::sort(sortedResourceIDs.begin(), sortedResourceIDs.end());
	sortedResourceIDs.erase(std::unique(sortedResourceIDs.begin(), sortedResourceIDs.end()), sortedResourceIDs.end());
	if (sortedResourceIDs.capacity() < resourceIDs.size() * 2) {
		sortedResourceIDs.reserve(resourceIDs.size() * 2);
	}
	const size_t baseResourceIDCount = sortedResourceIDs.size();
	for (size_t index = 0; index < baseResourceIDCount; ++index) {
		const uint64_t resourceID = sortedResourceIDs[index];
		const auto& equivalentIDs = GetSchedulingEquivalentIDsCached(resourceID);
		if (equivalentIDs.size() <= 1) {
			if (!equivalentIDs.empty() && equivalentIDs.front() != resourceID) {
				sortedResourceIDs.push_back(equivalentIDs.front());
			}
			continue;
		}
		for (uint64_t equivalentID : equivalentIDs) {
			sortedResourceIDs.push_back(equivalentID);
		}
	}
	std::sort(sortedResourceIDs.begin(), sortedResourceIDs.end());
	sortedResourceIDs.erase(std::unique(sortedResourceIDs.begin(), sortedResourceIDs.end()), sortedResourceIDs.end());

	m_frameSchedulingResourceIndexEntries.reserve(sortedResourceIDs.size() + m_dynamicResourcesByStableID.size());
	for (uint64_t resourceID : sortedResourceIDs) {
		m_frameSchedulingResourceIndexEntries.emplace_back(resourceID, m_frameSchedulingResourceCount++);
	}

	auto findEntry = [&](uint64_t resourceID) {
		return std::lower_bound(
			m_frameSchedulingResourceIndexEntries.begin(),
			m_frameSchedulingResourceIndexEntries.end(),
			resourceID,
			[](const auto& entry, uint64_t value) {
				return entry.first < value;
			});
	};

	for (const auto& [stableID, resource] : m_dynamicResourcesByStableID) {
		if (!resource) {
			continue;
		}
		const uint64_t backingID = resource->GetGlobalResourceID();
		auto stableIt = findEntry(stableID);
		const bool hasStable = stableIt != m_frameSchedulingResourceIndexEntries.end() && stableIt->first == stableID;
		auto backingIt = findEntry(backingID);
		const bool hasBacking = backingIt != m_frameSchedulingResourceIndexEntries.end() && backingIt->first == backingID;
		if (hasStable) {
			if (hasBacking) {
				backingIt->second = stableIt->second;
			}
			else {
				m_frameSchedulingResourceIndexEntries.emplace_back(backingID, stableIt->second);
			}
		}
		else if (hasBacking) {
			m_frameSchedulingResourceIndexEntries.emplace_back(stableID, backingIt->second);
		}
	}
	std::sort(
		m_frameSchedulingResourceIndexEntries.begin(),
		m_frameSchedulingResourceIndexEntries.end(),
		[](const auto& lhs, const auto& rhs) {
			return lhs.first < rhs.first;
		});
	if (m_frameSchedulingResourceIDByIndex.size() != m_frameSchedulingResourceCount) {
		m_frameSchedulingResourceIDByIndex.assign(m_frameSchedulingResourceCount, 0);
	}
	else {
		std::fill(m_frameSchedulingResourceIDByIndex.begin(), m_frameSchedulingResourceIDByIndex.end(), 0);
	}
	for (const auto& [resourceID, resourceIndex] : m_frameSchedulingResourceIndexEntries) {
		if (resourceIndex < m_frameSchedulingResourceIDByIndex.size()) {
			m_frameSchedulingResourceIDByIndex[resourceIndex] = resourceID;
		}
	}

	size_t lookupSize = 1;
	const size_t minLookupSize = (std::max)(size_t{ 2 }, m_frameSchedulingResourceIndexEntries.size() * 2);
	while (lookupSize < minLookupSize) {
		lookupSize <<= 1;
	}
	if (m_frameSchedulingResourceIndexLookup.size() != lookupSize) {
		m_frameSchedulingResourceIndexLookup.resize(lookupSize);
	}
	for (auto& entry : m_frameSchedulingResourceIndexLookup) {
		entry.occupied = 0;
	}
	const size_t lookupMask = lookupSize - 1;
	auto insertLookupEntry = [&](uint64_t resourceID, size_t resourceIndex) {
		size_t slot = (std::hash<uint64_t>{}(resourceID) * 11400714819323198485ull) & lookupMask;
		while (m_frameSchedulingResourceIndexLookup[slot].occupied != 0) {
			slot = (slot + 1) & lookupMask;
		}
		auto& lookupEntry = m_frameSchedulingResourceIndexLookup[slot];
		lookupEntry.resourceID = resourceID;
		lookupEntry.resourceIndex = resourceIndex;
		lookupEntry.occupied = 1;
	};
	for (const auto& [resourceID, resourceIndex] : m_frameSchedulingResourceIndexEntries) {
		insertLookupEntry(resourceID, resourceIndex);
	}

	{
		BT_ZONE_SCOPE("RenderGraph::RebuildFrameSchedulingResourceIndex::BuildDAGToSchedulingIndex");
		auto& schedulingIndexByDagIndex =
			m_compilerState->schedulingResourceIndexByDagResourceIndex;
		schedulingIndexByDagIndex.resize(m_frameDAGResourceIDsByIndex.size());
		for (size_t dagResourceIndex = 0;
			dagResourceIndex < m_frameDAGResourceIDsByIndex.size();
			++dagResourceIndex) {
			const auto schedulingResourceIndex =
				TryGetFrameSchedulingResourceIndex(m_frameDAGResourceIDsByIndex[dagResourceIndex]);
			schedulingIndexByDagIndex[dagResourceIndex] =
				schedulingResourceIndex.value_or(SIZE_MAX);
		}
	}

	m_aliasActivationPendingByResourceIndex.assign(m_frameSchedulingResourceCount, org::alias::AliasActivationReason::None);
	for (const auto& [resourceID, reason] : aliasActivationPending) {
		auto resourceIndex = TryGetFrameSchedulingResourceIndex(resourceID);
		if (resourceIndex.has_value()) {
			m_aliasActivationPendingByResourceIndex[*resourceIndex] = reason;
		}
	}
}

void RenderGraph::RebuildEquivalentResourceIndicesByResourceIndex() {
	BT_ZONE_SCOPE("RenderGraph::RebuildEquivalentResourceIndicesByResourceIndex");
	if (m_equivalentResourceIndicesByResourceIndex.size() != m_frameSchedulingResourceCount) {
		m_equivalentResourceIndicesByResourceIndex.resize(m_frameSchedulingResourceCount);
	}
	for (auto& equivalentIndices : m_equivalentResourceIndicesByResourceIndex) {
		equivalentIndices.clear();
	}
	const bool hasSchedulingPlacements = std::any_of(
		m_hasSchedulingPlacementByResourceIndex.begin(),
		m_hasSchedulingPlacementByResourceIndex.end(),
		[](uint8_t hasPlacement) { return hasPlacement != 0; });
	if (!hasSchedulingPlacements) {
		return;
	}

	for (const auto& [resourceID, resourceIndex] : m_frameSchedulingResourceIndexEntries) {
		if (resourceIndex >= m_equivalentResourceIndicesByResourceIndex.size()) {
			continue;
		}

		auto& equivalentIndices = m_equivalentResourceIndicesByResourceIndex[resourceIndex];
		for (uint64_t equivalentID : GetSchedulingEquivalentIDsCached(resourceID)) {
			if (equivalentID == resourceID) {
				continue;
			}

			auto equivalentIndex = TryGetFrameSchedulingResourceIndex(equivalentID);
			if (equivalentIndex.has_value()) {
				equivalentIndices.push_back(*equivalentIndex);
			}
		}

		std::sort(equivalentIndices.begin(), equivalentIndices.end());
		equivalentIndices.erase(std::unique(equivalentIndices.begin(), equivalentIndices.end()), equivalentIndices.end());
	}
}

void RenderGraph::RebuildFramePassSchedulingSummaries() {
	BT_ZONE_SCOPE("RenderGraph::RebuildFramePassSchedulingSummaries");
	m_framePassSchedulingSummaries.resize(m_framePassAccessSummaries.size());
	if (m_frameResourceAccessSummaries.size() != m_frameSchedulingResourceCount) {
		m_frameResourceAccessSummaries.assign(m_frameSchedulingResourceCount, FrameResourceAccessSummary{});
	}
	else {
		std::fill(m_frameResourceAccessSummaries.begin(), m_frameResourceAccessSummaries.end(), FrameResourceAccessSummary{});
	}
	auto& resourceEpochs = m_compilerState->schedulingSummaryResourceEpochs;
	auto& writeEpochs = m_compilerState->schedulingSummaryWriteEpochs;
	auto& uavEpochs = m_compilerState->schedulingSummaryUAVEpochs;
	if (resourceEpochs.size() < m_frameSchedulingResourceCount) {
		resourceEpochs.resize(m_frameSchedulingResourceCount, 0);
		writeEpochs.resize(m_frameSchedulingResourceCount, 0);
		uavEpochs.resize(m_frameSchedulingResourceCount, 0);
	}
	if (m_compilerState->schedulingSummaryEpoch
		>= std::numeric_limits<uint32_t>::max() - static_cast<uint32_t>(m_framePassAccessSummaries.size())) {
		std::fill(resourceEpochs.begin(), resourceEpochs.end(), 0);
		std::fill(writeEpochs.begin(), writeEpochs.end(), 0);
		std::fill(uavEpochs.begin(), uavEpochs.end(), 0);
		m_compilerState->schedulingSummaryEpoch = 1;
	}

	auto buildPassSchedulingSummary = [&](size_t passIndex) {
		const uint32_t passEpoch = m_compilerState->schedulingSummaryEpoch++;
		auto& summary = m_framePassSchedulingSummaries[passIndex];
		const auto& passAccess = m_framePassAccessSummaries[passIndex];
		summary.requirements.clear();
		summary.internalTransitions.clear();
		summary.requiredResourceIndices.clear();
		summary.waitDependencyResourceIndices.clear();
		summary.touchedResourceIndices.clear();
		summary.writtenResourceIndices.clear();
		summary.uavResourceIndices.clear();
		if (summary.requirements.capacity() < passAccess.requirementSummaries.size()) {
			summary.requirements.reserve(passAccess.requirementSummaries.size());
		}
		if (summary.internalTransitions.capacity() < passAccess.internalTransitionSummaries.size()) {
			summary.internalTransitions.reserve(passAccess.internalTransitionSummaries.size());
		}
		if (summary.requiredResourceIndices.capacity() < passAccess.requirementSummaries.size()) {
			summary.requiredResourceIndices.reserve(passAccess.requirementSummaries.size());
		}
		if (summary.waitDependencyResourceIndices.capacity() < passAccess.requirementSummaries.size()) {
			summary.waitDependencyResourceIndices.reserve(passAccess.requirementSummaries.size());
		}
		const size_t touchedResourceCapacity = passAccess.requirementSummaries.size() + passAccess.internalTransitionSummaries.size();
		if (summary.touchedResourceIndices.capacity() < touchedResourceCapacity) {
			summary.touchedResourceIndices.reserve(touchedResourceCapacity);
		}
		if (summary.uavResourceIndices.capacity() < passAccess.requirementSummaries.size()) {
			summary.uavResourceIndices.reserve(passAccess.requirementSummaries.size());
		}
		if (summary.writtenResourceIndices.capacity() < passAccess.requirementSummaries.size()) {
			summary.writtenResourceIndices.reserve(passAccess.requirementSummaries.size());
		}

		for (const auto& req : passAccess.requirementSummaries) {
			const size_t resourceIndex =
				req.dagResourceIndex < m_compilerState->schedulingResourceIndexByDagResourceIndex.size()
				? m_compilerState->schedulingResourceIndexByDagResourceIndex[req.dagResourceIndex]
				: SIZE_MAX;
			if (resourceIndex == SIZE_MAX) {
				continue;
			}

			const bool isWholeResource = IsWholeResourceRange(req.range, req.resource);
			DenseRequirementSummary denseRequirement{};
			denseRequirement.resource = req.resource;
			denseRequirement.resourceID = req.resourceID;
			denseRequirement.resourceIndex = resourceIndex;
			denseRequirement.range = req.range;
			denseRequirement.state = req.state;
			denseRequirement.isUAV = req.isUAV;
			denseRequirement.isWholeResource = isWholeResource;
			if (resourceIndex < m_equivalentResourceIndicesByResourceIndex.size()) {
				denseRequirement.equivalentResourceIndices = &m_equivalentResourceIndicesByResourceIndex[resourceIndex];
			}
			summary.requirements.push_back(std::move(denseRequirement));
			if (resourceEpochs[resourceIndex] != passEpoch) {
				resourceEpochs[resourceIndex] = passEpoch;
				summary.requiredResourceIndices.push_back(resourceIndex);
			}
			if (req.isUAV && uavEpochs[resourceIndex] != passEpoch) {
				uavEpochs[resourceIndex] = passEpoch;
				summary.uavResourceIndices.push_back(resourceIndex);
			}

			auto& accessSummary = m_frameResourceAccessSummaries[resourceIndex];
			const bool isWrite = AccessTypeIsWriteType(req.state.access);
			if (isWrite && writeEpochs[resourceIndex] != passEpoch) {
				writeEpochs[resourceIndex] = passEpoch;
				summary.writtenResourceIndices.push_back(resourceIndex);
			}
			accessSummary.hasWrite = accessSummary.hasWrite || isWrite;
			accessSummary.hasUAV = accessSummary.hasUAV || req.isUAV;
			accessSummary.hasAliasActivation = accessSummary.hasAliasActivation
				|| (resourceIndex < m_aliasActivationPendingByResourceIndex.size()
					&& m_aliasActivationPendingByResourceIndex[resourceIndex] != org::alias::AliasActivationReason::None);
			accessSummary.hasNonWholeResourceRange = accessSummary.hasNonWholeResourceRange || !isWholeResource;
		}

		for (const auto& transition : passAccess.internalTransitionSummaries) {
			const size_t resourceIndex =
				transition.dagResourceIndex < m_compilerState->schedulingResourceIndexByDagResourceIndex.size()
				? m_compilerState->schedulingResourceIndexByDagResourceIndex[transition.dagResourceIndex]
				: SIZE_MAX;
			if (resourceIndex == SIZE_MAX) {
				continue;
			}

			DenseEquivalentResourceSummary denseTransition{};
			denseTransition.resourceID = transition.resourceID;
			denseTransition.resourceIndex = resourceIndex;
			if (resourceIndex < m_equivalentResourceIndicesByResourceIndex.size()) {
				denseTransition.equivalentResourceIndices = &m_equivalentResourceIndicesByResourceIndex[resourceIndex];
			}
			summary.internalTransitions.push_back(std::move(denseTransition));
			m_frameResourceAccessSummaries[resourceIndex].hasInternalTransition = true;
		}

		summary.touchedResourceIndices.assign(
			summary.requiredResourceIndices.begin(),
			summary.requiredResourceIndices.end());
		if (!summary.internalTransitions.empty()) {
			for (const auto& transition : summary.internalTransitions) {
				if (resourceEpochs[transition.resourceIndex] != passEpoch) {
					resourceEpochs[transition.resourceIndex] = passEpoch;
					summary.touchedResourceIndices.push_back(transition.resourceIndex);
				}
			}
		}

		summary.waitDependencyResourceIndices.assign(
			summary.requiredResourceIndices.begin(),
			summary.requiredResourceIndices.end());
		bool addedEquivalentResource = false;
		for (size_t resourceIndex : summary.requiredResourceIndices) {
			if (resourceIndex >= m_equivalentResourceIndicesByResourceIndex.size()) {
				continue;
			}
			const auto& equivalents = m_equivalentResourceIndicesByResourceIndex[resourceIndex];
			if (!equivalents.empty()) {
				summary.waitDependencyResourceIndices.insert(
					summary.waitDependencyResourceIndices.end(),
					equivalents.begin(),
					equivalents.end());
				addedEquivalentResource = true;
			}
		}
		if (addedEquivalentResource && summary.waitDependencyResourceIndices.size() > 1) {
			std::sort(summary.waitDependencyResourceIndices.begin(), summary.waitDependencyResourceIndices.end());
			summary.waitDependencyResourceIndices.erase(
				std::unique(summary.waitDependencyResourceIndices.begin(), summary.waitDependencyResourceIndices.end()),
				summary.waitDependencyResourceIndices.end());
		}

	};

	for (size_t passIndex = 0; passIndex < m_framePassAccessSummaries.size(); ++passIndex) {
		buildPassSchedulingSummary(passIndex);
	}
}

void RenderGraph::RebuildFrameResourceAccessSummaries(const std::vector<Node>& nodes) {
	BT_ZONE_SCOPE("RenderGraph::RebuildFrameResourceAccessSummaries");
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

		std::array<size_t, 64> activeCompatibleSlots{};
		size_t activeCompatibleSlotCount = 0;
		auto appendActiveCompatibleSlot = [&](size_t queueSlot) {
			for (size_t index = 0; index < activeCompatibleSlotCount; ++index) {
				if (activeCompatibleSlots[index] == queueSlot) {
					return;
				}
			}
			if (activeCompatibleSlotCount < activeCompatibleSlots.size()) {
				activeCompatibleSlots[activeCompatibleSlotCount++] = queueSlot;
			}
		};
		for (size_t queueSlot : node.compatibleQueueSlots) {
			if (queueSlot < m_activeQueueSlotsThisFrame.size() && m_activeQueueSlotsThisFrame[queueSlot]) {
				appendActiveCompatibleSlot(queueSlot);
			}
		}
		if (activeCompatibleSlotCount == 0) {
			appendActiveCompatibleSlot(node.queueSlot);
		}

		for (const auto& requirement : passSummary.requirements) {
			if (requirement.resourceIndex >= m_frameResourceAccessSummaries.size()) {
				continue;
			}

			auto& accessSummary = m_frameResourceAccessSummaries[requirement.resourceIndex];
			for (size_t slotIndex = 0; slotIndex < activeCompatibleSlotCount; ++slotIndex) {
				const size_t queueSlot = activeCompatibleSlots[slotIndex];
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
	const RenderGraph::FramePassList& framePasses,
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

std::optional<size_t> RenderGraph::TryGetFrameSchedulingResourceIndex(uint64_t resourceID) const {
	if (m_frameSchedulingResourceIndexLookup.empty()) {
		return std::nullopt;
	}
	const size_t lookupMask = m_frameSchedulingResourceIndexLookup.size() - 1;
	size_t slot = (std::hash<uint64_t>{}(resourceID) * 11400714819323198485ull) & lookupMask;
	while (m_frameSchedulingResourceIndexLookup[slot].occupied != 0) {
		const auto& entry = m_frameSchedulingResourceIndexLookup[slot];
		if (entry.resourceID == resourceID) {
			return entry.resourceIndex;
		}
		slot = (slot + 1) & lookupMask;
	}
	return std::nullopt;
}

const org::alias::AliasPlacementRange* RenderGraph::TryGetAliasPlacementRangeByResourceIndex(size_t resourceIndex) const {
	if (resourceIndex >= m_hasAliasPlacementByResourceIndex.size() || resourceIndex >= m_aliasPlacementRangeByResourceIndex.size()) {
		return nullptr;
	}
	if (m_hasAliasPlacementByResourceIndex[resourceIndex] == 0) {
		return nullptr;
	}
	return &m_aliasPlacementRangeByResourceIndex[resourceIndex];
}

const org::alias::AliasPlacementRange* RenderGraph::TryGetAliasPlacementRange(uint64_t resourceID) const {
	auto resourceIndex = TryGetFrameSchedulingResourceIndex(resourceID);
	if (!resourceIndex.has_value()) {
		return nullptr;
	}
	return TryGetAliasPlacementRangeByResourceIndex(*resourceIndex);
}

const org::alias::AliasPlacementRange* RenderGraph::TryGetSchedulingPlacementRangeByResourceIndex(size_t resourceIndex) const {
	if (resourceIndex >= m_hasSchedulingPlacementByResourceIndex.size() || resourceIndex >= m_schedulingPlacementRangeByResourceIndex.size()) {
		return nullptr;
	}
	if (m_hasSchedulingPlacementByResourceIndex[resourceIndex] == 0) {
		return nullptr;
	}
	return &m_schedulingPlacementRangeByResourceIndex[resourceIndex];
}

const org::alias::AliasPlacementRange* RenderGraph::TryGetSchedulingPlacementRange(uint64_t resourceID) const {
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
	for (const auto& [candidateID, candidateIndex] : m_frameSchedulingResourceIndexEntries) {
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
	std::sort(out.begin(), out.end());
	out.erase(std::unique(out.begin(), out.end()), out.end());

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
		resource = UnwrapDynamicResource(resource);
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

		auto buffer = dynamic_cast<BufferBase*>(resource);
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
	BT_ZONE_SCOPE("RenderGraph::CollectFrameResourceIDs");
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

void RenderGraph::ApplyIdleDematerializationPolicy(std::span<const uint64_t> usedResourceIDs) {
	BT_ZONE_SCOPE("RenderGraph::ApplyIdleDematerializationPolicy");
	auto isResourceUsed = [&](uint64_t id) {
		if (id == kFrameDAGResourceIndexEmptyKey || m_frameDAGResourceIndexHashKeys.empty()) {
			return std::find(usedResourceIDs.begin(), usedResourceIDs.end(), id) != usedResourceIDs.end();
		}

		const size_t hashMask = m_frameDAGResourceIndexHashKeys.size() - 1;
		size_t hashSlot = static_cast<size_t>(MixFrameDAGResourceID(id)) & hashMask;
		for (;;) {
			const uint64_t key = m_frameDAGResourceIndexHashKeys[hashSlot];
			if (key == id) {
				return true;
			}
			if (key == kFrameDAGResourceIndexEmptyKey) {
				return false;
			}
			hashSlot = (hashSlot + 1) & hashMask;
		}
	};
	for (auto& [id, resource] : resourcesByID) {
		if (!resource) {
			continue;
		}

		auto texture = std::dynamic_pointer_cast<PixelBuffer>(resource);
		if (!texture || !texture->IsIdleDematerializationEnabled()) {
			continue;
		}

		if (isResourceUsed(id)) {
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

void RenderGraph::SnapshotCompiledResourceGenerations(std::span<const uint64_t> usedResourceIDs) {
	compiledResourceGenerations.clear();
	if (compiledResourceGenerations.capacity() < usedResourceIDs.size()) {
		compiledResourceGenerations.reserve(usedResourceIDs.size());
	}

	for (uint64_t id : usedResourceIDs) {
		auto it = resourcesByID.find(id);
		if (it == resourcesByID.end() || !it->second) {
			continue;
		}

		auto* backedResource = TryGetBackedResource(it->second.get());
		if (backedResource) {
			compiledResourceGenerations.emplace_back(id, backedResource->GetBackingGeneration());
		}
	}
}

void RenderGraph::ValidateCompiledResourceGenerations() const {
	for (auto const& [id, compiledGeneration] : compiledResourceGenerations) {
		auto it = resourcesByID.find(id);
		if (it == resourcesByID.end() || !it->second) {
			continue;
		}

		auto* backedResource = TryGetBackedResource(it->second.get());
		if (backedResource) {
			const uint64_t currentGeneration = backedResource->GetBackingGeneration();
			if (compiledGeneration != 0u &&
				currentGeneration != 0u &&
				currentGeneration != compiledGeneration) {
				throw std::runtime_error(fmt::format(
					"Resource backing generation changed after compile and before execute. Resource ID: {} name='{}' compiledGeneration={} currentGeneration={}",
					id,
					it->second->GetName(),
					compiledGeneration,
					currentGeneration));
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

void RenderGraph::ShutdownExtensions() {
	for (auto& ext : m_extensions) {
		if (ext) {
			ext->Shutdown(*this);
		}
	}
}

void RenderGraph::ClearExtensions() {
	ShutdownExtensions();
	m_extensions.clear();
	m_extensionRegistrationIds.clear();
}

void RenderGraph::ResetForRebuild()
{
	if (m_pCommandRecordingManager) {
		m_pCommandRecordingManager->ShutdownThreadLocal();
		m_pCommandRecordingManager.reset();
	}


	// Clear any existing compile state
	m_masterPassList.clear();
	m_compilerState->immediateModePassPointers.clear();
	m_compilerState->immediateModeInterfaces.clear();
	m_retainedDeclarationRefreshCandidateMasterIndices.clear();
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
	// Alias pools are deliberately persistent allocations. A full structural
	// rebuild invalidates placements and cached plans, but it does not make a
	// compatible pool allocation unsafe to reuse once the caller has stalled the
	// GPU. Keeping the allocation avoids overlapping two very large pool
	// generations during runtime pipeline replacement.
	auto preservedAliasPools = std::move(persistentAliasPools);
	m_aliasingSubsystem.ResetPersistentState(*this);
	persistentAliasPools = std::move(preservedAliasPools);
	for (auto& [poolID, poolState] : persistentAliasPools) {
		(void)poolID;
		poolState.usedThisFrame = false;
	}
	m_lastProducerByResourceAcrossFrames.clear();
	m_lastAccessByResourceAcrossFrames.clear();
	m_lastAliasPlacementProducersByPoolAcrossFrames.clear();
	m_compilerState->compiledLastProducerBatchByResourceByQueue.clear();
	m_compilerState->compiledLastAccessBatchByResourceByQueue.clear();
	m_hasPendingFrameStartQueueWait.clear();
	m_pendingFrameStartQueueWaitFenceValue.clear();
	// Queue slots, their command-list pools, and timelines are runtime-owned and
	// survive structural graph generations. They are released only at shutdown.
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
	{
		BT_ZONE_SCOPE("RenderGraph::ResetCompileFrameState::Batches");
		// Keep every previously allocated batch in the reuse pool. Swapping after
		// clearing discarded any pool entries left over when the new frame used
		// fewer batches than the preceding frame, paying their potentially large
		// nested-vector destruction cost at the next frame boundary. AcquireReusablePassBatch
		// resets all observable state before reuse, so retaining the high-water
		// pool does not carry frame data forward.
		m_reusablePassBatches.reserve(m_reusablePassBatches.size() + batches.size());
		std::move(batches.begin(), batches.end(), std::back_inserter(m_reusablePassBatches));
		batches.clear();
	}
	{
		BT_ZONE_SCOPE("RenderGraph::ResetCompileFrameState::ResourceMaps");
		compiledResourceGenerations.clear();
		m_transientFrameResourcesByID.clear();
		m_transientFrameResourcesByName.clear();
	}
	{
		BT_ZONE_SCOPE("RenderGraph::ResetCompileFrameState::CompileVectors");
		m_addTransitionDebugStatsByResource.clear();
		m_schedulingEquivalentIDsCache.clear();
		ClearFrameSchedulingResourceIndex();
		ClearFramePassSchedulingSummaries();
		m_assignedQueueSlotsByFramePass.clear();
		m_activeQueueSlotsThisFrame.clear();
		m_executionSchedule.Reset();
	}
	{
		BT_ZONE_SCOPE("RenderGraph::ResetCompileFrameState::QueueHistory");
		for (auto& producerMap : m_compilerState->compiledLastProducerBatchByResourceByQueue) {
			producerMap.clear();
		}
		for (auto& accessMap : m_compilerState->compiledLastAccessBatchByResourceByQueue) {
			accessMap.clear();
		}
		for (auto& row : m_hasPendingFrameStartQueueWait) {
			std::fill(row.begin(), row.end(), false);
		}
		for (auto& row : m_pendingFrameStartQueueWaitFenceValue) {
			std::fill(row.begin(), row.end(), UINT64(0));
		}
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
	m_framePassAccessSummaryCache.clear();
	m_compilerState->densePassAccessKeys.clear();
	m_retainedDeclarationRefreshCandidateMasterIndices.clear();
}

void RenderGraph::ResetForFrame() {
	BT_ZONE_SCOPE("RenderGraph::ResetForFrame");
	{
		BT_ZONE_SCOPE("RenderGraph::ResetForFrame::ReclaimExpiredAnonymous");
		_registry.ReclaimExpiredAnonymous();
	}
	{
		BT_ZONE_SCOPE("RenderGraph::ResetForFrame::ResetAliasPerFrameState");
		m_aliasingSubsystem.ResetPerFrameState(*this);
	}
	{
		BT_ZONE_SCOPE("RenderGraph::ResetForFrame::ResetCompileFrameState");
		ResetCompileFrameState();
	}
}

void RenderGraph::CompileStructural() {
	// Register resource providers from pass builders

	std::vector<unsigned int> empty;
	{
	BT_ZONE_SCOPE("RenderGraph::CompileStructural::FinalizeBasePasses");
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
	m_compilerState->immediateModePassPointers.clear();
	m_compilerState->immediateModeInterfaces.clear();
	m_structuralExplicitAfterByName.clear();

	// Gather extension passes into ExtItem list
	std::vector<ExtItem> extItems;
	extItems.reserve(64);

	size_t globalOrder = 0;

	// helper: stable synthetic key for unnamed passes
	auto makeSyntheticKey = [](size_t n) -> std::string {
		return "__rg_ext_" + std::to_string(n);
		};

	{
	BT_ZONE_SCOPE("RenderGraph::CompileStructural::GatherAndMaterializeExtensions");
	for (int ei = 0; ei < (int)m_extensions.size(); ++ei) {
		auto& ext = m_extensions[ei];
		if (!ext) continue;

		std::vector<ExternalPassDesc> local;
		local.reserve(16);
		if (m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled()) {
			spdlog::info("RG gather structural extension {} begin", ei);
		}
		{
			BT_ZONE_SCOPE("RenderGraph::CompileStructural::GatherExtensionPasses");
			const auto gatherBegin = std::chrono::steady_clock::now();
			ext->GatherStructuralPasses(*this, local);
			const auto gatherMs = std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - gatherBegin).count();
			if (gatherMs >= 10.0) {
				const std::string_view extensionId = static_cast<std::size_t>(ei) < m_extensionRegistrationIds.size()
					? std::string_view{ m_extensionRegistrationIds[static_cast<std::size_t>(ei)] }
					: std::string_view{ "unknown" };
				spdlog::info(
					"RenderGraph structural extension gather: id='{}' elapsed_ms={:.3f} passes={}",
					extensionId,
					gatherMs,
					local.size());
			}
		}
		if (m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled()) {
			spdlog::info("RG gather structural extension {} complete localPassCount={}", ei, local.size());
		}

		std::optional<std::string> prevKey; // for extension-local chaining
		int localOrder = 0;

		for (auto& d : local) {
			BT_ZONE_SCOPE("RenderGraph::CompileStructural::MaterializeExtensionPass");
			BT_ZONE_TEXT(d.name.data(), d.name.size());
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
	}

	BT_ZONE_SCOPE("RenderGraph::CompileStructural::OrderPasses");
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
				if (!e.key.starts_with("CLodShadow::")) {
					spdlog::warn("External pass '{}' requested Before('{}') but anchor not found; ignoring.", e.key, b);
				}
				continue;
			}
			addEdge(passNode, *idxOpt);
			if (b != kBeginKey && b != kAfterBaseKey && b != kEndKey && b != kFirstBaseKey) {
				m_structuralExplicitAfterByName.push_back({ e.key, b });
			}
			anyConstraint = true;
		}
	}
	spdlog::debug("RenderGraph structural merge: external constraints complete passes={} edges={}", extItems.size(), edgeSet.size());

	// Extension chaining edges: prev -> next (if keepExtensionOrder on the *next* pass)
	for (auto& [ei, v] : extOrder) {
		spdlog::debug("RenderGraph structural merge: chaining extension={} passes={}", ei, v.size());
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
	spdlog::debug("RenderGraph structural merge: extension chaining complete edges={}", edgeSet.size());

	// Topological sort (stable by priority then order)
	spdlog::debug("RenderGraph structural merge: sorting {} nodes", nodes.size());
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

	spdlog::debug("RenderGraph structural merge: topological sort complete nodes={}", topo.size());
	// Emit final m_masterPassList in topo order (skip sentinels)
	m_masterPassList.clear();
	m_compilerState->immediateModePassPointers.clear();
	m_compilerState->immediateModeInterfaces.clear();
	m_masterPassList.reserve(baseIdx.size() + extIdx.size());

	for (size_t u : topo) {
		if (!nodes[u].hasPass) {
			continue;
		}
		m_masterPassList.push_back(std::move(nodes[u].pass));
	}
	spdlog::debug("RenderGraph structural merge: emitted {} passes; rebuilding retained declarations", m_masterPassList.size());
	RebuildRetainedDeclarationRefreshCandidates();
	spdlog::debug("RenderGraph structural merge: compile complete");
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
	BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Render)");
	const bool traceLifecycle = m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled();
	const uint64_t previousDeclarationFingerprint = p.declarationCache.declarationFingerprint;
	if (!p.name.empty()) {
		BT_ZONE_TEXT(p.name.data(), p.name.size());
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
	{
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Render)::EnsureProviderRegistered");
		EnsureProviderRegistered(p.pass.get());
	}
	{
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Render)::DeclareResourceUsages");
		if (!p.name.empty()) {
			BT_ZONE_TEXT(p.name.data(), p.name.size());
		}
		p.pass->DeclareResourceUsages(&b);
	}
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh render pass '{}' declare complete requirements={} transitions={}", frameIndex, p.name, b.GatherResourceRequirements().size(), b.params.internalTransitions.size());
	}
	auto refreshedRequirements = b.GatherResourceRequirements();
	BT_PLOT("ORG.RefreshRetained.Render.Requirements", static_cast<int64_t>(refreshedRequirements.size()));
	BT_PLOT("ORG.RefreshRetained.Render.InternalTransitions", static_cast<int64_t>(b.params.internalTransitions.size()));
	BT_PLOT("ORG.RefreshRetained.Render.DeclaredIds", static_cast<int64_t>(b.DeclaredResourceIds().size()));

	// Update the frame view used by scheduling
	{
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Render)::StoreRequirements");
		p.resources.staticResourceRequirements = std::move(refreshedRequirements);
		p.resources.mergedFrameRequirementsDirty = true;

		// Internal transitions also affect scheduling
		p.resources.internalTransitions = std::move(b.params.internalTransitions);

		p.resources.identifierSet = std::move(b._declaredIds);
		p.resources.autoDescriptorShaderResources = std::move(b.params.autoDescriptorShaderResources);
		p.resources.autoDescriptorConstantBuffers = std::move(b.params.autoDescriptorConstantBuffers);
		p.resources.autoDescriptorUnorderedAccessViews = std::move(b.params.autoDescriptorUnorderedAccessViews);
		p.resources.activeFeatureDomains = std::move(b.params.activeFeatureDomains);
		p.resources.externalWaitsBeforeTransitions = std::move(b.params.externalWaitsBeforeTransitions);
		p.resources.externalWaitBindingsBeforeTransitions = std::move(b.params.externalWaitBindingsBeforeTransitions);
	}
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh render pass '{}' materialize referenced resources begin", frameIndex, p.name);
	}
	{
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Render)::MaterializeReferencedResources");
		MaterializeReferencedResources(p.resources.staticResourceRequirements, p.resources.internalTransitions);
	}

	// Transfer resolver snapshots for auto-invalidation tracking
	{
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Render)::CaptureResolverSnapshots");
		p.resolverSnapshots = b.TakeResolverSnapshots();
		p.retainedAnonymousKeepAlive = CaptureRetainedAnonymousKeepAlive(
			p.resources.staticResourceRequirements,
			p.resources.internalTransitions);
	}
	{
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Render)::UpdateRetainedDeclarationCache");
		UpdateRetainedDeclarationCache(PassType::Render, p.name, p);
	}

	// A versioned resolver can replace the concrete resource behind an otherwise
	// unchanged identifier.  Rebuild the pass view/setup in that case so automatic
	// descriptor bindings follow the newly resolved resource instead of retaining
	// the descriptor captured from the previous resolver version.
	const bool requiresPassRebind = !p.resolverSnapshots.empty() ||
		!p.declarationCache.dynamicInterface ||
		p.declarationCache.dynamicInterface->RequiresPassRebindAfterDeclarationRefresh();
	// Only retained passes that resolve through their view or perform declaration-
	// dependent setup need these execution helpers rebuilt. Immediate-only upload
	// and readback passes consume the resources captured in their bytecode.
	if (requiresPassRebind) {
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Render)::SetResourceRegistryView");
		p.pass->SetResourceRegistryView(
			std::make_unique<ResourceRegistryView>(_registry, p.resources.identifierSet),
			p.resources.activeFeatureDomains,
			p.resources.autoDescriptorShaderResources,
			p.resources.autoDescriptorConstantBuffers,
			p.resources.autoDescriptorUnorderedAccessViews
		);
	}
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh render pass '{}' setup begin", frameIndex, p.name);
	}
	if (requiresPassRebind) {
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Render)::Setup");
		p.pass->Setup();
	}
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh render pass '{}' setup complete", frameIndex, p.name);
	}
	return previousDeclarationFingerprint != p.declarationCache.declarationFingerprint;
}

bool RenderGraph::RefreshRetainedDeclarationsForFrame(ComputePassAndResources& p, uint8_t frameIndex)
{
	BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Compute)");
	const bool traceLifecycle = m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled();
	const uint64_t previousDeclarationFingerprint = p.declarationCache.declarationFingerprint;
	if (!p.name.empty()) {
		BT_ZONE_TEXT(p.name.data(), p.name.size());
	}
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh compute pass '{}' declare begin", frameIndex, p.name);
	}
	ComputePassBuilder b(this, p.name);
	b.pass = p.pass;
	b.built_ = true;

	b.params = {};
	b._declaredIds.clear();

	{
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Compute)::EnsureProviderRegistered");
		EnsureProviderRegistered(p.pass.get());
	}
	{
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Compute)::DeclareResourceUsages");
		if (!p.name.empty()) {
			BT_ZONE_TEXT(p.name.data(), p.name.size());
		}
		p.pass->DeclareResourceUsages(&b);
	}
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh compute pass '{}' declare complete requirements={} transitions={}", frameIndex, p.name, b.GatherResourceRequirements().size(), b.params.internalTransitions.size());
	}
	auto refreshedRequirements = b.GatherResourceRequirements();
	BT_PLOT("ORG.RefreshRetained.Compute.Requirements", static_cast<int64_t>(refreshedRequirements.size()));
	BT_PLOT("ORG.RefreshRetained.Compute.InternalTransitions", static_cast<int64_t>(b.params.internalTransitions.size()));
	BT_PLOT("ORG.RefreshRetained.Compute.DeclaredIds", static_cast<int64_t>(b.DeclaredResourceIds().size()));

	{
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Compute)::StoreRequirements");
		p.resources.staticResourceRequirements = std::move(refreshedRequirements);
		p.resources.mergedFrameRequirementsDirty = true;
		p.resources.internalTransitions = std::move(b.params.internalTransitions);
		p.resources.identifierSet = std::move(b._declaredIds);
		p.resources.autoDescriptorShaderResources = std::move(b.params.autoDescriptorShaderResources);
		p.resources.autoDescriptorConstantBuffers = std::move(b.params.autoDescriptorConstantBuffers);
		p.resources.autoDescriptorUnorderedAccessViews = std::move(b.params.autoDescriptorUnorderedAccessViews);
		p.resources.activeFeatureDomains = std::move(b.params.activeFeatureDomains);
		p.resources.externalWaitsBeforeTransitions = std::move(b.params.externalWaitsBeforeTransitions);
		p.resources.externalWaitBindingsBeforeTransitions = std::move(b.params.externalWaitBindingsBeforeTransitions);
	}
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh compute pass '{}' materialize referenced resources begin", frameIndex, p.name);
	}
	{
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Compute)::MaterializeReferencedResources");
		MaterializeReferencedResources(p.resources.staticResourceRequirements, p.resources.internalTransitions);
	}

	// Transfer resolver snapshots for auto-invalidation tracking
	{
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Compute)::CaptureResolverSnapshots");
		p.resolverSnapshots = b.TakeResolverSnapshots();
		p.retainedAnonymousKeepAlive = CaptureRetainedAnonymousKeepAlive(
			p.resources.staticResourceRequirements,
			p.resources.internalTransitions);
	}
	{
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Compute)::UpdateRetainedDeclarationCache");
		UpdateRetainedDeclarationCache(PassType::Compute, p.name, p);
	}

	const bool requiresPassRebind = !p.resolverSnapshots.empty() ||
		!p.declarationCache.dynamicInterface ||
		p.declarationCache.dynamicInterface->RequiresPassRebindAfterDeclarationRefresh();
	if (requiresPassRebind) {
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Compute)::SetResourceRegistryView");
		p.pass->SetResourceRegistryView(
			std::make_unique<ResourceRegistryView>(_registry, p.resources.identifierSet),
			p.resources.activeFeatureDomains,
			p.resources.autoDescriptorShaderResources,
			p.resources.autoDescriptorConstantBuffers,
			p.resources.autoDescriptorUnorderedAccessViews
		);
	}
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh compute pass '{}' setup begin", frameIndex, p.name);
	}

	if (requiresPassRebind) {
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Compute)::Setup");
		p.pass->Setup();
	}
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh compute pass '{}' setup complete", frameIndex, p.name);
	}
	return previousDeclarationFingerprint != p.declarationCache.declarationFingerprint;
}

bool RenderGraph::RefreshRetainedDeclarationsForFrame(CopyPassAndResources& p, uint8_t frameIndex)
{
	BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Copy)");
	const bool traceLifecycle = m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled();
	const uint64_t previousDeclarationFingerprint = p.declarationCache.declarationFingerprint;
	if (!p.name.empty()) {
		BT_ZONE_TEXT(p.name.data(), p.name.size());
	}
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh copy pass '{}' declare begin", frameIndex, p.name);
	}
	CopyPassBuilder b(this, p.name);
	b.pass = p.pass;
	b.built_ = true;

	b.params = {};
	b._declaredIds.clear();

	{
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Copy)::EnsureProviderRegistered");
		EnsureProviderRegistered(p.pass.get());
	}
	{
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Copy)::DeclareResourceUsages");
		if (!p.name.empty()) {
			BT_ZONE_TEXT(p.name.data(), p.name.size());
		}
		p.pass->DeclareResourceUsages(&b);
	}
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh copy pass '{}' declare complete requirements={} transitions={}", frameIndex, p.name, b.GatherResourceRequirements().size(), b.params.internalTransitions.size());
	}
	auto refreshedRequirements = b.GatherResourceRequirements();
	BT_PLOT("ORG.RefreshRetained.Copy.Requirements", static_cast<int64_t>(refreshedRequirements.size()));
	BT_PLOT("ORG.RefreshRetained.Copy.InternalTransitions", static_cast<int64_t>(b.params.internalTransitions.size()));
	BT_PLOT("ORG.RefreshRetained.Copy.DeclaredIds", static_cast<int64_t>(b.DeclaredResourceIds().size()));

	{
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Copy)::StoreRequirements");
		p.resources.staticResourceRequirements = std::move(refreshedRequirements);
		p.resources.mergedFrameRequirementsDirty = true;
		p.resources.internalTransitions = std::move(b.params.internalTransitions);
		p.resources.identifierSet = std::move(b._declaredIds);
		p.resources.externalWaitsBeforeTransitions = std::move(b.params.externalWaitsBeforeTransitions);
		p.resources.externalWaitBindingsBeforeTransitions = std::move(b.params.externalWaitBindingsBeforeTransitions);
	}
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh copy pass '{}' materialize referenced resources begin", frameIndex, p.name);
	}
	{
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Copy)::MaterializeReferencedResources");
		MaterializeReferencedResources(p.resources.staticResourceRequirements, p.resources.internalTransitions);
	}

	// Transfer resolver snapshots for auto-invalidation tracking
	{
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Copy)::CaptureResolverSnapshots");
		p.resolverSnapshots = b.TakeResolverSnapshots();
		p.retainedAnonymousKeepAlive = CaptureRetainedAnonymousKeepAlive(
			p.resources.staticResourceRequirements,
			p.resources.internalTransitions);
	}
	{
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Copy)::UpdateRetainedDeclarationCache");
		UpdateRetainedDeclarationCache(PassType::Copy, p.name, p);
	}

	const bool requiresPassRebind = !p.declarationCache.dynamicInterface ||
		p.declarationCache.dynamicInterface->RequiresPassRebindAfterDeclarationRefresh();
	if (requiresPassRebind) {
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Copy)::SetResourceRegistryView");
		p.pass->SetResourceRegistryView(
			std::make_unique<ResourceRegistryView>(_registry, p.resources.identifierSet)
		);
	}
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh copy pass '{}' setup begin", frameIndex, p.name);
	}

	if (requiresPassRebind) {
		BT_ZONE_SCOPE("RenderGraph::RefreshRetainedDeclarationsForFrame(Copy)::Setup");
		p.pass->Setup();
	}
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh copy pass '{}' setup complete", frameIndex, p.name);
	}
	return previousDeclarationFingerprint != p.declarationCache.declarationFingerprint;
}

std::tuple<int, int, int> RenderGraph::GetBatchesToWaitOn(
	std::string_view passName,
	size_t sourceQueueSlot,
	const FramePassSchedulingSummary& passSummary,
	const FrameEpochSet& resourcesTransitionedThisPass)
{
	(void)passName;

	const size_t queueCount = m_queueRegistry.SlotCount();
	const size_t resourceCount = m_frameSchedulingResourceCount;
	if (m_waitCachePassSummary != &passSummary
		|| m_waitCacheTransitionedResources != &resourcesTransitionedThisPass
		|| m_waitCacheTransitionedEpoch != resourcesTransitionedThisPass.epoch
		|| m_waitCacheQueueCount != queueCount) {
		if (m_waitCacheLatestTransitionByQueue.size() < queueCount) {
			m_waitCacheLatestTransitionByQueue.resize(queueCount, -1);
		}
		if (m_waitCacheLatestProducerByQueue.size() < queueCount) {
			m_waitCacheLatestProducerByQueue.resize(queueCount, -1);
		}
		if (m_waitCacheLatestUsageByQueue.size() < queueCount) {
			m_waitCacheLatestUsageByQueue.resize(queueCount, -1);
		}

		for (size_t queueSlot = 0; queueSlot < queueCount; ++queueSlot) {
			m_waitCacheLatestTransitionByQueue[queueSlot] = -1;
			m_waitCacheLatestProducerByQueue[queueSlot] = -1;
			m_waitCacheLatestUsageByQueue[queueSlot] = -1;
		}

		for (size_t queueSlot = 0; queueSlot < queueCount; ++queueSlot) {
			const size_t transitionRowOffset = queueSlot * resourceCount;
			const size_t producerRowOffset = queueSlot * resourceCount;
			const auto* transitionRow = transitionRowOffset < m_frameQueueLastTransitionBatch.size()
				? &m_frameQueueLastTransitionBatch[transitionRowOffset]
				: nullptr;
			const auto* producerRow = producerRowOffset < m_frameQueueLastProducerBatch.size()
				? &m_frameQueueLastProducerBatch[producerRowOffset]
				: nullptr;
			if (!transitionRow && !producerRow) {
				continue;
			}

			for (size_t resourceIndex : passSummary.waitDependencyResourceIndices) {
				if (resourceIndex >= resourceCount) {
					continue;
				}
				if (transitionRow) {
					const int transitionBatch = static_cast<int>(transitionRow[resourceIndex]);
					m_waitCacheLatestTransitionByQueue[queueSlot] = (std::max)(
						m_waitCacheLatestTransitionByQueue[queueSlot],
						transitionBatch);
				}
				if (producerRow) {
					const int producerBatch = static_cast<int>(producerRow[resourceIndex]);
					m_waitCacheLatestProducerByQueue[queueSlot] = (std::max)(
						m_waitCacheLatestProducerByQueue[queueSlot],
						producerBatch);
				}
			}
		}

		const auto processUsageResourceForWaits = [&](size_t resourceIndex) {
			if (resourceIndex >= resourceCount) {
				return;
			}
			for (size_t queueSlot = 0; queueSlot < queueCount; ++queueSlot) {
				const size_t usageRowOffset = queueSlot * resourceCount + resourceIndex;
				if (usageRowOffset >= m_frameQueueLastUsageBatch.size()) {
					continue;
				}
				const int usageBatch = static_cast<int>(m_frameQueueLastUsageBatch[usageRowOffset]);
				m_waitCacheLatestUsageByQueue[queueSlot] = (std::max)(
					m_waitCacheLatestUsageByQueue[queueSlot],
					usageBatch);
			}
		};

		for (size_t resourceIndex : resourcesTransitionedThisPass.Values()) {
			processUsageResourceForWaits(resourceIndex);
			if (resourceIndex >= m_equivalentResourceIndicesByResourceIndex.size()) {
				continue;
			}
			for (size_t equivalentResourceIndex : m_equivalentResourceIndicesByResourceIndex[resourceIndex]) {
				processUsageResourceForWaits(equivalentResourceIndex);
			}
		}

		m_waitCachePassSummary = &passSummary;
		m_waitCacheTransitionedResources = &resourcesTransitionedThisPass;
		m_waitCacheTransitionedEpoch = resourcesTransitionedThisPass.epoch;
		m_waitCacheQueueCount = queueCount;
	}

	if (sourceQueueSlot >= queueCount) {
		return { -1, -1, -1 };
	}

	return {
		m_waitCacheLatestTransitionByQueue[sourceQueueSlot],
		m_waitCacheLatestProducerByQueue[sourceQueueSlot],
		m_waitCacheLatestUsageByQueue[sourceQueueSlot]
	};
}

void RenderGraph::MaterializeUnmaterializedResources(std::span<const uint64_t> onlyResourceIDs) {
	BT_ZONE_SCOPE("RenderGraph::MaterializeUnmaterializedResources");
	const bool limitToResourceIDs = !onlyResourceIDs.empty();
	auto tryGetAliasMaterializeOptions = [&](uint64_t id) -> ResourceMaterializeOptions* {
		auto resourceIndex = TryGetFrameSchedulingResourceIndex(id);
		if (resourceIndex.has_value()
			&& *resourceIndex < m_aliasMaterializeOptionsByResourceIndex.size()
			&& m_aliasMaterializeOptionsByResourceIndex[*resourceIndex].has_value()) {
			return &m_aliasMaterializeOptionsByResourceIndex[*resourceIndex].value();
		}

		auto itAlias = aliasMaterializeOptionsByID.find(id);
		return itAlias != aliasMaterializeOptionsByID.end()
			? &itAlias->second
			: nullptr;
	};

	// Returns the backing generation if the resource was materialized (or already materialized), or nullopt if skipped.
	auto materializeOne = [&](uint64_t id, Resource* resource) -> std::optional<uint64_t> {
		if (!resource) {
			return std::nullopt;
		}
		resource = UnwrapDynamicResource(resource);
		if (!resource) {
			return std::nullopt;
		}

		auto texture = dynamic_cast<PixelBuffer*>(resource);
		if (texture) {
			if (!texture->IsMaterialized()) {
				if (auto* aliasOptions = tryGetAliasMaterializeOptions(id)) {
					if (std::holds_alternative<PixelBuffer::MaterializeOptions>(*aliasOptions)) {
						auto& options = std::get<PixelBuffer::MaterializeOptions>(*aliasOptions);
						texture->Materialize(&options);
					}
				}
				else {
					if (texture->GetDescription().allowAlias) {
						const bool hasManualAliasPool = texture->GetDescription().aliasingPoolID.has_value();
						const bool hasFrameAliasPlacement = TryGetAliasPlacementRange(id) != nullptr;
						const bool aliasPlacementRequiredThisFrame = hasManualAliasPool || hasFrameAliasPlacement;

						if (limitToResourceIDs && aliasPlacementRequiredThisFrame) {
							throw std::runtime_error(
								"Aliasing placement missing for used aliased resource during frame materialization. Resource ID: " + std::to_string(id));
						}

						if (limitToResourceIDs && !aliasPlacementRequiredThisFrame) {
							spdlog::debug(
								"RG alias fallback materialize: id={} name='{}' allowAlias=1 but no pool assignment this frame; materializing standalone",
								id,
								resource->GetName());
						}

						// Setup-time eager materialization happens before alias planning.
						// Defer aliased resources until compile-time placement is available.
						if (!limitToResourceIDs) {
							return std::nullopt;
						}
					}
					texture->Materialize();
				}
			}

			return texture->GetBackingGeneration();
		}

		auto buffer = dynamic_cast<BufferBase*>(resource);
		if (!buffer) {
			return std::nullopt;
		}

		if (!buffer->IsMaterialized()) {
			if (auto* aliasOptions = tryGetAliasMaterializeOptions(id)) {
				if (std::holds_alternative<BufferBase::MaterializeOptions>(*aliasOptions)) {
					auto& options = std::get<BufferBase::MaterializeOptions>(*aliasOptions);
					buffer->Materialize(&options);
				}
			}
			else {
				if (buffer->IsAliasingAllowed()) {
					const bool hasManualAliasPool = buffer->GetAliasingPoolHint().has_value();
					const bool hasFrameAliasPlacement = TryGetAliasPlacementRange(id) != nullptr;
					const bool aliasPlacementRequiredThisFrame = hasManualAliasPool || hasFrameAliasPlacement;

					if (limitToResourceIDs && aliasPlacementRequiredThisFrame) {
						throw std::runtime_error(
							"Aliasing placement missing for used aliased buffer during frame materialization. Resource ID: " + std::to_string(id));
					}

					if (limitToResourceIDs && !aliasPlacementRequiredThisFrame) {
						spdlog::debug(
							"RG alias fallback materialize (buffer): id={} name='{}' allowAlias=1 but no pool assignment this frame; materializing standalone",
							id,
							resource->GetName());
					}

					if (!limitToResourceIDs) {
						return std::nullopt;
					}
				}
				buffer->Materialize();
			}
		}

		return buffer->GetBackingGeneration();
	};

	// Collect all unique {id, resource*} items to materialize.
	auto& items = m_materializeScratchItems;
	items.clear();
	if (limitToResourceIDs) {
		const size_t candidateCapacity = m_frameDAGResourceIDsByIndex.size() + m_aliasMaterializeResourceIDs.size();
		if (items.capacity() < candidateCapacity) {
			items.reserve(candidateCapacity);
		}
	}

	auto skipIfAlreadyMaterialized = [&](uint64_t id, Resource* resource) {
		if (!resource) {
			return false;
		}
		auto* backedResource = TryGetBackedResource(UnwrapDynamicResource(resource));
		if (!backedResource) {
			return true;
		}
		if (!backedResource->IsMaterialized()) {
			return false;
		}
		return true;
	};

	auto resolveMaterializeResourceByID = [&](uint64_t id) -> Resource* {
		if (auto it = resourcesByID.find(id); it != resourcesByID.end() && it->second) {
			return it->second.get();
		}
		if (auto it = m_transientFrameResourcesByID.find(id); it != m_transientFrameResourcesByID.end() && it->second) {
			return it->second.get();
		}
		const size_t count = (std::min)(m_frameDAGResourceIDsByIndex.size(), m_frameDAGResourcePtrByIndex.size());
		for (size_t resourceIndex = 0; resourceIndex < count; ++resourceIndex) {
			if (m_frameDAGResourceIDsByIndex[resourceIndex] == id) {
				return m_frameDAGResourcePtrByIndex[resourceIndex];
			}
		}
		return nullptr;
	};

	auto queueLimitedMaterializeCandidate = [&](uint64_t id, Resource* resource) {
		if (!resource) {
			return;
		}
		if (skipIfAlreadyMaterialized(id, resource)) {
			return;
		}
		for (const auto& item : items) {
			if (item.first == id) {
				return;
			}
		}
		TrackTransientFrameResource(resource);
		items.emplace_back(id, resource);
	};

	if (limitToResourceIDs) {
		for (uint32_t resourceIndex : m_frameDAGUnmaterializedResourceIndices) {
			if (resourceIndex >= m_frameDAGResourceIDsByIndex.size()
				|| resourceIndex >= m_frameDAGResourcePtrByIndex.size()) {
				continue;
			}
			Resource* resource = m_frameDAGResourcePtrByIndex[resourceIndex];
			if (!resource) {
				continue;
			}
			const uint64_t id = m_frameDAGResourceIDsByIndex[resourceIndex];
			queueLimitedMaterializeCandidate(id, resource);
		}
		for (uint64_t resourceID : m_aliasMaterializeResourceIDs) {
			queueLimitedMaterializeCandidate(resourceID, resolveMaterializeResourceByID(resourceID));
		}
	}
	else {
		std::unordered_set<uint64_t> fallbackSeen;
		auto& seen = fallbackSeen;
		items.reserve(resourcesByID.size() + m_transientFrameResourcesByID.size());
		seen.reserve(items.capacity());
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
			if (!fallbackSeen.insert(id).second) {
				return;
			}
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
	}

	// Parallel materialize phase
	m_lastMaterializeCandidateCount = items.size();
	if (items.empty()) {
		m_materializeScratchGenerationResults.clear();
		return;
	}
	auto& genResults = m_materializeScratchGenerationResults;
	genResults.assign(items.size(), MaterializeGenerationResult{});

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

void RenderGraph::MaterializeMultiBackendRepresentations() {
	CollectRetiredInteropGenerations();
	if (m_backendDevices.size() < 2 || m_frameSchedulingResourceCount == 0) return;
	// Scheduling indices are deliberately compacted/merged (dynamic wrappers and
	// their current backings may share one), so they are not interchangeable with
	// DAG resource indices. Build an exact current-frame ID lookup from pass
	// requirements. This also covers pass-owned resources which are already
	// materialized and therefore never entered the transient materialize list.
	std::unordered_map<uint64_t, Resource*> currentResourcesBySchedulingID;
	currentResourcesBySchedulingID.reserve(m_frameSchedulingResourceCount);
	auto collectCurrentResource = [&](const ResourceRegistry::RegistryHandle& handle) {
		Resource* resource = handle.IsEphemeral() ? handle.GetEphemeralPtr() : _registry.Resolve(handle);
		resource = UnwrapDynamicResource(resource);
		if (!resource) return;
		currentResourcesBySchedulingID.try_emplace(resource->GetSchedulingResourceID(), resource);
		currentResourcesBySchedulingID.try_emplace(resource->GetGlobalResourceID(), resource);
	};
	for (const auto& framePass : m_framePasses) {
		std::visit([&](const auto& passAndResources) {
			using PassRecord = std::remove_cvref_t<decltype(passAndResources)>;
			if constexpr (std::is_same_v<PassRecord, std::monostate>) {
				return;
			}
			else {
			ForEachFrameRequirement(passAndResources.resources, [&](const auto& requirement) {
				collectCurrentResource(requirement.resourceHandleAndRange.resource);
			});
			for (const auto& transition : passAndResources.resources.internalTransitions)
				collectCurrentResource(transition.first.resource);
			}
		}, framePass.pass);
	}
	std::vector<std::vector<uint8_t>> uses(
		m_frameSchedulingResourceCount, std::vector<uint8_t>(m_backendDevices.size(), 0));
	for (size_t passIndex = 0; passIndex < m_framePassSchedulingSummaries.size() && passIndex < m_assignedQueueSlotsByFramePass.size(); ++passIndex) {
		const size_t slot = m_assignedQueueSlotsByFramePass[passIndex];
		if (slot >= m_queueRegistry.SlotCount()) continue;
		const auto instance = m_queueRegistry.GetBackendInstance(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot)));
		const size_t instanceIndex = static_cast<uint8_t>(instance);
		if (instanceIndex >= m_backendDevices.size()) continue;
		for (const size_t resourceIndex : m_framePassSchedulingSummaries[passIndex].touchedResourceIndices) {
			if (resourceIndex < uses.size()) uses[resourceIndex][instanceIndex] = 1;
		}
	}
	for (const auto& [resourceID, mask] : m_frameBackendUseMaskByResourceID) {
		const auto index = TryGetFrameSchedulingResourceIndex(resourceID);
		if (!index || *index >= uses.size()) continue;
		for (size_t backend = 0; backend < uses[*index].size(); ++backend)
			uses[*index][backend] |= (mask & (uint64_t{1} << backend)) != 0;
	}
	DeviceRegistryEntry* d3d12 = nullptr;
	DeviceRegistryEntry* vulkan = nullptr;
	for (auto& entry : m_backendDevices) {
		if (entry.backend == rhi::Backend::D3D12) d3d12 = &entry;
		if (entry.backend == rhi::Backend::Vulkan) vulkan = &entry;
	}
	if (!d3d12 || !vulkan) return;
	std::optional<RetiredInteropGeneration> retirement;
	auto ensureRetirement = [&]() -> RetiredInteropGeneration& {
		if (!retirement) retirement.emplace(BeginInteropRetirement());
		return *retirement;
	};

	for (size_t resourceIndex = 0; resourceIndex < uses.size(); ++resourceIndex) {
		size_t backendCount = 0;
		bool nonPrimaryUse = false;
		for (size_t i = 0; i < uses[resourceIndex].size(); ++i) {
			backendCount += uses[resourceIndex][i] != 0;
			nonPrimaryUse |= i != 0 && uses[resourceIndex][i] != 0;
		}
		if (!nonPrimaryUse && backendCount < 2) continue;
		// This function can run during alias planning, before
		// RebuildFrameCompileResources. Never consult that prior-frame pointer
		// cache here; resolve against the current frame's stable IDs/DAG.
		Resource* resource = nullptr;
		if (resourceIndex < m_frameSchedulingResourceIDByIndex.size()) {
			const uint64_t schedulingID = m_frameSchedulingResourceIDByIndex[resourceIndex];
			if (auto currentIt = currentResourcesBySchedulingID.find(schedulingID);
				currentIt != currentResourcesBySchedulingID.end()) {
				resource = currentIt->second;
			}
			else if (auto current = GetResourceByID(schedulingID)) {
				resource = UnwrapDynamicResource(current.get());
			}
		}
		if (!resource) {
			for (const auto& [candidateID, candidateIndex] : m_frameSchedulingResourceIndexEntries) {
				if (candidateIndex != resourceIndex) continue;
				if (auto candidate = GetResourceByID(candidateID)) {
					resource = UnwrapDynamicResource(candidate.get());
					break;
				}
			}
		}
		if (!resource && resourceIndex < m_frameSchedulingResourceIDByIndex.size()) {
			if (auto shared = GetResourceByID(m_frameSchedulingResourceIDByIndex[resourceIndex])) {
				resource = UnwrapDynamicResource(shared.get());
			}
		}
		if (!resource) continue;
		if (!resource->IsRenderGraphManaged()) {
			throw std::runtime_error("Multi-RHI resource '" + resource->GetName() + "' is externally managed");
		}
		const uint64_t schedulingID = resourceIndex < m_frameSchedulingResourceIDByIndex.size()
			? m_frameSchedulingResourceIDByIndex[resourceIndex] : resource->GetSchedulingResourceID();
		const auto generationPlacement = aliasPlacementRangesByID.find(schedulingID);
		const auto generationPool = generationPlacement == aliasPlacementRangesByID.end()
			? persistentAliasPools.end() : persistentAliasPools.find(generationPlacement->second.poolID);
		const bool staleSharedPoolGeneration = generationPool != persistentAliasPools.end() &&
			generationPool->second.multiBackendShared &&
			(!m_sharedAliasResourcePoolGeneration.contains(schedulingID) ||
			 m_sharedAliasResourcePoolGeneration.at(schedulingID) != generationPool->second.generation);
		bool complete = true;
		for (size_t i = 0; i < uses[resourceIndex].size(); ++i) {
			if (uses[resourceIndex][i] && !resource->HasAPIRepresentation(static_cast<BackendInstanceId>(static_cast<uint8_t>(i)))) {
				complete = false;
			}
		}
		if (staleSharedPoolGeneration) complete = false;
		if (complete) continue;
		if (staleSharedPoolGeneration &&
			(resource->HasAPIRepresentation(d3d12->id) || resource->HasAPIRepresentation(vulkan->id))) {
			auto old = resource->TakeAPIRepresentations();
			auto& retired = ensureRetirement().representations;
			retired.insert(retired.end(), std::make_move_iterator(old.begin()), std::make_move_iterator(old.end()));
		}

		rhi::ResourceDesc desc{};
		if (!resource->TryGetRHIResourceDesc(desc) || desc.heapType != rhi::HeapType::DeviceLocal ||
			(desc.type != rhi::ResourceType::Buffer && desc.type != rhi::ResourceType::Texture2D)) {
			throw std::runtime_error("Multi-RHI resource '" + resource->GetName() + "' has no supported device-local buffer/2D texture representation");
		}
		desc.heapFlags |= rhi::HeapFlags::Shared;
		spdlog::debug("RenderGraph materializing multi-RHI representation: index={} id={} name='{}' type={} flags=0x{:X}",
			resourceIndex, resource->GetGlobalResourceID(), resource->GetName(), static_cast<uint32_t>(desc.type),
			static_cast<uint32_t>(desc.resourceFlags));
		rhi::ResourcePtr d3Resource;
		rhi::ResourcePtr vkResource;
		if (desc.type == rhi::ResourceType::Texture2D &&
			!rhi::vulkan::query_d3d12_texture_support(vulkan->device, desc).supported) {
			throw std::runtime_error("Multi-RHI texture '" + resource->GetName() + "' is unsupported for its exact format/usage");
		}
		bool placedOnSharedHeap = false;
		// D3D12 shared heaps may contain buffers and RT/DS-class textures. Ordinary
		// non-RT textures must use committed D3D12_RESOURCE sharing.
		const bool sharedHeapCandidate = desc.type == rhi::ResourceType::Buffer ||
			(desc.type == rhi::ResourceType::Texture2D &&
				(desc.resourceFlags & rhi::ResourceFlags::RF_AllowRenderTarget) != 0 &&
				(desc.resourceFlags & rhi::ResourceFlags::RF_AllowDepthStencil) == 0);
		if (const auto placementIt = aliasPlacementRangesByID.find(schedulingID);
			sharedHeapCandidate && placementIt != aliasPlacementRangesByID.end()) {
			const auto& placement = placementIt->second;
			const auto primaryPool = persistentAliasPools.find(placement.poolID);
			if (primaryPool != persistentAliasPools.end()) {
				const uint8_t resourceClass = desc.type == rhi::ResourceType::Buffer ? 1u : 2u;
				rhi::ResourceAllocationInfo d3Requirements{}, vkRequirements{};
				d3d12->device.GetResourceAllocationInfo(&desc, 1, &d3Requirements);
				vulkan->device.GetResourceAllocationInfo(&desc, 1, &vkRequirements);
				const uint64_t requiredAlignment = (std::max)(d3Requirements.alignment, vkRequirements.alignment);
				const uint64_t requiredSize = (std::max)(d3Requirements.sizeInBytes, vkRequirements.sizeInBytes);
				const bool placementCompatible = requiredAlignment != 0 &&
					(placement.startByte % requiredAlignment) == 0 &&
					requiredSize <= placement.endByte - placement.startByte;
				if (!placementCompatible) {
					spdlog::warn("RenderGraph shared alias requirements rejected; using committed fallback: pool={} resource={} offset={} reserved={} requiredSize={} requiredAlignment={}",
						placement.poolID, schedulingID, placement.startByte,
						placement.endByte - placement.startByte, requiredSize, requiredAlignment);
					goto committed_multi_backend_resource;
				}
				auto* sharedPool = m_sharedAliasPools.Find(placement.poolID);
				const bool replacePool = !sharedPool || !sharedPool->d3d12 || !sharedPool->vulkan ||
					sharedPool->generation != primaryPool->second.generation ||
					sharedPool->capacityBytes < primaryPool->second.capacityBytes ||
					sharedPool->resourceClass != resourceClass;
				if (replacePool) {
					// Destroy every placed object bound to the old imported memory before
					// releasing that VkDeviceMemory/D3D12 heap pair.
					for (auto it = m_sharedAliasResourcePoolID.begin(); it != m_sharedAliasResourcePoolID.end();) {
						if (it->second != placement.poolID) { ++it; continue; }
						Resource* oldResource = nullptr;
						if (auto current = currentResourcesBySchedulingID.find(it->first); current != currentResourcesBySchedulingID.end())
							oldResource = current->second;
						else if (auto registered = GetResourceByID(it->first))
							oldResource = UnwrapDynamicResource(registered.get());
						if (oldResource) {
							auto old = oldResource->TakeAPIRepresentations();
							auto& retired = ensureRetirement().representations;
							retired.insert(retired.end(), std::make_move_iterator(old.begin()), std::make_move_iterator(old.end()));
						}
						m_sharedAliasResourcePoolGeneration.erase(it->first);
						it = m_sharedAliasResourcePoolID.erase(it);
					}
					auto& retiredHeaps = ensureRetirement().heaps;
					const auto heapResult = m_sharedAliasPools.EnsureD3D12VulkanPool(
						placement.poolID, primaryPool->second.generation, primaryPool->second.capacityBytes,
						(std::max)(primaryPool->second.alignment, requiredAlignment), resourceClass,
						*d3d12, *vulkan, retiredHeaps);
					if (rhi::IsOk(heapResult)) {
						sharedPool = m_sharedAliasPools.Find(placement.poolID);
						spdlog::debug("RenderGraph created shared alias pool: pool={} capacity={} alignment={} class={} generation={}",
							placement.poolID, primaryPool->second.capacityBytes,
							(std::max)(primaryPool->second.alignment, requiredAlignment), resourceClass,
							primaryPool->second.generation);
					}
					else {
						sharedPool = nullptr;
						spdlog::warn("RenderGraph shared alias pool creation failed; using committed fallback: pool={} result={}",
							placement.poolID, static_cast<uint32_t>(heapResult));
					}
				}
				if (sharedPool && sharedPool->d3d12 && sharedPool->vulkan && sharedPool->resourceClass == resourceClass &&
					placement.endByte <= sharedPool->capacityBytes) {
					rhi::ResourceDesc placedDesc = desc;
					placedDesc.heapFlags = rhi::HeapFlags::None;
					auto d3Placed = d3d12->device.CreatePlacedResource(sharedPool->d3d12->GetHandle(), placement.startByte, placedDesc, d3Resource);
					auto vkPlaced = rhi::IsOk(d3Placed)
						? vulkan->device.CreatePlacedResource(sharedPool->vulkan->GetHandle(), placement.startByte, placedDesc, vkResource)
						: d3Placed;
					placedOnSharedHeap = rhi::IsOk(d3Placed) && rhi::IsOk(vkPlaced);
					if (placedOnSharedHeap && desc.type == rhi::ResourceType::Texture2D) {
						// D3D12 requires every placed RT/DS resource to be initialized by
						// D3D12 before use, even when Vulkan will be its first producer.
						// A one-time enhanced discard transition initializes all
						// subresources without allocating or clearing a second backing.
						rhi::CommandAllocatorPtr allocator;
						rhi::CommandListPtr list;
						rhi::DescriptorHeapPtr initializationRtvHeap;
						auto initResult = d3d12->device.CreateCommandAllocator(rhi::QueueKind::Graphics, allocator);
						if (rhi::IsOk(initResult)) initResult = d3d12->device.CreateCommandList(rhi::QueueKind::Graphics, allocator.Get(), list);
						if (rhi::IsOk(initResult)) {
							rhi::DescriptorHeapDesc heapDesc{};
							heapDesc.type = rhi::DescriptorHeapType::RTV;
							heapDesc.capacity = 1;
							heapDesc.debugName = "RenderGraph shared texture initialization RTV";
							initResult = d3d12->device.CreateDescriptorHeap(heapDesc, initializationRtvHeap);
						}
						if (rhi::IsOk(initResult)) {
							initResult = d3d12->device.CreateRenderTargetView(
								{ initializationRtvHeap->GetHandle(), 0 }, d3Resource->GetHandle(), {});
						}
						if (rhi::IsOk(initResult)) {
							rhi::TextureBarrier initialize{
								.texture = d3Resource->GetHandle(),
								.range = {
									.baseMip = 0,
									.mipCount = desc.texture.mipLevels,
									.baseLayer = 0,
									.layerCount = desc.texture.depthOrLayers,
									.basePlane = 0,
									.planeCount = 1,
								},
								.beforeSync = rhi::ResourceSyncState::None,
								.afterSync = rhi::ResourceSyncState::All,
								.beforeAccess = rhi::ResourceAccessType::None,
								.afterAccess = rhi::ResourceAccessType::RenderTarget,
								.beforeLayout = rhi::ResourceLayout::Undefined,
								.afterLayout = rhi::ResourceLayout::RenderTarget,
								.discard = true,
							};
							list->Barriers(rhi::BarrierBatch{ .textures = { &initialize, 1 } });
							rhi::ColorAttachment attachment{};
							attachment.rtv = { initializationRtvHeap->GetHandle(), 0 };
							attachment.loadOp = rhi::LoadOp::DontCare;
							attachment.resource = d3Resource->GetHandle();
							attachment.mipSlice = -1;
							rhi::PassBeginInfo begin{};
							begin.colors = { &attachment, 1 };
							begin.width = desc.texture.width;
							begin.height = desc.texture.height;
							list->BeginPass(begin);
							list->EndPass();
							rhi::TextureBarrier toCommon = initialize;
							toCommon.beforeSync = rhi::ResourceSyncState::RenderTarget;
							toCommon.afterSync = rhi::ResourceSyncState::All;
							toCommon.beforeAccess = rhi::ResourceAccessType::RenderTarget;
							toCommon.afterAccess = rhi::ResourceAccessType::Common;
							toCommon.beforeLayout = rhi::ResourceLayout::RenderTarget;
							toCommon.afterLayout = rhi::ResourceLayout::Common;
							toCommon.discard = false;
							list->Barriers(rhi::BarrierBatch{ .textures = { &toCommon, 1 } });
							list->End();
							const rhi::CommandList submitted[] = { list.Get() };
							auto queue = d3d12->device.GetQueue(rhi::QueueKind::Graphics);
							rhi::TimelinePtr completion;
							if (rhi::IsOk(initResult))
								initResult = d3d12->device.CreateTimeline(completion, 0, "RenderGraph shared texture initialization");
							const rhi::TimelinePoint completed{ completion ? completion->GetHandle() : rhi::TimelineHandle{}, 1 };
							if (rhi::IsOk(initResult)) initResult = queue.Submit(submitted, { .signals = { &completed, 1 } });
							if (rhi::IsOk(initResult)) initResult = completion->HostWait(1, 30000);
						}
						if (rhi::Failed(initResult)) {
							d3Resource.Reset(); vkResource.Reset();
							placedOnSharedHeap = false;
							spdlog::warn("RenderGraph shared alias texture initialization failed; using committed fallback: pool={} resource={}",
								placement.poolID, schedulingID);
						}
					}
					if (!placedOnSharedHeap) {
						d3Resource.Reset(); vkResource.Reset();
						spdlog::warn("RenderGraph shared alias placement rejected; using committed fallback: pool={} resource={} offset={}",
							placement.poolID, schedulingID, placement.startByte);
					}
				}
			}
		}

	committed_multi_backend_resource:
		auto result = rhi::Result::Ok;
		if (!placedOnSharedHeap) {
			InteropResourcePair pair;
			result = InteropAllocator::CreateCommittedD3D12Vulkan(*d3d12, *vulkan, desc, pair);
			d3Resource = std::move(pair.canonical);
			vkResource = std::move(pair.imported);
		}
		if (rhi::Failed(result)) throw std::runtime_error("Failed to import Vulkan representation for '" + resource->GetName() + "'");
		const ResourceState commonInitial{
			rhi::ResourceAccessType::Common, rhi::ResourceLayout::Common, rhi::ResourceSyncState::All };
		const ResourceState d3dInitial = desc.type == rhi::ResourceType::Texture2D && !placedOnSharedHeap
			? ResourceState{ rhi::ResourceAccessType::None, rhi::ResourceLayout::Undefined, rhi::ResourceSyncState::None }
			: commonInitial;
		const ResourceState vulkanInitial = desc.type == rhi::ResourceType::Texture2D
			? ResourceState{ rhi::ResourceAccessType::None, rhi::ResourceLayout::Undefined, rhi::ResourceSyncState::None }
			: commonInitial;
		if (placedOnSharedHeap) {
			if (auto* texture = dynamic_cast<PixelBuffer*>(resource); texture && texture->IsMaterialized()) texture->Dematerialize();
			else if (auto* buffer = dynamic_cast<BufferBase*>(resource); buffer && buffer->IsMaterialized()) buffer->Dematerialize();
		}
		// Enhanced-barrier textures are created in the description's UNDEFINED
		// layout on both APIs; buffers use the bridge-compatible COMMON state.
		std::vector<Resource::PendingAPIRepresentation> representations;
		representations.push_back({ d3d12->id, std::move(d3Resource), d3dInitial });
		representations.push_back({ vulkan->id, std::move(vkResource), vulkanInitial });
		if (!resource->PublishAPIRepresentations(std::move(representations)))
			throw std::runtime_error("Failed to publish multi-RHI representations for '" + resource->GetName() + "'");
		if (placedOnSharedHeap) {
			const auto placement = aliasPlacementRangesByID.find(schedulingID);
			if (placement != aliasPlacementRangesByID.end())
			{
				const auto* pool = m_sharedAliasPools.Find(placement->second.poolID);
				if (!pool) throw std::runtime_error("Shared alias pool disappeared during representation publication");
				m_sharedAliasResourcePoolGeneration[schedulingID] = pool->generation;
				m_sharedAliasResourcePoolID[schedulingID] = placement->second.poolID;
			}
		}
		else {
			m_sharedAliasResourcePoolGeneration.erase(schedulingID);
			m_sharedAliasResourcePoolID.erase(schedulingID);
		}
		resource->RefreshAPIRepresentationDescriptors(d3d12->id);
		resource->RefreshAPIRepresentationDescriptors(vulkan->id);
		spdlog::debug("RenderGraph materialized multi-RHI representations: id={} name='{}' type={} sharedAlias={} D3D12Instance={} VulkanInstance={}",
			resource->GetGlobalResourceID(), resource->GetName(), static_cast<uint32_t>(desc.type),
			placedOnSharedHeap,
			static_cast<uint32_t>(d3d12->id), static_cast<uint32_t>(vulkan->id));
	}
	if (retirement && (!retirement->representations.empty() || !retirement->heaps.empty()))
		m_retiredInteropGenerations.push_back(std::move(*retirement));
}

void RenderGraph::PlanMultiBackendOwnershipTransfers() {
	BackendInstanceId canonicalD3D12 = BackendInstanceId::Primary;
	for (const auto& device : m_backendDevices) {
		if (device.backend == rhi::Backend::D3D12) canonicalD3D12 = device.id;
	}
	struct LastUse {
		BackendInstanceId backend = BackendInstanceId::Primary;
		size_t batchIndex = 0;
		size_t queueIndex = 0;
		std::vector<ExternalOwnershipBarrier>* releases = nullptr;
		SymbolicTracker tracker{};
	};
	std::unordered_map<Resource*, LastUse> lastUses;
	std::unordered_map<Resource*, std::unordered_map<uint8_t, SymbolicTracker>> representationTrackers;
	auto backingGeneration = [](Resource* resource) -> uint64_t {
		if (auto* backed = dynamic_cast<BackedResource*>(resource)) return backed->GetBackingGeneration();
		return 0;
	};
	auto getRepresentationTracker = [&](Resource* resource, BackendInstanceId backend) -> SymbolicTracker& {
		auto& byBackend = representationTrackers[resource];
		auto [it, inserted] = byBackend.try_emplace(static_cast<uint8_t>(backend));
		if (inserted) {
			if (auto* tracker = resource->GetStateTracker(backend)) it->second.CopyFrom(*tracker);
		}
		return it->second;
	};

	auto visitPass = [&](auto* passEntry, BackendInstanceId backend, size_t batchIndex, size_t queueIndex) {
		passEntry->backendPreTransitions.clear();
		passEntry->backendPostTransitions.clear();
		passEntry->externalAcquires.clear();
		passEntry->externalReleases.clear();
		ForEachFrameRequirement(passEntry->resources, [&](const auto& requirement) {
			const auto handle = requirement.resourceHandleAndRange.resource;
			Resource* resource = handle.IsEphemeral() ? handle.GetEphemeralPtr() : _registry.Resolve(handle);
			resource = UnwrapDynamicResource(resource);
			bool hasMultipleRepresentations = false;
			if (resource) {
				size_t representationCount = 0;
				for (const auto& device : m_backendDevices) representationCount += resource->HasAPIRepresentation(device.id) ? 1u : 0u;
				hasMultipleRepresentations = representationCount > 1;
			}
			if (!resource || !hasMultipleRepresentations) {
				return;
			}
			auto previous = lastUses.find(resource);
			if (previous == lastUses.end()) {
				const auto id = resource->GetGlobalResourceID();
				const auto generation = backingGeneration(resource);
				auto persisted = m_logicalExternalOwners.find(id);
				if (persisted != m_logicalExternalOwners.end()
					&& persisted->second.backingGeneration == generation) {
					LastUse restored{};
					restored.backend = persisted->second.backend;
					restored.batchIndex = batchIndex;
					restored.queueIndex = queueIndex;
					if (auto* tracker = resource->GetStateTracker(restored.backend)) restored.tracker.CopyFrom(*tracker);
					previous = lastUses.emplace(resource, std::move(restored)).first;
				}
				else if (persisted != m_logicalExternalOwners.end()) {
					m_logicalExternalOwners.erase(persisted);
				}
			}
			if (previous == lastUses.end() && backend != canonicalD3D12) {
				// Acquire every destination segment from its real local state. A
				// representation that has never been used is still UNDEFINED; one
				// used before a prior release is COMMON.
				auto& destinationTracker = getRepresentationTracker(resource, backend);
				for (const auto& segment : destinationTracker.GetSegments()) {
					passEntry->externalAcquires.push_back({
						resource, segment.state, segment.rangeSpec,
						segment.state.layout == rhi::ResourceLayout::Undefined });
				}
				LastUse initial{};
				initial.backend = backend;
				initial.batchIndex = batchIndex;
				initial.queueIndex = queueIndex;
				initial.releases = &passEntry->externalReleases;
				initial.tracker.CopyFrom(destinationTracker);
				previous = lastUses.emplace(resource, std::move(initial)).first;
				destinationTracker.Reset({}, { rhi::ResourceAccessType::Common, rhi::ResourceLayout::Common, rhi::ResourceSyncState::All });
				spdlog::debug("RenderGraph planned initial external texture acquire: resource='{}' toBackend={} consumer='{}'",
					resource->GetName(), static_cast<uint32_t>(backend), passEntry->name);
			}
			else if (previous == lastUses.end() && backend == canonicalD3D12) {
				// The canonical representation owns the allocation initially; its normal
				// per-device transition below initializes the requested subresources.
				LastUse initial{};
				initial.backend = backend;
				initial.batchIndex = batchIndex;
				initial.queueIndex = queueIndex;
				initial.releases = &passEntry->externalReleases;
				if (auto* tracker = resource->GetStateTracker(backend)) initial.tracker.CopyFrom(*tracker);
				previous = lastUses.emplace(resource, std::move(initial)).first;
			}
			else if (previous != lastUses.end() && previous->second.backend != backend) {
				const BackendInstanceId sourceBackend = previous->second.backend;
				const size_t sourceBatchIndex = previous->second.batchIndex;
				const size_t sourceQueueIndex = previous->second.queueIndex;
				if (previous->second.releases) {
					for (const auto& segment : previous->second.tracker.GetSegments()) {
						previous->second.releases->push_back({ resource, segment.state, segment.rangeSpec, false });
					}
				}
				auto& destinationTracker = getRepresentationTracker(resource, backend);
				for (const auto& segment : destinationTracker.GetSegments()) {
					passEntry->externalAcquires.push_back({
						resource, segment.state, segment.rangeSpec,
						segment.state.layout == rhi::ResourceLayout::Undefined });
				}
				getRepresentationTracker(resource, sourceBackend).Reset({}, { rhi::ResourceAccessType::Common, rhi::ResourceLayout::Common, rhi::ResourceSyncState::All });
				destinationTracker.Reset({}, { rhi::ResourceAccessType::Common, rhi::ResourceLayout::Common, rhi::ResourceSyncState::All });
				previous->second.backend = backend;
				// Ownership is ordered by the completion of the actual source pass,
				// rather than by the legacy logical-transition slot. The latter may
				// become signal-only after its cross-device barriers are removed.
				if (sourceBatchIndex > 0 && sourceQueueIndex != queueIndex) {
					auto& sourceBatch = batches[sourceBatchIndex];
					sourceBatch.MarkQueueSignal(BatchSignalPhase::AfterCompletion, sourceQueueIndex);
					batches[batchIndex].AddQueueWait(
						BatchWaitPhase::BeforeTransitions,
						queueIndex,
						sourceQueueIndex,
						sourceBatch.GetQueueSignalFenceValue(BatchSignalPhase::AfterCompletion, sourceQueueIndex));
				}
				previous->second.tracker.Reset({}, { rhi::ResourceAccessType::Common, rhi::ResourceLayout::Common, rhi::ResourceSyncState::All });
				spdlog::debug("RenderGraph planned external texture ownership transfer: resource='{}' fromBackend={} toBackend={} consumer='{}'",
					resource->GetName(), static_cast<uint32_t>(sourceBackend), static_cast<uint32_t>(backend), passEntry->name);
			}
			if (previous == lastUses.end()) {
				LastUse initial{};
				initial.backend = backend;
				initial.batchIndex = batchIndex;
				initial.queueIndex = queueIndex;
				initial.releases = &passEntry->externalReleases;
				if (auto* tracker = resource->GetStateTracker(backend)) initial.tracker.CopyFrom(*tracker);
				previous = lastUses.emplace(resource, std::move(initial)).first;
			}
			previous->second.backend = backend;
			previous->second.batchIndex = batchIndex;
			previous->second.queueIndex = queueIndex;
			previous->second.releases = &passEntry->externalReleases;
			auto& representationTracker = getRepresentationTracker(resource, backend);
			representationTracker.Apply(requirement.resourceHandleAndRange.range, resource,
				requirement.state, passEntry->backendPreTransitions);
			previous->second.tracker.CopyFrom(representationTracker);
		});
		for (const auto& [handleAndRange, targetState] : passEntry->resources.internalTransitions) {
			Resource* resource = handleAndRange.resource.IsEphemeral()
				? handleAndRange.resource.GetEphemeralPtr() : _registry.Resolve(handleAndRange.resource);
			resource = UnwrapDynamicResource(resource);
			if (!resource) continue;
			size_t representationCount = 0;
			for (const auto& device : m_backendDevices) representationCount += resource->HasAPIRepresentation(device.id) ? 1u : 0u;
			if (representationCount < 2) continue;
			auto& representationTracker = getRepresentationTracker(resource, backend);
			representationTracker.Apply(handleAndRange.range, resource, targetState, passEntry->backendPostTransitions);
			if (auto previous = lastUses.find(resource); previous != lastUses.end()) previous->second.tracker.CopyFrom(representationTracker);
		}
	};

	for (size_t batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
		auto& batch = batches[batchIndex];
		for (size_t queueIndex = 0; queueIndex < m_queueRegistry.SlotCount(); ++queueIndex) {
			const auto backend = m_queueRegistry.GetBackendInstance(static_cast<QueueSlotIndex>(static_cast<uint8_t>(queueIndex)));
			for (auto& passVariant : batch.Passes(queueIndex)) {
				std::visit([&](auto* passEntry) { visitPass(passEntry, backend, batchIndex, queueIndex); }, passVariant);
			}
		}
	}

	for (const auto& [resource, lastUse] : lastUses) {
		m_logicalExternalOwners[resource->GetGlobalResourceID()] = {
			backingGeneration(resource), lastUse.backend
		};
	}

	// Transition compilation historically followed one logical tracker. Multi-
	// representation resources now use the per-pass, per-backend transitions
	// built above, so remove every legacy batch transition for those resources.
	for (auto& batch : batches) {
		for (size_t queueIndex = 0; queueIndex < m_queueRegistry.SlotCount(); ++queueIndex) {
			for (const auto phase : { BatchTransitionPhase::BeforePasses, BatchTransitionPhase::AfterPasses }) {
				auto& transitions = batch.Transitions(queueIndex, phase);
				std::erase_if(transitions, [&](const ResourceTransition& transition) {
					auto* resource = UnwrapDynamicResource(transition.pResource);
					if (!resource) return false;
					size_t representationCount = 0;
					for (const auto& device : m_backendDevices) representationCount += resource->HasAPIRepresentation(device.id) ? 1u : 0u;
					return representationCount > 1;
				});
			}
		}
	}
	// Publish the independently compiled end state of every API representation.
	// Subsequent frame compiles must begin from these states, not from the
	// temporary COMMON state used at an ownership boundary.
	for (auto& [resource, byBackend] : representationTrackers) {
		for (auto& [backendValue, compiledTracker] : byBackend) {
			if (auto* tracker = resource->GetStateTracker(static_cast<BackendInstanceId>(backendValue))) {
				tracker->CopyFrom(compiledTracker);
			}
		}
	}

	// Removing legacy cross-device transitions can leave synchronization on a
	// queue slot that no longer contains any command-list work. Such a slot is
	// deliberately inactive at execution time, so its reserved fence value can
	// never be signaled. Remove waits on those orphan signals; the explicit
	// ownership edges above point at real source passes instead.
	for (size_t sourceBatchIndex = 0; sourceBatchIndex < batches.size(); ++sourceBatchIndex) {
		auto& sourceBatch = batches[sourceBatchIndex];
		for (size_t sourceQueueIndex = 0; sourceQueueIndex < sourceBatch.QueueCount(); ++sourceQueueIndex) {
			const bool hasWork = sourceBatch.HasTransitions(sourceQueueIndex, BatchTransitionPhase::BeforePasses)
				|| sourceBatch.HasPasses(sourceQueueIndex)
				|| sourceBatch.HasTransitions(sourceQueueIndex, BatchTransitionPhase::AfterPasses);
			if (hasWork) continue;
			for (size_t signalPhaseIndex = 0; signalPhaseIndex < PassBatch::kSignalPhaseCount; ++signalPhaseIndex) {
				const auto signalPhase = static_cast<BatchSignalPhase>(signalPhaseIndex);
				if (!sourceBatch.HasQueueSignal(signalPhase, sourceQueueIndex)) continue;
				const auto orphanValue = sourceBatch.GetQueueSignalFenceValue(signalPhase, sourceQueueIndex);
				for (auto& consumerBatch : batches) {
					for (size_t waitPhaseIndex = 0; waitPhaseIndex < PassBatch::kWaitPhaseCount; ++waitPhaseIndex) {
						const auto waitPhase = static_cast<BatchWaitPhase>(waitPhaseIndex);
						for (size_t destinationQueueIndex = 0; destinationQueueIndex < consumerBatch.QueueCount(); ++destinationQueueIndex) {
							if (consumerBatch.HasQueueWait(waitPhase, destinationQueueIndex, sourceQueueIndex)
								&& consumerBatch.GetQueueWaitFenceValue(waitPhase, destinationQueueIndex, sourceQueueIndex) == orphanValue) {
								consumerBatch.ClearQueueWait(waitPhase, destinationQueueIndex, sourceQueueIndex);
							}
						}
					}
				}
				sourceBatch.ClearQueueSignal(signalPhase, sourceQueueIndex);
			}
		}
	}
}

RenderGraph::RetiredInteropGeneration RenderGraph::BeginInteropRetirement() {
	RetiredInteropGeneration generation;
	generation.queueCompletionValues.reserve(m_queueRegistry.SlotCount());
	for (size_t slot = 0; slot < m_queueRegistry.SlotCount(); ++slot) {
		const auto nextValue = m_queueRegistry.GetCurrentFenceValue(
			static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot)));
		generation.queueCompletionValues.push_back(nextValue > 0 ? nextValue - 1 : 0);
	}
	return generation;
}

void RenderGraph::CollectRetiredInteropGenerations() {
	auto completed = [&](const RetiredInteropGeneration& generation) {
		if (generation.queueCompletionValues.size() > m_queueRegistry.SlotCount()) return false;
		for (size_t slot = 0; slot < generation.queueCompletionValues.size(); ++slot) {
			if (m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot))).GetCompletedValue() <
				generation.queueCompletionValues[slot]) return false;
		}
		return true;
	};
	std::erase_if(m_retiredInteropGenerations, completed);
}

void RenderGraph::ResizeQueueParallelVectors() {
	const size_t qc = m_queueRegistry.SlotCount();
	m_compilerState->compiledLastProducerBatchByResourceByQueue.resize(qc);
	m_compilerState->compiledLastAccessBatchByResourceByQueue.resize(qc);
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

bool RenderGraph::RetainedDeclarationMayNeedRefresh(const AnyPassAndResources& pass)
{
	return std::visit(
		[](const auto& passAndResources) -> bool {
			using T = std::decay_t<decltype(passAndResources)>;
			if constexpr (std::is_same_v<T, std::monostate>) {
				return false;
			}
			else {
				const auto& cache = passAndResources.declarationCache;
				return cache.dynamicInterface != nullptr
					|| !passAndResources.resolverSnapshots.empty()
					|| cache.requiresStaleHandleValidation;
			}
		},
		pass.pass);
}

void RenderGraph::RebuildRetainedDeclarationRefreshCandidates()
{
	BT_ZONE_SCOPE("RenderGraph::RebuildRetainedDeclarationRefreshCandidates");
	m_retainedDeclarationRefreshCandidateMasterIndices.clear();
	m_retainedDeclarationRefreshCandidateMasterIndices.reserve(m_masterPassList.size());
	for (size_t passIndex = 0; passIndex < m_masterPassList.size(); ++passIndex) {
		if (RetainedDeclarationMayNeedRefresh(m_masterPassList[passIndex])) {
			m_retainedDeclarationRefreshCandidateMasterIndices.push_back(passIndex);
		}
	}
}

void RenderGraph::Setup() {
	DeletionManager::GetInstance().Initialize();

	// Setup the statistics manager
	if (m_statisticsService) {
		m_statisticsService->ClearAll();
		m_statisticsService->Initialize();
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
		const auto primaryBackend = m_backendDevices.empty() ? rhi::Backend::Null : m_backendDevices.front().backend;
		m_queueRegistry.Register({ QueueKind::Graphics, 0, BackendInstanceId::Primary, primaryBackend }, gfxQ, device);
		m_queueRegistry.Register({ QueueKind::Compute, 0, BackendInstanceId::Primary, primaryBackend }, compQ, device);
		m_queueRegistry.Register({ QueueKind::Copy,    0, BackendInstanceId::Primary, primaryBackend }, copyQ, device);
		for (size_t i = 1; i < m_backendDevices.size(); ++i) {
			const auto& peer = m_backendDevices[i];
			auto peerDevice = peer.device;
			m_queueRegistry.Register({ QueueKind::Graphics, 0, peer.id, peer.backend }, peerDevice.GetQueue(rhi::QueueKind::Graphics), peerDevice);
			m_queueRegistry.Register({ QueueKind::Compute, 0, peer.id, peer.backend }, peerDevice.GetQueue(rhi::QueueKind::Compute), peerDevice);
			m_queueRegistry.Register({ QueueKind::Copy, 0, peer.id, peer.backend }, peerDevice.GetQueue(rhi::QueueKind::Copy), peerDevice);
		}
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

	// Queue extensions and the automatic scheduler can add slots after the
	// primary queues. Every slot must receive a destination-local imported
	// timeline before any cross-device dependency is compiled.
	if (m_backendDevices.size() > 1) {
		rhi::Device d3d12Device{};
		rhi::Device vulkanDevice{};
		for (const auto& entry : m_backendDevices) {
			if (entry.backend == rhi::Backend::D3D12) d3d12Device = entry.device;
			if (entry.backend == rhi::Backend::Vulkan) vulkanDevice = entry.device;
		}
		const auto interopResult = m_queueRegistry.EnableD3D12VulkanInterop(d3d12Device, vulkanDevice);
		if (rhi::Failed(interopResult)) {
			throw std::runtime_error(std::string("RenderGraph failed to initialize multi-RHI queue timelines: ") + rhi::ResultName(interopResult));
		}
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
	m_getQueueSchedulingSelectionPolicy = [this]() {
		return m_renderGraphSettingsService
			? m_renderGraphSettingsService->GetQueueSchedulingSelectionPolicy()
			: org::runtime::QueueSchedulingSelectionPolicy::FirstFit;
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
	m_getTransitionPlacementMode = [this]() {
		return m_renderGraphSettingsService
			? m_renderGraphSettingsService->GetTransitionPlacementMode()
			: org::runtime::TransitionPlacementMode::InlineEarlyPlacement;
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
	UpdateRetainedDeclarationCache(PassType::Render, passAndResources.name, passAndResources);
	AnyPassAndResources passAndResourcesAny;
	passAndResourcesAny.type = PassType::Render;
	passAndResourcesAny.pass = std::move(passAndResources);
	passAndResourcesAny.name = name;
	const bool mayNeedRefresh = RetainedDeclarationMayNeedRefresh(passAndResourcesAny);
	m_masterPassList.push_back(std::move(passAndResourcesAny));
	if (mayNeedRefresh) {
		m_retainedDeclarationRefreshCandidateMasterIndices.push_back(m_masterPassList.size() - 1);
	}
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
	UpdateRetainedDeclarationCache(PassType::Compute, passAndResources.name, passAndResources);
	AnyPassAndResources passAndResourcesAny;
	passAndResourcesAny.type = PassType::Compute;
	passAndResourcesAny.pass = std::move(passAndResources);
	passAndResourcesAny.name = name;
	const bool mayNeedRefresh = RetainedDeclarationMayNeedRefresh(passAndResourcesAny);
	m_masterPassList.push_back(std::move(passAndResourcesAny));
	if (mayNeedRefresh) {
		m_retainedDeclarationRefreshCandidateMasterIndices.push_back(m_masterPassList.size() - 1);
	}
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
	UpdateRetainedDeclarationCache(PassType::Copy, passAndResources.name, passAndResources);
	AnyPassAndResources passAndResourcesAny;
	passAndResourcesAny.type = PassType::Copy;
	passAndResourcesAny.pass = std::move(passAndResources);
	passAndResourcesAny.name = name;
	const bool mayNeedRefresh = RetainedDeclarationMayNeedRefresh(passAndResourcesAny);
	m_masterPassList.push_back(std::move(passAndResourcesAny));
	if (mayNeedRefresh) {
		m_retainedDeclarationRefreshCandidateMasterIndices.push_back(m_masterPassList.size() - 1);
	}
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

void RenderGraph::AddExplicitPassDependency(std::string beforePass, std::string afterPass) {
	if (beforePass.empty() || afterPass.empty() || beforePass == afterPass) {
		return;
	}
	m_structuralExplicitAfterByName.emplace_back(std::move(beforePass), std::move(afterPass));
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
	if (!resource) {
		throw std::invalid_argument("RenderGraph::AddResource received a null resource");
	}
	if (!resource->IsRenderGraphManaged()) {
		throw std::runtime_error(
			"RenderGraph::AddResource rejected externally managed shader resource '" +
			resource->GetName() + "' (id=" + std::to_string(resource->GetGlobalResourceID()) + ")");
	}
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
		else if (std::dynamic_pointer_cast<BufferBase>(resource)) {
			resourceType = "Buffer";
		}
		spdlog::info(
			"RenderGraph::AddResource type={} name='{}' id={} transition={}",
			resourceType,
			name,
			resourceID,
			transition);
	}

	if (auto* backedResource = TryGetBackedResource(resource.get())) {
		backedResource->EnsureVirtualDescriptorSlotsAllocated();
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
	BT_ZONE_SCOPE("RenderGraph::Update");
	const bool traceLifecycle = m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled();
	{
		BT_ZONE_SCOPE("RenderGraph::Update::ResetForFrame");
		ResetForFrame();
	}

	// Poll readback completions early so that passes (e.g. CLod streaming)
	// can react to GPU-produced data from the previous frame immediately,
	// rather than waiting until post-Present.
	if (m_readbackService) {
		BT_ZONE_SCOPE("RenderGraph::Update::ProcessReadbacks");
		m_readbackService->ProcessReadbackRequests();
	}

	if (m_statisticsService) {
		BT_ZONE_SCOPE("RenderGraph::Update::BeginStatisticsFrame");
		m_statisticsService->BeginFrame();
	}

	auto toMilliseconds = [](auto duration) {
		return std::chrono::duration<double, std::milli>(duration).count();
	};

	{
		BT_ZONE_SCOPE("RenderGraph::Update::PassUpdates");
		for (auto& pr : m_masterPassList) {	
			// Resolve into type and update
			std::visit([&](auto& obj) {
				using T = std::decay_t<decltype(obj)>;
				if constexpr (std::is_same_v<T, std::monostate>) {
					// no-op
				}
				else {
					BT_ZONE_SCOPE("RenderGraph::Update::PassUpdate");
					if (!obj.name.empty()) {
						BT_ZONE_TEXT(obj.name.data(), obj.name.size());
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

	if (context.beforeCompileFrame) {
		BT_ZONE_SCOPE("RenderGraph::Update::BeforeCompileFrame");
		context.beforeCompileFrame();
	}

	{
		BT_ZONE_SCOPE("RenderGraph::Update::CompileFrame");
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
		BackendInstanceId backendInstance,
		rhi::CommandList& commandList) {
		rhi::helpers::OwnedBarrierBatch batch;
		for (auto& transition : transitions) {
			SubresourceRange resolvedRange{};
			if (transition.pResource->HasLayout() && !ResolveNonEmptyTransitionRange(transition, resolvedRange)) {
				continue;
			}

			std::vector<ResourceTransition> dummy;
			transition.pResource->GetStateTracker(backendInstance)->Apply(
				transition.range, transition.pResource,
				{ transition.newAccessType, transition.newLayout, transition.newSyncState }, dummy);
			auto bg = transition.pResource->GetEnhancedBarrierGroup(backendInstance,
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
		BackendInstanceId backendInstance,
		rhi::CommandList& commandList) {
		if (transitions.empty()) return;

		rhi::helpers::OwnedBarrierBatch batch;
		batch.textures.reserve(transitions.size());
		batch.buffers.reserve(transitions.size());

		for (auto& t : transitions) {
			if (t.pResource->HasLayout()) {
				SubresourceRange resolvedRange{};
				if (!ResolveNonEmptyTransitionRange(t, resolvedRange)) {
					continue;
				}
				// Let the resource encode backend/import-specific constraints. In
				// particular, cross-API simultaneous-access textures must retain
				// COMMON layout even while their access and synchronization scopes
				// change. The old parallel path bypassed this virtual contract.
				const size_t textureStart = batch.textures.size();
				const size_t bufferStart = batch.buffers.size();
				batch.Append(t.pResource->GetEnhancedBarrierGroup(backendInstance,
					t.range, t.prevAccessType, t.newAccessType,
					t.prevLayout, t.newLayout, t.prevSyncState, t.newSyncState));
				if (t.discard) {
					for (size_t i = textureStart; i < batch.textures.size(); ++i) batch.textures[i].discard = true;
					for (size_t i = bufferStart; i < batch.buffers.size(); ++i) batch.buffers[i].discard = true;
				}
			} else {
				// Buffer barrier
				rhi::BufferBarrier bb{};
				bb.buffer       = t.pResource->GetAPIResource(backendInstance).GetHandle();
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
		void* deviceImpl = nullptr;
		uint32_t index = 0;
		uint32_t generation = 0;
		uint64_t value = 0;

		bool operator==(const ExternalFenceSignalKey& other) const noexcept {
			return deviceImpl == other.deviceImpl && index == other.index && generation == other.generation && value == other.value;
		}
	};

	struct ExternalFenceSignalKeyHash {
		size_t operator()(const ExternalFenceSignalKey& key) const noexcept {
			size_t seed = std::hash<void*>{}(key.deviceImpl);
			seed ^= static_cast<size_t>(key.index) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
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
	struct QueueSlotTimelineIdentity {
		void* deviceImpl = nullptr;
		uint64_t handleKey = 0;
	};

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
		auto logSignal = [&](const rhi::Timeline& timeline, uint64_t value) {
			auto handle = timeline.GetHandle();
			ExternalFenceSignalKey key{
				.deviceImpl = timeline.impl,
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
			logSignal(*passReturn.fence, passReturn.fenceValue);
		}
		for (const auto& signal : passReturn.externalSignalsAfterCompletion) {
			if (signal.timeline.IsValid()) {
				logSignal(signal.timeline, signal.value);
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
		std::span<const QueueSlotTimelineIdentity> queueSlotFenceTimelineKeys,
		std::unordered_map<ExternalFenceSignalKey, ExternalFenceSignalOrigin, ExternalFenceSignalKeyHash>& seenSignals,
		std::unordered_map<uint64_t, uint64_t>& lastExternalSignalValueByTimeline,
		std::vector<PassReturn>& externalFences) {
		BT_ZONE_SCOPE("RenderGraph::SignalExternalFences");
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
					if (fr.fence.value().impl == slotFence->impl &&
						a.index == b.index && a.generation == b.generation) {
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
				const uint64_t timelineKey = PackTimelineSignalKey(handle)
					^ (static_cast<uint64_t>(std::hash<void*>{}(fr.fence.value().impl)) * 0x9e3779b97f4a7c15ull);
				bool aliasesQueueSlotFence = false;
				for (size_t aliasedSlot = 0; aliasedSlot < queueSlotFenceTimelineKeys.size(); ++aliasedSlot) {
					if (queueSlotFenceTimelineKeys[aliasedSlot].deviceImpl != fr.fence.value().impl ||
						queueSlotFenceTimelineKeys[aliasedSlot].handleKey != timelineKey) {
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
					.deviceImpl = fr.fence.value().impl,
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
				if (SarpClodImportDebugLoggingEnabled()) {
					spdlog::info(
						"SARPDBG SignalExternalFences frame={} queue={} slot={} batch={} timeline(idx={}, gen={}) value={} completedBefore={}",
						frameIndex,
						QueueKindToString(queueKind),
						queueSlot,
						batchIndex,
						handle.index,
						handle.generation,
						fr.fenceValue,
						fr.fence.value().GetCompletedValue());
				}
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
		unsigned frameIndex,
		const std::vector<ExternalTimelineBindingValue>& bindingValues)
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
		for (const auto binding : batch.ExternalWaitBindingsBeforeTransitions(queueSlot)) {
			const auto value = std::find_if(bindingValues.begin(), bindingValues.end(),
				[binding](const auto& candidate) { return candidate.binding == binding; });
			if (value == bindingValues.end() || !value->point.timeline.IsValid() || value->point.value == 0 || value->point.value == UINT64_MAX)
				throw std::runtime_error("RenderGraph external wait binding was missing or invalid");
			if (queue.Wait({ value->point.timeline.GetHandle(), value->point.value }) != rhi::Result::Ok)
				throw std::runtime_error("RenderGraph external wait binding failed");
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
		org::runtime::IStatisticsService* statisticsService;
		std::vector<PassReturn>& outExternalFences;
		std::unordered_map<ExternalFenceSignalKey, ExternalFenceSignalOrigin, ExternalFenceSignalKeyHash>& queuedExternalFenceOrigins;
		UINT64& lastSignaledOnTimeline;
		UINT64& greatestActuallySignaledOnTimeline;
		bool batchTraceEnabled;
	};

	void ExecuteQueueBatch(
		ExecuteQueueBatchArgs& args,
		auto&& WaitOnSlot)
	{
		BT_ZONE_SCOPE("RenderGraph::ExecuteQueueBatch");
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
			static_cast<unsigned>(args.context.frameIndex),
			args.context.externalTimelineBindings);
		for (size_t srcIndex = 0; srcIndex < batch.QueueCount(); ++srcIndex) {
			if (!batch.HasQueueWait(RenderGraph::BatchWaitPhase::BeforeTransitions, qi, srcIndex))
				continue;
			UINT64 val = fenceOffset + batch.GetQueueWaitFenceValue(
				RenderGraph::BatchWaitPhase::BeforeTransitions, qi, srcIndex);
			if (args.batchTraceEnabled) {
				spdlog::info(
					"RenderGraph: frame {} batch {} slot {} enqueue wait phase=BeforeTransitions srcSlot={} value={}",
					static_cast<unsigned>(args.context.frameIndex), args.batchIndex, qi, srcIndex, val);
			}
			WaitOnSlot(qi, srcIndex, val, fmt::format("batch={} phase=BeforeTransitions", args.batchIndex));
		}

		// Open first CL and record pre-transitions
		auto& cl0 = sched.preallocatedCLs[clIndex];
		rhi::CommandList commandList = cl0.list.Get();

		auto& preTransitions = batch.Transitions(qi, RenderGraph::BatchTransitionPhase::BeforePasses);
		ExecuteTransitions(preTransitions, /*crm=*/nullptr, queue, args.context.backendInstance, commandList);

		// Waits: BeforeExecution
		for (size_t srcIndex = 0; srcIndex < batch.QueueCount(); ++srcIndex) {
			if (!batch.HasQueueWait(RenderGraph::BatchWaitPhase::BeforeExecution, qi, srcIndex))
				continue;
			UINT64 val = fenceOffset + batch.GetQueueWaitFenceValue(
				RenderGraph::BatchWaitPhase::BeforeExecution, qi, srcIndex);
			if (args.batchTraceEnabled) {
				spdlog::info(
					"RenderGraph: frame {} batch {} slot {} enqueue wait phase=BeforeExecution srcSlot={} value={}",
					static_cast<unsigned>(args.context.frameIndex), args.batchIndex, qi, srcIndex, val);
			}
			WaitOnSlot(qi, srcIndex, val, fmt::format("batch={} phase=BeforeExecution", args.batchIndex));
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
			args.greatestActuallySignaledOnTimeline =
				std::max(args.greatestActuallySignaledOnTimeline, signalValue);
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
				BT_ZONE_SCOPE("RenderGraph::ExecuteQueueBatch::PassExecute");
				BT_ZONE_TEXT(passName.data(), passName.size());
					if (args.batchTraceEnabled) {
						spdlog::info(
							"RenderGraph: frame {} queue {} slot {} batch {} begin pass {}",
							static_cast<unsigned>(args.context.frameIndex),
							QueueKindToString(queue),
							qi,
							args.batchIndex,
							passName);
					}
				rhi::debug::Scope scope(commandList, rhi::colors::Mint, passName.data());
				args.context.currentPassName = passName.data();
				args.context.currentTechniquePath = techniquePath;
				(void)rhi::debug::SetInstrumentationContext(commandList, args.context.currentPassName, args.context.currentTechniquePath);
				(void)commandList.BeginTracyGpuZone(rhiQueue, args.context.currentPassName);
				if (args.context.beginGpuPassRange) {
					args.context.beginGpuPassRange(commandList, rhiQueue, QueueKindToString(queue), args.context.currentPassName);
				}
				const bool hasStatistics = args.statisticsService && pr.statisticsIndex >= 0;
				const auto cpuStart = std::chrono::steady_clock::now();
				if (hasStatistics)
					args.statisticsService->BeginQuery(pr.statisticsIndex, args.context.frameIndex, rhiQueue, commandList);
				if ((pr.run & PassRunMask::Immediate) != PassRunMask::None)
					org::imm::Replay(pr.immediateBytecode, commandList, *args.context.immediateDispatch);
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
				if (args.context.endGpuPassRange) {
					args.context.endGpuPassRange(commandList, rhiQueue);
				}
				commandList.EndTracyGpuZone();
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
				if (args.context.endGpuPassRange) {
					args.context.endGpuPassRange(commandList, rhiQueue);
				}
				commandList.EndTracyGpuZone();
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
				if (args.context.endGpuPassRange) {
					args.context.endGpuPassRange(commandList, rhiQueue);
				}
				commandList.EndTracyGpuZone();
				(void)rhi::debug::SetInstrumentationContext(commandList, nullptr, nullptr);
				args.context.currentPassName = nullptr;
				args.context.currentTechniquePath = nullptr;
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
			args.greatestActuallySignaledOnTimeline =
				std::max(args.greatestActuallySignaledOnTimeline, signalValue);
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
			if (args.batchTraceEnabled) {
				spdlog::info(
					"RenderGraph: frame {} batch {} slot {} enqueue wait phase=BeforeAfterPasses srcSlot={} value={}",
					static_cast<unsigned>(args.context.frameIndex), args.batchIndex, qi, srcIndex, val);
			}
			WaitOnSlot(qi, srcIndex, val, fmt::format("batch={} phase=BeforeAfterPasses", args.batchIndex));
		}

		// Record post-transitions
		auto& postTransitions = batch.Transitions(qi, RenderGraph::BatchTransitionPhase::AfterPasses);
		if (!postTransitions.empty())
			ExecuteTransitions(postTransitions, /*crm=*/nullptr, queue, args.context.backendInstance, commandList);

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
			args.greatestActuallySignaledOnTimeline =
				std::max(args.greatestActuallySignaledOnTimeline, recycleFence);
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
		ResourceRegistry& registry;
		PassExecutionContext context;    // COPY: each task gets its own
		org::runtime::IStatisticsService* statisticsService;
		std::unordered_map<ExternalFenceSignalKey, ExternalFenceSignalOrigin, ExternalFenceSignalKeyHash>& queuedExternalFenceOrigins;
		bool batchTraceEnabled;
	};

	void RecordExternalOwnershipBarriers(
		const std::vector<RenderGraph::ExternalOwnershipBarrier>& barriers,
		BackendInstanceId backendInstance,
		bool acquire,
		rhi::CommandList commandList) {
		rhi::helpers::OwnedBarrierBatch batch;
		for (const auto& entry : barriers) {
			if (!entry.resource) continue;
			const auto apiResource = entry.resource->GetAPIResource(backendInstance);
			if (!apiResource) continue;
			if (entry.resource->HasLayout()) {
				rhi::TextureBarrier barrier{};
				barrier.texture = apiResource.GetHandle();
				const auto resolvedRange = ResolveRangeSpec(
					entry.range,
					entry.resource->GetMipLevels(),
					entry.resource->GetArraySize());
				barrier.range = {
					resolvedRange.firstMip,
					resolvedRange.mipCount,
					resolvedRange.firstSlice,
					resolvedRange.sliceCount };
				barrier.beforeSync = entry.state.sync;
				barrier.afterSync = rhi::ResourceSyncState::All;
				barrier.beforeAccess = entry.state.access;
				barrier.afterAccess = rhi::ResourceAccessType::Common;
				barrier.beforeLayout = entry.state.layout;
				barrier.afterLayout = rhi::ResourceLayout::Common;
				barrier.externalOwnership = acquire
					? rhi::TextureBarrier::ExternalOwnership::Acquire
					: rhi::TextureBarrier::ExternalOwnership::Release;
				batch.textures.push_back(barrier);
			}
			else {
				rhi::BufferBarrier barrier{};
				barrier.buffer = apiResource.GetHandle();
				barrier.beforeSync = entry.state.sync;
				barrier.afterSync = rhi::ResourceSyncState::All;
				barrier.beforeAccess = entry.state.access;
				barrier.afterAccess = rhi::ResourceAccessType::Common;
				barrier.externalOwnership = acquire
					? rhi::BufferBarrier::ExternalOwnership::Acquire
					: rhi::BufferBarrier::ExternalOwnership::Release;
				batch.buffers.push_back(barrier);
			}
		}
		if (!batch.Empty()) commandList.Barriers(batch.View());
	}

	void RecordQueueBatch(RecordQueueBatchArgs& args) {
		BT_ZONE_SCOPE("RenderGraph::RecordQueueBatch");
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
		RecordTransitionBarriers(preTransitions, args.context.backendInstance, commandList);

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
			if (args.batchTraceEnabled && (!pr.externalAcquires.empty() || !pr.backendPreTransitions.empty())) {
				spdlog::info("RenderGraph: frame {} pass {} backend={} externalAcquires={} backendPreTransitions={}",
					static_cast<unsigned>(args.context.frameIndex), pr.name,
					static_cast<unsigned>(args.context.backendInstance), pr.externalAcquires.size(), pr.backendPreTransitions.size());
				for (const auto& transition : pr.backendPreTransitions) {
					spdlog::info("RenderGraph: pass {} transition resource='{}' layout {}->{} access {}->{}",
						pr.name, transition.pResource ? transition.pResource->GetName() : "<null>",
						static_cast<unsigned>(transition.prevLayout), static_cast<unsigned>(transition.newLayout),
						static_cast<unsigned>(transition.prevAccessType), static_cast<unsigned>(transition.newAccessType));
				}
			}
			// A crossed representation remains externally owned until this exact
			// consumer.  Acquiring here avoids changing the layout before earlier
			// passes in the same queue batch have executed.
			RecordExternalOwnershipBarriers(
				pr.externalAcquires,
				args.context.backendInstance,
				true,
				commandList);
			RecordTransitionBarriers(pr.backendPreTransitions, args.context.backendInstance, commandList);
			const std::string_view passName = pr.name.empty() ? std::string_view("<unnamed>") : std::string_view(pr.name);
			const char* techniquePath = pr.techniquePath.empty() ? nullptr : pr.techniquePath.c_str();
			try {
				BT_ZONE_SCOPE("RenderGraph::RecordQueueBatch::PassRecord");
				BT_ZONE_TEXT(passName.data(), passName.size());
				if (args.batchTraceEnabled) {
					spdlog::info(
						"RenderGraph: frame {} batch {} queue {} slot {} begin pass {}",
						static_cast<unsigned>(args.context.frameIndex),
						args.batchIndex,
						QueueKindToString(queue),
						qi,
						passName);
				}
				rhi::debug::Scope scope(commandList, rhi::colors::Mint, passName.data());
				args.context.currentPassName = passName.data();
				args.context.currentTechniquePath = techniquePath;
				{
					BT_ZONE_SCOPE("RenderGraph::PassRecord::BeginInstrumentation");
					(void)rhi::debug::SetInstrumentationContext(commandList, args.context.currentPassName, args.context.currentTechniquePath);
					(void)commandList.BeginTracyGpuZone(args.rhiQueue, args.context.currentPassName);
					if (args.context.beginGpuPassRange) {
						args.context.beginGpuPassRange(commandList, args.rhiQueue, QueueKindToString(queue), args.context.currentPassName);
					}
				}
				const bool hasStatistics = args.statisticsService && pr.statisticsIndex >= 0;
				const auto cpuStart = std::chrono::steady_clock::now();
				{
					BT_ZONE_SCOPE("RenderGraph::PassRecord::ExecuteBody");
					BT_ZONE_TEXT(passName.data(), passName.size());
					if (hasStatistics)
						args.statisticsService->BeginQuery(pr.statisticsIndex, args.context.frameIndex, args.rhiQueue, commandList, sched.queryRecordingContext);
					if ((pr.run & PassRunMask::Immediate) != PassRunMask::None)
						org::imm::Replay(pr.immediateBytecode, commandList, *args.context.immediateDispatch);
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
				}
				{
					BT_ZONE_SCOPE("RenderGraph::PassRecord::Finalize");
					RecordTransitionBarriers(pr.backendPostTransitions, args.context.backendInstance, commandList);
					RecordExternalOwnershipBarriers(pr.externalReleases, args.context.backendInstance, false, commandList);
					if (hasStatistics)
						args.statisticsService->EndQuery(pr.statisticsIndex, args.context.frameIndex, args.rhiQueue, commandList, sched.queryRecordingContext);
				}
				if (hasStatistics) {
					args.statisticsService->RecordCpuExecuteTime(
						static_cast<unsigned>(pr.statisticsIndex),
						std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuStart).count());
				}
				{
					BT_ZONE_SCOPE("RenderGraph::PassRecord::EndInstrumentation");
					if (args.context.endGpuPassRange) {
						args.context.endGpuPassRange(commandList, args.rhiQueue);
					}
					commandList.EndTracyGpuZone();
					(void)rhi::debug::SetInstrumentationContext(commandList, nullptr, nullptr);
				}
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
				if (args.context.endGpuPassRange) {
					args.context.endGpuPassRange(commandList, args.rhiQueue);
				}
				commandList.EndTracyGpuZone();
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
				if (args.context.endGpuPassRange) {
					args.context.endGpuPassRange(commandList, args.rhiQueue);
				}
				commandList.EndTracyGpuZone();
				(void)rhi::debug::SetInstrumentationContext(commandList, nullptr, nullptr);
				args.context.currentPassName = nullptr;
				args.context.currentTechniquePath = nullptr;
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
			RecordTransitionBarriers(postTransitions, args.context.backendInstance, commandList);

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
		BT_ZONE_SCOPE("RenderGraph::SubmitQueueBatch");
		BT_ZONE_TEXT(QueueKindToString(args.queue), std::strlen(QueueKindToString(args.queue)));
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
			BT_ZONE_SCOPE("RenderGraph::SubmitQueueBatch::WaitBeforeTransitions");
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
			BT_ZONE_SCOPE("RenderGraph::SubmitQueueBatch::WaitBeforeExecution");
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
			BT_ZONE_SCOPE("RenderGraph::SubmitQueueBatch::SubmitAfterTransitions");
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
			BT_ZONE_SCOPE("RenderGraph::SubmitQueueBatch::SubmitAfterExecution");
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
			BT_ZONE_SCOPE("RenderGraph::SubmitQueueBatch::WaitBeforeAfterPasses");
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
			BT_ZONE_SCOPE("RenderGraph::SubmitQueueBatch::SubmitAfterCompletion");
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
	BT_ZONE_SCOPE("RenderGraph::BuildExecutionSchedule");
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
	BT_ZONE_SCOPE("RenderGraph::Execute");
	m_lastPresentDependency.reset();
	{
		BT_ZONE_SCOPE("RenderGraph::Execute::ValidateCompiledResourceGenerations");
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
		BT_ZONE_SCOPE("RenderGraph::Execute::CreateCommandRecordingManager");
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

	auto WaitOnSlot = [&](size_t dstSlot, size_t srcSlot, UINT64 absoluteFenceValue, std::string_view reason = {}) {
		if (dstSlot == srcSlot) return;
		BT_ZONE_SCOPE("RenderGraph::Execute::FrameStartWaitOnSlot");
		if (absoluteFenceValue == 0 || absoluteFenceValue == UINT64_MAX) {
			throw std::runtime_error(fmt::format(
				"WaitOnSlot rejected invalid fence value: dstSlot={} srcSlot={} value={} reason='{}'",
				dstSlot,
				srcSlot,
				absoluteFenceValue,
				reason));
		}
		auto& srcFence = SlotFence(srcSlot);
		auto& waitFence = m_queueRegistry.GetFenceForConsumer(
			static_cast<QueueSlotIndex>(static_cast<uint8_t>(srcSlot)),
			static_cast<QueueSlotIndex>(static_cast<uint8_t>(dstSlot)));
		const auto srcFenceHandle = srcFence.GetHandle();
		const auto waitFenceHandle = waitFence.GetHandle();
		const auto waitBegin = std::chrono::steady_clock::now();
		UINT64 completedFenceValue = 0;
		{
			BT_ZONE_SCOPE("RenderGraph::Execute::FrameStartWaitOnSlot::GetCompletedValue");
			completedFenceValue = srcFence.GetCompletedValue();
		}
		const uint64_t pendingDelta = completedFenceValue < absoluteFenceValue
			? absoluteFenceValue - completedFenceValue
			: 0;
		BT_ZONE_VALUE(pendingDelta);
		const std::string waitText = fmt::format(
			"frame={} dstSlot={} srcSlot={} fence={} completed={} delta={} reason='{}'",
			static_cast<unsigned>(context.frameIndex),
			dstSlot,
			srcSlot,
			absoluteFenceValue,
			completedFenceValue,
			pendingDelta,
			reason);
		BT_ZONE_TEXT(waitText.c_str(), waitText.size());
		BT_PLOT("RG.FrameStartWait.PendingDelta", static_cast<int64_t>(std::min<uint64_t>(pendingDelta, static_cast<uint64_t>(INT64_MAX))));
		if (completedFenceValue == UINT64_MAX) {
			throw std::runtime_error(fmt::format(
				"WaitOnSlot detected poisoned queue timeline before wait: dstSlot={} srcSlot={} requestedFence={} completed=UINT64_MAX timeline(idx={}, gen={}) reason='{}'",
				dstSlot,
				srcSlot,
				absoluteFenceValue,
				srcFenceHandle.index,
				srcFenceHandle.generation,
				reason));
		}
		if (completedFenceValue >= absoluteFenceValue) {
			BT_PLOT("RG.FrameStartWait.AlreadyCompleted", int64_t{ 1 });
			return;
		}
		BT_PLOT("RG.FrameStartWait.AlreadyCompleted", int64_t{ 0 });
		if (batchTraceEnabled) {
			spdlog::info("RenderGraph::Execute frame={} queue wait dstSlot={} srcSlot={} value={} srcCompleted={} sourceTimeline=({}, {}) consumerTimeline=({}, {}) reason='{}'",
				static_cast<unsigned>(context.frameIndex), dstSlot, srcSlot, absoluteFenceValue,
				completedFenceValue, srcFenceHandle.index, srcFenceHandle.generation,
				waitFenceHandle.index, waitFenceHandle.generation, reason);
		}

		auto dstQ = SlotQueue(dstSlot);
		rhi::Result waitResult = rhi::Result::Ok;
		{
			BT_ZONE_SCOPE("RenderGraph::Execute::FrameStartWaitOnSlot::QueueWait");
			waitResult = dstQ.Wait({ waitFenceHandle, absoluteFenceValue });
		}
		const auto waitElapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
			std::chrono::steady_clock::now() - waitBegin).count();
		BT_PLOT("RG.FrameStartWait.ElapsedUs", static_cast<int64_t>(waitElapsedUs));
		if (waitElapsedUs >= 1000) {
			spdlog::warn(
				"RenderGraph frame-start wait slow: frame={} dstSlot={} srcSlot={} fence={} completedBefore={} delta={} elapsed_us={} timeline(idx={}, gen={}) reason='{}'",
				static_cast<unsigned>(context.frameIndex),
				dstSlot,
				srcSlot,
				absoluteFenceValue,
				completedFenceValue,
				pendingDelta,
				waitElapsedUs,
				srcFenceHandle.index,
				srcFenceHandle.generation,
				reason);
		}
		if (waitResult != rhi::Result::Ok) {
			throw std::runtime_error(fmt::format(
				"WaitOnSlot failed: dstSlot={} srcSlot={} fence={} completed={} timeline(idx={}, gen={}) result={} reason='{}'",
				dstSlot,
				srcSlot,
				absoluteFenceValue,
				completedFenceValue,
				srcFenceHandle.index,
				srcFenceHandle.generation,
				rhi::ResultName(waitResult),
				reason));
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
		BT_ZONE_SCOPE("RenderGraph::Execute::ApplyFrameStartWaits");
		// Frame-start waits from previous frame's last-producer tracking.
		for (size_t dstIndex = 0; dstIndex < slotCount; ++dstIndex) {
			for (size_t srcIndex = 0; srcIndex < slotCount; ++srcIndex) {
				if (dstIndex == srcIndex) continue;
				if (dstIndex >= m_hasPendingFrameStartQueueWait.size() ||
					srcIndex >= m_hasPendingFrameStartQueueWait[dstIndex].size()) continue;
				if (!m_hasPendingFrameStartQueueWait[dstIndex][srcIndex]) continue;
				WaitOnSlot(dstIndex, srcIndex,
					m_pendingFrameStartQueueWaitFenceValue[dstIndex][srcIndex],
					fmt::format("FrameStart dst={} src={}", dstIndex, srcIndex));
			}
		}
	}
	if (batchTraceEnabled) {
		spdlog::info("RenderGraph::Execute frame={} completed frame-start waits", static_cast<unsigned>(context.frameIndex));
	}

	{
		BT_ZONE_SCOPE("RenderGraph::Execute::MarkCompletionSignals");
		// Cross-frame waits only need a monotonic signal that is guaranteed to fire
		// after the queue's final work for the frame. Marking every producer batch
		// forces extra submissions in the parallel path.
		for (size_t queueIndex = 0; queueIndex < slotCount; ++queueIndex) {
			const bool hasProducers =
				queueIndex < m_compilerState->compiledLastProducerBatchByResourceByQueue.size()
				&& !m_compilerState->compiledLastProducerBatchByResourceByQueue[queueIndex].empty();
			const bool hasAccesses =
				queueIndex < m_compilerState->compiledLastAccessBatchByResourceByQueue.size()
				&& !m_compilerState->compiledLastAccessBatchByResourceByQueue[queueIndex].empty();
			if (!hasProducers && !hasAccesses) continue;

			const unsigned int signalBatch = lastCrossFrameSignalBatchByQueue[queueIndex];
			if (signalBatch > 0 && signalBatch < batches.size()) {
				batches[signalBatch].MarkQueueSignal(BatchSignalPhase::AfterCompletion, queueIndex);
			}
			else {
				spdlog::warn(
					"RenderGraph::Execute frame={} queue slot {} has {} cross-frame producers but no active batch to signal.",
					static_cast<unsigned>(context.frameIndex),
					queueIndex,
					(hasProducers ? m_compilerState->compiledLastProducerBatchByResourceByQueue[queueIndex].size() : 0)
						+ (hasAccesses ? m_compilerState->compiledLastAccessBatchByResourceByQueue[queueIndex].size() : 0));
			}
		}
	}
	if (batchTraceEnabled) {
		spdlog::info("RenderGraph::Execute frame={} marked completion signals", static_cast<unsigned>(context.frameIndex));
	}

	{
		BT_ZONE_SCOPE("RenderGraph::Execute::AssignQueueSignalFenceValues");
		AssignQueueSignalFenceValuesInSubmissionOrder(batches);
	}
	if (batchTraceEnabled) {
		spdlog::info("RenderGraph::Execute frame={} assigned queue signal fences in submission order", static_cast<unsigned>(context.frameIndex));
	}

	auto& nextLastProducerByResourceAcrossFrames = m_lastProducerByResourceAcrossFrames;
	auto& nextLastAccessByResourceAcrossFrames = m_lastAccessByResourceAcrossFrames;
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
		const auto* placement = TryGetAliasPlacementRange(resourceID);
		if (!placement) {
			return;
		}
		// Most tracked producers are not alias placements. Do not scan every
		// persistent alias-pool producer list for those ordinary resources.
		removeResourceFromLastAliasPlacementCache(resourceID);

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
		BT_ZONE_SCOPE("RenderGraph::Execute::BuildExecutionSchedule");
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
		BT_ZONE_SCOPE("RenderGraph::Execute::DebugScheduleValidation");
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
			if (qi >= m_compilerState->compiledLastProducerBatchByResourceByQueue.size()) continue;
			size_t count = m_compilerState->compiledLastProducerBatchByResourceByQueue[qi].size();
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
		BT_ZONE_SCOPE("RenderGraph::Execute::PreallocateCommandLists");
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
					// Keep graph provenance on command buffers even in the normal parallel
					// path.  Vulkan synchronization validation reports the command-buffer
					// debug name, and numeric pool names alone make an intermittent hazard
					// impossible to associate with its transition/pass batch.
					if (qs.preallocatedCLs[ci].list) {
						const auto queueKind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(qi));
						auto debugName = MakeRenderGraphCommandListName(
							static_cast<unsigned>(context.frameIndex),
							bi,
							qi,
							queueKind,
							qs,
							ci);
						if (qi < batches[bi].queuePasses.size() && !batches[bi].queuePasses[qi].empty()) {
							debugName += " passes=";
							bool first = true;
							for (const auto& queuedPass : batches[bi].queuePasses[qi]) {
								std::visit([&](const auto* pass) {
									if (!pass) return;
									if (!first) debugName += ',';
									debugName += pass->name;
									first = false;
								}, queuedPass);
							}
						}
						qs.preallocatedCLs[ci].list->SetName(debugName.c_str());
					}
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

	// Advance each Tracy queue context exactly once per graph execution.  Vulkan
	// records query collection/reset commands into the first submitted command
	// list for that queue; D3D12 uses the same call as its per-frame resolve point.
	for (size_t qi = 0; qi < slotCount; ++qi) {
		for (auto& batchSchedule : m_executionSchedule.batches) {
			if (qi >= batchSchedule.queues.size()) {
				continue;
			}
			auto& queueSchedule = batchSchedule.queues[qi];
			if (!queueSchedule.active || queueSchedule.numCLs == 0 ||
				!queueSchedule.preallocatedCLs[0].list) {
				continue;
			}
			auto queue = SlotQueue(qi);
			auto firstCommandList = queueSchedule.preallocatedCLs[0].list.Get();
			queue.TracyGpuFrameBegin(firstCommandList);
			break;
		}
	}

	// Per-slot signal tracking for monotonic recycle signals.
	std::vector<UINT64> lastSignaledPerSlot(slotCount);
	std::vector<UINT64> greatestActuallySignaledPerSlot(slotCount);
	std::vector<QueueSlotTimelineIdentity> queueSlotFenceTimelineKeys(slotCount);
	std::unordered_map<ExternalFenceSignalKey, ExternalFenceSignalOrigin, ExternalFenceSignalKeyHash> seenExternalFenceSignalsThisFrame;
	seenExternalFenceSignalsThisFrame.reserve(32);
	std::unordered_map<ExternalFenceSignalKey, ExternalFenceSignalOrigin, ExternalFenceSignalKeyHash> queuedExternalFenceOriginsThisFrame;
	queuedExternalFenceOriginsThisFrame.reserve(32);
	for (size_t qi = 0; qi < slotCount; ++qi) {
		const auto slotIndex = static_cast<QueueSlotIndex>(static_cast<uint8_t>(qi));
		queueSlotFenceTimelineKeys[qi] = {
			.deviceImpl = SlotFence(qi).impl,
			.handleKey = PackTimelineSignalKey(SlotFence(qi).GetHandle())
		};
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
		BT_ZONE_SCOPE("RenderGraph::Execute::HeavyDebugPath");
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
				const auto slotIndex = static_cast<QueueSlotIndex>(static_cast<uint8_t>(qi));
				PassExecutionContext slotContext = context;
				slotContext.device = m_queueRegistry.GetDevice(slotIndex);
				slotContext.backendInstance = m_queueRegistry.GetBackendInstance(slotIndex);
				ExecuteQueueBatchArgs args{
					.sched = qs,
					.batch = batch,
					.batchIndex = bi,
					.queue = m_queueRegistry.GetKind(slotIndex),
					.queueSlot = qi,
					.rhiQueue = rhiQ,
					.fenceTimeline = SlotFence(qi),
					.pool = *SlotPool(qi),
					.fenceOffset = 0,
					.context = slotContext,
					.statisticsService = statisticsService,
					.outExternalFences = slotExternalFences[qi],
					.queuedExternalFenceOrigins = queuedExternalFenceOriginsThisFrame,
					.lastSignaledOnTimeline = lastSignaledPerSlot[qi],
					.greatestActuallySignaledOnTimeline = greatestActuallySignaledPerSlot[qi],
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
				const UINT64 completedBeforeWait = SlotFence(qi).GetCompletedValue();
				spdlog::info(
					"[HeavyDebug] drain begin frame={} batch={} queueSlot={} queue={} target={} completed={} passes=[{}]",
					static_cast<unsigned>(context.frameIndex),
					batchIndex,
					qi,
					QueueKindToString(m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(qi)))),
					highestSignal,
					completedBeforeWait,
					passNames);
				// Heavy debug is a fault-isolation mode. Do not wait forever: a
				// missing GPU signal must leave an actionable batch/pass diagnostic.
				auto result = SlotFence(qi).HostWait(highestSignal, 30000);
				DeviceManager::GetInstance().GetDevice().CheckDebugMessages();
				if (rhi::Failed(result)) {
					spdlog::error(
						"[HeavyDebug] GPU drain failed frame={} batch={} queueSlot={} target={} completed={} result={}. Passes=[{}]",
						static_cast<unsigned>(context.frameIndex), batchIndex, qi, highestSignal,
						SlotFence(qi).GetCompletedValue(), rhi::ResultName(result), passNames);
					throw std::runtime_error(fmt::format(
						"Heavy-debug GPU drain failed after batch {} on queue slot {} (target {}, passes [{}])",
						batchIndex, qi, highestSignal, passNames));
				}
				spdlog::info(
					"[HeavyDebug] drain end frame={} batch={} queueSlot={} target={} completed={}",
					static_cast<unsigned>(context.frameIndex), batchIndex, qi, highestSignal,
					SlotFence(qi).GetCompletedValue());
			}
			++batchIndex;
		}
	} else {
		BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath");
		if (batchTraceEnabled) {
			spdlog::info("RenderGraph::Execute frame={} entering parallel path", static_cast<unsigned>(context.frameIndex));
		}
		// Parallel recording path

		// Clear per-frame recording state from any previous frame.
		{
			BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::ResetRecordingState");
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
			BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::BuildRecordTasks");
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
			BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::RecordAllBatches");
			if (batchTraceEnabled) {
				spdlog::info(
					"RenderGraph::Execute frame={} record-all-batches begin taskCount={} (serialBypass={})",
					static_cast<unsigned>(context.frameIndex),
					tasks.size(),
					false);
			}
			// DX12 command recording scales poorly once too many independent lists enter
			// the driver concurrently. Keep the render thread participating, but leave
			// capacity for streaming and avoid the all-core contention cliff.
			ParallelForOptionalLimited("RecordAllBatches", tasks.size(), PassRecordingConcurrency(), [&](size_t taskIdx) {
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
						.registry = _registry,
					.context = context,
						.statisticsService = statisticsService,
						.queuedExternalFenceOrigins = queuedExternalFenceOriginsThisFrame,
						.batchTraceEnabled = batchTraceEnabled,
				};
				const auto slotIndex = static_cast<QueueSlotIndex>(static_cast<uint8_t>(task.queueIndex));
				args.context.device = m_queueRegistry.GetDevice(slotIndex);
				args.context.backendInstance = m_queueRegistry.GetBackendInstance(slotIndex);
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
				});
			if (batchTraceEnabled) {
				spdlog::info("RenderGraph::Execute frame={} record-all-batches complete", static_cast<unsigned>(context.frameIndex));
			}
		}

		// Merge per-task statistics contexts.
		if (statisticsService) {
			BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::MergePendingResolves");
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
			BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::SubmitAllBatches");
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
				BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::SubmitPendingWithoutSignal");
				BT_ZONE_TEXT(reason, std::strlen(reason));
				auto& pending = pendingSubmissions[queueIndex];
				if (!pending.HasPendingCommandLists()) {
					return;
				}

				auto rhiQueue = SlotQueue(queueIndex);
				const QueueKind queueKind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(queueIndex));
				BT_ZONE_TEXT(QueueKindToString(queueKind), std::strlen(QueueKindToString(queueKind)));
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

				{
					BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::SubmitPendingWithoutSignal::RHI Submit");
					rhiQueue.Submit({ pending.pendingCommandLists.data(), static_cast<uint32_t>(pending.pendingCommandLists.size()) }, {});
				}
				{
					BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::SubmitPendingWithoutSignal::TrackPairs");
					for (auto& pair : pending.pendingPairs) {
						pending.submittedPairsAwaitingRecycle.push_back(std::move(pair));
					}
					pending.pendingPairs.clear();
					pending.pendingCommandLists.clear();
				}
			};

			auto signalAndRecycleQueue = [&](size_t queueIndex, size_t batchIndex, UINT64 signalValue, const char* reason) {
				BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::SignalAndRecycleQueue");
				BT_ZONE_TEXT(reason, std::strlen(reason));
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
				BT_ZONE_TEXT(QueueKindToString(queueKind), std::strlen(QueueKindToString(queueKind)));

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
				greatestActuallySignaledPerSlot[queueIndex] =
					std::max(greatestActuallySignaledPerSlot[queueIndex], signalValue);
				auto [slotSignalIt, insertedSlotSignal] = m_lastExternalSignalValueByTimeline.try_emplace(
					PackTimelineSignalKey(fenceTimeline.GetHandle()),
					0);
				slotSignalIt->second = std::max(slotSignalIt->second, signalValue);

				if (pool) {
					BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::SignalAndRecycleQueue::RecyclePairs");
					for (auto& pair : pending.submittedPairsAwaitingRecycle) {
						pool->Recycle(std::move(pair), signalValue);
					}
				}
				pending.submittedPairsAwaitingRecycle.clear();
			};

			auto flushExternalFencesForQueue = [&](size_t queueIndex, size_t batchIndex, std::vector<PassReturn>& externalFences) {
				BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::FlushExternalFencesForQueue");
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
				BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::QueueRecordedCommandList");
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
				BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::ApplyBatchWaitPhase");
				BT_ZONE_TEXT(waitPhaseLabel, std::strlen(waitPhaseLabel));
				if (waitPhase == BatchWaitPhase::BeforeTransitions) {
					WaitExternalFencesBeforeTransitions(
						SlotQueue(queueIndex),
						batch,
						queueIndex,
						batchIndex,
						static_cast<unsigned>(context.frameIndex),
						context.externalTimelineBindings);
				}
				for (size_t srcIndex = 0; srcIndex < batch.QueueCount(); ++srcIndex) {
					if (!batch.HasQueueWait(waitPhase, queueIndex, srcIndex)) {
						continue;
					}
					WaitOnSlot(
						queueIndex,
						srcIndex,
						batch.GetQueueWaitFenceValue(waitPhase, queueIndex, srcIndex),
						fmt::format(
							"BatchWait frame={} batch={} phase={} dstQueue={} srcQueue={}",
							static_cast<unsigned>(context.frameIndex),
							batchIndex,
							waitPhaseLabel,
							queueIndex,
							srcIndex));
				}
			};

			for (size_t bi = 0; bi < batches.size(); ++bi) {
				BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::SubmitBatch");
				auto& batch = batches[bi];
				auto& batchSched = m_executionSchedule.batches[bi];

				for (size_t qi = 0; qi < batchSched.queues.size(); ++qi) {
					BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::SubmitQueue");
					auto& qs = batchSched.queues[qi];
					if (!qs.active) continue;
					const QueueKind queueKind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(qi));
					BT_ZONE_TEXT(QueueKindToString(queueKind), std::strlen(QueueKindToString(queueKind)));

					if (batchHasWaitsForQueue(batch, qi)) {
						submitPendingWithoutSignal(qi, bi, "BeforeQueueWaits");
					}

					uint8_t clIndex = 0;
					{
						BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::SubmitBatch::ApplyBeforeTransitionWaits");
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
						BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::SubmitBatch::ApplyAfterExecutionWaits");
						queueRecordedCommandList(qi, std::move(qs.preallocatedCLs[clIndex]));
						signalAndRecycleQueue(
							qi,
							bi,
							batch.GetQueueSignalFenceValue(BatchSignalPhase::AfterExecution, qi),
							"AfterExecution");
						++clIndex;
					}

					{
						BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::SubmitBatch::ApplyBeforeAfterPassesWaits");
						applyBatchWaitPhase(batch, bi, qi, BatchWaitPhase::BeforeAfterPasses);
						queueRecordedCommandList(qi, std::move(qs.preallocatedCLs[clIndex]));
					}
					if (qs.signalAfterCompletion) {
						BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::SubmitBatch::SignalAfterCompletion");
						signalAndRecycleQueue(
							qi,
							bi,
							batch.GetQueueSignalFenceValue(BatchSignalPhase::AfterCompletion, qi),
							"AfterCompletion");
					}

					{
						BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::SubmitBatch::FlushExternalFences");
						flushExternalFencesForQueue(qi, bi, qs.externalFences);
					}
				}
			}

			for (size_t qi = 0; qi < slotCount; ++qi) {
				BT_ZONE_SCOPE("RenderGraph::Execute::ParallelPath::EndOfFrameRecycleQueue");
				auto& pending = pendingSubmissions[qi];
				if (!pending.HasOutstandingWork()) {
					continue;
				}
				const QueueKind queueKind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(qi));
				BT_ZONE_TEXT(QueueKindToString(queueKind), std::strlen(QueueKindToString(queueKind)));
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
		BT_ZONE_SCOPE("RenderGraph::Execute::UpdateCrossFrameProducerTracking");
		const uint64_t publishSerial = ++m_crossFrameProducerPublishSerial;
		// Publish the end-of-frame signal for each queue that accessed a tracked
		// resource. Producers protect later reads; all accesses protect later
		// writes from prior read-only consumers and state transitions.
		for (size_t queueIndex = 0; queueIndex < slotCount; ++queueIndex) {
			const bool hasProducers =
				queueIndex < m_compilerState->compiledLastProducerBatchByResourceByQueue.size()
				&& !m_compilerState->compiledLastProducerBatchByResourceByQueue[queueIndex].empty();
			const bool hasAccesses =
				queueIndex < m_compilerState->compiledLastAccessBatchByResourceByQueue.size()
				&& !m_compilerState->compiledLastAccessBatchByResourceByQueue[queueIndex].empty();
			if (!hasProducers && !hasAccesses) continue;

			const unsigned int signalBatch = lastCrossFrameSignalBatchByQueue[queueIndex];
			if (signalBatch == 0 || signalBatch >= batches.size()) {
				spdlog::warn(
					"Cross-frame producer skip: slot {} has {} tracked resources but no active batch signal.",
					queueIndex,
					(hasProducers ? m_compilerState->compiledLastProducerBatchByResourceByQueue[queueIndex].size() : 0)
						+ (hasAccesses ? m_compilerState->compiledLastAccessBatchByResourceByQueue[queueIndex].size() : 0));
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

			if (hasAccesses) {
				for (const auto& accessEntry : m_compilerState->compiledLastAccessBatchByResourceByQueue[queueIndex]) {
					const uint64_t resourceID = accessEntry.resourceID;
					const unsigned int accessBatch = accessEntry.batchIndex;
					if (accessBatch == 0 || accessBatch >= batches.size()) continue;

					LastProducerAcrossFrames access{
						.queueSlot = queueIndex,
						.fenceValue = fenceValue,
						.publishSerial = publishSerial,
						.anonymous = accessEntry.anonymous,
					};
					auto& accesses = nextLastAccessByResourceAcrossFrames[resourceID];
					auto existing = std::find_if(
						accesses.begin(),
						accesses.end(),
						[&](const LastProducerAcrossFrames& prior) {
							return prior.queueSlot == queueIndex;
						});
					if (existing != accesses.end()) {
						*existing = access;
					}
					else {
						accesses.push_back(access);
					}
				}
			}

			if (!hasProducers) {
				continue;
			}
			for (const auto& producerEntry : m_compilerState->compiledLastProducerBatchByResourceByQueue[queueIndex]) {
				const uint64_t resourceID = producerEntry.resourceID;
				const unsigned int producerBatch = producerEntry.batchIndex;
				if (producerBatch == 0 || producerBatch >= batches.size()) continue;

				LastProducerAcrossFrames producer{
					.queueSlot = queueIndex,
					.fenceValue = fenceValue,
					.publishSerial = publishSerial,
					.anonymous = producerEntry.anonymous,
				};
				nextLastProducerByResourceAcrossFrames[resourceID] = producer;
				publishAliasPlacementProducer(resourceID, producer);
			}
		}
	}

	{
		BT_ZONE_SCOPE("RenderGraph::Execute::PruneCrossFrameProducerTracking");
		auto isLiveResourceID = [&](uint64_t resourceID) {
			auto itResource = resourcesByID.find(resourceID);
			if (itResource != resourcesByID.end() && itResource->second) {
				return true;
			}

			auto itTransient = m_transientFrameResourcesByID.find(resourceID);
			return itTransient != m_transientFrameResourcesByID.end() && itTransient->second;
		};

		std::erase_if(
			nextLastProducerByResourceAcrossFrames,
			[&](const auto& entry) {
				if (entry.second.anonymous) {
					return entry.second.publishSerial != m_crossFrameProducerPublishSerial;
				}
				return !isLiveResourceID(entry.first);
			});

		std::erase_if(
			nextLastAccessByResourceAcrossFrames,
			[&](auto& entry) {
				auto& accesses = entry.second;
				std::erase_if(
					accesses,
					[&](const LastProducerAcrossFrames& access) {
						if (access.anonymous) {
							return access.publishSerial != m_crossFrameProducerPublishSerial;
						}
						return !isLiveResourceID(entry.first);
					});
				return accesses.empty();
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
					return !isLiveResourceID(producer.resourceID);
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
		BT_ZONE_SCOPE("RenderGraph::Execute::FinalizeCommandRecording");
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
		BT_ZONE_SCOPE("RenderGraph::Execute::PublishDescriptorRetirementFences");
		std::vector<DescriptorHeapManager::QueueFenceSnapshotPoint> fenceSnapshot;
		fenceSnapshot.reserve(slotCount);
		for (size_t qi = 0; qi < slotCount; ++qi) {
			const UINT64 value = greatestActuallySignaledPerSlot[qi];
			if (value == 0 || value == UINT64_MAX) {
				continue;
			}
			fenceSnapshot.push_back(DescriptorHeapManager::QueueFenceSnapshotPoint{
				.timeline = SlotFence(qi),
				.value = value,
			});
		}
		DescriptorHeapManager::GetInstance().PublishQueueFenceSnapshot(std::move(fenceSnapshot));
	}

	{
		BT_ZONE_SCOPE("RenderGraph::Execute::RecycleCompletedCommandLists");
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
	const PassBatch& currentBatch,
	const BatchBuildState& batchBuildState,
	std::string_view,
	unsigned int currentBatchIndex,
	size_t candidateQueueSlot)
{
	auto findAliasedBatchOverlap = [&](const auto& summaryEntry) {
		std::pair<bool, bool> overlap{};
		if (summaryEntry.equivalentResourceIndices) {
			for (size_t equivalentResourceIndex : *summaryEntry.equivalentResourceIndices) {
				overlap.first = overlap.first || batchBuildState.ContainsResource(equivalentResourceIndex);
				overlap.second = overlap.second || batchBuildState.ContainsInternalTransition(equivalentResourceIndex);
				if (overlap.first && overlap.second) {
					break;
				}
			}
		}
		return overlap;
	};

	// For each internally modified resource
	for (const auto& transition : passSummary.internalTransitions) {
		// If this resource is used in the current batch, we need a new one
		if (batchBuildState.ContainsResource(transition.resourceIndex)) {
			return true;
		}
		if (findAliasedBatchOverlap(transition).first) {
			return true;
		}
	}

	// For each subresource requirement in this pass:
	for (const auto& requirement : passSummary.requirements) {
		const auto [overlapsAliasedResource, overlapsAliasedTransition] =
			findAliasedBatchOverlap(requirement);

		// Alias activations are emitted in BeforePasses of the consuming batch.
		// Only reject same-batch merging when that activation would clobber an
		// aliased-equivalent resource that is already live in the batch.
		if (requirement.resourceIndex < m_aliasActivationPendingByResourceIndex.size()
			&& m_aliasActivationPendingByResourceIndex[requirement.resourceIndex] != org::alias::AliasActivationReason::None
			&& (overlapsAliasedResource || overlapsAliasedTransition)) {
			return true;
		}

		// If this resource is internally modified in the current batch, we need a new one
		if (batchBuildState.ContainsInternalTransition(requirement.resourceIndex)) {
			return true;
		}
		if (overlapsAliasedResource || overlapsAliasedTransition) {
			return true;
		}

		ResourceState wantState{ requirement.state.access, requirement.state.layout, requirement.state.sync };

		// Changing state?
		SymbolicTracker* tracker = currentBatch.GetPassBatchTracker(requirement.resourceIndex);
		if (tracker && (requirement.isWholeResource
			? tracker->WouldModifyWholeResourceFast(wantState)
			: tracker->WouldModify(requirement.range, wantState))) {
			return true;
		}
		if (!tracker && batchBuildState.ContainsResource(requirement.resourceIndex)) {
			if (!requirement.isWholeResource) {
				return true;
			}
			if (requirement.resourceIndex < m_frameCompileResources.size()) {
				const auto& compileResource = m_frameCompileResources[requirement.resourceIndex];
				if (!compileResource.fastState.valid
					|| !compileResource.fastState.wholeResourceOnly
					|| !StatesExactlyEqual(compileResource.fastState.state, wantState)) {
					return true;
				}
			}
			else {
				return true;
			}
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

void RenderGraph::RegisterResolvedResourceAlias(
	ResourceIdentifier const& id, std::shared_ptr<Resource> resource) {
	if (!resource) return;

	const auto existing = RequestResourcePtr(id, true);
	const bool dynamicAlias = dynamic_cast<DynamicResource*>(existing.get()) != nullptr ||
		dynamic_cast<DynamicGloballyIndexedResource*>(existing.get()) != nullptr;
	if (dynamicAlias) {
		m_resolvedResourceAliases.insert(id);
		spdlog::info(
			"RenderGraph: enabled dynamic resolver alias '{}' resource={} name='{}'",
			id.ToString(), resource->GetGlobalResourceID(), resource->GetName());
	}
	if (dynamicAlias || m_resolvedResourceAliases.contains(id)) {
		RegisterResource(id, std::move(resource), nullptr);
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
	const char* logicalName = name ? name : "UserQueue";
	const auto existing = m_queueRegistry.FindNamedOwnedSlot(kind, logicalName);
	if (ToUnderlying(existing) != 0xFF) {
		if (m_queueRegistry.GetAutoAssignmentPolicy(existing) != autoAssignmentPolicy) {
			throw std::runtime_error(fmt::format(
				"Queue '{}' was recreated with a different automatic scheduling policy",
				logicalName));
		}
		return existing;
	}
	auto& deviceManager = DeviceManager::GetInstance();
	auto device = deviceManager.GetDevice();
	const rhi::Backend backend = m_backendDevices.empty()
		? rhi::Backend::Null
		: m_backendDevices.front().backend;
	rhi::Queue queue;
	auto result = device.CreateQueue(static_cast<rhi::QueueKind>(kind), logicalName, queue);
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
	return m_queueRegistry.Register(
		{ kind, instance, BackendInstanceId::Primary, backend },
		queue,
		device,
		autoAssignmentPolicy,
		true,
		logicalName);
}

void RenderGraph::SetMinimumAutomaticSchedulingQueues(QueueKind kind, uint8_t count) {
	const size_t kindIndex = static_cast<size_t>(kind);
	const uint8_t clampedCount = kind == QueueKind::Graphics ? (std::max)(uint8_t(1), count) : (std::max)(uint8_t(1), count);
	m_minAutomaticSchedulingQueuesByKind[kindIndex] = clampedCount;

	if (m_queueRegistry.SlotCount() > 0) {
		EnsureMinimumAutomaticSchedulingQueues();
	}
}


} // namespace org
