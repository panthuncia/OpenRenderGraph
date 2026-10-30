#include "Render/RenderGraph/RenderGraph.h"
#include "Render/Runtime/OpenRenderGraphSettings.h"
#include "RenderGraphCompilerState.h"
#include "PersistentLowering.h"
#include "FrameRecording.h"
#include "Render/RenderGraph/InteropAllocator.h"
#include "Render/RenderGraph/ExperimentalRhiExecution.h"

#include <span>
#include <algorithm>
#include <type_traits>
#include <atomic>
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
#include <future>
#include <thread>
#include "Render/Runtime/IUploadService.h"
#include "Render/Runtime/IDescriptorService.h"
#include "Render/Runtime/DescriptorServiceAccess.h"
#include "Render/Runtime/ScopedActiveGraphServices.h"
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
#include "Interfaces/IHasMemoryMetadata.h"
#include "Interfaces/IDynamicDeclaredResources.h"
#include "Resources/DynamicResource.h"
#include "Resources/ExternalTextureResource.h"
#include "Resources/BackedResource.h"
#include "Render/BufferBarrierHelpers.h"
#include "Resources/PixelBuffer.h"
#include "Resources/Buffers/DynamicBufferBase.h"
#include "Resources/MemoryStatisticsComponents.h"


namespace org {

namespace {
thread_local ResolverCaptureScope* t_resolverCaptures = nullptr;
}

ExternalPassDesc ExternalPassDesc::Compute(std::string name, std::shared_ptr<RenderPass> computePass) {
	ExternalPassDesc desc{};
	// PassType remains an ingestion/compiler shape tag during the migration;
	// actual execution queue is selected independently below.
	desc.type = PassType::Render;
	desc.name = std::move(name);
	desc.unifiedPass = std::move(computePass);
	desc.preferredQueueKind = QueueKind::Compute;
	return desc;
}

ExternalPassDesc ExternalPassDesc::Copy(std::string name, std::shared_ptr<RenderPass> copyPass) {
	ExternalPassDesc desc{};
	desc.type = PassType::Render;
	desc.name = std::move(name);
	desc.unifiedPass = std::move(copyPass);
	desc.preferredQueueKind = QueueKind::Copy;
	return desc;
}

ResolverCaptureScope::ResolverCaptureScope() : m_previous(t_resolverCaptures) { t_resolverCaptures = this; }
ResolverCaptureScope::~ResolverCaptureScope() { t_resolverCaptures = m_previous; }

std::shared_ptr<const ResolverDeclarationState> RenderGraph::CaptureResolverDeclarationState(const IResourceResolver& resolver) const {
	auto state = m_resolverCaptureContext ? resolver.CaptureDeclarationState(*m_resolverCaptureContext)
		: resolver.CaptureDeclarationState();
	auto* scope = t_resolverCaptures;
	if (!scope || !state || !state->dependencyIdentity) return state;
	for (const auto& captured : scope->m_captures)
		if (captured->dependencyIdentity == state->dependencyIdentity) return captured;
	scope->m_captures.push_back(state);
	return state;
}

namespace {
std::atomic_bool g_loggedLegacyPacketRecording{false};

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

void MergeResolverWaits(std::vector<ExternalTimelinePoint>& waits,
    const std::vector<ResolverSnapshot>& snapshots)
{
    for (const auto& snapshot : snapshots)
        if (snapshot.state && snapshot.state->waits) waits.insert(waits.end(), snapshot.state->waits->begin(), snapshot.state->waits->end());
    std::sort(waits.begin(), waits.end(), [](const auto& lhs, const auto& rhs) {
        const auto lh = lhs.timeline.GetHandle(); const auto rh = rhs.timeline.GetHandle();
        return lh.index != rh.index ? lh.index < rh.index :
            (lh.generation != rh.generation ? lh.generation < rh.generation : lhs.value < rhs.value);
    });
    std::vector<ExternalTimelinePoint> normalized;
    normalized.reserve(waits.size());
    for (const auto& wait : waits) {
        const auto handle = wait.timeline.GetHandle();
        if (!normalized.empty()) {
            const auto previous = normalized.back().timeline.GetHandle();
            if (previous.index == handle.index && previous.generation == handle.generation) {
                normalized.back().value = (std::max)(normalized.back().value, wait.value);
                continue;
            }
        }
        normalized.push_back(wait);
    }
    waits = std::move(normalized);
}

template<class PassResourceData>
std::unique_ptr<ResourceRegistryView> MakePassResourceRegistryView(
	ResourceRegistry& registry, const PassResourceData& resources)
{
	auto view = std::make_unique<ResourceRegistryView>(registry, resources.identifierSet);
	for (const auto& requirement : resources.staticResourceRequirements) {
		view->AllowConcreteResourceID(requirement.resourceHandleAndRange.resource.GetGlobalResourceID());
	}
	for (const auto& requirement : resources.frameResourceRequirements)
		view->AllowConcreteResourceID(requirement.resourceHandleAndRange.resource.GetGlobalResourceID());
	for (const auto& block : resources.resolverRequirementBlocks) {
		view->AllowConcretePredicate([slot = &block](uint64_t id) {
			if (!slot || !*slot) return false;
			return std::ranges::any_of((*slot)->requirements, [id](const auto& requirement) {
				return requirement.resourceHandleAndRange.resource.GetGlobalResourceID() == id;
			});
		});
	}
	for (const auto& transition : resources.internalTransitions)
		view->AllowConcreteResourceID(transition.first.resource.GetGlobalResourceID());
	return view;
}
}

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
		auto rp = d.unifiedPass
			? std::dynamic_pointer_cast<RenderPass>(d.unifiedPass)
			: std::get<std::shared_ptr<RenderPass>>(d.pass);
		if (!rp) {
			throw std::logic_error("Unified pass does not implement the unified render-graph pass contract");
		}
		RenderPassAndResources par;
		par.pass = rp;
		par.name = d.name;
		par.techniquePath = d.techniquePath;
		par.collectStatistics = d.collectStatistics;
		par.epoch = d.epoch;
		par.recordAtExecution = d.recordAtExecution;
		{
			RenderPassBuilder b(this, d.name);
			b.pass = rp;
			b.built_ = true;
			b.params = {};
			b.params.isGeometryPass = d.isGeometryPass;
			b._declaredIds.clear();
			if (traceLifecycle) {
				spdlog::info("RG materialize external render pass '{}' declare begin", d.name);
			}
			EnsureProviderRegistered(par.pass.get());
			{ const ResolverCaptureScope captures; rp->DeclareResourceUsages(&b); }
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
			par.resolverSnapshots = b.TakeResolverSnapshots();
			par.resolverGroupPlan = lowering::PlanResolverGroups(par);
		}

		if (callSetup) {
			par.pass->ConfigureRuntimeServices(m_uploadService, m_descriptorService);
			par.pass->ConfigureResourceRegistryView(
				MakePassResourceRegistryView(_registry, par.resources), par.resources);
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
		par.pass = cp;
		par.name = d.name;
		par.techniquePath = d.techniquePath;
		par.collectStatistics = d.collectStatistics;
		par.epoch = d.epoch;
		par.recordAtExecution = d.recordAtExecution;
		{
			ComputePassBuilder b(this, d.name);
			b.pass = cp;
			b.built_ = true;
			b.params = {};
			b._declaredIds.clear();
			if (traceLifecycle) {
				spdlog::info("RG materialize external compute pass '{}' declare begin", d.name);
			}
			EnsureProviderRegistered(par.pass.get());
			{ const ResolverCaptureScope captures; cp->DeclareResourceUsages(&b); }
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
			par.resolverSnapshots = b.TakeResolverSnapshots();
			par.resolverGroupPlan = lowering::PlanResolverGroups(par);
		}

		if (callSetup) {
			par.pass->ConfigureRuntimeServices(m_uploadService, m_descriptorService);
			par.pass->ConfigureResourceRegistryView(
				MakePassResourceRegistryView(_registry, par.resources), par.resources);
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
		par.pass = cp;
		par.name = d.name;
		par.techniquePath = d.techniquePath;
		par.collectStatistics = d.collectStatistics;
		par.epoch = d.epoch;
		par.recordAtExecution = d.recordAtExecution;
		{
			CopyPassBuilder b(this, d.name);
			b.pass = cp;
			b.built_ = true;
			b.params = {};
			b._declaredIds.clear();
			if (traceLifecycle) {
				spdlog::info("RG materialize external copy pass '{}' declare begin", d.name);
			}
			EnsureProviderRegistered(par.pass.get());
			{ const ResolverCaptureScope captures; cp->DeclareResourceUsages(&b); }
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
			par.resolverSnapshots = b.TakeResolverSnapshots();
			par.resolverGroupPlan = lowering::PlanResolverGroups(par);
		}

		if (callSetup) {
			par.pass->ConfigureRuntimeServices(m_uploadService, m_descriptorService);
			par.pass->ConfigureResourceRegistryView(
				MakePassResourceRegistryView(_registry, par.resources), par.resources);
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
			renderPassesByName[d.name] = std::dynamic_pointer_cast<RenderPass>(rp.pass);
		}
	}
	else if (d.type == PassType::Compute) {
		auto& cp = std::get<ComputePassAndResources>(any.pass);
		if (!d.name.empty()) {
			computePassesByName[d.name] = std::dynamic_pointer_cast<ComputePass>(cp.pass);
		}
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
	for (auto& entry : m_masterPassList) {
		std::visit([&](auto& obj) {
			using T = std::decay_t<decltype(obj)>;
			if constexpr (!std::is_same_v<T, std::monostate>) {
				if (obj.pass) obj.pass->ConfigureRuntimeServices(m_uploadService, m_descriptorService);
			}
		}, entry.pass);
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
	ShutdownOwnedState();
}

void RenderGraph::StopFrameProduction() {
    BT_ZONE_SCOPE("ORG.Frame.StopProduction");
    m_compilerState->frameProductionStopped = true;
    if (m_compilerState->persistentBuildScope) m_compilerState->persistentBuildScope->CancelAndWait();
    m_compilerState->persistentBuildScope.reset();
    if (m_compilerState->frameWorkerScope) m_compilerState->frameWorkerScope->CancelAndWait();
    m_compilerState->frameWorkerScope.reset();
    if (m_compilerState->ownershipRetirementScope) m_compilerState->ownershipRetirementScope->Wait();
    m_compilerState->ownershipRetirementScope.reset();
}

void RenderGraph::ShutdownTaskWorkers() {
    StopFrameProduction();
	m_queueRegistry.ShutdownTaskWorkers();
	m_taskService.reset();
	org::runtime::SetDefaultTaskService({});
}

void RenderGraph::ShutdownRuntime() {
	StatisticsManager::GetInstance().ClearAll();
	UploadManager::GetInstance().Cleanup(); // its pages and copy queue must not outlive the device (see ShutdownRuntimeDevice)
	DeletionManager::GetInstance().DrainAll();
	// Resource destruction above and during host teardown can enqueue further
	// descriptor/backing owners. Global arenas outlive all of those owners.
	DescriptorHeapManager::GetInstance().Cleanup();
	DeletionManager::GetInstance().DrainAll();
	DeletionManager::GetInstance().Cleanup();
	DeviceManager::GetInstance().Cleanup();
}

void RenderGraph::ShutdownOwnedState() {
	ShutdownTaskWorkers();
    m_resolverCaptureContext.reset();
	m_masterPassList.clear();
	m_compilerState->frameExtensions.clear();
	renderPassesByName.clear();
	computePassesByName.clear();
	resourcesByID.clear();
	resourcesByName.clear();
	m_transientFrameResourcesByID.clear();
	m_dynamicResourcesByStableID.clear();
	m_transientFrameResourcesByName.clear();

	m_passBuilderOrder.clear();
	m_passNamesSeenThisReset.clear();
	m_passBuildersByName.clear();
	m_extensions.clear();
    m_extensionRegistrationIds.clear();
	_providerMap.clear();
	_providers.clear();
	_resolverMap.clear();
	_registry = ResourceRegistry();
	++m_resourceRegistryGeneration;
	m_resolverHandleCache.clear();

	m_queueRegistry.Clear();

	m_readbackFence.Reset();
	m_copyReadbackFence.Reset();

	m_statisticsService.reset();
	// Keep service generations alive across graph rebuilds; existing pass
	// objects may still be initialized before the successor setup boundary.
	m_readbackService.reset();
	m_renderGraphSettingsService.reset();
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
	// A structural replacement is a generation boundary for every accepted frame. The caller has already made all devices
	// idle: join CPU users, then drop the persistent program with its executions, bindings and alias pools. The next frame
	// bootstraps a new one over the rebuilt pass list.
	StopFrameProduction();
	m_compilerState->persistent.reset();
	m_compilerState->invocationArenas.clear();
	m_compilerState->frameProductionStopped = false;
	basic_telemetry::AddCounter("ORG.Frame.GenerationBoundaryResets");

	m_masterPassList.clear();
	ResetStructuralBuildState();

	resourcesByID.clear();
	resourcesByName.clear();
	m_transientFrameResourcesByID.clear();
	m_dynamicResourcesByStableID.clear();
	m_transientFrameResourcesByName.clear();
	// Queue slots, their command-list pools, and timelines are runtime-owned and
	// survive structural graph generations. They are released only at shutdown.
	renderPassesByName.clear();
	computePassesByName.clear();

	_providerMap.clear();
	_providers.clear();
	_resolverMap.clear();
	_registry = ResourceRegistry();
	++m_resourceRegistryGeneration;
	m_resolverHandleCache.clear();

	// Notify extensions that the registry was replaced
	for (auto& ext : m_extensions) {
		if (ext) ext->OnRegistryReset(&_registry);
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
	BT_ZONE_SCOPE("RenderGraph::ResetForFrame::ReclaimExpiredAnonymous");
	_registry.ReclaimExpiredAnonymous();
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
			if (std::holds_alternative<std::monostate>(d.pass) && !d.unifiedPass) continue;

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
	m_masterPassList.reserve(baseIdx.size() + extIdx.size());

	for (size_t u : topo) {
		if (!nodes[u].hasPass) {
			continue;
		}
		m_masterPassList.push_back(std::move(nodes[u].pass));
	}
	spdlog::debug("RenderGraph structural merge: emitted {} passes", m_masterPassList.size());
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


void RenderGraph::RedeclarePass(PassAndResources& p, PassType type)
{
	BT_ZONE_SCOPE("RenderGraph::RedeclarePass");
	if (!p.name.empty()) { BT_ZONE_TEXT(p.name.data(), p.name.size()); }
	auto redeclare = [&]<class Builder, class Pass>(std::type_identity<Builder>, std::type_identity<Pass>, const char* kind) {
		auto pass = std::dynamic_pointer_cast<Pass>(p.pass);
		if (!pass)
			throw std::logic_error(std::string(kind) + "-tagged pass '" + p.name + "' does not implement its declaration interface");
		// Enough of a built builder for pass code that queries ResourceProvider().
		Builder b(this, p.name);
		b.pass = pass;
		b.built_ = true;
		EnsureProviderRegistered(p.pass.get());
		{
			BT_ZONE_SCOPE("RenderGraph::RedeclarePass::DeclareResourceUsages");
			const ResolverCaptureScope captures;
			pass->DeclareResourceUsages(&b);
		}
		auto& resources = p.resources;
		resources.staticResourceRequirements = b.GatherResourceRequirements();
		resources.resolverRequirementBlocks.clear();
		resources.mergedFrameRequirementsDirty = true;
		resources.internalTransitions = std::move(b.params.internalTransitions);
		resources.identifierSet = std::move(b._declaredIds);
		if constexpr (!std::is_same_v<Builder, CopyPassBuilder>) {
			resources.autoDescriptorShaderResources = std::move(b.params.autoDescriptorShaderResources);
			resources.autoDescriptorConstantBuffers = std::move(b.params.autoDescriptorConstantBuffers);
			resources.autoDescriptorUnorderedAccessViews = std::move(b.params.autoDescriptorUnorderedAccessViews);
			resources.activeFeatureDomains = std::move(b.params.activeFeatureDomains);
		}
		p.explicitExternalWaitsBeforeTransitions = b.params.externalWaitsBeforeTransitions;
		resources.externalWaitsBeforeTransitions = std::move(b.params.externalWaitsBeforeTransitions);
		resources.externalWaitBindingsBeforeTransitions = std::move(b.params.externalWaitBindingsBeforeTransitions);
		{
			BT_ZONE_SCOPE("RenderGraph::RedeclarePass::MaterializeReferencedResources");
			MaterializeReferencedResources(resources.staticResourceRequirements, resources.internalTransitions);
		}
		p.resolverSnapshots = b.TakeResolverSnapshots();
		MergeResolverWaits(resources.externalWaitsBeforeTransitions, p.resolverSnapshots);
		p.retainedAnonymousKeepAlive = CaptureRetainedAnonymousKeepAlive(resources.staticResourceRequirements, resources.internalTransitions);
		p.resolverGroupPlan = lowering::PlanResolverGroups(p);
		// A versioned resolver can replace the concrete resource behind an unchanged identifier: the view and Setup are
		// rebuilt so automatic descriptor bindings follow it. Immediate-only upload and readback passes consume the
		// resources captured in their bytecode and opt out.
		const auto* dynamic = dynamic_cast<IDynamicDeclaredResources*>(p.pass.get());
		const bool rebind = (!std::is_same_v<Builder, CopyPassBuilder> && !p.resolverSnapshots.empty())
			|| !dynamic || dynamic->RequiresPassRebindAfterDeclarationRefresh();
		if (rebind) {
			BT_ZONE_SCOPE("RenderGraph::RedeclarePass::Rebind");
			p.pass->ConfigureRuntimeServices(m_uploadService, m_descriptorService);
			p.pass->ConfigureResourceRegistryView(MakePassResourceRegistryView(_registry, resources), resources);
			p.pass->Setup();
		}
	};
	p.resolverGroupPlan.reset();
	switch (type) {
	case PassType::Render: redeclare(std::type_identity<RenderPassBuilder>{}, std::type_identity<RenderPass>{}, "Render"); break;
	case PassType::Compute: redeclare(std::type_identity<ComputePassBuilder>{}, std::type_identity<ComputePass>{}, "Compute"); break;
	case PassType::Copy: redeclare(std::type_identity<CopyPassBuilder>{}, std::type_identity<CopyPass>{}, "Copy"); break;
	default: throw std::logic_error("Pass '" + p.name + "' has no declaration interface");
	}
}

void RenderGraph::MaterializeUnmaterializedResources() {
	BT_ZONE_SCOPE("RenderGraph::MaterializeUnmaterializedResources");
	// Rebuilds may have retired the prior service generation before structural
	// materialization runs. Establish the successor generation before any pass
	// Setup/configuration callback can resolve descriptor or upload services.
	if (!m_uploadService) m_uploadService = org::runtime::CreateDefaultUploadService();
	if (!m_descriptorService) m_descriptorService = org::runtime::CreateDefaultDescriptorService();
	// Every registered and transient resource; an alias-eligible one waits for the persistent alias plan, which places it.
	auto materialize = [](Resource* resource) {
		resource = UnwrapDynamicResource(resource);
		if (auto* texture = dynamic_cast<PixelBuffer*>(resource)) {
			if (!texture->IsMaterialized() && !texture->GetDescription().allowAlias) texture->Materialize();
		} else if (auto* buffer = dynamic_cast<BufferBase*>(resource)) {
			if (!buffer->IsMaterialized() && !buffer->IsAliasingAllowed()) buffer->Materialize();
		}
	};
	for (auto& [id, resource] : resourcesByID)
		if (resource) materialize(resource.get());
	for (auto& [id, resource] : m_transientFrameResourcesByID)
		if (resource && !resourcesByID.contains(id)) materialize(resource.get());
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
}

void RenderGraph::Setup() {
	// Establish service generations before any pass initialization or setup
	// callback. Rebuilds may have retired the prior generation.
	if (!m_uploadService) m_uploadService = org::runtime::CreateDefaultUploadService();
	if (!m_descriptorService) m_descriptorService = org::runtime::CreateDefaultDescriptorService();
	if (!m_readbackService) m_readbackService = org::runtime::CreateDefaultReadbackService();
	if (!m_statisticsService) m_statisticsService = org::runtime::CreateDefaultStatisticsService();
	org::runtime::ScopedActiveGraphServices activeServices(m_uploadService.get(), m_descriptorService.get());
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
		// ClearAll above dropped every registration, but passes keep their old index. A graph compiled before
		// Setup (the persistent host: CompileStructural, then Setup) would then never register again and record
		// no timestamps at all - the query heap is sized from zero passes. Re-register what is compiled.
		for (auto& any : m_masterPassList)
			std::visit([](auto& pass) { if constexpr (!std::is_same_v<std::decay_t<decltype(pass)>, std::monostate>) pass.statisticsIndex = -1; }, any.pass);
		m_statisticsService->SetupQueryHeap();
	}

	auto device = DeviceManager::GetInstance().GetDevice();

	auto result = device.CreateTimeline(m_readbackFence);
	result = device.CreateTimeline(m_copyReadbackFence);

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

	// Settings the persistent compiler and its diagnostics read, live from the settings service.
	auto& settings = m_renderGraphSettingsService;
	m_getRenderGraphCompileDumpEnabled = [&settings] { return settings ? settings->GetRenderGraphCompileDumpEnabled() : false; };
	m_getRenderGraphVramDumpEnabled = [&settings] { return settings ? settings->GetRenderGraphVramDumpEnabled() : false; };
	m_getRenderGraphBatchTraceEnabled = [&settings] { return settings ? settings->GetRenderGraphBatchTraceEnabled() : false; };
	m_getAutoAliasMode = [&settings] {
		return static_cast<AutoAliasMode>(settings ? settings->GetAutoAliasMode() : static_cast<uint8_t>(AutoAliasMode::Off));
	};
	m_getAutoAliasPackingStrategy = [&settings] {
		return static_cast<AutoAliasPackingStrategy>(settings
			? settings->GetAutoAliasPackingStrategy() : static_cast<uint8_t>(AutoAliasPackingStrategy::GreedySweepLine));
	};
	m_getAutoAliasEnableLogging = [&settings] { return settings ? settings->GetAutoAliasEnableLogging() : false; };
	m_getAutoAliasLogExclusionReasons = [&settings] { return settings ? settings->GetAutoAliasLogExclusionReasons() : false; };
	m_getAutoAliasBuildDebugData = [&settings] { return settings ? settings->GetAutoAliasBuildDebugData() : false; };
	m_getAutoAliasPoolRetireIdleFrames = [&settings] { return settings ? settings->GetAutoAliasPoolRetireIdleFrames() : 120u; };
	m_getAutoAliasPoolGrowthHeadroom = [&settings] { return settings ? settings->GetAutoAliasPoolGrowthHeadroom() : 1.5f; };
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
			renderPass.pass->ConfigureRuntimeServices(m_uploadService, m_descriptorService);
			if (traceLifecycle) {
				spdlog::info("RG setup render pass '{}' begin", renderPass.name);
			}
			renderPass.pass->ConfigureResourceRegistryView(
				MakePassResourceRegistryView(_registry, renderPass.resources), renderPass.resources);
			renderPass.pass->Setup();
			if (traceLifecycle) {
				spdlog::info("RG setup render pass '{}' complete", renderPass.name);
			}
			break;
		}
		case PassType::Compute: {
			auto& computePass = std::get<ComputePassAndResources>(pass.pass);
			computePass.pass->ConfigureRuntimeServices(m_uploadService, m_descriptorService);
			if (traceLifecycle) {
				spdlog::info("RG setup compute pass '{}' begin", computePass.name);
			}
			computePass.pass->ConfigureResourceRegistryView(
				MakePassResourceRegistryView(_registry, computePass.resources), computePass.resources);
			computePass.pass->Setup();
			if (traceLifecycle) {
				spdlog::info("RG setup compute pass '{}' complete", computePass.name);
			}
			break;
		}
		case PassType::Copy: {
			auto& copyPass = std::get<CopyPassAndResources>(pass.pass);
			copyPass.pass->ConfigureRuntimeServices(m_uploadService, m_descriptorService);
			if (traceLifecycle) {
				spdlog::info("RG setup copy pass '{}' begin", copyPass.name);
			}
			copyPass.pass->ConfigureResourceRegistryView(
				MakePassResourceRegistryView(_registry, copyPass.resources), copyPass.resources);
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
	passAndResources.explicitExternalWaitsBeforeTransitions = resources.externalWaitsBeforeTransitions;
	MergeResolverWaits(passAndResources.resources.externalWaitsBeforeTransitions, passAndResources.resolverSnapshots);
	passAndResources.resolverGroupPlan = lowering::PlanResolverGroups(passAndResources);
	AnyPassAndResources passAndResourcesAny;
	passAndResourcesAny.type = PassType::Render;
	passAndResourcesAny.pass = std::move(passAndResources);
	passAndResourcesAny.name = name;
	m_masterPassList.push_back(std::move(passAndResourcesAny));
	if (name != "") {
		renderPassesByName[name] = std::move(pass);
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
	passAndResources.explicitExternalWaitsBeforeTransitions = resources.externalWaitsBeforeTransitions;
	MergeResolverWaits(passAndResources.resources.externalWaitsBeforeTransitions, passAndResources.resolverSnapshots);
	passAndResources.resolverGroupPlan = lowering::PlanResolverGroups(passAndResources);
	AnyPassAndResources passAndResourcesAny;
	passAndResourcesAny.type = PassType::Compute;
	passAndResourcesAny.pass = std::move(passAndResources);
	passAndResourcesAny.name = name;
	m_masterPassList.push_back(std::move(passAndResourcesAny));
	if (name != "") {
		computePassesByName[name] = std::move(pass);
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
	passAndResources.explicitExternalWaitsBeforeTransitions = resources.externalWaitsBeforeTransitions;
	MergeResolverWaits(passAndResources.resources.externalWaitsBeforeTransitions, passAndResources.resolverSnapshots);
	passAndResources.resolverGroupPlan = lowering::PlanResolverGroups(passAndResources);
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
    // A task service without a settings service is a structural/unit host: it admits no executable frames.
    if (m_taskService && !m_renderGraphSettingsService) return;
    UpdateOnPreparationOwner(context, device);
}

void RenderGraph::UpdateOnPreparationOwner(const UpdateExecutionContext& context, rhi::Device device) {
	BT_ZONE_SCOPE("RenderGraph::Update");
    if (m_compilerState->frameProductionStopped)
        throw std::logic_error("Frame production has stopped");
    m_resolverCaptureContext = context.resolverCaptureContext;
	m_asyncUpdateHostData = context.ownedHostData;
	// Graph replacement tears down the previous generation's services before the
	// next frame is admitted. Recreate the generation here before pass setup so
	// every pass receives a live owned service pair; otherwise a rebuilt graph can
	// enter Setup with empty weak service references.
	if (!m_uploadService) m_uploadService = org::runtime::CreateDefaultUploadService();
	if (!m_descriptorService) m_descriptorService = org::runtime::CreateDefaultDescriptorService();
	if (!m_readbackService) m_readbackService = org::runtime::CreateDefaultReadbackService();
	if (!m_statisticsService) m_statisticsService = org::runtime::CreateDefaultStatisticsService();
	// A graph rebuild may replace the service generation after passes were
	// registered. Refresh every pass' weak service bindings before Update/Setup
	// callbacks run so they cannot retain an expired generation.
	for (auto& pr : m_masterPassList) {
		std::visit([&](auto& obj) {
			using T = std::decay_t<decltype(obj)>;
			if constexpr (!std::is_same_v<T, std::monostate>) {
				if (obj.pass) obj.pass->ConfigureRuntimeServices(m_uploadService, m_descriptorService);
			}
		}, pr.pass);
	}
	org::runtime::ScopedActiveGraphServices activeServices(m_uploadService.get(), m_descriptorService.get());
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
				else if (RunsInPersistentEpoch(obj.epoch)) {
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
		// Registration above may have added passes. The persistent recorder only looks passes up by name, so
		// this is the one place its query heap gets sized: before it, every lookup found a pass with no query
		// slots, and no persistent graph ever recorded a timestamp. A no-op when the capacity already fits.
		if (m_statisticsService)
			m_statisticsService->SetupQueryHeap();
	}

	if (context.beforeCompileFrame) {
		BT_ZONE_SCOPE("RenderGraph::Update::BeforeCompileFrame");
		context.beforeCompileFrame();
	}

	{
		BT_ZONE_SCOPE("RenderGraph::Update::CompileFrame");
		CompileFrame(device, static_cast<uint8_t>(context.preparationSlot), context.hostData);
	}
}

void RenderGraph::CompileFrame(rhi::Device device, uint8_t frameIndex, const IHostExecutionData* hostData) {
	PrepareAndCompileFrame(device, frameIndex, hostData);
}

void RenderGraph::PrepareAndCompileFrame(rhi::Device device, uint8_t frameIndex,
	const IHostExecutionData* hostData, float deltaTime) {
	BT_ZONE_SCOPE("ORG.Frame.PrepareAndCompile");
	if (m_compilerState->frameProductionStopped)
		throw std::logic_error("Frame production has stopped");
	// Persistent structural builds, preparation and ownership retirement run on the task service; there is no inline fallback.
	if (!m_taskService) throw std::logic_error("Persistent execution requires a task service");
	PreparePersistentFrame(device, frameIndex, hostData, deltaTime);
}

void RenderGraph::ConfirmPresentationTailSubmission() {
	if (m_compilerState->persistent) ConfirmPersistentPresentationTail();
}

void RenderGraph::Execute(PassExecutionContext& context) {
	if (!context.beginGpuPassRange && !context.endGpuPassRange && m_gpuPassRangeBegin && m_gpuPassRangeEnd) {
		context.beginGpuPassRange = m_gpuPassRangeBegin;
		context.endGpuPassRange = m_gpuPassRangeEnd;
	}
	if (m_compilerState->frameProductionStopped)
		throw std::logic_error("Frame production has stopped");
	if (!m_compilerState->persistent) throw std::logic_error("RenderGraph::Execute before a persistent frame was prepared");
	try { ExecutePersistentFrame(context); }
	catch (...) { StopFrameProduction(); throw; }
}

std::optional<uint32_t> RenderGraph::GetLastExecutedPreparationSlot() const noexcept {
    return m_compilerState ? m_compilerState->lastExecutedPreparationSlot : std::nullopt;
}

std::shared_ptr<const IHostExecutionData> RenderGraph::GetLastSubmittedFrameData() const noexcept {
    return m_compilerState ? m_compilerState->lastSubmittedFrameData : nullptr;
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

std::shared_ptr<const std::vector<ResourceHandleAndRange>>
RenderGraph::RequestResolverResourceHandles(const ResolverDeclarationState& state) {
	BT_ZONE_SCOPE("RenderGraph::RequestResolverResourceHandles");
	const auto identity = state.dependencyIdentity.get();
	if (identity) {
		const auto found = m_resolverHandleCache.find(identity);
		if (found != m_resolverHandleCache.end() &&
			found->second.registryGeneration == m_resourceRegistryGeneration &&
			found->second.resourceSetIdentity == state.resourceSetIdentity) {
			++m_resolverHandleCacheHitsThisFrame;
			return found->second.handles;
		}
	}

	++m_resolverHandleCacheMissesThisFrame;
	auto handles = std::make_shared<std::vector<ResourceHandleAndRange>>();
	if (state.resources) {
		handles->reserve(state.resources->size());
		for (const auto& resource : *state.resources) {
			// Resolver declarations, like direct pointer declarations, are authorized
			// by their concrete registry handles.  Do not promote them into graph-owned
			// resources: published resolver sets may legitimately contain externally
			// managed immutable resources.
			handles->push_back(ResourceHandleAndRange{ RequestResourceHandle(resource.get()), {} });
		}
	}

	if (identity) {
		m_resolverHandleCache.insert_or_assign(identity, ResolverHandleCacheEntry{
			.dependencyIdentity = state.dependencyIdentity,
			.resourceSetIdentity = state.resourceSetIdentity,
			.registryGeneration = m_resourceRegistryGeneration,
			.handles = handles
		});
	}
	return handles;
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
	if (result == rhi::Result::Unsupported) {
		// The device has no further queue of this kind (a Vulkan family's queue count): share the last one of the kind, so the
		// work serializes there instead of failing the graph.
		for (size_t i = m_queueRegistry.SlotCount(); i-- > 0;) {
			const auto slot = static_cast<QueueSlotIndex>(static_cast<uint8_t>(i));
			if (m_queueRegistry.GetKind(slot) != kind) continue;
			spdlog::warn("Queue '{}' (kind {}): the device has no further queue of this kind; sharing queue slot {}",
				logicalName, static_cast<int>(kind), i);
			return slot;
		}
	}
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
