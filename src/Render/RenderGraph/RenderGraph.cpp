#include "Render/RenderGraph/RenderGraph.h"

#include <span>
#include <algorithm>
#include <cmath>
#include <cctype>
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

#include "Render/PassExecutionContext.h"
#include "Utilities/ORGUtilities.h"
#include "Managers/Singletons/DeviceManager.h"
#include "Managers/Singletons/DeletionManager.h"
#include "Managers/Singletons/UploadManager.h"
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
				rhi::ResourceAccessType::DepthRead |
				rhi::ResourceAccessType::DepthReadWrite;
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
				layout != rhi::ResourceLayout::DepthReadWrite &&
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

	bool SubresourceRangesOverlap(const SubresourceRange& lhs, const SubresourceRange& rhs) noexcept {
		if (lhs.isEmpty() || rhs.isEmpty()) {
			return false;
		}

		const uint32_t lhsMipEnd = lhs.firstMip + lhs.mipCount;
		const uint32_t rhsMipEnd = rhs.firstMip + rhs.mipCount;
		const uint32_t lhsSliceEnd = lhs.firstSlice + lhs.sliceCount;
		const uint32_t rhsSliceEnd = rhs.firstSlice + rhs.sliceCount;

		return lhs.firstMip < rhsMipEnd
			&& rhs.firstMip < lhsMipEnd
			&& lhs.firstSlice < rhsSliceEnd
			&& rhs.firstSlice < lhsSliceEnd;
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
			par.resources.frameResourceRequirements = par.resources.staticResourceRequirements;
			par.resources.internalTransitions = b.params.internalTransitions;
			par.resources.identifierSet = b.DeclaredResourceIds();
			par.resources.autoDescriptorShaderResources = b.params.autoDescriptorShaderResources;
			par.resources.autoDescriptorConstantBuffers = b.params.autoDescriptorConstantBuffers;
			par.resources.autoDescriptorUnorderedAccessViews = b.params.autoDescriptorUnorderedAccessViews;
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
		}

		if (callSetup) {
			par.pass->SetResourceRegistryView(
				std::make_unique<ResourceRegistryView>(_registry, par.resources.identifierSet),
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
			par.resources.frameResourceRequirements = par.resources.staticResourceRequirements;
			par.resources.internalTransitions = b.params.internalTransitions;
			par.resources.identifierSet = b.DeclaredResourceIds();
			par.resources.autoDescriptorShaderResources = b.params.autoDescriptorShaderResources;
			par.resources.autoDescriptorConstantBuffers = b.params.autoDescriptorConstantBuffers;
			par.resources.autoDescriptorUnorderedAccessViews = b.params.autoDescriptorUnorderedAccessViews;
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
		}

		if (callSetup) {
			par.pass->SetResourceRegistryView(
				std::make_unique<ResourceRegistryView>(_registry, par.resources.identifierSet),
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
			par.resources.frameResourceRequirements = par.resources.staticResourceRequirements;
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

void RenderGraph::WriteCompiledGraphDebugDump(uint8_t frameIndex, const std::vector<RenderGraph::Node>& nodes, std::string_view dumpVariant) const
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

		std::ostringstream dump;
		const std::string variantLabel = dumpVariant.empty() ? std::string("latest") : std::string(dumpVariant);

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
			if (passIndex < m_compiledFrameProgram.passes.size()) {
				const auto& passIR = m_compiledFrameProgram.passes[passIndex];
				dump << " pass_schedule_hash=" << passIR.scheduleHash
					 << " pass_barrier_hash=" << passIR.barrierHash
					 << " pass_execution_hash=" << passIR.executionHash
					 << " pass_full_hash=" << passIR.fullHash;
			}
			if (resources.pinnedQueueSlot.has_value()) {
				dump << " pinned_queue=" << static_cast<unsigned int>(static_cast<uint8_t>(*resources.pinnedQueueSlot));
			}
			if constexpr (requires { resources.isGeometryPass; }) {
				dump << " geometry_pass=" << (resources.isGeometryPass ? "true" : "false");
			}
			dump << " declared_requirements=" << resources.frameResourceRequirements.size()
				 << " internal_transitions=" << resources.internalTransitions.size()
				 << "\n";

			if (!resources.frameResourceRequirements.empty()) {
				dump << "  requirements:\n";
				for (const auto& req : resources.frameResourceRequirements) {
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
		dump << "dump_variant=" << variantLabel << "\n";
		dump << "pass_count=" << m_framePasses.size()
			 << " node_count=" << nodes.size()
			 << " batch_count=" << batches.size()
			 << " queue_slot_count=" << m_queueRegistry.SlotCount() << "\n";
		dump << "frame_program_passes=" << m_compileCacheStats.passIRCount
			 << " normalized_accesses=" << m_compileCacheStats.normalizedAccessCount
			 << " frame_schedule_hash=" << m_compiledFrameProgram.scheduleContentHash
			 << " frame_barrier_hash=" << m_compiledFrameProgram.barrierContentHash
			 << " frame_execution_hash=" << m_compiledFrameProgram.executionContentHash
			 << " barrier_transition_count=" << m_compiledBarrierIR.transitionCount
			 << " barrier_wait_count=" << m_compiledBarrierIR.waitCount
			 << " barrier_lowered_requirements=" << m_compiledBarrierIR.loweredRequirementCount
			 << " resource_access_chains=" << m_resourceAccessChains.size()
			 << " dependency_edge_ir_count=" << m_dependencyEdgeIR.size()
			 << " pass_ir_cache_hits=" << m_compileCacheStats.passIRCacheHits
			 << " pass_ir_cache_misses=" << m_compileCacheStats.passIRCacheMisses
			 << " schedule_cache_hits=" << m_compileCacheStats.scheduleCacheHits
			 << " schedule_cache_misses=" << m_compileCacheStats.scheduleCacheMisses
			 << " schedule_relaxed_cache_hits=" << m_compileCacheStats.scheduleRelaxedCacheHits
			 << " schedule_reused_passes=" << m_compileCacheStats.scheduleReusedPassCount
			 << " volatile_schedule_pass_changes=" << m_compileCacheStats.volatileSchedulePassChanges
			 << " nonvolatile_schedule_pass_changes=" << m_compileCacheStats.nonVolatileSchedulePassChanges
			 << " segments=" << m_compileCacheStats.segmentCount
			 << " lowered_requirements=" << m_compileCacheStats.loweredRequirementCount
			 << " barrier_segment_cache_hits=" << m_compileCacheStats.barrierSegmentCacheHits
			 << " barrier_segment_cache_misses=" << m_compileCacheStats.barrierSegmentCacheMisses
			 << " barrier_segment_reused=" << m_compileCacheStats.barrierSegmentReused
			 << " barrier_segment_recompiled=" << m_compileCacheStats.barrierSegmentRecompiled
			 << " barrier_segment_reusable_lowered_requirements=" << m_compileCacheStats.barrierSegmentReusableLoweredRequirements
			 << " planned_replay_segments=" << m_compileCacheStats.plannedReplaySegmentCount
			 << " planned_lower_segments=" << m_compileCacheStats.plannedLowerSegmentCount
			 << " planned_replay_lowered_requirements=" << m_compileCacheStats.plannedReplayLoweredRequirements
			 << " planned_lowered_requirements=" << m_compileCacheStats.plannedLoweredRequirements
			 << " materialized_replay_segments=" << m_compileCacheStats.materializedReplaySegmentCount
			 << " materialized_lower_segments=" << m_compileCacheStats.materializedLowerSegmentCount
			 << "\n";
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

		if (m_lastScheduleCacheKeyParts.has_value()) {
			const auto& parts = *m_lastScheduleCacheKeyParts;
			dump << "schedule_key_parts"
				 << " full=" << parts.fullHash
				 << " frame_structure=" << parts.frameStructureHash
				 << " frame_schedule=" << parts.frameScheduleHash
				 << " queue_config=" << parts.queueConfigHash
				 << " active_queues=" << parts.activeQueueHash
				 << " settings=" << parts.settingsHash
				 << " node_topology=" << parts.nodeTopologyHash
				 << " dependency_edges=" << parts.dependencyEdgeHash
				 << "\n\n";
		}

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

			if (!node.accessByID.empty()) {
				dump << "  access_by_id=[";
				for (size_t i = 0; i < node.accessByID.size(); ++i) {
					if (i != 0) {
						dump << ", ";
					}
					const auto& [resourceID, accessKind] = node.accessByID[i];
					dump << resourceID;
					auto resourceIt = resourcesByID.find(resourceID);
					if (resourceIt != resourcesByID.end() && resourceIt->second && !resourceIt->second->GetName().empty()) {
						dump << ":\"" << resourceIt->second->GetName() << "\"";
					}
					dump << ":" << (accessKind == AccessKind::Read ? "Read" : "Write");
				}
				dump << "]\n";
			}
		}

		if (!m_compiledSegments.empty()) {
			dump << "\n[CompiledSegments]\n";
			for (size_t segmentIndex = 0; segmentIndex < m_compiledSegments.size(); ++segmentIndex) {
				const auto& segment = m_compiledSegments[segmentIndex];
				const size_t replayExternalWaitCount = std::count_if(
					segment.replay.waits.begin(),
					segment.replay.waits.end(),
					[](const CachedBarrierWaitOp& wait) { return !wait.sourceSignalInSegment; });
				dump << "[" << segmentIndex << "]"
					 << " name=\"" << segment.name << "\""
					 << " first_pass_stream_index=" << segment.firstPassStreamIndex
					 << " pass_count=" << segment.passCount
					 << " batch_range=[" << segment.firstBatch << "," << segment.lastBatch << "]"
					 << " structure_hash=" << segment.segmentStructureHash
					 << " pass_content_hash=" << segment.passContentHash
					 << " alias_hash=" << segment.aliasSignatureHash
					 << " queue_hash=" << segment.queueAssignmentHash
					 << " entry_state_hash=" << segment.entryStateHash
					 << " exit_state_hash=" << segment.exitStateHash
					 << " lowered_requirements=" << segment.loweredRequirementCount
					 << " barrier_transition_count=" << segment.barriers.transitionCount
					 << " barrier_wait_count=" << segment.barriers.waitCount
					 << " barrier_transition_ops=" << segment.barriers.transitions.size()
					 << " barrier_wait_ops=" << segment.barriers.waits.size()
					 << " replay_signal_ops=" << segment.replay.signals.size()
					 << " replay_transition_ops=" << segment.replay.transitions.size()
					 << " replay_wait_ops=" << segment.replay.waits.size()
					 << " replay_external_waits=" << replayExternalWaitCount
					 << " barrier_transition_hash=" << segment.barriers.transitionHash
					 << " barrier_wait_hash=" << segment.barriers.waitHash
					 << " cache_hit=" << (segment.barrierCacheHit ? "true" : "false")
					 << " reused=" << (segment.barriersReused ? "true" : "false")
					 << " reason=\"" << segment.barrierReuseReason << "\""
					 << "\n";
			}
		}

		if (!m_lastSegmentPlans.empty()) {
			dump << "\n[SegmentPlan]\n";
			for (size_t segmentIndex = 0; segmentIndex < m_lastSegmentPlans.size(); ++segmentIndex) {
				const auto& plan = m_lastSegmentPlans[segmentIndex];
				dump << "[" << segmentIndex << "]"
					 << " name=\"" << plan.name << "\""
					 << " kind=" << (plan.kind == SegmentPlanKind::Replay ? "Replay" : "Lower")
					 << " first_pass_stream_index=" << plan.firstPassStreamIndex
					 << " pass_count=" << plan.passCount
					 << " batch_range=[" << plan.firstBatch << "," << plan.lastBatch << "]"
					 << " cache_key=" << plan.cacheKey
					 << " lowered_requirements=" << plan.loweredRequirementCount
					 << " reason=\"" << plan.reason << "\""
					 << "\n";
			}
		}

		if (!m_compileReuseEvents.empty()) {
			dump << "\n[CompileReuse]\n";
			for (const auto& event : m_compileReuseEvents) {
				dump << "- stage=\"" << event.stage << "\""
					 << " artifact=\"" << event.artifact << "\""
					 << " reused=" << (event.reused ? "true" : "false")
					 << " key=" << event.key
					 << " reason=\"" << event.reason << "\""
					 << "\n";
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

		auto sanitizeDumpVariant = [](std::string_view value) {
			std::string sanitized;
			sanitized.reserve(value.size());
			for (char ch : value) {
				if (std::isalnum(static_cast<unsigned char>(ch))) {
					sanitized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
				}
				else if (ch == '-' || ch == '_' || ch == '.') {
					sanitized.push_back(ch);
				}
				else {
					sanitized.push_back('_');
				}
			}
			return sanitized;
		};

		auto writeDumpFile = [&](const fs::path& dumpPath) -> bool {
			std::ofstream outFile(dumpPath, std::ios::out | std::ios::trunc);
			if (!outFile.is_open()) {
				spdlog::warn("Failed to open render graph debug dump '{}'", dumpPath.string());
				return false;
			}
			outFile << dump.str();
			return true;
		};

		const fs::path latestDumpPath = dumpDir / "rendergraph_compiled_state_latest.txt";
		if (!writeDumpFile(latestDumpPath)) {
			return;
		}

		fs::path variantDumpPath;
		if (!dumpVariant.empty()) {
			variantDumpPath = dumpDir / ("rendergraph_compiled_state_" + sanitizeDumpVariant(dumpVariant) + ".txt");
			writeDumpFile(variantDumpPath);
		}

		static bool announcedDumpPath = false;
		if (!announcedDumpPath) {
			announcedDumpPath = true;
			if (!variantDumpPath.empty()) {
				spdlog::info(
					"Render graph compiled-state dumps will be written to '{}' and '{}'",
					latestDumpPath.string(),
					variantDumpPath.string());
			}
			else {
				spdlog::info("Render graph compiled-state dump will be written to '{}'", latestDumpPath.string());
			}
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

RenderGraph::PassView RenderGraph::GetPassView(AnyPassAndResources& pr) {
	PassView v{};
	if (pr.type == PassType::Compute) {
		auto& p = std::get<ComputePassAndResources>(pr.pass);
		v.reqs = &p.resources.frameResourceRequirements;
		v.internalTransitions = &p.resources.internalTransitions;
	}
	else if (pr.type == PassType::Render) {
		auto& p = std::get<RenderPassAndResources>(pr.pass);
		v.reqs = &p.resources.frameResourceRequirements;
		v.internalTransitions = &p.resources.internalTransitions;
	}
	else if (pr.type == PassType::Copy) {
		auto& p = std::get<CopyPassAndResources>(pr.pass);
		v.reqs = &p.resources.frameResourceRequirements;
		v.internalTransitions = &p.resources.internalTransitions;
	}
	return v;
}

std::vector<RenderGraph::Node> RenderGraph::BuildNodes(RenderGraph& rg, std::vector<AnyPassAndResources>& passes) {
	ZoneScopedN("RenderGraph::BuildNodes");

	std::vector<Node> nodes;
	nodes.resize(passes.size());

	auto resolveCompatibleQueueSlotsForPass = [&rg](const AnyPassAndResources& pr) -> std::vector<size_t> {
		auto collectSlotsForKind = [&rg](QueueKind kind) {
			std::vector<size_t> slots;
			const size_t slotCount = rg.m_queueRegistry.SlotCount();
			slots.reserve(slotCount);
			for (size_t slotIndex = 0; slotIndex < slotCount; ++slotIndex) {
				const auto queueSlotIndex = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slotIndex));
				if (rg.m_queueRegistry.GetKind(queueSlotIndex) == kind && rg.m_queueRegistry.IsAutoAssignable(queueSlotIndex)) {
					slots.push_back(slotIndex);
				}
			}
			if (slots.empty()) {
				slots.push_back(QueueIndex(kind));
			}
			return slots;
		};

		auto collectAutomaticSlotsForPassType = [&rg](PassType type) {
			std::vector<size_t> slots;
			const size_t slotCount = rg.m_queueRegistry.SlotCount();
			slots.reserve(slotCount);
			for (size_t slotIndex = 0; slotIndex < slotCount; ++slotIndex) {
				const auto queueSlotIndex = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slotIndex));
				if (!rg.m_queueRegistry.IsAutoAssignable(queueSlotIndex)) {
					continue;
				}
				const QueueKind kind = rg.m_queueRegistry.GetKind(queueSlotIndex);
				if (IsPreferredQueueKindCompatible(type, kind)) {
					slots.push_back(slotIndex);
				}
			}
			return slots;
		};

		if (pr.type == PassType::Compute) {
			const auto& pass = std::get<ComputePassAndResources>(pr.pass);
			if (pass.resources.pinnedQueueSlot)
				return std::vector<size_t>{ static_cast<size_t>(static_cast<uint8_t>(*pass.resources.pinnedQueueSlot)) };
			if (pass.resources.queueAssignmentPolicy == QueueAssignmentPolicy::Automatic) {
				auto slots = collectAutomaticSlotsForPassType(pr.type);
				if (!slots.empty()) {
					return slots;
				}
			}
			return collectSlotsForKind(pass.resources.preferredQueueKind);
		}

		if (pr.type == PassType::Render) {
			const auto& pass = std::get<RenderPassAndResources>(pr.pass);
			if (pass.resources.pinnedQueueSlot)
				return std::vector<size_t>{ static_cast<size_t>(static_cast<uint8_t>(*pass.resources.pinnedQueueSlot)) };
			if (pass.resources.queueAssignmentPolicy == QueueAssignmentPolicy::Automatic) {
				auto slots = collectAutomaticSlotsForPassType(pr.type);
				if (!slots.empty()) {
					return slots;
				}
			}
			return collectSlotsForKind(pass.resources.preferredQueueKind);
		}

		if (pr.type == PassType::Copy) {
			const auto& pass = std::get<CopyPassAndResources>(pr.pass);
			if (pass.resources.pinnedQueueSlot)
				return std::vector<size_t>{ static_cast<size_t>(static_cast<uint8_t>(*pass.resources.pinnedQueueSlot)) };
			if (pass.resources.queueAssignmentPolicy == QueueAssignmentPolicy::Automatic) {
				auto slots = collectAutomaticSlotsForPassType(pr.type);
				if (!slots.empty()) {
					return slots;
				}
			}
			return collectSlotsForKind(pass.resources.preferredQueueKind);
		}

		return std::vector<size_t>{ QueueIndex(QueueKind::Graphics) };
	};

	for (size_t i = 0; i < passes.size(); ++i) {
		Node n{};
		n.passIndex = i;
		n.compatibleQueueSlots = resolveCompatibleQueueSlotsForPass(passes[i]);
		for (size_t slot : n.compatibleQueueSlots) {
			if (slot >= rg.m_queueRegistry.SlotCount()) {
				continue;
			}
			const QueueKind kind = rg.m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot)));
			n.compatibleQueueKindMask |= static_cast<uint8_t>(1u << QueueIndex(kind));
		}
		n.preferredQueueKind = DefaultPreferredQueueKind(passes[i].type);
		n.queueAssignmentPolicy = DefaultQueueAssignmentPolicy(passes[i].type);
		if (passes[i].type == PassType::Render) {
			const auto& pass = std::get<RenderPassAndResources>(passes[i].pass);
			n.preferredQueueKind = pass.resources.preferredQueueKind;
			n.queueAssignmentPolicy = pass.resources.queueAssignmentPolicy;
		}
		else if (passes[i].type == PassType::Compute) {
			const auto& pass = std::get<ComputePassAndResources>(passes[i].pass);
			n.preferredQueueKind = pass.resources.preferredQueueKind;
			n.queueAssignmentPolicy = pass.resources.queueAssignmentPolicy;
		}
		else if (passes[i].type == PassType::Copy) {
			const auto& pass = std::get<CopyPassAndResources>(passes[i].pass);
			n.preferredQueueKind = pass.resources.preferredQueueKind;
			n.queueAssignmentPolicy = pass.resources.queueAssignmentPolicy;
		}
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

		PassView view = GetPassView(passes[i]);

		auto mark = [&](uint64_t rid, AccessKind k, bool isUav) {
			n.touchedIDs.push_back(rid);
			if (isUav) n.uavIDs.push_back(rid);
			n.accessByID.push_back({rid, k});
			};

		// Alias placement is computed later in CompileFrame, after the dependency DAG
		// has already been built. Expanding through the previous frame's alias
		// placements here can inject stale hazards into the current frame, so node
		// construction must only reflect the pass's declared resources.
		// Current-frame alias overlap is handled after BuildAliasPlanAfterDag.
		// resource requirements
		for (auto& req : *view.reqs) {
			uint64_t base = req.resourceHandleAndRange.resource.GetGlobalResourceID();
			bool write = AccessTypeIsWriteType(req.state.access);
			bool isUav = IsUAVState(req.state);

			mark(base, write ? AccessKind::Write : AccessKind::Read, isUav);
		}

		// internal transitions: treat as "write" for scheduling conservatism
		for (auto& tr : *view.internalTransitions) {
			uint64_t base = tr.first.resource.GetGlobalResourceID();
			mark(base, AccessKind::Write, /*isUav=*/false);
		}

		// Deduplicate touchedIDs
		std::sort(n.touchedIDs.begin(), n.touchedIDs.end());
		n.touchedIDs.erase(std::unique(n.touchedIDs.begin(), n.touchedIDs.end()), n.touchedIDs.end());

		// Deduplicate uavIDs
		std::sort(n.uavIDs.begin(), n.uavIDs.end());
		n.uavIDs.erase(std::unique(n.uavIDs.begin(), n.uavIDs.end()), n.uavIDs.end());

		// Deduplicate accessByID: sort by ID, then collapse duplicates keeping Write over Read
		std::sort(n.accessByID.begin(), n.accessByID.end(), [](auto& a, auto& b) {
			if (a.first != b.first) return a.first < b.first;
			return a.second > b.second; // Write (1) before Read (0)
		});
		auto last = std::unique(n.accessByID.begin(), n.accessByID.end(),
			[](auto& a, auto& b) { return a.first == b.first; });
		n.accessByID.erase(last, n.accessByID.end());

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
	std::unordered_map<uint64_t, SeqState> seq;
	{
		size_t totalAccesses = 0;
		for (const auto& node : nodes) totalAccesses += node.accessByID.size();
		seq.reserve(totalAccesses);
	}

	std::unordered_set<uint64_t> edgeSet;
	edgeSet.reserve(nodes.size() * 8);

	// build deps in ORIGINAL order
	for (size_t i = 0; i < nodes.size(); ++i) {
		auto& node = nodes[i];

		for (auto& [rid, kind] : node.accessByID) {
			auto& s = seq[rid];

			if (kind == AccessKind::Read) {
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

	std::vector<size_t> q;
	q.reserve(nodes.size());
	for (size_t i = 0; i < nodes.size(); ++i)
		if (indeg[i] == 0) q.push_back(i);

	std::vector<size_t> topo;
	topo.reserve(nodes.size());

	for (size_t head = 0; head < q.size(); ++head) {
		size_t u = q[head];
		topo.push_back(u);
		for (size_t v : nodes[u].out) {
			if (--indeg[v] == 0) q.push_back(v);
		}
	}

	if (topo.size() != nodes.size()) {
		// cycle: invalid graph
		return false;
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
	if (aliasPlacementRangesByID.empty()) {
		return true;
	}

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

	std::unordered_set<uint64_t> edgeSet;
	edgeSet.reserve(existingEdgeCount + aliasPlacementRangesByID.size() * 4);
	for (size_t from = 0; from < nodes.size(); ++from) {
		for (size_t to : nodes[from].out) {
			edgeSet.insert((uint64_t(from) << 32) | uint64_t(to));
		}
	}

	std::vector<uint64_t> resourceIDs;
	resourceIDs.reserve(aliasPlacementRangesByID.size());
	for (const auto& [resourceID, placement] : aliasPlacementRangesByID) {
		(void)placement;
		resourceIDs.push_back(resourceID);
	}
	std::sort(resourceIDs.begin(), resourceIDs.end());

	for (size_t i = 0; i < resourceIDs.size(); ++i) {
		const uint64_t lhsResourceID = resourceIDs[i];
		const auto lhsIt = aliasPlacementRangesByID.find(lhsResourceID);
		if (lhsIt == aliasPlacementRangesByID.end()) {
			continue;
		}

		const auto& lhs = lhsIt->second;
		if (lhs.firstUsePassIndex == std::numeric_limits<size_t>::max() ||
			lhs.lastUsePassIndex == std::numeric_limits<size_t>::max()) {
			continue;
		}

		for (size_t j = i + 1; j < resourceIDs.size(); ++j) {
			const uint64_t rhsResourceID = resourceIDs[j];
			const auto rhsIt = aliasPlacementRangesByID.find(rhsResourceID);
			if (rhsIt == aliasPlacementRangesByID.end()) {
				continue;
			}

			const auto& rhs = rhsIt->second;
			if (lhs.poolID != rhs.poolID || !rangesOverlap(lhs, rhs)) {
				continue;
			}
			if (rhs.firstUsePassIndex == std::numeric_limits<size_t>::max() ||
				rhs.lastUsePassIndex == std::numeric_limits<size_t>::max()) {
				continue;
			}

			size_t fromPassIndex = std::numeric_limits<size_t>::max();
			size_t toPassIndex = std::numeric_limits<size_t>::max();
			if (lhs.lastUse < rhs.firstUse) {
				fromPassIndex = lhs.lastUsePassIndex;
				toPassIndex = rhs.firstUsePassIndex;
			}
			else if (rhs.lastUse < lhs.firstUse) {
				fromPassIndex = rhs.lastUsePassIndex;
				toPassIndex = lhs.firstUsePassIndex;
			}
			else {
				throw std::runtime_error(
					"Alias plan produced overlapping lifetimes for overlapping placements: resource " +
					std::to_string(lhsResourceID) + " ('" + resourceDebugName(lhsResourceID) + "') [" +
					std::to_string(lhs.startByte) + ", " + std::to_string(lhs.endByte) + ") firstUse=" +
					std::to_string(static_cast<uint64_t>(lhs.firstUse)) + " lastUse=" +
					std::to_string(static_cast<uint64_t>(lhs.lastUse)) + " and resource " +
					std::to_string(rhsResourceID) + " ('" + resourceDebugName(rhsResourceID) + "') [" +
					std::to_string(rhs.startByte) + ", " + std::to_string(rhs.endByte) + ") firstUse=" +
					std::to_string(static_cast<uint64_t>(rhs.firstUse)) + " lastUse=" +
					std::to_string(static_cast<uint64_t>(rhs.lastUse)));
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

namespace {
	uint64_t HashStringView64(std::string_view value) noexcept {
		uint64_t seed = 1469598103934665603ull;
		for (unsigned char c : value) {
			seed ^= c;
			seed *= 1099511628211ull;
		}
		return seed;
	}

	uint64_t HashBound64(const Bound& bound) noexcept {
		uint64_t seed = 0;
		seed = rg::HashCombine(seed, static_cast<uint64_t>(bound.type));
		seed = rg::HashCombine(seed, static_cast<uint64_t>(bound.value));
		return seed;
	}

	uint64_t HashRangeSpec64(const RangeSpec& range) noexcept {
		uint64_t seed = 0;
		seed = rg::HashCombine(seed, HashBound64(range.mipLower));
		seed = rg::HashCombine(seed, HashBound64(range.mipUpper));
		seed = rg::HashCombine(seed, HashBound64(range.sliceLower));
		seed = rg::HashCombine(seed, HashBound64(range.sliceUpper));
		return seed;
	}

	uint64_t HashResourceState64(const ResourceState& state) noexcept {
		uint64_t seed = 0;
		seed = rg::HashCombine(seed, static_cast<uint64_t>(state.access));
		seed = rg::HashCombine(seed, static_cast<uint64_t>(state.layout));
		seed = rg::HashCombine(seed, static_cast<uint64_t>(state.sync));
		return seed;
	}

	constexpr uint64_t kBarrierSegmentCachePolicyVersion = 5;

	uint64_t HashBytes64(std::span<const std::byte> bytes) noexcept {
		uint64_t seed = 1469598103934665603ull;
		for (std::byte b : bytes) {
			seed ^= static_cast<uint64_t>(std::to_integer<unsigned char>(b));
			seed *= 1099511628211ull;
		}
		return seed;
	}

	std::string LowerAscii(std::string_view value) {
		std::string result;
		result.reserve(value.size());
		for (char c : value) {
			result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
		}
		return result;
	}

	std::string_view SegmentNameForPass(std::string_view passName) {
		const std::string lower = LowerAscii(passName);
		if (lower.find("upload") != std::string::npos || lower.find("setup") != std::string::npos) {
			return "frame-setup";
		}
		if (lower.find("readback") != std::string::npos || lower.find("capture") != std::string::npos) {
			return "readback-tail";
		}
		if (lower.find("depth") != std::string::npos || lower.find("visibility") != std::string::npos) {
			return "depth-visibility";
		}
		if (lower.find("shadow") != std::string::npos) {
			return "shadows";
		}
		if (lower.find("gbuffer") != std::string::npos || lower.find("g-buffer") != std::string::npos || lower.find("main") != std::string::npos) {
			return "main";
		}
		if (lower.find("light") != std::string::npos) {
			return "lighting";
		}
		if (lower.find("transparent") != std::string::npos || lower.find("transparency") != std::string::npos) {
			return "transparency";
		}
		if (lower.find("post") != std::string::npos || lower.find("tonemap") != std::string::npos || lower.find("bloom") != std::string::npos) {
			return "postprocess";
		}
		if (lower.find("ui") != std::string::npos || lower.find("imgui") != std::string::npos || lower.find("present") != std::string::npos) {
			return "ui-present";
		}
		return "main";
	}

	bool IsVolatileSchedulePassName(std::string_view passName) {
		// TODO: Replace this manual volatility classification with a declared pass
		// or resource policy so schedule reuse does not depend on pass-name heuristics.
		const std::string lower = LowerAscii(passName);
		return lower.find("readback") != std::string::npos
			|| lower.find("capture") != std::string::npos;
	}

	template<typename ScheduleCacheKeyPartsT>
	uint64_t BuildRelaxedScheduleCacheKey(const ScheduleCacheKeyPartsT& parts) {
		uint64_t key = rg::HashCombine(0ull, parts.frameStructureHash);
		key = rg::HashCombine(key, parts.queueConfigHash);
		key = rg::HashCombine(key, parts.activeQueueHash);
		key = rg::HashCombine(key, parts.settingsHash);
		key = rg::HashCombine(key, parts.nodeTopologyHash);
		key = rg::HashCombine(key, parts.dependencyEdgeHash);
		return key;
	}

	template<typename ScheduleCacheKeyPartsT>
	std::string DescribeScheduleCacheMiss(
		const ScheduleCacheKeyPartsT& current,
		const std::optional<ScheduleCacheKeyPartsT>& previous)
	{
		if (!previous.has_value()) {
			return "no cached schedule for dependency/pass/queue signature";
		}

		const auto& prev = *previous;
		std::vector<std::string> changed;
		auto addChanged = [&](std::string_view label, uint64_t before, uint64_t after) {
			if (before == after) {
				return;
			}

			std::ostringstream item;
			item << label << "(" << before << "->" << after << ")";
			changed.push_back(item.str());
		};
		addChanged("frame-structure", prev.frameStructureHash, current.frameStructureHash);
		addChanged("frame-schedule", prev.frameScheduleHash, current.frameScheduleHash);
		addChanged("queue-config", prev.queueConfigHash, current.queueConfigHash);
		addChanged("active-queues", prev.activeQueueHash, current.activeQueueHash);
		addChanged("scheduler-settings", prev.settingsHash, current.settingsHash);
		addChanged("node-topology", prev.nodeTopologyHash, current.nodeTopologyHash);
		addChanged("dependency-edges", prev.dependencyEdgeHash, current.dependencyEdgeHash);

		if (changed.empty()) {
			return "no cached schedule for dependency/pass/queue signature; tracked key parts matched previous frame";
		}

		std::ostringstream oss;
		oss << "schedule signature changed: ";
		for (size_t i = 0; i < changed.size(); ++i) {
			if (i > 0) {
				oss << ",";
			}
			oss << changed[i];
		}
		return oss.str();
	}
}

void RenderGraph::BuildFrameProgramIR() {
	ZoneScopedN("RenderGraph::BuildFrameProgramIR");
	m_compileCacheStats = CompileCacheStats{};
	m_compileReuseEvents.clear();
	const bool disableGraphCaching = m_getRenderGraphDisableCaching ? m_getRenderGraphDisableCaching() : false;
	if (disableGraphCaching) {
		m_cachedPassIRByStableId.clear();
		m_cachedScheduleIRByKey.clear();
		m_cachedScheduleIRByRelaxedKey.clear();
		m_cachedScheduleKeyPartsByKey.clear();
		m_cachedBarrierSegments.clear();
		m_lastScheduleCacheKeyParts.reset();
	}
	FrameProgram program{};
	program.passes.reserve(m_framePasses.size());

	auto appendScheduleAccessHash = [](uint64_t seed, const NormalizedAccess& access) {
		seed = rg::HashCombine(seed, access.resourceID);
		seed = rg::HashCombine(seed, HashRangeSpec64(access.range));
		seed = rg::HashCombine(seed, static_cast<uint64_t>(access.accessKind));
		seed = rg::HashCombine(seed, access.isUAV ? 1ull : 0ull);
		seed = rg::HashCombine(seed, access.isInternalTransition ? 1ull : 0ull);
		return seed;
	};

	auto appendBarrierAccessHash = [](uint64_t seed, const NormalizedAccess& access) {
		seed = rg::HashCombine(seed, access.resourceID);
		seed = rg::HashCombine(seed, HashRangeSpec64(access.range));
		seed = rg::HashCombine(seed, HashResourceState64(access.state));
		seed = rg::HashCombine(seed, static_cast<uint64_t>(access.accessKind));
		seed = rg::HashCombine(seed, access.isUAV ? 1ull : 0ull);
		seed = rg::HashCombine(seed, access.isInternalTransition ? 1ull : 0ull);
		return seed;
	};

	for (size_t passIndex = 0; passIndex < m_framePasses.size(); ++passIndex) {
		const auto& any = m_framePasses[passIndex];
		const auto& summary = m_framePassSchedulingSummaries[passIndex];

		PassIR passIR{};
		passIR.id.value = rg::HashCombine(HashStringView64(any.name), static_cast<uint64_t>(passIndex));
		passIR.name = any.name;
		passIR.type = any.type;

		std::visit([&](const auto& pass) {
			using PassTypeT = std::decay_t<decltype(pass)>;
			if constexpr (!std::is_same_v<PassTypeT, std::monostate>) {
				passIR.run = pass.run;
				passIR.preferredQueueKind = pass.resources.preferredQueueKind;
				passIR.queuePolicy = pass.resources.pinnedQueueSlot.has_value()
					? QueueAssignmentPolicy::ForcePreferred
					: DefaultQueueAssignmentPolicy(any.type);
				passIR.pinnedQueueSlot = pass.resources.pinnedQueueSlot;
				passIR.immediateHash = HashBytes64(pass.immediateBytecode);
				for (const auto& snap : pass.resolverSnapshots) {
					const uint64_t version = snap.resolver ? snap.resolver->GetContentVersion() : snap.version;
					passIR.declarationHash = rg::HashCombine(passIR.declarationHash, version);
				}
			}
		}, any.pass);

		passIR.queuePolicyHash = rg::HashCombine(static_cast<uint64_t>(passIR.preferredQueueKind), static_cast<uint64_t>(passIR.queuePolicy));
		if (passIR.pinnedQueueSlot.has_value()) {
			passIR.queuePolicyHash = rg::HashCombine(passIR.queuePolicyHash, static_cast<uint64_t>(static_cast<uint8_t>(*passIR.pinnedQueueSlot)));
		}

		for (const auto& requirement : summary.requirements) {
			NormalizedAccess access{};
			access.resourceID = requirement.resourceID;
			access.resourceIndex = requirement.resourceIndex;
			access.range = requirement.range;
			access.state = requirement.state;
			access.accessKind = AccessTypeIsWriteType(requirement.state.access)
				? NormalizedAccessKind::Write
				: NormalizedAccessKind::Read;
			access.isUAV = requirement.isUAV;
			access.hash = appendBarrierAccessHash(appendScheduleAccessHash(0, access), access);
			passIR.accesses.push_back(access);
			passIR.scheduleHash = appendScheduleAccessHash(passIR.scheduleHash, access);
			passIR.barrierHash = appendBarrierAccessHash(passIR.barrierHash, access);
			passIR.declarationHash = rg::HashCombine(passIR.declarationHash, access.hash);

			auto itGeneration = compiledResourceGenerationByID.find(requirement.resourceID);
			if (itGeneration != compiledResourceGenerationByID.end()) {
				passIR.resourceGenerationHash = rg::HashCombine(passIR.resourceGenerationHash, itGeneration->second);
			}

			auto itAliasSignature = aliasPlacementSignatureByID.find(requirement.resourceID);
			if (itAliasSignature != aliasPlacementSignatureByID.end()) {
				passIR.aliasPolicyHash = rg::HashCombine(passIR.aliasPolicyHash, itAliasSignature->second);
			}
		}

		PassView passView = GetPassView(m_framePasses[passIndex]);
		for (const auto& transition : *passView.internalTransitions) {
			auto resourceIndex = TryGetFrameSchedulingResourceIndex(transition.first.resource.GetGlobalResourceID());
			if (!resourceIndex.has_value()) {
				continue;
			}

			NormalizedAccess access{};
			access.resourceID = transition.first.resource.GetGlobalResourceID();
			access.resourceIndex = *resourceIndex;
			access.range = transition.first.range;
			access.state = transition.second;
			access.accessKind = NormalizedAccessKind::Write;
			access.isInternalTransition = true;
			access.hash = appendBarrierAccessHash(appendScheduleAccessHash(0, access), access);
			passIR.internalTransitions.push_back(access);
			passIR.scheduleHash = appendScheduleAccessHash(passIR.scheduleHash, access);
			passIR.barrierHash = appendBarrierAccessHash(passIR.barrierHash, access);
			passIR.declarationHash = rg::HashCombine(passIR.declarationHash, access.hash);
		}

		passIR.scheduleHash = rg::HashCombine(passIR.scheduleHash, passIR.id.value);
		passIR.scheduleHash = rg::HashCombine(passIR.scheduleHash, passIR.queuePolicyHash);

		passIR.barrierHash = rg::HashCombine(passIR.barrierHash, passIR.scheduleHash);
		passIR.barrierHash = rg::HashCombine(passIR.barrierHash, passIR.aliasPolicyHash);

		passIR.executionHash = passIR.barrierHash;
		passIR.executionHash = rg::HashCombine(passIR.executionHash, passIR.immediateHash);
		passIR.executionHash = rg::HashCombine(passIR.executionHash, passIR.resourceGenerationHash);

		passIR.fullHash = passIR.declarationHash;
		passIR.fullHash = rg::HashCombine(passIR.fullHash, passIR.immediateHash);
		passIR.fullHash = rg::HashCombine(passIR.fullHash, passIR.queuePolicyHash);
		passIR.fullHash = rg::HashCombine(passIR.fullHash, passIR.resourceGenerationHash);
		passIR.fullHash = rg::HashCombine(passIR.fullHash, passIR.aliasPolicyHash);

		auto cacheIt = m_cachedPassIRByStableId.find(passIR.id.value);
		if (cacheIt != m_cachedPassIRByStableId.end() && cacheIt->second.fullHash == passIR.fullHash) {
			++m_compileCacheStats.passIRCacheHits;
			RecordCompileReuseEvent("PassIR", passIR.name, true, passIR.id.value, "stable pass fingerprint matched cached PassIR");
			passIR = cacheIt->second;
		}
		else {
			++m_compileCacheStats.passIRCacheMisses;
			std::string reason = "no cached PassIR for stable pass id";
			if (cacheIt != m_cachedPassIRByStableId.end()) {
				const PassIR& previous = cacheIt->second;
				std::vector<std::string> changed;
				if (previous.scheduleHash != passIR.scheduleHash) changed.push_back("schedule");
				if (previous.barrierHash != passIR.barrierHash) changed.push_back("barrier");
				if (previous.executionHash != passIR.executionHash) changed.push_back("execution");
				if (previous.declarationHash != passIR.declarationHash) changed.push_back("declaration");
				if (previous.immediateHash != passIR.immediateHash) changed.push_back("immediate");
				if (previous.queuePolicyHash != passIR.queuePolicyHash) changed.push_back("queue-policy");
				if (previous.resourceGenerationHash != passIR.resourceGenerationHash) changed.push_back("resource-generation");
				if (previous.aliasPolicyHash != passIR.aliasPolicyHash) changed.push_back("alias-policy");

				std::ostringstream oss;
				oss << "PassIR fingerprint changed: ";
				for (size_t i = 0; i < changed.size(); ++i) {
					if (i != 0) oss << ",";
					oss << changed[i];
				}
				if (changed.empty()) {
					oss << "full-hash-only";
				}
				reason = oss.str();

				if (previous.scheduleHash != passIR.scheduleHash) {
					if (IsVolatileSchedulePassName(passIR.name)) {
						++m_compileCacheStats.volatileSchedulePassChanges;
					}
					else {
						++m_compileCacheStats.nonVolatileSchedulePassChanges;
					}

					std::ostringstream scheduleReason;
					scheduleReason << "pass schedule hash changed: "
						<< previous.scheduleHash << "->" << passIR.scheduleHash
						<< " declaration=" << previous.declarationHash << "->" << passIR.declarationHash
						<< " queue_policy=" << previous.queuePolicyHash << "->" << passIR.queuePolicyHash
						<< " access_count=" << (previous.accesses.size() + previous.internalTransitions.size())
						<< "->" << (passIR.accesses.size() + passIR.internalTransitions.size());
					RecordCompileReuseEvent("SchedulePass", passIR.name, false, passIR.id.value, scheduleReason.str());
				}
			}
			RecordCompileReuseEvent("PassIR", passIR.name, false, passIR.id.value, reason);
			m_cachedPassIRByStableId[passIR.id.value] = passIR;
		}

		program.normalizedAccessCount += passIR.accesses.size() + passIR.internalTransitions.size();
		program.structureHash = rg::HashCombine(program.structureHash, passIR.id.value);
		program.passContentHash = rg::HashCombine(program.passContentHash, passIR.fullHash);
		program.scheduleContentHash = rg::HashCombine(program.scheduleContentHash, passIR.scheduleHash);
		program.barrierContentHash = rg::HashCombine(program.barrierContentHash, passIR.barrierHash);
		program.executionContentHash = rg::HashCombine(program.executionContentHash, passIR.executionHash);
		program.passes.push_back(std::move(passIR));
	}

	m_compiledFrameProgram = std::move(program);
	m_compileCacheStats.passIRCount = m_compiledFrameProgram.passes.size();
	m_compileCacheStats.normalizedAccessCount = m_compiledFrameProgram.normalizedAccessCount;
}

void RenderGraph::BuildResourceAccessChainIR(
	const std::vector<Node>& nodes,
	std::span<const std::pair<size_t, size_t>> explicitEdges)
{
	ZoneScopedN("RenderGraph::BuildResourceAccessChainIR");
	m_resourceAccessChains.clear();
	m_dependencyEdgeIR.clear();

	struct ChainState {
		std::optional<size_t> lastWriterNode;
		std::vector<size_t> readsSinceWriteNodes;
	};

	std::unordered_map<uint64_t, ChainState> chainStateByResource;
	std::unordered_set<uint64_t> edgeSet;
	edgeSet.reserve(nodes.size() * 8 + explicitEdges.size());

	auto addEdge = [&](size_t fromNode, size_t toNode, EdgeKind kind, uint64_t resourceID) {
		if (fromNode >= nodes.size() || toNode >= nodes.size() || fromNode == toNode) {
			return;
		}

		const uint64_t edgeKey = (uint64_t(fromNode) << 32) | uint64_t(toNode);
		if (!edgeSet.insert(edgeKey).second) {
			return;
		}

		DependencyEdgeIR edge{};
		edge.fromPassIndex = nodes[fromNode].passIndex;
		edge.toPassIndex = nodes[toNode].passIndex;
		edge.kind = kind;
		edge.resourceID = resourceID;
		edge.provenanceKey = rg::HashCombine(
			rg::HashCombine(static_cast<uint64_t>(kind), resourceID),
			edgeKey);
		m_dependencyEdgeIR.push_back(edge);

		if (resourceID != 0) {
			auto& chain = m_resourceAccessChains[resourceID];
			chain.resourceID = resourceID;
			chain.producedEdges.push_back(edge);
		}
	};

	for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
		const Node& node = nodes[nodeIndex];
		for (const auto& [resourceID, accessKind] : node.accessByID) {
			auto& chain = m_resourceAccessChains[resourceID];
			chain.resourceID = resourceID;
			chain.events.push_back(ResourceAccessEvent{
				.passIndex = node.passIndex,
				.resourceID = resourceID,
				.accessKind = accessKind,
				.stateHash = rg::HashCombine(resourceID, static_cast<uint64_t>(accessKind)),
			});
			chain.chainHash = rg::HashCombine(chain.chainHash, chain.events.back().stateHash);

			auto& state = chainStateByResource[resourceID];
			if (accessKind == AccessKind::Read) {
				if (state.lastWriterNode.has_value()) {
					addEdge(*state.lastWriterNode, nodeIndex, EdgeKind::ResourceHazard, resourceID);
				}
				state.readsSinceWriteNodes.push_back(nodeIndex);
			}
			else {
				if (state.lastWriterNode.has_value()) {
					addEdge(*state.lastWriterNode, nodeIndex, EdgeKind::ResourceHazard, resourceID);
				}
				for (size_t readNode : state.readsSinceWriteNodes) {
					addEdge(readNode, nodeIndex, EdgeKind::ResourceHazard, resourceID);
				}
				state.readsSinceWriteNodes.clear();
				state.lastWriterNode = nodeIndex;
			}
		}
	}

	for (const auto& [fromNode, toNode] : explicitEdges) {
		addEdge(fromNode, toNode, EdgeKind::Explicit, 0);
	}
}

void RenderGraph::SimulatePassForSchedule(
	const Node& node,
	BatchBuildState& batchBuildState,
	std::vector<SymbolicTracker*>& passBatchTrackersByResourceIndex,
	std::vector<ResourceTransition>& scratchTransitions)
{
	const size_t passQueueSlot = node.assignedQueueSlot.value_or(node.queueSlot);
	const QueueKind passQueue = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(passQueueSlot));
	const auto& passSummary = m_framePassSchedulingSummaries[node.passIndex];

	for (const auto& requirement : passSummary.requirements) {
		Resource* resource = requirement.resource.IsEphemeral()
			? requirement.resource.GetEphemeralPtr()
			: _registry.Resolve(requirement.resource);
		auto& tracker = GetOrCreateCompileTracker(resource, requirement.resourceID);
		const ResourceState requiredState = NormalizeStateForQueue(passQueue, requirement.state);
		scratchTransitions.clear();
		tracker.Apply(requirement.range, resource, requiredState, scratchTransitions);
		if (requirement.resourceIndex < passBatchTrackersByResourceIndex.size()) {
			passBatchTrackersByResourceIndex[requirement.resourceIndex] = &tracker;
		}
	}

	PassView view = GetPassView(m_framePasses[node.passIndex]);
	for (const auto& exit : *view.internalTransitions) {
		auto resource = _registry.Resolve(exit.first.resource);
		auto& tracker = GetOrCreateCompileTracker(resource, exit.first.resource.GetGlobalResourceID());
		scratchTransitions.clear();
		tracker.Apply(exit.first.range, resource, exit.second, scratchTransitions);
	}

	for (size_t resourceIndex : passSummary.requiredResourceIndices) {
		batchBuildState.MarkResource(resourceIndex);
	}
	for (const auto& transition : passSummary.internalTransitions) {
		batchBuildState.MarkInternalTransition(transition.resourceIndex);
	}
	for (size_t resourceIndex : passSummary.uavResourceIndices) {
		for (size_t queueIndex = 0; queueIndex < batchBuildState.queueCount; ++queueIndex) {
			if (queueIndex != passQueueSlot) {
				batchBuildState.MarkOtherQueueUAV(queueIndex, resourceIndex);
			}
		}
	}
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
		for (const auto& exit : pass.resources.internalTransitions) {
			std::vector<ResourceTransition> ignoredTransitions;
			auto pRes = _registry.Resolve(exit.first.resource);
			auto& compileTracker = GetOrCreateCompileTracker(pRes, exit.first.resource.GetGlobalResourceID());
			compileTracker.Apply(exit.first.range, nullptr, exit.second, ignoredTransitions);
			SortedInsert(currentBatch.internallyTransitionedResources, exit.first.resource.GetGlobalResourceID());
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

RenderGraph::ScheduleIR RenderGraph::BuildScheduleIR(
	RenderGraph& rg,
	std::vector<AnyPassAndResources>& passes,
	std::vector<Node>& nodes)
{
	ZoneScopedN("RenderGraph::BuildScheduleIR");
	ScheduleIR schedule{};

	std::vector<uint32_t> indeg(nodes.size());
	for (size_t i = 0; i < nodes.size(); ++i) indeg[i] = nodes[i].indegree;

	std::vector<size_t> ready;
	ready.reserve(nodes.size());
	for (size_t i = 0; i < nodes.size(); ++i) {
		if (indeg[i] == 0) ready.push_back(i);
	}

	const size_t queueCount = rg.m_queueRegistry.SlotCount();
	const size_t gfxSlot = QueueIndex(QueueKind::Graphics);
	BatchBuildState batchBuildState;
	batchBuildState.Initialize(nodes.size(), queueCount, rg.m_frameSchedulingResourceCount);
	std::vector<SymbolicTracker*> passBatchTrackersByResourceIndex(rg.m_frameSchedulingResourceCount, nullptr);
	std::vector<ResourceTransition> scratchTransitions;

	unsigned int currentBatchIndex = 1;
	SymbolicBatchIR currentBatch{};
	currentBatch.id = currentBatchIndex;
	currentBatch.queueHasWork.assign(queueCount, 0);

	auto closeBatch = [&]() {
		if (!currentBatch.passes.empty()) {
			schedule.batches.push_back(std::move(currentBatch));
		}
		++currentBatchIndex;
		currentBatch = SymbolicBatchIR{};
		currentBatch.id = currentBatchIndex;
		currentBatch.queueHasWork.assign(queueCount, 0);
		batchBuildState.ResetForNewBatch();
		std::fill(passBatchTrackersByResourceIndex.begin(), passBatchTrackersByResourceIndex.end(), nullptr);
	};

	auto commitScheduledPass = [&](size_t nodeIndex, size_t queueSlot) {
		auto& node = nodes[nodeIndex];
		node.assignedQueueSlot = queueSlot;
		if (node.passIndex < rg.m_assignedQueueSlotsByFramePass.size()) {
			rg.m_assignedQueueSlotsByFramePass[node.passIndex] = queueSlot;
		}

		ScheduledPass scheduled{};
		scheduled.nodeIndex = nodeIndex;
		scheduled.passIndex = node.passIndex;
		scheduled.queueSlot = queueSlot;
		scheduled.symbolicBatch = currentBatch.id;
		currentBatch.passes.push_back(scheduled);
		if (queueSlot < currentBatch.queueHasWork.size()) {
			currentBatch.queueHasWork[queueSlot] = 1;
		}
		schedule.passStream.push_back(scheduled);
		schedule.structureHash = rg::HashCombine(schedule.structureHash, static_cast<uint64_t>(node.passIndex));
		schedule.queueAssignmentHash = rg::HashCombine(
			schedule.queueAssignmentHash,
			rg::HashCombine(static_cast<uint64_t>(node.passIndex), static_cast<uint64_t>(queueSlot)));

		SimulatePassForSchedule(node, batchBuildState, passBatchTrackersByResourceIndex, scratchTransitions);
		batchBuildState.MarkNode(nodeIndex);
	};

	const double autoGraphicsBias = rg.m_getQueueSchedulingAutoGraphicsBias ? static_cast<double>(rg.m_getQueueSchedulingAutoGraphicsBias()) : 2.5;
	const double asyncOverlapBonus = rg.m_getQueueSchedulingAsyncOverlapBonus ? static_cast<double>(rg.m_getQueueSchedulingAsyncOverlapBonus()) : 3.0;
	const double crossQueueHandoffPenalty = rg.m_getQueueSchedulingCrossQueueHandoffPenalty ? static_cast<double>(rg.m_getQueueSchedulingCrossQueueHandoffPenalty()) : 2.0;

	size_t remaining = nodes.size();
	while (remaining > 0) {
		int bestIdxInReady = -1;
		size_t bestQueueSlot = 0;
		double bestScore = -1e300;

		{
			ZoneScopedN("RenderGraph::BuildScheduleIR::EvaluateCandidates");
			size_t readyGraphicsCapableCount = 0;
			for (size_t ni : ready) {
				if ((nodes[ni].compatibleQueueKindMask & static_cast<uint8_t>(1u << QueueIndex(QueueKind::Graphics))) != 0) {
					++readyGraphicsCapableCount;
				}
			}

			const bool batchHasGraphicsWork = gfxSlot < currentBatch.queueHasWork.size() && currentBatch.queueHasWork[gfxSlot] != 0;

			for (int ri = 0; ri < static_cast<int>(ready.size()); ++ri) {
				size_t ni = ready[ri];
				auto& n = nodes[ni];
				const auto& passSummary = rg.m_framePassSchedulingSummaries[n.passIndex];

				for (size_t nodeQueueSlot : n.compatibleQueueSlots) {
					if (nodeQueueSlot >= queueCount) {
						continue;
					}
					if (nodeQueueSlot >= rg.m_activeQueueSlotsThisFrame.size() || !rg.m_activeQueueSlotsThisFrame[nodeQueueSlot]) {
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

					if (rg.IsNewBatchNeeded(
						passSummary,
						passBatchTrackersByResourceIndex,
						batchBuildState,
						passes[n.passIndex].name,
						currentBatchIndex,
						nodeQueueSlot))
					{
						continue;
					}

					int reuse = 0, fresh = 0;
					for (size_t resourceIndex : passSummary.touchedResourceIndices) {
						if (batchBuildState.ContainsResource(resourceIndex)) ++reuse;
						else ++fresh;
					}

					double score = 3.0 * reuse - 1.0 * fresh;
					if (nodeQueueSlot < currentBatch.queueHasWork.size() && !currentBatch.queueHasWork[nodeQueueSlot]) score += 2.0;

					size_t queuedOnSlot = 0;
					for (const auto& scheduled : currentBatch.passes) {
						if (scheduled.queueSlot == nodeQueueSlot) {
							++queuedOnSlot;
						}
					}
					score -= 0.25 * double(queuedOnSlot);
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
			if (!currentBatch.passes.empty()) {
				closeBatch();
				continue;
			}

			ZoneScopedN("RenderGraph::BuildScheduleIR::CommitFallbackPass");
			size_t ni = ready.front();
			auto& n = nodes[ni];
			size_t fallbackSlot = n.queueSlot;
			for (size_t compatibleSlot : n.compatibleQueueSlots) {
				if (compatibleSlot < rg.m_activeQueueSlotsThisFrame.size() && rg.m_activeQueueSlotsThisFrame[compatibleSlot]) {
					fallbackSlot = compatibleSlot;
					break;
				}
			}
			commitScheduledPass(ni, fallbackSlot);
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

		size_t chosenNodeIndex = ready[bestIdxInReady];
		commitScheduledPass(chosenNodeIndex, bestQueueSlot);
		ready[bestIdxInReady] = ready.back();
		ready.pop_back();

		for (size_t v : nodes[chosenNodeIndex].out) {
			if (--indeg[v] == 0) ready.push_back(v);
		}
		--remaining;

		if (rg.m_getHeavyDebug && rg.m_getHeavyDebug()) {
			closeBatch();
		}
	}

	if (!currentBatch.passes.empty()) {
		schedule.batches.push_back(std::move(currentBatch));
	}

	return schedule;
}

RenderGraph::ScheduleCacheKeyParts RenderGraph::BuildScheduleCacheKeyParts(const std::vector<Node>& nodes) const {
	ScheduleCacheKeyParts parts{};
	parts.frameStructureHash = m_compiledFrameProgram.structureHash;
	parts.frameScheduleHash = m_compiledFrameProgram.scheduleContentHash;
	parts.queueConfigHash = rg::HashCombine(static_cast<uint64_t>(m_queueRegistry.SlotCount()), static_cast<uint64_t>(m_frameSchedulingResourceCount));
	parts.settingsHash = rg::HashCombine(0ull, (m_getHeavyDebug && m_getHeavyDebug()) ? 1ull : 0ull);
	parts.settingsHash = rg::HashCombine(parts.settingsHash, static_cast<uint64_t>((m_getQueueSchedulingAutoGraphicsBias ? m_getQueueSchedulingAutoGraphicsBias() : 2.5f) * 1000.0f));
	parts.settingsHash = rg::HashCombine(parts.settingsHash, static_cast<uint64_t>((m_getQueueSchedulingAsyncOverlapBonus ? m_getQueueSchedulingAsyncOverlapBonus() : 3.0f) * 1000.0f));
	parts.settingsHash = rg::HashCombine(parts.settingsHash, static_cast<uint64_t>((m_getQueueSchedulingCrossQueueHandoffPenalty ? m_getQueueSchedulingCrossQueueHandoffPenalty() : 2.0f) * 1000.0f));

	for (size_t queueIndex = 0; queueIndex < m_activeQueueSlotsThisFrame.size(); ++queueIndex) {
		if (m_activeQueueSlotsThisFrame[queueIndex]) {
			uint64_t queueHash = rg::HashCombine(static_cast<uint64_t>(queueIndex), 1ull);
			parts.activeQueueHash = rg::HashCombine(parts.activeQueueHash, queueHash);
		}
	}

	for (const auto& node : nodes) {
		uint64_t nodeHash = rg::HashCombine(static_cast<uint64_t>(node.passIndex), static_cast<uint64_t>(node.originalOrder));
		nodeHash = rg::HashCombine(nodeHash, static_cast<uint64_t>(node.queueSlot));
		nodeHash = rg::HashCombine(nodeHash, static_cast<uint64_t>(node.preferredQueueKind));
		nodeHash = rg::HashCombine(nodeHash, static_cast<uint64_t>(node.queueAssignmentPolicy));
		nodeHash = rg::HashCombine(nodeHash, static_cast<uint64_t>(node.compatibleQueueKindMask));
		nodeHash = rg::HashCombine(nodeHash, static_cast<uint64_t>(node.criticality));
		for (size_t compatibleSlot : node.compatibleQueueSlots) {
			nodeHash = rg::HashCombine(nodeHash, static_cast<uint64_t>(compatibleSlot));
		}
		for (size_t pred : node.in) {
			nodeHash = rg::HashCombine(nodeHash, rg::HashCombine(0x1ull, static_cast<uint64_t>(pred)));
		}
		for (size_t succ : node.out) {
			nodeHash = rg::HashCombine(nodeHash, rg::HashCombine(0x2ull, static_cast<uint64_t>(succ)));
		}
		parts.nodeTopologyHash = rg::HashCombine(parts.nodeTopologyHash, nodeHash);
	}

	for (const auto& edge : m_dependencyEdgeIR) {
		parts.dependencyEdgeHash = rg::HashCombine(parts.dependencyEdgeHash, edge.provenanceKey);
	}

	parts.fullHash = rg::HashCombine(0ull, parts.frameStructureHash);
	parts.fullHash = rg::HashCombine(parts.fullHash, parts.frameScheduleHash);
	parts.fullHash = rg::HashCombine(parts.fullHash, parts.queueConfigHash);
	parts.fullHash = rg::HashCombine(parts.fullHash, parts.activeQueueHash);
	parts.fullHash = rg::HashCombine(parts.fullHash, parts.settingsHash);
	parts.fullHash = rg::HashCombine(parts.fullHash, parts.nodeTopologyHash);
	parts.fullHash = rg::HashCombine(parts.fullHash, parts.dependencyEdgeHash);
	return parts;
}

uint64_t RenderGraph::BuildScheduleCacheKey(const std::vector<Node>& nodes) const {
	return BuildScheduleCacheKeyParts(nodes).fullHash;
}

void RenderGraph::ApplyCachedScheduleIR(
	const ScheduleIR& schedule,
	std::vector<Node>& nodes)
{
	ZoneScopedN("RenderGraph::ApplyCachedScheduleIR");
	m_assignedQueueSlotsByFramePass.assign(m_framePasses.size(), 0);

	for (const auto& scheduled : schedule.passStream) {
		if (scheduled.nodeIndex >= nodes.size()) {
			continue;
		}

		auto& node = nodes[scheduled.nodeIndex];
		node.assignedQueueSlot = scheduled.queueSlot;
		if (node.passIndex < m_assignedQueueSlotsByFramePass.size()) {
			m_assignedQueueSlotsByFramePass[node.passIndex] = scheduled.queueSlot;
		}
	}
}

void RenderGraph::RecordCompileReuseEvent(
	std::string stage,
	std::string artifact,
	bool reused,
	uint64_t key,
	std::string reason)
{
	constexpr size_t kMaxCompileReuseEvents = 512;
	if (m_compileReuseEvents.size() >= kMaxCompileReuseEvents) {
		return;
	}

	m_compileReuseEvents.push_back(CompileReuseEvent{
		.stage = std::move(stage),
		.artifact = std::move(artifact),
		.reused = reused,
		.key = key,
		.reason = std::move(reason),
	});
}

RenderGraph::BarrierLoweringOutput RenderGraph::LowerBarriers(
	RenderGraph& rg,
	std::vector<AnyPassAndResources>& passes,
	std::vector<Node>& nodes,
	const BarrierLoweringInput& input)
{
	ZoneScopedN("RenderGraph::LowerBarriers");
	BarrierLoweringOutput output{};
	if (!input.schedule) {
		return output;
	}

	rg.batches.clear();
	rg.batches.emplace_back(rg.m_queueRegistry.SlotCount(), rg.m_frameSchedulingResourceCount);

	auto openSymbolicBatch = [&](unsigned int symbolicBatchId) -> PassBatch {
		PassBatch b(rg.m_queueRegistry.SlotCount(), rg.m_frameSchedulingResourceCount);
		for (size_t qi = 0; qi < rg.m_queueRegistry.SlotCount(); ++qi) {
			for (size_t phase = 0; phase < PassBatch::kSignalPhaseCount; ++phase) {
				const UINT64 symbolicValue =
					static_cast<UINT64>(symbolicBatchId > 0 ? symbolicBatchId - 1 : symbolicBatchId)
					* static_cast<UINT64>(PassBatch::kSignalPhaseCount)
					+ static_cast<UINT64>(phase)
					+ 1;
				b.SetQueueSignalFenceValue(static_cast<BatchSignalPhase>(phase), qi, symbolicValue);
			}
		}
		return b;
	};

	std::unordered_set<uint64_t> scratchTransitioned;
	std::unordered_set<size_t> scratchFallback;
	std::vector<ResourceTransition> scratchTransitions;

	for (const auto& symbolicBatch : input.schedule->batches) {
		PassBatch currentBatch = openSymbolicBatch(symbolicBatch.id);
		for (const auto& scheduled : symbolicBatch.passes) {
			if (scheduled.passIndex < rg.m_framePassSchedulingSummaries.size()) {
				const uint64_t requirementCount = static_cast<uint64_t>(rg.m_framePassSchedulingSummaries[scheduled.passIndex].requirements.size());
				rg.m_compileCacheStats.loweredRequirementCount += requirementCount;
				output.barriers.loweredRequirementCount += requirementCount;
			}
			CommitPassToBatch(
				rg,
				passes[scheduled.passIndex],
				nodes[scheduled.nodeIndex],
				symbolicBatch.id,
				currentBatch,
				scratchTransitioned,
				scratchFallback,
				scratchTransitions);
		}
		rg.batches.push_back(std::move(currentBatch));
	}

	for (unsigned int batchIndex = 1; batchIndex < rg.batches.size(); ++batchIndex) {
		const auto& batch = rg.batches[batchIndex];
		for (size_t qi = 0; qi < batch.QueueCount(); ++qi) {
			for (size_t phase = 0; phase < PassBatch::kSignalPhaseCount; ++phase) {
				output.barriers.signals.push_back(SymbolicFenceToken{
					.batch = batchIndex,
					.queueSlot = qi,
					.phase = static_cast<BatchSignalPhase>(phase),
					.symbolicValue = batch.GetQueueSignalFenceValue(static_cast<BatchSignalPhase>(phase), qi),
				});
			}
			for (size_t transitionPhase = 0; transitionPhase < PassBatch::kTransitionPhaseCount; ++transitionPhase) {
				const auto phase = static_cast<BatchTransitionPhase>(transitionPhase);
				const auto& transitions = batch.Transitions(qi, phase);
				const uint64_t transitionCount = static_cast<uint64_t>(transitions.size());
				output.barriers.transitionCount += transitionCount;
				output.barriers.transitionHash = rg::HashCombine(
					output.barriers.transitionHash,
					transitionCount);
				for (const auto& transition : transitions) {
					output.barriers.transitions.push_back(SymbolicTransitionOp{
						.batch = batchIndex,
						.queueSlot = qi,
						.phase = phase,
						.resourceID = transition.pResource ? transition.pResource->GetGlobalResourceID() : 0ull,
						.range = transition.range,
						.prevAccessType = transition.prevAccessType,
						.newAccessType = transition.newAccessType,
						.prevLayout = transition.prevLayout,
						.newLayout = transition.newLayout,
						.prevSyncState = transition.prevSyncState,
						.newSyncState = transition.newSyncState,
						.discard = transition.discard,
					});
				}
			}
			for (size_t waitPhase = 0; waitPhase < PassBatch::kWaitPhaseCount; ++waitPhase) {
				for (size_t src = 0; src < batch.QueueCount(); ++src) {
					if (batch.HasQueueWait(static_cast<BatchWaitPhase>(waitPhase), qi, src)) {
						const auto phase = static_cast<BatchWaitPhase>(waitPhase);
						++output.barriers.waitCount;
						output.barriers.waits.push_back(SymbolicQueueWaitOp{
							.batch = batchIndex,
							.dstQueueSlot = qi,
							.srcQueueSlot = src,
							.phase = phase,
							.symbolicValue = batch.GetQueueWaitFenceValue(phase, qi, src),
						});
						output.barriers.waitHash = rg::HashCombine(
							output.barriers.waitHash,
							batch.GetQueueWaitFenceValue(phase, qi, src));
					}
				}
			}
		}
	}

	return output;
}

std::vector<RenderGraph::CompiledSegmentDesc> RenderGraph::BuildCompiledSegmentDescriptorsFromSchedule(
	const ScheduleIR& schedule) const
{
	ZoneScopedN("RenderGraph::BuildCompiledSegmentDescriptorsFromSchedule");
	std::vector<CompiledSegmentDesc> segments;
	auto closeSegment = [&](CompiledSegmentDesc& segment) {
		if (segment.passCount == 0) {
			return;
		}
		segments.push_back(std::move(segment));
	};

	auto beginSegment = [&](std::string_view name, size_t firstPassStreamIndex, unsigned int firstBatch) {
		CompiledSegmentDesc segment{};
		segment.name = std::string(name);
		segment.id = rg::HashCombine(HashStringView64(name), static_cast<uint64_t>(segments.size()));
		segment.firstPassStreamIndex = firstPassStreamIndex;
		segment.firstBatch = firstBatch;
		segment.lastBatch = firstBatch;
		return segment;
	};

	CompiledSegmentDesc current{};
	std::string currentName;
	std::unordered_set<uint64_t> currentSeenResources;
	std::unordered_set<uint64_t> currentSeenBoundaryRanges;
	std::unordered_map<uint64_t, size_t> currentExitBoundaryIndexByKey;
	bool hasCurrent = false;

	for (size_t streamIndex = 0; streamIndex < schedule.passStream.size(); ++streamIndex) {
		const ScheduledPass& scheduled = schedule.passStream[streamIndex];
		const PassIR* passIR = scheduled.passIndex < m_compiledFrameProgram.passes.size()
			? &m_compiledFrameProgram.passes[scheduled.passIndex]
			: nullptr;
		const std::string_view passName = passIR ? std::string_view(passIR->name) : std::string_view{};
		const std::string_view nextName = SegmentNameForPass(passName);

		// Keep segment cuts batch-aligned so barrier summaries are attributed to
		// one segment instead of being counted once per intra-batch phase change.
		const bool sameSymbolicBatch = hasCurrent && scheduled.symbolicBatch == current.lastBatch;
		if (!hasCurrent || (!sameSymbolicBatch && std::string_view(currentName) != nextName)) {
			if (hasCurrent) {
				closeSegment(current);
			}
			currentName = std::string(nextName);
			current = beginSegment(nextName, streamIndex, scheduled.symbolicBatch);
			currentSeenResources.clear();
			currentSeenBoundaryRanges.clear();
			currentExitBoundaryIndexByKey.clear();
			hasCurrent = true;
		}

		current.passCount++;
		current.lastBatch = scheduled.symbolicBatch;
		if (scheduled.passIndex < m_framePassSchedulingSummaries.size()) {
			current.loweredRequirementCount += static_cast<uint64_t>(m_framePassSchedulingSummaries[scheduled.passIndex].requirements.size());
		}
		current.schedule.passStream.push_back(scheduled);
		current.schedule.structureHash = rg::HashCombine(current.schedule.structureHash, static_cast<uint64_t>(scheduled.passIndex));
		current.schedule.queueAssignmentHash = rg::HashCombine(
			current.schedule.queueAssignmentHash,
			rg::HashCombine(static_cast<uint64_t>(scheduled.passIndex), static_cast<uint64_t>(scheduled.queueSlot)));
		current.queueAssignmentHash = current.schedule.queueAssignmentHash;

		if (!passIR) {
			continue;
		}

		current.passes.push_back(passIR->id);
		current.segmentStructureHash = rg::HashCombine(current.segmentStructureHash, passIR->id.value);
		current.passContentHash = rg::HashCombine(current.passContentHash, passIR->barrierHash);

		auto accumulateAccess = [&](const auto& access) {
			const auto aliasIt = aliasPlacementSignatureByID.find(access.resourceID);
			const uint64_t aliasSignature = aliasIt != aliasPlacementSignatureByID.end() ? aliasIt->second : 0ull;
			const uint64_t boundaryKey = rg::HashCombine(access.resourceID, HashRangeSpec64(access.range));
			if (currentSeenResources.insert(access.resourceID).second) {
				current.touchedResourceIDs.push_back(access.resourceID);
			}
			if (currentSeenBoundaryRanges.insert(boundaryKey).second) {
				current.entryBoundaryStates.push_back(CompiledSegmentDesc::BoundaryStateEntry{
			.resourceID = access.resourceID,
			.range = access.range,
			.state = access.state,
			.aliasSignature = aliasSignature,
		});
	}

	auto [exitIt, inserted] = currentExitBoundaryIndexByKey.emplace(boundaryKey, current.exitBoundaryStates.size());
	if (inserted) {
				current.exitBoundaryStates.push_back(CompiledSegmentDesc::BoundaryStateEntry{
					.resourceID = access.resourceID,
					.range = access.range,
					.state = access.state,
					.aliasSignature = aliasSignature,
				});
			}
			else {
				auto& exitEntry = current.exitBoundaryStates[exitIt->second];
				exitEntry.state = access.state;
				exitEntry.aliasSignature = aliasSignature;
			}

			if (aliasSignature != 0) {
				current.aliasSignatureHash = rg::HashCombine(current.aliasSignatureHash, aliasSignature);
			}
		};

		for (const auto& access : passIR->accesses) {
			accumulateAccess(access);
		}
		for (const auto& access : passIR->internalTransitions) {
			accumulateAccess(access);
		}
	}

	if (hasCurrent) {
		closeSegment(current);
	}

	for (auto& segment : segments) {
		auto sortBoundaryStates = [](std::vector<CompiledSegmentDesc::BoundaryStateEntry>& states) {
			std::sort(
				states.begin(),
				states.end(),
				[](const auto& lhs, const auto& rhs) {
					if (lhs.resourceID != rhs.resourceID) {
						return lhs.resourceID < rhs.resourceID;
					}
					const uint64_t lhsRange = HashRangeSpec64(lhs.range);
					const uint64_t rhsRange = HashRangeSpec64(rhs.range);
					if (lhsRange != rhsRange) {
						return lhsRange < rhsRange;
					}
					return lhs.aliasSignature < rhs.aliasSignature;
				});
		};
		sortBoundaryStates(segment.entryBoundaryStates);
		sortBoundaryStates(segment.exitBoundaryStates);
		for (const auto& symbolicBatch : schedule.batches) {
			if (symbolicBatch.id < segment.firstBatch || symbolicBatch.id > segment.lastBatch) {
				continue;
			}
			segment.schedule.batches.push_back(symbolicBatch);
		}
	}

	std::sort(
		segments.begin(),
		segments.end(),
		[](const CompiledSegmentDesc& lhs, const CompiledSegmentDesc& rhs) {
			if (lhs.firstBatch != rhs.firstBatch) {
				return lhs.firstBatch < rhs.firstBatch;
			}
			if (lhs.firstPassStreamIndex != rhs.firstPassStreamIndex) {
				return lhs.firstPassStreamIndex < rhs.firstPassStreamIndex;
			}
			if (lhs.lastBatch != rhs.lastBatch) {
				return lhs.lastBatch < rhs.lastBatch;
			}
			return lhs.name < rhs.name;
		});

	return segments;
}

std::vector<RenderGraph::SegmentPlan> RenderGraph::BuildSegmentPlans(
	const std::vector<CompiledSegmentDesc>& segmentDescs) const
{
	std::vector<SegmentPlan> plans;
	plans.reserve(segmentDescs.size());
	auto summarizeCompatibleCachedSegments = [&](const CompiledSegmentDesc& desc) -> std::string {
		size_t compatibleCount = 0;
		std::vector<uint64_t> sampleEntryHashes;
		sampleEntryHashes.reserve(4);
		for (const auto& [ignoredKey, cached] : m_cachedBarrierSegments) {
			(void)ignoredKey;
			if (cached.segmentStructureHash != desc.segmentStructureHash
				|| cached.passContentHash != desc.passContentHash
				|| cached.aliasSignatureHash != desc.aliasSignatureHash
				|| cached.queueAssignmentHash != desc.queueAssignmentHash) {
				continue;
			}

			++compatibleCount;
			if (sampleEntryHashes.size() < 4
				&& std::find(sampleEntryHashes.begin(), sampleEntryHashes.end(), cached.entryStateHash) == sampleEntryHashes.end()) {
				sampleEntryHashes.push_back(cached.entryStateHash);
			}
		}

		if (compatibleCount == 0) {
			return "no structurally compatible cached barrier segments";
		}

		std::ostringstream oss;
		oss << "structurally compatible cached segments found but entry-state hash differed: compatible_count="
			<< compatibleCount
			<< " descriptor_entry_hash=" << desc.entryStateHash
			<< " sample_cached_entry_hashes=[";
		for (size_t i = 0; i < sampleEntryHashes.size(); ++i) {
			if (i != 0) {
				oss << ", ";
			}
			oss << sampleEntryHashes[i];
		}
		oss << "]";
		return oss.str();
	};
	for (const auto& desc : segmentDescs) {
		const uint64_t cacheKey = ComputeBarrierSegmentCacheKey(desc);
		const bool cacheHit = m_cachedBarrierSegments.find(cacheKey) != m_cachedBarrierSegments.end();
		const bool includeDetailedMissReason = m_getRenderGraphCompileDumpEnabled && m_getRenderGraphCompileDumpEnabled();
		SegmentPlan plan{};
		plan.kind = cacheHit ? SegmentPlanKind::Replay : SegmentPlanKind::Lower;
		plan.segmentID = desc.id;
		plan.name = desc.name;
		plan.firstPassStreamIndex = desc.firstPassStreamIndex;
		plan.passCount = desc.passCount;
		plan.firstBatch = desc.firstBatch;
		plan.lastBatch = desc.lastBatch;
		plan.cacheKey = cacheKey;
		plan.loweredRequirementCount = desc.loweredRequirementCount;
		plan.reason = cacheHit
			? "cached barrier segment available for current descriptor cache key"
			: (includeDetailedMissReason ? summarizeCompatibleCachedSegments(desc) : std::string{});
		plans.push_back(std::move(plan));
	}
	return plans;
}

uint64_t RenderGraph::ComputeBarrierSegmentCacheKey(const CompiledSegmentDesc& desc) const
{
	return ComputeBarrierSegmentCacheKey(desc, desc.entryStateHash);
}

uint64_t RenderGraph::ComputeBarrierSegmentCacheKey(const CompiledSegmentDesc& desc, uint64_t entryStateHash) const
{
	uint64_t cacheKey = desc.segmentStructureHash;
	cacheKey = rg::HashCombine(cacheKey, desc.passContentHash);
	cacheKey = rg::HashCombine(cacheKey, desc.aliasSignatureHash);
	cacheKey = rg::HashCombine(cacheKey, desc.queueAssignmentHash);
	cacheKey = rg::HashCombine(cacheKey, entryStateHash);
	cacheKey = rg::HashCombine(cacheKey, static_cast<uint64_t>(m_queueRegistry.SlotCount()));
	cacheKey = rg::HashCombine(cacheKey, kBarrierSegmentCachePolicyVersion);
	return cacheKey;
}

uint64_t RenderGraph::HashBoundaryStateEntries(std::span<const CompiledSegmentDesc::BoundaryStateEntry> boundaryStates) const
{
	auto lessBoundary = [](const auto& a, const auto& b) {
		if (a.resourceID != b.resourceID) return a.resourceID < b.resourceID;
		if (a.range.mipLower.type != b.range.mipLower.type) return a.range.mipLower.type < b.range.mipLower.type;
		if (a.range.mipLower.value != b.range.mipLower.value) return a.range.mipLower.value < b.range.mipLower.value;
		if (a.range.mipUpper.type != b.range.mipUpper.type) return a.range.mipUpper.type < b.range.mipUpper.type;
		if (a.range.mipUpper.value != b.range.mipUpper.value) return a.range.mipUpper.value < b.range.mipUpper.value;
		if (a.range.sliceLower.type != b.range.sliceLower.type) return a.range.sliceLower.type < b.range.sliceLower.type;
		if (a.range.sliceLower.value != b.range.sliceLower.value) return a.range.sliceLower.value < b.range.sliceLower.value;
		if (a.range.sliceUpper.type != b.range.sliceUpper.type) return a.range.sliceUpper.type < b.range.sliceUpper.type;
		if (a.range.sliceUpper.value != b.range.sliceUpper.value) return a.range.sliceUpper.value < b.range.sliceUpper.value;
		return a.aliasSignature < b.aliasSignature;
	};
	auto hashBoundary = [](uint64_t hash, const auto& boundary) {
		hash = rg::HashCombine(hash, boundary.resourceID);
		hash = rg::HashCombine(hash, boundary.aliasSignature);
		hash = rg::HashCombine(hash, static_cast<uint64_t>(boundary.range.mipLower.type));
		hash = rg::HashCombine(hash, boundary.range.mipLower.value);
		hash = rg::HashCombine(hash, static_cast<uint64_t>(boundary.range.mipUpper.type));
		hash = rg::HashCombine(hash, boundary.range.mipUpper.value);
		hash = rg::HashCombine(hash, static_cast<uint64_t>(boundary.range.sliceLower.type));
		hash = rg::HashCombine(hash, boundary.range.sliceLower.value);
		hash = rg::HashCombine(hash, static_cast<uint64_t>(boundary.range.sliceUpper.type));
		hash = rg::HashCombine(hash, boundary.range.sliceUpper.value);
		hash = rg::HashCombine(hash, static_cast<uint64_t>(boundary.state.access));
		hash = rg::HashCombine(hash, static_cast<uint64_t>(boundary.state.layout));
		hash = rg::HashCombine(hash, static_cast<uint64_t>(boundary.state.sync));
		return hash;
	};

	uint64_t hash = 0;
	if (std::is_sorted(boundaryStates.begin(), boundaryStates.end(), lessBoundary)) {
		for (const auto& boundary : boundaryStates) {
			hash = hashBoundary(hash, boundary);
		}
		return hash;
	}

	std::vector<size_t> order(boundaryStates.size());
	std::iota(order.begin(), order.end(), 0);
	std::sort(
		order.begin(),
		order.end(),
		[&](size_t lhs, size_t rhs) {
			return lessBoundary(boundaryStates[lhs], boundaryStates[rhs]);
		});

	for (size_t index : order) {
		const auto& boundary = boundaryStates[index];
		hash = hashBoundary(hash, boundary);
	}

	return hash;
}

std::vector<RenderGraph::CompiledSegmentDesc::BoundaryStateEntry> RenderGraph::CaptureActualBoundaryStates(
	std::span<const CompiledSegmentDesc::BoundaryStateEntry> boundaryStates)
{
	std::vector<CompiledSegmentDesc::BoundaryStateEntry> captured;
	captured.reserve(boundaryStates.size());
	for (const auto& boundary : boundaryStates) {
		Resource* resource = nullptr;
		if (auto resourceRef = GetResourceByID(boundary.resourceID)) {
			resource = resourceRef.get();
		}

		auto& tracker = GetOrCreateCompileTracker(resource, boundary.resourceID);
		std::vector<Segment> segments = tracker.Query(boundary.range);
		if (segments.empty()) {
			captured.push_back(boundary);
			continue;
		}

		for (const auto& segment : segments) {
			captured.push_back(CompiledSegmentDesc::BoundaryStateEntry{
				.resourceID = boundary.resourceID,
				.range = segment.rangeSpec,
				.state = segment.state,
				.aliasSignature = boundary.aliasSignature,
			});
		}
	}

	return captured;
}

void RenderGraph::ApplyExactBoundaryStates(
	std::span<const CompiledSegmentDesc::BoundaryStateEntry> boundaryStates)
{
	for (const auto& boundary : boundaryStates) {
		Resource* resource = nullptr;
		if (auto resourceRef = GetResourceByID(boundary.resourceID)) {
			resource = resourceRef.get();
		}

		auto& tracker = GetOrCreateCompileTracker(resource, boundary.resourceID);
		tracker.SetExact(boundary.range, boundary.state);
		aliasActivationPending.erase(boundary.resourceID);
	}
}

RenderGraph::BarrierIR RenderGraph::BuildBarrierIRFromCompiledSegments(const std::vector<CompiledSegment>& segments) const
{
	BarrierIR barriers{};
	for (const auto& segment : segments) {
		barriers.loweredRequirementCount += segment.loweredRequirementCount;
		barriers.transitionCount += segment.barriers.transitionCount;
		barriers.waitCount += segment.barriers.waitCount;
		barriers.transitionHash = rg::HashCombine(barriers.transitionHash, segment.barriers.transitionHash);
		barriers.waitHash = rg::HashCombine(barriers.waitHash, segment.barriers.waitHash);
		barriers.signals.insert(barriers.signals.end(), segment.barriers.signals.begin(), segment.barriers.signals.end());
		barriers.transitions.insert(barriers.transitions.end(), segment.barriers.transitions.begin(), segment.barriers.transitions.end());
		barriers.waits.insert(barriers.waits.end(), segment.barriers.waits.begin(), segment.barriers.waits.end());
	}
	return barriers;
}

bool RenderGraph::TryBuildCompiledSegmentsFromCache(
	const std::vector<CompiledSegmentDesc>& segmentDescs,
	std::vector<CompiledSegment>& outSegments,
	bool recordReuseEvents)
{
	outSegments.clear();
	outSegments.reserve(segmentDescs.size());

	for (const auto& desc : segmentDescs) {
		const uint64_t cacheKey = ComputeBarrierSegmentCacheKey(desc);
		auto cacheIt = m_cachedBarrierSegments.find(cacheKey);
		if (cacheIt == m_cachedBarrierSegments.end()) {
			if (recordReuseEvents) {
				RecordCompileReuseEvent(
					"BarrierSegment",
					desc.name,
					false,
					cacheKey,
					"no cached barrier segment for structure/pass/alias/queue/entry-state signature");
			}
			outSegments.clear();
			return false;
		}

		const CompiledSegment& cached = cacheIt->second;
		CompiledSegment segment{};
		segment.id = desc.id;
		segment.name = desc.name;
		segment.firstPassStreamIndex = desc.firstPassStreamIndex;
		segment.passCount = desc.passCount;
		segment.firstBatch = desc.firstBatch;
		segment.lastBatch = desc.lastBatch;
		segment.passes = desc.passes;
		segment.touchedResourceIDs = desc.touchedResourceIDs;
		segment.entryBoundaryStates = desc.entryBoundaryStates;
		segment.exitBoundaryStates = desc.exitBoundaryStates;
		segment.segmentStructureHash = desc.segmentStructureHash;
		segment.passContentHash = desc.passContentHash;
		segment.aliasSignatureHash = desc.aliasSignatureHash;
		segment.queueAssignmentHash = desc.queueAssignmentHash;
		segment.entryStateHash = desc.entryStateHash;
		segment.exitStateHash = desc.exitStateHash;
		segment.loweredRequirementCount = desc.loweredRequirementCount;
		segment.schedule = desc.schedule;
		segment.barriers = cached.barriers;
		segment.replay = cached.replay;
		segment.cacheKey = cacheKey;
		segment.barrierCacheHit = true;
		segment.barriersReused = true;
		segment.barrierReuseReason = "symbolic barrier metadata reused from matching segment; concrete PassBatch materialization replayed cached segment";
		outSegments.push_back(std::move(segment));

		if (recordReuseEvents) {
			RecordCompileReuseEvent(
				"BarrierSegment",
				desc.name,
				true,
				cacheKey,
				"symbolic barrier metadata reused from matching segment; concrete PassBatch materialization replayed cached segment");
		}
	}

	return true;
}

RenderGraph::CachedBarrierSegmentReplay RenderGraph::BuildCachedBarrierSegmentReplay(
	const CompiledSegment& segment) const
{
	ZoneScopedN("RenderGraph::BuildCachedBarrierSegmentReplay");
	CachedBarrierSegmentReplay replay{};
	if (segment.lastBatch < segment.firstBatch) {
		return replay;
	}

	replay.localBatchCount = segment.lastBatch - segment.firstBatch + 1;
	replay.signals.reserve(segment.barriers.signals.size());
	replay.transitions.reserve(segment.barriers.transitions.size());
	replay.waits.reserve(segment.barriers.waits.size());
	replay.batchMembership.reserve(replay.localBatchCount);

	std::unordered_map<UINT64, CachedBarrierSignalRef> signalRefByQueueAndValue;
	signalRefByQueueAndValue.reserve(segment.barriers.signals.size());
	std::unordered_map<UINT64, CachedBarrierSignalToken> enabledSignalTokenByQueueAndValue;
	for (unsigned int batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
		const auto& batch = batches[batchIndex];
		for (size_t queueSlot = 0; queueSlot < batch.QueueCount(); ++queueSlot) {
			for (size_t phaseIndex = 0; phaseIndex < PassBatch::kSignalPhaseCount; ++phaseIndex) {
				const auto phase = static_cast<BatchSignalPhase>(phaseIndex);
				if (!batch.HasQueueSignal(phase, queueSlot)) {
					continue;
				}
				const UINT64 symbolicValue = batch.GetQueueSignalFenceValue(phase, queueSlot);
				if (symbolicValue == 0) {
					continue;
				}
				const UINT64 signalKey = (static_cast<UINT64>(queueSlot) << 56) | symbolicValue;
				enabledSignalTokenByQueueAndValue[signalKey] = CachedBarrierSignalToken{
					.batch = batchIndex,
					.queueSlot = static_cast<uint16_t>(queueSlot),
					.phase = phase,
				};
			}
		}
	}

	for (const auto& signal : segment.barriers.signals) {
		if (signal.batch < segment.firstBatch || signal.batch > segment.lastBatch) {
			continue;
		}

		const CachedBarrierSignalRef signalRef{
			.localBatch = static_cast<uint16_t>(signal.batch - segment.firstBatch),
			.queueSlot = static_cast<uint16_t>(signal.queueSlot),
			.phase = signal.phase,
		};
		replay.signals.push_back(CachedBarrierSignalOp{
			.signal = signalRef,
			.symbolicValue = signal.symbolicValue,
			.enabled = signal.batch < batches.size() && batches[signal.batch].HasQueueSignal(signal.phase, signal.queueSlot),
		});

		const bool signalEnabled = signal.batch < batches.size() && batches[signal.batch].HasQueueSignal(signal.phase, signal.queueSlot);
		if (signalEnabled) {
			const UINT64 signalKey = (static_cast<UINT64>(signal.queueSlot) << 56) | signal.symbolicValue;
			signalRefByQueueAndValue[signalKey] = signalRef;
		}
	}

	for (const auto& transition : segment.barriers.transitions) {
		if (transition.batch < segment.firstBatch || transition.batch > segment.lastBatch) {
			continue;
		}

		replay.transitions.push_back(CachedBarrierTransitionOp{
			.localBatch = static_cast<uint16_t>(transition.batch - segment.firstBatch),
			.queueSlot = static_cast<uint16_t>(transition.queueSlot),
			.phase = transition.phase,
			.resourceID = transition.resourceID,
			.range = transition.range,
			.before = ResourceState{
				.access = transition.prevAccessType,
				.layout = transition.prevLayout,
				.sync = transition.prevSyncState,
			},
			.after = ResourceState{
				.access = transition.newAccessType,
				.layout = transition.newLayout,
				.sync = transition.newSyncState,
			},
			.discard = transition.discard,
		});
	}

	for (const auto& wait : segment.barriers.waits) {
		if (wait.batch < segment.firstBatch || wait.batch > segment.lastBatch) {
			continue;
		}

		const UINT64 signalKey = (static_cast<UINT64>(wait.srcQueueSlot) << 56) | wait.symbolicValue;
		const auto signalIt = signalRefByQueueAndValue.find(signalKey);
		const auto globalSignalIt = enabledSignalTokenByQueueAndValue.find(signalKey);
		replay.waits.push_back(CachedBarrierWaitOp{
			.localBatch = static_cast<uint16_t>(wait.batch - segment.firstBatch),
			.dstQueueSlot = static_cast<uint16_t>(wait.dstQueueSlot),
			.srcQueueSlot = static_cast<uint16_t>(wait.srcQueueSlot),
			.phase = wait.phase,
			.sourceSignal = signalIt != signalRefByQueueAndValue.end() ? signalIt->second : CachedBarrierSignalRef{},
			.globalSourceSignal = globalSignalIt != enabledSignalTokenByQueueAndValue.end() ? globalSignalIt->second : CachedBarrierSignalToken{},
			.symbolicValue = wait.symbolicValue,
			.sourceSignalInSegment = signalIt != signalRefByQueueAndValue.end(),
			.sourceSignalGlobalKnown = globalSignalIt != enabledSignalTokenByQueueAndValue.end(),
		});
	}

	std::unordered_map<uint64_t, size_t> resourceIndexByID;
	resourceIndexByID.reserve(segment.touchedResourceIDs.size());
	for (uint64_t resourceID : segment.touchedResourceIDs) {
		auto resourceIndex = TryGetFrameSchedulingResourceIndex(resourceID);
		if (!resourceIndex.has_value()) {
			continue;
		}
		resourceIndexByID.emplace(resourceID, *resourceIndex);
	}

	for (unsigned int batchIndex = segment.firstBatch; batchIndex <= segment.lastBatch && batchIndex < batches.size(); ++batchIndex) {
		const auto& batch = batches[batchIndex];
		replay.batchMembership.push_back(CachedBatchMembership{
			.localBatch = static_cast<uint16_t>(batchIndex - segment.firstBatch),
			.allResources = batch.allResources,
			.internallyTransitionedResources = batch.internallyTransitionedResources,
		});
	}

	for (const auto& event : m_frameResourceEventLog) {
		if (event.batchIndex < segment.firstBatch || event.batchIndex > segment.lastBatch) {
			continue;
		}
		if (!resourceIndexByID.contains(event.resourceID)) {
			continue;
		}
		replay.resourceEvents.push_back(CachedResourceEvent{
			.localBatch = static_cast<uint16_t>(event.batchIndex - segment.firstBatch),
			.queueSlot = event.queueSlot,
			.resourceID = event.resourceID,
		});
	}

	auto appendHistoryEvent = [&](CachedHistoryEvent::Kind kind, unsigned int batchIndex, size_t queueSlot, uint64_t resourceID) {
		if (batchIndex < segment.firstBatch || batchIndex > segment.lastBatch) {
			return;
		}
		replay.historyEvents.push_back(CachedHistoryEvent{
			.kind = kind,
			.localBatch = static_cast<uint16_t>(batchIndex - segment.firstBatch),
			.queueSlot = static_cast<uint16_t>(queueSlot),
			.resourceID = resourceID,
		});
	};

	for (const auto& [resourceID, resourceIndex] : resourceIndexByID) {
		for (size_t queueSlot = 0; queueSlot < m_queueRegistry.SlotCount(); ++queueSlot) {
			appendHistoryEvent(
				CachedHistoryEvent::Kind::Usage,
				GetFrameQueueHistoryValue(m_frameQueueLastUsageBatch, queueSlot, resourceIndex),
				queueSlot,
				resourceID);
			appendHistoryEvent(
				CachedHistoryEvent::Kind::Producer,
				GetFrameQueueHistoryValue(m_frameQueueLastProducerBatch, queueSlot, resourceIndex),
				queueSlot,
				resourceID);
			appendHistoryEvent(
				CachedHistoryEvent::Kind::QueueTransition,
				GetFrameQueueHistoryValue(m_frameQueueLastTransitionBatch, queueSlot, resourceIndex),
				queueSlot,
				resourceID);
		}

		if (resourceIndex >= m_frameTransitionPlacementBatchesByResource.size()) {
			continue;
		}

		for (unsigned int placementBatch : m_frameTransitionPlacementBatchesByResource[resourceIndex]) {
			appendHistoryEvent(CachedHistoryEvent::Kind::TransitionPlacement, placementBatch, 0, resourceID);
		}
	}

	std::sort(
		replay.historyEvents.begin(),
		replay.historyEvents.end(),
		[](const CachedHistoryEvent& lhs, const CachedHistoryEvent& rhs) {
			if (lhs.localBatch != rhs.localBatch) {
				return lhs.localBatch < rhs.localBatch;
			}
			if (lhs.kind != rhs.kind) {
				return static_cast<uint8_t>(lhs.kind) < static_cast<uint8_t>(rhs.kind);
			}
			if (lhs.queueSlot != rhs.queueSlot) {
				return lhs.queueSlot < rhs.queueSlot;
			}
			return lhs.resourceID < rhs.resourceID;
		});

	return replay;
}

std::vector<RenderGraph::PassBatch> RenderGraph::BuildReplayedSegmentBatches(
	const CompiledSegment& segment,
	const std::unordered_map<uint64_t, UINT64>* materializedSignalValuesByToken,
	const std::unordered_set<uint64_t>* materializedEnabledSignalTokens)
{
	ZoneScopedN("RenderGraph::BuildReplayedSegmentBatches");
	std::vector<PassBatch> replayedBatches;
	replayedBatches.reserve(segment.replay.localBatchCount);
	for (uint32_t localBatch = 0; localBatch < segment.replay.localBatchCount; ++localBatch) {
		replayedBatches.emplace_back(m_queueRegistry.SlotCount(), m_frameSchedulingResourceCount);
	}

	auto makeQueuedPass = [&](AnyPassAndResources& any) -> PassBatch::QueuedPass {
		return std::visit(
			[&](auto& pass) -> PassBatch::QueuedPass {
				using PassT = std::decay_t<decltype(pass)>;
				if constexpr (std::is_same_v<PassT, std::monostate>) {
					throw std::runtime_error("Unexpected empty pass variant in RenderGraph::BuildReplayedSegmentBatches");
				}
				else {
					return &pass;
				}
			},
			any.pass);
	};

	for (const auto& scheduled : segment.schedule.passStream) {
		if (scheduled.passIndex >= m_framePasses.size()) {
			continue;
		}
		if (scheduled.symbolicBatch < segment.firstBatch || scheduled.symbolicBatch > segment.lastBatch) {
			continue;
		}

		AnyPassAndResources& any = m_framePasses[scheduled.passIndex];
		PassBatch& batch = replayedBatches[scheduled.symbolicBatch - segment.firstBatch];
		batch.Passes(scheduled.queueSlot).push_back(makeQueuedPass(any));
	}

	for (const auto& membership : segment.replay.batchMembership) {
		if (membership.localBatch >= replayedBatches.size()) {
			continue;
		}

		PassBatch& batch = replayedBatches[membership.localBatch];
		batch.allResources = membership.allResources;
		batch.internallyTransitionedResources = membership.internallyTransitionedResources;
	}

	for (const auto& signal : segment.replay.signals) {
		if (signal.signal.localBatch >= replayedBatches.size()) {
			continue;
		}

		PassBatch& batch = replayedBatches[signal.signal.localBatch];
		batch.SetQueueSignalFenceValue(signal.signal.phase, static_cast<size_t>(signal.signal.queueSlot), signal.symbolicValue);
		if (signal.enabled) {
			batch.MarkQueueSignal(signal.signal.phase, static_cast<size_t>(signal.signal.queueSlot));
		}
	}

	for (const auto& transition : segment.replay.transitions) {
		if (transition.localBatch >= replayedBatches.size()) {
			continue;
		}

		PassBatch& batch = replayedBatches[transition.localBatch];
		Resource* resource = nullptr;
		auto resourceRef = GetResourceByID(transition.resourceID);
		if (resourceRef) {
			resource = resourceRef.get();
		}

		auto& transitions = batch.Transitions(static_cast<size_t>(transition.queueSlot), transition.phase);
		transitions.emplace_back(
			resource,
			transition.range,
			transition.before.access,
			transition.after.access,
			transition.before.layout,
			transition.after.layout,
			transition.before.sync,
			transition.after.sync,
			transition.discard);
	}

	std::unordered_map<uint64_t, UINT64> signalValueByRefKey;
	signalValueByRefKey.reserve(segment.replay.signals.size());
	auto encodeSignalRef = [](const CachedBarrierSignalRef& signalRef) -> uint64_t {
		uint64_t key = 0;
		key = rg::HashCombine(key, static_cast<uint64_t>(signalRef.localBatch));
		key = rg::HashCombine(key, static_cast<uint64_t>(signalRef.queueSlot));
		key = rg::HashCombine(key, static_cast<uint64_t>(signalRef.phase));
		return key;
	};
	auto encodeSignalToken = [](const CachedBarrierSignalToken& signalToken) -> uint64_t {
		uint64_t key = 0;
		key = rg::HashCombine(key, static_cast<uint64_t>(signalToken.batch));
		key = rg::HashCombine(key, static_cast<uint64_t>(signalToken.queueSlot));
		key = rg::HashCombine(key, static_cast<uint64_t>(signalToken.phase));
		return key;
	};
	for (const auto& signal : segment.replay.signals) {
		signalValueByRefKey[encodeSignalRef(signal.signal)] = signal.symbolicValue;
	}

	for (const auto& wait : segment.replay.waits) {
		if (wait.localBatch >= replayedBatches.size()) {
			continue;
		}

		UINT64 waitValue = wait.symbolicValue;
		if (wait.sourceSignalInSegment) {
			auto signalIt = signalValueByRefKey.find(encodeSignalRef(wait.sourceSignal));
			if (signalIt == signalValueByRefKey.end()) {
				continue;
			}
			waitValue = signalIt->second;
		}
		else if (wait.sourceSignalGlobalKnown) {
			if (materializedSignalValuesByToken == nullptr || materializedEnabledSignalTokens == nullptr) {
				throw std::runtime_error("Segment replay external wait requires a materialized signal registry");
			}

			const uint64_t signalTokenKey = encodeSignalToken(wait.globalSourceSignal);
			if (!materializedEnabledSignalTokens->contains(signalTokenKey)) {
				throw std::runtime_error("Segment replay external wait references a disabled materialized signal");
			}

			auto signalIt = materializedSignalValuesByToken->find(signalTokenKey);
			if (signalIt == materializedSignalValuesByToken->end()) {
				throw std::runtime_error("Segment replay external wait references an unresolved materialized signal");
			}
			waitValue = signalIt->second;
		}
		else {
			throw std::runtime_error("Segment replay external wait has no source signal token");
		}

		PassBatch& batch = replayedBatches[wait.localBatch];
		batch.AddQueueWait(
			wait.phase,
			static_cast<size_t>(wait.dstQueueSlot),
			static_cast<size_t>(wait.srcQueueSlot),
			waitValue);
	}

	return replayedBatches;
}

void RenderGraph::ReplaySegmentIntoFrameBatches(
	const CompiledSegment& segment,
	std::vector<UINT64>& materializedSignalValuesByToken,
	std::vector<uint8_t>& materializedEnabledSignalTokens)
{
	ZoneScopedN("RenderGraph::ReplaySegmentIntoFrameBatches");
	while (batches.size() <= segment.lastBatch) {
		batches.emplace_back(m_queueRegistry.SlotCount(), m_frameSchedulingResourceCount);
	}
	for (uint32_t localBatch = 0; localBatch < segment.replay.localBatchCount; ++localBatch) {
		const unsigned int globalBatch = segment.firstBatch + localBatch;
		batches[globalBatch].ResetForReplay(m_queueRegistry.SlotCount());
	}

	auto makeQueuedPass = [&](AnyPassAndResources& any) -> PassBatch::QueuedPass {
		return std::visit(
			[&](auto& pass) -> PassBatch::QueuedPass {
				using PassT = std::decay_t<decltype(pass)>;
				if constexpr (std::is_same_v<PassT, std::monostate>) {
					throw std::runtime_error("Unexpected empty pass variant in RenderGraph::ReplaySegmentIntoFrameBatches");
				}
				else {
					return &pass;
				}
			},
			any.pass);
	};

	for (const auto& scheduled : segment.schedule.passStream) {
		if (scheduled.passIndex >= m_framePasses.size()) {
			continue;
		}
		if (scheduled.symbolicBatch < segment.firstBatch || scheduled.symbolicBatch > segment.lastBatch) {
			continue;
		}
		AnyPassAndResources& any = m_framePasses[scheduled.passIndex];
		PassBatch& batch = batches[scheduled.symbolicBatch];
		batch.Passes(scheduled.queueSlot).push_back(makeQueuedPass(any));
	}

	for (const auto& membership : segment.replay.batchMembership) {
		if (membership.localBatch >= segment.replay.localBatchCount) {
			continue;
		}
		PassBatch& batch = batches[segment.firstBatch + membership.localBatch];
		batch.allResources = membership.allResources;
		batch.internallyTransitionedResources = membership.internallyTransitionedResources;
	}

	for (const auto& signal : segment.replay.signals) {
		if (signal.signal.localBatch >= segment.replay.localBatchCount) {
			continue;
		}
		PassBatch& batch = batches[segment.firstBatch + signal.signal.localBatch];
		const size_t queueSlot = static_cast<size_t>(signal.signal.queueSlot);
		batch.SetQueueSignalFenceValue(signal.signal.phase, queueSlot, signal.symbolicValue);
		if (signal.enabled) {
			batch.MarkQueueSignal(signal.signal.phase, queueSlot);
		}
	}

	for (const auto& transition : segment.replay.transitions) {
		if (transition.localBatch >= segment.replay.localBatchCount) {
			continue;
		}
		PassBatch& batch = batches[segment.firstBatch + transition.localBatch];
		Resource* resource = nullptr;
		if (auto resourceRef = GetResourceByID(transition.resourceID)) {
			resource = resourceRef.get();
		}

		auto resourceIndex = TryGetFrameSchedulingResourceIndex(transition.resourceID);
		const size_t transitionResourceIndex = resourceIndex.value_or(0);
		auto& transitions = batch.Transitions(static_cast<size_t>(transition.queueSlot), transition.phase);
		transitions.emplace_back(
			resource,
			transition.range,
			transition.before.access,
			transition.after.access,
			transition.before.layout,
			transition.after.layout,
			transition.before.sync,
			transition.after.sync,
			transition.discard);
		if (resourceIndex.has_value()) {
			batch.RecordTransitionPosition(
				static_cast<size_t>(transition.queueSlot),
				transition.phase,
				transitionResourceIndex,
				static_cast<uint32_t>(transitions.size() - 1));
		}
	}

	std::unordered_map<uint64_t, UINT64> signalValueByRefKey;
	signalValueByRefKey.reserve(segment.replay.signals.size());
	auto encodeSignalRef = [](const CachedBarrierSignalRef& signalRef) -> uint64_t {
		uint64_t key = 0;
		key = rg::HashCombine(key, static_cast<uint64_t>(signalRef.localBatch));
		key = rg::HashCombine(key, static_cast<uint64_t>(signalRef.queueSlot));
		key = rg::HashCombine(key, static_cast<uint64_t>(signalRef.phase));
		return key;
	};
	const size_t queueCount = m_queueRegistry.SlotCount();
	auto signalTokenIndex = [&](const CachedBarrierSignalToken& signalToken) -> size_t {
		return ((static_cast<size_t>(signalToken.batch) * queueCount) + static_cast<size_t>(signalToken.queueSlot))
			* PassBatch::kSignalPhaseCount
			+ static_cast<size_t>(signalToken.phase);
	};
	for (const auto& signal : segment.replay.signals) {
		signalValueByRefKey[encodeSignalRef(signal.signal)] = signal.symbolicValue;
	}

	for (const auto& wait : segment.replay.waits) {
		if (wait.localBatch >= segment.replay.localBatchCount) {
			continue;
		}

		UINT64 waitValue = wait.symbolicValue;
		if (wait.sourceSignalInSegment) {
			auto signalIt = signalValueByRefKey.find(encodeSignalRef(wait.sourceSignal));
			if (signalIt == signalValueByRefKey.end()) {
				continue;
			}
			waitValue = signalIt->second;
		}
		else if (wait.sourceSignalGlobalKnown) {
			const size_t signalIndex = signalTokenIndex(wait.globalSourceSignal);
			if (signalIndex >= materializedEnabledSignalTokens.size() || materializedEnabledSignalTokens[signalIndex] == 0) {
				throw std::runtime_error("Segment replay external wait references a disabled materialized signal");
			}
			if (signalIndex >= materializedSignalValuesByToken.size() || materializedSignalValuesByToken[signalIndex] == 0) {
				throw std::runtime_error("Segment replay external wait references an unresolved materialized signal");
			}
			waitValue = materializedSignalValuesByToken[signalIndex];
		}
		else {
			throw std::runtime_error("Segment replay external wait has no source signal token");
		}

		PassBatch& batch = batches[segment.firstBatch + wait.localBatch];
		batch.AddQueueWait(
			wait.phase,
			static_cast<size_t>(wait.dstQueueSlot),
			static_cast<size_t>(wait.srcQueueSlot),
			waitValue);
	}
}

std::vector<RenderGraph::PassBatch> RenderGraph::BuildReplayedFrameBatches()
{
	ZoneScopedN("RenderGraph::BuildReplayedFrameBatches");
	std::vector<PassBatch> replayedFrameBatches;
	replayedFrameBatches.reserve(batches.size());
	for (size_t batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
		replayedFrameBatches.emplace_back(m_queueRegistry.SlotCount(), m_frameSchedulingResourceCount);
	}

	std::unordered_map<uint64_t, UINT64> materializedSignalValuesByToken;
	std::unordered_set<uint64_t> materializedEnabledSignalTokens;
	auto encodeSignalToken = [](unsigned int batchIndex, size_t queueSlot, BatchSignalPhase phase) -> uint64_t {
		uint64_t key = 0;
		key = rg::HashCombine(key, static_cast<uint64_t>(batchIndex));
		key = rg::HashCombine(key, static_cast<uint64_t>(queueSlot));
		key = rg::HashCombine(key, static_cast<uint64_t>(phase));
		return key;
	};
	auto registerReplayedSignals = [&](unsigned int firstBatch, const std::vector<PassBatch>& segmentBatches) {
		for (size_t localBatch = 0; localBatch < segmentBatches.size(); ++localBatch) {
			const unsigned int globalBatch = firstBatch + static_cast<unsigned int>(localBatch);
			const auto& batch = segmentBatches[localBatch];
			for (size_t queueSlot = 0; queueSlot < batch.QueueCount(); ++queueSlot) {
				for (size_t phaseIndex = 0; phaseIndex < PassBatch::kSignalPhaseCount; ++phaseIndex) {
					const auto phase = static_cast<BatchSignalPhase>(phaseIndex);
					if (!batch.HasQueueSignal(phase, queueSlot)) {
						continue;
					}
					const UINT64 symbolicValue = batch.GetQueueSignalFenceValue(phase, queueSlot);
					if (symbolicValue == 0) {
						continue;
					}
					const uint64_t tokenKey = encodeSignalToken(globalBatch, queueSlot, phase);
					materializedSignalValuesByToken[tokenKey] = symbolicValue;
					materializedEnabledSignalTokens.insert(tokenKey);
				}
			}
		}
	};

	for (const auto& segment : m_compiledSegments) {
		std::vector<PassBatch> replayedSegmentBatches = BuildReplayedSegmentBatches(
			segment,
			&materializedSignalValuesByToken,
			&materializedEnabledSignalTokens);
		registerReplayedSignals(segment.firstBatch, replayedSegmentBatches);
		for (size_t localBatch = 0; localBatch < replayedSegmentBatches.size(); ++localBatch) {
			const unsigned int globalBatch = segment.firstBatch + static_cast<unsigned int>(localBatch);
			if (globalBatch >= replayedFrameBatches.size()) {
				continue;
			}
			replayedFrameBatches[globalBatch] = std::move(replayedSegmentBatches[localBatch]);
		}
	}

	return replayedFrameBatches;
}

void RenderGraph::ApplyReplayedSegmentCompilerState(const CompiledSegment& segment)
{
	ZoneScopedN("RenderGraph::ApplyReplayedSegmentCompilerState");

	std::unordered_map<uint64_t, size_t> frameResourceIndexByID;
	frameResourceIndexByID.reserve(segment.touchedResourceIDs.size());
	for (uint64_t resourceID : segment.touchedResourceIDs) {
		auto resourceIndex = TryGetFrameSchedulingResourceIndex(resourceID);
		if (resourceIndex.has_value()) {
			frameResourceIndexByID.emplace(resourceID, *resourceIndex);
		}
	}
	auto findResourceIndex = [&](uint64_t resourceID) -> std::optional<size_t> {
		auto it = frameResourceIndexByID.find(resourceID);
		if (it == frameResourceIndexByID.end()) {
			return std::nullopt;
		}
		return it->second;
	};

	for (const auto& event : segment.replay.resourceEvents) {
		auto resourceIndex = findResourceIndex(event.resourceID);
		if (!resourceIndex.has_value()) {
			continue;
		}

		RecordFrameResourceEvent(
			static_cast<size_t>(event.queueSlot),
			*resourceIndex,
			segment.firstBatch + event.localBatch);
	}

	for (const auto& event : segment.replay.historyEvents) {
		auto resourceIndex = findResourceIndex(event.resourceID);
		if (!resourceIndex.has_value()) {
			continue;
		}

		const unsigned int batchIndex = segment.firstBatch + event.localBatch;
		switch (event.kind) {
		case CachedHistoryEvent::Kind::Usage:
			SetFrameQueueHistoryValue(m_frameQueueLastUsageBatch, static_cast<size_t>(event.queueSlot), *resourceIndex, batchIndex);
			break;
		case CachedHistoryEvent::Kind::Producer:
			SetFrameQueueHistoryValue(m_frameQueueLastProducerBatch, static_cast<size_t>(event.queueSlot), *resourceIndex, batchIndex);
			break;
		case CachedHistoryEvent::Kind::QueueTransition:
			SetFrameQueueHistoryValue(m_frameQueueLastTransitionBatch, static_cast<size_t>(event.queueSlot), *resourceIndex, batchIndex);
			break;
		case CachedHistoryEvent::Kind::TransitionPlacement:
			RecordFrameTransitionPlacementBatch(*resourceIndex, batchIndex);
			break;
		}
	}

	ApplyExactBoundaryStates(segment.exitBoundaryStates);

	for (uint64_t resourceID : segment.touchedResourceIDs) {
		aliasActivationPending.erase(resourceID);
	}
}

void RenderGraph::RefreshCompiledSegmentBoundaryMetadataFromReplay()
{
	ZoneScopedN("RenderGraph::RefreshCompiledSegmentBoundaryMetadataFromReplay");
	if (m_compiledSegments.empty()) {
		return;
	}

	const auto savedCompileTrackers = compileTrackers;
	const auto savedAliasActivationPending = aliasActivationPending;
	const auto savedFrameQueueLastUsageBatch = m_frameQueueLastUsageBatch;
	const auto savedFrameQueueLastProducerBatch = m_frameQueueLastProducerBatch;
	const auto savedFrameQueueLastTransitionBatch = m_frameQueueLastTransitionBatch;
	const auto savedFrameTransitionPlacementBatchesByResource = m_frameTransitionPlacementBatchesByResource;
	const auto savedFrameResourceEventSummaries = m_frameResourceEventSummaries;
	const auto savedFrameResourceEventLog = m_frameResourceEventLog;

	auto computeSegmentCacheKey = [&](const CompiledSegment& segment) {
		uint64_t cacheKey = segment.segmentStructureHash;
		cacheKey = rg::HashCombine(cacheKey, segment.passContentHash);
		cacheKey = rg::HashCombine(cacheKey, segment.aliasSignatureHash);
		cacheKey = rg::HashCombine(cacheKey, segment.queueAssignmentHash);
		cacheKey = rg::HashCombine(cacheKey, segment.entryStateHash);
		cacheKey = rg::HashCombine(cacheKey, static_cast<uint64_t>(m_queueRegistry.SlotCount()));
		cacheKey = rg::HashCombine(cacheKey, kBarrierSegmentCachePolicyVersion);
		return cacheKey;
	};

	compileTrackers.clear();
	aliasActivationPending.clear();
	ResetFrameQueueBatchHistoryTables();
	m_actualEntryBoundaryStatesBySegmentId.clear();
	m_actualExitBoundaryStatesBySegmentId.clear();

	std::vector<uint64_t> staleCacheKeys;
	staleCacheKeys.reserve(m_compiledSegments.size());
	for (auto& segment : m_compiledSegments) {
		staleCacheKeys.push_back(segment.cacheKey);

		segment.entryBoundaryStates = CaptureActualBoundaryStates(segment.entryBoundaryStates);
		segment.entryStateHash = HashBoundaryStateEntries(segment.entryBoundaryStates);
		m_actualEntryBoundaryStatesBySegmentId[segment.id] = segment.entryBoundaryStates;

		ApplyReplayedSegmentCompilerState(segment);

		segment.exitBoundaryStates = CaptureActualBoundaryStates(segment.exitBoundaryStates);
		segment.exitStateHash = HashBoundaryStateEntries(segment.exitBoundaryStates);
		m_actualExitBoundaryStatesBySegmentId[segment.id] = segment.exitBoundaryStates;
		ApplyExactBoundaryStates(segment.exitBoundaryStates);

		segment.cacheKey = computeSegmentCacheKey(segment);
	}

	for (uint64_t staleCacheKey : staleCacheKeys) {
		m_cachedBarrierSegments.erase(staleCacheKey);
	}
	for (const auto& segment : m_compiledSegments) {
		m_cachedBarrierSegments[segment.cacheKey] = segment;
	}

	compileTrackers = savedCompileTrackers;
	aliasActivationPending = savedAliasActivationPending;
	m_frameQueueLastUsageBatch = savedFrameQueueLastUsageBatch;
	m_frameQueueLastProducerBatch = savedFrameQueueLastProducerBatch;
	m_frameQueueLastTransitionBatch = savedFrameQueueLastTransitionBatch;
	m_frameTransitionPlacementBatchesByResource = savedFrameTransitionPlacementBatchesByResource;
	m_frameResourceEventSummaries = savedFrameResourceEventSummaries;
	m_frameResourceEventLog = savedFrameResourceEventLog;
}

void RenderGraph::AppendLoweredScheduleBatches(
	RenderGraph& rg,
	std::vector<AnyPassAndResources>& passes,
	std::vector<Node>& nodes,
	const ScheduleIR& schedule,
	unsigned int firstBatchToLower)
{
	ZoneScopedN("RenderGraph::AppendLoweredScheduleBatches");

	auto openSymbolicBatch = [&](unsigned int symbolicBatchId) -> PassBatch {
		PassBatch b(rg.m_queueRegistry.SlotCount(), rg.m_frameSchedulingResourceCount);
		for (size_t qi = 0; qi < rg.m_queueRegistry.SlotCount(); ++qi) {
			for (size_t phase = 0; phase < PassBatch::kSignalPhaseCount; ++phase) {
				const UINT64 symbolicValue =
					static_cast<UINT64>(symbolicBatchId > 0 ? symbolicBatchId - 1 : symbolicBatchId)
					* static_cast<UINT64>(PassBatch::kSignalPhaseCount)
					+ static_cast<UINT64>(phase)
					+ 1;
				b.SetQueueSignalFenceValue(static_cast<BatchSignalPhase>(phase), qi, symbolicValue);
			}
		}
		return b;
	};

	auto ensureBatchSlot = [&](unsigned int batchIndex) {
		while (rg.batches.size() <= batchIndex) {
			rg.batches.emplace_back(rg.m_queueRegistry.SlotCount(), rg.m_frameSchedulingResourceCount);
		}
	};

	std::unordered_set<uint64_t> scratchTransitioned;
	std::unordered_set<size_t> scratchFallback;
	std::vector<ResourceTransition> scratchTransitions;

	for (const auto& symbolicBatch : schedule.batches) {
		if (symbolicBatch.id < firstBatchToLower) {
			continue;
		}

		PassBatch currentBatch = openSymbolicBatch(symbolicBatch.id);
		for (const auto& scheduled : symbolicBatch.passes) {
			if (scheduled.passIndex < rg.m_framePassSchedulingSummaries.size()) {
				const uint64_t requirementCount = static_cast<uint64_t>(rg.m_framePassSchedulingSummaries[scheduled.passIndex].requirements.size());
				rg.m_compileCacheStats.loweredRequirementCount += requirementCount;
			}
			CommitPassToBatch(
				rg,
				passes[scheduled.passIndex],
				nodes[scheduled.nodeIndex],
				symbolicBatch.id,
				currentBatch,
				scratchTransitioned,
				scratchFallback,
				scratchTransitions);
		}
		ensureBatchSlot(symbolicBatch.id);
		rg.batches[symbolicBatch.id] = std::move(currentBatch);
	}
}

bool RenderGraph::TryMaterializeSegmentsWithReplayAndLowering(
	RenderGraph& rg,
	std::vector<AnyPassAndResources>& passes,
	std::vector<Node>& nodes,
	const std::vector<CompiledSegmentDesc>& segmentDescs,
	bool recordReuseEvents)
{
	ZoneScopedN("RenderGraph::TryMaterializeSegmentsWithReplayAndLowering");
	if (segmentDescs.empty()) {
		return false;
	}

	rg.batches.clear();
	rg.batches.emplace_back(rg.m_queueRegistry.SlotCount(), rg.m_frameSchedulingResourceCount);
	m_actualEntryBoundaryStatesBySegmentId.clear();
	m_actualExitBoundaryStatesBySegmentId.clear();

	auto describeBoundaryState = [](const ResourceState& state) -> std::string {
		std::ostringstream oss;
		oss << "access=" << static_cast<uint64_t>(state.access)
			<< ",layout=" << static_cast<uint64_t>(state.layout)
			<< ",sync=" << static_cast<uint64_t>(state.sync);
		return oss.str();
	};

	auto describeResourceName = [&](uint64_t resourceID) -> std::string {
		auto it = resourcesByID.find(resourceID);
		if (it != resourcesByID.end() && it->second && !it->second->GetName().empty()) {
			return it->second->GetName();
		}
		auto transientIt = m_transientFrameResourcesByID.find(resourceID);
		if (transientIt != m_transientFrameResourcesByID.end() && transientIt->second && !transientIt->second->GetName().empty()) {
			return transientIt->second->GetName();
		}
		return std::string("<unknown>");
	};

	auto summarizeCompatibleCachedSegments = [&](const CompiledSegmentDesc& desc) -> std::string {
		size_t compatibleCount = 0;
		std::vector<uint64_t> sampleEntryHashes;
		sampleEntryHashes.reserve(4);
		for (const auto& [ignoredKey, cached] : m_cachedBarrierSegments) {
			(void)ignoredKey;
			if (cached.segmentStructureHash != desc.segmentStructureHash
				|| cached.passContentHash != desc.passContentHash
				|| cached.aliasSignatureHash != desc.aliasSignatureHash
				|| cached.queueAssignmentHash != desc.queueAssignmentHash) {
				continue;
			}

			++compatibleCount;
			if (sampleEntryHashes.size() < 4
				&& std::find(sampleEntryHashes.begin(), sampleEntryHashes.end(), cached.entryStateHash) == sampleEntryHashes.end()) {
				sampleEntryHashes.push_back(cached.entryStateHash);
			}
		}

		std::ostringstream oss;
		oss << "compatible_cached_segments=" << compatibleCount;
		if (!sampleEntryHashes.empty()) {
			oss << " sample_cached_entry_hashes=[";
			for (size_t i = 0; i < sampleEntryHashes.size(); ++i) {
				if (i != 0) {
					oss << ", ";
				}
				oss << sampleEntryHashes[i];
			}
			oss << "]";
		}
		return oss.str();
	};

	auto summarizeBoundaryStateDifferences = [&](std::string_view label,
		std::span<const CompiledSegmentDesc::BoundaryStateEntry> expectedBoundaryStates,
		std::span<const CompiledSegmentDesc::BoundaryStateEntry> actualBoundaryStates) -> std::string {
		auto makeBoundaryKey = [&](const CompiledSegmentDesc::BoundaryStateEntry& entry) {
			uint64_t key = 0;
			key = rg::HashCombine(key, entry.resourceID);
			key = rg::HashCombine(key, HashRangeSpec64(entry.range));
			key = rg::HashCombine(key, entry.aliasSignature);
			return key;
		};

		std::unordered_map<uint64_t, const CompiledSegmentDesc::BoundaryStateEntry*> expectedByKey;
		expectedByKey.reserve(expectedBoundaryStates.size());
		for (const auto& entry : expectedBoundaryStates) {
			expectedByKey[makeBoundaryKey(entry)] = &entry;
		}

		std::unordered_set<uint64_t> seenKeys;
		seenKeys.reserve(actualBoundaryStates.size());
		std::vector<std::string> samples;
		samples.reserve(4);
		size_t diffCount = 0;

		auto appendSample = [&](std::string sample) {
			++diffCount;
			if (samples.size() < 4) {
				samples.push_back(std::move(sample));
			}
		};

		for (const auto& entry : actualBoundaryStates) {
			const uint64_t key = makeBoundaryKey(entry);
			seenKeys.insert(key);
			auto itExpected = expectedByKey.find(key);
			if (itExpected == expectedByKey.end()) {
				std::ostringstream oss;
				oss << "extra_actual resource=" << entry.resourceID << "('" << describeResourceName(entry.resourceID)
					<< "') alias=" << entry.aliasSignature
					<< " range_hash=" << HashRangeSpec64(entry.range)
					<< " state={" << describeBoundaryState(entry.state) << "}";
				appendSample(oss.str());
				continue;
			}

			const auto& expected = *itExpected->second;
			if (expected.state.access != entry.state.access
				|| expected.state.layout != entry.state.layout
				|| expected.state.sync != entry.state.sync) {
				std::ostringstream oss;
				oss << "state_mismatch resource=" << entry.resourceID << "('" << describeResourceName(entry.resourceID)
					<< "') alias=" << entry.aliasSignature
					<< " range_hash=" << HashRangeSpec64(entry.range)
					<< " expected={" << describeBoundaryState(expected.state) << "}"
					<< " actual={" << describeBoundaryState(entry.state) << "}";
				appendSample(oss.str());
			}
		}

		for (const auto& [key, expected] : expectedByKey) {
			if (seenKeys.contains(key)) {
				continue;
			}

			std::ostringstream oss;
			oss << "missing_actual resource=" << expected->resourceID << "('" << describeResourceName(expected->resourceID)
				<< "') alias=" << expected->aliasSignature
				<< " range_hash=" << HashRangeSpec64(expected->range)
				<< " expected={" << describeBoundaryState(expected->state) << "}";
			appendSample(oss.str());
		}

		if (diffCount == 0) {
			return std::string(label) + "_boundary_differences=<none>";
		}

		std::ostringstream oss;
		oss << label << "_boundary_differences=" << diffCount << " samples=[";
		for (size_t i = 0; i < samples.size(); ++i) {
			if (i != 0) {
				oss << "; ";
			}
			oss << samples[i];
		}
		oss << "]";
		return oss.str();
	};

	bool replayedAnySegment = false;
	bool loweredAnySegment = false;
	std::vector<UINT64> materializedSignalValuesByToken;
	std::vector<uint8_t> materializedEnabledSignalTokens;
	const size_t signalRegistryQueueCount = rg.m_queueRegistry.SlotCount();
	auto signalTokenIndex = [&](unsigned int batchIndex, size_t queueSlot, BatchSignalPhase phase) -> size_t {
		return ((static_cast<size_t>(batchIndex) * signalRegistryQueueCount) + queueSlot)
			* PassBatch::kSignalPhaseCount
			+ static_cast<size_t>(phase);
	};
	auto ensureSignalRegistryBatch = [&](unsigned int batchIndex) {
		const size_t requiredSize = signalTokenIndex(batchIndex, signalRegistryQueueCount - 1, BatchSignalPhase::AfterCompletion) + 1;
		if (materializedSignalValuesByToken.size() < requiredSize) {
			materializedSignalValuesByToken.resize(requiredSize, 0);
			materializedEnabledSignalTokens.resize(requiredSize, 0);
		}
	};
	auto registerMaterializedSignalsForRange = [&](unsigned int firstBatch, unsigned int lastBatch) {
		if (rg.batches.empty()) {
			return;
		}
		const unsigned int cappedLastBatch = std::min<unsigned int>(lastBatch, static_cast<unsigned int>(rg.batches.size() - 1));
		for (unsigned int batchIndex = firstBatch; batchIndex <= cappedLastBatch; ++batchIndex) {
			auto& batch = rg.batches[batchIndex];
			for (size_t queueSlot = 0; queueSlot < batch.QueueCount(); ++queueSlot) {
				for (size_t phaseIndex = 0; phaseIndex < PassBatch::kSignalPhaseCount; ++phaseIndex) {
					const auto phase = static_cast<BatchSignalPhase>(phaseIndex);
					if (!batch.HasQueueSignal(phase, queueSlot)) {
						continue;
					}
					const UINT64 symbolicValue = batch.GetQueueSignalFenceValue(phase, queueSlot);
					if (symbolicValue == 0) {
						continue;
					}
					ensureSignalRegistryBatch(batchIndex);
					const size_t tokenIndex = signalTokenIndex(batchIndex, queueSlot, phase);
					materializedSignalValuesByToken[tokenIndex] = symbolicValue;
					materializedEnabledSignalTokens[tokenIndex] = 1;
				}
			}
		}
	};
	auto markAndRegisterSegmentBoundarySignals = [&](const CompiledSegmentDesc& desc) {
		const size_t queueCount = rg.m_queueRegistry.SlotCount();
		for (size_t queueSlot = 0; queueSlot < queueCount; ++queueSlot) {
			std::optional<unsigned int> lastActiveBatch;
			for (unsigned int batchIndex = desc.firstBatch; batchIndex <= desc.lastBatch && batchIndex < rg.batches.size(); ++batchIndex) {
				PassBatch& batch = rg.batches[batchIndex];
				bool queueHasWork = batch.HasPasses(queueSlot);
				for (size_t phaseIndex = 0; phaseIndex < PassBatch::kTransitionPhaseCount && !queueHasWork; ++phaseIndex) {
					queueHasWork = batch.HasTransitions(queueSlot, static_cast<BatchTransitionPhase>(phaseIndex));
				}
				if (queueHasWork) {
					lastActiveBatch = batchIndex;
				}
			}
			if (!lastActiveBatch.has_value()) {
				continue;
			}
			rg.batches[*lastActiveBatch].MarkQueueSignal(BatchSignalPhase::AfterCompletion, queueSlot);
		}
		registerMaterializedSignalsForRange(desc.firstBatch, desc.lastBatch);
	};
	auto refreshReplaySignalEnablesFromBatches = [&](CompiledSegment& segment) {
		for (auto& signal : segment.replay.signals) {
			const unsigned int globalBatch = segment.firstBatch + static_cast<unsigned int>(signal.signal.localBatch);
			if (globalBatch >= rg.batches.size()) {
				signal.enabled = false;
				continue;
			}
			signal.enabled = rg.batches[globalBatch].HasQueueSignal(
				signal.signal.phase,
				static_cast<size_t>(signal.signal.queueSlot));
		}
	};
	const bool validateReplayForDump = m_getRenderGraphCompileDumpEnabled && m_getRenderGraphCompileDumpEnabled();
	auto extractLoweredCompiledSegment = [&](const CompiledSegmentDesc& desc,
		std::vector<CompiledSegmentDesc::BoundaryStateEntry> entryBoundaryStates,
		std::vector<CompiledSegmentDesc::BoundaryStateEntry> exitBoundaryStates,
		uint64_t entryStateHash) {
		CompiledSegment segment{};
		segment.id = desc.id;
		segment.name = desc.name;
		segment.firstPassStreamIndex = desc.firstPassStreamIndex;
		segment.passCount = desc.passCount;
		segment.firstBatch = desc.firstBatch;
		segment.lastBatch = desc.lastBatch;
		segment.passes = desc.passes;
		segment.touchedResourceIDs = desc.touchedResourceIDs;
		segment.entryBoundaryStates = std::move(entryBoundaryStates);
		segment.exitBoundaryStates = std::move(exitBoundaryStates);
		segment.segmentStructureHash = desc.segmentStructureHash;
		segment.passContentHash = desc.passContentHash;
		segment.aliasSignatureHash = desc.aliasSignatureHash;
		segment.queueAssignmentHash = desc.queueAssignmentHash;
		segment.entryStateHash = entryStateHash;
		segment.exitStateHash = HashBoundaryStateEntries(segment.exitBoundaryStates);
		segment.loweredRequirementCount = desc.loweredRequirementCount;
		segment.schedule = desc.schedule;

		for (unsigned int batchIndex = segment.firstBatch; batchIndex <= segment.lastBatch && batchIndex < rg.batches.size(); ++batchIndex) {
			const auto& batch = rg.batches[batchIndex];
			for (size_t qi = 0; qi < batch.QueueCount(); ++qi) {
				for (size_t phase = 0; phase < PassBatch::kSignalPhaseCount; ++phase) {
					segment.barriers.signals.push_back(SymbolicFenceToken{
						.batch = batchIndex,
						.queueSlot = qi,
						.phase = static_cast<BatchSignalPhase>(phase),
						.symbolicValue = batch.GetQueueSignalFenceValue(static_cast<BatchSignalPhase>(phase), qi),
					});
				}
				for (size_t transitionPhase = 0; transitionPhase < PassBatch::kTransitionPhaseCount; ++transitionPhase) {
					const auto phase = static_cast<BatchTransitionPhase>(transitionPhase);
					const auto& transitions = batch.Transitions(qi, phase);
					const uint64_t transitionCount = static_cast<uint64_t>(transitions.size());
					segment.barriers.transitionCount += transitionCount;
					segment.barriers.transitionHash = rg::HashCombine(segment.barriers.transitionHash, transitionCount);
					for (const auto& transition : transitions) {
						segment.barriers.transitions.push_back(SymbolicTransitionOp{
							.batch = batchIndex,
							.queueSlot = qi,
							.phase = phase,
							.resourceID = transition.pResource ? transition.pResource->GetGlobalResourceID() : 0ull,
							.range = transition.range,
							.prevAccessType = transition.prevAccessType,
							.newAccessType = transition.newAccessType,
							.prevLayout = transition.prevLayout,
							.newLayout = transition.newLayout,
							.prevSyncState = transition.prevSyncState,
							.newSyncState = transition.newSyncState,
							.discard = transition.discard,
						});
					}
				}
				for (size_t waitPhase = 0; waitPhase < PassBatch::kWaitPhaseCount; ++waitPhase) {
					for (size_t src = 0; src < batch.QueueCount(); ++src) {
						if (!batch.HasQueueWait(static_cast<BatchWaitPhase>(waitPhase), qi, src)) {
							continue;
						}
						const auto phase = static_cast<BatchWaitPhase>(waitPhase);
						++segment.barriers.waitCount;
						const UINT64 waitValue = batch.GetQueueWaitFenceValue(phase, qi, src);
						segment.barriers.waits.push_back(SymbolicQueueWaitOp{
							.batch = batchIndex,
							.dstQueueSlot = qi,
							.srcQueueSlot = src,
							.phase = phase,
							.symbolicValue = waitValue,
						});
						segment.barriers.waitHash = rg::HashCombine(segment.barriers.waitHash, waitValue);
					}
				}
			}
		}

		segment.replay = BuildCachedBarrierSegmentReplay(segment);
		segment.cacheKey = ComputeBarrierSegmentCacheKey(desc, segment.entryStateHash);
		if (validateReplayForDump) {
			try {
				std::unordered_map<uint64_t, UINT64> validationSignalValuesByToken;
				std::unordered_set<uint64_t> validationEnabledSignalTokens;
				for (unsigned int batchIndex = 0; batchIndex < rg.batches.size(); ++batchIndex) {
					const auto& batch = rg.batches[batchIndex];
					for (size_t queueSlot = 0; queueSlot < batch.QueueCount(); ++queueSlot) {
						for (size_t phaseIndex = 0; phaseIndex < PassBatch::kSignalPhaseCount; ++phaseIndex) {
							const auto phase = static_cast<BatchSignalPhase>(phaseIndex);
							if (!batch.HasQueueSignal(phase, queueSlot)) {
								continue;
							}
							const UINT64 symbolicValue = batch.GetQueueSignalFenceValue(phase, queueSlot);
							if (symbolicValue == 0) {
								continue;
							}
							uint64_t tokenKey = 0;
							tokenKey = rg::HashCombine(tokenKey, static_cast<uint64_t>(batchIndex));
							tokenKey = rg::HashCombine(tokenKey, static_cast<uint64_t>(queueSlot));
							tokenKey = rg::HashCombine(tokenKey, static_cast<uint64_t>(phase));
							validationSignalValuesByToken[tokenKey] = symbolicValue;
							validationEnabledSignalTokens.insert(tokenKey);
						}
					}
				}
				ValidateReplayedSegmentBatches(
					segment,
					BuildReplayedSegmentBatches(segment, &validationSignalValuesByToken, &validationEnabledSignalTokens));
			}
			catch (const std::exception& ex) {
				RecordCompileReuseEvent("ReplayValidation", segment.name, false, segment.cacheKey, ex.what());
			}
		}

		auto cacheIt = m_cachedBarrierSegments.find(segment.cacheKey);
		if (cacheIt != m_cachedBarrierSegments.end()) {
			segment.barrierCacheHit = true;
			++m_compileCacheStats.barrierSegmentCacheHits;
			const BarrierIR loweredBarriers = segment.barriers;
			const BarrierIR& cachedBarriers = cacheIt->second.barriers;
			const bool barrierSummaryMatches =
				loweredBarriers.transitionCount == cachedBarriers.transitionCount
				&& loweredBarriers.waitCount == cachedBarriers.waitCount
				&& loweredBarriers.loweredRequirementCount == cachedBarriers.loweredRequirementCount
				&& loweredBarriers.transitionHash == cachedBarriers.transitionHash
				&& loweredBarriers.waitHash == cachedBarriers.waitHash
				&& loweredBarriers.signals.size() == cachedBarriers.signals.size()
				&& loweredBarriers.transitions.size() == cachedBarriers.transitions.size()
				&& loweredBarriers.waits.size() == cachedBarriers.waits.size();
			if (barrierSummaryMatches) {
				segment.barriersReused = true;
				segment.barriers = cachedBarriers;
				segment.barrierReuseReason = "symbolic barrier metadata reused from matching segment; concrete PassBatch materialization still reran";
				++m_compileCacheStats.barrierSegmentReused;
				m_compileCacheStats.barrierSegmentReusableLoweredRequirements += segment.loweredRequirementCount;
			}
			else {
				segment.barriersReused = false;
				segment.barrierReuseReason = "barrier cache key matched but lowered barrier summary changed; cache key is missing an input";
			}
			cacheIt->second = segment;
			++m_compileCacheStats.barrierSegmentRecompiled;
		}
		else {
			segment.barrierCacheHit = false;
			++m_compileCacheStats.barrierSegmentCacheMisses;
			segment.barriersReused = false;
			segment.barrierReuseReason = "no cached barrier segment for structure/pass/alias/queue/entry-state signature";
			++m_compileCacheStats.barrierSegmentRecompiled;
			m_cachedBarrierSegments.emplace(segment.cacheKey, segment);
		}

		return segment;
	};
	m_compiledSegments.clear();
	m_compiledSegments.reserve(segmentDescs.size());

	for (size_t descIndex = 0; descIndex < segmentDescs.size(); ++descIndex) {
		const auto& desc = segmentDescs[descIndex];
		m_activeMaterializingSegmentName = desc.name;
		m_activeMaterializingSegmentBatchRange = std::pair<unsigned int, unsigned int>{ desc.firstBatch, desc.lastBatch };
		std::vector<CompiledSegmentDesc::BoundaryStateEntry> actualEntryBoundaryStates = CaptureActualBoundaryStates(desc.entryBoundaryStates);
		const uint64_t actualEntryStateHash = HashBoundaryStateEntries(actualEntryBoundaryStates);
		m_actualEntryBoundaryStatesBySegmentId[desc.id] = actualEntryBoundaryStates;

		const uint64_t descriptorCacheKey = ComputeBarrierSegmentCacheKey(desc);
		const bool descriptorCacheHit = m_cachedBarrierSegments.find(descriptorCacheKey) != m_cachedBarrierSegments.end();
		const uint64_t cacheKey = ComputeBarrierSegmentCacheKey(desc, actualEntryStateHash);
		auto cacheIt = m_cachedBarrierSegments.find(cacheKey);
		if (!recordReuseEvents) {
			if (cacheIt != m_cachedBarrierSegments.end()) {
				++m_compileCacheStats.plannedReplaySegmentCount;
				m_compileCacheStats.plannedReplayLoweredRequirements += desc.loweredRequirementCount;
			}
			else {
				++m_compileCacheStats.plannedLowerSegmentCount;
				m_compileCacheStats.plannedLoweredRequirements += desc.loweredRequirementCount;
			}
		}
		if (cacheIt != m_cachedBarrierSegments.end()) {
			m_activeMaterializingSegmentMode = "replay";
			CompiledSegment segment = cacheIt->second;
			segment.id = desc.id;
			segment.name = desc.name;
			segment.firstPassStreamIndex = desc.firstPassStreamIndex;
			segment.passCount = desc.passCount;
			segment.firstBatch = desc.firstBatch;
			segment.lastBatch = desc.lastBatch;
			segment.passes = desc.passes;
			segment.touchedResourceIDs = desc.touchedResourceIDs;
			segment.entryBoundaryStates = actualEntryBoundaryStates;
			segment.segmentStructureHash = desc.segmentStructureHash;
			segment.passContentHash = desc.passContentHash;
			segment.aliasSignatureHash = desc.aliasSignatureHash;
			segment.queueAssignmentHash = desc.queueAssignmentHash;
			segment.entryStateHash = actualEntryStateHash;
			segment.loweredRequirementCount = desc.loweredRequirementCount;
			segment.schedule = desc.schedule;
			segment.cacheKey = cacheKey;
			segment.barrierCacheHit = true;
			segment.barriersReused = true;
			segment.barrierReuseReason = "symbolic barrier metadata reused from matching segment; concrete PassBatch materialization replayed cached segment";

			try {
				ReplaySegmentIntoFrameBatches(
					segment,
					materializedSignalValuesByToken,
					materializedEnabledSignalTokens);
				markAndRegisterSegmentBoundarySignals(desc);
				refreshReplaySignalEnablesFromBatches(segment);
				ApplyReplayedSegmentCompilerState(segment);
				m_actualExitBoundaryStatesBySegmentId[desc.id] = segment.exitBoundaryStates;
				if (m_getRenderGraphCompileDumpEnabled && m_getRenderGraphCompileDumpEnabled()) {
					std::vector<CompiledSegmentDesc::BoundaryStateEntry> actualExitBoundaryStates = CaptureActualBoundaryStates(segment.exitBoundaryStates);
					const uint64_t actualExitStateHash = HashBoundaryStateEntries(actualExitBoundaryStates);
					if (actualExitStateHash != segment.exitStateHash) {
						std::ostringstream replayMismatchReason;
						replayMismatchReason
							<< "replayed segment exit-state mismatch: segment='" << desc.name
							<< "' batches=[" << desc.firstBatch << "," << desc.lastBatch << "]"
							<< " cache_key=" << cacheKey
							<< " expected_exit_hash=" << segment.exitStateHash
							<< " actual_exit_hash=" << actualExitStateHash
							<< ' ' << summarizeBoundaryStateDifferences("exit", segment.exitBoundaryStates, actualExitBoundaryStates);
						if (recordReuseEvents) {
							RecordCompileReuseEvent("BarrierSegmentReplayExitState", desc.name, false, cacheKey, replayMismatchReason.str());
						}
						spdlog::warn("RenderGraph::TryMaterializeSegmentsWithReplayAndLowering: {}", replayMismatchReason.str());
					}
				}
				replayedAnySegment = true;
				++m_compileCacheStats.materializedReplaySegmentCount;
				if (recordReuseEvents) {
					RecordCompileReuseEvent("BarrierSegment", segment.name, true, segment.cacheKey, segment.barrierReuseReason);
				}
				m_compiledSegments.push_back(std::move(segment));
				continue;
			}
			catch (const std::exception& ex) {
				if (recordReuseEvents) {
					RecordCompileReuseEvent("BarrierSegment", desc.name, false, cacheKey, ex.what());
				}
				spdlog::warn("RenderGraph::TryMaterializeSegmentsWithReplayAndLowering: rejected replay for segment='{}': {}", desc.name, ex.what());
			}
		}

		if (descriptorCacheHit) {
			std::ostringstream rejectionReason;
			rejectionReason
				<< "planned replay segment rejected at materialization: segment='" << desc.name
				<< "' batches=[" << desc.firstBatch << "," << desc.lastBatch << "]"
				<< " descriptor_cache_key=" << descriptorCacheKey
				<< " actual_cache_key=" << cacheKey
				<< " descriptor_entry_hash=" << desc.entryStateHash
				<< " actual_entry_hash=" << actualEntryStateHash
				<< ' ' << summarizeCompatibleCachedSegments(desc)
				<< ' ' << summarizeBoundaryStateDifferences("entry", desc.entryBoundaryStates, actualEntryBoundaryStates);
			if (recordReuseEvents) {
				RecordCompileReuseEvent("BarrierSegment", desc.name, false, cacheKey, rejectionReason.str());
			}
			spdlog::warn("RenderGraph::TryMaterializeSegmentsWithReplayAndLowering: {}", rejectionReason.str());
		}

		m_activeMaterializingSegmentMode = "lower";
		m_activeMaterializingSegmentBatchRange = std::pair<unsigned int, unsigned int>{ desc.firstBatch, desc.lastBatch };
		m_activeLoweringBatchRange = std::pair<unsigned int, unsigned int>{ desc.firstBatch, desc.lastBatch };
		AppendLoweredScheduleBatches(rg, passes, nodes, desc.schedule, desc.firstBatch);
		m_activeLoweringBatchRange.reset();
		markAndRegisterSegmentBoundarySignals(desc);
		std::vector<CompiledSegmentDesc::BoundaryStateEntry> actualExitBoundaryStates = CaptureActualBoundaryStates(desc.exitBoundaryStates);
		m_actualExitBoundaryStatesBySegmentId[desc.id] = actualExitBoundaryStates;
		loweredAnySegment = true;
		++m_compileCacheStats.materializedLowerSegmentCount;
		if (recordReuseEvents) {
			RecordCompileReuseEvent(
				"BarrierSegment",
				desc.name,
				false,
				cacheKey,
				"lowered this segment with segment-boundary-fenced transition placement; later segments remain eligible for replay");
		}
		m_compiledSegments.push_back(extractLoweredCompiledSegment(
			desc,
			std::move(actualEntryBoundaryStates),
			std::move(actualExitBoundaryStates),
			actualEntryStateHash));
	}

	if (!replayedAnySegment && !loweredAnySegment) {
		return false;
	}

	const char* materializationReason = loweredAnySegment
		? "materialized frame with mixed segment replay and segment-boundary-fenced lowering"
		: "replayed all segments from cache using actual tracker entry-state identity";
	if (recordReuseEvents) {
		RecordCompileReuseEvent(
			"BarrierMaterialization",
			"frame",
			true,
			m_compiledScheduleIR.structureHash,
			materializationReason);
	}

	m_compileCacheStats.segmentCount = m_compiledSegments.size();
	m_compiledBarrierIR = BuildBarrierIRFromCompiledSegments(m_compiledSegments);
	FinalizeBatchesFromIR(rg, passes, m_compiledScheduleIR, m_compiledBarrierIR);
	if (m_getRenderGraphCompileDumpEnabled && m_getRenderGraphCompileDumpEnabled()) {
		try {
			ValidateCompiledFrameReplay();
		}
		catch (const std::exception& ex) {
			RecordCompileReuseEvent(
				"ReplayValidation",
				"mixed segment materialization",
				false,
				m_compiledScheduleIR.structureHash,
				ex.what());
		}
	}
	m_activeMaterializingSegmentName.clear();
	m_activeMaterializingSegmentMode.clear();
	m_activeMaterializingSegmentBatchRange.reset();
	m_actualEntryBoundaryStatesBySegmentId.clear();
	m_actualExitBoundaryStatesBySegmentId.clear();
	return true;
}

void RenderGraph::ValidateReplayedSegmentBatches(
	const CompiledSegment& segment,
	const std::vector<PassBatch>& replayedBatches) const
{
	ZoneScopedN("RenderGraph::ValidateReplayedSegmentBatches");
	if (replayedBatches.size() != segment.replay.localBatchCount) {
		throw std::runtime_error("Segment replay batch count mismatch");
	}

	for (unsigned int batchIndex = segment.firstBatch; batchIndex <= segment.lastBatch && batchIndex < batches.size(); ++batchIndex) {
		const size_t localBatch = batchIndex - segment.firstBatch;
		const auto& expected = batches[batchIndex];
		const auto& replayed = replayedBatches[localBatch];

		for (size_t queueSlot = 0; queueSlot < m_queueRegistry.SlotCount(); ++queueSlot) {
			if (expected.Passes(queueSlot).size() != replayed.Passes(queueSlot).size()) {
				throw std::runtime_error("Segment replay pass placement mismatch");
			}

			for (size_t transitionPhase = 0; transitionPhase < PassBatch::kTransitionPhaseCount; ++transitionPhase) {
				const auto phase = static_cast<BatchTransitionPhase>(transitionPhase);
				if (expected.Transitions(queueSlot, phase).size() != replayed.Transitions(queueSlot, phase).size()) {
					throw std::runtime_error("Segment replay transition count mismatch");
				}
			}

			for (size_t signalPhase = 0; signalPhase < PassBatch::kSignalPhaseCount; ++signalPhase) {
				const auto phase = static_cast<BatchSignalPhase>(signalPhase);
				if (expected.GetQueueSignalFenceValue(phase, queueSlot) != replayed.GetQueueSignalFenceValue(phase, queueSlot)) {
					throw std::runtime_error("Segment replay signal fence mismatch");
				}
				if (expected.HasQueueSignal(phase, queueSlot) != replayed.HasQueueSignal(phase, queueSlot)) {
					throw std::runtime_error("Segment replay signal enable mismatch");
				}
			}

			for (size_t waitPhase = 0; waitPhase < PassBatch::kWaitPhaseCount; ++waitPhase) {
				const auto phase = static_cast<BatchWaitPhase>(waitPhase);
				for (size_t srcQueue = 0; srcQueue < m_queueRegistry.SlotCount(); ++srcQueue) {
					if (expected.HasQueueWait(phase, queueSlot, srcQueue) != replayed.HasQueueWait(phase, queueSlot, srcQueue)) {
						std::ostringstream message;
						message << "Segment replay wait enable mismatch: segment='" << segment.name
							<< "' global_batch=" << batchIndex
							<< " local_batch=" << localBatch
							<< " phase=" << waitPhase
							<< " dst_queue=" << queueSlot
							<< " src_queue=" << srcQueue
							<< " expected=" << expected.HasQueueWait(phase, queueSlot, srcQueue)
							<< " replayed=" << replayed.HasQueueWait(phase, queueSlot, srcQueue);
						throw std::runtime_error(message.str());
					}
					if (expected.GetQueueWaitFenceValue(phase, queueSlot, srcQueue) != replayed.GetQueueWaitFenceValue(phase, queueSlot, srcQueue)) {
						std::ostringstream message;
						message << "Segment replay wait fence mismatch: segment='" << segment.name
							<< "' global_batch=" << batchIndex
							<< " local_batch=" << localBatch
							<< " phase=" << waitPhase
							<< " dst_queue=" << queueSlot
							<< " src_queue=" << srcQueue
							<< " expected=" << expected.GetQueueWaitFenceValue(phase, queueSlot, srcQueue)
							<< " replayed=" << replayed.GetQueueWaitFenceValue(phase, queueSlot, srcQueue);
						throw std::runtime_error(message.str());
					}
				}
			}
		}

		if (expected.allResources != replayed.allResources) {
			throw std::runtime_error("Segment replay allResources mismatch");
		}
		if (expected.internallyTransitionedResources != replayed.internallyTransitionedResources) {
			throw std::runtime_error("Segment replay internally transitioned resources mismatch");
		}
	}
}

void RenderGraph::ValidateCompiledFrameReplay()
{
	ZoneScopedN("RenderGraph::ValidateCompiledFrameReplay");
	std::vector<PassBatch> replayedFrameBatches = BuildReplayedFrameBatches();
	if (replayedFrameBatches.size() != batches.size()) {
		throw std::runtime_error("Frame replay batch count mismatch");
	}

	std::unordered_map<UINT64, UINT64> concreteFenceBySymbolic;
	for (const auto& segment : m_compiledSegments) {
		for (const auto& signal : segment.replay.signals) {
			const unsigned int globalBatch = segment.firstBatch + signal.signal.localBatch;
			if (globalBatch >= batches.size()) {
				continue;
			}
			const auto& expectedBatch = batches[globalBatch];
			const size_t queueSlot = static_cast<size_t>(signal.signal.queueSlot);
			if (!expectedBatch.HasQueueSignal(signal.signal.phase, queueSlot)) {
				continue;
			}
			const UINT64 concreteValue = expectedBatch.GetQueueSignalFenceValue(signal.signal.phase, queueSlot);
			if (signal.symbolicValue == 0 || concreteValue == 0) {
				continue;
			}

			const UINT64 mapKey = (static_cast<UINT64>(queueSlot) << 56) | signal.symbolicValue;
			concreteFenceBySymbolic.emplace(mapKey, concreteValue);
		}
	}

	for (auto& batch : replayedFrameBatches) {
		for (size_t queueSlot = 0; queueSlot < m_queueRegistry.SlotCount(); ++queueSlot) {
			for (size_t signalPhase = 0; signalPhase < PassBatch::kSignalPhaseCount; ++signalPhase) {
				const auto phase = static_cast<BatchSignalPhase>(signalPhase);
				if (!batch.HasQueueSignal(phase, queueSlot)) {
					continue;
				}
				const UINT64 symbolicValue = batch.GetQueueSignalFenceValue(phase, queueSlot);
				if (symbolicValue != 0) {
					const UINT64 mapKey = (static_cast<UINT64>(queueSlot) << 56) | symbolicValue;
					auto it = concreteFenceBySymbolic.find(mapKey);
					if (it != concreteFenceBySymbolic.end()) {
						batch.SetQueueSignalFenceValue(phase, queueSlot, it->second);
					}
				}
			}

			for (size_t waitPhase = 0; waitPhase < PassBatch::kWaitPhaseCount; ++waitPhase) {
				const auto phase = static_cast<BatchWaitPhase>(waitPhase);
				for (size_t srcQueue = 0; srcQueue < m_queueRegistry.SlotCount(); ++srcQueue) {
					if (!batch.HasQueueWait(phase, queueSlot, srcQueue)) {
						continue;
					}

					const UINT64 symbolicValue = batch.GetQueueWaitFenceValue(phase, queueSlot, srcQueue);
					const UINT64 mapKey = (static_cast<UINT64>(srcQueue) << 56) | symbolicValue;
					auto it = concreteFenceBySymbolic.find(mapKey);
					if (it != concreteFenceBySymbolic.end()) {
						batch.queueWaitFenceValue[waitPhase][queueSlot][srcQueue] = it->second;
					}
					else {
						throw std::runtime_error("Frame replay wait references unresolved enabled signal");
					}
				}
			}
		}
	}

	auto validateBatch = [&](const PassBatch& expected, const PassBatch& replayed, size_t batchIndex, const char* context) {
		for (size_t queueSlot = 0; queueSlot < m_queueRegistry.SlotCount(); ++queueSlot) {
			if (expected.Passes(queueSlot).size() != replayed.Passes(queueSlot).size()) {
				throw std::runtime_error(std::string(context) + ": pass placement mismatch");
			}

			for (size_t transitionPhase = 0; transitionPhase < PassBatch::kTransitionPhaseCount; ++transitionPhase) {
				const auto phase = static_cast<BatchTransitionPhase>(transitionPhase);
				if (expected.Transitions(queueSlot, phase).size() != replayed.Transitions(queueSlot, phase).size()) {
					throw std::runtime_error(std::string(context) + ": transition count mismatch");
				}
			}

			for (size_t signalPhase = 0; signalPhase < PassBatch::kSignalPhaseCount; ++signalPhase) {
				const auto phase = static_cast<BatchSignalPhase>(signalPhase);
				if (expected.GetQueueSignalFenceValue(phase, queueSlot) != replayed.GetQueueSignalFenceValue(phase, queueSlot)) {
					std::ostringstream message;
					message << context
						<< ": signal fence mismatch"
						<< " batch=" << batchIndex
						<< " phase=" << signalPhase
						<< " queue=" << queueSlot
						<< " expected=" << expected.GetQueueSignalFenceValue(phase, queueSlot)
						<< " replayed=" << replayed.GetQueueSignalFenceValue(phase, queueSlot);
					throw std::runtime_error(message.str());
				}
				if (expected.HasQueueSignal(phase, queueSlot) != replayed.HasQueueSignal(phase, queueSlot)) {
					throw std::runtime_error(std::string(context) + ": signal enable mismatch");
				}
			}

			for (size_t waitPhase = 0; waitPhase < PassBatch::kWaitPhaseCount; ++waitPhase) {
				const auto phase = static_cast<BatchWaitPhase>(waitPhase);
				for (size_t srcQueue = 0; srcQueue < m_queueRegistry.SlotCount(); ++srcQueue) {
					if (expected.HasQueueWait(phase, queueSlot, srcQueue) != replayed.HasQueueWait(phase, queueSlot, srcQueue)) {
						std::ostringstream message;
						message << context
							<< ": wait enable mismatch"
							<< " batch=" << batchIndex
							<< " phase=" << waitPhase
							<< " dst_queue=" << queueSlot
							<< " src_queue=" << srcQueue
							<< " expected=" << expected.HasQueueWait(phase, queueSlot, srcQueue)
							<< " replayed=" << replayed.HasQueueWait(phase, queueSlot, srcQueue);
						throw std::runtime_error(message.str());
					}
					if (expected.GetQueueWaitFenceValue(phase, queueSlot, srcQueue) != replayed.GetQueueWaitFenceValue(phase, queueSlot, srcQueue)) {
						std::ostringstream message;
						message << context
							<< ": wait fence mismatch"
							<< " batch=" << batchIndex
							<< " phase=" << waitPhase
							<< " dst_queue=" << queueSlot
							<< " src_queue=" << srcQueue
							<< " expected=" << expected.GetQueueWaitFenceValue(phase, queueSlot, srcQueue)
							<< " replayed=" << replayed.GetQueueWaitFenceValue(phase, queueSlot, srcQueue);
						throw std::runtime_error(message.str());
					}
				}
			}
		}

		if (expected.allResources != replayed.allResources) {
			throw std::runtime_error(std::string(context) + ": allResources mismatch");
		}
		if (expected.internallyTransitionedResources != replayed.internallyTransitionedResources) {
			throw std::runtime_error(std::string(context) + ": internally transitioned resources mismatch");
		}
	};

	for (size_t batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
		validateBatch(batches[batchIndex], replayedFrameBatches[batchIndex], batchIndex, "Frame replay");
	}

	const size_t entryCount = m_queueRegistry.SlotCount() * m_frameSchedulingResourceCount;
	std::vector<unsigned int> replayedUsageHistory(entryCount, 0);
	std::vector<unsigned int> replayedProducerHistory(entryCount, 0);
	std::vector<unsigned int> replayedTransitionHistory(entryCount, 0);
	std::vector<std::vector<unsigned int>> replayedTransitionPlacementByResource(m_frameSchedulingResourceCount);
	std::vector<FrameResourceEventSummary> replayedResourceEventSummaries;
	if (m_queueRegistry.SlotCount() <= 64) {
		replayedResourceEventSummaries.assign(m_frameSchedulingResourceCount, FrameResourceEventSummary{});
	}

	auto setHistoryValue = [&](std::vector<unsigned int>& history, size_t queueSlot, size_t resourceIndex, unsigned int batchIndex) {
		if (batchIndex == 0 || queueSlot >= m_queueRegistry.SlotCount() || resourceIndex >= m_frameSchedulingResourceCount) {
			return;
		}
		history[queueSlot * m_frameSchedulingResourceCount + resourceIndex] = batchIndex;
	};

	auto recordResourceEvent = [&](size_t queueSlot, size_t resourceIndex, unsigned int batchIndex) {
		if (batchIndex == 0 || queueSlot >= 64 || resourceIndex >= replayedResourceEventSummaries.size()) {
			return;
		}

		auto& summary = replayedResourceEventSummaries[resourceIndex];
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
	};

	auto recordTransitionPlacement = [&](size_t resourceIndex, unsigned int batchIndex) {
		if (batchIndex == 0 || resourceIndex >= replayedTransitionPlacementByResource.size()) {
			return;
		}
		auto& batchHistory = replayedTransitionPlacementByResource[resourceIndex];
		if (batchHistory.empty() || batchHistory.back() < batchIndex) {
			batchHistory.push_back(batchIndex);
			return;
		}
		if (batchHistory.back() == batchIndex) {
			return;
		}
		auto insertIt = std::lower_bound(batchHistory.begin(), batchHistory.end(), batchIndex);
		if (insertIt == batchHistory.end() || *insertIt != batchIndex) {
			batchHistory.insert(insertIt, batchIndex);
		}
	};

	for (const auto& segment : m_compiledSegments) {
		for (const auto& event : segment.replay.historyEvents) {
			auto resourceIndex = TryGetFrameSchedulingResourceIndex(event.resourceID);
			if (!resourceIndex.has_value()) {
				continue;
			}

			const unsigned int globalBatch = segment.firstBatch + event.localBatch;
			switch (event.kind) {
			case CachedHistoryEvent::Kind::Usage:
				setHistoryValue(replayedUsageHistory, event.queueSlot, *resourceIndex, globalBatch);
				recordResourceEvent(event.queueSlot, *resourceIndex, globalBatch);
				break;
			case CachedHistoryEvent::Kind::Producer:
				setHistoryValue(replayedProducerHistory, event.queueSlot, *resourceIndex, globalBatch);
				break;
			case CachedHistoryEvent::Kind::QueueTransition:
				setHistoryValue(replayedTransitionHistory, event.queueSlot, *resourceIndex, globalBatch);
				recordResourceEvent(event.queueSlot, *resourceIndex, globalBatch);
				break;
			case CachedHistoryEvent::Kind::TransitionPlacement:
				recordTransitionPlacement(*resourceIndex, globalBatch);
				break;
			}
		}

		for (const auto& event : segment.replay.resourceEvents) {
			auto resourceIndex = TryGetFrameSchedulingResourceIndex(event.resourceID);
			if (!resourceIndex.has_value()) {
				continue;
			}
			const unsigned int globalBatch = segment.firstBatch + event.localBatch;
			recordResourceEvent(event.queueSlot, *resourceIndex, globalBatch);
		}
	}

	if (replayedUsageHistory != m_frameQueueLastUsageBatch) {
		throw std::runtime_error("Frame replay usage history mismatch");
	}
	if (replayedProducerHistory != m_frameQueueLastProducerBatch) {
		throw std::runtime_error("Frame replay producer history mismatch");
	}
	if (replayedTransitionHistory != m_frameQueueLastTransitionBatch) {
		throw std::runtime_error("Frame replay transition history mismatch");
	}
	if (replayedTransitionPlacementByResource != m_frameTransitionPlacementBatchesByResource) {
		throw std::runtime_error("Frame replay transition placement mismatch");
	}
	if (replayedResourceEventSummaries != m_frameResourceEventSummaries) {
		for (size_t resourceIndex = 0; resourceIndex < replayedResourceEventSummaries.size(); ++resourceIndex) {
			if (replayedResourceEventSummaries[resourceIndex] == m_frameResourceEventSummaries[resourceIndex]) {
				continue;
			}

			uint64_t resourceID = 0;
			std::string resourceName;
			if (resourceIndex < m_frameSchedulingResourceIDsByIndex.size()) {
				resourceID = m_frameSchedulingResourceIDsByIndex[resourceIndex];
				auto resourceIt = resourcesByID.find(resourceID);
				if (resourceIt != resourcesByID.end() && resourceIt->second) {
					resourceName = resourceIt->second->GetName();
				}
			}

			const auto& expected = m_frameResourceEventSummaries[resourceIndex];
			const auto& replayed = replayedResourceEventSummaries[resourceIndex];
			std::ostringstream message;
			message << "Frame replay resource event summary mismatch"
				<< ": resource_index=" << resourceIndex
				<< " resource_id=" << resourceID;
			if (!resourceName.empty()) {
				message << " resource='" << resourceName << "'";
			}
			message << " expected_latest_batch=" << expected.latestBatch
				<< " expected_latest_mask=" << expected.latestQueueMask
				<< " expected_previous_batch=" << expected.previousBatch
				<< " expected_previous_mask=" << expected.previousQueueMask
				<< " replayed_latest_batch=" << replayed.latestBatch
				<< " replayed_latest_mask=" << replayed.latestQueueMask
				<< " replayed_previous_batch=" << replayed.previousBatch
				<< " replayed_previous_mask=" << replayed.previousQueueMask;
			throw std::runtime_error(message.str());
		}

		throw std::runtime_error("Frame replay resource event summary mismatch");
	}

	const size_t queueCount = m_queueRegistry.SlotCount();
	std::vector<std::unordered_map<uint64_t, unsigned int>> replayedCrossFrameProducer(queueCount);
	for (unsigned int batchIndex = 1; batchIndex < static_cast<unsigned int>(replayedFrameBatches.size()); ++batchIndex) {
		auto& batch = replayedFrameBatches[batchIndex];
		for (size_t queueSlot = 0; queueSlot < queueCount; ++queueSlot) {
			for (auto& passVariant : batch.Passes(queueSlot)) {
				std::visit([&](const auto* passEntry) {
					for (const auto& req : passEntry->resources.frameResourceRequirements) {
						if (AccessTypeIsWriteType(req.state.access)) {
							replayedCrossFrameProducer[queueSlot][req.resourceHandleAndRange.resource.GetGlobalResourceID()] = batchIndex;
						}
					}
				}, passVariant);
			}
		}
	}

	if (replayedCrossFrameProducer != m_compiledLastProducerBatchByResourceByQueue) {
		for (size_t queueSlot = 0; queueSlot < queueCount; ++queueSlot) {
			const auto& expectedByResource = m_compiledLastProducerBatchByResourceByQueue[queueSlot];
			const auto& replayedByResource = replayedCrossFrameProducer[queueSlot];
			for (const auto& [resourceID, expectedBatch] : expectedByResource) {
				auto replayedIt = replayedByResource.find(resourceID);
				const unsigned int replayedBatch = replayedIt != replayedByResource.end() ? replayedIt->second : 0u;
				if (replayedBatch == expectedBatch) {
					continue;
				}

				std::string resourceName;
				auto resourceIt = resourcesByID.find(resourceID);
				if (resourceIt != resourcesByID.end() && resourceIt->second) {
					resourceName = resourceIt->second->GetName();
				}

				std::ostringstream message;
				message << "Frame replay cross-frame producer mismatch"
					<< ": queue=" << queueSlot
					<< " resource_id=" << resourceID;
				if (!resourceName.empty()) {
					message << " resource='" << resourceName << "'";
				}
				message << " expected_batch=" << expectedBatch
					<< " replayed_batch=" << replayedBatch;
				throw std::runtime_error(message.str());
			}

			for (const auto& [resourceID, replayedBatch] : replayedByResource) {
				if (expectedByResource.contains(resourceID)) {
					continue;
				}

				std::string resourceName;
				auto resourceIt = resourcesByID.find(resourceID);
				if (resourceIt != resourcesByID.end() && resourceIt->second) {
					resourceName = resourceIt->second->GetName();
				}

				std::ostringstream message;
				message << "Frame replay cross-frame producer mismatch"
					<< ": queue=" << queueSlot
					<< " resource_id=" << resourceID;
				if (!resourceName.empty()) {
					message << " resource='" << resourceName << "'";
				}
				message << " expected_batch=0"
					<< " replayed_batch=" << replayedBatch;
				throw std::runtime_error(message.str());
			}
		}

		throw std::runtime_error("Frame replay cross-frame producer mismatch");
	}
}

void RenderGraph::BuildCompiledSegments(
	const std::vector<CompiledSegmentDesc>& segmentDescs)
{
	ZoneScopedN("RenderGraph::BuildCompiledSegments");
	const bool validateReplayForDump = m_getRenderGraphCompileDumpEnabled && m_getRenderGraphCompileDumpEnabled();
	m_compiledSegments.clear();
	m_compiledSegments.reserve(segmentDescs.size());

	for (const auto& desc : segmentDescs) {
		CompiledSegment segment{};
		segment.id = desc.id;
		segment.name = desc.name;
		segment.firstPassStreamIndex = desc.firstPassStreamIndex;
		segment.passCount = desc.passCount;
		segment.firstBatch = desc.firstBatch;
		segment.lastBatch = desc.lastBatch;
		segment.passes = desc.passes;
		segment.touchedResourceIDs = desc.touchedResourceIDs;
		auto actualEntryIt = m_actualEntryBoundaryStatesBySegmentId.find(desc.id);
		segment.entryBoundaryStates = actualEntryIt != m_actualEntryBoundaryStatesBySegmentId.end()
			? actualEntryIt->second
			: desc.entryBoundaryStates;
		auto actualExitIt = m_actualExitBoundaryStatesBySegmentId.find(desc.id);
		segment.exitBoundaryStates = actualExitIt != m_actualExitBoundaryStatesBySegmentId.end()
			? actualExitIt->second
			: desc.exitBoundaryStates;
		segment.segmentStructureHash = desc.segmentStructureHash;
		segment.passContentHash = desc.passContentHash;
		segment.aliasSignatureHash = desc.aliasSignatureHash;
		segment.queueAssignmentHash = desc.queueAssignmentHash;
		segment.entryStateHash = actualEntryIt != m_actualEntryBoundaryStatesBySegmentId.end()
			? HashBoundaryStateEntries(segment.entryBoundaryStates)
			: desc.entryStateHash;
		segment.exitStateHash = actualExitIt != m_actualExitBoundaryStatesBySegmentId.end()
			? HashBoundaryStateEntries(segment.exitBoundaryStates)
			: desc.exitStateHash;
		segment.loweredRequirementCount = desc.loweredRequirementCount;
		segment.schedule = desc.schedule;
		m_compiledSegments.push_back(std::move(segment));
	}

	for (auto& segment : m_compiledSegments) {
		const auto descIndex = static_cast<size_t>(&segment - m_compiledSegments.data());
		const auto& desc = segmentDescs[descIndex];
		for (unsigned int batchIndex = segment.firstBatch; batchIndex <= segment.lastBatch && batchIndex < batches.size(); ++batchIndex) {
			const auto& batch = batches[batchIndex];
			for (size_t qi = 0; qi < batch.QueueCount(); ++qi) {
				for (size_t phase = 0; phase < PassBatch::kSignalPhaseCount; ++phase) {
					segment.barriers.signals.push_back(SymbolicFenceToken{
						.batch = batchIndex,
						.queueSlot = qi,
						.phase = static_cast<BatchSignalPhase>(phase),
						.symbolicValue = batch.GetQueueSignalFenceValue(static_cast<BatchSignalPhase>(phase), qi),
					});
				}
				for (size_t transitionPhase = 0; transitionPhase < PassBatch::kTransitionPhaseCount; ++transitionPhase) {
					const auto phase = static_cast<BatchTransitionPhase>(transitionPhase);
					const auto& transitions = batch.Transitions(qi, phase);
					const uint64_t transitionCount = static_cast<uint64_t>(transitions.size());
					segment.barriers.transitionCount += transitionCount;
					segment.barriers.transitionHash = rg::HashCombine(
						segment.barriers.transitionHash,
						transitionCount);
					for (const auto& transition : transitions) {
						segment.barriers.transitions.push_back(SymbolicTransitionOp{
							.batch = batchIndex,
							.queueSlot = qi,
							.phase = phase,
							.resourceID = transition.pResource ? transition.pResource->GetGlobalResourceID() : 0ull,
							.range = transition.range,
							.prevAccessType = transition.prevAccessType,
							.newAccessType = transition.newAccessType,
							.prevLayout = transition.prevLayout,
							.newLayout = transition.newLayout,
							.prevSyncState = transition.prevSyncState,
							.newSyncState = transition.newSyncState,
							.discard = transition.discard,
						});
					}
				}
				for (size_t waitPhase = 0; waitPhase < PassBatch::kWaitPhaseCount; ++waitPhase) {
					for (size_t src = 0; src < batch.QueueCount(); ++src) {
						if (batch.HasQueueWait(static_cast<BatchWaitPhase>(waitPhase), qi, src)) {
							const auto phase = static_cast<BatchWaitPhase>(waitPhase);
							++segment.barriers.waitCount;
							segment.barriers.waits.push_back(SymbolicQueueWaitOp{
								.batch = batchIndex,
								.dstQueueSlot = qi,
								.srcQueueSlot = src,
								.phase = phase,
								.symbolicValue = batch.GetQueueWaitFenceValue(phase, qi, src),
							});
							segment.barriers.waitHash = rg::HashCombine(
								segment.barriers.waitHash,
								batch.GetQueueWaitFenceValue(phase, qi, src));
						}
					}
				}
			}
		}

		segment.replay = BuildCachedBarrierSegmentReplay(segment);
		segment.cacheKey = ComputeBarrierSegmentCacheKey(desc, segment.entryStateHash);
		if (validateReplayForDump) {
			try {
				std::unordered_map<uint64_t, UINT64> validationSignalValuesByToken;
				std::unordered_set<uint64_t> validationEnabledSignalTokens;
				for (unsigned int batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
					const auto& batch = batches[batchIndex];
					for (size_t queueSlot = 0; queueSlot < batch.QueueCount(); ++queueSlot) {
						for (size_t phaseIndex = 0; phaseIndex < PassBatch::kSignalPhaseCount; ++phaseIndex) {
							const auto phase = static_cast<BatchSignalPhase>(phaseIndex);
							if (!batch.HasQueueSignal(phase, queueSlot)) {
								continue;
							}
							const UINT64 symbolicValue = batch.GetQueueSignalFenceValue(phase, queueSlot);
							if (symbolicValue == 0) {
								continue;
							}
							uint64_t tokenKey = 0;
							tokenKey = rg::HashCombine(tokenKey, static_cast<uint64_t>(batchIndex));
							tokenKey = rg::HashCombine(tokenKey, static_cast<uint64_t>(queueSlot));
							tokenKey = rg::HashCombine(tokenKey, static_cast<uint64_t>(phase));
							validationSignalValuesByToken[tokenKey] = symbolicValue;
							validationEnabledSignalTokens.insert(tokenKey);
						}
					}
				}
				ValidateReplayedSegmentBatches(
					segment,
					BuildReplayedSegmentBatches(segment, &validationSignalValuesByToken, &validationEnabledSignalTokens));
			}
			catch (const std::exception& ex) {
				RecordCompileReuseEvent(
					"ReplayValidation",
					segment.name,
					false,
					segment.cacheKey,
					ex.what());
			}
		}

		auto cacheIt = m_cachedBarrierSegments.find(segment.cacheKey);
		if (cacheIt != m_cachedBarrierSegments.end()) {
			segment.barrierCacheHit = true;
			++m_compileCacheStats.barrierSegmentCacheHits;
			const BarrierIR loweredBarriers = segment.barriers;
			const BarrierIR& cachedBarriers = cacheIt->second.barriers;
			const bool barrierSummaryMatches =
				loweredBarriers.transitionCount == cachedBarriers.transitionCount
				&& loweredBarriers.waitCount == cachedBarriers.waitCount
				&& loweredBarriers.loweredRequirementCount == cachedBarriers.loweredRequirementCount
				&& loweredBarriers.transitionHash == cachedBarriers.transitionHash
				&& loweredBarriers.waitHash == cachedBarriers.waitHash
				&& loweredBarriers.signals.size() == cachedBarriers.signals.size()
				&& loweredBarriers.transitions.size() == cachedBarriers.transitions.size()
				&& loweredBarriers.waits.size() == cachedBarriers.waits.size();
			if (barrierSummaryMatches) {
				segment.barriersReused = true;
				segment.barriers = cachedBarriers;
				segment.barrierReuseReason = "symbolic barrier metadata reused from matching segment; concrete PassBatch materialization still reran";
				cacheIt->second = segment;
				++m_compileCacheStats.barrierSegmentReused;
				m_compileCacheStats.barrierSegmentReusableLoweredRequirements += segment.loweredRequirementCount;
				RecordCompileReuseEvent("BarrierSegment", segment.name, true, segment.cacheKey, segment.barrierReuseReason);
			}
			else {
				segment.barriersReused = false;
				segment.barrierReuseReason = "barrier cache key matched but lowered barrier summary changed; cache key is missing an input";
				cacheIt->second = segment;
				RecordCompileReuseEvent("BarrierSegment", segment.name, false, segment.cacheKey, segment.barrierReuseReason);
			}
			++m_compileCacheStats.barrierSegmentRecompiled;
		}
		else {
			segment.barrierCacheHit = false;
			++m_compileCacheStats.barrierSegmentCacheMisses;
			segment.barriersReused = false;
			segment.barrierReuseReason = "no cached barrier segment for structure/pass/alias/queue/entry-state signature";
			++m_compileCacheStats.barrierSegmentRecompiled;
			RecordCompileReuseEvent("BarrierSegment", segment.name, false, segment.cacheKey, segment.barrierReuseReason);
			m_cachedBarrierSegments.emplace(segment.cacheKey, segment);
		}
	}

	m_compileCacheStats.segmentCount = m_compiledSegments.size();
}

void RenderGraph::FinalizeBatchesFromIR(
	RenderGraph& rg,
	std::vector<AnyPassAndResources>& passes,
	const ScheduleIR& schedule,
	const BarrierIR& barriers)
{
	ZoneScopedN("RenderGraph::FinalizeBatchesFromIR");
	(void)schedule;
	(void)barriers;

	std::unordered_map<UINT64, UINT64> concreteFenceBySymbolic;
	for (const auto& signal : barriers.signals) {
		if (signal.batch >= rg.batches.size()) {
			continue;
		}
		auto& batch = rg.batches[signal.batch];
		if (signal.queueSlot >= batch.QueueCount() || !batch.HasQueueSignal(signal.phase, signal.queueSlot)) {
			continue;
		}
		const UINT64 symbolicValue = batch.GetQueueSignalFenceValue(signal.phase, signal.queueSlot);
		if (symbolicValue == 0) {
			continue;
		}
		const UINT64 mapKey = (static_cast<UINT64>(signal.queueSlot) << 56) | symbolicValue;
		auto [it, inserted] = concreteFenceBySymbolic.emplace(mapKey, 0);
		if (inserted) {
			it->second = rg.GetNextQueueFenceValue(signal.queueSlot);
		}
		batch.SetQueueSignalFenceValue(signal.phase, signal.queueSlot, it->second);
	}

	for (const auto& wait : barriers.waits) {
		if (wait.batch >= rg.batches.size()) {
			continue;
		}
		auto& batch = rg.batches[wait.batch];
		if (wait.dstQueueSlot >= batch.QueueCount() || wait.srcQueueSlot >= batch.QueueCount()) {
			continue;
		}
		if (!batch.HasQueueWait(wait.phase, wait.dstQueueSlot, wait.srcQueueSlot)) {
			continue;
		}
		const UINT64 symbolicValue = batch.GetQueueWaitFenceValue(wait.phase, wait.dstQueueSlot, wait.srcQueueSlot);
		const UINT64 mapKey = (static_cast<UINT64>(wait.srcQueueSlot) << 56) | symbolicValue;
		auto it = concreteFenceBySymbolic.find(mapKey);
		if (it != concreteFenceBySymbolic.end()) {
			batch.queueWaitFenceValue[PassBatch::WaitPhaseIndex(wait.phase)][wait.dstQueueSlot][wait.srcQueueSlot] = it->second;
		}
		else {
			std::ostringstream oss;
			oss << "RenderGraph::FinalizeBatchesFromIR unresolved queue wait: dst=" << wait.dstQueueSlot
				<< " src=" << wait.srcQueueSlot
				<< " symbolic=" << symbolicValue
				<< " phase=" << static_cast<size_t>(wait.phase);
			spdlog::error("{}", oss.str());
			throw std::runtime_error(oss.str());
		}
	}

	CoalesceQueueWaits();
	BuildCrossFrameProducerTracking(passes);
}

void RenderGraph::CoalesceQueueWaits() {
	ZoneScopedN("RenderGraph::CoalesceQueueWaits");
	for (auto& batch : batches) {
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

void RenderGraph::BuildCrossFrameProducerTracking(const std::vector<AnyPassAndResources>& passes) {
	ZoneScopedN("RenderGraph::BuildCrossFrameProducerTracking");
	(void)passes;
	const size_t queueCount = m_queueRegistry.SlotCount();
	std::vector<std::unordered_map<uint64_t, unsigned int>> crossFrameProducer(queueCount);
	if (!m_compiledSegments.empty()) {
		for (const auto& segment : m_compiledSegments) {
			for (const auto& event : segment.replay.historyEvents) {
				if (event.kind != CachedHistoryEvent::Kind::Producer) {
					continue;
				}
				const size_t queueSlot = static_cast<size_t>(event.queueSlot);
				if (queueSlot >= crossFrameProducer.size()) {
					continue;
				}
				crossFrameProducer[queueSlot][event.resourceID] = segment.firstBatch + event.localBatch;
			}
		}
		m_compiledLastProducerBatchByResourceByQueue = std::move(crossFrameProducer);
		return;
	}

	for (unsigned int bi = 1; bi < static_cast<unsigned int>(batches.size()); ++bi) {
		auto& batch = batches[bi];
		for (size_t qi = 0; qi < queueCount; ++qi) {
			for (auto& passVariant : batch.Passes(qi)) {
				std::visit([&](const auto* passEntry) {
					for (auto& req : passEntry->resources.frameResourceRequirements) {
						if (AccessTypeIsWriteType(req.state.access)) {
							crossFrameProducer[qi][req.resourceHandleAndRange.resource.GetGlobalResourceID()] = bi;
						}
					}
				}, passVariant);
			}
		}
	}
	m_compiledLastProducerBatchByResourceByQueue = std::move(crossFrameProducer);
}

void RenderGraph::AutoScheduleAndBuildBatches(
	RenderGraph& rg,
	std::vector<RenderGraph::AnyPassAndResources>& passes,
	std::vector<RenderGraph::Node>& nodes,
	bool forceFullLowerForDiagnostics)
{
	ZoneScopedN("RenderGraph::AutoScheduleAndBuildBatches");
	const bool recordReuseEvents = (m_getRenderGraphCompileDumpEnabled && m_getRenderGraphCompileDumpEnabled()) || forceFullLowerForDiagnostics;
	const ScheduleCacheKeyParts scheduleCacheKeyParts = BuildScheduleCacheKeyParts(nodes);
	const uint64_t scheduleCacheKey = scheduleCacheKeyParts.fullHash;
	const uint64_t relaxedScheduleCacheKey = BuildRelaxedScheduleCacheKey(scheduleCacheKeyParts);
	auto scheduleCacheIt = m_cachedScheduleIRByKey.find(scheduleCacheKey);
	if (scheduleCacheIt != m_cachedScheduleIRByKey.end()) {
		++m_compileCacheStats.scheduleCacheHits;
		m_compiledScheduleIR = scheduleCacheIt->second;
		ApplyCachedScheduleIR(m_compiledScheduleIR, nodes);
		m_compileCacheStats.scheduleReusedPassCount += m_compiledScheduleIR.passStream.size();
		m_cachedScheduleIRByRelaxedKey.emplace(relaxedScheduleCacheKey, m_compiledScheduleIR);
		if (recordReuseEvents) {
			RecordCompileReuseEvent("ScheduleIR", "frame schedule", true, scheduleCacheKey, "schedule dependency/pass/queue signature matched cached schedule");
		}
	}
	else {
		const bool onlyVolatileScheduleChanged =
			m_compileCacheStats.volatileSchedulePassChanges > 0
			&& m_compileCacheStats.nonVolatileSchedulePassChanges == 0;
		auto relaxedCacheIt = m_cachedScheduleIRByRelaxedKey.find(relaxedScheduleCacheKey);
		if (onlyVolatileScheduleChanged && relaxedCacheIt != m_cachedScheduleIRByRelaxedKey.end()) {
			++m_compileCacheStats.scheduleCacheHits;
			++m_compileCacheStats.scheduleRelaxedCacheHits;
			m_compiledScheduleIR = relaxedCacheIt->second;
			ApplyCachedScheduleIR(m_compiledScheduleIR, nodes);
			m_compileCacheStats.scheduleReusedPassCount += m_compiledScheduleIR.passStream.size();
			m_cachedScheduleIRByKey.emplace(scheduleCacheKey, m_compiledScheduleIR);
			m_cachedScheduleKeyPartsByKey.emplace(scheduleCacheKey, scheduleCacheKeyParts);
			if (recordReuseEvents) {
				RecordCompileReuseEvent(
					"ScheduleIR",
					"frame schedule",
					true,
					scheduleCacheKey,
					"relaxed schedule cache hit: only readback/capture pass schedule hashes changed while topology and dependency edges matched");
			}
		}
		else {
			++m_compileCacheStats.scheduleCacheMisses;
			if (recordReuseEvents) {
				RecordCompileReuseEvent("ScheduleIR", "frame schedule", false, scheduleCacheKey, DescribeScheduleCacheMiss(scheduleCacheKeyParts, m_lastScheduleCacheKeyParts));
			}
			m_compiledScheduleIR = BuildScheduleIR(rg, passes, nodes);
			m_cachedScheduleIRByKey.emplace(scheduleCacheKey, m_compiledScheduleIR);
			m_cachedScheduleKeyPartsByKey.emplace(scheduleCacheKey, scheduleCacheKeyParts);
			m_cachedScheduleIRByRelaxedKey[relaxedScheduleCacheKey] = m_compiledScheduleIR;
		}
	}
	m_lastScheduleCacheKeyParts = scheduleCacheKeyParts;
	compileTrackers.clear();
	m_activeMaterializingSegmentName.clear();
	m_activeMaterializingSegmentMode.clear();
	m_activeMaterializingSegmentBatchRange.reset();
	m_actualEntryBoundaryStatesBySegmentId.clear();
	m_actualExitBoundaryStatesBySegmentId.clear();
	ResetFrameQueueBatchHistoryTables();
	const bool validateReplayForDump = forceFullLowerForDiagnostics;
	const std::vector<CompiledSegmentDesc> segmentDescs = BuildCompiledSegmentDescriptorsFromSchedule(m_compiledScheduleIR);
	if (recordReuseEvents) {
		m_lastSegmentPlans = BuildSegmentPlans(segmentDescs);
		for (const auto& plan : m_lastSegmentPlans) {
			if (plan.kind == SegmentPlanKind::Replay) {
				++m_compileCacheStats.plannedReplaySegmentCount;
				m_compileCacheStats.plannedReplayLoweredRequirements += plan.loweredRequirementCount;
			}
			else {
				++m_compileCacheStats.plannedLowerSegmentCount;
				m_compileCacheStats.plannedLoweredRequirements += plan.loweredRequirementCount;
			}
		}
	}
	else {
		m_lastSegmentPlans.clear();
	}
	if (!validateReplayForDump) {
		if (TryMaterializeSegmentsWithReplayAndLowering(rg, passes, nodes, segmentDescs, recordReuseEvents)) {
			return;
		}
	}

	BarrierLoweringInput loweringInput{ .schedule = &m_compiledScheduleIR };
	m_compileCacheStats.materializedLowerSegmentCount += static_cast<uint64_t>(segmentDescs.size());
	if (recordReuseEvents) {
		RecordCompileReuseEvent(
			"BarrierMaterialization",
			"frame",
			false,
			m_compiledScheduleIR.structureHash,
			"lowered the full frame because no replayable prefix or full-frame cache hit was available");
	}
	auto loweringOutput = LowerBarriers(rg, passes, nodes, loweringInput);
	m_compiledBarrierIR = std::move(loweringOutput.barriers);
	BuildCompiledSegments(segmentDescs);
	RefreshCompiledSegmentBoundaryMetadataFromReplay();
	FinalizeBatchesFromIR(rg, passes, m_compiledScheduleIR, m_compiledBarrierIR);
	if (validateReplayForDump) {
		try {
			ValidateCompiledFrameReplay();
		}
		catch (const std::exception& ex) {
			RecordCompileReuseEvent(
				"ReplayValidation",
				"frame replay",
				false,
				m_compiledScheduleIR.structureHash,
				ex.what());
		}
	}
	return;
	// Working indegrees
	std::vector<uint32_t> indeg(nodes.size());
	for (size_t i = 0; i < nodes.size(); ++i) indeg[i] = nodes[i].indegree;

	std::vector<size_t> ready;
	ready.reserve(nodes.size());
	for (size_t i = 0; i < nodes.size(); ++i)
		if (indeg[i] == 0) ready.push_back(i);

	auto openNewBatch = [&]() -> PassBatch {
		const size_t queueCount = rg.m_queueRegistry.SlotCount();
		PassBatch b(queueCount, rg.m_frameSchedulingResourceCount);
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
		rg.batches.push_back(std::move(currentBatch));
		currentBatch = openNewBatch();
		batchBuildState.ResetForNewBatch();
		++currentBatchIndex;
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

	while (remaining > 0) {
		// Collect "fits" and pick best by heuristic
		int bestIdxInReady = -1;
		size_t bestQueueSlot = 0;
		double bestScore = -1e300;
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
				CommitPassToBatch(
					rg, passes[n.passIndex], n,
					currentBatchIndex, currentBatch,
					scratchTransitioned,
					scratchFallback,
					scratchTransitions);
				updateBatchMembershipForCommittedPass(n);

				batchBuildState.MarkNode(ni);

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
			CommitPassToBatch(
				rg, passes[chosen.passIndex], chosen,
				currentBatchIndex, currentBatch,
				scratchTransitioned,
				scratchFallback,
				scratchTransitions);
			updateBatchMembershipForCommittedPass(chosen);
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

	// Coalesce redundant waits: for each (dstQueue, srcQueue) pair in a batch,
	// eliminate later-phase waits that are subsumed by earlier-phase waits, and
	// promote cross-batch fence values to earlier phases when safe.
	//
	// GPU execution order within a queue for one batch:
	//   BeforeTransitions waits -> transitions -> AfterTransitions signal
	//   BeforeExecution waits   -> passes      -> AfterExecution signal
	//   BeforeAfterPasses waits -> post-transitions -> AfterCompletion signal
	//
	// Fence values on a given queue are monotonically increasing, so a wait for
	// fence F at an earlier phase automatically satisfies any wait for F' <= F
	// at a later phase on the same source queue.
	//
	// Promoting a fence to an earlier phase is only safe when the signal producing
	// that fence will fire independently (i.e., it's from a PREVIOUS batch, not the
	// same batch). Same-batch fences reference signals produced within this batch's
	// own execution; promoting them to an earlier phase could create circular
	// cross-queue dependencies (GPU deadlock).
	{
		ZoneScopedN("RenderGraph::AutoScheduleAndBuildBatches::CoalesceQueueWaits");
		for (auto& batch : rg.batches) {
			const size_t batchQueueCount = batch.QueueCount();
			for (size_t dst = 0; dst < batchQueueCount; ++dst) {
				for (size_t src = 0; src < batchQueueCount; ++src) {
					if (dst == src) continue;

				int enabledCount = 0;
				for (size_t phase = 0; phase < PassBatch::kWaitPhaseCount; ++phase) {
					if (batch.queueWaitEnabled[phase][dst][src]) ++enabledCount;
				}
				if (enabledCount <= 1) continue;

				// Helper: true if the fence value matches one of this batch's own
				// pre-allocated signal fence values for the source queue.
				auto isSameBatchFence = [&](UINT64 f) -> bool {
					for (size_t sp = 0; sp < PassBatch::kSignalPhaseCount; ++sp) {
						if (f == batch.queueSignalFenceValue[sp][src]) return true;
					}
					return false;
				};

				// Step 1: Strict subsumption - remove later-phase waits whose fence
				// value is <= an earlier enabled phase's fence value.
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

				// Step 2: Promote cross-batch fences to the earliest phase.
				// Only promote fences that are NOT from the same batch.
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

					// Same-batch fences must stay at their original phase to avoid
					// creating circular cross-queue dependencies.
					if (isSameBatchFence(fj)) continue;

					// Promote to earliest phase (take the max)
					if (fj > batch.queueWaitFenceValue[earliest][dst][src]) {
						batch.queueWaitFenceValue[earliest][dst][src] = fj;
					}
					batch.queueWaitEnabled[j][dst][src] = false;
					batch.queueWaitFenceValue[j][dst][src] = 0;
				}

				// Step 3: Re-run subsumption - the promoted fence may now
				// subsume same-batch waits that we couldn't promote.
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
						for (auto& req : passEntry->resources.frameResourceRequirements) {
							if (AccessTypeIsWriteType(req.state.access)) {
								crossFrameProducer[qi][req.resourceHandleAndRange.resource.GetGlobalResourceID()] = bi;
							}
						}
					}, passVariant);
				}
			}
		}
		rg.m_compiledLastProducerBatchByResourceByQueue = std::move(crossFrameProducer);
	}
}


// Factory for the transition lambda
void RenderGraph::AppendBatchTransition(
	PassBatch& batch,
	unsigned int batchIndex,
	size_t queueSlot,
	BatchTransitionPhase phase,
	size_t resourceIndex,
	const ResourceTransition& transition)
{
	auto& batchTransitions = batch.Transitions(queueSlot, phase);
	batchTransitions.push_back(transition);
	batch.RecordTransitionPosition(
		queueSlot,
		phase,
		resourceIndex,
		static_cast<uint32_t>(batchTransitions.size() - 1));
	RecordFrameTransitionPlacementBatch(resourceIndex, batchIndex);
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

	auto resource = requirement.resource;
	const ResourceState requiredState = [&]() {
		ZoneScopedN("RenderGraph::AddTransition::NormalizeStateForQueue");
		return NormalizeStateForQueue(passQueue, requirement.state);
	}();

	// If this triggers, you're probably queueing an operation on an external/ephemeral resource, and then discarding it before the graph can use it.
	{
		ZoneScopedN("RenderGraph::AddTransition::ValidateResourceHandle");
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
	}
	scratchTransitions.clear();
	auto& transitions = scratchTransitions;
	auto pRes = [&]() {
		ZoneScopedN("RenderGraph::AddTransition::ResolveResource");
		return _registry.Resolve(resource);
	}(); // TODO: Can we get rid of pRes in transitions?
	if (pRes && !pRes->GetName().empty()) {
		ZoneText(pRes->GetName().data(), pRes->GetName().size());
	}
	{
		ZoneScopedN("RenderGraph::AddTransition::ValidateTextureState");
		if (pRes && pRes->HasLayout() && !ValidateResourceLayoutAndAccessType(requiredState.layout, requiredState.access)) {
		spdlog::error(
			"Invalid texture state in RenderGraph::AddTransition: pass='{}' resource='{}' id={} range={} access={} layout={} sync={} queue={}.",
			passName,
			pRes->GetName(),
			resource.GetGlobalResourceID(),
			FormatRangeSpec(requirement.range),
			rhi::helpers::ResourceAccessMaskToString(requiredState.access),
			static_cast<uint32_t>(requiredState.layout),
			static_cast<uint32_t>(requiredState.sync),
			QueueKindToString(passQueue));
		throw std::runtime_error("Invalid texture layout/access combination in RenderGraph::AddTransition");
		}
	}
	auto& compileTracker = [&]() -> SymbolicTracker& {
		ZoneScopedN("RenderGraph::AddTransition::GetOrCreateCompileTracker");
		return GetOrCreateCompileTracker(pRes, resource.GetGlobalResourceID());
	}();

	auto widenLatestCompatibleTransition = [&]() {
		ZoneScopedN("RenderGraph::AddTransition::WidenLatestCompatibleTransition");
		auto widenInBatch = [&](PassBatch& batch) {
			ZoneScopedN("RenderGraph::AddTransition::WidenInBatch");
			bool widened = false;
			const auto wantedRange = pRes && pRes->HasLayout()
				? ResolveRangeSpec(requirement.range, pRes->GetMipLevels(), pRes->GetArraySize())
				: SubresourceRange{ 0, 1, 0, 1 };

			for (size_t queueIndex = 0; queueIndex < batch.QueueCount(); ++queueIndex) {
				for (size_t phaseIndex = 0; phaseIndex < static_cast<size_t>(BatchTransitionPhase::Count); ++phaseIndex) {
					if (requirement.resourceIndex >= batch.TransitionIndexedResourceCount()) {
						continue;
					}

					auto& batchTransitions = batch.Transitions(queueIndex, static_cast<BatchTransitionPhase>(phaseIndex));
					for (uint32_t transitionPosition : batch.TransitionPositions(queueIndex, static_cast<BatchTransitionPhase>(phaseIndex), requirement.resourceIndex)) {
						if (transitionPosition >= batchTransitions.size()) {
							continue;
						}

						auto& existingTransition = batchTransitions[transitionPosition];
						if (existingTransition.pResource != pRes) {
							continue;
						}

						if (pRes && pRes->HasLayout()) {
							const auto existingRange = ResolveRangeSpec(
								existingTransition.range,
								pRes->GetMipLevels(),
								pRes->GetArraySize());
							if (!SubresourceRangesOverlap(existingRange, wantedRange)) {
								continue;
							}
						}

						const auto combinedAccess = ComposeCompatibleAccessTypes(existingTransition.newAccessType, requiredState.access);
						if (pRes && pRes->HasLayout()) {
							if (existingTransition.newLayout != requiredState.layout) {
								continue;
							}

							if (!ValidateResourceLayoutAndAccessType(existingTransition.newLayout, combinedAccess)) {
								continue;
							}
						}

						existingTransition.newAccessType = combinedAccess;
						existingTransition.newSyncState |= requiredState.sync;
						widened = true;
					}
				}
			}

			return widened;
		};

		if (widenInBatch(currentBatch)) {
			return;
		}

		if (requirement.resourceIndex >= m_frameTransitionPlacementBatchesByResource.size()) {
			return;
		}

		const auto& transitionBatchHistory = m_frameTransitionPlacementBatchesByResource[requirement.resourceIndex];
		for (auto historyIt = transitionBatchHistory.rbegin(); historyIt != transitionBatchHistory.rend(); ++historyIt) {
			const unsigned int priorBatchIndex = *historyIt;
			if (priorBatchIndex == 0 || priorBatchIndex >= batchIndex) {
				continue;
			}
			if (m_activeLoweringBatchRange.has_value()) {
				const auto [firstAllowedBatch, lastAllowedBatch] = *m_activeLoweringBatchRange;
				if (priorBatchIndex < firstAllowedBatch || priorBatchIndex > lastAllowedBatch) {
					continue;
				}
			}

			if (widenInBatch(batches[priorBatchIndex])) {
				return;
			}
		}
	};

	auto absorbIntoExistingTransition = [&](PassBatch& batch, unsigned int targetBatchIndex, size_t targetQueueSlot, BatchTransitionPhase targetPhase) {
		ZoneScopedN("RenderGraph::AddTransition::AbsorbIntoExistingTransition");
		if (requirement.resourceIndex >= batch.TransitionIndexedResourceCount()) {
			return false;
		}

		const auto wantedRange = pRes && pRes->HasLayout()
			? ResolveRangeSpec(requirement.range, pRes->GetMipLevels(), pRes->GetArraySize())
			: SubresourceRange{ 0, 1, 0, 1 };
		auto& batchTransitions = batch.Transitions(targetQueueSlot, targetPhase);
		bool absorbedAny = false;

		for (uint32_t transitionPosition : batch.TransitionPositions(targetQueueSlot, targetPhase, requirement.resourceIndex)) {
			if (transitionPosition >= batchTransitions.size()) {
				continue;
			}

			auto& existingTransition = batchTransitions[transitionPosition];
			if (existingTransition.pResource != pRes) {
				continue;
			}

			if (pRes && pRes->HasLayout()) {
				const auto existingRange = ResolveRangeSpec(
					existingTransition.range,
					pRes->GetMipLevels(),
					pRes->GetArraySize());
				if (!SubresourceRangesOverlap(existingRange, wantedRange)) {
					continue;
				}
			}

			for (const auto& transition : transitions) {
				if (existingTransition.prevAccessType != transition.prevAccessType
					|| existingTransition.prevLayout != transition.prevLayout
					|| existingTransition.prevSyncState != transition.prevSyncState
					|| existingTransition.newLayout != transition.newLayout
					|| existingTransition.discard != transition.discard) {
					continue;
				}

				const auto combinedAccess = ComposeCompatibleAccessTypes(existingTransition.newAccessType, transition.newAccessType);
				if (pRes && pRes->HasLayout() && !ValidateResourceLayoutAndAccessType(existingTransition.newLayout, combinedAccess)) {
					continue;
				}

				existingTransition.newAccessType = combinedAccess;
				existingTransition.newSyncState |= transition.newSyncState;
				RecordFrameTransitionPlacementBatch(requirement.resourceIndex, targetBatchIndex);
				absorbedAny = true;
			}
		}

		return absorbedAny;
	};

	bool isAliasActivation = false;
	{
		ZoneScopedN("RenderGraph::AddTransition::ResolveTransitions");
		if (aliasActivationPending.find(resource.GetGlobalResourceID()) != aliasActivationPending.end()) {
		isAliasActivation = true;
		const bool firstUseIsWrite = AccessTypeIsWriteType(requirement.state.access);
		const bool firstUseIsCommon = requirement.state.access == rhi::ResourceAccessType::Common;
		// Common counts as write for alias activation, as this is generally used to indicate that the resource will be
		// transitioned internally by an external system that still uses legacy barriers. Don't abuse this.
		if (firstUseIsWrite || firstUseIsCommon) {
			const uint64_t id = resource.GetGlobalResourceID();
			auto itSig = aliasPlacementSignatureByID.find(id);
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
				rhi::ResourceAccessType::None,
				requiredState.access,
				rhi::ResourceLayout::Undefined,
				requiredState.layout,
				rhi::ResourceSyncState::None,
				requiredState.sync,
				true);
		}
		else {
			throw std::runtime_error("Alias activation requires first use to be a write when explicit initialization is disabled");
		}
		{
			ZoneScopedN("RenderGraph::AddTransition::AliasActivationApply");
			std::vector<ResourceTransition> ignored;
			compileTracker.Apply(requirement.range, pRes, requiredState, ignored);
		}
		aliasActivationPending.erase(resource.GetGlobalResourceID());
	}
	else {
		{
			ZoneScopedN("RenderGraph::AddTransition::TrackerApply");
			compileTracker.Apply(requirement.range, pRes, requiredState, transitions);
		}
		if (transitions.empty()) {
			widenLatestCompatibleTransition();
		}
	}
	}

	{
		ZoneScopedN("RenderGraph::AddTransition::UpdateBatchTrackerState");
		if (!transitions.empty()) {
		outTransitionedResourceIDs.insert(resource.GetGlobalResourceID());
	}

	if (requirement.resourceIndex < currentBatch.passBatchTrackersByResourceIndex.size()) {
		currentBatch.passBatchTrackersByResourceIndex[requirement.resourceIndex] = &compileTracker;
	}
	}

	if (transitions.empty()) {
		return;
	}

	bool needsGraphicsQueueForTransitions = false;
	{
		ZoneScopedN("RenderGraph::AddTransition::CheckQueueTransitionSupport");
		for (auto& transition : transitions) {
			if (!QueueSupportsTransition(passQueue, transition)) {
				needsGraphicsQueueForTransitions = true;
				break;
			}
		}
	}

	const size_t gfxSlot = QueueIndex(QueueKind::Graphics);
	const size_t transitionSlot = (passQueue != QueueKind::Graphics && needsGraphicsQueueForTransitions)
		? gfxSlot : passQueueSlot;

	// Try early placement: move transitions to AfterPasses of the batch where the resource was last used.
	// This reduces GPU idle time by allowing transitions to overlap with unrelated work on other queues.
	// Skip alias activations - those must stay in the consuming batch (discard semantics at first use).
	if (!isAliasActivation) {
		ZoneScopedN("RenderGraph::AddTransition::TryEarlyPlacement");
		unsigned int lastUseBatch = 0;
		uint64_t lastUseQueueMask = 0;
		bool requiresCrossQueuePlacementCoordination = false;
		const bool canUseEventSummary = !m_frameResourceEventSummaries.empty();
		if (canUseEventSummary) {
			ZoneScopedN("RenderGraph::AddTransition::EarlyPlacementEventSummaryLookup");
			std::tie(lastUseBatch, lastUseQueueMask) = GetFrameResourceLastEventBeforeBatch(requirement.resourceIndex, batchIndex);
			requiresCrossQueuePlacementCoordination =
				lastUseQueueMask != 0 &&
				(lastUseQueueMask & ~(uint64_t{ 1 } << transitionSlot)) != 0;
		}
		else {
			ZoneScopedN("RenderGraph::AddTransition::EarlyPlacementHistoryScan");
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

		const bool lastUseInsideActiveLoweringSegment = [&]() {
			if (!m_activeLoweringBatchRange.has_value()) {
				return true;
			}
			const auto [firstAllowedBatch, lastAllowedBatch] = *m_activeLoweringBatchRange;
			return lastUseBatch >= firstAllowedBatch && lastUseBatch <= lastAllowedBatch;
		}();

		if (lastUseBatch > 0 && !requiresCrossQueuePlacementCoordination && lastUseInsideActiveLoweringSegment) { // > 0 to skip batch 0 (placeholder with no fence values)
			ZoneScopedN("RenderGraph::AddTransition::PlaceTransitionsInPriorBatch");
			PassBatch& targetBatch = batches[lastUseBatch];
			const bool absorbedIntoExisting = absorbIntoExistingTransition(
				targetBatch,
				lastUseBatch,
				transitionSlot,
				BatchTransitionPhase::AfterPasses);

			if (!absorbedIntoExisting) {
				for (auto& transition : transitions) {
					AppendBatchTransition(targetBatch, lastUseBatch, transitionSlot, BatchTransitionPhase::AfterPasses, requirement.resourceIndex, transition);
				}
			}

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
		ZoneScopedN("RenderGraph::AddTransition::PlaceTransitionsOnGraphicsFallback");
		// The consuming pass's queue can't support these transitions, so delegate
		// them to the graphics queue within the *current* batch's BeforePasses phase.
		// CommitPassToBatch will set up:
		//   1. BeforeTransitions waits on Graphics for any prior non-graphics producers
		//   2. AfterTransitions signal on Graphics so the consuming queue can wait
		for (auto& transition : transitions) {
			AppendBatchTransition(currentBatch, batchIndex, gfxSlot, BatchTransitionPhase::BeforePasses, requirement.resourceIndex, transition);
		}
		outFallbackResourceIndices.insert(requirement.resourceIndex);
	}
	else {
		ZoneScopedN("RenderGraph::AddTransition::PlaceTransitionsInCurrentBatch");
		for (auto& transition : transitions) {
			AppendBatchTransition(currentBatch, batchIndex, passQueueSlot, BatchTransitionPhase::BeforePasses, requirement.resourceIndex, transition);
		}
	}
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

	for (const auto& resourceRequirement : resourceRequirements) {
		AddTransition(batchIndex, currentBatch, passQueueSlot, passName, resourceRequirement, outTransitionedResourceIDs, outFallbackResourceIndices, scratchTransitions);

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
	DeletionManager::GetInstance().Cleanup();
	DeviceManager::GetInstance().Cleanup();
}

void RenderGraph::ShutdownOwnedState() {
	batches.clear();
	initialTransitions.clear();
	trackers.clear();
	compileTrackers.clear();
	m_masterPassList.clear();
	m_framePasses.clear();
	m_assignedQueueSlotsByFramePass.clear();
	m_activeQueueSlotsThisFrame.clear();
	renderPassesByName.clear();
	computePassesByName.clear();
	resourcesByID.clear();
	resourcesByName.clear();
	m_transientFrameResourcesByID.clear();
	m_transientFrameResourcesByName.clear();
	resourceBackingGenerationByID.clear();
	resourceIdleFrameCounts.clear();
	compiledResourceGenerationByID.clear();
	aliasMaterializeOptionsByID.clear();
	aliasPlacementSignatureByID.clear();
	aliasPlacementRangesByID.clear();
	schedulingPlacementRangesByID.clear();
	m_schedulingEquivalentIDsCache.clear();
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
	std::fill(m_lastSubmittedSignalValueByQueue.begin(), m_lastSubmittedSignalValueByQueue.end(), UINT64(0));

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

	m_statisticsService.reset();
	m_uploadService.reset();
	m_readbackService.reset();
	m_descriptorService.reset();
	m_renderGraphSettingsService.reset();
}

SymbolicTracker& RenderGraph::GetOrCreateCompileTracker(Resource* resource, uint64_t resourceID) {
	auto it = compileTrackers.find(resourceID);
	if (it != compileTrackers.end()) {
		return it->second;
	}

	SymbolicTracker seed{};
	if (resource) {
		bool hasLiveBacking = true;
		if (auto* texture = dynamic_cast<PixelBuffer*>(resource)) {
			hasLiveBacking = texture->IsMaterialized();
		}
		else if (auto* buffer = dynamic_cast<Buffer*>(resource)) {
			hasLiveBacking = buffer->IsMaterialized();
		}

		if (hasLiveBacking) {
			seed = *resource->GetStateTracker();
		}
	}

	auto [insertedIt, _] = compileTrackers.emplace(resourceID, std::move(seed));
	return insertedIt->second;
}

void RenderGraph::CaptureCompileTrackersForExecution(const std::unordered_set<uint64_t>& resourceIDs) {
	ZoneScopedN("RenderGraph::CaptureCompileTrackersForExecution");
	trackers.clear();
	trackers.reserve(resourceIDs.size());

	auto captureTracker = [&](uint64_t resourceID, Resource* resource) {
		if (!resource) {
			return;
		}

		if (!compileTrackers.contains(resourceID)) {
			return;
		}

		bool hasLiveBacking = true;
		if (auto* texture = dynamic_cast<PixelBuffer*>(resource)) {
			hasLiveBacking = texture->IsMaterialized();
		}
		else if (auto* buffer = dynamic_cast<Buffer*>(resource)) {
			hasLiveBacking = buffer->IsMaterialized();
		}

		if (!hasLiveBacking) {
			return;
		}

		if (auto* tracker = resource->GetStateTracker()) {
			trackers[resourceID] = tracker;
		}
	};

	for (uint64_t resourceID : resourceIDs) {
		auto itResource = resourcesByID.find(resourceID);
		if (itResource != resourcesByID.end() && itResource->second) {
			captureTracker(resourceID, itResource->second.get());
			continue;
		}

		auto itTransient = m_transientFrameResourcesByID.find(resourceID);
		if (itTransient != m_transientFrameResourcesByID.end() && itTransient->second) {
			captureTracker(resourceID, itTransient->second.get());
		}
	}
}

void RenderGraph::PublishCompiledTrackerStates() {
	for (const auto& [resourceID, liveTracker] : trackers) {
		if (!liveTracker) {
			continue;
		}

		auto itCompileTracker = compileTrackers.find(resourceID);
		if (itCompileTracker == compileTrackers.end()) {
			continue;
		}

		liveTracker->CopyFrom(itCompileTracker->second);
	}
}

void RenderGraph::RebuildSchedulingEquivalentIDCache(const std::unordered_set<uint64_t>& resourceIDs) {
	ZoneScopedN("RenderGraph::RebuildSchedulingEquivalentIDCache");
	m_schedulingEquivalentIDsCache.clear();
	m_schedulingEquivalentIDsCache.reserve(resourceIDs.size());

	for (uint64_t resourceID : resourceIDs) {
		m_schedulingEquivalentIDsCache.emplace(
			resourceID,
			m_aliasingSubsystem.GetSchedulingEquivalentIDs(resourceID, schedulingPlacementRangesByID));
	}
}

const std::vector<uint64_t>& RenderGraph::GetSchedulingEquivalentIDsCached(uint64_t resourceID) {
	auto it = m_schedulingEquivalentIDsCache.find(resourceID);
	if (it != m_schedulingEquivalentIDsCache.end()) {
		return it->second;
	}

	auto [insertedIt, inserted] = m_schedulingEquivalentIDsCache.emplace(
		resourceID,
		m_aliasingSubsystem.GetSchedulingEquivalentIDs(resourceID, schedulingPlacementRangesByID));
	(void)inserted;
	return insertedIt->second;
}

void RenderGraph::ClearFrameSchedulingResourceIndex() {
	m_frameSchedulingResourceIndexByID.clear();
	m_frameSchedulingResourceIDsByIndex.clear();
	m_frameSchedulingResourceCount = 0;
	m_frameQueueLastUsageBatch.clear();
	m_frameQueueLastProducerBatch.clear();
	m_frameQueueLastTransitionBatch.clear();
	m_frameTransitionPlacementBatchesByResource.clear();
	m_frameResourceEventSummaries.clear();
	m_frameResourceEventLog.clear();
}

void RenderGraph::ClearFramePassSchedulingSummaries() {
	m_framePassSchedulingSummaries.clear();
}

void RenderGraph::ResetFrameQueueBatchHistoryTables() {
	const size_t entryCount = m_queueRegistry.SlotCount() * m_frameSchedulingResourceCount;
	m_frameQueueLastUsageBatch.assign(entryCount, 0);
	m_frameQueueLastProducerBatch.assign(entryCount, 0);
	m_frameQueueLastTransitionBatch.assign(entryCount, 0);
	m_frameTransitionPlacementBatchesByResource.assign(m_frameSchedulingResourceCount, {});
	if (m_queueRegistry.SlotCount() <= 64) {
		m_frameResourceEventSummaries.assign(m_frameSchedulingResourceCount, FrameResourceEventSummary{});
	}
	else {
		m_frameResourceEventSummaries.clear();
	}
	m_frameResourceEventLog.clear();
}

void RenderGraph::RebuildFrameSchedulingResourceIndex(const std::unordered_set<uint64_t>& resourceIDs) {
	ZoneScopedN("RenderGraph::RebuildFrameSchedulingResourceIndex");
	m_frameSchedulingResourceIndexByID.clear();
	m_frameSchedulingResourceIndexByID.reserve(resourceIDs.size() * 2);
	m_frameSchedulingResourceIDsByIndex.clear();
	m_frameSchedulingResourceCount = 0;

	auto registerResourceID = [&](uint64_t resourceID) {
		auto [_, inserted] = m_frameSchedulingResourceIndexByID.emplace(resourceID, m_frameSchedulingResourceCount);
		if (inserted) {
			m_frameSchedulingResourceIDsByIndex.push_back(resourceID);
			++m_frameSchedulingResourceCount;
		}
	};

	for (uint64_t resourceID : resourceIDs) {
		registerResourceID(resourceID);
		for (uint64_t equivalentID : GetSchedulingEquivalentIDsCached(resourceID)) {
			registerResourceID(equivalentID);
		}
	}

	ResetFrameQueueBatchHistoryTables();
}

void RenderGraph::RebuildFramePassSchedulingSummaries() {
	ZoneScopedN("RenderGraph::RebuildFramePassSchedulingSummaries");
	m_framePassSchedulingSummaries.clear();
	m_framePassSchedulingSummaries.resize(m_framePasses.size());

	for (size_t passIndex = 0; passIndex < m_framePasses.size(); ++passIndex) {
		auto& summary = m_framePassSchedulingSummaries[passIndex];
		PassView view = GetPassView(m_framePasses[passIndex]);

		auto appendEquivalentResourceIndices = [&](uint64_t resourceID, std::vector<size_t>& outIndices) {
			for (uint64_t equivalentID : GetSchedulingEquivalentIDsCached(resourceID)) {
				if (equivalentID == resourceID) {
					continue;
				}
				auto resourceIndex = TryGetFrameSchedulingResourceIndex(equivalentID);
				if (!resourceIndex.has_value()) {
					continue;
				}
				outIndices.push_back(*resourceIndex);
			}
			std::sort(outIndices.begin(), outIndices.end());
			outIndices.erase(std::unique(outIndices.begin(), outIndices.end()), outIndices.end());
		};

		for (const auto& req : *view.reqs) {
			auto resourceIndex = TryGetFrameSchedulingResourceIndex(req.resourceHandleAndRange.resource.GetGlobalResourceID());
			if (!resourceIndex.has_value()) {
				continue;
			}

			DenseRequirementSummary denseRequirement{};
			denseRequirement.resource = req.resourceHandleAndRange.resource;
			denseRequirement.resourceID = req.resourceHandleAndRange.resource.GetGlobalResourceID();
			denseRequirement.resourceIndex = *resourceIndex;
			denseRequirement.range = req.resourceHandleAndRange.range;
			denseRequirement.state = req.state;
			denseRequirement.isUAV = IsUAVState(req.state);
			appendEquivalentResourceIndices(denseRequirement.resourceID, denseRequirement.equivalentResourceIndices);
			summary.requirements.push_back(std::move(denseRequirement));
			summary.requiredResourceIndices.push_back(*resourceIndex);
			summary.touchedResourceIndices.push_back(*resourceIndex);
			if (IsUAVState(req.state)) {
				summary.uavResourceIndices.push_back(*resourceIndex);
			}
		}

		for (const auto& transition : *view.internalTransitions) {
			auto resourceIndex = TryGetFrameSchedulingResourceIndex(transition.first.resource.GetGlobalResourceID());
			if (!resourceIndex.has_value()) {
				continue;
			}

			DenseEquivalentResourceSummary denseTransition{};
			denseTransition.resourceID = transition.first.resource.GetGlobalResourceID();
			denseTransition.resourceIndex = *resourceIndex;
			appendEquivalentResourceIndices(denseTransition.resourceID, denseTransition.equivalentResourceIndices);
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

std::optional<size_t> RenderGraph::TryGetFrameSchedulingResourceIndex(uint64_t resourceID) const {
	auto it = m_frameSchedulingResourceIndexByID.find(resourceID);
	if (it == m_frameSchedulingResourceIndexByID.end()) {
		return std::nullopt;
	}
	return it->second;
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
	if (batchIndex == 0 || queueSlot >= 64 || resourceIndex >= m_frameSchedulingResourceCount) {
		return;
	}

	if (resourceIndex < m_frameSchedulingResourceIDsByIndex.size()) {
		m_frameResourceEventLog.push_back(FrameResourceEventRecord{
			.batchIndex = batchIndex,
			.queueSlot = static_cast<uint16_t>(queueSlot),
			.resourceID = m_frameSchedulingResourceIDsByIndex[resourceIndex],
		});
	}

	if (resourceIndex >= m_frameResourceEventSummaries.size()) {
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

void RenderGraph::RecordFrameTransitionPlacementBatch(size_t resourceIndex, unsigned int batchIndex) {
	if (batchIndex == 0 || resourceIndex >= m_frameTransitionPlacementBatchesByResource.size()) {
		return;
	}

	auto& batchHistory = m_frameTransitionPlacementBatchesByResource[resourceIndex];
	if (batchHistory.empty() || batchHistory.back() < batchIndex) {
		batchHistory.push_back(batchIndex);
		return;
	}
	if (batchHistory.back() == batchIndex) {
		return;
	}

	auto insertIt = std::lower_bound(batchHistory.begin(), batchHistory.end(), batchIndex);
	if (insertIt == batchHistory.end() || *insertIt != batchIndex) {
		batchHistory.insert(insertIt, batchIndex);
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

	for (auto const& pr : m_framePasses) {
		std::visit([&](auto const& passAndResources) {
			using T = std::decay_t<decltype(passAndResources)>;
			if constexpr (!std::is_same_v<T, std::monostate>) {
				for (auto const& req : passAndResources.resources.frameResourceRequirements) {
					used.insert(req.resourceHandleAndRange.resource.GetGlobalResourceID());
				}
				for (auto const& t : passAndResources.resources.internalTransitions) {
					used.insert(t.first.resource.GetGlobalResourceID());
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

	const auto& incomingType = typeid(*ext);
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
	// Clear any existing compile state
	m_masterPassList.clear();
	m_framePasses.clear();
	m_assignedQueueSlotsByFramePass.clear();
	m_activeQueueSlotsThisFrame.clear();
	batches.clear();
	m_executionSchedule.batches.clear();
	trackers.clear();
	m_compiledFrameProgram = FrameProgram{};
	m_compiledScheduleIR = ScheduleIR{};
	m_compiledBarrierIR = BarrierIR{};
	m_compiledSegments.clear();
	m_cachedPassIRByStableId.clear();
	m_cachedScheduleIRByKey.clear();
	m_cachedScheduleIRByRelaxedKey.clear();
	m_cachedScheduleKeyPartsByKey.clear();
	m_lastScheduleCacheKeyParts.reset();
	m_cachedBarrierSegments.clear();
	m_resourceAccessChains.clear();
	m_dependencyEdgeIR.clear();
	m_compileCacheStats = CompileCacheStats{};
	m_compileReuseEvents.clear();

	// Full rebuilds must drop cached pass instances before clearing resources.
	// Builders reuse pass objects across frames, and many passes capture
	// resource-owning shared_ptrs through constructor arguments.
	for (auto& [name, builder] : m_passBuildersByName) {
		(void)name;
		if (builder) {
			builder->Reset();
		}
	}

	// Clear resources
	resourcesByID.clear();
	resourcesByName.clear();
	m_transientFrameResourcesByID.clear();
	m_transientFrameResourcesByName.clear();
	resourceBackingGenerationByID.clear();
	resourceIdleFrameCounts.clear();
	compiledResourceGenerationByID.clear();
	m_aliasingSubsystem.ResetPersistentState(*this);
	compileTrackers.clear();
	m_schedulingEquivalentIDsCache.clear();
	ClearFrameSchedulingResourceIndex();
	ClearFramePassSchedulingSummaries();
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
	m_compiledLastProducerBatchByResourceByQueue.clear();
	m_hasPendingFrameStartQueueWait.clear();
	m_pendingFrameStartQueueWaitFenceValue.clear();
	m_lastSubmittedSignalValueByQueue.clear();
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
	// clear pass ordering
	m_passBuilderOrder.clear();
	m_passNamesSeenThisReset.clear();

}

void RenderGraph::ResetForFrame() {
	ZoneScopedN("RenderGraph::ResetForFrame");
	batches.clear();
	compiledResourceGenerationByID.clear();
	m_transientFrameResourcesByID.clear();
	m_transientFrameResourcesByName.clear();
	_registry.ReclaimExpiredAnonymous();
	m_aliasingSubsystem.ResetPerFrameState(*this);
	compileTrackers.clear();
	m_schedulingEquivalentIDsCache.clear();
	ClearFrameSchedulingResourceIndex();
	ClearFramePassSchedulingSummaries();
	m_assignedQueueSlotsByFramePass.clear();
	m_activeQueueSlotsThisFrame.clear();
	m_executionSchedule.Reset();
	m_compiledFrameProgram = FrameProgram{};
	m_compiledScheduleIR = ScheduleIR{};
	m_compiledBarrierIR = BarrierIR{};
	m_compiledSegments.clear();
	m_resourceAccessChains.clear();
	m_dependencyEdgeIR.clear();
	m_compileCacheStats = CompileCacheStats{};
	m_compileReuseEvents.clear();
	for (auto& producerMap : m_compiledLastProducerBatchByResourceByQueue) {
		producerMap.clear();
	}
	for (auto& row : m_hasPendingFrameStartQueueWait) {
		std::fill(row.begin(), row.end(), false);
	}
	for (auto& row : m_pendingFrameStartQueueWaitFenceValue) {
		std::fill(row.begin(), row.end(), UINT64(0));
	}
	// reset pass builders
	for (auto& [name, builder] : m_passBuildersByName) {
		builder->Reset();
	}
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
	std::vector<ResourceRequirement> const& retained,
	std::vector<ResourceRequirement> const& immediate)
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


void RenderGraph::RefreshRetainedDeclarationsForFrame(RenderPassAndResources& p, uint8_t frameIndex)
{
	ZoneScopedN("RenderGraph::RefreshRetainedDeclarationsForFrame(Render)");
	const bool traceLifecycle = m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled();
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

	// Update the frame view used by scheduling
	p.resources.staticResourceRequirements = b.GatherResourceRequirements();

	// Internal transitions also affect scheduling
	p.resources.internalTransitions = b.params.internalTransitions;

	p.resources.identifierSet = b.DeclaredResourceIds();
	p.resources.autoDescriptorShaderResources = b.params.autoDescriptorShaderResources;
	p.resources.autoDescriptorConstantBuffers = b.params.autoDescriptorConstantBuffers;
	p.resources.autoDescriptorUnorderedAccessViews = b.params.autoDescriptorUnorderedAccessViews;
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh render pass '{}' materialize referenced resources begin", frameIndex, p.name);
	}
	MaterializeReferencedResources(p.resources.staticResourceRequirements, p.resources.internalTransitions);

	// Transfer resolver snapshots for auto-invalidation tracking
	p.resolverSnapshots = b.TakeResolverSnapshots();
	p.retainedAnonymousKeepAlive = CaptureRetainedAnonymousKeepAlive(
		p.resources.staticResourceRequirements,
		p.resources.internalTransitions);

	// Ensure the pass's view matches the refreshed identifier set
	p.pass->SetResourceRegistryView(
		std::make_unique<ResourceRegistryView>(_registry, p.resources.identifierSet),
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
}

void RenderGraph::RefreshRetainedDeclarationsForFrame(ComputePassAndResources& p, uint8_t frameIndex)
{
	ZoneScopedN("RenderGraph::RefreshRetainedDeclarationsForFrame(Compute)");
	const bool traceLifecycle = m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled();
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

	p.resources.staticResourceRequirements = b.GatherResourceRequirements();
	p.resources.internalTransitions = b.params.internalTransitions;
	p.resources.identifierSet = b.DeclaredResourceIds();
	p.resources.autoDescriptorShaderResources = b.params.autoDescriptorShaderResources;
	p.resources.autoDescriptorConstantBuffers = b.params.autoDescriptorConstantBuffers;
	p.resources.autoDescriptorUnorderedAccessViews = b.params.autoDescriptorUnorderedAccessViews;
	if (traceLifecycle) {
		spdlog::info("RG frame {} refresh compute pass '{}' materialize referenced resources begin", frameIndex, p.name);
	}
	MaterializeReferencedResources(p.resources.staticResourceRequirements, p.resources.internalTransitions);

	// Transfer resolver snapshots for auto-invalidation tracking
	p.resolverSnapshots = b.TakeResolverSnapshots();
	p.retainedAnonymousKeepAlive = CaptureRetainedAnonymousKeepAlive(
		p.resources.staticResourceRequirements,
		p.resources.internalTransitions);

	p.pass->SetResourceRegistryView(
		std::make_unique<ResourceRegistryView>(_registry, p.resources.identifierSet),
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
}

void RenderGraph::RefreshRetainedDeclarationsForFrame(CopyPassAndResources& p, uint8_t frameIndex)
{
	ZoneScopedN("RenderGraph::RefreshRetainedDeclarationsForFrame(Copy)");
	const bool traceLifecycle = m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled();
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

	p.resources.staticResourceRequirements = b.GatherResourceRequirements();
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
		compileTrackers.clear();
		m_schedulingEquivalentIDsCache.clear();
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

		auto* iFace = dynamic_cast<IDynamicDeclaredResources*>(p.pass.get());
		if (!iFace) {
			// if pass doesn't opt-in, assume no change
			return false;
		}

		return iFace->DeclaredResourcesChanged();
		};

	{
		ZoneScopedN("RenderGraph::CompileFrame::RefreshRetainedDeclarations");
		// First, refresh all retained declarations for this frame
		for (auto& pr : m_masterPassList) {
			if (pr.type == PassType::Compute) {
				auto& p = std::get<ComputePassAndResources>(pr.pass);
				p.immediateBytecode.clear();
				p.resources.frameResourceRequirements = {};// p.resources.staticResourceRequirements;
				if (needsRefresh(p)) {
					RefreshRetainedDeclarationsForFrame(p, frameIndex);
				}
			}
			else if (pr.type == PassType::Render) {
				auto& p = std::get<RenderPassAndResources>(pr.pass);
				p.immediateBytecode.clear();
				p.resources.frameResourceRequirements = {};// p.resources.staticResourceRequirements;
				if (needsRefresh(p)) {
					RefreshRetainedDeclarationsForFrame(p, frameIndex);
				}
			}
			else if (pr.type == PassType::Copy) {
				auto& p = std::get<CopyPassAndResources>(pr.pass);
				p.immediateBytecode.clear();
				p.resources.frameResourceRequirements = {};
				if (needsRefresh(p)) {
					RefreshRetainedDeclarationsForFrame(p, frameIndex);
				}
			}
		}
	}

	{
		ZoneScopedN("RenderGraph::CompileFrame::InitFramePassState");
		batches.clear();
		batches.emplace_back(m_queueRegistry.SlotCount(), m_frameSchedulingResourceCount); // Dummy batch 0 for pre-first-pass transitions
		m_framePasses.clear(); // Combined retained + immediate-mode passes for this frame
	}

	// Record immediate-mode commands + access for each pass and fold into per-frame requirements
	for (auto& pr : m_masterPassList) {

		if (pr.type == PassType::Compute) {
			auto& p = std::get<ComputePassAndResources>(pr.pass);

			// reset per-frame
			p.immediateBytecode.clear();
			p.resources.frameResourceRequirements = p.resources.staticResourceRequirements;

			ImmediateExecutionContext c{ device,
				{rg::imm::ImmediatePassKind::Compute,
				m_immediateDispatch,
				&ResolveByIdThunk,
				&ResolveByPtrThunk,
				this},
				frameIndex,
				hostData
			};

			// Record immediate-mode commands
			{
				ZoneScopedN("RenderGraph::CompileFrame::RecordImmediateCommands");
				if (!p.name.empty()) {
					ZoneText(p.name.data(), p.name.size());
				}
				if (traceLifecycle) {
					spdlog::info("RG frame {} compute pass '{}' RecordImmediateCommands begin", frameIndex, p.name);
				}
				p.pass->RecordImmediateCommands(c);
				if (traceLifecycle) {
					spdlog::info("RG frame {} compute pass '{}' RecordImmediateCommands complete", frameIndex, p.name);
				}
			}

			auto immediateFrameData = c.list.Finalize();
			// If there is a conflict between retained and immediate requirements, split the pass
			bool conflict = RequirementsConflict(
				p.resources.frameResourceRequirements,   // baseline retained for this frame
				immediateFrameData.requirements);
			if (conflict) {
				// Create new PassAndResources for the immediate requirements
				ComputePassAndResources immediatePassAndResources;
				immediatePassAndResources.pass = p.pass;
				immediatePassAndResources.resources.staticResourceRequirements = immediateFrameData.requirements;
				immediatePassAndResources.resources.frameResourceRequirements = immediateFrameData.requirements;
				immediatePassAndResources.resources.preferredQueueKind = p.resources.preferredQueueKind;
				immediatePassAndResources.resources.pinnedQueueSlot = p.resources.pinnedQueueSlot;
				immediatePassAndResources.immediateBytecode = std::move(immediateFrameData.bytecode);
				immediatePassAndResources.immediateKeepAlive = std::move(p.immediateKeepAlive);
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
				p.resources.frameResourceRequirements.insert(
					p.resources.frameResourceRequirements.end(),
					immediateFrameData.requirements.begin(),
					immediateFrameData.requirements.end());
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
				m_framePasses.push_back(pr);
			}
		}
		else if (pr.type == PassType::Render) {
			auto& p = std::get<RenderPassAndResources>(pr.pass);

			p.immediateBytecode.clear();
			p.resources.frameResourceRequirements = p.resources.staticResourceRequirements;

			ImmediateExecutionContext c{ device,
				{rg::imm::ImmediatePassKind::Render,
				m_immediateDispatch,
				&ResolveByIdThunk,
				&ResolveByPtrThunk,
				this},
				frameIndex,
				hostData
			};
			{
				ZoneScopedN("RenderGraph::CompileFrame::RecordImmediateCommands");
				if (!p.name.empty()) {
					ZoneText(p.name.data(), p.name.size());
				}
				if (traceLifecycle) {
					spdlog::info("RG frame {} render pass '{}' RecordImmediateCommands begin", frameIndex, p.name);
				}
				p.pass->RecordImmediateCommands(c);
				if (traceLifecycle) {
					spdlog::info("RG frame {} render pass '{}' RecordImmediateCommands complete", frameIndex, p.name);
				}
			}
			auto immediateFrameData = c.list.Finalize();

			bool conflict = RequirementsConflict(
				p.resources.frameResourceRequirements,   // baseline retained for this frame
				immediateFrameData.requirements);

			if (conflict) {
				// Create new PassAndResources for the immediate requirements
				RenderPassAndResources immediatePassAndResources;
				immediatePassAndResources.pass = p.pass;
				immediatePassAndResources.resources.staticResourceRequirements = immediateFrameData.requirements;
				immediatePassAndResources.resources.frameResourceRequirements = immediateFrameData.requirements;
				immediatePassAndResources.resources.preferredQueueKind = p.resources.preferredQueueKind;
				immediatePassAndResources.resources.pinnedQueueSlot = p.resources.pinnedQueueSlot;
				immediatePassAndResources.immediateBytecode = std::move(immediateFrameData.bytecode);
				immediatePassAndResources.immediateKeepAlive = std::move(p.immediateKeepAlive);
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
				p.resources.frameResourceRequirements.insert(
					p.resources.frameResourceRequirements.end(),
					immediateFrameData.requirements.begin(),
					immediateFrameData.requirements.end());
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
				m_framePasses.push_back(pr);
			}
		}
		else if (pr.type == PassType::Copy) {
			auto& p = std::get<CopyPassAndResources>(pr.pass);

			p.immediateBytecode.clear();
			p.resources.frameResourceRequirements = p.resources.staticResourceRequirements;

			ImmediateExecutionContext c{ device,
				{rg::imm::ImmediatePassKind::Copy,
				m_immediateDispatch,
				&ResolveByIdThunk,
				&ResolveByPtrThunk,
				this},
				frameIndex,
				hostData
			};

			{
				ZoneScopedN("RenderGraph::CompileFrame::RecordImmediateCommands");
				if (!p.name.empty()) {
					ZoneText(p.name.data(), p.name.size());
				}
				if (traceLifecycle) {
					spdlog::info("RG frame {} copy pass '{}' RecordImmediateCommands begin", frameIndex, p.name);
				}
				p.pass->RecordImmediateCommands(c);
				if (traceLifecycle) {
					spdlog::info("RG frame {} copy pass '{}' RecordImmediateCommands complete", frameIndex, p.name);
				}
			}
			auto immediateFrameData = c.list.Finalize();

			bool conflict = RequirementsConflict(
				p.resources.frameResourceRequirements,
				immediateFrameData.requirements);

			if (conflict) {
				CopyPassAndResources immediatePassAndResources;
				immediatePassAndResources.pass = p.pass;
				immediatePassAndResources.resources.staticResourceRequirements = immediateFrameData.requirements;
				immediatePassAndResources.resources.frameResourceRequirements = immediateFrameData.requirements;
				immediatePassAndResources.resources.preferredQueueKind = p.resources.preferredQueueKind;
				immediatePassAndResources.resources.pinnedQueueSlot = p.resources.pinnedQueueSlot;
				immediatePassAndResources.immediateBytecode = std::move(immediateFrameData.bytecode);
				immediatePassAndResources.immediateKeepAlive = std::move(p.immediateKeepAlive);
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
				p.resources.frameResourceRequirements.insert(
					p.resources.frameResourceRequirements.end(),
					immediateFrameData.requirements.begin(),
					immediateFrameData.requirements.end());
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
	std::vector<std::pair<std::string, std::string>> explicitAfterByName;
	explicitAfterByName.reserve(frameExt.size());

	if (!frameExt.empty()) {
		auto recordImmediateCommands = [&](AnyPassAndResources& pr) {
			if (pr.type == PassType::Compute) {
				auto& p = std::get<ComputePassAndResources>(pr.pass);
				p.immediateBytecode.clear();
				p.resources.frameResourceRequirements = p.resources.staticResourceRequirements;

				ImmediateExecutionContext c{ device,
					{rg::imm::ImmediatePassKind::Compute,
					m_immediateDispatch,
					&ResolveByIdThunk,
					&ResolveByPtrThunk,
					this},
					frameIndex,
					hostData
				};

				{
					ZoneScopedN("RenderGraph::CompileFrame::RecordImmediateCommands");
					if (!p.name.empty()) {
						ZoneText(p.name.data(), p.name.size());
					}
					if (traceLifecycle) {
						spdlog::info("RG frame {} compute pass '{}' RecordImmediateCommands begin", frameIndex, p.name);
					}
					p.pass->RecordImmediateCommands(c);
					if (traceLifecycle) {
						spdlog::info("RG frame {} compute pass '{}' RecordImmediateCommands complete", frameIndex, p.name);
					}
				}
				auto immediateFrameData = c.list.Finalize();
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				p.resources.frameResourceRequirements.insert(
					p.resources.frameResourceRequirements.end(),
					immediateFrameData.requirements.begin(),
					immediateFrameData.requirements.end());
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
			}
			else if (pr.type == PassType::Copy) {
				auto& p = std::get<CopyPassAndResources>(pr.pass);
				p.immediateBytecode.clear();
				p.resources.frameResourceRequirements = p.resources.staticResourceRequirements;

				ImmediateExecutionContext c{ device,
					{rg::imm::ImmediatePassKind::Copy,
					m_immediateDispatch,
					&ResolveByIdThunk,
					&ResolveByPtrThunk,
					this},
					frameIndex,
					hostData
				};

				{
					ZoneScopedN("RenderGraph::CompileFrame::RecordImmediateCommands");
					if (!p.name.empty()) {
						ZoneText(p.name.data(), p.name.size());
					}
					if (traceLifecycle) {
						spdlog::info("RG frame {} copy pass '{}' RecordImmediateCommands begin", frameIndex, p.name);
					}
					p.pass->RecordImmediateCommands(c);
					if (traceLifecycle) {
						spdlog::info("RG frame {} copy pass '{}' RecordImmediateCommands complete", frameIndex, p.name);
					}
				}
				auto immediateFrameData = c.list.Finalize();
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				p.resources.frameResourceRequirements.insert(
					p.resources.frameResourceRequirements.end(),
					immediateFrameData.requirements.begin(),
					immediateFrameData.requirements.end());
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
			}
			else {
				auto& p = std::get<RenderPassAndResources>(pr.pass);
				p.immediateBytecode.clear();
				p.resources.frameResourceRequirements = p.resources.staticResourceRequirements;

				ImmediateExecutionContext c{ device,
					{rg::imm::ImmediatePassKind::Render,
					m_immediateDispatch,
					&ResolveByIdThunk,
					&ResolveByPtrThunk,
					this},
					frameIndex,
					hostData
				};

				{
					ZoneScopedN("RenderGraph::CompileFrame::RecordImmediateCommands");
					if (!p.name.empty()) {
						ZoneText(p.name.data(), p.name.size());
					}
					if (traceLifecycle) {
						spdlog::info("RG frame {} render pass '{}' RecordImmediateCommands begin", frameIndex, p.name);
					}
					p.pass->RecordImmediateCommands(c);
					if (traceLifecycle) {
						spdlog::info("RG frame {} render pass '{}' RecordImmediateCommands complete", frameIndex, p.name);
					}
				}
				auto immediateFrameData = c.list.Finalize();
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				p.resources.frameResourceRequirements.insert(
					p.resources.frameResourceRequirements.end(),
					immediateFrameData.requirements.begin(),
					immediateFrameData.requirements.end());
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
			}
		};

		// Build name->index map for O(1) anchor lookup
		std::unordered_map<std::string, size_t> framePassIndexByName;
		framePassIndexByName.reserve(m_framePasses.size() + frameExt.size());
		for (size_t i = 0; i < m_framePasses.size(); ++i) {
			if (!m_framePasses[i].name.empty()) {
				framePassIndexByName[m_framePasses[i].name] = i;
			}
		}

		auto findPassIndexByName = [&](const std::string& name) -> std::optional<size_t> {
			if (name.empty()) return std::nullopt;
			auto it = framePassIndexByName.find(name);
			if (it != framePassIndexByName.end()) return it->second;
			return std::nullopt;
		};

		std::unordered_map<std::string, uint32_t> insertedAfterCount;
		insertedAfterCount.reserve(frameExt.size());

		for (auto& d : frameExt) {
			if (d.type == PassType::Unknown) continue;
			if (std::holds_alternative<std::monostate>(d.pass)) continue;
			if (d.name.empty()) {
				spdlog::warn("Frame extension emitted a pass with empty name; skipping.");
				continue;
			}

			AnyPassAndResources any = MaterializeExternalPass(d, true, false);
			recordImmediateCommands(any);

			// Default insertion: append
			size_t insertPos = m_framePasses.size();
			std::string anchorName;

			if (d.where.has_value()) {
				for (auto const& a : d.where->after) {
					auto idxOpt = findPassIndexByName(a);
					if (idxOpt.has_value()) {
						anchorName = a;
						uint32_t offset = insertedAfterCount[anchorName]++;
						insertPos = std::min(*idxOpt + 1 + (size_t)offset, m_framePasses.size());
						break;
					}
				}
			}

			if (!anchorName.empty()) {
				explicitAfterByName.push_back({ anchorName, any.name });
			}

			// Update indices in the map for entries at or after the insertion point
			for (auto& [name, idx] : framePassIndexByName) {
				if (idx >= insertPos) ++idx;
			}
			// Add the new pass to the map before inserting (insertPos is its index)
			if (!any.name.empty()) {
				framePassIndexByName[any.name] = insertPos;
			}

			m_framePasses.insert(m_framePasses.begin() + insertPos, std::move(any));
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
	{
		traceCompileStep("BuildNodes");
		ZoneScopedN("RenderGraph::CompileFrame::BuildNodes");
		nodes = BuildNodes(*this, m_framePasses);
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
	{
		traceCompileStep("BuildResourceAccessChainIR");
		ZoneScopedN("RenderGraph::CompileFrame::BuildResourceAccessChainIR");
		BuildResourceAccessChainIR(nodes, explicitEdges);
	}

	std::vector<rg::alias::AliasSchedulingNode> aliasNodes;
	aliasNodes.reserve(nodes.size());
	for (const auto& node : nodes) {
		aliasNodes.push_back(rg::alias::AliasSchedulingNode{
			.passIndex = node.passIndex,
			.originalOrder = node.originalOrder,
			.indegree = node.indegree,
			.criticality = node.criticality,
			.out = node.out,
		});
	}

	{
		traceCompileStep("AutoAssignAliasingPools");
		ZoneScopedN("RenderGraph::CompileFrame::AutoAssignAliasingPools");
		m_aliasingSubsystem.AutoAssignAliasingPools(*this, aliasNodes);
	}
	{
		traceCompileStep("BuildAliasPlanAfterDag");
		ZoneScopedN("RenderGraph::CompileFrame::BuildAliasPlanAfterDag");
		m_aliasingSubsystem.BuildAliasPlanAfterDag(*this, aliasNodes);
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
		traceCompileStep("MaterializeUnmaterializedResources");
		ZoneScopedN("RenderGraph::CompileFrame::MaterializeUnmaterializedResources");
		MaterializeUnmaterializedResources(&usedResourceIDs);
	}
	{
		traceCompileStep("SnapshotCompiledResourceGenerations");
		ZoneScopedN("RenderGraph::CompileFrame::SnapshotCompiledResourceGenerations");
		SnapshotCompiledResourceGenerations(usedResourceIDs);
	}
	{
		traceCompileStep("BuildFrameProgramIR");
		ZoneScopedN("RenderGraph::CompileFrame::BuildFrameProgramIR");
		BuildFrameProgramIR();
	}

	struct MaterializationSnapshot {
		std::vector<PassBatch> batches;
		std::unordered_map<uint64_t, SymbolicTracker> compileTrackers;
		std::unordered_set<uint64_t> aliasActivationPending;
		std::vector<unsigned int> frameQueueLastUsageBatch;
		std::vector<unsigned int> frameQueueLastProducerBatch;
		std::vector<unsigned int> frameQueueLastTransitionBatch;
		std::vector<std::vector<unsigned int>> frameTransitionPlacementBatchesByResource;
		std::vector<FrameResourceEventSummary> frameResourceEventSummaries;
		std::vector<FrameResourceEventRecord> frameResourceEventLog;
		std::unordered_map<uint64_t, std::vector<CompiledSegmentDesc::BoundaryStateEntry>> actualEntryBoundaryStatesBySegmentId;
		std::unordered_map<uint64_t, std::vector<CompiledSegmentDesc::BoundaryStateEntry>> actualExitBoundaryStatesBySegmentId;
		ScheduleIR compiledScheduleIR;
		BarrierIR compiledBarrierIR;
		std::vector<CompiledSegment> compiledSegments;
		std::unordered_map<uint64_t, PassIR> cachedPassIRByStableId;
		std::unordered_map<uint64_t, ScheduleIR> cachedScheduleIRByKey;
		std::unordered_map<uint64_t, ScheduleIR> cachedScheduleIRByRelaxedKey;
		std::unordered_map<uint64_t, ScheduleCacheKeyParts> cachedScheduleKeyPartsByKey;
		std::optional<ScheduleCacheKeyParts> lastScheduleCacheKeyParts;
		std::unordered_map<uint64_t, CompiledSegment> cachedBarrierSegments;
		std::vector<SegmentPlan> lastSegmentPlans;
		CompileCacheStats compileCacheStats;
		std::vector<CompileReuseEvent> compileReuseEvents;
		std::optional<std::pair<unsigned int, unsigned int>> activeLoweringBatchRange;
		std::string activeMaterializingSegmentName;
		std::string activeMaterializingSegmentMode;
		std::optional<std::pair<unsigned int, unsigned int>> activeMaterializingSegmentBatchRange;
		std::vector<std::unordered_map<uint64_t, unsigned int>> compiledLastProducerBatchByResourceByQueue;
		std::vector<UINT64> queueNextFenceValues;
	};

	auto captureMaterializationSnapshot = [&]() {
		MaterializationSnapshot snapshot{};
		snapshot.batches = batches;
		snapshot.compileTrackers = compileTrackers;
		snapshot.aliasActivationPending = aliasActivationPending;
		snapshot.frameQueueLastUsageBatch = m_frameQueueLastUsageBatch;
		snapshot.frameQueueLastProducerBatch = m_frameQueueLastProducerBatch;
		snapshot.frameQueueLastTransitionBatch = m_frameQueueLastTransitionBatch;
		snapshot.frameTransitionPlacementBatchesByResource = m_frameTransitionPlacementBatchesByResource;
		snapshot.frameResourceEventSummaries = m_frameResourceEventSummaries;
		snapshot.frameResourceEventLog = m_frameResourceEventLog;
		snapshot.actualEntryBoundaryStatesBySegmentId = m_actualEntryBoundaryStatesBySegmentId;
		snapshot.actualExitBoundaryStatesBySegmentId = m_actualExitBoundaryStatesBySegmentId;
		snapshot.compiledScheduleIR = m_compiledScheduleIR;
		snapshot.compiledBarrierIR = m_compiledBarrierIR;
		snapshot.compiledSegments = m_compiledSegments;
		snapshot.cachedPassIRByStableId = m_cachedPassIRByStableId;
		snapshot.cachedScheduleIRByKey = m_cachedScheduleIRByKey;
		snapshot.cachedScheduleIRByRelaxedKey = m_cachedScheduleIRByRelaxedKey;
		snapshot.cachedScheduleKeyPartsByKey = m_cachedScheduleKeyPartsByKey;
		snapshot.lastScheduleCacheKeyParts = m_lastScheduleCacheKeyParts;
		snapshot.cachedBarrierSegments = m_cachedBarrierSegments;
		snapshot.lastSegmentPlans = m_lastSegmentPlans;
		snapshot.compileCacheStats = m_compileCacheStats;
		snapshot.compileReuseEvents = m_compileReuseEvents;
		snapshot.activeLoweringBatchRange = m_activeLoweringBatchRange;
		snapshot.activeMaterializingSegmentName = m_activeMaterializingSegmentName;
		snapshot.activeMaterializingSegmentMode = m_activeMaterializingSegmentMode;
		snapshot.activeMaterializingSegmentBatchRange = m_activeMaterializingSegmentBatchRange;
		snapshot.compiledLastProducerBatchByResourceByQueue = m_compiledLastProducerBatchByResourceByQueue;
		snapshot.queueNextFenceValues.reserve(m_queueRegistry.SlotCount());
		for (size_t queueIndex = 0; queueIndex < m_queueRegistry.SlotCount(); ++queueIndex) {
			snapshot.queueNextFenceValues.push_back(
				m_queueRegistry.GetCurrentFenceValue(static_cast<QueueSlotIndex>(static_cast<uint8_t>(queueIndex))));
		}
		return snapshot;
	};

	auto restoreMaterializationSnapshot = [&](const MaterializationSnapshot& snapshot) {
		batches = snapshot.batches;
		compileTrackers = snapshot.compileTrackers;
		aliasActivationPending = snapshot.aliasActivationPending;
		m_frameQueueLastUsageBatch = snapshot.frameQueueLastUsageBatch;
		m_frameQueueLastProducerBatch = snapshot.frameQueueLastProducerBatch;
		m_frameQueueLastTransitionBatch = snapshot.frameQueueLastTransitionBatch;
		m_frameTransitionPlacementBatchesByResource = snapshot.frameTransitionPlacementBatchesByResource;
		m_frameResourceEventSummaries = snapshot.frameResourceEventSummaries;
		m_frameResourceEventLog = snapshot.frameResourceEventLog;
		m_actualEntryBoundaryStatesBySegmentId = snapshot.actualEntryBoundaryStatesBySegmentId;
		m_actualExitBoundaryStatesBySegmentId = snapshot.actualExitBoundaryStatesBySegmentId;
		m_compiledScheduleIR = snapshot.compiledScheduleIR;
		m_compiledBarrierIR = snapshot.compiledBarrierIR;
		m_compiledSegments = snapshot.compiledSegments;
		m_cachedPassIRByStableId = snapshot.cachedPassIRByStableId;
		m_cachedScheduleIRByKey = snapshot.cachedScheduleIRByKey;
		m_cachedScheduleIRByRelaxedKey = snapshot.cachedScheduleIRByRelaxedKey;
		m_cachedScheduleKeyPartsByKey = snapshot.cachedScheduleKeyPartsByKey;
		m_lastScheduleCacheKeyParts = snapshot.lastScheduleCacheKeyParts;
		m_cachedBarrierSegments = snapshot.cachedBarrierSegments;
		m_lastSegmentPlans = snapshot.lastSegmentPlans;
		m_compileCacheStats = snapshot.compileCacheStats;
		m_compileReuseEvents = snapshot.compileReuseEvents;
		m_activeLoweringBatchRange = snapshot.activeLoweringBatchRange;
		m_activeMaterializingSegmentName = snapshot.activeMaterializingSegmentName;
		m_activeMaterializingSegmentMode = snapshot.activeMaterializingSegmentMode;
		m_activeMaterializingSegmentBatchRange = snapshot.activeMaterializingSegmentBatchRange;
		m_compiledLastProducerBatchByResourceByQueue = snapshot.compiledLastProducerBatchByResourceByQueue;
		for (size_t queueIndex = 0; queueIndex < snapshot.queueNextFenceValues.size() && queueIndex < m_queueRegistry.SlotCount(); ++queueIndex) {
			m_queueRegistry.RestoreNextFenceValueForDiagnostics(
				static_cast<QueueSlotIndex>(static_cast<uint8_t>(queueIndex)),
				snapshot.queueNextFenceValues[queueIndex]);
		}
	};

	auto finalizeMaterializedBatches = [&](bool forceFullLowerForDiagnostics) {
		traceCompileStep("AutoScheduleAndBuildBatches");
		{
			ZoneScopedN("RenderGraph::CompileFrame::AutoScheduleAndBuildBatches");
			AutoScheduleAndBuildBatches(*this, m_framePasses, nodes, forceFullLowerForDiagnostics);
		}
		traceCompileStep("ApplyAliasQueueSynchronization");
		{
			ZoneScopedN("RenderGraph::CompileFrame::ApplyAliasQueueSynchronization");
			m_aliasingSubsystem.ApplyAliasQueueSynchronization(*this);
		}
		traceCompileStep("CaptureCompileTrackersForExecution");
		{
			ZoneScopedN("RenderGraph::CompileFrame::CaptureCompileTrackersForExecution");
			CaptureCompileTrackersForExecution(usedResourceIDs);
		}

		traceCompileStep("PlanCrossFrameQueueWaits");
		{
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

					auto itCurPlacement = aliasPlacementRangesByID.find(rid);
					if (itCurPlacement == aliasPlacementRangesByID.end()) {
						continue;
					}

					auto itPoolState = persistentAliasPools.find(itCurPlacement->second.poolID);
					if (itPoolState == persistentAliasPools.end()) {
						continue;
					}

					auto itPrevPool = m_lastAliasPlacementProducersByPoolAcrossFrames.find(itCurPlacement->second.poolID);
					if (itPrevPool == m_lastAliasPlacementProducersByPoolAcrossFrames.end()) {
						continue;
					}

					const uint64_t curStart = itCurPlacement->second.startByte;
					const uint64_t curEnd = itCurPlacement->second.endByte;
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
						for (auto const& req : passAndResources.resources.frameResourceRequirements) {
							accumulateCrossFrameWaitForHandle(passQueueSlot, req.resourceHandleAndRange.resource);
						}
						for (auto const& tr : passAndResources.resources.internalTransitions) {
							accumulateCrossFrameWaitForHandle(passQueueSlot, tr.first.resource);
						}
					}
				}, pr.pass);
			}
		}

		traceCompileStep("DeduplicateQueueWaits");
		{
			ZoneScopedN("RenderGraph::CompileFrame::DeduplicateQueueWaits");
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

		traceCompileStep("PruneUnusedQueueSignals");
		{
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
	};

	const bool captureCompileComparisonDumps = m_getRenderGraphCompileDumpEnabled && m_getRenderGraphCompileDumpEnabled();
	if (captureCompileComparisonDumps) {
		const MaterializationSnapshot preMaterializationSnapshot = captureMaterializationSnapshot();
		finalizeMaterializedBatches(true);
		const MaterializationSnapshot fullLowerSnapshot = captureMaterializationSnapshot();
		traceCompileStep("WriteCompiledGraphDebugDump.full_recompile");
		{
			ZoneScopedN("RenderGraph::CompileFrame::WriteCompiledGraphDebugDump.full_recompile");
			WriteCompiledGraphDebugDump(frameIndex, nodes, "full_recompile");
		}
		restoreMaterializationSnapshot(preMaterializationSnapshot);
		finalizeMaterializedBatches(false);
		auto validateCachedBatchesAgainstFullLower = [&]() {
			const auto& expectedBatches = fullLowerSnapshot.batches;
			const auto& replayedBatches = batches;
			auto fail = [&](std::string reason) {
				RecordCompileReuseEvent(
					"FinalBatchReplayValidation",
					"cached_segments",
					false,
					m_compiledScheduleIR.structureHash,
					reason);
				spdlog::warn("RenderGraph::CompileFrame final cached batch validation failed: {}", reason);
			};

			if (expectedBatches.size() != replayedBatches.size()) {
				std::ostringstream oss;
				oss << "batch count mismatch: full_lower=" << expectedBatches.size()
					<< " cached=" << replayedBatches.size();
				fail(oss.str());
				return;
			}

			for (size_t batchIndex = 0; batchIndex < expectedBatches.size(); ++batchIndex) {
				const auto& expected = expectedBatches[batchIndex];
				const auto& replayed = replayedBatches[batchIndex];
				if (expected.QueueCount() != replayed.QueueCount()) {
					std::ostringstream oss;
					oss << "queue count mismatch at batch " << batchIndex
						<< ": full_lower=" << expected.QueueCount()
						<< " cached=" << replayed.QueueCount();
					fail(oss.str());
					return;
				}

				for (size_t queueSlot = 0; queueSlot < expected.QueueCount(); ++queueSlot) {
					if (expected.Passes(queueSlot).size() != replayed.Passes(queueSlot).size()) {
						std::ostringstream oss;
						oss << "pass count mismatch at batch " << batchIndex
							<< " queue=" << queueSlot
							<< ": full_lower=" << expected.Passes(queueSlot).size()
							<< " cached=" << replayed.Passes(queueSlot).size();
						fail(oss.str());
						return;
					}

					for (size_t phaseIndex = 0; phaseIndex < PassBatch::kTransitionPhaseCount; ++phaseIndex) {
						const auto phase = static_cast<BatchTransitionPhase>(phaseIndex);
						if (expected.Transitions(queueSlot, phase).size() != replayed.Transitions(queueSlot, phase).size()) {
							std::ostringstream oss;
							oss << "transition count mismatch at batch " << batchIndex
								<< " queue=" << queueSlot
								<< " phase=" << phaseIndex
								<< ": full_lower=" << expected.Transitions(queueSlot, phase).size()
								<< " cached=" << replayed.Transitions(queueSlot, phase).size();
							fail(oss.str());
							return;
						}
					}

					for (size_t phaseIndex = 0; phaseIndex < PassBatch::kSignalPhaseCount; ++phaseIndex) {
						const auto phase = static_cast<BatchSignalPhase>(phaseIndex);
						const bool expectedSignal = expected.HasQueueSignal(phase, queueSlot);
						const bool replayedSignal = replayed.HasQueueSignal(phase, queueSlot);
						const UINT64 expectedFence = expected.GetQueueSignalFenceValue(phase, queueSlot);
						const UINT64 replayedFence = replayed.GetQueueSignalFenceValue(phase, queueSlot);
						if (expectedSignal != replayedSignal || expectedFence != replayedFence) {
							std::ostringstream oss;
							oss << "signal mismatch at batch " << batchIndex
								<< " queue=" << queueSlot
								<< " phase=" << phaseIndex
								<< ": full_lower=(" << expectedSignal << "," << expectedFence
								<< ") cached=(" << replayedSignal << "," << replayedFence << ")";
							fail(oss.str());
							return;
						}
					}

					for (size_t waitPhaseIndex = 0; waitPhaseIndex < PassBatch::kWaitPhaseCount; ++waitPhaseIndex) {
						const auto phase = static_cast<BatchWaitPhase>(waitPhaseIndex);
						for (size_t srcQueue = 0; srcQueue < expected.QueueCount(); ++srcQueue) {
							const bool expectedWait = expected.HasQueueWait(phase, queueSlot, srcQueue);
							const bool replayedWait = replayed.HasQueueWait(phase, queueSlot, srcQueue);
							const UINT64 expectedFence = expected.GetQueueWaitFenceValue(phase, queueSlot, srcQueue);
							const UINT64 replayedFence = replayed.GetQueueWaitFenceValue(phase, queueSlot, srcQueue);
							if (expectedWait != replayedWait || expectedFence != replayedFence) {
								std::ostringstream oss;
								oss << "wait mismatch at batch " << batchIndex
									<< " dst_queue=" << queueSlot
									<< " src_queue=" << srcQueue
									<< " phase=" << waitPhaseIndex
									<< ": full_lower=(" << expectedWait << "," << expectedFence
									<< ") cached=(" << replayedWait << "," << replayedFence << ")";
								fail(oss.str());
								return;
							}
						}
					}
				}

				if (expected.allResources != replayed.allResources) {
					std::ostringstream oss;
					oss << "allResources mismatch at batch " << batchIndex;
					fail(oss.str());
					return;
				}
				if (expected.internallyTransitionedResources != replayed.internallyTransitionedResources) {
					std::ostringstream oss;
					oss << "internallyTransitionedResources mismatch at batch " << batchIndex;
					fail(oss.str());
					return;
				}
			}

			RecordCompileReuseEvent(
				"FinalBatchReplayValidation",
				"cached_segments",
				true,
				m_compiledScheduleIR.structureHash,
				"cached final batches matched diagnostic full-lower final batches after alias sync and signal pruning");
		};
		validateCachedBatchesAgainstFullLower();
		traceCompileStep("WriteCompiledGraphDebugDump.cached_segments");
		{
			ZoneScopedN("RenderGraph::CompileFrame::WriteCompiledGraphDebugDump.cached_segments");
			WriteCompiledGraphDebugDump(frameIndex, nodes, "cached_segments");
		}
	}
	else {
		finalizeMaterializedBatches(false);
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

	for (auto const& req : pass.resources.frameResourceRequirements)
		processResource(req.resourceHandleAndRange.resource);

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

	for (auto const& req : pass.resources.frameResourceRequirements)
		processResource(req.resourceHandleAndRange.resource);

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

	for (auto const& req : pass.resources.frameResourceRequirements)
		processResource(req.resourceHandleAndRange.resource);

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
			for (auto const& req : p.resources.frameResourceRequirements) {
				collectFromHandle(req.resourceHandleAndRange.resource);
			}
			for (auto const& t : p.resources.internalTransitions) {
				collectFromHandle(t.first.resource);
			}
		}
		else if (pr.type == PassType::Render) {
			auto const& p = std::get<RenderPassAndResources>(pr.pass);
			for (auto const& req : p.resources.frameResourceRequirements) {
				collectFromHandle(req.resourceHandleAndRange.resource);
			}
			for (auto const& t : p.resources.internalTransitions) {
				collectFromHandle(t.first.resource);
			}
		}
		else if (pr.type == PassType::Copy) {
			auto const& p = std::get<CopyPassAndResources>(pr.pass);
			for (auto const& req : p.resources.frameResourceRequirements) {
				collectFromHandle(req.resourceHandleAndRange.resource);
			}
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
	m_lastSubmittedSignalValueByQueue.resize(qc, 0);
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
			CreateQueue(kind, queueName.c_str(), QueueAutoAssignmentPolicy::AllowAutomaticScheduling);
			++currentAutoQueueCount;
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
	result = device.CreateTimeline(m_frameStartSyncFence);

	if (m_readbackService) {
		m_readbackService->Initialize(m_readbackFence.Get());
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
	m_getRenderGraphDisableCaching = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetRenderGraphDisableCaching() : false;
	};
	m_getRenderGraphQueueSyncTraceEnabled = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetRenderGraphQueueSyncTraceEnabled() : false;
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
	if (resourcesByID.contains(resource->GetGlobalResourceID())) {
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
	resourcesByID[resource->GetGlobalResourceID()] = resource;
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
			resource->GetGlobalResourceID(),
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
			batch.Append(bg);
			if (transition.discard) {
				for (size_t i = textureStart; i < batch.textures.size(); ++i) {
					batch.textures[i].discard = true;
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
			if (!t.pResource) {
				spdlog::error(
					"RenderGraph::RecordTransitionBarriers: skipping transition with null resource pointer range={} layout:{}->{} access:{}->{} sync:{}->{} discard={}",
					FormatRangeSpec(t.range),
					rhi::helpers::ResourceLayoutToString(t.prevLayout),
					rhi::helpers::ResourceLayoutToString(t.newLayout),
					rhi::helpers::ResourceAccessMaskToString(t.prevAccessType),
					rhi::helpers::ResourceAccessMaskToString(t.newAccessType),
					rhi::helpers::ResourceSyncToString(t.prevSyncState),
					rhi::helpers::ResourceSyncToString(t.newSyncState),
					t.discard);
				continue;
			}

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
				batch.buffers.push_back(bb);
			}
		}

		if (!batch.Empty()) {
			commandList.Barriers(batch.View());
		}
	}

	// Signal external fences on the queue. Must be called AFTER the command list
	// containing the pass work has been flushed (submitted) so the signals fire
	// after the GPU work they depend on.
	void SignalExternalFences(
		rhi::Queue& queue,
		QueueKind queueKind,
		rhi::Timeline* slotFence,
		std::vector<PassReturn>& externalFences) {
		ZoneScopedN("RenderGraph::SignalExternalFences");
		if (externalFences.empty()) return;
		for (auto& fr : externalFences) {
			if (!fr.fence.has_value()) {
				spdlog::warn("Pass returned an external fence without a value. This should not happen.");
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
#if BUILD_TYPE == BUILD_TYPE_DEBUG
				// Detect external fences that accidentally use the queue's own fence
				// timeline. This would cause signal tracking to become stale,
				// leading to out-of-order signals. The pass must use a dedicated timeline.
				if (slotFence) {
					auto a = fr.fence.value().GetHandle();
					auto b = slotFence->GetHandle();
					if (a.index == b.index && a.generation == b.generation) {
						spdlog::error(
							"External fence from pass uses the queue's own fence timeline! "
							"This will cause signal tracking to become stale and produce "
							"out-of-order signals. The pass must use a dedicated timeline.");
						throw std::runtime_error("External fence aliases queue fence timeline!");
					}
				}
#endif
				spdlog::debug(
					"SignalExternalFences: queue={} signaling timeline(idx={}, gen={}) value={}",
					static_cast<int>(queueKind),
					fr.fence.value().GetHandle().index,
					fr.fence.value().GetHandle().generation,
					fr.fenceValue);
				queue.Signal({ fr.fence.value().GetHandle(), fr.fenceValue });
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
					if (passReturn.fence)
						args.outExternalFences.push_back(passReturn);
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
					if (passReturn.fence)
						sched.externalFences.push_back(passReturn);
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
	{
		ZoneScopedN("RenderGraph::Execute::ValidateCompiledResourceGenerations");
		ValidateCompiledResourceGenerations();
	}
	context.immediateDispatch = &m_immediateDispatch;

	const bool heavyDebug = m_getHeavyDebug ? m_getHeavyDebug() : false;
	const bool batchTraceEnabled = m_getRenderGraphBatchTraceEnabled ? m_getRenderGraphBatchTraceEnabled() : false;
	const bool queueSyncTraceEnabled = m_getRenderGraphQueueSyncTraceEnabled ? m_getRenderGraphQueueSyncTraceEnabled() : false;
	auto& manager = DeviceManager::GetInstance();
	const size_t slotCount = m_queueRegistry.SlotCount();
	if (batchTraceEnabled) {
		spdlog::info(
			"RenderGraph::Execute begin frame={} batches={} slotCount={} heavyDebug={} queueSyncTrace={}",
			static_cast<unsigned>(context.frameIndex),
			batches.size(),
			slotCount,
			heavyDebug,
			queueSyncTraceEnabled);
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

	std::vector<UINT64> submittedSignalWatermarkBySlot(slotCount, 0);
	if (m_lastSubmittedSignalValueByQueue.size() != slotCount) {
		m_lastSubmittedSignalValueByQueue.resize(slotCount, 0);
	}
	std::vector<std::vector<UINT64>> maxObservedWaitFenceByDstSrc(
		slotCount,
		std::vector<UINT64>(slotCount, 0));
	std::vector<std::vector<uint8_t>> observedWaitByDstSrc(
		slotCount,
		std::vector<uint8_t>(slotCount, 0));
	for (size_t qi = 0; qi < slotCount; ++qi) {
		const auto slotIndex = static_cast<QueueSlotIndex>(static_cast<uint8_t>(qi));
		const UINT64 completedFenceValue = SlotFence(qi).GetCompletedValue();
		UINT64 nextFenceValue = m_queueRegistry.GetCurrentFenceValue(slotIndex);
		if (completedFenceValue != UINT64_MAX) {
			const UINT64 minimumNextFenceValue = completedFenceValue + 1;
			if (nextFenceValue < minimumNextFenceValue) {
				m_queueRegistry.EnsureNextFenceValueAtLeast(slotIndex, minimumNextFenceValue);
				nextFenceValue = m_queueRegistry.GetCurrentFenceValue(slotIndex);
			}
		}
		submittedSignalWatermarkBySlot[qi] = std::max(
			m_lastSubmittedSignalValueByQueue[qi],
			completedFenceValue == UINT64_MAX ? UINT64(0) : completedFenceValue);
	}

	auto WaitOnSlot = [&](size_t dstSlot, size_t srcSlot, UINT64 absoluteFenceValue) {
		if (dstSlot == srcSlot) return;
		if (dstSlot < observedWaitByDstSrc.size() && srcSlot < observedWaitByDstSrc[dstSlot].size()) {
			observedWaitByDstSrc[dstSlot][srcSlot] = 1;
			maxObservedWaitFenceByDstSrc[dstSlot][srcSlot] =
				(std::max)(maxObservedWaitFenceByDstSrc[dstSlot][srcSlot], absoluteFenceValue);
		}
		const UINT64 submittedFenceValue =
			srcSlot < submittedSignalWatermarkBySlot.size() ? submittedSignalWatermarkBySlot[srcSlot] : 0;
		const UINT64 completedFenceValue = SlotFence(srcSlot).GetCompletedValue();
		if (queueSyncTraceEnabled) {
			spdlog::info(
				"RenderGraph::Execute frame={} queue-wait dstSlot={} srcSlot={} fence={} submittedSignalWatermark={} completedFence={}",
				static_cast<unsigned>(context.frameIndex),
				dstSlot,
				srcSlot,
				absoluteFenceValue,
				submittedFenceValue,
				completedFenceValue);
		}
		if (submittedFenceValue < absoluteFenceValue) {
			spdlog::warn(
				"RenderGraph::Execute frame={} queue-wait dstSlot={} srcSlot={} fence={} exceeds highest successfully submitted signal {} on the source timeline (completed={})",
				static_cast<unsigned>(context.frameIndex),
				dstSlot,
				srcSlot,
				absoluteFenceValue,
				submittedFenceValue,
				completedFenceValue);
		}
		auto dstQ = SlotQueue(dstSlot);
		dstQ.Wait({ SlotFence(srcSlot).GetHandle(), absoluteFenceValue });
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

		auto itPlacement = aliasPlacementRangesByID.find(resourceID);
		if (itPlacement == aliasPlacementRangesByID.end()) {
			return;
		}

		auto itPoolState = persistentAliasPools.find(itPlacement->second.poolID);
		if (itPoolState == persistentAliasPools.end()) {
			return;
		}

		nextLastAliasPlacementProducersByPoolAcrossFrames[itPlacement->second.poolID].push_back(
			LastAliasPlacementProducerAcrossFrames{
				.resourceID = resourceID,
				.poolID = itPlacement->second.poolID,
				.poolGeneration = itPoolState->second.generation,
				.startByte = itPlacement->second.startByte,
				.endByte = itPlacement->second.endByte,
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
	for (size_t qi = 0; qi < slotCount; ++qi) {
		const auto slotIndex = static_cast<QueueSlotIndex>(static_cast<uint8_t>(qi));
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
	for (size_t qi = 0; qi < slotCount; ++qi) {
		submittedSignalWatermarkBySlot[qi] = std::max(
			m_lastSubmittedSignalValueByQueue[qi],
			SlotFence(qi).GetCompletedValue() == UINT64_MAX ? UINT64(0) : SlotFence(qi).GetCompletedValue());
	}
	if (queueSyncTraceEnabled) {
		for (size_t qi = 0; qi < slotCount; ++qi) {
			spdlog::info(
				"RenderGraph::Execute frame={} slot={} queue={} initialFenceState completed={} nextFence={} submittedSignalWatermark={}",
				static_cast<unsigned>(context.frameIndex),
				qi,
				QueueKindToString(m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(qi)))),
				SlotFence(qi).GetCompletedValue(),
				m_queueRegistry.GetCurrentFenceValue(static_cast<QueueSlotIndex>(static_cast<uint8_t>(qi))),
				submittedSignalWatermarkBySlot[qi]);
		}
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
					.lastSignaledOnTimeline = lastSignaledPerSlot[qi],
					.batchTraceEnabled = batchTraceEnabled,
				};
				ExecuteQueueBatch(args, WaitOnSlot);
				submittedSignalWatermarkBySlot[qi] = std::max(submittedSignalWatermarkBySlot[qi], lastSignaledPerSlot[qi]);
				m_lastSubmittedSignalValueByQueue[qi] = std::max(m_lastSubmittedSignalValueByQueue[qi], lastSignaledPerSlot[qi]);
			}

			// Signal external fences AFTER all CLs in this batch are submitted.
			for (size_t qi = 0; qi < slotCount; ++qi) {
				if (!slotExternalFences[qi].empty()) {
					auto rhiQ = SlotQueue(qi);
					SignalExternalFences(rhiQ, m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(qi)),
						&SlotFence(qi), slotExternalFences[qi]);
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
					signalValue = lastSignaledPerSlot[queueIndex] + 1;
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

				SignalQueueFenceOrThrow(
					rhiQueue,
					fenceTimeline,
					signalValue,
					queueKind,
					queueIndex,
					batchIndex,
					reason,
					static_cast<unsigned>(context.frameIndex));
				lastSignaledPerSlot[queueIndex] = std::max(lastSignaledPerSlot[queueIndex], signalValue);
				submittedSignalWatermarkBySlot[queueIndex] = std::max(submittedSignalWatermarkBySlot[queueIndex], signalValue);
				m_lastSubmittedSignalValueByQueue[queueIndex] = std::max(m_lastSubmittedSignalValueByQueue[queueIndex], signalValue);
				const auto slotIndex = static_cast<QueueSlotIndex>(static_cast<uint8_t>(queueIndex));
				if (signalValue != UINT64_MAX) {
					m_queueRegistry.EnsureNextFenceValueAtLeast(slotIndex, signalValue + 1);
				}
				if (queueSyncTraceEnabled) {
					spdlog::info(
						"RenderGraph::Execute frame={} queue-signal queue={} slot={} batch={} reason={} fence={} completed={} nextFence={}",
						static_cast<unsigned>(context.frameIndex),
						QueueKindToString(queueKind),
						queueIndex,
						batchIndex,
						reason,
						signalValue,
						fenceTimeline.GetCompletedValue(),
						m_queueRegistry.GetCurrentFenceValue(slotIndex));
				}

				if (pool) {
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
					externalFences);
			};

			auto queueRecordedCommandList = [&](size_t queueIndex, CommandListPair&& pair) {
				ZoneScopedN("RenderGraph::Execute::ParallelPath::QueueRecordedCommandList");
				auto& pending = pendingSubmissions[queueIndex];
				pending.pendingCommandLists.push_back(pair.list.Get());
				pending.pendingPairs.push_back(std::move(pair));
			};

			auto applyBatchWaitPhase = [&](const PassBatch& batch, size_t queueIndex, BatchWaitPhase waitPhase) {
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

					applyBatchWaitPhase(batch, qi, BatchWaitPhase::BeforeTransitions);

					if (qs.splitAfterTransitions) {
						queueRecordedCommandList(qi, std::move(qs.preallocatedCLs[clIndex]));
						signalAndRecycleQueue(
							qi,
							bi,
							batch.GetQueueSignalFenceValue(BatchSignalPhase::AfterTransitions, qi),
							"AfterTransitions");
						++clIndex;
					}

					applyBatchWaitPhase(batch, qi, BatchWaitPhase::BeforeExecution);

					if (qs.splitAfterExecution) {
						queueRecordedCommandList(qi, std::move(qs.preallocatedCLs[clIndex]));
						signalAndRecycleQueue(
							qi,
							bi,
							batch.GetQueueSignalFenceValue(BatchSignalPhase::AfterExecution, qi),
							"AfterExecution");
						++clIndex;
					}

					applyBatchWaitPhase(batch, qi, BatchWaitPhase::BeforeAfterPasses);
					queueRecordedCommandList(qi, std::move(qs.preallocatedCLs[clIndex]));

					if (qs.signalAfterCompletion) {
						signalAndRecycleQueue(
							qi,
							bi,
							batch.GetQueueSignalFenceValue(BatchSignalPhase::AfterCompletion, qi),
							"AfterCompletion");
					}

					flushExternalFencesForQueue(qi, bi, qs.externalFences);
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
				signalAndRecycleQueue(qi, batches.size(), lastSignaledPerSlot[qi] + 1, "EndOfFrameRecycle");
			}
			if (batchTraceEnabled) {
				spdlog::info("RenderGraph::Execute frame={} submit-all-batches complete", static_cast<unsigned>(context.frameIndex));
			}
		}
	}

	{
		ZoneScopedN("RenderGraph::Execute::ValidateRuntimeQueueSync");
		for (size_t dstIndex = 0; dstIndex < slotCount; ++dstIndex) {
			for (size_t srcIndex = 0; srcIndex < slotCount; ++srcIndex) {
				if (dstIndex == srcIndex || !observedWaitByDstSrc[dstIndex][srcIndex]) {
					continue;
				}

				const UINT64 waitedFenceValue = maxObservedWaitFenceByDstSrc[dstIndex][srcIndex];
				const UINT64 submittedFenceValue = submittedSignalWatermarkBySlot[srcIndex];
				if (waitedFenceValue > submittedFenceValue) {
					spdlog::error(
						"RenderGraph::Execute frame={} queue sync deadlock risk: dstSlot={} waited on srcSlot={} fence={}, but the highest successfully submitted signal on the source timeline was {} (completed={} nextFence={})",
						static_cast<unsigned>(context.frameIndex),
						dstIndex,
						srcIndex,
						waitedFenceValue,
						submittedFenceValue,
						SlotFence(srcIndex).GetCompletedValue(),
						m_queueRegistry.GetCurrentFenceValue(static_cast<QueueSlotIndex>(static_cast<uint8_t>(srcIndex))));
				}
			}
		}
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

	m_lastProducerByResourceAcrossFrames = std::move(nextLastProducerByResourceAcrossFrames);
	m_lastAliasPlacementProducersByPoolAcrossFrames = std::move(nextLastAliasPlacementProducersByPoolAcrossFrames);
	{
		ZoneScopedN("RenderGraph::Execute::ProcessDeletions");
		DeletionManager::GetInstance().ProcessDeletions();
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
		const UINT64 graphicsSignal = crm->Flush(QueueKind::Graphics, { false, 0 });
		const UINT64 computeSignal = crm->Flush(QueueKind::Compute, { false, 0 });
		const UINT64 copySignal = crm->Flush(QueueKind::Copy, { false, 0 });
		const std::array<UINT64, static_cast<size_t>(QueueKind::Count)> crmSignals{
			graphicsSignal,
			computeSignal,
			copySignal,
		};
		for (size_t qi = 0; qi < std::min(slotCount, crmSignals.size()); ++qi) {
			if (crmSignals[qi] == 0) {
				continue;
			}
			m_lastSubmittedSignalValueByQueue[qi] = std::max(m_lastSubmittedSignalValueByQueue[qi], crmSignals[qi]);
		}
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
		for (size_t equivalentResourceIndex : summaryEntry.equivalentResourceIndices) {
			if (batchBuildState.ContainsResource(equivalentResourceIndex)) {
				return true;
			}
		}
		return false;
	};

	auto overlapsAliasedTransitionInBatch = [&](const auto& summaryEntry) {
		for (size_t equivalentResourceIndex : summaryEntry.equivalentResourceIndices) {
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
		// Alias activations are emitted in BeforePasses of the consuming batch.
		// Only reject same-batch merging when that activation would clobber an
		// aliased-equivalent resource that is already live in the batch.
		if (aliasActivationPending.find(requirement.resourceID) != aliasActivationPending.end()
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
		Resource* trackedResource = requirement.resource.IsEphemeral()
			? requirement.resource.GetEphemeralPtr()
			: _registry.Resolve(requirement.resource);
		const bool hasLayout = trackedResource && trackedResource->HasLayout();

		// Changing state?
		if (requirement.resourceIndex < passBatchTrackersByResourceIndex.size()) {
			SymbolicTracker* tracker = passBatchTrackersByResourceIndex[requirement.resourceIndex];
			if (tracker && tracker->WouldModify(requirement.range, wantState, hasLayout)) {
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
			std::string_view name = key.ToString();
			throw std::runtime_error("Resource provider already registered for key: " + std::string(name));
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
		throw std::runtime_error("Failed to create queue");
	}
	// Determine instance number: count existing slots of this kind.
	uint8_t instance = 0;
	for (size_t i = 0; i < m_queueRegistry.SlotCount(); ++i) {
		if (m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(i))) == kind)
			++instance;
	}
	return m_queueRegistry.Register({ kind, instance }, queue, device, autoAssignmentPolicy);
}

void RenderGraph::SetMinimumAutomaticSchedulingQueues(QueueKind kind, uint8_t count) {
	const size_t kindIndex = static_cast<size_t>(kind);
	const uint8_t clampedCount = kind == QueueKind::Graphics ? (std::max)(uint8_t(1), count) : (std::max)(uint8_t(1), count);
	m_minAutomaticSchedulingQueuesByKind[kindIndex] = clampedCount;

	if (m_queueRegistry.SlotCount() > 0) {
		EnsureMinimumAutomaticSchedulingQueues();
	}
}
