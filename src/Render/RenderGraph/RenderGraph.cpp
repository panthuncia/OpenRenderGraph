#include "RenderGraphInternal.h"
#include "Render/Runtime/RuntimeDevice.h"

namespace org {

using namespace graph_internal;

namespace {
thread_local ResolverCaptureScope* t_resolverCaptures = nullptr;
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


RenderGraph::PassView RenderGraph::GetPassView(const AnyPassAndResources& pr) {
	PassView v{};
	if (pr.type == PassType::Render) {
		const auto& p = std::get<RenderPassAndResources>(pr.pass);
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
	runtime::ShutdownRuntimeDevice();
}

void RenderGraph::ShutdownOwnedState() {
	ShutdownTaskWorkers();
    m_resolverCaptureContext.reset();
	m_masterPassList.clear();
	m_compilerState->frameExtensions.clear();
	renderPassesByName.clear();
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
