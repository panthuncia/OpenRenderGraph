// Persistent execution, structural builds: representations, inline and worker-side builds, and alias placement.
#include "RenderGraphPersistentState.h"
#include <rhi_interop_dx12.h>
#include <array>
#include <bit>
#include <functional>
#include <map>
#include <mutex>
#include <tuple>
#include <unordered_set>

namespace org {

namespace {
// Publishes a D3D12 resource and its Vulkan counterpart as a resource's representations on both devices, and points its
// descriptors on each device at them.
void PublishRepresentationPair(Resource& concrete, const DeviceRegistryEntry& d3d12, const DeviceRegistryEntry& vulkan,
	InteropResourcePair pair, bool texture, Resource::APIRepresentationPlacement placement) {
	// Textures start undefined on both APIs; buffers in the bridge-compatible COMMON state.
	const ResourceState initial = texture
		? ResourceState{rhi::ResourceAccessType::None, rhi::ResourceLayout::Undefined, rhi::ResourceSyncState::None}
		: ResourceState{rhi::ResourceAccessType::Common, rhi::ResourceLayout::Common, rhi::ResourceSyncState::All};
	std::vector<Resource::PendingAPIRepresentation> representations;
	representations.push_back({d3d12.id, std::move(pair.canonical), initial, placement.heap});
	representations.push_back({vulkan.id, std::move(pair.imported), initial, placement.heap});
	if (auto* indexed = dynamic_cast<GloballyIndexedResource*>(&concrete)) indexed->RotateDescriptorSlotsForPublication();
	if (!concrete.PublishAPIRepresentations(std::move(representations), std::move(placement)))
		throw std::runtime_error("Failed to publish the shared representations of '" + concrete.GetName() + "'");
	concrete.RefreshAPIRepresentationDescriptors(d3d12.id);
	concrete.RefreshAPIRepresentationDescriptors(vulkan.id);
}

size_t DescriptionKey(const rhi::ResourceDesc& desc) {
	size_t key = 0;
	boost::hash_combine(key, static_cast<uint32_t>(desc.type));
	boost::hash_combine(key, desc.buffer.sizeBytes);
	boost::hash_combine(key, static_cast<uint32_t>(desc.texture.format));
	boost::hash_combine(key, desc.texture.width);
	boost::hash_combine(key, desc.texture.height);
	boost::hash_combine(key, desc.texture.depthOrLayers);
	boost::hash_combine(key, desc.texture.mipLevels);
	boost::hash_combine(key, desc.texture.sampleCount);
	boost::hash_combine(key, static_cast<uint32_t>(desc.resourceFlags));
	return key;
}
}

void RenderGraph::EnsurePersistentRepresentations(Resource& resource) {
	auto* concrete = UnwrapDynamic(&resource);
	if (!concrete) return;
	DeviceRegistryEntry* d3d12 = nullptr;
	DeviceRegistryEntry* vulkan = nullptr;
	for (auto& entry : m_backendDevices) {
		if (entry.backend == rhi::Backend::D3D12 && !d3d12) d3d12 = &entry;
		if (entry.backend == rhi::Backend::Vulkan && !vulkan) vulkan = &entry;
	}
	if (!d3d12 || !vulkan) throw std::runtime_error("Multi-device resource '" + concrete->GetName() + "' needs one D3D12 and one Vulkan device");
	if (concrete->HasAPIRepresentation(d3d12->id) && concrete->HasAPIRepresentation(vulkan->id)) return;
	if (!concrete->IsRenderGraphManaged())
		throw std::runtime_error("Multi-device resource '" + concrete->GetName() + "' is externally managed");
	rhi::ResourceDesc desc{};
	if (!concrete->TryGetRHIResourceDesc(desc) || desc.heapType != rhi::HeapType::DeviceLocal
		|| (desc.type != rhi::ResourceType::Buffer && desc.type != rhi::ResourceType::Texture2D))
		throw std::runtime_error("Multi-device resource '" + concrete->GetName() + "' is not a device-local buffer or 2D texture");
	if (desc.type == rhi::ResourceType::Texture2D && desc.texture.sampleCount > 1)
		throw std::runtime_error("Multi-device multisampled texture '" + concrete->GetName() + "' is unsupported");
	desc.heapFlags |= rhi::HeapFlags::Shared;
	if (desc.type == rhi::ResourceType::Texture2D && !rhi::vulkan::query_d3d12_texture_support(vulkan->device, desc).supported)
		throw std::runtime_error("Multi-device texture '" + concrete->GetName() + "' is unsupported for its format and usage");
	InteropResourcePair pair;
	if (rhi::Failed(InteropAllocator::CreateCommittedD3D12Vulkan(*d3d12, *vulkan, desc, pair)))
		throw std::runtime_error("Failed to create the shared representations of '" + concrete->GetName() + "'");
	// D3D12 requires an RT or DS texture it did not zero to be initialized by D3D12 before any use, even when Vulkan produces
	// it first: one enhanced discard into RENDER_TARGET, an empty pass, and back to COMMON, waited for here.
	if (desc.type == rhi::ResourceType::Texture2D && (desc.resourceFlags & rhi::ResourceFlags::RF_AllowDepthStencil) != 0)
		throw std::runtime_error("Multi-device depth-stencil texture '" + concrete->GetName() + "' is unsupported");
	if (desc.type == rhi::ResourceType::Texture2D && (desc.resourceFlags & rhi::ResourceFlags::RF_AllowRenderTarget) != 0) {
		rhi::CommandAllocatorPtr allocator;
		rhi::CommandListPtr list;
		rhi::DescriptorHeapPtr rtvHeap;
		rhi::TimelinePtr completion;
		auto result = d3d12->device.CreateCommandAllocator(rhi::QueueKind::Graphics, allocator);
		if (rhi::IsOk(result)) result = d3d12->device.CreateCommandList(rhi::QueueKind::Graphics, allocator.Get(), list);
		if (rhi::IsOk(result)) {
			rhi::DescriptorHeapDesc heapDesc{};
			heapDesc.type = rhi::DescriptorHeapType::RTV;
			heapDesc.capacity = 1;
			heapDesc.debugName = "ORG multi-device texture initialization RTV";
			result = d3d12->device.CreateDescriptorHeap(heapDesc, rtvHeap);
		}
		if (rhi::IsOk(result)) result = d3d12->device.CreateRenderTargetView({rtvHeap->GetHandle(), 0}, pair.canonical->GetHandle(), {});
		if (rhi::IsOk(result)) result = d3d12->device.CreateTimeline(completion, 0, "ORG multi-device texture initialization");
		if (rhi::IsOk(result)) {
			rhi::TextureBarrier initialize{
				.texture = pair.canonical->GetHandle(),
				.range = {.baseMip = 0, .mipCount = desc.texture.mipLevels, .baseLayer = 0, .layerCount = desc.texture.depthOrLayers,
					.basePlane = 0, .planeCount = 1},
				.beforeSync = rhi::ResourceSyncState::None,
				.afterSync = rhi::ResourceSyncState::All,
				.beforeAccess = rhi::ResourceAccessType::None,
				.afterAccess = rhi::ResourceAccessType::RenderTarget,
				.beforeLayout = rhi::ResourceLayout::Undefined,
				.afterLayout = rhi::ResourceLayout::RenderTarget,
				.discard = true,
			};
			list->Barriers(rhi::BarrierBatch{.textures = {&initialize, 1}});
			rhi::ColorAttachment attachment{};
			attachment.rtv = {rtvHeap->GetHandle(), 0};
			attachment.loadOp = rhi::LoadOp::DontCare;
			attachment.resource = pair.canonical->GetHandle();
			attachment.mipSlice = -1;
			rhi::PassBeginInfo begin{};
			begin.colors = {&attachment, 1};
			begin.width = desc.texture.width;
			begin.height = desc.texture.height;
			list->BeginPass(begin);
			list->EndPass();
			auto toCommon = initialize;
			toCommon.beforeSync = rhi::ResourceSyncState::RenderTarget;
			toCommon.beforeAccess = rhi::ResourceAccessType::RenderTarget;
			toCommon.afterAccess = rhi::ResourceAccessType::Common;
			toCommon.beforeLayout = rhi::ResourceLayout::RenderTarget;
			toCommon.afterLayout = rhi::ResourceLayout::Common;
			toCommon.discard = false;
			list->Barriers(rhi::BarrierBatch{.textures = {&toCommon, 1}});
			list->End();
			const rhi::CommandList submitted[] = {list.Get()};
			const rhi::TimelinePoint completed{completion->GetHandle(), 1};
			result = d3d12->device.GetQueue(rhi::QueueKind::Graphics).Submit(submitted, {.signals = {&completed, 1}});
			if (rhi::IsOk(result)) result = completion->HostWait(1, 30000);
		}
		if (rhi::Failed(result)) throw std::runtime_error("Failed to initialize the D3D12 representation of '" + concrete->GetName() + "'");
	}
	const bool texture = desc.type == rhi::ResourceType::Texture2D;
	PublishRepresentationPair(*concrete, *d3d12, *vulkan, std::move(pair), texture, Resource::APIRepresentationPlacement{});
	basic_telemetry::AddCounter("ORG.Persistent.MultiDeviceRepresentations");
	spdlog::info("Persistent multi-device representations: '{}' ({})", concrete->GetName(), texture ? "texture" : "buffer");
}

void RenderGraph::PlaceSharedRepresentations(Resource& concrete, const std::shared_ptr<const AliasHeapGeneration>& heap,
	uint64_t poolID, uint64_t offset, uint64_t size) {
	const auto* shared = heap ? heap->Shared() : nullptr;
	if (!shared) throw std::logic_error("Shared placement of '" + concrete.GetName() + "' without a shared alias heap");
	const DeviceRegistryEntry* d3d12 = m_backendDevices.Find(static_cast<DeviceInstanceId>(shared->d3d12Device));
	const DeviceRegistryEntry* vulkan = m_backendDevices.Find(static_cast<DeviceInstanceId>(shared->vulkanDevice));
	if (!d3d12 || !vulkan) throw std::runtime_error("Shared placement of '" + concrete.GetName() + "': a device of its pool is gone");
	rhi::ResourceDesc desc{};
	if (!concrete.TryGetRHIResourceDesc(desc)) throw std::runtime_error("Shared placement of '" + concrete.GetName() + "' has no description");
	InteropResourcePair pair;
	auto d3d12Device = d3d12->device;
	auto vulkanDevice = vulkan->device;
	if (rhi::Failed(d3d12Device.CreatePlacedResource(shared->d3d12->GetHandle(), offset, desc, pair.canonical))
		|| rhi::Failed(vulkanDevice.CreatePlacedResource(shared->vulkan->GetHandle(), offset, desc, pair.imported)))
		throw std::runtime_error("Failed to place '" + concrete.GetName() + "' in a shared alias pool");
	PublishRepresentationPair(concrete, *d3d12, *vulkan, std::move(pair), desc.type == rhi::ResourceType::Texture2D,
		{heap, poolID, offset, size});
	basic_telemetry::AddCounter("ORG.Persistent.SharedAliasPlacements");
	spdlog::info("Persistent shared alias placement: '{}' pool={:#x} offset={} size={}", concrete.GetName(), poolID, offset, size);
}

bool RenderGraph::IsSharedAliasEligible(Resource* concrete) const {
	if (!concrete || !m_compilerState->persistent) return false;
	const DeviceRegistryEntry* vulkan = nullptr;
	bool hasD3D12 = false;
	for (const auto& entry : m_backendDevices) {
		hasD3D12 = hasD3D12 || entry.backend == rhi::Backend::D3D12;
		if (entry.backend == rhi::Backend::Vulkan && !vulkan) vulkan = &entry;
	}
	if (!hasD3D12 || !vulkan) return false;
	rhi::ResourceDesc desc{};
	if (!concrete->TryGetRHIResourceDesc(desc) || desc.heapType != rhi::HeapType::DeviceLocal) return false;
	auto& cache = m_compilerState->persistent->sharedAliasEligibility;
	const auto key = DescriptionKey(desc);
	if (const auto found = cache.find(concrete); found != cache.end() && found->second.first == key) return found->second.second;
	bool eligible = false;
	if (desc.type == rhi::ResourceType::Buffer) eligible = true;
	else if (desc.type == rhi::ResourceType::Texture2D && desc.texture.sampleCount <= 1
		// D3D12 requires a placed render target or depth texture to be initialized by D3D12 after each occupant switch, which a
		// first use on the Vulkan device does not do: those stay committed.
		&& (desc.resourceFlags & (rhi::ResourceFlags::RF_AllowRenderTarget | rhi::ResourceFlags::RF_AllowDepthStencil)) == 0)
		eligible = rhi::vulkan::query_d3d12_heap_texture_support(vulkan->device, desc).supported;
	cache[concrete] = {key, eligible};
	return eligible;
}

rhi::ResourceAllocationInfo RenderGraph::SharedAliasAllocationInfo(Resource* concrete) const {
	rhi::ResourceAllocationInfo result{};
	result.alignment = 64ull * 1024ull; // the placement granularity of a D3D12 heap
	rhi::ResourceDesc desc{};
	if (!concrete || !concrete->TryGetRHIResourceDesc(desc)) return result;
	for (const auto& entry : m_backendDevices) {
		if (entry.backend != rhi::Backend::D3D12 && entry.backend != rhi::Backend::Vulkan) continue;
		rhi::ResourceAllocationInfo info{};
		entry.device.GetResourceAllocationInfo(&desc, 1, &info);
		result.sizeInBytes = (std::max)(result.sizeInBytes, info.sizeInBytes);
		result.alignment = (std::max)(result.alignment, info.alignment);
	}
	result.sizeInBytes = (result.sizeInBytes + result.alignment - 1) / result.alignment * result.alignment;
	return result;
}

void RenderGraph::MaterializePersistentStandalone(std::span<Resource* const> resources, bool skipAliasCandidates) {
	for (auto* resource : resources) {
		auto* concrete = UnwrapDynamic(resource);
		auto* backed = concrete ? dynamic_cast<BackedResource*>(concrete) : nullptr;
		if (!backed || backed->IsMaterialized()) continue;
		if (skipAliasCandidates && IsPersistentAliasCandidate(concrete)) continue;
		if (auto* texture = dynamic_cast<PixelBuffer*>(concrete)) texture->Materialize();
		else if (auto* buffer = dynamic_cast<BufferBase*>(concrete)) buffer->Materialize();
		else continue;
		if (m_compilerState && m_compilerState->persistent) m_compilerState->persistent->ownMaterialized.insert(concrete);
		TrackTransientFrameResource(concrete);
		basic_telemetry::AddCounter("ORG.Persistent.StandaloneMaterializations");
	}
}

bool RenderGraph::IsPersistentAliasCandidate(Resource* concrete) const {
	if (!concrete) return false;
	// A resource a pass on another device uses is placed only in a shared pool, and only when it qualifies for one.
	if (m_compilerState->persistent && m_compilerState->persistent->multiDeviceResources.contains(concrete)
		&& !IsSharedAliasEligible(concrete)) return false;
	const auto mode = m_getAutoAliasMode ? m_getAutoAliasMode() : AutoAliasMode::Off;
	if (auto* texture = dynamic_cast<PixelBuffer*>(concrete)) {
		const auto& desc = texture->GetDescription();
		return desc.allowAlias && (mode != AutoAliasMode::Off || desc.aliasingPoolID.has_value());
	}
	if (auto* buffer = dynamic_cast<BufferBase*>(concrete))
		return buffer->IsAliasingAllowed() && buffer->GetAccessType() == rhi::HeapType::DeviceLocal
			&& (mode != AutoAliasMode::Off || buffer->GetAliasingPoolHint().has_value());
	return false;
}

// Binds every declared, unbound slot that is not a versioned slot before a structural build.
// Direct alias candidates stay unbound so the planner can place them, unless
// several unbound slots share one concrete backing: unbound slots carry unique
// placeholder identities and would not canonicalize, so the schedule would miss
// their dependencies. Those (and group members) are realized standalone here;
// the planner may still re-place a standalone candidate afterwards.
void RenderGraph::BindUnboundPersistentEntries(persistent::GraphEditTransaction& edit) {
	auto& state = *m_compilerState->persistent;
	std::unordered_map<Resource*, uint32_t> unboundSlotsByConcrete;
	for (const auto& entry : state.entries)
		if (entry.resource && !entry.versioned && !entry.isBound) ++unboundSlotsByConcrete[UnwrapDynamic(entry.resource)];
	for (uint32_t index = 0; index < state.entries.size(); ++index) {
		auto& entry = state.entries[index];
		if (!entry.resource || entry.versioned || entry.isBound) continue;
		auto* concrete = UnwrapDynamic(entry.resource);
		auto* backed = concrete ? dynamic_cast<BackedResource*>(concrete) : nullptr;
		if (backed && !backed->IsMaterialized()) {
			if (entry.direct && unboundSlotsByConcrete[concrete] <= 1 && IsPersistentAliasCandidate(concrete)) continue;
			MaterializePersistentStandalone(std::span<Resource* const>{&entry.resource, 1}, false);
		}
		BindEntry(state, edit, index);
	}
}

std::shared_ptr<const persistent::SelectedPublication> RenderGraph::BuildPersistentStructural(persistent::GraphEditTransaction& edit) {
	auto& compiler = *m_compilerState;
	auto& state = *compiler.persistent;
	BindUnboundPersistentEntries(edit);
	std::atomic_bool cancelled{false};
	std::shared_ptr<const persistent::SelectedPublication> scheduled;
	{
		// Schedule-only build: existing placements keep their orderings (they
		// still hold for unchanged users) but are not validated, since edited
		// passes may have gained users the old orderings do not cover. A stale
		// ordering can in principle contradict a new data dependency; then
		// schedule once more without any placement orderings.
		BT_ZONE_SCOPE("ORG.Persistent.ScheduleBuild");
		try {
			scheduled = edit.Build(compiler.synchronousCompileWorkspace, cancelled, false);
		} catch (const std::exception& e) {
			spdlog::warn("Persistent schedule build failed with stale placement orderings ({}); retrying without them", e.what());
			basic_telemetry::AddCounter("ORG.Persistent.StalePlacementOrderingRetries");
			edit.ClearPlacementOrderings();
			state.aliasEdges.clear();
			scheduled = edit.Build(compiler.synchronousCompileWorkspace, cancelled, false);
		}
	}
	if (!scheduled) throw std::runtime_error("Persistent structural build failed (schedule)");
	// The frame transients' spans cover this schedule's uses before anyone can trace it or run it.
	ReportFrameTransientUses(*scheduled);
	// A first build off the owner thread installs the schedule alone; its first preparation places (PlaceDeferredPersistentAliases).
	if (state.deferAliasPlacement) {
		state.deferAliasPlacement = false;
		state.aliasPlacementDeferred = true;
		return scheduled;
	}
	// Unchanged orderings and bindings mean every overlapping pair is still
	// ordered between all of its users, so the scheduled publication is valid.
	if (!PlanPersistentAliasPlacement(edit, *scheduled)) return scheduled;
	BT_ZONE_SCOPE("ORG.Persistent.PlacedBuild");
	basic_telemetry::AddCounter("ORG.Persistent.PlacedRebuilds");
	auto ready = edit.Build(compiler.synchronousCompileWorkspace, cancelled);
	if (!ready) throw std::runtime_error("Persistent structural build failed (placement)");
	return ready;
}

void RenderGraph::RunPersistentStructuralBuildPhase(bool validated) {
	auto& compiler = *m_compilerState;
	auto pending = compiler.persistent->pending;
	auto run = [pending, validated] {
		BT_ZONE_SCOPE("ORG.Persistent.WorkerBuild");
		try {
			std::shared_ptr<const persistent::SelectedPublication> ready;
			try {
				ready = pending->edit->Build(pending->workspace, pending->cancelled, validated);
			} catch (const std::exception& e) {
				if (validated) throw;
				// Stale placement orderings can contradict a new dependency: schedule
				// without them (the planner recomputes every ordering it needs).
				spdlog::warn("Persistent schedule build failed with stale placement orderings ({}); retrying without them", e.what());
				basic_telemetry::AddCounter("ORG.Persistent.StalePlacementOrderingRetries");
				pending->edit->ClearPlacementOrderings();
				pending->clearedOrderings.store(true, std::memory_order_release);
				ready = pending->edit->Build(pending->workspace, pending->cancelled, false);
			}
			if (!ready) throw std::runtime_error("build produced no publication");
			pending->result = std::move(ready);
			pending->phase.store(validated ? 3 : 1, std::memory_order_release);
		} catch (const std::exception& e) {
			pending->error = e.what();
			pending->phase.store(-1, std::memory_order_release);
		}
	};
	pending->phase.store(validated ? 2 : 0, std::memory_order_release);
	if (!compiler.persistentBuildScope && m_taskService) compiler.persistentBuildScope = m_taskService->CreateScope("ORG.Persistent.Build");
	if (!m_taskService || !compiler.persistentBuildScope
		|| !m_taskService->Submit(compiler.persistentBuildScope, runtime::TaskPriority::Streaming, "ORG.Persistent.WorkerBuild", run))
		run();
}

void RenderGraph::SubmitPersistentStructuralBuild(const std::vector<uint32_t>& structural, const std::vector<uint32_t>& growGroups, rhi::Backend primaryBackend,
	bool forceInline) {
	BT_ZONE_SCOPE("ORG.Persistent.SubmitStructuralBuild");
	auto& compiler = *m_compilerState;
	auto& state = *compiler.persistent;
	// A relowered pass that declares resources the live publication does not
	// have cannot be prepared against it while the build is pending: such
	// edits build inline (one frame hitch). Growth, new passes and relowers
	// that add nothing are safe to build on the worker.
	bool inlineRequired = forceInline;
	std::unordered_map<uint32_t, LoweredPass> lowerings;
	for (const auto mainIndex : structural) {
		auto& main = state.mainPasses[mainIndex];
		auto& any = m_masterPassList[main.masterIndex];
		std::visit([&](auto& value) {
			using T = std::decay_t<decltype(value)>;
			if constexpr (!std::is_same_v<T, std::monostate>) {
				auto lowered = lowering::LowerPass(*this, _registry, m_queueRegistry, primaryBackend, value, any.type);
				const auto* dynamic = dynamic_cast<const IDynamicDeclaredResources*>(value.pass.get());
				UseRebindSlots(main, lowered, dynamic);
				if (!(dynamic && dynamic->PreparesOnlyDeclaredWork()) && LoweringAddsDeclarations(main, lowered)) {
					// Name the pass, so an app can find what makes it rebuild inline.
					spdlog::info("Persistent structural build inline: pass '{}' declares resources the live executable lacks", main.name);
					inlineRequired = true;
				}
				lowerings.emplace(mainIndex, std::move(lowered));
			}
		}, any.pass);
	}
	auto pending = std::make_shared<State::PendingStructural>();
	pending->started = std::chrono::steady_clock::now();
	pending->workspace.SetScheduleCache(m_runtime->Schedules());
	pending->base = state.program.Select();
	pending->baseSlotCount = pending->base->bindings.Size();
	std::optional<persistent::GraphEditTransaction> inlineEdit;
	if (inlineRequired) inlineEdit.emplace(state.program.BeginEdit());
	else pending->edit = std::make_unique<persistent::GraphEditTransaction>(pending->base);
	persistent::GraphEditTransaction* const transactionPtr = inlineRequired ? &*inlineEdit : pending->edit.get();
	auto& transaction = *transactionPtr;
	ApplyDeferredSharing(state, transaction);
	for (const auto groupIndex : growGroups) {
		GrowGroup(state, transaction, groupIndex, state.groups[groupIndex].capacity + 1);
		state.groups[groupIndex].identity = {}; // Re-syncs (binding-only) after the install.
		pending->touchedGroups.push_back(groupIndex);
	}
	for (const auto mainIndex : structural) {
		auto& main = state.mainPasses[mainIndex];
		auto& any = m_masterPassList[main.masterIndex];
		std::visit([&](auto& value) {
			using T = std::decay_t<decltype(value)>;
			if constexpr (!std::is_same_v<T, std::monostate>) {
				auto lowered = std::move(lowerings.at(mainIndex));
				std::vector<Resource*> resources;
				for (const auto& use : lowered.entries) resources.push_back(use.resource);
				for (const auto& group : lowered.groups) for (const auto& [id, resource] : group.members) resources.push_back(resource.get());
				MaterializePersistentStandalone(resources, true);
				auto views = std::move(main.declaredViews);
				auto bindings = std::move(main.declaredBindings);
				auto preparedUses = main.preparedResourceUses;
				auto preparedBindings = main.preparedBindings;
				const auto preparedDevice = main.device;
				InstallMainPass(*this, state, transaction, main, std::move(lowered), mainIndex, true, std::nullopt,
					inlineRequired ? nullptr : &pending->touchedGroups);
				if (!inlineRequired) {
					// Frames keep preparing against the current publication until the install.
					std::swap(views, main.declaredViews);
					std::swap(bindings, main.declaredBindings);
					std::swap(preparedUses, main.preparedResourceUses);
					std::swap(preparedBindings, main.preparedBindings);
					const auto loweredDevice = main.device;
					main.device = preparedDevice;
					pending->declaredTokens.push_back({mainIndex, std::move(views), std::move(bindings),
						std::move(preparedUses), std::move(preparedBindings), loweredDevice});
				}
				basic_telemetry::AddCounter("ORG.Persistent.StructuralPassRelowerings");
			}
		}, any.pass);
	}
	if (inlineRequired) {
		BT_ZONE_SCOPE("ORG.Persistent.InlineStructuralBuild");
		auto ready = BuildPersistentStructural(*inlineEdit);
		if (!ready || !state.program.Install(*inlineEdit, ready)) throw std::runtime_error("Persistent structural edit failed to install");
		++state.structuralBuilds;
		basic_telemetry::AddCounter("ORG.Persistent.StructuralBuilds");
		basic_telemetry::AddCounter("ORG.Persistent.StructuralBuildsInline");
		return;
	}
	// Frames keep their current slot maps until the new executable installs.
	for (uint32_t mainIndex = 0; mainIndex < state.mainPasses.size(); ++mainIndex) {
		auto& main = state.mainPasses[mainIndex];
		if (!main.slotsDirty) continue;
		main.slotsDirty = false;
		pending->dirtyPasses.push_back(mainIndex);
	}
	BindUnboundPersistentEntries(transaction);
	state.pending = pending;
	RunPersistentStructuralBuildPhase(false);
	basic_telemetry::AddCounter("ORG.Persistent.StructuralBuildsSubmitted");
}

void RenderGraph::AdvancePersistentStructuralBuild(std::vector<uint32_t>& structural) {
	auto& compiler = *m_compilerState;
	auto& state = *compiler.persistent;
	auto pending = state.pending;
	if (!pending) return;
	const int phase = pending->phase.load(std::memory_order_acquire);
	if (phase == 0 || phase == 2) return;
	if (phase < 0) throw std::runtime_error("Persistent structural build failed: " + pending->error);
	BT_ZONE_SCOPE("ORG.Persistent.AdvanceStructuralBuild");
	if (phase == 1) {
		if (pending->clearedOrderings.exchange(false)) state.aliasEdges.clear();
		// Unchanged orderings and bindings mean every overlapping pair is still
		// ordered between all of its users, so the scheduled publication is valid.
		if (PlanPersistentAliasPlacement(*pending->edit, *pending->result)) {
			basic_telemetry::AddCounter("ORG.Persistent.PlacedRebuilds");
			RunPersistentStructuralBuildPhase(true);
			return;
		}
	}
	// Replay binding edits installed since the base onto the result.
	auto ready = pending->result;
	if (!pending->bindingLog.empty()) {
		BT_ZONE_SCOPE("ORG.Persistent.ReplayBindingLog");
		persistent::GraphEditTransaction replay(ready);
		size_t applied = 0;
		for (const auto& [index, version] : pending->bindingLog) {
			const auto* current = ready->bindings.TryAt(index);
			if (!current) continue; // retired by the structural edit
			const auto slot = ready->bindings.CurrentSlot(index);
			if (!version) {
				if (current->bound) { replay.Unbind(slot); ++applied; }
				continue;
			}
			if (current->bound && current->identity == version->identity && current->backingRevision == version->backingRevision
				&& current->recording == version->recording) continue;
			if (current->bound) replay.ReplaceBinding(slot, *version);
			else replay.BindReserved(slot, *version);
			++applied;
		}
		if (applied) {
			std::atomic_bool cancelled{false};
			ready = replay.Build(compiler.synchronousCompileWorkspace, cancelled);
			if (!ready) throw std::runtime_error("Persistent structural build failed to replay binding edits");
		}
		BT_ZONE_VALUE(static_cast<int64_t>(applied));
	}
	if (!state.program.InstallRebased(ready)) throw std::runtime_error("Persistent structural build failed to install");
	for (const auto mainIndex : pending->dirtyPasses) state.mainPasses[mainIndex].slotsDirty = true;
	for (auto& tokens : pending->declaredTokens) {
		auto& main = state.mainPasses[tokens.mainIndex];
		main.declaredViews = std::move(tokens.views);
		main.declaredBindings = std::move(tokens.bindings);
		main.preparedResourceUses = std::move(tokens.preparedResourceUses);
		main.preparedBindings = std::move(tokens.preparedBindings);
		main.device = tokens.device;
	}
	structural.insert(structural.end(), pending->deferredStructural.begin(), pending->deferredStructural.end());
	++state.structuralBuilds;
	basic_telemetry::AddCounter("ORG.Persistent.StructuralBuilds");
	spdlog::info("Persistent structural build installed: latency_ms={:.2f} replayed={} deferred={}",
		std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pending->started).count(),
		pending->bindingLog.size(), pending->deferredStructural.size());
	state.pending.reset();
}

void RenderGraph::DrainPersistentStructuralBuild(std::vector<uint32_t>& structural) {
	BT_ZONE_SCOPE("ORG.Persistent.DrainStructuralBuild");
	auto& state = *m_compilerState->persistent;
	while (state.pending) {
		AdvancePersistentStructuralBuild(structural);
		if (state.pending) std::this_thread::yield();
	}
	basic_telemetry::AddCounter("ORG.Persistent.StructuralBuildsDrained");
}


// Alias placement over the scheduled executable. Lifetimes are ranks in batch
// order (a linear extension of the scheduling edges), packed per pool with the
// same greedy sweep-line as the per-frame planner. Overlapping placements get
// pairwise placement orderings between all their users, so any schedule of the
// rebuilt executable keeps them disjoint in time (ValidateAliasOrder checks it).
void RenderGraph::RetireIdlePersistentAliasPools(PersistentExecutionState& state) {
	const uint64_t retireFrames = m_getAutoAliasPoolRetireIdleFrames ? (std::max)(1u, m_getAutoAliasPoolRetireIdleFrames()) : 120u;
	if (!state.aliasArena) return;
	// Each preparation retires what idled long enough; one that finds the arena busy (another graph's plan holds it, which may be
	// allocating a pool on its own thread) leaves it to the next. Waiting here put that allocation on the render thread.
	std::unique_lock arenaLock(state.aliasArena->mutex, std::try_to_lock);
	if (!arenaLock.owns_lock()) return;
	auto& pools = state.aliasArena->pools;
	for (auto it = pools.begin(); it != pools.end();) {
		auto& pool = it->second;
		// Idle for every graph: none places into it.
		if (!pool.required.empty() || !pool.idleSinceFrame || state.frameNumber < pool.idleSinceFrame
			|| state.frameNumber - pool.idleSinceFrame < retireFrames) { ++it; continue; }
		spdlog::info("Persistent alias pool retired: pool={:#x} capacity={} idleFrames={}", it->first, pool.capacityBytes,
			state.frameNumber - pool.idleSinceFrame);
		basic_telemetry::AddCounter("ORG.Persistent.AliasPoolRetirements");
		if (pool.allocation) DeletionManager::GetInstance().MarkForDelete(std::move(pool.allocation), "RenderGraph alias pool (idle)");
		it = pools.erase(it);
	}
}

namespace {
// The resources some graph placed in an alias pool and has not made standalone since, across every arena (private pools included). A
// planner counts them as ORG transients, not as materialized by someone else: a later build of the same epoch (a staged graph) places
// them again rather than keep the placement of the graph it replaces, whose pools go with it.
std::mutex g_aliasPlacedMutex;
// By resource: the epoch whose graph placed it last (RenderGraph pass epoch filter).
std::unordered_map<const void*, uint32_t> g_aliasPlaced;

bool AliasPlacedAnywhere(const void* resource) {
	std::lock_guard lock(g_aliasPlacedMutex);
	return g_aliasPlaced.contains(resource);
}

// Whether a graph of another epoch placed it: then it is that graph's too, and a plan of this epoch must not move it.
bool AliasPlacedByOtherEpoch(const void* resource, uint32_t epoch) {
	std::lock_guard lock(g_aliasPlacedMutex);
	const auto found = g_aliasPlaced.find(resource);
	return found != g_aliasPlaced.end() && found->second != epoch;
}

// Placed: recorded, with the epoch that placed it. Not placed: forgotten; returns whether it was recorded.
bool NoteAliasPlaced(const void* resource, bool placed, uint32_t epoch = UINT32_MAX) {
	std::lock_guard lock(g_aliasPlacedMutex);
	if (placed) return g_aliasPlaced.insert_or_assign(resource, epoch).second;
	return g_aliasPlaced.erase(resource) != 0;
}

// Resources placed from a frame placement: where, and at which backing generation. A graph that finds a resource already at its
// frame placement binds it as it is. Another epoch's graph placed it there from the same version, and its bindings stay valid.
struct FramePlacedRecord {
	std::weak_ptr<Resource> resource;
	const AliasHeapGeneration* heap = nullptr;
	uint64_t offset = 0, generation = 0;
	// The graph states that bind the placement: the one that placed it, and each newer build that found the resource at its frame
	// placement and claimed it. The last of them to end gives it back. So the build a newer one replaces does not take the backing
	// away from it when it ends, and a newer build dropped before its attach does not take it from the graph that keeps running.
	std::vector<const void*> owners;
	// For the memory log (FramePlacedResidents): its bytes, and the epoch whose graph placed it.
	uint64_t size = 0;
	uint32_t epoch = UINT32_MAX;
};
std::mutex g_framePlacedMutex;
std::unordered_map<const void*, FramePlacedRecord> g_framePlaced;

bool AtFramePlacement(Resource* concrete, const AliasHeapGeneration* heap, uint64_t offset, uint64_t generation) {
	std::lock_guard lock(g_framePlacedMutex);
	const auto found = g_framePlaced.find(concrete);
	if (found == g_framePlaced.end()) return false;
	const auto& record = found->second;
	return record.resource.lock().get() == concrete && record.heap == heap && record.offset == offset && record.generation == generation;
}

void NoteFramePlaced(Resource* concrete, const AliasHeapGeneration* heap, uint64_t offset, uint64_t generation, const void* owner,
	uint64_t size, uint32_t epoch) {
	std::lock_guard lock(g_framePlacedMutex);
	// Records of resources that are gone are dropped now and then.
	if (g_framePlaced.size() > 4096) std::erase_if(g_framePlaced, [](const auto& entry) { return entry.second.resource.expired(); });
	g_framePlaced[concrete] = {concrete->weak_from_this(), heap, offset, generation, {owner}, size, epoch};
}

// Where concrete is placed from a frame placement now (its heap, offset and bytes), if it still is.
bool FramePlacedWhere(Resource* concrete, const AliasHeapGeneration*& heap, uint64_t& offset, uint64_t& size) {
	std::lock_guard lock(g_framePlacedMutex);
	const auto found = g_framePlaced.find(concrete);
	if (found == g_framePlaced.end()) return false;
	const auto& record = found->second;
	const auto resource = record.resource.lock();
	auto* backed = resource.get() == concrete ? dynamic_cast<BackedResource*>(concrete) : nullptr;
	if (!backed || !backed->IsMaterialized() || backed->GetBackingGeneration() != record.generation) return false;
	heap = record.heap;
	offset = record.offset;
	size = record.size;
	return true;
}

// Makes owner the graph state that gives the frame placement of concrete back.
void ClaimFramePlaced(const void* concrete, const void* owner) {
	std::lock_guard lock(g_framePlacedMutex);
	if (const auto found = g_framePlaced.find(concrete); found != g_framePlaced.end() && std::ranges::find(found->second.owners, owner) == found->second.owners.end())
		found->second.owners.push_back(owner);
}

// Whether another graph state claimed the frame placement of concrete since owner placed it.
bool FramePlacementClaimedByOther(const void* concrete, const void* owner) {
	std::lock_guard lock(g_framePlacedMutex);
	const auto found = g_framePlaced.find(concrete);
	return found != g_framePlaced.end() && !found->second.resource.expired()
		&& std::ranges::any_of(found->second.owners, [&](const void* other) { return other && other != owner; });
}

// owner ends: it binds none of the placements it made or claimed any more.
void ForgetFramePlacementClaims(const void* owner) {
	std::lock_guard lock(g_framePlacedMutex);
	for (auto& [key, record] : g_framePlaced) std::erase(record.owners, owner);
}

// A candidate's key in usage summaries (PlacementUse::key): its name and memory description. Resources are made again when a
// graph is built again, so their ids change. Their names and descriptions do not.
uint64_t PlacementKey(const std::string& name, uint64_t size, uint64_t alignment, uint8_t resourceClass, uint64_t backendMask) {
	uint64_t key = std::hash<std::string>{}(name);
	boost::hash_combine(key, size);
	boost::hash_combine(key, alignment);
	boost::hash_combine(key, resourceClass);
	boost::hash_combine(key, backendMask);
	return key;
}
}

std::vector<FramePlacedResident> FramePlacedResidents() {
	std::vector<FramePlacedResident> out;
	std::lock_guard lock(g_framePlacedMutex);
	for (const auto& [key, record] : g_framePlaced) {
		const auto resource = record.resource.lock();
		auto* backed = resource ? dynamic_cast<BackedResource*>(resource.get()) : nullptr;
		// Only resources still at the placement the record names. A resource placed again elsewhere left that heap.
		if (!backed || !backed->IsMaterialized() || backed->GetBackingGeneration() != record.generation) continue;
		out.push_back({record.heap, resource->GetName(), record.offset, record.size, record.epoch, resource->GetGlobalResourceID() + 1});
	}
	return out;
}

void persistent_impl::ReleaseOwnAliasPlacements(const std::unordered_map<Resource*, AliasPlacementOwner>& owners, const void* self) noexcept {
	size_t released = 0;
	for (const auto& [concrete, owner] : owners) {
		// Another graph binds it: a newer build that placed from the same frame placement, or, when this is a build dropped before
		// its attach, the running graph it found the resource with.
		if (FramePlacementClaimedByOther(concrete, self)) continue;
		const auto resource = owner.resource.lock();
		auto* backed = resource ? dynamic_cast<BackedResource*>(resource.get()) : nullptr;
		if (!backed || !backed->IsMaterialized() || backed->GetBackingGeneration() != owner.generation) continue;
		try {
			if (auto* texture = dynamic_cast<PixelBuffer*>(resource.get())) texture->Dematerialize();
			else if (auto* buffer = dynamic_cast<BufferBase*>(resource.get())) buffer->Dematerialize();
			else continue;
		} catch (const std::exception& error) {
			spdlog::error("Persistent alias placement of '{}' not given back: {}", resource->GetName(), error.what());
			continue;
		}
		NoteAliasPlaced(concrete, false);
		++released;
	}
	ForgetFramePlacementClaims(self);
	if (released) spdlog::info("Persistent alias placements given back at a graph's end: {}", released);
}

namespace {
// By resource: the graph states that declare it, with their pass epoch filters (DeclareEpochUses).
std::mutex g_epochUsesMutex;
std::unordered_map<const void*, std::vector<std::pair<const void*, uint32_t>>> g_epochUses;
std::atomic<uint64_t> g_crossEpochSerial{0};
std::atomic<uint64_t> g_placementTakenSerial{0};
}

void persistent_impl::NoteAliasPlacementTaken() noexcept { g_placementTakenSerial.fetch_add(1, std::memory_order_acq_rel); }

uint64_t persistent_impl::AliasPlacementTakenSerial() noexcept { return g_placementTakenSerial.load(std::memory_order_acquire); }

bool persistent_impl::DeclareEpochUses(const void* state, uint32_t epoch, std::vector<const void*>& declared, std::vector<const void*> resources) {
	std::sort(resources.begin(), resources.end());
	resources.erase(std::unique(resources.begin(), resources.end()), resources.end());
	if (resources == declared) return false;
	bool crossed = false;
	{
		std::lock_guard lock(g_epochUsesMutex);
		std::vector<const void*> gone, added;
		std::set_difference(declared.begin(), declared.end(), resources.begin(), resources.end(), std::back_inserter(gone));
		std::set_difference(resources.begin(), resources.end(), declared.begin(), declared.end(), std::back_inserter(added));
		for (const auto* resource : gone) {
			const auto found = g_epochUses.find(resource);
			if (found == g_epochUses.end()) continue;
			std::erase_if(found->second, [&](const auto& use) { return use.first == state; });
			if (found->second.empty()) g_epochUses.erase(found);
		}
		for (const auto* resource : added) {
			auto& uses = g_epochUses[resource];
			for (const auto& use : uses) crossed = crossed || use.second != epoch;
			uses.emplace_back(state, epoch);
		}
	}
	declared = std::move(resources);
	if (crossed) g_crossEpochSerial.fetch_add(1, std::memory_order_acq_rel);
	return crossed;
}

void persistent_impl::ForgetEpochUses(const void* state, std::vector<const void*>& declared) noexcept {
	if (declared.empty()) return;
	std::lock_guard lock(g_epochUsesMutex);
	for (const auto* resource : declared) {
		const auto found = g_epochUses.find(resource);
		if (found == g_epochUses.end()) continue;
		std::erase_if(found->second, [&](const auto& use) { return use.first == state; });
		if (found->second.empty()) g_epochUses.erase(found);
	}
	declared.clear();
}

bool persistent_impl::DeclaredByOtherEpoch(const void* resource, uint32_t epoch) {
	std::lock_guard lock(g_epochUsesMutex);
	const auto found = g_epochUses.find(resource);
	if (found == g_epochUses.end()) return false;
	for (const auto& use : found->second)
		if (use.second != epoch) return true;
	return false;
}

uint64_t persistent_impl::CrossEpochUsesSerial() noexcept {
	return g_crossEpochSerial.load(std::memory_order_acquire);
}

void persistent_impl::SyncEpochUses(State& state, uint32_t epoch) {
	std::vector<const void*> resources;
	resources.reserve(state.entries.size());
	for (const auto& entry : state.entries) {
		if (entry.retired || entry.versioned || !entry.resource || entry.subscribers.empty()) continue;
		auto* concrete = UnwrapDynamic(entry.resource);
		// Only what an alias plan may place. Resources their makers materialize are bound as they are by every graph.
		bool aliasable = false;
		if (auto* texture = dynamic_cast<PixelBuffer*>(concrete)) aliasable = texture->GetDescription().allowAlias;
		else if (auto* buffer = dynamic_cast<BufferBase*>(concrete))
			aliasable = buffer->IsAliasingAllowed() && buffer->GetAccessType() == rhi::HeapType::DeviceLocal;
		if (aliasable) resources.push_back(concrete);
	}
	DeclareEpochUses(&state, epoch, state.declaredUses, std::move(resources));
}

void RenderGraph::ReportFrameTransientUses(const persistent::SelectedPublication& scheduled) {
	if (!m_frameTransientObserver || !scheduled.executable) return;
	const auto& state = *m_compilerState->persistent;
	const auto& executable = *scheduled.executable;
	const auto& structure = *executable.graph->structure;
	// Every resource the passes use, once each.
	std::vector<Resource*> used;
	std::unordered_set<Resource*> seen;
	const auto note = [&](uint32_t resourceIndex) {
		if (resourceIndex >= executable.resourceSlots.size()) return;
		const auto slot = executable.resourceSlots[resourceIndex];
		if (slot.index >= state.entries.size()) return;
		const auto& entry = state.entries[slot.index];
		Resource* concrete = nullptr;
		if (entry.versioned) {
			// A versioned slot (an imported image's): its current version's backing. A frame transient's slot has one version.
			if (entry.versioned->PerFrame()) return;
			const auto version = entry.versioned->SelectVersion(nullptr);
			if (!version || !version->resource) return;
			concrete = UnwrapDynamic(version->resource.get());
		} else if (entry.resource) {
			concrete = UnwrapDynamic(entry.resource);
		}
		if (concrete && seen.insert(concrete).second) used.push_back(concrete);
	};
	for (const auto& pass : structure.passes) {
		for (const auto& use : pass.entryStates) note(use.resource);
		for (const auto& use : pass.exitStates) note(use.resource);
		for (const auto& access : pass.accesses) note(access.resourceIndex);
	}
	m_frameTransientObserver(used);
}

bool RenderGraph::PlanPersistentAliasPlacement(persistent::GraphEditTransaction& edit, const persistent::SelectedPublication& scheduled) {
	BT_ZONE_SCOPE("ORG.Persistent.PlanAliases");
	using PlanClock = std::chrono::steady_clock;
	const auto planBegin = PlanClock::now();
	double acquireMs = 0.0;
	size_t materializedCount = 0;
	auto& state = *m_compilerState->persistent;
	const auto& executable = *scheduled.executable;
	const auto& graph = *executable.graph;
	const auto& structure = *graph.structure;
	const auto mode = m_getAutoAliasMode ? m_getAutoAliasMode() : AutoAliasMode::Off;
	const bool logging = m_getAutoAliasEnableLogging && m_getAutoAliasEnableLogging();
	const float headroom = m_getAutoAliasPoolGrowthHeadroom ? (std::max)(1.0f, m_getAutoAliasPoolGrowthHeadroom()) : 1.5f;
	const auto packingStrategy = m_getAutoAliasPackingStrategy ? m_getAutoAliasPackingStrategy() : AutoAliasPackingStrategy::GreedySweepLine;
	state.aliasPackingStrategy = packingStrategy;
	++state.aliasPlans;
	// Versioned slots are candidates only while the host places transients across the frame (SetFramePlacementEnabled). Off,
	// nothing here touches them.
	const bool placeVersions = m_framePlacementEnabled && m_framePlacementEnabled();
	for (auto& entry : state.entries) entry.aliasPlannedVersion = 0;
	// Which of this graph's transients other epochs' graphs declare too. The serial first: a declaration made after it makes the
	// next preparation look again (BindPersistentFrame).
	state.crossEpochSerialAtPlan = CrossEpochUsesSerial();
	state.placementTakenSerialAtPlan = AliasPlacementTakenSerial();
	SyncEpochUses(state, m_passEpochFilter);
	// Placements of this graph's that another graph has taken since (given a backing of its own, or placed again): no longer this
	// graph's, so this plan neither keeps them as they were nor gives their backings back.
	for (auto it = state.aliasPlacementOwners.begin(); it != state.aliasPlacementOwners.end();) {
		const auto resource = it->second.resource.lock();
		auto* backed = resource ? dynamic_cast<BackedResource*>(resource.get()) : nullptr;
		if (backed && backed->IsMaterialized() && backed->GetBackingGeneration() == it->second.generation) { ++it; continue; }
		state.aliasPlaced.erase(it->first);
		it = state.aliasPlacementOwners.erase(it);
	}
	// Another graph placed it in its own pools, not at a frame placement. Those pools see only that graph's uses, and every graph of
	// an arena packs its pools from offset 0, so binding it there puts two unordered placements into one heap. This graph gives it
	// a backing of its own instead, and the other graph, finding its placement taken, binds that one (see above).
	const auto inOtherOwnPool = [&](Resource* concrete) {
		if (state.aliasPlaced.contains(concrete) || !AliasPlacedAnywhere(concrete)) return false;
		const AliasHeapGeneration* heldHeap = nullptr;
		uint64_t heldOffset = 0, heldSize = 0;
		return !FramePlacedWhere(concrete, heldHeap, heldOffset, heldSize);
	};

	std::vector<uint32_t> rank(structure.passes.size(), UINT32_MAX);
	std::vector<uint8_t> backendByPass(structure.passes.size(), 0);
	uint32_t nextRank = 0;
	for (const auto& batch : graph.batches) for (const auto pass : batch.passes) {
		rank.at(pass) = nextRank++;
		backendByPass[pass] = static_cast<uint8_t>(structure.queues.at(batch.queue).backendInstance);
	}
	uint32_t maxCriticality = 1;
	for (const auto c : graph.criticality) maxCriticality = (std::max)(maxCriticality, c);

	// The host's frame transients: their spans grow to this graph's position before it runs. A throw fails the plan.
	ReportFrameTransientUses(scheduled);

	struct Candidate {
		Resource* concrete = nullptr;
		bool texture = false;
		bool direct = true;
		uint8_t resourceClass = 0;
		uint64_t sizeBytes = 0, alignment = 1;
		uint32_t firstUse = UINT32_MAX, lastUse = 0;
		bool firstUseIsWrite = false;
		// The pass of the first use, a compiled pass index. Its MayNotWrite flag decides whether the first write is guaranteed.
		uint32_t firstPass = UINT32_MAX;
		bool writeGuaranteed = false;
		uint32_t criticality = 0;
		uint64_t backendMask = 0;
		std::optional<uint64_t> manualPool;
		bool materialized = false; // By someone other than this planner.
		std::vector<uint32_t> users;   // Compiled pass indices.
		uint64_t poolID = 0, offset = 0;
		bool pooled = false;
		// Used by passes on more than one device: placed in a shared pool (PlaceSharedRepresentations).
		bool shared = false;
		const char* exclusion = nullptr;
		// For the usage summary (PlacementSummary): whether the planner would pool it automatically on one backend if its first
		// use wrote, and why not.
		bool scoreOk = false;
		bool eligible = false;
		const char* ineligible = nullptr;
		// Placed from the host's frame placement (FramePlacementVersion): poolID and offset are its block's and its entry's.
		bool framePlaced = false;
		uint32_t frameBlock = UINT32_MAX;
		// Several epochs use it, by the frame placement. Its placement is then not this graph's own: the graph's end leaves it.
		bool frameShared = false;
		// The frame placement knows other epochs use it, but does not cover it here. Another epoch's graph may have placed it,
		// so this graph keeps whatever backing it has rather than move it under that graph's executable.
		bool keepBacking = false;
		// The current version of a versioned slot (frame placement only). It is placed only at a frame placement's offset, never
		// in the graph's own pools. The graph's end never gives it back, since the version outlives the graph. version keeps
		// the version alive for the plan. Its key in usage summaries holds the version number, so each version is an entry of
		// its own.
		std::shared_ptr<const ResourceVersion> version;
		uint32_t versionedEntry = UINT32_MAX;
		// It keeps its contents across the runs of a repeated epoch (Resource::SetContentAcrossRuns). Only a frame placement
		// places it, with one lifetime over every run.
		bool acrossRuns = false;
		// Placed from a frame placement, its contents carry over into this graph's executions from an earlier one in the frame:
		// this is not the first epoch of the frame that uses it, or it keeps its contents across runs. It then enters without a
		// discard (PreparedBackingState::aliasContinues).
		bool continues = false;
		// Its key in usage summaries and frame placements (PlacementUse::key).
		uint64_t key = 0;
		// Placed from a frame placement: its lifetime in this graph's ranks, as the version's bytes see it. Its uses, from rank 0
		// when its contents come from an earlier epoch, to the last rank when they go on to a later one (or across runs).
		uint32_t frameFirst = 0, frameLast = 0;
		// A graph of another epoch declares it too (DeclaredByOtherEpoch). This graph's own pools see only its own uses, so it is
		// never placed there: only a frame placement places it, or it keeps a backing of its own.
		bool otherEpochs = false;
		// Its own path, as planned before the frame placement step. Restored when the frame placement does not take it.
		bool ownPooled = false;
		uint64_t ownPoolID = 0;
		const char* ownExclusion = nullptr;
	};
	std::vector<Candidate> candidates;
	std::unordered_map<Resource*, uint32_t> candidateByConcrete;
	std::unordered_map<Resource*, std::vector<uint32_t>> entriesByConcrete;
	for (uint32_t index = 0; index < state.entries.size(); ++index) {
		const auto& entry = state.entries[index];
		if (entry.resource && !entry.versioned) entriesByConcrete[UnwrapDynamic(entry.resource)].push_back(index);
	}
	auto device = DeviceManager::GetInstance().GetDevice();
	std::unordered_set<Resource*> notAliasable;
	auto candidateFor = [&](uint32_t resourceIndex) -> uint32_t {
		if (resourceIndex >= executable.resourceSlots.size()) return UINT32_MAX;
		const auto slot = executable.resourceSlots[resourceIndex];
		if (slot.index >= state.entries.size()) return UINT32_MAX;
		const auto& entry = state.entries[slot.index];
		Resource* concrete = nullptr;
		std::shared_ptr<const ResourceVersion> version;
		if (entry.versioned) {
			// A versioned slot's current version, while the frame placement is on. Never a per-frame resource (the swapchain), a
			// late-bound one or one used on several devices. A graph lowered for a pending slice count binds another version
			// than the current one, so it leaves the slot out.
			if (!placeVersions || entry.retired || entry.multiDevice || entry.versioned->PerFrame() || entry.versioned->LateBound())
				return UINT32_MAX;
			version = entry.versioned->SelectVersion(nullptr);
			if (!version || !version->resource || lowering::ShapeOf(*version->resource) != entry.shape) return UINT32_MAX;
			concrete = UnwrapDynamic(version->resource.get());
		} else if (entry.resource) {
			concrete = UnwrapDynamic(entry.resource);
		}
		if (!concrete) return UINT32_MAX;
		if (!IsPersistentAliasCandidate(concrete)) {
			if (logging) notAliasable.insert(concrete);
			return UINT32_MAX;
		}
		if (const auto found = candidateByConcrete.find(concrete); found != candidateByConcrete.end()) return found->second;
		Candidate c;
		c.concrete = concrete;
		if (version) {
			c.version = std::move(version);
			c.versionedEntry = slot.index;
		}
		for (const auto entryIndex : entriesByConcrete[concrete]) c.direct = c.direct && state.entries[entryIndex].direct;
		rhi::ResourceDesc desc{};
		if (auto* texture = dynamic_cast<PixelBuffer*>(concrete)) {
			const auto& d = texture->GetDescription();
			c.texture = true;
			c.resourceClass = (d.hasRTV || d.hasDSV) ? 2u : 3u;
			c.manualPool = d.aliasingPoolID;
			desc = alias::AliasTextureResourceDesc(d);
		} else {
			auto* buffer = static_cast<BufferBase*>(concrete);
			c.resourceClass = 1u;
			c.manualPool = buffer->GetAliasingPoolHint();
			desc = alias::AliasBufferResourceDesc(buffer->GetBufferSize(), buffer->IsUnorderedAccessEnabled(), buffer->GetAccessType());
		}
		// A version's place comes from a frame placement alone, so a manual pool means nothing for it.
		if (c.version) c.manualPool.reset();
		c.shared = state.multiDeviceResources.contains(concrete);
		c.otherEpochs = DeclaredByOtherEpoch(concrete, m_passEpochFilter);
		c.acrossRuns = concrete->ContentAcrossRuns();
		if (c.shared) c.resourceClass = c.texture ? 3u : 1u; // Render targets and depth do not qualify (IsSharedAliasEligible).
		rhi::ResourceAllocationInfo info{};
		if (c.shared) info = SharedAliasAllocationInfo(concrete);
		else device.GetResourceAllocationInfo(&desc, 1, &info);
		c.sizeBytes = info.sizeInBytes;
		c.alignment = (std::max<uint64_t>)(1, info.alignment);
		auto* backed = dynamic_cast<BackedResource*>(concrete);
		c.materialized = backed && backed->IsMaterialized() && !state.aliasPlaced.contains(concrete);
		// Placed by another graph's alias pool (the graph this one replaces, whatever its arena): an ORG transient all the same.
		if (c.materialized)
			c.materialized = !AliasPlacedAnywhere(concrete);
		const auto index = static_cast<uint32_t>(candidates.size());
		candidateByConcrete.emplace(concrete, index);
		candidates.push_back(std::move(c));
		return index;
	};
	for (uint32_t p = 0; p < structure.passes.size(); ++p) {
		const auto& pass = structure.passes[p];
		auto touch = [&](uint32_t resource, bool write) {
			const auto index = candidateFor(resource);
			if (index == UINT32_MAX) return;
			auto& c = candidates[index];
			const auto passRank = rank[p];
			if (passRank < c.firstUse) { c.firstUse = passRank; c.firstUseIsWrite = write; c.firstPass = p; }
			else if (passRank == c.firstUse) c.firstUseIsWrite = c.firstUseIsWrite || write;
			c.lastUse = (std::max)(c.lastUse, passRank);
			c.criticality = (std::max)(c.criticality, graph.criticality.at(p));
			c.backendMask |= uint64_t{1} << backendByPass[p];
			if (c.users.empty() || c.users.back() != p) c.users.push_back(p);
		};
		for (const auto& use : pass.entryStates)
			touch(use.resource, use.state.write || use.state.access == static_cast<uint64_t>(rhi::ResourceAccessType::Common));
		for (const auto& use : pass.exitStates) touch(use.resource, true);
		for (const auto& access : pass.accesses) touch(access.resourceIndex, access.write);
	}

	constexpr uint64_t kAutoPoolGlobal = 0xA171000000000000ull;
	const float threshold = mode == AutoAliasMode::Conservative ? 1.0f : mode == AutoAliasMode::Balanced ? 0.25f
		: mode == AutoAliasMode::Aggressive ? -0.5f : std::numeric_limits<float>::infinity();
	// Each epoch's graph packs its transients from offset 0, so the epochs sharing an arena overlap in memory. The alias ledger
	// orders that: a run whose transient overlaps another epoch's committed use enters it from Undefined, after that use (a wait
	// on another timeline, sync All on its own queue).
	// A transient holds nothing from one run to the next, and with shared pools its memory holds another epoch's data in between.
	// A pass that reads texels no pass wrote in that run shows the other epoch's data. A screen-space GI pass did (its radiance
	// prefilter read past the frame), which gave the green streaks once blamed on sharing.
	size_t excludedCount = 0;
	for (auto& c : candidates) {
		std::sort(c.users.begin(), c.users.end());
		c.users.erase(std::unique(c.users.begin(), c.users.end()), c.users.end());
		// A first write by a pass that may skip it (MayNotWrite) leaves the memory's old content in the runs where it does.
		c.writeGuaranteed = c.firstUseIsWrite && c.firstPass < structure.passes.size() && !structure.passes[c.firstPass].mayNotWrite;
		if (c.firstUse != UINT32_MAX && c.direct && !c.manualPool && mode != AutoAliasMode::Off) {
			const float benefitMB = static_cast<float>(c.sizeBytes) / (1024.0f * 1024.0f);
			const float critNorm = static_cast<float>(c.criticality) / static_cast<float>(maxCriticality);
			const float penalty = c.materialized ? 1.0f : 0.0f;
			const float score = mode == AutoAliasMode::Conservative ? benefitMB - 2.0f * critNorm - 1.0f * penalty
				: mode == AutoAliasMode::Balanced ? benefitMB - 1.25f * critNorm - 0.5f * penalty
				: benefitMB - 0.5f * critNorm - 0.25f * penalty;
			c.scoreOk = score >= threshold;
		}
		if (c.firstUse == UINT32_MAX) c.ineligible = "unused";
		else if (!c.direct) c.ineligible = "group member";
		else if (c.manualPool) c.ineligible = "manual pool";
		else if (mode == AutoAliasMode::Off) c.ineligible = "auto aliasing off";
		else if (!c.scoreOk) c.ineligible = "score below threshold";
		else if (c.shared) c.ineligible = "used on several devices";
		else if (std::popcount(c.backendMask) > 1) c.ineligible = "multi-backend placement unsupported";
		c.eligible = !c.ineligible;
		if (c.firstUse == UINT32_MAX) { c.exclusion = "unused"; }
		else if (!c.direct) { c.exclusion = "group member"; }
		else if (c.manualPool) {
			if (!c.firstUseIsWrite)
				throw std::runtime_error("Aliasing candidate '" + c.concrete->GetName() + "' in manual pool has a first-use read");
			c.poolID = *c.manualPool; c.pooled = true;
		// A gated first writer (MayNotWrite) does not keep a resource out of this graph's own pools: its readers in the graph
		// share the gate. It only keeps the resource out of frame placement, where another epoch may read it.
		} else if (mode == AutoAliasMode::Off) c.exclusion = "auto aliasing off";
		else if (!c.firstUseIsWrite) c.exclusion = "first use is read";
		else if (!c.scoreOk) c.exclusion = "score below threshold";
		// Contents kept across runs: another epoch's transients in the shared pools could use its bytes between two runs. Only a
		// frame placement, which spans every run, may place it. It stays eligible for one.
		else if (c.acrossRuns) c.exclusion = "keeps its contents across runs";
		else { c.poolID = kAutoPoolGlobal; c.pooled = true; }
		if (c.pooled && c.shared) {
			// One pool per class for every resource the devices share, whichever device uses it.
			boost::hash_combine(c.poolID, 0x5A4EDu);
			boost::hash_combine(c.poolID, c.resourceClass);
		} else if (c.pooled) {
			boost::hash_combine(c.poolID, c.backendMask);
			boost::hash_combine(c.poolID, c.resourceClass);
			if (std::popcount(c.backendMask) > 1) { c.pooled = false; c.exclusion = "multi-backend placement unsupported"; }
		}
		if (c.exclusion) ++excludedCount;
	}

	// Keys (PlacementUse::key): the name and memory description, and the ordinal among this graph's candidates with the same
	// ones, in first-use order (then in the order the passes first touch them). A rebuild of the same graph makes new resources,
	// and they get the keys their predecessors had. A version's key also holds its number, so a new version is a new entry.
	{
		std::vector<uint32_t> order(candidates.size());
		for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
		std::stable_sort(order.begin(), order.end(), [&](uint32_t l, uint32_t r) {
			return std::tuple(candidates[l].firstUse, candidates[l].lastUse) < std::tuple(candidates[r].firstUse, candidates[r].lastUse);
		});
		std::unordered_map<uint64_t, uint64_t> ordinals;
		for (const auto index : order) {
			auto& c = candidates[index];
			auto key = PlacementKey(c.concrete->GetName(), c.sizeBytes, c.alignment, c.resourceClass, c.backendMask);
			if (const auto ordinal = ordinals[key]++) boost::hash_combine(key, ordinal);
			if (c.version) boost::hash_combine(key, c.version->number);
			c.key = key;
		}
	}
	// A resource's identity in usage summaries (PlacementUse::objectId).
	const auto objectOf = [](const Candidate& c) { return c.concrete->GetGlobalResourceID() + 1; };
	uint64_t framePlacedBytes = 0, framePlacedCount = 0, frameWithheld = 0, frameOwnPath = 0, frameMoved = 0, frameFitted = 0;
	uint64_t requiredBytes = 0;
	std::array<uint64_t, 4> requiredByClass{};
	// The usage summary: every used candidate with its ranks, for the host's frame-wide placement.
	// Candidates the plan would pool but did not place from its version (PlacementSummary::uncovered).
	const auto uncoveredCount = [&] {
		uint32_t count = 0;
		for (const auto& c : candidates)
			if (c.eligible && !c.framePlaced && (c.pooled || c.keepBacking)) ++count;
		return count;
	};
	// countUncovered false: the first summary, before any version placed anything.
	const auto publishSummary = [&](bool countUncovered) {
		if (!m_placementSummaryObserver) return;
		PlacementSummary summary;
		summary.epoch = m_passEpochFilter;
		summary.rankCount = nextRank;
		summary.pooledBytesByClass = requiredByClass;
		summary.pooledBytes = requiredBytes;
		summary.framePlacedBytes = framePlacedBytes;
		if (countUncovered) summary.uncovered = uncoveredCount();
		summary.uses.reserve(candidates.size());
		for (const auto& c : candidates) {
			if (c.firstUse == UINT32_MAX || !c.direct) continue;
			PlacementUse use;
			use.name = c.concrete->GetName();
			use.key = c.key;
			use.objectId = objectOf(c);
			use.firstRank = c.firstUse;
			use.lastRank = c.lastUse;
			use.firstUseWrites = c.firstUseIsWrite;
			use.writeGuaranteed = c.writeGuaranteed;
			use.loopCrossing = c.acrossRuns;
			use.eligible = c.eligible;
			if (c.ineligible) use.exclusion = c.ineligible;
			else if (c.exclusion) use.exclusion = c.exclusion;
			use.size = c.sizeBytes;
			use.alignment = c.alignment;
			use.memoryClass = c.resourceClass;
			use.backendMask = c.backendMask;
			summary.uses.push_back(std::move(use));
		}
		summary.Finish();
		try {
			m_placementSummaryObserver(std::move(summary));
		} catch (const std::exception& error) {
			spdlog::warn("Persistent alias plan: the usage summary of epoch {} was not published: {}", m_passEpochFilter, error.what());
		}
	};

	const auto planGathered = PlanClock::now();
	// Frame placement (SetFramePlacementSource). The graph's first plan takes the host's current version, and keeps it until the
	// host publishes a newer one: the graph then places again from that, in place (BindPersistentFrame). A candidate the version
	// covers takes the version's offset in its block instead of a place in this graph's pools. Covered means:
	// - the planner would pool it on one backend if its first use wrote (eligible),
	// - the version has its entry, found by its identity, or else by its key in this epoch (a rebuilt graph's new resource),
	// - the version placed it,
	// - in the frame's first epoch that uses it, its first use here is a guaranteed write.
	// Its ranks here may differ from the version's (a rebuild that added passes). That is safe for the bytes the version shares
	// with other epochs: they hold it over whole slots of the epochs in between, and other epochs' resources lie in other
	// slots. Only this graph's own candidates in its bytes may now overlap in time: those pairs are checked below, and one of each
	// goes back to its own path. Byte overlaps between epochs are ordered by the alias ledger, as for any placed resource.
	// A plan for any other reason (a structural build at a preparation) while the host has a newer version takes that one: placing
	// from the older version would move the resources several epochs share back to its offsets, under the other epochs' graphs,
	// which placed them from the newer one.
	if (state.summaryOnly) {
		// PublishPlacementSummaryAtBuild: the summary the plan would publish first, and nothing else.
		--state.aliasPlans;
		if (!state.framePlacementTaken && m_framePlacementEnabled && m_framePlacementEnabled()) publishSummary(false);
		return false;
	}
	if (state.framePlacementTaken && m_framePlacementProbe && m_framePlacementProbe() != (state.framePlacement ? state.framePlacement->Id() : 0))
		state.framePlacementTaken = false;
	if (!state.framePlacementTaken) {
		state.framePlacementTaken = true;
		// The summary goes first, and again at the end with the bytes this plan placed (the same uses, so no new computation).
		// The summary that completes the frame's epochs makes the first version at once (GraphHost), so the plan that completes
		// it places from it instead of on its own path.
		if (m_framePlacementEnabled && m_framePlacementEnabled()) publishSummary(false);
		state.framePlacement = m_framePlacementSource ? m_framePlacementSource() : nullptr;
		if (state.framePlacement)
			spdlog::info("Persistent alias plan: epoch {} places from frame placement {}", m_passEpochFilter, state.framePlacement->Id());
	}
	for (auto& c : candidates) {
		c.ownPooled = c.pooled;
		c.ownPoolID = c.poolID;
		c.ownExclusion = c.exclusion;
	}
	// Not placed from the frame placement after all: another epoch may hold it, or its own path takes it.
	const auto withhold = [&](Candidate& c) {
		c.pooled = false;
		c.keepBacking = true;
		c.exclusion = "frame placement does not cover it (used by several epochs)";
		++frameWithheld;
	};
	// A transient a graph of another epoch declares too, which no frame placement places here: it keeps the backing it has (where
	// another graph put it), or gets one of its own. This graph's own pools would reuse its bytes after its last use here, while the
	// other epoch still reads it (or write it before an earlier epoch's contents arrive). The usage summary counts it uncovered, so
	// the host computes a version that places it across the frame.
	uint64_t crossEpochKept = 0;
	const auto keepCrossEpoch = [&] {
		crossEpochKept = 0;
		for (auto& c : candidates) {
			if (!c.otherEpochs || c.framePlaced || !c.pooled || c.manualPool || c.shared || c.version) continue;
			c.pooled = false;
			c.keepBacking = true;
			c.exclusion = "another epoch's graph uses it and no frame placement places it here";
			++crossEpochKept;
		}
	};
	// The frame placement step: places the candidates from a version, starting from their own paths.
	const auto placeFrom = [&](const std::shared_ptr<const FramePlacementVersion>& framePlacement) {
		for (auto& c : candidates) {
			c.pooled = c.ownPooled;
			c.poolID = c.ownPoolID;
			c.exclusion = c.ownExclusion;
			c.framePlaced = c.frameShared = c.keepBacking = c.continues = false;
			c.frameBlock = UINT32_MAX;
			c.offset = 0;
		}
		framePlacedBytes = framePlacedCount = frameWithheld = frameOwnPath = frameMoved = frameFitted = 0;
		if (!framePlacement) {
			keepCrossEpoch();
			return;
		}
		const auto epoch = m_passEpochFilter;
		const auto slot = std::ranges::find_if(framePlacement->Slots(), [&](const auto& s) { return s.epoch == epoch; });
		const bool repeatedSlot = slot != framePlacement->Slots().end() && slot->repeated;
		const uint32_t lastRank = nextRank ? nextRank - 1 : 0;
		for (auto& c : candidates) {
			if (!c.eligible) continue;
			// The resource itself when the version knows it (an object several epochs' graphs share, or this graph's own).
			// Otherwise its key in this epoch: a rebuilt graph's new resource takes its predecessor's entry. An entry several
			// epochs share is the object they shared, so another object under its key is a new resource.
			const auto* entry = framePlacement->FindObject(objectOf(c));
			if (!entry) {
				entry = framePlacement->Find(c.key, epoch);
				if (entry && entry->MultiEpoch()) entry = nullptr;
			}
			if (!entry) {
				// New since the version was computed. If another epoch's graph placed it, that backing stays: moving it would
				// break that graph's executable. Otherwise it goes on this graph's own path until a version covers it.
				if (AliasPlacedByOtherEpoch(c.concrete, epoch)) withhold(c);
				else ++frameOwnPath;
				continue;
			}
			const auto* use = entry->UseIn(epoch);
			if (!use) {
				// The version knows it from other epochs only (it was computed before this build). This graph's own pools see
				// only its own uses, so they could hand its bytes to something live in another epoch. A later version covers it.
				withhold(c);
				continue;
			}
			const bool firstInFrame = entry->uses.front().epoch == epoch;
			const bool lastInFrame = entry->uses.back().epoch == epoch;
			// A resource that keeps its contents across runs needs a version that gave it one lifetime over every run. A version
			// computed before it was marked did not.
			const bool covered = entry->placed && (!firstInFrame || (c.firstUseIsWrite && c.writeGuaranteed))
				&& (!c.acrossRuns || entry->acrossRuns);
			if (covered) {
				c.exclusion = nullptr;
				c.pooled = true;
				c.framePlaced = true;
				c.frameBlock = entry->block;
				c.frameShared = entry->MultiEpoch();
				// Its lifetime in the version covers everything between its earlier use and this one, so no other occupant of its
				// bytes runs in this graph before it.
				c.continues = !firstInFrame || entry->acrossRuns;
				const bool wholeSlot = entry->acrossRuns || (entry->MultiEpoch() && repeatedSlot);
				c.frameFirst = firstInFrame && !wholeSlot ? c.firstUse : 0;
				c.frameLast = lastInFrame && !wholeSlot ? c.lastUse : lastRank;
				c.poolID = framePlacement->Blocks().at(entry->block).poolID;
				c.offset = entry->offset;
				framePlacedBytes += c.sizeBytes;
				++framePlacedCount;
			} else if (entry->MultiEpoch()) {
				// Other epochs use it too, but the version does not place it (a first use that reads, a gated first write).
				withhold(c);
			} else {
				++frameOwnPath;
			}
		}
		// The version kept this graph's candidates apart over the ranks it saw. A rebuilt graph's ranks may differ, so two that
		// share bytes and are live together here cannot both take their offsets. One goes back: a resource of this epoch alone
		// to its own path, one several epochs share to the backing it has.
		std::map<uint32_t, std::vector<uint32_t>> byBlock;
		for (uint32_t i = 0; i < candidates.size(); ++i)
			if (candidates[i].framePlaced) byBlock[candidates[i].frameBlock].push_back(i);
		for (auto& [block, members] : byBlock) {
			std::sort(members.begin(), members.end(), [&](uint32_t l, uint32_t r) { return candidates[l].offset < candidates[r].offset; });
			for (size_t i = 0; i < members.size(); ++i) {
				for (size_t j = i + 1; j < members.size(); ++j) {
					auto& a = candidates[members[i]];
					auto& b = candidates[members[j]];
					if (b.offset >= a.offset + a.sizeBytes) break;
					if (!a.framePlaced) break;
					if (!b.framePlaced || a.frameLast < b.frameFirst || b.frameLast < a.frameFirst) continue;
					auto& moved = a.frameShared && !b.frameShared ? b : !a.frameShared && b.frameShared ? a : b;
					moved.framePlaced = false;
					framePlacedBytes -= moved.sizeBytes;
					--framePlacedCount;
					++frameMoved;
					if (moved.frameShared) {
						moved.frameShared = false;
						moved.continues = false;
						withhold(moved);
					} else {
						moved.continues = false;
						moved.pooled = moved.ownPooled;
						moved.poolID = moved.ownPoolID;
						moved.exclusion = moved.ownExclusion;
						moved.offset = 0;
						++frameOwnPath;
					}
				}
			}
		}
		// A resource this graph keeps where another epoch's graph placed it (withheld) holds those bytes from this graph's start
		// (its contents come from that epoch) to its last use here. The version may give the same bytes to one of this graph's own
		// candidates (it did not know this graph uses it): that candidate goes back to its own path.
		for (const auto& k : candidates) {
			if (!k.keepBacking) continue;
			const AliasHeapGeneration* heldHeap = nullptr;
			uint64_t heldOffset = 0, heldSize = 0;
			if (!FramePlacedWhere(k.concrete, heldHeap, heldOffset, heldSize)) continue;
			for (auto& c : candidates) {
				if (&c == &k || !c.framePlaced || c.frameShared) continue;
				if (framePlacement->PeekBlock(c.frameBlock).get() != heldHeap) continue;
				if (c.offset >= heldOffset + heldSize || heldOffset >= c.offset + c.sizeBytes) continue;
				if (c.frameFirst > k.lastUse) continue;
				c.framePlaced = false;
				framePlacedBytes -= c.sizeBytes;
				--framePlacedCount;
				++frameMoved;
				c.continues = false;
				c.pooled = c.ownPooled;
				c.poolID = c.ownPoolID;
				c.exclusion = c.ownExclusion;
				c.offset = 0;
				++frameOwnPath;
			}
		}
		// Resources the version does not know yet, or does not place here, that this graph alone uses: into the free bytes of the
		// version's blocks, where this graph's own pools would have put them, so a rebuild with new transients does not allocate
		// pools beside the blocks. Free means: no candidate this graph placed from the block is live then (its lifetime here),
		// and no entry the version gives another epoch holds the bytes through this epoch's slot. An entry of this epoch alone
		// that no candidate took is free: its resource is gone. Those that do not fit take their own path.
		std::vector<uint32_t> unplaced;
		for (uint32_t i = 0; i < candidates.size(); ++i) {
			const auto& c = candidates[i];
			if (c.framePlaced || !c.ownPooled || c.shared || c.manualPool || c.acrossRuns || c.keepBacking || c.version || c.otherEpochs
				|| c.firstUse == UINT32_MAX) continue;
			unplaced.push_back(i);
		}
		std::stable_sort(unplaced.begin(), unplaced.end(), [&](uint32_t l, uint32_t r) {
			const auto& a = candidates[l];
			const auto& b = candidates[r];
			return std::tuple(a.firstUse, ~a.sizeBytes) < std::tuple(b.firstUse, ~b.sizeBytes);
		});
		if (!unplaced.empty()) {
			struct Occupant {
				uint64_t begin = 0, end = 0;
				uint32_t first = 0, last = 0;
			};
			std::map<uint32_t, std::vector<Occupant>> occupants;
			const auto& slots = framePlacement->Slots();
			const uint32_t slotBase = slot != slots.end() ? slot->base : 0;
			const uint32_t slotEnd = slot != slots.end() ? slot->base + (std::max)(slot->rankCount, 1u) - 1 : 0;
			std::unordered_set<const FramePlacementVersion::Entry*> taken;
			for (const auto& c : candidates) {
				if (!c.framePlaced) continue;
				occupants[c.frameBlock].push_back({c.offset, c.offset + c.sizeBytes, c.frameFirst, c.frameLast});
				if (const auto* entry = framePlacement->FindObject(objectOf(c))) taken.insert(entry);
				else if (const auto* byKey = framePlacement->Find(c.key, epoch)) taken.insert(byKey);
			}
			// A resource another epoch's graph placed in this version's blocks that this graph keeps where it is (withheld: the version
			// does not give this epoch its use, or that graph fitted it into free bytes) holds its bytes through this graph's whole
			// slot. Its place is where it is now, which the version may not know.
			for (const auto& c : candidates) {
				if (!c.keepBacking) continue;
				const AliasHeapGeneration* heldHeap = nullptr;
				uint64_t heldOffset = 0, heldSize = 0;
				if (!FramePlacedWhere(c.concrete, heldHeap, heldOffset, heldSize)) continue;
				for (uint32_t b = 0; b < framePlacement->Blocks().size(); ++b)
					if (framePlacement->PeekBlock(b).get() == heldHeap)
						occupants[b].push_back({heldOffset, heldOffset + heldSize, 0, lastRank});
				if (const auto* entry = framePlacement->FindObject(objectOf(c))) taken.insert(entry);
			}
			for (const auto& entry : framePlacement->Entries()) {
				if (!entry.placed || taken.contains(&entry)) continue;
				// Another epoch's resource whose lifetime reaches into this slot: its bytes are held through the whole slot here.
				const bool ownEpochOnly = !entry.MultiEpoch() && entry.uses.front().epoch == epoch;
				if (ownEpochOnly || slot == slots.end() || entry.frameLast < slotBase || entry.frameFirst > slotEnd) continue;
				occupants[entry.block].push_back({entry.offset, entry.offset + entry.size, 0, lastRank});
			}
			for (const auto index : unplaced) {
				auto& c = candidates[index];
				for (uint32_t b = 0; b < framePlacement->Blocks().size(); ++b) {
					const auto& block = framePlacement->Blocks()[b];
					if (block.memoryClass != c.resourceClass || block.backendMask != c.backendMask) continue;
					auto& held = occupants[b];
					std::vector<std::pair<uint64_t, uint64_t>> busy;
					for (const auto& o : held)
						if (!(o.last < c.firstUse || c.lastUse < o.first)) busy.emplace_back(o.begin, o.end);
					std::sort(busy.begin(), busy.end());
					uint64_t offset = 0;
					const uint64_t alignment = (std::max<uint64_t>)(c.alignment, 1);
					for (const auto& [begin, end] : busy) {
						if (offset + c.sizeBytes <= begin) break;
						if (end > offset) offset = (end + alignment - 1) / alignment * alignment;
					}
					if (offset + c.sizeBytes > block.capacity) continue;
					held.push_back({offset, offset + c.sizeBytes, c.firstUse, c.lastUse});
					c.exclusion = nullptr;
					c.pooled = true;
					c.framePlaced = true;
					c.frameBlock = b;
					c.frameShared = false;
					c.continues = false;
					c.frameFirst = c.firstUse;
					c.frameLast = c.lastUse;
					c.poolID = block.poolID;
					c.offset = offset;
					framePlacedBytes += c.sizeBytes;
					++framePlacedCount;
					++frameFitted;
					--frameOwnPath;
					break;
				}
			}
		}
		keepCrossEpoch();
	};
	placeFrom(state.framePlacement);
	// Candidates the version leaves on this graph's own path or held back (new transients that did not fit, new resources other
	// epochs share): the summary that says so makes the host compute the next version at once (FramePlacement::Publish), and
	// this plan places from that one. So a rebuild that brings new transients neither allocates pools beside the blocks nor
	// places again a settle time later.
	const auto baseVersion = state.framePlacement;
	if (state.framePlacement && uncoveredCount() > 0 && m_framePlacementSource) {
		publishSummary(true);
		if (auto newer = m_framePlacementSource(); newer && newer != state.framePlacement) {
			spdlog::info("Persistent alias plan: epoch {} places from frame placement {}, made for its {} uncovered candidates", m_passEpochFilter,
				newer->Id(), uncoveredCount());
			state.framePlacement = std::move(newer);
			placeFrom(state.framePlacement);
		}
	}
	// Whether the plan's apply gives a candidate it does not pool, which another graph placed, a backing of its own: one in another
	// graph's own pools, or one an older build of this epoch placed (unless another epoch's graph binds it).
	const auto takesOffPlacement = [&](const Candidate& c) {
		if (!AliasPlacedAnywhere(c.concrete)) return false;
		if (inOtherOwnPool(c.concrete)) return true;
		return !c.keepBacking && !c.otherEpochs && !AliasPlacedByOtherEpoch(c.concrete, m_passEpochFilter);
	};
	// Placed before the graph is attached: resources the running graphs bind (materialized by someone else) must stay where they
	// are. The first offender under a version, or null.
	const auto movesForeign = [&](const std::shared_ptr<const FramePlacementVersion>& version) -> const Candidate* {
		for (const auto& c : candidates) {
			auto* backed = dynamic_cast<BackedResource*>(c.concrete);
			if (!backed || !backed->IsMaterialized() || state.ownMaterialized.contains(c.concrete) || state.aliasPlaced.contains(c.concrete))
				continue;
			if (c.pooled && c.framePlaced) {
				if (c.shared) return &c;
				// Without allocating: a version the set may decline allocates nothing (a heap it would need is a move anyway).
				const auto heap = version->PeekBlock(c.frameBlock);
				if (!heap || !AtFramePlacement(c.concrete, heap.get(), c.offset, backed->GetBackingGeneration())) return &c;
			} else if (c.pooled) {
				return &c;
			} else if (!c.version && takesOffPlacement(c)) {
				return &c;
			}
		}
		return nullptr;
	};
	// A new version made for this graph's uncovered candidates may pack the frame anew, moving resources the running graphs bind.
	// Before the attach the set may decline it (no graph of the set placed from it yet): this graph keeps the version it had, and
	// its new transients go to its own pool until a later version takes them in place. Only when the old version withholds nothing:
	// a resource this graph newly shares with another epoch needs the new version, or the other epoch's graph reuses its bytes
	// while this one still reads it (wrong contents until the new version comes). Then the placement waits for the attach.
	if (state.placingAtBuild && baseVersion && state.framePlacement != baseVersion && m_framePlacementDecline) {
		if (const auto* offender = movesForeign(state.framePlacement)) {
			const auto pending = state.framePlacement;
			const auto offenderName = offender->concrete->GetName();
			placeFrom(baseVersion);
			if (frameWithheld == 0 && m_framePlacementDecline()) {
				spdlog::info("Persistent alias plan: epoch {} keeps frame placement {} (frame placement {} would move '{}', which running graphs "
					"bind); its new transients go to its own pool for now", m_passEpochFilter, baseVersion->Id(), pending->Id(), offenderName);
				state.framePlacement = baseVersion;
				state.atBuildOwnPools = true;
			} else {
				placeFrom(pending);
			}
		}
	} else if (state.placingAtBuild && baseVersion && state.framePlacement == baseVersion && m_framePlacementDecline && uncoveredCount() > 0
		&& frameWithheld == 0 && m_framePlacementDecline()) {
		// An earlier graph of the set declined the version these candidates would need: the set stays on its base, and this
		// graph's new transients go to its own pool too.
		state.atBuildOwnPools = true;
	}
	const auto framePlacement = state.framePlacement;
	const auto planFramed = PlanClock::now();

	// A version goes only where a frame placement puts it. Other graphs may bind it, and this graph's pools see only its own
	// uses. Otherwise it keeps the backing it has: memory of its own from whoever made it, or the frame placement another
	// epoch's graph gave it.
	for (auto& c : candidates) {
		if (!c.version) continue;
		if (c.versionedEntry < state.entries.size()) state.entries[c.versionedEntry].aliasPlannedVersion = c.version->number;
		if (c.framePlaced) continue;
		if (!c.exclusion) {
			++excludedCount;
			c.exclusion = "a version is placed only by a frame placement";
		}
		c.pooled = false;
		c.keepBacking = true;
	}

	excludedCount = 0;
	for (const auto& c : candidates)
		if (c.exclusion) ++excludedCount;

	// The resources whose bindings continue from an earlier execution of the frame. Bindings made from here on read it.
	state.aliasContinues.clear();
	for (const auto& c : candidates)
		if (c.framePlaced && c.continues) state.aliasContinues.insert(c.concrete);

	// Placed before the graph is attached (PlaceDeferredAliasesAtBuild): the running graphs bind what is materialized by others,
	// so the plan goes on only if it moves none of it, and allocates nothing in the arena's pools, which the running graphs share.
	// Otherwise nothing has changed yet: the placement is left to the attach.
	if (state.placingAtBuild) {
		const char* refusal = nullptr;
		const Candidate* refused = nullptr;
		for (const auto& c : candidates) {
			auto* backed = dynamic_cast<BackedResource*>(c.concrete);
			if (!backed) continue;
			const bool foreign = backed->IsMaterialized() && !state.ownMaterialized.contains(c.concrete) && !state.aliasPlaced.contains(c.concrete);
			// The graph's own pool: only after it declined a new version, and only a pool no other graph places into (growing it
			// would leave their placements in an old generation).
			if (c.pooled && !c.framePlaced) {
				// A resource another graph binds (an older build of this epoch) cannot move to this graph's pool before the attach.
				if (foreign) refusal = "it would move a resource another graph binds into the graph's own pool";
				else if (!state.atBuildOwnPools || c.manualPool || c.shared || m_frameAliasArena) refusal = "a transient goes to the graph's own pools";
				else if (state.aliasArena) {
					std::lock_guard arenaLock(state.aliasArena->mutex);
					const auto found = state.aliasArena->pools.find(c.poolID);
					if (found != state.aliasArena->pools.end())
						for (const auto& [owner, bytes] : found->second.required)
							if (owner != &state && bytes) refusal = "a transient goes to a pool other graphs place into";
				}
			}
			else if (!foreign) continue;
			else if (c.shared && c.pooled) refusal = "a resource shared between devices is placed";
			else if (c.pooled) {
				const auto heap = framePlacement->AcquireBlock(c.frameBlock);
				if (!AtFramePlacement(c.concrete, heap.get(), c.offset, backed->GetBackingGeneration()))
					refusal = "it would move a resource another graph binds";
			} else if (!c.version && takesOffPlacement(c))
				refusal = "it would take a resource another graph placed off its placement";
			if (refusal) {
				refused = &c;
				break;
			}
		}
		if (refusal) {
			state.atBuildRefusal = std::string(refusal) + " ('" + refused->concrete->GetName() + "')";
			// The plan at the attach takes the host's version then.
			state.framePlacementTaken = false;
			state.framePlacement.reset();
			state.aliasContinues.clear();
			return false;
		}
	}
	// Pack per pool, into the arena's pools (shared with the other graphs of a host): each graph from offset 0.
	if (!state.aliasArena) state.aliasArena = PersistentAliasArena::Create();
	std::lock_guard arenaLock(state.aliasArena->mutex);
	auto& aliasPools = state.aliasArena->pools;
	state.aliasArena->stale.erase(&state);
	std::map<uint64_t, std::vector<uint32_t>> byPool;
	std::unordered_set<uint64_t> plannedPools;
	// The frame alias arena's ranges this plan reserves (SetFrameAliasRegion), by pool id. They replace the previous plan's at the end.
	std::unordered_map<uint64_t, std::shared_ptr<FrameAliasArena::Reservation>> newFrameRegions;
	auto heapOf = [&](uint64_t poolID) -> const std::shared_ptr<const AliasHeapGeneration>& {
		if (const auto found = newFrameRegions.find(poolID); found != newFrameRegions.end()) return found->second->Heap();
		return aliasPools.at(poolID).heap;
	};
	// The frame placement's candidates are placed already. They are kept by block.
	std::map<uint64_t, std::vector<uint32_t>> byFrameBlock;
	for (uint32_t i = 0; i < candidates.size(); ++i) {
		if (!candidates[i].pooled) continue;
		if (candidates[i].framePlaced) byFrameBlock[candidates[i].poolID].push_back(i);
		else byPool[candidates[i].poolID].push_back(i);
	}
	std::vector<std::pair<uint32_t, uint32_t>> edges; // (logical slot, logical slot)
	uint64_t independentBytes = 0, reservedBytes = 0;
	size_t overlapPairs = 0;
	// Placement orderings between byte-overlapping occupants of one pool or block.
	auto orderOverlaps = [&](const std::vector<uint32_t>& members) {
		for (size_t i = 0; i < members.size(); ++i) for (size_t j = i + 1; j < members.size(); ++j) {
			const auto& a = candidates[members[i]]; const auto& b = candidates[members[j]];
			if (a.offset >= b.offset + b.sizeBytes || b.offset >= a.offset + a.sizeBytes) continue;
			const auto& earlier = a.lastUse < b.firstUse ? a : b;
			const auto& later = a.lastUse < b.firstUse ? b : a;
			if (!(earlier.lastUse < later.firstUse))
				throw std::logic_error("Persistent alias plan produced overlapping lifetimes for '" + a.concrete->GetName() + "' and '" + b.concrete->GetName() + "'");
			++overlapPairs;
			// Which transients share memory, for checking a plan by hand (with alias logging on).
			if (logging)
				spdlog::info("Persistent alias overlap: '{}' [{}, {}) ranks {}-{} then '{}' [{}, {}) ranks {}-{}", earlier.concrete->GetName(),
					earlier.offset, earlier.offset + earlier.sizeBytes, earlier.firstUse, earlier.lastUse, later.concrete->GetName(), later.offset,
					later.offset + later.sizeBytes, later.firstUse, later.lastUse);
			for (const auto u : earlier.users) for (const auto v : later.users)
				edges.emplace_back(structure.passes[u].preparedPassIndex, structure.passes[v].preparedPassIndex);
		}
	};
	for (const auto& [blockPool, members] : byFrameBlock) orderOverlaps(members);
	for (auto& [poolID, members] : byPool) {
		std::sort(members.begin(), members.end(), [&](uint32_t l, uint32_t r) {
			const auto& a = candidates[l]; const auto& b = candidates[r];
			if (a.firstUse != b.firstUse) return a.firstUse < b.firstUse;
			if (a.sizeBytes != b.sizeBytes) return a.sizeBytes > b.sizeBytes;
			if (a.lastUse != b.lastUse) return a.lastUse < b.lastUse;
			return a.concrete->GetGlobalResourceID() < b.concrete->GetGlobalResourceID();
		});
		std::vector<alias::PackingItem> items;
		items.reserve(members.size());
		for (const auto index : members) {
			const auto& c = candidates[index];
			independentBytes += c.sizeBytes;
			items.push_back({c.sizeBytes, c.alignment, c.firstUse, c.lastUse, c.concrete->GetGlobalResourceID()});
		}
		auto packed = packingStrategy == AutoAliasPackingStrategy::BranchAndBound
			? alias::PackBeamSearch(items) : alias::PackGreedySweepLine(items);
		if (packingStrategy == AutoAliasPackingStrategy::GreedySweepLine && members.size() > 1) {
			// The sweep places in this order, so among resources that start together the ones placed first take the low
			// offsets. Placing the longest-lived first keeps the memory the others free as one range rather than split around
			// a survivor: path tracing's culling frontiers, freed before the scene build, then hold its scratch and CLAS
			// arena. Neither order wins everywhere, so the smaller heap is kept.
			std::vector<size_t> order(members.size());
			for (size_t i = 0; i < order.size(); ++i) order[i] = i;
			std::stable_sort(order.begin(), order.end(), [&](size_t l, size_t r) {
				const auto& a = items[l]; const auto& b = items[r];
				if (a.firstUse != b.firstUse) return a.firstUse < b.firstUse;
				if (a.lastUse != b.lastUse) return a.lastUse > b.lastUse;
				return a.sizeBytes > b.sizeBytes;
			});
			std::vector<alias::PackingItem> reordered(items.size());
			for (size_t i = 0; i < order.size(); ++i) reordered[i] = items[order[i]];
			const auto longestFirst = alias::PackGreedySweepLine(reordered);
			if (longestFirst.heapEnd < packed.heapEnd) {
				packed.heapEnd = longestFirst.heapEnd;
				packed.alignment = longestFirst.alignment;
				for (size_t i = 0; i < order.size(); ++i) packed.offsets[order[i]] = longestFirst.offsets[i];
			}
		}
		for (size_t i = 0; i < members.size(); ++i) candidates[members[i]].offset = packed.offsets[i];
		const uint64_t heapEnd = packed.heapEnd, poolAlignment = packed.alignment;
		requiredBytes += heapEnd;
		requiredByClass[candidates[members.front()].resourceClass & 3u] += heapEnd;
		if (heapEnd == 0) continue;
		const bool sharedPool = candidates[members.front()].shared;
		if (m_frameAliasArena && !sharedPool) {
			// A range of the frame alias arena for this pool's transients, reserved for the graph's span. The packing above stays;
			// it moves to the range's start. A forced re-placement takes bytes apart from the graph's current ones (lane 0).
			const uint32_t lane = state.forceNewAliasHeaps ? 0u : m_frameAliasLane;
			auto region = m_frameAliasArena->Reserve(candidates[members.front()].resourceClass, m_frameAliasSpan, lane, heapEnd, poolAlignment,
				"epoch " + std::to_string(m_passEpochFilter) + " pool " + std::to_string(poolID & 0xFFFF));
			for (const auto index : members) candidates[index].offset += region->Offset();
			reservedBytes += heapEnd;
			newFrameRegions[poolID] = std::move(region);
		} else {
		// Pool heap: grows with headroom, and shrinks to that headroom when the graphs placed in it need less than half of it. A
		// replaced generation stays alive through the leases held by in-flight publications and by the placements made in it.
		auto& pool = aliasPools[poolID];
		pool.shared = sharedPool;
		plannedPools.insert(poolID);
		pool.idleSinceFrame = 0;
		pool.required[&state] = heapEnd;
		const auto needed = PersistentAliasArena::Required(pool);
		const auto withHeadroom = static_cast<uint64_t>(std::ceil(static_cast<double>(needed) * headroom));
		const bool initial = sharedPool ? !pool.heap : !static_cast<bool>(pool.allocation);
		const bool grow = !initial && (heapEnd > pool.capacityBytes || poolAlignment > pool.alignment);
		const bool shrink = !initial && !grow && pool.capacityBytes > 2 * withHeadroom;
		const bool forced = !initial && !grow && !shrink && state.forceNewAliasHeaps;
		if (initial || grow || shrink || forced) {
			uint64_t capacity = needed;
			if (forced) capacity = (std::max)(capacity, pool.capacityBytes);
			else if (shrink) capacity = withHeadroom;
			else if (!initial && pool.capacityBytes)
				capacity = (std::max)(capacity, static_cast<uint64_t>(std::ceil(static_cast<double>(pool.capacityBytes) * headroom)));
			if (sharedPool) {
				// A D3D12 heap created shared and imported on the Vulkan device. The earlier generation's memory stays alive
				// through its heap generation, which every placement in it and every binding of one holds.
				const DeviceRegistryEntry* d3d12 = nullptr;
				const DeviceRegistryEntry* vulkan = nullptr;
				for (const auto& entry : m_backendDevices) {
					if (entry.backend == rhi::Backend::D3D12 && !d3d12) d3d12 = &entry;
					if (entry.backend == rhi::Backend::Vulkan && !vulkan) vulkan = &entry;
				}
				if (!d3d12 || !vulkan) throw std::logic_error("A shared alias pool needs one D3D12 and one Vulkan device");
				rhi::HeapDesc heapDesc{};
				heapDesc.sizeBytes = (capacity + 65535ull) / 65536ull * 65536ull;
				heapDesc.alignment = 64ull * 1024ull;
				heapDesc.memory = rhi::HeapType::DeviceLocal;
				heapDesc.flags = rhi::HeapFlags::Shared | (candidates[members.front()].resourceClass == 1u
					? rhi::HeapFlags::AllowOnlyBuffers : rhi::HeapFlags::AllowOnlyNonRtDsTextures);
				heapDesc.debugName = "RenderGraph shared alias pool";
				auto heaps = std::make_shared<SharedAliasHeaps>();
				heaps->d3d12Device = static_cast<uint8_t>(d3d12->id);
				heaps->vulkanDevice = static_cast<uint8_t>(vulkan->id);
				rhi::ExternalHandle handle{};
				const bool created = [&] {
					BT_ZONE_SCOPE("ORG.Persistent.RealizeSharedAliasPool");
					return rhi::IsOk(d3d12->device.CreateHeap(heapDesc, heaps->d3d12))
						&& rhi::IsOk(rhi::dx12::export_shared_heap(d3d12->device, heaps->d3d12.Get(), handle))
						&& rhi::IsOk(rhi::vulkan::import_d3d12_heap(vulkan->device, handle, heapDesc, heaps->vulkan));
				}();
				if (!created) throw std::runtime_error("Failed to allocate shared persistent alias pool memory");
				capacity = heapDesc.sizeBytes;
				pool.capacityBytes = capacity;
				pool.alignment = poolAlignment;
				++pool.generation;
				pool.heap = AliasHeapGeneration::CaptureShared(std::move(heaps), pool.generation);
			} else {
			rhi::ma::AllocationDesc allocDesc{};
			allocDesc.heapType = rhi::HeapType::DeviceLocal;
			allocDesc.flags = rhi::ma::AllocationFlagCanAlias;
			allocDesc.extraHeapFlags = alias::AliasClassHeapFlags(candidates[members.front()].resourceClass);
			rhi::ResourceAllocationInfo allocInfo{};
			allocInfo.alignment = poolAlignment;
			allocInfo.sizeInBytes = capacity;
			TrackedHandle fresh;
			AllocationTrackDesc trackDesc(0);
			trackDesc.attach
				.Set<MemoryStatisticsComponents::ResourceName>({"RenderGraph Alias Pool"})
				.Set<MemoryStatisticsComponents::ResourceType>({rhi::ResourceType::Unknown})
				.Set<MemoryStatisticsComponents::ResourceUsage>({"RenderGraph alias pools"})
				.Set<MemoryStatisticsComponents::AliasingPool>({poolID});
			const auto result = [&] {
				BT_ZONE_SCOPE("ORG.Persistent.RealizeAliasPool");
				return DeviceManager::GetInstance().AllocateMemoryTracked(allocDesc, allocInfo, fresh, trackDesc);
			}();
			if (!rhi::IsOk(result)) throw std::runtime_error("Failed to allocate persistent alias pool memory");
			// Frame targets and per-frame working buffers: over the OS budget, the costliest to leave in system memory.
			DeviceManager::GetInstance().SetResidencyPriority(fresh, rhi::ResidencyPriorityHigh);
			if (pool.allocation) DeletionManager::GetInstance().MarkForDelete(std::move(pool.allocation), "RenderGraph alias pool (replaced)");
			pool.allocation = std::move(fresh);
			pool.capacityBytes = capacity;
			pool.alignment = poolAlignment;
			++pool.generation;
			pool.heap = AliasHeapGeneration::Capture(pool.allocation, pool.generation);
			}
			// The other graphs placed in it keep the earlier generation until they place again.
			for (const auto& [owner, bytes] : pool.required)
				if (owner != &state) state.aliasArena->stale.insert(owner);
			spdlog::info("Persistent alias pool {}: pool={:#x}{} capacity={} required={} alignment={} placements={} generation={}",
				initial ? "allocated" : forced ? "reallocated (forced)" : shrink ? "shrank" : "grew", poolID, sharedPool ? " (shared between devices)" : "",
				capacity, heapEnd, poolAlignment, members.size(), pool.generation);
			basic_telemetry::AddCounter("ORG.Persistent.AliasPoolAllocations");
		}
		pool.resourceClass = candidates[members.front()].resourceClass;
		reservedBytes += pool.capacityBytes;
		}
		orderOverlaps(members);
	}
	std::sort(edges.begin(), edges.end());
	edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
	// A pool this plan places nothing into idles once no graph places into it; RetireIdlePersistentAliasPools frees it once it has
	// idled long enough.
	for (auto& [poolID, pool] : aliasPools) {
		if (plannedPools.contains(poolID)) continue;
		pool.required.erase(&state);
		if (pool.required.empty() && !pool.idleSinceFrame) pool.idleSinceFrame = (std::max)(state.frameNumber, uint64_t{1});
	}

	const auto planPacked = PlanClock::now();
	// Apply: orderings, then bindings.
	bool changed = false;
	if (edges != state.aliasEdges) {
		changed = true;
		edit.ClearPlacementOrderings();
		std::unordered_map<uint32_t, persistent::PassId> idBySlot;
		for (const auto& main : state.mainPasses) idBySlot.emplace(main.id.index, main.id);
		for (const auto& [from, to] : edges) {
			const auto a = idBySlot.find(from), b = idBySlot.find(to);
			if (a == idBySlot.end() || b == idBySlot.end()) throw std::logic_error("Persistent alias planning: ordering references an unknown pass slot");
			edit.AddPlacementOrdering(a->second, b->second);
		}
		state.aliasEdges = edges;
	}
	size_t placed = 0, rematerialized = 0, standalone = 0;
	auto dematerialize = [](Resource* concrete) {
		if (auto* texture = dynamic_cast<PixelBuffer*>(concrete)) texture->Dematerialize();
		else if (auto* buffer = dynamic_cast<BufferBase*>(concrete)) buffer->Dematerialize();
	};
	auto rebind = [&](Resource* concrete) {
		for (const auto entryIndex : entriesByConcrete[concrete]) {
			if (!BindEntry(state, edit, entryIndex))
				throw std::runtime_error("Persistent alias planning could not bind '" + concrete->GetName() + "'");
			for (const auto subscriber : state.entries[entryIndex].subscribers) state.mainPasses[subscriber].slotsDirty = true;
		}
		changed = true;
	};
	state.adoptedBackings.clear();
	for (auto& c : candidates) {
		auto* backed = dynamic_cast<BackedResource*>(c.concrete);
		if (!backed) continue;
		if (c.pooled) {
			++placed;
			// A frame placement's block is allocated by the first graph that places into it.
			const auto acquireBegin = PlanClock::now();
			const std::shared_ptr<const AliasHeapGeneration> frameHeap = c.framePlaced ? framePlacement->AcquireBlock(c.frameBlock) : nullptr;
			acquireMs += std::chrono::duration<double, std::milli>(PlanClock::now() - acquireBegin).count();
			const auto& heap = c.framePlaced ? frameHeap : heapOf(c.poolID);
			const State::AliasPlacementRecord record{c.poolID, c.offset, c.sizeBytes, heap.get(), c.framePlaced && c.continues};
			const auto previous = state.aliasPlaced.find(c.concrete);
			if (c.shared) {
				// Placed on both devices. A new placement replaces the representations, so nothing is dematerialized first.
				const auto set = c.concrete->AcquireAPIRepresentationSet();
				if (previous != state.aliasPlaced.end() && previous->second == record && set && set->placement.heap == heap
					&& set->placement.offset == c.offset) continue;
				PlaceSharedRepresentations(*c.concrete, heap, c.poolID, c.offset, c.sizeBytes);
				TrackTransientFrameResource(c.concrete);
				state.aliasPlaced[c.concrete] = record;
				rebind(c.concrete);
				continue;
			}
			const bool atFramePlacement = c.framePlaced && backed->IsMaterialized()
				&& AtFramePlacement(c.concrete, heap.get(), c.offset, backed->GetBackingGeneration());
			if (previous != state.aliasPlaced.end() && previous->second == record && backed->IsMaterialized()
				&& (!c.framePlaced || atFramePlacement)) {
				// Where it was. Whether it is this graph's own can still change: a new version may share it with another epoch now
				// (or no longer). A shared one is left at the graph's end; giving it back then would take it from the other epochs.
				if (c.framePlaced) {
					if (c.frameShared || c.version) state.aliasPlacementOwners.erase(c.concrete);
					else state.aliasPlacementOwners[c.concrete] = {c.concrete->weak_from_this(), backed->GetBackingGeneration()};
				}
				continue;
			}
			if (atFramePlacement) {
				// Already at its frame placement, by another epoch's graph. Its backing stays, so that graph's bindings stay valid.
				// This graph's bindings may still name the backing it had before that graph placed it. Then they are bound again
				// here, in this structural edit. Otherwise the next recording would meet the new placement as a plain binding
				// change, which the executable refuses.
				const bool movedSincePlan = previous == state.aliasPlaced.end() || !(previous->second == record);
				state.aliasPlaced[c.concrete] = record;
				NoteAliasPlaced(c.concrete, true, m_passEpochFilter);
				// A version outlives the graph, so the graph's end never gives it back.
				if (c.frameShared || c.version) state.aliasPlacementOwners.erase(c.concrete);
				else state.aliasPlacementOwners[c.concrete] = {c.concrete->weak_from_this(), backed->GetBackingGeneration()};
				// Claimed either way: the graph that placed it (an older build of an epoch that used it alone, under a version that
				// kept it where it is) must not give it back at its end, since this graph binds it now.
				ClaimFramePlaced(c.concrete, &state);
				// A version's slot is bound by each preparation (BindVersion), never by the publication.
				if (c.version) continue;
				bool unbound = false;
				for (const auto entryIndex : entriesByConcrete[c.concrete]) unbound = unbound || !state.entries[entryIndex].isBound;
				if (unbound || movedSincePlan) rebind(c.concrete);
				continue;
			}
			if (backed->IsMaterialized()) { dematerialize(c.concrete); ++rematerialized; }
			if (c.texture) {
				PixelBuffer::MaterializeOptions options{};
				options.aliasPlacement = TextureAliasPlacement{heap, c.offset, c.poolID};
				static_cast<PixelBuffer*>(c.concrete)->Materialize(&options);
			} else {
				BufferBase::MaterializeOptions options{};
				options.aliasPlacement = BufferAliasPlacement{heap, c.offset, c.poolID};
				static_cast<BufferBase*>(c.concrete)->Materialize(&options);
			}
			++materializedCount;
			TrackTransientFrameResource(c.concrete);
			state.aliasPlaced[c.concrete] = record;
			NoteAliasPlaced(c.concrete, true, m_passEpochFilter);
			if (c.framePlaced)
				NoteFramePlaced(c.concrete, heap.get(), c.offset, backed->GetBackingGeneration(), &state, c.sizeBytes, m_passEpochFilter);
			// A resource several epochs use is not this graph's alone: the graph's end leaves its placement to the others. A version
			// outlives the graph, so its end never gives one back.
			if (c.frameShared || c.version) state.aliasPlacementOwners.erase(c.concrete);
			else state.aliasPlacementOwners[c.concrete] = {c.concrete->weak_from_this(), backed->GetBackingGeneration()};
			// A version's slot is bound by each preparation, which takes the new backing (BindVersion).
			if (!c.version) rebind(c.concrete);
		} else if (c.version) {
			// A version this graph does not place keeps its backing: memory of its own, or the frame placement another epoch's
			// graph gave it. Preparations bind it as it is, so nothing is bound here.
			state.aliasPlacementOwners.erase(c.concrete);
			state.aliasPlaced.erase(c.concrete);
			if (!backed->IsMaterialized()) {
				MaterializePersistentStandalone(std::span<Resource* const>{&c.concrete, 1}, false);
				++standalone;
			}
		} else if (c.keepBacking && backed->IsMaterialized() && !state.aliasPlaced.contains(c.concrete) && !inOtherOwnPool(c.concrete)) {
			// Placed by another graph and shared with it: bound as it is, in this structural edit, so the executable admits its
			// current placement. If that graph moves it, this one re-places (adoptedBackings).
			state.aliasPlacementOwners.erase(c.concrete);
			rebind(c.concrete);
			state.adoptedBackings[c.concrete] = {c.concrete->weak_from_this(), backed->GetBackingGeneration()};
		} else {
			state.aliasPlacementOwners.erase(c.concrete);
			const bool own = state.aliasPlaced.erase(c.concrete) != 0;
			if (own && backed->IsMaterialized()) { dematerialize(c.concrete); ++rematerialized; }
			// Placed by an older build of this epoch, whose pools go with it, or in another graph's own pools, and standalone in this
			// one: it leaves that placement for a backing of its own. A frame placement another epoch's graph made, or a backing of
			// its own another epoch's graph binds, stays: moving it would take it from under that graph's executable.
			const bool otherHolds = !own && backed->IsMaterialized() && !inOtherOwnPool(c.concrete)
				&& (AliasPlacedByOtherEpoch(c.concrete, m_passEpochFilter) || c.otherEpochs);
			if (!otherHolds && NoteAliasPlaced(c.concrete, false) && backed->IsMaterialized()) {
				dematerialize(c.concrete);
				++rematerialized;
				NoteAliasPlacementTaken();
			}
			if (!backed->IsMaterialized()) {
				MaterializePersistentStandalone(std::span<Resource* const>{&c.concrete, 1}, false);
				++standalone;
				rebind(c.concrete);
			} else {
				// Materialized by someone else (another graph's placement or standalone backing): bound where it is now, in this
				// structural edit. A later preparation's binding refresh could not admit a placement change, which only a structural
				// edit may make. If its backing changes after this, this graph re-places (adoptedBackings).
				bool stale = false;
				for (const auto entryIndex : entriesByConcrete[c.concrete]) {
					auto& entry = state.entries[entryIndex];
					stale = stale || !entry.isBound || !(RotationKeyOf(entry) == entry.bound);
				}
				if (stale) rebind(c.concrete);
			}
			// Another epoch's graph may place it later (a frame placement that covers it there): followed the same way. Only what
			// ORG graphs place or share: a backing its maker materialized stays where it is, and each preparation checks the list.
			if (backed->IsMaterialized() && (c.otherEpochs || (!state.ownMaterialized.contains(c.concrete) && AliasPlacedAnywhere(c.concrete))))
				state.adoptedBackings[c.concrete] = {c.concrete->weak_from_this(), backed->GetBackingGeneration()};
		}
	}
	state.aliasRematerializations += rematerialized;
	{
		const auto ms = [](PlanClock::time_point a, PlanClock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
		const auto planApplied = PlanClock::now();
		// Where a slow plan's time went (a plan at a build or an attach, a re-placement on a preparation).
		if (ms(planBegin, planApplied) > 2.0)
		spdlog::info("Persistent alias plan timing: epoch {} total {:.2f} ms: gather {:.2f}, frame placement {:.2f}, pack {:.2f}, apply {:.2f} "
			"(acquire {:.2f}, {} materialized)", m_passEpochFilter, ms(planBegin, planApplied), ms(planBegin, planGathered),
			ms(planGathered, planFramed), ms(planFramed, planPacked), ms(planPacked, planApplied), acquireMs, materializedCount);
	}
	m_framePlacementVersion.store(framePlacement ? framePlacement->Id() : 0, std::memory_order_release);
	m_framePlacementPlaced.store(framePlacedCount, std::memory_order_release);
	m_framePlacementPlanned.store(true, std::memory_order_release);
	// The previous plan's ranges go. Its placements that this plan did not replace keep their memory through their leases.
	state.frameRegions = std::move(newFrameRegions);
	// The usage summary, with the bytes this plan placed.
	publishSummary(true);
	BT_PLOT("ORG.Persistent.AliasPlacements", static_cast<int64_t>(placed));
	BT_PLOT("ORG.Persistent.AliasRequiredBytes", static_cast<int64_t>(requiredBytes));
	BT_PLOT("ORG.Persistent.AliasReservedBytes", static_cast<int64_t>(reservedBytes));
	BT_PLOT("ORG.Persistent.AliasIndependentBytes", static_cast<int64_t>(independentBytes));
	basic_telemetry::AddCounter("ORG.Persistent.AliasPlans");
	// The AutoAlias panel's snapshot (GetAutoAliasDebugSnapshot): this plan's candidates, exclusions and pool layouts.
	// Kept for every plan, not only while autoAliasBuildDebugData is set: plans run only on structural builds, so a
	// panel opened later would otherwise show nothing until the next one.
	autoAliasModeLastFrame = mode;
	autoAliasPackingStrategyLastFrame = packingStrategy;
	{
		AutoAliasPlannerStats stats;
		std::map<std::string, size_t> reasons;
		std::map<uint64_t, AutoAliasPoolDebug> pools;
		autoAliasExcludedResources.clear();
		for (const auto& c : candidates) {
			++stats.candidatesSeen;
			stats.candidateBytes += c.sizeBytes;
			if (c.exclusion) {
				++stats.excluded;
				++reasons[c.exclusion];
				autoAliasExcludedResources.push_back({c.concrete->GetGlobalResourceID(), c.concrete->GetName(), c.sizeBytes, c.exclusion});
				continue;
			}
			if (!c.pooled) continue;
			if (c.manualPool) ++stats.manuallyAssigned;
			else { ++stats.autoAssigned; stats.autoAssignedBytes += c.sizeBytes; }
			auto& pool = pools[c.poolID];
			pool.poolID = c.poolID;
			pool.requiredBytes = (std::max)(pool.requiredBytes, c.offset + c.sizeBytes);
			pool.ranges.push_back({c.concrete->GetGlobalResourceID(), c.concrete->GetName(), c.offset, c.offset + c.sizeBytes,
				c.sizeBytes, c.firstUse, c.lastUse, false});
		}
		autoAliasPoolDebug.clear();
		for (auto& [poolID, pool] : pools) {
			if (const auto found = aliasPools.find(poolID); found != aliasPools.end()) pool.reservedBytes = found->second.capacityBytes;
			std::sort(pool.ranges.begin(), pool.ranges.end(), [](const auto& a, const auto& b) { return a.startByte < b.startByte; });
			for (size_t i = 0; i < pool.ranges.size(); ++i)
				for (size_t j = i + 1; j < pool.ranges.size() && pool.ranges[j].startByte < pool.ranges[i].endByte; ++j)
					pool.ranges[i].overlapsByteRange = pool.ranges[j].overlapsByteRange = true;
			autoAliasPoolDebug.push_back(std::move(pool));
		}
		stats.pooledIndependentBytes = independentBytes;
		stats.pooledActualBytes = requiredBytes;
		stats.pooledSavedBytes = independentBytes > requiredBytes ? independentBytes - requiredBytes : 0;
		autoAliasPlannerStats = stats;
		autoAliasExclusionReasonSummary.clear();
		for (const auto& [reason, count] : reasons) autoAliasExclusionReasonSummary.push_back({reason, count});
		std::sort(autoAliasExclusionReasonSummary.begin(), autoAliasExclusionReasonSummary.end(),
			[](const auto& a, const auto& b) { return a.count > b.count; });
	}
	// Persistent memory changes only with a plan, so the VRAM dump is written per plan rather than per frame.
	if (m_getRenderGraphVramDumpEnabled && m_getRenderGraphVramDumpEnabled()) {
		BT_ZONE_SCOPE("ORG.Persistent.WriteVramUsageDebugDump");
		WriteVramUsageDebugDump(state.frameIndex);
	}
	if (rematerialized) basic_telemetry::AddCounter("ORG.Persistent.AliasRematerializations", static_cast<int64_t>(rematerialized));
	if (logging || state.aliasPlans <= 2 || rematerialized)
		spdlog::info("Persistent alias plan #{}: epoch={} inHostStream={} candidates={} placed={} excluded={} pools={} overlapPairs={} "
			"orderings={} rematerialized={} standalone={} independentMB={:.2f} requiredMB={:.2f} reservedMB={:.2f} rebuild={} "
			"framePlacement={} framePlaced={} framePlacedMB={:.2f} frameWithheld={} frameOwnPath={} frameMoved={} frameFitted={} crossEpochKept={}",
			state.aliasPlans, m_passEpochFilter, PassEpochInHostStream(), candidates.size(), placed, excludedCount, byPool.size(),
			overlapPairs, edges.size(), rematerialized, standalone, independentBytes / (1024.0 * 1024.0), requiredBytes / (1024.0 * 1024.0),
			reservedBytes / (1024.0 * 1024.0), changed, framePlacement ? framePlacement->Id() : 0, framePlacedCount,
			framePlacedBytes / (1024.0 * 1024.0), frameWithheld, frameOwnPath, frameMoved, frameFitted, crossEpochKept);
	// Which queue slots this epoch's batches use, once per plan under logging: with inHostStream, which of them are in the host's stream.
	if (logging) {
		std::vector<uint32_t> queues;
		for (const auto& batch : graph.batches)
			if (std::ranges::find(queues, batch.queue) == queues.end()) queues.push_back(batch.queue);
		std::string list;
		for (const auto queue : queues) list += (list.empty() ? "" : ",") + std::to_string(queue);
		spdlog::info("Persistent alias epoch {}: batches={} queue slots={} inHostStream={}", m_passEpochFilter, graph.batches.size(), list,
			PassEpochInHostStream());
	}
	if (logging) for (const auto& c : candidates) if (c.exclusion)
		spdlog::info("Persistent alias exclusion: '{}' reason='{}' bytes={}", c.concrete->GetName(), c.exclusion, c.sizeBytes);
	// Graph resources never offered to the planner: not marked aliasable, or a manual pool with auto aliasing off.
	if (logging) for (auto* concrete : notAliasable) {
		rhi::ResourceDesc desc{};
		if (auto* texture = dynamic_cast<PixelBuffer*>(concrete)) desc = alias::AliasTextureResourceDesc(texture->GetDescription());
		else if (auto* buffer = dynamic_cast<BufferBase*>(concrete))
			desc = alias::AliasBufferResourceDesc(buffer->GetBufferSize(), buffer->IsUnorderedAccessEnabled(), buffer->GetAccessType());
		else continue;
		rhi::ResourceAllocationInfo info{};
		device.GetResourceAllocationInfo(&desc, 1, &info);
		spdlog::info("Persistent alias exclusion: '{}' reason='not aliasable' bytes={}", concrete->GetName(), info.sizeInBytes);
	}
	if (logging) {
		std::unordered_map<uint32_t, const std::string*> nameBySlot;
		for (const auto& main : state.mainPasses) nameBySlot.emplace(main.id.index, &main.name);
		std::vector<std::string> nameByRank(nextRank);
		for (uint32_t p = 0; p < structure.passes.size(); ++p) {
			if (rank[p] == UINT32_MAX) continue;
			const auto found = nameBySlot.find(structure.passes[p].preparedPassIndex);
			nameByRank[rank[p]] = found != nameBySlot.end() ? *found->second : "<pass " + std::to_string(p) + ">";
		}
		if (state.aliasPlans <= 2) {
			uint32_t r = 0;
			for (const auto& batch : graph.batches) for (const auto pass : batch.passes)
				spdlog::info("Persistent alias rank {}: '{}' queue={}", r++, nameByRank[rank[pass]], batch.queue);
		}
		for (const auto& c : candidates) if (c.pooled)
			spdlog::info("Persistent alias placement:'{}' pool={:#x} offset={} bytes={} first={}:{}({}) last={}:{} users={}",
				c.concrete->GetName(), c.poolID, c.offset, c.sizeBytes, c.firstUse, nameByRank[c.firstUse],
				c.firstUseIsWrite ? "write" : "read", c.lastUse, nameByRank[c.lastUse], c.users.size());
	}
	return changed;
}

// ---- Re-placement prepared on a worker (PrepareFramePlacementReplacement, InstallPreparedReplacement) ----

namespace {
// Prepared backings by frame placement version and resource. Graphs of several epochs that bind one resource bind one prepared
// backing, made by whichever graph prepares first.
std::mutex g_preparedBackingsMutex;
std::map<std::pair<uint64_t, const Resource*>, std::weak_ptr<PreparedPlacedBacking>> g_preparedBackings;

std::shared_ptr<PreparedPlacedBacking> PreparedBackingFor(uint64_t versionId, Resource& concrete, BackedResource& backed,
	const std::shared_ptr<const AliasHeapGeneration>& heap, uint64_t offset, uint64_t poolID, std::string& why) {
	std::lock_guard lock(g_preparedBackingsMutex);
	std::erase_if(g_preparedBackings, [&](const auto& entry) { return entry.second.expired() || entry.first.first < versionId; });
	auto& slot = g_preparedBackings[{versionId, &concrete}];
	if (auto existing = slot.lock()) {
		if (existing->heap == heap && existing->offset == offset) return existing;
		why = "another graph prepared '" + concrete.GetName() + "' at another placement";
		return {};
	}
	auto made = backed.PreparePlacedBacking(heap, offset, poolID, &why);
	if (!made) why = "'" + concrete.GetName() + "': " + why;
	slot = made;
	return made;
}

// What a preparation reads of the graph, captured on the owner thread when it starts.
struct PreparedReplacementInputs {
	uint32_t epoch = UINT32_MAX;
	struct Item {
		Resource* concrete = nullptr;
		std::shared_ptr<Resource> keep;  // alive while the preparation runs
		State::AliasPlacementRecord record;
		bool own = false;    // this graph gives the placement back at its end (aliasPlacementOwners)
		bool frame = false;  // placed from a frame placement, not in the graph's own pools
		std::vector<uint32_t> entries;  // bound slot entries of it
	};
	std::vector<Item> items;
	// By slot (entry) index: the concrete resource of a direct, unversioned slot, and the slot with its shape.
	std::vector<Resource*> slotResource;
	std::vector<std::pair<persistent::ResourceSlotId, CompileResourceShape>> slots;
	std::unordered_map<uint32_t, persistent::PassId> idBySlot;
};

using PreparedJob = State::PreparedReplacement;

// The worker's half. Reads only the inputs, the immutable base publication and the version, and makes the prepared backings, which
// change nothing a running graph uses.
void RunPreparedReplacement(PreparedJob& job, const PreparedReplacementInputs& in) {
	using Clock = std::chrono::steady_clock;
	const auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
	const auto begin = Clock::now();
	const auto refuse = [&](std::string why) {
		job.why = std::move(why);
		job.phase.store(-1, std::memory_order_release);
	};
	try {
		const auto& version = *job.version;
		const auto& executable = *job.base->executable;
		const auto& graph = *executable.graph;
		const auto& structure = *graph.structure;
		// Ranks and users as the plan computes them, from the publication the graph runs now.
		std::vector<uint32_t> rank(structure.passes.size(), UINT32_MAX);
		uint32_t nextRank = 0;
		for (const auto& batch : graph.batches)
			for (const auto pass : batch.passes) rank.at(pass) = nextRank++;
		std::unordered_map<Resource*, uint32_t> itemOf;
		for (uint32_t i = 0; i < in.items.size(); ++i) itemOf.emplace(in.items[i].concrete, i);
		struct Life {
			uint32_t first = UINT32_MAX, last = 0;
			std::vector<uint32_t> users;
		};
		std::vector<Life> life(in.items.size());
		const auto itemFor = [&](uint32_t resourceIndex) -> uint32_t {
			if (resourceIndex >= executable.resourceSlots.size()) return UINT32_MAX;
			const auto slot = executable.resourceSlots[resourceIndex];
			if (slot.index >= in.slotResource.size() || !in.slotResource[slot.index]) return UINT32_MAX;
			const auto found = itemOf.find(in.slotResource[slot.index]);
			return found == itemOf.end() ? UINT32_MAX : found->second;
		};
		for (uint32_t p = 0; p < structure.passes.size(); ++p) {
			const auto& pass = structure.passes[p];
			const auto touch = [&](uint32_t resource) {
				const auto index = itemFor(resource);
				if (index == UINT32_MAX || rank[p] == UINT32_MAX) return;
				auto& l = life[index];
				l.first = (std::min)(l.first, rank[p]);
				l.last = (std::max)(l.last, rank[p]);
				if (l.users.empty() || l.users.back() != p) l.users.push_back(p);
			};
			for (const auto& use : pass.entryStates) touch(use.resource);
			for (const auto& use : pass.exitStates) touch(use.resource);
			for (const auto& access : pass.accesses) touch(access.resourceIndex);
		}
		for (auto& l : life) {
			std::sort(l.users.begin(), l.users.end());
			l.users.erase(std::unique(l.users.begin(), l.users.end()), l.users.end());
		}
		// Where each placement goes under the version, by the rules TakeFramePlacementInPlace checks, except that a resource may move.
		const auto slotIt = std::ranges::find_if(version.Slots(), [&](const auto& s) { return s.epoch == in.epoch; });
		const bool repeatedSlot = slotIt != version.Slots().end() && slotIt->repeated;
		const uint32_t lastRank = nextRank ? nextRank - 1 : 0;
		struct Placed {
			std::shared_ptr<const AliasHeapGeneration> heap;
			State::AliasPlacementRecord record;
			uint32_t block = UINT32_MAX, frameFirst = 0, frameLast = 0;
			bool multiEpoch = false;
		};
		std::vector<Placed> placed(in.items.size());
		for (uint32_t i = 0; i < in.items.size(); ++i) {
			const auto& item = in.items[i];
			const auto* entry = version.FindObject(item.concrete->GetGlobalResourceID() + 1);
			const auto name = "'" + item.concrete->GetName() + "'";
			if (life[i].first == UINT32_MAX) return refuse(name + " is not used by the running publication");
			if (!item.frame) {
				if (entry && entry->placed) return refuse("the version places " + name + ", which is in the graph's own pools");
				placed[i].record = item.record;
				continue;
			}
			if (!entry || !entry->placed) return refuse("the version does not place " + name);
			if (!entry->UseIn(in.epoch)) return refuse("the version has no use of " + name + " in this epoch");
			auto heap = version.PeekBlock(entry->block);
			if (!heap) return refuse("the version's block for " + name + " has no heap yet");
			if (entry->size != item.record.sizeBytes) return refuse("the size of " + name + " changed");
			const bool firstInFrame = entry->uses.front().epoch == in.epoch;
			const bool lastInFrame = entry->uses.back().epoch == in.epoch;
			const bool continues = !firstInFrame || entry->acrossRuns;
			if (continues != item.record.continues) return refuse("the contents flow of " + name + " changed");
			const bool wholeSlot = entry->acrossRuns || (entry->MultiEpoch() && repeatedSlot);
			auto& p = placed[i];
			p.block = entry->block;
			p.frameFirst = firstInFrame && !wholeSlot ? life[i].first : 0;
			p.frameLast = lastInFrame && !wholeSlot ? life[i].last : lastRank;
			p.multiEpoch = entry->MultiEpoch();
			p.record = {version.Blocks().at(entry->block).poolID, entry->offset, entry->size, heap.get(), continues};
			p.heap = std::move(heap);
		}
		// Two of this graph's frame placed resources that share bytes and are live together here: the plan would move one of them.
		// Then byte overlaps per pool or block: placement orderings between their users, as the plan derives them.
		std::map<uint64_t, std::vector<uint32_t>> byPool;
		for (uint32_t i = 0; i < in.items.size(); ++i) byPool[placed[i].record.poolID].push_back(i);
		std::vector<std::pair<uint32_t, uint32_t>> edges;
		for (auto& [poolID, members] : byPool) {
			for (size_t a = 0; a < members.size(); ++a)
				for (size_t b = a + 1; b < members.size(); ++b) {
					const auto& pa = placed[members[a]];
					const auto& pb = placed[members[b]];
					if (pa.record.heap != pb.record.heap) continue;
					if (pa.record.offset >= pb.record.offset + pb.record.sizeBytes || pb.record.offset >= pa.record.offset + pa.record.sizeBytes) continue;
					const auto& la = life[members[a]];
					const auto& lb = life[members[b]];
					const auto& nameA = in.items[members[a]].concrete->GetName();
					const auto& nameB = in.items[members[b]].concrete->GetName();
					if (in.items[members[a]].frame && in.items[members[b]].frame
						&& !(pa.frameLast < pb.frameFirst || pb.frameLast < pa.frameFirst))
						return refuse("'" + nameA + "' and '" + nameB + "' share bytes and are live together");
					const bool aFirst = la.last < lb.first;
					const auto& earlier = aFirst ? la : lb;
					const auto& later = aFirst ? lb : la;
					if (!(earlier.last < later.first)) return refuse("'" + nameA + "' and '" + nameB + "' overlap in bytes and in time");
					for (const auto u : earlier.users)
						for (const auto v : later.users) edges.emplace_back(structure.passes[u].preparedPassIndex, structure.passes[v].preparedPassIndex);
				}
		}
		std::sort(edges.begin(), edges.end());
		edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
		// Moved: another heap, offset or size, as the plan's apply decides (AtFramePlacement). Only another block pool id (a newer
		// version's block in the same heap) or another set of epochs changes the records alone, as the plan's apply does for a
		// resource already at its frame placement: its backing and bindings stay.
		std::vector<uint32_t> movedItems, updatedItems;
		for (uint32_t i = 0; i < in.items.size(); ++i) {
			const auto& item = in.items[i];
			if (!item.frame) continue;
			const auto& to = placed[i].record;
			if (to.heap != item.record.heap || to.offset != item.record.offset || to.sizeBytes != item.record.sizeBytes) movedItems.push_back(i);
			else if (!(to == item.record) || placed[i].multiEpoch == item.own) updatedItems.push_back(i);
		}
		if (movedItems.empty()) return refuse("it moves nothing");
		for (const auto i : updatedItems) {
			PreparedJob::Moved updated;
			updated.resource = in.items[i].keep;
			updated.from = in.items[i].record;
			updated.to = placed[i].record;
			updated.multiEpoch = placed[i].multiEpoch;
			job.updated.push_back(std::move(updated));
		}
		const auto planned = Clock::now();
		job.planMs = ms(begin, planned);
		// The backings the moved resources switch to, made now. A resource another epoch's graph binds gets one backing for both.
		for (const auto i : movedItems) {
			const auto& item = in.items[i];
			auto* backed = dynamic_cast<BackedResource*>(item.concrete);
			if (!backed) return refuse("'" + item.concrete->GetName() + "' has no backing to switch");
			std::string why;
			auto prepared = PreparedBackingFor(job.versionId, *item.concrete, *backed, placed[i].heap, placed[i].record.offset,
				placed[i].record.poolID, why);
			if (!prepared) return refuse(why);
			PreparedJob::Moved moved;
			moved.resource = item.keep;
			moved.prepared = std::move(prepared);
			moved.from = item.record;
			moved.to = placed[i].record;
			moved.multiEpoch = placed[i].multiEpoch;
			moved.entries = item.entries;
			job.moved.push_back(std::move(moved));
		}
		const auto backingsMade = Clock::now();
		job.backingsMs = ms(planned, backingsMade);
		// The publication: the running one with the new orderings and the moved resources' bindings, validated here.
		persistent::GraphEditTransaction edit(job.base);
		edit.ClearPlacementOrderings();
		for (const auto& [from, to] : edges) {
			const auto a = in.idBySlot.find(from), b = in.idBySlot.find(to);
			if (a == in.idBySlot.end() || b == in.idBySlot.end()) return refuse("an ordering names an unknown pass slot");
			edit.AddPlacementOrdering(a->second, b->second);
		}
		for (const auto& moved : job.moved) {
			for (const auto entryIndex : moved.entries) {
				const auto& [slot, shape] = in.slots.at(entryIndex);
				auto binding = CapturePreparedSlotBinding(*moved.resource, *moved.prepared, slot.index, shape, moved.to.continues);
				if (!binding) return refuse("no binding for '" + moved.resource->GetName() + "'");
				edit.ReplaceBinding(slot, std::move(*binding));
			}
		}
		job.ready = edit.Build(job.workspace, job.cancelled, true);
		if (!job.ready) return refuse("the validated build produced no publication");
		job.edges = std::move(edges);
		job.kept = in.items.size() - movedItems.size() - updatedItems.size();
		job.buildMs = ms(backingsMade, Clock::now());
		job.phase.store(1, std::memory_order_release);
	} catch (const std::exception& error) {
		refuse(std::string("it failed: ") + error.what());
	}
}

// Slots whose binding the live publication changed since base: false when they differ in size.
bool ChangedSlots(const persistent::SelectedPublication& base, const persistent::SelectedPublication& live, std::vector<uint32_t>& out) {
	if (base.bindings.Size() != live.bindings.Size()) return false;
	for (uint32_t i = 0; i < base.bindings.Size(); ++i) {
		const auto* a = base.bindings.TryAt(i);
		const auto* b = live.bindings.TryAt(i);
		if (a == b) continue;
		if (!a || !b) return false;
		if (a->identity == b->identity && a->backingRevision == b->backingRevision && a->descriptorRevision == b->descriptorRevision
			&& a->contentRevision == b->contentRevision && a->recording == b->recording && a->bound == b->bound && a->active == b->active)
			continue;
		out.push_back(i);
	}
	return true;
}
}  // namespace

bool RenderGraph::PrepareFramePlacementReplacement(std::shared_ptr<const FramePlacementVersion> version) {
	if (!version || !m_compilerState || m_compilerState->frameProductionStopped || !m_compilerState->persistent || !m_taskService) return false;
	auto& state = *m_compilerState->persistent;
	if (state.preparedReplacement && state.preparedReplacement->versionId == version->Id()) return true;
	state.preparedReplacement.reset();
	if (state.pending || !state.framePlacementTaken || !state.framePlacement || state.aliasPlacementDeferred || !version->HasEpoch(m_passEpochFilter)
		|| state.framePlacement->Id() >= version->Id())
		return false;
	// A resource this graph binds where another epoch's graph placed it on its own path: that graph's re-placement moves it, so this
	// graph re-places after it, as before.
	if (!state.adoptedBackings.empty()) return false;
	auto live = state.program.Select();
	if (!live || !live->executable) return false;
	auto inputs = std::make_shared<PreparedReplacementInputs>();
	inputs->epoch = m_passEpochFilter;
	std::unordered_map<Resource*, uint32_t> itemOf;
	bool anyFrame = false;
	for (const auto& [concrete, record] : state.aliasPlaced) {
		PreparedReplacementInputs::Item item;
		item.concrete = concrete;
		item.keep = concrete->weak_from_this().lock();
		if (!item.keep) return false;
		item.record = record;
		item.own = state.aliasPlacementOwners.contains(concrete);
		item.frame = FramePlacementVersion::IsBlockPoolID(record.poolID);
		anyFrame = anyFrame || item.frame;
		itemOf.emplace(concrete, static_cast<uint32_t>(inputs->items.size()));
		inputs->items.push_back(std::move(item));
	}
	if (!anyFrame) return false;
	inputs->slotResource.assign(state.entries.size(), nullptr);
	inputs->slots.resize(state.entries.size());
	for (uint32_t index = 0; index < state.entries.size(); ++index) {
		const auto& entry = state.entries[index];
		if (entry.retired || entry.versioned || !entry.resource) continue;
		auto* concrete = UnwrapDynamic(entry.resource);
		inputs->slotResource[index] = concrete;
		inputs->slots[index] = {entry.slot, entry.shape};
		if (const auto found = itemOf.find(concrete); found != itemOf.end() && entry.isBound) inputs->items[found->second].entries.push_back(index);
	}
	for (const auto& main : state.mainPasses) inputs->idBySlot.emplace(main.id.index, main.id);
	auto job = std::make_shared<State::PreparedReplacement>();
	job->versionId = version->Id();
	job->version = std::move(version);
	job->base = std::move(live);
	job->aliasPlans = state.aliasPlans;
	job->startedFrame = state.frameNumber;
	job->started = std::chrono::steady_clock::now();
	job->workspace.SetScheduleCache(m_runtime->Schedules());
	auto& compiler = *m_compilerState;
	if (!compiler.persistentBuildScope) compiler.persistentBuildScope = m_taskService->CreateScope("ORG.Persistent.Build");
	auto run = [job, inputs] {
		BT_ZONE_SCOPE("ORG.Persistent.PrepareReplacement");
		RunPreparedReplacement(*job, *inputs);
	};
	if (!compiler.persistentBuildScope
		|| !m_taskService->Submit(compiler.persistentBuildScope, runtime::TaskPriority::Streaming, "ORG.Persistent.PrepareReplacement", run))
		return false;
	state.preparedReplacement = std::move(job);
	return true;
}

bool RenderGraph::FramePlacementReplacementSettled(uint64_t versionId) const {
	if (!m_compilerState || !m_compilerState->persistent) return true;
	const auto job = m_compilerState->persistent->preparedReplacement;
	return !job || job->versionId != versionId || job->phase.load(std::memory_order_acquire) != 0;
}

bool RenderGraph::PersistentStructuralBuildPending() const {
	return m_compilerState && m_compilerState->persistent && m_compilerState->persistent->pending;
}

std::pair<uint64_t, uint64_t> RenderGraph::PreparedReplacementCounts() const {
	if (!m_compilerState || !m_compilerState->persistent) return {0, 0};
	const auto& state = *m_compilerState->persistent;
	return {state.preparedReplacementsInstalled, state.preparedReplacementsRefused};
}

bool RenderGraph::InstallPreparedReplacement() {
	auto& state = *m_compilerState->persistent;
	auto job = state.preparedReplacement;
	if (!job) return false;
	const uint64_t current = m_framePlacementProbe ? m_framePlacementProbe() : 0;
	if (job->versionId != current) {
		// Made for another version: an older one is of no use any more, a newer one waits for its publication.
		if (job->versionId < current) state.preparedReplacement.reset();
		return false;
	}
	state.preparedReplacement.reset();
	const auto begin = std::chrono::steady_clock::now();
	const auto decline = [&](const std::string& why) {
		++state.preparedReplacementsRefused;
		spdlog::info("Persistent alias re-placement: epoch {} does not install the one prepared for frame placement {} ({}); it places again "
					 "here: frame={}", m_passEpochFilter, job->versionId, why, state.frameNumber);
		return false;
	};
	const int phase = job->phase.load(std::memory_order_acquire);
	if (phase == 0) {
		// Still running: cancelled, and this preparation places as before.
		job->cancelled.store(true, std::memory_order_release);
		return decline("its preparation had not finished");
	}
	if (phase < 0) {
		if (job->why == "it moves nothing") return false;
		return decline(job->why);
	}
	const auto live = state.program.Select();
	if (!live || live->executable != job->base->executable) return decline("the graph's executable changed since");
	if (state.aliasPlans != job->aliasPlans || state.pending) return decline("the graph planned again since");
	// Everything is checked before anything changes.
	for (const auto& moved : job->moved) {
		auto* concrete = moved.resource.get();
		auto* backed = dynamic_cast<BackedResource*>(concrete);
		const auto found = state.aliasPlaced.find(concrete);
		if (!backed || found == state.aliasPlaced.end() || !(found->second == moved.from))
			return decline("'" + concrete->GetName() + "' was placed again since");
		const auto generation = backed->GetBackingGeneration();
		const bool adopted = moved.prepared->adopted.load(std::memory_order_acquire);
		if (adopted ? generation != moved.prepared->fromGeneration + 1 : generation != moved.prepared->fromGeneration)
			return decline("the backing of '" + concrete->GetName() + "' changed since");
	}
	for (const auto& updated : job->updated) {
		auto* concrete = updated.resource.get();
		const auto found = state.aliasPlaced.find(concrete);
		if (found == state.aliasPlaced.end() || !(found->second == updated.from)) return decline("'" + concrete->GetName() + "' was placed again since");
	}
	// Binding edits installed since the preparation started go onto its publication.
	auto ready = job->ready;
	std::vector<uint32_t> changed;
	if (live != job->base) {
		if (!ChangedSlots(*job->base, *live, changed)) return decline("the graph's slots changed since");
		if (!changed.empty()) {
			std::unordered_set<uint32_t> movedSlots;
			for (const auto& moved : job->moved) movedSlots.insert(moved.entries.begin(), moved.entries.end());
			try {
				persistent::GraphEditTransaction replay(ready);
				for (const auto index : changed) {
					if (movedSlots.contains(index)) return decline("a moved resource's binding changed since");
					const auto* binding = live->bindings.TryAt(index);
					const auto* prepared = ready->bindings.TryAt(index);
					if (!binding || !prepared) return decline("a slot retired since");
					const auto slot = ready->bindings.CurrentSlot(index);
					if (!binding->bound) {
						if (prepared->bound) replay.Unbind(slot);
					} else if (prepared->bound) replay.ReplaceBinding(slot, *binding);
					else replay.BindReserved(slot, *binding);
				}
				std::atomic_bool cancelled{false};
				ready = replay.Build(m_compilerState->synchronousCompileWorkspace, cancelled);
			} catch (const std::exception& error) {
				return decline(std::string("its binding replay failed: ") + error.what());
			}
			if (!ready) return decline("its binding replay produced no publication");
		}
	}
	// The switch: the moved resources take their prepared backings (a resource another epoch's graph adopted earlier in the frame
	// has it already), then the publication that binds them installs.
	size_t adoptedHere = 0;
	std::vector<bool> adoptedBy(job->moved.size(), false);
	for (size_t i = 0; i < job->moved.size(); ++i) {
		auto& moved = job->moved[i];
		if (moved.prepared->adopted.load(std::memory_order_acquire)) continue;
		auto* backed = dynamic_cast<BackedResource*>(moved.resource.get());
		if (!backed->AdoptPreparedBacking(*moved.prepared))
			throw std::runtime_error("Persistent prepared re-placement: '" + moved.resource->GetName() + "' could not switch to its prepared backing");
		adoptedBy[i] = true;
		++adoptedHere;
	}
	if (!state.program.InstallRebased(ready)) throw std::runtime_error("Persistent prepared re-placement failed to install");
	// The plan's records, as its apply leaves them for a resource placed again (or found at its new placement, when another epoch's
	// graph adopted it first).
	for (size_t i = 0; i < job->moved.size(); ++i) {
		const auto& moved = job->moved[i];
		auto* concrete = moved.resource.get();
		auto* backed = dynamic_cast<BackedResource*>(concrete);
		const auto generation = backed->GetBackingGeneration();
		state.aliasPlaced[concrete] = moved.to;
		NoteAliasPlaced(concrete, true, m_passEpochFilter);
		if (adoptedBy[i]) NoteFramePlaced(concrete, moved.to.heap, moved.to.offset, generation, &state, moved.to.sizeBytes, m_passEpochFilter);
		else ClaimFramePlaced(concrete, &state);
		if (moved.multiEpoch) state.aliasPlacementOwners.erase(concrete);
		else state.aliasPlacementOwners[concrete] = {concrete->weak_from_this(), generation};
		for (const auto entryIndex : moved.entries) {
			auto& entry = state.entries[entryIndex];
			entry.isBound = true;
			entry.bound = RotationKeyOf(entry);
			for (const auto subscriber : entry.subscribers) state.mainPasses[subscriber].slotsDirty = true;
		}
	}
	// Already where the version puts them: the records follow the version, and whether the placement is this graph's own its epochs.
	// Their claims stay as they are (as TakeFramePlacementInPlace leaves them): a newer build of this epoch, staged and placed from
	// this version meanwhile, may have claimed one, and taking the claim back would make this graph's end give its backing away.
	for (const auto& updated : job->updated) {
		auto* concrete = updated.resource.get();
		auto* backed = dynamic_cast<BackedResource*>(concrete);
		state.aliasPlaced[concrete] = updated.to;
		if (updated.multiEpoch) state.aliasPlacementOwners.erase(concrete);
		else if (backed && !state.aliasPlacementOwners.contains(concrete))
			state.aliasPlacementOwners[concrete] = {concrete->weak_from_this(), backed->GetBackingGeneration()};
	}
	state.aliasEdges = job->edges;
	state.framePlacement = job->version;
	++state.placementSerial;
	m_framePlacementVersion.store(job->versionId, std::memory_order_release);
	++state.structuralBuilds;
	++state.preparedReplacementsInstalled;
	basic_telemetry::AddCounter("ORG.Persistent.PreparedReplacements");
	spdlog::info("Persistent alias re-placement (prepared on a worker): epoch {} takes frame placement {}: {} moved ({} switched here), {} stay ({} with new records), "
				 "{} orderings, {} bindings replayed; install {:.2f} ms, prepared in {:.2f} ms (plan {:.2f}, backings {:.2f}, validated build {:.2f}) "
				 "over {} frames: frame={}",
		m_passEpochFilter, job->versionId, job->moved.size(), adoptedHere, job->kept + job->updated.size(), job->updated.size(), job->edges.size(), changed.size(),
		std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count(), job->planMs + job->backingsMs + job->buildMs,
		job->planMs, job->backingsMs, job->buildMs, state.frameNumber - job->startedFrame, state.frameNumber);
	// What the preparation holds (its compile workspace, the publication it started from, the prepared backings' records) goes on a
	// worker, not in this frame.
	if (m_taskService && m_compilerState->persistentBuildScope)
		(void)m_taskService->Submit(m_compilerState->persistentBuildScope, runtime::TaskPriority::Background, "ORG.Persistent.ReleasePreparedReplacement",
			[job = std::move(job)]() mutable { job.reset(); });
	return true;
}

} // namespace org
