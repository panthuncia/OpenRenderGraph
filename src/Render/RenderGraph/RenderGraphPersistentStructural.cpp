// Persistent execution, structural builds: representations, inline and worker-side builds, frame-interrupting passes and
// alias placement.
#include "RenderGraphPersistentState.h"

namespace org {

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
	// Textures start undefined on both APIs; buffers in the bridge-compatible COMMON state.
	const bool texture = desc.type == rhi::ResourceType::Texture2D;
	const ResourceState initial = texture
		? ResourceState{rhi::ResourceAccessType::None, rhi::ResourceLayout::Undefined, rhi::ResourceSyncState::None}
		: ResourceState{rhi::ResourceAccessType::Common, rhi::ResourceLayout::Common, rhi::ResourceSyncState::All};
	std::vector<Resource::PendingAPIRepresentation> representations;
	representations.push_back({d3d12->id, std::move(pair.canonical), initial});
	representations.push_back({vulkan->id, std::move(pair.imported), initial});
	if (auto* indexed = dynamic_cast<GloballyIndexedResource*>(concrete)) indexed->RotateDescriptorSlotsForPublication();
	if (!concrete->PublishAPIRepresentations(std::move(representations)))
		throw std::runtime_error("Failed to publish the shared representations of '" + concrete->GetName() + "'");
	concrete->RefreshAPIRepresentationDescriptors(d3d12->id);
	concrete->RefreshAPIRepresentationDescriptors(vulkan->id);
	basic_telemetry::AddCounter("ORG.Persistent.MultiDeviceRepresentations");
	spdlog::info("Persistent multi-device representations: '{}' ({})", concrete->GetName(), texture ? "texture" : "buffer");
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
		TrackTransientFrameResource(concrete);
		basic_telemetry::AddCounter("ORG.Persistent.StandaloneMaterializations");
	}
}

bool RenderGraph::IsPersistentAliasCandidate(Resource* concrete) const {
	if (!concrete) return false;
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

// Binds every declared, unbound, non-swapchain slot before a structural build.
// Direct alias candidates stay unbound so the planner can place them, unless
// several unbound slots share one concrete backing: unbound slots carry unique
// placeholder identities and would not canonicalize, so the schedule would miss
// their dependencies. Those (and group members) are realized standalone here;
// the planner may still re-place a standalone candidate afterwards.
void RenderGraph::BindUnboundPersistentEntries(persistent::GraphEditTransaction& edit) {
	auto& state = *m_compilerState->persistent;
	std::unordered_map<Resource*, uint32_t> unboundSlotsByConcrete;
	for (const auto& entry : state.entries)
		if (entry.resource && !entry.swapchain && !entry.isBound) ++unboundSlotsByConcrete[UnwrapDynamic(entry.resource)];
	for (uint32_t index = 0; index < state.entries.size(); ++index) {
		auto& entry = state.entries[index];
		if (!entry.resource || entry.swapchain || entry.isBound) continue;
		auto* concrete = UnwrapDynamic(entry.resource);
		auto* backed = concrete ? dynamic_cast<BackedResource*>(concrete) : nullptr;
		if (backed && !backed->IsMaterialized()) {
			if (entry.direct && unboundSlotsByConcrete[concrete] <= 1 && IsPersistentAliasCandidate(concrete)) continue;
			MaterializePersistentStandalone(std::span<Resource* const>{&entry.resource, 1});
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
				if (LoweringAddsDeclarations(main, lowered)) inlineRequired = true;
				lowerings.emplace(mainIndex, std::move(lowered));
			}
		}, any.pass);
	}
	auto pending = std::make_shared<State::PendingStructural>();
	pending->started = std::chrono::steady_clock::now();
	pending->base = state.program.Select();
	pending->baseSlotCount = pending->base->bindings.Size();
	std::optional<persistent::GraphEditTransaction> inlineEdit;
	if (inlineRequired) inlineEdit.emplace(state.program.BeginEdit());
	else pending->edit = std::make_unique<persistent::GraphEditTransaction>(pending->base);
	persistent::GraphEditTransaction* const transactionPtr = inlineRequired ? &*inlineEdit : pending->edit.get();
	auto& transaction = *transactionPtr;
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
				InstallMainPass(*this, state, transaction, main, std::move(lowered), mainIndex, true, std::nullopt,
					inlineRequired ? nullptr : &pending->touchedGroups);
				if (!inlineRequired) {
					// Frames keep preparing against the current publication until the install.
					std::swap(views, main.declaredViews);
					std::swap(bindings, main.declaredBindings);
					pending->declaredTokens.push_back({mainIndex, std::move(views), std::move(bindings)});
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


// Frame-interrupting per-frame passes (ExternalPassDesc::interruptsFrame, e.g.
// mid-frame debug readbacks) must observe resources at their insert point, so
// they cannot run in the Tail segment. Each frame that has them, or had them the
// frame before, pays one inline structural build: the previous set is removed
// and the new set is added at authored orders just after (or before) their
// anchors. Hazards follow the authored order, so a capture after a pass reads
// the state that pass left, and alias placement treats it as a user.
std::vector<RenderGraph::ExternalPassDesc> RenderGraph::UpdatePersistentFrameInterruptingPasses(
	std::vector<ExternalPassDesc> requested, rhi::Backend primaryBackend) {
	auto& state = *m_compilerState->persistent;
	std::vector<ExternalPassDesc> unhosted;
	if (!state.deferredInterrupting.empty()) {
		requested.insert(requested.begin(), std::make_move_iterator(state.deferredInterrupting.begin()),
			std::make_move_iterator(state.deferredInterrupting.end()));
		state.deferredInterrupting.clear();
	}
	if (requested.empty() && state.hostedMainPasses.empty()) return unhosted;
	if (state.pending) {
		// The program cannot be edited inline under an in-flight structural
		// build. Installed passes keep recording nothing; new ones wait.
		state.deferredInterrupting = std::move(requested);
		return unhosted;
	}
	BT_ZONE_SCOPE("ORG.Persistent.FrameInterruptingPasses");
	const auto started = std::chrono::steady_clock::now();
	auto edit = state.program.BeginEdit();
	const size_t removed = state.hostedMainPasses.size();
	for (const auto mainIndex : state.hostedMainPasses) {
		auto& main = state.mainPasses[mainIndex];
		edit.RemovePass(main.id);
		for (const auto groupIndex : main.groups) {
			auto& group = state.groups[groupIndex];
			std::erase(group.subscribers, mainIndex);
			for (const auto entryIndex : group.memberEntries) std::erase(state.entries[entryIndex].subscribers, mainIndex);
		}
		for (const auto entryIndex : main.directEntries) {
			auto& entry = state.entries[entryIndex];
			std::erase(entry.subscribers, mainIndex);
			if (!entry.direct || !entry.subscribers.empty()) continue;
			// Declared only by the removed pass: release the slot, and with it the
			// raw resource pointer (published versions retire independently).
			edit.RemoveResource(entry.slot);
			if (auto found = state.entriesByResourceID.find(entry.resourceID); found != state.entriesByResourceID.end()) {
				std::erase(found->second, entryIndex);
				if (found->second.empty()) state.entriesByResourceID.erase(found);
			}
			entry = {};
			entry.retired = true;
		}
		main = {};
		main.retired = true;
	}
	state.hostedMainPasses.clear();

	std::unordered_map<std::string, uint32_t> byName;
	for (uint32_t i = 0; i < state.mainPasses.size(); ++i)
		if (!state.mainPasses[i].retired && !state.mainPasses[i].hosted) byName.emplace(state.mainPasses[i].name, i);
	std::unordered_map<uint32_t, uint32_t> insertedAt; // anchor main index -> passes inserted there
	constexpr uint32_t stride = persistent::GraphEditTransaction::kAuthoredOrderStride;
	for (auto& desc : requested) {
		if (desc.type == PassType::Unknown || !desc.pass || desc.name.empty()) continue;
		std::vector<uint32_t> after, before;
		if (desc.where) {
			for (const auto& name : desc.where->after) if (const auto found = byName.find(name); found != byName.end()) after.push_back(found->second);
			for (const auto& name : desc.where->before) if (const auto found = byName.find(name); found != byName.end()) before.push_back(found->second);
		}
		if (after.empty() && before.empty()) {
			static std::atomic<uint32_t> reported{0};
			if (reported.fetch_add(1) < 8)
				spdlog::warn("Frame-interrupting pass '{}' names no main pass to insert against; running it after the main executable", desc.name);
			unhosted.push_back(std::move(desc));
			continue;
		}
		// Latest 'after' anchor, else earliest 'before' anchor; consecutive
		// insertions at one anchor keep their request order.
		uint32_t anchor = after.empty() ? before.front() : after.front();
		for (const auto a : after) if (edit.AuthoredOrder(state.mainPasses[a].id) > edit.AuthoredOrder(state.mainPasses[anchor].id)) anchor = a;
		if (after.empty()) for (const auto b : before) if (edit.AuthoredOrder(state.mainPasses[b].id) < edit.AuthoredOrder(state.mainPasses[anchor].id)) anchor = b;
		const auto k = ++insertedAt[anchor];
		if (k >= stride / 2) throw std::runtime_error("Too many frame-interrupting passes at '" + state.mainPasses[anchor].name + "'");
		const auto anchorOrder = edit.AuthoredOrder(state.mainPasses[anchor].id);
		const uint32_t order = after.empty() ? anchorOrder - stride / 2 + k : anchorOrder + k;

		uint32_t mainIndex = 0;
		while (mainIndex < state.mainPasses.size() && !state.mainPasses[mainIndex].retired) ++mainIndex;
		if (mainIndex == state.mainPasses.size()) state.mainPasses.emplace_back();
		auto& main = state.mainPasses[mainIndex];
		main = {};
		main.hosted = std::make_unique<AnyPassAndResources>(MaterializeExternalPass(desc, true, false));
		main.name = main.hosted->name;
		auto& any = *main.hosted;
		std::visit([&](auto& value) {
			using T = std::decay_t<decltype(value)>;
			if constexpr (!std::is_same_v<T, std::monostate>) {
				auto lowered = lowering::LowerPass(*this, _registry, m_queueRegistry, primaryBackend, value, any.type);
				std::vector<Resource*> resources;
				for (const auto& use : lowered.entries) resources.push_back(use.resource);
				for (const auto& group : lowered.groups) for (const auto& [id, resource] : group.members) resources.push_back(resource.get());
				MaterializePersistentStandalone(resources, true);
				InstallMainPass(*this, state, edit, main, std::move(lowered), mainIndex, false, order);
			}
		}, any.pass);
		for (const auto a : after) edit.AddOrdering(state.mainPasses[a].id, main.id);
		for (const auto b : before) edit.AddOrdering(main.id, state.mainPasses[b].id);
		state.hostedMainPasses.push_back(mainIndex);
	}
	state.forceNewAliasHeaps = ForcedRealias().withCaptures && !state.hostedMainPasses.empty();
	auto ready = BuildPersistentStructural(edit);
	if (state.forceNewAliasHeaps) spdlog::info("Persistent forced alias re-placement with frame-interrupting passes: frame={}", state.frameNumber);
	state.forceNewAliasHeaps = false;
	if (!ready || !state.program.Install(edit, ready)) throw std::runtime_error("Persistent frame-interrupting pass edit failed to install");
	++state.structuralBuilds;
	basic_telemetry::AddCounter("ORG.Persistent.StructuralBuilds");
	basic_telemetry::AddCounter("ORG.Persistent.FrameInterruptingBuilds");
	std::string names;
	for (const auto mainIndex : state.hostedMainPasses) {
		if (names.size() > 512) { names += " ..."; break; }
		names += (names.empty() ? "" : ", ") + state.mainPasses[mainIndex].name;
	}
	spdlog::info("Persistent frame-interrupting passes: added={} removed={} build_ms={:.2f} [{}]", state.hostedMainPasses.size(), removed,
		std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(), names);
	return unhosted;
}

// Alias placement over the scheduled executable. Lifetimes are ranks in batch
// order (a linear extension of the scheduling edges), packed per pool with the
// same greedy sweep-line as the per-frame planner. Overlapping placements get
// pairwise placement orderings between all their users, so any schedule of the
// rebuilt executable keeps them disjoint in time (ValidateAliasOrder checks it).
void RenderGraph::RetireIdlePersistentAliasPools(PersistentExecutionState& state) {
	const uint64_t retireFrames = m_getAutoAliasPoolRetireIdleFrames ? (std::max)(1u, m_getAutoAliasPoolRetireIdleFrames()) : 120u;
	for (auto it = state.aliasPools.begin(); it != state.aliasPools.end();) {
		auto& pool = it->second;
		if (!pool.idleSinceFrame || state.frameNumber - pool.idleSinceFrame < retireFrames) { ++it; continue; }
		spdlog::info("Persistent alias pool retired: pool={:#x} capacity={} idleFrames={}", it->first, pool.capacityBytes,
			state.frameNumber - pool.idleSinceFrame);
		basic_telemetry::AddCounter("ORG.Persistent.AliasPoolRetirements");
		if (pool.allocation) DeletionManager::GetInstance().MarkForDelete(std::move(pool.allocation));
		it = state.aliasPools.erase(it);
	}
}

bool RenderGraph::PlanPersistentAliasPlacement(persistent::GraphEditTransaction& edit, const persistent::SelectedPublication& scheduled) {
	BT_ZONE_SCOPE("ORG.Persistent.PlanAliases");
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

	std::vector<uint32_t> rank(structure.passes.size(), UINT32_MAX);
	std::vector<uint8_t> backendByPass(structure.passes.size(), 0);
	uint32_t nextRank = 0;
	for (const auto& batch : graph.batches) for (const auto pass : batch.passes) {
		rank.at(pass) = nextRank++;
		backendByPass[pass] = static_cast<uint8_t>(structure.queues.at(batch.queue).backendInstance);
	}
	uint32_t maxCriticality = 1;
	for (const auto c : graph.criticality) maxCriticality = (std::max)(maxCriticality, c);

	struct Candidate {
		Resource* concrete = nullptr;
		bool texture = false;
		bool direct = true;
		uint8_t resourceClass = 0;
		uint64_t sizeBytes = 0, alignment = 1;
		uint32_t firstUse = UINT32_MAX, lastUse = 0;
		bool firstUseIsWrite = false;
		uint32_t criticality = 0;
		uint64_t backendMask = 0;
		std::optional<uint64_t> manualPool;
		bool materialized = false; // By someone other than this planner.
		std::vector<uint32_t> users;   // Compiled pass indices.
		uint64_t poolID = 0, offset = 0;
		bool pooled = false;
		const char* exclusion = nullptr;
	};
	std::vector<Candidate> candidates;
	std::unordered_map<Resource*, uint32_t> candidateByConcrete;
	std::unordered_map<Resource*, std::vector<uint32_t>> entriesByConcrete;
	for (uint32_t index = 0; index < state.entries.size(); ++index) {
		const auto& entry = state.entries[index];
		if (entry.resource && !entry.swapchain) entriesByConcrete[UnwrapDynamic(entry.resource)].push_back(index);
	}
	auto device = DeviceManager::GetInstance().GetDevice();
	auto candidateFor = [&](uint32_t resourceIndex) -> uint32_t {
		if (resourceIndex >= executable.resourceSlots.size()) return UINT32_MAX;
		const auto slot = executable.resourceSlots[resourceIndex];
		if (slot.index >= state.entries.size()) return UINT32_MAX;
		const auto& entry = state.entries[slot.index];
		if (!entry.resource || entry.swapchain) return UINT32_MAX;
		auto* concrete = UnwrapDynamic(entry.resource);
		if (!IsPersistentAliasCandidate(concrete)) return UINT32_MAX;
		if (const auto found = candidateByConcrete.find(concrete); found != candidateByConcrete.end()) return found->second;
		Candidate c;
		c.concrete = concrete;
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
		rhi::ResourceAllocationInfo info{};
		device.GetResourceAllocationInfo(&desc, 1, &info);
		c.sizeBytes = info.sizeInBytes;
		c.alignment = (std::max<uint64_t>)(1, info.alignment);
		auto* backed = dynamic_cast<BackedResource*>(concrete);
		c.materialized = backed && backed->IsMaterialized() && !state.aliasPlaced.contains(concrete);
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
			if (passRank < c.firstUse) { c.firstUse = passRank; c.firstUseIsWrite = write; }
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
	// Diagnostic bisection: comma-separated name substrings kept out of the pools ("*": every resource).
	static const std::vector<std::string> diagnosticExclusions = [] {
		std::vector<std::string> out;
		if (const char* list = std::getenv("ORG_PERSISTENT_ALIAS_EXCLUDE")) {
			std::string text(list);
			for (size_t pos = 0; pos < text.size();) {
				const auto next = text.find(',', pos);
				auto token = text.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
				if (!token.empty()) out.push_back(std::move(token));
				pos = next == std::string::npos ? text.size() : next + 1;
			}
		}
		return out;
	}();
	size_t excludedCount = 0;
	for (auto& c : candidates) {
		std::sort(c.users.begin(), c.users.end());
		c.users.erase(std::unique(c.users.begin(), c.users.end()), c.users.end());
		const auto& name = c.concrete->GetName();
		if (std::any_of(diagnosticExclusions.begin(), diagnosticExclusions.end(),
				[&](const std::string& token) { return token == "*" || name.find(token) != std::string::npos; }))
			c.exclusion = "ORG_PERSISTENT_ALIAS_EXCLUDE";
		else if (c.firstUse == UINT32_MAX) { c.exclusion = "unused"; }
		else if (!c.direct) { c.exclusion = "group member"; }
		else if (c.manualPool) {
			if (!c.firstUseIsWrite)
				throw std::runtime_error("Aliasing candidate '" + c.concrete->GetName() + "' in manual pool has a first-use read");
			c.poolID = *c.manualPool; c.pooled = true;
		} else if (mode == AutoAliasMode::Off) c.exclusion = "auto aliasing off";
		else if (!c.firstUseIsWrite) c.exclusion = "first use is read";
		else {
			const float benefitMB = static_cast<float>(c.sizeBytes) / (1024.0f * 1024.0f);
			const float critNorm = static_cast<float>(c.criticality) / static_cast<float>(maxCriticality);
			const float penalty = c.materialized ? 1.0f : 0.0f;
			const float score = mode == AutoAliasMode::Conservative ? benefitMB - 2.0f * critNorm - 1.0f * penalty
				: mode == AutoAliasMode::Balanced ? benefitMB - 1.25f * critNorm - 0.5f * penalty
				: benefitMB - 0.5f * critNorm - 0.25f * penalty;
			if (score < threshold) c.exclusion = "score below threshold";
			else { c.poolID = kAutoPoolGlobal; c.pooled = true; }
		}
		if (c.pooled) {
			boost::hash_combine(c.poolID, c.backendMask);
			boost::hash_combine(c.poolID, c.resourceClass);
			if (std::popcount(c.backendMask) > 1) { c.pooled = false; c.exclusion = "multi-backend placement unsupported"; }
		}
		if (c.exclusion) ++excludedCount;
	}

	// Pack per pool.
	std::map<uint64_t, std::vector<uint32_t>> byPool;
	std::unordered_set<uint64_t> plannedPools;
	for (uint32_t i = 0; i < candidates.size(); ++i) if (candidates[i].pooled) byPool[candidates[i].poolID].push_back(i);
	std::vector<std::pair<uint32_t, uint32_t>> edges; // (logical slot, logical slot)
	uint64_t independentBytes = 0, requiredBytes = 0, reservedBytes = 0;
	size_t overlapPairs = 0;
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
		const auto packed = packingStrategy == AutoAliasPackingStrategy::BranchAndBound
			? alias::PackBeamSearch(items) : alias::PackGreedySweepLine(items);
		for (size_t i = 0; i < members.size(); ++i) candidates[members[i]].offset = packed.offsets[i];
		const uint64_t heapEnd = packed.heapEnd, poolAlignment = packed.alignment;
		requiredBytes += heapEnd;
		if (heapEnd == 0) continue;
		// Pool heap: grows with headroom, and shrinks to that headroom when the plan needs less than half of it. A
		// replaced generation stays alive through the leases held by in-flight publications.
		auto& pool = state.aliasPools[poolID];
		plannedPools.insert(poolID);
		pool.idleSinceFrame = 0;
		const auto withHeadroom = static_cast<uint64_t>(std::ceil(static_cast<double>(heapEnd) * headroom));
		const bool initial = !static_cast<bool>(pool.allocation);
		const bool grow = !initial && (heapEnd > pool.capacityBytes || poolAlignment > pool.alignment);
		const bool shrink = !initial && !grow && pool.capacityBytes > 2 * withHeadroom;
		const bool forced = !initial && !grow && !shrink && state.forceNewAliasHeaps;
		if (initial || grow || shrink || forced) {
			uint64_t capacity = heapEnd;
			if (forced) capacity = (std::max)(capacity, pool.capacityBytes);
			else if (shrink) capacity = withHeadroom;
			else if (!initial && pool.capacityBytes)
				capacity = (std::max)(capacity, static_cast<uint64_t>(std::ceil(static_cast<double>(pool.capacityBytes) * headroom)));
			rhi::ma::AllocationDesc allocDesc{};
			allocDesc.heapType = rhi::HeapType::DeviceLocal;
			allocDesc.flags = rhi::ma::AllocationFlagCanAlias;
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
			if (pool.allocation) DeletionManager::GetInstance().MarkForDelete(std::move(pool.allocation));
			pool.allocation = std::move(fresh);
			pool.capacityBytes = capacity;
			pool.alignment = poolAlignment;
			++pool.generation;
			pool.heap = AliasHeapGeneration::Capture(pool.allocation, pool.generation);
			spdlog::info("Persistent alias pool {}: pool={:#x} capacity={} required={} alignment={} placements={} generation={}",
				initial ? "allocated" : forced ? "reallocated (forced)" : shrink ? "shrank" : "grew", poolID, capacity, heapEnd, poolAlignment, members.size(), pool.generation);
			basic_telemetry::AddCounter("ORG.Persistent.AliasPoolAllocations");
		}
		pool.resourceClass = candidates[members.front()].resourceClass;
		reservedBytes += pool.capacityBytes;
		// Placement orderings between byte-overlapping occupants.
		for (size_t i = 0; i < members.size(); ++i) for (size_t j = i + 1; j < members.size(); ++j) {
			const auto& a = candidates[members[i]]; const auto& b = candidates[members[j]];
			if (a.offset >= b.offset + b.sizeBytes || b.offset >= a.offset + a.sizeBytes) continue;
			const auto& earlier = a.lastUse < b.firstUse ? a : b;
			const auto& later = a.lastUse < b.firstUse ? b : a;
			if (!(earlier.lastUse < later.firstUse))
				throw std::logic_error("Persistent alias plan produced overlapping lifetimes for '" + a.concrete->GetName() + "' and '" + b.concrete->GetName() + "'");
			++overlapPairs;
			for (const auto u : earlier.users) for (const auto v : later.users)
				edges.emplace_back(structure.passes[u].preparedPassIndex, structure.passes[v].preparedPassIndex);
		}
	}
	std::sort(edges.begin(), edges.end());
	edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
	// A pool this plan places nothing into idles; RetireIdlePersistentAliasPools frees it once it has idled long enough.
	for (auto& [poolID, pool] : state.aliasPools)
		if (!plannedPools.contains(poolID) && !pool.idleSinceFrame) pool.idleSinceFrame = (std::max)(state.frameNumber, uint64_t{1});

	// Apply: orderings, then bindings.
	bool changed = false;
	if (edges != state.aliasEdges) {
		changed = true;
		edit.ClearPlacementOrderings();
		std::unordered_map<uint32_t, persistent::PassId> idBySlot;
		for (const auto& main : state.mainPasses) if (!main.retired) idBySlot.emplace(main.id.index, main.id);
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
	for (auto& c : candidates) {
		auto* backed = dynamic_cast<BackedResource*>(c.concrete);
		if (!backed) continue;
		if (c.pooled) {
			++placed;
			const auto& pool = state.aliasPools.at(c.poolID);
			const State::AliasPlacementRecord record{c.poolID, c.offset, c.sizeBytes, pool.heap.get()};
			const auto previous = state.aliasPlaced.find(c.concrete);
			if (previous != state.aliasPlaced.end() && previous->second == record && backed->IsMaterialized()) continue;
			if (backed->IsMaterialized()) { dematerialize(c.concrete); ++rematerialized; }
			if (c.texture) {
				PixelBuffer::MaterializeOptions options{};
				options.aliasPlacement = TextureAliasPlacement{pool.heap, c.offset, c.poolID};
				static_cast<PixelBuffer*>(c.concrete)->Materialize(&options);
			} else {
				BufferBase::MaterializeOptions options{};
				options.aliasPlacement = BufferAliasPlacement{pool.heap, c.offset, c.poolID};
				static_cast<BufferBase*>(c.concrete)->Materialize(&options);
			}
			TrackTransientFrameResource(c.concrete);
			state.aliasPlaced[c.concrete] = record;
			rebind(c.concrete);
		} else {
			if (state.aliasPlaced.erase(c.concrete) && backed->IsMaterialized()) { dematerialize(c.concrete); ++rematerialized; }
			if (!backed->IsMaterialized()) {
				MaterializePersistentStandalone(std::span<Resource* const>{&c.concrete, 1});
				++standalone;
				rebind(c.concrete);
			}
		}
	}
	state.aliasRematerializations += rematerialized;
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
			if (const auto found = state.aliasPools.find(poolID); found != state.aliasPools.end()) pool.reservedBytes = found->second.capacityBytes;
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
		spdlog::info("Persistent alias plan #{}: candidates={} placed={} excluded={} pools={} overlapPairs={} orderings={} rematerialized={} standalone={} "
			"independentMB={:.2f} requiredMB={:.2f} reservedMB={:.2f} rebuild={}",
			state.aliasPlans, candidates.size(), placed, excludedCount, byPool.size(), overlapPairs, edges.size(), rematerialized, standalone,
			independentBytes / (1024.0 * 1024.0), requiredBytes / (1024.0 * 1024.0), reservedBytes / (1024.0 * 1024.0), changed);
	if (logging) for (const auto& c : candidates) if (c.exclusion)
		spdlog::info("Persistent alias exclusion: '{}' reason='{}' bytes={}", c.concrete->GetName(), c.exclusion, c.sizeBytes);
	if (logging) {
		std::unordered_map<uint32_t, const std::string*> nameBySlot;
		for (const auto& main : state.mainPasses) if (!main.retired) nameBySlot.emplace(main.id.index, &main.name);
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

} // namespace org
