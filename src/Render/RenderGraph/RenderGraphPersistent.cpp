// Persistent execution, frame preparation: installing passes into the program, polling declarations, resolver groups and
// backings, and preparing the frame's invocations and dynamic segments. See RenderGraphPersistentState.h.
#include "OpenRenderGraph/Trace.h"
#include "RenderGraphPersistentState.h"

namespace org {

// ---------------------------------------------------------------------------
// Program editing helpers (owner thread)

namespace persistent_impl {

// Directory entry for a slot the program just allocated. Slots are dense except
// that the program reuses the indices of removed slots, whose entries are
// retired rather than erased.
State::SlotEntry& ClaimEntry(State& state, persistent::ResourceSlotId slot) {
	if (slot.index == state.entries.size()) return state.entries.emplace_back();
	if (slot.index > state.entries.size() || !state.entries[slot.index].retired)
		throw std::logic_error("Persistent slot directory expects dense slot allocation");
	auto& entry = state.entries[slot.index];
	entry = {};
	return entry;
}

// Finds or creates the directory slot for a scheduling identity. Direct slots
// are shared by every pass declaring the resource directly; group member slots
// are private to their group (see ReserveGroupMember).
uint32_t DirectEntry(State& state, persistent::GraphEditTransaction& edit, uint64_t resourceID, Resource* resource,
	CompileResourceShape shape, bool swapchain, std::shared_ptr<Resource> owner) {
	auto& indices = state.entriesByResourceID[resourceID];
	for (const auto index : indices) {
		auto& entry = state.entries[index];
		if (entry.direct && entry.resource == resource && entry.shape == shape) {
			if (owner && !entry.member) entry.member = std::move(owner);
			return index;
		}
	}
	const auto slot = edit.ReserveResource(shape);
	auto& entry = ClaimEntry(state, slot);
	entry.direct = true;
	entry.resource = resource;
	entry.member = std::move(owner);
	entry.resourceID = resourceID;
	entry.shape = shape;
	entry.swapchain = swapchain;
	entry.slot = slot;
	indices.push_back(slot.index);
	return slot.index;
}

// While a structural build is pending, edits against the live publication are
// logged for replay onto the result; slots the live publication does not have
// yet are logged only.
bool LiveEdit(const State& state, const persistent::GraphEditTransaction& edit) {
	return state.pending && &edit != state.pending->edit.get();
}
void UnbindSlot(State& state, persistent::GraphEditTransaction& edit, const State::SlotEntry& entry) {
	if (LiveEdit(state, edit)) {
		state.pending->bindingLog[entry.slot.index] = std::nullopt;
		if (entry.slot.index >= state.pending->baseSlotCount) return;
	}
	edit.Unbind(entry.slot);
}
bool BindEntryOnly(State& state, persistent::GraphEditTransaction& edit, uint32_t index, uint64_t* identity = nullptr) {
	auto& entry = state.entries[index];
	if (entry.swapchain || !entry.resource) return false;
	if (entry.multiDevice && state.ensureRepresentations) state.ensureRepresentations(*entry.resource);
	auto binding = CaptureSlotBinding(*entry.resource, entry.slot.index, entry.shape);
	if (binding && identity) *identity = binding->identity;
	if (!binding) {
		if (entry.isBound) { UnbindSlot(state, edit, entry); entry.isBound = false; entry.bound = {}; }
		static std::atomic<uint32_t> reported{0};
		if (reported.fetch_add(1) < 16)
			spdlog::warn("Persistent slot {} could not capture a backing for '{}' (unmaterialized or invalid)", index, entry.resource->GetName());
		basic_telemetry::AddCounter("ORG.Persistent.UnboundCaptures");
		return false;
	}
	bool apply = true;
	if (LiveEdit(state, edit)) {
		state.pending->bindingLog[entry.slot.index] = *binding;
		apply = entry.slot.index < state.pending->baseSlotCount;
	}
	if (apply) {
		BT_ZONE_SCOPE("ORG.Persistent.BindEntry.Edit");
		if (entry.isBound) edit.ReplaceBinding(entry.slot, std::move(*binding));
		else edit.BindReserved(entry.slot, std::move(*binding));
	}
	entry.isBound = true;
	entry.bound = RotationKeyOf(entry);
	return true;
}
// Every slot bound to one physical backing must carry the same admission
// state. A capture reads the resource's live state tracker, so binding one slot
// (a relowered pass's new direct entry, a group member arriving for a backing
// another slot already binds) next to a sibling captured frames earlier pairs
// two different incoming states, and the edit fails to build with "Shared
// physical slots disagree on incoming state". Recapture the bound siblings at
// the same moment. Admission seeds a backing from these regions only until it
// tracks the backing, so a refresh at most moves an untracked seed to the
// tracker's current state.
bool BindEntry(State& state, persistent::GraphEditTransaction& edit, uint32_t index) {
	uint64_t identity = 0;
	if (!BindEntryOnly(state, edit, index, &identity)) return false;
	for (const auto sibling : edit.SlotsSharingIdentity(identity)) {
		if (sibling == state.entries[index].slot.index || sibling >= state.entries.size()) continue;
		const auto& other = state.entries[sibling];
		if (other.retired || !other.isBound || other.swapchain || !other.resource || other.slot.index != sibling) continue;
		basic_telemetry::AddCounter("ORG.Persistent.SharedBackingRecaptures");
		BindEntryOnly(state, edit, sibling);
	}
	return true;
}

// The (id, slot) pairs a bound entry contributes to the slot maps of its
// subscribers: scheduling ID, registry object ID and (for wrappers) the
// concrete backing ID. Direct entries additionally expose registry handle
// IDs, which never change and are only produced by the full rebuild.
void QueueSlotPatchIDs(State& state, const State::SlotEntry& entry, const std::array<uint64_t, 3>& ids, bool add) {
	for (const auto subscriber : entry.subscribers) {
		auto& pass = state.mainPasses[subscriber];
		auto& list = add ? pass.slotAdds : pass.slotRemoves;
		for (const auto id : ids) if (id) list.emplace_back(id, entry.slot.index);
	}
}
void QueueSlotPatch(State& state, const State::SlotEntry& entry, uint64_t concreteID, bool add) {
	QueueSlotPatchIDs(state, entry, {entry.resourceID, entry.resource ? entry.resource->GetGlobalResourceID() : 0, concreteID}, add);
}

// Reserves member positions for a group (a structural edit), doubling its capacity until `needed` fit. The new
// positions start free and carry every subscriber's group views.
void GrowGroup(State& state, persistent::GraphEditTransaction& edit, uint32_t groupIndex, size_t needed) {
	auto& group = state.groups[groupIndex];
	while (group.capacity < needed) {
		const auto more = edit.ReserveGroupMembers(group.group, group.capacity);
		for (const auto slot : more) {
			auto& entry = ClaimEntry(state, slot);
			entry.slot = slot;
			entry.shape = group.shape;
			entry.subscribers = group.subscribers;
			group.memberEntries.push_back(slot.index);
			group.memberReplacedIDs.emplace_back();
			group.freeMembers.push_back(static_cast<uint32_t>(group.memberEntries.size() - 1));
		}
		for (const auto subscriber : group.subscribers) {
			const auto& main = state.mainPasses[subscriber];
			for (const auto& requirement : main.groupViews) if (requirement.group == groupIndex)
				for (const auto slot : more) {
					const auto binding = edit.BindGroupMember(main.id, group.group, slot);
					for (const auto view : requirement.views) edit.RequireView(binding, view);
				}
		}
		group.capacity *= 2;
		basic_telemetry::AddCounter("ORG.Persistent.GroupCapacityGrowth");
	}
}

struct GroupSyncCounts { size_t added = 0, removed = 0; };
// Brings a group's bound members to `members`, one resolver capture; the caller has made room (GrowGroup). A member
// that stays keeps its position. Members that left keep their positions bound until the arrivals are placed, so a
// member arriving in the same capture (a rotated publication of the same logical resource: a VersionedBuffer's next
// version) replaces the binding in place, and the passes whose uses name that position follow it. Positions are reused
// lowest-first in resolver order, so a set republished every frame lands on the same positions each time; a physical
// backing shared with another group then keeps its equivalence class (a position swap would be a structural change).
// Positions nobody refilled are unbound. An unmaterialized arrival is left unbound for BindUnboundPersistentEntries.
GroupSyncCounts SyncGroupMembers(State& state, persistent::GraphEditTransaction& edit, State::ResolverGroup& group,
	const std::vector<std::pair<uint64_t, std::shared_ptr<Resource>>>& members) {
	GroupSyncCounts counts;
	std::unordered_set<uint64_t> wanted;
	wanted.reserve(members.size());
	for (const auto& [id, member] : members) if (member) wanted.insert(id);
	std::vector<uint32_t> vacated;
	for (auto it = group.memberIndexByID.begin(); it != group.memberIndexByID.end();) {
		const auto [id, i] = *it;
		if (wanted.contains(id)) { ++it; continue; }
		std::erase(state.entriesByResourceID[id], group.memberEntries[i]);
		group.memberReplacedIDs[i] = id;
		vacated.push_back(i);
		it = group.memberIndexByID.erase(it);
	}
	std::sort(vacated.begin(), vacated.end(), std::greater<>());
	std::sort(group.freeMembers.begin(), group.freeMembers.end(), std::greater<>());
	for (const auto& [id, member] : members) {
		if (!member || group.memberIndexByID.contains(id)) continue;
		Resource* const resource = member.get();
		uint32_t i;
		if (!vacated.empty()) { i = vacated.back(); vacated.pop_back(); }
		else if (!group.freeMembers.empty()) { i = group.freeMembers.back(); group.freeMembers.pop_back(); group.memberReplacedIDs[i].reset(); }
		else throw std::logic_error("Persistent group capacity accounting mismatch");
		auto& entry = state.entries[group.memberEntries[i]];
		const bool replacing = entry.isBound;
		const std::array<uint64_t, 3> previousIDs = replacing && entry.resource
			? std::array<uint64_t, 3>{entry.resourceID, entry.resource->GetGlobalResourceID(), entry.bound.concreteID} : std::array<uint64_t, 3>{};
		entry.resource = resource; entry.resourceID = id; entry.member = member;
		state.entriesByResourceID[id].push_back(group.memberEntries[i]);
		group.memberIndexByID.emplace(id, i);
		if (replacing) { QueueSlotPatchIDs(state, entry, previousIDs, false); ++counts.removed; }
		auto* concrete = UnwrapDynamic(resource);
		auto* backed = concrete ? dynamic_cast<BackedResource*>(concrete) : nullptr;
		if (backed && !backed->IsMaterialized()) {
			if (replacing) { UnbindSlot(state, edit, entry); entry.isBound = false; entry.bound = {}; }
			continue;
		}
		BT_ZONE_SCOPE("ORG.Persistent.PollGroups.Diff.Bind");
		if (BindEntry(state, edit, group.memberEntries[i])) { QueueSlotPatch(state, entry, entry.bound.concreteID, true); ++counts.added; }
	}
	for (const auto i : vacated) {
		auto& entry = state.entries[group.memberEntries[i]];
		if (entry.isBound) { UnbindSlot(state, edit, entry); QueueSlotPatch(state, entry, entry.bound.concreteID, false); ++counts.removed; }
		entry.isBound = false; entry.bound = {}; entry.resource = nullptr; entry.resourceID = 0; entry.member.reset();
		group.memberReplacedIDs[i].reset(); // The logical member left.
		group.freeMembers.push_back(i);
	}
	return counts;
}

// The member slot that names `id`: its current member, or the one that replaced it in place.
std::optional<uint32_t> MemberPosition(const State::ResolverGroup& group, uint64_t id) {
	if (const auto found = group.memberIndexByID.find(id); found != group.memberIndexByID.end()) return found->second;
	for (uint32_t i = 0; i < group.memberReplacedIDs.size(); ++i)
		if (group.memberReplacedIDs[i] == id) return i;
	return std::nullopt;
}

// Whether a pass's declaration (one resolver capture) names a member the group has not seen: a publication newer than
// the group's, which the poll would bind a frame later. A declaration the group has rotated past names its members'
// predecessors, which MemberPosition still resolves.
bool DeclarationIsAhead(const State::ResolverGroup& group, const std::vector<std::pair<uint64_t, std::shared_ptr<Resource>>>& members) {
	return std::any_of(members.begin(), members.end(), [&](const auto& member) { return member.second && !MemberPosition(group, member.first); });
}

} // namespace

// Installs (or reinstalls) a main pass declaration into the transaction.
void persistent_impl::InstallMainPass(RenderGraph& graph, State& state, persistent::GraphEditTransaction& edit,
	State::MainPass& main, LoweredPass lowered, uint32_t mainIndex, bool replace, std::optional<uint32_t> authoredOrder,
	std::vector<uint32_t>* touchedGroups) {
	std::vector<uint32_t> previousDirect;
	if (replace) {
		previousDirect = main.directEntries;
		for (const auto groupIndex : main.groups) {
			auto& group = state.groups[groupIndex];
			edit.RemoveGroupAccess(main.id, group.group);
			std::erase(group.subscribers, mainIndex);
			for (const auto entryIndex : group.memberEntries) std::erase(state.entries[entryIndex].subscribers, mainIndex);
		}
		for (const auto entryIndex : main.directEntries) std::erase(state.entries[entryIndex].subscribers, mainIndex);
		main.groups.clear();
		main.directEntries.clear();
		edit.ReplacePass(main.id, std::move(lowered.declaration));
	} else {
		main.id = edit.AddPass(std::move(lowered.declaration), authoredOrder);
	}
	edit.SetPassEpoch(main.id, main.epoch);
	edit.SetPassDebugName(main.id, main.name);
    main.resourceUses = lowered.resourceUses;
    main.declaredViews.clear();
    main.declaredBindings.clear();
    main.resolverUseEntries.clear();
    main.groupViews.clear();
	main.fingerprint = FingerprintLowering(lowered);
	main.loweredDirect.clear();
	for (const auto& use : lowered.entries) main.loweredDirect.emplace_back(use.resourceID, use.resource ? use.resource->GetName() : std::string{});
	main.loweredGroupKeys.clear();
	for (const auto& group : lowered.groups) main.loweredGroupKeys.push_back(group.key.get());
	main.explicitWaits = std::move(lowered.explicitWaits);
	main.directResolvers = std::move(lowered.directResolvers);
	main.device = lowered.device;
	main.slotsDirty = true;
	std::unordered_map<uint64_t, persistent::BindingToken> tokens;
	auto declare = [&](const LoweredUse& use, bool entry) {
		const bool swapchain = use.resource && use.resource->GetName() == "Backbuffer"
			&& dynamic_cast<DynamicResource*>(use.resource) && !dynamic_cast<BackedResource*>(UnwrapDynamic(use.resource));
		const auto index = DirectEntry(state, edit, use.resourceID, use.resource, ShapeOf(*use.resource), swapchain, use.owner);
		auto& slotEntry = state.entries[index];
		if (std::find(slotEntry.subscribers.begin(), slotEntry.subscribers.end(), mainIndex) == slotEntry.subscribers.end())
			slotEntry.subscribers.push_back(mainIndex);
		if (main.device != BackendInstanceId::Primary && !slotEntry.multiDevice) {
			slotEntry.multiDevice = true;
			if (slotEntry.isBound) BindEntry(state, edit, index); // Rebound with its representations.
		}
		if (std::find(main.directEntries.begin(), main.directEntries.end(), index) == main.directEntries.end())
			main.directEntries.push_back(index);
		if (entry) {
			tokens[index] = edit.Declare(main.id, slotEntry.slot, use.range, use.state);
		} else {
			auto found = tokens.find(index);
			if (found == tokens.end()) found = tokens.emplace(index, edit.DeclareDependency(main.id, slotEntry.slot, use.state.write)).first;
			edit.DeclarePostcondition(found->second, use.range, use.state);
		}
	};
	for (const auto& use : lowered.entries) declare(use, true);
	for (const auto& use : lowered.exits) declare(use, false);
	for (auto& loweredGroup : lowered.groups) {
		uint32_t groupIndex = UINT32_MAX;
		for (uint32_t i = 0; i < state.groups.size(); ++i)
			if (state.groups[i].key == loweredGroup.key && state.groups[i].shape == loweredGroup.shape) { groupIndex = i; break; }
		const bool created = groupIndex == UINT32_MAX;
		if (created) {
			State::ResolverGroup group;
			group.key = loweredGroup.key;
			group.resolver = std::move(loweredGroup.resolver);
			group.shape = loweredGroup.shape;
			group.waitRevision = loweredGroup.waitRevision;
			group.waits = std::move(loweredGroup.waits);
			// Exact published selections have one member that rotates in place;
			// reserving 16 slots for each only inflates the tables and ledgers.
			group.capacity = loweredGroup.members.size() <= 1 ? 2u
				: (std::max<uint32_t>)(16, static_cast<uint32_t>(loweredGroup.members.size() * 2));
			group.group = edit.AddGroup(loweredGroup.shape, {});
			const auto reserved = edit.ReserveGroupMembers(group.group, group.capacity);
			for (const auto slot : reserved) {
				auto& entry = ClaimEntry(state, slot);
				entry.slot = slot;
				entry.shape = loweredGroup.shape;
				group.freeMembers.push_back(static_cast<uint32_t>(group.memberEntries.size()));
				group.memberEntries.push_back(slot.index);
			group.memberReplacedIDs.emplace_back();
			}
			groupIndex = static_cast<uint32_t>(state.groups.size());
			state.groups.push_back(std::move(group));
		}
		// The pass's uses name the members of its own capture; each needs a member slot before its tokens are made. A new
		// group starts as that capture, in its order; a group behind it moves forward to it, as the poll would have (a
		// rotated member replaced in place). The poll moves it on from there.
		if (created || DeclarationIsAhead(state.groups[groupIndex], loweredGroup.members)) {
			GrowGroup(state, edit, groupIndex, loweredGroup.members.size());
			SyncGroupMembers(state, edit, state.groups[groupIndex], loweredGroup.members);
			state.groups[groupIndex].identity = loweredGroup.identity;
			if (!created && touchedGroups) touchedGroups->push_back(groupIndex);
		}
		auto& group = state.groups[groupIndex];
		group.subscribers.push_back(mainIndex);
		for (const auto entryIndex : group.memberEntries) state.entries[entryIndex].subscribers.push_back(mainIndex);
		for (const auto& [range, stateValue] : loweredGroup.templates)
			edit.DeclareGroupAccess(main.id, group.group, stateValue, mainIndex, range);
		main.groups.push_back(groupIndex);
	}
    if (main.resourceUses) {
        // Requirements belong to resolver positions, including reserved capacity,
        // rather than to the transient resource IDs captured by Declare.
        for (const auto& requirement : main.resourceUses->resolverViews) {
            for (const auto groupIndex : main.groups) {
                const auto& group = state.groups[groupIndex];
                if (group.key != requirement.identity) continue;
                main.groupViews.push_back({groupIndex, requirement.views});
                for (const auto index : group.memberEntries) {
                    auto found = tokens.find(index);
                    if (found == tokens.end()) found = tokens.emplace(index,
                        edit.BindGroupMember(main.id, group.group, state.entries[index].slot)).first;
                    for (const auto view : requirement.views) edit.RequireView(found->second, view);
                }
            }
        }
        for (const auto& use : main.resourceUses->uses) {
            main.declaredViews.emplace_back();
            main.declaredBindings.emplace_back();
            uint32_t index = UINT32_MAX;
            if (use.resolverIdentity) for (const auto groupIndex : main.groups) {
                const auto& group = state.groups[groupIndex];
                if (group.key != use.resolverIdentity) continue;
                if (const auto position = MemberPosition(group, use.binding.globalResourceID)) index = group.memberEntries[*position];
                break;
            }
            main.resolverUseEntries.push_back(index);
            if (use.requiredViews.empty()) continue;
            // A direct use, or a resolver lowered direct (mixed member shapes): its members are direct entries. A resource
            // declared by pointer may also be another resolver's member.
            const auto names = [&](uint32_t candidate) {
                const auto& entry = state.entries[candidate];
                return entry.resourceID == use.binding.globalResourceID
                    || (entry.resource && entry.resource->GetGlobalResourceID() == use.binding.registryResourceID);
            };
            if (index == UINT32_MAX) for (const auto candidate : main.directEntries) if (names(candidate)) { index = candidate; break; }
            if (index == UINT32_MAX) for (const auto groupIndex : main.groups) {
                for (const auto candidate : state.groups[groupIndex].memberEntries) if (names(candidate)) { index = candidate; break; }
                if (index != UINT32_MAX) break;
            }
            if (index == UINT32_MAX) throw std::logic_error("Declared view has no lowered resource slot: pass='"
                + main.name + "' resource=" + std::to_string(use.binding.globalResourceID)
                + " registry=" + std::to_string(use.binding.registryResourceID));
            auto token = tokens.find(index);
            if (token == tokens.end()) token = tokens.emplace(index,
                edit.DeclareDependency(main.id, state.entries[index].slot, false)).first;
            main.declaredBindings.back() = token->second;
            for (const auto view : use.requiredViews)
                main.declaredViews.back().push_back(edit.RequireView(token->second, view));
        }
    }
	// Direct slots the new lowering no longer declares, and no other pass does: released with the pass's old declaration.
	// Kept, they would name a resource nothing keeps alive (a resolver's old member) and still be polled every frame.
	for (const auto entryIndex : previousDirect) {
		auto& entry = state.entries[entryIndex];
		if (!entry.direct || entry.retired || !entry.subscribers.empty()) continue;
		edit.RemoveResource(entry.slot);
		if (auto found = state.entriesByResourceID.find(entry.resourceID); found != state.entriesByResourceID.end()) {
			std::erase(found->second, entryIndex);
			if (found->second.empty()) state.entriesByResourceID.erase(found);
		}
		entry = {};
		entry.retired = true;
		basic_telemetry::AddCounter("ORG.Persistent.RetiredDirectSlots");
	}
	(void)graph;
}

// ---------------------------------------------------------------------------
// Bootstrap

static void LogRelowerDiff(const std::string& name, const State::MainPass& main, const LoweredPass& lowered) {
	static std::atomic<uint32_t> reported{0};
	if (reported.fetch_add(1) >= 48) return;
	std::string added, removed;
	for (const auto& use : lowered.entries) {
		bool found = false;
		for (const auto& d : main.loweredDirect) found = found || d.first == use.resourceID;
		if (!found) added += (use.resource ? use.resource->GetName() : std::string("?")) + "(" + std::to_string(use.resourceID) + ") ";
	}
	for (const auto& [id, resourceName] : main.loweredDirect) {
		bool found = false;
		for (const auto& use : lowered.entries) found = found || use.resourceID == id;
		if (!found) removed += resourceName + "(" + std::to_string(id) + ") ";
	}
	size_t keyChanges = 0;
	for (const auto& group : lowered.groups) {
		bool found = false;
		for (const auto* key : main.loweredGroupKeys) found = found || key == group.key.get();
		if (!found) ++keyChanges;
	}
	spdlog::info("Persistent structural relower: pass='{}' direct={} groups={} newGroupKeys={} added=[{}] removed=[{}]",
		name, lowered.entries.size(), lowered.groups.size(), keyChanges, added, removed);
}

// True when the new lowering declares a direct resource or group the pass
// did not declare before (the live publication has no slot for it).
bool persistent_impl::LoweringAddsDeclarations(const State::MainPass& main, const LoweredPass& lowered) {
    // A new view layout cannot prepare against the old publication while its
    // structural edit is pending, even when the resource set is unchanged.
    if (lowered.resourceUses && !lowered.resourceUses->uses.empty()
        && lowered.resourceUses != main.resourceUses) return true;
	for (const auto& use : lowered.entries) {
		bool known = false;
		for (const auto& previous : main.loweredDirect) if (previous.first == use.resourceID) { known = true; break; }
		if (!known) return true;
	}
	for (const auto& group : lowered.groups) {
		bool known = false;
		for (const auto* key : main.loweredGroupKeys) if (key == group.key.get()) { known = true; break; }
		if (!known) return true;
	}
	return false;
}

void RenderGraph::PreparePersistentFrame(rhi::Device device, uint8_t frameIndex, const IHostExecutionData* hostData, float deltaTime) {
	BT_ZONE_SCOPE("ORG.Persistent.PrepareFrame");
	auto& compiler = *m_compilerState;
	const bool bootstrap = !compiler.persistent;
	if (bootstrap) {
		BT_ZONE_SCOPE("ORG.Persistent.Bootstrap");
		compiler.persistent = std::make_shared<PersistentExecutionState>();
		auto& state = *compiler.persistent;
		const size_t framesInFlight = m_renderGraphSettingsService ? m_renderGraphSettingsService->GetNumFramesInFlight() : 3;
		state.admission = std::make_unique<persistent::SynchronousAdmission>(framesInFlight * 3 + 3,
			[this](std::shared_ptr<const void> owner) {
				if (!m_compilerState->ownershipRetirementScope)
					m_compilerState->ownershipRetirementScope = m_taskService->CreateScope("ORG.Ownership.Retirement");
				if (!m_taskService->Submit(m_compilerState->ownershipRetirementScope, runtime::TaskPriority::Background,
					"ORG.Ownership.RetirePublication", [owner = std::move(owner)]() mutable { owner.reset(); }))
					basic_telemetry::AddCounter("ORG.Ownership.RetirementRejected");
			}, m_runtime->Sequencer());
		state.admission->SetClosedExecutions(m_persistentClosedExecutions);
		state.ensureRepresentations = [this](Resource& resource) { EnsurePersistentRepresentations(resource); };
		// Tickets record their uploads on the graphics queue, or on the queue that first uses them (RecordPendingUploads).
		if (m_queueRegistry.SlotCount())
			state.admission->SetUploadQueue(static_cast<uint32_t>(ToUnderlying(m_queueRegistry.FindGraphicsSlot())));
		// Classify master passes into segments.
		state.mainPassByMaster.assign(m_masterPassList.size(), UINT32_MAX);
		state.masterHostedInMain.assign(m_masterPassList.size(), 0);
		std::vector<Resource*> usedResources;
		for (size_t masterIndex = 0; masterIndex < m_masterPassList.size(); ++masterIndex) {
			auto& any = m_masterPassList[masterIndex];
			auto kind = PersistentSegmentKind::Main;
			if (const auto found = m_persistentSegmentKinds.find(any.name); found != m_persistentSegmentKinds.end()) kind = found->second;
			uint32_t epoch = persistent::AllEpochs;
			std::visit([&](auto& value) {
				using T = std::decay_t<decltype(value)>;
				if constexpr (!std::is_same_v<T, std::monostate>) {
					epoch = value.epoch;
					if (!value.pass->UsesTypedPreparation() && dynamic_cast<IHasImmediateModeCommands*>(value.pass.get()))
						kind = PersistentSegmentKind::Pre;
				}
			}, any.pass);
			if (kind == PersistentSegmentKind::Pre) state.preMasterIndices.push_back(masterIndex);
			else if (kind == PersistentSegmentKind::Tail) state.tailMasterIndices.push_back(masterIndex);
			else {
				state.mainPassByMaster[masterIndex] = static_cast<uint32_t>(state.mainPasses.size());
				state.masterHostedInMain[masterIndex] = 1;
				state.mainPasses.push_back({masterIndex, {}, any.name});
				state.mainPasses.back().epoch = epoch;
				const auto view = GetPassView(any);
				for (const auto& requirement : view.reqs) {
					auto* resource = requirement.resourceHandleAndRange.resource.IsEphemeral()
						? requirement.resourceHandleAndRange.resource.GetEphemeralPtr()
						: _registry.Resolve(requirement.resourceHandleAndRange.resource);
					if (resource) usedResources.push_back(resource);
				}
			}
		}
		std::sort(usedResources.begin(), usedResources.end());
		usedResources.erase(std::unique(usedResources.begin(), usedResources.end()), usedResources.end());
		MaterializePersistentStandalone(usedResources, true);
		const auto primaryBackend = m_backendDevices.empty() ? rhi::Backend::Null : m_backendDevices.front().backend;
		auto edit = state.program.BeginEdit();
		edit.SetQueues(RegistryQueues(m_queueRegistry, m_runtime->Backend()));
		edit.SetEpochOrder(m_persistentEpochOrder);
		for (uint32_t mainIndex = 0; mainIndex < state.mainPasses.size(); ++mainIndex) {
			auto& main = state.mainPasses[mainIndex];
			auto& any = m_masterPassList[main.masterIndex];
			std::visit([&](auto& value) {
				using T = std::decay_t<decltype(value)>;
				if constexpr (!std::is_same_v<T, std::monostate>) {
					InstallMainPass(*this, state, edit, main,
						lowering::LowerPass(*this, _registry, m_queueRegistry, primaryBackend, value, any.type), mainIndex, false);
				}
			}, any.pass);
		}
		// Explicit structural ordering: the master list is already topologically
		// merged, so registration order is the authored order for equal phases.
		std::unordered_map<std::string, uint32_t> byName;
		for (uint32_t i = 0; i < state.mainPasses.size(); ++i) byName.emplace(state.mainPasses[i].name, i);
		for (const auto& [before, after] : m_structuralExplicitAfterByName) {
			const auto a = byName.find(before), b = byName.find(after);
			if (a != byName.end() && b != byName.end() && a->second != b->second)
				edit.AddOrdering(state.mainPasses[a->second].id, state.mainPasses[b->second].id);
		}
		// Registration order is preserved through originalOrder for tie-breaking;
		// the legacy compiler never treated list order as a hard edge either.
		const auto started = std::chrono::steady_clock::now();
		auto ready = [&] {
			BT_ZONE_SCOPE("ORG.Persistent.InlineStructuralBuild");
			return BuildPersistentStructural(edit);
		}();
		if (!ready || !state.program.Install(edit, ready)) throw std::runtime_error("Persistent graph bootstrap failed to build");
		++state.structuralBuilds;
		basic_telemetry::AddCounter("ORG.Persistent.StructuralBuilds");
		size_t unbound = 0; std::string unboundNames;
		for (const auto& entry : state.entries) {
			if (!entry.resource || entry.swapchain || entry.isBound) continue;
			if (unbound++ < 8) unboundNames += (unboundNames.empty() ? "" : ", ") + entry.resource->GetName();
		}
		if (unbound) spdlog::warn("Persistent graph bootstrap left {} declared slots unbound (first: {})", unbound, unboundNames);
		{
			// Diagnostic: physical resources bound in several groups (or a group
			// and a direct slot) share one equivalence class; joining another
			// group later is structural. Report the overlapping groups once.
			std::unordered_map<uint64_t, std::vector<uint32_t>> groupsByResource;
			for (uint32_t g = 0; g < state.groups.size(); ++g)
				for (const auto& [id, position] : state.groups[g].memberIndexByID) groupsByResource[id].push_back(g);
			std::map<std::vector<uint32_t>, uint32_t> sharedSets;
			for (const auto& [id, groups] : groupsByResource) if (groups.size() > 1) ++sharedSets[groups];
			for (const auto& [groups, count] : sharedSets) {
				std::string names;
				for (const auto g : groups) {
					const auto& group = state.groups[g];
					const IResourceResolver* resolver = group.resolver.get();
					names += (names.empty() ? "" : " | ") + std::string(resolver ? typeid(*resolver).name() : "?")
						+ "#" + std::to_string(g) + "(members=" + std::to_string(group.memberIndexByID.size()) + ",subs=" + std::to_string(group.subscribers.size())
						+ (group.subscribers.empty() ? "" : ",first=" + state.mainPasses[group.subscribers.front()].name) + ")";
				}
				spdlog::info("Persistent groups sharing {} member resource(s): {}", count, names);
			}
			size_t directAndGroup = 0;
			for (const auto& [id, entries] : state.entriesByResourceID) {
				bool direct = false, member = false;
				for (const auto index : entries) (state.entries[index].direct ? direct : member) = true;
				if (direct && member) {
					++directAndGroup;
					std::string detail;
					for (const auto index : entries) {
						const auto& entry = state.entries[index];
						detail += " slot=" + std::to_string(index) + (entry.direct ? "(direct" : "(member") + " subs=";
						for (const auto sub : entry.subscribers) detail += state.mainPasses[sub].name + ",";
						detail += ")";
					}
					spdlog::info("Persistent direct+group resource '{}' id={}:{}", state.entries[entries.front()].resource->GetName(), id, detail);
				}
			}
			spdlog::info("Persistent slot directory: resources={} sharedAcrossGroups={} directAndGroup={}", state.entriesByResourceID.size(),
				std::count_if(groupsByResource.begin(), groupsByResource.end(), [](const auto& kv) { return kv.second.size() > 1; }), directAndGroup);
		}
		if (const char* dump = std::getenv("ORG_PERSISTENT_DUMP_SLOTS")) {
			std::string list(dump);
			for (size_t pos = 0; pos < list.size();) {
				const auto next = list.find(',', pos);
				const auto index = static_cast<uint32_t>(std::stoul(list.substr(pos, next == std::string::npos ? std::string::npos : next - pos)));
				pos = next == std::string::npos ? list.size() : next + 1;
				if (index >= state.entries.size()) continue;
				const auto& entry = state.entries[index];
				std::string subs; for (const auto sub : entry.subscribers) subs += state.mainPasses[sub].name + ",";
				std::string groupName = "-";
				for (uint32_t g = 0; g < state.groups.size(); ++g)
					if (std::find(state.groups[g].memberEntries.begin(), state.groups[g].memberEntries.end(), index) != state.groups[g].memberEntries.end()) {
						const IResourceResolver* resolver = state.groups[g].resolver.get();
						groupName = std::to_string(g) + ":" + (resolver ? typeid(*resolver).name() : "?") + "(members=" + std::to_string(state.groups[g].memberIndexByID.size()) + ")";
					}
				spdlog::info("Persistent slot {}: resource='{}' id={} direct={} group={} subs=[{}]", index,
					entry.resource ? entry.resource->GetName() : "-", entry.resourceID, entry.direct, groupName, subs);
			}
		}
		spdlog::info("Persistent graph bootstrap: mainPasses={} pre={} tail={} slots={} groups={} batches={} build_ms={:.2f}",
			state.mainPasses.size(), state.preMasterIndices.size(), state.tailMasterIndices.size(), state.entries.size(), state.groups.size(),
			ready->executable->graph->batches.size(),
			std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
	}
	auto& state = *compiler.persistent;
	state.frameIndex = frameIndex;
	state.frameNumber = ++compiler.asyncPreparationFrameNumber;
	state.frameData = m_asyncUpdateHostData;
	RetireIdlePersistentAliasPools(state);
	org::runtime::ScopedActiveGraphServices activeServices(m_uploadService.get(), m_descriptorService.get());
	const auto primaryBackend = m_backendDevices.empty() ? rhi::Backend::Null : m_backendDevices.front().backend;

	// 0. Per-frame extension passes, gathered before declarations are polled as
	// in the fresh compiler: frame hooks may publish work that retained passes
	// consume this frame (streaming uploads).
	auto& frameExt = compiler.frameExtensions;
	std::vector<ExternalPassDesc> interrupting;
	{
		BT_ZONE_SCOPE("ORG.Persistent.GatherFramePasses");
		frameExt.clear();
		for (auto& ext : m_extensions) if (ext) ext->GatherFramePasses(*this, frameExt);
		for (auto it = frameExt.begin(); it != frameExt.end();) {
			if (!it->interruptsFrame) { ++it; continue; }
			interrupting.push_back(std::move(*it));
			it = frameExt.erase(it);
		}
		if (m_drivesFrameServices) PlacePersistentCaptures(*this, state, m_persistentTicketTarget != nullptr, frameExt, interrupting);
	}
	// 1. Structural and membership changes reported by legacy passes. One
	// transaction carries both so the slot directory never diverges from the
	// installed program: a directory change is only made against an edit that
	// is built and installed below (or throws).
	{
		BT_ZONE_SCOPE("ORG.Persistent.PollDeclarations");
		std::vector<uint32_t> structural;
		std::vector<uint32_t> growGroups;
		std::unordered_map<uint32_t, LoweredPass> relowered;
		std::optional<persistent::GraphEditTransaction> edit;
		AdvancePersistentStructuralBuild(structural);
		for (auto& desc : UpdatePersistentFrameInterruptingPasses(std::move(interrupting), primaryBackend))
			frameExt.push_back(std::move(desc));
		// Alias re-placement: on the ORG_PERSISTENT_FORCE_REALIAS interval, or when the packing strategy changed since the
		// latest plan (plans run only on structural builds).
		const auto interval = ForcedRealias().interval;
		const bool realiasInterval = interval && state.frameNumber % interval == 0;
		const bool packingChanged = m_getAutoAliasPackingStrategy && m_getAutoAliasPackingStrategy() != state.aliasPackingStrategy;
		if ((realiasInterval || packingChanged) && !state.pending) {
			BT_ZONE_SCOPE("ORG.Persistent.ForcedRealias");
			auto forced = state.program.BeginEdit();
			// Re-placement is only valid in a structural edit (the executable must
			// re-validate the new overlaps). Dropping the orderings makes it one;
			// the planner re-derives them for the new placements.
			forced.ClearPlacementOrderings();
			state.aliasEdges.clear();
			state.forceNewAliasHeaps = true;
			auto ready = BuildPersistentStructural(forced);
			state.forceNewAliasHeaps = false;
			if (!ready || !state.program.Install(forced, ready)) throw std::runtime_error("Persistent forced alias re-placement failed to install");
			++state.structuralBuilds;
			spdlog::info("Persistent alias re-placement ({}): frame={}", packingChanged ? "packing strategy changed" : "forced", state.frameNumber);
		}
		auto ensureEdit = [&]() -> persistent::GraphEditTransaction& {
			if (!edit) edit.emplace(state.program.BeginEdit());
			return *edit;
		};
		size_t refreshed = 0;
		// Set when a relower cannot wait for a worker build: the frame would prepare the pass against a publication that does
		// not have its declarations.
		bool inlineStructural = false;
		{ BT_ZONE_SCOPE("ORG.Persistent.PollDynamicDeclared");
		for (uint32_t mainIndex = 0; mainIndex < state.mainPasses.size(); ++mainIndex) {
			auto& main = state.mainPasses[mainIndex];
			if (main.hosted || main.retired) continue; // One-shot; never relowered.
			auto& any = m_masterPassList[main.masterIndex];
			std::visit([&](auto& value) {
				using T = std::decay_t<decltype(value)>;
				if constexpr (!std::is_same_v<T, std::monostate>) {
					auto* dynamic = dynamic_cast<IDynamicDeclaredResources*>(value.pass.get());
					const bool declared = dynamic && dynamic->DeclaredResourcesChanged() && !dynamic->DeclarationsProvidedByImmediateCommands();
					// A resolver declared directly is lowered as the set it resolved to; another set means the pass is lowered
					// again, from a fresh declaration, before it is prepared.
					bool directStale = false;
					for (const auto& [resolver, identity] : main.directResolvers) {
						const auto captured = CaptureResolverDeclarationState(*resolver);
						if (captured && !(captured->resourceSetIdentity == identity)) { directStale = true; break; }
					}
					if (!declared && !directStale) return;
					BT_ZONE_SCOPE("ORG.Persistent.RefreshDeclaration");
					BT_ZONE_TEXT(any.name.data(), any.name.size());
					++refreshed;
					RedeclarePass(value, any.type);
					auto lowered = lowering::LowerPass(*this, _registry, m_queueRegistry, primaryBackend, value, any.type);
					// Membership-only change: the same structure under the same declared layout. A new layout re-issues the pass's
					// binding tokens, which only a reinstalled lowering resolves.
					if (FingerprintLowering(lowered) == main.fingerprint && lowered.resourceUses == main.resourceUses) {
						main.directResolvers = std::move(lowered.directResolvers);
						return;
					}
					LogRelowerDiff(any.name, main, lowered);
					const bool mustInline = directStale || LoweringAddsDeclarations(main, lowered);
					if (state.pending) {
						if (!mustInline) { state.pending->deferredStructural.push_back(mainIndex); return; }
						DrainPersistentStructuralBuild(structural);
					}
					inlineStructural |= mustInline;
					relowered.emplace(mainIndex, std::move(lowered));
					structural.push_back(mainIndex);
				}
			}, any.pass);
		}
		}
		BT_PLOT("ORG.Persistent.RefreshedDeclarations", static_cast<int64_t>(refreshed));
		// Resolver membership: within reserved capacity this is a binding edit.
		{ BT_ZONE_SCOPE("ORG.Persistent.PollGroups");
		for (uint32_t groupIndex = 0; groupIndex < state.groups.size(); ++groupIndex) {
			auto& group = state.groups[groupIndex];
			if (group.subscribers.empty() || !group.resolver) continue;
			if (state.pending && std::find(state.pending->touchedGroups.begin(), state.pending->touchedGroups.end(), groupIndex) != state.pending->touchedGroups.end()) continue;
			const auto hint = group.resolver->DeclarationVersionHint();
			if (hint && hint == group.versionHint) continue;
			std::shared_ptr<const ResolverDeclarationState> captured;
			{
				BT_ZONE_SCOPE("ORG.Persistent.PollGroups.Capture");
				captured = CaptureResolverDeclarationState(*group.resolver);
			}
			if (!captured) continue;
			group.versionHint = hint;
			if (captured->waitRevision != group.waitRevision) {
				group.waitRevision = captured->waitRevision;
				group.waits = captured->waits ? *captured->waits : std::vector<ExternalTimelinePoint>{};
			}
			if (captured->resourceSetIdentity == group.identity) continue;
			BT_ZONE_SCOPE("ORG.Persistent.PollGroups.Diff");
			if (!group.subscribers.empty()) { const auto& n = state.mainPasses[group.subscribers.front()].name; BT_ZONE_TEXT(n.data(), n.size()); }
			size_t added = 0, removed = 0;
			std::vector<std::pair<uint64_t, std::shared_ptr<Resource>>> members;
			bool uniform = true;
			if (captured->resources) for (const auto& member : *captured->resources) {
				if (!member) continue;
				const auto id = member->GetSchedulingResourceID();
				// Shapes of already-bound members were checked when they joined.
				if (!group.memberIndexByID.contains(id) && ShapeOf(*member) != group.shape) uniform = false;
				members.emplace_back(id, member);
			}
			if (!uniform) {
				// Shape contract broken: subscribers re-lower (direct declarations).
				{
					static std::atomic<uint32_t> reported{0};
					if (reported.fetch_add(1) < 8)
						spdlog::info("Persistent group shape changed: subscriber='{}' members={} capacity={}", state.mainPasses[group.subscribers.front()].name,
							members.size(), group.capacity);
				}
				if (state.pending) { group.versionHint = 0; for (const auto subscriber : group.subscribers) state.pending->deferredStructural.push_back(subscriber); continue; }
				for (const auto subscriber : group.subscribers) structural.push_back(subscriber);
				continue;
			}
			if (members.size() > group.capacity) {
				if (state.pending) { group.versionHint = 0; continue; } // re-detected after the install
				growGroups.push_back(groupIndex);
				continue;
			}
			auto& transaction = ensureEdit();
			{
				BT_ZONE_SCOPE("ORG.Persistent.PollGroups.Diff.Materialize");
				std::vector<Resource*> arriving;
				for (const auto& [id, member] : members) if (!group.memberIndexByID.contains(id)) arriving.push_back(member.get());
				MaterializePersistentStandalone(arriving);
			}
			const auto counts = SyncGroupMembers(state, transaction, group, members);
			added = counts.added;
			removed = counts.removed;
			group.identity = captured->resourceSetIdentity;
			BT_ZONE_VALUE(static_cast<int64_t>(added + removed));
			basic_telemetry::AddCounter("ORG.Persistent.MembershipEdits");
		}
		}
		std::sort(structural.begin(), structural.end());
		structural.erase(std::unique(structural.begin(), structural.end()), structural.end());
		// Backing rotation, in the same transaction as membership: a rotating
		// wrapper and a group slot that tracks the same backing must move
		// together, or the backing's equivalence class changes between the two
		// edits (structural).
		size_t rotated = 0;
		{
			BT_ZONE_SCOPE("ORG.Persistent.PollBackings");
			for (uint32_t index = 0; index < state.entries.size(); ++index) {
				auto& entry = state.entries[index];
				if (!entry.resource || entry.swapchain) continue;
				const auto key = RotationKeyOf(entry);
				if (entry.isBound && key == entry.bound) continue;
				if (!entry.isBound && !key.concreteID) continue;
				auto& transaction = ensureEdit();
				// Rotated backings may not be materialized yet (per-frame presentation
				// targets are realized lazily by the legacy path).
				MaterializePersistentStandalone(std::span<Resource* const>{&entry.resource, 1});
				// Slot maps are keyed by resource IDs: only a wrapper whose concrete
				// backing object changed exposes a new ID to its subscribers.
				const bool concreteChanged = key.concreteID != entry.bound.concreteID;
				const bool wasBound = entry.isBound;
				const auto previousConcreteID = entry.bound.concreteID;
				if (BindEntry(state, transaction, index)) {
					++rotated;
					if (concreteChanged || !wasBound) {
						if (wasBound) QueueSlotPatch(state, entry, previousConcreteID, false);
						QueueSlotPatch(state, entry, entry.bound.concreteID, true);
					}
				}
				{
					static std::atomic<uint32_t> reported{0};
					if (reported.load() < 6 && entry.concrete)
						for (uint32_t other = 0; other < state.entries.size(); ++other) {
							const auto& candidate = state.entries[other];
							if (other == index || !candidate.isBound || candidate.concrete != entry.concrete) continue;
							if (reported.fetch_add(1) < 6)
								spdlog::info("Persistent rotation shares a backing: slot={} resource='{}' direct={} -> backing '{}' also at slot={} resource='{}' direct={} groupSubs={}",
									index, entry.resource->GetName(), entry.direct, entry.concrete->GetName(), other,
									candidate.resource ? candidate.resource->GetName() : "?", candidate.direct,
									candidate.subscribers.empty() ? "" : state.mainPasses[candidate.subscribers.front()].name);
							break;
						}
				}
			}
			BT_PLOT("ORG.Persistent.RotatedBindings", static_cast<int64_t>(rotated));
		}
		if (edit) {
			BT_ZONE_SCOPE("ORG.Persistent.InlineEditBuild");
			std::atomic_bool cancelled{false};
			auto ready = edit->Build(compiler.synchronousCompileWorkspace, cancelled);
			if (!ready || !state.program.Install(*edit, ready)) throw std::runtime_error("Persistent edit failed to install");
			++state.bindingBuilds;
			if (rotated) basic_telemetry::AddCounter("ORG.Persistent.BindingEdits", static_cast<int64_t>(rotated));
		}
		// Structural edits build on a worker against the publication that now
		// includes this frame's binding edits.
		if (!state.pending && (!structural.empty() || !growGroups.empty())) {
			// What asked for the build: the first builds, then every 256th, so a build every frame names its cause.
			static std::atomic<uint64_t> submitted{0};
			if (const auto count = submitted.fetch_add(1); count < 16 || count % 256 == 0) {
				std::string passes, groups;
				for (const auto index : structural) passes += state.mainPasses[index].name + " ";
				for (const auto index : growGroups) {
					const auto& group = state.groups[index];
					groups += (group.subscribers.empty() ? std::string("?") : state.mainPasses[group.subscribers.front()].name)
						+ "(capacity=" + std::to_string(group.capacity) + ") ";
				}
				spdlog::info("Persistent structural build #{} requested: frame={} relower=[{}] grow=[{}]", count, state.frameNumber, passes, groups);
			}
			SubmitPersistentStructuralBuild(structural, growGroups, primaryBackend, inlineStructural);
		}
		// Direct declarations of a resolver bind the set it resolved to when the pass was lowered. A preparation that resolves
		// another set would record against resources it does not name - read past a grown buffer's old backing, say. A change
		// is relowered inline above, so by here every pass must match; never silently.
		{ BT_ZONE_SCOPE("ORG.Persistent.CheckDirectResolvers");
		for (const auto& main : state.mainPasses) {
			if (main.hosted || main.retired) continue;
			for (const auto& [resolver, identity] : main.directResolvers) {
				const auto captured = CaptureResolverDeclarationState(*resolver);
				if (captured && !(captured->resourceSetIdentity == identity))
					throw std::logic_error("Persistent pass '" + main.name
						+ "' binds a resolver's resources directly, as they were when it was lowered, but this preparation resolves "
						  "another set of them");
			}
		}
		}
	}
	// 3. Select and prepare the main segment.
	auto selected = state.program.Select();
	if (m_getRenderGraphCompileDumpEnabled && m_getRenderGraphCompileDumpEnabled() && state.dumpedExecutable != selected->executable.get()) {
		state.dumpedExecutable = selected->executable.get();
		WritePersistentGraphDebugDump(*selected);
	}
	if (m_scheduleTraceEnabled.load(std::memory_order_relaxed)) RefreshPersistentScheduleTraces(*selected);
	else if (state.scheduleTraces.executable) {
		state.scheduleTraces = {};
		m_scheduleSnapshot.store(nullptr, std::memory_order_release);
	}
	if (state.bindingsFor != selected.get()) {
		BT_ZONE_SCOPE("ORG.Persistent.BuildLegacyBindingTable");
		std::vector<FrozenExecutionBindings::ResourceBinding> table(selected->bindings.Size());
		for (uint32_t index = 0; index < selected->bindings.Size(); ++index) {
			const auto* version = selected->bindings.TryAt(index);
			if (!version || !version->bound || !version->recording) continue;
			const auto& recording = *version->recording;
			table[index] = {recording.resource, recording.allocationOwner, recording.views, recording.descriptorOwner, nullptr};
		}
		state.bindings = FrozenExecutionBindings::Sparse(std::move(table));
		state.bindingsFor = selected.get();
		for (auto& main : state.mainPasses)
			if (main.slotsExecutable != selected->executable.get()) main.slotsDirty = true;
	}
	if (!compiler.ownershipRetirementScope)
		compiler.ownershipRetirementScope = m_taskService->CreateScope("ORG.Ownership.Retirement");
	auto& invocationArena = compiler.invocationArenas[frameIndex];
	if (!invocationArena) invocationArena = std::make_shared<PreparedInvocationArena>();
	FramePreparationContext preparation{
		.invocationArena = invocationArena,
		.retireOwnership = [tasks = m_taskService, scope = compiler.ownershipRetirementScope](std::shared_ptr<const void> owner) {
			if (!tasks->Submit(scope, runtime::TaskPriority::Background, "ORG.Ownership.Retire",
				[owner = std::move(owner)]() mutable { owner.reset(); }))
				basic_telemetry::AddCounter("ORG.Ownership.RetirementRejected");
		},
		.device = device,
		.frameIndex = frameIndex,
		.preparationSlot = frameIndex,
		.frameNumber = state.frameNumber,
		.deltaTime = deltaTime,
		.bindings = state.bindings,
		.preparationData = m_asyncUpdateHostData ? m_asyncUpdateHostData.get() : hostData,
		.admissionData = m_asyncUpdateHostData ? m_asyncUpdateHostData.get() : hostData,
		.frameData = m_asyncUpdateHostData,
	};
	// Swapchain slots are rebound per frame from the wrapper's current backing.
	std::vector<persistent::FrameRebinding> rebindings;
	std::vector<FrozenExecutionBindings::ResourceBinding> overlay;
	for (const auto& entry : state.entries) {
		if (!entry.swapchain || !entry.resource) continue;
		auto* concrete = UnwrapDynamic(entry.resource);
		if (!concrete) throw std::runtime_error("Persistent frame has no swapchain backing");
		auto snapshot = PublicationBindingBundle::Capture(*entry.resource);
		if (!snapshot) throw std::runtime_error("Persistent frame could not capture the swapchain backing");
		persistent::FrameRebinding rebinding;
		rebinding.slot = entry.slot;
		rebinding.backing.graphResourceID = uint64_t{entry.slot.index} + 1;
		if (trace::Any()) rebinding.backing.debugName = entry.resource->GetName();
		rebinding.backing.resource = snapshot->resource.GetHandle();
		rebinding.backing.shape = entry.shape;
		rebinding.backing.regions = CaptureRegions(*concrete, entry.shape);
		rhi::ResourceDesc description{};
		if (concrete->TryGetRHIResourceDesc(description)) rebinding.backing.heapType = description.heapType;
		rebinding.recording = snapshot;
		if (overlay.empty()) overlay = state.bindings->Resources();
		overlay[entry.slot.index] = {snapshot->resource, snapshot->allocationOwner, snapshot->views, snapshot->descriptorOwner, nullptr};
		rebindings.push_back(std::move(rebinding));
	}
	if (!overlay.empty()) preparation.bindings = FrozenExecutionBindings::Sparse(std::move(overlay));
	// The epoch's own executable when the program has epochs; an epoch with none of its
	// passes has no main segment at all (its Pre/Tail work still runs).
	std::shared_ptr<const persistent::SelectedPublication> executing = selected;
	if (m_persistentEpoch != persistent::AllEpochs && selected->HasEpochs()) executing = selected->ForEpoch(m_persistentEpoch);
	PersistentExecutionState::PreparedSegment main;
	main.label = "Main";
	main.publication = executing;
	main.rebindings = std::move(rebindings);
	main.legacyBindings = preparation.bindings;
	main.invocations.resize(selected->logical->passSlots.size());
	{
		BT_ZONE_SCOPE("ORG.Persistent.PrepareInvocations");
		auto passPreparation = preparation;
		state.prepareCostByMain.resize(state.mainPasses.size());
		// Pass names per logical slot only change with the executable.
		const bool namesDirty = !state.mainNames || state.mainNamesExecutable != selected->executable.get();
		std::shared_ptr<std::vector<std::string>> mainNames;
		if (namesDirty) {
			mainNames = std::make_shared<std::vector<std::string>>(selected->logical->passSlots.size());
			state.mainNames = mainNames;
			state.mainNamesExecutable = selected->executable.get();
		}
		main.names = state.mainNames;
		// Every pass's name, not only this epoch's: the table is rebuilt once per executable.
		if (namesDirty)
			for (const auto& pass : state.mainPasses)
				if (!pass.retired && pass.id.index < mainNames->size())
					(*mainNames)[pass.id.index] = (pass.hosted ? *pass.hosted : m_masterPassList[pass.masterIndex]).name;
		// A resolver use's resource tokens name the member it had when the pass declared it. They resolve as its views do: to
		// the resource bound now at that member's position (a versioned buffer's current version: VersionedBuffer), so a
		// membership change stays a binding edit.
		auto addResolverUses = [&](const auto& a_pass, FramePreparationContext::ResourceSlots& a_slots) {
			if (!a_pass.resourceUses) return;
			const auto& uses = a_pass.resourceUses->uses;
			for (size_t u = 0; u < uses.size() && u < a_pass.resolverUseEntries.size(); ++u) {
				const auto index = a_pass.resolverUseEntries[u];
				if (index == UINT32_MAX) continue;
				const auto& entry = state.entries[index];
				if (!entry.resource || !entry.isBound || entry.slot.index >= selected->bindings.Size()) continue;
				a_slots.emplace_back(uses[u].binding.registryResourceID, entry.slot.index);
				a_slots.emplace_back(uses[u].binding.globalResourceID, entry.slot.index);
			}
		};
		for (uint32_t mainIndex = 0; mainIndex < state.mainPasses.size() && executing; ++mainIndex) {
			auto& pass = state.mainPasses[mainIndex];
			if (pass.retired) continue;
			// Another epoch's pass: neither prepared nor recorded now. Its slot patches wait.
			if (!RunsInPersistentEpoch(pass.epoch)) continue;
			if (pass.slotsDirty || pass.slotsExecutable != selected->executable.get()) {
				auto slots = std::make_shared<FramePreparationContext::ResourceSlots>();
				auto expose = [&](uint32_t entryIndex) {
					const auto& entry = state.entries[entryIndex];
					if (!entry.resource || (!entry.isBound && !entry.swapchain)) return;
					const auto slot = entry.slot.index;
					if (slot >= selected->bindings.Size()) return; // reserved by a pending structural build
					slots->emplace_back(entry.resourceID, slot);
					slots->emplace_back(entry.resource->GetGlobalResourceID(), slot);
					if (auto* concrete = UnwrapDynamic(entry.resource)) slots->emplace_back(concrete->GetGlobalResourceID(), slot);
				};
				for (const auto index : pass.directEntries) expose(index);
				for (const auto groupIndex : pass.groups) for (const auto index : state.groups[groupIndex].memberEntries) expose(index);
				// Registry handle identities may differ from scheduling identities.
				const auto view = GetPassView(pass.hosted ? *pass.hosted : m_masterPassList[pass.masterIndex]);
				for (const auto& requirement : view.reqs) {
					auto* resource = requirement.resourceHandleAndRange.resource.IsEphemeral()
						? requirement.resourceHandleAndRange.resource.GetEphemeralPtr() : _registry.Resolve(requirement.resourceHandleAndRange.resource);
					if (!resource) continue;
					const auto found = state.entriesByResourceID.find(resource->GetSchedulingResourceID());
					if (found == state.entriesByResourceID.end()) continue;
					for (const auto entryIndex : found->second)
						if (std::find(pass.directEntries.begin(), pass.directEntries.end(), entryIndex) != pass.directEntries.end())
							slots->emplace_back(requirement.resourceHandleAndRange.resource.GetGlobalResourceID(), state.entries[entryIndex].slot.index);
				}
				addResolverUses(pass, *slots);
				std::sort(slots->begin(), slots->end());
				slots->erase(std::unique(slots->begin(), slots->end()), slots->end());
				slots->sorted = true;
				pass.slots = std::move(slots);
				pass.slotsExecutable = selected->executable.get();
				pass.slotsDirty = false;
				pass.slotAdds.clear(); pass.slotRemoves.clear();
			} else if (!pass.slotAdds.empty() || !pass.slotRemoves.empty()) {
				// Copy-on-write patch: typed passes key their recipe layout on the
				// object identity, so a changed map must be a new object.
				BT_ZONE_SCOPE("ORG.Persistent.PatchPassSlots");
				BT_ZONE_VALUE(static_cast<int64_t>(pass.slotAdds.size() + pass.slotRemoves.size()));
				auto slots = std::make_shared<FramePreparationContext::ResourceSlots>(*pass.slots);
				if (!pass.slotRemoves.empty()) {
					auto& removes = pass.slotRemoves;
					std::sort(removes.begin(), removes.end());
					slots->erase(std::remove_if(slots->begin(), slots->end(),
						[&](const auto& pair) { return std::binary_search(removes.begin(), removes.end(), pair); }), slots->end());
				}
				for (const auto& add : pass.slotAdds)
					if (add.second < selected->bindings.Size()) slots->push_back(add);
				// A removed member's own pair may have been a declared token's: map those again.
				addResolverUses(pass, *slots);
				std::sort(slots->begin(), slots->end());
				slots->erase(std::unique(slots->begin(), slots->end()), slots->end());
				slots->sorted = true;
				pass.slots = std::move(slots);
				pass.slotAdds.clear(); pass.slotRemoves.clear();
			}
			if (pass.id.index >= main.invocations.size()) continue; // added by a pending structural build
			auto& any = pass.hosted ? *pass.hosted : m_masterPassList[pass.masterIndex];
			if (pass.hosted && pass.hostedPrepared) {
				// Already ran; stays in the executable until its removal installs.
				main.invocations[pass.id.index] = PreparedPass::NoOp();
				if (namesDirty) (*mainNames)[pass.id.index] = any.name;
				continue;
			}
			pass.hostedPrepared = static_cast<bool>(pass.hosted);
			passPreparation.resourceSlots = pass.slots;
			passPreparation.backendInstance = pass.device;
			if (const auto* entry = m_backendDevices.Find(pass.device)) passPreparation.device = entry->device;
            passPreparation.resolveDeclaredView = [&pass, selected](uint32_t use, uint32_t view) {
                const auto slot = selected->Resolve(pass.declaredViews.at(use).at(view));
                return pass.device == BackendInstanceId::Primary ? slot
                    : DescriptorHeapManager::GetInstance().TranslateDescriptorSlot(slot, pass.device);
            };
            passPreparation.resolveDeclaredResource = [&pass, selected](uint32_t use) {
                return selected->ResolveNative(pass.declaredBindings.at(use));
            };
            // A name declared through a resolver group: its first member bound this frame, in capture order.
            passPreparation.resolveResolverView = [&pass, &state, bindings = passPreparation.bindings](
                uint32_t resolverGroup, BindlessViewRequest request) -> rhi::DescriptorSlot {
                if (!pass.resourceUses || resolverGroup >= pass.resourceUses->resolverViews.size() || !bindings) return {};
                const auto& identity = pass.resourceUses->resolverViews[resolverGroup].identity;
                for (const auto groupIndex : pass.groups) {
                    const auto& group = state.groups[groupIndex];
                    if (group.key != identity) continue;
                    for (const auto entryIndex : group.memberEntries) {
                        const auto& entry = state.entries[entryIndex];
                        if (!entry.resource || !entry.isBound || entry.slot.index >= bindings->Resources().size()) continue;
                        const auto slot = bindings->Views(PreparedResourceReference{entry.slot.index}).Resolve(request);
                        return pass.device == BackendInstanceId::Primary ? slot
                            : DescriptorHeapManager::GetInstance().TranslateDescriptorSlot(slot, pass.device);
                    }
                }
                return {};
            };

			PreparedPass packet;
			const auto prepareStarted = std::chrono::steady_clock::now();
			try {
				BT_ZONE_SCOPE("ORG.Persistent.PrepareInvocation");
				BT_ZONE_TEXT(any.name.data(), any.name.size());
				packet = std::visit([&](auto& value) -> PreparedPass {
					using T = std::decay_t<decltype(value)>;
					if constexpr (std::is_same_v<T, std::monostate>) return {};
					else return value.pass->PrepareFrame(passPreparation);
				}, any.pass);
			} catch (const std::exception& e) {
				throw std::runtime_error("Pass '" + any.name + "' persistent preparation failed: " + e.what());
			}
			if (!packet) throw std::runtime_error("Pass '" + any.name + "' has no owned preparation; persistent execution requires PrepareFrame");
			{
				auto& cost = state.prepareCostByMain[mainIndex];
				cost.first += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - prepareStarted).count());
				++cost.second;
			}
			// A capture anchored after this pass records in its packet: the packet is this frame's only.
			bool captured = false;
			if (!state.pendingCaptures.empty() && state.pendingCaptures.contains(any.name)) {
				packet = AppendPersistentCaptures(*this, state, pass, any.name, std::move(packet), passPreparation,
					executing ? *executing : *selected);
				captured = true;
			}
			packet.SetDebugName(any.name);
			main.invocations[pass.id.index] = std::move(packet);
			if (auto* ticket = m_persistentTicketTarget) {
				const RenderGraphPass* object = std::visit([](auto& value) -> const RenderGraphPass* {
					using T = std::decay_t<decltype(value)>;
					if constexpr (std::is_same_v<T, std::monostate>) return nullptr;
					else return value.pass.get();
				}, any.pass);
				uint64_t hash = 0;
				if (object && !pass.hosted && !captured && object->RevisionHash(hash)) ticket->revisions.emplace_back(object, hash);
				else ticket->uncheckable = true;
			}
			if (namesDirty) (*mainNames)[pass.id.index] = any.name;
			// External waits (explicit + resolver-provided) become per-frame producer waits.
			auto appendWait = [&](const ExternalTimelinePoint& wait) {
				if (!wait.timeline || !wait.value) return;
				const auto handle = wait.timeline.GetHandle();
				const uint64_t identity = m_runtime->TimelineIdentity(wait.timeline);
				bool bound = std::ranges::any_of(main.foreignTimelines, [&](const auto& binding) { return binding.identity == identity; });
				for (size_t slot = 0; slot < m_queueRegistry.SlotCount() && !bound; ++slot)
					bound = SameTimeline(m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot))), wait.timeline);
				if (!bound) main.foreignTimelines.push_back({identity, handle});
				main.waits.push_back({pass.id, {identity, wait.value}});
			};
			for (const auto& wait : pass.explicitWaits) appendWait(wait);
			for (const auto groupIndex : pass.groups) for (const auto& wait : state.groups[groupIndex].waits) appendWait(wait);
		}
	}
	if (executing) state.main = std::move(main);
	else state.main.reset();
	state.framePrepared = true;
	// 4. Dynamic segments. A ticket holds none: the host records its uploads when it submits the ticket, and
	// no other Pre/Tail pass is supported with tickets.
	state.pre.reset();
	state.tail.reset();
	if (m_persistentTicketTarget) {
		for (const auto index : state.preMasterIndices)
			if (m_masterPassList[index].name != "org.host.uploads")
				throw std::logic_error("Tickets do not support the Pre pass '" + m_masterPassList[index].name + "'");
		if (!state.tailMasterIndices.empty())
			throw std::logic_error("Tickets do not support Tail passes");
	} else {
		BT_ZONE_SCOPE("ORG.Persistent.PrepareDynamicSegments");
		auto buildSegment = [&](const char* label, std::span<const size_t> masterIndices,
			std::vector<AnyPassAndResources>& extensionPasses, bool immediate,
			PersistentExecutionState::SegmentCache& cache,
			std::vector<std::unique_ptr<PersistentExecutionState::SegmentProgram>>& programs) -> std::optional<PersistentExecutionState::PreparedSegment> {
			struct Entry { persistent::ResourceSlotId slot; Resource* resource; CompileResourceShape shape; };
			std::vector<AnyPassAndResources*> passes;
			for (const auto masterIndex : masterIndices) passes.push_back(&m_masterPassList[masterIndex]);
			for (auto& pass : extensionPasses) passes.push_back(&pass);
			if (passes.empty()) return std::nullopt;
			std::unordered_map<uint64_t, uint32_t> entriesByID;
			std::vector<Entry> entries;               // slot index == entry index (fresh path)
			std::vector<LoweredPass> loweredPasses;
			std::vector<AnyPassAndResources*> loweredOwners; // parallel to loweredPasses
			std::vector<persistent::PassId> ids;
			std::vector<PreparedPass> packets;
			std::vector<std::string> names;
			std::vector<persistent::FrameRebinding> rebindings;
			std::vector<std::pair<uint32_t, ExternalTimelinePoint>> waits;
			auto slotFor = [&](Resource* resource) -> uint32_t {
				const auto id = resource->GetSchedulingResourceID();
				if (const auto found = entriesByID.find(id); found != entriesByID.end()) return found->second;
				const auto index = static_cast<uint32_t>(entries.size());
				entries.push_back({persistent::ResourceSlotId{index, 0}, resource, ShapeOf(*resource)});
				entriesByID.emplace(id, index);
				return index;
			};
			BT_ZONE_SCOPE("ORG.Persistent.DynamicSegment");
			for (uint32_t passIndex = 0; passIndex < passes.size(); ++passIndex) {
				BT_ZONE_SCOPE("ORG.Persistent.DynamicSegment.Lower");
				auto& any = *passes[passIndex];
				const auto lowerStarted = std::chrono::steady_clock::now();
				struct CostScope {
					State& state; const std::string& name; std::chrono::steady_clock::time_point started;
					~CostScope() {
						// Extension passes are recreated per frame; only master-list passes are tracked.
						auto found = state.prepareCostByPass.find(name);
						if (found == state.prepareCostByPass.end()) return;
						found->second.first += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
						++found->second.second;
					}
				} costScope{state, any.name, lowerStarted};
				if (passIndex < masterIndices.size()) state.prepareCostByPass.try_emplace(any.name);
				std::visit([&](auto& value) {
					using T = std::decay_t<decltype(value)>;
					if constexpr (!std::is_same_v<T, std::monostate>) {
						PreparedPass packet;
						LoweredPass lowered;
						auto* immediateCommands = immediate && !value.pass->UsesTypedPreparation()
							? dynamic_cast<IHasImmediateModeCommands*>(value.pass.get()) : nullptr;
						if (immediateCommands) {
							ImmediateExecutionContext context{device, {org::imm::ImmediatePassKind::Render, m_immediateDispatch,
								&PersistentResolveByIdThunk, &PersistentResolveByPtrThunk, this}, frameIndex, hostData};
							{
								BT_ZONE_SCOPE("ORG.Persistent.RecordImmediateCommands");
								immediateCommands->RecordImmediateCommands(context);
							}
							const bool hasWork = context.list.HasRecordedWork();
							auto frame = context.list.Finalize();
							auto effect = immediateCommands->TakeOwnedImmediateSubmissionEffect();
							if (!hasWork && !effect) return;
							for (const auto& requirement : frame.requirements) {
								auto* resource = requirement.resourceHandleAndRange.resource.IsEphemeral()
									? requirement.resourceHandleAndRange.resource.GetEphemeralPtr() : _registry.Resolve(requirement.resourceHandleAndRange.resource);
								if (!resource) throw std::runtime_error("Immediate pass '" + any.name + "' references an unresolvable resource");
								const auto shape = ShapeOf(*resource);
								const auto range = LowerRange(requirement.resourceHandleAndRange.range, shape);
								if (range) lowered.entries.push_back({resource->GetSchedulingResourceID(), resource, *range, LowerState(requirement.state)});
							}
							lowered.declaration.compatibleQueueSlots = {0};
							lowered.declaration.preferredQueueSlot = 0;
							lowered.declaration.backend = static_cast<uint32_t>(primaryBackend);
							lowered.declaration.forceBatchIsolation = true;
							struct ReplayData {
								std::vector<std::byte> bytecode;
								org::imm::ImmediateDispatch dispatch;
								std::shared_ptr<org::imm::KeepAliveBag> keepAlive;
								std::function<void()> commit;
							};
							ReplayData replay{std::move(frame.bytecode), m_immediateDispatch,
								std::shared_ptr<org::imm::KeepAliveBag>(std::move(frame.keepAlive)), effect ? std::move(effect->commit) : nullptr};
							packet = PreparedPass::MakeOwned(std::move(replay),
								+[](const ReplayData& data, RecordingContext& recording) {
									if (!data.bytecode.empty()) org::imm::Replay(data.bytecode, recording.Commands(), data.dispatch);
								}, +[](const ReplayData& data) { if (data.commit) data.commit(); },
								effect ? std::move(effect->completionSignals) : std::vector<ExternalTimelinePoint>{});
						} else {
							if (auto* dynamic = dynamic_cast<IDynamicDeclaredResources*>(value.pass.get()); dynamic && dynamic->DeclaredResourcesChanged()) {
								BT_ZONE_SCOPE("ORG.Persistent.DynamicSegment.RefreshDeclarations");
								RedeclarePass(value, any.type);
							}
							{ BT_ZONE_SCOPE("ORG.Persistent.DynamicSegment.LowerPass"); lowered = lowering::LowerPass(*this, _registry, m_queueRegistry, primaryBackend, value, any.type); }
							// Groups are meaningless for one-frame programs: declare members directly.
							for (auto& group : lowered.groups)
								for (const auto& [id, resource] : group.members)
									for (const auto& [range, stateValue] : group.templates)
										lowered.entries.push_back({id, resource.get(), range, stateValue});
							lowered.groups.clear();
							lowered.declaration.forceBatchIsolation = true;
						}
						std::vector<Resource*> resources;
						for (const auto& use : lowered.entries) resources.push_back(use.resource);
						MaterializePersistentStandalone(resources);
						for (const auto& use : lowered.entries) slotFor(use.resource);
						for (const auto& use : lowered.exits) slotFor(use.resource);
						for (const auto& wait : lowered.explicitWaits) waits.emplace_back(static_cast<uint32_t>(loweredPasses.size()), wait);
						loweredPasses.push_back(std::move(lowered));
						loweredOwners.push_back(&any);
						packets.push_back(std::move(packet));
						names.push_back(any.name);
					}
				}, any.pass);
			}
			if (loweredPasses.empty()) return std::nullopt;

			// Packets, names and waits over a selected publication; shared by the
			// persistent segment program and the fresh (fallback) build.
			auto finish = [&](std::shared_ptr<const persistent::SelectedPublication> ready,
				std::shared_ptr<const FrozenExecutionBindings> legacyBindings, std::span<const persistent::PassId> passIds,
				std::vector<persistent::FrameRebinding> segmentRebindings,
				auto&& slotsForPass) -> PersistentExecutionState::PreparedSegment {
				PersistentExecutionState::PreparedSegment segment;
				segment.label = label;
				segment.publication = ready;
				segment.legacyBindings = std::move(legacyBindings);
				segment.rebindings = std::move(segmentRebindings);
				segment.invocations.resize(ready->logical->passSlots.size());
				auto segmentNames = std::make_shared<std::vector<std::string>>(ready->logical->passSlots.size());
				auto segmentPreparation = preparation;
				segmentPreparation.bindings = segment.legacyBindings;
				{
				BT_ZONE_SCOPE("ORG.Persistent.DynamicSegment.Prepare");
				for (size_t i = 0; i < passIds.size(); ++i) {
					auto& any = *loweredOwners[i];
					if (!packets[i]) {
						auto passPreparation = segmentPreparation;
						passPreparation.resourceSlots = slotsForPass(i);
						try {
							packets[i] = std::visit([&](auto& value) -> PreparedPass {
								using T = std::decay_t<decltype(value)>;
								if constexpr (std::is_same_v<T, std::monostate>) return {};
								else return value.pass->PrepareFrame(passPreparation);
							}, any.pass);
						} catch (const std::exception& e) {
							throw std::runtime_error("Pass '" + any.name + "' persistent preparation failed: " + e.what());
						}
						if (!packets[i]) throw std::runtime_error("Pass '" + any.name + "' has no owned preparation for the " + label + " segment");
					}
					packets[i].SetDebugName(any.name);
					segment.invocations[passIds[i].index] = std::move(packets[i]);
					(*segmentNames)[passIds[i].index] = names[i];
				}
				}
				segment.names = std::move(segmentNames);
				for (const auto& [passIndex, wait] : waits) {
					if (!wait.timeline || !wait.value) continue;
					const auto handle = wait.timeline.GetHandle();
					const uint64_t identity = m_runtime->TimelineIdentity(wait.timeline);
					bool bound = std::ranges::any_of(segment.foreignTimelines, [&](const auto& binding) { return binding.identity == identity; });
					for (size_t slot = 0; slot < m_queueRegistry.SlotCount() && !bound; ++slot)
						bound = SameTimeline(m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot))), wait.timeline);
					if (!bound) segment.foreignTimelines.push_back({identity, handle});
					segment.waits.push_back({passIds[passIndex], {identity, wait.value}});
				}
				return segment;
			};
			// Registry-handle identities a legacy pass may resolve in addition to
			// the scheduling/global identities every bound slot exposes.
			auto appendRequirementIDs = [&](AnyPassAndResources& any, FramePreparationContext::ResourceSlots& slots,
				auto&& slotOfSchedulingID) {
				const auto view = GetPassView(any);
				for (const auto& requirement : view.reqs) {
					auto* resource = requirement.resourceHandleAndRange.resource.IsEphemeral()
						? requirement.resourceHandleAndRange.resource.GetEphemeralPtr() : _registry.Resolve(requirement.resourceHandleAndRange.resource);
					if (!resource) continue;
					const auto slot = slotOfSchedulingID(resource->GetSchedulingResourceID());
					if (slot != UINT32_MAX) slots.emplace_back(requirement.resourceHandleAndRange.resource.GetGlobalResourceID(), slot);
				}
			};

			// ---- Persistent segment program: bind this frame's resource set into
			// reserved-capacity groups; no relowering into a new program, no compile
			// unless the pass structure or a pool's capacity changes.
			auto persistentSegment = [&]() -> std::optional<PersistentExecutionState::PreparedSegment> {
				using Program = PersistentExecutionState::SegmentProgram;
				struct Want { uint32_t pass; uint64_t templateHash; uint64_t physical; Resource* resource; RotationKey key; };
				std::vector<Want> wants;
				std::vector<uint64_t> key;
				key.push_back(loweredPasses.size());
				std::vector<std::vector<std::pair<uint64_t, Program::Template>>> passTemplates(loweredPasses.size());
				for (size_t p = 0; p < loweredPasses.size(); ++p) {
					const auto& lowered = loweredPasses[p];
					// Postconditions and swapchain images stay on the fresh build.
					if (!lowered.exits.empty() || !lowered.groups.empty()) return std::nullopt;
					key.push_back(std::hash<std::string_view>{}(names[p]));
					key.push_back(FingerprintDeclaration(lowered.declaration));
					auto& templates = passTemplates[p];
					for (const auto& use : lowered.entries) {
						if (IsSwapchainResource(*use.resource)) return std::nullopt;
						const auto rotation = RotationKeyOfResource(*use.resource);
						if (!rotation.concreteID) return std::nullopt;
						const Program::Template request{ShapeOf(*use.resource), use.range, use.state};
						const auto hash = HashSegmentTemplate(request);
						if (std::none_of(templates.begin(), templates.end(), [&](const auto& e) { return e.first == hash; })) templates.emplace_back(hash, request);
						wants.push_back({static_cast<uint32_t>(p), hash, rotation.concreteID, use.resource, rotation});
					}
				}
				// One slot per physical backing: bound in two pools it would form a
				// multi-slot equivalence class, which is structural and changes with
				// every frame the pairing differs.
				std::sort(wants.begin(), wants.end(), [](const Want& a, const Want& b) {
					return std::tie(a.physical, a.pass, a.templateHash) < std::tie(b.physical, b.pass, b.templateHash); });
				for (size_t i = 1; i < wants.size(); ++i)
					if (wants[i].physical == wants[i - 1].physical
						&& (wants[i].pass != wants[i - 1].pass || wants[i].templateHash != wants[i - 1].templateHash)) {
						basic_telemetry::AddCounter("ORG.Persistent.SegmentPoolConflicts");
						return std::nullopt;
					}
				const auto allWants = wants; // every registry object, for the slot maps
				wants.erase(std::unique(wants.begin(), wants.end(), [](const Want& a, const Want& b) { return a.physical == b.physical; }), wants.end());
				// Select the program for this pass structure. Pools and capacities
				// grow on the same program (structural, rare); the structure key
				// deliberately excludes them so a new access template or a larger
				// target set never starts a new program.
				Program* program = nullptr;
				for (auto& candidate : programs) if (candidate->key == key) { program = candidate.get(); break; }
				auto poolIndexOf = [&](const Program& candidate, uint32_t pass, uint64_t templateHash) -> uint32_t {
					for (const auto index : candidate.passes[pass].pools)
						if (candidate.pools[index].templateHash == templateHash) return index;
					return UINT32_MAX;
				};
				std::optional<persistent::GraphEditTransaction> edit;
				if (!program) {
					BT_ZONE_SCOPE("ORG.Persistent.DynamicSegment.CreateProgram");
					basic_telemetry::AddCounter("ORG.Persistent.SegmentProgramBuilds");
					if (programs.size() >= 4) programs.erase(programs.begin());
					programs.push_back(std::make_unique<Program>());
					program = programs.back().get();
					program->key = key;
					edit.emplace(program->program.BeginEdit());
					edit->SetQueues(RegistryQueues(m_queueRegistry, m_runtime->Backend()));
					for (size_t p = 0; p < loweredPasses.size(); ++p) {
						Program::Pass pass;
						pass.id = edit->AddPass(loweredPasses[p].declaration);
						program->passes.push_back(std::move(pass));
					}
				}
				auto ensureEdit = [&]() -> persistent::GraphEditTransaction& {
					if (!edit) edit.emplace(program->program.BeginEdit());
					return *edit;
				};
				auto addPositions = [&](Program::Pool& pool, uint32_t count) {
					const auto more = ensureEdit().ReserveGroupMembers(pool.group, count);
					for (const auto slot : more) {
						pool.slots.push_back(slot);
						if (slot.index >= program->table.size()) program->table.resize(slot.index + 1);
					}
					for (uint32_t i = 0; i < count; ++i) pool.free.push_back(pool.capacity + count - 1 - i);
					pool.capacity += count;
					pool.physical.resize(pool.capacity, 0);
					pool.resources.resize(pool.capacity, nullptr);
					pool.bound.resize(pool.capacity, RotationKey{});
					pool.isBound.resize(pool.capacity, 0);
					pool.wanted.resize(pool.capacity, 0);
				};
				// Missing templates become new pools; overfull pools double.
				for (size_t p = 0; p < loweredPasses.size(); ++p)
					for (const auto& [hash, request] : passTemplates[p]) {
						if (poolIndexOf(*program, static_cast<uint32_t>(p), hash) != UINT32_MAX) continue;
						BT_ZONE_SCOPE("ORG.Persistent.DynamicSegment.AddPool");
						basic_telemetry::AddCounter("ORG.Persistent.SegmentPoolAdds");
						uint32_t count = 0;
						for (const auto& want : wants) if (want.pass == p && want.templateHash == hash) ++count;
						Program::Pool pool;
						pool.pass = static_cast<uint32_t>(p);
						pool.templateHash = hash;
						pool.request = request;
						pool.group = ensureEdit().AddGroup(request.shape, {});
						program->passes[p].pools.push_back(static_cast<uint32_t>(program->pools.size()));
						program->pools.push_back(std::move(pool));
						addPositions(program->pools.back(), (std::max<uint32_t>)(4, count * 2));
						ensureEdit().DeclareGroupAccess(program->passes[p].id, program->pools.back().group, request.state, static_cast<uint32_t>(p), request.range);
					}
				{
					std::vector<uint32_t> counts(program->pools.size(), 0);
					for (const auto& want : wants) ++counts[poolIndexOf(*program, want.pass, want.templateHash)];
					for (size_t i = 0; i < counts.size(); ++i) {
						auto& pool = program->pools[i];
						if (counts[i] <= pool.capacity) continue;
						basic_telemetry::AddCounter("ORG.Persistent.SegmentPoolGrowth");
						addPositions(pool, (std::max)(pool.capacity, counts[i] * 2 - pool.capacity));
					}
				}
				std::unordered_set<uint64_t> capturedPhysicals;
				std::set<std::pair<const Program::Pool*, uint32_t>> capturedPositions;
				auto bindPosition = [&](Program::Pool& pool, uint32_t i, const Want& want) {
					BT_ZONE_SCOPE("ORG.Persistent.DynamicSegment.Bind");
					capturedPhysicals.insert(want.physical);
					capturedPositions.emplace(&pool, i);
					const auto slot = pool.slots[i];
					auto snapshot = PublicationBindingBundle::Capture(*want.resource);
					if (!snapshot) throw std::runtime_error(std::string(label) + " segment resource has no backing: " + want.resource->GetName());
					program->table[slot.index] = {snapshot->resource, snapshot->allocationOwner, snapshot->views, snapshot->descriptorOwner, nullptr};
					auto binding = CaptureSlotBinding(*want.resource, slot.index, pool.request.shape, std::move(snapshot));
					if (!binding) throw std::runtime_error(std::string(label) + " segment resource could not be bound: " + want.resource->GetName());
					auto& transaction = ensureEdit();
					if (pool.isBound[i]) transaction.ReplaceBinding(slot, std::move(*binding));
					else transaction.BindReserved(slot, std::move(*binding));
					pool.isBound[i] = 1;
					pool.physical[i] = want.physical;
					pool.resources[i] = want.resource;
					pool.bound[i] = want.key;
					pool.positionByPhysical[want.physical] = i;
				};
				// Diff against the bound set: rotated backings rebind in place,
				// departed members vacate, arrivals refill vacated positions first.
				for (auto& pool : program->pools) std::fill(pool.wanted.begin(), pool.wanted.end(), uint8_t{0});
				std::vector<std::pair<const Want*, uint32_t>> arriving;
				size_t rebound = 0;
				for (const auto& want : wants) {
					const auto poolIndex = poolIndexOf(*program, want.pass, want.templateHash);
					auto& pool = program->pools[poolIndex];
					const auto found = pool.positionByPhysical.find(want.physical);
					if (found == pool.positionByPhysical.end()) { arriving.emplace_back(&want, poolIndex); continue; }
					pool.wanted[found->second] = 1;
					if (pool.bound[found->second] != want.key) { bindPosition(pool, found->second, want); ++rebound; }
				}
				for (auto& pool : program->pools) {
					pool.vacated.clear();
					for (uint32_t i = 0; i < pool.capacity; ++i) {
						if (!pool.physical[i] || pool.wanted[i]) continue;
						pool.positionByPhysical.erase(pool.physical[i]);
						pool.physical[i] = 0;
						pool.resources[i] = nullptr;
						pool.vacated.push_back(i);
					}
					std::sort(pool.vacated.begin(), pool.vacated.end(), std::greater<>());
					std::sort(pool.free.begin(), pool.free.end(), std::greater<>());
				}
				for (const auto& [want, poolIndex] : arriving) {
					auto& pool = program->pools[poolIndex];
					uint32_t i;
					if (!pool.vacated.empty()) { i = pool.vacated.back(); pool.vacated.pop_back(); }
					else if (!pool.free.empty()) { i = pool.free.back(); pool.free.pop_back(); }
					else throw std::logic_error("Dynamic segment pool capacity accounting mismatch");
					bindPosition(pool, i, *want);
				}
				size_t unbound = 0;
				for (auto& pool : program->pools) {
					for (const auto i : pool.vacated) {
						if (pool.isBound[i]) {
							ensureEdit().Unbind(pool.slots[i]);
							pool.isBound[i] = 0;
							pool.bound[i] = RotationKey{};
							program->table[pool.slots[i].index] = {};
							++unbound;
						}
						pool.free.push_back(i);
					}
					pool.vacated.clear();
				}
				// A backing bound in several pools must carry one admission state
				// (see BindEntry): recapture the positions this edit did not touch.
				if (!capturedPhysicals.empty())
					for (auto& pool : program->pools)
						for (uint32_t i = 0; i < pool.capacity; ++i) {
							if (!pool.isBound[i] || !pool.physical[i] || !capturedPhysicals.contains(pool.physical[i])
								|| capturedPositions.contains({&pool, i})) continue;
							basic_telemetry::AddCounter("ORG.Persistent.SharedBackingRecaptures");
							bindPosition(pool, i, Want{pool.pass, pool.templateHash, pool.physical[i], pool.resources[i], pool.bound[i]});
						}
				if (edit) {
					BT_ZONE_SCOPE("ORG.Persistent.BuildDynamicSegment");
					BT_ZONE_VALUE(static_cast<int64_t>(arriving.size() + rebound + unbound));
					std::atomic_bool cancelled{false};
					auto ready = edit->Build(compiler.synchronousCompileWorkspace, cancelled);
					if (!ready || !program->program.Install(*edit, ready)) throw std::runtime_error(std::string(label) + " segment program failed to build");
					program->publication = std::move(ready);
					program->legacyBindings = FrozenExecutionBindings::Sparse(program->table);
					basic_telemetry::AddCounter("ORG.Persistent.SegmentBindingBuilds");
				} else basic_telemetry::AddCounter("ORG.Persistent.SegmentReuses");
				++program->frames;
				std::vector<persistent::PassId> passIds;
				for (const auto& pass : program->passes) passIds.push_back(pass.id);
				// Slot maps for legacy (non-immediate) passes: every identity of every
				// bound registry object, plus the pass's registry-handle identities.
				std::shared_ptr<FramePreparationContext::ResourceSlots> base;
				auto slotOfPhysical = [&](uint64_t physical) -> uint32_t {
					for (const auto& pool : program->pools)
						if (const auto found = pool.positionByPhysical.find(physical); found != pool.positionByPhysical.end())
							return pool.slots[found->second].index;
					return UINT32_MAX;
				};
				auto slotsForPass = [&](size_t i) -> std::shared_ptr<const FramePreparationContext::ResourceSlots> {
					if (!base) {
						base = std::make_shared<FramePreparationContext::ResourceSlots>();
						for (const auto& want : allWants) {
							const auto slot = slotOfPhysical(want.physical);
							if (slot == UINT32_MAX) continue;
							base->emplace_back(want.resource->GetSchedulingResourceID(), slot);
							base->emplace_back(want.resource->GetGlobalResourceID(), slot);
							base->emplace_back(want.physical, slot);
						}
					}
					auto slots = std::make_shared<FramePreparationContext::ResourceSlots>(*base);
					appendRequirementIDs(*loweredOwners[i], *slots, [&](uint64_t schedulingID) -> uint32_t {
						for (const auto& want : allWants)
							if (want.resource->GetSchedulingResourceID() == schedulingID) return slotOfPhysical(want.physical);
						return UINT32_MAX;
					});
					std::sort(slots->begin(), slots->end());
					slots->erase(std::unique(slots->begin(), slots->end()), slots->end());
					slots->sorted = true;
					return slots;
				};
				return finish(program->publication, program->legacyBindings, passIds, {}, slotsForPass);
			};
			if (auto segment = persistentSegment()) return segment;
			basic_telemetry::AddCounter("ORG.Persistent.SegmentFreshBuilds");

			// ---- Fresh build (fallback): a throwaway program for this frame.
			// Reuse key: lowered structure of every pass plus the exact backing bound
			// to every entry. Swapchain entries are rebound per frame and never cached.
			std::vector<uint64_t> key;
			key.reserve(loweredPasses.size() + entries.size() * 5 + 1);
			key.push_back(loweredPasses.size());
			for (const auto& lowered : loweredPasses) key.push_back(FingerprintLowering(lowered));
			bool cacheable = true;
			for (const auto& entry : entries) {
				if (IsSwapchainResource(*entry.resource)) { cacheable = false; break; }
				const auto rotation = RotationKeyOfResource(*entry.resource);
				key.push_back(entry.resource->GetSchedulingResourceID());
				key.push_back(rotation.concreteID); key.push_back(rotation.backingGeneration);
				key.push_back(reinterpret_cast<uintptr_t>(rotation.views));
				key.push_back((uint64_t{rotation.apiGeneration} << 32) | rotation.apiIndex);
			}
			std::shared_ptr<const persistent::SelectedPublication> ready;
			std::shared_ptr<const FrozenExecutionBindings> legacyBindings;
			if (cacheable && cache.publication && cache.key == key) {
				basic_telemetry::AddCounter("ORG.Persistent.DynamicSegmentReuses");
				ready = cache.publication;
				legacyBindings = cache.legacyBindings;
				ids = cache.ids;
			} else {
			persistent::GraphProgram program;
			auto edit = program.BeginEdit();
			edit.SetQueues(RegistryQueues(m_queueRegistry, m_runtime->Backend()));
			for (auto& entry : entries) {
				const auto slot = edit.ReserveResource(entry.shape);
				if (slot.index != entry.slot.index) throw std::logic_error("Dynamic segment slot allocation is not dense");
				entry.slot = slot;
			}
			for (auto& lowered : loweredPasses) {
				const auto id = edit.AddPass(std::move(lowered.declaration));
				std::unordered_map<uint32_t, persistent::BindingToken> tokens;
				for (const auto& use : lowered.entries) tokens[slotFor(use.resource)] = edit.Declare(id, entries[slotFor(use.resource)].slot, use.range, use.state);
				for (const auto& use : lowered.exits) {
					const auto slot = slotFor(use.resource);
					auto found = tokens.find(slot);
					if (found == tokens.end()) found = tokens.emplace(slot, edit.DeclareDependency(id, entries[slot].slot, use.state.write)).first;
					edit.DeclarePostcondition(found->second, use.range, use.state);
				}
				ids.push_back(id);
			}
			std::vector<FrozenExecutionBindings::ResourceBinding> table(entries.size());
			{
			BT_ZONE_SCOPE("ORG.Persistent.DynamicSegment.Bind");
			for (uint32_t index = 0; index < entries.size(); ++index) {
				auto& entry = entries[index];
				const bool swapchain = IsSwapchainResource(*entry.resource);
				auto snapshot = PublicationBindingBundle::Capture(*entry.resource);
				if (!snapshot) throw std::runtime_error(std::string(label) + " segment resource has no backing: " + entry.resource->GetName());
				table[index] = {snapshot->resource, snapshot->allocationOwner, snapshot->views, snapshot->descriptorOwner, nullptr};
				if (swapchain) {
					auto* concrete = UnwrapDynamic(entry.resource);
					persistent::FrameRebinding rebinding;
					rebinding.slot = entry.slot;
					rebinding.backing.graphResourceID = uint64_t{entry.slot.index} + 1;
					if (trace::Any()) rebinding.backing.debugName = entry.resource->GetName();
					rebinding.backing.resource = snapshot->resource.GetHandle();
					rebinding.backing.shape = entry.shape;
					rebinding.backing.regions = CaptureRegions(*concrete, entry.shape);
					rhi::ResourceDesc description{};
					if (concrete->TryGetRHIResourceDesc(description)) rebinding.backing.heapType = description.heapType;
					rebinding.recording = std::move(snapshot);
					rebindings.push_back(std::move(rebinding));
					continue;
				}
				auto binding = CaptureSlotBinding(*entry.resource, entry.slot.index, entry.shape, std::move(snapshot));
				if (!binding) throw std::runtime_error(std::string(label) + " segment resource could not be bound: " + entry.resource->GetName());
				edit.BindReserved(entry.slot, std::move(*binding));
			}
			}
			std::atomic_bool cancelled{false};
			ready = [&] {
				BT_ZONE_SCOPE("ORG.Persistent.BuildDynamicSegment");
				return edit.Build(compiler.synchronousCompileWorkspace, cancelled);
			}();
			if (!ready || !program.Install(edit, ready)) throw std::runtime_error(std::string(label) + " segment failed to build");
			legacyBindings = FrozenExecutionBindings::Sparse(std::move(table));
			if (cacheable) cache = {std::move(key), ready, legacyBindings, ids};
			else cache = {};
			}
			auto slotsForPass = [&](size_t i) -> std::shared_ptr<const FramePreparationContext::ResourceSlots> {
				auto slots = std::make_shared<FramePreparationContext::ResourceSlots>();
				for (const auto& entry : entries) {
					slots->emplace_back(entry.resource->GetSchedulingResourceID(), entry.slot.index);
					slots->emplace_back(entry.resource->GetGlobalResourceID(), entry.slot.index);
					if (auto* concrete = UnwrapDynamic(entry.resource)) slots->emplace_back(concrete->GetGlobalResourceID(), entry.slot.index);
				}
				appendRequirementIDs(*loweredOwners[i], *slots, [&](uint64_t schedulingID) -> uint32_t {
					const auto found = entriesByID.find(schedulingID);
					return found == entriesByID.end() ? UINT32_MAX : found->second;
				});
				std::sort(slots->begin(), slots->end());
				slots->erase(std::unique(slots->begin(), slots->end()), slots->end());
				slots->sorted = true;
				return slots;
			};
			return finish(ready, legacyBindings, ids, std::move(rebindings), slotsForPass);
		};
		std::vector<AnyPassAndResources> none;
		state.pre = buildSegment("Pre", state.preMasterIndices, none, true, state.preCache, state.prePrograms);
		// Per-frame extension passes (gathered in step 0) are materialized fresh,
		// as the legacy path does.
		state.frameExtensionPasses.clear();
		for (auto& d : frameExt) {
			if (d.type == PassType::Unknown || !d.pass || d.name.empty()) continue;
			state.frameExtensionPasses.push_back(MaterializeExternalPass(d, true, false));
		}
		state.tail = buildSegment("Tail", state.tailMasterIndices, state.frameExtensionPasses, false, state.tailCache, state.tailPrograms);
	}
	if (state.frameNumber % 900 == 0 && (!state.prepareCostByPass.empty() || !state.prepareCostByMain.empty())) {
		std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> rows(state.prepareCostByPass.begin(), state.prepareCostByPass.end());
		for (uint32_t i = 0; i < state.prepareCostByMain.size() && i < state.mainPasses.size(); ++i)
			if (state.prepareCostByMain[i].second) rows.emplace_back(state.mainPasses[i].name, state.prepareCostByMain[i]);
		std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
		std::string text;
		for (size_t i = 0; i < rows.size() && i < 48; ++i)
			text += rows[i].first + "=" + std::to_string(rows[i].second.first / (std::max<uint64_t>)(1, rows[i].second.second) / 1000) + "us x" + std::to_string(rows[i].second.second) + "; ";
		spdlog::info("Persistent preparation cost top passes (mean per call): {}", text);
	}
	BT_PLOT("ORG.Persistent.StructuralBuildsTotal", static_cast<int64_t>(state.structuralBuilds));
	BT_PLOT("ORG.Persistent.BindingBuildsTotal", static_cast<int64_t>(state.bindingBuilds));
}

} // namespace org
