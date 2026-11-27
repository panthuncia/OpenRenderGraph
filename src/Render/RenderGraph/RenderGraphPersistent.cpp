// Persistent execution, frame preparation: installing passes into the program, polling declarations, resolver groups and
// backings, and preparing the frame's invocations and dynamic segments. See RenderGraphPersistentState.h.
#include "RenderGraphPersistentState.h"
#include "Resources/Relocation.h"

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
// are private to their group (see ReserveGroupMember). A versioned resource has
// one slot whatever version a declaration names.
uint32_t DirectEntry(State& state, persistent::GraphEditTransaction& edit, uint64_t resourceID, Resource* resource,
	CompileResourceShape shape, std::shared_ptr<const IVersionedResource> versioned, std::shared_ptr<Resource> owner) {
	auto& indices = state.entriesByResourceID[resourceID];
	for (const auto index : indices) {
		auto& entry = state.entries[index];
		const bool same = versioned ? entry.versioned && entry.versioned->Key() == versioned->Key() : entry.resource == resource;
		if (entry.direct && same && entry.shape == shape) {
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
	if (versioned) state.declaredVersionsChanged = true;
	entry.versioned = std::move(versioned);
	entry.slot = slot;
	indices.push_back(slot.index);
	return slot.index;
}

// A direct slot is retiring: its resource's pin can go once nothing executing binds it (State::retiredPins).
void NoteRetiredDirectResource(State& state, const State::SlotEntry& entry) {
	if (!entry.resource || entry.versioned) return;
	auto* concrete = UnwrapDynamic(entry.resource);
	state.retiredPins.push_back({entry.resource, entry.resource->GetGlobalResourceID(),
		concrete ? concrete->GetGlobalResourceID() : 0u});
}

// While a structural build is pending, edits against the live publication are
// logged for replay onto the result; slots the live publication does not have
// yet are logged only.
bool LiveEdit(const State& state, const persistent::GraphEditTransaction& edit) {
	return state.pending && &edit != state.pending->edit.get();
}
void UnbindSlot(State& state, persistent::GraphEditTransaction& edit, const State::SlotEntry& entry) {
	state.deferredSharing.erase(entry.slot.index); // superseded
	if (LiveEdit(state, edit)) {
		state.pending->bindingLog[entry.slot.index] = std::nullopt;
		if (entry.slot.index >= state.pending->baseSlotCount) return;
	}
	edit.Unbind(entry.slot);
}
bool BindEntryOnly(State& state, persistent::GraphEditTransaction& edit, uint32_t index, uint64_t* identity = nullptr) {
	auto& entry = state.entries[index];
	if (entry.versioned || !entry.resource) return false;
	if (entry.multiDevice && state.ensureRepresentations) state.ensureRepresentations(*entry.resource);
	state.deferredSharing.erase(entry.slot.index); // superseded
	auto binding = CaptureSlotBinding(*entry.resource, entry.slot.index, entry.shape, nullptr,
		state.aliasContinues.contains(UnwrapDynamic(entry.resource)));
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
		if (other.retired || !other.isBound || other.versioned || !other.resource || other.slot.index != sibling) continue;
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

// The pass's slot for the versioned resource `key` (IVersionedResource::Key), or UINT32_MAX.
uint32_t VersionedEntry(const State& state, const State::MainPass& main, const void* key) {
	for (const auto index : main.directEntries) {
		const auto& entry = state.entries[index];
		if (entry.versioned && entry.versioned->Key() == key) return index;
	}
	return UINT32_MAX;
}

// A versioned slot's binding of `version`. It has two parts:
// - the frame rebinding that admission takes, so barriers and state tracking follow the version's backing,
// - the binding that the overlay gives preparation and recording.
// The binding owns the version. So the version and its backing live while a preparation, a recording or an execution
// holds the bindings. The slot keeps the binding for the next preparation that selects the same version, unless the
// resource is per frame.
// aliasContinues: the version continues from an earlier execution of the frame (State::aliasContinues).
State::SlotEntry::FrameBinding BindVersion(State::SlotEntry& entry, std::shared_ptr<const ResourceVersion> version, bool aliasContinues) {
	auto& resource = *version->resource;
	const auto key = RotationKeyOfResource(resource);
	if (entry.frameBinding && entry.frameBinding->version == version && entry.frameBinding->key == key
		&& entry.frameBinding->rebinding.backing.aliasContinues == (aliasContinues && entry.frameBinding->rebinding.backing.aliasSize != 0))
		return *entry.frameBinding;
	auto* concrete = UnwrapDynamic(&resource);
	auto snapshot = concrete ? PublicationBindingBundle::Capture(resource) : nullptr;
	if (!snapshot || !snapshot->resource.GetHandle().valid())
		throw std::runtime_error("Persistent slot of '" + resource.GetName() + "' selected a version with no backing. A version "
			"must be materialized before a preparation binds it");
	if (const auto shape = ShapeOf(resource); shape != entry.shape)
		throw std::logic_error("Persistent slot of '" + resource.GetName() + "' selected a version of " + std::to_string(shape.slices)
			+ " slices and " + std::to_string(shape.mips) + " mips, but the graph was lowered for " + std::to_string(entry.shape.slices)
			+ " and " + std::to_string(entry.shape.mips) + ". A new slice count needs graphs built for it, attached when its version "
			"is adopted (VersionedResource, \"Array-size versions\")");
	State::SlotEntry::FrameBinding made;
	made.rebinding.slot = entry.slot;
	made.rebinding.backing = BackingStateOf(*concrete, *snapshot, entry.slot.index, entry.shape);
	// A version placed from a frame placement lies in a block it shares with other transients. Admission then treats it as an
	// aliased backing: the alias ledger orders the other occupants of its bytes, and each execution enters it from Undefined.
	// Without a frame placement no version is placed, and these stay empty.
	made.rebinding.backing.aliasHeap = snapshot->aliasHeap;
	made.rebinding.backing.aliasHeapIdentity = snapshot->aliasHeap.get();
	made.rebinding.backing.aliasPoolID = snapshot->aliasPoolID;
	made.rebinding.backing.aliasOffset = snapshot->aliasOffset;
	made.rebinding.backing.aliasSize = snapshot->aliasSize;
	made.rebinding.backing.aliasContinues = aliasContinues && snapshot->aliasSize != 0;
	made.rebinding.recording = snapshot;
	made.rebinding.perFrame = entry.versioned->PerFrame();
	auto owner = std::make_shared<const std::pair<std::shared_ptr<const ResourceVersion>, PublicationBindingBundle::Snapshot>>(version, snapshot);
	made.binding = {snapshot->resource, std::move(owner), snapshot->views, snapshot->descriptorOwner, nullptr};
	made.version = std::move(version);
	made.key = key;
	if (entry.versioned->PerFrame()) entry.frameBinding.reset();
	else entry.frameBinding = made;
	return made;
}

} // namespace

// Installs (or reinstalls) a main pass declaration into the transaction.
void persistent_impl::InstallMainPass(RenderGraph& graph, State& state, persistent::GraphEditTransaction& edit,
	State::MainPass& main, LoweredPass lowered, uint32_t mainIndex, bool replace, std::optional<uint32_t> authoredOrder,
	std::vector<uint32_t>* touchedGroups) {
	std::vector<uint32_t> previousDirect;
	// Before the declaration moves into the edit: the fingerprint covers it.
	main.fingerprint = FingerprintLowering(lowered);
	main.structuralFingerprint = FingerprintLowering(lowered, false);
	main.structuralParts = FingerprintParts(lowered);
	main.declarationSummary = DescribeDeclaration(lowered);
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
    main.preparedResourceUses = lowered.resourceUses;
    main.preparedBindings = lowered.declaredBindings;
    main.declaredViews.clear();
    main.declaredBindings.clear();
    main.resolverUseEntries.clear();
    main.groupViews.clear();
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
		const auto index = DirectEntry(state, edit, use.resourceID, use.resource, ShapeOf(*use.resource), use.versioned, use.owner);
		auto& slotEntry = state.entries[index];
		if (std::find(slotEntry.subscribers.begin(), slotEntry.subscribers.end(), mainIndex) == slotEntry.subscribers.end())
			slotEntry.subscribers.push_back(mainIndex);
		if (main.device != BackendInstanceId::Primary && !slotEntry.multiDevice) {
			slotEntry.multiDevice = true;
			auto* concrete = UnwrapDynamic(slotEntry.resource);
			state.multiDeviceResources.insert(concrete);
			state.aliasPlaced.erase(concrete); // Its placement, if any, is left unused: the representations replace it.
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
			// A resolver that expects more members says so (IResourceResolver::CapacityHint).
			group.capacity = loweredGroup.members.size() <= 1 ? 2u
				: (std::max<uint32_t>)(16, static_cast<uint32_t>(loweredGroup.members.size() * 2));
			if (group.resolver) group.capacity = (std::max)(group.capacity, group.resolver->CapacityHint());
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
            if (use.resolverIdentity && index == UINT32_MAX) index = VersionedEntry(state, main, use.resolverIdentity.get());
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
		NoteRetiredDirectResource(state, entry);
		if (entry.versioned) state.declaredVersionsChanged = true;
		entry = {};
		entry.retired = true;
		basic_telemetry::AddCounter("ORG.Persistent.RetiredDirectSlots");
	}
	(void)graph;
}

// ---------------------------------------------------------------------------
// Bootstrap

static void LogRelowerDiff(const std::string& name, const State::MainPass& main, const LoweredPass& lowered,
	persistent_impl::RelowerCause cause, const char* notRemapped = "structure changed") {
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
	// The uses whose state or views differ from the installed layout's, by position: what changed when nothing was
	// added or removed.
	std::string changedUses;
	if (main.resourceUses && lowered.resourceUses && main.resourceUses != lowered.resourceUses) {
		const auto& before = main.resourceUses->uses;
		const auto& after = lowered.resourceUses->uses;
		if (before.size() != after.size()) {
			// The global resource IDs declared a different number of times, with how the count changed.
			const auto resolverUses = [](const std::vector<ResourceUseDeclaration>& uses) {
				return std::count_if(uses.begin(), uses.end(), [](const auto& use) { return use.resolverIdentity != nullptr; });
			};
			changedUses = "uses " + std::to_string(before.size()) + "->" + std::to_string(after.size())
				+ " (resolver " + std::to_string(resolverUses(before)) + "->" + std::to_string(resolverUses(after)) + "): ";
			std::unordered_map<uint64_t, int> counts;
			for (const auto& use : after) ++counts[use.binding.globalResourceID];
			for (const auto& use : before) --counts[use.binding.globalResourceID];
			for (const auto& [id, delta] : counts)
				if (delta) changedUses += std::to_string(id) + (delta > 0 ? "+" : "") + std::to_string(delta) + " ";
		}
		else for (size_t index = 0, shown = 0; index < after.size() && shown < 6; ++index) {
			const auto& a = before[index];
			const auto& b = after[index];
			bool sameViews = a.requiredViews.size() == b.requiredViews.size();
			for (size_t view = 0; sameViews && view < b.requiredViews.size(); ++view) sameViews = SameView(a.requiredViews[view], b.requiredViews[view]);
			const char* what = a.binding.globalResourceID != b.binding.globalResourceID ? "resource"
				: !(a.state == b.state) ? "state" : !sameViews ? "views" : nullptr;
			if (!what) continue;
			changedUses += "#" + std::to_string(index) + ":" + what + "(" + std::to_string(b.binding.globalResourceID) + ") ";
			++shown;
		}
	}
	spdlog::info("Persistent structural relower: pass='{}' cause={} notRemapped='{}' direct={} groups={} newGroupKeys={} added=[{}] removed=[{}] changedUses=[{}]",
		name, persistent_impl::RelowerCauseName(cause), notRemapped, lowered.entries.size(), lowered.groups.size(), keyChanges,
		added, removed, changedUses);
}

const char* persistent_impl::RelowerCauseName(RelowerCause cause) noexcept {
	switch (cause) {
	case RelowerCause::NoOp: return "NoOp";
	case RelowerCause::Rebind: return "Rebind";
	case RelowerCause::Shape: return "Shape";
	case RelowerCause::DirectResolver: return "DirectResolver";
	case RelowerCause::Remapped: return "Remapped";
	}
	return "?";
}

persistent_impl::RelowerCause persistent_impl::ClassifyRelower(const State::MainPass& main, const LoweredPass& lowered,
	bool directStale) {
	if (directStale) return RelowerCause::DirectResolver;
	// Groups the pass didn't declare before, or no longer declares, change its shape.
	if (lowered.groups.size() != main.loweredGroupKeys.size()) return RelowerCause::Shape;
	for (const auto& group : lowered.groups)
		if (std::find(main.loweredGroupKeys.begin(), main.loweredGroupKeys.end(), group.key.get()) == main.loweredGroupKeys.end())
			return RelowerCause::Shape;
	if (!main.resourceUses || !lowered.resourceUses) return main.resourceUses == lowered.resourceUses ? RelowerCause::NoOp : RelowerCause::Shape;
	if (main.resourceUses == lowered.resourceUses || SameLayoutContent(*main.resourceUses, *lowered.resourceUses, true))
		return RelowerCause::NoOp;
	return SameLayoutContent(*main.resourceUses, *lowered.resourceUses, false) ? RelowerCause::Rebind : RelowerCause::Shape;
}

void persistent_impl::CountRelower(RelowerCause cause, bool inlineBuild) {
	switch (cause) {
	case RelowerCause::NoOp: basic_telemetry::AddCounter("ORG.Persistent.Relower.NoOp"); break;
	case RelowerCause::Rebind: basic_telemetry::AddCounter("ORG.Persistent.Relower.Rebind"); break;
	case RelowerCause::Shape: basic_telemetry::AddCounter("ORG.Persistent.Relower.Shape"); break;
	case RelowerCause::DirectResolver: basic_telemetry::AddCounter("ORG.Persistent.Relower.DirectResolver"); break;
	case RelowerCause::Remapped: basic_telemetry::AddCounter("ORG.Persistent.Relower.Remapped"); break;
	}
	if (inlineBuild) basic_telemetry::AddCounter("ORG.Persistent.Relower.Inline");
}

void persistent_impl::DeferSharingChanges(State& state, persistent::GraphEditTransaction& edit) {
	auto taken = edit.SplitSharingChanges();
	if (!taken) {
		// A reserved slot bound onto a shared backing, say: no binding to keep, so it builds inline.
		basic_telemetry::AddCounter("ORG.Persistent.InlineSharingChanges");
		return;
	}
	if (taken->empty()) return;
	const auto live = state.program.Select();
	for (auto& [slot, version] : *taken) {
		// A pending build replays what the live publication binds, which is the old binding again.
		if (state.pending) {
			if (const auto* current = live->bindings.TryAt(slot.index)) state.pending->bindingLog[slot.index] = *current;
		}
		state.deferredSharing.insert_or_assign(slot.index, std::move(version));
	}
	basic_telemetry::AddCounter("ORG.Persistent.DeferredSharingChanges", static_cast<int64_t>(taken->size()));
}

void persistent_impl::ApplyDeferredSharing(State& state, persistent::GraphEditTransaction& edit) {
	for (auto& [index, version] : state.deferredSharing) {
		// The slot may have been retired or unbound by a build since; then there is nothing to swap.
		const auto* current = edit.BindingAt(index);
		if (!current || !current->bound || current->slotGeneration != version.slotGeneration) continue;
		edit.ReplaceBinding({index, version.slotGeneration}, std::move(version));
	}
	state.deferredSharing.clear();
}

const char* persistent_impl::RemapDeclaration(State& state, const persistent::SelectedPublication& live, State::MainPass& main,
	LoweredPass& lowered) {
	if (!main.resourceUses || !lowered.resourceUses || main.preparedResourceUses != main.resourceUses) return "no installed layout";
	if (FingerprintLowering(lowered, false) != main.structuralFingerprint) return "structure changed";
	const auto& before = *main.resourceUses;
	const auto& after = *lowered.resourceUses;
	// Group views are made per group position when the pass is installed, so they must not change.
	if (before.resolverViews.size() != after.resolverViews.size()) return "group views changed";
	for (size_t i = 0; i < after.resolverViews.size(); ++i) {
		const auto& x = before.resolverViews[i];
		const auto& y = after.resolverViews[i];
		if (x.identity != y.identity || x.views.size() != y.views.size()) return "group views changed";
		for (size_t v = 0; v < x.views.size(); ++v) if (!SameView(x.views[v], y.views[v])) return "group views changed";
	}
	std::vector<std::vector<persistent::ViewToken>> views;
	std::vector<persistent::BindingToken> bindings;
	std::vector<uint32_t> resolverEntries;
	for (const auto& use : after.uses) {
		views.emplace_back();
		bindings.emplace_back();
		// The use's slot, found as InstallMainPass finds it.
		uint32_t index = UINT32_MAX;
		if (use.resolverIdentity) for (const auto groupIndex : main.groups) {
			const auto& group = state.groups[groupIndex];
			if (group.key != use.resolverIdentity) continue;
			const auto position = MemberPosition(group, use.binding.globalResourceID);
			// A member the group has no slot for yet (past its capacity, say): only a build gives it one.
			if (!position) return "group member without a slot";
			index = group.memberEntries[*position];
			break;
		}
		if (use.resolverIdentity && index == UINT32_MAX) index = VersionedEntry(state, main, use.resolverIdentity.get());
		resolverEntries.push_back(index);
		if (use.requiredViews.empty()) continue;
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
		if (index == UINT32_MAX) return "use without a slot";
		// The executable must already have the pass's binding of that slot with these views.
		auto found = live.FindBinding(main.id, state.entries[index].slot, use.requiredViews);
		if (!found) return "view the executable lacks";
		bindings.back() = found->first;
		views.back() = std::move(found->second);
	}
	main.resourceUses = lowered.resourceUses;
	main.preparedResourceUses = lowered.resourceUses;
	main.preparedBindings = std::move(lowered.declaredBindings);
	main.declaredViews = std::move(views);
	main.declaredBindings = std::move(bindings);
	main.resolverUseEntries = std::move(resolverEntries);
	main.directResolvers = std::move(lowered.directResolvers);
	main.fingerprint = FingerprintLowering(lowered);
	main.slotsDirty = true;
	return nullptr;
}

void persistent_impl::UseRebindSlots(State::MainPass& main, LoweredPass& lowered,
	const IDynamicDeclaredResources* dynamic) {
	if (!dynamic || !dynamic->RebindsConcreteResources()) return;
	if (dynamic->PreparesOnlyDeclaredWork())
		throw std::logic_error("Pass '" + main.name + "' says RebindsConcreteResources, so it must say PreparesOnlyDeclaredWork false");
	if (lowered.device != BackendInstanceId::Primary) return;
	lowered.rebindable = true;
	std::vector<Resource*> places; // Distinct resources, in the order the lowering names them.
	const auto rebind = [&](LoweredUse& use) {
		if (!use.resource || use.owner || use.versioned) return;
		if (dynamic_cast<DynamicResource*>(use.resource)) return;
		auto concrete = use.resource->weak_from_this().lock();
		if (!concrete) return; // Not owned by a shared_ptr: nothing a slot could hold.
		const auto place = static_cast<size_t>(std::find(places.begin(), places.end(), use.resource) - places.begin());
		if (place == places.size()) places.push_back(use.resource);
		if (place == main.rebindSlots.size()) {
			main.rebindSlots.push_back(std::make_shared<DynamicResource>(concrete));
			main.rebindSlots.back()->SetName(main.name + " slot " + std::to_string(place));
		}
		auto& slot = main.rebindSlots[place];
		if (slot->GetResource() != concrete) slot->SetResource(concrete);
		use.resource = slot.get();
		use.resourceID = slot->GetSchedulingResourceID();
	};
	for (auto& use : lowered.entries) rebind(use);
	for (auto& use : lowered.exits) rebind(use);
}

// True when the new lowering declares a direct resource or group the pass
// did not declare before (the live publication has no slot for it).
bool persistent_impl::LoweringAddsDeclarations(const State::MainPass& main, const LoweredPass& lowered) {
    // A new view layout cannot prepare against the old publication while its
    // structural edit is pending, even when the resource set is unchanged.
    // An equal layout can: the executable's tokens are per (use, view).
    if (lowered.resourceUses && !lowered.resourceUses->uses.empty() && lowered.resourceUses != main.resourceUses
        && !(main.resourceUses && SameLayoutContent(*main.resourceUses, *lowered.resourceUses, !lowered.rebindable))) return true;
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

namespace {
// Where a slow preparation's time went (a graph's first ones, a re-placement): the phases that took 0.1 ms or more, and the passes
// whose PrepareFrame took longest. Logged for a graph's first two preparations and any over 3 ms.
struct PreparationLaps {
	using Clock = std::chrono::steady_clock;
	Clock::time_point begin = Clock::now(), last = begin;
	std::string phases;
	std::vector<std::pair<double, std::string>> passes;
	void Mark(const char* name) {
		const auto now = Clock::now();
		const double ms = std::chrono::duration<double, std::milli>(now - last).count();
		last = now;
		if (ms >= 0.1) phases += fmt::format("{}{} {:.2f}", phases.empty() ? "" : ", ", name, ms);
		if (UpdateProfileOn()) UpdateProfileAdd("prepare", name, ms * 1000.0);
	}
	void Pass(double ms, const std::string& name) {
		if (ms < 0.05) return;
		passes.emplace_back(ms, name);
		std::sort(passes.begin(), passes.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
		if (passes.size() > 4) passes.pop_back();
	}
};
thread_local PreparationLaps* t_preparationLaps = nullptr;
void PreparationLap(const char* name) {
	if (t_preparationLaps) t_preparationLaps->Mark(name);
}
}  // namespace

void RenderGraph::PreparePersistentFrame(rhi::Device device, uint32_t frameIndex, const IHostExecutionData* hostData, float deltaTime,
	const PersistentExecutionOptions& options) {
	BT_ZONE_SCOPE("ORG.Persistent.PrepareFrame");
	PreparationLaps laps;
	struct LapsScope {
		PreparationLaps* previous;
		explicit LapsScope(PreparationLaps* laps) : previous(t_preparationLaps) { t_preparationLaps = laps; }
		~LapsScope() { t_preparationLaps = previous; }
	} lapsScope(&laps);
	if (m_compilerState->frameProductionStopped) throw std::logic_error("Frame production has stopped");
	// Persistent structural builds, preparation and ownership retirement run on the task service; there is no inline fallback.
	if (!m_taskService) throw std::logic_error("Persistent execution requires a task service");
	auto& compiler = *m_compilerState;
	const bool bootstrap = !compiler.persistent;
	if (bootstrap) BootstrapPersistentGraph();
	auto& state = *compiler.persistent;
	state.frameIndex = frameIndex;
	state.frameNumber = ++compiler.asyncPreparationFrameNumber;
	RetireIdlePersistentAliasPools(state);
	org::runtime::ScopedActiveGraphServices activeServices(m_uploadService.get(), m_descriptorService.get());
	PreparationLap(bootstrap ? "bootstrap" : "begin");
	PlaceDeferredPersistentAliases();
	PreparationLap("deferred placement");
	BindPersistentFrame(bootstrap);
	PublishDeclaredVersions();
	PreparationLap("publish versions");
	state.prepared = PreparePersistentExecution(device, frameIndex, hostData, deltaTime, options);
	PreparationLap("prepare execution");
	const double total = std::chrono::duration<double, std::milli>(PreparationLaps::Clock::now() - laps.begin).count();
	if (state.frameNumber <= 2 || total > 3.0) {
		std::string passes;
		for (const auto& [ms, name] : laps.passes) passes += fmt::format("{}{} {:.2f}", passes.empty() ? "" : ", ", name, ms);
		spdlog::info("Persistent preparation timing: epoch {} preparation {} total {:.2f} ms: {}; slowest passes: {}", m_passEpochFilter,
			state.frameNumber, total, laps.phases, passes.empty() ? "none" : passes);
	}
}

void RenderGraph::ReleaseAliasPlacements() {
	if (m_compilerState && m_compilerState->persistent) m_compilerState->persistent->ReleaseAliasPlacements();
}

bool RenderGraph::DeclaredVersions::Declares(const void* key) const noexcept {
	const auto keys = Keys();
	return keys && std::binary_search(keys->begin(), keys->end(), key);
}

bool RenderGraph::DeclaredVersions::DeclaresOtherSlices(const void* key, uint32_t slices) const noexcept {
	const auto declared = m_slices.load(std::memory_order_acquire);
	if (!declared) return false;
	const auto first = std::lower_bound(declared->begin(), declared->end(), std::pair<const void*, uint32_t>{key, 0u});
	for (auto it = first; it != declared->end() && it->first == key; ++it)
		if (it->second != slices) return true;
	return false;
}

void RenderGraph::PublishDeclaredVersions() {
	auto& state = *m_compilerState->persistent;
	if (!state.declaredVersionsChanged) return;
	state.declaredVersionsChanged = false;
	auto keys = std::make_shared<std::vector<const void*>>();
	auto slices = std::make_shared<std::vector<std::pair<const void*, uint32_t>>>();
	for (const auto& entry : state.entries)
		if (entry.versioned && !entry.retired) {
			keys->push_back(entry.versioned->Key());
			if (entry.shape.hasLayout) slices->emplace_back(entry.versioned->Key(), entry.shape.slices);
		}
	std::ranges::sort(*keys);
	keys->erase(std::unique(keys->begin(), keys->end()), keys->end());
	std::ranges::sort(*slices);
	slices->erase(std::unique(slices->begin(), slices->end()), slices->end());
	m_declaredVersions->m_keys.store(std::move(keys), std::memory_order_release);
	m_declaredVersions->m_slices.store(std::move(slices), std::memory_order_release);
}

void RenderGraph::FreezeAliasedBindings() {
	if (m_compilerState && m_compilerState->persistent) m_compilerState->persistent->freezeAliased.store(true, std::memory_order_release);
}

void RenderGraph::PrepareFirstPersistentBuild() {
	BT_ZONE_SCOPE("ORG.Persistent.PrepareFirstBuild");
	if (m_compilerState->frameProductionStopped) throw std::logic_error("Frame production has stopped");
	if (!m_taskService) throw std::logic_error("Persistent execution requires a task service");
	if (m_compilerState->persistent) return;
	// The same bootstrap the first preparation runs. The first preparation then finds the graph built, and only polls the
	// passes' declarations: a pass whose declaration its first Update changes is lowered again then, like any later change.
	org::runtime::ScopedActiveGraphServices activeServices(m_uploadService.get(), m_descriptorService.get());
	m_deferFirstAliasPlacement = true;
	try {
		BootstrapPersistentGraph();
	} catch (...) {
		m_deferFirstAliasPlacement = false;
		throw;
	}
	m_deferFirstAliasPlacement = false;
	// A staged graph is asked what it declares before it is ever prepared.
	PublishDeclaredVersions();
}

void RenderGraph::PlaceDeferredAliasesNow() {
	if (!m_compilerState || m_compilerState->frameProductionStopped) return;
	if (!m_compilerState->persistent || !m_compilerState->persistent->aliasPlacementDeferred) return;
	// The services the first preparation would make current: the placement re-materializes resources and rebinds their views.
	org::runtime::ScopedActiveGraphServices activeServices(m_uploadService.get(), m_descriptorService.get());
	PlaceDeferredPersistentAliases();
}

bool RenderGraph::PlaceDeferredAliasesAtBuild(std::string* why) {
	if (!m_compilerState || m_compilerState->frameProductionStopped) {
		if (why) *why = "frame production stopped";
		return false;
	}
	if (!m_compilerState->persistent || !m_compilerState->persistent->aliasPlacementDeferred) return true;
	auto& state = *m_compilerState->persistent;
	// As the attach's (PlaceDeferredAliasesNow), on the building thread, which owns the graph alone.
	org::runtime::ScopedActiveGraphServices activeServices(m_uploadService.get(), m_descriptorService.get());
	state.placingAtBuild = true;
	state.atBuildOwnPools = false;
	state.atBuildRefusal.clear();
	struct Reset {
		bool& flag;
		~Reset() { flag = false; }
	} reset{state.placingAtBuild};
	PlaceDeferredPersistentAliases();
	if (state.atBuildRefusal.empty()) return true;
	state.aliasPlacementDeferred = true;
	if (why) *why = state.atBuildRefusal;
	return false;
}

bool RenderGraph::PublishPlacementSummaryAtBuild() {
	if (!m_compilerState || m_compilerState->frameProductionStopped || !m_compilerState->persistent) return false;
	auto& state = *m_compilerState->persistent;
	if (!state.aliasPlacementDeferred || state.framePlacementTaken) return false;
	org::runtime::ScopedActiveGraphServices activeServices(m_uploadService.get(), m_descriptorService.get());
	const auto live = state.program.Select();
	if (!live) return false;
	// The plan's gather only: nothing is placed, and the placement stays deferred (PlaceDeferredAliasesAtBuild places).
	auto edit = state.program.BeginEdit();
	state.summaryOnly = true;
	struct Reset {
		bool& flag;
		~Reset() { flag = false; }
	} reset{state.summaryOnly};
	(void)PlanPersistentAliasPlacement(edit, *live);
	return true;
}

bool RenderGraph::PlaceAgainAtBuild(std::string* why) {
	if (!m_compilerState || !m_compilerState->persistent) return PlaceDeferredAliasesAtBuild(why);
	auto& state = *m_compilerState->persistent;
	// Left for its attach: it tries again, by the same rules.
	if (state.aliasPlacementDeferred) return PlaceDeferredAliasesAtBuild(why);
	// Placed: only a version that leaves its placements where they are is taken. A new plan would treat what this graph adopted
	// from the running graphs as its own and could move it under them (a resize moved the running Visibility graph's 'CLod RT
	// CLAS build arena' that way). Otherwise it places again at its first preparation, as before.
	std::string reason;
	if (TakeFramePlacementInPlace(&reason, true)) return true;
	if (why) *why = (reason.empty() ? std::string("the version cannot be taken") : reason) + "; it places again at its first preparation";
	return false;
}

std::shared_ptr<const FramePlacementVersion> RenderGraph::PersistentFramePlacement() const {
	if (!m_compilerState || !m_compilerState->persistent || !m_compilerState->persistent->framePlacementTaken) return nullptr;
	return m_compilerState->persistent->framePlacement;
}

bool RenderGraph::TakeFramePlacementInPlace(std::string* why, bool atBuild) {
	if (!m_compilerState || !m_compilerState->persistent || !m_framePlacementSource) return false;
	auto& state = *m_compilerState->persistent;
	if (!state.framePlacementTaken || !state.framePlacement) return false;
	const auto next = m_framePlacementSource();
	if (!next || next == state.framePlacement || !next->HasEpoch(m_passEpochFilter)) return false;
	const auto refuse = [&](Resource* concrete, const char* reason) {
		if (why) *why = std::string(reason) + " ('" + concrete->GetName() + "')";
		return false;
	};
	size_t kept = 0;
	// At the build (a staged set's graph that has run nothing): a resource the newer version shows shared with another epoch now (or
	// no longer) changes only who gives its placement back, as a plan would: a shared one is the frame placement's, not this graph's.
	std::vector<Resource*> nowShared, nowOwn;
	for (const auto& [concrete, record] : state.aliasPlaced) {
		const auto* entry = next->FindObject(concrete->GetGlobalResourceID() + 1);
		if (!FramePlacementVersion::IsBlockPoolID(record.poolID)) {
			// In this graph's own pools. A version that places it should be taken by a plan, which moves it there.
			if (entry && entry->placed) return refuse(concrete, "a resource of its own pools is placed by the version");
			continue;
		}
		if (!entry || !entry->placed) return refuse(concrete, "the version does not place a resource it placed");
		const auto heap = next->PeekBlock(entry->block);
		if (!heap || heap.get() != record.heap || entry->offset != record.offset || entry->size != record.sizeBytes)
			return refuse(concrete, heap ? "the version moves a resource it placed" : "the version's block has no heap yet");
		// Its contents come from an earlier epoch (or across runs) exactly as before, so its bindings enter it the same way.
		const bool firstInFrame = !entry->uses.empty() && entry->uses.front().epoch == m_passEpochFilter;
		if ((!firstInFrame || entry->acrossRuns) != record.continues) return refuse(concrete, "its contents flow changed");
		// This graph's own placement exactly when no other epoch uses it, so its end gives back the same ones.
		if (entry->MultiEpoch() == state.aliasPlacementOwners.contains(concrete)) {
			if (!atBuild) return refuse(concrete, "its epochs changed");
			(entry->MultiEpoch() ? nowShared : nowOwn).push_back(concrete);
		}
		++kept;
	}
	for (auto* concrete : nowShared) state.aliasPlacementOwners.erase(concrete);
	for (auto* concrete : nowOwn)
		if (auto* backed = dynamic_cast<BackedResource*>(concrete))
			state.aliasPlacementOwners[concrete] = {concrete->weak_from_this(), backed->GetBackingGeneration()};
	const auto previous = state.framePlacement->Id();
	state.framePlacement = next;
	++state.placementSerial;
	m_framePlacementVersion.store(next->Id(), std::memory_order_release);
	spdlog::info("Persistent alias placement: epoch {} takes frame placement {} as it is (from {}; its {} frame placements stay): frame={}",
		m_passEpochFilter, next->Id(), previous, kept, state.frameNumber);
	return true;
}

void RenderGraph::PlaceDeferredPersistentAliases() {
	auto& compiler = *m_compilerState;
	if (!compiler.persistent || !compiler.persistent->aliasPlacementDeferred) return;
	BT_ZONE_SCOPE("ORG.Persistent.DeferredAliasPlacement");
	auto& state = *compiler.persistent;
	state.aliasPlacementDeferred = false;
	auto edit = state.program.BeginEdit();
	// A structural edit, as every placement is: the executable re-validates the new overlaps (and the bindings the planner
	// replaces may carry another graph's placement, which only a structural edit may change). There are no orderings yet.
	edit.ClearPlacementOrderings();
	state.aliasEdges.clear();
	const auto begin = std::chrono::steady_clock::now();
	const auto live = state.program.Select();
	if (!live || !PlanPersistentAliasPlacement(edit, *live)) return;
	const auto planned = std::chrono::steady_clock::now();
	std::atomic_bool cancelled{false};
	auto ready = edit.Build(compiler.synchronousCompileWorkspace, cancelled);
	const auto built = std::chrono::steady_clock::now();
	if (!ready || !state.program.Install(edit, ready)) throw std::runtime_error("Persistent deferred alias placement failed to install");
	++state.placementSerial;
	const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
	spdlog::info("Persistent deferred alias placement timing: epoch {} plan {:.2f} ms, validated build {:.2f} ms, install {:.2f} ms", m_passEpochFilter,
		ms(begin, planned), ms(planned, built), ms(built, std::chrono::steady_clock::now()));
	++state.structuralBuilds;
	basic_telemetry::AddCounter("ORG.Persistent.DeferredAliasPlacements");
}

// The first preparation's: the passes are lowered and the first publication is built inline.
void RenderGraph::BootstrapPersistentGraph() {
	BT_ZONE_SCOPE("ORG.Persistent.Bootstrap");
	auto& compiler = *m_compilerState;
	{
		compiler.persistent = std::make_shared<PersistentExecutionState>();
		auto& state = *compiler.persistent;
		state.aliasArena = m_persistentAliasArena ? m_persistentAliasArena : PersistentAliasArena::Create();
		const size_t framesInFlight = m_renderGraphSettingsService ? m_renderGraphSettingsService->GetNumFramesInFlight() : 3;
		state.admission = std::make_unique<persistent::SynchronousAdmission>(framesInFlight * 3 + 3,
			[this](std::shared_ptr<const void> owner) {
				if (!m_compilerState->ownershipRetirementScope)
					m_compilerState->ownershipRetirementScope = m_taskService->CreateScope("ORG.Ownership.Retirement");
				if (!m_taskService->Submit(m_compilerState->ownershipRetirementScope, runtime::TaskPriority::Background,
					"ORG.Ownership.RetirePublication", [owner = std::move(owner)]() mutable { owner.reset(); }))
					basic_telemetry::AddCounter("ORG.Ownership.RetirementRejected");
			}, m_runtime->Sequencer());
		state.admission->SetExternalHomeStates(m_persistentExternalHomeStates);
		state.ensureRepresentations = [this](Resource& resource) { EnsurePersistentRepresentations(resource); };
		state.deferAliasPlacement = m_deferFirstAliasPlacement;
		// Tickets record their uploads on the graphics queue, or on the queue that first uses them (RecordTakenUploads).
		if (m_queueRegistry.SlotCount())
			state.admission->SetUploadQueue(static_cast<uint32_t>(ToUnderlying(m_queueRegistry.FindGraphicsSlot())));
		// Passes with immediate-mode commands are uploads; every other pass is a main pass.
		state.mainPassByMaster.assign(m_masterPassList.size(), UINT32_MAX);
		state.masterHostedInMain.assign(m_masterPassList.size(), 0);
		std::vector<Resource*> usedResources;
		for (size_t masterIndex = 0; masterIndex < m_masterPassList.size(); ++masterIndex) {
			auto& any = m_masterPassList[masterIndex];
			bool upload = false;
			uint32_t epoch = persistent::AllEpochs;
			std::visit([&](auto& value) {
				using T = std::decay_t<decltype(value)>;
				if constexpr (!std::is_same_v<T, std::monostate>) {
					epoch = value.epoch;
					if (!value.pass->UsesTypedPreparation() && dynamic_cast<IHasImmediateModeCommands*>(value.pass.get()))
						upload = true;
				}
			}, any.pass);
			if (upload) state.uploadMasterIndices.push_back(masterIndex);
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
					// The passes were declared at Setup, and this frame's Update has run since. A pass whose declaration that
					// Update changed is declared again now, so the first build already has it.
					auto* dynamic = dynamic_cast<IDynamicDeclaredResources*>(value.pass.get());
					if (dynamic && dynamic->DeclaredResourcesChanged() && !dynamic->DeclarationsProvidedByImmediateCommands())
						RedeclarePass(value, any.type);
					auto lowered = lowering::LowerPass(*this, _registry, m_queueRegistry, primaryBackend, value, any.type);
					UseRebindSlots(main, lowered, dynamic);
					InstallMainPass(*this, state, edit, main, std::move(lowered), mainIndex, false);
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
		// The transients it binds, for the other epochs' plans: a set's graphs are all built before any of them places (a deferred
		// placement), so the first to place already knows what the others share with it.
		SyncEpochUses(state, m_passEpochFilter);
		size_t unbound = 0; std::string unboundNames;
		for (const auto& entry : state.entries) {
			if (!entry.resource || entry.versioned || entry.isBound) continue;
			if (unbound++ < 8) unboundNames += (unboundNames.empty() ? "" : ", ") + entry.resource->GetName();
		}
		// Alias candidates are bound by the deferred placement (PlaceDeferredPersistentAliases).
		if (unbound && !state.aliasPlacementDeferred) spdlog::warn("Persistent graph bootstrap left {} declared slots unbound (first: {})", unbound, unboundNames);
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
		spdlog::info("Persistent graph bootstrap: mainPasses={} uploads={} slots={} groups={} batches={} build_ms={:.2f}",
			state.mainPasses.size(), state.uploadMasterIndices.size(), state.entries.size(), state.groups.size(),
			ready->executable->graph->batches.size(),
			std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
	}
}

// A pass that declares a versioned resource captured the version current at its declaration. Its resolver snapshot owns
// that version's resource list, its requirements name the version, and its keep-alives and the graph's transient pin hold
// it. Preparations bind the version they select, so a new version needs no new declaration. Without this, the graph would
// hold the old version until its next build.
// So once a preparation binds a newer version, each such declaration moves to it here, as a redeclaration would. Its uses lower to
// the same slot, so the installed executable stays. The pass's use layout names resources by ID only, so it stays too.
// Frames in flight keep what they run: their bindings own the versions they bind (BindVersion).
void RenderGraph::ReleaseReplacedVersionDeclarations() {
	auto& state = *m_compilerState->persistent;
	// A cheap test first: has any versioned slot's current version changed since the last sweep?
	bool changed = std::exchange(state.versionSweepRetry, false);
	for (auto& entry : state.entries) {
		if (!entry.versioned || entry.retired || entry.versioned->PerFrame() || entry.versioned->LateBound()) continue;
		// The version this preparation binds: a revision's pending one while its recordings are made, else the current one.
		// A revision's recordings are replayed after its adoption without another preparation, so the move happens here.
		const auto current = entry.versioned->SelectVersion(m_resolverCaptureContext.get());
		if (!current || current->number == entry.sweptVersion) continue;
		entry.sweptVersion = current->number;
		changed = true;
	}
	if (!changed) return;
	BT_ZONE_SCOPE("ORG.Persistent.ReleaseReplacedVersionDeclarations");
	// The replaced versions' resource IDs. Resource IDs are never reused.
	std::unordered_set<uint64_t> replaced;
	size_t passes = 0;
	// Snapshots left on an older version, by why: not one resource, no declaration yet, another shape.
	size_t leftSet = 0, leftWaiting = 0, leftShape = 0;
	std::string movedNames;  // the first passes moved, so the log says which graph it is
	for (auto& any : m_masterPassList) {
		std::visit([&](auto& value) {
			using T = std::decay_t<decltype(value)>;
			if constexpr (!std::is_same_v<T, std::monostate>) {
				// Old resource ID to the current version's resource, for this pass.
				std::unordered_map<uint64_t, Resource*> moves;
				for (auto& snapshot : value.resolverSnapshots) {
					const auto* versioned = lowering::PreparedVersions(snapshot.resolver.get());
					if (!versioned || !snapshot.state || !snapshot.state->resources) continue;
					if (snapshot.state->resources->size() != 1) {
						++leftSet;
						continue;
					}
					const auto current = versioned->SelectVersion(m_resolverCaptureContext.get());
					if (!current || current->declaration == snapshot.state) continue;
					// Only an older version moves. A newer one, such as a revision's or a pending reshape's, stays.
					if (snapshot.state->resourceSetIdentity.high >= current->number) continue;
					// The current version has no declaration or resource yet. The pass moves at a later preparation.
					if (!current->declaration || !current->resource) {
						state.versionSweepRetry = true;
						++leftWaiting;
						continue;
					}
					const auto* declared = snapshot.state->resources->front().get();
					// A version of another slice count: the graph was lowered for the old shape, and graphs built for the new
					// one replace it.
					if (!declared || lowering::ShapeOf(*declared) != lowering::ShapeOf(*current->resource)) {
						++leftShape;
						continue;
					}
					const auto oldID = declared->GetGlobalResourceID();
					moves.emplace(oldID, current->resource.get());
					replaced.insert(oldID);
					// This may release the old version's resource list. Only IDs are used from here on.
					snapshot.state = current->declaration;
				}
				if (moves.empty()) return;
				++passes;
				if (passes <= 3) movedNames += (movedNames.empty() ? "" : ", ") + any.name;
				auto& resources = value.resources;
				const auto move = [&](ResourceHandleAndRange& use) {
					if (use.resource.IsEphemeral()) return;
					if (const auto found = moves.find(use.resource.GetGlobalResourceID()); found != moves.end())
						use.resource = RequestResourceHandle(found->second);
				};
				for (auto& requirement : resources.staticResourceRequirements) move(requirement.resourceHandleAndRange);
				for (auto& requirement : resources.frameResourceRequirements) move(requirement.resourceHandleAndRange);
				for (auto& transition : resources.internalTransitions) move(transition.first);
				resources.mergedFrameRequirementsDirty = true;
				value.retainedAnonymousKeepAlive = CaptureRetainedAnonymousKeepAlive(
					resources.staticResourceRequirements, resources.internalTransitions);
			}
		}, any.pass);
	}
	if (leftSet || leftWaiting || leftShape)
		spdlog::info("Persistent declarations left on replaced versions: {} of a resource set, {} waiting for the current declaration, {} of another shape",
			leftSet, leftWaiting, leftShape);
	if (replaced.empty()) return;
	// The graph's transient pin of a replaced version goes, unless a pass still names it, for example by pointer.
	std::unordered_set<uint64_t> named;
	for (const auto& any : m_masterPassList) {
		std::visit([&](const auto& value) {
			using T = std::decay_t<decltype(value)>;
			if constexpr (!std::is_same_v<T, std::monostate>) {
				const auto note = [&](const ResourceHandleAndRange& use) {
					const auto id = use.resource.GetGlobalResourceID();
					if (replaced.contains(id)) named.insert(id);
				};
				const auto& resources = value.resources;
				for (const auto& requirement : resources.staticResourceRequirements) note(requirement.resourceHandleAndRange);
				for (const auto& requirement : resources.frameResourceRequirements) note(requirement.resourceHandleAndRange);
				for (const auto& transition : resources.internalTransitions) note(transition.first);
			}
		}, any.pass);
	}
	size_t released = 0;
	for (const auto id : replaced) {
		if (named.contains(id) || resourcesByID.contains(id)) continue;
		const auto found = m_transientFrameResourcesByID.find(id);
		if (found == m_transientFrameResourcesByID.end()) continue;
		if (found->second) {
			if (const auto byName = m_transientFrameResourcesByName.find(found->second->GetName());
				byName != m_transientFrameResourcesByName.end() && byName->second == found->second)
				m_transientFrameResourcesByName.erase(byName);
		}
		m_transientFrameResourcesByID.erase(found);
		++released;
	}
	spdlog::info("Persistent declarations moved off {} replaced versions in {} passes ({}); {} transient pins released",
		replaced.size(), passes, movedNames, released);
}

// Steps 1-2 of a preparation: the passes' declarations, group memberships and backings, and the edits they make.
void RenderGraph::BindPersistentFrame(bool bootstrap) {
	auto& compiler = *m_compilerState;
	auto& state = *compiler.persistent;
	const auto primaryBackend = m_backendDevices.empty() ? rhi::Backend::Null : m_backendDevices.front().backend;

	// 1. Structural and membership changes reported by legacy passes. One
	// transaction carries both so the slot directory never diverges from the
	// installed program: a directory change is only made against an edit that
	// is built and installed below (or throws).
	{
		BT_ZONE_SCOPE("ORG.Persistent.PollDeclarations");
		std::vector<uint32_t> structural;
		std::vector<uint32_t> growGroups;
		std::unordered_map<uint32_t, LoweredPass> relowered;
		// Relowers with the same structure, tried against the installed executable once this frame's group members are
		// placed (RemapDeclaration); the ones that don't map become structural.
		struct RemapCandidate { uint32_t mainIndex; LoweredPass lowered; const IDynamicDeclaredResources* dynamic; };
		std::vector<RemapCandidate> remapCandidates;
		std::optional<persistent::GraphEditTransaction> edit;
		AdvancePersistentStructuralBuild(structural);
		// Declarations of replaced versions move to the current ones. A pending build lowered the old declarations, so
		// this waits until it installs.
		if (!bootstrap && !state.pending) ReleaseReplacedVersionDeclarations();
		PreparationLap("structural build and replaced versions");
		// Alias re-placement when the packing strategy changed since the latest plan (plans run only on structural builds).
		const bool packingChanged = m_getAutoAliasPackingStrategy && m_getAutoAliasPackingStrategy() != state.aliasPackingStrategy;
		// Alias re-placement when the host published a new frame placement since this graph's plan took one.
		bool framePlacementChanged = m_framePlacementProbe && state.framePlacementTaken
			&& m_framePlacementProbe() != (state.framePlacement ? state.framePlacement->Id() : 0);
		// A re-placement prepared on a worker for this version installs here: no plan, no build on this thread.
		if (framePlacementChanged && !state.pending) {
			PreparationLap("before the prepared re-placement");
			if (InstallPreparedReplacement()) framePlacementChanged = false;
			PreparationLap("prepared re-placement");
		}
		// A version that leaves this graph's placements where they are is taken as it is: no plan, no build.
		if (framePlacementChanged && !state.pending && TakeFramePlacementInPlace()) framePlacementChanged = false;
		// A resource several epochs share that this graph bound at another graph's placement, moved by that graph since (its
		// re-placement for a newer version): this graph's executable must admit the new backing, which only a structural edit does.
		bool adoptedMoved = false;
		for (const auto& [concrete, adopted] : state.adoptedBackings) {
			const auto resource = adopted.resource.lock();
			const auto* backed = resource ? dynamic_cast<const BackedResource*>(resource.get()) : nullptr;
			if (backed && backed->GetBackingGeneration() != adopted.generation) {
				adoptedMoved = true;
				break;
			}
		}
		// A graph of another epoch declared one of this graph's transients since its plan (a graph built later binds it too). If
		// this graph placed it in its own pools, it moves out: those pools see only this graph's uses. Or another graph took one of
		// this graph's placements (gave it a backing of its own): this graph binds that backing instead, which only a structural
		// edit admits.
		bool ownPoolShared = false;
		if (const auto taken = AliasPlacementTakenSerial(); taken != state.placementTakenSerialAtPlan) {
			bool ownTaken = false;
			for (const auto& [concrete, owner] : state.aliasPlacementOwners) {
				const auto resource = owner.resource.lock();
				const auto* backed = resource ? dynamic_cast<const BackedResource*>(resource.get()) : nullptr;
				if (backed && backed->GetBackingGeneration() != owner.generation) {
					ownTaken = true;
					break;
				}
			}
			if (ownTaken) adoptedMoved = true;
			else state.placementTakenSerialAtPlan = taken;
		}
		if (!state.pending && state.framePlacementTaken) {
			if (const auto serial = CrossEpochUsesSerial(); serial != state.crossEpochSerialAtPlan) {
				for (const auto& [concrete, record] : state.aliasPlaced)
					if (!FramePlacementVersion::IsBlockPoolID(record.poolID) && DeclaredByOtherEpoch(concrete, m_passEpochFilter)) {
						ownPoolShared = true;
						break;
					}
				if (!ownPoolShared) state.crossEpochSerialAtPlan = serial;
			}
		}
		if ((packingChanged || framePlacementChanged || adoptedMoved || ownPoolShared) && !state.pending) {
			BT_ZONE_SCOPE("ORG.Persistent.ForcedRealias");
			const auto forcedBegin = std::chrono::steady_clock::now();
			auto forced = state.program.BeginEdit();
			// Re-placement is only valid in a structural edit (the executable must
			// re-validate the new overlaps). Dropping the orderings makes it one;
			// the planner re-derives them for the new placements.
			forced.ClearPlacementOrderings();
			state.aliasEdges.clear();
			// A packing strategy change re-places every pool in new memory. A new frame placement does not: the candidates it
			// covers move to its heaps, and the rest stay in the graph's pools and frame arena ranges, which grow or shrink as for
			// any plan. New pool memory here made a new pool generation at each version, so every other graph placed in the pool
			// went stale and was built again, and the frame arena took new bytes beside the graph's current ranges.
			state.forceNewAliasHeaps = packingChanged;
			// The plan takes the host's current version again.
			if (framePlacementChanged) state.framePlacementTaken = false;
			if (packingChanged) {
				auto ready = BuildPersistentStructural(forced);
				state.forceNewAliasHeaps = false;
				if (!ready || !state.program.Install(forced, ready)) throw std::runtime_error("Persistent forced alias re-placement failed to install");
				++state.structuralBuilds;
				++state.placementSerial;
			} else {
				// A new frame placement or a moved shared backing changes placements only: planned against the live publication's
				// ranks (as the deferred placement), so one validated build instead of a schedule build and a placed one. The
				// orderings the plan derives follow those ranks, so the live order stays valid under them.
				state.forceNewAliasHeaps = false;
				const auto live = state.program.Select();
				if (live && PlanPersistentAliasPlacement(forced, *live)) {
					std::atomic_bool cancelled{false};
					auto ready = forced.Build(compiler.synchronousCompileWorkspace, cancelled);
					if (!ready || !state.program.Install(forced, ready)) throw std::runtime_error("Persistent forced alias re-placement failed to install");
					++state.structuralBuilds;
					++state.placementSerial;
				}
			}
			PreparationLap("re-placement");
			spdlog::info("Persistent alias re-placement ({}): frame={} in {:.2f} ms",
				packingChanged ? "packing strategy changed" : framePlacementChanged ? "new frame placement"
				: adoptedMoved ? "a shared resource another epoch placed moved"
				: ownPoolShared ? "another epoch's graph now uses a resource in its own pools" : "forced", state.frameNumber,
				std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - forcedBegin).count());
		} else if (!state.pending && state.framePlacementTaken && m_framePlacementEnabled && m_framePlacementEnabled()) {
			// A versioned candidate whose current version changed since the latest plan (a new extent, a new version). The new
			// version has a key of its own, so the graph plans again to publish it in its usage summary. It keeps the memory its
			// maker gave it until a frame placement covers it. The graph's own pools do not change, so their heaps stay. As for
			// the deferred placement, the plan runs against the live publication and builds only when it changed something.
			bool versionsMoved = false;
			for (const auto& entry : state.entries) {
				if (!entry.aliasPlannedVersion || entry.retired || !entry.versioned) continue;
				const auto current = entry.versioned->SelectVersion(nullptr);
				if (current && current->number != entry.aliasPlannedVersion) {
					versionsMoved = true;
					break;
				}
			}
			if (versionsMoved) {
				BT_ZONE_SCOPE("ORG.Persistent.VersionRealias");
				auto moved = state.program.BeginEdit();
				moved.ClearPlacementOrderings();
				state.aliasEdges.clear();
				const auto live = state.program.Select();
				if (live && PlanPersistentAliasPlacement(moved, *live)) {
					std::atomic_bool cancelled{false};
					auto ready = moved.Build(compiler.synchronousCompileWorkspace, cancelled);
					if (!ready || !state.program.Install(moved, ready))
						throw std::runtime_error("Persistent alias re-placement for new versions failed to install");
					++state.structuralBuilds;
					++state.placementSerial;
				}
				spdlog::info("Persistent alias re-placement (new versions of versioned candidates): frame={}", state.frameNumber);
			}
		}
		auto ensureEdit = [&]() -> persistent::GraphEditTransaction& {
			if (!edit) edit.emplace(state.program.BeginEdit());
			return *edit;
		};
		size_t refreshed = 0;
		// Set when a relower cannot wait for a worker build: the frame would prepare the pass against a publication that does
		// not have its declarations.
		bool inlineStructural = false;
		PreparationLap("versions moved");
		{ BT_ZONE_SCOPE("ORG.Persistent.PollDynamicDeclared");
		for (uint32_t mainIndex = 0; mainIndex < state.mainPasses.size(); ++mainIndex) {
			auto& main = state.mainPasses[mainIndex];
			auto& any = m_masterPassList[main.masterIndex];
			std::visit([&](auto& value) {
				using T = std::decay_t<decltype(value)>;
				if constexpr (!std::is_same_v<T, std::monostate>) {
					auto* dynamic = dynamic_cast<IDynamicDeclaredResources*>(value.pass.get());
					// On the bootstrap frame every pass was just declared from this frame's state.
					const bool declared = !bootstrap
						&& (dynamic && dynamic->DeclaredResourcesChanged() && !dynamic->DeclarationsProvidedByImmediateCommands());
					// A resolver declared directly is lowered as the set it resolved to; another set means the pass is lowered
					// again, from a fresh declaration, before it is prepared.
					bool directStale = false;
					for (const auto& [resolver, identity] : main.directResolvers) {
						const auto captured = CaptureResolverDeclarationState(*resolver);
						if (captured && !(captured->resourceSetIdentity == identity)) { directStale = true; break; }
					}
					if (!declared && !directStale) return;
					// A build in flight lowers the pass's current declaration, and frames keep preparing against the publication
					// that has it until that build installs. Redeclaring now would hand the pass declarations that publication
					// lacks; the pass still reports the change, so it is redeclared once the build installs.
					if (state.pending && !directStale) return;
					BT_ZONE_SCOPE("ORG.Persistent.RefreshDeclaration");
					BT_ZONE_TEXT(any.name.data(), any.name.size());
					++refreshed;
					RedeclarePass(value, any.type);
					auto lowered = lowering::LowerPass(*this, _registry, m_queueRegistry, primaryBackend, value, any.type);
					// A rebindable pass's places now hold what it just declared; the backing poll below rebinds them.
					UseRebindSlots(main, lowered, dynamic);
					// Membership-only change: the same structure under a layout with the same content. The installed executable's
					// view and binding tokens are per (use, view), which an equal layout keeps. So the pass takes the new layout
					// and bindings without a build; their tokens name the new layout. For a rebindable pass, the content may
					// name other resources: they are its slots' new backings.
					// While a pending build has a new declaration of the pass, the pass prepares against an older one, so only
					// the very same layout counts then.
					const bool sameLayout = lowered.resourceUses == main.resourceUses
						|| (main.preparedResourceUses == main.resourceUses && main.resourceUses && lowered.resourceUses
							&& SameLayoutContent(*main.resourceUses, *lowered.resourceUses, !lowered.rebindable));
					if (FingerprintLowering(lowered) == main.fingerprint && sameLayout) {
						main.directResolvers = std::move(lowered.directResolvers);
						if (lowered.resourceUses != main.resourceUses) {
							CountRelower(SameLayoutContent(*main.resourceUses, *lowered.resourceUses, true)
								? RelowerCause::NoOp : RelowerCause::Rebind, false);
							main.resourceUses = lowered.resourceUses;
							main.preparedResourceUses = lowered.resourceUses;
						}
						main.preparedBindings = std::move(lowered.declaredBindings);
						return;
					}
					if (!directStale && !state.pending && FingerprintLowering(lowered, false) == main.structuralFingerprint) {
						remapCandidates.push_back({mainIndex, std::move(lowered), dynamic});
						return;
					}
					const auto cause = ClassifyRelower(main, lowered, directStale);
					std::string why = directStale ? "direct resolver changed" : state.pending ? "build pending" : "structure changed:";
					if (!directStale && !state.pending) {
						const auto parts = FingerprintParts(lowered);
						for (size_t part = 0; part < parts.size(); ++part)
							if (parts[part] != main.structuralParts[part]) why += std::string(" ") + FingerprintPartName(part);
						if (parts[0] != main.structuralParts[0])
							why += " (" + main.declarationSummary + " -> " + DescribeDeclaration(lowered) + ")";
					}
					LogRelowerDiff(any.name, main, lowered, cause, why.c_str());
					const bool mustInline = directStale || (!(dynamic && dynamic->PreparesOnlyDeclaredWork()) && LoweringAddsDeclarations(main, lowered));
					CountRelower(cause, mustInline);
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
		PreparationLap("poll declarations");
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
				basic_telemetry::AddCounter("ORG.Persistent.Relower.GroupShape", static_cast<int64_t>(group.subscribers.size()));
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
				MaterializePersistentStandalone(arriving, false);
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
			PreparationLap("poll groups");
			BT_ZONE_SCOPE("ORG.Persistent.PollBackings");
			for (uint32_t index = 0; index < state.entries.size(); ++index) {
				auto& entry = state.entries[index];
				if (!entry.resource || entry.versioned) continue;
				const auto key = RotationKeyOf(entry);
				if (entry.isBound && key == entry.bound) continue;
				// Frozen (FreezeAliasedBindings): a transient this graph placed keeps the backing it bound.
				if (entry.isBound && state.freezeAliased.load(std::memory_order_acquire) && state.aliasPlaced.contains(UnwrapDynamic(entry.resource))) continue;
				if (!entry.isBound && !key.concreteID) continue;
				// A backing of another shape (a texture swapped for one with more mips, say) is a new declaration, not a
				// binding edit: the slot was admitted with its shape. Its subscribers re-lower, which gives the resource a
				// slot of the new shape; until that installs they keep the bound backing.
				if (const auto shape = ShapeOf(*entry.resource); shape != entry.shape) {
					{
						static std::atomic<uint32_t> reported{0};
						if (reported.fetch_add(1) < 8)
							spdlog::info("Persistent backing shape changed: resource='{}' mips={}->{} slices={}->{}",
								entry.resource->GetName(), entry.shape.mips, shape.mips, entry.shape.slices, shape.slices);
					}
					basic_telemetry::AddCounter("ORG.Persistent.BackingShapeChanges");
					auto& relower = state.pending ? state.pending->deferredStructural : structural;
					for (const auto subscriber : entry.subscribers)
						if (std::find(relower.begin(), relower.end(), subscriber) == relower.end()) relower.push_back(subscriber);
					continue;
				}
				auto& transaction = ensureEdit();
				// Rotated backings may not be materialized yet (per-frame presentation
				// targets are realized lazily by the legacy path).
				MaterializePersistentStandalone(std::span<Resource* const>{&entry.resource, 1}, false);
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
		if (!remapCandidates.empty()) {
			PreparationLap("poll backings");
			BT_ZONE_SCOPE("ORG.Persistent.RemapDeclarations");
			const auto live = state.program.Select();
			for (auto& candidate : remapCandidates) {
				auto& main = state.mainPasses[candidate.mainIndex];
				const auto cause = ClassifyRelower(main, candidate.lowered, false);
				const char* why = RemapDeclaration(state, *live, main, candidate.lowered);
				if (!why) {
					CountRelower(RelowerCause::Remapped, false);
					continue;
				}
				LogRelowerDiff(main.name, main, candidate.lowered, cause, why);
				const bool mustInline = !(candidate.dynamic && candidate.dynamic->PreparesOnlyDeclaredWork())
					&& LoweringAddsDeclarations(main, candidate.lowered);
				CountRelower(cause, mustInline);
				inlineStructural |= mustInline;
				structural.push_back(candidate.mainIndex);
			}
			std::sort(structural.begin(), structural.end());
			structural.erase(std::unique(structural.begin(), structural.end()), structural.end());
		}
		if (edit) {
			PreparationLap("remap declarations");
			BT_ZONE_SCOPE("ORG.Persistent.InlineEditBuild");
			if (m_sharingChangePolicy == SharingChangePolicy::Deferred) DeferSharingChanges(state, *edit);
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
		// Deferred sharing changes build on a worker of their own when no other build takes them.
		if (!state.pending && !state.deferredSharing.empty()) {
			++state.deferredSharingBuilds;
			SubmitPersistentStructuralBuild({}, {}, primaryBackend, false);
		}
		// Direct declarations of a resolver bind the set it resolved to when the pass was lowered. A preparation that resolves
		// another set would record against resources it does not name - read past a grown buffer's old backing, say. A change
		// is relowered inline above, so by here every pass must match; never silently.
		PreparationLap("edit build");
		{ BT_ZONE_SCOPE("ORG.Persistent.CheckDirectResolvers");
		for (const auto& main : state.mainPasses) {
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
}

namespace {
// Slots whose binding `live` changed since `base`: false when they differ in size (as ChangedSlots in RenderGraphPersistentStructural.cpp).
bool ChangedPublicationSlots(const persistent::SelectedPublication& base, const persistent::SelectedPublication& live, std::vector<uint32_t>& out) {
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

// The currency probe (SetCurrencyProbe): what of a kept recording's key this preparation changed against the previous one.
void RenderGraph::ProbePersistentCurrency(const std::shared_ptr<const persistent::SelectedPublication>& selected,
	const persistent::SelectedPublication& executing, std::shared_ptr<const void> bindings,
	const std::vector<std::pair<std::shared_ptr<const IVersionedResource>, uint64_t>>& boundVersions, bool perFrame) {
	auto& state = *m_compilerState->persistent;
	auto& probe = state.currencyProbe;
	auto& stats = probe.stats;
	const auto& compiled = executing.executable->resourceIndexBySlot;
	const auto boundHere = [&](uint32_t slot) { return slot < compiled.size() && compiled[slot] != UINT32_MAX; };
	const auto nameOf = [&](uint32_t slot) -> std::string {
		for (const auto& entry : state.entries)
			if (entry.slot.index == slot && entry.resource) return entry.resource->GetName();
		return "slot " + std::to_string(slot);
	};
	const auto note = [&](std::string name) {
		if (stats.names.size() < 256 || stats.names.contains(name)) ++stats.names[std::move(name)];
	};
	std::vector<std::pair<const void*, uint64_t>> versions;
	versions.reserve(boundVersions.size());
	for (const auto& [versioned, number] : boundVersions) versions.emplace_back(versioned.get(), number);
	uint64_t head = 0;
	try { head = relocation::Service::Get().Journal().Head(); } catch (...) {}
	const uint64_t placementId = state.framePlacement ? state.framePlacement->Id() : 0;
	if (!probe.primed) {
		++stats.firstRuns;
	} else {
		++stats.runs;
		bool any = false;
		if (probe.publication != selected) {
			++stats.publication;
			std::vector<uint32_t> changed;
			bool bound = false;
			if (!probe.publication || !ChangedPublicationSlots(*probe.publication, *selected, changed)) {
				bound = true;
				note("slot:<table size changed>");
			}
			for (const auto slot : changed) {
				if (!boundHere(slot)) continue;
				bound = true;
				note("slot:" + nameOf(slot));
			}
			if (bound) ++stats.publicationBound;
			any = any || bound;
		}
		// By identity, as a kept recording is checked today. Per slot (O-M6-2), a new publication counts only when it changed a
		// slot this epoch binds; an overlay of other versions on the same publication still counts.
		const bool identity = probe.publication != selected || probe.bindings != bindings;
		if (probe.bindings != bindings) {
			++stats.bindings;
			if (probe.publication == selected) any = true;
		}
		if (perFrame) ++stats.perFrame;
		if (versions != probe.versions) {
			++stats.versions;
			any = true;
			for (const auto& [versioned, number] : boundVersions) {
				const auto found = std::find_if(probe.versions.begin(), probe.versions.end(), [&](const auto& v) { return v.first == versioned.get(); });
				if (found != probe.versions.end() && found->second == number) continue;
				const auto current = versioned->SelectVersion(nullptr);
				note("version:" + (current && current->resource ? current->resource->GetName() : std::string("?")));
			}
		}
		if (head != probe.journalHead && probe.publication) {
			const auto& service = relocation::Service::Get();
			bool moved = false;
			const auto& table = probe.publication->bindings;
			for (uint32_t index = 0; index < table.Size(); ++index) {
				const auto* binding = table.TryAt(index);
				if (!binding || !binding->bound || !binding->recording || !boundHere(index)) continue;
				if (!service.MovedFrom(binding->recording->resourceID, binding->recording->backingGeneration)) continue;
				moved = true;
				note("moved:" + nameOf(index));
			}
			if (moved) {
				++stats.backings;
				any = true;
			}
		}
		if (placementId != probe.placementId || state.aliasPlans != probe.aliasPlans) {
			++stats.placement;
			any = true;
		}
		if (any) ++stats.any;
		if (any || identity) ++stats.anyByIdentity;
	}
	probe.primed = true;
	probe.publication = selected;
	probe.bindings = std::move(bindings);
	probe.versions = std::move(versions);
	probe.journalHead = head;
	probe.placementId = placementId;
	probe.aliasPlans = state.aliasPlans;
}

RenderGraph::CurrencyProbeStats RenderGraph::TakeCurrencyProbeStats() {
	if (!m_compilerState || !m_compilerState->persistent) return {};
	return std::exchange(m_compilerState->persistent->currencyProbe.stats, {});
}

void RenderGraph::DropKeptRecordings() {
	m_lastKeptRun = {};
	if (!m_compilerState || !m_compilerState->persistent) return;
	// A run still in flight holds its own reference to the packets it submitted.
	m_compilerState->persistent->syncKept.clear();
}

// Steps 3-4 of a preparation: the main segment's invocations against the selected publication, and the uploads.
RenderGraph::PersistentExecution RenderGraph::PreparePersistentExecution(rhi::Device device, uint32_t frameIndex,
	const IHostExecutionData* hostData, float deltaTime, const PersistentExecutionOptions& options) {
	auto& compiler = *m_compilerState;
	auto& state = *compiler.persistent;
	PersistentExecution execution;
	m_lastKeptRun = {};
	execution.options = options;
	execution.number = state.frameNumber;
	execution.frameIndex = frameIndex;
	execution.frameData = m_asyncUpdateHostData;
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
		PreparationLap("select");
		BT_ZONE_SCOPE("ORG.Persistent.BuildBindingTable");
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
		.executionIndex = state.frameNumber,
		.frameNumber = state.frameNumber,
		.deltaTime = deltaTime,
		.bindings = state.bindings,
		.preparationData = m_asyncUpdateHostData ? m_asyncUpdateHostData.get() : hostData,
		.admissionData = m_asyncUpdateHostData ? m_asyncUpdateHostData.get() : hostData,
		.frameData = m_asyncUpdateHostData,
	};
	// The epoch's own executable when the program has epochs; an epoch with none of its
	// passes has no main segment at all (its uploads still run).
	std::shared_ptr<const persistent::SelectedPublication> executing = selected;
	if (m_persistentEpoch != persistent::AllEpochs && selected->HasEpochs()) executing = selected->ForEpoch(m_persistentEpoch);
	// Versioned slots, the swapchain among them. No publication binds them. Each preparation binds the version it selects:
	// the one its host data's revision names (IResourceVersions), or else the current one. Admission takes that version as a
	// frame rebinding. Preparation and recording take it through an overlay of the bindings.
	std::vector<persistent::FrameRebinding> rebindings;
	std::vector<std::pair<std::shared_ptr<const IVersionedResource>, uint64_t>> versionStamps;
	bool perFrame = false;
	if (executing) {
		PreparationLap("binding table and setup");
		BT_ZONE_SCOPE("ORG.Persistent.BindVersions");
		const auto& compiled = executing->executable->resourceIndexBySlot;
		std::vector<std::pair<uint32_t, FrozenExecutionBindings::ResourceBinding>> overlay;
		for (auto& entry : state.entries) {
			if (!entry.versioned) continue;
			const auto index = entry.slot.index;
			// Only the slots this execution compiled. A slot that a pending structural build added is not compiled yet.
			if (index >= selected->bindings.Size() || index >= compiled.size() || compiled[index] == UINT32_MAX) continue;
			auto version = entry.versioned->SelectVersion(m_resolverCaptureContext.get());
			// A graph built for a pending slice count binds that version until it is current. It runs only once it is.
			if (version && version->resource && ShapeOf(*version->resource) != entry.shape)
				if (auto reshaped = entry.versioned->Reshaped(); reshaped && reshaped->resource && ShapeOf(*reshaped->resource) == entry.shape)
					version = std::move(reshaped);
			if (version) versionStamps.emplace_back(entry.versioned, version->number);
			if (!version || !version->resource)
				throw std::runtime_error("Persistent slot of '" + (entry.resource ? entry.resource->GetName() : std::string("?"))
					+ "' has no version to bind");
			// A versioned resource's slot names the version it binds now, so a pass that looks the version up by its own ID finds
			// the slot. The swapchain's slot names its wrapper, which never changes.
			if (entry.member && entry.member != version->resource) {
				QueueSlotPatchIDs(state, entry, {entry.member->GetGlobalResourceID(), 0, 0}, false);
				QueueSlotPatchIDs(state, entry, {version->resource->GetGlobalResourceID(), 0, 0}, true);
				entry.member = version->resource;
				entry.resource = entry.member.get();
			}
			// A pass on another device binds the version's representation there. A per-frame resource's versions (swapchain
			// images) belong to the host and get none, as before.
			if (entry.multiDevice && !entry.versioned->PerFrame() && state.ensureRepresentations)
				state.ensureRepresentations(*version->resource);
			const bool continues = state.aliasContinues.contains(UnwrapDynamic(version->resource.get()));
			auto binding = BindVersion(entry, std::move(version), continues);
			rebindings.push_back(std::move(binding.rebinding));
			overlay.emplace_back(index, std::move(binding.binding));
			perFrame = perFrame || entry.versioned->PerFrame();
		}
		if (!overlay.empty()) {
			auto& last = state.versionedOverlay;
			bool same = !perFrame && last.base == state.bindings && last.bound.size() == overlay.size();
			for (size_t i = 0; same && i < overlay.size(); ++i)
				same = last.bound[i].first == overlay[i].first && last.bound[i].second == overlay[i].second.owner.get();
			if (same) {
				preparation.bindings = last.bindings;
			} else {
				std::vector<std::pair<uint32_t, const void*>> bound;
				bound.reserve(overlay.size());
				for (const auto& [slot, binding] : overlay) bound.emplace_back(slot, binding.owner.get());
				preparation.bindings = FrozenExecutionBindings::Overlay(state.bindings, std::move(overlay));
				if (perFrame) last = {};
				else last = {state.bindings, preparation.bindings, std::move(bound)};
			}
		}
	}
	PersistentExecution::PreparedSegment main;
	main.label = "Main";
	main.publication = executing;
	main.rebindings = std::move(rebindings);
	main.versions = std::move(versionStamps);
	main.perFrame = perFrame;
	main.bindings = preparation.bindings;
	main.invocations.resize(selected->logical->passSlots.size());
	// A kept copy (synchronous executions, PersistentExecutionOptions::keptCopy): its kept recording is current when what it was
	// recorded for still holds. The checks come in this order, and the first that fails names the re-record.
	const bool keeping = executing && !options.ticket && options.keptCopy != UINT32_MAX;
	PersistentExecutionsState::SyncKept* kept = nullptr;
	if (keeping) {
		BT_ZONE_SCOPE("ORG.Persistent.CheckKeptRecording");
		main.keptCopy = options.keptCopy;
		if (state.syncKept.size() <= options.keptCopy) state.syncKept.resize(size_t{options.keptCopy} + 1);
		kept = &state.syncKept[options.keptCopy];
		const auto nameOfSlot = [&](uint32_t slot) -> std::string {
			for (const auto& entry : state.entries)
				if (entry.slot.index == slot && entry.resource && !entry.resource->GetName().empty()) return entry.resource->GetName();
			return "slot " + std::to_string(slot);
		};
		main.staleReason = [&]() -> std::string {
			if (!kept->recording) return kept->unkeptReason.empty() ? std::string("first run") : "not kept: " + kept->unkeptReason;
			if (options.run && kept->lease != options.run->lease) return "copy lease";
			const auto& recording = *kept->recording;
			// O-M6-3: a frame placement version taken in place, a prepared re-placement installed, or an alias plan installed.
			if (kept->placementSerial != state.placementSerial) return "frame placement";
			if (recording.publication != executing) {
				if (!recording.publication || recording.publication->executable != executing->executable) return "new executable";
				std::vector<uint32_t> changed;
				if (!ChangedPublicationSlots(*recording.publication, *executing, changed)) return "binding table resized";
				const auto& compiled = executing->executable->resourceIndexBySlot;
				for (const auto slot : changed)
					if (slot < compiled.size() && compiled[slot] != UINT32_MAX) return "binding: " + nameOfSlot(slot);
				return "publication (a slot this epoch does not bind)";
			}
			// O-M6-2: the versions of the versioned slots this epoch binds, each by its own number.
			if (recording.versions != main.versions) {
				for (const auto& [versioned, number] : main.versions) {
					const auto found = std::ranges::find_if(recording.versions, [&](const auto& stamp) { return stamp.first == versioned; });
					if (found != recording.versions.end() && found->second == number) continue;
					const auto current = versioned ? versioned->SelectVersion(nullptr) : nullptr;
					return "version: " + (current && current->resource ? current->resource->GetName() : std::string("?"));
				}
				return "version: a versioned slot is no longer bound";
			}
			if (recording.bindings != preparation.bindings) return "bindings";
			const auto resourceHeap = DescriptorHeapManager::GetInstance().GetSRVDescriptorHeap(BackendInstanceId::Primary).GetHandle();
			const auto samplerHeap = DescriptorHeapManager::GetInstance().GetSamplerDescriptorHeap(BackendInstanceId::Primary).GetHandle();
			if (recording.resourceHeap.index != resourceHeap.index || recording.resourceHeap.generation != resourceHeap.generation
				|| recording.samplerHeap.index != samplerHeap.index || recording.samplerHeap.generation != samplerHeap.generation)
				return "descriptor heaps";
			if (!PersistentRecordingBackingsCurrent(recording)) return "backing moved: " + PersistentRecordingStaleBacking(recording);
			// Batch reuse: the passes' revisions and skip answers are checked per batch, below. A staged recording is confirmed whole.
			if (m_keptBatchReuse && !kept->staged) return {};
			std::string firstStale;
			for (size_t i = 0; i < recording.revisions.size(); ++i) {
				const auto& [pass, hash] = recording.revisions[i];
				uint64_t now = 0;
				if (!pass->RevisionHash(now) || now != hash) {
					const std::string name = i < kept->revisionNames.size() ? kept->revisionNames[i] : std::string("?");
					if (kept->staged && i < kept->stagedRevisionRows.size()) {
						// Diagnostics (ORG_STAGED_REVISION_DIAG): the rows that changed since the staging.
						std::vector<uint64_t> rows;
						(void)pass->RevisionRows(rows);
						const auto& staged = kept->stagedRevisionRows[i];
						std::string changed;
						int shown = 0;
						for (size_t row = 0; row < (std::max)(rows.size(), staged.size()) && shown < 8; ++row) {
							const uint64_t a = row < staged.size() ? staged[row] : ~0ull, b = row < rows.size() ? rows[row] : ~0ull;
							if (a == b) continue;
							changed += fmt::format(" [{}] {:#x} -> {:#x}", row, a, b);
							++shown;
						}
						spdlog::info("Staged revision diagnostics: copy {} pass '{}': {} rows staged, {} now;{}", options.keptCopy, name,
							staged.size(), rows.size(), changed);
						// Every stale pass is logged; the first names the re-record.
						if (firstStale.empty()) firstStale = "revision: " + name;
						continue;
					}
					return "revision: " + name;
				}
			}
			if (!firstStale.empty()) return firstStale;
			for (size_t i = 0; i < kept->skips.size(); ++i) {
				const auto& [pass, skipped] = kept->skips[i];
				if (pass->SkipsFor(preparation.preparationData) != (skipped != 0))
					return "skip: " + (i < kept->revisionNames.size() ? kept->revisionNames[i] : std::string("?"));
			}
			return {};
		}();
		if (!main.staleReason.empty()) {
			// A staged recording that is stale at its first use: named so (KeptRun::staged).
			if (kept->staged) {
				main.staleReason = "staged: " + main.staleReason;
				m_lastKeptRun.staged = true;
				kept->staged = false;
			}
			kept->recording.reset();
		}
	}
	// Batch reuse (O-M6-6, SetKeptBatchReuse): what every batch shares holds, so each batch is checked on its own. A batch whose
	// passes' revisions and skip answers are unchanged, and whose packet was kept, is submitted again; the others record. A batch
	// body depends only on its own passes and the compiled graph: its entry barriers are recorded at submission.
	std::vector<uint8_t> reuseBatches;
	if (keeping && main.staleReason.empty() && m_keptBatchReuse && !kept->staged) {
		BT_ZONE_SCOPE("ORG.Persistent.CheckKeptBatches");
		const auto& recording = *kept->recording;
		const size_t count = executing->executable->graph->batches.size();
		reuseBatches.assign(count, 1);
		main.staleBatches.assign(count, {});
		std::string whole;
		if (!kept->batchReusable.empty() && kept->batchReusable.size() != count) whole = "batches changed";
		for (size_t batch = 0; whole.empty() && batch < kept->batchReusable.size(); ++batch)
			if (!kept->batchReusable[batch]) {
				reuseBatches[batch] = 0;
				main.staleBatches[batch] = "not kept: " + (batch < kept->batchWhy.size() ? kept->batchWhy[batch] : std::string("?"));
			}
		const bool mapped = kept->revisionBatches.size() == recording.revisions.size();
		const auto nameOf = [&](size_t i) { return i < kept->revisionNames.size() ? kept->revisionNames[i] : std::string("?"); };
		for (size_t i = 0; whole.empty() && i < recording.revisions.size(); ++i) {
			const auto batch = mapped ? kept->revisionBatches[i] : UINT32_MAX;
			if (batch < count && !reuseBatches[batch]) continue;
			const auto& [pass, hash] = recording.revisions[i];
			uint64_t now = 0;
			std::string why;
			if (!pass->RevisionHash(now) || now != hash) why = "revision: " + nameOf(i);
			else if (i < kept->skips.size() && kept->skips[i].first == pass
				&& pass->SkipsFor(preparation.preparationData) != (kept->skips[i].second != 0))
				why = "skip: " + nameOf(i);
			if (why.empty()) continue;
			if (batch >= count) whole = std::move(why);
			else {
				reuseBatches[batch] = 0;
				main.staleBatches[batch] = std::move(why);
			}
		}
		const auto reused = static_cast<size_t>(std::ranges::count(reuseBatches, uint8_t{1}));
		if (whole.empty() && reused == count) {
			// Every batch current: the recording is submitted as it is.
			reuseBatches.clear();
			main.staleBatches.clear();
		} else if (!whole.empty() || reused == 0) {
			main.staleReason = !whole.empty() ? whole : *std::ranges::find_if(main.staleBatches, [](const auto& why) { return !why.empty(); });
			if (!whole.empty()) main.staleBatches.clear();
			reuseBatches.clear();
			kept->recording.reset();
		}
	}
	bool partial = !reuseBatches.empty();
	// External waits (explicit and resolver-provided) become per-frame producer waits.
	const auto appendWaits = [&](const auto& pass) {
		const auto appendWait = [&](const ExternalTimelinePoint& wait) {
			if (!wait.timeline || !wait.value) return;
			const auto handle = wait.timeline.GetHandle();
			const uint64_t identity = m_runtime->TimelineIdentity(wait.timeline);
			bool bound = std::ranges::any_of(main.foreignTimelines, [&](const auto& binding) { return binding.identity == identity; });
			for (size_t slot = 0; slot < m_queueRegistry.SlotCount() && !bound; ++slot)
				bound = SameTimeline(m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(static_cast<uint32_t>(slot))), wait.timeline);
			if (!bound) main.foreignTimelines.push_back({identity, handle});
			main.waits.push_back({pass.id, {identity, wait.value}});
		};
		for (const auto& wait : pass.explicitWaits) appendWait(wait);
		for (const auto groupIndex : pass.groups) for (const auto& wait : state.groups[groupIndex].waits) appendWait(wait);
	};
	// A current kept recording: only this frame's waits are taken. Nothing is prepared. Its packets bind the timelines they were
	// recorded with, so a wait on another timeline records again.
	if (keeping && main.staleReason.empty()) {
		for (const auto& pass : state.mainPasses)
			if (RunsInPersistentEpoch(pass.epoch) && pass.id.index < main.invocations.size()) appendWaits(pass);
		std::vector<uint64_t> foreign;
		for (const auto& binding : main.foreignTimelines) foreign.push_back(binding.identity);
		std::ranges::sort(foreign);
		if (foreign == kept->foreignTimelines) {
			main.replay = kept->recording;
			// The names of the executable it was recorded on: the same one.
			main.names = state.mainNames;
			if (partial) main.reuseBatches = std::move(reuseBatches);
		} else {
			main.staleReason = kept->staged ? "staged: foreign timelines" : "foreign timelines";
			main.staleBatches.clear();
			if (kept->staged) m_lastKeptRun.staged = true;
			kept->staged = false;
			kept->recording.reset();
			main.waits.clear();
			main.foreignTimelines.clear();
			partial = false;
		}
	}
	// The revision audit (O-M6-7, SetKeptRecordingAudit): every N-th reuse of a copy prepares the passes anyway, compares each
	// pass's frame data with the kept recording's and discards what it prepared. The run still submits the kept recording.
	// A staged recording's first use (RecordStagedCopy) is confirmed: the passes are prepared once more, as an audit does, and
	// their frame data must equal the staged ones. The staging took its host data from the application's frame for the build,
	// which may differ from this frame's in what a revision does not name. On a difference the run records instead.
	const bool confirming = main.replay && kept && kept->staged;
	if (confirming) {
		kept->staged = false;
		m_lastKeptRun.staged = true;
	}
	const bool auditing = !confirming && main.replay && kept && m_keptAuditEvery != 0 && ++kept->reusesSinceAudit >= m_keptAuditEvery;
	if (auditing) {
		kept->reusesSinceAudit = 0;
		m_lastKeptRun.audited = true;
	}
	std::string confirmRefusal;
	// One pass's new preparation against the kept recording's, the pass's index among the recording's revisioned passes
	// counted by auditIndex.
	size_t auditIndex = 0;
	// 0: the same frame data, 1: other frame data, 2: not comparable, 3: a pass with no revision (a current recording has none).
	const auto compareFrameData = [&](const auto& any, const PreparedPass& packet) -> int {
		const RenderGraphPass* object = std::visit([](auto& value) -> const RenderGraphPass* {
			using T = std::decay_t<decltype(value)>;
			if constexpr (std::is_same_v<T, std::monostate>) return nullptr;
			else return value.pass.get();
		}, any.pass);
		uint64_t hash = 0;
		if (!object || !object->RevisionHash(hash)) return 3;
		// By the pass: a recording merged by batch reuse holds its entries in another order than this preparation's.
		const auto& revisions = main.replay->revisions;
		size_t at = auditIndex;
		if (at >= revisions.size() || revisions[at].first != object)
			at = static_cast<size_t>(std::ranges::find(revisions, object, &std::pair<const RenderGraphPass*, uint64_t>::first) - revisions.begin());
		auditIndex = at + 1;
		const auto audit = packet.AuditFrameData();
		if (at >= revisions.size() || at >= kept->audits.size() || !kept->audits[at] || !audit) return 2;
		return audit->Equal(*kept->audits[at]) ? 0 : 1;
	};
	const auto confirmPass = [&](const auto& any, const PreparedPass& packet) {
		if (!confirmRefusal.empty()) return;
		const auto compared = compareFrameData(any, packet);
		if (compared == 1) confirmRefusal = "staged: frame data of " + any.name;
		else if (compared == 2) confirmRefusal = "staged: frame data not comparable: " + any.name;
	};
	const auto auditPass = [&](const auto& any, const PreparedPass& packet) {
		const auto compared = compareFrameData(any, packet);
		if (compared == 3) return;
		if (compared == 2) {
			++m_lastKeptRun.auditUncomparable;
			return;
		}
		++m_lastKeptRun.auditCompared;
		if (compared == 0) return;
		m_lastKeptRun.auditMismatches.push_back(any.name);
		if (m_keptAuditWarned.size() < 256 && m_keptAuditWarned.insert(any.name).second)
			spdlog::warn("Kept recording audit: pass '{}' (epoch {}) prepared other frame data than its kept recording's while its "
				"revision is the same: its Revision() misses an input", any.name, m_passEpochFilter);
	};
	// A partial run (batch reuse) prepares the passes of the batches it records, and with the audit also the reused ones, to compare.
	const std::vector<uint32_t>* batchOf = partial ? &BatchOfPrepared(state, executing->executable) : nullptr;
	const auto reusedPass = [&](uint32_t prepared) {
		if (!batchOf) return false;
		const auto batch = prepared < batchOf->size() ? (*batchOf)[prepared] : UINT32_MAX;
		return batch < main.reuseBatches.size() && main.reuseBatches[batch] != 0;
	};
	if (!main.replay || auditing || confirming || partial) {
		PreparationLap("bind versions and setup");
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
				if (pass.id.index < mainNames->size())
					(*mainNames)[pass.id.index] = m_masterPassList[pass.masterIndex].name;
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
				if (!entry.resource || (!entry.isBound && !entry.versioned) || entry.slot.index >= selected->bindings.Size()) continue;
				a_slots.emplace_back(uses[u].binding.registryResourceID, entry.slot.index);
				a_slots.emplace_back(uses[u].binding.globalResourceID, entry.slot.index);
			}
		};
		for (uint32_t mainIndex = 0; mainIndex < state.mainPasses.size() && executing; ++mainIndex) {
			auto& pass = state.mainPasses[mainIndex];
			// Another epoch's pass: neither prepared nor recorded now. Its slot patches wait.
			if (!RunsInPersistentEpoch(pass.epoch)) continue;
			// In a batch submitted again: nothing to prepare, unless the audit compares it. Its slot patches wait too.
			const bool reused = reusedPass(pass.id.index);
			if (reused && !auditing) continue;
			if (pass.slotsDirty || pass.slotsExecutable != selected->executable.get()) {
				auto slots = std::make_shared<FramePreparationContext::ResourceSlots>();
				auto expose = [&](uint32_t entryIndex) {
					const auto& entry = state.entries[entryIndex];
					if (!entry.resource || (!entry.isBound && !entry.versioned)) return;
					const auto slot = entry.slot.index;
					if (slot >= selected->bindings.Size()) return; // reserved by a pending structural build
					slots->emplace_back(entry.resourceID, slot);
					slots->emplace_back(entry.resource->GetGlobalResourceID(), slot);
					if (auto* concrete = UnwrapDynamic(entry.resource)) slots->emplace_back(concrete->GetGlobalResourceID(), slot);
				};
				for (const auto index : pass.directEntries) expose(index);
				for (const auto groupIndex : pass.groups) for (const auto index : state.groups[groupIndex].memberEntries) expose(index);
				// Registry handle identities may differ from scheduling identities.
				const auto view = GetPassView(m_masterPassList[pass.masterIndex]);
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
			auto& any = m_masterPassList[pass.masterIndex];
			passPreparation.resourceSlots = pass.slots;
			passPreparation.preparedResourceUses = pass.preparedResourceUses;
			passPreparation.preparedBindings = pass.preparedBindings;
			passPreparation.backendInstance = pass.device;
			if (const auto* entry = m_backendDevices.Find(pass.device)) passPreparation.device = entry->device;
            // A declared use of a slot the publication leaves unbound (a versioned slot) resolves through this preparation's
            // bindings, which hold the version it binds.
            passPreparation.resolveDeclaredView = [&pass, selected, bindings = passPreparation.bindings](uint32_t use, uint32_t view) {
                const auto& token = pass.declaredViews.at(use).at(view);
                const auto resource = selected->SlotOf(token);
                const auto* version = selected->bindings.TryAt(resource.index);
                const auto slot = version && !version->bound
                    ? bindings->Views(PreparedResourceReference{resource.index}).Resolve(selected->RequestOf(token))
                    : selected->Resolve(token);
                return pass.device == BackendInstanceId::Primary ? slot
                    : DescriptorHeapManager::GetInstance().TranslateDescriptorSlot(slot, pass.device);
            };
            passPreparation.resolveDeclaredResource = [&pass, selected, bindings = passPreparation.bindings](uint32_t use) {
                const auto& token = pass.declaredBindings.at(use);
                const auto resource = selected->SlotOf(token);
                const auto* version = selected->bindings.TryAt(resource.index);
                return version && !version->bound ? bindings->Resolve(PreparedResourceReference{resource.index}) : selected->ResolveNative(token);
            };
            // A name declared through a resolver: a group's first member bound this frame, in capture order, or the version of a
            // versioned resource bound this frame.
            passPreparation.resolveResolverView = [&pass, &state, bindings = passPreparation.bindings](
                uint32_t resolverGroup, BindlessViewRequest request) -> rhi::DescriptorSlot {
                if (!pass.preparedResourceUses || resolverGroup >= pass.preparedResourceUses->resolverViews.size() || !bindings) return {};
                const auto& identity = pass.preparedResourceUses->resolverViews[resolverGroup].identity;
                const auto translate = [&](rhi::DescriptorSlot slot) {
                    return pass.device == BackendInstanceId::Primary ? slot
                        : DescriptorHeapManager::GetInstance().TranslateDescriptorSlot(slot, pass.device);
                };
                for (const auto groupIndex : pass.groups) {
                    const auto& group = state.groups[groupIndex];
                    if (group.key != identity) continue;
                    for (const auto entryIndex : group.memberEntries) {
                        const auto& entry = state.entries[entryIndex];
                        if (!entry.resource || !entry.isBound || entry.slot.index >= bindings->Size()) continue;
                        return translate(bindings->Views(PreparedResourceReference{entry.slot.index}).Resolve(request));
                    }
                }
                if (const auto entryIndex = VersionedEntry(state, pass, identity.get()); entryIndex != UINT32_MAX) {
                    const auto& entry = state.entries[entryIndex];
                    if (entry.slot.index < bindings->Size())
                        return translate(bindings->Views(PreparedResourceReference{entry.slot.index}).Resolve(request));
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
				const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - prepareStarted).count();
				cost.first += static_cast<uint64_t>(ns);
				++cost.second;
				if (t_preparationLaps) t_preparationLaps->Pass(static_cast<double>(ns) / 1e6, any.name);
				if (UpdateProfileOn()) UpdateProfileAdd("prepare pass", any.name, static_cast<double>(ns) / 1e3);
			}
			packet.SetDebugName(any.name);
			// A pass the host turned off for this execution by its host data (RenderGraphPass::SetSkipWhen) records only its barriers.
			if (const RenderGraphPass* object = std::visit([](auto& value) -> const RenderGraphPass* {
					using T = std::decay_t<decltype(value)>;
					if constexpr (std::is_same_v<T, std::monostate>) return nullptr;
					else return value.pass.get();
				}, any.pass);
				object && object->SkipsFor(passPreparation.preparationData))
				packet.MarkSkippedAhead();
			if (auditing && (!partial || reused)) {
				auditPass(any, packet);
				packet.Abandon(AbandonReason::GenerationInvalidated);
				continue;
			}
			// Kept as a normal preparation's, in case the confirmation fails and the run records.
			if (confirming) confirmPass(any, packet);
			main.invocations[pass.id.index] = std::move(packet);
			if (auto* ticket = options.ticket) {
				const RenderGraphPass* object = std::visit([](auto& value) -> const RenderGraphPass* {
					using T = std::decay_t<decltype(value)>;
					if constexpr (std::is_same_v<T, std::monostate>) return nullptr;
					else return value.pass.get();
				}, any.pass);
				uint64_t hash = 0;
				if (object && object->RevisionHash(hash)) {
					ticket->revisions.emplace_back(object, hash);
					ticket->revisionNames.push_back(any.name);
				}
				else {
					if (!ticket->uncheckable) ticket->uncheckableBy = any.name + (!object ? " (no pass object)" : " (no revision hash)");
					ticket->uncheckable = true;
				}
			} else if (keeping) {
				// What a kept copy's next run checks its recording against: each pass's revision and skip answer.
				const RenderGraphPass* object = std::visit([](auto& value) -> const RenderGraphPass* {
					using T = std::decay_t<decltype(value)>;
					if constexpr (std::is_same_v<T, std::monostate>) return nullptr;
					else return value.pass.get();
				}, any.pass);
				uint64_t hash = 0;
				if (object && object->RevisionHash(hash)) {
					main.revisions.emplace_back(object, hash);
					main.revisionNames.push_back(any.name);
					main.revisionPrepared.push_back(pass.id.index);
					main.skips.emplace_back(object, object->SkipsFor(passPreparation.preparationData) ? uint8_t{1} : uint8_t{0});
					if (m_keptAuditEvery || options.staged || confirming)
						main.audits.push_back(main.invocations[pass.id.index].AuditFrameData());
				} else {
					const auto why = any.name + (!object ? " (no pass object)" : " (no revision hash)");
					if (!main.uncheckable) main.uncheckableBy = why;
					main.uncheckable = true;
					main.uncheckablePrepared.emplace_back(pass.id.index, why);
				}
			}
			if (namesDirty) (*mainNames)[pass.id.index] = any.name;
			// A confirmed reuse, and a partial run, took the frame's waits already, above.
			if (!confirming && !partial) appendWaits(pass);
		}
	}
	if (confirming) {
		if (confirmRefusal.empty()) {
			// Confirmed: the staged recording is submitted, and what was prepared to compare is abandoned.
			for (auto& invocation : main.invocations)
				if (invocation) {
					invocation.Abandon(AbandonReason::GenerationInvalidated);
					invocation = PreparedPass{};
				}
			main.revisions.clear();
			main.revisionNames.clear();
			main.skips.clear();
			main.audits.clear();
			main.uncheckable = false;
			main.uncheckableBy.clear();
		} else {
			// Recorded from this preparation instead, as a stale copy's run is.
			main.replay.reset();
			main.staleReason = std::move(confirmRefusal);
			kept->recording.reset();
		}
	}
	if (executing && !options.ticket && m_currencyProbe.load(std::memory_order_relaxed)) ProbePersistentCurrency(selected, *executing, main.bindings, main.versions, main.perFrame);
	if (executing) execution.main = std::move(main);
	PreparationLap("prepare invocations");
	// 4. Uploads.
	// Tickets have none: the host records its uploads itself when it submits the ticket, and tickets don't
	// support any other upload pass.
	if (options.staged) {
		// A staged recording takes no uploads: they belong to the frames that run.
	} else if (options.ticket) {
		for (const auto index : state.uploadMasterIndices)
			if (m_masterPassList[index].name != "org.host.uploads")
				throw std::logic_error("Tickets do not support the upload pass '" + m_masterPassList[index].name + "'");
	} else {
		BT_ZONE_SCOPE("ORG.Persistent.PrepareUploads");
		// Each upload pass records its commands now. They are submitted through the graph's external client just before
		// the main execution, with the resources the commands touch as the client's accesses.
		for (const auto masterIndex : state.uploadMasterIndices) {
			BT_ZONE_SCOPE("ORG.Persistent.RecordUploads");
			auto& any = m_masterPassList[masterIndex];
			std::visit([&](auto& value) {
				using T = std::decay_t<decltype(value)>;
				if constexpr (!std::is_same_v<T, std::monostate>) {
					auto* immediate = !value.pass->UsesTypedPreparation()
						? dynamic_cast<IHasImmediateModeCommands*>(value.pass.get()) : nullptr;
					if (!immediate) throw std::logic_error("Pre pass '" + any.name + "' does not record immediate commands");
					ImmediateExecutionContext context{device, {org::imm::ImmediatePassKind::Render, m_immediateDispatch,
						&PersistentResolveByIdThunk, &PersistentResolveByPtrThunk, this}, frameIndex, hostData};
					immediate->RecordImmediateCommands(context);
					const bool hasWork = context.list.HasRecordedWork();
					auto frame = context.list.Finalize();
					auto effect = immediate->TakeOwnedImmediateSubmissionEffect();
					if (!hasWork && !effect) return;
					PersistentExecution::PendingUpload upload;
					upload.name = any.name;
					std::vector<Resource*> resources;
					for (const auto& requirement : frame.requirements) {
						auto* resource = requirement.resourceHandleAndRange.resource.IsEphemeral()
							? requirement.resourceHandleAndRange.resource.GetEphemeralPtr() : _registry.Resolve(requirement.resourceHandleAndRange.resource);
						if (!resource) throw std::runtime_error("Pre pass '" + any.name + "' references an unresolvable resource");
						resources.push_back(resource);
						upload.accesses.push_back({resource, requirement.resourceHandleAndRange.range, requirement.state});
					}
					MaterializePersistentStandalone(resources, false);
					upload.bytecode = std::move(frame.bytecode);
					upload.keepAlive = std::shared_ptr<org::imm::KeepAliveBag>(std::move(frame.keepAlive));
					if (effect) {
						upload.commit = std::move(effect->commit);
						upload.signals = std::move(effect->completionSignals);
					}
					execution.uploads.push_back(std::move(upload));
				}
			}, any.pass);
		}
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
	return execution;
}

} // namespace org
