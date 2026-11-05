// Persistent execution, frame preparation: installing passes into the program, polling declarations, resolver groups and
// backings, and preparing the frame's invocations and dynamic segments. See RenderGraphPersistentState.h.
#include "OpenRenderGraph/Trace.h"
#include "RenderGraphPersistentState.h"
#include "Render/PipelineLibrary.h"

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
	state.deferredSharing.erase(entry.slot.index); // superseded
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
	state.deferredSharing.erase(entry.slot.index); // superseded
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
		const bool swapchain = graph.IsSwapchainResource(use.resource);
		const auto index = DirectEntry(state, edit, use.resourceID, use.resource, ShapeOf(*use.resource), swapchain, use.owner);
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

void persistent_impl::UseRebindSlots(const RenderGraph& graph, State::MainPass& main, LoweredPass& lowered,
	const IDynamicDeclaredResources* dynamic) {
	if (!dynamic || !dynamic->RebindsConcreteResources()) return;
	if (dynamic->PreparesOnlyDeclaredWork())
		throw std::logic_error("Pass '" + main.name + "' says RebindsConcreteResources, so it must say PreparesOnlyDeclaredWork false");
	if (lowered.device != BackendInstanceId::Primary) return;
	lowered.rebindable = true;
	std::vector<Resource*> places; // Distinct resources, in the order the lowering names them.
	const auto rebind = [&](LoweredUse& use) {
		if (!use.resource || use.owner || graph.IsSwapchainResource(use.resource)) return;
		if (dynamic_cast<DynamicResource*>(use.resource) || dynamic_cast<DynamicGloballyIndexedResource*>(use.resource)) return;
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

void RenderGraph::PreparePersistentFrame(rhi::Device device, uint8_t frameIndex, const IHostExecutionData* hostData, float deltaTime,
	const PersistentExecutionOptions& options) {
	BT_ZONE_SCOPE("ORG.Persistent.PrepareFrame");
	if (m_compilerState->frameProductionStopped) throw std::logic_error("Frame production has stopped");
	// Persistent structural builds, preparation and ownership retirement run on the task service; there is no inline fallback.
	if (!m_taskService) throw std::logic_error("Persistent execution requires a task service");
	auto& compiler = *m_compilerState;
	const bool bootstrap = !compiler.persistent;
	m_runtime->Pipelines()->PublishRebuilt();
	if (bootstrap) BootstrapPersistentGraph();
	auto& state = *compiler.persistent;
	state.frameIndex = frameIndex;
	state.frameNumber = ++compiler.asyncPreparationFrameNumber;
	RetireIdlePersistentAliasPools(state);
	org::runtime::ScopedActiveGraphServices activeServices(m_uploadService.get(), m_descriptorService.get());
	BindPersistentFrame(options.ticket != nullptr, bootstrap);
	state.prepared = PreparePersistentExecution(device, frameIndex, hostData, deltaTime, options);
}

bool RenderGraph::RequestPassPipelines(const PassParameters& resources, DeviceInstanceId device, const std::string& passName, std::string* failure) {
	auto& pipelines = *m_runtime->Pipelines();
	bool ready = true;
	for (const auto& entry : resources.declaredCommandSignatures) {
		try {
			pipelines.EnsureCommandSignature(entry, device);
		} catch (const std::exception& e) {
			const auto message = "Pass '" + passName + "' on device " + std::to_string(static_cast<uint8_t>(device)) + ": " + e.what();
			if (!failure) throw std::runtime_error(message);
			*failure = message;
			return false;
		}
	}
	for (const auto& entry : resources.declaredPipelines) {
		pipelines.Request(entry, device);
		const auto status = entry->Status(device);
		if (status == PipelineBuildStatus::Failed) {
			const auto message = "Pipeline '" + entry->Name() + "' of pass '" + passName + "' failed to build on device "
				+ std::to_string(static_cast<uint8_t>(device)) + ": " + entry->Error(device);
			if (!failure) throw std::runtime_error(message);
			*failure = message;
			return false;
		}
		ready &= status == PipelineBuildStatus::Ready;
	}
	return ready;
}

void RenderGraph::AwaitPassPipelines(const PassParameters& resources, DeviceInstanceId device, const std::string& passName) {
	auto& pipelines = *m_runtime->Pipelines();
	for (const auto& entry : resources.declaredPipelines) pipelines.Request(entry, device);
	for (const auto& entry : resources.declaredPipelines) pipelines.Wait(entry, device);
	RequestPassPipelines(resources, device, passName);
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
		state.admission->SetClosedExecutions(m_persistentClosedExecutions);
		state.admission->SetExternalHomeStates(m_persistentExternalHomeStates);
		state.ensureRepresentations = [this](Resource& resource) { EnsurePersistentRepresentations(resource); };
		// Tickets record their uploads on the graphics queue, or on the queue that first uses them (RecordPendingUploads).
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
					UseRebindSlots(*this, main, lowered, dynamic);
					RequestPassPipelines(value.resources, lowered.device, main.name);
					InstallMainPass(*this, state, edit, main, std::move(lowered), mainIndex, false);
				}
			}, any.pass);
		}
		// Every pass's pipelines build in parallel; the first publication waits for all of them.
		{
			BT_ZONE_SCOPE("ORG.Persistent.AwaitPipelines");
			for (auto& main : state.mainPasses) {
				std::visit([&](auto& value) {
					using T = std::decay_t<decltype(value)>;
					if constexpr (!std::is_same_v<T, std::monostate>) AwaitPassPipelines(value.resources, main.device, main.name);
				}, m_masterPassList[main.masterIndex].pass);
			}
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
		spdlog::info("Persistent graph bootstrap: mainPasses={} uploads={} slots={} groups={} batches={} build_ms={:.2f}",
			state.mainPasses.size(), state.uploadMasterIndices.size(), state.entries.size(), state.groups.size(),
			ready->executable->graph->batches.size(),
			std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
	}
}

// Steps 0-2 of a preparation: captures, the passes' declarations, group memberships and backings, and the edits they make.
void RenderGraph::BindPersistentFrame(bool ticket, bool bootstrap) {
	auto& compiler = *m_compilerState;
	auto& state = *compiler.persistent;
	const auto primaryBackend = m_backendDevices.empty() ? rhi::Backend::Null : m_backendDevices.front().backend;

	// 0. Readback captures. Captures after a pass become passes inserted into the main graph (step 1).
	std::vector<ExternalPassDesc> interrupting;
	if (m_drivesFrameServices) PlacePersistentCaptures(*this, state, ticket, interrupting);
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
		UpdatePersistentFrameInterruptingPasses(std::move(interrupting), primaryBackend);
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
					// On the bootstrap frame every pass was just declared from this frame's state.
					const bool declared = !bootstrap && (main.awaitingPipelines
						|| (dynamic && dynamic->DeclaredResourcesChanged() && !dynamic->DeclarationsProvidedByImmediateCommands()));
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
					// A pipeline the declaration newly names may still be building. The pass keeps preparing against the declaration it
					// has until it is ready. A stale direct resolver cannot wait: its old declaration names resources that may be gone.
					if (directStale) AwaitPassPipelines(value.resources, lowered.device, main.name);
					else {
						std::string failure;
						const bool ready = RequestPassPipelines(value.resources, lowered.device, main.name, &failure);
						if (!failure.empty()) spdlog::error("Pass '{}' keeps its previous declaration: {}", main.name, failure);
						main.awaitingPipelines = !ready && failure.empty();
						if (!ready) return;
					}
					main.awaitingPipelines = false;
					// A rebindable pass's places now hold what it just declared; the backing poll below rebinds them.
					UseRebindSlots(*this, main, lowered, dynamic);
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
		if (!remapCandidates.empty()) {
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
}

// Steps 3-4 of a preparation: the main segment's invocations against the selected publication, and the uploads.
RenderGraph::PersistentExecution RenderGraph::PreparePersistentExecution(rhi::Device device, uint8_t frameIndex,
	const IHostExecutionData* hostData, float deltaTime, const PersistentExecutionOptions& options) {
	auto& compiler = *m_compilerState;
	auto& state = *compiler.persistent;
	PersistentExecution execution;
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
		rebinding.backing.hostOwned = concrete->HostOwned();
		rebinding.recording = snapshot;
		if (overlay.empty()) overlay = state.bindings->Resources();
		overlay[entry.slot.index] = {snapshot->resource, snapshot->allocationOwner, snapshot->views, snapshot->descriptorOwner, nullptr};
		rebindings.push_back(std::move(rebinding));
	}
	if (!overlay.empty()) preparation.bindings = FrozenExecutionBindings::Sparse(std::move(overlay));
	// The epoch's own executable when the program has epochs; an epoch with none of its
	// passes has no main segment at all (its uploads still run).
	std::shared_ptr<const persistent::SelectedPublication> executing = selected;
	if (m_persistentEpoch != persistent::AllEpochs && selected->HasEpochs()) executing = selected->ForEpoch(m_persistentEpoch);
	PersistentExecution::PreparedSegment main;
	main.label = "Main";
	main.publication = executing;
	main.rebindings = std::move(rebindings);
	main.bindings = preparation.bindings;
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
			passPreparation.preparedResourceUses = pass.preparedResourceUses;
			passPreparation.preparedBindings = pass.preparedBindings;
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
                if (!pass.preparedResourceUses || resolverGroup >= pass.preparedResourceUses->resolverViews.size() || !bindings) return {};
                const auto& identity = pass.preparedResourceUses->resolverViews[resolverGroup].identity;
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
			if (auto* ticket = options.ticket) {
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
	if (executing) execution.main = std::move(main);
	// 4. Uploads.
	// Tickets have none: the host records its uploads itself when it submits the ticket, and tickets don't
	// support any other upload pass.
	if (options.ticket) {
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
					MaterializePersistentStandalone(resources);
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
