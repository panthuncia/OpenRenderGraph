// Persistent execution mode: the main graph runs from a selected persistent
// publication that is rebuilt only by explicit structural edits. Truly dynamic
// passes (immediate-mode uploads, per-frame readbacks) run in Pre/Tail segments
// built per frame from tiny throwaway programs. All segments are admitted in
// order through one SynchronousAdmission, so cross-segment hazards are ordinary
// previous-submission waits and barriers.
//
// Migration status: every existing pass is hosted through the legacy adapter
// below (declaration lowered once, PrepareFrame per frame). Structural rebuilds
// run inline on the owner thread. Alias placement is planned inside the
// structural build (PlanPersistentAliasPlacement): the executable is scheduled
// once, transient lifetimes are packed into persistent pool heaps, placement
// orderings are added, and the executable is rebuilt over the placed bindings.
#include "Render/RenderGraph/RenderGraph.h"
#include "RenderGraphCompilerState.h"
#include "FrameRecording.h"
#include "Render/RenderGraph/ExperimentalRhiExecution.h"
#include "Render/RenderGraph/PersistentGraph.h"
#include "Render/Runtime/ScopedActiveGraphServices.h"
#include "Interfaces/IDynamicDeclaredResources.h"
#include "Resources/DynamicResource.h"
#include "Resources/BackedResource.h"
#include "Resources/GloballyIndexedResource.h"
#include "Resources/PixelBuffer.h"
#include "Resources/Buffers/DynamicBufferBase.h"
#include "Resources/MemoryStatisticsComponents.h"
#include "Managers/Singletons/DeviceManager.h"
#include "Managers/Singletons/DeletionManager.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <optional>
#include <bit>
#include <boost/functional/hash.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <map>
#include <numeric>
#include <queue>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include "RenderPasses/Base/TypedRenderGraphPass.h"
#include <BasicTelemetry/Tracy.h>
#include <spdlog/spdlog.h>

namespace org {

namespace {

using experimental::CompileRange;
using experimental::CompileResourceShape;
using experimental::CompileResourceState;
using experimental::PreparedBackingState;
using experimental::PreparedStateRegion;

ResourceRegistry::RegistryHandle PersistentResolveByIdThunk(void* user, ResourceIdentifier const& id, bool allowFailure) {
	return static_cast<RenderGraph*>(user)->RequestResourceHandle(id, allowFailure);
}
ResourceRegistry::RegistryHandle PersistentResolveByPtrThunk(void* user, Resource* ptr, bool allowFailure) {
	return static_cast<RenderGraph*>(user)->RequestResourceHandle(ptr, allowFailure);
}

Resource* UnwrapDynamic(Resource* resource) noexcept {
	while (resource) {
		if (auto* dynamic = dynamic_cast<DynamicResource*>(resource)) {
			auto backing = dynamic->GetResource();
			if (!backing) return nullptr;
			resource = backing.get();
			continue;
		}
		if (auto* dynamic = dynamic_cast<DynamicGloballyIndexedResource*>(resource)) {
			auto backing = dynamic->GetResource();
			if (!backing) return nullptr;
			resource = backing.get();
			continue;
		}
		return resource;
	}
	return nullptr;
}

// Shape from the logical description when the texture has no backing yet: an
// unmaterialized PixelBuffer reports default mip/array counts until realized.
CompileResourceShape ShapeOf(const Resource& resource) {
	const auto* concrete = UnwrapDynamic(const_cast<Resource*>(&resource));
	if (const auto* texture = dynamic_cast<const PixelBuffer*>(concrete); texture && !texture->IsMaterialized()) {
		const auto& desc = texture->GetDescription();
		const uint32_t slices = desc.isCubemap ? 6u * desc.arraySize : (desc.isArray ? desc.arraySize : 1u);
		uint32_t mips = 1;
		if (!desc.imageDimensions.empty()) {
			if (slices > 0 && desc.imageDimensions.size() > slices && desc.imageDimensions.size() % slices == 0)
				mips = static_cast<uint32_t>(desc.imageDimensions.size() / slices);
			else if (desc.generateMipMaps)
				mips = static_cast<uint32_t>(std::floor(std::log2((std::max)(desc.imageDimensions[0].width, desc.imageDimensions[0].height)))) + 1;
		}
		return {(std::max)(1u, mips), (std::max)(1u, slices), resource.HasLayout(), resource.CommonLayoutOnly()};
	}
	return {(std::max)(1u, resource.GetMipLevels()), (std::max)(1u, resource.GetArraySize()), resource.HasLayout(), resource.CommonLayoutOnly()};
}

CompileResourceState LowerState(const ResourceState& state) {
	return {static_cast<uint64_t>(state.access), static_cast<uint64_t>(state.layout),
		static_cast<uint64_t>(state.sync), rhi::AccessTypeIsWriteType(state.access)};
}

std::optional<CompileRange> LowerRange(const RangeSpec& spec, CompileResourceShape shape) {
	const auto resolved = ResolveRangeSpec(spec, shape.mips, shape.slices);
	if (resolved.isEmpty()) return std::nullopt;
	return CompileRange{resolved.firstMip, resolved.mipCount, resolved.firstSlice, resolved.sliceCount};
}

std::shared_ptr<const std::vector<PreparedStateRegion>> CaptureRegions(Resource& resource, CompileResourceShape shape) {
	auto regions = std::make_shared<std::vector<PreparedStateRegion>>();
	if (auto* tracker = resource.GetStateTracker()) {
		for (const auto& segment : tracker->GetSegments()) {
			const auto range = LowerRange(segment.rangeSpec, shape);
			if (!range) continue;
			regions->push_back({*range, LowerState(segment.state)});
		}
	}
	if (regions->empty() && shape.mips && shape.slices)
		regions->push_back({{0, shape.mips, 0, shape.slices},
			{static_cast<uint64_t>(rhi::ResourceAccessType::None), static_cast<uint64_t>(rhi::ResourceLayout::Undefined),
				static_cast<uint64_t>(rhi::ResourceSyncState::None), false}});
	return regions;
}

// Exact binding for one slot from the live resource. Null when the resource has
// no backing yet (unmaterialized) — the slot stays reserved for that frame.
std::optional<persistent::BindingVersion> CaptureSlotBinding(Resource& resource, uint32_t slotIndex, CompileResourceShape shape,
	PublicationBindingBundle::Snapshot snapshot = nullptr) {
	if (!snapshot) { BT_ZONE_SCOPE("ORG.Persistent.CaptureSlotBinding.Snapshot"); snapshot = PublicationBindingBundle::Capture(resource); }
	if (!snapshot || !snapshot->resource.GetHandle().valid()) return std::nullopt;
	auto* concrete = UnwrapDynamic(&resource);
	if (!concrete) return std::nullopt;
	PreparedBackingState initial;
	initial.graphResourceID = uint64_t{slotIndex} + 1;
	initial.resource = snapshot->resource.GetHandle();
	initial.shape = shape;
	initial.regions = CaptureRegions(*concrete, shape);
	rhi::ResourceDesc description{};
	if (concrete->TryGetRHIResourceDesc(description)) initial.heapType = description.heapType;
	initial.aliasHeap = snapshot->aliasHeap;
	initial.aliasHeapIdentity = snapshot->aliasHeap.get();
	initial.aliasPoolID = snapshot->aliasPoolID;
	initial.aliasOffset = snapshot->aliasOffset;
	initial.aliasSize = snapshot->aliasSize;
	return persistent::BindingVersion::FromSnapshot(std::move(snapshot), initial,
		reinterpret_cast<uintptr_t>(initial.regions.get()), initial.resource.generation);
}

struct RotationKey {
	uint64_t concreteID = 0, backingGeneration = 0;
	const void* views = nullptr;
	uint32_t apiIndex = 0, apiGeneration = 0;
	bool operator==(const RotationKey&) const = default;
};
RotationKey RotationKeyOfResource(Resource& resource) {
	RotationKey key;
	auto* concrete = UnwrapDynamic(&resource);
	if (!concrete) return key;
	key.concreteID = concrete->GetGlobalResourceID();
	if (auto* backed = dynamic_cast<BackedResource*>(concrete)) key.backingGeneration = backed->GetBackingGeneration();
	else {
		const auto handle = concrete->GetAPIResource().GetHandle();
		key.apiIndex = handle.index; key.apiGeneration = handle.generation;
	}
	if (auto* indexed = dynamic_cast<GloballyIndexedResource*>(concrete)) key.views = indexed->CaptureBindlessViews().get();
	return key;
}
} // namespace

// ---------------------------------------------------------------------------

struct RenderGraph::PersistentExecutionState {
	struct SlotEntry {
		persistent::ResourceSlotId slot;
		Resource* resource = nullptr;      // Registry object (wrapper for dynamic resources).
		uint64_t resourceID = 0;           // Scheduling identity.
		CompileResourceShape shape;
		RotationKey bound;                 // Rotation state of the currently bound backing.
		// Cached classification of the registry object (polled every frame).
		Resource* classified = nullptr;    // resource this classification was made for
		Resource* concrete = nullptr;      // last unwrapped backing
		BackedResource* backed = nullptr;
		GloballyIndexedResource* indexed = nullptr;
		bool wrapper = false;
		bool isBound = false;
		bool direct = false;               // Shared by direct declarations; never a group member.
		bool swapchain = false;            // Rebound per frame at admission.
		bool retired = false;              // Slot removed from the program; the index may be reclaimed.
		std::vector<uint32_t> subscribers; // Main pass indices resolving this slot.
	};
	// One persistent group per resolver identity, shared by every subscribing
	// pass. Membership is polled once per group, not once per pass.
	struct ResolverGroup {
		persistent::ResourceGroupId group;
		std::shared_ptr<const void> key;          // Resolver dependency identity.
		std::unique_ptr<IResourceResolver> resolver;
		CompileResourceShape shape;
		uint32_t capacity = 0;
		std::vector<uint32_t> memberEntries;      // Reserved member slots (entry indices).
		std::vector<uint64_t> memberResourceIDs;  // Bound scheduling ID per member slot, 0 = free.
		std::unordered_map<uint64_t, uint32_t> memberIndexByID; // Scheduling ID -> member position.
		std::vector<uint32_t> freeMembers;        // Member positions with no bound resource.
		ResolverResourceSetIdentity identity{};
		uint64_t waitRevision = 0;
		uint64_t versionHint = 0;                 // Last IResourceResolver::DeclarationVersionHint seen.
		std::vector<ExternalTimelinePoint> waits;
		std::vector<uint32_t> subscribers;        // Main pass indices.
	};
	struct MainPass {
		size_t masterIndex = 0;
		persistent::PassId id;
		std::string name;
		std::vector<uint32_t> directEntries;  // Entry indices of direct declarations.
		std::vector<uint32_t> groups;         // Indices into PersistentExecutionState::groups.
		std::shared_ptr<const FramePreparationContext::ResourceSlots> slots;
		const void* slotsExecutable = nullptr;
		bool slotsDirty = true;
		// Pending incremental slot-map patches (membership and rotation edits);
		// applied copy-on-write before preparation instead of a full rebuild.
		std::vector<std::pair<uint64_t, uint32_t>> slotAdds, slotRemoves;
		uint64_t fingerprint = 0; // Structural identity of the lowered declaration (membership excluded).
		std::vector<std::pair<uint64_t, std::string>> loweredDirect; // Diagnostic: (scheduling ID, name).
		std::vector<const void*> loweredGroupKeys;
		std::vector<ExternalTimelinePoint> explicitWaits;
		// Frame-interrupting per-frame pass hosted in the main executable (not in
		// the master list). It is prepared once and records nothing afterwards
		// until its removal is installed.
		std::unique_ptr<AnyPassAndResources> hosted;
		bool hostedPrepared = false;
		bool retired = false; // Hosted pass removed; the main index may be reused.
	};
	struct PreparedSegment {
		const char* label = "";
		std::shared_ptr<const persistent::SelectedPublication> publication;
		std::vector<PreparedPass> invocations;
		std::shared_ptr<const std::vector<std::string>> names; // Per logical pass slot, for statistics/present detection.
		std::vector<persistent::FrameRebinding> rebindings;
		std::vector<persistent::FrameProducerWait> waits;
		std::shared_ptr<const FrozenExecutionBindings> legacyBindings;
		std::vector<experimental::PreparedTimelineBinding> foreignTimelines;
		std::vector<std::shared_ptr<const void>> leases;
	};

	// A dynamic segment whose lowered structure and bound backings match the
	// previous frame's reuses that publication: no binding capture, no build.
	struct SegmentCache {
		std::vector<uint64_t> key;
		std::shared_ptr<const persistent::SelectedPublication> publication;
		std::shared_ptr<const FrozenExecutionBindings> legacyBindings;
		std::vector<persistent::PassId> ids;
	};
	SegmentCache preCache, tailCache;
	// Persistent program for a dynamic segment: the passes are lowered once per
	// structure; each frame's resource set is bound into reserved-capacity
	// groups (one per pass and access template) by diff. A frame whose set and
	// backings repeat does no edit and no build. Capacity overflow and
	// structure changes rebuild the program (rare); postconditions, swapchain
	// images and a backing wanted by two pools fall back to a fresh build.
	struct SegmentProgram {
		struct Template {
			CompileResourceShape shape; CompileRange range; CompileResourceState state;
			bool operator==(const Template&) const = default;
		};
		struct Pool {
			uint32_t pass = 0;
			uint64_t templateHash = 0;
			Template request;
			persistent::ResourceGroupId group;
			uint32_t capacity = 0;
			std::vector<persistent::ResourceSlotId> slots;    // member slots (positions)
			std::vector<uint64_t> physical;                   // wanted physical identity per position, 0 = none
			std::vector<Resource*> resources;
			std::vector<RotationKey> bound;
			std::vector<uint8_t> isBound;                     // position holds a binding (kept until refilled or unbound)
			std::unordered_map<uint64_t, uint32_t> positionByPhysical;
			std::vector<uint32_t> free, vacated;
			std::vector<uint8_t> wanted;                      // per-frame scratch
		};
		struct Pass { persistent::PassId id; std::vector<uint32_t> pools; };
		std::vector<uint64_t> key;                            // structure: pass names, declarations, templates
		persistent::GraphProgram program;
		std::vector<Pool> pools;
		std::vector<Pass> passes;
		std::vector<FrozenExecutionBindings::ResourceBinding> table; // per slot
		std::shared_ptr<const persistent::SelectedPublication> publication;
		std::shared_ptr<const FrozenExecutionBindings> legacyBindings;
		uint64_t frames = 0;
	};
	std::vector<std::unique_ptr<SegmentProgram>> prePrograms, tailPrograms;

	persistent::GraphProgram program;
	std::unique_ptr<persistent::SynchronousAdmission> admission;
	std::unique_ptr<experimental::ExecutionTimelineAdmission> timelines;
	std::vector<SlotEntry> entries;                                // Index == slot index.
	std::unordered_map<uint64_t, std::vector<uint32_t>> entriesByResourceID;
	std::vector<MainPass> mainPasses;
	std::vector<ResolverGroup> groups;
	std::vector<size_t> preMasterIndices, tailMasterIndices;
	std::vector<uint32_t> mainPassByMaster;                        // UINT32_MAX when not main.
	std::vector<uint8_t> masterHostedInMain;
	// Per-publication legacy binding table cache.
	const persistent::SelectedPublication* bindingsFor = nullptr;
	std::shared_ptr<const FrozenExecutionBindings> bindings;
	// Frame in preparation.
	std::optional<PreparedSegment> pre, main, tail;
	std::vector<AnyPassAndResources> frameExtensionPasses;
	std::vector<uint32_t> hostedMainPasses;             // Main indices of installed frame-interrupting passes.
	std::vector<ExternalPassDesc> deferredInterrupting; // Requested while a structural build was pending.
	uint64_t frameNumber = 0;
	uint8_t frameIndex = 0;
	std::shared_ptr<const IHostExecutionData> frameData;
	// Alias planning (structural builds only).
	struct AliasPool {
		TrackedHandle allocation;
		std::shared_ptr<const AliasHeapGeneration> heap;
		uint64_t capacityBytes = 0, alignment = 1, generation = 0;
		uint8_t resourceClass = 0;
	};
	struct AliasPlacementRecord {
		uint64_t poolID = 0, offset = 0, sizeBytes = 0;
		const AliasHeapGeneration* heap = nullptr;
		bool operator==(const AliasPlacementRecord&) const = default;
	};
	std::unordered_map<uint64_t, AliasPool> aliasPools;
	std::unordered_map<Resource*, AliasPlacementRecord> aliasPlaced; // Concrete resources placed by this planner.
	std::vector<std::pair<uint32_t, uint32_t>> aliasEdges;          // Installed placement orderings (logical pass slots).
	uint64_t aliasPlans = 0, aliasRematerializations = 0;
	bool forceNewAliasHeaps = false; // Diagnostic (ORG_PERSISTENT_FORCE_REALIAS): next plan re-places every pool.
	const void* dumpedExecutable = nullptr; // Compile-dump mode: last executable written.
	// Diagnostic: per-pass preparation cost (logged periodically).
	std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> prepareCostByPass; // lowering: name -> (ns, calls)
	std::vector<std::pair<uint64_t, uint64_t>> prepareCostByMain;                    // main pass index -> (ns, calls)
	std::shared_ptr<const std::vector<std::string>> mainNames;
	const void* mainNamesExecutable = nullptr;
	// A structural build in flight on a worker. Frames keep using the selected
	// publication; the slot directory already describes the new structure, so
	// subscriber slot maps are frozen (no rebuild) until the install.
	struct PendingStructural {
		std::unique_ptr<persistent::GraphEditTransaction> edit;
		std::shared_ptr<const persistent::SelectedPublication> base;
		uint32_t baseSlotCount = 0;
		std::atomic<int> phase{0}; // 0 scheduling, 1 scheduled, 2 placing, 3 placed, -1 failed
		std::atomic_bool cancelled{false};
		std::atomic_bool clearedOrderings{false};
		std::shared_ptr<const persistent::SelectedPublication> result;
		std::string error;
		experimental::CompileWorkspace workspace;
		std::vector<uint32_t> dirtyPasses;        // slotsDirty deferred to the install
		std::vector<uint32_t> touchedGroups;      // grown in the transaction: not polled meanwhile
		std::vector<uint32_t> deferredStructural; // relower requests that arrived meanwhile
		// Binding edits installed on the live publication meanwhile, replayed
		// onto the result (nullopt = unbind). Keyed by slot index.
		std::unordered_map<uint32_t, std::optional<persistent::BindingVersion>> bindingLog;
		std::chrono::steady_clock::time_point started;
	};
	std::shared_ptr<PendingStructural> pending;
	// Submitted this frame.
	uint64_t pendingPresentSubmission = 0;
	uint64_t structuralBuilds = 0, bindingBuilds = 0;
	std::vector<uint8_t> tracyFrameBegun;
};

namespace {

using State = RenderGraph::PersistentExecutionState;

// Rotation state of an entry's current backing. Type classification is cached
// per registry object; only dynamic wrappers are unwrapped every frame.
RotationKey RotationKeyOf(State::SlotEntry& entry) {
	RotationKey key;
	if (!entry.resource) return key;
	if (entry.classified != entry.resource) {
		entry.classified = entry.resource;
		entry.wrapper = dynamic_cast<DynamicResource*>(entry.resource) || dynamic_cast<DynamicGloballyIndexedResource*>(entry.resource);
		entry.concrete = nullptr;
	}
	auto* concrete = entry.wrapper ? UnwrapDynamic(entry.resource) : entry.resource;
	if (!concrete) return key;
	if (concrete != entry.concrete) {
		entry.concrete = concrete;
		entry.backed = dynamic_cast<BackedResource*>(concrete);
		entry.indexed = dynamic_cast<GloballyIndexedResource*>(concrete);
	}
	key.concreteID = concrete->GetGlobalResourceID();
	if (entry.backed) key.backingGeneration = entry.backed->GetBackingGeneration();
	else {
		const auto handle = concrete->GetAPIResource().GetHandle();
		key.apiIndex = handle.index; key.apiGeneration = handle.generation;
	}
	if (entry.indexed) key.views = entry.indexed->CaptureBindlessViews().get();
	return key;
}

std::vector<experimental::CompileQueue> RegistryQueues(const QueueRegistry& registry) {
	std::vector<experimental::CompileQueue> queues;
	for (size_t slot = 0; slot < registry.SlotCount(); ++slot)
		queues.push_back({static_cast<uint32_t>(registry.GetBackendInstance(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot)))), true});
	return queues;
}

} // namespace

// ---------------------------------------------------------------------------
// Declaration lowering (legacy adapter)

namespace {

bool IsSwapchainResource(Resource& resource) {
	return resource.GetName() == "Backbuffer" && dynamic_cast<DynamicResource*>(&resource)
		&& !dynamic_cast<BackedResource*>(UnwrapDynamic(&resource));
}
uint64_t FingerprintDeclaration(const experimental::CompilePass& d) {
	uint64_t h = 0x9e3779b97f4a7c15ull;
	auto mix = [&](uint64_t value) { h ^= value + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
	mix(d.backend); mix(d.preferredQueueSlot); mix(d.forceBatchIsolation ? 1 : 0);
	auto slots = d.compatibleQueueSlots; std::sort(slots.begin(), slots.end());
	for (const auto slot : slots) mix(0x100 + slot);
	return h;
}
uint64_t HashSegmentTemplate(const State::SegmentProgram::Template& t) {
	uint64_t h = 0x7f4a7c159e3779b9ull;
	auto mix = [&](uint64_t value) { h ^= value + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
	mix(t.shape.mips); mix(t.shape.slices); mix(t.shape.hasLayout ? 1 : 0);
	mix(t.range.mip); mix(t.range.mips); mix(t.range.slice); mix(t.range.slices);
	mix(t.state.access); mix(t.state.layout); mix(t.state.sync); mix(t.state.write ? 1 : 0);
	return h ? h : 1;
}

struct LoweredUse {
	uint64_t resourceID = 0;         // Scheduling identity.
	Resource* resource = nullptr;
	CompileRange range;
	CompileResourceState state;
};
struct LoweredGroup {
	std::shared_ptr<const void> key;
	std::unique_ptr<IResourceResolver> resolver;
	std::vector<std::pair<uint64_t, Resource*>> members;
	std::vector<std::pair<CompileRange, CompileResourceState>> templates;
	CompileResourceShape shape;
	ResolverResourceSetIdentity identity{};
	uint64_t waitRevision = 0;
	std::vector<ExternalTimelinePoint> waits;
	bool uniform = true;
};
struct LoweredPass {
	experimental::CompilePass declaration;
	std::vector<LoweredUse> entries, exits;
	std::vector<LoweredGroup> groups;
	std::vector<ExternalTimelinePoint> explicitWaits;
};

// Structural identity of a lowering. Group membership is deliberately absent:
// it is a binding edit. A pass whose refreshed declaration hashes equal needs no
// executable rebuild even though it reported DeclaredResourcesChanged().
// Order-independent: a refreshed declaration may enumerate the same
// requirements in a different order (resolver blocks are re-interned).
uint64_t FingerprintLowering(const LoweredPass& lowered) {
	auto hashUse = [](const LoweredUse& use) {
		uint64_t h = 0x9e3779b97f4a7c15ull;
		auto mix = [&](uint64_t value) { h ^= value + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
		mix(use.resourceID); mix(use.range.mip); mix(use.range.mips); mix(use.range.slice); mix(use.range.slices);
		mix(use.state.access); mix(use.state.layout); mix(use.state.sync); mix(use.state.write);
		return h;
	};
	auto hashTemplate = [](const std::pair<CompileRange, CompileResourceState>& request) {
		uint64_t h = 0x7f4a7c159e3779b9ull;
		auto mix = [&](uint64_t value) { h ^= value + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
		const auto& [range, state] = request;
		mix(range.mip); mix(range.mips); mix(range.slice); mix(range.slices);
		mix(state.access); mix(state.layout); mix(state.sync); mix(state.write);
		return h;
	};
	std::vector<uint64_t> parts;
	const auto& d = lowered.declaration;
	parts.push_back(d.backend); parts.push_back(d.preferredQueueSlot); parts.push_back(d.forceBatchIsolation ? 1 : 0);
	{
		auto slots = d.compatibleQueueSlots; std::sort(slots.begin(), slots.end());
		for (auto slot : slots) parts.push_back(0x100 + slot);
	}
	std::vector<uint64_t> entries, exits, groups;
	for (const auto& use : lowered.entries) entries.push_back(hashUse(use));
	for (const auto& use : lowered.exits) exits.push_back(hashUse(use));
	for (const auto& group : lowered.groups) {
		std::vector<uint64_t> templates;
		for (const auto& request : group.templates) templates.push_back(hashTemplate(request));
		std::sort(templates.begin(), templates.end());
		uint64_t h = reinterpret_cast<uintptr_t>(group.key.get());
		h ^= (uint64_t{group.shape.mips} << 40) ^ (uint64_t{group.shape.slices} << 20) ^ (group.shape.hasLayout ? 1 : 0);
		for (auto t : templates) h ^= t + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
		groups.push_back(h);
	}
	std::sort(entries.begin(), entries.end()); std::sort(exits.begin(), exits.end()); std::sort(groups.begin(), groups.end());
	entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
	uint64_t h = 0xcbf29ce484222325ull;
	auto fold = [&](uint64_t value) { h ^= value; h *= 0x100000001b3ull; h ^= h >> 29; };
	for (auto v : parts) fold(v);
	fold(0xE1); for (auto v : entries) fold(v);
	fold(0xE2); for (auto v : exits) fold(v);
	fold(0xE3); for (auto v : groups) fold(v);
	return h;
}

} // namespace

// Lowers one pass's current declaration state. Resolver snapshots become
// reserved-capacity groups only when the pass keeps them separately (patchable);
// otherwise their members are already merged into the static requirements.
static LoweredPass LowerLegacyPass(RenderGraph& graph, ResourceRegistry& registry, const QueueRegistry& queues,
	rhi::Backend primaryBackend, const RenderGraph::PassAndResources& pass, RenderGraph::PassType type, uint32_t phase) {
	LoweredPass result;
	auto& declaration = result.declaration;
	// Queue compatibility mirrors BuildNodes for the retained path.
	const auto& resources = pass.resources;
	std::vector<uint32_t> compatible;
	if (resources.pinnedQueueSlot) compatible.push_back(static_cast<uint32_t>(static_cast<uint8_t>(*resources.pinnedQueueSlot)));
	else {
		auto kindCompatible = [&](QueueKind kind) {
			switch (type) {
			case RenderGraph::PassType::Render: return kind == QueueKind::Graphics;
			case RenderGraph::PassType::Compute: return kind == QueueKind::Graphics || kind == QueueKind::Compute;
			default: return true;
			}
		};
		for (size_t slot = 0; slot < queues.SlotCount(); ++slot) {
			const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot));
			if (!queues.IsAutoAssignable(index)) continue;
			const auto kind = queues.GetKind(index);
			const bool automatic = resources.queueAssignmentPolicy == QueueAssignmentPolicy::Automatic && kindCompatible(kind);
			if (automatic || kind == resources.preferredQueueKind) compatible.push_back(static_cast<uint32_t>(slot));
		}
		if (compatible.empty()) compatible.push_back(static_cast<uint32_t>(resources.preferredQueueKind));
	}
	const bool d3d12OwnedQueueTransfers = queues.SlotCount() != 0
		&& queues.GetBackend(static_cast<QueueSlotIndex>(0)) == rhi::Backend::D3D12;
	if (!d3d12OwnedQueueTransfers) compatible.assign(1, 0u);
	declaration.compatibleQueueSlots = compatible;
	declaration.preferredQueueSlot = compatible.front();
	declaration.backend = static_cast<uint32_t>(resources.backendAffinity.strength == BackendAffinityStrength::Primary
		? primaryBackend : resources.backendAffinity.backend);
	declaration.forceBatchIsolation = !resources.externalWaitsBeforeTransitions.empty()
		|| !resources.externalWaitBindingsBeforeTransitions.empty();
	result.explicitWaits = resources.externalWaitsBeforeTransitions;

	auto resolve = [&](const ResourceRegistry::RegistryHandle& handle) -> Resource* {
		return handle.IsEphemeral() ? handle.GetEphemeralPtr() : registry.Resolve(handle);
	};
	auto lowerRequirement = [&](const ResourceRequirement& requirement) {
		auto* resource = resolve(requirement.resourceHandleAndRange.resource);
		if (!resource) throw std::runtime_error("Persistent lowering: pass '" + pass.name + "' declares an unresolvable resource");
		const auto shape = ShapeOf(*resource);
		const auto range = LowerRange(requirement.resourceHandleAndRange.range, shape);
		if (!range) return;
		result.entries.push_back({resource->GetSchedulingResourceID(), resource, *range, LowerState(requirement.state)});
	};
	const bool grouped = pass.declarationCache.incrementalResolverPatchable
		&& !resources.resolverRequirementBlocks.empty()
		&& resources.resolverRequirementBlocks.size() == pass.resolverSnapshots.size();
	if (grouped) {
		for (const auto& requirement : resources.staticResourceRequirements) lowerRequirement(requirement);
		for (size_t i = 0; i < pass.resolverSnapshots.size(); ++i) {
			const auto& snapshot = pass.resolverSnapshots[i];
			LoweredGroup group;
			group.key = snapshot.dependencyIdentity;
			group.resolver = snapshot.resolver ? snapshot.resolver->Clone() : nullptr;
			group.identity = snapshot.resourceSetIdentity;
			group.waitRevision = snapshot.waitRevision;
			group.waits = snapshot.waits;
			const auto state = graph.CaptureResolverDeclarationState(*snapshot.resolver);
			std::optional<CompileResourceShape> shape;
			if (state && state->resources) for (const auto& member : *state->resources) {
				if (!member) continue;
				const auto memberShape = ShapeOf(*member);
				if (!shape) shape = memberShape;
				else if (*shape != memberShape) group.uniform = false;
				group.members.emplace_back(member->GetSchedulingResourceID(), member.get());
			}
			group.shape = shape.value_or(CompileResourceShape{1, 1, false});
			for (const auto& request : snapshot.requirementTemplates) {
				const auto range = LowerRange(request.range, group.shape);
				if (!range) { group.uniform = false; continue; }
				group.templates.emplace_back(*range, LowerState(request.state));
			}
			if (group.uniform && group.key && group.resolver && !group.templates.empty()) { result.groups.push_back(std::move(group)); continue; }
			// Mixed shapes: fall back to direct declarations for this resolver.
			for (const auto& [id, member] : group.members)
				for (const auto& request : snapshot.requirementTemplates) {
					const auto memberShape = ShapeOf(*member);
					const auto range = LowerRange(request.range, memberShape);
					if (range) result.entries.push_back({id, member, *range, LowerState(request.state)});
				}
		}
	} else {
		for (const auto& requirement : GetFrameRequirementsSpan(resources)) lowerRequirement(requirement);
	}
	for (const auto& [handleAndRange, state] : resources.internalTransitions) {
		auto* resource = resolve(handleAndRange.resource);
		if (!resource) throw std::runtime_error("Persistent lowering: pass '" + pass.name + "' transitions an unresolvable resource");
		const auto shape = ShapeOf(*resource);
		const auto range = LowerRange(handleAndRange.range, shape);
		if (!range) continue;
		result.exits.push_back({resource->GetSchedulingResourceID(), resource, *range, LowerState(state)});
	}
	(void)phase;
	return result;
}

// ---------------------------------------------------------------------------
// Program editing helpers (owner thread)

namespace {

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
	CompileResourceShape shape, bool swapchain) {
	auto& indices = state.entriesByResourceID[resourceID];
	for (const auto index : indices) {
		const auto& entry = state.entries[index];
		if (entry.direct && entry.resource == resource && entry.shape == shape) return index;
	}
	const auto slot = edit.ReserveResource(shape);
	auto& entry = ClaimEntry(state, slot);
	entry.direct = true;
	entry.resource = resource;
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
bool BindEntry(State& state, persistent::GraphEditTransaction& edit, uint32_t index) {
	auto& entry = state.entries[index];
	if (entry.swapchain || !entry.resource) return false;
	auto binding = CaptureSlotBinding(*entry.resource, entry.slot.index, entry.shape);
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

} // namespace

// Installs (or reinstalls) a main pass declaration into the transaction.
static void InstallMainPass(RenderGraph& graph, State& state, persistent::GraphEditTransaction& edit,
	State::MainPass& main, LoweredPass lowered, uint32_t mainIndex, bool replace, std::optional<uint32_t> authoredOrder = std::nullopt) {
	if (replace) {
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
	main.fingerprint = FingerprintLowering(lowered);
	main.loweredDirect.clear();
	for (const auto& use : lowered.entries) main.loweredDirect.emplace_back(use.resourceID, use.resource ? use.resource->GetName() : std::string{});
	main.loweredGroupKeys.clear();
	for (const auto& group : lowered.groups) main.loweredGroupKeys.push_back(group.key.get());
	main.explicitWaits = std::move(lowered.explicitWaits);
	main.slotsDirty = true;
	std::unordered_map<uint64_t, persistent::BindingToken> tokens;
	auto declare = [&](const LoweredUse& use, bool entry) {
		const bool swapchain = use.resource && use.resource->GetName() == "Backbuffer"
			&& dynamic_cast<DynamicResource*>(use.resource) && !dynamic_cast<BackedResource*>(UnwrapDynamic(use.resource));
		const auto index = DirectEntry(state, edit, use.resourceID, use.resource, ShapeOf(*use.resource), swapchain);
		auto& slotEntry = state.entries[index];
		if (std::find(slotEntry.subscribers.begin(), slotEntry.subscribers.end(), mainIndex) == slotEntry.subscribers.end())
			slotEntry.subscribers.push_back(mainIndex);
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
		if (groupIndex == UINT32_MAX) {
			State::ResolverGroup group;
			group.key = loweredGroup.key;
			group.resolver = std::move(loweredGroup.resolver);
			group.shape = loweredGroup.shape;
			group.identity = loweredGroup.identity;
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
				group.memberEntries.push_back(slot.index);
				group.memberResourceIDs.push_back(0);
			}
			for (uint32_t i = static_cast<uint32_t>(group.memberEntries.size()); i-- > loweredGroup.members.size();)
				group.freeMembers.push_back(i);
			for (size_t i = 0; i < loweredGroup.members.size(); ++i) {
				const auto [id, resource] = loweredGroup.members[i];
				auto& entry = state.entries[group.memberEntries[i]];
				entry.resource = resource;
				entry.resourceID = id;
				state.entriesByResourceID[id].push_back(group.memberEntries[i]);
				group.memberResourceIDs[i] = id;
				group.memberIndexByID.emplace(id, static_cast<uint32_t>(i));
				// Unmaterialized members are realized and bound by
				// BindUnboundPersistentEntries before the structural build.
				auto* concrete = UnwrapDynamic(resource);
				auto* backed = concrete ? dynamic_cast<BackedResource*>(concrete) : nullptr;
				if (!backed || backed->IsMaterialized()) BindEntry(state, edit, group.memberEntries[i]);
			}
			groupIndex = static_cast<uint32_t>(state.groups.size());
			state.groups.push_back(std::move(group));
		}
		auto& group = state.groups[groupIndex];
		group.subscribers.push_back(mainIndex);
		for (const auto entryIndex : group.memberEntries) state.entries[entryIndex].subscribers.push_back(mainIndex);
		for (const auto& [range, stateValue] : loweredGroup.templates)
			edit.DeclareGroupAccess(main.id, group.group, stateValue, mainIndex, range);
		main.groups.push_back(groupIndex);
	}
	(void)graph;
}

// ---------------------------------------------------------------------------
// Diagnostic: forced alias re-placement
//
// ORG_PERSISTENT_FORCE_REALIAS realizes every alias pool on a fresh heap, which
// rematerializes every placed resource and rotates its descriptor slots: the
// event a pool growth causes once at startup, made repeatable so consumers of
// descriptor indices can be tested on the exact frame it happens.
//   interval:N  every N frames (a structural build each time)
//   captures    in the build that hosts frame-interrupting passes, so a
//               mid-frame readback observes the rotation frame itself

namespace {
struct ForcedRealiasConfig { uint64_t interval = 0; bool withCaptures = false; };
const ForcedRealiasConfig& ForcedRealias() {
	static const ForcedRealiasConfig config = [] {
		ForcedRealiasConfig result;
		const char* value = std::getenv("ORG_PERSISTENT_FORCE_REALIAS");
		if (!value) return result;
		const std::string text(value);
		if (text == "captures") result.withCaptures = true;
		else if (text.rfind("interval:", 0) == 0) result.interval = std::strtoull(text.c_str() + 9, nullptr, 10);
		if (result.interval || result.withCaptures) spdlog::warn("Persistent graph: forced alias re-placement enabled ({})", text);
		return result;
	}();
	return config;
}
} // namespace

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
			});
		// Classify master passes into segments.
		state.mainPassByMaster.assign(m_masterPassList.size(), UINT32_MAX);
		state.masterHostedInMain.assign(m_masterPassList.size(), 0);
		std::vector<Resource*> usedResources;
		for (size_t masterIndex = 0; masterIndex < m_masterPassList.size(); ++masterIndex) {
			auto& any = m_masterPassList[masterIndex];
			auto kind = PersistentSegmentKind::Main;
			if (const auto found = m_persistentSegmentKinds.find(any.name); found != m_persistentSegmentKinds.end()) kind = found->second;
			std::visit([&](auto& value) {
				using T = std::decay_t<decltype(value)>;
				if constexpr (!std::is_same_v<T, std::monostate>) {
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
		edit.SetQueues(RegistryQueues(m_queueRegistry));
		for (uint32_t mainIndex = 0; mainIndex < state.mainPasses.size(); ++mainIndex) {
			auto& main = state.mainPasses[mainIndex];
			auto& any = m_masterPassList[main.masterIndex];
			std::visit([&](auto& value) {
				using T = std::decay_t<decltype(value)>;
				if constexpr (!std::is_same_v<T, std::monostate>)
					InstallMainPass(*this, state, edit, main,
						LowerLegacyPass(*this, _registry, m_queueRegistry, primaryBackend, value, any.type, mainIndex), mainIndex, false);
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
				for (const auto id : state.groups[g].memberResourceIDs) if (id) groupsByResource[id].push_back(g);
			std::map<std::vector<uint32_t>, uint32_t> sharedSets;
			for (const auto& [id, groups] : groupsByResource) if (groups.size() > 1) ++sharedSets[groups];
			for (const auto& [groups, count] : sharedSets) {
				std::string names;
				for (const auto g : groups) {
					const auto& group = state.groups[g];
					names += (names.empty() ? "" : " | ") + std::string(group.resolver ? typeid(*group.resolver).name() : "?")
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
					if (std::find(state.groups[g].memberEntries.begin(), state.groups[g].memberEntries.end(), index) != state.groups[g].memberEntries.end())
						groupName = std::to_string(g) + ":" + (state.groups[g].resolver ? typeid(*state.groups[g].resolver).name() : "?") + "(members=" + std::to_string(state.groups[g].memberIndexByID.size()) + ")";
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
	compiler.preparingFrame.reset();
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
		if (const auto interval = ForcedRealias().interval; interval && !state.pending && state.frameNumber % interval == 0) {
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
			spdlog::info("Persistent forced alias re-placement: frame={}", state.frameNumber);
		}
		auto ensureEdit = [&]() -> persistent::GraphEditTransaction& {
			if (!edit) edit.emplace(state.program.BeginEdit());
			return *edit;
		};
		size_t refreshed = 0;
		{ BT_ZONE_SCOPE("ORG.Persistent.PollDynamicDeclared");
		for (uint32_t mainIndex = 0; mainIndex < state.mainPasses.size(); ++mainIndex) {
			auto& main = state.mainPasses[mainIndex];
			if (main.hosted || main.retired) continue; // One-shot; never relowered.
			auto& any = m_masterPassList[main.masterIndex];
			std::visit([&](auto& value) {
				using T = std::decay_t<decltype(value)>;
				if constexpr (!std::is_same_v<T, std::monostate>) {
					auto* dynamic = value.declarationCache.dynamicInterface;
					if (!dynamic || !dynamic->DeclaredResourcesChanged() || dynamic->DeclarationsProvidedByImmediateCommands()) return;
					BT_ZONE_SCOPE("ORG.Persistent.RefreshDeclaration");
					BT_ZONE_TEXT(any.name.data(), any.name.size());
					++refreshed;
					if (!RefreshRetainedDeclarationsForFrame(value, frameIndex)) return;
					auto lowered = LowerLegacyPass(*this, _registry, m_queueRegistry, primaryBackend, value, any.type, mainIndex);
					if (FingerprintLowering(lowered) == main.fingerprint) return; // Membership-only change.
					LogRelowerDiff(any.name, main, lowered);
					if (state.pending) { state.pending->deferredStructural.push_back(mainIndex); return; }
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
			std::vector<std::pair<uint64_t, Resource*>> members;
			bool uniform = true;
			if (captured->resources) for (const auto& member : *captured->resources) {
				if (!member) continue;
				const auto id = member->GetSchedulingResourceID();
				// Shapes of already-bound members were checked when they joined.
				if (!group.memberIndexByID.contains(id) && ShapeOf(*member) != group.shape) uniform = false;
				members.emplace_back(id, member.get());
			}
			if (!uniform) {
				// Shape contract broken: subscribers re-lower (direct declarations).
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
			// Diff against the bound set: O(members) with the per-group index.
			std::unordered_set<uint64_t> wanted;
			wanted.reserve(members.size());
			for (const auto& [id, resource] : members) wanted.insert(id);
			// Members that left: their positions stay bound for now so that a
			// member arriving in the same poll (a rotated publication of the same
			// logical resource) replaces the binding in place. Positions are
			// reused lowest-first in resolver order so a resource set that is
			// republished every frame lands on the same member slots each time;
			// a physical backing shared with another group then keeps its
			// equivalence class (a position swap would be a structural change).
			std::vector<uint32_t> vacated;
			for (auto it = group.memberIndexByID.begin(); it != group.memberIndexByID.end();) {
				const auto [id, i] = *it;
				if (wanted.contains(id)) { ++it; continue; }
				std::erase(state.entriesByResourceID[id], group.memberEntries[i]);
				group.memberResourceIDs[i] = 0;
				vacated.push_back(i);
				it = group.memberIndexByID.erase(it);
			}
			std::sort(vacated.begin(), vacated.end(), std::greater<>());
			std::sort(group.freeMembers.begin(), group.freeMembers.end(), std::greater<>());
			for (const auto& [id, resource] : members) {
				if (group.memberIndexByID.contains(id)) continue;
				uint32_t i;
				if (!vacated.empty()) { i = vacated.back(); vacated.pop_back(); }
				else if (!group.freeMembers.empty()) { i = group.freeMembers.back(); group.freeMembers.pop_back(); }
				else throw std::logic_error("Persistent group capacity accounting mismatch");
				auto& entry = state.entries[group.memberEntries[i]];
				const bool replacing = entry.isBound;
				const std::array<uint64_t, 3> previousIDs = replacing && entry.resource
					? std::array<uint64_t, 3>{entry.resourceID, entry.resource->GetGlobalResourceID(), entry.bound.concreteID} : std::array<uint64_t, 3>{};
				entry.resource = resource; entry.resourceID = id;
				state.entriesByResourceID[id].push_back(group.memberEntries[i]);
				group.memberResourceIDs[i] = id;
				group.memberIndexByID.emplace(id, i);
				{ BT_ZONE_SCOPE("ORG.Persistent.PollGroups.Diff.Materialize"); MaterializePersistentStandalone(std::span<Resource* const>{&resource, 1}); }
				BT_ZONE_SCOPE("ORG.Persistent.PollGroups.Diff.Bind");
				if (replacing) { QueueSlotPatchIDs(state, entry, previousIDs, false); ++removed; }
				if (BindEntry(state, transaction, group.memberEntries[i])) { QueueSlotPatch(state, entry, entry.bound.concreteID, true); ++added; }
			}
			// Positions nobody refilled are unbound.
			for (const auto i : vacated) {
				auto& entry = state.entries[group.memberEntries[i]];
				if (entry.isBound) { UnbindSlot(state, transaction, entry); QueueSlotPatch(state, entry, entry.bound.concreteID, false); ++removed; }
				entry.isBound = false; entry.bound = {}; entry.resource = nullptr; entry.resourceID = 0;
				group.freeMembers.push_back(i);
			}
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
		if (!state.pending && (!structural.empty() || !growGroups.empty()))
			SubmitPersistentStructuralBuild(structural, growGroups, primaryBackend);
	}
	// 3. Select and prepare the main segment.
	auto selected = state.program.Select();
	if (m_getRenderGraphCompileDumpEnabled && m_getRenderGraphCompileDumpEnabled() && state.dumpedExecutable != selected->executable.get()) {
		state.dumpedExecutable = selected->executable.get();
		WritePersistentGraphDebugDump(*selected);
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
			if (!tasks->Submit(scope, runtime::TaskPriority::Background, "ORG.Ownership.RetireRecipe",
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
		.borrowedDependencies = true,
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
	PersistentExecutionState::PreparedSegment main;
	main.label = "Main";
	main.publication = selected;
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
		for (uint32_t mainIndex = 0; mainIndex < state.mainPasses.size(); ++mainIndex) {
			auto& pass = state.mainPasses[mainIndex];
			if (pass.retired) continue;
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
			packet.SetDebugName(any.name);
			main.invocations[pass.id.index] = std::move(packet);
			if (namesDirty) (*mainNames)[pass.id.index] = any.name;
			// External waits (explicit + resolver-provided) become per-frame producer waits.
			auto appendWait = [&](const ExternalTimelinePoint& wait) {
				if (!wait.timeline || !wait.value) return;
				const auto handle = wait.timeline.GetHandle();
				uint64_t identity = 0;
				for (const auto& binding : main.foreignTimelines)
					if (binding.handle.index == handle.index && binding.handle.generation == handle.generation) identity = binding.identity;
				for (size_t slot = 0; slot < m_queueRegistry.SlotCount() && !identity; ++slot) {
					const auto fence = m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot))).GetHandle();
					if (fence.index == handle.index && fence.generation == handle.generation) identity = slot + 1;
				}
				if (!identity) {
					identity = m_queueRegistry.SlotCount() + 1 + main.foreignTimelines.size();
					main.foreignTimelines.push_back({identity, handle});
				}
				main.waits.push_back({pass.id, {identity, wait.value}});
			};
			for (const auto& wait : pass.explicitWaits) appendWait(wait);
			for (const auto groupIndex : pass.groups) for (const auto& wait : state.groups[groupIndex].waits) appendWait(wait);
		}
	}
	state.main = std::move(main);
	// 4. Dynamic segments.
	state.pre.reset();
	state.tail.reset();
	{
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
							if (auto* dynamic = value.declarationCache.dynamicInterface; dynamic && dynamic->DeclaredResourcesChanged()) {
								BT_ZONE_SCOPE("ORG.Persistent.DynamicSegment.RefreshDeclarations");
								RefreshRetainedDeclarationsForFrame(value, frameIndex);
							}
							{ BT_ZONE_SCOPE("ORG.Persistent.DynamicSegment.LowerLegacyPass"); lowered = LowerLegacyPass(*this, _registry, m_queueRegistry, primaryBackend, value, any.type, passIndex); }
							// Groups are meaningless for one-frame programs: declare members directly.
							for (auto& group : lowered.groups)
								for (const auto& [id, resource] : group.members)
									for (const auto& [range, stateValue] : group.templates)
										lowered.entries.push_back({id, resource, range, stateValue});
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
					uint64_t identity = 0;
					for (const auto& binding : segment.foreignTimelines)
						if (binding.handle.index == handle.index && binding.handle.generation == handle.generation) identity = binding.identity;
					for (size_t slot = 0; slot < m_queueRegistry.SlotCount() && !identity; ++slot) {
						const auto fence = m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot))).GetHandle();
						if (fence.index == handle.index && fence.generation == handle.generation) identity = slot + 1;
					}
					if (!identity) {
						identity = m_queueRegistry.SlotCount() + 1 + segment.foreignTimelines.size();
						segment.foreignTimelines.push_back({identity, handle});
					}
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
					edit->SetQueues(RegistryQueues(m_queueRegistry));
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
				auto bindPosition = [&](Program::Pool& pool, uint32_t i, const Want& want) {
					BT_ZONE_SCOPE("ORG.Persistent.DynamicSegment.Bind");
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
			edit.SetQueues(RegistryQueues(m_queueRegistry));
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
			if (d.type == PassType::Unknown || (std::holds_alternative<std::monostate>(d.pass) && !d.unifiedPass) || d.name.empty()) continue;
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
		std::string reuse;
		{
			std::lock_guard<std::mutex> lock(InvocationReuseRegistryMutex());
			for (const auto* stats : InvocationReuseRegistry()) {
				const uint64_t hits = stats->hits.load();
				const uint64_t misses = stats->misses.load();
				reuse += std::string(stats->name) + " hits=" + std::to_string(hits) + " misses=" + std::to_string(misses)
					+ " check=" + std::to_string(stats->checkNs.load() / (std::max<uint64_t>)(1, hits + misses) / 1000) + "us"
					+ " build=" + std::to_string(stats->buildNs.load() / (std::max<uint64_t>)(1, misses) / 1000) + "us"
					+ " package=" + std::to_string(stats->packageNs.load() / (std::max<uint64_t>)(1, hits) / 1000) + "us"
					+ " [rev=" + std::to_string(stats->revisionNs.load() / (std::max<uint64_t>)(1, hits + misses)) + "ns"
					+ " bind=" + std::to_string(stats->bindingNs.load() / (std::max<uint64_t>)(1, hits + misses)) + "ns"
					+ " desc=" + std::to_string(stats->descriptorNs.load() / (std::max<uint64_t>)(1, hits + misses)) + "ns]; ";
			}
		}
		spdlog::info("Invocation reuse: {}", reuse);
	}
	BT_PLOT("ORG.Persistent.StructuralBuildsTotal", static_cast<int64_t>(state.structuralBuilds));
	BT_PLOT("ORG.Persistent.BindingBuildsTotal", static_cast<int64_t>(state.bindingBuilds));
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

// True when the new lowering declares a direct resource or group the pass
// did not declare before (the live publication has no slot for it).
static bool LoweringAddsDeclarations(const State::MainPass& main, const LoweredPass& lowered) {
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

void RenderGraph::SubmitPersistentStructuralBuild(const std::vector<uint32_t>& structural, const std::vector<uint32_t>& growGroups, rhi::Backend primaryBackend) {
	BT_ZONE_SCOPE("ORG.Persistent.SubmitStructuralBuild");
	auto& compiler = *m_compilerState;
	auto& state = *compiler.persistent;
	// A relowered pass that declares resources the live publication does not
	// have cannot be prepared against it while the build is pending: such
	// edits build inline (one frame hitch). Growth, new passes and relowers
	// that add nothing are safe to build on the worker.
	bool inlineRequired = false;
	std::unordered_map<uint32_t, LoweredPass> lowerings;
	for (const auto mainIndex : structural) {
		auto& main = state.mainPasses[mainIndex];
		auto& any = m_masterPassList[main.masterIndex];
		std::visit([&](auto& value) {
			using T = std::decay_t<decltype(value)>;
			if constexpr (!std::is_same_v<T, std::monostate>) {
				auto lowered = LowerLegacyPass(*this, _registry, m_queueRegistry, primaryBackend, value, any.type, mainIndex);
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
		auto& group = state.groups[groupIndex];
		const auto more = transaction.ReserveGroupMembers(group.group, group.capacity);
		for (const auto slot : more) {
			auto& entry = ClaimEntry(state, slot);
			entry.slot = slot;
			entry.shape = group.shape;
			entry.subscribers = group.subscribers;
			group.memberEntries.push_back(slot.index);
			group.memberResourceIDs.push_back(0);
			group.freeMembers.push_back(static_cast<uint32_t>(group.memberEntries.size() - 1));
		}
		group.capacity *= 2;
		group.identity = {}; // Re-syncs (binding-only) after the install.
		pending->touchedGroups.push_back(groupIndex);
		basic_telemetry::AddCounter("ORG.Persistent.GroupCapacityGrowth");
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
				for (const auto& group : lowered.groups) for (const auto& [id, resource] : group.members) resources.push_back(resource);
				MaterializePersistentStandalone(resources, true);
				InstallMainPass(*this, state, transaction, main, std::move(lowered), mainIndex, true);
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
	structural.insert(structural.end(), pending->deferredStructural.begin(), pending->deferredStructural.end());
	++state.structuralBuilds;
	basic_telemetry::AddCounter("ORG.Persistent.StructuralBuilds");
	spdlog::info("Persistent structural build installed: latency_ms={:.2f} replayed={} deferred={}",
		std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pending->started).count(),
		pending->bindingLog.size(), pending->deferredStructural.size());
	state.pending.reset();
}

// Compile-dump mode for the persistent graph. The legacy compiler rewrites its
// dump every compiled frame; the persistent executable changes only on
// structural builds, so it is written once per selected executable: batches in
// submission order with each pass's authored order and declared states, then
// every slot with its alias placement.
void RenderGraph::WritePersistentGraphDebugDump(const persistent::SelectedPublication& selected) const {
	try {
		const auto& state = *m_compilerState->persistent;
		const auto& executable = *selected.executable;
		const auto& graph = *executable.graph;
		const auto& structure = *graph.structure;
		std::unordered_map<uint32_t, const std::string*> nameBySlot;
		for (const auto& main : state.mainPasses) if (!main.retired) nameBySlot.emplace(main.id.index, &main.name);
		auto passName = [&](uint32_t pass) -> std::string {
			const auto found = nameBySlot.find(structure.passes[pass].preparedPassIndex);
			return found != nameBySlot.end() ? *found->second : "<pass " + std::to_string(pass) + ">";
		};
		auto resourceName = [&](uint32_t resource) -> std::string {
			if (resource >= executable.resourceSlots.size()) return "<resource " + std::to_string(resource) + ">";
			const auto slot = executable.resourceSlots[resource].index;
			const auto* entry = slot < state.entries.size() ? &state.entries[slot] : nullptr;
			return "slot=" + std::to_string(slot) + " name=\"" + (entry && entry->resource ? entry->resource->GetName() : std::string("-")) + "\"";
		};
		auto appendState = [](std::ostringstream& out, const CompileResourceState& value) {
			out << "access=" << rhi::helpers::ResourceAccessMaskToString(static_cast<rhi::ResourceAccessType>(value.access))
				<< " layout=" << rhi::helpers::ResourceLayoutToString(static_cast<rhi::ResourceLayout>(value.layout))
				<< " sync=" << rhi::helpers::ResourceSyncToString(static_cast<rhi::ResourceSyncState>(value.sync))
				<< (value.write ? " write" : "");
		};
		auto appendRange = [](std::ostringstream& out, const CompileRange& range) {
			out << "mips=[" << range.mip << "+" << range.mips << "] slices=[" << range.slice << "+" << range.slices << "]";
		};
		std::ostringstream dump;
		dump << "RenderGraph Persistent State\n"
			<< "structural_revision=" << executable.structuralRevision << " structural_builds=" << state.structuralBuilds
			<< " alias_plans=" << state.aliasPlans << "\n"
			<< "passes=" << structure.passes.size() << " batches=" << graph.batches.size() << " resources=" << structure.resourceIDs.size()
			<< " explicit_edges=" << structure.explicitEdges.size() << " placement_edges=" << structure.placementEdges.size()
			<< " pre_passes=" << state.preMasterIndices.size() << " tail_passes=" << state.tailMasterIndices.size()
			<< " hosted_frame_interrupting=" << state.hostedMainPasses.size() << "\n\n[Batches]\n";
		for (uint32_t batchIndex = 0; batchIndex < graph.batches.size(); ++batchIndex) {
			const auto& batch = graph.batches[batchIndex];
			dump << "batch " << batchIndex << " queue=" << batch.queue << "\n";
			for (const auto pass : batch.passes) {
				const auto& declaration = structure.passes[pass];
				dump << "  [" << pass << "] \"" << passName(pass) << "\" order=" << declaration.originalOrder
					<< " backend=" << declaration.backend << "\n";
				for (const auto& use : declaration.entryStates) {
					dump << "    entry " << resourceName(use.resource) << " "; appendRange(dump, use.range); dump << " "; appendState(dump, use.state); dump << "\n";
				}
				for (const auto& use : declaration.exitStates) {
					dump << "    exit " << resourceName(use.resource) << " "; appendRange(dump, use.range); dump << " "; appendState(dump, use.state); dump << "\n";
				}
				for (const auto& access : declaration.accesses)
					dump << "    access " << resourceName(access.resourceIndex) << (access.write ? " write" : " read") << "\n";
			}
		}
		dump << "\n[Resources]\n";
		for (uint32_t resource = 0; resource < executable.resourceSlots.size(); ++resource) {
			dump << "  [" << resource << "] " << resourceName(resource);
			const auto slot = executable.resourceSlots[resource];
			if (const auto* version = selected.bindings.TryAt(slot.index)) {
				dump << (version->bound ? " bound" : " reserved");
				if (const auto& admission = version->admission; admission && admission->aliasSize)
					dump << " alias_pool=0x" << std::hex << admission->aliasPoolID << std::dec
						<< " alias_offset=" << admission->aliasOffset << " alias_bytes=" << admission->aliasSize;
			}
			dump << "\n";
		}
		namespace fs = std::filesystem;
		std::error_code error;
		fs::path directory = fs::current_path(error);
		if (error) directory.clear();
		directory /= "rendergraph_dumps";
		fs::create_directories(directory, error);
		const auto path = directory / "rendergraph_persistent_state.txt";
		std::ofstream file(path, std::ios::out | std::ios::trunc);
		if (!file.is_open()) { spdlog::warn("Failed to open persistent render graph dump '{}'", path.string()); return; }
		file << dump.str();
		spdlog::info("Persistent render graph dump written to '{}' (structural revision {})", path.string(), executable.structuralRevision);
	} catch (const std::exception& e) {
		spdlog::warn("Failed to write persistent render graph dump: {}", e.what());
	}
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
		if (desc.type == PassType::Unknown || (std::holds_alternative<std::monostate>(desc.pass) && !desc.unifiedPass) || desc.name.empty()) continue;
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
				auto lowered = LowerLegacyPass(*this, _registry, m_queueRegistry, primaryBackend, value, any.type, mainIndex);
				std::vector<Resource*> resources;
				for (const auto& use : lowered.entries) resources.push_back(use.resource);
				for (const auto& group : lowered.groups) for (const auto& [id, resource] : group.members) resources.push_back(resource);
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
bool RenderGraph::PlanPersistentAliasPlacement(persistent::GraphEditTransaction& edit, const persistent::SelectedPublication& scheduled) {
	BT_ZONE_SCOPE("ORG.Persistent.PlanAliases");
	auto& state = *m_compilerState->persistent;
	const auto& executable = *scheduled.executable;
	const auto& graph = *executable.graph;
	const auto& structure = *graph.structure;
	const auto mode = m_getAutoAliasMode ? m_getAutoAliasMode() : AutoAliasMode::Off;
	const bool logging = m_getAutoAliasEnableLogging && m_getAutoAliasEnableLogging();
	const float headroom = m_getAutoAliasPoolGrowthHeadroom ? (std::max)(1.0f, m_getAutoAliasPoolGrowthHeadroom()) : 1.5f;
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
	// Diagnostic bisection: comma-separated name substrings kept out of the pools.
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
				[&](const std::string& token) { return name.find(token) != std::string::npos; }))
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
	for (uint32_t i = 0; i < candidates.size(); ++i) if (candidates[i].pooled) byPool[candidates[i].poolID].push_back(i);
	std::vector<std::pair<uint32_t, uint32_t>> edges; // (logical slot, logical slot)
	uint64_t independentBytes = 0, requiredBytes = 0, reservedBytes = 0;
	size_t overlapPairs = 0;
	auto alignUp = [](uint64_t value, uint64_t alignment) { return alignment <= 1 ? value : (value + alignment - 1) / alignment * alignment; };
	for (auto& [poolID, members] : byPool) {
		std::sort(members.begin(), members.end(), [&](uint32_t l, uint32_t r) {
			const auto& a = candidates[l]; const auto& b = candidates[r];
			if (a.firstUse != b.firstUse) return a.firstUse < b.firstUse;
			if (a.sizeBytes != b.sizeBytes) return a.sizeBytes > b.sizeBytes;
			if (a.lastUse != b.lastUse) return a.lastUse < b.lastUse;
			return a.concrete->GetGlobalResourceID() < b.concrete->GetGlobalResourceID();
		});
		struct Active { uint32_t lastUse; uint64_t start, end; bool operator>(const Active& o) const { return lastUse > o.lastUse; } };
		struct Free { uint64_t start, end; };
		std::priority_queue<Active, std::vector<Active>, std::greater<Active>> active;
		std::vector<Free> freeRanges;
		auto release = [&](Free range) {
			if (range.end <= range.start) return;
			auto it = std::lower_bound(freeRanges.begin(), freeRanges.end(), range.start, [](const Free& f, uint64_t s) { return f.start < s; });
			size_t pos = static_cast<size_t>(it - freeRanges.begin());
			freeRanges.insert(it, range);
			if (pos > 0 && freeRanges[pos - 1].end >= freeRanges[pos].start) {
				freeRanges[pos - 1].end = (std::max)(freeRanges[pos - 1].end, freeRanges[pos].end);
				freeRanges.erase(freeRanges.begin() + static_cast<std::ptrdiff_t>(pos)); --pos;
			}
			while (pos + 1 < freeRanges.size() && freeRanges[pos].end >= freeRanges[pos + 1].start) {
				freeRanges[pos].end = (std::max)(freeRanges[pos].end, freeRanges[pos + 1].end);
				freeRanges.erase(freeRanges.begin() + static_cast<std::ptrdiff_t>(pos + 1));
			}
		};
		uint64_t heapEnd = 0, poolAlignment = 1;
		for (const auto index : members) {
			auto& c = candidates[index];
			independentBytes += c.sizeBytes;
			while (!active.empty() && active.top().lastUse < c.firstUse) { release({active.top().start, active.top().end}); active.pop(); }
			bool found = false; size_t bestRange = 0; uint64_t bestStart = 0, bestSlack = UINT64_MAX;
			for (size_t r = 0; r < freeRanges.size(); ++r) {
				const auto start = alignUp(freeRanges[r].start, c.alignment);
				const auto end = start + c.sizeBytes;
				if (start < freeRanges[r].start || end > freeRanges[r].end) continue;
				const auto slack = freeRanges[r].end - end;
				if (!found || slack < bestSlack || (slack == bestSlack && start < bestStart)) { found = true; bestRange = r; bestStart = start; bestSlack = slack; }
			}
			uint64_t start;
			if (found) {
				const auto selected = freeRanges[bestRange];
				start = bestStart;
				const auto end = start + c.sizeBytes;
				if (selected.start < start && end < selected.end) {
					freeRanges[bestRange].end = start;
					freeRanges.insert(freeRanges.begin() + static_cast<std::ptrdiff_t>(bestRange + 1), Free{end, selected.end});
				} else if (selected.start < start) freeRanges[bestRange].end = start;
				else if (end < selected.end) freeRanges[bestRange].start = end;
				else freeRanges.erase(freeRanges.begin() + static_cast<std::ptrdiff_t>(bestRange));
			} else {
				start = alignUp(heapEnd, c.alignment);
			}
			heapEnd = (std::max)(heapEnd, start + c.sizeBytes);
			poolAlignment = (std::max)(poolAlignment, c.alignment);
			active.push({c.lastUse, start, start + c.sizeBytes});
			c.offset = start;
		}
		requiredBytes += heapEnd;
		if (heapEnd == 0) continue;
		// Pool heap: grow with headroom, never shrink; old generations stay alive
		// through the leases held by in-flight publications.
		auto& pool = state.aliasPools[poolID];
		const bool initial = !static_cast<bool>(pool.allocation);
		const bool grow = !initial && (heapEnd > pool.capacityBytes || poolAlignment > pool.alignment);
		const bool forced = !initial && !grow && state.forceNewAliasHeaps;
		if (initial || grow || forced) {
			uint64_t capacity = heapEnd;
			if (forced) capacity = (std::max)(capacity, pool.capacityBytes);
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
				initial ? "allocated" : forced ? "reallocated (forced)" : "grew", poolID, capacity, heapEnd, poolAlignment, members.size(), pool.generation);
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

// ---------------------------------------------------------------------------
// Execution

void RenderGraph::ExecutePersistentFrame(PassExecutionContext& context) {
	BT_ZONE_SCOPE("ORG.Persistent.Execute");
	auto& compiler = *m_compilerState;
	auto& state = *compiler.persistent;
	if (!state.main) throw std::logic_error("Persistent execution has no prepared frame");
	m_lastPresentDependency.reset();
	context.immediateDispatch = &m_immediateDispatch;
	const size_t slotCount = m_queueRegistry.SlotCount();
	if (!state.timelines) {
		std::vector<experimental::ExecutionTimelinePoint> queues;
		for (size_t slot = 0; slot < slotCount; ++slot) {
			const auto next = m_queueRegistry.GetCurrentFenceValue(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot)));
			queues.push_back({slot + 1, next ? next - 1 : 0});
		}
		const size_t framesInFlight = m_renderGraphSettingsService ? m_renderGraphSettingsService->GetNumFramesInFlight() : 3;
		state.timelines = std::make_unique<experimental::ExecutionTimelineAdmission>(std::move(queues), framesInFlight * 3 + 6);
	}
	// Retirement runs before this frame submits: the presentation tail of the
	// previous frame has been confirmed by now, so completed executions and
	// their publications can be released.
	{
		BT_ZONE_SCOPE("ORG.Persistent.Retire");
		std::vector<experimental::ExecutionTimelinePoint> completed;
		const auto submitted = state.timelines->Submitted();
		for (size_t slot = 0; slot < submitted.size(); ++slot) {
			const auto fence = m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot))).GetCompletedValue();
			completed.push_back({submitted[slot].timeline, (std::min)(submitted[slot].value, fence)});
		}
		if (!state.pendingPresentSubmission) state.timelines->RetireCompleted(completed);
		for (auto& point : completed) point.value = (std::min)(point.value, state.admission->Submitted(point.timeline));
		state.admission->RetireCompleted(completed);
		auto retired = state.timelines->TakeRetiredGarbage();
		if (!retired.empty()) {
			auto garbage = std::make_shared<decltype(retired)>(std::move(retired));
			if (!compiler.frameWorkerScope) compiler.frameWorkerScope = m_taskService->CreateScope("ORG.Frame.Worker");
			if (!m_taskService->Submit(compiler.frameWorkerScope, runtime::TaskPriority::Streaming, "ORG.Frame.RetiredOwnership",
				[garbage] { BT_ZONE_SCOPE("ORG.Frame.DestroyRetiredOwnership"); garbage->clear(); }))
				garbage->clear();
		}
		BT_PLOT("ORG.Persistent.RetainedFrames", static_cast<int64_t>(state.admission->RetainedFrames()));
		BT_PLOT("ORG.Persistent.InFlight", static_cast<int64_t>(state.timelines->InFlight()));
	}
	std::unordered_map<std::string_view, unsigned> statisticsIndices;
	if (m_statisticsService) {
		const auto& names = m_statisticsService->GetPassNames();
		for (unsigned i = 0; i < names.size(); ++i) statisticsIndices.emplace(names[i], i);
	}
	const auto defaultResourceHeap = context.GetResourceDescriptorHeap().GetHandle();
	const auto defaultSamplerHeap = context.GetSamplerDescriptorHeap().GetHandle();
	state.tracyFrameBegun.assign(slotCount, 0);
	const auto presentationSlot = m_queueRegistry.FindGraphicsSlot();

	std::vector<PersistentExecutionState::PreparedSegment*> segments;
	if (state.pre) segments.push_back(&*state.pre);
	segments.push_back(&*state.main);
	if (state.tail) segments.push_back(&*state.tail);
	// ExternalQueueBoundary: each queue's first and last batch across the frame's segments.
	struct BoundaryBatch { size_t segment = SIZE_MAX; uint32_t batch = 0; };
	std::vector<BoundaryBatch> firstBatchOnSlot(slotCount), lastBatchOnSlot(slotCount);
	if (m_externalQueueBoundary.entry || m_externalQueueBoundary.exit) {
		for (size_t segmentIndex = 0; segmentIndex < segments.size(); ++segmentIndex) {
			const auto& batches = segments[segmentIndex]->publication->executable->graph->batches;
			for (uint32_t batch = 0; batch < batches.size(); ++batch) {
				const auto slot = batches[batch].queue;
				if (slot >= slotCount) continue;
				if (firstBatchOnSlot[slot].segment == SIZE_MAX) firstBatchOnSlot[slot] = {segmentIndex, batch};
				lastBatchOnSlot[slot] = {segmentIndex, batch};
			}
		}
	}

	auto submitSegment = [&](PersistentExecutionState::PreparedSegment& segment, size_t segmentIndex) -> std::shared_ptr<const experimental::GraphExecutionTimeline> {
		BT_ZONE_SCOPE("ORG.Persistent.SubmitSegment");
		BT_ZONE_TEXT(segment.label, std::strlen(segment.label));
		std::vector<experimental::ExecutionTimelinePoint> queues;
		std::vector<experimental::PreparedTimelineBinding> timelineBindings;
		for (size_t slot = 0; slot < slotCount; ++slot) {
			const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot));
			const auto next = m_queueRegistry.GetCurrentFenceValue(index);
			auto submitted = next ? next - 1 : 0;
			// A pending presentation tail owns the next graphics value.
			if (state.pendingPresentSubmission && slot == static_cast<size_t>(ToUnderlying(presentationSlot))) submitted = next;
			queues.push_back({slot + 1, submitted});
			timelineBindings.push_back({slot + 1, m_queueRegistry.GetFence(index).GetHandle()});
		}
		timelineBindings.insert(timelineBindings.end(), segment.foreignTimelines.begin(), segment.foreignTimelines.end());
		auto admission = [&] {
			BT_ZONE_SCOPE("ORG.Persistent.PrepareAdmission");
			return state.admission->Prepare(segment.publication, queues, segment.waits, {}, segment.rebindings);
		}();
		std::shared_ptr<const experimental::RenderFrameSnapshot> sealed;
		try {
			sealed = experimental::SealPersistentFrame(state.frameNumber, admission, std::move(segment.invocations));
		} catch (...) { state.admission->Abandon(admission); throw; }
		const auto& graph = *segment.publication->executable->graph;
		std::vector<experimental::FrameRecordingJob> jobs(graph.batches.size());
		std::vector<std::shared_ptr<experimental::OwnedRecordingStatistics>> recordingStatistics;
		std::vector<size_t> demand(slotCount);
		for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
			const auto slot = graph.batches[batch].queue;
			if (slot >= slotCount) throw std::runtime_error("Persistent batch queue is unavailable");
			++demand[slot];
			const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot));
			const auto queueKind = m_queueRegistry.GetKind(index);
			const auto rhiKind = queueKind == QueueKind::Graphics ? rhi::QueueKind::Graphics
				: queueKind == QueueKind::Compute ? rhi::QueueKind::Compute : rhi::QueueKind::Copy;
			auto& job = jobs[batch];
			job.slot = static_cast<uint32_t>(slot);
			job.queueKind = rhiKind;
			job.device = m_queueRegistry.GetDevice(index);
			job.queue = m_queueRegistry.GetQueue(index);
			job.pool = m_queueRegistry.GetSharedPool(index);
			if (!job.pool) throw std::logic_error("Recording queue has no command-list pool");
			job.recording = experimental::BuildPersistentRecordingList(sealed, batch);
			job.recording.bindings = segment.legacyBindings;
			job.recording.externalEntryBarrier = m_externalQueueBoundary.entry
				&& firstBatchOnSlot[slot].segment == segmentIndex && firstBatchOnSlot[slot].batch == batch;
			job.recording.externalExitBarrier = m_externalQueueBoundary.exit
				&& lastBatchOnSlot[slot].segment == segmentIndex && lastBatchOnSlot[slot].batch == batch;
			if (queueKind != QueueKind::Copy) {
				job.recording.resourceDescriptorHeap = defaultResourceHeap;
				job.recording.samplerDescriptorHeap = defaultSamplerHeap;
			}
			if (m_statisticsService) {
				auto statistics = std::make_shared<experimental::OwnedRecordingStatistics>();
				statistics->service = m_statisticsService;
				statistics->frameIndex = state.frameIndex;
				statistics->queueKind = rhiKind;
				statistics->gpuQueries = queueKind != QueueKind::Copy
					&& m_queueRegistry.GetBackendInstance(index) == BackendInstanceId::Primary;
				for (const auto& pass : job.recording.passes) {
					const auto found = statisticsIndices.find(pass.DebugName());
					statistics->passIndices.push_back(found == statisticsIndices.end() ? -1 : static_cast<int>(found->second));
				}
				job.recording.statistics = statistics;
				recordingStatistics.push_back(std::move(statistics));
			}
		}
		{
			BT_ZONE_SCOPE("ORG.Persistent.AcquireCommandLists");
			for (size_t slot = 0; slot < slotCount; ++slot) {
				if (!demand[slot]) continue;
				const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot));
				auto pool = m_queueRegistry.GetSharedPool(index);
				const auto completed = m_queueRegistry.GetFence(index).GetCompletedValue();
				pool->PublishDemand(demand[slot], completed);
				auto ready = pool->AcquireBatch(demand[slot], completed);
				size_t readyIndex = 0;
				for (auto& job : jobs) {
					if (job.slot != slot) continue;
					job.recording.allocation->pair = std::move(ready[readyIndex++]);
					job.recording.allocation->pool = job.pool;
				}
				if (readyIndex != ready.size()) throw std::logic_error("Command-list batch demand did not match recording jobs");
				if (!state.tracyFrameBegun[slot]) {
					const auto first = std::ranges::find_if(jobs, [slot](const auto& job) { return job.slot == slot; });
					if (first != jobs.end()) {
						first->queue.TracyGpuFrameBegin(first->recording.allocation->pair.list.Get());
						state.tracyFrameBegun[slot] = 1;
					}
				}
			}
		}
		experimental::PlannedFrame plan{sealed, std::move(jobs), std::move(timelineBindings), admission.incomingWaits};
		const size_t concurrency = (std::min<size_t>)(graph.batches.size(), 4u);
		std::shared_ptr<const experimental::GraphExecutionTimeline> execution;
		try {
			auto recorded = experimental::RecordFrame(std::move(plan), m_taskService, concurrency);
			execution = std::move(recorded).Submit(*state.timelines);
		} catch (...) {
			state.admission->Abandon(admission);
			throw;
		}
		{
			BT_ZONE_SCOPE("ORG.Persistent.CommitAdmission");
			state.admission->Commit(admission, *execution);
		}
		for (const auto& statistics : recordingStatistics) statistics->Publish();
		for (size_t batch = 0; batch < execution->batches.size(); ++batch)
			m_queueRegistry.EnsureNextFenceValueAtLeast(static_cast<QueueSlotIndex>(static_cast<uint8_t>(graph.batches[batch].queue)),
				execution->batches[batch].signal.value + 1);
		// Present dependency: the batch containing the presentation pass.
		for (size_t batch = 0; batch < graph.batches.size(); ++batch)
			for (const auto pass : graph.batches[batch].passes) {
				const auto prepared = graph.structure->passes[pass].preparedPassIndex;
				if (!segment.names || prepared >= segment.names->size() || (*segment.names)[prepared] != "PresentationReadyPass") continue;
				const auto slot = static_cast<QueueSlotIndex>(static_cast<uint8_t>(graph.batches[batch].queue));
				m_lastPresentDependency = PresentDependency{
					.queue = m_queueRegistry.GetQueue(slot),
					.wait = {m_queueRegistry.GetFence(slot).GetHandle(), execution->batches[batch].signal.value},
					.queueSlot = slot, .batchIndex = batch, .valid = execution->batches[batch].signal.value != 0};
				state.pendingPresentSubmission = execution->submission;
			}
		return execution;
	};
	for (size_t segmentIndex = 0; segmentIndex < segments.size(); ++segmentIndex)
		submitSegment(*segments[segmentIndex], segmentIndex);
	state.pre.reset(); state.main.reset(); state.tail.reset();
	compiler.lastExecutedPreparationSlot = state.frameIndex;
	compiler.lastSubmittedFrameData = state.frameData;
	basic_telemetry::AddCounter("ORG.Persistent.SubmittedFrames");
}

void RenderGraph::ConfirmPersistentPresentationTail() {
	auto& state = *m_compilerState->persistent;
	if (!state.pendingPresentSubmission || !state.timelines) return;
	const auto slot = m_queueRegistry.FindGraphicsSlot();
	auto queue = m_queueRegistry.GetQueue(slot);
	auto& fence = m_queueRegistry.GetFence(slot);
	const auto value = m_queueRegistry.GetNextFenceValue(slot);
	if (queue.Signal({fence.GetHandle(), value}) != rhi::Result::Ok)
		throw std::runtime_error("Presentation tail completion signal failed");
	const experimental::ExecutionTimelinePoint point{static_cast<uint64_t>(ToUnderlying(slot)) + 1u, value};
	state.timelines->ExtendSubmittedExecution(state.pendingPresentSubmission, point);
	state.admission->ExtendSubmitted(point);
	state.pendingPresentSubmission = 0;
	basic_telemetry::AddCounter("ORG.PresentationTail.RetirementReceipts");
}

} // namespace org
