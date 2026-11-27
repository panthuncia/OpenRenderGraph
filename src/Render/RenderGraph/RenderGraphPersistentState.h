#pragma once

// Persistent execution mode: the graph runs from a selected persistent publication. It is rebuilt only by
// structural edits. Upload passes (immediate-mode commands) and per-frame readbacks are not part of it: they are
// submitted through the graph's external client, before and after the main execution. The client and the graph
// share the runtime's resource ledgers, so the GPU work stays ordered.
//
// Every pass's declaration is lowered (PersistentLowering.h) when it is installed
// and again when it redeclares; it is prepared through PrepareFrame every frame.
// Structural rebuilds run inline on the owner thread or on a worker. Alias placement is planned inside the
// structural build (PlanPersistentAliasPlacement): the executable is scheduled
// once, transient lifetimes are packed into persistent pool heaps, placement
// orderings are added, and the executable is rebuilt over the placed bindings.
//
// RenderGraphPersistentState.h: what the persistent route's units share (the execution state, tickets and
// recordings, and the helpers over them). Private to src/Render/RenderGraph.
#include "Render/RenderGraph/RenderGraph.h"
#include "OpenRenderGraph/Runtime.h"
#include "RenderGraphCompilerState.h"
#include "PersistentLowering.h"
#include "AliasPacking.h"
#include "FrameRecording.h"
#include "ScheduleTraceBuild.h"
#include "Render/RenderGraph/RhiExecution.h"
#include "Render/RenderGraph/ExternalEpochExecution.h"
#include "OpenRenderGraph/ExternalClient.h"
#include "Render/RenderGraph/PersistentGraph.h"
#include "Render/RenderGraph/PersistentAliasArena.h"
#include "Render/Runtime/ScopedActiveGraphServices.h"
#include "Interfaces/IDynamicDeclaredResources.h"
#include "Resources/DynamicResource.h"
#include "Resources/BackedResource.h"
#include "Resources/GloballyIndexedResource.h"
#include "Resources/PixelBuffer.h"
#include "Resources/Buffers/DynamicBufferBase.h"
#include "Resources/Buffers/VersionedBuffer.h"
#include "Resources/VersionedResource.h"
#include "Resources/MemoryStatisticsComponents.h"
#include "Render/FrameGraphSnapshot.h"
#include "Managers/Singletons/DeviceManager.h"
#include "Managers/Singletons/DeletionManager.h"
#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Render/RenderGraph/InteropAllocator.h"
#include <rhi_interop_vulkan.h>
#include <thread>

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
#include <set>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include "RenderPasses/Base/TypedRenderGraphPass.h"
#include <BasicTelemetry/Tracy.h>
#include <spdlog/spdlog.h>

namespace org {

// ORG_UPDATE_PROFILE=N (diagnostic): RenderGraph::Update adds each pass's TakeFrameValues + Update time and each preparation
// phase's time (UpdateProfileAdd), and every N updates of an epoch logs the means per update, largest first. Calling thread only.
bool UpdateProfileOn() noexcept;
void UpdateProfileAdd(const char* phase, std::string_view name, double microseconds);

// ORG_STAGED_REVISION_DIAG=1: a staged recording that is stale by a pass's revision logs the rows that changed (diagnostic).
inline bool StagedRevisionDiagnostics() {
	static const bool on = [] {
		char* value = nullptr;
		size_t size = 0;
		const bool set = _dupenv_s(&value, &size, "ORG_STAGED_REVISION_DIAG") == 0 && value && value[0] && value[0] != '0';
		std::free(value);
		return set;
	}();
	return on;
}

namespace persistent_impl {

using compiler::CompileRange;
using compiler::CompileResourceShape;
using compiler::CompileResourceState;
using compiler::PreparedBackingState;
using compiler::PreparedStateRegion;
using lowering::ConcurrentlyShared;
using lowering::LowerRange;
using lowering::LowerState;
using lowering::ShapeOf;
using lowering::UnwrapDynamic;

inline ResourceRegistry::RegistryHandle PersistentResolveByIdThunk(void* user, ResourceIdentifier const& id, bool allowFailure) {
	return static_cast<RenderGraph*>(user)->RequestResourceHandle(id, allowFailure);
}
inline ResourceRegistry::RegistryHandle PersistentResolveByPtrThunk(void* user, Resource* ptr, bool allowFailure) {
	return static_cast<RenderGraph*>(user)->RequestResourceHandle(ptr, allowFailure);
}

inline std::shared_ptr<const std::vector<PreparedStateRegion>> CaptureTrackerRegions(SymbolicTracker* tracker, CompileResourceShape shape) {
	auto regions = std::make_shared<std::vector<PreparedStateRegion>>();
	if (tracker) {
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
inline std::shared_ptr<const std::vector<PreparedStateRegion>> CaptureRegions(Resource& resource, CompileResourceShape shape) {
	return CaptureTrackerRegions(resource.GetStateTracker(), shape);
}

// The admission state of a slot's backing, from a snapshot of `concrete`. It has no alias placement.
inline PreparedBackingState BackingStateOf(Resource& concrete, const ResourceBindingSnapshot& snapshot, uint32_t slotIndex,
	CompileResourceShape shape) {
	PreparedBackingState initial;
	initial.graphResourceID = uint64_t{slotIndex} + 1;
	initial.resource = snapshot.resource.GetHandle();
	initial.shape = shape;
	initial.regions = CaptureRegions(concrete, shape);
	if (!snapshot.representations.empty()) {
		for (const auto& representation : snapshot.representations) initial.deviceHandles.push_back(representation.GetHandle());
		// A new representation pair starts undefined (a texture) or common (a buffer) on every device; the implicit backing's
		// tracker describes another resource.
		auto regions = std::make_shared<std::vector<PreparedStateRegion>>();
		regions->push_back({{0, shape.mips, 0, shape.slices}, shape.hasLayout
			? CompileResourceState{static_cast<uint64_t>(rhi::ResourceAccessType::None), static_cast<uint64_t>(rhi::ResourceLayout::Undefined),
				static_cast<uint64_t>(rhi::ResourceSyncState::None), false}
			: CompileResourceState{static_cast<uint64_t>(rhi::ResourceAccessType::Common), static_cast<uint64_t>(rhi::ResourceLayout::Common),
				static_cast<uint64_t>(rhi::ResourceSyncState::All), false}});
		initial.regions = std::move(regions);
	}
	rhi::ResourceDesc description{};
	if (concrete.TryGetRHIResourceDesc(description)) initial.heapType = description.heapType;
	initial.hostOwned = concrete.GoesHome();
	initial.exempt = concrete.GetGraphOwnership() == Resource::GraphOwnership::ExternalImmutableShaderResource;
	return initial;
}

// Exact binding for one slot from the live resource. Null when the resource has
// no backing yet (unmaterialized) — the slot stays reserved for that frame.
// aliasContinues: an aliased backing whose contents carry over from an earlier execution of the frame
// (PreparedBackingState::aliasContinues). Only the graph's alias plan knows it (State::aliasContinues).
inline std::optional<persistent::BindingVersion> CaptureSlotBinding(Resource& resource, uint32_t slotIndex, CompileResourceShape shape,
	PublicationBindingBundle::Snapshot snapshot = nullptr, bool aliasContinues = false) {
	if (!snapshot) { BT_ZONE_SCOPE("ORG.Persistent.CaptureSlotBinding.Snapshot"); snapshot = PublicationBindingBundle::Capture(resource); }
	if (!snapshot || !snapshot->resource.GetHandle().valid()) return std::nullopt;
	auto* concrete = UnwrapDynamic(&resource);
	if (!concrete) return std::nullopt;
	auto initial = BackingStateOf(*concrete, *snapshot, slotIndex, shape);
	initial.aliasHeap = snapshot->aliasHeap;
	initial.aliasHeapIdentity = snapshot->aliasHeap.get();
	initial.aliasPoolID = snapshot->aliasPoolID;
	initial.aliasOffset = snapshot->aliasOffset;
	initial.aliasSize = snapshot->aliasSize;
	initial.aliasContinues = aliasContinues && snapshot->aliasSize != 0;
	try {
		return persistent::BindingVersion::FromSnapshot(std::move(snapshot), initial,
			reinterpret_cast<uintptr_t>(initial.regions.get()), initial.resource.generation);
	} catch (const std::exception& error) {
		const auto live = ShapeOf(resource);
		throw std::logic_error(std::string(error.what()) + " (slot " + std::to_string(slotIndex) + " '" + resource.GetName()
			+ "': live mips=" + std::to_string(live.mips) + " slices=" + std::to_string(live.slices)
			+ ", captured with mips=" + std::to_string(shape.mips) + " slices=" + std::to_string(shape.slices) + ")");
	}
}

// The binding a slot of concrete takes once it adopted a prepared backing (BackedResource::PreparePlacedBacking): as CaptureSlotBinding
// makes it from a capture then. The regions are the new backing's: a new resource's state. Any thread.
inline std::optional<persistent::BindingVersion> CapturePreparedSlotBinding(Resource& concrete, PreparedPlacedBacking& prepared, uint32_t slotIndex,
	CompileResourceShape shape, bool aliasContinues) {
	const auto& snapshot = prepared.snapshot;
	if (!snapshot || !snapshot->resource.GetHandle().valid()) return std::nullopt;
	PreparedBackingState initial;
	initial.graphResourceID = uint64_t{slotIndex} + 1;
	initial.resource = snapshot->resource.GetHandle();
	initial.shape = shape;
	initial.regions = CaptureTrackerRegions(prepared.StateTracker(), shape);
	rhi::ResourceDesc description{};
	if (concrete.TryGetRHIResourceDesc(description)) initial.heapType = description.heapType;
	initial.hostOwned = concrete.GoesHome();
	initial.exempt = concrete.GetGraphOwnership() == Resource::GraphOwnership::ExternalImmutableShaderResource;
	initial.aliasHeap = snapshot->aliasHeap;
	initial.aliasHeapIdentity = snapshot->aliasHeap.get();
	initial.aliasPoolID = snapshot->aliasPoolID;
	initial.aliasOffset = snapshot->aliasOffset;
	initial.aliasSize = snapshot->aliasSize;
	initial.aliasContinues = aliasContinues && snapshot->aliasSize != 0;
	return persistent::BindingVersion::FromSnapshot(snapshot, initial, reinterpret_cast<uintptr_t>(initial.regions.get()),
		initial.resource.generation);
}

// A transient a graph placed in its alias memory: the resource, and its backing generation right after the placement. A later
// placement by another graph, or a standalone backing, moves the generation.
struct AliasPlacementOwner {
	std::weak_ptr<Resource> resource;
	uint64_t generation = 0;
};
// A graph's end: each transient it placed that no other graph placed since is given back (dematerialized). Otherwise the
// placement stayed bound with no graph that uses it, and kept its pool's memory alive.
// self: the state ending. Placements another state claimed since are left alone.
void ReleaseOwnAliasPlacements(const std::unordered_map<Resource*, AliasPlacementOwner>& owners, const void* self) noexcept;
// Which graphs declare which transients, across every graph (by graph state, with its pass epoch filter). A graph's own pools see
// only its own uses, so a transient a graph of another epoch also binds must never go there: only a frame placement places it,
// or it keeps a backing of its own. DeclareEpochUses replaces the state's set (sorted, unique). It returns whether this made a
// resource newly used by two epochs; the serial then advances (CrossEpochUsesSerial). ForgetEpochUses at the state's end.
// declared: the state's own copy of its set (PersistentExecutionState::declaredUses).
bool DeclareEpochUses(const void* state, uint32_t epoch, std::vector<const void*>& declared, std::vector<const void*> resources);
void ForgetEpochUses(const void* state, std::vector<const void*>& declared) noexcept;
bool DeclaredByOtherEpoch(const void* resource, uint32_t epoch);
uint64_t CrossEpochUsesSerial() noexcept;
// Advances each time a plan takes a resource off a placement another graph made (gives it a backing of its own). A graph checks
// its own placements' backings only when it moved since its plan (BindPersistentFrame).
void NoteAliasPlacementTaken() noexcept;
uint64_t AliasPlacementTakenSerial() noexcept;

struct RotationKey {
	uint64_t concreteID = 0, backingGeneration = 0;
	const void* views = nullptr;
	uint32_t apiIndex = 0, apiGeneration = 0;
	bool operator==(const RotationKey&) const = default;
};
inline RotationKey RotationKeyOfResource(Resource& resource) {
	RotationKey key;
	auto* concrete = UnwrapDynamic(&resource);
	if (!concrete) return key;
	key.concreteID = concrete->GetGlobalResourceID();
	// The generation and the views come from one backing (BackedResource::BackingMutex).
	std::unique_lock<std::recursive_mutex> backingLock;
	if (auto* backed = dynamic_cast<BackedResource*>(concrete)) {
		if (auto* mutex = backed->BackingMutex()) backingLock = std::unique_lock(*mutex);
		key.backingGeneration = backed->GetBackingGeneration();
	} else {
		const auto handle = concrete->GetAPIResource().GetHandle();
		key.apiIndex = handle.index; key.apiGeneration = handle.generation;
	}
	if (auto* indexed = dynamic_cast<GloballyIndexedResource*>(concrete)) key.views = indexed->CaptureBindlessViews().get();
	return key;
}
} // namespace

using namespace persistent_impl;

// ---------------------------------------------------------------------------

// A segment's recorded packets, and what they were recorded for. A revision's published copy, when replayable: then submitted by
// every ticket that takes it, one at a time (a copy's tickets follow each other's completion).
struct RenderGraph::PersistentRecording {
	std::shared_ptr<const compiler::RenderFrameSnapshot> snapshot;
	std::vector<std::shared_ptr<const compiler::IPreparedExecutionBatch>> packets;
	std::vector<std::shared_ptr<compiler::OwnedRecordingStatistics>> statistics;
	bool replayable = false;
	// The resources every use of which is skipped ahead: its lists left their barriers out, so every execution of it leaves them
	// unmoved (FindUntouched's ahead). Null for none.
	std::shared_ptr<const compiler::UntouchedResources> ahead;
	// The untouched resources its last submission found (SubmitPersistentTicket), with the skip answers (every batch's, in order)
	// and the backing set it found them for: a submission with the same answers takes them as they are. The submitting thread's,
	// under the sequencer.
	mutable std::vector<uint8_t> untouchedSkips;
	mutable uint64_t untouchedBackings = 0;
	mutable compiler::UntouchedResources untouched;
	// The reuse key.
	uint64_t hostTag = 0;
	std::shared_ptr<const persistent::SelectedPublication> publication;
	std::shared_ptr<const FrozenExecutionBindings> bindings;
	rhi::DescriptorHeapHandle resourceHeap{}, samplerHeap{};
	std::vector<std::pair<const RenderGraphPass*, uint64_t>> revisions;
	mutable std::atomic<uint64_t> lastUse{0};  // the host frame that last took it
	// What a revision ticket admits it with (AddPersistentTicketCandidate): its producer waits, the versions it bound (its frame
	// rebindings), and the leases its preparation held.
	std::vector<persistent::FrameProducerWait> waits;
	std::vector<persistent::FrameRebinding> rebindings;
	std::vector<std::shared_ptr<const void>> leases;
	// The version number each versioned slot bound when it was recorded (RenderGraph::PersistentRecordingVersionsCurrent).
	std::vector<std::pair<std::shared_ptr<const IVersionedResource>, uint64_t>> versions;
	// The relocation journal's head when every backing it bound was last found current
	// (RenderGraph::PersistentRecordingBackingsCurrent). While the head has not moved, nothing needs checking.
	mutable std::atomic<uint64_t> backingsCheckedAt{0};

	bool Recorded(uint64_t a_hostTag, const persistent::SelectedPublication* a_publication, const FrozenExecutionBindings* a_bindings,
		rhi::DescriptorHeapHandle a_resourceHeap, rhi::DescriptorHeapHandle a_samplerHeap,
		const std::vector<std::pair<const RenderGraphPass*, uint64_t>>& a_revisions) const {
		return hostTag == a_hostTag && publication.get() == a_publication && bindings.get() == a_bindings
			&& resourceHeap.index == a_resourceHeap.index && resourceHeap.generation == a_resourceHeap.generation
			&& samplerHeap.index == a_samplerHeap.index && samplerHeap.generation == a_samplerHeap.generation && revisions == a_revisions;
	}
};

// An epoch's main execution, prepared and recorded ahead of its submission (RenderGraph::PreparePersistentTicket).
struct RenderGraph::PersistentTicket {
	uint32_t epoch = 0;
	std::shared_ptr<detail::ExecutionRecord> execution;  // made by its preparation
	uint32_t slot = 0;  // the run's lease index (its frame index)
	RunRef run{};
	uint64_t hostFrame = 0;
	struct Segment {
		// Until the ticket is admitted (AdmitPersistentTicket): the plan its body was recorded from (PreparePlan; for a
		// candidate, only its publication). Then: the admission made at submission, which SubmitPersistentTicket commits.
		persistent::FrameAdmission admission;
		std::shared_ptr<const PersistentRecording> recording;
		// What the segment was prepared with, which a recording bound later must have been recorded for.
		std::shared_ptr<const FrozenExecutionBindings> bindings;
		rhi::DescriptorHeapHandle resourceHeap{}, samplerHeap{};
		// What the admission at submission is prepared with: the producers' waits and the frame's rebindings (the versions of its
		// versioned slots).
		std::vector<persistent::FrameProducerWait> waits;
		std::vector<persistent::FrameRebinding> rebindings;
		bool admitted = false;   // admission is the submission's, pending
		bool committed = false;  // and committed
	};
	std::vector<Segment> segments;
	// The first pass whose preparation cannot be checked (no revision hash, hosted or captured): diagnostics for a stale ticket.
	std::string uncheckableBy;
	std::vector<std::string> revisionNames;  // the pass name of each entry of revisions (diagnostics)
	// Recordings a candidate displaced: released by the owner thread, with the ticket.
	std::vector<std::shared_ptr<const PersistentRecording>> replaced;
	// What each prepared pass's preparation depended on (RenderGraphPass::RevisionHash).
	std::vector<std::pair<const RenderGraphPass*, uint64_t>> revisions;
	bool uncheckable = false;  // a pass that cannot report its revision: never current
	uint64_t hostTag = 0;      // the host's, for its own currency checks (GraphHost: the backing version)
	bool external = false;     // an external epoch's (SetExternalPersistentEpochs): its submission is the host's execution
	// A revision ticket (PrepareRevisionTicket), or a live one with alternatives to its own segment: its candidates, pushed by the
	// owner thread (newest first) and read by the bind on the submitting thread; the one the bind took. A candidate is a recording
	// only: the ticket is admitted at submission, whichever it takes.
	struct Candidate {
		Segment segment;
		Candidate* next = nullptr;
	};
	bool revision = false;
	std::atomic<Candidate*> candidates{nullptr};
	const Candidate* chosen = nullptr;
	std::vector<Segment> displaced;  // a live ticket's own segment, which a bound candidate replaced
	// What SubmitPersistentTicket added to every batch's waits (schedule introspection, read when the owner completes it).
	std::pair<uint32_t, uint64_t> submittedUploads{UINT32_MAX, 0};
	uint64_t submittedFrameWait = 0;
	// The ticket's uploads planned by queue (RecordTakenUploads with queue lists), for SubmitPersistentTicket; written by the
	// submitting thread before it submits, read by the owner after the ticket's Submitted message (schedule introspection).
	struct QueueUploads {
		bool planned = false;  // false: every batch off the uploads' queue waits for them
		std::vector<uint8_t> waitsForUploads;  // per batch of segments.front(): it uses what the graphics uploads write
		uint32_t spared = 0;  // batches off the graphics queue that do not (statistics)
		// Destinations that are a late-bound version's backing (statistics): routed to a declaring batch's queue, or waited for off graphics.
		uint32_t lateRouted = 0, lateWaits = 0;
		// Destinations no batch declares (statistics). Shaders may still read them through a bindless index, so every batch off
		// the graphics queue waited for the graphics list.
		uint32_t undeclared = 0;
		// A list of uploads recorded on a batch's queue, submitted just ahead of that batch (its queue's first in the ticket).
		struct Leading {
			uint32_t batch = 0;
			rhi::CommandList list{};
			size_t copies = 0;
			std::vector<compiler::ExecutionTimelinePoint> waits;  // against earlier executions' use of its destinations
		};
		std::vector<Leading> leading;
	} queueUploads;
	PersistentTicket() = default;
	PersistentTicket(const PersistentTicket&) = delete;
	PersistentTicket& operator=(const PersistentTicket&) = delete;
	~PersistentTicket() {
		for (auto* candidate = candidates.exchange(nullptr, std::memory_order_acquire); candidate;)
			delete std::exchange(candidate, candidate->next);
	}
};

// The persistent graph has three parts, each a base of PersistentExecutionState:
// - the instance: its passes, slot directory, resolver groups, alias pools and the edits in flight (owner thread);
// - the layout: what follows from the selected publication, rebuilt only when another is selected;
// - the executions: admission, the one prepared and the submitted ones not yet complete.

// The instance.
struct RenderGraph::PersistentInstanceState {
	struct SlotEntry {
		persistent::ResourceSlotId slot;
		Resource* resource = nullptr;      // Registry object (wrapper for dynamic resources).
		// A resolver group member's own reference: the resolver may drop the member while the slot still names it, and every
		// preparation polls the slots' backings (PollBackings). Released when the slot is rebound. A versioned slot's: the version
		// the latest preparation bound.
		std::shared_ptr<Resource> member;
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
		// A versioned resource's slot (LoweredUse::versioned, the swapchain among them). No publication binds it. Each preparation
		// binds the version it selects, as a frame rebinding and an overlay of the bindings (PrepareVersionedBindings).
		std::shared_ptr<const IVersionedResource> versioned;
		// The latest preparation's binding of a versioned slot, kept for the next one that selects the same version. Not kept for
		// a per-frame resource (IVersionedResource::PerFrame), whose versions must not outlive the frames that bind them.
		struct FrameBinding {
			std::shared_ptr<const ResourceVersion> version;
			RotationKey key;
			persistent::FrameRebinding rebinding;
			FrozenExecutionBindings::ResourceBinding binding;
		};
		std::optional<FrameBinding> frameBinding;
		// A versioned slot's current version number when the passes' declarations last moved to it
		// (RenderGraph::ReleaseReplacedVersionDeclarations).
		uint64_t sweptVersion = 0;
		// A versioned slot whose current version the latest alias plan took as a candidate (frame placement only): that version's
		// number. 0 when it was not a candidate. A preparation that finds another current version places again, so the new
		// version gets a key of its own in the next usage summary.
		uint64_t aliasPlannedVersion = 0;
		bool multiDevice = false;         // A pass on another device than the primary uses it: bound with representations.
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
		// Per member slot: the scheduling ID its member replaced in place (a rotated publication). A declaration captured
		// before the rotation names the old ID; it resolves here (MemberPosition).
		std::vector<std::optional<uint64_t>> memberReplacedIDs;
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
        std::shared_ptr<const ResourceUseLayout> resourceUses;
        // The declaration the selected executable was built from, which preparation hands the pass: a relowering joins
        // it when its executable installs, with declaredViews and declaredBindings.
        std::shared_ptr<const ResourceUseLayout> preparedResourceUses;
        std::shared_ptr<const void> preparedBindings;
        std::vector<std::vector<persistent::ViewToken>> declaredViews;
        std::vector<persistent::BindingToken> declaredBindings;
        // Per declared use: the group member entry a resolver use names (UINT32_MAX for a direct use). Resolved once, at
        // install, against the declaration's own capture; a rotated publication replaces the member in place, so the
        // entry keeps naming the logical resource the pass declared.
        std::vector<uint32_t> resolverUseEntries;
        BackendInstanceId device = BackendInstanceId::Primary; // The pass's device (its lowering's).
        struct GroupViews { uint32_t group; std::vector<BindlessViewRequest> views; };
        std::vector<GroupViews> groupViews;
		uint64_t fingerprint = 0; // Structural identity of the lowered declaration (membership excluded).
		uint64_t structuralFingerprint = 0; // The same without the declared layout's uses (FingerprintLowering).
		std::array<uint64_t, 5> structuralParts{}; // Its parts (FingerprintParts), to log which one a relower changed.
		std::string declarationSummary; // DescribeDeclaration of the installed lowering, for the relower log.
		std::vector<std::pair<uint64_t, std::string>> loweredDirect; // Diagnostic: (scheduling ID, name).
		std::vector<const void*> loweredGroupKeys;
		std::vector<ExternalTimelinePoint> explicitWaits;
		// Resolvers lowered as direct declarations, with the resource set they were lowered with: their slots hold those
		// resources until the pass is lowered again, so every preparation checks that it resolves the same set. A versioned
		// resource is never one: its slot takes each preparation's version.
		std::vector<std::pair<std::shared_ptr<const IResourceResolver>, ResolverResourceSetIdentity>> directResolvers;
		// RebindsConcreteResources: the wrappers its pointer declarations go through, one per distinct resource in
		// declaration order (UseRebindSlots). Each is the pass's own slot; its backing is the resource declared there last.
		std::vector<std::shared_ptr<DynamicResource>> rebindSlots;
		uint32_t epoch = persistent::AllEpochs; // Host epoch (ExternalPassDesc::Epoch).
	};

	persistent::GraphProgram program;
	std::vector<SlotEntry> entries;                                // Index == slot index.
	// A sweep left a pass on a replaced version it could not move yet, so the next preparation sweeps again.
	bool versionSweepRetry = false;
	std::unordered_map<uint64_t, std::vector<uint32_t>> entriesByResourceID;
	// Resources whose last direct slot retired. A declared resource is pinned in the graph's transient maps
	// (TrackTransientFrameResource), which keep it alive for the executions that bind it, raw slot pointers and publication
	// snapshots being unowned. RetirePersistentExecutions lets the pin go once the selected publication no longer binds the
	// resource and every execution submitted until then has completed (ReleaseRetiredPins). Without it a graph keeps
	// everything it ever declared until it is rebuilt, and a job graph, which never is, keeps it for the session.
	struct RetiredPin {
		Resource* resource = nullptr;  // Kept alive by the pin itself.
		uint64_t registryID = 0, concreteID = 0;
		bool unbound = false;          // The selected publication no longer binds it; then `after` is set.
		std::vector<compiler::ExecutionTimelinePoint> after;
	};
	std::vector<RetiredPin> retiredPins;
	std::vector<MainPass> mainPasses;
	std::vector<ResolverGroup> groups;
	std::vector<size_t> uploadMasterIndices;                        // Passes with immediate-mode commands.
	std::vector<uint32_t> mainPassByMaster;                        // UINT32_MAX when not main.
	std::vector<uint8_t> masterHostedInMain;
	// Alias planning (structural builds only), into the pools of the graph's arena (RenderGraph::SetPersistentAliasArena; one of
	// its own without one). This state is the arena's owner key for its requirements.
	std::shared_ptr<PersistentAliasArena> aliasArena;
	PersistentInstanceState() = default;
	PersistentInstanceState(const PersistentInstanceState&) = delete;
	PersistentInstanceState& operator=(const PersistentInstanceState&) = delete;
	~PersistentInstanceState() { ReleaseAliasPlacements(); }
	// Gives back what this state holds in what other graphs place into: the placements it made (ReleaseOwnAliasPlacements),
	// its frame regions, and its requirements in the arena's pools. Its end does it otherwise; a graph destroyed on another
	// thread does it first on the owner thread (RenderGraph::ReleaseAliasPlacements), since a resource it gives back may be
	// placed again by another graph at once. Once.
	void ReleaseAliasPlacements() noexcept {
		if (aliasPlacementsReleased) return;
		aliasPlacementsReleased = true;
		ReleaseOwnAliasPlacements(aliasPlacementOwners, this);
		ForgetEpochUses(this, declaredUses);
		frameRegions.clear();
		if (!aliasArena) return;
		std::lock_guard lock(aliasArena->mutex);
		for (auto& [id, pool] : aliasArena->pools) pool.required.erase(this);
		aliasArena->stale.erase(this);
	}
	bool aliasPlacementsReleased = false;
	struct AliasPlacementRecord {
		uint64_t poolID = 0, offset = 0, sizeBytes = 0;
		const AliasHeapGeneration* heap = nullptr;
		// The placement continues from an earlier execution of the frame (aliasContinues). A change binds it again.
		bool continues = false;
		bool operator==(const AliasPlacementRecord&) const = default;
	};
	std::unordered_map<Resource*, AliasPlacementRecord> aliasPlaced; // Concrete resources placed by this planner.
	// Concrete resources the latest alias plan placed from a frame placement whose contents carry over into this graph's
	// executions from an earlier one in the frame (PreparedBackingState::aliasContinues): a later epoch of a resource several
	// epochs use, or a resource that keeps its contents across runs. Bindings of these enter without a discard.
	std::unordered_set<Resource*> aliasContinues;
	std::unordered_map<Resource*, AliasPlacementOwner> aliasPlacementOwners; // The same, for this state's end (ReleaseOwnAliasPlacements).
	// Resources several epochs use that the latest plan bound where another epoch's graph placed them (keepBacking), with their
	// backing generation then. When that graph re-places one (its own re-placement for a newer frame placement), this graph's
	// executable no longer admits the backing: the next preparation re-places this graph too (BindPersistentFrame).
	std::unordered_map<Resource*, AliasPlacementOwner> adoptedBackings;
	// The transients this graph declares, as last given to DeclareEpochUses, and the cross-epoch serial its latest alias plan saw. A
	// newer serial (another epoch's graph now binds a resource too) makes the next preparation check whether one of them lies in this
	// graph's own pools, and re-place if so (BindPersistentFrame).
	std::vector<const void*> declaredUses;
	uint64_t crossEpochSerialAtPlan = 0;
	uint64_t placementTakenSerialAtPlan = 0;
	// With a frame alias arena (RenderGraph::SetFrameAliasRegion): the range reserved for each pool's transients, by pool id. Each
	// plan reserves its ranges again and gives back the previous ones. The state's end gives back the last ones.
	std::unordered_map<uint64_t, std::shared_ptr<FrameAliasArena::Reservation>> frameRegions;
	// The frame placement the graph's first alias plan took (RenderGraph::SetFramePlacementSource), kept for its life. Null when
	// there was none. framePlacementTaken: the first plan has run.
	std::shared_ptr<const FramePlacementVersion> framePlacement;
	bool framePlacementTaken = false;
	// Moves each time the graph's placements change: a frame placement version taken in place, a prepared re-placement installed,
	// and every alias plan that installs. A kept recording made under another serial is not current (O-M6-3).
	uint64_t placementSerial = 0;
	// Concrete resources a pass on another device uses. Their representations are committed allocations shared between the
	// devices, so they are never alias candidates: a placement would be replaced by those, and rematerializing one drops them.
	std::unordered_set<Resource*> multiDeviceResources;
	// Shared alias eligibility (RenderGraph::IsSharedAliasEligible) by resource, with the description it was decided for.
	std::unordered_map<Resource*, std::pair<size_t, bool>> sharedAliasEligibility;
	std::vector<std::pair<uint32_t, uint32_t>> aliasEdges;          // Installed placement orderings (logical pass slots).
	uint64_t aliasPlans = 0, aliasRematerializations = 0;
	bool forceNewAliasHeaps = false; // the next plan re-places every pool (a packing strategy change)
	// Built off the owner thread (PrepareFirstPersistentBuild) without its alias placement: the first preparation places.
	bool deferAliasPlacement = false;
	bool aliasPlacementDeferred = false;
	// RenderGraph::PlaceDeferredAliasesAtBuild: the plan runs before the graph is attached, so it must move nothing another graph
	// binds. A plan that would sets atBuildRefusal and changes nothing.
	bool placingAtBuild = false;
	// RenderGraph::PublishPlacementSummaryAtBuild: the plan publishes the graph's usage summary and stops there.
	bool summaryOnly = false;
	std::string atBuildRefusal;
	// The plan at the build declined a new version (RenderGraph::SetFramePlacementDecline): its new transients may go to its pool.
	bool atBuildOwnPools = false;
	// Resources this graph materialized standalone itself (MaterializePersistentStandalone). Any other materialized candidate
	// may be bound by another graph.
	std::unordered_set<Resource*> ownMaterialized;
	// RenderGraph::FreezeAliasedBindings: the graph keeps the backings it bound for the transients it placed, whatever they point at
	// now (a staged build of the same epoch placed them elsewhere).
	std::atomic<bool> freezeAliased{false};
	// A versioned slot was added or a slot retired since RenderGraph::PublishDeclaredVersions last ran.
	bool declaredVersionsChanged = true;
	AutoAliasPackingStrategy aliasPackingStrategy = AutoAliasPackingStrategy::GreedySweepLine; // The latest plan's.
	// Diagnostic: per-pass preparation cost (logged periodically).
	std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> prepareCostByPass; // lowering: name -> (ns, calls)
	// A structural build in flight on a worker. Frames keep using the selected
	// publication; the slot directory already describes the new structure, so
	// subscriber slot maps are frozen (no rebuild) until the install.
	std::function<void(Resource&)> ensureRepresentations; // RenderGraph::EnsurePersistentRepresentations
	struct PendingStructural {
		std::unique_ptr<persistent::GraphEditTransaction> edit;
		std::shared_ptr<const persistent::SelectedPublication> base;
		uint32_t baseSlotCount = 0;
		std::atomic<int> phase{0}; // 0 scheduling, 1 scheduled, 2 placing, 3 placed, -1 failed
		std::atomic_bool cancelled{false};
		std::atomic_bool clearedOrderings{false};
		std::shared_ptr<const persistent::SelectedPublication> result;
		std::string error;
		compiler::CompileWorkspace workspace;
		std::vector<uint32_t> dirtyPasses;        // slotsDirty deferred to the install
		// Relowered passes' declared tokens, which resolve only against the new publication: swapped in at the install.
		struct DeclaredTokens {
			uint32_t mainIndex;
			std::vector<std::vector<persistent::ViewToken>> views;
			std::vector<persistent::BindingToken> bindings;
			std::shared_ptr<const ResourceUseLayout> preparedResourceUses;
			std::shared_ptr<const void> preparedBindings;
			BackendInstanceId device = BackendInstanceId::Primary;
		};
		std::vector<DeclaredTokens> declaredTokens;
		std::vector<uint32_t> touchedGroups;      // grown in the transaction: not polled meanwhile
		std::vector<uint32_t> deferredStructural; // relower requests that arrived meanwhile
		// Binding edits installed on the live publication meanwhile, replayed
		// onto the result (nullopt = unbind). Keyed by slot index.
		std::unordered_map<uint32_t, std::optional<persistent::BindingVersion>> bindingLog;
		std::chrono::steady_clock::time_point started;
	};
	std::shared_ptr<PendingStructural> pending;
	// A re-placement for a newer frame placement version, prepared on a worker while the graph keeps running
	// (RenderGraph::PrepareFramePlacementReplacement): the plan's placements and orderings, the resources it moves given backings
	// made ahead, and the validated publication. The preparation that first sees the version current installs it
	// (InstallPreparedReplacement), or re-places as before when anything it was made from changed.
	struct PreparedReplacement {
		uint64_t versionId = 0;
		std::shared_ptr<const FramePlacementVersion> version;
		std::shared_ptr<const persistent::SelectedPublication> base;
		uint64_t aliasPlans = 0;
		uint64_t startedFrame = 0;
		std::atomic<int> phase{0}; // 0 running, 1 ready, -1 refused or failed
		std::string why;
		std::shared_ptr<const persistent::SelectedPublication> ready;
		std::vector<std::pair<uint32_t, uint32_t>> edges;
		struct Moved {
			std::shared_ptr<Resource> resource;
			std::shared_ptr<PreparedPlacedBacking> prepared;
			AliasPlacementRecord from, to;
			bool multiEpoch = false;
			std::vector<uint32_t> entries; // slot entry indices bound to it
		};
		std::vector<Moved> moved;
		// Frame placed resources the version leaves where they are, whose records it changes (a block pool id, its epochs).
		std::vector<Moved> updated;
		size_t kept = 0;
		double planMs = 0.0, backingsMs = 0.0, buildMs = 0.0;
		compiler::CompileWorkspace workspace;
		std::atomic_bool cancelled{false};
		std::chrono::steady_clock::time_point started;
	};
	std::shared_ptr<PreparedReplacement> preparedReplacement;
	uint64_t preparedReplacementsInstalled = 0, preparedReplacementsRefused = 0;
	// Submitted this frame.
	uint64_t structuralBuilds = 0, bindingBuilds = 0;
	// Bindings taken out of frames' edits because they change which slots share a backing (SharingChangePolicy::
	// Deferred), by slot index. The next structural build applies them. A newer edit of the slot drops its entry.
	std::unordered_map<uint32_t, persistent::BindingVersion> deferredSharing;
	uint64_t deferredSharingBuilds = 0;
};

// The layout: tables built from the selected publication.
struct RenderGraph::PersistentLayoutState {
	// The binding table recording resolves slots through (resources, views), per publication.
	const persistent::SelectedPublication* bindingsFor = nullptr;
	std::shared_ptr<const FrozenExecutionBindings> bindings;
	// The latest preparation's overlay of the versioned slots on `bindings`, with what it bound in each slot (the binding's
	// owner). A preparation that binds the same versions takes the same object, so a kept recording made with them still
	// matches (PersistentRecording::Recorded compares bindings by identity).
	struct VersionedOverlay {
		std::shared_ptr<const FrozenExecutionBindings> base, bindings;
		std::vector<std::pair<uint32_t, const void*>> bound;
	};
	VersionedOverlay versionedOverlay;
	// The currency probe's previous preparation (RenderGraph::SetCurrencyProbe), and its counts since the last take. Owner thread.
	struct CurrencyProbe {
		bool primed = false;
		std::shared_ptr<const persistent::SelectedPublication> publication;
		std::shared_ptr<const void> bindings;
		std::vector<std::pair<const void*, uint64_t>> versions;
		uint64_t journalHead = 0;
		uint64_t placementId = 0, aliasPlans = 0;
		RenderGraph::CurrencyProbeStats stats;
	};
	CurrencyProbe currencyProbe;
	// Pass names per logical pass slot, per executable.
	std::shared_ptr<const std::vector<std::string>> mainNames;
	const void* mainNamesExecutable = nullptr;
	const void* dumpedExecutable = nullptr; // Compile-dump mode: last executable written.
	// Schedule introspection (RenderGraph::SetScheduleTraceEnabled), owner thread.
	struct ScheduleTraces {
		const void* executable = nullptr;  // the traced whole-frame executable (not retained), with its revision
		uint64_t structuralRevision = 0;
		std::vector<std::pair<const void*, std::shared_ptr<const ScheduleTrace>>> traces;  // by executable: the frame's, then each epoch's
		std::vector<ScheduleSubmissionWaits> submissions;  // the latest of each label
		std::vector<uint8_t> recorded;                     // per submission: recorded since the last publication
		ScheduleSubmissionWaits scratch;                   // one execution's, filled before it is recorded (capacity reused)
	};
	ScheduleTraces scheduleTraces;
};

// One execution, from its preparation (PreparePersistentExecution) to its submission (ExecutePersistentFrame).
struct RenderGraph::PersistentExecution {
	// How it executes. A ticket's (PreparePersistentTicket) is recorded for the ticket's host to submit, not submitted.
	// - hostTag: the host's tag, which a kept recording must have been recorded under too;
	// - admissionOnly: it only admits, and the ticket binds a recording made ahead (BindPersistentTicketRecording);
	// - hostData: prepared for host data (a revision). Its passes' revision hashes do not see that data (RevisionHash reads
	//   live state), so its recording is neither taken from nor added to the kept sets.
	using Options = PersistentExecutionOptions;
	struct PreparedSegment {
		const char* label = "";
		std::shared_ptr<const persistent::SelectedPublication> publication;
		std::vector<PreparedPass> invocations;
		std::shared_ptr<const std::vector<std::string>> names; // Per logical pass slot, for statistics/present detection.
		std::vector<persistent::FrameRebinding> rebindings;
		// The version each versioned slot bound (its number), for a recording's VersionsCurrent check.
		std::vector<std::pair<std::shared_ptr<const IVersionedResource>, uint64_t>> versions;
		bool perFrame = false;  // a rebinding is a per-frame resource's (IVersionedResource::PerFrame): never kept
		std::vector<persistent::FrameProducerWait> waits;
		std::shared_ptr<const FrozenExecutionBindings> bindings;
		std::vector<compiler::PreparedTimelineBinding> foreignTimelines;
		std::vector<std::shared_ptr<const void>> leases;
		// A synchronous execution's kept copy (PersistentExecutionOptions::keptCopy), UINT32_MAX for none.
		uint32_t keptCopy = UINT32_MAX;
		// The copy's kept recording, current for this execution: it is submitted instead of a new recording, and the segment
		// prepared nothing (invocations is empty).
		std::shared_ptr<const PersistentRecording> replay;
		// Recorded for a kept copy: every pass's revision hash, in preparation order with its name, and whether each was skipped
		// for the host data (RenderGraphPass::SetSkipWhen). uncheckable: a pass reported no revision (uncheckableBy names it).
		std::vector<std::pair<const RenderGraphPass*, uint64_t>> revisions;
		std::vector<std::string> revisionNames;
		std::vector<std::pair<const RenderGraphPass*, uint8_t>> skips;
		// With the revision audit on (RenderGraph::SetKeptRecordingAudit): each revisions entry's frame data, null where it
		// cannot be compared.
		std::vector<std::shared_ptr<const PreparedFrameDataAudit>> audits;
		bool uncheckable = false;
		std::string uncheckableBy;
		// Why the copy's kept recording was not current (LastKeptRun::reason).
		std::string staleReason;
		// Batch reuse (RenderGraph::SetKeptBatchReuse), recorded for a kept copy: each revisions entry's prepared pass index, and
		// the prepared passes with no revision, with their names.
		std::vector<uint32_t> revisionPrepared;
		std::vector<std::pair<uint32_t, std::string>> uncheckablePrepared;
		// A partial run: per batch, 1 where the batch is submitted from `replay` (its passes were not prepared), 0 where it is
		// recorded now; and per batch why it records. Empty for any other run.
		std::vector<uint8_t> reuseBatches;
		std::vector<std::string> staleBatches;
	};
	// An upload recorded by an upload pass. Submitted through the graph's external client (RenderGraph::GetExternalClient)
	// before the main segment.
	struct PendingUpload {
		std::string name;
		std::vector<ExternalClient::Access> accesses;
		std::vector<std::byte> bytecode;
		std::shared_ptr<const void> keepAlive;
		std::function<void()> commit;  // called once the upload is submitted
		std::vector<ExternalTimelinePoint> signals;  // signalled when the upload finishes
	};

	Options options;
	uint64_t number = 0;     // its index (PassPrepareContext::executionIndex)
	uint32_t frameIndex = 0;  // its preparation slot
	std::shared_ptr<const IHostExecutionData> frameData;
	// Empty for an epoch with no main passes.
	std::optional<PreparedSegment> main;
	std::vector<PendingUpload> uploads;
};

// The executions.
struct RenderGraph::PersistentExecutionsState {
	std::unique_ptr<persistent::SynchronousAdmission> admission;
	std::unique_ptr<compiler::ExecutionTimelineAdmission> timelines;
	// The latest preparation's execution index and slot.
	uint64_t frameNumber = 0;
	uint32_t frameIndex = 0;
	// The execution prepared and not yet executed.
	std::optional<PersistentExecution> prepared;
	// The latest execution (its index is frameNumber), and the submitted ones not yet seen complete.
	std::shared_ptr<detail::ExecutionRecord> lastExecution;
	std::vector<std::shared_ptr<detail::ExecutionRecord>> pendingExecutions;
	// Lists submitted after an execution's batches (its home acquires), kept until their own signal completes.
	std::vector<std::pair<compiler::ExecutionTimelinePoint, std::shared_ptr<const void>>> trailingLeases;
	// What the last full retirement pass (RetirePersistentExecutions) left:
	// - submitted: the timelines' submitted value per queue slot it saw,
	// - lowest: per queue slot, the lowest value that anything still held waits for (UINT64_MAX for none).
	// While the submitted values stay the same and every queue's completed value is below its lowest, nothing can be released,
	// and retirement returns at once. skippable is false when something held waits for a timeline that is not a queue slot's.
	struct RetireWatch {
		bool skippable = false;
		std::vector<uint64_t> submitted, lowest;
	} retireWatch;
	// Per frame slot: the handoff lists tickets recorded from predicted plans (FrameAdmission::sharedBarriers), the most recent
	// first, kHandoffCacheEntries at most. A later ticket in the slot whose admission takes one of these plans submits its lists
	// again: the slot's earlier work has completed by then. Several, for an executable whose plans alternate.
	struct HandoffCache {
		std::shared_ptr<const compiler::PreparedExecutionBarrierPlan> plan;
		bool external = false;
		std::vector<rhi::CommandList> handoff, streamHandoff;  // per batch
		std::vector<uint32_t> targets;  // per batch, the batch whose handoff list holds its entry barriers
		uint64_t untouched = 0;  // the untouched resources' signature (compiler::UntouchedResources): their entry barriers are left out
		std::vector<std::shared_ptr<const void>> allocations;
	};
	static constexpr size_t kHandoffCacheEntries = 4;
	std::vector<std::vector<HandoffCache>> handoffCache;
	std::vector<uint8_t> tracyFrameBegun;
	// Kept recordings of synchronous executions, by kept copy (PersistentExecutionOptions::keptCopy): the copy's last replayable
	// recording and what it was recorded for beyond the recording's own key (RenderGraph::PersistentRecording).
	struct SyncKept {
		std::shared_ptr<const PersistentRecording> recording;
		std::vector<std::string> revisionNames;  // per recording->revisions entry
		std::vector<std::pair<const RenderGraphPass*, uint8_t>> skips;
		std::vector<std::shared_ptr<const PreparedFrameDataAudit>> audits;  // per recording->revisions entry, with the audit on
		uint64_t reusesSinceAudit = 0;
		std::vector<uint64_t> foreignTimelines;  // the identities its packets bind beyond the queues'
		uint64_t placementSerial = 0;
		// Why the copy's last recording was not kept (empty when it was, or before the copy's first run).
		std::string unkeptReason;
		// Recorded ahead (RecordStagedCopy) and not used yet: its first run confirms it by its passes' frame data.
		bool staged = false;
		// The lease index the recording was made for: its latch regions are that index's. A copy served by another lease since
		// (GraphHost's second set of copies for staged runs) records again.
		uint32_t lease = UINT32_MAX;
		// Diagnostics (ORG_STAGED_REVISION_DIAG=1): each recording->revisions entry's revision rows at the staging, so a staged
		// recording stale by a revision logs which rows changed.
		std::vector<std::vector<uint64_t>> stagedRevisionRows;
		// Batch reuse (RenderGraph::SetKeptBatchReuse): each recording->revisions entry's batch; per batch whether its packet can be
		// submitted again (empty: every batch can), and why not.
		std::vector<uint32_t> revisionBatches;
		std::vector<uint8_t> batchReusable;
		std::vector<std::string> batchWhy;
	};
	// Each prepared pass's batch in the executable batchOfPreparedFor, for batch reuse. UINT32_MAX: in none.
	std::weak_ptr<const persistent::ExecutableGeneration> batchOfPreparedFor;
	std::vector<uint32_t> batchOfPrepared;
	// The executable whose kept batches were last logged (batch reuse: once per executable).
	std::weak_ptr<const persistent::ExecutableGeneration> keptBatchesLogged;
	std::vector<SyncKept> syncKept;
	// Diagnostic: preparation cost per main pass index (ns, calls).
	std::vector<std::pair<uint64_t, uint64_t>> prepareCostByMain;
};

struct RenderGraph::PersistentExecutionState : PersistentInstanceState, PersistentLayoutState, PersistentExecutionsState {};

// What the phases of one segment execution share (RenderGraph::AdmitPersistentSegment, RecordPersistentSegment and
// SubmitPersistentSegment).
struct RenderGraph::PersistentSegmentRun {
	PersistentExecution& execution;
	PersistentExecution::PreparedSegment& segment;
	PersistentExecuteTimings& timings;
	std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
	// Adds the time since the previous lap to a timing.
	void Lap(double& into) {
		const auto now = std::chrono::steady_clock::now();
		into += std::chrono::duration<double, std::micro>(now - last).count();
		last = now;
	}
	bool traceSchedule = false;
	size_t slotCount = 0;
	QueueSlotIndex presentationSlot{};
	rhi::DescriptorHeapHandle resourceHeap{}, samplerHeap{};
	const std::unordered_map<std::string_view, unsigned>* statisticsIndices = nullptr;
	// Each queue's first and last batch, for ExternalQueueBoundary. UINT32_MAX when the queue has none.
	std::vector<uint32_t> firstBatchOnSlot, lastBatchOnSlot;

	// Set by admission.
	std::vector<compiler::PreparedTimelineBinding> timelineBindings;
	persistent::FrameAdmission admission;
	bool keepable = false;
	bool external = false;
	// A staged recording's (RecordStagedCopy): admitted from a plan, recorded and kept, never submitted.
	bool staged = false;
	// A kept copy's execution that cannot be kept: why (LastKeptRun::reason).
	std::string unkeptReason;
	// Batch reuse (RenderGraph::SetKeptBatchReuse), a kept copy's run: per batch whether its new recording can be kept (its passes
	// all report revisions and a late one is replayable), and why not. Empty without batch reuse.
	std::vector<uint8_t> batchKeepable;
	std::vector<std::string> batchUnkept;
	// Set by recording.
	std::shared_ptr<const compiler::RenderFrameSnapshot> sealed;
	std::vector<compiler::FrameRecordingJob> jobs;
	// An open ticket's resources every use of which is skipped ahead: its lists leave out their barriers (compiler::UntouchedResources).
	std::shared_ptr<const compiler::UntouchedResources> ahead;
	std::vector<std::shared_ptr<compiler::OwnedRecordingStatistics>> recordingStatistics;
	std::optional<compiler::RecordedFrame> recorded;
};

namespace persistent_impl {

using State = RenderGraph::PersistentExecutionState;

// Queues the pin of a retiring direct slot's resource for release (State::retiredPins). RenderGraphPersistent.cpp.
void NoteRetiredDirectResource(State& state, const State::SlotEntry& entry);

// Gives DeclareEpochUses the transients the state's live slots bind now (aliasable resources only). RenderGraphPersistentStructural.cpp.
void SyncEpochUses(State& state, uint32_t epoch);

// Each prepared pass's batch in the executable (batch reuse, RenderGraph::SetKeptBatchReuse): UINT32_MAX for a prepared pass in no
// batch. Kept in the state until the executable changes.
inline const std::vector<uint32_t>& BatchOfPrepared(State& state, const std::shared_ptr<const persistent::ExecutableGeneration>& executable) {
	if (state.batchOfPreparedFor.lock() == executable) return state.batchOfPrepared;
	const auto& graph = *executable->graph;
	auto& map = state.batchOfPrepared;
	map.clear();
	for (uint32_t batch = 0; batch < graph.batches.size(); ++batch)
		for (const auto pass : graph.batches[batch].passes) {
			const auto prepared = graph.structure->passes.at(pass).preparedPassIndex;
			if (prepared == UINT32_MAX) continue;
			if (map.size() <= prepared) map.resize(size_t{prepared} + 1, UINT32_MAX);
			map[prepared] = batch;
		}
	state.batchOfPreparedFor = executable;
	return map;
}

// Rotation state of an entry's current backing. Type classification is cached
// per registry object; only dynamic wrappers are unwrapped every frame.
inline RotationKey RotationKeyOf(State::SlotEntry& entry) {
	RotationKey key;
	if (!entry.resource) return key;
	if (entry.classified != entry.resource) {
		entry.classified = entry.resource;
		entry.wrapper = dynamic_cast<DynamicResource*>(entry.resource) != nullptr;
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
	std::unique_lock<std::recursive_mutex> backingLock;
	if (entry.backed) {
		if (auto* mutex = entry.backed->BackingMutex()) backingLock = std::unique_lock(*mutex);
		key.backingGeneration = entry.backed->GetBackingGeneration();
	} else {
		const auto handle = concrete->GetAPIResource().GetHandle();
		key.apiIndex = handle.index; key.apiGeneration = handle.generation;
	}
	if (entry.indexed) key.views = entry.indexed->CaptureBindlessViews().get();
	return key;
}

// The admission identity of a queue slot's timeline (Runtime::TimelineIdentity).
inline uint64_t QueueTimelineIdentity(Runtime& runtime, const QueueRegistry& registry, size_t slot) {
	return runtime.TimelineIdentity(registry.GetFence(static_cast<QueueSlotIndex>(static_cast<uint32_t>(slot))));
}

// Handles are per device: a timeline is its device's implementation and its handle there.
inline bool SameTimeline(const rhi::Timeline& a, const rhi::Timeline& b) noexcept {
	return a.impl == b.impl && a.GetHandle().index == b.GetHandle().index && a.GetHandle().generation == b.GetHandle().generation;
}

inline std::vector<compiler::CompileQueue> RegistryQueues(const QueueRegistry& registry, rhi::Backend backend) {
	std::vector<compiler::CompileQueue> queues;
	for (size_t slot = 0; slot < registry.SlotCount(); ++slot) {
		const auto index = static_cast<QueueSlotIndex>(static_cast<uint32_t>(slot));
		queues.push_back({static_cast<uint32_t>(registry.GetBackendInstance(index)), true,
			registry.GetKind(index) == QueueKind::Copy && backend == rhi::Backend::D3D12});
	}
	return queues;
}

} // namespace

// ---------------------------------------------------------------------------
// Lowered-declaration helpers

namespace persistent_impl {

inline uint64_t FingerprintDeclaration(const compiler::CompilePass& d) {
	uint64_t h = 0x9e3779b97f4a7c15ull;
	auto mix = [&](uint64_t value) { h ^= value + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
	mix(d.backend); mix(d.preferredQueueSlot); mix(d.forceBatchIsolation ? 1 : 0); mix(d.recordAtExecution ? 1 : 0);
	mix(d.mayNotWrite ? 1 : 0);
	auto slots = d.compatibleQueueSlots; std::sort(slots.begin(), slots.end());
	for (const auto slot : slots) mix(0x100 + slot);
	return h;
}
using lowering::FingerprintLowering;
using lowering::FingerprintParts;
using lowering::FingerprintPartName;
using lowering::DescribeDeclaration;
using lowering::LoweredGroup;
using lowering::LoweredPass;
using lowering::LoweredUse;
} // namespace

namespace persistent_impl {

// RenderGraphPersistent.cpp
bool BindEntry(State& state, persistent::GraphEditTransaction& edit, uint32_t index);
void GrowGroup(State& state, persistent::GraphEditTransaction& edit, uint32_t groupIndex, size_t needed);
void InstallMainPass(RenderGraph& graph, State& state, persistent::GraphEditTransaction& edit,
	State::MainPass& main, LoweredPass lowered, uint32_t mainIndex, bool replace, std::optional<uint32_t> authoredOrder = std::nullopt,
	std::vector<uint32_t>* touchedGroups = nullptr);
bool LoweringAddsDeclarations(const State::MainPass& main, const LoweredPass& lowered);
// For a pass that RebindsConcreteResources: points the lowering's uses of resources declared by pointer at the pass's
// rebind slots, and gives each slot the resource now declared in its place. Resolver members, versioned resources (the
// swapchain among them), passes on another device and resources that already are wrappers keep their own entries. Does
// nothing for other passes.
// A relower against the installed executable. When the pass declares the same direct uses and groups, and every use and
// view of its new layout maps onto a slot and view `live` already has (a group member that arrived within the group's
// capacity, say), the pass takes the new layout, bindings and tokens with no build. Returns null then; otherwise, changing
// nothing, why it doesn't map (for the relower log).
const char* RemapDeclaration(State& state, const persistent::SelectedPublication& live, State::MainPass& main, LoweredPass& lowered);
void UseRebindSlots(State::MainPass& main, LoweredPass& lowered, const IDynamicDeclaredResources* dynamic);
// Why a pass's new declaration differs from the one installed:
// - NoOp: the same declaration again;
// - Rebind: the same uses with other resources of the same shape;
// - Shape: uses, views, groups or names added, removed or changed;
// - DirectResolver: a resolver declared directly resolves to another set;
// - Remapped: not a cause, an outcome: a Shape relower that RemapDeclaration took with no build.
enum class RelowerCause : uint8_t { NoOp, Rebind, Shape, DirectResolver, Remapped };
RelowerCause ClassifyRelower(const State::MainPass& main, const LoweredPass& lowered, bool directStale);
const char* RelowerCauseName(RelowerCause cause) noexcept;
// Adds the relower to the ORG.Persistent.Relower.* counters.
void CountRelower(RelowerCause cause, bool inlineBuild);
// SharingChangePolicy::Deferred: takes the edits of `edit` that change which slots share a backing into
// state.deferredSharing, so `edit` builds binding-only. Leaves `edit` alone when it can't be split.
void DeferSharingChanges(State& state, persistent::GraphEditTransaction& edit);
// Applies state.deferredSharing to `edit` (a structural build's transaction) and clears it.
void ApplyDeferredSharing(State& state, persistent::GraphEditTransaction& edit);


// RenderGraphPersistentTrace.cpp
void FillScheduleSubmissionWaits(ScheduleSubmissionWaits& out, const compiler::Schedule& graph,
	const std::vector<std::vector<compiler::ExecutionTimelinePoint>>& incoming, std::pair<uint32_t, uint64_t> uploadsDone, uint64_t frameWait,
	const RenderGraph::PersistentTicket::QueueUploads* queueUploads = nullptr);

} // namespace persistent_impl

} // namespace org

namespace org::persistent {
// Per batch of `graph`: whether one of its passes declares a late-bound buffer whose versions are ready at the frame wait
// (VersionedBuffer::LateReady::FrameWait), through `backings` (an admission's, by compiler resource). Every batch counts when
// the graph does not say what each batch accesses.
template <class Graph, class Backings>
std::vector<uint8_t> FrameWaitReaders(const Graph& graph, const Backings& backings) {
	const auto batchCount = graph.batches.size();
	std::vector<uint8_t> reads(batchCount, 0);
	if (graph.boundaryAccessesByBatch.size() != batchCount) {
		reads.assign(batchCount, 1);
		return reads;
	}
	for (size_t batch = 0; batch < batchCount; ++batch)
		for (const auto& access : graph.boundaryAccessesByBatch[batch]) {
			if (access.resource >= backings.size()) continue;
			const auto& backing = backings[access.resource];
			if (backing.resource.valid() && detail::IsFrameWaitStandIn(backing.resource)) {
				reads[batch] = 1;
				break;
			}
		}
	return reads;
}
}  // namespace org::persistent
