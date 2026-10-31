#pragma once

// Persistent execution mode: the main graph runs from a selected persistent
// publication that is rebuilt only by explicit structural edits. Truly dynamic
// passes (immediate-mode uploads, per-frame readbacks) run in Pre/Tail segments
// built per frame from tiny throwaway programs. All segments are admitted in
// order through one SynchronousAdmission, so cross-segment hazards are ordinary
// previous-submission waits and barriers.
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
#include "OpenRenderGraph/Trace.h"
#include "Render/RenderGraph/RenderGraph.h"
#include "OpenRenderGraph/Runtime.h"
#include "RenderGraphCompilerState.h"
#include "PersistentLowering.h"
#include "AliasPacking.h"
#include "FrameRecording.h"
#include "ScheduleTraceBuild.h"
#include "Render/RenderGraph/RhiExecution.h"
#include "Render/RenderGraph/ExternalEpochExecution.h"
#include "Render/RenderGraph/PersistentGraph.h"
#include "Render/Runtime/ScopedActiveGraphServices.h"
#include "Interfaces/IDynamicDeclaredResources.h"
#include "Resources/DynamicResource.h"
#include "Resources/BackedResource.h"
#include "Resources/GloballyIndexedResource.h"
#include "Resources/PixelBuffer.h"
#include "Resources/Buffers/DynamicBufferBase.h"
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

inline std::shared_ptr<const std::vector<PreparedStateRegion>> CaptureRegions(Resource& resource, CompileResourceShape shape) {
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
inline std::optional<persistent::BindingVersion> CaptureSlotBinding(Resource& resource, uint32_t slotIndex, CompileResourceShape shape,
	PublicationBindingBundle::Snapshot snapshot = nullptr) {
	if (!snapshot) { BT_ZONE_SCOPE("ORG.Persistent.CaptureSlotBinding.Snapshot"); snapshot = PublicationBindingBundle::Capture(resource); }
	if (!snapshot || !snapshot->resource.GetHandle().valid()) return std::nullopt;
	auto* concrete = UnwrapDynamic(&resource);
	if (!concrete) return std::nullopt;
	PreparedBackingState initial;
	initial.graphResourceID = uint64_t{slotIndex} + 1;
	if (trace::Any()) initial.debugName = resource.GetName();
	initial.resource = snapshot->resource.GetHandle();
	initial.shape = shape;
	initial.regions = CaptureRegions(*concrete, shape);
	if (!snapshot->representations.empty()) {
		for (const auto& representation : snapshot->representations) initial.deviceHandles.push_back(representation.GetHandle());
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
inline RotationKey RotationKeyOfResource(Resource& resource) {
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

using namespace persistent_impl;

// ---------------------------------------------------------------------------

// A segment's recorded packets, and what they were recorded for. Kept (RenderGraph::m_keptRecordings) when replayable: then
// submitted by every ticket that takes it, one at a time (a frame slot's tickets follow each other's completion).
struct RenderGraph::PersistentRecording {
	std::shared_ptr<const compiler::RenderFrameSnapshot> snapshot;
	std::vector<std::shared_ptr<const compiler::IPreparedExecutionBatch>> packets;
	std::vector<std::shared_ptr<compiler::OwnedRecordingStatistics>> statistics;
	bool replayable = false;
	// The reuse key.
	uint64_t hostTag = 0;
	std::shared_ptr<const persistent::SelectedPublication> publication;
	std::shared_ptr<const FrozenExecutionBindings> bindings;
	rhi::DescriptorHeapHandle resourceHeap{}, samplerHeap{};
	std::vector<std::pair<const RenderGraphPass*, uint64_t>> revisions;
	mutable std::atomic<uint64_t> lastUse{0};  // the host frame that last took it
	// What a revision ticket admits it with (AddPersistentTicketCandidate): its producer waits, and the leases its preparation held.
	std::vector<persistent::FrameProducerWait> waits;
	std::vector<std::shared_ptr<const void>> leases;

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
	uint8_t slot = 0;
	uint64_t hostFrame = 0;
	struct Segment {
		persistent::FrameAdmission admission;
		std::shared_ptr<const PersistentRecording> recording;
		// What the segment was prepared with, which a recording bound later must have been recorded for.
		std::shared_ptr<const FrozenExecutionBindings> bindings;
		rhi::DescriptorHeapHandle resourceHeap{}, samplerHeap{};
	};
	std::vector<Segment> segments;
	// Recordings a kept one replaced (AdoptKeptPersistentRecording): released by the owner thread, with the ticket.
	std::vector<std::shared_ptr<const PersistentRecording>> replaced;
	// What each prepared pass's preparation depended on (RenderGraphPass::RevisionHash).
	std::vector<std::pair<const RenderGraphPass*, uint64_t>> revisions;
	bool uncheckable = false;  // a pass that cannot report its revision: never current
	uint64_t hostTag = 0;      // the host's, for its own currency checks (GraphHost: the backing version)
	bool external = false;     // an external epoch's (SetExternalPersistentEpochs): its submission is the host's execution
	// A revision ticket (PrepareRevisionTicket), or a live one with alternatives to its own segment: its candidates, pushed by the
	// owner thread (newest first) and read by the bind on the submitting thread; the one the bind took. Completion abandons every
	// other candidate's admission, and a live ticket's own segment when a candidate displaced it.
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
	// The ticket's uploads planned by queue (RecordPendingUploads with queue lists), for SubmitPersistentTicket; written by the
	// submitting thread before it submits, read by the owner after the ticket's Submitted message (schedule introspection).
	struct QueueUploads {
		bool planned = false;  // false: every batch off the uploads' queue waits for them
		std::vector<uint8_t> waitsForUploads;  // per batch of segments.front(): it uses what the graphics uploads write
		uint32_t spared = 0;  // batches off the graphics queue that do not (statistics)
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

struct RenderGraph::PersistentExecutionState {
	struct SlotEntry {
		persistent::ResourceSlotId slot;
		Resource* resource = nullptr;      // Registry object (wrapper for dynamic resources).
		// A resolver group member's own reference: the resolver may drop the member (a versioned buffer's old version) while the
		// slot still names it, and every preparation polls the slots' backings (PollBackings). Released when the slot is rebound.
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
		bool swapchain = false;            // Rebound per frame at admission.
		bool multiDevice = false;          // A pass on another device than the primary uses it: bound with representations.
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
		std::vector<std::pair<uint64_t, std::string>> loweredDirect; // Diagnostic: (scheduling ID, name).
		std::vector<const void*> loweredGroupKeys;
		std::vector<ExternalTimelinePoint> explicitWaits;
		// Resolvers lowered as direct declarations, with the resource set they were lowered with: their slots hold those
		// resources until the pass is lowered again, so every preparation checks that it resolves the same set.
		std::vector<std::pair<std::shared_ptr<const IResourceResolver>, ResolverResourceSetIdentity>> directResolvers;
		// Frame-interrupting per-frame pass hosted in the main executable (not in
		// the master list). It is prepared once and records nothing afterwards
		// until its removal is installed.
		std::unique_ptr<AnyPassAndResources> hosted;
		bool hostedPrepared = false;
		bool retired = false; // Hosted pass removed; the main index may be reused.
		uint32_t epoch = persistent::AllEpochs; // Host epoch (ExternalPassDesc::Epoch).
	};
	struct PreparedSegment {
		const char* label = "";
		std::shared_ptr<const persistent::SelectedPublication> publication;
		std::vector<PreparedPass> invocations;
		std::shared_ptr<const std::vector<std::string>> names; // Per logical pass slot, for statistics/present detection.
		std::vector<persistent::FrameRebinding> rebindings;
		std::vector<persistent::FrameProducerWait> waits;
		std::shared_ptr<const FrozenExecutionBindings> legacyBindings;
		std::vector<compiler::PreparedTimelineBinding> foreignTimelines;
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
	std::unique_ptr<compiler::ExecutionTimelineAdmission> timelines;
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
	// A frame was prepared; main may still be empty (an epoch with no main passes).
	bool framePrepared = false;
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
		uint64_t idleSinceFrame = 0; // The frame the latest plan stopped placing into it; 0 while placed into.
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
	AutoAliasPackingStrategy aliasPackingStrategy = AutoAliasPackingStrategy::GreedySweepLine; // The latest plan's.
	const void* dumpedExecutable = nullptr; // Compile-dump mode: last executable written.
	// Diagnostic: per-pass preparation cost (logged periodically).
	std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> prepareCostByPass; // lowering: name -> (ns, calls)
	std::vector<std::pair<uint64_t, uint64_t>> prepareCostByMain;                    // main pass index -> (ns, calls)
	std::shared_ptr<const std::vector<std::string>> mainNames;
	const void* mainNamesExecutable = nullptr;
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
	// Submitted this frame.
	uint64_t pendingPresentSubmission = 0;
	uint64_t structuralBuilds = 0, bindingBuilds = 0;
	std::vector<uint8_t> tracyFrameBegun;
	// Readback captures anchored after a main pass (RenderGraphPersistentCaptures.cpp), by pass name: appended to the
	// pass's own packet when it is next prepared, so a capture never changes the executable.
	struct PendingCapture { runtime::ReadbackCaptureInfo info; uint64_t requestedFrame = 0; };
	std::unordered_map<std::string, std::vector<PendingCapture>> pendingCaptures;
	// Captures the anchor pass could not take (it does not bind the resource, or not over the whole range): inserted
	// next frame as frame-interrupting capture passes.
	std::vector<runtime::ReadbackCaptureInfo> capturesToInsert;
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

namespace persistent_impl {

using State = RenderGraph::PersistentExecutionState;

// Rotation state of an entry's current backing. Type classification is cached
// per registry object; only dynamic wrappers are unwrapped every frame.
inline RotationKey RotationKeyOf(State::SlotEntry& entry) {
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

// The admission identity of a queue slot's timeline (Runtime::TimelineIdentity).
inline uint64_t QueueTimelineIdentity(Runtime& runtime, const QueueRegistry& registry, size_t slot) {
	return runtime.TimelineIdentity(registry.GetFence(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot))));
}

// Handles are per device: a timeline is its device's implementation and its handle there.
inline bool SameTimeline(const rhi::Timeline& a, const rhi::Timeline& b) noexcept {
	return a.impl == b.impl && a.GetHandle().index == b.GetHandle().index && a.GetHandle().generation == b.GetHandle().generation;
}

inline std::vector<compiler::CompileQueue> RegistryQueues(const QueueRegistry& registry, rhi::Backend backend) {
	std::vector<compiler::CompileQueue> queues;
	for (size_t slot = 0; slot < registry.SlotCount(); ++slot) {
		const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot));
		queues.push_back({static_cast<uint32_t>(registry.GetBackendInstance(index)), true,
			registry.GetKind(index) == QueueKind::Copy && backend == rhi::Backend::D3D12});
	}
	return queues;
}

} // namespace

// ---------------------------------------------------------------------------
// Lowered-declaration helpers

namespace persistent_impl {

inline bool IsSwapchainResource(Resource& resource) {
	return resource.GetName() == "Backbuffer" && dynamic_cast<DynamicResource*>(&resource)
		&& !dynamic_cast<BackedResource*>(UnwrapDynamic(&resource));
}
inline uint64_t FingerprintDeclaration(const compiler::CompilePass& d) {
	uint64_t h = 0x9e3779b97f4a7c15ull;
	auto mix = [&](uint64_t value) { h ^= value + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
	mix(d.backend); mix(d.preferredQueueSlot); mix(d.forceBatchIsolation ? 1 : 0); mix(d.recordAtExecution ? 1 : 0);
	auto slots = d.compatibleQueueSlots; std::sort(slots.begin(), slots.end());
	for (const auto slot : slots) mix(0x100 + slot);
	return h;
}
using lowering::FingerprintLowering;
using lowering::LoweredGroup;
using lowering::LoweredPass;
using lowering::LoweredUse;
inline uint64_t HashSegmentTemplate(const State::SegmentProgram::Template& t) {
	uint64_t h = 0x7f4a7c159e3779b9ull;
	auto mix = [&](uint64_t value) { h ^= value + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
	mix(t.shape.mips); mix(t.shape.slices); mix(t.shape.hasLayout ? 1 : 0);
	mix(t.range.mip); mix(t.range.mips); mix(t.range.slice); mix(t.range.slices);
	mix(t.state.access); mix(t.state.layout); mix(t.state.sync); mix(t.state.write ? 1 : 0);
	return h ? h : 1;
}


} // namespace

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

namespace persistent_impl {
struct ForcedRealiasConfig { uint64_t interval = 0; bool withCaptures = false; };
inline const ForcedRealiasConfig& ForcedRealias() {
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

namespace persistent_impl {

// RenderGraphPersistent.cpp
bool BindEntry(State& state, persistent::GraphEditTransaction& edit, uint32_t index);
void GrowGroup(State& state, persistent::GraphEditTransaction& edit, uint32_t groupIndex, size_t needed);
void InstallMainPass(RenderGraph& graph, State& state, persistent::GraphEditTransaction& edit,
	State::MainPass& main, LoweredPass lowered, uint32_t mainIndex, bool replace, std::optional<uint32_t> authoredOrder = std::nullopt,
	std::vector<uint32_t>* touchedGroups = nullptr);
bool LoweringAddsDeclarations(const State::MainPass& main, const LoweredPass& lowered);

// RenderGraphPersistentCaptures.cpp
// Takes the readback service's new capture requests: after-graph captures become per-frame passes at the end (frameExt),
// anchored ones wait for their pass's next preparation; the ones a pass could not take last frame become interrupting passes.
void PlacePersistentCaptures(RenderGraph& graph, State& state, bool ticket, std::vector<ExternalPassDesc>& frameExt,
	std::vector<ExternalPassDesc>& interrupting);
// The pass's packet with the captures waiting for it recorded after its own commands.
PreparedPass AppendPersistentCaptures(RenderGraph& graph, State& state, const State::MainPass& pass, const std::string& name,
	PreparedPass packet, const FramePreparationContext& preparation, const persistent::SelectedPublication& publication);

// RenderGraphPersistentTrace.cpp
void FillScheduleSubmissionWaits(ScheduleSubmissionWaits& out, const compiler::CompiledGraph& graph,
	const std::vector<std::vector<compiler::ExecutionTimelinePoint>>& incoming, std::pair<uint32_t, uint64_t> uploadsDone, uint64_t frameWait,
	const RenderGraph::PersistentTicket::QueueUploads* queueUploads = nullptr);

} // namespace persistent_impl

} // namespace org
