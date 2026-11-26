#pragma once

#include "Render/RenderGraph/FrameAliasArena.h"
#include "Render/RenderGraph/FramePlacement.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <string>
#include <string_view>
#include <memory>
#include <optional>
#include <variant>
#include <functional>
#include <span>
#include <utility>
#include <array>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <spdlog/spdlog.h>
#include <rhi.h>
#include <BasicTelemetry/Tracy.h>

#include "RenderPasses/Base/RenderPass.h"
#include "Resources/ResourceStateTracker.h"
#include "Interfaces/IResourceProvider.h"
#include "Render/ResourceRegistry.h"
#include "Render/RecordingTags.h"
#include "Render/CommandListPool.h"
#include "Interfaces/IPassBuilder.h"
#include "Render/MemoryIntrospectionAPI.h"
#include "Render/Runtime/IStatisticsService.h"
#include "Render/Runtime/IUploadService.h"
#include "Render/Runtime/IDescriptorService.h"
#include "Render/Runtime/IRenderGraphSettingsService.h"
#include "Render/Runtime/ITaskService.h"
#include "OpenRenderGraph/ExecutionHandle.h"
#include "Render/Runtime/TaskServiceAccess.h"
#include "Render/QueueKind.h"
#include "Render/RunRef.h"
#include "Render/QueueRegistry.h"
#include "Render/DeviceRegistry.h"
#include "Resources/PixelBuffer.h"
#include "Resources/Buffers/Buffer.h"
#include "Resources/TrackedAllocation.h"
#include "Render/RenderGraph/Aliasing/RenderGraphAliasingSubsystem.h"
#include "Render/RenderGraph/ScheduleTrace.h"
#include "Interfaces/IResourceResolver.h"
#include "Render/RenderGraph/RenderGraphExtension.h"

namespace org::compiler { struct ExecutionTimelinePoint; struct PreparedTimelineBinding; }

namespace org {

class Runtime;
class Resource;
class RenderPassBuilder;
class ExternalEpochExecution;
class ExternalClient;
class PersistentAliasArena;
class AliasHeapGeneration;
class IVersionedResource;
struct IPassBuilder;
struct IDynamicDeclaredResources;

namespace ui {
	struct FrameGraphSnapshot;
}

template<typename T>
concept DerivedResource = std::derived_from<T, Resource>;

enum class PassRunMask : uint8_t;
[[nodiscard]] constexpr PassRunMask operator|(PassRunMask a, PassRunMask b) noexcept;

enum class PassRunMask : uint8_t {
	None = 0,
	Immediate = 1u << 0,
	Retained = 1u << 1,
	Both = Immediate | Retained
};

constexpr uint8_t to_u8(PassRunMask v) noexcept {
	return static_cast<uint8_t>(v);
}

[[nodiscard]] constexpr PassRunMask operator&(PassRunMask a, PassRunMask b) noexcept {
	return static_cast<PassRunMask>(to_u8(a) & to_u8(b));
}

[[nodiscard]] constexpr PassRunMask operator|(PassRunMask a, PassRunMask b) noexcept {
	return static_cast<PassRunMask>(to_u8(a) | to_u8(b));
}

[[nodiscard]] constexpr PassRunMask operator^(PassRunMask a, PassRunMask b) noexcept {
	return static_cast<PassRunMask>(to_u8(a) ^ to_u8(b));
}

[[nodiscard]] constexpr PassRunMask operator~(PassRunMask a) noexcept {
	return static_cast<PassRunMask>(~to_u8(a));
}

constexpr PassRunMask& operator&=(PassRunMask& a, PassRunMask b) noexcept { return a = (a & b); }
constexpr PassRunMask& operator|=(PassRunMask& a, PassRunMask b) noexcept { return a = (a | b); }
constexpr PassRunMask& operator^=(PassRunMask& a, PassRunMask b) noexcept { return a = (a ^ b); }

namespace persistent { class GraphEditTransaction; class SelectedPublication; struct ExecutableGeneration; }

class RenderGraph {
public:
	// Timestamps in recordings kept for resubmission (SetPersistentRecordingTimestampMode).
	enum class RecordingTimestampMode : uint8_t { Off, Passes, Lists };
	// Declared at namespace scope (RenderGraphExtension.h), so extension headers need not include the graph.
	using ExternalInsertKind = org::ExternalInsertKind;
	using ExternalInsertPoint = org::ExternalInsertPoint;
	using PassType = org::PassType;
	using ExternalPassDesc = org::ExternalPassDesc;
	using IRenderGraphExtension = org::IRenderGraphExtension;

	inline bool Has(PassRunMask m, PassRunMask f) {
		return (uint8_t(m) & uint8_t(f)) != 0;
	}

	// How a pass's resolvers lower as persistent groups, decided when the pass declares (while its resolver snapshots
	// still list their declared members): one requirement template list per resolver snapshot, and the declared members
	// the groups cover. Absent when one of the resolvers is not patchable; they then lower as direct entries.
	struct ResolverGroupPlan {
		std::vector<std::vector<ResolverSnapshot::RequirementTemplate>> templates;
		std::unordered_set<uint64_t> memberIDs;
	};

	struct PassAndResources {
		std::shared_ptr<RenderGraphPass> pass;
		bool unifiedDeclaration = false;
		PassParameters resources;
		std::string name;
		std::string techniquePath;
		int statisticsIndex = -1;
		bool collectStatistics = true;
		uint32_t epoch = UINT32_MAX; // persistent::AllEpochs
		bool recordAtExecution = false; // ExternalPassDesc::recordAtExecution

		PassRunMask run = PassRunMask::Both; // default behavior
		std::vector<std::byte> immediateBytecode; // Stores the immediate execution bytecode
		std::shared_ptr<org::imm::KeepAliveBag> immediateKeepAlive = nullptr; // Keeps alive resources used by immediate execution bytecode
		std::vector<std::shared_ptr<Resource>> retainedAnonymousKeepAlive; // Keeps retained anonymous handles alive across frames
		std::vector<ResolverSnapshot> resolverSnapshots; // Versioned resolver snapshots for auto-invalidation
		std::vector<ExternalTimelinePoint> explicitExternalWaitsBeforeTransitions;
		std::optional<ResolverGroupPlan> resolverGroupPlan;
		std::shared_ptr<const org::imm::PreparedBufferCopies> preparedBufferCopies;
	};

	// Transitional tags preserve the current compiler variant while all three
	// ingestion paths now share one queue-agnostic stored representation.
	struct RenderPassAndResources : PassAndResources {};

	struct PresentDependency {
		rhi::Queue queue;
		rhi::TimelinePoint wait{};
		QueueSlotIndex queueSlot{};
		uint64_t batchIndex = 0;
		bool valid = false;
	};

	// A graph on `runtime`'s device, using its services (until replaced with the Set*Service calls).
	explicit RenderGraph(std::shared_ptr<Runtime> runtime);
	Runtime& GetRuntime() const noexcept { return *m_runtime; }
	DeviceInstanceId RegisterBackendDevice(rhi::Backend backend, rhi::Device device);
	const DeviceRegistry& GetDeviceRegistry() const noexcept { return m_backendDevices; }
	DeviceInstanceId GetPrimaryDeviceInstance() const noexcept { return m_backendDevices.PrimaryId(); }
	~RenderGraph();
	using AutoAliasReasonCount = org::alias::AutoAliasReasonCount;
	using AutoAliasExcludedResourceDebug = org::alias::AutoAliasExcludedResourceDebug;
	using AutoAliasPoolRangeDebug = org::alias::AutoAliasPoolRangeDebug;
	using AutoAliasPoolDebug = org::alias::AutoAliasPoolDebug;
	using AutoAliasDebugSnapshot = org::alias::AutoAliasDebugSnapshot;
	AutoAliasDebugSnapshot GetAutoAliasDebugSnapshot() const;
	void BuildMemoryIntrospectionFrameGraphSnapshot(
		ui::FrameGraphSnapshot& out,
		const std::vector<org::memory::ResourceMemoryRecord>& memoryRecords) const;
	void AddRenderPass(std::shared_ptr<RenderPass> pass, RenderPassParameters& resources, std::string name = "", std::vector<ResolverSnapshot> resolverSnapshots = {});
	// GPU ranges around every pass this graph records (a GPU profiler's, e.g. Telemetry/NvPerfCapture): used by an
	// execution whose context brings none of its own, which is how a host that builds the context itself
	// (GraphHost, persistent tickets) gets them. Set before executions that should carry them.
	using GpuPassRangeBegin = std::function<void(rhi::CommandList, rhi::Queue, const char* queueName, const char* passName)>;
	using GpuPassRangeEnd = std::function<void(rhi::CommandList, rhi::Queue)>;
	void SetGpuPassRangeCallbacks(GpuPassRangeBegin begin, GpuPassRangeEnd end) {
		m_gpuPassRangeBegin = std::move(begin);
		m_gpuPassRangeEnd = std::move(end);
	}
	// Persistent execution: the graph runs from a selected persistent publication, rebuilt only by structural edits.
	// Upload passes (immediate-mode commands) are submitted through the external client before it.
	// Its state (RenderGraphPersistentState.h, opaque elsewhere): the instance, the layout and the executions.
	struct PersistentInstanceState;
	struct PersistentLayoutState;
	struct PersistentExecutionsState;
	struct PersistentExecutionState;

	// The calling thread's CPU time in the last persistent Execute, by phase. Diagnostics: a host that
	// reports where an epoch's time goes reads it after Execute returns.
	struct PersistentExecuteTimings {
		double retireUs = 0.0;     // retirement of completed executions
		double admissionUs = 0.0;  // admission, seal and recording-job setup
		double acquireUs = 0.0;    // command-list acquisition
		double recordUs = 0.0;     // batch recording (the calling thread joins the task service)
		double submitUs = 0.0;     // queue submission of the recorded batches
		double commitUs = 0.0;     // admission commit, statistics publication, fence bookkeeping
		uint32_t segments = 0;
		uint32_t batches = 0;
	};
	const PersistentExecuteTimings& LastPersistentExecuteTimings() const noexcept { return m_lastPersistentExecuteTimings; }
	// The calling thread's CPU time in the last Update, by phase. Diagnostics.
	struct PersistentUpdateTimings {
		double passUpdatesUs = 0.0;    // the passes' Update calls
		double beforeCompileUs = 0.0;  // UpdateExecutionContext::beforeCompileFrame
		double prepareUs = 0.0;        // the frame's preparation (PrepareFrame of every pass, bindings, versions, uploads)
	};
	const PersistentUpdateTimings& LastPersistentUpdateTimings() const noexcept { return m_lastPersistentUpdateTimings; }
	// Currency probe (diagnostic, off by default). While on, each synchronous preparation of the main segment compares what a kept
	// recording of it would be keyed on with the previous preparation's: the publication (and whether a slot this epoch binds
	// changed in it), the bindings object, the versioned slots' versions, moved backings and the frame placement. Pass revisions
	// are not part of it. Owner thread.
	struct CurrencyProbeStats {
		uint32_t runs = 0;              // preparations compared with a previous one
		uint32_t firstRuns = 0;         // preparations with no previous one (a new graph)
		uint32_t publication = 0;       // a new publication was selected
		uint32_t publicationBound = 0;  // and it changed a slot this epoch binds
		uint32_t bindings = 0;          // the bindings object differs (it follows the publication and the versioned overlay)
		uint32_t perFrame = 0;          // a per-frame versioned slot is bound (the overlay is new each frame)
		uint32_t versions = 0;          // a versioned slot this epoch binds moved to another version
		uint32_t backings = 0;          // a backing the previous preparation bound moved (relocation)
		uint32_t placement = 0;         // another frame placement version, or another alias plan
		uint32_t any = 0;               // a kept recording would not be current with per-slot checks (publicationBound, bindings on
		                                // the same publication, versions, backings, placement)
		uint32_t anyByIdentity = 0;     // nor with the bindings compared by identity (any new publication counts)
		std::map<std::string, uint32_t> names;  // what changed: "slot:<resource>", "version:<resource>", "moved:<resource id>"
	};
	void SetCurrencyProbe(bool on) noexcept { m_currencyProbe.store(on, std::memory_order_relaxed); }
	bool CurrencyProbeOn() const noexcept { return m_currencyProbe.load(std::memory_order_relaxed); }
	// Since the last call. Owner thread.
	CurrencyProbeStats TakeCurrencyProbeStats();
	// Kept recordings of synchronous executions (GraphHost::SetKeptEpochs). A run that names a kept copy
	// (PersistentExecutionOptions::keptCopy) checks the copy's kept recording after the passes' Update calls and the frame's
	// bindings. When it is current, the run submits it: nothing is prepared or recorded. Otherwise the run prepares and records
	// as before, and keeps its recording for the copy when it is replayable. What the last run did, and why it recorded. Owner
	// thread.
	enum class KeptRunOutcome : uint8_t {
		None,        // the run named no kept copy
		Reused,      // the copy's kept recording was submitted
		Kept,        // recorded, and the recording is kept for the copy
		Unkeepable,  // recorded, and the recording cannot be kept (reason says why)
		// Batch reuse (SetKeptBatchReuse): the current batches of the copy's kept recording were submitted again, and only the
		// stale ones were prepared and recorded (reason names the first stale batch's). The merged recording is kept.
		Partial,
	};
	struct KeptRun {
		KeptRunOutcome outcome = KeptRunOutcome::None;
		// Kept: why the run recorded, the first condition of the copy's kept recording that failed ("revision: <pass>",
		// "version: <resource>", "frame placement", "binding: <resource>", ...), or "first run" when the copy had none.
		// Unkeepable: why the new recording cannot be kept ("uncheckable: <pass> (no revision hash)"). The copy's next run then
		// records for "not kept: <that reason>".
		std::string reason;
		// The revision audit (SetKeptRecordingAudit), on a reused run it audited: the passes whose frame data was compared, the
		// ones whose frame data cannot be compared, and the passes that prepared other frame data than the kept recording's
		// while their revision was the same (a revision that misses an input).
		bool audited = false;
		uint32_t auditCompared = 0;
		uint32_t auditUncomparable = 0;
		std::vector<std::string> auditMismatches;
		// The copy's kept recording was a staged one (RecordStagedCopy) at its first use: Reused when its frame data matched this
		// run's preparation, else Kept with reason "staged: ...".
		bool staged = false;
		// Batch reuse (SetKeptBatchReuse): the batches submitted from the kept recording and the batches recorded by this run, and
		// why each recorded batch recorded or cannot be kept, by reason (the first pass of the batch that failed its check, or has
		// no revision). A run that records every batch counts them all as recorded.
		uint32_t batchesReused = 0, batchesRecorded = 0;
		std::vector<std::pair<std::string, uint32_t>> batchReasons;
	};
	const KeptRun& LastKeptRun() const noexcept { return m_lastKeptRun; }
	/**
	 * @brief The revision audit of kept recordings (O-M6-7), off by default (0). Every everyN-th reuse of a kept copy also prepares
	 * the epoch's passes, compares each pass's frame data with the kept recording's (PreparedPass::AuditFrameData), and reports
	 * the passes that differ while their revision says equal (KeptRun::auditMismatches, a warning per pass the first time). The
	 * prepared result is discarded (abandoned): the run still submits the kept recording. Frame data is kept with a recording
	 * only while the audit is on, so a recording kept before it was turned on is not compared. It sees only frame data: what a
	 * pass's Record reads from elsewhere (a manager's state, a pipeline's payload) is outside it. Owner thread.
	 */
	void SetKeptRecordingAudit(uint32_t everyN) noexcept { m_keptAuditEvery = everyN; }
	uint32_t KeptRecordingAudit() const noexcept { return m_keptAuditEvery; }
	/**
	 * @brief Batch reuse of kept recordings (O-M6-6), off by default. A kept recording is then checked per batch: a batch whose
	 * passes all report revisions that still hold, and whose packet is replayable, is submitted again from the kept recording; the
	 * other batches' passes are prepared and recorded, and the merged recording is kept (KeptRunOutcome::Partial). A batch body
	 * depends only on its own passes and the compiled graph (entry barriers are recorded at submission), so an epoch with passes
	 * that cannot report revisions still reuses the batches without them. What every batch shares is still checked for the whole
	 * recording first, and a failure there records every batch: the build, frame placement, publication, versions, bindings,
	 * descriptor heaps, backings and foreign timelines. The audit (SetKeptRecordingAudit) compares the reused batches' passes. A
	 * staged recording (RecordStagedCopy) is confirmed whole, as before. Owner thread.
	 */
	void SetKeptBatchReuse(bool on) noexcept { m_keptBatchReuse = on; }
	bool KeptBatchReuse() const noexcept { return m_keptBatchReuse; }
	// Drops every kept recording (GraphHost::SetKeptEpochs, when the epoch stops keeping). Owner thread.
	void DropKeptRecordings();
	// Schedule introspection (ScheduleTrace.h), off by default. While on, the owner thread traces each executable it selects
	// (the whole-frame one and each epoch's, once per executable: a steady frame builds nothing) and records what each execution
	// waits for at submission beyond its batches' own waits, publishing snapshots of both. Off costs one relaxed load per
	// preparation, execution and ticket completion. Any thread.
	void SetScheduleTraceEnabled(bool enabled) noexcept;
	bool ScheduleTraceEnabled() const noexcept;
	// Any thread: the latest published snapshot; null while off, or before anything executed with it on.
	std::shared_ptr<const PersistentScheduleSnapshot> GetPersistentScheduleSnapshot() const;
	// The schedule of one epoch of the selected executable (UINT32_MAX: the whole frame's), traced now whether introspection is on
	// or not: its passes in order with what each reads and writes. Null before the graph selected an executable, or for an epoch it
	// has no split for. For an application that checks what its graphs read and write against each other once a graph is built
	// (a graph from GraphHost::BuildEpochGraph has its first schedule). Any thread that owns the graph or holds it unattached.
	std::shared_ptr<const ScheduleTrace> TraceSelectedSchedule(uint32_t epoch) const;
	// What happens when a frame's binding edits change which slots share a backing (a resource swapped onto a backing
	// another slot already uses, say). The compiler treats slots that share a backing as one resource, so this needs a
	// new schedule, with the passes of both slots ordered and transitioned against each other.
	// - Inline (default): the frame builds the new schedule before it prepares. The swap takes effect this frame.
	// - Deferred: the frame keeps those slots' current bindings, and a worker builds the new schedule with the new ones.
	//   They take effect when it installs, a frame or more later. Until then the graph still binds and orders the old
	//   backings: a pass that reads its resources through declared views sees the old ones, consistently. A pass that
	//   reads a resource's live descriptors directly would see the new backing without its barriers, and data written
	//   into the new backing on the CPU isn't read until the install.
	// Either way, edits replayed onto a worker build at its install that change these sets build inline.
	enum class SharingChangePolicy : uint8_t { Inline, Deferred };
	void SetSharingChangePolicy(SharingChangePolicy policy) noexcept { m_sharingChangePolicy = policy; }
	SharingChangePolicy GetSharingChangePolicy() const noexcept { return m_sharingChangePolicy; }
	// The frame-wait timeline: a timeline a host's producer of per-frame values signals, once per frame, after the values it
	// wrote for that frame; every batch of an execution waits for the value set for it (SetFrameWaitValue here, or
	// SubmitPersistentTicket's frameWait), so a late producer costs GPU idle, never a CPU wait. Set before the first execution:
	// every packet prepared from then on can wait on it.
	static constexpr uint64_t kFrameWaitTimelineIdentity = UINT64_MAX - 1;
	void SetFrameWaitTimeline(std::shared_ptr<rhi::TimelinePtr> timeline) { m_frameWaitTimeline = std::move(timeline); }
	bool HasFrameWaitTimeline() const noexcept { return m_frameWaitTimeline && *m_frameWaitTimeline; }
	// The synchronous executions' wait (Execute): until changed. 0: none.
	void SetFrameWaitValue(uint64_t value) noexcept { m_frameWaitValue = value; }
	// Only the batches that declare a late-bound buffer ready at the frame wait (VersionedBuffer::LateReady::FrameWait) wait for
	// it (GraphHost::Desc::scopedFrameWait). Off: every batch does.
	void SetScopedFrameWait(bool scoped) noexcept { m_scopedFrameWait = scoped; }
	bool ScopedFrameWait() const noexcept { return m_scopedFrameWait; }

	// Tickets (GraphHost's async epochs) and their recordings, which the host's private calls below take and return.
	struct PersistentTicket;
	// A segment's recorded packets and what they were recorded for (the publication, bindings, heaps, host tag and every pass's
	// revision). Shared, immutable once made.
	struct PersistentRecording;
	struct PersistentTicketSubmission {
		bool submitted = false;
		std::vector<uint64_t> signals;  // per batch of every segment, in order: the value it signalled
		// The handoff and release lists the submission recorded: the host keeps them until its slot completes.
		std::vector<std::shared_ptr<const void>> leases;
		// An external epoch's ticket (SetExternalPersistentEpochs): its graphics batches were not submitted; they are the host's
		// to execute, segment by segment, with their waits and signals (the values above among them).
		std::shared_ptr<ExternalEpochExecution> external;
	};

	// Host-only (GraphHost, a friend): epochs, tickets, kept recordings and the settings they run under. An application
	// drives graphs through GraphHost or GraphInstance, never through these.
private:
	// Host epochs (persistent execution). A host that submits the graph at several
	// points of its own frame tags each pass with its epoch (ExternalPassDesc::Epoch)
	// and names the execution order once. The graph compiles over the whole frame, so
	// scheduling and transient alias placement see every lifetime; each execution then
	// prepares, admits, records and submits only the current epoch's passes (and the
	// untagged ones). Changing the order is structural.
	void SetPersistentEpochOrder(std::vector<uint32_t> order) { m_persistentEpochOrder = std::move(order); }
	// External home states (persistent::SynchronousAdmission::SetExternalHomeStates), an interop option: every execution gives
	// each host-owned texture it touches (Resource::HostOwned) back in its home state on the graphics queue. Set before the first
	// frame.
	void SetPersistentExternalHomeStates(bool externalHome) noexcept { m_persistentExternalHomeStates = externalHome; }
	bool PersistentExternalHomeStates() const noexcept { return m_persistentExternalHomeStates; }
	// The alias pools the graph places its transients into (PersistentAliasArena), shared with the other graphs given the same
	// arena. Without one the graph has pools of its own. Set before the first frame.
	void SetPersistentAliasArena(std::shared_ptr<PersistentAliasArena> arena) noexcept { m_persistentAliasArena = std::move(arena); }
	// The graph's transients go in ranges of a frame alias arena reserved for span (FrameAliasArena), one range per pool, instead
	// of the persistent alias arena's pools. lane: the graph's epoch, so its builds may share bytes. Set before the first frame.
	void SetFrameAliasRegion(std::shared_ptr<FrameAliasArena> arena, AliasSpan span, uint32_t lane) noexcept {
		m_frameAliasArena = std::move(arena);
		m_frameAliasSpan = span;
		m_frameAliasLane = lane;
	}
	// Frame transients (GraphHost::AddFrameTransients): called at each alias plan with every resource the graph's passes use,
	// before the plan places or binds anything, so before the graph executes. The host derives the frame transients' spans from
	// it. It may throw: the plan fails then. Set before the first frame.
	using FrameTransientObserver = std::function<void(std::span<Resource* const>)>;
	void SetFrameTransientObserver(FrameTransientObserver observer) { m_frameTransientObserver = std::move(observer); }
	// Usage summaries: called at the end of each alias plan with the plan's candidates (PlacementSummary), on the thread that
	// plans. The host keeps the latest summary of each epoch and places transients across the frame from them. It must not
	// throw. The graph fills in epoch, ranks, uses and pooled bytes; the host fills in memory. Set before the first frame.
	using PlacementSummaryObserver = std::function<void(PlacementSummary)>;
	void SetPlacementSummaryObserver(PlacementSummaryObserver observer) { m_placementSummaryObserver = std::move(observer); }
	// Frame placement (GraphHost::FramePlacementSettings::enabled). The graph's first alias plan asks source for the host's
	// current version and keeps it until the host publishes a newer one (SetFramePlacementProbe). Each plan
	// places the candidates the version covers at its offsets in its blocks. The rest go to the graph's own pools, as without
	// it. A null source or version: the graph places on its own. Set before the first frame.
	using FramePlacementSource = std::function<std::shared_ptr<const FramePlacementVersion>()>;
	void SetFramePlacementSource(FramePlacementSource source) { m_framePlacementSource = std::move(source); }
	// The id of the version source would return now (0 for none). Each preparation compares it with the version the graph
	// placed from. When they differ, the graph places its aliases again, from the new version, in a structural edit. So every
	// graph moves to a new version before it records against it, and a resource that several epochs share never reaches a
	// graph as a plain binding change. It must be cheap: it runs once per preparation.
	using FramePlacementProbe = std::function<uint64_t()>;
	void SetFramePlacementProbe(FramePlacementProbe probe) { m_framePlacementProbe = std::move(probe); }
	// A plan before the graph is attached (PlaceDeferredAliasesAtBuild) whose new version would move resources the running graphs
	// bind asks this to keep the version it had instead. True: it does (its new transients go to its own pool until a later version
	// takes them). Unset, or false: the placement is left to the attach.
	using FramePlacementDecline = std::function<bool()>;
	void SetFramePlacementDecline(FramePlacementDecline decline) { m_framePlacementDecline = std::move(decline); }
	// Whether the host places transients across the frame (GraphHost::FramePlacementSettings::enabled), even before its first
	// version. Only then are the current versions of versioned slots alias candidates. A versioned resource is often used by
	// several graphs, and a graph's own pools see only its own uses. So a version is placed only at a frame placement's offset,
	// and keeps the backing it has otherwise. Unset or false: versioned slots are never candidates. Set before the first frame.
	using FramePlacementEnabled = std::function<bool()>;
	void SetFramePlacementEnabled(FramePlacementEnabled enabled) { m_framePlacementEnabled = std::move(enabled); }
	// What the graph's latest alias plan took from a frame placement. Any thread.
	struct FramePlacementUse {
		bool planned = false;   // the graph has made an alias plan
		uint64_t version = 0;   // the version it placed from (FramePlacementVersion::Id), 0 for none
		uint64_t placed = 0;    // the candidates it placed from it
	};
	FramePlacementUse FramePlacementInUse() const noexcept {
		return {m_framePlacementPlanned.load(std::memory_order_acquire), m_framePlacementVersion.load(std::memory_order_acquire),
			m_framePlacementPlaced.load(std::memory_order_acquire)};
	}
	// False while some of its placements are in a pool generation another graph has replaced (the pool grew): placed again, by a
	// rebuild, they share the pool's memory again.
	bool PersistentAliasPlacementsCurrent() const;
	// From now on the graph keeps the backings it bound for the transients it placed, even when a resource now points at another backing.
	// For a graph about to be replaced while its successor is recorded (GraphHost::StageEpochGraph): the successor's alias placement
	// may move those transients, and this graph's kept recordings still bind the old backings. Its new preparations would read the
	// moved resources' descriptors, so it records nothing new after this. Any thread that does not prepare the graph meanwhile.
	void FreezeAliasedBindings();
	// The versioned resources the graph's passes declare, by IVersionedResource::Key, sorted. The graph's first build sets them
	// and each preparation keeps them current. Any thread may read them. Keys only: nothing is kept alive.
	class DeclaredVersions {
	public:
		std::shared_ptr<const std::vector<const void*>> Keys() const noexcept { return m_keys.load(std::memory_order_acquire); }
		bool Declares(const void* key) const noexcept;
		// Whether a slot of `key` was lowered for another slice count than `slices` (array-size versions, VersionedResource).
		bool DeclaresOtherSlices(const void* key, uint32_t slices) const noexcept;
	private:
		friend class RenderGraph;
		std::atomic<std::shared_ptr<const std::vector<const void*>>> m_keys;
		// Each versioned slot's key and slice count, sorted. A key has more than one entry when slots of two counts are declared.
		std::atomic<std::shared_ptr<const std::vector<std::pair<const void*, uint32_t>>>> m_slices;
	};
public:
	// Any thread. The same object for the graph's life. Hosts and tests read what a built graph declares.
	std::shared_ptr<const DeclaredVersions> PersistentDeclaredVersions() const noexcept { return m_declaredVersions; }

private:
	// Owner thread: whether RecordTakenUploads would record anything now (IUploadService::HasPendingUploads). A host that
	// gets false skips its upload lists and RecordTakenUploads for the ticket.
	bool HasPendingUploads() const;

	// Tickets (async epochs, GraphHost::EnableAsyncEpochs). The graph's owner thread prepares an
	// epoch's main execution ahead of its epoch point - Update, invocations, recording - into a ticket, for a
	// frame slot it has waited for; its body is recorded from a plan (SynchronousAdmission::PreparePlan), not an
	// admission. The host's submitting thread later checks the ticket is still current, admits it
	// (AdmitPersistentTicket: against the ledgers as they are then, under the sequencer, which it holds through
	// the submission), records the uploads queued since (RecordTakenUploads) and submits the ticket
	// (SubmitPersistentTicket: the execution's handoff and release lists, then its packets, then the commit),
	// assigning timeline values in submission order. The owner then completes the ticket (its statistics and
	// retirement). No upload pass but the host's.
	/**
	 * @brief Owner thread, while no ticket is prepared (the host's async epochs stopped): the persistent epochs whose graphics
	 * work the host executes itself, in its own command buffers (ExternalEpochExecution). Their tickets record their graphics
	 * batches into secondary lists, and a pass of theirs recorded at execution (ExternalPassDesc::recordAtExecution) is recorded by
	 * the host's execution; it must run on the graphics queue. Their batches on other queues are recorded and submitted as any
	 * other ticket's. A recording of one with a pass recorded at execution is never kept for reuse.
	 */
	void SetExternalPersistentEpochs(std::vector<uint32_t> epochs);
	bool IsExternalPersistentEpoch(uint32_t epoch) const noexcept;
	/**
	 * @brief Before CompileStructural: a pass recorded at execution (ExternalPassDesc::recordAtExecution) of an epoch that is not
	 * external is recorded inline. Its late batch is recorded when the execution submits it, on the submitting thread (a ticket's:
	 * after the run's commit), into a primary list of its queue's shared pool, with the admission's barriers before and after it,
	 * and that list is submitted in the batch's place, with the batch's waits and signals. No secondary list, no extra submission,
	 * any queue. For a host that owns its queues (no DXVK stream to run external epochs in). A kept recording's late batch is
	 * recorded again at each submission (PreparedPass::RecordAgain), into the same list. A replayable late batch whose passes are
	 * all keyed by their skip answers (PreparedPass::KeyedBySkip: gated copies and clears) submits its list again while those
	 * answers are unchanged, instead of recording it. Off (the default): such a pass must be an external epoch's.
	 * GraphHost::SetInlineAtExecution sets it on every graph it builds.
	 */
	void SetInlineAtExecution(bool enabled) noexcept;
	bool InlineAtExecution() const noexcept { return m_inlineAtExecution; }
	// A queue slot the host's stream runs: the graphics queue's, or one on the same native queue (a kind the device aliases to it).
	// An external epoch's batches on it are the host's to execute, in its stream's order.
	bool InHostStream(uint32_t slot) const noexcept;
	// The secondary lists an external epoch's batches on that slot record into.
	std::shared_ptr<CommandListPool> ExternalSecondaryPool(uint32_t slot);
	// Owner thread. run: the run it is prepared for (GraphHost's run lease, or a published copy's run). Its lease index is the
	// frame index of the run's statistics range, upload pages and preparation data, and the run is what its recording's
	// RecordingContext::Run names. hostFrame numbers the execution (its statistics and lifecycle). hostTag: the host's currency
	// tag the ticket is prepared under (SetPersistentTicketHostTag). hostData: what the passes prepare for
	// (FramePreparationContext::preparationData), e.g. a scene revision; retained while it records. admissionOnly: the ticket
	// carries the admission alone, and its recording is bound before submission (BindPersistentTicketRecording) - an epoch
	// recorded ahead, for a revision (GraphHost::RequestEpochRecording).
	std::shared_ptr<PersistentTicket> PreparePersistentTicket(rhi::Device device, uint32_t epoch, RunRef run, uint64_t hostFrame,
		uint64_t hostTag = 0, std::shared_ptr<const IHostExecutionData> hostData = {}, bool admissionOnly = false);
	// The ticket's recording (one segment), or null; the ticket keeps it.
	static std::shared_ptr<const PersistentRecording> PersistentTicketRecording(const PersistentTicket& ticket) noexcept;
	// Whether it may be submitted again and again (every pass replayable, recorded for resubmission).
	static bool PersistentRecordingReplayable(const PersistentRecording& recording) noexcept;
	// Any thread: whether every versioned slot it bound still has the version it was recorded with as its current one.
	static bool PersistentRecordingVersionsCurrent(const PersistentRecording& recording) noexcept;
	// Any thread: the first versioned slot whose current version is not the one the recording bound, named with both version
	// numbers. Empty when every slot is current.
	static std::string PersistentRecordingStaleVersion(const PersistentRecording& recording);
	// Any thread: whether no backing the recording bound has moved since (relocation::Service). A recording that bound a backing
	// that moved still holds it, so it stays safe to submit; but it keeps the old memory alive and reads the old copy, so a
	// kept recording that is not current is recorded again.
	static bool PersistentRecordingBackingsCurrent(const PersistentRecording& recording) noexcept;
	// Any thread: the first resource whose bound backing moved, by global ID. Empty when every backing is current.
	static std::string PersistentRecordingStaleBacking(const PersistentRecording& recording);
	// Submitting thread: binds a recording made ahead (PreparePersistentTicket for the same epoch and slot) to an admission-only
	// ticket. It must have been recorded for the ticket's publication, bindings, descriptor heaps and hostTag (the host's backing
	// version now); otherwise false, with why, and the ticket unchanged. A revision ticket (PrepareRevisionTicket) instead takes the
	// candidate admitted for exactly this recording, which needs no tag: the recording holds every backing it was recorded with.
	static bool BindPersistentTicketRecording(PersistentTicket& ticket, std::shared_ptr<const PersistentRecording> recording, uint64_t hostTag,
		std::string* why = nullptr);
	// Revision-driven epochs (immutable inputs). A recording made for host data (a revision) resolves the graph's resolvers through
	// it (ResolverCaptureContext of IHostExecutionData), so its publication and bindings are the revision's own, and it holds them.
	// Owner thread: a ticket for a published copy's run that prepares nothing from live state: its admissions are its candidates',
	// one per recording it may be submitted with (AddPersistentTicketCandidate); the bind takes one, completion abandons the rest.
	// retire: release the executions the GPU has finished first. A host whose own loop retires them already passes false.
	std::shared_ptr<PersistentTicket> PrepareRevisionTicket(uint32_t epoch, RunRef run, uint64_t hostFrame, bool retire = true);
	// Owner thread: admits the recording's publication as a candidate of the revision ticket. Lock-free against a bind on the
	// submitting thread (an append-only list); a recording published to that thread after this call is always found.
	void AddPersistentTicketCandidate(PersistentTicket& ticket, std::shared_ptr<const PersistentRecording> recording);
	// Replayable recordings. A ticket's recording, when every pass of it is replayable (PreparedPass::IsReplayable) and its
	// execution has nothing late-bound, is recorded for resubmission. Only a recording made for host data (a revision's
	// published copy, GraphHost::RequestEpochRecording) is submitted again. A live ticket's recording is used once, and nothing
	// keeps it. Replayable recordings have no profiler GPU zones, and no GPU statistics queries in the shared pool (recreated
	// as passes register, and indexed by run): with SetPersistentRecordingTimestamps, each records into queries of its own.
	void SetPersistentRecordingReuse(bool reuse) noexcept { m_persistentRecordingReuse = reuse; }
	bool PersistentRecordingReuse() const noexcept { return m_persistentRecordingReuse; }
	// Pass and barrier-batch timestamps in recordings kept for resubmission (off by default): each such list records into
	// queries it owns (IStatisticsService::CreateOwnedTimestamps), resolved by every submission and read when its frame slot
	// completes (IStatisticsService::GetCompletedTimestamps). Read when a list is recorded: set it before the graph records
	// (GraphHost::SetRecordingTimestamps rebuilds the graph for it).
	// Lists: one range per kept list (its entry barriers to its exit), no pass or barrier ranges: a stage's GPU time at about
	// one drain per list instead of one per range.
	void SetPersistentRecordingTimestamps(bool enabled) noexcept {
		m_persistentRecordingTimestamps = enabled ? RecordingTimestampMode::Passes : RecordingTimestampMode::Off;
	}
	void SetPersistentRecordingTimestampMode(RecordingTimestampMode mode) noexcept { m_persistentRecordingTimestamps = mode; }
	bool PersistentRecordingTimestamps() const noexcept { return m_persistentRecordingTimestamps != RecordingTimestampMode::Off; }
	RecordingTimestampMode PersistentRecordingTimestampMode() const noexcept { return m_persistentRecordingTimestamps; }
	struct PersistentRecordingStats {
		uint64_t recorded = 0;    // tickets recorded (owner thread)
		// Tickets admitted at submission with their prediction's plan (the ledger held what the executable found last), and those
		// planned against the ledger (SynchronousAdmission::TakePredictionCounts).
		uint64_t predicted = 0, planned = 0;
	};
	// Since the last call (any thread).
	PersistentRecordingStats TakePersistentRecordingStats() noexcept;
	// Returns the last value the submission signals on each queue slot it used (empty when abandoned).
	std::vector<std::pair<uint32_t, uint64_t>> CompletePersistentTicket(PersistentTicket& ticket, const PersistentTicketSubmission& submission);
	// GPU completion values for one pass of retirement over several graphs. Each value is read once per timeline, because
	// each read is a driver call and the host's graphs share their queue timelines. Clear it before each pass.
	class CompletedTimelineValues {
	public:
		uint64_t Of(rhi::Timeline& timeline) {
			for (const auto& [known, value] : m_values)
				if (known == &timeline) return value;
			const uint64_t value = timeline.GetCompletedValue();
			m_values.emplace_back(&timeline, value);
			return value;
		}
		void Clear() noexcept { m_values.clear(); }
	private:
		std::vector<std::pair<const rhi::Timeline*, uint64_t>> m_values;
	};
	// Owner thread: releases executions the GPU has finished (the retirement Execute does before submitting). A graph none of
	// whose executions the GPU has reached returns at once. completed: the values read already in this pass, or null.
	void RetirePersistentExecutions(CompletedTimelineValues* completed = nullptr);
	// Submitting thread; they touch nothing the owner thread uses.
	static uint32_t PersistentTicketEpoch(const PersistentTicket& ticket) noexcept;
	// The run it was prepared for (PreparePersistentTicket, PrepareRevisionTicket).
	static RunRef PersistentTicketRun(const PersistentTicket& ticket) noexcept;
	static uint64_t PersistentTicketHostFrame(const PersistentTicket& ticket) noexcept;
	// The execution the ticket prepared. It is submitted, or poisoned, when the ticket completes (CompletePersistentTicket).
	static ExecutionHandle PersistentTicketExecution(const PersistentTicket& ticket) noexcept;
	// A value the host keeps with the ticket for its own currency checks (GraphHost: the backing version it was
	// prepared under). Set by the owner thread before the ticket is published.
	static uint64_t PersistentTicketHostTag(const PersistentTicket& ticket) noexcept;
	static void SetPersistentTicketHostTag(PersistentTicket& ticket, uint64_t tag) noexcept;
	// After RecordTakenUploads planned the ticket's uploads by queue: its batches off the graphics queue that will not wait for
	// the graphics queue's uploads (statistics); 0 when not planned.
	static uint32_t PersistentTicketSparedUploadWaits(const PersistentTicket& ticket) noexcept;
	// After RecordTakenUploads planned the ticket's uploads by queue (statistics; zeros when not planned):
	// - lateRouted, lateWaits: its destinations that are a late-bound version's backing (VersionedBuffer::CreateLateBound) that went to a
	//   declaring batch's queue, and those a batch off the graphics queue waits for;
	// - undeclared: its destinations no batch declares. Each one made every batch off the graphics queue wait.
	struct TicketUploadStats {
		uint32_t lateRouted = 0, lateWaits = 0, undeclared = 0;
	};
	static TicketUploadStats PersistentTicketUploadStats(const PersistentTicket& ticket) noexcept;
	// Every pass it prepared would still prepare the same invocation (RenderGraphPass::RevisionHash).
	static bool PersistentTicketCurrent(const PersistentTicket& ticket);
	// Why a ticket is not current (diagnostics): the first unchecked pass, or the first pass whose revision changed.
	static std::string PersistentTicketStaleReason(const PersistentTicket& ticket);
	// Submits the packets with values from nextQueueValues (the last value assigned per queue slot, advanced
	// here). Returns false when a packet failed.
	// Consecutive batches on one queue go to that queue in one submission, which waits for all their waits and signals the last
	// batch's value. Their order and the barriers between them are unchanged. A ticket that is not external puts the entry
	// barriers of a submission's batches in one handoff list ahead of its first batch. Barriers for a queue-family transfer, and
	// those that discard or start from Undefined, stay ahead of their own batch.
	// uploadsDone: the queue slot and value that signal the uploads recorded ahead of the ticket (RecordTakenUploads), when
	// there are any. They precede the ticket on their own queue only, so a batch on another queue waits for them: every such
	// batch, unless RecordTakenUploads planned the ticket's uploads by queue (queueLists), and then only a batch that uses
	// what they write. The uploads it recorded on another queue go to that queue just ahead of its first batch there.
	// frameWait: the frame-wait timeline's value (SetFrameWaitTimeline) every batch waits for; 0 for none.
	// An external epoch's ticket (out.external) submits its batches off the graphics queue here and gives the rest to its host
	// (ExternalEpochExecution). streamUploads: its graphics-queue uploads were recorded for the host's stream, ahead of its first
	// segment. A batch off the graphics queue that uses them waits for a segment of their own (the first).
	// carry: several tickets submitted back to back, holding the sequencer across them, share queue submissions. This
	// ticket's first group of batches joins the group the carry holds when both are on one queue and the new group
	// waits for nothing the held one does not. Any other submission of this ticket submits the held group first. carryOut: this
	// ticket's last group is held in the carry instead of submitted, for the next ticket to join. FlushSubmitCarry submits it.
	struct SubmitCarry;
	// The ticket's upload lists, recorded by the caller, for SubmitPersistentTicket to submit before its batches. They signal `signal`
	// on their queue's timeline. They go in the ticket's first queue submission when they can; otherwise in a submission of their
	// own, before anything else of the ticket.
	struct TicketUploads {
		std::span<const rhi::CommandList> lists;
		rhi::Queue queue{};
		uint32_t queueSlot = UINT32_MAX;
		rhi::TimelinePoint signal{};
		bool withBatches = false;  // set when they went in the ticket's first submission
	};
	bool SubmitPersistentTicket(PersistentTicket& ticket, std::span<uint64_t> nextQueueValues, PersistentTicketSubmission& out,
		std::pair<uint32_t, uint64_t> uploadsDone = {UINT32_MAX, 0}, uint64_t frameWait = 0, bool streamUploads = false,
		SubmitCarry* carry = nullptr, bool carryOut = false, TicketUploads* uploads = nullptr);
	// A carry for SubmitPersistentTicket, and its submission. FlushSubmitCarry submits the group the carry holds (nothing when it
	// holds none). Its batches were committed as submitted when they were held. False when the queue refused it. merged: how many
	// tickets' groups it held.
	static std::shared_ptr<SubmitCarry> MakeSubmitCarry();
	static bool FlushSubmitCarry(SubmitCarry& carry, uint32_t* merged = nullptr);
	// The tickets whose held group the next ticket's joined, since the last call.
	static uint64_t TakeSubmitCarryJoins(SubmitCarry& carry) noexcept;
	// Submitting thread, holding the sequencer (HoldPersistentSequencer) until the ticket is submitted: admits each of the
	// ticket's segments against the ledgers as they are now. queueValues: what the host has submitted on each queue slot.
	void AdmitPersistentTicket(PersistentTicket& ticket, std::span<const uint64_t> queueValues);
	// An admitted ticket that will not be submitted: its admissions go (holding the sequencer).
	void AbandonPersistentTicketAdmission(PersistentTicket& ticket);
	std::unique_lock<std::recursive_mutex> HoldPersistentSequencer() const;
	// Ticket uploads, in two parts, for a host whose submissions run on another thread than the commits that queue uploads.
	// TakePendingUploads runs on the upload service's owner thread, right after the commit: it takes everything the service holds
	// for the next submission. Uploads queued later wait for the next take.
	// RecordTakenUploads then records what was taken, later and on any thread, into `list` as plain buffer copies, without the
	// graph's registry or admission: a full barrier (against the work before them), then the copies, whose destinations are left
	// for the next execution's entry barrier. The list ends with a full barrier, so the execution after it on its queue sees the
	// copies. keepAlive owns the staging they read until the GPU is done with the slot. False when an upload is not a
	// pointer-targeted buffer copy: the host fails the ticket.
	// Queue-scoped uploads (ticket and queueLists given): queueLists holds, per queue slot, an open list of that queue (an empty
	// handle for none; the graphics slot's is ignored). A staged upload whose destination the ticket first uses on another queue -
	// with every batch that uses it ordered after that queue's first batch, and nothing in `list` touching it - is recorded into
	// that queue's list instead (after a full barrier against the work before it on that queue), which SubmitPersistentTicket
	// submits just ahead of the queue's first batch, with the waits a write into the destination needs against earlier
	// executions. The plan also records which batches on other queues use what `list` writes: only they wait for uploadsDone.
	// `recorded` then means that `list` itself took a copy; queueCopies, when given, receives the copies recorded into
	// queueLists. Without a ticket or queueLists, everything goes into `list` and every batch on another queue waits.
	struct TakenUploads {
		std::vector<rhi::ResourceHandle> graphicsBuffers;  // what the upload pass's copies write
		std::function<void(rhi::CommandList&)> recordCopies;  // records the upload pass's copies; null when it queued none
		std::shared_ptr<void> keepAlive;                      // the staging those copies read
		std::vector<std::shared_ptr<runtime::StagedUploadBatch>> staged;  // the staged batches, in order
		bool supported = true;  // false: an upload is not a pointer-targeted buffer copy
		bool Any() const noexcept { return recordCopies || !staged.empty(); }
	};
	TakenUploads TakePendingUploads(rhi::Device device, uint32_t slot);
	bool RecordTakenUploads(TakenUploads& taken, rhi::CommandList& list, std::shared_ptr<void>& keepAlive, bool& recorded,
		PersistentTicket* ticket = nullptr, std::span<rhi::CommandList> queueLists = {}, size_t* queueCopies = nullptr);
	// Submitting thread, holding the sequencer, before AdmitPersistentTicket: each texture the staged uploads write gets its
	// ledger state, Common on the graphics queue, as its copies leave it (StagedUploadBatch::RecordCopies). The ticket's own
	// entry barriers then start from there. Throws when a queue has used one of the textures already.
	void SeedTakenTextureUploads(const TakenUploads& taken, uint32_t graphicsSlot);
	// The epoch the next Update/Execute runs; persistent::AllEpochs runs every pass.
	void SetPersistentEpoch(uint32_t epoch) noexcept { m_persistentEpoch = epoch; }
	uint32_t PersistentEpoch() const noexcept { return m_persistentEpoch; }
	// Whether a pass of that epoch takes part in the current execution.
	bool RunsInPersistentEpoch(uint32_t passEpoch) const noexcept {
		return m_persistentEpoch == UINT32_MAX || passEpoch == UINT32_MAX || passEpoch == m_persistentEpoch;
	}

public:
	// The group held for the next ticket, submitted now if this thread is recording an inline late batch while one is held (else
	// nothing). Whatever submits on a queue while a pass records at execution calls it first: the held batches are committed as
	// submitted, so nothing may come ahead of them on the queue (a wait on a value they signal would never end).
	static bool FlushRecordingSubmitCarry();
	// Hosts that share a queue with another API (for example an adopted DXVK
	// VkQueue whose translated D3D11 work is submitted before and after the graph)
	// order that work against the graph with full memory barriers: `entry` at the
	// start of each queue's first batch in the frame, `exit` at the end of each
	// queue's last batch. Barrier scopes span submissions on the same queue, so no
	// semaphores or per-resource transitions are needed. Honored by persistent
	// execution. A graph queue the host does not share (an async compute queue)
	// is the host's to order: its first submission of an execution after the
	// host's work before the execution, and the host's later work after its last
	// submission's signals.
	struct ExternalQueueBoundary {
		bool entry = false;
		bool exit = false;
	};
	void SetExternalQueueBoundary(ExternalQueueBoundary boundary) noexcept { m_externalQueueBoundary = boundary; }
	ExternalQueueBoundary GetExternalQueueBoundary() const noexcept { return m_externalQueueBoundary; }
	std::optional<uint32_t> GetLastExecutedPreparationSlot() const noexcept;
	std::shared_ptr<const IHostExecutionData> GetLastSubmittedFrameData() const noexcept;
	void PrepareExtensionsForBuild();
	// Ends what PrepareExtensionsForBuild began (IRenderGraphExtension::FinishBuild), on the same thread.
	void FinishExtensionsBuild();
	void ShutdownExtensions();
	// Serialized producer boundary. Joins CPU frame work without releasing
	// submitted ownership; the caller can then wait for every device to idle.
	void StopFrameProduction();
	// Requires GPU idle. Releases frame ownership after CPU production stopped.
	void ShutdownTaskWorkers();
	// Owner thread, after ShutdownTaskWorkers: gives back the alias placements the graph made and its requirements in the
	// shared alias arena, as its destruction would. Then the graph may be destroyed on another thread (GraphHost::Desc::
	// destroyRetiredGraphsDetached): nothing else it holds is placed into, or used, by the graphs still running.
	void ReleaseAliasPlacements();
	// The runtime's services for those this graph has none of (never set, or dropped by a rebuild).
	void UseRuntimeServices();
	void ClearExtensions();
	void RegisterExtension(std::unique_ptr<IRenderGraphExtension> ext, std::optional<std::string_view> id = std::nullopt);
	std::optional<PresentDependency> GetLastPresentDependency() const noexcept { return m_lastPresentDependency; }
	// The latest execution (Execute, or the one a ticket prepared). Empty before the first. Owner thread.
	ExecutionHandle LastExecution() const;
	// The pass that leaves the swapchain image ready to present. The signal of its batch is the present dependency
	// (GetLastPresentDependency). Passes that must run after it anchor on GetPresentPass(). Work after the graph that
	// reads its output (a presentation copy) submits through GetExternalClient(), which orders it after the frame.
	void SetPresentPass(std::string name) { m_presentPass = std::move(name); }
	const std::string& GetPresentPass() const noexcept { return m_presentPass; }
	// The swapchain image. It must be a DynamicResource that points at the image being rendered. The graph treats it as a
	// versioned resource whose version for each preparation is the image the wrapper points at then (SwapchainVersions).
	// Recordings that bind it are never kept for reuse (IVersionedResource::PerFrame).
	void SetSwapchainResource(std::shared_ptr<Resource> resource);
	bool IsSwapchainResource(const Resource* resource) const noexcept { return resource && resource == m_swapchain.get(); }
	// The swapchain as a versioned resource. Null without one.
	const std::shared_ptr<const IVersionedResource>& SwapchainVersions() const noexcept { return m_swapchainVersions; }
	org::memory::SnapshotProvider& GetMemorySnapshotProvider() { return m_memorySnapshotProvider; }
	const org::memory::SnapshotProvider& GetMemorySnapshotProvider() const { return m_memorySnapshotProvider; }
	void WriteVramUsageDebugDumpNow(uint32_t frameIndex) const { WriteVramUsageDebugDump(frameIndex); }
	void SetStatisticsService(std::shared_ptr<org::runtime::IStatisticsService> service) { m_statisticsService = std::move(service); }
	// Whether the graph drives the runtime's frame-indexed services: statistics query slots. One graph on a runtime does
	// (the per-frame one) until those services are per execution (docs/job-graphs.md, Phase 3); a graph executing alongside
	// it (a job graph, on another thread) does not: it records no statistics. Set before Setup.
	void SetDrivesFrameServices(bool drives) {
		m_drivesFrameServices = drives;
		if (!drives) m_statisticsService.reset();
	}
	bool DrivesFrameServices() const noexcept { return m_drivesFrameServices; }
	org::runtime::IStatisticsService* GetStatisticsService() { return m_statisticsService.get(); }
	const org::runtime::IStatisticsService* GetStatisticsService() const { return m_statisticsService.get(); }
	void SetUploadService(std::shared_ptr<org::runtime::IUploadService> service) { m_uploadService = std::move(service); }
	org::runtime::IUploadService* GetUploadService() { return m_uploadService.get(); }
	const org::runtime::IUploadService* GetUploadService() const { return m_uploadService.get(); }
	std::shared_ptr<org::runtime::IUploadService> RetainUploadService() const { return m_uploadService; }
	void SetDescriptorService(std::shared_ptr<org::runtime::IDescriptorService> service) { m_descriptorService = std::move(service); }
	org::runtime::IDescriptorService* GetDescriptorService() { return m_descriptorService.get(); }
	const org::runtime::IDescriptorService* GetDescriptorService() const { return m_descriptorService.get(); }
	// Host shutdown retains the service beyond graph/manager resource owners.
	std::shared_ptr<org::runtime::IDescriptorService> RetainDescriptorService() const { return m_descriptorService; }
	void SetRenderGraphSettingsService(std::shared_ptr<org::runtime::IRenderGraphSettingsService> service) { m_renderGraphSettingsService = std::move(service); }
	org::runtime::IRenderGraphSettingsService* GetRenderGraphSettingsService() { return m_renderGraphSettingsService.get(); }
	const org::runtime::IRenderGraphSettingsService* GetRenderGraphSettingsService() const { return m_renderGraphSettingsService.get(); }
	void SetTaskService(std::shared_ptr<org::runtime::ITaskService> service) {
		m_taskService = std::move(service);
	}
	org::runtime::ITaskService* GetTaskService() { return m_taskService.get(); }
	const org::runtime::ITaskService* GetTaskService() const { return m_taskService.get(); }
	void SetStructuralMaterializeCheckpointCallback(std::function<void(std::string_view)> callback) {
		m_structuralMaterializeCheckpointCallback = std::move(callback);
	}
	void SetStructuralMaterializeResourceCheckpointCallback(std::function<void(std::string_view, std::string_view)> callback) {
		m_structuralMaterializeResourceCheckpointCallback = std::move(callback);
	}
	//void AllocateResources(PassExecutionContext& context);
	//void CreateResource(std::wstring name);
	std::shared_ptr<Resource> GetResourceByName(const std::string& name);
	std::shared_ptr<Resource> GetResourceByID(const uint64_t id);
	std::shared_ptr<RenderPass> GetRenderPassByName(const std::string& name);

	void RegisterProvider(IResourceProvider* prov);
	void EnsureProviderRegistered(IResourceProvider* prov);
	void RegisterResource(ResourceIdentifier id, std::shared_ptr<Resource> resource, IResourceProvider* provider = nullptr);
	// Replace a symbolic resource only when that identifier was originally
	// registered as a dynamic wrapper.  This preserves the alias across later
	// resolver revisions without changing the handle shape of ordinary
	// resolver-backed resources (for example terrain resource sets).
	void RegisterResolvedResourceAlias(ResourceIdentifier const& id, std::shared_ptr<Resource> resource);

	std::unordered_map<ResourceIdentifier, std::shared_ptr<IResourceResolver>, ResourceIdentifier::Hasher> _resolverMap;

	void RegisterResolver(ResourceIdentifier id, const std::shared_ptr<IResourceResolver>& resolver);
	std::shared_ptr<IResourceResolver> RequestResolver(ResourceIdentifier const& rid, bool allowFailure = false);
	void SetPassTechnique(std::string passName, std::string techniquePath);
	// Adds a structural ordering edge from `beforePass` to `afterPass`.
	// Both passes must exist when the graph is compiled.
	void AddExplicitPassDependency(std::string beforePass, std::string afterPass);

	// The resource registered under rid (RegisterResource), or null. Unlike RequestResourcePtr it asks no provider and is not
	// a lookup (WasLookedUp), so a host can check a registration before it adds its own. The building thread only.
	std::shared_ptr<Resource> RegisteredResource(ResourceIdentifier const& rid) const { return _registry.RequestShared(rid); }
	// The graph's registry: what its passes' registry handles resolve in. A host of several graphs points the shared upload
	// service at the registry of the graph it prepares (GraphHost::ExecuteFrame).
	ResourceRegistry& GetResourceRegistry() noexcept { return _registry; }
	std::shared_ptr<Resource> RequestResourcePtr(ResourceIdentifier const& rid, bool allowFailure = false);
	ResourceRegistry::RegistryHandle RequestResourceHandle(ResourceIdentifier const& rid, bool allowFailure = false);
	ResourceRegistry::RegistryHandle RequestResourceHandle(Resource* const& pResource, bool allowFailure = false);
	/**
	 * @brief Any thread: whether this graph looked rid up by name (RequestResourcePtr, RequestResourceHandle), found or not. A host
	 * that changes what a name means rebuilds only the graphs that looked it up (GraphHost::RequestRebuildFor).
	 */
	bool LookedUp(ResourceIdentifier const& rid) const;
	// Intern the concrete resources from one coherent resolver snapshot once per
	// resource set.  Retained passes commonly share the same resolver and would
	// otherwise repeat thousands of registry lookups during a loading frame.
	std::shared_ptr<const std::vector<ResourceHandleAndRange>>
		RequestResolverResourceHandles(const ResolverDeclarationState& state);
    // Within a ResolverCaptureScope, every capture of one resolver (by dependency identity) returns the scope's first.
    std::shared_ptr<const ResolverDeclarationState> CaptureResolverDeclarationState(const IResourceResolver& resolver) const;
    using ResolverCaptureScope = org::ResolverCaptureScope;

	//void RegisterECSRenderPhaseEntities(const std::unordered_map<RenderPhase, flecs::entity, RenderPhase::Hasher>& phaseEntities);

	template<DerivedResource T>
	std::shared_ptr<T> RequestResourcePtr(ResourceIdentifier const& rid, bool allowFailure = false) {
		auto basePtr = RequestResourcePtr(rid, allowFailure);

		if (!basePtr) {
			if (allowFailure) {
				return nullptr;
			}

			throw std::runtime_error(
				"RequestResource<" + std::string(typeid(T).name()) +
				">: underlying Resource* is null (rid = " + rid.ToString() + ")"
			);
		}

		auto derivedPtr = std::dynamic_pointer_cast<T>(basePtr);
		if (!derivedPtr) {
			throw std::runtime_error(
				"Requested resource is not a " + std::string(typeid(T).name()) +
				": " + rid.ToString()
			);
		}

		return derivedPtr;
	}

	template<typename PassT, org::PassInputs InputsT, typename... StableCtorArgs>
	RenderPassBuilder& BuildRenderPass(std::string const& name, InputsT&& inputs, StableCtorArgs&&... ctorArgs);

	template<typename PassT, typename... StableCtorArgs>
	RenderPassBuilder& BuildRenderPass(std::string const& name, StableCtorArgs&&... ctorArgs);

	// Unified author-facing path. Queue class is a declaration property, not a
	// C++ base-class choice; typed passes select it through PassBuilder.
	template<typename PassT, org::PassInputs InputsT, typename... StableCtorArgs>
	RenderPassBuilder& BuildPass(std::string const& name, InputsT&& inputs, StableCtorArgs&&... ctorArgs) {
		return BuildRenderPass<PassT>(name, std::forward<InputsT>(inputs),
			std::forward<StableCtorArgs>(ctorArgs)...);
	}

	template<typename PassT, typename... StableCtorArgs>
	RenderPassBuilder& BuildPass(std::string const& name, StableCtorArgs&&... ctorArgs) {
		return BuildRenderPass<PassT>(name, std::forward<StableCtorArgs>(ctorArgs)...);
	}
	// The builder of a pass built since the last reset, for settings applied after the fact (placement). Null if none.
	RenderPassBuilder* FindPassBuilder(std::string const& name);
	// The passes built since the last reset, in build order: how many, and the names of those from `first` on. A host
	// that labels the passes a declaration builds (its own epoch tags, say) reads the count before it and the names after.
	size_t PassBuilderCount() const noexcept { return m_passBuilderOrder.size(); }
	std::vector<std::string> PassBuilderNamesFrom(size_t first) const;

	/// Create a new queue of the given kind and register it with the render graph.
	/// Must be called after Setup() and before the first Execute().
	QueueSlotIndex CreateQueue(
		QueueKind kind,
		const char* name,
		QueueAutoAssignmentPolicy autoAssignmentPolicy = QueueAutoAssignmentPolicy::AllowAutomaticScheduling);
	void SetMinimumAutomaticSchedulingQueues(QueueKind kind, uint8_t count);
	uint8_t GetMinimumAutomaticSchedulingQueues(QueueKind kind) const noexcept {
		return m_minAutomaticSchedulingQueuesByKind[static_cast<size_t>(kind)];
	}

	/// Access the queue registry (read-only).
	const QueueRegistry& GetQueueRegistry() const noexcept { return m_queueRegistry; }
	QueueRegistry& GetQueueRegistry() noexcept { return m_queueRegistry; }
	// Before Setup: the primary queues' timelines (graphics, compute, copy, in that order) come from another registry
	// (QueueRegistry::SharedTimelineOf), so this graph's work and the other graph's share them. Executions of graphs on one
	// queue's timeline need no waits on each other there; the ledger drops a wait on the consumer's own timeline. The graph
	// neither waits on nor retires shared timelines when it goes. Empty: timelines of its own.
	void SetSharedQueueTimelines(std::vector<SharedQueueTimeline> timelines) { m_sharedQueueTimelines = std::move(timelines); }
	// Before CompileStructural: the graph takes only the passes of this host epoch (ExternalPassDesc::epoch) and the untagged
	// ones from its extensions, so it compiles, admits and executes that epoch alone (GraphHost: one graph per epoch).
	// persistent::AllEpochs (the default) takes every pass.
	void SetPassEpochFilter(uint32_t epoch) noexcept { m_passEpochFilter = epoch; }
	uint32_t PassEpochFilter() const noexcept { return m_passEpochFilter; }
	// A value that every recording of this graph carries under key (RecordingContext::Tag), such as the pipeline versions its
	// passes bind. Each extension tags under its own key, so several can tag one graph. Set while the graph is built (an
	// extension's PrepareForBuild), before it first executes. A null tag removes the key.
	void SetRecordingTag(const RecordingTagKey& key, std::shared_ptr<const void> tag) {
		const org::RecordingTags& now = m_recordingTags ? *m_recordingTags : org::RecordingTags{};
		m_recordingTags = std::make_shared<const org::RecordingTags>(now.With(key, std::move(tag)));
	}
	// The tag under key, null without one.
	const void* RecordingTag(const RecordingTagKey& key) const noexcept { return m_recordingTags ? m_recordingTags->Find(key) : nullptr; }
	// Every tag, as the graph's recordings carry them. Null before the first SetRecordingTag.
	const std::shared_ptr<const org::RecordingTags>& RecordingTagSet() const noexcept { return m_recordingTags; }
	// Whether the graph's epoch (PassEpochFilter) is an external one, whose graphics work the host runs in its own stream. Its
	// extensions may then declare passes recorded at execution (ExternalPassDesc::RecordAtExecution). GraphHost sets this before
	// it builds the graph, so an extension can read it while it gathers its passes.
	bool PassEpochInHostStream() const noexcept { return IsExternalPersistentEpoch(m_passEpochFilter); }
	// Whether a pass of the graph's epoch declared RecordAtExecution is recorded at execution: in the host's stream (an external
	// epoch) or inline at submission (SetInlineAtExecution). Otherwise an extension records it ahead, as any other pass.
	bool PassEpochRecordsAtExecution() const noexcept { return PassEpochInHostStream() || m_inlineAtExecution; }

	/// The graph's external client (OpenRenderGraph/ExternalClient.h). Use it to submit GPU work recorded outside the
	/// graph, such as uploads and readbacks. It uses this graph's queue slots, and its work is ordered with the graph's
	/// executions in submission order. The graph's own uploads go through it too.
	/// Created on first use. Call it after Setup. Any thread.
	ExternalClient& GetExternalClient();
	// The queue slots its external client has used (ExternalClient::UsedSlots), 0 when it has none yet. Does not make one.
	uint32_t ExternalClientSlots();

	// Owner thread: re-placements installed from a preparation (PrepareFramePlacementReplacement), and preparations the graph did not
	// install, so far.
	std::pair<uint64_t, uint64_t> PreparedReplacementCounts() const;
	// Owner thread: a structural build is in flight (its preparations cannot re-place until it installs).
	bool PersistentStructuralBuildPending() const;
private:
	// Driving the graph is GraphHost's: its declared-graph calls (Build, PrepareFrame, ExecuteFrame)
	// or its own frames.
	friend class GraphHost;
	friend class GraphInstance;
	// The executable of the selected publication. Null before the first preparation. Owner thread.
	std::shared_ptr<const persistent::ExecutableGeneration> SelectedExecutable() const;
	void CompileStructural();
	void ResetForFrame();
	void ResetForRebuild();
	void Setup();
	// Builds off the owner thread (GraphHost::BuildEpochGraph). Setup leaves out the steps that touch objects other graphs share
	// while those graphs run:
	// - sizing the shared statistics query heap,
	// - setting up the named passes, whose pass objects every graph of the runtime shares (the host's upload pass).
	// FinishSharedSetup runs them later, on the owner thread, while nothing else uses those objects. Call before Setup.
	void DeferSharedSetup(std::vector<std::string> sharedPasses) {
		m_sharedSetupDeferred = true;
		m_sharedSetupPasses = std::move(sharedPasses);
	}
	void FinishSharedSetup();
	// The work the first preparation would otherwise do inline: lower every pass, request its pipelines and wait for them, and
	// build and install the first publication (its schedule). After Setup, before the first preparation. Any thread that owns the
	// graph alone. A graph already prepared is left as it is.
	// The alias placement is left to the first preparation (PlaceDeferredPersistentAliases): placing re-materializes the transients,
	// and those resource objects are shared with the graph the epoch still runs (and with any other build of it in flight), whose
	// next preparation would then find its bindings moved under it.
	void PrepareFirstPersistentBuild();
	// The alias placement PrepareFirstPersistentBuild left to the first preparation, run now. Owner thread, once the graph is
	// attached and before its first preparation. Nothing when the graph has placed already. GraphHost calls it at a synchronous
	// host's attach, so the graphs attached together place in epoch order before any of them records.
	void PlaceDeferredAliasesNow();
	// The same placement on the thread that built the graph, before it is staged or attached (GraphHost::PlaceEpochGraphs). It
	// places only when that moves nothing another graph binds: every resource it would place or bind that is materialized by
	// someone else must already be where the plan puts it, and nothing may go to the arena's shared pools. Otherwise it changes
	// nothing, leaves the placement to the attach and returns false with why. True when it placed (or had nothing to place).
	bool PlaceDeferredAliasesAtBuild(std::string* why = nullptr);
	// Before any graph of a staged set places (GraphHost::PlaceEpochGraphs): publishes this graph's usage summary to its staging
	// and changes nothing else, so the version the set's first graph needs is computed from every graph of the set, and the later
	// graphs need no newer one that moves what the first placed. True when it published one. Same thread.
	bool PublishPlacementSummaryAtBuild();
	// For a graph of a staged set that placed from an older version of the set's staging (a later graph of the set made a newer
	// one): takes the staging's current version in place when that moves none of its placements (TakeFramePlacementInPlace), so its
	// first preparation does not place again. A graph left for its attach tries PlaceDeferredAliasesAtBuild again. Same thread.
	bool PlaceAgainAtBuild(std::string* why = nullptr);
	// The frame placement version the graph's latest alias plan placed from, null for none. Owner thread.
	std::shared_ptr<const FramePlacementVersion> PersistentFramePlacement() const;
	// Staged recordings (GraphHost::RecordStagedCopies): on the thread that built the graph, before it is staged or attached,
	// records the epoch's kept copy `copy` for `run` (the copy's lease) and keeps it, as the copy's first run would. The passes'
	// TakeFrameValues run with `frame`'s host data; their Update does not. Nothing is admitted against the ledgers or submitted:
	// the body is recorded from a plan (SynchronousAdmission::PreparePlan), and the copy's first run admits it. That run checks
	// it as any kept recording, and also prepares the passes once and compares their frame data with the staged ones
	// (KeptRun::staged): a difference records instead, named "staged: frame data of <pass>". Returns false with why when the
	// copy cannot be recorded ahead: a pass without a revision, the alias placement left to the attach, a recording that cannot
	// be kept.
	bool RecordStagedCopy(rhi::Device device, const UpdateExecutionContext& frame, uint32_t copy, RunRef run, std::string& why);
	// Owner thread, at a preparation that finds a new frame placement version: takes it without a plan when it leaves every
	// resource this graph placed from a frame placement where it is, with the same contents flow (whether it continues from an
	// earlier epoch, and whether it is this graph's own), and places nothing of this graph's own pools. A version that kept the
	// running graphs' resources (FramePlacementVersion::Keeping) is one. True when it took it.
	// atBuild: the graph has run nothing yet (PlaceAgainAtBuild); a resource now shared with another epoch (or no longer) only changes
	// who gives its placement back. why: what refused it.
	bool TakeFramePlacementInPlace(std::string* why = nullptr, bool atBuild = false);
	// Owner thread, while the host holds a new frame placement version back (FramePlacementSettings::prepareReplacements): starts
	// preparing this graph's re-placement for it on a worker. The worker places what this graph placed from a frame placement where
	// version puts it, derives the placement orderings, gives each resource that moves a backing made ahead (BackedResource::
	// PreparePlacedBacking) and builds the validated publication, while this graph keeps running. The first preparation that finds
	// version current installs it: the backings switch, the publication installs, and no plan or build runs on this thread. When
	// anything it was made from changed meanwhile, that preparation re-places as before. False when there is nothing to prepare (the
	// graph has not placed from a frame placement, has a build in flight, or binds a resource another epoch's graph placed).
	bool PrepareFramePlacementReplacement(std::shared_ptr<const FramePlacementVersion> version);
	// Any thread: no preparation for this version is running (it finished, was refused, or none was started).
	bool FramePlacementReplacementSettled(uint64_t versionId) const;
	// Before CompileStructural, on a graph built detached: the context its passes' resolvers are captured with and its first
	// publication binds versions with. Each preparation replaces it with its own (UpdateExecutionContext::resolverCaptureContext).
	// GraphHost::BuildEpochGraph gives one that names each texture's reshaped version (array-size versions, VersionedResource).
	void SetBuildCaptureContext(std::shared_ptr<const ResolverCaptureContext> context) noexcept { m_resolverCaptureContext = std::move(context); }
	// How the next execution runs (RenderGraphPersistentState.h, PersistentExecution::Options). Plain executions use the
	// defaults; a ticket's set its ticket.
	struct PersistentExecutionOptions {
		PersistentTicket* ticket = nullptr;
		uint64_t hostTag = 0;
		bool admissionOnly = false;
		bool hostData = false;
		// What the recording's RecordingContext::Run names. Without one: UpdateExecutionContext::run, else the execution's frame
		// index outside a lease (RunRef::Unleased).
		std::optional<RunRef> run;
		// A synchronous execution's kept copy (GraphHost::SetKeptEpochs): the run reuses the copy's kept recording when it is
		// current, and keeps its own recording for the copy otherwise. The run must be the copy's own (its latch regions stay at
		// its index), and the copy's last run must be complete. UINT32_MAX: no kept copy.
		uint32_t keptCopy = UINT32_MAX;
		// A staged recording (RecordStagedCopy): no uploads are prepared, and every pass's frame data is kept to confirm it.
		bool staged = false;
	};
	void Update(const UpdateExecutionContext& context, rhi::Device device) { Update(context, device, PersistentExecutionOptions{}); }
	void Update(const UpdateExecutionContext& context, rhi::Device device, const PersistentExecutionOptions& options);
	void Execute(PassExecutionContext& context);
	GpuPassRangeBegin m_gpuPassRangeBegin;
	GpuPassRangeEnd m_gpuPassRangeEnd;
	// Update: the passes' own updates, then the next execution's preparation (PreparePersistentFrame).
	void UpdateOnPreparationOwner(const UpdateExecutionContext& context, rhi::Device device,
		const PersistentExecutionOptions& options);
	std::string GetTechniquePathForPassName(std::string_view passName) const;

	struct AnyPassAndResources {
		PassType type = PassType::Unknown;
		std::variant<std::monostate, RenderPassAndResources> pass;
		std::string name;

		AnyPassAndResources() = default;

		explicit AnyPassAndResources(RenderPassAndResources const& rp)
			: type(PassType::Render), pass(rp) {}
		explicit AnyPassAndResources(RenderPassAndResources&& rp)
			: type(PassType::Render), pass(std::move(rp)) {}

	};
	// Sets up one render pass: its runtime services, its registry view, then its own Setup.
	void SetupRenderPass(AnyPassAndResources& pass);

	


	// Compiler implementation data deliberately lives outside this public header.
	// This keeps compiler algorithm/layout changes from rebuilding all ORG clients.
	struct Node;
	struct CompilerState;
	// Preparation, on the owner thread: BindPersistentFrame polls the passes' declarations and backings and installs the
	// edits they make (a structural one builds on a worker); PreparePersistentExecution then prepares an execution
	// against the selected publication.
	struct PersistentExecution;
	void PreparePersistentFrame(rhi::Device device, uint32_t frameIndex, const IHostExecutionData* hostData, float deltaTime,
		const PersistentExecutionOptions& options);
	void BootstrapPersistentGraph();
	// bootstrap: the first, whose passes were all just declared.
	void BindPersistentFrame(bool bootstrap);
	PersistentExecution PreparePersistentExecution(rhi::Device device, uint32_t frameIndex, const IHostExecutionData* hostData,
		float deltaTime, const PersistentExecutionOptions& options);
	// Execution of the prepared one: its uploads, then its main segment's admission, recording and submission.
	void ExecutePersistentFrame(PassExecutionContext& context);
	void SubmitPersistentUploads(PersistentExecution& execution);
	// One execution of the prepared segment, in phases (RenderGraphPersistentExecute.cpp). Admit returns false when the
	// segment needs nothing more: it was empty, admission-only, or took a kept recording.
	struct PersistentSegmentRun;
	// The execution's waits with those on another device's timelines relayed to the batch's device (docs/pipelines.md, "When pipelines build").
	std::vector<std::vector<compiler::ExecutionTimelinePoint>> RelayCrossDeviceWaits(PersistentSegmentRun& run,
		const std::vector<std::vector<compiler::ExecutionTimelinePoint>>& waits, std::vector<compiler::PreparedTimelineBinding>& extraTimelines);
	bool AdmitPersistentSegment(PersistentSegmentRun& run);
	void RecordPersistentSegment(PersistentSegmentRun& run);
	void SubmitPersistentSegment(PersistentSegmentRun& run);
	// skipAliasCandidates leaves alias-eligible transients unmaterialized so the
	// structural build can place them (see PlanPersistentAliasPlacement).
	void MaterializePersistentStandalone(std::span<Resource* const> resources, bool skipAliasCandidates = false);
	// A resource a pass on another device uses: one shared D3D12 allocation, imported into Vulkan, as its representation on
	// every device (the primary's included). Its descriptors on each device name that device's representation. Kept while
	// the backing lives; the resource's next materialization clears them and the next binding makes new ones.
	void EnsurePersistentRepresentations(Resource& resource);
	// A multi-device resource placed in a shared alias pool (SharedAliasHeaps): its representations are resources placed at
	// `offset` of the pool's D3D12 heap and of its Vulkan import, published with the placement so bindings treat them as aliased.
	void PlaceSharedRepresentations(Resource& concrete, const std::shared_ptr<const AliasHeapGeneration>& heap, uint64_t poolID,
		uint64_t offset, uint64_t size);
	// Whether a multi-device resource can be placed in a shared alias pool: a buffer, or a 2D texture without render target or
	// depth use whose format the Vulkan device can place in an imported D3D12 heap.
	bool IsSharedAliasEligible(Resource* concrete) const;
	// A shared candidate's allocation size and alignment: the larger of the two devices'.
	rhi::ResourceAllocationInfo SharedAliasAllocationInfo(Resource* concrete) const;
	bool IsPersistentAliasCandidate(Resource* concrete) const;
	// Returns true when the edit changed (placement edges or bindings) and needs
	// another build before installation.
	// The memory widget's frame-graph timeline over the selected persistent executable: per batch, the bytes its passes touch
	// (memory: resource ID -> bytes and category) and the live bytes with and without the alias pools.
	void BuildPersistentFrameGraphSnapshot(ui::FrameGraphSnapshot& out,
		const std::unordered_map<uint64_t, std::pair<uint64_t, std::string>>& memory) const;
	// Frees the alias pools no structural plan has placed into for autoAliasPoolRetireIdleFrames frames.
	void RetireIdlePersistentAliasPools(PersistentExecutionState& state);
	bool PlanPersistentAliasPlacement(persistent::GraphEditTransaction& edit, const persistent::SelectedPublication& scheduled);
	// Gives the frame transient observer (SetFrameTransientObserver) every resource the schedule's passes use.
	void ReportFrameTransientUses(const persistent::SelectedPublication& scheduled);
	// The first preparation of a graph PrepareFirstPersistentBuild built: its alias placement, planned against the installed
	// schedule, built and installed (a structural edit). Nothing otherwise.
	void PlaceDeferredPersistentAliases();
	// BindPersistentFrame: installs the prepared re-placement for the current frame placement version (PrepareFramePlacementReplacement).
	bool InstallPreparedReplacement();
	void BindUnboundPersistentEntries(persistent::GraphEditTransaction& edit);
	// Structural build: bind, schedule, plan alias placement, rebuild if needed.
	std::shared_ptr<const persistent::SelectedPublication> BuildPersistentStructural(persistent::GraphEditTransaction& edit);
	// Worker-side structural builds: the transaction is prepared on the owner,
	// its schedule/placed builds run on a task worker while frames keep the
	// selected publication; binding edits made meanwhile are replayed onto the
	// result before a rebased install.
	void SubmitPersistentStructuralBuild(const std::vector<uint32_t>& structural, const std::vector<uint32_t>& growGroups, rhi::Backend primaryBackend,
		bool forceInline = false);
	void AdvancePersistentStructuralBuild(std::vector<uint32_t>& structural);
	// Blocks until the pending structural build is installed (a relower that cannot be prepared against the live publication).
	void DrainPersistentStructuralBuild(std::vector<uint32_t>& structural);
	void RunPersistentStructuralBuildPhase(bool validated);
	// Compile-dump mode for the persistent graph: written once per selected executable.
	void WritePersistentGraphDebugDump(const persistent::SelectedPublication& selected) const;
	// Schedule introspection, owner thread. Traces the selected executable and its epochs' when it is not the traced one.
	void RefreshPersistentScheduleTraces(const persistent::SelectedPublication& selected);
	void ProbePersistentCurrency(const std::shared_ptr<const persistent::SelectedPublication>& selected,
		const persistent::SelectedPublication& executing, std::shared_ptr<const void> bindings,
		const std::vector<std::pair<std::shared_ptr<const IVersionedResource>, uint64_t>>& boundVersions, bool perFrame);
	// Records the submission waits filled into the persistent state's scratch for one execution of `executable` (a segment
	// labelled segmentLabel), and publishes a snapshot once per round of executions.
	void RecordPersistentScheduleSubmission(const persistent::ExecutableGeneration& executable, uint64_t frame);
	void RecordPersistentTicketScheduleSubmissions(const PersistentTicket& ticket);
	void PublishPersistentScheduleSnapshot();
	std::atomic<bool> m_scheduleTraceEnabled{false};
	std::atomic<std::shared_ptr<const PersistentScheduleSnapshot>> m_scheduleSnapshot;
	// PersistentDeclaredVersions. PublishDeclaredVersions refreshes it on the preparing thread when the versioned slots changed.
	std::shared_ptr<DeclaredVersions> m_declaredVersions = std::make_shared<DeclaredVersions>();
	void PublishDeclaredVersions();
	std::vector<uint32_t> m_persistentEpochOrder;
	bool m_persistentExternalHomeStates = false;
	std::shared_ptr<PersistentAliasArena> m_persistentAliasArena;
	std::shared_ptr<FrameAliasArena> m_frameAliasArena;  // SetFrameAliasRegion
	AliasSpan m_frameAliasSpan{};
	uint32_t m_frameAliasLane = 0;
	FrameTransientObserver m_frameTransientObserver;  // SetFrameTransientObserver
	PlacementSummaryObserver m_placementSummaryObserver;  // SetPlacementSummaryObserver
	FramePlacementProbe m_framePlacementProbe;            // SetFramePlacementProbe
	FramePlacementDecline m_framePlacementDecline;        // SetFramePlacementDecline
	FramePlacementSource m_framePlacementSource;          // SetFramePlacementSource
	FramePlacementEnabled m_framePlacementEnabled;        // SetFramePlacementEnabled
	// FramePlacementInUse, written by each alias plan.
	std::atomic<bool> m_framePlacementPlanned{false};
	std::atomic<uint64_t> m_framePlacementVersion{0}, m_framePlacementPlaced{0};
	// SetExternalPersistentEpochs; and the secondary lists their tickets record into, per queue slot in the host's stream (graphics
	// lists, made at their first ticket).
	std::vector<uint32_t> m_externalPersistentEpochs;
	bool m_inlineAtExecution = false;  // SetInlineAtExecution
	std::shared_ptr<const org::RecordingTags> m_recordingTags;  // SetRecordingTag
	std::mutex m_externalSecondaryPoolsMutex;
	std::vector<std::shared_ptr<CommandListPool>> m_externalSecondaryPools;
	bool m_persistentRecordingReuse = false;
	SharingChangePolicy m_sharingChangePolicy = SharingChangePolicy::Inline;
	RecordingTimestampMode m_persistentRecordingTimestamps = RecordingTimestampMode::Off;
	std::atomic<uint64_t> m_recordingsRecorded{0};
	uint32_t m_persistentEpoch = UINT32_MAX;
	PersistentExecuteTimings m_lastPersistentExecuteTimings{};
	PersistentUpdateTimings m_lastPersistentUpdateTimings{};
	std::atomic<bool> m_currencyProbe{false};
	KeptRun m_lastKeptRun{};
	uint32_t m_keptAuditEvery = 0;
	bool m_keptBatchReuse = false;
	// Pass names the audit warned about once.
	std::unordered_set<std::string> m_keptAuditWarned;
	ExternalQueueBoundary m_externalQueueBoundary{};
	std::unique_ptr<CompilerState> m_compilerState;
    std::shared_ptr<const ResolverCaptureContext> m_resolverCaptureContext;
    std::shared_ptr<const IHostExecutionData> m_asyncUpdateHostData;

	std::vector<IResourceProvider*> _providers;
	ResourceRegistry _registry;
	struct ResolverHandleCacheEntry {
		std::shared_ptr<const void> dependencyIdentity;
		ResolverResourceSetIdentity resourceSetIdentity{};
		uint64_t registryGeneration = 0;
		std::shared_ptr<const std::vector<ResourceHandleAndRange>> handles;
	};
	std::unordered_map<const void*, ResolverHandleCacheEntry> m_resolverHandleCache;
	uint64_t m_resourceRegistryGeneration = 1;
	uint64_t m_resolverHandleCacheHitsThisFrame = 0;
	uint64_t m_resolverHandleCacheMissesThisFrame = 0;
	std::unordered_map<ResourceIdentifier, IResourceProvider*, ResourceIdentifier::Hasher> _providerMap;
	std::unordered_set<ResourceIdentifier, ResourceIdentifier::Hasher> m_resolvedResourceAliases;
	// The names looked up (LookedUp). Locked: a pass may resolve a name while its graph is prepared on the host's thread.
	void NoteLookup(ResourceIdentifier const& rid);
	mutable std::mutex m_lookupMutex;
	std::unordered_set<ResourceIdentifier, ResourceIdentifier::Hasher> m_lookedUp;

	std::vector<IPassBuilder*> m_passBuilderOrder;
	std::unordered_map<std::string, std::unique_ptr<IPassBuilder>> m_passBuildersByName;
	std::unordered_set<std::string> m_passNamesSeenThisReset;

	std::vector<AnyPassAndResources> m_masterPassList;
	std::vector<std::pair<std::string, std::string>> m_structuralExplicitAfterByName;
	std::shared_ptr<Runtime> m_runtime;
	bool m_drivesFrameServices = true;  // SetDrivesFrameServices
	bool m_sharedSetupDeferred = false;  // DeferSharedSetup, until FinishSharedSetup
	bool m_deferFirstAliasPlacement = false;  // PrepareFirstPersistentBuild: its bootstrap leaves the alias placement to the first preparation
	std::vector<std::string> m_sharedSetupPasses;
	DeviceRegistry m_backendDevices;
	std::array<uint8_t, static_cast<size_t>(QueueKind::Count)> m_minAutomaticSchedulingQueuesByKind = { 1, 1, 1 };
	std::unordered_map<std::string, std::shared_ptr<RenderPass>> renderPassesByName;
	std::unordered_map<std::string, std::string> m_passTechniquePathsByName;
	std::unordered_map<std::string, std::shared_ptr<Resource>> resourcesByName;
	std::unordered_map<uint64_t, std::shared_ptr<Resource>> resourcesByID;
	std::unordered_map<uint64_t, std::shared_ptr<Resource>> m_transientFrameResourcesByID;
	std::unordered_map<uint64_t, std::shared_ptr<Resource>> m_dynamicResourcesByStableID;
	std::unordered_map<std::string, std::shared_ptr<Resource>> m_transientFrameResourcesByName;
	org::memory::SnapshotProvider m_memorySnapshotProvider;
	std::shared_ptr<org::runtime::IStatisticsService> m_statisticsService;
	std::shared_ptr<org::runtime::IUploadService> m_uploadService;
	std::shared_ptr<org::runtime::IDescriptorService> m_descriptorService;
	std::shared_ptr<org::runtime::IRenderGraphSettingsService> m_renderGraphSettingsService;
	std::shared_ptr<org::runtime::ITaskService> m_taskService;
	QueueRegistry m_queueRegistry;
	// Declared after the queue registry, so it is destroyed before the queues it submits to.
	std::mutex m_externalClientMutex;
	std::unique_ptr<ExternalClient> m_externalClient;
	std::shared_ptr<rhi::TimelinePtr> m_frameWaitTimeline;  // SetFrameWaitTimeline
	std::vector<SharedQueueTimeline> m_sharedQueueTimelines;  // SetSharedQueueTimelines
	uint32_t m_passEpochFilter = UINT32_MAX;  // SetPassEpochFilter (persistent::AllEpochs)
	uint64_t m_frameWaitValue = 0;
	bool m_scopedFrameWait = false;  // SetScopedFrameWait
	std::optional<PresentDependency> m_lastPresentDependency;
	std::string m_presentPass;              // SetPresentPass
	std::shared_ptr<Resource> m_swapchain;  // SetSwapchainResource
	std::shared_ptr<const IVersionedResource> m_swapchainVersions;


	void EnsureMinimumAutomaticSchedulingQueues();

	org::imm::ImmediateDispatch m_immediateDispatch{};

	std::vector<std::unique_ptr<IRenderGraphExtension>> m_extensions;
    std::vector<std::string> m_extensionRegistrationIds;

	UINT64 GetNextQueueFenceValue(size_t queueSlotIndex) {
		return m_queueRegistry.GetNextFenceValue(static_cast<QueueSlotIndex>(static_cast<uint8_t>(queueSlotIndex)));
	}


	void AddResource(std::shared_ptr<Resource> resource, bool transition = false);
	void TrackTransientFrameResource(const std::shared_ptr<Resource>& resource);
public:
	// A structural build is in flight or waiting to be installed: executions still run the publication before it.
	bool HasPendingStructuralBuild() const;
private:
	// Lets go of the transient pins of resources no execution binds any more (PersistentInstanceState::retiredPins).
	void ReleaseRetiredPins(std::span<const compiler::ExecutionTimelinePoint> reached);
	// Moves the passes' declarations of a versioned resource off a replaced version, so the graph holds it no more.
	// Owner thread, in BindPersistentFrame.
	void ReleaseReplacedVersionDeclarations();
	void TrackTransientFrameResource(Resource* resource);
	void ShutdownOwnedState();

	/// Dispatches to the injected ITaskService when available, otherwise runs a serial loop.
	void ParallelForOptional(std::string_view taskName, size_t itemCount, std::function<void(size_t)> func, bool override = false) {
		const auto telemetryContext = basic_telemetry::CaptureCurrentContext();
		if (m_taskService && !override) {
			m_taskService->ParallelFor(taskName, itemCount, [telemetryContext, func = std::move(func)](size_t index) mutable {
				basic_telemetry::ContextBinding scopedContext(telemetryContext);
				func(index);
			});
		} else {
			for (size_t i = 0; i < itemCount; ++i) {
				basic_telemetry::ContextBinding scopedContext(telemetryContext);
				func(i);
			}
		}
	}

	void ParallelForOptionalLimited(std::string_view taskName, size_t itemCount, size_t maximumConcurrency,
		std::function<void(size_t)> func) {
		const auto telemetryContext = basic_telemetry::CaptureCurrentContext();
		if (m_taskService) {
			m_taskService->ParallelForLimited(taskName, itemCount, maximumConcurrency,
				[telemetryContext, func = std::move(func)](size_t index) mutable {
					basic_telemetry::ContextBinding scopedContext(telemetryContext);
					func(index);
				});
		} else {
			for (size_t i = 0; i < itemCount; ++i) {
				basic_telemetry::ContextBinding scopedContext(telemetryContext);
				func(i);
			}
		}
	}

	void MaterializeUnmaterializedResources();
	void MaterializeReferencedResources(
		const std::vector<ResourceRequirement>& resourceRequirements,
		const std::vector<std::pair<ResourceHandleAndRange, ResourceState>>& internalTransitions,
		std::string_view debugPassName = {});
	std::vector<std::shared_ptr<Resource>> CaptureRetainedAnonymousKeepAlive(
		const std::vector<ResourceRequirement>& resourceRequirements,
		const std::vector<std::pair<ResourceHandleAndRange, ResourceState>>& internalTransitions) const;
	void ResetStructuralBuildState();


	// Re-runs the pass's declaration against current per-frame state: requirements, transitions, waits, resolver
	// snapshots, referenced resources and (when they depend on it) its registry view and Setup. Persistent execution
	// lowers from this directly; the old compiler's RefreshRetainedDeclarationsForFrame also updates its declaration cache.
	void RedeclarePass(PassAndResources& p, PassType type);
	void WriteVramUsageDebugDump(uint32_t frameIndex) const;
	AnyPassAndResources MaterializeExternalPass(const ExternalPassDesc& desc);
	void RegisterExternalPassName(const ExternalPassDesc& desc, AnyPassAndResources& any);

	struct PassView {
		bool isCompute = false;
		std::span<const ResourceRequirement> reqs{};
		const std::vector<std::pair<ResourceHandleAndRange, ResourceState>>* internalTransitions = nullptr;
	};

	using AutoAliasPlannerStats = org::alias::AutoAliasPlannerStats;

	static PassView GetPassView(const AnyPassAndResources& pr);
	std::vector<AutoAliasReasonCount> autoAliasExclusionReasonSummary;
	std::vector<AutoAliasExcludedResourceDebug> autoAliasExcludedResources;
	std::vector<AutoAliasPoolDebug> autoAliasPoolDebug;
	AutoAliasPlannerStats autoAliasPlannerStats;
	AutoAliasMode autoAliasModeLastFrame = AutoAliasMode::Off;
	AutoAliasPackingStrategy autoAliasPackingStrategyLastFrame = AutoAliasPackingStrategy::GreedySweepLine;
	std::function<AutoAliasMode()> m_getAutoAliasMode;
	std::function<AutoAliasPackingStrategy()> m_getAutoAliasPackingStrategy;
	std::function<bool()> m_getRenderGraphCompileDumpEnabled;
	std::function<bool()> m_getRenderGraphVramDumpEnabled;
	std::function<bool()> m_getRenderGraphBatchTraceEnabled;
	std::function<bool()> m_getAutoAliasEnableLogging;
	std::function<bool()> m_getAutoAliasLogExclusionReasons;
	std::function<bool()> m_getAutoAliasBuildDebugData;
	std::function<uint32_t()> m_getAutoAliasPoolRetireIdleFrames;
	std::function<float()> m_getAutoAliasPoolGrowthHeadroom;
	std::function<void(std::string_view)> m_structuralMaterializeCheckpointCallback;
	std::function<void(std::string_view, std::string_view)> m_structuralMaterializeResourceCheckpointCallback;

	RenderPassBuilder& GetOrCreateRenderPassBuilder(std::string const& name);

	friend class RenderPassBuilder;
	friend struct BuilderGraphAccess;
};

} // namespace org

#include "Render/PassBuilders.h"

namespace org {

template<typename PassT, org::PassInputs InputsT, typename... StableCtorArgs>
RenderPassBuilder& RenderGraph::BuildRenderPass(std::string const& name, InputsT&& inputs, StableCtorArgs&&... ctorArgs) {
    static_assert(DerivedRenderPass<PassT>);
    auto& builder = GetOrCreateRenderPassBuilder(name);
    builder.template Instantiate<PassT>(std::forward<InputsT>(inputs), std::forward<StableCtorArgs>(ctorArgs)...);
    return builder;
}

template<typename PassT, typename... StableCtorArgs>
RenderPassBuilder& RenderGraph::BuildRenderPass(std::string const& name, StableCtorArgs&&... ctorArgs) {
    static_assert(DerivedRenderPass<PassT>);
    auto& builder = GetOrCreateRenderPassBuilder(name);
    builder.template Instantiate<PassT>(std::forward<StableCtorArgs>(ctorArgs)...);
    return builder;
}

} // namespace org
