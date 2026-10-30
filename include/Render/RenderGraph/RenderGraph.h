#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <string_view>
#include <memory>
#include <optional>
#include <variant>
#include <span>
#include <utility>
#include <array>
#include <atomic>
#include <algorithm>
#include <spdlog/spdlog.h>
#include <rhi.h>
#include <BasicTelemetry/Tracy.h>

#include "RenderPasses/Base/RenderPass.h"
#include "Resources/ResourceStateTracker.h"
#include "Interfaces/IResourceProvider.h"
#include "Render/ResourceRegistry.h"
#include "Render/CommandListPool.h"
#include "Interfaces/IPassBuilder.h"
#include "Render/MemoryIntrospectionAPI.h"
#include "Render/Runtime/IStatisticsService.h"
#include "Render/Runtime/IUploadService.h"
#include "Render/Runtime/IReadbackService.h"
#include "Render/Runtime/IDescriptorService.h"
#include "Render/Runtime/IRenderGraphSettingsService.h"
#include "Render/Runtime/ITaskService.h"
#include "Render/Runtime/TaskServiceAccess.h"
#include "Render/QueueKind.h"
#include "Render/QueueRegistry.h"
#include "Render/DeviceRegistry.h"
#include "Resources/PixelBuffer.h"
#include "Resources/Buffers/Buffer.h"
#include "Resources/TrackedAllocation.h"
#include "Render/RenderGraph/Aliasing/RenderGraphAliasingSubsystem.h"
#include "Render/RenderGraph/ScheduleTrace.h"
#include "Interfaces/IResourceResolver.h"
#include "Render/RenderGraph/RenderGraphExtension.h"

namespace org {

class Resource;
class RenderPassBuilder;
class ExternalEpochExecution;
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

	RenderGraph(rhi::Device device, rhi::Backend primaryBackend = rhi::Backend::Null);
	DeviceInstanceId RegisterBackendDevice(rhi::Backend backend, rhi::Device device);
	const DeviceRegistry& GetDeviceRegistry() const noexcept { return m_backendDevices; }
	DeviceInstanceId GetPrimaryDeviceInstance() const noexcept { return m_backendDevices.PrimaryId(); }
	~RenderGraph();
	static void ShutdownRuntime();
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
	// Persistent execution: the main graph runs from a selected persistent
	// publication rebuilt only by explicit structural edits; truly dynamic passes
	// run in per-frame Pre/Tail segments admitted through the same ledgers.
	// Requires synchronous scheduling during migration.
	enum class PersistentSegmentKind : uint8_t { Main, Pre, Tail };
	struct PersistentExecutionState; // Defined in RenderGraphPersistent.cpp; opaque elsewhere.

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
		uint32_t cachedAdmissions = 0;  // segments whose barrier plan came from the admission cache
		uint32_t batches = 0;
	};
	const PersistentExecuteTimings& LastPersistentExecuteTimings() const noexcept { return m_lastPersistentExecuteTimings; }
	// Schedule introspection (ScheduleTrace.h), off by default. While on, the owner thread traces each executable it selects
	// (the whole-frame one and each epoch's, once per executable: a steady frame builds nothing) and records what each execution
	// waits for at submission beyond its batches' own waits, publishing snapshots of both. Off costs one relaxed load per
	// preparation, execution and ticket completion. Any thread.
	void SetScheduleTraceEnabled(bool enabled) noexcept;
	bool ScheduleTraceEnabled() const noexcept;
	// Any thread: the latest published snapshot; null while off, or before anything executed with it on.
	std::shared_ptr<const PersistentScheduleSnapshot> GetPersistentScheduleSnapshot() const;
	void SetPersistentSegment(std::string passName, PersistentSegmentKind kind) {
		m_persistentSegmentKinds[std::move(passName)] = kind;
	}
	// Host epochs (persistent execution). A host that submits the graph at several
	// points of its own frame tags each pass with its epoch (ExternalPassDesc::Epoch)
	// and names the execution order once. The graph compiles over the whole frame, so
	// scheduling and transient alias placement see every lifetime; each execution then
	// prepares, admits, records and submits only the current epoch's passes (and the
	// untagged ones). Changing the order is structural.
	void SetPersistentEpochOrder(std::vector<uint32_t> order) { m_persistentEpochOrder = std::move(order); }
	// Closed executions (persistent::SynchronousAdmission::SetClosedExecutions): each segment the host
	// executes leaves every resource it touches in its home state and starts with a full memory barrier, so
	// its admission does not depend on what ran before. Set before the first frame.
	void SetPersistentClosedExecutions(bool closed) noexcept { m_persistentClosedExecutions = closed; }
	bool PersistentClosedExecutions() const noexcept { return m_persistentClosedExecutions; }

	// Tickets (async epochs, GraphHost::EnableAsyncEpochs). The graph's owner thread prepares an
	// epoch's main execution ahead of its epoch point - Update, invocations, admission, recording - into a
	// ticket, for a frame slot it has waited for. The host's submitting thread later checks the ticket is
	// still current, records the uploads queued since (RecordPendingUploads) and submits the ticket's
	// packets, assigning their timeline values then, in submission order. The owner completes every ticket
	// in that order, with those values, or abandons it. Requires closed executions (a ticket's admission
	// must not depend on the executions around it) and no Pre/Tail passes other than the host's uploads.
	struct PersistentTicket;
	struct PersistentTicketSubmission {
		bool submitted = false;
		std::vector<uint64_t> signals;  // per batch of every segment, in order: the value it signalled
		// An external epoch's ticket (SetExternalPersistentEpochs): nothing was submitted; its packets are the host's to execute,
		// with these waits and signals (the values above among them).
		std::shared_ptr<ExternalEpochExecution> external;
	};
	/**
	 * @brief Owner thread, while no ticket is prepared (the host's async epochs stopped): the persistent epochs the host executes
	 * itself, in a command buffer of its own (ExternalEpochExecution). Their tickets record into secondary lists, and a pass of
	 * theirs recorded at execution (ExternalPassDesc::recordAtExecution) is recorded by the host's execution. Every pass of such an
	 * epoch must run on the graphics queue (the host's buffer is one graphics submission), and their batches must not wait for each
	 * other across queues; a ticket that breaks either throws. A recording of one with a pass recorded at execution is never kept for reuse.
	 */
	void SetExternalPersistentEpochs(std::vector<uint32_t> epochs);
	bool IsExternalPersistentEpoch(uint32_t epoch) const noexcept;
	// Owner thread. hostFrame numbers the execution (its statistics and lifecycle). hostTag: the host's currency tag the ticket
	// is prepared under (SetPersistentTicketHostTag), which a kept recording must have been recorded under too. hostData: what
	// the passes prepare for (FramePreparationContext::preparationData), e.g. a scene revision; retained while it records.
	// admissionOnly: the ticket carries the admission alone, and its recording is bound before submission
	// (BindPersistentTicketRecording) - an epoch recorded ahead, for a revision (GraphHost::RequestEpochRecording).
	std::shared_ptr<PersistentTicket> PreparePersistentTicket(rhi::Device device, uint32_t epoch, uint8_t slot, uint64_t hostFrame,
		uint64_t hostTag = 0, std::shared_ptr<const IHostExecutionData> hostData = {}, bool admissionOnly = false);
	// A segment's recorded packets and what they were recorded for (the publication, bindings, heaps, host tag and every pass's
	// revision). Shared, immutable once made.
	struct PersistentRecording;
	// The ticket's recording (one segment), or null; the ticket keeps it.
	static std::shared_ptr<const PersistentRecording> PersistentTicketRecording(const PersistentTicket& ticket) noexcept;
	// Whether it may be submitted again and again (every pass replayable, recorded for resubmission).
	static bool PersistentRecordingReplayable(const PersistentRecording& recording) noexcept;
	// Submitting thread: binds a recording made ahead (PreparePersistentTicket for the same epoch and slot) to an admission-only
	// ticket. It must have been recorded for the ticket's publication, bindings, descriptor heaps and hostTag (the host's backing
	// version now); otherwise false, with why, and the ticket unchanged. A revision ticket (PrepareRevisionTicket) instead takes the
	// candidate admitted for exactly this recording, which needs no tag: the recording holds every backing it was recorded with.
	static bool BindPersistentTicketRecording(PersistentTicket& ticket, std::shared_ptr<const PersistentRecording> recording, uint64_t hostTag,
		std::string* why = nullptr);
	// Whether BindPersistentTicketRecording would bind it, the ticket unchanged (the same rules; why, when not).
	static bool CanBindPersistentTicketRecording(const PersistentTicket& ticket, const PersistentRecording& recording, uint64_t hostTag,
		std::string* why = nullptr);
	// Revision-driven epochs (immutable inputs). A recording made for host data (a revision) resolves the graph's resolvers through
	// it (ResolverCaptureContext of IHostExecutionData), so its publication and bindings are the revision's own, and it holds them.
	// Owner thread: a ticket for the epoch's slot that prepares nothing from live state: its admissions are its candidates', one
	// per recording it may be submitted with (AddPersistentTicketCandidate); the bind takes one, completion abandons the rest.
	std::shared_ptr<PersistentTicket> PrepareRevisionTicket(uint32_t epoch, uint8_t slot, uint64_t hostFrame);
	// Owner thread: admits the recording's publication as a candidate of the revision ticket. Lock-free against a bind on the
	// submitting thread (an append-only list); a recording published to that thread after this call is always found.
	void AddPersistentTicketCandidate(PersistentTicket& ticket, std::shared_ptr<const PersistentRecording> recording);
	// Recording reuse. A ticket's recording, when every pass of it is replayable (PreparedPass::IsReplayable) and its
	// execution closed with nothing late-bound, is kept per (epoch, frame slot) - up to kKeptRecordingVariants of them, by
	// what they were recorded for: the publication, the bindings and descriptor heaps, the host tag, and each pass's
	// revision. A ticket prepared for what a kept recording was recorded for takes it instead of recording (its lists are
	// submitted again), and a ticket the commit made stale can take the kept recording of what it depends on now
	// (AdoptKeptPersistentRecording) instead of being prepared again. Kept recordings have no profiler GPU zones, and no GPU
	// statistics queries in the shared pool (recreated as passes register, and frame-indexed): with
	// SetPersistentRecordingTimestamps, each records into queries of its own instead.
	static constexpr size_t kKeptRecordingVariants = 4;
	void SetPersistentRecordingReuse(bool reuse) noexcept { m_persistentRecordingReuse = reuse; }
	bool PersistentRecordingReuse() const noexcept { return m_persistentRecordingReuse; }
	// Pass and barrier-batch timestamps in recordings kept for resubmission (off by default): each such list records into
	// queries it owns (IStatisticsService::CreateOwnedTimestamps), resolved by every submission and read when its frame slot
	// completes (IStatisticsService::GetCompletedTimestamps). Read when a list is recorded: set it before the graph records
	// (GraphHost::SetRecordingTimestamps rebuilds the graph for it).
	void SetPersistentRecordingTimestamps(bool enabled) noexcept { m_persistentRecordingTimestamps = enabled; }
	bool PersistentRecordingTimestamps() const noexcept { return m_persistentRecordingTimestamps; }
	// Submitting thread: swaps the ticket's recording for the kept one of what its passes depend on now and a_hostTag, and
	// makes the ticket current. False (the ticket unchanged) when none is kept. Reads the kept sets only (lock-free).
	bool AdoptKeptPersistentRecording(PersistentTicket& ticket, uint64_t hostTag);
	// Owner thread, with no ticket in flight: lets every kept recording go (their lists return to their pools).
	void ClearKeptPersistentRecordings() noexcept { m_keptRecordings.reset(); }
	struct PersistentRecordingStats {
		uint64_t recorded = 0;    // tickets recorded (owner thread)
		uint64_t kept = 0;        // of them, kept for reuse
		uint64_t reused = 0;      // tickets that took a kept recording instead of recording (owner thread)
		uint64_t adopted = 0;     // stale tickets made current by a kept recording (submitting thread)
	};
	// Since the last call (any thread).
	PersistentRecordingStats TakePersistentRecordingStats() noexcept;
	// Returns the last value the submission signals on each queue slot it used (empty when abandoned).
	std::vector<std::pair<uint32_t, uint64_t>> CompletePersistentTicket(PersistentTicket& ticket, const PersistentTicketSubmission& submission);
	// Owner thread: releases executions the GPU has finished (the retirement Execute does before submitting).
	void RetirePersistentExecutions();
	// Submitting thread; they touch nothing the owner thread uses.
	static uint32_t PersistentTicketEpoch(const PersistentTicket& ticket) noexcept;
	static uint8_t PersistentTicketSlot(const PersistentTicket& ticket) noexcept;
	static uint64_t PersistentTicketHostFrame(const PersistentTicket& ticket) noexcept;
	// A value the host keeps with the ticket for its own currency checks (GraphHost: the backing version it was
	// prepared under). Set by the owner thread before the ticket is published.
	static uint64_t PersistentTicketHostTag(const PersistentTicket& ticket) noexcept;
	static void SetPersistentTicketHostTag(PersistentTicket& ticket, uint64_t tag) noexcept;
	// After RecordPendingUploads planned the ticket's uploads by queue: its batches off the graphics queue that will not wait for
	// the graphics queue's uploads (statistics); 0 when not planned.
	static uint32_t PersistentTicketSparedUploadWaits(const PersistentTicket& ticket) noexcept;
	// Every pass it prepared would still prepare the same invocation (RenderGraphPass::InvocationRevisionHash).
	static bool PersistentTicketCurrent(const PersistentTicket& ticket);
	// Submits the packets with values from nextQueueValues (the last value assigned per queue slot, advanced
	// here). Returns false when a packet failed.
	// uploadsDone: the queue slot and value that signal the uploads recorded ahead of the ticket (RecordPendingUploads), when
	// there are any. They precede the ticket on their own queue only, so a batch on another queue waits for them: every such
	// batch, unless RecordPendingUploads planned the ticket's uploads by queue (queueLists), and then only a batch that uses
	// what they write. The uploads it recorded on another queue go to that queue just ahead of its first batch there.
	// frameWait: the frame-wait timeline's value (SetFrameWaitTimeline) every batch waits for; 0 for none.
	static bool SubmitPersistentTicket(PersistentTicket& ticket, std::span<uint64_t> nextQueueValues, PersistentTicketSubmission& out,
		std::pair<uint32_t, uint64_t> uploadsDone = {UINT32_MAX, 0}, uint64_t frameWait = 0);
	// The frame-wait timeline: a timeline a host's producer of per-frame values signals, once per frame, after the values it
	// wrote for that frame; every batch of an execution waits for the value set for it (SetFrameWaitValue here, or
	// SubmitPersistentTicket's frameWait), so a late producer costs GPU idle, never a CPU wait. Set before the first execution:
	// every packet prepared from then on can wait on it.
	static constexpr uint64_t kFrameWaitTimelineIdentity = UINT64_MAX - 1;
	void SetFrameWaitTimeline(std::shared_ptr<rhi::TimelinePtr> timeline) { m_frameWaitTimeline = std::move(timeline); }
	bool HasFrameWaitTimeline() const noexcept { return m_frameWaitTimeline && *m_frameWaitTimeline; }
	// The synchronous executions' wait (Execute): until changed. 0: none.
	void SetFrameWaitValue(uint64_t value) noexcept { m_frameWaitValue = value; }
	// Submitting thread (the upload service's owner): records the uploads queued since the last call into
	// `list` as plain buffer copies, without the graph's registry or admission: a full barrier (against the
	// work before them), then the copies, whose destinations are left for the next execution's entry
	// barrier - so the list must precede a closed execution on its queue.
	// keepAlive owns the staging they read until the GPU is done with the slot. False when an upload is not
	// a pointer-targeted buffer copy (the caller must use the synchronous path).
	// recordedLists receives, first, the lists of batches their producers recorded (IUploadService::SubmitRecordedUploads),
	// which go to the queue ahead of `list`, as they are: each starts with its own full barrier, and `list` starts with one
	// whenever it records a copy.
	// Queue-scoped uploads (ticket and queueLists given): queueLists holds, per queue slot, an open list of that queue (an empty
	// handle for none; the graphics slot's is ignored). A staged upload whose destination the ticket first uses on another queue -
	// with every batch that uses it ordered after that queue's first batch, and nothing in `list` or the recorded lists touching it -
	// is recorded into that queue's list instead (after a full barrier against the work before it on that queue), which
	// SubmitPersistentTicket submits just ahead of the queue's first batch, with the waits a write into the destination needs
	// against earlier executions. The plan also records which batches on other queues use what `list` and the recorded lists
	// write: only they wait for uploadsDone (all of them when a destination there is not known). `recorded` then means that
	// `list` itself took a copy; queueCopies, when given, receives the copies recorded into queueLists. Without a ticket or
	// queueLists, everything goes into `list` and every batch on another queue waits, as before.
	bool RecordPendingUploads(rhi::Device device, rhi::CommandList& list, uint8_t slot, std::shared_ptr<void>& keepAlive, bool& recorded,
		std::vector<rhi::CommandList>& recordedLists, PersistentTicket* ticket = nullptr, std::span<rhi::CommandList> queueLists = {},
		size_t* queueCopies = nullptr);
	// The epoch the next Update/Execute runs; persistent::AllEpochs runs every pass.
	void SetPersistentEpoch(uint32_t epoch) noexcept { m_persistentEpoch = epoch; }
	uint32_t PersistentEpoch() const noexcept { return m_persistentEpoch; }
	// Whether a pass of that epoch takes part in the current execution.
	bool RunsInPersistentEpoch(uint32_t passEpoch) const noexcept {
		return m_persistentEpoch == UINT32_MAX || passEpoch == UINT32_MAX || passEpoch == m_persistentEpoch;
	}
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
	void ShutdownExtensions();
	// Serialized producer boundary. Joins CPU frame work without releasing
	// submitted ownership; the caller can then wait for every device to idle.
	void StopFrameProduction();
	// Requires GPU idle. Releases frame ownership after CPU production stopped.
	void ShutdownTaskWorkers();
	void ClearExtensions();
	void RegisterExtension(std::unique_ptr<IRenderGraphExtension> ext, std::optional<std::string_view> id = std::nullopt);
	std::optional<PresentDependency> GetLastPresentDependency() const noexcept { return m_lastPresentDependency; }
	org::memory::SnapshotProvider& GetMemorySnapshotProvider() { return m_memorySnapshotProvider; }
	const org::memory::SnapshotProvider& GetMemorySnapshotProvider() const { return m_memorySnapshotProvider; }
	void WriteVramUsageDebugDumpNow(uint8_t frameIndex) const { WriteVramUsageDebugDump(frameIndex); }
	void SetStatisticsService(std::shared_ptr<org::runtime::IStatisticsService> service) { m_statisticsService = std::move(service); }
	org::runtime::IStatisticsService* GetStatisticsService() { return m_statisticsService.get(); }
	const org::runtime::IStatisticsService* GetStatisticsService() const { return m_statisticsService.get(); }
	void SetUploadService(std::shared_ptr<org::runtime::IUploadService> service) { m_uploadService = std::move(service); }
	org::runtime::IUploadService* GetUploadService() { return m_uploadService.get(); }
	const org::runtime::IUploadService* GetUploadService() const { return m_uploadService.get(); }
	std::shared_ptr<org::runtime::IUploadService> RetainUploadService() const { return m_uploadService; }
	void SetReadbackService(std::shared_ptr<org::runtime::IReadbackService> service) { m_readbackService = std::move(service); }
	std::shared_ptr<org::runtime::IReadbackService> GetReadbackServiceOwner() const { return m_readbackService; }
	org::runtime::IReadbackService* GetReadbackService() { return m_readbackService.get(); }
	const org::runtime::IReadbackService* GetReadbackService() const { return m_readbackService.get(); }
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
		org::runtime::SetDefaultTaskService(m_taskService);
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
	// The resource a persistent executable schedules under this identity (ScheduleTrace::Resource::id), for a debug capture;
	// null when none does.
	std::shared_ptr<Resource> GetScheduledResource(uint64_t schedulingID) const;
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

	std::shared_ptr<Resource> RequestResourcePtr(ResourceIdentifier const& rid, bool allowFailure = false);
	ResourceRegistry::RegistryHandle RequestResourceHandle(ResourceIdentifier const& rid, bool allowFailure = false);
	ResourceRegistry::RegistryHandle RequestResourceHandle(Resource* const& pResource, bool allowFailure = false);
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

private:
	// Driving the graph is GraphHost's: its declared-graph calls (Build, PrepareFrame, ExecuteFrame, ConfirmPresentation)
	// or its own frames.
	friend class GraphHost;
	void CompileStructural();
	void ResetForFrame();
	void ResetForRebuild();
	void Setup();
	void Update(const UpdateExecutionContext& context, rhi::Device device);
	void Execute(PassExecutionContext& context);
	void ConfirmPresentationTailSubmission();
	GpuPassRangeBegin m_gpuPassRangeBegin;
	GpuPassRangeEnd m_gpuPassRangeEnd;
	void UpdateOnPreparationOwner(const UpdateExecutionContext& context, rhi::Device device);
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

	


	// Compiler implementation data deliberately lives outside this public header.
	// This keeps compiler algorithm/layout changes from rebuilding all ORG clients.
	struct Node;
	struct CompilerState;
	void PreparePersistentFrame(rhi::Device device, uint8_t frameIndex, const IHostExecutionData* hostData, float deltaTime);
	void ExecutePersistentFrame(PassExecutionContext& context);
	void ConfirmPersistentPresentationTail();
	// skipAliasCandidates leaves alias-eligible transients unmaterialized so the
	// structural build can place them (see PlanPersistentAliasPlacement).
	void MaterializePersistentStandalone(std::span<Resource* const> resources, bool skipAliasCandidates = false);
	// A resource a pass on another device uses: one shared D3D12 allocation, imported into Vulkan, as its representation on
	// every device (the primary's included). Its descriptors on each device name that device's representation. Kept while
	// the backing lives; the resource's next materialization clears them and the next binding makes new ones.
	void EnsurePersistentRepresentations(Resource& resource);
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
	// Hosts this frame's frame-interrupting passes in the main executable and
	// removes the previous frame's. Returns requested passes that name no main
	// pass to insert against; they run in the Tail segment instead.
	std::vector<ExternalPassDesc> UpdatePersistentFrameInterruptingPasses(std::vector<ExternalPassDesc> requested, rhi::Backend primaryBackend);
	// Compile-dump mode for the persistent graph: written once per selected executable.
	void WritePersistentGraphDebugDump(const persistent::SelectedPublication& selected) const;
	// Schedule introspection, owner thread. Traces the selected executable and its epochs' when it is not the traced one.
	void RefreshPersistentScheduleTraces(const persistent::SelectedPublication& selected);
	// Records the submission waits filled into the persistent state's scratch for one execution of `executable` (a segment
	// labelled segmentLabel), and publishes a snapshot once per round of executions.
	void RecordPersistentScheduleSubmission(const persistent::ExecutableGeneration& executable, const char* segmentLabel, uint64_t frame);
	void RecordPersistentTicketScheduleSubmissions(const PersistentTicket& ticket);
	void PublishPersistentScheduleSnapshot();
	std::atomic<bool> m_scheduleTraceEnabled{false};
	std::atomic<std::shared_ptr<const PersistentScheduleSnapshot>> m_scheduleSnapshot;
	std::vector<uint32_t> m_persistentEpochOrder;
	bool m_persistentClosedExecutions = false;
	// SetExternalPersistentEpochs; and the secondary lists their tickets record into (graphics, made at their first ticket).
	std::vector<uint32_t> m_externalPersistentEpochs;
	std::shared_ptr<CommandListPool> m_externalSecondaryPool;
	// Set while PreparePersistentTicket runs: Update skips the dynamic segments, Execute stops after recording.
	PersistentTicket* m_persistentTicketTarget = nullptr;
	uint64_t m_persistentTicketHostTag = 0;
	bool m_persistentTicketAdmissionOnly = false;
	// The ticket prepares for host data (a revision): its passes' revision hashes do not see it (InvocationRevisionHash reads
	// live state), so its recording is neither taken from nor added to the kept sets.
	bool m_persistentTicketHostData = false;
	bool m_persistentRecordingReuse = false;
	bool m_persistentRecordingTimestamps = false;
	// Kept recordings per (epoch, frame slot): replaced whole by the owner thread, read by the submitting thread. A host's frame
	// slots are its frames in flight of every epoch (GraphHost::FrameSlots), so an epoch meets several.
	// Slots: every frame slot a host can have (GraphHost refuses more than 255): a host with frames in flight times epochs past
	// 32 (Community Shaders' 4 x 9 since DCLF's first-person epochs) had its last epoch's slots unkeepable, so a revision-driven epoch's
	// recording there was refused as not replayable.
	static constexpr size_t kKeptRecordingEpochs = 32, kKeptRecordingSlots = 255;
	using KeptRecordingSet = std::vector<std::shared_ptr<const PersistentRecording>>;
	std::unique_ptr<std::array<std::array<std::atomic<std::shared_ptr<const KeptRecordingSet>>, kKeptRecordingSlots>, kKeptRecordingEpochs>> m_keptRecordings;
	std::atomic<uint64_t> m_recordingsRecorded{0}, m_recordingsKept{0}, m_recordingsReused{0}, m_recordingsAdopted{0};
	uint32_t m_persistentEpoch = UINT32_MAX;
	PersistentExecuteTimings m_lastPersistentExecuteTimings{};
	std::unordered_map<std::string, PersistentSegmentKind> m_persistentSegmentKinds;
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

	std::vector<IPassBuilder*> m_passBuilderOrder;
	std::unordered_map<std::string, std::unique_ptr<IPassBuilder>> m_passBuildersByName;
	std::unordered_set<std::string> m_passNamesSeenThisReset;

	std::vector<AnyPassAndResources> m_masterPassList;
	std::vector<std::pair<std::string, std::string>> m_structuralExplicitAfterByName;
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
	std::shared_ptr<org::runtime::IReadbackService> m_readbackService;
	std::shared_ptr<org::runtime::IDescriptorService> m_descriptorService;
	std::shared_ptr<org::runtime::IRenderGraphSettingsService> m_renderGraphSettingsService;
	std::shared_ptr<org::runtime::ITaskService> m_taskService;
	QueueRegistry m_queueRegistry;
	std::shared_ptr<rhi::TimelinePtr> m_frameWaitTimeline;  // SetFrameWaitTimeline
	uint64_t m_frameWaitValue = 0;
	std::optional<PresentDependency> m_lastPresentDependency;

	rhi::TimelinePtr m_readbackFence;
	rhi::TimelinePtr m_copyReadbackFence;

	void EnsureMinimumAutomaticSchedulingQueues();

	org::imm::ImmediateDispatch m_immediateDispatch{};

	std::vector<std::unique_ptr<IRenderGraphExtension>> m_extensions;
    std::vector<std::string> m_extensionRegistrationIds;

	UINT64 GetNextQueueFenceValue(size_t queueSlotIndex) {
		return m_queueRegistry.GetNextFenceValue(static_cast<QueueSlotIndex>(static_cast<uint8_t>(queueSlotIndex)));
	}


	void AddResource(std::shared_ptr<Resource> resource, bool transition = false);
	void TrackTransientFrameResource(const std::shared_ptr<Resource>& resource);
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
	void CompileFrame(rhi::Device device, uint8_t frameIndex, const IHostExecutionData* hostData);
	void PrepareAndCompileFrame(rhi::Device device, uint8_t frameIndex,
		const IHostExecutionData* hostData, float deltaTime = 0.0f);
	void WriteVramUsageDebugDump(uint8_t frameIndex) const;
	AnyPassAndResources MaterializeExternalPass(const ExternalPassDesc& desc, bool callSetup, bool materializeReferencedResources);
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
