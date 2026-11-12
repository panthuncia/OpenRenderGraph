#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <rhi.h>

#include "Render/RenderGraph/RenderGraph.h"
#include "Render/RenderGraph/ExternalEpochExecution.h"
#include "Render/FrameArena.h"
#include "Render/RunRef.h"
#include "OpenRenderGraph/Runtime.h"

namespace org::runtime {
class ITaskService;
class IUploadService;
class ResourceCleanupQueue;
}

namespace org {

struct IHostExecutionData;

// Drives persistent (static) execution for an application that embeds ORG in its own renderer, for example a host that
// adopted another API's device. It owns what every such embedding needs and nothing application-specific (the device,
// task service, services and retirement are the Runtime's, which the host is given):
//   - one graph per host epoch (Desc::epochOrder), or one graph without epochs. Each is built from the registered
//     extension factories and takes only its epoch's passes and the untagged ones (RenderGraph::SetPassEpochFilter), so
//     it compiles, admits and executes that epoch alone, and one epoch's rebuild leaves the others running,
//   - one timeline per queue, shared by every graph (RenderGraph::SetSharedQueueTimelines): an epoch's work on a queue
//     needs no wait on an earlier epoch's work there, as with one graph,
//   - the built-in upload pass (submitted through the graph's external client ahead of each execution),
//   - persistent execution with the requested external queue boundary,
//   - rebuilding graphs from the registered extension factories on request,
//   - synchronous per-frame Update + Execute (Execute submits before returning),
//   - run leases: each run (an execution, or a ticket's submission) holds a lease for what it uses beyond its recording
//     (its upload lists, frame arena regions, statistics range, upload pages and keep-alives). The lease retires once the
//     run's GPU work is done, and what it held is freed then (CurrentRun, RecordingContext::Run),
//   - an orderly shutdown that waits only for the graphs' own work.
//
// Backpressure. The only blocking wait for the GPU is frame pacing: ExecuteFrame waits for the run of its position
// framesInFlight frames back, and a host that runs async epochs paces its own frames. Leases never wait for a run of the
// current frame: a host that holds its submissions in a stream until the frame's end (an external queue boundary,
// DXVK's in-stream epochs) has not handed those runs to the device yet, so the wait would never end. When something runs
// dry, it waits only for a run behind a seal (SealSubmitted), or it grows.
// Not thread-safe: all calls come from the host's render thread. With async epochs (SetAsyncEpochs) the host
// runs one thread of its own that owns the graphs from then on; the render thread submits, and builds a graph only
// while the host's thread has let go of it.
class GraphHost {
public:
	struct Desc {
		// The runtime the host's graphs run on (their device, task service and services).
		std::shared_ptr<Runtime> runtime;
		RenderGraph::ExternalQueueBoundary queueBoundary{};
		// Frames the CPU may run ahead of the GPU. ExecuteFrame waits for its run of framesInFlight frames back (a frame is each
		// epoch of epochOrder once). A revision-driven epoch's recordings are published as framesInFlight copies.
		uint32_t framesInFlight = 3;
		// Host epochs: the order the host runs them in within its own frame. Each is a graph of its own. Passes declare
		// theirs with ExternalPassDesc::Epoch (untagged ones run in every epoch); ExecuteFrame and SubmitEpoch name the one
		// to run.
		std::vector<uint32_t> epochOrder;
		// RenderGraph::SetPersistentClosedExecutions, an interop option: every execution leaves its resources in their home
		// states and starts with a full barrier, so its admission is independent of what ran before. Nothing in the host
		// requires it: async epochs and recording reuse work with open executions, admitted at submission.
		bool closedExecutions = false;
		// RenderGraph::SetPersistentExternalHomeStates, an interop option: every execution gives each host-owned texture it
		// touches (ExternalTextureResource) back in its home state on the graphics queue, so the host finds its own resources
		// where it left them. Everything else flows through the ledgers.
		bool externalHomeStates = false;
		// One alias arena for every epoch's graph (RenderGraph::SetPersistentAliasArena): their transients share the same pools,
		// ordered by the runtime's alias ledger. Off: each graph has pools of its own.
		bool sharedAliasArena = true;
		// Async epochs: tickets record replayable recordings (RenderGraph::SetPersistentRecordingReuse), which revision-driven
		// epochs need: their published copies are submitted again and again. A live ticket's recording is used once and
		// nothing keeps it. Replayable recordings carry no profiler GPU zones.
		bool reuseRecordings = false;
		// Async epochs: a ticket's staged uploads whose destination it first uses on another queue than graphics (async compute)
		// are recorded on that queue, just ahead of the ticket's first batch there, instead of on the graphics queue; and a batch
		// on another queue waits for the graphics queue's uploads only when it uses what they write (RenderGraph::
		// RecordPendingUploads' queue-scoped uploads). Off: every upload on the graphics queue, and every batch on another queue
		// waits for them.
		bool queueScopedUploads = true;
	};

	using ExtensionFactory = std::function<std::unique_ptr<RenderGraph::IRenderGraphExtension>()>;

	explicit GraphHost(Desc desc);

	~GraphHost();
	GraphHost(const GraphHost&) = delete;
	GraphHost& operator=(const GraphHost&) = delete;

	// Extensions are instantiated from their factories each time a graph is (re)built, once per graph, so they may size
	// resources from current host state. Each graph takes the passes of its own epoch from them.
	// epochs: the epochs whose graphs the extension is made into (empty, or one of them persistent::AllEpochs: every graph).
	// An extension whose passes and registered resources only some epochs use names them: the other epochs' graphs never
	// make it, and adding it again, or removing it, rebuilds only its epochs' graphs. Adding or removing one without epochs
	// rebuilds every graph.
	void AddExtension(std::string id, ExtensionFactory factory, std::vector<uint32_t> epochs = {});
	void RemoveExtension(const std::string& id);
	/** @brief Whether an extension is registered under id (AddExtension). Owner thread. */
	bool HasExtension(std::string_view id) const noexcept {
		return std::ranges::any_of(m_extensions, [&](const Registered& entry) { return entry.id == id; });
	}

	// Every graph is built again before its next execution or submission. reason: why, named in the build's log line
	// ("GraphHost: epoch N's graph built again ... why: ..."); requests with the same reason are counted together.
	void RequestRebuild(std::string_view reason = {}) noexcept;
	// The epoch's graph alone is built again; the others keep running as built. With async epochs, the epoch's ticket is given
	// back and its graph replaced at its next submission (or BuildIfRequested), and the old graph is released once its own
	// submitted work completes, without waiting for it. An epoch the host does not have is ignored.
	void RequestRebuild(uint32_t epoch, std::string_view reason = {}) noexcept;
	/**
	 * @brief The graphs that looked id up by name (RenderGraph::LookedUp, found or not) are built again; the others keep theirs.
	 * For a name whose meaning changed without anything being dropped: a resource registered under a new name, which no graph that
	 * never asked for it needs. A graph that still holds a resource being dropped needs RequestRebuild instead. Returns how many
	 * epochs were marked.
	 */
	size_t RequestRebuildFor(const ResourceIdentifier& id, std::string_view reason = {}) noexcept;
	// Pass and barrier-batch timestamps in the recordings kept across frames (RenderGraph::SetPersistentRecordingTimestamps),
	// off by default: read back with the other pass timestamps (IStatisticsService::GetCompletedTimestamps, in the completed
	// frame callback). A change requests a rebuild, so every kept recording (a revision's epochs) is recorded again with them,
	// or without; until that build, kept recordings stay as they were recorded.
	void SetRecordingTimestamps(bool enabled) noexcept {
		if (m_recordingTimestamps.exchange(enabled, std::memory_order_acq_rel) != enabled) RequestRebuild("recording timestamps switched");
	}
	bool RecordingTimestamps() const noexcept { return m_recordingTimestamps.load(std::memory_order_acquire); }
	// Diagnostics: the external queue boundary's entry and exit barriers in place of Desc::queueBoundary's, from the next build
	// (dropping them is unsafe: the graphs' first and last accesses are then unordered against the host's work).
	void SetQueueBoundary(RenderGraph::ExternalQueueBoundary boundary) noexcept {
		m_boundaryEntry.store(boundary.entry, std::memory_order_release);
		m_boundaryExit.store(boundary.exit, std::memory_order_release);
		m_boundaryOverridden.store(true, std::memory_order_release);
		RequestRebuild("queue boundary overridden");
	}
	// Schedule introspection (RenderGraph::SetScheduleTraceEnabled) on every graph and on every graph built after, so it
	// survives rebuilds. Host thread (it reads the current graphs); read the snapshots through Graph(epoch).
	void SetScheduleTraceEnabled(bool enabled) noexcept {
		m_scheduleTrace.store(enabled, std::memory_order_release);
		for (auto& instance : m_instances)
			if (instance.graph) instance.graph->SetScheduleTraceEnabled(enabled);
	}
	bool ScheduleTraceEnabled() const noexcept { return m_scheduleTrace.load(std::memory_order_acquire); }

	// Explicit builds: a rebuild requested (an extension added or removed, RequestRebuild) waits for the caller's build point,
	// BuildIfRequested; until then every execution, submission and preparation runs the graphs as built, whose extensions keep
	// the state they were made with. For a caller that decides once a frame what its recordings cover (RequestEpochRecording,
	// UseEpochRecording): a recording of the current build stays its tickets' until that point. Without, the next execution or
	// submission of an epoch builds its graph (the default). A graph is built whenever there is none.
	void SetExplicitBuilds(bool explicitBuilds) noexcept { m_explicitBuilds = explicitBuilds; }
	// Owner thread: builds every graph whose rebuild was requested (or that has none); true when it built one. With async epochs,
	// the other epochs keep their tickets.
	bool BuildIfRequested();
	// A rebuild was requested and has not been built (with explicit builds, until BuildIfRequested).
	bool RebuildRequested() const noexcept;

	// Failure isolation. A failure while one epoch's graph is built or prepared stays with that epoch: the host's thread keeps
	// running, and every other epoch keeps its graph and its tickets. Only host-wide errors (a failed GPU completion wait, a lost
	// device) still end the host's thread.
	/**
	 * @brief One epoch's graph failed to build or to prepare.
	 *
	 * A failed build keeps the graph the epoch had (keptGraph), which runs as before; with none, the epoch runs nothing. A failed
	 * preparation (a pass's Prepare, a bindless view it resolves, its recording) leaves the epoch running nothing. Either way the
	 * epoch is not built again until a rebuild is requested (RequestRebuild), and a build that succeeds clears its failure.
	 */
	struct EpochFailure {
		uint32_t epoch = UINT32_MAX;  // persistent::AllEpochs without epochs
		uint64_t buildGeneration = 0;  // the epoch's build that failed to prepare, or the build kept when a build failed
		bool keptGraph = false;  // a build failed and the epoch still runs its previous graph
		std::string error;
	};
	/** @brief Called on the owner thread by PollEpochFailures, once per failure. It must not throw. */
	using EpochFailedCallback = std::function<void(const EpochFailure&)>;
	/** @brief The owner's handler of epoch failures (PollEpochFailures calls it). Owner thread. */
	void SetEpochFailedCallback(EpochFailedCallback callback) { m_epochFailed = std::move(callback); }
	/**
	 * @brief Owner thread: hands the failures reported since the last call to the callback, in the order they happened. A failure
	 * of a build the epoch has since replaced is dropped. Call it where nothing of the failed epoch's is in progress, for example
	 * before deciding what a frame's epochs run. Returns how many were handed over.
	 */
	size_t PollEpochFailures();
	/** @brief Owner thread: whether the epoch runs nothing because its graph failed (until a rebuild of it succeeds). */
	bool EpochFailed(uint32_t epoch) const noexcept;
	/**
	 * @brief What ExecuteFrame and SubmitEpoch throw for an epoch that runs nothing (EpochFailed). Nothing was submitted, and the
	 * host is fine: the caller skips the epoch, and its owner learns why through PollEpochFailures.
	 */
	struct EpochFailedError : std::runtime_error {
		EpochFailedError(uint32_t a_epoch, const std::string& a_error)
			: std::runtime_error("epoch " + std::to_string(a_epoch) + "'s graph failed: " + a_error), epoch(a_epoch) {}
		uint32_t epoch = UINT32_MAX;
	protected:
		struct Message {};
		EpochFailedError(Message, uint32_t a_epoch, const std::string& a_what) : std::runtime_error(a_what), epoch(a_epoch) {}
	};
	/**
	 * @brief What SubmitEpoch throws for a revision-driven epoch whose published copies are all busy: each holds a run the host has
	 * not sealed yet (SealSubmitted) and that is not done, so no copy can be submitted again without waiting for a run that may
	 * still be in the host's stream. It needs more runs between two seals than there are copies (Desc::framesInFlight). Nothing was
	 * submitted, the epoch has not failed (nothing is reported to PollEpochFailures), and its runs go on after the next seal or once
	 * a copy's run completes. The caller skips this run. An EpochFailedError, so a caller that skips a failed epoch skips this too.
	 */
	struct EpochCopiesBusyError : EpochFailedError {
		EpochCopiesBusyError(uint32_t a_epoch, uint32_t a_copies)
			: EpochFailedError(Message{}, a_epoch, "epoch " + std::to_string(a_epoch) + " ran more than its " + std::to_string(a_copies)
				+ " published copies between two frame boundaries; this run is skipped until the next SealSubmitted"),
			  copies(a_copies) {}
		uint32_t copies = 0;
	};

	// Runs after any rebuild and before the frame is prepared, with the graphs' upload
	// and descriptor services active: the place to queue this frame's uploads
	// (BUFFER_UPLOAD) into graph resources.
	using FrameCallback = std::function<void(RenderGraph&)>;

	// Called on the thread that retires runs (ExecuteFrame's, or the async host's thread) once a run is complete on the GPU, after
	// its pass timestamps were read back: the passes whose gpuSampleSerial equals stats.GetFrameSerial() are the ones that run
	// recorded. frameNumber is the run's host frame number.
	using CompletedFrameCallback = std::function<void(uint64_t frameNumber, const runtime::IStatisticsService& stats)>;
	void SetCompletedFrameCallback(CompletedFrameCallback callback) { m_completedFrame = std::move(callback); }
	// Called on the thread that sees completions (ExecuteFrame's, or the async host's thread) with n: every submission numbered n or
	// less (SubmissionsMade) has completed on the GPU. Never decreases. Frame numbers cannot say this: an async ticket's is given when
	// it is prepared, so a frame's submissions are not numbered in their order, and a frame number completing says nothing of a
	// ticket submitted after it with a lower one. What a submitter keeps from reuse until its work is done retires on this.
	using RetiredCallback = std::function<void(uint64_t submissions)>;
	void SetRetiredCallback(RetiredCallback callback) { m_retired = std::move(callback); }

	// GPU ranges around every pass (RenderGraph::SetGpuPassRangeCallbacks), kept across rebuilds. Set them while no
	// frame executes: the host's thread reads them when it prepares.
	void SetGpuPassRangeCallbacks(RenderGraph::GpuPassRangeBegin begin, RenderGraph::GpuPassRangeEnd end);

	// Builds if needed, then prepares and submits one frame. hostData is visible to
	// passes through PassPrepareContext::preparationData for this frame only.
	// With an epoch, that epoch's graph runs (its passes and the untagged ones); without, the first graph.
	// Throws EpochFailedError when the epoch's graph failed to build or prepare (now or before): it is then marked failed and
	// reported (PollEpochFailures), and the other epochs run on.
	void ExecuteFrame(const IHostExecutionData* hostData = nullptr, const FrameCallback& beforePrepare = {},
		uint32_t epoch = UINT32_MAX, std::shared_ptr<const void> resourceOwner = {});  // persistent::AllEpochs

	// Retires every graph: stops frame production and waits for their work.
	void DestroyGraph();

	// Async epochs. The host prepares each listed epoch's execution ahead of time, as a ticket
	// (RenderGraph::PreparePersistentTicket), on a thread of its own - which from then on is the only thread
	// that touches the graphs: it waits for frame slots, prepares, admits and records tickets, completes them
	// in submission order and retires finished work. The render thread calls SubmitEpoch instead of
	// ExecuteFrame. Each ticket is admitted when it is submitted, open or closed. Empty stops it (and waits for the thread); an
	// epoch's rebuild pauses that epoch alone.
	// revisionEpochs (a subset of epochs): recorded ahead for a revision (RequestEpochRecording) and submitted with that
	// recording (SubmitEpoch's recording). Their tickets carry the admission alone; nothing about them is recorded or prepared
	// again at submission. They need Desc::reuseRecordings' replayable recordings.
	void SetAsyncEpochs(std::vector<uint32_t> epochs, std::vector<uint32_t> revisionEpochs = {});
	bool AsyncEpochs() const noexcept { return static_cast<bool>(m_async); }
	/**
	 * @brief Async epochs whose graphics-queue work the caller executes itself, in its own command buffers
	 * (RenderGraph::SetExternalPersistentEpochs): SubmitEpoch submits only their batches on other queues. Their graphics uploads
	 * are recorded into a secondary list, the execution is taken with TakeExternalExecution, and the caller executes its
	 * segments in order, each in a submission that waits for and signals what the segment names (a segment marked split in a new
	 * one). Their uploads must be plain copies (no producer-recorded lists). The epochs that change are built again
	 * (RequestRebuild(epoch)); kept across rebuilds.
	 */
	void SetExternalEpochs(std::vector<uint32_t> epochs);
	/**
	 * @brief Submitting thread, after SubmitEpoch of an external epoch: its execution (null: none since the last call). The caller
	 * must execute its segments (ExternalEpochExecution::Execute), or signal the ones it did not unexecuted: the graphs' later
	 * work on their timelines waits for those signals.
	 */
	std::shared_ptr<ExternalEpochExecution> TakeExternalExecution() noexcept { return std::move(m_externalExecution); }
	/**
	 * @brief Submitting thread: the segments of an external execution the caller will not execute, from a_first on (SIZE_MAX: from
	 * ExecutedSegments on), completed without running them (per segment, a submission on the graphics queue of no lists, with its
	 * waits and signals). What their passes would have written is left as it was.
	 */
	void SignalExternalExecutionUnexecuted(const ExternalEpochExecution& execution, size_t a_first = SIZE_MAX);

	// An epoch recorded for one revision: its published copies, one per copy of the epoch (Desc::framesInFlight of them). Copy i is
	// recorded for the run RunRef::Copy(its copy lease, i), so what it reads per run (LatchBlock regions) stays in place for every
	// submission of it. A list cannot be pending twice, so each submission takes a copy whose last run is done. Immutable.
	// While the caller holds it, the epoch's tickets admit it as a candidate (RenderGraph::PrepareRevisionTicket), so it can be
	// submitted at any frame; released, it is never admitted again.
	struct EpochRecording {
		uint32_t epoch = 0;
		std::vector<std::shared_ptr<const RenderGraph::PersistentRecording>> copies;  // by copy index
		std::shared_ptr<const IHostExecutionData> revision;  // what it was recorded for
		uint64_t buildGeneration = 0;  // its epoch's build it was recorded on (BuildGeneration(epoch)): no later build's tickets take it
	};
	using EpochRecordingCallback = std::function<void(std::shared_ptr<const EpochRecording>, std::exception_ptr)>;
	// The error a request's done gets when it was dropped unrecorded: its epoch's graph was built again, or the host's thread stopped
	// (SetAsyncEpochs, DestroyGraph). Nothing was recorded for it; the caller asks again for the graph as built now, once.
	struct EpochRecordingDropped : std::runtime_error {
		explicit EpochRecordingDropped(uint64_t generation)
			: std::runtime_error("an epoch recording request was dropped by a graph build"), buildGeneration(generation) {}
		uint64_t buildGeneration = 0;  // its epoch's build current when it was dropped (BuildGeneration(epoch))
	};
	// Any thread, lock-free, safe across a build: records the revision-driven epoch for hostData (what its passes prepare for: the
	// revision) on the host's thread, one recording per published copy. Recording a copy waits for nothing: the copy's next
	// submission waits for its last run instead. The request goes onto a queue of the host's own lifetime, never through the async
	// state a build replaces; the host's thread takes it whole, in posting order. done runs exactly once: on the host's thread once
	// every copy is recorded (the recording carries the build it was recorded on, EpochRecording::buildGeneration), or with the
	// error that stopped it (an epoch that is not a revision-driven async epoch is one), or with EpochRecordingDropped when its
	// epoch's graph was built again, or the host's thread stopped, before it was recorded. False only when async epochs were never
	// started or are off (SetAsyncEpochs with none): done is not called then. The host's end drops what is queued without calling
	// done (its callers may be gone).
	bool RequestEpochRecording(uint32_t epoch, std::shared_ptr<const IHostExecutionData> hostData, EpochRecordingCallback done);
	// Render thread, for an async epoch it did not submit this frame: its ready ticket, prepared and not submitted, goes back to the
	// host's thread (abandoned, its run lease freed, a new one prepared). False when no ticket was ready (or the mailbox is full: it
	// stays).
	bool ReleaseEpochTicket(uint32_t epoch);
	// Submits the epoch at the current point of the host's queue stream. Builds the epoch's graph first when its rebuild is due.
	// Waits for its ticket if the host's thread has not finished it; runs beforeSubmit with the graphs' services active and
	// CurrentRun() the ticket's run (write latches and queue uploads there); has the ticket prepared again (and waits)
	// if a pass would now prepare differently; records the uploads queued since the last submission; and
	// hands both to the queue - timeline values are assigned here, in submission order. Throws when the
	// graph failed (on either thread): EpochFailedError when only this epoch's did, and nothing was submitted.
	// resourceOwner pins immutable descriptors/imports named by this execution
	// until all accepted queue submissions complete.
	// recording: a revision-driven epoch's (RequestEpochRecording), whose copy for the ticket's run is submitted; it must be one
	// the ticket admitted, or this throws (it is never prepared again). Throws EpochCopiesBusyError when no copy can take the run.
	void SubmitEpoch(uint32_t epoch, const FrameCallback& beforeSubmit = {}, std::shared_ptr<const void> resourceOwner = {},
		std::shared_ptr<const EpochRecording> recording = {});
	// Submitting thread, inside SubmitEpoch's beforeSubmit for a revision-driven epoch: submit this revision recording (one of the
	// ticket's candidates: RequestEpochRecording) when SubmitEpoch was given none. A recording the ticket did not admit makes
	// SubmitEpoch throw.
	void UseEpochRecording(std::shared_ptr<const EpochRecording> recording) noexcept { m_chosenRecording = std::move(recording); }
	// Submitting thread, inside SubmitEpoch's beforeSubmit for a revision-driven epoch: whether the ticket being submitted would take
	// this recording (UseEpochRecording) - false, with why, for one of another epoch or one the ticket did not admit (recorded for a
	// graph since rebuilt, or made after the ticket). A caller that writes values into what the recording reads decides by this
	// before writing, so SubmitEpoch never throws for it.
	bool CanUseEpochRecording(const EpochRecording& recording, std::string* why = nullptr) const;
	// Owner thread: moves with every graph build, of any epoch.
	uint64_t BuildGeneration() const noexcept { return m_buildGeneration; }
	// Owner thread: moves with every build of the epoch's graph: a recording (RequestEpochRecording) is of the build it was made on,
	// and no later one's tickets take it. 0 for an epoch the host does not have.
	uint64_t BuildGeneration(uint32_t epoch) const noexcept;
	// Owner thread: whether a recording is of its epoch's graph as built now - with explicit builds, what decides before a frame's
	// submissions whether they can take it (until the next BuildIfRequested).
	bool EpochRecordingCurrent(const EpochRecording& recording) const noexcept;
	// Submitting thread: a versioned buffer (VersionedBuffer) published a new version. Nothing changed in place, so nothing waits;
	// a ticket prepared before resolved the old version and is not current (it is prepared again, or takes a kept recording).
	// Revision-driven epochs need no call: their recordings resolve the versions their revisions name.
	void NoteNewVersions() noexcept;
	// SubmitEpoch counters since the last call.
	struct AsyncStats {
		uint64_t submitted = 0;
		uint64_t waited = 0;   // the ticket was not ready yet
		uint64_t stale = 0;    // prepared again after beforeSubmit
		uint64_t uploads = 0;  // submissions that carried uploads
		uint64_t recordedUploadLists = 0;  // producers' recorded upload lists submitted as they were (IUploadService::SubmitRecordedUploads)
		// Desc::queueScopedUploads: staged copies recorded on another queue than graphics, ahead of the batch that first uses them;
		// and batches off the graphics queue spared the wait for the graphics queue's uploads (they use nothing those write).
		uint64_t queueUploadCopies = 0, sparedUploadWaits = 0;
		uint64_t wakesSkipped = 0;  // submissions whose completion the host's thread took at a GPU wake-up instead of a signal
		uint64_t staleBacking = 0;  // tickets prepared again only because a new version was published since their preparation
		uint64_t rebuilds = 0;      // epoch graphs built again (an epoch's rebuild, or every graph's)
		uint64_t recorded = 0;      // tickets recorded
		// Tickets admitted with their prediction's plan, and those planned against the ledgers (RenderGraph::PersistentRecordingStats).
		uint64_t predicted = 0, planned = 0;
		// Revision-driven epochs: copies recorded for requests (RequestEpochRecording), requests completed, and submissions of a
		// revision's recording.
		uint64_t revisionCopiesRecorded = 0, revisionRecordings = 0, revisionSubmissions = 0;
		// Revision-driven tickets held for the next seal because every copy held an unsealed run (EpochCopiesBusyError), and copy
		// waits: a submission whose copy's last run was sealed but not done, waited for on the host's thread.
		uint64_t copiesBusy = 0, copyWaits = 0;
	};
	AsyncStats TakeAsyncStats() noexcept;

	// What the runs in flight hold, for a memory log line. Any thread.
	struct RunMemoryStats {
		uint32_t leasesInFlight = 0;  // open or submitted, not retired (published copies' leases while in use)
		uint32_t leasesPeak = 0;      // the most in flight since the last call
		uint32_t leaseIndices = 0;    // lease indices made (the high water of runs in flight, plus the published copies)
		uint32_t externalRuns = 0;    // of leasesInFlight, external runs (BeginExternalRun)
		uint32_t copyLeases = 0;      // published copies' leases (revision-driven epochs times framesInFlight)
		uint32_t statisticsRanges = 0;  // statistics query ranges held (IStatisticsService::FrameCapacity)
		FrameArena::Stats arena{};
	};
	RunMemoryStats TakeRunMemoryStats();

	// The first epoch's graph (the only one without epochs), null before the first build. Services (uploads, descriptors,
	// readbacks, statistics) are the runtime's, the same in every graph.
	RenderGraph* Graph() noexcept { return m_instances.empty() ? nullptr : m_instances.front().graph.get(); }
	// The epoch's graph, null for an epoch the host does not have or before its build. Diagnostics and schedule snapshots.
	RenderGraph* Graph(uint32_t epoch) noexcept;
	// Every graph, in epoch order (null entries before their builds), and the epoch of each.
	size_t GraphCount() const noexcept { return m_instances.size(); }
	RenderGraph* GraphAt(size_t index) noexcept { return index < m_instances.size() ? m_instances[index].graph.get() : nullptr; }
	uint32_t EpochAt(size_t index) const noexcept { return index < m_instances.size() ? m_instances[index].epoch : UINT32_MAX; }
	/** @brief The device the host was made with (Desc::device), e.g. for a producer recording its uploads ahead (StagedUploadBatch::Record). */
	rhi::Device Device() const noexcept { return m_device; }
	Runtime& GetRuntime() const noexcept { return *m_desc.runtime; }
	/** @brief The device-generation cleanup lane used for immutable binding roots. */
	std::shared_ptr<runtime::ResourceCleanupQueue> ResourceCleanup() const;
	runtime::IUploadService* Uploads() noexcept;
	// Owner thread: the graphs' descriptor service (its resource and sampler heaps: what a recording binds), null without a graph.
	runtime::IDescriptorService* Descriptors() noexcept;
	// The same, kept alive by the caller: a producer on another thread (growth as graph work, its worker upload path).
	std::shared_ptr<runtime::IUploadService> RetainUploads() noexcept;
	uint64_t FramesExecuted() const noexcept { return m_frameNumber; }
	bool ClosedExecutions() const noexcept { return m_desc.closedExecutions; }
	// The epochs' graphs' shared alias pools (Desc::sharedAliasArena), made with the first graph; null without.
	std::shared_ptr<PersistentAliasArena> AliasArena() const noexcept { return m_aliasArena; }
	/**
	 * @brief Submitting thread: the run the host may write per-run data for now (LatchBlock regions, CPU payloads). Inside
	 * SubmitEpoch's beforeSubmit, the ticket's run; inside ExecuteFrame's beforePrepare, the execution's run; otherwise the external
	 * run (BeginExternalRun) when one is active; otherwise the last run submitted. RecordingContext::Run names the same run.
	 */
	RunRef CurrentRun() const noexcept {
		if (m_externalRun && !m_submittingTicket) return *m_externalRun;
		return m_async ? m_asyncRun : m_syncRun;
	}
	/**
	 * @brief Submitting thread: a client that records work reading its LatchBlocks outside the host's executions (into another
	 * submitter's command buffer: Community Shaders' DCLF CPU pipeline, DXVK's own list) opens a run of its own. CurrentRun() returns
	 * it from now until EndExternalRun, except inside a ticket's beforeSubmit. The run's lease retires once `timeline` reaches
	 * `value`, which the client signals after its last work that reads the run (its frame's end, maybe later than this call). Its
	 * upload pages are taken now (what was staged since the last submission), and every descriptor slot, resource or backing
	 * retired from now on waits for the point too. Opening another external run replaces this one as CurrentRun(); both stay
	 * leased until their points.
	 */
	RunRef BeginExternalRun(std::shared_ptr<rhi::TimelinePtr> timeline, uint64_t value);
	void EndExternalRun() noexcept { m_externalRun.reset(); }
	bool ExternalRunActive() const noexcept { return m_externalRun.has_value(); }
	// Migration shims, removed once every client names runs: the current run's lease index, and the host's frames in flight.
	uint32_t CurrentFrameSlot() const noexcept { return CurrentRun().lease; }
	// The host frame number of the last execution (ExecuteFrame or SubmitEpoch); the completed-frame
	// callback reports frames by it.
	uint64_t LastHostFrame() const noexcept { return m_lastHostFrame; }
	// Submitting thread: the executions submitted so far (ExecuteFrame and SubmitEpoch), each numbered in submission order from 1.
	// Work submitted before this returns has retired once the retired callback reports this value.
	uint64_t SubmissionsMade() const noexcept { return m_submissions; }

	// A point of the GPU's progress: per queue, its timeline (shared: valid after a rebuild) and a value it reaches. Any thread.
	struct GpuPoint {
		std::vector<std::pair<std::shared_ptr<rhi::TimelinePtr>, uint64_t>> points;
		bool Reached() const noexcept {
			for (const auto& [timeline, value] : points)
				if (timeline && *timeline && (*timeline)->GetCompletedValue() < value) return false;
			return true;
		}
		// False on a timeout or a failed wait.
		bool Wait(uint32_t timeoutMs = UINT32_MAX) const noexcept {
			for (const auto& [timeline, value] : points)
				if (timeline && *timeline && (*timeline)->HostWait(value, timeoutMs) != rhi::Result::Ok) return false;
			return true;
		}
	};
	// Submitting thread: the point every queue reaches once everything submitted so far (ExecuteFrame, SubmitEpoch) is done.
	// Submitted to the host, not always to the device: a host that queues its submissions behind a stream (an external queue
	// boundary) may hold them until its next flush. So a producer whose signal later work of the same frame waits for must not
	// wait for this point: the work it names may be queued behind that later work, or flushed only once the frame ends. It
	// waits for SealedPoint instead.
	GpuPoint SubmittedPoint() const;
	// Submitting thread, at a frame boundary: everything submitted so far is on the device's queues (the host flushed its stream).
	// Seals SubmittedPoint's current value as SealedPoint, and counts a frame boundary (Seals): a run is waited for only once a seal
	// came after it. Also trims the frame arena's idle chunks.
	void SealSubmitted();
	// Any thread: the SealSubmitted calls so far.
	uint64_t Seals() const noexcept { return m_seals.load(std::memory_order_acquire); }
	// Any thread: the point SealSubmitted last captured (none before the first). It names only work already on the device, and
	// none of the current frame's, so a mid-frame producer may wait for it whatever the frame's later work waits for.
	GpuPoint SealedPoint() const;
	// Any thread, for a failure message only: the host's thread's state (each async epoch's pause and ticket, the run leases in
	// flight and what each waits for, the queued requests, the wake), read racily.
	std::string DescribeAsyncState() const;
	// A thread waiting for recording requests: wakes the host's thread when it sleeps with one queued (a lost wake, logged once).
	// True when it did.
	bool NudgeHostThread();

	// The frame-wait timeline (RenderGraph::SetFrameWaitTimeline): a host producer of per-frame values signals it once per frame,
	// after the values it wrote for that frame, and every batch submitted from then on waits for SetFrameWaitValue's value. Set
	// once, before the frames that wait; set on built graphs, they are built again (packets prepared before cannot wait).
	void SetFrameWaitTimeline(std::shared_ptr<rhi::TimelinePtr> timeline);
	// Submitting thread: what every batch submitted from now on waits for (0: nothing). The producer must signal it: the waits
	// are on the GPU, and a value never signalled stalls the queue.
	void SetFrameWaitValue(uint64_t value) noexcept { m_frameWaitValue = value; }
	uint64_t FrameWaitValue() const noexcept { return m_frameWaitValue; }
	// Migration shim, removed once every client names runs: Desc::framesInFlight.
	uint32_t FrameSlots() const noexcept { return (std::max)(m_desc.framesInFlight, 1u); }

	// The calling thread's CPU time in the last ExecuteFrame, by phase. Diagnostics only.
	struct FrameTimings {
		double buildUs = 0.0;          // a requested rebuild
		double waitUs = 0.0;           // the frame-pacing wait for the run framesInFlight frames back, and its retirement
		double releaseUs = 0.0;        // deferred upload and descriptor releases
		double beforePrepareUs = 0.0;  // the host's callback (its uploads and inputs)
		double updateUs = 0.0;         // RenderGraph::Update: pass updates and frame preparation
		double executeUs = 0.0;        // RenderGraph::Execute: admission, recording, submission
		double signalUs = 0.0;         // recording the frame's completion points
		// SubmitEpoch (async epochs) fills these instead.
		bool async = false;
		double ticketWaitUs = 0.0;     // waiting for the ticket (zero when it was ready)
		double checkUs = 0.0;          // checking it is current, and preparing it again when not
		double uploadsUs = 0.0;        // admitting the ticket and recording the queued uploads
		double submitUs = 0.0;         // handing the uploads and the ticket to the queue, and committing it
		double postUs = 0.0;           // handing the submitted ticket back to the host's thread (its completion)
		RenderGraph::PersistentExecuteTimings execute{};
	};
	const FrameTimings& LastFrameTimings() const noexcept { return m_lastTimings; }
	const Desc& GetDesc() const noexcept { return m_desc; }

private:
	// One epoch's graph (the only one, of every pass, without epochs).
	struct Instance {
		uint32_t epoch = UINT32_MAX;  // persistent::AllEpochs without epochs
		std::unique_ptr<RenderGraph> graph;
		bool rebuildRequested = true;
		// Why the rebuild was requested (RequestRebuild's reasons, each with its count), logged with the build and cleared by it.
		std::vector<std::pair<std::string, uint32_t>> rebuildReasons;
		uint64_t buildGeneration = 0;  // BuildGeneration(epoch)
		// Its graph failed to build (with none to keep) or to prepare: it runs nothing until a build of it succeeds (EpochFailed).
		bool failed = false;
		std::string failure;
	};
	// A graph an epoch's rebuild replaced, released once the work it submitted is done.
	struct RetiredGraph {
		std::unique_ptr<RenderGraph> graph;
		GpuPoint done;
	};
	static constexpr size_t kNoInstance = ~size_t{0};
	size_t InstanceOf(uint32_t epoch) const noexcept;
	// The first graph built, null with none: what the runtime's services are reached through on the owner thread.
	RenderGraph* AnyGraph() const noexcept;
	// The epoch's graph is to be built before an execution or submission: there is none, or a rebuild was requested without
	// explicit builds.
	// An epoch whose build failed waits for a rebuild request: it is not built again every frame.
	bool BuildDue(size_t index) const noexcept {
		const auto& instance = m_instances[index];
		return (!instance.graph && !instance.failed) || (instance.rebuildRequested && !m_explicitBuilds);
	}
	bool AnyBuildDue() const noexcept;
	// Owner thread, the instance not in the host's thread's hands (async epochs stopped, or the host's thread holding): builds its graph
	// anew. The one it replaces goes to the retired graphs once the new one is built. False when the build failed: the old graph is
	// kept (or there is none), the failure is reported (PollEpochFailures), and nothing is thrown. Host-wide errors still throw.
	bool BuildInstance(size_t index);
	// Owner thread: whether the instance runs nothing (its own failure, or its async epoch's).
	bool InstanceFailed(size_t index) const noexcept;
	// Any thread: queues a failure for PollEpochFailures.
	void ReportEpochFailure(EpochFailure failure);
	struct Async;
	// Host thread: the async epoch's preparation failed. It runs nothing until its instance is built again.
	void FailAsyncEpoch(Async& state, uint32_t epochIndex, std::string error);
	// Owner thread: builds every instance BuildDue names (all: every one requested, explicit builds or not).
	bool BuildDueInstances(bool all);
	// Releases the retired graphs whose work is done (all: waits for the rest).
	void ReleaseRetiredGraphs(bool all);
	// Host-owned queue timelines, made at the first build: graphics, compute, copy.
	void EnsureTimelines();

	Desc m_desc;
	rhi::Device m_device;
	struct Registered {
		std::string id;
		ExtensionFactory factory;
		std::vector<uint32_t> epochs;  // AddExtension's: empty for every graph
	};
	// Whether the extension is made into the epoch's graph (Registered::epochs).
	static bool ExtensionInEpoch(const Registered& extension, uint32_t epoch) noexcept;
	// Marks the instance's graph for a rebuild and records why (Instance::rebuildReasons).
	static void MarkRebuild(Instance& instance, std::string_view reason) noexcept;
	// Marks the graphs the extension is made into.
	void RequestRebuildOf(const Registered& extension, std::string_view reason) noexcept;
	std::vector<Registered> m_extensions;
	std::vector<Instance> m_instances;  // by Desc::epochOrder (one without epochs)
	std::vector<RetiredGraph> m_retiredGraphs;
	std::vector<SharedQueueTimeline> m_timelines;  // EnsureTimelines
	std::shared_ptr<PersistentAliasArena> m_aliasArena;  // Desc::sharedAliasArena: the epochs' graphs' alias pools
	// Run leases (RunRef). A lease is opened for each run and retired when the run's GPU work is done; what it held is freed
	// then. The table lives as long as the host, so a run submitted by the sync path or by one async state is retired by
	// whichever comes next. m_leaseMutex guards the free list and the store; a lease's own fields are its owner's: the
	// thread that opened it until it is submitted, then the thread that retires runs (the host's thread with async epochs,
	// the render thread without).
	struct RunLease;
	mutable std::mutex m_leaseMutex;
	std::vector<std::unique_ptr<RunLease>> m_leaseStore;     // every lease made, by index
	std::unique_ptr<std::atomic<RunLease*>[]> m_leaseTable;  // the same, read without the lock (Runtime::kMaxSlots entries)
	std::vector<uint32_t> m_freeLeases;                      // retired live leases, reused last in first out
	std::vector<RunLease*> m_submittedLeases;                // submitted and not retired, in submission order
	uint32_t m_leasesInFlight = 0, m_leasesPeak = 0;
	// Each revision-driven epoch's published copies' leases, framesInFlight of them, made at the epoch's first use and kept for
	// the host's life, so a copy's LatchBlock regions stay at its index (RunRef::Copy).
	std::vector<std::pair<uint32_t, std::vector<RunLease*>>> m_copyLeases;
	// The sync path's frame pacing: its runs of the last framesInFlight frames, by position.
	std::vector<RunLease*> m_syncLeases;
	// Opens a live lease (or, at Runtime::kMaxSlots, null). With m_leaseMutex held.
	RunLease* OpenLeaseLocked(bool external);
	// Opens a live lease. At Runtime::kMaxSlots in flight, waits for the oldest sealed run and retires it first (throws when none
	// is sealed). Then sizes the statistics ranges for its index. The thread that retires runs.
	RunLease& OpenLease(Async* state);
	// The published copies' leases of the epoch, made on first use. The thread that retires runs.
	std::vector<RunLease*>& CopyLeases(uint32_t epoch);
	// The run's GPU work is done: its statistics read back (the completed-frame callback), then its memory freed. The thread
	// that retires runs; state: the async state whose pools gave its lists (null on the sync path).
	void RetireLease(RunLease& lease, Async* state);
	// Frees what the lease holds (arena regions, upload lists, keep-alives) and returns it: to the free list, or idle for a copy.
	void ReleaseLease(RunLease& lease, Async* state);
	// Retires every submitted lease whose points are reached, and reports the retired submissions (the retired callback).
	// Returns how many it retired.
	size_t RetireReachedLeases(Async* state);
	// Every submission numbered this or less has completed: below the oldest submitted lease whose points are not reached, or seen
	// (the newest submission retirement knows of) when there is none. With m_leaseMutex held.
	uint64_t RetiredSubmissionsLocked(uint64_t seen) const;
	uint64_t m_retiredReported = 0;
	std::atomic<bool> m_recordingTimestamps{false};
	std::atomic<bool> m_boundaryOverridden{false}, m_boundaryEntry{false}, m_boundaryExit{false};
	std::atomic<bool> m_scheduleTrace{false};
	bool m_explicitBuilds = false;
	uint64_t m_frameNumber = 0;
	CompletedFrameCallback m_completedFrame;
	RetiredCallback m_retired;
	uint64_t m_submissions = 0;  // submitting thread (SubmissionsMade)
	RenderGraph::GpuPassRangeBegin m_gpuPassRangeBegin;
	RenderGraph::GpuPassRangeEnd m_gpuPassRangeEnd;
	FrameTimings m_lastTimings{};
	uint64_t m_lastHostFrame = 0;
	std::shared_ptr<rhi::TimelinePtr> m_frameWaitTimeline;  // SetFrameWaitTimeline
	uint64_t m_frameWaitValue = 0;
	mutable std::mutex m_sealedMutex;  // SealedPoint is read on other threads
	GpuPoint m_sealed;
	std::atomic<uint64_t> m_seals{0};  // SealSubmitted calls (Seals); the host's thread reads it
	uint64_t m_buildGeneration = 0;
	std::unique_ptr<Async> m_async;
	EpochFailedCallback m_epochFailed;
	// Failures not handed to the callback yet: the host's thread and the owner thread both report them.
	std::mutex m_failureMutex;
	std::vector<EpochFailure> m_failures;
	// Recording requests (RequestEpochRecording) and the wake of the host's thread: the host's lifetime, never a build's, so a
	// requester on any thread never touches the async state a build replaces.
	struct RecordingQueue;
	std::unique_ptr<RecordingQueue> m_recordingQueue;
	std::unique_ptr<rhi::CompletionWait> m_completionWake;
	bool m_ending = false;  // the destructor: requests still queued are dropped without their callbacks
	// A failed/uncertain GPU completion cannot release submitted owners during
	// a graph rebuild. Keep them until device teardown has stopped execution.
	std::vector<std::shared_ptr<const void>> m_uncertainExecutionOwners;
	std::vector<std::shared_ptr<void>> m_uncertainUploadOwners;
	// Submitting thread: CurrentRun's sources.
	RunRef m_syncRun{}, m_asyncRun{};
	std::optional<RunRef> m_externalRun;
	// Submitting thread, with async epochs: what releases at an external run's start retired, handed to the host's thread with the
	// next submission (its driver frees take the device's memory locks).
	std::vector<std::shared_ptr<void>> m_externalRetired;
	std::vector<uint32_t> m_asyncEpochs;  // restarted with these after a rebuild
	std::vector<uint32_t> m_revisionEpochs;
	std::vector<uint32_t> m_externalEpochs;  // SetExternalEpochs; set on every build's graph
	std::shared_ptr<ExternalEpochExecution> m_externalExecution;  // submitting thread: the last external SubmitEpoch's
	std::shared_ptr<const EpochRecording> m_chosenRecording;  // submitting thread: UseEpochRecording's, until SubmitEpoch takes it
	// Submitting thread, during SubmitTicket's beforeSubmit: the ticket being submitted and its epoch's index (CanUseEpochRecording).
	const RenderGraph::PersistentTicket* m_submittingTicket = nullptr;
	uint32_t m_submittingIndex = 0;
	AsyncStats m_asyncStats{};
	void StartAsync();
	void StopAsync();
	// Owner thread, async epochs running: the instances are given back by the host's thread (their tickets abandoned, their recording
	// requests dropped), which holds still while they are built again, and handed back. The other epochs keep their tickets.
	void RebuildAsyncInstances(std::span<const size_t> indices);
	// Owner thread, async epochs running: the host's thread lets go of the instances to detach and holds still while work runs on
	// this thread (a build, a retired graph's release: both change services every graph shares).
	void HoldAsync(std::span<const size_t> detach, const std::function<void()>& work);
	// Host thread: prepares the epoch's next ticket. reprepare: a stale ticket's run lease, prepared again (its latch regions were
	// written); otherwise a new lease, or for a revision-driven epoch a published copy's (TakeCopyLease).
	void PrepareTicket(Async& state, uint32_t epochIndex, RunLease* reprepare);
	// Host thread: the published copy a revision-driven epoch's next ticket takes: an idle one, else the least recently submitted
	// one whose last run is done or sealed (waited for here: never a run that may still be in the host's stream). Null when every
	// copy holds an unsealed run that is not done: the copies are busy until the next seal or one of their runs completes.
	RunLease* TakeCopyLease(Async& state, uint32_t epochIndex);
	bool AdvanceRecordingRequests(Async& state);
	// Host thread: admits the recording's copy for the ticket's run as one of the revision ticket's candidates.
	void AddRevisionCandidate(Async& state, uint32_t epochIndex, RenderGraph::PersistentTicket& ticket, const EpochRecording& recording);
	void SubmitTicket(Async& state, uint32_t epochIndex, std::shared_ptr<RenderGraph::PersistentTicket> ticket,
		const FrameCallback& beforeSubmit, std::shared_ptr<const void> resourceOwner = {}, std::shared_ptr<const EpochRecording> recording = {});
};

} // namespace org
