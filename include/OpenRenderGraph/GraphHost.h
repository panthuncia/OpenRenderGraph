#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <rhi.h>

#include "Render/RenderGraph/RenderGraph.h"
#include "Render/RenderGraph/ExternalEpochExecution.h"
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
//   - per-frame-slot retirement: before a slot is reused, its previous frame's GPU work is
//     waited for and the upload pages and descriptors it retired are released (an async epoch that runs many times a
//     frame grows its ring of slots instead of waiting for a run of the same frame: Desc::epochRuns),
//   - an orderly shutdown that waits only for the graphs' own work.
// Not thread-safe: all calls come from the host's render thread. With async epochs (SetAsyncEpochs) the host
// runs one thread of its own that owns the graphs from then on; the render thread submits, and builds a graph only
// while the host's thread has let go of it.
class GraphHost {
public:
	struct Desc {
		// The runtime the host's graphs run on (their device, task service and services). Its slots must cover the host's frame
		// slots (FrameSlots).
		std::shared_ptr<Runtime> runtime;
		RenderGraph::ExternalQueueBoundary queueBoundary{};
		// Frames the CPU may run ahead of the GPU: whole frames of the host's epochs (each epoch of epochOrder once, or as many
		// times as epochRuns says), or of ExecuteFrame calls without epochs. The host's frame slots - the ring every execution
		// takes the next slot of, and waits for that slot's previous execution before reusing - number framesInFlight times the
		// epochs' runs (FrameSlots, FrameSlotsFor), so the ring covers the same frames however many epochs there are.
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
		// Async epochs: keep an epoch's replayable recordings per frame slot and submit them again
		// (RenderGraph::SetPersistentRecordingReuse). A ticket prepared for what a kept recording was recorded for takes it
		// instead of recording, and a ticket the beforeSubmit callback made stale takes the kept recording of what it depends
		// on now instead of being prepared again (a wait). Kept recordings carry no profiler GPU zones.
		bool reuseRecordings = false;
		// Async epochs: a ticket's staged uploads whose destination it first uses on another queue than graphics (async compute)
		// are recorded on that queue, just ahead of the ticket's first batch there, instead of on the graphics queue; and a batch
		// on another queue waits for the graphics queue's uploads only when it uses what they write (RenderGraph::
		// RecordPendingUploads' queue-scoped uploads). Off: every upload on the graphics queue, and every batch on another queue
		// waits for them.
		bool queueScopedUploads = true;
		/**
		 * @brief An async epoch that may run more than once in a host frame, and the most runs a frame it is sized for.
		 *
		 * An async epoch takes the slots of its own ring in turn, and the host's thread waits for a slot's last run on the GPU
		 * before it prepares the slot again. A host that holds its submissions in a stream until the frame's end (an external
		 * queue boundary, external epochs in another API's command list) has not handed a run of the current frame to the
		 * device yet: waiting for it deadlocks, since the render thread is waiting for the host's thread at the epoch's next
		 * run. An epoch listed here has a ring of framesInFlight times runsPerFrame slots (its capacity) and starts using
		 * framesInFlight of them. When the slot it would take next ran too recently, the ring grows by one slot instead of
		 * waiting. "Too recently" counts frame boundaries, which the host declares with SealSubmitted: a slot is waited for
		 * only once its last run is at least max(framesInFlight - 1, 1) seals old. A seal flushed it, and the wait is
		 * backpressure from as many frames back as an epoch that runs once a frame gets. So the ring settles at about its
		 * runs a frame times framesInFlight, and waits only target runs of earlier frames.
		 * At capacity, a slot whose last run was sealed is waited for (safe: it is on the device). When even the oldest slot's
		 * run is still unsealed (more runs since the last seal than the capacity), the epoch's next ticket is not prepared
		 * until the next seal, and SubmitEpoch throws EpochRingFullError for it meanwhile: the caller skips that run, nothing
		 * hangs. A host that lists an epoch here must call SealSubmitted at every frame boundary. Epochs not listed keep a ring
		 * of framesInFlight slots and do not grow. Once the host has sealed, they follow the same rule: a ticket takes the least
		 * recently submitted slot whose last run is done or sealed, and when there is none the ring is full as above. A host that
		 * never seals keeps waiting for any slot. Each epoch must be one of epochOrder; runsPerFrame at least 1.
		 */
		struct EpochRuns {
			uint32_t epoch = 0;
			uint32_t runsPerFrame = 1;
		};
		std::vector<EpochRuns> epochRuns;
	};

	/**
	 * @brief The frame slots a host made with desc has (FrameSlots): framesInFlight times the sum, over epochOrder (one epoch
	 * without), of each epoch's runsPerFrame (Desc::epochRuns; 1 when not listed). The runtime's slots must cover it. Throws for
	 * an invalid Desc::epochRuns (an epoch not in epochOrder, listed twice, or 0 runs), and for more than Runtime::kMaxSlots.
	 */
	static uint32_t FrameSlotsFor(const Desc& desc);

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
	 * @brief What SubmitEpoch throws for an epoch of Desc::epochRuns whose ring is full: every slot of it holds a run the host has
	 * not sealed yet (SealSubmitted), so its next ticket waits for the next seal. Nothing was submitted, the epoch has not failed
	 * (nothing is reported to PollEpochFailures), and its runs go on after the next seal. The caller skips this run (does its
	 * work another way). An EpochFailedError, so a caller that skips a failed epoch skips this too.
	 */
	struct EpochRingFullError : EpochFailedError {
		EpochRingFullError(uint32_t a_epoch, uint32_t a_capacity)
			: EpochFailedError(Message{}, a_epoch, "epoch " + std::to_string(a_epoch) + " ran more than its " + std::to_string(a_capacity)
				+ " ring slots between two frame boundaries (Desc::epochRuns); this run is skipped until the next SealSubmitted"),
			  capacity(a_capacity) {}
		uint32_t capacity = 0;
	};

	// Runs after any rebuild and before the frame is prepared, with the graphs' upload
	// and descriptor services active: the place to queue this frame's uploads
	// (BUFFER_UPLOAD) into graph resources.
	using FrameCallback = std::function<void(RenderGraph&)>;

	// Called from ExecuteFrame once an earlier frame is known complete on the GPU (its slot is about to be
	// reused), after that frame's pass timestamps were read back: the passes whose gpuSampleSerial equals
	// stats.GetFrameSerial() are the ones that frame recorded. frameNumber is that frame's 0-based number.
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

	// An epoch recorded for one revision: a recording per frame slot of the epoch's ring (each async epoch has a run of slots of its
	// own, firstSlot .. + its ring's capacity - 1: framesInFlight, times Desc::epochRuns' runsPerFrame for an epoch listed there; its
	// tickets and its recordings meet there). Immutable.
	// A revision's recordings of an epoch, one per slot of its ring. While the caller holds it, the epoch's tickets admit it as a
	// candidate (RenderGraph::PrepareRevisionTicket), so it can be submitted at any frame; released, it is never admitted again.
	struct EpochRecording {
		uint32_t epoch = 0;
		uint32_t firstSlot = 0;
		std::vector<std::shared_ptr<const RenderGraph::PersistentRecording>> slots;  // by ring position
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
	// revision) on the host's thread, into every slot of its ring once the slot's last work is done (the host's thread waits for
	// that, never the caller; a slot the epoch's admission-only ticket holds is recorded too). The request goes onto a queue of the
	// host's own lifetime, never through the async state a build replaces; the host's thread takes it whole, in posting order.
	// done runs exactly once: on the host's thread once every slot is recorded (the recording carries the build it was recorded on,
	// EpochRecording::buildGeneration), or with the error that stopped it (an epoch that is not an async one is one), or with
	// EpochRecordingDropped when its epoch's graph was built again, or the host's thread stopped, before it was recorded. False only
	// when async epochs were never started or are off (SetAsyncEpochs with none): done is not called then. A live (not
	// revision-driven) async epoch's is made the same way (a slot its live ticket holds once that ticket is submitted, a slot with
	// work in flight once it completes) and admitted beside its tickets' own preparations, while held: the submitting thread may
	// submit it instead (UseEpochRecording). The host's end drops what is queued without calling done (its callers may be gone).
	bool RequestEpochRecording(uint32_t epoch, std::shared_ptr<const IHostExecutionData> hostData, EpochRecordingCallback done);
	// Render thread, for an async epoch it did not submit this frame: its ready ticket, prepared and not submitted, goes back to the
	// host's thread (abandoned, its slot freed, a new one prepared). A live epoch's recording waits for the slot its unsubmitted ticket
	// holds, so one requested for an epoch the caller stopped submitting completes only once the caller gives that ticket back. False
	// when no ticket was ready (or the mailbox is full: it stays).
	bool ReleaseEpochTicket(uint32_t epoch);
	// Submits the epoch at the current point of the host's queue stream. Builds the epoch's graph first when its rebuild is due.
	// Waits for its ticket if the host's thread has not finished it; runs beforeSubmit with the graphs' services active and
	// CurrentFrameSlot() the ticket's slot (write latches and queue uploads there); has the ticket prepared again (and waits)
	// if a pass would now prepare differently; records the uploads queued since the last submission; and
	// hands both to the queue - timeline values are assigned here, in submission order. Throws when the
	// graph failed (on either thread): EpochFailedError when only this epoch's did, and nothing was submitted.
	// resourceOwner pins immutable descriptors/imports named by this execution
	// until all accepted queue submissions complete, even without slot reuse.
	// recording: a revision-driven epoch's (RequestEpochRecording), whose recording for the ticket's slot is submitted; it must
	// match the ticket's publication, bindings and backing version, or this throws (it is never prepared again).
	void SubmitEpoch(uint32_t epoch, const FrameCallback& beforeSubmit = {}, std::shared_ptr<const void> resourceOwner = {},
		std::shared_ptr<const EpochRecording> recording = {});
	// Submitting thread, inside SubmitEpoch's beforeSubmit for a live async epoch: submit this revision recording (one of the
	// ticket's candidates: RequestEpochRecording) instead of the ticket's own preparation, which is abandoned. Its stale check does
	// not apply: the recording is what the caller's revision drew. A recording the ticket did not admit makes SubmitEpoch throw.
	void UseEpochRecording(std::shared_ptr<const EpochRecording> recording) noexcept { m_chosenRecording = std::move(recording); }
	// Submitting thread, inside SubmitEpoch's beforeSubmit for a live async epoch: whether the ticket being submitted would take
	// this recording (UseEpochRecording), by the bind's own rules (RenderGraph::CanBindPersistentTicketRecording) - false, with why,
	// for one recorded for a graph since rebuilt, other bindings, or another backing version. A caller that writes values into what
	// the recording reads decides by this before writing, so SubmitEpoch never throws for it.
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
		// Desc::reuseRecordings: tickets recorded, recordings kept, tickets that took a kept recording instead of recording,
		// and stale tickets made current by a kept recording (not prepared again).
		uint64_t recorded = 0, kept = 0, reused = 0, adopted = 0;
		// Tickets admitted with their prediction's plan, and those planned against the ledgers (RenderGraph::PersistentRecordingStats).
		uint64_t predicted = 0, planned = 0;
		// Revision-driven epochs: slots recorded for requests (RequestEpochRecording), requests completed, and submissions of a
		// revision's recording.
		uint64_t revisionSlotsRecorded = 0, revisionRecordings = 0, revisionSubmissions = 0;
		// Desc::epochRuns: slots added to an epoch's ring instead of waiting for a run too recent, and tickets held for the next seal
		// because every slot of the ring held an unsealed run (EpochRingFullError).
		uint64_t ringGrowths = 0, ringFull = 0;
	};
	AsyncStats TakeAsyncStats() noexcept;

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
	// The frame slot of the execution ExecuteFrame is running (valid from its beforePrepare callback until
	// it returns): the region of a LatchBlock the host may write for it. Also RecordingContext::FrameSlot.
	bool ClosedExecutions() const noexcept { return m_desc.closedExecutions; }
	// The epochs' graphs' shared alias pools (Desc::sharedAliasArena), made with the first graph; null without.
	std::shared_ptr<PersistentAliasArena> AliasArena() const noexcept { return m_aliasArena; }
	// An external frame slot set (BeginExternalFrameSlot) wins outside a ticket's beforeSubmit: the slot an epoch submitted meanwhile
	// writes is its ticket's own.
	uint32_t CurrentFrameSlot() const noexcept {
		if (m_externalSlot != kNoExternalSlot && !m_submittingTicket) return m_externalSlot;
		return m_async ? m_asyncSlot : static_cast<uint32_t>(m_frameNumber % m_frameSlots);
	}
	// Submitting thread: a client that records the work reading its LatchBlocks outside the host's executions (into another
	// submitter's command buffer: Community Shaders' DCLF CPU pipeline, DXVK's own list) writes its latch regions at `slot`
	// (taken modulo FrameSlots()) from now until EndExternalFrameSlot: CurrentFrameSlot() returns it, except inside a ticket's
	// beforeSubmit. The client owns the slot's reuse: it waits, before Begin, until the work that read the slot last has retired
	// (the host's slot waits cover its own submissions only), and it does not use a slot under an execution of the host's that
	// writes the same blocks. The slot's upload pages and staged batches are released here once the host's own last work at the
	// slot has completed too: a client's frames go round every slot, and an execution of the host's releases only the slot it takes.
	void BeginExternalFrameSlot(uint32_t slot);
	void EndExternalFrameSlot() noexcept { m_externalSlot = kNoExternalSlot; }
	bool ExternalFrameSlotActive() const noexcept { return m_externalSlot != kNoExternalSlot; }
	// Submitting thread: the client's work at frame slot `slot` is complete once `timeline` reaches `value`, which the client may signal
	// later (its frame's end). The host's slot waits cover its own submissions alone, so until then an execution of the host's taking
	// the slot leaves the slot's upload pages for a later release (the client's work may still read them; BeginExternalFrameSlot's
	// when the client takes the slot again, if no execution of the host's took it first), and every descriptor slot,
	// resource or backing retired from now on waits for the point too (an execution of the host's mid-frame would otherwise release
	// what the client's pending work still reads).
	void NoteExternalSlotUse(uint32_t slot, std::shared_ptr<rhi::TimelinePtr> timeline, uint64_t value);
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
	// Seals SubmittedPoint's current value as SealedPoint, and counts a frame boundary (Seals): the epochs of Desc::epochRuns
	// size their rings by these.
	void SealSubmitted();
	// Any thread: the SealSubmitted calls so far.
	uint64_t Seals() const noexcept { return m_seals.load(std::memory_order_acquire); }
	// Any thread: the point SealSubmitted last captured (none before the first). It names only work already on the device, and
	// none of the current frame's, so a mid-frame producer may wait for it whatever the frame's later work waits for.
	GpuPoint SealedPoint() const;
	// Any thread, for a failure message only: the host's thread's state (each async epoch's pause, ticket and ring, the recording
	// requests it holds and what each waits for, the queued requests, the wake), read racily.
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
	// The frame slots: Desc::framesInFlight frames of every epoch in the order, each its runs a frame (Desc::epochRuns; one
	// execution a frame without epochs). Fixed when the host is made (FrameSlotsFor): a LatchBlock sized from it stays right.
	uint32_t FrameSlots() const noexcept { return m_frameSlots; }

	// The calling thread's CPU time in the last ExecuteFrame, by phase. Diagnostics only.
	struct FrameTimings {
		double buildUs = 0.0;          // a requested rebuild
		double waitUs = 0.0;           // the host wait for the slot's previous frame, and its completion callback
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
	uint32_t m_frameSlots = 1;  // FrameSlots: Desc::framesInFlight times the epochs in the order
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
	// A slot's last frame: the value each queue's timeline reached with that frame's last batch.
	// Waiting on those gates the slot's reuse; the host adds no submission of its own.
	struct SlotCompletion {
		std::vector<std::pair<std::shared_ptr<rhi::TimelinePtr>, uint64_t>> points;
		uint64_t frameNumber = 0;
		uint64_t submission = 0;  // its number (SubmissionsMade)
		uint64_t seal = 0;        // the seals before it was submitted (Seals)
		bool pending = false;
	};
	std::vector<SlotCompletion> m_slotCompletions;
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
	// Desc::epochRuns' runsPerFrame for an epoch (1 when not listed).
	uint32_t RunsPerFrame(uint32_t epoch) const noexcept;
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
	uint32_t m_asyncSlot = 0;
	static constexpr uint32_t kNoExternalSlot = ~0u;
	uint32_t m_externalSlot = kNoExternalSlot;  // submitting thread: BeginExternalFrameSlot's
	// Submitting thread: NoteExternalSlotUse's timeline and each slot's last point on it (0: none).
	std::shared_ptr<rhi::TimelinePtr> m_externalTimeline;
	std::vector<uint64_t> m_externalSlotValues;
	// Submitting thread: per slot, every queue's value assigned when the host last submitted work for it (SubmittedPoint), which
	// BeginExternalFrameSlot's release waits for without blocking; and, with async epochs, what its releases retired, handed to
	// the host's thread with the next submission (its driver frees take the device's memory locks).
	std::vector<GpuPoint> m_slotSubmitted;
	std::vector<std::shared_ptr<void>> m_externalRetired;
	bool ExternalSlotInUse(uint32_t slot) const noexcept;
	// Submitting thread, after work for `slot` was submitted: m_slotSubmitted's point for it.
	void NoteSlotSubmitted(uint32_t slot);
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
	void PrepareTicket(Async& state, uint32_t epochIndex, int32_t requestedSlot);
	// Host thread: the slot of its ring an epoch's next ticket takes, the least recently submitted one whose last run may be waited
	// for (done, or sealed: never a run that may still be in the host's stream). A ring of Desc::epochRuns grows instead when that run
	// is too recent. UINT32_MAX when no slot qualifies: the ring is full until the next seal or one of its runs completes.
	uint32_t RingSlot(Async& state, uint32_t epochIndex);
	// Host thread: waits for the slot's last run on the GPU. Logs once per epoch if that run is neither done nor sealed.
	void WaitSlot(Async& state, uint32_t slot, uint32_t epochIndex);
	bool AdvanceRecordingRequests(Async& state);
	// Host thread: admits the recording of the ticket's slot as one of the revision ticket's candidates.
	void AddRevisionCandidate(Async& state, uint32_t epochIndex, RenderGraph::PersistentTicket& ticket, const EpochRecording& recording);
	void SubmitTicket(Async& state, uint32_t epochIndex, std::shared_ptr<RenderGraph::PersistentTicket> ticket,
		const FrameCallback& beforeSubmit, std::shared_ptr<const void> resourceOwner = {}, std::shared_ptr<const EpochRecording> recording = {});
};

} // namespace org
