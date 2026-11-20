#pragma once

#include "Render/RenderGraph/FrameAliasArena.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
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

namespace org::detail {
class HostUploadExtension;
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
		// Async live epochs keep their recordings (needs reuseRecordings). Each run of a live epoch takes one of the epoch's
		// kept copies: a lease whose LatchBlock regions stay at its index, like a published copy's. A copy keeps the last
		// replayable recording made for it. While that recording is current, the next ticket of the copy binds it: nothing is
		// prepared or recorded. Current means every pass's revision is unchanged, no new version was published since
		// (NoteNewVersions), and the epoch's graph was not built again. Otherwise the ticket is prepared and recorded as
		// before, and its recording is kept if it is replayable. Each reuse clears the copy's latch regions first
		// (LatchBlock::ZeroStableRun), so a block the run's commit does not write reads zeros, as in a new lease.
		// Passes recorded at execution are still recorded at every submission.
		bool reuseLiveRecordings = false;
		// The most kept copies a live epoch makes (reuseLiveRecordings). When every copy holds a run that is neither done nor
		// sealed, the run takes a lease of its own and records as before (AsyncStats::liveSpilled).
		uint32_t liveCopiesMost = 64;
		// A ticket's consecutive batches on one queue are submitted together, in one queue submission
		// (RenderGraph::SetCoalescedSubmissions). Off: one submission per batch.
		bool coalesceSubmissions = false;
		// With coalesced submissions: one handoff list of entry barriers per queue submission, ahead of its first batch, instead of
		// one per batch (RenderGraph::SetMergedHandoffs).
		bool mergeHandoffs = false;
		// With coalesced submissions: a ticket's uploads on the graphics queue go in its first queue submission when all its batches
		// are on that queue, instead of in a submission of their own (RenderGraph::SetUploadsWithBatches).
		bool uploadsWithBatches = false;
		// With inline recording at execution: a kept recording's late batch made only of passes keyed by their skip answers
		// (gated copies and clears) keeps its list while the answers are unchanged (RenderGraph::SetKeptGatedInline).
		bool keepGatedInline = false;
		// Async epochs: a ticket's staged uploads whose destination it first uses on another queue than graphics (async compute)
		// are recorded on that queue, just ahead of the ticket's first batch there, instead of on the graphics queue; and a batch
		// on another queue waits for the graphics queue's uploads only when it uses what they write (RenderGraph::
		// RecordTakenUploads' queue-scoped uploads). Off: every upload on the graphics queue, and every batch on another queue
		// waits for them.
		bool queueScopedUploads = true;
		// A batch waits for the frame wait (SetFrameWaitValue) only when one of its passes declares a late-bound buffer whose
		// versions are ready at the frame wait (VersionedBuffer::LateReady::FrameWait). Off: every batch waits for it.
		bool scopedFrameWait = false;
		// Alias placements that are no longer current (another graph's build grew a pool they are in) are reported by
		// StaleAliasPlacementEpochs instead of being rebuilt at the next build point. For an application that builds its epochs'
		// graphs itself (BuildEpochGraph, AttachEpochGraph). Off: the build point rebuilds those graphs.
		bool reportStaleAliasPlacements = false;
		// Async epochs: SubmitEpoch runs the caller's commit and the revision check on the calling thread, then hands the rest of
		// the submission to a submission thread of the host's, in order. That rest is the upload lists, the admission, the passes
		// recorded at execution and the queue submissions. External epochs still submit on the calling thread. A queue user that
		// must come after the epochs submitted so far drains the thread first (DrainSubmissions).
		bool submissionThread = false;
		// With a submission thread and coalesced submissions: the tickets queued together are finished as one group, and a
		// ticket's first queue submission joins the one the ticket before it left held, when both are on one queue and it waits
		// for nothing new (RenderGraph::SubmitCarry). Only what is already queued is grouped: the thread never waits for more.
		bool mergeTicketSubmissions = false;
		// With a submission thread: SealSubmitted does not drain the thread. It queues the seal behind the tickets queued so far,
		// and the thread passes it in order. Until then SealedPoint has a pending part (GpuPoint::Pending), and Seals counts the
		// seal already. Off: SealSubmitted drains the thread first.
		bool queuedSeals = false;
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
		SetRecordingTimestampMode(enabled ? RenderGraph::RecordingTimestampMode::Passes : RenderGraph::RecordingTimestampMode::Off);
	}
	// Per pass (Passes) or one range per kept list (Lists; RenderGraph::SetPersistentRecordingTimestampMode).
	void SetRecordingTimestampMode(RenderGraph::RecordingTimestampMode mode) noexcept {
		if (m_recordingTimestamps.exchange(mode, std::memory_order_acq_rel) == mode) return;
		m_hostConfiguration.fetch_add(1, std::memory_order_acq_rel);
		RequestRebuild("recording timestamps switched");
	}
	bool RecordingTimestamps() const noexcept {
		return m_recordingTimestamps.load(std::memory_order_acquire) != RenderGraph::RecordingTimestampMode::Off;
	}
	// Diagnostics: the external queue boundary's entry and exit barriers in place of Desc::queueBoundary's, from the next build
	// (dropping them is unsafe: the graphs' first and last accesses are then unordered against the host's work).
	void SetQueueBoundary(RenderGraph::ExternalQueueBoundary boundary) noexcept {
		m_boundaryEntry.store(boundary.entry, std::memory_order_release);
		m_boundaryExit.store(boundary.exit, std::memory_order_release);
		m_boundaryOverridden.store(true, std::memory_order_release);
		m_hostConfiguration.fetch_add(1, std::memory_order_acq_rel);
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
	/** @brief Owner thread: a rebuild of the epoch's graph was requested, and no build or attach has answered it yet. */
	bool RebuildRequested(uint32_t epoch) const noexcept;
	/**
	 * @brief Owner thread: how many rebuilds of the epoch's graph were requested so far (0 for an epoch the host does not have). It only
	 * grows. An application that builds with BuildEpochGraph compares it across a build: a request made after the build started is not
	 * answered by attaching it.
	 */
	uint64_t RebuildRequests(uint32_t epoch) const noexcept;
	/** @brief Owner thread: the reasons of the epoch's rebuild requests not answered yet, each with its count ("" with none). */
	std::string RebuildReasons(uint32_t epoch) const;

	// Application-driven builds. The two primitives below let an application build an epoch's graph wherever it likes (a worker
	// thread of its own) while the host keeps running the old one, and attach it at a point it chooses. The policy is the
	// application's: which thread builds, when, what supersedes what, and what a failure does. RequestRebuild and
	// BuildIfRequested stay the synchronous path.
	/**
	 * @brief A graph built for one epoch and not attached yet (BuildEpochGraph). Dropping it destroys the graph on the dropping
	 * thread. It never ran, so that touches nothing the running graphs use. Drop or attach it before the host is destroyed.
	 */
	class EpochGraph {
	public:
		~EpochGraph();
		EpochGraph(const EpochGraph&) = delete;
		EpochGraph& operator=(const EpochGraph&) = delete;
		uint32_t Epoch() const noexcept { return m_epoch; }
		/** @brief False when the build failed. Error says why, and there is nothing to attach. */
		bool Built() const noexcept { return static_cast<bool>(m_graph); }
		const std::string& Error() const noexcept { return m_error; }
		/** @brief The build's CPU time, on the thread that built it. */
		double BuildMilliseconds() const noexcept { return m_buildMs; }
		/** @brief The graph, for diagnostics before it is attached (its schedule trace, say). Null when the build failed. */
		RenderGraph* Graph() noexcept { return m_graph.get(); }
	private:
		friend class GraphHost;
		EpochGraph() = default;
		uint32_t m_epoch = UINT32_MAX;
		size_t m_index = 0;
		std::unique_ptr<RenderGraph> m_graph;
		detail::HostUploadExtension* m_uploads = nullptr;  // owned by m_graph
		// The host's state the build saw. AttachEpochGraph compares it with the state at the attach.
		uint64_t m_hostConfiguration = 0, m_epochConfiguration = 0, m_requests = 0, m_serial = 0;
		std::string m_reason, m_error;
		double m_buildMs = 0.0;
	};
	/**
	 * @brief Any thread: builds a new graph for the epoch and returns it detached. The host and its graphs are not touched: the old
	 * graph keeps running, preparing and submitting meanwhile.
	 *
	 * It does everything the host's own build does: the extensions made for the epoch, the shared queue timelines, the epoch
	 * filter, external epochs, the alias arena, the settings, CompileStructural and Setup. It also does the work the first
	 * preparation would do inline. It lowers every pass, waits for the passes' pipelines, and builds and installs the first schedule.
	 * So AttachEpochGraph only swaps. The alias placement waits for the graph's first preparation (on the host's thread with async
	 * epochs): it re-materializes transients the epoch's running graph shares. The steps that touch objects every graph shares (the statistics
	 * query heap, the host's upload pass, the upload service's registry) wait for the attach.
	 *
	 * The extension factories run on the calling thread (see "Extension factories" in persistent-epochs.md). Builds of different
	 * epochs may run in parallel. Two builds of one epoch may run in parallel too, each making a graph of its own. A graph can be
	 * attached only if its build started after the build of the graph the epoch runs. A build that throws returns a graph that
	 * is not Built(), with the error. Throws std::invalid_argument for an epoch the host does not have. The host must outlive
	 * the call. The build blocks while its pipelines compile on the runtime's task service, so call it from a thread of your own
	 * (or a pool with workers to spare), never from the last free worker of a small pool.
	 */
	std::unique_ptr<EpochGraph> BuildEpochGraph(uint32_t epoch, std::string_view reason = {});
	// Why a build was refused at its attach or staging. Counted per reason (AttachStats).
	enum class AttachRefusal : uint8_t {
		None,
		NoGraph,             // no graph was given
		UnknownEpoch,        // the epoch is not one of the host's
		Duplicate,           // a set names the epoch twice
		NotStaged,           // no graph is staged for the epoch
		BuildFailed,         // the build failed
		HostConfiguration,   // the host's configuration changed since the build started
		EpochConfiguration,  // the epoch's extensions or external state changed since the build started
		NewerAttached,       // the epoch runs a graph whose build started later
		NewerStaged,         // a graph whose build started later is staged
		BeingSubmitted,      // the call is inside a submission
		SharedSetupFailed,   // the graph's shared setup failed
		Count
	};
	/** @brief A short fixed name for the reason, for counters and logs. */
	static const char* AttachRefusalName(AttachRefusal refusal) noexcept;
	struct AttachResult {
		bool attached = false;
		std::string reason;  // why it was rejected
		AttachRefusal refusal = AttachRefusal::None;
		uint32_t epoch = UINT32_MAX;  // the epoch refused (a set names its first refused epoch)
	};
	/**
	 * @brief Owner thread, where the epoch is not being submitted: attaches a graph BuildEpochGraph made. On success `graph` is
	 * taken. Otherwise it stays with the caller, who drops it or builds again.
	 *
	 * The swap is small. The old graph stops and is released once the work it submitted completes (no device wait). The epoch's
	 * failure is cleared and its build generation moves (BuildGeneration). With async epochs the host's thread drops its
	 * unsubmitted tickets of the old graph. The host's thread is held only for the swap and the shared setup steps, never for a
	 * build. A rebuild requested before the build started is answered. One requested since stays requested.
	 *
	 * Rejected, with a reason, when:
	 * - the build failed,
	 * - the host's configuration changed since the build started: an extension of the epoch added or removed, its epoch made
	 *   external or not, the frame-wait timeline, the queue boundary, recording timestamps, DestroyGraph,
	 * - the epoch runs a graph whose build started after this one's.
	 */
	AttachResult AttachEpochGraph(std::unique_ptr<EpochGraph>& graph);
	/**
	 * @brief Staged builds: a built graph the host's thread records for before it runs (persistent-epochs.md, "Staged graphs").
	 *
	 * Owner thread, where the epoch is not being submitted. The graph is checked as AttachEpochGraph checks it, then handed to the host's
	 * thread as the epoch's staged graph. The attached graph keeps preparing and submitting. A recording request that names the staged
	 * generation (RequestEpochRecording's generation) is recorded on the staged graph. Its recording is current once
	 * AttachStagedEpochGraphs swaps the graph in, so a caller can switch graph and recordings in one step at a frame boundary.
	 *
	 * From the staged graph's first recording, its alias placement may move transients the attached graph binds. So while a graph is
	 * staged, the attached graph keeps the backings it bound (RenderGraph::FreezeAliasedBindings) and takes no new recording: a request
	 * for it is dropped (EpochRecordingDropped). Its kept recordings still submit.
	 *
	 * A second staging replaces the first, whose graph is retired. AttachEpochGraph on the epoch drops a staged graph too. Without async
	 * epochs, or for an epoch with no graph, the graph is attached at once (StageResult::attached).
	 */
	struct StageResult {
		bool staged = false;      // the host's thread records it for the staged generation from now on
		bool attached = false;    // no staging was needed: it was attached (no async epochs, or the epoch had no graph)
		uint64_t generation = 0;  // the build generation it runs at once attached
		std::string reason;       // why it was rejected (neither staged nor attached)
		AttachRefusal refusal = AttachRefusal::None;
	};
	StageResult StageEpochGraph(std::unique_ptr<EpochGraph>& graph);
	/** @brief Owner thread: the staged graph's build generation (0 with none). */
	uint64_t StagedGeneration(uint32_t epoch) const noexcept;
	/**
	 * @brief Owner thread, at a frame boundary: swaps the staged graphs of several epochs in as one step (frame revisions, rule R3).
	 *
	 * Every epoch is checked before anything changes:
	 * - it is one of the host's, and named once;
	 * - a graph is staged for it;
	 * - the host's and the epoch's configuration are unchanged since the staged build started;
	 * - it runs no graph whose build started later;
	 * - the call is not inside a submission.
	 *
	 * If any check fails, nothing is attached and the result names the first refused epoch and why. A refusal drops no staged
	 * graph: the caller decides what to do. Otherwise the host's thread is held once and every staged graph is swapped in with its
	 * recordings. An empty set attaches nothing and succeeds.
	 */
	AttachResult AttachStagedEpochGraphs(std::span<const uint32_t> epochs);
	/**
	 * @brief Any thread: the epochs whose attached or staged graph declares any of `slots` (a closure's members). In the host's epoch
	 * order, each once.
	 *
	 * A graph's declared versioned resources are set by its first build and kept current by its preparations
	 * (RenderGraph::PersistentDeclaredVersions). A graph built on the owner thread declares nothing until its first preparation.
	 * Resources match by IVersionedResource::Key, so a clone matches its original. Null entries are skipped.
	 */
	std::vector<uint32_t> EpochsDeclaring(std::span<const IVersionedResource* const> slots) const;
	/** @brief Totals since the host was made: builds attached or staged, staged sets attached, and refusals by reason. */
	struct AttachStats {
		uint64_t attached = 0;    // AttachEpochGraph and AttachStagedEpochGraphs, per epoch
		uint64_t staged = 0;      // StageEpochGraph
		uint64_t setsAttached = 0;  // AttachStagedEpochGraphs calls that attached
		std::array<uint64_t, static_cast<size_t>(AttachRefusal::Count)> refusals{};
	};
	/** @brief Any thread. */
	AttachStats AttachStatsTotal() const noexcept;
	/**
	 * @brief Owner thread: the epochs whose graph has alias placements in a pool generation another graph replaced (the pool
	 * grew). With Desc::reportStaleAliasPlacements the application rebuilds them (BuildEpochGraph). Otherwise the next build point
	 * does.
	 */
	std::vector<uint32_t> StaleAliasPlacementEpochs() const;

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
	// ExecuteFrame. Each ticket is admitted when it is submitted. Empty stops it (and waits for the thread); an
	// epoch's rebuild pauses that epoch alone.
	// revisionEpochs (a subset of epochs): recorded ahead for a revision (RequestEpochRecording) and submitted with that
	// recording (SubmitEpoch's recording). Their tickets carry the admission alone; nothing about them is recorded or prepared
	// again at submission. They need Desc::reuseRecordings' replayable recordings.
	void SetAsyncEpochs(std::vector<uint32_t> epochs, std::vector<uint32_t> revisionEpochs = {});
	bool AsyncEpochs() const noexcept { return static_cast<bool>(m_async); }
	/**
	 * @brief Owner thread, before SetAsyncEpochs (it takes effect when the host's thread starts, and is kept across rebuilds): a live
	 * async epoch that runs several times a frame keeps up to `most` tickets prepared, as many as it ran in the last frame. Its
	 * back-to-back runs then find their tickets ready instead of each waiting for a preparation.
	 *
	 * - 1 (the default): the next ticket is prepared once the last one was submitted.
	 * - The extra tickets are prepared while earlier ones of the epoch are submitted, so the epoch's passes are prepared on the
	 *   host's thread while the render thread runs the epoch's commits and checks its passes' revisions. Enable it only for epochs
	 *   whose passes' Prepare and Revision read nothing a commit writes.
	 * - Each ticket holds a run lease. Only the first one queued has its upload lists begun on the host's thread; the render
	 *   thread begins the others' when it submits them.
	 * - When a ticket was stale at submission, the queued ones that are stale too are dropped and prepared again.
	 * - A revision-driven epoch always keeps one.
	 */
	void SetEpochTicketsAhead(uint32_t epoch, uint32_t most);
	/**
	 * @brief Owner thread, before the listed epochs' graphs are built: their transients get pools of their own instead of the shared
	 * alias arena (Desc::sharedAliasArena). Another epoch's transient then never takes their memory between two of their
	 * executions, so their admissions find no alias activation and can take a predicted plan. Kept across rebuilds.
	 */
	void SetPrivateAliasEpochs(std::vector<uint32_t> epochs);
	/** @brief An epoch's place in the frame (SetFramePositions). */
	struct FramePosition {
		uint32_t epoch = 0;
		uint32_t position = 0;  // the stage's place in the frame; epochs of one stage share it
		bool ownTransients = false;  // the epoch's own transients go in the frame alias arena, in a range for its position
	};
	/**
	 * @brief Owner thread, before the listed epochs' graphs are built: each epoch's place in the frame, and the host's frame alias
	 * arena (FrameAliasArena, made by the first call).
	 * - An epoch with ownTransients puts its transients in the arena, reserved for its position. Epochs whose positions differ
	 *   share memory. It leaves the shared alias arena and SetPrivateAliasEpochs.
	 * - Every listed epoch's graphs count for the frame transients' spans (AddFrameTransients): a resource's span is the
	 *   positions of the graphs that use it, first to last. An epoch that runs several times a frame keeps one position.
	 * Kept across rebuilds.
	 */
	void SetFramePositions(std::vector<FramePosition> positions);
	/** @brief The frame alias arena SetFramePositions made; null before. */
	std::shared_ptr<FrameAliasArena> FrameArena() const;
	/**
	 * @brief Frame transients: textures several epochs' graphs use within a frame, placed together in the frame alias arena. Their
	 * span is derived, never declared: each graph's alias plan (before it runs) reports the resources its passes use, and a
	 * texture's span grows to cover that graph's position (a graph of an epoch with no position: the whole frame). It starts empty
	 * (nothing uses it yet), so it may sit over other ranges' bytes, and only grows, in place: a placement never moves. Ranges
	 * placed later avoid it where its span conflicts with theirs. A growth into bytes another range holds at those positions fails
	 * that graph's plan, with both named (its content would not survive there).
	 *
	 * Textures made again under the same label (a resize) start with the span the last ones reached.
	 *
	 * The textures must be unmaterialized; they are materialized at their places now. The placement lasts as long as the
	 * returned object. Null when the host has no frame arena (SetFramePositions was not called): the caller materializes them.
	 * Throws when the memory cannot be allocated. Any thread.
	 */
	std::shared_ptr<const void> AddFrameTransients(std::span<const std::shared_ptr<PixelBuffer>> textures, std::string label);
	/** @brief A frame transient's label and its span now, by its resource name; empty for anything else. Any thread. */
	struct FrameTransientInfo {
		std::string label;
		AliasSpan span = AliasSpan::None();
	};
	std::optional<FrameTransientInfo> FrameTransientOf(std::string_view name) const;
	/** @brief Every frame transient group: its label, span and member names. Any thread. */
	struct FrameTransientGroup {
		std::string label;
		AliasSpan span = AliasSpan::None();
		std::vector<std::string> members;
	};
	std::vector<FrameTransientGroup> FrameTransientGroups() const;
	/**
	 * @brief Submitting thread, for a revision-driven async epoch: its runs take their published copies by the caller's index
	 * from now on, starting with copy `copy` for the next SubmitEpoch.
	 *
	 * A host that keys per-run data by its own frame slot (a slot is the frame number modulo Desc::framesInFlight) calls this
	 * before each submission with the slot. The copy's LatchBlock regions then belong to that slot, so the host can write them
	 * by slot (EpochCopyRun) and a slot's recording always reads the same place.
	 *
	 * - The ticket prepared ahead takes the copy after the one submitted last. When the caller names another copy, the ready
	 *   ticket is given back and one for the named copy is prepared, and SubmitEpoch waits for it (AsyncStats::copyRepins).
	 * - Taking a copy waits for its last run as for any copy: a run behind a seal is waited for on the host's thread, and a
	 *   run neither sealed nor done makes SubmitEpoch throw EpochCopiesBusyError.
	 * Throws std::invalid_argument for an epoch that is not a revision-driven async epoch. Nothing before async epochs start.
	 */
	void SetEpochCopy(uint32_t epoch, uint32_t copy);
	/**
	 * @brief Any thread, after SetAsyncEpochs listed the epoch as revision-driven: the run of its published copy `copy`
	 * (RunRef::Copy of the copy's lease). Every submission of that copy names this run, so what a host writes for it
	 * (LatchBlock::Write, run gates) is what the copy's recording reads at its next submission. Write it only while the
	 * copy's last run is done: inside SubmitEpoch's beforeSubmit for that copy. Throws std::invalid_argument otherwise.
	 */
	RunRef EpochCopyRun(uint32_t epoch, uint32_t copy) const;
	/**
	 * @brief Any thread, after SetAsyncEpochs listed the epoch as revision-driven: the staging run of its published copy `copy`. It
	 * is a lease index of its own, never handed to a run and never read by a recording. A host that gathers a copy's per-run values
	 * at several points before it takes the copy (its frame's hooks) writes them here (UseRun), then copies them into the copy's run
	 * inside the copy's beforeSubmit (LatchBlock::CopyStableRun). No GPU work reads the staging regions, so they can be written at
	 * any time. Throws std::invalid_argument for an epoch without published copies.
	 */
	RunRef EpochStagingRun(uint32_t epoch, uint32_t copy) const;
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
	 * @brief Passes recorded at execution (ExternalPassDesc::RecordAtExecution) of the epochs that are not external are recorded
	 * inline (RenderGraph::SetInlineAtExecution): SubmitEpoch records each such batch on the calling thread, after the commit
	 * (beforeSubmit), into a primary list of its own, and submits it in its place in the graph's order with the graph's barriers.
	 * Nothing is taken or executed by the caller. For a host that owns its queues; external epochs keep their behaviour. Every epoch
	 * whose setting changes is built again; kept across rebuilds.
	 */
	void SetInlineAtExecution(bool enabled);
	/** @brief Whether SetInlineAtExecution is on. */
	bool InlineAtExecution() const;
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
	// copy: record that published copy alone (a host that pins its runs to copies, SetEpochCopy). The recording's other copies stay
	// null, and it is a candidate only for runs of that copy. A copy whose last run is done is recorded at once, so a host that asks
	// for the copy it submits next waits for no GPU work.
	// generation: the build to record on (0: the attached one). The staged graph's generation (StageEpochGraph) records on the staged
	// graph. Any other generation that is not the attached one completes with EpochRecordingDropped.
	bool RequestEpochRecording(uint32_t epoch, std::shared_ptr<const IHostExecutionData> hostData, EpochRecordingCallback done,
		uint32_t copy = RunRef::kNoCopy, uint64_t generation = 0);
	// Render thread, for an async epoch it did not submit this frame: its ready ticket, prepared and not submitted, goes back to the
	// host's thread (abandoned, its run lease freed, a new one prepared). False when no ticket was ready (or the mailbox is full: it
	// stays).
	bool ReleaseEpochTicket(uint32_t epoch);
	/**
	 * @brief Render thread: CurrentRun() is `run` from now until EndHeldRun, outside a ticket's beforeSubmit. For a staging run
	 * (EpochStagingRun): what is written for CurrentRun() meanwhile goes to its regions.
	 */
	void UseRun(const RunRef& run) noexcept { m_heldRun = run; }
	void EndHeldRun() noexcept { m_heldRun.reset(); }
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
	// Owner thread: moves with every graph build, of any epoch.
	uint64_t BuildGeneration() const noexcept { return m_buildGeneration; }
	// Owner thread: moves with every build of the epoch's graph: a recording (RequestEpochRecording) is of the build it was made on,
	// and no later one's tickets take it. 0 for an epoch the host does not have.
	uint64_t BuildGeneration(uint32_t epoch) const noexcept;
	// Owner thread: whether a recording is of its epoch's graph as built now - with explicit builds, what decides before a frame's
	// submissions whether they can take it (until the next BuildIfRequested).
	bool EpochRecordingCurrent(const EpochRecording& recording) const noexcept;
	// Any thread: whether copy `copy` of the recording was recorded with the versions its versioned slots have now (a recording
	// for host data that names no versions binds the current ones). False for a copy it does not have. A host whose revision does
	// not name versions checks this instead of counting NoteNewVersions: a version of a resource the epoch does not use leaves it
	// true.
	bool EpochRecordingVersionsCurrent(const EpochRecording& recording, uint32_t copy) const noexcept;
	// Any thread: the versioned slot that makes copy `copy` of the recording not current, by name (empty when none).
	std::string EpochRecordingStaleVersion(const EpochRecording& recording, uint32_t copy) const;
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
		// Desc::queueScopedUploads: staged copies recorded on another queue than graphics, ahead of the batch that first uses them;
		// and batches off the graphics queue spared the wait for the graphics queue's uploads (they use nothing those write).
		uint64_t queueUploadCopies = 0, sparedUploadWaits = 0;
		// Of the staged destinations that are a late-bound version's backing (VersionedBuffer::CreateLateBound): those recorded on a declaring
		// batch's queue, and those a batch off the graphics queue waited for.
		uint64_t lateRoutedUploads = 0, lateUploadWaits = 0;
		// Upload destinations no batch declared. Each made every batch of its ticket off the graphics queue wait for the uploads.
		uint64_t undeclaredUploads = 0;
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
		// Revision-driven tickets given back at submission because they held another copy than the one the caller named
		// (SetEpochCopy), each replaced by a ticket for the named copy.
		uint64_t copyRepins = 0;
		// Tickets prepared ahead (SetEpochTicketsAhead) dropped before their submission, because one before them was stale.
		uint64_t staleAhead = 0;
		// Desc::reuseLiveRecordings: live tickets that bound a kept recording (nothing prepared or recorded), live recordings
		// kept for a copy, live recordings made for a copy that were not replayable, and live runs that found no copy free.
		uint64_t liveReused = 0, liveKept = 0, liveUnkeepable = 0, liveSpilled = 0;
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
	// The epochs' graphs' shared alias pools (Desc::sharedAliasArena), made with the first graph; null without.
	std::shared_ptr<PersistentAliasArena> AliasArena() const noexcept { return m_aliasArena; }
	/**
	 * @brief Submitting thread: the run the host may write per-run data for now (LatchBlock regions, CPU payloads). Inside
	 * SubmitEpoch's beforeSubmit, the ticket's run; inside ExecuteFrame's beforePrepare, the execution's run; otherwise the run
	 * UseRun named; otherwise the external run (BeginExternalRun) when one is active; otherwise the last run submitted.
	 * RecordingContext::Run names the same run.
	 */
	RunRef CurrentRun() const noexcept {
		if (m_heldRun && !m_submittingTicket) return *m_heldRun;
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
	// The host frame number of the last execution (ExecuteFrame or SubmitEpoch); the completed-frame
	// callback reports frames by it.
	uint64_t LastHostFrame() const noexcept { return m_lastHostFrame; }
	// Submitting thread: the executions submitted so far (ExecuteFrame and SubmitEpoch), each numbered in submission order from 1.
	// Work submitted before this returns has retired once the retired callback reports this value.
	uint64_t SubmissionsMade() const noexcept { return m_submissions; }

	// A point of the GPU's progress: per queue, its timeline (shared: valid after a rebuild) and a value it reaches. Any thread.
	struct GpuPoint {
		using Points = std::vector<std::pair<std::shared_ptr<rhi::TimelinePtr>, uint64_t>>;
		// With a submission thread (Desc::submissionThread), the part of the point that is still in the thread's queue. Its values
		// are not handed out yet. The thread fills them in when it passes the point's place in its queue, then sets passed.
		struct Pending {
			std::atomic<bool> passed{false};
			Points points;  // written before passed is set
		};
		Points points;
		std::shared_ptr<const Pending> pending;
		// Whether the thread has passed the pending part (true without one). Until then the point names work that is not on the
		// device's queues yet.
		bool Passed() const noexcept { return !pending || pending->passed.load(std::memory_order_acquire); }
		bool Reached() const noexcept {
			if (!Passed()) return false;
			if (pending && !Reached(pending->points)) return false;
			return Reached(points);
		}
		// False on a timeout or a failed wait. A pending part is waited for first: the thread passes it without waiting for
		// anything this thread does, so the wait ends.
		bool Wait(uint32_t timeoutMs = UINT32_MAX) const noexcept {
			if (pending) {
				if (!WaitPassed(timeoutMs)) return false;
				if (!Wait(pending->points, timeoutMs)) return false;
			}
			return Wait(points, timeoutMs);
		}
		// Every timeline value of the point, the pending part's included. Waits for the thread to pass the pending part first.
		Points Resolve() const {
			Points all = points;
			if (pending) {
				(void)WaitPassed(UINT32_MAX);
				all.insert(all.end(), pending->points.begin(), pending->points.end());
			}
			return all;
		}

	private:
		static bool Reached(const Points& a_points) noexcept {
			for (const auto& [timeline, value] : a_points)
				if (timeline && *timeline && (*timeline)->GetCompletedValue() < value) return false;
			return true;
		}
		static bool Wait(const Points& a_points, uint32_t timeoutMs) noexcept {
			for (const auto& [timeline, value] : a_points)
				if (timeline && *timeline && (*timeline)->HostWait(value, timeoutMs) != rhi::Result::Ok) return false;
			return true;
		}
		bool WaitPassed(uint32_t timeoutMs) const noexcept {
			if (Passed()) return true;
			if (timeoutMs == UINT32_MAX) {
				while (!pending->passed.load(std::memory_order_acquire)) pending->passed.wait(false, std::memory_order_acquire);
				return true;
			}
			const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
			while (!pending->passed.load(std::memory_order_acquire)) {
				if (std::chrono::steady_clock::now() >= until) return false;
				std::this_thread::yield();
			}
			return true;
		}
	};
	// Submitting thread: the point every queue reaches once everything submitted so far (ExecuteFrame, SubmitEpoch) is done.
	// Submitted to the host, not always to the device: a host that queues its submissions behind a stream (an external queue
	// boundary) may hold them until its next flush. So a producer whose signal later work of the same frame waits for must not
	// wait for this point: the work it names may be queued behind that later work, or flushed only once the frame ends. It
	// waits for SealedPoint instead.
	// With a submission thread, a point taken while tickets are still queued there has a pending part (GpuPoint::Pending): it is not
	// reached, and its waits wait, until the thread has submitted them.
	GpuPoint SubmittedPoint() const;
	// Submitting thread, at a frame boundary: everything submitted so far is on the device's queues (the host flushed its stream).
	// With a submission thread, it drains the thread first, so the sealed point is complete. With Desc::queuedSeals it queues the
	// seal on the thread instead, behind the tickets queued so far.
	// Seals SubmittedPoint's current value as SealedPoint, and counts a frame boundary (Seals): a run is waited for only once a seal
	// came after it. Also trims the frame arena's idle chunks.
	void SealSubmitted();
	// Any thread: the SealSubmitted calls so far, queued seals included.
	uint64_t Seals() const noexcept { return m_sealsPushed.load(std::memory_order_acquire); }
	// Any thread: the point SealSubmitted last captured (none before the first). It names only work already on the device, and
	// none of the current frame's, so a mid-frame producer may wait for it whatever the frame's later work waits for.
	// With a queued seal the point has a pending part until the thread passes the seal. Its waits wait for the thread first.
	GpuPoint SealedPoint() const;
	// Any thread: the last seal the thread has passed. It never has a pending part, so its values can be read at once. It may be
	// a frame older than SealedPoint. Use it to ask whether a point is sealed, not to wait for the frame boundary.
	GpuPoint PassedSealedPoint() const;
	// Render owner: runs a_closure on the submission thread, in order after everything queued so far. Without a submission thread
	// it runs at once on the calling thread. A closure that throws is logged. Waits for room when the queue is full.
	void EnqueueOnSubmitThread(std::function<void()> a_closure);
	// Any thread, for a failure message only: the host's thread's state (each async epoch's pause and ticket, the run leases in
	// flight and what each waits for, the queued requests, the wake), read racily.
	// a_pollTimelines false: no RHI call and no blocking lock. Use it where the caller may hold an RHI lock, such as inside a
	// queue submission hook (completed values then read "?").
	std::string DescribeAsyncState(bool a_pollTimelines = true) const;
	// A thread waiting for recording requests: wakes the host's thread when it sleeps with one queued (a lost wake, logged once).
	// True when it did.
	bool NudgeHostThread();
	// For a profiler that samples the host's own threads: their native handles (std::thread::native_handle, a HANDLE on
	// Windows). Null when the thread does not run. Read them after the host's threads started; the host keeps them.
	void* SubmitThreadNativeHandle() noexcept;
	void* HostThreadNativeHandle() noexcept;

	// The frame-wait timeline (RenderGraph::SetFrameWaitTimeline): a host producer of per-frame values signals it once per frame,
	// after the values it wrote for that frame, and every batch submitted from then on waits for SetFrameWaitValue's value. Set
	// once, before the frames that wait; set on built graphs, they are built again (packets prepared before cannot wait).
	void SetFrameWaitTimeline(std::shared_ptr<rhi::TimelinePtr> timeline);
	// Submitting thread: what every batch submitted from now on waits for (0: nothing). The producer must signal it: the waits
	// are on the GPU, and a value never signalled stalls the queue.
	void SetFrameWaitValue(uint64_t value) noexcept { m_frameWaitValue = value; }
	uint64_t FrameWaitValue() const noexcept { return m_frameWaitValue; }
	// Desc::framesInFlight (at least 1): the frames the CPU runs ahead, and a revision-driven epoch's published copies.
	uint32_t FramesInFlight() const noexcept { return (std::max)(m_desc.framesInFlight, 1u); }

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

	// The submission thread (Desc::submissionThread).
	// Any thread: whether the host has one.
	bool SubmissionThreadOn() const noexcept { return static_cast<bool>(m_submitQueue); }
	// Blocks until every submission handed to the thread so far is on its queue. Call it before work that must reach the GPU after
	// those submissions: another queue user's submission, a present, a host wait on the GPU. why names the caller in the
	// statistics. It returns at once when nothing is queued, without the thread, and on the thread itself.
	void DrainSubmissions(const char* why = "unnamed");
	// Diagnostics since the last call. Any thread.
	struct SubmitThreadStats {
		uint64_t tickets = 0;             // tickets the thread finished
		uint64_t groups = 0;              // groups of tickets finished together (those queued at once)
		uint64_t joined = 0;              // tickets whose last queue submission the next ticket's first joined
		uint64_t discarded = 0;           // tickets it gave back unsubmitted after a failure
		uint64_t peakDepth = 0;           // the most entries queued or in progress at once
		uint64_t backpressureWaits = 0;   // pushes that waited for room
		double backpressureUs = 0.0;      // the time those waited
		uint64_t drains = 0;              // drains that had to wait for the thread
		double drainUs = 0.0;             // the time those waited
		double busyUs = 0.0;              // the thread's time on entries
		std::vector<std::pair<std::string, std::pair<uint64_t, double>>> drainsByCaller;  // why -> (waits, us)
	};
	SubmitThreadStats TakeSubmitThreadStats();

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
		// Read by BuildEpochGraph on any thread. requests counts rebuild requests (MarkRebuild). configuration counts changes to
		// what the epoch's graph is built from (its extensions, whether its epoch is external), moved under m_configMutex.
		std::shared_ptr<std::atomic<uint64_t>> requests = std::make_shared<std::atomic<uint64_t>>(0);
		std::shared_ptr<std::atomic<uint64_t>> configuration = std::make_shared<std::atomic<uint64_t>>(0);
		uint64_t attachedSerial = 0;  // the build serial of the graph it runs (AttachEpochGraph refuses older ones)
		// StageEpochGraph: the graph staged for the epoch and the generation it takes at its attach. lastGeneration is the highest
		// generation given out, attached or staged, so every build's generation is new.
		std::unique_ptr<EpochGraph> staged;
		uint64_t stagedGeneration = 0;
		uint64_t lastGeneration = 0;
	};
	// Owner thread, the host's thread holding or stopped: retires the instance's staged graph and clears the host thread's view of it.
	void RetireStaged(size_t index);
	// Owner thread, the instance not in the host's thread's hands: swaps its staged graph in, with the recordings made on it.
	bool SwapStagedIn(size_t index);
	// Owner thread: publishes what the instance's attached and staged graphs declare, for EpochsDeclaring.
	void PublishDeclared(size_t index);
	// Counts a refusal and returns it as an AttachResult.
	AttachResult Refuse(AttachRefusal refusal, uint32_t epoch, std::string reason) noexcept;
	// EpochsDeclaring's view, per instance (index as m_instances). Written on the owner thread under the mutex.
	struct DeclaredGraphs {
		std::shared_ptr<const RenderGraph::DeclaredVersions> attached, staged;
	};
	mutable std::mutex m_declaredMutex;
	std::vector<DeclaredGraphs> m_declared;
	// AttachStatsTotal's counters.
	std::atomic<uint64_t> m_attachCount{0}, m_stageCount{0}, m_setAttachCount{0};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(AttachRefusal::Count)> m_refusalCounts{};
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
	// What a build of one instance is made from, taken under m_configMutex (any thread).
	struct BuildInputs;
	BuildInputs SnapshotBuildInputs(size_t index);
	// Any thread: makes the instance's graph from a snapshot, through Setup. With detached it also runs the first build
	// (RenderGraph::PrepareFirstPersistentBuild) and defers the shared setup steps. Throws on failure.
	std::unique_ptr<RenderGraph> MakeGraph(const BuildInputs& inputs, bool detached, detail::HostUploadExtension** uploads);
	// Owner thread, the instance not in the host's thread's hands: the graph replaces the instance's. The old one is retired by
	// the work submitted so far.
	void InstallGraph(size_t index, std::unique_ptr<RenderGraph> graph, uint64_t serial, uint64_t generation = 0);
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
	// Moves the configuration of the instances the extension is made into. With m_configMutex held.
	void NoteConfigurationOf(const Registered& extension) noexcept;
	// Guards what a build on another thread reads: m_extensions, m_externalEpochs, m_inlineAtExecution, m_frameWaitTimeline,
	// the GPU pass range callbacks, m_timelines and m_aliasArena. The owner thread writes them under it, and reads them without it.
	mutable std::mutex m_configMutex;
	std::atomic<uint64_t> m_hostConfiguration{0};  // host-wide changes that a graph is built with (EpochGraph)
	std::atomic<uint64_t> m_buildSerial{0};        // every build of any instance takes the next
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
	// Each revision-driven epoch's staging leases, one per published copy (EpochStagingRun), made with the copies and never run.
	std::vector<std::pair<uint32_t, std::vector<RunLease*>>> m_stagingLeases;
	// Each live epoch's kept copies' leases (Desc::reuseLiveRecordings), made as its runs need them and kept for the host's life.
	std::vector<std::pair<uint32_t, std::vector<RunLease*>>> m_liveCopyLeases;
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
	std::atomic<RenderGraph::RecordingTimestampMode> m_recordingTimestamps{RenderGraph::RecordingTimestampMode::Off};
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
	GpuPoint m_sealedPassed;  // the last seal the submission thread passed (PassedSealedPoint)
	// Seals the submission thread has passed. The lease logic reads it: a run is flushed once a seal after it was passed.
	std::atomic<uint64_t> m_seals{0};
	// SealSubmitted calls, queued seals included (Seals). The render owner's commits take it: the seals queued before them.
	std::atomic<uint64_t> m_sealsPushed{0};
	// Where a seal is passed: on the submission thread for a queued seal (a_marker set), else on the calling thread.
	void PassSeal(const std::shared_ptr<GpuPoint::Pending>& a_marker);
	// Submitting thread: an admitted ticket of a_segment, and the frame's end (the seal).
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
	std::optional<RunRef> m_heldRun;  // UseRun's, until EndHeldRun
	// Submitting thread, with async epochs: what releases at an external run's start retired, handed to the host's thread with the
	// next submission (its driver frees take the device's memory locks).
	std::vector<std::shared_ptr<void>> m_externalRetired;
	std::vector<uint32_t> m_asyncEpochs;  // restarted with these after a rebuild
	std::vector<uint32_t> m_revisionEpochs;
	std::vector<std::pair<uint32_t, uint32_t>> m_ticketsAhead;  // SetEpochTicketsAhead: (epoch, most)
	std::vector<uint32_t> m_privateAliasEpochs;  // SetPrivateAliasEpochs, under m_configMutex
	std::vector<FramePosition> m_framePositions;         // SetFramePositions, under m_configMutex
	std::shared_ptr<FrameAliasArena> m_frameAliasArena;  // made by SetFramePositions, under m_configMutex
	// AddFrameTransients: the groups, their derived spans and reservations. Shared with each placement's owner, which removes its
	// group when it goes.
	struct FrameTransients;
	std::shared_ptr<FrameTransients> m_frameTransients;  // made by SetFramePositions, under m_configMutex
	// A graph's plan reports what its passes use (RenderGraph::SetFrameTransientObserver). lane: the epoch's lane in the frame
	// arena (its own transients' ranges), 0 for none.
	void NoteFrameTransientUses(uint32_t epoch, uint32_t position, uint32_t lane, std::span<Resource* const> resources);
	std::vector<uint32_t> m_externalEpochs;  // SetExternalEpochs; set on every build's graph
	bool m_inlineAtExecution = false;        // SetInlineAtExecution; set on every build's graph
	std::shared_ptr<ExternalEpochExecution> m_externalExecution;  // submitting thread: the last external SubmitEpoch's
	std::shared_ptr<const EpochRecording> m_chosenRecording;  // submitting thread: UseEpochRecording's, until SubmitEpoch takes it
	// Submitting thread, during SubmitTicket's beforeSubmit: the ticket being submitted.
	const RenderGraph::PersistentTicket* m_submittingTicket = nullptr;
	AsyncStats m_asyncStats{};
	void StartAsync();
	void StopAsync();
	// Owner thread, async epochs running: the instances are given back by the host's thread (their tickets abandoned, their recording
	// requests dropped), which holds still while they are built again, and handed back. The other epochs keep their tickets.
	void RebuildAsyncInstances(std::span<const size_t> indices);
	// The same, with replace(index) in place of BuildInstance. replace returns true when the instance's graph was replaced.
	void ReplaceAsyncInstances(std::span<const size_t> indices, const std::function<bool(size_t)>& replace);
	// Owner thread, async epochs running: the host's thread lets go of the instances to detach and holds still while work runs on
	// this thread (a build, a retired graph's release: both change services every graph shares).
	void HoldAsync(std::span<const size_t> detach, const std::function<void()>& work);
	// Host thread: prepares the epoch's next ticket. reprepare: a stale ticket's run lease, prepared again (its latch regions were
	// written); otherwise a new lease, or for a revision-driven epoch a published copy's (TakeCopyLease).
	void PrepareTicket(Async& state, uint32_t epochIndex, RunLease* reprepare);
	// Host thread, after a stale ticket's replacement: the epoch's queued tickets that are no longer current are abandoned and
	// their leases freed, so the render thread does not find them stale one by one.
	void DropStaleTickets(Async& state, uint32_t epochIndex);
	// Host thread: the published copy a revision-driven epoch's next ticket takes: an idle one, else the least recently submitted
	// one whose last run is done or sealed (waited for here: never a run that may still be in the host's stream). Null when every
	// copy holds an unsealed run that is not done: the copies are busy until the next seal or one of their runs completes.
	RunLease* TakeCopyLease(Async& state, uint32_t epochIndex);
	// Host thread: the kept copy a live epoch's next run takes (Desc::reuseLiveRecordings). An idle copy whose kept recording is
	// still of the epoch's graph comes first, then any idle copy, then a new copy (up to Desc::liveCopiesMost), then the least
	// recently submitted copy whose last run is sealed (waited for here). Null when none of these exists.
	RunLease* TakeLiveCopy(Async& state, uint32_t epochIndex);
	bool AdvanceRecordingRequests(Async& state);
	// Host thread: admits the recording's copy for the ticket's run as one of the revision ticket's candidates.
	void AddRevisionCandidate(Async& state, uint32_t epochIndex, RenderGraph::PersistentTicket& ticket, const EpochRecording& recording);
	void SubmitTicket(Async& state, uint32_t epochIndex, std::shared_ptr<RenderGraph::PersistentTicket> ticket,
		const FrameCallback& beforeSubmit, std::shared_ptr<const void> resourceOwner = {}, std::shared_ptr<const EpochRecording> recording = {});
	// A submission in two parts. CommitTicket runs on the render thread: the caller's commit (beforeSubmit), the recording
	// choice, and the revision check with its re-preparation, which all read the caller's state. FinishTicket does the rest, in
	// the same order as before: the upload lists, the admission, the submission and the post to the host's thread.
	struct TicketCommit;
	TicketCommit CommitTicket(Async& state, uint32_t epochIndex, std::shared_ptr<RenderGraph::PersistentTicket> ticket,
		const FrameCallback& beforeSubmit, std::shared_ptr<const void> resourceOwner, std::shared_ptr<const EpochRecording> recording);
	// timings and stats: where its times and counts go (the render thread's own, or the submission thread's).
	// carry and carryOut: the submission thread's group of tickets (RenderGraph::SubmitPersistentTicket's). posts: the ticket's
	// completion message goes there instead of to the host's thread, for the caller to post once the group's held submission is
	// on its queue.
	struct DeferredPosts;
	void FinishTicket(Async& state, TicketCommit& commit, FrameTimings& timings, AsyncStats& stats, RenderGraph::SubmitCarry* carry = nullptr,
		bool carryOut = false, DeferredPosts* posts = nullptr);
	// The submission thread (Desc::submissionThread): its queue, made with the host and stopped first by its destructor.
	struct SubmitQueue;
	std::unique_ptr<SubmitQueue> m_submitQueue;
	// The submission thread's held group of batches (RenderGraph::SubmitCarry), across the tickets of one group.
	std::shared_ptr<RenderGraph::SubmitCarry> m_submitCarry;
	// SubmittedPoint's values now: every queue's last value handed out.
	GpuPoint SubmittedPointNow() const;
	// The thread's counts (FinishTicket's stats there), added to TakeAsyncStats.
	std::mutex m_threadStatsMutex;
	AsyncStats m_threadStats{};
	void RunSubmitThread();
	void StopSubmitThread();
	// Render owner: hands a committed ticket to the thread. Waits for room when the queue is full.
	void EnqueueTicket(Async& state, TicketCommit commit);
	// Submission thread: records the first failure. SubmitEpoch throws it from then on, and PollEpochFailures reports it.
	void FailSubmitThread(Async& state, uint32_t epochIndex, const std::string& error);
	// Throws the thread's failure, if it failed.
	void ThrowIfSubmitThreadFailed() const;
};

} // namespace org
