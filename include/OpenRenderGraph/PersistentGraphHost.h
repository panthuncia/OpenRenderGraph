#pragma once

#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <rhi.h>

#include "Render/RenderGraph/RenderGraph.h"

namespace org::runtime {
class ITaskService;
class IUploadService;
class ResourceCleanupQueue;
}

namespace org {

struct IHostExecutionData;

// Drives one RenderGraph in persistent (static) execution for an application that
// embeds ORG in its own renderer, for example a host that adopted another API's
// device. It owns what every such embedding needs and nothing application-specific:
//   - the runtime device registration and a task service,
//   - the built-in upload pass (persistent Pre segment),
//   - persistent execution with the requested external queue boundary,
//   - rebuilding the graph from the registered extension factories on request,
//   - synchronous per-frame Update + Execute (Execute submits before returning),
//   - per-frame-slot retirement: before a slot is reused, its previous frame's GPU work is
//     waited for and the upload pages and descriptors it retired are released,
//   - an orderly shutdown that waits only for the graph's own work.
// Not thread-safe: all calls come from the host's render thread. With async epochs (SetAsyncEpochs) the host
// runs one thread of its own that owns the graph from then on; the render thread only submits.
class PersistentGraphHost {
public:
	struct Desc {
		rhi::Device device;
		rhi::Backend backend = rhi::Backend::Null;
		// Optional; a ThreadPoolTaskService is created when absent.
		std::shared_ptr<runtime::ITaskService> tasks;
		RenderGraph::ExternalQueueBoundary queueBoundary{};
		// Frames the CPU may run ahead of the GPU: whole frames of the host's epochs (each epoch of epochOrder once), or of
		// ExecuteFrame calls without epochs. The host's frame slots - the ring every execution takes the next slot of, and waits
		// for that slot's previous execution before reusing - number framesInFlight times the epochs (FrameSlots), so the ring
		// covers the same frames however many epochs there are.
		uint32_t framesInFlight = 3;
		// Host epochs: the order the host runs them in within its own frame (see
		// RenderGraph::SetPersistentEpochOrder). Passes declare theirs with
		// ExternalPassDesc::Epoch; ExecuteFrame names the one to run.
		std::vector<uint32_t> epochOrder;
		// RenderGraph::SetPersistentClosedExecutions: every execution leaves its resources in their home
		// states and starts with a full barrier, so its admission is independent of what ran before.
		bool closedExecutions = false;
		// Async epochs: keep an epoch's replayable recordings per frame slot and submit them again
		// (RenderGraph::SetPersistentRecordingReuse). A ticket prepared for what a kept recording was recorded for takes it
		// instead of recording, and a ticket the beforeSubmit callback made stale takes the kept recording of what it depends
		// on now instead of being prepared again (a wait). Kept recordings carry no profiler GPU zones.
		bool reuseRecordings = false;
	};

	using ExtensionFactory = std::function<std::unique_ptr<RenderGraph::IRenderGraphExtension>()>;

	explicit PersistentGraphHost(Desc desc);
	~PersistentGraphHost();
	PersistentGraphHost(const PersistentGraphHost&) = delete;
	PersistentGraphHost& operator=(const PersistentGraphHost&) = delete;

	// Extensions are instantiated from their factories each time the graph is
	// (re)built, so they may size resources from current host state.
	void AddExtension(std::string id, ExtensionFactory factory);
	void RemoveExtension(const std::string& id);

	// The next ExecuteFrame builds a new graph (after retiring the current one).
	void RequestRebuild() noexcept { m_rebuildRequested = true; }

	// Explicit builds: a rebuild requested (an extension added or removed, RequestRebuild) waits for the caller's build point,
	// BuildIfRequested; until then every execution, submission and preparation runs the graph as built, whose extensions keep
	// the state they were made with. For a caller that decides once a frame what its recordings cover (RequestEpochRecording,
	// UseEpochRecording): a recording of the current build stays its tickets' until that point. Without, the next execution or
	// submission builds (the default). A graph is built whenever there is none.
	void SetExplicitBuilds(bool explicitBuilds) noexcept { m_explicitBuilds = explicitBuilds; }
	// Owner thread: builds the graph again when a rebuild was requested (or there is none), async epochs stopped and restarted
	// around it; true when it built.
	bool BuildIfRequested();
	// A rebuild was requested and has not been built (with explicit builds, until BuildIfRequested).
	bool RebuildRequested() const noexcept { return m_rebuildRequested; }

	// Runs after any rebuild and before the frame is prepared, with the graph's upload
	// and descriptor services active: the place to queue this frame's uploads
	// (BUFFER_UPLOAD) into graph resources.
	using FrameCallback = std::function<void(RenderGraph&)>;

	// Called from ExecuteFrame once an earlier frame is known complete on the GPU (its slot is about to be
	// reused), after that frame's pass timestamps were read back: the passes whose gpuSampleSerial equals
	// stats.GetFrameSerial() are the ones that frame recorded. frameNumber is that frame's 0-based number.
	using CompletedFrameCallback = std::function<void(uint64_t frameNumber, const runtime::IStatisticsService& stats)>;
	void SetCompletedFrameCallback(CompletedFrameCallback callback) { m_completedFrame = std::move(callback); }

	// GPU ranges around every pass (RenderGraph::SetGpuPassRangeCallbacks), kept across rebuilds. Set them while no
	// frame executes: the host's thread reads them when it prepares.
	void SetGpuPassRangeCallbacks(RenderGraph::GpuPassRangeBegin begin, RenderGraph::GpuPassRangeEnd end);

	// Builds if needed, then prepares and submits one frame. hostData is visible to
	// passes through PassPrepareContext::preparationData for this frame only.
	// With an epoch, only that epoch's passes (and untagged ones) are prepared,
	// admitted, recorded and submitted: the graph was compiled once for the whole
	// host frame, and this is one execution split of it.
	void ExecuteFrame(const IHostExecutionData* hostData = nullptr, const FrameCallback& beforePrepare = {},
		uint32_t epoch = UINT32_MAX, std::shared_ptr<const void> resourceOwner = {});  // persistent::AllEpochs

	// Retires the current graph: stops frame production and waits for its work.
	void DestroyGraph();

	// Async epochs. The host prepares each listed epoch's execution ahead of time, as a ticket
	// (RenderGraph::PreparePersistentTicket), on a thread of its own - which from then on is the only thread
	// that touches the graph: it waits for frame slots, prepares, admits and records tickets, completes them
	// in submission order and retires finished work. The render thread calls SubmitEpoch instead of
	// ExecuteFrame. Requires Desc::closedExecutions. Empty stops it (and waits for the thread); a rebuild
	// stops and restarts it.
	// revisionEpochs (a subset of epochs): recorded ahead for a revision (RequestEpochRecording) and submitted with that
	// recording (SubmitEpoch's recording). Their tickets carry the admission alone; nothing about them is recorded or prepared
	// again at submission. They need Desc::reuseRecordings' replayable recordings.
	void SetAsyncEpochs(std::vector<uint32_t> epochs, std::vector<uint32_t> revisionEpochs = {});
	bool AsyncEpochs() const noexcept { return static_cast<bool>(m_async); }

	// An epoch recorded for one revision: a recording per frame slot of the epoch's ring (async epoch i uses slots
	// i * framesInFlight .. + framesInFlight - 1, so its tickets and its recordings meet there). Immutable.
	// A revision's recordings of an epoch, one per slot of its ring. While the caller holds it, the epoch's tickets admit it as a
	// candidate (RenderGraph::PrepareRevisionTicket), so it can be submitted at any frame; released, it is never admitted again.
	struct EpochRecording {
		uint32_t epoch = 0;
		uint32_t firstSlot = 0;
		std::vector<std::shared_ptr<const RenderGraph::PersistentRecording>> slots;  // by ring position
		std::shared_ptr<const IHostExecutionData> revision;  // what it was recorded for
		uint64_t buildGeneration = 0;  // the build it was recorded on (BuildGeneration): no later build's tickets take it
	};
	using EpochRecordingCallback = std::function<void(std::shared_ptr<const EpochRecording>, std::exception_ptr)>;
	// Any thread, lock-free: records the revision-driven epoch for hostData (what its passes prepare for: the revision) on the
	// host's thread, into every slot of its ring once the slot's last work is done (the host's thread waits for that, never the
	// caller; a slot the epoch's admission-only ticket holds is recorded too). done runs on the host's thread once every slot is recorded, or with the error
	// that stopped it. False when async epochs are not running or the epoch is not an async one. A live (not revision-driven) async
	// epoch's is made the same way (a slot its live ticket holds once that ticket is submitted, a slot with work in flight once it
	// completes) and admitted beside its tickets' own preparations, while held: the submitting thread may submit it instead
	// (UseEpochRecording).
	bool RequestEpochRecording(uint32_t epoch, std::shared_ptr<const IHostExecutionData> hostData, EpochRecordingCallback done);
	// Render thread, for an async epoch it did not submit this frame: its ready ticket, prepared and not submitted, goes back to the
	// host's thread (abandoned, its slot freed, a new one prepared). A live epoch's recording waits for the slot its unsubmitted ticket
	// holds, so one requested for an epoch the caller stopped submitting completes only once the caller gives that ticket back. False
	// when no ticket was ready (or the mailbox is full: it stays).
	bool ReleaseEpochTicket(uint32_t epoch);
	// Submits the epoch at the current point of the host's queue stream. Waits for its ticket if the host's
	// thread has not finished it; runs beforeSubmit with the graph's services active and CurrentFrameSlot()
	// the ticket's slot (write latches and queue uploads there); has the ticket prepared again (and waits)
	// if a pass would now prepare differently; records the uploads queued since the last submission; and
	// hands both to the queue - timeline values are assigned here, in submission order. Throws when the
	// graph failed (on either thread).
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
	// Owner thread: moves with every graph build (an extension added or removed, applied at the next submission): a recording
	// (RequestEpochRecording) is of the build it was made on, and no later one's tickets take it.
	uint64_t BuildGeneration() const noexcept { return m_buildGeneration; }
	// Owner thread: whether a recording is of the graph as built now - with explicit builds, what decides before a frame's
	// submissions whether they can take it (until the next BuildIfRequested).
	bool EpochRecordingCurrent(const EpochRecording& recording) const noexcept { return m_graph && recording.buildGeneration == m_buildGeneration; }
	// A render-thread snapshot of complete, current tickets for all required
	// epochs. The async worker never removes ticket cells; only this render
	// thread can consume them. It also retains one control-mailbox credit per
	// unsubmitted epoch; Clear (or destruction) releases unused credits. Retaining
	// a ticket does not reserve a future graph rebuild or make arbitrary
	// beforeSubmit mutations compatible.
	class FrameTicketReservation {
		friend class PersistentGraphHost;
	public:
		FrameTicketReservation() = default;
		~FrameTicketReservation() { Clear(); }
		FrameTicketReservation(const FrameTicketReservation&) = delete;
		FrameTicketReservation& operator=(const FrameTicketReservation&) = delete;
		FrameTicketReservation(FrameTicketReservation&& other) noexcept { *this = std::move(other); }
		FrameTicketReservation& operator=(FrameTicketReservation&& other) noexcept {
			if (this != &other) {
				Clear();
				tickets = std::move(other.tickets);
				generation = std::exchange(other.generation, 0);
				controlCredits = std::move(other.controlCredits);
				heldCredits = std::exchange(other.heldCredits, 0);
				resourceOwner = std::move(other.resourceOwner);
			}
			return *this;
		}
		void Clear() {
			if (controlCredits && heldCredits) controlCredits->fetch_sub(heldCredits, std::memory_order_relaxed);
			controlCredits.reset();
			heldCredits = 0;
			tickets.clear();
			resourceOwner.reset();
			generation = 0;
		}
		bool Empty() const { return tickets.empty(); }
		// Retain immutable resource bindings from ownership selection through every
		// submitted ticket. The owner itself handles cleanup-lane final release.
		void HoldResources(std::shared_ptr<const void> owner) { resourceOwner = std::move(owner); }
	private:
		struct Entry { uint32_t epoch = 0; std::shared_ptr<RenderGraph::PersistentTicket> ticket; };
		std::vector<Entry> tickets;
		uint64_t generation = 0;
		std::shared_ptr<std::atomic<uint32_t>> controlCredits;
		uint32_t heldCredits = 0;
		std::shared_ptr<const void> resourceOwner;
	};
	// Returns false immediately when any ticket is absent/stale, a rebuild is
	// pending, or the async host is unavailable. Leaves destination empty on
	// failure; no graph build, ticket wait, or inline preparation occurs.
	bool TryReserveReadyEpochs(std::span<const uint32_t> epochs, FrameTicketReservation& destination);
	// Consumes one exact ticket from the reservation, with no ticket wait or
	// reprepare. BeforeSubmit may patch latches/uploads but must not invalidate
	// ticket invocation revisions; doing so throws instead of skipping a draw.
	bool TrySubmitReservedEpoch(FrameTicketReservation& reservation, uint32_t epoch, const FrameCallback& beforeSubmit = {});
	// Render-thread producer, nonblocking admission. The callback must own every
	// input it uses; it runs on the graph-owning async thread with upload and
	// descriptor services active, outside an epoch callback. False means the
	// bounded mailbox is full or async epochs are unavailable; the caller retains
	// its request for retry. A callback exception faults the async host.
	bool TryPostOwnedPreparation(FrameCallback& request);
	// A change of graph resources' backings (Buffer::ResizeBytes, ResizeStructured) from the submitting thread. With async
	// epochs the host's thread captures every slot's backing as it prepares a ticket, and capture and backing mutation must
	// not overlap (BackedResource): a resize has no backing between releasing the old one and creating the new one, and a
	// capture then binds nothing, or materializes the buffer itself. The scope waits for a preparation in progress to end,
	// holds the next one off until it closes, and lets this thread change backings (BufferBase::ScopedBackingMutation);
	// every ticket prepared before it is prepared again when submitted. Without async epochs it only opens the mutation
	// scope. Held around the change itself, never across a ticket wait; nests.
	class BackingMutation {
	public:
		BackingMutation(BackingMutation&& other) noexcept;
		BackingMutation& operator=(BackingMutation&&) = delete;
		BackingMutation(const BackingMutation&) = delete;
		BackingMutation& operator=(const BackingMutation&) = delete;
		~BackingMutation();

	private:
		friend class PersistentGraphHost;
		explicit BackingMutation(PersistentGraphHost* host);
		PersistentGraphHost* m_host = nullptr;
	};
	[[nodiscard]] BackingMutation MutateBackings();
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
		uint64_t wakesSkipped = 0;  // submissions whose completion the host's thread took at a GPU wake-up instead of a signal
		uint64_t backingMutations = 0;  // BackingMutation scopes opened while async epochs ran
		uint64_t backingWaits = 0;      // of them, those that found a ticket preparation in progress and waited it out
		uint64_t preparationWaits = 0;  // ticket preparations the host's thread held off for a scope
		uint64_t staleBacking = 0;      // tickets prepared again only because a backing changed since their preparation
		// Desc::reuseRecordings: tickets recorded, recordings kept, tickets that took a kept recording instead of recording,
		// and stale tickets made current by a kept recording (not prepared again).
		uint64_t recorded = 0, kept = 0, reused = 0, adopted = 0;
		// Revision-driven epochs: slots recorded for requests (RequestEpochRecording), requests completed, and submissions of a
		// revision's recording.
		uint64_t revisionSlotsRecorded = 0, revisionRecordings = 0, revisionSubmissions = 0;
	};
	AsyncStats TakeAsyncStats() noexcept;

	RenderGraph* Graph() noexcept { return m_graph.get(); }
	/** @brief The device the host was made with (Desc::device), e.g. for a producer recording its uploads ahead (StagedUploadBatch::Record). */
	rhi::Device Device() const noexcept { return m_desc.device; }
	/** @brief The device-generation cleanup lane used for immutable binding roots. */
	std::shared_ptr<runtime::ResourceCleanupQueue> ResourceCleanup() const;
	runtime::IUploadService* Uploads() noexcept;
	// Owner thread: the graph's descriptor service (its resource and sampler heaps: what a recording binds), null without a graph.
	runtime::IDescriptorService* Descriptors() noexcept;
	// The same, kept alive by the caller: a producer on another thread (growth as graph work, its worker upload path).
	std::shared_ptr<runtime::IUploadService> RetainUploads() noexcept;
	uint64_t FramesExecuted() const noexcept { return m_frameNumber; }
	// The frame slot of the execution ExecuteFrame is running (valid from its beforePrepare callback until
	// it returns): the region of a LatchBlock the host may write for it. Also RecordingContext::FrameSlot.
	bool ClosedExecutions() const noexcept { return m_desc.closedExecutions; }
	uint32_t CurrentFrameSlot() const noexcept {
		return m_async ? m_asyncSlot : static_cast<uint32_t>(m_frameNumber % m_frameSlots);
	}
	// The host frame number of the last execution (ExecuteFrame or SubmitEpoch); the completed-frame
	// callback reports frames by it.
	uint64_t LastHostFrame() const noexcept { return m_lastHostFrame; }
	// The frame slots: Desc::framesInFlight frames of every epoch in the order (one execution a frame without epochs).
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
		double uploadsUs = 0.0;        // recording the queued uploads
		double submitUs = 0.0;         // handing the uploads and the ticket to the queue
		double postUs = 0.0;           // handing the submitted ticket back to the host's thread (its completion)
		RenderGraph::PersistentExecuteTimings execute{};
	};
	const FrameTimings& LastFrameTimings() const noexcept { return m_lastTimings; }
	const Desc& GetDesc() const noexcept { return m_desc; }

private:
	void Build();
	// The graph is to be built before an execution or submission: there is none, or a rebuild was requested without explicit builds.
	bool BuildDue() const noexcept { return !m_graph || (m_rebuildRequested && !m_explicitBuilds); }

	Desc m_desc;
	uint32_t m_frameSlots = 1;  // FrameSlots: Desc::framesInFlight times the epochs in the order
	struct Registered {
		std::string id;
		ExtensionFactory factory;
	};
	std::vector<Registered> m_extensions;
	std::unique_ptr<RenderGraph> m_graph;
	// A slot's last frame: the value each queue's own timeline reached with that frame's last batch.
	// Waiting on those gates the slot's reuse; the host adds no submission of its own. Queue timelines
	// belong to the graph, so a rebuild (which waits for the device) clears them.
	struct SlotCompletion {
		std::vector<std::pair<rhi::Timeline*, uint64_t>> points;
		uint64_t frameNumber = 0;
		bool pending = false;
	};
	std::vector<SlotCompletion> m_slotCompletions;
	bool m_rebuildRequested = true;
	bool m_explicitBuilds = false;
	uint64_t m_frameNumber = 0;
	CompletedFrameCallback m_completedFrame;
	RenderGraph::GpuPassRangeBegin m_gpuPassRangeBegin;
	RenderGraph::GpuPassRangeEnd m_gpuPassRangeEnd;
	FrameTimings m_lastTimings{};
	uint64_t m_lastHostFrame = 0;
	uint64_t m_ticketGeneration = 0;
	uint64_t m_buildGeneration = 0;
	struct Async;
	std::unique_ptr<Async> m_async;
	uint32_t m_backingMutationDepth = 0;  // submitting thread: open BackingMutation scopes
	// A failed/uncertain GPU completion cannot release submitted owners during
	// a graph rebuild. Keep them until device teardown has stopped execution.
	std::vector<std::shared_ptr<const void>> m_uncertainExecutionOwners;
	std::vector<std::shared_ptr<void>> m_uncertainUploadOwners;
	uint32_t m_asyncSlot = 0;
	std::vector<uint32_t> m_asyncEpochs;  // restarted with these after a rebuild
	std::vector<uint32_t> m_revisionEpochs;
	std::shared_ptr<const EpochRecording> m_chosenRecording;  // submitting thread: UseEpochRecording's, until SubmitEpoch takes it
	// Submitting thread, during SubmitTicket's beforeSubmit: the ticket being submitted and its epoch's index (CanUseEpochRecording).
	const RenderGraph::PersistentTicket* m_submittingTicket = nullptr;
	uint32_t m_submittingIndex = 0;
	AsyncStats m_asyncStats{};
	void StartAsync();
	void StopAsync();
	void PrepareTicket(Async& state, uint32_t epochIndex, int32_t requestedSlot);
	void WaitSlot(Async& state, uint32_t slot);
	bool AdvanceRecordingRequests(Async& state);
	// Host thread: admits the recording of the ticket's slot as one of the revision ticket's candidates.
	void AddRevisionCandidate(Async& state, RenderGraph::PersistentTicket& ticket, const EpochRecording& recording);
	void SubmitTicket(Async& state, uint32_t epochIndex, std::shared_ptr<RenderGraph::PersistentTicket> ticket,
		const FrameCallback& beforeSubmit, bool allowReprepare, FrameTicketReservation* reservation = nullptr,
		std::shared_ptr<const void> resourceOwner = {}, std::shared_ptr<const EpochRecording> recording = {});
};

} // namespace org
