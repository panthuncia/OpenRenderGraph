#include "OpenRenderGraph/GraphHost.h"

#include <BasicTelemetry/Tracy.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Resources/Buffers/DynamicBufferBase.h"
#include "Render/FrameArena.h"
#include "Render/LatchBlock.h"
#include "Render/MemoryIntrospectionBackend.h"
#include "Render/PassExecutionContext.h"
#include "Render/RenderGraph/PersistentGraph.h"
#include "Render/RenderGraph/PersistentAliasArena.h"
#include "Render/Runtime/IUploadService.h"
#include "Render/Runtime/OpenRenderGraphSettings.h"
#include "Render/Runtime/RuntimeDevice.h"
#include "Render/Runtime/ScopedActiveGraphServices.h"
#include "Resources/VersionedResource.h"
#include "Resources/VersionedTexture.h"
#include "Interfaces/IResourceResolver.h"
#include "HostUploadExtension.h"
#include "FramePlacement.h"
#include "Resources/PixelBuffer.h"
#include "Resources/Defragmenter.h"
#include "Resources/Relocation.h"
#include "Managers/Singletons/DeviceManager.h"
#include "../Render/RenderGraph/AliasPacking.h"

#include <unordered_map>

// Owner-thread checks (GraphHost::CheckOwner). On by default in builds without NDEBUG. Define ORG_THREAD_CHECKS as 0 or 1 to choose.
#ifndef ORG_THREAD_CHECKS
#ifdef NDEBUG
#define ORG_THREAD_CHECKS 0
#else
#define ORG_THREAD_CHECKS 1
#endif
#endif

namespace org {

namespace {

using detail::HostUploadExtension;

using TimelinePoints = std::vector<std::pair<std::shared_ptr<rhi::TimelinePtr>, uint64_t>>;

bool PointsReached(const TimelinePoints& points) {
	for (const auto& [timeline, value] : points)
		if (timeline && *timeline && (*timeline)->GetCompletedValue() < value) return false;
	return true;
}

// Completed values read once per timeline, for a pass over many leases: each read is a driver call.
struct CompletedValues {
	std::vector<std::pair<const rhi::TimelinePtr*, uint64_t>> values;
	uint64_t Of(const std::shared_ptr<rhi::TimelinePtr>& timeline) {
		for (const auto& [known, value] : values)
			if (known == timeline.get()) return value;
		const uint64_t value = (*timeline)->GetCompletedValue();
		values.emplace_back(timeline.get(), value);
		return value;
	}
	bool Reached(const TimelinePoints& points) {
		for (const auto& [timeline, value] : points) {
			if (!timeline || !*timeline) continue;
			const auto completed = Of(timeline);
			// UINT64_MAX: a lost device, whose pending work will never read anything.
			if (completed != UINT64_MAX && completed < value) return false;
		}
		return true;
	}
};

} // namespace

// One run's lease (RunRef): what the run holds beyond its recording, until its GPU work is done.
struct GraphHost::RunLease {
	enum class State : uint8_t {
		Free,       // retired: on the free list, or a published copy's lease waiting for its next run
		Open,       // opened for a run that is not submitted yet (a ticket prepared, an execution running)
		Submitted,  // its points name the run's completion
	};
	uint32_t index = 0;
	uint32_t generation = 0;  // bumped at every opening, so two runs of the index differ (RunRef::generation)
	State state = State::Free;
	uint32_t copy = RunRef::kNoCopy;  // a published copy's lease: never on the free list, reused by its epoch's copies alone
	bool external = false;            // BeginExternalRun's: its client signals its point, and nothing of it is recorded here
	TimelinePoints points;            // Submitted: every queue's value at the run's last batch
	uint64_t hostFrame = 0;
	uint64_t submission = 0;  // its number (SubmissionsMade), 0 for an external run
	uint64_t seal = 0;        // the seals before it was submitted (Seals)
	std::shared_ptr<void> keepAlive;            // the uploads' staging and the submission's handoff lists
	std::shared_ptr<const void> resourceOwner;  // what the caller pinned for the run (SubmitEpoch's resourceOwner)
	// Its upload lists: begun by the host's thread when it prepares the run's ticket, recorded and submitted by the render
	// thread, recycled when the run retires. An external epoch's secondary, and a list per compute or copy queue
	// (Desc::queueScopedUploads), likewise.
	CommandListPair upload, uploadSecondary;
	std::vector<CommandListPair> queueUploads;  // by queue slot
	RunRef Ref() const noexcept { return copy != RunRef::kNoCopy ? RunRef::Copy(index, copy) : RunRef{index, generation, RunRef::kNoCopy}; }
};

// Defined before the constructor: a throwing constructor destroys m_async, and clang instantiates
// ~unique_ptr<Async> there.
struct GraphHost::Async {
	using Ticket = RenderGraph::PersistentTicket;
	struct Message {
		// Detach: the host's thread lets go of instances (their epochs' tickets abandoned, their recording requests dropped), says
		// so (detached) and holds still until the owner thread has built them (holding). Attach: the instances, built again, are the
		// host's thread's once more, at new build generations.
		enum class Kind : uint8_t { Submitted, Discard, Detach, Attach, Stop } kind = Kind::Stop;
		std::shared_ptr<Ticket> ticket;
		RenderGraph::PersistentTicketSubmission submission;
		uint32_t epochIndex = 0;
		std::vector<std::pair<size_t, uint64_t>> instances;  // Detach, Attach: the instances (and Attach: their new build generations)
		uint64_t hold = 0;  // Detach: the hold it starts, which lasts until released reaches it
		bool reprepare = false;  // Discard: the epoch's replacement ticket is prepared into the same run lease
		std::shared_ptr<void> keepAlive;          // the uploads' staging, until the run completes
		std::shared_ptr<void> retired;            // Submitted: GPU objects that retired at this submission, destroyed on the host's thread
		std::shared_ptr<const void> resourceOwner;
		std::pair<uint32_t, uint64_t> uploadSignal{UINT32_MAX, 0};
		uint64_t number = 0;  // Submitted: its number (SubmissionsMade)
		uint64_t seal = 0;    // Submitted: the seals before it (GraphHost::Seals)
	};

	std::vector<uint32_t> epochs;
	std::vector<size_t> instanceOf;       // per epoch index: its instance (GraphHost::m_instances)
	std::vector<uint8_t> revisionDriven;  // per epoch index: recorded ahead for revisions, its tickets admission-only
	// Host thread, per epoch index: let go of (Detach) until its instance is attached again; and its instance's build generation.
	std::vector<uint8_t> paused;
	std::vector<uint64_t> generation;
	// The host's seal counts: the seals passed (GraphHost::m_seals) and the seals made, queued ones included (m_sealsPushed).
	const std::atomic<uint64_t>* seals = nullptr;
	const std::atomic<uint64_t>* sealsPushed = nullptr;
	// A revision-driven epoch's published copies (Desc::framesInFlight).
	uint32_t copyCount = 1;
	// Per epoch index: the seal count at which its copies were busy (every one an unsealed run's), or kNoSeal. Its next ticket is
	// prepared once the count moves or a copy's run completes; until then WaitTicket throws EpochCopiesBusyError. anyCopiesBusy:
	// one is set, so a seal wakes the host's thread.
	static constexpr uint64_t kNoSeal = ~uint64_t{0};
	std::unique_ptr<std::atomic<uint64_t>[]> copiesBusyAt;
	std::atomic<bool> anyCopiesBusy{false};
	std::vector<uint8_t> copiesBusyLogged;
	std::atomic<uint64_t> copiesBusy{0}, copyWaits{0};
	// Per epoch index: the published copy its next ticket takes (GraphHost::SetEpochCopy), or RunRef::kNoCopy to take any. The
	// submitting thread writes it, the host's thread reads it when it prepares the epoch's ticket.
	std::unique_ptr<std::atomic<uint32_t>[]> pinnedCopy;
	// Submitting thread, per epoch index: the copy its last pinned submission took, or RunRef::kNoCopy.
	std::vector<uint32_t> lastPinned;
	// Recording requests taken from the host's queue (RecordingQueue), by the host's thread, in posting order.
	struct RecordingRequest {
		uint32_t epochIndex = 0;
		uint32_t copy = RunRef::kNoCopy;  // the one copy to record, or every copy
		uint64_t generation = 0;          // the build to record on (0: the attached one)
		std::shared_ptr<const IHostExecutionData> hostData;
		GraphHost::EpochRecordingCallback done;
		std::shared_ptr<GraphHost::EpochRecording> result;
		std::vector<uint8_t> recorded;  // by copy index
	};
	std::deque<std::unique_ptr<RecordingRequest>> pendingRequests;  // host thread; StopAsync drops what is left
	// Host thread, per epoch index: the revision recordings made (admitted by each ticket while their callers hold them), and the
	// ticket prepared and not yet submitted or discarded, which a recording finished meanwhile is added to before its callback.
	std::vector<std::vector<std::weak_ptr<const GraphHost::EpochRecording>>> liveRecordings;
	std::vector<std::shared_ptr<RenderGraph::PersistentTicket>> outstanding;
	std::atomic<uint64_t> revisionCopiesRecorded{0}, revisionRecordings{0};
	// Per epoch index: its staged graph (GraphHost::StageEpochGraph) and the generation it takes at its attach (0: none), and the
	// recordings made on it, which become liveRecordings at the attach. Written by the owner thread while this thread holds still.
	std::vector<RenderGraph*> stagedGraph;
	std::vector<uint64_t> stagedGeneration;
	std::vector<std::vector<std::weak_ptr<const GraphHost::EpochRecording>>> stagedRecordings;
	// Whether a request records on the epoch's staged graph.
	bool RecordsStaged(uint32_t index, uint64_t requestGeneration) const {
		return requestGeneration && stagedGraph[index] && requestGeneration == stagedGeneration[index];
	}
	// Host thread, per live epoch index and kept copy (RunLease::copy): the last replayable recording made for the copy, and the
	// build generation of the graph it was recorded on (live epochs with Desc::reuseRecordings).
	struct LiveCopy {
		std::shared_ptr<const RenderGraph::PersistentRecording> recording;
		uint64_t generation = 0;
		uint64_t version = 0;  // the backing version it was recorded under (NoteNewVersions)
	};
	std::vector<std::vector<LiveCopy>> liveKept;
	std::atomic<uint64_t> liveReused{0}, liveKeptCount{0}, liveUnkeepable{0}, liveSpilled{0};
	// Per epoch index: its prepared tickets, in the order they are submitted, and a stale ticket's replacement, which that
	// submission takes before any other. The host's thread pushes, the render thread takes. The mutex is held only to push or
	// take. count is ready's size, for a reader without the mutex.
	struct TicketQueue {
		std::mutex mutex;
		std::deque<std::shared_ptr<Ticket>> ready;
		std::shared_ptr<Ticket> reprepared;
		std::atomic<uint32_t> count{0};
	};
	std::unique_ptr<TicketQueue[]> queues;
	std::shared_ptr<Ticket> TakeReady(uint32_t index) {
		auto& queue = queues[index];
		std::scoped_lock lock(queue.mutex);
		if (queue.ready.empty()) return nullptr;
		auto ticket = std::move(queue.ready.front());
		queue.ready.pop_front();
		queue.count.store(static_cast<uint32_t>(queue.ready.size()), std::memory_order_release);
		return ticket;
	}
	std::shared_ptr<Ticket> TakeReprepared(uint32_t index) {
		auto& queue = queues[index];
		std::scoped_lock lock(queue.mutex);
		return std::exchange(queue.reprepared, nullptr);
	}
	// front: a ticket the render thread took and gives back (ReleaseEpochTicket), first again.
	void PushReady(uint32_t index, std::shared_ptr<Ticket> ticket, bool front = false) {
		auto& queue = queues[index];
		std::scoped_lock lock(queue.mutex);
		if (front) queue.ready.push_front(std::move(ticket));
		else queue.ready.push_back(std::move(ticket));
		queue.count.store(static_cast<uint32_t>(queue.ready.size()), std::memory_order_release);
	}
	void PushReprepared(uint32_t index, std::shared_ptr<Ticket> ticket) {
		auto& queue = queues[index];
		std::scoped_lock lock(queue.mutex);
		queue.reprepared = std::move(ticket);
	}
	// Everything the epoch has prepared and not handed out: the replacement first, then the queue in order.
	std::vector<std::shared_ptr<Ticket>> TakeAll(uint32_t index) {
		auto& queue = queues[index];
		std::scoped_lock lock(queue.mutex);
		std::vector<std::shared_ptr<Ticket>> taken;
		if (queue.reprepared) taken.push_back(std::exchange(queue.reprepared, nullptr));
		for (auto& ticket : queue.ready) taken.push_back(std::move(ticket));
		queue.ready.clear();
		queue.count.store(0, std::memory_order_release);
		return taken;
	}
	// Host thread, per epoch index:
	// - inHand: tickets prepared and not yet back (submitted or discarded), queued or with the render thread,
	// - mostAhead: the most tickets it may have in hand (GraphHost::SetEpochTicketsAhead),
	// - runsThisFrame and runsLastFrame: its submissions since the seal count lastSeal, and in the frame before that.
	std::vector<uint32_t> inHand, mostAhead, runsThisFrame, runsLastFrame;
	std::vector<uint64_t> lastSeal;
	uint32_t nextAhead = 0;  // host thread: the epoch index whose turn it is to prepare a ticket ahead
	// How many tickets the epoch keeps in hand: as many as it ran last frame (or this frame, when more), at most mostAhead. A
	// revision-driven epoch keeps one.
	uint32_t Depth(uint32_t index) const {
		if (revisionDriven[index]) return 1;
		const auto runs = (std::max)({runsLastFrame[index], runsThisFrame[index], 1u});
		return (std::min)(runs, (std::max)(mostAhead[index], 1u));
	}
	std::atomic<uint64_t> staleAhead{0};
	// Render thread and submission thread -> host thread, single consumer, in order. Two threads post (the submission thread
	// its tickets' completions, the render thread discards and holds), so a post takes postLock around its slot and the write
	// count. The host's thread reads without it.
	static constexpr uint64_t kMailbox = 64;
	std::array<Message, kMailbox> mailbox;
	std::atomic<uint64_t> written{0}, read{0};
	std::mutex postLock;
	rhi::CompletionWait* completionWake = nullptr;  // the host's (m_completionWake): requesters wake the thread through it too
	// Whether the host's thread, if it sleeps, sleeps on GPU completions as well as on completionWake: then a submission's
	// message needs no wake of its own (vkSignalSemaphore, several microseconds on the render thread), because the next GPU
	// completion wakes the thread and it reads the mailbox then. Written by the host's thread before its last look at the
	// mailbox, read by a poster after its write: both sequentially consistent, so one of them sees the other.
	std::atomic<bool> wakesOnGpu{false};
	std::atomic<uint64_t> sleepingAt{0};  // the wake version this thread sleeps on, plus one; 0 while awake
	std::atomic<bool> nudgeLogged{false};
	uint64_t wakesSkipped = 0;  // posters, under postLock
	// Ticket publication and terminal failure both advance this counter. A
	// notify on an unchanged null shared_ptr cannot release atomic::wait(nullptr).
	std::atomic<uint32_t> ticketWake{0};
	// Detach messages handled: the render thread waits for its own to be counted. Then the host's thread holds still, touching
	// nothing, until released reaches the Detach's hold: a graph's build changes services every graph shares (statistics,
	// uploads, descriptors). Holds are numbered, so one released and the next begun at once is still two.
	std::atomic<uint64_t> detached{0};
	std::atomic<uint64_t> released{0};
	uint64_t holds = 0;  // render thread: the last hold begun
	bool holding = false;  // render thread: inside a hold
	std::atomic<bool> failed{false};
	std::string error;  // written before failed is set
	// Per epoch index: its preparation failed (FailAsyncEpoch). Set by the host's thread. Cleared by the owner thread when a build of
	// the epoch's instance succeeds, while the host's thread holds still (RebuildAsyncInstances). The error is written before the
	// flag is set.
	std::unique_ptr<std::atomic<bool>[]> epochFailed;
	std::vector<std::string> epochErrors;
	std::thread thread;
	// Advanced by NoteNewVersions. A ticket prepared under an older value may hold a version that is not the newest: it is not
	// current (Current).
	std::atomic<uint64_t> backingVersion{0};

	// Host thread only.
	// The worker stops on the first completion exception. Preserve the one
	// submitted message it was processing until device teardown.
	std::shared_ptr<void> uncertainUploadOwner;
	std::shared_ptr<const void> uncertainResourceOwner;
	uint64_t hostFrame = 0;
	// The highest submission number a Submitted message brought (the mailbox is in order: every one not read yet is higher).
	uint64_t submissionsSeen = 0;

	// Render thread only.
	std::vector<uint64_t> queueValues;  // SubmitTicket's: the last value assigned per queue slot of the epoch's graph
	// The pools every run lease's upload lists come from (RunLease::upload), requested when the host's thread prepares a run's
	// ticket and recycled when the run retires. Their allocators are shared too (CommandListPool's threading rule). The host's
	// thread begins a run's list and so takes its allocator's claim. The render thread gives the claim up once it has closed
	// the list. Every lease's lists go back to these pools before the state ends (StopAsync).
	std::shared_ptr<CommandListPool> uploadPool;
	// An external epoch's uploads secondary (SetExternalEpochs): begun by the render thread, executed first by the caller's
	// execution (ExternalEpochExecution::leading).
	std::shared_ptr<CommandListPool> uploadSecondaryPool;
	std::array<std::shared_ptr<CommandListPool>, 3> queueUploadPools;  // by rhi::QueueKind (Compute and Copy only)
	// Per queue slot: the kind of the queue that takes queue-scoped uploads, or none. A run has a list for each compute or copy
	// queue of the graphics queue's device.
	std::vector<std::optional<rhi::QueueKind>> queueUploadKinds;
	std::vector<rhi::CommandList> queueLists;     // SubmitTicket's scratch: the run's queue upload lists, by queue slot
	std::vector<rhi::CommandList> uploadLists;    // SubmitTicket's scratch: the run's upload list, which TicketUploads names
	uint32_t graphicsSlot = 0;
	rhi::Queue graphicsQueue;
	rhi::TimelineHandle graphicsFence{};

	uint32_t IndexOf(uint32_t epoch) const {
		for (uint32_t i = 0; i < epochs.size(); ++i) if (epochs[i] == epoch) return i;
		throw std::invalid_argument("SubmitEpoch: epoch " + std::to_string(epoch) + " has no ticket");
	}
	// lazyWake: a submission's completion, which the host's thread can take at its next GPU wake-up (wakesOnGpu).
	bool TryPost(Message& message, bool lazyWake = false) {
		std::scoped_lock lock(postLock);
		const auto index = written.load(std::memory_order_relaxed);
		if (failed.load(std::memory_order_acquire) || index - read.load(std::memory_order_acquire) >= kMailbox) return false;
		mailbox[index % kMailbox] = std::move(message);
		written.store(index + 1, std::memory_order_seq_cst);
		if (lazyWake && wakesOnGpu.load(std::memory_order_seq_cst)) {
			++wakesSkipped;
			return true;
		}
		(void)completionWake->Notify();
		return true;
	}
	bool Post(Message& message, bool lazyWake = false) {
		while (!TryPost(message, lazyWake)) {
			if (failed.load(std::memory_order_acquire)) return false;
			std::this_thread::yield();
		}
		return true;
	}
	// Host thread: a ticket's preparation, which captures every slot's backing and may change backings itself
	// (BufferBase::EnterBackingMutation).
	struct Preparation {
		Preparation() { BufferBase::EnterBackingMutation(); }
		~Preparation() { BufferBase::LeaveBackingMutation(); }
		Preparation(const Preparation&) = delete;
		Preparation& operator=(const Preparation&) = delete;
	};
	// Every pass it prepared would prepare the same invocation, and no version was published since.
	bool Current(const Ticket& ticket) const {
		return RenderGraph::PersistentTicketCurrent(ticket)
			&& RenderGraph::PersistentTicketHostTag(ticket) == backingVersion.load(std::memory_order_acquire);
	}
	void ThrowIfFailed() const {
		if (failed.load(std::memory_order_acquire)) throw std::runtime_error("Async epochs failed: " + error);
	}
	// reprepared: the replacement of a stale ticket (TicketQueue::reprepared), not the next one queued.
	std::shared_ptr<Ticket> WaitTicket(uint32_t index, bool reprepared = false) {
		BT_ZONE_SCOPE("ORG.Host.WaitTicket");
		// About to block on the host's thread: it must not be left asleep on GPU work that waits for this thread.
		(void)completionWake->Notify();
		for (;;) {
			const auto seen = ticketWake.load(std::memory_order_acquire);
			if (auto ticket = reprepared ? TakeReprepared(index) : TakeReady(index)) return ticket;
			ThrowIfFailed();
			// Only this epoch failed: no ticket is coming until its graph is built again.
			if (epochFailed[index].load(std::memory_order_acquire)) throw EpochFailedError(epochs[index], epochErrors[index]);
			// Its copies are all busy with unsealed runs: no ticket is coming before the next seal, which this thread makes. Once the
			// count has moved past the one they were busy at, the host's thread prepares one (the seal woke it). A seal this thread
			// queued already is passed by the submission thread without this thread, so then the wait ends.
			if (const auto busy = copiesBusyAt[index].load(std::memory_order_acquire);
				busy != kNoSeal && busy == seals->load(std::memory_order_acquire) && busy == sealsPushed->load(std::memory_order_acquire))
				throw EpochCopiesBusyError(epochs[index], copyCount);
			ticketWake.wait(seen, std::memory_order_acquire);
		}
	}
};

// What CommitTicket hands to FinishTicket: the ticket, checked current after the caller's commit, and the values of the
// render thread's state that the rest of the submission needs.
struct GraphHost::TicketCommit {
	uint32_t index = 0;  // the epoch's index in the async state
	std::shared_ptr<RenderGraph::PersistentTicket> ticket;
	RunRef run{};
	bool external = false;  // an external epoch's ticket (RenderGraph::IsExternalPersistentEpoch)
	// What retired at the commit (the runtime's deletion queues, external runs' releases), destroyed on the host's thread once
	// the submission completes.
	std::shared_ptr<void> retired;
	std::shared_ptr<const void> resourceOwner;
	uint64_t number = 0;     // the submission's number (SubmissionsMade)
	uint64_t seal = 0;       // the seals before it (Seals)
	uint64_t frameWait = 0;  // what its batches wait for (SetFrameWaitValue), as it was at the commit
	// The uploads the commit queued, taken from the upload service at its end (RenderGraph::TakePendingUploads). Uploads the render
	// thread queues later belong to later tickets, so a ticket finished on the submission thread records only its own.
	RenderGraph::TakenUploads uploads;
	// Open from the commit until FinishTicket publishes the ticket's queue values. Meanwhile the frontier misses the ticket's work,
	// so deferred deletions, descriptor releases and freed pinned regions made on the render thread wait for a settled frontier
	// (Runtime::AssignmentScope).
	std::unique_ptr<Runtime::AssignmentScope> assigned;
};

// FinishTicket's completion messages, held until the group's last submission is on its queue (FinishTicket's posts).
struct GraphHost::DeferredPosts {
	std::vector<Async::Message> messages;
};

// The submission thread's queue (Desc::submissionThread). The render thread pushes an entry after each commit. The thread
// finishes the entries in order.
struct GraphHost::SubmitQueue {
	struct Entry {
		enum class Kind : uint8_t { Ticket, Closure, Marker, Stop } kind = Kind::Stop;
		Async* async = nullptr;  // a ticket's async state
		std::optional<TicketCommit> commit;
		std::function<void()> closure;
		std::shared_ptr<GpuPoint::Pending> marker;  // SubmittedPoint's pending part, filled in when the thread gets here
	};
	// The most entries queued at once. A full queue makes the render thread wait for room (backpressure), so the render thread
	// never runs further ahead of the queue than this.
	static constexpr size_t kCapacity = 64;
	// The most tickets the thread finishes as one group (RunSubmitThread).
	static constexpr size_t kMostGrouped = 16;
	std::mutex mutex;
	std::condition_variable work;      // the thread waits here for entries
	std::condition_variable progress;  // a full push waits here for room, a drain for the thread to pass its entries
	std::deque<Entry> entries;
	uint64_t pushed = 0;  // entries pushed so far
	uint64_t passed = 0;  // entries the thread has finished
	std::atomic<bool> failed{false};
	std::string error;  // the first failure, written under the mutex before failed is set
	std::thread thread;
	std::thread::id threadId;
	// HoldSubmissions: the holding thread's entries, not yet pushed. Under the mutex.
	std::thread::id holder;
	uint32_t holdDepth = 0;
	std::vector<Entry> held;
	// Statistics, under the mutex.
	GraphHost::SubmitThreadStats stats;
	std::map<std::string, std::pair<uint64_t, double>> drains;  // by caller
};

// The host's recording requests (RequestEpochRecording): any thread pushes onto a Treiber stack, the host's thread takes it whole
// and reverses it into posting order. It and the wake live as long as the host, so a request never meets a build's teardown.
struct GraphHost::RecordingQueue {
	struct Request {
		uint32_t epoch = 0;
		std::shared_ptr<const IHostExecutionData> hostData;
		GraphHost::EpochRecordingCallback done;
		uint32_t copy = RunRef::kNoCopy;  // the one copy to record, or every copy
		uint64_t generation = 0;          // the build to record on (0: the attached one)
		Request* next = nullptr;
	};
	std::atomic<Request*> head{nullptr};
	std::atomic<rhi::CompletionWait*> wake{nullptr};  // the host's thread's, once async epochs first started
	std::atomic<bool> accepting{false};
	// The host's end: what is left goes without its callbacks.
	~RecordingQueue() {
		for (auto* request = head.exchange(nullptr); request;) {
			auto* next = request->next;
			delete request;
			request = next;
		}
	}
	// Takes every request pushed so far, in posting order.
	std::vector<std::unique_ptr<Request>> Take() {
		std::vector<std::unique_ptr<Request>> taken;
		for (auto* request = head.exchange(nullptr, std::memory_order_acquire); request;) {
			auto* next = request->next;
			request->next = nullptr;
			taken.emplace_back(request);
			request = next;
		}
		std::reverse(taken.begin(), taken.end());
		return taken;
	}
};

GraphHost::GraphHost(Desc desc)
	: m_desc(std::move(desc)), m_recordingQueue(std::make_unique<RecordingQueue>()) {
	if (!m_desc.runtime) throw std::invalid_argument("GraphHost requires a runtime");
	m_device = m_desc.runtime->Device();
	m_leaseTable = std::make_unique<std::atomic<RunLease*>[]>(Runtime::kMaxSlots);
	// A graph per epoch, in the frame's order; one of every pass without epochs.
	if (m_desc.epochOrder.empty()) m_instances.emplace_back();
	for (const auto epoch : m_desc.epochOrder) {
		if (epoch == persistent::AllEpochs) throw std::invalid_argument("GraphHost: an epoch order names AllEpochs");
		if (std::ranges::any_of(m_instances, [&](const Instance& other) { return other.epoch == epoch; }))
			throw std::invalid_argument("GraphHost: an epoch order names an epoch twice");
		m_instances.emplace_back().epoch = epoch;
	}
	m_declared.resize(m_instances.size());
	m_builds = std::make_unique<BuildsCell[]>(m_instances.size());
	m_framePlacement = std::make_shared<FramePlacement>(m_desc.epochOrder);
	// Made with the host, so the queue's pointer never changes while another thread drains it.
	if (m_desc.submissionThread) {
		if (m_desc.mergeTicketSubmissions) m_submitCarry = RenderGraph::MakeSubmitCarry();
		m_submitQueue = std::make_unique<SubmitQueue>();
		m_submitQueue->thread = std::thread([this] { RunSubmitThread(); });
	}
}

GraphHost::~GraphHost() {
	// Nothing queued is completed from here on: its callers may be torn down before the host.
	m_ending = true;
	m_recordingQueue->accepting.store(false, std::memory_order_release);
	// What the submission thread still holds is finished first: it submits on the async state that StopAsync ends.
	StopSubmitThread();
	StopAsync();
	DestroyGraph();
	// Its worker stops here. Graphs still alive hold it weakly and publish nothing more.
	m_framePlacement.reset();
	// The host's queue timelines go last: what was submitted on them completes first, and the runtime's ledgers stop waiting on them.
	for (const auto& timeline : m_timelines) {
		const auto next = timeline.nextValue->load(std::memory_order_acquire);
		if (next > 1) (void)(*timeline.fence)->HostWait(next - 1);
		m_desc.runtime->UnregisterAssignedTimeline(timeline.fence);
		m_desc.runtime->Sequencer()->RetireTimeline(m_desc.runtime->TimelineIdentity(timeline.fence->Get()));
		m_desc.runtime->NoteTimelineDestroyed(timeline.fence->Get());
	}
	m_timelines.clear();
	m_desc.runtime->Retire();
	m_uncertainExecutionOwners.clear();
	m_uncertainUploadOwners.clear();
}

// ---- run leases ----

GraphHost::RunLease* GraphHost::OpenLeaseLocked(bool external) {
	RunLease* lease = nullptr;
	if (!m_freeLeases.empty()) {
		lease = m_leaseStore[m_freeLeases.back()].get();
		m_freeLeases.pop_back();
	} else {
		if (m_leaseStore.size() >= Runtime::kMaxSlots) return nullptr;
		auto made = std::make_unique<RunLease>();
		made->index = static_cast<uint32_t>(m_leaseStore.size());
		lease = made.get();
		m_leaseTable[made->index].store(lease, std::memory_order_release);
		m_leaseStore.push_back(std::move(made));
	}
	// Generation 0 means "not a leased run" (RunRef::Leased).
	if (++lease->generation == 0) lease->generation = 1;
	lease->state = RunLease::State::Open;
	lease->external = external;
	lease->points.clear();
	lease->hostFrame = lease->submission = lease->seal = 0;
	m_leasesPeak = (std::max)(m_leasesPeak, ++m_leasesInFlight);
	return lease;
}

GraphHost::RunLease& GraphHost::OpenLease(Async* state) {
	for (;;) {
		RunLease* lease = nullptr;
		RunLease* oldest = nullptr;
		{
			std::scoped_lock lock(m_leaseMutex);
			lease = OpenLeaseLocked(false);
			// Every index is in flight: the oldest sealed run is waited for (backpressure, never a run that may still be in the
			// host's stream). Never reached in practice: Runtime::kMaxSlots runs in flight.
			if (!lease) {
				const uint64_t seals = m_seals.load(std::memory_order_acquire);
				for (auto* submitted : m_submittedLeases)
					if (!submitted->external && (submitted->seal < seals || seals == 0)) {
						oldest = submitted;
						break;
					}
				if (!oldest)
					throw std::runtime_error("GraphHost: " + std::to_string(Runtime::kMaxSlots) + " runs in flight and none of them sealed (SealSubmitted)");
			}
		}
		if (lease) {
			// Its statistics range: the pools grow for the most runs in flight.
			if (const auto& stats = m_desc.runtime->Statistics()) stats->EnsureFrameCapacity(lease->index + 1);
			return *lease;
		}
		BT_ZONE_SCOPE("ORG.Host.WaitOldestRun");
		for (const auto& [timeline, value] : oldest->points) (void)(*timeline)->HostWait(value);
		RetireLease(*oldest, state);
	}
}

std::vector<GraphHost::RunLease*>& GraphHost::CopyLeases(uint32_t epoch) {
	for (auto& [owner, copies] : m_copyLeases)
		if (owner == epoch) return copies;
	const uint32_t count = (std::max)(m_desc.framesInFlight, 1u);
	std::vector<RunLease*> copies;
	{
		std::scoped_lock lock(m_leaseMutex);
		for (uint32_t copy = 0; copy < count; ++copy) {
			if (m_leaseStore.size() >= Runtime::kMaxSlots) throw std::runtime_error("GraphHost: no lease index left for a published copy");
			auto made = std::make_unique<RunLease>();
			made->index = static_cast<uint32_t>(m_leaseStore.size());
			made->copy = copy;
			copies.push_back(made.get());
			m_leaseTable[made->index].store(made.get(), std::memory_order_release);
			m_leaseStore.push_back(std::move(made));
		}
	}
	if (const auto& stats = m_desc.runtime->Statistics()) stats->EnsureFrameCapacity(copies.back()->index + 1);
	// A staging lease per copy: an index no run takes, for the values a host gathers before it takes the copy (EpochStagingRun).
	std::vector<RunLease*> staging;
	{
		std::scoped_lock lock(m_leaseMutex);
		for (uint32_t copy = 0; copy < count; ++copy) {
			if (m_leaseStore.size() >= Runtime::kMaxSlots) throw std::runtime_error("GraphHost: no lease index left for a staging run");
			auto made = std::make_unique<RunLease>();
			made->index = static_cast<uint32_t>(m_leaseStore.size());
			LatchBlock::MarkHostOnlyRun(made->index);
			staging.push_back(made.get());
			m_leaseTable[made->index].store(made.get(), std::memory_order_release);
			m_leaseStore.push_back(std::move(made));
		}
	}
	m_stagingLeases.emplace_back(epoch, std::move(staging));
	return m_copyLeases.emplace_back(epoch, std::move(copies)).second;
}

void GraphHost::RetireLease(RunLease& lease, Async* state) {
	// The run's queries were resolved by its submission: read them back now. The runtime's statistics, which every graph shares
	// (a graph's own pointer may be in the middle of a rebuild).
	if (!lease.external && lease.submission)
		if (const auto& stats = m_desc.runtime->Statistics()) {
			// Every queue's: a run's async compute passes resolve into the compute queue's range of the run's index. With several
			// epochs a frame (several runs), only the run's retirement knows its index. The callback sees each queue's read.
			for (const auto kind : { rhi::QueueKind::Compute, rhi::QueueKind::Graphics }) {
				rhi::Queue queue = m_device.GetQueue(kind);
				if (!queue) continue;
				stats->OnFrameComplete(lease.index, queue);
				if (m_completedFrame) m_completedFrame(lease.hostFrame, *stats);
			}
		}
	ReleaseLease(lease, state);
}

void GraphHost::ReleaseLease(RunLease& lease, Async* state) {
	// Its regions in the frame arena (a published copy's are pinned, and stay).
	if (const auto& arena = m_desc.runtime->Arena()) arena->ReleaseRun(lease.index, lease.generation);
	// Its lists: the GPU is done with them, or they were never submitted. Each is reset when it is begun again.
	const auto recycle = [](const std::shared_ptr<CommandListPool>& pool, CommandListPair& pair) {
		// Still claimed: begun for a ticket that never reached the record step that closes it (discarded, failed, or prepared
		// and replaced). Close it here, so its next Begin resets a closed list (D3D12 cannot reset one that records). It holds
		// no commands: uploads are recorded only at submission.
		// Not a secondary: a recycled one does not record until BeginSecondary, so a claimed one may be closed already.
		if (pair.Recording() && pair.list && !(pool && pool->Secondary())) pair.list->End();
		pair.EndRecording();
		if (!pair.list) return;
		if (pool) pool->RecycleForNextRequest(std::move(pair));
		else pair = {};
	};
	recycle(state ? state->uploadPool : nullptr, lease.upload);
	recycle(state ? state->uploadSecondaryPool : nullptr, lease.uploadSecondary);
	for (size_t queue = 0; queue < lease.queueUploads.size(); ++queue) {
		const auto kind = state && queue < state->queueUploadKinds.size() ? state->queueUploadKinds[queue] : std::nullopt;
		recycle(kind ? state->queueUploadPools[static_cast<size_t>(*kind)] : nullptr, lease.queueUploads[queue]);
	}
	lease.keepAlive.reset();
	lease.resourceOwner.reset();
	lease.points.clear();
	std::scoped_lock lock(m_leaseMutex);
	if (lease.state == RunLease::State::Submitted) std::erase(m_submittedLeases, &lease);
	if (lease.state != RunLease::State::Free) --m_leasesInFlight;
	if (lease.state != RunLease::State::Free && lease.copy == RunRef::kNoCopy) m_freeLeases.push_back(lease.index);
	lease.state = RunLease::State::Free;
	lease.external = false;
}

uint64_t GraphHost::RetiredSubmissionsLocked(uint64_t seen) const {
	uint64_t retired = seen;
	for (const auto* lease : m_submittedLeases)
		if (lease->submission) retired = (std::min)(retired, lease->submission - 1);
	return retired;
}

size_t GraphHost::RetireReachedLeases(Async* state) {
	std::vector<RunLease*> reached;
	{
		std::scoped_lock lock(m_leaseMutex);
		if (m_submittedLeases.empty()) return 0;
		CompletedValues completed;
		for (auto* lease : m_submittedLeases)
			if (completed.Reached(lease->points)) reached.push_back(lease);
	}
	for (auto* lease : reached) RetireLease(*lease, state);
	// Retired: every submission below the oldest one still running (all that were seen, with none). Runs complete in any order,
	// so a run's own number says nothing of the ones before it.
	uint64_t retired = 0;
	{
		std::scoped_lock lock(m_leaseMutex);
		retired = RetiredSubmissionsLocked(state ? state->submissionsSeen : m_submissions);
	}
	if (retired > m_retiredReported) {
		m_retiredReported = retired;
		if (m_retired) m_retired(retired);
	}
	return reached.size();
}

RunRef GraphHost::BeginExternalRun(std::shared_ptr<rhi::TimelinePtr> timeline, uint64_t value) {
	if (!timeline || !*timeline || !value) throw std::invalid_argument("BeginExternalRun needs the client's timeline and the value it signals");
	RunLease* lease = nullptr;
	{
		std::scoped_lock lock(m_leaseMutex);
		lease = OpenLeaseLocked(true);
		if (!lease) throw std::runtime_error("GraphHost: no lease index left for an external run");
		// Submitted at once: the client's point is its completion, whenever the client signals it.
		lease->points.emplace_back(timeline, value);
		lease->state = RunLease::State::Submitted;
		m_submittedLeases.push_back(lease);
	}
	m_externalRun = lease->Ref();
	// The client's own work may use ORG objects too (draws in its stream that read a graph's buffers). No ORG queue sees that
	// work, so deferred deletions and descriptor releases wait for its value.
	DescriptorHeapManager::GetInstance().PublishQueueFenceSnapshot({ DescriptorHeapManager::QueueFenceSnapshotPoint{ .timeline = timeline->Get(), .value = value } });
	m_desc.runtime->NoteExternalPoint(timeline->Get(), value);
	// The run's upload pages: what was staged since the last submission is the client's run's now. Its index's last run retired.
	auto* graph = AnyGraph();
	auto* uploads = graph ? graph->GetUploadService() : nullptr;
	if (!uploads) return *m_externalRun;
	uploads->ProcessDeferredReleases(lease->index);
	if (!m_async) {
		m_desc.runtime->Retire();
		return *m_externalRun;
	}
	if (auto retired = m_desc.runtime->RetireElsewhere()) m_externalRetired.push_back(std::move(retired));
	// No submission came to take them for a while: they go to the runtime's deleter rather than being kept.
	if (m_externalRetired.size() > 16)
		for (auto& retired : std::exchange(m_externalRetired, {})) m_desc.runtime->DestroyDetached(std::move(retired), "an external run's deferred deletions");
	return *m_externalRun;
}

GraphHost::RunMemoryStats GraphHost::TakeRunMemoryStats() {
	RunMemoryStats stats;
	{
		std::scoped_lock lock(m_leaseMutex);
		stats.leasesInFlight = m_leasesInFlight;
		stats.leasesPeak = std::exchange(m_leasesPeak, m_leasesInFlight);
		stats.leaseIndices = static_cast<uint32_t>(m_leaseStore.size());
		for (const auto* lease : m_submittedLeases) stats.externalRuns += lease->external ? 1u : 0u;
		for (const auto& [epoch, copies] : m_copyLeases) stats.copyLeases += static_cast<uint32_t>(copies.size());
	}
	if (const auto& statistics = m_desc.runtime->Statistics()) stats.statisticsRanges = statistics->FrameCapacity();
	if (const auto& arena = m_desc.runtime->Arena()) stats.arena = arena->TakeStats();
	return stats;
}

size_t GraphHost::InstanceOf(uint32_t epoch) const noexcept {
	for (size_t index = 0; index < m_instances.size(); ++index)
		if (m_instances[index].epoch == epoch) return index;
	return kNoInstance;
}

RenderGraph* GraphHost::AnyGraph() const noexcept {
	for (const auto& instance : m_instances)
		if (instance.graph) return instance.graph.get();
	return nullptr;
}

RenderGraph* GraphHost::Graph(uint32_t epoch) noexcept {
	const auto index = InstanceOf(epoch);
	return index == kNoInstance ? nullptr : m_instances[index].graph.get();
}

uint64_t GraphHost::BuildGeneration(uint32_t epoch) const noexcept {
	CheckOwner("BuildGeneration");
	const auto index = InstanceOf(epoch);
	return index == kNoInstance ? 0 : m_instances[index].buildGeneration;
}

bool GraphHost::EpochRecordingCurrent(const EpochRecording& recording) const noexcept {
	// The attached build is 0 while the epoch has no graph, and a recording's build is never 0.
	return recording.buildGeneration != 0 && BuildsOf(recording.epoch).attached == recording.buildGeneration;
}

bool GraphHost::EpochRecordingBackingsCurrent(const EpochRecording& recording, uint32_t copy) const noexcept {
	return copy < recording.copies.size() && recording.copies[copy] && RenderGraph::PersistentRecordingBackingsCurrent(*recording.copies[copy]);
}

GraphHost::Builds GraphHost::BuildsOf(uint32_t epoch) const noexcept {
	const auto index = InstanceOf(epoch);
	if (index == kNoInstance || !m_builds) return {};
	const auto& cell = m_builds[index];
	for (;;) {
		const auto before = cell.sequence.load(std::memory_order_acquire);
		Builds builds{cell.attached.load(std::memory_order_relaxed), cell.staged.load(std::memory_order_relaxed)};
		std::atomic_thread_fence(std::memory_order_acquire);
		if ((before & 1) == 0 && cell.sequence.load(std::memory_order_relaxed) == before) return builds;
		// The owner is inside a publish: a few stores, never a wait of its own.
		std::this_thread::yield();
	}
}

void GraphHost::PublishBuilds(size_t index) noexcept {
	if (!m_builds || index >= m_instances.size()) return;
	const auto& instance = m_instances[index];
	const uint64_t attached = instance.graph ? instance.buildGeneration : 0;
	const uint64_t staged = instance.staged && instance.staged->m_graph ? instance.stagedGeneration : 0;
	auto& cell = m_builds[index];
	if (cell.attached.load(std::memory_order_relaxed) == attached && cell.staged.load(std::memory_order_relaxed) == staged) return;
	// One writer: the count goes odd, the pair is written, and the count goes even again.
	const auto sequence = cell.sequence.load(std::memory_order_relaxed);
	cell.sequence.store(sequence + 1, std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_release);
	cell.attached.store(attached, std::memory_order_relaxed);
	cell.staged.store(staged, std::memory_order_relaxed);
	cell.sequence.store(sequence + 2, std::memory_order_release);
}

void GraphHost::SetOwnerThread() noexcept {
	m_ownerThread.store(std::this_thread::get_id(), std::memory_order_release);
}

bool GraphHost::IsOwnerThread() const noexcept {
	return m_ownerThread.load(std::memory_order_acquire) == std::this_thread::get_id();
}

void GraphHost::CheckOwner(const char* api) const noexcept {
#if ORG_THREAD_CHECKS
	const auto self = std::this_thread::get_id();
	auto owner = m_ownerThread.load(std::memory_order_acquire);
	if (owner == self) return;
	// An unowned host is claimed by its first checked call.
	if (owner == std::thread::id{} && m_ownerThread.compare_exchange_strong(owner, self, std::memory_order_acq_rel, std::memory_order_acquire)) return;
	if (owner == self) return;
	const auto id = [](std::thread::id a_thread) { return static_cast<unsigned long long>(std::hash<std::thread::id>{}(a_thread)); };
	spdlog::critical("GraphHost: {} is an owner-thread call. It was made on thread {:#x}, and thread {:#x} owns the host (SetOwnerThread)", api,
		id(self), id(owner));
	if (auto logger = spdlog::default_logger()) logger->flush();
	std::abort();
#else
	(void)api;
#endif
}

void GraphHost::MarkRebuild(Instance& instance, std::string_view reason) noexcept {
	instance.rebuildRequested = true;
	instance.requests->fetch_add(1, std::memory_order_acq_rel);
	// The reasons are for the log line alone: a failure to record one never stops the rebuild.
	try {
		if (reason.empty()) reason = "unnamed request";
		const auto known = std::ranges::find_if(instance.rebuildReasons, [&](const auto& entry) { return entry.first == reason; });
		if (known != instance.rebuildReasons.end()) {
			++known->second;
			return;
		}
		// Bounded: a storm of distinct reasons ends in one "others" entry.
		static constexpr size_t kMaxReasons = 12;
		if (instance.rebuildReasons.size() < kMaxReasons) {
			instance.rebuildReasons.emplace_back(std::string(reason), 1u);
			return;
		}
		if (instance.rebuildReasons.back().first != "others") instance.rebuildReasons.emplace_back("others", 0u);
		++instance.rebuildReasons.back().second;
	} catch (...) {
	}
}

void GraphHost::RequestRebuild(std::string_view reason) noexcept {
	for (auto& instance : m_instances) MarkRebuild(instance, reason);
}

void GraphHost::RequestRebuild(uint32_t epoch, std::string_view reason) noexcept {
	if (const auto index = InstanceOf(epoch); index != kNoInstance) MarkRebuild(m_instances[index], reason);
}

size_t GraphHost::RequestRebuildFor(const ResourceIdentifier& id, std::string_view reason) noexcept {
	size_t marked = 0;
	for (auto& instance : m_instances) {
		// A graph's lookups are locked (RenderGraph::LookedUp): the host's thread may be preparing it.
		if (!instance.graph || !instance.graph->LookedUp(id)) continue;
		MarkRebuild(instance, reason);
		++marked;
	}
	return marked;
}

bool GraphHost::ExtensionInEpoch(const Registered& extension, uint32_t epoch) noexcept {
	if (extension.epochs.empty() || epoch == persistent::AllEpochs) return true;
	return std::ranges::any_of(extension.epochs, [&](uint32_t listed) { return listed == epoch || listed == persistent::AllEpochs; });
}

void GraphHost::RequestRebuildOf(const Registered& extension, std::string_view reason) noexcept {
	for (auto& instance : m_instances)
		if (ExtensionInEpoch(extension, instance.epoch)) MarkRebuild(instance, reason);
}

void GraphHost::NoteConfigurationOf(const Registered& extension) noexcept {
	for (auto& instance : m_instances)
		if (ExtensionInEpoch(extension, instance.epoch)) instance.configuration->fetch_add(1, std::memory_order_acq_rel);
}

bool GraphHost::RebuildRequested() const noexcept {
	return std::ranges::any_of(m_instances, [](const Instance& instance) { return instance.rebuildRequested; });
}

bool GraphHost::RebuildRequested(uint32_t epoch) const noexcept {
	CheckOwner("RebuildRequested");
	const auto index = InstanceOf(epoch);
	return index != kNoInstance && m_instances[index].rebuildRequested;
}

uint64_t GraphHost::RebuildRequests(uint32_t epoch) const noexcept {
	CheckOwner("RebuildRequests");
	const auto index = InstanceOf(epoch);
	return index == kNoInstance ? 0 : m_instances[index].requests->load(std::memory_order_acquire);
}

std::string GraphHost::RebuildReasons(uint32_t epoch) const {
	CheckOwner("RebuildReasons");
	const auto index = InstanceOf(epoch);
	std::string why;
	if (index == kNoInstance) return why;
	for (const auto& [reason, count] : m_instances[index].rebuildReasons)
		why += (why.empty() ? "" : "; ") + reason + (count > 1 ? " (x" + std::to_string(count) + ")" : std::string());
	return why;
}

bool GraphHost::InstanceFailed(size_t index) const noexcept {
	if (m_instances[index].failed) return true;
	if (!m_async) return false;
	// The host's thread may know first: its preparation failed and the owner has not polled yet.
	for (uint32_t i = 0; i < m_async->epochs.size(); ++i)
		if (m_async->instanceOf[i] == index && m_async->epochFailed[i].load(std::memory_order_acquire)) return true;
	return false;
}

bool GraphHost::EpochFailed(uint32_t epoch) const noexcept {
	CheckOwner("EpochFailed");
	const auto index = InstanceOf(epoch);
	return index != kNoInstance && InstanceFailed(index);
}

void GraphHost::ReportEpochFailure(EpochFailure failure) {
	std::scoped_lock lock(m_failureMutex);
	m_failures.push_back(std::move(failure));
}

size_t GraphHost::PollEpochFailures() {
	CheckOwner("PollEpochFailures");
	std::vector<EpochFailure> failures;
	{
		std::scoped_lock lock(m_failureMutex);
		failures.swap(m_failures);
	}
	size_t delivered = 0;
	for (auto& failure : failures) {
		const auto index = InstanceOf(failure.epoch);
		if (index == kNoInstance) continue;
		auto& instance = m_instances[index];
		// A build since replaced the graph that failed: the failure is no longer the epoch's.
		if (failure.buildGeneration != instance.buildGeneration) continue;
		if (!failure.keptGraph) {
			instance.failed = true;
			instance.failure = failure.error;
		}
		++delivered;
		if (m_epochFailed) m_epochFailed(failure);
	}
	return delivered;
}

bool GraphHost::AnyBuildDue() const noexcept {
	for (size_t index = 0; index < m_instances.size(); ++index)
		if (BuildDue(index)) return true;
	return false;
}

void GraphHost::AddExtension(std::string id, ExtensionFactory factory, std::vector<uint32_t> epochs) {
	// Added again: the graphs it was made into, and the ones it is now, are built again with it; the others keep theirs.
	const auto existing = std::ranges::find(m_extensions, id, &Registered::id);
	const bool again = existing != m_extensions.end();
	const std::string reason = "extension '" + id + (again ? "' added again" : "' added");
	std::scoped_lock lock(m_configMutex);
	if (again) {
		RequestRebuildOf(*existing, reason);
		NoteConfigurationOf(*existing);
		m_extensions.erase(existing);
	}
	m_extensions.push_back({std::move(id), std::move(factory), std::move(epochs)});
	RequestRebuildOf(m_extensions.back(), reason);
	NoteConfigurationOf(m_extensions.back());
}

void GraphHost::RemoveExtension(const std::string& id) {
	const auto existing = std::ranges::find(m_extensions, id, &Registered::id);
	if (existing == m_extensions.end()) return;
	std::scoped_lock lock(m_configMutex);
	RequestRebuildOf(*existing, "extension '" + id + "' removed");
	NoteConfigurationOf(*existing);
	m_extensions.erase(existing);
}

runtime::IUploadService* GraphHost::Uploads() noexcept {
	auto* graph = AnyGraph();
	return graph ? graph->GetUploadService() : nullptr;
}

runtime::IDescriptorService* GraphHost::Descriptors() noexcept {
	CheckOwner("Descriptors");
	auto* graph = AnyGraph();
	return graph ? graph->GetDescriptorService() : nullptr;
}

std::shared_ptr<runtime::IUploadService> GraphHost::RetainUploads() noexcept {
	auto* graph = AnyGraph();
	return graph ? graph->RetainUploadService() : nullptr;
}

void GraphHost::DestroyGraph() {
	StopAsync();
	// A graph built before this (BuildEpochGraph) is not attached after it.
	m_hostConfiguration.fetch_add(1, std::memory_order_acq_rel);
	if (!AnyGraph() && m_retiredGraphs.empty()) return;
	for (auto& instance : m_instances)
		if (instance.graph) instance.graph->StopFrameProduction();
	// Adopted devices wait on their own timelines only (never the whole device).
	(void)m_device.WaitIdle();
	// Every run of the host's own is done: their leases retire. An external run waits for its client's point.
	{
		std::vector<RunLease*> runs;
		{
			std::scoped_lock lock(m_leaseMutex);
			for (auto* lease : m_submittedLeases)
				if (!lease->external) runs.push_back(lease);
		}
		for (auto* lease : runs) {
			for (const auto& [timeline, value] : lease->points) (void)(*timeline)->HostWait(value);
			RetireLease(*lease, nullptr);
		}
	}
	m_syncLeases.clear();
	m_externalRetired.clear();
	for (auto& instance : m_instances) {
		// The next build starts afresh: a failure is the old graphs'.
		instance.failed = false;
		instance.failure.clear();
		if (!instance.graph) continue;
		instance.graph->ShutdownExtensions();
		instance.graph->ShutdownTaskWorkers();
		instance.graph.reset();
		instance.rebuildRequested = true;
	}
	for (size_t index = 0; index < m_instances.size(); ++index) PublishDeclared(index);
	ReleaseRetiredGraphs(true);
}

void GraphHost::ReleaseRetiredGraphs(bool all) {
	if (m_retiredGraphs.empty()) return;
	if (!all && std::ranges::none_of(m_retiredGraphs, [](const RetiredGraph& retired) { return retired.done.Reached(); })) return;
	const auto release = [&] {
		std::erase_if(m_retiredGraphs, [&](RetiredGraph& retired) {
			if (!all && !retired.done.Reached()) return false;
			if (all) (void)retired.done.Wait();
			BT_ZONE_SCOPE("ORG.Host.ReleaseRetiredGraph");
			retired.graph->ShutdownExtensions();
			retired.graph->ShutdownTaskWorkers();
			// Everything else the graph holds is its own, or shared through services that other threads use too (the runtime's
			// registries and ledgers, deferred deletion and descriptor release, the upload policy and backing resize registries,
			// the ECS entity hooks, which defer to their thread). So the rest of its destruction (about 10 ms for a large graph:
			// its external client's and queues' timelines, its passes and resources, its compiled state) can run on the runtime's
			// deleter thread. Not with async epochs, whose host thread holds still for it (HoldAsync), nor at a release of all.
			const bool detached = m_desc.destroyRetiredGraphsDetached && !all && !m_async && m_desc.runtime;
			if (detached) {
				// What other graphs place into goes back here, on the owner thread: they may place it again at once.
				retired.graph->ReleaseAliasPlacements();
				m_desc.runtime->DestroyDetached(std::shared_ptr<RenderGraph>(std::move(retired.graph)), "a retired epoch graph");
			} else retired.graph.reset();
			return true;
		});
	};
	// A graph's teardown touches services the host's thread uses (its upload client's timelines, descriptors): it holds still.
	if (m_async && !m_async->holding) HoldAsync({}, release);
	else release();
}

void GraphHost::SetGpuPassRangeCallbacks(RenderGraph::GpuPassRangeBegin begin, RenderGraph::GpuPassRangeEnd end) {
	{
		std::scoped_lock lock(m_configMutex);
		m_gpuPassRangeBegin = std::move(begin);
		m_gpuPassRangeEnd = std::move(end);
	}
	for (auto& instance : m_instances)
		if (instance.graph) instance.graph->SetGpuPassRangeCallbacks(m_gpuPassRangeBegin, m_gpuPassRangeEnd);
}

void GraphHost::EnsureTimelines() {
	// With m_configMutex held: a build on another thread may make them.
	if (!m_timelines.empty()) return;
	static constexpr const char* names[] = {"ORG host graphics timeline", "ORG host compute timeline", "ORG host copy timeline"};
	for (const auto* name : names) {
		auto fence = std::make_shared<rhi::TimelinePtr>();
		if (m_device.CreateTimeline(*fence, 0, name) != rhi::Result::Ok || !*fence)
			throw std::runtime_error("GraphHost could not create its queue timelines");
		m_timelines.push_back({std::move(fence), std::make_shared<std::atomic<uint64_t>>(1)});
	}
	// In-stream epochs' values are signalled by the host's own stream, which no ORG queue sees: the runtime's retirement
	// frontier covers every value given out on them, so nothing an in-stream list names is freed before that list ran.
	for (const auto& timeline : m_timelines) m_desc.runtime->RegisterAssignedTimeline(timeline.fence, timeline.nextValue);
}

// What one instance's graph is built from. Copied under m_configMutex, so a build on another thread sees one consistent state.
struct GraphHost::BuildInputs {
	size_t index = 0;
	uint32_t epoch = UINT32_MAX;
	std::vector<std::pair<std::string, ExtensionFactory>> factories;  // the extensions made into the epoch's graph
	std::vector<SharedQueueTimeline> timelines;
	std::vector<uint32_t> externalEpochs;
	bool inlineAtExecution = false;
	std::shared_ptr<PersistentAliasArena> aliasArena;
	std::shared_ptr<FrameAliasArena> frameArena;  // with frameSpan: SetFramePositions listed the epoch with its own transients
	AliasSpan frameSpan{};
	// The epoch's frame position (SetFramePositions), for the frame transients' spans; kNoPosition when it has none.
	uint32_t framePosition = alias::FrameSpans::kNoPosition;
	bool frameTransients = false;  // the host has frame transients: the graph reports what its passes use
	std::shared_ptr<rhi::TimelinePtr> frameWaitTimeline;
	RenderGraph::GpuPassRangeBegin gpuPassRangeBegin;
	RenderGraph::GpuPassRangeEnd gpuPassRangeEnd;
	uint64_t hostConfiguration = 0, epochConfiguration = 0, requests = 0, serial = 0;
	// A detached build's resolver capture context (BuildEpochGraph); null: the reshaped versions.
	std::shared_ptr<const ResolverCaptureContext> captureContext;
};

GraphHost::BuildInputs GraphHost::SnapshotBuildInputs(size_t index) {
	std::scoped_lock lock(m_configMutex);
	// Host-wide: without the timelines no epoch can run.
	EnsureTimelines();
	if (m_desc.sharedAliasArena && !m_aliasArena) m_aliasArena = PersistentAliasArena::Create();
	const auto& instance = m_instances[index];
	BuildInputs inputs;
	inputs.index = index;
	inputs.epoch = instance.epoch;
	// Only the extensions made into this epoch's graph (AddExtension's epochs): the others are not made, and gather nothing.
	for (const auto& entry : m_extensions)
		if (ExtensionInEpoch(entry, instance.epoch)) inputs.factories.emplace_back(entry.id, entry.factory);
	inputs.timelines = m_timelines;
	inputs.externalEpochs = m_externalEpochs;
	inputs.inlineAtExecution = m_inlineAtExecution;
	if (m_desc.sharedAliasArena && std::ranges::find(m_privateAliasEpochs, instance.epoch) == m_privateAliasEpochs.end()) inputs.aliasArena = m_aliasArena;
	for (const auto& entry : m_framePositions) {
		if (entry.epoch != instance.epoch) continue;
		inputs.framePosition = entry.position;
		if (entry.ownTransients) {
			inputs.frameArena = m_frameAliasArena;
			inputs.frameSpan = {entry.position, entry.position};
			inputs.aliasArena = nullptr;
		}
	}
	inputs.frameTransients = static_cast<bool>(m_frameTransients);
	inputs.frameWaitTimeline = m_frameWaitTimeline;
	inputs.gpuPassRangeBegin = m_gpuPassRangeBegin;
	inputs.gpuPassRangeEnd = m_gpuPassRangeEnd;
	inputs.hostConfiguration = m_hostConfiguration.load(std::memory_order_acquire);
	inputs.epochConfiguration = instance.configuration->load(std::memory_order_acquire);
	inputs.requests = instance.requests->load(std::memory_order_acquire);
	inputs.serial = m_buildSerial.fetch_add(1, std::memory_order_acq_rel) + 1;
	return inputs;
}

namespace {
// What a detached build lowers against: each texture's reshaped version (VersionedResource, "Array-size versions"), else the
// current one. Buffers have no slice count, so they take the current one.
struct ReshapedVersions final : IResourceVersions, IHostExecutionData {
	std::shared_ptr<const BufferVersion> Find(const VersionedBuffer&) const noexcept override { return nullptr; }
	std::shared_ptr<const TextureVersion> FindTexture(const VersionedTexture& texture) const noexcept override {
		try {
			return std::static_pointer_cast<const TextureVersion>(texture.Reshaped());
		} catch (...) {
			return nullptr;
		}
	}
	const void* TryGet(std::type_index type) const noexcept override {
		return type == std::type_index(typeid(IResourceVersions)) ? static_cast<const IResourceVersions*>(this) : nullptr;
	}
};

std::shared_ptr<const ResolverCaptureContext> ReshapedCaptureContext() {
	static const auto context = std::make_shared<const ResolverCaptureContext>(
		std::shared_ptr<const IHostExecutionData>(std::make_shared<const ReshapedVersions>()));
	return context;
}
} // namespace

std::unique_ptr<RenderGraph> GraphHost::MakeGraph(const BuildInputs& inputs, bool detached, detail::HostUploadExtension** uploads) {
	auto graph = std::make_unique<RenderGraph>(m_desc.runtime);
	try {
		// A detached build may be staged ahead of a new slice count, so it lowers for the pending version.
		if (detached) graph->SetBuildCaptureContext(inputs.captureContext ? inputs.captureContext : ReshapedCaptureContext());
		graph->SetSharedQueueTimelines(inputs.timelines);
		graph->SetPassEpochFilter(inputs.epoch);
		// Before the extensions gather their passes: one may record a pass at execution when the epoch is external
		// (RenderGraph::PassEpochInHostStream).
		graph->SetExternalPersistentEpochs(inputs.externalEpochs);
		// Likewise for a pass recorded at execution inline (RenderGraph::PassEpochRecordsAtExecution).
		graph->SetInlineAtExecution(inputs.inlineAtExecution);
		// A detached graph sets up nothing other graphs share until it is attached (RenderGraph::DeferSharedSetup): the upload
		// service's registry, the shared upload pass and the statistics query heap.
		auto hostUploads = std::make_unique<HostUploadExtension>(graph->GetUploadService(), detached);
		if (uploads) *uploads = hostUploads.get();
		if (detached) graph->DeferSharedSetup({detail::kUploadPassName});
		graph->RegisterExtension(std::move(hostUploads), "org.host.io");
		// The factories run here, on the building thread.
		for (const auto& [id, factory] : inputs.factories)
			if (auto extension = factory()) graph->RegisterExtension(std::move(extension), id);
		runtime::ScopedActiveGraphServices services(graph->GetUploadService(), graph->GetDescriptorService());
		graph->PrepareExtensionsForBuild();
		// The graph's memory snapshots (its VRAM dump, a host's memory UI): ORG's tracking world, as the contributor host's.
		graph->GetMemorySnapshotProvider().SetProvider(memory::CreateECSMemorySnapshotProvider());
		graph->CompileStructural();
		if (inputs.epoch != persistent::AllEpochs) graph->SetPersistentEpochOrder({inputs.epoch});
		graph->SetPersistentExternalHomeStates(m_desc.externalHomeStates);
		if (inputs.aliasArena) graph->SetPersistentAliasArena(inputs.aliasArena);
		// Lane: the epoch plus one, so the epoch's builds may share bytes (lane 0 shares with nothing whose span overlaps).
		if (inputs.frameArena) graph->SetFrameAliasRegion(inputs.frameArena, inputs.frameSpan, inputs.epoch + 1);
		// The frame transients' spans grow to this graph's position before it runs (its alias plan reports its resources).
		if (inputs.frameTransients)
			graph->SetFrameTransientObserver([this, epoch = inputs.epoch, position = inputs.framePosition, lane = inputs.frameArena ? inputs.epoch + 1 : 0u](
												 std::span<Resource* const> resources) { NoteFrameTransientUses(epoch, position, lane, resources); });
		// Each alias plan publishes the graph's usage summary to the frame placement, with where its own transients go today.
		{
			const auto memory = inputs.frameArena ? PlacementMemory::FrameArena
				: inputs.aliasArena ? PlacementMemory::SharedArena : PlacementMemory::PrivatePools;
			graph->SetPlacementSummaryObserver([placement = std::weak_ptr<FramePlacement>(m_framePlacement), memory](PlacementSummary summary) {
				summary.memory = memory;
				summary.Finish();
				if (const auto locked = placement.lock()) locked->Publish(std::move(summary));
			});
			// The version the graph's first plan takes: null while the frame placement is off.
			graph->SetFramePlacementSource([placement = std::weak_ptr<FramePlacement>(m_framePlacement)]() -> std::shared_ptr<const FramePlacementVersion> {
				const auto locked = placement.lock();
				return locked ? locked->Current() : nullptr;
			});
			graph->SetFramePlacementProbe([placement = std::weak_ptr<FramePlacement>(m_framePlacement)]() -> uint64_t {
				const auto locked = placement.lock();
				return locked ? locked->CurrentId() : 0;
			});
			// Versioned slots are candidates only while the frame placement is on.
			graph->SetFramePlacementEnabled([placement = std::weak_ptr<FramePlacement>(m_framePlacement)]() -> bool {
				const auto locked = placement.lock();
				return locked && locked->Enabled();
			});
		}
		graph->SetPersistentRecordingReuse(m_desc.reuseRecordings);
		graph->SetPersistentRecordingTimestampMode(m_recordingTimestamps.load(std::memory_order_acquire));
		graph->SetScheduleTraceEnabled(m_scheduleTrace.load(std::memory_order_acquire));
		{
			auto boundary = m_desc.queueBoundary;
			if (m_boundaryOverridden.load(std::memory_order_acquire)) {
				boundary.entry = m_boundaryEntry.load(std::memory_order_acquire);
				boundary.exit = m_boundaryExit.load(std::memory_order_acquire);
			}
			graph->SetExternalQueueBoundary(boundary);
		}
		graph->SetFrameWaitTimeline(inputs.frameWaitTimeline);
		graph->SetScopedFrameWait(m_desc.scopedFrameWait);
		graph->Setup();
		graph->SetGpuPassRangeCallbacks(inputs.gpuPassRangeBegin, inputs.gpuPassRangeEnd);
		// The lowering, pipelines and first schedule the first preparation would otherwise build inline on the host's thread.
		if (detached) graph->PrepareFirstPersistentBuild();
		graph->FinishExtensionsBuild();
	} catch (...) {
		// The half-built graph goes now: nothing was submitted with it.
		try {
			graph->FinishExtensionsBuild();
			graph->ShutdownExtensions();
			graph->ShutdownTaskWorkers();
		} catch (...) {
		}
		if (uploads) *uploads = nullptr;
		throw;
	}
	return graph;
}

void GraphHost::InstallGraph(size_t index, std::unique_ptr<RenderGraph> graph, uint64_t serial, uint64_t generation) {
	auto& instance = m_instances[index];
	// The graph it replaces: its own submitted work is what every queue has been given so far (the timelines are shared), taken
	// while it is still listed. It is released once that is done, the other epochs running meanwhile.
	auto done = instance.graph ? SubmittedPoint() : GpuPoint{};
	if (auto old = std::move(instance.graph)) {
		old->StopFrameProduction();
		m_retiredGraphs.push_back({std::move(old), std::move(done)});
		++m_asyncStats.rebuilds;
	}
	instance.graph = std::move(graph);
	instance.failed = false;
	instance.failure.clear();
	instance.attachedSerial = serial;
	// Every build's generation is new, staged builds included: a staged graph's is the one it was given (StageEpochGraph).
	instance.buildGeneration = generation ? generation : (std::max)(instance.buildGeneration, instance.lastGeneration) + 1;
	instance.lastGeneration = (std::max)(instance.lastGeneration, instance.buildGeneration);
	++m_buildGeneration;
	PublishDeclared(index);
}

bool GraphHost::BuildInstance(size_t index) {
	BT_ZONE_SCOPE("ORG.Host.Build");
	const auto begin = std::chrono::steady_clock::now();
	auto& instance = m_instances[index];
	const bool replacing = static_cast<bool>(instance.graph);
	const auto epochName = instance.epoch == persistent::AllEpochs ? std::string("(all)") : std::to_string(instance.epoch);
	// Why this build: the requests since the last one, each with its count. This build answers them, whether it succeeds or not.
	std::string why;
	for (const auto& [reason, count] : instance.rebuildReasons)
		why += (why.empty() ? "" : "; ") + reason + (count > 1 ? " (x" + std::to_string(count) + ")" : std::string());
	instance.rebuildReasons.clear();
	if (why.empty()) why = replacing ? "no request named" : "first build";
	const auto inputs = SnapshotBuildInputs(index);
	// The new graph is built before the old one stops: a build that fails leaves the epoch its old graph.
	std::unique_ptr<RenderGraph> graph;
	std::string error;
	try {
		graph = MakeGraph(inputs, false, nullptr);
	} catch (const std::exception& e) {
		error = e.what();
	} catch (...) {
		error = "unknown exception";
	}
	if (!error.empty() || !graph) {
		if (error.empty()) error = "the graph could not be made";
		// Not built again until a rebuild is requested: a broken declaration would fail every frame.
		instance.rebuildRequested = false;
		const bool kept = static_cast<bool>(instance.graph) && !InstanceFailed(index);
		if (!kept) {
			instance.failed = true;
			instance.failure = error;
		}
		spdlog::error("GraphHost: epoch {}'s graph failed to build ({}); the other epochs keep running: {}; why it was built: {}", epochName,
			kept ? "its previous graph is kept" : "it runs nothing", error, why);
		ReportEpochFailure({instance.epoch, instance.buildGeneration, kept, std::move(error)});
		return false;
	}
	InstallGraph(index, std::move(graph), inputs.serial);
	instance.rebuildRequested = false;
	if (replacing)
		spdlog::info("GraphHost: epoch {}'s graph built again (build {}) in {:.2f} ms; the other epochs' graphs kept; why: {}", epochName,
			instance.buildGeneration, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count(), why);
	return true;
}

GraphHost::EpochGraph::~EpochGraph() {
	if (!m_graph) return;
	// Never attached, so never prepared or submitted: its teardown waits for nothing.
	try {
		m_graph->ShutdownExtensions();
		m_graph->ShutdownTaskWorkers();
	} catch (...) {
	}
	m_graph.reset();
}

std::unique_ptr<GraphHost::EpochGraph> GraphHost::BuildEpochGraph(uint32_t epoch, std::string_view reason) {
	return BuildEpochGraph(epoch, reason, nullptr);
}

std::unique_ptr<GraphHost::EpochGraph> GraphHost::BuildEpochGraph(uint32_t epoch, std::string_view reason,
	std::shared_ptr<const ResolverCaptureContext> captureContext) {
	BT_ZONE_SCOPE("ORG.Host.BuildEpochGraph");
	const auto index = m_desc.epochOrder.empty() && epoch == persistent::AllEpochs ? 0 : InstanceOf(epoch);
	if (index == kNoInstance) throw std::invalid_argument("BuildEpochGraph: epoch " + std::to_string(epoch) + " is not one of the host's");
	const auto begin = std::chrono::steady_clock::now();
	std::unique_ptr<EpochGraph> result(new EpochGraph());
	result->m_epoch = m_instances[index].epoch;
	result->m_index = index;
	result->m_reason = reason.empty() ? std::string("unnamed request") : std::string(reason);
	try {
		auto inputs = SnapshotBuildInputs(index);
		inputs.captureContext = std::move(captureContext);
		result->m_hostConfiguration = inputs.hostConfiguration;
		result->m_epochConfiguration = inputs.epochConfiguration;
		result->m_requests = inputs.requests;
		result->m_serial = inputs.serial;
		result->m_graph = MakeGraph(inputs, true, &result->m_uploads);
	} catch (const std::exception& e) {
		result->m_error = e.what();
	} catch (...) {
		result->m_error = "unknown exception";
	}
	if (!result->m_graph && result->m_error.empty()) result->m_error = "the graph could not be made";
	result->m_buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
	return result;
}

GraphHost::AttachResult GraphHost::AttachEpochGraph(std::unique_ptr<EpochGraph>& built) {
	BT_ZONE_SCOPE("ORG.Host.AttachEpochGraph");
	CheckOwner("AttachEpochGraph");
	if (!built) return Refuse(AttachRefusal::NoGraph, UINT32_MAX, "no graph");
	if (built->m_index >= m_instances.size() || m_instances[built->m_index].epoch != built->m_epoch)
		return Refuse(AttachRefusal::UnknownEpoch, built->m_epoch, "the graph is not one of this host's epochs");
	if (!built->m_graph) return Refuse(AttachRefusal::BuildFailed, built->m_epoch, "its build failed: " + built->m_error);
	const auto index = built->m_index;
	auto& instance = m_instances[index];
	if (built->m_hostConfiguration != m_hostConfiguration.load(std::memory_order_acquire))
		return Refuse(AttachRefusal::HostConfiguration, instance.epoch, "the host's configuration changed since its build started");
	if (built->m_epochConfiguration != instance.configuration->load(std::memory_order_acquire))
		return Refuse(AttachRefusal::EpochConfiguration, instance.epoch, "the epoch's extensions or external state changed since its build started");
	if (built->m_serial < instance.attachedSerial)
		return Refuse(AttachRefusal::NewerAttached, instance.epoch, "the epoch runs a graph whose build started later");
	const auto epochName = instance.epoch == persistent::AllEpochs ? std::string("(all)") : std::to_string(instance.epoch);
	const auto begin = std::chrono::steady_clock::now();
	const bool replacing = static_cast<bool>(instance.graph);
	std::string failure;
	const auto replace = [&](size_t) {
		auto& graph = *built->m_graph;
		// The steps the build left out because other graphs share their objects. Nothing else uses them now: the host's thread
		// holds still. A failure here leaves the epoch its old graph.
		try {
			graph.FinishSharedSetup();
			if (built->m_uploads) built->m_uploads->ApplyResolveContext();
		} catch (const std::exception& e) {
			failure = std::string("its shared setup failed: ") + e.what();
			return false;
		}
		// What changed since the build without changing what it is built from.
		graph.SetGpuPassRangeCallbacks(m_gpuPassRangeBegin, m_gpuPassRangeEnd);
		graph.SetScheduleTraceEnabled(m_scheduleTrace.load(std::memory_order_acquire));
		built->m_uploads = nullptr;
		// A graph staged for the epoch is older than this attach's (or would have been rejected below): it goes.
		RetireStaged(index);
		InstallGraph(index, std::move(built->m_graph), built->m_serial);
		// Requests made before the build started are answered by it. A later one stays.
		if (instance.requests->load(std::memory_order_acquire) == built->m_requests) {
			instance.rebuildRequested = false;
			instance.rebuildReasons.clear();
		}
		return true;
	};
	if (m_async && !m_async->holding) ReplaceAsyncInstances({&index, 1}, replace);
	else (void)replace(index);
	if (!failure.empty()) return Refuse(AttachRefusal::SharedSetupFailed, instance.epoch, std::move(failure));
	PlaceAttachedAliases(index);
	m_attachCount.fetch_add(1, std::memory_order_relaxed);
	spdlog::info("GraphHost: epoch {}'s graph {} (build {}): built in {:.2f} ms off the host, attached in {:.3f} ms; why: {}", epochName,
		replacing ? "replaced" : "attached", instance.buildGeneration, built->m_buildMs,
		std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count(), built->m_reason);
	built.reset();
	return {true, {}};
}

std::vector<uint32_t> GraphHost::StaleAliasPlacementEpochs() const {
	CheckOwner("StaleAliasPlacementEpochs");
	std::vector<uint32_t> stale;
	// An epoch with a staged graph is not stale: the staged graph's placement grew the pool, and it replaces the attached one.
	// A graph placed from an older frame placement is not stale either: it places again from the current one in place, at its
	// next preparation, before it records (RenderGraph::SetFramePlacementProbe).
	for (const auto& instance : m_instances)
		if (instance.graph && !instance.staged && !instance.graph->PersistentAliasPlacementsCurrent()) stale.push_back(instance.epoch);
	return stale;
}

void GraphHost::RetireStaged(size_t index) {
	auto& instance = m_instances[index];
	if (m_async)
		for (uint32_t i = 0; i < m_async->epochs.size(); ++i) {
			if (m_async->instanceOf[i] != index) continue;
			m_async->stagedGraph[i] = nullptr;
			m_async->stagedGeneration[i] = 0;
			m_async->stagedRecordings[i].clear();
		}
	if (instance.staged && instance.staged->m_graph) {
		// Never submitted: it is released once the work submitted so far is done, like a replaced graph.
		instance.staged->m_uploads = nullptr;
		auto graph = std::move(instance.staged->m_graph);
		graph->StopFrameProduction();
		m_retiredGraphs.push_back({std::move(graph), SubmittedPoint()});
	}
	instance.staged.reset();
	instance.stagedGeneration = 0;
	PublishDeclared(index);
}

GraphHost::StageResult GraphHost::StageEpochGraph(std::unique_ptr<EpochGraph>& built) {
	BT_ZONE_SCOPE("ORG.Host.StageEpochGraph");
	CheckOwner("StageEpochGraph");
	StageResult result;
	const auto refuse = [&](AttachRefusal refusal, std::string reason) {
		auto refused = Refuse(refusal, built ? built->m_epoch : UINT32_MAX, std::move(reason));
		result.reason = std::move(refused.reason);
		result.refusal = refusal;
		return result;
	};
	if (!built) return refuse(AttachRefusal::NoGraph, "no graph");
	if (built->m_index >= m_instances.size() || m_instances[built->m_index].epoch != built->m_epoch)
		return refuse(AttachRefusal::UnknownEpoch, "the graph is not one of this host's epochs");
	const auto index = built->m_index;
	auto& instance = m_instances[index];
	// Nothing runs that a staging would keep: attached at once.
	if ((m_async && m_async->holding) || !instance.graph) {
		const auto attached = AttachEpochGraph(built);
		result.attached = attached.attached;
		result.reason = attached.reason;
		result.refusal = attached.refusal;
		result.generation = instance.buildGeneration;
		return result;
	}
	if (!built->m_graph) return refuse(AttachRefusal::BuildFailed, "its build failed: " + built->m_error);
	if (built->m_hostConfiguration != m_hostConfiguration.load(std::memory_order_acquire))
		return refuse(AttachRefusal::HostConfiguration, "the host's configuration changed since its build started");
	if (built->m_epochConfiguration != instance.configuration->load(std::memory_order_acquire))
		return refuse(AttachRefusal::EpochConfiguration, "the epoch's extensions or external state changed since its build started");
	if (built->m_serial < instance.attachedSerial) return refuse(AttachRefusal::NewerAttached, "the epoch runs a graph whose build started later");
	if (instance.staged && built->m_serial < instance.staged->m_serial)
		return refuse(AttachRefusal::NewerStaged, "a graph whose build started later is staged");
	const auto epochName = instance.epoch == persistent::AllEpochs ? std::string("(all)") : std::to_string(instance.epoch);
	const auto begin = std::chrono::steady_clock::now();
	const auto generation = (std::max)(instance.buildGeneration, instance.lastGeneration) + 1;
	std::string failure;
	const auto stage = [&] {
		auto& graph = *built->m_graph;
		// The shared setup steps but the upload service's registry, which the attach points at the new graph: the running graphs'
		// uploads keep resolving as before until then.
		try {
			graph.FinishSharedSetup();
		} catch (const std::exception& e) {
			failure = std::string("its shared setup failed: ") + e.what();
			return;
		}
		graph.SetGpuPassRangeCallbacks(m_gpuPassRangeBegin, m_gpuPassRangeEnd);
		graph.SetScheduleTraceEnabled(m_scheduleTrace.load(std::memory_order_acquire));
		RetireStaged(index);
		auto* raw = built->m_graph.get();
		instance.staged = std::move(built);
		instance.stagedGeneration = generation;
		instance.lastGeneration = generation;
		if (m_async) {
			// The staged graph's first recording places its transients, which may move ones the attached graph binds. From here the
			// attached graph keeps what it bound, and records nothing new.
			instance.graph->FreezeAliasedBindings();
			for (uint32_t i = 0; i < m_async->epochs.size(); ++i) {
				if (m_async->instanceOf[i] != index) continue;
				m_async->stagedGraph[i] = raw;
				m_async->stagedGeneration[i] = generation;
				m_async->stagedRecordings[i].clear();
			}
			// Requests made before the build started are answered by it. A later one stays. Without async epochs the attach
			// answers them (SwapStagedIn): a staged graph dropped before its attach answers nothing.
			if (instance.requests->load(std::memory_order_acquire) == instance.staged->m_requests) {
				instance.rebuildRequested = false;
				instance.rebuildReasons.clear();
			}
		}
		PublishDeclared(index);
	};
	// With async epochs the host's thread holds still while its view of the epoch changes. Nothing is detached: the attached graph
	// keeps its tickets. Without them nothing else runs now.
	if (m_async) HoldAsync({}, stage);
	else stage();
	if (!failure.empty()) return refuse(AttachRefusal::SharedSetupFailed, std::move(failure));
	m_stageCount.fetch_add(1, std::memory_order_relaxed);
	result.staged = true;
	result.generation = generation;
	spdlog::info("GraphHost: epoch {}'s graph staged (build {}): built in {:.2f} ms off the host, staged in {:.3f} ms; why: {}", epochName, generation,
		instance.staged->m_buildMs, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count(), instance.staged->m_reason);
	return result;
}

bool GraphHost::SwapStagedIn(size_t index) {
	auto& instance = m_instances[index];
	auto& built = instance.staged;
	const auto generation = instance.stagedGeneration;
	if (built->m_uploads) built->m_uploads->ApplyResolveContext();
	built->m_uploads = nullptr;
	std::vector<std::weak_ptr<const EpochRecording>> recordings;
	if (m_async)
		for (uint32_t i = 0; i < m_async->epochs.size(); ++i) {
			if (m_async->instanceOf[i] != index) continue;
			recordings = std::move(m_async->stagedRecordings[i]);
			m_async->stagedRecordings[i].clear();
			m_async->stagedGraph[i] = nullptr;
			m_async->stagedGeneration[i] = 0;
		}
	InstallGraph(index, std::move(built->m_graph), built->m_serial, generation);
	// Without async epochs the requests made before the build started are answered here (with them, at the staging).
	if (!m_async && instance.requests->load(std::memory_order_acquire) == built->m_requests) {
		instance.rebuildRequested = false;
		instance.rebuildReasons.clear();
	}
	// The recordings made on it are its tickets' candidates from now on.
	if (m_async)
		for (uint32_t i = 0; i < m_async->epochs.size(); ++i)
			if (m_async->instanceOf[i] == index) m_async->liveRecordings[i] = recordings;
	instance.staged.reset();
	instance.stagedGeneration = 0;
	PublishDeclared(index);
	return true;
}

GraphHost::AttachResult GraphHost::AttachStagedEpochGraphs(std::span<const uint32_t> epochs) {
	BT_ZONE_SCOPE("ORG.Host.AttachStagedEpochGraphs");
	CheckOwner("AttachStagedEpochGraphs");
	// Every epoch is checked before anything changes. A refusal leaves every staged graph staged.
	// The set itself first: every epoch the host's, each once.
	std::vector<size_t> indices;
	indices.reserve(epochs.size());
	for (const auto epoch : epochs) {
		const auto index = InstanceOf(epoch);
		if (index == kNoInstance) return Refuse(AttachRefusal::UnknownEpoch, epoch, "the epoch is not one of the host's");
		if (std::ranges::find(indices, index) != indices.end()) return Refuse(AttachRefusal::Duplicate, epoch, "the set names the epoch twice");
		indices.push_back(index);
	}
	for (const auto index : indices) {
		const auto epoch = m_instances[index].epoch;
		const auto& instance = m_instances[index];
		if (!instance.staged) return Refuse(AttachRefusal::NotStaged, epoch, "no graph is staged");
		const auto& built = *instance.staged;
		if (built.m_hostConfiguration != m_hostConfiguration.load(std::memory_order_acquire))
			return Refuse(AttachRefusal::HostConfiguration, epoch, "the host's configuration changed since its build started");
		if (built.m_epochConfiguration != instance.configuration->load(std::memory_order_acquire))
			return Refuse(AttachRefusal::EpochConfiguration, epoch, "the epoch's extensions or external state changed since its build started");
		if (built.m_serial < instance.attachedSerial)
			return Refuse(AttachRefusal::NewerAttached, epoch, "the epoch runs a graph whose build started later");
	}
	// Inside a submission callback on this thread the host's thread cannot be held.
	if (!m_desc.submissionThread && m_submittingTicket && !indices.empty())
		return Refuse(AttachRefusal::BeingSubmitted, epochs.front(), "called inside a submission");
	if (indices.empty()) return {true, {}};
	const auto begin = std::chrono::steady_clock::now();
	std::string names;
	for (const auto index : indices) {
		const auto& instance = m_instances[index];
		names += (names.empty() ? "" : ", ")
			+ (instance.epoch == persistent::AllEpochs ? std::string("(all)") : std::to_string(instance.epoch)) + " (build "
			+ std::to_string(instance.stagedGeneration) + ")";
	}
	if (m_async && !m_async->holding) ReplaceAsyncInstances(indices, [&](size_t i) { return SwapStagedIn(i); });
	else
		for (const auto index : indices) (void)SwapStagedIn(index);
	// Without async epochs: the set's graphs place in the host's epoch order (instances are in that order), before any prepares.
	if (!m_async) {
		auto ordered = indices;
		std::ranges::sort(ordered);
		for (const auto index : ordered) PlaceAttachedAliases(index);
	}
	m_attachCount.fetch_add(indices.size(), std::memory_order_relaxed);
	m_setAttachCount.fetch_add(1, std::memory_order_relaxed);
	spdlog::info("GraphHost: staged graphs of epochs {} attached together in {:.3f} ms", names,
		std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
	return {true, {}};
}

std::vector<uint32_t> GraphHost::EpochsDeclaring(std::span<const IVersionedResource* const> slots) const {
	std::vector<const void*> keys;
	keys.reserve(slots.size());
	for (const auto* slot : slots)
		if (slot) keys.push_back(slot->Key());
	std::vector<uint32_t> epochs;
	if (keys.empty()) return epochs;
	std::vector<DeclaredGraphs> declared;
	{
		std::scoped_lock lock(m_declaredMutex);
		declared = m_declared;
	}
	const auto declares = [&](const std::shared_ptr<const RenderGraph::DeclaredVersions>& graph) {
		if (!graph) return false;
		const auto set = graph->Keys();
		if (!set) return false;
		for (const auto* key : keys)
			if (std::binary_search(set->begin(), set->end(), key)) return true;
		return false;
	};
	for (size_t index = 0; index < declared.size() && index < m_instances.size(); ++index)
		if (declares(declared[index].attached) || declares(declared[index].staged)) epochs.push_back(m_instances[index].epoch);
	return epochs;
}

std::vector<uint32_t> GraphHost::EpochsAwaitingSlices(const IVersionedResource& slot, uint32_t slices) const {
	const void* key = slot.Key();
	std::vector<DeclaredGraphs> declared;
	{
		std::scoped_lock lock(m_declaredMutex);
		declared = m_declared;
	}
	std::vector<uint32_t> epochs;
	for (size_t index = 0; index < declared.size() && index < m_instances.size(); ++index) {
		const auto& next = declared[index].staged ? declared[index].staged : declared[index].attached;
		if (next && next->DeclaresOtherSlices(key, slices)) epochs.push_back(m_instances[index].epoch);
	}
	return epochs;
}

size_t GraphHost::DropStagedEpochGraphs(std::span<const uint32_t> epochs) {
	BT_ZONE_SCOPE("ORG.Host.DropStagedEpochGraphs");
	CheckOwner("DropStagedEpochGraphs");
	std::vector<size_t> indices;
	for (const auto epoch : epochs) {
		const auto index = InstanceOf(epoch);
		if (index == kNoInstance || !m_instances[index].staged || std::ranges::find(indices, index) != indices.end()) continue;
		indices.push_back(index);
	}
	if (indices.empty()) return 0;
	const auto drop = [&] {
		for (const auto index : indices) RetireStaged(index);
	};
	// RetireStaged changes the host thread's view of the epochs: it holds still meanwhile.
	if (m_async && !m_async->holding) HoldAsync({}, drop);
	else drop();
	spdlog::info("GraphHost: {} staged graph(s) dropped before their attach", indices.size());
	return indices.size();
}

void GraphHost::PlaceAttachedAliases(size_t index) {
	auto& instance = m_instances[index];
	if (m_async || !instance.graph) return;
	try {
		instance.graph->PlaceDeferredAliasesNow();
	} catch (const std::exception& e) {
		instance.failed = true;
		instance.failure = std::string("its alias placement at the attach failed: ") + e.what();
		spdlog::error("GraphHost: epoch {}'s graph failed to place its transients at its attach (build {}); it runs nothing until it is built "
			"again: {}", instance.epoch == persistent::AllEpochs ? std::string("(all)") : std::to_string(instance.epoch), instance.buildGeneration,
			e.what());
		ReportEpochFailure({instance.epoch, instance.buildGeneration, false, instance.failure});
	}
}

GraphHost::AttachStats GraphHost::AttachStatsTotal() const noexcept {
	AttachStats stats;
	stats.attached = m_attachCount.load(std::memory_order_relaxed);
	stats.staged = m_stageCount.load(std::memory_order_relaxed);
	stats.setsAttached = m_setAttachCount.load(std::memory_order_relaxed);
	for (size_t i = 0; i < stats.refusals.size(); ++i) stats.refusals[i] = m_refusalCounts[i].load(std::memory_order_relaxed);
	return stats;
}

const char* GraphHost::AttachRefusalName(AttachRefusal refusal) noexcept {
	switch (refusal) {
	case AttachRefusal::None: return "none";
	case AttachRefusal::NoGraph: return "no graph";
	case AttachRefusal::UnknownEpoch: return "unknown epoch";
	case AttachRefusal::Duplicate: return "epoch named twice";
	case AttachRefusal::NotStaged: return "not staged";
	case AttachRefusal::BuildFailed: return "build failed";
	case AttachRefusal::HostConfiguration: return "host configuration changed";
	case AttachRefusal::EpochConfiguration: return "epoch configuration changed";
	case AttachRefusal::NewerAttached: return "newer build attached";
	case AttachRefusal::NewerStaged: return "newer build staged";
	case AttachRefusal::BeingSubmitted: return "being submitted";
	case AttachRefusal::SharedSetupFailed: return "shared setup failed";
	case AttachRefusal::Count: break;
	}
	return "?";
}

GraphHost::AttachResult GraphHost::Refuse(AttachRefusal refusal, uint32_t epoch, std::string reason) noexcept {
	m_refusalCounts[static_cast<size_t>(refusal)].fetch_add(1, std::memory_order_relaxed);
	AttachResult result;
	result.reason = std::move(reason);
	result.refusal = refusal;
	result.epoch = epoch;
	return result;
}

void GraphHost::PublishDeclared(size_t index) {
	const auto& instance = m_instances[index];
	DeclaredGraphs graphs;
	if (instance.graph) graphs.attached = instance.graph->PersistentDeclaredVersions();
	if (instance.staged && instance.staged->m_graph) graphs.staged = instance.staged->m_graph->PersistentDeclaredVersions();
	{
		std::scoped_lock lock(m_declaredMutex);
		if (index < m_declared.size()) m_declared[index] = std::move(graphs);
	}
	PublishBuilds(index);
}

uint64_t GraphHost::StagedGeneration(uint32_t epoch) const noexcept {
	return BuildsOf(epoch).staged;
}

bool GraphHost::BuildDueInstances(bool all) {
	std::vector<size_t> due;
	for (size_t index = 0; index < m_instances.size(); ++index) {
		auto& instance = m_instances[index];
		// Placements in a pool generation another epoch's graph replaced (its pool grew): placed again by a rebuild, they share
		// the pool's memory again, and the old generation is freed.
		// With Desc::reportStaleAliasPlacements the application rebuilds them instead (StaleAliasPlacementEpochs).
		if (!m_desc.reportStaleAliasPlacements && instance.graph && !instance.rebuildRequested && !instance.staged
			&& !instance.graph->PersistentAliasPlacementsCurrent()) {
			MarkRebuild(instance, "its alias placements are in a pool another epoch's build grew");
			basic_telemetry::AddCounter("ORG.Host.AliasArenaRebuilds");
		}
		// A failed build waits for a rebuild request (BuildDue).
		if (all ? (!instance.graph && !instance.failed) || instance.rebuildRequested : BuildDue(index)) due.push_back(index);
	}
	const auto begin = std::chrono::steady_clock::now();
	if (m_async) RebuildAsyncInstances(due);
	else for (const auto index : due) (void)BuildInstance(index);
	// Many graphs at one build point: one line totals them (each graph's own line names its reasons).
	if (due.size() > 1)
		spdlog::info("GraphHost: {} epochs' graphs built at one build point in {:.2f} ms", due.size(),
			std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
	return !due.empty();
}

void GraphHost::SetFrameWaitTimeline(std::shared_ptr<rhi::TimelinePtr> timeline) {
	{
		std::scoped_lock lock(m_configMutex);
		m_frameWaitTimeline = std::move(timeline);
	}
	m_hostConfiguration.fetch_add(1, std::memory_order_acq_rel);
	// Every packet binds its timelines when it is prepared: the graphs' present ones cannot wait on it.
	if (AnyGraph()) RequestRebuild("frame-wait timeline set");
}

namespace {
// Pushes a_entries onto the submission thread's queue, in order, waiting for room first when it is full. The thread never
// waits for the render thread, so the wait ends. a_lock holds the queue's mutex; it is released before the wake.
template <class Queue, class Entries>
void PushSubmitEntries(Queue& queue, std::unique_lock<std::mutex>& a_lock, Entries&& a_entries) {
	if (queue.entries.size() >= Queue::kCapacity) {
		BT_ZONE_SCOPE("ORG.Host.SubmitBackpressure");
		const auto start = std::chrono::steady_clock::now();
		queue.progress.wait(a_lock, [&] { return queue.entries.size() < Queue::kCapacity; });
		++queue.stats.backpressureWaits;
		queue.stats.backpressureUs += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
	}
	for (auto& entry : a_entries) queue.entries.push_back(std::move(entry));
	queue.pushed += a_entries.size();
	queue.stats.peakDepth = (std::max)(queue.stats.peakDepth, queue.pushed - queue.passed);
	a_lock.unlock();
	queue.work.notify_one();
}

// Pushes a_entry, or keeps it with the held ones when the calling thread holds (GraphHost::HoldSubmissions).
template <class Queue>
void PushOrHold(Queue& queue, typename Queue::Entry a_entry) {
	std::unique_lock lock(queue.mutex);
	if (queue.holdDepth && queue.holder == std::this_thread::get_id()) {
		queue.held.push_back(std::move(a_entry));
		return;
	}
	std::array<typename Queue::Entry, 1> one{std::move(a_entry)};
	PushSubmitEntries(queue, lock, one);
}

// Pushes what the calling thread holds, with a_lock held; a_lock is held again on return.
template <class Queue>
void PushHeld(Queue& queue, std::unique_lock<std::mutex>& a_lock) {
	if (queue.held.empty()) return;
	BT_ZONE_SCOPE("ORG.Host.PushHeldSubmissions");
	auto held = std::exchange(queue.held, {});
	PushSubmitEntries(queue, a_lock, held);
	a_lock.lock();
}
}  // namespace

GraphHost::GpuPoint GraphHost::SubmittedPoint() const {
	if (auto* queue = m_submitQueue.get()) {
		std::unique_lock lock(queue->mutex);
		// What this thread holds goes first: the point must come after it.
		if (queue->holder == std::this_thread::get_id()) PushHeld(*queue, lock);
		// Tickets still queued have no values yet. The point waits for the thread to get past them: a marker behind them, which
		// the thread fills in with the values then. The thread's own calls are in order already.
		if (std::this_thread::get_id() != queue->threadId && queue->passed < queue->pushed) {
			std::shared_ptr<GpuPoint::Pending> marker;
			if (!queue->entries.empty() && queue->entries.back().kind == SubmitQueue::Entry::Kind::Marker) {
				marker = queue->entries.back().marker;
			} else {
				marker = std::make_shared<GpuPoint::Pending>();
				SubmitQueue::Entry entry;
				entry.kind = SubmitQueue::Entry::Kind::Marker;
				entry.marker = marker;
				queue->entries.push_back(std::move(entry));
				++queue->pushed;
				lock.unlock();
				queue->work.notify_one();
			}
			GpuPoint point;
			point.pending = std::move(marker);
			return point;
		}
	}
	return SubmittedPointNow();
}

GraphHost::GpuPoint GraphHost::SubmittedPointNow() const {
	// Every queue of every graph, each timeline once: the shared ones' values are the host's, the same in each registry.
	GpuPoint point;
	for (const auto& instance : m_instances) {
		if (!instance.graph) continue;
		const auto& queues = instance.graph->GetQueueRegistry();
		for (size_t index = 0; index < queues.SlotCount(); ++index) {
			const auto slot = static_cast<QueueSlotIndex>(static_cast<uint8_t>(index));
			auto owner = queues.GetFenceOwner(slot);
			const auto next = queues.GetCurrentFenceValue(slot);
			if (!owner || next <= 1
				|| std::ranges::any_of(point.points, [&](const auto& known) { return known.first == owner; }))
				continue;
			point.points.emplace_back(std::move(owner), next - 1);
		}
	}
	return point;
}

void GraphHost::SealSubmitted() {
	auto* queue = m_submitQueue.get();
	if (queue && m_desc.queuedSeals && std::this_thread::get_id() != queue->threadId) {
		// The seal goes behind the tickets queued so far. Until the thread passes it, the sealed point is that pending part: its
		// waits wait for the thread first, which never waits for this thread.
		auto marker = std::make_shared<GpuPoint::Pending>();
		{
			std::scoped_lock lock(m_sealedMutex);
			m_sealed = {};
			m_sealed.pending = marker;
		}
		// Counted now: the commits after this one are behind the seal in the queue.
		m_sealsPushed.fetch_add(1, std::memory_order_seq_cst);
		EnqueueOnSubmitThread([this, marker] { PassSeal(marker); });
	} else {
		// Sealed means on the device's queues: what the submission thread still holds goes first.
		DrainSubmissions("a seal (the frame's end)");
		m_sealsPushed.fetch_add(1, std::memory_order_seq_cst);
		PassSeal(nullptr);
	}
	// Once a frame: chunks the runs in flight have not needed for a while go.
	if (const auto& arena = m_desc.runtime->Arena()) arena->Trim();
	// The frame boundary: relocations switch here, before the next frame prepares anything.
	(void)CommitRelocationsNow(false);
	// Now and then, where alias memory goes (FramePlacementSettings::memoryLogSeconds). The owner thread reads the instances.
	if (m_framePlacement && IsOwnerThread() && m_framePlacement->MemoryLogDue()) LogFramePlacementMemory();
}

void GraphHost::LogFramePlacementMemory() const {
	// Which frame placement version each graph's latest alias plan placed from, with the epochs. A graph that has not planned
	// yet (built, never prepared) holds nothing.
	std::map<uint64_t, std::pair<uint64_t, std::string>> byVersion;  // version: (resources placed, epochs)
	size_t unplanned = 0;
	for (const auto& instance : m_instances) {
		if (!instance.graph) continue;
		const auto use = instance.graph->FramePlacementInUse();
		if (!use.planned) {
			++unplanned;
			continue;
		}
		auto& entry = byVersion[use.version];
		entry.first += use.placed;
		entry.second += (entry.second.empty() ? "" : ",") + std::to_string(instance.epoch);
	}
	std::string graphs;
	for (const auto& [version, entry] : byVersion)
		graphs += (graphs.empty() ? "" : "; ") + (version ? "version " + std::to_string(version) : std::string("no version")) + ": "
			+ std::to_string(entry.first) + " resources placed, epochs " + entry.second;
	if (unplanned) graphs += (graphs.empty() ? "" : "; ") + std::to_string(unplanned) + " graphs not planned yet";
	m_framePlacement->LogMemory(m_aliasArena.get(), graphs);
}

size_t GraphHost::CommitRelocations(bool gpuIdle) {
	CheckOwner("CommitRelocations");
	return CommitRelocationsNow(gpuIdle);
}

size_t GraphHost::CommitRelocationsNow(bool gpuIdle) {
	// Before the first build there is no queue to copy on. Sources keep what they have until then.
	auto* graph = AnyGraph();
	if (!graph) return 0;
	try {
		relocation::CommitContext context;
		context.gpuIdle = gpuIdle;
		const auto& queues = graph->GetQueueRegistry();
		context.client = &graph->GetExternalClient();
		context.copySlot = static_cast<uint32_t>(ToUnderlying(queues.FindGraphicsSlot()));
		for (size_t index = 0; index < queues.SlotCount(); ++index)
			context.queues.push_back(queues.GetQueue(static_cast<QueueSlotIndex>(static_cast<uint8_t>(index))));
		return relocation::Service::Get().Commit(context);
	} catch (const std::exception& error) {
		spdlog::error("GraphHost: committing relocations failed: {}", error.what());
		return 0;
	}
}

relocation::DefragmentReport GraphHost::DefragmentAtIdle(const relocation::DefragmentBudget& budget) {
	CheckOwner("DefragmentAtIdle");
	relocation::DefragmentReport report;
	auto* allocator = DeviceManager::GetInstance().GetAllocator();
	if (!AnyGraph() || !allocator) return report;
	// Only at a settled point. A graph whose shared pools another graph changed (a resource it pooled went to memory of its own)
	// waits for its rebuild. A move would make its kept recordings record again against the old placement, which the binding
	// check refuses. The next call tries again.
	for (const auto& instance : m_instances)
		if (instance.graph && !instance.graph->PersistentAliasPlacementsCurrent()) {
			static uint32_t logged = 0;
			if (logged < 8) {
				++logged;
				spdlog::info("ORG defragmenter: waiting, epoch {}'s graph is waiting for a rebuild of its alias placements", instance.epoch);
			}
			return report;
		}
	auto& defragmenter = relocation::Defragmenter::Get();
	if (!defragmenter.WantsCall(budget, *allocator)) return report;
	BT_ZONE_SCOPE("ORG.Defragmenter.Call");
	using Clock = std::chrono::steady_clock;
	const auto milliseconds = [](Clock::time_point from, Clock::time_point to) {
		return std::chrono::duration<double, std::milli>(to - from).count();
	};
	const auto start = Clock::now();
	// Every queue the graphs use reaches what was submitted so far. A bounded wait: a GPU that does not go idle in time skips the
	// call rather than stall the loading screen.
	constexpr uint32_t kIdleTimeoutMs = 2000;
	bool idle = true;
	const auto waitIdle = [&] {
		const auto from = Clock::now();
		DrainSubmissions("defragmentation");
		idle = SubmittedPoint().Wait(kIdleTimeoutMs);
		report.waitMilliseconds += milliseconds(from, Clock::now());
	};
	while (report.passes < budget.maxPasses && milliseconds(start, Clock::now()) < budget.maxMilliseconds) {
		waitIdle();
		if (!idle) {
			spdlog::warn("ORG defragmenter: the GPU did not go idle within {} ms. No pass this call", kIdleTimeoutMs);
			break;
		}
		// Retirements whose GPU work is done go now. Old ranges of the last pass free their blocks here, so the next pass counts them.
		(void)DescriptorHeapManager::GetInstance().ProcessDeferredReleasesElsewhere();
		if (!defragmenter.ArmPass(budget, *allocator)) break;
		const auto passStart = Clock::now();
		(void)CommitRelocationsNow(true);
		const double passMs = milliseconds(passStart, Clock::now());
		defragmenter.FinishPass(passMs, report, waitIdle);
		report.passMilliseconds += passMs;
		++report.passes;
	}
	report.campaignActive = defragmenter.CampaignActive();
	if (report.moves) basic_telemetry::AddCounter("ORG.Defragmenter.Moves", static_cast<int64_t>(report.moves));
	return report;
}

void GraphHost::EndDefragmentation() {
	CheckOwner("EndDefragmentation");
	relocation::Defragmenter::Get().EndCampaign("the idle point ended");
}

void GraphHost::PassSeal(const std::shared_ptr<GpuPoint::Pending>& a_marker) {
	auto point = SubmittedPointNow();
	if (a_marker) {
		a_marker->points = point.points;
		a_marker->passed.store(true, std::memory_order_release);
		a_marker->passed.notify_all();
	}
	{
		std::scoped_lock lock(m_sealedMutex);
		m_sealedPassed = point;
		// A later seal may be queued already. Then the sealed point stays that one.
		if (!a_marker || m_sealed.pending == a_marker) m_sealed = std::move(point);
	}
	m_seals.fetch_add(1, std::memory_order_seq_cst);
	// An epoch whose copies were busy with unsealed runs can take one again: the host's thread prepares its ticket now. The flag is
	// set before the host's thread reads the seals again, so either it sees this count or this sees its flag.
	if (m_async && m_async->anyCopiesBusy.exchange(false, std::memory_order_seq_cst)) (void)m_async->completionWake->Notify();
}

GraphHost::GpuPoint GraphHost::PassedSealedPoint() const {
	std::scoped_lock lock(m_sealedMutex);
	return m_sealedPassed;
}

GraphHost::GpuPoint GraphHost::SealedPoint() const {
	std::scoped_lock lock(m_sealedMutex);
	return m_sealed;
}

bool GraphHost::NudgeHostThread() {
	// A requester that waits for the host's thread wakes it again: the thread sleeping with a request queued means a wake was lost.
	auto* async = m_async.get();
	if (!async || !async->completionWake) return false;
	const auto sleeping = async->sleepingAt.load(std::memory_order_acquire);
	if (!sleeping || !m_recordingQueue->head.load(std::memory_order_acquire)) return false;
	if (!async->nudgeLogged.exchange(true))
		spdlog::warn("GraphHost: the host's thread slept on wake version {} with a recording request queued (version now {}): woken again", sleeping - 1,
			async->completionWake->WakeVersion());
	(void)async->completionWake->Notify();
	return true;
}

void* GraphHost::SubmitThreadNativeHandle() noexcept {
	if (!m_submitQueue || !m_submitQueue->thread.joinable()) return nullptr;
	return reinterpret_cast<void*>(m_submitQueue->thread.native_handle());
}

void* GraphHost::HostThreadNativeHandle() noexcept {
	if (!m_async || !m_async->thread.joinable()) return nullptr;
	return reinterpret_cast<void*>(m_async->thread.native_handle());
}

std::string GraphHost::DescribeAsyncState(bool a_pollTimelines) const {
	// Read racily from another thread: a message for a failure, never a decision.
	const auto* async = m_async.get();
	if (!async) return "no async host thread";
	const auto n = [](uint64_t a_value) { return std::to_string(a_value); };
	// A completed value takes the RHI's timeline lock. A caller inside a submission may hold it already.
	const auto completed = [&](auto& a_timeline) {
		return a_pollTimelines ? n(a_timeline->GetCompletedValue()) : std::string("?");
	};
	const uint64_t seals = m_seals.load(std::memory_order_acquire);
	std::string out = "host thread: failed " + n(async->failed.load()) + ", messages written " + n(async->written.load()) + " read " +
		n(async->read.load()) + ", wakes on GPU " + n(async->wakesOnGpu.load()) + ", sleeping on " + n(async->sleepingAt.load()) + " (version + 1), wake version " +
		n(async->completionWake ? async->completionWake->WakeVersion() : 0) + ", requests queued " + n(m_recordingQueue->head.load() != nullptr) +
		", accepting " + n(m_recordingQueue->accepting.load()) + ", seals " + n(seals) + ", requests held " + n(async->pendingRequests.size());
	out += "; epochs (index:epoch paused/in hand/queued/failed/copiesBusy/generation):";
	for (size_t i = 0; i < async->epochs.size(); ++i) {
		const auto busy = async->copiesBusyAt ? async->copiesBusyAt[i].load() : Async::kNoSeal;
		out += " " + n(i) + ":" + n(async->epochs[i]) + " " + n(async->paused[i]) + "/" + n(i < async->inHand.size() ? async->inHand[i] : 0) +
			"/" + n(async->queues ? async->queues[i].count.load() : 0) + "/" + n(async->epochFailed && async->epochFailed[i].load()) + "/" +
			(busy == Async::kNoSeal ? std::string("-") : n(busy)) + "/" + n(async->generation[i]);
	}
	out += "; host timelines (t<index> last assigned/completed):";
	for (const auto& timeline : m_timelines)
		if (timeline.fence && *timeline.fence)
			out += " t" + n((*timeline.fence)->GetHandle().index) + " " + n(timeline.nextValue->load() - 1) + "/" + completed(*timeline.fence);
	// The runs in flight, oldest first, and what each waits for. Without polling, never block: the lease lock's holder may be
	// waiting for the RHI lock the caller holds.
	std::unique_lock lock(m_leaseMutex, std::defer_lock);
	if (a_pollTimelines)
		lock.lock();
	else if (!lock.try_lock())
		return out + "; leases: (locked)";
	out += "; leases: " + n(m_leasesInFlight) + " in flight of " + n(m_leaseStore.size()) + "; submitted (index#submission s=seal, points):";
	size_t listed = 0;
	for (const auto* lease : m_submittedLeases) {
		if (++listed > 32) {
			out += " ...";
			break;
		}
		out += " [" + n(lease->index) + (lease->external ? " external" : " #" + n(lease->submission)) + " s" + n(lease->seal);
		for (const auto& [timeline, value] : lease->points)
			if (timeline && *timeline)
				out += " t" + n((*timeline)->GetHandle().index) + " " + n(value) + "/" + completed(*timeline);
		out += "]";
	}
	return out;
}

bool GraphHost::BuildIfRequested() {
	CheckOwner("BuildIfRequested");
	ReleaseRetiredGraphs(false);
	return BuildDueInstances(true);
}

void GraphHost::ExecuteFrame(const UpdateExecutionContext& frame, const FrameCallback& beforePrepare, uint32_t epoch,
	std::shared_ptr<const void> resourceOwner) {
	struct Restore {
		const UpdateExecutionContext*& context;
		~Restore() { context = nullptr; }
	} restore{m_frameContext};
	m_frameContext = &frame;
	ExecuteFrame(frame.hostData, beforePrepare, epoch, std::move(resourceOwner));
}

void GraphHost::ExecuteFrame(const IHostExecutionData* hostData, const FrameCallback& beforePrepare,
	uint32_t epoch, std::shared_ptr<const void> resourceOwner) {
	BT_ZONE_SCOPE("ORG.Host.ExecuteFrame");
	if (m_async) throw std::logic_error("ExecuteFrame while async epochs run: the host's thread owns the graphs (use SubmitEpoch)");
	m_lastTimings = {};
	auto lap = [last = std::chrono::steady_clock::now()](double& a_into) mutable {
		const auto now = std::chrono::steady_clock::now();
		a_into += std::chrono::duration<double, std::micro>(now - last).count();
		last = now;
	};
	const auto index = epoch == persistent::AllEpochs ? 0 : InstanceOf(epoch);
	if (index == kNoInstance) throw std::invalid_argument("ExecuteFrame: epoch " + std::to_string(epoch) + " is not one of the host's");
	ReleaseRetiredGraphs(false);
	if (BuildDue(index)) (void)BuildInstance(index);
	lap(m_lastTimings.buildUs);
	// An epoch whose graph failed runs nothing; the others are not affected.
	if (!m_instances[index].graph || m_instances[index].failed)
		throw EpochFailedError(m_instances[index].epoch, m_instances[index].failure);
	auto& graph = *m_instances[index].graph;
	// A failure preparing or recording this epoch's graph is the epoch's: it is marked failed and reported, and runs nothing until
	// it is built again. What Execute may have submitted before it failed keeps its owner (m_uncertainExecutionOwners).
	const auto failEpoch = [&](const std::string& error) {
		auto& instance = m_instances[index];
		instance.failed = true;
		instance.failure = error;
		spdlog::error("GraphHost: epoch {}'s graph failed to prepare (build {}); the other epochs keep running: {}",
			instance.epoch == persistent::AllEpochs ? std::string("(all)") : std::to_string(instance.epoch), instance.buildGeneration, error);
		ReportEpochFailure({instance.epoch, instance.buildGeneration, false, error});
		return EpochFailedError(instance.epoch, error);
	};
	runtime::ScopedActiveGraphServices services(graph.GetUploadService(), graph.GetDescriptorService());
	// Frame pacing: this epoch's run of framesInFlight of its executions back is waited for and retired. It was submitted to the device by its own Execute, so the wait always ends. Every other run whose GPU work is done
	// retires too (their statistics are read, their memory freed).
	{
		// Per epoch: its run framesInFlight of its own executions back, so no epoch's graph has more runs in flight than its
		// admission holds, however often each epoch executes in a frame.
		const size_t inFlight = (std::max)(m_desc.framesInFlight, 1u);
		const size_t positions = inFlight * (std::max<size_t>)(m_instances.size(), 1);
		if (m_syncLeases.size() != positions) {
			m_syncLeases.assign(positions, nullptr);
			m_syncSubmissions.assign(positions, 0);
			m_syncExecutions.assign(m_instances.size(), 0);
		}
		m_syncPosition = index * inFlight + m_syncExecutions[index]++ % inFlight;
		// The run at a position, if its lease still holds it and it is not complete.
		const auto pending = [&](size_t position) -> RunLease* {
			auto* run = m_syncLeases[position];
			return run && run->state == RunLease::State::Submitted && run->submission == m_syncSubmissions[position] ? run : nullptr;
		};
		auto* previous = pending(m_syncPosition);
		m_syncLeases[m_syncPosition] = nullptr;
		if (previous) {
			BT_ZONE_SCOPE("ORG.Host.WaitFramePacing");
			for (const auto& [timeline, value] : previous->points) (void)(*timeline)->HostWait(value);
		}
		(void)RetireReachedLeases(nullptr);
	}
	lap(m_lastTimings.waitUs);
	auto& lease = OpenLease(nullptr);
	const uint32_t slot = lease.index;
	m_syncRun = lease.Ref();
	// A failure before submission gives the lease back: nothing of the run reached the GPU. Once Execute ran, the lease holds its
	// points (below), and it retires with a later pacing pass.
	struct ReleaseUnsubmitted {
		GraphHost& host;
		RunLease& lease;
		~ReleaseUnsubmitted() {
			if (lease.state == RunLease::State::Open) host.ReleaseLease(lease, nullptr);
		}
	} releaseUnsubmitted{*this, lease};
	// The run's index retired before it was opened: its upload pages are free again, and what was staged since takes it.
	if (auto* uploads = graph.GetUploadService()) uploads->ProcessDeferredReleases(slot);
	m_desc.runtime->Retire();
	lap(m_lastTimings.releaseUs);
	graph.SetPersistentEpoch(epoch);
	graph.SetFrameWaitValue(m_frameWaitValue);
	// The shared upload service resolves registry handles in one registry. A pass queues an upload by a handle of its own graph's
	// registry in its Update, and this graph's upload pass records it in Execute below, so the service resolves in this graph's
	// registry while the graph is prepared and executed. Otherwise a host of several graphs resolves every handle in the registry
	// of the graph attached last, which names another resource (or none) at that index.
	if (m_instances.size() > 1)
		if (auto* uploads = graph.GetUploadService()) uploads->SetUploadResolveContext({&graph.GetResourceRegistry(), 0});
	if (beforePrepare) {
		BT_ZONE_SCOPE("ORG.Host.CommitInputs");
		beforePrepare(graph);
	}
	lap(m_lastTimings.beforePrepareUs);
	UpdateExecutionContext update{};
	update.frameIndex = slot;
	update.preparationSlot = slot;
	update.run = m_syncRun;
	update.frameFenceValue = m_frameNumber + 1;
	update.hostData = hostData;
	if (m_frameContext) {
		update.resolverCaptureContext = m_frameContext->resolverCaptureContext;
		update.deltaTime = m_frameContext->deltaTime;
		update.ownedHostData = m_frameContext->ownedHostData;
		update.beforeCompileFrame = m_frameContext->beforeCompileFrame;
	}
	try {
		graph.Update(update, m_device);
	} catch (const std::exception& e) {
		throw failEpoch(e.what());
	}
	lap(m_lastTimings.updateUs);
	PassExecutionContext execute{};
	execute.device = m_device;
	execute.frameIndex = slot;
	execute.frameFenceValue = m_frameNumber + 1;
	execute.hostData = hostData;
	if (m_frameContext) execute.ownedHostData = m_frameContext->ownedHostData;
	// Reserve CPU ownership before submission. Execute may fail after accepting a
	// prefix of its batches, in which case recovery retains this owner.
	const bool heldResources = static_cast<bool>(resourceOwner);
	if (heldResources) m_uncertainExecutionOwners.push_back(std::move(resourceOwner));
	try {
		// Execute takes and publishes the queues' values: deferred deletions are not sealed meanwhile.
		Runtime::AssignmentScope assigning(*m_desc.runtime);
		graph.Execute(execute);
	} catch (const std::exception& e) {
		// Some of its batches may have been accepted: the lease waits for every queue's current value.
		auto& queues = graph.GetQueueRegistry();
		lease.points.clear();
		for (size_t queueIndex = 0; queueIndex < queues.SlotCount(); ++queueIndex) {
			const auto queueSlot = static_cast<QueueSlotIndex>(static_cast<uint8_t>(queueIndex));
			if (const auto next = queues.GetCurrentFenceValue(queueSlot); next > 1) lease.points.emplace_back(queues.GetFenceOwner(queueSlot), next - 1);
		}
		{
			std::scoped_lock lock(m_leaseMutex);
			lease.state = RunLease::State::Submitted;
			m_submittedLeases.push_back(&lease);
		}
		throw failEpoch(e.what());
	}
	lap(m_lastTimings.executeUs);
	m_lastTimings.execute = graph.LastPersistentExecuteTimings();
	// Execute submitted the frame: every batch signalled its queue's timeline, so the value each queue
	// has reached is the run's completion on it.
	{
		auto& queues = graph.GetQueueRegistry();
		lease.points.clear();
		for (size_t queueIndex = 0; queueIndex < queues.SlotCount(); ++queueIndex) {
			const auto queueSlot = static_cast<QueueSlotIndex>(static_cast<uint8_t>(queueIndex));
			if (const auto next = queues.GetCurrentFenceValue(queueSlot); next > 1)
				lease.points.emplace_back(queues.GetFenceOwner(queueSlot), next - 1);
		}
		lease.hostFrame = m_frameNumber;
		lease.submission = ++m_submissions;
		lease.seal = m_sealsPushed.load(std::memory_order_relaxed);
		if (heldResources) {
			const auto& owner = m_uncertainExecutionOwners.back();
			if (owner && !lease.points.empty()) {
				std::vector<DescriptorHeapManager::QueueFenceSnapshotPoint> points;
				points.reserve(lease.points.size());
				for (const auto& [timeline, value] : lease.points)
					points.push_back({timeline->Get(), value});
				DescriptorHeapManager::GetInstance().RetireExecutionLease(owner, std::move(points));
			}
			m_uncertainExecutionOwners.pop_back();
		}
		{
			std::scoped_lock lock(m_leaseMutex);
			lease.state = RunLease::State::Submitted;
			m_submittedLeases.push_back(&lease);
		}
		m_syncLeases[m_syncPosition] = &lease;
		m_syncSubmissions[m_syncPosition] = lease.submission;
	}
	lap(m_lastTimings.signalUs);
	m_lastHostFrame = m_frameNumber;
	++m_frameNumber;
}

// ---- async epochs ----


void GraphHost::SetAsyncEpochs(std::vector<uint32_t> epochs, std::vector<uint32_t> revisionEpochs) {
	StopAsync();
	for (const auto epoch : epochs)
		if (InstanceOf(epoch) == kNoInstance) throw std::invalid_argument("An async epoch must be one of the host's epochs");
	m_asyncEpochs = std::move(epochs);
	for (const auto epoch : revisionEpochs)
		if (std::ranges::find(m_asyncEpochs, epoch) == m_asyncEpochs.end())
			throw std::invalid_argument("A revision-driven epoch must be an async epoch");
	if (!revisionEpochs.empty() && !m_desc.reuseRecordings)
		throw std::logic_error("Revision-driven epochs need Desc::reuseRecordings (replayable recordings)");
	m_revisionEpochs = std::move(revisionEpochs);
	// Their copies' leases are made now, on the owner thread before the host's thread starts. EpochCopyRun then reads them on any
	// thread, and nothing adds to the list afterwards.
	for (const auto epoch : m_revisionEpochs) (void)CopyLeases(epoch);
	if (m_asyncEpochs.empty()) {
		m_recordingQueue->accepting.store(false, std::memory_order_release);
		return;
	}
	BuildDueInstances(false);
	StartAsync();
}

void GraphHost::SetPrivateAliasEpochs(std::vector<uint32_t> epochs) {
	CheckOwner("SetPrivateAliasEpochs");
	std::scoped_lock lock(m_configMutex);
	m_privateAliasEpochs = std::move(epochs);
}

struct GraphHost::FrameTransients {
	std::mutex mutex;
	alias::FrameSpans spans;
	struct Group {
		std::vector<std::shared_ptr<FrameAliasArena::Reservation>> reservations;
		std::vector<std::weak_ptr<PixelBuffer>> members;
	};
	std::unordered_map<uint64_t, Group> groups;             // by FrameSpans group id
	std::unordered_map<std::string, uint64_t> groupByName;  // member resource name -> group
	std::shared_ptr<FrameAliasArena> arena;
};

namespace {
AliasSpan ToAliasSpan(alias::FrameSpans::Span span) { return span.Empty() ? AliasSpan::None() : AliasSpan{span.first, span.last}; }
std::string DescribeSpan(alias::FrameSpans::Span span) {
	return span.Empty() ? std::string("none") : std::to_string(span.first) + "-" + std::to_string(span.last);
}
}

void GraphHost::SetFramePositions(std::vector<FramePosition> positions) {
	CheckOwner("SetFramePositions");
	std::scoped_lock lock(m_configMutex);
	if (!m_frameAliasArena) m_frameAliasArena = FrameAliasArena::Create();
	if (!m_frameTransients) {
		m_frameTransients = std::make_shared<FrameTransients>();
		m_frameTransients->arena = m_frameAliasArena;
	}
	uint32_t last = 0;
	for (const auto& entry : positions) last = (std::max)(last, entry.position);
	{
		std::scoped_lock spans(m_frameTransients->mutex);
		m_frameTransients->spans.SetLastPosition(last);
	}
	m_framePositions = std::move(positions);
	m_framePlacement->SetFrameArena(m_frameAliasArena);
	// The frame placement's worker makes no version until each epoch with a position has a summary (the first plan with
	// candidates makes one at once).
	std::vector<uint32_t> epochs;
	for (const auto& entry : m_framePositions) epochs.push_back(entry.epoch);
	m_framePlacement->SetFrameEpochs(std::move(epochs));
}

std::shared_ptr<FrameAliasArena> GraphHost::FrameArena() const {
	std::scoped_lock lock(m_configMutex);
	return m_frameAliasArena;
}

std::shared_ptr<const void> GraphHost::AddFrameTransients(std::span<const std::shared_ptr<PixelBuffer>> textures, std::string label) {
	std::shared_ptr<FrameTransients> state;
	{
		std::scoped_lock lock(m_configMutex);
		state = m_frameTransients;
	}
	if (!state || !state->arena) return nullptr;
	std::scoped_lock lock(state->mutex);
	std::vector<uint64_t> ids;
	for (const auto& texture : textures)
		if (texture) ids.push_back(texture->GetGlobalResourceID());
	const auto group = state->spans.AddGroup(label, ids);
	// The span its label reached before (a resize), else none: nothing uses the textures yet.
	const auto span = state->spans.SpanOf(group);
	std::vector<std::shared_ptr<FrameAliasArena::Reservation>> reservations;
	try {
		reservations = state->arena->ReserveTextures(ToAliasSpan(span), textures, label);
	} catch (...) {
		state->spans.RemoveGroup(group);
		throw;
	}
	auto& entry = state->groups[group];
	entry.reservations = reservations;
	for (const auto& texture : textures)
		if (texture) {
			entry.members.push_back(texture);
			state->groupByName[texture->GetName()] = group;
		}
	spdlog::info("GraphHost: frame transients '{}' placed ({} textures), span {}: derived from the graphs that use them", label, ids.size(),
		DescribeSpan(span));
	// The owner keeps the reservations; when it goes, the group leaves the derivation.
	struct Owner {
		std::weak_ptr<FrameTransients> state;
		uint64_t group = 0;
		std::vector<std::shared_ptr<FrameAliasArena::Reservation>> reservations;
		~Owner() {
			if (const auto locked = state.lock()) {
				std::scoped_lock lock(locked->mutex);
				locked->spans.RemoveGroup(group);
				std::erase_if(locked->groupByName, [&](const auto& byName) { return byName.second == group; });
				locked->groups.erase(group);
			}
		}
	};
	auto owner = std::make_shared<Owner>();
	owner->state = state;
	owner->group = group;
	owner->reservations = std::move(reservations);
	return owner;
}

void GraphHost::NoteFrameTransientUses(uint32_t epoch, uint32_t position, uint32_t lane, std::span<Resource* const> resources) {
	std::shared_ptr<FrameTransients> state;
	{
		std::scoped_lock lock(m_configMutex);
		state = m_frameTransients;
	}
	if (!state) return;
	std::scoped_lock lock(state->mutex);
	if (state->groups.empty()) return;
	std::vector<uint64_t> ids;
	ids.reserve(resources.size());
	for (auto* resource : resources)
		if (resource) ids.push_back(resource->GetGlobalResourceID());
	const auto grown = state->spans.NoteUses(position, ids);
	const auto where = position == alias::FrameSpans::kNoPosition ? std::string("none (the whole frame)") : std::to_string(position);
	for (const auto& growth : grown) {
		const auto& label = state->spans.LabelOf(growth.group);
		auto& entry = state->groups.at(growth.group);
		std::string blocker;
		bool widened = true;
		// The epoch's earlier builds' ranges do not block: this build replaces them, and its own plan places its transients after this.
		for (const auto& reservation : entry.reservations)
			if (!state->arena->Widen(*reservation, ToAliasSpan(growth.to), &blocker, lane)) {
				widened = false;
				break;
			}
		if (!widened) {
			// All or nothing per group: the reservations already widened go back.
			for (const auto& reservation : entry.reservations) state->arena->Widen(*reservation, ToAliasSpan(growth.from));
			state->spans.SetSpan(growth.group, growth.from);
			throw std::runtime_error("Frame transients '" + label + "' are used by epoch " + std::to_string(epoch) + " at position " + where
				+ ", outside their span " + DescribeSpan(growth.from) + ", where '" + blocker
				+ "' holds their bytes: their content would not survive there");
		}
		spdlog::info("GraphHost: frame transients '{}' span {} (was {}): epoch {}'s graph uses them at position {}", label, DescribeSpan(growth.to),
			DescribeSpan(growth.from), epoch, where);
	}
}

std::optional<GraphHost::FrameTransientInfo> GraphHost::FrameTransientOf(std::string_view name) const {
	std::shared_ptr<FrameTransients> state;
	{
		std::scoped_lock lock(m_configMutex);
		state = m_frameTransients;
	}
	if (!state) return std::nullopt;
	std::scoped_lock lock(state->mutex);
	const auto found = state->groupByName.find(std::string(name));
	if (found == state->groupByName.end()) return std::nullopt;
	return FrameTransientInfo{state->spans.LabelOf(found->second), ToAliasSpan(state->spans.SpanOf(found->second))};
}

std::vector<GraphHost::FrameTransientGroup> GraphHost::FrameTransientGroups() const {
	std::shared_ptr<FrameTransients> state;
	{
		std::scoped_lock lock(m_configMutex);
		state = m_frameTransients;
	}
	std::vector<FrameTransientGroup> out;
	if (!state) return out;
	std::scoped_lock lock(state->mutex);
	for (const auto group : state->spans.Groups()) {
		FrameTransientGroup info{state->spans.LabelOf(group), ToAliasSpan(state->spans.SpanOf(group)), {}};
		if (const auto found = state->groups.find(group); found != state->groups.end())
			for (const auto& member : found->second.members)
				if (const auto texture = member.lock()) info.members.push_back(texture->GetName());
		out.push_back(std::move(info));
	}
	return out;
}

void GraphHost::SetEpochCopy(uint32_t epoch, uint32_t copy) {
	if (!m_async) return;
	auto& async = *m_async;
	const auto index = async.IndexOf(epoch);
	if (!async.revisionDriven[index]) throw std::invalid_argument("SetEpochCopy: epoch " + std::to_string(epoch) + " is not revision-driven");
	async.pinnedCopy[index].store(copy % async.copyCount, std::memory_order_release);
}

RunRef GraphHost::EpochCopyRun(uint32_t epoch, uint32_t copy) const {
	for (const auto& [owner, copies] : m_copyLeases)
		if (owner == epoch) {
			if (copy >= copies.size()) break;
			return copies[copy]->Ref();
		}
	throw std::invalid_argument("EpochCopyRun: epoch " + std::to_string(epoch) + " has no published copy " + std::to_string(copy));
}

RunRef GraphHost::EpochStagingRun(uint32_t epoch, uint32_t copy) const {
	for (const auto& [owner, leases] : m_stagingLeases)
		if (owner == epoch) {
			if (copy >= leases.size()) break;
			// Generation 0 and no copy: a stable run (RunRef::Stable) that no lease ever opens.
			return RunRef{leases[copy]->index, 0, RunRef::kNoCopy};
		}
	throw std::invalid_argument("EpochStagingRun: epoch " + std::to_string(epoch) + " has no staging run " + std::to_string(copy));
}

void GraphHost::SetEpochTicketsAhead(uint32_t epoch, uint32_t most) {
	CheckOwner("SetEpochTicketsAhead");
	std::erase_if(m_ticketsAhead, [&](const auto& entry) { return entry.first == epoch; });
	if (most > 1) m_ticketsAhead.emplace_back(epoch, most);
	// An epoch with several tickets ahead runs several times a frame: a repeated epoch for the frame placement.
	ApplyFramePlacementSettings();
}

void GraphHost::SetFramePlacement(FramePlacementSettings settings) {
	CheckOwner("SetFramePlacement");
	{
		std::scoped_lock lock(m_configMutex);
		// The frame placement packs the graphs, so their arena regions go. Frame transients kept apart then free those blocks.
		if (!m_frameAliasArena) m_frameAliasArena = FrameAliasArena::Create();
		m_frameAliasArena->SetSeparateFrameTransients(settings.enabled);
		m_framePlacementSettings = std::move(settings);
	}
	ApplyFramePlacementSettings();
}

GraphHost::FramePlacementSettings GraphHost::GetFramePlacement() const {
	std::scoped_lock lock(m_configMutex);
	return m_framePlacementSettings;
}

void GraphHost::ApplyFramePlacementSettings() {
	FramePlacement::Settings applied;
	{
		std::scoped_lock lock(m_configMutex);
		applied.enabled = m_framePlacementSettings.enabled;
		applied.repeatedEpochs = m_framePlacementSettings.repeatedEpochs;
		applied.beamSearch = m_framePlacementSettings.beamSearch;
		applied.logLifetimes = m_framePlacementSettings.logLifetimes;
		applied.settle = std::chrono::milliseconds(m_framePlacementSettings.settleMilliseconds);
		applied.sizing.headroom = m_framePlacementSettings.heapHeadroom;
		applied.sizing.shrinkFactor = m_framePlacementSettings.heapShrinkFactor;
		applied.memoryLogPeriod = std::chrono::seconds(m_framePlacementSettings.memoryLogSeconds);
		applied.trimAfter = std::chrono::milliseconds(m_framePlacementSettings.heapTrimMilliseconds);
	}
	for (const auto& [epoch, most] : m_ticketsAhead)
		if (std::ranges::find(applied.repeatedEpochs, epoch) == applied.repeatedEpochs.end()) applied.repeatedEpochs.push_back(epoch);
	m_framePlacement->SetSettings(std::move(applied));
}

void GraphHost::LogFramePlacement(bool detailed) const {
	if (m_framePlacement) m_framePlacement->Log(detailed);
}

std::vector<std::shared_ptr<const PlacementSummary>> GraphHost::PlacementSummaries() const {
	return m_framePlacement ? m_framePlacement->Summaries() : std::vector<std::shared_ptr<const PlacementSummary>>{};
}

std::shared_ptr<const FramePlacementVersion> GraphHost::LatestFramePlacement() const {
	return m_framePlacement ? m_framePlacement->Latest() : nullptr;
}

GraphHost::FramePlacementUse GraphHost::FramePlacementInUse(uint32_t epoch) const {
	CheckOwner("FramePlacementInUse");
	for (const auto& instance : m_instances) {
		if (instance.epoch != epoch || !instance.graph) continue;
		const auto use = instance.graph->FramePlacementInUse();
		return {use.planned, use.version, use.placed};
	}
	return {};
}

std::vector<FramePlacementMemory::HeapInfo> GraphHost::FramePlacementHeaps() const {
	return m_framePlacement ? m_framePlacement->Heaps() : std::vector<FramePlacementMemory::HeapInfo>{};
}

void GraphHost::SetExternalEpochs(std::vector<uint32_t> epochs) {
	std::scoped_lock lock(m_configMutex);
	// An epoch whose graph becomes or stops being external is built again; the rest keep theirs.
	for (const auto& instance : m_instances) {
		const bool was = std::ranges::find(m_externalEpochs, instance.epoch) != m_externalEpochs.end();
		const bool is = std::ranges::find(epochs, instance.epoch) != epochs.end();
		if (was != is) {
			RequestRebuild(instance.epoch, is ? "its epoch made external" : "its epoch no longer external");
			instance.configuration->fetch_add(1, std::memory_order_acq_rel);
		}
	}
	m_externalEpochs = std::move(epochs);
}

void GraphHost::SetInlineAtExecution(bool enabled) {
	std::scoped_lock lock(m_configMutex);
	if (m_inlineAtExecution == enabled) return;
	// Every external epoch keeps its graph: its passes recorded at execution are its host's either way.
	for (const auto& instance : m_instances) {
		if (std::ranges::find(m_externalEpochs, instance.epoch) != m_externalEpochs.end()) continue;
		RequestRebuild(instance.epoch, enabled ? "passes recorded at execution made inline" : "passes recorded at execution no longer inline");
		instance.configuration->fetch_add(1, std::memory_order_acq_rel);
	}
	m_inlineAtExecution = enabled;
}

bool GraphHost::InlineAtExecution() const {
	std::scoped_lock lock(m_configMutex);
	return m_inlineAtExecution;
}

void GraphHost::SignalExternalExecutionUnexecuted(const ExternalEpochExecution& execution, size_t a_first) {
	if (!m_async) throw std::logic_error("An external execution without async epochs");
	// Each segment not executed, in its own submission: a later segment may wait for a batch on another queue that waits for an
	// earlier one's signal.
	for (size_t index = a_first == SIZE_MAX ? execution.ExecutedSegments() : a_first; index < execution.segments.size(); ++index) {
		const auto& segment = execution.segments[index];
		auto waits = segment.waits;
		auto signals = segment.signals;
		const auto result = m_async->graphicsQueue.Submit({}, {{waits.data(), static_cast<uint32_t>(waits.size())},
			{signals.data(), static_cast<uint32_t>(signals.size())}});
		if (result != rhi::Result::Ok)
			throw std::runtime_error(std::string("An unexecuted external execution could not be signalled (rhi::Result ") + rhi::ResultName(result) + ")");
	}
}

void GraphHost::StartAsync() {
	if (m_async || m_asyncEpochs.empty()) return;
	// Every epoch built, or failed: a failed epoch starts paused and runs nothing, and the others start without it.
	if (std::ranges::any_of(m_instances, [](const Instance& instance) { return !instance.graph && !instance.failed; })) return;
	auto* firstGraph = AnyGraph();
	if (!firstGraph) return;
	auto async = std::make_unique<Async>();
	// The host's wake, made once: a requester's Notify may reach it whenever, so no build destroys it.
	if (!m_completionWake && m_device.CreateCompletionWait(m_completionWake) != rhi::Result::Ok)
		throw std::runtime_error("Async epochs require wakeable GPU completion waits");
	async->completionWake = m_completionWake.get();
	async->epochs = m_asyncEpochs;
	async->revisionDriven.assign(async->epochs.size(), 0);
	for (uint32_t index = 0; index < async->epochs.size(); ++index) {
		async->revisionDriven[index] = std::ranges::find(m_revisionEpochs, async->epochs[index]) != m_revisionEpochs.end();
		const auto instance = InstanceOf(async->epochs[index]);
		async->instanceOf.push_back(instance);
		async->generation.push_back(m_instances[instance].buildGeneration);
	}
	async->paused.assign(async->epochs.size(), 0);
	async->epochFailed = std::make_unique<std::atomic<bool>[]>(async->epochs.size());
	async->epochErrors.assign(async->epochs.size(), {});
	for (uint32_t index = 0; index < async->epochs.size(); ++index) {
		const auto& instance = m_instances[async->instanceOf[index]];
		// No graph (its build failed): nothing of it until it is built and attached.
		if (!instance.graph) async->paused[index] = 1;
		if (instance.failed) {
			async->epochErrors[index] = instance.failure;
			async->epochFailed[index].store(true, std::memory_order_release);
		}
	}
	async->copyCount = (std::max)(m_desc.framesInFlight, 1u);
	async->seals = &m_seals;
	async->sealsPushed = &m_sealsPushed;
	async->copiesBusyAt = std::make_unique<std::atomic<uint64_t>[]>(async->epochs.size());
	for (uint32_t index = 0; index < async->epochs.size(); ++index) async->copiesBusyAt[index].store(Async::kNoSeal, std::memory_order_relaxed);
	async->copiesBusyLogged.assign(async->epochs.size(), 0);
	async->pinnedCopy = std::make_unique<std::atomic<uint32_t>[]>(async->epochs.size());
	for (uint32_t index = 0; index < async->epochs.size(); ++index) async->pinnedCopy[index].store(RunRef::kNoCopy, std::memory_order_relaxed);
	async->lastPinned.assign(async->epochs.size(), RunRef::kNoCopy);
	async->queues = std::make_unique<Async::TicketQueue[]>(async->epochs.size());
	async->inHand.assign(async->epochs.size(), 0);
	async->mostAhead.assign(async->epochs.size(), 1);
	for (uint32_t index = 0; index < async->epochs.size(); ++index)
		for (const auto& [epoch, most] : m_ticketsAhead)
			if (epoch == async->epochs[index]) async->mostAhead[index] = (std::max)(most, 1u);
	async->runsThisFrame.assign(async->epochs.size(), 0);
	async->runsLastFrame.assign(async->epochs.size(), 0);
	async->lastSeal.assign(async->epochs.size(), 0);
	async->liveRecordings.assign(async->epochs.size(), {});
	async->stagedGraph.assign(async->epochs.size(), nullptr);
	async->stagedGeneration.assign(async->epochs.size(), 0);
	async->stagedRecordings.assign(async->epochs.size(), {});
	async->liveKept.assign(async->epochs.size(), {});
	async->outstanding.assign(async->epochs.size(), {});
	// The synchronous frames' runs still in flight are the host thread's to retire now, like its own. The sync path's pacing
	// starts afresh when async epochs stop.
	m_syncLeases.clear();
	async->hostFrame = m_frameNumber;
	async->submissionsSeen = m_submissions;
	// The queues' layout is every graph's: the primary queues first, on the host's timelines.
	auto& queues = firstGraph->GetQueueRegistry();
	const auto graphics = queues.FindGraphicsSlot();
	async->graphicsSlot = static_cast<uint32_t>(ToUnderlying(graphics));
	async->graphicsQueue = queues.GetQueue(graphics);
	async->graphicsFence = queues.GetFence(graphics).GetHandle();
	// The upload lists come from host-wide pools: each run lease requests its lists when its ticket is prepared, and recycles
	// them when it retires. The pools share their allocators between the lists, so a few command pools serve every run.
	async->uploadPool = std::make_shared<CommandListPool>(m_device, rhi::QueueKind::Graphics);
	// An external epoch's uploads secondaries: an epoch may become external at any of its rebuilds.
	async->uploadSecondaryPool = std::make_shared<CommandListPool>(m_device, rhi::QueueKind::Graphics, true);
	// Queue-scoped uploads: one list per compute or copy queue on the graphics queue's device (a copy needs no more).
	async->queueUploadKinds.assign(queues.SlotCount(), std::nullopt);
	for (size_t index = 0; index < queues.SlotCount(); ++index) {
		const auto queueSlot = static_cast<QueueSlotIndex>(static_cast<uint8_t>(index));
		const auto kind = queues.GetKind(queueSlot);
		if (queueSlot == graphics || (kind != QueueKind::Compute && kind != QueueKind::Copy)
			|| queues.GetBackendInstance(queueSlot) != queues.GetBackendInstance(graphics) || !queues.GetQueue(queueSlot))
			continue;
		const auto queueKind = kind == QueueKind::Compute ? rhi::QueueKind::Compute : rhi::QueueKind::Copy;
		async->queueUploadKinds[index] = queueKind;
		auto& pool = async->queueUploadPools[static_cast<size_t>(queueKind)];
		if (!pool) pool = std::make_shared<CommandListPool>(m_device, queueKind);
	}
	async->queueLists.resize(queues.SlotCount());
	// Staged uploads go straight into each submission's upload list (RenderGraph::RecordTakenUploads).
	if (auto* uploads = firstGraph->GetUploadService()) uploads->SetStagedUploadsRecordedDirectly(true);
	auto* state = async.get();
	state->thread = std::thread([this, state] {
		const auto graphOf = [&](uint32_t epochIndex) -> RenderGraph& { return *m_instances[state->instanceOf[epochIndex]].graph; };
		RenderGraph::CompletedTimelineValues completedValues;
		try {
			for (;;) {
				const auto seen = state->completionWake->WakeVersion();
				bool stop = false;
				// Messages, in the order the render thread posted them.
				for (auto read = state->read.load(std::memory_order_relaxed); read != state->written.load(std::memory_order_acquire); ++read) {
					auto message = std::move(state->mailbox[read % Async::kMailbox]);
					state->read.store(read + 1, std::memory_order_release);
					switch (message.kind) {
					case Async::Message::Kind::Submitted: {
						auto& graph = graphOf(message.epochIndex);
						auto& lease = *m_leaseTable[RenderGraph::PersistentTicketRun(*message.ticket).lease].load(std::memory_order_acquire);
						state->uncertainUploadOwner = std::move(message.keepAlive);
						state->uncertainResourceOwner = std::move(message.resourceOwner);
						auto completion = graph.CompletePersistentTicket(*message.ticket, message.submission);
						if (message.uploadSignal.first != UINT32_MAX) {
							auto found = std::ranges::find(completion, message.uploadSignal.first, &std::pair<uint32_t, uint64_t>::first);
							if (found == completion.end()) completion.push_back(message.uploadSignal);
							else found->second = (std::max)(found->second, message.uploadSignal.second);
						}
						lease.points.clear();
						for (const auto& [queue, value] : completion)
							lease.points.emplace_back(graph.GetQueueRegistry().GetFenceOwner(static_cast<QueueSlotIndex>(static_cast<uint8_t>(queue))), value);
						lease.hostFrame = RenderGraph::PersistentTicketHostFrame(*message.ticket);
						lease.submission = message.number;
						lease.seal = message.seal;
						state->submissionsSeen = (std::max)(state->submissionsSeen, message.number);
						if (state->outstanding[message.epochIndex] == message.ticket) state->outstanding[message.epochIndex].reset();
						lease.keepAlive = std::move(state->uncertainUploadOwner);
						lease.resourceOwner = std::move(state->uncertainResourceOwner);
						{
							std::scoped_lock lock(m_leaseMutex);
							lease.state = RunLease::State::Submitted;
							m_submittedLeases.push_back(&lease);
						}
						{
							const auto index = message.epochIndex;
							if (state->inHand[index]) --state->inHand[index];
							// The epoch's runs per frame (between seals), which set how many tickets it keeps prepared (Depth).
							if (message.seal != state->lastSeal[index]) {
								state->runsLastFrame[index] = state->runsThisFrame[index];
								state->runsThisFrame[index] = 0;
								state->lastSeal[index] = message.seal;
							}
							++state->runsThisFrame[index];
						}
						// Never freed on this thread: a driver free can wait for GPU work that waits for a signal this thread prepares.
						m_desc.runtime->DestroyDetached(std::move(message.retired), "a submission's deferred deletions");
						break;
					}
					case Async::Message::Kind::Discard: {
						auto& lease = *m_leaseTable[RenderGraph::PersistentTicketRun(*message.ticket).lease].load(std::memory_order_acquire);
						if (state->outstanding[message.epochIndex] == message.ticket) state->outstanding[message.epochIndex].reset();
						(void)graphOf(message.epochIndex).CompletePersistentTicket(*message.ticket, {});
						// A reprepare is a stale ticket's: its replacement is prepared here, in the same step, into the run lease whose
						// latch regions the render thread wrote and whose upload lists it records next. Marking the epoch as needing a
						// ticket instead would let the proactive pass below prepare one into a new lease.
						// A failed epoch prepares nothing: the render thread waiting for it sees the failure (WaitTicket).
						if (message.reprepare && !state->epochFailed[message.epochIndex].load(std::memory_order_relaxed)) {
							// The replacement takes the discarded ticket's place in hand.
							PrepareTicket(*state, message.epochIndex, &lease);
							// The tickets queued behind it were prepared before the same change, most likely. The ones no longer
							// current go now, before the render thread reaches them, and others are prepared in their place.
							DropStaleTickets(*state, message.epochIndex);
						} else {
							// Never submitted: nothing of it reached the GPU.
							ReleaseLease(lease, state);
							if (state->inHand[message.epochIndex]) --state->inHand[message.epochIndex];
						}
						break;
					}
					case Async::Message::Kind::Detach: {
						// The instances' epochs: their unsubmitted tickets abandoned, their recording requests dropped, and nothing prepared
						// for them until the instances are attached again. Their submitted work completes as before (its leases keep their points).
						const auto detaching = [&](size_t instance) {
							return std::ranges::any_of(message.instances, [&](const auto& entry) { return entry.first == instance; });
						};
						for (uint32_t index = 0; index < state->epochs.size(); ++index) {
							if (!detaching(state->instanceOf[index])) continue;
							state->paused[index] = 1;
							// The render thread holds none of them: it waits for this detach.
							for (const auto& ticket : state->TakeAll(index)) {
								(void)graphOf(index).CompletePersistentTicket(*ticket, {});
								ReleaseLease(*m_leaseTable[RenderGraph::PersistentTicketRun(*ticket).lease].load(std::memory_order_acquire), state);
							}
							state->inHand[index] = 0;
							state->outstanding[index].reset();
							state->liveRecordings[index].clear();
							state->liveKept[index].clear();
							const auto dropped = std::make_exception_ptr(EpochRecordingDropped(state->generation[index]));
							std::erase_if(state->pendingRequests, [&](const std::unique_ptr<Async::RecordingRequest>& request) {
								if (request->epochIndex != index) return false;
								// A request for the staged graph survives: the attach that follows makes it the epoch's graph.
								if (state->RecordsStaged(index, request->generation)) return false;
								if (!m_ending && request->done) request->done(nullptr, dropped);
								return true;
							});
						}
						state->detached.fetch_add(1, std::memory_order_acq_rel);
						state->detached.notify_all();
						{
							BT_ZONE_SCOPE("ORG.Host.HoldForBuild.Worker");
							for (auto seen = state->released.load(std::memory_order_acquire); seen < message.hold;
								seen = state->released.load(std::memory_order_acquire))
								state->released.wait(seen, std::memory_order_acquire);
						}
						break;
					}
					case Async::Message::Kind::Attach:
						for (const auto& [instance, generation] : message.instances)
							for (uint32_t index = 0; index < state->epochs.size(); ++index) {
								if (state->instanceOf[index] != instance) continue;
								// A new build's failure was cleared while this thread held still (RebuildAsyncInstances).
								state->paused[index] = 0;
								state->generation[index] = generation;
								state->inHand[index] = 0;
							}
						break;
					case Async::Message::Kind::Stop:
						stop = true;
						break;
					}
				}
				if (stop) break;
				// Recording requests: taken in posting order, each recording its published copies. One for an epoch that
				// is not a revision-driven async epoch completes with that error now.
				for (auto& node : m_recordingQueue->Take()) {
					const auto found = std::ranges::find(state->epochs, node->epoch);
					const auto epochIndex = static_cast<uint32_t>(found - state->epochs.begin());
					// Only a revision-driven epoch publishes recordings: a live epoch records each run afresh.
					if (found == state->epochs.end() || !state->revisionDriven[epochIndex]) {
						if (node->done)
							node->done(nullptr, std::make_exception_ptr(std::invalid_argument("RequestEpochRecording: epoch " + std::to_string(node->epoch) +
								" is not a revision-driven async epoch")));
						continue;
					}
					const bool staged = state->RecordsStaged(epochIndex, node->generation);
					// Its graph is being built again (the request is for the old one), it names a build that is neither attached nor staged,
					// or a staged graph froze the attached one: dropped.
					if (!staged && (state->paused[epochIndex] || (node->generation && node->generation != state->generation[epochIndex])
						|| state->stagedGraph[epochIndex])) {
						if (node->done) node->done(nullptr, std::make_exception_ptr(EpochRecordingDropped(state->generation[epochIndex])));
						continue;
					}
					if (!staged && state->epochFailed[epochIndex].load(std::memory_order_acquire)) {
						// Its graph failed: nothing is recorded on it until it is built again.
						if (node->done)
							node->done(nullptr, std::make_exception_ptr(EpochFailedError(node->epoch, state->epochErrors[epochIndex])));
						continue;
					}
					auto request = std::make_unique<Async::RecordingRequest>();
					request->epochIndex = epochIndex;
					request->copy = node->copy;
					request->generation = node->generation;
					request->hostData = std::move(node->hostData);
					request->done = std::move(node->done);
					state->pendingRequests.push_back(std::move(request));
				}
				// Each queue timeline's completed value is read once for every graph (they share the timelines). A graph none of
				// whose work the GPU has reached returns at once.
				completedValues.Clear();
				for (uint32_t index = 0; index < state->epochs.size(); ++index)
					if (!state->paused[index]) graphOf(index).RetirePersistentExecutions(&completedValues);
				// Every run whose GPU work is done retires: its statistics are read, and its lists, arena regions, upload staging and
				// owners are freed, even if no later run takes its lease again.
				(void)RetireReachedLeases(state);
				// The next tickets: an epoch keeps Async::Depth of them in hand. That is one, prepared once the last one was submitted,
				// or up to its runs last frame (SetEpochTicketsAhead).
				// - First, a ticket for every epoch with none in hand: the render thread may be about to wait for it.
				// - Otherwise, one more ticket for one epoch below its depth, the epochs taking turns. Then the messages are read
				//   again, so a submission's completion or a discard is never held up behind a queue being filled.
				bool prepared = false;
				const auto epochCount = static_cast<uint32_t>(state->epochs.size());
				for (uint32_t round = 0; round < 2 && !prepared; ++round)
				for (uint32_t turn = 0; turn < epochCount; ++turn) {
					const uint32_t index = round == 0 ? turn : (state->nextAhead + turn) % epochCount;
					if (state->paused[index] || state->epochFailed[index].load(std::memory_order_relaxed)) continue;
					const bool wanted = round == 0 ? state->inHand[index] == 0
						: (state->inHand[index] != 0 && state->inHand[index] < state->Depth(index));
					if (!wanted) continue;
					// Copies busy with unsealed runs wait for the next seal (its SealSubmitted wakes this thread), or for one of their runs to
					// complete (this thread sleeps on the submitted runs' completions too).
					if (const auto busy = state->copiesBusyAt[index].load(std::memory_order_acquire); busy != Async::kNoSeal) {
						if (busy == m_seals.load(std::memory_order_acquire)) {
							bool done = false;
							for (const auto* copy : CopyLeases(state->epochs[index]))
								done = done || copy->state == RunLease::State::Free;
							if (!done) continue;
						}
						state->copiesBusyAt[index].store(Async::kNoSeal, std::memory_order_release);
					}
					PrepareTicket(*state, index, nullptr);
					prepared = true;
					// A message waits (an attach holding for this thread, say): read it before the next epoch's first ticket. The
					// epochs still without one come first again on the next pass.
					if (round == 0 && state->read.load(std::memory_order_relaxed) != state->written.load(std::memory_order_acquire)) break;
					if (round == 1) {
						state->nextAhead = (index + 1) % epochCount;
						break;
					}
				}
				// Then the recording requests: the frames' tickets come first, so recording ahead never delays a submission.
				const bool requestsAdvanced = AdvanceRecordingRequests(*state);
				if (!prepared && !requestsAdvanced && !m_recordingQueue->head.load(std::memory_order_acquire)
					&& state->read.load(std::memory_order_relaxed) == state->written.load(std::memory_order_acquire)) {
					std::array<rhi::TimelinePoint, rhi::CompletionWait::MaxTimelines> heads{};
					uint32_t count = 0;
					bool retireAgain = false;
					{
						std::scoped_lock lock(m_leaseMutex);
						for (const auto* lease : m_submittedLeases) {
							// An external run's point is its client's: it is retired at the next wake-up, never slept on.
							if (lease->external) continue;
							bool incomplete = false;
							for (const auto& [timeline, value] : lease->points) {
								const auto reached = (*timeline)->GetCompletedValue();
								if (reached == UINT64_MAX) throw std::runtime_error("GPU completion timeline failed");
								if (reached >= value) continue;
								incomplete = true;
								const auto handle = (*timeline)->GetHandle();
								uint32_t i = 0;
								for (; i < count; ++i) if (heads[i].t.index == handle.index && heads[i].t.generation == handle.generation) break;
								if (i < count) heads[i].value = (std::min)(heads[i].value, value);
								else {
									if (count == heads.size()) throw std::runtime_error("Too many GPU completion timelines");
									heads[count++] = {handle, value};
								}
							}
							// Completion may race the retirement pass above. Do not go to sleep with a run whose final signal already
							// arrived.
							if (!incomplete) retireAgain = true;
						}
					}
					if (retireAgain) continue;
					// Posters skip the wake while this sleep has GPU completions to end it; a message written before this store
					// is seen by the look below, and one written after it sees the store.
					state->wakesOnGpu.store(count != 0, std::memory_order_seq_cst);
					if (state->read.load(std::memory_order_relaxed) != state->written.load(std::memory_order_seq_cst)) {
						state->wakesOnGpu.store(false, std::memory_order_seq_cst);
						continue;
					}
					state->sleepingAt.store(seen + 1, std::memory_order_release);
					const auto waited = state->completionWake->Wait({heads.data(), count}, seen);
					state->sleepingAt.store(0, std::memory_order_release);
					state->wakesOnGpu.store(false, std::memory_order_seq_cst);
					if (waited != rhi::Result::Ok)
						throw std::runtime_error("GPU completion wait failed");
				}
			}
		} catch (const std::exception& e) {
			state->error = e.what();
			spdlog::error("GraphHost: the host's thread failed: {}", state->error);
			state->failed.store(true, std::memory_order_release);
		} catch (...) {
			state->error = "unknown exception";
			spdlog::error("GraphHost: the host's thread failed: {}", state->error);
			state->failed.store(true, std::memory_order_release);
		}
		// Wake a render thread waiting for a ticket or a detach, so it sees the failure.
		state->ticketWake.fetch_add(1, std::memory_order_release);
		state->ticketWake.notify_all();
		state->detached.fetch_add(1, std::memory_order_acq_rel);
		state->detached.notify_all();
	});
	// From here the host's thread captures backings while the submitting thread runs.
	BufferBase::RegisterConcurrentBackingCapture();
	m_async = std::move(async);
	// Requests from now on (and any queued across the build that started this) are this thread's.
	m_recordingQueue->wake.store(m_completionWake.get(), std::memory_order_release);
	m_recordingQueue->accepting.store(true, std::memory_order_release);
	(void)m_completionWake->Notify();
}

void GraphHost::HoldAsync(std::span<const size_t> detach, const std::function<void()>& work) {
	// The work changes graphs that tickets still on the submission thread were prepared on: they are submitted first.
	DrainSubmissions("a hold of the host's thread (a build)");
	auto& async = *m_async;
	async.ThrowIfFailed();
	// The host's thread lets go of the instances to detach and holds still while the work runs: whatever it was given before
	// (submissions' completions) it handles first, on the graphs as built. The other epochs keep their tickets.
	const auto before = async.detached.load(std::memory_order_acquire);
	const auto hold = ++async.holds;
	async.holding = true;
	struct Release {
		Async& async;
		uint64_t hold;
		~Release() {
			async.holding = false;
			async.released.store(hold, std::memory_order_release);
			async.released.notify_all();
		}
	} release{async, hold};
	m_heldRun.reset();
	Async::Message message;
	message.kind = Async::Message::Kind::Detach;
	message.hold = hold;
	for (const auto index : detach) message.instances.emplace_back(index, 0);
	if (!async.Post(message)) async.ThrowIfFailed();
	{
		BT_ZONE_SCOPE("ORG.Host.WaitHold");
		for (auto seen = async.detached.load(std::memory_order_acquire); seen == before; seen = async.detached.load(std::memory_order_acquire)) {
			async.ThrowIfFailed();
			async.detached.wait(seen, std::memory_order_acquire);
		}
	}
	async.ThrowIfFailed();
	work();
}

void GraphHost::RebuildAsyncInstances(std::span<const size_t> indices) {
	BT_ZONE_SCOPE("ORG.Host.RebuildAsyncInstances");
	ReplaceAsyncInstances(indices, [&](size_t index) { return BuildInstance(index); });
}

void GraphHost::ReplaceAsyncInstances(std::span<const size_t> indices, const std::function<bool(size_t)>& replace) {
	if (indices.empty()) return;
	Async::Message attach;
	attach.kind = Async::Message::Kind::Attach;
	HoldAsync(indices, [&] {
		auto& async = *m_async;
		for (const auto index : indices) {
			if (replace(index)) {
				// Built: its async epoch's failure goes with the old graph. Written while the host's thread holds still.
				for (uint32_t i = 0; i < async.epochs.size(); ++i) {
					if (async.instanceOf[i] != index) continue;
					async.epochFailed[i].store(false, std::memory_order_release);
					async.epochErrors[i].clear();
				}
			}
			// A failed build with no graph to keep stays detached: its epoch runs nothing until a build of it succeeds.
			if (m_instances[index].graph) attach.instances.emplace_back(index, m_instances[index].buildGeneration);
		}
	});
	if (!m_async->Post(attach)) m_async->ThrowIfFailed();
}

std::shared_ptr<runtime::ResourceCleanupQueue> GraphHost::ResourceCleanup() const {
	return DescriptorHeapManager::GetInstance().GetResourceCleanupQueue();
}

bool GraphHost::ReleaseEpochTicket(uint32_t epoch) {
	if (!m_async) return false;
	auto& async = *m_async;
	const auto found = std::ranges::find(async.epochs, epoch);
	if (found == async.epochs.end()) return false;
	const auto index = static_cast<uint32_t>(found - async.epochs.begin());
	auto ticket = async.TakeReady(index);
	if (!ticket) return false;
	// The host's thread counts the ticket in hand until it reads the discard. A ticket the mailbox has no room for goes back to
	// the front of the queue.
	Async::Message discard;
	discard.kind = Async::Message::Kind::Discard;
	discard.epochIndex = index;
	discard.ticket = ticket;
	if (async.TryPost(discard)) return true;
	async.PushReady(index, std::move(ticket), true);
	return false;
}

bool GraphHost::RequestEpochRecording(uint32_t epoch, std::shared_ptr<const IHostExecutionData> hostData, EpochRecordingCallback done, uint32_t copy,
	uint64_t generation) {
	// The host's queue alone (never m_async, which a build moves out from under any other thread): the host's thread validates the
	// epoch when it takes it, and a build drops what it did not take (StopAsync, an epoch's Detach), each completed.
	auto& queue = *m_recordingQueue;
	if (!done || !queue.accepting.load(std::memory_order_acquire)) return false;
	auto* node = new RecordingQueue::Request{epoch, std::move(hostData), std::move(done), copy, generation, nullptr};
	auto* head = queue.head.load(std::memory_order_relaxed);
	do node->next = head;
	while (!queue.head.compare_exchange_weak(head, node, std::memory_order_release, std::memory_order_relaxed));
	if (auto* wake = queue.wake.load(std::memory_order_acquire)) {
		static std::atomic<bool> logged{false};
		if (const auto result = wake->Notify(); result != rhi::Result::Ok && !logged.exchange(true))
			spdlog::error("GraphHost: waking the host's thread for a recording request failed ({})", static_cast<int>(result));
	}
	return true;
}

bool GraphHost::AdvanceRecordingRequests(Async& state) {
	bool advanced = false;
	for (auto it = state.pendingRequests.begin(); it != state.pendingRequests.end();) {
		// A message from the render thread waits (an attach holding for this thread, a submission): it is read before the next copy, so
		// a hold never waits behind every pending request's copy (a load posts one per revision-driven epoch).
		if (advanced && state.read.load(std::memory_order_relaxed) != state.written.load(std::memory_order_acquire)) break;
		auto& request = **it;
		std::exception_ptr error;
		const bool staged = state.RecordsStaged(request.epochIndex, request.generation);
		try {
			if (!staged && state.paused[request.epochIndex]) {
				++it;
				continue;
			}
			// For a build that is neither attached nor staged now (a staged graph dropped), or for an attached graph a staged one froze:
			// dropped.
			if (!staged && ((request.generation && request.generation != state.generation[request.epochIndex]) || state.stagedGraph[request.epochIndex]))
				throw EpochRecordingDropped(state.generation[request.epochIndex]);
			// Its graph failed since the request was taken: it completes with that failure.
			if (!staged && state.epochFailed[request.epochIndex].load(std::memory_order_relaxed))
				throw EpochFailedError(state.epochs[request.epochIndex], state.epochErrors[request.epochIndex]);
			auto& graph = staged ? *state.stagedGraph[request.epochIndex] : *m_instances[state.instanceOf[request.epochIndex]].graph;
			if (!request.result) {
				request.result = std::make_shared<EpochRecording>();
				request.result->epoch = state.epochs[request.epochIndex];
				request.result->copies.resize(state.copyCount);
				request.result->revision = request.hostData;
				request.result->buildGeneration = staged ? state.stagedGeneration[request.epochIndex] : state.generation[request.epochIndex];
				request.recorded.assign(state.copyCount, 0);
				// One copy only: the others count as done and stay empty.
				if (request.copy != RunRef::kNoCopy) {
					std::fill(request.recorded.begin(), request.recorded.end(), uint8_t{1});
					request.recorded[request.copy % state.copyCount] = 0;
				}
			}
			auto& copies = CopyLeases(request.result->epoch);
			for (uint32_t copy = 0; copy < request.recorded.size(); ++copy) {
				if (request.recorded[copy]) continue;
				// A copy whose last run is in flight is recorded once that completes (the loop wakes at completions), so the host's
				// thread never waits on the GPU here and the epochs' tickets are not held up behind it. A copy the epoch's
				// admission-only ticket holds is recorded too: the ticket recorded nothing, and it cannot be submitted without this.
				auto& lease = *copies[copy];
				if (lease.state == RunLease::State::Submitted) {
					if (!PointsReached(lease.points)) continue;
					RetireLease(lease, &state);
				}
				BT_ZONE_SCOPE("ORG.Host.RecordForRevision");
				runtime::ScopedActiveGraphServices services(graph.GetUploadService(), graph.GetDescriptorService());
				std::shared_ptr<const RenderGraph::PersistentRecording> recording;
				{
					Async::Preparation preparation;
					const auto version = state.backingVersion.load(std::memory_order_acquire);
					auto ticket = graph.PreparePersistentTicket(m_device, request.result->epoch, lease.Ref(), state.hostFrame, version, request.hostData);
					recording = RenderGraph::PersistentTicketRecording(*ticket);
					// Recorded only: its admission is abandoned, never submitted.
					(void)graph.CompletePersistentTicket(*ticket, {});
				}
				if (!recording || !RenderGraph::PersistentRecordingReplayable(*recording))
					throw std::logic_error("A revision-driven epoch's recording is not replayable (a pass with lifecycle effects, or a late-bound slot)");
				request.result->copies[copy] = std::move(recording);
				request.recorded[copy] = 1;
				state.revisionCopiesRecorded.fetch_add(1, std::memory_order_relaxed);
				advanced = true;
				// One copy per pass: the epochs' tickets come first between copies.
				break;
			}
		} catch (...) {
			error = std::current_exception();
		}
		const bool complete = !error && std::ranges::all_of(request.recorded, [](uint8_t a_done) { return a_done != 0; });
		if (!error && !complete) {
			++it;
			continue;
		}
		auto done = std::move(request.done);
		auto result = error ? nullptr : std::shared_ptr<const EpochRecording>(std::move(request.result));
		const auto epochIndex = request.epochIndex;
		it = state.pendingRequests.erase(it);
		if (!error && staged) {
			// The staged graph's: its tickets take it once the attach makes it the epoch's graph.
			state.revisionRecordings.fetch_add(1, std::memory_order_relaxed);
			state.stagedRecordings[epochIndex].push_back(result);
		} else if (!error) {
			state.revisionRecordings.fetch_add(1, std::memory_order_relaxed);
			// Admitted from now on: by every ticket prepared next, and by the one waiting for submission, before the caller can
			// publish it to the submitting thread.
			state.liveRecordings[epochIndex].push_back(result);
			if (const auto& ticket = state.outstanding[epochIndex]) {
				try {
					AddRevisionCandidate(state, epochIndex, *ticket, *result);
				} catch (...) {
					error = std::current_exception();
					result = nullptr;
				}
			}
		}
		done(std::move(result), error);
		advanced = true;
	}
	return advanced;
}

GraphHost::RunLease* GraphHost::TakeCopyLease(Async& state, uint32_t epochIndex) {
	auto& copies = CopyLeases(state.epochs[epochIndex]);
	const uint64_t seals = m_seals.load(std::memory_order_acquire);
	// An idle copy first, the least recently submitted. Else the least recently submitted copy whose last run may be waited for:
	// done, or behind a seal (never a run that may still be in the host's stream). A host that never sealed hands its runs to the
	// device itself, so then every run may be waited for.
	RunLease* idle = nullptr;
	RunLease* waitable = nullptr;
	// A pinned epoch (SetEpochCopy) takes the copy its caller named, under the same rules: idle, or its last run waitable.
	const uint32_t pinned = state.pinnedCopy[epochIndex].load(std::memory_order_acquire);
	for (auto* copy : copies) {
		if (pinned < copies.size() && copy != copies[pinned]) continue;
		if (copy->state == RunLease::State::Open) continue;
		if (copy->state == RunLease::State::Submitted && PointsReached(copy->points)) RetireLease(*copy, &state);
		if (copy->state == RunLease::State::Free) {
			if (!idle || copy->submission < idle->submission) idle = copy;
			continue;
		}
		if ((copy->seal < seals || seals == 0) && (!waitable || copy->submission < waitable->submission)) waitable = copy;
	}
	RunLease* taken = idle ? idle : waitable;
	if (!taken) {
		// Every copy holds a run that is neither sealed nor done: the ticket waits for the next seal or for one of those runs to
		// complete, and the render thread skips the epoch's runs meanwhile.
		state.copiesBusy.fetch_add(1, std::memory_order_relaxed);
		if (!state.copiesBusyLogged[epochIndex]) {
			state.copiesBusyLogged[epochIndex] = 1;
			spdlog::warn("GraphHost: epoch {} has no published copy to submit: each of its {} copies holds a run neither sealed nor done (seals "
				"now {}). Its runs are skipped (EpochCopiesBusyError) until the next SealSubmitted or until one of those runs completes",
				state.epochs[epochIndex], copies.size(), seals);
		}
		state.copiesBusyAt[epochIndex].store(seals, std::memory_order_release);
		state.anyCopiesBusy.store(true, std::memory_order_seq_cst);
		state.ticketWake.fetch_add(1, std::memory_order_release);
		state.ticketWake.notify_all();
		return nullptr;
	}
	if (taken->state == RunLease::State::Submitted) {
		// Sealed: on the device, so the wait ends. Its lists, statistics range and pinned regions are reused next.
		BT_ZONE_SCOPE("ORG.Host.WaitCopy");
		state.copyWaits.fetch_add(1, std::memory_order_relaxed);
		for (const auto& [timeline, value] : taken->points) (void)(*timeline)->HostWait(value);
		RetireLease(*taken, &state);
	}
	std::scoped_lock lock(m_leaseMutex);
	if (++taken->generation == 0) taken->generation = 1;
	taken->state = RunLease::State::Open;
	m_leasesPeak = (std::max)(m_leasesPeak, ++m_leasesInFlight);
	return taken;
}

GraphHost::RunLease* GraphHost::TakeLiveCopy(Async& state, uint32_t epochIndex) {
	const uint32_t epoch = state.epochs[epochIndex];
	auto found = std::ranges::find(m_liveCopyLeases, epoch, &std::pair<uint32_t, std::vector<RunLease*>>::first);
	if (found == m_liveCopyLeases.end()) found = m_liveCopyLeases.insert(m_liveCopyLeases.end(), std::pair<uint32_t, std::vector<RunLease*>>{epoch, {}});
	auto& copies = found->second;
	const auto& kept = state.liveKept[epochIndex];
	// An idle copy with a recording of the epoch's current graph first, so the run can bind it. Then any idle copy. Among
	// equals, the least recently submitted one.
	RunLease* idleKept = nullptr;
	RunLease* idle = nullptr;
	for (auto* copy : copies) {
		if (copy->state == RunLease::State::Open) continue;
		if (copy->state == RunLease::State::Submitted && PointsReached(copy->points)) RetireLease(*copy, &state);
		if (copy->state != RunLease::State::Free) continue;
		const bool hasRecording = copy->copy < kept.size() && kept[copy->copy].recording && kept[copy->copy].generation == state.generation[epochIndex];
		auto& pick = hasRecording ? idleKept : idle;
		if (!pick || copy->submission < pick->submission) pick = copy;
	}
	RunLease* taken = idleKept ? idleKept : idle;
	std::scoped_lock lock(m_leaseMutex);
	if (!taken) {
		// Every copy is in use: a new one, until the epoch has the most it may make. The caller then takes a lease of its own.
		if (copies.size() >= (std::max)(m_desc.liveCopiesMost, 1u) || m_leaseStore.size() >= Runtime::kMaxSlots) return nullptr;
		auto made = std::make_unique<RunLease>();
		made->index = static_cast<uint32_t>(m_leaseStore.size());
		made->copy = static_cast<uint32_t>(copies.size());
		taken = made.get();
		copies.push_back(taken);
		m_leaseTable[made->index].store(taken, std::memory_order_release);
		m_leaseStore.push_back(std::move(made));
		if (const auto& stats = m_desc.runtime->Statistics()) stats->EnsureFrameCapacity(taken->index + 1);
	}
	if (++taken->generation == 0) taken->generation = 1;
	taken->state = RunLease::State::Open;
	m_leasesPeak = (std::max)(m_leasesPeak, ++m_leasesInFlight);
	return taken;
}

void GraphHost::PrepareTicket(Async& state, uint32_t epochIndex, RunLease* reprepare) {
	BT_ZONE_SCOPE("ORG.Host.PrepareTicket.Worker");
	auto& graph = *m_instances[state.instanceOf[epochIndex]].graph;
	// A stale ticket's reprepare takes the lease it held: its latch regions were written for it. Otherwise a new run lease, or a
	// published copy's for a revision-driven epoch, or none: then nothing is prepared until a seal or a completion frees a copy
	// (EpochCopiesBusyError meanwhile).
	RunLease* lease = reprepare;
	const bool keepsLive = !state.revisionDriven[epochIndex] && m_desc.reuseRecordings;
	if (!lease && state.revisionDriven[epochIndex]) {
		lease = TakeCopyLease(state, epochIndex);
	} else if (!lease && keepsLive) {
		lease = TakeLiveCopy(state, epochIndex);
		if (!lease) {
			state.liveSpilled.fetch_add(1, std::memory_order_relaxed);
			lease = &OpenLease(&state);
		}
	} else if (!lease) {
		lease = &OpenLease(&state);
	}
	if (!lease) return;
	runtime::ScopedActiveGraphServices services(graph.GetUploadService(), graph.GetDescriptorService());
	// The render thread records this ticket's uploads into an open list: resetting and beginning it is several microseconds of
	// driver work that belongs here. Begin takes the list's allocator claim, which the render thread gives up after closing the
	// list (SubmitTicket). A new lease's lists come from the pools; a reprepare's are begun again.
	// A ticket prepared behind others of its epoch leaves its lists to the render thread (SubmitTicket begins them). An open list
	// holds its allocator, and on Vulkan each allocator is a command pool with driver memory of its own, so a queue of tickets
	// with open lists would make a pool for each.
	lease->queueUploads.resize(state.queueUploadKinds.size());
	// Every epoch's lists, external ones' too, are begun by FinishTicket, and only for a ticket with uploads (most have none):
	// a list begun here would have to be closed on the render thread for nothing.
	// The descriptors are released here; the objects behind them and the deferred deletions are freed on the runtime's deleter
	// thread. A driver free can block until the GPU is done with the memory. The GPU may be waiting for a signal of work this
	// thread prepares, so a free here could wait forever.
	m_desc.runtime->RetireDetached();
	std::shared_ptr<Async::Ticket> ticket;
	// The epoch's own preparation: its passes' Prepare and recording, the bindless views they resolve. A failure here is the
	// epoch's alone (FailAsyncEpoch); the host's thread and the other epochs go on.
	std::string error;
	try {
		if (state.revisionDriven[epochIndex]) {
			// Nothing from live state: the ticket admits the revisions' recordings' copies for its run, the ones still held.
			ticket = graph.PrepareRevisionTicket(state.epochs[epochIndex], lease->Ref(), state.hostFrame++);
			auto& live = state.liveRecordings[epochIndex];
			std::erase_if(live, [](const auto& a_recording) { return a_recording.expired(); });
			for (const auto& weak : live)
				if (const auto recording = weak.lock()) AddRevisionCandidate(state, epochIndex, *ticket, *recording);
		} else {
			const auto version = state.backingVersion.load(std::memory_order_acquire);
			// A kept recording of an older build or older versions never binds again, yet holds what it bound. Every copy's goes
			// now, not only when its copy is next taken: a copy may not be taken again for a long time. A run still in flight
			// holds its own reference.
			// A kept recording that bound a backing that moved since goes too (relocation::Service): it would keep the old memory.
			if (keepsLive)
				for (auto& entry : state.liveKept[epochIndex])
					if (entry.recording && (entry.generation != state.generation[epochIndex] || entry.version != version
						|| !RenderGraph::PersistentRecordingBackingsCurrent(*entry.recording))) entry = {};
			// A live epoch's kept copy.
			const bool liveCopy = keepsLive && lease->copy != RunRef::kNoCopy;
			if (liveCopy && !reprepare) {
				// The copy's regions still hold its last run's values. A new lease's would be zeros, and a client whose commit does
				// not write its block this run relies on that.
				LatchBlock::ZeroStableRun(lease->index);
				auto& kept = state.liveKept[epochIndex];
				if (lease->copy < kept.size() && kept[lease->copy].recording) {
					auto& entry = kept[lease->copy];
					// The kept recording, bound now. It is checked again at submission, after the commit, like any ticket.
					if (entry.generation == state.generation[epochIndex] && entry.version == version) {
						// The host's loop retires every graph's finished executions on each pass: no retirement here.
						auto reuse = graph.PrepareRevisionTicket(state.epochs[epochIndex], lease->Ref(), state.hostFrame++, false);
						graph.AddPersistentTicketCandidate(*reuse, entry.recording);
						if (RenderGraph::BindPersistentTicketRecording(*reuse, entry.recording, version, nullptr)) {
							RenderGraph::SetPersistentTicketHostTag(*reuse, version);
							if (state.Current(*reuse)) ticket = std::move(reuse);
						}
						if (reuse) (void)graph.CompletePersistentTicket(*reuse, {});
					}
					if (ticket) state.liveReused.fetch_add(1, std::memory_order_relaxed);
					else entry = {};
				}
			}
			if (!ticket) {
				Async::Preparation preparation;
				ticket = graph.PreparePersistentTicket(m_device, state.epochs[epochIndex], lease->Ref(), state.hostFrame++, version);
				RenderGraph::SetPersistentTicketHostTag(*ticket, version);
				if (liveCopy) {
					auto& kept = state.liveKept[epochIndex];
					if (kept.size() <= lease->copy) kept.resize(size_t{lease->copy} + 1);
					auto recording = RenderGraph::PersistentTicketRecording(*ticket);
					if (recording && RenderGraph::PersistentRecordingReplayable(*recording)) {
						kept[lease->copy] = {std::move(recording), state.generation[epochIndex], version};
						state.liveKeptCount.fetch_add(1, std::memory_order_relaxed);
					} else {
						kept[lease->copy] = {};
						state.liveUnkeepable.fetch_add(1, std::memory_order_relaxed);
					}
				}
			}
		}
	} catch (const std::exception& e) {
		error = e.what();
	} catch (...) {
		error = "unknown exception";
	}
	if (!error.empty() || !ticket) {
		// The lease is free again: no ticket holds it. A ticket made before the failure gives up what it prepared.
		if (ticket) {
			try {
				(void)graph.CompletePersistentTicket(*ticket, {});
			} catch (...) {
			}
		}
		ReleaseLease(*lease, &state);
		FailAsyncEpoch(state, epochIndex, error.empty() ? std::string("no ticket was prepared") : std::move(error));
		return;
	}
	state.outstanding[epochIndex] = ticket;
	// A replacement goes where its submission waits for it, and takes the discarded ticket's place in hand.
	if (reprepare) {
		state.PushReprepared(epochIndex, std::move(ticket));
	} else {
		++state.inHand[epochIndex];
		state.PushReady(epochIndex, std::move(ticket));
	}
	state.ticketWake.fetch_add(1, std::memory_order_release);
	state.ticketWake.notify_all();
}

void GraphHost::DropStaleTickets(Async& state, uint32_t epochIndex) {
	if (state.revisionDriven[epochIndex]) return;
	auto& queue = state.queues[epochIndex];
	std::vector<std::shared_ptr<Async::Ticket>> stale;
	{
		// The render thread waits for the replacement (WaitTicket's reprepared) and takes none of these meanwhile.
		std::scoped_lock lock(queue.mutex);
		std::erase_if(queue.ready, [&](std::shared_ptr<Async::Ticket>& ticket) {
			if (state.Current(*ticket)) return false;
			stale.push_back(std::move(ticket));
			return true;
		});
		queue.count.store(static_cast<uint32_t>(queue.ready.size()), std::memory_order_release);
	}
	auto& graph = *m_instances[state.instanceOf[epochIndex]].graph;
	for (const auto& ticket : stale) {
		(void)graph.CompletePersistentTicket(*ticket, {});
		ReleaseLease(*m_leaseTable[RenderGraph::PersistentTicketRun(*ticket).lease].load(std::memory_order_acquire), &state);
		if (state.inHand[epochIndex]) --state.inHand[epochIndex];
		state.staleAhead.fetch_add(1, std::memory_order_relaxed);
	}
}

void GraphHost::FailAsyncEpoch(Async& state, uint32_t epochIndex, std::string error) {
	const auto epoch = state.epochs[epochIndex];
	spdlog::error("GraphHost: epoch {}'s graph failed to prepare (build {}); the other epochs keep running: {}", epoch,
		state.generation[epochIndex], error);
	// Nothing more is prepared or recorded for it until its instance is attached at a new build (epochFailed).
	state.outstanding[epochIndex].reset();
	state.liveRecordings[epochIndex].clear();
	state.liveKept[epochIndex].clear();
	state.epochErrors[epochIndex] = error;
	// The owner hears of it at its next PollEpochFailures; the render thread sees the flag first if it submits the epoch before that.
	ReportEpochFailure({epoch, state.generation[epochIndex], false, std::move(error)});
	state.epochFailed[epochIndex].store(true, std::memory_order_release);
	// A render thread waiting for this epoch's ticket wakes and sees the failure.
	state.ticketWake.fetch_add(1, std::memory_order_release);
	state.ticketWake.notify_all();
}

void GraphHost::AddRevisionCandidate(Async& state, uint32_t epochIndex, RenderGraph::PersistentTicket& ticket, const EpochRecording& recording) {
	const auto run = RenderGraph::PersistentTicketRun(ticket);
	if (!run.Published() || run.copy >= recording.copies.size())
		throw std::logic_error("A revision recording has no copy for its epoch's ticket");
	// A recording of some copies only (RequestEpochRecording's copy) is no candidate for a run of another copy.
	if (!recording.copies[run.copy]) return;
	m_instances[state.instanceOf[epochIndex]].graph->AddPersistentTicketCandidate(ticket, recording.copies[run.copy]);
}

void GraphHost::NoteNewVersions() noexcept {
	// After the version's publication (release), so a preparation that reads the new value resolves the new version.
	if (m_async) m_async->backingVersion.fetch_add(1, std::memory_order_acq_rel);
}

bool GraphHost::EpochRecordingVersionsCurrent(const EpochRecording& recording, uint32_t copy) const noexcept {
	return copy < recording.copies.size() && recording.copies[copy] && RenderGraph::PersistentRecordingVersionsCurrent(*recording.copies[copy]);
}

std::string GraphHost::EpochRecordingStaleVersion(const EpochRecording& recording, uint32_t copy) const {
	if (copy >= recording.copies.size() || !recording.copies[copy]) return "no copy " + std::to_string(copy);
	if (auto stale = RenderGraph::PersistentRecordingStaleVersion(*recording.copies[copy]); !stale.empty()) return stale;
	return RenderGraph::PersistentRecordingStaleBacking(*recording.copies[copy]);
}

void GraphHost::StopAsync() {
	BT_ZONE_SCOPE("ORG.Host.StopAsync");
	if (!m_async) return;
	// Tickets still on the submission thread submit on this async state: they go first.
	DrainSubmissions("async epochs stopping");
	auto async = std::move(m_async);
	Async::Message stop;
	stop.kind = Async::Message::Kind::Stop;
	(void)async->Post(stop);
	if (async->thread.joinable()) async->thread.join();
	BufferBase::UnregisterConcurrentBackingCapture();
	// An epoch whose preparation failed stays failed without the host's thread: its graph is the one that failed.
	for (uint32_t index = 0; index < async->epochs.size(); ++index) {
		auto& instance = m_instances[async->instanceOf[index]];
		if (async->epochFailed[index].load(std::memory_order_acquire) && async->generation[index] == instance.buildGeneration) {
			instance.failed = true;
			instance.failure = async->epochErrors[index];
		}
	}
	// The recording requests not recorded: dropped, each completed so (EpochRecordingDropped), here, on the thread that builds -
	// the taken ones and the ones still queued (a request posted after this goes to the next thread, on the next build's graph).
	// The host's end drops them without their callbacks.
	if (!m_ending) {
		for (auto& request : async->pendingRequests)
			if (request && request->done)
				request->done(nullptr, std::make_exception_ptr(EpochRecordingDropped(async->generation[request->epochIndex])));
		const auto dropped = std::make_exception_ptr(EpochRecordingDropped(m_buildGeneration));
		for (auto& node : m_recordingQueue->Take())
			if (node->done) node->done(nullptr, dropped);
	}
	async->pendingRequests.clear();
	// Staged graphs go with the host's thread that recorded for them (m_async is gone, so only the instances' views are cleared).
	for (size_t index = 0; index < m_instances.size(); ++index)
		if (m_instances[index].staged) {
			RetireStaged(index);
			MarkRebuild(m_instances[index], "its staged graph was dropped with the host's thread");
		}
	// Unsubmitted tickets are abandoned, and their leases given back.
	m_heldRun.reset();
	for (uint32_t index = 0; index < async->epochs.size(); ++index) {
		auto tickets = async->TakeAll(index);
		for (const auto& ticket : tickets) {
			if (!async->failed.load())
				if (auto& graph = m_instances[async->instanceOf[index]].graph) (void)graph->CompletePersistentTicket(*ticket, {});
			ReleaseLease(*m_leaseTable[RenderGraph::PersistentTicketRun(*ticket).lease].load(std::memory_order_acquire), async.get());
		}
	}
	m_frameNumber = (std::max)(m_frameNumber, async->hostFrame);
	if (auto* graph = AnyGraph())
		if (auto* uploads = graph->GetUploadService()) uploads->SetStagedUploadsRecordedDirectly(false);
	// Every run of the host's own: its lists came from this state's pools, which go with it. Their work is waited for, then they
	// retire. An external run waits for its client's point: it holds no lists, and whoever retires runs next retires it.
	{
		std::vector<RunLease*> runs;
		{
			std::scoped_lock lock(m_leaseMutex);
			for (const auto& lease : m_leaseStore)
				if (lease->state != RunLease::State::Free && !lease->external) runs.push_back(lease.get());
		}
		for (auto* lease : runs) {
			bool complete = true;
			for (const auto& [timeline, value] : lease->points)
				if ((*timeline)->HostWait(value) != rhi::Result::Ok) complete = false;
			if (!complete) {
				if (lease->resourceOwner) m_uncertainExecutionOwners.push_back(std::move(lease->resourceOwner));
				if (lease->keepAlive) m_uncertainUploadOwners.push_back(std::move(lease->keepAlive));
			}
			if (lease->state == RunLease::State::Submitted) RetireLease(*lease, async.get());
			else ReleaseLease(*lease, async.get());
		}
	}
	if (async->uncertainResourceOwner)
		m_uncertainExecutionOwners.push_back(std::move(async->uncertainResourceOwner));
	if (async->uncertainUploadOwner)
		m_uncertainUploadOwners.push_back(std::move(async->uncertainUploadOwner));
}

GraphHost::AsyncStats GraphHost::TakeAsyncStats() noexcept {
	{
		// What the submission thread counted (FinishTicket's counts there).
		std::scoped_lock lock(m_threadStatsMutex);
		m_asyncStats.submitted += m_threadStats.submitted;
		m_asyncStats.uploads += m_threadStats.uploads;
		m_asyncStats.queueUploadCopies += m_threadStats.queueUploadCopies;
		m_asyncStats.sparedUploadWaits += m_threadStats.sparedUploadWaits;
		m_asyncStats.lateRoutedUploads += m_threadStats.lateRoutedUploads;
		m_asyncStats.lateUploadWaits += m_threadStats.lateUploadWaits;
		m_asyncStats.undeclaredUploads += m_threadStats.undeclaredUploads;
		m_threadStats = {};
	}
	m_asyncStats.recorded = 0;
	m_asyncStats.predicted = m_asyncStats.planned = 0;
	for (auto& instance : m_instances) {
		if (!instance.graph) continue;
		const auto recordings = instance.graph->TakePersistentRecordingStats();
		m_asyncStats.recorded += recordings.recorded;
		m_asyncStats.predicted += recordings.predicted;
		m_asyncStats.planned += recordings.planned;
	}
	if (m_async) {
		m_asyncStats.revisionCopiesRecorded = m_async->revisionCopiesRecorded.exchange(0, std::memory_order_relaxed);
		m_asyncStats.revisionRecordings = m_async->revisionRecordings.exchange(0, std::memory_order_relaxed);
		{
			std::scoped_lock lock(m_async->postLock);
			m_asyncStats.wakesSkipped = std::exchange(m_async->wakesSkipped, 0);
		}
		m_asyncStats.copiesBusy = m_async->copiesBusy.exchange(0, std::memory_order_relaxed);
		m_asyncStats.copyWaits = m_async->copyWaits.exchange(0, std::memory_order_relaxed);
		m_asyncStats.staleAhead = m_async->staleAhead.exchange(0, std::memory_order_relaxed);
		m_asyncStats.liveReused = m_async->liveReused.exchange(0, std::memory_order_relaxed);
		m_asyncStats.liveKept = m_async->liveKeptCount.exchange(0, std::memory_order_relaxed);
		m_asyncStats.liveUnkeepable = m_async->liveUnkeepable.exchange(0, std::memory_order_relaxed);
		m_asyncStats.liveSpilled = m_async->liveSpilled.exchange(0, std::memory_order_relaxed);
	}
	return std::exchange(m_asyncStats, {});
}

void GraphHost::SubmitEpoch(uint32_t epoch, const FrameCallback& beforeSubmit, std::shared_ptr<const void> resourceOwner,
	std::shared_ptr<const EpochRecording> recording) {
	BT_ZONE_SCOPE("ORG.Host.SubmitEpoch");
	// A failed submission thread submits nothing more. The caller hears of it here, as of a failed submission on this thread.
	ThrowIfSubmitThreadFailed();
	m_lastTimings = {};
	m_lastTimings.async = true;
	auto lap = [last = std::chrono::steady_clock::now()](double& a_into) mutable {
		const auto now = std::chrono::steady_clock::now();
		a_into += std::chrono::duration<double, std::micro>(now - last).count();
		last = now;
	};
	ReleaseRetiredGraphs(false);
	const auto instance = InstanceOf(epoch);
	if (instance == kNoInstance) throw std::invalid_argument("SubmitEpoch: epoch " + std::to_string(epoch) + " is not one of the host's");
	if (!m_async && !m_asyncEpochs.empty()) {
		BuildDueInstances(false);
		StartAsync();
	}
	if (!m_async) throw std::logic_error("SubmitEpoch requires async epochs");
	// The epoch's own graph: built again here when due, the other epochs' tickets kept.
	if (BuildDue(instance)) RebuildAsyncInstances({&instance, 1});
	lap(m_lastTimings.buildUs);
	auto& async = *m_async;
	async.ThrowIfFailed();
	const auto index = async.IndexOf(epoch);
	// An epoch whose graph failed runs nothing until a build of it succeeds; the others run on.
	if (!m_instances[instance].graph || m_instances[instance].failed)
		throw EpochFailedError(epoch, m_instances[instance].failure);
	if (async.epochFailed[index].load(std::memory_order_acquire)) throw EpochFailedError(epoch, async.epochErrors[index]);
	m_heldRun.reset();
	auto ticket = async.TakeReady(index);
	if (!ticket) {
		++m_asyncStats.waited;
		ticket = async.WaitTicket(index);
	}
	// A pinned epoch (SetEpochCopy): a ticket of another copy goes back, and one for the named copy takes its place. Then the
	// next ticket is prepared for the copy the epoch's runs step to, which is what a caller that names its frame slots asks for
	// next: the step between its last two runs. An epoch that runs every other frame steps two copies, not one.
	if (async.revisionDriven[index]) {
		const uint32_t pinned = async.pinnedCopy[index].load(std::memory_order_acquire);
		if (pinned != RunRef::kNoCopy) {
			if (RenderGraph::PersistentTicketRun(*ticket).copy != pinned) {
				++m_asyncStats.copyRepins;
				Async::Message discard;
				discard.kind = Async::Message::Kind::Discard;
				discard.epochIndex = index;
				discard.ticket = std::move(ticket);
				if (!async.Post(discard)) async.ThrowIfFailed();
				ticket = async.WaitTicket(index);
			}
			const uint32_t last = std::exchange(async.lastPinned[index], pinned);
			const uint32_t step = last == RunRef::kNoCopy ? 1 : (pinned + async.copyCount - last) % async.copyCount;
			async.pinnedCopy[index].store((pinned + step) % async.copyCount, std::memory_order_release);
		}
	}
	lap(m_lastTimings.ticketWaitUs);
	if (m_submitQueue && !m_instances[instance].graph->IsExternalPersistentEpoch(epoch)) {
		// The commit here; the rest on the submission thread, in order after the tickets before it.
		auto commit = CommitTicket(async, index, std::move(ticket), beforeSubmit, std::move(resourceOwner), std::move(recording));
		EnqueueTicket(async, std::move(commit));
	} else {
		// An external epoch's submission stays on this thread: it comes after every ticket the thread still holds.
		DrainSubmissions("a submission on the calling thread");
		SubmitTicket(async, index, std::move(ticket), beforeSubmit, std::move(resourceOwner), std::move(recording));
	}
}

void GraphHost::RunSubmitThread() {
	auto& queue = *m_submitQueue;
	constexpr size_t kMostGrouped = SubmitQueue::kMostGrouped;
	{
		std::scoped_lock lock(queue.mutex);
		queue.threadId = std::this_thread::get_id();
	}
	for (;;) {
		SubmitQueue::Entry entry;
		{
			std::unique_lock lock(queue.mutex);
			queue.work.wait(lock, [&] { return !queue.entries.empty(); });
			entry = std::move(queue.entries.front());
			queue.entries.pop_front();
		}
		// A push that waited for room can go on.
		queue.progress.notify_all();
		const auto kind = entry.kind;
		if (kind == SubmitQueue::Entry::Kind::Stop) {
			{
				std::scoped_lock lock(queue.mutex);
				++queue.passed;
			}
			queue.progress.notify_all();
			return;
		}
		const auto start = std::chrono::steady_clock::now();
		uint64_t discarded = 0, finished = 0, joined = 0, entries = 1;
		if (kind == SubmitQueue::Entry::Kind::Ticket) {
			BT_ZONE_SCOPE("ORG.Host.SubmitThread.Ticket");
			// The tickets queued right behind it go with it, as one group: their queue submissions may share a vkQueueSubmit
			// (RenderGraph::SubmitCarry). Only what is queued now: the thread never waits for more.
			std::vector<SubmitQueue::Entry> group;
			group.push_back(std::move(entry));
			if (m_submitCarry) {
				{
					std::scoped_lock lock(queue.mutex);
					while (!queue.entries.empty() && queue.entries.front().kind == SubmitQueue::Entry::Kind::Ticket && group.size() < kMostGrouped) {
						group.push_back(std::move(queue.entries.front()));
						queue.entries.pop_front();
					}
				}
				queue.progress.notify_all();
			}
			entries = group.size();
			const bool merge = group.size() > 1;
			// The sequencer stays held across the group: no other submitter's work comes between a held submission and the ticket
			// that joins it (a wait on a value the held one signals would then never end).
			std::unique_lock<std::recursive_mutex> sequencer;
			if (merge && !queue.failed.load(std::memory_order_acquire)) {
				auto& first = *group.front().async;
				sequencer = m_instances[first.instanceOf[group.front().commit->index]].graph->HoldPersistentSequencer();
			}
			DeferredPosts posts;
			AsyncStats stats{};
			for (size_t i = 0; i < group.size(); ++i) {
				auto& async = *group[i].async;
				auto& commit = *group[i].commit;
				if (queue.failed.load(std::memory_order_acquire)) {
					// After a failure nothing more is submitted. The ticket goes back to the host's thread unadmitted, which frees its lease.
					if (commit.ticket) {
						Async::Message discard;
						discard.kind = Async::Message::Kind::Discard;
						discard.epochIndex = commit.index;
						discard.ticket = std::move(commit.ticket);
						(void)async.TryPost(discard);
					}
					++discarded;
					continue;
				}
				FrameTimings timings{};
				const bool carryOut = merge && i + 1 < group.size();
				try {
					FinishTicket(async, commit, timings, stats, merge ? m_submitCarry.get() : nullptr, carryOut, merge ? &posts : nullptr);
					++finished;
				} catch (const std::exception& e) {
					FailSubmitThread(async, commit.index, e.what());
				} catch (...) {
					FailSubmitThread(async, commit.index, "an unknown exception");
				}
			}
			if (merge) {
				// The group's last held submission, then the completions, in order, once the sequencer is free (the host's thread
				// takes it to retire runs).
				uint32_t held = 0;
				if (!RenderGraph::FlushSubmitCarry(*m_submitCarry, &held))
					FailSubmitThread(*group.back().async, group.back().commit->index, "a held queue submission was refused");
				if (sequencer.owns_lock()) sequencer.unlock();
				for (auto& message : posts.messages)
					(void)group.front().async->Post(message, message.kind == Async::Message::Kind::Submitted);
				joined = RenderGraph::TakeSubmitCarryJoins(*m_submitCarry);
			}
			std::scoped_lock lock(m_threadStatsMutex);
			m_threadStats.submitted += stats.submitted;
			m_threadStats.uploads += stats.uploads;
			m_threadStats.queueUploadCopies += stats.queueUploadCopies;
			m_threadStats.sparedUploadWaits += stats.sparedUploadWaits;
			m_threadStats.lateRoutedUploads += stats.lateRoutedUploads;
			m_threadStats.lateUploadWaits += stats.lateUploadWaits;
			m_threadStats.undeclaredUploads += stats.undeclaredUploads;
		} else if (kind == SubmitQueue::Entry::Kind::Marker && entry.marker) {
			// Every ticket before the marker is on its queue: the point's values are the queues' last ones now.
			entry.marker->points = SubmittedPointNow().points;
			entry.marker->passed.store(true, std::memory_order_release);
			entry.marker->passed.notify_all();
		} else if (kind == SubmitQueue::Entry::Kind::Closure && entry.closure) {
			BT_ZONE_SCOPE("ORG.Host.SubmitThread.Closure");
			try {
				entry.closure();
			} catch (const std::exception& e) {
				spdlog::error("GraphHost: a closure on the submission thread threw: {}", e.what());
			} catch (...) {
				spdlog::error("GraphHost: a closure on the submission thread threw");
			}
		}
		// The entry's objects (the ticket, the closure's captures) go before a drain that waits for it returns.
		entry = {};
		{
			std::scoped_lock lock(queue.mutex);
			queue.passed += entries;
			queue.stats.busyUs += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
			queue.stats.discarded += discarded;
			queue.stats.tickets += finished;
			queue.stats.joined += joined;
			if (kind == SubmitQueue::Entry::Kind::Ticket) ++queue.stats.groups;
		}
		queue.progress.notify_all();
	}
}

void GraphHost::StopSubmitThread() {
	if (!m_submitQueue || !m_submitQueue->thread.joinable()) return;
	DrainSubmissions("the host's end");
	{
		std::scoped_lock lock(m_submitQueue->mutex);
		m_submitQueue->entries.emplace_back();  // Kind::Stop
		++m_submitQueue->pushed;
	}
	m_submitQueue->work.notify_one();
	m_submitQueue->thread.join();
}


void GraphHost::HoldSubmissions() {
	auto* queue = m_submitQueue.get();
	if (!queue) return;
	std::scoped_lock lock(queue->mutex);
	if (queue->holdDepth && queue->holder != std::this_thread::get_id())
		throw std::logic_error("HoldSubmissions: another thread holds the submission thread's entries");
	queue->holder = std::this_thread::get_id();
	++queue->holdDepth;
}

void GraphHost::ReleaseSubmissions() {
	auto* queue = m_submitQueue.get();
	if (!queue) return;
	std::unique_lock lock(queue->mutex);
	if (!queue->holdDepth || queue->holder != std::this_thread::get_id()) return;
	if (--queue->holdDepth) return;
	queue->holder = {};
	PushHeld(*queue, lock);
}

void GraphHost::EnqueueTicket(Async& async, TicketCommit commit) {
	BT_ZONE_SCOPE("ORG.Host.EnqueueTicket");
	SubmitQueue::Entry entry;
	entry.kind = SubmitQueue::Entry::Kind::Ticket;
	entry.async = &async;
	entry.commit.emplace(std::move(commit));
	PushOrHold(*m_submitQueue, std::move(entry));
}

void GraphHost::EnqueueOnSubmitThread(std::function<void()> a_closure) {
	if (!m_submitQueue) {
		a_closure();
		return;
	}
	SubmitQueue::Entry entry;
	entry.kind = SubmitQueue::Entry::Kind::Closure;
	entry.closure = std::move(a_closure);
	PushOrHold(*m_submitQueue, std::move(entry));
}

void GraphHost::DrainSubmissions(const char* why) {
	auto* queue = m_submitQueue.get();
	if (!queue) return;
	std::unique_lock lock(queue->mutex);
	// The thread's own work (a pass recorded there that reaches a drain) is already in order.
	if (std::this_thread::get_id() == queue->threadId) return;
	// What this thread holds goes first, or the drain would wait for entries never pushed.
	if (queue->holder == std::this_thread::get_id()) PushHeld(*queue, lock);
	const auto target = queue->pushed;
	if (queue->passed >= target) return;
	BT_ZONE_SCOPE("ORG.Host.DrainSubmissions");
	const auto start = std::chrono::steady_clock::now();
	queue->progress.wait(lock, [&] { return queue->passed >= target; });
	const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
	++queue->stats.drains;
	queue->stats.drainUs += us;
	auto& caller = queue->drains[why ? why : "unnamed"];
	++caller.first;
	caller.second += us;
}

GraphHost::SubmitThreadStats GraphHost::TakeSubmitThreadStats() {
	if (!m_submitQueue) return {};
	std::scoped_lock lock(m_submitQueue->mutex);
	auto stats = std::exchange(m_submitQueue->stats, {});
	for (auto& [why, counts] : m_submitQueue->drains) stats.drainsByCaller.emplace_back(why, counts);
	m_submitQueue->drains.clear();
	return stats;
}

void GraphHost::FailSubmitThread(Async& async, uint32_t epochIndex, const std::string& error) {
	auto& queue = *m_submitQueue;
	{
		std::scoped_lock lock(queue.mutex);
		if (queue.failed.load(std::memory_order_relaxed)) return;
		queue.error = error;
		queue.failed.store(true, std::memory_order_release);
	}
	spdlog::error("GraphHost: the submission thread failed, and submits nothing more: {}", error);
	// Builds drain the thread first, so the instance's build generation is stable here.
	const auto& instance = m_instances[async.instanceOf[epochIndex]];
	ReportEpochFailure({async.epochs[epochIndex], instance.buildGeneration, false, "The submission thread failed: " + error});
}

void GraphHost::ThrowIfSubmitThreadFailed() const {
	if (!m_submitQueue || !m_submitQueue->failed.load(std::memory_order_acquire)) return;
	std::scoped_lock lock(m_submitQueue->mutex);
	throw std::runtime_error("The submission thread failed: " + m_submitQueue->error);
}

void GraphHost::SubmitTicket(Async& async, uint32_t index, std::shared_ptr<RenderGraph::PersistentTicket> ticket,
	const FrameCallback& beforeSubmit, std::shared_ptr<const void> resourceOwner, std::shared_ptr<const EpochRecording> recording) {
	BT_ZONE_SCOPE("ORG.Host.SubmitTicket");
	BT_ZONE_VALUE(async.epochs[index]);
	auto commit = CommitTicket(async, index, std::move(ticket), beforeSubmit, std::move(resourceOwner), std::move(recording));
	FinishTicket(async, commit, m_lastTimings, m_asyncStats);
}

GraphHost::TicketCommit GraphHost::CommitTicket(Async& async, uint32_t index, std::shared_ptr<RenderGraph::PersistentTicket> ticket,
	const FrameCallback& beforeSubmit, std::shared_ptr<const void> resourceOwner, std::shared_ptr<const EpochRecording> recording) {
	BT_ZONE_SCOPE("ORG.Host.CommitTicket");
	// A discard always wakes the host's thread.
	auto postDiscard = [&](Async::Message& message) {
		if (!async.Post(message)) throw std::runtime_error("Async epoch host stopped before accepting its completion");
	};
	auto lap = [last = std::chrono::steady_clock::now()](double& a_into) mutable {
		const auto now = std::chrono::steady_clock::now();
		a_into += std::chrono::duration<double, std::micro>(now - last).count();
		last = now;
	};
	auto& graph = *m_instances[async.instanceOf[index]].graph;
	const auto run = RenderGraph::PersistentTicketRun(*ticket);
	const uint32_t slot = run.lease;
	m_asyncRun = run;
	runtime::ScopedActiveGraphServices services(graph.GetUploadService(), graph.GetDescriptorService());
	// The lease's last run retired before the host's thread opened it: its upload pages are free again. What retires from
	// the deletion queues goes to the host's thread with the submission: its driver frees take the device's memory locks,
	// which the render thread must not wait on here.
	if (auto* uploads = graph.GetUploadService()) uploads->ProcessDeferredReleases(slot);
	std::shared_ptr<void> retired = m_desc.runtime->RetireElsewhere();
	lap(m_lastTimings.releaseUs);
	if (beforeSubmit) {
		BT_ZONE_SCOPE("ORG.Host.CommitInputs");
		struct Submitting {
			GraphHost& host;
			~Submitting() { host.m_submittingTicket = nullptr; }
		} submitting{*this};
		m_submittingTicket = ticket.get();
		beforeSubmit(graph);
	}
	lap(m_lastTimings.beforePrepareUs);
	// A revision-driven epoch: the revision's copy for this run, recorded ahead. Never prepared again: a recording that does not
	// match the ticket is the caller's error. A live epoch takes none: it records every run.
	if (!recording) recording = std::exchange(m_chosenRecording, nullptr);
	m_chosenRecording.reset();
	if (!async.revisionDriven[index] && recording) {
		Async::Message discard;
		discard.kind = Async::Message::Kind::Discard;
		discard.epochIndex = index;
		discard.ticket = std::move(ticket);
		postDiscard(discard);
		throw std::logic_error("A revision's recording was given for a live epoch, which records every run");
	}
	if (async.revisionDriven[index]) {
		std::string why = "no recording given";
		const bool bound = recording && recording->epoch == async.epochs[index] && run.Published() && run.copy < recording->copies.size()
			&& RenderGraph::BindPersistentTicketRecording(*ticket, recording->copies[run.copy], async.backingVersion.load(std::memory_order_acquire), &why);
		if (!bound) {
			Async::Message discard;
			discard.kind = Async::Message::Kind::Discard;
			discard.epochIndex = index;
			discard.ticket = std::move(ticket);
			postDiscard(discard);
			throw std::logic_error("A revision-driven epoch was submitted without its revision's copy for the run: " + why);
		}
		++m_asyncStats.revisionSubmissions;
	}
	// The revision check stays with the commit. Each pass's revision reads the caller's state, and so does the preparation of a
	// replacement. Both must see that state as the commit left it, before the render thread moves on to the next run.
	if (!async.revisionDriven[index] && !async.Current(*ticket)) {
		BT_ZONE_SCOPE("ORG.Host.ReprepareStaleTicket");
		m_asyncStats.staleBacking += RenderGraph::PersistentTicketCurrent(*ticket);
		{
			// Why tickets were stale, by epoch and reason: the counts are logged every 500 stale tickets.
			static std::mutex reasonsLock;
			static std::map<std::string, uint32_t> reasons;
			static uint32_t seen = 0;
			auto reason = "epoch " + std::to_string(async.epochs[index]) + ": " + RenderGraph::PersistentTicketStaleReason(*ticket);
			std::lock_guard guard(reasonsLock);
			++reasons[reason];
			if (++seen % 500 == 0) {
				std::string text;
				for (const auto& [why, count] : reasons) text += "; " + std::to_string(count) + " " + why;
				spdlog::info("GraphHost: the last 500 stale tickets, by reason:{}", text);
				reasons.clear();
			}
		}
		// Prepared before beforeSubmit changed what a pass depends on, or a version it resolved: prepare it again, for the
		// same run (its latch regions were just written), and wait. The new one is prepared after the change, so it is current.
		++m_asyncStats.stale;
		Async::Message discard;
		discard.kind = Async::Message::Kind::Discard;
		discard.epochIndex = index;
		discard.ticket = std::move(ticket);
		// The discard and the reprepare are one message, so the worker prepares the replacement into this lease before it
		// can prepare the epoch's next ticket into another (the Discard handler).
		discard.reprepare = true;
		postDiscard(discard);
		ticket = async.WaitTicket(index, true);
		if (!(RenderGraph::PersistentTicketRun(*ticket) == run))
			throw std::logic_error("A reprepared async epoch ticket is not for the run whose upload list the render thread records");
	}
	// The submitted ticket's frame: one prepared again above has a new number, and the completed-frame callback
	// reports it by that one.
	m_lastHostFrame = RenderGraph::PersistentTicketHostFrame(*ticket);
	TicketCommit commit;
	commit.index = index;
	commit.run = run;
	commit.external = graph.IsExternalPersistentEpoch(async.epochs[index]);
	if (!m_externalRetired.empty()) {
		m_externalRetired.push_back(std::move(retired));
		retired = std::make_shared<std::vector<std::shared_ptr<void>>>(std::exchange(m_externalRetired, {}));
	}
	commit.retired = std::move(retired);
	commit.resourceOwner = std::move(resourceOwner);
	commit.number = ++m_submissions;
	// The frame boundaries before it, queued seals included: a later seal flushed it (a published copy waits for its run only then).
	commit.seal = m_sealsPushed.load(std::memory_order_relaxed);
	commit.frameWait = m_frameWaitValue;
	// What the commit queued goes with this ticket, wherever it is finished.
	commit.uploads = graph.TakePendingUploads(m_device, slot);
	commit.assigned = std::make_unique<Runtime::AssignmentScope>(*m_desc.runtime);
	commit.ticket = std::move(ticket);
	lap(m_lastTimings.checkUs);
	return commit;
}

void GraphHost::FinishTicket(Async& async, TicketCommit& commit, FrameTimings& timings, AsyncStats& stats, RenderGraph::SubmitCarry* carry,
	bool carryOut, DeferredPosts* posts) {
	BT_ZONE_SCOPE("ORG.Host.FinishTicket");
	// The epoch, so a capture can attribute the submission thread's time to it.
	BT_ZONE_VALUE(async.epochs[commit.index]);
	const auto index = commit.index;
	auto& ticket = commit.ticket;
	// A submission's completion message wakes the host's thread lazily (wakesOnGpu); a discard always wakes it.
	auto postControl = [&](Async::Message& message) {
		const bool lazy = message.kind == Async::Message::Kind::Submitted;
		if (!async.Post(message, lazy)) throw std::runtime_error("Async epoch host stopped before accepting its completion");
	};
	auto lap = [last = std::chrono::steady_clock::now()](double& a_into) mutable {
		const auto now = std::chrono::steady_clock::now();
		a_into += std::chrono::duration<double, std::micro>(now - last).count();
		last = now;
	};
	auto& graph = *m_instances[async.instanceOf[index]].graph;
	const uint32_t slot = commit.run.lease;
	// The ticket's lease: published by the host's thread before the ticket, and the submitter's until it posts the ticket back.
	auto& lease = *m_leaseTable[slot].load(std::memory_order_acquire);
	runtime::ScopedActiveGraphServices services(graph.GetUploadService(), graph.GetDescriptorService());
	// An external epoch's uploads go into the slot's secondary, which its execution runs first: nothing is submitted here.
	const bool external = commit.external;
	// Taken at the commit: a ticket without uploads begins, records and closes no list (most have none). Its lists are begun
	// here, external epochs' too. The other queues' lists only for a ticket with batches on them: otherwise nothing is routed there.
	const bool hasUploads = commit.uploads.Any() || !commit.uploads.supported;
	if (hasUploads && !(external ? lease.uploadSecondary : lease.upload).Recording()) {
		BT_ZONE_SCOPE("ORG.Host.BeginUploadLists");
		if (external) async.uploadSecondaryPool->Begin(lease.uploadSecondary);
		else async.uploadPool->Begin(lease.upload);
		lease.queueUploads.resize(async.queueUploadKinds.size());
		const bool otherQueues = m_desc.queueScopedUploads && !RenderGraph::PersistentTicketOnQueue(*ticket, async.graphicsSlot);
		for (size_t queue = 0; queue < async.queueUploadKinds.size() && otherQueues; ++queue)
			if (async.queueUploadKinds[queue] && !lease.queueUploads[queue].Recording())
				async.queueUploadPools[static_cast<size_t>(*async.queueUploadKinds[queue])]->Begin(lease.queueUploads[queue]);
	}
	lap(timings.checkUs);
	// The queues' last values, which every graph assigns from (the host's shared timelines): this submission's come after them,
	// and go back to the registry as soon as they are assigned, before any other graph's submission.
	// From taking them until they are published, the frontier would miss the values this submission signals: deferred
	// deletions are not sealed meanwhile. It ends at the publish below, or with it on a throw.
	std::optional<Runtime::AssignmentScope> assigning(std::in_place, *m_desc.runtime);
	auto& queues = graph.GetQueueRegistry();
	async.queueValues.resize(queues.SlotCount());
	for (size_t queue = 0; queue < queues.SlotCount(); ++queue) {
		const auto next = queues.GetCurrentFenceValue(static_cast<QueueSlotIndex>(static_cast<uint8_t>(queue)));
		async.queueValues[queue] = next ? next - 1 : 0;
	}
	struct PublishQueueValues {
		QueueRegistry& queues;
		const std::vector<uint64_t>& values;
		void Publish() const noexcept {
			for (size_t queue = 0; queue < values.size(); ++queue)
				queues.EnsureNextFenceValueAtLeast(static_cast<QueueSlotIndex>(static_cast<uint8_t>(queue)), values[queue] + 1);
		}
		~PublishQueueValues() { Publish(); }
	} publishQueueValues{queues, async.queueValues};
	// Admitted now, against the ledgers as they are: the upload plan by queue and the submission's handoff read it. The sequencer
	// stays held until the ticket is submitted and committed, so no other admission comes between.
	auto sequencer = graph.HoldPersistentSequencer();
	// Fresh textures the uploads write: their ledger states first, so the ticket's entry barriers start from what the copies leave.
	if (hasUploads) graph.SeedTakenTextureUploads(commit.uploads, async.graphicsSlot);
	graph.AdmitPersistentTicket(*ticket, async.queueValues);
	struct AbandonUnsubmitted {
		RenderGraph& graph;
		RenderGraph::PersistentTicket* ticket;
		~AbandonUnsubmitted() { if (ticket) graph.AbandonPersistentTicketAdmission(*ticket); }
	} abandonUnsubmitted{graph, ticket.get()};
	// A ticket handed back to the host's thread unsubmitted gives up its admission first (the host's thread completes it).
	auto abandonBeforeDiscard = [&] {
		abandonUnsubmitted.ticket = nullptr;
		graph.AbandonPersistentTicketAdmission(*ticket);
	};
	// The uploads queued since the last submission, as plain copies ahead of the ticket (the list ends with a full barrier).
	auto& list = lease.upload;  // begun by the host's thread (PrepareTicket), or above; unused by an external epoch
	std::shared_ptr<void> keepAlive;
	bool recorded = false;
	auto& secondary = lease.uploadSecondary;
	if (external && hasUploads) {
		rhi::SecondaryInheritance outside{};
		outside.insideRendering = false;
		outside.debugName = "ORG external ticket uploads";
		// Run by the host's command buffers: never chained into one, as the epoch's own secondaries
		// (RecordPreparedRhiExecutionBatch).
		outside.simultaneousUse = true;
		if (!secondary.list || secondary.list->BeginSecondary(outside) != rhi::Result::Ok)
			throw std::runtime_error("An external epoch's uploads secondary could not begin");
	}
	auto commands = external ? secondary.list.Get() : list.list.Get();
	auto& uploadLists = async.uploadLists;
	// Queue-scoped uploads (Desc::queueScopedUploads): the run's open lists of the other queues, which take the uploads their
	// queue uses first, and go to it just ahead of the ticket's first batch there (SubmitPersistentTicket).
	auto& queueUploadLists = lease.queueUploads;
	const bool queueScoped = m_desc.queueScopedUploads;
	for (size_t queue = 0; queue < async.queueLists.size(); ++queue)
		async.queueLists[queue] = queueScoped && queue < queueUploadLists.size() && queueUploadLists[queue].list && queueUploadLists[queue].Recording()
			? queueUploadLists[queue].list.Get() : rhi::CommandList{};
	size_t queueCopies = 0;
	uploadLists.clear();
	// BasicRHI's command census names the upload lists (diagnostics).
	if (rhi::census::Enabled()) rhi::census::SetLabel("upload");
	const bool supported = !hasUploads || graph.RecordTakenUploads(commit.uploads, commands, keepAlive, recorded,
		queueScoped ? ticket.get() : nullptr, queueScoped ? std::span<rhi::CommandList>(async.queueLists) : std::span<rhi::CommandList>{}, &queueCopies);
	if (hasUploads) {
		BT_ZONE_SCOPE("ORG.Host.CloseUploadList");
		(external ? secondary : list).list->End();
		// Every one the host's thread began, used or not (they go back to their pools when the run retires).
		for (auto& queueUploads : queueUploadLists)
			if (queueUploads.list && queueUploads.Recording()) queueUploads.list->End();
		// Closed, so their allocators' claims go.
		list.EndRecording();
		secondary.EndRecording();
		for (auto& queueUploads : queueUploadLists) queueUploads.EndRecording();
	}
	stats.queueUploadCopies += queueCopies;
	if (!supported) {
		abandonBeforeDiscard();
		Async::Message discard;
		discard.kind = Async::Message::Kind::Discard;
		discard.epochIndex = index;
		discard.ticket = std::move(ticket);
		postControl(discard);
		throw std::runtime_error("Async epochs cannot record these uploads (only pointer-targeted buffer copies)");
	}
	// The ticket's uploads, for SubmitPersistentTicket to submit (none when it recorded nothing).
	RenderGraph::TicketUploads uploads;
	Async::Message submitted;
	submitted.kind = Async::Message::Kind::Submitted;
	submitted.number = commit.number;
	submitted.seal = commit.seal;
	submitted.epochIndex = index;
	submitted.keepAlive = std::move(keepAlive);
	submitted.retired = std::move(commit.retired);
	submitted.resourceOwner = std::move(commit.resourceOwner);
	if (!external && recorded) {
		BT_ZONE_SCOPE("ORG.Host.EnqueueUploads");
		++stats.uploads;
		const auto value = ++async.queueValues.at(async.graphicsSlot);
		const rhi::TimelinePoint signal{async.graphicsFence, value};
		uploadLists.push_back(commands);
		// SubmitPersistentTicket submits it, after the group an earlier ticket left held (the submission thread's carry): in the
		// ticket's first queue submission when it can, otherwise on their own first.
		uploads.lists = {uploadLists.data(), uploadLists.size()};
		uploads.queue = async.graphicsQueue;
		uploads.queueSlot = async.graphicsSlot;
		uploads.signal = signal;
		submitted.uploadSignal = {async.graphicsSlot, value};
	}
	lap(timings.uploadsUs);
	bool ok = false;
	try {
		// Submits and commits, or abandons its admission itself.
		abandonUnsubmitted.ticket = nullptr;
		ok = graph.SubmitPersistentTicket(*ticket, async.queueValues, submitted.submission, submitted.uploadSignal, commit.frameWait,
			external && recorded, carry, carryOut, uploads.lists.empty() ? nullptr : &uploads);
	} catch (...) {
		// The queue may have accepted a prefix of this ticket. Completion is
		// unknown, so preserve both owners until device teardown.
		if (submitted.resourceOwner) m_uncertainExecutionOwners.push_back(std::move(submitted.resourceOwner));
		if (submitted.keepAlive) m_uncertainUploadOwners.push_back(std::move(submitted.keepAlive));
		throw;
	}
	publishQueueValues.Publish();
	assigning.reset();
	// Published: the frontier covers this ticket's work from here on.
	commit.assigned.reset();
	sequencer.unlock();
	lap(timings.submitUs);
	// The handoff and release lists the submission recorded: kept until the run completes, with the uploads' staging.
	if (!submitted.submission.leases.empty()) {
		struct Kept {
			std::shared_ptr<void> uploads;
			std::vector<std::shared_ptr<const void>> leases;
		};
		submitted.keepAlive = std::make_shared<Kept>(Kept{std::move(submitted.keepAlive), std::move(submitted.submission.leases)});
	}
	if (external && ok) {
		// The caller's to execute (TakeExternalExecution): the uploads first, then the handoff into the execution.
		auto execution = submitted.submission.external;
		if (!execution) throw std::logic_error("An external epoch's ticket gave no execution");
		if (recorded) {
			if (execution->segments.empty()) execution->segments.emplace_back();
			auto& leading = execution->segments.front().leading;
			leading.insert(leading.begin(), commands);
		}
		if (!m_externalExecution) {
			m_externalExecution = std::move(execution);
		} else {
			// The caller has not taken the earlier tickets' execution (TakeExternalExecution): this ticket's segments run after
			// theirs, in one execution. When a batch of that execution off the stream waits for one of its segments, its exit
			// needed a host submission of its own (exitSplit): this ticket's segments, which may wait for that batch, start one.
			auto& into = *m_externalExecution;
			if (!execution->segments.empty() && into.exitSplit) execution->segments.front().split = true;
			into.exitSplit = into.exitSplit || execution->exitSplit;
			for (auto& segment : execution->segments) into.segments.push_back(std::move(segment));
			into.keepAlive.insert(into.keepAlive.end(), execution->keepAlive.begin(), execution->keepAlive.end());
		}
	}
	if (submitted.uploadSignal.second) stats.sparedUploadWaits += RenderGraph::PersistentTicketSparedUploadWaits(*ticket);
	{
		const auto uploads = RenderGraph::PersistentTicketUploadStats(*ticket);
		stats.lateRoutedUploads += uploads.lateRouted;
		stats.lateUploadWaits += uploads.lateWaits;
		stats.undeclaredUploads += uploads.undeclared;
	}
	submitted.ticket = std::move(ticket);
	if (posts) {
		// Posted by the caller once the group's held submission is on its queue: the host's thread then never handles a ticket
		// whose batches are still held.
		++stats.submitted;
		posts->messages.push_back(std::move(submitted));
		lap(timings.postUs);
		if (!ok) throw std::runtime_error("Async epoch submission failed");
		return;
	}
	try {
		postControl(submitted);
	} catch (...) {
		if (submitted.resourceOwner) m_uncertainExecutionOwners.push_back(std::move(submitted.resourceOwner));
		if (submitted.keepAlive) m_uncertainUploadOwners.push_back(std::move(submitted.keepAlive));
		throw;
	}
	lap(timings.postUs);
	++stats.submitted;
	if (!ok) throw std::runtime_error("Async epoch submission failed");
}

} // namespace org
