#include "OpenRenderGraph/GraphHost.h"

#include <BasicTelemetry/Tracy.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Resources/Buffers/DynamicBufferBase.h"
#include "Render/FrameArena.h"
#include "Render/MemoryIntrospectionBackend.h"
#include "Render/PassExecutionContext.h"
#include "Render/RenderGraph/PersistentGraph.h"
#include "Render/RenderGraph/PersistentAliasArena.h"
#include "Render/Runtime/IUploadService.h"
#include "Render/Runtime/OpenRenderGraphSettings.h"
#include "Render/Runtime/RuntimeDevice.h"
#include "Render/Runtime/ScopedActiveGraphServices.h"
#include "Render/Runtime/ThreadPoolTaskService.h"
#include "HostUploadExtension.h"

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
	// The host's seal count (GraphHost::m_seals), written by the render thread.
	const std::atomic<uint64_t>* seals = nullptr;
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
	// Recording requests taken from the host's queue (RecordingQueue), by the host's thread, in posting order.
	struct RecordingRequest {
		uint32_t epochIndex = 0;
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
	// One ticket cell per epoch: stored by the host's thread, taken by the render thread.
	std::unique_ptr<std::atomic<std::shared_ptr<Ticket>>[]> cells;
	// Render thread -> host thread, single producer / single consumer, in order.
	static constexpr uint64_t kMailbox = 64;
	std::array<Message, kMailbox> mailbox;
	std::atomic<uint64_t> written{0}, read{0};
	rhi::CompletionWait* completionWake = nullptr;  // the host's (m_completionWake): requesters wake the thread through it too
	// Whether the host's thread, if it sleeps, sleeps on GPU completions as well as on completionWake: then a submission's
	// message needs no wake of its own (vkSignalSemaphore, several microseconds on the render thread), because the next GPU
	// completion wakes the thread and it reads the mailbox then. Written by the host's thread before its last look at the
	// mailbox, read by a poster after its write: both sequentially consistent, so one of them sees the other.
	std::atomic<bool> wakesOnGpu{false};
	std::atomic<uint64_t> sleepingAt{0};  // the wake version this thread sleeps on, plus one; 0 while awake
	std::atomic<bool> nudgeLogged{false};
	uint64_t wakesSkipped = 0;  // render thread
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
	std::vector<uint8_t> needsTicket;
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
	std::vector<rhi::CommandList> recordedLists;  // SubmitTicket's scratch: the producers' recorded upload lists, then the run's
	uint32_t graphicsSlot = 0;
	rhi::Queue graphicsQueue;
	rhi::TimelineHandle graphicsFence{};

	uint32_t IndexOf(uint32_t epoch) const {
		for (uint32_t i = 0; i < epochs.size(); ++i) if (epochs[i] == epoch) return i;
		throw std::invalid_argument("SubmitEpoch: epoch " + std::to_string(epoch) + " has no ticket");
	}
	// lazyWake: a submission's completion, which the host's thread can take at its next GPU wake-up (wakesOnGpu).
	bool TryPost(Message& message, bool lazyWake = false) {
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
	std::shared_ptr<Ticket> WaitTicket(uint32_t index) {
		BT_ZONE_SCOPE("ORG.Host.WaitTicket");
		// About to block on the host's thread: it must not be left asleep on GPU work that waits for this thread.
		(void)completionWake->Notify();
		for (;;) {
			const auto seen = ticketWake.load(std::memory_order_acquire);
			if (auto ticket = cells[index].exchange(nullptr, std::memory_order_acq_rel)) return ticket;
			ThrowIfFailed();
			// Only this epoch failed: no ticket is coming until its graph is built again.
			if (epochFailed[index].load(std::memory_order_acquire)) throw EpochFailedError(epochs[index], epochErrors[index]);
			// Its copies are all busy with unsealed runs: no ticket is coming before the next seal, which this thread makes. Once the
			// count has moved past the one they were busy at, the host's thread prepares one (the seal woke it).
			if (const auto busy = copiesBusyAt[index].load(std::memory_order_acquire); busy != kNoSeal && busy == seals->load(std::memory_order_acquire))
				throw EpochCopiesBusyError(epochs[index], copyCount);
			ticketWake.wait(seen, std::memory_order_acquire);
		}
	}
};

// The host's recording requests (RequestEpochRecording): any thread pushes onto a Treiber stack, the host's thread takes it whole
// and reverses it into posting order. It and the wake live as long as the host, so a request never meets a build's teardown.
struct GraphHost::RecordingQueue {
	struct Request {
		uint32_t epoch = 0;
		std::shared_ptr<const IHostExecutionData> hostData;
		GraphHost::EpochRecordingCallback done;
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

GraphHost::GraphHost(Desc desc) : m_desc(std::move(desc)), m_recordingQueue(std::make_unique<RecordingQueue>()) {
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
}

GraphHost::~GraphHost() {
	// Nothing queued is completed from here on: its callers may be torn down before the host.
	m_ending = true;
	m_recordingQueue->accepting.store(false, std::memory_order_release);
	StopAsync();
	DestroyGraph();
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
	return m_copyLeases.emplace_back(epoch, std::move(copies)).second;
}

void GraphHost::RetireLease(RunLease& lease, Async* state) {
	// The run's queries were resolved by its submission: read them back now. The runtime's statistics, which every graph shares
	// (a graph's own pointer may be in the middle of a rebuild).
	if (!lease.external && lease.submission)
		if (const auto& stats = m_desc.runtime->Statistics()) {
			rhi::Queue queue = m_device.GetQueue(rhi::QueueKind::Graphics);
			stats->OnFrameComplete(lease.index, queue);
			if (m_completedFrame) m_completedFrame(lease.hostFrame, *stats);
		}
	ReleaseLease(lease, state);
}

void GraphHost::ReleaseLease(RunLease& lease, Async* state) {
	// Its regions in the frame arena (a published copy's are pinned, and stay).
	if (const auto& arena = m_desc.runtime->Arena()) arena->ReleaseRun(lease.index, lease.generation);
	// Its lists: the GPU is done with them, or they were never submitted. Each is reset when it is begun again.
	const auto recycle = [](const std::shared_ptr<CommandListPool>& pool, CommandListPair& pair) {
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
	const auto index = InstanceOf(epoch);
	return index == kNoInstance ? 0 : m_instances[index].buildGeneration;
}

bool GraphHost::EpochRecordingCurrent(const EpochRecording& recording) const noexcept {
	const auto index = InstanceOf(recording.epoch);
	return index != kNoInstance && m_instances[index].graph && recording.buildGeneration == m_instances[index].buildGeneration;
}

void GraphHost::MarkRebuild(Instance& instance, std::string_view reason) noexcept {
	instance.rebuildRequested = true;
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

bool GraphHost::RebuildRequested() const noexcept {
	return std::ranges::any_of(m_instances, [](const Instance& instance) { return instance.rebuildRequested; });
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
	const auto index = InstanceOf(epoch);
	return index != kNoInstance && InstanceFailed(index);
}

void GraphHost::ReportEpochFailure(EpochFailure failure) {
	std::scoped_lock lock(m_failureMutex);
	m_failures.push_back(std::move(failure));
}

size_t GraphHost::PollEpochFailures() {
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
	if (again) {
		RequestRebuildOf(*existing, reason);
		m_extensions.erase(existing);
	}
	m_extensions.push_back({std::move(id), std::move(factory), std::move(epochs)});
	RequestRebuildOf(m_extensions.back(), reason);
}

void GraphHost::RemoveExtension(const std::string& id) {
	const auto existing = std::ranges::find(m_extensions, id, &Registered::id);
	if (existing == m_extensions.end()) return;
	RequestRebuildOf(*existing, "extension '" + id + "' removed");
	m_extensions.erase(existing);
}

runtime::IUploadService* GraphHost::Uploads() noexcept {
	auto* graph = AnyGraph();
	return graph ? graph->GetUploadService() : nullptr;
}

runtime::IDescriptorService* GraphHost::Descriptors() noexcept {
	auto* graph = AnyGraph();
	return graph ? graph->GetDescriptorService() : nullptr;
}

std::shared_ptr<runtime::IUploadService> GraphHost::RetainUploads() noexcept {
	auto* graph = AnyGraph();
	return graph ? graph->RetainUploadService() : nullptr;
}

void GraphHost::DestroyGraph() {
	StopAsync();
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
			retired.graph.reset();
			return true;
		});
	};
	// A graph's teardown touches services the host's thread uses (its upload client's timelines, descriptors): it holds still.
	if (m_async && !m_async->holding) HoldAsync({}, release);
	else release();
}

void GraphHost::SetGpuPassRangeCallbacks(RenderGraph::GpuPassRangeBegin begin, RenderGraph::GpuPassRangeEnd end) {
	m_gpuPassRangeBegin = std::move(begin);
	m_gpuPassRangeEnd = std::move(end);
	for (auto& instance : m_instances)
		if (instance.graph) instance.graph->SetGpuPassRangeCallbacks(m_gpuPassRangeBegin, m_gpuPassRangeEnd);
}

void GraphHost::EnsureTimelines() {
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

bool GraphHost::BuildInstance(size_t index) {
	BT_ZONE_SCOPE("ORG.Host.Build");
	const auto begin = std::chrono::steady_clock::now();
	// Host-wide: without the timelines no epoch can run.
	EnsureTimelines();
	auto& instance = m_instances[index];
	const bool replacing = static_cast<bool>(instance.graph);
	const auto epochName = instance.epoch == persistent::AllEpochs ? std::string("(all)") : std::to_string(instance.epoch);
	// Why this build: the requests since the last one, each with its count. This build answers them, whether it succeeds or not.
	std::string why;
	for (const auto& [reason, count] : instance.rebuildReasons)
		why += (why.empty() ? "" : "; ") + reason + (count > 1 ? " (x" + std::to_string(count) + ")" : std::string());
	instance.rebuildReasons.clear();
	if (why.empty()) why = replacing ? "no request named" : "first build";
	// The graph it replaces: its own submitted work is what every queue has been given so far (the timelines are shared), taken
	// while it is still listed. It is released once that is done, the other epochs running meanwhile.
	auto done = instance.graph ? SubmittedPoint() : GpuPoint{};
	// The new graph is built before the old one stops: a build that fails leaves the epoch its old graph.
	std::unique_ptr<RenderGraph> graph;
	std::string error;
	try {
		graph = std::make_unique<RenderGraph>(m_desc.runtime);
		graph->SetSharedQueueTimelines(m_timelines);
		graph->SetPassEpochFilter(instance.epoch);
		graph->RegisterExtension(std::make_unique<HostUploadExtension>(graph->GetUploadService()), "org.host.io");
		// Only the extensions made into this epoch's graph (AddExtension's epochs): the others are not made, and gather nothing.
		for (const auto& entry : m_extensions)
			if (ExtensionInEpoch(entry, instance.epoch))
				if (auto extension = entry.factory()) graph->RegisterExtension(std::move(extension), entry.id);
		runtime::ScopedActiveGraphServices services(graph->GetUploadService(), graph->GetDescriptorService());
		graph->PrepareExtensionsForBuild();
		// The graph's memory snapshots (its VRAM dump, a host's memory UI): ORG's tracking world, as the contributor host's.
		graph->GetMemorySnapshotProvider().SetProvider(memory::CreateECSMemorySnapshotProvider());
		graph->CompileStructural();
		if (instance.epoch != persistent::AllEpochs) graph->SetPersistentEpochOrder({instance.epoch});
		graph->SetPersistentClosedExecutions(m_desc.closedExecutions);
		graph->SetPersistentExternalHomeStates(m_desc.externalHomeStates);
		if (m_desc.sharedAliasArena) {
			if (!m_aliasArena) m_aliasArena = PersistentAliasArena::Create();
			graph->SetPersistentAliasArena(m_aliasArena);
		}
		graph->SetExternalPersistentEpochs(m_externalEpochs);
		graph->SetPersistentRecordingReuse(m_desc.reuseRecordings);
		graph->SetPersistentRecordingTimestamps(m_recordingTimestamps.load(std::memory_order_acquire));
		graph->SetScheduleTraceEnabled(m_scheduleTrace.load(std::memory_order_acquire));
		{
			auto boundary = m_desc.queueBoundary;
			if (m_boundaryOverridden.load(std::memory_order_acquire)) {
				boundary.entry = m_boundaryEntry.load(std::memory_order_acquire);
				boundary.exit = m_boundaryExit.load(std::memory_order_acquire);
			}
			graph->SetExternalQueueBoundary(boundary);
		}
		graph->SetFrameWaitTimeline(m_frameWaitTimeline);
		graph->Setup();
		graph->SetGpuPassRangeCallbacks(m_gpuPassRangeBegin, m_gpuPassRangeEnd);
	} catch (const std::exception& e) {
		error = e.what();
	} catch (...) {
		error = "unknown exception";
	}
	if (!error.empty() || !graph) {
		if (error.empty()) error = "the graph could not be made";
		// The half-built graph goes now: nothing was submitted with it.
		if (graph) {
			try {
				graph->ShutdownExtensions();
				graph->ShutdownTaskWorkers();
			} catch (...) {
			}
			graph.reset();
		}
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
	if (auto old = std::move(instance.graph)) {
		old->StopFrameProduction();
		m_retiredGraphs.push_back({std::move(old), std::move(done)});
		++m_asyncStats.rebuilds;
	}
	instance.graph = std::move(graph);
	instance.rebuildRequested = false;
	instance.failed = false;
	instance.failure.clear();
	++instance.buildGeneration;
	++m_buildGeneration;
	if (replacing)
		spdlog::info("GraphHost: epoch {}'s graph built again (build {}) in {:.2f} ms; the other epochs' graphs kept; why: {}", epochName,
			instance.buildGeneration, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count(), why);
	return true;
}

bool GraphHost::BuildDueInstances(bool all) {
	std::vector<size_t> due;
	for (size_t index = 0; index < m_instances.size(); ++index) {
		auto& instance = m_instances[index];
		// Placements in a pool generation another epoch's graph replaced (its pool grew): placed again by a rebuild, they share
		// the pool's memory again, and the old generation is freed.
		if (instance.graph && !instance.rebuildRequested && !instance.graph->PersistentAliasPlacementsCurrent()) {
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
	m_frameWaitTimeline = std::move(timeline);
	// Every packet binds its timelines when it is prepared: the graphs' present ones cannot wait on it.
	if (AnyGraph()) RequestRebuild("frame-wait timeline set");
}

GraphHost::GpuPoint GraphHost::SubmittedPoint() const {
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
	auto point = SubmittedPoint();
	{
		std::scoped_lock lock(m_sealedMutex);
		m_sealed = std::move(point);
	}
	m_seals.fetch_add(1, std::memory_order_seq_cst);
	// An epoch whose copies were busy with unsealed runs can take one again: the host's thread prepares its ticket now. The flag is
	// set before the host's thread reads the seals again, so either it sees this count or this sees its flag.
	if (m_async && m_async->anyCopiesBusy.exchange(false, std::memory_order_seq_cst)) (void)m_async->completionWake->Notify();
	// Once a frame: chunks the runs in flight have not needed for a while go.
	if (const auto& arena = m_desc.runtime->Arena()) arena->Trim();
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

std::string GraphHost::DescribeAsyncState() const {
	// Read racily from another thread: a message for a failure, never a decision.
	const auto* async = m_async.get();
	if (!async) return "no async host thread";
	const auto n = [](uint64_t a_value) { return std::to_string(a_value); };
	const uint64_t seals = m_seals.load(std::memory_order_acquire);
	std::string out = "host thread: failed " + n(async->failed.load()) + ", messages written " + n(async->written.load()) + " read " +
		n(async->read.load()) + ", wakes on GPU " + n(async->wakesOnGpu.load()) + ", sleeping on " + n(async->sleepingAt.load()) + " (version + 1), wake version " +
		n(async->completionWake ? async->completionWake->WakeVersion() : 0) + ", requests queued " + n(m_recordingQueue->head.load() != nullptr) +
		", accepting " + n(m_recordingQueue->accepting.load()) + ", seals " + n(seals) + ", requests held " + n(async->pendingRequests.size());
	out += "; epochs (index:epoch paused/needsTicket/ticket/failed/copiesBusy/generation):";
	for (size_t i = 0; i < async->epochs.size(); ++i) {
		const auto busy = async->copiesBusyAt ? async->copiesBusyAt[i].load() : Async::kNoSeal;
		out += " " + n(i) + ":" + n(async->epochs[i]) + " " + n(async->paused[i]) + "/" + n(i < async->needsTicket.size() ? async->needsTicket[i] : 0) +
			"/" + n(async->cells && async->cells[i].load() != nullptr) + "/" + n(async->epochFailed && async->epochFailed[i].load()) + "/" +
			(busy == Async::kNoSeal ? std::string("-") : n(busy)) + "/" + n(async->generation[i]);
	}
	out += "; host timelines (t<index> last assigned/completed):";
	for (const auto& timeline : m_timelines)
		if (timeline.fence && *timeline.fence)
			out += " t" + n((*timeline.fence)->GetHandle().index) + " " + n(timeline.nextValue->load() - 1) + "/" + n((*timeline.fence)->GetCompletedValue());
	// The runs in flight, oldest first, and what each waits for.
	std::scoped_lock lock(m_leaseMutex);
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
				out += " t" + n((*timeline)->GetHandle().index) + " " + n(value) + "/" + n((*timeline)->GetCompletedValue());
		out += "]";
	}
	return out;
}

bool GraphHost::BuildIfRequested() {bool GraphHost::BuildIfRequested() {
	ReleaseRetiredGraphs(false);
	return BuildDueInstances(true);
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
	// Frame pacing: this position's run of framesInFlight frames back (a frame: each epoch of the order once) is waited for and
	// retired. It was submitted to the device by its own Execute, so the wait always ends. Every other run whose GPU work is done
	// retires too (their statistics are read, their memory freed).
	{
		const size_t positions = size_t{(std::max)(m_desc.framesInFlight, 1u)} * (std::max<size_t>)(m_desc.epochOrder.size(), 1);
		if (m_syncLeases.size() != positions) m_syncLeases.assign(positions, nullptr);
		if (auto* previous = std::exchange(m_syncLeases[m_frameNumber % positions], nullptr); previous && previous->state == RunLease::State::Submitted) {
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
		lease.seal = m_seals.load(std::memory_order_relaxed);
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
		const size_t positions = m_syncLeases.size();
		m_syncLeases[m_frameNumber % positions] = &lease;
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
	if (m_asyncEpochs.empty()) {
		m_recordingQueue->accepting.store(false, std::memory_order_release);
		return;
	}
	BuildDueInstances(false);
	StartAsync();
}

void GraphHost::SetExternalEpochs(std::vector<uint32_t> epochs) {
	// An epoch whose graph becomes or stops being external is built again; the rest keep theirs.
	for (const auto& instance : m_instances) {
		const bool was = std::ranges::find(m_externalEpochs, instance.epoch) != m_externalEpochs.end();
		const bool is = std::ranges::find(epochs, instance.epoch) != epochs.end();
		if (was != is) RequestRebuild(instance.epoch, is ? "its epoch made external" : "its epoch no longer external");
	}
	m_externalEpochs = std::move(epochs);
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
	async->copiesBusyAt = std::make_unique<std::atomic<uint64_t>[]>(async->epochs.size());
	for (uint32_t index = 0; index < async->epochs.size(); ++index) async->copiesBusyAt[index].store(Async::kNoSeal, std::memory_order_relaxed);
	async->copiesBusyLogged.assign(async->epochs.size(), 0);
	async->cells = std::make_unique<std::atomic<std::shared_ptr<Async::Ticket>>[]>(async->epochs.size());
	async->needsTicket.assign(async->epochs.size(), 1);
	async->liveRecordings.assign(async->epochs.size(), {});
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
	// Staged uploads go straight into each submission's upload list (RenderGraph::RecordPendingUploads).
	if (auto* uploads = firstGraph->GetUploadService()) uploads->SetStagedUploadsRecordedDirectly(true);
	auto* state = async.get();
	state->thread = std::thread([this, state] {
		const auto graphOf = [&](uint32_t epochIndex) -> RenderGraph& { return *m_instances[state->instanceOf[epochIndex]].graph; };
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
						if (!state->paused[message.epochIndex]) state->needsTicket[message.epochIndex] = 1;
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
							PrepareTicket(*state, message.epochIndex, &lease);
						} else {
							// Never submitted: nothing of it reached the GPU.
							ReleaseLease(lease, state);
							if (!state->paused[message.epochIndex]) state->needsTicket[message.epochIndex] = 1;
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
							state->needsTicket[index] = 0;
							if (auto ticket = state->cells[index].exchange(nullptr, std::memory_order_acq_rel)) {
								(void)graphOf(index).CompletePersistentTicket(*ticket, {});
								ReleaseLease(*m_leaseTable[RenderGraph::PersistentTicketRun(*ticket).lease].load(std::memory_order_acquire), state);
							}
							state->outstanding[index].reset();
							state->liveRecordings[index].clear();
							const auto dropped = std::make_exception_ptr(EpochRecordingDropped(state->generation[index]));
							std::erase_if(state->pendingRequests, [&](const std::unique_ptr<Async::RecordingRequest>& request) {
								if (request->epochIndex != index) return false;
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
								state->needsTicket[index] = 1;
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
					if (state->paused[epochIndex]) {
						// Its graph is being built again: the request is for the old one.
						if (node->done) node->done(nullptr, std::make_exception_ptr(EpochRecordingDropped(state->generation[epochIndex])));
						continue;
					}
					if (state->epochFailed[epochIndex].load(std::memory_order_acquire)) {
						// Its graph failed: nothing is recorded on it until it is built again.
						if (node->done)
							node->done(nullptr, std::make_exception_ptr(EpochFailedError(node->epoch, state->epochErrors[epochIndex])));
						continue;
					}
					auto request = std::make_unique<Async::RecordingRequest>();
					request->epochIndex = epochIndex;
					request->hostData = std::move(node->hostData);
					request->done = std::move(node->done);
					state->pendingRequests.push_back(std::move(request));
				}
				for (uint32_t index = 0; index < state->epochs.size(); ++index)
					if (!state->paused[index]) graphOf(index).RetirePersistentExecutions();
				// Every run whose GPU work is done retires: its statistics are read, and its lists, arena regions, upload staging and
				// owners are freed, even if no later run takes its lease again.
				(void)RetireReachedLeases(state);
				// The next ticket of every epoch whose last one was submitted: one frame ahead.
				bool prepared = false;
				for (uint32_t index = 0; index < state->epochs.size(); ++index) {
					if (!state->needsTicket[index] || state->paused[index] || state->epochFailed[index].load(std::memory_order_relaxed)
						|| state->cells[index].load(std::memory_order_acquire))
						continue;
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
	if (indices.empty()) return;
	Async::Message attach;
	attach.kind = Async::Message::Kind::Attach;
	HoldAsync(indices, [&] {
		auto& async = *m_async;
		for (const auto index : indices) {
			if (BuildInstance(index)) {
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
	auto ticket = async.cells[index].exchange(nullptr, std::memory_order_acq_rel);
	if (!ticket) return false;
	// The host's thread prepares into an empty cell only once the ticket it last put there is submitted or discarded: the cell is the
	// render thread's until the discard is read, so a ticket the mailbox has no room for goes back.
	Async::Message discard;
	discard.kind = Async::Message::Kind::Discard;
	discard.epochIndex = index;
	discard.ticket = ticket;
	if (async.TryPost(discard)) return true;
	async.cells[index].store(std::move(ticket), std::memory_order_release);
	return false;
}

bool GraphHost::RequestEpochRecording(uint32_t epoch, std::shared_ptr<const IHostExecutionData> hostData, EpochRecordingCallback done) {
	// The host's queue alone (never m_async, which a build moves out from under any other thread): the host's thread validates the
	// epoch when it takes it, and a build drops what it did not take (StopAsync, an epoch's Detach), each completed.
	auto& queue = *m_recordingQueue;
	if (!done || !queue.accepting.load(std::memory_order_acquire)) return false;
	auto* node = new RecordingQueue::Request{epoch, std::move(hostData), std::move(done), nullptr};
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
		auto& request = **it;
		std::exception_ptr error;
		try {
			if (state.paused[request.epochIndex]) {
				++it;
				continue;
			}
			// Its graph failed since the request was taken: it completes with that failure.
			if (state.epochFailed[request.epochIndex].load(std::memory_order_relaxed))
				throw EpochFailedError(state.epochs[request.epochIndex], state.epochErrors[request.epochIndex]);
			auto& graph = *m_instances[state.instanceOf[request.epochIndex]].graph;
			if (!request.result) {
				request.result = std::make_shared<EpochRecording>();
				request.result->epoch = state.epochs[request.epochIndex];
				request.result->copies.resize(state.copyCount);
				request.result->revision = request.hostData;
				request.result->buildGeneration = state.generation[request.epochIndex];
				request.recorded.assign(state.copyCount, 0);
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
		if (!error) {
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
	for (auto* copy : copies) {
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

void GraphHost::PrepareTicket(Async& state, uint32_t epochIndex, RunLease* reprepare) {
	BT_ZONE_SCOPE("ORG.Host.PrepareTicket.Worker");
	auto& graph = *m_instances[state.instanceOf[epochIndex]].graph;
	// A stale ticket's reprepare takes the lease it held: its latch regions were written for it. Otherwise a new run lease, or a
	// published copy's for a revision-driven epoch, or none: then nothing is prepared until a seal or a completion frees a copy
	// (EpochCopiesBusyError meanwhile).
	RunLease* lease = reprepare;
	if (!lease) lease = state.revisionDriven[epochIndex] ? TakeCopyLease(state, epochIndex) : &OpenLease(&state);
	if (!lease) return;
	runtime::ScopedActiveGraphServices services(graph.GetUploadService(), graph.GetDescriptorService());
	// The render thread records this ticket's uploads into an open list: resetting and beginning it is several microseconds of
	// driver work that belongs here. Begin takes the list's allocator claim, which the render thread gives up after closing the
	// list (SubmitTicket). A new lease's lists come from the pools; a reprepare's are begun again.
	state.uploadPool->Begin(lease->upload);
	// An external epoch's uploads secondary: recycled with the lease, begun where it is recorded.
	state.uploadSecondaryPool->Begin(lease->uploadSecondary);
	// The run's queue upload lists likewise.
	lease->queueUploads.resize(state.queueUploadKinds.size());
	for (size_t queue = 0; queue < state.queueUploadKinds.size(); ++queue) {
		if (!state.queueUploadKinds[queue]) continue;
		state.queueUploadPools[static_cast<size_t>(*state.queueUploadKinds[queue])]->Begin(lease->queueUploads[queue]);
	}
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
			Async::Preparation preparation;
			const auto version = state.backingVersion.load(std::memory_order_acquire);
			ticket = graph.PreparePersistentTicket(m_device, state.epochs[epochIndex], lease->Ref(), state.hostFrame++, version);
			RenderGraph::SetPersistentTicketHostTag(*ticket, version);
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
	state.needsTicket[epochIndex] = 0;
	state.cells[epochIndex].store(std::move(ticket), std::memory_order_release);
	state.ticketWake.fetch_add(1, std::memory_order_release);
	state.ticketWake.notify_all();
}

void GraphHost::FailAsyncEpoch(Async& state, uint32_t epochIndex, std::string error) {
	const auto epoch = state.epochs[epochIndex];
	spdlog::error("GraphHost: epoch {}'s graph failed to prepare (build {}); the other epochs keep running: {}", epoch,
		state.generation[epochIndex], error);
	// Nothing more is prepared or recorded for it until its instance is attached at a new build.
	state.needsTicket[epochIndex] = 0;
	state.outstanding[epochIndex].reset();
	state.liveRecordings[epochIndex].clear();
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
	if (!run.Published() || run.copy >= recording.copies.size() || !recording.copies[run.copy])
		throw std::logic_error("A revision recording has no copy for its epoch's ticket");
	m_instances[state.instanceOf[epochIndex]].graph->AddPersistentTicketCandidate(ticket, recording.copies[run.copy]);
}

bool GraphHost::CanUseEpochRecording(const EpochRecording& recording, std::string* why) const {
	auto fail = [&](const char* reason) {
		if (why) *why = reason;
		return false;
	};
	if (!m_async || !m_submittingTicket) return fail("not inside an async epoch's submission");
	const auto& async = *m_async;
	if (!async.revisionDriven[m_submittingIndex]) return fail("a live epoch records every run (it takes no revision recording)");
	if (recording.epoch != async.epochs[m_submittingIndex]) return fail("recorded for another epoch");
	const auto run = RenderGraph::PersistentTicketRun(*m_submittingTicket);
	if (!run.Published() || run.copy >= recording.copies.size() || !recording.copies[run.copy]) return fail("no copy for the ticket's run");
	return RenderGraph::CanBindPersistentTicketRecording(*m_submittingTicket, *recording.copies[run.copy],
		async.backingVersion.load(std::memory_order_acquire), why);
}

void GraphHost::NoteNewVersions() noexcept {
	// After the version's publication (release), so a preparation that reads the new value resolves the new version.
	if (m_async) m_async->backingVersion.fetch_add(1, std::memory_order_acq_rel);
}

void GraphHost::StopAsync() {
	BT_ZONE_SCOPE("ORG.Host.StopAsync");
	if (!m_async) return;
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
	// Unsubmitted tickets are abandoned, and their leases given back.
	for (uint32_t index = 0; index < async->epochs.size(); ++index)
		if (auto ticket = async->cells[index].exchange(nullptr)) {
			if (!async->failed.load())
				if (auto& graph = m_instances[async->instanceOf[index]].graph) (void)graph->CompletePersistentTicket(*ticket, {});
			ReleaseLease(*m_leaseTable[RenderGraph::PersistentTicketRun(*ticket).lease].load(std::memory_order_acquire), async.get());
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
		m_asyncStats.wakesSkipped = std::exchange(m_async->wakesSkipped, 0);
		m_asyncStats.copiesBusy = m_async->copiesBusy.exchange(0, std::memory_order_relaxed);
		m_asyncStats.copyWaits = m_async->copyWaits.exchange(0, std::memory_order_relaxed);
	}
	return std::exchange(m_asyncStats, {});
}

void GraphHost::SubmitEpoch(uint32_t epoch, const FrameCallback& beforeSubmit, std::shared_ptr<const void> resourceOwner,
	std::shared_ptr<const EpochRecording> recording) {
	BT_ZONE_SCOPE("ORG.Host.SubmitEpoch");
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
	auto ticket = async.cells[index].exchange(nullptr, std::memory_order_acq_rel);
	if (!ticket) {
		++m_asyncStats.waited;
		ticket = async.WaitTicket(index);
	}
	lap(m_lastTimings.ticketWaitUs);
	SubmitTicket(async, index, std::move(ticket), beforeSubmit, std::move(resourceOwner), std::move(recording));
	// Completed readback captures reach their callbacks here: a ticket's frame never runs Update, which polls them.
	if (auto* readback = m_instances[instance].graph->GetReadbackService()) readback->ProcessReadbackRequests();
}

void GraphHost::SubmitTicket(Async& async, uint32_t index, std::shared_ptr<RenderGraph::PersistentTicket> ticket,
	const FrameCallback& beforeSubmit, std::shared_ptr<const void> resourceOwner, std::shared_ptr<const EpochRecording> recording) {
	BT_ZONE_SCOPE("ORG.Host.SubmitTicket");
	BT_ZONE_VALUE(async.epochs[index]);
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
	const auto run = RenderGraph::PersistentTicketRun(*ticket);
	const uint32_t slot = run.lease;
	// The ticket's lease: published by the host's thread before the ticket, and the render thread's until it posts the ticket back.
	auto& lease = *m_leaseTable[slot].load(std::memory_order_acquire);
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
		m_submittingIndex = index;
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
		postControl(discard);
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
			postControl(discard);
			throw std::logic_error("A revision-driven epoch was submitted without its revision's copy for the run: " + why);
		}
		++m_asyncStats.revisionSubmissions;
	}
	if (!async.revisionDriven[index] && !async.Current(*ticket)) {
		BT_ZONE_SCOPE("ORG.Host.ReprepareStaleTicket");
		m_asyncStats.staleBacking += RenderGraph::PersistentTicketCurrent(*ticket);
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
		postControl(discard);
		ticket = async.WaitTicket(index);
		if (!(RenderGraph::PersistentTicketRun(*ticket) == run))
			throw std::logic_error("A reprepared async epoch ticket is not for the run whose upload list the render thread records");
	}
	// The submitted ticket's frame: one prepared again above has a new number, and the completed-frame callback
	// reports it by that one.
	m_lastHostFrame = RenderGraph::PersistentTicketHostFrame(*ticket);
	lap(m_lastTimings.checkUs);
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
	auto& list = lease.upload;  // begun by the host's thread (PrepareTicket)
	std::shared_ptr<void> keepAlive;
	bool recorded = false;
	// An external epoch's uploads go into the slot's secondary, which its execution runs first: nothing is submitted here.
	const bool external = graph.IsExternalPersistentEpoch(async.epochs[index]);
	auto& secondary = lease.uploadSecondary;
	if (external) {
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
	auto& recordedLists = async.recordedLists;
	// Queue-scoped uploads (Desc::queueScopedUploads): the run's open lists of the other queues, which take the uploads their
	// queue uses first, and go to it just ahead of the ticket's first batch there (SubmitPersistentTicket).
	auto& queueUploadLists = lease.queueUploads;
	const bool queueScoped = m_desc.queueScopedUploads;
	for (size_t queue = 0; queue < async.queueLists.size(); ++queue)
		async.queueLists[queue] = queueScoped && queue < queueUploadLists.size() && queueUploadLists[queue].list
			? queueUploadLists[queue].list.Get() : rhi::CommandList{};
	size_t queueCopies = 0;
	const bool supported = graph.RecordPendingUploads(m_device, commands, static_cast<uint32_t>(slot), keepAlive, recorded, recordedLists,
		queueScoped ? ticket.get() : nullptr, queueScoped ? std::span<rhi::CommandList>(async.queueLists) : std::span<rhi::CommandList>{}, &queueCopies);
	{
		BT_ZONE_SCOPE("ORG.Host.CloseUploadList");
		list.list->End();
		if (external) secondary.list->End();
		// Every one the host's thread began, used or not (they go back to their pools when the run retires).
		for (auto& queueUploads : queueUploadLists)
			if (queueUploads.list) queueUploads.list->End();
		// Closed, so their allocators' claims go. The secondary was reset by the host's thread even when unused.
		list.EndRecording();
		secondary.EndRecording();
		for (auto& queueUploads : queueUploadLists) queueUploads.EndRecording();
	}
	m_asyncStats.queueUploadCopies += queueCopies;
	if (external && !recordedLists.empty()) {
		recordedLists.clear();
		abandonBeforeDiscard();
		Async::Message discard;
		discard.kind = Async::Message::Kind::Discard;
		discard.epochIndex = index;
		discard.ticket = std::move(ticket);
		postControl(discard);
		throw std::runtime_error("An external epoch's uploads include producer-recorded lists, which its execution cannot run");
	}
	if (!supported) {
		abandonBeforeDiscard();
		Async::Message discard;
		discard.kind = Async::Message::Kind::Discard;
		discard.epochIndex = index;
		discard.ticket = std::move(ticket);
		postControl(discard);
		throw std::runtime_error("Async epochs cannot record these uploads (only pointer-targeted buffer copies)");
	}
	Async::Message submitted;
	submitted.kind = Async::Message::Kind::Submitted;
	submitted.number = ++m_submissions;
	// The frame boundaries before it: a later seal flushed it (a published copy waits for its run only then).
	submitted.seal = m_seals.load(std::memory_order_relaxed);
	submitted.epochIndex = index;
	submitted.keepAlive = std::move(keepAlive);
	if (!m_externalRetired.empty()) {
		m_externalRetired.push_back(std::move(retired));
		retired = std::make_shared<std::vector<std::shared_ptr<void>>>(std::exchange(m_externalRetired, {}));
	}
	submitted.retired = std::move(retired);
	submitted.resourceOwner = std::move(resourceOwner);
	if (!external && (recorded || !recordedLists.empty())) {
		BT_ZONE_SCOPE("ORG.Host.EnqueueUploads");
		++m_asyncStats.uploads;
		m_asyncStats.recordedUploadLists += recordedLists.size();
		const auto value = ++async.queueValues.at(async.graphicsSlot);
		const rhi::TimelinePoint signal{async.graphicsFence, value};
		// The producers' recorded lists, then this thread's (an empty one costs nothing but its begin and end).
		recordedLists.push_back(commands);
		rhi::Result uploadResult = rhi::Result::Ok;
		try {
			uploadResult = async.graphicsQueue.Submit({recordedLists.data(), static_cast<uint32_t>(recordedLists.size())}, {{}, {&signal, 1}});
		} catch (...) {
			if (submitted.resourceOwner) m_uncertainExecutionOwners.push_back(std::move(submitted.resourceOwner));
			if (submitted.keepAlive) m_uncertainUploadOwners.push_back(std::move(submitted.keepAlive));
			throw;
		}
		if (uploadResult != rhi::Result::Ok) {
			if (submitted.resourceOwner) m_uncertainExecutionOwners.push_back(std::move(submitted.resourceOwner));
			if (submitted.keepAlive) m_uncertainUploadOwners.push_back(std::move(submitted.keepAlive));
			throw std::runtime_error(std::string("Async epochs could not submit their uploads (rhi::Result ") + rhi::ResultName(uploadResult) + ")");
		}
		submitted.uploadSignal = {async.graphicsSlot, value};
	}
	lap(m_lastTimings.uploadsUs);
	bool ok = false;
	try {
		// Submits and commits, or abandons its admission itself.
		abandonUnsubmitted.ticket = nullptr;
		ok = graph.SubmitPersistentTicket(*ticket, async.queueValues, submitted.submission, submitted.uploadSignal, m_frameWaitValue,
			external && recorded);
	} catch (...) {
		// The queue may have accepted a prefix of this ticket. Completion is
		// unknown, so preserve both owners until device teardown.
		if (submitted.resourceOwner) m_uncertainExecutionOwners.push_back(std::move(submitted.resourceOwner));
		if (submitted.keepAlive) m_uncertainUploadOwners.push_back(std::move(submitted.keepAlive));
		throw;
	}
	publishQueueValues.Publish();
	assigning.reset();
	sequencer.unlock();
	lap(m_lastTimings.submitUs);
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
		m_externalExecution = std::move(execution);
	}
	if (submitted.uploadSignal.second) m_asyncStats.sparedUploadWaits += RenderGraph::PersistentTicketSparedUploadWaits(*ticket);
	submitted.ticket = std::move(ticket);
	try {
		postControl(submitted);
	} catch (...) {
		if (submitted.resourceOwner) m_uncertainExecutionOwners.push_back(std::move(submitted.resourceOwner));
		if (submitted.keepAlive) m_uncertainUploadOwners.push_back(std::move(submitted.keepAlive));
		throw;
	}
	lap(m_lastTimings.postUs);
	++m_asyncStats.submitted;
	if (!ok) throw std::runtime_error("Async epoch submission failed");
}

} // namespace org
