#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include <rhi.h>

#include "Render/ResourceRequirements.h"

namespace org::persistent { class GraphProgram; class SynchronousAdmission; }
namespace org::compiler { class CompileWorkspace; }

namespace org {

class Runtime;
class Resource;
class CommandListPool;

// Where an external client's submissions go while a host submits its executions on a thread of its own
// (GraphHost::Desc::executeOnSubmissionThread). A submission made on another thread must not reach the queue ahead of an
// execution that thread queued before it: the client records it at once and queues the rest behind the queued work.
class ExternalSubmitRouter {
public:
	virtual ~ExternalSubmitRouter() = default;
	// Whether a submission made on the calling thread now must go behind the thread's queue: something is queued that it must
	// follow (or the calling thread holds entries back, GraphHost::HoldSubmissions). False on the submission thread itself.
	virtual bool MustQueue() const = 0;
	// Pushes `submit` behind what is queued (or with what the calling thread holds). False when the thread takes no more
	// work (the host's end): the caller submits itself then.
	virtual bool Queue(std::function<void()> submit) = 0;
	// Whether the calling thread is the submission thread.
	virtual bool OnThread() const = 0;
};

// Submits GPU work that is recorded outside any graph, such as uploads, readbacks and streaming copies.
// See docs/job-graphs.md, "External clients".
//
// Each submission lists the resources it uses. The client then:
// - finds who used each resource last, and adds the barriers and queue waits needed to continue from there,
// - records the new state of each resource, so the next graph or client starts from it.
//
// Work runs in the order it is submitted, across all graphs and clients on the runtime. So if a client reads
// something a graph writes, submit it after that graph's execution.
//
// Any thread may call it. Submissions are serialized.
//
// With a router (Desc::router, a host's submission thread), a submission made off that thread while it still has work queued
// is recorded at once, on the calling thread, and its admission and queue submission go behind that work. So it follows
// every execution queued before it, as it would without the thread. See docs/persistent-epochs.md, "External client
// submissions behind the submission thread".
class ExternalClient {
public:
	struct Queue {
		rhi::Queue queue;
		rhi::QueueKind kind = rhi::QueueKind::Graphics;
	};
	struct Desc {
		std::shared_ptr<Runtime> runtime;
		// One entry per queue slot. Use the same slot numbers as the graphs on this runtime (QueueRegistry order:
		// graphics, compute, copy, then queues made with CreateQueue). Handoffs between a graph and the client
		// match queues by slot number.
		std::vector<Queue> queues;
		std::string name = "ExternalClient";
		// How many submissions can be in flight. When it is reached, Submit waits for the oldest one to finish.
		size_t capacity = 64;
		// Start each submission with a full memory barrier. Set this when the queue is shared with another API whose
		// earlier work the ledgers don't know about (for example DXVK's).
		bool entryFullBarrier = false;
		// A host's submission thread (ExternalSubmitRouter). Null: every submission is made on the calling thread.
		std::shared_ptr<ExternalSubmitRouter> router;
	};
	struct Access {
		Resource* resource = nullptr;
		RangeSpec range{};
		ResourceState state{};
	};
	// The timeline value a submission signals when it finishes. Wait on it or poll it.
	// A queued submission (Desc::router) returns a value reserved at Submit on a timeline of the client's for queued
	// submissions (one per slot): it is signalled by that submission, once the thread has submitted it. If the queued
	// submission fails, the value and the caller's `signals` are signalled on the queue without the work, so no wait hangs,
	// and the failure is logged. The caller learns of it from `failed`, which is set before the values are signalled, or from
	// the `onQueuedFailure` callback it gave Submit.
	struct Signal {
		rhi::Timeline timeline;
		uint64_t value = 0;
		// Set when a queued submission failed on the submission thread. Null for a submission made at once: that one throws.
		std::shared_ptr<const std::atomic<bool>> failed;
		// Whether the submission failed on the submission thread, so its work never ran. Check it once the value is reached.
		bool Failed() const noexcept { return failed && failed->load(std::memory_order_acquire); }
	};

	explicit ExternalClient(Desc desc);
	ExternalClient(const ExternalClient&) = delete;
	ExternalClient& operator=(const ExternalClient&) = delete;
	// Waits for all submissions to finish.
	~ExternalClient();

	// Submits work on queue slot `slot`. With a router it may be queued behind the host's submission thread (Desc::router).
	// `accesses` lists every resource the work uses, with its range and state. Each resource must already have a backing.
	// `record` records the work. It runs on the calling thread before the barriers are computed, so it must not depend on
	// the state a resource is in beforehand. The barriers are recorded into a separate list that runs first.
	// Throws if the submission fails. Nothing is recorded in the ledgers in that case.
	// `signals` are extra timeline points the submission signals when it finishes.
	// `keepAlive` is held until the submission finishes (for example the staging memory its commands read).
	// `waits` are extra timeline points on this queue's device that the submission waits for before it runs.
	// `cachePublication`: keep what was built from these accesses for a later submission with the same ones. The cache holds
	// the backings it names. A one-off submission whose backings should go once its GPU work is done (a relocation copy)
	// passes false.
	// `onQueuedFailure` is called when the submission was queued behind the submission thread (Desc::router) and then failed
	// there. It is called before the submission's values are signalled. It runs on the submission thread, so it must be safe
	// there: take a lock, or set a flag. A caller that retries a submission that throws uses it to retry a queued one too.
	// A submission made at once throws instead and never calls it.
	Signal Submit(uint32_t slot, std::span<const Access> accesses, const std::function<void(rhi::CommandList&)>& record,
		std::span<const rhi::TimelinePoint> signals = {}, std::shared_ptr<const void> keepAlive = {},
		std::span<const rhi::TimelinePoint> waits = {}, bool cachePublication = true, std::function<void()> onQueuedFailure = {});

	// Work recorded by Record and not yet submitted. Dropping it unsubmitted drops its commands.
	struct Recorded;
	// Submit in two steps, for a caller that records on one thread and submits on another. Record runs `record` now, as Submit
	// does, and takes what the submission needs. SubmitRecorded admits and submits it later, in its place among the runtime's
	// submissions then. The accesses' backings are the ones current at Record. Takes the same arguments as Submit.
	std::shared_ptr<Recorded> Record(uint32_t slot, std::span<const Access> accesses, const std::function<void(rhi::CommandList&)>& record,
		std::span<const rhi::TimelinePoint> signals = {}, std::shared_ptr<const void> keepAlive = {},
		std::span<const rhi::TimelinePoint> waits = {}, bool cachePublication = true, std::function<void()> onQueuedFailure = {});
	// Throws if the submission fails, as Submit does.
	Signal SubmitRecorded(std::shared_ptr<Recorded> recorded);

	// Frees what finished submissions held. Submit also does this.
	void Retire();
	// Stops the background work of its command-list pools. Call it before the task service shuts down.
	void ShutdownTaskWorkers();
	// The queue slots it has submitted on (or made ready), as a bit mask.
	uint32_t UsedSlots();
	// Makes these slots (a bit mask) ready ahead of their first submission: their timelines, and command lists made in the
	// background. A slot is made at its first submission otherwise, and that submission waits for its first list.
	void Prewarm(uint32_t slots);
	// Routes later submissions through `router` (Desc::router); null: none. Submissions already queued are not affected.
	void SetRouter(std::shared_ptr<ExternalSubmitRouter> router);
	// Submissions queued behind a router so far, and those still queued.
	uint64_t QueuedSubmissions() const noexcept { return m_queuedTotal.load(std::memory_order_relaxed); }
	uint32_t PendingQueuedSubmissions() const noexcept { return m_queuedPending.load(std::memory_order_acquire); }
	// Queued submissions that failed on the submission thread so far. Each is also logged, and reported through its
	// Signal::failed and its onQueuedFailure.
	uint64_t FailedQueuedSubmissions() const noexcept { return m_queuedFailed.load(std::memory_order_relaxed); }

private:
	struct SlotState;
	struct PublicationCache;
	void RetireLocked();
	SlotState& Slot(uint32_t slot);
	std::shared_ptr<ExternalSubmitRouter> Router();
	// The submission thread's part of a queued submission.
	void SubmitQueued(const std::shared_ptr<Recorded>& recorded);

	Desc m_desc;
	std::mutex m_mutex;
	std::unique_ptr<persistent::GraphProgram> m_empty;
	std::unique_ptr<persistent::SynchronousAdmission> m_admission;
	std::unique_ptr<compiler::CompileWorkspace> m_workspace;
	std::vector<std::unique_ptr<SlotState>> m_slots;
	std::unique_ptr<PublicationCache> m_publications;
	struct InFlight {
		uint32_t slot = 0;
		uint64_t value = 0;
	};
	std::deque<InFlight> m_inFlight;
	// Queued submissions (Desc::router). m_queueMutex orders the values reserved on m_queuedTimelines with the pushes, so each
	// slot's queued timeline is signalled in increasing order. m_routerMutex guards m_desc.router.
	std::mutex m_routerMutex;
	std::mutex m_queueMutex;
	std::vector<rhi::TimelinePtr> m_queuedTimelines;  // by slot, made at the slot's first queued submission
	std::vector<uint64_t> m_queuedValues;             // by slot: the last value reserved
	std::atomic<uint64_t> m_queuedTotal{0};
	std::atomic<uint32_t> m_queuedPending{0};
	std::atomic<uint64_t> m_queuedFailed{0};
	std::mutex m_queuedDoneMutex;
	std::condition_variable m_queuedDone;
};

} // namespace org
