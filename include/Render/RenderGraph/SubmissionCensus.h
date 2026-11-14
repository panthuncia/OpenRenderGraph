#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <span>
#include <vector>

namespace org::diagnostics {

// What epoch submissions make: command lists by kind, queue submissions by kind, and why each submission is not part of the
// one before it. It counts only while enabled (a host's statistics switch). Relaxed counters: the report is approximate when
// several threads submit, and exact on one.
struct SubmissionCensus {
	enum class List : uint8_t {
		Batch,               // recorded ahead, in a packet
		Inline,              // a late batch recorded at submission (RecordInline)
		InlineBarriersOnly,  // of those, every pass skipped: only barriers
		Handoff,             // entry barriers, recorded at submission
		HandoffReused,       // entry barriers, from the slot's cache
		Release,             // releases to another queue
		HomeAcquire,         // host-owned textures acquired back
		Upload,              // the ticket's upload list, begun and closed
		UploadEmpty,         // of those, nothing recorded and nothing submitted
		QueueUpload,         // a queue-scoped upload list, begun and closed
		QueueUploadEmpty,    // of those, no copy routed to it
		Count
	};
	enum class Submission : uint8_t {
		Batches,      // one or more batches (coalesced or not)
		Uploads,      // the ticket's uploads ahead of it
		Release,      // a release list, on its own
		Leading,      // queue-scoped uploads ahead of a batch
		HomeAcquire,  // home acquires, on their own
		Count
	};
	// Why a submission is separate from the one before it (the last one counted here).
	enum class Split : uint8_t {
		OtherQueue,           // the one before went to another queue
		SameTicket,           // the same ticket, the same queue: split by kind or by a packet that cannot gather
		EarlierTicketWaits,   // an earlier ticket on the same queue; this one waits for something the one before did not
		EarlierTicketCovered, // an earlier ticket on the same queue; the one before waited for all this one waits for
		Count
	};
	// A wait on a timeline other than the submission's own queue's.
	enum class WaitKind : uint8_t {
		OtherQueue,  // another queue's timeline
		FrameWait,   // the frame's values (RenderGraph::SetFrameWaitTimeline)
		Other,
		Count
	};
	struct Wait {
		uint64_t timeline = 0;  // (index << 32) | generation
		uint64_t value = 0;
		WaitKind kind = WaitKind::Other;
	};
	enum class HandoffMiss : uint8_t {
		Planned,     // admitted against the ledgers, not a prediction's plan: nothing to key a cache on
		Segments,    // a ticket of several segments
		NewPlan,     // the slot's cache held another plan
		Count
	};

	std::atomic<bool> enabled{false};
	std::array<std::atomic<uint64_t>, size_t(List::Count)> lists{};
	std::array<std::array<std::atomic<uint64_t>, size_t(Split::Count)>, size_t(Submission::Count)> submissions{};
	std::array<std::atomic<uint64_t>, size_t(Submission::Count)> submittedLists{};
	std::array<std::atomic<uint64_t>, size_t(HandoffMiss::Count)> handoffMisses{};
	std::atomic<uint64_t> tickets{0};
	std::atomic<uint64_t> sequencerWaits{0};    // HoldSequencer found the sequencer held by another thread
	std::atomic<uint64_t> sequencerWaitNs{0};
	std::atomic<uint64_t> sequencerHeldNs{0};   // GraphHost::SubmitTicket's hold, admission to commit
	// Waits that a submission following an earlier ticket's on its queue had and the one before did not, by kind.
	std::array<std::atomic<uint64_t>, size_t(WaitKind::Count)> uncoveredWaits{};
	std::mutex lastLock;  // the last submission's queue, ticket and waits
	uint32_t lastQueue = UINT32_MAX;
	uint64_t lastTicket = 0;
	std::vector<Wait> lastWaits;

	bool On() const noexcept { return enabled.load(std::memory_order_relaxed); }
	void Add(List kind, uint64_t count = 1) noexcept {
		if (On() && count) lists[size_t(kind)].fetch_add(count, std::memory_order_relaxed);
	}
	void Miss(HandoffMiss reason) noexcept {
		if (On()) handoffMisses[size_t(reason)].fetch_add(1, std::memory_order_relaxed);
	}
	// A ticket's submission starts: what it submits next is "the same ticket" until the next call.
	void BeginTicket() noexcept {
		if (On()) tickets.fetch_add(1, std::memory_order_relaxed);
	}
	// A queue submission of `listCount` lists on queue slot `queue`, with its waits on timelines other than its queue's.
	void Submitted(Submission kind, uint32_t queue, uint32_t listCount, std::span<const Wait> waits) noexcept {
		if (!On()) return;
		try {
			const uint64_t ticket = tickets.load(std::memory_order_relaxed);
			std::lock_guard lock(lastLock);
			Split split = Split::OtherQueue;
			if (lastQueue == queue && lastTicket == ticket) {
				split = Split::SameTicket;
			} else if (lastQueue == queue) {
				split = Split::EarlierTicketCovered;
				for (const auto& wait : waits) {
					bool covered = false;
					for (const auto& before : lastWaits) covered = covered || (before.timeline == wait.timeline && before.value >= wait.value);
					if (covered) continue;
					split = Split::EarlierTicketWaits;
					uncoveredWaits[size_t(wait.kind)].fetch_add(1, std::memory_order_relaxed);
				}
			}
			submissions[size_t(kind)][size_t(split)].fetch_add(1, std::memory_order_relaxed);
			submittedLists[size_t(kind)].fetch_add(listCount, std::memory_order_relaxed);
			lastQueue = queue;
			lastTicket = ticket;
			lastWaits.assign(waits.begin(), waits.end());
		} catch (...) {
		}
	}
	// Takes the counts and zeroes them. Copies the plain values into `out`.
	struct Snapshot {
		std::array<uint64_t, size_t(List::Count)> lists{};
		std::array<std::array<uint64_t, size_t(Split::Count)>, size_t(Submission::Count)> submissions{};
		std::array<uint64_t, size_t(Submission::Count)> submittedLists{};
		std::array<uint64_t, size_t(HandoffMiss::Count)> handoffMisses{};
		std::array<uint64_t, size_t(WaitKind::Count)> uncoveredWaits{};
		uint64_t sequencerWaits = 0, sequencerWaitNs = 0, sequencerHeldNs = 0;
	};
	Snapshot Take() noexcept {
		Snapshot out;
		for (size_t i = 0; i < lists.size(); ++i) out.lists[i] = lists[i].exchange(0, std::memory_order_relaxed);
		for (size_t k = 0; k < submissions.size(); ++k) {
			for (size_t s = 0; s < submissions[k].size(); ++s) out.submissions[k][s] = submissions[k][s].exchange(0, std::memory_order_relaxed);
			out.submittedLists[k] = submittedLists[k].exchange(0, std::memory_order_relaxed);
		}
		for (size_t i = 0; i < handoffMisses.size(); ++i) out.handoffMisses[i] = handoffMisses[i].exchange(0, std::memory_order_relaxed);
		for (size_t i = 0; i < uncoveredWaits.size(); ++i) out.uncoveredWaits[i] = uncoveredWaits[i].exchange(0, std::memory_order_relaxed);
		out.sequencerWaits = sequencerWaits.exchange(0, std::memory_order_relaxed);
		out.sequencerWaitNs = sequencerWaitNs.exchange(0, std::memory_order_relaxed);
		out.sequencerHeldNs = sequencerHeldNs.exchange(0, std::memory_order_relaxed);
		return out;
	}
};

inline SubmissionCensus& Census() noexcept {
	static SubmissionCensus census;
	return census;
}

} // namespace org::diagnostics
