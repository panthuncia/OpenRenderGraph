// Persistent execution, tickets (async epochs): preparation, kept recordings, submission and pending uploads.
#include "RenderGraphPersistentState.h"
#include "Resources/Buffers/VersionedBuffer.h"
#include "Render/Runtime/StagedUploadBatch.h"

namespace org {

std::shared_ptr<RenderGraph::PersistentTicket> RenderGraph::PreparePersistentTicket(rhi::Device device, uint32_t epoch, RunRef run, uint64_t hostFrame,
	uint64_t hostTag, std::shared_ptr<const IHostExecutionData> hostData, bool admissionOnly) {
	BT_ZONE_SCOPE("ORG.Persistent.PrepareTicket");
	const uint32_t slot = run.lease;
	auto ticket = std::make_shared<PersistentTicket>();
	ticket->epoch = epoch;
	ticket->slot = slot;
	ticket->run = run;
	ticket->hostFrame = hostFrame;
	ticket->hostTag = hostTag;
	ticket->external = IsExternalPersistentEpoch(epoch);
	const PersistentExecutionOptions options{.ticket = ticket.get(), .hostTag = hostTag, .admissionOnly = admissionOnly,
		.hostData = static_cast<bool>(hostData), .run = run};
	SetPersistentEpoch(epoch);
	UpdateExecutionContext update{};
	update.frameIndex = slot;
	update.preparationSlot = slot;
	update.frameFenceValue = hostFrame + 1;
	update.hostData = hostData.get();
	update.ownedHostData = hostData;
	// What it prepares for resolves the resolvers too: a revision names its resources (its versions of them), never the live ones.
	if (hostData) update.resolverCaptureContext = std::make_shared<const ResolverCaptureContext>(hostData);
	Update(update, device, options);
	PassExecutionContext execute{};
	execute.device = device;
	execute.frameIndex = slot;
	execute.frameFenceValue = hostFrame + 1;
	execute.hostData = hostData.get();
	execute.ownedHostData = hostData;
	Execute(execute);
	return ticket;
}

std::shared_ptr<RenderGraph::PersistentTicket> RenderGraph::PrepareRevisionTicket(uint32_t epoch, RunRef run, uint64_t hostFrame, bool retire) {
	BT_ZONE_SCOPE("ORG.Persistent.PrepareRevisionTicket");
	auto ticket = std::make_shared<PersistentTicket>();
	ticket->epoch = epoch;
	ticket->slot = run.lease;
	ticket->run = run;
	ticket->hostFrame = hostFrame;
	ticket->revision = true;
	ticket->external = IsExternalPersistentEpoch(epoch);
	// What Execute does before it admits: release the executions the GPU has finished.
	if (retire) RetirePersistentExecutions();
	return ticket;
}

void RenderGraph::AddPersistentTicketCandidate(PersistentTicket& ticket, std::shared_ptr<const PersistentRecording> recording) {
	BT_ZONE_SCOPE("ORG.Persistent.AddTicketCandidate");
	if (!recording || !recording->replayable || !recording->publication)
		throw std::logic_error("A revision ticket's candidate must be a replayable recording");
	for (const auto* candidate = ticket.candidates.load(std::memory_order_relaxed); candidate; candidate = candidate->next)
		if (candidate->segment.recording == recording) return;
	// Admitted with the ticket at submission (AdmitPersistentTicket), if the bind takes it.
	auto* candidate = new PersistentTicket::Candidate{};
	candidate->segment.admission.publication = recording->publication;
	candidate->segment.recording = recording;
	candidate->segment.bindings = recording->bindings;
	candidate->segment.resourceHeap = recording->resourceHeap;
	candidate->segment.samplerHeap = recording->samplerHeap;
	candidate->segment.waits = recording->waits;
	candidate->segment.rebindings = recording->rebindings;
	candidate->next = ticket.candidates.load(std::memory_order_relaxed);
	ticket.candidates.store(candidate, std::memory_order_release);  // the owner thread is the only writer
}

std::vector<std::pair<uint32_t, uint64_t>> RenderGraph::CompletePersistentTicket(PersistentTicket& ticket, const PersistentTicketSubmission& submission) {
	BT_ZONE_SCOPE("ORG.Persistent.CompleteTicket");
	auto& state = *m_compilerState->persistent;
	std::vector<std::pair<uint32_t, uint64_t>> completion;
	size_t next = 0;
	for (auto& segment : ticket.segments) {
		const auto& graph = *segment.admission.publication->executable->graph;
		if (!submission.submitted || !segment.recording || !segment.committed) {
			if (segment.admitted && !segment.committed) state.admission->Abandon(segment.admission);
			segment.admitted = false;
			continue;
		}
		std::vector<compiler::ExecutionTimelinePoint> signals;
		signals.reserve(graph.batches.size());
		for (uint32_t batch = 0; batch < graph.batches.size(); ++batch)
			signals.push_back({segment.admission.queueTimelines.at(graph.batches[batch].queue), submission.signals.at(next++)});
		// Committed by the submission (SubmitPersistentTicket); recorded here for retirement.
		(void)state.timelines->RecordSubmitted(segment.recording->snapshot->layout->bundle,
			segment.admission.incomingWaits, segment.recording->packets, signals);
		for (const auto& statistics : segment.recording->statistics) statistics->Publish();
		for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
			const auto queue = graph.batches[batch].queue;
			m_queueRegistry.EnsureNextFenceValueAtLeast(static_cast<QueueSlotIndex>(static_cast<uint8_t>(queue)), signals[batch].value + 1);
			auto found = std::ranges::find(completion, queue, &std::pair<uint32_t, uint64_t>::first);
			if (found == completion.end()) completion.emplace_back(queue, signals[batch].value);
			else found->second = (std::max)(found->second, signals[batch].value);
		}
	}
	if (submission.submitted && m_scheduleTraceEnabled.load(std::memory_order_relaxed)) RecordPersistentTicketScheduleSubmissions(ticket);
	if (ticket.execution) {
		if (submission.submitted) {
			std::vector<ExecutionSignal> signals;
			for (const auto& segment : ticket.segments) {
				if (!segment.recording) continue;
				const auto& graph = *segment.admission.publication->executable->graph;
				for (uint32_t batch = 0; batch < graph.batches.size(); ++batch)
					signals.push_back({segment.admission.queueTimelines.at(graph.batches[batch].queue), 0});
			}
			for (size_t i = 0; i < signals.size(); ++i) signals[i].value = submission.signals.at(i);
			ticket.execution->MarkSubmitted(std::move(signals));
			state.pendingExecutions.push_back(ticket.execution);
		} else {
			ticket.execution->Finish(true);
		}
	}
	for (auto& segment : ticket.displaced)
		if (segment.recording) ticket.replaced.push_back(std::move(segment.recording));
	ticket.displaced.clear();
	// Abandoned packets are destroyed here, never submitted: their lifecycle effects abandon and their
	// command lists return to the pool. A revision's recording stays with its caller.
	ticket.segments.clear();
	ticket.replaced.clear();
	basic_telemetry::AddCounter(submission.submitted ? "ORG.Persistent.TicketsSubmitted" : "ORG.Persistent.TicketsAbandoned");
	return completion;
}

uint32_t RenderGraph::PersistentTicketEpoch(const PersistentTicket& ticket) noexcept { return ticket.epoch; }
RunRef RenderGraph::PersistentTicketRun(const PersistentTicket& ticket) noexcept { return ticket.run; }
uint64_t RenderGraph::PersistentTicketHostFrame(const PersistentTicket& ticket) noexcept { return ticket.hostFrame; }
ExecutionHandle RenderGraph::PersistentTicketExecution(const PersistentTicket& ticket) noexcept { return ExecutionHandle(ticket.execution); }
uint64_t RenderGraph::PersistentTicketHostTag(const PersistentTicket& ticket) noexcept { return ticket.hostTag; }
void RenderGraph::SetPersistentTicketHostTag(PersistentTicket& ticket, uint64_t tag) noexcept { ticket.hostTag = tag; }
uint32_t RenderGraph::PersistentTicketSparedUploadWaits(const PersistentTicket& ticket) noexcept {
	return ticket.queueUploads.planned ? ticket.queueUploads.spared : 0;
}
RenderGraph::TicketUploadStats RenderGraph::PersistentTicketUploadStats(const PersistentTicket& ticket) noexcept {
	const auto& plan = ticket.queueUploads;
	if (!plan.planned) return {};
	return {plan.lateRouted, plan.lateWaits, plan.undeclared};
}

std::shared_ptr<const RenderGraph::PersistentRecording> RenderGraph::PersistentTicketRecording(const PersistentTicket& ticket) noexcept {
	return ticket.segments.size() == 1 ? ticket.segments.front().recording : nullptr;
}

bool RenderGraph::PersistentRecordingReplayable(const PersistentRecording& recording) noexcept {
	return recording.replayable;
}

bool RenderGraph::BindPersistentTicketRecording(PersistentTicket& ticket, std::shared_ptr<const PersistentRecording> recording, uint64_t hostTag,
	std::string* why) {
	auto fail = [&](const char* reason) {
		if (why) *why = reason;
		return false;
	};
	if (!recording) return fail("no recording");
	if (ticket.revision) {
		if (!ticket.segments.empty()) return fail("the ticket already has a recording");
		for (const auto* candidate = ticket.candidates.load(std::memory_order_acquire); candidate; candidate = candidate->next) {
			if (candidate->segment.recording != recording) continue;
			recording->lastUse.store(ticket.hostFrame, std::memory_order_relaxed);
			ticket.segments.push_back(candidate->segment);
			ticket.chosen = candidate;
			ticket.revisions = recording->revisions;
			return true;
		}
		return fail("the ticket admitted no candidate for this recording (made after the ticket, and not added to it)");
	}
	// A live ticket's candidate (a revision's recording admitted beside the ticket's own): it displaces the ticket's segment.
	for (const auto* candidate = ticket.candidates.load(std::memory_order_acquire); candidate; candidate = candidate->next) {
		if (candidate->segment.recording != recording) continue;
		if (ticket.chosen) return fail("the ticket already took a candidate");
		recording->lastUse.store(ticket.hostFrame, std::memory_order_relaxed);
		for (auto& segment : ticket.segments) ticket.displaced.push_back(std::move(segment));
		ticket.segments.clear();
		ticket.segments.push_back(candidate->segment);
		ticket.chosen = candidate;
		ticket.revisions = recording->revisions;
		return true;
	}
	if (ticket.segments.size() != 1) return fail("the ticket has no single segment");
	auto& segment = ticket.segments.front();
	if (segment.recording) return fail("the ticket already has a recording");
	if (!recording->replayable) return fail("the recording is not replayable");
	if (recording->publication != segment.admission.publication) return fail("recorded for another publication");
	if (recording->bindings != segment.bindings) return fail("recorded for other bindings");
	if (recording->resourceHeap.index != segment.resourceHeap.index || recording->resourceHeap.generation != segment.resourceHeap.generation
		|| recording->samplerHeap.index != segment.samplerHeap.index || recording->samplerHeap.generation != segment.samplerHeap.generation)
		return fail("recorded for other descriptor heaps");
	if (recording->hostTag != hostTag) return fail("recorded under another backing version");
	recording->lastUse.store(ticket.hostFrame, std::memory_order_relaxed);
	segment.recording = std::move(recording);
	ticket.revisions = segment.recording->revisions;
	ticket.hostTag = hostTag;
	return true;
}

bool RenderGraph::CanBindPersistentTicketRecording(const PersistentTicket& ticket, const PersistentRecording& recording, uint64_t hostTag,
	std::string* why) {
	auto fail = [&](const char* reason) {
		if (why) *why = reason;
		return false;
	};
	const auto admitted = [&] {
		for (const auto* candidate = ticket.candidates.load(std::memory_order_acquire); candidate; candidate = candidate->next)
			if (candidate->segment.recording.get() == &recording) return true;
		return false;
	};
	if (ticket.revision) {
		if (!ticket.segments.empty()) return fail("the ticket already has a recording");
		return admitted() || fail("the ticket admitted no candidate for this recording (made after the ticket, and not added to it)");
	}
	if (admitted()) return !ticket.chosen || fail("the ticket already took a candidate");
	if (ticket.segments.size() != 1) return fail("the ticket has no single segment");
	const auto& segment = ticket.segments.front();
	if (segment.recording) return fail("the ticket already has a recording");
	if (!recording.replayable) return fail("the recording is not replayable");
	if (recording.publication != segment.admission.publication) return fail("recorded for another publication");
	if (recording.bindings != segment.bindings) return fail("recorded for other bindings");
	if (recording.resourceHeap.index != segment.resourceHeap.index || recording.resourceHeap.generation != segment.resourceHeap.generation
		|| recording.samplerHeap.index != segment.samplerHeap.index || recording.samplerHeap.generation != segment.samplerHeap.generation)
		return fail("recorded for other descriptor heaps");
	if (recording.hostTag != hostTag) return fail("recorded under another backing version");
	return true;
}

RenderGraph::PersistentRecordingStats RenderGraph::TakePersistentRecordingStats() noexcept {
	PersistentRecordingStats stats;
	stats.recorded = m_recordingsRecorded.exchange(0, std::memory_order_relaxed);
	if (m_compilerState && m_compilerState->persistent && m_compilerState->persistent->admission)
		std::tie(stats.predicted, stats.planned) = m_compilerState->persistent->admission->TakePredictionCounts();
	return stats;
}

std::string RenderGraph::PersistentTicketStaleReason(const PersistentTicket& ticket) {
	if (ticket.uncheckable) return "uncheckable: " + ticket.uncheckableBy;
	for (size_t i = 0; i < ticket.revisions.size(); ++i) {
		const auto& [pass, hash] = ticket.revisions[i];
		const std::string name = i < ticket.revisionNames.size() ? ticket.revisionNames[i] : std::string("?");
		uint64_t now = 0;
		if (!pass->RevisionHash(now)) return "revision hash gone: " + name;
		if (now != hash) return "revision changed: " + name;
	}
	return "current";
}

bool RenderGraph::PersistentRecordingVersionsCurrent(const PersistentRecording& recording) noexcept {
	for (const auto& [versioned, number] : recording.versions) {
		const auto current = versioned ? versioned->SelectVersion(nullptr) : nullptr;
		if (!current || current->number != number) return false;
	}
	return true;
}

std::string RenderGraph::PersistentRecordingStaleVersion(const PersistentRecording& recording) {
	for (const auto& [versioned, number] : recording.versions) {
		const auto current = versioned ? versioned->SelectVersion(nullptr) : nullptr;
		if (current && current->number == number) continue;
		const std::string name = current && current->resource ? current->resource->GetName() : std::string("?");
		return "'" + name + "' (slot " + std::to_string(versioned ? versioned->SlotID() : 0) + ", version " + std::to_string(current ? current->number : 0)
			+ ", recorded with " + std::to_string(number) + ")";
	}
	return {};
}

RenderGraph::PersistentTicketKey RenderGraph::PersistentTicketKeyOf(const PersistentTicket& ticket) noexcept {
	PersistentTicketKey key;
	const auto mix = [](uint64_t a_hash, const void* a_pointer) {
		return (a_hash ^ reinterpret_cast<uintptr_t>(a_pointer)) * 1099511628211ull;
	};
	key.recordings = key.plans = 1469598103934665603ull;
	const auto add = [&](const PersistentTicket::Segment& a_segment) {
		key.recordings = mix(key.recordings, a_segment.recording.get());
		key.plans = mix(key.plans, a_segment.admission.sharedBarriers.get());
		key.planned = key.planned && a_segment.admission.predicted;
		const auto& plan = a_segment.admission.Barriers();
		for (const auto& batch : plan.batches) key.entryBarriers += batch.textures.size() + batch.buffers.size() + batch.acquired.size();
		for (const auto& release : plan.releases) key.entryBarriers += release.textures.size();
		if (a_segment.recording) {
			key.bodyBarriers += a_segment.recording->bodyBarriers;
			key.bodyFullBarriers += a_segment.recording->bodyFullBarriers;
		}
	};
	// A bound candidate's segment is in segments by now (its own went to displaced).
	for (const auto& segment : ticket.segments) add(segment);
	return key;
}

bool RenderGraph::PersistentTicketCurrent(const PersistentTicket& ticket) {
	BT_ZONE_SCOPE("ORG.Persistent.CheckTicket");
	if (ticket.uncheckable) return false;
	for (const auto& [pass, hash] : ticket.revisions) {
		uint64_t now = 0;
		if (!pass->RevisionHash(now) || now != hash) return false;
	}
	return true;
}

namespace {
// A submitted ticket segment's commit, under the sequencer hold its submission took: the next admission (this graph's or
// another's) starts from the states it leaves.
void CommitTicketSegment(persistent::SynchronousAdmission& admission, RenderGraph::PersistentTicket::Segment& segment,
	std::span<const compiler::ExecutionBatchTimeline> batches) {
	compiler::GraphExecutionTimeline receipt;
	receipt.batches.reserve(batches.size());
	for (const auto& batch : batches) receipt.batches.push_back({batch.signal, batch.waits, {}, {}});
	admission.Commit(segment.admission, receipt);
	segment.committed = true;
}
}

std::unique_lock<std::recursive_mutex> RenderGraph::HoldPersistentSequencer() const {
	BT_ZONE_SCOPE("ORG.Persistent.HoldSequencer");
	if (!m_compilerState || !m_compilerState->persistent || !m_compilerState->persistent->admission)
		throw std::logic_error("The persistent sequencer before the graph executed");
	return m_compilerState->persistent->admission->HoldSequencer();
}

void RenderGraph::AdmitPersistentTicket(PersistentTicket& ticket, std::span<const uint64_t> queueValues) {
	BT_ZONE_SCOPE("ORG.Persistent.AdmitTicket");
	auto& state = *m_compilerState->persistent;
	const size_t slotCount = m_queueRegistry.SlotCount();
	if (queueValues.size() < slotCount) throw std::logic_error("A ticket admitted without every queue's submitted value");
	// An open admission commits before the next is prepared: a ticket's segments are admitted together, so only closed
	// executions can have several.
	if (!state.admission->ClosedExecutions() && ticket.segments.size() > 1)
		throw std::logic_error("An open execution's ticket with several segments");
	std::vector<compiler::ExecutionTimelinePoint> queues;
	queues.reserve(slotCount);
	for (size_t slot = 0; slot < slotCount; ++slot) queues.push_back({QueueTimelineIdentity(*m_runtime, m_queueRegistry, slot), queueValues[slot]});
	for (auto& segment : ticket.segments) {
		if (segment.admitted) throw std::logic_error("A ticket admitted twice");
		auto publication = segment.admission.publication;
		if (!publication) throw std::logic_error("A ticket's segment without a publication");
		// Its body was recorded from a plan of this publication and these rebindings: the ledgers' part alone.
		segment.admission = state.admission->PrepareAtSubmission(std::move(publication), queues, segment.waits, segment.rebindings);
		segment.admitted = true;
	}
}

void RenderGraph::AbandonPersistentTicketAdmission(PersistentTicket& ticket) {
	auto& state = *m_compilerState->persistent;
	for (auto& segment : ticket.segments) {
		if (segment.admitted && !segment.committed) state.admission->Abandon(segment.admission);
		segment.admitted = segment.committed;
	}
}

// A group of batches held back for the next ticket to join (SubmitPersistentTicket's carry). Its batches are committed as
// submitted when they are held. It keeps the recordings that own its lists.
struct RenderGraph::SubmitCarry {
	rhi::Queue queue{};
	rhi::TimelineHandle own{};  // the queue's timeline
	uint32_t queueSlot = UINT32_MAX;
	uint64_t firstSignal = 0;   // the held group's first signal on its own timeline
	compiler::PreparedRhiExecutionBatch::GatheredSubmission gathered;
	std::vector<std::shared_ptr<const void>> keep;  // the tickets' recordings
	uint32_t tickets = 0;  // the tickets whose groups it holds
	uint64_t joins = 0;    // statistics: groups joined since the last TakeSubmitCarryJoins
	bool Active() const noexcept { return !gathered.lists.empty(); }
	void Clear() {
		gathered.Clear();
		keep.clear();
		tickets = 0;
		firstSignal = 0;
		queueSlot = UINT32_MAX;
	}
};

std::shared_ptr<RenderGraph::SubmitCarry> RenderGraph::MakeSubmitCarry() { return std::make_shared<SubmitCarry>(); }

uint64_t RenderGraph::TakeSubmitCarryJoins(SubmitCarry& carry) noexcept { return std::exchange(carry.joins, 0); }

bool RenderGraph::FlushSubmitCarry(SubmitCarry& carry, uint32_t* merged) {
	if (merged) *merged = carry.tickets;
	if (!carry.Active()) return true;
	BT_ZONE_SCOPE("ORG.Persistent.FlushSubmitCarry");
	auto& census = diagnostics::Census();
	const auto submitted = carry.queue.Submit({carry.gathered.lists.data(), static_cast<uint32_t>(carry.gathered.lists.size())},
		{{carry.gathered.waits.data(), static_cast<uint32_t>(carry.gathered.waits.size())},
		 {carry.gathered.signals.data(), static_cast<uint32_t>(carry.gathered.signals.size())}});
	const bool ok = submitted == rhi::Result::Ok;
	if (ok) {
		basic_telemetry::AddCounter("ORG.Persistent.CoalescedSubmissions");
		if (census.On()) {
			// Its waits are all on other timelines than its queue's (the held group's own were dropped), so they count as such.
			std::vector<diagnostics::SubmissionCensus::Wait> waits;
			for (const auto& wait : carry.gathered.waits)
				waits.push_back({(uint64_t{wait.t.index} << 32) | wait.t.generation, wait.value, diagnostics::SubmissionCensus::WaitKind::Other});
			census.Submitted(diagnostics::SubmissionCensus::Submission::Batches, carry.queueSlot, static_cast<uint32_t>(carry.gathered.lists.size()), waits);
		}
	}
	carry.Clear();
	return ok;
}

bool RenderGraph::SubmitPersistentTicket(PersistentTicket& ticket, std::span<uint64_t> nextQueueValues, PersistentTicketSubmission& out,
	std::pair<uint32_t, uint64_t> uploadsDone, uint64_t frameWait, bool streamUploads, SubmitCarry* carry, bool carryOut, TicketUploads* uploads) {
	BT_ZONE_SCOPE("ORG.Persistent.SubmitTicket");
	out = {};
	auto& state = *m_compilerState->persistent;
	using Census = diagnostics::SubmissionCensus;
	auto& census = diagnostics::Census();
	// The admission (AdmitPersistentTicket), the handoff and the commit with nothing admitted in between.
	auto hold = state.admission->HoldSequencer();
	struct AbandonUncommitted {
		RenderGraph& graph;
		PersistentTicket& ticket;
		~AbandonUncommitted() { graph.AbandonPersistentTicketAdmission(ticket); }
	} abandonUncommitted{*this, ticket};
	ticket.submittedUploads = uploadsDone;
	ticket.submittedFrameWait = frameWait;
	const auto& planned = ticket.queueUploads;
	if (planned.planned && ticket.segments.size() != 1) throw std::logic_error("Queue-scoped uploads planned for a ticket of several segments");
	const auto graphicsSlot = static_cast<uint32_t>(ToUnderlying(m_queueRegistry.FindGraphicsSlot()));
	// The caller's upload lists. They ride in the ticket's first queue submission when every batch is on their queue, the ticket has
	// one segment and no release, and its first batch can be gathered. Otherwise they are submitted on their own, before anything
	// else of the ticket, as the caller did before.
	bool uploadsPending = uploads && !uploads->lists.empty();
	if (uploadsPending) {
		bool ride = m_uploadsWithBatches && m_coalesceSubmissions && !ticket.external && ticket.segments.size() == 1;
		if (ride) {
			const auto& segment = ticket.segments.front();
			const auto& graph = *segment.admission.publication->executable->graph;
			const auto& packets = segment.recording->packets;
			ride = !graph.batches.empty() && packets.size() == graph.batches.size() && segment.admission.Barriers().releases.empty()
				&& std::ranges::all_of(graph.batches, [&](const auto& batch) { return batch.queue == uploads->queueSlot; })
				&& dynamic_cast<const compiler::PreparedRhiExecutionBatch*>(packets.front().get()) != nullptr
				&& !std::ranges::any_of(planned.leading, [](const auto& leading) { return leading.batch == 0 && leading.copies; });
		}
		if (!ride) {
			uploadsPending = false;
			if (carry && !FlushSubmitCarry(*carry)) return false;
			rhi::SubmitDesc submit{};
			submit.signals = {&uploads->signal, 1u};
			if (uploads->queue.Submit({uploads->lists.data(), static_cast<uint32_t>(uploads->lists.size())}, submit) != rhi::Result::Ok) return false;
			census.Submitted(Census::Submission::Uploads, uploads->queueSlot, static_cast<uint32_t>(uploads->lists.size()), {});
		}
	}
	for (auto& segment : ticket.segments) {
		if (!segment.admitted) throw std::logic_error("A ticket submitted before it was admitted (AdmitPersistentTicket)");
		auto& admission = segment.admission;
		const auto& graph = *admission.publication->executable->graph;
		const auto batchCount = static_cast<uint32_t>(graph.batches.size());
		// The resources no running pass of this execution touches (an open ticket the graph submits itself): every pass that uses
		// one skips the run, so it stays as the execution found it. Its entry barriers and releases are left out here, its body
		// barriers by the lists (left out ahead when every use was skipped ahead, else at execution), and the commit leaves its
		// ledger state as it was (FrameAdmission::unmoved). The late batches' answers are asked once, here, and passed on.
		std::vector<std::vector<uint8_t>> skips(batchCount);
		compiler::UntouchedResources untouched;
		if (!ticket.external && !admission.closed && !state.admission->ExternalHomeStates()
			&& !diagnostics::DiagnosticKeepSkippedPassBarriers().load(std::memory_order_relaxed)) {
			BT_ZONE_SCOPE("ORG.Persistent.TicketUntouched");
			const auto& packets = segment.recording->packets;
			const auto* ahead = segment.recording->ahead.get();
			std::vector<uint8_t> late(batchCount);
			bool lateSkips = false;
			for (uint32_t batch = 0; batch < batchCount; ++batch) {
				const auto* packet = batch < packets.size() ? dynamic_cast<const compiler::PreparedRhiExecutionBatch*>(packets[batch].get()) : nullptr;
				if (packet && packet->Inline()) {
					packet->AskSkips(skips[batch]);
					late[batch] = 1;
					lateSkips = lateSkips || std::ranges::any_of(skips[batch], [](uint8_t skip) { return skip != 0; });
				} else if (ahead) {
					// Recorded ahead: only a pass skipped ahead skips.
					const auto& snapshot = *segment.recording->snapshot;
					for (const auto pass : graph.batches[batch].passes) {
						const auto prepared = graph.structure->passes.at(pass).preparedPassIndex;
						skips[batch].push_back(prepared < snapshot.passes.size() && snapshot.passes[prepared].SkippedAhead() ? 1 : 0);
					}
				}
			}
			if (lateSkips || ahead) {
				// A frame like the last one (the same answers, the same backings) finds the same resources.
				const auto& recording = *segment.recording;
				thread_local std::vector<uint8_t> flat;
				flat.clear();
				for (const auto& batch : skips) flat.insert(flat.end(), batch.begin(), batch.end());
				if (recording.untouchedBackings != admission.backings.Id() || recording.untouchedSkips != flat) {
					recording.untouched = compiler::FindUntouched(graph, admission.backings, skips, late, ahead);
					recording.untouchedSkips = flat;
					recording.untouchedBackings = admission.backings.Id();
				}
				untouched = recording.untouched;
			}
			admission.unmoved = untouched.indices;
		}
		// A release's textures less the untouched ones.
		std::vector<rhi::TextureBarrier> releasedScratch;
		const auto released = [&](const compiler::PreparedExecutionBarrierPlan::Release& release) -> std::span<const rhi::TextureBarrier> {
			if (untouched.Empty()) return release.textures;
			releasedScratch.clear();
			for (const auto& texture : release.textures)
				if (!untouched.Holds(texture.texture)) releasedScratch.push_back(texture);
			diagnostics::UntouchedEntryBarriers().fetch_add(release.textures.size() - releasedScratch.size(), std::memory_order_relaxed);
			return releasedScratch;
		};
		// An external epoch's batches on the host's stream's queue (InHostStream) go to its host's stream; the rest are submitted here.
		std::vector<uint8_t> streamSlots(m_queueRegistry.SlotCount());
		for (uint32_t slot = 0; slot < streamSlots.size(); ++slot) streamSlots[slot] = ticket.external && InHostStream(slot) ? 1 : 0;
		const auto inStream = [&](uint32_t batch) {
			const auto queue = graph.batches[batch].queue;
			return queue < streamSlots.size() && streamSlots[queue] != 0;
		};
		// The same timeline composition the owner's RecordSubmitted will derive from these values: the
		// execution's incoming waits plus its batches' waits on each other.
		std::vector<compiler::ExecutionBatchTimeline> batches(batchCount);
		auto incoming = admission.incomingWaits;
		// An open execution's handoff, against the ledgers as admitted now: textures an earlier execution left on another
		// queue are released there first (the consumers wait for it), and each batch's entry barriers go ahead of it: in a list
		// submitted with it, or (in the host's stream) in a secondary its segment runs first. A resource's entry is its first use,
		// so no earlier batch touches it.
		std::vector<rhi::CommandList> handoff(batchCount);
		std::vector<rhi::CommandList> streamHandoff(batchCount);
		// Releases on a queue the host's stream runs: in a secondary its first segment runs, with a value of its own on the graphics
		// timeline that the consumers off the stream wait for.
		std::vector<rhi::CommandList> streamReleases;
		uint64_t streamReleasesSignal = 0;
		std::vector<uint32_t> streamReleaseConsumers;
		const auto streamSecondary = [&](uint32_t slot, const char* name) {
			auto allocation = std::make_shared<compiler::FrameCommandAllocation>();
			allocation->pool = ExternalSecondaryPool(slot);
			allocation->pair = allocation->pool->Request();
			rhi::SecondaryInheritance outside{};
			outside.insideRendering = false;
			outside.debugName = name;
			// Run by the host's command buffers, and again from the handoff cache: never chained into one, as the epoch's
			// own secondaries (RecordPreparedRhiExecutionBatch).
			outside.simultaneousUse = true;
			if (allocation->pair.list->BeginSecondary(outside) != rhi::Result::Ok)
				throw std::runtime_error("An external ticket's secondary could not begin");
			return allocation;
		};
		if (!admission.closed) {
			BT_ZONE_SCOPE("ORG.Persistent.TicketHandoff");
			for (const auto& release : admission.Barriers().releases) {
				if (release.queue >= nextQueueValues.size()) throw std::logic_error("A ticket's release has no queue value to signal");
				const auto textures = released(release);
				if (textures.empty()) continue;
				const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(release.queue));
				if (release.queue < streamSlots.size() && streamSlots[release.queue]) {
					auto allocation = streamSecondary(release.queue, "ORG external ticket release");
					auto list = allocation->pair.list.Get();
					rhi::BarrierBatch barriers{};
					barriers.textures = {textures.data(), static_cast<uint32_t>(textures.size())};
					list.Barriers(barriers);
					if (list.EndChecked() != rhi::Result::Ok) throw std::runtime_error("A ticket's release list could not close");
					allocation->Close();
					streamReleases.push_back(list);
					for (const auto consumer : release.consumers)
						if (std::ranges::find(streamReleaseConsumers, consumer) == streamReleaseConsumers.end()) streamReleaseConsumers.push_back(consumer);
					out.leases.push_back(std::move(allocation));
					basic_telemetry::AddCounter("ORG.Persistent.HandoffReleases");
					continue;
				}
				auto allocation = std::make_shared<compiler::FrameCommandAllocation>();
				allocation->pool = m_queueRegistry.GetSharedPool(index);
				allocation->pair = allocation->pool->Request();
				auto list = allocation->pair.list.Get();
				rhi::BarrierBatch barriers{};
				barriers.textures = {textures.data(), static_cast<uint32_t>(textures.size())};
				list.Barriers(barriers);
				if (list.EndChecked() != rhi::Result::Ok) throw std::runtime_error("A ticket's release list could not close");
				allocation->Close();
				const auto value = ++nextQueueValues[release.queue];
				const rhi::TimelinePoint point{m_queueRegistry.GetFence(index).GetHandle(), value};
				rhi::SubmitDesc submit{};
				submit.signals = {&point, 1u};
				// Every submission of this ticket comes after the group an earlier ticket left held.
				if (carry && !FlushSubmitCarry(*carry)) return false;
				if (m_queueRegistry.GetQueue(index).Submit({&list, 1u}, submit) != rhi::Result::Ok) return false;
				census.Add(Census::List::Release);
				census.Submitted(Census::Submission::Release, release.queue, 1, {});
				m_queueRegistry.EnsureNextFenceValueAtLeast(index, value + 1);
				for (const auto consumer : release.consumers) incoming.at(consumer).push_back({admission.queueTimelines.at(release.queue), value});
				out.leases.push_back(std::move(allocation));
				basic_telemetry::AddCounter("ORG.Persistent.HandoffReleases");
			}
			// The lists a ticket of the slot recorded from the same predicted plan, again; otherwise recorded, and kept for the slot's
			// later tickets.
			// Merged handoffs: per batch, the first batch of the queue submission the coalescing below gives it. A batch submitted on
			// its own (not a gatherable packet, or queue-scoped uploads ahead of it) keeps its own list, and the next batch starts a
			// new submission.
			std::vector<uint32_t> targets(batchCount);
			for (uint32_t batch = 0; batch < batchCount; ++batch) targets[batch] = batch;
			if (m_mergeHandoffs && m_coalesceSubmissions && !ticket.external) {
				const auto& packets = segment.recording->packets;
				uint32_t first = UINT32_MAX, firstQueue = UINT32_MAX;
				for (uint32_t batch = 0; batch < batchCount && batch < packets.size(); ++batch) {
					const bool gatherable = dynamic_cast<const compiler::PreparedRhiExecutionBatch*>(packets[batch].get()) != nullptr
						&& !std::ranges::any_of(planned.leading, [&](const auto& leading) { return leading.batch == batch && leading.copies; });
					if (!gatherable) {
						first = UINT32_MAX;
						continue;
					}
					if (first == UINT32_MAX || graph.batches[batch].queue != firstQueue) {
						first = batch;
						firstQueue = graph.batches[batch].queue;
					}
					targets[batch] = first;
				}
			}
			// The barriers each handoff list records. A barrier that discards, starts from Undefined, or transfers a queue family stays
			// with its own batch, and so does every barrier of a batch with acquired textures.
			struct HandoffBarriers {
				std::vector<rhi::TextureBarrier> textures;
				std::vector<rhi::BufferBarrier> buffers;
				std::vector<rhi::TextureBarrier> acquired;
				bool Empty() const noexcept { return textures.empty() && buffers.empty() && acquired.empty(); }
			};
			std::vector<HandoffBarriers> lists(batchCount);
			uint64_t leftOut = 0;  // entry barriers on untouched resources
			for (uint32_t batch = 0; batch < batchCount; ++batch) {
				const auto& entry = admission.Barriers().batches[batch];
				const bool own = targets[batch] == batch || !entry.acquired.empty();
				auto& local = lists[batch];
				auto& merged = lists[own ? batch : targets[batch]];
				for (const auto& texture : entry.textures) {
					if (untouched.Holds(texture.texture)) {
						++leftOut;
						continue;
					}
					const bool movable = !texture.discard && texture.beforeLayout != rhi::ResourceLayout::Undefined
						&& texture.queueOwnership == rhi::QueueOwnership::None && texture.externalOwnership == rhi::TextureBarrier::ExternalOwnership::None;
					(movable ? merged : local).textures.push_back(texture);
				}
				for (const auto& buffer : entry.buffers) {
					if (untouched.Holds(buffer.buffer)) {
						++leftOut;
						continue;
					}
					const bool movable = buffer.queueOwnership == rhi::QueueOwnership::None && buffer.externalOwnership == rhi::BufferBarrier::ExternalOwnership::None;
					(movable ? merged : local).buffers.push_back(buffer);
				}
				for (const auto& texture : entry.acquired) {
					if (!untouched.Holds(texture.texture)) local.acquired.push_back(texture);
					else ++leftOut;
				}
			}
			if (leftOut) diagnostics::UntouchedEntryBarriers().fetch_add(leftOut, std::memory_order_relaxed);
			PersistentExecutionsState::HandoffCache* cache = nullptr;
			auto miss = !admission.sharedBarriers ? Census::HandoffMiss::Planned
				: ticket.segments.size() != 1 ? Census::HandoffMiss::Segments : Census::HandoffMiss::NewPlan;
			uint32_t recordedHandoffs = 0;
			if (admission.sharedBarriers && ticket.segments.size() == 1) {
				if (state.handoffCache.size() <= ticket.slot) state.handoffCache.resize(size_t{ticket.slot} + 1);
				auto& entries = state.handoffCache[ticket.slot];
				const auto found = std::ranges::find_if(entries, [&](const auto& entry) {
					return entry.plan == admission.sharedBarriers && entry.external == ticket.external && entry.handoff.size() == batchCount
						&& entry.targets == targets && entry.untouched == untouched.signature;
				});
				// The one used goes first; a new one replaces the least recent when the slot keeps as many as it may.
				const bool hit = found != entries.end();
				if (hit) {
					std::rotate(entries.begin(), found, found + 1);
				} else {
					if (entries.size() >= PersistentExecutionsState::kHandoffCacheEntries) entries.pop_back();
					entries.insert(entries.begin(), PersistentExecutionsState::HandoffCache{});
				}
				cache = &entries.front();
				if (hit) {
					handoff = cache->handoff;
					streamHandoff = cache->streamHandoff;
					out.leases.insert(out.leases.end(), cache->allocations.begin(), cache->allocations.end());
					basic_telemetry::AddCounter("ORG.Persistent.HandoffListsReused");
					census.Add(Census::List::HandoffReused, static_cast<uint64_t>(std::ranges::count_if(handoff, [](rhi::CommandList list) { return static_cast<bool>(list); })));
				}
			}
			for (uint32_t batch = 0; batch < batchCount && !(cache && cache->plan); ++batch) {
				const auto& entry = lists[batch];
				if (entry.Empty()) continue;
				std::shared_ptr<compiler::FrameCommandAllocation> allocation;
				if (inStream(batch)) {
					allocation = streamSecondary(graph.batches[batch].queue, "ORG external ticket handoff");
				} else {
					allocation = std::make_shared<compiler::FrameCommandAllocation>();
					allocation->pool = m_queueRegistry.GetSharedPool(static_cast<QueueSlotIndex>(static_cast<uint8_t>(graph.batches[batch].queue)));
					allocation->pair = allocation->pool->Request();
				}
				auto list = allocation->pair.list.Get();
				// BasicRHI's command census names the list (diagnostics).
				if (rhi::census::Enabled()) {
					// Named after its batch's first pass, so a barrier's cost can be placed.
					std::string label = "handoff";
					const auto& passes = graph.batches[batch].passes;
					const auto& names = graph.structure->diagnosticPassNames;
					if (!passes.empty() && passes.front() < names.size() && !names[passes.front()].empty()) label += " > " + names[passes.front()];
					rhi::census::SetLabel(label.c_str());
				}
				rhi::BarrierBatch barriers{};
				barriers.textures = {entry.textures.data(), static_cast<uint32_t>(entry.textures.size())};
				barriers.buffers = {entry.buffers.data(), static_cast<uint32_t>(entry.buffers.size())};
				list.Barriers(barriers);
				if (!entry.acquired.empty()) {
					rhi::BarrierBatch acquired{};
					acquired.textures = {entry.acquired.data(), static_cast<uint32_t>(entry.acquired.size())};
					list.Barriers(acquired);
				}
				const auto closed = [&] {
					BT_ZONE_SCOPE("ORG.Persistent.TicketHandoff.End");
					return list.EndChecked();
				}();
				if (closed != rhi::Result::Ok) throw std::runtime_error("A ticket's handoff list could not close");
				allocation->Close();
				census.Add(Census::List::Handoff);
				++recordedHandoffs;
				(inStream(batch) ? streamHandoff : handoff)[batch] = list;
				if (cache) cache->allocations.push_back(allocation);
				out.leases.push_back(std::move(allocation));
			}
			if (recordedHandoffs) census.Miss(miss);
			if (cache && !cache->plan) {
				cache->plan = admission.sharedBarriers;
				cache->external = ticket.external;
				cache->handoff = handoff;
				cache->streamHandoff = streamHandoff;
				cache->targets = targets;
				cache->untouched = untouched.signature;
			}
		}
		if (planned.planned && planned.waitsForUploads.size() != batchCount)
			throw std::logic_error("Queue-scoped uploads planned for other batches");
		// An external ticket's graphics-queue uploads run in its host's stream, first. A batch off it that uses them waits for a
		// segment of their own, whose signal on the graphics timeline comes before its batches' values.
		const auto usesUploads = [&](uint32_t batch) { return !planned.planned || planned.waitsForUploads[batch]; };
		// The stream's releases come first: in the stream (no wait) for its own batches, by their value for the others.
		const bool releasesOffStream = std::ranges::any_of(streamReleaseConsumers, [&](uint32_t consumer) { return !inStream(consumer); });
		if (!streamReleases.empty() && releasesOffStream) {
			if (graphicsSlot >= nextQueueValues.size()) throw std::logic_error("A ticket's releases have no graphics queue value to signal");
			streamReleasesSignal = ++nextQueueValues[graphicsSlot];
		}
		uint64_t streamUploadsSignal = 0;
		if (ticket.external && streamUploads)
			for (uint32_t batch = 0; batch < batchCount && !streamUploadsSignal; ++batch)
				if (!inStream(batch) && usesUploads(batch)) {
					if (graphicsSlot >= nextQueueValues.size()) throw std::logic_error("A ticket's uploads have no graphics queue value to signal");
					streamUploadsSignal = ++nextQueueValues[graphicsSlot];
				}
		// Scoped frame wait: only the batches that read the frame's values wait for it.
		std::vector<uint8_t> frameReaders;
		if (frameWait && m_scopedFrameWait) frameReaders = persistent::FrameWaitReaders(graph, admission.backings);
		uint32_t frameWaitsSpared = 0;
		for (uint32_t batch = 0; batch < batchCount; ++batch) {
			const auto queue = graph.batches[batch].queue;
			if (queue >= nextQueueValues.size()) throw std::logic_error("Ticket batch on an unknown queue");
			batches[batch].signal = {admission.queueTimelines.at(queue), ++nextQueueValues[queue]};
			if (batch < incoming.size()) batches[batch].waits = incoming[batch];
			// The uploads ahead of the ticket order only the batches on their own queue (its entry barrier): any other queue
			// would otherwise read what this epoch's uploads write before they land. Planned by queue (RecordPendingUploads),
			// only the batches that use what they write wait (the rest of theirs went to their own queue, ahead of them).
			if (uploadsDone.second && queue != uploadsDone.first && usesUploads(batch))
				batches[batch].waits.push_back({admission.queueTimelines.at(uploadsDone.first), uploadsDone.second});
			if (streamUploadsSignal && !inStream(batch) && usesUploads(batch))
				batches[batch].waits.push_back({admission.queueTimelines.at(graphicsSlot), streamUploadsSignal});
			if (streamReleasesSignal && !inStream(batch) && std::ranges::find(streamReleaseConsumers, batch) != streamReleaseConsumers.end())
				batches[batch].waits.push_back({admission.queueTimelines.at(graphicsSlot), streamReleasesSignal});
			// The frame's values (SetFrameWaitTimeline): every packet prepared since it was set binds the timeline.
			if (frameWait && (frameReaders.empty() || frameReaders[batch]))
				batches[batch].waits.push_back({kFrameWaitTimelineIdentity, frameWait});
			else if (frameWait)
				++frameWaitsSpared;
			out.signals.push_back(batches[batch].signal.value);
		}
		if (frameWaitsSpared) basic_telemetry::AddCounter("ORG.Persistent.FrameWaitsSpared", frameWaitsSpared);
		for (const auto wait : graph.relativeWaits)
			batches.at(wait.consumerBatch).waits.push_back(batches.at(wait.producerBatch).signal);
		for (auto& batch : batches) {
			std::sort(batch.waits.begin(), batch.waits.end(), [](auto a, auto b) { return a.timeline < b.timeline; });
			size_t count = 0;
			for (auto wait : batch.waits) {
				if (!wait.value) continue;
				if (count && batch.waits[count - 1].timeline == wait.timeline)
					batch.waits[count - 1].value = (std::max)(batch.waits[count - 1].value, wait.value);
				else batch.waits[count++] = wait;
			}
			batch.waits.resize(count);
		}
		const auto& packets = segment.recording->packets;
		if (packets.size() != batches.size()) throw std::logic_error("Ticket packets do not match its batches");
		// Submits a batch on its queue: waits on timelines it does not bind (another graph's, from the admission) bind through
		// the runtime; its handoff goes ahead of it, and so do the uploads recorded on its queue (on the queue just ahead of
		// it, so it needs no wait for them: the list ends with a full barrier), with the waits their destinations need.
		std::vector<std::vector<compiler::PreparedTimelineBinding>> extra(packets.size());
		// The census's view of a submission's waits: those off its own queue's timeline, by kind.
		std::vector<Census::Wait> censusWaits;
		const auto censusWait = [&](rhi::TimelineHandle handle, uint64_t value, uint32_t queue) {
			const auto own = m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(static_cast<uint8_t>(queue))).GetHandle();
			if (!handle.valid() || (handle.index == own.index && handle.generation == own.generation)) return;
			auto kind = Census::WaitKind::Other;
			if (m_frameWaitTimeline && *m_frameWaitTimeline) {
				const auto frame = (*m_frameWaitTimeline)->GetHandle();
				if (frame.index == handle.index && frame.generation == handle.generation) kind = Census::WaitKind::FrameWait;
			}
			for (size_t slot = 0; slot < m_queueRegistry.SlotCount() && kind == Census::WaitKind::Other; ++slot) {
				const auto fence = m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot))).GetHandle();
				if (fence.index == handle.index && fence.generation == handle.generation) kind = Census::WaitKind::OtherQueue;
			}
			censusWaits.push_back({(uint64_t{handle.index} << 32) | handle.generation, value, kind});
		};
		const auto submitPacket = [&](uint32_t i) -> bool {
			if (carry && !FlushSubmitCarry(*carry)) return false;
			const auto* packet = dynamic_cast<const compiler::PreparedRhiExecutionBatch*>(packets[i].get());
			for (const auto wait : batches[i].waits) {
				if (!packet || packet->BindsTimeline(wait.timeline)) continue;
				const auto handle = m_runtime->TimelineHandleOf(wait.timeline);
				if (handle.valid()) extra[i].push_back({wait.timeline, handle});
			}
			batches[i].extraTimelines = extra[i];
			batches[i].skips = skips[i];
			batches[i].untouched = untouched.Empty() ? nullptr : &untouched;
			if (handoff[i]) batches[i].leading = {&handoff[i], 1};
			for (const auto& leading : planned.leading) {
				if (leading.batch != i || !leading.copies) continue;
				if (!packet) throw std::logic_error("Queue-scoped uploads ahead of a packet that cannot take them");
				auto list = leading.list;
				if (!packet->SubmitLeading({&list, 1}, leading.waits)) return false;
				if (census.On()) {
					censusWaits.clear();
					for (const auto wait : leading.waits) censusWait(m_runtime->TimelineHandleOf(wait.timeline), wait.value, graph.batches[i].queue);
					census.Submitted(Census::Submission::Leading, graph.batches[i].queue, 1, censusWaits);
				}
			}
			if (!packets[i]->Submit(batches[i])) return false;
			if (census.On()) {
				const auto queue = graph.batches[i].queue;
				censusWaits.clear();
				for (const auto wait : batches[i].waits) {
					auto handle = packet ? packet->ResolveTimeline(wait.timeline) : rhi::TimelineHandle{};
					if (!handle.valid()) handle = m_runtime->TimelineHandleOf(wait.timeline);
					censusWait(handle, wait.value, queue);
				}
				const uint32_t lists = packet && !packet->Inline() ? static_cast<uint32_t>(packet->Lists().size()) : 1u;
				if (packet && !packet->Inline()) census.Add(Census::List::Batch, lists);
				census.Submitted(Census::Submission::Batches, queue, lists + (handoff[i] ? 1u : 0u), censusWaits);
			}
			return true;
		};
		if (ticket.external) {
			// The batches on the graphics queue are not submitted: the host executes them in its own stream, segment by segment
			// (ExternalEpochExecution). A segment's submission waits for its batches' waits and signals their values.
			BT_ZONE_SCOPE("ORG.Persistent.ExternalExecution");
			if (carry && !FlushSubmitCarry(*carry)) return false;
			if (!out.external) out.external = std::make_shared<ExternalEpochExecution>();
			auto& execution = *out.external;
			const auto graphicsTimeline = admission.queueTimelines.at(graphicsSlot);
			const auto graphicsFence = m_queueRegistry.GetFence(m_queueRegistry.FindGraphicsSlot()).GetHandle();
			if (!streamReleases.empty()) {
				auto& releases = execution.segments.emplace_back();
				releases.leading = streamReleases;
				if (streamReleasesSignal) {
					releases.signals.push_back({graphicsFence, streamReleasesSignal});
					state.admission->ExtendSubmitted({graphicsTimeline, streamReleasesSignal});
				}
			}
			if (streamUploadsSignal) {
				auto& uploads = execution.segments.emplace_back();
				uploads.signals.push_back({m_queueRegistry.GetFence(m_queueRegistry.FindGraphicsSlot()).GetHandle(), streamUploadsSignal});
				state.admission->ExtendSubmitted({graphicsTimeline, streamUploadsSignal});
			}
			// Per batch off the stream: whether it waits (through other batches) for a segment of the stream. The batches are in an
			// order every wait follows, so a batch's producers come before it.
			std::vector<uint8_t> afterStream(batchCount);
			std::vector<std::vector<uint32_t>> producers(batchCount);
			for (const auto wait : graph.relativeWaits)
				if (wait.consumerBatch < batchCount && wait.producerBatch < batchCount) producers[wait.consumerBatch].push_back(wait.producerBatch);
			for (uint32_t i = 0; i < batchCount; ++i) {
				if (!inStream(i)) {
					bool after = (streamUploadsSignal && usesUploads(i))
						|| (streamReleasesSignal && std::ranges::find(streamReleaseConsumers, i) != streamReleaseConsumers.end());
					for (const auto producer : producers[i]) {
						if (producer >= i) throw std::logic_error("A ticket's batch waits for a later one");
						after = after || inStream(producer) || afterStream[producer];
					}
					afterStream[i] = after ? 1 : 0;
					if (!submitPacket(i)) return false;
					continue;
				}
				auto packet = std::dynamic_pointer_cast<const compiler::PreparedRhiExecutionBatch>(packets[i]);
				if (!packet) throw std::logic_error("An external epoch's packet the host cannot execute");
				// A new segment where the batch waits for a batch off the stream; split from the ones before when that batch waits
				// for one of them.
				bool waitsOffStream = false, split = false;
				for (const auto producer : producers[i])
					if (!inStream(producer)) {
						waitsOffStream = true;
						split = split || afterStream[producer];
					}
				if (execution.segments.empty() || waitsOffStream) execution.segments.emplace_back().split = split;
				auto& stream = execution.segments.back();
				if (streamHandoff[i]) stream.leading.push_back(streamHandoff[i]);
				for (const auto wait : batches[i].waits) {
					// A timeline of a queue the stream runs: everything submitted on that queue is in the stream, in order before this.
					bool streamOrdered = false;
					for (uint32_t slot = 0; slot < streamSlots.size() && !streamOrdered; ++slot)
						streamOrdered = streamSlots[slot] && slot < admission.queueTimelines.size() && admission.queueTimelines[slot] == wait.timeline;
					if (streamOrdered) continue;
					// Another graph's timeline (an open admission's wait on its earlier execution) resolves through the runtime.
					auto timeline = packet->ResolveTimeline(wait.timeline);
					if (!timeline.valid()) timeline = m_runtime->TimelineHandleOf(wait.timeline);
					if (!timeline.valid()) throw std::logic_error("An external epoch's wait on a timeline neither its packet nor the runtime binds");
					stream.waits.push_back({timeline, wait.value});
				}
				if (!packet->CommitExternal(batches[i].signal.value, stream.signals)) return false;
				stream.items.push_back({packet, segment.resourceHeap, segment.samplerHeap});
				const auto timeline = packet->ResolveTimeline(batches[i].signal.timeline);
				if (!timeline.valid()) throw std::logic_error("An external epoch's signal on a timeline its packet does not bind");
				stream.signals.push_back({timeline, batches[i].signal.value});
			}
			execution.exitSplit = execution.exitSplit || std::ranges::any_of(afterStream, [](uint8_t after) { return after != 0; });
			execution.keepAlive.push_back(segment.recording);
			CommitTicketSegment(*state.admission, segment, batches);
			// Host-owned textures last used off the stream, acquired back in it after their producers (external home states).
			const auto& home = admission.Barriers().homeAcquires;
			if (!home.textures.empty()) {
				auto allocation = streamSecondary(graphicsSlot, "ORG external ticket home acquires");
				auto list = allocation->pair.list.Get();
				rhi::BarrierBatch barriers{};
				barriers.textures = {home.textures.data(), static_cast<uint32_t>(home.textures.size())};
				list.Barriers(barriers);
				if (!home.home.empty()) {
					rhi::BarrierBatch toHome{};
					toHome.textures = {home.home.data(), static_cast<uint32_t>(home.home.size())};
					list.Barriers(toHome);
				}
				if (list.EndChecked() != rhi::Result::Ok) throw std::runtime_error("A ticket's home acquires could not close");
				allocation->Close();
				bool split = false;
				std::vector<rhi::TimelinePoint> waits;
				for (const auto producer : home.producers) {
					if (producer >= batchCount || inStream(producer)) continue;
					split = split || afterStream[producer];
					const auto* prepared = dynamic_cast<const compiler::PreparedRhiExecutionBatch*>(packets[producer].get());
					auto timeline = prepared ? prepared->ResolveTimeline(batches[producer].signal.timeline) : rhi::TimelineHandle{};
					if (!timeline.valid()) timeline = m_runtime->TimelineHandleOf(batches[producer].signal.timeline);
					if (!timeline.valid()) throw std::logic_error("A ticket's home acquires wait on a timeline nothing binds");
					waits.push_back({timeline, batches[producer].signal.value});
				}
				// No split when the segments since the last split already wait for each of these: the host's submission that holds
				// them waits for those anyway.
				if (split) {
					bool covered = true;
					for (const auto& wait : waits) {
						bool found = false;
						for (auto segment = execution.segments.rbegin(); segment != execution.segments.rend() && !found; ++segment) {
							found = std::ranges::any_of(segment->waits, [&](const rhi::TimelinePoint& other) {
								return other.t.index == wait.t.index && other.t.generation == wait.t.generation && other.value >= wait.value;
							});
							if (segment->split) break;
						}
						covered = covered && found;
					}
					split = !covered;
				}
				auto& trailing = execution.segments.emplace_back();
				trailing.split = split;
				trailing.leading.push_back(list);
				trailing.waits = std::move(waits);
				const auto value = ++nextQueueValues[graphicsSlot];
				trailing.signals.push_back({graphicsFence, value});
				state.admission->ExtendSubmitted({graphicsTimeline, value});
				out.leases.push_back(std::move(allocation));
				basic_telemetry::AddCounter("ORG.Persistent.HomeAcquires");
			}
			continue;
		}
		if (m_coalesceSubmissions) {
			BT_ZONE_SCOPE("ORG.Frame.SubmitPacketsCoalesced");
			// Consecutive batches on one queue share one queue submission (SetCoalescedSubmissions). A batch with queue-scoped
			// uploads ahead of it, or a packet of another kind, is submitted on its own.
			compiler::PreparedRhiExecutionBatch::GatheredSubmission gathered;
			std::vector<uint32_t> group;
			uint32_t groupQueue = UINT32_MAX;
			rhi::Queue groupRhiQueue{};
			// The ticket's last group may be held for the next ticket (carryOut): only when nothing of this ticket is submitted after it
			// (no home acquires; the last segment).
			const bool holdLast = carry && carryOut && &segment == &ticket.segments.back() && admission.Barriers().homeAcquires.textures.empty();
			const auto flush = [&](bool last) -> bool {
				if (group.empty()) return true;
				// The ticket's uploads ahead of its first group's lists. Their signal is below the group's own values on the same
				// timeline, so the group's highest signal covers it.
				if (uploadsPending) {
					uploadsPending = false;
					uploads->withBatches = true;
					gathered.lists.insert(gathered.lists.begin(), uploads->lists.begin(), uploads->lists.end());
					gathered.signals.push_back(uploads->signal);
				}
				// The batches' own queue timeline: a wait on a value one of them signals is covered by queue order. Every
				// signal on one timeline collapses to its highest value, which covers the lower ones.
				const auto own = m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(static_cast<uint8_t>(groupQueue))).GetHandle();
				uint64_t firstSignal = batches[group.front()].signal.value;
				// The group an earlier ticket left held: this group joins it when both are on one queue and timeline, and this group
				// waits for nothing the held one did not. A wait on a value the held group signals is then covered by queue order, as
				// within a ticket. Otherwise the held group is submitted on its own first.
				bool joined = false;
				if (carry && carry->Active()) {
					// One queue: each queue signals its own timeline (the host's, shared by every graph), so the same timeline means
					// the same queue.
					const bool sameTimeline = carry->own.index == own.index && carry->own.generation == own.generation;
					bool covered = sameTimeline;
					bool waitsOnHeld = false;
					for (const auto& wait : gathered.waits) {
						if (!covered) break;
						if (wait.t.index == own.index && wait.t.generation == own.generation && wait.value >= carry->firstSignal) {
							waitsOnHeld = true;
							continue;
						}
						covered = std::ranges::any_of(carry->gathered.waits, [&](const rhi::TimelinePoint& held) {
							return held.t.index == wait.t.index && held.t.generation == wait.t.generation && held.value >= wait.value;
						});
					}
					if (covered) {
						joined = true;
						firstSignal = carry->firstSignal;
						// Between the held lists and this group's, when this group waited for a value the held group signals: a full
						// memory barrier. Apart, that wait ordered this group's work after the held group's and made its writes
						// visible. In one submission the wait goes, and the barrier takes its place.
						if (waitsOnHeld) {
							auto allocation = std::make_shared<compiler::FrameCommandAllocation>();
							allocation->pool = m_queueRegistry.GetSharedPool(static_cast<QueueSlotIndex>(static_cast<uint8_t>(groupQueue)));
							allocation->pair = allocation->pool->Request();
							auto barrierList = allocation->pair.list.Get();
							if (rhi::census::Enabled()) rhi::census::SetLabel("joined barrier");
							const auto full = rhi::FullMemoryBarrier();
							rhi::BarrierBatch barriers{};
							barriers.globals = {&full, 1u};
							barrierList.Barriers(barriers);
							if (barrierList.EndChecked() != rhi::Result::Ok) throw std::runtime_error("A joined submission's barrier list could not close");
							allocation->Close();
							out.leases.push_back(std::move(allocation));
							gathered.lists.insert(gathered.lists.begin(), barrierList);
						}
						gathered.waits.insert(gathered.waits.begin(), carry->gathered.waits.begin(), carry->gathered.waits.end());
						gathered.lists.insert(gathered.lists.begin(), carry->gathered.lists.begin(), carry->gathered.lists.end());
						gathered.signals.insert(gathered.signals.begin(), carry->gathered.signals.begin(), carry->gathered.signals.end());
						++carry->joins;
						basic_telemetry::AddCounter("ORG.Persistent.CarriedSubmissionsJoined");
					} else if (!FlushSubmitCarry(*carry)) {
						return false;
					}
				}
				const auto merge = [](std::vector<rhi::TimelinePoint>& points) {
					std::vector<rhi::TimelinePoint> merged;
					for (const auto& point : points) {
						auto found = std::ranges::find_if(merged, [&](const rhi::TimelinePoint& other) {
							return other.t.index == point.t.index && other.t.generation == point.t.generation;
						});
						if (found == merged.end()) merged.push_back(point);
						else found->value = (std::max)(found->value, point.value);
					}
					points = std::move(merged);
				};
				std::erase_if(gathered.waits, [&](const rhi::TimelinePoint& wait) {
					return wait.t.index == own.index && wait.t.generation == own.generation && wait.value >= firstSignal;
				});
				merge(gathered.waits);
				merge(gathered.signals);
				// Held for the next ticket instead of submitted: the joined group's batches and this one's, with their recordings.
				if (last && holdLast) {
					if (!joined) {
						carry->Clear();
						carry->queue = groupRhiQueue;
						carry->own = own;
						carry->queueSlot = groupQueue;
					}
					carry->firstSignal = firstSignal;
					carry->gathered = std::move(gathered);
					carry->keep.push_back(segment.recording);
					carry->tickets = joined ? carry->tickets + 1 : 1;
					// Committed as submitted now, not when the held submission reaches the queue: the next ticket's preparation and
					// recording read what these passes publish at their submission (a new version, for one), and the queue takes the
					// held submission before any other submitter's, under the same hold of the sequencer.
					bool committed = true;
					for (const auto batch : group) {
						const auto* packet = static_cast<const compiler::PreparedRhiExecutionBatch*>(packets[batch].get());
						committed = static_cast<bool>(packet->CommitGathered(batches[batch])) && committed;
					}
					group.clear();
					gathered.Clear();
					return committed;
				}
				if (census.On()) {
					censusWaits.clear();
					for (const auto& wait : gathered.waits) censusWait(wait.t, wait.value, groupQueue);
				}
				const auto submitted = groupRhiQueue.Submit({gathered.lists.data(), static_cast<uint32_t>(gathered.lists.size())},
					{{gathered.waits.data(), static_cast<uint32_t>(gathered.waits.size())},
					 {gathered.signals.data(), static_cast<uint32_t>(gathered.signals.size())}});
				basic_telemetry::AddCounter("ORG.Persistent.CoalescedSubmissions");
				basic_telemetry::AddCounter("ORG.Persistent.CoalescedBatches", group.size());
				const bool ok = submitted == rhi::Result::Ok;
				if (ok) census.Submitted(Census::Submission::Batches, groupQueue, static_cast<uint32_t>(gathered.lists.size()), censusWaits);
				// Each batch the queue accepted commits its passes, even when another one's commit failed.
				bool committed = true;
				if (joined) {
					// The held batches were committed when they were held.
					if (ok) basic_telemetry::AddCounter("ORG.Persistent.CarriedTicketsSubmitted", carry->tickets);
					carry->Clear();
				}
				for (const auto batch : group) {
					if (!ok) break;
					const auto* packet = static_cast<const compiler::PreparedRhiExecutionBatch*>(packets[batch].get());
					committed = static_cast<bool>(packet->CommitGathered(batches[batch])) && committed;
				}
				group.clear();
				gathered.Clear();
				return ok && committed;
			};
			for (uint32_t i = 0; i < batchCount; ++i) {
				const auto* packet = dynamic_cast<const compiler::PreparedRhiExecutionBatch*>(packets[i].get());
				const bool leadingUploads = std::ranges::any_of(planned.leading, [&](const auto& leading) { return leading.batch == i && leading.copies; });
				if (!packet || leadingUploads) {
					if (!flush(false) || !submitPacket(i)) return false;
					continue;
				}
				const auto queue = graph.batches[i].queue;
				if (queue != groupQueue && !flush(false)) return false;
				groupQueue = queue;
				groupRhiQueue = packet->SubmitQueue();
				for (const auto wait : batches[i].waits) {
					if (packet->BindsTimeline(wait.timeline)) continue;
					const auto handle = m_runtime->TimelineHandleOf(wait.timeline);
					if (handle.valid()) extra[i].push_back({wait.timeline, handle});
				}
				batches[i].extraTimelines = extra[i];
				batches[i].skips = skips[i];
				batches[i].untouched = untouched.Empty() ? nullptr : &untouched;
				if (handoff[i]) batches[i].leading = {&handoff[i], 1};
				// An inline late batch is recorded here, and a pass recorded at execution may submit work of its own on the queue (an
				// external client's, after a drain). The held group goes first, so nothing comes ahead of it on the queue.
				if (packet->Inline() && carry && !FlushSubmitCarry(*carry)) return false;
				if (!packet->Gather(batches[i], gathered)) {
					// Its effects are abandoned. The batches gathered before it still go to the queue.
					(void)flush(false);
					return false;
				}
				if (!packet->Inline()) census.Add(Census::List::Batch, packet->Lists().size());
				group.push_back(i);
			}
			if (!flush(true)) return false;
		} else {
			BT_ZONE_SCOPE("ORG.Frame.SubmitPackets");
			for (uint32_t i = 0; i < batchCount; ++i)
				if (!submitPacket(i)) return false;
		}
		CommitTicketSegment(*state.admission, segment, batches);
		// Host-owned textures last used off the graphics queue, acquired back on it after their producers (external home states).
		const auto& home = admission.Barriers().homeAcquires;
		if (!home.textures.empty()) {
			const auto index = m_queueRegistry.FindGraphicsSlot();
			auto allocation = std::make_shared<compiler::FrameCommandAllocation>();
			allocation->pool = m_queueRegistry.GetSharedPool(index);
			allocation->pair = allocation->pool->Request();
			auto list = allocation->pair.list.Get();
			rhi::BarrierBatch barriers{};
			barriers.textures = {home.textures.data(), static_cast<uint32_t>(home.textures.size())};
			list.Barriers(barriers);
			if (!home.home.empty()) {
				rhi::BarrierBatch toHome{};
				toHome.textures = {home.home.data(), static_cast<uint32_t>(home.home.size())};
				list.Barriers(toHome);
			}
			if (list.EndChecked() != rhi::Result::Ok) throw std::runtime_error("A ticket's home acquires could not close");
			allocation->Close();
			if (graphicsSlot >= nextQueueValues.size()) throw std::logic_error("A ticket's home acquires have no graphics queue value to signal");
			const auto value = ++nextQueueValues[graphicsSlot];
			std::vector<rhi::TimelinePoint> waits;
			for (const auto producer : home.producers) {
				if (producer >= batchCount || graph.batches[producer].queue == graphicsSlot) continue;
				const auto producerSlot = static_cast<QueueSlotIndex>(static_cast<uint8_t>(graph.batches[producer].queue));
				waits.push_back({m_queueRegistry.GetFence(producerSlot).GetHandle(), batches[producer].signal.value});
			}
			const rhi::TimelinePoint point{m_queueRegistry.GetFence(index).GetHandle(), value};
			rhi::SubmitDesc submit{};
			submit.waits = {waits.data(), static_cast<uint32_t>(waits.size())};
			submit.signals = {&point, 1u};
			if (carry && !FlushSubmitCarry(*carry)) return false;
			if (m_queueRegistry.GetQueue(index).Submit({&list, 1u}, submit) != rhi::Result::Ok) return false;
			census.Add(Census::List::HomeAcquire);
			if (census.On()) {
				censusWaits.clear();
				for (const auto& wait : waits) censusWait(wait.t, wait.value, graphicsSlot);
				census.Submitted(Census::Submission::HomeAcquire, graphicsSlot, 1, censusWaits);
			}
			m_queueRegistry.EnsureNextFenceValueAtLeast(index, value + 1);
			state.admission->ExtendSubmitted({admission.queueTimelines.at(graphicsSlot), value});
			out.leases.push_back(std::move(allocation));
			basic_telemetry::AddCounter("ORG.Persistent.HomeAcquires");
		}
	}
	// An external ticket without segments (none had batches: ExecutePersistentFrame) executes nothing but its uploads.
	if (ticket.external && !out.external) out.external = std::make_shared<ExternalEpochExecution>();
	out.submitted = true;
	return true;
}

namespace {
ResourceRegistry::RegistryHandle TicketUploadResolveById(void*, ResourceIdentifier const& id, bool) {
	throw std::runtime_error("Ticket uploads resolve only resource pointers, not '" + id.ToString() + "'");
}
ResourceRegistry::RegistryHandle TicketUploadResolveByPtr(void*, Resource* resource, bool) {
	return ResourceRegistry::RegistryHandle::MakeEphemeral(resource);
}

uint64_t UploadBufferKey(rhi::ResourceHandle handle) { return (uint64_t{handle.generation} << 32) | handle.index; }

// Queue-scoped ticket uploads (RenderGraph::RecordPendingUploads): the staged destinations recorded on the queue that first uses
// them instead of the graphics list, and the batches off the graphics queue that use what stays there.
struct TicketUploadRouting {
	std::unordered_map<uint64_t, uint32_t> routed;  // destination backing -> the queue slot its copies go to
	std::vector<uint8_t> waitsForUploads;           // per batch: uses what the graphics list (or a recorded list) writes
	std::vector<uint32_t> firstOnQueue;             // per queue slot: its first batch, UINT32_MAX for none
	std::vector<size_t> copies;                     // per queue slot: the staged copies routed to it
	std::vector<std::vector<compiler::ExecutionTimelinePoint>> waits;  // per queue slot: its upload list's waits
	// Destinations that are a late-bound version's backing (statistics): routed to a declaring batch's queue, or waited for off graphics.
	uint32_t lateRouted = 0, lateWaits = 0;
	uint32_t undeclared = 0;  // destinations no batch declares (statistics)
};

// graphicsBuffers: everything the graphics list and the recorded lists touch (known). staged: the staged copies' destinations, one
// per copy. False when the ticket cannot be planned: every batch off the graphics queue then waits for the uploads, as before.
bool PlanTicketUploads(const RenderGraph::PersistentTicket& ticket, uint32_t graphicsQueue, std::span<rhi::CommandList> queueLists,
	std::span<const rhi::ResourceHandle> graphicsBuffers, std::span<const rhi::ResourceHandle> staged, TicketUploadRouting& out) {
	BT_ZONE_SCOPE("ORG.Persistent.PlanTicketUploads");
	if (ticket.segments.size() != 1) return false;
	const auto& segment = ticket.segments.front();
	if (!segment.recording || !segment.admission.publication || !segment.admission.publication->executable
		|| !segment.admission.publication->executable->graph)
		return false;
	const auto& graph = *segment.admission.publication->executable->graph;
	const auto& backings = segment.admission.backings;
	const auto& packets = segment.recording->packets;
	const auto batchCount = static_cast<uint32_t>(graph.batches.size());
	if (!graph.structure || backings.size() != graph.structure->resourceIDs.size() || packets.size() != batchCount
		|| graph.boundaryAccessesByBatch.size() != batchCount)
		return false;
	uint32_t queueCount = 0;
	for (const auto& batch : graph.batches) queueCount = (std::max)(queueCount, batch.queue + 1);
	out.firstOnQueue.assign(queueCount, UINT32_MAX);
	for (uint32_t batch = 0; batch < batchCount; ++batch)
		if (out.firstOnQueue[graph.batches[batch].queue] == UINT32_MAX) out.firstOnQueue[graph.batches[batch].queue] = batch;
	out.copies.assign(queueCount, 0);
	out.waits.assign(queueCount, {});
	out.waitsForUploads.assign(batchCount, 0);

	// Every destination's uses in the ticket, in batch order: (batch, compiler resource), through each resource's backing for
	// this execution. Several compiler resources may share a backing; an aliased one is never routed (another backing may share
	// its memory).
	struct Destination {
		std::vector<std::pair<uint32_t, uint32_t>> uses;
		bool graphics = false;  // written by the graphics list or a recorded list
		bool staged = false;
		bool aliased = false;
	};
	std::unordered_map<uint64_t, Destination> destinations;
	destinations.reserve(graphicsBuffers.size() + staged.size());
	for (const auto handle : graphicsBuffers) destinations[UploadBufferKey(handle)].graphics = true;
	for (const auto handle : staged) destinations[UploadBufferKey(handle)].staged = true;
	// Late-bound versions (VersionedBuffer::CreateLateBound): a destination that is a live version's backing is used by every
	// batch that declares the buffer (its stand-in), as if that batch declared the destination itself. By the stand-in's
	// backing: its destinations.
	std::unordered_map<uint64_t, std::vector<uint64_t>> lateDestinations;
	if (detail::AnyLateBacking()) {
		auto bindLate = [&](rhi::ResourceHandle handle) {
			const auto standIn = detail::LateStandInOf(handle);
			if (!standIn.valid()) return;
			auto& members = lateDestinations[UploadBufferKey(standIn)];
			const auto key = UploadBufferKey(handle);
			if (std::ranges::find(members, key) == members.end()) members.push_back(key);
		};
		for (const auto handle : graphicsBuffers) bindLate(handle);
		for (const auto handle : staged) bindLate(handle);
	}
	auto addUse = [](Destination& destination, uint32_t batch, uint32_t resource) {
		auto& uses = destination.uses;
		if (uses.empty() || uses.back() != std::pair{batch, resource}) uses.push_back({batch, resource});
	};
	for (uint32_t batch = 0; batch < batchCount; ++batch)
		for (const auto& access : graph.boundaryAccessesByBatch[batch]) {
			if (access.resource >= backings.size()) continue;
			const auto& backing = backings[access.resource];
			if (!backing.resource.valid()) continue;
			const auto key = UploadBufferKey(backing.resource);
			if (const auto found = destinations.find(key); found != destinations.end()) {
				addUse(found->second, batch, access.resource);
				if (backing.aliasSize || backing.aliasHeapIdentity || backing.aliasHeap) found->second.aliased = true;
			}
			// The stand-in's own aliasing says nothing of its destinations' memory.
			if (const auto standIn = lateDestinations.find(key); standIn != lateDestinations.end())
				for (const auto member : standIn->second) addUse(destinations.at(member), batch, access.resource);
		}

	// Whether each batch is ordered after a queue's first batch: on that queue, or behind a batch that is (its queue's order,
	// or a wait). Only waits on earlier batches count; anything else is taken as unordered.
	std::vector<std::vector<uint32_t>> producers(batchCount);
	for (const auto wait : graph.relativeWaits)
		if (wait.consumerBatch < batchCount && wait.producerBatch < wait.consumerBatch) producers[wait.consumerBatch].push_back(wait.producerBatch);
	std::vector<std::vector<uint8_t>> after(queueCount);
	auto orderedAfter = [&](uint32_t queue) -> const std::vector<uint8_t>& {
		auto& ordered = after[queue];
		if (!ordered.empty()) return ordered;
		ordered.assign(batchCount, 0);
		std::vector<uint32_t> previousOnQueue(queueCount, UINT32_MAX);
		for (uint32_t batch = 0; batch < batchCount; ++batch) {
			const auto batchQueue = graph.batches[batch].queue;
			bool behind = batchQueue == queue;  // every batch of the queue is at or after its first
			if (!behind && previousOnQueue[batchQueue] != UINT32_MAX) behind = ordered[previousOnQueue[batchQueue]] != 0;
			for (const auto producer : producers[batch]) behind = behind || ordered[producer] != 0;
			ordered[batch] = behind ? 1 : 0;
			previousOnQueue[batchQueue] = batch;
		}
		return ordered;
	};

	// A staged destination goes to the queue of its first use, when that is not the graphics queue and has a list, nothing on the
	// graphics list touches it, every use is ordered after that queue's first batch (where its list goes), and the earlier
	// executions' use of it is known (the admission's boundary access waits) and can be waited for there.
	const auto& incoming = segment.admission.incomingWaits;
	const auto& boundary = segment.admission.boundaryAccessWaits;
	const bool earlierUseKnown = boundary.size() == batchCount;
	std::vector<compiler::ExecutionTimelinePoint> waits;
	for (auto& [key, destination] : destinations) {
		if (!destination.staged || destination.graphics || destination.aliased || destination.uses.empty() || !earlierUseKnown) continue;
		const auto queue = graph.batches[destination.uses.front().first].queue;
		if (queue == graphicsQueue || queue >= queueLists.size() || !queueLists[queue]) continue;
		const auto* packet = dynamic_cast<const compiler::PreparedRhiExecutionBatch*>(packets[out.firstOnQueue[queue]].get());
		if (!packet) continue;
		const auto& ordered = orderedAfter(queue);
		if (!std::ranges::all_of(destination.uses, [&](const auto& use) { return ordered[use.first] != 0; })) continue;
		// What a write of it on this queue, ahead of the execution, must wait for: every earlier use the ledger holds (any queue;
		// its own queue is ordered by the list's barrier), and what each batch using it waits for at entry.
		waits.clear();
		const uint64_t ownTimeline = segment.admission.queueTimelines.at(queue);
		auto addWait = [&](compiler::ExecutionTimelinePoint point) {
			if (!point.timeline || !point.value || point.timeline == ownTimeline) return;
			const auto found = std::ranges::find(waits, point.timeline, &compiler::ExecutionTimelinePoint::timeline);
			if (found == waits.end()) waits.push_back(point);
			else found->value = (std::max)(found->value, point.value);
		};
		for (const auto& [batch, resource] : destination.uses) {
			if (batch < incoming.size()) for (const auto point : incoming[batch]) addWait(point);
			for (const auto& [waitResource, point] : boundary[batch])
				if (waitResource == resource) addWait(point);
		}
		if (!std::ranges::all_of(waits, [&](const auto& point) { return packet->BindsTimeline(point.timeline); })) continue;
		out.routed.emplace(key, queue);
		for (const auto point : waits) {
			auto& queueWaits = out.waits[queue];
			const auto found = std::ranges::find(queueWaits, point.timeline, &compiler::ExecutionTimelinePoint::timeline);
			if (found == queueWaits.end()) queueWaits.push_back(point);
			else found->value = (std::max)(found->value, point.value);
		}
	}
	for (const auto handle : staged)
		if (const auto found = out.routed.find(UploadBufferKey(handle)); found != out.routed.end()) ++out.copies[found->second];

	// The batches off the graphics queue that use what stays on it wait for the graphics list.
	for (const auto& [key, destination] : destinations) {
		if (out.routed.contains(key)) continue;
		for (const auto& use : destination.uses)
			if (graph.batches[use.first].queue != graphicsQueue) out.waitsForUploads[use.first] = 1;
	}
	// A destination that no batch declares has no known reader. A shader can still read it through a bindless index, on any
	// queue. So no batch off the graphics queue may skip the wait.
	for (const auto& [key, destination] : destinations)
		if (destination.uses.empty()) ++out.undeclared;
	if (out.undeclared)
		for (uint32_t batch = 0; batch < batchCount; ++batch)
			if (graph.batches[batch].queue != graphicsQueue) out.waitsForUploads[batch] = 1;
	// Statistics: the late-bound versions' backings that a declaring batch took to its own queue, and those that made a batch off
	// the graphics queue wait for the graphics list.
	for (const auto& [standIn, members] : lateDestinations)
		for (const auto member : members) {
			const auto& destination = destinations.at(member);
			if (destination.uses.empty()) continue;
			if (out.routed.contains(member)) ++out.lateRouted;
			else if (std::ranges::any_of(destination.uses, [&](const auto& use) { return graph.batches[use.first].queue != graphicsQueue; }))
				++out.lateWaits;
		}
	return true;
}
}

bool RenderGraph::HasPendingUploads() const {
	return m_uploadService && m_uploadService->HasPendingUploads();
}

bool RenderGraph::RecordPendingUploads(rhi::Device device, rhi::CommandList& list, uint32_t slot, std::shared_ptr<void>& keepAlive, bool& recorded,
	std::vector<rhi::CommandList>& recordedLists, PersistentTicket* ticket, std::span<rhi::CommandList> queueLists, size_t* queueCopies) {
	BT_ZONE_SCOPE("ORG.Persistent.RecordPendingUploads");
	auto taken = TakePendingUploads(device, slot);
	return RecordTakenUploads(taken, list, keepAlive, recorded, recordedLists, ticket, queueLists, queueCopies);
}

RenderGraph::TakenUploads RenderGraph::TakePendingUploads(rhi::Device device, uint32_t slot) {
	BT_ZONE_SCOPE("ORG.Persistent.TakePendingUploads");
	TakenUploads taken;
	if (!m_uploadService) return taken;
	// Before the staged batches: an outdated recording goes back among them, first. What the lists write is taken too, in case
	// the ticket's uploads are planned by queue.
	if (m_queueRegistry.SlotCount()) taken.graphicsKnown = m_uploadService->TakeRecordedUploadsWithDestinations(taken.recordedLists, slot, taken.graphicsBuffers);
	else m_uploadService->TakeRecordedUploads(taken.recordedLists, slot);
	const auto pass = m_uploadService->GetUploadPass();
	if (auto* immediate = dynamic_cast<IHasImmediateModeCommands*>(pass.get())) {
		try {
			ImmediateExecutionContext context{device, {org::imm::ImmediatePassKind::Copy, m_immediateDispatch,
				&TicketUploadResolveById, &TicketUploadResolveByPtr, nullptr}, slot, nullptr};
			{
				BT_ZONE_SCOPE("ORG.Upload.DrainImmediateCommands");
				immediate->RecordImmediateCommands(context);
			}
			auto effect = immediate->TakeOwnedImmediateSubmissionEffect();
			auto frame = context.list.Finalize();
			if (effect) {
				taken.supported = false;  // tracked uploads signal their completion: the synchronous path's job
				return taken;
			}
			if (!frame.bytecode.empty()) {
				// The copies' destinations are captured now, with the backings the commit left.
				auto copies = org::imm::PreparedBufferCopies::Capture(frame.bytecode, [](ResourceRegistry::RegistryHandle handle) -> BackingAllocationSnapshot {
					if (!handle.IsEphemeral()) return {};
					auto* resource = handle.GetEphemeralPtr();
					if (!resource || resource->HasLayout()) return {};
					auto* backed = dynamic_cast<BackedResource*>(UnwrapDynamic(resource));
					return backed ? backed->CaptureBackingAllocation() : BackingAllocationSnapshot{};
				});
				if (!copies) {
					taken.supported = false;
					return taken;
				}
				copies->AppendBuffers(taken.graphicsBuffers);
				struct Owned {
					std::shared_ptr<const org::imm::PreparedBufferCopies> copies;
					std::unique_ptr<org::imm::KeepAliveBag> bag;
				};
				auto owned = std::make_shared<Owned>(Owned{std::move(copies), std::move(frame.keepAlive)});
				taken.recordCopies = [copies = owned->copies](rhi::CommandList& target) { copies->Record(target); };
				taken.keepAlive = std::move(owned);
			}
		} catch (const std::exception& e) {
			spdlog::error("Ticket uploads could not be taken: {}", e.what());
			taken.supported = false;
			return taken;
		}
	}
	if (!m_uploadService->TakeStagedUploads(slot, taken.staged)) taken.supported = false;
	return taken;
}

bool RenderGraph::RecordTakenUploads(TakenUploads& taken, rhi::CommandList& list, std::shared_ptr<void>& keepAlive, bool& recorded,
	std::vector<rhi::CommandList>& recordedLists, PersistentTicket* ticket, std::span<rhi::CommandList> queueLists, size_t* queueCopies) {
	BT_ZONE_SCOPE("ORG.Persistent.RecordTakenUploads");
	recorded = false;
	recordedLists = std::move(taken.recordedLists);
	taken.recordedLists.clear();
	if (queueCopies) *queueCopies = 0;
	if (ticket) ticket->queueUploads = {};
	if (!taken.supported) return false;
	// Queue-scoped (see RecordPendingUploads' declaration): everything the graphics list and the recorded lists touch is known.
	const bool scoped = ticket && !queueLists.empty() && m_queueRegistry.SlotCount();
	constexpr std::string_view kOwner = "ORG ticket uploads";
	// A list of copies starts with one full barrier against the work before it on its queue, and only when it has a copy.
	auto orderAfterPreviousWork = [](rhi::CommandList& target) {
		const auto full = rhi::FullMemoryBarrier();
		rhi::BarrierBatch barriers{};
		barriers.globals = {&full, 1u};
		target.Barriers(barriers);
	};
	// Every list of copies ends with a full barrier, so the execution after it on its queue sees them (an open execution's
	// handoff does not know of them: they bypass the ledgers).
	auto orderBeforeLaterWork = orderAfterPreviousWork;
	try {
		if (taken.recordCopies) {
			// The copies overwrite buffers the previous execution may still be reading: nothing else orders them
			// after it (the next execution's full barrier comes after them), so they start with one of their own.
			orderAfterPreviousWork(list);
			taken.recordCopies(list);
			keepAlive = std::move(taken.keepAlive);
			recorded = true;
		}
		// Direct staged batches carry resolved destinations: one copy each, after the queued uploads, so a producer's payload
		// replaces what was queued before it. They follow the work before them (the queued copies, or the previous execution)
		// after one barrier, and none is recorded without a batch.
		auto recordAllInto = [&](rhi::CommandList& target) -> size_t {
			if (taken.staged.empty()) return 0;
			orderAfterPreviousWork(target);
			return org::runtime::StagedUploadBatch::RecordCopies(target, taken.staged, kOwner);
		};
		std::vector<rhi::ResourceHandle> staged;
		TicketUploadRouting routing;
		const auto graphicsQueue = scoped ? static_cast<uint32_t>(ToUnderlying(m_queueRegistry.FindGraphicsSlot())) : UINT32_MAX;
		if (!scoped || !taken.graphicsKnown || !org::runtime::StagedUploadBatch::AppendDestinations(taken.staged, staged)
			|| !PlanTicketUploads(*ticket, graphicsQueue, queueLists, taken.graphicsBuffers, staged, routing)) {
			// Not planned: every batch off the graphics queue waits for the uploads.
			if (recordAllInto(list)) recorded = true;
			if (recorded || !recordedLists.empty()) orderBeforeLaterWork(list);
			return true;
		}
		// Planned by queue: each routed destination's copies into its queue's list (behind that list's own barrier against the
		// work before it on that queue), the rest into `list`. Every copy of one destination goes to the same list, in order, so
		// the batches' order between overlapping copies (the later write wins) holds within it.
		size_t routed = 0, copies = 0;
		if (!taken.staged.empty()) {
			std::unordered_map<uint64_t, rhi::CommandList*> byDestination;
			std::vector<rhi::CommandList*> targets;
			for (const auto destination : staged) {
				const auto key = UploadBufferKey(destination);
				if (byDestination.contains(key)) continue;
				const auto found = routing.routed.find(key);
				auto* target = found == routing.routed.end() ? &list : &queueLists[found->second];
				byDestination.emplace(key, target);
				if (std::ranges::find(targets, target) == targets.end()) targets.push_back(target);
			}
			for (auto* target : targets) {
				orderAfterPreviousWork(*target);
				const size_t count = org::runtime::StagedUploadBatch::RecordCopies(*target, taken.staged, kOwner,
					[&](rhi::ResourceHandle destination) {
						const auto found = byDestination.find(UploadBufferKey(destination));
						return found != byDestination.end() && found->second == target;
					});
				copies += count;
				if (target != &list) routed += count;
			}
		}
		size_t expected = 0;
		for (const auto count : routing.copies) expected += count;
		// The batches that use a routed destination do not wait for the graphics list: a copy kept there would race them.
		if (routed != expected) throw std::logic_error("The staged uploads were not recorded on the queues their plan named");
		if (copies > routed) recorded = true;
		auto& plan = ticket->queueUploads;
		plan.planned = true;
		plan.waitsForUploads = std::move(routing.waitsForUploads);
		plan.lateRouted = routing.lateRouted;
		plan.lateWaits = routing.lateWaits;
		plan.undeclared = routing.undeclared;
		const auto& planGraph = *ticket->segments.front().admission.publication->executable->graph;
		for (uint32_t batch = 0; batch < plan.waitsForUploads.size(); ++batch)
			if (planGraph.batches[batch].queue != graphicsQueue && !plan.waitsForUploads[batch]) ++plan.spared;
		for (uint32_t queue = 0; queue < routing.copies.size(); ++queue) {
			if (!routing.copies[queue]) continue;
			plan.leading.push_back({routing.firstOnQueue[queue], queueLists[queue], routing.copies[queue], std::move(routing.waits[queue])});
		}
		if (queueCopies) *queueCopies = routed;
		if (recorded || !recordedLists.empty()) orderBeforeLaterWork(list);
		for (uint32_t queue = 0; queue < routing.copies.size(); ++queue)
			if (routing.copies[queue] && queue < queueLists.size() && queueLists[queue]) orderBeforeLaterWork(queueLists[queue]);
		return true;
	} catch (const std::exception& e) {
		spdlog::error("Ticket uploads could not be recorded: {}", e.what());
		return false;
	}
}

} // namespace org
