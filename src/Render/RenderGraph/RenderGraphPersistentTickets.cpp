// Persistent execution, tickets (async epochs): preparation, kept recordings, submission and pending uploads.
#include "RenderGraphPersistentState.h"

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

std::shared_ptr<RenderGraph::PersistentTicket> RenderGraph::PrepareRevisionTicket(uint32_t epoch, RunRef run, uint64_t hostFrame) {
	BT_ZONE_SCOPE("ORG.Persistent.PrepareRevisionTicket");
	auto ticket = std::make_shared<PersistentTicket>();
	ticket->epoch = epoch;
	ticket->slot = run.lease;
	ticket->run = run;
	ticket->hostFrame = hostFrame;
	ticket->revision = true;
	ticket->external = IsExternalPersistentEpoch(epoch);
	// What Execute does before it admits: release the executions the GPU has finished.
	RetirePersistentExecutions();
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

bool RenderGraph::SubmitPersistentTicket(PersistentTicket& ticket, std::span<uint64_t> nextQueueValues, PersistentTicketSubmission& out,
	std::pair<uint32_t, uint64_t> uploadsDone, uint64_t frameWait, bool streamUploads) {
	BT_ZONE_SCOPE("ORG.Persistent.SubmitTicket");
	out = {};
	auto& state = *m_compilerState->persistent;
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
	for (auto& segment : ticket.segments) {
		if (!segment.admitted) throw std::logic_error("A ticket submitted before it was admitted (AdmitPersistentTicket)");
		auto& admission = segment.admission;
		const auto& graph = *admission.publication->executable->graph;
		const auto batchCount = static_cast<uint32_t>(graph.batches.size());
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
				const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(release.queue));
				if (release.queue < streamSlots.size() && streamSlots[release.queue]) {
					auto allocation = streamSecondary(release.queue, "ORG external ticket release");
					auto list = allocation->pair.list.Get();
					rhi::BarrierBatch barriers{};
					barriers.textures = {release.textures.data(), static_cast<uint32_t>(release.textures.size())};
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
				barriers.textures = {release.textures.data(), static_cast<uint32_t>(release.textures.size())};
				list.Barriers(barriers);
				if (list.EndChecked() != rhi::Result::Ok) throw std::runtime_error("A ticket's release list could not close");
				allocation->Close();
				const auto value = ++nextQueueValues[release.queue];
				const rhi::TimelinePoint point{m_queueRegistry.GetFence(index).GetHandle(), value};
				rhi::SubmitDesc submit{};
				submit.signals = {&point, 1u};
				if (m_queueRegistry.GetQueue(index).Submit({&list, 1u}, submit) != rhi::Result::Ok) return false;
				m_queueRegistry.EnsureNextFenceValueAtLeast(index, value + 1);
				for (const auto consumer : release.consumers) incoming.at(consumer).push_back({admission.queueTimelines.at(release.queue), value});
				out.leases.push_back(std::move(allocation));
				basic_telemetry::AddCounter("ORG.Persistent.HandoffReleases");
			}
			// The lists the slot's last ticket recorded from the same predicted plan, again; otherwise recorded, and kept for the slot's
			// next ticket.
			PersistentExecutionsState::HandoffCache* cache = nullptr;
			if (admission.sharedBarriers && ticket.segments.size() == 1) {
				if (state.handoffCache.size() <= ticket.slot) state.handoffCache.resize(size_t{ticket.slot} + 1);
				cache = &state.handoffCache[ticket.slot];
				if (cache->plan == admission.sharedBarriers && cache->external == ticket.external && cache->handoff.size() == batchCount) {
					handoff = cache->handoff;
					streamHandoff = cache->streamHandoff;
					out.leases.insert(out.leases.end(), cache->allocations.begin(), cache->allocations.end());
					basic_telemetry::AddCounter("ORG.Persistent.HandoffListsReused");
				} else {
					*cache = {};
				}
			}
			for (uint32_t batch = 0; batch < batchCount && !(cache && cache->plan); ++batch) {
				const auto& entry = admission.Barriers().batches[batch];
				if (entry.textures.empty() && entry.buffers.empty() && entry.acquired.empty()) continue;
				std::shared_ptr<compiler::FrameCommandAllocation> allocation;
				if (inStream(batch)) {
					allocation = streamSecondary(graph.batches[batch].queue, "ORG external ticket handoff");
				} else {
					allocation = std::make_shared<compiler::FrameCommandAllocation>();
					allocation->pool = m_queueRegistry.GetSharedPool(static_cast<QueueSlotIndex>(static_cast<uint8_t>(graph.batches[batch].queue)));
					allocation->pair = allocation->pool->Request();
				}
				auto list = allocation->pair.list.Get();
				rhi::BarrierBatch barriers{};
				barriers.textures = {entry.textures.data(), static_cast<uint32_t>(entry.textures.size())};
				barriers.buffers = {entry.buffers.data(), static_cast<uint32_t>(entry.buffers.size())};
				list.Barriers(barriers);
				if (!entry.acquired.empty()) {
					rhi::BarrierBatch acquired{};
					acquired.textures = {entry.acquired.data(), static_cast<uint32_t>(entry.acquired.size())};
					list.Barriers(acquired);
				}
				if (list.EndChecked() != rhi::Result::Ok) throw std::runtime_error("A ticket's handoff list could not close");
				allocation->Close();
				(inStream(batch) ? streamHandoff : handoff)[batch] = list;
				if (cache) cache->allocations.push_back(allocation);
				out.leases.push_back(std::move(allocation));
			}
			if (cache && !cache->plan) {
				cache->plan = admission.sharedBarriers;
				cache->external = ticket.external;
				cache->handoff = handoff;
				cache->streamHandoff = streamHandoff;
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
			if (frameWait)
				batches[batch].waits.push_back({kFrameWaitTimelineIdentity, frameWait});
			out.signals.push_back(batches[batch].signal.value);
		}
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
		const auto submitPacket = [&](uint32_t i) -> bool {
			const auto* packet = dynamic_cast<const compiler::PreparedRhiExecutionBatch*>(packets[i].get());
			for (const auto wait : batches[i].waits) {
				if (!packet || packet->BindsTimeline(wait.timeline)) continue;
				const auto handle = m_runtime->TimelineHandleOf(wait.timeline);
				if (handle.valid()) extra[i].push_back({wait.timeline, handle});
			}
			batches[i].extraTimelines = extra[i];
			if (handoff[i]) batches[i].leading = {&handoff[i], 1};
			for (const auto& leading : planned.leading) {
				if (leading.batch != i || !leading.copies) continue;
				if (!packet) throw std::logic_error("Queue-scoped uploads ahead of a packet that cannot take them");
				auto list = leading.list;
				if (!packet->SubmitLeading({&list, 1}, leading.waits)) return false;
			}
			return static_cast<bool>(packets[i]->Submit(batches[i]));
		};
		if (ticket.external) {
			// The batches on the graphics queue are not submitted: the host executes them in its own stream, segment by segment
			// (ExternalEpochExecution). A segment's submission waits for its batches' waits and signals their values.
			BT_ZONE_SCOPE("ORG.Persistent.ExternalExecution");
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
		{
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
			if (m_queueRegistry.GetQueue(index).Submit({&list, 1u}, submit) != rhi::Result::Ok) return false;
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
	for (uint32_t batch = 0; batch < batchCount; ++batch)
		for (const auto& access : graph.boundaryAccessesByBatch[batch]) {
			if (access.resource >= backings.size()) continue;
			const auto& backing = backings[access.resource];
			if (!backing.resource.valid()) continue;
			const auto found = destinations.find(UploadBufferKey(backing.resource));
			if (found == destinations.end()) continue;
			auto& uses = found->second.uses;
			if (uses.empty() || uses.back() != std::pair{batch, access.resource}) uses.push_back({batch, access.resource});
			if (backing.aliasSize || backing.aliasHeapIdentity || backing.aliasHeap) found->second.aliased = true;
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
	return true;
}
}

bool RenderGraph::RecordPendingUploads(rhi::Device device, rhi::CommandList& list, uint32_t slot, std::shared_ptr<void>& keepAlive, bool& recorded,
	std::vector<rhi::CommandList>& recordedLists, PersistentTicket* ticket, std::span<rhi::CommandList> queueLists, size_t* queueCopies) {
	BT_ZONE_SCOPE("ORG.Persistent.RecordPendingUploads");
	recorded = false;
	recordedLists.clear();
	if (queueCopies) *queueCopies = 0;
	if (ticket) ticket->queueUploads = {};
	if (!m_uploadService) return true;
	// Queue-scoped (see the declaration): everything the graphics list and the recorded lists touch, while it is known.
	const bool scoped = ticket && !queueLists.empty() && m_queueRegistry.SlotCount();
	std::vector<rhi::ResourceHandle> graphicsBuffers;
	bool graphicsKnown = true;
	// Before the staged batches: an outdated recording goes back among them, first.
	if (scoped) graphicsKnown = m_uploadService->TakeRecordedUploadsWithDestinations(recordedLists, slot, graphicsBuffers);
	else m_uploadService->TakeRecordedUploads(recordedLists, slot);
	// Every copy follows one full barrier against the work before them (see the declaration).
	bool barrier = false;
	auto orderAfterPreviousWork = [&] {
		if (barrier) return;
		const auto full = rhi::FullMemoryBarrier();
		rhi::BarrierBatch barriers{};
		barriers.globals = {&full, 1u};
		list.Barriers(barriers);
		barrier = true;
	};
	const auto pass = m_uploadService->GetUploadPass();
	auto* immediate = dynamic_cast<IHasImmediateModeCommands*>(pass.get());
	auto recordStaged = [&]() -> bool {
		// Direct staged batches carry resolved destinations: one copy each, after the queued uploads, so a
		// producer's payload replaces what was queued before it. The service orders them after the work before them
		// (the queued copies, or the previous execution) with one barrier, and records none when it has no batch.
		std::vector<rhi::ResourceHandle> staged;
		TicketUploadRouting routing;
		// Every list of copies ends with a full barrier, so the execution after it on its queue sees them (an open execution's
		// handoff does not know of them: they bypass the ledgers).
		auto orderBeforeLaterWork = [](rhi::CommandList& target) {
			const auto full = rhi::FullMemoryBarrier();
			rhi::BarrierBatch barriers{};
			barriers.globals = {&full, 1u};
			target.Barriers(barriers);
		};
		const auto graphicsQueue = scoped ? static_cast<uint32_t>(ToUnderlying(m_queueRegistry.FindGraphicsSlot())) : UINT32_MAX;
		if (!scoped || !graphicsKnown || !m_uploadService->StagedUploadDestinations(staged)
			|| !PlanTicketUploads(*ticket, graphicsQueue, queueLists, graphicsBuffers, staged, routing)) {
			// Not planned: every batch off the graphics queue waits for the uploads.
			if (m_uploadService->RecordStagedUploads(list, slot, true)) recorded = true;
			if (recorded || !recordedLists.empty()) orderBeforeLaterWork(list);
			return true;
		}
		// Planned by queue: each routed destination's copies into its queue's list (behind that list's own barrier against the
		// work before it on that queue), the rest into `list`.
		size_t routed = 0;
		const size_t copies = m_uploadService->RecordStagedUploadsRouted(list, slot, true,
			[&](rhi::ResourceHandle destination) -> rhi::CommandList* {
				const auto found = routing.routed.find(UploadBufferKey(destination));
				return found == routing.routed.end() ? nullptr : &queueLists[found->second];
			}, routed);
		size_t expected = 0;
		for (const auto count : routing.copies) expected += count;
		// The batches that use a routed destination do not wait for the graphics list: a service that kept a copy there would
		// race them.
		if (routed != expected) throw std::logic_error("The upload service did not record the staged uploads on the queues asked");
		if (copies > routed) recorded = true;
		auto& plan = ticket->queueUploads;
		plan.planned = true;
		plan.waitsForUploads = std::move(routing.waitsForUploads);
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
	};
	if (!immediate) {
		try {
			return recordStaged();
		} catch (const std::exception& e) {
			spdlog::error("Ticket uploads could not be recorded: {}", e.what());
			return false;
		}
	}
	try {
		ImmediateExecutionContext context{device, {org::imm::ImmediatePassKind::Copy, m_immediateDispatch,
			&TicketUploadResolveById, &TicketUploadResolveByPtr, nullptr}, slot, nullptr};
		{
			BT_ZONE_SCOPE("ORG.Upload.DrainImmediateCommands");
			immediate->RecordImmediateCommands(context);
		}
		auto effect = immediate->TakeOwnedImmediateSubmissionEffect();
		auto frame = context.list.Finalize();
		if (effect) return false;  // tracked uploads signal their completion: the synchronous path's job
		if (frame.bytecode.empty()) return recordStaged();
		auto copies = org::imm::PreparedBufferCopies::Capture(frame.bytecode, [](ResourceRegistry::RegistryHandle handle) -> BackingAllocationSnapshot {
			if (!handle.IsEphemeral()) return {};
			auto* resource = handle.GetEphemeralPtr();
			if (!resource || resource->HasLayout()) return {};
			auto* backed = dynamic_cast<BackedResource*>(UnwrapDynamic(resource));
			return backed ? backed->CaptureBackingAllocation() : BackingAllocationSnapshot{};
		});
		if (!copies) return false;
		// The copies overwrite buffers the previous execution may still be reading: nothing else orders them
		// after it (the next execution's full barrier comes after them), so they start with one of their own.
		orderAfterPreviousWork();
		copies->Record(list);
		if (scoped) copies->AppendBuffers(graphicsBuffers);
		struct Owned {
			std::shared_ptr<const org::imm::PreparedBufferCopies> copies;
			std::unique_ptr<org::imm::KeepAliveBag> bag;
		};
		keepAlive = std::make_shared<Owned>(Owned{std::move(copies), std::move(frame.keepAlive)});
		recorded = true;
		return recordStaged();
	} catch (const std::exception& e) {
		spdlog::error("Ticket uploads could not be recorded: {}", e.what());
		return false;
	}
}

} // namespace org
