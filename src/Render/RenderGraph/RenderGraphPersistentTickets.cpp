// Persistent execution, tickets (async epochs): preparation, kept recordings, submission and pending uploads.
#include "RenderGraphPersistentState.h"

namespace org {

std::shared_ptr<RenderGraph::PersistentTicket> RenderGraph::PreparePersistentTicket(rhi::Device device, uint32_t epoch, uint8_t slot, uint64_t hostFrame,
	uint64_t hostTag, std::shared_ptr<const IHostExecutionData> hostData, bool admissionOnly) {
	BT_ZONE_SCOPE("ORG.Persistent.PrepareTicket");
	if (!m_persistentClosedExecutions) throw std::logic_error("Tickets require closed executions");
	auto ticket = std::make_shared<PersistentTicket>();
	ticket->epoch = epoch;
	ticket->slot = slot;
	ticket->hostFrame = hostFrame;
	ticket->hostTag = hostTag;
	ticket->external = IsExternalPersistentEpoch(epoch);
	m_persistentTicketHostTag = hostTag;
	m_persistentTicketAdmissionOnly = admissionOnly;
	m_persistentTicketHostData = static_cast<bool>(hostData);
	struct TargetScope {
		RenderGraph& graph;
		~TargetScope() {
			graph.m_persistentTicketTarget = nullptr;
			graph.m_persistentTicketAdmissionOnly = false;
			graph.m_persistentTicketHostData = false;
		}
	} scope{*this};
	m_persistentTicketTarget = ticket.get();
	SetPersistentEpoch(epoch);
	UpdateExecutionContext update{};
	update.frameIndex = slot;
	update.preparationSlot = slot;
	update.frameFenceValue = hostFrame + 1;
	update.hostData = hostData.get();
	update.ownedHostData = hostData;
	// What it prepares for resolves the resolvers too: a revision names its resources (its versions of them), never the live ones.
	if (hostData) update.resolverCaptureContext = std::make_shared<const ResolverCaptureContext>(hostData);
	Update(update, device);
	PassExecutionContext execute{};
	execute.device = device;
	execute.frameIndex = slot;
	execute.frameFenceValue = hostFrame + 1;
	execute.hostData = hostData.get();
	execute.ownedHostData = hostData;
	Execute(execute);
	return ticket;
}

std::shared_ptr<RenderGraph::PersistentTicket> RenderGraph::PrepareRevisionTicket(uint32_t epoch, uint8_t slot, uint64_t hostFrame) {
	BT_ZONE_SCOPE("ORG.Persistent.PrepareRevisionTicket");
	if (!m_persistentClosedExecutions) throw std::logic_error("Tickets require closed executions");
	auto ticket = std::make_shared<PersistentTicket>();
	ticket->epoch = epoch;
	ticket->slot = slot;
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
	auto& state = *m_compilerState->persistent;
	if (!state.timelines || !state.admission) throw std::logic_error("A revision ticket's candidate before the graph executed");
	// The queues as submitSegment takes them: what each has submitted, a pending presentation tail owning the next graphics value.
	const size_t slotCount = m_queueRegistry.SlotCount();
	const auto presentationSlot = m_queueRegistry.FindGraphicsSlot();
	std::vector<compiler::ExecutionTimelinePoint> queues;
	for (size_t slot = 0; slot < slotCount; ++slot) {
		const auto next = m_queueRegistry.GetCurrentFenceValue(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot)));
		auto submitted = next ? next - 1 : 0;
		if (state.pendingPresentSubmission && slot == static_cast<size_t>(ToUnderlying(presentationSlot))) submitted = next;
		queues.push_back({QueueTimelineIdentity(*m_runtime, m_queueRegistry, slot), submitted});
	}
	auto* candidate = new PersistentTicket::Candidate{
		{state.admission->Prepare(recording->publication, queues, recording->waits, {}, {}), recording, recording->bindings, recording->resourceHeap,
			recording->samplerHeap}};
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
		if (!submission.submitted || !segment.recording) {
			state.admission->Abandon(segment.admission);
			continue;
		}
		std::vector<compiler::ExecutionTimelinePoint> signals;
		signals.reserve(graph.batches.size());
		for (uint32_t batch = 0; batch < graph.batches.size(); ++batch)
			signals.push_back({segment.admission.queueTimelines.at(graph.batches[batch].queue), submission.signals.at(next++)});
		const auto execution = state.timelines->RecordSubmitted(segment.recording->snapshot->layout->bundle,
			segment.admission.incomingWaits, segment.recording->packets, signals);
		state.admission->Commit(segment.admission, *execution);
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
	for (auto* candidate = ticket.candidates.load(std::memory_order_acquire); candidate; candidate = candidate->next)
		if (candidate != ticket.chosen) state.admission->Abandon(candidate->segment.admission);
	for (auto& segment : ticket.displaced) state.admission->Abandon(segment.admission);
	for (auto& segment : ticket.displaced)
		if (segment.recording) ticket.replaced.push_back(std::move(segment.recording));
	ticket.displaced.clear();
	// Abandoned packets are destroyed here, never submitted: their lifecycle effects abandon and their
	// command lists return to the pool. A kept recording stays with its set.
	ticket.segments.clear();
	ticket.replaced.clear();
	basic_telemetry::AddCounter(submission.submitted ? "ORG.Persistent.TicketsSubmitted" : "ORG.Persistent.TicketsAbandoned");
	return completion;
}

uint32_t RenderGraph::PersistentTicketEpoch(const PersistentTicket& ticket) noexcept { return ticket.epoch; }
uint8_t RenderGraph::PersistentTicketSlot(const PersistentTicket& ticket) noexcept { return ticket.slot; }
uint64_t RenderGraph::PersistentTicketHostFrame(const PersistentTicket& ticket) noexcept { return ticket.hostFrame; }
uint64_t RenderGraph::PersistentTicketHostTag(const PersistentTicket& ticket) noexcept { return ticket.hostTag; }
void RenderGraph::SetPersistentTicketHostTag(PersistentTicket& ticket, uint64_t tag) noexcept { ticket.hostTag = tag; }
uint32_t RenderGraph::PersistentTicketSparedUploadWaits(const PersistentTicket& ticket) noexcept {
	return ticket.queueUploads.planned ? ticket.queueUploads.spared : 0;
}

bool RenderGraph::AdoptKeptPersistentRecording(PersistentTicket& ticket, uint64_t hostTag) {
	BT_ZONE_SCOPE("ORG.Persistent.AdoptKeptRecording");
	if (!m_keptRecordings || ticket.uncheckable || ticket.segments.size() != 1 || ticket.epoch >= kKeptRecordingEpochs || ticket.slot >= kKeptRecordingSlots)
		return false;
	auto& segment = ticket.segments.front();
	if (!segment.recording) return false;
	const auto set = (*m_keptRecordings)[ticket.epoch][ticket.slot].load(std::memory_order_acquire);
	if (!set || set->empty()) return false;
	// What the ticket's passes depend on now: the same passes (the same publication), their revisions as they are.
	std::vector<std::pair<const RenderGraphPass*, uint64_t>> now;
	now.reserve(ticket.revisions.size());
	for (const auto& [pass, recorded] : ticket.revisions) {
		uint64_t hash = 0;
		if (!pass->RevisionHash(hash)) return false;
		now.emplace_back(pass, hash);
	}
	const auto& current = *segment.recording;
	for (const auto& kept : *set) {
		if (kept.get() == &current || !kept->Recorded(hostTag, segment.admission.publication.get(), current.bindings.get(), current.resourceHeap, current.samplerHeap, now))
			continue;
		ticket.replaced.push_back(std::move(segment.recording));
		segment.recording = kept;
		kept->lastUse.store(ticket.hostFrame, std::memory_order_relaxed);
		ticket.revisions = std::move(now);
		ticket.hostTag = hostTag;
		m_recordingsAdopted.fetch_add(1, std::memory_order_relaxed);
		return true;
	}
	return false;
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
	stats.kept = m_recordingsKept.exchange(0, std::memory_order_relaxed);
	stats.reused = m_recordingsReused.exchange(0, std::memory_order_relaxed);
	stats.adopted = m_recordingsAdopted.exchange(0, std::memory_order_relaxed);
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

bool RenderGraph::SubmitPersistentTicket(PersistentTicket& ticket, std::span<uint64_t> nextQueueValues, PersistentTicketSubmission& out,
	std::pair<uint32_t, uint64_t> uploadsDone, uint64_t frameWait) {
	BT_ZONE_SCOPE("ORG.Persistent.SubmitTicket");
	out = {};
	ticket.submittedUploads = uploadsDone;
	ticket.submittedFrameWait = frameWait;
	const auto& planned = ticket.queueUploads;
	if (planned.planned && ticket.segments.size() != 1) throw std::logic_error("Queue-scoped uploads planned for a ticket of several segments");
	if (ticket.external && planned.planned) throw std::logic_error("Queue-scoped uploads planned for an external epoch's ticket");
	for (auto& segment : ticket.segments) {
		const auto& graph = *segment.admission.publication->executable->graph;
		// The same timeline composition the owner's RecordSubmitted will derive from these values: the
		// execution's incoming waits plus its batches' waits on each other.
		std::vector<compiler::ExecutionBatchTimeline> batches(graph.batches.size());
		const auto& incoming = segment.admission.incomingWaits;
		if (planned.planned && planned.waitsForUploads.size() != graph.batches.size())
			throw std::logic_error("Queue-scoped uploads planned for other batches");
		for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
			const auto queue = graph.batches[batch].queue;
			if (queue >= nextQueueValues.size()) throw std::logic_error("Ticket batch on an unknown queue");
			batches[batch].signal = {segment.admission.queueTimelines.at(queue), ++nextQueueValues[queue]};
			if (batch < incoming.size()) batches[batch].waits = incoming[batch];
			// The uploads ahead of the ticket order only the batches on their own queue (its entry barrier): any other queue
			// would otherwise read what this epoch's uploads write before they land. Planned by queue (RecordPendingUploads),
			// only the batches that use what they write wait (the rest of theirs went to their own queue, ahead of them).
			if (uploadsDone.second && queue != uploadsDone.first && (!planned.planned || planned.waitsForUploads[batch]))
				batches[batch].waits.push_back({segment.admission.queueTimelines.at(uploadsDone.first), uploadsDone.second});
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
		if (ticket.external) {
			// Not submitted: the host executes the packets in its own submission, which waits for every batch's waits and
			// signals the last batch's value (the batches are all the graphics queue's, in order).
			BT_ZONE_SCOPE("ORG.Persistent.ExternalExecution");
			const auto& packets = segment.recording->packets;
			if (packets.size() != batches.size()) throw std::logic_error("Ticket packets do not match its batches");
			if (!out.external) out.external = std::make_shared<ExternalEpochExecution>();
			auto& execution = *out.external;
			for (size_t i = 0; i < packets.size(); ++i) {
				auto packet = std::dynamic_pointer_cast<const compiler::PreparedRhiExecutionBatch>(packets[i]);
				if (!packet) throw std::logic_error("An external epoch's packet the host cannot execute");
				for (const auto wait : batches[i].waits) {
					const auto timeline = packet->ResolveTimeline(wait.timeline);
					if (!timeline.valid()) throw std::logic_error("An external epoch's wait on a timeline its packet does not bind");
					execution.waits.push_back({timeline, wait.value});
				}
				if (!packet->CommitExternal(batches[i].signal.value, execution.signals)) return false;
				execution.items.push_back({packet, segment.resourceHeap, segment.samplerHeap});
				if (i + 1 == packets.size()) {
					const auto timeline = packet->ResolveTimeline(batches[i].signal.timeline);
					if (!timeline.valid()) throw std::logic_error("An external epoch's signal on a timeline its packet does not bind");
					execution.signals.push_back({timeline, batches[i].signal.value});
				}
			}
			execution.keepAlive.push_back(segment.recording);
			continue;
		}
		{
			BT_ZONE_SCOPE("ORG.Frame.SubmitPackets");
			const auto& packets = segment.recording->packets;
			if (packets.size() != batches.size()) throw std::logic_error("Ticket packets do not match its batches");
			for (size_t i = 0; i < packets.size(); ++i) {
				// The uploads recorded on this batch's queue: on the queue just ahead of it, so it needs no wait for them (its
				// closed execution's entry barrier orders it after them), with the waits their destinations need.
				for (const auto& leading : planned.leading) {
					if (leading.batch != i || !leading.copies) continue;
					const auto* packet = dynamic_cast<const compiler::PreparedRhiExecutionBatch*>(packets[i].get());
					if (!packet) throw std::logic_error("Queue-scoped uploads ahead of a packet that cannot take them");
					auto list = leading.list;
					if (!packet->SubmitLeading({&list, 1}, leading.waits)) return false;
				}
				if (!packets[i]->Submit(batches[i])) return false;
			}
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

bool RenderGraph::RecordPendingUploads(rhi::Device device, rhi::CommandList& list, uint8_t slot, std::shared_ptr<void>& keepAlive, bool& recorded,
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
		const auto graphicsQueue = scoped ? static_cast<uint32_t>(ToUnderlying(m_queueRegistry.FindGraphicsSlot())) : UINT32_MAX;
		if (!scoped || !graphicsKnown || !m_uploadService->StagedUploadDestinations(staged)
			|| !PlanTicketUploads(*ticket, graphicsQueue, queueLists, graphicsBuffers, staged, routing)) {
			// Not planned: every batch off the graphics queue waits for the uploads.
			if (m_uploadService->RecordStagedUploads(list, slot, true)) recorded = true;
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

void RenderGraph::ConfirmPersistentPresentationTail() {
	auto& state = *m_compilerState->persistent;
	if (!state.pendingPresentSubmission || !state.timelines) return;
	const auto slot = m_queueRegistry.FindGraphicsSlot();
	auto queue = m_queueRegistry.GetQueue(slot);
	auto& fence = m_queueRegistry.GetFence(slot);
	const auto value = m_queueRegistry.GetNextFenceValue(slot);
	if (queue.Signal({fence.GetHandle(), value}) != rhi::Result::Ok)
		throw std::runtime_error("Presentation tail completion signal failed");
	const compiler::ExecutionTimelinePoint point{QueueTimelineIdentity(*m_runtime, m_queueRegistry, ToUnderlying(slot)), value};
	state.timelines->ExtendSubmittedExecution(state.pendingPresentSubmission, point);
	state.admission->ExtendSubmitted(point);
	state.pendingPresentSubmission = 0;
	basic_telemetry::AddCounter("ORG.PresentationTail.RetirementReceipts");
}

} // namespace org
