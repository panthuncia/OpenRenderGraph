// Persistent execution, the synchronous route: admission, recording and submission of a frame's segment, and retirement.
#include "RenderGraphPersistentState.h"

namespace org {

namespace {
// The pass indices of a graph without a statistics service: none.
const std::unordered_map<std::string_view, unsigned>& NoStatisticsIndices() {
	static const std::unordered_map<std::string_view, unsigned> none;
	return none;
}
}

// ---------------------------------------------------------------------------
// Execution

void RenderGraph::ExecutePersistentFrame(PassExecutionContext& context) {
	BT_ZONE_SCOPE("ORG.Persistent.Execute");
	auto& compiler = *m_compilerState;
	auto& state = *compiler.persistent;
	if (!state.prepared) throw std::logic_error("Persistent execution has no prepared frame");
	// Taken out of the state now: whatever happens below, it is not executed twice.
	auto execution = std::move(*state.prepared);
	state.prepared.reset();
	ExecutePersistentRun(execution, context, false);
}

std::shared_ptr<RenderGraph::PersistentExecution> RenderGraph::TakePersistentExecution() {
	BT_ZONE_SCOPE("ORG.Persistent.TakeExecution");
	if (m_compilerState->frameProductionStopped) throw std::logic_error("Frame production has stopped");
	if (!m_compilerState->persistent) throw std::logic_error("RenderGraph::Execute before a persistent frame was prepared");
	auto& compiler = *m_compilerState;
	auto& state = *compiler.persistent;
	if (!state.prepared) throw std::logic_error("Persistent execution has no prepared frame");
	if (state.prepared->options.ticket) throw std::logic_error("A ticket's execution cannot be taken");
	auto execution = std::make_shared<PersistentExecution>(std::move(*state.prepared));
	state.prepared.reset();
	// Retirement reads and changes graph state that the owner thread also reads (the transient resources by name, the retired
	// pins), so it stays here. The graph's earlier executions have gone through the submitting thread by now.
	RetirePersistentExecutions();
	compiler.lastExecutedPreparationSlot = execution->frameIndex;
	compiler.lastSubmittedFrameData = execution->frameData;
	return execution;
}

void RenderGraph::ExecuteTakenPersistentFrame(PersistentExecution& execution, PassExecutionContext& context) {
	BT_ZONE_SCOPE("ORG.Persistent.Execute");
	if (!context.beginGpuPassRange && !context.endGpuPassRange && m_gpuPassRangeBegin && m_gpuPassRangeEnd) {
		context.beginGpuPassRange = m_gpuPassRangeBegin;
		context.endGpuPassRange = m_gpuPassRangeEnd;
	}
	ExecutePersistentRun(execution, context, true);
}

void RenderGraph::RunPersistentAfterExecute() {
	BT_ZONE_SCOPE("ORG.Persistent.AfterExecute");
	for (auto& extension : m_extensions)
		if (extension) extension->AfterExecute(*this);
}

void RenderGraph::ExecutePersistentRun(PersistentExecution& execution, PassExecutionContext& context, bool taken) {
	auto& compiler = *m_compilerState;
	auto& state = *compiler.persistent;
	const auto* const ticket = execution.options.ticket;
	auto& timings = m_lastPersistentExecuteTimings;
	timings = {};
	m_lastPresentDependency.reset();
	context.immediateDispatch = &m_immediateDispatch;
	const size_t slotCount = m_queueRegistry.SlotCount();
	if (!state.timelines) {
		std::vector<compiler::ExecutionTimelinePoint> queues;
		for (size_t slot = 0; slot < slotCount; ++slot) {
			const auto next = m_queueRegistry.GetCurrentFenceValue(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot)));
			queues.push_back({QueueTimelineIdentity(*m_runtime, m_queueRegistry, slot), next ? next - 1 : 0});
		}
		const size_t framesInFlight = m_renderGraphSettingsService ? m_renderGraphSettingsService->GetNumFramesInFlight() : 3;
		state.timelines = std::make_unique<compiler::ExecutionTimelineAdmission>(std::move(queues), framesInFlight * 3 + 6);
	}
	// A timeline other graphs share (SetSharedQueueTimelines): a synchronous execution's values come after theirs. A ticket's are
	// assigned at its submission, after everything submitted before it, and recorded in that order (CompletePersistentTicket).
	if (!ticket) for (size_t slot = 0; slot < slotCount; ++slot) {
		const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot));
		if (!m_queueRegistry.SharesTimeline(index)) continue;
		if (const auto next = m_queueRegistry.GetCurrentFenceValue(index); next > 1) state.timelines->AdvanceTo(static_cast<uint32_t>(slot), next - 1);
	}
	// Retirement runs before this frame submits: the presentation tail of the
	// previous frame has been confirmed by now, so completed executions and
	// their publications can be released.
	// A taken execution's retirement ran on the owner thread when it was taken.
	if (!taken) {
		auto retireStart = std::chrono::steady_clock::now();
		RetirePersistentExecutions();
		timings.retireUs += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - retireStart).count();
	}
	// A table of the pass names now: passes register on build threads meanwhile (RegisterPass).
	const auto statisticsIndices = m_statisticsService ? m_statisticsService->PassIndices() : nullptr;
	state.tracyFrameBegun.assign(slotCount, 0);
	const auto presentationSlot = m_queueRegistry.FindGraphicsSlot();

	// The uploads go first. The main segment's admission comes after them, so it starts from the states they leave.
	SubmitPersistentUploads(execution);

	auto record = std::make_shared<detail::ExecutionRecord>(execution.number);
	state.lastExecution = record;
	if (ticket) execution.options.ticket->execution = record;
	if (execution.main) {
		BT_ZONE_SCOPE("ORG.Persistent.SubmitSegment");
		auto& segment = *execution.main;
		BT_ZONE_TEXT(segment.label, std::strlen(segment.label));
		PersistentSegmentRun run{.execution = execution, .segment = segment, .timings = timings};
		run.traceSchedule = m_scheduleTraceEnabled.load(std::memory_order_relaxed);
		run.slotCount = slotCount;
		run.presentationSlot = presentationSlot;
		run.resourceHeap = context.GetResourceDescriptorHeap().GetHandle();
		run.samplerHeap = context.GetSamplerDescriptorHeap().GetHandle();
		run.statisticsIndices = statisticsIndices ? &statisticsIndices->indices : &NoStatisticsIndices();
		run.firstBatchOnSlot.assign(slotCount, UINT32_MAX);
		run.lastBatchOnSlot.assign(slotCount, UINT32_MAX);
		const auto& batches = segment.publication->executable->graph->batches;
		for (uint32_t batch = 0; batch < batches.size(); ++batch) {
			const auto slot = batches[batch].queue;
			if (slot >= slotCount) continue;
			if (run.firstBatchOnSlot[slot] == UINT32_MAX) run.firstBatchOnSlot[slot] = batch;
			run.lastBatchOnSlot[slot] = batch;
		}
		try {
			if (AdmitPersistentSegment(run)) {
				// A current kept recording (PersistentExecutionOptions::keptCopy) is submitted again: nothing is recorded. A partial
				// run (batch reuse) records its stale batches.
				if (segment.replay && segment.reuseBatches.empty()) {
					run.Lap(timings.admissionUs);
					++timings.segments;
					timings.batches += static_cast<uint32_t>(segment.publication->executable->graph->batches.size());
				} else {
					RecordPersistentSegment(run);
				}
				SubmitPersistentSegment(run);
			}
		} catch (...) {
			record->Finish(true);
			throw;
		}
	}
	// A ticket's execution is submitted when the ticket completes. One with nothing submitted here is complete now.
	record->MarkAdmitted();
	if (!ticket && record->milestone.load(std::memory_order_acquire) == ExecutionMilestone::Admitted) record->Finish(false);
	// A taken execution's owner set these when it took it, and runs AfterExecute itself.
	if (taken) {
		basic_telemetry::AddCounter("ORG.Persistent.SubmittedFrames");
		return;
	}
	compiler.lastExecutedPreparationSlot = execution.frameIndex;
	if (!ticket) {
		BT_ZONE_SCOPE("ORG.Persistent.AfterExecute");
		for (auto& extension : m_extensions)
			if (extension) extension->AfterExecute(*this);
	}
	compiler.lastSubmittedFrameData = execution.frameData;
	basic_telemetry::AddCounter("ORG.Persistent.SubmittedFrames");
}

// Each upload goes through the graph's external client, on the graphics queue.
void RenderGraph::SubmitPersistentUploads(PersistentExecution& execution) {
	if (execution.uploads.empty()) return;
	BT_ZONE_SCOPE("ORG.Persistent.SubmitUploads");
	auto& client = GetExternalClient();
	const auto graphics = static_cast<uint32_t>(ToUnderlying(m_queueRegistry.FindGraphicsSlot()));
	for (auto& upload : execution.uploads) {
		std::vector<rhi::TimelinePoint> signals;
		for (const auto& signal : upload.signals) signals.push_back({signal.timeline.GetHandle(), signal.value});
		const auto& bytecode = upload.bytecode;
		client.Submit(graphics, upload.accesses, [&](rhi::CommandList& list) {
			if (!bytecode.empty()) org::imm::Replay(bytecode, list, m_immediateDispatch);
		}, signals, std::move(upload.keepAlive));
		if (upload.commit) upload.commit();
	}
	execution.uploads.clear();
}

// Prepares the segment's admission against the ledgers, and settles the cases that need nothing more: an empty segment, a
// ticket that only admits, and a ticket that takes a kept recording.
bool RenderGraph::AdmitPersistentSegment(PersistentSegmentRun& run) {
	auto& state = *m_compilerState->persistent;
	auto& segment = run.segment;
	const size_t slotCount = run.slotCount;
	std::vector<compiler::ExecutionTimelinePoint> queues;
	for (size_t slot = 0; slot < slotCount; ++slot) {
		const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot));
		const auto next = m_queueRegistry.GetCurrentFenceValue(index);
		const auto submitted = next ? next - 1 : 0;
		const auto identity = QueueTimelineIdentity(*m_runtime, m_queueRegistry, slot);
		queues.push_back({identity, submitted});
		run.timelineBindings.push_back({identity, m_queueRegistry.GetFence(index).GetHandle()});
	}
	run.timelineBindings.insert(run.timelineBindings.end(), segment.foreignTimelines.begin(), segment.foreignTimelines.end());
	if (HasFrameWaitTimeline())
		run.timelineBindings.push_back({kFrameWaitTimelineIdentity, (*m_frameWaitTimeline)->GetHandle()});
	run.admission = [&] {
		BT_ZONE_SCOPE("ORG.Persistent.PrepareAdmission");
		// A ticket's body is recorded from a plan: it is admitted when it is submitted (AdmitPersistentTicket). So is a staged
		// recording's (RecordStagedCopy): its copy's first run admits it.
		if (run.execution.options.ticket || run.staged)
			return state.admission->PreparePlan(segment.publication, queues, segment.waits, segment.rebindings);
		return state.admission->Prepare(segment.publication, queues, segment.waits, {}, segment.rebindings);
	}();
	auto& admission = run.admission;
	// A segment with no batches (an epoch none of whose passes the graph has, as in the empty bootstrap graph) records and submits
	// nothing: its admission is abandoned, and a ticket gets no segment for it (an external one an empty execution).
	if (segment.publication->executable->graph->batches.empty()) {
		for (const auto& invocation : segment.invocations)
			if (invocation) invocation.Abandon(AbandonReason::AdmissionFailed);
		state.admission->Abandon(admission);
		return false;
	}
	const auto& options = run.execution.options;
	// Executed now (not a ticket, which takes its wait when submitted): every batch waits for the frame's values, or with a
	// scoped frame wait only the batches that read them.
	if (!options.ticket && !run.staged && HasFrameWaitTimeline() && m_frameWaitValue) {
		std::vector<uint8_t> readers;
		if (m_scopedFrameWait) readers = persistent::FrameWaitReaders(*segment.publication->executable->graph, admission.backings);
		for (auto* set : {&admission.incomingWaits, &admission.ownWaits})  // ownWaits: kept by its readmission
			for (size_t batch = 0; batch < set->size(); ++batch)
				if (readers.empty() || (batch < readers.size() && readers[batch])) (*set)[batch].push_back({kFrameWaitTimelineIdentity, m_frameWaitValue});
	}
	// Replayable recordings (SetPersistentRecordingReuse): a ticket's execution with nothing late-bound, whose passes all
	// report their revisions, is recorded for resubmission. Only a revision's published copies are submitted again.
	auto* const ticketTarget = options.ticket;
	if (ticketTarget && options.admissionOnly) {
		// Recorded ahead (BindPersistentTicketRecording): this preparation's invocations are never recorded.
		for (const auto& invocation : segment.invocations)
			if (invocation) invocation.Abandon(AbandonReason::AdmissionFailed);
		ticketTarget->segments.push_back({std::move(admission), nullptr, segment.bindings, run.resourceHeap, run.samplerHeap,
			segment.waits, segment.rebindings});
		return false;
	}
	// An external epoch's recording is replayable as any other's (its secondaries executed again by each ticket that takes it); a
	// late batch's passes are recorded by each execution (the host's, or the submission's when inline), so it is replayable only
	// when they are (PreparedPass::RecordAgain: their invocation recorded again by each).
	bool lateUnreplayable = false;
	const auto& structure = *segment.publication->executable->graph->structure;
	const auto& batches = segment.publication->executable->graph->batches;
	// A partial run (batch reuse): the batches submitted from the kept recording prepared nothing.
	const bool partial = !segment.reuseBatches.empty();
	std::vector<uint8_t> lateBatch(batches.size());
	if (!segment.replay || partial)
		for (uint32_t index = 0; index < batches.size(); ++index) {
			if (partial && index < segment.reuseBatches.size() && segment.reuseBatches[index]) continue;
			for (const auto pass : batches[index].passes) {
				if (!structure.passes[pass].recordAtExecution) continue;
				const auto prepared = structure.passes[pass].preparedPassIndex;
				const bool unreplayable = prepared >= segment.invocations.size() || !segment.invocations[prepared].IsReplayable();
				lateUnreplayable = lateUnreplayable || unreplayable;
				if (unreplayable) lateBatch[index] = 1;
			}
		}
	run.external = ticketTarget && ticketTarget->external;
	// A body depends only on the compiled graph (its entry barriers are recorded at submission), so a ticket's recording can be
	// replayable.
	// A recording made with versions keeps its frame rebindings and is admitted with them again. A per-frame resource's
	// version (a swapchain image) must be the frame's own, so such a recording is never kept.
	run.keepable = ticketTarget && m_persistentRecordingReuse && !segment.perFrame
		&& admission.incomingEffects.empty() && !ticketTarget->uncheckable
		&& !lateUnreplayable;
	// A synchronous execution for a kept copy (PersistentExecutionOptions::keptCopy) is recorded for resubmission by the same
	// rules, its passes' revisions taken by its preparation.
	const bool batchReuse = m_keptBatchReuse && !ticketTarget && segment.keptCopy != UINT32_MAX && !run.staged;
	if (!ticketTarget && segment.keptCopy != UINT32_MAX && (!segment.replay || partial) && batchReuse) {
		// Batch reuse (SetKeptBatchReuse): what every batch shares decides whether anything is kept; each batch's own passes
		// decide whether its packet is.
		run.keepable = !segment.perFrame && admission.incomingEffects.empty();
		if (segment.perFrame) run.unkeptReason = "a per-frame version is bound (a swapchain image)";
		else if (!admission.incomingEffects.empty()) run.unkeptReason = "its admission has incoming effects";
		run.batchKeepable.assign(batches.size(), run.keepable ? 1 : 0);
		run.batchUnkept.assign(batches.size(), run.unkeptReason);
		if (run.keepable) {
			const auto& batchOf = BatchOfPrepared(state, segment.publication->executable);
			for (const auto& [prepared, why] : segment.uncheckablePrepared) {
				const auto batch = prepared < batchOf.size() ? batchOf[prepared] : UINT32_MAX;
				if (batch >= batches.size() || !run.batchKeepable[batch]) continue;
				run.batchKeepable[batch] = 0;
				run.batchUnkept[batch] = "uncheckable: " + why;
			}
			for (uint32_t index = 0; index < batches.size(); ++index)
				if (lateBatch[index] && run.batchKeepable[index]) {
					run.batchKeepable[index] = 0;
					run.batchUnkept[index] = "a pass recorded at execution is not replayable";
				}
			// A batch submitted again was kept already.
			for (uint32_t index = 0; partial && index < batches.size() && index < segment.reuseBatches.size(); ++index)
				if (segment.reuseBatches[index]) run.batchKeepable[index] = 1;
			if (std::ranges::none_of(run.batchKeepable, [](uint8_t keep) { return keep != 0; })) {
				run.keepable = false;
				const auto first = std::ranges::find_if(run.batchUnkept, [](const auto& why) { return !why.empty(); });
				run.unkeptReason = first != run.batchUnkept.end() ? *first : std::string("no batch can be kept");
			}
		}
	} else if (!ticketTarget && segment.keptCopy != UINT32_MAX && !segment.replay) {
		run.keepable = !segment.uncheckable && !segment.perFrame && admission.incomingEffects.empty() && !lateUnreplayable;
		if (segment.uncheckable) run.unkeptReason = "uncheckable: " + segment.uncheckableBy;
		else if (segment.perFrame) run.unkeptReason = "a per-frame version is bound (a swapchain image)";
		else if (!admission.incomingEffects.empty()) run.unkeptReason = "its admission has incoming effects";
		else if (lateUnreplayable) run.unkeptReason = "a pass recorded at execution is not replayable";
	}
	return true;
}

// Seals the admitted invocations, sets up one recording job per batch with its command list, and records them.
void RenderGraph::RecordPersistentSegment(PersistentSegmentRun& run) {
	auto& state = *m_compilerState->persistent;
	const auto frameIndex = run.execution.frameIndex;
	auto& segment = run.segment;
	auto& admission = run.admission;
	auto& timings = run.timings;
	const size_t slotCount = run.slotCount;
	const bool keepable = run.keepable;
	const bool external = run.external;
	auto* const ticketTarget = run.execution.options.ticket;
	// A partial run (batch reuse): the batches submitted from the kept recording have no invocations and record nothing.
	const bool partial = !segment.reuseBatches.empty() && segment.replay;
	const auto reusedBatch = [&](uint32_t batch) { return partial && batch < segment.reuseBatches.size() && segment.reuseBatches[batch] != 0; };
	std::vector<uint8_t> absent;
	if (partial) {
		const auto& batchOf = BatchOfPrepared(state, segment.publication->executable);
		absent.resize(segment.invocations.size());
		for (size_t prepared = 0; prepared < absent.size() && prepared < batchOf.size(); ++prepared)
			absent[prepared] = batchOf[prepared] != UINT32_MAX && reusedBatch(batchOf[prepared]) ? 1 : 0;
	}
	try {
		run.sealed = compiler::SealPersistentFrame(run.execution.number, admission, std::move(segment.invocations), absent);
	} catch (...) { state.admission->Abandon(admission); throw; }
	const auto& graph = *segment.publication->executable->graph;
	auto& jobs = run.jobs;
	jobs.resize(graph.batches.size());
	// An open ticket's resources every use of which is skipped ahead: no list records their barriers, and every execution of the
	// recording leaves them where it found them (SubmitPersistentTicket's untouched resources hold them).
	run.ahead.reset();
	if (ticketTarget && !external && !state.admission->ExternalHomeStates()
		&& !diagnostics::DiagnosticKeepSkippedPassBarriers().load(std::memory_order_relaxed)) {
		std::vector<std::vector<uint8_t>> skippedAhead(graph.batches.size());
		bool any = false;
		for (uint32_t batch = 0; batch < graph.batches.size(); ++batch)
			for (const auto pass : graph.batches[batch].passes) {
				const auto prepared = graph.structure->passes.at(pass).preparedPassIndex;
				const bool skipped = prepared < run.sealed->passes.size() && run.sealed->passes[prepared].SkippedAhead();
				skippedAhead[batch].push_back(skipped ? 1 : 0);
				any = any || skipped;
			}
		if (any) {
			auto ahead = std::make_shared<compiler::UntouchedResources>(compiler::FindUntouched(graph, admission.backings, skippedAhead));
			if (!ahead->Empty()) run.ahead = std::move(ahead);
		}
	}
	std::array<std::shared_ptr<const compiler::OwnedRecordingGpuRanges>, 3> gpuPassRanges;  // per rhi::QueueKind
	std::vector<size_t> demand(slotCount);
	// An external epoch's ticket (SetExternalPersistentEpochs): its batches on the graphics queue are recorded into secondaries,
	// which the host executes in its own stream, and a late batch's pass (recordAtExecution) by the host's execution. Its batches on
	// other queues are recorded and submitted as any other's.
	// Any other epoch's late batch, under SetInlineAtExecution: recorded by its packet's submission, into a primary list of the
	// queue's shared pool, in the batch's place (OwnedRecordingList::inlineAtExecution).
	for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
		const auto slot = graph.batches[batch].queue;
		if (slot >= slotCount) throw std::runtime_error("Persistent batch queue is unavailable");
		if (reusedBatch(batch)) {
			auto& job = jobs[batch];
			job.slot = static_cast<uint32_t>(slot);
			job.reuse = batch < segment.replay->packets.size() ? segment.replay->packets[batch] : nullptr;
			if (!job.reuse) throw std::logic_error("A reused batch has no kept packet");
			// Its statistics stay the kept recording's (one per batch, or none without a statistics service).
			if (m_statisticsService)
				run.recordingStatistics.push_back(batch < segment.replay->statistics.size() ? segment.replay->statistics[batch] : nullptr);
			continue;
		}
		// Batch reuse: whether this batch's new recording can be kept (its passes all report revisions).
		const bool batchKeepable = keepable && (run.batchKeepable.empty() || (batch < run.batchKeepable.size() && run.batchKeepable[batch]));
		bool late = false;
		for (const auto pass : graph.batches[batch].passes) late = late || graph.structure->passes[pass].recordAtExecution;
		const bool inStream = external && InHostStream(slot);
		if (late && external && !inStream) {
			std::string names;
			for (const auto pass : graph.batches[batch].passes) {
				const auto prepared = graph.structure->passes[pass].preparedPassIndex;
				if (!names.empty()) names += ", ";
				names += segment.names && prepared < segment.names->size() ? (*segment.names)[prepared] : std::string("<unnamed>");
			}
			throw std::logic_error("An external epoch's passes recorded at execution must run on the graphics queue: " + names);
		}
		if (late && !external && !m_inlineAtExecution)
			throw std::logic_error("A pass recorded at execution is not an external epoch's, and the graph records none inline (SetInlineAtExecution)");
		if (!late) ++demand[slot];
		const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot));
		const auto queueKind = m_queueRegistry.GetKind(index);
		const auto rhiKind = queueKind == QueueKind::Graphics ? rhi::QueueKind::Graphics
			: queueKind == QueueKind::Compute ? rhi::QueueKind::Compute : rhi::QueueKind::Copy;
		auto& job = jobs[batch];
		job.slot = static_cast<uint32_t>(slot);
		job.queueKind = rhiKind;
		job.device = m_queueRegistry.GetDevice(index);
		job.queue = m_queueRegistry.GetQueue(index);
		job.pool = inStream ? ExternalSecondaryPool(slot) : m_queueRegistry.GetSharedPool(index);
		if (!job.pool) throw std::logic_error("Recording queue has no command-list pool");
		const auto device = m_queueRegistry.GetBackendInstance(index);
		if (m_backendDevices.size() > 1) {
			// Each device waits on and signals its own copy of every queue timeline.
			job.timelines = run.timelineBindings;
			for (size_t source = 0; source < slotCount; ++source)
				job.timelines[source].handle = m_queueRegistry.GetFenceForConsumer(
					static_cast<QueueSlotIndex>(static_cast<uint8_t>(source)), index).GetHandle();
		}
		job.deferred = late;
		job.recording = compiler::BuildPersistentRecordingList(run.sealed, batch, run.ahead.get());
		// The handoff into the execution (its batches' entry barriers) is recorded when it submits, against the ledger as it is
		// then: the body has none.
		job.recording.textureBarriers.clear();
		job.recording.bufferBarriers.clear();
		job.recording.acquiredBarriers.clear();
		job.recording.external = inStream;
		// Inline: the packet records the batch when it submits it, from the queue's shared pool (AcquireCommandLists skips it).
		job.recording.inlineAtExecution = late && !external;
		if (job.recording.inlineAtExecution) job.recording.allocation->pool = job.pool;
		job.recording.bindings = segment.bindings;
		job.recording.run = run.execution.options.run.value_or(RunRef::Unleased(frameIndex));
		job.recording.tags = m_recordingTags;
		job.recording.externalEntryBarrier = m_externalQueueBoundary.entry && run.firstBatchOnSlot[slot] == batch;
		job.recording.externalExitBarrier = m_externalQueueBoundary.exit && run.lastBatchOnSlot[slot] == batch;
		job.recording.replayable = batchKeepable;
		if (queueKind != QueueKind::Copy && device == BackendInstanceId::Primary) {
			job.recording.resourceDescriptorHeap = run.resourceHeap;
			job.recording.samplerDescriptorHeap = run.samplerHeap;
		} else if (queueKind != QueueKind::Copy) {
			auto& heaps = DescriptorHeapManager::GetInstance();
			job.recording.resourceDescriptorHeap = heaps.GetSRVDescriptorHeap(device).GetHandle();
			job.recording.samplerDescriptorHeap = heaps.GetSamplerDescriptorHeap(device).GetHandle();
		}
		// A late batch is recorded after its ticket completes on the owner thread (which publishes the statistics): its frame
		// slot's completion publishes its (OwnedRecordingStatistics::atCompletion). It is never kept: the shared pool's queries.
		if (m_statisticsService) {
			auto statistics = std::make_shared<compiler::OwnedRecordingStatistics>();
			statistics->service = m_statisticsService;
			statistics->frameIndex = frameIndex;
			statistics->queueKind = rhiKind;
			statistics->atCompletion = late;
			// A recording kept for resubmission records no queries of the shared pool: the statistics service recreates it and
			// its readback buffers as passes register (a resubmitted list would resolve into the freed ones), and its slots are
			// frame-indexed. With recording timestamps it owns its own instead: a range per pass, per barrier batch around one
			// (before and after it) and for the list's entry and exit barriers.
			const bool listTimestamps = m_persistentRecordingTimestamps == RecordingTimestampMode::Lists;
			statistics->gpuQueries = queueKind != QueueKind::Copy && (!batchKeepable || m_persistentRecordingTimestamps != RecordingTimestampMode::Off)
				&& m_queueRegistry.GetBackendInstance(index) == BackendInstanceId::Primary;
			for (const auto& pass : job.recording.passes) {
				const auto found = run.statisticsIndices->find(pass.DebugName());
				statistics->passIndices.push_back(found == run.statisticsIndices->end() ? -1 : static_cast<int>(found->second));
			}
			if (batchKeepable && statistics->gpuQueries) {
				// Each pass's range and barrier batches, plus sub-ranges the passes time inside themselves (a draw pass's views
				// or draw groups: BeginRecordingSubRange).
				constexpr uint32_t kSubRanges = 256;
				const auto ranges = listTimestamps ? 1u : static_cast<uint32_t>(job.recording.passes.size() * 3 + 3) + kSubRanges;
				statistics->queries.owned = m_statisticsService->CreateOwnedTimestamps(rhiKind, ranges);
				if (statistics->queries.owned) statistics->queries.owned->listOnly = listTimestamps;
				statistics->gpuQueries = static_cast<bool>(statistics->queries.owned);
			}
			job.recording.statistics = statistics;
			run.recordingStatistics.push_back(std::move(statistics));
		}
		if (m_gpuPassRangeBegin && m_gpuPassRangeEnd) {
			auto& ranges = gpuPassRanges[static_cast<size_t>(rhiKind)];
			if (!ranges) {
				ranges = std::make_shared<compiler::OwnedRecordingGpuRanges>(compiler::OwnedRecordingGpuRanges{m_gpuPassRangeBegin, m_gpuPassRangeEnd,
					rhiKind == rhi::QueueKind::Graphics ? "Graphics" : rhiKind == rhi::QueueKind::Compute ? "Compute" : "Copy"});
			}
			job.recording.gpuPassRanges = ranges;
		}
	}
	// Schedule introspection: an execution submitted here (a ticket's is recorded when the owner completes it) waits for its
	// admission's incoming waits, the frame's values among them, and has the boundary barriers its jobs record.
	if (run.traceSchedule && !ticketTarget) {
		auto& waits = state.scheduleTraces.scratch;
		FillScheduleSubmissionWaits(waits, graph, admission.incomingWaits, {UINT32_MAX, 0}, 0);
		for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
			waits.batches[batch].externalEntryBarrier = jobs[batch].recording.externalEntryBarrier;
			waits.batches[batch].externalExitBarrier = jobs[batch].recording.externalExitBarrier;
		}
	}
	run.Lap(timings.admissionUs);
	++timings.segments;
	timings.batches += static_cast<uint32_t>(graph.batches.size());
	{
		BT_ZONE_SCOPE("ORG.Persistent.AcquireCommandLists");
		for (size_t slot = 0; slot < slotCount; ++slot) {
			if (!demand[slot]) continue;
			const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot));
			// An external ticket's secondaries retire with its execution, on the graphics timeline as well.
			auto pool = external && InHostStream(static_cast<uint32_t>(slot)) ? ExternalSecondaryPool(static_cast<uint32_t>(slot))
				: m_queueRegistry.GetSharedPool(index);
			const auto completed = m_queueRegistry.GetFence(index).GetCompletedValue();
			pool->PublishDemand(demand[slot], completed);
			auto ready = pool->AcquireBatch(demand[slot], completed);
			size_t readyIndex = 0;
			for (auto& job : jobs) {
				if (job.slot != slot || job.deferred || job.reuse) continue;
				job.recording.allocation->pair = std::move(ready[readyIndex++]);
				job.recording.allocation->pool = job.pool;
			}
			if (readyIndex != ready.size()) throw std::logic_error("Command-list batch demand did not match recording jobs");
			if (!state.tracyFrameBegun[slot] && !external) {
				// A late batch has no list yet, so the mark goes in the slot's first list recorded ahead. Not in a list kept for
				// resubmission.
				const auto first = std::ranges::find_if(jobs, [slot](const auto& job) { return job.slot == slot && !job.deferred && !job.reuse; });
				if (first != jobs.end() && !first->recording.replayable) {
					// The frame mark is recorded here, so the list is begun here. Its recorder sees it begun and keeps it.
					auto& allocation = *first->recording.allocation;
					allocation.pool->Begin(allocation.pair);
					first->queue.TracyGpuFrameBegin(allocation.pair.list.Get());
					state.tracyFrameBegun[slot] = 1;
				}
			}
		}
	}
	run.Lap(timings.acquireUs);
	compiler::PlannedFrame plan{run.sealed, std::move(jobs), run.timelineBindings, admission.incomingWaits};
	const size_t concurrency = (std::min<size_t>)(graph.batches.size(), 4u);
	try {
		run.recorded.emplace(compiler::RecordFrame(std::move(plan), m_taskService, concurrency));
	} catch (...) { state.admission->Abandon(admission); throw; }
	run.Lap(timings.recordUs);
}

// A device waits only on its own timelines. A graph's queue timelines are shared between its devices (QueueRegistry's bridge
// fences), but a batch can also wait on timelines the graph does not own: another graph's queues through the runtime's ledgers,
// the frame-wait timeline, a producer's. A batch whose device is not such a timeline's gets a relay: an empty submission on the
// timeline's device waits for the points and signals a fence shared with the batch's device, and the batch waits on that
// (Runtime::RelayWaits).
std::vector<std::vector<compiler::ExecutionTimelinePoint>> RenderGraph::RelayCrossDeviceWaits(PersistentSegmentRun& run,
	const std::vector<std::vector<compiler::ExecutionTimelinePoint>>& waits, std::vector<compiler::PreparedTimelineBinding>& extraTimelines) {
	auto result = waits;
	if (m_backendDevices.size() < 2) return result;
	BT_ZONE_SCOPE("ORG.Persistent.RelayCrossDeviceWaits");
	const auto& graph = *run.segment.publication->executable->graph;
	const size_t slotCount = run.slotCount;
	const auto bindingOf = [&](uint64_t identity) -> rhi::TimelineHandle {
		for (const auto& binding : run.timelineBindings) if (binding.identity == identity) return binding.handle;
		for (const auto& binding : extraTimelines) if (binding.identity == identity) return binding.handle;
		return m_runtime->TimelineHandleOf(identity);
	};
	const auto queueIdentity = [&](uint64_t identity) {
		for (size_t slot = 0; slot < slotCount && slot < run.timelineBindings.size(); ++slot)
			if (run.timelineBindings[slot].identity == identity) return true;  // shared between devices by the bridge fences
		return false;
	};
	const auto ownerOf = [&](uint64_t identity) -> const void* {
		if (identity == kFrameWaitTimelineIdentity) return HasFrameWaitTimeline() ? (*m_frameWaitTimeline)->impl : nullptr;
		return m_runtime->TimelineDeviceOf(identity);
	};
	struct Group {
		const void* owner = nullptr;
		const void* consumer = nullptr;
		std::map<uint64_t, uint64_t> points;  // identity -> highest value
		std::vector<uint32_t> batches;
	};
	std::vector<Group> groups;
	for (uint32_t batch = 0; batch < result.size() && batch < graph.batches.size(); ++batch) {
		const auto slot = static_cast<QueueSlotIndex>(static_cast<uint8_t>(graph.batches[batch].queue));
		const auto* consumerEntry = m_backendDevices.Find(m_queueRegistry.GetBackendInstance(slot));
		if (!consumerEntry) continue;
		const void* consumer = consumerEntry->device.impl;
		for (const auto wait : result[batch]) {
			if (!wait.value || queueIdentity(wait.timeline)) continue;
			const auto* owner = ownerOf(wait.timeline);
			if (!owner || owner == consumer) continue;
			auto group = std::ranges::find_if(groups, [&](const Group& g) { return g.owner == owner && g.consumer == consumer; });
			if (group == groups.end()) group = groups.insert(groups.end(), Group{owner, consumer, {}, {}});
			auto& point = group->points[wait.timeline];
			point = (std::max)(point, wait.value);
			if (std::ranges::find(group->batches, batch) == group->batches.end()) group->batches.push_back(batch);
		}
	}
	for (const auto& group : groups) {
		std::vector<rhi::TimelinePoint> relayed;
		relayed.reserve(group.points.size());
		for (const auto& [identity, pointValue] : group.points) {
			const auto handle = bindingOf(identity);
			if (!handle.valid()) throw std::logic_error("A relayed wait's timeline has no handle");
			relayed.push_back({handle, pointValue});
		}
		const auto relay = m_runtime->RelayWaits(group.owner, group.consumer, relayed);
		basic_telemetry::AddCounter("ORG.Persistent.CrossDeviceRelays");
		for (const auto batch : group.batches) {
			auto& batchWaits = result[batch];
			std::erase_if(batchWaits, [&](const auto& wait) { return group.points.contains(wait.timeline) && ownerOf(wait.timeline) == group.owner; });
			batchWaits.push_back({relay.identity, relay.point.value});
		}
		if (std::ranges::none_of(extraTimelines, [&](const auto& binding) { return binding.identity == relay.identity; }))
			extraTimelines.push_back({relay.identity, relay.point.t});
	}
	return result;
}

// A ticket keeps the recording for its host to submit. Otherwise the handoff is recorded under the sequencer, and the
// execution is submitted and committed.
void RenderGraph::SubmitPersistentSegment(PersistentSegmentRun& run) {
	auto& state = *m_compilerState->persistent;
	auto& segment = run.segment;
	auto& admission = run.admission;
	auto& timings = run.timings;
	const size_t slotCount = run.slotCount;
	const bool keepable = run.keepable;
	const auto& options = run.execution.options;
	const auto& graph = *segment.publication->executable->graph;
	std::shared_ptr<const compiler::GraphExecutionTimeline> execution;
	std::unique_lock<std::recursive_mutex> sequencer;
	// A kept copy's current recording, submitted instead of a new one; or the new recording kept for the copy. A partial run (batch
	// reuse) submits the recorded frame, whose reused batches are the kept recording's packets, and keeps the merged recording.
	const auto& replay = segment.replay;
	const bool partial = replay && !segment.reuseBatches.empty();
	std::shared_ptr<PersistentRecording> keep;
	// Batch reuse: per batch whether the kept recording's packet can be submitted again, and why not.
	std::vector<uint8_t> batchReusable;
	std::vector<std::string> batchWhy;
	try {
		if (auto* ticket = options.ticket) {
			auto& recorded = *run.recorded;
			// Submitted later, by the host's submitting thread; completed (or abandoned) by the owner.
			auto recording = std::make_shared<PersistentRecording>();
			recording->snapshot = recorded.Snapshot();
			recording->packets = recorded.Packets();
			recording->statistics = std::move(run.recordingStatistics);
			recording->replayable = keepable && std::ranges::all_of(recording->packets, [](const auto& packet) { return packet->Replayable(); });
			recording->ahead = run.ahead;
			m_recordingsRecorded.fetch_add(1, std::memory_order_relaxed);
			if (recording->replayable) {
				recording->hostTag = options.hostTag;
				recording->publication = segment.publication;
				recording->bindings = segment.bindings;
				recording->resourceHeap = run.resourceHeap;
				recording->samplerHeap = run.samplerHeap;
				recording->revisions = ticket->revisions;
				recording->lastUse.store(ticket->hostFrame, std::memory_order_relaxed);
				recording->waits = segment.waits;
				recording->rebindings = segment.rebindings;
				recording->versions = segment.versions;
				recording->leases = segment.leases;
			}
			// A recording made for host data is its caller's (BindPersistentTicketRecording). Nothing else keeps a recording: the
			// ticket's is used once.
			ticket->segments.push_back({std::move(admission), std::move(recording), segment.bindings, run.resourceHeap, run.samplerHeap,
				segment.waits, segment.rebindings});
			return;
		}
		if ((!replay || partial) && keepable && segment.keptCopy != UINT32_MAX) {
			// The packets stay valid after their submission: a replayable list is submitted again by the copy's later runs.
			const auto& recorded = *run.recorded;
			keep = std::make_shared<PersistentRecording>();
			keep->snapshot = recorded.Snapshot();
			keep->packets = recorded.Packets();
			keep->statistics = run.recordingStatistics;
			if (!run.batchKeepable.empty()) {
				batchReusable.assign(keep->packets.size(), 0);
				batchWhy.assign(keep->packets.size(), {});
				for (size_t batch = 0; batch < keep->packets.size(); ++batch) {
					const bool reused = partial && batch < segment.reuseBatches.size() && segment.reuseBatches[batch];
					if (reused) batchReusable[batch] = 1;
					else if (batch < run.batchKeepable.size() && !run.batchKeepable[batch]) batchWhy[batch] = run.batchUnkept[batch];
					else if (!keep->packets[batch]->Replayable()) batchWhy[batch] = "a packet is not replayable (a pass's lifecycle effect or callback)";
					else batchReusable[batch] = 1;
				}
				keep->replayable = std::ranges::any_of(batchReusable, [](uint8_t reusable) { return reusable != 0; });
				// Only the packets a later run can submit again are kept. The others stay with the execution until it retires, as
				// an unkept recording's do, and are not released when the copy's next run replaces this recording.
				for (size_t batch = 0; batch < batchReusable.size(); ++batch) {
					if (batchReusable[batch]) continue;
					keep->packets[batch] = nullptr;
					if (batch < keep->statistics.size()) keep->statistics[batch] = nullptr;
				}
			} else {
				keep->replayable = std::ranges::all_of(keep->packets, [](const auto& packet) { return packet->Replayable(); });
			}
			keep->publication = segment.publication;
			keep->bindings = segment.bindings;
			keep->resourceHeap = run.resourceHeap;
			keep->samplerHeap = run.samplerHeap;
			keep->revisions = segment.revisions;
			keep->waits = segment.waits;
			keep->rebindings = segment.rebindings;
			keep->versions = segment.versions;
			keep->leases = segment.leases;
			m_recordingsRecorded.fetch_add(1, std::memory_order_relaxed);
		}
		// Batch reuse, once per executable: each batch's passes (those without a revision marked) and why a batch cannot be kept,
		// which is what converting its passes needs.
		if (!run.batchKeepable.empty() && (!replay || partial) && state.keptBatchesLogged.lock() != segment.publication->executable) {
			state.keptBatchesLogged = segment.publication->executable;
			std::vector<uint8_t> unchecked(segment.names ? segment.names->size() : 0);
			for (const auto& [prepared, why] : segment.uncheckablePrepared)
				if (prepared < unchecked.size()) unchecked[prepared] = 1;
			std::string lines;
			size_t reusableCount = 0;
			for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
				const bool reusable = batch < batchReusable.size() ? batchReusable[batch] != 0 : batch < run.batchKeepable.size() && run.batchKeepable[batch];
				const auto& why = batch < batchWhy.size() && !batchWhy[batch].empty() ? batchWhy[batch]
					: batch < run.batchUnkept.size() ? run.batchUnkept[batch] : std::string();
				std::string passes;
				for (const auto pass : graph.batches[batch].passes) {
					const auto prepared = graph.structure->passes[pass].preparedPassIndex;
					passes += (passes.empty() ? "" : ", ")
						+ (segment.names && prepared < segment.names->size() ? (*segment.names)[prepared] : std::string("?"))
						+ (prepared < unchecked.size() && unchecked[prepared] ? " (no revision)" : "");
				}
				reusableCount += reusable ? 1 : 0;
				lines += fmt::format("\n  batch {} (queue {}, {} passes){}: {}", batch, graph.batches[batch].queue, graph.batches[batch].passes.size(),
					reusable ? "" : " not kept, " + why, passes);
			}
			spdlog::info("Kept recording batches (epoch {}): {} of {} batches can be kept{}", m_passEpochFilter, reusableCount,
				graph.batches.size(), lines);
		}
		// Admission at submission (docs/job-graphs.md, "Phase 2 investigation"): under the sequencer, the ledger's handoff is
		// computed again, recorded into a list ahead of each batch that has one, submitted and committed, with no other
		// admission in between.
		sequencer = state.admission->HoldSequencer();
		// With a submission thread, executions run there, and it holds the sequencer for most of a busy frame. The owner thread's
		// retirement skips the admission's part while the sequencer is held (RetirePersistentExecutions), so it could fall
		// behind until the admission runs out of frames. So the admission's completed frames are also retired here, where the
		// sequencer is held already. It costs one completed-value read per queue.
		{
			std::vector<compiler::ExecutionTimelinePoint> completed;
			completed.reserve(slotCount);
			for (size_t slot = 0; slot < slotCount; ++slot) {
				const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot));
				const auto identity = QueueTimelineIdentity(*m_runtime, m_queueRegistry, slot);
				const uint64_t reached = m_queueRegistry.GetFence(index).GetCompletedValue();
				completed.push_back({identity, (std::min)(reached, state.admission->Submitted(identity))});
			}
			state.admission->RetireCompleted(completed);
		}
		std::vector<rhi::CommandList> handoffLists(graph.batches.size());
		std::vector<std::span<const rhi::CommandList>> leading;
		std::vector<std::shared_ptr<const void>> handoffLeases;
		{
			state.admission->Readmit(admission);
			// Textures an earlier execution left on another queue: released there first, the consumers waiting for it.
			for (const auto& release : admission.barriers.releases) {
				BT_ZONE_SCOPE("ORG.Persistent.RecordHandoffRelease");
				const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(release.queue));
				if (release.queue >= slotCount)
					throw std::logic_error("A queue handoff's release has no queue value to signal");
				auto allocation = std::make_shared<compiler::FrameCommandAllocation>();
				allocation->pool = m_queueRegistry.GetSharedPool(index);
				allocation->pair = allocation->pool->Request();
				auto list = allocation->pair.list.Get();
				rhi::BarrierBatch barriers{};
				barriers.textures = {release.textures.data(), static_cast<uint32_t>(release.textures.size())};
				list.Barriers(barriers);
				if (list.EndChecked() != rhi::Result::Ok) throw std::runtime_error("Handoff release list close failed");
				allocation->Close();
				const auto signal = state.timelines->SubmitOutside(release.queue);
				const rhi::TimelinePoint point{m_queueRegistry.GetFence(index).GetHandle(), signal.value};
				rhi::SubmitDesc submit{};
				submit.signals = {&point, 1u};
				if (m_queueRegistry.GetQueue(index).Submit({&list, 1u}, submit) != rhi::Result::Ok)
					throw std::runtime_error("Handoff release submission failed");
				m_queueRegistry.EnsureNextFenceValueAtLeast(index, signal.value + 1);
				for (const auto consumer : release.consumers) admission.incomingWaits.at(consumer).push_back(signal);
				handoffLeases.push_back(std::move(allocation));
				basic_telemetry::AddCounter("ORG.Persistent.HandoffReleases");
			}
			leading.resize(graph.batches.size());
			for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
				const auto& entry = admission.barriers.batches[batch];
				if (entry.textures.empty() && entry.buffers.empty() && entry.acquired.empty()) continue;
				BT_ZONE_SCOPE("ORG.Persistent.RecordHandoff");
				const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(graph.batches[batch].queue));
				auto allocation = std::make_shared<compiler::FrameCommandAllocation>();
				allocation->pool = m_queueRegistry.GetSharedPool(index);
				allocation->pair = allocation->pool->Request();
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
				if (list.EndChecked() != rhi::Result::Ok) throw std::runtime_error("Handoff command list close failed");
				allocation->Close();
				handoffLists[batch] = list;
				leading[batch] = {&handoffLists[batch], 1};
				handoffLeases.push_back(std::move(allocation));
			}
		}
		// Waits the readmission found on timelines the packets do not bind: another graph's, on the runtime.
		std::vector<compiler::PreparedTimelineBinding> extraTimelines;
		for (const auto& waits : admission.incomingWaits)
			for (const auto wait : waits) {
				const auto bound = [&](uint64_t identity) {
					return std::ranges::any_of(run.timelineBindings, [&](const auto& binding) { return binding.identity == identity; })
						|| std::ranges::any_of(extraTimelines, [&](const auto& binding) { return binding.identity == identity; });
				};
				if (bound(wait.timeline)) continue;
				const auto handle = m_runtime->TimelineHandleOf(wait.timeline);
				if (handle.valid()) extraTimelines.push_back({wait.timeline, handle});
			}
		auto submitWaits = RelayCrossDeviceWaits(run, admission.incomingWaits, extraTimelines);
		if (replay && !partial)
			execution = state.timelines->SubmitPrepared(replay->snapshot->layout->bundle, submitWaits, replay->packets, {}, leading,
				std::move(handoffLeases), extraTimelines);
		else
			execution = std::move(*run.recorded).Submit(*state.timelines, submitWaits, leading, std::move(handoffLeases),
				extraTimelines);
		run.Lap(timings.submitUs);
	} catch (...) {
		state.admission->Abandon(admission);
		throw;
	}
	{
		BT_ZONE_SCOPE("ORG.Persistent.CommitAdmission");
		state.admission->Commit(admission, *execution);
	}
	// Host-owned textures last used off the graphics queue, acquired back on it after their producers (external home states).
	if (const auto& home = admission.barriers.homeAcquires; !home.textures.empty()) {
		const auto index = m_queueRegistry.FindGraphicsSlot();
		const auto graphicsSlot = static_cast<uint32_t>(ToUnderlying(index));
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
		if (list.EndChecked() != rhi::Result::Ok) throw std::runtime_error("Home acquire list close failed");
		allocation->Close();
		std::vector<rhi::TimelinePoint> waits;
		for (const auto producer : home.producers) {
			if (producer >= execution->batches.size() || graph.batches[producer].queue == graphicsSlot) continue;
			const auto producerSlot = static_cast<QueueSlotIndex>(static_cast<uint8_t>(graph.batches[producer].queue));
			waits.push_back({m_queueRegistry.GetFence(producerSlot).GetHandle(), execution->batches[producer].signal.value});
		}
		const auto signal = state.timelines->SubmitOutside(graphicsSlot);
		const rhi::TimelinePoint point{m_queueRegistry.GetFence(index).GetHandle(), signal.value};
		rhi::SubmitDesc submit{};
		submit.waits = {waits.data(), static_cast<uint32_t>(waits.size())};
		submit.signals = {&point, 1u};
		if (m_queueRegistry.GetQueue(index).Submit({&list, 1u}, submit) != rhi::Result::Ok)
			throw std::runtime_error("Home acquire submission failed");
		m_queueRegistry.EnsureNextFenceValueAtLeast(index, signal.value + 1);
		state.admission->ExtendSubmitted(signal);
		state.trailingLeases.push_back({signal, std::move(allocation)});
		basic_telemetry::AddCounter("ORG.Persistent.HomeAcquires");
	}
	if (sequencer) sequencer.unlock();
	{
		std::vector<ExecutionSignal> signals;
		for (const auto& batch : execution->batches) signals.push_back({batch.signal.timeline, batch.signal.value});
		state.lastExecution->MarkAdmitted();
		state.lastExecution->MarkSubmitted(std::move(signals));
		state.pendingExecutions.push_back(state.lastExecution);
	}
	if (run.traceSchedule) RecordPersistentScheduleSubmission(*segment.publication->executable, run.execution.number);
	for (const auto& statistics : replay && !partial ? replay->statistics : run.recordingStatistics)
		if (statistics) statistics->Publish();
	for (size_t batch = 0; batch < execution->batches.size(); ++batch)
		m_queueRegistry.EnsureNextFenceValueAtLeast(static_cast<QueueSlotIndex>(static_cast<uint8_t>(graph.batches[batch].queue)),
			execution->batches[batch].signal.value + 1);
	// Present dependency: the batch containing the present pass (SetPresentPass).
	if (!m_presentPass.empty()) for (size_t batch = 0; batch < graph.batches.size(); ++batch)
		for (const auto pass : graph.batches[batch].passes) {
			const auto prepared = graph.structure->passes[pass].preparedPassIndex;
			if (!segment.names || prepared >= segment.names->size() || (*segment.names)[prepared] != m_presentPass) continue;
			const auto slot = static_cast<QueueSlotIndex>(static_cast<uint8_t>(graph.batches[batch].queue));
			m_lastPresentDependency = PresentDependency{
				.queue = m_queueRegistry.GetQueue(slot),
				.wait = {m_queueRegistry.GetFence(slot).GetHandle(), execution->batches[batch].signal.value},
				.queueSlot = slot, .batchIndex = batch, .valid = execution->batches[batch].signal.value != 0};
		}
	// The kept copy's recording from now on, and what this run did (LastKeptRun).
	if (segment.keptCopy != UINT32_MAX && segment.keptCopy < state.syncKept.size()) {
		auto& kept = state.syncKept[segment.keptCopy];
		const auto batchCount = static_cast<uint32_t>(graph.batches.size());
		// Batch reuse: how many batches were submitted again and recorded, and why each recorded batch recorded.
		if (replay && !partial) {
			m_lastKeptRun.batchesReused = batchCount;
		} else {
			std::map<std::string, uint32_t> reasons;
			for (uint32_t batch = 0; batch < batchCount; ++batch) {
				if (partial && batch < segment.reuseBatches.size() && segment.reuseBatches[batch]) {
					++m_lastKeptRun.batchesReused;
					continue;
				}
				++m_lastKeptRun.batchesRecorded;
				const auto& why = batch < segment.staleBatches.size() && !segment.staleBatches[batch].empty() ? segment.staleBatches[batch]
					: segment.staleReason;
				if (!why.empty()) ++reasons[why];
			}
			m_lastKeptRun.batchReasons.assign(reasons.begin(), reasons.end());
		}
		// The preparation's audit answers (SetKeptRecordingAudit) stay in LastKeptRun.
		if (replay && !partial) {
			m_lastKeptRun.outcome = KeptRunOutcome::Reused;
			m_lastKeptRun.reason.clear();
		} else if (keep && keep->replayable) {
			// The revisions of the passes in its batches, with their names, skip answers, audit frame data and batches: a partial
			// run's prepared ones, and the reused batches' from the kept recording.
			const auto& batchOf = BatchOfPrepared(state, segment.publication->executable);
			auto revisions = segment.revisions;
			auto names = segment.revisionNames;
			auto skips = segment.skips;
			auto audits = segment.audits;
			std::vector<uint32_t> revisionBatches;
			for (const auto prepared : segment.revisionPrepared) revisionBatches.push_back(prepared < batchOf.size() ? batchOf[prepared] : UINT32_MAX);
			if (!audits.empty()) audits.resize(revisions.size());
			if (partial && kept.recording) {
				const auto& old = kept.recording->revisions;
				for (size_t i = 0; i < old.size() && i < kept.revisionBatches.size(); ++i) {
					const auto batch = kept.revisionBatches[i];
					if (batch >= segment.reuseBatches.size() || !segment.reuseBatches[batch]) continue;
					revisions.push_back(old[i]);
					names.push_back(i < kept.revisionNames.size() ? kept.revisionNames[i] : std::string("?"));
					skips.push_back(i < kept.skips.size() ? kept.skips[i] : std::pair<const RenderGraphPass*, uint8_t>{old[i].first, 0});
					revisionBatches.push_back(batch);
					if (m_keptAuditEvery || !audits.empty()) {
						audits.resize(revisions.size() - 1);
						audits.push_back(i < kept.audits.size() ? kept.audits[i] : nullptr);
					}
				}
			}
			keep->revisions = std::move(revisions);
			kept.recording = std::move(keep);
			kept.lease = run.execution.options.run ? run.execution.options.run->lease : UINT32_MAX;
			kept.revisionNames = std::move(names);
			kept.skips = std::move(skips);
			kept.audits = std::move(audits);
			kept.revisionBatches = std::move(revisionBatches);
			kept.batchReusable = std::move(batchReusable);
			kept.batchWhy = std::move(batchWhy);
			if (!partial) kept.reusesSinceAudit = 0;
			kept.foreignTimelines.clear();
			for (const auto& binding : segment.foreignTimelines) kept.foreignTimelines.push_back(binding.identity);
			std::ranges::sort(kept.foreignTimelines);
			kept.placementSerial = state.placementSerial;
			kept.unkeptReason.clear();
			kept.staged = false;
			if (partial) {
				m_lastKeptRun.outcome = KeptRunOutcome::Partial;
				const auto first = std::ranges::find_if(segment.staleBatches, [](const auto& why) { return !why.empty(); });
				m_lastKeptRun.reason = first != segment.staleBatches.end() ? *first : std::string("?");
			} else {
				m_lastKeptRun.outcome = KeptRunOutcome::Kept;
				m_lastKeptRun.reason = segment.staleReason;
			}
		} else {
			kept = {};
			const auto firstWhy = std::ranges::find_if(batchWhy, [](const auto& why) { return !why.empty(); });
			kept.unkeptReason = keep && firstWhy != batchWhy.end() ? *firstWhy
				: keep ? std::string("a packet is not replayable (a pass's lifecycle effect or callback)")
				: run.unkeptReason.empty() ? std::string("not replayable") : run.unkeptReason;
			m_lastKeptRun.outcome = KeptRunOutcome::Unkeepable;
			m_lastKeptRun.reason = kept.unkeptReason;
		}
	}
	run.Lap(timings.commitUs);
}

// ---------------------------------------------------------------------------
// Staged recordings (GraphHost::RecordStagedCopies)

bool RenderGraph::RecordStagedCopy(rhi::Device device, const UpdateExecutionContext& frame, uint32_t copy, RunRef runRef, std::string& why) {
	BT_ZONE_SCOPE("ORG.Persistent.RecordStagedCopy");
	auto& compiler = *m_compilerState;
	if (compiler.frameProductionStopped) {
		why = "frame production has stopped";
		return false;
	}
	if (!m_taskService || !compiler.persistent) {
		why = "the graph was not built ahead (PrepareFirstPersistentBuild)";
		return false;
	}
	auto& state = *compiler.persistent;
	// A graph not placed at its build (GraphHost::PlaceEpochGraphs) places now, by the same rule: only when that moves nothing the
	// running graphs bind. Otherwise the attach places it, and its first run records.
	if (state.aliasPlacementDeferred) {
		std::string placement;
		if (!PlaceDeferredAliasesAtBuild(&placement)) {
			why = "its transients are placed at its attach (" + placement + ")";
			return false;
		}
	}
	if (state.pending) {
		why = "a structural build is pending";
		return false;
	}
	// Every pass of the epoch names its revision. A recording with one that cannot is never current, and that pass's Prepare may
	// do what only the host's thread may (an upload, a manager's state).
	for (const auto& pass : state.mainPasses) {
		if (!RunsInPersistentEpoch(pass.epoch)) continue;
		const auto& any = m_masterPassList[pass.masterIndex];
		const RenderGraphPass* object = std::visit([](auto& value) -> const RenderGraphPass* {
			using T = std::decay_t<decltype(value)>;
			if constexpr (std::is_same_v<T, std::monostate>) return nullptr;
			else return value.pass.get();
		}, any.pass);
		uint64_t hash = 0;
		if (!object || !object->RevisionHash(hash)) {
			why = "uncheckable: " + any.name + (!object ? " (no pass object)" : " (no revision hash)");
			return false;
		}
	}
	// The frame the copy's first run prepares stands in for it: its host data, its capture context and the passes' values.
	m_resolverCaptureContext = frame.resolverCaptureContext;
	m_asyncUpdateHostData = frame.ownedHostData;
	struct ResetFrame {
		RenderGraph& graph;
		~ResetFrame() {
			graph.m_asyncUpdateHostData.reset();
			graph.m_resolverCaptureContext.reset();
		}
	} resetFrame{*this};
	UseRuntimeServices();
	org::runtime::ScopedActiveGraphServices activeServices(m_uploadService.get(), m_descriptorService.get());
	PersistentExecutionOptions options{};
	options.run = runRef;
	options.keptCopy = copy;
	options.staged = true;
	const auto frameIndex = static_cast<uint32_t>(frame.preparationSlot);
	try {
		// A pass whose values cannot be taken from the stand-in frame (one that checks what an earlier pass's Update sizes,
		// which does not run here) refuses the staging, named; its build is unaffected and the copy's first run records.
		for (auto& any : m_masterPassList)
			std::visit([&](auto& obj) {
				using T = std::decay_t<decltype(obj)>;
				if constexpr (!std::is_same_v<T, std::monostate>)
					if (RunsInPersistentEpoch(obj.epoch)) obj.pass->TakeFrameValues(frame);
			}, any.pass);
		state.frameIndex = frameIndex;
		state.frameNumber = ++compiler.asyncPreparationFrameNumber;
		BindPersistentFrame(false);
		PublishDeclaredVersions();
		if (state.pending) {
			why = "a structural build is pending";
			return false;
		}
		auto execution = PreparePersistentExecution(device, frameIndex, frame.hostData, frame.deltaTime, options);
		if (!execution.main) {
			why = "the epoch has nothing to record";
			return false;
		}
		auto& segment = *execution.main;
		const auto abandon = [&] {
			for (auto& invocation : segment.invocations)
				if (invocation) invocation.Abandon(AbandonReason::AdmissionFailed);
		};
		if (segment.uncheckable) {
			abandon();
			why = "uncheckable: " + segment.uncheckableBy;
			return false;
		}
		PersistentExecuteTimings timings{};
		const auto statisticsIndices = m_statisticsService ? m_statisticsService->PassIndices() : nullptr;
		const size_t slotCount = m_queueRegistry.SlotCount();
		state.tracyFrameBegun.assign(slotCount, 0);
		PersistentSegmentRun run{.execution = execution, .segment = segment, .timings = timings};
		run.staged = true;
		run.slotCount = slotCount;
		run.presentationSlot = m_queueRegistry.FindGraphicsSlot();
		// The heaps the copy's run checks the recording against (PreparePersistentExecution).
		run.resourceHeap = DescriptorHeapManager::GetInstance().GetSRVDescriptorHeap(BackendInstanceId::Primary).GetHandle();
		run.samplerHeap = DescriptorHeapManager::GetInstance().GetSamplerDescriptorHeap(BackendInstanceId::Primary).GetHandle();
		run.statisticsIndices = statisticsIndices ? &statisticsIndices->indices : &NoStatisticsIndices();
		run.firstBatchOnSlot.assign(slotCount, UINT32_MAX);
		run.lastBatchOnSlot.assign(slotCount, UINT32_MAX);
		const auto& batches = segment.publication->executable->graph->batches;
		for (uint32_t batch = 0; batch < batches.size(); ++batch) {
			const auto slot = batches[batch].queue;
			if (slot >= slotCount) continue;
			if (run.firstBatchOnSlot[slot] == UINT32_MAX) run.firstBatchOnSlot[slot] = batch;
			run.lastBatchOnSlot[slot] = batch;
		}
		if (!AdmitPersistentSegment(run)) {
			why = "the epoch has nothing to record";
			return false;
		}
		if (!run.keepable) {
			abandon();
			why = run.unkeptReason.empty() ? std::string("not keepable") : run.unkeptReason;
			return false;
		}
		RecordPersistentSegment(run);
		const auto& recorded = *run.recorded;
		auto keep = std::make_shared<PersistentRecording>();
		keep->snapshot = recorded.Snapshot();
		keep->packets = recorded.Packets();
		keep->statistics = run.recordingStatistics;
		keep->replayable = std::ranges::all_of(keep->packets, [](const auto& packet) { return packet->Replayable(); });
		if (!keep->replayable) {
			why = "a packet is not replayable (a pass's lifecycle effect or callback)";
			return false;
		}
		keep->publication = segment.publication;
		keep->bindings = segment.bindings;
		keep->resourceHeap = run.resourceHeap;
		keep->samplerHeap = run.samplerHeap;
		keep->revisions = segment.revisions;
		keep->waits = segment.waits;
		keep->rebindings = segment.rebindings;
		keep->versions = segment.versions;
		keep->leases = segment.leases;
		m_recordingsRecorded.fetch_add(1, std::memory_order_relaxed);
		if (state.syncKept.size() <= copy) state.syncKept.resize(size_t{copy} + 1);
		auto& kept = state.syncKept[copy];
		kept = {};
		kept.recording = std::move(keep);
		kept.lease = runRef.lease;
		kept.revisionNames = segment.revisionNames;
		kept.skips = segment.skips;
		kept.audits = segment.audits;
		{
			const auto& batchOf = BatchOfPrepared(state, segment.publication->executable);
			for (const auto prepared : segment.revisionPrepared) kept.revisionBatches.push_back(prepared < batchOf.size() ? batchOf[prepared] : UINT32_MAX);
		}
		for (const auto& binding : segment.foreignTimelines) kept.foreignTimelines.push_back(binding.identity);
		std::ranges::sort(kept.foreignTimelines);
		kept.placementSerial = state.placementSerial;
		kept.staged = true;
		if (StagedRevisionDiagnostics()) {
			kept.stagedRevisionRows.resize(kept.recording->revisions.size());
			for (size_t i = 0; i < kept.recording->revisions.size(); ++i)
				(void)kept.recording->revisions[i].first->RevisionRows(kept.stagedRevisionRows[i]);
		}
		return true;
	} catch (const std::exception& e) {
		why = std::string("its recording failed: ") + e.what();
		return false;
	}
}

std::shared_ptr<const persistent::ExecutableGeneration> RenderGraph::SelectedExecutable() const {
	if (!m_compilerState || !m_compilerState->persistent) return nullptr;
	const auto selected = m_compilerState->persistent->program.Select();
	return selected ? selected->executable : nullptr;
}

ExecutionHandle RenderGraph::LastExecution() const {
	return m_compilerState && m_compilerState->persistent ? ExecutionHandle(m_compilerState->persistent->lastExecution) : ExecutionHandle{};
}

bool RenderGraph::HasPendingStructuralBuild() const {
	return m_compilerState && m_compilerState->persistent && m_compilerState->persistent->pending;
}

// See State::retiredPins. Owner thread.
void RenderGraph::ReleaseRetiredPins(std::span<const compiler::ExecutionTimelinePoint> reached) {
	auto& state = *m_compilerState->persistent;
	// An execution prepared but not submitted may bind what the selected publication no longer does.
	if (state.retiredPins.empty() || state.prepared) return;
	BT_ZONE_SCOPE("ORG.Persistent.ReleaseRetiredPins");
	const auto live = [&](uint64_t id) { return id && state.entriesByResourceID.contains(id); };
	std::optional<std::unordered_set<uint64_t>> bound;  // Concrete IDs the selected publication binds, built once.
	const auto boundBySelected = [&](uint64_t id) {
		if (!bound) {
			bound.emplace();
			if (const auto selected = state.program.Select()) {
				for (uint32_t index = 0; index < selected->bindings.Size(); ++index)
					if (const auto* binding = selected->bindings.TryAt(index); binding && binding->recording)
						bound->insert(binding->recording->resourceID);
			}
		}
		return bound->contains(id);
	};
	const auto completed = [&](const std::vector<compiler::ExecutionTimelinePoint>& after) {
		for (const auto& point : after) {
			const auto found = std::ranges::find(reached, point.timeline, &compiler::ExecutionTimelinePoint::timeline);
			if (found == reached.end() || found->value < point.value) return false;
		}
		return true;
	};
	uint64_t released = 0;
	std::erase_if(state.retiredPins, [&](State::RetiredPin& pin) {
		// Declared again (a new slot): pinned as before.
		if (live(pin.registryID) || live(pin.concreteID)) return true;
		if (!pin.unbound) {
			if (boundBySelected(pin.concreteID)) return false;
			pin.unbound = true;
			const auto submitted = state.timelines->Submitted();
			pin.after.assign(submitted.begin(), submitted.end());
		}
		if (!completed(pin.after)) return false;
		for (const auto id : {pin.registryID, pin.concreteID}) {
			if (!id || resourcesByID.contains(id)) continue;
			const auto found = m_transientFrameResourcesByID.find(id);
			if (found == m_transientFrameResourcesByID.end()) continue;
			if (const auto byName = m_transientFrameResourcesByName.find(found->second->GetName());
				byName != m_transientFrameResourcesByName.end() && byName->second == found->second)
				m_transientFrameResourcesByName.erase(byName);
			m_transientFrameResourcesByID.erase(found);
			++released;
		}
		return true;
	});
	if (released) basic_telemetry::AddCounter("ORG.Persistent.ReleasedRetiredPins", released);
}

void RenderGraph::RetirePersistentExecutions(CompletedTimelineValues* known) {
	auto& compiler = *m_compilerState;
	if (!compiler.persistent || !compiler.persistent->timelines) return;
	auto& state = *compiler.persistent;
	CompletedTimelineValues local;
	auto& values = known ? *known : local;
	const auto submitted = state.timelines->Submitted();
	// Per queue slot: what the GPU has completed. Nothing can be released while the timelines' submitted values are the ones
	// the last full pass saw (no execution recorded since) and every queue is below the lowest value anything held waits for.
	// Only the GPU's real progress is compared, so nothing another graph observed on a shared timeline can be missed: an
	// observation is never above it.
	std::array<uint64_t, 8> fences{};
	const bool fewSlots = submitted.size() <= fences.size();
	for (size_t slot = 0; slot < submitted.size() && fewSlots; ++slot)
		fences[slot] = values.Of(m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot))));
	auto& watch = state.retireWatch;
	if (fewSlots && watch.skippable && watch.submitted.size() == submitted.size() && watch.lowest.size() == submitted.size()) {
		bool reached = false;
		for (size_t slot = 0; slot < submitted.size() && !reached; ++slot)
			reached = watch.submitted[slot] != submitted[slot].value || fences[slot] >= watch.lowest[slot];
		if (!reached) return;
	}
	BT_ZONE_SCOPE("ORG.Persistent.Retire");
	std::vector<compiler::ExecutionTimelinePoint> completed;
	completed.reserve(submitted.size());
	for (size_t slot = 0; slot < submitted.size(); ++slot) {
		const auto fence = fewSlots ? fences[slot]
			: values.Of(m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot))));
		completed.push_back({submitted[slot].timeline, (std::min)(submitted[slot].value, fence)});
	}
	state.timelines->RetireCompleted(completed);
	const auto reached = completed;  // what the queues finished, before the admission's limit below
	// The admission's part needs the sequencer, which a submission holds through its recording. Rather than wait for it,
	// the pass leaves that part to its next call (the watch is not skippable then).
	bool admissionDeferred = false;
	if (auto sequencer = state.admission->TryHoldSequencer(); sequencer.owns_lock()) {
		for (auto& point : completed) point.value = (std::min)(point.value, state.admission->Submitted(point.timeline));
		state.admission->RetireCompleted(completed);
	} else {
		admissionDeferred = true;
	}
	std::erase_if(state.trailingLeases, [&](const auto& lease) {
		const auto found = std::ranges::find(reached, lease.first.timeline, &compiler::ExecutionTimelinePoint::timeline);
		return found != reached.end() && found->value >= lease.first.value;
	});
	// Handles of executions every batch of which has completed.
	std::erase_if(state.pendingExecutions, [&](const std::shared_ptr<detail::ExecutionRecord>& record) {
		for (const auto& signal : record->signals) {
			const auto found = std::ranges::find(reached, signal.timeline, &compiler::ExecutionTimelinePoint::timeline);
			if (found == reached.end() || found->value < signal.value) return false;
		}
		record->Finish(false);
		return true;
	});
	ReleaseRetiredPins(reached);
	auto retired = state.timelines->TakeRetiredGarbage();
	if (!retired.empty()) {
		auto garbage = std::make_shared<decltype(retired)>(std::move(retired));
		if (!compiler.frameWorkerScope) compiler.frameWorkerScope = m_taskService->CreateScope("ORG.Frame.Worker");
		if (!m_taskService->Submit(compiler.frameWorkerScope, runtime::TaskPriority::Streaming, "ORG.Frame.RetiredOwnership",
			[garbage] { BT_ZONE_SCOPE("ORG.Frame.DestroyRetiredOwnership"); garbage->clear(); }))
			garbage->clear();
	}
	// What the next pass must see before anything can be released (RetireWatch): the lowest value per queue slot of what is
	// still held, over the timelines' executions, the admission's frames, the trailing lists and the execution handles.
	watch.submitted.resize(submitted.size());
	watch.lowest.assign(submitted.size(), UINT64_MAX);
	std::vector<uint64_t> identities(submitted.size());
	for (size_t slot = 0; slot < submitted.size(); ++slot) {
		watch.submitted[slot] = submitted[slot].value;
		identities[slot] = submitted[slot].timeline;
	}
	const auto lower = [&](uint64_t timeline, uint64_t value) {
		const auto found = std::ranges::find(identities, timeline);
		if (found == identities.end()) return false;
		auto& lowest = watch.lowest[static_cast<size_t>(found - identities.begin())];
		lowest = (std::min)(lowest, value);
		return true;
	};
	bool skippable = fewSlots && !admissionDeferred && state.timelines->LowestRetainedSignals(watch.lowest)
		&& state.admission->LowestRetainedCompletions(identities, watch.lowest);
	for (const auto& lease : state.trailingLeases) skippable = skippable && lower(lease.first.timeline, lease.first.value);
	for (const auto& record : state.pendingExecutions) {
		// A handle with no signal is released by the next full pass.
		skippable = skippable && !record->signals.empty();
		for (const auto& signal : record->signals) skippable = skippable && lower(signal.timeline, signal.value);
	}
	watch.skippable = skippable;
	BT_PLOT("ORG.Persistent.RetainedFrames", static_cast<int64_t>(state.admission->RetainedFrames()));
	BT_PLOT("ORG.Persistent.InFlight", static_cast<int64_t>(state.timelines->InFlight()));
}

void RenderGraph::SetExternalPersistentEpochs(std::vector<uint32_t> epochs) {
	std::sort(epochs.begin(), epochs.end());
	epochs.erase(std::unique(epochs.begin(), epochs.end()), epochs.end());
	m_externalPersistentEpochs = std::move(epochs);
}

bool RenderGraph::InHostStream(uint32_t slot) const noexcept {
	const auto graphics = m_queueRegistry.FindGraphicsSlot();
	if (slot == static_cast<uint32_t>(ToUnderlying(graphics))) return true;
	if (slot >= m_queueRegistry.SlotCount()) return false;
	const auto queue = m_queueRegistry.GetQueue(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot))).GetQueueHandle();
	const auto host = m_queueRegistry.GetQueue(graphics).GetQueueHandle();
	return queue.valid() && queue.index == host.index && queue.generation == host.generation;
}

std::shared_ptr<CommandListPool> RenderGraph::ExternalSecondaryPool(uint32_t slot) {
	// The host's thread records and the submitting thread hands off into them.
	std::lock_guard lock(m_externalSecondaryPoolsMutex);
	if (m_externalSecondaryPools.size() <= slot) m_externalSecondaryPools.resize(slot + 1);
	auto& pool = m_externalSecondaryPools[slot];
	if (!pool) {
		auto device = m_queueRegistry.GetDevice(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot)));
		pool = std::make_shared<CommandListPool>(device, rhi::QueueKind::Graphics, true);
	}
	return pool;
}

bool RenderGraph::PersistentAliasPlacementsCurrent() const {
	if (!m_compilerState || !m_compilerState->persistent) return true;
	const auto& state = *m_compilerState->persistent;
	if (!state.aliasArena) return true;
	std::lock_guard lock(state.aliasArena->mutex);
	return !state.aliasArena->stale.contains(&state);
}

void RenderGraph::SetInlineAtExecution(bool enabled) noexcept { m_inlineAtExecution = enabled; }

bool RenderGraph::IsExternalPersistentEpoch(uint32_t epoch) const noexcept {
	return std::binary_search(m_externalPersistentEpochs.begin(), m_externalPersistentEpochs.end(), epoch);
}

} // namespace org
