// Persistent execution, the synchronous route: admission, recording and submission of a frame's segment, and retirement.
#include "OpenRenderGraph/Trace.h"
#include "RenderGraphPersistentState.h"

namespace org {

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
	auto retireStart = std::chrono::steady_clock::now();
	RetirePersistentExecutions();
	timings.retireUs += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - retireStart).count();
	std::unordered_map<std::string_view, unsigned> statisticsIndices;
	if (m_statisticsService) {
		const auto& names = m_statisticsService->GetPassNames();
		for (unsigned i = 0; i < names.size(); ++i) statisticsIndices.emplace(names[i], i);
	}
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
		run.statisticsIndices = &statisticsIndices;
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
				RecordPersistentSegment(run);
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
	compiler.lastExecutedPreparationSlot = execution.frameIndex;
	if (!ticket) {
		SubmitAfterGraphCaptures(*this, state);
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
		// A ticket's body is recorded from a plan: it is admitted when it is submitted (AdmitPersistentTicket).
		if (run.execution.options.ticket)
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
	// Executed now (not a ticket, which takes its wait when submitted): every batch waits for the frame's values.
	if (!options.ticket && HasFrameWaitTimeline() && m_frameWaitValue)
		for (auto* set : {&admission.incomingWaits, &admission.ownWaits})  // ownWaits: kept by its readmission
			for (auto& waits : *set) waits.push_back({kFrameWaitTimelineIdentity, m_frameWaitValue});
	// Recording reuse (SetPersistentRecordingReuse): a ticket's closed execution with nothing late-bound, whose passes all
	// report their revisions, is recorded for resubmission and kept per (epoch, slot); one prepared for what a kept
	// recording was recorded for takes it.
	auto* const ticketTarget = options.ticket;
	if (ticketTarget && options.admissionOnly) {
		// Recorded ahead (BindPersistentTicketRecording): this preparation's invocations are never recorded.
		for (const auto& invocation : segment.invocations)
			if (invocation) invocation.Abandon(AbandonReason::AdmissionFailed);
		ticketTarget->segments.push_back({std::move(admission), nullptr, segment.bindings, run.resourceHeap, run.samplerHeap,
			segment.waits, segment.rebindings});
		return false;
	}
	// An external epoch's recording is kept as any other's (its secondaries executed again by each ticket that takes it); a late
	// batch's passes are recorded by each execution, so it is kept only when they are replayable (PreparedPass::RecordAgain: their
	// invocation recorded again by each).
	bool lateUnreplayable = false;
	const auto& structure = *segment.publication->executable->graph->structure;
	for (const auto& batch : segment.publication->executable->graph->batches)
		for (const auto pass : batch.passes) {
			if (!structure.passes[pass].recordAtExecution) continue;
			const auto prepared = structure.passes[pass].preparedPassIndex;
			lateUnreplayable = lateUnreplayable || prepared >= segment.invocations.size() || !segment.invocations[prepared].IsReplayable();
		}
	run.external = ticketTarget && ticketTarget->external;
	// A body depends only on the compiled graph (open: its entry barriers are recorded at submission) or on the executable and
	// backings (closed), so a ticket's recording can be kept, open or closed.
	run.keepable = ticketTarget && m_persistentRecordingReuse && admission.rebound.empty()
		&& admission.incomingEffects.empty() && !ticketTarget->uncheckable
		&& !(run.external && lateUnreplayable);
	if (run.keepable && !m_keptRecordings) m_keptRecordings = std::make_unique<KeptRecordingStore>();
	if (run.keepable && !options.hostData && m_keepLiveRecordings) {
		const auto set = m_keptRecordings->Load(ticketTarget->epoch, run.execution.frameIndex);
		for (const auto& kept : set ? *set : KeptRecordingSet{}) {
			if (!kept->Recorded(options.hostTag, segment.publication.get(), segment.bindings.get(), run.resourceHeap, run.samplerHeap,
					ticketTarget->revisions))
				continue;
			BT_ZONE_SCOPE("ORG.Persistent.ReuseRecording");
			// This preparation's invocations are the kept ones' equals (the same revisions): never recorded.
			for (const auto& invocation : segment.invocations)
				if (invocation) invocation.Abandon(AbandonReason::AdmissionFailed);
			kept->lastUse.store(ticketTarget->hostFrame, std::memory_order_relaxed);
			ticketTarget->segments.push_back({std::move(admission), kept, segment.bindings, run.resourceHeap, run.samplerHeap,
				segment.waits, segment.rebindings});
			m_recordingsReused.fetch_add(1, std::memory_order_relaxed);
			run.Lap(run.timings.recordUs);
			return false;
		}
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
	try {
		run.sealed = compiler::SealPersistentFrame(run.execution.number, admission, std::move(segment.invocations));
	} catch (...) { state.admission->Abandon(admission); throw; }
	const auto& graph = *segment.publication->executable->graph;
	timings.cachedAdmissions += admission.cachedPlan ? 1u : 0u;
	// A closed execution starts with a full memory barrier on each of its queues.
	std::vector<uint8_t> closedEntryDone(slotCount, 0);
	auto& jobs = run.jobs;
	jobs.resize(graph.batches.size());
	std::array<std::shared_ptr<const compiler::OwnedRecordingGpuRanges>, 3> gpuPassRanges;  // per rhi::QueueKind
	std::vector<size_t> demand(slotCount);
	// An external epoch's ticket (SetExternalPersistentEpochs): its batches on the graphics queue are recorded into secondaries,
	// which the host executes in its own stream, and a late batch's pass (recordAtExecution) by the host's execution. Its batches on
	// other queues are recorded and submitted as any other's.
	for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
		const auto slot = graph.batches[batch].queue;
		if (slot >= slotCount) throw std::runtime_error("Persistent batch queue is unavailable");
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
		if (late && !external) throw std::logic_error("A pass recorded at execution is not an external epoch's");
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
		job.recording = compiler::BuildPersistentRecordingList(run.sealed, batch);
		// An open admission's handoff into the execution (its batches' entry barriers) is recorded when it submits, against the
		// ledger as it is then: the body has none.
		if (!admission.closed) {
			job.recording.textureBarriers.clear();
			job.recording.bufferBarriers.clear();
			job.recording.acquiredBarriers.clear();
		}
		job.recording.external = inStream;
		job.recording.bindings = segment.bindings;
		job.recording.frameSlot = frameIndex;
		job.recording.externalEntryBarrier = m_externalQueueBoundary.entry && run.firstBatchOnSlot[slot] == batch;
		if (admission.closed && !closedEntryDone[slot]) {
			job.recording.externalEntryBarrier = true;
			closedEntryDone[slot] = 1;
		}
		job.recording.externalExitBarrier = m_externalQueueBoundary.exit && run.lastBatchOnSlot[slot] == batch;
		job.recording.replayable = keepable;
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
			statistics->gpuQueries = queueKind != QueueKind::Copy && (!keepable || m_persistentRecordingTimestamps)
				&& m_queueRegistry.GetBackendInstance(index) == BackendInstanceId::Primary;
			for (const auto& pass : job.recording.passes) {
				const auto found = run.statisticsIndices->find(pass.DebugName());
				statistics->passIndices.push_back(found == run.statisticsIndices->end() ? -1 : static_cast<int>(found->second));
			}
			if (keepable && statistics->gpuQueries) {
				// Each pass's range and barrier batches, plus sub-ranges the passes time inside themselves (a draw pass's views
				// or draw groups: BeginRecordingSubRange).
				constexpr uint32_t kSubRanges = 256;
				const auto ranges = static_cast<uint32_t>(job.recording.passes.size() * 3 + 3) + kSubRanges;
				statistics->queries.owned = m_statisticsService->CreateOwnedTimestamps(rhiKind, ranges);
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
				if (job.slot != slot || job.deferred) continue;
				job.recording.allocation->pair = std::move(ready[readyIndex++]);
				job.recording.allocation->pool = job.pool;
			}
			if (readyIndex != ready.size()) throw std::logic_error("Command-list batch demand did not match recording jobs");
			if (!state.tracyFrameBegun[slot] && !keepable && !external) {
				const auto first = std::ranges::find_if(jobs, [slot](const auto& job) { return job.slot == slot; });
				if (first != jobs.end()) {
					first->queue.TracyGpuFrameBegin(first->recording.allocation->pair.list.Get());
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
		if (trace::Enabled(trace::Category::Submission)) {
			std::string waited, batches;
			for (const auto& [identity, pointValue] : group.points) waited += (waited.empty() ? "" : ",") + std::to_string(identity) + "@" + std::to_string(pointValue);
			for (const auto batch : group.batches) batches += (batches.empty() ? "" : ",") + std::to_string(batch);
			ORG_TRACE(Submission, "relay-submitted").Add("waits", waited)
				.Add("signal", std::to_string(relay.identity) + "@" + std::to_string(relay.point.value)).Add("consumerBatches", batches);
		}
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
	try {
		auto& recorded = *run.recorded;
		if (auto* ticket = options.ticket) {
			// Submitted later, by the host's submitting thread; completed (or abandoned) by the owner.
			auto recording = std::make_shared<PersistentRecording>();
			recording->snapshot = recorded.Snapshot();
			recording->packets = recorded.Packets();
			recording->statistics = std::move(run.recordingStatistics);
			recording->replayable = keepable && std::ranges::all_of(recording->packets, [](const auto& packet) { return packet->Replayable(); });
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
				recording->leases = segment.leases;
			}
			// A recording made for host data is its caller's (BindPersistentTicketRecording), never kept here.
			if (recording->replayable && !options.hostData && m_keepLiveRecordings) {
				// Kept with the others recorded under the same publication, bindings and host tag (any other can never be
				// taken again), the most recently used first.
				const auto previous = m_keptRecordings->Load(ticket->epoch, run.execution.frameIndex);
				auto next = std::make_shared<KeptRecordingSet>();
				next->push_back(recording);
				for (const auto& kept : previous ? *previous : KeptRecordingSet{})
					if (kept->hostTag == recording->hostTag && kept->publication == recording->publication && kept->bindings == recording->bindings
						&& kept->resourceHeap.index == recording->resourceHeap.index && kept->samplerHeap.index == recording->samplerHeap.index)
						next->push_back(kept);
				std::stable_sort(next->begin() + 1, next->end(), [](const auto& a, const auto& b) {
					return a->lastUse.load(std::memory_order_relaxed) > b->lastUse.load(std::memory_order_relaxed);
				});
				if (next->size() > kKeptRecordingVariants) next->resize(kKeptRecordingVariants);
				m_keptRecordings->Store(ticket->epoch, run.execution.frameIndex, std::move(next));
				m_recordingsKept.fetch_add(1, std::memory_order_relaxed);
			}
			ticket->segments.push_back({std::move(admission), std::move(recording), segment.bindings, run.resourceHeap, run.samplerHeap,
				segment.waits, segment.rebindings});
			return;
		}
		// Admission at submission (docs/job-graphs.md, "Phase 2 investigation"): under the sequencer, the ledger's handoff is
		// computed again, recorded into a list ahead of each batch that has one, submitted and committed, with no other
		// admission in between.
		sequencer = state.admission->HoldSequencer();
		std::vector<rhi::CommandList> handoffLists(graph.batches.size());
		std::vector<std::span<const rhi::CommandList>> leading;
		std::vector<std::shared_ptr<const void>> handoffLeases;
		if (!admission.closed) {
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
				allocation->closed = true;
				const auto signal = state.timelines->SubmitOutside(release.queue);
				const rhi::TimelinePoint point{m_queueRegistry.GetFence(index).GetHandle(), signal.value};
				rhi::SubmitDesc submit{};
				submit.signals = {&point, 1u};
				if (m_queueRegistry.GetQueue(index).Submit({&list, 1u}, submit) != rhi::Result::Ok)
					throw std::runtime_error("Handoff release submission failed");
				m_queueRegistry.EnsureNextFenceValueAtLeast(index, signal.value + 1);
				for (const auto consumer : release.consumers) admission.incomingWaits.at(consumer).push_back(signal);
				if (trace::Enabled(trace::Category::Submission)) {
					std::string consumers;
					for (const auto consumer : release.consumers) consumers += (consumers.empty() ? "" : ",") + std::to_string(consumer);
					ORG_TRACE(Submission, "release-list-submitted").Add("queue", release.queue)
						.Add("signal", std::to_string(signal.timeline) + "@" + std::to_string(signal.value))
						.Add("textures", static_cast<uint64_t>(release.textures.size())).Add("consumerBatches", consumers);
				}
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
				allocation->closed = true;
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
		execution = std::move(recorded).Submit(*state.timelines, submitWaits, leading, std::move(handoffLeases),
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
		allocation->closed = true;
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
	for (const auto& statistics : run.recordingStatistics) statistics->Publish();
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
	run.Lap(timings.commitUs);
}

std::shared_ptr<const persistent::ExecutableGeneration> RenderGraph::SelectedExecutable() const {
	if (!m_compilerState || !m_compilerState->persistent) return nullptr;
	const auto selected = m_compilerState->persistent->program.Select();
	return selected ? selected->executable : nullptr;
}

ExecutionHandle RenderGraph::LastExecution() const {
	return m_compilerState && m_compilerState->persistent ? ExecutionHandle(m_compilerState->persistent->lastExecution) : ExecutionHandle{};
}

void RenderGraph::RetirePersistentExecutions() {
	auto& compiler = *m_compilerState;
	if (!compiler.persistent || !compiler.persistent->timelines) return;
	auto& state = *compiler.persistent;
	BT_ZONE_SCOPE("ORG.Persistent.Retire");
	std::vector<compiler::ExecutionTimelinePoint> completed;
	const auto submitted = state.timelines->Submitted();
	for (size_t slot = 0; slot < submitted.size(); ++slot) {
		const auto fence = m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot))).GetCompletedValue();
		completed.push_back({submitted[slot].timeline, (std::min)(submitted[slot].value, fence)});
	}
	state.timelines->RetireCompleted(completed);
	const auto reached = completed;  // what the queues finished, before the admission's limit below
	for (auto& point : completed) point.value = (std::min)(point.value, state.admission->Submitted(point.timeline));
	state.admission->RetireCompleted(completed);
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
	auto retired = state.timelines->TakeRetiredGarbage();
	if (!retired.empty()) {
		auto garbage = std::make_shared<decltype(retired)>(std::move(retired));
		if (!compiler.frameWorkerScope) compiler.frameWorkerScope = m_taskService->CreateScope("ORG.Frame.Worker");
		if (!m_taskService->Submit(compiler.frameWorkerScope, runtime::TaskPriority::Streaming, "ORG.Frame.RetiredOwnership",
			[garbage] { BT_ZONE_SCOPE("ORG.Frame.DestroyRetiredOwnership"); garbage->clear(); }))
			garbage->clear();
	}
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

bool RenderGraph::IsExternalPersistentEpoch(uint32_t epoch) const noexcept {
	return std::binary_search(m_externalPersistentEpochs.begin(), m_externalPersistentEpochs.end(), epoch);
}

} // namespace org
