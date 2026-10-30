// Persistent execution, the synchronous route: admission, recording and submission of a frame's segments, and retirement.
#include "RenderGraphPersistentState.h"

namespace org {

// ---------------------------------------------------------------------------
// Execution

void RenderGraph::ExecutePersistentFrame(PassExecutionContext& context) {
	BT_ZONE_SCOPE("ORG.Persistent.Execute");
	auto& compiler = *m_compilerState;
	auto& state = *compiler.persistent;
	if (!state.framePrepared) throw std::logic_error("Persistent execution has no prepared frame");
	auto& timings = m_lastPersistentExecuteTimings;
	timings = {};
	const bool traceSchedule = m_scheduleTraceEnabled.load(std::memory_order_relaxed);
	auto lap = [last = std::chrono::steady_clock::now()](double& a_into) mutable {
		const auto now = std::chrono::steady_clock::now();
		a_into += std::chrono::duration<double, std::micro>(now - last).count();
		last = now;
	};
	m_lastPresentDependency.reset();
	context.immediateDispatch = &m_immediateDispatch;
	const size_t slotCount = m_queueRegistry.SlotCount();
	if (!state.timelines) {
		std::vector<compiler::ExecutionTimelinePoint> queues;
		for (size_t slot = 0; slot < slotCount; ++slot) {
			const auto next = m_queueRegistry.GetCurrentFenceValue(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot)));
			queues.push_back({slot + 1, next ? next - 1 : 0});
		}
		const size_t framesInFlight = m_renderGraphSettingsService ? m_renderGraphSettingsService->GetNumFramesInFlight() : 3;
		state.timelines = std::make_unique<compiler::ExecutionTimelineAdmission>(std::move(queues), framesInFlight * 3 + 6);
	}
	// Retirement runs before this frame submits: the presentation tail of the
	// previous frame has been confirmed by now, so completed executions and
	// their publications can be released.
	RetirePersistentExecutions();
	lap(timings.retireUs);
	std::unordered_map<std::string_view, unsigned> statisticsIndices;
	if (m_statisticsService) {
		const auto& names = m_statisticsService->GetPassNames();
		for (unsigned i = 0; i < names.size(); ++i) statisticsIndices.emplace(names[i], i);
	}
	const auto defaultResourceHeap = context.GetResourceDescriptorHeap().GetHandle();
	const auto defaultSamplerHeap = context.GetSamplerDescriptorHeap().GetHandle();
	state.tracyFrameBegun.assign(slotCount, 0);
	const auto presentationSlot = m_queueRegistry.FindGraphicsSlot();

	std::vector<PersistentExecutionState::PreparedSegment*> segments;
	if (state.pre) segments.push_back(&*state.pre);
	if (state.main) segments.push_back(&*state.main);
	if (state.tail) segments.push_back(&*state.tail);
	// ExternalQueueBoundary: each queue's first and last batch across the frame's segments.
	struct BoundaryBatch { size_t segment = SIZE_MAX; uint32_t batch = 0; };
	std::vector<BoundaryBatch> firstBatchOnSlot(slotCount), lastBatchOnSlot(slotCount);
	if (m_externalQueueBoundary.entry || m_externalQueueBoundary.exit) {
		for (size_t segmentIndex = 0; segmentIndex < segments.size(); ++segmentIndex) {
			const auto& batches = segments[segmentIndex]->publication->executable->graph->batches;
			for (uint32_t batch = 0; batch < batches.size(); ++batch) {
				const auto slot = batches[batch].queue;
				if (slot >= slotCount) continue;
				if (firstBatchOnSlot[slot].segment == SIZE_MAX) firstBatchOnSlot[slot] = {segmentIndex, batch};
				lastBatchOnSlot[slot] = {segmentIndex, batch};
			}
		}
	}

	auto submitSegment = [&](PersistentExecutionState::PreparedSegment& segment, size_t segmentIndex) -> std::shared_ptr<const compiler::GraphExecutionTimeline> {
		BT_ZONE_SCOPE("ORG.Persistent.SubmitSegment");
		BT_ZONE_TEXT(segment.label, std::strlen(segment.label));
		std::vector<compiler::ExecutionTimelinePoint> queues;
		std::vector<compiler::PreparedTimelineBinding> timelineBindings;
		for (size_t slot = 0; slot < slotCount; ++slot) {
			const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot));
			const auto next = m_queueRegistry.GetCurrentFenceValue(index);
			auto submitted = next ? next - 1 : 0;
			// A pending presentation tail owns the next graphics value.
			if (state.pendingPresentSubmission && slot == static_cast<size_t>(ToUnderlying(presentationSlot))) submitted = next;
			queues.push_back({slot + 1, submitted});
			timelineBindings.push_back({slot + 1, m_queueRegistry.GetFence(index).GetHandle()});
		}
		timelineBindings.insert(timelineBindings.end(), segment.foreignTimelines.begin(), segment.foreignTimelines.end());
		if (HasFrameWaitTimeline())
			timelineBindings.push_back({kFrameWaitTimelineIdentity, (*m_frameWaitTimeline)->GetHandle()});
		auto admission = [&] {
			BT_ZONE_SCOPE("ORG.Persistent.PrepareAdmission");
			return state.admission->Prepare(segment.publication, queues, segment.waits, {}, segment.rebindings);
		}();
		// A segment with no batches (an epoch none of whose passes the graph has, as in the empty bootstrap graph) records and submits
		// nothing: its admission is abandoned, and a ticket gets no segment for it (an external one an empty execution).
		if (segment.publication->executable->graph->batches.empty()) {
			for (const auto& invocation : segment.invocations)
				if (invocation) invocation.Abandon(AbandonReason::AdmissionFailed);
			state.admission->Abandon(admission);
			return nullptr;
		}
		// Executed now (not a ticket, which takes its wait when submitted): every batch waits for the frame's values.
		if (!m_persistentTicketTarget && HasFrameWaitTimeline() && m_frameWaitValue)
			for (auto& waits : admission.incomingWaits)
				waits.push_back({kFrameWaitTimelineIdentity, m_frameWaitValue});
		// Recording reuse (SetPersistentRecordingReuse): a ticket's closed execution with nothing late-bound, whose passes all
		// report their revisions, is recorded for resubmission and kept per (epoch, slot); one prepared for what a kept
		// recording was recorded for takes it.
		auto* const ticketTarget = m_persistentTicketTarget;
		if (ticketTarget && m_persistentTicketAdmissionOnly) {
			// Recorded ahead (BindPersistentTicketRecording): this preparation's invocations are never recorded.
			for (const auto& invocation : segment.invocations)
				if (invocation) invocation.Abandon(AbandonReason::AdmissionFailed);
			ticketTarget->segments.push_back({std::move(admission), nullptr, segment.legacyBindings, defaultResourceHeap, defaultSamplerHeap});
			return nullptr;
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
		const bool keepable = ticketTarget && m_persistentRecordingReuse && admission.closed && admission.rebound.empty()
			&& admission.incomingEffects.empty() && !ticketTarget->uncheckable && segments.size() == 1
			&& ticketTarget->epoch < kKeptRecordingEpochs && state.frameIndex < kKeptRecordingSlots && !(ticketTarget->external && lateUnreplayable);
		if (keepable && !m_keptRecordings) m_keptRecordings = std::make_unique<std::remove_reference_t<decltype(*m_keptRecordings)>>();
		if (keepable && !m_persistentTicketHostData) {
			if (!m_keptRecordings) m_keptRecordings = std::make_unique<std::remove_reference_t<decltype(*m_keptRecordings)>>();
			const auto set = (*m_keptRecordings)[ticketTarget->epoch][state.frameIndex].load(std::memory_order_acquire);
			for (const auto& kept : set ? *set : KeptRecordingSet{}) {
				if (!kept->Recorded(m_persistentTicketHostTag, segment.publication.get(), segment.legacyBindings.get(), defaultResourceHeap, defaultSamplerHeap,
						ticketTarget->revisions))
					continue;
				BT_ZONE_SCOPE("ORG.Persistent.ReuseRecording");
				// This preparation's invocations are the kept ones' equals (the same revisions): never recorded.
				for (const auto& invocation : segment.invocations)
					if (invocation) invocation.Abandon(AbandonReason::AdmissionFailed);
				kept->lastUse.store(ticketTarget->hostFrame, std::memory_order_relaxed);
				ticketTarget->segments.push_back({std::move(admission), kept, segment.legacyBindings, defaultResourceHeap, defaultSamplerHeap});
				m_recordingsReused.fetch_add(1, std::memory_order_relaxed);
				lap(timings.recordUs);
				return nullptr;
			}
		}
		std::shared_ptr<const compiler::RenderFrameSnapshot> sealed;
		try {
			sealed = compiler::SealPersistentFrame(state.frameNumber, admission, std::move(segment.invocations));
		} catch (...) { state.admission->Abandon(admission); throw; }
		const auto& graph = *segment.publication->executable->graph;
		timings.cachedAdmissions += admission.cachedPlan ? 1u : 0u;
		// A closed execution starts with a full memory barrier on each of its queues.
		std::vector<uint8_t> closedEntryDone(slotCount, 0);
		std::vector<compiler::FrameRecordingJob> jobs(graph.batches.size());
		std::vector<std::shared_ptr<compiler::OwnedRecordingStatistics>> recordingStatistics;
		std::array<std::shared_ptr<const compiler::OwnedRecordingGpuRanges>, 3> gpuPassRanges;  // per rhi::QueueKind
		std::vector<size_t> demand(slotCount);
		// An external epoch's ticket (SetExternalPersistentEpochs): every batch on the graphics queue, none waiting for another,
		// recorded into secondaries; a late batch's pass (recordAtExecution) is recorded by the host's execution.
		const bool external = ticketTarget && ticketTarget->external;
		const auto graphicsSlot = static_cast<uint32_t>(ToUnderlying(m_queueRegistry.FindGraphicsSlot()));
		if (external && !graph.relativeWaits.empty())
			throw std::logic_error("An external epoch's batches wait for each other across queues");
		if (external && !m_externalSecondaryPool) {
			auto device = m_queueRegistry.GetDevice(static_cast<QueueSlotIndex>(static_cast<uint8_t>(graphicsSlot)));
			m_externalSecondaryPool = std::make_shared<CommandListPool>(device, rhi::QueueKind::Graphics, true);
		}
		for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
			const auto slot = graph.batches[batch].queue;
			if (slot >= slotCount) throw std::runtime_error("Persistent batch queue is unavailable");
			bool late = false;
			for (const auto pass : graph.batches[batch].passes) late = late || graph.structure->passes[pass].recordAtExecution;
			if (external && slot != graphicsSlot) {
				std::string names;
				for (const auto pass : graph.batches[batch].passes) {
					const auto prepared = graph.structure->passes[pass].preparedPassIndex;
					if (!names.empty()) names += ", ";
					names += segment.names && prepared < segment.names->size() ? (*segment.names)[prepared] : std::string("<unnamed>");
				}
				throw std::logic_error("An external epoch's passes must run on the graphics queue: " + names);
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
			job.pool = external ? m_externalSecondaryPool : m_queueRegistry.GetSharedPool(index);
			if (!job.pool) throw std::logic_error("Recording queue has no command-list pool");
			const auto device = m_queueRegistry.GetBackendInstance(index);
			if (m_backendDevices.size() > 1) {
				// Each device waits on and signals its own copy of every queue timeline.
				job.timelines = timelineBindings;
				for (size_t source = 0; source < slotCount; ++source)
					job.timelines[source].handle = m_queueRegistry.GetFenceForConsumer(
						static_cast<QueueSlotIndex>(static_cast<uint8_t>(source)), index).GetHandle();
			}
			job.deferred = late;
			job.recording = compiler::BuildPersistentRecordingList(sealed, batch);
			job.recording.external = external;
			job.recording.bindings = segment.legacyBindings;
			job.recording.frameSlot = state.frameIndex;
			job.recording.externalEntryBarrier = m_externalQueueBoundary.entry
				&& firstBatchOnSlot[slot].segment == segmentIndex && firstBatchOnSlot[slot].batch == batch;
			if (admission.closed && !closedEntryDone[slot]) {
				job.recording.externalEntryBarrier = true;
				closedEntryDone[slot] = 1;
			}
			job.recording.externalExitBarrier = m_externalQueueBoundary.exit
				&& lastBatchOnSlot[slot].segment == segmentIndex && lastBatchOnSlot[slot].batch == batch;
			job.recording.replayable = keepable;
			if (queueKind != QueueKind::Copy && device == BackendInstanceId::Primary) {
				job.recording.resourceDescriptorHeap = defaultResourceHeap;
				job.recording.samplerDescriptorHeap = defaultSamplerHeap;
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
				statistics->frameIndex = state.frameIndex;
				statistics->queueKind = rhiKind;
				statistics->atCompletion = late;
				// A recording kept for resubmission records no queries of the shared pool: the statistics service recreates it and
				// its readback buffers as passes register (a resubmitted list would resolve into the freed ones), and its slots are
				// frame-indexed. With recording timestamps it owns its own instead: a range per pass, per barrier batch around one
				// (before and after it) and for the list's entry and exit barriers.
				statistics->gpuQueries = queueKind != QueueKind::Copy && (!keepable || m_persistentRecordingTimestamps)
					&& m_queueRegistry.GetBackendInstance(index) == BackendInstanceId::Primary;
				for (const auto& pass : job.recording.passes) {
					const auto found = statisticsIndices.find(pass.DebugName());
					statistics->passIndices.push_back(found == statisticsIndices.end() ? -1 : static_cast<int>(found->second));
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
				recordingStatistics.push_back(std::move(statistics));
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
		const bool traceSubmission = traceSchedule && !ticketTarget;
		if (traceSubmission) {
			auto& waits = state.scheduleTraces.scratch;
			FillScheduleSubmissionWaits(waits, graph, admission.incomingWaits, {UINT32_MAX, 0}, 0);
			for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
				waits.batches[batch].externalEntryBarrier = jobs[batch].recording.externalEntryBarrier;
				waits.batches[batch].externalExitBarrier = jobs[batch].recording.externalExitBarrier;
			}
		}
		lap(timings.admissionUs);
		++timings.segments;
		timings.batches += static_cast<uint32_t>(graph.batches.size());
		{
			BT_ZONE_SCOPE("ORG.Persistent.AcquireCommandLists");
			for (size_t slot = 0; slot < slotCount; ++slot) {
				if (!demand[slot]) continue;
				const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot));
				// An external ticket's secondaries retire with its execution, on the graphics timeline as well.
				auto pool = external ? m_externalSecondaryPool : m_queueRegistry.GetSharedPool(index);
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
		compiler::PlannedFrame plan{sealed, std::move(jobs), std::move(timelineBindings), admission.incomingWaits};
		const size_t concurrency = (std::min<size_t>)(graph.batches.size(), 4u);
		std::shared_ptr<const compiler::GraphExecutionTimeline> execution;
		lap(timings.acquireUs);
		try {
			auto recorded = compiler::RecordFrame(std::move(plan), m_taskService, concurrency);
			lap(timings.recordUs);
			if (auto* ticket = m_persistentTicketTarget) {
				// Submitted later, by the host's submitting thread; completed (or abandoned) by the owner.
				auto recording = std::make_shared<PersistentRecording>();
				recording->snapshot = recorded.Snapshot();
				recording->packets = recorded.Packets();
				recording->statistics = std::move(recordingStatistics);
				recording->replayable = keepable && std::ranges::all_of(recording->packets, [](const auto& packet) { return packet->Replayable(); });
				m_recordingsRecorded.fetch_add(1, std::memory_order_relaxed);
				if (recording->replayable) {
					recording->hostTag = m_persistentTicketHostTag;
					recording->publication = segment.publication;
					recording->bindings = segment.legacyBindings;
					recording->resourceHeap = defaultResourceHeap;
					recording->samplerHeap = defaultSamplerHeap;
					recording->revisions = ticket->revisions;
					recording->lastUse.store(ticket->hostFrame, std::memory_order_relaxed);
					recording->waits = segment.waits;
					recording->leases = segment.leases;
				}
				// A recording made for host data is its caller's (BindPersistentTicketRecording), never kept here.
				if (recording->replayable && !m_persistentTicketHostData) {
					// Kept with the others recorded under the same publication, bindings and host tag (any other can never be
					// taken again), the most recently used first.
					auto& cell = (*m_keptRecordings)[ticket->epoch][state.frameIndex];
					const auto previous = cell.load(std::memory_order_acquire);
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
					cell.store(std::move(next), std::memory_order_release);
					m_recordingsKept.fetch_add(1, std::memory_order_relaxed);
				}
				ticket->segments.push_back({std::move(admission), std::move(recording), segment.legacyBindings, defaultResourceHeap, defaultSamplerHeap});
				return nullptr;
			}
			execution = std::move(recorded).Submit(*state.timelines);
			lap(timings.submitUs);
		} catch (...) {
			state.admission->Abandon(admission);
			throw;
		}
		{
			BT_ZONE_SCOPE("ORG.Persistent.CommitAdmission");
			state.admission->Commit(admission, *execution);
		}
		if (traceSubmission) RecordPersistentScheduleSubmission(*segment.publication->executable, segment.label, state.frameNumber);
		for (const auto& statistics : recordingStatistics) statistics->Publish();
		for (size_t batch = 0; batch < execution->batches.size(); ++batch)
			m_queueRegistry.EnsureNextFenceValueAtLeast(static_cast<QueueSlotIndex>(static_cast<uint8_t>(graph.batches[batch].queue)),
				execution->batches[batch].signal.value + 1);
		// Present dependency: the batch containing the presentation pass.
		for (size_t batch = 0; batch < graph.batches.size(); ++batch)
			for (const auto pass : graph.batches[batch].passes) {
				const auto prepared = graph.structure->passes[pass].preparedPassIndex;
				if (!segment.names || prepared >= segment.names->size() || (*segment.names)[prepared] != "PresentationReadyPass") continue;
				const auto slot = static_cast<QueueSlotIndex>(static_cast<uint8_t>(graph.batches[batch].queue));
				m_lastPresentDependency = PresentDependency{
					.queue = m_queueRegistry.GetQueue(slot),
					.wait = {m_queueRegistry.GetFence(slot).GetHandle(), execution->batches[batch].signal.value},
					.queueSlot = slot, .batchIndex = batch, .valid = execution->batches[batch].signal.value != 0};
				state.pendingPresentSubmission = execution->submission;
			}
		lap(timings.commitUs);
		return execution;
	};
	for (size_t segmentIndex = 0; segmentIndex < segments.size(); ++segmentIndex)
		submitSegment(*segments[segmentIndex], segmentIndex);
	state.pre.reset(); state.main.reset(); state.tail.reset();
	state.framePrepared = false;
	compiler.lastExecutedPreparationSlot = state.frameIndex;
	compiler.lastSubmittedFrameData = state.frameData;
	basic_telemetry::AddCounter("ORG.Persistent.SubmittedFrames");
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
	if (!state.pendingPresentSubmission) state.timelines->RetireCompleted(completed);
	for (auto& point : completed) point.value = (std::min)(point.value, state.admission->Submitted(point.timeline));
	state.admission->RetireCompleted(completed);
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

bool RenderGraph::IsExternalPersistentEpoch(uint32_t epoch) const noexcept {
	return std::binary_search(m_externalPersistentEpochs.begin(), m_externalPersistentEpochs.end(), epoch);
}

} // namespace org
