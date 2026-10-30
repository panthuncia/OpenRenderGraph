// Persistent execution, introspection: the compile dump, schedule traces and snapshots, and the memory frame-graph timeline.
#include "RenderGraphPersistentState.h"

namespace org {

namespace persistent_impl {

// Schedule introspection names: a pass by its logical slot's debug name (the main pass's name when it has none), a resource by
// its slot's registry object, a queue by its kind and instance.
ScheduleTraceNames ScheduleTraceNamesOf(const State& state, const persistent::SelectedPublication& selected, const QueueRegistry& registry) {
	ScheduleTraceNames names;
	names.pass = [&state, logical = selected.logical.get()](uint32_t prepared) -> std::string {
		if (logical && prepared < logical->passSlots.size() && !logical->passSlots[prepared].debugName.empty())
			return logical->passSlots[prepared].debugName;
		for (const auto& main : state.mainPasses)
			if (!main.retired && main.id.index == prepared) return main.name;
		return {};
	};
	names.resource = [&state](uint32_t slot) -> std::string {
		if (slot >= state.entries.size()) return {};
		const auto& entry = state.entries[slot];
		return entry.resource && !entry.retired ? entry.resource->GetName() : std::string{};
	};
	for (size_t slot = 0; slot < registry.SlotCount(); ++slot) {
		const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot));
		const auto kind = registry.GetKind(index);
		std::string name = kind == QueueKind::Graphics ? "graphics" : kind == QueueKind::Compute ? "compute" : "copy";
		if (const auto instance = registry.GetInstance(index)) name += " " + std::to_string(instance);
		names.queues.push_back(std::move(name));
	}
	return names;
}

std::string ScheduleTraceEpochLabel(uint32_t epoch) { return "epoch " + std::to_string(epoch); }

// One execution's waits beyond its batches' own (ScheduleTrace::waits), per batch: its admission's incoming waits (the frame
// values' timeline among them for a synchronous execution), and what a ticket adds at submission (SubmitPersistentTicket): the
// uploads ahead of it, on every other queue (or, planned by queue, on the batches that use what they write, with the uploads
// recorded on a batch's own queue noted on it), and the frame values. Zero values are no wait (submission drops them).
void FillScheduleSubmissionWaits(ScheduleSubmissionWaits& out, const experimental::CompiledGraph& graph,
	const std::vector<std::vector<experimental::ExecutionTimelinePoint>>& incoming, std::pair<uint32_t, uint64_t> uploadsDone, uint64_t frameWait,
	const RenderGraph::PersistentTicket::QueueUploads* queueUploads) {
	using Source = ScheduleSubmissionWaits::Point::Source;
	const bool planned = queueUploads && queueUploads->planned && queueUploads->waitsForUploads.size() == graph.batches.size();
	out.batches.resize(graph.batches.size());
	for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
		auto& traced = out.batches[batch];
		traced.points.clear();
		traced.externalEntryBarrier = traced.externalExitBarrier = false;
		traced.queueUploadCopies = 0;
		traced.queueUploadWaits.clear();
		if (batch < incoming.size())
			for (const auto& wait : incoming[batch])
				if (wait.value)
					traced.points.push_back({wait.timeline == RenderGraph::kFrameWaitTimelineIdentity ? Source::FrameValues : Source::Incoming,
						wait.timeline, wait.value});
		if (uploadsDone.second && graph.batches[batch].queue != uploadsDone.first && (!planned || queueUploads->waitsForUploads[batch]))
			traced.points.push_back({Source::Uploads, uint64_t{uploadsDone.first} + 1, uploadsDone.second});
		if (frameWait) traced.points.push_back({Source::FrameValues, RenderGraph::kFrameWaitTimelineIdentity, frameWait});
	}
	if (planned)
		for (const auto& leading : queueUploads->leading) {
			if (leading.batch >= out.batches.size() || !leading.copies) continue;
			auto& traced = out.batches[leading.batch];
			traced.queueUploadCopies += leading.copies;
			for (const auto wait : leading.waits)
				if (wait.value) traced.queueUploadWaits.push_back({Source::Incoming, wait.timeline, wait.value});
		}
}

} // namespace

// Compile-dump mode for the persistent graph. The legacy compiler rewrites its
// dump every compiled frame; the persistent executable changes only on
// structural builds, so it is written once per selected executable: batches in
// submission order with each pass's authored order and declared states, then
// every slot with its alias placement.
void RenderGraph::WritePersistentGraphDebugDump(const persistent::SelectedPublication& selected) const {
	try {
		const auto& state = *m_compilerState->persistent;
		const auto& executable = *selected.executable;
		const auto& graph = *executable.graph;
		const auto& structure = *graph.structure;
		std::unordered_map<uint32_t, const std::string*> nameBySlot;
		for (const auto& main : state.mainPasses) if (!main.retired) nameBySlot.emplace(main.id.index, &main.name);
		auto passName = [&](uint32_t pass) -> std::string {
			const auto found = nameBySlot.find(structure.passes[pass].preparedPassIndex);
			return found != nameBySlot.end() ? *found->second : "<pass " + std::to_string(pass) + ">";
		};
		auto resourceName = [&](uint32_t resource) -> std::string {
			if (resource >= executable.resourceSlots.size()) return "<resource " + std::to_string(resource) + ">";
			const auto slot = executable.resourceSlots[resource].index;
			const auto* entry = slot < state.entries.size() ? &state.entries[slot] : nullptr;
			return "slot=" + std::to_string(slot) + " name=\"" + (entry && entry->resource ? entry->resource->GetName() : std::string("-")) + "\"";
		};
		auto appendState = [](std::ostringstream& out, const CompileResourceState& value) {
			out << "access=" << rhi::helpers::ResourceAccessMaskToString(static_cast<rhi::ResourceAccessType>(value.access))
				<< " layout=" << rhi::helpers::ResourceLayoutToString(static_cast<rhi::ResourceLayout>(value.layout))
				<< " sync=" << rhi::helpers::ResourceSyncToString(static_cast<rhi::ResourceSyncState>(value.sync))
				<< (value.write ? " write" : "");
		};
		auto appendRange = [](std::ostringstream& out, const CompileRange& range) {
			out << "mips=[" << range.mip << "+" << range.mips << "] slices=[" << range.slice << "+" << range.slices << "]";
		};
		std::ostringstream dump;
		dump << "RenderGraph Persistent State\n"
			<< "structural_revision=" << executable.structuralRevision << " structural_builds=" << state.structuralBuilds
			<< " alias_plans=" << state.aliasPlans << "\n"
			<< "passes=" << structure.passes.size() << " batches=" << graph.batches.size() << " resources=" << structure.resourceIDs.size()
			<< " explicit_edges=" << structure.explicitEdges.size() << " placement_edges=" << structure.placementEdges.size()
			<< " pre_passes=" << state.preMasterIndices.size() << " tail_passes=" << state.tailMasterIndices.size()
			<< " hosted_frame_interrupting=" << state.hostedMainPasses.size() << "\n\n[Batches]\n";
		for (uint32_t batchIndex = 0; batchIndex < graph.batches.size(); ++batchIndex) {
			const auto& batch = graph.batches[batchIndex];
			dump << "batch " << batchIndex << " queue=" << batch.queue << "\n";
			for (const auto pass : batch.passes) {
				const auto& declaration = structure.passes[pass];
				dump << "  [" << pass << "] \"" << passName(pass) << "\" order=" << declaration.originalOrder
					<< " backend=" << declaration.backend << "\n";
				for (const auto& use : declaration.entryStates) {
					dump << "    entry " << resourceName(use.resource) << " "; appendRange(dump, use.range); dump << " "; appendState(dump, use.state); dump << "\n";
				}
				for (const auto& use : declaration.exitStates) {
					dump << "    exit " << resourceName(use.resource) << " "; appendRange(dump, use.range); dump << " "; appendState(dump, use.state); dump << "\n";
				}
				for (const auto& access : declaration.accesses)
					dump << "    access " << resourceName(access.resourceIndex) << (access.write ? " write" : " read") << "\n";
			}
		}
		dump << "\n[Resources]\n";
		for (uint32_t resource = 0; resource < executable.resourceSlots.size(); ++resource) {
			dump << "  [" << resource << "] " << resourceName(resource);
			const auto slot = executable.resourceSlots[resource];
			if (const auto* version = selected.bindings.TryAt(slot.index)) {
				dump << (version->bound ? " bound" : " reserved");
				if (const auto& admission = version->admission; admission && admission->aliasSize)
					dump << " alias_pool=0x" << std::hex << admission->aliasPoolID << std::dec
						<< " alias_offset=" << admission->aliasOffset << " alias_bytes=" << admission->aliasSize;
			}
			dump << "\n";
		}
		// The schedules explained: every cross-queue wait of the executable and of each epoch's, with its causes.
		const auto traceNames = ScheduleTraceNamesOf(state, selected, m_queueRegistry);
		dump << "\n" << ScheduleTraceToText(*BuildScheduleTrace(executable, traceNames, "frame", UINT32_MAX));
		for (const auto& [epoch, split] : executable.epochs)
			if (split) dump << "\n" << ScheduleTraceToText(*BuildScheduleTrace(*split, traceNames, ScheduleTraceEpochLabel(epoch), epoch));
		namespace fs = std::filesystem;
		std::error_code error;
		fs::path directory = fs::current_path(error);
		if (error) directory.clear();
		directory /= "rendergraph_dumps";
		fs::create_directories(directory, error);
		const auto path = directory / "rendergraph_persistent_state.txt";
		std::ofstream file(path, std::ios::out | std::ios::trunc);
		if (!file.is_open()) { spdlog::warn("Failed to open persistent render graph dump '{}'", path.string()); return; }
		file << dump.str();
		spdlog::info("Persistent render graph dump written to '{}' (structural revision {})", path.string(), executable.structuralRevision);
	} catch (const std::exception& e) {
		spdlog::warn("Failed to write persistent render graph dump: {}", e.what());
	}
}

void RenderGraph::SetScheduleTraceEnabled(bool enabled) noexcept {
	m_scheduleTraceEnabled.store(enabled, std::memory_order_release);
	if (!enabled) m_scheduleSnapshot.store(nullptr, std::memory_order_release);
}

bool RenderGraph::ScheduleTraceEnabled() const noexcept { return m_scheduleTraceEnabled.load(std::memory_order_acquire); }

std::shared_ptr<const PersistentScheduleSnapshot> RenderGraph::GetPersistentScheduleSnapshot() const {
	if (!m_scheduleTraceEnabled.load(std::memory_order_acquire)) return {};
	return m_scheduleSnapshot.load(std::memory_order_acquire);
}

void RenderGraph::BuildPersistentFrameGraphSnapshot(ui::FrameGraphSnapshot& out,
	const std::unordered_map<uint64_t, std::pair<uint64_t, std::string>>& memory) const {
	out.batches.clear();
	if (!m_compilerState || !m_compilerState->persistent) return;
	const auto& state = *m_compilerState->persistent;
	const auto selected = state.program.Select();
	if (!selected || !selected->executable || !selected->executable->graph) return;
	const auto& executable = *selected->executable;
	const auto& graph = *executable.graph;
	const auto& structure = *graph.structure;
	const auto names = ScheduleTraceNamesOf(state, *selected, m_queueRegistry);
	// Per compiler resource: its memory record, its alias pool (0: placed on its own) and the batches it is live across.
	struct Use { const std::pair<uint64_t, std::string>* memory = nullptr; uint64_t pool = 0; uint32_t first = UINT32_MAX, last = 0; };
	std::vector<Use> uses(executable.resourceSlots.size());
	for (size_t index = 0; index < uses.size(); ++index) {
		const auto slot = executable.resourceSlots[index].index;
		if (slot >= state.entries.size() || state.entries[slot].retired || !state.entries[slot].resource) continue;
		auto* resource = state.entries[slot].resource;
		auto* concrete = UnwrapDynamic(resource);
		auto found = concrete ? memory.find(concrete->GetGlobalResourceID()) : memory.end();
		if (found == memory.end()) found = memory.find(resource->GetGlobalResourceID());
		if (found == memory.end()) continue;
		uses[index].memory = &found->second;
		if (const auto placed = state.aliasPlaced.find(concrete); placed != state.aliasPlaced.end()) uses[index].pool = placed->second.poolID;
	}
	std::vector<std::vector<uint32_t>> touched(graph.batches.size());
	for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
		for (const auto pass : graph.batches[batch].passes)
			for (const auto& access : structure.passes[pass].accesses) {
				if (access.resourceIndex >= uses.size()) continue;
				touched[batch].push_back(access.resourceIndex);
				auto& use = uses[access.resourceIndex];
				use.first = (std::min)(use.first, batch);
				use.last = (std::max)(use.last, batch);
			}
		std::sort(touched[batch].begin(), touched[batch].end());
		touched[batch].erase(std::unique(touched[batch].begin(), touched[batch].end()), touched[batch].end());
	}
	out.batches.reserve(graph.batches.size());
	for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
		ui::FrameGraphBatchRow row;
		const auto queue = graph.batches[batch].queue;
		row.label = "Batch " + std::to_string(batch) + (queue < names.queues.size() ? " (" + names.queues[queue] + ")" : std::string{});
		for (const auto pass : graph.batches[batch].passes) row.passNames.push_back(names.pass(structure.passes[pass].preparedPassIndex));
		std::map<std::string, uint64_t> categories;
		for (const auto index : touched[batch]) {
			if (const auto* record = uses[index].memory) {
				row.footprintBytes += record->first;
				categories[record->second] += record->first;
			}
		}
		for (auto& [label, bytes] : categories) row.categories.push_back({label, bytes});
		std::sort(row.categories.begin(), row.categories.end(), [](const auto& a, const auto& b) { return a.bytes > b.bytes; });
		// Live across this batch: an aliased resource is counted through its pool's heap, once per pool.
		std::unordered_set<uint64_t> pools;
		for (const auto& use : uses) {
			if (!use.memory || use.first > batch || use.last < batch) continue;
			row.peakNaiveLiveBytes += use.memory->first;
			if (!use.pool) row.peakLiveBytes += use.memory->first;
			else if (pools.insert(use.pool).second)
				if (const auto pool = state.aliasPools.find(use.pool); pool != state.aliasPools.end()) row.peakLiveBytes += pool->second.capacityBytes;
		}
		row.aliasSavingsBytes = row.peakNaiveLiveBytes > row.peakLiveBytes ? row.peakNaiveLiveBytes - row.peakLiveBytes : 0;
		out.batches.push_back(std::move(row));
	}
}

void RenderGraph::RefreshPersistentScheduleTraces(const persistent::SelectedPublication& selected) {
	auto& state = *m_compilerState->persistent;
	auto& traces = state.scheduleTraces;
	const auto* executable = selected.executable.get();
	if (!executable || (traces.executable == executable && traces.structuralRevision == executable->structuralRevision)) return;
	BT_ZONE_SCOPE("ORG.Persistent.BuildScheduleTraces");
	const auto names = ScheduleTraceNamesOf(state, selected, m_queueRegistry);
	traces.traces.clear();
	traces.traces.emplace_back(executable, BuildScheduleTrace(*executable, names, "frame", UINT32_MAX));
	for (const auto& [epoch, split] : executable->epochs)
		if (split) traces.traces.emplace_back(split.get(), BuildScheduleTrace(*split, names, ScheduleTraceEpochLabel(epoch), epoch));
	traces.executable = executable;
	traces.structuralRevision = executable->structuralRevision;
	// Recorded submissions index the previous executables' batches.
	traces.submissions.clear();
	traces.recorded.clear();
	PublishPersistentScheduleSnapshot();
}

void RenderGraph::RecordPersistentScheduleSubmission(const persistent::ExecutableGeneration& executable, const char* segmentLabel, uint64_t frame) {
	auto& state = *m_compilerState->persistent;
	auto& traces = state.scheduleTraces;
	auto traceOf = [&]() -> const ScheduleTrace* {
		for (const auto& [key, value] : traces.traces)
			if (key == &executable) return value.get();
		return nullptr;
	};
	const std::string_view segment = segmentLabel;
	const bool dynamic = segment == "Pre" || segment == "Tail";
	const auto* trace = traceOf();
	if (!trace && !dynamic) {
		// Selected since this graph last prepared a frame (a ticket prepared from a revision's recording).
		RefreshPersistentScheduleTraces(*state.program.Select());
		trace = traceOf();
	}
	// An executable no longer selected (an older revision's recording) matches no trace: not recorded.
	if (!trace && !dynamic) return;
	const std::string_view label = trace ? std::string_view(trace->label) : segment == "Pre" ? std::string_view("pre") : std::string_view("tail");
	size_t index = 0;
	while (index < traces.submissions.size() && traces.submissions[index].label != label) ++index;
	if (index == traces.submissions.size()) {
		traces.submissions.emplace_back().label = label;
		traces.recorded.push_back(0);
	} else if (traces.recorded[index]) {
		// A label's second execution since the last publication: the round before it is complete.
		PublishPersistentScheduleSnapshot();
	}
	auto& submission = traces.submissions[index];
	submission.frame = frame;
	std::swap(submission.batches, traces.scratch.batches);
	traces.recorded[index] = 1;
}

void RenderGraph::RecordPersistentTicketScheduleSubmissions(const PersistentTicket& ticket) {
	auto& state = *m_compilerState->persistent;
	const size_t slotCount = m_queueRegistry.SlotCount();
	// The boundary barriers ExecutePersistentFrame records a ticket's segments with (a ticket has no Pre/Tail segments, so they are
	// its execution's only ones; a kept or revision recording was recorded as a ticket's too): the host's queue boundary at each
	// queue's first and last batch across them, and a closed execution's full barrier at each queue's first batch of a segment.
	using Position = std::pair<size_t, uint32_t>;
	std::vector<Position> first(slotCount, {SIZE_MAX, 0}), last(slotCount, {SIZE_MAX, 0});
	for (size_t segmentIndex = 0; segmentIndex < ticket.segments.size(); ++segmentIndex) {
		const auto& segment = ticket.segments[segmentIndex];
		if (!segment.recording) continue;
		const auto& segmentBatches = segment.admission.publication->executable->graph->batches;
		for (uint32_t batch = 0; batch < segmentBatches.size(); ++batch) {
			const auto slot = segmentBatches[batch].queue;
			if (slot >= slotCount) continue;
			if (first[slot].first == SIZE_MAX) first[slot] = {segmentIndex, batch};
			last[slot] = {segmentIndex, batch};
		}
	}
	std::vector<uint8_t> closedEntryDone(slotCount);
	for (size_t segmentIndex = 0; segmentIndex < ticket.segments.size(); ++segmentIndex) {
		const auto& segment = ticket.segments[segmentIndex];
		if (!segment.recording) continue;
		const auto& executable = *segment.admission.publication->executable;
		const auto& graph = *executable.graph;
		auto& waits = state.scheduleTraces.scratch;
		FillScheduleSubmissionWaits(waits, graph, segment.admission.incomingWaits, ticket.submittedUploads, ticket.submittedFrameWait,
			&ticket.queueUploads);
		std::fill(closedEntryDone.begin(), closedEntryDone.end(), uint8_t{0});
		for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
			const auto slot = graph.batches[batch].queue;
			if (slot >= slotCount) continue;
			const Position position{segmentIndex, batch};
			bool entry = m_externalQueueBoundary.entry && first[slot] == position;
			if (segment.admission.closed && !closedEntryDone[slot]) {
				entry = true;
				closedEntryDone[slot] = 1;
			}
			waits.batches[batch].externalEntryBarrier = entry;
			waits.batches[batch].externalExitBarrier = m_externalQueueBoundary.exit && last[slot] == position;
		}
		RecordPersistentScheduleSubmission(executable, "Main", ticket.hostFrame);
	}
}

void RenderGraph::PublishPersistentScheduleSnapshot() {
	auto& traces = m_compilerState->persistent->scheduleTraces;
	auto snapshot = std::make_shared<PersistentScheduleSnapshot>();
	snapshot->executables.reserve(traces.traces.size());
	for (const auto& entry : traces.traces) snapshot->executables.push_back(entry.second);
	snapshot->lastSubmission = traces.submissions;
	std::fill(traces.recorded.begin(), traces.recorded.end(), uint8_t{0});
	m_scheduleSnapshot.store(std::move(snapshot), std::memory_order_release);
}

} // namespace org
