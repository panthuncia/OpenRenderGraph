// External clients: GPU work recorded outside any graph and ordered through the runtime's sequencer. See ExternalClient.h.
#include "OpenRenderGraph/ExternalClient.h"
#include "RenderGraphPersistentState.h"
#include "Render/CommandListPool.h"

#include <spdlog/spdlog.h>

namespace org {

struct ExternalClient::SlotState {
	Queue queue;
	rhi::TimelinePtr timeline;
	uint64_t identity = 0;
	uint64_t next = 1;
	std::shared_ptr<CommandListPool> pool;
	// Command lists of submissions that may still be running, with the timeline value that marks each one done.
	std::deque<std::pair<uint64_t, CommandListPair>> running;
	// Objects held for submissions that may still be running, with the same timeline values.
	std::deque<std::pair<uint64_t, std::shared_ptr<const void>>> held;
	// The completed value the last retirement saw. Retirement is skipped while no slot moves past it.
	uint64_t retired = 0;
};

// Recently built publications, keyed by what they were built from: the slot, and for each access its resource, its current
// backing (RotationKeyOfResource), range and state. Consecutive uploads usually touch the same backings, so a hit skips
// capturing bindings and building. The bindings' captured states may be out of date by then, but admission only uses them
// for a backing the ledgers don't know yet.
struct ExternalClient::PublicationCache {
	static constexpr size_t kEntries = 32;
	struct Entry {
		std::vector<uint64_t> key;
		std::shared_ptr<const persistent::SelectedPublication> publication;
		uint64_t lastUse = 0;
	};
	std::vector<Entry> entries;
	uint64_t clock = 0;
};

ExternalClient::ExternalClient(Desc desc) : m_desc(std::move(desc)) {
	if (!m_desc.runtime) throw std::invalid_argument("ExternalClient requires a runtime");
	if (m_desc.queues.empty()) throw std::invalid_argument("ExternalClient requires queues");
	m_empty = std::make_unique<persistent::GraphProgram>();
	m_admission = std::make_unique<persistent::SynchronousAdmission>(m_desc.capacity,
		std::function<void(std::shared_ptr<const void>)>{},
		m_desc.runtime->Sequencer());
	m_workspace = std::make_unique<compiler::CompileWorkspace>();
	m_workspace->SetScheduleCache(m_desc.runtime->Schedules());
	m_slots.resize(m_desc.queues.size());
	m_publications = std::make_unique<PublicationCache>();
}

ExternalClient::~ExternalClient() {
	// Queued submissions name this client: they go to the queue first. Not on the submission thread itself, which would wait
	// for its own queue (the host retires a graph only once its queued work has passed, so none is left then).
	if (m_queuedPending.load(std::memory_order_acquire)) {
		const auto router = Router();
		if (router && router->OnThread()) {
			spdlog::error("ExternalClient {}: destroyed on the submission thread with {} queued submissions", m_desc.name,
				m_queuedPending.load(std::memory_order_acquire));
		} else {
			std::unique_lock done(m_queuedDoneMutex);
			m_queuedDone.wait(done, [&] { return m_queuedPending.load(std::memory_order_acquire) == 0; });
		}
	}
	std::lock_guard lock(m_mutex);
	for (auto& slot : m_slots) {
		if (!slot || !slot->timeline || slot->next == 1) continue;
		slot->timeline->HostWait(slot->next - 1);
	}
	RetireLocked();
	for (auto& slot : m_slots)
		if (slot && slot->timeline) {
			m_desc.runtime->Sequencer()->RetireTimeline(slot->identity);
			m_desc.runtime->NoteTimelineDestroyed(slot->timeline.Get());
		}
	// The queued timelines were signalled with the slots' submissions (or on their queues after a failure), waited for above.
	for (size_t slot = 0; slot < m_queuedTimelines.size(); ++slot)
		if (m_queuedTimelines[slot] && m_queuedValues[slot]) (void)m_queuedTimelines[slot]->HostWait(m_queuedValues[slot]);
}

std::shared_ptr<ExternalSubmitRouter> ExternalClient::Router() {
	std::lock_guard lock(m_routerMutex);
	return m_desc.router;
}

void ExternalClient::SetRouter(std::shared_ptr<ExternalSubmitRouter> router) {
	std::lock_guard lock(m_routerMutex);
	m_desc.router = std::move(router);
}

ExternalClient::SlotState& ExternalClient::Slot(uint32_t slot) {
	if (slot >= m_slots.size()) throw std::out_of_range("ExternalClient queue slot");
	auto& state = m_slots[slot];
	if (!state) {
		BT_ZONE_SCOPE("ORG.ExternalClient.CreateSlot");
		state = std::make_unique<SlotState>();
		state->queue = m_desc.queues[slot];
		auto device = m_desc.runtime->Device();
		const auto name = m_desc.name + ".Slot" + std::to_string(slot);
		if (device.CreateTimeline(state->timeline, 0, name.c_str()) != rhi::Result::Ok || !state->timeline)
			throw std::runtime_error("ExternalClient timeline creation failed");
		state->identity = m_desc.runtime->TimelineIdentity(state->timeline.Get());
		state->pool = std::make_shared<CommandListPool>(device, state->queue.kind);
	}
	return *state;
}

uint32_t ExternalClient::UsedSlots() {
	std::lock_guard lock(m_mutex);
	uint32_t mask = 0;
	for (size_t slot = 0; slot < m_slots.size() && slot < 32; ++slot)
		if (m_slots[slot]) mask |= 1u << slot;
	return mask;
}

void ExternalClient::Prewarm(uint32_t slots) {
	std::lock_guard lock(m_mutex);
	for (size_t slot = 0; slot < m_slots.size() && slot < 32; ++slot) {
		if (!(slots & (1u << slot))) continue;
		auto& state = Slot(static_cast<uint32_t>(slot));
		// Each submission takes a list for its work and one for its barriers, and a frame submits several (readbacks, streaming
		// uploads): enough for one frame stand ready, made in the background.
		if (state.pool) state.pool->PublishDemand(6, 0);
	}
}

void ExternalClient::ShutdownTaskWorkers() {
	std::lock_guard lock(m_mutex);
	for (auto& slot : m_slots)
		if (slot && slot->pool) slot->pool->ShutdownBackgroundReset();
}

void ExternalClient::Retire() {
	std::lock_guard lock(m_mutex);
	RetireLocked();
}

void ExternalClient::RetireLocked() {
	std::vector<compiler::ExecutionTimelinePoint> completed;
	bool advanced = false;
	for (auto& slot : m_slots) {
		if (!slot || !slot->timeline) continue;
		const auto value = (std::min)(slot->timeline->GetCompletedValue(), slot->next - 1);
		advanced = advanced || value > slot->retired;
		slot->retired = (std::max)(slot->retired, value);
		completed.push_back({slot->identity, (std::min)(value, m_admission->Submitted(slot->identity))});
		while (!slot->running.empty() && slot->running.front().first <= value) {
			slot->pool->Recycle(std::move(slot->running.front().second), 0);
			slot->running.pop_front();
		}
		while (!slot->held.empty() && slot->held.front().first <= value) slot->held.pop_front();
	}
	while (!m_inFlight.empty() && m_slots[m_inFlight.front().slot]->timeline->GetCompletedValue() >= m_inFlight.front().value)
		m_inFlight.pop_front();
	if (advanced) m_admission->RetireCompleted(completed);
}

// What Record took for SubmitRecorded.
struct ExternalClient::Recorded {
	uint32_t slot = 0;
	std::shared_ptr<const persistent::SelectedPublication> publication;
	CommandListPair body;  // closed
	std::vector<rhi::TimelinePoint> signals;
	std::vector<rhi::TimelinePoint> waits;
	std::shared_ptr<const void> keepAlive;
	// A queued submission's failure report: the flag its Signal shares, and the caller's callback.
	std::shared_ptr<std::atomic<bool>> failed;
	std::function<void()> onQueuedFailure;
};

ExternalClient::Signal ExternalClient::Submit(uint32_t slotIndex, std::span<const Access> accesses,
	const std::function<void(rhi::CommandList&)>& record, std::span<const rhi::TimelinePoint> signals,
	std::shared_ptr<const void> keepAlive, std::span<const rhi::TimelinePoint> extraWaits, bool cachePublication,
	std::function<void()> onQueuedFailure) {
	const auto router = Router();
	if (!router || !router->MustQueue())
		return SubmitRecorded(Record(slotIndex, accesses, record, signals, std::move(keepAlive), extraWaits, cachePublication));
	// Off the submission thread, with work queued there: record now (the caller's captures are valid only during this call),
	// and admit and submit behind that work. The admission then starts from the states the queued executions leave.
	BT_ZONE_SCOPE("ORG.ExternalClient.Queue");
	auto recorded = Record(slotIndex, accesses, record, signals, std::move(keepAlive), extraWaits, cachePublication,
		std::move(onQueuedFailure));
	recorded->failed = std::make_shared<std::atomic<bool>>(false);
	// Values are reserved and pushed under one lock: the thread submits them in push order, so each slot's queued timeline
	// is signalled in increasing order.
	std::lock_guard queueLock(m_queueMutex);
	if (m_queuedTimelines.size() < m_desc.queues.size()) {
		m_queuedTimelines.resize(m_desc.queues.size());
		m_queuedValues.resize(m_desc.queues.size(), 0);
	}
	auto& timeline = m_queuedTimelines[slotIndex];
	if (!timeline) {
		const auto name = m_desc.name + ".Queued" + std::to_string(slotIndex);
		if (m_desc.runtime->Device().CreateTimeline(timeline, 0, name.c_str()) != rhi::Result::Ok || !timeline)
			throw std::runtime_error("ExternalClient queued timeline creation failed");
	}
	const auto value = ++m_queuedValues[slotIndex];
	recorded->signals.push_back({timeline->GetHandle(), value});
	const Signal result{timeline.Get(), value, recorded->failed};
	m_queuedPending.fetch_add(1, std::memory_order_acq_rel);
	if (router->Queue([this, recorded] { SubmitQueued(recorded); })) {
		m_queuedTotal.fetch_add(1, std::memory_order_relaxed);
		basic_telemetry::AddCounter("ORG.ExternalClient.Queued");
		return result;
	}
	// The thread takes no more work (the host's end): submitted here, still in value order (the lock is held).
	m_queuedPending.fetch_sub(1, std::memory_order_acq_rel);
	(void)SubmitRecorded(recorded);
	return result;
}

void ExternalClient::SubmitQueued(const std::shared_ptr<Recorded>& recorded) {
	BT_ZONE_SCOPE("ORG.ExternalClient.SubmitQueued");
	// The caller has returned, so a failure cannot throw to it. It is reported instead:
	// - the Signal's failed flag is set, and the caller's onQueuedFailure is called,
	// - then the values are signalled anyway, without the work, so no wait on them hangs.
	const auto slot = recorded->slot;
	const auto signals = recorded->signals;
	const auto failed = recorded->failed;
	auto onFailure = std::move(recorded->onQueuedFailure);
	const auto fail = [&](const char* what) {
		spdlog::error("ExternalClient {}: a queued submission failed: {}", m_desc.name, what);
		m_queuedFailed.fetch_add(1, std::memory_order_relaxed);
		basic_telemetry::AddCounter("ORG.ExternalClient.QueuedFailed");
		if (failed) failed->store(true, std::memory_order_release);
		if (onFailure) {
			try {
				onFailure();
			} catch (const std::exception& error) {
				spdlog::error("ExternalClient {}: a queued submission's failure callback threw: {}", m_desc.name, error.what());
			} catch (...) {
				spdlog::error("ExternalClient {}: a queued submission's failure callback threw", m_desc.name);
			}
		}
		for (const auto& point : signals) (void)m_desc.queues[slot].queue.Signal(point);
	};
	try {
		(void)SubmitRecorded(recorded);
	} catch (const std::exception& error) {
		fail(error.what());
	} catch (...) {
		fail("unknown error");
	}
	{
		std::lock_guard done(m_queuedDoneMutex);
		m_queuedPending.fetch_sub(1, std::memory_order_acq_rel);
	}
	m_queuedDone.notify_all();
}

std::shared_ptr<ExternalClient::Recorded> ExternalClient::Record(uint32_t slotIndex, std::span<const Access> accesses,
	const std::function<void(rhi::CommandList&)>& record, std::span<const rhi::TimelinePoint> signals,
	std::shared_ptr<const void> keepAlive, std::span<const rhi::TimelinePoint> extraWaits, bool cachePublication,
	std::function<void()> onQueuedFailure) {
	BT_ZONE_SCOPE("ORG.ExternalClient.RecordWork");
	std::lock_guard lock(m_mutex);
	auto& slot = Slot(slotIndex);
	std::vector<compiler::CompileQueue> queues;
	for (const auto& queue : m_desc.queues)
		queues.push_back({0, true, queue.kind == rhi::QueueKind::Copy && m_desc.runtime->Backend() == rhi::Backend::D3D12});
	// Look for a publication built from the same accesses.
	std::vector<uint64_t> key{slotIndex};
	key.reserve(1 + accesses.size() * 12);
	for (const auto& access : accesses) {
		if (!access.resource) throw std::invalid_argument("ExternalClient access has no resource");
		const auto shape = ShapeOf(*access.resource);
		const auto range = LowerRange(access.range, shape);
		if (!range) continue;
		const auto state = LowerState(access.state);
		const auto backing = RotationKeyOfResource(*access.resource);
		key.insert(key.end(), {reinterpret_cast<uintptr_t>(access.resource), backing.concreteID, backing.backingGeneration,
			reinterpret_cast<uintptr_t>(backing.views), (uint64_t{backing.apiGeneration} << 32) | backing.apiIndex,
			(uint64_t{range->mip} << 32) | range->mips, (uint64_t{range->slice} << 32) | range->slices,
			state.access, state.layout, state.sync, state.write ? 1u : 0u});
	}
	basic_telemetry::AddCounter("ORG.ExternalClient.Accesses", static_cast<int64_t>(accesses.size()));
	auto& cache = *m_publications;
	std::shared_ptr<const persistent::SelectedPublication> publication;
	auto found = cachePublication ? std::ranges::find(cache.entries, key, &PublicationCache::Entry::key) : cache.entries.end();
	if (found != cache.entries.end()) {
		found->lastUse = ++cache.clock;
		publication = found->publication;
		basic_telemetry::AddCounter("ORG.ExternalClient.PublicationReuses");
	} else {
		std::vector<persistent::ExternalAccess> lowered;
		lowered.reserve(accesses.size());
		{
		BT_ZONE_SCOPE("ORG.ExternalClient.CaptureBindings");
		for (uint32_t index = 0; index < accesses.size(); ++index) {
			const auto& access = accesses[index];
			const auto shape = ShapeOf(*access.resource);
			const auto range = LowerRange(access.range, shape);
			if (!range) continue;
			auto binding = CaptureSlotBinding(*access.resource, index, shape);
			if (!binding) throw std::runtime_error("ExternalClient resource has no backing: " + access.resource->GetName());
			lowered.push_back({std::move(*binding), *range, LowerState(access.state)});
		}
		}
		publication = persistent::BuildExternalPublication(*m_empty, queues, slotIndex, lowered, *m_workspace);
		if (!publication) throw std::runtime_error("ExternalClient publication failed to build");
		if (cachePublication) {
			if (cache.entries.size() >= PublicationCache::kEntries)
				cache.entries.erase(std::ranges::min_element(cache.entries, {}, &PublicationCache::Entry::lastUse));
			cache.entries.push_back({std::move(key), publication, ++cache.clock});
		}
	}

	// Record the caller's commands first, outside the sequencer lock. They don't depend on the barriers computed below.
	CommandListPair body;
	{ BT_ZONE_SCOPE("ORG.ExternalClient.RequestList"); body = slot.pool->Request(); }
	auto bodyList = body.list.Get();
	{ BT_ZONE_SCOPE("ORG.ExternalClient.Record"); if (record) record(bodyList); }
	if (bodyList.EndChecked() != rhi::Result::Ok) throw std::runtime_error("ExternalClient command list close failed");
	body.EndRecording();
	auto recorded = std::make_shared<Recorded>();
	recorded->slot = slotIndex;
	recorded->publication = std::move(publication);
	recorded->body = std::move(body);
	recorded->signals.assign(signals.begin(), signals.end());
	recorded->waits.assign(extraWaits.begin(), extraWaits.end());
	recorded->keepAlive = std::move(keepAlive);
	recorded->onQueuedFailure = std::move(onQueuedFailure);
	return recorded;
}

ExternalClient::Signal ExternalClient::SubmitRecorded(std::shared_ptr<Recorded> recorded) {
	BT_ZONE_SCOPE("ORG.ExternalClient.Submit");
	if (!recorded || !recorded->body.list) throw std::invalid_argument("ExternalClient: SubmitRecorded needs work from Record, once");
	// Called while a pass records at execution: the submitting thread's held group goes to the queue first.
	if (!RenderGraph::FlushRecordingSubmitCarry()) throw std::runtime_error("ExternalClient: a held queue submission was refused");
	std::lock_guard lock(m_mutex);
	const uint32_t slotIndex = recorded->slot;
	auto& slot = Slot(slotIndex);
	{ BT_ZONE_SCOPE("ORG.ExternalClient.Retire"); RetireLocked(); }
	// The admission keeps each submission's publication until the GPU finishes it. When it is full, wait for the oldest.
	while (m_admission->RetainedFrames() + 1 >= m_desc.capacity && !m_inFlight.empty()) {
		BT_ZONE_SCOPE("ORG.ExternalClient.WaitForCapacity");
		const auto oldest = m_inFlight.front();
		m_slots[oldest.slot]->timeline->HostWait(oldest.value);
		RetireLocked();
	}
	const auto& publication = recorded->publication;
	auto& body = recorded->body;
	auto bodyList = body.list.Get();
	const std::span<const rhi::TimelinePoint> signals = recorded->signals;
	const std::span<const rhi::TimelinePoint> extraWaits = recorded->waits;
	auto keepAlive = std::move(recorded->keepAlive);

	std::vector<compiler::ExecutionTimelinePoint> points;
	for (uint32_t index = 0; index < m_slots.size(); ++index) {
		auto& other = Slot(index);
		points.push_back({other.identity, other.next - 1});
	}
	auto admission = [&] { BT_ZONE_SCOPE("ORG.ExternalClient.Prepare"); return m_admission->Prepare(publication, points); }();
	try {
		auto hold = m_admission->HoldSequencer();
		{
			BT_ZONE_SCOPE("ORG.ExternalClient.Readmit");
			m_admission->Readmit(admission);
		}
		auto& waits = admission.incomingWaits.at(0);
		// A texture last used on another queue must be released on that queue first.
		// We submit the release there with its own signal, and this submission waits for it.
		for (const auto& release : admission.barriers.releases) {
			auto& from = Slot(release.queue);
			auto pair = from.pool->Request();
			auto list = pair.list.Get();
			rhi::BarrierBatch barriers{};
			barriers.textures = {release.textures.data(), static_cast<uint32_t>(release.textures.size())};
			list.Barriers(barriers);
			if (list.EndChecked() != rhi::Result::Ok) throw std::runtime_error("ExternalClient release list close failed");
			pair.EndRecording();
			const auto value = from.next++;
			const rhi::TimelinePoint signal{from.timeline->GetHandle(), value};
			rhi::SubmitDesc submit{};
			submit.signals = {&signal, 1u};
			if (from.queue.queue.Submit({&list, 1u}, submit) != rhi::Result::Ok)
				throw std::runtime_error("ExternalClient release submission failed");
			m_admission->ExtendSubmitted({from.identity, value});
			from.running.emplace_back(value, std::move(pair));
			waits.push_back({from.identity, value});
			basic_telemetry::AddCounter("ORG.ExternalClient.HandoffReleases");
		}
		// Before the caller's commands, record:
		// - the entry barriers that hand each resource over from its previous user,
		// - for textures acquired from another queue family, the barriers that move them out of Common,
		// - a full memory barrier when the queue is shared with another API,
		// - the pass's own barriers.
		// After the caller's commands, record the pass's closing barriers (declared final states).
		const auto& entry = admission.barriers.batches.at(0);
		const auto* before = entry.beforePass.empty() ? nullptr : &entry.beforePass.front();
		const auto* after = entry.afterPass.empty() ? nullptr : &entry.afterPass.front();
		auto barriersOf = [](rhi::CommandList& list, std::span<const rhi::TextureBarrier> textures, std::span<const rhi::BufferBarrier> buffers) {
			if (textures.empty() && buffers.empty()) return;
			rhi::BarrierBatch barriers{};
			barriers.textures = {textures.data(), static_cast<uint32_t>(textures.size())};
			barriers.buffers = {buffers.data(), static_cast<uint32_t>(buffers.size())};
			list.Barriers(barriers);
		};
		std::vector<rhi::CommandList> lists;
		CommandListPair entryPair, exitPair;
		const bool fullBarrier = m_desc.entryFullBarrier;
		const bool entryWork = !entry.textures.empty() || !entry.buffers.empty() || !entry.acquired.empty() || fullBarrier
			|| (before && (!before->textures.empty() || !before->buffers.empty()));
		if (entryWork) {
			entryPair = slot.pool->Request();
			auto list = entryPair.list.Get();
			barriersOf(list, entry.textures, entry.buffers);
			barriersOf(list, entry.acquired, {});
			if (fullBarrier) {
				const auto full = rhi::FullMemoryBarrier();
				rhi::BarrierBatch barriers{};
				barriers.globals = {&full, 1u};
				list.Barriers(barriers);
			}
			if (before) barriersOf(list, before->textures, before->buffers);
			if (list.EndChecked() != rhi::Result::Ok) throw std::runtime_error("ExternalClient entry list close failed");
			entryPair.EndRecording();
			lists.push_back(list);
		}
		lists.push_back(bodyList);
		if (after && (!after->textures.empty() || !after->buffers.empty())) {
			exitPair = slot.pool->Request();
			auto list = exitPair.list.Get();
			barriersOf(list, after->textures, after->buffers);
			if (list.EndChecked() != rhi::Result::Ok) throw std::runtime_error("ExternalClient exit list close failed");
			exitPair.EndRecording();
			lists.push_back(list);
		}
		std::vector<rhi::TimelinePoint> waitPoints;
		// A point on another device's timeline (a graph's queue on a peer device) is relayed: the queue's device waits only
		// on its own timelines.
		const void* device = slot.timeline->impl;
		std::vector<std::pair<const void*, std::vector<rhi::TimelinePoint>>> relayed;
		for (const auto wait : waits) {
			if (wait.timeline == slot.identity) continue;  // same queue: earlier submissions already run first
			const auto handle = m_desc.runtime->TimelineHandleOf(wait.timeline);
			if (!handle.valid()) throw std::runtime_error("ExternalClient waits on an unknown timeline");
			const auto* owner = m_desc.runtime->TimelineDeviceOf(wait.timeline);
			if (!owner || owner == device) {
				waitPoints.push_back({handle, wait.value});
				continue;
			}
			auto group = std::ranges::find_if(relayed, [&](const auto& entry) { return entry.first == owner; });
			if (group == relayed.end()) group = relayed.insert(relayed.end(), {owner, {}});
			group->second.push_back({handle, wait.value});
		}
		for (const auto& [owner, points] : relayed) {
			const auto relay = m_desc.runtime->RelayWaits(owner, device, points);
			waitPoints.push_back(relay.point);
			basic_telemetry::AddCounter("ORG.ExternalClient.CrossDeviceRelays");
		}
		// The caller's own waits, already on this queue's device.
		waitPoints.insert(waitPoints.end(), extraWaits.begin(), extraWaits.end());
		const auto value = slot.next++;
		std::vector<rhi::TimelinePoint> signalPoints{{slot.timeline->GetHandle(), value}};
		signalPoints.insert(signalPoints.end(), signals.begin(), signals.end());
		rhi::SubmitDesc submit{};
		submit.waits = {waitPoints.data(), static_cast<uint32_t>(waitPoints.size())};
		submit.signals = {signalPoints.data(), static_cast<uint32_t>(signalPoints.size())};
		{
		BT_ZONE_SCOPE("ORG.ExternalClient.QueueSubmit");
		if (const auto submitted = slot.queue.queue.Submit({lists.data(), static_cast<uint32_t>(lists.size())}, submit); submitted != rhi::Result::Ok) {
			--slot.next; // Nothing will signal this value, and shutdown waits for the last one taken.
			throw std::runtime_error(std::string("ExternalClient submission failed: ") + rhi::ResultName(submitted));
		}
		}
		compiler::GraphExecutionTimeline receipt;
		receipt.batches.resize(1);
		receipt.batches[0].signal = {slot.identity, value};
		{ BT_ZONE_SCOPE("ORG.ExternalClient.Commit"); m_admission->Commit(admission, receipt); }
		if (entryPair.list) slot.running.emplace_back(value, std::move(entryPair));
		if (exitPair.list) slot.running.emplace_back(value, std::move(exitPair));
		slot.running.emplace_back(value, std::move(body));
		if (keepAlive) slot.held.emplace_back(value, std::move(keepAlive));
		m_inFlight.push_back({slotIndex, value});
		basic_telemetry::AddCounter("ORG.ExternalClient.Submissions");
		return {slot.timeline.Get(), value};
	} catch (...) {
		m_admission->Abandon(admission);
		slot.pool->Recycle(std::move(body), 0);
		throw;
	}
}

} // namespace org
