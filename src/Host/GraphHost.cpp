#include "OpenRenderGraph/GraphHost.h"

#include <BasicTelemetry/Tracy.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Resources/Buffers/DynamicBufferBase.h"
#include "Render/MemoryIntrospectionBackend.h"
#include "Render/PassExecutionContext.h"
#include "Render/RenderGraph/PersistentGraph.h"
#include "Render/RenderGraph/PersistentAliasArena.h"
#include "Render/Runtime/IUploadService.h"
#include "Render/Runtime/OpenRenderGraphSettings.h"
#include "Render/Runtime/RuntimeDevice.h"
#include "Render/Runtime/ScopedActiveGraphServices.h"
#include "Render/Runtime/ThreadPoolTaskService.h"
#include "HostUploadExtension.h"

namespace org {

namespace {

using detail::HostUploadExtension;

using TimelinePoints = std::vector<std::pair<std::shared_ptr<rhi::TimelinePtr>, uint64_t>>;

bool PointsReached(const TimelinePoints& points) {
	for (const auto& [timeline, value] : points)
		if (timeline && *timeline && (*timeline)->GetCompletedValue() < value) return false;
	return true;
}

} // namespace

// Defined before the constructor: a throwing constructor destroys m_async, and clang instantiates
// ~unique_ptr<Async> there.
struct GraphHost::Async {
	using Ticket = RenderGraph::PersistentTicket;
	struct Message {
		// Detach: the host's thread lets go of instances (their epochs' tickets abandoned, their recording requests dropped), says
		// so (detached) and holds still until the owner thread has built them (holding). Attach: the instances, built again, are the
		// host's thread's once more, at new build generations.
		enum class Kind : uint8_t { Submitted, Discard, Detach, Attach, Stop } kind = Kind::Stop;
		std::shared_ptr<Ticket> ticket;
		RenderGraph::PersistentTicketSubmission submission;
		uint32_t epochIndex = 0;
		std::vector<std::pair<size_t, uint64_t>> instances;  // Detach, Attach: the instances (and Attach: their new build generations)
		uint64_t hold = 0;  // Detach: the hold it starts, which lasts until released reaches it
		int32_t slot = -1;  // Discard: the slot to prepare the epoch's replacement ticket into, or -1 for none
		std::shared_ptr<void> keepAlive;          // the uploads' staging, until the slot completes
		std::shared_ptr<void> retired;            // Submitted: GPU objects that retired at this submission, destroyed on the host's thread
		std::shared_ptr<const void> resourceOwner;
		std::pair<uint32_t, uint64_t> uploadSignal{UINT32_MAX, 0};
		uint64_t number = 0;  // Submitted: its number (SubmissionsMade)
	};

	std::vector<uint32_t> epochs;
	std::vector<size_t> instanceOf;       // per epoch index: its instance (GraphHost::m_instances)
	std::vector<uint8_t> revisionDriven;  // per epoch index: recorded ahead for revisions, its tickets admission-only
	// Host thread, per epoch index: let go of (Detach) until its instance is attached again; and its instance's build generation.
	std::vector<uint8_t> paused;
	std::vector<uint64_t> generation;
	// Per epoch index, its ring of frame slots: [index * ringSize, + ringSize), the next position. 0: one ring of every slot.
	uint32_t ringSize = 0;
	std::vector<uint64_t> ringNext;
	// Recording requests taken from the host's queue (RecordingQueue), by the host's thread, in posting order.
	struct RecordingRequest {
		uint32_t epochIndex = 0;
		std::shared_ptr<const IHostExecutionData> hostData;
		GraphHost::EpochRecordingCallback done;
		std::shared_ptr<GraphHost::EpochRecording> result;
		std::vector<uint8_t> recorded;  // by ring position
	};
	std::deque<std::unique_ptr<RecordingRequest>> pendingRequests;  // host thread; StopAsync drops what is left
	// Host thread, per epoch index: the revision recordings made (admitted by each ticket while their callers hold them), and the
	// ticket prepared and not yet submitted or discarded, which a recording finished meanwhile is added to before its callback.
	std::vector<std::vector<std::weak_ptr<const GraphHost::EpochRecording>>> liveRecordings;
	std::vector<std::shared_ptr<RenderGraph::PersistentTicket>> outstanding;
	std::atomic<uint64_t> revisionSlotsRecorded{0}, revisionRecordings{0};
	// One ticket cell per epoch: stored by the host's thread, taken by the render thread.
	std::unique_ptr<std::atomic<std::shared_ptr<Ticket>>[]> cells;
	// Render thread -> host thread, single producer / single consumer, in order.
	static constexpr uint64_t kMailbox = 64;
	std::array<Message, kMailbox> mailbox;
	std::atomic<uint64_t> written{0}, read{0};
	rhi::CompletionWait* completionWake = nullptr;  // the host's (m_completionWake): requesters wake the thread through it too
	// Whether the host's thread, if it sleeps, sleeps on GPU completions as well as on completionWake: then a submission's
	// message needs no wake of its own (vkSignalSemaphore, several microseconds on the render thread), because the next GPU
	// completion wakes the thread and it reads the mailbox then. Written by the host's thread before its last look at the
	// mailbox, read by a poster after its write: both sequentially consistent, so one of them sees the other.
	std::atomic<bool> wakesOnGpu{false};
	uint64_t wakesSkipped = 0;  // render thread
	// Ticket publication and terminal failure both advance this counter. A
	// notify on an unchanged null shared_ptr cannot release atomic::wait(nullptr).
	std::atomic<uint32_t> ticketWake{0};
	// Detach messages handled: the render thread waits for its own to be counted. Then the host's thread holds still, touching
	// nothing, until released reaches the Detach's hold: a graph's build changes services every graph shares (statistics,
	// uploads, descriptors). Holds are numbered, so one released and the next begun at once is still two.
	std::atomic<uint64_t> detached{0};
	std::atomic<uint64_t> released{0};
	uint64_t holds = 0;  // render thread: the last hold begun
	bool holding = false;  // render thread: inside a hold
	std::atomic<bool> failed{false};
	std::string error;  // written before failed is set
	std::thread thread;
	// Advanced by NoteNewVersions. A ticket prepared under an older value may hold a version that is not the newest: it is not
	// current (Current).
	std::atomic<uint64_t> backingVersion{0};

	// Host thread only.
	struct Slot {
		TimelinePoints points;  // its last submission's completion
		uint64_t hostFrame = 0;
		uint64_t submission = 0;  // its last submission's number (SubmissionsMade)
		bool pending = false;   // submitted work not yet waited for
		bool reserved = false;  // held by a ticket that has not been submitted
		std::shared_ptr<void> keepAlive;
		std::shared_ptr<const void> resourceOwner;
	};
	std::vector<Slot> slots;
	// The worker stops on the first completion exception. Preserve the one
	// submitted message it was processing until device teardown.
	std::shared_ptr<void> uncertainUploadOwner;
	std::shared_ptr<const void> uncertainResourceOwner;
	std::vector<uint8_t> needsTicket;
	uint64_t nextSlot = 0;
	uint64_t hostFrame = 0;
	// The highest submission number a Submitted message brought (the mailbox is in order: every one not read yet is higher), and
	// the last value the retired callback was given.
	uint64_t submissionsSeen = 0, retiredReported = 0;

	// Render thread only.
	std::vector<uint64_t> queueValues;  // SubmitTicket's: the last value assigned per queue slot of the epoch's graph
	// Per frame slot. Reset and begun by the host's thread when it prepares the slot's ticket (after the slot's
	// wait: the slot's last upload list is done), then recorded, ended and submitted by the render thread; the
	// ticket cell orders the two.
	std::vector<CommandListPair> uploadLists;
	// The same for an external epoch's ticket (SetExternalEpochs): a secondary, recycled by the host's thread with the slot's
	// list and begun by the render thread, executed first by the caller's execution (ExternalEpochExecution::leading).
	std::vector<CommandListPair> uploadSecondaries;
	// The same, per frame slot and per queue slot, for the uploads recorded on another queue than graphics (Desc::queueScopedUploads):
	// a list for each compute or copy queue of the graphics queue's device, none (empty) for the others.
	std::vector<std::vector<CommandListPair>> queueUploadLists;
	std::vector<rhi::CommandList> queueLists;     // SubmitTicket's scratch: the slot's queue upload lists, by queue slot
	std::vector<rhi::CommandList> recordedLists;  // SubmitTicket's scratch: the producers' recorded upload lists, then the slot's
	uint32_t graphicsSlot = 0;
	rhi::Queue graphicsQueue;
	rhi::TimelineHandle graphicsFence{};

	uint32_t IndexOf(uint32_t epoch) const {
		for (uint32_t i = 0; i < epochs.size(); ++i) if (epochs[i] == epoch) return i;
		throw std::invalid_argument("SubmitEpoch: epoch " + std::to_string(epoch) + " has no ticket");
	}
	// lazyWake: a submission's completion, which the host's thread can take at its next GPU wake-up (wakesOnGpu).
	bool TryPost(Message& message, bool lazyWake = false) {
		const auto index = written.load(std::memory_order_relaxed);
		if (failed.load(std::memory_order_acquire) || index - read.load(std::memory_order_acquire) >= kMailbox) return false;
		mailbox[index % kMailbox] = std::move(message);
		written.store(index + 1, std::memory_order_seq_cst);
		if (lazyWake && wakesOnGpu.load(std::memory_order_seq_cst)) {
			++wakesSkipped;
			return true;
		}
		(void)completionWake->Notify();
		return true;
	}
	bool Post(Message& message, bool lazyWake = false) {
		while (!TryPost(message, lazyWake)) {
			if (failed.load(std::memory_order_acquire)) return false;
			std::this_thread::yield();
		}
		return true;
	}
	// Host thread: a ticket's preparation, which captures every slot's backing and may change backings itself
	// (BufferBase::EnterBackingMutation).
	struct Preparation {
		Preparation() { BufferBase::EnterBackingMutation(); }
		~Preparation() { BufferBase::LeaveBackingMutation(); }
		Preparation(const Preparation&) = delete;
		Preparation& operator=(const Preparation&) = delete;
	};
	// Every pass it prepared would prepare the same invocation, and no version was published since.
	bool Current(const Ticket& ticket) const {
		return RenderGraph::PersistentTicketCurrent(ticket)
			&& RenderGraph::PersistentTicketHostTag(ticket) == backingVersion.load(std::memory_order_acquire);
	}
	void ThrowIfFailed() const {
		if (failed.load(std::memory_order_acquire)) throw std::runtime_error("Async epochs failed: " + error);
	}
	std::shared_ptr<Ticket> WaitTicket(uint32_t index) {
		BT_ZONE_SCOPE("ORG.Host.WaitTicket");
		// About to block on the host's thread: it must not be left asleep on GPU work that waits for this thread.
		(void)completionWake->Notify();
		for (;;) {
			const auto seen = ticketWake.load(std::memory_order_acquire);
			if (auto ticket = cells[index].exchange(nullptr, std::memory_order_acq_rel)) return ticket;
			ThrowIfFailed();
			ticketWake.wait(seen, std::memory_order_acquire);
		}
	}
};

// The host's recording requests (RequestEpochRecording): any thread pushes onto a Treiber stack, the host's thread takes it whole
// and reverses it into posting order. It and the wake live as long as the host, so a request never meets a build's teardown.
struct GraphHost::RecordingQueue {
	struct Request {
		uint32_t epoch = 0;
		std::shared_ptr<const IHostExecutionData> hostData;
		GraphHost::EpochRecordingCallback done;
		Request* next = nullptr;
	};
	std::atomic<Request*> head{nullptr};
	std::atomic<rhi::CompletionWait*> wake{nullptr};  // the host's thread's, once async epochs first started
	std::atomic<bool> accepting{false};
	// The host's end: what is left goes without its callbacks.
	~RecordingQueue() {
		for (auto* request = head.exchange(nullptr); request;) {
			auto* next = request->next;
			delete request;
			request = next;
		}
	}
	// Takes every request pushed so far, in posting order.
	std::vector<std::unique_ptr<Request>> Take() {
		std::vector<std::unique_ptr<Request>> taken;
		for (auto* request = head.exchange(nullptr, std::memory_order_acquire); request;) {
			auto* next = request->next;
			request->next = nullptr;
			taken.emplace_back(request);
			request = next;
		}
		std::reverse(taken.begin(), taken.end());
		return taken;
	}
};

GraphHost::GraphHost(Desc desc) : m_desc(std::move(desc)), m_recordingQueue(std::make_unique<RecordingQueue>()) {
	if (!m_desc.runtime) throw std::invalid_argument("GraphHost requires a runtime");
	m_device = m_desc.runtime->Device();
	// The frame slots: framesInFlight frames of every epoch in the order, so that adding an epoch keeps the CPU's lead over
	// the GPU the same rather than shortening it (each slot's next execution waits for its last one).
	const uint32_t epochs = (std::max)(static_cast<uint32_t>(m_desc.epochOrder.size()), 1u);
	m_frameSlots = (std::max)(m_desc.framesInFlight, 1u) * epochs;
	// Upload pages, statistics query ranges and admission capacity are sized from the runtime's slots.
	if (m_frameSlots > m_desc.runtime->Slots())
		throw std::invalid_argument("GraphHost: more frame slots (frames in flight times epochs) than the runtime has");
	m_slotCompletions.assign(m_frameSlots, {});
	// A graph per epoch, in the frame's order; one of every pass without epochs.
	if (m_desc.epochOrder.empty()) m_instances.emplace_back();
	for (const auto epoch : m_desc.epochOrder) {
		if (epoch == persistent::AllEpochs) throw std::invalid_argument("GraphHost: an epoch order names AllEpochs");
		if (std::ranges::any_of(m_instances, [&](const Instance& other) { return other.epoch == epoch; }))
			throw std::invalid_argument("GraphHost: an epoch order names an epoch twice");
		m_instances.emplace_back().epoch = epoch;
	}
}

GraphHost::~GraphHost() {
	// Nothing queued is completed from here on: its callers may be torn down before the host.
	m_ending = true;
	m_recordingQueue->accepting.store(false, std::memory_order_release);
	StopAsync();
	DestroyGraph();
	// The host's queue timelines go last: what was submitted on them completes first, and the runtime's ledgers stop waiting on them.
	for (const auto& timeline : m_timelines) {
		const auto next = timeline.nextValue->load(std::memory_order_acquire);
		if (next > 1) (void)(*timeline.fence)->HostWait(next - 1);
		m_desc.runtime->Sequencer()->RetireTimeline(m_desc.runtime->TimelineIdentity(timeline.fence->Get()));
		m_desc.runtime->NoteTimelineDestroyed(timeline.fence->Get());
	}
	m_timelines.clear();
	m_desc.runtime->Retire();
	m_uncertainExecutionOwners.clear();
	m_uncertainUploadOwners.clear();
}

size_t GraphHost::InstanceOf(uint32_t epoch) const noexcept {
	for (size_t index = 0; index < m_instances.size(); ++index)
		if (m_instances[index].epoch == epoch) return index;
	return kNoInstance;
}

RenderGraph* GraphHost::AnyGraph() const noexcept {
	for (const auto& instance : m_instances)
		if (instance.graph) return instance.graph.get();
	return nullptr;
}

RenderGraph* GraphHost::Graph(uint32_t epoch) noexcept {
	const auto index = InstanceOf(epoch);
	return index == kNoInstance ? nullptr : m_instances[index].graph.get();
}

uint64_t GraphHost::BuildGeneration(uint32_t epoch) const noexcept {
	const auto index = InstanceOf(epoch);
	return index == kNoInstance ? 0 : m_instances[index].buildGeneration;
}

bool GraphHost::EpochRecordingCurrent(const EpochRecording& recording) const noexcept {
	const auto index = InstanceOf(recording.epoch);
	return index != kNoInstance && m_instances[index].graph && recording.buildGeneration == m_instances[index].buildGeneration;
}

void GraphHost::RequestRebuild() noexcept {
	for (auto& instance : m_instances) instance.rebuildRequested = true;
}

void GraphHost::RequestRebuild(uint32_t epoch) noexcept {
	if (const auto index = InstanceOf(epoch); index != kNoInstance) m_instances[index].rebuildRequested = true;
}

bool GraphHost::RebuildRequested() const noexcept {
	return std::ranges::any_of(m_instances, [](const Instance& instance) { return instance.rebuildRequested; });
}

bool GraphHost::AnyBuildDue() const noexcept {
	for (size_t index = 0; index < m_instances.size(); ++index)
		if (BuildDue(index)) return true;
	return false;
}

void GraphHost::AddExtension(std::string id, ExtensionFactory factory) {
	RemoveExtension(id);
	m_extensions.push_back({std::move(id), std::move(factory)});
	RequestRebuild();
}

void GraphHost::RemoveExtension(const std::string& id) {
	const auto erased = std::erase_if(m_extensions, [&](const Registered& entry) { return entry.id == id; });
	if (erased) RequestRebuild();
}

runtime::IUploadService* GraphHost::Uploads() noexcept {
	auto* graph = AnyGraph();
	return graph ? graph->GetUploadService() : nullptr;
}

runtime::IDescriptorService* GraphHost::Descriptors() noexcept {
	auto* graph = AnyGraph();
	return graph ? graph->GetDescriptorService() : nullptr;
}

std::shared_ptr<runtime::IUploadService> GraphHost::RetainUploads() noexcept {
	auto* graph = AnyGraph();
	return graph ? graph->RetainUploadService() : nullptr;
}

void GraphHost::DestroyGraph() {
	StopAsync();
	if (!AnyGraph() && m_retiredGraphs.empty()) return;
	for (auto& instance : m_instances)
		if (instance.graph) instance.graph->StopFrameProduction();
	// Adopted devices wait on their own timelines only (never the whole device).
	(void)m_device.WaitIdle();
	for (auto& completion : m_slotCompletions) completion = {};
	m_slotSubmitted.clear();
	m_externalRetired.clear();
	for (auto& instance : m_instances) {
		if (!instance.graph) continue;
		instance.graph->ShutdownExtensions();
		instance.graph->ShutdownTaskWorkers();
		instance.graph.reset();
		instance.rebuildRequested = true;
	}
	ReleaseRetiredGraphs(true);
}

void GraphHost::ReleaseRetiredGraphs(bool all) {
	if (m_retiredGraphs.empty()) return;
	if (!all && std::ranges::none_of(m_retiredGraphs, [](const RetiredGraph& retired) { return retired.done.Reached(); })) return;
	const auto release = [&] {
		std::erase_if(m_retiredGraphs, [&](RetiredGraph& retired) {
			if (!all && !retired.done.Reached()) return false;
			if (all) (void)retired.done.Wait();
			BT_ZONE_SCOPE("ORG.Host.ReleaseRetiredGraph");
			retired.graph->ShutdownExtensions();
			retired.graph->ShutdownTaskWorkers();
			retired.graph.reset();
			return true;
		});
	};
	// A graph's teardown touches services the host's thread uses (its upload client's timelines, descriptors): it holds still.
	if (m_async && !m_async->holding) HoldAsync({}, release);
	else release();
}

void GraphHost::SetGpuPassRangeCallbacks(RenderGraph::GpuPassRangeBegin begin, RenderGraph::GpuPassRangeEnd end) {
	m_gpuPassRangeBegin = std::move(begin);
	m_gpuPassRangeEnd = std::move(end);
	for (auto& instance : m_instances)
		if (instance.graph) instance.graph->SetGpuPassRangeCallbacks(m_gpuPassRangeBegin, m_gpuPassRangeEnd);
}

void GraphHost::EnsureTimelines() {
	if (!m_timelines.empty()) return;
	static constexpr const char* names[] = {"ORG host graphics timeline", "ORG host compute timeline", "ORG host copy timeline"};
	for (const auto* name : names) {
		auto fence = std::make_shared<rhi::TimelinePtr>();
		if (m_device.CreateTimeline(*fence, 0, name) != rhi::Result::Ok || !*fence)
			throw std::runtime_error("GraphHost could not create its queue timelines");
		m_timelines.push_back({std::move(fence), std::make_shared<std::atomic<uint64_t>>(1)});
	}
}

void GraphHost::BuildInstance(size_t index, bool wait) {
	BT_ZONE_SCOPE("ORG.Host.Build");
	const auto begin = std::chrono::steady_clock::now();
	EnsureTimelines();
	auto& instance = m_instances[index];
	const bool replacing = static_cast<bool>(instance.graph);
	// The graph it replaces: its own submitted work is what every queue has been given so far (the timelines are shared), taken
	// while it is still listed. It is released once that is done, the other epochs running meanwhile.
	auto done = instance.graph ? SubmittedPoint() : GpuPoint{};
	if (auto old = std::move(instance.graph)) {
		old->StopFrameProduction();
		m_retiredGraphs.push_back({std::move(old), std::move(done)});
		++m_asyncStats.rebuilds;
		if (wait) ReleaseRetiredGraphs(true);
	}
	auto graph = std::make_unique<RenderGraph>(m_desc.runtime);
	graph->SetSharedQueueTimelines(m_timelines);
	graph->SetPassEpochFilter(instance.epoch);
	graph->RegisterExtension(std::make_unique<HostUploadExtension>(graph->GetUploadService()), "org.host.io");
	for (const auto& entry : m_extensions)
		if (auto extension = entry.factory()) graph->RegisterExtension(std::move(extension), entry.id);
	runtime::ScopedActiveGraphServices services(graph->GetUploadService(), graph->GetDescriptorService());
	graph->PrepareExtensionsForBuild();
	// The graph's memory snapshots (its VRAM dump, a host's memory UI): ORG's tracking world, as the contributor host's.
	graph->GetMemorySnapshotProvider().SetProvider(memory::CreateECSMemorySnapshotProvider());
	graph->CompileStructural();
	if (instance.epoch != persistent::AllEpochs) graph->SetPersistentEpochOrder({instance.epoch});
	graph->SetPersistentClosedExecutions(m_desc.closedExecutions);
	graph->SetPersistentExternalHomeStates(m_desc.externalHomeStates);
	if (m_desc.sharedAliasArena) {
		if (!m_aliasArena) m_aliasArena = PersistentAliasArena::Create();
		graph->SetPersistentAliasArena(m_aliasArena);
	}
	graph->SetExternalPersistentEpochs(m_externalEpochs);
	graph->SetPersistentRecordingReuse(m_desc.reuseRecordings);
	graph->SetPersistentRecordingTimestamps(m_recordingTimestamps.load(std::memory_order_acquire));
	graph->SetScheduleTraceEnabled(m_scheduleTrace.load(std::memory_order_acquire));
	{
		auto boundary = m_desc.queueBoundary;
		if (m_boundaryOverridden.load(std::memory_order_acquire)) {
			boundary.entry = m_boundaryEntry.load(std::memory_order_acquire);
			boundary.exit = m_boundaryExit.load(std::memory_order_acquire);
		}
		graph->SetExternalQueueBoundary(boundary);
	}
	graph->SetFrameWaitTimeline(m_frameWaitTimeline);
	graph->Setup();
	graph->SetGpuPassRangeCallbacks(m_gpuPassRangeBegin, m_gpuPassRangeEnd);
	instance.graph = std::move(graph);
	instance.rebuildRequested = false;
	++instance.buildGeneration;
	++m_buildGeneration;
	if (replacing)
		spdlog::info("GraphHost: epoch {}'s graph built again (build {}) in {:.2f} ms; the other epochs' graphs kept",
			instance.epoch == persistent::AllEpochs ? std::string("(all)") : std::to_string(instance.epoch), instance.buildGeneration,
			std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
}

bool GraphHost::BuildDueInstances(bool all) {
	std::vector<size_t> due;
	for (size_t index = 0; index < m_instances.size(); ++index) {
		auto& instance = m_instances[index];
		// Placements in a pool generation another epoch's graph replaced (its pool grew): placed again by a rebuild, they share
		// the pool's memory again, and the old generation is freed.
		if (instance.graph && !instance.rebuildRequested && !instance.graph->PersistentAliasPlacementsCurrent()) {
			instance.rebuildRequested = true;
			basic_telemetry::AddCounter("ORG.Host.AliasArenaRebuilds");
		}
		if (all ? !instance.graph || instance.rebuildRequested : BuildDue(index)) due.push_back(index);
	}
	if (m_async) RebuildAsyncInstances(due);
	else for (const auto index : due) BuildInstance(index, false);
	return !due.empty();
}

void GraphHost::SetFrameWaitTimeline(std::shared_ptr<rhi::TimelinePtr> timeline) {
	m_frameWaitTimeline = std::move(timeline);
	// Every packet binds its timelines when it is prepared: the graphs' present ones cannot wait on it.
	if (AnyGraph()) RequestRebuild();
}

GraphHost::GpuPoint GraphHost::SubmittedPoint() const {
	// Every queue of every graph, each timeline once: the shared ones' values are the host's, the same in each registry.
	GpuPoint point;
	for (const auto& instance : m_instances) {
		if (!instance.graph) continue;
		const auto& queues = instance.graph->GetQueueRegistry();
		for (size_t index = 0; index < queues.SlotCount(); ++index) {
			const auto slot = static_cast<QueueSlotIndex>(static_cast<uint8_t>(index));
			auto owner = queues.GetFenceOwner(slot);
			const auto next = queues.GetCurrentFenceValue(slot);
			if (!owner || next <= 1
				|| std::ranges::any_of(point.points, [&](const auto& known) { return known.first == owner; }))
				continue;
			point.points.emplace_back(std::move(owner), next - 1);
		}
	}
	return point;
}

void GraphHost::SealSubmitted() {
	auto point = SubmittedPoint();
	std::scoped_lock lock(m_sealedMutex);
	m_sealed = std::move(point);
}

GraphHost::GpuPoint GraphHost::SealedPoint() const {
	std::scoped_lock lock(m_sealedMutex);
	return m_sealed;
}

bool GraphHost::BuildIfRequested() {
	ReleaseRetiredGraphs(false);
	return BuildDueInstances(true);
}

void GraphHost::ExecuteFrame(const IHostExecutionData* hostData, const FrameCallback& beforePrepare,
	uint32_t epoch, std::shared_ptr<const void> resourceOwner) {
	BT_ZONE_SCOPE("ORG.Host.ExecuteFrame");
	if (m_async) throw std::logic_error("ExecuteFrame while async epochs run: the host's thread owns the graphs (use SubmitEpoch)");
	m_lastTimings = {};
	auto lap = [last = std::chrono::steady_clock::now()](double& a_into) mutable {
		const auto now = std::chrono::steady_clock::now();
		a_into += std::chrono::duration<double, std::micro>(now - last).count();
		last = now;
	};
	const auto index = epoch == persistent::AllEpochs ? 0 : InstanceOf(epoch);
	if (index == kNoInstance) throw std::invalid_argument("ExecuteFrame: epoch " + std::to_string(epoch) + " is not one of the host's");
	ReleaseRetiredGraphs(false);
	if (BuildDue(index)) BuildInstance(index, false);
	lap(m_lastTimings.buildUs);
	auto& graph = *m_instances[index].graph;
	runtime::ScopedActiveGraphServices services(graph.GetUploadService(), graph.GetDescriptorService());
	const auto slot = static_cast<uint32_t>(m_frameNumber % m_frameSlots);
	// The slot's previous frame must be done on the GPU before the upload pages it used are recycled
	// (and before this frame records uploads into the slot).
	if (auto& completion = m_slotCompletions[slot]; completion.pending) {
		{
			BT_ZONE_SCOPE("ORG.Host.WaitSlotCompletion");
			for (const auto& [timeline, value] : completion.points) (void)(*timeline)->HostWait(value);
		}
		completion.pending = false;
		if (m_retired) m_retired(completion.submission);
		// Nothing else completes a frame's statistics on this path: without it the pass timestamps were
		// written and resolved every frame and never read, and their pending resolves only grew.
		if (auto* stats = graph.GetStatisticsService()) {
			rhi::Queue queue = m_device.GetQueue(rhi::QueueKind::Graphics);
			stats->OnFrameComplete(slot, queue);
			if (m_completedFrame) m_completedFrame(completion.frameNumber, *stats);
		}
	}
	lap(m_lastTimings.waitUs);
	// A slot an external client's work still uses (NoteExternalSlotUse) keeps its upload pages for a later release.
	if (auto* uploads = graph.GetUploadService(); uploads && !ExternalSlotInUse(slot)) uploads->ProcessDeferredReleases(static_cast<uint8_t>(slot));
	m_desc.runtime->Retire();
	lap(m_lastTimings.releaseUs);
	graph.SetPersistentEpoch(epoch);
	graph.SetFrameWaitValue(m_frameWaitValue);
	if (beforePrepare) {
		BT_ZONE_SCOPE("ORG.Host.CommitInputs");
		beforePrepare(graph);
	}
	lap(m_lastTimings.beforePrepareUs);
	UpdateExecutionContext update{};
	update.frameIndex = slot;
	update.preparationSlot = slot;
	update.frameFenceValue = m_frameNumber + 1;
	update.hostData = hostData;
	graph.Update(update, m_device);
	lap(m_lastTimings.updateUs);
	PassExecutionContext execute{};
	execute.device = m_device;
	execute.frameIndex = slot;
	execute.frameFenceValue = m_frameNumber + 1;
	execute.hostData = hostData;
	// Reserve CPU ownership before submission. Execute may fail after accepting a
	// prefix of its batches, in which case recovery retains this owner.
	const bool heldResources = static_cast<bool>(resourceOwner);
	if (heldResources) m_uncertainExecutionOwners.push_back(std::move(resourceOwner));
	graph.Execute(execute);
	lap(m_lastTimings.executeUs);
	m_lastTimings.execute = graph.LastPersistentExecuteTimings();
	// Execute submitted the frame: every batch signalled its queue's timeline, so the value each queue
	// has reached is the frame's completion on it.
	{
		auto& completion = m_slotCompletions[slot];
		auto& queues = graph.GetQueueRegistry();
		completion.points.clear();
		for (size_t queueIndex = 0; queueIndex < queues.SlotCount(); ++queueIndex) {
			const auto queueSlot = static_cast<QueueSlotIndex>(static_cast<uint8_t>(queueIndex));
			if (const auto next = queues.GetCurrentFenceValue(queueSlot); next > 1)
				completion.points.emplace_back(queues.GetFenceOwner(queueSlot), next - 1);
		}
		completion.frameNumber = m_frameNumber;
		completion.submission = ++m_submissions;
		completion.pending = true;
		if (heldResources) {
			const auto& owner = m_uncertainExecutionOwners.back();
			if (owner && !completion.points.empty()) {
				std::vector<DescriptorHeapManager::QueueFenceSnapshotPoint> points;
				points.reserve(completion.points.size());
				for (const auto& [timeline, value] : completion.points)
					points.push_back({timeline->Get(), value});
				DescriptorHeapManager::GetInstance().RetireExecutionLease(owner, std::move(points));
			}
			m_uncertainExecutionOwners.pop_back();
		}
	}
	NoteSlotSubmitted(slot);
	lap(m_lastTimings.signalUs);
	m_lastHostFrame = m_frameNumber;
	++m_frameNumber;
}

// ---- async epochs ----


void GraphHost::SetAsyncEpochs(std::vector<uint32_t> epochs, std::vector<uint32_t> revisionEpochs) {
	StopAsync();
	for (const auto epoch : epochs)
		if (InstanceOf(epoch) == kNoInstance) throw std::invalid_argument("An async epoch must be one of the host's epochs");
	m_asyncEpochs = std::move(epochs);
	for (const auto epoch : revisionEpochs)
		if (std::ranges::find(m_asyncEpochs, epoch) == m_asyncEpochs.end())
			throw std::invalid_argument("A revision-driven epoch must be an async epoch");
	if (!revisionEpochs.empty() && !m_desc.reuseRecordings)
		throw std::logic_error("Revision-driven epochs need Desc::reuseRecordings (replayable recordings)");
	m_revisionEpochs = std::move(revisionEpochs);
	if (m_asyncEpochs.empty()) {
		m_recordingQueue->accepting.store(false, std::memory_order_release);
		return;
	}
	BuildDueInstances(false);
	StartAsync();
}

void GraphHost::SetExternalEpochs(std::vector<uint32_t> epochs) {
	// An epoch whose graph becomes or stops being external is built again; the rest keep theirs.
	for (const auto& instance : m_instances) {
		const bool was = std::ranges::find(m_externalEpochs, instance.epoch) != m_externalEpochs.end();
		const bool is = std::ranges::find(epochs, instance.epoch) != epochs.end();
		if (was != is) RequestRebuild(instance.epoch);
	}
	m_externalEpochs = std::move(epochs);
}

void GraphHost::SignalExternalExecutionUnexecuted(const ExternalEpochExecution& execution, size_t a_first) {
	if (!m_async) throw std::logic_error("An external execution without async epochs");
	// Each segment not executed, in its own submission: a later segment may wait for a batch on another queue that waits for an
	// earlier one's signal.
	for (size_t index = a_first == SIZE_MAX ? execution.ExecutedSegments() : a_first; index < execution.segments.size(); ++index) {
		const auto& segment = execution.segments[index];
		auto waits = segment.waits;
		auto signals = segment.signals;
		const auto result = m_async->graphicsQueue.Submit({}, {{waits.data(), static_cast<uint32_t>(waits.size())},
			{signals.data(), static_cast<uint32_t>(signals.size())}});
		if (result != rhi::Result::Ok)
			throw std::runtime_error(std::string("An unexecuted external execution could not be signalled (rhi::Result ") + rhi::ResultName(result) + ")");
	}
}

void GraphHost::StartAsync() {
	if (m_async || m_asyncEpochs.empty()) return;
	if (std::ranges::any_of(m_instances, [](const Instance& instance) { return !instance.graph; })) return;
	auto async = std::make_unique<Async>();
	// The host's wake, made once: a requester's Notify may reach it whenever, so no build destroys it.
	if (!m_completionWake && m_device.CreateCompletionWait(m_completionWake) != rhi::Result::Ok)
		throw std::runtime_error("Async epochs require wakeable GPU completion waits");
	async->completionWake = m_completionWake.get();
	async->epochs = m_asyncEpochs;
	async->revisionDriven.assign(async->epochs.size(), 0);
	for (uint32_t index = 0; index < async->epochs.size(); ++index) {
		async->revisionDriven[index] = std::ranges::find(m_revisionEpochs, async->epochs[index]) != m_revisionEpochs.end();
		const auto instance = InstanceOf(async->epochs[index]);
		async->instanceOf.push_back(instance);
		async->generation.push_back(m_instances[instance].buildGeneration);
	}
	async->paused.assign(async->epochs.size(), 0);
	// Each epoch its own ring of framesInFlight slots, when the slots hold one per async epoch (they hold framesInFlight of every
	// epoch in the order); otherwise one ring of all.
	if (const uint32_t ring = (std::max)(m_desc.framesInFlight, 1u); m_frameSlots >= ring * async->epochs.size())
		async->ringSize = ring;
	async->ringNext.assign(async->epochs.size(), 0);
	async->cells = std::make_unique<std::atomic<std::shared_ptr<Async::Ticket>>[]>(async->epochs.size());
	async->needsTicket.assign(async->epochs.size(), 1);
	async->liveRecordings.assign(async->epochs.size(), {});
	async->outstanding.assign(async->epochs.size(), {});
	const uint32_t slotCount = m_frameSlots;
	async->slots.resize(slotCount);
	// What the synchronous frames left: their slots' completions.
	for (uint32_t slot = 0; slot < slotCount && slot < m_slotCompletions.size(); ++slot) {
		auto& completion = m_slotCompletions[slot];
		async->slots[slot].points = completion.points;
		async->slots[slot].hostFrame = completion.frameNumber;
		async->slots[slot].submission = completion.submission;
		async->slots[slot].pending = completion.pending;
		completion = {};
	}
	async->nextSlot = m_frameNumber;
	async->hostFrame = m_frameNumber;
	async->submissionsSeen = m_submissions;
	// The queues' layout is every graph's: the primary queues first, on the host's timelines.
	auto& queues = m_instances.front().graph->GetQueueRegistry();
	const auto graphics = queues.FindGraphicsSlot();
	async->graphicsSlot = static_cast<uint32_t>(ToUnderlying(graphics));
	async->graphicsQueue = queues.GetQueue(graphics);
	async->graphicsFence = queues.GetFence(graphics).GetHandle();
	for (uint32_t slot = 0; slot < slotCount; ++slot) {
		CommandListPair pair;
		if (!rhi::IsOk(m_device.CreateCommandAllocator(rhi::QueueKind::Graphics, pair.allocator))
			|| !rhi::IsOk(m_device.CreateCommandList(rhi::QueueKind::Graphics, pair.allocator.Get(), pair.list)))
			throw std::runtime_error("Async epochs could not create their upload command lists");
		pair.list->SetName(("ORG ticket uploads #" + std::to_string(slot)).c_str());
		pair.list->End();
		async->uploadLists.push_back(std::move(pair));
		// An external epoch's uploads secondary, for every slot: an epoch may become external at any of its rebuilds.
		CommandListPair secondary;
		if (!rhi::IsOk(m_device.CreateCommandAllocator(rhi::QueueKind::Graphics, secondary.allocator))
			|| !rhi::IsOk(m_device.CreateSecondaryCommandList(rhi::QueueKind::Graphics, secondary.allocator.Get(), secondary.list)))
			throw std::runtime_error("External epochs could not create their upload secondary lists");
		secondary.list->SetName(("ORG external ticket uploads #" + std::to_string(slot)).c_str());
		async->uploadSecondaries.push_back(std::move(secondary));
		// Queue-scoped uploads: one list per compute or copy queue on the graphics queue's device (a copy needs no more).
		auto& perQueue = async->queueUploadLists.emplace_back(queues.SlotCount());
		for (size_t index = 0; index < queues.SlotCount(); ++index) {
			const auto queueSlot = static_cast<QueueSlotIndex>(static_cast<uint8_t>(index));
			const auto kind = queues.GetKind(queueSlot);
			if (queueSlot == graphics || (kind != QueueKind::Compute && kind != QueueKind::Copy)
				|| queues.GetBackendInstance(queueSlot) != queues.GetBackendInstance(graphics) || !queues.GetQueue(queueSlot))
				continue;
			const auto queueKind = kind == QueueKind::Compute ? rhi::QueueKind::Compute : rhi::QueueKind::Copy;
			CommandListPair queuePair;
			if (!rhi::IsOk(m_device.CreateCommandAllocator(queueKind, queuePair.allocator))
				|| !rhi::IsOk(m_device.CreateCommandList(queueKind, queuePair.allocator.Get(), queuePair.list)))
				throw std::runtime_error("Async epochs could not create their queue upload command lists");
			queuePair.list->SetName(("ORG ticket queue uploads #" + std::to_string(slot) + " q" + std::to_string(index)).c_str());
			queuePair.list->End();
			perQueue[index] = std::move(queuePair);
		}
	}
	async->queueLists.resize(queues.SlotCount());
	// Staged uploads go straight into each submission's upload list (RenderGraph::RecordPendingUploads).
	if (auto* uploads = m_instances.front().graph->GetUploadService()) uploads->SetStagedUploadsRecordedDirectly(true);
	auto* state = async.get();
	state->thread = std::thread([this, state] {
		const auto graphOf = [&](uint32_t epochIndex) -> RenderGraph& { return *m_instances[state->instanceOf[epochIndex]].graph; };
		try {
			for (;;) {
				const auto seen = state->completionWake->WakeVersion();
				bool stop = false;
				// Messages, in the order the render thread posted them.
				for (auto read = state->read.load(std::memory_order_relaxed); read != state->written.load(std::memory_order_acquire); ++read) {
					auto message = std::move(state->mailbox[read % Async::kMailbox]);
					state->read.store(read + 1, std::memory_order_release);
					switch (message.kind) {
					case Async::Message::Kind::Submitted: {
						auto& graph = graphOf(message.epochIndex);
						auto& slot = state->slots[RenderGraph::PersistentTicketSlot(*message.ticket)];
						state->uncertainUploadOwner = std::move(message.keepAlive);
						state->uncertainResourceOwner = std::move(message.resourceOwner);
						auto completion = graph.CompletePersistentTicket(*message.ticket, message.submission);
						if (message.uploadSignal.first != UINT32_MAX) {
							auto found = std::ranges::find(completion, message.uploadSignal.first, &std::pair<uint32_t, uint64_t>::first);
							if (found == completion.end()) completion.push_back(message.uploadSignal);
							else found->second = (std::max)(found->second, message.uploadSignal.second);
						}
						slot.points.clear();
						for (const auto& [queue, value] : completion)
							slot.points.emplace_back(graph.GetQueueRegistry().GetFenceOwner(static_cast<QueueSlotIndex>(static_cast<uint8_t>(queue))), value);
						slot.pending = !slot.points.empty();
						slot.reserved = false;
						slot.hostFrame = RenderGraph::PersistentTicketHostFrame(*message.ticket);
						slot.submission = message.number;
						state->submissionsSeen = (std::max)(state->submissionsSeen, message.number);
						if (state->outstanding[message.epochIndex] == message.ticket) state->outstanding[message.epochIndex].reset();
						slot.keepAlive = std::move(state->uncertainUploadOwner);
						slot.resourceOwner = std::move(state->uncertainResourceOwner);
						if (!state->paused[message.epochIndex]) state->needsTicket[message.epochIndex] = 1;
						if (message.retired) {
							BT_ZONE_SCOPE("ORG.Host.DestroyRetired.Worker");
							message.retired.reset();
						}
						break;
					}
					case Async::Message::Kind::Discard:
						state->slots[RenderGraph::PersistentTicketSlot(*message.ticket)].reserved = false;
						if (state->outstanding[message.epochIndex] == message.ticket) state->outstanding[message.epochIndex].reset();
						(void)graphOf(message.epochIndex).CompletePersistentTicket(*message.ticket, {});
						// A discard naming a slot is a stale ticket's reprepare: its replacement is prepared here, in the same
						// step, into the slot the render thread is recording. Marking the epoch as needing a ticket instead
						// would let the proactive pass below prepare one into a slot of its own choosing, which the render
						// thread would take while it records into the other slot's upload list.
						if (message.slot >= 0)
							PrepareTicket(*state, message.epochIndex, message.slot);
						else if (!state->paused[message.epochIndex])
							state->needsTicket[message.epochIndex] = 1;
						break;
					case Async::Message::Kind::Detach: {
						// The instances' epochs: their unsubmitted tickets abandoned, their recording requests dropped, and nothing prepared
						// for them until the instances are attached again. Their submitted work completes as before (its slots keep their points).
						const auto detaching = [&](size_t instance) {
							return std::ranges::any_of(message.instances, [&](const auto& entry) { return entry.first == instance; });
						};
						for (uint32_t index = 0; index < state->epochs.size(); ++index) {
							if (!detaching(state->instanceOf[index])) continue;
							state->paused[index] = 1;
							state->needsTicket[index] = 0;
							if (auto ticket = state->cells[index].exchange(nullptr, std::memory_order_acq_rel)) {
								state->slots[RenderGraph::PersistentTicketSlot(*ticket)].reserved = false;
								(void)graphOf(index).CompletePersistentTicket(*ticket, {});
							}
							state->outstanding[index].reset();
							state->liveRecordings[index].clear();
							const auto dropped = std::make_exception_ptr(EpochRecordingDropped(state->generation[index]));
							std::erase_if(state->pendingRequests, [&](const std::unique_ptr<Async::RecordingRequest>& request) {
								if (request->epochIndex != index) return false;
								if (!m_ending && request->done) request->done(nullptr, dropped);
								return true;
							});
						}
						state->detached.fetch_add(1, std::memory_order_acq_rel);
						state->detached.notify_all();
						{
							BT_ZONE_SCOPE("ORG.Host.HoldForBuild.Worker");
							for (auto seen = state->released.load(std::memory_order_acquire); seen < message.hold;
								seen = state->released.load(std::memory_order_acquire))
								state->released.wait(seen, std::memory_order_acquire);
						}
						break;
					}
					case Async::Message::Kind::Attach:
						for (const auto& [instance, generation] : message.instances)
							for (uint32_t index = 0; index < state->epochs.size(); ++index) {
								if (state->instanceOf[index] != instance) continue;
								state->paused[index] = 0;
								state->generation[index] = generation;
								state->needsTicket[index] = 1;
							}
						break;
					case Async::Message::Kind::Stop:
						stop = true;
						break;
					}
				}
				if (stop) break;
				// Recording requests: taken in posting order, each recording the slots of its ring that are free. One for an epoch that
				// is not an async one (or without rings) completes with that error now.
				for (auto& node : m_recordingQueue->Take()) {
					const auto found = std::ranges::find(state->epochs, node->epoch);
					if (found == state->epochs.end() || !state->ringSize) {
						if (node->done)
							node->done(nullptr, std::make_exception_ptr(std::invalid_argument("RequestEpochRecording: epoch " + std::to_string(node->epoch) +
								" is not an async epoch with a slot ring")));
						continue;
					}
					const auto epochIndex = static_cast<uint32_t>(found - state->epochs.begin());
					if (state->paused[epochIndex]) {
						// Its graph is being built again: the request is for the old one.
						if (node->done) node->done(nullptr, std::make_exception_ptr(EpochRecordingDropped(state->generation[epochIndex])));
						continue;
					}
					auto request = std::make_unique<Async::RecordingRequest>();
					request->epochIndex = epochIndex;
					request->hostData = std::move(node->hostData);
					request->done = std::move(node->done);
					state->pendingRequests.push_back(std::move(request));
				}
				for (uint32_t index = 0; index < state->epochs.size(); ++index)
					if (!state->paused[index]) graphOf(index).RetirePersistentExecutions();
				// A submitted upload batch owns its staging until the GPU has actually
				// signalled every queue that used the slot. Retire it on this worker even
				// if no later frame ever requests this slot again.
				for (auto& slot : state->slots) {
					if (!slot.keepAlive && !slot.resourceOwner) continue;
					bool complete = true;
					for (const auto& [timeline, value] : slot.points) {
						const auto reached = (*timeline)->GetCompletedValue();
						if (reached == UINT64_MAX || reached < value) {
							complete = false;
							break;
						}
					}
					if (complete) {
						slot.keepAlive.reset();
						slot.resourceOwner.reset();
					}
				}
				// The next ticket of every epoch whose last one was submitted: one frame ahead.
				bool prepared = false;
				for (uint32_t index = 0; index < state->epochs.size(); ++index) {
					if (!state->needsTicket[index] || state->paused[index] || state->cells[index].load(std::memory_order_acquire)) continue;
					PrepareTicket(*state, index, -1);
					prepared = true;
				}
				// Then the recording requests: the frames' tickets come first, so recording ahead never delays a submission.
				const bool requestsAdvanced = AdvanceRecordingRequests(*state);
				if (!prepared && !requestsAdvanced && !m_recordingQueue->head.load(std::memory_order_acquire)
					&& state->read.load(std::memory_order_relaxed) == state->written.load(std::memory_order_acquire)) {
					std::array<rhi::TimelinePoint, rhi::CompletionWait::MaxTimelines> heads{};
					uint32_t count = 0;
					bool retireAgain = false;
					for (const auto& slot : state->slots) if (slot.pending) {
						bool incomplete = false;
						for (const auto& [timeline, value] : slot.points) {
							const auto reached = (*timeline)->GetCompletedValue();
							if (reached == UINT64_MAX) throw std::runtime_error("GPU completion timeline failed");
							if (reached >= value) continue;
							incomplete = true;
							const auto handle = (*timeline)->GetHandle();
							uint32_t i = 0;
							for (; i < count; ++i) if (heads[i].t.index == handle.index && heads[i].t.generation == handle.generation) break;
							if (i < count) heads[i].value = (std::min)(heads[i].value, value);
							else {
								if (count == heads.size()) throw std::runtime_error("Too many GPU completion timelines");
								heads[count++] = {handle, value};
							}
						}
						// Completion may race the release pass above. Do not go to
						// sleep with an owner whose final signal already arrived.
						if (!incomplete && (slot.keepAlive || slot.resourceOwner)) retireAgain = true;
					}
					if (retireAgain) continue;
					// Posters skip the wake while this sleep has GPU completions to end it; a message written before this store
					// is seen by the look below, and one written after it sees the store.
					state->wakesOnGpu.store(count != 0, std::memory_order_seq_cst);
					if (state->read.load(std::memory_order_relaxed) != state->written.load(std::memory_order_seq_cst)) {
						state->wakesOnGpu.store(false, std::memory_order_seq_cst);
						continue;
					}
					const auto waited = state->completionWake->Wait({heads.data(), count}, seen);
					state->wakesOnGpu.store(false, std::memory_order_seq_cst);
					if (waited != rhi::Result::Ok)
						throw std::runtime_error("GPU completion wait failed");
				}
			}
		} catch (const std::exception& e) {
			state->error = e.what();
			spdlog::error("GraphHost: the host's thread failed: {}", state->error);
			state->failed.store(true, std::memory_order_release);
		} catch (...) {
			state->error = "unknown exception";
			spdlog::error("GraphHost: the host's thread failed: {}", state->error);
			state->failed.store(true, std::memory_order_release);
		}
		// Wake a render thread waiting for a ticket or a detach, so it sees the failure.
		state->ticketWake.fetch_add(1, std::memory_order_release);
		state->ticketWake.notify_all();
		state->detached.fetch_add(1, std::memory_order_acq_rel);
		state->detached.notify_all();
	});
	// From here the host's thread captures backings while the submitting thread runs.
	BufferBase::RegisterConcurrentBackingCapture();
	m_async = std::move(async);
	// Requests from now on (and any queued across the build that started this) are this thread's.
	m_recordingQueue->wake.store(m_completionWake.get(), std::memory_order_release);
	m_recordingQueue->accepting.store(true, std::memory_order_release);
	(void)m_completionWake->Notify();
}

void GraphHost::HoldAsync(std::span<const size_t> detach, const std::function<void()>& work) {
	auto& async = *m_async;
	async.ThrowIfFailed();
	// The host's thread lets go of the instances to detach and holds still while the work runs: whatever it was given before
	// (submissions' completions) it handles first, on the graphs as built. The other epochs keep their tickets.
	const auto before = async.detached.load(std::memory_order_acquire);
	const auto hold = ++async.holds;
	async.holding = true;
	struct Release {
		Async& async;
		uint64_t hold;
		~Release() {
			async.holding = false;
			async.released.store(hold, std::memory_order_release);
			async.released.notify_all();
		}
	} release{async, hold};
	Async::Message message;
	message.kind = Async::Message::Kind::Detach;
	message.hold = hold;
	for (const auto index : detach) message.instances.emplace_back(index, 0);
	if (!async.Post(message)) async.ThrowIfFailed();
	{
		BT_ZONE_SCOPE("ORG.Host.WaitHold");
		for (auto seen = async.detached.load(std::memory_order_acquire); seen == before; seen = async.detached.load(std::memory_order_acquire)) {
			async.ThrowIfFailed();
			async.detached.wait(seen, std::memory_order_acquire);
		}
	}
	async.ThrowIfFailed();
	work();
}

void GraphHost::RebuildAsyncInstances(std::span<const size_t> indices) {
	BT_ZONE_SCOPE("ORG.Host.RebuildAsyncInstances");
	if (indices.empty()) return;
	Async::Message attach;
	attach.kind = Async::Message::Kind::Attach;
	HoldAsync(indices, [&] {
		for (const auto index : indices) {
			BuildInstance(index, false);
			attach.instances.emplace_back(index, m_instances[index].buildGeneration);
		}
	});
	if (!m_async->Post(attach)) m_async->ThrowIfFailed();
}

std::shared_ptr<runtime::ResourceCleanupQueue> GraphHost::ResourceCleanup() const {
	return DescriptorHeapManager::GetInstance().GetResourceCleanupQueue();
}

void GraphHost::WaitSlot(Async& state, uint32_t slot) {
	// The slot's previous execution must be done before its command lists, statistics range and latch region are reused, or
	// before it is recorded for again: the wait happens here, on this thread, never on the render thread.
	auto& entry = state.slots[slot];
	if (!entry.pending) return;
	{
		BT_ZONE_SCOPE("ORG.Host.WaitSlotCompletion.Worker");
		for (const auto& [timeline, value] : entry.points) (void)(*timeline)->HostWait(value);
	}
	entry.pending = false;
	// Retired: every submission below the oldest one still running (all that were seen, with none). Slots complete in the order the
	// host's thread reuses them, not in submission order, so a slot's own number says nothing of the ones before it; a pending slot
	// whose points are reached counts as done.
	{
		uint64_t retired = state.submissionsSeen;
		for (const auto& other : state.slots) {
			if (!other.pending) continue;
			if (!PointsReached(other.points)) retired = (std::min)(retired, other.submission - 1);
		}
		if (retired > state.retiredReported) {
			state.retiredReported = retired;
			if (m_retired) m_retired(retired);
		}
	}
	// The runtime's statistics, which every graph shares (a graph's own pointer may be in the middle of a rebuild).
	if (const auto& stats = m_desc.runtime->Statistics()) {
		rhi::Queue queue = m_device.GetQueue(rhi::QueueKind::Graphics);
		stats->OnFrameComplete(slot, queue);
		if (m_completedFrame) m_completedFrame(entry.hostFrame, *stats);
	}
	entry.keepAlive.reset();
	entry.resourceOwner.reset();
}

bool GraphHost::ReleaseEpochTicket(uint32_t epoch) {
	if (!m_async) return false;
	auto& async = *m_async;
	const auto found = std::ranges::find(async.epochs, epoch);
	if (found == async.epochs.end()) return false;
	const auto index = static_cast<uint32_t>(found - async.epochs.begin());
	auto ticket = async.cells[index].exchange(nullptr, std::memory_order_acq_rel);
	if (!ticket) return false;
	// The host's thread prepares into an empty cell only once the ticket it last put there is submitted or discarded: the cell is the
	// render thread's until the discard is read, so a ticket the mailbox has no room for goes back.
	Async::Message discard;
	discard.kind = Async::Message::Kind::Discard;
	discard.epochIndex = index;
	discard.ticket = ticket;
	if (async.TryPost(discard)) return true;
	async.cells[index].store(std::move(ticket), std::memory_order_release);
	return false;
}

bool GraphHost::RequestEpochRecording(uint32_t epoch, std::shared_ptr<const IHostExecutionData> hostData, EpochRecordingCallback done) {
	// The host's queue alone (never m_async, which a build moves out from under any other thread): the host's thread validates the
	// epoch when it takes it, and a build drops what it did not take (StopAsync, an epoch's Detach), each completed.
	auto& queue = *m_recordingQueue;
	if (!done || !queue.accepting.load(std::memory_order_acquire)) return false;
	auto* node = new RecordingQueue::Request{epoch, std::move(hostData), std::move(done), nullptr};
	auto* head = queue.head.load(std::memory_order_relaxed);
	do node->next = head;
	while (!queue.head.compare_exchange_weak(head, node, std::memory_order_release, std::memory_order_relaxed));
	if (auto* wake = queue.wake.load(std::memory_order_acquire)) (void)wake->Notify();
	return true;
}

bool GraphHost::AdvanceRecordingRequests(Async& state) {
	bool advanced = false;
	for (auto it = state.pendingRequests.begin(); it != state.pendingRequests.end();) {
		auto& request = **it;
		std::exception_ptr error;
		try {
			if (state.paused[request.epochIndex]) {
				++it;
				continue;
			}
			auto& graph = *m_instances[state.instanceOf[request.epochIndex]].graph;
			if (!request.result) {
				request.result = std::make_shared<EpochRecording>();
				request.result->epoch = state.epochs[request.epochIndex];
				request.result->firstSlot = request.epochIndex * state.ringSize;
				request.result->slots.resize(state.ringSize);
				request.result->revision = request.hostData;
				request.result->buildGeneration = state.generation[request.epochIndex];
				request.recorded.assign(state.ringSize, 0);
			}
			for (uint32_t position = 0; position < state.ringSize; ++position) {
				const uint32_t slot = request.result->firstSlot + position;
				// A slot its epoch's unsubmitted ticket holds is recorded too: a revision-driven ticket holds the admission alone (its
				// preparation recorded nothing, and already waited out the slot's last work), and it cannot be submitted without this.
				if (request.recorded[position]) continue;
				// A slot whose work is in flight is recorded once that completes (the loop wakes at completions), so the host's thread never
				// waits on the GPU here and the epochs' tickets are not held up behind it. A live epoch's slot its ticket holds is recorded once
				// the ticket is submitted.
				const auto& entry = state.slots[slot];
				if (!state.revisionDriven[request.epochIndex] && entry.reserved) continue;
				if (entry.pending && std::ranges::any_of(entry.points, [](const auto& a_point) {
						const auto reached = (*a_point.first)->GetCompletedValue();
						return reached == UINT64_MAX || reached < a_point.second;
					}))
					continue;
				BT_ZONE_SCOPE("ORG.Host.RecordForRevision");
				runtime::ScopedActiveGraphServices services(graph.GetUploadService(), graph.GetDescriptorService());
				WaitSlot(state, slot);
				std::shared_ptr<const RenderGraph::PersistentRecording> recording;
				{
					Async::Preparation preparation;
					const auto version = state.backingVersion.load(std::memory_order_acquire);
					auto ticket = graph.PreparePersistentTicket(m_device, request.result->epoch, static_cast<uint8_t>(slot), state.hostFrame, version,
						request.hostData);
					recording = RenderGraph::PersistentTicketRecording(*ticket);
					// Recorded only: its admission is abandoned, never submitted.
					(void)graph.CompletePersistentTicket(*ticket, {});
				}
				if (!recording || !RenderGraph::PersistentRecordingReplayable(*recording))
					throw std::logic_error("A revision-driven epoch's recording is not replayable (a pass with lifecycle effects, or a late-bound slot)");
				request.result->slots[position] = std::move(recording);
				request.recorded[position] = 1;
				state.revisionSlotsRecorded.fetch_add(1, std::memory_order_relaxed);
				advanced = true;
			}
		} catch (...) {
			error = std::current_exception();
		}
		const bool complete = !error && std::ranges::all_of(request.recorded, [](uint8_t a_done) { return a_done != 0; });
		if (!error && !complete) {
			++it;
			continue;
		}
		auto done = std::move(request.done);
		auto result = error ? nullptr : std::shared_ptr<const EpochRecording>(std::move(request.result));
		const auto epochIndex = request.epochIndex;
		it = state.pendingRequests.erase(it);
		if (!error) {
			state.revisionRecordings.fetch_add(1, std::memory_order_relaxed);
			// Admitted from now on: by every ticket prepared next, and by the one waiting for submission, before the caller can
			// publish it to the submitting thread.
			state.liveRecordings[epochIndex].push_back(result);
			if (const auto& ticket = state.outstanding[epochIndex]) {
				try {
					AddRevisionCandidate(state, epochIndex, *ticket, *result);
				} catch (...) {
					error = std::current_exception();
					result = nullptr;
				}
			}
		}
		done(std::move(result), error);
		advanced = true;
	}
	return advanced;
}

void GraphHost::PrepareTicket(Async& state, uint32_t epochIndex, int32_t requestedSlot) {
	BT_ZONE_SCOPE("ORG.Host.PrepareTicket.Worker");
	auto& graph = *m_instances[state.instanceOf[epochIndex]].graph;
	const uint32_t slotCount = static_cast<uint32_t>(state.slots.size());
	uint32_t slot = 0;
	if (requestedSlot >= 0) {
		slot = static_cast<uint32_t>(requestedSlot);
	} else if (state.ringSize) {
		// The next slot of the epoch's ring no unsubmitted ticket holds.
		for (uint32_t attempt = 0;; ++attempt) {
			if (attempt == state.ringSize) throw std::logic_error("Every frame slot of the epoch's ring is held by an unsubmitted ticket");
			slot = epochIndex * state.ringSize + static_cast<uint32_t>(state.ringNext[epochIndex]++ % state.ringSize);
			if (!state.slots[slot].reserved) break;
		}
	} else {
		// The next slot no unsubmitted ticket holds.
		for (uint32_t attempt = 0;; ++attempt) {
			if (attempt == slotCount) throw std::logic_error("Every frame slot is held by an unsubmitted ticket");
			slot = static_cast<uint32_t>(state.nextSlot++ % slotCount);
			if (!state.slots[slot].reserved) break;
		}
	}
	auto& entry = state.slots[slot];
	runtime::ScopedActiveGraphServices services(graph.GetUploadService(), graph.GetDescriptorService());
	WaitSlot(state, slot);
	// The render thread records this ticket's uploads into an open list: resetting and beginning it is several
	// microseconds of driver work that belongs here.
	auto& uploads = state.uploadLists[slot];
	uploads.allocator->Recycle();
	uploads.list->Recycle(uploads.allocator.Get());
	// An external epoch's uploads secondary: recycled here, begun where it is recorded.
	if (auto& secondary = state.uploadSecondaries[slot]; secondary.list) {
		secondary.allocator->Recycle();
		secondary.list->Recycle(secondary.allocator.Get());
	}
	// The slot's queue upload lists likewise: their last submission went ahead of a batch of the slot's last ticket on their queue,
	// whose completion the wait above covers.
	for (auto& queueUploads : state.queueUploadLists[slot]) {
		if (!queueUploads.list) continue;
		queueUploads.allocator->Recycle();
		queueUploads.list->Recycle(queueUploads.allocator.Get());
	}
	m_desc.runtime->Retire();
	entry.reserved = true;
	std::shared_ptr<Async::Ticket> ticket;
	if (state.revisionDriven[epochIndex]) {
		// Nothing from live state: the ticket admits the revisions' recordings for its slot, the ones still held.
		ticket = graph.PrepareRevisionTicket(state.epochs[epochIndex], static_cast<uint8_t>(slot), state.hostFrame++);
		auto& live = state.liveRecordings[epochIndex];
		std::erase_if(live, [](const auto& a_recording) { return a_recording.expired(); });
		for (const auto& weak : live)
			if (const auto recording = weak.lock()) AddRevisionCandidate(state, epochIndex, *ticket, *recording);
		state.outstanding[epochIndex] = ticket;
	} else {
		Async::Preparation preparation;
		const auto version = state.backingVersion.load(std::memory_order_acquire);
		ticket = graph.PreparePersistentTicket(m_device, state.epochs[epochIndex], static_cast<uint8_t>(slot), state.hostFrame++, version);
		RenderGraph::SetPersistentTicketHostTag(*ticket, version);
		// The revisions' recordings still held, as alternatives the submitting thread may take instead (UseEpochRecording).
		auto& live = state.liveRecordings[epochIndex];
		std::erase_if(live, [](const auto& a_recording) { return a_recording.expired(); });
		for (const auto& weak : live)
			if (const auto recording = weak.lock()) AddRevisionCandidate(state, epochIndex, *ticket, *recording);
		state.outstanding[epochIndex] = ticket;
	}
	state.needsTicket[epochIndex] = 0;
	state.cells[epochIndex].store(std::move(ticket), std::memory_order_release);
	state.ticketWake.fetch_add(1, std::memory_order_release);
	state.ticketWake.notify_all();
}

void GraphHost::AddRevisionCandidate(Async& state, uint32_t epochIndex, RenderGraph::PersistentTicket& ticket, const EpochRecording& recording) {
	const uint32_t slot = RenderGraph::PersistentTicketSlot(ticket);
	if (slot < recording.firstSlot || slot - recording.firstSlot >= recording.slots.size())
		throw std::logic_error("A revision recording does not cover its epoch's ticket slot");
	m_instances[state.instanceOf[epochIndex]].graph->AddPersistentTicketCandidate(ticket, recording.slots[slot - recording.firstSlot]);
}

bool GraphHost::CanUseEpochRecording(const EpochRecording& recording, std::string* why) const {
	auto fail = [&](const char* reason) {
		if (why) *why = reason;
		return false;
	};
	if (!m_async || !m_submittingTicket) return fail("not inside a live async epoch's submission");
	const auto& async = *m_async;
	if (async.revisionDriven[m_submittingIndex]) return fail("a revision-driven epoch (its recording is SubmitEpoch's)");
	if (recording.epoch != async.epochs[m_submittingIndex]) return fail("recorded for another epoch");
	const uint32_t slot = RenderGraph::PersistentTicketSlot(*m_submittingTicket);
	if (slot < recording.firstSlot || slot - recording.firstSlot >= recording.slots.size() || !recording.slots[slot - recording.firstSlot])
		return fail("no recording for the ticket's slot");
	return RenderGraph::CanBindPersistentTicketRecording(*m_submittingTicket, *recording.slots[slot - recording.firstSlot],
		async.backingVersion.load(std::memory_order_acquire), why);
}

void GraphHost::NoteNewVersions() noexcept {
	// After the version's publication (release), so a preparation that reads the new value resolves the new version.
	if (m_async) m_async->backingVersion.fetch_add(1, std::memory_order_acq_rel);
}

void GraphHost::StopAsync() {
	BT_ZONE_SCOPE("ORG.Host.StopAsync");
	if (!m_async) return;
	auto async = std::move(m_async);
	Async::Message stop;
	stop.kind = Async::Message::Kind::Stop;
	(void)async->Post(stop);
	if (async->thread.joinable()) async->thread.join();
	BufferBase::UnregisterConcurrentBackingCapture();
	// The recording requests not recorded: dropped, each completed so (EpochRecordingDropped), here, on the thread that builds -
	// the taken ones and the ones still queued (a request posted after this goes to the next thread, on the next build's graph).
	// The host's end drops them without their callbacks.
	if (!m_ending) {
		for (auto& request : async->pendingRequests)
			if (request && request->done)
				request->done(nullptr, std::make_exception_ptr(EpochRecordingDropped(async->generation[request->epochIndex])));
		const auto dropped = std::make_exception_ptr(EpochRecordingDropped(m_buildGeneration));
		for (auto& node : m_recordingQueue->Take())
			if (node->done) node->done(nullptr, dropped);
	}
	async->pendingRequests.clear();
	// Unsubmitted tickets are abandoned; submitted work keeps its slots' completions for the synchronous path.
	if (!async->failed.load()) {
		for (uint32_t index = 0; index < async->epochs.size(); ++index)
			if (auto ticket = async->cells[index].exchange(nullptr))
				if (auto& graph = m_instances[async->instanceOf[index]].graph) (void)graph->CompletePersistentTicket(*ticket, {});
	}
	for (uint32_t slot = 0; slot < async->slots.size() && slot < m_slotCompletions.size(); ++slot) {
		auto& completion = m_slotCompletions[slot];
		completion.points = async->slots[slot].points;
		completion.frameNumber = async->slots[slot].hostFrame;
		completion.submission = async->slots[slot].submission;
		completion.pending = async->slots[slot].pending;
	}
	m_frameNumber = (std::max)(m_frameNumber, async->hostFrame);
	if (auto* graph = AnyGraph())
		if (auto* uploads = graph->GetUploadService()) uploads->SetStagedUploadsRecordedDirectly(false);
	// The render thread's upload lists may still be executing: wait for the queue before they go.
	for (auto& slot : async->slots) {
		bool complete = true;
		for (const auto& [timeline, value] : slot.points)
			if ((*timeline)->HostWait(value) != rhi::Result::Ok) complete = false;
		if (!complete) {
			if (slot.resourceOwner) m_uncertainExecutionOwners.push_back(std::move(slot.resourceOwner));
			if (slot.keepAlive) m_uncertainUploadOwners.push_back(std::move(slot.keepAlive));
		}
	}
	// Every slot's work is done (or its owners kept as uncertain): the kept recordings can go.
	for (auto& instance : m_instances)
		if (instance.graph) instance.graph->ClearKeptPersistentRecordings();
	if (async->uncertainResourceOwner)
		m_uncertainExecutionOwners.push_back(std::move(async->uncertainResourceOwner));
	if (async->uncertainUploadOwner)
		m_uncertainUploadOwners.push_back(std::move(async->uncertainUploadOwner));
}

GraphHost::AsyncStats GraphHost::TakeAsyncStats() noexcept {
	m_asyncStats.recorded = m_asyncStats.kept = m_asyncStats.reused = m_asyncStats.adopted = 0;
	m_asyncStats.predicted = m_asyncStats.planned = 0;
	for (auto& instance : m_instances) {
		if (!instance.graph) continue;
		const auto recordings = instance.graph->TakePersistentRecordingStats();
		m_asyncStats.recorded += recordings.recorded;
		m_asyncStats.kept += recordings.kept;
		m_asyncStats.reused += recordings.reused;
		m_asyncStats.adopted += recordings.adopted;
		m_asyncStats.predicted += recordings.predicted;
		m_asyncStats.planned += recordings.planned;
	}
	if (m_async) {
		m_asyncStats.revisionSlotsRecorded = m_async->revisionSlotsRecorded.exchange(0, std::memory_order_relaxed);
		m_asyncStats.revisionRecordings = m_async->revisionRecordings.exchange(0, std::memory_order_relaxed);
		m_asyncStats.wakesSkipped = std::exchange(m_async->wakesSkipped, 0);
	}
	return std::exchange(m_asyncStats, {});
}

void GraphHost::SubmitEpoch(uint32_t epoch, const FrameCallback& beforeSubmit, std::shared_ptr<const void> resourceOwner,
	std::shared_ptr<const EpochRecording> recording) {
	BT_ZONE_SCOPE("ORG.Host.SubmitEpoch");
	m_lastTimings = {};
	m_lastTimings.async = true;
	auto lap = [last = std::chrono::steady_clock::now()](double& a_into) mutable {
		const auto now = std::chrono::steady_clock::now();
		a_into += std::chrono::duration<double, std::micro>(now - last).count();
		last = now;
	};
	ReleaseRetiredGraphs(false);
	const auto instance = InstanceOf(epoch);
	if (instance == kNoInstance) throw std::invalid_argument("SubmitEpoch: epoch " + std::to_string(epoch) + " is not one of the host's");
	if (!m_async && !m_asyncEpochs.empty()) {
		BuildDueInstances(false);
		StartAsync();
	}
	if (!m_async) throw std::logic_error("SubmitEpoch requires async epochs");
	// The epoch's own graph: built again here when due, the other epochs' tickets kept.
	if (BuildDue(instance)) RebuildAsyncInstances({&instance, 1});
	lap(m_lastTimings.buildUs);
	auto& async = *m_async;
	async.ThrowIfFailed();
	const auto index = async.IndexOf(epoch);
	auto ticket = async.cells[index].exchange(nullptr, std::memory_order_acq_rel);
	if (!ticket) {
		++m_asyncStats.waited;
		ticket = async.WaitTicket(index);
	}
	lap(m_lastTimings.ticketWaitUs);
	SubmitTicket(async, index, std::move(ticket), beforeSubmit, std::move(resourceOwner), std::move(recording));
	// Completed readback captures reach their callbacks here: a ticket's frame never runs Update, which polls them.
	if (auto* readback = m_instances[instance].graph->GetReadbackService()) readback->ProcessReadbackRequests();
}

void GraphHost::SubmitTicket(Async& async, uint32_t index, std::shared_ptr<RenderGraph::PersistentTicket> ticket,
	const FrameCallback& beforeSubmit, std::shared_ptr<const void> resourceOwner, std::shared_ptr<const EpochRecording> recording) {
	BT_ZONE_SCOPE("ORG.Host.SubmitTicket");
	BT_ZONE_VALUE(async.epochs[index]);
	// A submission's completion message wakes the host's thread lazily (wakesOnGpu); a discard always wakes it.
	auto postControl = [&](Async::Message& message) {
		const bool lazy = message.kind == Async::Message::Kind::Submitted;
		if (!async.Post(message, lazy)) throw std::runtime_error("Async epoch host stopped before accepting its completion");
	};
	auto lap = [last = std::chrono::steady_clock::now()](double& a_into) mutable {
		const auto now = std::chrono::steady_clock::now();
		a_into += std::chrono::duration<double, std::micro>(now - last).count();
		last = now;
	};
	auto& graph = *m_instances[async.instanceOf[index]].graph;
	const uint32_t slot = RenderGraph::PersistentTicketSlot(*ticket);
	m_asyncSlot = slot;
	runtime::ScopedActiveGraphServices services(graph.GetUploadService(), graph.GetDescriptorService());
	// The host's thread waited for this slot before preparing the ticket: its upload pages are free again. What retires from
	// the deletion queues goes to the host's thread with the submission: its driver frees take the device's memory locks,
	// which the render thread must not wait on here.
	if (auto* uploads = graph.GetUploadService(); uploads && !ExternalSlotInUse(slot))
		uploads->ProcessDeferredReleases(static_cast<uint8_t>(slot));
	std::shared_ptr<void> retired = m_desc.runtime->RetireElsewhere();
	lap(m_lastTimings.releaseUs);
	if (beforeSubmit) {
		BT_ZONE_SCOPE("ORG.Host.CommitInputs");
		struct Submitting {
			GraphHost& host;
			~Submitting() { host.m_submittingTicket = nullptr; }
		} submitting{*this};
		m_submittingTicket = ticket.get();
		m_submittingIndex = index;
		beforeSubmit(graph);
	}
	lap(m_lastTimings.beforePrepareUs);
	// A revision-driven epoch: the revision's recording for this slot, recorded ahead. Never prepared again: a recording that
	// does not match the ticket is the caller's error.
	// A live epoch's alternative: the revision's recording the commit chose (UseEpochRecording), admitted as the ticket's candidate.
	if (!recording) recording = std::exchange(m_chosenRecording, nullptr);
	m_chosenRecording.reset();
	if (!async.revisionDriven[index] && recording) {
		std::string why;
		const uint32_t position = slot - recording->firstSlot;
		const bool bound = recording->epoch == async.epochs[index] && slot >= recording->firstSlot && position < recording->slots.size()
			&& RenderGraph::BindPersistentTicketRecording(*ticket, recording->slots[position], async.backingVersion.load(std::memory_order_acquire), &why);
		if (!bound) {
			Async::Message discard;
			discard.kind = Async::Message::Kind::Discard;
			discard.epochIndex = index;
			discard.ticket = std::move(ticket);
			postControl(discard);
			throw std::logic_error("A revision's recording chosen for a live epoch is not its ticket's candidate: " + why);
		}
		++m_asyncStats.revisionSubmissions;
	}
	const bool chosen = !async.revisionDriven[index] && recording;
	if (async.revisionDriven[index]) {
		std::string why = "no recording given";
		const uint32_t position = recording ? slot - recording->firstSlot : 0;
		const bool bound = recording && recording->epoch == async.epochs[index] && slot >= recording->firstSlot && position < recording->slots.size()
			&& RenderGraph::BindPersistentTicketRecording(*ticket, recording->slots[position], async.backingVersion.load(std::memory_order_acquire), &why);
		if (!bound) {
			Async::Message discard;
			discard.kind = Async::Message::Kind::Discard;
			discard.epochIndex = index;
			discard.ticket = std::move(ticket);
			postControl(discard);
			throw std::logic_error("A revision-driven epoch was submitted without its revision's recording for the slot: " + why);
		}
		++m_asyncStats.revisionSubmissions;
	}
	// Stale, but recorded before for what its passes depend on now (Desc::reuseRecordings): it takes that recording, and
	// nothing is prepared again.
	if (!async.revisionDriven[index] && !chosen && !async.Current(*ticket)
		&& graph.AdoptKeptPersistentRecording(*ticket, async.backingVersion.load(std::memory_order_acquire))
		&& !async.Current(*ticket))
		throw std::logic_error("An adopted recording left its ticket stale");
	if (!async.revisionDriven[index] && !chosen && !async.Current(*ticket)) {
		BT_ZONE_SCOPE("ORG.Host.ReprepareStaleTicket");
		m_asyncStats.staleBacking += RenderGraph::PersistentTicketCurrent(*ticket);
		// Prepared before beforeSubmit changed what a pass depends on, or a version it resolved: prepare it again, for the
		// same slot (its latch region was just written), and wait. The new one is prepared after the change, so it is current.
		++m_asyncStats.stale;
		Async::Message discard;
		discard.kind = Async::Message::Kind::Discard;
		discard.epochIndex = index;
		discard.ticket = std::move(ticket);
		// The discard and the reprepare are one message, so the worker prepares the replacement into this slot before it
		// can prepare the epoch's next ticket anywhere else (the Discard handler).
		discard.slot = static_cast<int32_t>(slot);
		postControl(discard);
		ticket = async.WaitTicket(index);
		if (RenderGraph::PersistentTicketSlot(*ticket) != slot)
			throw std::logic_error("A reprepared async epoch ticket is not for the slot whose upload list the render thread records");
	}
	// The submitted ticket's frame: one prepared again above has a new number, and the completed-frame callback
	// reports it by that one.
	m_lastHostFrame = RenderGraph::PersistentTicketHostFrame(*ticket);
	lap(m_lastTimings.checkUs);
	// The queues' last values, which every graph assigns from (the host's shared timelines): this submission's come after them,
	// and go back to the registry as soon as they are assigned, before any other graph's submission.
	auto& queues = graph.GetQueueRegistry();
	async.queueValues.resize(queues.SlotCount());
	for (size_t queue = 0; queue < queues.SlotCount(); ++queue) {
		const auto next = queues.GetCurrentFenceValue(static_cast<QueueSlotIndex>(static_cast<uint8_t>(queue)));
		async.queueValues[queue] = next ? next - 1 : 0;
	}
	struct PublishQueueValues {
		QueueRegistry& queues;
		const std::vector<uint64_t>& values;
		void Publish() const noexcept {
			for (size_t queue = 0; queue < values.size(); ++queue)
				queues.EnsureNextFenceValueAtLeast(static_cast<QueueSlotIndex>(static_cast<uint8_t>(queue)), values[queue] + 1);
		}
		~PublishQueueValues() { Publish(); }
	} publishQueueValues{queues, async.queueValues};
	// Admitted now, against the ledgers as they are: the upload plan by queue and the submission's handoff read it. The sequencer
	// stays held until the ticket is submitted and committed, so no other admission comes between.
	auto sequencer = graph.HoldPersistentSequencer();
	graph.AdmitPersistentTicket(*ticket, async.queueValues);
	struct AbandonUnsubmitted {
		RenderGraph& graph;
		RenderGraph::PersistentTicket* ticket;
		~AbandonUnsubmitted() { if (ticket) graph.AbandonPersistentTicketAdmission(*ticket); }
	} abandonUnsubmitted{graph, ticket.get()};
	// A ticket handed back to the host's thread unsubmitted gives up its admission first (the host's thread completes it).
	auto abandonBeforeDiscard = [&] {
		abandonUnsubmitted.ticket = nullptr;
		graph.AbandonPersistentTicketAdmission(*ticket);
	};
	// The uploads queued since the last submission, as plain copies ahead of the ticket (the list ends with a full barrier).
	auto& list = async.uploadLists[slot];  // begun by the host's thread (PrepareTicket)
	std::shared_ptr<void> keepAlive;
	bool recorded = false;
	// An external epoch's uploads go into the slot's secondary, which its execution runs first: nothing is submitted here.
	const bool external = graph.IsExternalPersistentEpoch(async.epochs[index]);
	auto& secondary = async.uploadSecondaries[slot];
	if (external) {
		rhi::SecondaryInheritance outside{};
		outside.insideRendering = false;
		outside.debugName = "ORG external ticket uploads";
		if (!secondary.list || secondary.list->BeginSecondary(outside) != rhi::Result::Ok)
			throw std::runtime_error("An external epoch's uploads secondary could not begin");
	}
	auto commands = external ? secondary.list.Get() : list.list.Get();
	auto& recordedLists = async.recordedLists;
	// Queue-scoped uploads (Desc::queueScopedUploads): the slot's open lists of the other queues, which take the uploads their
	// queue uses first, and go to it just ahead of the ticket's first batch there (SubmitPersistentTicket).
	auto& queueUploadLists = async.queueUploadLists[slot];
	const bool queueScoped = m_desc.queueScopedUploads;
	for (size_t queue = 0; queue < async.queueLists.size(); ++queue)
		async.queueLists[queue] = queueScoped && queue < queueUploadLists.size() && queueUploadLists[queue].list
			? queueUploadLists[queue].list.Get() : rhi::CommandList{};
	size_t queueCopies = 0;
	const bool supported = graph.RecordPendingUploads(m_device, commands, static_cast<uint8_t>(slot), keepAlive, recorded, recordedLists,
		queueScoped ? ticket.get() : nullptr, queueScoped ? std::span<rhi::CommandList>(async.queueLists) : std::span<rhi::CommandList>{}, &queueCopies);
	{
		BT_ZONE_SCOPE("ORG.Host.CloseUploadList");
		list.list->End();
		if (external) secondary.list->End();
		// Every one the host's thread began, used or not (it begins them all again for the slot's next ticket).
		for (auto& queueUploads : queueUploadLists)
			if (queueUploads.list) queueUploads.list->End();
	}
	m_asyncStats.queueUploadCopies += queueCopies;
	if (external && !recordedLists.empty()) {
		recordedLists.clear();
		abandonBeforeDiscard();
		Async::Message discard;
		discard.kind = Async::Message::Kind::Discard;
		discard.epochIndex = index;
		discard.ticket = std::move(ticket);
		postControl(discard);
		throw std::runtime_error("An external epoch's uploads include producer-recorded lists, which its execution cannot run");
	}
	if (!supported) {
		abandonBeforeDiscard();
		Async::Message discard;
		discard.kind = Async::Message::Kind::Discard;
		discard.epochIndex = index;
		discard.ticket = std::move(ticket);
		postControl(discard);
		throw std::runtime_error("Async epochs cannot record these uploads (only pointer-targeted buffer copies)");
	}
	Async::Message submitted;
	submitted.kind = Async::Message::Kind::Submitted;
	submitted.number = ++m_submissions;
	submitted.epochIndex = index;
	submitted.keepAlive = std::move(keepAlive);
	if (!m_externalRetired.empty()) {
		m_externalRetired.push_back(std::move(retired));
		retired = std::make_shared<std::vector<std::shared_ptr<void>>>(std::exchange(m_externalRetired, {}));
	}
	submitted.retired = std::move(retired);
	submitted.resourceOwner = std::move(resourceOwner);
	if (!external && (recorded || !recordedLists.empty())) {
		BT_ZONE_SCOPE("ORG.Host.EnqueueUploads");
		++m_asyncStats.uploads;
		m_asyncStats.recordedUploadLists += recordedLists.size();
		const auto value = ++async.queueValues.at(async.graphicsSlot);
		const rhi::TimelinePoint signal{async.graphicsFence, value};
		// The producers' recorded lists, then this thread's (an empty one costs nothing but its begin and end).
		recordedLists.push_back(commands);
		rhi::Result uploadResult = rhi::Result::Ok;
		try {
			uploadResult = async.graphicsQueue.Submit({recordedLists.data(), static_cast<uint32_t>(recordedLists.size())}, {{}, {&signal, 1}});
		} catch (...) {
			if (submitted.resourceOwner) m_uncertainExecutionOwners.push_back(std::move(submitted.resourceOwner));
			if (submitted.keepAlive) m_uncertainUploadOwners.push_back(std::move(submitted.keepAlive));
			throw;
		}
		if (uploadResult != rhi::Result::Ok) {
			if (submitted.resourceOwner) m_uncertainExecutionOwners.push_back(std::move(submitted.resourceOwner));
			if (submitted.keepAlive) m_uncertainUploadOwners.push_back(std::move(submitted.keepAlive));
			throw std::runtime_error(std::string("Async epochs could not submit their uploads (rhi::Result ") + rhi::ResultName(uploadResult) + ")");
		}
		submitted.uploadSignal = {async.graphicsSlot, value};
	}
	lap(m_lastTimings.uploadsUs);
	bool ok = false;
	try {
		// Submits and commits, or abandons its admission itself.
		abandonUnsubmitted.ticket = nullptr;
		ok = graph.SubmitPersistentTicket(*ticket, async.queueValues, submitted.submission, submitted.uploadSignal, m_frameWaitValue,
			external && recorded);
	} catch (...) {
		// The queue may have accepted a prefix of this ticket. Completion is
		// unknown, so preserve both owners until device teardown.
		if (submitted.resourceOwner) m_uncertainExecutionOwners.push_back(std::move(submitted.resourceOwner));
		if (submitted.keepAlive) m_uncertainUploadOwners.push_back(std::move(submitted.keepAlive));
		throw;
	}
	publishQueueValues.Publish();
	sequencer.unlock();
	lap(m_lastTimings.submitUs);
	// The handoff and release lists the submission recorded: kept until the slot completes, with the uploads' staging.
	if (!submitted.submission.leases.empty()) {
		struct Kept {
			std::shared_ptr<void> uploads;
			std::vector<std::shared_ptr<const void>> leases;
		};
		submitted.keepAlive = std::make_shared<Kept>(Kept{std::move(submitted.keepAlive), std::move(submitted.submission.leases)});
	}
	if (external && ok) {
		// The caller's to execute (TakeExternalExecution): the uploads first, then the handoff into the execution.
		auto execution = submitted.submission.external;
		if (!execution) throw std::logic_error("An external epoch's ticket gave no execution");
		if (recorded) {
			if (execution->segments.empty()) execution->segments.emplace_back();
			auto& leading = execution->segments.front().leading;
			leading.insert(leading.begin(), commands);
		}
		m_externalExecution = std::move(execution);
	}
	if (submitted.uploadSignal.second) m_asyncStats.sparedUploadWaits += RenderGraph::PersistentTicketSparedUploadWaits(*ticket);
	submitted.ticket = std::move(ticket);
	try {
		postControl(submitted);
	} catch (...) {
		if (submitted.resourceOwner) m_uncertainExecutionOwners.push_back(std::move(submitted.resourceOwner));
		if (submitted.keepAlive) m_uncertainUploadOwners.push_back(std::move(submitted.keepAlive));
		throw;
	}
	lap(m_lastTimings.postUs);
	NoteSlotSubmitted(slot);
	++m_asyncStats.submitted;
	if (!ok) throw std::runtime_error("Async epoch submission failed");
}

void GraphHost::NoteExternalSlotUse(uint32_t slot, std::shared_ptr<rhi::TimelinePtr> timeline, uint64_t value) {
	if (!timeline || !*timeline || !value || !m_frameSlots) return;
	slot %= m_frameSlots;
	if (m_externalSlotValues.size() != m_frameSlots) m_externalSlotValues.assign(m_frameSlots, 0);
	m_externalTimeline = timeline;
	m_externalSlotValues[slot] = (std::max)(m_externalSlotValues[slot], value);
	DescriptorHeapManager::GetInstance().PublishQueueFenceSnapshot({ DescriptorHeapManager::QueueFenceSnapshotPoint{ .timeline = timeline->Get(), .value = value } });
}

void GraphHost::BeginExternalFrameSlot(uint32_t slot) {
	m_externalSlot = m_frameSlots ? slot % m_frameSlots : 0u;
	// The client waited out its own last work at the slot (the contract above). The host's last work there must be done too, or
	// the slot is left to the host's next execution taking it, or to the client's next frame at it.
	auto* graph = AnyGraph();
	auto* uploads = graph ? graph->GetUploadService() : nullptr;
	if (!uploads || (m_externalSlot < m_slotSubmitted.size() && !m_slotSubmitted[m_externalSlot].Reached())) return;
	uploads->ProcessDeferredReleases(static_cast<uint8_t>(m_externalSlot));
	if (!m_async) {
		m_desc.runtime->Retire();
		return;
	}
	if (auto retired = m_desc.runtime->RetireElsewhere()) m_externalRetired.push_back(std::move(retired));
	// No submission came to take them for a round of slots: they are freed here rather than kept.
	if (m_externalRetired.size() > m_frameSlots) m_externalRetired.clear();
}

void GraphHost::NoteSlotSubmitted(uint32_t slot) {
	if (m_slotSubmitted.size() != m_frameSlots) m_slotSubmitted.resize(m_frameSlots);
	m_slotSubmitted[slot % m_frameSlots] = SubmittedPoint();
}

bool GraphHost::ExternalSlotInUse(uint32_t slot) const noexcept {
	if (!m_externalTimeline || !*m_externalTimeline || m_frameSlots == 0 || m_externalSlotValues.size() != m_frameSlots) return false;
	const uint64_t value = m_externalSlotValues[slot % m_frameSlots];
	if (!value) return false;
	const uint64_t completed = (*m_externalTimeline)->GetCompletedValue();
	// UINT64_MAX: a lost device, whose pending work will never read anything.
	return completed != UINT64_MAX && completed < value;
}

} // namespace org
