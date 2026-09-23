#include "OpenRenderGraph/PersistentGraphHost.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>

#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Render/PassExecutionContext.h"
#include "Render/Runtime/IUploadService.h"
#include "Render/Runtime/OpenRenderGraphSettings.h"
#include "Render/Runtime/RuntimeDevice.h"
#include "Render/Runtime/ScopedActiveGraphServices.h"
#include "Render/Runtime/ThreadPoolTaskService.h"

namespace org {

namespace {

constexpr const char* kUploadPassName = "org.host.uploads";

// The built-in upload pass: graph-ordered buffer/texture uploads requested with
// BUFFER_UPLOAD and friends. It records immediate-mode copies, so it runs in the
// persistent Pre segment ahead of the main executable.
class HostUploadExtension final : public RenderGraph::IRenderGraphExtension {
public:
	explicit HostUploadExtension(runtime::IUploadService* uploads) : m_uploads(uploads) {}
	void OnRegistryReset(ResourceRegistry* registry) override {
		if (m_uploads) m_uploads->SetUploadResolveContext({registry, 0});
	}
	void GatherStructuralPasses(RenderGraph&, std::vector<RenderGraph::ExternalPassDesc>& out) override {
		if (!m_uploads) return;
		if (auto pass = m_uploads->GetUploadPass())
			out.push_back(RenderGraph::ExternalPassDesc::Render(kUploadPassName, std::move(pass))
				.At(RenderGraph::ExternalInsertPoint::Begin(-1000)).CollectStatistics(false));
	}
private:
	runtime::IUploadService* m_uploads;
};

} // namespace

PersistentGraphHost::PersistentGraphHost(Desc desc) : m_desc(std::move(desc)) {
	if (!m_desc.device) throw std::invalid_argument("PersistentGraphHost requires a device");
	if (m_desc.backend == rhi::Backend::Null) throw std::invalid_argument("PersistentGraphHost requires a device backend");
	if (!m_desc.tasks) m_desc.tasks = std::make_shared<runtime::ThreadPoolTaskService>();
	// The host's slot count is the one frames-in-flight: upload pages, statistics query ranges, deletion
	// and admission capacity all size themselves from the runtime settings, so they follow it.
	{
		auto settings = runtime::GetOpenRenderGraphSettings();
		settings.numFramesInFlight = static_cast<uint8_t>((std::clamp)(m_desc.framesInFlight, 1u, 255u));
		runtime::SetOpenRenderGraphSettings(settings);
	}
	runtime::InitializeRuntimeDevice(m_desc.device);
	m_slotCompletions.assign((std::max)(m_desc.framesInFlight, 1u), {});
}

PersistentGraphHost::~PersistentGraphHost() {
	StopAsync();
	DestroyGraph();
	runtime::ShutdownRuntimeDevice();
}

void PersistentGraphHost::AddExtension(std::string id, ExtensionFactory factory) {
	RemoveExtension(id);
	m_extensions.push_back({std::move(id), std::move(factory)});
	m_rebuildRequested = true;
}

void PersistentGraphHost::RemoveExtension(const std::string& id) {
	const auto erased = std::erase_if(m_extensions, [&](const Registered& entry) { return entry.id == id; });
	if (erased) m_rebuildRequested = true;
}

runtime::IUploadService* PersistentGraphHost::Uploads() noexcept {
	return m_graph ? m_graph->GetUploadService() : nullptr;
}

void PersistentGraphHost::DestroyGraph() {
	StopAsync();
	if (!m_graph) return;
	m_graph->StopFrameProduction();
	// Adopted devices wait on their own timelines only (never the whole device).
	(void)m_desc.device.WaitIdle();
	for (auto& completion : m_slotCompletions) completion = {};
	m_graph->ShutdownExtensions();
	m_graph->ShutdownTaskWorkers();
	m_graph.reset();
}

void PersistentGraphHost::Build() {
	DestroyGraph();
	auto graph = std::make_unique<RenderGraph>(m_desc.device, m_desc.backend);
	graph->SetTaskService(m_desc.tasks);
	graph->RegisterExtension(std::make_unique<HostUploadExtension>(graph->GetUploadService()), "org.host.io");
	for (const auto& entry : m_extensions)
		if (auto extension = entry.factory()) graph->RegisterExtension(std::move(extension), entry.id);
	runtime::ScopedActiveGraphServices services(graph->GetUploadService(), graph->GetDescriptorService());
	graph->PrepareExtensionsForBuild();
	graph->CompileStructural();
	graph->SetPersistentSegment(kUploadPassName, RenderGraph::PersistentSegmentKind::Pre);
	graph->SetPersistentEpochOrder(m_desc.epochOrder);
	graph->SetPersistentClosedExecutions(m_desc.closedExecutions);
	graph->SetPersistentExecutionEnabled(true);
	graph->SetExternalQueueBoundary(m_desc.queueBoundary);
	graph->Setup();
	m_graph = std::move(graph);
	m_rebuildRequested = false;
}

void PersistentGraphHost::ExecuteFrame(const IHostExecutionData* hostData, const FrameCallback& beforePrepare, uint32_t epoch) {
	if (m_async) throw std::logic_error("ExecuteFrame while async epochs run: the host's thread owns the graph (use SubmitEpoch)");
	m_lastTimings = {};
	auto lap = [last = std::chrono::steady_clock::now()](double& a_into) mutable {
		const auto now = std::chrono::steady_clock::now();
		a_into += std::chrono::duration<double, std::micro>(now - last).count();
		last = now;
	};
	if (m_rebuildRequested || !m_graph) Build();
	lap(m_lastTimings.buildUs);
	runtime::ScopedActiveGraphServices services(m_graph->GetUploadService(), m_graph->GetDescriptorService());
	const auto slot = static_cast<uint32_t>(m_frameNumber % (std::max)(m_desc.framesInFlight, 1u));
	// The slot's previous frame must be done on the GPU before the upload pages it used are recycled
	// (and before this frame records uploads into the slot).
	if (auto& completion = m_slotCompletions[slot]; completion.pending) {
		for (const auto& [timeline, value] : completion.points) (void)timeline->HostWait(value);
		completion.pending = false;
		// Nothing else completes a frame's statistics on this path: without it the pass timestamps were
		// written and resolved every frame and never read, and their pending resolves only grew.
		if (auto* stats = m_graph->GetStatisticsService()) {
			rhi::Queue queue = m_desc.device.GetQueue(rhi::QueueKind::Graphics);
			stats->OnFrameComplete(slot, queue);
			if (m_completedFrame) m_completedFrame(completion.frameNumber, *stats);
		}
	}
	lap(m_lastTimings.waitUs);
	if (auto* uploads = m_graph->GetUploadService()) uploads->ProcessDeferredReleases(static_cast<uint8_t>(slot));
	DescriptorHeapManager::GetInstance().ProcessDeferredReleases(static_cast<uint8_t>(slot));
	lap(m_lastTimings.releaseUs);
	m_graph->SetPersistentEpoch(epoch);
	if (beforePrepare) beforePrepare(*m_graph);
	lap(m_lastTimings.beforePrepareUs);
	UpdateExecutionContext update{};
	update.frameIndex = slot;
	update.preparationSlot = slot;
	update.frameFenceValue = m_frameNumber + 1;
	update.hostData = hostData;
	m_graph->Update(update, m_desc.device);
	lap(m_lastTimings.updateUs);
	PassExecutionContext execute{};
	execute.device = m_desc.device;
	execute.frameIndex = slot;
	execute.frameFenceValue = m_frameNumber + 1;
	execute.hostData = hostData;
	m_graph->Execute(execute);
	lap(m_lastTimings.executeUs);
	m_lastTimings.execute = m_graph->LastPersistentExecuteTimings();
	// Execute submitted the frame: every batch signalled its queue's timeline, so the value each queue
	// has reached is the frame's completion on it.
	{
		auto& completion = m_slotCompletions[slot];
		auto& queues = m_graph->GetQueueRegistry();
		completion.points.clear();
		for (size_t index = 0; index < queues.SlotCount(); ++index) {
			const auto queueSlot = static_cast<QueueSlotIndex>(static_cast<uint8_t>(index));
			if (const auto next = queues.GetCurrentFenceValue(queueSlot); next > 1)
				completion.points.emplace_back(&queues.GetFence(queueSlot), next - 1);
		}
		completion.frameNumber = m_frameNumber;
		completion.pending = true;
	}
	lap(m_lastTimings.signalUs);
	m_lastHostFrame = m_frameNumber;
	++m_frameNumber;
}

// ---- async epochs ----

struct PersistentGraphHost::Async {
	using Ticket = RenderGraph::PersistentTicket;
	struct Message {
		enum class Kind : uint8_t { Submitted, Discard, PrepareNow, Stop } kind = Kind::Stop;
		std::shared_ptr<Ticket> ticket;
		RenderGraph::PersistentTicketSubmission submission;
		uint32_t epochIndex = 0;
		int32_t slot = -1;
		std::shared_ptr<void> keepAlive;          // the uploads' staging, until the slot completes
		std::pair<uint32_t, uint64_t> uploadSignal{UINT32_MAX, 0};
	};

	std::vector<uint32_t> epochs;
	// One ticket cell per epoch: stored by the host's thread, taken by the render thread.
	std::unique_ptr<std::atomic<std::shared_ptr<Ticket>>[]> cells;
	// Render thread -> host thread, single producer / single consumer, in order.
	static constexpr uint64_t kMailbox = 64;
	std::array<Message, kMailbox> mailbox;
	std::atomic<uint64_t> written{0}, read{0};
	std::atomic<uint32_t> wake{0};
	std::atomic<bool> failed{false};
	std::string error;  // written before failed is set
	std::thread thread;

	// Host thread only.
	struct Slot {
		std::vector<std::pair<rhi::Timeline*, uint64_t>> points;  // its last submission's completion
		uint64_t hostFrame = 0;
		bool pending = false;   // submitted work not yet waited for
		bool reserved = false;  // held by a ticket that has not been submitted
		std::shared_ptr<void> keepAlive;
	};
	std::vector<Slot> slots;
	std::vector<uint8_t> needsTicket;
	uint64_t nextSlot = 0;
	uint64_t hostFrame = 0;

	// Render thread only.
	std::vector<uint64_t> queueValues;  // the last value assigned per queue slot
	// Per frame slot. Reset and begun by the host's thread when it prepares the slot's ticket (after the slot's
	// wait: the slot's last upload list is done), then recorded, ended and submitted by the render thread; the
	// ticket cell orders the two.
	std::vector<CommandListPair> uploadLists;
	uint32_t graphicsSlot = 0;
	rhi::Queue graphicsQueue;
	rhi::TimelineHandle graphicsFence{};

	uint32_t IndexOf(uint32_t epoch) const {
		for (uint32_t i = 0; i < epochs.size(); ++i) if (epochs[i] == epoch) return i;
		throw std::invalid_argument("SubmitEpoch: epoch " + std::to_string(epoch) + " has no ticket");
	}
	void Post(Message message) {
		const auto index = written.load(std::memory_order_relaxed);
		// The host thread drains in order; with one ticket per epoch outstanding this never fills.
		while (index - read.load(std::memory_order_acquire) >= kMailbox) {
			if (failed.load(std::memory_order_acquire)) return;
			std::this_thread::yield();
		}
		mailbox[index % kMailbox] = std::move(message);
		written.store(index + 1, std::memory_order_release);
		wake.fetch_add(1, std::memory_order_release);
		wake.notify_one();
	}
	void ThrowIfFailed() const {
		if (failed.load(std::memory_order_acquire)) throw std::runtime_error("Async epochs failed: " + error);
	}
	std::shared_ptr<Ticket> WaitTicket(uint32_t index) {
		for (;;) {
			if (auto ticket = cells[index].exchange(nullptr, std::memory_order_acq_rel)) return ticket;
			ThrowIfFailed();
			cells[index].wait(nullptr, std::memory_order_acquire);
		}
	}
};

void PersistentGraphHost::SetAsyncEpochs(std::vector<uint32_t> epochs) {
	StopAsync();
	m_asyncEpochs = std::move(epochs);
	if (m_asyncEpochs.empty()) return;
	if (!m_desc.closedExecutions) throw std::logic_error("Async epochs require closed executions");
	if (m_rebuildRequested || !m_graph) Build();
	StartAsync();
}

void PersistentGraphHost::StartAsync() {
	if (m_async || m_asyncEpochs.empty() || !m_graph) return;
	auto async = std::make_unique<Async>();
	async->epochs = m_asyncEpochs;
	async->cells = std::make_unique<std::atomic<std::shared_ptr<Async::Ticket>>[]>(async->epochs.size());
	async->needsTicket.assign(async->epochs.size(), 1);
	const uint32_t slotCount = (std::max)(m_desc.framesInFlight, 1u);
	async->slots.resize(slotCount);
	// What the synchronous frames left: their slots' completions, and the values the queues have reached.
	for (uint32_t slot = 0; slot < slotCount && slot < m_slotCompletions.size(); ++slot) {
		auto& completion = m_slotCompletions[slot];
		async->slots[slot].points = completion.points;
		async->slots[slot].hostFrame = completion.frameNumber;
		async->slots[slot].pending = completion.pending;
		completion = {};
	}
	async->nextSlot = m_frameNumber;
	async->hostFrame = m_frameNumber;
	auto& queues = m_graph->GetQueueRegistry();
	for (size_t index = 0; index < queues.SlotCount(); ++index) {
		const auto next = queues.GetCurrentFenceValue(static_cast<QueueSlotIndex>(static_cast<uint8_t>(index)));
		async->queueValues.push_back(next ? next - 1 : 0);
	}
	const auto graphics = queues.FindGraphicsSlot();
	async->graphicsSlot = static_cast<uint32_t>(ToUnderlying(graphics));
	async->graphicsQueue = queues.GetQueue(graphics);
	async->graphicsFence = queues.GetFence(graphics).GetHandle();
	for (uint32_t slot = 0; slot < slotCount; ++slot) {
		CommandListPair pair;
		if (!rhi::IsOk(m_desc.device.CreateCommandAllocator(rhi::QueueKind::Graphics, pair.allocator))
			|| !rhi::IsOk(m_desc.device.CreateCommandList(rhi::QueueKind::Graphics, pair.allocator.Get(), pair.list)))
			throw std::runtime_error("Async epochs could not create their upload command lists");
		pair.list->SetName(("ORG ticket uploads #" + std::to_string(slot)).c_str());
		pair.list->End();
		async->uploadLists.push_back(std::move(pair));
	}
	// Staged uploads go straight into each submission's upload list (RenderGraph::RecordPendingUploads).
	if (auto* uploads = m_graph->GetUploadService()) uploads->SetStagedUploadsRecordedDirectly(true);
	auto* state = async.get();
	auto* graph = m_graph.get();
	state->thread = std::thread([this, state, graph] {
		try {
			for (;;) {
				const auto seen = state->wake.load(std::memory_order_acquire);
				bool stop = false;
				// Messages, in the order the render thread posted them.
				for (auto read = state->read.load(std::memory_order_relaxed); read != state->written.load(std::memory_order_acquire); ++read) {
					auto message = std::move(state->mailbox[read % Async::kMailbox]);
					state->read.store(read + 1, std::memory_order_release);
					switch (message.kind) {
					case Async::Message::Kind::Submitted: {
						auto& slot = state->slots[RenderGraph::PersistentTicketSlot(*message.ticket)];
						auto completion = graph->CompletePersistentTicket(*message.ticket, message.submission);
						if (message.uploadSignal.first != UINT32_MAX) {
							auto found = std::ranges::find(completion, message.uploadSignal.first, &std::pair<uint32_t, uint64_t>::first);
							if (found == completion.end()) completion.push_back(message.uploadSignal);
							else found->second = (std::max)(found->second, message.uploadSignal.second);
						}
						slot.points.clear();
						for (const auto& [queue, value] : completion)
							slot.points.emplace_back(&graph->GetQueueRegistry().GetFence(static_cast<QueueSlotIndex>(static_cast<uint8_t>(queue))), value);
						slot.pending = !slot.points.empty();
						slot.reserved = false;
						slot.hostFrame = RenderGraph::PersistentTicketHostFrame(*message.ticket);
						slot.keepAlive = std::move(message.keepAlive);
						state->needsTicket[message.epochIndex] = 1;
						break;
					}
					case Async::Message::Kind::Discard:
						state->slots[RenderGraph::PersistentTicketSlot(*message.ticket)].reserved = false;
						(void)graph->CompletePersistentTicket(*message.ticket, {});
						break;
					case Async::Message::Kind::PrepareNow:
						PrepareTicket(*state, message.epochIndex, message.slot);
						break;
					case Async::Message::Kind::Stop:
						stop = true;
						break;
					}
				}
				if (stop) break;
				graph->RetirePersistentExecutions();
				// The next ticket of every epoch whose last one was submitted: one frame ahead.
				bool prepared = false;
				for (uint32_t index = 0; index < state->epochs.size(); ++index) {
					if (!state->needsTicket[index] || state->cells[index].load(std::memory_order_acquire)) continue;
					PrepareTicket(*state, index, -1);
					prepared = true;
				}
				if (!prepared && state->read.load(std::memory_order_relaxed) == state->written.load(std::memory_order_acquire))
					state->wake.wait(seen, std::memory_order_acquire);
			}
		} catch (const std::exception& e) {
			state->error = e.what();
			state->failed.store(true, std::memory_order_release);
		} catch (...) {
			state->error = "unknown exception";
			state->failed.store(true, std::memory_order_release);
		}
		// Wake a render thread waiting for a ticket, so it sees the failure.
		for (uint32_t index = 0; index < state->epochs.size(); ++index) state->cells[index].notify_all();
	});
	m_async = std::move(async);
}

void PersistentGraphHost::PrepareTicket(Async& state, uint32_t epochIndex, int32_t requestedSlot) {
	const uint32_t slotCount = static_cast<uint32_t>(state.slots.size());
	uint32_t slot = 0;
	if (requestedSlot >= 0) {
		slot = static_cast<uint32_t>(requestedSlot);
	} else {
		// The next slot no unsubmitted ticket holds.
		for (uint32_t attempt = 0;; ++attempt) {
			if (attempt == slotCount) throw std::logic_error("Every frame slot is held by an unsubmitted ticket");
			slot = static_cast<uint32_t>(state.nextSlot++ % slotCount);
			if (!state.slots[slot].reserved) break;
		}
	}
	auto& entry = state.slots[slot];
	runtime::ScopedActiveGraphServices services(m_graph->GetUploadService(), m_graph->GetDescriptorService());
	// The slot's previous execution must be done before its command lists, statistics range and latch region
	// are reused: the wait happens here, on this thread, never on the render thread.
	if (entry.pending) {
		for (const auto& [timeline, value] : entry.points) (void)timeline->HostWait(value);
		entry.pending = false;
		if (auto* stats = m_graph->GetStatisticsService()) {
			rhi::Queue queue = m_desc.device.GetQueue(rhi::QueueKind::Graphics);
			stats->OnFrameComplete(slot, queue);
			if (m_completedFrame) m_completedFrame(entry.hostFrame, *stats);
		}
		entry.keepAlive.reset();
	}
	// The render thread records this ticket's uploads into an open list: resetting and beginning it is several
	// microseconds of driver work that belongs here.
	auto& uploads = state.uploadLists[slot];
	uploads.allocator->Recycle();
	uploads.list->Recycle(uploads.allocator.Get());
	DescriptorHeapManager::GetInstance().ProcessDeferredReleases(static_cast<uint8_t>(slot));
	entry.reserved = true;
	auto ticket = m_graph->PreparePersistentTicket(m_desc.device, state.epochs[epochIndex], static_cast<uint8_t>(slot), state.hostFrame++);
	state.needsTicket[epochIndex] = 0;
	state.cells[epochIndex].store(std::move(ticket), std::memory_order_release);
	state.cells[epochIndex].notify_all();
}

void PersistentGraphHost::StopAsync() {
	if (!m_async) return;
	auto async = std::move(m_async);
	Async::Message stop;
	stop.kind = Async::Message::Kind::Stop;
	async->Post(std::move(stop));
	if (async->thread.joinable()) async->thread.join();
	// Unsubmitted tickets are abandoned; submitted work keeps its slots' completions for the synchronous path.
	if (m_graph && !async->failed.load()) {
		for (uint32_t index = 0; index < async->epochs.size(); ++index)
			if (auto ticket = async->cells[index].exchange(nullptr)) (void)m_graph->CompletePersistentTicket(*ticket, {});
	}
	for (uint32_t slot = 0; slot < async->slots.size() && slot < m_slotCompletions.size(); ++slot) {
		auto& completion = m_slotCompletions[slot];
		completion.points = async->slots[slot].points;
		completion.frameNumber = async->slots[slot].hostFrame;
		completion.pending = async->slots[slot].pending;
	}
	m_frameNumber = (std::max)(m_frameNumber, async->hostFrame);
	if (m_graph)
		if (auto* uploads = m_graph->GetUploadService()) uploads->SetStagedUploadsRecordedDirectly(false);
	// The render thread's upload lists may still be executing: wait for the queue before they go.
	for (const auto& slot : async->slots)
		for (const auto& [timeline, value] : slot.points) (void)timeline->HostWait(value);
}

void PersistentGraphHost::SubmitEpoch(uint32_t epoch, const FrameCallback& beforeSubmit) {
	m_lastTimings = {};
	m_lastTimings.async = true;
	auto lap = [last = std::chrono::steady_clock::now()](double& a_into) mutable {
		const auto now = std::chrono::steady_clock::now();
		a_into += std::chrono::duration<double, std::micro>(now - last).count();
		last = now;
	};
	if (m_rebuildRequested || !m_graph) {
		StopAsync();
		Build();
		StartAsync();
	}
	lap(m_lastTimings.buildUs);
	if (!m_async) throw std::logic_error("SubmitEpoch requires async epochs");
	auto& async = *m_async;
	async.ThrowIfFailed();
	const auto index = async.IndexOf(epoch);
	auto ticket = async.cells[index].exchange(nullptr, std::memory_order_acq_rel);
	if (!ticket) {
		++m_asyncStats.waited;
		ticket = async.WaitTicket(index);
	}
	lap(m_lastTimings.ticketWaitUs);
	const uint32_t slot = RenderGraph::PersistentTicketSlot(*ticket);
	m_asyncSlot = slot;
	m_lastHostFrame = RenderGraph::PersistentTicketHostFrame(*ticket);
	runtime::ScopedActiveGraphServices services(m_graph->GetUploadService(), m_graph->GetDescriptorService());
	// The host's thread waited for this slot before preparing the ticket: its upload pages are free again.
	if (auto* uploads = m_graph->GetUploadService()) uploads->ProcessDeferredReleases(static_cast<uint8_t>(slot));
	lap(m_lastTimings.releaseUs);
	if (beforeSubmit) beforeSubmit(*m_graph);
	lap(m_lastTimings.beforePrepareUs);
	if (!RenderGraph::PersistentTicketCurrent(*ticket)) {
		// Prepared before beforeSubmit changed what a pass depends on: prepare it again, for the same slot (its
		// latch region was just written), and wait. The new one is prepared after the change, so it is current.
		++m_asyncStats.stale;
		Async::Message discard;
		discard.kind = Async::Message::Kind::Discard;
		discard.ticket = std::move(ticket);
		async.Post(std::move(discard));
		Async::Message prepare;
		prepare.kind = Async::Message::Kind::PrepareNow;
		prepare.epochIndex = index;
		prepare.slot = static_cast<int32_t>(slot);
		async.Post(std::move(prepare));
		ticket = async.WaitTicket(index);
	}
	lap(m_lastTimings.checkUs);
	// The uploads queued since the last submission, as plain copies ahead of the ticket (its first batch
	// starts with a full barrier, so they need none of their own).
	auto& list = async.uploadLists[slot];  // begun by the host's thread (PrepareTicket)
	std::shared_ptr<void> keepAlive;
	bool recorded = false;
	auto commands = list.list.Get();
	const bool supported = m_graph->RecordPendingUploads(m_desc.device, commands, static_cast<uint8_t>(slot), keepAlive, recorded);
	list.list->End();
	if (!supported) {
		Async::Message discard;
		discard.kind = Async::Message::Kind::Discard;
		discard.ticket = std::move(ticket);
		async.Post(std::move(discard));
		throw std::runtime_error("Async epochs cannot record these uploads (only pointer-targeted buffer copies)");
	}
	Async::Message submitted;
	submitted.kind = Async::Message::Kind::Submitted;
	submitted.epochIndex = index;
	if (recorded) {
		++m_asyncStats.uploads;
		const auto value = ++async.queueValues.at(async.graphicsSlot);
		const rhi::TimelinePoint signal{async.graphicsFence, value};
		rhi::CommandList lists[] = {commands};
		if (async.graphicsQueue.Submit({lists, 1}, {{}, {&signal, 1}}) != rhi::Result::Ok)
			throw std::runtime_error("Async epochs could not submit their uploads");
		submitted.uploadSignal = {async.graphicsSlot, value};
	}
	submitted.keepAlive = std::move(keepAlive);
	lap(m_lastTimings.uploadsUs);
	const bool ok = RenderGraph::SubmitPersistentTicket(*ticket, async.queueValues, submitted.submission);
	lap(m_lastTimings.submitUs);
	submitted.ticket = std::move(ticket);
	async.Post(std::move(submitted));
	++m_asyncStats.submitted;
	if (!ok) throw std::runtime_error("Async epoch submission failed");
}

} // namespace org
