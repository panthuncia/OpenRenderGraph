#include "OpenRenderGraph/PersistentGraphHost.h"

#include <BasicTelemetry/Tracy.h>
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

// Defined before the constructor: a throwing constructor destroys m_async, and clang instantiates
// ~unique_ptr<Async> there.
struct PersistentGraphHost::Async {
	using Ticket = RenderGraph::PersistentTicket;
	struct Message {
		enum class Kind : uint8_t { Submitted, Discard, OwnedPrepare, Stop } kind = Kind::Stop;
		std::shared_ptr<Ticket> ticket;
		RenderGraph::PersistentTicketSubmission submission;
		uint32_t epochIndex = 0;
		int32_t slot = -1;  // Discard: the slot to prepare the epoch's replacement ticket into, or -1 for none
		std::shared_ptr<void> keepAlive;          // the uploads' staging, until the slot completes
		std::shared_ptr<void> retired;            // Submitted: GPU objects that retired at this submission, destroyed on the host's thread
		std::shared_ptr<const void> resourceOwner;
		std::pair<uint32_t, uint64_t> uploadSignal{UINT32_MAX, 0};
		FrameCallback ownedPrepare;
	};

	std::vector<uint32_t> epochs;
	std::vector<uint8_t> revisionDriven;  // per epoch index: recorded ahead for revisions, its tickets admission-only
	// Per epoch index, its ring of frame slots: [index * ringSize, + ringSize), the next position. 0: one ring of every slot.
	uint32_t ringSize = 0;
	std::vector<uint64_t> ringNext;
	// Recording requests (RequestEpochRecording): any thread pushes, the host's thread takes them all (a Treiber stack, taken
	// whole and reversed into posting order).
	struct RecordingRequest {
		uint32_t epochIndex = 0;
		std::shared_ptr<const IHostExecutionData> hostData;
		PersistentGraphHost::EpochRecordingCallback done;
		std::shared_ptr<PersistentGraphHost::EpochRecording> result;
		std::vector<uint8_t> recorded;  // by ring position
		RecordingRequest* next = nullptr;
	};
	std::atomic<RecordingRequest*> requests{nullptr};
	std::deque<std::unique_ptr<RecordingRequest>> pendingRequests;  // host thread
	// Host thread, per epoch index: the revision recordings made (admitted by each ticket while their callers hold them), and the
	// ticket prepared and not yet submitted or discarded, which a recording finished meanwhile is added to before its callback.
	std::vector<std::vector<std::weak_ptr<const PersistentGraphHost::EpochRecording>>> liveRecordings;
	std::vector<std::shared_ptr<RenderGraph::PersistentTicket>> outstanding;
	std::atomic<uint64_t> revisionSlotsRecorded{0}, revisionRecordings{0};
	~Async() {
		for (auto* request = requests.exchange(nullptr); request;) {
			auto* next = request->next;
			delete request;
			request = next;
		}
	}
	// One ticket cell per epoch: stored by the host's thread, taken by the render thread.
	std::unique_ptr<std::atomic<std::shared_ptr<Ticket>>[]> cells;
	// Render thread -> host thread, single producer / single consumer, in order.
	static constexpr uint64_t kMailbox = 64;
	std::array<Message, kMailbox> mailbox;
	std::atomic<uint64_t> written{0}, read{0};
	// Credits held by render-thread frame reservations. Owned preparation may not
	// consume these cells, so a reserved submission always has a control-message
	// slot even when the worker is busy.
	std::shared_ptr<std::atomic<uint32_t>> controlCredits = std::make_shared<std::atomic<uint32_t>>(0);
	std::unique_ptr<rhi::CompletionWait> completionWake;
	// Whether the host's thread, if it sleeps, sleeps on GPU completions as well as on completionWake: then a submission's
	// message needs no wake of its own (vkSignalSemaphore, several microseconds on the render thread), because the next GPU
	// completion wakes the thread and it reads the mailbox then. Written by the host's thread before its last look at the
	// mailbox, read by a poster after its write: both sequentially consistent, so one of them sees the other.
	std::atomic<bool> wakesOnGpu{false};
	uint64_t wakesSkipped = 0;  // render thread
	// Ticket publication and terminal failure both advance this counter. A
	// notify on an unchanged null shared_ptr cannot release atomic::wait(nullptr).
	std::atomic<uint32_t> ticketWake{0};
	std::atomic<bool> failed{false};
	std::string error;  // written before failed is set
	std::thread thread;
	// Backing changes from the submitting thread (BackingMutation) against this thread's ticket preparation, which captures
	// every slot's backing (BackedResource: capture and backing mutation never overlap). A Dekker pair, both sides
	// sequentially consistent: the submitting thread raises mutating, then waits for preparing to fall; the host's thread
	// raises preparing, then backs off while mutating is up. So a preparation sees every backing whole, before or after.
	std::atomic<uint32_t> preparing{0}, mutating{0};
	// Advanced by every BackingMutation. A ticket prepared under an older value may hold a backing that has changed since
	// (the render thread then writes the new one): it is not current (Current).
	std::atomic<uint64_t> backingVersion{0};
	std::atomic<uint64_t> preparationWaits{0};  // AsyncStats::preparationWaits, from the host's thread

	// Host thread only.
	struct Slot {
		std::vector<std::pair<rhi::Timeline*, uint64_t>> points;  // its last submission's completion
		uint64_t hostFrame = 0;
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

	// Render thread only.
	std::vector<uint64_t> queueValues;  // the last value assigned per queue slot
	// Per frame slot. Reset and begun by the host's thread when it prepares the slot's ticket (after the slot's
	// wait: the slot's last upload list is done), then recorded, ended and submitted by the render thread; the
	// ticket cell orders the two.
	std::vector<CommandListPair> uploadLists;
	std::vector<rhi::CommandList> recordedLists;  // SubmitTicket's scratch: the producers' recorded upload lists, then the slot's
	uint32_t graphicsSlot = 0;
	rhi::Queue graphicsQueue;
	rhi::TimelineHandle graphicsFence{};

	uint32_t IndexOf(uint32_t epoch) const {
		for (uint32_t i = 0; i < epochs.size(); ++i) if (epochs[i] == epoch) return i;
		throw std::invalid_argument("SubmitEpoch: epoch " + std::to_string(epoch) + " has no ticket");
	}
	// lazyWake: a submission's completion, which the host's thread can take at its next GPU wake-up (wakesOnGpu).
	bool TryPost(Message& message, bool spendsReservedCredit = false, bool lazyWake = false) {
		const auto index = written.load(std::memory_order_relaxed);
		const auto credits = controlCredits->load(std::memory_order_relaxed);
		if (spendsReservedCredit && !credits) return false;
		const auto protectedCredits = credits - static_cast<uint32_t>(spendsReservedCredit);
		if (failed.load(std::memory_order_acquire)
			|| index - read.load(std::memory_order_acquire) >= kMailbox - protectedCredits)
			return false;
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
		// Legacy SubmitEpoch still blocks for admission; published callers use
		// TryPostOwnedPreparation and never take this path on a full mailbox.
		while (!TryPost(message, false, lazyWake)) {
			if (failed.load(std::memory_order_acquire)) return false;
			std::this_thread::yield();
		}
		return true;
	}
	// Host thread: a preparation, kept clear of backing mutations (preparing, mutating). It may change backings itself.
	struct Preparation {
		explicit Preparation(Async& a_async) : async(a_async) {
			for (;;) {
				async.preparing.store(1, std::memory_order_seq_cst);
				if (!async.mutating.load(std::memory_order_seq_cst)) break;
				Leave();
				async.preparationWaits.fetch_add(1, std::memory_order_relaxed);
				BT_ZONE_SCOPE("ORG.Host.WaitBackingMutation.Worker");
				async.mutating.wait(1, std::memory_order_seq_cst);
			}
			BufferBase::EnterBackingMutation();
		}
		~Preparation() {
			BufferBase::LeaveBackingMutation();
			Leave();
		}
		void Leave() {
			async.preparing.store(0, std::memory_order_seq_cst);
			async.preparing.notify_all();
		}
		Async& async;
	};
	// Every pass it prepared would prepare the same invocation, and no backing has changed since.
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

PersistentGraphHost::PersistentGraphHost(Desc desc) : m_desc(std::move(desc)) {
	if (!m_desc.device) throw std::invalid_argument("PersistentGraphHost requires a device");
	if (m_desc.backend == rhi::Backend::Null) throw std::invalid_argument("PersistentGraphHost requires a device backend");
	if (!m_desc.tasks) m_desc.tasks = std::make_shared<runtime::ThreadPoolTaskService>();
	// The frame slots: framesInFlight frames of every epoch in the order, so that adding an epoch keeps the CPU's lead over
	// the GPU the same rather than shortening it (each slot's next execution waits for its last one).
	const uint32_t epochs = (std::max)(static_cast<uint32_t>(m_desc.epochOrder.size()), 1u);
	m_frameSlots = (std::max)(m_desc.framesInFlight, 1u) * epochs;
	if (m_frameSlots > 255) throw std::invalid_argument("PersistentGraphHost: more than 255 frame slots (frames in flight times epochs)");
	// The host's slot count is the runtime's frames-in-flight: upload pages, statistics query ranges, deletion
	// and admission capacity all size themselves from the runtime settings, so they follow it.
	{
		auto settings = runtime::GetOpenRenderGraphSettings();
		settings.numFramesInFlight = static_cast<uint8_t>(m_frameSlots);
		runtime::SetOpenRenderGraphSettings(settings);
	}
	runtime::InitializeRuntimeDevice(m_desc.device);
	m_slotCompletions.assign(m_frameSlots, {});
}

PersistentGraphHost::~PersistentGraphHost() {
	StopAsync();
	DestroyGraph();
	runtime::ShutdownRuntimeDevice();
	m_uncertainExecutionOwners.clear();
	m_uncertainUploadOwners.clear();
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

runtime::IDescriptorService* PersistentGraphHost::Descriptors() noexcept {
	return m_graph ? m_graph->GetDescriptorService() : nullptr;
}

std::shared_ptr<runtime::IUploadService> PersistentGraphHost::RetainUploads() noexcept {
	return m_graph ? m_graph->RetainUploadService() : nullptr;
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

void PersistentGraphHost::SetGpuPassRangeCallbacks(RenderGraph::GpuPassRangeBegin begin, RenderGraph::GpuPassRangeEnd end) {
	m_gpuPassRangeBegin = std::move(begin);
	m_gpuPassRangeEnd = std::move(end);
	if (m_graph) m_graph->SetGpuPassRangeCallbacks(m_gpuPassRangeBegin, m_gpuPassRangeEnd);
}

void PersistentGraphHost::Build() {
	BT_ZONE_SCOPE("ORG.Host.Build");
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
	graph->SetPersistentRecordingReuse(m_desc.reuseRecordings && m_desc.closedExecutions);
	graph->SetPersistentExecutionEnabled(true);
	graph->SetExternalQueueBoundary(m_desc.queueBoundary);
	graph->SetFrameWaitTimeline(m_frameWaitTimeline);
	graph->Setup();
	m_graph = std::move(graph);
	m_graph->SetGpuPassRangeCallbacks(m_gpuPassRangeBegin, m_gpuPassRangeEnd);
	m_rebuildRequested = false;
	++m_buildGeneration;
}

void PersistentGraphHost::SetFrameWaitTimeline(std::shared_ptr<rhi::TimelinePtr> timeline) {
	m_frameWaitTimeline = std::move(timeline);
	// Every packet binds its timelines when it is prepared: the graph's present ones cannot wait on it.
	if (m_graph) m_rebuildRequested = true;
}

PersistentGraphHost::GpuPoint PersistentGraphHost::SubmittedPoint() const {
	GpuPoint point;
	if (!m_graph) return point;
	const auto& queues = m_graph->GetQueueRegistry();
	for (size_t index = 0; index < queues.SlotCount(); ++index) {
		const auto slot = static_cast<QueueSlotIndex>(static_cast<uint8_t>(index));
		// Async epochs: the values the submitting thread assigned; synchronous frames: the registry's.
		uint64_t value = 0;
		if (m_async) value = index < m_async->queueValues.size() ? m_async->queueValues[index] : 0;
		else if (const auto next = queues.GetCurrentFenceValue(slot); next > 1) value = next - 1;
		if (value) point.points.emplace_back(queues.GetFenceOwner(slot), value);
	}
	return point;
}

bool PersistentGraphHost::BuildIfRequested() {
	if (m_graph && !m_rebuildRequested) return false;
	const bool async = static_cast<bool>(m_async);
	StopAsync();
	Build();
	if (async) StartAsync();
	return true;
}

void PersistentGraphHost::ExecuteFrame(const IHostExecutionData* hostData, const FrameCallback& beforePrepare,
	uint32_t epoch, std::shared_ptr<const void> resourceOwner) {
	BT_ZONE_SCOPE("ORG.Host.ExecuteFrame");
	if (m_async) throw std::logic_error("ExecuteFrame while async epochs run: the host's thread owns the graph (use SubmitEpoch)");
	m_lastTimings = {};
	auto lap = [last = std::chrono::steady_clock::now()](double& a_into) mutable {
		const auto now = std::chrono::steady_clock::now();
		a_into += std::chrono::duration<double, std::micro>(now - last).count();
		last = now;
	};
	if (BuildDue()) Build();
	lap(m_lastTimings.buildUs);
	runtime::ScopedActiveGraphServices services(m_graph->GetUploadService(), m_graph->GetDescriptorService());
	const auto slot = static_cast<uint32_t>(m_frameNumber % m_frameSlots);
	// The slot's previous frame must be done on the GPU before the upload pages it used are recycled
	// (and before this frame records uploads into the slot).
	if (auto& completion = m_slotCompletions[slot]; completion.pending) {
		{
			BT_ZONE_SCOPE("ORG.Host.WaitSlotCompletion");
			for (const auto& [timeline, value] : completion.points) (void)timeline->HostWait(value);
		}
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
	m_graph->SetFrameWaitValue(m_frameWaitValue);
	if (beforePrepare) {
		BT_ZONE_SCOPE("ORG.Host.CommitInputs");
		beforePrepare(*m_graph);
	}
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
	// Reserve CPU ownership before submission. Execute may fail after accepting a
	// prefix of its batches, in which case recovery retains this owner.
	const bool heldResources = static_cast<bool>(resourceOwner);
	if (heldResources) m_uncertainExecutionOwners.push_back(std::move(resourceOwner));
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
		if (heldResources) {
			const auto& owner = m_uncertainExecutionOwners.back();
			if (owner && !completion.points.empty()) {
				std::vector<DescriptorHeapManager::QueueFenceSnapshotPoint> points;
				points.reserve(completion.points.size());
				for (const auto& [timeline, value] : completion.points)
					points.push_back({*timeline, value});
				DescriptorHeapManager::GetInstance().RetireExecutionLease(owner, std::move(points));
			}
			m_uncertainExecutionOwners.pop_back();
		}
	}
	lap(m_lastTimings.signalUs);
	m_lastHostFrame = m_frameNumber;
	++m_frameNumber;
}

// ---- async epochs ----


void PersistentGraphHost::SetAsyncEpochs(std::vector<uint32_t> epochs, std::vector<uint32_t> revisionEpochs) {
	StopAsync();
	m_asyncEpochs = std::move(epochs);
	for (const auto epoch : revisionEpochs)
		if (std::ranges::find(m_asyncEpochs, epoch) == m_asyncEpochs.end())
			throw std::invalid_argument("A revision-driven epoch must be an async epoch");
	if (!revisionEpochs.empty() && !m_desc.reuseRecordings)
		throw std::logic_error("Revision-driven epochs need Desc::reuseRecordings (replayable recordings)");
	m_revisionEpochs = std::move(revisionEpochs);
	if (m_asyncEpochs.empty()) return;
	if (!m_desc.closedExecutions) throw std::logic_error("Async epochs require closed executions");
	if (BuildDue()) Build();
	StartAsync();
}

void PersistentGraphHost::StartAsync() {
	if (m_async || m_asyncEpochs.empty() || !m_graph) return;
	auto async = std::make_unique<Async>();
	if (m_desc.device.CreateCompletionWait(async->completionWake) != rhi::Result::Ok)
		throw std::runtime_error("Async epochs require wakeable GPU completion waits");
	async->epochs = m_asyncEpochs;
	async->revisionDriven.assign(async->epochs.size(), 0);
	for (uint32_t index = 0; index < async->epochs.size(); ++index)
		async->revisionDriven[index] = std::ranges::find(m_revisionEpochs, async->epochs[index]) != m_revisionEpochs.end();
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
				const auto seen = state->completionWake->WakeVersion();
				bool stop = false;
				// Messages, in the order the render thread posted them.
				for (auto read = state->read.load(std::memory_order_relaxed); read != state->written.load(std::memory_order_acquire); ++read) {
					auto message = std::move(state->mailbox[read % Async::kMailbox]);
					state->read.store(read + 1, std::memory_order_release);
					switch (message.kind) {
					case Async::Message::Kind::Submitted: {
						auto& slot = state->slots[RenderGraph::PersistentTicketSlot(*message.ticket)];
						state->uncertainUploadOwner = std::move(message.keepAlive);
						state->uncertainResourceOwner = std::move(message.resourceOwner);
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
						if (state->outstanding[message.epochIndex] == message.ticket) state->outstanding[message.epochIndex].reset();
							slot.keepAlive = std::move(state->uncertainUploadOwner);
							slot.resourceOwner = std::move(state->uncertainResourceOwner);
						state->needsTicket[message.epochIndex] = 1;
						if (message.retired) {
							BT_ZONE_SCOPE("ORG.Host.DestroyRetired.Worker");
							message.retired.reset();
						}
						break;
					}
					case Async::Message::Kind::Discard:
						state->slots[RenderGraph::PersistentTicketSlot(*message.ticket)].reserved = false;
						if (state->outstanding[message.epochIndex] == message.ticket) state->outstanding[message.epochIndex].reset();
						(void)graph->CompletePersistentTicket(*message.ticket, {});
						// A discard naming a slot is a stale ticket's reprepare: its replacement is prepared here, in the same
						// step, into the slot the render thread is recording. Marking the epoch as needing a ticket instead
						// would let the proactive pass below prepare one into a slot of its own choosing, which the render
						// thread would take while it records into the other slot's upload list.
						if (message.slot >= 0)
							PrepareTicket(*state, message.epochIndex, message.slot);
						else
							state->needsTicket[message.epochIndex] = 1;
						break;
					case Async::Message::Kind::OwnedPrepare: {
						runtime::ScopedActiveGraphServices services(graph->GetUploadService(), graph->GetDescriptorService());
						Async::Preparation preparation(*state);
						message.ownedPrepare(*graph);
						break;
					}
					case Async::Message::Kind::Stop:
						stop = true;
						break;
					}
				}
				if (stop) break;
				// Recording requests: taken in posting order, each recording the slots of its ring that are free.
				if (auto* head = state->requests.exchange(nullptr, std::memory_order_acquire)) {
					std::vector<Async::RecordingRequest*> taken;
					for (; head; head = head->next) taken.push_back(head);
					for (auto it = taken.rbegin(); it != taken.rend(); ++it) {
						(*it)->next = nullptr;
						state->pendingRequests.emplace_back(*it);
					}
				}
				graph->RetirePersistentExecutions();
				// A submitted upload batch owns its staging until the GPU has actually
				// signalled every queue that used the slot. Retire it on this worker even
				// if no later frame ever requests this slot again.
				for (auto& slot : state->slots) {
					if (!slot.keepAlive && !slot.resourceOwner) continue;
					bool complete = true;
					for (const auto& [timeline, value] : slot.points) {
						const auto reached = timeline->GetCompletedValue();
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
					if (!state->needsTicket[index] || state->cells[index].load(std::memory_order_acquire)) continue;
					PrepareTicket(*state, index, -1);
					prepared = true;
				}
				// Then the recording requests: the frames' tickets come first, so recording ahead never delays a submission.
				const bool requestsAdvanced = AdvanceRecordingRequests(*state);
				if (!prepared && !requestsAdvanced && !state->requests.load(std::memory_order_acquire)
					&& state->read.load(std::memory_order_relaxed) == state->written.load(std::memory_order_acquire)) {
					std::array<rhi::TimelinePoint, rhi::CompletionWait::MaxTimelines> heads{};
					uint32_t count = 0;
					bool retireAgain = false;
					for (const auto& slot : state->slots) if (slot.pending) {
						bool incomplete = false;
						for (const auto& [timeline, value] : slot.points) {
							const auto reached = timeline->GetCompletedValue();
							if (reached == UINT64_MAX) throw std::runtime_error("GPU completion timeline failed");
							if (reached >= value) continue;
							incomplete = true;
							const auto handle = timeline->GetHandle();
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
			state->failed.store(true, std::memory_order_release);
		} catch (...) {
			state->error = "unknown exception";
			state->failed.store(true, std::memory_order_release);
		}
		// Wake a render thread waiting for a ticket, so it sees the failure.
		state->ticketWake.fetch_add(1, std::memory_order_release);
		state->ticketWake.notify_all();
	});
	// From here the host's thread captures backings while the submitting thread runs: their changes need BackingMutation.
	BufferBase::RegisterConcurrentBackingCapture();
	m_async = std::move(async);
	++m_ticketGeneration;
}

std::shared_ptr<runtime::ResourceCleanupQueue> PersistentGraphHost::ResourceCleanup() const {
	return DescriptorHeapManager::GetInstance().GetResourceCleanupQueue();
}

void PersistentGraphHost::WaitSlot(Async& state, uint32_t slot) {
	// The slot's previous execution must be done before its command lists, statistics range and latch region are reused, or
	// before it is recorded for again: the wait happens here, on this thread, never on the render thread.
	auto& entry = state.slots[slot];
	if (!entry.pending) return;
	{
		BT_ZONE_SCOPE("ORG.Host.WaitSlotCompletion.Worker");
		for (const auto& [timeline, value] : entry.points) (void)timeline->HostWait(value);
	}
	entry.pending = false;
	if (auto* stats = m_graph->GetStatisticsService()) {
		rhi::Queue queue = m_desc.device.GetQueue(rhi::QueueKind::Graphics);
		stats->OnFrameComplete(slot, queue);
		if (m_completedFrame) m_completedFrame(entry.hostFrame, *stats);
	}
	entry.keepAlive.reset();
	entry.resourceOwner.reset();
}

bool PersistentGraphHost::ReleaseEpochTicket(uint32_t epoch) {
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

bool PersistentGraphHost::RequestEpochRecording(uint32_t epoch, std::shared_ptr<const IHostExecutionData> hostData, EpochRecordingCallback done) {
	auto* async = m_async.get();
	if (!async || !done || async->failed.load(std::memory_order_acquire)) return false;
	const auto found = std::ranges::find(async->epochs, epoch);
	if (found == async->epochs.end()) return false;
	const auto index = static_cast<uint32_t>(found - async->epochs.begin());
	if (!async->ringSize) return false;
	auto request = std::make_unique<Async::RecordingRequest>();
	request->epochIndex = index;
	request->hostData = std::move(hostData);
	request->done = std::move(done);
	auto* node = request.release();
	auto* head = async->requests.load(std::memory_order_relaxed);
	do node->next = head;
	while (!async->requests.compare_exchange_weak(head, node, std::memory_order_release, std::memory_order_relaxed));
	(void)async->completionWake->Notify();
	return true;
}

bool PersistentGraphHost::AdvanceRecordingRequests(Async& state) {
	bool advanced = false;
	for (auto it = state.pendingRequests.begin(); it != state.pendingRequests.end();) {
		auto& request = **it;
		std::exception_ptr error;
		try {
			if (!request.result) {
				request.result = std::make_shared<EpochRecording>();
				request.result->epoch = state.epochs[request.epochIndex];
				request.result->firstSlot = request.epochIndex * state.ringSize;
				request.result->slots.resize(state.ringSize);
				request.result->revision = request.hostData;
				request.result->buildGeneration = m_buildGeneration;
				request.recorded.assign(state.ringSize, 0);
			}
			for (uint32_t position = 0; position < state.ringSize; ++position) {
				const uint32_t slot = request.result->firstSlot + position;
				// A slot its epoch's unsubmitted ticket holds is recorded too: a revision-driven ticket holds the admission alone (its
				// preparation recorded nothing, and already waited out the slot's last work), and it cannot be submitted without this.
				if (request.recorded[position]) continue;
				// A live epoch's: a slot its ticket holds is recorded once the ticket is submitted, and a slot whose work is in flight once
				// that completes (the loop wakes at completions), so its frames never wait for this.
				if (!state.revisionDriven[request.epochIndex]) {
					const auto& entry = state.slots[slot];
					if (entry.reserved) continue;
					if (entry.pending && std::ranges::any_of(entry.points, [](const auto& a_point) {
							const auto reached = a_point.first->GetCompletedValue();
							return reached == UINT64_MAX || reached < a_point.second;
						}))
						continue;
				}
				BT_ZONE_SCOPE("ORG.Host.RecordForRevision");
				runtime::ScopedActiveGraphServices services(m_graph->GetUploadService(), m_graph->GetDescriptorService());
				WaitSlot(state, slot);
				std::shared_ptr<const RenderGraph::PersistentRecording> recording;
				{
					Async::Preparation preparation(state);
					const auto version = state.backingVersion.load(std::memory_order_acquire);
					auto ticket = m_graph->PreparePersistentTicket(m_desc.device, request.result->epoch, static_cast<uint8_t>(slot), state.hostFrame, version,
						request.hostData);
					recording = RenderGraph::PersistentTicketRecording(*ticket);
					// Recorded only: its admission is abandoned, never submitted.
					(void)m_graph->CompletePersistentTicket(*ticket, {});
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
					AddRevisionCandidate(state, *ticket, *result);
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

void PersistentGraphHost::PrepareTicket(Async& state, uint32_t epochIndex, int32_t requestedSlot) {
	BT_ZONE_SCOPE("ORG.Host.PrepareTicket.Worker");
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
	runtime::ScopedActiveGraphServices services(m_graph->GetUploadService(), m_graph->GetDescriptorService());
	WaitSlot(state, slot);
	// The render thread records this ticket's uploads into an open list: resetting and beginning it is several
	// microseconds of driver work that belongs here.
	auto& uploads = state.uploadLists[slot];
	uploads.allocator->Recycle();
	uploads.list->Recycle(uploads.allocator.Get());
	DescriptorHeapManager::GetInstance().ProcessDeferredReleases(static_cast<uint8_t>(slot));
	entry.reserved = true;
	std::shared_ptr<Async::Ticket> ticket;
	if (state.revisionDriven[epochIndex]) {
		// Nothing from live state: the ticket admits the revisions' recordings for its slot, the ones still held.
		ticket = m_graph->PrepareRevisionTicket(state.epochs[epochIndex], static_cast<uint8_t>(slot), state.hostFrame++);
		auto& live = state.liveRecordings[epochIndex];
		std::erase_if(live, [](const auto& a_recording) { return a_recording.expired(); });
		for (const auto& weak : live)
			if (const auto recording = weak.lock()) AddRevisionCandidate(state, *ticket, *recording);
		state.outstanding[epochIndex] = ticket;
	} else {
		// Clear of the submitting thread's backing changes: the preparation captures every slot's backing.
		Async::Preparation preparation(state);
		const auto version = state.backingVersion.load(std::memory_order_acquire);
		ticket = m_graph->PreparePersistentTicket(m_desc.device, state.epochs[epochIndex], static_cast<uint8_t>(slot), state.hostFrame++, version);
		RenderGraph::SetPersistentTicketHostTag(*ticket, version);
		// The revisions' recordings still held, as alternatives the submitting thread may take instead (UseEpochRecording).
		auto& live = state.liveRecordings[epochIndex];
		std::erase_if(live, [](const auto& a_recording) { return a_recording.expired(); });
		for (const auto& weak : live)
			if (const auto recording = weak.lock()) AddRevisionCandidate(state, *ticket, *recording);
		state.outstanding[epochIndex] = ticket;
	}
	state.needsTicket[epochIndex] = 0;
	state.cells[epochIndex].store(std::move(ticket), std::memory_order_release);
	state.ticketWake.fetch_add(1, std::memory_order_release);
	state.ticketWake.notify_all();
}

void PersistentGraphHost::AddRevisionCandidate(Async& state, RenderGraph::PersistentTicket& ticket, const EpochRecording& recording) {
	const uint32_t slot = RenderGraph::PersistentTicketSlot(ticket);
	if (slot < recording.firstSlot || slot - recording.firstSlot >= recording.slots.size())
		throw std::logic_error("A revision recording does not cover its epoch's ticket slot");
	(void)state;
	m_graph->AddPersistentTicketCandidate(ticket, recording.slots[slot - recording.firstSlot]);
}

PersistentGraphHost::BackingMutation::BackingMutation(PersistentGraphHost* host) : m_host(host) {
	BufferBase::EnterBackingMutation();
	if (m_host->m_backingMutationDepth++ || !m_host->m_async) return;
	auto& async = *m_host->m_async;
	BT_ZONE_SCOPE("ORG.Host.BackingMutation");
	async.mutating.store(1, std::memory_order_seq_cst);
	++m_host->m_asyncStats.backingMutations;
	// A preparation in progress ends within its own work (the slot's GPU wait comes before it); a failed host has none.
	bool waited = false;
	for (auto preparing = async.preparing.load(std::memory_order_seq_cst); preparing && !async.failed.load(std::memory_order_acquire);
		preparing = async.preparing.load(std::memory_order_seq_cst)) {
		waited = true;
		async.preparing.wait(preparing, std::memory_order_seq_cst);
	}
	m_host->m_asyncStats.backingWaits += waited;
	async.backingVersion.fetch_add(1, std::memory_order_acq_rel);
}

PersistentGraphHost::BackingMutation::BackingMutation(BackingMutation&& other) noexcept : m_host(std::exchange(other.m_host, nullptr)) {}

PersistentGraphHost::BackingMutation::~BackingMutation() {
	if (!m_host) return;
	if (!--m_host->m_backingMutationDepth && m_host->m_async) {
		auto& async = *m_host->m_async;
		async.mutating.store(0, std::memory_order_seq_cst);
		async.mutating.notify_all();
	}
	BufferBase::LeaveBackingMutation();
}

PersistentGraphHost::BackingMutation PersistentGraphHost::MutateBackings() {
	return BackingMutation(this);
}

bool PersistentGraphHost::CanUseEpochRecording(const EpochRecording& recording, std::string* why) const {
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

void PersistentGraphHost::NoteNewVersions() noexcept {
	// After the version's publication (release), so a preparation that reads the new value resolves the new version.
	if (m_async) m_async->backingVersion.fetch_add(1, std::memory_order_acq_rel);
}

void PersistentGraphHost::StopAsync() {
	BT_ZONE_SCOPE("ORG.Host.StopAsync");
	if (!m_async) return;
	auto async = std::move(m_async);
	Async::Message stop;
	stop.kind = Async::Message::Kind::Stop;
	(void)async->Post(stop);
	if (async->thread.joinable()) async->thread.join();
	BufferBase::UnregisterConcurrentBackingCapture();
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
	for (auto& slot : async->slots) {
		bool complete = true;
		for (const auto& [timeline, value] : slot.points)
			if (timeline->HostWait(value) != rhi::Result::Ok) complete = false;
		if (!complete) {
			if (slot.resourceOwner) m_uncertainExecutionOwners.push_back(std::move(slot.resourceOwner));
			if (slot.keepAlive) m_uncertainUploadOwners.push_back(std::move(slot.keepAlive));
		}
	}
	// Every slot's work is done (or its owners kept as uncertain): the kept recordings can go.
	if (m_graph) m_graph->ClearKeptPersistentRecordings();
	if (async->uncertainResourceOwner)
		m_uncertainExecutionOwners.push_back(std::move(async->uncertainResourceOwner));
	if (async->uncertainUploadOwner)
		m_uncertainUploadOwners.push_back(std::move(async->uncertainUploadOwner));
}

PersistentGraphHost::AsyncStats PersistentGraphHost::TakeAsyncStats() noexcept {
	if (m_graph) {
		const auto recordings = m_graph->TakePersistentRecordingStats();
		m_asyncStats.recorded = recordings.recorded;
		m_asyncStats.kept = recordings.kept;
		m_asyncStats.reused = recordings.reused;
		m_asyncStats.adopted = recordings.adopted;
	}
	if (m_async) {
		m_asyncStats.revisionSlotsRecorded = m_async->revisionSlotsRecorded.exchange(0, std::memory_order_relaxed);
		m_asyncStats.revisionRecordings = m_async->revisionRecordings.exchange(0, std::memory_order_relaxed);
	}
	if (m_async) {
		m_asyncStats.wakesSkipped = std::exchange(m_async->wakesSkipped, 0);
		m_asyncStats.preparationWaits = m_async->preparationWaits.exchange(0, std::memory_order_relaxed);
	}
	return std::exchange(m_asyncStats, {});
}

bool PersistentGraphHost::TryPostOwnedPreparation(FrameCallback& request) {
	if (!request || !m_async || BuildDue())
		return false;
	Async::Message message;
	message.kind = Async::Message::Kind::OwnedPrepare;
	message.ownedPrepare = std::move(request);
	if (m_async->TryPost(message))
		return true;
	request = std::move(message.ownedPrepare);
	return false;
}

bool PersistentGraphHost::TryReserveReadyEpochs(std::span<const uint32_t> epochs, FrameTicketReservation& destination) {
	BT_ZONE_SCOPE("ORG.Host.TryReserveReadyEpochs");
	if (!destination.Empty() || epochs.empty() || !m_async || BuildDue() || m_async->failed.load(std::memory_order_acquire))
		return false;
	std::vector<FrameTicketReservation::Entry> ready;
	ready.reserve(epochs.size());
	for (const auto epoch : epochs) {
		const auto found = std::ranges::find(m_async->epochs, epoch);
		if (found == m_async->epochs.end() || std::ranges::find(ready, epoch, &FrameTicketReservation::Entry::epoch) != ready.end()) {
			// Duplicate epochs cannot be consumed twice from a single ticket cell.
			return false;
		}
		const auto index = static_cast<uint32_t>(found - m_async->epochs.begin());
		auto ticket = m_async->cells[index].load(std::memory_order_acquire);
		if (!ticket || !m_async->Current(*ticket)) return false;
		ready.push_back({epoch, std::move(ticket)});
	}
	const auto outstanding = m_async->written.load(std::memory_order_relaxed) - m_async->read.load(std::memory_order_acquire);
	const auto credits = m_async->controlCredits->load(std::memory_order_relaxed);
	if (outstanding + credits + ready.size() > Async::kMailbox) return false;
	m_async->controlCredits->fetch_add(static_cast<uint32_t>(ready.size()), std::memory_order_relaxed);
	destination.tickets = std::move(ready);
	destination.generation = m_ticketGeneration;
	destination.controlCredits = m_async->controlCredits;
	destination.heldCredits = static_cast<uint32_t>(destination.tickets.size());
	return true;
}

bool PersistentGraphHost::TrySubmitReservedEpoch(FrameTicketReservation& reservation, uint32_t epoch, const FrameCallback& beforeSubmit) {
	BT_ZONE_SCOPE("ORG.Host.TrySubmitReservedEpoch");
	if (!m_async || BuildDue() || reservation.generation != m_ticketGeneration
		|| m_async->failed.load(std::memory_order_acquire)) return false;
	auto found = std::ranges::find(reservation.tickets, epoch, &FrameTicketReservation::Entry::epoch);
	if (found == reservation.tickets.end() || !found->ticket) return false;
	const auto index = m_async->IndexOf(epoch);
	auto expected = found->ticket;
	if (!m_async->cells[index].compare_exchange_strong(expected, {}, std::memory_order_acq_rel)) return false;
	auto ticket = std::move(found->ticket);
	m_lastTimings = {};
	m_lastTimings.async = true;
	SubmitTicket(*m_async, index, std::move(ticket), beforeSubmit, false, &reservation);
	return true;
}

void PersistentGraphHost::SubmitEpoch(uint32_t epoch, const FrameCallback& beforeSubmit, std::shared_ptr<const void> resourceOwner,
	std::shared_ptr<const EpochRecording> recording) {
	BT_ZONE_SCOPE("ORG.Host.SubmitEpoch");
	m_lastTimings = {};
	m_lastTimings.async = true;
	auto lap = [last = std::chrono::steady_clock::now()](double& a_into) mutable {
		const auto now = std::chrono::steady_clock::now();
		a_into += std::chrono::duration<double, std::micro>(now - last).count();
		last = now;
	};
	if (BuildDue()) {
		StopAsync();
		Build();
		StartAsync();
	}
	lap(m_lastTimings.buildUs);
	if (!m_async) throw std::logic_error("SubmitEpoch requires async epochs");
	auto& async = *m_async;
	if (async.controlCredits->load(std::memory_order_relaxed))
		throw std::logic_error("SubmitEpoch cannot mix with outstanding frame ticket reservations");
	async.ThrowIfFailed();
	const auto index = async.IndexOf(epoch);
	auto ticket = async.cells[index].exchange(nullptr, std::memory_order_acq_rel);
	if (!ticket) {
		++m_asyncStats.waited;
		ticket = async.WaitTicket(index);
	}
	lap(m_lastTimings.ticketWaitUs);
	SubmitTicket(async, index, std::move(ticket), beforeSubmit, true, nullptr, std::move(resourceOwner), std::move(recording));
}

void PersistentGraphHost::SubmitTicket(Async& async, uint32_t index, std::shared_ptr<RenderGraph::PersistentTicket> ticket,
	const FrameCallback& beforeSubmit, bool allowReprepare, FrameTicketReservation* reservation,
	std::shared_ptr<const void> resourceOwner, std::shared_ptr<const EpochRecording> recording) {
	// A submission's completion message wakes the host's thread lazily (wakesOnGpu); a discard always wakes it.
	auto postControl = [&](Async::Message& message) {
		const bool lazy = message.kind == Async::Message::Kind::Submitted;
		if (!reservation) {
			if (!async.Post(message, lazy)) throw std::runtime_error("Async epoch host stopped before accepting its completion");
			return;
		}
		if (!async.TryPost(message, true, lazy)) throw std::logic_error("Reserved async epoch lost its control-mailbox credit");
		--reservation->heldCredits;
		reservation->controlCredits->fetch_sub(1, std::memory_order_relaxed);
	};
	auto lap = [last = std::chrono::steady_clock::now()](double& a_into) mutable {
		const auto now = std::chrono::steady_clock::now();
		a_into += std::chrono::duration<double, std::micro>(now - last).count();
		last = now;
	};
	const uint32_t slot = RenderGraph::PersistentTicketSlot(*ticket);
	m_asyncSlot = slot;
	runtime::ScopedActiveGraphServices services(m_graph->GetUploadService(), m_graph->GetDescriptorService());
	// The host's thread waited for this slot before preparing the ticket: its upload pages are free again. What retires from
	// the deletion queues goes to the host's thread with the submission: its driver frees take the device's memory locks,
	// which the render thread must not wait on here.
	std::shared_ptr<void> retired;
	if (auto* uploads = m_graph->GetUploadService()) retired = uploads->ProcessDeferredReleasesRetiringElsewhere(static_cast<uint8_t>(slot));
	lap(m_lastTimings.releaseUs);
	if (beforeSubmit) {
		BT_ZONE_SCOPE("ORG.Host.CommitInputs");
		struct Submitting {
			PersistentGraphHost& host;
			~Submitting() { host.m_submittingTicket = nullptr; }
		} submitting{*this};
		m_submittingTicket = ticket.get();
		m_submittingIndex = index;
		beforeSubmit(*m_graph);
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
		&& m_graph->AdoptKeptPersistentRecording(*ticket, async.backingVersion.load(std::memory_order_acquire))
		&& !async.Current(*ticket))
		throw std::logic_error("An adopted recording left its ticket stale");
	if (!async.revisionDriven[index] && !chosen && !async.Current(*ticket)) {
		BT_ZONE_SCOPE("ORG.Host.ReprepareStaleTicket");
		m_asyncStats.staleBacking += RenderGraph::PersistentTicketCurrent(*ticket);
		// Prepared before beforeSubmit changed what a pass depends on, or a backing it captured: prepare it again, for the
		// same slot (its latch region was just written), and wait. The new one is prepared after the change, so it is current.
		++m_asyncStats.stale;
		Async::Message discard;
		discard.kind = Async::Message::Kind::Discard;
		discard.epochIndex = index;
		discard.ticket = std::move(ticket);
		if (!allowReprepare) {
			postControl(discard);
			// A reserved ticket becoming stale is an execution-contract violation. Never wait or
			// silently omit a draw after native ownership may have been claimed.
			throw std::logic_error("Reserved async epoch ticket became stale before submission");
		}
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
	// The uploads queued since the last submission, as plain copies ahead of the ticket (its first batch
	// starts with a full barrier, so they need none of their own).
	auto& list = async.uploadLists[slot];  // begun by the host's thread (PrepareTicket)
	std::shared_ptr<void> keepAlive;
	bool recorded = false;
	auto commands = list.list.Get();
	auto& recordedLists = async.recordedLists;
	const bool supported = m_graph->RecordPendingUploads(m_desc.device, commands, static_cast<uint8_t>(slot), keepAlive, recorded, recordedLists);
	{
		BT_ZONE_SCOPE("ORG.Host.CloseUploadList");
		list.list->End();
	}
	if (!supported) {
		Async::Message discard;
		discard.kind = Async::Message::Kind::Discard;
		discard.epochIndex = index;
		discard.ticket = std::move(ticket);
		postControl(discard);
		throw std::runtime_error("Async epochs cannot record these uploads (only pointer-targeted buffer copies)");
	}
	Async::Message submitted;
	submitted.kind = Async::Message::Kind::Submitted;
	submitted.epochIndex = index;
	submitted.keepAlive = std::move(keepAlive);
	submitted.retired = std::move(retired);
	submitted.resourceOwner = reservation ? reservation->resourceOwner : std::move(resourceOwner);
	if (recorded || !recordedLists.empty()) {
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
		ok = RenderGraph::SubmitPersistentTicket(*ticket, async.queueValues, submitted.submission, submitted.uploadSignal, m_frameWaitValue);
	} catch (...) {
		// The queue may have accepted a prefix of this ticket. Completion is
		// unknown, so preserve both owners until device teardown.
		if (submitted.resourceOwner) m_uncertainExecutionOwners.push_back(std::move(submitted.resourceOwner));
		if (submitted.keepAlive) m_uncertainUploadOwners.push_back(std::move(submitted.keepAlive));
		throw;
	}
	lap(m_lastTimings.submitUs);
	submitted.ticket = std::move(ticket);
	try {
		postControl(submitted);
	} catch (...) {
		if (submitted.resourceOwner) m_uncertainExecutionOwners.push_back(std::move(submitted.resourceOwner));
		if (submitted.keepAlive) m_uncertainUploadOwners.push_back(std::move(submitted.keepAlive));
		throw;
	}
	lap(m_lastTimings.postUs);
	++m_asyncStats.submitted;
	if (!ok) throw std::runtime_error("Async epoch submission failed");
}

} // namespace org
