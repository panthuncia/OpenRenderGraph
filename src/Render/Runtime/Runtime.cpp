#include "OpenRenderGraph/Trace.h"
#include "OpenRenderGraph/Runtime.h"

#include <BasicTelemetry/Tracy.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <stdexcept>
#include <string>
#include <thread>

#include "Render/PipelineLibrary.h"
#include "Render/QueueRegistry.h"
#include "Render/RenderGraph/PersistentGraph.h"
#include "Render/Runtime/IDescriptorService.h"
#include "Render/Runtime/IReadbackService.h"
#include "Render/Runtime/IRenderGraphSettingsService.h"
#include "Render/Runtime/IStatisticsService.h"
#include "Render/Runtime/IUploadService.h"
#include "Render/Runtime/OpenRenderGraphSettings.h"
#include "Render/Runtime/RuntimeDevice.h"
#include "Render/Runtime/TaskServiceAccess.h"
#include "Render/Runtime/ThreadPoolTaskService.h"

#include "Managers/Singletons/DeletionManager.h"
#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Managers/Singletons/DeviceManager.h"
#include "Managers/Singletons/StatisticsManager.h"
#include "Managers/Singletons/UploadManager.h"

namespace org {

struct Runtime::DeviceRelay {
	const void* owner = nullptr;
	const void* consumer = nullptr;
	rhi::Device ownerDevice{};
	rhi::TimelinePtr ownerSide, consumerSide;
	uint64_t identity = 0;
	std::mutex mutex;   // A value is taken and signalled together: a fence's signals must not go backwards.
	uint64_t next = 1;
};

// Drops what it is handed on its own thread, in order. A driver free can block in the kernel until the GPU is done with
// the memory (WDDM frees GPU virtual addresses that way). On this thread such a wait holds up nothing the GPU waits for.
// The render thread and a graph host's thread hand their frees here.
class Runtime::Deleter {
public:
	Deleter() { m_thread = std::thread([this] { Run(); }); }
	~Deleter() {
		{
			std::scoped_lock lock(m_mutex);
			m_stopping = true;
		}
		m_wake.notify_one();
		m_thread.join();
	}
	Deleter(const Deleter&) = delete;
	Deleter& operator=(const Deleter&) = delete;

	void Post(std::shared_ptr<void> objects) {
		size_t queued = 0;
		{
			std::scoped_lock lock(m_mutex);
			m_queue.push_back(std::move(objects));
			queued = m_queue.size();
		}
		m_wake.notify_one();
		// A free that does not return shows here, once per stuck free: the thread that would have frozen goes on.
		const auto since = m_busySince.load(std::memory_order_acquire);
		if (!since) return;
		const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
		const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::duration(now - since)).count();
		if (ms >= kStuckFreeMs && m_stuckLogged.exchange(since) != since)
			spdlog::warn("ORG runtime: the deleter thread has been in one free for {:.0f} ms ({} batches queued); the driver's free waits "
				"for the GPU, which may wait for later work", ms, queued);
	}

private:
	// A free slower than this is logged: it waited for the GPU.
	static constexpr double kSlowFreeMs = 100.0;
	// A free still running after this is logged by the next Post.
	static constexpr double kStuckFreeMs = 1000.0;

	void Run() {
		std::unique_lock lock(m_mutex);
		for (;;) {
			m_wake.wait(lock, [&] { return m_stopping || !m_queue.empty(); });
			// Stopping drains the queue first: nothing handed over is left behind.
			if (m_queue.empty()) return;
			auto objects = std::move(m_queue.front());
			m_queue.pop_front();
			const auto queued = m_queue.size();
			lock.unlock();
			const auto begin = std::chrono::steady_clock::now();
			m_busySince.store(begin.time_since_epoch().count(), std::memory_order_release);
			{
				BT_ZONE_SCOPE("ORG.Runtime.DestroyDetached");
				objects.reset();
			}
			m_busySince.store(0, std::memory_order_release);
			const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
			if (ms >= kSlowFreeMs)
				spdlog::warn("ORG runtime: a deferred free took {:.0f} ms on the deleter thread ({} batches queued behind it); the driver's "
					"free waited for the GPU", ms, queued);
			lock.lock();
		}
	}

	std::mutex m_mutex;
	std::condition_variable m_wake;
	std::deque<std::shared_ptr<void>> m_queue;
	bool m_stopping = false;
	// When the free in progress started (steady clock ticks), or 0; and the start last reported as stuck.
	std::atomic<int64_t> m_busySince{0};
	std::atomic<int64_t> m_stuckLogged{0};
	std::thread m_thread;
};

namespace {
std::atomic<Runtime*> g_current{nullptr};

bool SameQueue(const rhi::Queue& a, const rhi::Queue& b) noexcept {
	const auto ha = a.GetQueueHandle(), hb = b.GetQueueHandle();
	return a.impl == b.impl && ha.index == hb.index && ha.generation == hb.generation;
}
} // namespace

std::shared_ptr<Runtime> Runtime::Create(Desc desc) {
	if (!desc.device) throw std::invalid_argument("Runtime requires a device");
	if (desc.backend == rhi::Backend::Null) throw std::invalid_argument("Runtime requires a device backend");
	if (desc.slots == 0 || desc.slots > kMaxSlots)
		throw std::invalid_argument("Runtime slots must be 1-" + std::to_string(kMaxSlots) + ", not " + std::to_string(desc.slots));
	static std::once_flag traceFromEnvironment;
	std::call_once(traceFromEnvironment, [] { trace::ConfigureFromEnvironment(); });
	Runtime* expected = nullptr;
	std::shared_ptr<Runtime> runtime(new Runtime(std::move(desc)));
	if (!g_current.compare_exchange_strong(expected, runtime.get()))
		throw std::logic_error("Runtime: one runtime per process (its services are still process-wide)");
	try {
		runtime->Initialize();
	} catch (...) {
		g_current.store(nullptr);
		throw;
	}
	return runtime;
}

Runtime* Runtime::Current() noexcept {
	return g_current.load(std::memory_order_acquire);
}

Runtime::Runtime(Desc desc)
	: m_desc(std::move(desc)), m_sequencer(std::make_shared<persistent::ResourceSequencer>()),
	  m_schedules(std::make_shared<compiler::ScheduleCache>()) {}

void Runtime::Initialize() {
	// Slot-indexed services size themselves from the settings: the runtime's slots, set once here.
	auto settings = runtime::GetOpenRenderGraphSettings();
	settings.numFramesInFlight = m_desc.slots;
	runtime::SetOpenRenderGraphSettings(settings);

	runtime::InitializeRuntimeDevice(m_desc.device);
	if (!m_desc.tasks) m_desc.tasks = std::make_shared<runtime::ThreadPoolTaskService>();
	runtime::SetDefaultTaskService(m_desc.tasks);
	m_pipelines = std::make_shared<PipelineLibrary>(m_desc.tasks, m_desc.pipelineCompiler);
	m_pipelines->RegisterDevice(DeviceInstanceId::Primary, m_desc.backend, m_desc.device);
	{
		std::scoped_lock lock(m_relaysMutex);
		m_devices.emplace_back(m_desc.device, m_desc.backend);
	}

	for (auto kind : { rhi::QueueKind::Graphics, rhi::QueueKind::Compute, rhi::QueueKind::Copy })
		RegisterQueue(m_desc.device.GetQueue(kind));
	m_deleter = std::make_unique<Deleter>();
	DeletionManager::GetInstance().Initialize();
	DeletionManager::GetInstance().SetFrontier([this] { return Frontier(); }, [this](const runtime::SubmittedPoint& point) { return Reached(point); },
		[this] { return m_assigning.load(std::memory_order_acquire) == 0; });
	DescriptorHeapManager::GetInstance().SetFrontier([this] { return Frontier(); }, [this](const runtime::SubmittedPoint& point) { return Reached(point); });

	m_descriptors = runtime::CreateDefaultDescriptorService();
	m_settings = runtime::CreateDefaultRenderGraphSettingsService();
	m_uploads = runtime::CreateDefaultUploadService();
	m_uploads->Initialize();
	if (auto queue = UploadManager::GetInstance().GetDedicatedUploadQueue()) RegisterQueue(queue);

	m_statistics = runtime::CreateDefaultStatisticsService();
	m_statistics->Initialize();
	m_statistics->RegisterQueue(rhi::QueueKind::Graphics);
	m_statistics->RegisterQueue(rhi::QueueKind::Compute);
	m_statistics->SetupQueryHeap();

	if (rhi::Failed(m_desc.device.CreateTimeline(m_readbackFence, 0, "ORG.Readback.Graphics")) ||
		rhi::Failed(m_desc.device.CreateTimeline(m_copyReadbackFence, 0, "ORG.Readback.Copy")))
		throw std::runtime_error("Runtime failed to create its readback timelines");
	m_readbacks = runtime::CreateDefaultReadbackService();
	m_readbacks->Initialize(m_readbackFence.Get(), m_copyReadbackFence.Get());
}

Runtime::~Runtime() {
	if (g_current.load() != this) return;  // Create failed before publishing it
	m_pipelines.reset(); // Joins its builds, which use the device.
	m_deleter.reset();   // Drops what it still holds while the device and the services are alive.
	{
		std::scoped_lock lock(m_relaysMutex);
		m_relays.clear(); // Their last signals were waited for with the queues that consumed them.
	}
	DeletionManager::GetInstance().SetFrontier({}, {});
	DescriptorHeapManager::GetInstance().SetFrontier({}, {});
	StatisticsManager::GetInstance().ClearAll();
	m_readbacks.reset();
	m_uploads.reset();
	m_statistics.reset();
	m_descriptors.reset();
	m_settings.reset();
	runtime::ShutdownRuntimeDevice();
	m_readbackFence.Reset();
	m_copyReadbackFence.Reset();
	{
		std::scoped_lock lock(m_queuesMutex);
		m_queues.clear();
		m_assignedTimelines.clear();
		m_externalPoints.clear();
	}
	if (runtime::GetDefaultTaskService() == m_desc.tasks) runtime::SetDefaultTaskService({});
	g_current.store(nullptr, std::memory_order_release);
}

void Runtime::RegisterQueue(rhi::Queue queue) {
	if (!queue) return;
	std::scoped_lock lock(m_queuesMutex);
	if (std::none_of(m_queues.begin(), m_queues.end(), [&](const rhi::Queue& known) { return SameQueue(known, queue); }))
		m_queues.push_back(queue);
}

void Runtime::UnregisterQueue(rhi::Queue queue) {
	if (!queue) return;
	// Its progress timeline goes with it, and a destroyed timeline reads as never reached: wait for what was submitted, then
	// answer its points as reached from now on.
	auto progress = queue.Progress();
	if (progress.timeline && progress.submitted) (void)progress.timeline.HostWait(progress.submitted);
	std::scoped_lock lock(m_queuesMutex);
	const auto before = m_queues.size();
	std::erase_if(m_queues, [&](const rhi::Queue& known) { return SameQueue(known, queue); });
	if (m_queues.size() != before && progress.timeline) m_retiredTimelines.emplace_back(progress.timeline.impl, progress.timeline.GetHandle());
}

void Runtime::RegisterAssignedTimeline(std::shared_ptr<rhi::TimelinePtr> fence, std::shared_ptr<const std::atomic<uint64_t>> nextValue) {
	if (!fence || !*fence || !nextValue) return;
	std::scoped_lock lock(m_queuesMutex);
	if (std::ranges::any_of(m_assignedTimelines, [&](const auto& known) { return known.first == fence; })) return;
	m_assignedTimelines.emplace_back(std::move(fence), std::move(nextValue));
}

void Runtime::UnregisterAssignedTimeline(const std::shared_ptr<rhi::TimelinePtr>& fence) {
	std::scoped_lock lock(m_queuesMutex);
	std::erase_if(m_assignedTimelines, [&](const auto& known) { return known.first == fence; });
}

void Runtime::NoteExternalPoint(const rhi::Timeline& timeline, uint64_t value) {
	if (!timeline.IsValid() || !value || value == UINT64_MAX) return;
	const auto handle = timeline.GetHandle();
	std::scoped_lock lock(m_queuesMutex);
	for (auto& point : m_externalPoints) {
		const auto known = point.timeline.GetHandle();
		if (point.timeline.impl != timeline.impl || known.index != handle.index || known.generation != handle.generation) continue;
		point.value = (std::max)(point.value, value);
		return;
	}
	m_externalPoints.push_back({timeline, value});
}

void Runtime::NoteTimelineDestroyed(const rhi::Timeline& timeline) {
	if (!timeline.IsValid()) return;
	std::scoped_lock lock(m_queuesMutex);
	m_retiredTimelines.emplace_back(timeline.impl, timeline.GetHandle());
	// Its external points go too: a destroyed timeline would read as never reached.
	const auto handle = timeline.GetHandle();
	std::erase_if(m_externalPoints, [&](const runtime::SubmittedPoint& point) {
		const auto known = point.timeline.GetHandle();
		return point.timeline.impl == timeline.impl && known.index == handle.index && known.generation == handle.generation;
	});
}

uint64_t Runtime::TimelineIdentity(const rhi::Timeline& timeline) {
	const auto handle = timeline.GetHandle();
	std::lock_guard lock(m_identitiesMutex);
	auto [found, inserted] = m_identities.try_emplace({timeline.impl, handle.index, handle.generation}, m_nextIdentity);
	if (inserted) {
		m_owners.emplace(m_nextIdentity, timeline.impl);
		m_handles.emplace(m_nextIdentity++, handle);
	}
	return found->second;
}

const void* Runtime::TimelineDeviceOf(uint64_t identity) const {
	std::lock_guard lock(m_identitiesMutex);
	const auto found = m_owners.find(identity);
	return found == m_owners.end() ? nullptr : found->second;
}

rhi::TimelineHandle Runtime::TimelineHandleOf(uint64_t identity) const {
	std::lock_guard lock(m_identitiesMutex);
	const auto found = m_handles.find(identity);
	return found == m_handles.end() ? rhi::TimelineHandle{} : found->second;
}

bool Runtime::Reached(const runtime::SubmittedPoint& point) const {
	if (!point.timeline.IsValid()) return true;
	{
		std::scoped_lock lock(m_queuesMutex);
		const auto handle = point.timeline.GetHandle();
		for (const auto& [impl, retired] : m_retiredTimelines)
			if (impl == point.timeline.impl && retired.index == handle.index && retired.generation == handle.generation) return true;
	}
	return runtime::TimelineReached(point);
}

Runtime::RelayedPoint Runtime::RelayWaits(const void* owner, const void* consumer, std::span<const rhi::TimelinePoint> points) {
	if (!owner || !consumer || owner == consumer) throw std::invalid_argument("RelayWaits needs two different devices");
	DeviceRelay* relay = nullptr;
	{
		std::scoped_lock lock(m_relaysMutex);
		for (const auto& existing : m_relays)
			if (existing->owner == owner && existing->consumer == consumer) relay = existing.get();
		if (!relay) {
			const std::pair<rhi::Device, rhi::Backend>* ownerEntry = nullptr;
			const std::pair<rhi::Device, rhi::Backend>* consumerEntry = nullptr;
			for (const auto& entry : m_devices) {
				if (entry.first.impl == owner) ownerEntry = &entry;
				if (entry.first.impl == consumer) consumerEntry = &entry;
			}
			if (!ownerEntry || !consumerEntry || ownerEntry->second == rhi::Backend::Null || consumerEntry->second == rhi::Backend::Null)
				throw std::logic_error("RelayWaits: a device is not registered with its API");
			auto created = std::make_unique<DeviceRelay>();
			created->owner = owner;
			created->consumer = consumer;
			created->ownerDevice = ownerEntry->first;
			if (rhi::Failed(CreateBridgeTimelinePair(ownerEntry->first, ownerEntry->second, consumerEntry->first, consumerEntry->second,
					"ORG cross-device relay", created->ownerSide, created->consumerSide)))
				throw std::runtime_error("RelayWaits: failed to create a fence shared by the two devices");
			created->identity = TimelineIdentity(created->consumerSide.Get());
			relay = created.get();
			m_relays.push_back(std::move(created));
		}
	}
	std::scoped_lock lock(relay->mutex);
	const uint64_t value = relay->next++;
	const rhi::TimelinePoint signal{relay->ownerSide->GetHandle(), value};
	rhi::SubmitDesc submit{};
	submit.waits = {points.data(), static_cast<uint32_t>(points.size())};
	submit.signals = {&signal, 1u};
	if (const auto result = relay->ownerDevice.GetQueue(rhi::QueueKind::Graphics).Submit({}, submit); result != rhi::Result::Ok) {
		--relay->next;
		throw std::runtime_error(std::string("RelayWaits: the relay submission failed: ") + rhi::ResultName(result));
	}
	return {relay->identity, {relay->consumerSide->GetHandle(), value}};
}

void Runtime::RegisterDevice(rhi::Device device, rhi::Backend backend) {
	if (!device) return;
	{
		std::scoped_lock lock(m_relaysMutex);
		auto found = std::ranges::find_if(m_devices, [&](const auto& entry) { return entry.first.impl == device.impl; });
		if (found == m_devices.end()) m_devices.emplace_back(device, backend);
		else if (found->second == rhi::Backend::Null) found->second = backend;
	}
	for (auto kind : { rhi::QueueKind::Graphics, rhi::QueueKind::Compute, rhi::QueueKind::Copy })
		RegisterQueue(device.GetQueue(kind));
}

runtime::SubmissionFrontier Runtime::Frontier() const {
	runtime::SubmissionFrontier frontier;
	std::scoped_lock lock(m_queuesMutex);
	frontier.reserve(m_queues.size());
	for (const auto& queue : m_queues) {
		auto progress = queue.Progress();
		if (progress.timeline && progress.submitted) frontier.push_back({ progress.timeline, progress.submitted });
	}
	// Values given out on a host's timelines: work in a host's own stream (in-stream epochs) that no queue above was given.
	for (const auto& [fence, nextValue] : m_assignedTimelines) {
		const auto next = nextValue->load(std::memory_order_acquire);
		if (next > 1 && *fence) frontier.push_back({ fence->Get(), next - 1 });
	}
	// Values external streams signal after their own uses of ORG objects (NoteExternalPoint).
	for (const auto& point : m_externalPoints) frontier.push_back(point);
	return frontier;
}

void Runtime::Retire() {
	DescriptorHeapManager::GetInstance().ProcessDeferredReleases(0);
	(void)DeletionManager::GetInstance().Retire();
}

std::shared_ptr<void> Runtime::RetireElsewhere() {
	auto retired = DeletionManager::GetInstance().Retire();
	if (retired.Empty()) return {};
	return std::make_shared<DeletionManager::Retired>(std::move(retired));
}

void Runtime::RetireDetached() {
	DestroyDetached(DescriptorHeapManager::GetInstance().ProcessDeferredReleasesElsewhere());
	DestroyDetached(RetireElsewhere());
}

void Runtime::DestroyDetached(std::shared_ptr<void> objects) {
	if (!objects) return;
	// No deleter (the runtime is being torn down): dropped here.
	if (!m_deleter) return;
	m_deleter->Post(std::move(objects));
}

} // namespace org
