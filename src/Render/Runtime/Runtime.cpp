#include "OpenRenderGraph/Trace.h"
#include "OpenRenderGraph/Runtime.h"

#include <algorithm>
#include <atomic>
#include <stdexcept>

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
	if (desc.slots == 0 || desc.slots > 255) throw std::invalid_argument("Runtime slots must be 1-255");
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

Runtime::Runtime(Desc desc) : m_desc(std::move(desc)), m_sequencer(std::make_shared<persistent::ResourceSequencer>()) {}

void Runtime::Initialize() {
	// Slot-indexed services size themselves from the settings: the runtime's slots, set once here.
	auto settings = runtime::GetOpenRenderGraphSettings();
	settings.numFramesInFlight = static_cast<uint8_t>(m_desc.slots);
	runtime::SetOpenRenderGraphSettings(settings);

	runtime::InitializeRuntimeDevice(m_desc.device);
	if (!m_desc.tasks) m_desc.tasks = std::make_shared<runtime::ThreadPoolTaskService>();
	runtime::SetDefaultTaskService(m_desc.tasks);

	for (auto kind : { rhi::QueueKind::Graphics, rhi::QueueKind::Compute, rhi::QueueKind::Copy })
		RegisterQueue(m_desc.device.GetQueue(kind));
	DeletionManager::GetInstance().Initialize();
	DeletionManager::GetInstance().SetFrontier([this] { return Frontier(); }, [this](const runtime::SubmittedPoint& point) { return Reached(point); });
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

uint64_t Runtime::TimelineIdentity(const rhi::Timeline& timeline) {
	const auto handle = timeline.GetHandle();
	std::lock_guard lock(m_identitiesMutex);
	auto [found, inserted] = m_identities.try_emplace({timeline.impl, handle.index, handle.generation}, m_nextIdentity);
	if (inserted) m_handles.emplace(m_nextIdentity++, handle);
	return found->second;
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

void Runtime::RegisterDevice(rhi::Device device) {
	if (!device) return;
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

} // namespace org
