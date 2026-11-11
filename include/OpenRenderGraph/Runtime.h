#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <tuple>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include <rhi.h>

#include "Render/Runtime/SubmissionFrontier.h"

namespace org::runtime {
class ITaskService;
class IUploadService;
class IReadbackService;
class IDescriptorService;
class IStatisticsService;
class IRenderGraphSettingsService;
}

namespace org::persistent { class ResourceSequencer; }
namespace org::compiler { class ScheduleCache; }

namespace org {

class PipelineCompiler;
class PipelineLibrary;

// The ORG runtime of a device: what every graph on it shares. One per process for now (the services behind it are still
// process singletons); graphs and hosts take it and never initialize, reset or shut down shared state themselves.
// - The device, and the queues whose progress bounds retirement (the device's three, plus RegisterQueue's).
// - The services: tasks, uploads, readbacks, descriptors, statistics, settings.
// - Retirement by submission frontier (Retire): deferred deletions and descriptor releases are destroyed once every queue
//   has completed what was submitted to it when they were released, whoever submitted it.
// Thread safety: Frontier, Retire and RegisterQueue may be called from any thread. Synchronization across threads is a first
// cut (docs/job-graphs.md, "WIP: cross-thread synchronization and locking").
class Runtime : public std::enable_shared_from_this<Runtime> {
public:
	struct Desc {
		rhi::Device device;
		rhi::Backend backend = rhi::Backend::Null;
		// Capacity of the services still indexed by a ring of slots (upload pages, statistics query ranges): the most
		// executions any graph keeps in flight (GraphHost: frames in flight times epochs). At most kMaxSlots.
		uint32_t slots = 3;
		// Optional; a ThreadPoolTaskService is created when absent.
		std::shared_ptr<runtime::ITaskService> tasks;
		// Optional; passes can declare pipelines by description only with one (Render/PipelineDesc.h). Its first call comes
		// when a graph lowers a pass, so it may rely on host state made after the runtime.
		std::shared_ptr<PipelineCompiler> pipelineCompiler;
	};

	// The most slots a runtime can have. Slot indices are 32-bit everywhere, so this is not a type limit. It is a sanity bound.
	// Each slot costs memory in every slot-indexed table: a range of the statistics query pools (two timestamps per pass), an
	// upload page list, and a region in every LatchBlock a host sizes from GraphHost::FrameSlots.
	static constexpr uint32_t kMaxSlots = 4096;

	static std::shared_ptr<Runtime> Create(Desc desc);
	// The live runtime, for code with none in reach (resource teardown). Null when there is none.
	static Runtime* Current() noexcept;

	Runtime(const Runtime&) = delete;
	Runtime& operator=(const Runtime&) = delete;
	// The caller has joined its CPU work and completed (or abandoned) its GPU work, and destroyed its graphs.
	~Runtime();

	rhi::Device Device() const noexcept { return m_desc.device; }
	rhi::Backend Backend() const noexcept { return m_desc.backend; }
	uint32_t Slots() const noexcept { return m_desc.slots; }

	const std::shared_ptr<runtime::ITaskService>& Tasks() const noexcept { return m_desc.tasks; }
	const std::shared_ptr<runtime::IUploadService>& Uploads() const noexcept { return m_uploads; }
	const std::shared_ptr<runtime::IReadbackService>& Readbacks() const noexcept { return m_readbacks; }
	const std::shared_ptr<runtime::IDescriptorService>& Descriptors() const noexcept { return m_descriptors; }
	const std::shared_ptr<runtime::IStatisticsService>& Statistics() const noexcept { return m_statistics; }
	const std::shared_ptr<runtime::IRenderGraphSettingsService>& Settings() const noexcept { return m_settings; }
	// The pipelines passes declared by description, built for the devices they run on.
	const std::shared_ptr<PipelineLibrary>& Pipelines() const noexcept { return m_pipelines; }

	// Another queue whose submissions can use ORG objects (a peer device's, one made with CreateQueue): the frontier covers it
	// from now on. Registering twice is harmless.
	void RegisterQueue(rhi::Queue queue);
	// The queue is about to be destroyed: its submissions have completed, or are covered by whoever destroys it.
	void UnregisterQueue(rhi::Queue queue);
	// A timeline whose values are given out before the work that signals them reaches a queue ORG submits to: a graph host's
	// shared queue timeline, which a host's own command stream signals for in-stream epochs (DXVK, at its submission). The
	// frontier covers every value given out so far (nextValue - 1), so a deferred deletion or descriptor release waits for
	// that work too. Registering twice is harmless.
	void RegisterAssignedTimeline(std::shared_ptr<rhi::TimelinePtr> fence, std::shared_ptr<const std::atomic<uint64_t>> nextValue);
	// The owner is about to destroy it, after waiting for every value it gave out (then call NoteTimelineDestroyed).
	void UnregisterAssignedTimeline(const std::shared_ptr<rhi::TimelinePtr>& fence);
	// A timeline its owner is about to destroy, after waiting for everything signalled on it (a graph's queue fence, an
	// external client's slot timeline). Points on it read as reached from now on, as an unregistered queue's do: a
	// destroyed timeline would read as never reached.
	void NoteTimelineDestroyed(const rhi::Timeline& timeline);
	// Another device: its queues join the frontier, and with its API given, other devices' work can wait on its timelines
	// (RelayWaits).
	void RegisterDevice(rhi::Device device, rhi::Backend backend = rhi::Backend::Null);
	// A device waits only on its own timelines. For points on timelines of the device `owner` (an rhi::Device::impl, as
	// TimelineDeviceOf names it) that work on `consumer` must wait for: `owner` waits for them in an empty submission on its
	// graphics queue and signals a fence it shares with `consumer`. Returned is that fence's point as `consumer` sees it, with its
	// identity: wait on it instead. Any thread. Throws when either device is not registered with its API, or the two are not one
	// D3D12 and one Vulkan device.
	struct RelayedPoint { uint64_t identity = 0; rhi::TimelinePoint point{}; };
	RelayedPoint RelayWaits(const void* owner, const void* consumer, std::span<const rhi::TimelinePoint> points);

	// A value an external stream signals after work of its own that may use ORG objects: a host's frame timeline signalled
	// in its command stream after the frame's last use (GraphHost::NoteExternalSlotUse). No ORG queue sees that work. The
	// frontier covers the latest value noted per timeline, so a deferred deletion waits for it. The value may still be
	// unsignalled, or not yet submitted: retirement waits until the GPU reaches it. So the stream must signal it (or a later
	// value) eventually, or deferred deletions wait forever. Its owner calls NoteTimelineDestroyed before destroying the
	// timeline. Any thread.
	void NoteExternalPoint(const rhi::Timeline& timeline, uint64_t value);

	// Held by a submitter from taking values on the host's shared timelines until it publishes them (GraphHost::SubmitTicket).
	// Meanwhile the work that signals them may already be queued, and Frontier would miss them: deferred deletions are not
	// sealed until no scope is open. Any thread.
	class AssignmentScope {
	public:
		explicit AssignmentScope(Runtime& runtime) noexcept : m_runtime(&runtime) {
			m_runtime->m_assigning.fetch_add(1, std::memory_order_acq_rel);
		}
		~AssignmentScope() { m_runtime->m_assigning.fetch_sub(1, std::memory_order_acq_rel); }
		AssignmentScope(const AssignmentScope&) = delete;
		AssignmentScope& operator=(const AssignmentScope&) = delete;
	private:
		Runtime* m_runtime;
	};

	// Every registered queue's progress now.
	runtime::SubmissionFrontier Frontier() const;
	// Whether the GPU reached a point of a frontier, including the points of queues unregistered since (reached: they were
	// drained when they went).
	bool Reached(const runtime::SubmittedPoint& point) const;
	// Releases the descriptors and destroys the deferred deletions whose frontier is reached.
	void Retire();
	// The deferred deletions alone, handed back to be destroyed wherever the caller drops them: for a latency-critical thread
	// (the driver frees take the device's memory locks). Descriptor releases wait for the next Retire.
	std::shared_ptr<void> RetireElsewhere();
	// Retire, with every driver free on the runtime's deleter thread instead of the caller's. For a thread whose work the GPU
	// may wait on (a graph host's thread produces the signals of the work it prepares). On Windows a driver free can block
	// until the GPU is done with the memory, and the GPU may be waiting for a signal only that thread will give. The
	// descriptor slots are released here; only the objects behind them go to the deleter.
	void RetireDetached();
	// Hands objects to the runtime's deleter thread, which drops them in the order given. For what RetireElsewhere returned,
	// on a thread that must not block on a driver free. Null is ignored. Any thread.
	void DestroyDetached(std::shared_ptr<void> objects);

	// The process-unique identity of a timeline, what admission and the ledgers name its points by (never 0; below
	// RenderGraph::kFrameWaitTimelineIdentity). Stable for the timeline's life; a timeline created later in its place (the same
	// handle index, another generation) gets a new one.
	uint64_t TimelineIdentity(const rhi::Timeline& timeline);
	// The timeline an identity names, for a submission that waits on another graph's (its packets bind only their own).
	// Invalid for an unknown identity.
	rhi::TimelineHandle TimelineHandleOf(uint64_t identity) const;
	// The device a timeline identity's timeline belongs to, as its rhi::Device::impl (a timeline names its device by it). Null for an
	// unknown identity.
	const void* TimelineDeviceOf(uint64_t identity) const;
	// The resource ledgers every graph's admission on the runtime shares (docs/job-graphs.md, "Phase 2 investigation").
	const std::shared_ptr<persistent::ResourceSequencer>& Sequencer() const noexcept { return m_sequencer; }
	// The schedule cache shared by every graph on this runtime (compiler::ScheduleCache).
	const std::shared_ptr<compiler::ScheduleCache>& Schedules() const noexcept { return m_schedules; }

private:
	explicit Runtime(Desc desc);
	void Initialize();

	Desc m_desc;
	std::shared_ptr<runtime::IUploadService> m_uploads;
	std::shared_ptr<runtime::IReadbackService> m_readbacks;
	std::shared_ptr<runtime::IDescriptorService> m_descriptors;
	std::shared_ptr<runtime::IStatisticsService> m_statistics;
	std::shared_ptr<runtime::IRenderGraphSettingsService> m_settings;
	std::shared_ptr<PipelineLibrary> m_pipelines;
	rhi::TimelinePtr m_readbackFence, m_copyReadbackFence;
	std::shared_ptr<persistent::ResourceSequencer> m_sequencer;
	std::shared_ptr<compiler::ScheduleCache> m_schedules;

	mutable std::mutex m_queuesMutex;
	std::vector<rhi::Queue> m_queues;
	// The progress timelines of unregistered queues (destroyed with them), and timelines noted destroyed: device and handle.
	std::vector<std::pair<void*, rhi::TimelineHandle>> m_retiredTimelines;
	// RegisterAssignedTimeline's timelines and the next value each gives out. Guarded by m_queuesMutex.
	std::vector<std::pair<std::shared_ptr<rhi::TimelinePtr>, std::shared_ptr<const std::atomic<uint64_t>>>> m_assignedTimelines;
	// NoteExternalPoint's latest point per timeline. Guarded by m_queuesMutex.
	std::vector<runtime::SubmittedPoint> m_externalPoints;
	// Open AssignmentScopes.
	std::atomic<uint32_t> m_assigning{0};

	// The thread DestroyDetached and RetireDetached hand objects to.
	class Deleter;
	std::unique_ptr<Deleter> m_deleter;

	struct DeviceRelay;
	std::mutex m_relaysMutex;
	std::vector<std::pair<rhi::Device, rhi::Backend>> m_devices;  // Guarded by m_relaysMutex.
	std::vector<std::unique_ptr<DeviceRelay>> m_relays;            // Guarded by m_relaysMutex.

	mutable std::mutex m_identitiesMutex;
	std::map<std::tuple<void*, uint32_t, uint32_t>, uint64_t> m_identities;
	std::map<uint64_t, rhi::TimelineHandle> m_handles;
	std::map<uint64_t, const void*> m_owners;
	uint64_t m_nextIdentity = 1;
};

} // namespace org
