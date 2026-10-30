#pragma once

#include <cstdint>
#include <map>
#include <tuple>
#include <memory>
#include <mutex>
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

namespace org {

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
		// executions any graph keeps in flight (GraphHost: frames in flight times epochs). At most 255.
		uint32_t slots = 3;
		// Optional; a ThreadPoolTaskService is created when absent.
		std::shared_ptr<runtime::ITaskService> tasks;
	};

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

	// Another queue whose submissions can use ORG objects (a peer device's, one made with CreateQueue): the frontier covers it
	// from now on. Registering twice is harmless.
	void RegisterQueue(rhi::Queue queue);
	// The queue is about to be destroyed: its submissions have completed, or are covered by whoever destroys it.
	void UnregisterQueue(rhi::Queue queue);
	void RegisterDevice(rhi::Device device);

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

	// The process-unique identity of a timeline, what admission and the ledgers name its points by (never 0; below
	// RenderGraph::kFrameWaitTimelineIdentity). Stable for the timeline's life; a timeline created later in its place (the same
	// handle index, another generation) gets a new one.
	uint64_t TimelineIdentity(const rhi::Timeline& timeline);
	// The timeline an identity names, for a submission that waits on another graph's (its packets bind only their own).
	// Invalid for an unknown identity.
	rhi::TimelineHandle TimelineHandleOf(uint64_t identity) const;
	// The resource ledgers every graph's admission on the runtime shares (docs/job-graphs.md, "Phase 2 investigation").
	const std::shared_ptr<persistent::ResourceSequencer>& Sequencer() const noexcept { return m_sequencer; }

private:
	explicit Runtime(Desc desc);
	void Initialize();

	Desc m_desc;
	std::shared_ptr<runtime::IUploadService> m_uploads;
	std::shared_ptr<runtime::IReadbackService> m_readbacks;
	std::shared_ptr<runtime::IDescriptorService> m_descriptors;
	std::shared_ptr<runtime::IStatisticsService> m_statistics;
	std::shared_ptr<runtime::IRenderGraphSettingsService> m_settings;
	rhi::TimelinePtr m_readbackFence, m_copyReadbackFence;
	std::shared_ptr<persistent::ResourceSequencer> m_sequencer;

	mutable std::mutex m_queuesMutex;
	std::vector<rhi::Queue> m_queues;
	// The progress timelines of unregistered queues (destroyed with them): device and handle.
	std::vector<std::pair<void*, rhi::TimelineHandle>> m_retiredTimelines;

	mutable std::mutex m_identitiesMutex;
	std::map<std::tuple<void*, uint32_t, uint32_t>, uint64_t> m_identities;
	std::map<uint64_t, rhi::TimelineHandle> m_handles;
	uint64_t m_nextIdentity = 1;
};

} // namespace org
