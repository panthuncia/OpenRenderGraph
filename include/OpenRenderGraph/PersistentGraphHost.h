#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <rhi.h>

#include "Render/RenderGraph/RenderGraph.h"

namespace org::runtime {
class ITaskService;
class IUploadService;
}

namespace org {

struct IHostExecutionData;

// Drives one RenderGraph in persistent (static) execution for an application that
// embeds ORG in its own renderer, for example a host that adopted another API's
// device. It owns what every such embedding needs and nothing application-specific:
//   - the runtime device registration and a task service,
//   - the built-in upload pass (persistent Pre segment),
//   - persistent execution with the requested external queue boundary,
//   - rebuilding the graph from the registered extension factories on request,
//   - synchronous per-frame Update + Execute (Execute submits before returning),
//   - per-frame-slot retirement: before a slot is reused, its previous frame's GPU work is
//     waited for and the upload pages and descriptors it retired are released,
//   - an orderly shutdown that waits only for the graph's own work.
// Not thread-safe: all calls come from the host's render thread. With async epochs (SetAsyncEpochs) the host
// runs one thread of its own that owns the graph from then on; the render thread only submits.
class PersistentGraphHost {
public:
	struct Desc {
		rhi::Device device;
		rhi::Backend backend = rhi::Backend::Null;
		// Optional; a ThreadPoolTaskService is created when absent.
		std::shared_ptr<runtime::ITaskService> tasks;
		RenderGraph::ExternalQueueBoundary queueBoundary{};
		uint32_t framesInFlight = 3;
		// Host epochs: the order the host runs them in within its own frame (see
		// RenderGraph::SetPersistentEpochOrder). Passes declare theirs with
		// ExternalPassDesc::Epoch; ExecuteFrame names the one to run.
		std::vector<uint32_t> epochOrder;
		// RenderGraph::SetPersistentClosedExecutions: every execution leaves its resources in their home
		// states and starts with a full barrier, so its admission is independent of what ran before.
		bool closedExecutions = false;
	};

	using ExtensionFactory = std::function<std::unique_ptr<RenderGraph::IRenderGraphExtension>()>;

	explicit PersistentGraphHost(Desc desc);
	~PersistentGraphHost();
	PersistentGraphHost(const PersistentGraphHost&) = delete;
	PersistentGraphHost& operator=(const PersistentGraphHost&) = delete;

	// Extensions are instantiated from their factories each time the graph is
	// (re)built, so they may size resources from current host state.
	void AddExtension(std::string id, ExtensionFactory factory);
	void RemoveExtension(const std::string& id);

	// The next ExecuteFrame builds a new graph (after retiring the current one).
	void RequestRebuild() noexcept { m_rebuildRequested = true; }

	// Runs after any rebuild and before the frame is prepared, with the graph's upload
	// and descriptor services active: the place to queue this frame's uploads
	// (BUFFER_UPLOAD) into graph resources.
	using FrameCallback = std::function<void(RenderGraph&)>;

	// Called from ExecuteFrame once an earlier frame is known complete on the GPU (its slot is about to be
	// reused), after that frame's pass timestamps were read back: the passes whose gpuSampleSerial equals
	// stats.GetFrameSerial() are the ones that frame recorded. frameNumber is that frame's 0-based number.
	using CompletedFrameCallback = std::function<void(uint64_t frameNumber, const runtime::IStatisticsService& stats)>;
	void SetCompletedFrameCallback(CompletedFrameCallback callback) { m_completedFrame = std::move(callback); }

	// Builds if needed, then prepares and submits one frame. hostData is visible to
	// passes through PassPrepareContext::preparationData for this frame only.
	// With an epoch, only that epoch's passes (and untagged ones) are prepared,
	// admitted, recorded and submitted: the graph was compiled once for the whole
	// host frame, and this is one execution split of it.
	void ExecuteFrame(const IHostExecutionData* hostData = nullptr, const FrameCallback& beforePrepare = {},
		uint32_t epoch = UINT32_MAX);  // persistent::AllEpochs

	// Retires the current graph: stops frame production and waits for its work.
	void DestroyGraph();

	// Async epochs. The host prepares each listed epoch's execution ahead of time, as a ticket
	// (RenderGraph::PreparePersistentTicket), on a thread of its own - which from then on is the only thread
	// that touches the graph: it waits for frame slots, prepares, admits and records tickets, completes them
	// in submission order and retires finished work. The render thread calls SubmitEpoch instead of
	// ExecuteFrame. Requires Desc::closedExecutions. Empty stops it (and waits for the thread); a rebuild
	// stops and restarts it.
	void SetAsyncEpochs(std::vector<uint32_t> epochs);
	bool AsyncEpochs() const noexcept { return static_cast<bool>(m_async); }
	// Submits the epoch at the current point of the host's queue stream. Waits for its ticket if the host's
	// thread has not finished it; runs beforeSubmit with the graph's services active and CurrentFrameSlot()
	// the ticket's slot (write latches and queue uploads there); has the ticket prepared again (and waits)
	// if a pass would now prepare differently; records the uploads queued since the last submission; and
	// hands both to the queue - timeline values are assigned here, in submission order. Throws when the
	// graph failed (on either thread).
	void SubmitEpoch(uint32_t epoch, const FrameCallback& beforeSubmit = {});
	// SubmitEpoch counters since the last call.
	struct AsyncStats {
		uint64_t submitted = 0;
		uint64_t waited = 0;   // the ticket was not ready yet
		uint64_t stale = 0;    // prepared again after beforeSubmit
		uint64_t uploads = 0;  // submissions that carried uploads
	};
	AsyncStats TakeAsyncStats() noexcept { return std::exchange(m_asyncStats, {}); }

	RenderGraph* Graph() noexcept { return m_graph.get(); }
	runtime::IUploadService* Uploads() noexcept;
	uint64_t FramesExecuted() const noexcept { return m_frameNumber; }
	// The frame slot of the execution ExecuteFrame is running (valid from its beforePrepare callback until
	// it returns): the region of a LatchBlock the host may write for it. Also RecordingContext::FrameSlot.
	bool ClosedExecutions() const noexcept { return m_desc.closedExecutions; }
	uint32_t CurrentFrameSlot() const noexcept {
		return m_async ? m_asyncSlot : static_cast<uint32_t>(m_frameNumber % (std::max)(m_desc.framesInFlight, 1u));
	}
	// The host frame number of the last execution (ExecuteFrame or SubmitEpoch); the completed-frame
	// callback reports frames by it.
	uint64_t LastHostFrame() const noexcept { return m_lastHostFrame; }
	uint32_t FrameSlots() const noexcept { return (std::max)(m_desc.framesInFlight, 1u); }

	// The calling thread's CPU time in the last ExecuteFrame, by phase. Diagnostics only.
	struct FrameTimings {
		double buildUs = 0.0;          // a requested rebuild
		double waitUs = 0.0;           // the host wait for the slot's previous frame, and its completion callback
		double releaseUs = 0.0;        // deferred upload and descriptor releases
		double beforePrepareUs = 0.0;  // the host's callback (its uploads and inputs)
		double updateUs = 0.0;         // RenderGraph::Update: pass updates and frame preparation
		double executeUs = 0.0;        // RenderGraph::Execute: admission, recording, submission
		double signalUs = 0.0;         // recording the frame's completion points
		// SubmitEpoch (async epochs) fills these instead.
		bool async = false;
		double ticketWaitUs = 0.0;     // waiting for the ticket (zero when it was ready)
		double checkUs = 0.0;          // checking it is current, and preparing it again when not
		double uploadsUs = 0.0;        // recording the queued uploads
		double submitUs = 0.0;         // handing the uploads and the ticket to the queue
		RenderGraph::PersistentExecuteTimings execute{};
	};
	const FrameTimings& LastFrameTimings() const noexcept { return m_lastTimings; }
	const Desc& GetDesc() const noexcept { return m_desc; }

private:
	void Build();

	Desc m_desc;
	struct Registered {
		std::string id;
		ExtensionFactory factory;
	};
	std::vector<Registered> m_extensions;
	std::unique_ptr<RenderGraph> m_graph;
	// A slot's last frame: the value each queue's own timeline reached with that frame's last batch.
	// Waiting on those gates the slot's reuse; the host adds no submission of its own. Queue timelines
	// belong to the graph, so a rebuild (which waits for the device) clears them.
	struct SlotCompletion {
		std::vector<std::pair<rhi::Timeline*, uint64_t>> points;
		uint64_t frameNumber = 0;
		bool pending = false;
	};
	std::vector<SlotCompletion> m_slotCompletions;
	bool m_rebuildRequested = true;
	uint64_t m_frameNumber = 0;
	CompletedFrameCallback m_completedFrame;
	FrameTimings m_lastTimings{};
	uint64_t m_lastHostFrame = 0;
	struct Async;
	std::unique_ptr<Async> m_async;
	uint32_t m_asyncSlot = 0;
	std::vector<uint32_t> m_asyncEpochs;  // restarted with these after a rebuild
	AsyncStats m_asyncStats{};
	void StartAsync();
	void StopAsync();
	void PrepareTicket(Async& state, uint32_t epochIndex, int32_t requestedSlot);
};

} // namespace org
