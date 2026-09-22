#pragma once

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
// Not thread-safe: all calls come from the host's render thread.
class PersistentGraphHost {
public:
	struct Desc {
		rhi::Device device;
		rhi::Backend backend = rhi::Backend::Null;
		// Optional; a ThreadPoolTaskService is created when absent.
		std::shared_ptr<runtime::ITaskService> tasks;
		RenderGraph::ExternalQueueBoundary queueBoundary{};
		uint32_t framesInFlight = 3;
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
	void ExecuteFrame(const IHostExecutionData* hostData = nullptr, const FrameCallback& beforePrepare = {});

	// Retires the current graph: stops frame production and waits for its work.
	void DestroyGraph();

	RenderGraph* Graph() noexcept { return m_graph.get(); }
	runtime::IUploadService* Uploads() noexcept;
	uint64_t FramesExecuted() const noexcept { return m_frameNumber; }
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
	// Signalled on the graphics queue after each frame; a slot's last value gates its reuse.
	rhi::TimelinePtr m_frameTimeline;
	std::vector<uint64_t> m_slotFrameValues;
	bool m_rebuildRequested = true;
	uint64_t m_frameNumber = 0;
	CompletedFrameCallback m_completedFrame;
};

} // namespace org
