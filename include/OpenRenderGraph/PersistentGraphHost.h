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
	bool m_rebuildRequested = true;
	uint64_t m_frameNumber = 0;
};

} // namespace org
