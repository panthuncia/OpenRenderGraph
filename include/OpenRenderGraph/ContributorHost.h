#pragma once

#include <OpenRenderGraph/ContributorAPI.h>
#include <OpenRenderGraph/ContributorRegistry.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <rhi.h>

#include "Render/RenderGraph/RenderGraphExtension.h"

namespace org {
class GraphHost;
class Resource;
struct FramePreparationContext;
}

namespace org::contributor {

/**
 * @brief The graph side of the contributor ABI (ContributorAPI.h): the registry, the C table, and the extension that adds
 * every registered contributor's resources and passes to the host's graph.
 *
 * One per process (the C table has no context). The hosting module exports ORG_GetRenderGraphAPI by forwarding to GetAPI,
 * registers CreateExtension() with its GraphHost, and calls Poll once per frame on the graph's owner thread. Destroy it
 * after the graph: contributor packets in flight call into their modules until the graph lets them go.
 */
class ContributorHost {
public:
	/// A point in the host's frame contributors order passes against: `pass` is the graph pass (or the frame's ends,
	/// "__rg_begin__" / "__rg_end__") it stands for.
	struct Anchor { std::string id, displayName, pass; };
	/// One of the host's persistent epochs (RenderGraph::SetPersistentEpochOrder).
	struct Epoch { std::string id, displayName; uint32_t epoch = 0; };
	/// A host resource contributors may access. `resource` is called on the owner thread at each structural build.
	struct Export {
		std::string id, displayName;
		ORGExportDescriptor desc{};  // shape, allowed usages and READABLE/WRITABLE; id and displayName are filled in
		std::function<std::shared_ptr<Resource>()> resource;
	};
	/// What a pass prepares for: the frame (frameIndex, frameSlot, extents, epoch) and the host's per-frame snapshot.
	struct FrameView { ORGFrameInfo frame{}; ORGHostSnapshot snapshot{}; };
	struct Extents { uint32_t renderWidth = 0, renderHeight = 0, outputWidth = 0, outputHeight = 0; };

	struct Desc {
		std::string id, displayName, version;
		rhi::Device device;
		rhi::Backend backend = rhi::Backend::Null;
		std::vector<Anchor> anchors;
		std::vector<Epoch> epochs;
		std::vector<Export> exports;
		/// The bindless layouts contributor pipelines are created against (the pipeline service).
		rhi::PipelineLayoutHandle graphicsLayout{}, computeLayout{};
		/// Owner thread, at each structural build: what render- and output-relative resources are sized against.
		std::function<Extents()> extents;
		/// Preparation thread: the frame the graph prepares for. Without it, frames carry only the generation.
		std::function<FrameView(const FramePreparationContext&)> frame;
	};

	explicit ContributorHost(Desc desc);
	/// Shuts the registry down: every contributor's generations retire and its shutdown callback runs.
	~ContributorHost();
	ContributorHost(const ContributorHost&) = delete;
	ContributorHost& operator=(const ContributorHost&) = delete;

	/// The table ORG_GetRenderGraphAPI returns; ORG_RG_E_RUNTIME_UNAVAILABLE while no host exists.
	static ORGStatus GetAPI(uint32_t version, ORGRenderGraphAPI* out) noexcept;

	/// The extension that adds the contributors' passes (GraphHost::AddExtension's factory; one graph at a time).
	std::unique_ptr<IRenderGraphExtension> CreateExtension();

	/// Owner thread, once per frame: requests a structural build when contributors registered, unregistered or asked
	/// for one, or when the extents render-relative resources were sized against changed.
	void Poll(GraphHost& host);

	ContributorRegistry& Registry() noexcept;

	struct Impl;

private:
	std::shared_ptr<Impl> m_impl;
};

} // namespace org::contributor
