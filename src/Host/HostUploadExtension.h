#pragma once

// The upload pass every host adds to its graphs (GraphHost, GraphInstance). Private to src/Host.
#include "Render/RenderGraph/RenderGraph.h"
#include "Render/Runtime/IUploadService.h"

#include <vector>

namespace org::detail {

inline constexpr const char* kUploadPassName = "org.host.uploads";

// The built-in upload pass: graph-ordered buffer/texture uploads requested with
// BUFFER_UPLOAD and friends. It records immediate-mode copies, so it is submitted
// through the graph's external client ahead of the main execution.
class HostUploadExtension final : public RenderGraph::IRenderGraphExtension {
public:
	// deferResolveContext: a graph built off the owner thread (GraphHost::BuildEpochGraph). The shared upload service is pointed at
	// its registry only by ApplyResolveContext, once the graph is attached: until then the running graphs' uploads resolve as before.
	explicit HostUploadExtension(runtime::IUploadService* uploads, bool deferResolveContext = false)
		: m_uploads(uploads), m_deferResolveContext(deferResolveContext) {}
	void OnRegistryReset(ResourceRegistry* registry) override {
		m_registry = registry;
		if (m_uploads && !m_deferResolveContext) m_uploads->SetUploadResolveContext({registry, 0});
	}
	// Owner thread: the shared upload service resolves registry handles against this graph's registry.
	void ApplyResolveContext() {
		if (m_uploads && m_registry) m_uploads->SetUploadResolveContext({m_registry, 0});
	}
	void GatherStructuralPasses(RenderGraph&, std::vector<RenderGraph::ExternalPassDesc>& out) override {
		if (!m_uploads) return;
		if (auto pass = m_uploads->GetUploadPass())
			out.push_back(RenderGraph::ExternalPassDesc::Render(kUploadPassName, std::move(pass))
				.At(RenderGraph::ExternalInsertPoint::Begin(-1000)).CollectStatistics(false));
	}
private:
	runtime::IUploadService* m_uploads;
	ResourceRegistry* m_registry = nullptr;
	bool m_deferResolveContext = false;
};

} // namespace org::detail
