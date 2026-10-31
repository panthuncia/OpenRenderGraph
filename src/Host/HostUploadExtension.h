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

} // namespace org::detail
