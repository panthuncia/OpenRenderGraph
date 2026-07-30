#include "Render/RenderGraph/RenderGraphCompileProfile.h"

namespace rg::profile
{
ScopedCompileProfileStep::ScopedCompileProfileStep(const char* stepName)
	: m_callsite(stepName ? stepName : "unnamed", "OpenRenderGraph.Compile")
{
	if (basic_telemetry::Enabled()) {
		m_scope.emplace(m_callsite);
	}
}
}
