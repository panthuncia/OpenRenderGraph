#include "Render/RenderGraph/RenderGraphCompileProfile.h"

#include <map>
#include <mutex>
#include <string>

namespace org::profile
{
ScopedCompileProfileStep::ScopedCompileProfileStep(const char* stepName)
{
	if (basic_telemetry::Enabled()) {
		// Telemetry retains callsite pointers until the session is exported.
		// A step object is stack-local and usually dies many frames before that.
		// Map nodes keep both the callsite and its name stable across insertions.
		static std::mutex mutex;
		static std::map<std::string, basic_telemetry::Callsite, std::less<>> callsites;
		std::scoped_lock lock(mutex);
		const auto name = stepName ? stepName : "unnamed";
		auto found = callsites.find(name);
		if (found == callsites.end()) {
			found = callsites.try_emplace(name, "", "OpenRenderGraph.Compile").first;
			found->second = basic_telemetry::Callsite(found->first, "OpenRenderGraph.Compile");
		}
		m_scope.emplace(found->second);
	}
}
}
