#pragma once

#include <Render/RenderGraph/RenderGraph.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// In-tree contributors deliberately use ORG's C++ API directly. This registry
// supplies only cross-module policy and lifetime; it does not wrap native
// passes in another callback abstraction.
namespace org::contributor {

class ExtensionRegistry
{
public:
	enum class Kind : uint8_t { Required, Optional, Diagnostic };
	using Handle = uint64_t;
	using Factory = std::function<std::unique_ptr<org::RenderGraph::IRenderGraphExtension>()>;
	using GenerationCallback = std::function<void(uint64_t)>;
	using DiagnosticCallback = std::function<void(std::string_view)>;

	struct Descriptor
	{
		std::string id;
		Kind kind{ Kind::Required };
		std::vector<std::string> exportedResources;
		std::vector<std::string> requiredImports;
		Factory factory;
		GenerationCallback activated;
		GenerationCallback retired;
		DiagnosticCallback diagnostic;
	};

	struct Installed
	{
		std::string id;
		std::unique_ptr<org::RenderGraph::IRenderGraphExtension> extension;
		GenerationCallback activated;
		GenerationCallback retired;
	};

	ExtensionRegistry() = default;
	Handle Register(Descriptor descriptor);
	void BeginUnregister(Handle handle);
	std::vector<Installed> BuildCandidate(const std::unordered_set<std::string>& externalResources) const;

private:
	struct Entry { Handle handle{}; Descriptor descriptor; bool unregistering{}; };
	mutable std::mutex mutex_;
	std::unordered_map<Handle, Entry> entries_;
	Handle nextHandle_{ 1 };
};

} // namespace org::contributor
