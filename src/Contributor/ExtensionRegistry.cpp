#include <OpenRenderGraph/ExtensionRegistry.h>

#include <algorithm>
#include <stdexcept>

namespace org::contributor {

ExtensionRegistry::Handle ExtensionRegistry::Register(Descriptor descriptor)
{
	if (descriptor.id.empty() || !descriptor.factory)
		throw std::invalid_argument("Native render-graph contributors require an ID and factory");
	std::scoped_lock lock(mutex_);
	for (const auto& [_, entry] : entries_)
		if (!entry.unregistering && entry.descriptor.id == descriptor.id)
			throw std::invalid_argument("Duplicate native render-graph contributor ID: " + descriptor.id);
	const auto handle = nextHandle_++;
	entries_.emplace(handle, Entry{ handle, std::move(descriptor), false });
	return handle;
}

void ExtensionRegistry::BeginUnregister(Handle handle)
{
	std::scoped_lock lock(mutex_);
	if (auto found = entries_.find(handle); found != entries_.end()) found->second.unregistering = true;
}

std::vector<ExtensionRegistry::Installed> ExtensionRegistry::BuildCandidate(
	const std::unordered_set<std::string>& externalResources) const
{
	std::vector<Descriptor> descriptors;
	{
		std::scoped_lock lock(mutex_);
		for (const auto& [_, entry] : entries_)
			if (!entry.unregistering) descriptors.push_back(entry.descriptor);
	}
	std::ranges::sort(descriptors, {}, &Descriptor::id);

	std::unordered_map<std::string, size_t> exporters;
	for (size_t i = 0; i < descriptors.size(); ++i) {
		for (const auto& resource : descriptors[i].exportedResources) {
			if (resource.empty() || externalResources.contains(resource) ||
				!exporters.emplace(resource, i).second)
				throw std::runtime_error("Duplicate or empty native symbolic export: " + resource);
		}
	}

	std::vector<bool> selected(descriptors.size(), true);
	bool changed = true;
	while (changed) {
		changed = false;
		for (size_t i = 0; i < descriptors.size(); ++i) {
			if (!selected[i]) continue;
			for (const auto& imported : descriptors[i].requiredImports) {
				const auto provider = exporters.find(imported);
				const bool available = externalResources.contains(imported) ||
					(provider != exporters.end() && selected[provider->second]);
				if (available) continue;
				if (descriptors[i].kind == Kind::Required)
					throw std::runtime_error("Required native contributor '" + descriptors[i].id +
						"' is missing symbolic resource '" + imported + "'");
				selected[i] = false;
				changed = true;
				if (descriptors[i].diagnostic)
					try { descriptors[i].diagnostic("omitted: missing symbolic resource " + imported); } catch (...) {}
				break;
			}
		}
	}

	std::vector<std::unique_ptr<org::RenderGraph::IRenderGraphExtension>> extensions(descriptors.size());
	for (size_t i = 0; i < descriptors.size(); ++i) {
		if (!selected[i]) continue;
		try {
			extensions[i] = descriptors[i].factory();
			if (!extensions[i]) throw std::runtime_error("factory returned null");
		} catch (const std::exception& error) {
			if (descriptors[i].diagnostic) try { descriptors[i].diagnostic(error.what()); } catch (...) {}
			if (descriptors[i].kind == Kind::Required) throw;
			selected[i] = false;
		}
	}

	// A factory failure removes that contributor's exports. Prune optional
	// consumers transitively and reject any required consumer, just as the C ABI
	// candidate builder does. Registration order never participates.
	changed = true;
	while (changed) {
		changed = false;
		for (size_t i = 0; i < descriptors.size(); ++i) {
			if (!selected[i]) continue;
			for (const auto& imported : descriptors[i].requiredImports) {
				const auto provider = exporters.find(imported);
				const bool available = externalResources.contains(imported) ||
					(provider != exporters.end() && selected[provider->second] && extensions[provider->second]);
				if (available) continue;
				if (descriptors[i].kind == Kind::Required)
					throw std::runtime_error("Required native contributor '" + descriptors[i].id +
						"' lost symbolic resource '" + imported + "' while preparing the candidate");
				selected[i] = false;
				extensions[i].reset();
				changed = true;
				if (descriptors[i].diagnostic) try {
					descriptors[i].diagnostic("omitted: provider failed for symbolic resource " + imported);
				} catch (...) {}
				break;
			}
		}
	}

	std::vector<Installed> installed;
	for (size_t i = 0; i < descriptors.size(); ++i) {
		if (selected[i] && extensions[i])
			installed.push_back({ descriptors[i].id, std::move(extensions[i]),
				descriptors[i].activated, descriptors[i].retired });
	}
	return installed;
}

} // namespace org::contributor
