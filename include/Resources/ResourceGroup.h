#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <rhi.h>

#include "Resources/Resource.h"
#include "Resources/GloballyIndexedResource.h"


namespace org {

class ResourceGroup {
public:
    ResourceGroup(const std::string& groupName) : name(groupName) {
    }

	const std::vector<std::shared_ptr<Resource>>& GetChildren() const {
		return resources;
	}

	void AddResource(const std::shared_ptr<Resource>& resource) {
		auto id = resource->GetGlobalResourceID();
		if (!resourcesByID.contains(id)) {
			resourcesByID[resource->GetGlobalResourceID()] = resource;
			resources.push_back(resource);
			++m_contentVersion;
		}
	}

	void RemoveResource(const Resource* resource) {
		const auto id = resource->GetGlobalResourceID();
		auto it = resourcesByID.find(id);
		if (it != resourcesByID.end()) {
			const auto& sp = it->second;
			resources.erase(std::remove(resources.begin(), resources.end(), sp), resources.end());
			resourcesByID.erase(it);
			++m_contentVersion;
		}
	}

	void ClearResources() {
		resources.clear();
		resourcesByID.clear();
		++m_contentVersion;
	}

	/// Monotonically-increasing version, bumped on every mutation.
	uint64_t GetContentVersion() const { return m_contentVersion; }

	/// Stable identity of this group as a resolver dependency. Every resolver
	/// wrapping the same group reports the same identity, so declaration caches
	/// and persistent resource groups are shared instead of duplicated per pass.
	std::shared_ptr<const void> DependencyIdentity() const { return m_dependencyIdentity; }

protected:


    std::unordered_map<uint64_t, std::shared_ptr<Resource>> resourcesByID;
	std::vector<std::shared_ptr<Resource>> resources;

	std::string name = "";
	uint64_t m_contentVersion = 1;
	struct DependencyIdentityTag {};
	std::shared_ptr<const DependencyIdentityTag> m_dependencyIdentity = std::make_shared<const DependencyIdentityTag>();

private:

	std::vector<uint64_t> GetChildIDs() const {
		std::vector<uint64_t> children;
		for (auto& resource : resources) {
			children.push_back(resource->GetGlobalResourceID());
		}
		return children;
	}

};


} // namespace org
