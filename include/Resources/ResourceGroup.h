#pragma once

#include <atomic>
#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <rhi.h>

#include "Resources/Resource.h"
#include "Resources/GloballyIndexedResource.h"


namespace org {

class ResourceGroup {
public:
    ResourceGroup(const std::string& groupName) : name(groupName) {
    }

	// Groups are filled by producer threads (e.g. the CLod streaming worker adding
	// page slabs) while the graph thread resolves them, so every access is locked
	// and readers receive a snapshot rather than a reference to live storage.
	std::vector<std::shared_ptr<Resource>> GetChildren() const {
		std::lock_guard lock(m_mutex);
		return resources;
	}

	void AddResource(const std::shared_ptr<Resource>& resource) {
		auto id = resource->GetGlobalResourceID();
		std::lock_guard lock(m_mutex);
		if (!resourcesByID.contains(id)) {
			resourcesByID[id] = resource;
			resources.push_back(resource);
			m_contentVersion.fetch_add(1, std::memory_order_release);
		}
	}

	void RemoveResource(const Resource* resource) {
		const auto id = resource->GetGlobalResourceID();
		std::lock_guard lock(m_mutex);
		auto it = resourcesByID.find(id);
		if (it != resourcesByID.end()) {
			const auto& sp = it->second;
			resources.erase(std::remove(resources.begin(), resources.end(), sp), resources.end());
			resourcesByID.erase(it);
			m_contentVersion.fetch_add(1, std::memory_order_release);
		}
	}

	void ClearResources() {
		std::lock_guard lock(m_mutex);
		resources.clear();
		resourcesByID.clear();
		m_contentVersion.fetch_add(1, std::memory_order_release);
	}

	/// Monotonically-increasing version, bumped on every mutation.
	uint64_t GetContentVersion() const { return m_contentVersion.load(std::memory_order_acquire); }

	/// Stable identity of this group as a resolver dependency. Every resolver
	/// wrapping the same group reports the same identity, so declaration caches
	/// and persistent resource groups are shared instead of duplicated per pass.
	std::shared_ptr<const void> DependencyIdentity() const { return m_dependencyIdentity; }

protected:


    std::unordered_map<uint64_t, std::shared_ptr<Resource>> resourcesByID;
	std::vector<std::shared_ptr<Resource>> resources;

	std::string name = "";
	mutable std::mutex m_mutex;
	std::atomic<uint64_t> m_contentVersion{ 1 };
	struct DependencyIdentityTag {};
	std::shared_ptr<const DependencyIdentityTag> m_dependencyIdentity = std::make_shared<const DependencyIdentityTag>();

private:

	std::vector<uint64_t> GetChildIDs() const {
		std::lock_guard lock(m_mutex);
		std::vector<uint64_t> children;
		for (auto& resource : resources) {
			children.push_back(resource->GetGlobalResourceID());
		}
		return children;
	}

};


} // namespace org
