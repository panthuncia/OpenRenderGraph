#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include <spdlog/spdlog.h>

#include "Resources/DynamicResource.h"
#include "Render/FeatureDomainRegistry.h"
#include "Render/ResourceRegistry.h"

class ResourceIndexOrDynamicResource {
public:
	bool isDynamic = false;
	unsigned int index = 0;
	ResourceRegistry::RegistryHandle handle{}; // for dynamic case

	static ResourceIndexOrDynamicResource Static(unsigned idx) {
		ResourceIndexOrDynamicResource r; r.isDynamic = false; r.index = idx; return r;
	}
	static ResourceIndexOrDynamicResource Dynamic(ResourceRegistry::RegistryHandle h) {
		ResourceIndexOrDynamicResource r; r.isDynamic = true; r.handle = h; return r;
	}
};


enum class DescriptorType {
	SRV,
	UAV,
	CBV
};

struct DescriptorAccessor {
	DescriptorType type; // Type of the descriptor (SRV or UAV)
	bool hasSRVViewType = false; // Indicates if a specific SRVViewType is set
	SRVViewType SRVType; // Type of the SRV
	bool hasUAVViewType = false; // Indicates if a specific UAVViewType is set
	UAVViewType UAVType; // Type of the UAV
	unsigned int mip; // Mip level
	unsigned int slice; // Slice index
};

inline bool operator==(const DescriptorAccessor& lhs, const DescriptorAccessor& rhs) {
	return lhs.type == rhs.type
		&& lhs.hasSRVViewType == rhs.hasSRVViewType
		&& (!lhs.hasSRVViewType || lhs.SRVType == rhs.SRVType)
		&& lhs.hasUAVViewType == rhs.hasUAVViewType
		&& (!lhs.hasUAVViewType || lhs.UAVType == rhs.UAVType)
		&& lhs.mip == rhs.mip
		&& lhs.slice == rhs.slice;
}

struct AutoDescriptorRegistration {
	ResourceIdentifier resourceId;
	DescriptorAccessor accessor;
};

inline bool operator==(const AutoDescriptorRegistration& lhs, const AutoDescriptorRegistration& rhs) {
	return lhs.resourceId == rhs.resourceId && lhs.accessor == rhs.accessor;
}

struct ResourceAndAccessor {
	ResourceIndexOrDynamicResource resource;
	DescriptorAccessor accessor; // Accessor for the descriptor
};

class ResourceDescriptorIndexHelper {
public:
	ResourceDescriptorIndexHelper(
		std::shared_ptr<ResourceRegistryView> registryView,
		std::unordered_set<FeatureDomainIdentifier, FeatureDomainIdentifier::Hasher> activeFeatureDomains = {})
		: m_resourceRegistryView(std::move(registryView)),
		  m_activeFeatureDomains(std::move(activeFeatureDomains)) {

	}
	void RegisterDescriptor(const AutoDescriptorRegistration& registration) {
		switch (registration.accessor.type) {
		case DescriptorType::SRV:
			if (registration.accessor.hasSRVViewType) {
				RegisterSRV(registration.accessor.SRVType, registration.resourceId, registration.accessor.mip, registration.accessor.slice);
			}
			else {
				RegisterSRV(registration.resourceId, registration.accessor.mip, registration.accessor.slice);
			}
			break;
		case DescriptorType::UAV:
			if (registration.accessor.hasUAVViewType) {
				RegisterUAV(registration.accessor.UAVType, registration.resourceId, registration.accessor.mip, registration.accessor.slice);
			}
			else {
				RegisterUAV(registration.resourceId, registration.accessor.mip, registration.accessor.slice);
			}
			break;
		case DescriptorType::CBV:
			RegisterCBV(registration.resourceId);
			break;
		default:
			throw std::runtime_error("Unsupported descriptor type");
		}
	}
	void RegisterSRV(SRVViewType type, ResourceIdentifier id, unsigned int mip, unsigned int slice = 0) {
		DescriptorAccessor accessor;
		accessor.type = DescriptorType::SRV;
		accessor.hasSRVViewType = true;
		accessor.SRVType = type;
		accessor.mip = mip;
		accessor.slice = slice;

		auto h = m_resourceRegistryView->RequestHandle(id);
		Resource* res = m_resourceRegistryView->Resolve<Resource>(h);

		auto entry = GetResourceIndexOrDynamicResource(h, res, accessor);
		m_resourceMap[id.hash] = ResourceAndAccessor{ entry, accessor };
	}
	void RegisterSRV(ResourceIdentifier id, unsigned int mip, unsigned int slice = 0) {
		DescriptorAccessor accessor;
		accessor.type = DescriptorType::SRV;
		accessor.hasSRVViewType = false;
		accessor.mip = mip;
		accessor.slice = slice;
		auto h = m_resourceRegistryView->RequestHandle(id);
		Resource* res = m_resourceRegistryView->Resolve<Resource>(h);

		auto entry = GetResourceIndexOrDynamicResource(h, res, accessor);
		m_resourceMap[id.hash] = ResourceAndAccessor{ entry, accessor };
	}
	void RegisterUAV(ResourceIdentifier id, unsigned int mip, unsigned int slice = 0) {
		DescriptorAccessor accessor;
		accessor.type = DescriptorType::UAV;
		accessor.mip = mip;
		accessor.slice = slice;
		auto h = m_resourceRegistryView->RequestHandle(id);
		Resource* res = m_resourceRegistryView->Resolve<Resource>(h);

		auto entry = GetResourceIndexOrDynamicResource(h, res, accessor);
		m_resourceMap[id.hash] = ResourceAndAccessor{ entry, accessor };
	}
	void RegisterUAV(UAVViewType type, ResourceIdentifier id, unsigned int mip, unsigned int slice = 0) {
		DescriptorAccessor accessor;
		accessor.type = DescriptorType::UAV;
		accessor.hasUAVViewType = true;
		accessor.UAVType = type;
		accessor.mip = mip;
		accessor.slice = slice;
		auto h = m_resourceRegistryView->RequestHandle(id);
		Resource* res = m_resourceRegistryView->Resolve<Resource>(h);

		auto entry = GetResourceIndexOrDynamicResource(h, res, accessor);
		m_resourceMap[id.hash] = ResourceAndAccessor{ entry, accessor };
	}
	void RegisterCBV(ResourceIdentifier id) {
		DescriptorAccessor accessor;
		accessor.type = DescriptorType::CBV;
		auto h = m_resourceRegistryView->RequestHandle(id);
		Resource* res = m_resourceRegistryView->Resolve<Resource>(h);

		auto entry = GetResourceIndexOrDynamicResource(h, res, accessor);
		m_resourceMap[id.hash] = ResourceAndAccessor{ entry, accessor };
	}
	unsigned int GetResourceDescriptorIndex(size_t hash, bool allowFail = true, const std::string* name = nullptr) const {
		auto it = m_resourceMap.find(hash);
		if (it == m_resourceMap.end()) {
			if (allowFail) {
				return (std::numeric_limits<unsigned int>().max)(); // Return max value if the resource is not found and allowFail is true
			}
			if (name && ShouldAllowMissingForInactiveFeature(ResourceIdentifier{ *name })) {
				return (std::numeric_limits<unsigned int>().max)();
			}
			std::string resourceName = name ? *name : "Unknown";
			throw std::runtime_error("Resource "+ resourceName +" not found!");
		}
		const auto& resourceAndAccessor = it->second;
		unsigned int resolvedIndex = 0;
		if (resourceAndAccessor.resource.isDynamic) {
			resolvedIndex = AccessResourceByHandle(resourceAndAccessor.resource.handle, resourceAndAccessor.accessor);
		}
		else {
			resolvedIndex = resourceAndAccessor.resource.index;
		}

		auto lastIt = m_lastResolvedDescriptorIndices.find(hash);
		if (lastIt == m_lastResolvedDescriptorIndices.end()) {
			m_lastResolvedDescriptorIndices.emplace(hash, resolvedIndex);
		}
		else if (lastIt->second != resolvedIndex) {
			spdlog::warn(
				"ResourceDescriptorIndexHelper: descriptor index changed for '{}' old={} new={}; using refreshed bind-time index",
				name ? *name : std::string("Unknown"),
				lastIt->second,
				resolvedIndex);
			lastIt->second = resolvedIndex;
		}

		return resolvedIndex;
	}
	unsigned int GetResourceDescriptorIndex(const ResourceIdentifier& id, bool allowFail = true) const {
		return GetResourceDescriptorIndex(id.hash, allowFail, &id.name);
	}

	void SetActiveFeatureDomains(std::unordered_set<FeatureDomainIdentifier, FeatureDomainIdentifier::Hasher> activeFeatureDomains) {
		m_activeFeatureDomains = std::move(activeFeatureDomains);
	}
private:
	std::unordered_map<size_t, ResourceAndAccessor> m_resourceMap; // Maps resource identifiers to descriptor indices
	std::unordered_set<FeatureDomainIdentifier, FeatureDomainIdentifier::Hasher> m_activeFeatureDomains;
	mutable std::unordered_map<size_t, unsigned int> m_lastResolvedDescriptorIndices;

	bool ShouldAllowMissingForInactiveFeature(const ResourceIdentifier& id) const {
		auto domain = FeatureDomainRegistry::Get().FindResourceDomain(id);
		return domain.has_value() && !m_activeFeatureDomains.contains(*domain);
	}

	unsigned int AccessGloballyIndexedResource(const std::shared_ptr<GloballyIndexedResource> resource, const DescriptorAccessor& accessor) const {
		switch (accessor.type) {
		case DescriptorType::SRV:
			if (accessor.hasSRVViewType) {
				return resource->GetSRVInfo(accessor.SRVType, accessor.mip, accessor.slice).slot.index;
			}
			else {
				return resource->GetSRVInfo(accessor.mip, accessor.slice).slot.index;
			}
		case DescriptorType::UAV:
			if (accessor.hasUAVViewType) {
				return resource->GetUAVShaderVisibleInfo(accessor.UAVType, accessor.mip, accessor.slice).slot.index;
			}
			return resource->GetUAVShaderVisibleInfo(accessor.mip, accessor.slice).slot.index;
		case DescriptorType::CBV:
			return resource->GetCBVInfo().slot.index;
		default:
			throw std::runtime_error("Unsupported descriptor type");
		}
	}

	unsigned int AccessGloballyIndexedResource(
		const GloballyIndexedResource& resource,
		const DescriptorAccessor& accessor) const
	{
		switch (accessor.type) {
		case DescriptorType::SRV:
			if (accessor.hasSRVViewType) {
				return resource.GetSRVInfo(accessor.SRVType, accessor.mip, accessor.slice).slot.index;
			}
			else {
				return resource.GetSRVInfo(accessor.mip, accessor.slice).slot.index;
			}

		case DescriptorType::UAV:
			if (accessor.hasUAVViewType) {
				return resource.GetUAVShaderVisibleInfo(accessor.UAVType, accessor.mip, accessor.slice).slot.index;
			}
			return resource.GetUAVShaderVisibleInfo(accessor.mip, accessor.slice).slot.index;

		case DescriptorType::CBV:
			return resource.GetCBVInfo().slot.index;

		default:
			throw std::runtime_error("Unsupported descriptor type");
		}
	}

	template<class T>
	static T* PtrFrom(T* p) noexcept { return p; }

	template<class T>
	static T* PtrFrom(const std::shared_ptr<T>& p) noexcept { return p.get(); }

	unsigned int AccessResourceByHandle(
		const ResourceRegistry::RegistryHandle& h,
		const DescriptorAccessor& accessor) const
	{
		Resource* base = m_resourceRegistryView->Resolve<Resource>(h);
		if (!base) {
			throw std::runtime_error("Resource descriptor handle no longer resolves");
		}

		if (auto* dyn = dynamic_cast<DynamicGloballyIndexedResource*>(base)) {
			auto backing = dyn->GetResource();
			auto* gi = PtrFrom(backing);
			if (!gi) {
				throw std::runtime_error("Dynamic resource has null backing resource");
			}

			return AccessGloballyIndexedResource(*gi, accessor);
		}

		if (auto* gi = dynamic_cast<GloballyIndexedResource*>(base)) {
			return AccessGloballyIndexedResource(*gi, accessor);
		}

		throw std::runtime_error(
			"Resource descriptor handle does not resolve to a GloballyIndexedResource or DynamicGloballyIndexedResource");
	}


	ResourceIndexOrDynamicResource GetResourceIndexOrDynamicResource(
		const ResourceRegistry::RegistryHandle& h,
		Resource* resource,
		const DescriptorAccessor& accessor) const
	{
		if (!resource) {
			throw std::runtime_error("Resource is null");
		}

		if (dynamic_cast<DynamicGloballyIndexedResource*>(resource) ||
			dynamic_cast<GloballyIndexedResource*>(resource)) {
			// Descriptor slots for ordinary globally indexed resources can change
			// when dynamic buffers grow. Re-resolve by handle at bind time so
			// long-lived passes do not keep stale descriptor root constants until
			// the next render-graph rebuild.
			return ResourceIndexOrDynamicResource::Dynamic(h);
		}

		throw std::runtime_error(
			"Resource is not a GloballyIndexedResource or DynamicGloballyIndexedResource");
	}


	std::shared_ptr<ResourceRegistryView> m_resourceRegistryView;
};
