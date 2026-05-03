#pragma once

#include <memory>
#include <span>
#include <vector>

#include "Resources/ResourceStateTracker.h"
#include "Render/ResourceRegistry.h"

class Resource;

struct ResourceHandleAndRange {
	ResourceHandleAndRange() : resource({}) {}
    ResourceHandleAndRange(ResourceRegistry::RegistryHandle resource) : resource(resource) {}
	ResourceHandleAndRange(ResourceRegistry::RegistryHandle resource, const RangeSpec& range) : resource(resource), range(range) {}
    ResourceRegistry::RegistryHandle resource;
    RangeSpec range;

	RG_DEFINE_PASS_INPUTS(ResourceHandleAndRange, &ResourceHandleAndRange::resource, &ResourceHandleAndRange::range);
};

struct ResourcePtrAndRange {
	ResourcePtrAndRange(std::shared_ptr<Resource> resource) : resource(resource) {}
	ResourcePtrAndRange(std::shared_ptr<Resource> resource, const RangeSpec& range) : resource(resource), range(range) {}
	std::shared_ptr<Resource> resource;
	RangeSpec range;
};

struct ResourceRequirement {
	ResourceRequirement(const ResourceHandleAndRange& resourceAndRange)
		: resourceHandleAndRange(resourceAndRange) {
	}
	ResourceHandleAndRange resourceHandleAndRange;    // resource and range
    ResourceState state;
};

template<class PassResourceData>
void ClearImmediateFrameRequirements(PassResourceData& resources) {
	resources.frameResourceRequirements.clear();
	resources.mergedFrameRequirementsDirty = true;
}

template<class PassResourceData>
void SetImmediateFrameRequirements(PassResourceData& resources, std::vector<ResourceRequirement>&& requirements) {
	resources.frameResourceRequirements = std::move(requirements);
	resources.mergedFrameRequirementsDirty = true;
}

template<class PassResourceData>
size_t GetFrameRequirementCount(const PassResourceData& resources) {
	return resources.staticResourceRequirements.size() + resources.frameResourceRequirements.size();
}

template<class PassResourceData, class Fn>
void ForEachFrameRequirement(PassResourceData& resources, Fn&& fn) {
	for (auto& req : resources.staticResourceRequirements) {
		fn(req);
	}
	for (auto& req : resources.frameResourceRequirements) {
		fn(req);
	}
}

template<class PassResourceData, class Fn>
void ForEachFrameRequirement(const PassResourceData& resources, Fn&& fn) {
	for (const auto& req : resources.staticResourceRequirements) {
		fn(req);
	}
	for (const auto& req : resources.frameResourceRequirements) {
		fn(req);
	}
}

template<class PassResourceData>
std::span<const ResourceRequirement> GetFrameRequirementsSpan(const PassResourceData& resources) {
	if (resources.frameResourceRequirements.empty()) {
		return resources.staticResourceRequirements;
	}
	if (resources.staticResourceRequirements.empty()) {
		return resources.frameResourceRequirements;
	}
	if (resources.mergedFrameRequirementsDirty) {
		auto& merged = resources.mergedFrameResourceRequirements;
		merged.clear();
		merged.reserve(resources.staticResourceRequirements.size() + resources.frameResourceRequirements.size());
		merged.insert(merged.end(), resources.staticResourceRequirements.begin(), resources.staticResourceRequirements.end());
		merged.insert(merged.end(), resources.frameResourceRequirements.begin(), resources.frameResourceRequirements.end());
		resources.mergedFrameRequirementsDirty = false;
	}
	return resources.mergedFrameResourceRequirements;
}