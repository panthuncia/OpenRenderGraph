#pragma once

#include <memory>
#include <span>
#include <vector>

#include "Resources/ResourceStateTracker.h"
#include "Render/ResourceRegistry.h"


namespace org {

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

struct ResolverRequirementBlock {
	std::shared_ptr<const void> dependencyIdentity;
	uint64_t resourceSetIdentityLow = 0;
	uint64_t resourceSetIdentityHigh = 0;
	uint64_t registryGeneration = 0;
	uint64_t bindingHash = 0;
	std::shared_ptr<const std::vector<std::shared_ptr<Resource>>> resourceOwnership;
	std::vector<ResourceRequirement> requirements;
	uint64_t membershipHash = 0;
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
	size_t count = resources.staticResourceRequirements.size() + resources.frameResourceRequirements.size();
	if constexpr (requires { resources.resolverRequirementBlocks; }) {
		for (const auto& block : resources.resolverRequirementBlocks)
			if (block) count += block->requirements.size();
	}
	return count;
}

template<class PassResourceData, class Fn>
void ForEachFrameRequirement(PassResourceData& resources, Fn&& fn) {
	for (auto& req : resources.staticResourceRequirements) {
		fn(req);
	}
	if constexpr (requires { resources.resolverRequirementBlocks; }) {
		for (const auto& block : resources.resolverRequirementBlocks)
			if (block) for (const auto& req : block->requirements) fn(req);
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
	if constexpr (requires { resources.resolverRequirementBlocks; }) {
		for (const auto& block : resources.resolverRequirementBlocks)
			if (block) for (const auto& req : block->requirements) fn(req);
	}
	for (const auto& req : resources.frameResourceRequirements) {
		fn(req);
	}
}

template<class PassResourceData>
std::span<const ResourceRequirement> GetFrameRequirementsSpan(const PassResourceData& resources) {
	const bool hasResolverBlocks = [&] {
		if constexpr (requires { resources.resolverRequirementBlocks; }) return !resources.resolverRequirementBlocks.empty();
		return false;
	}();
	if (resources.frameResourceRequirements.empty() && !hasResolverBlocks) {
		return resources.staticResourceRequirements;
	}
	if (resources.staticResourceRequirements.empty() && !hasResolverBlocks) {
		return resources.frameResourceRequirements;
	}
	if (resources.mergedFrameRequirementsDirty) {
		auto& merged = resources.mergedFrameResourceRequirements;
		merged.clear();
		merged.reserve(GetFrameRequirementCount(resources));
		merged.insert(merged.end(), resources.staticResourceRequirements.begin(), resources.staticResourceRequirements.end());
		if constexpr (requires { resources.resolverRequirementBlocks; }) {
			for (const auto& block : resources.resolverRequirementBlocks)
				if (block) merged.insert(merged.end(), block->requirements.begin(), block->requirements.end());
		}
		merged.insert(merged.end(), resources.frameResourceRequirements.begin(), resources.frameResourceRequirements.end());
		resources.mergedFrameRequirementsDirty = false;
	}
	return resources.mergedFrameResourceRequirements;
}


} // namespace org
