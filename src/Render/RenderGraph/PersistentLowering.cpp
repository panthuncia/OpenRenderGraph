#include "PersistentLowering.h"

#include "Interfaces/IResourceResolver.h"
#include "Render/QueueRegistry.h"
#include "Render/ResourceRequirements.h"
#include "Resources/DynamicResource.h"
#include "Resources/GloballyIndexedResource.h"
#include "Resources/PixelBuffer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace org::lowering {

using experimental::CompileRange;
using experimental::CompileResourceShape;
using experimental::CompileResourceState;

Resource* UnwrapDynamic(Resource* resource) noexcept {
	while (resource) {
		if (auto* dynamic = dynamic_cast<DynamicResource*>(resource)) {
			auto backing = dynamic->GetResource();
			if (!backing) return nullptr;
			resource = backing.get();
			continue;
		}
		if (auto* dynamic = dynamic_cast<DynamicGloballyIndexedResource*>(resource)) {
			auto backing = dynamic->GetResource();
			if (!backing) return nullptr;
			resource = backing.get();
			continue;
		}
		return resource;
	}
	return nullptr;
}

bool ConcurrentlyShared(const Resource& resource) {
	rhi::ResourceDesc description{};
	return resource.TryGetRHIResourceDesc(description) && description.queueSharing == rhi::QueueSharing::Concurrent;
}

CompileResourceShape ShapeOf(const Resource& resource) {
	const auto* concrete = UnwrapDynamic(const_cast<Resource*>(&resource));
	if (const auto* texture = dynamic_cast<const PixelBuffer*>(concrete); texture && !texture->IsMaterialized()) {
		const auto& desc = texture->GetDescription();
		const uint32_t slices = desc.isCubemap ? 6u * desc.arraySize : (desc.isArray ? desc.arraySize : 1u);
		uint32_t mips = 1;
		if (!desc.imageDimensions.empty()) {
			if (slices > 0 && desc.imageDimensions.size() > slices && desc.imageDimensions.size() % slices == 0)
				mips = static_cast<uint32_t>(desc.imageDimensions.size() / slices);
			else if (desc.generateMipMaps)
				mips = static_cast<uint32_t>(std::floor(std::log2((std::max)(desc.imageDimensions[0].width, desc.imageDimensions[0].height)))) + 1;
		}
		return {(std::max)(1u, mips), (std::max)(1u, slices), resource.HasLayout(), resource.CommonLayoutOnly()};
	}
	return {(std::max)(1u, resource.GetMipLevels()), (std::max)(1u, resource.GetArraySize()), resource.HasLayout(), resource.CommonLayoutOnly()};
}

CompileResourceState LowerState(const ResourceState& state) {
	return {static_cast<uint64_t>(state.access), static_cast<uint64_t>(state.layout),
		static_cast<uint64_t>(state.sync), rhi::AccessTypeIsWriteType(state.access)};
}

std::optional<CompileRange> LowerRange(const RangeSpec& spec, CompileResourceShape shape) {
	const auto resolved = ResolveRangeSpec(spec, shape.mips, shape.slices);
	if (resolved.isEmpty()) return std::nullopt;
	return CompileRange{resolved.firstMip, resolved.mipCount, resolved.firstSlice, resolved.sliceCount};
}

std::optional<RenderGraph::ResolverGroupPlan> PlanResolverGroups(const RenderGraph::PassAndResources& pass) {
	using Template = ResolverSnapshot::RequirementTemplate;
	if (pass.resolverSnapshots.empty()) return std::nullopt;
	std::unordered_map<uint64_t, std::vector<Template>> templatesByResource;
	templatesByResource.reserve(pass.resources.staticResourceRequirements.size());
	for (const auto& requirement : pass.resources.staticResourceRequirements)
		templatesByResource[requirement.resourceHandleAndRange.resource.GetGlobalResourceID()].push_back(
			{.range = requirement.resourceHandleAndRange.range, .state = requirement.state});
	RenderGraph::ResolverGroupPlan plan;
	for (const auto& snapshot : pass.resolverSnapshots) {
		std::vector<uint64_t> resourceIDs;
		if (snapshot.state && snapshot.state->resources)
			for (const auto& resource : *snapshot.state->resources) resourceIDs.push_back(resource ? resource->GetGlobalResourceID() : 0u);
		if (snapshot.hasUnclassifiedDeclaration) return std::nullopt;
		auto templates = snapshot.declaredRequirementTemplates;
		if (std::ranges::any_of(templates, [](const Template& t) { return t.state.access == rhi::ResourceAccessType::Common; }))
			return std::nullopt;
		if (resourceIDs.empty()) {
			// Without concrete subresource dimensions, several authored uses may overlap or merge.
			if (templates.size() != 1) return std::nullopt;
			plan.templates.push_back(std::move(templates));
			continue;
		}
		for (const auto id : resourceIDs)
			if (!plan.memberIDs.insert(id).second) return std::nullopt; // A resource of two resolvers.
		const auto first = templatesByResource.find(resourceIDs.front());
		if (first == templatesByResource.end() || first->second.empty()) return std::nullopt;
		// The pass's requirements for the resolver's resources are what its declarations of them came to: several uses of
		// one resolver (indirect arguments and a shader read, say) merge into fewer requirements than they were declared
		// as. The members' requirements are the recipe, so such a pass is still lowered as a group (re-resolved by every
		// preparation under its own capture context) rather than direct, where a preparation for another version - a
		// revision's - would bind the version current when the pass was lowered.
		if (templates.size() != first->second.size()) templates = first->second;
		for (const auto id : resourceIDs) {
			const auto found = templatesByResource.find(id);
			if (found == templatesByResource.end() || found->second.size() != templates.size()) return std::nullopt;
			for (size_t i = 0; i < templates.size(); ++i)
				if (!(found->second[i].range == templates[i].range) || !(found->second[i].state == templates[i].state)
					|| found->second[i].state.sync != templates[i].state.sync)
					return std::nullopt;
		}
		plan.templates.push_back(std::move(templates));
	}
	return plan;
}

uint64_t FingerprintLowering(const LoweredPass& lowered) {
	auto hashUse = [](const LoweredUse& use) {
		uint64_t h = 0x9e3779b97f4a7c15ull;
		auto mix = [&](uint64_t value) { h ^= value + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
		mix(use.resourceID); mix(use.range.mip); mix(use.range.mips); mix(use.range.slice); mix(use.range.slices);
		mix(use.state.access); mix(use.state.layout); mix(use.state.sync); mix(use.state.write);
		return h;
	};
	auto hashTemplate = [](const std::pair<CompileRange, CompileResourceState>& request) {
		uint64_t h = 0x7f4a7c159e3779b9ull;
		auto mix = [&](uint64_t value) { h ^= value + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); };
		const auto& [range, state] = request;
		mix(range.mip); mix(range.mips); mix(range.slice); mix(range.slices);
		mix(state.access); mix(state.layout); mix(state.sync); mix(state.write);
		return h;
	};
	std::vector<uint64_t> parts;
	const auto& d = lowered.declaration;
	parts.push_back(d.backend); parts.push_back(d.preferredQueueSlot); parts.push_back(d.forceBatchIsolation ? 1 : 0);
	parts.push_back(d.recordAtExecution ? 1 : 0);
	{
		auto slots = d.compatibleQueueSlots; std::sort(slots.begin(), slots.end());
		for (auto slot : slots) parts.push_back(0x100 + slot);
	}
	std::vector<uint64_t> entries, exits, groups;
	for (const auto& use : lowered.entries) entries.push_back(hashUse(use));
	for (const auto& use : lowered.exits) exits.push_back(hashUse(use));
	for (const auto& group : lowered.groups) {
		std::vector<uint64_t> templates;
		for (const auto& request : group.templates) templates.push_back(hashTemplate(request));
		std::sort(templates.begin(), templates.end());
		uint64_t h = reinterpret_cast<uintptr_t>(group.key.get());
		h ^= (uint64_t{group.shape.mips} << 40) ^ (uint64_t{group.shape.slices} << 20) ^ (group.shape.hasLayout ? 1 : 0);
		for (auto t : templates) h ^= t + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
		groups.push_back(h);
	}
	std::sort(entries.begin(), entries.end()); std::sort(exits.begin(), exits.end()); std::sort(groups.begin(), groups.end());
	entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
	uint64_t h = 0xcbf29ce484222325ull;
	auto fold = [&](uint64_t value) { h ^= value; h *= 0x100000001b3ull; h ^= h >> 29; };
	for (auto v : parts) fold(v);
	fold(0xE1); for (auto v : entries) fold(v);
	fold(0xE2); for (auto v : exits) fold(v);
	fold(0xE3); for (auto v : groups) fold(v);
	if (lowered.resourceUses) for (const auto& group : lowered.resourceUses->resolverViews) {
		fold(reinterpret_cast<uintptr_t>(group.identity.get()));
		for (const auto view : group.views) {
			fold(static_cast<uint64_t>(view.kind)); fold(view.variant); fold(view.mip); fold(view.slice);
		}
		fold(UINT64_MAX);
	}
	if (lowered.resourceUses) for (const auto& use : lowered.resourceUses->uses) {
		fold(use.resolverIdentity ? reinterpret_cast<uintptr_t>(use.resolverIdentity.get()) : use.binding.registryResourceID);
		fold(static_cast<uint64_t>(use.state.access));
		for (const auto view : use.requiredViews) {
			fold(static_cast<uint64_t>(view.kind)); fold(view.variant); fold(view.mip); fold(view.slice);
		}
		fold(UINT64_MAX);
	}
	return h;
}

LoweredPass LowerPass(RenderGraph& graph, ResourceRegistry& registry, const QueueRegistry& queues,
	rhi::Backend primaryBackend, const RenderGraph::PassAndResources& pass, RenderGraph::PassType type) {
	LoweredPass result;
	result.resourceUses = pass.pass->ResourceUses();
	auto& declaration = result.declaration;
	const auto& resources = pass.resources;
	std::vector<uint32_t> compatible;
	if (resources.pinnedQueueSlot) compatible.push_back(static_cast<uint32_t>(static_cast<uint8_t>(*resources.pinnedQueueSlot)));
	else {
		auto kindCompatible = [&](QueueKind kind) {
			switch (type) {
			case RenderGraph::PassType::Render: return kind == QueueKind::Graphics;
			case RenderGraph::PassType::Compute: return kind == QueueKind::Graphics || kind == QueueKind::Compute;
			default: return true;
			}
		};
		for (size_t slot = 0; slot < queues.SlotCount(); ++slot) {
			const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot));
			if (!queues.IsAutoAssignable(index)) continue;
			const auto kind = queues.GetKind(index);
			const bool automatic = resources.queueAssignmentPolicy == QueueAssignmentPolicy::Automatic && kindCompatible(kind);
			if (automatic || kind == resources.preferredQueueKind) compatible.push_back(static_cast<uint32_t>(slot));
		}
		// The pass's device: the primary unless its backend affinity names another; a preferred one falls back to the primary.
		const auto& affinity = resources.backendAffinity;
		auto onDevice = [&](auto&& accept) {
			std::vector<uint32_t> kept;
			for (const auto slot : compatible) if (accept(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot)))) kept.push_back(slot);
			return kept;
		};
		auto primary = [&](QueueSlotIndex slot) { return queues.GetBackendInstance(slot) == BackendInstanceId::Primary; };
		if (affinity.strength == BackendAffinityStrength::Primary) compatible = onDevice(primary);
		else {
			auto kept = onDevice([&](QueueSlotIndex slot) {
				return affinity.device ? queues.GetBackendInstance(slot) == *affinity.device : queues.GetBackend(slot) == affinity.backend;
			});
			if (kept.empty() && affinity.strength == BackendAffinityStrength::Required)
				throw std::runtime_error("Persistent lowering: pass '" + pass.name + "' requires an unavailable backend");
			compatible = kept.empty() ? onDevice(primary) : std::move(kept);
		}
		if (compatible.empty()) compatible.push_back(static_cast<uint32_t>(resources.preferredQueueKind));
	}
	const auto device = queues.SlotCount() != 0
		? queues.GetBackendInstance(static_cast<QueueSlotIndex>(static_cast<uint8_t>(compatible.front()))) : BackendInstanceId::Primary;
	result.device = device;
	// The device's graphics slot: the route of a pass that may not leave graphics.
	uint32_t graphicsSlot = 0;
	for (size_t slot = 0; slot < queues.SlotCount(); ++slot) {
		const auto index = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot));
		if (queues.GetBackendInstance(index) == device && queues.GetKind(index) == QueueKind::Graphics) { graphicsSlot = static_cast<uint32_t>(slot); break; }
	}
	const bool d3d12OwnedQueueTransfers = queues.SlotCount() != 0
		&& queues.GetBackend(static_cast<QueueSlotIndex>(static_cast<uint8_t>(graphicsSlot))) == rhi::Backend::D3D12;
	declaration.compatibleQueueSlots = compatible;
	declaration.preferredQueueSlot = compatible.front();
	declaration.backend = static_cast<uint32_t>(queues.SlotCount() != 0
		? queues.GetBackend(static_cast<QueueSlotIndex>(static_cast<uint8_t>(graphicsSlot))) : primaryBackend);
	// A pass recorded at execution is a batch of its own: the batch is recorded by the host's execution, the rest ahead.
	declaration.recordAtExecution = pass.recordAtExecution;
	declaration.forceBatchIsolation = !resources.externalWaitsBeforeTransitions.empty()
		|| !resources.externalWaitBindingsBeforeTransitions.empty() || pass.recordAtExecution;
	result.explicitWaits = resources.externalWaitsBeforeTransitions;

	auto resolve = [&](const ResourceRegistry::RegistryHandle& handle) -> Resource* {
		return handle.IsEphemeral() ? handle.GetEphemeralPtr() : registry.Resolve(handle);
	};
	auto lowerRequirement = [&](const ResourceRequirement& requirement) {
		auto* resource = resolve(requirement.resourceHandleAndRange.resource);
		if (!resource) throw std::runtime_error("Persistent lowering: pass '" + pass.name + "' declares an unresolvable resource");
		const auto range = LowerRange(requirement.resourceHandleAndRange.range, ShapeOf(*resource));
		if (!range) return;
		result.entries.push_back({resource->GetSchedulingResourceID(), resource, *range, LowerState(requirement.state)});
	};
	// Every requirement the pass declared, resolver members included (the old compiler's requirement blocks too).
	const auto requirements = GetFrameRequirementsSpan(resources);
	const auto& plan = pass.resolverGroupPlan;
	if (plan && plan->templates.size() == pass.resolverSnapshots.size()) {
		for (const auto& requirement : requirements)
			if (!plan->memberIDs.contains(requirement.resourceHandleAndRange.resource.GetGlobalResourceID())) lowerRequirement(requirement);
		for (size_t i = 0; i < pass.resolverSnapshots.size(); ++i) {
			const auto& snapshot = pass.resolverSnapshots[i];
			LoweredGroup group;
			// The declaration's own capture: its uses name these members.
			const auto& state = snapshot.state;
			group.key = snapshot.DependencyIdentity();
			group.resolver = snapshot.resolver ? snapshot.resolver->Clone() : nullptr;
			if (state) {
				group.identity = state->resourceSetIdentity;
				group.waitRevision = state->waitRevision;
				if (state->waits) group.waits = *state->waits;
			}
			std::optional<CompileResourceShape> shape;
			if (state && state->resources) for (const auto& member : *state->resources) {
				if (!member) continue;
				const auto memberShape = ShapeOf(*member);
				if (!shape) shape = memberShape;
				else if (*shape != memberShape) group.uniform = false;
				group.members.emplace_back(member->GetSchedulingResourceID(), member);
			}
			group.shape = shape.value_or(CompileResourceShape{1, 1, false});
			for (const auto& request : plan->templates[i]) {
				const auto range = LowerRange(request.range, group.shape);
				if (!range) { group.uniform = false; continue; }
				group.templates.emplace_back(*range, LowerState(request.state));
			}
			if (group.uniform && group.key && group.resolver && !group.templates.empty()) { result.groups.push_back(std::move(group)); continue; }
			// Mixed shapes: this resolver's members are declared directly, and the resolver is polled for a new set.
			if (snapshot.resolver)
				result.directResolvers.emplace_back(snapshot.resolver->Clone(), state ? state->resourceSetIdentity : ResolverResourceSetIdentity{});
			for (const auto& [id, member] : group.members)
				for (const auto& request : plan->templates[i]) {
					const auto range = LowerRange(request.range, ShapeOf(*member));
					if (range) result.entries.push_back({id, member.get(), *range, LowerState(request.state), member});
				}
		}
	} else {
		for (const auto& requirement : requirements) lowerRequirement(requirement);
		std::unordered_map<const Resource*, std::shared_ptr<Resource>> owners;
		for (const auto& snapshot : pass.resolverSnapshots) {
			if (!snapshot.resolver) continue;
			const auto& state = snapshot.state;
			result.directResolvers.emplace_back(snapshot.resolver->Clone(), state ? state->resourceSetIdentity : ResolverResourceSetIdentity{});
			if (state && state->resources)
				for (const auto& member : *state->resources) if (member) owners.emplace(member.get(), member);
		}
		for (auto& use : result.entries)
			if (const auto found = owners.find(use.resource); found != owners.end()) use.owner = found->second;
	}
	for (const auto& [handleAndRange, state] : resources.internalTransitions) {
		auto* resource = resolve(handleAndRange.resource);
		if (!resource) throw std::runtime_error("Persistent lowering: pass '" + pass.name + "' transitions an unresolvable resource");
		const auto range = LowerRange(handleAndRange.range, ShapeOf(*resource));
		if (!range) continue;
		result.exits.push_back({resource->GetSchedulingResourceID(), resource, *range, LowerState(state)});
	}
	// D3D12 admission emits the producer's release and the consumer's acquire around a cross-queue wait. Elsewhere a pass
	// leaves the graphics route only when nothing it touches needs a queue-family ownership transfer.
	if (!d3d12OwnedQueueTransfers) {
		bool concurrent = true;
		for (const auto& use : result.entries) concurrent = concurrent && use.resource && ConcurrentlyShared(*use.resource);
		for (const auto& use : result.exits) concurrent = concurrent && use.resource && ConcurrentlyShared(*use.resource);
		for (const auto& group : result.groups)
			for (const auto& [id, member] : group.members) concurrent = concurrent && member && ConcurrentlyShared(*member);
		if (!concurrent) {
			declaration.compatibleQueueSlots.assign(1, graphicsSlot);
			declaration.preferredQueueSlot = graphicsSlot;
		}
	}
	return result;
}

} // namespace org::lowering
