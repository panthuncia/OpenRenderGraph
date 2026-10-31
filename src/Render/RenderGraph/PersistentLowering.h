#pragma once
// Lowers a pass's declaration into what the persistent graph installs:
// - its fixed resource uses,
// - its resolver groups. A change in group membership is a binding edit, not a structural one.
// - its end states,
// - its queue, backend and wait policy.
// It reads the pass's current declaration (RenderGraph::RedeclarePass). It never reads the old compiler's declaration cache.
#include "Render/RenderGraph/RenderGraph.h"
#include "Render/RenderGraph/GraphCompiler.h"

#include <cstdint>
#include <memory>
#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace org::lowering {

// The concrete resource behind dynamic wrappers; null when a wrapper has no backing.
Resource* UnwrapDynamic(Resource* resource) noexcept;
// Whether the resource can move between queues of different families without an ownership transfer. That is true when
// it was created with QueueSharing::Concurrent, as every graph buffer is. Unknown resources return false.
bool ConcurrentlyShared(const Resource& resource);
// Takes the shape from the logical description when the texture has no backing yet. An unmaterialized PixelBuffer
// reports default mip and array counts until it is realized.
compiler::CompileResourceShape ShapeOf(const Resource& resource);
compiler::CompileResourceState LowerState(const ResourceState& state);
std::optional<compiler::CompileRange> LowerRange(const RangeSpec& spec, compiler::CompileResourceShape shape);

struct LoweredUse {
	uint64_t resourceID = 0; // Scheduling identity.
	Resource* resource = nullptr;
	compiler::CompileRange range;
	compiler::CompileResourceState state;
	// For a resolver's resource that is declared directly, the slot's own reference, as for a group member. The resolver
	// may drop the resource while the slot still names it. Examples are a removed group entry or the old version of a
	// versioned buffer. Every preparation polls the slots' backings.
	std::shared_ptr<Resource> owner;
};

struct LoweredGroup {
	std::shared_ptr<const void> key;
	std::unique_ptr<IResourceResolver> resolver;
	std::vector<std::pair<uint64_t, std::shared_ptr<Resource>>> members;
	std::vector<std::pair<compiler::CompileRange, compiler::CompileResourceState>> templates;
	compiler::CompileResourceShape shape;
	ResolverResourceSetIdentity identity{};
	uint64_t waitRevision = 0;
	std::vector<ExternalTimelinePoint> waits;
	bool uniform = true;
};

struct LoweredPass {
	std::shared_ptr<const ResourceUseLayout> resourceUses;
	std::shared_ptr<const void> declaredBindings; // The typed bindings of the same declaration (RenderGraphPass::DeclaredBindings).
	compiler::CompilePass declaration;
	std::vector<LoweredUse> entries, exits;
	std::vector<LoweredGroup> groups;
	// The resolvers whose resources are among the entries (not a group), with the set they resolved to.
	std::vector<std::pair<std::shared_ptr<const IResourceResolver>, ResolverResourceSetIdentity>> directResolvers;
	std::vector<ExternalTimelinePoint> explicitWaits;
	BackendInstanceId device = BackendInstanceId::Primary; // The device of every compatible queue slot.
	// The pass's pointer declarations go through its rebind slots (IDynamicDeclaredResources::RebindsConcreteResources):
	// the fingerprint leaves out which resources they name, since another resource in a place is a rebind.
	bool rebindable = false;
};

// Builds the pass's resolver group plan from its current declaration (RenderGraph::ResolverGroupPlan). It must run
// before the old compiler's declaration cache finalizes the pass, because that clears a lone resolver's declared member
// list.
std::optional<RenderGraph::ResolverGroupPlan> PlanResolverGroups(const RenderGraph::PassAndResources& pass);

// The structural identity of a lowering. Group membership is left out on purpose, because it is a binding edit. When a
// pass's redeclaration hashes equal, the executable needs no rebuild, even if the pass reported
// DeclaredResourcesChanged(). The hash does not depend on order, because a redeclaration may list the same requirements
// in a different order.
// With `layoutUses` false, leaves out the declared layout's uses: what stays the same when only which group members a
// pass names, or which views table rows route, changes (a relower against the installed executable).
uint64_t FingerprintLowering(const LoweredPass& lowered, bool layoutUses = true);
// The structural fingerprint in parts (declaration, entries, exits, groups, group views), for saying which one changed.
std::array<uint64_t, 5> FingerprintParts(const LoweredPass& lowered);
const char* FingerprintPartName(size_t part) noexcept;
// The lowered declaration's queue and batch fields, as text (the fingerprint's "declaration" part).
std::string DescribeDeclaration(const LoweredPass& lowered);

// Lowers one pass's current declaration. A resolver becomes a group when both of these hold:
// - the pass's resolver group plan says so,
// - the group's members share one shape.
// Otherwise its members are lowered as direct entries, and the resolver is polled for a new set.
LoweredPass LowerPass(RenderGraph& graph, ResourceRegistry& registry, const QueueRegistry& queues,
	rhi::Backend primaryBackend, const RenderGraph::PassAndResources& pass, RenderGraph::PassType type);

} // namespace org::lowering
