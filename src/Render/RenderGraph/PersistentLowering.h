#pragma once
// Lowering a pass's declaration into what the persistent graph installs: its fixed resource uses, its resolver groups
// (whose membership changes are binding edits, not structural ones), its end states and its queue, backend and wait
// policy. Reads the pass's current declaration (RenderGraph::RedeclarePass), never the old compiler's declaration cache.
#include "Render/RenderGraph/RenderGraph.h"
#include "Render/RenderGraph/GraphCompiler.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace org::lowering {

// The concrete resource behind dynamic wrappers; null when a wrapper has no backing.
Resource* UnwrapDynamic(Resource* resource) noexcept;
// Whether moving the resource between queues of different families needs no ownership transfer: it was created
// QueueSharing::Concurrent (every graph buffer is). Unknown resources answer no.
bool ConcurrentlyShared(const Resource& resource);
// Shape from the logical description when the texture has no backing yet: an unmaterialized PixelBuffer reports default
// mip/array counts until realized.
compiler::CompileResourceShape ShapeOf(const Resource& resource);
compiler::CompileResourceState LowerState(const ResourceState& state);
std::optional<compiler::CompileRange> LowerRange(const RangeSpec& spec, compiler::CompileResourceShape shape);

struct LoweredUse {
	uint64_t resourceID = 0; // Scheduling identity.
	Resource* resource = nullptr;
	compiler::CompileRange range;
	compiler::CompileResourceState state;
	// A resolver's resource declared directly: the slot's own reference, as for a group member. The resolver may drop it
	// (a removed group entry, a versioned buffer's old version) while the slot still names it, and every preparation
	// polls the slots' backings.
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
};

// The pass's resolver group plan from its current declaration (RenderGraph::ResolverGroupPlan). Must run before the old
// compiler's declaration cache finalizes the pass: that clears a lone resolver's declared member list.
std::optional<RenderGraph::ResolverGroupPlan> PlanResolverGroups(const RenderGraph::PassAndResources& pass);

// Structural identity of a lowering. Group membership is deliberately absent: it is a binding edit. A pass whose
// redeclaration hashes equal needs no executable rebuild even though it reported DeclaredResourcesChanged().
// Order-independent: a redeclaration may enumerate the same requirements in a different order.
uint64_t FingerprintLowering(const LoweredPass& lowered);

// Lowers one pass's current declaration. Its resolvers become groups when its resolver group plan says so and a group's
// members share one shape; otherwise their members are lowered as direct entries and the resolvers are polled for a new
// set.
LoweredPass LowerPass(RenderGraph& graph, ResourceRegistry& registry, const QueueRegistry& queues,
	rhi::Backend primaryBackend, const RenderGraph::PassAndResources& pass, RenderGraph::PassType type);

} // namespace org::lowering
