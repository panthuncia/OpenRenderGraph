// RenderGraph's structural build: materializing the extensions' passes, CompileStructural, and redeclaration.
#include "RenderGraphInternal.h"

namespace org {

using namespace graph_internal;

namespace {

	QueueKind ResolveExternalPreferredQueueKind(const RenderGraph::ExternalPassDesc& desc) {
		const QueueKind preferredQueueKind = desc.preferredQueueKind.value_or(QueueKind::Graphics);
		if (!IsQueueKindSupportedByRenderPass(preferredQueueKind)) {
			throw std::runtime_error("External pass '" + desc.name + "' requested an incompatible queue kind");
		}
		return preferredQueueKind;
	}

	QueueAssignmentPolicy ResolveExternalQueueAssignmentPolicy(const RenderGraph::ExternalPassDesc& desc) {
		if (desc.queueAssignmentPolicy.has_value()) {
			return *desc.queueAssignmentPolicy;
		}

		if (desc.preferredQueueKind.has_value()) {
			return QueueAssignmentPolicy::ForcePreferred;
		}

		return QueueAssignmentPolicy::ForcePreferred;
	}

} // namespace

RenderGraph::AnyPassAndResources RenderGraph::MaterializeExternalPass(
	const ExternalPassDesc& d,
	bool callSetup,
	bool materializeReferencedResources)
{
	const bool traceLifecycle = m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled();
	const bool traceStructuralMaterialize = materializeReferencedResources && !callSetup;
	const bool logStructuralMaterialize = traceStructuralMaterialize && traceLifecycle;
	AnyPassAndResources any;
	any.type = d.type;
	any.name = d.name;

	if (logStructuralMaterialize) {
		spdlog::info(
			"RG structural materialize begin pass='{}' type={} preferredQueue={} pinnedQueue={} registerName={}",
			d.name,
			PassTypeToString(d.type),
			QueueKindToString(ResolveExternalPreferredQueueKind(d)),
			d.pinnedQueueSlot.has_value() ? std::to_string(static_cast<unsigned int>(static_cast<uint8_t>(*d.pinnedQueueSlot))) : std::string("none"),
			d.registerName);
	}

	if (d.type == PassType::Render) {
		auto rp = d.pass;
		if (!rp) {
			throw std::logic_error("External pass '" + d.name + "' has no pass object");
		}
		RenderPassAndResources par;
		par.pass = rp;
		par.name = d.name;
		par.techniquePath = d.techniquePath;
		par.collectStatistics = d.collectStatistics;
		par.epoch = d.epoch;
		par.recordAtExecution = d.recordAtExecution;
		{
			RenderPassBuilder b(this, d.name);
			b.pass = rp;
			b.built_ = true;
			b.params = {};
			b.params.isGeometryPass = d.isGeometryPass;
			b._declaredIds.clear();
			if (traceLifecycle) {
				spdlog::info("RG materialize external render pass '{}' declare begin", d.name);
			}
			EnsureProviderRegistered(par.pass.get());
			{ const ResolverCaptureScope captures; rp->DeclareResourceUsages(&b); }
			if (logStructuralMaterialize) {
				spdlog::info(
					"RG structural materialize render pass='{}' declare complete requirements={} transitions={} identifiers={}",
					d.name,
					b.GatherResourceRequirements().size(),
					b.params.internalTransitions.size(),
					b.DeclaredResourceIds().size());
			}
			par.resources.staticResourceRequirements = b.GatherResourceRequirements();
			par.resources.internalTransitions = b.params.internalTransitions;
			par.resources.identifierSet = b.DeclaredResourceIds();
			par.resources.autoDescriptorShaderResources = b.params.autoDescriptorShaderResources;
			par.resources.autoDescriptorConstantBuffers = b.params.autoDescriptorConstantBuffers;
			par.resources.autoDescriptorUnorderedAccessViews = b.params.autoDescriptorUnorderedAccessViews;
			par.resources.activeFeatureDomains = b.params.activeFeatureDomains;
			par.resources.externalWaitsBeforeTransitions = b.params.externalWaitsBeforeTransitions;
			par.resources.externalWaitBindingsBeforeTransitions = b.params.externalWaitBindingsBeforeTransitions;
			par.resources.isGeometryPass = b.params.isGeometryPass;
			par.resources.preferredQueueKind = ResolveExternalPreferredQueueKind(d);
			par.resources.queueAssignmentPolicy = ResolveExternalQueueAssignmentPolicy(d);
			par.resources.pinnedQueueSlot = d.pinnedQueueSlot;
			if (materializeReferencedResources) {
				if (traceLifecycle) {
					spdlog::info("RG materialize external render pass '{}' materialize referenced resources begin", d.name);
				}
				if (logStructuralMaterialize) {
					spdlog::info("RG structural materialize render pass='{}' referenced resources begin", d.name);
				}
				MaterializeReferencedResources(par.resources.staticResourceRequirements, par.resources.internalTransitions, d.name);
				if (logStructuralMaterialize) {
					spdlog::info("RG structural materialize render pass='{}' referenced resources complete", d.name);
				}
			}
			par.retainedAnonymousKeepAlive = CaptureRetainedAnonymousKeepAlive(
				par.resources.staticResourceRequirements,
				par.resources.internalTransitions);
			par.resolverSnapshots = b.TakeResolverSnapshots();
			par.resolverGroupPlan = lowering::PlanResolverGroups(par);
		}

		if (callSetup) {
			par.pass->ConfigureRuntimeServices(m_uploadService, m_descriptorService);
			par.pass->ConfigureResourceRegistryView(
				MakePassResourceRegistryView(_registry, par.resources), par.resources);
			if (traceLifecycle) {
				spdlog::info("RG materialize external render pass '{}' setup begin", d.name);
			}
			par.pass->Setup();
			if (traceLifecycle) {
				spdlog::info("RG materialize external render pass '{}' setup complete", d.name);
			}
		}

		any.pass = std::move(par);
	}

	if (logStructuralMaterialize) {
		spdlog::info("RG structural materialize complete pass='{}' type={}", d.name, PassTypeToString(d.type));
	}
	if (traceStructuralMaterialize) {
		if (m_structuralMaterializeCheckpointCallback) {
			m_structuralMaterializeCheckpointCallback(d.name);
		}
	}

	return any;
}

void RenderGraph::RegisterExternalPassName(const ExternalPassDesc& d, AnyPassAndResources& any)
{
	if (!d.registerName) {
		return;
	}

	if (d.type == PassType::Render) {
		auto& rp = std::get<RenderPassAndResources>(any.pass);
		if (!d.name.empty()) {
			renderPassesByName[d.name] = std::dynamic_pointer_cast<RenderPass>(rp.pass);
		}
	}
}

void RenderGraph::CompileStructural() {
	// Register resource providers from pass builders

	std::vector<unsigned int> empty;
	{
	BT_ZONE_SCOPE("RenderGraph::CompileStructural::FinalizeBasePasses");
	// Go backwards to build skip list
	for (int i = static_cast<int>(m_passBuilderOrder.size()) - 1; i >= 0; i--) {
		auto ptr = m_passBuilderOrder[i];
		auto prov = ptr->ResourceProvider();
		if (!prov) {
			empty.push_back(i); // This pass was not built
			continue;
		}
		EnsureProviderRegistered(prov);
	}
	unsigned int i = 0;
	for (auto ptr : m_passBuilderOrder) {
		if (!empty.empty() && empty.back() == i) {
			empty.pop_back();
			continue;
		}
		ptr->Finalize();
		i++;
	}
	}


	struct Pending {
		AnyPassAndResources pr;

		std::string anchorKey;     // internal unique key used for anchoring/emission

		ExternalInsertPoint where; // concrete
		bool chained = false;      // true if where was synthesized from "natural ordering"
		int chainOrder = 0;        // preserves extension-local order among chained followers

		int priority = 0;
		size_t order = 0;          // global stable order across all gathered externals
	};

	auto makeAny = [&](ExternalPassDesc const& d) -> AnyPassAndResources {
		return MaterializeExternalPass(d, false, true);
	};

	// Sentinels (must not collide with real pass names)
	static constexpr const char* kBeginKey = "__rg_begin__";
	static constexpr const char* kAfterBaseKey = "__rg_after_base__";
	static constexpr const char* kEndKey = "__rg_end__"; // legacy alias for end-of-base anchor
	static constexpr const char* kFirstBaseKey = "__rg_first_base__"; // optional helper token

	struct ExtItem {
		AnyPassAndResources pr;
		std::string key;                 // unique key for anchoring
		ExternalInsertPoint where;        // constraints
		int priority = 0;
		size_t order = 0;                // global stable order
		int extIndex = 0;                // which extension emitted it
		int extLocalOrder = 0;           // order within that extension
	};

	struct MergeNode {
		std::string key;
		bool hasPass = false;            // sentinel nodes have no pass payload
		AnyPassAndResources pass{};      // valid iff hasPass
		int priority = 0;
		size_t order = 0;
		std::vector<size_t> out;
		uint32_t indeg = 0;
	};

	// Keep base passes
	auto base = std::move(m_masterPassList);
	m_masterPassList.clear();
	m_structuralExplicitAfterByName.clear();

	// Gather extension passes into ExtItem list
	std::vector<ExtItem> extItems;
	extItems.reserve(64);

	size_t globalOrder = 0;

	// helper: stable synthetic key for unnamed passes
	auto makeSyntheticKey = [](size_t n) -> std::string {
		return "__rg_ext_" + std::to_string(n);
		};

	{
	BT_ZONE_SCOPE("RenderGraph::CompileStructural::GatherAndMaterializeExtensions");
	for (int ei = 0; ei < (int)m_extensions.size(); ++ei) {
		auto& ext = m_extensions[ei];
		if (!ext) continue;

		std::vector<ExternalPassDesc> local;
		local.reserve(16);
		if (m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled()) {
			spdlog::info("RG gather structural extension {} begin", ei);
		}
		{
			BT_ZONE_SCOPE("RenderGraph::CompileStructural::GatherExtensionPasses");
			const auto gatherBegin = std::chrono::steady_clock::now();
			ext->GatherStructuralPasses(*this, local);
			const auto gatherMs = std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - gatherBegin).count();
			if (gatherMs >= 10.0) {
				const std::string_view extensionId = static_cast<std::size_t>(ei) < m_extensionRegistrationIds.size()
					? std::string_view{ m_extensionRegistrationIds[static_cast<std::size_t>(ei)] }
					: std::string_view{ "unknown" };
				spdlog::info(
					"RenderGraph structural extension gather: id='{}' elapsed_ms={:.3f} passes={}",
					extensionId,
					gatherMs,
					local.size());
			}
		}
		if (m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled()) {
			spdlog::info("RG gather structural extension {} complete localPassCount={}", ei, local.size());
		}

		std::optional<std::string> prevKey; // for extension-local chaining
		int localOrder = 0;

		for (auto& d : local) {
			BT_ZONE_SCOPE("RenderGraph::CompileStructural::MaterializeExtensionPass");
			BT_ZONE_TEXT(d.name.data(), d.name.size());
			if (d.type == PassType::Unknown) continue;
			if (!d.pass) continue;

			ExtItem it;
			if (m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled()) {
				spdlog::info("RG gather structural extension {} materialize begin name='{}' type={} localOrder={}", ei, d.name, static_cast<int>(d.type), localOrder);
			}
			it.pr = MaterializeExternalPass(d, false, true);
			if (m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled()) {
				spdlog::info("RG gather structural extension {} materialize complete name='{}'", ei, d.name);
			}
			RegisterExternalPassName(d, it.pr);
			if (m_getRenderGraphBatchTraceEnabled && m_getRenderGraphBatchTraceEnabled()) {
				spdlog::info("RG gather structural extension {} register external pass name complete name='{}'", ei, d.name);
			}

			it.order = globalOrder++;
			it.key = !d.name.empty() ? d.name : makeSyntheticKey(it.order);

			if (d.where.has_value()) {
				it.where = *d.where;
			}
			else {
				it.where = ExternalInsertPoint{};
				it.where.keepExtensionOrder = true;

				// Only the *first* unconstrained pass gets anchored after base.
				// Followers will be ordered by chain edges.
				if (!prevKey.has_value()) {
					it.where.after.push_back(kAfterBaseKey);
				}
			}

			it.priority = it.where.priority;
			it.extIndex = ei;
			it.extLocalOrder = localOrder++;

			// Store prevKey for later chaining edges (we apply chaining even if explicit where)
			// (We don't add edges here because we haven't built node indices yet.)
			// We'll reconstruct chaining using extIndex/extLocalOrder + keepExtensionOrder below.
			extItems.push_back(std::move(it));
			prevKey = extItems.back().key;
		}
	}
	}

	BT_ZONE_SCOPE("RenderGraph::CompileStructural::OrderPasses");
	// Build nodes list: sentinels + base + externals
	std::vector<MergeNode> nodes;
	nodes.reserve(2 + base.size() + extItems.size());

	auto addNode = [&](std::string key, bool hasPass, AnyPassAndResources&& pass, int prio, size_t ord) -> size_t {
		MergeNode n;
		n.key = std::move(key);
		n.hasPass = hasPass;
		if (hasPass) n.pass = std::move(pass);
		n.priority = prio;
		n.order = ord;
		nodes.push_back(std::move(n));
		return nodes.size() - 1;
		};

	const size_t beginIdx = addNode(std::string(kBeginKey), false, AnyPassAndResources{}, INT_MIN, 0);
	const size_t afterBaseIdx = addNode(std::string(kAfterBaseKey), false, AnyPassAndResources{}, INT_MAX, 1);

	// Map key->node index (detect collisions early)
	std::unordered_map<std::string, size_t> keyToIdx;
	keyToIdx.reserve(2 + base.size() + extItems.size());
	keyToIdx.emplace(nodes[beginIdx].key, beginIdx);
	keyToIdx.emplace(nodes[afterBaseIdx].key, afterBaseIdx);
	keyToIdx.emplace(kEndKey, afterBaseIdx);

	// Add base pass nodes (preserve current base order deterministically)
	std::vector<size_t> baseIdx;
	baseIdx.reserve(base.size());

	for (size_t bi = 0; bi < base.size(); ++bi) {
		AnyPassAndResources bp = std::move(base[bi]);

		// Anchor keys for base: use name if present, else synthetic (not anchorable by user)
		std::string key = !bp.name.empty() ? bp.name : ("__rg_base_" + std::to_string(bi));

		if (keyToIdx.contains(key)) {
			throw std::runtime_error("Pass name/key collision during structural merge: " + key);
		}

		size_t idx = addNode(key, true, std::move(bp), /*prio=*/0, /*ord=*/1000 + bi);
		keyToIdx.emplace(nodes[idx].key, idx);
		baseIdx.push_back(idx);
	}

	// Add external pass nodes
	std::vector<size_t> extIdx;
	extIdx.reserve(extItems.size());

	for (size_t i = 0; i < extItems.size(); ++i) {
		auto& e = extItems[i];

		if (keyToIdx.contains(e.key)) {
			spdlog::error("External pass name/key collision during structural merge: {}", e.key);
			throw std::runtime_error("External pass name/key collision during structural merge: " + e.key);
		}

		size_t idx = addNode(e.key, true, std::move(e.pr), e.priority, /*ord=*/2000 + e.order);
		keyToIdx.emplace(nodes[idx].key, idx);
		extIdx.push_back(idx);
	}

	// Edge helper (dedup)
	std::unordered_set<uint64_t> edgeSet;
	edgeSet.reserve(nodes.size() * 8);

	auto addEdge = [&](size_t from, size_t to) {
		if (from == to) return;
		uint64_t k = (uint64_t(from) << 32) | uint64_t(to);
		if (!edgeSet.insert(k).second) return;
		nodes[from].out.push_back(to);
		nodes[to].indeg++;
		};

	// Base order edges: BEGIN -> base0 -> base1 -> ... -> lastBase -> AFTER_BASE
	if (!baseIdx.empty()) {
		addEdge(beginIdx, baseIdx.front());
		for (size_t i = 0; i + 1 < baseIdx.size(); ++i) addEdge(baseIdx[i], baseIdx[i + 1]);
		addEdge(baseIdx.back(), afterBaseIdx);
	}
	else {
		addEdge(beginIdx, afterBaseIdx);
	}

	// Apply external constraints + extension chaining
	// Build per-extension ordering list (by extLocalOrder)
	std::unordered_map<int, std::vector<std::pair<int, size_t>>> extOrder; // extIndex -> [(localOrder, nodeIdx)]
	extOrder.reserve(m_extensions.size());

	for (size_t i = 0; i < extItems.size(); ++i) {
		extOrder[extItems[i].extIndex].push_back({ extItems[i].extLocalOrder, extIdx[i] });
	}

	for (auto& [ei, v] : extOrder) {
		std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first < b.first; });
	}

	// Now attach constraints and chain edges
	for (size_t i = 0; i < extItems.size(); ++i) {
		auto& e = extItems[i];
		const size_t passNode = extIdx[i];

		// Helper: resolve special token for "first base"
		auto resolveAnchor = [&](std::string const& anchor) -> std::optional<size_t> {
			if (anchor == kFirstBaseKey) {
				if (!baseIdx.empty()) return baseIdx.front();
				return afterBaseIdx;
			}
			auto it = keyToIdx.find(anchor);
			if (it == keyToIdx.end()) return std::nullopt;
			return it->second;
			};

		bool anyConstraint = false;

		// after[] : anchor -> pass
		for (auto const& a : e.where.after) {
			auto idxOpt = resolveAnchor(a);
			if (!idxOpt) {
				if (!a.starts_with("CLodShadow::")) {
					spdlog::warn("External pass '{}' requested After('{}') but anchor not found; ignoring.", e.key, a);
				}
				continue;
			}
			addEdge(*idxOpt, passNode);
			if (a != kBeginKey && a != kAfterBaseKey && a != kEndKey && a != kFirstBaseKey) {
				m_structuralExplicitAfterByName.push_back({ a, e.key });
			}
			anyConstraint = true;
		}

		// before[] : pass -> anchor
		for (auto const& b : e.where.before) {
			auto idxOpt = resolveAnchor(b);
			if (!idxOpt) {
				if (!e.key.starts_with("CLodShadow::")) {
					spdlog::warn("External pass '{}' requested Before('{}') but anchor not found; ignoring.", e.key, b);
				}
				continue;
			}
			addEdge(passNode, *idxOpt);
			if (b != kBeginKey && b != kAfterBaseKey && b != kEndKey && b != kFirstBaseKey) {
				m_structuralExplicitAfterByName.push_back({ e.key, b });
			}
			anyConstraint = true;
		}
	}
	spdlog::debug("RenderGraph structural merge: external constraints complete passes={} edges={}", extItems.size(), edgeSet.size());

	// Extension chaining edges: prev -> next (if keepExtensionOrder on the *next* pass)
	for (auto& [ei, v] : extOrder) {
		spdlog::debug("RenderGraph structural merge: chaining extension={} passes={}", ei, v.size());
		for (size_t j = 1; j < v.size(); ++j) {
			// Find the extItems entry for this node to check keepExtensionOrder
			// (We can check by key because keys are unique.)
			const size_t prevNode = v[j - 1].second;
			const size_t nextNode = v[j].second;

			const std::string& nextKey = nodes[nextNode].key;

			// locate corresponding ext item (small N; linear is fine)
			bool keep = true;
			for (auto& e : extItems) {
				if (e.key == nextKey) { keep = e.where.keepExtensionOrder; break; }
			}
			if (keep) addEdge(prevNode, nextNode);
		}
	}
	spdlog::debug("RenderGraph structural merge: extension chaining complete edges={}", edgeSet.size());

	// Topological sort (stable by priority then order)
	spdlog::debug("RenderGraph structural merge: sorting {} nodes", nodes.size());
	std::vector<uint32_t> indeg(nodes.size());
	for (size_t n = 0; n < nodes.size(); ++n) indeg[n] = nodes[n].indeg;

	std::vector<size_t> ready;
	ready.reserve(nodes.size());
	for (size_t n = 0; n < nodes.size(); ++n) if (indeg[n] == 0) ready.push_back(n);

	auto better = [&](size_t a, size_t b) {
		if (nodes[a].priority != nodes[b].priority) return nodes[a].priority < nodes[b].priority;
		return nodes[a].order < nodes[b].order;
		};

	std::vector<size_t> topo;
	topo.reserve(nodes.size());

	while (!ready.empty()) {
		auto it = std::min_element(ready.begin(), ready.end(), [&](size_t a, size_t b) { return better(a, b); });
		size_t u = *it;
		ready.erase(it);

		topo.push_back(u);
		for (size_t vtx : nodes[u].out) {
			if (--indeg[vtx] == 0) ready.push_back(vtx);
		}
	}

	if (topo.size() != nodes.size()) {
		spdlog::error("Structural merge has a cycle (extension anchors/chains impossible).");
		throw std::runtime_error("RenderGraph structural merge cycle");
	}

	spdlog::debug("RenderGraph structural merge: topological sort complete nodes={}", topo.size());
	// Emit final m_masterPassList in topo order (skip sentinels)
	m_masterPassList.clear();
	m_masterPassList.reserve(baseIdx.size() + extIdx.size());

	for (size_t u : topo) {
		if (!nodes[u].hasPass) {
			continue;
		}
		m_masterPassList.push_back(std::move(nodes[u].pass));
	}
	spdlog::debug("RenderGraph structural merge: emitted {} passes", m_masterPassList.size());
	spdlog::debug("RenderGraph structural merge: compile complete");
}


static ResourceRegistry::RegistryHandle ResolveByIdThunk(void* user, ResourceIdentifier const& id, bool allowFailure) {
	return static_cast<RenderGraph*>(user)->RequestResourceHandle(id, allowFailure);
}

static ResourceRegistry::RegistryHandle ResolveByPtrThunk(void* user, Resource* ptr, bool allowFailure) {
	return static_cast<RenderGraph*>(user)->RequestResourceHandle(ptr, allowFailure);
}

static bool Overlap(SubresourceRange a, SubresourceRange b) {
	auto aMipEnd = a.firstMip + a.mipCount;
	auto bMipEnd = b.firstMip + b.mipCount;
	auto aSlEnd = a.firstSlice + a.sliceCount;
	auto bSlEnd = b.firstSlice + b.sliceCount;
	return (a.firstMip < bMipEnd && b.firstMip < aMipEnd) &&
		(a.firstSlice < bSlEnd && b.firstSlice < aSlEnd);
}

static bool RequirementsConflict(
	std::span<const ResourceRequirement> retained,
	std::span<const ResourceRequirement> immediate)
{
	if (retained.empty() || immediate.empty()) return false;

	// Group immediate requirements by resource ID for O(N+M) lookup
	std::unordered_map<uint64_t, std::vector<const ResourceRequirement*>> immediateByID;
	immediateByID.reserve(immediate.size());
	for (auto const& ib : immediate) {
		immediateByID[ib.resourceHandleAndRange.resource.GetGlobalResourceID()].push_back(&ib);
	}

	for (auto const& ra : retained) {
		auto res = ra.resourceHandleAndRange.resource;
		uint64_t rid = res.GetGlobalResourceID();
		auto it = immediateByID.find(rid);
		if (it == immediateByID.end()) continue;

		auto a = ResolveRangeSpec(ra.resourceHandleAndRange.range, res.GetNumMipLevels(), res.GetArraySize());
		if (a.isEmpty()) continue;

		for (auto const* ib : it->second) {
			auto b = ResolveRangeSpec(ib->resourceHandleAndRange.range, res.GetNumMipLevels(), res.GetArraySize());
			if (b.isEmpty()) continue;

			if (Overlap(a, b) && !(ra.state == ib->state)) {
				return true;
			}
		}
	}
	return false;
}


void RenderGraph::RedeclarePass(PassAndResources& p, PassType type)
{
	BT_ZONE_SCOPE("RenderGraph::RedeclarePass");
	if (!p.name.empty()) { BT_ZONE_TEXT(p.name.data(), p.name.size()); }
	auto redeclare = [&]<class Builder, class Pass>(std::type_identity<Builder>, std::type_identity<Pass>, const char* kind) {
		auto pass = std::dynamic_pointer_cast<Pass>(p.pass);
		if (!pass)
			throw std::logic_error(std::string(kind) + "-tagged pass '" + p.name + "' does not implement its declaration interface");
		// Enough of a built builder for pass code that queries ResourceProvider().
		Builder b(this, p.name);
		b.pass = pass;
		b.built_ = true;
		EnsureProviderRegistered(p.pass.get());
		{
			BT_ZONE_SCOPE("RenderGraph::RedeclarePass::DeclareResourceUsages");
			const ResolverCaptureScope captures;
			pass->DeclareResourceUsages(&b);
		}
		auto& resources = p.resources;
		resources.staticResourceRequirements = b.GatherResourceRequirements();
		resources.resolverRequirementBlocks.clear();
		resources.mergedFrameRequirementsDirty = true;
		resources.internalTransitions = std::move(b.params.internalTransitions);
		resources.identifierSet = std::move(b._declaredIds);
		resources.autoDescriptorShaderResources = std::move(b.params.autoDescriptorShaderResources);
		resources.autoDescriptorConstantBuffers = std::move(b.params.autoDescriptorConstantBuffers);
		resources.autoDescriptorUnorderedAccessViews = std::move(b.params.autoDescriptorUnorderedAccessViews);
		resources.activeFeatureDomains = std::move(b.params.activeFeatureDomains);
		p.explicitExternalWaitsBeforeTransitions = b.params.externalWaitsBeforeTransitions;
		resources.externalWaitsBeforeTransitions = std::move(b.params.externalWaitsBeforeTransitions);
		resources.externalWaitBindingsBeforeTransitions = std::move(b.params.externalWaitBindingsBeforeTransitions);
		{
			BT_ZONE_SCOPE("RenderGraph::RedeclarePass::MaterializeReferencedResources");
			MaterializeReferencedResources(resources.staticResourceRequirements, resources.internalTransitions);
		}
		p.resolverSnapshots = b.TakeResolverSnapshots();
		MergeResolverWaits(resources.externalWaitsBeforeTransitions, p.resolverSnapshots);
		p.retainedAnonymousKeepAlive = CaptureRetainedAnonymousKeepAlive(resources.staticResourceRequirements, resources.internalTransitions);
		p.resolverGroupPlan = lowering::PlanResolverGroups(p);
		// A versioned resolver can replace the concrete resource behind an unchanged identifier: the view and Setup are
		// rebuilt so automatic descriptor bindings follow it. Immediate-only upload and readback passes consume the
		// resources captured in their bytecode and opt out.
		const auto* dynamic = dynamic_cast<IDynamicDeclaredResources*>(p.pass.get());
		const bool rebind = !p.resolverSnapshots.empty()
			|| !dynamic || dynamic->RequiresPassRebindAfterDeclarationRefresh();
		if (rebind) {
			BT_ZONE_SCOPE("RenderGraph::RedeclarePass::Rebind");
			p.pass->ConfigureRuntimeServices(m_uploadService, m_descriptorService);
			p.pass->ConfigureResourceRegistryView(MakePassResourceRegistryView(_registry, resources), resources);
			p.pass->Setup();
		}
	};
	p.resolverGroupPlan.reset();
	switch (type) {
	case PassType::Render: redeclare(std::type_identity<RenderPassBuilder>{}, std::type_identity<RenderPass>{}, "Render"); break;
	default: throw std::logic_error("Pass '" + p.name + "' has no declaration interface");
	}
}

} // namespace org
