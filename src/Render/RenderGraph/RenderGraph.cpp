#include "Render/RenderGraph/RenderGraph.h"

#include <span>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <rhi_helpers.h>
#include <rhi_debug.h>

#include "Render/PassExecutionContext.h"
#include "Utilities/ORGUtilities.h"
#include "Managers/Singletons/DeviceManager.h"
#include "Managers/Singletons/DeletionManager.h"
#include "Render/PassBuilders.h"
#include "Resources/ResourceGroup.h"
#include "Managers/CommandRecordingManager.h"
#include "Interfaces/IHasMemoryMetadata.h"
#include "Interfaces/IDynamicDeclaredResources.h"
#include "Resources/PixelBuffer.h"
#include "Resources/MemoryStatisticsComponents.h"

namespace {
	constexpr size_t QueueIndex(QueueKind queue) noexcept {
		return static_cast<size_t>(queue);
	}

	// Insert into a sorted vector, maintaining sorted order. No-op if already present.
	inline void SortedInsert(std::vector<uint64_t>& v, uint64_t val) {
		auto it = std::lower_bound(v.begin(), v.end(), val);
		if (it == v.end() || *it != val) v.insert(it, val);
	}

	bool QueueSupportsSyncState(QueueKind queue, rhi::ResourceSyncState state) {
		switch (queue) {
		case QueueKind::Graphics:
			return true;
		case QueueKind::Compute:
			return !ResourceSyncStateIsNotComputeSyncState(state);
		case QueueKind::Copy:
			switch (state) {
			case rhi::ResourceSyncState::None:
			case rhi::ResourceSyncState::All:
			case rhi::ResourceSyncState::Copy:
			case rhi::ResourceSyncState::Resolve:
				return true;
			default:
				return false;
			}
		default:
			return false;
		}
	}

	bool QueueSupportsAccessType(QueueKind queue, rhi::ResourceAccessType access) {
		if (queue == QueueKind::Graphics) {
			return true;
		}

		if (queue == QueueKind::Compute) {
			const auto unsupported = rhi::ResourceAccessType::RenderTarget |
				rhi::ResourceAccessType::DepthRead |
				rhi::ResourceAccessType::DepthReadWrite;
			return (access & unsupported) == 0;
		}

		if (queue == QueueKind::Copy) {
			const auto supported = rhi::ResourceAccessType::None |
				rhi::ResourceAccessType::Common |
				rhi::ResourceAccessType::CopySource |
				rhi::ResourceAccessType::CopyDest;
			return (access & ~supported) == 0;
		}

		return false;
	}

	bool QueueSupportsTransition(QueueKind queue, const ResourceTransition& transition) {
		if (!QueueSupportsSyncState(queue, transition.prevSyncState)) {
			return false;
		}
		if (!QueueSupportsSyncState(queue, transition.newSyncState)) {
			return false;
		}
		if (!QueueSupportsAccessType(queue, transition.prevAccessType)) {
			return false;
		}
		if (!QueueSupportsAccessType(queue, transition.newAccessType)) {
			return false;
		}
		return true;
	}
}

RenderGraph::PassView RenderGraph::GetPassView(AnyPassAndResources& pr) {
	PassView v{};
	if (pr.type == PassType::Compute) {
		auto& p = std::get<ComputePassAndResources>(pr.pass);
		v.reqs = &p.resources.frameResourceRequirements;
		v.internalTransitions = &p.resources.internalTransitions;
	}
	else if (pr.type == PassType::Render) {
		auto& p = std::get<RenderPassAndResources>(pr.pass);
		v.reqs = &p.resources.frameResourceRequirements;
		v.internalTransitions = &p.resources.internalTransitions;
	}
	else if (pr.type == PassType::Copy) {
		auto& p = std::get<CopyPassAndResources>(pr.pass);
		v.reqs = &p.resources.frameResourceRequirements;
		v.internalTransitions = &p.resources.internalTransitions;
	}
	return v;
}

std::vector<RenderGraph::Node> RenderGraph::BuildNodes(RenderGraph& rg, std::vector<AnyPassAndResources>& passes) {

	std::vector<Node> nodes;
	nodes.resize(passes.size());

	auto resolveCompatibleQueueSlotsForPass = [&rg](const AnyPassAndResources& pr) -> std::vector<size_t> {
		auto collectSlotsForKind = [&rg](QueueKind kind) {
			std::vector<size_t> slots;
			const size_t slotCount = rg.m_queueRegistry.SlotCount();
			slots.reserve(slotCount);
			for (size_t slotIndex = 0; slotIndex < slotCount; ++slotIndex) {
				const auto queueSlotIndex = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slotIndex));
				if (rg.m_queueRegistry.GetKind(queueSlotIndex) == kind && rg.m_queueRegistry.IsAutoAssignable(queueSlotIndex)) {
					slots.push_back(slotIndex);
				}
			}
			if (slots.empty()) {
				slots.push_back(QueueIndex(kind));
			}
			return slots;
		};

		if (pr.type == PassType::Compute) {
			const auto& pass = std::get<ComputePassAndResources>(pr.pass);
			if (pass.resources.queueSlotOverride)
				return std::vector<size_t>{ static_cast<size_t>(static_cast<uint8_t>(*pass.resources.queueSlotOverride)) };
			return collectSlotsForKind(ResolveQueueKind(pass.resources.queueSelection));
		}

		if (pr.type == PassType::Render) {
			const auto& pass = std::get<RenderPassAndResources>(pr.pass);
			if (pass.resources.queueSlotOverride)
				return std::vector<size_t>{ static_cast<size_t>(static_cast<uint8_t>(*pass.resources.queueSlotOverride)) };
			return collectSlotsForKind(ResolveQueueKind(pass.resources.queueSelection));
		}

		if (pr.type == PassType::Copy) {
			const auto& pass = std::get<CopyPassAndResources>(pr.pass);
			if (pass.resources.queueSlotOverride)
				return std::vector<size_t>{ static_cast<size_t>(static_cast<uint8_t>(*pass.resources.queueSlotOverride)) };
			return collectSlotsForKind(ResolveQueueKind(pass.resources.queueSelection));
		}

		return std::vector<size_t>{ QueueIndex(QueueKind::Graphics) };
	};

	for (size_t i = 0; i < passes.size(); ++i) {
		Node n{};
		n.passIndex = i;
		n.compatibleQueueSlots = resolveCompatibleQueueSlotsForPass(passes[i]);
		n.queueSlot = n.compatibleQueueSlots.empty() ? QueueIndex(QueueKind::Graphics) : n.compatibleQueueSlots.front();
		n.assignedQueueSlot = n.queueSlot;
		n.originalOrder = static_cast<uint32_t>(i);

		PassView view = GetPassView(passes[i]);

		auto mark = [&](uint64_t rid, AccessKind k, bool isUav) {
			n.touchedIDs.push_back(rid);
			if (isUav) n.uavIDs.push_back(rid);
			n.accessByID.push_back({rid, k});
			};

		// resource requirements
		for (auto& req : *view.reqs) {
			uint64_t base = req.resourceHandleAndRange.resource.GetGlobalResourceID();
			bool write = AccessTypeIsWriteType(req.state.access);
			bool isUav = IsUAVState(req.state);

			for (uint64_t rid : rg.m_aliasingSubsystem.GetSchedulingEquivalentIDs(base, rg.aliasPlacementRangesByID)) {
				mark(rid, write ? AccessKind::Write : AccessKind::Read, isUav);
			}
		}

		// internal transitions: treat as "write" for scheduling conservatism
		for (auto& tr : *view.internalTransitions) {
			uint64_t base = tr.first.resource.GetGlobalResourceID();
			for (uint64_t rid : rg.m_aliasingSubsystem.GetSchedulingEquivalentIDs(base, rg.aliasPlacementRangesByID)) {
				mark(rid, AccessKind::Write, /*isUav=*/false);
			}
		}

		// Deduplicate touchedIDs
		std::sort(n.touchedIDs.begin(), n.touchedIDs.end());
		n.touchedIDs.erase(std::unique(n.touchedIDs.begin(), n.touchedIDs.end()), n.touchedIDs.end());

		// Deduplicate uavIDs
		std::sort(n.uavIDs.begin(), n.uavIDs.end());
		n.uavIDs.erase(std::unique(n.uavIDs.begin(), n.uavIDs.end()), n.uavIDs.end());

		// Deduplicate accessByID: sort by ID, then collapse duplicates keeping Write over Read
		std::sort(n.accessByID.begin(), n.accessByID.end(), [](auto& a, auto& b) {
			if (a.first != b.first) return a.first < b.first;
			return a.second > b.second; // Write (1) before Read (0)
		});
		auto last = std::unique(n.accessByID.begin(), n.accessByID.end(),
			[](auto& a, auto& b) { return a.first == b.first; });
		n.accessByID.erase(last, n.accessByID.end());

		nodes[i] = std::move(n);
	}

	return nodes;
}

std::vector<uint8_t> RenderGraph::PlanActiveQueueSlots(
	RenderGraph& rg,
	const std::vector<AnyPassAndResources>& passes,
	const std::vector<Node>& nodes)
{
	const size_t slotCount = rg.m_queueRegistry.SlotCount();
	std::vector<uint8_t> activeSlots(slotCount, 0);
	if (slotCount == 0) {
		return activeSlots;
	}

	auto passHasExplicitQueuePin = [](const AnyPassAndResources& pr) {
		return std::visit([](auto const& passEntry) -> bool {
			using T = std::decay_t<decltype(passEntry)>;
			if constexpr (std::is_same_v<T, std::monostate>) {
				return false;
			}
			else {
				return passEntry.resources.queueSlotOverride.has_value();
			}
		}, pr.pass);
	};

	std::vector<uint32_t> indeg(nodes.size());
	std::vector<uint32_t> level(nodes.size(), 0);
	std::vector<size_t> ready;
	ready.reserve(nodes.size());
	for (size_t i = 0; i < nodes.size(); ++i) {
		indeg[i] = nodes[i].indegree;
		if (indeg[i] == 0) {
			ready.push_back(i);
		}
	}
	for (size_t head = 0; head < ready.size(); ++head) {
		const size_t u = ready[head];
		for (size_t v : nodes[u].out) {
			level[v] = (std::max)(level[v], level[u] + 1);
			if (--indeg[v] == 0) {
				ready.push_back(v);
			}
		}
	}

	std::array<std::vector<size_t>, static_cast<size_t>(QueueKind::Count)> slotsByKind;
	std::array<std::vector<size_t>, static_cast<size_t>(QueueKind::Count)> autoAssignableSlotsByKind;
	for (size_t slotIndex = 0; slotIndex < slotCount; ++slotIndex) {
		const auto queueSlotIndex = static_cast<QueueSlotIndex>(static_cast<uint8_t>(slotIndex));
		const QueueKind kind = rg.m_queueRegistry.GetKind(queueSlotIndex);
		slotsByKind[QueueIndex(kind)].push_back(slotIndex);
		if (rg.m_queueRegistry.IsAutoAssignable(queueSlotIndex)) {
			autoAssignableSlotsByKind[QueueIndex(kind)].push_back(slotIndex);
		}
	}

	const bool allowAsyncCompute = rg.m_getUseAsyncCompute ? rg.m_getUseAsyncCompute() : true;
	const bool enableQueueSchedulingLogging = rg.m_getQueueSchedulingEnableLogging ? rg.m_getQueueSchedulingEnableLogging() : false;
	const double widthScale = rg.m_getQueueSchedulingWidthScale ? static_cast<double>(rg.m_getQueueSchedulingWidthScale()) : 1.0;
	const double penaltyBias = rg.m_getQueueSchedulingPenaltyBias ? static_cast<double>(rg.m_getQueueSchedulingPenaltyBias()) : 0.0;
	const double minPenalty = rg.m_getQueueSchedulingMinPenalty ? static_cast<double>(rg.m_getQueueSchedulingMinPenalty()) : 1.0;
	const double resourcePressureWeight = rg.m_getQueueSchedulingResourcePressureWeight ? static_cast<double>(rg.m_getQueueSchedulingResourcePressureWeight()) : 1.0;
	const double uavPressureWeight = rg.m_getQueueSchedulingUavPressureWeight ? static_cast<double>(rg.m_getQueueSchedulingUavPressureWeight()) : 0.5;
	auto queueKindName = [](QueueKind kind) -> const char* {
		switch (kind) {
		case QueueKind::Graphics: return "graphics";
		case QueueKind::Compute:  return "compute";
		case QueueKind::Copy:     return "copy";
		default:                 return "unknown";
		}
	};

	for (size_t kindIndex = 0; kindIndex < static_cast<size_t>(QueueKind::Count); ++kindIndex) {
		const QueueKind kind = static_cast<QueueKind>(kindIndex);
		auto& slots = slotsByKind[kindIndex];
		auto& autoAssignableSlots = autoAssignableSlotsByKind[kindIndex];
		if (slots.empty()) {
			continue;
		}

		std::unordered_set<size_t> pinnedSlots;
		std::unordered_set<uint64_t> uniqueTouchedIDs;
		std::unordered_map<uint32_t, size_t> widthByLevel;
		size_t compatibleNodeCount = 0;
		size_t totalTouched = 0;
		size_t totalUAV = 0;

		for (size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
			const auto& node = nodes[nodeIndex];
			bool compatibleWithKind = false;
			for (size_t slot : node.compatibleQueueSlots) {
				if (slot < slotCount && rg.m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot))) == kind) {
					compatibleWithKind = true;
					break;
				}
			}
			if (!compatibleWithKind) {
				continue;
			}

			++compatibleNodeCount;
			++widthByLevel[level[nodeIndex]];
			totalTouched += node.touchedIDs.size();
			totalUAV += node.uavIDs.size();
			uniqueTouchedIDs.insert(node.touchedIDs.begin(), node.touchedIDs.end());

			if (node.passIndex < passes.size() && passHasExplicitQueuePin(passes[node.passIndex])) {
				const size_t pinnedSlot = node.compatibleQueueSlots.empty() ? node.queueSlot : node.compatibleQueueSlots.front();
				if (pinnedSlot < slotCount
					&& rg.m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(pinnedSlot))) == kind) {
					pinnedSlots.insert(pinnedSlot);
				}
			}
		}

		if (kind == QueueKind::Graphics) {
			activeSlots[slots.front()] = 1;
			for (size_t pinnedSlot : pinnedSlots) {
				activeSlots[pinnedSlot] = 1;
			}
			continue;
		}

		if (compatibleNodeCount == 0 && pinnedSlots.empty()) {
			continue;
		}

		size_t maxLevelWidth = 0;
		for (const auto& [_, width] : widthByLevel) {
			maxLevelWidth = (std::max)(maxLevelWidth, width);
		}

		double resourcePressure = 1.0;
		if (!uniqueTouchedIDs.empty()) {
			resourcePressure = static_cast<double>(totalTouched) / static_cast<double>(uniqueTouchedIDs.size());
		}
		const double uavPressure = totalTouched == 0 ? 0.0 : static_cast<double>(totalUAV) / static_cast<double>(totalTouched);
		const double weightedPressure = penaltyBias + resourcePressureWeight * resourcePressure + uavPressureWeight * uavPressure;
		const double parallelismPenalty = (std::max)(minPenalty, weightedPressure);
		const double widthToPenaltyRatio = parallelismPenalty > 0.0
			? ((widthScale * static_cast<double>(maxLevelWidth)) / parallelismPenalty)
			: 0.0;
		size_t targetCount = compatibleNodeCount > 0 ? 1u : 0u;
		if (maxLevelWidth > 0) {
			targetCount = static_cast<size_t>(std::ceil(widthToPenaltyRatio));
			targetCount = (std::max)(size_t(1), targetCount);
		}
		if (kind == QueueKind::Compute && !allowAsyncCompute) {
			targetCount = (std::min)(targetCount, size_t(1));
		}
		targetCount = (std::min)(targetCount, autoAssignableSlots.size());
		targetCount = (std::max)(targetCount, pinnedSlots.size());

		for (size_t pinnedSlot : pinnedSlots) {
			activeSlots[pinnedSlot] = 1;
		}
		if (compatibleNodeCount > 0) {
			if (!autoAssignableSlots.empty()) {
				activeSlots[autoAssignableSlots.front()] = 1;
			}
		}

		size_t activeCount = 0;
		for (size_t slot : autoAssignableSlots) {
			activeCount += activeSlots[slot] ? 1u : 0u;
		}

		for (size_t slot : autoAssignableSlots) {
			if (activeCount >= targetCount) {
				break;
			}
			if (activeSlots[slot]) {
				continue;
			}
			activeSlots[slot] = 1;
			++activeCount;
		}

		if (enableQueueSchedulingLogging) {
			std::ostringstream activeSlotStream;
			bool firstActiveSlot = true;
			for (size_t slot : slots) {
				if (!activeSlots[slot]) {
					continue;
				}
				if (!firstActiveSlot) {
					activeSlotStream << ",";
				}
				activeSlotStream << slot;
				firstActiveSlot = false;
			}

			std::ostringstream pinnedSlotStream;
			bool firstPinnedSlot = true;
			for (size_t slot : pinnedSlots) {
				if (!firstPinnedSlot) {
					pinnedSlotStream << ",";
				}
				pinnedSlotStream << slot;
				firstPinnedSlot = false;
			}

			spdlog::info(
				"RG queue planner [{}]: registered={} autoAssignable={} compatibleNodes={} maxLevelWidth={} resourcePressure={:.2f} uavPressure={:.2f} widthScale={:.2f} penaltyBias={:.2f} minPenalty={:.2f} resourceWeight={:.2f} uavWeight={:.2f} weightedPressure={:.2f} parallelismPenalty={:.2f} widthToPenaltyRatio={:.2f} targetActive={} activeSlots=[{}] pinnedSlots=[{}] asyncComputeAllowed={}",
				queueKindName(kind),
				slots.size(),
				autoAssignableSlots.size(),
				compatibleNodeCount,
				maxLevelWidth,
				resourcePressure,
				uavPressure,
				widthScale,
				penaltyBias,
				minPenalty,
				resourcePressureWeight,
				uavPressureWeight,
				weightedPressure,
				parallelismPenalty,
				widthToPenaltyRatio,
				targetCount,
				activeSlotStream.str(),
				pinnedSlotStream.str(),
				allowAsyncCompute);
		}
	}

	return activeSlots;
}

bool RenderGraph::AddEdgeDedup(
	size_t from, size_t to,
	std::vector<Node>& nodes,
	std::unordered_set<uint64_t>& edgeSet)
{
	if (from == to) return false;
	uint64_t key = (uint64_t(from) << 32) | uint64_t(to);
	if (!edgeSet.insert(key).second) return false;

	nodes[from].out.push_back(to);
	nodes[to].in.push_back(from);
	nodes[to].indegree++;
	return true;
}

bool RenderGraph::BuildDependencyGraph(
	std::vector<Node>& nodes)
{
	return BuildDependencyGraph(nodes, {});
}

bool RenderGraph::BuildDependencyGraph(
	std::vector<Node>& nodes,
	std::span<const std::pair<size_t, size_t>> explicitEdges)
{
	std::unordered_map<uint64_t, SeqState> seq;
	{
		size_t totalAccesses = 0;
		for (const auto& node : nodes) totalAccesses += node.accessByID.size();
		seq.reserve(totalAccesses);
	}

	std::unordered_set<uint64_t> edgeSet;
	edgeSet.reserve(nodes.size() * 8);

	// build deps in ORIGINAL order
	for (size_t i = 0; i < nodes.size(); ++i) {
		auto& node = nodes[i];

		for (auto& [rid, kind] : node.accessByID) {
			auto& s = seq[rid];

			if (kind == AccessKind::Read) {
				if (s.lastWriter) AddEdgeDedup(*s.lastWriter, i, nodes, edgeSet);
				s.readsSinceWrite.push_back(i);
			}
			else { // Write
				if (s.lastWriter) AddEdgeDedup(*s.lastWriter, i, nodes, edgeSet);
				for (size_t r : s.readsSinceWrite)
					AddEdgeDedup(r, i, nodes, edgeSet);
				s.readsSinceWrite.clear();
				s.lastWriter = i;
			}
		}
	}

	// Apply explicit edges (e.g. "After(passName)")
	for (auto const& e : explicitEdges) {
		if (e.first >= nodes.size() || e.second >= nodes.size()) continue;
		AddEdgeDedup(e.first, e.second, nodes, edgeSet);
	}

	// topo + criticality (longest path)
	std::vector<uint32_t> indeg(nodes.size());
	for (size_t i = 0; i < nodes.size(); ++i) indeg[i] = nodes[i].indegree;

	std::vector<size_t> q;
	q.reserve(nodes.size());
	for (size_t i = 0; i < nodes.size(); ++i)
		if (indeg[i] == 0) q.push_back(i);

	std::vector<size_t> topo;
	topo.reserve(nodes.size());

	for (size_t head = 0; head < q.size(); ++head) {
		size_t u = q[head];
		topo.push_back(u);
		for (size_t v : nodes[u].out) {
			if (--indeg[v] == 0) q.push_back(v);
		}
	}

	if (topo.size() != nodes.size()) {
		// cycle: invalid graph
		return false;
	}

	// reverse topo DP
	for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
		size_t u = *it;
		uint32_t best = 0;
		for (size_t v : nodes[u].out)
			best = std::max(best, uint32_t(1 + nodes[v].criticality));
		nodes[u].criticality = best;
	}

	return true;
}

void RenderGraph::CommitPassToBatch(
	RenderGraph& rg,
	AnyPassAndResources& pr,
	const Node& node,

	unsigned int currentBatchIndex,
	PassBatch& currentBatch,

	std::vector<std::unordered_set<uint64_t>>& queueUAVs,

	std::vector<std::unordered_map<uint64_t, unsigned int>>& batchOfLastQueueTransition,
	std::vector<std::unordered_map<uint64_t, unsigned int>>& batchOfLastQueueProducer,
	std::vector<std::unordered_map<uint64_t, unsigned int>>& batchOfLastQueueUsage,
	std::unordered_set<uint64_t>& scratchTransitioned,
	std::unordered_set<uint64_t>& scratchFallback,
	std::vector<ResourceTransition>& scratchTransitions)
{
	const size_t passQueueSlot = node.assignedQueueSlot.value_or(node.queueSlot);
	const size_t queueCount = currentBatch.QueueCount();
	const size_t gfxSlot = QueueIndex(QueueKind::Graphics);
	scratchTransitioned.clear();
	auto& resourcesTransitionedThisPass = scratchTransitioned;

	if (pr.type == PassType::Compute) {
		auto& pass = std::get<ComputePassAndResources>(pr.pass);

		scratchFallback.clear();
		auto& fallbackResourceIDs = scratchFallback;
		rg.ProcessResourceRequirements(
			passQueueSlot,
			pass.resources.frameResourceRequirements,
			batchOfLastQueueUsage,
			batchOfLastQueueTransition,
			currentBatchIndex,
			currentBatch,
			resourcesTransitionedThisPass,
			fallbackResourceIDs,
			scratchTransitions);

		// For fallback transitions (delegated to graphics queue in this batch's
		// BeforePasses), update the graphics transition tracking and add waits so the
		// graphics queue doesn't start transitioning before prior producers finish.
		if (!fallbackResourceIDs.empty()) {
			for (auto& resID : fallbackResourceIDs) {
				batchOfLastQueueTransition[gfxSlot][resID] = currentBatchIndex;
			}
			for (size_t qi = 0; qi < queueCount; ++qi) {
				if (qi == gfxSlot) continue;
				int latestBatch = -1;
				for (auto& resID : fallbackResourceIDs) {
					auto itT = batchOfLastQueueTransition[qi].find(resID);
					if (itT != batchOfLastQueueTransition[qi].end())
						latestBatch = std::max(latestBatch, (int)itT->second);
					auto itU = batchOfLastQueueUsage[qi].find(resID);
					if (itU != batchOfLastQueueUsage[qi].end())
						latestBatch = std::max(latestBatch, (int)itU->second);
				}
				if (latestBatch > 0 && static_cast<unsigned int>(latestBatch) != currentBatchIndex) {
					rg.batches[latestBatch].MarkQueueSignal(BatchSignalPhase::AfterCompletion, qi);
					currentBatch.AddQueueWait(
						BatchWaitPhase::BeforeTransitions,
						gfxSlot,
						qi,
						rg.batches[latestBatch].GetQueueSignalFenceValue(BatchSignalPhase::AfterCompletion, qi));
				}
			}
		}

		currentBatch.Passes(passQueueSlot).emplace_back(pass);

		for (auto& exit : pass.resources.internalTransitions) {
			std::vector<ResourceTransition> _;
			auto pRes = _registry.Resolve(exit.first.resource);
			auto& compileTracker = GetOrCreateCompileTracker(pRes, exit.first.resource.GetGlobalResourceID());
			compileTracker.Apply(
				exit.first.range, nullptr, exit.second, _); // TODO: Do we really need the ptr?
			SortedInsert(currentBatch.internallyTransitionedResources, exit.first.resource.GetGlobalResourceID());
		}

		for (auto& req : pass.resources.frameResourceRequirements) {
			uint64_t id = req.resourceHandleAndRange.resource.GetGlobalResourceID();
			SortedInsert(currentBatch.allResources, id);
			batchOfLastQueueUsage[passQueueSlot][id] = currentBatchIndex;
			if (AccessTypeIsWriteType(req.state.access)) {
				batchOfLastQueueProducer[passQueueSlot][id] = currentBatchIndex;
			}
		}

		// track UAV usage for cross-queue "same batch" rejection
		queueUAVs[passQueueSlot].insert(node.uavIDs.begin(), node.uavIDs.end());

		for (size_t qi = 0; qi < queueCount; ++qi) {
			if (qi == passQueueSlot) {
				continue;
			}

			rg.applySynchronization(
				passQueueSlot,
				qi,
				currentBatch,
				currentBatchIndex,
				std::get<ComputePassAndResources>(pr.pass),
				batchOfLastQueueTransition[qi],
				batchOfLastQueueProducer[qi],
				batchOfLastQueueUsage[qi],
				resourcesTransitionedThisPass);
		}

	}
	else if (pr.type == PassType::Render) {
		auto& pass = std::get<RenderPassAndResources>(pr.pass);

		scratchFallback.clear();
		auto& fallbackResourceIDs = scratchFallback;
		rg.ProcessResourceRequirements(
			passQueueSlot,
			pass.resources.frameResourceRequirements,
			batchOfLastQueueUsage,
			batchOfLastQueueTransition,
			currentBatchIndex,
			currentBatch,
			resourcesTransitionedThisPass,
			fallbackResourceIDs,
			scratchTransitions);

		// Render passes normally run on the graphics queue, so fallbacks should
		// be rare here, but handle them for correctness with Compute render passes.
		if (!fallbackResourceIDs.empty()) {
			for (auto& resID : fallbackResourceIDs) {
				batchOfLastQueueTransition[gfxSlot][resID] = currentBatchIndex;
			}
			for (size_t qi = 0; qi < queueCount; ++qi) {
				if (qi == gfxSlot) continue;
				int latestBatch = -1;
				for (auto& resID : fallbackResourceIDs) {
					auto itT = batchOfLastQueueTransition[qi].find(resID);
					if (itT != batchOfLastQueueTransition[qi].end())
						latestBatch = std::max(latestBatch, (int)itT->second);
					auto itU = batchOfLastQueueUsage[qi].find(resID);
					if (itU != batchOfLastQueueUsage[qi].end())
						latestBatch = std::max(latestBatch, (int)itU->second);
				}
				if (latestBatch > 0 && static_cast<unsigned int>(latestBatch) != currentBatchIndex) {
					rg.batches[latestBatch].MarkQueueSignal(BatchSignalPhase::AfterCompletion, qi);
					currentBatch.AddQueueWait(
						BatchWaitPhase::BeforeTransitions,
						gfxSlot,
						qi,
						rg.batches[latestBatch].GetQueueSignalFenceValue(BatchSignalPhase::AfterCompletion, qi));
				}
			}
		}

		currentBatch.Passes(passQueueSlot).emplace_back(pass);

		for (auto& exit : pass.resources.internalTransitions) {
			std::vector<ResourceTransition> _;
			auto pRes = _registry.Resolve(exit.first.resource);
			auto& compileTracker = GetOrCreateCompileTracker(pRes, exit.first.resource.GetGlobalResourceID());
			compileTracker.Apply(
				exit.first.range, nullptr, exit.second, _);
			SortedInsert(currentBatch.internallyTransitionedResources, exit.first.resource.GetGlobalResourceID());
		}

		for (auto& req : pass.resources.frameResourceRequirements) {
			uint64_t id = req.resourceHandleAndRange.resource.GetGlobalResourceID();
			SortedInsert(currentBatch.allResources, id);
			batchOfLastQueueUsage[passQueueSlot][id] = currentBatchIndex;
			if (AccessTypeIsWriteType(req.state.access)) {
				batchOfLastQueueProducer[passQueueSlot][id] = currentBatchIndex;
			}
		}

		queueUAVs[passQueueSlot].insert(node.uavIDs.begin(), node.uavIDs.end());

		for (size_t qi = 0; qi < queueCount; ++qi) {
			if (qi == passQueueSlot) {
				continue;
			}

			rg.applySynchronization(
				passQueueSlot,
				qi,
				currentBatch,
				currentBatchIndex,
				std::get<RenderPassAndResources>(pr.pass),
				batchOfLastQueueTransition[qi],
				batchOfLastQueueProducer[qi],
				batchOfLastQueueUsage[qi],
				resourcesTransitionedThisPass);
		}
	}
	else if (pr.type == PassType::Copy) {
		auto& pass = std::get<CopyPassAndResources>(pr.pass);

		scratchFallback.clear();
		auto& fallbackResourceIDs = scratchFallback;
		rg.ProcessResourceRequirements(
			passQueueSlot,
			pass.resources.frameResourceRequirements,
			batchOfLastQueueUsage,
			batchOfLastQueueTransition,
			currentBatchIndex,
			currentBatch,
			resourcesTransitionedThisPass,
			fallbackResourceIDs,
			scratchTransitions);

		// For fallback transitions (delegated to graphics queue in this batch's
		// BeforePasses), update the graphics transition tracking and add waits so the
		// graphics queue doesn't start transitioning before prior producers finish.
		if (!fallbackResourceIDs.empty()) {
			for (auto& resID : fallbackResourceIDs) {
				batchOfLastQueueTransition[gfxSlot][resID] = currentBatchIndex;
			}
			for (size_t qi = 0; qi < queueCount; ++qi) {
				if (qi == gfxSlot) continue;
				int latestBatch = -1;
				for (auto& resID : fallbackResourceIDs) {
					auto itT = batchOfLastQueueTransition[qi].find(resID);
					if (itT != batchOfLastQueueTransition[qi].end())
						latestBatch = std::max(latestBatch, (int)itT->second);
					auto itU = batchOfLastQueueUsage[qi].find(resID);
					if (itU != batchOfLastQueueUsage[qi].end())
						latestBatch = std::max(latestBatch, (int)itU->second);
				}
				if (latestBatch > 0 && static_cast<unsigned int>(latestBatch) != currentBatchIndex) {
					rg.batches[latestBatch].MarkQueueSignal(BatchSignalPhase::AfterCompletion, qi);
					currentBatch.AddQueueWait(
						BatchWaitPhase::BeforeTransitions,
						gfxSlot,
						qi,
						rg.batches[latestBatch].GetQueueSignalFenceValue(BatchSignalPhase::AfterCompletion, qi));
				}
			}
		}

		currentBatch.Passes(passQueueSlot).emplace_back(pass);

		for (auto& exit : pass.resources.internalTransitions) {
			std::vector<ResourceTransition> _;
			auto pRes = _registry.Resolve(exit.first.resource);
			auto& compileTracker = GetOrCreateCompileTracker(pRes, exit.first.resource.GetGlobalResourceID());
			compileTracker.Apply(
				exit.first.range, nullptr, exit.second, _);
			SortedInsert(currentBatch.internallyTransitionedResources, exit.first.resource.GetGlobalResourceID());
		}

		for (auto& req : pass.resources.frameResourceRequirements) {
			uint64_t id = req.resourceHandleAndRange.resource.GetGlobalResourceID();
			SortedInsert(currentBatch.allResources, id);
			batchOfLastQueueUsage[passQueueSlot][id] = currentBatchIndex;
			if (AccessTypeIsWriteType(req.state.access)) {
				batchOfLastQueueProducer[passQueueSlot][id] = currentBatchIndex;
			}
		}

		queueUAVs[passQueueSlot].insert(node.uavIDs.begin(), node.uavIDs.end());

		for (size_t qi = 0; qi < queueCount; ++qi) {
			if (qi == passQueueSlot) {
				continue;
			}

			rg.applySynchronization(
				passQueueSlot,
				qi,
				currentBatch,
				currentBatchIndex,
				std::get<CopyPassAndResources>(pr.pass),
				batchOfLastQueueTransition[qi],
				batchOfLastQueueProducer[qi],
				batchOfLastQueueUsage[qi],
				resourcesTransitionedThisPass);
		}
	}
}

void RenderGraph::AutoScheduleAndBuildBatches(
	RenderGraph& rg,
	std::vector<AnyPassAndResources>& passes,
	std::vector<Node>& nodes)
{
	struct QueueSchedulingDiagnostics {
		uint32_t candidateChecks = 0;
		uint32_t assignedPasses = 0;
		uint32_t rejectedInactive = 0;
		uint32_t rejectedCrossQueuePred = 0;
		uint32_t rejectedBatchNeeded = 0;
	};

	// Working indegrees
	std::vector<uint32_t> indeg(nodes.size());
	for (size_t i = 0; i < nodes.size(); ++i) indeg[i] = nodes[i].indegree;

	std::vector<size_t> ready;
	ready.reserve(nodes.size());
	for (size_t i = 0; i < nodes.size(); ++i)
		if (indeg[i] == 0) ready.push_back(i);

	std::vector<uint8_t> inBatch(nodes.size(), 0);
	std::vector<size_t>  batchMembers;
	batchMembers.reserve(nodes.size());

	auto openNewBatch = [&]() -> PassBatch {
		const size_t queueCount = rg.m_queueRegistry.SlotCount();
		PassBatch b(queueCount);
		for (size_t qi = 0; qi < queueCount; ++qi) {
			b.SetQueueSignalFenceValue(RenderGraph::BatchSignalPhase::AfterTransitions, qi, rg.GetNextQueueFenceValue(qi));
			b.SetQueueSignalFenceValue(RenderGraph::BatchSignalPhase::AfterExecution, qi, rg.GetNextQueueFenceValue(qi));
			b.SetQueueSignalFenceValue(RenderGraph::BatchSignalPhase::AfterCompletion, qi, rg.GetNextQueueFenceValue(qi));
		}
		return b;
		};

	PassBatch currentBatch = openNewBatch();
	unsigned int currentBatchIndex = 1; // Start at batch 1- batch 0 is reserved for inserting transitions before first batch

	const size_t queueCount = rg.m_queueRegistry.SlotCount();

	std::vector<std::unordered_set<uint64_t>> queueUAVs(queueCount);

	std::vector<std::unordered_map<uint64_t, unsigned int>> batchOfLastQueueTransition(queueCount);
	std::vector<std::unordered_map<uint64_t, unsigned int>> batchOfLastQueueProducer(queueCount);
	std::vector<std::unordered_map<uint64_t, unsigned int>> batchOfLastQueueUsage(queueCount);

	// Scratch sets reused across CommitPassToBatch calls to avoid per-call allocation
	std::unordered_set<uint64_t> scratchTransitioned;
	std::unordered_set<uint64_t> scratchFallback;
	std::vector<ResourceTransition> scratchTransitions;
	const bool enableQueueSchedulingLogging = rg.m_getQueueSchedulingEnableLogging ? rg.m_getQueueSchedulingEnableLogging() : false;
	std::vector<QueueSchedulingDiagnostics> queueDiagnostics(queueCount);
	auto queueKindName = [&rg](size_t queueIndex) -> const char* {
		if (queueIndex >= rg.m_queueRegistry.SlotCount()) {
			return "unknown";
		}
		switch (rg.m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(queueIndex)))) {
		case QueueKind::Graphics: return "graphics";
		case QueueKind::Compute:  return "compute";
		case QueueKind::Copy:     return "copy";
		default:                 return "unknown";
		}
	};

	auto closeBatch = [&]() {
		// clear inBatch marks for members
		for (size_t p : batchMembers) inBatch[p] = 0;
		batchMembers.clear();

		rg.batches.push_back(std::move(currentBatch));
		currentBatch = openNewBatch();
		for (auto& queueUAVSet : queueUAVs) {
			queueUAVSet.clear();
		}
		++currentBatchIndex;
		};

	size_t remaining = nodes.size();

	while (remaining > 0) {
		// Collect "fits" and pick best by heuristic
		int bestIdxInReady = -1;
		size_t bestQueueSlot = 0;
		double bestScore = -1e300;

		std::vector<uint8_t> batchHasQueue(queueCount);
		for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
			batchHasQueue[queueIndex] = currentBatch.HasPasses(queueIndex);
		}

		// Precompute the "other queues' UAVs" sorted vector for each queue kind once per iteration,
		// instead of rebuilding it for every candidate in the ready list.
		std::vector<std::vector<uint64_t>> otherQueueUAVsByQueue(queueCount);
		for (size_t q = 0; q < queueUAVs.size(); ++q) {
			auto& merged = otherQueueUAVsByQueue[q];
			for (size_t other = 0; other < queueUAVs.size(); ++other) {
				if (other == q) continue;
				merged.insert(merged.end(), queueUAVs[other].begin(), queueUAVs[other].end());
			}
			std::sort(merged.begin(), merged.end());
			merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
		}

		for (int ri = 0; ri < (int)ready.size(); ++ri) {
			size_t ni = ready[ri];

			auto& n = nodes[ni];

			PassView view = GetPassView(passes[n.passIndex]);

			for (size_t nodeQueueSlot : n.compatibleQueueSlots) {
				if (nodeQueueSlot >= queueCount) {
					continue;
				}
				queueDiagnostics[nodeQueueSlot].candidateChecks++;
				if (nodeQueueSlot >= rg.m_activeQueueSlotsThisFrame.size() || !rg.m_activeQueueSlotsThisFrame[nodeQueueSlot]) {
					queueDiagnostics[nodeQueueSlot].rejectedInactive++;
					continue;
				}

				// Extra constraint: disallow cross-queue deps within same batch.
				if (batchHasQueue[nodeQueueSlot]) {
					bool hasCrossQueuePredInBatch = false;
					for (size_t pred : n.in) {
						if (!inBatch[pred]) {
							continue;
						}
						const size_t predQueueSlot = nodes[pred].assignedQueueSlot.value_or(nodes[pred].queueSlot);
						if (predQueueSlot != nodeQueueSlot) {
							hasCrossQueuePredInBatch = true;
							break;
						}
					}
					if (hasCrossQueuePredInBatch) {
						queueDiagnostics[nodeQueueSlot].rejectedCrossQueuePred++;
						continue;
					}
				}

				if (rg.IsNewBatchNeeded(
					*view.reqs,
					*view.internalTransitions,
					currentBatch.passBatchTrackers,
					currentBatch.internallyTransitionedResources,
					currentBatch.allResources,
					otherQueueUAVsByQueue[nodeQueueSlot]))
				{
					queueDiagnostics[nodeQueueSlot].rejectedBatchNeeded++;
					continue;
				}

				// Score: pack by reusing resources already in batch, and encourage overlap
				int reuse = 0, fresh = 0;
				for (uint64_t rid : n.touchedIDs) {
					if (std::binary_search(currentBatch.allResources.begin(), currentBatch.allResources.end(), rid)) ++reuse;
					else ++fresh;
				}

				double score = 3.0 * reuse - 1.0 * fresh;

				// Encourage having more queues represented when legal.
				if (!batchHasQueue[nodeQueueSlot]) score += 2.0;
				// Encourage spreading compatible work across less-populated queues.
				score -= 0.25 * double(currentBatch.Passes(nodeQueueSlot).size());

				// Tie-break
				score += 0.05 * double(n.criticality);

				// Deterministic tie-break: prefer earlier original order slightly
				score += 1e-6 * double(nodes.size() - n.originalOrder);

				if (score > bestScore) {
					bestScore = score;
					bestIdxInReady = ri;
					bestQueueSlot = nodeQueueSlot;
				}
			}
		}

		if (bestIdxInReady < 0) {
			// Nothing ready fits: must end batch
			bool hasAnyQueuedPasses = false;
			for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
				hasAnyQueuedPasses = hasAnyQueuedPasses || !currentBatch.Passes(queueIndex).empty();
			}
			if (hasAnyQueuedPasses) {
				closeBatch();
				continue;
			}
			else {
				// Should be rare; fall back by forcing one ready pass in.
				// If this happens, IsNewBatchNeeded is likely too strict on empty batch.
				size_t ni = ready.front();
				auto& n = nodes[ni];
				size_t fallbackSlot = n.queueSlot;
				for (size_t compatibleSlot : n.compatibleQueueSlots) {
					if (compatibleSlot < rg.m_activeQueueSlotsThisFrame.size() && rg.m_activeQueueSlotsThisFrame[compatibleSlot]) {
						fallbackSlot = compatibleSlot;
						break;
					}
				}
				n.assignedQueueSlot = fallbackSlot;
				if (n.passIndex < rg.m_assignedQueueSlotsByFramePass.size()) {
					rg.m_assignedQueueSlotsByFramePass[n.passIndex] = *n.assignedQueueSlot;
				}
				if (fallbackSlot < queueDiagnostics.size()) {
					queueDiagnostics[fallbackSlot].assignedPasses++;
				}
				CommitPassToBatch(
					rg, passes[n.passIndex], n,
					currentBatchIndex, currentBatch,
					queueUAVs,
					batchOfLastQueueTransition,
					batchOfLastQueueProducer,
					batchOfLastQueueUsage,
					scratchTransitioned,
					scratchFallback,
					scratchTransitions);

				inBatch[ni] = 1;
				batchMembers.push_back(ni);

				// Pop from ready
				ready[0] = ready.back();
				ready.pop_back();

				for (size_t v : nodes[ni].out) {
					if (--indeg[v] == 0) ready.push_back(v);
				}
				--remaining;
				if (rg.m_getHeavyDebug && rg.m_getHeavyDebug()) {
					closeBatch();
				}
				continue;
			}
		}

		// Commit chosen pass
		size_t chosenNodeIndex = ready[bestIdxInReady];
		auto& chosen = nodes[chosenNodeIndex];
		chosen.assignedQueueSlot = bestQueueSlot;
		if (chosen.passIndex < rg.m_assignedQueueSlotsByFramePass.size()) {
			rg.m_assignedQueueSlotsByFramePass[chosen.passIndex] = bestQueueSlot;
		}
		if (bestQueueSlot < queueDiagnostics.size()) {
			queueDiagnostics[bestQueueSlot].assignedPasses++;
		}

		CommitPassToBatch(
			rg, passes[chosen.passIndex], chosen,
			currentBatchIndex, currentBatch,
			queueUAVs,
			batchOfLastQueueTransition,
			batchOfLastQueueProducer,
			batchOfLastQueueUsage,
			scratchTransitioned,
			scratchFallback,
			scratchTransitions);

		inBatch[chosenNodeIndex] = 1;
		batchMembers.push_back(chosenNodeIndex);

		// Remove from ready
		ready[bestIdxInReady] = ready.back();
		ready.pop_back();

		// Release successors
		for (size_t v : chosen.out) {
			if (--indeg[v] == 0) ready.push_back(v);
		}

		--remaining;

		// Heavy-debug: isolate every pass into its own batch.
		if (rg.m_getHeavyDebug && rg.m_getHeavyDebug()) {
			closeBatch();
		}
	}

	// Final batch
	bool hasAnyQueuedPasses = false;
	for (size_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
		hasAnyQueuedPasses = hasAnyQueuedPasses || !currentBatch.Passes(queueIndex).empty();
	}
	if (hasAnyQueuedPasses) {
		rg.batches.push_back(std::move(currentBatch));
	}

	if (enableQueueSchedulingLogging) {
		for (size_t queueIndex = 0; queueIndex < queueDiagnostics.size(); ++queueIndex) {
			const auto& d = queueDiagnostics[queueIndex];
			spdlog::info(
				"RG queue scheduler [slot={} kind={} active={}]: candidateChecks={} assignedPasses={} rejectedInactive={} rejectedCrossQueuePred={} rejectedBatchNeeded={}",
				queueIndex,
				queueKindName(queueIndex),
				(queueIndex < rg.m_activeQueueSlotsThisFrame.size() ? int(rg.m_activeQueueSlotsThisFrame[queueIndex]) : 0),
				d.candidateChecks,
				d.assignedPasses,
				d.rejectedInactive,
				d.rejectedCrossQueuePred,
				d.rejectedBatchNeeded);
		}
	}

	// Coalesce redundant waits: for each (dstQueue, srcQueue) pair in a batch,
	// eliminate later-phase waits that are subsumed by earlier-phase waits, and
	// promote cross-batch fence values to earlier phases when safe.
	//
	// GPU execution order within a queue for one batch:
	//   BeforeTransitions waits -> transitions -> AfterTransitions signal
	//   BeforeExecution waits   -> passes      -> AfterExecution signal
	//   BeforeAfterPasses waits -> post-transitions -> AfterCompletion signal
	//
	// Fence values on a given queue are monotonically increasing, so a wait for
	// fence F at an earlier phase automatically satisfies any wait for F' <= F
	// at a later phase on the same source queue.
	//
	// Promoting a fence to an earlier phase is only safe when the signal producing
	// that fence will fire independently (i.e., it's from a PREVIOUS batch, not the
	// same batch). Same-batch fences reference signals produced within this batch's
	// own execution; promoting them to an earlier phase could create circular
	// cross-queue dependencies (GPU deadlock).
	for (auto& batch : rg.batches) {
		const size_t batchQueueCount = batch.QueueCount();
		for (size_t dst = 0; dst < batchQueueCount; ++dst) {
			for (size_t src = 0; src < batchQueueCount; ++src) {
				if (dst == src) continue;

				int enabledCount = 0;
				for (size_t phase = 0; phase < PassBatch::kWaitPhaseCount; ++phase) {
					if (batch.queueWaitEnabled[phase][dst][src]) ++enabledCount;
				}
				if (enabledCount <= 1) continue;

				// Helper: true if the fence value matches one of this batch's own
				// pre-allocated signal fence values for the source queue.
				auto isSameBatchFence = [&](UINT64 f) -> bool {
					for (size_t sp = 0; sp < PassBatch::kSignalPhaseCount; ++sp) {
						if (f == batch.queueSignalFenceValue[sp][src]) return true;
					}
					return false;
				};

				// Step 1: Strict subsumption - remove later-phase waits whose fence
				// value is <= an earlier enabled phase's fence value.
				for (size_t i = 0; i < PassBatch::kWaitPhaseCount; ++i) {
					if (!batch.queueWaitEnabled[i][dst][src]) continue;
					UINT64 fi = batch.queueWaitFenceValue[i][dst][src];
					for (size_t j = i + 1; j < PassBatch::kWaitPhaseCount; ++j) {
						if (!batch.queueWaitEnabled[j][dst][src]) continue;
						if (batch.queueWaitFenceValue[j][dst][src] <= fi) {
							batch.queueWaitEnabled[j][dst][src] = false;
							batch.queueWaitFenceValue[j][dst][src] = 0;
						}
					}
				}

				// Step 2: Promote cross-batch fences to the earliest phase.
				// Only promote fences that are NOT from the same batch.
				int earliest = -1;
				for (size_t phase = 0; phase < PassBatch::kWaitPhaseCount; ++phase) {
					if (batch.queueWaitEnabled[phase][dst][src]) {
						earliest = static_cast<int>(phase);
						break;
					}
				}
				if (earliest < 0) continue;

				for (size_t j = static_cast<size_t>(earliest) + 1; j < PassBatch::kWaitPhaseCount; ++j) {
					if (!batch.queueWaitEnabled[j][dst][src]) continue;
					UINT64 fj = batch.queueWaitFenceValue[j][dst][src];

					// Same-batch fences must stay at their original phase to avoid
					// creating circular cross-queue dependencies.
					if (isSameBatchFence(fj)) continue;

					// Promote to earliest phase (take the max)
					if (fj > batch.queueWaitFenceValue[earliest][dst][src]) {
						batch.queueWaitFenceValue[earliest][dst][src] = fj;
					}
					batch.queueWaitEnabled[j][dst][src] = false;
					batch.queueWaitFenceValue[j][dst][src] = 0;
				}

				// Step 3: Re-run subsumption - the promoted fence may now
				// subsume same-batch waits that we couldn't promote.
				for (size_t i = 0; i < PassBatch::kWaitPhaseCount; ++i) {
					if (!batch.queueWaitEnabled[i][dst][src]) continue;
					UINT64 fi = batch.queueWaitFenceValue[i][dst][src];
					for (size_t j = i + 1; j < PassBatch::kWaitPhaseCount; ++j) {
						if (!batch.queueWaitEnabled[j][dst][src]) continue;
						if (batch.queueWaitFenceValue[j][dst][src] <= fi) {
							batch.queueWaitEnabled[j][dst][src] = false;
							batch.queueWaitFenceValue[j][dst][src] = 0;
						}
					}
				}
			}
		}
	}

	// Build cross-frame producer tracking from the committed batch schedule.
	// Cross-frame tracking needs to know which queue
	// wrote each resource and in which batch so the next frame can insert
	// frame-start waits.
	{
		std::vector<std::unordered_map<uint64_t, unsigned int>> crossFrameProducer(queueCount);
		for (unsigned int bi = 1; bi < static_cast<unsigned int>(rg.batches.size()); ++bi) {
			auto& batch = rg.batches[bi];
			for (size_t qi = 0; qi < queueCount; ++qi) {
				for (auto& passVariant : batch.Passes(qi)) {
					std::visit([&](const auto& passEntry) {
						for (auto& req : passEntry.resources.frameResourceRequirements) {
							if (AccessTypeIsWriteType(req.state.access)) {
								crossFrameProducer[qi][req.resourceHandleAndRange.resource.GetGlobalResourceID()] = bi;
							}
						}
					}, passVariant);
				}
			}
		}
		rg.m_compiledLastProducerBatchByResourceByQueue = std::move(crossFrameProducer);
	}
}


// Factory for the transition lambda
void RenderGraph::AddTransition(
	std::vector<std::unordered_map<uint64_t, unsigned int>>& batchOfLastQueueUsage,
	std::vector<std::unordered_map<uint64_t, unsigned int>>& batchOfLastQueueTransition,
	unsigned int batchIndex,
	PassBatch& currentBatch,
	size_t passQueueSlot,
	const ResourceRequirement& r,
	std::unordered_set<uint64_t>& outTransitionedResourceIDs,
	std::unordered_set<uint64_t>& outFallbackResourceIDs,
	std::vector<ResourceTransition>& scratchTransitions)
{
	const QueueKind passQueue = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(passQueueSlot));

	auto& resource = r.resourceHandleAndRange.resource;

	// If this triggers, you're probably queueing an operation on an external/ephemeral resource, and then discarding it before the graph can use it.
	if (!resource.IsEphemeral() && !_registry.IsValid(resource)) {
		spdlog::error("Invalid resource handle");
		throw (std::runtime_error("Invalid resource handle in RenderGraph::AddTransition"));
	}
	scratchTransitions.clear();
	auto& transitions = scratchTransitions;
	auto pRes = _registry.Resolve(resource); // TODO: Can we get rid of pRes in transitions?
	auto& compileTracker = GetOrCreateCompileTracker(pRes, resource.GetGlobalResourceID());

	bool isAliasActivation = false;
	if (aliasActivationPending.find(resource.GetGlobalResourceID()) != aliasActivationPending.end()) {
		isAliasActivation = true;
		const bool firstUseIsWrite = AccessTypeIsWriteType(r.state.access);
		const bool firstUseIsCommon = r.state.access == rhi::ResourceAccessType::Common;
		// Common counts as write for alias activation, as this is generally used to indicate that the resource will be
		// transitioned internally by an external system that still uses legacy barriers. Don't abuse this.
		if (firstUseIsWrite || firstUseIsCommon) { 
			const uint64_t id = resource.GetGlobalResourceID();
			auto itSig = aliasPlacementSignatureByID.find(id);
			spdlog::info(
				"RG alias activate: id={} name='{}' signature={} accessAfter={} layoutAfter={} syncAfter={} discard=1",
				id,
				pRes ? pRes->GetName() : std::string("<null>"),
				itSig != aliasPlacementSignatureByID.end() ? itSig->second : 0ull,
				static_cast<uint32_t>(r.state.access),
				static_cast<uint32_t>(r.state.layout),
				static_cast<uint32_t>(r.state.sync));
			transitions.emplace_back(
				pRes,
				r.resourceHandleAndRange.range,
				rhi::ResourceAccessType::None,
				r.state.access,
				rhi::ResourceLayout::Undefined,
				r.state.layout,
				rhi::ResourceSyncState::None,
				r.state.sync,
				true);
		}
		else {
			throw std::runtime_error("Alias activation requires first use to be a write when explicit initialization is disabled");
		}
		std::vector<ResourceTransition> ignored;
		compileTracker.Apply(r.resourceHandleAndRange.range, pRes, r.state, ignored);
		aliasActivationPending.erase(resource.GetGlobalResourceID());
	}
	else {
		compileTracker.Apply(r.resourceHandleAndRange.range, pRes, r.state, transitions);
	}

	if (!transitions.empty()) {
		outTransitionedResourceIDs.insert(resource.GetGlobalResourceID());
	}

	currentBatch.passBatchTrackers[resource.GetGlobalResourceID()] = &compileTracker; // We will need to check subsequent passes against this

	if (transitions.empty()) {
		return;
	}

	bool needsGraphicsQueueForTransitions = false;
	for (auto& transition : transitions) {
		if (!QueueSupportsTransition(passQueue, transition)) {
			needsGraphicsQueueForTransitions = true;
			break;
		}
	}

	const size_t gfxSlot = QueueIndex(QueueKind::Graphics);
	const size_t transitionSlot = (passQueue != QueueKind::Graphics && needsGraphicsQueueForTransitions)
		? gfxSlot : passQueueSlot;

	// Try early placement: move transitions to AfterPasses of the batch where the resource was last used.
	// This reduces GPU idle time by allowing transitions to overlap with unrelated work on other queues.
	// Skip alias activations - those must stay in the consuming batch (discard semantics at first use).
	if (!isAliasActivation) {
		const uint64_t resourceID = resource.GetGlobalResourceID();
		int lastUseBatch = -1;
		for (size_t qi = 0; qi < batchOfLastQueueUsage.size(); ++qi) {
			auto itU = batchOfLastQueueUsage[qi].find(resourceID);
			if (itU != batchOfLastQueueUsage[qi].end() && static_cast<int>(itU->second) > lastUseBatch && itU->second < batchIndex)
				lastUseBatch = static_cast<int>(itU->second);
			auto itT = batchOfLastQueueTransition[qi].find(resourceID);
			if (itT != batchOfLastQueueTransition[qi].end() && static_cast<int>(itT->second) > lastUseBatch && itT->second < batchIndex)
				lastUseBatch = static_cast<int>(itT->second);
		}

		if (lastUseBatch > 0) { // > 0 to skip batch 0 (placeholder with no fence values)
			PassBatch& targetBatch = batches[lastUseBatch];

			for (auto& transition : transitions) {
				targetBatch.Transitions(transitionSlot, BatchTransitionPhase::AfterPasses).push_back(transition);
			}

			// If the transition queue differs from the queue(s) that last used the resource in the target batch,
			// the transition queue must wait for those queues to finish before executing AfterPasses transitions.
			for (size_t qi = 0; qi < batchOfLastQueueUsage.size(); ++qi) {
				if (qi == transitionSlot) continue;

				bool usedInTargetBatch = false;
				auto itU = batchOfLastQueueUsage[qi].find(resourceID);
				if (itU != batchOfLastQueueUsage[qi].end() && itU->second == static_cast<unsigned int>(lastUseBatch))
					usedInTargetBatch = true;
				auto itT = batchOfLastQueueTransition[qi].find(resourceID);
				if (itT != batchOfLastQueueTransition[qi].end() && itT->second == static_cast<unsigned int>(lastUseBatch))
					usedInTargetBatch = true;

				if (usedInTargetBatch) {
					targetBatch.MarkQueueSignal(BatchSignalPhase::AfterExecution, qi);
					targetBatch.AddQueueWait(
						BatchWaitPhase::BeforeAfterPasses,
						transitionSlot,
						qi,
						targetBatch.GetQueueSignalFenceValue(BatchSignalPhase::AfterExecution, qi));
				}
			}

			// Signal AfterCompletion on the transition queue so downstream consumers can wait on it
			targetBatch.MarkQueueSignal(BatchSignalPhase::AfterCompletion, transitionSlot);

			// Update tracking: the transition is now in the earlier batch
			batchOfLastQueueTransition[transitionSlot][resourceID] = lastUseBatch;

			// Do NOT add to outFallbackResourceIDs- applySynchronization will handle
			// cross-queue waits based on the updated tracking maps.
			return;
		}
	}

	// Fallback: place in current batch's BeforePasses (existing behavior for first use or alias activations)
	if (passQueue != QueueKind::Graphics && needsGraphicsQueueForTransitions) {
		// The consuming pass's queue can't support these transitions, so delegate
		// them to the graphics queue within the *current* batch's BeforePasses phase.
		// CommitPassToBatch will set up:
		//   1. BeforeTransitions waits on Graphics for any prior non-graphics producers
		//   2. AfterTransitions signal on Graphics so the consuming queue can wait
		for (auto& transition : transitions) {
			currentBatch.Transitions(gfxSlot, BatchTransitionPhase::BeforePasses).push_back(transition);
		}
		outFallbackResourceIDs.insert(resource.GetGlobalResourceID());
	}
	else {
		for (auto& transition : transitions) {
			currentBatch.Transitions(passQueueSlot, BatchTransitionPhase::BeforePasses).push_back(transition);
		}
	}
}

void RenderGraph::ProcessResourceRequirements(
	size_t passQueueSlot,
	std::vector<ResourceRequirement>& resourceRequirements,
	std::vector<std::unordered_map<uint64_t, unsigned int>>& batchOfLastQueueUsage,
	std::vector<std::unordered_map<uint64_t, unsigned int>>& batchOfLastQueueTransition,
	unsigned int batchIndex,
	PassBatch& currentBatch, std::unordered_set<uint64_t>& outTransitionedResourceIDs,
	std::unordered_set<uint64_t>& outFallbackResourceIDs,
	std::vector<ResourceTransition>& scratchTransitions) {

	for (auto& resourceRequirement : resourceRequirements) {

		const auto& id = resourceRequirement.resourceHandleAndRange.resource.GetGlobalResourceID();

		AddTransition(batchOfLastQueueUsage, batchOfLastQueueTransition, batchIndex, currentBatch, passQueueSlot, resourceRequirement, outTransitionedResourceIDs, outFallbackResourceIDs, scratchTransitions);

		if (AccessTypeIsWriteType(resourceRequirement.state.access)) {
			batchOfLastQueueTransition[passQueueSlot][id] = batchIndex;
		}
	}
}

bool ResolveFirstMipSlice(ResourceRegistry::RegistryHandle r, RangeSpec range, uint32_t& outMip, uint32_t& outSlice) noexcept
{
	const uint32_t totalMips = r.GetNumMipLevels();
	const uint32_t totalSlices = r.GetArraySize();
	if (totalMips == 0 || totalSlices == 0) return false;

	SubresourceRange sr = ResolveRangeSpec(range, totalMips, totalSlices);
	if (sr.isEmpty()) return false;

	outMip = sr.firstMip;
	outSlice = sr.firstSlice;
	return true;
}

RenderGraph::RenderGraph(rhi::Device device) {
	DeviceManager::GetInstance().Initialize(device);

	auto MakeDefaultImmediateDispatch = [&]() noexcept -> rg::imm::ImmediateDispatch
		{
			rg::imm::ImmediateDispatch d{};
			d.user = this;

			d.GetResourceHandle = [](RenderGraph* user, ResourceRegistry::RegistryHandle r) noexcept -> rhi::ResourceHandle {
				Resource* ptr;
				if (r.IsEphemeral()) {
					ptr = r.GetEphemeralPtr();
				}
				else {
					ptr = user->_registry.Resolve(r);
				}
				return ptr ? ptr->GetAPIResource().GetHandle() : rhi::ResourceHandle{};
				};

			d.GetRTV = +[](RenderGraph* user, ResourceRegistry::RegistryHandle r, RangeSpec range) noexcept -> rhi::DescriptorSlot {
				auto* gir = dynamic_cast<GloballyIndexedResource*>(user->_registry.Resolve(r));
				if (!gir || !gir->HasRTV()) return {};

				uint32_t mip = 0, slice = 0;
				if (!ResolveFirstMipSlice(r, range, mip, slice)) return {};

				return gir->GetRTVInfo(mip, slice).slot;
				};

			d.GetDSV = +[](RenderGraph* user, ResourceRegistry::RegistryHandle r, RangeSpec range) noexcept -> rhi::DescriptorSlot {
				auto* gir = dynamic_cast<GloballyIndexedResource*>(user->_registry.Resolve(r));
				if (!gir || !gir->HasDSV()) return {};

				uint32_t mip = 0, slice = 0;
				if (!ResolveFirstMipSlice(r, range, mip, slice)) return {};

				return gir->GetDSVInfo(mip, slice).slot;
				};

			d.GetUavClearInfo = +[](RenderGraph* user, ResourceRegistry::RegistryHandle r, RangeSpec range, rhi::UavClearInfo& out) noexcept -> bool {
				auto* gir = dynamic_cast<GloballyIndexedResource*>(user->_registry.Resolve(r));

				// DX12 path requires both a shader-visible and CPU-visible UAV descriptor.
				if (!gir || !gir->HasUAVShaderVisible() || !gir->HasUAVNonShaderVisible()) return false;

				uint32_t mip = 0, slice = 0;
				if (!ResolveFirstMipSlice(r, range, mip, slice)) return false;

				out.shaderVisible = gir->GetUAVShaderVisibleInfo(mip, slice).slot;
				out.cpuVisible = gir->GetUAVNonShaderVisibleInfo(mip, slice).slot;

				out.resource = gir->GetAPIResource();

				return true;
				};

			return d;
		};

	m_immediateDispatch = MakeDefaultImmediateDispatch();
	if (!m_statisticsService) {
		m_statisticsService = rg::runtime::CreateDefaultStatisticsService();
	}
	if (!m_uploadService) {
		m_uploadService = rg::runtime::CreateDefaultUploadService();
	}
	if (!m_readbackService) {
		m_readbackService = rg::runtime::CreateDefaultReadbackService();
	}
	if (!m_descriptorService) {
		m_descriptorService = rg::runtime::CreateDefaultDescriptorService();
	}
	if (!m_renderGraphSettingsService) {
		m_renderGraphSettingsService = rg::runtime::CreateDefaultRenderGraphSettingsService();
	}
}

RenderGraph::~RenderGraph() {
	if (m_pCommandRecordingManager) {
		m_pCommandRecordingManager->ShutdownThreadLocal(); // Clears thread-local storage
	}
	ShutdownOwnedState();
	DeletionManager::GetInstance().Cleanup();
	DeviceManager::GetInstance().Cleanup();
}

void RenderGraph::ShutdownOwnedState() {
	batches.clear();
	initialTransitions.clear();
	trackers.clear();
	compileTrackers.clear();
	m_masterPassList.clear();
	m_framePasses.clear();
	m_assignedQueueSlotsByFramePass.clear();
	m_activeQueueSlotsThisFrame.clear();
	renderPassesByName.clear();
	computePassesByName.clear();
	resourcesByID.clear();
	resourcesByName.clear();
	m_transientFrameResourcesByID.clear();
	m_transientFrameResourcesByName.clear();
	resourceBackingGenerationByID.clear();
	resourceIdleFrameCounts.clear();
	compiledResourceGenerationByID.clear();
	aliasMaterializeOptionsByID.clear();
	aliasPlacementSignatureByID.clear();
	aliasPlacementRangesByID.clear();
	aliasPlacementPoolByID.clear();
	aliasActivationPending.clear();
	persistentAliasPools.clear();
	autoAliasPoolByID.clear();
	m_lastProducerByResourceAcrossFrames.clear();
	m_lastAliasPlacementProducersByPoolAcrossFrames.clear();
	for (auto& producerMap : m_compiledLastProducerBatchByResourceByQueue) {
		producerMap.clear();
	}
	for (auto& row : m_hasPendingFrameStartQueueWait) {
		std::fill(row.begin(), row.end(), false);
	}
	for (auto& row : m_pendingFrameStartQueueWaitFenceValue) {
		std::fill(row.begin(), row.end(), UINT64(0));
	}

	m_passBuilderOrder.clear();
	m_passNamesSeenThisReset.clear();
	m_passBuildersByName.clear();
	m_extensions.clear();
    m_extensionRegistrationIds.clear();
	_providerMap.clear();
	_providers.clear();
	_resolverMap.clear();
	_registry = ResourceRegistry();

	m_pCommandRecordingManager.reset();
	m_queueRegistry.Clear();

	initialTransitionCommandAllocator.Reset();
	m_initialTransitionFence.Reset();
	m_frameStartSyncFence.Reset();
	m_readbackFence.Reset();

	m_statisticsService.reset();
	m_uploadService.reset();
	m_readbackService.reset();
	m_descriptorService.reset();
	m_renderGraphSettingsService.reset();
}

SymbolicTracker& RenderGraph::GetOrCreateCompileTracker(Resource* resource, uint64_t resourceID) {
	auto it = compileTrackers.find(resourceID);
	if (it != compileTrackers.end()) {
		return it->second;
	}

	SymbolicTracker seed{};
	if (resource) {
		bool hasLiveBacking = true;
		if (auto* texture = dynamic_cast<PixelBuffer*>(resource)) {
			hasLiveBacking = texture->IsMaterialized();
		}
		else if (auto* buffer = dynamic_cast<Buffer*>(resource)) {
			hasLiveBacking = buffer->IsMaterialized();
		}

		if (hasLiveBacking) {
			seed = *resource->GetStateTracker();
		}
	}

	auto [insertedIt, _] = compileTrackers.emplace(resourceID, std::move(seed));
	return insertedIt->second;
}

void RenderGraph::CaptureCompileTrackersForExecution(const std::unordered_set<uint64_t>& resourceIDs) {
	trackers.clear();
	trackers.reserve(resourceIDs.size());

	auto captureTracker = [&](uint64_t resourceID, Resource* resource) {
		if (!resource) {
			return;
		}

		if (!compileTrackers.contains(resourceID)) {
			return;
		}

		bool hasLiveBacking = true;
		if (auto* texture = dynamic_cast<PixelBuffer*>(resource)) {
			hasLiveBacking = texture->IsMaterialized();
		}
		else if (auto* buffer = dynamic_cast<Buffer*>(resource)) {
			hasLiveBacking = buffer->IsMaterialized();
		}

		if (!hasLiveBacking) {
			return;
		}

		if (auto* tracker = resource->GetStateTracker()) {
			trackers[resourceID] = tracker;
		}
	};

	for (uint64_t resourceID : resourceIDs) {
		auto itResource = resourcesByID.find(resourceID);
		if (itResource != resourcesByID.end() && itResource->second) {
			captureTracker(resourceID, itResource->second.get());
			continue;
		}

		auto itTransient = m_transientFrameResourcesByID.find(resourceID);
		if (itTransient != m_transientFrameResourcesByID.end() && itTransient->second) {
			captureTracker(resourceID, itTransient->second.get());
		}
	}
}

void RenderGraph::PublishCompiledTrackerStates() {
	for (const auto& [resourceID, liveTracker] : trackers) {
		if (!liveTracker) {
			continue;
		}

		auto itCompileTracker = compileTrackers.find(resourceID);
		if (itCompileTracker == compileTrackers.end()) {
			continue;
		}

		liveTracker->CopyFrom(itCompileTracker->second);
	}
}

void RenderGraph::MaterializeReferencedResources(
	const std::vector<ResourceRequirement>& resourceRequirements,
	const std::vector<std::pair<ResourceHandleAndRange, ResourceState>>& internalTransitions)
{
	auto materializeIfNeeded = [&](const ResourceRegistry::RegistryHandle& handle) {
		if (handle.IsEphemeral()) {
			return;
		}

		auto resource = _registry.Resolve(handle);
		if (!resource) {
			return;
		}

		auto texture = dynamic_cast<PixelBuffer*>(resource);
		if (texture) {
			if (texture->IsMaterialized()) {
				return;
			}

			if (texture->GetDescription().allowAlias) {
				// Alias placement is frame-dependent and produced later in CompileFrame.
				// Defer materialization until RenderGraph::MaterializeUnmaterializedResources,
				// after BuildAliasPlanAfterDag has produced placement options.
				return;
			}

			texture->Materialize();
			resourceBackingGenerationByID[handle.GetGlobalResourceID()] = texture->GetBackingGeneration();
			return;
		}

		auto buffer = dynamic_cast<Buffer*>(resource);
		if (buffer) {
			if (buffer->IsMaterialized()) {
				return;
			}

			if (buffer->IsAliasingAllowed()) {
				return;
			}

			buffer->Materialize();
			resourceBackingGenerationByID[handle.GetGlobalResourceID()] = buffer->GetBackingGeneration();
		}
	};

	for (auto const& req : resourceRequirements) {
		materializeIfNeeded(req.resourceHandleAndRange.resource);
	}

	for (auto const& transition : internalTransitions) {
		materializeIfNeeded(transition.first.resource);
	}
}

void RenderGraph::CollectFrameResourceIDs(std::unordered_set<uint64_t>& used) const {
	used.clear();
	used.reserve(m_framePasses.size() * 4);

	for (auto const& pr : m_framePasses) {
		std::visit([&](auto const& passAndResources) {
			using T = std::decay_t<decltype(passAndResources)>;
			if constexpr (!std::is_same_v<T, std::monostate>) {
				for (auto const& req : passAndResources.resources.frameResourceRequirements) {
					used.insert(req.resourceHandleAndRange.resource.GetGlobalResourceID());
				}
				for (auto const& t : passAndResources.resources.internalTransitions) {
					used.insert(t.first.resource.GetGlobalResourceID());
				}
			}
		}, pr.pass);
	}
}

void RenderGraph::ApplyIdleDematerializationPolicy(const std::unordered_set<uint64_t>& usedResourceIDs) {
	for (auto& [id, resource] : resourcesByID) {
		if (!resource) {
			continue;
		}

		auto texture = std::dynamic_pointer_cast<PixelBuffer>(resource);
		if (!texture || !texture->IsIdleDematerializationEnabled()) {
			continue;
		}

		if (usedResourceIDs.find(id) != usedResourceIDs.end()) {
			resourceIdleFrameCounts[id] = 0;
			continue;
		}

		auto& idleFrames = resourceIdleFrameCounts[id];
		idleFrames++;

		if (texture->IsMaterialized() && idleFrames >= texture->GetIdleDematerializationThreshold()) {
			texture->Dematerialize();
			resourceBackingGenerationByID[id] = texture->GetBackingGeneration();
		}
	}
}

void RenderGraph::SnapshotCompiledResourceGenerations(const std::unordered_set<uint64_t>& usedResourceIDs) {
	compiledResourceGenerationByID.clear();
	compiledResourceGenerationByID.reserve(usedResourceIDs.size());

	for (uint64_t id : usedResourceIDs) {
		auto it = resourcesByID.find(id);
		if (it == resourcesByID.end() || !it->second) {
			continue;
		}

		auto texture = std::dynamic_pointer_cast<PixelBuffer>(it->second);
		if (texture) {
			compiledResourceGenerationByID[id] = texture->GetBackingGeneration();
			continue;
		}

		auto buffer = std::dynamic_pointer_cast<Buffer>(it->second);
		if (buffer) {
			compiledResourceGenerationByID[id] = buffer->GetBackingGeneration();
		}
	}
}

void RenderGraph::ValidateCompiledResourceGenerations() const {
	for (auto const& [id, compiledGeneration] : compiledResourceGenerationByID) {
		auto it = resourcesByID.find(id);
		if (it == resourcesByID.end() || !it->second) {
			continue;
		}

		auto texture = std::dynamic_pointer_cast<PixelBuffer>(it->second);
		if (texture) {
			const uint64_t currentGeneration = texture->GetBackingGeneration();
			if (currentGeneration != compiledGeneration) {
				throw std::runtime_error("Resource backing generation changed after compile and before execute. Resource ID: " + std::to_string(id));
			}
			continue;
		}

		auto buffer = std::dynamic_pointer_cast<Buffer>(it->second);
		if (buffer) {
			const uint64_t currentGeneration = buffer->GetBackingGeneration();
			if (currentGeneration != compiledGeneration) {
				throw std::runtime_error("Resource backing generation changed after compile and before execute. Resource ID: " + std::to_string(id));
			}
		}
	}
}


void RenderGraph::RegisterExtension(std::unique_ptr<IRenderGraphExtension> ext, std::optional<std::string_view> id) {
	if (!ext) return;

	const auto& incomingType = typeid(*ext);
    const std::string registrationId = id.has_value()
        ? std::string(*id)
        : std::string(incomingType.name());

	for (const auto& existingId : m_extensionRegistrationIds) {
		if (existingId == registrationId) {
			spdlog::error("Duplicate RenderGraph extension registration: {}", registrationId);
			throw std::runtime_error("Duplicate RenderGraph extension registration");
		}
	}

	// Let the extension see the current registry immediately.
	ext->OnRegistryReset(&_registry);
	m_extensions.push_back(std::move(ext));
    try {
        m_extensionRegistrationIds.push_back(registrationId);
    }
    catch (...) {
        m_extensions.pop_back();
        throw;
    }
}

void RenderGraph::ResetForRebuild()
{
	// Clear any existing compile state
	m_masterPassList.clear();
	m_framePasses.clear();
	m_assignedQueueSlotsByFramePass.clear();
	m_activeQueueSlotsThisFrame.clear();
	batches.clear();
	m_executionSchedule.batches.clear();
	trackers.clear();

	// Full rebuilds must drop cached pass instances before clearing resources.
	// Builders reuse pass objects across frames, and many passes capture
	// resource-owning shared_ptrs through constructor arguments.
	for (auto& [name, builder] : m_passBuildersByName) {
		(void)name;
		if (builder) {
			builder->Reset();
		}
	}

	// Clear resources
	resourcesByID.clear();
	resourcesByName.clear();
	m_transientFrameResourcesByID.clear();
	m_transientFrameResourcesByName.clear();
	resourceBackingGenerationByID.clear();
	resourceIdleFrameCounts.clear();
	compiledResourceGenerationByID.clear();
	m_aliasingSubsystem.ResetPersistentState(*this);
	compileTrackers.clear();
	m_lastProducerByResourceAcrossFrames.clear();
	m_lastAliasPlacementProducersByPoolAcrossFrames.clear();
	for (auto& producerMap : m_compiledLastProducerBatchByResourceByQueue) {
		producerMap.clear();
	}
	for (auto& row : m_hasPendingFrameStartQueueWait) {
		std::fill(row.begin(), row.end(), false);
	}
	for (auto& row : m_pendingFrameStartQueueWaitFenceValue) {
		std::fill(row.begin(), row.end(), UINT64(0));
	}
	m_compiledLastProducerBatchByResourceByQueue.clear();
	m_hasPendingFrameStartQueueWait.clear();
	m_pendingFrameStartQueueWaitFenceValue.clear();
	m_queueRegistry.Clear();
	renderPassesByName.clear();
	computePassesByName.clear();

	// Clear providers
	_providerMap.clear();
	_providers.clear();
	_resolverMap.clear();
	_registry = ResourceRegistry();

	// Notify extensions that the registry was replaced
	for (auto& ext : m_extensions) {
		if (ext) ext->OnRegistryReset(&_registry);
	}
	// clear pass ordering
	m_passBuilderOrder.clear();
	m_passNamesSeenThisReset.clear();

}

void RenderGraph::ResetForFrame() {
	batches.clear();
	compiledResourceGenerationByID.clear();
	m_transientFrameResourcesByID.clear();
	m_transientFrameResourcesByName.clear();
	m_aliasingSubsystem.ResetPerFrameState(*this);
	compileTrackers.clear();
	m_assignedQueueSlotsByFramePass.clear();
	m_activeQueueSlotsThisFrame.clear();
	for (auto& producerMap : m_compiledLastProducerBatchByResourceByQueue) {
		producerMap.clear();
	}
	for (auto& row : m_hasPendingFrameStartQueueWait) {
		std::fill(row.begin(), row.end(), false);
	}
	for (auto& row : m_pendingFrameStartQueueWaitFenceValue) {
		std::fill(row.begin(), row.end(), UINT64(0));
	}
	// reset pass builders
	for (auto& [name, builder] : m_passBuildersByName) {
		builder->Reset();
	}
}

void RenderGraph::CompileStructural() {
	// Register resource providers from pass builders

	std::vector<unsigned int> empty;
	// Go backwards to build skip list
	for (int i = static_cast<int>(m_passBuilderOrder.size()) - 1; i >= 0; i--) {
		auto ptr = m_passBuilderOrder[i];
		auto prov = ptr->ResourceProvider();
		if (!prov) {
			empty.push_back(i); // This pass was not built
			continue;
		}
		RegisterProvider(prov);
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

	batches.clear();

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
		AnyPassAndResources any;
		any.type = d.type;
		any.name = d.name;

		if (d.type == PassType::Render) {
			auto rp = std::get<std::shared_ptr<RenderPass>>(d.pass);
			RenderPassAndResources par;
			par.pass = std::move(rp);
			par.name = d.name;
			{
				RenderPassBuilder b(this, d.name);
				b.pass = par.pass;
				b.built_ = true;
				b.params = {};
				b.params.isGeometryPass = d.isGeometryPass;
				b._declaredIds.clear();
				par.pass->DeclareResourceUsages(&b);
				par.resources.staticResourceRequirements = b.GatherResourceRequirements();
				par.resources.frameResourceRequirements = par.resources.staticResourceRequirements;
				par.resources.internalTransitions = b.params.internalTransitions;
				par.resources.identifierSet = b.DeclaredResourceIds();
				par.resources.isGeometryPass = b.params.isGeometryPass;
				par.resources.queueSelection = d.renderQueueSelection.value_or(RenderQueueSelection::Graphics);
				par.resources.queueSlotOverride = d.queueSlotOverride;
				MaterializeReferencedResources(par.resources.staticResourceRequirements, par.resources.internalTransitions);
			}
 			any.pass = std::move(par);
 		}
 		else if (d.type == PassType::Compute) {
 			auto cp = std::get<std::shared_ptr<ComputePass>>(d.pass);
 			ComputePassAndResources par;
 			par.pass = std::move(cp);
 			par.name = d.name;
			{
				ComputePassBuilder b(this, d.name);
				b.pass = par.pass;
				b.built_ = true;
				b.params = {};
				b._declaredIds.clear();
				par.pass->DeclareResourceUsages(&b);
				par.resources.staticResourceRequirements = b.GatherResourceRequirements();
				par.resources.frameResourceRequirements = par.resources.staticResourceRequirements;
				par.resources.internalTransitions = b.params.internalTransitions;
				par.resources.identifierSet = b.DeclaredResourceIds();
				par.resources.queueSelection = d.computeQueueSelection.value_or(ComputeQueueSelection::Compute);
				par.resources.queueSlotOverride = d.queueSlotOverride;
				MaterializeReferencedResources(par.resources.staticResourceRequirements, par.resources.internalTransitions);
			}
 			any.pass = std::move(par);
 		}
		else if (d.type == PassType::Copy) {
			auto cp = std::get<std::shared_ptr<CopyPass>>(d.pass);
			CopyPassAndResources par;
			par.pass = std::move(cp);
			par.name = d.name;
			{
				CopyPassBuilder b(this, d.name);
				b.pass = par.pass;
				b.built_ = true;
				b.params = {};
				b._declaredIds.clear();
				par.pass->DeclareResourceUsages(&b);
				par.resources.staticResourceRequirements = b.GatherResourceRequirements();
				par.resources.frameResourceRequirements = par.resources.staticResourceRequirements;
				par.resources.internalTransitions = b.params.internalTransitions;
				par.resources.identifierSet = b.DeclaredResourceIds();
				par.resources.queueSelection = d.copyQueueSelection.value_or(CopyQueueSelection::Copy);
				par.resources.queueSlotOverride = d.queueSlotOverride;
				MaterializeReferencedResources(par.resources.staticResourceRequirements, par.resources.internalTransitions);
			}
			any.pass = std::move(par);
		}
 		return any;
 		};

	auto registerName = [&](ExternalPassDesc const& d, AnyPassAndResources& any) {
		if (!d.registerName) return;
		if (d.type == PassType::Render) {
			auto& rp = std::get<RenderPassAndResources>(any.pass);
			if (!d.name.empty()) renderPassesByName[d.name] = rp.pass;
		}
		else if (d.type == PassType::Compute) {
			auto& cp = std::get<ComputePassAndResources>(any.pass);
			if (!d.name.empty()) computePassesByName[d.name] = cp.pass;
		}
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

	// Gather extension passes into ExtItem list
	std::vector<ExtItem> extItems;
	extItems.reserve(64);

	size_t globalOrder = 0;

	// helper: stable synthetic key for unnamed passes
	auto makeSyntheticKey = [](size_t n) -> std::string {
		return "__rg_ext_" + std::to_string(n);
		};

	for (int ei = 0; ei < (int)m_extensions.size(); ++ei) {
		auto& ext = m_extensions[ei];
		if (!ext) continue;

		std::vector<ExternalPassDesc> local;
		local.reserve(16);
		ext->GatherStructuralPasses(*this, local);

		std::optional<std::string> prevKey; // for extension-local chaining
		int localOrder = 0;

		for (auto& d : local) {
			if (d.type == PassType::Unknown) continue;
			if (std::holds_alternative<std::monostate>(d.pass)) continue;

			ExtItem it;
			it.pr = makeAny(d);
			registerName(d, it.pr);

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
				spdlog::warn("External pass '{}' requested After('{}') but anchor not found; ignoring.", e.key, a);
				continue;
			}
			addEdge(*idxOpt, passNode);
			anyConstraint = true;
		}

		// before[] : pass -> anchor
		for (auto const& b : e.where.before) {
			auto idxOpt = resolveAnchor(b);
			if (!idxOpt) {
				spdlog::warn("External pass '{}' requested Before('{}') but anchor not found; ignoring.", e.key, b);
				continue;
			}
			addEdge(passNode, *idxOpt);
			anyConstraint = true;
		}
	}

	// Extension chaining edges: prev -> next (if keepExtensionOrder on the *next* pass)
	for (auto& [ei, v] : extOrder) {
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

	// Topological sort (stable by priority then order)
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

	// Emit final m_masterPassList in topo order (skip sentinels)
	m_masterPassList.clear();
	m_masterPassList.reserve(baseIdx.size() + extIdx.size());

	for (size_t u : topo) {
		if (!nodes[u].hasPass) {
			continue;
		}
		m_masterPassList.push_back(std::move(nodes[u].pass));
	}
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
	std::vector<ResourceRequirement> const& retained,
	std::vector<ResourceRequirement> const& immediate)
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


void RenderGraph::RefreshRetainedDeclarationsForFrame(RenderPassAndResources& p, uint8_t frameIndex)
{
	RenderPassBuilder b(this, p.name);

	// Make it look like a normal builder enough for any pass code that queries ResourceProvider()
	b.pass = p.pass;
	b.built_ = true;

	// Clear any previous declarations
	b.params = {};
	b._declaredIds.clear();

	// Let the pass declare based on current per-frame state (queued mip jobs etc.)
	p.pass->DeclareResourceUsages(&b);

	// Update the frame view used by scheduling
	p.resources.staticResourceRequirements = b.GatherResourceRequirements();

	// Internal transitions also affect scheduling
	p.resources.internalTransitions = b.params.internalTransitions;

	p.resources.identifierSet = b.DeclaredResourceIds();
	p.resources.autoDescriptorShaderResources = b.params.autoDescriptorShaderResources;
	p.resources.autoDescriptorUnorderedAccessViews = b.params.autoDescriptorUnorderedAccessViews;
	MaterializeReferencedResources(p.resources.staticResourceRequirements, p.resources.internalTransitions);

	// Transfer resolver snapshots for auto-invalidation tracking
	p.resolverSnapshots = b.TakeResolverSnapshots();

	// Ensure the pass's view matches the refreshed identifier set
	p.pass->SetResourceRegistryView(
		std::make_unique<ResourceRegistryView>(_registry, p.resources.identifierSet),
		p.resources.autoDescriptorShaderResources,
		p.resources.autoDescriptorUnorderedAccessViews
	);
	p.pass->Setup();
}

void RenderGraph::RefreshRetainedDeclarationsForFrame(ComputePassAndResources& p, uint8_t frameIndex)
{
	ComputePassBuilder b(this, p.name);
	b.pass = p.pass;
	b.built_ = true;

	b.params = {};
	b._declaredIds.clear();

	p.pass->DeclareResourceUsages(&b);

	p.resources.staticResourceRequirements = b.GatherResourceRequirements();
	p.resources.internalTransitions = b.params.internalTransitions;
	p.resources.identifierSet = b.DeclaredResourceIds();
	p.resources.autoDescriptorShaderResources = b.params.autoDescriptorShaderResources;
	p.resources.autoDescriptorConstantBuffers = b.params.autoDescriptorConstantBuffers;
	p.resources.autoDescriptorUnorderedAccessViews = b.params.autoDescriptorUnorderedAccessViews;
	MaterializeReferencedResources(p.resources.staticResourceRequirements, p.resources.internalTransitions);

	// Transfer resolver snapshots for auto-invalidation tracking
	p.resolverSnapshots = b.TakeResolverSnapshots();

	p.pass->SetResourceRegistryView(
		std::make_unique<ResourceRegistryView>(_registry, p.resources.identifierSet),
		p.resources.autoDescriptorShaderResources,
		p.resources.autoDescriptorConstantBuffers,
		p.resources.autoDescriptorUnorderedAccessViews
	);

	p.pass->Setup();
}

void RenderGraph::RefreshRetainedDeclarationsForFrame(CopyPassAndResources& p, uint8_t frameIndex)
{
	CopyPassBuilder b(this, p.name);
	b.pass = p.pass;
	b.built_ = true;

	b.params = {};
	b._declaredIds.clear();

	p.pass->DeclareResourceUsages(&b);

	p.resources.staticResourceRequirements = b.GatherResourceRequirements();
	p.resources.internalTransitions = b.params.internalTransitions;
	p.resources.identifierSet = b.DeclaredResourceIds();
	MaterializeReferencedResources(p.resources.staticResourceRequirements, p.resources.internalTransitions);

	// Transfer resolver snapshots for auto-invalidation tracking
	p.resolverSnapshots = b.TakeResolverSnapshots();

	p.pass->SetResourceRegistryView(
		std::make_unique<ResourceRegistryView>(_registry, p.resources.identifierSet)
	);

	p.pass->Setup();
}

void RenderGraph::CompileFrame(rhi::Device device, uint8_t frameIndex, const IHostExecutionData* hostData) {
	compileTrackers.clear();
	m_aliasingSubsystem.ResetPerFrameState(*this);

	auto needsRefresh = [&](auto& p) -> bool {
		// Check if any stored resolver's content version has changed
		for (const auto& snap : p.resolverSnapshots) {
			uint64_t cv = snap.resolver->GetContentVersion();
			if (cv != 0 && cv != snap.version) {
				return true;
			}
		}

		auto* iFace = dynamic_cast<IDynamicDeclaredResources*>(p.pass.get());
		if (!iFace) {
			// if pass doesn't opt-in, assume no change
			return false;
		}

		return iFace->DeclaredResourcesChanged();
		};

	// First, refresh all retained declarations for this frame
	for (auto& pr : m_masterPassList) {
		if (pr.type == PassType::Compute) {
			auto& p = std::get<ComputePassAndResources>(pr.pass);
			p.immediateBytecode.clear();
			p.resources.frameResourceRequirements = {};// p.resources.staticResourceRequirements;
			if (needsRefresh(p)) {
				RefreshRetainedDeclarationsForFrame(p, frameIndex);
			}
		}
		else if (pr.type == PassType::Render) {
			auto& p = std::get<RenderPassAndResources>(pr.pass);
			p.immediateBytecode.clear();
			p.resources.frameResourceRequirements = {};// p.resources.staticResourceRequirements;
			if (needsRefresh(p)) {
				RefreshRetainedDeclarationsForFrame(p, frameIndex);
			}
		}
		else if (pr.type == PassType::Copy) {
			auto& p = std::get<CopyPassAndResources>(pr.pass);
			p.immediateBytecode.clear();
			p.resources.frameResourceRequirements = {};
			if (needsRefresh(p)) {
				RefreshRetainedDeclarationsForFrame(p, frameIndex);
			}
		}
	}

	batches.clear();
	batches.emplace_back(m_queueRegistry.SlotCount()); // Dummy batch 0 for pre-first-pass transitions
	m_framePasses.clear(); // Combined retained + immediate-mode passes for this frame

	// Record immediate-mode commands + access for each pass and fold into per-frame requirements
	for (auto& pr : m_masterPassList) {

		if (pr.type == PassType::Compute) {
			auto& p = std::get<ComputePassAndResources>(pr.pass);

			// reset per-frame
			p.immediateBytecode.clear();
			p.resources.frameResourceRequirements = p.resources.staticResourceRequirements;

			ImmediateExecutionContext c{ device,
				{rg::imm::ImmediatePassKind::Compute,
				m_immediateDispatch,
				&ResolveByIdThunk,
				&ResolveByPtrThunk,
				this},
				frameIndex,
				hostData
			};

			// Record immediate-mode commands
			p.pass->ExecuteImmediate(c);

			auto immediateFrameData = c.list.Finalize();
			// If there is a conflict between retained and immediate requirements, split the pass
			bool conflict = RequirementsConflict(
				p.resources.frameResourceRequirements,   // baseline retained for this frame
				immediateFrameData.requirements);
			if (conflict) {
				// Create new PassAndResources for the immediate requirements
				ComputePassAndResources immediatePassAndResources;
				immediatePassAndResources.pass = p.pass;
				immediatePassAndResources.resources.staticResourceRequirements = immediateFrameData.requirements;
				immediatePassAndResources.resources.frameResourceRequirements = immediateFrameData.requirements;
				immediatePassAndResources.resources.queueSelection = p.resources.queueSelection;
				immediatePassAndResources.immediateBytecode = std::move(immediateFrameData.bytecode);
				immediatePassAndResources.immediateKeepAlive = std::move(p.immediateKeepAlive);
				immediatePassAndResources.run = PassRunMask::Immediate;
				AnyPassAndResources immediateAnyPassAndResources;
				immediateAnyPassAndResources.type = PassType::Compute;
				immediateAnyPassAndResources.pass = immediatePassAndResources;
				m_framePasses.push_back(immediateAnyPassAndResources);
				p.run = PassRunMask::Retained;
				m_framePasses.push_back(pr); // Retained pass
			}
			else {
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				p.resources.frameResourceRequirements.insert(
					p.resources.frameResourceRequirements.end(),
					immediateFrameData.requirements.begin(),
					immediateFrameData.requirements.end());
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
				m_framePasses.push_back(pr);
			}
		}
		else if (pr.type == PassType::Render) {
			auto& p = std::get<RenderPassAndResources>(pr.pass);

			p.immediateBytecode.clear();
			p.resources.frameResourceRequirements = p.resources.staticResourceRequirements;

			ImmediateExecutionContext c{ device,
				{rg::imm::ImmediatePassKind::Render,
				m_immediateDispatch,
				&ResolveByIdThunk,
				&ResolveByPtrThunk,
				this},
				frameIndex,
				hostData
			};
			p.pass->ExecuteImmediate(c);
			auto immediateFrameData = c.list.Finalize();

			bool conflict = RequirementsConflict(
				p.resources.frameResourceRequirements,   // baseline retained for this frame
				immediateFrameData.requirements);

			if (conflict) {
				// Create new PassAndResources for the immediate requirements
				RenderPassAndResources immediatePassAndResources;
				immediatePassAndResources.pass = p.pass;
				immediatePassAndResources.resources.staticResourceRequirements = immediateFrameData.requirements;
				immediatePassAndResources.resources.frameResourceRequirements = immediateFrameData.requirements;
				immediatePassAndResources.resources.queueSelection = p.resources.queueSelection;
				immediatePassAndResources.immediateBytecode = std::move(immediateFrameData.bytecode);
				immediatePassAndResources.immediateKeepAlive = std::move(p.immediateKeepAlive);
				immediatePassAndResources.run = PassRunMask::Immediate;
				AnyPassAndResources immediateAnyPassAndResources;
				immediateAnyPassAndResources.type = PassType::Render;
				immediateAnyPassAndResources.pass = immediatePassAndResources;
				m_framePasses.push_back(immediateAnyPassAndResources);
				p.run = PassRunMask::Retained;
				m_framePasses.push_back(pr); // Retained pass
			}
			else {
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				p.resources.frameResourceRequirements.insert(
					p.resources.frameResourceRequirements.end(),
					immediateFrameData.requirements.begin(),
					immediateFrameData.requirements.end());
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
				m_framePasses.push_back(pr);
			}
		}
		else if (pr.type == PassType::Copy) {
			auto& p = std::get<CopyPassAndResources>(pr.pass);

			p.immediateBytecode.clear();
			p.resources.frameResourceRequirements = p.resources.staticResourceRequirements;

			ImmediateExecutionContext c{ device,
				{rg::imm::ImmediatePassKind::Copy,
				m_immediateDispatch,
				&ResolveByIdThunk,
				&ResolveByPtrThunk,
				this},
				frameIndex,
				hostData
			};

			p.pass->ExecuteImmediate(c);
			auto immediateFrameData = c.list.Finalize();

			bool conflict = RequirementsConflict(
				p.resources.frameResourceRequirements,
				immediateFrameData.requirements);

			if (conflict) {
				CopyPassAndResources immediatePassAndResources;
				immediatePassAndResources.pass = p.pass;
				immediatePassAndResources.resources.staticResourceRequirements = immediateFrameData.requirements;
				immediatePassAndResources.resources.frameResourceRequirements = immediateFrameData.requirements;
				immediatePassAndResources.resources.queueSelection = p.resources.queueSelection;
				immediatePassAndResources.immediateBytecode = std::move(immediateFrameData.bytecode);
				immediatePassAndResources.immediateKeepAlive = std::move(p.immediateKeepAlive);
				immediatePassAndResources.run = PassRunMask::Immediate;
				AnyPassAndResources immediateAnyPassAndResources;
				immediateAnyPassAndResources.type = PassType::Copy;
				immediateAnyPassAndResources.pass = immediatePassAndResources;
				m_framePasses.push_back(immediateAnyPassAndResources);
				p.run = PassRunMask::Retained;
				m_framePasses.push_back(pr);
			}
			else {
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				p.resources.frameResourceRequirements.insert(
					p.resources.frameResourceRequirements.end(),
					immediateFrameData.requirements.begin(),
					immediateFrameData.requirements.end());
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
				m_framePasses.push_back(pr);
			}
		}
	}

	// Per-frame extension passes (ephemeral)
	// These are injected into the per-frame pass list (not m_masterPassList) so they do not accumulate.
	std::vector<ExternalPassDesc> frameExt;
	frameExt.reserve(16);
	for (auto& ext : m_extensions) {
		if (!ext) continue;
		ext->GatherFramePasses(*this, frameExt);
	}

	// explicit After(anchor) edges (anchorName -> injectedName)
	std::vector<std::pair<std::string, std::string>> explicitAfterByName;
	explicitAfterByName.reserve(frameExt.size());

	if (!frameExt.empty()) {
		auto makeAny = [&](ExternalPassDesc const& d) -> AnyPassAndResources {
			AnyPassAndResources any;
			any.type = d.type;
			any.name = d.name;

			if (d.type == PassType::Render) {
				auto rp = std::get<std::shared_ptr<RenderPass>>(d.pass);
				RenderPassAndResources par;
				par.pass = std::move(rp);
				par.name = d.name;
				{
					RenderPassBuilder b(this, d.name);
					b.pass = par.pass;
					b.built_ = true;
					b.params = {};
					b._declaredIds.clear();
					par.pass->DeclareResourceUsages(&b);
					par.resources.staticResourceRequirements = b.GatherResourceRequirements();
					par.resources.frameResourceRequirements = par.resources.staticResourceRequirements;
					par.resources.internalTransitions = b.params.internalTransitions;
					par.resources.identifierSet = b.DeclaredResourceIds();
					par.resources.autoDescriptorShaderResources = b.params.autoDescriptorShaderResources;
					par.resources.autoDescriptorUnorderedAccessViews = b.params.autoDescriptorUnorderedAccessViews;
					par.resources.isGeometryPass = b.params.isGeometryPass;
					par.resources.queueSelection = d.renderQueueSelection.value_or(RenderQueueSelection::Graphics);
					par.resources.queueSlotOverride = d.queueSlotOverride;
				}

				par.pass->SetResourceRegistryView(
						std::make_unique<ResourceRegistryView>(_registry, par.resources.identifierSet),
						par.resources.autoDescriptorShaderResources,
						par.resources.autoDescriptorUnorderedAccessViews
				);
				par.pass->Setup();
				any.pass = std::move(par);
			}
			else if (d.type == PassType::Compute) {
				auto cp = std::get<std::shared_ptr<ComputePass>>(d.pass);
				ComputePassAndResources par;
				par.pass = std::move(cp);
				par.name = d.name;
				{
					ComputePassBuilder b(this, d.name);
					b.pass = par.pass;
					b.built_ = true;
					b.params = {};
					b._declaredIds.clear();
					par.pass->DeclareResourceUsages(&b);
					par.resources.staticResourceRequirements = b.GatherResourceRequirements();
					par.resources.frameResourceRequirements = par.resources.staticResourceRequirements;
					par.resources.internalTransitions = b.params.internalTransitions;
					par.resources.identifierSet = b.DeclaredResourceIds();
					par.resources.autoDescriptorShaderResources = b.params.autoDescriptorShaderResources;
					par.resources.autoDescriptorConstantBuffers = b.params.autoDescriptorConstantBuffers;
					par.resources.autoDescriptorUnorderedAccessViews = b.params.autoDescriptorUnorderedAccessViews;
					par.resources.queueSelection = d.computeQueueSelection.value_or(ComputeQueueSelection::Compute);
					par.resources.queueSlotOverride = d.queueSlotOverride;
				}

				par.pass->SetResourceRegistryView(
						std::make_unique<ResourceRegistryView>(_registry, par.resources.identifierSet),
						par.resources.autoDescriptorShaderResources,
						par.resources.autoDescriptorConstantBuffers,
						par.resources.autoDescriptorUnorderedAccessViews
				);
				par.pass->Setup();
				any.pass = std::move(par);
			}
			else if (d.type == PassType::Copy) {
				auto cp = std::get<std::shared_ptr<CopyPass>>(d.pass);
				CopyPassAndResources par;
				par.pass = std::move(cp);
				par.name = d.name;
				{
					CopyPassBuilder b(this, d.name);
					b.pass = par.pass;
					b.built_ = true;
					b.params = {};
					b._declaredIds.clear();
					par.pass->DeclareResourceUsages(&b);
					par.resources.staticResourceRequirements = b.GatherResourceRequirements();
					par.resources.frameResourceRequirements = par.resources.staticResourceRequirements;
					par.resources.internalTransitions = b.params.internalTransitions;
					par.resources.identifierSet = b.DeclaredResourceIds();
					par.resources.queueSelection = d.copyQueueSelection.value_or(CopyQueueSelection::Copy);
					par.resources.queueSlotOverride = d.queueSlotOverride;
				}

				par.pass->SetResourceRegistryView(
					std::make_unique<ResourceRegistryView>(_registry, par.resources.identifierSet)
				);
				par.pass->Setup();
				any.pass = std::move(par);
			}

			return any;
		};

		auto recordImmediate = [&](AnyPassAndResources& pr) {
			if (pr.type == PassType::Compute) {
				auto& p = std::get<ComputePassAndResources>(pr.pass);
				p.immediateBytecode.clear();
				p.resources.frameResourceRequirements = p.resources.staticResourceRequirements;

				ImmediateExecutionContext c{ device,
					{rg::imm::ImmediatePassKind::Compute,
					m_immediateDispatch,
					&ResolveByIdThunk,
					&ResolveByPtrThunk,
					this},
					frameIndex,
					hostData
				};

				p.pass->ExecuteImmediate(c);
				auto immediateFrameData = c.list.Finalize();
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				p.resources.frameResourceRequirements.insert(
					p.resources.frameResourceRequirements.end(),
					immediateFrameData.requirements.begin(),
					immediateFrameData.requirements.end());
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
			}
			else if (pr.type == PassType::Copy) {
				auto& p = std::get<CopyPassAndResources>(pr.pass);
				p.immediateBytecode.clear();
				p.resources.frameResourceRequirements = p.resources.staticResourceRequirements;

				ImmediateExecutionContext c{ device,
					{rg::imm::ImmediatePassKind::Copy,
					m_immediateDispatch,
					&ResolveByIdThunk,
					&ResolveByPtrThunk,
					this},
					frameIndex,
					hostData
				};

				p.pass->ExecuteImmediate(c);
				auto immediateFrameData = c.list.Finalize();
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				p.resources.frameResourceRequirements.insert(
					p.resources.frameResourceRequirements.end(),
					immediateFrameData.requirements.begin(),
					immediateFrameData.requirements.end());
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
			}
			else {
				auto& p = std::get<RenderPassAndResources>(pr.pass);
				p.immediateBytecode.clear();
				p.resources.frameResourceRequirements = p.resources.staticResourceRequirements;

				ImmediateExecutionContext c{ device,
					{rg::imm::ImmediatePassKind::Render,
					m_immediateDispatch,
					&ResolveByIdThunk,
					&ResolveByPtrThunk,
					this},
					frameIndex,
					hostData
				};

				p.pass->ExecuteImmediate(c);
				auto immediateFrameData = c.list.Finalize();
				p.immediateBytecode = std::move(immediateFrameData.bytecode);
				p.immediateKeepAlive = std::move(immediateFrameData.keepAlive);
				p.resources.frameResourceRequirements.insert(
					p.resources.frameResourceRequirements.end(),
					immediateFrameData.requirements.begin(),
					immediateFrameData.requirements.end());
				p.run = p.immediateBytecode.empty() ? PassRunMask::Retained : PassRunMask::Both;
			}
		};

		// Build name->index map for O(1) anchor lookup instead of linear scan
		std::unordered_map<std::string, size_t> framePassIndexByName;
		framePassIndexByName.reserve(m_framePasses.size() + frameExt.size());
		for (size_t i = 0; i < m_framePasses.size(); ++i) {
			if (!m_framePasses[i].name.empty()) {
				framePassIndexByName[m_framePasses[i].name] = i;
			}
		}

		auto findPassIndexByName = [&](const std::string& name) -> std::optional<size_t> {
			if (name.empty()) return std::nullopt;
			auto it = framePassIndexByName.find(name);
			if (it != framePassIndexByName.end()) return it->second;
			return std::nullopt;
		};

		std::unordered_map<std::string, uint32_t> insertedAfterCount;
		insertedAfterCount.reserve(frameExt.size());

		for (auto& d : frameExt) {
			if (d.type == PassType::Unknown) continue;
			if (std::holds_alternative<std::monostate>(d.pass)) continue;
			if (d.name.empty()) {
				spdlog::warn("Frame extension emitted a pass with empty name; skipping.");
				continue;
			}

			AnyPassAndResources any = makeAny(d);
			recordImmediate(any);

			// Default insertion: append
			size_t insertPos = m_framePasses.size();
			std::string anchorName;

			if (d.where.has_value()) {
				for (auto const& a : d.where->after) {
					auto idxOpt = findPassIndexByName(a);
					if (idxOpt.has_value()) {
						anchorName = a;
						uint32_t offset = insertedAfterCount[anchorName]++;
						insertPos = std::min(*idxOpt + 1 + (size_t)offset, m_framePasses.size());
						break;
					}
				}
			}

			if (!anchorName.empty()) {
				explicitAfterByName.push_back({ anchorName, any.name });
			}

			// Update indices in the map for entries at or after the insertion point
			for (auto& [name, idx] : framePassIndexByName) {
				if (idx >= insertPos) ++idx;
			}
			// Add the new pass to the map before inserting (insertPos is its index)
			if (!any.name.empty()) {
				framePassIndexByName[any.name] = insertPos;
			}

			m_framePasses.insert(m_framePasses.begin() + insertPos, std::move(any));
		}
	}

	// Register/refresh pass statistics indices for this frame's concrete pass list.
	// This supports transient passes and per-frame retained/immediate splits.
	if (m_statisticsService) {
		for (size_t i = 0; i < m_framePasses.size(); ++i) {
			auto& any = m_framePasses[i];
			if (any.type == PassType::Render) {
				auto& p = std::get<RenderPassAndResources>(any.pass);
				if (p.name.empty()) {
					p.name = "RenderPass#" + std::to_string(i);
				}
				any.name = p.name;
				p.statisticsIndex = static_cast<int>(m_statisticsService->RegisterPass(p.name, p.resources.isGeometryPass));
			}
			else if (any.type == PassType::Compute) {
				auto& p = std::get<ComputePassAndResources>(any.pass);
				if (p.name.empty()) {
					p.name = "ComputePass#" + std::to_string(i);
				}
				any.name = p.name;
				p.statisticsIndex = static_cast<int>(m_statisticsService->RegisterPass(p.name, false));
			}
			else if (any.type == PassType::Copy) {
				auto& p = std::get<CopyPassAndResources>(any.pass);
				if (p.name.empty()) {
					p.name = "CopyPass#" + std::to_string(i);
				}
				any.name = p.name;
				p.statisticsIndex = static_cast<int>(m_statisticsService->RegisterPass(p.name, false));
			}
		}

		m_statisticsService->SetupQueryHeap();
	}

	std::unordered_set<uint64_t> usedResourceIDs;
	CollectFrameResourceIDs(usedResourceIDs);
	ApplyIdleDematerializationPolicy(usedResourceIDs);

	// Convert explicit After(anchorName)->(passName) constraints into node-index edges.
	std::vector<std::pair<size_t, size_t>> explicitEdges;
	explicitEdges.reserve(explicitAfterByName.size());
	if (!explicitAfterByName.empty()) {
		std::unordered_map<std::string, size_t> nameToIndex;
		nameToIndex.reserve(m_framePasses.size());
		for (size_t i = 0; i < m_framePasses.size(); ++i) {
			if (!m_framePasses[i].name.empty()) {
				nameToIndex[m_framePasses[i].name] = i;
			}
		}
		for (auto const& e : explicitAfterByName) {
			auto itA = nameToIndex.find(e.first);
			auto itB = nameToIndex.find(e.second);
			if (itA == nameToIndex.end() || itB == nameToIndex.end()) {
				spdlog::warn("Explicit After edge dropped (anchor='{}', pass='{}'): name not found in frame pass list.", e.first, e.second);
				continue;
			}
			explicitEdges.push_back({ itA->second, itB->second });
		}
	}

	auto nodes = BuildNodes(*this, m_framePasses);
	if (!BuildDependencyGraph(nodes, explicitEdges)) {
		// Cycle detected
		spdlog::error("Render graph contains a dependency cycle! Render graph compilation failed.");
		throw std::runtime_error("Render graph contains a dependency cycle");
	}

	m_activeQueueSlotsThisFrame = PlanActiveQueueSlots(*this, m_framePasses, nodes);
	m_assignedQueueSlotsByFramePass.resize(m_framePasses.size());
	for (auto& node : nodes) {
		size_t defaultSlot = node.queueSlot;
		for (size_t compatibleSlot : node.compatibleQueueSlots) {
			if (compatibleSlot < m_activeQueueSlotsThisFrame.size() && m_activeQueueSlotsThisFrame[compatibleSlot]) {
				defaultSlot = compatibleSlot;
				break;
			}
		}
		node.assignedQueueSlot = defaultSlot;
		if (node.passIndex < m_assignedQueueSlotsByFramePass.size()) {
			m_assignedQueueSlotsByFramePass[node.passIndex] = defaultSlot;
		}
	}

	std::vector<rg::alias::AliasSchedulingNode> aliasNodes;
	aliasNodes.reserve(nodes.size());
	for (const auto& node : nodes) {
		aliasNodes.push_back(rg::alias::AliasSchedulingNode{
			.passIndex = node.passIndex,
			.originalOrder = node.originalOrder,
			.indegree = node.indegree,
			.criticality = node.criticality,
			.out = node.out,
		});
	}

	m_aliasingSubsystem.AutoAssignAliasingPools(*this, aliasNodes);
	m_aliasingSubsystem.BuildAliasPlanAfterDag(*this, aliasNodes);
	MaterializeUnmaterializedResources(&usedResourceIDs);
	SnapshotCompiledResourceGenerations(usedResourceIDs);

	AutoScheduleAndBuildBatches(*this, m_framePasses, nodes);
	m_aliasingSubsystem.ApplyAliasQueueSynchronization(*this);
	CaptureCompileTrackersForExecution(usedResourceIDs);

	for (auto& row : m_hasPendingFrameStartQueueWait) {
		std::fill(row.begin(), row.end(), false);
	}
	for (auto& row : m_pendingFrameStartQueueWaitFenceValue) {
		std::fill(row.begin(), row.end(), UINT64(0));
	}
	uint32_t overlapTriggeredWaitCount = 0;
	uint64_t overlapSampleCurrentResourceId = 0;
	uint64_t overlapSamplePreviousResourceId = 0;

	auto markCrossFrameWait = [&](size_t dstSlot, size_t srcSlot, uint64_t fenceValue) {
		if (dstSlot == srcSlot) {
			return;
		}
		auto& enabled = m_hasPendingFrameStartQueueWait[dstSlot][srcSlot];
		auto& maxFence = m_pendingFrameStartQueueWaitFenceValue[dstSlot][srcSlot];
		enabled = true;
		maxFence = std::max(maxFence, fenceValue);
	};

	auto accumulateCrossFrameWaitForHandle = [&](size_t passQueueSlot, const ResourceRegistry::RegistryHandle& handle) {
		if (handle.IsEphemeral()) {
			return;
		}

		const uint64_t id = handle.GetGlobalResourceID();
		for (uint64_t rid : m_aliasingSubsystem.GetSchedulingEquivalentIDs(id, aliasPlacementRangesByID)) {
			auto it = m_lastProducerByResourceAcrossFrames.find(rid);
			if (it != m_lastProducerByResourceAcrossFrames.end()) {
				markCrossFrameWait(passQueueSlot, it->second.queueSlot, it->second.fenceValue);
			}

			auto itCurPlacement = aliasPlacementRangesByID.find(rid);
			if (itCurPlacement == aliasPlacementRangesByID.end()) {
				continue;
			}

			auto itPoolState = persistentAliasPools.find(itCurPlacement->second.poolID);
			if (itPoolState == persistentAliasPools.end()) {
				continue;
			}

			auto itPrevPool = m_lastAliasPlacementProducersByPoolAcrossFrames.find(itCurPlacement->second.poolID);
			if (itPrevPool == m_lastAliasPlacementProducersByPoolAcrossFrames.end()) {
				continue;
			}

			const uint64_t curStart = itCurPlacement->second.startByte;
			const uint64_t curEnd = itCurPlacement->second.endByte;
			const uint64_t curPoolGeneration = itPoolState->second.generation;

			for (const auto& prevPlacementProducer : itPrevPool->second) {
				if (prevPlacementProducer.poolGeneration != curPoolGeneration) {
					continue;
				}

				const uint64_t overlapStart = std::max(curStart, prevPlacementProducer.startByte);
				const uint64_t overlapEnd = std::min(curEnd, prevPlacementProducer.endByte);
				if (overlapStart >= overlapEnd) {
					continue;
				}

				markCrossFrameWait(passQueueSlot, prevPlacementProducer.producer.queueSlot, prevPlacementProducer.producer.fenceValue);
				overlapTriggeredWaitCount++;
				if (overlapSampleCurrentResourceId == 0) {
					overlapSampleCurrentResourceId = rid;
					overlapSamplePreviousResourceId = prevPlacementProducer.resourceID;
				}
			}
		}
	};

	for (const auto& pr : m_framePasses) {
		const size_t passIndex = static_cast<size_t>(&pr - m_framePasses.data());
		std::visit([&](auto const& passAndResources) {
			using T = std::decay_t<decltype(passAndResources)>;
			if constexpr (!std::is_same_v<T, std::monostate>) {
				const size_t fallbackQueueSlot = passAndResources.resources.queueSlotOverride
					? static_cast<size_t>(static_cast<uint8_t>(*passAndResources.resources.queueSlotOverride))
					: QueueIndex(ResolveQueueKind(passAndResources.resources.queueSelection));
				const size_t passQueueSlot = passIndex < m_assignedQueueSlotsByFramePass.size()
					? m_assignedQueueSlotsByFramePass[passIndex]
					: fallbackQueueSlot;
				for (auto const& req : passAndResources.resources.frameResourceRequirements) {
					accumulateCrossFrameWaitForHandle(passQueueSlot, req.resourceHandleAndRange.resource);
				}
				for (auto const& tr : passAndResources.resources.internalTransitions) {
					accumulateCrossFrameWaitForHandle(passQueueSlot, tr.first.resource);
				}
			}
		}, pr.pass);
	}

	//if (overlapTriggeredWaitCount > 0) {
	//	spdlog::info(
	//		"RG cross-frame overlap waits: hits={} sampleCurrentResourceId={} samplePreviousResourceId={}",
	//		overlapTriggeredWaitCount,
	//		overlapSampleCurrentResourceId,
	//		overlapSamplePreviousResourceId);
	//}

	// Insert transitions to loop resources back to their initial states
	//ComputeResourceLoops();

	// Cut out repeat waits on the same fence per destination queue.
	const size_t slotCount = m_queueRegistry.SlotCount();
	std::vector<uint64_t> lastWaitFenceByDstQueue(slotCount, 0);
	for (auto& batch : batches) {
		const size_t batchQueueCount = batch.QueueCount();
		for (size_t dstIndex = 0; dstIndex < batchQueueCount; ++dstIndex) {
			for (auto phase : { BatchWaitPhase::BeforeTransitions, BatchWaitPhase::BeforeExecution }) {
				for (size_t srcIndex = 0; srcIndex < batchQueueCount; ++srcIndex) {
					if (!batch.HasQueueWait(phase, dstIndex, srcIndex)) {
						continue;
					}

					const auto waitFence = batch.GetQueueWaitFenceValue(phase, dstIndex, srcIndex);
					if (waitFence <= lastWaitFenceByDstQueue[dstIndex]) {
						batch.ClearQueueWait(phase, dstIndex, srcIndex);
					}
					else {
						lastWaitFenceByDstQueue[dstIndex] = waitFence;
					}
				}
			}
		}
	}

#if BUILD_TYPE == BUILD_TYPE_DEBUG
	// Sanity checks:
	// 1. No conflicting resource transitions in a batch

	auto queueName = [](QueueKind queue) -> const char* {
		switch (queue) {
		case QueueKind::Graphics: return "Graphics";
		case QueueKind::Compute: return "Compute";
		case QueueKind::Copy: return "Copy";
		default: return "Unknown";
		}
		};

	auto phaseName = [](BatchTransitionPhase phase) -> const char* {
		switch (phase) {
		case BatchTransitionPhase::BeforePasses: return "BeforePasses";
		case BatchTransitionPhase::AfterPasses: return "AfterPasses";
		default: return "Unknown";
		}
		};

	auto dumpTransitionsForBatchPhase = [&](size_t batchIndex, const PassBatch& batch, BatchTransitionPhase phase) {
		for (size_t queueIndex = 0; queueIndex < batch.QueueCount(); ++queueIndex) {
			const auto& transitions = batch.Transitions(queueIndex, phase);
			spdlog::error(
				"RG transition dump: batch={} phase={} queue={} count={}",
				batchIndex,
				phaseName(phase),
				queueIndex < static_cast<size_t>(QueueKind::Count) ? queueName(static_cast<QueueKind>(queueIndex)) : "Dynamic",
				transitions.size());

			for (size_t ti = 0; ti < transitions.size(); ++ti) {
				const auto& transition = transitions[ti];
				if (!transition.pResource) {
					spdlog::error(
						"  [{}] resource=<null>",
						ti);
					continue;
				}

				auto prevAccess = rhi::helpers::ResourceAccessMaskToString(transition.prevAccessType);
				auto newAccess = rhi::helpers::ResourceAccessMaskToString(transition.newAccessType);
				std::string mipLower = transition.range.mipLower.ToString();
				std::string mipUpper = transition.range.mipUpper.ToString();
				std::string sliceLower = transition.range.sliceLower.ToString();
				std::string sliceUpper = transition.range.sliceUpper.ToString();

				spdlog::error(
					"  [{}] resource='{}' id={} mip=[{}..{}] slice=[{}..{}] discard={} layout:{}->{} access:{}->{} sync:{}->{}",
					ti,
					transition.pResource->GetName(),
					transition.pResource->GetGlobalResourceID(),
					mipLower,
					mipUpper,
					sliceLower,
					sliceUpper,
					transition.discard,
					rhi::helpers::ResourceLayoutToString(transition.prevLayout),
					rhi::helpers::ResourceLayoutToString(transition.newLayout),
					prevAccess,
					newAccess,
					rhi::helpers::ResourceSyncToString(transition.prevSyncState),
					rhi::helpers::ResourceSyncToString(transition.newSyncState));
			}
		}
		};

	// Validate per transition phase.
	// NOTE: transitions for the same resource can be valid across phases
	// (e.g. BeforePasses transitions into RT state, then AfterPasses transitions back for consumers).
	for (size_t bi = 0; bi < batches.size(); bi++) {
		auto& batch = batches[bi];
		for (size_t phaseIndex = 0; phaseIndex < static_cast<size_t>(BatchTransitionPhase::Count); ++phaseIndex) {
			std::vector<ResourceTransition> phaseTransitions;
			const auto phase = static_cast<BatchTransitionPhase>(phaseIndex);
			for (size_t queueIndex = 0; queueIndex < batch.QueueCount(); ++queueIndex) {
				const auto& transitions = batch.Transitions(queueIndex, phase);
				phaseTransitions.insert(phaseTransitions.end(), transitions.begin(), transitions.end());
			}
			

			// Validate this phase
			TransitionConflict out;
			if (bool ok = ValidateNoConflictingTransitions(phaseTransitions, &out); !ok) {
				const uint32_t conflictMip = static_cast<uint32_t>(out.mip);
				const uint32_t conflictSlice = static_cast<uint32_t>(out.slice);
				spdlog::error(
					"Render graph has conflicting resource transitions in batch {} phase {} ({}) (resource='{}' mip={} slice={})",
					bi,
					phaseIndex,
					phaseName(phase),
					out.resource ? out.resource->GetName() : std::string("<null>"),
					conflictMip,
					conflictSlice);
				dumpTransitionsForBatchPhase(bi, batch, phase);
				throw std::runtime_error("Render graph has conflicting resource transitions!");
			}
		}
	}

	// No out-of-order fence signals on any queue.
	// For each queue, enabled signals must be strictly monotonically increasing
	// in execution order: within a batch (AfterTransitions < AfterExecution < AfterCompletion)
	// and across batches (last signal of batch N < first signal of batch N+1).
	{
		auto signalPhaseName = [](BatchSignalPhase phase) -> const char* {
			switch (phase) {
			case BatchSignalPhase::AfterTransitions: return "AfterTransitions";
			case BatchSignalPhase::AfterExecution:   return "AfterExecution";
			case BatchSignalPhase::AfterCompletion:  return "AfterCompletion";
			default: return "Unknown";
			}
		};

		std::vector<UINT64> lastSignalValue(slotCount, 0);
		std::vector<int> lastSignalBatch(slotCount, -1);
		std::vector<BatchSignalPhase> lastSignalPhase(slotCount);

		for (size_t bi = 0; bi < batches.size(); ++bi) {
			const auto& batch = batches[bi];
			for (size_t qi = 0; qi < batch.QueueCount(); ++qi) {
				for (size_t sp = 0; sp < PassBatch::kSignalPhaseCount; ++sp) {
					const auto phase = static_cast<BatchSignalPhase>(sp);
					if (!batch.HasQueueSignal(phase, qi)) continue;

					UINT64 fenceVal = batch.GetQueueSignalFenceValue(phase, qi);
					if (fenceVal == 0) {
						spdlog::error(
							"Zero-value fence signal on queue {}: "
							"batch {} phase {} has signal enabled with fence value 0. "
							"This will produce an invalid timeline signal at execution time.",
							qi, bi, signalPhaseName(phase));
						throw std::runtime_error("Render graph has zero-value fence signal!");
					}
					if (lastSignalBatch[qi] >= 0 && fenceVal <= lastSignalValue[qi]) {
						spdlog::error(
							"Out-of-order fence signal on queue {}: "
							"batch {} phase {} signals fence={}, but batch {} phase {} already signaled fence={}",
							qi,
							bi, signalPhaseName(phase), fenceVal,
							lastSignalBatch[qi], signalPhaseName(lastSignalPhase[qi]), lastSignalValue[qi]);
						throw std::runtime_error("Render graph has out-of-order fence signals!");
					}
					lastSignalValue[qi] = fenceVal;
					lastSignalBatch[qi] = static_cast<int>(bi);
					lastSignalPhase[qi] = phase;
				}
			}
		}
	}

	// No wait references a fence value that no signal on that source queue will produce.
	{
		// Collect the set of all signaled fence values per queue.
		std::vector<std::unordered_set<UINT64>> signaledValues(slotCount);
		for (const auto& batch : batches) {
			for (size_t qi = 0; qi < batch.QueueCount(); ++qi) {
				for (size_t sp = 0; sp < PassBatch::kSignalPhaseCount; ++sp) {
					if (batch.HasQueueSignal(static_cast<BatchSignalPhase>(sp), qi))
						signaledValues[qi].insert(batch.GetQueueSignalFenceValue(static_cast<BatchSignalPhase>(sp), qi));
				}
			}
		}

		auto waitPhaseName = [](BatchWaitPhase phase) -> const char* {
			switch (phase) {
			case BatchWaitPhase::BeforeTransitions: return "BeforeTransitions";
			case BatchWaitPhase::BeforeExecution:   return "BeforeExecution";
			case BatchWaitPhase::BeforeAfterPasses:  return "BeforeAfterPasses";
			default: return "Unknown";
			}
		};

		for (size_t bi = 0; bi < batches.size(); ++bi) {
			const auto& batch = batches[bi];
			for (size_t wp = 0; wp < PassBatch::kWaitPhaseCount; ++wp) {
				const auto waitPhase = static_cast<BatchWaitPhase>(wp);
				for (size_t dst = 0; dst < batch.QueueCount(); ++dst) {
					for (size_t src = 0; src < batch.QueueCount(); ++src) {
						if (dst == src) continue;
						if (!batch.HasQueueWait(waitPhase, dst, src)) continue;

						UINT64 waitVal = batch.GetQueueWaitFenceValue(waitPhase, dst, src);
						if (signaledValues[src].find(waitVal) == signaledValues[src].end()) {
							spdlog::error(
								"Dangling fence wait: batch {} queue {} waits on queue {} fence={} "
								"at phase {}, but no signal for that value exists",
								bi, dst, src,
								waitVal, waitPhaseName(waitPhase));
							throw std::runtime_error("Render graph has a wait on an unsignaled fence value!");
						}
					}
				}
			}
		}
	}

	// Deadlock detection: check for circular wait-for-signal dependencies within
	// each batch. Within a batch, queues execute concurrently with this timeline:
	//   [BeforeTransitions waits] -> transitions -> AfterTransitions signal
	//   [BeforeExecution waits]   -> passes      -> AfterExecution signal
	//   [BeforeAfterPasses waits] -> post-trans  -> AfterCompletion signal
	// A deadlock occurs when queue A waits for a signal that queue B can only
	// produce after B is blocked waiting for A (or transitively through others).
	//
	// Algorithm: fixed-point iteration over per-queue "progress" levels.
	// progress[q] = maximum signal ordinal q can reach (3 = healthy).
	//   0 = stuck before transitions (no signals produced)
	//   1 = AfterTransitions produced, stuck before execution
	//   2 = AfterExecution produced, stuck before after-passes
	//   3 = AfterCompletion produced (no deadlock)
	{
		constexpr int kWaitBlockLevel[] = { 0, 1, 2 };   // BeforeTransitions, BeforeExecution, BeforeAfterPasses
		constexpr int kSignalRequired[] = { 1, 2, 3 };   // AfterTransitions, AfterExecution, AfterCompletion

		auto signalPhaseName = [](int sp) -> const char* {
			switch (sp) {
			case 0: return "AfterTransitions";
			case 1: return "AfterExecution";
			case 2: return "AfterCompletion";
			default: return "Unknown";
			}
		};

		auto waitPhaseName = [](int wp) -> const char* {
			switch (wp) {
			case 0: return "BeforeTransitions";
			case 1: return "BeforeExecution";
			case 2: return "BeforeAfterPasses";
			default: return "Unknown";
			}
		};

		for (size_t bi = 0; bi < batches.size(); ++bi) {
			const auto& batch = batches[bi];
			const size_t qc = batch.QueueCount();

			std::vector<int> progress(qc, 3);

			bool changed = true;
			while (changed) {
				changed = false;
				for (size_t dst = 0; dst < qc; ++dst) {
					for (size_t wp = 0; wp < PassBatch::kWaitPhaseCount; ++wp) {
						int blockLevel = kWaitBlockLevel[wp];
						if (progress[dst] <= blockLevel) continue; // already stuck here or earlier

						for (size_t src = 0; src < qc; ++src) {
							if (dst == src) continue;
							if (!batch.queueWaitEnabled[wp][dst][src]) continue;

							UINT64 fv = batch.queueWaitFenceValue[wp][dst][src];

							// Determine which signal phase of src this wait targets.
							int requiredSrcProgress = -1;
							for (size_t sp = 0; sp < PassBatch::kSignalPhaseCount; ++sp) {
								if (fv == batch.queueSignalFenceValue[sp][src]) {
									requiredSrcProgress = kSignalRequired[sp];
									break;
								}
							}
							if (requiredSrcProgress < 0) continue; // cross-batch wait, always safe

							if (progress[src] < requiredSrcProgress) {
								progress[dst] = std::min(progress[dst], blockLevel);
								changed = true;
							}
						}
					}
				}
			}

			for (size_t q = 0; q < qc; ++q) {
				if (progress[q] >= 3) continue;

				auto kind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(q)));
				auto inst = m_queueRegistry.GetInstance(static_cast<QueueSlotIndex>(static_cast<uint8_t>(q)));

				spdlog::error("DEADLOCK DETECTED: batch {} queue slot {} ({}:{}) stuck at progress {} "
					"(0=before-transitions, 1=after-transitions, 2=after-execution, 3=healthy)",
					bi, q, queueName(kind), inst, progress[q]);

				// Log the same-batch waits contributing to the cycle
				for (size_t wp = 0; wp < PassBatch::kWaitPhaseCount; ++wp) {
					for (size_t src = 0; src < qc; ++src) {
						if (q == src || !batch.queueWaitEnabled[wp][q][src]) continue;
						UINT64 fv = batch.queueWaitFenceValue[wp][q][src];
						for (size_t sp = 0; sp < PassBatch::kSignalPhaseCount; ++sp) {
							if (fv == batch.queueSignalFenceValue[sp][src]) {
								spdlog::error("  slot {} waits at {} for slot {} {} (fence={}), src progress={}",
									q, waitPhaseName(static_cast<int>(wp)),
									src, signalPhaseName(static_cast<int>(sp)),
									fv, progress[src]);
							}
						}
					}
				}

				throw std::runtime_error("Render graph has a GPU queue deadlock!");
			}
		}
	}

#endif
}

std::tuple<int, int, int> RenderGraph::GetBatchesToWaitOn(
	const ComputePassAndResources& pass,
	std::unordered_map<uint64_t, unsigned int> const& transitionHistory,
	std::unordered_map<uint64_t, unsigned int> const& producerHistory,
	std::unordered_map<uint64_t, unsigned int> const& usageHistory,
	std::unordered_set<uint64_t> const& resourcesTransitionedThisPass)
{
	int latestTransition = -1, latestProducer = -1, latestUsage = -1;

	auto processResource = [&](ResourceRegistry::RegistryHandle const& res) {
		uint64_t id = res.GetGlobalResourceID();
		for (auto rid : m_aliasingSubsystem.GetSchedulingEquivalentIDs(id, aliasPlacementRangesByID)) {
			auto itT = transitionHistory.find(rid);
			if (itT != transitionHistory.end())
				latestTransition = std::max(latestTransition, (int)itT->second);

			auto itP = producerHistory.find(rid);
			if (itP != producerHistory.end())
				latestProducer = std::max(latestProducer, (int)itP->second);
		}
		};

	for (auto const& req : pass.resources.frameResourceRequirements)
		processResource(req.resourceHandleAndRange.resource);

	for (auto& transitionID : resourcesTransitionedThisPass) { // We only need to wait on the latest usage for resources that will be transitioned in this batch
		for (auto rid : m_aliasingSubsystem.GetSchedulingEquivalentIDs(transitionID, aliasPlacementRangesByID)) {
			if (usageHistory.contains(rid)) {
				latestUsage = std::max(latestUsage, (int)usageHistory.at(rid));
			}
		}
	}

	return { latestTransition, latestProducer, latestUsage };
}

std::tuple<int, int, int> RenderGraph::GetBatchesToWaitOn(
	const RenderPassAndResources& pass,
	std::unordered_map<uint64_t, unsigned int> const& transitionHistory,
	std::unordered_map<uint64_t, unsigned int> const& producerHistory,
	std::unordered_map<uint64_t, unsigned int> const& usageHistory,
	std::unordered_set<uint64_t> const& resourcesTransitionedThisPass)
{
	int latestTransition = -1, latestProducer = -1, latestUsage = -1;

	auto processResource = [&](ResourceRegistry::RegistryHandle const& res) {
		uint64_t id = res.GetGlobalResourceID();
		for (auto rid : m_aliasingSubsystem.GetSchedulingEquivalentIDs(id, aliasPlacementRangesByID)) {
			auto itT = transitionHistory.find(rid);
			if (itT != transitionHistory.end())
				latestTransition = std::max(latestTransition, (int)itT->second);

			auto itP = producerHistory.find(rid);
			if (itP != producerHistory.end())
				latestProducer = std::max(latestProducer, (int)itP->second);
		}
		};

	for (auto const& req : pass.resources.frameResourceRequirements)
		processResource(req.resourceHandleAndRange.resource);

	for (auto& transitionID : resourcesTransitionedThisPass) { // We only need to wait on the latest usage for resources that will be transitioned in this batch
		for (auto rid : m_aliasingSubsystem.GetSchedulingEquivalentIDs(transitionID, aliasPlacementRangesByID)) {
			if (usageHistory.contains(rid)) {
				latestUsage = std::max(latestUsage, (int)usageHistory.at(rid));
			}
		}
	}

	return { latestTransition, latestProducer, latestUsage };
}

std::tuple<int, int, int> RenderGraph::GetBatchesToWaitOn(
	const CopyPassAndResources& pass,
	std::unordered_map<uint64_t, unsigned int> const& transitionHistory,
	std::unordered_map<uint64_t, unsigned int> const& producerHistory,
	std::unordered_map<uint64_t, unsigned int> const& usageHistory,
	std::unordered_set<uint64_t> const& resourcesTransitionedThisPass)
{
	int latestTransition = -1, latestProducer = -1, latestUsage = -1;

	auto processResource = [&](ResourceRegistry::RegistryHandle const& res) {
		uint64_t id = res.GetGlobalResourceID();
		for (auto rid : m_aliasingSubsystem.GetSchedulingEquivalentIDs(id, aliasPlacementRangesByID)) {
			auto itT = transitionHistory.find(rid);
			if (itT != transitionHistory.end())
				latestTransition = std::max(latestTransition, (int)itT->second);

			auto itP = producerHistory.find(rid);
			if (itP != producerHistory.end())
				latestProducer = std::max(latestProducer, (int)itP->second);
		}
		};

	for (auto const& req : pass.resources.frameResourceRequirements)
		processResource(req.resourceHandleAndRange.resource);

	for (auto& transitionID : resourcesTransitionedThisPass) {
		for (auto rid : m_aliasingSubsystem.GetSchedulingEquivalentIDs(transitionID, aliasPlacementRangesByID)) {
			if (usageHistory.contains(rid)) {
				latestUsage = std::max(latestUsage, (int)usageHistory.at(rid));
			}
		}
	}

	return { latestTransition, latestProducer, latestUsage };
}

void RenderGraph::MaterializeUnmaterializedResources(const std::unordered_set<uint64_t>* onlyResourceIDs) {
	// Returns the backing generation if the resource was materialized (or already materialized), or nullopt if skipped.
	auto materializeOne = [&](uint64_t id, Resource* resource) -> std::optional<uint64_t> {
		if (onlyResourceIDs && onlyResourceIDs->find(id) == onlyResourceIDs->end()) {
			return std::nullopt;
		}
		if (!resource) {
			return std::nullopt;
		}

		auto texture = dynamic_cast<PixelBuffer*>(resource);
		if (texture) {
			if (!texture->IsMaterialized()) {
				auto itAlias = aliasMaterializeOptionsByID.find(id);
				if (itAlias != aliasMaterializeOptionsByID.end()) {
					if (std::holds_alternative<PixelBuffer::MaterializeOptions>(itAlias->second)) {
						auto& options = std::get<PixelBuffer::MaterializeOptions>(itAlias->second);
						if (options.aliasPlacement.has_value()) {
							const auto& ap = options.aliasPlacement.value();
							spdlog::info(
								"RG alias materialize: id={} name='{}' pool={} offset={}",
								id,
								resource->GetName(),
								ap.poolID.has_value() ? ap.poolID.value() : 0ull,
								ap.offset);
						}
						texture->Materialize(&options);
					}
				}
				else {
					if (texture->GetDescription().allowAlias) {
						const bool hasManualAliasPool = texture->GetDescription().aliasingPoolID.has_value();
						const bool hasAutoAliasPool = autoAliasPoolByID.find(id) != autoAliasPoolByID.end();
						const bool aliasPlacementRequiredThisFrame = hasManualAliasPool || hasAutoAliasPool;

						if (onlyResourceIDs && aliasPlacementRequiredThisFrame) {
							throw std::runtime_error(
								"Aliasing placement missing for used aliased resource during frame materialization. Resource ID: " + std::to_string(id));
						}

						if (onlyResourceIDs && !aliasPlacementRequiredThisFrame) {
							spdlog::debug(
								"RG alias fallback materialize: id={} name='{}' allowAlias=1 but no pool assignment this frame; materializing standalone",
								id,
								resource->GetName());
						}

						// Setup-time eager materialization happens before alias planning.
						// Defer aliased resources until compile-time placement is available.
						if (!onlyResourceIDs) {
							return std::nullopt;
						}
					}
					texture->Materialize();
				}
			}

			return texture->GetBackingGeneration();
		}

		auto buffer = dynamic_cast<Buffer*>(resource);
		if (!buffer) {
			return std::nullopt;
		}

		if (!buffer->IsMaterialized()) {
			auto itAlias = aliasMaterializeOptionsByID.find(id);
			if (itAlias != aliasMaterializeOptionsByID.end()) {
				if (std::holds_alternative<BufferBase::MaterializeOptions>(itAlias->second)) {
					auto& options = std::get<BufferBase::MaterializeOptions>(itAlias->second);
					if (options.aliasPlacement.has_value()) {
						const auto& ap = options.aliasPlacement.value();
						spdlog::info(
							"RG alias materialize (buffer): id={} name='{}' pool={} offset={}",
							id,
							resource->GetName(),
							ap.poolID.has_value() ? ap.poolID.value() : 0ull,
							ap.offset);
					}
					buffer->Materialize(&options);
				}
			}
			else {
				if (buffer->IsAliasingAllowed()) {
					const bool hasManualAliasPool = buffer->GetAliasingPoolHint().has_value();
					const bool hasAutoAliasPool = autoAliasPoolByID.find(id) != autoAliasPoolByID.end();
					const bool aliasPlacementRequiredThisFrame = hasManualAliasPool || hasAutoAliasPool;

					if (onlyResourceIDs && aliasPlacementRequiredThisFrame) {
						throw std::runtime_error(
							"Aliasing placement missing for used aliased buffer during frame materialization. Resource ID: " + std::to_string(id));
					}

					if (onlyResourceIDs && !aliasPlacementRequiredThisFrame) {
						spdlog::debug(
							"RG alias fallback materialize (buffer): id={} name='{}' allowAlias=1 but no pool assignment this frame; materializing standalone",
							id,
							resource->GetName());
					}

					if (!onlyResourceIDs) {
						return std::nullopt;
					}
				}
				buffer->Materialize();
			}
		}

		return buffer->GetBackingGeneration();
	};

	// Collect all unique {id, resource*} items to materialize
	std::vector<std::pair<uint64_t, Resource*>> items;
	items.reserve(resourcesByID.size() + m_transientFrameResourcesByID.size());
	std::unordered_set<uint64_t> seen;

	for (auto& [id, resource] : resourcesByID) {
		if (seen.insert(id).second && resource) {
			items.emplace_back(id, resource.get());
		}
	}

	for (auto& [id, resource] : m_transientFrameResourcesByID) {
		if (resourcesByID.contains(id)) continue;
		if (seen.insert(id).second && resource) {
			items.emplace_back(id, resource.get());
		}
	}

	auto collectFromHandle = [&](const ResourceRegistry::RegistryHandle& handle) {
		const uint64_t id = handle.GetGlobalResourceID();
		if (!seen.insert(id).second) return;
		Resource* resource = handle.IsEphemeral() ? handle.GetEphemeralPtr() : _registry.Resolve(handle);
		if (resource) {
			TrackTransientFrameResource(resource);
			items.emplace_back(id, resource);
		}
	};

	for (const auto& pr : m_framePasses) {
		if (pr.type == PassType::Compute) {
			auto const& p = std::get<ComputePassAndResources>(pr.pass);
			for (auto const& req : p.resources.frameResourceRequirements) {
				collectFromHandle(req.resourceHandleAndRange.resource);
			}
			for (auto const& t : p.resources.internalTransitions) {
				collectFromHandle(t.first.resource);
			}
		}
		else if (pr.type == PassType::Render) {
			auto const& p = std::get<RenderPassAndResources>(pr.pass);
			for (auto const& req : p.resources.frameResourceRequirements) {
				collectFromHandle(req.resourceHandleAndRange.resource);
			}
			for (auto const& t : p.resources.internalTransitions) {
				collectFromHandle(t.first.resource);
			}
		}
		else if (pr.type == PassType::Copy) {
			auto const& p = std::get<CopyPassAndResources>(pr.pass);
			for (auto const& req : p.resources.frameResourceRequirements) {
				collectFromHandle(req.resourceHandleAndRange.resource);
			}
			for (auto const& t : p.resources.internalTransitions) {
				collectFromHandle(t.first.resource);
			}
		}
	}

	// Parallel materialize phase
	struct GenerationResult {
		uint64_t id = 0;
		uint64_t generation = 0;
		bool valid = false;
	};
	std::vector<GenerationResult> genResults(items.size());

	ParallelForOptional("Materialize", items.size(), [&](size_t i) {
		auto [id, resource] = items[i];
		auto gen = materializeOne(id, resource);
		if (gen.has_value()) {
			genResults[i] = { id, gen.value(), true };
		}
	}, true); // Disable or now, resource creation creates flecs entities in renderer

	// Merge generation results
	for (auto& r : genResults) {
		if (r.valid) {
			resourceBackingGenerationByID[r.id] = r.generation;
		}
	}
}

void RenderGraph::ResizeQueueParallelVectors() {
	const size_t qc = m_queueRegistry.SlotCount();
	m_compiledLastProducerBatchByResourceByQueue.resize(qc);
	m_hasPendingFrameStartQueueWait.assign(qc, std::vector<uint8_t>(qc, 0));
	m_pendingFrameStartQueueWaitFenceValue.assign(qc, std::vector<UINT64>(qc, 0));
}

void RenderGraph::EnsureMinimumAutomaticSchedulingQueues() {
	auto autoAssignableCountForKind = [this](QueueKind kind) {
		uint8_t count = 0;
		for (size_t i = 0; i < m_queueRegistry.SlotCount(); ++i) {
			const auto slotIndex = static_cast<QueueSlotIndex>(static_cast<uint8_t>(i));
			if (m_queueRegistry.GetKind(slotIndex) == kind && m_queueRegistry.IsAutoAssignable(slotIndex)) {
				++count;
			}
		}
		return count;
	};

	auto queueNamePrefix = [](QueueKind kind) -> const char* {
		switch (kind) {
		case QueueKind::Graphics: return "AutoGraphics";
		case QueueKind::Compute: return "AutoCompute";
		case QueueKind::Copy: return "AutoCopy";
		default: return "AutoQueue";
		}
	};

	for (size_t kindIndex = 0; kindIndex < static_cast<size_t>(QueueKind::Count); ++kindIndex) {
		const QueueKind kind = static_cast<QueueKind>(kindIndex);
		const uint8_t minimumQueueCount = m_minAutomaticSchedulingQueuesByKind[kindIndex];
		uint8_t currentAutoQueueCount = autoAssignableCountForKind(kind);
		while (currentAutoQueueCount < minimumQueueCount) {
			std::string queueName = std::string(queueNamePrefix(kind)) + std::to_string(currentAutoQueueCount);
			CreateQueue(kind, queueName.c_str(), QueueAutoAssignmentPolicy::AllowAutomaticScheduling);
			++currentAutoQueueCount;
		}
	}

	ResizeQueueParallelVectors();
}

void RenderGraph::Setup() {
	DeletionManager::GetInstance().Initialize();

	// Setup the statistics manager
	if (m_statisticsService) {
		m_statisticsService->ClearAll();
	}
	auto& manager = DeviceManager::GetInstance();
	if (m_statisticsService) {
		m_statisticsService->RegisterQueue(manager.GetGraphicsQueue().GetKind());
		m_statisticsService->RegisterQueue(manager.GetComputeQueue().GetKind());
		m_statisticsService->SetupQueryHeap();
	}

	auto device = DeviceManager::GetInstance().GetDevice();

	auto result = device.CreateTimeline(m_readbackFence);
	result = device.CreateTimeline(m_frameStartSyncFence);

	if (m_readbackService) {
		m_readbackService->Initialize(m_readbackFence.Get());
	}

	// Populate the queue registry with the 3 primary queues.
	// Each Register() call creates a CommandListPool and Timeline for the slot.
	{
		auto& gfxQ = DeviceManager::GetInstance().GetGraphicsQueue();
		auto& compQ = DeviceManager::GetInstance().GetComputeQueue();
		auto& copyQ = DeviceManager::GetInstance().GetCopyQueue();
		m_queueRegistry.Register({ QueueKind::Graphics, 0 }, gfxQ, device);
		m_queueRegistry.Register({ QueueKind::Compute, 0 }, compQ, device);
		m_queueRegistry.Register({ QueueKind::Copy,    0 }, copyQ, device);
	}
	EnsureMinimumAutomaticSchedulingQueues();

	// Size queue-parallel member vectors to match the registry.
	// Done after both primary queue registration and extension initialization
	// since extensions may create additional queues.
	ResizeQueueParallelVectors();

	// Notify extensions that the render graph is set up.
	// Extensions may create additional queues or allocate resources here.
	for (auto& ext : m_extensions) {
		if (ext) ext->Initialize(*this);
	}

	// Re-size in case extensions added queues.
	ResizeQueueParallelVectors();

	m_getUseAsyncCompute = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetUseAsyncCompute() : false;
	};
	m_getHeavyDebug = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetHeavyDebug() : false;
	};
	m_getAutoAliasMode = [this]() {
		const auto mode = m_renderGraphSettingsService
			? m_renderGraphSettingsService->GetAutoAliasMode()
			: static_cast<uint8_t>(AutoAliasMode::Off);
		return static_cast<AutoAliasMode>(mode);
	};
	m_getAutoAliasPackingStrategy = [this]() {
		const auto strategy = m_renderGraphSettingsService
			? m_renderGraphSettingsService->GetAutoAliasPackingStrategy()
			: static_cast<uint8_t>(AutoAliasPackingStrategy::GreedySweepLine);
		return static_cast<AutoAliasPackingStrategy>(strategy);
	};
	m_getAutoAliasEnableLogging = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetAutoAliasEnableLogging() : false;
	};
	m_getAutoAliasLogExclusionReasons = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetAutoAliasLogExclusionReasons() : false;
	};
	m_getQueueSchedulingEnableLogging = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetQueueSchedulingEnableLogging() : false;
	};
	m_getQueueSchedulingWidthScale = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetQueueSchedulingWidthScale() : 1.0f;
	};
	m_getQueueSchedulingPenaltyBias = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetQueueSchedulingPenaltyBias() : 0.0f;
	};
	m_getQueueSchedulingMinPenalty = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetQueueSchedulingMinPenalty() : 1.0f;
	};
	m_getQueueSchedulingResourcePressureWeight = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetQueueSchedulingResourcePressureWeight() : 1.0f;
	};
	m_getQueueSchedulingUavPressureWeight = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetQueueSchedulingUavPressureWeight() : 0.5f;
	};
	m_getAutoAliasPoolRetireIdleFrames = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetAutoAliasPoolRetireIdleFrames() : 120u;
	};
	m_getAutoAliasPoolGrowthHeadroom = [this]() {
		return m_renderGraphSettingsService ? m_renderGraphSettingsService->GetAutoAliasPoolGrowthHeadroom() : 1.5f;
	};
	MaterializeUnmaterializedResources();

	// Pass setup is intentionally serial. Many passes touch shared ECS/flecs world state
	// and singleton managers during Setup(), and iterating flecs queries from our task
	// worker threads can corrupt flecs iterator stack state.
	ParallelForOptional("PassSetup", m_masterPassList.size(), [this](size_t i) {
		auto& pass = m_masterPassList[i];
		switch (pass.type) {
		case PassType::Render: {
			auto& renderPass = std::get<RenderPassAndResources>(pass.pass);
			renderPass.pass->SetResourceRegistryView(
				std::make_unique<ResourceRegistryView>(_registry, renderPass.resources.identifierSet),
				renderPass.resources.autoDescriptorShaderResources,
				renderPass.resources.autoDescriptorUnorderedAccessViews);
			renderPass.pass->Setup();
			break;
		}
		case PassType::Compute: {
			auto& computePass = std::get<ComputePassAndResources>(pass.pass);
			computePass.pass->SetResourceRegistryView(
				std::make_unique<ResourceRegistryView>(_registry, computePass.resources.identifierSet),
				computePass.resources.autoDescriptorShaderResources,
				computePass.resources.autoDescriptorConstantBuffers,
				computePass.resources.autoDescriptorUnorderedAccessViews);
			computePass.pass->Setup();
			break;
		}
		case PassType::Copy: {
			auto& copyPass = std::get<CopyPassAndResources>(pass.pass);
			copyPass.pass->SetResourceRegistryView(std::make_unique<ResourceRegistryView>(_registry, copyPass.resources.identifierSet));
			copyPass.pass->Setup();
			break;
		}
		}
	}, true);
}

void RenderGraph::AddRenderPass(std::shared_ptr<RenderPass> pass, RenderPassParameters& resources, std::string name, std::vector<ResolverSnapshot> resolverSnapshots) {
	RenderPassAndResources passAndResources;
	passAndResources.pass = pass;
	passAndResources.resources = resources;
	passAndResources.name = name;
	passAndResources.resolverSnapshots = std::move(resolverSnapshots);
	AnyPassAndResources passAndResourcesAny;
	passAndResourcesAny.type = PassType::Render;
	passAndResourcesAny.pass = std::move(passAndResources);
	passAndResourcesAny.name = name;
	m_masterPassList.push_back(std::move(passAndResourcesAny));
	if (name != "") {
		renderPassesByName[name] = pass;
	}
}

void RenderGraph::AddComputePass(std::shared_ptr<ComputePass> pass, ComputePassParameters& resources, std::string name, std::vector<ResolverSnapshot> resolverSnapshots) {
	ComputePassAndResources passAndResources;
	passAndResources.pass = pass;
	passAndResources.resources = resources;
	passAndResources.name = name;
	passAndResources.resolverSnapshots = std::move(resolverSnapshots);
	AnyPassAndResources passAndResourcesAny;
	passAndResourcesAny.type = PassType::Compute;
	passAndResourcesAny.pass = std::move(passAndResources);
	passAndResourcesAny.name = name;
	m_masterPassList.push_back(std::move(passAndResourcesAny));
	if (name != "") {
		computePassesByName[name] = pass;
	}
}

void RenderGraph::AddCopyPass(std::shared_ptr<CopyPass> pass, CopyPassParameters& resources, std::string name, std::vector<ResolverSnapshot> resolverSnapshots) {
	CopyPassAndResources passAndResources;
	passAndResources.pass = pass;
	passAndResources.resources = resources;
	passAndResources.name = name;
	passAndResources.resolverSnapshots = std::move(resolverSnapshots);
	AnyPassAndResources passAndResourcesAny;
	passAndResourcesAny.type = PassType::Copy;
	passAndResourcesAny.pass = std::move(passAndResources);
	passAndResourcesAny.name = name;
	m_masterPassList.push_back(std::move(passAndResourcesAny));
}

void RenderGraph::AddResource(std::shared_ptr<Resource> resource, bool transition) {
	if (resourcesByID.contains(resource->GetGlobalResourceID())) {
		return; // Resource already added
	}
	auto& name = resource->GetName();

#ifdef _DEBUG
	//if (name == L"") {
	//	throw std::runtime_error("Resource name cannot be empty");
	//}
	//else if (resourcesByName.find(name) != resourcesByName.end()) {
	//	throw std::runtime_error("Resource with name " + ws2s(name) + " already exists");
	//}
#endif

	resourcesByName[name] = resource;
	resourcesByID[resource->GetGlobalResourceID()] = resource;

	if (auto texture = std::dynamic_pointer_cast<PixelBuffer>(resource)) {
		texture->EnsureVirtualDescriptorSlotsAllocated();
	}
	if (auto buffer = std::dynamic_pointer_cast<Buffer>(resource)) {
		buffer->EnsureVirtualDescriptorSlotsAllocated();
	}
}

void RenderGraph::TrackTransientFrameResource(const std::shared_ptr<Resource>& resource) {
	if (!resource) {
		return;
	}

	const uint64_t resourceID = resource->GetGlobalResourceID();
	if (resourcesByID.contains(resourceID)) {
		return;
	}

	m_transientFrameResourcesByID[resourceID] = resource;
	const auto& resourceName = resource->GetName();
	if (!resourceName.empty()) {
		m_transientFrameResourcesByName[resourceName] = resource;
	}
}

void RenderGraph::TrackTransientFrameResource(Resource* resource) {
	if (!resource) {
		return;
	}

	if (auto shared = resource->weak_from_this().lock()) {
		TrackTransientFrameResource(shared);
	}
}

std::shared_ptr<Resource> RenderGraph::GetResourceByName(const std::string& name) {
	auto transientIt = m_transientFrameResourcesByName.find(name);
	if (transientIt != m_transientFrameResourcesByName.end()) {
		return transientIt->second;
	}
	auto it = resourcesByName.find(name);
	if (it != resourcesByName.end()) {
		return it->second;
	}
	return nullptr;
}

std::shared_ptr<Resource> RenderGraph::GetResourceByID(const uint64_t id) {
	auto transientIt = m_transientFrameResourcesByID.find(id);
	if (transientIt != m_transientFrameResourcesByID.end()) {
		return transientIt->second;
	}
	auto it = resourcesByID.find(id);
	if (it != resourcesByID.end()) {
		return it->second;
	}
	return nullptr;
}
std::shared_ptr<RenderPass> RenderGraph::GetRenderPassByName(const std::string& name) {
	if (renderPassesByName.find(name) != renderPassesByName.end()) {
		return renderPassesByName[name];
	}
	else {
		return nullptr;
	}
}

std::shared_ptr<ComputePass> RenderGraph::GetComputePassByName(const std::string& name) {
	if (computePassesByName.find(name) != computePassesByName.end()) {
		return computePassesByName[name];
	}
	else {
		return nullptr;
	}
}

void RenderGraph::Update(const UpdateExecutionContext& context, rhi::Device device) {
	ResetForFrame();

	// Poll readback completions early so that passes (e.g. CLod streaming)
	// can react to GPU-produced data from the previous frame immediately,
	// rather than waiting until post-Present.
	if (m_readbackService) {
		m_readbackService->ProcessReadbackRequests();
	}

	if (m_statisticsService) {
		m_statisticsService->BeginFrame();
	}

	auto toMilliseconds = [](auto duration) {
		return std::chrono::duration<double, std::milli>(duration).count();
	};

	for (auto& pr : m_masterPassList) {	
		// Resolve into type and update
		std::visit([&](auto& obj) {
			using T = std::decay_t<decltype(obj)>;
			if constexpr (std::is_same_v<T, std::monostate>) {
				// no-op
			}
			else {
				if (m_statisticsService && obj.statisticsIndex < 0) {
					if constexpr (std::is_same_v<T, RenderPassAndResources>) {
						obj.statisticsIndex = static_cast<int>(m_statisticsService->RegisterPass(obj.name, obj.resources.isGeometryPass));
					}
					else {
						obj.statisticsIndex = static_cast<int>(m_statisticsService->RegisterPass(obj.name, false));
					}
				}

				const auto start = std::chrono::steady_clock::now();
				obj.pass->Update(context);
				if (m_statisticsService && obj.statisticsIndex >= 0) {
					m_statisticsService->RecordCpuUpdateTime(
						static_cast<unsigned>(obj.statisticsIndex),
						toMilliseconds(std::chrono::steady_clock::now() - start));
				}
			}
			}, pr.pass);
	}

	CompileFrame(device, context.frameIndex, context.hostData);
}

#define IFDEBUG(x) 

namespace {
	// ExecuteTransitions: applies state-tracker bookkeeping AND records
	// barriers.  Used only in the non-parallel fallback path.
	void ExecuteTransitions(std::vector<ResourceTransition>& transitions,
		CommandRecordingManager* crm,
		QueueKind queueKind,
		rhi::CommandList& commandList) {
		rhi::helpers::OwnedBarrierBatch batch;
		for (auto& transition : transitions) {
			std::vector<ResourceTransition> dummy;
			transition.pResource->GetStateTracker()->Apply(
				transition.range, transition.pResource,
				{ transition.newAccessType, transition.newLayout, transition.newSyncState }, dummy);
			auto bg = transition.pResource->GetEnhancedBarrierGroup(
				transition.range, transition.prevAccessType, transition.newAccessType,
				transition.prevLayout, transition.newLayout,
				transition.prevSyncState, transition.newSyncState);
			const size_t textureStart = batch.textures.size();
			batch.Append(bg);
			if (transition.discard) {
				for (size_t i = textureStart; i < batch.textures.size(); ++i) {
					batch.textures[i].discard = true;
				}
			}
		}
		if (!batch.Empty()) {
			commandList.Barriers(batch.View());
		}
	}

	// RecordTransitionBarriers: records barrier commands into a CL
	void RecordTransitionBarriers(std::vector<ResourceTransition>& transitions,
		rhi::CommandList& commandList) {
		if (transitions.empty()) return;

		rhi::helpers::OwnedBarrierBatch batch;
		batch.textures.reserve(transitions.size());
		batch.buffers.reserve(transitions.size());

		for (auto& t : transitions) {
			if (t.pResource->HasLayout()) {
				// Texture barrier
				auto resolvedRange = ResolveRangeSpec(
					t.range, t.pResource->GetMipLevels(), t.pResource->GetArraySize());

				rhi::TextureBarrier tb{};
				tb.beforeAccess = t.prevAccessType;
				tb.afterAccess  = t.newAccessType;
				tb.beforeLayout = t.prevLayout;
				tb.afterLayout  = t.newLayout;
				tb.beforeSync   = t.prevSyncState;
				tb.afterSync    = t.newSyncState;
				tb.discard      = t.discard;
				tb.range = { resolvedRange.firstMip, resolvedRange.mipCount,
				             resolvedRange.firstSlice, resolvedRange.sliceCount };
				tb.texture = t.pResource->GetAPIResource().GetHandle();
				batch.textures.push_back(tb);
			} else {
				// Buffer barrier
				rhi::BufferBarrier bb{};
				bb.buffer       = t.pResource->GetAPIResource().GetHandle();
				bb.offset       = 0;
				bb.size         = UINT64_MAX;
				bb.beforeSync   = t.prevSyncState;
				bb.afterSync    = t.newSyncState;
				bb.beforeAccess = t.prevAccessType;
				bb.afterAccess  = t.newAccessType;
				batch.buffers.push_back(bb);
			}
		}

		if (!batch.Empty()) {
			commandList.Barriers(batch.View());
		}
	}

	// Signal external fences on the queue. Must be called AFTER the command list
	// containing the pass work has been flushed (submitted) so the signals fire
	// after the GPU work they depend on.
	void SignalExternalFences(
		rhi::Queue& queue,
		QueueKind queueKind,
		rhi::Timeline* slotFence,
		std::vector<PassReturn>& externalFences) {
		if (externalFences.empty()) return;
		for (auto& fr : externalFences) {
			if (!fr.fence.has_value()) {
				spdlog::warn("Pass returned an external fence without a value. This should not happen.");
			}
            else {
				if (fr.fenceValue == 0) {
					auto h = fr.fence.value().GetHandle();
					spdlog::error(
						"SignalExternalFences: pass returned fence (index={}, gen={}) "
						"with value 0 - this will violate monotonic signal ordering. "
						"Skipping signal.",
						h.index, h.generation);
					continue;
				}
#if BUILD_TYPE == BUILD_TYPE_DEBUG
				// Detect external fences that accidentally use the queue's own fence
				// timeline. This would cause signal tracking to become stale,
				// leading to out-of-order signals. The pass must use a dedicated timeline.
				if (slotFence) {
					auto a = fr.fence.value().GetHandle();
					auto b = slotFence->GetHandle();
					if (a.index == b.index && a.generation == b.generation) {
						spdlog::error(
							"External fence from pass uses the queue's own fence timeline! "
							"This will cause signal tracking to become stale and produce "
							"out-of-order signals. The pass must use a dedicated timeline.");
						throw std::runtime_error("External fence aliases queue fence timeline!");
					}
				}
#endif
				spdlog::debug(
					"SignalExternalFences: queue={} signaling timeline(idx={}, gen={}) value={}",
					static_cast<int>(queueKind),
					fr.fence.value().GetHandle().index,
					fr.fence.value().GetHandle().generation,
					fr.fenceValue);
				queue.Signal({ fr.fence.value().GetHandle(), fr.fenceValue });
			}
		}
		externalFences.clear();
	}

	// ExecuteQueueBatch: unified per-queue-per-batch execution using pre-allocated
	// command lists from the execution schedule.
	struct ExecuteQueueBatchArgs {
		QueueBatchSchedule& sched;
		RenderGraph::PassBatch& batch;
		size_t batchIndex;
		QueueKind queue;
		size_t queueSlot;
		rhi::Queue& rhiQueue;
		rhi::Timeline& fenceTimeline;
		CommandListPool& pool;
		UINT64 fenceOffset;             // always 0 currently
		PassExecutionContext& context;
		rg::runtime::IStatisticsService* statisticsService;
		std::vector<PassReturn>& outExternalFences;
		UINT64& lastSignaledOnTimeline;
	};

	void ExecuteQueueBatch(
		ExecuteQueueBatchArgs& args,
		auto&& WaitOnSlot)
	{
		auto& sched    = args.sched;
		auto& batch    = args.batch;
		auto  queue    = args.queue;
		auto  qi       = args.queueSlot;
		auto& rhiQueue = args.rhiQueue;
		auto& pool     = args.pool;
		auto  fenceOffset = args.fenceOffset;

		uint8_t clIndex = 0; // index into preallocatedCLs

		// Waits: BeforeTransitions
		for (size_t srcIndex = 0; srcIndex < batch.QueueCount(); ++srcIndex) {
			if (!batch.HasQueueWait(RenderGraph::BatchWaitPhase::BeforeTransitions, qi, srcIndex))
				continue;
			UINT64 val = fenceOffset + batch.GetQueueWaitFenceValue(
				RenderGraph::BatchWaitPhase::BeforeTransitions, qi, srcIndex);
			WaitOnSlot(qi, srcIndex, val);
		}

		// Open first CL and record pre-transitions
		auto& cl0 = sched.preallocatedCLs[clIndex];
		rhi::CommandList commandList = cl0.list.Get();

		auto& preTransitions = batch.Transitions(qi, RenderGraph::BatchTransitionPhase::BeforePasses);
		ExecuteTransitions(preTransitions, /*crm=*/nullptr, queue, commandList);

		// Waits: BeforeExecution
		for (size_t srcIndex = 0; srcIndex < batch.QueueCount(); ++srcIndex) {
			if (!batch.HasQueueWait(RenderGraph::BatchWaitPhase::BeforeExecution, qi, srcIndex))
				continue;
			UINT64 val = fenceOffset + batch.GetQueueWaitFenceValue(
				RenderGraph::BatchWaitPhase::BeforeExecution, qi, srcIndex);
			WaitOnSlot(qi, srcIndex, val);
		}

		// Split after transitions if needed
		if (sched.splitAfterTransitions) {
			UINT64 signalValue = fenceOffset + batch.GetQueueSignalFenceValue(
				RenderGraph::BatchSignalPhase::AfterTransitions, qi);
			commandList.End();
			rhiQueue.Submit({ &commandList, 1 }, {});
			rhiQueue.Signal({ args.fenceTimeline.GetHandle(), signalValue });
			args.lastSignaledOnTimeline = std::max(args.lastSignaledOnTimeline, signalValue);
			pool.Recycle(std::move(cl0), signalValue);

			++clIndex;
			commandList = sched.preallocatedCLs[clIndex].list.Get();
		}

		// Record all passes into the current CL
		args.context.commandList = commandList;

		auto executeOne = [&](auto& pr) {
			if (!pr.pass->IsInvalidated())
				return;
			rhi::debug::Scope scope(commandList, rhi::colors::Mint, pr.name.c_str());
			const bool hasStatistics = args.statisticsService && pr.statisticsIndex >= 0;
			const auto cpuStart = std::chrono::steady_clock::now();
			if (hasStatistics)
				args.statisticsService->BeginQuery(pr.statisticsIndex, args.context.frameIndex, rhiQueue, commandList);
			if ((pr.run & PassRunMask::Immediate) != PassRunMask::None)
				rg::imm::Replay(pr.immediateBytecode, commandList);
			pr.immediateKeepAlive.reset();
			if ((pr.run & PassRunMask::Retained) != PassRunMask::None) {
				auto passReturn = pr.pass->Execute(args.context);
				if (passReturn.fence)
					args.outExternalFences.push_back(passReturn);
			}
			if (hasStatistics)
				args.statisticsService->EndQuery(pr.statisticsIndex, args.context.frameIndex, rhiQueue, commandList);
			if (hasStatistics) {
				args.statisticsService->RecordCpuExecuteTime(
					static_cast<unsigned>(pr.statisticsIndex),
					std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuStart).count());
			}
		};

		for (auto& passVariant : batch.Passes(qi)) {
			std::visit([&](auto& passEntry) { executeOne(passEntry); }, passVariant);
		}

		if (args.statisticsService)
			args.statisticsService->ResolveQueries(args.context.frameIndex, rhiQueue, commandList);

		// Split after execution if needed
		if (sched.splitAfterExecution) {
			UINT64 signalValue = fenceOffset + batch.GetQueueSignalFenceValue(
				RenderGraph::BatchSignalPhase::AfterExecution, qi);
			commandList.End();
			rhiQueue.Submit({ &commandList, 1 }, {});
			rhiQueue.Signal({ args.fenceTimeline.GetHandle(), signalValue });
			args.lastSignaledOnTimeline = std::max(args.lastSignaledOnTimeline, signalValue);
			pool.Recycle(std::move(sched.preallocatedCLs[clIndex]), signalValue);

			++clIndex;
			commandList = sched.preallocatedCLs[clIndex].list.Get();
			args.context.commandList = commandList;
		}

		// Waits: BeforeAfterPasses
		for (size_t srcIndex = 0; srcIndex < batch.QueueCount(); ++srcIndex) {
			if (!batch.HasQueueWait(RenderGraph::BatchWaitPhase::BeforeAfterPasses, qi, srcIndex))
				continue;
			UINT64 val = fenceOffset + batch.GetQueueWaitFenceValue(
				RenderGraph::BatchWaitPhase::BeforeAfterPasses, qi, srcIndex);
			WaitOnSlot(qi, srcIndex, val);
		}

		// Record post-transitions
		auto& postTransitions = batch.Transitions(qi, RenderGraph::BatchTransitionPhase::AfterPasses);
		if (!postTransitions.empty())
			ExecuteTransitions(postTransitions, /*crm=*/nullptr, queue, commandList);

		// Final submit + optional AfterCompletion signal
		{
			commandList.End();
			rhiQueue.Submit({ &commandList, 1 }, {});

			UINT64 recycleFence;
			if (sched.signalAfterCompletion) {
				recycleFence = fenceOffset + batch.GetQueueSignalFenceValue(
					RenderGraph::BatchSignalPhase::AfterCompletion, qi);
			} else {
				recycleFence = ++args.lastSignaledOnTimeline;
			}
			if (recycleFence == 0) {
				spdlog::error("ExecuteQueueBatch: recycleFence is 0 for batch {} slot {} "
					"(signalAfterCompletion={}, fenceOffset={}, fenceValue={}). "
					"Falling back to monotonic signal.",
					args.batchIndex, qi, sched.signalAfterCompletion, fenceOffset,
					batch.GetQueueSignalFenceValue(
						RenderGraph::BatchSignalPhase::AfterCompletion, qi));
				recycleFence = ++args.lastSignaledOnTimeline;
			}
			rhiQueue.Signal({ args.fenceTimeline.GetHandle(), recycleFence });
			args.lastSignaledOnTimeline = std::max(args.lastSignaledOnTimeline, recycleFence);
			pool.Recycle(std::move(sched.preallocatedCLs[clIndex]), recycleFence);
		}
	}

	// RecordQueueBatch: records barrier + pass commands into pre-allocated
	// CLs and calls End() on each.  Does NOT Submit, Signal, Wait, or
	// Recycle.  Safe to call from a worker thread.
	struct RecordQueueBatchArgs {
		QueueBatchSchedule& sched;
		RenderGraph::PassBatch& batch;
		QueueKind queue;
		size_t queueSlot;
		rhi::Queue& rhiQueue;           // needed for statistics Begin/EndQuery
		PassExecutionContext context;    // COPY: each task gets its own
		rg::runtime::IStatisticsService* statisticsService;
	};

	void RecordQueueBatch(RecordQueueBatchArgs& args) {
		auto& sched = args.sched;
		auto& batch = args.batch;
		auto  queue = args.queue;
		auto  qi    = args.queueSlot;

		uint8_t clIndex = 0;
		rhi::CommandList commandList = sched.preallocatedCLs[clIndex].list.Get();

		// Record pre-transitions
		auto& preTransitions = batch.Transitions(qi, RenderGraph::BatchTransitionPhase::BeforePasses);
		RecordTransitionBarriers(preTransitions, commandList);

		// Split after transitions?
		if (sched.splitAfterTransitions) {
			commandList.End();
			++clIndex;
			commandList = sched.preallocatedCLs[clIndex].list.Get();
		}

		// Record all passes.
		args.context.commandList = commandList;

		auto executeOne = [&](auto& pr) {
			if (!pr.pass->IsInvalidated())
				return;
			rhi::debug::Scope scope(commandList, rhi::colors::Mint, pr.name.c_str());
			const bool hasStatistics = args.statisticsService && pr.statisticsIndex >= 0;
			const auto cpuStart = std::chrono::steady_clock::now();
			if (hasStatistics)
				args.statisticsService->BeginQuery(pr.statisticsIndex, args.context.frameIndex, args.rhiQueue, commandList, sched.queryRecordingContext);
			if ((pr.run & PassRunMask::Immediate) != PassRunMask::None)
				rg::imm::Replay(pr.immediateBytecode, commandList);
			pr.immediateKeepAlive.reset();
			if ((pr.run & PassRunMask::Retained) != PassRunMask::None) {
				auto passReturn = pr.pass->Execute(args.context);
				if (passReturn.fence)
					sched.externalFences.push_back(passReturn);
			}
			if (hasStatistics)
				args.statisticsService->EndQuery(pr.statisticsIndex, args.context.frameIndex, args.rhiQueue, commandList, sched.queryRecordingContext);
			if (hasStatistics) {
				args.statisticsService->RecordCpuExecuteTime(
					static_cast<unsigned>(pr.statisticsIndex),
					std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuStart).count());
			}
		};

		for (auto& passVariant : batch.Passes(qi)) {
			std::visit([&](auto& passEntry) { executeOne(passEntry); }, passVariant);
		}

		if (args.statisticsService)
			args.statisticsService->ResolveQueries(args.context.frameIndex, args.rhiQueue, commandList, sched.queryRecordingContext);

		// Split after execution?
		if (sched.splitAfterExecution) {
			commandList.End();
			++clIndex;
			commandList = sched.preallocatedCLs[clIndex].list.Get();
		}

		// Record post-transitions (barriers only).
		auto& postTransitions = batch.Transitions(qi, RenderGraph::BatchTransitionPhase::AfterPasses);
		if (!postTransitions.empty())
			RecordTransitionBarriers(postTransitions, commandList);

		// End the last CL.
		commandList.End();
	}

	// -----------------------------------------------------------------------
	// SubmitQueueBatch: submits pre-recorded CLs with the correct wait /
	// signal / recycle ordering.  Must be called on the main thread.
	// CLs must already have had End() called (by RecordQueueBatch).
	// -----------------------------------------------------------------------
	struct SubmitQueueBatchArgs {
		QueueBatchSchedule& sched;
		RenderGraph::PassBatch& batch;
		QueueKind queue;
		size_t queueSlot;
		rhi::Queue& rhiQueue;
		rhi::Timeline& fenceTimeline;
		CommandListPool& pool;
		UINT64 fenceOffset;
		UINT64& lastSignaledOnTimeline;
	};

	void SubmitQueueBatch(
		SubmitQueueBatchArgs& args,
		auto&& WaitOnSlot)
	{
		auto& sched      = args.sched;
		auto& batch      = args.batch;
		auto  queue      = args.queue;
		auto  qi         = args.queueSlot;
		auto& rhiQueue   = args.rhiQueue;
		auto  fenceOffset = args.fenceOffset;

		uint8_t clIndex = 0;

		// Waits: BeforeTransitions + BeforeExecution
		for (size_t srcIndex = 0; srcIndex < batch.QueueCount(); ++srcIndex) {
			if (batch.HasQueueWait(RenderGraph::BatchWaitPhase::BeforeTransitions, qi, srcIndex)) {
				UINT64 val = fenceOffset + batch.GetQueueWaitFenceValue(
					RenderGraph::BatchWaitPhase::BeforeTransitions, qi, srcIndex);
				WaitOnSlot(qi, srcIndex, val);
			}
		}
		for (size_t srcIndex = 0; srcIndex < batch.QueueCount(); ++srcIndex) {
			if (batch.HasQueueWait(RenderGraph::BatchWaitPhase::BeforeExecution, qi, srcIndex)) {
				UINT64 val = fenceOffset + batch.GetQueueWaitFenceValue(
					RenderGraph::BatchWaitPhase::BeforeExecution, qi, srcIndex);
				WaitOnSlot(qi, srcIndex, val);
			}
		}

		// Submit + signal for the transitions CL if it was split out.
		if (sched.splitAfterTransitions) {
			rhi::CommandList cl = sched.preallocatedCLs[clIndex].list.Get();
			rhiQueue.Submit({ &cl, 1 }, {});
			UINT64 signalValue = fenceOffset + batch.GetQueueSignalFenceValue(
				RenderGraph::BatchSignalPhase::AfterTransitions, qi);
			rhiQueue.Signal({ args.fenceTimeline.GetHandle(), signalValue });
			args.lastSignaledOnTimeline = std::max(args.lastSignaledOnTimeline, signalValue);
			args.pool.Recycle(std::move(sched.preallocatedCLs[clIndex]), signalValue);
			++clIndex;
		}

		// Submit + signal for the passes CL if it was split out.
		if (sched.splitAfterExecution) {
			rhi::CommandList cl = sched.preallocatedCLs[clIndex].list.Get();
			rhiQueue.Submit({ &cl, 1 }, {});
			UINT64 signalValue = fenceOffset + batch.GetQueueSignalFenceValue(
				RenderGraph::BatchSignalPhase::AfterExecution, qi);
			rhiQueue.Signal({ args.fenceTimeline.GetHandle(), signalValue });
			args.lastSignaledOnTimeline = std::max(args.lastSignaledOnTimeline, signalValue);
			args.pool.Recycle(std::move(sched.preallocatedCLs[clIndex]), signalValue);
			++clIndex;
		}

		// Waits: BeforeAfterPasses
		for (size_t srcIndex = 0; srcIndex < batch.QueueCount(); ++srcIndex) {
			if (batch.HasQueueWait(RenderGraph::BatchWaitPhase::BeforeAfterPasses, qi, srcIndex)) {
				UINT64 val = fenceOffset + batch.GetQueueWaitFenceValue(
					RenderGraph::BatchWaitPhase::BeforeAfterPasses, qi, srcIndex);
				WaitOnSlot(qi, srcIndex, val);
			}
		}

		// Submit the final CL and signal for recycle.
		{
			rhi::CommandList cl = sched.preallocatedCLs[clIndex].list.Get();
			rhiQueue.Submit({ &cl, 1 }, {});

			UINT64 recycleFence;
			if (sched.signalAfterCompletion) {
				recycleFence = fenceOffset + batch.GetQueueSignalFenceValue(
					RenderGraph::BatchSignalPhase::AfterCompletion, qi);
			} else {
				recycleFence = ++args.lastSignaledOnTimeline;
			}
			if (recycleFence == 0) {
				spdlog::error(
					"SubmitQueueBatch: recycleFence is 0 (signalAfterCompletion={}, "
					"fenceOffset={}, fenceValue={}). Falling back to monotonic signal.",
					sched.signalAfterCompletion, fenceOffset,
					batch.GetQueueSignalFenceValue(
						RenderGraph::BatchSignalPhase::AfterCompletion, qi));
				recycleFence = ++args.lastSignaledOnTimeline;
			}
			rhiQueue.Signal({ args.fenceTimeline.GetHandle(), recycleFence });
			args.lastSignaledOnTimeline = std::max(args.lastSignaledOnTimeline, recycleFence);
			args.pool.Recycle(std::move(sched.preallocatedCLs[clIndex]), recycleFence);
		}
	}

} // namespace

void RenderGraph::BuildExecutionSchedule() {
	auto& schedule = m_executionSchedule;
	const size_t qc = m_queueRegistry.SlotCount();
	schedule.batches.clear();
	schedule.batches.reserve(batches.size());
	for (size_t i = 0; i < batches.size(); ++i) {
		schedule.batches.emplace_back(qc);
	}

	for (size_t bi = 0; bi < batches.size(); ++bi) {
		auto& batch = batches[bi];
		auto& batchSched = schedule.batches[bi];

		for (size_t qi = 0; qi < batchSched.queues.size(); ++qi) {
			auto& qs = batchSched.queues[qi];

			bool hasPre  = batch.HasTransitions(qi, BatchTransitionPhase::BeforePasses);
			bool hasPost = batch.HasTransitions(qi, BatchTransitionPhase::AfterPasses);
			bool hasPasses = batch.HasPasses(qi);
			qs.active = hasPre || hasPasses || hasPost;

			if (!qs.active) {
				qs.numCLs = 0;
				continue;
			}

			// Batch 0 is a dummy sentinel with all-zero fence values.
			// It should never be active. If it is, something upstream
			// incorrectly added transitions or passes to it.
			if (bi == 0) {
				spdlog::error(
					"BuildExecutionSchedule: batch 0 (sentinel) is unexpectedly "
					"active on queue {} (hasPre={}, hasPasses={}, hasPost={}). "
					"This will produce a zero-value fence signal. Forcing inactive.",
					qi, hasPre, hasPasses, hasPost);
				qs.active = false;
				qs.numCLs = 0;
				continue;
			}

			qs.splitAfterTransitions =
				batch.HasQueueSignal(BatchSignalPhase::AfterTransitions, qi);
			qs.splitAfterExecution =
				batch.HasQueueSignal(BatchSignalPhase::AfterExecution, qi);
			qs.signalAfterCompletion =
				batch.HasQueueSignal(BatchSignalPhase::AfterCompletion, qi);

			qs.numCLs = 1
				+ static_cast<uint8_t>(qs.splitAfterTransitions)
				+ static_cast<uint8_t>(qs.splitAfterExecution);
		}
	}
}

void RenderGraph::Execute(PassExecutionContext& context) {
	ValidateCompiledResourceGenerations();

	const bool heavyDebug = m_getHeavyDebug ? m_getHeavyDebug() : false;
	auto& manager = DeviceManager::GetInstance();
	const size_t slotCount = m_queueRegistry.SlotCount();

	// Create CRM from the primary queue slots in the registry.
	CommandRecordingManager::Init init{
		.graphicsQ = &manager.GetGraphicsQueue(),
		.graphicsF = &m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(0)),
		.graphicsPool = m_queueRegistry.GetPool(static_cast<QueueSlotIndex>(0)),

		.computeQ = &manager.GetComputeQueue(),
		.computeF = &m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(1)),
		.computePool = m_queueRegistry.GetPool(static_cast<QueueSlotIndex>(1)),

		.copyQ = &manager.GetCopyQueue(),
		.copyF = &m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(2)),
		.copyPool = m_queueRegistry.GetPool(static_cast<QueueSlotIndex>(2)),
	};

	m_pCommandRecordingManager = std::make_unique<CommandRecordingManager>(init);
	auto crm = m_pCommandRecordingManager.get();

	// Registry-based queue/fence/pool resolution — no aliasing.
	auto SlotQueue = [&](size_t qi) -> rhi::Queue {
		return m_queueRegistry.GetQueue(static_cast<QueueSlotIndex>(qi));
	};
	auto SlotFence = [&](size_t qi) -> rhi::Timeline& {
		return m_queueRegistry.GetFence(static_cast<QueueSlotIndex>(qi));
	};
	auto SlotPool = [&](size_t qi) -> CommandListPool* {
		return m_queueRegistry.GetPool(static_cast<QueueSlotIndex>(qi));
	};

	// Wait: every queue is distinct, so waits always execute.
	auto WaitOnSlot = [&](size_t dstSlot, size_t srcSlot, UINT64 absoluteFenceValue) {
		if (dstSlot == srcSlot) return;
		auto dstQ = SlotQueue(dstSlot);
		dstQ.Wait({ SlotFence(srcSlot).GetHandle(), absoluteFenceValue });
	};

	// Frame-start waits from previous frame's last-producer tracking.
	for (size_t dstIndex = 0; dstIndex < slotCount; ++dstIndex) {
		for (size_t srcIndex = 0; srcIndex < slotCount; ++srcIndex) {
			if (dstIndex == srcIndex) continue;
			if (dstIndex >= m_hasPendingFrameStartQueueWait.size() ||
				srcIndex >= m_hasPendingFrameStartQueueWait[dstIndex].size()) continue;
			if (!m_hasPendingFrameStartQueueWait[dstIndex][srcIndex]) continue;
			WaitOnSlot(dstIndex, srcIndex,
				m_pendingFrameStartQueueWaitFenceValue[dstIndex][srcIndex]);
		}
	}

	// Mark completion signals on batches that produced resources tracked across frames.
	for (size_t queueIndex = 0; queueIndex < slotCount; ++queueIndex) {
		if (queueIndex >= m_compiledLastProducerBatchByResourceByQueue.size()) continue;
		for (const auto& [resourceID, producerBatch] : m_compiledLastProducerBatchByResourceByQueue[queueIndex]) {
			(void)resourceID;
			if (producerBatch > 0 && producerBatch < batches.size()) {
				batches[producerBatch].MarkQueueSignal(BatchSignalPhase::AfterCompletion, queueIndex);
			}
		}
	}

	auto nextLastProducerByResourceAcrossFrames = m_lastProducerByResourceAcrossFrames;
	auto nextLastAliasPlacementProducersByPoolAcrossFrames = m_lastAliasPlacementProducersByPoolAcrossFrames;

	auto removeResourceFromLastAliasPlacementCache = [&](uint64_t resourceID) {
		for (auto& [poolID, producers] : nextLastAliasPlacementProducersByPoolAcrossFrames) {
			(void)poolID;
			producers.erase(
				std::remove_if(
					producers.begin(),
					producers.end(),
					[&](const LastAliasPlacementProducerAcrossFrames& p) {
						return p.resourceID == resourceID;
					}),
				producers.end());
		}
	};

	auto publishAliasPlacementProducer = [&](uint64_t resourceID, LastProducerAcrossFrames producer) {
		removeResourceFromLastAliasPlacementCache(resourceID);

		auto itPlacement = aliasPlacementRangesByID.find(resourceID);
		if (itPlacement == aliasPlacementRangesByID.end()) {
			return;
		}

		auto itPoolState = persistentAliasPools.find(itPlacement->second.poolID);
		if (itPoolState == persistentAliasPools.end()) {
			return;
		}

		nextLastAliasPlacementProducersByPoolAcrossFrames[itPlacement->second.poolID].push_back(
			LastAliasPlacementProducerAcrossFrames{
				.resourceID = resourceID,
				.poolID = itPlacement->second.poolID,
				.poolGeneration = itPoolState->second.generation,
				.startByte = itPlacement->second.startByte,
				.endByte = itPlacement->second.endByte,
				.producer = producer,
			});
	};

	auto* statisticsService = m_statisticsService.get();

	// Build the execution schedule and pre-allocate command lists.
	BuildExecutionSchedule();

#if BUILD_TYPE == BUILD_TYPE_DEBUG
	// ---- Post-schedule validation: detect signals that will never fire ----
	// A signal is "live" only when the queue is active in that batch.
	// A wait that references a dead signal will deadlock the GPU.
	{
		// 1. Collect the set of fence values that will actually be signaled.
		std::vector<std::unordered_set<UINT64>> liveSignalValues(slotCount);
		// Also track the highest value each queue will signal this frame so
		// we can verify frame-start waits from the previous frame.
		std::vector<UINT64> highestLiveSignal(slotCount, 0);

		for (size_t bi = 0; bi < batches.size(); ++bi) {
			auto& batch = batches[bi];
			auto& batchSched = m_executionSchedule.batches[bi];

			for (size_t qi = 0; qi < std::min(batchSched.queues.size(), slotCount); ++qi) {
				auto& qs = batchSched.queues[qi];
				if (!qs.active) {
					// Check: does this inactive queue have signals marked?
					for (size_t sp = 0; sp < PassBatch::kSignalPhaseCount; ++sp) {
						if (batch.HasQueueSignal(static_cast<BatchSignalPhase>(sp), qi)) {
							spdlog::error(
								"SIGNAL ON INACTIVE QUEUE: batch {} slot {} has {} signal "
								"enabled (fence={}) but queue is inactive (no passes/transitions). "
								"This signal will never fire on the GPU!",
								bi, qi,
								sp == 0 ? "AfterTransitions" : sp == 1 ? "AfterExecution" : "AfterCompletion",
								batch.GetQueueSignalFenceValue(static_cast<BatchSignalPhase>(sp), qi));
						}
					}
					continue;
				}

				// Active queue: record which signals will actually fire.
				if (qs.splitAfterTransitions) {
					UINT64 v = batch.GetQueueSignalFenceValue(BatchSignalPhase::AfterTransitions, qi);
					liveSignalValues[qi].insert(v);
					highestLiveSignal[qi] = std::max(highestLiveSignal[qi], v);
				}
				if (qs.splitAfterExecution) {
					UINT64 v = batch.GetQueueSignalFenceValue(BatchSignalPhase::AfterExecution, qi);
					liveSignalValues[qi].insert(v);
					highestLiveSignal[qi] = std::max(highestLiveSignal[qi], v);
				}
				// The final submit always signals. If signalAfterCompletion is true,
				// it signals the pre-assigned AfterCompletion value. Otherwise, it
				// signals a monotonic value that won't match any pre-assigned value.
				if (qs.signalAfterCompletion) {
					UINT64 v = batch.GetQueueSignalFenceValue(BatchSignalPhase::AfterCompletion, qi);
					liveSignalValues[qi].insert(v);
					highestLiveSignal[qi] = std::max(highestLiveSignal[qi], v);
				}
			}
		}

		// 2. Validate within-frame waits: every wait on an active queue must
		//    reference a live signal or a value already completed (cross-frame).
		bool foundDeadWait = false;
		for (size_t bi = 0; bi < batches.size(); ++bi) {
			auto& batch = batches[bi];
			auto& batchSched = m_executionSchedule.batches[bi];

			for (size_t qi = 0; qi < std::min(batchSched.queues.size(), slotCount); ++qi) {
				if (!batchSched.queues[qi].active) continue;

				for (size_t wp = 0; wp < PassBatch::kWaitPhaseCount; ++wp) {
					auto waitPhase = static_cast<BatchWaitPhase>(wp);
					for (size_t src = 0; src < slotCount; ++src) {
						if (qi == src) continue;
						if (!batch.HasQueueWait(waitPhase, qi, src)) continue;

						UINT64 fv = batch.GetQueueWaitFenceValue(waitPhase, qi, src);

						// Check if this is a live signal from the current frame.
						bool isLive = liveSignalValues[src].count(fv) > 0;

						// Check if already completed from a previous frame.
						UINT64 completedValue = SlotFence(src).GetCompletedValue();
						bool isAlreadyCompleted = (completedValue >= fv);

						if (!isLive && !isAlreadyCompleted) {
							spdlog::error(
								"DEADLOCK: batch {} active slot {} waits at phase {} "
								"on slot {} fence={}, but that value is not a live signal "
								"(slot {} may be inactive) and not already completed "
								"(completed={}). GPU WILL HANG.",
								bi, qi,
								wp == 0 ? "BeforeTransitions" : wp == 1 ? "BeforeExecution" : "BeforeAfterPasses",
								src, fv, src, completedValue);
							foundDeadWait = true;
						}
					}
				}
			}
		}

		// 3. Validate frame-start waits: these reference previous-frame values.
		for (size_t dst = 0; dst < slotCount; ++dst) {
			if (dst >= m_hasPendingFrameStartQueueWait.size()) continue;
			for (size_t src = 0; src < slotCount; ++src) {
				if (dst == src) continue;
				if (src >= m_hasPendingFrameStartQueueWait[dst].size()) continue;
				if (!m_hasPendingFrameStartQueueWait[dst][src]) continue;

				UINT64 fv = m_pendingFrameStartQueueWaitFenceValue[dst][src];
				UINT64 completedValue = SlotFence(src).GetCompletedValue();

				if (completedValue < fv) {
					spdlog::warn(
						"Frame-start wait: slot {} waiting on slot {} fence={}, "
						"currently completed={}. Delta={}. "
						"This wait will block until the previous frame's queue "
						"signals this value.",
						dst, src, fv, completedValue, fv - completedValue);
				}
			}
		}

		// 4. Log cross-frame producer summary for diagnostics.
		for (size_t qi = 0; qi < slotCount; ++qi) {
			if (qi >= m_compiledLastProducerBatchByResourceByQueue.size()) continue;
			size_t count = m_compiledLastProducerBatchByResourceByQueue[qi].size();
			if (count > 0) {
				spdlog::debug(
					"Cross-frame producer tracking: slot {} has {} resources tracked",
					qi, count);
			}
		}

		if (foundDeadWait) {
			// Dump the full active/inactive map for debugging.
			for (size_t bi = 0; bi < batches.size(); ++bi) {
				auto& batchSched = m_executionSchedule.batches[bi];
				std::string activeStr;
				for (size_t qi = 0; qi < std::min(batchSched.queues.size(), slotCount); ++qi) {
					if (!activeStr.empty()) activeStr += ", ";
					activeStr += "slot" + std::to_string(qi) + "=" +
						(batchSched.queues[qi].active ? "ACTIVE" : "inactive");
				}
				spdlog::error("  batch {} queue activity: [{}]", bi, activeStr);
			}
			__debugbreak();
		}
	}
#endif

	// Pre-allocate all CLs from the main thread using registry pools.
	for (size_t bi = 0; bi < m_executionSchedule.batches.size(); ++bi) {
		auto& batchSched = m_executionSchedule.batches[bi];
		for (size_t qi = 0; qi < batchSched.queues.size(); ++qi) {
			auto& qs = batchSched.queues[qi];
			for (uint8_t ci = 0; ci < qs.numCLs; ++ci) {
				qs.preallocatedCLs[ci] = SlotPool(qi)->Request();
			}
		}
	}

	// Per-slot signal tracking for monotonic recycle signals.
	std::vector<UINT64> lastSignaledPerSlot(slotCount);
	for (size_t qi = 0; qi < slotCount; ++qi) {
		lastSignaledPerSlot[qi] = SlotFence(qi).GetCompletedValue();
	}

	// Execution, two paths: heavyDebug (serial) or normal (parallel).
	if (heavyDebug) {
		// ---- Serial path: record + submit + drain per batch. ----
		unsigned int batchIndex = 0;
		for (size_t bi = 0; bi < batches.size(); ++bi) {
			auto& batch = batches[bi];
			auto& batchSched = m_executionSchedule.batches[bi];

			// Execute each queue slot in order.
			std::vector<std::vector<PassReturn>> slotExternalFences(slotCount);
			for (size_t qi = 0; qi < slotCount; ++qi) {
				auto& qs = batchSched.queues[qi];
				if (!qs.active) continue;
				auto rhiQ = SlotQueue(qi);
				ExecuteQueueBatchArgs args{
					.sched = qs,
					.batch = batch,
					.batchIndex = bi,
					.queue = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(qi)),
					.queueSlot = qi,
					.rhiQueue = rhiQ,
					.fenceTimeline = SlotFence(qi),
					.pool = *SlotPool(qi),
					.fenceOffset = 0,
					.context = context,
					.statisticsService = statisticsService,
					.outExternalFences = slotExternalFences[qi],
					.lastSignaledOnTimeline = lastSignaledPerSlot[qi],
				};
				ExecuteQueueBatch(args, WaitOnSlot);
			}

			// Signal external fences AFTER all CLs in this batch are submitted.
			for (size_t qi = 0; qi < slotCount; ++qi) {
				if (!slotExternalFences[qi].empty()) {
					auto rhiQ = SlotQueue(qi);
					SignalExternalFences(rhiQ, m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(qi)),
						&SlotFence(qi), slotExternalFences[qi]);
				}
			}

			// Drain all queues after every batch.
			for (size_t qi = 0; qi < slotCount; ++qi) {
				auto& qs = batchSched.queues[qi];
				if (!qs.active) continue;
				UINT64 highestSignal = 0;
				for (size_t sp = 0; sp < PassBatch::kSignalPhaseCount; ++sp) {
					if (batch.HasQueueSignal(static_cast<BatchSignalPhase>(sp), qi)) {
						UINT64 v = batch.GetQueueSignalFenceValue(static_cast<BatchSignalPhase>(sp), qi);
						if (v > highestSignal) highestSignal = v;
					}
				}
				if (highestSignal == 0) continue;
				auto result = SlotFence(qi).HostWait(highestSignal);
				DeviceManager::GetInstance().GetDevice().CheckDebugMessages();
				if (rhi::Failed(result)) {
					std::string passNames;
					auto collectNames = [&](size_t q) {
						for (auto& pv : batch.Passes(q)) {
							std::visit([&](auto& pr) {
								if (!passNames.empty()) passNames += ", ";
								passNames += pr.name;
							}, pv);
						}
					};
					for (size_t q = 0; q < slotCount; ++q) collectNames(q);
					spdlog::error(
						"[HeavyDebug] GPU fault after batch {} on queue slot {}. "
						"Passes in batch: [{}]",
						batchIndex, qi, passNames);
				}
			}
			++batchIndex;
		}
	} else {
		// Parallel recording path

		// Clear per-frame recording state from any previous frame.
		for (size_t bi = 0; bi < batches.size(); ++bi) {
			auto& batchSched = m_executionSchedule.batches[bi];
			for (size_t qi = 0; qi < batchSched.queues.size(); ++qi) {
				auto& qs = batchSched.queues[qi];
				qs.externalFences.clear();
				qs.queryRecordingContext.recordedIndices.clear();
				qs.queryRecordingContext.pendingRanges.clear();
			}
		}

		// Build flat task list: one entry per active (batch, queue) pair.
		struct RecordTask {
			size_t batchIndex;
			size_t queueIndex;
		};
		std::vector<RecordTask> tasks;
		tasks.reserve(batches.size() * slotCount);
		for (size_t bi = 0; bi < batches.size(); ++bi) {
			auto& batchSched = m_executionSchedule.batches[bi];
			for (size_t qi = 0; qi < batchSched.queues.size(); ++qi) {
				if (batchSched.queues[qi].active)
					tasks.push_back({bi, qi});
			}
		}

		ParallelForOptional("RecordAllBatches", tasks.size(), [&](size_t taskIdx) {
			auto& task = tasks[taskIdx];
			auto& qs = m_executionSchedule.batches[task.batchIndex].queues[task.queueIndex];
			auto rhiQ = SlotQueue(task.queueIndex);
			RecordQueueBatchArgs args{
				.sched = qs,
				.batch = batches[task.batchIndex],
				.queue = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(task.queueIndex)),
				.queueSlot = task.queueIndex,
				.rhiQueue = rhiQ,
				.context = context,
				.statisticsService = statisticsService,
			};
			RecordQueueBatch(args);
		}, true);

		// Merge per-task statistics contexts.
		if (statisticsService) {
			for (size_t bi = 0; bi < batches.size(); ++bi) {
				auto& batchSched = m_executionSchedule.batches[bi];
				for (size_t qi = 0; qi < batchSched.queues.size(); ++qi) {
					auto& qs = batchSched.queues[qi];
					if (!qs.active) continue;
					if (qs.queryRecordingContext.pendingRanges.empty()) continue;
					const auto kind = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(qi));
					statisticsService->MergePendingResolves(
						static_cast<rhi::QueueKind>(kind), context.frameIndex,
						qs.queryRecordingContext);
				}
			}
		}

		// Sequential submission on the main thread.
		for (size_t bi = 0; bi < batches.size(); ++bi) {
			auto& batch = batches[bi];
			auto& batchSched = m_executionSchedule.batches[bi];

			for (size_t qi = 0; qi < batchSched.queues.size(); ++qi) {
				auto& qs = batchSched.queues[qi];
				if (!qs.active) continue;
				auto rhiQ = SlotQueue(qi);
				SubmitQueueBatchArgs args{
					.sched = qs,
					.batch = batch,
					.queue = m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(qi)),
					.queueSlot = qi,
					.rhiQueue = rhiQ,
					.fenceTimeline = SlotFence(qi),
					.pool = *SlotPool(qi),
					.fenceOffset = 0,
					.lastSignaledOnTimeline = lastSignaledPerSlot[qi],
				};
				SubmitQueueBatch(args, WaitOnSlot);
			}

			// Signal external fences AFTER all CLs are submitted.
			for (size_t qi = 0; qi < batchSched.queues.size(); ++qi) {
				auto& qs = batchSched.queues[qi];
				if (qs.externalFences.empty()) continue;
				auto rhiQ = SlotQueue(qi);
				SignalExternalFences(rhiQ, m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(qi)),
					&SlotFence(qi), qs.externalFences);
			}
		}
	}

	// Update across-frame producer tracking (no aliasing remapping).
	// Only store fence values that were actually signaled during this frame's
	// execution. If a queue was inactive in a batch (no passes or transitions),
	// the pre-assigned fence value was never signaled on the GPU. Storing it
	// would cause next frame's frame-start waits to deadlock.
	for (size_t queueIndex = 0; queueIndex < slotCount; ++queueIndex) {
		if (queueIndex >= m_compiledLastProducerBatchByResourceByQueue.size()) continue;
		for (const auto& [resourceID, producerBatch] : m_compiledLastProducerBatchByResourceByQueue[queueIndex]) {
			if (producerBatch == 0 || producerBatch >= batches.size()) continue;

			const uint64_t fenceValue =
				batches[producerBatch].GetQueueSignalFenceValue(BatchSignalPhase::AfterCompletion, queueIndex);

			// Skip if this fence value was never actually signaled on the GPU.
			if (fenceValue > lastSignaledPerSlot[queueIndex]) {
				spdlog::warn(
					"Cross-frame producer skip: slot {} resource {} batch {} "
					"fenceValue={} > lastSignaled={}. Queue was likely inactive.",
					queueIndex, resourceID, producerBatch,
					fenceValue, lastSignaledPerSlot[queueIndex]);
				continue;
			}

			LastProducerAcrossFrames producer{
				.queueSlot = queueIndex,
				.fenceValue = fenceValue,
			};
			nextLastProducerByResourceAcrossFrames[resourceID] = producer;
			publishAliasPlacementProducer(resourceID, producer);
		}
	}

	m_lastProducerByResourceAcrossFrames = std::move(nextLastProducerByResourceAcrossFrames);
	m_lastAliasPlacementProducersByPoolAcrossFrames = std::move(nextLastAliasPlacementProducersByPoolAcrossFrames);
	DeletionManager::GetInstance().ProcessDeletions();

	// Sync CRM signal tracking with values we signaled directly.
	// Capped at primary queue count: CRM only tracks the 3 primary queues (Graphics/Compute/Copy).
	for (size_t qi = 0; qi < std::min(slotCount, static_cast<size_t>(QueueKind::Count)); ++qi) {
		UINT64 val = lastSignaledPerSlot[qi];
		if (val > 0) {
			crm->EnsureMinSignaledValue(static_cast<QueueKind>(qi), val);
		}
	}

	crm->Flush(QueueKind::Graphics, { false, 0 });
	crm->Flush(QueueKind::Compute, { false, 0 });
	crm->Flush(QueueKind::Copy, { false, 0 });
	PublishCompiledTrackerStates();
	crm->EndFrame();

	// Recycle completed command lists on registry pools.
	// CRM::EndFrame() only recycles the old member pools; the registry owns
	// separate pools that receive Recycle() calls during execution above.
	for (size_t qi = 0; qi < slotCount; ++qi) {
		auto* pool = SlotPool(qi);
		if (pool) {
			uint64_t completed = SlotFence(qi).GetCompletedValue();
			pool->RecycleCompleted(completed);
		}
	}
}

bool RenderGraph::IsNewBatchNeeded(
	const std::vector<ResourceRequirement>& reqs,
	const std::vector<std::pair<ResourceHandleAndRange, ResourceState>> passInternalTransitions,
	const std::unordered_map<uint64_t, SymbolicTracker*>& passBatchTrackers,
	const std::vector<uint64_t>& currentBatchInternallyTransitionedResources,
	const std::vector<uint64_t>& currentBatchAllResources,
	const std::vector<uint64_t>& otherQueueUAVs)
{
	auto overlapsAliasedResourceInBatch = [&](uint64_t resourceID) {
		for (uint64_t equivalentID : m_aliasingSubsystem.GetSchedulingEquivalentIDs(resourceID, aliasPlacementRangesByID)) {
			if (equivalentID == resourceID) {
				continue;
			}
			if (std::binary_search(currentBatchAllResources.begin(), currentBatchAllResources.end(), equivalentID)) {
				return true;
			}
		}
		return false;
	};

	auto overlapsAliasedTransitionInBatch = [&](uint64_t resourceID) {
		for (uint64_t equivalentID : m_aliasingSubsystem.GetSchedulingEquivalentIDs(resourceID, aliasPlacementRangesByID)) {
			if (equivalentID == resourceID) {
				continue;
			}
			if (std::binary_search(
				currentBatchInternallyTransitionedResources.begin(),
				currentBatchInternallyTransitionedResources.end(),
				equivalentID)) {
				return true;
			}
		}
		return false;
	};

	// For each internally modified resource
	for (auto const& r : passInternalTransitions) {
		auto id = r.first.resource.GetGlobalResourceID();
		// If this resource is used in the current batch, we need a new one
		if (std::binary_search(currentBatchAllResources.begin(), currentBatchAllResources.end(), id)) {
			return true;
		}
		if (overlapsAliasedResourceInBatch(id)) {
			return true;
		}
	}

	// For each subresource requirement in this pass:
	for (auto const& r : reqs) {

		uint64_t id = r.resourceHandleAndRange.resource.GetGlobalResourceID();

		// If this resource is internally modified in the current batch, we need a new one
		if (std::binary_search(currentBatchInternallyTransitionedResources.begin(), currentBatchInternallyTransitionedResources.end(), id)) {
			return true;
		}
		if (overlapsAliasedResourceInBatch(id) || overlapsAliasedTransitionInBatch(id)) {
			return true;
		}

		ResourceState wantState{ r.state.access, r.state.layout, r.state.sync };

		// Changing state?
		auto it = passBatchTrackers.find(id);
		if (it != passBatchTrackers.end()) {
			if (it->second->WouldModify(r.resourceHandleAndRange.range, wantState))
				return true;
		}
		// first-use in this batch never forces a split.

		// Reusing the same UAV in later passes of the same batch requires a UAV
		// barrier even when the logical state remains UnorderedAccess. The batch
		// model only inserts state transitions at batch boundaries, so keep each
		// same-resource UAV use in its own batch.
		if (((r.state.access & rhi::ResourceAccessType::UnorderedAccess) != 0
				|| r.state.layout == rhi::ResourceLayout::UnorderedAccess)
			&& std::binary_search(currentBatchAllResources.begin(), currentBatchAllResources.end(), id)) {
			return true;
		}

		// Cross-queue UAV hazard?
		if ((r.state.access & rhi::ResourceAccessType::UnorderedAccess)
			&& std::binary_search(otherQueueUAVs.begin(), otherQueueUAVs.end(), id))
			return true;
		if (r.state.layout == rhi::ResourceLayout::UnorderedAccess
			&& std::binary_search(otherQueueUAVs.begin(), otherQueueUAVs.end(), id))
			return true;
	}
	return false;
}

void RenderGraph::RegisterProvider(IResourceProvider* prov) {
	auto keys = prov->GetSupportedKeys();
	for (const auto& key : keys) {
		if (_providerMap.find(key) != _providerMap.end()) {
			std::string_view name = key.ToString();
			throw std::runtime_error("Resource provider already registered for key: " + std::string(name));
		}
		_providerMap[key] = prov;
	}
	_providers.push_back(prov);

	for (const auto& key : prov->GetSupportedKeys()) {
		auto resource = prov->ProvideResource(key);
		if (resource) {
			RegisterResource(key, resource, prov);
		}
		else {
			spdlog::warn("Provider returned null for advertised key: {}", key.ToString());
		}
	}

	// Register resolvers from this provider
	for (const auto& key : prov->GetSupportedResolverKeys()) {
		if (const auto resolver = prov->ProvideResolver(key); resolver) {
			RegisterResolver(key, resolver);
		}
		else {
			spdlog::warn("Provider returned null resolver for advertised key: {}", key.ToString());
		}
	}
}

void RenderGraph::RegisterResolver(ResourceIdentifier id, const std::shared_ptr<IResourceResolver>& resolver) {
	if (_resolverMap.contains(id)) {
		throw std::runtime_error("Resolver already registered for key: " + id.ToString());
	}
	// Resolve it and register its resources
	for (const auto& resource : resolver->Resolve()) {
		if (resource) {
			resourcesByID[resource->GetGlobalResourceID()] = resource;
			// Anonymous registration
			_registry.RegisterAnonymous(resource);
		}
	}
	_resolverMap[id] = resolver;
	_registry.RegisterResolver(id, resolver);
}

std::shared_ptr<IResourceResolver> RenderGraph::RequestResolver(ResourceIdentifier const& rid, bool allowFailure) {
	if (auto it = _resolverMap.find(rid); it != _resolverMap.end()) {
		return it->second;
	}

	if (allowFailure) return nullptr;
	throw std::runtime_error("No resolver registered for key: " + rid.ToString());
}

void RenderGraph::RegisterResource(ResourceIdentifier id, std::shared_ptr<Resource> resource,
	IResourceProvider* provider) {

	auto key = _registry.RegisterOrUpdate(id, resource);
	AddResource(resource);
	if (provider) {
		_providerMap[id] = provider;
	}

	// If resource can be cast to IHasMemoryMetadata, tag it with this ResouceIdentifier
	if (const auto hasMemoryMetadata = std::dynamic_pointer_cast<IHasMemoryMetadata>(resource); hasMemoryMetadata) {
		hasMemoryMetadata->ApplyMetadataComponentBundle(EntityComponentBundle().Set<ResourceIdentifier>(id));
	}
}

std::shared_ptr<Resource> RenderGraph::RequestResourcePtr(ResourceIdentifier const& rid, bool allowFailure) {
	// If it's already in our registry, return it
	auto cached = _registry.RequestShared(rid);
	if (cached) {
		return cached;
	}

	// We don't have it in our registry, check if we have a provider for it
	auto providerIt = _providerMap.find(rid);
	if (providerIt != _providerMap.end()) {
		// If we have a provider for this key, use it to provide the resource
		auto provider = providerIt->second;
		if (provider) {
			auto resource = provider->ProvideResource(rid);
			if (resource) {
				// Register the resource in our registry
				_registry.RegisterOrUpdate(rid, resource);
				AddResource(resource);
				return resource;
			}
			else {
				throw std::runtime_error("Provider returned null for key: " + rid.ToString());
			}
		}
	}

	// No provider registered for this key
	if (allowFailure) {
		// If we are allowed to fail, return nullptr
		return nullptr;
	}
	throw std::runtime_error("No resource provider registered for key: " + rid.ToString());
}

ResourceRegistry::RegistryHandle RenderGraph::RequestResourceHandle(ResourceIdentifier const& rid, bool allowFailure) {
	// If it's already in our registry, return it
	auto cached = _registry.GetHandleFor(rid);
	if (cached.has_value()) {
		return cached.value();
	}

	// We don't have it in our registry, check if we have a provider for it
	auto providerIt = _providerMap.find(rid);
	if (providerIt != _providerMap.end()) {
		// If we have a provider for this key, use it to provide the resource
		auto provider = providerIt->second;
		if (provider) {
			auto resource = provider->ProvideResource(rid);
			if (resource) {
				// Register the resource in our registry
				_registry.RegisterOrUpdate(rid, resource);
				AddResource(resource);
				return _registry.GetHandleFor(rid).value();
			}
			else {
				throw std::runtime_error("Provider returned null for key: " + rid.ToString());
			}
		}
	}

	// No provider registered for this key
	if (allowFailure) {
		// If we are allowed to fail, return nullptr
		return {};
	}
	throw std::runtime_error("No resource provider registered for key: " + rid.ToString());
}

ResourceRegistry::RegistryHandle RenderGraph::RequestResourceHandle(Resource* const& pResource, bool allowFailure) {
	if (!pResource) {
		if (allowFailure) {
			return {};
		}
		throw std::runtime_error("Null resource pointer passed to RequestResourceHandle(Resource*)");
	}

	auto pinTransientForFrame = [&]() {
		const uint64_t resourceID = pResource->GetGlobalResourceID();
		if (resourcesByID.contains(resourceID) || m_transientFrameResourcesByID.contains(resourceID)) {
			return;
		}

		TrackTransientFrameResource(pResource);
	};

	// If it's already in our registry, return it
	auto cached = _registry.GetHandleFor(pResource);
	if (cached.has_value()) {
		if (_registry.IsValid(cached.value())) {
			pinTransientForFrame();
			return cached.value();
		}

		//spdlog::warn(
		//	"Stale cached registry handle for resource '{}' (id={}) detected; reminting anonymous handle.",
		//	pResource ? pResource->GetName() : std::string("<null>"),
		//	pResource ? pResource->GetGlobalResourceID() : 0ull);

		// Fall through and remint a fresh handle for this live resource pointer.
		// This can happen if a resource was replaced but an old reverse-map entry remained.
		const auto reminted = _registry.RegisterAnonymousWeak(pResource->weak_from_this());
		pinTransientForFrame();
		return reminted;
	}

	if (allowFailure) {
		return {};
	}

	// Register anonymous resource
	const auto handle = _registry.RegisterAnonymousWeak(pResource->weak_from_this());
	pinTransientForFrame();

	return handle;
}


ComputePassBuilder& RenderGraph::BuildComputePass(std::string const& name) {
	if (auto it = m_passBuildersByName.find(name); it != m_passBuildersByName.end()) {
		if (m_passNamesSeenThisReset.contains(name)) {
			throw std::runtime_error("Pass names must be unique.");
		}
		if (it->second->Kind() != PassBuilderKind::Compute) {
			throw std::runtime_error("Pass builder name collision (render/compute/copy): " + name);
		}
		m_passBuilderOrder.push_back(it->second.get());
		return static_cast<ComputePassBuilder&>(*(it->second));
	}
	m_passNamesSeenThisReset.insert(name);
	auto ptr = std::unique_ptr<ComputePassBuilder>(new ComputePassBuilder(this, name));
	m_passBuilderOrder.push_back(ptr.get());
	m_passBuildersByName.emplace(name, std::move(ptr));
	return static_cast<ComputePassBuilder&>(*(m_passBuildersByName[name]));
}
RenderPassBuilder& RenderGraph::BuildRenderPass(std::string const& name) {
	if (auto it = m_passBuildersByName.find(name); it != m_passBuildersByName.end()) {
		if (m_passNamesSeenThisReset.contains(name)) {
			throw std::runtime_error("Pass names must be unique.");
		}
		if (it->second->Kind() != PassBuilderKind::Render) {
			throw std::runtime_error("Pass builder name collision (render/compute/copy): " + name);
		}
		m_passBuilderOrder.push_back(it->second.get());
		return static_cast<RenderPassBuilder&>(*(it->second));
	}
	m_passNamesSeenThisReset.insert(name);
	auto ptr = std::unique_ptr<RenderPassBuilder>(new RenderPassBuilder(this, name));
	m_passBuilderOrder.push_back(ptr.get());
	m_passBuildersByName.emplace(name, std::move(ptr));
	return static_cast<RenderPassBuilder&>(*(m_passBuildersByName[name]));
}

CopyPassBuilder& RenderGraph::BuildCopyPass(std::string const& name) {
	if (auto it = m_passBuildersByName.find(name); it != m_passBuildersByName.end()) {
		if (m_passNamesSeenThisReset.contains(name)) {
			throw std::runtime_error("Pass names must be unique.");
		}
		if (it->second->Kind() != PassBuilderKind::Copy) {
			throw std::runtime_error("Pass builder name collision (render/compute/copy): " + name);
		}
		m_passBuilderOrder.push_back(it->second.get());
		return static_cast<CopyPassBuilder&>(*(it->second));
	}
	m_passNamesSeenThisReset.insert(name);
	auto ptr = std::unique_ptr<CopyPassBuilder>(new CopyPassBuilder(this, name));
	m_passBuilderOrder.push_back(ptr.get());
	m_passBuildersByName.emplace(name, std::move(ptr));
	return static_cast<CopyPassBuilder&>(*(m_passBuildersByName[name]));
}

//void RenderGraph::RegisterPassBuilder(RenderPassBuilder&& builder) {
//	m_passBuildersByName[builder.passName] = std::move(builder);
//}
//void RenderGraph::RegisterPassBuilder(ComputePassBuilder&& builder) {
//	m_passBuildersByName[builder.passName] = std::move(builder);
//}

QueueSlotIndex RenderGraph::CreateQueue(QueueKind kind, const char* name, QueueAutoAssignmentPolicy autoAssignmentPolicy) {
	auto device = DeviceManager::GetInstance().GetDevice();
	rhi::Queue queue;
	auto result = device.CreateQueue(static_cast<rhi::QueueKind>(kind), name ? name : "UserQueue", queue);
	if (result != rhi::Result::Ok) {
		throw std::runtime_error("Failed to create queue");
	}
	// Determine instance number: count existing slots of this kind.
	uint8_t instance = 0;
	for (size_t i = 0; i < m_queueRegistry.SlotCount(); ++i) {
		if (m_queueRegistry.GetKind(static_cast<QueueSlotIndex>(static_cast<uint8_t>(i))) == kind)
			++instance;
	}
	return m_queueRegistry.Register({ kind, instance }, queue, device, autoAssignmentPolicy);
}

void RenderGraph::SetMinimumAutomaticSchedulingQueues(QueueKind kind, uint8_t count) {
	const size_t kindIndex = static_cast<size_t>(kind);
	const uint8_t clampedCount = kind == QueueKind::Graphics ? (std::max)(uint8_t(1), count) : (std::max)(uint8_t(1), count);
	m_minAutomaticSchedulingQueuesByKind[kindIndex] = clampedCount;

	if (m_queueRegistry.SlotCount() > 0) {
		EnsureMinimumAutomaticSchedulingQueues();
	}
}
