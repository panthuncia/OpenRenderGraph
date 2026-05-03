#include "Render/RenderGraph/Aliasing/RenderGraphAliasingSubsystem.h"

#include <algorithm>

#include "Render/RenderGraph/RenderGraph.h"
#include "Managers/Singletons/DeletionManager.h"

rg::alias::AutoAliasDebugSnapshot rg::alias::RenderGraphAliasingSubsystem::BuildDebugSnapshot(
	AutoAliasMode mode,
	AutoAliasPackingStrategy packingStrategy,
	const rg::alias::AutoAliasPlannerStats& plannerStats,
	const std::vector<rg::alias::AutoAliasReasonCount>& exclusionReasons,
	const std::vector<rg::alias::AutoAliasPoolDebug>& poolDebug) const
{
	rg::alias::AutoAliasDebugSnapshot out{};
	out.mode = mode;
	out.packingStrategy = packingStrategy;
	out.candidatesSeen = plannerStats.candidatesSeen;
	out.manuallyAssigned = plannerStats.manuallyAssigned;
	out.autoAssigned = plannerStats.autoAssigned;
	out.excluded = plannerStats.excluded;
	out.candidateBytes = plannerStats.candidateBytes;
	out.autoAssignedBytes = plannerStats.autoAssignedBytes;
	out.pooledIndependentBytes = plannerStats.pooledIndependentBytes;
	out.pooledActualBytes = plannerStats.pooledActualBytes;
	out.pooledSavedBytes = plannerStats.pooledSavedBytes;
	out.planCacheHits = plannerStats.planCacheHits;
	out.planCacheMisses = plannerStats.planCacheMisses;
	out.primaryPlanCacheMissReason = plannerStats.primaryPlanCacheMissReason != nullptr
		? plannerStats.primaryPlanCacheMissReason
		: std::string{};
	out.exclusionReasons = exclusionReasons;
	out.poolDebug = poolDebug;
	return out;
}

void rg::alias::RenderGraphAliasingSubsystem::ResetPerFrameState(RenderGraph& renderGraph) const {
	renderGraph.aliasMaterializeOptionsByID.clear();
	renderGraph.aliasActivationPending.clear();
	renderGraph.m_aliasPlacementRangeByResourceIndex.clear();
	renderGraph.m_hasAliasPlacementByResourceIndex.clear();
	renderGraph.m_schedulingPlacementRangeByResourceIndex.clear();
	renderGraph.m_hasSchedulingPlacementByResourceIndex.clear();
	renderGraph.m_aliasActivationPendingByResourceIndex.clear();
	renderGraph.autoAliasPoolByID.clear();
	renderGraph.autoAliasExclusionReasonByID.clear();
	renderGraph.autoAliasExclusionReasonSummary.clear();
	renderGraph.schedulingPlacementRangesByID.clear();
	renderGraph.autoAliasPlannerStats = {};
	renderGraph.autoAliasPreviousMode = renderGraph.autoAliasModeLastFrame;
}

void rg::alias::RenderGraphAliasingSubsystem::ResetPersistentState(RenderGraph& renderGraph) const {
	ResetPerFrameState(renderGraph);
	renderGraph.autoAliasPreviousMode = AutoAliasMode::Off;
	renderGraph.autoAliasModeLastFrame = AutoAliasMode::Off;
	renderGraph.aliasPlacementSignatureByID.clear();
	renderGraph.aliasPlacementRangesByID.clear();
	renderGraph.schedulingPlacementRangesByID.clear();
	renderGraph.aliasPlacementPoolByID.clear();
	renderGraph.cachedAliasPlanByPoolID.clear();

	for (auto& [poolID, poolState] : renderGraph.persistentAliasPools) {
		(void)poolID;
		if (poolState.allocation) {
			DeletionManager::GetInstance().MarkForDelete(std::move(poolState.allocation));
		}
	}
	renderGraph.persistentAliasPools.clear();
	renderGraph.aliasPoolPlanFrameIndex = 0;
}
