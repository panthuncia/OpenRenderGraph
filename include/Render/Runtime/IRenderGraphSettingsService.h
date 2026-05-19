#pragma once

#include <cstdint>
#include <memory>

#include "Render/Runtime/OpenRenderGraphSettings.h"

namespace rg::runtime {

class IRenderGraphSettingsService {
public:
    virtual ~IRenderGraphSettingsService() = default;

    virtual bool GetUseAsyncCompute() const = 0;
    virtual bool GetRenderGraphCompileDumpEnabled() const = 0;
    virtual bool GetRenderGraphVramDumpEnabled() const = 0;
    virtual bool GetRenderGraphBatchTraceEnabled() const = 0;
    virtual bool GetRenderGraphLightweightCompileSummaryEnabled() const = 0;
    virtual bool GetReadOnlyUniformTransitionElisionEnabled() const = 0;
    virtual uint8_t GetAutoAliasMode() const = 0;
    virtual uint8_t GetAutoAliasPackingStrategy() const = 0;
    virtual bool GetAutoAliasEnableLogging() const = 0;
    virtual bool GetAutoAliasLogExclusionReasons() const = 0;
    virtual bool GetAutoAliasBuildDebugData() const = 0;
    virtual bool GetQueueSchedulingEnableLogging() const = 0;
    virtual float GetQueueSchedulingWidthScale() const = 0;
    virtual float GetQueueSchedulingPenaltyBias() const = 0;
    virtual float GetQueueSchedulingMinPenalty() const = 0;
    virtual float GetQueueSchedulingResourcePressureWeight() const = 0;
    virtual float GetQueueSchedulingUavPressureWeight() const = 0;
    virtual float GetQueueSchedulingAutoGraphicsBias() const = 0;
    virtual float GetQueueSchedulingAsyncOverlapBonus() const = 0;
    virtual float GetQueueSchedulingCrossQueueHandoffPenalty() const = 0;
    virtual uint32_t GetAutoAliasPoolRetireIdleFrames() const = 0;
    virtual float GetAutoAliasPoolGrowthHeadroom() const = 0;
    virtual RenderGraphRegionMode GetRenderGraphRegionMode() const = 0;
    virtual TransitionPlacementMode GetTransitionPlacementMode() const = 0;
    virtual uint32_t GetRenderGraphRegionMinPassCount() const = 0;
    virtual bool GetRenderGraphRegionDiagnosticsEnabled() const = 0;
    virtual bool GetRenderGraphRegionShadowStrictBatchMatch() const = 0;
    virtual uint32_t GetRenderGraphReplaySegmentCacheMaxEntries() const = 0;
    virtual uint32_t GetRenderGraphReplaySegmentCacheMaxVariants() const = 0;
    virtual uint32_t GetRenderGraphReplaySegmentCacheMaxVariantsPerKey() const = 0;
    virtual uint32_t GetRenderGraphReplaySegmentCacheMaxAgeFrames() const = 0;
    virtual bool GetRenderGraphReplayRelaxAliasPlacement() const = 0;
    virtual bool GetHeavyDebug() const = 0;
};

std::shared_ptr<IRenderGraphSettingsService> CreateDefaultRenderGraphSettingsService();

}
