#include "Render/Runtime/IRenderGraphSettingsService.h"

#include "Render/Runtime/OpenRenderGraphSettings.h"

namespace rg::runtime {

namespace {
class DefaultRenderGraphSettingsService final : public IRenderGraphSettingsService {
public:
    bool GetUseAsyncCompute() const override {
        return GetOpenRenderGraphSettings().useAsyncCompute;
    }

    bool GetRenderGraphCompileDumpEnabled() const override {
        return GetOpenRenderGraphSettings().renderGraphCompileDumpEnabled;
    }

    bool GetRenderGraphVramDumpEnabled() const override {
        return GetOpenRenderGraphSettings().renderGraphVramDumpEnabled;
    }

    bool GetRenderGraphBatchTraceEnabled() const override {
        return GetOpenRenderGraphSettings().renderGraphBatchTraceEnabled;
    }

    bool GetRenderGraphLightweightCompileSummaryEnabled() const override {
        return GetOpenRenderGraphSettings().renderGraphLightweightCompileSummaryEnabled;
    }

    bool GetReadOnlyUniformTransitionElisionEnabled() const override {
        return GetOpenRenderGraphSettings().readOnlyUniformTransitionElisionEnabled;
    }

    uint8_t GetAutoAliasMode() const override {
        return GetOpenRenderGraphSettings().autoAliasMode;
    }

    uint8_t GetAutoAliasPackingStrategy() const override {
        return GetOpenRenderGraphSettings().autoAliasPackingStrategy;
    }

    bool GetAutoAliasEnableLogging() const override {
        return GetOpenRenderGraphSettings().autoAliasEnableLogging;
    }

    bool GetAutoAliasLogExclusionReasons() const override {
        return GetOpenRenderGraphSettings().autoAliasLogExclusionReasons;
    }

    bool GetAutoAliasBuildDebugData() const override {
        return GetOpenRenderGraphSettings().autoAliasBuildDebugData;
    }

    bool GetQueueSchedulingEnableLogging() const override {
        return GetOpenRenderGraphSettings().queueSchedulingEnableLogging;
    }

    float GetQueueSchedulingWidthScale() const override {
        return GetOpenRenderGraphSettings().queueSchedulingWidthScale;
    }

    float GetQueueSchedulingPenaltyBias() const override {
        return GetOpenRenderGraphSettings().queueSchedulingPenaltyBias;
    }

    float GetQueueSchedulingMinPenalty() const override {
        return GetOpenRenderGraphSettings().queueSchedulingMinPenalty;
    }

    float GetQueueSchedulingResourcePressureWeight() const override {
        return GetOpenRenderGraphSettings().queueSchedulingResourcePressureWeight;
    }

    float GetQueueSchedulingUavPressureWeight() const override {
        return GetOpenRenderGraphSettings().queueSchedulingUavPressureWeight;
    }

    float GetQueueSchedulingAutoGraphicsBias() const override {
        return GetOpenRenderGraphSettings().queueSchedulingAutoGraphicsBias;
    }

    float GetQueueSchedulingAsyncOverlapBonus() const override {
        return GetOpenRenderGraphSettings().queueSchedulingAsyncOverlapBonus;
    }

    float GetQueueSchedulingCrossQueueHandoffPenalty() const override {
        return GetOpenRenderGraphSettings().queueSchedulingCrossQueueHandoffPenalty;
    }

    uint32_t GetAutoAliasPoolRetireIdleFrames() const override {
        return GetOpenRenderGraphSettings().autoAliasPoolRetireIdleFrames;
    }

    float GetAutoAliasPoolGrowthHeadroom() const override {
        return GetOpenRenderGraphSettings().autoAliasPoolGrowthHeadroom;
    }

    RenderGraphRegionMode GetRenderGraphRegionMode() const override {
        return GetOpenRenderGraphSettings().renderGraphRegionMode;
    }

    TransitionPlacementMode GetTransitionPlacementMode() const override {
        return GetOpenRenderGraphSettings().transitionPlacementMode;
    }

    uint32_t GetRenderGraphRegionMinPassCount() const override {
        return GetOpenRenderGraphSettings().renderGraphRegionMinPassCount;
    }

    bool GetRenderGraphRegionDiagnosticsEnabled() const override {
        return GetOpenRenderGraphSettings().renderGraphRegionDiagnosticsEnabled;
    }

    bool GetRenderGraphRegionShadowStrictBatchMatch() const override {
        return GetOpenRenderGraphSettings().renderGraphRegionShadowStrictBatchMatch;
    }

    uint32_t GetRenderGraphReplaySegmentCacheMaxEntries() const override {
        return GetOpenRenderGraphSettings().renderGraphReplaySegmentCacheMaxEntries;
    }

    uint32_t GetRenderGraphReplaySegmentCacheMaxVariants() const override {
        return GetOpenRenderGraphSettings().renderGraphReplaySegmentCacheMaxVariants;
    }

    uint32_t GetRenderGraphReplaySegmentCacheMaxVariantsPerKey() const override {
        return GetOpenRenderGraphSettings().renderGraphReplaySegmentCacheMaxVariantsPerKey;
    }

    uint32_t GetRenderGraphReplaySegmentCacheMaxAgeFrames() const override {
        return GetOpenRenderGraphSettings().renderGraphReplaySegmentCacheMaxAgeFrames;
    }

    bool GetRenderGraphReplayRelaxAliasPlacement() const override {
        return GetOpenRenderGraphSettings().renderGraphReplayRelaxAliasPlacement;
    }

    bool GetHeavyDebug() const override {
        return GetOpenRenderGraphSettings().heavyDebug;
    }
};
}

std::shared_ptr<IRenderGraphSettingsService> CreateDefaultRenderGraphSettingsService() {
    return std::make_shared<DefaultRenderGraphSettingsService>();
}

}
