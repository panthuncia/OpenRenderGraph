#pragma once

#include <algorithm>
#include <cstdint>
#include <mutex>

namespace rg::runtime {

enum class RenderGraphRegionMode : uint8_t {
    Disabled = 0,
    ExtractOnly,
    ValidateOnly,
    ShadowReplay,
    ReplayAuthoritative,
};

enum class TransitionPlacementMode : uint8_t {
    InlineEarlyPlacement = 0,
    CanonicalThenOptimize,
};

struct OpenRenderGraphSettings {
    uint8_t numFramesInFlight = 3;
    bool collectPassStatistics = true;
    bool collectPipelineStatistics = false;

    bool useAsyncCompute = true;
    bool renderGraphCompileDumpEnabled = false;
    bool renderGraphVramDumpEnabled = false;
    bool renderGraphBatchTraceEnabled = false;
    bool renderGraphLightweightCompileSummaryEnabled = false;
    bool readOnlyUniformTransitionElisionEnabled = false;
    uint8_t autoAliasMode = 2;
    uint8_t autoAliasPackingStrategy = 0;
    bool autoAliasEnableLogging = false;
    bool autoAliasLogExclusionReasons = false;
    bool autoAliasBuildDebugData = false;
    bool queueSchedulingEnableLogging = false;
    float queueSchedulingWidthScale = 1.0f;
    float queueSchedulingPenaltyBias = 0.0f;
    float queueSchedulingMinPenalty = 1.0f;
    float queueSchedulingResourcePressureWeight = 1.0f;
    float queueSchedulingUavPressureWeight = 0.5f;
    float queueSchedulingAutoGraphicsBias = 2.5f;
    float queueSchedulingAsyncOverlapBonus = 3.0f;
    float queueSchedulingCrossQueueHandoffPenalty = 2.0f;
    uint32_t autoAliasPoolRetireIdleFrames = 120u;
    float autoAliasPoolGrowthHeadroom = 1.5f;
    RenderGraphRegionMode renderGraphRegionMode = RenderGraphRegionMode::Disabled;
    TransitionPlacementMode transitionPlacementMode = TransitionPlacementMode::InlineEarlyPlacement;
    uint32_t renderGraphRegionMinPassCount = 4u;
    bool renderGraphRegionDiagnosticsEnabled = false;
    bool renderGraphRegionShadowStrictBatchMatch = false;
    uint32_t renderGraphReplaySegmentCacheMaxEntries = 256u;
    uint32_t renderGraphReplaySegmentCacheMaxVariants = 128u;
    uint32_t renderGraphReplaySegmentCacheMaxVariantsPerKey = 32u;
    uint32_t renderGraphReplaySegmentCacheMaxAgeFrames = 0u;
    bool heavyDebug = false;
};

namespace detail {
struct OpenRenderGraphSettingsState {
    std::mutex mutex;
    OpenRenderGraphSettings settings;
};

inline OpenRenderGraphSettingsState& GetOpenRenderGraphSettingsState() {
    static OpenRenderGraphSettingsState state;
    return state;
}
}

inline void SetOpenRenderGraphSettings(const OpenRenderGraphSettings& settings) {
    auto& state = detail::GetOpenRenderGraphSettingsState();
    std::scoped_lock lock(state.mutex);
    state.settings = settings;
    state.settings.numFramesInFlight = (std::max)(uint8_t{ 1 }, state.settings.numFramesInFlight);
    state.settings.queueSchedulingWidthScale = (std::max)(0.0f, state.settings.queueSchedulingWidthScale);
    state.settings.queueSchedulingMinPenalty = (std::max)(0.0f, state.settings.queueSchedulingMinPenalty);
    state.settings.queueSchedulingResourcePressureWeight = (std::max)(0.0f, state.settings.queueSchedulingResourcePressureWeight);
    state.settings.queueSchedulingUavPressureWeight = (std::max)(0.0f, state.settings.queueSchedulingUavPressureWeight);
    state.settings.queueSchedulingAutoGraphicsBias = (std::max)(0.0f, state.settings.queueSchedulingAutoGraphicsBias);
    state.settings.queueSchedulingAsyncOverlapBonus = (std::max)(0.0f, state.settings.queueSchedulingAsyncOverlapBonus);
    state.settings.queueSchedulingCrossQueueHandoffPenalty = (std::max)(0.0f, state.settings.queueSchedulingCrossQueueHandoffPenalty);
    state.settings.autoAliasPoolRetireIdleFrames = (std::max)(1u, state.settings.autoAliasPoolRetireIdleFrames);
    state.settings.autoAliasPoolGrowthHeadroom = (std::max)(1.0f, state.settings.autoAliasPoolGrowthHeadroom);
    state.settings.renderGraphRegionMinPassCount = (std::max)(1u, state.settings.renderGraphRegionMinPassCount);
    state.settings.renderGraphReplaySegmentCacheMaxEntries = (std::max)(1u, state.settings.renderGraphReplaySegmentCacheMaxEntries);
    state.settings.renderGraphReplaySegmentCacheMaxVariants = (std::max)(1u, state.settings.renderGraphReplaySegmentCacheMaxVariants);
    state.settings.renderGraphReplaySegmentCacheMaxVariantsPerKey = (std::max)(1u, state.settings.renderGraphReplaySegmentCacheMaxVariantsPerKey);
}

inline OpenRenderGraphSettings GetOpenRenderGraphSettings() {
    auto& state = detail::GetOpenRenderGraphSettingsState();
    std::scoped_lock lock(state.mutex);
    return state.settings;
}

}
