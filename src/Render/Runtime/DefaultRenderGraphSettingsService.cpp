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

    uint32_t GetAutoAliasPoolRetireIdleFrames() const override {
        return GetOpenRenderGraphSettings().autoAliasPoolRetireIdleFrames;
    }

    float GetAutoAliasPoolGrowthHeadroom() const override {
        return GetOpenRenderGraphSettings().autoAliasPoolGrowthHeadroom;
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
