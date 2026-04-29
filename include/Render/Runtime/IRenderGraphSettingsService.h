#pragma once

#include <cstdint>
#include <memory>

namespace rg::runtime {

class IRenderGraphSettingsService {
public:
    virtual ~IRenderGraphSettingsService() = default;

    virtual bool GetUseAsyncCompute() const = 0;
    virtual bool GetRenderGraphCompileDumpEnabled() const = 0;
    virtual bool GetRenderGraphVramDumpEnabled() const = 0;
    virtual bool GetRenderGraphBatchTraceEnabled() const = 0;
    virtual uint8_t GetAutoAliasMode() const = 0;
    virtual uint8_t GetAutoAliasPackingStrategy() const = 0;
    virtual bool GetAutoAliasEnableLogging() const = 0;
    virtual bool GetAutoAliasLogExclusionReasons() const = 0;
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
    virtual bool GetHeavyDebug() const = 0;
};

std::shared_ptr<IRenderGraphSettingsService> CreateDefaultRenderGraphSettingsService();

}
