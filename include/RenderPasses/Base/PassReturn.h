#pragma once

#include <rhi.h>
#include <optional>
#include <cstdint>
#include <vector>

struct ExternalTimelinePoint {
	rhi::Timeline timeline;
	uint64_t value = 0;
};

enum class ExternalSignalPhase : uint8_t {
	AfterCompletion,
};

enum class ExternalWaitPhase : uint8_t {
	BeforeTransitions,
};

struct PassReturn {
	std::optional<rhi::Timeline> fence;
	uint64_t fenceValue = 0;
	std::vector<ExternalTimelinePoint> externalSignalsAfterCompletion;
};
