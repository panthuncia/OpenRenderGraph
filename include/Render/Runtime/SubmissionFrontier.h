#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <rhi.h>

namespace org::runtime {

// A point on a queue's progress timeline (rhi::Queue::Progress): everything submitted to that queue when it was taken.
struct SubmittedPoint {
	rhi::Timeline timeline;
	uint64_t value = 0;
	// What the point stands for, for logs: "queue", "host timeline" or "external", and its index among those.
	const char* source = nullptr;
	uint32_t sourceIndex = 0;
};

// Every queue's progress at one moment. Something released at that moment is no longer reachable by work recorded
// later. It is safe to destroy once each point is reached, because any GPU work that could still use it was
// submitted before.
using SubmissionFrontier = std::vector<SubmittedPoint>;

// Whether the GPU reached a point. It uses the runtime's check (Runtime::Reached), which also knows the points of
// queues the runtime no longer has.
using PointReached = std::function<bool(const SubmittedPoint&)>;

inline bool TimelineReached(SubmittedPoint point) noexcept {
	if (!point.timeline.IsValid()) return true;
	const uint64_t completed = point.timeline.GetCompletedValue();
	return completed != UINT64_MAX && completed >= point.value;
}

inline bool Reached(const SubmissionFrontier& frontier, const PointReached& reached) {
	for (const auto& point : frontier)
		if (!(reached ? reached(point) : TimelineReached(point))) return false;
	return true;
}

} // namespace org::runtime
