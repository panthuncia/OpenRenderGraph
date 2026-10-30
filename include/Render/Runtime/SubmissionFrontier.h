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
};

// Every queue's progress at one moment. Anything released then (no longer reachable by work recorded later) is safe to
// destroy once each point is reached: whatever GPU work could still use it was submitted before.
using SubmissionFrontier = std::vector<SubmittedPoint>;

// Whether the GPU reached a point: the runtime's (Runtime::Reached), which also knows the points of queues it no longer has.
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
