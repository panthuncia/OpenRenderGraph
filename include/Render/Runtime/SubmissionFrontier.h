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

// When a frontier taken now is complete. A submitter holds an assignment scope from taking values on a shared timeline
// until it publishes them (Runtime::AssignmentScope); work still queued in such a scope may use what is released now, and
// a frontier taken meanwhile misses its values. So a release notes mark() and takes its frontier once closedBefore(mark):
// every scope open at the release has published. Scopes opened later cannot reach what was released. Without a gate, a
// frontier taken now is complete.
struct AssignmentGate {
	std::function<uint64_t()> mark;
	std::function<bool(uint64_t)> closedBefore;

	explicit operator bool() const noexcept { return mark && closedBefore; }
	// 0 without a gate: closed at once.
	uint64_t Mark() const { return *this ? mark() : 0; }
	bool ClosedBefore(uint64_t a_mark) const { return !*this || closedBefore(a_mark); }
};

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
