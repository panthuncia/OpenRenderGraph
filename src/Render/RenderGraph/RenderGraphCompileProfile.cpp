#include "Render/RenderGraph/RenderGraphCompileProfile.h"

#include "Render/RenderGraph/RenderGraph.h"

#include <algorithm>
#include <chrono>

namespace rg::profile {
namespace {
	thread_local AllocationContext t_allocationContext{};

	uint64_t NowNs() noexcept {
		const auto now = std::chrono::steady_clock::now().time_since_epoch();
		return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
	}

	void AtomicMax(std::atomic<uint64_t>& target, uint64_t value) noexcept {
		uint64_t current = target.load(std::memory_order_relaxed);
		while (current < value && !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
		}
	}
}

void AllocationCounters::Reset() noexcept {
	allocationCount.store(0, std::memory_order_relaxed);
	freeCount.store(0, std::memory_order_relaxed);
	allocatedBytes.store(0, std::memory_order_relaxed);
	freedBytes.store(0, std::memory_order_relaxed);
	liveBytes.store(0, std::memory_order_relaxed);
	peakLiveBytes.store(0, std::memory_order_relaxed);
	largestAllocationBytes.store(0, std::memory_order_relaxed);
}

void AllocationCounters::CopyFrom(const AllocationCounters& other) noexcept {
	allocationCount.store(other.allocationCount.load(std::memory_order_relaxed), std::memory_order_relaxed);
	freeCount.store(other.freeCount.load(std::memory_order_relaxed), std::memory_order_relaxed);
	allocatedBytes.store(other.allocatedBytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
	freedBytes.store(other.freedBytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
	liveBytes.store(other.liveBytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
	peakLiveBytes.store(other.peakLiveBytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
	largestAllocationBytes.store(other.largestAllocationBytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

void CompileProfileStep::CopyFrom(const CompileProfileStep& other) noexcept {
	name = other.name;
	durationNs.store(other.durationNs.load(std::memory_order_relaxed), std::memory_order_relaxed);
	allocations.CopyFrom(other.allocations);
}

void CompileProfileFrame::Reset(uint8_t newFrameIndex, uint64_t newSerial) {
	serial = newSerial;
	frameIndex = newFrameIndex;
	totalDurationNs.store(0, std::memory_order_relaxed);
	liveAllocationBytes.store(0, std::memory_order_relaxed);
	peakLiveAllocationBytes.store(0, std::memory_order_relaxed);
	passCount = 0;
	resourceCount = 0;
	requirementCount = 0;
	dagEdgeCount = 0;
	batchCount = 0;
	transitionCount = 0;
	queueWaitCount = 0;
	queueSignalCount = 0;
	aliasPlacementCount = 0;
	materializationCandidateCount = 0;
	steps.clear();
	steps.reserve(48);
}

void CompileProfileFrame::CopyFrom(const CompileProfileFrame& other) {
	serial = other.serial;
	frameIndex = other.frameIndex;
	totalDurationNs.store(other.totalDurationNs.load(std::memory_order_relaxed), std::memory_order_relaxed);
	liveAllocationBytes.store(other.liveAllocationBytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
	peakLiveAllocationBytes.store(other.peakLiveAllocationBytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
	passCount = other.passCount;
	resourceCount = other.resourceCount;
	requirementCount = other.requirementCount;
	dagEdgeCount = other.dagEdgeCount;
	batchCount = other.batchCount;
	transitionCount = other.transitionCount;
	queueWaitCount = other.queueWaitCount;
	queueSignalCount = other.queueSignalCount;
	aliasPlacementCount = other.aliasPlacementCount;
	materializationCandidateCount = other.materializationCandidateCount;
	steps = other.steps;
}

ScopedAllocationContext::ScopedAllocationContext(AllocationContext context) noexcept
	: m_previous(t_allocationContext) {
	t_allocationContext = context;
}

ScopedAllocationContext::~ScopedAllocationContext() {
	t_allocationContext = m_previous;
}

ScopedCompileProfileStep::ScopedCompileProfileStep(RenderGraph& graph, const char* stepName)
	: m_graph(&graph)
	, m_stepIndex(graph.BeginCompileProfileStep(stepName))
	, m_startedAtNs(NowNs())
	, m_previousContext(t_allocationContext) {
	if (m_stepIndex != UINT32_MAX) {
		t_allocationContext = AllocationContext{ graph.ActiveCompileProfileFrame(), m_stepIndex };
	}
}

ScopedCompileProfileStep::~ScopedCompileProfileStep() {
	if (m_graph && m_stepIndex != UINT32_MAX) {
		m_graph->EndCompileProfileStep(m_stepIndex, NowNs() - m_startedAtNs);
	}
	t_allocationContext = m_previousContext;
}

AllocationContext GetCurrentAllocationContext() noexcept {
	return t_allocationContext;
}

void SetCurrentAllocationContext(AllocationContext context) noexcept {
	t_allocationContext = context;
}

void RecordAllocation(AllocationContext context, uint64_t bytes) noexcept {
	if (!context.IsValid()) {
		return;
	}
	auto* frame = static_cast<CompileProfileFrame*>(context.frame);
	if (context.stepIndex >= frame->steps.size()) {
		return;
	}

	auto& step = frame->steps[context.stepIndex];
	step.allocations.allocationCount.fetch_add(1, std::memory_order_relaxed);
	step.allocations.allocatedBytes.fetch_add(bytes, std::memory_order_relaxed);
	AtomicMax(step.allocations.largestAllocationBytes, bytes);
	const uint64_t stepLive = step.allocations.liveBytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
	AtomicMax(step.allocations.peakLiveBytes, stepLive);

	const uint64_t frameLive = frame->liveAllocationBytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
	AtomicMax(frame->peakLiveAllocationBytes, frameLive);
}

void RecordFree(AllocationContext context, uint64_t bytes) noexcept {
	if (!context.IsValid()) {
		return;
	}
	auto* frame = static_cast<CompileProfileFrame*>(context.frame);
	if (context.stepIndex >= frame->steps.size()) {
		return;
	}

	auto& step = frame->steps[context.stepIndex];
	step.allocations.freeCount.fetch_add(1, std::memory_order_relaxed);
	step.allocations.freedBytes.fetch_add(bytes, std::memory_order_relaxed);
	uint64_t stepLive = step.allocations.liveBytes.load(std::memory_order_relaxed);
	while (stepLive != 0) {
		const uint64_t next = stepLive > bytes ? stepLive - bytes : 0;
		if (step.allocations.liveBytes.compare_exchange_weak(stepLive, next, std::memory_order_relaxed)) {
			break;
		}
	}

	uint64_t frameLive = frame->liveAllocationBytes.load(std::memory_order_relaxed);
	while (frameLive != 0) {
		const uint64_t next = frameLive > bytes ? frameLive - bytes : 0;
		if (frame->liveAllocationBytes.compare_exchange_weak(frameLive, next, std::memory_order_relaxed)) {
			break;
		}
	}
}

} // namespace rg::profile
