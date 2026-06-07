#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class RenderGraph;

namespace rg::profile {

struct AllocationContext {
	void* frame = nullptr;
	uint32_t stepIndex = 0;

	bool IsValid() const noexcept {
		return frame != nullptr;
	}
};

struct AllocationCounters {
	std::atomic<uint64_t> allocationCount{ 0 };
	std::atomic<uint64_t> freeCount{ 0 };
	std::atomic<uint64_t> allocatedBytes{ 0 };
	std::atomic<uint64_t> freedBytes{ 0 };
	std::atomic<uint64_t> liveBytes{ 0 };
	std::atomic<uint64_t> peakLiveBytes{ 0 };
	std::atomic<uint64_t> largestAllocationBytes{ 0 };

	AllocationCounters() = default;

	AllocationCounters(const AllocationCounters& other) {
		CopyFrom(other);
	}

	AllocationCounters& operator=(const AllocationCounters& other) {
		if (this != &other) {
			CopyFrom(other);
		}
		return *this;
	}

	void Reset() noexcept;
	void CopyFrom(const AllocationCounters& other) noexcept;
};

struct CompileProfileStep {
	std::string name;
	std::atomic<uint64_t> durationNs{ 0 };
	AllocationCounters allocations;

	CompileProfileStep() = default;
	explicit CompileProfileStep(std::string stepName)
		: name(std::move(stepName)) {}

	CompileProfileStep(const CompileProfileStep& other) {
		CopyFrom(other);
	}

	CompileProfileStep& operator=(const CompileProfileStep& other) {
		if (this != &other) {
			CopyFrom(other);
		}
		return *this;
	}

	CompileProfileStep(CompileProfileStep&& other) noexcept {
		CopyFrom(other);
		name = std::move(other.name);
	}

	CompileProfileStep& operator=(CompileProfileStep&& other) noexcept {
		if (this != &other) {
			name = std::move(other.name);
			durationNs.store(other.durationNs.load(std::memory_order_relaxed), std::memory_order_relaxed);
			allocations.CopyFrom(other.allocations);
		}
		return *this;
	}

	void CopyFrom(const CompileProfileStep& other) noexcept;
};

struct CompileProfileFrame {
	uint64_t serial = 0;
	uint8_t frameIndex = 0;
	std::atomic<uint64_t> totalDurationNs{ 0 };
	std::atomic<uint64_t> liveAllocationBytes{ 0 };
	std::atomic<uint64_t> peakLiveAllocationBytes{ 0 };
	uint64_t passCount = 0;
	uint64_t resourceCount = 0;
	uint64_t requirementCount = 0;
	uint64_t dagEdgeCount = 0;
	uint64_t batchCount = 0;
	uint64_t transitionCount = 0;
	uint64_t queueWaitCount = 0;
	uint64_t queueSignalCount = 0;
	uint64_t aliasPlacementCount = 0;
	uint64_t materializationCandidateCount = 0;
	std::vector<CompileProfileStep> steps;

	CompileProfileFrame() = default;
	CompileProfileFrame(const CompileProfileFrame& other) {
		CopyFrom(other);
	}

	CompileProfileFrame& operator=(const CompileProfileFrame& other) {
		if (this != &other) {
			CopyFrom(other);
		}
		return *this;
	}

	void Reset(uint8_t newFrameIndex, uint64_t newSerial);
	void CopyFrom(const CompileProfileFrame& other);
};

class ICompileProfileSink {
public:
	virtual ~ICompileProfileSink() = default;
	virtual void OnCompileProfileFrame(const CompileProfileFrame& frame) = 0;
};

class ScopedAllocationContext {
public:
	explicit ScopedAllocationContext(AllocationContext context) noexcept;
	~ScopedAllocationContext();

	ScopedAllocationContext(const ScopedAllocationContext&) = delete;
	ScopedAllocationContext& operator=(const ScopedAllocationContext&) = delete;

private:
	AllocationContext m_previous{};
};

class ScopedCompileProfileStep {
public:
	ScopedCompileProfileStep(RenderGraph& graph, const char* stepName);
	~ScopedCompileProfileStep();

	ScopedCompileProfileStep(const ScopedCompileProfileStep&) = delete;
	ScopedCompileProfileStep& operator=(const ScopedCompileProfileStep&) = delete;

private:
	RenderGraph* m_graph = nullptr;
	uint32_t m_stepIndex = UINT32_MAX;
	uint64_t m_startedAtNs = 0;
	AllocationContext m_previousContext{};
};

AllocationContext GetCurrentAllocationContext() noexcept;
void SetCurrentAllocationContext(AllocationContext context) noexcept;
void RecordAllocation(AllocationContext context, uint64_t bytes) noexcept;
void RecordFree(AllocationContext context, uint64_t bytes) noexcept;

} // namespace rg::profile
