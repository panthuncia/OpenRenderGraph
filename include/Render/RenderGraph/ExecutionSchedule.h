#pragma once

#include <array>
#include <vector>
#include <cstdint>
#include "Render/CommandListPool.h"
#include "Render/QueueKind.h"

// Describes the command-list layout for one queue within one batch.
// Computed deterministically from the batch's signal flags.
struct QueueBatchSchedule {
	bool active = false;              // queue has transitions or passes in this batch
	uint8_t numCLs = 0;              // 0..3, computed from signal flags
	bool splitAfterTransitions = false; // AfterTransitions signal → CL boundary
	bool splitAfterExecution = false;   // AfterExecution signal → CL boundary
	bool signalAfterCompletion = false; // AfterCompletion signal present

	// Pre-allocated CL pairs (indexed 0..numCLs-1).
	// Filled during the pre-allocation phase of Execute().
	std::array<CommandListPair, 3> preallocatedCLs;
};

// Per-batch schedule, one entry per queue kind.
struct BatchSchedule {
	static constexpr size_t kQueueCount = static_cast<size_t>(QueueKind::Count);
	std::array<QueueBatchSchedule, kQueueCount> queues{};
};

// The full pre-computed execution schedule for a frame.
struct ExecutionSchedule {
	std::vector<BatchSchedule> batches;
};
