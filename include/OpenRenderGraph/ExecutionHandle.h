#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace org {

// How far one execution of a graph has got. Each milestone implies the ones before it, except Poisoned:
// a poisoned execution never submits, or submitted nothing a successor may read as fresh.
enum class ExecutionMilestone : uint8_t {
	Pending = 0,
	Admitted = 1,   // ordered against everything admitted before it; its waits and handoffs are known
	Submitted = 2,  // every batch is on its queue
	Completed = 3,  // the GPU finished every batch
	Poisoned = 4,   // abandoned: it will not submit, or it failed
};

// A point on a timeline the execution signals: the runtime's timeline identity and the value.
struct ExecutionSignal {
	uint64_t timeline = 0;
	uint64_t value = 0;
};

namespace detail {
struct ExecutionRecord;
}

// A shared view of one execution. Copies see the same execution. Any thread may read it or add callbacks.
// A default-constructed handle names no execution.
class ExecutionHandle {
public:
	ExecutionHandle() = default;
	explicit ExecutionHandle(std::shared_ptr<detail::ExecutionRecord> record) noexcept : m_record(std::move(record)) {}

	explicit operator bool() const noexcept { return static_cast<bool>(m_record); }
	ExecutionMilestone Milestone() const noexcept;
	bool Admitted() const noexcept;
	bool Submitted() const noexcept;
	bool Completed() const noexcept;
	bool Poisoned() const noexcept { return Milestone() == ExecutionMilestone::Poisoned; }
	// The graph's count of its executions when this one was made, from 1.
	uint64_t Index() const noexcept;
	// What the execution signals, one point per batch. Empty until it is submitted.
	std::vector<ExecutionSignal> Signals() const;
	// Runs callback once the execution completes or is poisoned, with true when it was poisoned. The graph that ran
	// it calls it when it retires completed work (its next execution, at the latest). It runs now, on the calling
	// thread, when that already happened. Callbacks must not call back into the graph.
	void OnCompleted(std::function<void(bool poisoned)> callback) const;

	const std::shared_ptr<detail::ExecutionRecord>& Record() const noexcept { return m_record; }

private:
	std::shared_ptr<detail::ExecutionRecord> m_record;
};

namespace detail {
// The state behind a handle. The graph that runs the execution moves it forward.
struct ExecutionRecord {
	explicit ExecutionRecord(uint64_t a_index) noexcept : index(a_index) {}
	const uint64_t index;
	std::atomic<ExecutionMilestone> milestone{ExecutionMilestone::Pending};
	std::mutex mutex;
	std::vector<ExecutionSignal> signals;                      // under mutex
	std::vector<std::function<void(bool)>> callbacks;          // under mutex

	void MarkAdmitted() noexcept;
	void MarkSubmitted(std::vector<ExecutionSignal> points);
	// Completed or poisoned: the last milestone. Runs the callbacks, on this thread, outside the lock.
	void Finish(bool poisoned);
};
}

} // namespace org
