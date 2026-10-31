#include "OpenRenderGraph/ExecutionHandle.h"

namespace org {

namespace {
bool Reached(ExecutionMilestone milestone, ExecutionMilestone at) noexcept {
	return milestone != ExecutionMilestone::Poisoned && static_cast<uint8_t>(milestone) >= static_cast<uint8_t>(at);
}
}

ExecutionMilestone ExecutionHandle::Milestone() const noexcept {
	return m_record ? m_record->milestone.load(std::memory_order_acquire) : ExecutionMilestone::Pending;
}
bool ExecutionHandle::Admitted() const noexcept { return Reached(Milestone(), ExecutionMilestone::Admitted); }
bool ExecutionHandle::Submitted() const noexcept { return Reached(Milestone(), ExecutionMilestone::Submitted); }
bool ExecutionHandle::Completed() const noexcept { return Milestone() == ExecutionMilestone::Completed; }
uint64_t ExecutionHandle::Index() const noexcept { return m_record ? m_record->index : 0; }

std::vector<ExecutionSignal> ExecutionHandle::Signals() const {
	if (!m_record) return {};
	std::scoped_lock lock(m_record->mutex);
	return m_record->signals;
}

void ExecutionHandle::OnCompleted(std::function<void(bool poisoned)> callback) const {
	if (!m_record || !callback) return;
	{
		std::scoped_lock lock(m_record->mutex);
		const auto milestone = m_record->milestone.load(std::memory_order_acquire);
		if (milestone != ExecutionMilestone::Completed && milestone != ExecutionMilestone::Poisoned) {
			m_record->callbacks.push_back(std::move(callback));
			return;
		}
	}
	callback(Poisoned());
}

namespace detail {

void ExecutionRecord::MarkAdmitted() noexcept {
	auto expected = ExecutionMilestone::Pending;
	milestone.compare_exchange_strong(expected, ExecutionMilestone::Admitted, std::memory_order_acq_rel);
}

void ExecutionRecord::MarkSubmitted(std::vector<ExecutionSignal> points) {
	std::scoped_lock lock(mutex);
	signals = std::move(points);
	const auto current = milestone.load(std::memory_order_acquire);
	if (current == ExecutionMilestone::Pending || current == ExecutionMilestone::Admitted)
		milestone.store(ExecutionMilestone::Submitted, std::memory_order_release);
}

void ExecutionRecord::Finish(bool poisoned) {
	std::vector<std::function<void(bool)>> run;
	{
		std::scoped_lock lock(mutex);
		const auto current = milestone.load(std::memory_order_acquire);
		if (current == ExecutionMilestone::Completed || current == ExecutionMilestone::Poisoned) return;
		milestone.store(poisoned ? ExecutionMilestone::Poisoned : ExecutionMilestone::Completed, std::memory_order_release);
		run.swap(callbacks);
	}
	for (auto& callback : run) callback(poisoned);
}

} // namespace detail
} // namespace org
