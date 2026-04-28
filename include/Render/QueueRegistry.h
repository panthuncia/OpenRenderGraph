#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <rhi.h>
#include "QueueKind.h"

class CommandListPool;

enum class QueueAutoAssignmentPolicy : uint8_t {
	AllowAutomaticScheduling = 0,
	ManualOnly = 1,
};

/// Identifies a logical queue by its kind and instance number.
struct QueueSlot {
	QueueKind kind{};
	uint8_t instance{};

	bool operator==(const QueueSlot& o) const noexcept { return kind == o.kind && instance == o.instance; }
	bool operator!=(const QueueSlot& o) const noexcept { return !(*this == o); }
};

// QueueSlotIndex is defined in QueueKind.h

constexpr uint8_t ToUnderlying(QueueSlotIndex i) noexcept { return static_cast<uint8_t>(i); }

/// Manages the set of queues available to the render graph.
/// Primary queues (Graphics:0, Compute:0, Copy:0) always occupy slots 0, 1, 2.
class QueueRegistry {
public:
	QueueRegistry() = default;

	/// Registers a queue slot backed by the given rhi::Queue.
	/// Creates a CommandListPool and Timeline for the slot.
	/// Returns the slot index assigned.
	QueueSlotIndex Register(QueueSlot slot, rhi::Queue queue, rhi::Device& device,
		QueueAutoAssignmentPolicy autoAssignmentPolicy = QueueAutoAssignmentPolicy::AllowAutomaticScheduling);

	/// Register a queue slot with an externally-supplied timeline and pool.
	QueueSlotIndex Register(QueueSlot slot, rhi::Queue queue, rhi::TimelinePtr fence, std::unique_ptr<CommandListPool> pool,
		QueueAutoAssignmentPolicy autoAssignmentPolicy = QueueAutoAssignmentPolicy::AllowAutomaticScheduling);

	/// Look up slot index by kind + instance. Returns empty optional if not found.
	QueueSlotIndex FindSlot(QueueSlot slot) const;

	/// Returns true if the given slot has been registered.
	bool HasSlot(QueueSlot slot) const;

	/// Number of registered queue slots.
	size_t SlotCount() const noexcept { return m_slots.size(); }

	// ---- Per-slot accessors ----

	QueueKind      GetKind(QueueSlotIndex i)     const noexcept { return m_slots[ToUnderlying(i)].kind; }
	uint8_t        GetInstance(QueueSlotIndex i)  const noexcept { return m_slots[ToUnderlying(i)].instance; }
	QueueSlot      GetSlot(QueueSlotIndex i)      const noexcept { return { m_slots[ToUnderlying(i)].kind, m_slots[ToUnderlying(i)].instance }; }
	rhi::Queue     GetQueue(QueueSlotIndex i)     const noexcept { return m_slots[ToUnderlying(i)].queue; }
	QueueAutoAssignmentPolicy GetAutoAssignmentPolicy(QueueSlotIndex i) const noexcept { return m_slots[ToUnderlying(i)].autoAssignmentPolicy; }
	bool IsAutoAssignable(QueueSlotIndex i) const noexcept { return GetAutoAssignmentPolicy(i) == QueueAutoAssignmentPolicy::AllowAutomaticScheduling; }
	rhi::Timeline& GetFence(QueueSlotIndex i)           noexcept { return m_slots[ToUnderlying(i)].fence.Get(); }
	const rhi::Timeline& GetFence(QueueSlotIndex i) const noexcept { return m_slots[ToUnderlying(i)].fence.Get(); }
	rhi::TimelinePtr& GetFencePtr(QueueSlotIndex i)     noexcept { return m_slots[ToUnderlying(i)].fence; }
	CommandListPool* GetPool(QueueSlotIndex i)    const noexcept { return m_slots[ToUnderlying(i)].pool.get(); }

	/// Atomically retrieve and increment the per-slot fence value.
	uint64_t GetNextFenceValue(QueueSlotIndex i) noexcept { return m_slots[ToUnderlying(i)].fenceValue++; }

	/// Current fence value for a slot (without incrementing).
	uint64_t GetCurrentFenceValue(QueueSlotIndex i) const noexcept { return m_slots[ToUnderlying(i)].fenceValue; }

	/// Raises the next fence value for a slot if it has fallen behind an externally observed floor.
	void EnsureNextFenceValueAtLeast(QueueSlotIndex i, uint64_t minValue) noexcept {
		auto& nextFenceValue = m_slots[ToUnderlying(i)].fenceValue;
		if (nextFenceValue < minValue) {
			nextFenceValue = minValue;
		}
	}

	/// Find any Graphics-kind queue slot (for transition fallback).
	QueueSlotIndex FindGraphicsSlot() const noexcept;

	/// Returns true if the queue at this slot supports the full range of resource state transitions.
	bool SupportsFullTransitions(QueueSlotIndex i) const noexcept {
		return GetKind(i) == QueueKind::Graphics;
	}

	/// Resets all pools and fences. Called during shutdown.
	void Clear();

private:
	struct SlotEntry {
		QueueKind kind{};
		uint8_t instance{};
		rhi::Queue queue{};
		rhi::TimelinePtr fence;
		std::unique_ptr<CommandListPool> pool;
		QueueAutoAssignmentPolicy autoAssignmentPolicy = QueueAutoAssignmentPolicy::AllowAutomaticScheduling;
		uint64_t fenceValue = 1;
	};

	std::vector<SlotEntry> m_slots;
};
