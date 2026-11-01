#pragma once

#include <atomic>
#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include <string_view>
#include <rhi.h>
#include "QueueKind.h"


namespace org {

class CommandListPool;

enum class QueueAutoAssignmentPolicy : uint8_t {
	AllowAutomaticScheduling = 0,
	ManualOnly = 1,
};

/// Identifies a logical queue by its kind and instance number.
struct QueueSlot {
	QueueKind kind{};
	uint8_t instance{};
	BackendInstanceId backendInstance = BackendInstanceId::Primary;
	rhi::Backend backend = rhi::Backend::Null;

	bool operator==(const QueueSlot& o) const noexcept { return kind == o.kind && instance == o.instance && backendInstance == o.backendInstance; }
	bool operator!=(const QueueSlot& o) const noexcept { return !(*this == o); }
};

// QueueSlotIndex is defined in QueueKind.h

constexpr uint8_t ToUnderlying(QueueSlotIndex i) noexcept { return static_cast<uint8_t>(i); }

/// A queue's timeline shared by several graphs: the fence and the next value to signal, which every registry holding it
/// assigns from. Graphs that share a queue's timeline need no waits on each other's work on that queue, as one graph does.
struct SharedQueueTimeline {
	std::shared_ptr<rhi::TimelinePtr> fence;
	std::shared_ptr<std::atomic<uint64_t>> nextValue;
	explicit operator bool() const noexcept { return fence && *fence && nextValue; }
};

/// Manages the set of queues available to the render graph.
/// Primary queues (Graphics:0, Compute:0, Copy:0) always occupy slots 0, 1, 2.
class QueueRegistry {
public:
	QueueRegistry() = default;

	/// Registers a queue slot backed by the given rhi::Queue.
	/// Creates a CommandListPool and Timeline for the slot.
	/// Returns the slot index assigned.
	QueueSlotIndex Register(QueueSlot slot, rhi::Queue queue, rhi::Device& device,
		QueueAutoAssignmentPolicy autoAssignmentPolicy = QueueAutoAssignmentPolicy::AllowAutomaticScheduling,
		bool ownsQueue = false,
		std::string_view logicalName = {});

	/// Register a queue slot with an externally-supplied timeline and pool.
	QueueSlotIndex Register(QueueSlot slot, rhi::Queue queue, rhi::TimelinePtr fence, std::unique_ptr<CommandListPool> pool,
		QueueAutoAssignmentPolicy autoAssignmentPolicy = QueueAutoAssignmentPolicy::AllowAutomaticScheduling,
		bool ownsQueue = false,
		rhi::Device device = {},
		std::string_view logicalName = {});

	/// Register a queue slot on a timeline other registries share (SharedTimelineOf): its values come from the shared counter,
	/// and clearing the registry neither waits on the timeline nor retires it (its owner does).
	QueueSlotIndex Register(QueueSlot slot, rhi::Queue queue, const SharedQueueTimeline& timeline, rhi::Device& device,
		QueueAutoAssignmentPolicy autoAssignmentPolicy = QueueAutoAssignmentPolicy::AllowAutomaticScheduling);

	/// Returns a persistent, graph-owned queue registered under this logical name.
	/// Named queues survive structural graph rebuilds, so callers must reuse them.
	QueueSlotIndex FindNamedOwnedSlot(QueueKind kind, std::string_view logicalName) const noexcept;

	/// Look up slot index by kind + instance. Returns empty optional if not found.
	QueueSlotIndex FindSlot(QueueSlot slot) const;

	/// Returns true if the given slot has been registered.
	bool HasSlot(QueueSlot slot) const;

	/// Number of registered queue slots.
	size_t SlotCount() const noexcept { return m_slots.size(); }

	// ---- Per-slot accessors ----

	QueueKind      GetKind(QueueSlotIndex i)     const noexcept { return m_slots[ToUnderlying(i)].kind; }
	uint8_t        GetInstance(QueueSlotIndex i)  const noexcept { return m_slots[ToUnderlying(i)].instance; }
	BackendInstanceId GetBackendInstance(QueueSlotIndex i) const noexcept { return m_slots[ToUnderlying(i)].backendInstance; }
	rhi::Backend GetBackend(QueueSlotIndex i) const noexcept { return m_slots[ToUnderlying(i)].backend; }
	rhi::Device GetDevice(QueueSlotIndex i) const noexcept { return m_slots[ToUnderlying(i)].device; }
	QueueSlot      GetSlot(QueueSlotIndex i)      const noexcept { return { m_slots[ToUnderlying(i)].kind, m_slots[ToUnderlying(i)].instance, m_slots[ToUnderlying(i)].backendInstance, m_slots[ToUnderlying(i)].backend }; }
	rhi::Queue     GetQueue(QueueSlotIndex i)     const noexcept { return m_slots[ToUnderlying(i)].queue; }
	bool           OwnsQueue(QueueSlotIndex i)    const noexcept { return m_slots[ToUnderlying(i)].ownsQueue; }
	QueueAutoAssignmentPolicy GetAutoAssignmentPolicy(QueueSlotIndex i) const noexcept { return m_slots[ToUnderlying(i)].autoAssignmentPolicy; }
	bool IsAutoAssignable(QueueSlotIndex i) const noexcept { return GetAutoAssignmentPolicy(i) == QueueAutoAssignmentPolicy::AllowAutomaticScheduling; }
	rhi::Timeline& GetFence(QueueSlotIndex i)           noexcept { return m_slots[ToUnderlying(i)].fence->Get(); }
	const rhi::Timeline& GetFence(QueueSlotIndex i) const noexcept { return m_slots[ToUnderlying(i)].fence->Get(); }
	rhi::TimelinePtr& GetFencePtr(QueueSlotIndex i)     noexcept { return *m_slots[ToUnderlying(i)].fence; }
	/// The slot's fence, as a shared pointer. It stays valid for its holder after the registry is cleared, for example by
	/// a graph rebuild.
	std::shared_ptr<rhi::TimelinePtr> GetFenceOwner(QueueSlotIndex i) const noexcept { return m_slots[ToUnderlying(i)].fence; }
	/// The slot's timeline, to register in another registry.
	SharedQueueTimeline SharedTimelineOf(QueueSlotIndex i) const noexcept { return {m_slots[ToUnderlying(i)].fence, m_slots[ToUnderlying(i)].fenceValue}; }
	/// Whether the slot's timeline came from another registry (Register with a SharedQueueTimeline).
	bool SharesTimeline(QueueSlotIndex i) const noexcept { return m_slots[ToUnderlying(i)].sharedTimeline; }
	CommandListPool* GetPool(QueueSlotIndex i)    const noexcept { return m_slots[ToUnderlying(i)].pool.get(); }
	std::shared_ptr<CommandListPool> GetSharedPool(QueueSlotIndex i) const noexcept { return m_slots[ToUnderlying(i)].pool; }
	rhi::Timeline& GetFenceForConsumer(QueueSlotIndex source, QueueSlotIndex consumer) noexcept {
		auto& entry = m_slots[ToUnderlying(source)];
		// Slots on the same device instance always consume the source's local
		// timeline. Backend metadata is diagnostic and must not turn a same-device
		// wait into an external-handle lookup if a custom slot was incompletely
		// described.
		return GetBackendInstance(source) == GetBackendInstance(consumer) || !entry.peerFence
			? entry.fence->Get()
			: entry.peerFence.Get();
	}

	/// Replaces backend-local slot fences with two API representations of the
	/// same D3D12 fence payload. Must be called after all queues are registered
	/// and before graph execution begins.
	rhi::Result EnableD3D12VulkanInterop(rhi::Device d3d12Device, rhi::Device vulkanDevice);

	/// Atomically retrieve and increment the per-slot fence value.
	uint64_t GetNextFenceValue(QueueSlotIndex i) noexcept { return m_slots[ToUnderlying(i)].fenceValue->fetch_add(1, std::memory_order_acq_rel); }

	/// Current fence value for a slot (without incrementing).
	uint64_t GetCurrentFenceValue(QueueSlotIndex i) const noexcept { return m_slots[ToUnderlying(i)].fenceValue->load(std::memory_order_acquire); }

	/// Raises the next fence value for a slot if it has fallen behind an externally observed floor.
	void EnsureNextFenceValueAtLeast(QueueSlotIndex i, uint64_t minValue) noexcept {
		auto& nextFenceValue = *m_slots[ToUnderlying(i)].fenceValue;
		for (auto current = nextFenceValue.load(std::memory_order_acquire);
			current < minValue && !nextFenceValue.compare_exchange_weak(current, minValue, std::memory_order_acq_rel);) {}
	}

	/// Find any Graphics-kind queue slot (for transition fallback).
	QueueSlotIndex FindGraphicsSlot() const noexcept;

	/// Returns true if the queue at this slot supports the full range of resource state transitions.
	bool SupportsFullTransitions(QueueSlotIndex i) const noexcept {
		return GetKind(i) == QueueKind::Graphics;
	}

	/// Resets all pools and fences. Called during shutdown.
	void Clear();
	void ShutdownTaskWorkers();

private:
	struct SlotEntry {
		QueueKind kind{};
		uint8_t instance{};
		BackendInstanceId backendInstance = BackendInstanceId::Primary;
		rhi::Backend backend = rhi::Backend::Null;
		rhi::Queue queue{};
		rhi::Device device{};
		std::shared_ptr<rhi::TimelinePtr> fence;
		rhi::TimelinePtr peerFence;
		std::shared_ptr<CommandListPool> pool;
		QueueAutoAssignmentPolicy autoAssignmentPolicy = QueueAutoAssignmentPolicy::AllowAutomaticScheduling;
		bool ownsQueue = false;
		std::string logicalName;
		std::shared_ptr<std::atomic<uint64_t>> fenceValue = std::make_shared<std::atomic<uint64_t>>(1);
		bool sharedTimeline = false;  // Register with a SharedQueueTimeline
	};

	std::vector<SlotEntry> m_slots;
};


} // namespace org
