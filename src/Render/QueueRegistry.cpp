#include "Render/QueueRegistry.h"
#include "Render/CommandListPool.h"

QueueSlotIndex QueueRegistry::Register(QueueSlot slot, rhi::Queue queue, rhi::Device& device, QueueAutoAssignmentPolicy autoAssignmentPolicy) {
	auto pool = std::make_unique<CommandListPool>(device, static_cast<rhi::QueueKind>(slot.kind));
	rhi::TimelinePtr fence;
	device.CreateTimeline(fence);
	return Register(slot, queue, std::move(fence), std::move(pool), autoAssignmentPolicy);
}

QueueSlotIndex QueueRegistry::Register(QueueSlot slot, rhi::Queue queue, rhi::TimelinePtr fence, std::unique_ptr<CommandListPool> pool, QueueAutoAssignmentPolicy autoAssignmentPolicy) {
	auto idx = static_cast<QueueSlotIndex>(static_cast<uint8_t>(m_slots.size()));
	m_slots.push_back({ slot.kind, slot.instance, queue, std::move(fence), std::move(pool), autoAssignmentPolicy, 1 });
	return idx;
}

QueueSlotIndex QueueRegistry::FindSlot(QueueSlot slot) const {
	for (size_t i = 0; i < m_slots.size(); ++i) {
		if (m_slots[i].kind == slot.kind && m_slots[i].instance == slot.instance)
			return static_cast<QueueSlotIndex>(static_cast<uint8_t>(i));
	}
	return static_cast<QueueSlotIndex>(0xFF);
}

bool QueueRegistry::HasSlot(QueueSlot slot) const {
	for (auto& s : m_slots) {
		if (s.kind == slot.kind && s.instance == slot.instance) return true;
	}
	return false;
}

QueueSlotIndex QueueRegistry::FindGraphicsSlot() const noexcept {
	for (size_t i = 0; i < m_slots.size(); ++i) {
		if (m_slots[i].kind == QueueKind::Graphics)
			return static_cast<QueueSlotIndex>(static_cast<uint8_t>(i));
	}
	return static_cast<QueueSlotIndex>(0);
}

void QueueRegistry::Clear() {
	m_slots.clear();
}
