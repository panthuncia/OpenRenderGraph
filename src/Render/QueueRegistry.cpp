#include "Render/QueueRegistry.h"
#include "Render/CommandListPool.h"

#include <string>
#include <spdlog/spdlog.h>
#include <rhi_interop_dx12.h>
#include <rhi_interop_vulkan.h>
#ifdef _WIN32
#include <Windows.h>
#endif


namespace org {

namespace {
	const char* QueueKindDebugName(QueueKind kind) noexcept {
		switch (kind) {
		case QueueKind::Graphics: return "Graphics";
		case QueueKind::Compute: return "Compute";
		case QueueKind::Copy: return "Copy";
		default: return "Unknown";
		}
	}
}

QueueSlotIndex QueueRegistry::Register(QueueSlot slot, rhi::Queue queue, rhi::Device& device, QueueAutoAssignmentPolicy autoAssignmentPolicy, bool ownsQueue, std::string_view logicalName) {
	auto pool = std::make_unique<CommandListPool>(device, static_cast<rhi::QueueKind>(slot.kind));
	rhi::TimelinePtr fence;
	device.CreateTimeline(fence);
	return Register(slot, queue, std::move(fence), std::move(pool), autoAssignmentPolicy, ownsQueue, ownsQueue ? device : rhi::Device{}, logicalName);
}

QueueSlotIndex QueueRegistry::Register(QueueSlot slot, rhi::Queue queue, rhi::TimelinePtr fence, std::unique_ptr<CommandListPool> pool, QueueAutoAssignmentPolicy autoAssignmentPolicy, bool ownsQueue, rhi::Device device, std::string_view logicalName) {
	auto idx = static_cast<QueueSlotIndex>(static_cast<uint8_t>(m_slots.size()));
	const std::string queueName = "ORG QueueSlot " + std::to_string(static_cast<uint8_t>(idx)) +
		" backend=" + std::to_string(static_cast<uint8_t>(slot.backendInstance)) +
		" " + QueueKindDebugName(slot.kind) + ":" + std::to_string(slot.instance);
	if (queue) {
		queue.SetName(queueName.c_str());
		const rhi::Result tracyResult = queue.InitializeTracyGpuContext(queueName.c_str());
		if (tracyResult != rhi::Result::Ok && tracyResult != rhi::Result::Unsupported) {
			spdlog::warn(
				"Failed to initialize Tracy GPU profiling for '{}': {}",
				queueName,
				rhi::ResultName(tracyResult));
		}
	}
	if (fence) {
		const std::string fenceName = queueName + " Fence";
		fence->SetName(fenceName.c_str());
	}
	m_slots.push_back({ slot.kind, slot.instance, slot.backendInstance, slot.backend, queue, device, std::move(fence), {}, std::move(pool), autoAssignmentPolicy, ownsQueue, std::string(logicalName), 1 });
	return idx;
}

QueueSlotIndex QueueRegistry::FindNamedOwnedSlot(QueueKind kind, std::string_view logicalName) const noexcept {
	if (logicalName.empty()) return static_cast<QueueSlotIndex>(0xFF);
	for (size_t i = 0; i < m_slots.size(); ++i) {
		const auto& slot = m_slots[i];
		if (slot.ownsQueue && slot.kind == kind && slot.logicalName == logicalName) {
			return static_cast<QueueSlotIndex>(static_cast<uint8_t>(i));
		}
	}
	return static_cast<QueueSlotIndex>(0xFF);
}

rhi::Result QueueRegistry::EnableD3D12VulkanInterop(rhi::Device d3d12Device, rhi::Device vulkanDevice) {
#ifdef _WIN32
	if (!d3d12Device || !vulkanDevice) return rhi::Result::InvalidArgument;
	for (size_t i = 0; i < m_slots.size(); ++i) {
		auto& slot = m_slots[i];
		if (slot.backend != rhi::Backend::D3D12 && slot.backend != rhi::Backend::Vulkan) continue;
		rhi::TimelinePtr d3dFence;
		auto result = d3d12Device.CreateTimeline(d3dFence, 0, "ORG Multi-RHI Bridge Fence", true);
		if (rhi::Failed(result)) return result;
		rhi::dx12::SharedHandle shared{};
		result = rhi::dx12::export_shared_timeline(d3d12Device, d3dFence.Get(), shared);
		if (rhi::Failed(result)) return result;
		rhi::TimelinePtr vkFence;
		result = rhi::vulkan::import_d3d12_timeline(vulkanDevice, shared.value, 0, "ORG Multi-RHI Bridge Timeline", vkFence);
		CloseHandle(static_cast<HANDLE>(shared.value));
		if (rhi::Failed(result)) return result;
		if (slot.backend == rhi::Backend::D3D12) {
			slot.fence = std::move(d3dFence);
			slot.peerFence = std::move(vkFence);
		} else {
			slot.fence = std::move(vkFence);
			slot.peerFence = std::move(d3dFence);
		}
		slot.fenceValue = 1;
	}
	return rhi::Result::Ok;
#else
	(void)d3d12Device; (void)vulkanDevice;
	return rhi::Result::Unsupported;
#endif
}

QueueSlotIndex QueueRegistry::FindSlot(QueueSlot slot) const {
	for (size_t i = 0; i < m_slots.size(); ++i) {
		if (m_slots[i].kind == slot.kind && m_slots[i].instance == slot.instance && m_slots[i].backendInstance == slot.backendInstance)
			return static_cast<QueueSlotIndex>(static_cast<uint8_t>(i));
	}
	return static_cast<QueueSlotIndex>(0xFF);
}

bool QueueRegistry::HasSlot(QueueSlot slot) const {
	for (auto& s : m_slots) {
		if (s.kind == slot.kind && s.instance == slot.instance && s.backendInstance == slot.backendInstance) return true;
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
	for (auto& slot : m_slots) {
		if (slot.ownsQueue && slot.device && slot.queue) {
			slot.device.DestroyQueue(slot.queue.GetQueueHandle());
		}
	}
	m_slots.clear();
}


} // namespace org
