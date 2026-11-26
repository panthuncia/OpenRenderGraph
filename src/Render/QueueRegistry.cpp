#include "Render/QueueRegistry.h"
#include "Render/CommandListPool.h"

#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <spdlog/spdlog.h>
#include <rhi_interop_dx12.h>
#include <rhi_interop_vulkan.h>


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

	// The command list pools of slots on shared timelines, one per device and queue kind: every graph of a host records from
	// them. A pool keeps at least one command pool, and each costs the driver about 2 MB (RX 6600), so one pool per graph made
	// one per epoch. The pools are thread-safe. Held weakly: the last graph or recording that holds one frees it.
	std::shared_ptr<CommandListPool> SharedPoolFor(rhi::Device& device, rhi::QueueKind kind) {
		static std::mutex mutex;
		static std::map<std::pair<const void*, int>, std::weak_ptr<CommandListPool>> pools;
		std::lock_guard lock(mutex);
		auto& entry = pools[{ device.impl, static_cast<int>(kind) }];
		if (auto pool = entry.lock()) return pool;
		auto pool = std::make_shared<CommandListPool>(device, kind);
		entry = pool;
		return pool;
	}
}

QueueSlotIndex QueueRegistry::Register(QueueSlot slot, rhi::Queue queue, rhi::Device& device, QueueAutoAssignmentPolicy autoAssignmentPolicy, bool ownsQueue, std::string_view logicalName) {
	auto pool = std::make_unique<CommandListPool>(device, static_cast<rhi::QueueKind>(slot.kind));
	rhi::TimelinePtr fence;
	device.CreateTimeline(fence);
	return Register(slot, queue, std::move(fence), std::move(pool), autoAssignmentPolicy, ownsQueue, device, logicalName);
}

QueueSlotIndex QueueRegistry::Register(QueueSlot slot, rhi::Queue queue, rhi::TimelinePtr fence, std::unique_ptr<CommandListPool> pool, QueueAutoAssignmentPolicy autoAssignmentPolicy, bool ownsQueue, rhi::Device device, std::string_view logicalName) {
	return RegisterWithPool(slot, queue, std::move(fence), std::shared_ptr<CommandListPool>(std::move(pool)), autoAssignmentPolicy, ownsQueue, device, logicalName);
}

QueueSlotIndex QueueRegistry::RegisterWithPool(QueueSlot slot, rhi::Queue queue, rhi::TimelinePtr fence, std::shared_ptr<CommandListPool> pool, QueueAutoAssignmentPolicy autoAssignmentPolicy, bool ownsQueue, rhi::Device device, std::string_view logicalName) {
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
	m_slots.push_back({ slot.kind, slot.instance, slot.backendInstance, slot.backend, queue, device, std::make_shared<rhi::TimelinePtr>(std::move(fence)), {}, std::move(pool), autoAssignmentPolicy, ownsQueue, std::string(logicalName) });
	return idx;
}

QueueSlotIndex QueueRegistry::Register(QueueSlot slot, rhi::Queue queue, const SharedQueueTimeline& timeline, rhi::Device& device,
	QueueAutoAssignmentPolicy autoAssignmentPolicy) {
	if (!timeline) throw std::invalid_argument("QueueRegistry::Register: an empty shared timeline");
	const auto idx = RegisterWithPool(slot, queue, rhi::TimelinePtr{}, SharedPoolFor(device, static_cast<rhi::QueueKind>(slot.kind)),
		autoAssignmentPolicy, false, device, {});
	auto& entry = m_slots.back();
	entry.sharedPool = true;
	entry.fence = timeline.fence;
	entry.fenceValue = timeline.nextValue;
	entry.sharedTimeline = true;
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

rhi::Result CreateBridgeTimelinePair(rhi::Device first, rhi::Backend firstBackend, rhi::Device second, rhi::Backend secondBackend,
	const char* name, rhi::TimelinePtr& onFirst, rhi::TimelinePtr& onSecond) {
#ifdef _WIN32
	const bool firstIsD3D12 = firstBackend == rhi::Backend::D3D12 && secondBackend == rhi::Backend::Vulkan;
	const bool firstIsVulkan = firstBackend == rhi::Backend::Vulkan && secondBackend == rhi::Backend::D3D12;
	if (!first || !second || (!firstIsD3D12 && !firstIsVulkan)) return rhi::Result::Unsupported;
	auto d3d12Device = firstIsD3D12 ? first : second;
	auto vulkanDevice = firstIsD3D12 ? second : first;
	rhi::TimelinePtr d3dFence;
	auto result = d3d12Device.CreateTimeline(d3dFence, 0, name, true);
	if (rhi::Failed(result)) return result;
	rhi::ExternalHandle shared{};
	result = rhi::dx12::export_shared_timeline(d3d12Device, d3dFence.Get(), shared);
	if (rhi::Failed(result)) return result;
	rhi::TimelinePtr vkFence;
	result = rhi::vulkan::import_d3d12_timeline(vulkanDevice, shared, 0, name, vkFence);
	if (rhi::Failed(result)) return result;
	onFirst = firstIsD3D12 ? std::move(d3dFence) : std::move(vkFence);
	onSecond = firstIsD3D12 ? std::move(vkFence) : std::move(d3dFence);
	return rhi::Result::Ok;
#else
	(void)first; (void)firstBackend; (void)second; (void)secondBackend; (void)name; (void)onFirst; (void)onSecond;
	return rhi::Result::Unsupported;
#endif
}

rhi::Result QueueRegistry::EnableD3D12VulkanInterop(rhi::Device d3d12Device, rhi::Device vulkanDevice) {
#ifdef _WIN32
	if (!d3d12Device || !vulkanDevice) return rhi::Result::InvalidArgument;
	for (size_t i = 0; i < m_slots.size(); ++i) {
		auto& slot = m_slots[i];
		if (slot.backend != rhi::Backend::D3D12 && slot.backend != rhi::Backend::Vulkan) continue;
		// Setup runs again on every structural rebuild, and timelines survive those: replacing one would destroy it while the
		// runtime's ledgers still wait on it, and restart its values. Only slots added since the last Setup need converting.
		if (slot.peerFence) continue;
		rhi::TimelinePtr d3dFence;
		auto result = d3d12Device.CreateTimeline(d3dFence, 0, "ORG Multi-RHI Bridge Fence", true);
		if (rhi::Failed(result)) return result;
		rhi::ExternalHandle shared{};
		result = rhi::dx12::export_shared_timeline(d3d12Device, d3dFence.Get(), shared);
		if (rhi::Failed(result)) return result;
		rhi::TimelinePtr vkFence;
		result = rhi::vulkan::import_d3d12_timeline(vulkanDevice, shared, 0, "ORG Multi-RHI Bridge Timeline", vkFence);
		if (rhi::Failed(result)) return result;
		if (slot.backend == rhi::Backend::D3D12) {
			slot.fence = std::make_shared<rhi::TimelinePtr>(std::move(d3dFence));
			slot.peerFence = std::move(vkFence);
		} else {
			slot.fence = std::make_shared<rhi::TimelinePtr>(std::move(vkFence));
			slot.peerFence = std::move(d3dFence);
		}
		slot.fenceValue = std::make_shared<std::atomic<uint64_t>>(1);
		slot.sharedTimeline = false;
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

QueueSlotIndex QueueRegistry::PhysicalSlot(QueueSlotIndex i) const noexcept {
	const auto& slot = m_slots[ToUnderlying(i)];
	if (!slot.queue) return i;
	const auto handle = slot.queue.GetQueueHandle();
	for (size_t earlier = 0; earlier < ToUnderlying(i); ++earlier) {
		const auto& other = m_slots[earlier];
		if (other.backendInstance != slot.backendInstance || !other.queue) continue;
		const auto otherHandle = other.queue.GetQueueHandle();
		if (otherHandle.index == handle.index && otherHandle.generation == handle.generation)
			return static_cast<QueueSlotIndex>(static_cast<uint8_t>(earlier));
	}
	return i;
}

void QueueRegistry::Clear() {
	ShutdownTaskWorkers();
	for (auto& slot : m_slots) {
		if (slot.ownsQueue && slot.device && slot.queue) {
			slot.device.DestroyQueue(slot.queue.GetQueueHandle());
		}
	}
	m_slots.clear();
}

void QueueRegistry::ShutdownTaskWorkers() {
	for (auto& slot : m_slots) {
		// Other graphs still record from a shared pool: it stops when its last holder frees it.
		if (slot.pool && !slot.sharedPool) {
			slot.pool->ShutdownBackgroundReset();
		}
	}
}


} // namespace org
