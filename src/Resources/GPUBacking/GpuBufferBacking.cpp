#include "Resources/GPUBacking/GpuBufferBacking.h"

#include <rhi_helpers.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

#include "Managers/Singletons/DeviceManager.h"
#include "Managers/Singletons/DeletionManager.h"
#include "Resources/MemoryStatisticsComponents.h"

GpuBufferBacking::GpuBufferBacking(
	const rhi::HeapType accessType,
	const uint64_t bufferSize,
	uint64_t owningResourceID,
	const bool unorderedAccess,
    const char* name,
    const BufferAliasPlacement* aliasPlacement) {
    m_accessType = accessType;

    rhi::ResourceDesc desc = rhi::helpers::ResourceDesc::Buffer(bufferSize);
    if (unorderedAccess) {
        desc.resourceFlags |= rhi::ResourceFlags::RF_AllowUnorderedAccess;
    }
    desc.heapType = accessType;
    auto device = DeviceManager::GetInstance().GetDevice();

    rhi::ResourceAllocationInfo allocInfo;
    device.GetResourceAllocationInfo(&desc, 1, &allocInfo);

    AllocationTrackDesc trackDesc(owningResourceID);
    EntityComponentBundle allocationBundle;
    if (name != nullptr) {
        allocationBundle.Set<MemoryStatisticsComponents::ResourceName>({ name });
    }

    allocationBundle
        .Set<MemoryStatisticsComponents::MemSizeBytes>({ allocInfo.sizeInBytes })
        .Set<MemoryStatisticsComponents::ResourceType>({ rhi::ResourceType::Buffer })
        .Set<MemoryStatisticsComponents::ResourceID>({ owningResourceID });
    if (aliasPlacement && aliasPlacement->poolID.has_value()) {
        allocationBundle.Set<MemoryStatisticsComponents::AliasingPool>({ aliasPlacement->poolID });
    }
    trackDesc.attach = allocationBundle;

    if (aliasPlacement && aliasPlacement->allocation) {
        const auto result = DeviceManager::GetInstance().CreateAliasingResourceTracked(
            *aliasPlacement->allocation,
            aliasPlacement->offset,
            desc,
            0,
            nullptr,
            m_bufferAllocation,
            trackDesc);
        if (!rhi::IsOk(result)) {
            throw std::runtime_error("Failed to create aliased buffer resource backing");
        }
    }
    else {
        rhi::ma::AllocationDesc allocationDesc;
        allocationDesc.heapType = accessType;
        const auto result = DeviceManager::GetInstance().CreateResourceTracked(
            allocationDesc,
            desc,
            0,
            nullptr,
            m_bufferAllocation,
            trackDesc);
        if (!rhi::IsOk(result)) {
            throw std::runtime_error("Failed to create committed buffer resource backing");
        }
    }

    m_size = bufferSize;

    RegisterLiveAlloc();
    UpdateLiveAllocName(name);
}

GpuBufferBacking::~GpuBufferBacking() {
    UnregisterLiveAlloc();
    if (m_bufferAllocation) {
        DeletionManager::GetInstance().MarkForDelete(std::move(m_bufferAllocation));
    }
}

void GpuBufferBacking::SetName(const char* name)
{
	m_bufferAllocation.ApplyComponentBundle(EntityComponentBundle().Set<MemoryStatisticsComponents::ResourceName>({ name }));
	m_bufferAllocation.GetResource().SetName(name);
    UpdateLiveAllocName(name);
}

rhi::BarrierBatch GpuBufferBacking::GetEnhancedBarrierGroup(RangeSpec range, rhi::ResourceAccessType prevAccessType, rhi::ResourceAccessType newAccessType, rhi::ResourceLayout prevLayout, rhi::ResourceLayout newLayout, rhi::ResourceSyncState prevSyncState, rhi::ResourceSyncState newSyncState) {

	rhi::BarrierBatch batch = {};
	m_barrier = rhi::BufferBarrier{
	   .buffer = GetAPIResource().GetHandle(),
	   .offset = 0,
	   .size = UINT64_MAX,
	   .beforeSync = prevSyncState,
	   .afterSync = newSyncState,
	   .beforeAccess = prevAccessType,
	   .afterAccess = newAccessType
	};
	batch.buffers = { &m_barrier };

	return batch;
}

std::mutex& GpuBufferBacking::LiveAllocMutex() {
	static auto* mutex = new std::mutex();
	return *mutex;
}

std::unordered_map<const GpuBufferBacking*, GpuBufferBacking::LiveAllocInfo>& GpuBufferBacking::LiveAllocs() {
	static auto* liveAllocs = new std::unordered_map<const GpuBufferBacking*, LiveAllocInfo>();
	return *liveAllocs;
}

void GpuBufferBacking::RegisterLiveAlloc() {
    auto& liveAllocs = LiveAllocs();
    std::scoped_lock lock(LiveAllocMutex());
    LiveAllocInfo info{};
    info.size = m_size;
    //info.uav = m_bufferAllocation.GetResource().IsValid() && m_bufferAllocation.GetResource().GetDesc().resourceFlags & rhi::ResourceFlags::RF_AllowUnorderedAccess;
    liveAllocs[this] = info;
}

void GpuBufferBacking::UnregisterLiveAlloc() {
    auto& liveAllocs = LiveAllocs();
    std::scoped_lock lock(LiveAllocMutex());
	liveAllocs.erase(this);
}

void GpuBufferBacking::UpdateLiveAllocName(const char* name) {
    auto& liveAllocs = LiveAllocs();
    std::scoped_lock lock(LiveAllocMutex());
    auto it = liveAllocs.find(this);
    if (it != liveAllocs.end()) {
        it->second.name = name ? name : "";
    }
}

unsigned int GpuBufferBacking::DumpLiveBuffers() {
    auto& liveAllocs = LiveAllocs();
    std::scoped_lock lock(LiveAllocMutex());
    for (const auto& [ptr, info] : liveAllocs) {
        spdlog::warn("Live buffer still tracked: size={} bytes, name='{}'", info.size, info.name);
    }
	return static_cast<unsigned int>(liveAllocs.size());
}
