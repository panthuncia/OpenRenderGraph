#include "Resources/ExternalBufferResource.h"

#include <cstring>

#include "Managers/Singletons/DescriptorHeapManager.h"

namespace org {

ExternalBufferResource::ExternalBufferResource(
	rhi::ResourcePtr resource, uint64_t byteSize, ViewRequirements views)
	: m_resource(std::move(resource)), m_byteSize(byteSize), m_views(std::move(views))
{
	m_hasLayout = false;
	m_mipLevels = 1;
	m_arraySize = 1;
}

std::shared_ptr<ExternalBufferResource> ExternalBufferResource::CreateShared(
	rhi::ResourcePtr resource, uint64_t byteSize, ViewRequirements views)
{
	if (!resource || !byteSize) return {};
	auto result = std::shared_ptr<ExternalBufferResource>(
		new ExternalBufferResource(std::move(resource), byteSize, std::move(views)));
	runtime::DescriptorViewRequirements requirements{};
	requirements.views = result->m_views;
	auto apiResource = result->m_resource.Get();
	DescriptorHeapManager::GetInstance().AssignDescriptorSlots(*result, apiResource, requirements);
	return result;
}

bool ExternalBufferResource::RefreshShared(
	rhi::ResourcePtr resource, uint64_t byteSize, const ViewRequirements& views)
{
	if (!resource || byteSize != m_byteSize || views.createCBV != m_views.createCBV ||
		views.createSRV != m_views.createSRV || views.createUAV != m_views.createUAV ||
		views.createNonShaderVisibleUAV != m_views.createNonShaderVisibleUAV ||
		memcmp(&views.cbvDesc, &m_views.cbvDesc, sizeof(views.cbvDesc)) != 0 ||
		memcmp(&views.srvDesc, &m_views.srvDesc, sizeof(views.srvDesc)) != 0 ||
		memcmp(&views.uavDesc, &m_views.uavDesc, sizeof(views.uavDesc)) != 0 ||
		views.uavCounterOffset != m_views.uavCounterOffset) return false;
	auto previous = std::move(m_resource);
	m_resource = std::move(resource);
	m_stateTracker = SymbolicTracker{};
	runtime::DescriptorViewRequirements requirements{};
	requirements.views = m_views;
	auto apiResource = m_resource.Get();
	DescriptorHeapManager::GetInstance().UpdateDescriptorContents(*this, apiResource, requirements);
	DescriptorHeapManager::GetInstance().RetireNativeResource(std::move(previous));
	return true;
}

void ExternalBufferResource::ResetState(ResourceState state)
{
	RangeSpec wholeRange{};
	m_stateTracker = SymbolicTracker(wholeRange, state);
}

rhi::BarrierBatch ExternalBufferResource::GetEnhancedBarrierGroup(RangeSpec,
	rhi::ResourceAccessType previousAccess, rhi::ResourceAccessType nextAccess,
	rhi::ResourceLayout, rhi::ResourceLayout,
	rhi::ResourceSyncState previousSync, rhi::ResourceSyncState nextSync)
{
	thread_local rhi::BufferBarrier barrier{};
	barrier = {
		.buffer = m_resource.Get().GetHandle(),
		.offset = 0,
		.size = UINT64_MAX,
		.beforeSync = previousSync,
		.afterSync = nextSync,
		.beforeAccess = previousAccess,
		.afterAccess = nextAccess
	};
	rhi::BarrierBatch result{};
	result.buffers = { &barrier };
	return result;
}

} // namespace org
