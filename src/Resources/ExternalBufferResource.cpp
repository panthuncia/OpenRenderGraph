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

namespace {
	// Field-wise: the descriptors contain unions and padding, whose bytes are not
	// preserved by member-wise copies, so memcmp would reject identical views.
	bool SameBufferViews(const ExternalBufferResource::ViewRequirements& a, const ExternalBufferResource::ViewRequirements& b) {
		const auto sameSrv = [](const rhi::SrvDesc& x, const rhi::SrvDesc& y) {
			return x.dimension == y.dimension && x.formatOverride == y.formatOverride && x.componentMapping == y.componentMapping
				&& x.buffer.kind == y.buffer.kind && x.buffer.firstElement == y.buffer.firstElement
				&& x.buffer.numElements == y.buffer.numElements && x.buffer.structureByteStride == y.buffer.structureByteStride;
		};
		const auto sameUav = [](const rhi::UavDesc& x, const rhi::UavDesc& y) {
			return x.dimension == y.dimension && x.formatOverride == y.formatOverride
				&& x.buffer.kind == y.buffer.kind && x.buffer.firstElement == y.buffer.firstElement
				&& x.buffer.numElements == y.buffer.numElements && x.buffer.structureByteStride == y.buffer.structureByteStride
				&& x.buffer.counterOffsetInBytes == y.buffer.counterOffsetInBytes;
		};
		return a.createCBV == b.createCBV && a.createSRV == b.createSRV && a.createUAV == b.createUAV
			&& a.createNonShaderVisibleUAV == b.createNonShaderVisibleUAV && a.uavCounterOffset == b.uavCounterOffset
			&& (!a.createCBV || (a.cbvDesc.byteOffset == b.cbvDesc.byteOffset && a.cbvDesc.byteSize == b.cbvDesc.byteSize))
			&& (!a.createSRV || sameSrv(a.srvDesc, b.srvDesc))
			&& (!(a.createUAV || a.createNonShaderVisibleUAV) || sameUav(a.uavDesc, b.uavDesc));
	}
}

bool ExternalBufferResource::RefreshShared(
	rhi::ResourcePtr resource, uint64_t byteSize, const ViewRequirements& views)
{
	if (!resource || byteSize != m_byteSize || !SameBufferViews(views, m_views)) return false;
	auto previous = std::move(m_resource);
	RotateDescriptorSlotsForPublication();
	m_resource = std::move(resource);
	m_stateTracker = SymbolicTracker{};
	runtime::DescriptorViewRequirements requirements{};
	requirements.views = m_views;
	auto apiResource = m_resource.Get();
	DescriptorHeapManager::GetInstance().AssignDescriptorSlots(*this, apiResource, requirements);
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
