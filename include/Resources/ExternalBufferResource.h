#pragma once

#include <memory>
#include <rhi.h>

#include "Resources/GloballyIndexedResource.h"
#include "Resources/ResourceStateTracker.h"
#include "Render/Runtime/DescriptorServiceTypes.h"

namespace org {

// Retained wrapper for externally allocated buffers. Structural size changes
// require a graph rebuild; native identity can be refreshed in place.
class ExternalBufferResource final : public GloballyIndexedResource {
public:
	using ViewRequirements = runtime::DescriptorViewRequirements::BufferViews;
	static std::shared_ptr<ExternalBufferResource> CreateShared(
		rhi::ResourcePtr resource, uint64_t byteSize, ViewRequirements views = {});
	bool RefreshShared(rhi::ResourcePtr resource, uint64_t byteSize, const ViewRequirements& views);
	void ResetState(ResourceState state);

	rhi::Resource GetAPIResource() override { return m_resource.Get(); }
	rhi::BarrierBatch GetEnhancedBarrierGroup(RangeSpec,
		rhi::ResourceAccessType, rhi::ResourceAccessType,
		rhi::ResourceLayout, rhi::ResourceLayout,
		rhi::ResourceSyncState, rhi::ResourceSyncState) override;
	SymbolicTracker* GetStateTracker() override { return &m_stateTracker; }
	bool TryGetBufferByteSize(uint64_t& out) const override { out = m_byteSize; return true; }

private:
	ExternalBufferResource(rhi::ResourcePtr resource, uint64_t byteSize, ViewRequirements views);
	rhi::ResourcePtr m_resource;
	uint64_t m_byteSize{};
	ViewRequirements m_views{};
	rhi::BufferBarrier m_barrier{};
	SymbolicTracker m_stateTracker{};
};

} // namespace org
