#pragma once

#include <memory>

#include <rhi.h>

#include "Render/Runtime/DescriptorServiceTypes.h"
#include "Render/OwnedDescriptorBinding.h"
#include "Render/Runtime/ResourceCleanupQueue.h"

namespace org { class GloballyIndexedResource; }

namespace org::runtime {

class IDescriptorService {
public:
    virtual ~IDescriptorService() = default;

    virtual void Initialize() = 0;
    virtual void Cleanup() = 0;

    virtual void AssignDescriptorSlots(
        GloballyIndexedResource& target,
        rhi::Resource& apiResource,
        const DescriptorViewRequirements& req) = 0;

    virtual void ReserveDescriptorSlots(
        GloballyIndexedResource& target,
        const DescriptorViewRequirements& req) = 0;

    virtual void UpdateDescriptorContents(
        GloballyIndexedResource& target,
        rhi::Resource& apiResource,
        const DescriptorViewRequirements& req) = 0;

    virtual rhi::DescriptorHeap GetSRVDescriptorHeap() const = 0;
    virtual rhi::DescriptorHeap GetSamplerDescriptorHeap() const = 0;
	virtual rhi::DescriptorHeap GetRTVDescriptorHeap() const = 0;
	virtual rhi::DescriptorHeap GetDSVDescriptorHeap() const = 0;
	virtual rhi::DescriptorHeap GetNonShaderVisibleDescriptorHeap() const = 0;
	// Extension-owned exact views still allocate from ORG's global arenas and
	// retire against ORG's queue-fence snapshot.
	virtual rhi::DescriptorSlot AllocateDescriptorSlot(rhi::DescriptorHeapType type, bool shaderVisible) = 0;
	virtual void RetireDescriptorSlot(rhi::DescriptorSlot slot) = 0;
	// Retire an extension-owned view and its backing at the same GPU completion
	// point as its descriptor. Outstanding CPU descriptor leases also retain the
	// owner; its final release runs outside descriptor locks.
	virtual void RetireDescriptorSlotWithOwner(rhi::DescriptorSlot slot, std::shared_ptr<const void> owner) = 0;
    virtual UINT CreateIndexedSampler(const rhi::SamplerDesc& samplerDesc) = 0;
    // New owned path. Compatibility services may reject it without changing
    // their existing raw-slot behavior. The host supplies real device ownership.
    virtual OwnedDescriptorBinding CreateOwnedShaderResourceView(std::shared_ptr<const void>,
        rhi::Resource, std::shared_ptr<const void>, const rhi::SrvDesc&) { return {}; }
    virtual OwnedDescriptorBinding CreateOwnedSampler(std::shared_ptr<const void>, const rhi::SamplerDesc&) { return {}; }
    virtual std::shared_ptr<ResourceCleanupQueue> GetResourceCleanupQueue() const { return {}; }
};

std::shared_ptr<IDescriptorService> CreateDefaultDescriptorService();

} // namespace org::runtime
