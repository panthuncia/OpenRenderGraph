#pragma once

#include <memory>

#include <rhi.h>

#include "Render/Runtime/DescriptorServiceTypes.h"

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
    virtual UINT CreateIndexedSampler(const rhi::SamplerDesc& samplerDesc) = 0;
};

std::shared_ptr<IDescriptorService> CreateDefaultDescriptorService();

} // namespace org::runtime
