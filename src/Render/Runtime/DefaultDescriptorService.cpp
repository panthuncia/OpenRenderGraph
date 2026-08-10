#include "Render/Runtime/IDescriptorService.h"

#include "Managers/Singletons/DescriptorHeapManager.h"

namespace org::runtime {

namespace {
class DefaultDescriptorService final : public IDescriptorService {
public:
    void Initialize() override {
        DescriptorHeapManager::GetInstance().Initialize();
    }

    void Cleanup() override {
        DescriptorHeapManager::GetInstance().Cleanup();
    }

    void AssignDescriptorSlots(
        GloballyIndexedResource& target,
        rhi::Resource& apiResource,
        const DescriptorViewRequirements& req) override {
        DescriptorHeapManager::GetInstance().AssignDescriptorSlots(target, apiResource, req);
    }

    void ReserveDescriptorSlots(
        GloballyIndexedResource& target,
        const DescriptorViewRequirements& req) override {
        DescriptorHeapManager::GetInstance().ReserveDescriptorSlots(target, req);
    }

    void UpdateDescriptorContents(
        GloballyIndexedResource& target,
        rhi::Resource& apiResource,
        const DescriptorViewRequirements& req) override {
        DescriptorHeapManager::GetInstance().UpdateDescriptorContents(target, apiResource, req);
    }

    rhi::DescriptorHeap GetSRVDescriptorHeap() const override {
        return DescriptorHeapManager::GetInstance().GetSRVDescriptorHeap();
    }

    rhi::DescriptorHeap GetSamplerDescriptorHeap() const override {
        return DescriptorHeapManager::GetInstance().GetSamplerDescriptorHeap();
    }

	rhi::DescriptorHeap GetRTVDescriptorHeap() const override {
		return DescriptorHeapManager::GetInstance().GetRTVHeap()->GetHeap();
	}

	rhi::DescriptorHeap GetDSVDescriptorHeap() const override {
		return DescriptorHeapManager::GetInstance().GetDSVHeap()->GetHeap();
	}

	rhi::DescriptorHeap GetNonShaderVisibleDescriptorHeap() const override {
		return DescriptorHeapManager::GetInstance().GetNonShaderVisibleHeap()->GetHeap();
	}

	rhi::DescriptorSlot AllocateDescriptorSlot(rhi::DescriptorHeapType type, bool shaderVisible) override {
		return DescriptorHeapManager::GetInstance().AllocateDescriptorSlot(type, shaderVisible);
	}

	void RetireDescriptorSlot(rhi::DescriptorSlot slot) override {
		DescriptorHeapManager::GetInstance().RetireDescriptorSlot(slot);
	}

    UINT CreateIndexedSampler(const rhi::SamplerDesc& samplerDesc) override {
        return DescriptorHeapManager::GetInstance().CreateIndexedSampler(samplerDesc);
    }
};
} // namespace

std::shared_ptr<IDescriptorService> CreateDefaultDescriptorService() {
    return std::make_shared<DefaultDescriptorService>();
}

} // namespace org::runtime
