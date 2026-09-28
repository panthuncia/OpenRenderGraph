#include "Render/Runtime/IDescriptorService.h"

#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Managers/Singletons/DeviceManager.h"

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

	void RetireDescriptorSlotWithOwner(rhi::DescriptorSlot slot, std::shared_ptr<const void> owner) override {
		DescriptorHeapManager::GetInstance().RetireDescriptorSlotWithOwner(slot, std::move(owner));
	}

    UINT CreateIndexedSampler(const rhi::SamplerDesc& samplerDesc) override {
        return DescriptorHeapManager::GetInstance().CreateIndexedSampler(samplerDesc);
    }
    OwnedDescriptorBinding CreateOwnedShaderResourceView(std::shared_ptr<const void> deviceOwner,
        rhi::Resource resource, std::shared_ptr<const void> backingOwner, const rhi::SrvDesc& description) override {
        auto& manager = DescriptorHeapManager::GetInstance();
        return OwnedDescriptorBinding::CreateShaderResourceView(DeviceManager::GetInstance().GetDevice(),
            std::move(deviceOwner), manager.DeviceGeneration(), manager.GetCBVSRVUAVHeap(), manager.GetResourceCleanupQueue(),
            resource, std::move(backingOwner), description);
    }
    OwnedDescriptorBinding CreateOwnedSampler(std::shared_ptr<const void> deviceOwner,
        const rhi::SamplerDesc& description) override {
        auto& manager = DescriptorHeapManager::GetInstance();
        return OwnedDescriptorBinding::CreateSampler(DeviceManager::GetInstance().GetDevice(), std::move(deviceOwner),
            manager.DeviceGeneration(), manager.GetSamplerHeap(), manager.GetResourceCleanupQueue(), description);
    }
    std::shared_ptr<ResourceCleanupQueue> GetResourceCleanupQueue() const override {
        return DescriptorHeapManager::GetInstance().GetResourceCleanupQueue();
    }
};
} // namespace

std::shared_ptr<IDescriptorService> CreateDefaultDescriptorService() {
    return std::make_shared<DefaultDescriptorService>();
}

} // namespace org::runtime
