#pragma once

#pragma once

#include <memory>
#include <shared_mutex>
#include <string>
#include <rhi.h>
#include "Resources/Resource.h"
#include "Resources/GloballyIndexedResource.h"


namespace org {

class DynamicResource : public Resource {
public:
    DynamicResource(std::shared_ptr<Resource> initialResource)
        : resource(std::move(initialResource)) {
        if (resource) {
            //currentState = resource->GetState();
            name = resource->GetName();
            m_hasLayout = resource->HasLayout();
            m_mipLevels = resource->GetMipLevels();
            m_arraySize = resource->GetArraySize();
        }
    }

    // Allow swapping the underlying resource dynamically
    void SetResource(std::shared_ptr<Resource> newResource) {
        if (!newResource) {
            throw std::runtime_error("Cannot set a null resource.");
        }

        std::unique_lock lock(resourceMutex);
        resource = std::move(newResource);
        //currentState = resource->GetState();
        name = resource->GetName();
        m_hasLayout = resource->HasLayout();
        m_mipLevels = resource->GetMipLevels();
        m_arraySize = resource->GetArraySize();
    }

    std::shared_ptr<Resource> GetResource() const {
        std::shared_lock lock(resourceMutex);
        return resource;
    }

    rhi::BarrierBatch GetEnhancedBarrierGroup(RangeSpec range, rhi::ResourceAccessType prevAccessType, rhi::ResourceAccessType newAccessType, rhi::ResourceLayout prevLayout, rhi::ResourceLayout newLayout, rhi::ResourceSyncState prevSyncState, rhi::ResourceSyncState newSyncState) override {
        auto backing = GetResource();
        if (backing) {
            return backing->GetEnhancedBarrierGroup(range, prevAccessType, newAccessType, prevLayout, newLayout, prevSyncState, newSyncState);
        }
        return {};
    }

    rhi::Resource GetAPIResource() override {
		return GetResource()->GetAPIResource();
	}
    bool HasResource() const {
        return static_cast<bool>(GetResource());
	}

    virtual uint64_t GetGlobalResourceID() const override {
        if (auto backing = GetResource()) {
            return backing->GetGlobalResourceID();
        }
        return m_globalResourceID;
	}

    uint64_t GetDynamicWrapperGlobalResourceID() const {
        return Resource::GetGlobalResourceID();
    }
	uint64_t GetSchedulingResourceID() const override {
		return GetDynamicWrapperGlobalResourceID();
	}

    SymbolicTracker* GetStateTracker() override {
        return GetResource()->GetStateTracker();
    }
    bool TryGetRHIResourceDesc(rhi::ResourceDesc& outDesc) const override {
        auto backing = GetResource();
        return backing && backing->TryGetRHIResourceDesc(outDesc);
    }

protected:
    void OnSetName() override {
        if (auto backing = GetResource()) {
            backing->SetName(name);
        }
    }

private:
    mutable std::shared_mutex resourceMutex;
    std::shared_ptr<Resource> resource; // T actual resource
};

class DynamicGloballyIndexedResource : public GloballyIndexedResourceBase {
public:
    DynamicGloballyIndexedResource(std::shared_ptr<GloballyIndexedResource> initialResource)
        : m_resource(std::move(initialResource)) {
        if (m_resource) {
            //currentState = m_resource->GetState();
            name = m_resource->GetName();
        }
    }

    // Allow swapping the underlying resource dynamically
    void SetResource(std::shared_ptr<GloballyIndexedResource> newResource) {
        if (!newResource) {
            throw std::runtime_error("Cannot set a null resource.");
        }
        std::unique_lock lock(m_resourceMutex);
        m_resource = std::move(newResource);
        //currentState = m_resource->GetState();
        name = m_resource->GetName();
    }

    std::shared_ptr<GloballyIndexedResource> GetResource() const {
        std::shared_lock lock(m_resourceMutex);
        return m_resource;
    }

    rhi::BarrierBatch GetEnhancedBarrierGroup(RangeSpec range, rhi::ResourceAccessType prevAccessType, rhi::ResourceAccessType newAccessType, rhi::ResourceLayout prevLayout, rhi::ResourceLayout newLayout, rhi::ResourceSyncState prevSyncState, rhi::ResourceSyncState newSyncState) override {
		if (m_resource) {
			//SetState(newState); // Keep the wrapper's state in sync
			return m_resource->GetEnhancedBarrierGroup(range, prevAccessType, newAccessType, prevLayout, newLayout, prevSyncState, newSyncState);
		}
        return {};
    }

    rhi::Resource GetAPIResource() override {
        return m_resource->GetAPIResource();
    }
    bool HasResource() const {
        return m_resource != nullptr;
    }
    SymbolicTracker* GetStateTracker() override {
        return m_resource->GetStateTracker();
    }
protected:
    void OnSetName() override {
        if (m_resource) {
            m_resource->SetName(name);
        }
    }

private:
    mutable std::shared_mutex m_resourceMutex;
    std::shared_ptr<GloballyIndexedResource> m_resource; // actual resource
};


} // namespace org
