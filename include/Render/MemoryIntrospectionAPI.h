#pragma once

#include <cstdint>
#include <mutex>
#include <memory>
#include <string>
#include <vector>

#include "Interfaces/IHasMemoryMetadata.h"

namespace org::memory {

struct ResourceMemoryRecord {
    uint64_t resourceID = 0;
    uint64_t bytes = 0;
    rhi::ResourceType resourceType = rhi::ResourceType::Unknown;
    // The allocation's heap: Upload and Readback are host memory on a discrete GPU, the rest (GPUUpload too) device memory.
    rhi::HeapType heapType = rhi::HeapType::DeviceLocal;
    std::string resourceName;
    std::string usage;
    std::string identifier;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipLevels = 0;
    uint32_t arraySize = 0;
    rhi::Format format = rhi::Format::Unknown;
    bool aliased = false;
    // Who the memory is for (SetResourceOwner); '/' nests levels.
    std::string owner;
    // The aliasing pool the record belongs to: its placements and the pool's own memory. 0 for none.
    uint64_t aliasPoolID = 0;
    // Placed in an aliasing pool's memory: the pool's record already counts these bytes.
    bool placedInAliasPool = false;

    // Whether the bytes add to a total; placements in an aliasing pool would count the pool's memory twice.
    bool CountsTowardTotal() const { return !placedInAliasPool; }
};

class IMemorySnapshotProvider {
public:
    virtual ~IMemorySnapshotProvider() = default;
    virtual void BuildSnapshot(std::vector<ResourceMemoryRecord>& out) = 0;
};

class SnapshotProvider {
public:
    void SetProvider(std::shared_ptr<IMemorySnapshotProvider> provider);
    void ResetProvider();
    void BuildSnapshot(std::vector<ResourceMemoryRecord>& out) const;

private:
    mutable std::mutex m_providerMutex;
    std::shared_ptr<IMemorySnapshotProvider> m_provider;
};

inline void SetResourceUsageHint(IHasMemoryMetadata& resource, std::string usage) {
    resource.SetMemoryUsageHint(std::move(usage));
}

inline void SetResourceMemoryIdentifier(IHasMemoryMetadata& resource, std::string identifier) {
    resource.SetMemoryIdentifier(std::move(identifier));
}

inline void SetResourceOwner(IHasMemoryMetadata& resource, std::string owner) {
    resource.SetMemoryOwner(std::move(owner));
}

}
