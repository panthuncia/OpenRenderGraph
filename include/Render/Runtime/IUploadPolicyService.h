#pragma once

#include <cstdint>
#include <memory>

namespace rg::runtime {

class IUploadPolicyClient {
public:
    virtual ~IUploadPolicyClient() = default;
    virtual void OnUploadPolicyBeginFrame() = 0;
    virtual void OnUploadPolicyFlush() = 0;
};

struct UploadPolicyServiceStats {
    uint64_t beginFrameCalls = 0;
    uint64_t flushCalls = 0;
    uint64_t registeredClients = 0;
};

class IUploadPolicyService {
public:
    virtual ~IUploadPolicyService() = default;

    virtual void Initialize() = 0;
    virtual void Cleanup() = 0;

    virtual void RegisterClient(IUploadPolicyClient* client) = 0;
    virtual void UnregisterClient(IUploadPolicyClient* client) = 0;

    virtual void BeginFrame() = 0;
    virtual void FlushAll() = 0;

    virtual UploadPolicyServiceStats GetStats() const = 0;
};

std::shared_ptr<IUploadPolicyService> CreateDefaultUploadPolicyService();

}
