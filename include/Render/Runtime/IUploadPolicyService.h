#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace org::runtime {

class IUploadService;

class IUploadPolicyClient {
public:
    virtual ~IUploadPolicyClient() = default;
    virtual void OnUploadPolicyBeginFrame() = 0;
    virtual void OnUploadPolicyFlush() = 0;
    virtual bool HasPendingUploadPolicyWork() const { return false; }
    virtual std::string GetUploadPolicyDebugName() const { return {}; }
    virtual uint64_t GetUploadPolicyLastFlushWrites() const { return 0; }
    virtual uint64_t GetUploadPolicyLastFlushBytes() const { return 0; }

    void ConfigureUploadService(const std::shared_ptr<IUploadService>& service) { m_uploadService = service; }

protected:
    std::shared_ptr<IUploadService> RetainUploadService() const { return m_uploadService.lock(); }

private:
    std::weak_ptr<IUploadService> m_uploadService;
};

struct UploadPolicyServiceStats {
    uint64_t beginFrameCalls = 0;
    uint64_t flushCalls = 0;
    uint64_t registeredClients = 0;
    uint64_t dirtyClientsFlushed = 0;
    uint64_t flushedWrites = 0;
    uint64_t flushedBytes = 0;
};

class IUploadPolicyService {
public:
    virtual ~IUploadPolicyService() = default;

    virtual void Initialize() = 0;
    virtual void Cleanup() = 0;
    virtual void SetUploadService(std::shared_ptr<IUploadService> service) = 0;

    virtual void RegisterClient(IUploadPolicyClient* client) = 0;
    virtual void UnregisterClient(IUploadPolicyClient* client) = 0;
    virtual void MarkClientDirty(IUploadPolicyClient* client) = 0;

    virtual void BeginFrame() = 0;
    virtual void FlushAll() = 0;

    virtual UploadPolicyServiceStats GetStats() const = 0;
};

std::shared_ptr<IUploadPolicyService> CreateDefaultUploadPolicyService(std::shared_ptr<IUploadService> uploadService);

}
