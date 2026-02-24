#include "Render/Runtime/IUploadPolicyService.h"

#include <mutex>
#include <unordered_set>
#include <vector>

namespace rg::runtime {

namespace {

class DefaultUploadPolicyService final : public IUploadPolicyService {
public:
    void Initialize() override {}

    void Cleanup() override {
        std::scoped_lock lock(m_mutex);
        m_clients.clear();
        m_stats = {};
    }

    void RegisterClient(IUploadPolicyClient* client) override {
        if (!client) {
            return;
        }

        std::scoped_lock lock(m_mutex);
        m_clients.insert(client);
        m_stats.registeredClients = static_cast<uint64_t>(m_clients.size());
    }

    void UnregisterClient(IUploadPolicyClient* client) override {
        if (!client) {
            return;
        }

        std::scoped_lock lock(m_mutex);
        m_clients.erase(client);
        m_stats.registeredClients = static_cast<uint64_t>(m_clients.size());
    }

    void BeginFrame() override {
        auto clients = SnapshotClients();
        for (auto* client : clients) {
            if (client) {
                client->OnUploadPolicyBeginFrame();
            }
        }

        std::scoped_lock lock(m_mutex);
        ++m_stats.beginFrameCalls;
        m_stats.registeredClients = static_cast<uint64_t>(m_clients.size());
    }

    void FlushAll() override {
        auto clients = SnapshotClients();
        for (auto* client : clients) {
            if (client) {
                client->OnUploadPolicyFlush();
            }
        }

        std::scoped_lock lock(m_mutex);
        ++m_stats.flushCalls;
        m_stats.registeredClients = static_cast<uint64_t>(m_clients.size());
    }

    UploadPolicyServiceStats GetStats() const override {
        std::scoped_lock lock(m_mutex);
        return m_stats;
    }

private:
    std::vector<IUploadPolicyClient*> SnapshotClients() {
        std::scoped_lock lock(m_mutex);
        std::vector<IUploadPolicyClient*> out;
        out.reserve(m_clients.size());
        for (auto* client : m_clients) {
            out.push_back(client);
        }
        return out;
    }

    mutable std::mutex m_mutex;
    std::unordered_set<IUploadPolicyClient*> m_clients;
    UploadPolicyServiceStats m_stats{};
};

}

std::shared_ptr<IUploadPolicyService> CreateDefaultUploadPolicyService() {
    return std::make_shared<DefaultUploadPolicyService>();
}

}
