#include "Render/Runtime/PublicationChannels.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace org::runtime {
namespace {

uint64_t Key(rhi::ResourceHandle handle) noexcept { return (uint64_t{handle.generation} << 32) | handle.index; }

struct Entry {
    rhi::ResourceHandle channel;
    uint64_t serial = 0;  // the binding that made it, so releasing an older one keeps a newer one
};

struct Registry {
    std::mutex mutex;
    std::unordered_map<uint64_t, Entry> byBacking;
    uint64_t nextSerial = 0;
    std::atomic<size_t> count{0};
};

Registry& Get() {
    static Registry registry;
    return registry;
}

struct Binding {
    uint64_t backing = 0;
    uint64_t serial = 0;
    ~Binding() {
        auto& registry = Get();
        std::lock_guard lock(registry.mutex);
        const auto found = registry.byBacking.find(backing);
        if (found == registry.byBacking.end() || found->second.serial != serial) return;
        registry.byBacking.erase(found);
        registry.count.store(registry.byBacking.size(), std::memory_order_release);
    }
};

} // namespace

std::shared_ptr<void> PublicationChannels::Bind(rhi::ResourceHandle backing, rhi::ResourceHandle channel) {
    if (!backing.valid() || !channel.valid()) return nullptr;
    auto& registry = Get();
    auto binding = std::make_shared<Binding>();
    binding->backing = Key(backing);
    std::lock_guard lock(registry.mutex);
    binding->serial = ++registry.nextSerial;
    registry.byBacking[binding->backing] = {channel, binding->serial};
    registry.count.store(registry.byBacking.size(), std::memory_order_release);
    return binding;
}

rhi::ResourceHandle PublicationChannels::ChannelOf(rhi::ResourceHandle backing) noexcept {
    auto& registry = Get();
    if (registry.count.load(std::memory_order_acquire) == 0) return {};
    std::lock_guard lock(registry.mutex);
    const auto found = registry.byBacking.find(Key(backing));
    return found == registry.byBacking.end() ? rhi::ResourceHandle{} : found->second.channel;
}

bool PublicationChannels::Any() noexcept { return Get().count.load(std::memory_order_acquire) != 0; }

} // namespace org::runtime
