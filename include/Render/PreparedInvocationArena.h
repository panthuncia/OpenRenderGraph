#pragma once

#include <memory>
#include <memory_resource>
#include <utility>
#include <atomic>
#include <cstddef>

namespace org {
// Shared control blocks retain their allocator. No slot reset can recycle
// memory underneath CPU recording, a submitted frame, or its lifecycle effects.
class PreparedInvocationArena {
    struct Upstream final : std::pmr::memory_resource {
        std::atomic_size_t allocations{0}, bytes{0}, liveBytes{0}, peakBytes{0};
        void* do_allocate(size_t count, size_t alignment) override {
            auto* result = std::pmr::new_delete_resource()->allocate(count,alignment);
            allocations.fetch_add(1,std::memory_order_relaxed);
            bytes.fetch_add(count,std::memory_order_relaxed);
            const auto live = liveBytes.fetch_add(count,std::memory_order_relaxed) + count;
            auto peak = peakBytes.load(std::memory_order_relaxed);
            while (peak < live && !peakBytes.compare_exchange_weak(peak,live,std::memory_order_relaxed)) {}
            return result;
        }
        void do_deallocate(void* pointer, size_t count, size_t alignment) override {
            std::pmr::new_delete_resource()->deallocate(pointer,count,alignment);
            liveBytes.fetch_sub(count,std::memory_order_relaxed);
        }
        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
    };
    struct State {
        Upstream upstream;
        // Destruction order keeps the upstream alive until the pool releases
        // all of its chunks. Allocators retain this entire state.
        std::pmr::synchronized_pool_resource resource{&upstream};
    };
    template<class T> struct Allocator {
        using value_type = T;
        std::shared_ptr<State> resource;
        explicit Allocator(std::shared_ptr<State> r) : resource(std::move(r)) {}
        template<class U> Allocator(const Allocator<U>& other) : resource(other.resource) {}
        T* allocate(size_t n) { return static_cast<T*>(resource->resource.allocate(n * sizeof(T), alignof(T))); }
        void deallocate(T* p, size_t n) { resource->resource.deallocate(p, n * sizeof(T), alignof(T)); }
        template<class U> bool operator==(const Allocator<U>& other) const noexcept { return resource == other.resource; }
    };
public:
    struct Statistics { size_t allocations, allocatedBytes, liveBytes, peakBytes; };
    Statistics MemoryStatistics() const noexcept {
        const auto& counters = m_resource->upstream;
        return {counters.allocations.load(std::memory_order_relaxed),counters.bytes.load(std::memory_order_relaxed),
            counters.liveBytes.load(std::memory_order_relaxed),counters.peakBytes.load(std::memory_order_relaxed)};
    }
    // Owner-thread maintenance after preparation/recording tasks join. Shared
    // allocator control blocks (including weak-only blocks) prevent release.
    // Normal frame preparation keeps capacity and never calls this operation.
    bool ReleaseUnusedStorage() {
        if (m_resource.use_count() != 1) return false;
        m_resource->resource.release();
        return true;
    }
    template<class T, class... Args> std::shared_ptr<T> MakeShared(Args&&... args) const {
        return std::allocate_shared<T>(Allocator<T>{m_resource}, std::forward<Args>(args)...);
    }
private:
    std::shared_ptr<State> m_resource = std::make_shared<State>();
};
}
