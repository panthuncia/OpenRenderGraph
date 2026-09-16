#pragma once

#include <memory>
#include <memory_resource>
#include <utility>

namespace org {
// Shared control blocks retain their allocator. No slot reset can recycle
// memory underneath CPU recording, a submitted frame, or its lifecycle effects.
class PreparedInvocationArena {
    template<class T> struct Allocator {
        using value_type = T;
        std::shared_ptr<std::pmr::synchronized_pool_resource> resource;
        explicit Allocator(std::shared_ptr<std::pmr::synchronized_pool_resource> r) : resource(std::move(r)) {}
        template<class U> Allocator(const Allocator<U>& other) : resource(other.resource) {}
        T* allocate(size_t n) { return static_cast<T*>(resource->allocate(n * sizeof(T), alignof(T))); }
        void deallocate(T* p, size_t n) { resource->deallocate(p, n * sizeof(T), alignof(T)); }
        template<class U> bool operator==(const Allocator<U>& other) const noexcept { return resource == other.resource; }
    };
public:
    template<class T, class... Args> std::shared_ptr<T> MakeShared(Args&&... args) const {
        return std::allocate_shared<T>(Allocator<T>{m_resource}, std::forward<Args>(args)...);
    }
private:
    std::shared_ptr<std::pmr::synchronized_pool_resource> m_resource =
        std::make_shared<std::pmr::synchronized_pool_resource>();
};
}
