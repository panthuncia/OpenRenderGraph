#pragma once

#include <atomic>
#include <memory>
#include <utility>

namespace org::runtime {

// CPU destruction only. Submitted owners must first be retained until their
// actual GPU completion. No fence snapshot is taken by this queue.
class ResourceCleanupQueue : public std::enable_shared_from_this<ResourceCleanupQueue> {
public:
    static std::shared_ptr<ResourceCleanupQueue> Create();
    ~ResourceCleanupQueue();
    ResourceCleanupQueue(const ResourceCleanupQueue&) = delete;
    ResourceCleanupQueue& operator=(const ResourceCleanupQueue&) = delete;

    template<class T, class... Args>
    std::shared_ptr<T> Make(Args&&... args) {
        struct Node final : RetirementNode {
            explicit Node(Args&&... values) : value(std::forward<Args>(values)...) {
                destroy = [](RetirementNode* node) noexcept { delete static_cast<Node*>(node); };
            }
            T value;
        };
        auto queue = shared_from_this();
        auto* node = new Node(std::forward<Args>(args)...);
        // shared_ptr calls the deleter if allocating its control block fails.
        return std::shared_ptr<T>(&node->value, [queue = std::move(queue), node](T*) noexcept { queue->Post(node); });
    }

    // Explicit quiescent/test boundary only; never an ordinary frame operation.
    // Includes destruction cascades posted by nodes already being destroyed.
    void Drain();
    uint64_t Pending() const noexcept;

private:
    struct RetirementNode {
        RetirementNode* next = nullptr;
        void (*destroy)(RetirementNode*) noexcept = nullptr;
    };
    struct State;
    ResourceCleanupQueue();
    void Post(RetirementNode*) noexcept;
    std::shared_ptr<State> m_state;
};

} // namespace org::runtime
