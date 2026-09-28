#include "Render/Runtime/ResourceCleanupQueue.h"

#include <thread>

namespace org::runtime {
struct ResourceCleanupQueue::State {
    std::atomic<RetirementNode*> head{nullptr};
    std::atomic<uint64_t> pending{0}, wake{0};
    std::atomic<bool> closing{false};

    void Notify() noexcept {
        wake.fetch_add(1, std::memory_order_release);
        wake.notify_one();
    }
    void Run() noexcept {
        for (;;) {
            const auto observed = wake.load(std::memory_order_acquire);
            auto* nodes = head.exchange(nullptr, std::memory_order_acquire);
            while (nodes) {
                auto* node = nodes;
                nodes = node->next;
                node->destroy(node);
                pending.fetch_sub(1, std::memory_order_acq_rel);
                pending.notify_all();
            }
            if (closing.load(std::memory_order_acquire) && !pending.load(std::memory_order_acquire)) return;
            if (!head.load(std::memory_order_acquire)) wake.wait(observed, std::memory_order_acquire);
        }
    }
};

ResourceCleanupQueue::ResourceCleanupQueue() : m_state(std::make_shared<State>()) {}
std::shared_ptr<ResourceCleanupQueue> ResourceCleanupQueue::Create() {
    auto result = std::shared_ptr<ResourceCleanupQueue>(new ResourceCleanupQueue());
    // The thread owns State, not the queue: dropping the last public owner
    // merely wakes it. There is no thread join in a final-reference deleter.
    std::thread([state = result->m_state] { state->Run(); }).detach();
    return result;
}
ResourceCleanupQueue::~ResourceCleanupQueue() {
    m_state->closing.store(true, std::memory_order_release);
    m_state->Notify();
}
void ResourceCleanupQueue::Post(RetirementNode* node) noexcept {
    m_state->pending.fetch_add(1, std::memory_order_relaxed);
    auto* head = m_state->head.load(std::memory_order_relaxed);
    do { node->next = head; }
    while (!m_state->head.compare_exchange_weak(head, node, std::memory_order_release, std::memory_order_relaxed));
    m_state->Notify();
}
uint64_t ResourceCleanupQueue::Pending() const noexcept { return m_state->pending.load(std::memory_order_acquire); }
void ResourceCleanupQueue::Drain() {
    for (auto count = Pending(); count; count = Pending()) m_state->pending.wait(count, std::memory_order_acquire);
}
} // namespace org::runtime
