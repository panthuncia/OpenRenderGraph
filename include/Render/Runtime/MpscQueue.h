#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

namespace org::runtime {

// Intrusive multi-producer single-consumer queue (Vyukov). Push is wait-free
// for any number of producers; Pop must only be called by one consumer at a
// time. Used where a producer must never block on the consumer: worker
// uploads posted to the render thread's upload instance, and copies posted to
// the copy-queue submitter.
template <typename T>
class MpscQueue {
public:
    MpscQueue() {
        m_stub = new Node();
        m_head.store(m_stub, std::memory_order_relaxed);
        m_tail = m_stub;
    }
    ~MpscQueue() {
        Node* node = m_tail;
        while (node) {
            Node* next = node->next.load(std::memory_order_relaxed);
            delete node;
            node = next;
        }
    }
    MpscQueue(const MpscQueue&) = delete;
    MpscQueue& operator=(const MpscQueue&) = delete;

    void Push(T&& value) {
        auto* node = new Node();
        node->value = std::move(value);
        node->next.store(nullptr, std::memory_order_relaxed);
        Node* previous = m_head.exchange(node, std::memory_order_acq_rel);
        previous->next.store(node, std::memory_order_release);
        m_count.fetch_add(1, std::memory_order_acq_rel);
    }

    // Consumer only. Returns false when empty or when a producer is mid-push
    // (the value becomes visible on a later call).
    bool Pop(T& out) {
        Node* tail = m_tail;
        Node* next = tail->next.load(std::memory_order_acquire);
        if (tail == m_stub) {
            if (!next) return false;
            m_tail = next;
            tail = next;
            next = next->next.load(std::memory_order_acquire);
        }
        if (next) {
            out = std::move(tail->value);
            m_tail = next;
            delete tail;
            m_count.fetch_sub(1, std::memory_order_acq_rel);
            return true;
        }
        Node* head = m_head.load(std::memory_order_acquire);
        if (tail != head) return false;
        m_stub->next.store(nullptr, std::memory_order_relaxed);
        Node* previous = m_head.exchange(m_stub, std::memory_order_acq_rel);
        previous->next.store(m_stub, std::memory_order_release);
        next = tail->next.load(std::memory_order_acquire);
        if (!next) return false;
        out = std::move(tail->value);
        m_tail = next;
        delete tail;
        m_count.fetch_sub(1, std::memory_order_acq_rel);
        return true;
    }

    // Approximate: exact once every in-flight push has completed.
    uint64_t Count() const noexcept { return m_count.load(std::memory_order_acquire); }
    bool Empty() const noexcept { return Count() == 0; }

private:
    struct Node {
        std::atomic<Node*> next{ nullptr };
        T value{};
    };
    std::atomic<Node*> m_head{ nullptr };
    Node* m_tail = nullptr;
    Node* m_stub = nullptr;
    std::atomic<uint64_t> m_count{ 0 };
};

} // namespace org::runtime
