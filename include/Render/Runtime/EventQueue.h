#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace org::runtime {

// What a queue does with its storage and its events when it is destroyed.
enum class QueueTeardown : uint8_t {
    Free,  // frees its ring and destroys the events still in it
    Leak,  // frees nothing: for queues that live as long as the process and hold objects that must not be released at its end
};

// A multi-producer, single-consumer queue. Producers on any thread push without blocking and without allocating while the ring
// has room; a push to a full ring spills onto a lock-free stack. No event is ever lost.
// The one consumer drains (Drain) or pops one at a time (Pop). It sees the ring's events in push order, then the spilled ones in
// push order. While spilled events wait, later pushes spill too, so one producer's events stay in push order. A push still being
// written is seen by a later call.
// A bounded Vyukov ring. T must be default-constructible and movable.
template <class T, std::size_t Capacity = 4096, QueueTeardown Teardown = QueueTeardown::Free>
class EventQueue {
    static_assert(Capacity && (Capacity & (Capacity - 1)) == 0, "a power of two");

public:
    EventQueue() {
        for (std::size_t i = 0; i < Capacity; ++i) m_cells[i].sequence.store(i, std::memory_order_relaxed);
    }
    // onPush runs after every push, on the pushing thread, to wake the consumer.
    explicit EventQueue(void (*onPush)()) : EventQueue() { m_onPush = onPush; }
    ~EventQueue() {
        if constexpr (Teardown == QueueTeardown::Free) {
            for (auto* node = m_popped; node;) {
                auto* next = node->next;
                delete node;
                node = next;
            }
            for (auto* node = m_spill.load(std::memory_order_acquire); node;) {
                auto* next = node->next;
                delete node;
                node = next;
            }
            delete[] m_cells;
        }
    }
    EventQueue(const EventQueue&) = delete;
    EventQueue& operator=(const EventQueue&) = delete;

    void Push(T value) {
        PushOnly(std::move(value));
        if (m_onPush) m_onPush();
    }

    // The one consumer. Visits every event pushed before the call, oldest first, and returns how many. Bounded by the ring's
    // head at the call, so a drain always ends even while producers keep pushing.
    template <class F>
    std::size_t Drain(F&& visit) {
        std::size_t count = 0;
        const std::uint64_t bound = m_head.load(std::memory_order_acquire);
        while (m_tail != bound) {
            auto& cell = m_cells[m_tail & (Capacity - 1)];
            if (cell.sequence.load(std::memory_order_acquire) != m_tail + 1) break;
            visit(std::move(cell.value));
            cell.value = T{};
            cell.sequence.store(m_tail + Capacity, std::memory_order_release);
            ++m_tail;
            ++count;
        }
        TakeSpill();
        std::uint64_t spilled = 0;
        for (auto* node = m_popped; node;) {
            visit(std::move(node->value));
            auto* next = node->next;
            delete node;
            node = next;
            ++count;
            ++spilled;
        }
        m_popped = nullptr;
        m_spillWaiting.fetch_sub(spilled, std::memory_order_acq_rel);
        m_count.fetch_sub(count, std::memory_order_acq_rel);
        return count;
    }
    // The one consumer. Takes the oldest event into `out`. False when there is none (or the next one is still being written).
    bool Pop(T& out) {
        auto& cell = m_cells[m_tail & (Capacity - 1)];
        if (cell.sequence.load(std::memory_order_acquire) == m_tail + 1) {
            out = std::move(cell.value);
            cell.value = T{};
            cell.sequence.store(m_tail + Capacity, std::memory_order_release);
            ++m_tail;
            m_count.fetch_sub(1, std::memory_order_acq_rel);
            return true;
        }
        if (!m_popped) TakeSpill();
        if (!m_popped) return false;
        auto* node = m_popped;
        m_popped = node->next;
        out = std::move(node->value);
        delete node;
        m_spillWaiting.fetch_sub(1, std::memory_order_acq_rel);
        m_count.fetch_sub(1, std::memory_order_acq_rel);
        return true;
    }
    // The one consumer: drops every event pushed before the call.
    void Discard() {
        Drain([](T&&) {});
    }
    // Any thread. The events pushed and not yet taken. A push in flight counts already, before the consumer can take it.
    std::uint64_t Count() const noexcept { return m_count.load(std::memory_order_acquire); }
    bool Empty() const noexcept { return Count() == 0; }
    // Pushes that found the ring full since the start.
    std::uint64_t Spilled() const noexcept { return m_spilled.load(std::memory_order_relaxed); }

private:
    void PushOnly(T value) {
        // Spilled events not yet taken: this push follows them onto the stack, or the consumer would take it from the ring first.
        if (m_spillWaiting.load(std::memory_order_acquire) != 0) {
            Spill(std::move(value));
            return;
        }
        std::uint64_t position = m_head.load(std::memory_order_relaxed);
        for (;;) {
            auto& cell = m_cells[position & (Capacity - 1)];
            const std::uint64_t sequence = cell.sequence.load(std::memory_order_acquire);
            const auto difference = static_cast<std::int64_t>(sequence) - static_cast<std::int64_t>(position);
            if (difference == 0) {
                if (m_head.compare_exchange_weak(position, position + 1, std::memory_order_relaxed)) {
                    // Counted before it is published, so the consumer never takes it before the count holds it.
                    m_count.fetch_add(1, std::memory_order_acq_rel);
                    cell.value = std::move(value);
                    cell.sequence.store(position + 1, std::memory_order_release);
                    return;
                }
            } else if (difference < 0) {
                // Full, so spill. Never drop.
                Spill(std::move(value));
                return;
            } else {
                position = m_head.load(std::memory_order_relaxed);
            }
        }
    }
    void Spill(T value) {
        auto* node = new SpillNode{std::move(value), m_spill.load(std::memory_order_relaxed)};
        m_count.fetch_add(1, std::memory_order_acq_rel);
        m_spillWaiting.fetch_add(1, std::memory_order_acq_rel);
        while (!m_spill.compare_exchange_weak(node->next, node, std::memory_order_release, std::memory_order_relaxed)) {
        }
        m_spilled.fetch_add(1, std::memory_order_relaxed);
    }
    // The consumer's: appends the spilled events, in push order, to the ones it took before.
    void TakeSpill() {
        SpillNode* reversed = nullptr;
        for (auto* node = m_spill.exchange(nullptr, std::memory_order_acquire); node;) {
            auto* next = node->next;
            node->next = reversed;
            reversed = node;
            node = next;
        }
        if (!reversed) return;
        if (!m_popped) {
            m_popped = reversed;
            return;
        }
        auto* tail = m_popped;
        while (tail->next) tail = tail->next;
        tail->next = reversed;
    }

    struct Cell {
        std::atomic<std::uint64_t> sequence{0};
        T value{};
    };
    struct SpillNode {
        T value;
        SpillNode* next;
    };
    Cell* m_cells = new Cell[Capacity];
    std::atomic<std::uint64_t> m_head{0};
    std::uint64_t m_tail = 0;     // the consumer's
    SpillNode* m_popped = nullptr;  // the consumer's: spilled events taken from the stack, in push order
    std::atomic<SpillNode*> m_spill{nullptr};
    std::atomic<std::uint64_t> m_spillWaiting{0};  // spilled events not yet taken
    std::atomic<std::uint64_t> m_count{0};
    std::atomic<std::uint64_t> m_spilled{0};
    void (*m_onPush)() = nullptr;  // set at construction, before the queue is shared
};

} // namespace org::runtime
