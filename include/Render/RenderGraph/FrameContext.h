#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <vector>

namespace org {

struct FrameCompletionPoint {
    uint64_t timeline = 0;
    uint64_t value = 0;
    bool operator==(const FrameCompletionPoint&) const = default;
};

// Every queue that touched the frame must complete. A graphics-only frame
// fence is sufficient only when submission explicitly joined all other queues.
class CompletionSet {
public:
    void Include(FrameCompletionPoint point) {
        if (!point.timeline || !point.value || point.value == UINT64_MAX)
            throw std::invalid_argument("Invalid frame completion point");
        auto found = std::ranges::find(m_points, point.timeline, &FrameCompletionPoint::timeline);
        if (found == m_points.end()) m_points.push_back(point);
        else found->value = (std::max)(found->value, point.value);
    }
    bool IsComplete(std::span<const FrameCompletionPoint> observed) const noexcept {
        for (auto required : m_points) {
            const auto found = std::ranges::find(observed, required.timeline, &FrameCompletionPoint::timeline);
            if (found == observed.end() || found->value == UINT64_MAX || found->value < required.value)
                return false;
        }
        return true;
    }
    std::span<const FrameCompletionPoint> Points() const noexcept { return m_points; }
private:
    std::vector<FrameCompletionPoint> m_points;
};

enum class FrameStage : uint8_t {
    Preparing, Compiling, Planned, Recording, Ready, Submitted, Retired, Cancelled, Recovery
};

// The pool stores weak leases: a slot cannot be reused merely because it has
// no submitted fence yet. The lease also covers pending CPU tasks and ready work.
class FrameSlotLease {
    friend class FrameSlotPool;
    explicit FrameSlotLease(uint32_t slot) : m_slot(slot) {}
public:
    uint32_t Index() const noexcept { return m_slot; }
private:
    uint32_t m_slot;
};

class FrameSlotPool {
public:
    explicit FrameSlotPool(uint32_t capacity) : m_slots(capacity) {
        if (!capacity || capacity > 64) throw std::invalid_argument("Invalid frame slot capacity");
    }
    std::shared_ptr<const FrameSlotLease> TryAcquire(uint32_t slot) {
        std::scoped_lock lock(m_mutex);
        if (slot >= m_slots.size()) throw std::out_of_range("Frame preparation slot");
        if (!m_slots[slot].expired()) return {};
        auto lease = std::shared_ptr<const FrameSlotLease>(new FrameSlotLease(slot));
        m_slots[slot] = lease;
        return lease;
    }
    size_t Active() const {
        std::scoped_lock lock(m_mutex);
        return std::ranges::count_if(m_slots, [](const auto& slot) { return !slot.expired(); });
    }
private:
    mutable std::mutex m_mutex;
    std::vector<std::weak_ptr<const FrameSlotLease>> m_slots;
};

// Shared by CPU jobs, the ready queue and the GPU retirement owner. Callers
// release unsubmitted frames only after joining their jobs. Submission failure
// keeps this object recovery-owned until the device is known to be quiescent.
class FrameContext {
public:
    FrameContext(uint64_t frame, uint64_t generation, std::shared_ptr<const FrameSlotLease> slot)
        : m_frame(frame), m_generation(generation), m_slot(std::move(slot)) {
        if (!frame || !generation || !m_slot) throw std::invalid_argument("Incomplete frame identity");
    }
    uint64_t Number() const noexcept { return m_frame; }
    uint64_t Generation() const noexcept { return m_generation; }
    uint32_t Slot() const noexcept { return m_slot->Index(); }
    FrameStage Stage() const noexcept { return m_stage.load(std::memory_order_acquire); }
    std::vector<FrameCompletionPoint> CompletionPoints() const {
        std::scoped_lock lock(m_mutex);
        const auto points = m_completion.Points();
        return {points.begin(), points.end()};
    }
    void Advance(FrameStage expected, FrameStage next) {
        if (static_cast<unsigned>(next) != static_cast<unsigned>(expected) + 1
            || expected >= FrameStage::Ready)
            throw std::logic_error("Invalid frame stage transition");
        if (!m_stage.compare_exchange_strong(expected, next, std::memory_order_acq_rel))
            throw std::logic_error("Out-of-order frame stage transition");
    }
    void Retain(std::shared_ptr<const void> owner) {
        if (!owner) throw std::invalid_argument("Empty frame dependency");
        std::scoped_lock lock(m_mutex);
        if (Stage() >= FrameStage::Ready) throw std::logic_error("Frame dependencies already frozen");
        m_owners.push_back(std::move(owner));
    }
    void MarkSubmitted(CompletionSet completion) {
        if (completion.Points().empty()) throw std::invalid_argument("Submitted frame has no completion point");
        std::scoped_lock lock(m_mutex);
        auto expected = FrameStage::Ready;
        if (!m_stage.compare_exchange_strong(expected, FrameStage::Submitted))
            throw std::logic_error("Frame is not ready for submission");
        m_completion = std::move(completion);
    }
    bool Retire(std::span<const FrameCompletionPoint> observed) {
        std::scoped_lock lock(m_mutex);
        if (Stage() != FrameStage::Submitted || !m_completion.IsComplete(observed)) return false;
        m_stage.store(FrameStage::Retired, std::memory_order_release);
        return true;
    }
    void CancelAfterJoin() {
        std::scoped_lock lock(m_mutex);
        if (Stage() >= FrameStage::Submitted)
            throw std::logic_error("Submitted frame cannot be cancelled");
        m_stage.store(FrameStage::Cancelled, std::memory_order_release);
    }
    void EnterRecovery() {
        std::scoped_lock lock(m_mutex);
        if (Stage() != FrameStage::Ready) throw std::logic_error("Invalid recovery transition");
        m_stage.store(FrameStage::Recovery, std::memory_order_release);
    }
private:
    const uint64_t m_frame;
    const uint64_t m_generation;
    // Declared before owners so the slot is released after its dependencies.
    const std::shared_ptr<const FrameSlotLease> m_slot;
    std::atomic<FrameStage> m_stage{FrameStage::Preparing};
    mutable std::mutex m_mutex;
    CompletionSet m_completion;
    std::vector<std::shared_ptr<const void>> m_owners;
};

} // namespace org
