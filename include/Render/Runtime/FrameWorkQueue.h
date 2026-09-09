#pragma once

#include "Render/PreparedPass.h"
#include <algorithm>
#include <list>
#include <mutex>

namespace org::runtime {

// Service-owned work, independent of pass and graph generations. A reservation
// consumes work on submission, or returns it in publication order on CPU abort.
// Work payloads contain owned inputs; ordinary recording never receives this queue.
template<class Work>
class FrameWorkQueue {
public:
    struct Counters { uint64_t accepted = 0, pending = 0, reserved = 0, submitted = 0, returned = 0, discarded = 0; };
    struct Item { uint64_t sequence; Work work; mutable bool discarded = false; };
    using Entry = std::shared_ptr<const Item>;
    using Snapshot = std::vector<Entry>;
private:
    struct State {
        std::mutex mutex;
        std::list<Entry> pending;
        std::unordered_map<uint64_t, std::weak_ptr<const Item>> known;
        uint64_t nextSequence = 1;
        Counters counters;
    };
    struct Reservation final : PreparedLifecycleEffect {
        explicit Reservation(std::shared_ptr<State> state) : state(std::move(state)) {}
        ~Reservation() override { ReturnUnsubmitted(); }
        void Submitted(SubmissionContext) const override {
            if (resolved.exchange(true)) return;
            {
                std::lock_guard lock(state->mutex);
                state->counters.reserved -= work.size();
                state->counters.submitted += work.size();
                for (const auto& entry : work) state->known.erase(entry->sequence);
            }
            if constexpr (requires(const Work& value) { value.Commit(); })
                for (const auto& entry : work) if (!entry->discarded) entry->work.Commit();
        }
        void Abandoned(AbandonReason) const override { ReturnUnsubmitted(); }
        void ReturnUnsubmitted() const {
            if (resolved.exchange(true)) return;
            std::lock_guard lock(state->mutex);
            const auto count = work.size();
            state->counters.reserved -= count;
            work.remove_if([](const Entry& entry) { return entry->discarded; });
            state->counters.discarded += count - work.size();
            state->counters.returned += work.size();
            state->pending.merge(work, [](const Entry& a, const Entry& b) {
                return a->sequence < b->sequence;
            });
        }
        std::shared_ptr<State> state;
        mutable std::list<Entry> work;
        mutable std::atomic<bool> resolved{false};
    };
public:
    void Enqueue(Work work) {
        std::lock_guard lock(m_state->mutex);
        auto entry = std::make_shared<const Item>(Item{m_state->nextSequence++, std::move(work)});
        m_state->pending.push_back(entry);
        m_state->known.emplace(entry->sequence, entry);
        ++m_state->counters.accepted;
    }
    Counters ReadCounters() const {
        std::lock_guard lock(m_state->mutex);
        auto counters = m_state->counters;
        counters.pending = m_state->pending.size();
        return counters;
    }
    Snapshot Pending() const {
        std::lock_guard lock(m_state->mutex);
        return {m_state->pending.begin(), m_state->pending.end()};
    }
    void Reserve(const Snapshot& selected, const FramePreparationContext& preparation) const {
        if (selected.empty()) return;
        auto reservation = std::make_shared<Reservation>(m_state);
        {
            std::lock_guard lock(m_state->mutex);
            // Validate the entire selection before removing any work.
            uint64_t previous = 0;
            for (const auto& entry : selected) {
                if (!entry || entry->sequence <= previous ||
                    std::find(m_state->pending.begin(), m_state->pending.end(), entry) == m_state->pending.end())
                    throw std::logic_error("Stale or duplicate service work reservation");
                previous = entry->sequence;
            }
            for (const auto& entry : selected) {
                auto found = std::find(m_state->pending.begin(), m_state->pending.end(), entry);
                reservation->work.splice(reservation->work.end(), m_state->pending, found);
            }
            m_state->counters.reserved += selected.size();
        }
        // If publication throws, destruction returns the removed nodes without allocation.
        preparation.Reserve(std::move(reservation));
    }
    // Prevent obsolete jobs from returning after a frame is cancelled. This
    // does not revoke already recorded GPU work; callers still drain/join it.
    template<class Predicate>
    void DiscardUnsubmitted(Predicate predicate) {
        Snapshot discarded;
        {
            std::lock_guard lock(m_state->mutex);
            discarded.reserve(m_state->known.size());
            for (auto it = m_state->known.begin(); it != m_state->known.end();) {
                auto entry = it->second.lock();
                if (!entry || predicate(entry->work)) {
                    if (entry) { entry->discarded = true; discarded.push_back(entry); }
                    it = m_state->known.erase(it);
                } else ++it;
            }
            const auto previous = m_state->pending.size();
            m_state->pending.remove_if([](const Entry& entry) { return entry->discarded; });
            m_state->counters.discarded += previous - m_state->pending.size();
        }
        if constexpr (requires(const Work& value) { value.Discard(); })
            for (const auto& entry : discarded) entry->work.Discard();
    }

private:
    std::shared_ptr<State> m_state = std::make_shared<State>();
};

} // namespace org::runtime
