#pragma once

#include "Render/PreparedPass.h"

namespace org::runtime {

// A service reserves a signal and owns its rollback. Typed passes only retain
// the reservation; the framework places the signal after the recorded work.
class ExternalSignalReservation final : public PreparedLifecycleEffect {
public:
    ExternalSignalReservation(std::shared_ptr<const rhi::TimelinePtr> owner,
        uint64_t value, std::function<void()> cancelled)
        : m_owner(std::move(owner)), m_cancelled(std::move(cancelled)) {
        if (!m_owner || !*m_owner)
            throw std::invalid_argument("External signal reservation requires an owned timeline");
        m_signal = {m_owner->Get(), value};
    }
    ~ExternalSignalReservation() override { Cancel(); }
    std::span<const ExternalTimelinePoint> SignalsAfterCompletion() const override {
        return {&m_signal, 1};
    }
    void Submitted(SubmissionContext) const override { m_resolved.exchange(true); }
    void Abandoned(AbandonReason) const override { Cancel(); }
private:
    void Cancel() const {
        if (!m_resolved.exchange(true) && m_cancelled) m_cancelled();
    }
    std::shared_ptr<const rhi::TimelinePtr> m_owner;
    ExternalTimelinePoint m_signal{};
    std::function<void()> m_cancelled;
    mutable std::atomic<bool> m_resolved{false};
};

} // namespace org::runtime
