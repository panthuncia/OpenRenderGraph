#pragma once

#include <BasicTelemetry/Telemetry.h>
#include <cstdint>
#include <optional>
#include <string_view>

namespace org {

// Detailed pass/resource zones, on only where the compile input carries diagnostic names. Normal compilation neither copies
// diagnostic names into the IR nor emits these high-volume events.
class CompileDetailScope {
public:
    CompileDetailScope(const basic_telemetry::Callsite& callsite,
        std::string_view label, uint64_t workItems,
        bool enabled) {
        if (enabled) {
            m_scope.emplace(callsite);
            m_scope->Text(label);
            m_scope->Value(workItems);
        }
    }
private:
    std::optional<basic_telemetry::Scope> m_scope;
};

} // namespace org
