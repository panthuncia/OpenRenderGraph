#pragma once

#include <BasicTelemetry/Telemetry.h>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string_view>

namespace org {

// Detailed pass/resource zones are opt-in. Normal compilation neither copies
// diagnostic names into the IR nor emits these high-volume events.
inline bool CompileDetailTracingEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ORG_COMPILE_TRACE_PASSES");
        return value && value[0] == '1';
    }();
    return enabled;
}

class CompileDetailScope {
public:
    CompileDetailScope(const basic_telemetry::Callsite& callsite,
        std::string_view label, uint64_t workItems,
        bool enabled = CompileDetailTracingEnabled()) {
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
