#pragma once

#include <BasicTelemetry/Telemetry.h>

#include <optional>

namespace rg::profile
{
class ScopedCompileProfileStep
{
public:
	explicit ScopedCompileProfileStep(const char* stepName);
	~ScopedCompileProfileStep() = default;

	ScopedCompileProfileStep(const ScopedCompileProfileStep&) = delete;
	ScopedCompileProfileStep& operator=(const ScopedCompileProfileStep&) = delete;

private:
	basic_telemetry::Callsite m_callsite;
	std::optional<basic_telemetry::Scope> m_scope;
};
}
