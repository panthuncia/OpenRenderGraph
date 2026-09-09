#pragma once

#include <BasicTelemetry/Telemetry.h>

#include <optional>

namespace org::profile
{
class ScopedCompileProfileStep
{
public:
	explicit ScopedCompileProfileStep(const char* stepName);
	~ScopedCompileProfileStep() = default;

	ScopedCompileProfileStep(const ScopedCompileProfileStep&) = delete;
	ScopedCompileProfileStep& operator=(const ScopedCompileProfileStep&) = delete;

private:
	std::optional<basic_telemetry::Scope> m_scope;
};
}
