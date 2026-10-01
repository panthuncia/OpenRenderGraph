#pragma once

#include <cstdlib>

namespace org::detail {

// True when the environment variable is set to a non-empty value that does not start with '0'.
// Diagnostic switches read once at startup; _dupenv_s on Windows keeps MSVC's /WX builds quiet.
inline bool EnvironmentFlagEnabled(const char* name) noexcept {
#ifdef _WIN32
	char* value = nullptr;
	size_t length = 0;
	const bool enabled = _dupenv_s(&value, &length, name) == 0 && value && value[0] != '\0' && value[0] != '0';
	std::free(value);
	return enabled;
#else
	const char* value = std::getenv(name);
	return value && value[0] != '\0' && value[0] != '0';
#endif
}

}
