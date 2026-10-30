#pragma once

// OpenRenderGraph's test assertions, one report format for every test: "FAIL <file>:<line>: <what failed>".

#include <cstdio>
#include <cstdlib>

#define ORG_TEST_REPORT(what) std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, what)

// In a function returning int (0: passed): returns the failing line.
#define CHECK(...) do { if (!(__VA_ARGS__)) { ORG_TEST_REPORT(#__VA_ARGS__); return __LINE__; } } while (false)

// Anywhere (lambdas, void functions): ends the test.
#define ENSURE(...) do { if (!(__VA_ARGS__)) { ORG_TEST_REPORT(#__VA_ARGS__); std::abort(); } } while (false)

// In a function returning int: returns 1, with a description of what failed.
#define REQUIRE(condition, message) do { if (!(condition)) { ORG_TEST_REPORT(message); return 1; } } while (false)
