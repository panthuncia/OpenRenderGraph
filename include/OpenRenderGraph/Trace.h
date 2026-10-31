#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

// ORG's decision trace. It is a stream of the decisions the graph makes while it compiles, admits, plans barriers for,
// submits and retires executions. Each event carries the inputs that decided it. Use it to diagnose scheduling and
// synchronization problems, for example:
// - which queue a pass went to, and why,
// - which barrier a resource got, and from which path: in-batch, cross-queue, the ledger's handoff, or an invalidated
//   or seeded backing,
// - why a handoff was or was not planned,
// - what each execution waited on and committed.
//
// The trace is off by default. When it is off, a trace point costs one relaxed atomic load. To use it:
// - enable categories with SetCategories, or with the ORG_TRACE environment variable (read by ConfigureFromEnvironment),
// - give it a sink: a JSON-lines file (MakeJsonLinesSink, ORG_TRACE_FILE) or a MemorySink for tests.
// ORG_TRACE_RESOURCE (SetResourceFilter) keeps only the events that name no resource, or that name a resource whose
// name contains the filter.
//
// Names: an event names passes and resources by the labels the graph captured while tracing was on. So enable tracing
// before the graphs of interest build. The environment is read when the runtime is created. Events are emitted from
// whichever thread made the decision. A sink must be thread-safe. Both provided sinks are.
namespace org::trace {

enum class Category : uint32_t {
	None = 0,
	Schedule = 1u << 0,    // queue assignment, batching, intra-execution waits (at compile)
	Barriers = 1u << 1,    // each state step's barrier plan
	Handoff = 1u << 2,     // cross-queue and cross-execution handoffs: releases, acquires, why one was not planned
	Admission = 1u << 3,   // admission, readmission, incoming waits, ledger commits
	Aliasing = 1u << 4,    // transient alias placement
	Submission = 1u << 5,  // what each batch submitted: leading and release lists, waits, signals
	Retirement = 1u << 6,  // completion and retirement of executions
	All = 0xFFFFFFFFu,
};
constexpr Category operator|(Category a, Category b) { return Category(uint32_t(a) | uint32_t(b)); }
std::string_view CategoryName(Category category);

using Value = std::variant<int64_t, uint64_t, double, bool, std::string>;
struct Field {
	std::string key;
	Value value;
};

struct Event {
	Category category = Category::None;
	std::string decision;     // what was decided, e.g. "state-step", "handoff-skipped"
	uint64_t timestampNs = 0; // steady clock
	uint32_t thread = 0;
	std::vector<Field> fields;
	// The resources the event concerns (for the resource filter), by name.
	std::vector<std::string> resources;

	const Value* Find(std::string_view key) const;
	// The field's value as text ("" if absent).
	std::string Text(std::string_view key) const;
};

class Sink {
public:
	virtual ~Sink() = default;
	virtual void Write(const Event& event) = 0;
	virtual void Flush() {}
};

// One JSON object per line. Thread-safe.
std::shared_ptr<Sink> MakeJsonLinesSink(const std::string& path);

// Keeps events in memory. Thread-safe.
class MemorySink final : public Sink {
public:
	void Write(const Event& event) override;
	std::vector<Event> Take();
	// Events of one decision, in emission order.
	std::vector<Event> Of(std::string_view decision) const;

private:
	struct State;
	std::shared_ptr<State> m_state = MakeState();
	static std::shared_ptr<State> MakeState();
};

namespace detail {
inline std::atomic<uint32_t> g_categories{0};
}
inline bool Enabled(Category category) noexcept {
	return (detail::g_categories.load(std::memory_order_relaxed) & uint32_t(category)) != 0;
}
// Whether any category is on. While one is, the graph captures names for events.
inline bool Any() noexcept { return detail::g_categories.load(std::memory_order_relaxed) != 0; }

// A null sink turns the trace off, whatever the categories.
void SetSink(std::shared_ptr<Sink> sink);
void SetCategories(Category categories);
void SetResourceFilter(std::string substring);
void Flush();
// Reads ORG_TRACE (comma-separated category names, or "all"), ORG_TRACE_FILE (default "org_trace.jsonl") and
// ORG_TRACE_RESOURCE. Does nothing when ORG_TRACE is unset or empty.
void ConfigureFromEnvironment();

// Builds one event and emits it when it goes out of scope. Only construct one where Enabled(category) is true.
// The ORG_TRACE macro checks this first.
class Builder {
public:
	Builder(Category category, std::string_view decision);
	Builder(const Builder&) = delete;
	Builder& operator=(const Builder&) = delete;
	~Builder();

	Builder& Add(std::string_view key, Value value);
	Builder& Add(std::string_view key, std::string_view value) { return Add(key, Value{std::string(value)}); }
	Builder& Add(std::string_view key, const char* value) { return Add(key, Value{std::string(value ? value : "")}); }
	Builder& Add(std::string_view key, const std::string& value) { return Add(key, Value{value}); }
	Builder& Add(std::string_view key, uint32_t value) { return Add(key, Value{uint64_t{value}}); }
	Builder& Add(std::string_view key, int32_t value) { return Add(key, Value{int64_t{value}}); }
	Builder& Add(std::string_view key, uint64_t value) { return Add(key, Value{value}); }
	Builder& Add(std::string_view key, int64_t value) { return Add(key, Value{value}); }
	Builder& Add(std::string_view key, double value) { return Add(key, Value{value}); }
	Builder& Add(std::string_view key, bool value) { return Add(key, Value{value}); }
	// Marks a resource the event concerns. It adds a field named `key` ("resource" by default) that holds the resource's
	// name. The resource filter matches against that name.
	Builder& Resource(std::string_view name, std::string_view key = "resource");
	// Drops the event instead of emitting it.
	void Cancel() { m_cancelled = true; }

private:
	Event m_event;
	bool m_cancelled = false;
};

} // namespace org::trace

// Usage: ORG_TRACE(Barriers, "state-step").Add("pass", name)...
// The event is emitted at the end of the full expression. It is built only when the category is enabled.
// Otherwise the arguments are not evaluated.
#define ORG_TRACE(category, decision) \
	if (!::org::trace::Enabled(::org::trace::Category::category)) {} else ::org::trace::Builder(::org::trace::Category::category, decision)
