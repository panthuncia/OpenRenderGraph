#include "OpenRenderGraph/Trace.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>

namespace org::trace {

namespace {
struct Global {
	std::mutex mutex;
	std::shared_ptr<Sink> sink;
	std::string resourceFilter;
	uint32_t categories = 0;
};
Global& State() {
	static Global global;
	return global;
}
// The categories take effect only with a sink.
void Publish(Global& global) {
	detail::g_categories.store(global.sink ? global.categories : 0u, std::memory_order_relaxed);
}

uint32_t ThreadNumber() {
	static std::atomic<uint32_t> next{1};
	thread_local const uint32_t number = next.fetch_add(1, std::memory_order_relaxed);
	return number;
}

void AppendJsonString(std::string& out, std::string_view text) {
	out += '"';
	for (const char c : text) {
		switch (c) {
		case '"': out += "\\\""; break;
		case '\\': out += "\\\\"; break;
		case '\n': out += "\\n"; break;
		case '\r': out += "\\r"; break;
		case '\t': out += "\\t"; break;
		default:
			if (static_cast<unsigned char>(c) < 0x20) {
				char escaped[8];
				std::snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned>(c));
				out += escaped;
			} else {
				out += c;
			}
		}
	}
	out += '"';
}

std::string ValueText(const Value& value) {
	return std::visit([](const auto& v) -> std::string {
		using T = std::decay_t<decltype(v)>;
		if constexpr (std::is_same_v<T, std::string>) return v;
		else if constexpr (std::is_same_v<T, bool>) return v ? "true" : "false";
		else return std::to_string(v);
	}, value);
}

class JsonLinesSink final : public Sink {
public:
	explicit JsonLinesSink(const std::string& path) : m_file(std::fopen(path.c_str(), "w")) {}
	~JsonLinesSink() override {
		if (m_file) std::fclose(m_file);
	}
	void Write(const Event& event) override {
		if (!m_file) return;
		std::string line;
		line.reserve(256);
		line += "{\"t\":";
		line += std::to_string(event.timestampNs);
		line += ",\"thread\":";
		line += std::to_string(event.thread);
		line += ",\"category\":";
		AppendJsonString(line, CategoryName(event.category));
		line += ",\"decision\":";
		AppendJsonString(line, event.decision);
		for (const auto& field : event.fields) {
			line += ',';
			AppendJsonString(line, field.key);
			line += ':';
			std::visit([&](const auto& v) {
				using T = std::decay_t<decltype(v)>;
				if constexpr (std::is_same_v<T, std::string>) AppendJsonString(line, v);
				else if constexpr (std::is_same_v<T, bool>) line += v ? "true" : "false";
				else line += std::to_string(v);
			}, field.value);
		}
		line += "}\n";
		std::lock_guard lock(m_mutex);
		std::fwrite(line.data(), 1, line.size(), m_file);
	}
	void Flush() override {
		std::lock_guard lock(m_mutex);
		if (m_file) std::fflush(m_file);
	}

private:
	std::mutex m_mutex;
	FILE* m_file = nullptr;
};
}

std::string_view CategoryName(Category category) {
	switch (category) {
	case Category::Schedule: return "schedule";
	case Category::Barriers: return "barriers";
	case Category::Handoff: return "handoff";
	case Category::Admission: return "admission";
	case Category::Aliasing: return "aliasing";
	case Category::Submission: return "submission";
	case Category::Retirement: return "retirement";
	default: return "unknown";
	}
}

const Value* Event::Find(std::string_view key) const {
	for (const auto& field : fields)
		if (field.key == key) return &field.value;
	return nullptr;
}

std::string Event::Text(std::string_view key) const {
	const auto* value = Find(key);
	return value ? ValueText(*value) : std::string{};
}

std::shared_ptr<Sink> MakeJsonLinesSink(const std::string& path) {
	return std::make_shared<JsonLinesSink>(path);
}

struct MemorySink::State {
	mutable std::mutex mutex;
	std::vector<Event> events;
};
std::shared_ptr<MemorySink::State> MemorySink::MakeState() { return std::make_shared<State>(); }

void MemorySink::Write(const Event& event) {
	std::lock_guard lock(m_state->mutex);
	m_state->events.push_back(event);
}

std::vector<Event> MemorySink::Take() {
	std::lock_guard lock(m_state->mutex);
	return std::exchange(m_state->events, {});
}

std::vector<Event> MemorySink::Of(std::string_view decision) const {
	std::lock_guard lock(m_state->mutex);
	std::vector<Event> result;
	for (const auto& event : m_state->events)
		if (event.decision == decision) result.push_back(event);
	return result;
}

void SetSink(std::shared_ptr<Sink> sink) {
	auto& global = State();
	std::lock_guard lock(global.mutex);
	if (global.sink) global.sink->Flush();
	global.sink = std::move(sink);
	Publish(global);
}

void SetCategories(Category categories) {
	auto& global = State();
	std::lock_guard lock(global.mutex);
	global.categories = uint32_t(categories);
	Publish(global);
}

void SetResourceFilter(std::string substring) {
	auto& global = State();
	std::lock_guard lock(global.mutex);
	global.resourceFilter = std::move(substring);
}

void Flush() {
	auto& global = State();
	std::lock_guard lock(global.mutex);
	if (global.sink) global.sink->Flush();
}

void ConfigureFromEnvironment() {
	const char* categories = std::getenv("ORG_TRACE");
	if (!categories || !*categories) return;
	uint32_t mask = 0;
	std::string_view rest(categories);
	while (!rest.empty()) {
		const auto comma = rest.find(',');
		const auto name = rest.substr(0, comma);
		rest = comma == std::string_view::npos ? std::string_view{} : rest.substr(comma + 1);
		if (name == "all") mask = uint32_t(Category::All);
		for (const auto category : {Category::Schedule, Category::Barriers, Category::Handoff, Category::Admission,
				 Category::Aliasing, Category::Submission, Category::Retirement})
			if (name == CategoryName(category)) mask |= uint32_t(category);
	}
	const char* file = std::getenv("ORG_TRACE_FILE");
	if (const char* resource = std::getenv("ORG_TRACE_RESOURCE")) SetResourceFilter(resource);
	SetSink(MakeJsonLinesSink(file && *file ? file : "org_trace.jsonl"));
	SetCategories(Category(mask));
}

Builder::Builder(Category category, std::string_view decision) {
	m_event.category = category;
	m_event.decision = decision;
	m_event.timestampNs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		std::chrono::steady_clock::now().time_since_epoch()).count());
	m_event.thread = ThreadNumber();
}

Builder& Builder::Add(std::string_view key, Value value) {
	m_event.fields.push_back({std::string(key), std::move(value)});
	return *this;
}

Builder& Builder::Resource(std::string_view name, std::string_view key) {
	m_event.resources.emplace_back(name);
	return Add(key, name);
}

Builder::~Builder() {
	if (m_cancelled) return;
	auto& global = State();
	std::shared_ptr<Sink> sink;
	{
		std::lock_guard lock(global.mutex);
		if (!global.resourceFilter.empty() && !m_event.resources.empty()) {
			bool matched = false;
			for (const auto& name : m_event.resources)
				matched = matched || name.find(global.resourceFilter) != std::string::npos;
			if (!matched) return;
		}
		sink = global.sink;
	}
	if (sink) sink->Write(m_event);
}

} // namespace org::trace
