#include "Render/RunGate.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <span>
#include <unordered_set>
#include <vector>

#include "OpenRenderGraph/Runtime.h"

namespace org {

namespace {
// Every live gate, so a host's staging copy can move each one's open bits.
std::mutex& GatesMutex() {
	static auto* mutex = new std::mutex;
	return *mutex;
}
std::vector<const RunGate*>& Gates() {
	static auto* gates = new std::vector<const RunGate*>;
	return *gates;
}
// Per lease index, the gates opened since the last ClearOpenAll, so a copy moves only those. Under GatesMutex.
std::vector<std::unordered_set<const RunGate*>>& Touched() {
	static auto* touched = new std::vector<std::unordered_set<const RunGate*>>;
	return *touched;
}
void NoteTouched(const RunGate* gate, uint32_t lease) {
	const std::lock_guard lock(GatesMutex());
	auto& touched = Touched();
	if (touched.size() <= lease) touched.resize(static_cast<size_t>(lease) + 1);
	touched[lease].insert(gate);
}
}

std::shared_ptr<LatchBlock> RunGate::CreateLatch(std::string_view name, uint32_t entries, std::string* error) {
	const auto fail = [&](const char* reason) -> std::shared_ptr<LatchBlock> {
		if (error) *error = reason;
		return nullptr;
	};
	if (!entries) return fail("no entries");
	const auto* runtime = Runtime::Current();
	if (!runtime || !runtime->Arena()) return fail("no runtime");
	// A run's region is zeroed when allocated, so every run starts closed.
	return LatchBlock::Create(std::string(name), entries * kEntryBytes);
}

RunGate::RunGate(std::shared_ptr<LatchBlock> latch, uint32_t entries)
	: m_latch(std::move(latch)), m_entries(entries), m_words((entries + 63) / 64) {
	m_open = std::make_unique<std::atomic<uint64_t>[]>(static_cast<size_t>(Runtime::kMaxSlots) * m_words);
	const std::lock_guard lock(GatesMutex());
	Gates().push_back(this);
}

RunGate::~RunGate() {
	const std::lock_guard lock(GatesMutex());
	std::erase(Gates(), this);
	for (auto& touched : Touched()) touched.erase(this);
}

void RunGate::CopyOpenAll(const RunRef& from, const RunRef& to) {
	if (from.lease >= Runtime::kMaxSlots || to.lease >= Runtime::kMaxSlots || from.lease == to.lease) return;
	const std::lock_guard lock(GatesMutex());
	auto& touched = Touched();
	const auto size = static_cast<size_t>((std::max)(from.lease, to.lease)) + 1;
	if (touched.size() < size) touched.resize(size);
	// A gate open in the target's last run but not in the source is copied closed.
	for (const auto* gate : touched[to.lease])
		if (!touched[from.lease].contains(gate)) gate->CopyOpen(from, to);
	for (const auto* gate : touched[from.lease]) gate->CopyOpen(from, to);
	touched[to.lease] = touched[from.lease];
}

void RunGate::ClearOpenAll(const RunRef& run) {
	if (run.lease >= Runtime::kMaxSlots) return;
	const std::lock_guard lock(GatesMutex());
	auto& touched = Touched();
	if (run.lease >= touched.size()) return;
	for (const auto* gate : touched[run.lease])
		for (uint32_t i = 0; i < gate->m_words; ++i)
			gate->m_open[static_cast<size_t>(run.lease) * gate->m_words + i].store(0, std::memory_order_release);
	touched[run.lease].clear();
}

void RunGate::Write(const RunRef& run, uint32_t entry, const std::array<uint32_t, 4>& words, bool work) const {
	if (entry >= m_entries) return;
	m_latch->Write(run, Offset(entry), std::as_bytes(std::span(words)));
	if (run.lease >= Runtime::kMaxSlots) return;
	auto& word = m_open[static_cast<size_t>(run.lease) * m_words + entry / 64];
	const uint64_t bit = uint64_t{1} << (entry % 64);
	if (work) {
		if ((word.fetch_or(bit, std::memory_order_release) & bit) == 0) NoteTouched(this, run.lease);
	} else {
		word.fetch_and(~bit, std::memory_order_release);
	}
}

void RunGate::Close(const RunRef& run) const {
	const auto bytes = m_latch->Bytes(run);
	std::memset(bytes.data(), 0, static_cast<size_t>(m_entries) * kEntryBytes);
	if (run.lease >= Runtime::kMaxSlots) return;
	for (uint32_t i = 0; i < m_words; ++i) m_open[static_cast<size_t>(run.lease) * m_words + i].store(0, std::memory_order_release);
}

void RunGate::CopyOpen(const RunRef& from, const RunRef& to) const {
	if (from.lease >= Runtime::kMaxSlots || to.lease >= Runtime::kMaxSlots || from.lease == to.lease) return;
	for (uint32_t i = 0; i < m_words; ++i)
		m_open[static_cast<size_t>(to.lease) * m_words + i].store(m_open[static_cast<size_t>(from.lease) * m_words + i].load(std::memory_order_acquire),
			std::memory_order_release);
}

bool RunGate::IsOpen(const RunRef& run, uint32_t entry) const {
	if (entry >= m_entries || run.lease >= Runtime::kMaxSlots) return true;
	const auto word = m_open[static_cast<size_t>(run.lease) * m_words + entry / 64].load(std::memory_order_acquire);
	return ((word >> (entry % 64)) & 1) != 0;
}

void RunGate::OpenDispatch(const RunRef& run, uint32_t entry, uint32_t x, uint32_t y, uint32_t z) const {
	Write(run, entry, {x, y, z, 0}, x && y && z);
}

void RunGate::OpenDraw(const RunRef& run, uint32_t entry, uint32_t vertices, uint32_t instances) const {
	Write(run, entry, {vertices, instances, 0, 0}, vertices && instances);
}

} // namespace org
