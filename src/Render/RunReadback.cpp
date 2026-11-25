#include "Render/RunReadback.h"

#include "Render/LatchBlock.h"
#include "Render/PassBuilders.h"
#include "RenderPasses/Base/TypedRenderGraphPass.h"
#include "Resources/Buffers/Buffer.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <format>
#include <mutex>
#include <utility>

#include <spdlog/spdlog.h>

namespace org {

namespace {
// Enough lease indices for every run in flight.
constexpr uint32_t kLeases = 256;
// A texture copy's offset must be 512-aligned on D3D12. Each part has its stamp, then its data, both at that alignment.
constexpr uint32_t kAlign = 512;

constexpr uint32_t AlignPart(uint32_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

// Hands an error to the host. Without a handler it is only logged.
void Report(const RunReadback::ErrorHandler& onError, const std::string& why) {
	if (onError) onError("run readback: " + why);
	else spdlog::critical("[ORG] run readback: {}", why);
}

enum class Scheme : uint8_t { Parts, List, Regions };
} // namespace

static_assert(sizeof(RunReadback::Stamp) == 16);

struct RunReadback::ListSet {
	uint32_t items = 0;
	uint64_t bytes = 0;              // one region: the header and every item
	std::shared_ptr<Buffer> buffer;  // Regions() regions
	ListSet* next = nullptr;
};

struct RunReadback::State {
	struct Run {
		RunRef ref;
		uint32_t stamp = 0;
		bool committed = false, submitted = false;
	};

	std::string name;
	ErrorHandler onError;
	Scheme scheme = Scheme::Parts;
	std::shared_ptr<Buffer> buffer;  // parts and regions
	std::vector<uint32_t> sizes;     // by part
	std::vector<uint32_t> bases;     // by part, the stamp's offset in a region. The data follows at kAlign.
	uint32_t stride = 0;             // one lease's region, or one stable run's

	// The current build's latch and stamp offset. Render thread.
	std::shared_ptr<const LatchBlock> latch;
	uint32_t stampOffset = 0;

	mutable std::mutex mutex;
	mutable std::vector<Run> runs;  // by lease index, or by region for a list
	mutable uint32_t lastCommitted = ~0u;
	mutable std::vector<std::vector<std::byte>> latest;  // by part, not taken yet
	mutable std::vector<uint32_t> latestStamps;          // by part
	uint32_t sequence = 0;

	// The stable runs' regions: a list's and a region readback's.
	uint32_t regions = 0;
	mutable std::vector<uint32_t> regionOf;  // by lease index, under the mutex
	mutable uint32_t regionsGiven = 0;

	// A list's part and its sets.
	List list{};
	std::atomic<ListSet*> sets{nullptr};
	std::atomic<uint32_t> wanted{0};
	std::atomic<uint64_t> overflowed{0}, missed{0}, setsAdded{0};

	State() = default;
	~State() {
		Unregister();
		for (ListSet* set = sets.exchange(nullptr); set;) {
			ListSet* following = set->next;
			delete set;
			set = following;
		}
	}
	State(const State&) = delete;
	State& operator=(const State&) = delete;

	// Part readbacks move their staging runs' commits (MoveStagingRun).
	static std::mutex& RegistryMutex() {
		static auto* mutex = new std::mutex;
		return *mutex;
	}
	static std::vector<State*>& Registry() {
		// Never destroyed, since readbacks in static objects leave it during static destruction.
		static auto* registry = new std::vector<State*>;
		return *registry;
	}
	void Register() {
		std::scoped_lock lock(RegistryMutex());
		Registry().push_back(this);
	}
	void Unregister() {
		if (scheme != Scheme::Parts) return;
		std::scoped_lock lock(RegistryMutex());
		std::erase(Registry(), this);
		std::scoped_lock committed(CommittedMutex());
		for (auto& list : CommittedAt()) std::erase(list, this);
	}
	// By lease index: the part readbacks with a commit there that no move took yet, so a stage's move asks only those. A leaf lock,
	// taken inside a state's mutex (Commit) and inside RegistryMutex.
	static std::mutex& CommittedMutex() {
		static auto* mutex = new std::mutex;
		return *mutex;
	}
	static std::vector<std::vector<State*>>& CommittedAt() {
		static auto* lists = new std::vector<std::vector<State*>>;
		return *lists;
	}
	void NoteCommitted(uint32_t lease) {
		if (scheme != Scheme::Parts) return;
		std::scoped_lock lock(CommittedMutex());
		auto& lists = CommittedAt();
		if (lists.size() <= lease) lists.resize(size_t{lease} + 1);
		if (std::ranges::find(lists[lease], this) == lists[lease].end()) lists[lease].push_back(this);
	}

	void Fail(const std::string& why) const { Report(onError, why); }

	// False, after the error, when the lease is past the regions.
	bool CheckLease(uint32_t lease) const {
		if (lease < kLeases) return true;
		Fail(std::format("{} got run lease {}, past its {} regions", name, lease, kLeases));
		return false;
	}

	// Under the mutex. The lease's last run is done, so its region holds what that run copied.
	void Harvest(uint32_t lease) const {
		if (lease >= runs.size()) return;
		const auto run = std::exchange(runs[lease], Run{});
		// A run that was committed but not submitted copied nothing of its own.
		if (!run.committed || !run.submitted) return;
		void* mapped = nullptr;
		buffer->GetAPIResource().Map(&mapped);
		if (!mapped) {
			Fail(std::format("{} could not be mapped", name));
			return;
		}
		const auto* region = static_cast<const std::byte*>(mapped) + uint64_t(lease) * stride;
		for (size_t part = 0; part < sizes.size(); ++part) {
			uint32_t written = 0;
			std::memcpy(&written, region + bases[part], sizeof(written));
			// Another stamp means this part's pass was not in the run, or a later uncommitted run wrote over it.
			if (written != run.stamp) continue;
			const auto* data = region + bases[part] + kAlign;
			latest[part].assign(data, data + sizes[part]);
			latestStamps[part] = run.stamp;
		}
		buffer->GetAPIResource().Unmap(0, 0);
	}

	// Under the mutex. A stable run's region, given on first use.
	uint32_t RegionLocked(const RunRef& run) const {
		if (!run.Stable()) return kNoRegion;
		if (regionOf.size() <= run.lease) regionOf.resize(size_t{run.lease} + 1, kNoRegion);
		auto& region = regionOf[run.lease];
		if (region == kNoRegion && regionsGiven < regions) region = regionsGiven++;
		return region;
	}

	// Under the mutex. Reads a list region's run of stamp from whichever set's copy holds it.
	void HarvestList(uint32_t region, uint32_t stamp, const std::function<void(const ListRead&)>& read) {
		for (const ListSet* set = sets.load(std::memory_order_acquire); set; set = set->next) {
			auto resource = set->buffer->GetAPIResource();
			void* mapped = nullptr;
			resource.Map(&mapped);
			if (!mapped) continue;
			const auto* words = reinterpret_cast<const uint32_t*>(static_cast<const std::byte*>(mapped) + uint64_t(region) * set->bytes);
			if (words[list.stampOffset / sizeof(uint32_t)] != stamp) {
				resource.Unmap(0, 0);
				continue;
			}
			const uint32_t count = words[0];
			if (count > set->items) {
				uint32_t asked = wanted.load(std::memory_order_relaxed);
				while (asked < count && !wanted.compare_exchange_weak(asked, count, std::memory_order_release, std::memory_order_relaxed)) {}
				overflowed.fetch_add(1, std::memory_order_relaxed);
			}
			read({.region = region, .stamp = stamp, .words = words, .count = count, .capacity = set->items});
			resource.Unmap(0, 0);
			return;
		}
		missed.fetch_add(1, std::memory_order_relaxed);
		read({.region = region, .stamp = stamp});
	}

	// A stage's commit wrote its staging run. The stage now submits it as to, whose last run is done.
	void MoveRun(const RunRef& from, const RunRef& to) {
		if (from.lease == to.lease) return;
		const std::scoped_lock lock(mutex);
		if (from.lease >= runs.size() || !runs[from.lease].committed || !(runs[from.lease].ref == from)) return;
		if (!CheckLease(to.lease)) return;
		if (runs.size() <= to.lease) runs.resize(size_t{to.lease} + 1);
		Harvest(to.lease);
		auto moved = std::exchange(runs[from.lease], Run{});
		moved.ref = to;
		runs[to.lease] = moved;
		if (lastCommitted == from.lease) lastCommitted = to.lease;
	}

	uint32_t NextStamp() {
		uint32_t stamp = ++sequence;
		if (stamp == 0) stamp = ++sequence;
		return stamp;
	}
};

namespace {

// A texture copy's row pitch: 16 bytes a texel at most, aligned to 256.
constexpr uint32_t TexturePitch(uint32_t width) { return (width * 16u + 255u) & ~255u; }

struct PartSpec {
	std::shared_ptr<const RunReadback::State> state;
	std::shared_ptr<const LatchBlock> latch;
	uint32_t stamp = 0;  // the stamp's offset in the latch
	uint32_t part = 0;
	std::shared_ptr<Resource> source;
	bool texture = false;
	uint32_t width = 0, height = 0;
};

struct PartBindings {
	ResourceBindingToken source;
};

struct PartFrame {
	std::shared_ptr<const PartSpec> spec;
};

// Copies the run's stamp and one part into the run's region. Its commands depend only on the run, so it is replayable.
class PartPass final : public TypedRenderGraphPass<PartPass, PartFrame, PartBindings> {
public:
	explicit PartPass(std::shared_ptr<const PartSpec> spec) : m_spec(std::move(spec)) {}

	PartBindings Declare(PassBuilder& builder) {
		builder.PreferQueue(QueueKind::Graphics);
		PartBindings bindings{};
		bindings.source = builder.CopySource(m_spec->source);
		builder.CopyDestination(std::static_pointer_cast<Resource>(m_spec->state->buffer));
		return bindings;
	}

	void Revision(std::vector<uint64_t>& out) const { out.push_back(reinterpret_cast<std::uintptr_t>(m_spec.get())); }

	PartFrame Prepare(const PartBindings&, const PassPrepareContext&) const { return {m_spec}; }

	static void Record(const PartBindings& bindings, const PartFrame& frame, PassRecordContext& recording) {
		const auto& spec = *frame.spec;
		const auto& state = *spec.state;
		const auto& run = recording.Run();
		if (!state.CheckLease(run.lease)) return;
		auto& commands = recording.Commands();
		const auto destination = state.buffer->GetAPIResource().GetHandle();
		const uint64_t base = uint64_t(run.lease) * state.stride + state.bases[spec.part];
		const auto latch = spec.latch->At(run);
		commands.CopyBufferRegion(destination, base, latch.buffer->GetAPIResource().GetHandle(), latch.offset + spec.stamp, sizeof(uint32_t));
		const auto source = recording.Resolve(bindings.source).GetHandle();
		if (!spec.texture) {
			commands.CopyBufferRegion(destination, base + kAlign, source, 0, state.sizes[spec.part]);
			return;
		}
		rhi::BufferTextureCopyFootprint copy{};
		copy.texture = source;
		copy.buffer = destination;
		copy.mip = 0;
		copy.arraySlice = 0;
		copy.footprint = {base + kAlign, TexturePitch(spec.width), spec.width, spec.height, 1};
		commands.CopyTextureToBuffer(copy);
	}

private:
	std::shared_ptr<const PartSpec> m_spec;
};

// Null, after the error, when it cannot be made.
std::shared_ptr<Buffer> MakeReadbackBuffer(const RunReadback::State& state, uint64_t bytes) {
	auto buffer = Buffer::CreateShared(rhi::HeapType::Readback, bytes);
	if (!buffer) {
		state.Fail(std::format("the buffer {} ({} bytes) could not be made", state.name, bytes));
		return nullptr;
	}
	buffer->SetName(state.name);
	return buffer;
}

} // namespace

std::shared_ptr<RunReadback> RunReadback::Create(std::string_view name, std::vector<uint32_t> parts, ErrorHandler onError) {
	auto made = std::make_shared<State>();
	made->name = std::string(name);
	made->onError = std::move(onError);
	for (const auto bytes : parts) {
		made->bases.push_back(made->stride);
		made->sizes.push_back(bytes);
		made->stride += kAlign + AlignPart(bytes);
	}
	made->latest.resize(parts.size());
	made->latestStamps.resize(parts.size());
	made->buffer = MakeReadbackBuffer(*made, uint64_t(made->stride) * kLeases);
	if (!made->buffer) return nullptr;
	made->Register();
	return std::make_shared<RunReadback>(std::move(made));
}

void RunReadback::Bind(std::shared_ptr<const LatchBlock> latch, uint32_t stampOffset) {
	m_state->latch = std::move(latch);
	m_state->stampOffset = stampOffset;
}

std::function<std::shared_ptr<RenderPass>()> RunReadback::BufferPass(uint32_t part, std::shared_ptr<Resource> source, std::string& why) const {
	if (part >= m_state->sizes.size() || !source || !m_state->latch) {
		why = std::format("the readback {}: the part, its source or its Bind is missing", m_state->name);
		return {};
	}
	auto spec = std::make_shared<const PartSpec>(PartSpec{
		.state = m_state, .latch = m_state->latch, .stamp = m_state->stampOffset, .part = part, .source = std::move(source)});
	return [spec]() -> std::shared_ptr<RenderPass> { return std::make_shared<PartPass>(spec); };
}

std::function<std::shared_ptr<RenderPass>()> RunReadback::TexturePass(uint32_t part, std::shared_ptr<Resource> source, uint32_t width,
	uint32_t height, std::string& why) const {
	if (part >= m_state->sizes.size() || !source || !m_state->latch) {
		why = std::format("the readback {}: the part, its source or its Bind is missing", m_state->name);
		return {};
	}
	// Rows are 256-byte aligned, with room for 16 bytes a texel.
	if (!width || !height || uint64_t(height) * TexturePitch(width) > m_state->sizes[part]) {
		why = std::format("the readback {}: a {}x{} texture does not fit its part", m_state->name, width, height);
		return {};
	}
	auto spec = std::make_shared<const PartSpec>(PartSpec{.state = m_state, .latch = m_state->latch, .stamp = m_state->stampOffset,
		.part = part, .source = std::move(source), .texture = true, .width = width, .height = height});
	return [spec]() -> std::shared_ptr<RenderPass> { return std::make_shared<PartPass>(spec); };
}

uint32_t RunReadback::Commit(const RunRef& run) {
	auto& state = *m_state;
	const std::scoped_lock lock(state.mutex);
	if (!state.CheckLease(run.lease)) return 0;
	if (state.runs.size() <= run.lease) state.runs.resize(size_t{run.lease} + 1);
	// A leased run's last run on this index retired before the lease reopened. A stage's staging run holds nothing.
	state.Harvest(run.lease);
	const uint32_t stamp = state.NextStamp();
	state.runs[run.lease] = {.ref = run, .stamp = stamp, .committed = true};
	state.lastCommitted = run.lease;
	state.NoteCommitted(run.lease);
	return stamp;
}

void RunReadback::Executed() {
	const std::scoped_lock lock(m_state->mutex);
	if (m_state->lastCommitted < m_state->runs.size()) m_state->runs[m_state->lastCommitted].submitted = true;
}

std::vector<std::byte> RunReadback::Take(uint32_t part, uint32_t* stamp) {
	const std::scoped_lock lock(m_state->mutex);
	if (part >= m_state->latest.size()) return {};
	if (stamp) *stamp = m_state->latestStamps[part];
	return std::exchange(m_state->latest[part], {});
}

void RunReadback::MoveStagingRun(const RunRef& from, const RunRef& to) {
	std::scoped_lock lock(State::RegistryMutex());
	std::vector<State*> committed;
	{
		std::scoped_lock lists(State::CommittedMutex());
		auto& at = State::CommittedAt();
		if (from.lease < at.size()) committed.swap(at[from.lease]);
	}
	for (auto* state : committed) state->MoveRun(from, to);
}

std::shared_ptr<RunReadback> RunReadback::CreateList(std::string_view name, const List& list, uint32_t items, ErrorHandler onError) {
	if (!list.itemBytes || list.headerBytes < 2 * sizeof(uint32_t) || list.headerBytes % sizeof(uint32_t) || list.stampOffset % sizeof(uint32_t)
		|| list.stampOffset < sizeof(uint32_t) || list.stampOffset >= list.headerBytes || list.itemBytes % sizeof(uint32_t) || !list.regions) {
		Report(onError, std::format("the list readback {} has a malformed part", name));
		return nullptr;
	}
	auto made = std::make_shared<State>();
	made->name = std::string(name);
	made->onError = std::move(onError);
	made->scheme = Scheme::List;
	made->list = list;
	made->regions = list.regions;
	made->runs.resize(list.regions);
	auto readback = std::make_shared<RunReadback>(std::move(made));
	readback->Grow(items);
	if (!readback->SetFor(items)) return nullptr;
	return readback;
}

void RunReadback::Grow(uint32_t items) {
	auto& state = *m_state;
	if (state.scheme != Scheme::List || SetFor(items)) return;
	auto set = std::make_unique<ListSet>();
	set->items = items;
	set->bytes = state.list.headerBytes + uint64_t(items) * state.list.itemBytes;
	set->buffer = MakeReadbackBuffer(state, set->bytes * state.list.regions);
	if (!set->buffer) return;
	auto* added = set.release();
	added->next = state.sets.load(std::memory_order_relaxed);
	while (!state.sets.compare_exchange_weak(added->next, added, std::memory_order_release, std::memory_order_relaxed)) {}
	state.setsAdded.fetch_add(1, std::memory_order_relaxed);
}

const RunReadback::ListSet* RunReadback::SetFor(uint32_t items) const {
	for (const ListSet* set = m_state->sets.load(std::memory_order_acquire); set; set = set->next)
		if (set->items == items) return set;
	return nullptr;
}

void RunReadback::RecordList(rhi::CommandList& commands, const RunRef& run, rhi::ResourceHandle source, const ListSet& set) const {
	const uint32_t region = RegionOf(run);
	if (region == kNoRegion) return;
	commands.CopyBufferRegion(set.buffer->GetAPIResource().GetHandle(), uint64_t(region) * set.bytes, source, 0, set.bytes);
}

RunReadback::ListCommit RunReadback::CommitList(const RunRef& run, const std::function<void(const ListRead&)>& read) {
	auto& state = *m_state;
	const std::scoped_lock lock(state.mutex);
	ListCommit out;
	out.region = state.RegionLocked(run);
	if (out.region == kNoRegion) return out;
	auto& last = state.runs[out.region];
	if (last.committed) state.HarvestList(out.region, last.stamp, read);
	const uint32_t stamp = state.NextStamp();
	last = {.ref = run, .stamp = stamp, .committed = true};
	out.stamp = stamp;
	return out;
}

uint32_t RunReadback::Wanted() const {
	return m_state->wanted.load(std::memory_order_acquire);
}

uint32_t RunReadback::Regions() const {
	return m_state->regions;
}

RunReadback::ListStats RunReadback::Stats(bool take) {
	auto& state = *m_state;
	ListStats stats;
	const auto read = [&](std::atomic<uint64_t>& counter) {
		return take ? counter.exchange(0, std::memory_order_relaxed) : counter.load(std::memory_order_relaxed);
	};
	stats.overflowed = read(state.overflowed);
	stats.missed = read(state.missed);
	stats.setsAdded = read(state.setsAdded);
	for (const ListSet* set = state.sets.load(std::memory_order_acquire); set; set = set->next) {
		++stats.sets;
		stats.largest = (std::max)(stats.largest, set->items);
	}
	return stats;
}

std::shared_ptr<RunReadback> RunReadback::CreateRegions(std::string_view name, uint32_t bytes, uint32_t regions, ErrorHandler onError) {
	if (!bytes || !regions) {
		Report(onError, std::format("the region readback {} needs a size and a region count", name));
		return nullptr;
	}
	auto made = std::make_shared<State>();
	made->name = std::string(name);
	made->onError = std::move(onError);
	made->scheme = Scheme::Regions;
	made->regions = regions;
	// Buffer copies need 4-byte offsets.
	made->stride = (bytes + 3u) & ~3u;
	made->buffer = MakeReadbackBuffer(*made, uint64_t(made->stride) * regions);
	if (!made->buffer) return nullptr;
	return std::make_shared<RunReadback>(std::move(made));
}

uint32_t RunReadback::RegionOf(const RunRef& run) const {
	const std::scoped_lock lock(m_state->mutex);
	return m_state->RegionLocked(run);
}

void RunReadback::RecordCopy(rhi::CommandList& commands, uint32_t region, uint64_t offset, rhi::ResourceHandle source, uint64_t sourceOffset,
	uint64_t bytes) const {
	const auto& state = *m_state;
	if (state.scheme != Scheme::Regions || region >= state.regions || offset + bytes > state.stride) {
		state.Fail(std::format("{}: a copy of {} bytes at {} is outside region {}", state.name, bytes, offset, region));
		return;
	}
	commands.CopyBufferRegion(state.buffer->GetAPIResource().GetHandle(), uint64_t(region) * state.stride + offset, source, sourceOffset, bytes);
}

void RunReadback::ReadRegion(uint32_t region, const std::function<void(const std::byte*)>& read) const {
	const auto& state = *m_state;
	if (state.scheme != Scheme::Regions || region >= state.regions) {
		state.Fail(std::format("{}: no region {} to read", state.name, region));
		return;
	}
	auto resource = state.buffer->GetAPIResource();
	void* mapped = nullptr;
	resource.Map(&mapped);
	if (!mapped) {
		state.Fail(std::format("{} could not be mapped", state.name));
		return;
	}
	read(static_cast<const std::byte*>(mapped) + uint64_t(region) * state.stride);
	resource.Unmap(0, 0);
}

} // namespace org
