#include "FramePlacement.h"

#include <algorithm>
#include <array>
#include <map>
#include <string>

#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include "Render/RenderGraph/FrameAliasArena.h"
#include "Render/RenderGraph/PersistentAliasArena.h"

namespace org {

namespace {
double MB(uint64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

const char* MemoryName(PlacementMemory memory) {
	switch (memory) {
	case PlacementMemory::SharedArena: return "shared arena";
	case PlacementMemory::PrivatePools: return "private pools";
	case PlacementMemory::FrameArena: return "frame arena";
	}
	return "?";
}
}  // namespace

FramePlacement::FramePlacement(std::vector<uint32_t> epochOrder) : m_epochOrder(std::move(epochOrder)) {}

FramePlacement::~FramePlacement() {
	{
		std::lock_guard lock(m_mutex);
		m_stopping = true;
	}
	m_wake.notify_all();
	if (m_thread.joinable()) m_thread.join();
}

void FramePlacement::SetSettings(Settings settings) {
	std::sort(settings.repeatedEpochs.begin(), settings.repeatedEpochs.end());
	{
		std::lock_guard lock(m_mutex);
		m_settings = std::move(settings);
		m_memory->SetSizing(m_settings.sizing);
		m_enabled.store(m_settings.enabled, std::memory_order_release);
		// Disabled: graphs place on their own from their next plan. Enabled: the next computation publishes a version.
		if (!m_settings.enabled) m_current.reset();
		++m_changes;
		m_lastChange = std::chrono::steady_clock::now();
		if (!m_summaries.empty() && !m_thread.joinable() && !m_stopping) m_thread = std::thread([this] { Run(); });
	}
	m_wake.notify_all();
}

FramePlacement::Settings FramePlacement::GetSettings() const {
	std::lock_guard lock(m_mutex);
	return m_settings;
}

void FramePlacement::SetFrameArena(std::weak_ptr<FrameAliasArena> arena) {
	std::lock_guard lock(m_mutex);
	m_frameArena = std::move(arena);
}

void FramePlacement::SetFrameEpochs(std::vector<uint32_t> epochs) {
	{
		std::lock_guard lock(m_mutex);
		m_frameEpochs = std::move(epochs);
		++m_changes;
		m_lastChange = std::chrono::steady_clock::now();
	}
	m_wake.notify_all();
}

bool FramePlacement::SummariesCompleteLocked() const {
	const auto& epochs = m_frameEpochs.empty() ? m_epochOrder : m_frameEpochs;
	return std::ranges::all_of(epochs, [&](uint32_t epoch) { return m_summaries.contains(epoch); });
}

FramePlacementInputs FramePlacement::InputsLocked(std::vector<std::shared_ptr<const PlacementSummary>>& summaries) const {
	FramePlacementInputs inputs;
	inputs.epochOrder = m_epochOrder;
	inputs.repeatedEpochs = m_settings.repeatedEpochs;
	inputs.beamSearch = m_settings.beamSearch;
	summaries.clear();
	for (const auto& [epoch, summary] : m_summaries) summaries.push_back(summary);
	inputs.summaries = summaries;
	return inputs;
}

void FramePlacement::Publish(PlacementSummary summary) {
	auto published = std::make_shared<const PlacementSummary>(std::move(summary));
	std::shared_ptr<const FramePlacementVersion> first;
	bool firstEver = false;
	std::vector<std::shared_ptr<const PlacementSummary>> summaries;
	bool detailed = false;
	double computeMs = 0.0;
	{
		std::unique_lock lock(m_mutex);
		auto& kept = m_summaries[published->epoch];
		const bool changed = !kept || kept->hash != published->hash;
		// The same uses give the same placement. The new summary is kept for its pooled bytes alone.
		kept = std::move(published);
		if (!changed) return;
		++m_changes;
		m_lastChange = std::chrono::steady_clock::now();
		if (!m_thread.joinable() && !m_stopping) m_thread = std::thread([this] { Run(); });
		// The first summary with candidates makes the first version now, from the summaries so far, so the plan that published
		// it places from it instead of on its own path. With several frame epochs the later epochs' first plans then fit their
		// candidates into the version's free bytes, and report what it leaves uncovered (a resource the earlier epoch shares
		// with them: its span in the version ends in the earlier epoch). One with candidates the current version left uncovered
		// (PlacementSummary::uncovered) makes the next version now, and its plan places from it at once. (Waiting for every
		// epoch's summary instead put the first epochs' transients in pools beside the version's heaps for 120 frames.)
		const bool uncovered = m_current && kept->uncovered != 0;
		const bool firstVersion = !m_current && !kept->uses.empty();
		if (m_settings.enabled && (firstVersion || uncovered)) {
			const auto replacing = m_current;
			const auto inputs = InputsLocked(summaries);
			const auto changes = m_changes;
			const auto id = m_nextId++;
			detailed = m_settings.logLifetimes;
			// Every other epoch's graph keeps running: the plan that published this summary is the only one placing now. With no
			// other epoch that has transients, nothing is kept and the version is packed from scratch.
			std::vector<uint32_t> keptEpochs;
			bool othersPlace = false;
			for (const auto& [epoch, summary] : m_summaries)
				if (epoch != kept->epoch) {
					keptEpochs.push_back(epoch);
					othersPlace = othersPlace || !summary->uses.empty();
				}
			const bool keep = m_settings.keepRunning && othersPlace;
			lock.unlock();
			std::shared_ptr<FramePlacementVersion> computed;
			const auto computeBegin = std::chrono::steady_clock::now();
			try {
				computed = FramePlacementVersion::Compute(inputs, id);
				if (computed && replacing) computed = KeepRunning(std::move(computed), replacing, std::move(keptEpochs), keep, "a graph's uncovered candidates");
				if (computed) computed->SetMemory(m_memory);
			} catch (const std::exception& error) {
				spdlog::error("Frame placement: the computation failed: {}", error.what());
			}
			computeMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - computeBegin).count();
			lock.lock();
			// Another thread may have published one meanwhile: that one stands.
			if (computed && m_settings.enabled && m_current == replacing) {
				if (m_computed < changes) m_computed = changes;
				m_latest = computed;
				m_current = computed;
				m_currentId.store(computed->Id(), std::memory_order_release);
				m_lastPublish = std::chrono::steady_clock::now();
				DropOfferLocked("a version was published at once for a plan");
				// A plan of this frame needs it now (its candidates are uncovered by the frame's version, which may give it no valid
				// layout at all), so it is the frame's version from here on. Only the worker's versions wait for the frame's end.
				if (m_frozen.load(std::memory_order_relaxed)) {
					m_frozenVersion = computed;
					m_frozenId.store(computed->Id(), std::memory_order_release);
				}
				firstEver = !replacing;
				first = computed;
			}
		}
	}
	m_wake.notify_all();
	if (first) {
		std::string epochs;
		for (const auto& summary : summaries) epochs += (epochs.empty() ? "" : ",") + std::to_string(summary->epoch);
		spdlog::info("Frame placement {}: published at once, {} (summaries of epochs {}), computed in {:.2f} ms{}", first->Id(),
			firstEver ? "the first" : "for a graph's uncovered candidates", epochs, computeMs,
			first->KeptFrom() ? "; it keeps the running graphs' resources where version " + std::to_string(first->KeptFrom()) + " has them"
							  : std::string());
		LogVersion(*first, summaries, detailed);
	}
}

std::shared_ptr<const FramePlacementVersion> FramePlacement::Current() const {
	std::lock_guard lock(m_mutex);
	return m_frozen.load(std::memory_order_relaxed) ? m_frozenVersion : m_current;
}

void FramePlacement::Freeze(std::shared_ptr<const FramePlacementVersion> version) {
	std::lock_guard lock(m_mutex);
	m_frozenVersion = std::move(version);
	m_frozenId.store(m_frozenVersion ? m_frozenVersion->Id() : 0, std::memory_order_release);
	m_frozen.store(true, std::memory_order_release);
}

void FramePlacement::Thaw() {
	std::lock_guard lock(m_mutex);
	m_frozen.store(false, std::memory_order_release);
	m_frozenVersion.reset();
}

FramePlacement::Staging::~Staging() {
	if (m_closed.load(std::memory_order_acquire)) return;
	if (const auto owner = m_owner.lock()) {
		{
			std::lock_guard lock(owner->m_mutex);
			owner->CloseStagingLocked(*this);
		}
		owner->m_wake.notify_all();
	}
}

void FramePlacement::CloseStagingLocked(Staging& staging) {
	if (staging.m_closed.exchange(true, std::memory_order_acq_rel)) return;
	if (m_openStagings) --m_openStagings;
}

std::shared_ptr<FramePlacement::Staging> FramePlacement::BeginStaging(std::vector<uint32_t> epochs) {
	std::shared_ptr<Staging> staging(new Staging());
	staging->m_owner = weak_from_this();
	staging->m_epochs = std::move(epochs);
	std::lock_guard lock(m_mutex);
	staging->m_base = m_current;
	// A set that rebuilds every epoch with transients (a recipe swap, a resize) leaves no graph running beside it after its attach:
	// it does not compact. Its versions still keep what its graphs share with the running ones (PublishStaged).
	staging->m_keep = m_settings.keepRunning && std::ranges::any_of(m_summaries, [&](const auto& entry) {
		return !entry.second->uses.empty() && std::ranges::find(staging->m_epochs, entry.first) == staging->m_epochs.end();
	});
	staging->m_currentId.store(m_current ? m_current->Id() : 0, std::memory_order_release);
	++m_openStagings;
	DropOfferLocked("a staged set began");
	return staging;
}

void FramePlacement::PublishStaged(Staging& staging, PlacementSummary summary) {
	auto published = std::make_shared<const PlacementSummary>(std::move(summary));
	std::unique_lock lock(m_mutex);
	if (staging.Closed()) return;
	auto& kept = staging.m_summaries[published->epoch];
	const bool changed = !kept || kept->hash != published->hash;
	kept = published;
	if (!changed || !m_settings.enabled || staging.m_declined || staging.m_collecting) return;
	// As Publish: the first version, or one for the candidates the staging's version leaves uncovered, computed at once. Here it
	// stays the staging's (pending) until its set attaches, so the running graphs keep the current version and their backings.
	const auto versionNow = staging.m_pending ? staging.m_pending : staging.m_base;
	const bool firstVersion = !versionNow && !kept->uses.empty();
	const bool uncovered = versionNow && kept->uncovered != 0;
	// A layout that kept the running graphs' resources in place may need clearly more than one packed from scratch (a heap lane
	// added for a graph whose resources were larger then). The set's graphs place anyway, so a version that keeps the running
	// graphs' resources and packs the set's own again (a lane nothing lies in any more goes) is made for them now, if it is
	// clearly smaller. Otherwise only the trim packs again, once quiet, and every graph places again then.
	const bool compact = !firstVersion && !uncovered && !staging.m_pending && versionNow && versionNow->KeptFrom() && staging.m_keep
		&& !kept->uses.empty();
	if (!firstVersion && !uncovered && !compact) return;
	// The host's summaries, with the staged epochs' in their place.
	auto merged = m_summaries;
	for (const auto& [epoch, staged] : staging.m_summaries) merged[epoch] = staged;
	FramePlacementInputs inputs;
	inputs.epochOrder = m_epochOrder;
	inputs.repeatedEpochs = m_settings.repeatedEpochs;
	inputs.beamSearch = m_settings.beamSearch;
	std::vector<std::shared_ptr<const PlacementSummary>> summaries;
	for (const auto& [epoch, s] : merged) summaries.push_back(s);
	inputs.summaries = summaries;
	const auto id = m_nextId++;
	const bool detailed = m_settings.logLifetimes;
	// Even a set that rebuilds every epoch (m_keep false: a resize, a recipe swap) keeps what its graphs share with the running ones
	// (pinEpochs: what the set's published graphs use, say the CLod RT CLAS build arena) and what its graphs that placed already put
	// somewhere. A version packed from scratch would move those under a running graph or under the set's own placed graph, and the
	// set's graph would decline it or leave its placement to the attach (a resize: 6-7 ms on the render thread, M5 step 2).
	const bool keep = staging.m_keep || m_settings.keepRunning;
	// The graphs that keep running: every epoch the set does not rebuild, and the set's graphs that placed already.
	std::vector<uint32_t> keptEpochs = staging.m_placedEpochs;
	for (const auto& [epoch, s] : merged)
		if (std::ranges::find(staging.m_epochs, epoch) == staging.m_epochs.end()) keptEpochs.push_back(epoch);
	// What stays where it is: what the graphs that keep running use, and what the set's new graphs that published use (their running
	// graphs bind it until the attach). What only the running graphs of the set's other epochs use goes with them at the attach.
	std::vector<uint32_t> pinEpochs = keptEpochs;
	for (const auto& [epoch, s] : staging.m_summaries) pinEpochs.push_back(epoch);
	lock.unlock();
	std::shared_ptr<FramePlacementVersion> computed;
	const auto computeBegin = std::chrono::steady_clock::now();
	try {
		computed = FramePlacementVersion::Compute(inputs, id);
		// A compaction that would not save memory, or could not keep the running graphs' resources, is not made.
		if (compact && computed && versionNow->Bytes() <= computed->Bytes() + FramePlacementMemory::TrimSlack(computed->Bytes())) computed.reset();
		if (computed && versionNow)
			computed = KeepRunning(std::move(computed), versionNow, std::move(keptEpochs), keep, compact ? "a staged set, to pack it again" : "a staged set",
				&pinEpochs);
		if (compact && computed
			&& (!computed->KeptFrom() || computed->Bytes() + FramePlacementMemory::TrimSlack(versionNow->Bytes()) >= versionNow->Bytes()))
			computed.reset();
		if (computed) computed->SetMemory(m_memory);
	} catch (const std::exception& error) {
		spdlog::error("Frame placement: the staged computation failed: {}", error.what());
	}
	const double computeMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - computeBegin).count();
	if (!computed) return;
	lock.lock();
	if (staging.Closed()) return;
	staging.m_pending = computed;
	staging.m_currentId.store(computed->Id(), std::memory_order_release);
	lock.unlock();
	std::string epochs;
	for (const auto& s : summaries) epochs += (epochs.empty() ? "" : ",") + std::to_string(s->epoch);
	spdlog::info("Frame placement {}: computed for a staged set, {} (summaries of epochs {}), in {:.2f} ms; published at the set's attach{}",
		computed->Id(), firstVersion ? "the first" : compact ? "to pack its graphs' transients again" : "for a graph's uncovered candidates", epochs, computeMs,
		computed->KeptFrom() ? "; it keeps the running graphs' resources where version " + std::to_string(computed->KeptFrom()) + " has them"
							 : std::string());
	LogVersion(*computed, summaries, detailed);
}

void FramePlacement::CollectStaged(Staging& staging, bool collecting) {
	std::lock_guard lock(m_mutex);
	staging.m_collecting = collecting;
}

std::shared_ptr<const FramePlacementVersion> FramePlacement::StagedCurrent(const Staging& staging) const {
	std::lock_guard lock(m_mutex);
	return staging.m_pending ? staging.m_pending : staging.m_base;
}

bool FramePlacement::DeclineStaged(Staging& staging) {
	uint64_t declined = 0;
	{
		std::lock_guard lock(m_mutex);
		// Declined already: the set's later graphs place from the base too.
		if (staging.m_declined) return !staging.Closed();
		if (staging.Closed() || staging.m_pendingUsed || !staging.m_pending) return false;
		declined = staging.m_pending->Id();
		staging.m_declined = true;
		staging.m_pending.reset();
		staging.m_currentId.store(staging.m_base ? staging.m_base->Id() : 0, std::memory_order_release);
	}
	// Its heaps stay this memory's current ones: the worker's next version, which places the set's new transients, takes them.
	spdlog::info("Frame placement {}: declined by its staged set (it would move resources the running graphs bind); the set places from {}",
		declined, staging.m_currentId.load(std::memory_order_acquire));
	return true;
}

void FramePlacement::NoteStagedPlacement(Staging& staging, uint64_t versionId, uint32_t epoch, bool placed) {
	std::lock_guard lock(m_mutex);
	if (staging.m_pending && staging.m_pending->Id() == versionId) staging.m_pendingUsed = true;
	if (placed && epoch != UINT32_MAX && std::ranges::find(staging.m_placedEpochs, epoch) == staging.m_placedEpochs.end())
		staging.m_placedEpochs.push_back(epoch);
}

std::shared_ptr<FramePlacementVersion> FramePlacement::KeepRunning(std::shared_ptr<FramePlacementVersion> fresh,
	const std::shared_ptr<const FramePlacementVersion>& base, std::vector<uint32_t> keptEpochs, bool keep, const char* what,
	const std::vector<uint32_t>* pinEpochs) const {
	if (!keep || !fresh || !base) return fresh;
	// What the graphs bind now: every resource still where a frame placement put it.
	std::vector<FramePlacementPin> pins;
	for (const auto& resident : FramePlacedResidents())
		if (resident.objectId) pins.push_back({resident.objectId, resident.heap, resident.offset, resident.size});
	std::string why;
	auto keeping = FramePlacementVersion::Keeping(*fresh, *base, pins, keptEpochs, fresh->Id(), &why, pinEpochs);
	if (!keeping) {
		spdlog::info("Frame placement {}: for {}, packed from scratch: it cannot keep the running graphs' resources where version {} has "
					 "them ({}); every graph places again from it",
			fresh->Id(), what, base->Id(), why);
		return fresh;
	}
	uint64_t laneBytes = 0;
	for (const auto& block : keeping->Blocks())
		if (block.lane) laneBytes += block.capacity;
	std::string moved;
	for (size_t i = 0; i < keeping->KeptMoved().size(); ++i) {
		moved += (moved.empty() ? "'" : ", '") + keeping->KeptMoved()[i] + "'";
		if (i < keeping->KeptMovedWith().size()) moved += " (with '" + keeping->KeptMovedWith()[i] + "')";
	}
	spdlog::info("Frame placement {}: for {}, keeps the running graphs' resources where version {} has them{}: {:.2f} MB in {} blocks "
				 "({:.2f} MB in added heap lanes) against {:.2f} MB packed from scratch",
		keeping->Id(), what, base->Id(),
		keeping->KeptMovedCount() ? fmt::format(" but {} that are now live with another in their bytes ({}{})", keeping->KeptMovedCount(), moved,
										keeping->KeptMovedCount() > keeping->KeptMoved().size() ? ", ..." : "")
								  : std::string(),
		MB(keeping->Bytes()), keeping->Blocks().size(), MB(laneBytes), MB(fresh->Bytes()));
	return keeping;
}

void FramePlacement::AcquireStagedBlocks(const Staging& staging) {
	std::shared_ptr<const FramePlacementVersion> pending;
	{
		std::lock_guard lock(m_mutex);
		pending = staging.m_pending;
	}
	if (!pending) return;
	// The memory keeps each class's current heap, so a block no graph placed into yet stays allocated until the attach takes it.
	for (uint32_t block = 0; block < pending->Blocks().size(); ++block) (void)pending->AcquireBlock(block);
}

bool FramePlacement::CommitStaging(Staging& staging) {
	std::shared_ptr<const FramePlacementVersion> published;
	std::string epochs;
	{
		std::lock_guard lock(m_mutex);
		if (staging.Closed()) return false;
		const auto now = std::chrono::steady_clock::now();
		for (const auto& [epoch, summary] : staging.m_summaries) {
			auto& kept = m_summaries[epoch];
			if (!kept || kept->hash != summary->hash) {
				++m_changes;
				m_lastChange = now;
			}
			kept = summary;
			epochs += (epochs.empty() ? "" : ",") + std::to_string(epoch);
		}
		// The set's graphs placed from it. The worker computes again from the summaries once they settle; the same inputs give the
		// same layout, which it does not publish again.
		if (staging.m_pending && m_settings.enabled) {
			m_latest = staging.m_pending;
			m_current = staging.m_pending;
			m_currentId.store(m_current->Id(), std::memory_order_release);
			m_lastPublish = now;
			published = m_current;
		}
		if (!m_thread.joinable() && !m_stopping && !m_summaries.empty()) m_thread = std::thread([this] { Run(); });
		CloseStagingLocked(staging);
	}
	m_wake.notify_all();
	if (published)
		spdlog::info("Frame placement {}: published at the attach of the staged set that placed from it (summaries of epochs {})", published->Id(),
			epochs);
	return static_cast<bool>(published);
}

void FramePlacement::DropStaging(Staging& staging) {
	{
		std::lock_guard lock(m_mutex);
		CloseStagingLocked(staging);
	}
	m_wake.notify_all();
}

std::shared_ptr<const FramePlacementVersion> FramePlacement::Offered() const {
	std::lock_guard lock(m_mutex);
	return m_offered;
}

bool FramePlacement::PublishOffered(uint64_t id) {
	{
		std::lock_guard lock(m_mutex);
		if (!m_offered || m_offered->Id() != id) return false;
		if (m_openStagings || !m_settings.enabled || m_changes != m_computed || m_current != m_offeredOver) {
			DropOfferLocked("it went stale before its publication");
			return false;
		}
		PublishOfferLocked();
	}
	m_wake.notify_all();
	return true;
}

void FramePlacement::PublishOfferLocked() {
	if (!m_offered) return;
	if (m_offeredTrim) {
		// The heaps no block uses go now: their placements keep them alive until they move.
		m_memory->ReleaseUnused(m_offeredNeeds);
		m_trimChecked = m_offered->Id();
	}
	m_current = m_offered;
	m_currentId.store(m_current->Id(), std::memory_order_release);
	m_lastPublish = std::chrono::steady_clock::now();
	spdlog::info("Frame placement {}: published{} (offered {:.1f} ms before); graphs place from it at their next preparation", m_current->Id(),
		m_offeredTrim ? " as the trim of " + std::to_string(m_offeredOver ? m_offeredOver->Id() : 0) : std::string(),
		std::chrono::duration<double, std::milli>(m_lastPublish - m_offeredAt).count());
	m_offered.reset();
	m_offeredOver.reset();
	m_offeredNeeds.clear();
	m_offeredTrim = false;
}

void FramePlacement::DropOfferLocked(const char* why) {
	if (!m_offered) return;
	spdlog::info("Frame placement {}: offer dropped ({}); the worker computes again", m_offered->Id(), why);
	// The worker checks the trim again for the current version, or computes the layout again (a dropped set changes no summary).
	if (m_offeredTrim) m_trimChecked = 0;
	else if (m_computed && m_computed == m_changes) --m_computed;
	m_offered.reset();
	m_offeredOver.reset();
	m_offeredNeeds.clear();
	m_offeredTrim = false;
}

std::shared_ptr<const FramePlacementVersion> FramePlacement::Latest() const {
	std::lock_guard lock(m_mutex);
	return m_latest;
}

std::vector<std::shared_ptr<const PlacementSummary>> FramePlacement::Summaries() const {
	std::lock_guard lock(m_mutex);
	std::vector<std::shared_ptr<const PlacementSummary>> out;
	out.reserve(m_summaries.size());
	for (const auto& [epoch, summary] : m_summaries) out.push_back(summary);
	return out;
}

void FramePlacement::Log(bool detailed) const {
	const auto latest = Latest();
	if (!latest) {
		spdlog::info("Frame placement: nothing computed yet ({} usage summaries)", Summaries().size());
		return;
	}
	LogVersion(*latest, Summaries(), detailed);
}

bool FramePlacement::MemoryLogDue() {
	std::lock_guard lock(m_mutex);
	if (!m_settings.logLifetimes || m_settings.memoryLogPeriod.count() <= 0) return false;
	const auto now = std::chrono::steady_clock::now();
	// The first call starts the first period.
	if (m_nextMemoryLog == std::chrono::steady_clock::time_point{}) {
		m_nextMemoryLog = now + m_settings.memoryLogPeriod;
		return false;
	}
	if (now < m_nextMemoryLog) return false;
	m_nextMemoryLog = now + m_settings.memoryLogPeriod;
	return true;
}

void FramePlacement::LogMemory(PersistentAliasArena* pools, const std::string& graphs) const {
	std::shared_ptr<FrameAliasArena> arena;
	std::shared_ptr<const FramePlacementVersion> current;
	{
		std::lock_guard lock(m_mutex);
		arena = m_frameArena.lock();
		current = m_current;
	}
	// The resources still placed from a frame placement, by heap generation.
	struct Holders {
		size_t count = 0;
		uint64_t bytes = 0;
		std::map<uint32_t, size_t> byEpoch;
		std::vector<std::string> names;
	};
	std::map<const AliasHeapGeneration*, Holders> holders;
	for (const auto& resident : FramePlacedResidents()) {
		auto& held = holders[resident.heap];
		++held.count;
		held.bytes += resident.size;
		++held.byEpoch[resident.epoch];
		if (held.names.size() < 4) held.names.push_back(resident.name);
	}
	const auto heaps = m_memory->Heaps();
	uint64_t currentBytes = 0, replacedBytes = 0;
	for (const auto& heap : heaps) (heap.current ? currentBytes : replacedBytes) += heap.capacity;
	uint64_t arenaBytes = 0, arenaReserved = 0;
	size_t arenaBlocks = 0, arenaRanges = 0;
	if (arena) {
		for (const auto& block : arena->Blocks()) {
			++arenaBlocks;
			arenaBytes += block.capacity;
			// Ranges whose spans do not overlap share bytes, so the bytes in use are the highest range end, not the sum.
			uint64_t end = 0;
			for (const auto& range : block.ranges) {
				++arenaRanges;
				end = (std::max)(end, range.end);
			}
			arenaReserved += end;
		}
	}
	uint64_t poolBytes = 0, poolRequired = 0;
	size_t poolCount = 0, poolsStale = 0;
	std::string poolList;
	if (pools) {
		auto& locked = *pools;
		std::lock_guard lock(locked.mutex);
		for (const auto& [poolID, pool] : locked.pools) {
			++poolCount;
			poolBytes += pool.capacityBytes;
			const auto required = PersistentAliasArena::Required(pool);
			poolRequired += required;
			poolList += (poolList.empty() ? "" : "; ") + fmt::format("{:#x} class {} {:.2f}/{:.2f} MB gen {} graphs {}{}", poolID, pool.resourceClass,
				MB(required), MB(pool.capacityBytes), pool.generation, pool.required.size(), pool.idleSinceFrame ? " idle" : "");
		}
		poolsStale = locked.stale.size();
	}
	spdlog::info("Frame placement memory: placement heaps {:.2f} MB current + {:.2f} MB replaced and still alive ({} heaps); frame arena "
				 "{:.2f} MB in {} blocks, {:.2f} MB of it reserved by {} ranges; shared arena pools {:.2f} MB in {} pools, {:.2f} MB required, "
				 "{} graphs with placements in replaced pool generations; current version {}{}{}",
		MB(currentBytes), MB(replacedBytes), heaps.size(), MB(arenaBytes), arenaBlocks, MB(arenaReserved), arenaRanges, MB(poolBytes), poolCount,
		MB(poolRequired), poolsStale, current ? current->Id() : 0, graphs.empty() ? "" : "; graphs: ", graphs);
	for (const auto& heap : heaps) {
		const auto found = holders.find(heap.identity);
		std::string held = "no resources";
		if (found != holders.end()) {
			std::string epochs;
			for (const auto& [epoch, count] : found->second.byEpoch) epochs += (epochs.empty() ? "" : ",") + fmt::format("{}x{}", epoch, count);
			std::string names;
			for (const auto& name : found->second.names) names += (names.empty() ? "'" : ", '") + name + "'";
			held = fmt::format("{} resources {:.2f} MB (by placing epoch {}: {}{})", found->second.count, MB(found->second.bytes), epochs, names,
				found->second.count > found->second.names.size() ? ", ..." : "");
		}
		// What the current version packs for the heap's class and backends.
		uint64_t packed = 0;
		if (current)
			for (const auto& block : current->Blocks())
				if (block.memoryClass == heap.memoryClass && block.backendMask == heap.backendMask && block.lane == heap.lane) packed += block.capacity;
		spdlog::info("Frame placement memory: heap class {} backends {:#x} lane {} generation {} {}: reserved {:.2f} MB, current version packs "
					 "{:.2f} MB{}; held by {} (references {})",
			heap.memoryClass, heap.backendMask, heap.lane, heap.generation, heap.current ? "current" : "replaced", MB(heap.capacity), MB(packed),
			heap.current && heap.needed ? fmt::format(", last taken by version {} for {:.2f} MB", heap.neededBy, MB(heap.needed)) : std::string(),
			held, heap.references);
	}
	if (!poolList.empty()) spdlog::info("Frame placement memory: shared arena pools: {}", poolList);
}

std::optional<std::chrono::steady_clock::time_point> FramePlacement::TrimDueLocked() const {
	if (!m_settings.enabled || !m_current || m_settings.trimAfter.count() <= 0 || m_trimChecked == m_current->Id()) return std::nullopt;
	return (std::max)(m_lastChange, m_lastPublish) + m_settings.trimAfter;
}

void FramePlacement::Run() {
	std::unique_lock lock(m_mutex);
	while (!m_stopping) {
		// A staged set places from the current version (or its own pending one): nothing is published until it attaches or goes.
		if (m_openStagings) {
			m_wake.wait(lock);
			continue;
		}
		// An offered version waits for the host to publish it at a frame boundary, once the graphs prepared their re-placements. A change
		// makes it stale; an offer no frame boundary took is published here after the timeout, as without offers.
		if (m_offered) {
			if (m_changes != m_computed || m_current != m_offeredOver) {
				DropOfferLocked("the summaries or the current version changed");
				continue;
			}
			const auto due = m_offeredAt + m_settings.offerTimeout;
			if (std::chrono::steady_clock::now() < due) {
				m_wake.wait_until(lock, due);
				continue;
			}
			spdlog::warn("Frame placement {}: offered {:.0f} ms ago and not published at a frame boundary; published now", m_offered->Id(),
				std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - m_offeredAt).count());
			PublishOfferLocked();
			continue;
		}
		if (m_changes == m_computed) {
			// Quiet. Once the current version has been quiet for the trim time, heaps larger than its need (growth headroom, a
			// class it no longer uses) are trimmed.
			const auto trimDue = TrimDueLocked();
			if (!trimDue) {
				m_wake.wait(lock);
				continue;
			}
			if (std::chrono::steady_clock::now() < *trimDue) {
				m_wake.wait_until(lock, *trimDue);
				continue;
			}
			const auto current = m_current;
			m_trimChecked = current->Id();
			// The latest layout (the current one, or a newer one the current version served in place of) on exact heaps, when
			// that trims a heap or is a newer layout.
			auto latest = m_latest ? m_latest : current;
			// A layout that kept the running graphs' resources in place (Keeping) stays while it needs no more than the layout
			// packed from scratch: packing again would make every graph place again for nothing. Only its heaps are trimmed.
			const bool keepCurrent = current->KeptFrom() && latest != current
				&& current->Bytes() <= latest->Bytes() + FramePlacementMemory::TrimSlack(latest->Bytes());
			if (keepCurrent) latest = current;
			const auto needs = latest->BlockNeeds();
			if (!m_memory->WouldTrim(needs) && latest->LayoutHash() == current->LayoutHash()) continue;
			// The same layout under a new id, its blocks on exact heaps. Graphs take it in place at their next preparation, as
			// any new version. The heaps no block uses go now: their placements keep them alive until they move.
			std::vector<std::shared_ptr<const PlacementSummary>> summaries;
			const auto inputs = InputsLocked(summaries);
			const auto id = m_nextId++;
			lock.unlock();
			std::shared_ptr<FramePlacementVersion> trimmed;
			try {
				trimmed = keepCurrent ? FramePlacementVersion::Relabel(*current, id) : FramePlacementVersion::Compute(inputs, id);
			} catch (const std::exception& error) {
				spdlog::error("Frame placement: the trim's computation failed: {}", error.what());
			}
			if (trimmed) {
				trimmed->SetMemory(m_memory);
				trimmed->SetExactHeaps(true);
				// Its heaps are made here, on this thread: otherwise the first graph that places from it allocates them at its
				// preparation, on the render thread (a heap of a GB took 20-40 ms there).
				const auto acquireBegin = std::chrono::steady_clock::now();
				try {
					for (uint32_t block = 0; block < trimmed->Blocks().size(); ++block) (void)trimmed->AcquireBlock(block);
				} catch (const std::exception& error) {
					spdlog::warn("Frame placement {}: its heaps were not allocated ahead ({}); the graphs allocate them", trimmed->Id(), error.what());
				}
				const double acquireMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - acquireBegin).count();
				if (acquireMs > 1.0) spdlog::info("Frame placement {}: heaps allocated ahead of its publication in {:.2f} ms", trimmed->Id(), acquireMs);
			}
			lock.lock();
			// Only while nothing changed meanwhile; otherwise a new computation is due. The trim takes the latest layout, which the
			// current version may have served in its place (Run's serves).
			if (!trimmed || m_changes != m_computed || m_current != current) continue;
			if (m_settings.offerVersions) {
				// The graphs prepare their re-placements for it first (Offered); its unused heaps go at its publication.
				m_latest = trimmed;
				m_offered = trimmed;
				m_offeredOver = current;
				m_offeredAt = std::chrono::steady_clock::now();
				m_offeredTrim = true;
				m_offeredNeeds = needs;
				spdlog::info("Frame placement {}: offered once quiet, on heaps of its need ({:.2f} MB; the current version is {}); published at a frame "
							 "boundary once the graphs prepared their re-placements for it", trimmed->Id(), MB(trimmed->Bytes()), current->Id());
				continue;
			}
			m_memory->ReleaseUnused(needs);
			m_latest = trimmed;
			m_current = trimmed;
			m_currentId.store(trimmed->Id(), std::memory_order_release);
			m_lastPublish = std::chrono::steady_clock::now();
			m_trimChecked = trimmed->Id();
			lock.unlock();
			spdlog::info("Frame placement {}: published once quiet, on heaps of its need ({:.2f} MB; the current version was {}); graphs "
						 "place from it at their next preparation",
				trimmed->Id(), MB(trimmed->Bytes()), current->Id());
			lock.lock();
			continue;
		}
		// Wait until the summaries have been quiet for the settle time, so a burst of builds gives one computation.
		const auto due = m_lastChange + m_settings.settle;
		if (std::chrono::steady_clock::now() < due) {
			m_wake.wait_until(lock, due);
			continue;
		}
		// Without a plan's summary that made one at once, the first version waits for a summary of every frame epoch
		// (SetFrameEpochs), so no version comes from no summary at all.
		if (!m_current && !SummariesCompleteLocked()) {
			m_computed = m_changes;
			continue;
		}
		std::vector<std::shared_ptr<const PlacementSummary>> summaries;
		const auto inputs = InputsLocked(summaries);
		const auto changes = m_changes;
		const bool detailed = m_settings.logLifetimes;
		const auto id = m_nextId++;
		lock.unlock();
		std::shared_ptr<const FramePlacementVersion> version;
		try {
			auto computed = FramePlacementVersion::Compute(inputs, id);
			if (computed) computed->SetMemory(m_memory);
			version = std::move(computed);
		} catch (const std::exception& error) {
			spdlog::error("Frame placement: the computation failed: {}", error.what());
		}
		lock.lock();
		m_computed = changes;
		if (!version) continue;
		// An unchanged layout keeps the version already published, so graphs see no change.
		const bool newLayout = !m_latest || m_latest->LayoutHash() != version->LayoutHash();
		if (newLayout) m_latest = version;
		// Each new version makes every graph place again (a structural edit at its next preparation). While the current one still
		// places every candidate of every graph (each rebuilt graph found its entries or fitted its new transients into free
		// bytes), a new layout is taken only when it needs clearly less memory. The trim publishes the settled layout later.
		const bool uncovered = std::ranges::any_of(summaries, [](const auto& summary) { return summary->uncovered != 0; });
		// A layout that kept the running graphs' resources in place (Keeping) is packed again only by the trim, once quiet: every
		// graph places again for it, and while settings are toggled the next kept layout often needs less anyway.
		const bool smaller = m_current && !m_current->KeptFrom()
			&& m_latest->Bytes() + FramePlacementMemory::TrimSlack(m_current->Bytes()) < m_current->Bytes();
		const bool serves = m_current && m_current->Slots().size() == m_latest->Slots().size() && !uncovered && !smaller;
		bool published = false;
		std::shared_ptr<const FramePlacementVersion> toOffer, offerOver;
		if (m_settings.enabled && (!m_current || m_current->LayoutHash() != m_latest->LayoutHash()) && !serves) {
			if (m_settings.offerVersions && m_current) {
				toOffer = m_latest;
				offerOver = m_current;
			} else {
				m_current = m_latest;
				m_currentId.store(m_current->Id(), std::memory_order_release);
				m_lastPublish = std::chrono::steady_clock::now();
				published = true;
			}
		}
		if (!newLayout && !published && !toOffer) continue;
		const auto toLog = m_latest;
		const bool allocateOffered = m_settings.allocateOffered;
		lock.unlock();
		if (published) spdlog::info("Frame placement {}: published; graphs place from it at their next alias plan", toLog->Id());
		if (newLayout) LogVersion(*toLog, summaries, detailed);
		if (toOffer && allocateOffered) {
			// Its heaps are made here, as the trim's are: the graphs' preparations place into them, and no graph allocates one.
			try {
				for (uint32_t block = 0; block < toOffer->Blocks().size(); ++block) (void)toOffer->AcquireBlock(block);
			} catch (const std::exception& error) {
				spdlog::warn("Frame placement {}: its heaps were not allocated ahead ({})", toOffer->Id(), error.what());
			}
		}
		lock.lock();
		// Offered only while nothing changed meanwhile.
		if (toOffer && !m_offered && !m_openStagings && m_changes == changes && m_current == offerOver && m_latest == toOffer) {
			m_offered = toOffer;
			m_offeredOver = offerOver;
			m_offeredAt = std::chrono::steady_clock::now();
			m_offeredTrim = false;
			m_offeredNeeds.clear();
			spdlog::info("Frame placement {}: offered; published at a frame boundary once the graphs prepared their re-placements for it", toOffer->Id());
		}
	}
}

void FramePlacement::LogVersion(const FramePlacementVersion& version, const std::vector<std::shared_ptr<const PlacementSummary>>& summaries,
	bool detailed) const {
	std::shared_ptr<FrameAliasArena> arena;
	{
		std::lock_guard lock(m_mutex);
		arena = m_frameArena.lock();
	}
	// Today's own pools. Epochs in the shared arena pack from offset 0 of the same pools, so the arena needs the largest epoch
	// of each memory class. Private pools add up. The frame arena's blocks are read as they are now.
	std::array<uint64_t, 4> sharedByClass{};
	uint64_t privateBytes = 0, frameArenaPlanned = 0, framePlaced = 0;
	for (const auto& summary : summaries) {
		framePlaced += summary->framePlacedBytes;
		switch (summary->memory) {
		case PlacementMemory::SharedArena:
			for (size_t c = 0; c < sharedByClass.size(); ++c) sharedByClass[c] = (std::max)(sharedByClass[c], summary->pooledBytesByClass[c]);
			break;
		case PlacementMemory::PrivatePools: privateBytes += summary->pooledBytes; break;
		case PlacementMemory::FrameArena: frameArenaPlanned += summary->pooledBytes; break;
		}
	}
	uint64_t sharedBytes = 0;
	for (const auto bytes : sharedByClass) sharedBytes += bytes;
	const uint64_t arenaBytes = arena ? arena->Bytes() : 0;
	size_t placed = 0, placedMulti = 0, previousFrame = 0, gated = 0, other = 0;
	uint64_t placedBytes = 0;
	for (const auto& entry : version.Entries()) {
		if (entry.placed) {
			++placed;
			placedBytes += entry.size;
			if (entry.MultiEpoch()) ++placedMulti;
		} else if (entry.reason.starts_with("previous-frame read")) ++previousFrame;
		else if (entry.reason.find("may not write") != std::string::npos) ++gated;
		else ++other;
	}
	spdlog::info("Frame placement {}: {} epochs, {} candidates: {} placed ({} used by several epochs), {} previous-frame reads, {} with a "
				 "first writer that may not write, {} not eligible. Whole-frame packing needs {:.2f} MB in {} blocks (buffers {:.2f}, "
				 "targets {:.2f}, textures {:.2f}) for {:.2f} MB of resources. Today's own pools: shared arena {:.2f} MB (the largest "
				 "epoch per class), private pools {:.2f} MB, frame arena {:.2f} MB of blocks ({:.2f} MB planned by its epochs); placed from "
				 "a frame placement now {:.2f} MB",
		version.Id(), version.Slots().size(), version.Entries().size(), placed, placedMulti, previousFrame, gated, other, MB(version.Bytes()),
		version.Blocks().size(), MB(version.BytesOfClass(1)), MB(version.BytesOfClass(2)), MB(version.BytesOfClass(3)), MB(placedBytes),
		MB(sharedBytes), MB(privateBytes), MB(arenaBytes), MB(frameArenaPlanned), MB(framePlaced));
	// Resources several epochs use whose first use in frame order reads: either they hold content across frames, or an epoch
	// reads before the frame writes them. Always worth a line.
	for (const auto& entry : version.Entries()) {
		if (!entry.MultiEpoch() || !entry.reason.starts_with("previous-frame read")) continue;
		std::string epochs;
		for (const auto& use : entry.uses) epochs += (epochs.empty() ? "" : ",") + std::to_string(use.epoch);
		spdlog::info("Frame placement {} finding: '{}' (epochs {}) {}", version.Id(), entry.name, epochs, entry.reason);
	}
	if (!detailed) return;
	for (const auto& slot : version.Slots())
		spdlog::info("Frame placement {} slot: epoch {} at {} ranks={}{} memory={}", version.Id(), slot.epoch, slot.base, slot.rankCount,
			slot.repeated ? " repeated" : "", MemoryName(slot.memory));
	for (const auto& entry : version.Entries()) {
		std::string uses;
		for (const auto& use : entry.uses)
			uses += (uses.empty() ? "" : " ") + std::to_string(use.epoch) + ":" + std::to_string(use.firstRank) + "-" + std::to_string(use.lastRank)
				+ (use.firstUseWrites ? "w" : "r");
		if (entry.placed) {
			const auto& block = version.Blocks()[entry.block];
			spdlog::info("Frame placement {} resource '{}': uses {} frame {}-{} bytes={} block {:#x} [{}, {}){}", version.Id(), entry.name, uses,
				entry.frameFirst, entry.frameLast, entry.size, block.poolID, entry.offset, entry.offset + entry.size,
				entry.acrossRuns ? " across runs" : "");
		} else {
			spdlog::info("Frame placement {} resource '{}': uses {} frame {}-{} bytes={} not placed: {}", version.Id(), entry.name, uses,
				entry.frameFirst, entry.frameLast, entry.size, entry.reason);
		}
	}
}

}  // namespace org
