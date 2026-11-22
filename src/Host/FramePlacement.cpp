#include "FramePlacement.h"

#include <algorithm>
#include <array>
#include <string>

#include <spdlog/spdlog.h>

#include "Render/RenderGraph/FrameAliasArena.h"

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

void FramePlacement::Publish(PlacementSummary summary) {
	auto published = std::make_shared<const PlacementSummary>(std::move(summary));
	{
		std::lock_guard lock(m_mutex);
		auto& kept = m_summaries[published->epoch];
		const bool changed = !kept || kept->hash != published->hash;
		// The same uses give the same placement. The new summary is kept for its pooled bytes alone.
		kept = std::move(published);
		if (!changed) return;
		++m_changes;
		m_lastChange = std::chrono::steady_clock::now();
		if (!m_thread.joinable() && !m_stopping) m_thread = std::thread([this] { Run(); });
	}
	m_wake.notify_all();
}

std::shared_ptr<const FramePlacementVersion> FramePlacement::Current() const {
	std::lock_guard lock(m_mutex);
	return m_current;
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

void FramePlacement::Run() {
	std::unique_lock lock(m_mutex);
	while (!m_stopping) {
		if (m_changes == m_computed) {
			m_wake.wait(lock);
			continue;
		}
		// Wait until the summaries have been quiet for the settle time, so a burst of builds gives one computation.
		const auto due = m_lastChange + m_settings.settle;
		if (std::chrono::steady_clock::now() < due) {
			m_wake.wait_until(lock, due);
			continue;
		}
		FramePlacementInputs inputs;
		inputs.epochOrder = m_epochOrder;
		inputs.repeatedEpochs = m_settings.repeatedEpochs;
		inputs.beamSearch = m_settings.beamSearch;
		std::vector<std::shared_ptr<const PlacementSummary>> summaries;
		for (const auto& [epoch, summary] : m_summaries) summaries.push_back(summary);
		inputs.summaries = summaries;
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
		bool published = false;
		if (m_settings.enabled && (!m_current || m_current->LayoutHash() != m_latest->LayoutHash())) {
			m_current = m_latest;
			m_currentId.store(m_current->Id(), std::memory_order_release);
			published = true;
		}
		if (!newLayout && !published) continue;
		const auto toLog = m_latest;
		lock.unlock();
		if (published) spdlog::info("Frame placement {}: published; graphs place from it at their next alias plan", toLog->Id());
		if (newLayout) LogVersion(*toLog, summaries, detailed);
		lock.lock();
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
