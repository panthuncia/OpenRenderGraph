#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "Resources/AliasingPlacement.h"
#include "Resources/TrackedAllocation.h"

namespace org {

/**
 * @brief The alias pools persistent graphs place their transient resources into (RenderGraph::SetPersistentAliasArena).
 * Each graph packs its own transients from offset 0 of a pool. Graphs that share an arena share its pools, so the
 * transients of different graphs overlap the same memory. The runtime's alias ledger orders them: a placement's first use
 * in one graph's execution waits for the last use of any placement it overlaps in another graph's earlier execution.
 *
 * A pool is as large as the largest requirement of any graph placed in it. When a graph needs more, the pool grows into a
 * new generation. The placements made in an earlier generation keep that memory alive, and share nothing with the new
 * generation, until their graph places them again (PersistentAliasPlacementsCurrent).
 */
class PersistentAliasArena {
public:
	struct Pool {
		TrackedHandle allocation;
		std::shared_ptr<const AliasHeapGeneration> heap;
		uint64_t capacityBytes = 0, alignment = 1, generation = 0;
		uint8_t resourceClass = 0;
		// Each graph's latest requirement (its plan's heap end), by the graph's state.
		std::unordered_map<const void*, uint64_t> required;
		uint64_t idleSinceFrame = 0;  // The frame the last graph placed into it stopped placing; 0 while placed into.
	};

	static std::shared_ptr<PersistentAliasArena> Create() { return std::make_shared<PersistentAliasArena>(); }

	// The largest requirement any graph has in a pool.
	static uint64_t Required(const Pool& pool) noexcept {
		uint64_t required = 0;
		for (const auto& [owner, bytes] : pool.required) required = (std::max)(required, bytes);
		return required;
	}

	std::mutex mutex;
	std::unordered_map<uint64_t, Pool> pools;
	// Graphs (their states) with placements in a pool's earlier generation: set when the pool grows, cleared by their next plan.
	std::unordered_set<const void*> stale;
};

}  // namespace org
