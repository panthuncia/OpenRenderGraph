#pragma once

#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "Render/PipelineLibrary.h"

namespace org {

/// @brief What PipelinePermutations::Select does when the pipeline has no build on the device yet.
enum class PipelineMiss : uint8_t {
	Skip, ///< Start the build and return without a pipeline. The caller skips the work that needs it.
	Wait, ///< Start the build and block until it ends.
};

/// @brief The result of a selection: the device's build when it is ready, and the build's status.
struct PipelineSelection {
	std::shared_ptr<const PipelineStatePayload> payload; ///< Set only when status is Ready.
	PipelineBuildStatus status = PipelineBuildStatus::Idle;
	explicit operator bool() const noexcept { return payload != nullptr; }
};

/// @brief Pipelines a pass picks while it prepares a frame, by a key of its own, such as a shader permutation.
///
/// Pipelines a pass declares in Declare are built before the pass is published. That does not work when the pipeline
/// depends on the frame's draws, as it does for engine shaders with many permutations. A pass keeps one of these instead,
/// and selects in Prepare:
///
/// - The key is the caller's. It must identify the description: equal keys must describe the same pipeline. A small key,
///   such as a technique id and state bits, avoids building a description on every lookup.
/// - The description is built only on a miss, and is acquired from the runtime's PipelineLibrary. So passes that describe
///   the same pipeline share one build, and reloading rebuilds these pipelines like any other.
/// - A miss starts the device's build. With PipelineMiss::Skip the selection is empty until the build is ready.
///
/// The prepared frame keeps the selected payload, and RecordingContext::BindGraphicsPipeline(payload) retains it until the
/// recording retires. So releasing a pipeline here never frees one that recorded work still uses.
///
/// Select may be called from several threads at once.
class PipelinePermutations {
public:
	/// @param library The runtime's library (Runtime::Pipelines, or RenderPassBuilder::Pipelines in Declare).
	/// @param name Names the set in failure messages.
	explicit PipelinePermutations(std::shared_ptr<PipelineLibrary> library, std::string name = {});

	/// @brief The pipeline of `key` on `device`.
	/// @param describe Called on a miss only. Returns the PipelineDesc of the key.
	/// @param miss Whether to block on a build that has not ended.
	template<class Describe>
	PipelineSelection Select(uint64_t key, DeviceInstanceId device, Describe&& describe, PipelineMiss miss = PipelineMiss::Skip) {
		auto entry = Find(key);
		if (!entry) entry = Insert(key, PipelineDesc(describe()));
		return Select(key, entry, device, miss);
	}

	/// @brief The entry of a key, or null when it was never selected. Builds nothing.
	PipelineLibraryEntryPtr Find(uint64_t key) const;
	/// @brief Starts the device's build of the key ahead of its first use, such as for permutations a loading screen knows.
	void Prewarm(uint64_t key, DeviceInstanceId device, PipelineDesc desc);
	/// @brief How many keys the set holds.
	size_t Size() const;
	/// @brief Releases every key. Builds in flight finish in the library. Recordings keep the payloads they bound.
	void Clear();

private:
	PipelineLibraryEntryPtr Insert(uint64_t key, PipelineDesc desc);
	PipelineSelection Select(uint64_t key, const PipelineLibraryEntryPtr& entry, DeviceInstanceId device, PipelineMiss miss);

	std::shared_ptr<PipelineLibrary> m_library;
	std::string m_name;
	mutable std::shared_mutex m_mutex;
	std::unordered_map<uint64_t, PipelineLibraryEntryPtr> m_entries;
	std::unordered_set<uint64_t> m_reportedFailures; // Keys whose failure was logged. A failure is logged once.
};

} // namespace org
