#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <vector>

#include "Render/LatchedCopies.h"
#include "RenderPasses/Base/TypedRenderGraphPass.h"

namespace org {

// A pass that copies a run's latched values to their targets (RecordLatchedCopies). It belongs at the start of the work
// that reads them, and it prefers the graphics queue.
// - targets() is called when the pass is declared. Each target is declared as a copy destination. A copy to a target it
//   did not return is skipped.
// - source() is called when the pass is prepared, once per run. It names what the run copies.
// - copyFixed false leaves out the fixed lists of every source.
// - declared, when set, receives the declared targets' keys at each declaration, so a host can latch only what the pass
//   will copy.
// The pass opts into kept-recording reuse: its commands depend only on the source and the run's regions.
std::shared_ptr<RenderPass> MakeLatchedCopiesPass(bool copyFixed, std::function<std::vector<LatchedTarget>()> targets,
	std::function<LatchedSource(const PassPrepareContext&)> source,
	std::atomic<std::shared_ptr<const std::vector<const void*>>>* declared = nullptr);

} // namespace org
