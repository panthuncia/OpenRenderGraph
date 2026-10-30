#pragma once
#include <map>

#include "Render/RenderGraph/RenderGraph.h"
#include "Render/RenderGraph/ExperimentalGraphCompiler.h"
#include "Render/RenderGraph/ExperimentalExecutionState.h"
#include "FrameRecording.h"


// Private compiler representation. Keep compiler-only data here rather than in
// RenderGraph's installed public header so compile algorithm changes have a
// narrow incremental-build footprint.
namespace org {

struct RenderGraph::CompilerState {
    experimental::CompileWorkspace synchronousCompileWorkspace;
    std::map<uint32_t, std::shared_ptr<PreparedInvocationArena>> invocationArenas;
    std::shared_ptr<runtime::ITaskScope> ownershipRetirementScope;
    std::shared_ptr<runtime::ITaskScope> persistentBuildScope; // worker-side structural builds
    std::shared_ptr<runtime::ITaskScope> frameWorkerScope;
    bool frameProductionStopped = false;
    uint64_t asyncPreparationFrameNumber = 0;
    std::optional<uint32_t> lastExecutedPreparationSlot;
    std::shared_ptr<const IHostExecutionData> lastSubmittedFrameData;
    // Owned by RenderGraphPersistent.cpp; shared_ptr keeps the type opaque here.
    std::shared_ptr<PersistentExecutionState> persistent;
    std::vector<ExternalPassDesc> frameExtensions;
};


} // namespace org
