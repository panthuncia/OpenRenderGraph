#pragma once
#include "Render/RenderGraph/ExperimentalGraphCompiler.h"
#include "Render/RenderGraph/ExperimentalExecutionState.h"
#include <iosfwd>
#include <memory>

namespace org::experimental {
// Versioned numeric graph snapshots. Never serialize leases or native handles.
void WriteGraphReplay(std::ostream& output, const GraphCompileStructure& structure);
GraphCompileStructure ReadGraphReplay(std::istream& input);
struct ReplayBindingEdit {
    uint64_t frame = 0, publicationRevision = 0, allocationIdentity = 0;
    uint64_t backingRevision = 0, descriptorRevision = 0, contentRevision = 0;
    uint32_t slot = 0, slotGeneration = 0;
    CompileResourceShape shape;
};
struct ReplayCompletion {
    uint64_t frame = 0, timeline = 0, value = 0;
};
struct ReplaySubmission {
    uint64_t frame = 0;
    std::vector<ExecutionTimelinePoint> batchSignals, tailCompletions;
    uint32_t signaledBatches = UINT32_MAX;
    struct ProducerWait { uint32_t pass = 0, generation = 1; ExecutionTimelinePoint completion; };
    struct IncomingState {
        uint32_t slot = 0, generation = 1;
        uint64_t allocationIdentity = 0, revision = 1;
        ExecutionTimelinePoint completion;
        std::vector<PreparedStateRegion> regions;
    };
    std::vector<ProducerWait> producerWaits;
    std::vector<IncomingState> incomingStates;
};
// Allocation identities are fixture-local numbers, not encoded RHI handles.
// The sequence schema covers binding churn, external states/waits and completion; further
// structural/program/invocation event records remain separate migration work.
struct GraphReplaySequence {
    GraphCompileStructure initial;
    std::vector<ReplayBindingEdit> bindingEdits;
    std::vector<ReplaySubmission> submissions;
    std::vector<ReplayCompletion> completions;
};
void WriteGraphReplaySequence(std::ostream&, const GraphReplaySequence&);
GraphReplaySequence ReadGraphReplaySequence(std::istream&);
// Opt-in ORG_GRAPH_REPLAY_OUTPUT captures the latest numeric structure and
// writes it at process shutdown, outside the frame's measured compile interval.
void ObserveGraphReplay(std::shared_ptr<const GraphCompileStructure> structure);
}
