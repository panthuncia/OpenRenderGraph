#pragma once
#include "Render/RenderGraph/GraphReplay.h"
#include "Render/RenderGraph/PersistentGraph.h"
#include <algorithm>
#include <chrono>
#include <map>
#include <stdexcept>

namespace org::test {
struct ReplayRunResult {
    size_t submissions = 0, bindingChanges = 0, retiredFrames = 0, peakRetainedFrames = 0;
    size_t producerWaits = 0, incomingReports = 0, incomingRevisions = 0;
    size_t submissionWaits = 0;
    size_t retiredBackings = 0, peakStateBackings = 0, finalStateBackings = 0;
    uint64_t digest = 0;
    double bootstrapMs = 0, publicationMs = 0, criticalPathMs = 0;
    std::vector<double> admissionMs;
};
// GPU-free fixture adapter. Only synthetic physical identities are supplied by
// this layer; graph edits, compilation, selection and admission are production.
// Sequence v3 includes external waits/states, but not recording workloads or authored physical aliasing.
inline ReplayRunResult RunGraphReplaySequence(const experimental::GraphReplaySequence& sequence) {
    using namespace experimental;
    using namespace persistent;
    using Clock = std::chrono::steady_clock;
    const auto milliseconds = [](auto since) {
        return std::chrono::duration<double,std::milli>(Clock::now()-since).count();
    };
    const auto require = [](bool valid, const char* message) {
        if (!valid) throw std::invalid_argument(message);
    };
    const auto ordered = [](const auto& events) {
        return std::is_sorted(events.begin(),events.end(),[](const auto& a,const auto& b) { return a.frame < b.frame; });
    };
    require(ordered(sequence.bindingEdits) && ordered(sequence.submissions) && ordered(sequence.completions),
        "Replay events must be ordered within each stream");
    require(sequence.initial.resourceIDs.size() == sequence.initial.resourceShapes.size(),"Replay shapes are missing");
    ReplayRunResult result;
    GraphProgram program; CompileWorkspace workspace; std::atomic_bool cancelled{false};
    std::map<uint64_t,uint32_t> allocations;
    uint32_t nextAllocation = 1;
    const auto bind = [&](uint64_t allocation, CompileResourceShape shape, uint32_t slot) {
        auto [where,inserted] = allocations.try_emplace(allocation,nextAllocation);
        if (inserted) {
            require(nextAllocation != UINT32_MAX,"Replay allocation identities exhausted");
            ++nextAllocation;
        }
        BindingVersion binding;
        binding.identity = (uint64_t{1} << 32) | where->second;
        binding.backingRevision = 1; binding.shape = shape;
        binding.owner = std::make_shared<const uint64_t>(allocation);
        auto state = std::make_shared<PreparedBackingState>();
        state->resource = {where->second,1}; state->graphResourceID = slot+1; state->shape = shape;
        state->regions = std::make_shared<const std::vector<PreparedStateRegion>>();
        binding.admission = std::move(state);
        return binding;
    };
    const auto bootstrapStart = Clock::now();
    {
        auto edit = program.BeginEdit(); edit.SetQueues(sequence.initial.queues);
        for (uint32_t i = 0; i < sequence.initial.resourceIDs.size(); ++i)
            edit.AddResource(sequence.initial.resourceShapes[i],bind(uint64_t{i}+1,sequence.initial.resourceShapes[i],i));
        for (auto pass : sequence.initial.passes) edit.AddPass(std::move(pass));
        for (auto [from,to] : sequence.initial.explicitEdges) edit.AddOrdering({from,1},{to,1});
        for (auto [from,to] : sequence.initial.placementEdges) edit.AddPlacementOrdering({from,1},{to,1});
        require(program.Install(edit.Build(workspace,cancelled)),"Replay bootstrap was rejected");
    }
    result.bootstrapMs = milliseconds(bootstrapStart);
    const auto selectedGraph = program.Select()->executable->graph;
    std::vector<ExecutionTimelinePoint> queues(sequence.initial.queues.size());
    // Queue identities are recovered from receipts, never from completion order.
    for (const auto& submission : sequence.submissions) {
        require(submission.batchSignals.size() == selectedGraph->batches.size(),"Replay receipt batch count mismatch");
        for (size_t i = 0; i < submission.batchSignals.size(); ++i) {
            const auto timeline = submission.batchSignals[i].timeline;
            if (!timeline) continue;
            auto& queue = queues.at(selectedGraph->batches[i].queue);
            require(!queue.timeline || queue.timeline == timeline,"Replay queue timeline changed without an edit");
            queue.timeline = timeline;
        }
    }
    uint64_t unusedTimeline = 1;
    for (auto& queue : queues) if (!queue.timeline) {
        while (std::ranges::any_of(queues,[&](auto point) { return point.timeline == unusedTimeline; })) ++unusedTimeline;
        queue.timeline = unusedTimeline++;
    }
    SynchronousAdmission admission;
    std::vector<SynchronousAdmission::RetirementTicket> retiredBindings;
    const auto prune = [&] {
        std::erase_if(retiredBindings,[&](const auto& ticket) {
            if (!admission.RetireBackingMetadata(ticket)) return false;
            ++result.retiredBackings; return true;
        });
    };
    size_t editIndex = 0, submissionIndex = 0, completionIndex = 0;
    const auto criticalStart = Clock::now();
    while (editIndex < sequence.bindingEdits.size() || submissionIndex < sequence.submissions.size()
        || completionIndex < sequence.completions.size()) {
        uint64_t frame = UINT64_MAX;
        if (editIndex < sequence.bindingEdits.size()) frame = (std::min)(frame,sequence.bindingEdits[editIndex].frame);
        if (submissionIndex < sequence.submissions.size()) frame = (std::min)(frame,sequence.submissions[submissionIndex].frame);
        if (completionIndex < sequence.completions.size()) frame = (std::min)(frame,sequence.completions[completionIndex].frame);
        // Earlier submissions may complete while this frame's publication is
        // installed. Same-frame completion events precede new submissions.
        while (completionIndex < sequence.completions.size() && sequence.completions[completionIndex].frame == frame) {
            const auto& event = sequence.completions[completionIndex++];
            const std::array points{ExecutionTimelinePoint{event.timeline,event.value}};
            result.retiredFrames += admission.RetireCompleted(points);
            prune();
        }
        while (editIndex < sequence.bindingEdits.size() && sequence.bindingEdits[editIndex].frame == frame) {
            const auto start = Clock::now();
            const auto revision = sequence.bindingEdits[editIndex].publicationRevision;
            auto edit = program.BeginEdit();
            do {
                const auto& event = sequence.bindingEdits[editIndex++];
                auto binding = bind(event.allocationIdentity,event.shape,event.slot);
                const auto& previous = program.Select()->bindings.At({event.slot,event.slotGeneration});
                if (previous.identity != binding.identity)
                    retiredBindings.push_back(SynchronousAdmission::CaptureRetirement(previous));
                binding.backingRevision = event.backingRevision;
                binding.descriptorRevision = event.descriptorRevision; binding.contentRevision = event.contentRevision;
                edit.ReplaceBinding({event.slot,event.slotGeneration},std::move(binding));
                ++result.bindingChanges;
            } while (editIndex < sequence.bindingEdits.size() && sequence.bindingEdits[editIndex].frame == frame
                && sequence.bindingEdits[editIndex].publicationRevision == revision);
            auto ready = edit.Build(workspace,cancelled);
            require(ready->revision == revision,"Replay publication revision mismatch");
            require(ready->executable == program.Select()->executable,"Compatible replay binding rebuilt the executable");
            require(program.Install(ready),"Replay publication was rejected");
            result.publicationMs += milliseconds(start);
        }
        prune();
        while (submissionIndex < sequence.submissions.size() && sequence.submissions[submissionIndex].frame == frame) {
            const auto& event = sequence.submissions[submissionIndex++];
            const auto start = Clock::now();
            std::vector<FrameProducerWait> waits;
            for (const auto& wait : event.producerWaits) waits.push_back({{wait.pass,wait.generation},wait.completion});
            std::vector<FrameIncomingState> incoming;
            for (const auto& state : event.incomingStates) {
                const auto allocation = allocations.find(state.allocationIdentity);
                require(allocation != allocations.end(),"Replay incoming allocation is unknown");
                incoming.push_back({{state.slot,state.generation},{allocation->second,1},
                    std::make_shared<const std::vector<PreparedStateRegion>>(state.regions),state.completion,state.revision});
            }
            auto prepared = admission.Prepare(program.Select(),queues,waits,incoming);
            result.producerWaits += waits.size(); result.incomingReports += incoming.size();
            result.incomingRevisions += prepared.incomingEffects.size();
            const auto count = event.signaledBatches == UINT32_MAX ? event.batchSignals.size() : event.signaledBatches;
            require(count <= event.batchSignals.size(),"Replay submitted prefix exceeds batch count");
            // Plan every batch through the production timeline owner. A receipt
            // records only submitted signals; reserve synthetic values for the
            // unsubmitted suffix so relative waits can still be instantiated.
            auto reserved = event.batchSignals;
            auto plannedQueues = queues;
            for (size_t i = 0; i < reserved.size(); ++i) {
                auto& queue = plannedQueues.at(selectedGraph->batches[i].queue);
                if (i >= count) {
                    require(!reserved[i].value,"Replay unsubmitted batch has a signal");
                    require(queue.value < UINT64_MAX-1,"Replay queue timeline exhausted");
                    reserved[i] = {queue.timeline,queue.value+1};
                }
                queue.value = reserved[i].value;
            }
            // A partial failure closes a real timeline owner. Replay constructs
            // a fresh numeric owner from actual submitted values for each event;
            // this models recovery without retaining any native submission work.
            ExecutionTimelineAdmission timeline(queues);
            const auto planned = timeline.Prepare(prepared.publication->executable->executionLayout->bundle,
                prepared.incomingWaits,{}, {},reserved);
            GraphExecutionTimeline receipt = *planned;
            for (size_t i = 0; i < count; ++i) {
                timeline.CommitBatch(planned->submission,static_cast<uint32_t>(i));
                result.submissionWaits += receipt.batches[i].waits.size();
            }
            for (size_t i = count; i < receipt.batches.size(); ++i) receipt.batches[i].signal = event.batchSignals[i];
            if (count < receipt.batches.size()) timeline.Fail(planned->submission);
            receipt.tailCompletions = event.tailCompletions;
            admission.Commit(prepared,receipt,event.signaledBatches);
            for (size_t i = 0; i < count; ++i) queues.at(selectedGraph->batches[i].queue).value = receipt.batches[i].signal.value;
            result.admissionMs.push_back(milliseconds(start));
            for (const auto& backing : prepared.backings)
                result.digest = result.digest*1099511628211ull + backing.resource.index;
            ++result.submissions;
            result.peakStateBackings = (std::max)(result.peakStateBackings,admission.StateBackingCount());
            result.peakRetainedFrames = (std::max)(result.peakRetainedFrames,admission.RetainedFrames());
        }
    }
    result.criticalPathMs = milliseconds(criticalStart);
    require(admission.RetainedFrames() == 0,"Replay ended with incomplete GPU consumers");
    prune();
    require(retiredBindings.empty(),"Replay ended with retired backing owners");
    result.finalStateBackings = admission.StateBackingCount();
    return result;
}
}
