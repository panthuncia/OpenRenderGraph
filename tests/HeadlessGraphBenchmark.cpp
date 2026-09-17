#include "Render/RenderGraph/PersistentGraph.h"
#include "Render/RenderGraph/GraphReplay.h"
#include "GraphReplayRunner.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <fstream>
#include <string>

using namespace org::persistent;
using namespace org::experimental;
void SetAdmission(BindingVersion& binding, uint32_t index, uint32_t generation) {
    auto state = std::make_shared<PreparedBackingState>();
    state->graphResourceID = index + 1;
    state->resource = {index + 1,generation};
    state->shape = binding.shape;
    auto regions = std::make_shared<std::vector<PreparedStateRegion>>();
    if (binding.shape.mips && binding.shape.slices)
        regions->push_back({{0,binding.shape.mips,0,binding.shape.slices},
            {static_cast<uint64_t>(rhi::ResourceAccessType::Common),0,
                static_cast<uint64_t>(rhi::ResourceSyncState::All),false}});
    state->regions = std::move(regions);
    binding.identity = (uint64_t{generation} << 32) | (index + 1);
    binding.admission = std::move(state);
}
int Run(int argc, char** argv) {
    std::string replayPath, exportPath, sequencePath;
    bool streaming = true;
    bool grouped = false;
    bool checkTarget = false, passed = true;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 < argc && std::string(argv[i]) == "--replay") replayPath = argv[++i];
        else if (i + 1 < argc && std::string(argv[i]) == "--sequence") sequencePath = argv[++i];
        else if (i + 1 < argc && std::string(argv[i]) == "--write-replay") exportPath = argv[++i];
        else if (std::string(argv[i]) == "--static") streaming = false;
        else if (std::string(argv[i]) == "--groups") grouped = true;
        else if (std::string(argv[i]) == "--check-target") checkTarget = true;
        else { std::fprintf(stderr,"Usage: [--replay FILE] [--write-replay FILE] [--static] [--groups] [--check-target] | --sequence FILE\n"); return 2; }
    }
    if (!sequencePath.empty()) {
        if (!replayPath.empty() || !exportPath.empty() || !streaming || grouped || checkTarget)
            throw std::invalid_argument("Sequence replay must be run without synthetic benchmark options");
        std::ifstream input(sequencePath);
        const auto result = org::test::RunGraphReplaySequence(ReadGraphReplaySequence(input));
        auto samples = result.admissionMs; std::sort(samples.begin(),samples.end());
        const auto percentile = [&](double fraction) {
            return samples.empty() ? 0.0 : samples[static_cast<size_t>(std::ceil(samples.size()*fraction))-1];
        };
        const auto mean = samples.empty() ? 0.0 : std::accumulate(samples.begin(),samples.end(),0.0)/samples.size();
        std::printf("{\"scope\":\"sequence_numeric_bindings_and_admission\",\"submissions\":%zu,\"binding_changes\":%zu,\"producer_waits\":%zu,\"incoming_reports\":%zu,\"incoming_revisions\":%zu,\"retired_frames\":%zu,\"peak_retained_frames\":%zu,\"bootstrap_ms\":%.6f,\"publication_ms\":%.6f,\"critical_path_ms\":%.6f,\"admission_mean_ms\":%.6f,\"admission_p50_ms\":%.6f,\"admission_p95_ms\":%.6f,\"admission_p99_ms\":%.6f,\"admission_max_ms\":%.6f,\"digest\":%llu}\n",
            result.submissions,result.bindingChanges,result.producerWaits,result.incomingReports,result.incomingRevisions,result.retiredFrames,result.peakRetainedFrames,
            result.bootstrapMs,result.publicationMs,result.criticalPathMs,mean,percentile(.5),percentile(.95),percentile(.99),
            samples.empty() ? 0.0 : samples.back(),static_cast<unsigned long long>(result.digest));
        return 0;
    }
    if (grouped && !replayPath.empty()) throw std::invalid_argument("Group workload uses synthetic declarations; omit --replay");
    GraphCompileStructure replay;
    if (!replayPath.empty()) {
        std::ifstream input(replayPath);
        replay = ReadGraphReplay(input);
    }
    GraphProgram program; CompileWorkspace workspace; std::atomic_bool cancelled{false};
    const auto resourceCount = replayPath.empty() ? 650u : static_cast<uint32_t>(replay.resourceIDs.size());
    if (!resourceCount) return 2;
    const auto passCount = replayPath.empty() ? 176u : static_cast<uint32_t>(replay.passes.size());
    std::vector<ResourceGroupId> groups;
    std::vector<BindingToken> recordingBindings;
    {
        auto initial = program.BeginEdit();
        if (!replayPath.empty()) initial.SetQueues(replay.queues);
        for (uint32_t i = 0; i < resourceCount; ++i) {
            BindingVersion binding; binding.identity = i + 1; binding.backingRevision = 1;
            binding.shape = replayPath.empty() ? CompileResourceShape{1,1,false} : replay.resourceShapes[i];
            binding.owner = std::make_shared<const uint32_t>(i);
            SetAdmission(binding, i, 1);
            initial.AddResource(binding.shape, std::move(binding));
        }
        if (grouped) {
            for (uint32_t start = 0; start < resourceCount; start += 32) {
                std::vector<ResourceSlotId> members;
                for (uint32_t i = start; i < (std::min)(start + 32,resourceCount); ++i) members.push_back({i,1});
                groups.push_back(initial.AddGroup({1,1,false},std::move(members)));
            }
        }
        for (uint32_t i = 0; i < passCount; ++i) {
            if (!replayPath.empty()) { initial.AddPass(replay.passes[i]); continue; }
            CompilePass pass;
            if (grouped) {
                const auto id = initial.AddPass(std::move(pass));
                const bool write = i < groups.size();
                initial.DeclareGroupAccess(id,groups[i % groups.size()],
                    {static_cast<uint64_t>(write ? rhi::ResourceAccessType::UnorderedAccess : rhi::ResourceAccessType::ShaderResource),
                        0,static_cast<uint64_t>(rhi::ResourceSyncState::ComputeShading),write},i);
                continue;
            }
            const auto id = initial.AddPass(std::move(pass));
            for (uint32_t k = 0; k < 22; ++k) {
                const uint32_t resource = (i * 17 + k) % 650;
                const bool write = k == 0;
                recordingBindings.push_back(initial.Declare(id,{resource,1},{},
                    {static_cast<uint64_t>(write ? rhi::ResourceAccessType::UnorderedAccess
                    : rhi::ResourceAccessType::ShaderResource),0,static_cast<uint64_t>(rhi::ResourceSyncState::ComputeShading),write}));
            }
        }
        for (auto [from,to] : replay.explicitEdges) initial.AddOrdering({from,1},{to,1});
        for (auto [from,to] : replay.placementEdges) initial.AddPlacementOrdering({from,1},{to,1});
        if (!program.Install(initial, initial.Build(workspace, cancelled))) return 1;
    } // Bootstrap edit must not retain every initial backing for the replay.
    if (!exportPath.empty()) {
        std::ofstream output(exportPath);
        WriteGraphReplay(output,*program.Select()->executable->graph->structure);
    }
    for (int run = 0; run < 5; ++run) {
        SynchronousAdmission admission;
        std::vector<ExecutionTimelinePoint> queues(program.Select()->executable->graph->structure->queues.size());
        for (uint32_t i = 0; i < queues.size(); ++i) queues[i] = {i+1,0};
        std::vector<double> samples; samples.reserve(10000);
        uint64_t digest = 0;
        uint64_t generationBuilds = 0, bindingChanges = 0;
        double publicationBuildMs = 0;
        std::vector<SynchronousAdmission::RetirementTicket> retirements;
        size_t peakStateBackings = 0, peakHazardBackings = 0, peakRetirements = 0;
        uint64_t retiredBindings = 0;
        for (uint32_t frame = 0; frame < 11000; ++frame) {
            const auto begin = std::chrono::steady_clock::now();
            // Include maintenance in the timed scope rather than hiding its
            // owner-thread cost. The previous iteration's CPU roots are gone.
            std::erase_if(retirements,[&](const auto& ticket) {
                if (!admission.RetireBackingMetadata(ticket)) return false;
                if (!admission.RetireAliasMetadata(ticket)) return false;
                ++retiredBindings;
                return true;
            });
            if (streaming && frame % 7 == 0) {
                auto edit = program.BeginEdit();
                auto binding = program.Select()->bindings.At({frame % resourceCount,1});
                retirements.push_back(SynchronousAdmission::CaptureRetirement(binding));
                ++binding.backingRevision;
                SetAdmission(binding, frame % resourceCount, static_cast<uint32_t>(binding.backingRevision));
                binding.owner = std::make_shared<const uint64_t>(binding.identity);
                edit.ReplaceBinding({frame % resourceCount,1}, std::move(binding));
                if (!program.Install(edit, edit.Build(workspace, cancelled))) return 2;
                if (frame >= 1000) ++bindingChanges;
            }
            if (grouped && streaming && frame % 127 == 0) {
                auto edit = program.BeginEdit();
                const auto group = groups[(frame / 127) % groups.size()];
                const uint32_t start = group.index * 32;
                const uint32_t full = (std::min)(32u,resourceCount-start);
                const auto selected = program.Select();
                const uint32_t size = selected->logical->groups[group.index].members.size() == full ? full / 2 : full;
                std::vector<ResourceSlotId> members;
                for (uint32_t i = 0; i < size; ++i) members.push_back({start+i,1});
                edit.ReplaceGroupMembers(group,std::move(members));
                const auto buildBegin = std::chrono::steady_clock::now();
                auto ready = edit.Build(workspace,cancelled);
                const auto buildEnd = std::chrono::steady_clock::now();
                if (!program.Install(edit,std::move(ready))) return 2;
                if (frame >= 1000) {
                    ++generationBuilds;
                    publicationBuildMs += std::chrono::duration<double,std::milli>(buildEnd-buildBegin).count();
                }
            }
            const auto selected = program.Select();
            auto prepared = admission.Prepare(selected, queues);
            for (uint32_t i = 0; i < resourceCount; ++i) digest += selected->bindings.At({i,1}).identity;
            for (auto token : recordingBindings) digest += selected->Resolve(token).identity;
            const auto end = std::chrono::steady_clock::now();
            if (frame >= 1000) samples.push_back(std::chrono::duration<double,std::milli>(end-begin).count());
            GraphExecutionTimeline receipt;
            receipt.batches.resize(selected->executable->graph->batches.size());
            for (uint32_t i = 0; i < receipt.batches.size(); ++i) {
                auto& queue = queues[selected->executable->graph->batches[i].queue];
                ++queue.value;
                receipt.batches[i].signal = queue;
            }
            admission.Commit(prepared, receipt);
            admission.RetireCompleted(queues);
            peakStateBackings = (std::max)(peakStateBackings,admission.StateBackingCount());
            peakHazardBackings = (std::max)(peakHazardBackings,admission.HazardBackingCount());
            peakRetirements = (std::max)(peakRetirements,retirements.size());
        }
        std::erase_if(retirements,[&](const auto& ticket) {
            if (!admission.RetireBackingMetadata(ticket) || !admission.RetireAliasMetadata(ticket)) return false;
            ++retiredBindings;
            return true;
        });
        if (!retirements.empty() || admission.StateBackingCount() > resourceCount
            || admission.HazardBackingCount() > resourceCount)
            throw std::runtime_error("Streaming retirement failed to bound admission metadata");
        const auto mean = std::accumulate(samples.begin(),samples.end(),0.0)/samples.size();
        std::sort(samples.begin(),samples.end());
        passed &= mean <= 1.0 && samples[9899] <= 2.0;
        std::printf("{\"run\":%d,\"scope\":\"%s\",\"workload\":\"%s\",\"frames\":10000,\"mean_ms\":%.6f,\"p50_ms\":%.6f,\"p95_ms\":%.6f,\"p99_ms\":%.6f,\"max_ms\":%.6f,\"digest\":%llu,\"binding_changes\":%llu,\"generation_builds\":%llu,\"publication_build_ms\":%.6f,\"retired_bindings_including_warmup\":%llu,\"peak_state_backings\":%zu,\"peak_hazard_backings\":%zu,\"peak_retirement_tickets\":%zu}\n",
            run,grouped ? "synthetic_groups_bindings_and_admission" :
                (recordingBindings.empty() ? "synthetic_bindings_and_admission" : "synthetic_typed_bindings_and_admission"),
            streaming ? "streaming" : "static",mean,samples[4999],samples[9499],samples[9899],samples.back(),static_cast<unsigned long long>(digest),
            static_cast<unsigned long long>(bindingChanges),static_cast<unsigned long long>(generationBuilds),publicationBuildMs,
            static_cast<unsigned long long>(retiredBindings),peakStateBackings,peakHazardBackings,peakRetirements);
    }
    return checkTarget && !passed ? 3 : 0;
}
int main(int argc, char** argv) {
    try { return Run(argc,argv); }
    catch (const std::exception& error) { std::fprintf(stderr,"%s\n",error.what()); return 1; }
}
