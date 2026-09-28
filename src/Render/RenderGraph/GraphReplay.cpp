#include "Render/RenderGraph/GraphReplay.h"
#include <istream>
#include <ostream>
#include <stdexcept>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <mutex>

namespace org::experimental {
namespace {
constexpr size_t Limit = 2000000;
GraphCompileStructure ReadSnapshot(std::istream&, bool requireEnd);
template<class T> void Read(std::istream& input, T& value) {
    if (!(input >> value)) throw std::invalid_argument("Truncated or invalid graph replay");
}
size_t Count(std::istream& input) {
    uint64_t count; Read(input, count);
    if (count > Limit) throw std::invalid_argument("Graph replay count exceeds limit");
    return static_cast<size_t>(count);
}
void WriteUses(std::ostream& out, const std::vector<CompileStateUse>& uses) {
    out << uses.size() << '\n';
    for (const auto& use : uses)
        out << use.resource << ' ' << use.range.mip << ' ' << use.range.mips << ' '
            << use.range.slice << ' ' << use.range.slices << ' ' << use.state.access << ' '
            << use.state.layout << ' ' << use.state.sync << ' ' << use.state.write << '\n';
}
void ReadUses(std::istream& in, std::vector<CompileStateUse>& uses) {
    uses.resize(Count(in));
    for (auto& use : uses) {
        Read(in,use.resource); Read(in,use.range.mip); Read(in,use.range.mips);
        Read(in,use.range.slice); Read(in,use.range.slices); Read(in,use.state.access);
        Read(in,use.state.layout); Read(in,use.state.sync); Read(in,use.state.write);
    }
}
}
void WriteGraphReplay(std::ostream& out, const GraphCompileStructure& s) {
    if (s.resourceIDs.size() != s.resourceShapes.size()) throw std::invalid_argument("Replay requires resource shapes");
    out << "ORG_GRAPH_REPLAY 2\n" << s.resourceIDs.size() << '\n';
    for (size_t i = 0; i < s.resourceIDs.size(); ++i)
        out << s.resourceIDs[i] << ' ' << s.resourceShapes[i].mips << ' ' << s.resourceShapes[i].slices
            << ' ' << s.resourceShapes[i].hasLayout << '\n';
    out << s.resourceKeys.size() << '\n';
    for (const auto& key : s.resourceKeys) out << std::quoted(key) << '\n';
    out << s.queues.size() << '\n';
    for (const auto& queue : s.queues) out << queue.backendInstance << ' ' << queue.active << '\n';
    out << s.passes.size() << '\n';
    for (const auto& pass : s.passes) {
        out << pass.originalOrder << ' ' << pass.backend << ' ' << pass.preferredQueueSlot << ' '
            << pass.preparedPassIndex << ' ' << pass.forceBatchIsolation << '\n';
        out << pass.compatibleQueueSlots.size();
        for (auto queue : pass.compatibleQueueSlots) out << ' ' << queue;
        out << '\n' << pass.accesses.size() << '\n';
        for (const auto& access : pass.accesses) out << access.resourceIndex << ' ' << access.write << '\n';
        WriteUses(out,pass.entryStates); WriteUses(out,pass.exitStates);
    }
    for (const auto* edges : {&s.explicitEdges,&s.placementEdges}) {
        out << edges->size() << '\n';
        for (auto [from,to] : *edges) out << from << ' ' << to << '\n';
    }
    if (!out) throw std::runtime_error("Graph replay write failed");
}
void WriteGraphReplaySequence(std::ostream& out, const GraphReplaySequence& sequence) {
    out << "ORG_GRAPH_SEQUENCE 3\n";
    WriteGraphReplay(out,sequence.initial);
    out << sequence.bindingEdits.size() << '\n';
    for (const auto& edit : sequence.bindingEdits)
        out << edit.frame << ' ' << edit.publicationRevision << ' ' << edit.allocationIdentity << ' '
            << edit.backingRevision << ' ' << edit.descriptorRevision << ' ' << edit.contentRevision << ' '
            << edit.slot << ' ' << edit.slotGeneration << ' ' << edit.shape.mips << ' ' << edit.shape.slices << ' '
            << edit.shape.hasLayout << '\n';
    out << sequence.submissions.size() << '\n';
    for (const auto& event : sequence.submissions) {
        out << event.frame << ' ' << event.signaledBatches << '\n';
        for (const auto* points : {&event.batchSignals,&event.tailCompletions}) {
            out << points->size() << '\n';
            for (auto point : *points) out << point.timeline << ' ' << point.value << '\n';
        }
        out << event.producerWaits.size() << '\n';
        for (const auto& wait : event.producerWaits)
            out << wait.pass << ' ' << wait.generation << ' ' << wait.completion.timeline << ' ' << wait.completion.value << '\n';
        out << event.incomingStates.size() << '\n';
        for (const auto& state : event.incomingStates) {
            out << state.slot << ' ' << state.generation << ' ' << state.allocationIdentity << ' ' << state.revision << ' '
                << state.completion.timeline << ' ' << state.completion.value << ' ' << state.regions.size() << '\n';
            for (const auto& region : state.regions)
                out << region.range.mip << ' ' << region.range.mips << ' ' << region.range.slice << ' ' << region.range.slices << ' '
                    << region.state.access << ' ' << region.state.layout << ' ' << region.state.sync << ' ' << region.state.write << '\n';
        }
    }
    out << sequence.completions.size() << '\n';
    for (const auto& event : sequence.completions) out << event.frame << ' ' << event.timeline << ' ' << event.value << '\n';
    if (!out) throw std::runtime_error("Graph sequence write failed");
}
GraphReplaySequence ReadGraphReplaySequence(std::istream& in) {
    std::string magic; uint32_t version;
    Read(in,magic); Read(in,version);
    if (magic != "ORG_GRAPH_SEQUENCE" || version < 1 || version > 3) throw std::invalid_argument("Unsupported graph sequence");
    GraphReplaySequence sequence; sequence.initial = ReadSnapshot(in,false);
    sequence.bindingEdits.resize(Count(in));
    for (auto& edit : sequence.bindingEdits) {
        Read(in,edit.frame); Read(in,edit.publicationRevision); Read(in,edit.allocationIdentity);
        Read(in,edit.backingRevision); Read(in,edit.descriptorRevision); Read(in,edit.contentRevision);
        Read(in,edit.slot); Read(in,edit.slotGeneration); Read(in,edit.shape.mips); Read(in,edit.shape.slices); Read(in,edit.shape.hasLayout);
        if (!edit.publicationRevision || !edit.allocationIdentity || !edit.backingRevision || !edit.slotGeneration
            || edit.slot >= sequence.initial.resourceIDs.size()) throw std::invalid_argument("Invalid sequence binding edit");
    }
    if (version >= 2) {
        sequence.submissions.resize(Count(in));
        for (auto& event : sequence.submissions) {
            Read(in,event.frame); Read(in,event.signaledBatches);
            for (auto* points : {&event.batchSignals,&event.tailCompletions}) {
                points->resize(Count(in));
                for (auto& point : *points) { Read(in,point.timeline); Read(in,point.value); }
            }
            const auto count = event.signaledBatches == UINT32_MAX ? event.batchSignals.size() : event.signaledBatches;
            if (count > event.batchSignals.size()) throw std::invalid_argument("Invalid replay submitted prefix");
            for (size_t i = 0; i < count; ++i)
                if (!event.batchSignals[i].timeline || !event.batchSignals[i].value) throw std::invalid_argument("Invalid replay submission signal");
            for (auto point : event.tailCompletions)
                if (!point.timeline || !point.value) throw std::invalid_argument("Invalid replay submission tail");
            if (version >= 3) {
                event.producerWaits.resize(Count(in));
                for (auto& wait : event.producerWaits) {
                    Read(in,wait.pass); Read(in,wait.generation); Read(in,wait.completion.timeline); Read(in,wait.completion.value);
                    if (wait.pass >= sequence.initial.passes.size() || !wait.generation || !wait.completion.timeline || !wait.completion.value)
                        throw std::invalid_argument("Invalid replay producer wait");
                }
                event.incomingStates.resize(Count(in));
                for (auto& state : event.incomingStates) {
                    Read(in,state.slot); Read(in,state.generation); Read(in,state.allocationIdentity); Read(in,state.revision);
                    Read(in,state.completion.timeline); Read(in,state.completion.value);
                    if (state.slot >= sequence.initial.resourceIDs.size() || !state.generation || !state.allocationIdentity || !state.revision
                        || !state.completion.timeline || !state.completion.value) throw std::invalid_argument("Invalid replay incoming state");
                    state.regions.resize(Count(in));
                    for (auto& region : state.regions) {
                        Read(in,region.range.mip); Read(in,region.range.mips); Read(in,region.range.slice); Read(in,region.range.slices);
                        Read(in,region.state.access); Read(in,region.state.layout); Read(in,region.state.sync); Read(in,region.state.write);
                    }
                }
            }
        }
    }
    sequence.completions.resize(Count(in));
    for (auto& event : sequence.completions) {
        Read(in,event.frame); Read(in,event.timeline); Read(in,event.value);
        if (!event.timeline) throw std::invalid_argument("Invalid sequence completion timeline");
    }
    in >> std::ws;
    if (!in.eof()) throw std::invalid_argument("Trailing graph sequence data");
    return sequence;
}
namespace {
GraphCompileStructure ReadSnapshot(std::istream& in, bool requireEnd) {
    std::string magic; uint32_t version;
    Read(in,magic); Read(in,version);
    if (magic != "ORG_GRAPH_REPLAY" || (version != 1 && version != 2)) throw std::invalid_argument("Unsupported graph replay version");
    GraphCompileStructure s;
    s.resourceIDs.resize(Count(in)); s.resourceShapes.resize(s.resourceIDs.size());
    for (size_t i = 0; i < s.resourceIDs.size(); ++i) {
        Read(in,s.resourceIDs[i]); Read(in,s.resourceShapes[i].mips);
        Read(in,s.resourceShapes[i].slices); Read(in,s.resourceShapes[i].hasLayout);
    }
    if (version == 2) {
        s.resourceKeys.resize(Count(in));
        if (!s.resourceKeys.empty() && s.resourceKeys.size() != s.resourceIDs.size())
            throw std::invalid_argument("Invalid replay semantic key count");
        for (auto& key : s.resourceKeys)
            if (!(in >> std::quoted(key))) throw std::invalid_argument("Invalid replay semantic key");
    }
    s.queues.resize(Count(in));
    for (auto& queue : s.queues) { Read(in,queue.backendInstance); Read(in,queue.active); }
    s.passes.resize(Count(in));
    for (auto& pass : s.passes) {
        Read(in,pass.originalOrder); Read(in,pass.backend); Read(in,pass.preferredQueueSlot);
        Read(in,pass.preparedPassIndex); Read(in,pass.forceBatchIsolation);
        pass.compatibleQueueSlots.resize(Count(in));
        for (auto& queue : pass.compatibleQueueSlots) Read(in,queue);
        pass.accesses.resize(Count(in));
        for (auto& access : pass.accesses) { Read(in,access.resourceIndex); Read(in,access.write); }
        ReadUses(in,pass.entryStates); ReadUses(in,pass.exitStates);
    }
    for (auto* edges : {&s.explicitEdges,&s.placementEdges}) {
        edges->resize(Count(in));
        for (auto& [from,to] : *edges) { Read(in,from); Read(in,to); }
    }
    if (requireEnd) {
        in >> std::ws;
        if (!in.eof()) throw std::invalid_argument("Trailing graph replay data");
    }
    return s;
}
}
GraphCompileStructure ReadGraphReplay(std::istream& in) { return ReadSnapshot(in,true); }
void ObserveGraphReplay(std::shared_ptr<const GraphCompileStructure> structure) {
    struct Recorder {
        std::string path;
        std::mutex mutex;
        std::shared_ptr<const GraphCompileStructure> latest;
        Recorder() {
            if (const auto* value = std::getenv("ORG_GRAPH_REPLAY_OUTPUT")) path = value;
        }
        ~Recorder() {
            if (path.empty() || !latest) return;
            try {
                std::ofstream output(path);
                WriteGraphReplay(output,*latest);
            } catch (const std::exception& error) {
                std::fprintf(stderr,"Graph replay export failed: %s\n",error.what());
            }
        }
    };
    static Recorder recorder;
    if (recorder.path.empty()) return;
    std::lock_guard lock(recorder.mutex);
    // CompiledGraph::structure may alias GraphCompileInput's control block.
    // Copy numeric metadata to avoid retaining frame payloads and semantic pins.
    recorder.latest = std::make_shared<const GraphCompileStructure>(*structure);
}
}
