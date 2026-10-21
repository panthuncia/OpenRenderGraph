// Schedule introspection: a persistent executable's batches, passes, planned state changes and cross-queue waits, with each wait's
// cause. The compiler keeps no provenance for its waits, so causes are reconstructed here from the same inputs and rules it orders
// passes by (ExperimentalGraphCompiler.cpp: AnalyzeDependencies, ForEachScheduledUse, OrderUse).
#include "ScheduleTraceBuild.h"
#include "Render/RenderGraph/PersistentGraph.h"

#include <rhi_helpers.h>

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string_view>
#include <utility>

namespace org {

namespace {

using experimental::CompilePass;
using experimental::GraphCompileStructure;
using Cause = ScheduleTrace::WaitCause;
using ScheduledUses = std::vector<std::pair<uint32_t, bool>>;

// A pass's scheduled uses, once per resource and sorted by it: its accesses, entry and exit states, a write when any of them
// writes (ForEachScheduledUse).
void CollectScheduledUses(const CompilePass& pass, ScheduledUses& out) {
    out.clear();
    out.reserve(pass.accesses.size() + pass.entryStates.size() + pass.exitStates.size());
    for (const auto& access : pass.accesses) out.emplace_back(access.resourceIndex, access.write);
    for (const auto& use : pass.entryStates) out.emplace_back(use.resource, use.state.write);
    for (const auto& use : pass.exitStates) out.emplace_back(use.resource, use.state.write);
    std::sort(out.begin(), out.end());
    size_t count = 0;
    for (size_t index = 0; index < out.size(); ++index) {
        const auto use = out[index];
        if (count && out[count - 1].first == use.first) out[count - 1].second = out[count - 1].second || use.second;
        else out[count++] = use;
    }
    out.resize(count);
}

// The hazards that order `to` after `from` on the resources both use. A write orders against every use (RAW, WAR, WAW). Two reads
// are ordered only for a texture, whose uses OrderUse serializes (its layout is planned state), or across backends, which
// AppendDependencyAccess orders for API ownership; two reads of a buffer on one backend are not ordered at all.
void AppendResourceCauses(const GraphCompileStructure& structure, uint32_t from, uint32_t to, const ScheduledUses& fromUses,
    const ScheduledUses& toUses, std::vector<Cause>& causes) {
    const bool crossBackend = structure.passes[from].backend != structure.passes[to].backend;
    auto left = fromUses.begin();
    auto right = toUses.begin();
    while (left != fromUses.end() && right != toUses.end()) {
        if (left->first < right->first) { ++left; continue; }
        if (right->first < left->first) { ++right; continue; }
        const auto resource = left->first;
        const bool fromWrites = left->second, toWrites = right->second;
        ++left; ++right;
        auto hazard = Cause::Hazard::None;
        if (fromWrites && toWrites) hazard = Cause::Hazard::WriteAfterWrite;
        else if (fromWrites) hazard = Cause::Hazard::ReadAfterWrite;
        else if (toWrites) hazard = Cause::Hazard::WriteAfterRead;
        else {
            const bool overlappingReads = resource < structure.resourceShapes.size() && !structure.resourceShapes[resource].hasLayout;
            if (overlappingReads && !crossBackend) continue;
        }
        causes.push_back({Cause::Kind::Resource, from, to, resource, hazard});
    }
}

const char* KindName(Cause::Kind kind) {
    switch (kind) {
    case Cause::Kind::Resource: return "resource";
    case Cause::Kind::Explicit: return "explicit";
    case Cause::Kind::Placement: return "placement";
    case Cause::Kind::Unknown: break;
    }
    return "unknown";
}

const char* HazardName(Cause::Hazard hazard) {
    switch (hazard) {
    case Cause::Hazard::ReadAfterWrite: return "RAW";
    case Cause::Hazard::WriteAfterRead: return "WAR";
    case Cause::Hazard::WriteAfterWrite: return "WAW";
    case Cause::Hazard::None: break;
    }
    return "none";
}

const char* SourceName(ScheduleSubmissionWaits::Point::Source source) {
    switch (source) {
    case ScheduleSubmissionWaits::Point::Source::Incoming: return "incoming";
    case ScheduleSubmissionWaits::Point::Source::Uploads: return "uploads";
    case ScheduleSubmissionWaits::Point::Source::FrameValues: return "frame-values";
    }
    return "unknown";
}

std::string AccessName(uint64_t access) {
    return rhi::helpers::ResourceAccessMaskToString(static_cast<rhi::ResourceAccessType>(access));
}
std::string LayoutName(uint64_t layout) {
    return rhi::helpers::ResourceLayoutToString(static_cast<rhi::ResourceLayout>(layout));
}
std::string SyncName(uint64_t sync) {
    return rhi::helpers::ResourceSyncToString(static_cast<rhi::ResourceSyncState>(sync));
}

// Minimal JSON writing: ORG has no JSON dependency.
void AppendEscaped(std::string& out, std::string_view text) {
    out += '"';
    for (const char character : text) {
        switch (character) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(character) < 0x20) {
                char escaped[8];
                std::snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(character)));
                out += escaped;
            } else out += character;
        }
    }
    out += '"';
}

class JsonWriter {
public:
    std::string out;
    void BeginObject() { Separate(); out += '{'; m_first = true; }
    void EndObject() { out += '}'; m_first = false; }
    void BeginArray() { Separate(); out += '['; m_first = true; }
    void EndArray() { out += ']'; m_first = false; }
    void Key(std::string_view key) { Separate(); AppendEscaped(out, key); out += ':'; m_first = true; }
    void WriteString(std::string_view value) { Separate(); AppendEscaped(out, value); }
    void WriteNumber(uint64_t value) { Separate(); out += std::to_string(value); }
    void WriteBool(bool value) { Separate(); out += value ? "true" : "false"; }
    void WriteNull() { Separate(); out += "null"; }
    // UINT32_MAX (no batch, no pass, the whole frame) as null.
    void WriteIndex(uint32_t value) { if (value == UINT32_MAX) WriteNull(); else WriteNumber(value); }
private:
    void Separate() {
        if (!m_first) out += ',';
        m_first = false;
    }
    bool m_first = true;
};

void WriteState(JsonWriter& json, uint64_t access, uint64_t layout, uint64_t sync, bool write, bool hasLayout) {
    json.BeginObject();
    json.Key("access"); json.WriteNumber(access);
    json.Key("accessName"); json.WriteString(AccessName(access));
    json.Key("layout"); json.WriteNumber(layout);
    json.Key("layoutName"); json.WriteString(hasLayout ? LayoutName(layout) : std::string{});
    json.Key("sync"); json.WriteNumber(sync);
    json.Key("syncName"); json.WriteString(SyncName(sync));
    json.Key("write"); json.WriteBool(write);
    json.EndObject();
}

void WriteTrace(JsonWriter& json, const ScheduleTrace& trace) {
    json.BeginObject();
    json.Key("label"); json.WriteString(trace.label);
    json.Key("structuralRevision"); json.WriteNumber(trace.structuralRevision);
    json.Key("epoch"); json.WriteIndex(trace.epoch);
    json.Key("stateFallbackReason"); json.WriteString(trace.stateFallbackReason);
    json.Key("queueNames"); json.BeginArray();
    for (const auto& name : trace.queueNames) json.WriteString(name);
    json.EndArray();
    json.Key("batches"); json.BeginArray();
    for (const auto& batch : trace.batches) {
        json.BeginObject();
        json.Key("queue"); json.WriteNumber(batch.queue);
        json.Key("passes"); json.BeginArray();
        for (const auto pass : batch.passes) json.WriteNumber(pass);
        json.EndArray();
        json.EndObject();
    }
    json.EndArray();
    json.Key("passes"); json.BeginArray();
    for (const auto& pass : trace.passes) {
        json.BeginObject();
        json.Key("name"); json.WriteString(pass.name);
        json.Key("preparedPassIndex"); json.WriteIndex(pass.preparedPassIndex);
        json.Key("batch"); json.WriteIndex(pass.batch);
        json.Key("position"); json.WriteIndex(pass.position);
        json.Key("queue"); json.WriteNumber(pass.queue);
        json.Key("preferredQueue"); json.WriteNumber(pass.preferredQueue);
        json.Key("forceBatchIsolation"); json.WriteBool(pass.forceBatchIsolation);
        json.Key("accesses"); json.BeginArray();
        for (const auto& access : pass.accesses) {
            json.BeginObject();
            json.Key("resource"); json.WriteNumber(access.resource);
            json.Key("write"); json.WriteBool(access.write);
            json.EndObject();
        }
        json.EndArray();
        json.Key("stateChanges"); json.BeginArray();
        for (const auto& change : pass.stateChanges) {
            const bool hasLayout = change.resource < trace.resources.size() && trace.resources[change.resource].hasLayout;
            json.BeginObject();
            json.Key("resource"); json.WriteNumber(change.resource);
            json.Key("previousBatch"); json.WriteIndex(change.previousBatch);
            json.Key("previousPass"); json.WriteIndex(change.previousPass);
            json.Key("before"); WriteState(json, change.beforeAccess, change.beforeLayout, change.beforeSync, change.beforeWrite, hasLayout);
            json.Key("after"); WriteState(json, change.afterAccess, change.afterLayout, change.afterSync, change.afterWrite, hasLayout);
            json.Key("firstMip"); json.WriteNumber(change.firstMip);
            json.Key("mipCount"); json.WriteNumber(change.mipCount);
            json.Key("firstSlice"); json.WriteNumber(change.firstSlice);
            json.Key("sliceCount"); json.WriteNumber(change.sliceCount);
            json.EndObject();
        }
        json.EndArray();
        json.EndObject();
    }
    json.EndArray();
    json.Key("resources"); json.BeginArray();
    for (const auto& resource : trace.resources) {
        json.BeginObject();
        json.Key("name"); json.WriteString(resource.name);
        json.Key("id"); json.WriteNumber(resource.id);
        json.Key("hasLayout"); json.WriteBool(resource.hasLayout);
        json.EndObject();
    }
    json.EndArray();
    json.Key("waits"); json.BeginArray();
    for (const auto& wait : trace.waits) {
        json.BeginObject();
        json.Key("consumerBatch"); json.WriteNumber(wait.consumerBatch);
        json.Key("producerBatch"); json.WriteNumber(wait.producerBatch);
        json.Key("causes"); json.BeginArray();
        for (const auto& cause : wait.causes) {
            json.BeginObject();
            json.Key("kind"); json.WriteString(KindName(cause.kind));
            json.Key("fromPass"); json.WriteNumber(cause.fromPass);
            json.Key("toPass"); json.WriteNumber(cause.toPass);
            json.Key("resource"); json.WriteIndex(cause.resource);
            json.Key("hazard"); json.WriteString(HazardName(cause.hazard));
            json.EndObject();
        }
        json.EndArray();
        json.EndObject();
    }
    json.EndArray();
    json.EndObject();
}

}  // namespace

std::shared_ptr<const ScheduleTrace> BuildScheduleTrace(const persistent::ExecutableGeneration& executable, const ScheduleTraceNames& names,
    std::string label, uint32_t epoch) {
    auto trace = std::make_shared<ScheduleTrace>();
    trace->label = std::move(label);
    trace->structuralRevision = executable.structuralRevision;
    trace->epoch = epoch;
    trace->queueNames = names.queues;
    if (!executable.graph || !executable.graph->structure) return trace;
    const auto& graph = *executable.graph;
    const auto& structure = *graph.structure;
    const auto passCount = static_cast<uint32_t>(structure.passes.size());
    const auto batchCount = static_cast<uint32_t>(graph.batches.size());

    trace->resources.resize(structure.resourceIDs.size());
    for (uint32_t resource = 0; resource < trace->resources.size(); ++resource) {
        auto& traced = trace->resources[resource];
        traced.id = structure.resourceIDs[resource];
        traced.hasLayout = resource < structure.resourceShapes.size() && structure.resourceShapes[resource].hasLayout;
        if (names.resource && resource < executable.resourceSlots.size()) traced.name = names.resource(executable.resourceSlots[resource].index);
        if (traced.name.empty()) traced.name = "<resource " + std::to_string(resource) + ">";
    }

    trace->passes.resize(passCount);
    for (uint32_t pass = 0; pass < passCount; ++pass) {
        const auto& declaration = structure.passes[pass];
        auto& traced = trace->passes[pass];
        traced.preparedPassIndex = declaration.preparedPassIndex;
        if (names.pass) traced.name = names.pass(declaration.preparedPassIndex);
        if (traced.name.empty()) traced.name = "<pass " + std::to_string(pass) + ">";
        traced.batch = pass < graph.batchByPass.size() ? graph.batchByPass[pass] : UINT32_MAX;
        traced.position = pass < graph.positionByPass.size() ? graph.positionByPass[pass] : UINT32_MAX;
        traced.queue = traced.batch < batchCount ? graph.batches[traced.batch].queue : 0;
        traced.preferredQueue = declaration.preferredQueueSlot;
        traced.forceBatchIsolation = declaration.forceBatchIsolation;
        traced.accesses.reserve(declaration.accesses.size());
        for (const auto& access : declaration.accesses) traced.accesses.push_back({access.resourceIndex, access.write});
    }

    trace->batches.resize(batchCount);
    for (uint32_t batch = 0; batch < batchCount; ++batch) {
        trace->batches[batch].queue = graph.batches[batch].queue;
        trace->batches[batch].passes = graph.batches[batch].passes;
    }

    for (const auto& step : graph.states.steps) {
        if (step.pass >= passCount) continue;
        ScheduleTrace::StateChange change;
        change.resource = step.resource;
        change.previousBatch = step.previousBatch;
        change.previousPass = step.previousPass;
        change.beforeAccess = step.before.access;
        change.beforeLayout = step.before.layout;
        change.beforeSync = step.before.sync;
        change.beforeWrite = step.before.write;
        change.afterAccess = step.after.access;
        change.afterLayout = step.after.layout;
        change.afterSync = step.after.sync;
        change.afterWrite = step.after.write;
        change.firstMip = step.range.mip;
        change.mipCount = step.range.mips;
        change.firstSlice = step.range.slice;
        change.sliceCount = step.range.slices;
        trace->passes[step.pass].stateChanges.push_back(change);
    }
    if (!graph.states.complete) trace->stateFallbackReason = graph.states.fallbackReason;

    // A relative wait {consumer, producer} is the latest producer batch on its queue that any scheduling edge into the consumer
    // comes from (Compile's latestProducer): it covers every edge into the consumer from that queue's batches up to it.
    std::vector<std::vector<uint32_t>> edgesByConsumer(batchCount);
    for (uint32_t index = 0; index < graph.schedulingEdges.size(); ++index) {
        const auto to = graph.schedulingEdges[index].second;
        if (to < graph.batchByPass.size() && graph.batchByPass[to] < batchCount) edgesByConsumer[graph.batchByPass[to]].push_back(index);
    }
    auto explicitEdges = structure.explicitEdges;
    auto placementEdges = structure.placementEdges;
    std::sort(explicitEdges.begin(), explicitEdges.end());
    std::sort(placementEdges.begin(), placementEdges.end());
    std::vector<ScheduledUses> uses(passCount);
    std::vector<uint8_t> usesCollected(passCount, 0);
    auto usesOf = [&](uint32_t pass) -> const ScheduledUses& {
        if (!usesCollected[pass]) {
            CollectScheduledUses(structure.passes[pass], uses[pass]);
            usesCollected[pass] = 1;
        }
        return uses[pass];
    };
    trace->waits.reserve(graph.relativeWaits.size());
    for (const auto& relative : graph.relativeWaits) {
        ScheduleTrace::Wait wait;
        wait.consumerBatch = relative.consumerBatch;
        wait.producerBatch = relative.producerBatch;
        if (relative.consumerBatch < batchCount && relative.producerBatch < batchCount) {
            const auto producerQueue = graph.batches[relative.producerBatch].queue;
            for (const auto index : edgesByConsumer[relative.consumerBatch]) {
                const auto edge = graph.schedulingEdges[index];
                const auto from = edge.first, to = edge.second;
                if (from >= passCount || to >= passCount) continue;
                const auto source = graph.batchByPass[from];
                if (source >= batchCount || source > relative.producerBatch || graph.batches[source].queue != producerQueue) continue;
                if (std::binary_search(explicitEdges.begin(), explicitEdges.end(), edge)) {
                    wait.causes.push_back({Cause::Kind::Explicit, from, to});
                    continue;
                }
                if (std::binary_search(placementEdges.begin(), placementEdges.end(), edge)) {
                    wait.causes.push_back({Cause::Kind::Placement, from, to});
                    continue;
                }
                const auto before = wait.causes.size();
                AppendResourceCauses(structure, from, to, usesOf(from), usesOf(to), wait.causes);
                if (wait.causes.size() == before) wait.causes.push_back({Cause::Kind::Unknown, from, to});
            }
        }
        trace->waits.push_back(std::move(wait));
    }
    return trace;
}

std::string ScheduleTraceToJson(const PersistentScheduleSnapshot& snapshot) {
    JsonWriter json;
    json.BeginObject();
    json.Key("executables"); json.BeginArray();
    for (const auto& trace : snapshot.executables) {
        if (trace) WriteTrace(json, *trace);
        else json.WriteNull();
    }
    json.EndArray();
    json.Key("lastSubmission"); json.BeginArray();
    for (const auto& submission : snapshot.lastSubmission) {
        json.BeginObject();
        json.Key("label"); json.WriteString(submission.label);
        json.Key("frame"); json.WriteNumber(submission.frame);
        json.Key("batches"); json.BeginArray();
        for (const auto& batch : submission.batches) {
            json.BeginObject();
            json.Key("externalEntryBarrier"); json.WriteBool(batch.externalEntryBarrier);
            json.Key("externalExitBarrier"); json.WriteBool(batch.externalExitBarrier);
            auto writePoints = [&](const std::vector<ScheduleSubmissionWaits::Point>& points) {
                json.BeginArray();
                for (const auto& point : points) {
                    json.BeginObject();
                    json.Key("source"); json.WriteString(SourceName(point.source));
                    json.Key("timeline"); json.WriteNumber(point.timeline);
                    json.Key("value"); json.WriteNumber(point.value);
                    json.EndObject();
                }
                json.EndArray();
            };
            json.Key("points"); writePoints(batch.points);
            if (batch.queueUploadCopies) {
                json.Key("queueUploadCopies"); json.WriteNumber(batch.queueUploadCopies);
                json.Key("queueUploadWaits"); writePoints(batch.queueUploadWaits);
            }
            json.EndObject();
        }
        json.EndArray();
        json.EndObject();
    }
    json.EndArray();
    json.EndObject();
    return std::move(json.out);
}

std::string ScheduleTraceToText(const ScheduleTrace& trace) {
    auto passName = [&](uint32_t pass) -> std::string {
        if (pass == UINT32_MAX) return "<entry>";
        return pass < trace.passes.size() ? "[" + std::to_string(pass) + "] \"" + trace.passes[pass].name + "\"" : "<pass " + std::to_string(pass) + ">";
    };
    auto resourceName = [&](uint32_t resource) -> std::string {
        return resource < trace.resources.size() ? "\"" + trace.resources[resource].name + "\"" : "<resource " + std::to_string(resource) + ">";
    };
    auto queueName = [&](uint32_t queue) -> std::string {
        return queue < trace.queueNames.size() ? trace.queueNames[queue] : "queue " + std::to_string(queue);
    };
    std::vector<std::vector<uint32_t>> waitsByConsumer(trace.batches.size());
    for (uint32_t index = 0; index < trace.waits.size(); ++index)
        if (trace.waits[index].consumerBatch < waitsByConsumer.size()) waitsByConsumer[trace.waits[index].consumerBatch].push_back(index);

    std::ostringstream out;
    out << "[Schedule \"" << trace.label << "\"] structural_revision=" << trace.structuralRevision;
    if (trace.epoch != UINT32_MAX) out << " epoch=" << trace.epoch;
    out << " passes=" << trace.passes.size() << " batches=" << trace.batches.size() << " waits=" << trace.waits.size() << "\n";
    if (!trace.stateFallbackReason.empty()) out << "state plan incomplete: " << trace.stateFallbackReason << "\n";
    for (uint32_t batch = 0; batch < trace.batches.size(); ++batch) {
        out << "batch " << batch << " queue=" << queueName(trace.batches[batch].queue) << "\n";
        for (const auto index : waitsByConsumer[batch]) {
            const auto& wait = trace.waits[index];
            const auto producerQueue = wait.producerBatch < trace.batches.size() ? trace.batches[wait.producerBatch].queue : UINT32_MAX;
            out << "  waits for batch " << wait.producerBatch << " (" << queueName(producerQueue) << "):\n";
            for (const auto& cause : wait.causes) {
                out << "    " << passName(cause.fromPass) << " -> " << passName(cause.toPass) << ": " << KindName(cause.kind);
                if (cause.kind == Cause::Kind::Resource)
                    out << " " << HazardName(cause.hazard) << " on " << resourceName(cause.resource);
                out << "\n";
            }
        }
        for (const auto pass : trace.batches[batch].passes) {
            if (pass >= trace.passes.size()) continue;
            const auto& traced = trace.passes[pass];
            out << "  " << passName(pass) << " position=" << traced.position << " accesses=" << traced.accesses.size();
            if (traced.preferredQueue != traced.queue) out << " preferred=" << queueName(traced.preferredQueue);
            if (traced.forceBatchIsolation) out << " isolated";
            out << "\n";
            for (const auto& change : traced.stateChanges) {
                const bool hasLayout = change.resource < trace.resources.size() && trace.resources[change.resource].hasLayout;
                out << "    state " << resourceName(change.resource) << " mips=[" << change.firstMip << "+" << change.mipCount << "] slices=["
                    << change.firstSlice << "+" << change.sliceCount << "] after ";
                if (change.previousPass == UINT32_MAX) out << "<execution entry>";
                else out << passName(change.previousPass) << " (batch " << change.previousBatch << ")";
                out << ": " << AccessName(change.beforeAccess);
                if (hasLayout) out << " " << LayoutName(change.beforeLayout);
                out << " " << SyncName(change.beforeSync) << (change.beforeWrite ? " write" : "") << " -> " << AccessName(change.afterAccess);
                if (hasLayout) out << " " << LayoutName(change.afterLayout);
                out << " " << SyncName(change.afterSync) << (change.afterWrite ? " write" : "") << "\n";
            }
        }
    }
    return out.str();
}

}  // namespace org
