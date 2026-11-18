#pragma once

// Shared by the persistent-layer tests: a binding for a test identity, and two independent checks of a compiled
// schedule (hazard order and state continuity) that call no compiler helpers.

#include "Render/RenderGraph/PersistentGraph.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace org_test {
using namespace org::persistent;
using namespace org::compiler;

inline BindingVersion Binding(uint64_t id) {
    BindingVersion b;
    b.identity = id; b.backingRevision = id; b.shape = {1,1,false};
    b.owner = std::make_shared<const uint64_t>(id);
    return b;
}
// Independent small reference check: each conflicting authored access must
// precede its later user in execution order. No compiler helpers are called.
inline bool CheckConflicts(const Schedule& graph) {
    if (!graph.structure) return false;
    std::vector<uint32_t> position(graph.structure->passes.size(), UINT32_MAX);
    uint32_t n = 0;
    for (const auto& batch : graph.batches)
        for (auto pass : batch.passes) {
            if (pass >= position.size() || position[pass] != UINT32_MAX) return false;
            position[pass] = n++;
        }
    if (n != position.size()) return false;
    const auto count = position.size();
    std::vector<uint8_t> happensBefore(count * count);
    std::vector<uint8_t> authored(count * count);
    for (const auto* edges : {&graph.structure->explicitEdges,&graph.structure->placementEdges})
        for (auto [from,to] : *edges) {
            if (from >= count || to >= count) return false;
            authored[from * count + to] = 1;
        }
    for (size_t k = 0; k < count; ++k)
        for (size_t i = 0; i < count; ++i)
            for (size_t j = 0; j < count; ++j)
                if (authored[i * count + k] && authored[k * count + j]) authored[i * count + j] = 1;
    std::vector<uint32_t> lastOnQueue(graph.structure->queues.size(),UINT32_MAX);
    for (const auto& batch : graph.batches) {
        if (batch.queue >= lastOnQueue.size()) return false;
        const auto& queue = graph.structure->queues[batch.queue];
        if (!queue.active) return false;
        auto& previous = lastOnQueue[batch.queue];
        for (auto pass : batch.passes) {
            const auto& declaration = graph.structure->passes[pass];
            // Pass backend affinity and queue backend-instance identity are
            // different domains; authored compatibility is the numeric contract.
            if (std::find(declaration.compatibleQueueSlots.begin(),declaration.compatibleQueueSlots.end(),batch.queue)
                    == declaration.compatibleQueueSlots.end()) return false;
            if (previous != UINT32_MAX) happensBefore[previous * count + pass] = 1;
            previous = pass;
        }
    }
    for (const auto& wait : graph.relativeWaits) {
        if (wait.producerBatch >= graph.batches.size() || wait.consumerBatch >= graph.batches.size()) return false;
        const auto& producer = graph.batches[wait.producerBatch];
        const auto& consumer = graph.batches[wait.consumerBatch];
        if (producer.passes.empty() || consumer.passes.empty()) return false;
        happensBefore[producer.passes.back() * count + consumer.passes.front()] = 1;
    }
    for (size_t k = 0; k < count; ++k)
        for (size_t i = 0; i < count; ++i)
            for (size_t j = 0; j < count; ++j)
                if (happensBefore[i * count + k] && happensBefore[k * count + j]) happensBefore[i * count + j] = 1;
    for (auto [from,to] : graph.structure->explicitEdges)
        if (from >= count || to >= count || !happensBefore[from * count + to]) return false;
    for (auto [from,to] : graph.structure->placementEdges)
        if (from >= count || to >= count || !happensBefore[from * count + to]) return false;
    for (uint32_t a = 0; a < position.size(); ++a)
        for (uint32_t b = a + 1; b < position.size(); ++b)
            for (const auto& x : graph.structure->passes[a].accesses)
                for (const auto& y : graph.structure->passes[b].accesses)
                    if (x.resourceIndex == y.resourceIndex && (x.write || y.write)
                        && !happensBefore[a * count + b]
                        && !(authored[b * count + a] && happensBefore[b * count + a])) return false;
    return true;
}
// Test-only full-cell simulation, deliberately independent of production range
// partitioning, dense placement maps and symbolic-state validation helpers.
inline bool CheckStates(const Schedule& graph) {
    if (!graph.structure || !graph.states.complete) return false;
    const auto& structure = *graph.structure;
    if (structure.resourceShapes.size() != structure.resourceIDs.size()) return false;
    struct Cell {
        CompileResourceState state;
        uint32_t pass = UINT32_MAX, batch = UINT32_MAX;
        bool terminal = false;
    };
    std::vector<std::vector<Cell>> cells;
    for (const auto shape : structure.resourceShapes) {
        if (uint64_t{shape.mips} * shape.slices > 2'000'000) return false;
        cells.emplace_back(size_t{shape.mips} * shape.slices);
    }
    auto visit = [&](uint32_t resource, CompileRange range, auto&& fn) {
        if (resource >= cells.size()) return false;
        const auto shape = structure.resourceShapes[resource];
        if (!range.mips || !range.slices || range.mip >= shape.mips || range.slice >= shape.slices
            || range.mips > shape.mips - range.mip || range.slices > shape.slices - range.slice) return false;
        for (uint32_t slice = range.slice; slice < range.slice + range.slices; ++slice)
            for (uint32_t mip = range.mip; mip < range.mip + range.mips; ++mip)
                if (!fn(cells[resource][size_t{slice} * shape.mips + mip])) return false;
        return true;
    };
    auto normalized = [&](uint32_t resource, CompileResourceState state) {
        if (!structure.resourceShapes[resource].hasLayout) state.layout = 0;
        return state;
    };
    std::vector<bool> consumed(graph.states.steps.size());
    for (uint32_t batch = 0; batch < graph.batches.size(); ++batch) {
        for (const auto passIndex : graph.batches[batch].passes) {
            if (passIndex >= structure.passes.size()) return false;
            for (size_t i = 0; i < graph.states.steps.size(); ++i) {
                const auto& step = graph.states.steps[i];
                if (step.pass != passIndex || step.batch != batch) continue;
                if (!visit(step.resource, step.range, [&](Cell& cell) {
                    if (cell.pass != step.previousPass || cell.batch != step.previousBatch
                        || cell.state != step.before || cell.pass == passIndex) return false;
                    cell = {step.after,passIndex,batch,false};
                    return true;
                })) return false;
                consumed[i] = true;
            }
            const auto& pass = structure.passes[passIndex];
            for (const auto& use : pass.entryStates) {
                if (!visit(use.resource,use.range,[&](Cell& cell) {
                    return cell.pass == passIndex && cell.batch == batch
                        && cell.state == normalized(use.resource,use.state);
                })) return false;
            }
            for (const auto& use : pass.exitStates) {
                if (!visit(use.resource,use.range,[&](Cell& cell) {
                    if (cell.pass != passIndex || cell.batch != batch) return false;
                    cell.state = normalized(use.resource,use.state);
                    return true;
                })) return false;
            }
        }
    }
    if (std::find(consumed.begin(),consumed.end(),false) != consumed.end()) return false;
    for (const auto& final : graph.states.finalStates) {
        if (!visit(final.resource,final.range,[&](Cell& cell) {
            if (cell.terminal || cell.batch != final.batch || cell.state != final.state) return false;
            cell.terminal = true;
            return true;
        })) return false;
    }
    for (const auto& resource : cells)
        for (const auto& cell : resource)
            if ((cell.pass != UINT32_MAX) != cell.terminal) return false;
    return true;
}
} // namespace org_test

using org_test::Binding;
using org_test::CheckConflicts;
using org_test::CheckStates;
