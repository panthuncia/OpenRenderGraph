#include "Render/RenderGraph/ExperimentalGraphCompiler.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <BasicTelemetry/Tracy.h>

namespace org::experimental {

namespace {
template<class Fn>
void ForEachScheduledResource(const CompilePass& pass, Fn&& fn) {
    for (auto access : pass.accesses) fn(access.resourceIndex);
    // The legacy dependency DAG deliberately omits globally read-only
    // resources. They still require state/queue ordering in a compiled plan.
    for (const auto& use : pass.entryStates) fn(use.resource);
    for (const auto& use : pass.exitStates) fn(use.resource);
}
bool ValidRange(CompileRange r, CompileResourceShape shape) {
    return r.mips && r.slices && r.mip < shape.mips && r.slice < shape.slices
        && r.mips <= shape.mips - r.mip && r.slices <= shape.slices - r.slice;
}
CompileResourceState StateForShape(CompileResourceState state, CompileResourceShape shape) {
    // Buffer barriers have no layout field. Preserve all access/sync bits.
    if (!shape.hasLayout) state.layout = 0;
    return state;
}
CompileRange Intersection(CompileRange a, CompileRange b) {
    const auto m = (std::max)(a.mip, b.mip), s = (std::max)(a.slice, b.slice);
    const auto me = (std::min)(a.mip + a.mips, b.mip + b.mips);
    const auto se = (std::min)(a.slice + a.slices, b.slice + b.slices);
    return {m, me > m ? me - m : 0, s, se > s ? se - s : 0};
}
struct StateRegion {
    CompileRange range;
    CompileResourceState state;
    uint32_t batch = UINT32_MAX;
};

SymbolicStatePlan BuildStatePlan(const GraphCompileStructure& s,
    std::span<const SymbolicBatch> batches, const std::atomic_bool& cancelled) {
    BT_ZONE_SCOPE("ORG.AsyncCompile.SymbolicStates");
    SymbolicStatePlan plan;
    auto fallback = [&](const char* reason) {
        BT_ZONE_SCOPE("ORG.AsyncCompile.StateFallback");
        plan.steps.clear(); plan.finalStates.clear(); plan.fallbackReason = reason;
        BT_ZONE_TEXT(plan.fallbackReason.data(), plan.fallbackReason.size());
        // A bounded set of literal reasons: diagnostic attempts include jobs
        // later cancelled/coalesced, unlike the coordinator acceptance counts.
        basic_telemetry::AddCounter(std::string("ORG.AsyncCompile.StateFallback.") + reason);
        return plan;
    };
    if (s.resourceShapes.size() != s.resourceIDs.size()) return fallback("Resource shapes not captured");
    uint64_t cells = 0;
    for (auto shape : s.resourceShapes) {
        if (!shape.mips && !shape.slices && !shape.hasLayout) continue;
        if (!shape.mips || !shape.slices) return fallback("Invalid resource dimensions");
        cells += uint64_t{shape.mips} * shape.slices;
        // Bound the independent full-cell validation oracle during migration.
        if (cells > 2'000'000) return fallback("State oracle subresource budget exceeded");
    }
    uint64_t oracleWork = 0;
    std::vector<bool> written(s.resourceIDs.size());
    for (const auto& pass : s.passes) {
        for (auto access : pass.accesses) if (access.resourceIndex < written.size())
            written[access.resourceIndex] = written[access.resourceIndex] || access.write;
        for (auto* uses : {&pass.entryStates, &pass.exitStates}) for (const auto& use : *uses) {
            if (use.resource >= s.resourceShapes.size() || !ValidRange(use.range, s.resourceShapes[use.resource]))
                return fallback("Invalid captured state range");
            oracleWork += uint64_t{use.range.mips} * use.range.slices;
            if (oracleWork > 2'000'000) return fallback("State oracle work budget exceeded");
        }
    }
    std::vector<std::vector<StateRegion>> regions(s.resourceShapes.size());
    for (size_t r = 0; r < regions.size(); ++r) if (s.resourceShapes[r].mips)
        regions[r].push_back({{0, s.resourceShapes[r].mips, 0, s.resourceShapes[r].slices}, {}});
    std::vector<StateRegion> next;
    for (uint32_t b = 0; b < batches.size(); ++b) {
        if (cancelled.load(std::memory_order_relaxed)) return fallback("Cancelled");
        const auto& pass = s.passes[batches[b].pass];
        auto apply = [&](const CompileStateUse& use, bool exit) -> const char* {
            if (use.resource >= regions.size() || !ValidRange(use.range, s.resourceShapes[use.resource]))
                return "Invalid captured state range";
            if ((written[use.resource] || use.state.write) && std::none_of(pass.accesses.begin(), pass.accesses.end(),
                [&](auto a) { return a.resourceIndex == use.resource && (!use.state.write || a.write); }))
                return "State use missing required hazard access";
            const auto state = StateForShape(use.state, s.resourceShapes[use.resource]);
            next.clear();
            for (const auto& previous : regions[use.resource]) {
                const auto overlap = Intersection(previous.range, use.range);
                if (!overlap.mips || !overlap.slices) { next.push_back(previous); continue; }
                if (exit && previous.batch != b) return "Internal transition lacks matching entry declaration";
                if (!exit && previous.batch == b && previous.state != state)
                    return "Conflicting overlapping entry states";
                if (!exit && previous.batch != b)
                    plan.steps.push_back({use.resource, b, previous.batch, overlap, previous.state, state});
                const auto old = previous.range;
                auto keep = [&](CompileRange range) {
                    if (range.mips && range.slices) next.push_back({range, previous.state, previous.batch});
                };
                keep({old.mip, overlap.mip - old.mip, old.slice, old.slices});
                keep({overlap.mip + overlap.mips, old.mip + old.mips - overlap.mip - overlap.mips, old.slice, old.slices});
                keep({overlap.mip, overlap.mips, old.slice, overlap.slice - old.slice});
                keep({overlap.mip, overlap.mips, overlap.slice + overlap.slices, old.slice + old.slices - overlap.slice - overlap.slices});
                next.push_back({overlap, state, b});
            }
            regions[use.resource].swap(next);
            return nullptr;
        };
        for (const auto& use : pass.entryStates) if (auto error = apply(use, false)) return fallback(error);
        for (const auto& use : pass.exitStates) if (auto error = apply(use, true)) return fallback(error);
    }
    for (uint32_t r = 0; r < regions.size(); ++r) for (const auto& region : regions[r])
        if (region.batch != UINT32_MAX) plan.finalStates.push_back({r, region.batch, region.range, region.state});
    plan.complete = true;
    return plan;
}
} // namespace

void NormalizeCompileInput(GraphCompileInput& input) {
    auto& s = input.structure;
    if (s.resourceIDs.size() >= UINT32_MAX || s.passes.size() >= UINT32_MAX)
        throw std::invalid_argument("Compile input exceeds index capacity");
    std::vector<uint32_t> order(s.resourceIDs.size()), remap(order.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](auto a, auto b) { return s.resourceIDs[a] < s.resourceIDs[b]; });
    std::vector<uint64_t> ids;
    std::vector<CompileResourceShape> shapes;
    if (!s.resourceShapes.empty() && s.resourceShapes.size() != order.size())
        throw std::invalid_argument("Captured resource shape count mismatch");
    ids.reserve(order.size());
    for (uint32_t i = 0; i < order.size(); ++i) {
        const auto id = s.resourceIDs[order[i]];
        if (!ids.empty() && ids.back() == id)
            throw std::invalid_argument("Duplicate captured global resource ID");
        ids.push_back(id); remap[order[i]] = i;
        if (!s.resourceShapes.empty()) shapes.push_back(s.resourceShapes[order[i]]);
    }
    s.resourceIDs = std::move(ids);
    s.resourceShapes = std::move(shapes);
    for (auto& pass : s.passes) {
        for (auto* uses : {&pass.entryStates, &pass.exitStates}) for (auto& use : *uses) {
            if (use.resource >= remap.size()) throw std::invalid_argument("Invalid captured state resource index");
            use.resource = remap[use.resource];
            if (!s.resourceShapes.empty()) use.state = StateForShape(use.state, s.resourceShapes[use.resource]);
        }
        for (auto& access : pass.accesses) {
            if (access.resourceIndex >= remap.size()) throw std::invalid_argument("Invalid captured resource index");
            access.resourceIndex = remap[access.resourceIndex];
        }
        std::sort(pass.accesses.begin(), pass.accesses.end(), [](auto a, auto b) { return a.resourceIndex < b.resourceIndex; });
        size_t used = 0;
        for (auto access : pass.accesses) {
            if (used && pass.accesses[used - 1].resourceIndex == access.resourceIndex)
                pass.accesses[used - 1].write |= access.write;
            else pass.accesses[used++] = access;
        }
        pass.accesses.resize(used);
    }
    auto normalizeEdges = [](DependencyEdges& edges) {
        std::erase_if(edges, [](auto edge) { return edge.first == edge.second; });
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    };
    normalizeEdges(s.explicitEdges);
    normalizeEdges(s.placementEdges);
}

std::string ValidateSymbolicSchedule(const GraphCompileInput& input, const CompiledGraph& graph) {
    BT_ZONE_SCOPE("ORG.AsyncCompile.ValidateSchedule");
    const auto& s = input.structure;
    const size_t count = s.passes.size(), words = (count + 63) / 64;
    if (graph.stage != CompiledGraphStage::SymbolicSchedule || graph.batches.size() != count)
        return "Missing complete symbolic schedule";
    std::vector<uint32_t> batchByPass(count, UINT32_MAX), lastQueue(s.queues.size(), UINT32_MAX);
    std::vector<std::vector<uint32_t>> waits(count);
    for (auto wait : graph.relativeWaits) {
        if (wait.consumerBatch >= count || wait.producerBatch >= wait.consumerBatch)
            return "Relative wait targets an invalid or future batch";
        waits[wait.consumerBatch].push_back(wait.producerBatch);
    }
    std::vector<std::vector<uint64_t>> ancestors(count, std::vector<uint64_t>(words));
    for (uint32_t b = 0; b < count; ++b) {
        auto batch = graph.batches[b];
        if (batch.pass >= count || batchByPass[batch.pass] != UINT32_MAX) return "Duplicate or invalid scheduled pass";
        if (batch.queue >= s.queues.size() || !s.queues[batch.queue].active) return "Inactive schedule queue";
        const auto& compatible = s.passes[batch.pass].compatibleQueueSlots;
        if (std::find(compatible.begin(), compatible.end(), batch.queue) == compatible.end()) return "Incompatible schedule queue";
        batchByPass[batch.pass] = b;
        auto inherit = [&](uint32_t pred) {
            for (size_t w = 0; w < words; ++w) ancestors[b][w] |= ancestors[pred][w];
            ancestors[b][pred / 64] |= uint64_t{1} << (pred % 64);
        };
        if (lastQueue[batch.queue] != UINT32_MAX) inherit(lastQueue[batch.queue]);
        for (auto pred : waits[b]) inherit(pred);
        lastQueue[batch.queue] = b;
    }
    auto ordered = [&](uint32_t from, uint32_t to) {
        if (from >= count || to >= count) return false;
        auto a = batchByPass[from], b = batchByPass[to];
        return a < b && (ancestors[b][a / 64] & (uint64_t{1} << (a % 64))) != 0;
    };
    auto checkEdges = [&](const DependencyEdges& edges) {
        return std::all_of(edges.begin(), edges.end(), [&](auto e) { return e.first == e.second || ordered(e.first, e.second); });
    };
    if (!checkEdges(graph.edges) || !checkEdges(s.explicitEdges) || !checkEdges(s.placementEdges)
        || (input.expectedSchedulingEdges && !checkEdges(*input.expectedSchedulingEdges)))
        return "Schedule does not enforce a dependency or alias-placement edge";
    // The initial scheduler deliberately serializes all uses of a resource,
    // including read/read. This keeps state transitions safe until owned range
    // and state planning can prove which reads may overlap.
    std::vector<uint32_t> lastResource(s.resourceIDs.size(), UINT32_MAX);
    for (auto batch : graph.batches) {
        bool invalid = false;
        ForEachScheduledResource(s.passes[batch.pass], [&](uint32_t resource) {
            if (resource >= lastResource.size()) { invalid = true; return; }
            auto& previous = lastResource[resource];
            if (previous != UINT32_MAX && previous != batch.pass && !ordered(previous, batch.pass)) invalid = true;
            previous = batch.pass;
        });
        if (invalid) return "Invalid or unordered resource accesses across queues";
    }
    return {};
}

std::string ValidateSymbolicStates(const GraphCompileInput& input, const CompiledGraph& graph) {
    BT_ZONE_SCOPE("ORG.AsyncCompile.ValidateStates");
    // Independent dense-cell oracle; it does not share the planner's rectangle
    // splitting algorithm. Only the worker invokes this on captured host inputs.
    const auto& s = input.structure;
    if (!graph.states.complete || s.resourceShapes.size() != s.resourceIDs.size()) return "Missing state plan";
    struct Cell { CompileResourceState state; uint32_t batch = UINT32_MAX; };
    std::vector<std::vector<Cell>> cells(s.resourceShapes.size());
    uint64_t total = 0;
    for (size_t r = 0; r < cells.size(); ++r) {
        const auto shape = s.resourceShapes[r];
        if (!shape.mips && !shape.slices && !shape.hasLayout) continue;
        const auto size = uint64_t{shape.mips} * shape.slices;
        total += size;
        if (!size || total > 2'000'000) return "Invalid oracle dimensions";
        cells[r].resize(static_cast<size_t>(size));
    }
    std::vector<SymbolicStateStep> expected, actual;
    auto visit = [&](uint32_t r, CompileRange range, auto&& fn) {
        if (r >= cells.size() || !ValidRange(range, s.resourceShapes[r])) return false;
        for (uint32_t slice = range.slice; slice < range.slice + range.slices; ++slice)
            for (uint32_t mip = range.mip; mip < range.mip + range.mips; ++mip)
                fn(CompileRange{mip, 1, slice, 1}, cells[r][size_t{slice} * s.resourceShapes[r].mips + mip]);
        return true;
    };
    std::vector<bool> seen(s.passes.size());
    for (uint32_t b = 0; b < graph.batches.size(); ++b) {
        const auto p = graph.batches[b].pass;
        if (p >= seen.size() || seen[p]) return "Invalid state-plan pass layout";
        seen[p] = true;
        const auto& pass = s.passes[p];
        bool conflict = false;
        for (const auto& use : pass.entryStates) {
            if (!visit(use.resource, use.range, [&](CompileRange unit, Cell& cell) {
                const auto state = StateForShape(use.state, s.resourceShapes[use.resource]);
                if (cell.batch == b) { conflict |= cell.state != state; return; }
                expected.push_back({use.resource, b, cell.batch, unit, cell.state, state});
                cell = {state, b};
            })) return "Invalid oracle entry range";
        }
        for (const auto& use : pass.exitStates) {
            if (!visit(use.resource, use.range, [&](CompileRange, Cell& cell) {
                conflict |= cell.batch != b;
                cell = {StateForShape(use.state, s.resourceShapes[use.resource]), b};
            })) return "Invalid oracle exit range";
        }
        if (conflict) return "Conflicting state declarations";
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) return "Missing state-plan pass";
    for (const auto& step : graph.states.steps) {
        if (!visit(step.resource, step.range, [&](CompileRange unit, Cell&) {
            auto copy = step; copy.range = unit; actual.push_back(copy);
        })) return "Invalid symbolic step range";
    }
    auto stateKey = [](CompileResourceState state) { return std::tuple{state.access, state.layout, state.sync, state.write}; };
    auto key = [&](const SymbolicStateStep& step) {
        return std::tuple{step.resource, step.batch, step.range.slice, step.range.mip,
            step.previousBatch, stateKey(step.before), stateKey(step.after)};
    };
    auto order = [&](const auto& a, const auto& b) { return key(a) < key(b); };
    std::sort(expected.begin(), expected.end(), order);
    std::sort(actual.begin(), actual.end(), order);
    if (expected != actual) return "Symbolic entry/transition state mismatch";
    for (const auto& final : graph.states.finalStates) {
        bool mismatch = false;
        if (!visit(final.resource, final.range, [&](CompileRange, Cell& cell) {
            mismatch |= cell.batch == UINT32_MAX || cell.batch != final.batch || cell.state != final.state;
            cell.batch = UINT32_MAX; // Detect duplicate final-state coverage.
        }) || mismatch) return "Symbolic final-state mismatch";
    }
    for (const auto& resource : cells) for (const auto& cell : resource)
        if (cell.batch != UINT32_MAX) return "Missing symbolic final-state coverage";
    return {};
}

std::shared_ptr<const CompiledGraph> CompileWorkspace::Compile(
    std::shared_ptr<const GraphCompileInput> input, const std::atomic_bool& cancelled) {
    BT_ZONE_SCOPE("ORG.AsyncCompile.DependencyCompile");
    if (!input) throw std::invalid_argument("Null graph compile input");
    const auto& structure = input->structure;
    if (structure.passes.size() >= UINT32_MAX || structure.resourceIDs.size() >= UINT32_MAX)
        throw std::invalid_argument("Graph exceeds dependency index capacity");
    auto result = std::make_shared<CompiledGraph>();
    result->structure = std::make_shared<const GraphCompileStructure>(structure);
    const auto passCount = static_cast<uint32_t>(structure.passes.size());
    m_resources.clear();
    m_resources.resize(structure.resourceIDs.size());
    auto edge = [&](uint32_t from, uint32_t to) {
        if (from != UINT32_MAX && from != to) result->edges.emplace_back(from, to);
    };
    {
        BT_ZONE_SCOPE("ORG.AsyncCompile.BuildDependencies");
        for (uint32_t index = 0; index < passCount; ++index) {
            if (cancelled.load(std::memory_order_relaxed)) return {};
            const auto& pass = structure.passes[index];
            for (const auto& access : pass.accesses) {
                if (access.resourceIndex >= m_resources.size())
                    throw std::invalid_argument("Invalid captured dependency resource index");
                auto& state = m_resources[access.resourceIndex];
                edge(state.writer, index);
                if (access.write) {
                    for (auto reader : state.readers) edge(reader, index);
                    state.readers.clear();
                    state.writer = index;
                } else {
                    state.readers.push_back(index);
                }
                // Exclusive imported API ownership also orders read/read uses.
                if (state.backend != pass.backend) edge(state.lastAccess, index);
                state.lastAccess = index;
                state.backend = pass.backend;
            }
        }
        for (auto [from, to] : structure.explicitEdges) {
            if (from >= passCount || to >= passCount)
                throw std::invalid_argument("Invalid captured explicit dependency edge");
            edge(from, to);
        }
        std::sort(result->edges.begin(), result->edges.end());
        result->edges.erase(std::unique(result->edges.begin(), result->edges.end()), result->edges.end());
    }
    if (cancelled.load(std::memory_order_relaxed)) return {};
    result->schedulingEdges = result->edges;
    for (auto edge : structure.placementEdges) {
        if (edge.first >= passCount || edge.second >= passCount) throw std::invalid_argument("Invalid alias placement edge");
        if (edge.first != edge.second) result->schedulingEdges.push_back(edge);
    }
    std::sort(result->schedulingEdges.begin(), result->schedulingEdges.end());
    result->schedulingEdges.erase(std::unique(result->schedulingEdges.begin(), result->schedulingEdges.end()), result->schedulingEdges.end());
    {
        BT_ZONE_SCOPE("ORG.AsyncCompile.Topology");
        m_successors.clear(); m_successors.resize(passCount);
        m_indegrees.assign(passCount, 0);
        for (auto [from, to] : result->schedulingEdges) {
            m_successors[from].push_back(to);
            ++m_indegrees[to];
        }
        auto order = [&](uint32_t lhs, uint32_t rhs) {
            const auto l = structure.passes[lhs].originalOrder;
            const auto r = structure.passes[rhs].originalOrder;
            return l != r ? l > r : lhs > rhs;
        };
        m_ready.clear();
        for (uint32_t index = 0; index < passCount; ++index)
            if (!m_indegrees[index]) m_ready.push_back(index);
        std::make_heap(m_ready.begin(), m_ready.end(), order);
        while (!m_ready.empty()) {
            if (cancelled.load(std::memory_order_relaxed)) return {};
            std::pop_heap(m_ready.begin(), m_ready.end(), order);
            auto node = m_ready.back(); m_ready.pop_back();
            result->topologicalOrder.push_back(node);
            for (auto next : m_successors[node]) if (--m_indegrees[next] == 0) {
                m_ready.push_back(next);
                std::push_heap(m_ready.begin(), m_ready.end(), order);
            }
        }
        if (result->topologicalOrder.size() != passCount)
            throw std::runtime_error("Captured dependency graph contains a cycle");
        result->criticality.assign(passCount, 0);
        for (auto it = result->topologicalOrder.rbegin(); it != result->topologicalOrder.rend(); ++it)
            for (auto next : m_successors[*it])
                result->criticality[*it] = (std::max)(result->criticality[*it], 1 + result->criticality[next]);
    }
    {
        BT_ZONE_SCOPE("ORG.AsyncCompile.SymbolicSchedule");
        std::vector<uint32_t> batchByPass(passCount, UINT32_MAX), lastResource(structure.resourceIDs.size(), UINT32_MAX);
        for (auto passIndex : result->topologicalOrder) {
            if (cancelled.load(std::memory_order_relaxed)) return {};
            const auto& pass = structure.passes[passIndex];
            auto usable = [&](uint32_t queue) {
                return queue < structure.queues.size() && structure.queues[queue].active
                    && std::find(pass.compatibleQueueSlots.begin(), pass.compatibleQueueSlots.end(), queue) != pass.compatibleQueueSlots.end();
            };
            auto queue = pass.preferredQueueSlot;
            if (!usable(queue)) {
                auto it = std::find_if(pass.compatibleQueueSlots.begin(), pass.compatibleQueueSlots.end(), usable);
                if (it == pass.compatibleQueueSlots.end()) throw std::runtime_error("Pass has no active compatible compile queue");
                queue = *it;
            }
            batchByPass[passIndex] = static_cast<uint32_t>(result->batches.size());
            result->batches.push_back({passIndex, queue});
            ForEachScheduledResource(pass, [&](uint32_t resource) {
                if (resource >= lastResource.size()) throw std::invalid_argument("Invalid scheduled state resource");
                auto& previous = lastResource[resource];
                if (previous != UINT32_MAX && previous != passIndex) result->schedulingEdges.emplace_back(previous, passIndex);
                previous = passIndex;
            });
        }
        std::sort(result->schedulingEdges.begin(), result->schedulingEdges.end());
        result->schedulingEdges.erase(std::unique(result->schedulingEdges.begin(), result->schedulingEdges.end()), result->schedulingEdges.end());
        std::vector<std::vector<uint32_t>> latestProducer(passCount, std::vector<uint32_t>(structure.queues.size(), UINT32_MAX));
        for (auto [from, to] : result->schedulingEdges) {
            auto source = batchByPass[from], destination = batchByPass[to];
            auto sourceQueue = result->batches[source].queue;
            if (sourceQueue == result->batches[destination].queue) continue;
            auto& latest = latestProducer[destination][sourceQueue];
            if (latest == UINT32_MAX || source > latest) latest = source;
        }
        for (uint32_t b = 0; b < passCount; ++b) for (auto producer : latestProducer[b])
            if (producer != UINT32_MAX) result->relativeWaits.push_back({b, producer});
        result->stage = CompiledGraphStage::SymbolicSchedule;
    }
    if (!structure.resourceShapes.empty()) {
        result->states = BuildStatePlan(structure, result->batches, cancelled);
        if (cancelled.load(std::memory_order_relaxed)) return {};
        if (result->states.complete) result->stateValidationError = ValidateSymbolicStates(*input, *result);
    }
    return cancelled.load(std::memory_order_relaxed) ? nullptr : result;
}

struct GraphCompileCoordinator::Job {
    RequestState request;
    std::shared_ptr<const GraphCompileInput> input;
    std::atomic_bool cancel{false}, done{false};
    std::shared_ptr<const CompiledGraph> result;
    std::string error;
    uint64_t originalSequence = 0;
    std::chrono::steady_clock::time_point queued;
};

GraphCompileCoordinator::GraphCompileCoordinator(std::shared_ptr<runtime::ITaskService> tasks,
    size_t concurrency) : m_tasks(std::move(tasks)), m_concurrency(std::clamp(concurrency, size_t{1}, size_t{4})) {
    if (!m_tasks) throw std::invalid_argument("Async compilation requires a task service");
    m_scope = m_tasks->CreateScope("ORG.AsyncCompile");
    if (!m_scope) throw std::runtime_error("Task service rejected compile scope");
}
GraphCompileCoordinator::~GraphCompileCoordinator() { Shutdown(); }

void GraphCompileCoordinator::SetConcurrency(size_t concurrency) {
    m_concurrency = std::clamp(concurrency, size_t{1}, size_t{4});
}

uint64_t GraphCompileCoordinator::Request(GraphCompileInput input) {
    if (m_stopped) return 0;
    Pump();
    if (input.structure.generation < m_generation) { ++m_stats.rejected; return 0; }
    if (input.structure.generation != m_generation) Reset(input.structure.generation);
    NormalizeCompileInput(input);
    if (auto previous = m_previousRequest.lock()) {
        const auto& before = previous->structure;
        const auto& after = input.structure;
        m_stats.membershipChanges += before.resourceIDs != after.resourceIDs;
        m_stats.passChanges += before.passes != after.passes;
        m_stats.constraintChanges += before.explicitEdges != after.explicitEdges || before.placementEdges != after.placementEdges;
        m_stats.queueChanges += before.queues != after.queues;
    }
    RequestState request{++m_sequence, std::make_shared<const GraphCompileInput>(std::move(input)), std::chrono::steady_clock::now()};
    m_previousRequest = request.input;
    ++m_stats.requested;
    if (m_latest && m_latest->input->structure == request.input->structure) {
        ++m_stats.coalesced;
        Accept(request, m_latest->graph);
        // A newer request returning to the selected structure supersedes queued
        // intermediate publications as well.
        if (m_pending) { m_pending.reset(); ++m_stats.cancelled; }
        return request.sequence;
    }
    for (auto& job : m_jobs) {
        if (!job->cancel.load() && job->input->structure == request.input->structure) {
            job->request = request;
            ++m_stats.coalesced;
            if (m_pending) { m_pending.reset(); ++m_stats.cancelled; }
            return request.sequence;
        }
    }
    for (auto& plan : m_completedPlans) {
        if (*plan->structure == request.input->structure) {
            ++m_stats.coalesced;
            ++m_stats.completedCacheHits;
            // Accept may reorder the cache; retain the value across that call.
            auto cached = plan;
            Accept(request, std::move(cached));
            if (m_pending) { m_pending.reset(); ++m_stats.cancelled; }
            return request.sequence;
        }
    }
    if (m_pending) ++m_stats.cancelled;
    m_pending = std::make_unique<RequestState>(request);
    StartPending();
    return request.sequence;
}

void GraphCompileCoordinator::StartPending() {
    if (!m_pending || m_stopped || m_jobs.size() >= m_concurrency) return;
    auto job = std::make_shared<Job>();
    job->request = std::move(*m_pending); m_pending.reset();
    job->input = job->request.input;
    job->originalSequence = job->request.sequence;
    job->queued = job->request.queued;
    auto running = m_running;
    const bool accepted = m_tasks->Submit(m_scope, runtime::TaskPriority::Background,
        "ORG.AsyncCompile.Job", [job, running] {
            BT_ZONE_SCOPE("ORG.AsyncCompile.Job");
            BT_ZONE_VALUE(job->originalSequence);
            BT_PLOT("ORG.AsyncCompile.QueueDelayUs", static_cast<int64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - job->queued).count()));
            const size_t count = running->running.fetch_add(1) + 1;
            auto peak = running->peak.load();
            while (count > peak && !running->peak.compare_exchange_weak(peak, count)) {}
            try {
                CompileWorkspace workspace;
                job->result = workspace.Compile(job->input, job->cancel);
            } catch (const std::exception& e) { job->error = e.what(); }
            catch (...) { job->error = "Unknown graph compile exception"; }
            running->running.fetch_sub(1);
            job->done.store(true, std::memory_order_release);
        });
    if (!accepted) {
        ++m_stats.rejected;
        m_stats.lastError = "Task service rejected graph compile job";
        return;
    }
    m_jobs.push_back(std::move(job));
    ++m_stats.started;
    m_stats.peakActive = (std::max)(m_stats.peakActive, m_jobs.size());
}

void GraphCompileCoordinator::Accept(const RequestState& request,
    std::shared_ptr<const CompiledGraph> result) {
    if (!result || request.input->structure.generation != m_generation) return;
    if (!request.input->structure.resourceShapes.empty()) {
        if (!result->states.complete) {
            ++m_stats.stateFallbacks;
            m_stats.lastStateFallback = result->states.fallbackReason;
        }
        else {
            ++m_stats.stateComparisons;
            if (!result->stateValidationError.empty()) {
                ++m_stats.stateFailures;
                m_stats.lastError = "Async state oracle at request " + std::to_string(request.sequence) + ": " + result->stateValidationError;
                return;
            }
        }
    }
    if (request.input->expectedEdges) {
        ++m_stats.oracleComparisons;
        if (result->edges != *request.input->expectedEdges) {
            ++m_stats.oracleFailures;
            m_stats.lastError = "Async dependency oracle mismatch at request " + std::to_string(request.sequence);
            return;
        }
    }
    ++m_stats.scheduleComparisons;
    if (auto error = ValidateSymbolicSchedule(*request.input, *result); !error.empty()) {
        ++m_stats.scheduleFailures;
        m_stats.lastError = "Async schedule validation at request " + std::to_string(request.sequence) + ": " + error;
        return;
    }
    if (!m_latest || request.sequence > m_latest->sequence) {
        std::erase(m_completedPlans, result);
        m_completedPlans.push_back(result);
        if (m_completedPlans.size() > kMaximumCompletedPlans) m_completedPlans.erase(m_completedPlans.begin());
        m_latest = std::make_shared<const CompiledGraphBundle>(CompiledGraphBundle{
            request.sequence, std::move(result), request.input});
        m_stats.selectedSequence = request.sequence;
        for (const auto& job : m_jobs)
            if (job->request.sequence < request.sequence) job->cancel.store(true, std::memory_order_relaxed);
    }
}

void GraphCompileCoordinator::Pump() {
    BT_ZONE_SCOPE("ORG.AsyncCompile.Poll");
    for (auto it = m_jobs.begin(); it != m_jobs.end();) {
        auto& job = *it;
        if (!job->done.load(std::memory_order_acquire)) { ++it; continue; }
        if (job->cancel.load() || !job->result) {
            if (!job->error.empty()) { ++m_stats.failed; m_stats.lastError = job->error; }
            else ++m_stats.cancelled;
        } else {
            ++m_stats.completed;
            Accept(job->request, job->result);
        }
        it = m_jobs.erase(it);
    }
    StartPending();
}

void GraphCompileCoordinator::Reset(uint64_t generation) {
    if (generation < m_generation) throw std::invalid_argument("Compile generation regression");
    m_generation = generation;
    if (m_pending) { ++m_stats.cancelled; m_pending.reset(); }
    for (auto& job : m_jobs) job->cancel.store(true);
    m_latest.reset();
    m_completedPlans.clear();
    m_stats.selectedSequence = 0;
}

void GraphCompileCoordinator::Shutdown() {
    if (m_stopped) return;
    m_stopped = true;
    m_pending.reset();
    for (auto& job : m_jobs) job->cancel.store(true);
    // Do not cancel the scheduler scope: queued closures must run so their
    // completion mailboxes and scheduler accounting finish normally.
    if (m_scope) m_scope->Wait();
    Pump();
    m_jobs.clear(); m_latest.reset(); m_completedPlans.clear();
}

CompileCoordinatorStatistics GraphCompileCoordinator::Statistics() const {
    auto result = m_stats;
    result.active = m_jobs.size();
    result.pending = m_pending ? 1 : 0;
    result.peakRunning = m_running->peak.load();
    // Count retained compiler-owned vectors. Opaque publication/GPU leases are
    // accounted by their owning subsystem, not charged as zero-cost resources.
    std::vector<const GraphCompileInput*> inputs;
    auto structureBytes = [](const GraphCompileStructure& structure) {
        size_t bytes = structure.resourceIDs.capacity() * sizeof(uint64_t)
            + structure.resourceShapes.capacity() * sizeof(CompileResourceShape)
            + structure.passes.capacity() * sizeof(CompilePass)
            + (structure.explicitEdges.capacity() + structure.placementEdges.capacity()) * sizeof(DependencyEdges::value_type)
            + structure.queues.capacity() * sizeof(CompileQueue);
        for (const auto& pass : structure.passes)
            bytes += pass.accesses.capacity() * sizeof(CompileAccess) + pass.compatibleQueueSlots.capacity() * sizeof(uint32_t)
                + (pass.entryStates.capacity() + pass.exitStates.capacity()) * sizeof(CompileStateUse);
        return bytes;
    };
    auto accountInput = [&](const std::shared_ptr<const GraphCompileInput>& input) {
        if (!input || std::find(inputs.begin(), inputs.end(), input.get()) != inputs.end()) return;
        inputs.push_back(input.get());
        result.retainedBytes += sizeof(GraphCompileInput) + structureBytes(input->structure)
            + input->leases.capacity() * sizeof(std::shared_ptr<const void>);
        if (input->expectedEdges) result.retainedBytes += input->expectedEdges->capacity() * sizeof(DependencyEdges::value_type);
        if (input->expectedSchedulingEdges) result.retainedBytes += input->expectedSchedulingEdges->capacity() * sizeof(DependencyEdges::value_type);
    };
    for (const auto& job : m_jobs) { accountInput(job->input); accountInput(job->request.input); }
    if (m_pending) accountInput(m_pending->input);
    if (m_latest) accountInput(m_latest->input);
    for (const auto& graph : m_completedPlans) {
        const auto& structure = *graph->structure;
        result.retainedBytes += sizeof(GraphCompileStructure) + structureBytes(structure);
        result.retainedBytes += sizeof(CompiledGraph) + graph->edges.capacity() * sizeof(DependencyEdges::value_type)
            + graph->schedulingEdges.capacity() * sizeof(DependencyEdges::value_type)
            + graph->batches.capacity() * sizeof(SymbolicBatch)
            + graph->relativeWaits.capacity() * sizeof(RelativeQueueWait)
            + graph->states.steps.capacity() * sizeof(SymbolicStateStep)
            + graph->states.finalStates.capacity() * sizeof(SymbolicFinalState)
            + graph->topologicalOrder.capacity() * sizeof(uint32_t)
            + graph->criticality.capacity() * sizeof(uint32_t);
    }
    return result;
}

} // namespace org::experimental
