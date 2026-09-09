#include "Render/RenderGraph/ExperimentalGraphCompiler.h"
#include "FrameTrace.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <BasicTelemetry/Tracy.h>
#include <spdlog/spdlog.h>

namespace org::experimental {

bool IsExecutionCompatible(const CompiledGraphBundle& bundle,
    const GraphCompileInput& prepared) noexcept {
    if (!bundle.graph || !bundle.input || !bundle.graph->structure
        || !bundle.graph->scheduleValidated || !bundle.graph->scheduleValidationError.empty()
        || bundle.input->structure != prepared.structure
        || *bundle.graph->structure != prepared.structure) return false;
    size_t scheduledPasses = 0;
    for (const auto& batch : bundle.graph->batches) scheduledPasses += batch.passes.size();
    if (scheduledPasses != prepared.structure.passes.size()) return false;
    std::vector<uint8_t> mapped(prepared.structure.passes.size());
    for (const auto& pass : prepared.structure.passes) {
        if (pass.preparedPassIndex >= mapped.size() || mapped[pass.preparedPassIndex]) return false;
        mapped[pass.preparedPassIndex] = 1;
    }
    return std::find(mapped.begin(), mapped.end(), uint8_t{0}) == mapped.end();
}

std::shared_ptr<const GraphExecutionLayout> BuildExecutionLayout(
    std::shared_ptr<const CompiledGraphBundle> bundle,
    const GraphCompileInput& prepared) {
    BT_ZONE_SCOPE("ORG.AsyncExecution.BuildLayout");
    if (!bundle || !IsExecutionCompatible(*bundle, prepared))
        throw std::invalid_argument("Compiled graph is incompatible with prepared frame");
    auto result = std::make_shared<GraphExecutionLayout>();
    result->bundle = std::move(bundle);
    result->placements.resize(prepared.structure.passes.size());
    std::vector<uint8_t> assigned(prepared.structure.passes.size());
    for (uint32_t batch = 0; batch < result->bundle->graph->batches.size(); ++batch) {
        const auto& symbolic = result->bundle->graph->batches[batch];
        for (const auto pass : symbolic.passes) {
            const auto preparedPass = prepared.structure.passes.at(pass).preparedPassIndex;
            if (preparedPass >= assigned.size() || assigned[preparedPass])
                throw std::invalid_argument("Compiled graph has duplicate prepared pass placement");
            result->placements.at(preparedPass) = {preparedPass, batch, symbolic.queue};
            assigned[preparedPass] = 1;
        }
    }
    if (std::find(assigned.begin(), assigned.end(), uint8_t{0}) != assigned.end())
        throw std::invalid_argument("Compiled graph has incomplete prepared pass placement");
    return result;
}


ExecutionTimelineAdmission::ExecutionTimelineAdmission(std::vector<ExecutionTimelinePoint> queues,
    size_t maximumInFlight)
    : m_reserved(std::move(queues)), m_submitted(m_reserved), m_completed(m_reserved),
      m_maximumInFlight(maximumInFlight) {
    if (!maximumInFlight || maximumInFlight > 64) throw std::invalid_argument("Invalid execution capacity");
    m_retained.reserve(maximumInFlight); // Commit cannot allocate after submission.
    for (auto& completed : m_completed) completed.value = 0;
    for (size_t i = 0; i < m_reserved.size(); ++i) {
        if (!m_reserved[i].timeline) throw std::invalid_argument("Missing queue timeline");
        for (size_t j = 0; j < i; ++j)
            if (m_reserved[i].timeline == m_reserved[j].timeline)
                throw std::invalid_argument("Queue timelines must be exclusive");
    }
}

std::shared_ptr<const GraphExecutionTimeline> ExecutionTimelineAdmission::Prepare(
    std::shared_ptr<const CompiledGraphBundle> bundle,
    const std::vector<std::vector<ExecutionTimelinePoint>>& incomingWaits,
    std::vector<std::shared_ptr<const void>> executionLeases,
    std::vector<std::shared_ptr<const IPreparedExecutionBatch>> preparedBatches) {
    BT_ZONE_SCOPE("ORG.AsyncExecution.PrepareTimelines");
    if (m_failed || m_pending || InFlight() >= m_maximumInFlight)
        throw std::logic_error("Admission unavailable");
    if (!bundle || !bundle->graph || !bundle->input || !bundle->graph->structure
        || *bundle->graph->structure != bundle->input->structure)
        throw std::invalid_argument("Incompatible execution bundle");
    const auto& graph = *bundle->graph;
    if (graph.structure->queues.size() != m_reserved.size()
        || incomingWaits.size() != graph.batches.size() || graph.batches.empty())
        throw std::invalid_argument("Invalid execution timeline dimensions");
    if (!graph.scheduleValidated || !graph.scheduleValidationError.empty())
        throw std::invalid_argument("Invalid execution schedule");
    if (m_sequence == UINT64_MAX) throw std::overflow_error("Submission sequence exhausted");
    auto reserved = m_reserved; // Transactional: errors never consume values.
    auto result = std::make_shared<GraphExecutionTimeline>();
    result->submission = m_sequence + 1;
    result->bundle = std::move(bundle);
    result->executionLeases = std::move(executionLeases);
    result->preparedBatches = std::move(preparedBatches);
    result->batches.resize(graph.batches.size());
    for (size_t i = 0; i < graph.batches.size(); ++i) {
        auto& point = reserved.at(graph.batches[i].queue);
        if (point.value >= UINT64_MAX - 1) throw std::overflow_error("Queue timeline exhausted");
        result->batches[i].signal = {point.timeline, ++point.value};
        result->batches[i].waits = incomingWaits[i];
        for (auto wait : incomingWaits[i]) {
            if (!wait.timeline) throw std::invalid_argument("Missing wait timeline");
            // Cross-frame waits on our exclusive timelines cannot reference
            // reserved-but-unsubmitted work, including this execution itself.
            for (auto submitted : m_submitted)
                if (wait.timeline == submitted.timeline && wait.value > submitted.value)
                    throw std::invalid_argument("Incoming wait references unsubmitted work");
        }
    }
    for (auto wait : graph.relativeWaits)
        result->batches.at(wait.consumerBatch).waits.push_back(
            result->batches.at(wait.producerBatch).signal);
    for (auto& batch : result->batches) {
        std::sort(batch.waits.begin(), batch.waits.end(), [](auto a, auto b) {
            return a.timeline < b.timeline;
        });
        size_t count = 0;
        for (auto wait : batch.waits) {
            if (!wait.value) continue;
            if (count && batch.waits[count - 1].timeline == wait.timeline)
                batch.waits[count - 1].value = std::max(batch.waits[count - 1].value, wait.value);
            else batch.waits[count++] = wait;
        }
        batch.waits.resize(count);
    }
    m_reserved = std::move(reserved);
    m_sequence = result->submission;
    m_nextBatch = 0;
    m_pending = result;
    return result;
}

std::shared_ptr<const GraphExecutionTimeline> ExecutionTimelineAdmission::SubmitPrepared(
    std::shared_ptr<const CompiledGraphBundle> bundle,
    const std::vector<std::vector<ExecutionTimelinePoint>>& incomingWaits,
    const std::vector<std::shared_ptr<const IPreparedExecutionBatch>>& packets) {
    BT_ZONE_SCOPE("ORG.AsyncExecution.SubmitPrepared");
    if (!bundle || !bundle->graph || packets.size() != bundle->graph->batches.size())
        throw std::invalid_argument("Prepared packet count mismatch");
    std::vector<std::shared_ptr<const void>> leases;
    leases.reserve(packets.size());
    for (size_t i = 0; i < packets.size(); ++i) {
        const auto& packet = packets[i];
        if (!packet) throw std::invalid_argument("Missing prepared packet");
        if (packet->QueueSlot() != bundle->graph->batches[i].queue)
            throw std::invalid_argument("Prepared packet queue mismatch");
        leases.push_back(packet);
    }
    auto execution = Prepare(std::move(bundle), incomingWaits, std::move(leases), packets);
    CompletionSet completion;
    for (const auto& batch : execution->batches) completion.Include(batch.signal);
    for (uint32_t i = 0; i < packets.size(); ++i) {
        const auto receipt = packets[i]->Submit(execution->batches[i]);
        if (!receipt) {
            Fail(execution->submission, receipt);
            throw std::runtime_error("Prepared GPU submission failed; recovery required");
        }
        CommitBatch(execution->submission, i);
    }
    if (execution->bundle->input->frameContext)
        execution->bundle->input->frameContext->MarkSubmitted(std::move(completion));
    return execution;
}

void ExecutionTimelineAdmission::CommitBatch(uint64_t submission, uint32_t batch) {
    BT_ZONE_SCOPE("ORG.AsyncExecution.CommitTimeline");
    if (m_failed || !m_pending || submission != m_pending->submission || batch != m_nextBatch)
        throw std::logic_error("Out-of-order execution submission");
    const auto queue = m_pending->bundle->graph->batches.at(batch).queue;
    m_submitted.at(queue) = m_pending->batches.at(batch).signal;
    if (++m_nextBatch == m_pending->batches.size()) {
        m_retained.push_back(std::move(m_pending));
    }
}

void ExecutionTimelineAdmission::Fail(uint64_t submission, SubmissionReceipt receipt) {
    if (m_failed || !m_pending || submission != m_pending->submission || receipt)
        throw std::logic_error("Unknown failed execution");
    m_failed = true;
    if (const auto& frame = m_pending->bundle->input->frameContext) frame->EnterRecovery();
    m_failure = FailedExecutionBatch{submission, m_nextBatch, receipt};
    basic_telemetry::AddCounter(receipt.state == SubmissionState::SubmittedWithoutSignal
        ? "ORG.AsyncExecution.SubmittedWithoutSignal"
        : receipt.state == SubmissionState::SubmissionUncertain
            ? "ORG.AsyncExecution.SubmissionUncertain" : "ORG.AsyncExecution.FailedBeforeSubmit");
    // Retain the pending bundle: it may own partially submitted GPU work. Only
    // recovery/retirement can release the admission owner after GPU completion.
}

size_t ExecutionTimelineAdmission::RetireCompleted(std::span<const ExecutionTimelinePoint> completed) {
    BT_ZONE_SCOPE("ORG.AsyncExecution.RetireTimelines");
    if (completed.size() != m_completed.size()) throw std::invalid_argument("Invalid completion dimensions");
    for (size_t i = 0; i < completed.size(); ++i)
        if (completed[i].timeline != m_completed[i].timeline
            || completed[i].value < m_completed[i].value
            || completed[i].value > m_submitted[i].value)
            throw std::invalid_argument("Invalid completion observation");
    std::copy(completed.begin(), completed.end(), m_completed.begin());
    const auto before = m_retained.size();
    std::erase_if(m_retained, [&](const auto& execution) {
        for (size_t i = 0; i < execution->batches.size(); ++i)
            if (execution->batches[i].signal.value > completed[execution->bundle->graph->batches[i].queue].value)
                return false;
        BT_ZONE_SCOPE("ORG.Frame.Retire");
        AnnotateFrameTrace(execution->bundle->input->frameContext);
        for (const auto& packet : execution->preparedBatches)
            if (packet) packet->Complete(execution->submission);
        if (const auto& frame = execution->bundle->input->frameContext)
            if (!frame->Retire(completed)) throw std::logic_error("Frame completion disagrees with admission");
        return true;
    });
    // Failed partial packets remain recovery-owned: completion alone cannot
    // establish that recording/backend error recovery has released its objects.
    return before - m_retained.size();
}

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
    uint32_t pass = UINT32_MAX;
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
        for (const auto passIndex : batches[b].passes) {
        if (passIndex >= s.passes.size()) return fallback("Invalid scheduled pass");
        const auto& pass = s.passes[passIndex];
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
                if (exit && (previous.batch != b || previous.pass != passIndex))
                    return "Internal transition lacks matching entry declaration";
                if (!exit && previous.batch == b && previous.pass == passIndex && previous.state != state)
                    return "Conflicting overlapping entry states";
                if (!exit && (previous.batch != b || previous.pass != passIndex))
                    plan.steps.push_back({use.resource, b, previous.batch, overlap, previous.state, state,
                        passIndex, previous.pass});
                const auto old = previous.range;
                auto keep = [&](CompileRange range) {
                    if (range.mips && range.slices)
                        next.push_back({range, previous.state, previous.batch, previous.pass});
                };
                keep({old.mip, overlap.mip - old.mip, old.slice, old.slices});
                keep({overlap.mip + overlap.mips, old.mip + old.mips - overlap.mip - overlap.mips, old.slice, old.slices});
                keep({overlap.mip, overlap.mips, old.slice, overlap.slice - old.slice});
                keep({overlap.mip, overlap.mips, overlap.slice + overlap.slices, old.slice + old.slices - overlap.slice - overlap.slices});
                next.push_back({overlap, state, b, passIndex});
            }
            regions[use.resource].swap(next);
            return nullptr;
        };
        for (const auto& use : pass.entryStates) if (auto error = apply(use, false)) return fallback(error);
        for (const auto& use : pass.exitStates) if (auto error = apply(use, true)) return fallback(error);
        }
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
    if (!s.resourceKeys.empty() && s.resourceKeys.size() != order.size())
        throw std::invalid_argument("Captured resource key count mismatch");
    auto keyAt = [&](uint32_t i) -> std::string_view {
        return s.resourceKeys.empty() ? std::string_view{} : std::string_view{s.resourceKeys[i]};
    };
    std::sort(order.begin(), order.end(), [&](auto a, auto b) {
        return std::tuple{s.resourceIDs[a], keyAt(a)} < std::tuple{s.resourceIDs[b], keyAt(b)};
    });
    std::vector<uint64_t> ids;
    std::vector<std::string> keys;
    std::vector<CompileResourceShape> shapes;
    std::vector<uint64_t> backingGenerations;
    if (!s.resourceShapes.empty() && s.resourceShapes.size() != order.size())
        throw std::invalid_argument("Captured resource shape count mismatch");
    if (!input.backingGenerations.empty() && input.backingGenerations.size() != order.size())
        throw std::invalid_argument("Captured backing generation count mismatch");
    ids.reserve(order.size());
    if (!s.resourceKeys.empty()) keys.reserve(order.size());
    for (uint32_t i = 0; i < order.size(); ++i) {
        const auto id = s.resourceIDs[order[i]];
        const auto key = keyAt(order[i]);
        if (!ids.empty() && ids.back() == id
            && (keys.empty() || keys.back() == key))
            throw std::invalid_argument("Duplicate captured resource identity");
        ids.push_back(id); remap[order[i]] = i;
        if (!s.resourceKeys.empty()) keys.emplace_back(key);
        if (!s.resourceShapes.empty()) shapes.push_back(s.resourceShapes[order[i]]);
        if (!input.backingGenerations.empty()) backingGenerations.push_back(input.backingGenerations[order[i]]);
    }
    s.resourceIDs = std::move(ids);
    s.resourceKeys = std::move(keys);
    s.resourceShapes = std::move(shapes);
    input.backingGenerations = std::move(backingGenerations);
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
    const size_t count = s.passes.size(), batchCount = graph.batches.size();
    const size_t words = (batchCount + 63) / 64;
    if (graph.stage != CompiledGraphStage::SymbolicSchedule || graph.batches.empty())
        return "Missing complete symbolic schedule";
    std::vector<uint32_t> batchByPass(count, UINT32_MAX), positionByPass(count, UINT32_MAX);
    std::vector<uint32_t> lastQueue(s.queues.size(), UINT32_MAX);
    std::vector<std::vector<uint32_t>> waits(batchCount);
    for (auto wait : graph.relativeWaits) {
        if (wait.consumerBatch >= batchCount || wait.producerBatch >= wait.consumerBatch)
            return "Relative wait targets an invalid or future batch";
        waits[wait.consumerBatch].push_back(wait.producerBatch);
    }
    std::vector<std::vector<uint64_t>> ancestors(batchCount, std::vector<uint64_t>(words));
    for (uint32_t b = 0; b < batchCount; ++b) {
        const auto& batch = graph.batches[b];
        if (batch.passes.empty()) return "Empty symbolic batch";
        if (batch.queue >= s.queues.size() || !s.queues[batch.queue].active) return "Inactive schedule queue";
        for (uint32_t position = 0; position < batch.passes.size(); ++position) {
            const auto pass = batch.passes[position];
            if (pass >= count || batchByPass[pass] != UINT32_MAX) return "Duplicate or invalid scheduled pass";
            const auto& compatible = s.passes[pass].compatibleQueueSlots;
            if (std::find(compatible.begin(), compatible.end(), batch.queue) == compatible.end()) return "Incompatible schedule queue";
            batchByPass[pass] = b;
            positionByPass[pass] = position;
        }
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
        if (a == UINT32_MAX || b == UINT32_MAX) return false;
        if (a == b) return positionByPass[from] < positionByPass[to];
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
    for (const auto& batch : graph.batches) {
        for (const auto pass : batch.passes) {
            bool invalid = false;
            ForEachScheduledResource(s.passes[pass], [&](uint32_t resource) {
                if (resource >= lastResource.size()) { invalid = true; return; }
                auto& previous = lastResource[resource];
                if (previous != UINT32_MAX && previous != pass && !ordered(previous, pass)) invalid = true;
                previous = pass;
            });
            if (invalid) return "Invalid or unordered resource accesses across queues";
        }
    }
    if (std::find(batchByPass.begin(), batchByPass.end(), UINT32_MAX) != batchByPass.end())
        return "Missing scheduled pass";
    return {};
}

std::string ValidateSymbolicStates(const GraphCompileInput& input, const CompiledGraph& graph) {
    BT_ZONE_SCOPE("ORG.AsyncCompile.ValidateStates");
    // Independent dense-cell oracle; it does not share the planner's rectangle
    // splitting algorithm. Only the worker invokes this on captured host inputs.
    const auto& s = input.structure;
    if (!graph.states.complete || s.resourceShapes.size() != s.resourceIDs.size()) return "Missing state plan";
    struct Cell { CompileResourceState state; uint32_t batch = UINT32_MAX, pass = UINT32_MAX; };
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
        for (const auto p : graph.batches[b].passes) {
        if (p >= seen.size() || seen[p]) return "Invalid state-plan pass layout";
        seen[p] = true;
        const auto& pass = s.passes[p];
        bool conflict = false;
        for (const auto& use : pass.entryStates) {
            if (!visit(use.resource, use.range, [&](CompileRange unit, Cell& cell) {
                const auto state = StateForShape(use.state, s.resourceShapes[use.resource]);
                if (cell.batch == b && cell.pass == p) { conflict |= cell.state != state; return; }
                expected.push_back({use.resource, b, cell.batch, unit, cell.state, state, p, cell.pass});
                cell = {state, b, p};
            })) return "Invalid oracle entry range";
        }
        for (const auto& use : pass.exitStates) {
            if (!visit(use.resource, use.range, [&](CompileRange, Cell& cell) {
                conflict |= cell.batch != b || cell.pass != p;
                cell = {StateForShape(use.state, s.resourceShapes[use.resource]), b, p};
            })) return "Invalid oracle exit range";
        }
        if (conflict) return "Conflicting state declarations";
        }
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) return "Missing state-plan pass";
    for (const auto& step : graph.states.steps) {
        if (!visit(step.resource, step.range, [&](CompileRange unit, Cell&) {
            auto copy = step; copy.range = unit; actual.push_back(copy);
        })) return "Invalid symbolic step range";
    }
    auto stateKey = [](CompileResourceState state) { return std::tuple{state.access, state.layout, state.sync, state.write}; };
    auto key = [&](const SymbolicStateStep& step) {
        return std::tuple{step.resource, step.batch, step.pass, step.range.slice, step.range.mip,
            step.previousBatch, step.previousPass, stateKey(step.before), stateKey(step.after)};
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
    m_resources.resize(structure.resourceIDs.size());
    for (auto& state : m_resources) state.Reset();
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
                compiler::AppendDependencyAccess(m_resources[access.resourceIndex], index,
                    access.write, pass.backend, edge);
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
        // The kernel snapshots each indegree before mutating scratch, so this
        // accessor may read the same vector supplied as remaining-indegree storage.
        const auto topology = compiler::BuildTopologicalOrder<uint32_t>(passCount,
            [&](uint32_t i) { return structure.passes[i].originalOrder; },
            [&](uint32_t i) -> const auto& { return m_successors[i]; },
            [&](uint32_t i) { return m_indegrees[i]; },
            [&] { return cancelled.load(std::memory_order_relaxed); },
            m_indegrees, m_ready, result->topologicalOrder);
        if (topology == compiler::TopologyResult::Cancelled) return {};
        if (topology == compiler::TopologyResult::Cycle)
            throw std::runtime_error("Captured dependency graph contains a cycle");
        result->criticality.assign(passCount, 0);
        compiler::ComputeCriticality(result->topologicalOrder,
            [&](uint32_t i) -> const auto& { return m_successors[i]; },
            [&](uint32_t i) { return result->criticality[i]; },
            [&](uint32_t i, uint32_t value) { result->criticality[i] = value; });
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
            const std::array ready{passIndex};
            const auto selected = compiler::SelectFirstFitCandidate<uint32_t>(
                std::span<const uint32_t>{ready},
                [&](uint32_t node) -> std::span<const uint32_t> {
                    return structure.passes[node].compatibleQueueSlots;
                },
                [&](uint32_t node) { return structure.passes[node].preferredQueueSlot; },
                [](uint32_t) { return false; },
                [&](uint32_t, uint32_t queue) { return usable(queue); });
            if (!selected) throw std::runtime_error("Pass has no active compatible compile queue");
            const auto queue = selected->second;
            const bool extendPrevious = !pass.forceBatchIsolation && !result->batches.empty()
                && result->batches.back().queue == queue
                && !structure.passes[result->batches.back().passes.back()].forceBatchIsolation;
            if (!extendPrevious) result->batches.push_back({passIndex, queue});
            else result->batches.back().passes.push_back(passIndex);
            batchByPass[passIndex] = static_cast<uint32_t>(result->batches.size() - 1);
            ForEachScheduledResource(pass, [&](uint32_t resource) {
                if (resource >= lastResource.size()) throw std::invalid_argument("Invalid scheduled state resource");
                auto& previous = lastResource[resource];
                if (previous != UINT32_MAX && previous != passIndex) result->schedulingEdges.emplace_back(previous, passIndex);
                previous = passIndex;
            });
        }
        std::sort(result->schedulingEdges.begin(), result->schedulingEdges.end());
        result->schedulingEdges.erase(std::unique(result->schedulingEdges.begin(), result->schedulingEdges.end()), result->schedulingEdges.end());
        std::vector<std::vector<uint32_t>> latestProducer(result->batches.size(),
            std::vector<uint32_t>(structure.queues.size(), UINT32_MAX));
        for (auto [from, to] : result->schedulingEdges) {
            auto source = batchByPass[from], destination = batchByPass[to];
            auto sourceQueue = result->batches[source].queue;
            if (sourceQueue == result->batches[destination].queue) continue;
            auto& latest = latestProducer[destination][sourceQueue];
            if (latest == UINT32_MAX || source > latest) latest = source;
        }
        for (uint32_t b = 0; b < result->batches.size(); ++b) for (auto producer : latestProducer[b])
            if (producer != UINT32_MAX) result->relativeWaits.push_back({b, producer});
        result->stage = CompiledGraphStage::SymbolicSchedule;
    }
    if (!structure.resourceShapes.empty()) {
        result->states = BuildStatePlan(structure, result->batches, cancelled);
        if (cancelled.load(std::memory_order_relaxed)) return {};
        if (result->states.complete) result->stateValidationError = ValidateSymbolicStates(*input, *result);
    }
    result->scheduleValidationError = ValidateSymbolicSchedule(*input, *result);
    result->scheduleValidated = true;
    return cancelled.load(std::memory_order_relaxed) ? nullptr : result;
}

std::shared_ptr<const CompiledGraph> CompileGraph(
    std::shared_ptr<const GraphCompileInput> input,
    CompileWorkspace& workspace,
    const std::atomic_bool& cancelled) {
    return workspace.Compile(std::move(input), cancelled);
}

struct GraphCompileCoordinator::Job {
    RequestState request;
    // Structurally identical frame requests share compiler work, but retain
    // distinct owned payloads and each publish an executable queue entry.
    std::vector<RequestState> followers;
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
    return RequestOwned(std::move(input)).sequence;
}

GraphCompileCoordinator::RequestReceipt GraphCompileCoordinator::RequestOwned(
    GraphCompileInput input, bool compileInline) {
    if (m_stopped) return {};
    Pump();
    if (input.structure.generation < m_generation) { ++m_stats.rejected; return {}; }
    if (input.structure.generation != m_generation) Reset(input.structure.generation);
    NormalizeCompileInput(input);
    if (auto previous = m_previousRequest.lock()) {
        const auto& before = previous->structure;
        const auto& after = input.structure;
        const bool membershipChanged = before.resourceIDs != after.resourceIDs;
        m_stats.membershipChanges += membershipChanged;
        if (membershipChanged && (m_stats.membershipChanges == 1
            || (m_stats.membershipChanges % 128) == 0)) {
            auto removed = before.resourceIDs.begin(), added = after.resourceIDs.begin();
            while (removed != before.resourceIDs.end()
                && std::binary_search(after.resourceIDs.begin(), after.resourceIDs.end(), *removed)) ++removed;
            while (added != after.resourceIDs.end()
                && std::binary_search(before.resourceIDs.begin(), before.resourceIDs.end(), *added)) ++added;
            const auto beforeIndex = static_cast<size_t>(removed - before.resourceIDs.begin());
            const auto afterIndex = static_cast<size_t>(added - after.resourceIDs.begin());
            const auto beforeID = removed == before.resourceIDs.end() ? uint64_t{0} : *removed;
            const auto afterID = added == after.resourceIDs.end() ? uint64_t{0} : *added;
            const auto beforeKey = beforeIndex < before.resourceKeys.size() ? before.resourceKeys[beforeIndex] : std::string{};
            const auto afterKey = afterIndex < after.resourceKeys.size() ? after.resourceKeys[afterIndex] : std::string{};
            spdlog::info("Async compile membership changed count={} beforeCount={} afterCount={} firstBefore={}:'{}' firstAfter={}:'{}'",
                m_stats.membershipChanges, before.resourceIDs.size(), after.resourceIDs.size(),
                beforeID, beforeKey, afterID, afterKey);
        }
        const bool passesChanged = before.passes != after.passes;
        m_stats.passChanges += passesChanged;
        if (passesChanged && (m_stats.passChanges == 1 || (m_stats.passChanges % 128) == 0)) {
            const auto mismatch = std::mismatch(before.passes.begin(), before.passes.end(),
                after.passes.begin(), after.passes.end());
            const auto index = static_cast<size_t>(mismatch.first - before.passes.begin());
            const auto* a = mismatch.first == before.passes.end() ? nullptr : &*mismatch.first;
            const auto* b = mismatch.second == after.passes.end() ? nullptr : &*mismatch.second;
            spdlog::info("Async compile pass changed count={} index={} beforeCount={} afterCount={} access={} entry={} exit={} isolation={} placement={} queue={}",
                m_stats.passChanges, index, before.passes.size(), after.passes.size(),
                a && b && a->accesses != b->accesses,
                a && b && a->entryStates != b->entryStates,
                a && b && a->exitStates != b->exitStates,
                a && b && a->forceBatchIsolation != b->forceBatchIsolation,
                a && b && (a->originalOrder != b->originalOrder || a->preparedPassIndex != b->preparedPassIndex),
                a && b && (a->compatibleQueueSlots != b->compatibleQueueSlots
                    || a->preferredQueueSlot != b->preferredQueueSlot));
        }
        m_stats.constraintChanges += before.explicitEdges != after.explicitEdges || before.placementEdges != after.placementEdges;
        m_stats.queueChanges += before.queues != after.queues;
        m_stats.realizationChanges += previous->backingGenerations != input.backingGenerations;
    }
    RequestState request{++m_sequence, std::make_shared<const GraphCompileInput>(std::move(input)), std::chrono::steady_clock::now()};
    m_previousRequest = request.input;
    ++m_stats.requested;
    for (auto& job : m_jobs) {
        if (!job->cancel.load() && job->input->structure == request.input->structure) {
            job->followers.push_back(request);
            ++m_stats.coalesced;
            return {request.sequence, request.input};
        }
    }
    for (auto& plan : m_completedPlans) {
        if (*plan->structure == request.input->structure) {
            ++m_stats.coalesced;
            ++m_stats.completedCacheHits;
            // Accept may reorder the cache; retain the value across that call.
            auto cached = plan;
            Accept(request, std::move(cached));
            return {request.sequence, request.input};
        }
    }
    if (compileInline) {
        BT_ZONE_SCOPE("ORG.AsyncCompile.InlineBootstrap");
        ++m_stats.started;
        try {
            std::atomic_bool cancelled{false};
            CompileWorkspace workspace;
            auto result = CompileGraph(request.input, workspace, cancelled);
            if (!result) throw std::runtime_error("Inline graph compile was cancelled");
            ++m_stats.completed;
            Accept(request, std::move(result));
        } catch (const std::exception& error) {
            ++m_stats.failed;
            m_stats.lastError = error.what();
            throw;
        }
        return {request.sequence, request.input};
    }
    m_pending.push_back(request);
    StartPending();
    return {request.sequence, request.input};
}

void GraphCompileCoordinator::StartPending() {
    while (!m_pending.empty() && !m_stopped && m_jobs.size() < m_concurrency) {
        auto job = std::make_shared<Job>();
        job->request = std::move(m_pending.front());
        m_pending.pop_front();
        job->input = job->request.input;
        job->originalSequence = job->request.sequence;
        job->queued = job->request.queued;
        auto running = m_running;
        auto completion = m_completion;
        const bool accepted = m_tasks->Submit(m_scope, runtime::TaskPriority::Background,
        "ORG.AsyncCompile.Job", [job, running, completion] {
            BT_ZONE_SCOPE("ORG.AsyncCompile.Job");
            BT_ZONE_VALUE(job->originalSequence);
            AnnotateFrameTrace(job->input->frameContext);
            BT_PLOT("ORG.AsyncCompile.QueueDelayUs", static_cast<int64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - job->queued).count()));
            const size_t count = running->running.fetch_add(1) + 1;
            auto peak = running->peak.load();
            while (count > peak && !running->peak.compare_exchange_weak(peak, count)) {}
            try {
                CompileWorkspace workspace;
                job->result = CompileGraph(job->input, workspace, job->cancel);
            } catch (const std::exception& e) { job->error = e.what(); }
            catch (...) { job->error = "Unknown graph compile exception"; }
            running->running.fetch_sub(1);
            job->done.store(true, std::memory_order_release);
            {
                std::lock_guard lock(completion->mutex);
                ++completion->revision;
            }
            completion->changed.notify_all();
            });
        ++m_stats.started;
        if (!accepted) {
            // A scheduler rejection must not discard an owned one-shot packet.
            // Compile on the preparation owner through the identical compiler.
            try {
                CompileWorkspace workspace;
                job->result = CompileGraph(job->input, workspace, job->cancel);
            } catch (const std::exception& error) { job->error = error.what(); }
            catch (...) { job->error = "Unknown inline graph compile exception"; }
            job->done.store(true, std::memory_order_release);
            {
                std::lock_guard lock(m_completion->mutex);
                ++m_completion->revision;
            }
            m_completion->changed.notify_all();
        }
        m_jobs.push_back(std::move(job));
        m_stats.peakActive = (std::max)(m_stats.peakActive, m_jobs.size());
    }
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
                m_failures.emplace(request.sequence, m_stats.lastError);
                return;
            }
        }
    }
    if (request.input->expectedEdges) {
        ++m_stats.oracleComparisons;
        if (result->edges != *request.input->expectedEdges) {
            ++m_stats.oracleFailures;
            m_stats.lastError = "Async dependency oracle mismatch at request " + std::to_string(request.sequence);
            m_failures.emplace(request.sequence, m_stats.lastError);
            return;
        }
    }
    ++m_stats.scheduleComparisons;
    if (!result->scheduleValidated || !result->scheduleValidationError.empty()) {
        ++m_stats.scheduleFailures;
        m_stats.lastError = "Async schedule validation at request " + std::to_string(request.sequence) + ": "
            + (result->scheduleValidated ? result->scheduleValidationError : "validation was not performed");
        m_failures.emplace(request.sequence, m_stats.lastError);
        return;
    }
    std::erase(m_completedPlans, result);
    m_completedPlans.push_back(result);
    if (m_completedPlans.size() > kMaximumCompletedPlans) m_completedPlans.erase(m_completedPlans.begin());
    auto bundle = std::make_shared<const CompiledGraphBundle>(CompiledGraphBundle{
        request.sequence, std::move(result), request.input});
    m_ready.emplace(request.sequence, bundle);
    m_stats.highestReadySequence = (std::max)(m_stats.highestReadySequence, request.sequence);
}

std::shared_ptr<const CompiledGraphBundle> GraphCompileCoordinator::PeekReady(uint64_t sequence) const {
    const auto found = m_ready.find(sequence);
    if (found == m_ready.end()) return {};
    return found->second;
}

void GraphCompileCoordinator::Pump() {
    BT_ZONE_SCOPE("ORG.AsyncCompile.Poll");
    for (auto it = m_jobs.begin(); it != m_jobs.end();) {
        auto& job = *it;
        if (!job->done.load(std::memory_order_acquire)) { ++it; continue; }
        if (job->cancel.load() || !job->result) {
            const std::string error = !job->error.empty() ? job->error : "Graph compilation cancelled";
            if (!job->error.empty()) { ++m_stats.failed; m_stats.lastError = job->error; }
            else ++m_stats.cancelled;
            m_failures.emplace(job->request.sequence, error);
            for (const auto& follower : job->followers) m_failures.emplace(follower.sequence, error);
        } else {
            ++m_stats.completed;
            Accept(job->request, job->result);
            for (const auto& follower : job->followers) Accept(follower, job->result);
        }
        it = m_jobs.erase(it);
    }
    StartPending();
}

void GraphCompileCoordinator::WaitForSequence(uint64_t sequence) {
    while (!m_stopped && !m_ready.contains(sequence) && !m_failures.contains(sequence)) {
        uint64_t observed = 0;
        {
            std::lock_guard lock(m_completion->mutex);
            observed = m_completion->revision;
        }
        Pump();
        if (m_jobs.empty() && m_pending.empty()) break;
        if (m_ready.contains(sequence) || m_failures.contains(sequence)) break;
        std::unique_lock lock(m_completion->mutex);
        m_completion->changed.wait(lock, [&] {
            return m_completion->revision != observed || m_stopped;
        });
    }
}

std::shared_ptr<const CompiledGraphBundle> GraphCompileCoordinator::WaitAndPop(uint64_t sequence) {
    BT_ZONE_SCOPE("ORG.AsyncCompile.WaitForQueueHead");
    WaitForSequence(sequence);
    if (auto failure = m_failures.find(sequence); failure != m_failures.end()) {
        auto error = std::move(failure->second);
        m_failures.erase(failure);
        throw std::runtime_error("Asynchronous graph compilation failed for frame "
            + std::to_string(sequence) + ": " + error);
    }
    auto found = m_ready.find(sequence);
    if (found == m_ready.end()) return {};
    auto result = std::move(found->second);
    m_ready.erase(found);
    return result;
}

void GraphCompileCoordinator::Reset(uint64_t generation) {
    if (generation < m_generation) throw std::invalid_argument("Compile generation regression");
    m_generation = generation;
    m_sequence = 0;
    m_stats.cancelled += m_pending.size();
    for (const auto& request : m_pending)
        if (request.input->executionLifecycle) request.input->executionLifecycle->Abandon(1);
    m_pending.clear();
    for (auto& job : m_jobs) {
        job->cancel.store(true);
        if (job->request.input->executionLifecycle) job->request.input->executionLifecycle->Abandon(1);
        for (const auto& follower : job->followers)
            if (follower.input->executionLifecycle) follower.input->executionLifecycle->Abandon(1);
    }
    for (const auto& [_, bundle] : m_ready)
        if (bundle && bundle->input && bundle->input->executionLifecycle)
            bundle->input->executionLifecycle->Abandon(1);
    m_ready.clear();
    m_failures.clear();
    m_completedPlans.clear();
    m_stats.highestReadySequence = 0;
}

void GraphCompileCoordinator::Shutdown() {
    if (m_stopped) return;
    m_stopped = true;
    for (const auto& request : m_pending)
        if (request.input->executionLifecycle) request.input->executionLifecycle->Abandon(0);
    m_pending.clear();
    for (auto& job : m_jobs) {
        job->cancel.store(true);
        if (job->request.input->executionLifecycle) job->request.input->executionLifecycle->Abandon(0);
        for (const auto& follower : job->followers)
            if (follower.input->executionLifecycle) follower.input->executionLifecycle->Abandon(0);
    }
    for (const auto& [_, bundle] : m_ready)
        if (bundle && bundle->input && bundle->input->executionLifecycle)
            bundle->input->executionLifecycle->Abandon(0);
    // Do not cancel the scheduler scope: queued closures must run so their
    // completion mailboxes and scheduler accounting finish normally.
    if (m_scope) m_scope->Wait();
    Pump();
    m_jobs.clear(); m_ready.clear(); m_failures.clear(); m_completedPlans.clear();
}

CompileCoordinatorStatistics GraphCompileCoordinator::Statistics() const {
    auto result = m_stats;
    result.active = m_jobs.size();
    result.pending = m_pending.size();
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
            + input->leases.capacity() * sizeof(std::shared_ptr<const void>)
            + input->backingGenerations.capacity() * sizeof(uint64_t);
        if (input->expectedEdges) result.retainedBytes += input->expectedEdges->capacity() * sizeof(DependencyEdges::value_type);
        if (input->expectedSchedulingEdges) result.retainedBytes += input->expectedSchedulingEdges->capacity() * sizeof(DependencyEdges::value_type);
    };
    for (const auto& job : m_jobs) {
        accountInput(job->input);
        accountInput(job->request.input);
    }
    for (const auto& pending : m_pending) accountInput(pending.input);
    for (const auto& [sequence, bundle] : m_ready) {
        (void)sequence;
        accountInput(bundle->input);
    }
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
        for (const auto& batch : graph->batches)
            result.retainedBytes += batch.passes.capacity() * sizeof(uint32_t);
    }
    return result;
}

} // namespace org::experimental
