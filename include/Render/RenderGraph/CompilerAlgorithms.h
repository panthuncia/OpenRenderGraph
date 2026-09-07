#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

// Pure compiler kernels shared by live and owned-input compilation. Callers
// provide views/accessors and own all scratch; no graph/resource/pass callbacks.
namespace org::compiler {

template<class Index>
struct DependencySequence {
    static constexpr Index absent = (std::numeric_limits<Index>::max)();
    Index writer = absent, lastAccess = absent;
    uint32_t backend = 0;
    std::vector<Index> readers;

    void Reset() {
        writer = lastAccess = absent;
        backend = 0;
        readers.clear();
    }
};

template<class Index, class Emit>
void AppendDependencyAccess(DependencySequence<Index>& state, Index pass,
    bool write, uint32_t backend, Emit&& emit) {
    auto edge = [&](Index from) {
        if (from != DependencySequence<Index>::absent && from != pass) emit(from, pass);
    };
    edge(state.writer);
    if (write) {
        for (auto reader : state.readers) edge(reader);
        state.readers.clear();
        state.writer = pass;
    } else {
        state.readers.push_back(pass);
    }
    // Imported API ownership orders read/read as well as data hazards.
    if (state.backend != backend) edge(state.lastAccess);
    state.backend = backend;
    state.lastAccess = pass;
}

enum class TopologyResult { Complete, Cycle, Cancelled };

template<class Index, class Order, class Successors, class Indegree, class Cancel>
TopologyResult BuildTopologicalOrder(size_t count, Order&& originalOrder,
    Successors&& successors, Indegree&& indegree, Cancel&& cancelled,
    std::vector<uint32_t>& remaining, std::vector<Index>& ready,
    std::vector<Index>& result) {
    if (count >= (std::numeric_limits<Index>::max)())
        throw std::invalid_argument("Dependency graph exceeds index capacity");
    remaining.resize(count);
    ready.clear(); ready.reserve(count);
    result.clear(); result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        remaining[i] = indegree(static_cast<Index>(i));
        if (!remaining[i]) ready.push_back(static_cast<Index>(i));
    }
    auto less = [&](Index lhs, Index rhs) {
        auto l = originalOrder(lhs), r = originalOrder(rhs);
        return l != r ? l > r : lhs > rhs;
    };
    std::make_heap(ready.begin(), ready.end(), less);
    while (!ready.empty()) {
        if (cancelled()) return TopologyResult::Cancelled;
        std::pop_heap(ready.begin(), ready.end(), less);
        auto node = ready.back(); ready.pop_back();
        result.push_back(node);
        for (auto next : successors(node)) {
            if (next >= count || !remaining[next])
                throw std::invalid_argument("Invalid dependency adjacency/indegree");
            if (--remaining[next] == 0) {
                ready.push_back(next);
                std::push_heap(ready.begin(), ready.end(), less);
            }
        }
    }
    return result.size() == count ? TopologyResult::Complete : TopologyResult::Cycle;
}

template<class Order, class Successors, class Get, class Set>
void ComputeCriticality(const Order& topologicalOrder, Successors&& successors,
    Get&& get, Set&& set) {
    for (auto it = topologicalOrder.rbegin(); it != topologicalOrder.rend(); ++it) {
        uint32_t best = 0;
        for (auto next : successors(*it)) best = (std::max)(best, 1u + get(next));
        set(*it, best);
    }
}

// Shared deterministic first-fit queue selection.  The live compiler and the
// owned-input compiler intentionally use this exact routine; policy belongs in
// compiler input/callbacks, never in the worker route.  The ready vector stores
// node indices, while the returned first component is an index into that vector.
template<class Index, class CompatibleQueues, class PreferredQueue,
    class QueueHasWork, class Fits>
std::optional<std::pair<size_t, uint32_t>> SelectFirstFitCandidate(
    std::span<const Index> ready,
    CompatibleQueues&& compatibleQueues,
    PreferredQueue&& preferredQueue,
    QueueHasWork&& queueHasWork,
    Fits&& fits) {
    for (size_t readyIndex = 0; readyIndex < ready.size(); ++readyIndex) {
        const Index node = ready[readyIndex];
        const auto queues = compatibleQueues(node);
        auto tryQueue = [&](uint32_t queue) {
            return fits(node, queue);
        };

        // Prefer extending work already present on a compatible queue, then
        // the pass preference, and finally the remaining compatible queues.
        for (auto queue : queues)
            if (queueHasWork(static_cast<uint32_t>(queue))
                && tryQueue(static_cast<uint32_t>(queue)))
                return std::pair{readyIndex, static_cast<uint32_t>(queue)};

        const auto preferred = static_cast<uint32_t>(preferredQueue(node));
        if (tryQueue(preferred)) return std::pair{readyIndex, preferred};

        for (auto queue : queues) {
            const auto candidate = static_cast<uint32_t>(queue);
            if (candidate != preferred && tryQueue(candidate))
                return std::pair{readyIndex, candidate};
        }
    }
    return std::nullopt;
}

} // namespace org::compiler
