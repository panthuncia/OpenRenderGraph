#include "Render/RenderGraph/ExecutionBoundary.h"
#include "Render/RenderGraph/GraphReplay.h"
#include <cstdio>
#include <cstdlib>
#include <sstream>

using namespace org::experimental;
#define CHECK(...) do { if (!(__VA_ARGS__)) { std::fprintf(stderr, "Check failed at %d: %s\n", __LINE__, #__VA_ARGS__); std::abort(); } } while (false)

template<class F> void Rejects(F&& fn) {
    bool rejected = false;
    try { fn(); } catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
}

int main() {
    auto structure = std::make_shared<GraphCompileStructure>();
    structure->resourceIDs = {17, 42};
    CompiledGraph graph;
    graph.structure = structure;
    graph.batches.resize(1);
    graph.batches[0].queue = 3;
    const CompileResourceState read{1, 2, 4, false};
    const CompileResourceState write{8, 2, 16, true};
    graph.boundaryAccessesByBatch = {{
        {0, {0, 1, 0, 1}, read},
        {0, {0, 1, 0, 1}, write},
        {1, {1, 2, 3, 1}, write}}};
    graph.boundaryAccessesByBatch[0][0].byteOffset = 32;
    graph.boundaryAccessesByBatch[0][0].byteSize = 64;
    graph.boundaryAccessesByBatch[0][2].aspects = 2;
    structure->resourceShapes = {{1, 1, false}, {4, 6, true}};
    structure->passes.resize(1);
    structure->passes[0].entryStates = graph.boundaryAccessesByBatch[0];
    std::stringstream replay;
    WriteGraphReplay(replay, *structure);
    CHECK(ReadGraphReplay(replay).passes[0].entryStates == structure->passes[0].entryStates);
    std::vector<PreparedBackingState> backings(2);
    backings[0].resource = {7, 11};
    backings[1].resource = {8, 12};
    backings[1].shape = {4, 6, true};
    auto lease = std::make_shared<int>(123);
    std::weak_ptr<int> lifetime = lease;
    auto manifest = BuildExecutionBoundaryManifest(graph, 0, backings, {lease});
    lease.reset();
    CHECK(!lifetime.expired());
    CHECK(manifest->queueSlot == 3 && manifest->accesses.size() == 3);
    CHECK(manifest->accesses[0].backing.index == 7 && manifest->accesses[0].backing.generation == 11);
    CHECK(manifest->accesses[0].state == read && manifest->accesses[1].state == write);
    CHECK(manifest->accesses[0].offset == 32 && manifest->accesses[0].size == 64);
    CHECK(manifest->accesses[2].aspects == 2);
    CHECK(!manifest->accesses[0].image && manifest->accesses[2].image);
    CHECK(manifest->accesses[2].subresources == (CompileRange{1, 2, 3, 1}));
    backings[0].resource.generation++;
    const auto replacement = BuildExecutionBoundaryManifest(graph, 0, backings);
    CHECK(replacement->accesses[0].backing.generation == 12);
    CHECK(manifest->accesses[0].backing.generation == 11);
    manifest.reset();
    CHECK(lifetime.expired());
    graph.boundaryAccessesByBatch[0][2].range.mips = UINT32_MAX;
    Rejects([&] { BuildExecutionBoundaryManifest(graph, 0, backings); });
    graph.boundaryAccessesByBatch[0][2].range.mips = 2;
    graph.boundaryAccessesByBatch[0][2].resource = 2;
    Rejects([&] { BuildExecutionBoundaryManifest(graph, 0, backings); });
    graph.boundaryAccessesByBatch[0].clear();
    CHECK(BuildExecutionBoundaryManifest(graph, 0, backings)->accesses.empty());
    graph.boundaryAccessesByBatch.clear();
    Rejects([&] { BuildExecutionBoundaryManifest(graph, 0, backings); });
    std::puts("Execution boundary manifest tests passed");
}
