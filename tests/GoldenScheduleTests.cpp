// Golden schedules: each replay fixture is compiled and its schedule (order, batches, waits, state steps) compared with a
// checked-in text dump, so a compiler refactor that changes a schedule fails here and its diff reads in review.
// ORG_UPDATE_GOLDEN=1 rewrites the dumps instead of comparing.
#include "Render/RenderGraph/GraphReplay.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace org::experimental;

namespace {
void Fail(const std::string& message)
{
    std::fprintf(stderr, "Golden schedule check failed: %s\n", message.c_str());
    std::exit(1);
}

void WriteState(std::ostream& out, const CompileResourceState& state)
{
    out << state.access << '/' << state.layout << '/' << state.sync << (state.write ? "w" : "r");
}

void WriteRange(std::ostream& out, const CompileRange& range)
{
    out << 'm' << range.mip << '+' << range.mips << 's' << range.slice << '+' << range.slices;
}

std::string DumpSchedule(GraphCompileStructure structure)
{
    GraphCompileInput input;
    input.structure = std::move(structure);
    NormalizeCompileInput(input);
    CompileWorkspace workspace;
    std::atomic_bool cancelled{ false };
    const auto graph = CompileGraph(std::make_shared<const GraphCompileInput>(std::move(input)), workspace, cancelled);
    if (!graph) Fail("compile returned no graph");

    std::ostringstream out;
    out << "passes " << graph->structure->passes.size() << " resources " << graph->structure->resourceIDs.size()
        << " queues " << graph->structure->queues.size() << '\n';
    out << "stage " << static_cast<int>(graph->stage) << '\n';
    out << "schedule-error " << graph->scheduleValidationError << '\n';
    out << "state-error " << graph->stateValidationError << '\n';
    out << "order";
    for (auto pass : graph->topologicalOrder) out << ' ' << pass;
    out << '\n';
    for (size_t i = 0; i < graph->batches.size(); ++i) {
        out << "batch " << i << " q" << graph->batches[i].queue << ':';
        for (auto pass : graph->batches[i].passes) out << ' ' << pass;
        out << '\n';
    }
    for (const auto& wait : graph->relativeWaits) out << "wait " << wait.consumerBatch << " <- " << wait.producerBatch << '\n';
    out << "states complete " << graph->states.complete << " fallback " << graph->states.fallbackReason << '\n';
    for (const auto& step : graph->states.steps) {
        out << "step r" << step.resource << " b" << step.batch << " p" << step.pass << " from b"
            << static_cast<int64_t>(step.previousBatch == UINT32_MAX ? -1 : step.previousBatch) << " p"
            << static_cast<int64_t>(step.previousPass == UINT32_MAX ? -1 : step.previousPass) << ' ';
        WriteRange(out, step.range);
        out << ' ';
        WriteState(out, step.before);
        out << " -> ";
        WriteState(out, step.after);
        out << '\n';
    }
    for (const auto& final : graph->states.finalStates) {
        out << "final r" << final.resource << " b" << final.batch << ' ';
        WriteRange(out, final.range);
        out << ' ';
        WriteState(out, final.state);
        out << '\n';
    }
    return out.str();
}

std::string ReadFile(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    // A checkout may convert the golden files to CRLF; the dumps use LF.
    auto content = text.str();
    std::erase(content, '\r');
    return content;
}

void CheckGolden(const std::filesystem::path& fixture, const std::string& dump, bool update)
{
    auto golden = fixture;
    golden.replace_extension(".schedule");
    if (update) {
        std::ofstream(golden, std::ios::binary) << dump;
        std::printf("updated %s\n", golden.filename().string().c_str());
        return;
    }
    if (!std::filesystem::exists(golden)) Fail(golden.string() + " is missing; run with ORG_UPDATE_GOLDEN=1");
    if (ReadFile(golden) != dump) {
        auto actual = golden;
        actual += ".actual";
        std::ofstream(actual, std::ios::binary) << dump;
        Fail(golden.filename().string() + " differs; the new schedule is in " + actual.string());
    }
}
}

int main()
{
    const bool update = std::getenv("ORG_UPDATE_GOLDEN") != nullptr;
    const std::filesystem::path fixtures = ORG_REPLAY_FIXTURE_DIR;
    size_t checked = 0;
    for (const auto& entry : std::filesystem::directory_iterator(fixtures)) {
        const auto extension = entry.path().extension();
        if (extension == ".orggraph") {
            std::ifstream in(entry.path(), std::ios::binary);
            CheckGolden(entry.path(), DumpSchedule(ReadGraphReplay(in)), update);
            ++checked;
        } else if (extension == ".orgsequence") {
            std::ifstream in(entry.path(), std::ios::binary);
            CheckGolden(entry.path(), DumpSchedule(ReadGraphReplaySequence(in).initial), update);
            ++checked;
        }
    }
    if (checked == 0) Fail("no fixtures found in " + fixtures.string());
    std::printf("golden schedules: %zu fixtures %s\n", checked, update ? "updated" : "match");
    return 0;
}
