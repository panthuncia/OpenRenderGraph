// The read-order check across epoch graphs (Render/RenderGraph/ReadOrderCheck.h). No device.
#include "Render/RenderGraph/ReadOrderCheck.h"
#include "Render/RenderGraph/ScheduleTrace.h"
#include "TestSupport.h"

#include <string>
#include <vector>

using namespace org;

namespace {

ReadOrderPass Pass(uint32_t position, std::string name, std::vector<std::string> reads, std::vector<std::string> writes) {
	return { position, std::move(name), std::move(reads), std::move(writes) };
}

// Plan order is batch then position, and each resource is listed once per access kind.
int TestAppendOrder() {
	ScheduleTrace trace;
	trace.resources = { { "colour", 1, true }, { "", 7, false } };
	ScheduleTrace::Pass late;
	late.name = "late";
	late.batch = 1;
	late.position = 0;
	late.accesses = { { 0, false }, { 0, false }, { 1, true } };
	ScheduleTrace::Pass early;
	early.name = "early";
	early.batch = 0;
	early.position = 3;
	early.accesses = { { 0, true }, { 5, false } };
	trace.passes = { late, early };
	std::vector<ReadOrderPass> passes;
	AppendReadOrderPasses(2, trace, passes);
	CHECK(passes.size() == 2);
	CHECK(passes[0].name == "early" && passes[0].position == 2);
	CHECK(passes[0].writes == std::vector<std::string>{ "colour" });
	CHECK(passes[0].reads == std::vector<std::string>{ "#5" });
	CHECK(passes[1].name == "late");
	CHECK(passes[1].reads == std::vector<std::string>{ "colour" });
	CHECK(passes[1].writes == std::vector<std::string>{ "#7" });
	return 0;
}

// A read whose first writer comes later is a finding, unless it is the previous frame's or the reader's position
// allows it. A read nothing writes comes from outside.
int TestLaterWriters() {
	const std::vector<ReadOrderPass> passes = {
		Pass(0, "a", { "history", "late", "late", "outside" }, {}),
		Pass(1, "b", { "declared" }, { "x" }),
		Pass(2, "c", { "x" }, { "history", "late", "declared" }),
	};
	ReadOrderInputs inputs;
	inputs.previousFrame = [](std::string_view name) { return name == "history"; };
	inputs.allowsLaterWriter = [](uint32_t position) { return position == 1; };
	const auto report = CheckReadOrder(passes, inputs);
	CHECK(report.findings.size() == 1);
	CHECK(report.findings[0].kind == ReadOrderFinding::Kind::LaterWriter);
	CHECK(report.findings[0].pass == 0 && report.findings[0].writer == 2 && report.findings[0].resource == "late");
	CHECK(report.previousInputs == 1 && report.outsideInputs == 1 && report.laterDeclared == 1);
	return 0;
}

// A transient read needs a writer in its own position where the host asks for one. Each resource is reported once.
int TestUnwrittenTransients() {
	const std::vector<ReadOrderPass> passes = {
		Pass(0, "w", {}, { "t" }),
		Pass(1, "r1", { "t", "u" }, {}),
		Pass(1, "r2", { "t" }, {}),
		Pass(2, "r3", { "t" }, {}),
	};
	ReadOrderInputs inputs;
	inputs.requiresLocalWriter = [](uint32_t position) { return position == 1; };
	inputs.transient = [](std::string_view name) { return name == "t"; };
	const auto report = CheckReadOrder(passes, inputs);
	CHECK(report.findings.size() == 1);
	CHECK(report.findings[0].kind == ReadOrderFinding::Kind::UnwrittenTransient);
	CHECK(report.findings[0].pass == 1 && report.findings[0].resource == "t");
	return 0;
}

// Frame transients: a use outside the span is reported per pass, and a first use that reads is reported once.
int TestFrameTransients() {
	const std::vector<ReadOrderPass> passes = {
		Pass(0, "early", { "f" }, {}),
		Pass(1, "inside", { "f" }, { "f" }),
		Pass(3, "late", { "f" }, {}),
		Pass(3, "late2", { "g" }, { "g" }),
	};
	ReadOrderInputs inputs;
	// "early" reads f before "inside" writes it, so f is the previous frame's here, to keep that finding out.
	inputs.previousFrame = [](std::string_view name) { return name == "f"; };
	inputs.frameTransient = [](std::string_view name) -> std::optional<ReadOrderTransient> {
		if (name == "f") return ReadOrderTransient{ "group", { 1, 2 } };
		if (name == "g") return ReadOrderTransient{ "unused", { 1, 0 } };
		return std::nullopt;
	};
	const auto report = CheckReadOrder(passes, inputs);
	using Kind = ReadOrderFinding::Kind;
	CHECK(report.findings.size() == 4);
	CHECK(report.findings[0].kind == Kind::OutsideSpan && report.findings[0].pass == 0 && report.findings[0].transient.group == "group");
	CHECK(report.findings[1].kind == Kind::FirstUseReads && report.findings[1].pass == 0);
	CHECK(report.findings[2].kind == Kind::OutsideSpan && report.findings[2].pass == 2);
	// An empty span puts every use outside it. Its first use writes, so that is not a finding.
	CHECK(report.findings[3].kind == Kind::OutsideSpan && report.findings[3].pass == 3 && report.findings[3].resource == "g");
	CHECK(report.findings[3].transient.span.Empty());
	CHECK(report.previousInputs == 1);
	return 0;
}

// Findings come grouped: unwritten transients, later writers, then frame transients.
int TestFindingOrder() {
	const std::vector<ReadOrderPass> passes = {
		Pass(0, "a", { "later", "f" }, {}),
		Pass(1, "b", { "t" }, { "later" }),
	};
	ReadOrderInputs inputs;
	inputs.requiresLocalWriter = [](uint32_t) { return true; };
	inputs.transient = [](std::string_view name) { return name == "t"; };
	inputs.frameTransient = [](std::string_view name) -> std::optional<ReadOrderTransient> {
		if (name == "f") return ReadOrderTransient{ "group", { 0, 0 } };
		return std::nullopt;
	};
	const auto report = CheckReadOrder(passes, inputs);
	using Kind = ReadOrderFinding::Kind;
	CHECK(report.findings.size() == 3);
	CHECK(report.findings[0].kind == Kind::UnwrittenTransient && report.findings[0].pass == 1);
	CHECK(report.findings[1].kind == Kind::LaterWriter && report.findings[1].pass == 0);
	CHECK(report.findings[2].kind == Kind::FirstUseReads && report.findings[2].pass == 0);
	return 0;
}

}  // namespace

int main() {
	if (const int line = TestAppendOrder()) return line;
	if (const int line = TestLaterWriters()) return line;
	if (const int line = TestUnwrittenTransients()) return line;
	if (const int line = TestFrameTransients()) return line;
	if (const int line = TestFindingOrder()) return line;
	std::printf("ReadOrderCheck tests passed\n");
	return 0;
}
