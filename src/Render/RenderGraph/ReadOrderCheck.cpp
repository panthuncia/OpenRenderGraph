// The read-order check across epoch graphs. See Render/RenderGraph/ReadOrderCheck.h.
#include "Render/RenderGraph/ReadOrderCheck.h"
#include "Render/RenderGraph/ScheduleTrace.h"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>

namespace org {

namespace {

std::string ResourceName(const ScheduleTrace& trace, uint32_t index) {
	if (index >= trace.resources.size()) return "#" + std::to_string(index);
	const auto& resource = trace.resources[index];
	return resource.name.empty() ? "#" + std::to_string(resource.id) : resource.name;
}

bool Holds(const std::function<bool(uint32_t)>& predicate, uint32_t position) { return predicate && predicate(position); }
bool Holds(const std::function<bool(std::string_view)>& predicate, std::string_view name) { return predicate && predicate(name); }

}  // namespace

void AppendReadOrderPasses(uint32_t position, const ScheduleTrace& trace, std::vector<ReadOrderPass>& out) {
	std::vector<const ScheduleTrace::Pass*> ordered;
	for (const auto& pass : trace.passes) ordered.push_back(&pass);
	std::ranges::stable_sort(ordered, [](const auto* left, const auto* right) {
		return std::tie(left->batch, left->position) < std::tie(right->batch, right->position);
	});
	for (const auto* pass : ordered) {
		ReadOrderPass entry{ .position = position, .name = pass->name };
		for (const auto& access : pass->accesses) {
			auto name = ResourceName(trace, access.resource);
			auto& list = access.write ? entry.writes : entry.reads;
			if (std::ranges::find(list, name) == list.end()) list.push_back(std::move(name));
		}
		out.push_back(std::move(entry));
	}
}

ReadOrderReport CheckReadOrder(std::span<const ReadOrderPass> passes, const ReadOrderInputs& inputs) {
	ReadOrderReport report;
	auto& findings = report.findings;
	// What each position's passes write.
	std::map<uint32_t, std::set<std::string>> positionWrites;
	for (const auto& pass : passes)
		for (const auto& name : pass.writes) positionWrites[pass.position].insert(name);
	// A transient read with no writer in its own position: a writer absent from the build while its reader is present.
	std::set<std::string> unwritten;
	for (uint32_t i = 0; i < passes.size(); ++i) {
		const auto& pass = passes[i];
		if (!Holds(inputs.requiresLocalWriter, pass.position)) continue;
		for (const auto& name : pass.reads)
			if (!positionWrites[pass.position].contains(name) && Holds(inputs.transient, name) && unwritten.insert(name).second)
				findings.push_back({ .kind = ReadOrderFinding::Kind::UnwrittenTransient, .pass = i, .resource = name });
	}
	// The first pass that writes each resource.
	std::map<std::string, uint32_t, std::less<>> firstWriter;
	for (uint32_t i = 0; i < passes.size(); ++i)
		for (const auto& name : passes[i].writes) firstWriter.try_emplace(name, i);
	std::set<std::string> previous, outside, found, laterDeclared;
	for (uint32_t i = 0; i < passes.size(); ++i)
		for (const auto& name : passes[i].reads) {
			const auto writer = firstWriter.find(name);
			if (writer == firstWriter.end()) {
				outside.insert(name);
				continue;
			}
			if (passes[writer->second].position <= passes[i].position) continue;
			if (Holds(inputs.previousFrame, name)) {
				previous.insert(name);
				continue;
			}
			if (Holds(inputs.allowsLaterWriter, passes[i].position)) {
				laterDeclared.insert(name);
				continue;
			}
			if (!found.insert(name).second) continue;
			findings.push_back({ .kind = ReadOrderFinding::Kind::LaterWriter, .pass = i, .writer = writer->second, .resource = name });
		}
	// Frame transients against their derived spans: a pass outside the span uses bytes another resource holds. The first
	// pass that uses one must write it, since the bytes hold another resource's data.
	std::set<std::string> spanFindings, seen;
	for (uint32_t i = 0; i < passes.size(); ++i) {
		const auto& pass = passes[i];
		const auto check = [&](const std::string& name, bool write) {
			if (!inputs.frameTransient) return;
			const auto transient = inputs.frameTransient(name);
			if (!transient) return;
			const auto span = transient->span;
			if ((span.Empty() || pass.position < span.first || pass.position > span.last) && spanFindings.insert(name + "@" + pass.name).second)
				findings.push_back({ .kind = ReadOrderFinding::Kind::OutsideSpan, .pass = i, .resource = name, .transient = *transient });
			if (seen.insert(name).second && !write)
				findings.push_back({ .kind = ReadOrderFinding::Kind::FirstUseReads, .pass = i, .resource = name, .transient = *transient });
		};
		for (const auto& name : pass.writes) check(name, true);
		for (const auto& name : pass.reads) check(name, std::ranges::find(pass.writes, name) != pass.writes.end());
	}
	report.previousInputs = static_cast<uint32_t>(previous.size());
	report.outsideInputs = static_cast<uint32_t>(outside.size());
	report.laterDeclared = static_cast<uint32_t>(laterDeclared.size());
	return report;
}

}  // namespace org
