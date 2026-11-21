#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace org {

struct ScheduleTrace;

// A read-order check across a host's epoch graphs, run in the host's epoch order (GraphHost::Desc::epochOrder).
//
// Each epoch's graph orders its own passes, so a read inside one epoch always follows its writer. Between epochs nothing
// orders them but the epoch order itself. The check lists every epoch's passes in that order and reports:
// - a read whose first writer is a later epoch, unless the resource is the previous frame's;
// - a transient read with no writer in its own epoch, in the epochs that require one, since a writer absent from the
//   build leaves the reader with whatever the memory held;
// - a use of a frame transient outside its derived span, or a first use that reads it, since its bytes hold another
//   resource's data there.
// It is plain data in and out: the host gives it the traces and its predicates, and formats the findings itself.

// One pass of an epoch's schedule.
struct ReadOrderPass {
	uint32_t position = 0;  // the epoch's index in the epoch order
	std::string name;
	std::vector<std::string> reads, writes;  // resource names, each once, in access order
};

// Appends trace's passes in plan order (by batch, then position), at `position` in the epoch order. A resource
// without a name is named "#<id>".
void AppendReadOrderPasses(uint32_t position, const ScheduleTrace& trace, std::vector<ReadOrderPass>& out);

// Epoch positions, first to last, inclusive. Empty when first > last.
struct ReadOrderSpan {
	uint32_t first = 0, last = 0;
	bool Empty() const { return first > last; }
};

// A frame transient: the group its memory is placed with, and the positions its memory is its own.
struct ReadOrderTransient {
	std::string group;
	ReadOrderSpan span;
};

// The host's predicates. Each may be empty, which reads as false (or no frame transient).
struct ReadOrderInputs {
	// Whether a transient read at this position needs a writer in the same position.
	std::function<bool(uint32_t position)> requiresLocalWriter;
	// Whether reads at this position may name resources only later positions write, such as outputs an epoch declares
	// for later epochs to fill. They are counted, not reported.
	std::function<bool(uint32_t position)> allowsLaterWriter;
	// Whether a resource is read from the previous frame, so a later writer is expected.
	std::function<bool(std::string_view name)> previousFrame;
	// Whether a resource is transient within a frame.
	std::function<bool(std::string_view name)> transient;
	// A frame transient's group and span, or none for any other resource.
	std::function<std::optional<ReadOrderTransient>(std::string_view name)> frameTransient;
};

struct ReadOrderFinding {
	enum class Kind : uint8_t {
		UnwrittenTransient,  // pass reads a transient that no pass of its position writes
		LaterWriter,         // pass reads a resource whose first writer (writer) is at a later position
		OutsideSpan,         // pass uses a frame transient outside its span
		FirstUseReads,       // pass is the frame transient's first use, and reads it
	};
	Kind kind = Kind::LaterWriter;
	uint32_t pass = 0;              // index into the checked passes
	uint32_t writer = UINT32_MAX;   // LaterWriter: the first writer's index
	std::string resource;
	ReadOrderTransient transient;   // OutsideSpan and FirstUseReads
};

struct ReadOrderReport {
	std::vector<ReadOrderFinding> findings;  // unwritten transients, then later writers, then frame transients, each in pass order
	uint32_t previousInputs = 0;  // distinct resources read from the previous frame
	uint32_t outsideInputs = 0;   // distinct resources read that no pass writes
	uint32_t laterDeclared = 0;   // distinct resources read where allowsLaterWriter holds and only later positions write
};

// Checks passes, listed in epoch order. Each finding is reported once: per resource for unwritten transients, later
// writers and first uses, per resource and pass for uses outside a span.
ReadOrderReport CheckReadOrder(std::span<const ReadOrderPass> passes, const ReadOrderInputs& inputs);

}  // namespace org
