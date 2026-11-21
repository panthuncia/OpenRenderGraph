// Run gates (RunGate) and first-run work (FirstRun): the latch words and open bits a run's commit writes, and their
// copy from a staging run. Needs a runtime with a frame arena (Runtime::Current).

#include "OpenRenderGraph/Runtime.h"
#include "Render/RunGate.h"
#include "TestSupport.h"

#include <cstdint>
#include <cstring>
#include <string>

namespace {
std::uint32_t Word(const org::RunGate& gate, const org::RunRef& run, std::uint32_t entry, std::uint32_t word) {
	std::uint32_t value = 0;
	std::memcpy(&value, gate.Latch().Bytes(run).data() + gate.Offset(entry) + word * 4, sizeof(value));
	return value;
}

struct DerivedGate final : org::RunGate {
	using org::RunGate::RunGate;
	int extra = 5;
};
}

int TestRunGates() {
	CHECK(org::Runtime::Current());
	std::string error;
	CHECK(!org::RunGate::Create("test.gate.empty", 0, &error) && error == "no entries");
	auto derived = org::RunGate::Create<DerivedGate>("test.gate.derived", 1);
	CHECK(derived && derived->extra == 5 && derived->Entries() == 1);

	auto gate = org::RunGate::Create("test.gate", 70);
	CHECK(gate && gate->Entries() == 70);
	const auto run = org::RunRef::Unleased(100);
	const auto other = org::RunRef::Unleased(101);
	// A run starts closed.
	CHECK(!gate->IsOpen(run, 0) && Word(*gate, run, 0, 0) == 0);
	// Out of range: the gate cannot tell, so it says open.
	CHECK(gate->IsOpen(run, 70));

	gate->OpenDispatch(run, 0, 4, 2);
	CHECK(gate->IsOpen(run, 0));
	CHECK(Word(*gate, run, 0, 0) == 4 && Word(*gate, run, 0, 1) == 2 && Word(*gate, run, 0, 2) == 1 && Word(*gate, run, 0, 3) == 0);
	gate->OpenDraw(run, 65);
	CHECK(gate->IsOpen(run, 65) && Word(*gate, run, 65, 0) == 3 && Word(*gate, run, 65, 1) == 1);
	// No work: the words are written, the entry is not open.
	gate->OpenDispatch(run, 1, 0);
	CHECK(!gate->IsOpen(run, 1));
	gate->OpenDispatch(run, 65, 0);
	CHECK(!gate->IsOpen(run, 65) && Word(*gate, run, 65, 0) == 0);
	CHECK(!gate->IsOpen(other, 0));

	// A staging run's open bits move onto the run it submits.
	org::RunGate::CopyOpenAll(run, other);
	CHECK(gate->IsOpen(other, 0) && !gate->IsOpen(other, 65));
	org::RunGate::ClearOpenAll(run);
	CHECK(!gate->IsOpen(run, 0) && gate->IsOpen(other, 0));
	// Close zeroes the words and the bits.
	gate->OpenDispatch(run, 0, 1);
	gate->Close(run);
	CHECK(!gate->IsOpen(run, 0) && Word(*gate, run, 0, 0) == 0 && Word(*gate, run, 0, 1) == 0);
	gate->Close(other);
	CHECK(!gate->IsOpen(other, 0));

	// First-run work opens its entry until a run that opened it executes.
	org::FirstRun first;
	CHECK(first.Pending());
	CHECK(!first.Executed());  // no commit opened it yet
	CHECK(first.Pending());
	first.Open(*gate, run, 2);
	CHECK(gate->IsOpen(run, 2) && Word(*gate, run, 2, 0) == 1);
	CHECK(first.Executed() && !first.Pending());
	gate->Close(run);
	first.Open(*gate, run, 2);
	CHECK(!gate->IsOpen(run, 2) && !first.Executed());
	gate->Close(run);
	return 0;
}
