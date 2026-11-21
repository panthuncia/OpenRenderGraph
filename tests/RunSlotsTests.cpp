// Per-run values (RunSlots) and their move from a staging run onto the run it submits (RunSlotsRegistry::MoveAll).
// No device.

#include "Render/RunSlots.h"
#include "TestSupport.h"

#include <cstdio>
#include <memory>

namespace {
using org::RunRef;

// A value is found and taken only by its own run, not by a later run of the same lease index.
int TestStoreFindTake() {
	org::RunSlots<int> slots;
	const RunRef run{3, 1};
	const RunRef later{3, 2};
	CHECK(!slots.Find(run));
	slots.Store(run, std::make_shared<const int>(7));
	CHECK(!slots.Find(later) && !slots.Take(later));
	CHECK(slots.Find(run) && *slots.Find(run) == 7);
	CHECK(slots.Find(run) && *slots.Find(run) == 7);  // Find leaves it
	const auto taken = slots.Take(run);
	CHECK(taken && *taken == 7);
	CHECK(!slots.Take(run) && !slots.Find(run));
	// A later run replaces the index's value.
	slots.Store(run, std::make_shared<const int>(1));
	slots.Store(later, std::make_shared<const int>(2));
	CHECK(!slots.Find(run));
	CHECK(slots.Find(later) && *slots.Find(later) == 2);
	slots.Clear();
	CHECK(!slots.Find(later));
	return 0;
}

// MoveAll moves every live RunSlots' value of the staging run onto the target run, and only from the staging run.
int TestMoveAll() {
	org::RunSlots<int> a;
	org::RunSlots<float> b;
	const RunRef staging = RunRef::Unleased(40);
	const RunRef target{5, 9};
	a.Store(staging, std::make_shared<const int>(11));
	b.Store(RunRef{40, 3}, std::make_shared<const float>(2.5f));  // another run at the staging index: not moved
	org::RunSlotsRegistry::MoveAll(staging, target);
	CHECK(!a.Find(staging));
	CHECK(a.Find(target) && *a.Find(target) == 11);
	CHECK(!b.Find(target));
	CHECK(b.Find(RunRef{40, 3}));
	{
		// A RunSlots removes itself when destroyed, so a later MoveAll does not reach it.
		org::RunSlots<int> gone;
		gone.Store(staging, std::make_shared<const int>(1));
	}
	org::RunSlotsRegistry::MoveAll(staging, target);
	CHECK(a.Find(target) && *a.Find(target) == 11);  // nothing at the staging run: the target keeps its value
	return 0;
}
}

int main() {
	if (const auto failure = TestStoreFindTake()) return failure;
	if (const auto failure = TestMoveAll()) return failure;
	std::printf("RunSlots tests passed\n");
	return 0;
}
