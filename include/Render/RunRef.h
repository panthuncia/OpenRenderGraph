#pragma once

#include <cstdint>

namespace org {

// One run of an epoch: one submission, with everything it uses that must outlive its recording. A GraphHost opens a run
// lease for each run and retires it once the run's GPU work is done (GraphHost::CurrentRun, RecordingContext::Run).
//
// A run reference names the lease:
// - lease: the lease's index. Indices are dense and reused once a lease retires, so a table may be indexed by them.
// - generation: how many times the index has been opened. It tells two runs of one index apart. 0 means the index is not
//   a leased run: a published copy's run, or an execution outside a host.
// - copy: for a published recording (a revision-driven epoch's), the copy it is one of. Each copy is recorded once and
//   submitted many times, so what it reads must stay where the recording put it.
//
// Per-run memory (LatchBlock regions, CPU payloads) is keyed by the reference. A leased run's memory is freed when its lease
// retires. A stable run's (Stable) is kept for the index, because its recording is submitted again.
struct RunRef {
	static constexpr uint32_t kNoCopy = UINT32_MAX;

	uint32_t lease = 0;
	uint32_t generation = 0;
	uint32_t copy = kNoCopy;

	// A run of a lease the host opened for it alone.
	constexpr bool Leased() const noexcept { return generation != 0; }
	// A published copy's run.
	constexpr bool Published() const noexcept { return copy != kNoCopy; }
	// Memory keyed by this run stays at its index for good: a published copy's, or an index outside a lease.
	constexpr bool Stable() const noexcept { return Published() || !Leased(); }

	// An execution outside a host's leases (a graph executed on its own), at a frame index of the caller's.
	static constexpr RunRef Unleased(uint32_t index) noexcept { return RunRef{index, 0, kNoCopy}; }
	// The run of copy `copy` of a published recording, whose lease index is `lease`.
	static constexpr RunRef Copy(uint32_t lease, uint32_t copy) noexcept { return RunRef{lease, 0, copy}; }

	friend constexpr bool operator==(const RunRef&, const RunRef&) = default;
};

} // namespace org
