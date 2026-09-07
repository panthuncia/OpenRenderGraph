# Experimental asynchronous compilation

Status: dependency, conservative symbolic-schedule, and declared resource-state shadow compilation. This
is not yet an asynchronous execution path. Neither current compiled stage can be admitted to GPU
execution, and no benchmark result from this mode demonstrates that full frame
compilation is asynchronous.

## Scene-execution continuation (2026-09-05)

- Worker jobs and validation use the single owned-input `CompileGraph` entry point.
- Captured passes retain an explicit prepared-pass index; completed plans become immutable execution layouts without pointers into compiler scratch.
- Compile requests can carry an owned execution payload. Coalescing and out-of-order completion pair a reused structural plan with the newest matching backing leases and prepared payload.
- `RenderFrameSnapshot` owns the selected layout, single-use prepared passes, frozen resource bindings, and publication/backing leases. Publication fails if any pass remains legacy.
- Imported handle-only resources require a concrete-resource owner. This covers swapchain images without treating mutable dynamic wrappers as lifetime leases.
- Named frame-transient resources have collision-safe semantic compile slots, and helper-only dynamic identities are excluded from worker inputs.
- The radius-zero scene reaches the pass-migration gate. `ClearVisibilityBufferPass` is the first owned adapter; remaining scene passes still report explicit migration fallback, so async execution is not enabled yet and the legacy synchronous compile remains authoritative.

## Implemented boundary

`GraphCompileInput` owns all dependency-stage inputs. `CompileWorkspace` reads
only those values, builds hazard/API-ownership/explicit edges, topological order
and criticality, and returns an immutable `CompiledGraph`. It does not call any
pass, resource, registry, device, or publication interface. A workspace belongs
to one job. Publication leases are retained by requests and completed bundles;
reusing a structural graph does not pin its first content publication forever.

`GraphCompileCoordinator` uses the existing task service and has a single owner
for request/publication operations. Default concurrency is two (bounded 1–4),
with one replaceable pending request. Each worker writes its own completion
mailbox and publishes it with release/acquire ordering. Out-of-order completion
cannot regress the selected sequence. Coalescing compares complete owned
structural inputs, including registry generation, and preserves the newest
request's leases. Shutdown cooperatively cancels jobs and joins their scope
before the task service is destroyed.

The live graph retains its dependency oracle before aliasing, then captures
owned input after alias edges and active queue capabilities are prepared. The
worker independently reconstructs resource and explicit dependencies and adds
the captured alias-placement constraints. Dependency edges are compared against
the pre-alias oracle, not copied into the worker's resource dependency input.
The shadow
result has no authority over the live graph. Configuration is default-off. One
`experimentalAsyncCompileMode` setting selects `Off`, `Shadow`, or `Async` and
`experimentalCompileConcurrency` remains bounded 1-4. The startup override is
`SARP_ASYNC_COMPILE_MODE=Off|Shadow|Async`. Until scene execution migration is
complete, `Async` runs the same compiler/shadow capture and reports the exact
`SceneExecutionNotMigrated` synchronous fallback.

The first pure queue scheduler uses isolated single-pass batches and orders all
uses of each resource, including read/read. It chooses only active compatible
queues and emits relative producer-batch waits, coalesced per source queue.
It never allocates fence values or applies transitions. An independent validator
computes reachability through same-queue order and actual waits, checks every
live final-DAG constraint (including alias edges), and rejects missing, invalid,
or duplicate pass placements. This conservative schedule is not claimed to
match the legacy scheduler's batching efficiency or to be GPU-executable without
barrier, backing, and admission extraction.

Capture canonicalizes concrete resource enumeration and access ordering before
full-key comparison. Counters distinguish membership, pass/access, queue, and
constraint changes between retained consecutive requests. Normalization does
not erase concrete identity, queue policy, alias constraints, or pass order.

Resolver capture accepts an owned `ResolverCaptureContext`. The renderer
supplies its selected manifest lease before pass updates. Builders and retained
declaration capture use that same context. Published resolvers never reload
the process source in the explicit-context overload and return their locally
captured result, even if a clone concurrently replaces the shared cache.
Selection/fallback policy changes remain preparation-owner operations.

## Remaining side-effect extraction

| Existing operation | Required owner in the final architecture | Current status |
| --- | --- | --- |
| `Update`, declaration refresh, `Setup`, registry view changes | Preparation owner producing immutable pass packets | Still legacy mutable callbacks |
| Resolver expansion and resource metadata capture | Preparation with an explicit manifest lease | Explicit lease wired; backing metadata not yet frozen |
| Frame extensions and immediate command capture | Preparation, with owned single-consumption packets | Still stored on the live graph |
| Pass-access summaries and dependency graph | Independent compile workspace | Dependency stage extracted; live summaries remain mutable |
| Queue assignment and batching | Pure compiler using frozen capabilities | Conservative shadow schedule extracted; live optimized scheduler unchanged |
| Transition planning | Pure compiler with symbolic boundary states | Declared state progression extracted; backend barrier encoding remains live |
| Alias analysis and packing | Independent workspace producing symbolic placements | Still coupled to the live aliasing subsystem |
| Materialization and multi-backend representations | Resource realization dependency | Still performed by `CompileFrame` |
| Alias activation, entry barriers, cross-frame waits, absolute fence allocation | Ordered frame admission against submitted state | Still split between `CompileFrame` and `Execute` |
| Tracker publication and partial submission accounting | Execution runtime submission ledger | Existing synchronous implementation only |
| GPU allocation/descriptor retirement, temporal history advancement | Submitted execution instance and GPU retirement | Existing synchronous implementation only |

Next extraction must separate scheduling from `AddTransition` and physical
backing lookup; copying `RenderGraph`, invoking `CompileFrame` on another worker,
or locking the whole graph around compilation does not satisfy that boundary.
Prepared pass migration must cover all passes in the selected graph before
enabling asynchronous execution. A graph with legacy callbacks must report a
synchronous fallback. The renderer-level `RenderFrameSnapshot`, realization
leases and ordered `GraphExecutionInstance` are still required; dependency-only
shadow results must not be attached as executable frame snapshots.

## Validation

`OpenRenderGraphAsyncCompileTests` covers independent dependency-oracle checks,
cycles, cancellation, reversed completion, pending replacement, generation
invalidation, rejected work, retained publication lifetime and actual concurrent
compilation of a 1,000-resource / 60-pass graph with a ten-resource replacement.
`AsyncStateGraphTests` covers captures through clones against different explicit
manifest leases, including content-only publications and bootstrap.

Build through SARP's `build.cmd`. `SARP_SKIP_INSTALL=1` permits focused tests
without deploying an unrelated binary from a different preset.
`SARP_SKIP_CONFIGURE=1` is only for already configured, current build trees.
ASAN uses `SARP_PRESET_OVERRIDE=vs2026-renderer-host-asan`.

### Validation checkpoint: 2026-09-04

SARP's full `build.cmd` build and deployment succeeded. CTest now runs both
BasicRenderer and OpenRenderGraph tests with the SARP deployed renderer directory
as their working directory and first runtime DLL search path. The normal
`AsyncStateGraphTests`, `OpenRenderGraphExternalResourceTests`, and
`OpenRenderGraphAsyncCompileTests` passed; the new async compiler test also passed
under the ASAN preset. This is not full-renderer ASAN coverage.

An initial benchmark crashed before reporting. Dump inspection showed the
culling pass loading `hostData` at the old context offset 0x18 while the caller's
updated context stored it at 0x28: it read fence value 1 as a pointer. Rebuilding
the context consumers together removed that reproduced startup failure without
a synchronization or lifetime workaround.

Two subsequent radius-100 shadow runs used `--benchmark-auto-exit
--benchmark-stable-frames 300 --benchmark-timeout-ms 90000`. Both reported stable
and exited with code zero. BasicTelemetry reported:

| Measurement | Run 1 | Run 2 |
| --- | ---: | ---: |
| Completed dependency-oracle comparisons at report capture | 815 | 745 |
| Oracle mismatches / failed jobs | 0 / 0 | 0 / 0 |
| Peak concurrently running jobs | 2 | 2 |
| Peak active / pending jobs | 2 / 1 | 2 / 1 |
| Shadow input capture p95 | 0.260 ms | 0.204 ms |
| Synchronous CompileFrame p95 / max | 9.021 / 53.513 ms | 10.817 / 83.313 ms |
| Retained declaration Apply p95 / max | 0.788 / 30.812 ms | 0.844 / 77.032 ms |
| Peak accounted input and selected-result bytes | 232736 | 194564 |

These are whole-run distributions, not isolated steady-state measurements.
The byte counter excludes opaque resource leases and active worker scratch.
Background queue-delay maxima were 389 and 240 ms; executable latest-bundle
selection must account for that publication age. Neither run coalesced structural
requests. The remaining synchronous declaration spikes are not resolved by
dependency shadow compilation. Cleanup also reported live buffers/textures;
their ownership remains uninvestigated, so GPU retirement acceptance is open.

Reports are preserved in SARP's build directory as
`async-compile-shadow-run{1,2}-summary.json` and
`async-compile-shadow-run{1,2}-report.txt`.

The full plan's acceptance gates remain open until executable snapshot
selection, backing/descriptor leases, ordered admission, temporal and
single-consumption work semantics, and complete scheduling/alias/wait oracles
have been implemented and tested. Shadow dependency comparisons alone do not
validate those properties.

### Symbolic-schedule checkpoint: 2026-09-04

The worker now produces a conservative multi-queue schedule and relative waits.
Tests cover inactive/incompatible queues, missing and future waits, duplicate
placements, alias constraints/cycles, randomized dependency graphs, reversed
completion order, cancellation, generation changes, and two real concurrent
workers on the 1,000-resource/60-pass workload. Concrete resource ID zero is
valid and has a regression test. Capture exceptions are reported and contained
in shadow mode; they cannot abort the authoritative synchronous update.

Completed structural plans have an eight-entry generation-scoped cache with
full-key equality and no publication leases. Cache reuse, eviction, lease
release, generation invalidation, and failed jobs are tested. The retained-byte
counter includes cached structural plans but not opaque leases or worker scratch.

The corrected radius-100 symbolic-schedule run reached 300 stable frames and
auto-exited: 721 dependency and schedule comparisons, zero failures, and two
overlapping jobs. Capture p95 was 0.579 ms; worker scheduling p95 was 0.219 ms.
Whole-run synchronous CompileFrame p95/max remained 8.774/83.500 ms and
declaration Apply p95/max 0.640/76.502 ms. A subsequent cache-enabled run also
auto-exited with 770 checks and zero failures. It reported zero completed-cache
hits: 790 of 790 consecutive inputs actually changed concrete membership after
canonicalization. Queue capabilities did not change. Cache reuse in that scene
is therefore not demonstrated by the synthetic reuse tests.

A BasicTelemetry scope trace attributes its 79.140 ms Apply spike to frame 2:
53 full redeclarations, not a worker or coordinator wait. The largest compute
refresh was EvaluateMaterialGroupsPass at 5.668 ms, of which declaration itself
took 4.052 ms. Summary counters subsequently reported up to 50 empty-initial-set
fallbacks. Fluent builders previously inferred resolver access recipes from
expanded resources, losing the recipe when an initial selection was empty.
Builders now retain authored state/range even for empty sets. The new graph
regression exercises empty/populated/empty/populated transitions for direct and
identifier resolvers, requires no repeated declaration or view replacement,
checks compatibility Setup calls, and verifies content-only updates do not
call Setup. Multi-use empty recipes remain on the conservative full path.

Preserved evidence: `build/async-schedule-summary-run1.json`,
`build/async-schedule-trace-run1.sqlite`, and
`build/async-schedule-trace-run1-summary.json` in the SARP checkout.

The empty-recipe fix passed normal and ASAN graph tests, then two more radius-100
runs reached 300 stable frames and auto-exited. The summary run recorded 727
dependency/schedule checks, zero failures, and two overlapping jobs; Apply
p95/max fell to 0.052/2.645 ms. The detailed run recorded 742 checks, zero
failures, and Apply p95/max of 0.061/2.216 ms. Both reported zero incremental
patch fallbacks. These Apply numbers must not be confused with the entire
refresh cost: incremental patches execute inside CheckCandidates. Its detailed
run p95/max was still 3.311/14.739 ms. No combined preparation/admission target
is claimed.

The detailed run's maximum CompileFrame event (52.377 ms, frame 1) comprised
33.171 ms alias-plan construction and 11.767 ms materialization. A later
24.834 ms frame included 14.739 ms in CheckCandidates. These remain preparation
and realization extraction targets; moving only the existing dependency stage
off-thread does not resolve them. Evidence is in
`build/async-empty-recipe-{run1-summary.json,trace-summary.json,trace.sqlite}`.

An additional concrete-permission regression initially failed. Live debugger
inspection showed RequestHandle allowing numeric ID zero but then searching
only symbolic registry keys, throwing `Unknown resource: "0"`. The registry
now maintains a numeric reverse index and resolves numeric identifiers lazily.
Tests cover updated resolver permissions, denied access after removal, mixed
resolver collisions requiring the full merge path, expired anonymous slots,
slot reuse, and registry replacement. This compatibility fix does not recreate
decimal-string identifiers in pass declarations.

### Open host failure: 2026-09-05

The next normal host run crashed during loading with `0xC0000374`; it is not an
accepted benchmark. WER captured `SARPRendererHost.exe.40324.dmp`. Debugger stack
inspection locates detection at destruction of a static-import artifact's
`transactionBuild.drawRecordCounts` vector from `AsyncStateGraph::Impl::Drain`.
This identifies the failing free, not the original corrupting write. The dump
does not contain the relevant heap pages. No speculative resolver lifetime or
synchronization change was made. Full-host ASAN reproduction is the next
diagnostic gate. The ASAN build's obsolete cached zlib library names were
refreshed through `build.cmd` CMake arguments, without substituting another
preset's libraries. Crash analysis is preserved in
`build/async-final-crash-{analysis,vector,vector-fields}.txt`.

Full-host ASAN subsequently caught a separate shutdown use-after-free. An
indirect-command desired-state worker called RendererStateRequestService's
SubmitLatest after Renderer::Cleanup freed the service. The sanitizer records
both the read (RendererStateRequestService.cpp:42) and freeing stack
(Renderer::Cleanup). IndirectCommandBufferManager now exposes idempotent
Shutdown: stop scheduling, detach the mutation callback, join its preparation
scope, and clear borrowed service pointers. Renderer calls it while those
services remain alive. The destructor delegates to the same operation.

The launcher previously classified every exit code 3 as a timeout. That masked
post-report ASAN failures; classification now requires an actual timeout report.
Two earlier instrumented runs wrote stable reports but exited 3 and are not
accepted runs. A debugger run with Windows-heap interception was much slower
and genuinely timed out. The matched-settings debugger run caught the service
use-after-free during teardown. No evidence yet links that teardown defect to
the earlier normal-build heap corruption during loading. The ASAN report and
live stacks are in `build/async-host-asan-strict-live-debug.txt`.

After the shutdown fix, the matched-settings ASAN host reached 300 stable
frames, completed teardown, and exited zero without a sanitizer report.
This diagnostic used a 180-second limit and took 101.9 seconds wall-clock;
it does not meet the normal 90-second performance gate. The normal six test
targets and ASAN async compiler, external-resource/declaration, and
AsyncStateGraphTests targets passed. The successful ASAN report is preserved
under `build/async-shutdown-fix-asan-run1-*`. Cleanup still reported 228 live
buffers and three textures, so complete GPU retirement acceptance remains open.

### Final normal-build checkpoint: 2026-09-05

The normal host was rebuilt/deployed through `build.cmd` and its deployed hash
verified against the normal build. Two consecutive radius-100 runs with the
original 90-second limit reached 300 stable frames and exited zero:

| Whole-run measurement | Run 1 | Run 2 |
| --- | ---: | ---: |
| Dependency / schedule comparisons | 774 / 774 | 715 / 715 |
| Oracle / schedule / worker failures | 0 / 0 / 0 | 0 / 0 / 0 |
| Peak overlapping jobs | 2 | 2 |
| CompileFrame p95 / max | 9.455 / 54.327 ms | 11.519 / 53.968 ms |
| CheckCandidates p95 / max (includes incremental patches) | 3.293 / 15.793 ms | 3.986 / 15.118 ms |
| Apply p95 / max | 0.053 / 2.381 ms | 0.057 / 2.368 ms |
| Incremental resolver patch fallbacks | 0 | 0 |
| Largest alias-pool allocation | 23.417 ms | 23.662 ms |

Evidence is preserved as `build/async-shutdown-fix-normal-run{1,2}-*`. The second
run accounted for at most 950,268 retained input/structural-plan bytes (opaque
leases and worker scratch excluded). The successful retries do not establish
the cause of the earlier loading-time heap corruption, and that failure remains
open. This checkpoint proves the bounded concurrent shadow stages and fixes the
reproduced declaration/permission/shutdown defects; it does not prove immutable
GPU execution bundles, latest-graph frame admission, backing leases, or complete
resource retirement. Experimental GPU execution remains unavailable/default-off.

### Radius-zero ASAN checkpoint: 2026-09-05

Radius-zero iteration exposed a missing producer notification, not a new ASAN
memory error. Before the fix, the host timed out with 245 (and then 229)
completed static-cell results still queued. Live inspection found the static
import coordinator Idle, requested/drained epochs both 0x1001, no delayed wake,
zero rejected submissions, and eight already-queued worldspace-resident LOD4
results. LOD4 used a direct TryPush whereas ordinary cell producers used the
wake-notifying handoff helper. It now uses that same helper; no polling,
readiness relaxation, or additional lifetime hold was introduced. The debugger
evidence is `build/async-radius0-wake-debug.txt` in SARP. That diagnostic process
was intentionally terminated after capturing its state and thread stacks; it
is not an accepted benchmark run.

Two subsequent full-host ASAN radius-zero runs reached 300 stable frames,
drained the completed/preparation queues to zero, completed cleanup, and exited
zero without sanitizer reports. Both enabled `SARP_ASYNC_COMPILE_SHADOW=1`:

| Measurement | Run 1 | Run 2 |
| --- | ---: | ---: |
| Benchmark elapsed | 43.430 s | 37.444 s |
| Whole process | 73.802 s | 61.688 s |
| Timeout setting | 180 s | 90 s |
| Windows heap interception | Off | On |
| Dependency / schedule comparisons | 365 / 365 | 365 / 365 |
| Worker / oracle / schedule failures | 0 / 0 / 0 | 0 / 0 / 0 |
| Peak overlapping jobs | 1 | 1 |

Run 1 used `halt_on_error=1:abort_on_error=1:log_path=asan_radius0_fixed`.
Run 2 additionally used `windows_hook_rtl_allocators=true` and
`alloc_dealloc_mismatch=0`, with `log_path=asan_radius0_fixed_rtl`.
Reports, renderer/launch logs, and BasicTelemetry summaries are preserved as
`build/async-radius0-wake-fix-run{1,2}-*`. The radius-zero workload does not prove
two-job overlap; the synthetic concurrent test remains the coverage for that
case here. Radius limits static-cell streaming, not worldspace terrain and
distant LOD initialization. ASAN timings are not performance acceptance data.

Added coordinator lifetime tests cover outstanding closures plus a replaceable
pending input, generation reset, explicit repeated shutdown, destruction, and
publication-lease release. The async compiler suite passed 20 consecutive ASAN
runs. ASAN async compiler, external-resource, AsyncStateGraph, and streaming
queue tests passed. StreamingQueueTests now uses SARP's deployment directory for
its CTest working directory and DLL search path; without it the ASAN executable
failed DLL loading (0xc0000135), while the same executable passed with that path.
The seven focused normal test targets also passed. Both hosts were built through
`build.cmd`; the normal deployment was restored afterward and its executable
hash verified against the normal build. No diagnostic host/debugger was left running.

These runs do not establish the cause of the earlier normal-build loading-time
heap corruption. Live-buffer/texture cleanup warnings also remain. Full async
GPU execution and its realization/admission ownership extraction are still
unimplemented; the experiment remains a default-off shadow path.

### Symbolic-state checkpoint: 2026-09-05

Preparation now copies resource dimensions, layout capability, entry states and
declared transition exit states into the owned compile structure. Workers plan
state progression with independently owned subresource rectangles. Initial
states remain unresolved admission inputs; exit states are callback
postconditions, not additional generated barriers. No live resource tracker,
queue timeline, allocation, or pass callback is touched by the worker.

An independent dense-cell oracle validates rectangle steps and final states on
the worker. This checks captured declaration semantics, not equivalence with
the legacy backend's emitted barriers. The oracle has explicit two-million-cell
storage/work limits. Invalid ranges, conflicting overlapping declarations and
unsupported transition coverage produce a reported fallback with no partial
state plan. Dependency-only identities may have unknown dimensions provided
they carry no state declarations. Buffer layout is canonicalized as irrelevant;
access and synchronization remain part of full structural equality. Globally
read-only resources omitted from the legacy hazard DAG still participate in
state-use queue ordering.

Tests cover randomized subresource rectangles, exit states, deliberately damaged
oracle outputs, buffer versus texture layouts, missing hazard declarations,
normalization, generation invalidation, and state planning on two actual workers
using the 1,000-resource/60-pass workload. Seven focused normal tests and four
focused ASAN tests passed after builds through `build.cmd`. The standalone ASAN
compiler test additionally passed five consecutive runs. This checkpoint did
not repeat full-host ASAN; earlier full-host results above predate this stage.

The radius-100 run reached 300 stable frames and auto-exited with 755 dependency
and schedule comparisons, 754 state comparisons, zero validation/worker failures,
and two overlapping jobs. One request reported an invalid captured state range;
its cause remains to be investigated before executable admission. State planning
p95/max was 0.818/2.332 ms and worker oracle validation 6.119/18.804 ms. Capture
p95/max was 1.745/8.482 ms. ASAN tests ran concurrently, so these whole-run timings
are diagnostic, not performance acceptance. Evidence is preserved in SARP's
`build/async-symbolic-states-radius100-run1-*` files.

A second radius-100 run also auto-exited successfully at 300 stable frames:
774 dependency/schedule comparisons, 773 state comparisons, zero validation
failures, and the same single invalid-range fallback. Evidence is preserved as
`build/async-symbolic-states-radius100-run2-*`. Existing live-buffer/texture
cleanup warnings remain; successful exit does not establish complete retirement.

Next boundaries remain backing/descriptor leases, backend barrier policy,
owned recording packets, realization, and ordered admission against submitted
GPU state. These symbolic plans are not executable frame bundles. The earlier
loading-time heap corruption did not recur and was not investigated at this
checkpoint, following the user's requested scope.

### Execution timeline admission extraction

`ExecutionTimelineAdmission` now instantiates absolute per-execution signals
and normalized waits from an immutable bundle's relative schedule. It is an
owner-thread component, not a compile-worker operation. Preparation is
transactional: incompatible inputs, future incoming waits on owned timelines,
overflow, and invalid schedules consume no signal values. Each queue must have
an exclusive timeline. Incoming cross-frame/resource waits remain an explicit
input from the future backing-state ledger; they are not inferred here.

Submission accounting accepts batches only in admission order and advances
only successfully submitted signal values. Partial failure closes admission
while preserving committed values and the failed packet's leases for recovery.
Fully submitted packets stay retained until observed completion covers every
batch. Capacity defaults to three executions and is bounded at 64. Commit does
not allocate; capacity is reserved before submission. This owner is noncopyable.
The runtime must keep the owner alive through GPU retirement/recovery, and must
not interpret CPU submission completion as GPU completion.

Tests exercise two queues, repeated graph execution with fresh absolute values,
wait normalization, transactional rejection, partial failure, capacity pressure,
completion validation, and packet lifetime before/after retirement. Both normal
and ASAN focused compiler tests pass. This component is not connected to live
GPU submission: backend barriers, backing leases, recording packets, and the
resource/alias ownership ledger remain required. No host benchmark can yet
validate this new admission component because the renderer does not call it.

### Owned RHI submission prototype

`SubmitPrepared` now drives prepared batch submission, checks queue-slot parity,
commits successful signals in order, and enters recovery on failure without
submitting subsequent batches. Execution retention includes the packets, not
just the compile publication. Signal value UINT64_MAX is reserved, matching the
live executor's invalid-value rule.

`PreparedRhiExecutionBatch` implements actual RHI wait/submit/signal operations
on closed command lists. Packets are single-consumption even after failure and
retain an explicit ownership lease. The preparation owner must provide ownership
of concrete backing, descriptors, allocators, device/queue and timelines; this
interface cannot turn a mutable Resource reference into a backing lease.

The external-resource test uses D3D12 WARP with an independently compiled,
two-pass graph. It uploads into a default-heap buffer, encodes the compiler's
inter-pass CopyDest-to-CopySource state step as an RHI buffer barrier, copies to
readback, submits both packets, waits for GPU completion and verifies all 4096
bytes. It reuses the compiled structure with fresh concrete backing and changed
upload data for a second execution. Concrete allocations, imported handles,
allocators and lists belong to the test's owned lease; device and timeline owners
outlive the admission owner. This is actual backend submission/readback, not a
mock, but WARP is software execution rather than hardware-GPU coverage.

Production backing-version leases, general texture/import barrier policies,
alias ownership, cross-frame state-ledger resolution and migration of mutable
scene passes remain unimplemented. The prototype's initial backing states are
known from allocation; it does not validate those general admission problems.
The scene renderer remains shadow-only and no scene benchmark result validates
this submission prototype. Mock failure tests additionally verify that a failed
middle packet stops submission and preserves already committed values.

Validation: builds exclusively through `build.cmd`; seven focused normal tests
and four focused ASAN tests passed. After extending the readback test to two
passes, it passed normally and five consecutive times under ASAN. No full-host
benchmark or hardware-GPU validation was performed for this prototype.

### Scene allocation ownership integration

`TrackedHandle::CaptureAllocationLease` now lazily transfers concrete allocation
and tracking-token ownership to a stable shared holder. Logical reset, backing
replacement and deferred-deletion handoff can release their reference without
invalidating a captured allocation. Capture/mutation remain preparation-owner
operations; workers only retain the immutable lease. Disarming a holder with
outstanding leases is explicitly rejected. No lease is allocated on the normal
path unless capture is requested.

Ordinary BufferBase and PixelBuffer backings expose allocation snapshots with
logical resource ID, backing generation, API resource and ownership lease.
Aliased and attached representations are unsupported: keeping a placed resource
alive does not reserve its physical alias range. Descriptor slots are also not
covered by this allocation-only contract. Scene shadow preparation captures
supported allocation leases and includes backing generation in structural keys.
Telemetry distinguishes captured leases from unsupported backed resources.

This wires production allocation ownership into shadow requests, not scene GPU
execution. Descriptor-version ownership, imported/alias leases, prepared scene
pass recording, and the cross-frame state ledger remain mandatory gates. The
readback regression drops each original TrackedHandle before recording and
submission; only captured allocation ownership keeps its RHI resource alive.

Validation checkpoint: normal focused tests passed 7/7 and ASAN focused tests
4/4; the backing-generation cache regression passed in both presets. The host
was rebuilt/deployed using `build.cmd`, with matching executable hashes. A
normal radius-zero shadow benchmark reached 300 stable frames in 19.679 seconds
and exited zero. It reported 181,210 allocation-lease captures, 27,286 unsupported
backed-resource encounters (counts across frames, not unique resources), 362
dependency/schedule comparisons, 361 state comparisons, zero worker/oracle/state
failures and two overlapping jobs. Per-resource capture p95/max was
0.0007/1.3606 ms; this is not aggregate preparation timing. Evidence is preserved
as `build/async-scene-leases-radius0-{report.txt,renderer.log,summary.json}` in
SARP. The full host was not run under ASAN at this checkpoint. Scene GPU
execution remains disabled; this benchmark validates allocation capture and
retention in the live shadow path, not owned scene recording or submission.

### Owned scene buffer-copy recording

In any non-Off async compile mode, the first scene recording migration is enabled.
After materialization, the execution owner captures descriptor-free immediate
buffer-copy bytecode into `PreparedBufferCopies`. The packet contains concrete
handles, immutable operands and allocation leases, validates byte ranges without
overflow, and preserves repeated-destination write ordering. Unsupported opcodes,
unknown sizes, imported/aliased backing and non-primary backends remain on the
legacy path. Capture never consumes the source bytecode. Recording uses no
registry, resolver, resource wrapper or host-data pointer, and rejects replay.

Both legacy recording modes consume supported packets. On successful submission
the packet is handed to the existing deferred-release owner against the just-
published actual queue-fence snapshot, not CPU frame count. Failed execution
keeps its frame packet for the existing recovery/device-idle cleanup path.
Telemetry counts prepared, recorded and unsupported scene copy passes.

This option changes recording of supported scene copies, not graph selection:
legacy compilation still supplies scheduling, barriers and cross-frame states.
It must not be described as full asynchronous scene GPU execution. Remaining
scene passes, descriptor versions, alias/import ownership, and executable
latest-bundle selection/admission are still open. The WARP readback regression
records two copies to the same destination from a prepared packet after dropping
the original allocation owners and bytecode, then verifies all copied bytes.
It additionally checks replay rejection, unsupported operations and overflowed
copy ranges.

Validation: normal focused tests passed 7/7 and ASAN focused tests 4/4, with five
additional consecutive ASAN readback tests. The radius-zero scene run recorded
1,278 owned copy-pass executions and three unsupported encounters, reached 300
stable frames and exited zero. Copy preparation p95/max was 0.114/0.334 ms.
The radius-100 run recorded 3,257 owned copy-pass executions, five unsupported
encounters, 796 dependency/schedule checks, 795 state checks, zero failures and
two overlapping shadow jobs. It reached 300 stable frames and exited zero in
44.213 seconds. Copy preparation p95/max was 0.335/8.146 ms; the maximum remains
above the operation spike target and is not attributed here. Evidence is in
`build/async-scene-copies-radius{0,100}-{report.txt,renderer.log,summary.json}`.
These runs preceded the exception-safety refinement retaining frame ownership
until deferred-retirement registration succeeds. They are not evidence of full
async graph execution or full-host ASAN coverage.

After the retirement refinement, normal tests passed 7/7 and focused ASAN tests
4/4 again. The rebuilt/deployed host (hash verified) completed another radius-zero
run: 300 stable frames, 1,312 recorded packets, three fallback encounters, zero
failures, clean exit. Evidence is `build/async-scene-copies-radius0-final-*`.

### Shared compiler helpers

The live and owned-input routes now use `CompilerAlgorithms.h` for hazard/API
ordering, deterministic topological sorting and criticality. Callers provide
compiler-owned views/accessors and private reusable scratch; the helpers have
no device, registry, resource or mutable pass dependencies. The live route folds
API ownership into its resource-access traversal, removing the second traversal
and its per-frame temporary arrays. The owned workspace preserves reader-list
capacity across builds. Queue assignment, alias planning and realization are
not duplicated or moved by this extraction.

Sharing the algorithms changes the meaning of the live/shadow comparison: it
now verifies captured-input parity for these stages, not independent algorithm
implementations. The quadratic hazard oracle remains independent in tests.
An additional randomized oracle selects ready nodes by linear search and computes
longest paths recursively, comparing both size_t and uint32_t helper views,
including tied authored order and cancellation/reuse. Existing cycle, invalid
input, generation and concurrent-workspace tests remain.

`MakeWholeBufferBarrier` is likewise shared by live transition recording, legacy
immediate replay, prepared copy recording and the owned GPU readback test.
Resource-specific texture/import encoding remains under its existing owner;
this helper does not pretend a general texture barrier can ignore that policy.
Normal focused tests passed 7/7 and ASAN focused tests 4/4 after extraction.
Full async scene execution remains incomplete; further work should extract
existing compiler stages through owned interfaces rather than build a second
independent compiler implementation.

The rebuilt/deployed radius-100 host reached 300 stable frames and exited zero
in 32.170 seconds with both experimental flags enabled: 790 dependency/schedule
checks, 789 state checks, zero failures, two overlapping compile jobs and 3,360
owned scene-copy recordings (five fallback encounters). Live dependency
construction p95/max was 0.172/2.228 ms; worker dependency construction was
0.675/2.328 ms and worker topology 0.175/5.514 ms. These whole-run measurements
are not full async execution or performance acceptance. Evidence is preserved as
`build/async-shared-helpers-radius100-{report.txt,renderer.log,summary.json}`.

### Typed preparation and submission receipts

`PreparedPass.h` introduces owned typed data with a non-capturing recording
function. Packet copies share single-consumption state, including after a
recording failure. `RecordingContext` exposes command recording and checked
resource/descriptor slots from owned frozen bindings, not live registry,
manager, tracker, or RenderContext access. The base graphics/compute/copy pass
interfaces provide preparation-owner `PrepareFrame`; an empty result explicitly
means unsupported legacy preparation. These hooks are not yet dispatched for
ordinary scene passes. Data providers must own their snapshots; the type-erased
adapter cannot prove that arbitrary user data contains no borrowed pointers.

Prepared RHI submission now returns a receipt distinguishing no command
submission, uncertain submission, submitted-but-unsignaled work, and confirmed
signals, with the failure stage and backend result. Admission preserves the
first failed batch and retains pending ownership without advancing an
unconfirmed completion value. Recovery integration remains required before
using this prototype for scene submission.

The D3D12 copy/readback test records through the typed adapter and verifies
slot bounds and replay rejection. An inert queue injects wait, submit, and
signal failures and verifies receipts and call ordering without executing GPU
work. Descriptor heap versioning, execution-instance ownership assembly and
scene pass migration remain outstanding; these interfaces alone do not make
legacy descriptors or live renderer data immutable.

This checkpoint passed normal tests 7/7 and focused ASAN tests 4/4. The ASAN
streaming test initially took 59.96 seconds during the host rebuild, then passed
in 0.29 seconds on repeat; the delay is not attributed. It exited before a
debugger could attach. Normal host build/deployment also succeeded. The next
radius-zero launch waited in MO2 without starting a renderer; its Qt event-loop
stacks are retained in `build/async-owned-interface-mo2-stacks.txt`. Do not count
the older report on disk as validation of this checkpoint.
The full ASAN host subsequently built/deployed successfully through `build.cmd`;
host execution remains unvalidated while the MO2 handoff is pending.

### Versioned descriptor snapshot foundation

The retried full-host ASAN radius-zero launch completed with 300 stable frames,
exit code zero, and no new ASAN report. Telemetry recorded 210 dependency and
schedule comparisons, 209 state comparisons, zero failures, two overlapping
compile jobs, and 1,422 owned scene-copy recordings. This validates the existing
shadow/copy route, not full async scene selection. Evidence is preserved in
`build/async-descriptor-radius0-asan-{report.txt,renderer.log,summary.json}`.

`DescriptorSnapshots.h` adds an admission-owner pool bounded by execution slots,
with immutable owned descriptor recipes and a persistent paged dirty-slot journal.
Logical indices are unchanged. Unchanged captures are cached; each dirty page
is copied on write, and heap reuse compares captured pages/slot recipes rather
than applying the latest publication indiscriminately. Older selected captures
therefore reproduce their own descriptor contents. Each returned snapshot owns
its device and descriptor backing recipes. Its shared ownership must be retained
through recording and actual GPU completion; an outstanding snapshot prevents
that execution slot from being overwritten, without reserving alias ranges.

Initial population writes populated slots only. Later reuse updates changed
slots only. Removal denies slot resolution and conservatively retains the old
physical slot's backing until overwrite or heap destruction, bounded by capacity.
Allocation/population are separate BasicTelemetry zones. A failed candidate
population discards only its unleased heap; other selected heaps survive.

This is independently tested infrastructure, not yet attached to the legacy
DescriptorHeapManager producers or scene frame snapshots. Shader-visible and
CPU-view heap pools share the mechanism. Integration must capture concrete
backing ownership at every descriptor producer, preserve published GPU table
versions, group all heap kinds transactionally into a realized bundle, and
register the returned snapshot owners with execution retirement. No existing
mutable descriptor heap is declared safe for async scene recording by this work.
After adding historical capture, CPU-view, busy-slot, generation, population and
allocation-failure coverage, normal tests passed 7/7 and focused ASAN tests 4/4.
Both builds used root `build.cmd`. The new pool is header-only and not yet used
by the host; the host run above validates the preceding scene implementation.

### Owned descriptor recording and retirement

`RecordPreparedRhiExecutionBatch` now transfers frozen binding tables, typed pass
data, command allocators and lists into the submission packet's retirement
ownership. It validates every packet before consuming any pass, calls only
owned recording functions, and is callable on a recording worker. Descriptor
snapshots retained by the bindings therefore cannot be overwritten between
recording and actual GPU retirement. Queue/device/timeline ownership remains an
explicit runtime input; declaration and Setup callbacks are never invoked.

A real D3D12 test clears a UAV using matching shader-visible and CPU descriptor
snapshots, records its compiler-produced transition, copies to readback, and
checks all 1,024 words. Two executions reuse the same compiled structure with
fresh backing and clear values. After caller references are dropped, admission
alone retains recording packets/bindings. Both descriptor pools remain busy
until actual fence completion is passed to retirement; then bindings expire
and the slots become reusable. Recording runs on a worker.

BasicRHI command-list ABI 6 appends checked close support after the version
field, preserving existing table offsets. D3D12 reports Close's result; Vulkan
reports pending recording errors and vkEndCommandBuffer's result. Existing
void End calls use the same implementation but retain their historical API.
Owned recording requires EndChecked capability and fails before returning a
submission packet when close fails. A deliberately already-closed D3D12 list
tests this path without submitting invalid work; ABI-5 tables report unsupported.

This still is not scene migration: descriptor producers, published table
versions, complete realized bundles, submission-ledger alias hazards and scene
selection remain to be connected. The helper does not make borrowed legacy
callbacks safe or supply a general retry policy for single-consumption work.

Validation after checked-close integration: normal tests 7/7, focused ASAN tests
4/4. The ASAN host built/deployed through root `build.cmd` with matching executable
hashes, then completed radius zero at 300 stable frames and exited zero. It
reported 217 dependency/schedule checks, 216 state checks, zero failures, two
overlapping compile jobs and 1,421 owned scene-copy recordings. No new ASAN
report was generated. Evidence is
`build/async-recording-descriptors-radius0-asan-{report.txt,renderer.log,summary.json}`.
Vulkan checked-close code was updated, but this GPU validation uses D3D12/WARP
and the D3D12 scene; it does not establish Vulkan runtime coverage.

### Alias heap generations and realization identity

Alias materialization no longer passes borrowed allocator pointers. Each
persistent alias pool publishes an `AliasHeapGeneration` owning the tracked
allocation and generation number; placements hold that shared owner plus their
offset. Placed resource handles retain the generation through their own tracked
handle, so backing capture, deferred deletion, recording and GPU retirement keep
the heap alive. This is ordinary lifetime ownership only: two placed resources
may retain and use the same physical interval, and no CPU-side range reservation
or global frame serialization is introduced. Submission-ledger hazard ordering
is still required before scene execution can select these bundles.

The ownership test creates two resources at the same offset, replaces the pool
allocation, and proves the old heap survives until both placed-resource leases
retire. Disarming a resource with lifetime dependencies is rejected. The D3D12
descriptor GPU test now uses a fresh placed UAV and drops its mutable pool owner
before worker recording, submission and readback.

Backing generations have also moved out of `CompileResourceShape` into ordered
realization metadata on `GraphCompileInput`. Layout-equivalent backing changes
now coalesce with a selected or in-flight symbolic compile while updating the
published bundle to the newest generation and ownership leases. A dedicated
in-flight race verifies one compile starts, the newest realization is selected,
and the superseded lease is released after harvesting. Normalization preserves
the backing-generation/global-ID correspondence. Telemetry reports realization
changes separately from structural membership, pass, constraint and queue changes.

Normal and focused ASAN suites passed 7/7 and 4/4 after this split. A full-host
ASAN radius-zero run completed 300 stable frames with no new sanitizer report,
zero comparison failures and two overlapping workers; it recorded 337
realization changes independently. Evidence is
`build/async-realization-key-radius0-asan-{report.txt,renderer.log,summary.json}`.

The separate shadow/copy environment switches were then replaced by the single
mode described above. `Async` radius zero completed 300 stable frames and
reported `ORG.AsyncCompile.Fallback.SceneExecutionNotMigrated=1`, proving that
unsupported execution is explicit. `Shadow` radius 100 completed 300 stable
frames in 39.263 seconds: 808 requests, 791 completed jobs, 16 cooperative
cancellations, peak running workers of two, 733 independently classified
realization changes, and zero compile, dependency-oracle, schedule or state
failures. Evidence is `build/async-mode-fallback-radius0-*` and
`build/async-unified-mode-radius100-*`. This establishes parallel compilation
correctness under the target scene workload; it does not establish selected
compiled-graph GPU execution.

### Functional scene execution and packed-batch barrier correction

Async mode now selects compiled scene layouts, prepares fresh typed/immediate
packets at admission, records owned D3D12 command lists, submits them through
the ordered timeline admission owner and supplies the selected batch signal to
presentation. Synchronous bootstrap and worker compilation call the same
`CompileGraph` entry point, and both compiler routes use the shared dependency,
topology, criticality and queue-selection kernels.

Packing consecutive compatible passes initially exposed a real state-placement
bug: symbolic transitions retained only a batch index, so every transition was
recorded at command-list entry. A producer-to-consumer transition inside one
packed batch therefore ran before the producer and produced large white blocks
in scene color. Symbolic state steps now retain their consuming pass; admission
builds per-pass barrier groups; and recording emits each group immediately
before that pass in compiler order. The dense state oracle also compares the
pass placement. BasicTelemetry reports total and intra-batch state steps.

The D3D12 regression now packs a UAV producer and readback consumer into one
batch, relies exclusively on framework-emitted barriers, replaces the concrete
backing twice, and verifies every readback word. The deployed radius-zero async
scene completed normally after the correction, and visual inspection confirmed
that the white-block corruption was gone.

Readback capture is now an owned async submission effect. Copy bytecode records
into the admitted command list, the packet signals the readback timeline after
successful command submission, and only then commits `FinalizeCapture`.
Graphics- and copy-queue capture pass variants use the same generic owned
immediate-effect contract. An opt-in `SARP_COLOR_OUTPUT_READBACK_PATH` capture
at `PresentPass` writes the pitched GPU bytes plus metadata and reports white
pixels and fully-white 16x16 tiles through BasicTelemetry. The first successful
async scene capture was 2560x1440 (14,745,600 bytes), with zero fully-white
16x16 tiles and no legacy-pass or dropped-token warning.

The remaining live-resource warnings at host cleanup are not specific to async
retirement: a synchronous radius-zero control reported the same published/CLod
owners (and more live buffers). That separate renderer teardown-order issue is
not treated as evidence against async execution ownership.

### Multi-queue admission and current structural-churn boundary

D3D12 async scene execution now preserves compiler queue placement. Symbolic
state steps retain both producer and consumer passes; admission emits a
release-to-common after the producer, an acquire-from-common before the
consumer, and the relative timeline wait between their batches. Backing and
physical-alias ledgers add cross-frame waits by concrete backing/subresource or
heap interval, and partial submission commits only the submitted batch prefix.
The D3D12 checked-close path dumps InfoQueue diagnostics on failure.

Debug-layer validation found and fixed invalid `COMMON + SYNC_NONE` handoff
barriers and copy-list entry barriers carrying shader access masks. Owned
graphics/compute recording lists also bind the admission-captured default
resource and sampler heaps before any directly-indexed root signature. A
radius-zero debug-layer run then submitted 86 selected async scene frames with
zero InfoQueue messages, legacy passes, or compiler/state/schedule/oracle
failures. A radius-100 run submitted 713 async frames and reached 300 stable
frames with the same zero-failure compiler telemetry.

Compiler slot capture now separates named logical slots from rotating concrete
backings. Numeric execution-slot suffixes used by CLod readback rings and the
swapchain are normalized while occurrence indices keep simultaneous resources
distinct. On radius zero this reduced membership changes from every request to
initial-loading changes and allowed most later requests to coalesce with the
selected graph.

The remaining structural churn is genuine pass-access churn at pass index zero:
the upload/immediate pass changes its resource accesses and entry states while
assets stream. The current route waits for a compatible graph because consuming
or dropping that one-shot packet would be incorrect. Completing no-wait scene
admission therefore requires reservable immediate/upload packets that can stay
pending while an older compatible graph renders, then commit exactly once when
their matching compile is admitted. Recording is also still performed on the
admission thread; moving frozen recording batches to bounded workers remains a
major performance milestone.
