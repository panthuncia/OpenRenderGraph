# Experimental asynchronous compilation

Status: dependency, conservative symbolic-schedule, and declared resource-state shadow compilation. This
is not yet an asynchronous execution path. Neither current compiled stage can be admitted to GPU
execution, and no benchmark result from this mode demonstrates that full frame
compilation is asynchronous.

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
result has no authority over the live graph. Configuration is default-off:
`experimentalAsyncCompileShadow`, `experimentalCompileConcurrency`;
`SARP_ASYNC_COMPILE_SHADOW=1` enables the renderer's shadow setting at startup.

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
