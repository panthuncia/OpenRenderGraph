# Async frame implementation and validation

Target: immutable host inputs, one worker preparation owner, parallel compilation,
ordered worker planning, parallel recording, FIFO submission and fence retirement.
The frame-slot bound includes queued CPU work as well as submitted GPU work.
The progress table and framework audit describe the latest state; later sections
preserve earlier migration checkpoints and their historical inventory counts.

## Progress

**Visual regression repaired:** normal raster recording pushed resource-index
constants before binding a layout. A native debugger caught the rejected write in
`cl_pushConstants`, called by `RecordPreparedRenderIndirectSequence`. The helper
now binds each captured layout before writing its constants. The user confirmed
the repaired Async run renders correctly. Earlier stable benchmark reports did
not establish visual parity and must not serve as valid performance baselines.

| Phase | Status |
| --- | --- |
| 0: checkpoint baseline and telemetry | Four original failures archived; four repaired baseline runs stable; frame-stage annotation added |
| 1: frame ownership and retirement | Slot leases, retained dependencies, completion sets and recovery retention integrated; runtime descriptor snapshot integration and full program-version audit remain |
| 2: separate recording/submission | Recording boundary and slot/queue command pools implemented; focused tests and Off/Async radius-1/100 runs pass; full phase acceptance remains pending |
| 3: ordered planning and recording ahead | Planned/submitted ledgers, symbolic waits, worker planning/recording and deterministic concurrent-frame tests implemented; scene recording-ahead integration remains gated |
| 4: worker preparation and presentation tail | Pending |
| 5: pass migration and legacy API removal | 138 BasicRenderer/plugin entries use typed authoring; broader audit still finds the framework upload adapter and explicit contributor boundary paths |
| 6: full correctness/performance matrix | Pending |

## Framework follow-up audit

### Raster binding regression investigation

Artifacts: `build/async-frame-validation/visual-regression`. The same-camera,
frame-1000 Async GPU snapshots show visibility coverage increasing from 10,828
pixels (0.294%) to all 3,686,400 pixels. In both snapshots, the material histogram,
pixel list and indirect argument counts agree with visibility; all listed pixels
are unique, in bounds and match their captured visibility keys. The defect was
upstream of material evaluation. The repaired color capture restores terrain and
meshes, and the user independently confirmed the run. The capture run completed
with exit zero and `status=stable` (1,500 stable frames).

`AsyncStateGraphTests` now checks the shared raster recorder on a fresh command
list and across two different captured layouts. Each indirect draw requires the
correct constants after its layout is bound. This suite and the ORG frame-context,
async-compiler and external-resource suites pass.

Final root build/install succeeds. Radius-1 Off/run1 and Async/run2 under
`build/async-frame-validation/raster-layout-fix` have verified executable identity,
fresh stable reports, exit zero, no failing telemetry counters, no dropped events,
and no renderer errors. Both drain frame slots and GPU-retained ownership.
Async/run1 instead crashed in `WinPixGpuCapturer`'s background thread while locking
a mutex; its dump and native stack are archived. Async/run2 disables only PIX
injection (`SARP_ENABLE_PIX_GPU_CAPTURER=0`), preserving rendering techniques.
The PIX-enabled failure remains unresolved; this control is not a waiver of that
diagnostic integration issue or full migration acceptance.

For repeatable GPU-work inspection, set `SARP_GPU_WORK_READBACK_DIR` and optionally
`SARP_GPU_WORK_READBACK_FRAME` (default 120). The one-shot probe saves visibility,
material counts/offsets/indirect arguments, pixel list and surface identity, with
frame/resource/texture-layout metadata. `SARP_COLOR_OUTPUT_READBACK_PATH` uses the
same capture frame. Use a later frame, such as 1,000, to avoid startup-only images.
Diagnostic captures are not performance measurements. Full migration acceptance
and controlled performance validation remain pending.

The original `--scope renderer` inventory confirms 138 typed BasicRenderer and
plugin entries. This verifies authoring shape, not immutable recording inputs or
runtime coverage of all techniques. The default inventory now also scans ORG
framework passes and the external contributor boundary. It exposed active
`ReadbackCapturePass`/`ReadbackCopyCapturePass` and the central `UploadPass`
that the original source roots missed. Contributor adapters and host boundary
passes are reported explicitly rather than silently excluded.

Graphics and copy readback now share one typed implementation with the queue
expressed in its declaration. Preparation freezes the source binding and retains
the concrete destination allocation. A service reservation owns a dedicated
completion timeline and publishes its complete request only after submission;
cancelled captures never enter the service's pending queue. The extension and
frame retain the service owner. StreamingUploadPass also uses typed declarations
and direct copies from frozen bindings, including declared upload sources.

The central UploadManager pass still uses immediate recording; its queue drain
and resource-declaration boundary require a service migration before removing
the compiler's immediate adapters. Vendor paths also remain a blocker to
recording ahead: UpscalingFrameData retains mutable PixelBuffer wrappers and
Record calls the global UpscalingManager. Serializing Evaluate does not freeze
those inputs or protect against concurrent preparation/rebuild. Descriptor
snapshot integration, immutable vendor inputs, history versions, worker
preparation and scene recording ahead therefore remain required. Async is not
being made the default by this follow-up.

Follow-up validation artifacts are under
`build/async-frame-validation/framework-pass-unification`. Refreshed ORG ownership,
compiler and external-resource suites pass. The new readback test records after
pass destruction, verifies copied bytes on actual graphics and copy queues,
and checks service retention and cancellation without a pending capture. The
root build/install and all 11 affected CTest suites pass. Off and Async each
completed radius-1 and radius-100 runs with fresh stable reports,
verified executable identity, zero monitored failures, no dropped telemetry
and no renderer errors. Active slots peak at two in Off and three in Async;
all runs drain to zero with no pending GPU work or deferred releases.

Single-run whole-host-frame median/p95 (milliseconds, 119 samples each):

| Mode | Radius 1 | Radius 100 |
| --- | --- | --- |
| Off | 31.77 / 38.16 | 36.24 / 47.46 |
| Async | 28.87 / 36.57 | 35.31 / 44.65 |

These measurements predate the discovered raster binding bug and are invalid
as correctness or performance references. Async traces contain worker recording, but still show preparation on
the main thread and only one logical frame recording at a time. No scene
recording-ahead benefit, full visual coverage or final acceptance is claimed.
`correctness-summary.json` and `migration-audit.json` preserve the measurements,
shutdown evidence and remaining work.

## BasicRenderer authoring migration


The migration inventory now contains 138 typed entries and no prepared,
immediate, or legacy adapters. Active construction sites use the unified
`BuildPass` entrypoint. A source audit of BasicRenderer and its plugins finds no
ordinary `Execute`, `PrepareFrame`, immediate-command, worker-safety opt-in, or
submission/completion/abandonment callback entrypoints.

The final batch migrated the structural CLOD upload/readback passes, material
texture readback, BC7 compression/copy/readback, generic texture readback,
software cluster rasterization, and ray-traced reflections. Streaming and
readback side effects are represented by frame-owned reservation objects. The
ray-tracing backend remains an explicitly serialized service because its build
and trace operations share mutable vendor/backend state; the frame retains a
shared service owner plus the concrete buffers and program versions it records.

The dynamic CLOD upload declaration key now includes both destination and
staging-buffer identities. Tracking only destinations allowed a fresh staging
allocation to be captured against the preceding frame's declaration in Off
mode. The corrected implementation passes Off and Async radius-1 validation.
Async radius-100 also produces a fresh stable report, confirming that the exact
object-publication dependency correction continues to hold under the completed
streaming migration.

Validation artifacts are under
`build/async-frame-validation/remaining-pass-migration-138`. The repository-root
build succeeds, and all 11 affected ownership, async compilation, external
resource, pipeline, state-graph, and contributor ABI/loader tests pass. Visual
and performance evaluation remain deferred.

## Texture preparation and streaming signal reservations

Four additional entries now use typed preparation and static recording:
`MipmappingPass`, `DownsamplePass`, `CLodStreamingFeedbackSortPass`, and
`CLodDirectStorageLaunchPass`. Feedback sorting records its fixed radix-sort
algorithm directly from captured programs, resources, constants and an indirect
command signature, replacing the generic step packet and duplicate Execute body.

Mip generation reserves precisely the jobs included in resource declaration.
Jobs arriving afterwards wait for the next declaration. Cancellation returns
reserved jobs for retry; submitted jobs retain their resources through frame
retirement. Each job owns an immutable constants allocation. Depth downsampling
likewise gives each map backing generation its own constants allocation and
declares every active map, constants buffer and counter. This also fixes detection
of backing replacement through an unchanged resource wrapper.

Typed frame dependencies now collect service-owned external signals. The
DirectStorage launch reservation owns its timeline and rolls back an unsubmitted
armed launch on cancellation. Its detachable service owner prevents callbacks
from accessing a destroyed streaming system. Ordinary recording is empty and
does not implement submission or abandonment callbacks. A focused external
resource test checks signal propagation, exactly-once cancellation and timeline
retention through dependency release.

The unused `CLodAsyncUploadPass` and `CLodStreamingReadbackCopyPass`
implementations were removed after checking supported construction paths. The
active structural streaming implementations remain; their shared readback source
description now lives in `CLodStreamingReadbackSources.h`.

The inventory contains 118 typed, 12 prepared-adapter, four legacy and four
immediate-adapter entries (138 total). This does not establish runtime coverage
for every optional pass. The remaining service migrations, descriptor snapshots,
worker preparation and scene recording ahead are still pending. Scheduling
defaults are unchanged; visual and performance validation remain deferred.

Artifacts are archived under
`build/async-frame-validation/texture-service-unification`. The root build and
all nine focused CTest suites pass. The initial parallel build encountered a
vcpkg deployment file lock (exit 32); a serialized root build completed.
Off and Async radius-1 and radius-100 correctness runs all pass with verified
executable identity and fresh stable reports. Structured telemetry reports zero
monitored failures, dropped events or renderer errors. All runs release their
three frame slots, leave no pending GPU frames and fully drain retirement.
`correctness-summary.json` records this evidence without a performance comparison.

## History publication, forward rendering, wind debug and upscaling

Five additional entries now use typed preparation and static recording. The
Reyes tessellation-table upload tracks concrete buffer backing generations, so
replacement allocations receive the immutable table data exactly once. Linear
depth history validity is published by a ViewManager-owned submission effect.
It captures view, source resource, backing generation and history epoch, ignores
stale publications after depth replacement, and detaches safely at manager
destruction.

Forward indirect rendering captures coherent pipeline payloads, descriptor
bindings, the dispatch-mesh signature and concrete indirect allocations. Its
explicit disabled packet preserves the prior no-op behavior while published
workloads are unavailable. The first typed conversion omitted this flag and
produced five invalid RTV errors in the archived
`remaining-pass-migration-122/Async/radius1/run1` diagnostic run; the repaired
run is clean.

The procedural-wind skeleton diagnostic now freezes both custom pipelines,
their bindings, the command signature, indirect resources, output slot, counts
and constants. Upscaling frame data owns its exact input/output resource
wrappers, and vendor evaluation is serialized because the backend integrations
mutate shared temporal context.

The current inventory contains 123 typed, seven prepared-adapter, four legacy
and four immediate-adapter entries. Fifteen transitional entries remain.

Screen-space reflections now use owned texture inputs and serialized FFX context
evaluation. The two built-in debug overlays snapshot ECS-derived sphere and
skeleton geometry during preparation. Their recording packets retain custom
pipelines, descriptor bindings and concrete input allocations, and contain no
queries into the live ECS world. Skeleton line uploads also move out of the
recording callback. The current inventory is 126 typed, seven prepared-adapter,
one legacy and four immediate-adapter entries; twelve remain.

The 126-entry checkpoint passes the serialized root build and all nine focused
CTest suites. Off and Async radius-1 plus Async radius-100 correctness runs have
fresh stable reports, verified executable identity, zero monitored failures,
zero dropped events and zero renderer errors. Visual and performance comparison
remain deferred.

`VoxelSoftwareRasterizationPass` now records its two build/raster variants from
captured program bindings, concrete indirect allocations and a retained command
signature. Its duplicate Execute body and prepared-command ownership fields are
removed. The 127-entry checkpoint passes the root build, all nine focused tests
and an Async radius-1 correctness run with a fresh stable report and no monitored
errors or dropped events. Eleven transitional entries remain.

`ClusterRasterizationPass` now uses the same typed indirect packet for all six
supported outputs: visibility, virtual shadow, deep visibility, AVBOIT
occupancy, AVBOIT raster and AVBOIT shading. Programs, descriptor mappings,
the indirect allocation and command signature are captured through frame
ownership. The transparency fallback and duplicate Execute path are removed.
The 128-entry checkpoint passes the root build, focused tests and Async
radius-1 correctness validation. Ten transitional entries remain.

## Object publication correction and Reyes/page-job migration

The radius-100 object-publication failure is reproduced deterministically in
`AsyncStateGraphTests`. A valid DrawRecords root describes buffer revision 1;
publishing buffer revision 2 invalidates the root's `LatestAtLeast` dependency
and rebuilds the unchanged revision-1 input against revision 2. The builder
correctly rejects the mixed cut. This can happen before the replacement root
is admitted, even with ObjectManager's one-root-at-a-time consumer gate.

ObjectManager now requests `Exact` dependencies using the admitted revision
and generation handles, including the visibility-generation sidecar. It does
not relax ABI/revision validation. The regression retains the original cut
across buffer replacement and then publishes a new cut, checking both payloads
and their mutation coverage. With the old dependency policy it reproduces the
same error as the scene; with exact dependencies it passes. A DebugMCP launch
did not establish a session; deterministic CTest execution supplies the
reproduction evidence. Temporary debugger configuration was restored.

Nine more passes use typed preparation and static recording: Reyes raster-work
histogram and compaction/argument generation, virtual-shadow block expansion,
software-raster page-job expansion and rasterization, deep-visibility resolve,
Reyes deep-visibility rasterization, and Reyes virtual-shadow compute and
hardware rasterization. Their programs and indirect command signatures are
retained by captured frame dependencies.
Preparation freezes bucket variants, settings, constants, argument offsets
and concrete barrier resources. Recording preserves clear/dispatch/barrier
ordering, including the page-job clears when the bucket count is zero.
The legacy execution bodies and empty setup/cleanup overrides are removed.
The hardware shadow recorder now explicitly balances BeginPass with EndPass.

The inventory is 114 typed entries, 16 prepared adapters, four legacy entries
and six immediate adapters. This is source migration evidence, not execution
coverage of every optional technique. Worker preparation, descriptor snapshot
integration, scene recording ahead and the remaining authoring/service work
are still pending. Visual and performance validation remain deferred; the
default scheduling mode is unchanged.

Artifacts under `build/async-frame-validation/object-cut-unification` preserve
the failing reproducer, fixed tests, builds and the first repaired Async
radius-100 run. That run reaches stable completion with verified executable
identity, zero renderer errors, zero failing counters and no dropped telemetry.
The combined implementation passes the root build and all nine refreshed
focused tests. An intermediate five-pass checkpoint passed Off and Async at
both radii under `build/async-frame-validation/reyes-page-job-unification`.
The final nine-pass build passes the same four correctness runs under
`build/async-frame-validation/reyes-deep-visibility-unification`. Every final
run has verified executable identity, a fresh stable report, zero monitored
failures, no dropped events, no renderer errors and complete frame-slot/GPU
retirement drainage. `correctness-summary.json` records these checks without
performing a performance comparison. Three post-fix Async radius-100 runs
across these checkpoints complete successfully.

## Shadow upload service and pass unification

The predicted-page deduplication, page marking and page admission passes now
use typed preparation and static recording. Their duplicate execution bodies
and page admission's per-pass submission state are removed. Programs, indirect
arguments, command signatures and barrier resources are captured during
preparation. Clear/dispatch ordering and UAV barriers are preserved.

Virtual-shadow upgrade publications now use `FrameWorkQueue` reservations.
Each publication owns a CPU-written upload-slot lease. Cancellation returns
the reservation for retry; submission retains it until frame dependencies
retire. Reset discards unsubmitted publications without freeing submitted
slots. Reinstalling the same buffers preserves their busy slot ownership.
Retirement callbacks use a detachable wake owner so they may safely outlive
the streaming system. The ordinary admission pass no longer acquires and
releases producer slots using frame-index callbacks.

Structured telemetry exports accepted, pending, reserved, submitted, returned
and discarded shadow-upgrade work. The analyzer reconciles these counts using
the same conservation rule as the environment services. The ownership test
now retains a publication through cancellation/retry, submission and service
replacement, and verifies exactly one final release after frame ownership ends.

Artifacts are under `build/async-frame-validation/shadow-service-unification`.
The root build/install and all nine affected CTest tests pass. Off and Async
radius-1 run1 pass with verified executable identity, stable completion, zero
renderer errors, zero failing counters, no dropped telemetry and no outstanding
GPU ownership at shutdown. Async's upgrade-job counters are zero; that run
does not establish active upgrade-work coverage.

Async radius-100 run1 is a failed correctness gate: it times out before stable
completion after `ObjectBufferStateArtifact` reports an ABI/revision mismatch
(artifact 7:0:0, revision 122, generation 25356). Shutdown still drains all
three slots and pending GPU ownership. ObjectManager requests `LatestAtLeast`
buffer dependencies while the object-state builder requires exact revisions;
this is a suspected cause, not a verified diagnosis. A separate debugger-attach
run under `shadow-service-diagnostic` did not reach the breakpoint and exited
with an access violation; it supplies no acceptance evidence. The temporary
debugger configuration and breakpoint were removed.

Off radius-100 run1 passes with a fresh stable report, verified executable,
zero failing counters, zero dropped events and zero renderer errors. This
single successful Off run did not prove the failed Async run was caused by
scheduling. The publication mismatch was an open correctness blocker at this
checkpoint; the follow-up above records its reproduction and correction.

Visual and performance validation are deferred at the user's request. This
does not waive the radius-100 correctness failure or change the scheduling
default. The inventory is 105 typed entries, 16 prepared adapters, 13 legacy
entries and six immediate adapters. Production descriptor snapshots, worker
preparation, scene recording ahead and the remaining authoring migration are
still unfinished.

## Pass unification work in progress

The root `scripts/async_pass_inventory.py` inventories supported source paths,
including optional techniques and abstract bases. It is a migration checklist,
not evidence that every pass has executed. `--require-unified` fails while legacy
bases, adapters or ordinary lifecycle callbacks remain.

That batch's header-only inventory recorded 43 typed passes, 44 prepared adapters, 38 legacy
classes, three immediate adapters and two inherited legacy classes. Nineteen
passes moved to typed authoring in this batch: the three GTAO stages, luminance
histogram and average, motion-vector dilation, Reyes dispatch-argument creation,
ten virtual-shadow-map stages, Specular IBL and BloomBlend. Compute declarations
preserve automatic queue assignment with a compute preference. Duplicate legacy
recording bodies were removed from these passes and seven previously migrated
CLOD implementations.

Typed authoring supports an optional preparation method. A direct pass supplies
`static Record(PassRecordContext&)`; the framework supplies empty frame data.
Prepared passes retain `Prepare -> FrameData -> static Record`. This is still a
transitional base: inherited update machinery, lifecycle adaptation and shared
heap handles remain. Typed classification alone does not establish the final
immutable-binding contract.

Program payloads now retain their root-layout generation with the PSO and
binding metadata. PSOManager factories and fullscreen custom pipelines capture
that ownership. Recording helpers resolve the captured layout; the transitional
layout fallback remains for unmigrated payloads. The external-resource suite
creates actual compute pipelines/layouts, replaces the current version, records
and submits the retained old version, and checks release after GPU completion.
The direct-authoring test records after destroying the pass object.

Root build/install and all nine affected CTest suites pass, including refreshed
ORG ownership/compiler/external-resource tests, renderer state/pipeline recipes
and contributor ABI/header/loader checks. Runtime artifacts
for this batch are under `build/async-frame-validation/unification` in the host
repository. The first Async radius-1 run crashed before rendering: four migrated shadow
passes retained a duplicate descriptor-mapping read after moving their payload
into captured ownership. Crash-dump inspection identified the moved-from
pointer, and those reads were removed. The failed run and dump are archived;
post-fix Async radius-1 run2 and Off radius-1 run1 pass with verified executable
identity, stable reports, zero failing counters, zero dropped telemetry and no
renderer errors. Both drain three owned slots to zero with no pending GPU work.
Quick-run whole-host-frame median/p95 is 46.23/54.83 ms for Async and
41.12/52.15 ms for Off (119 samples each). These single runs are correctness
checks, not performance acceptance; Async is slower and the controlled
comparison remains mandatory. This work does not complete descriptor
snapshot integration, all custom program/signature/work-graph capture, service
migration, worker preparation or scene recording ahead. Scheduling defaults
remain unchanged.

## Indirect-compute unification follow-up

Eight more passes use typed preparation: Reyes classification, seed patches,
dicing, replay merge, raster-work construction, patch rasterization, virtual
shadow page allocation and terrain RVT material-page generation. Their duplicate
Execute bodies and empty authoring overrides are removed. Declaration queue
preferences and the existing admission order are preserved.

Preparation now captures command-signature ownership into the dependency
snapshot. These passes and procedural-wind indirect simulation no longer carry
signature ownership manually. The manager's raw dispatch signature is versioned
by shared ownership so cleanup/reinitialization does not destroy a captured
signature. Other manager signature APIs remain migration work.

All callers of the single indirect-dispatch helper now use captured programs
and argument resources. That helper no longer has legacy pipeline, layout,
pipeline-owner, signature-owner or raw argument-resource fields. Disabled work
still returns without recording; enabled work requires captured bindings. The
native program-version test also checks that a captured command signature
survives release of its original owner and is released with frame dependencies.

The source inventory now scans renderer implementation files and plugins as well
as headers. It reports 57 typed classes, 38 prepared adapters, 38 legacy classes,
six immediate adapters and two inherited legacy classes. The larger inventory
includes previously omitted streaming and procedural-wind implementations; the
counts must not be compared directly with the old header-only total.

Validation artifacts are under `build/async-frame-validation/indirect-unification`.
Root build/install and all nine affected CTest suites pass. Off and Async
radius-1 run1 both pass with stable reports, verified executable identity,
zero failing counters, zero dropped events and no renderer errors. Both modes
retain at most three frame slots and drain them to zero at shutdown. The initial
build exposed an unmigrated terrain caller, which is now included above; a
subsequent LNK1163 in TextureFactory.obj cleared after rebuilding that generated
object. The final root build exits zero. These quick checks do not replace
representative visual/technique coverage or the controlled performance matrix.
This remains authoring/ownership work;
production descriptor snapshots, service unification, worker preparation and
scene recording ahead are not claimed complete. Defaults remain unchanged.

## Remaining migration follow-up (in progress)

This follow-up migrates 27 additional passes: the seven remaining terrain RVT
stages; streaming frame reset and four multi-dispatch virtual-shadow stages;
all eleven AVBOIT stages; bloom sampling; and all three environment stages.
The expanded inventory now reports 84 typed classes, 22 prepared adapters,
27 legacy classes, six immediate adapters and two inherited legacy classes.
Thus 57 inventory entries still require migration or an explicit unused-code
removal decision. Typed authoring does not by itself establish final descriptor
or host-input immutability.

AVBOIT setup captures clear targets/resources; early depth captures its custom
PSO, layout, indirect signature and argument/count buffers. Ordinary callbacks
record from plain values and captured bindings. The once-only fit-state upload
in AVBOIT Update remains a service-migration item.

Bloom sampling uses the common fullscreen recorder with captured mip targets.
The fullscreen helper has no raw pipeline/layout/owner fallback. Multi-dispatch
recording resolves each captured program's layout while preserving dispatch
order and internal UAV barriers.

Environment conversion, prefilter and SH jobs now live in manager-owned
FrameWorkQueue services rather than consumable vectors of Environment pointers.
Preparation reserves a snapshot; submission consumes it exactly once, while
CPU cancellation or failed publication returns nodes in publication order.
Discarded jobs are not resurrected by later cancellation. Reservations retain
work and service state independently of pass/graph lifetime. Source-publication
release belongs to the service, under an owned publication mutex; it does not
release the frame's GPU ownership. Discarding service intent does not revoke
already recorded work: generation changes still require cancelling/joining the
unsubmitted suffix. Moving preparation off-thread must also route mutable source
publication changes through its owner.

Environment recording contains only precomputed face/mip dispatch values and
captured custom program versions. It no longer looks up texture views, dimensions,
or manager state. Conversion no longer rejects owned preparation when jobs exist.
All ordinary environment submission/cancellation behavior is in the service.

Structured telemetry exports accepted, pending, reserved, submitted, returned
and discarded job counts for each environment queue. The analyzer checks
accepted = pending + reserved + submitted + discarded where these counters are
present. Focused infrastructure tests cover return ordering, duplicate lifecycle
calls, failed reservation publication, obsolete-job discard, stale selections,
service-side commit and counter reconciliation.

Root build/install and all nine affected CTest suites pass. Intermediate Off/Async
radius-1 runs (run1) passed before the environment-service changes. Current-binary
radius-1 run2 and radius-100 run1 pass in both modes with verified executable
identity, fresh stable reports, zero failing counters, no dropped events and no
renderer errors. Artifacts are under
`build/async-frame-validation/remaining-migration`. All four runs drain frame
ownership and GPU retirement. Environment prefilter and SH each submit one job
and reconcile; conversion has zero jobs, so this scene does not establish
runtime conversion coverage.

Radius-100 whole-host-frame median/p95 is 41.08/50.97 ms for Async and
48.97/60.47 ms for Off (119 samples each). These single runs do not satisfy the
three-run controlled performance gate or resolve the historical regression.
Async traces place planning/recording off the main thread, but preparation
remains on it. Peak simultaneous recording frames is one and measured
preparation/recording overlap across different frames is zero. No actual scene
recording-ahead benefit is claimed. No complete technique/visual matrix,
diagnostic scene validation or final performance acceptance is claimed.

Descriptor snapshot integration, the remaining authoring/service adapters,
worker preparation, scene recording ahead, final cleanup and the default switch
remain unfinished. Off/Async still use the existing scene admission ordering.

## Terrain/material and clear/compute follow-up

The terrain-region family now uses typed recording: counter reset, histogram,
block scan/offsets, pixel list, command-build dispatch arguments, indirect
command construction and material group evaluation. Its shared range base is
typed too. General material evaluation and primary depth copy also migrated.
Material variants, per-variant programs/descriptors, settings and indirect
argument offsets are selected during preparation. Recording does not revisit
the material publication, pipeline manager or resource registry. The specialized
optional terrain-RVT binding policy is preserved during program capture.

All five manager command-signature types now support shared version capture;
cleanup/reinitialization releases the manager's ownership without destroying
versions retained by queued frames. The terrain/material passes retain their
signatures through the preparation context. Unmigrated raw getter callers still
need conversion. Indirect sequence recording resolves each captured layout with
its program; the software-raster adapter still needs the transitional fallback.

The root build and all nine affected discovered tests pass for this checkpoint.
Both modes at radii 1 and 100 pass with fresh stable reports, verified executable
identity, no renderer errors, no failing counters or dropped events, and clean
frame/GPU retirement. Artifacts are in
`build/async-frame-validation/terrain-material-unification`. Radius-100 host
frame median/p95 is 28.56/38.04 ms for Async and 36.40/45.28 ms for Off (119
samples each). These single runs are not the final controlled performance gate.
Preparation remains on the main thread and scene cross-frame overlap is zero.

The next group migrates skybox, debug grid, software page-job argument building,
virtual-shadow raster argument building and deep-visibility clears. Debug-grid
configuration is construction-owned; the unused mutable parameter accessor was
removed. The common resource-clear recorder shares the AVBOIT and deep-visibility
implementation with captured resource/descriptor references. Existing deep
visibility uploads remain Update/service-migration work.

`ClearIndirectDrawCommandUAVsPass` and its input type were removed as unused.
A repository source search found only their definitions and an unused Renderer
include, with no construction paths; the include was removed too.

Reyes queue reset and split use the common compute recording helpers and captured
program/signature/resource references. Their duplicate Execute bodies and
manual prepared ownership were deleted. Split now inserts a buffer UAV barrier
between clearing its two output counters and the split shader's counter atomics;
both previous recording bodies omitted this required dependency.

The latter group passes the root build and all nine affected tests. Off/Async
radius-1 and radius-100 run1 have fresh stable reports, verified executable
identity, zero renderer errors/failing counters/dropped events, at most three
owned slots, and clean shutdown retirement. Artifacts are under
`build/async-frame-validation/clear-compute-unification`. The inventory is now
102 typed entries, 19 prepared adapters, 13 legacy entries and six immediate
adapters (140 total, including the shared terrain base). This turn migrated 17
concrete passes plus that base and removed one unused pass.

Three radius-100 runs per mode on the same final executable all pass correctness
and telemetry checks. Host-frame median/p95 (ms, 119 samples per run):

| Run | Async | Off | Async median/p95 change |
| --- | --- | --- | --- |
| 1 | 44.80/63.18 | 39.38/58.17 | +13.8%/+8.6% |
| 2 | 49.73/61.37 | 60.62/72.63 | -18.0%/-15.5% |
| 3 | 52.17/64.82 | 37.96/44.34 | +37.4%/+46.2% |

Two of three pairs exceed the 5% threshold for both median and p95. The
performance gate fails and remains an acceptance blocker, despite substantial
run variation. These measurements do not waive the historical regression or
substitute for reference/GPU-time comparisons. The
first Async run spends a median 8.06 ms waiting for the compile queue head,
6.15 ms waiting for planning and 3.00 ms waiting for recording. Preparation
still runs on the main thread; scene recording overlap remains one frame with
zero preparation/recording overlap across different frames. The run comparison
and executable hash are in `radius100-comparison.json` beside the artifacts.

Visual capture was attempted using the computer-use helper, which returned
`Computer Use native pipe is unavailable` (Windows error 2). No screenshot or
visual parity claim is made. D3D diagnostic and representative optional-technique
runs remain open. Descriptor snapshots,
remaining adapters/services, worker preparation, scene recording ahead,
representative visual/diagnostic coverage and final performance acceptance are
still unfinished. Scheduling defaults remain unchanged.

## Baseline

BasicRenderer `db6c6dbb`, OpenRenderGraph `a6a875e`; pre-rework references
`3a4af093` and `3580776`. Existing SARP and BasicRHI working-tree modifications
were preserved. Inventory and build logs are in the root repository's
`build/async-frame-validation/phase0` directory.

`build.cmd` passed. Off/Async runs at radii 1/100 all reached profiling but
failed during export, so none supplies an accepted performance baseline.
The radius-1 crash dump resolves to `SnapshotState`, copying a dangling
compile-step callsite name. `ScopedCompileProfileStep` stored a callsite on the
stack although telemetry retains its address through export. The fix interns
compile-step callsites and owns their names. A regression test exports after
all dynamically named step objects have been destroyed.

After that repair, Off and Async at both radii produced fresh `status=stable`
reports with executable identity and structured telemetry. These runs are under
`phase0-fixed`, separate from the failing original checkpoint runs. They are
single preliminary measurements, not the final three-run performance matrix.

The external-resource GPU test also had stale batch/barrier assumptions: its
two same-queue passes can legally merge unless isolation is declared, and a
fresh COMMON buffer need not receive an initial enhanced buffer barrier. The
test now requests the two batches it exercises and checks the actual intervening
copy dependency. Its descriptor execution test still validates GPU readback.

## Implemented ownership and recording boundaries

`FrameContext` retains prepared dependencies and a `FrameSlotLease` across
queued CPU work and execution. `CompletionSet` requires every touched queue;
unrelated fence advancement and device-removal sentinel values cannot retire
the frame. Partial/uncertain submission transitions ownership into recovery,
which ordinary cancellation and completion polling cannot release.

`PlannedFrame` supplies frozen recording jobs, bindings, barriers and waits.
`RecordFrame` allocates and records lists, checks closure, joins child work on
failure, and returns a move-only `RecordedFrame`. Submission consumes closed
recordings once. Command pools are keyed by preparation slot and queue, and
allocator reset happens on the next recording worker after ownership retirement.
The existing pass lifecycle packets remain transitional; this does not yet
implement the final simplified pass-authoring API.

The artificial queue-prefill delay has been removed. A phase-1 Async run caught
slot wrap into retained work with that delay; the failing run remains archived.
Both modes subsequently passed radius-1 with no monitored failures or dropped
trace events. This establishes no claim of multi-frame recording overlap.
The later recording-pool run exposed incomplete GPU retirement at slot reuse:
the host's graphics frame fence did not cover every queue in the frame's
completion set. Reuse now waits those explicit completion points. It still
rejects wrapping into unsubmitted CPU work rather than overwriting it.

Async scheduling now runs ordered planning and ordinary batch recording on
workers. Off invokes the same implementation inline. The host uses a non-helping
readiness wait so it cannot steal ordinary recording callbacks. Scheduler
rejection or a non-worker-safe packet is an explicit error, not an inline Async
fallback. Preparation remains on the host, and it still waits for the current
frame before submission. The presentation path binds the acquired swapchain
image at admission. Descriptor snapshots, worker preparation, presentation
tails, full pass migration and Shadow removal remain required work.

## Ordered planning and recording foundation

`FramePlanningState` serializes planning and confirmation. It tracks a planned
tail separately from confirmed submitted resource state, access history and
alias occupancy. Recording receives immutable barriers and bindings. Cross-frame
dependencies retain predecessor batch tokens; FIFO submission resolves these
to actual signals, including gaps caused by other submissions. Symbolic values
never reach the RHI. Confirmation removes resolved tokens from the planner.

Only the FIFO head may submit, even when two frames need no explicit queue wait.
After all affected recording jobs join, cancelling an unsubmitted frame cancels
the entire planned suffix and restores the tail from confirmed state. A partial
or uncertain submission confirms only known signaled batches and blocks further
planning/submission. Potentially submitted ownership remains retained for
recovery; this is not a complete device-recovery implementation.

Successful planning and confirmation avoid copying the historical ledgers each
frame. An exceptional update blocks further work until suffix cancellation or
recovery. Planner operations are serialized by their owner; recording workers
only read snapshots and do not access mutable ledgers.

Deterministic tests cover RAW/WAR/WAW dependencies, alias handoffs, cancellation
restoring state and alias occupancy, actual signal gaps, FIFO rejection, and
uncertain submission. A WARP test holds two recording callbacks active together,
finishes the second frame first, rejects its early submission, then submits both
in order. A second iteration fails recording and joins/cancels the suffix without
advancing the GPU timeline. Its native lists are intentionally no-op; existing
descriptor/readback tests supply actual GPU resource-use coverage.

This establishes concurrent recording capability, not scene recording ahead.
Enabling that production queue still requires frame-stable descriptor contents,
CPU-written ranges, immutable host inputs and presentation separation. The
current scene path deliberately preserves one-frame admission ordering.

The initial optimized Off radius-1 run (`phase3/Off/radius1/run2`) failed with
`Backing state was not seeded before commit`. Dump inspection identified an
alias-reactivation seed omitted when avoiding ledger copies. The fix treats an
invalidated historical handle as requiring a fresh seed and invalidates confirmed
state only at its first committed use. An A-to-B-to-A alias regression covers
the case. The failed run and dump analysis remain archived; it is not counted
as a passing benchmark.

Focused tests cover queued ownership across backing/program-owner replacement,
multi-queue retirement, cancellation after CPU join, recovery retention, actual
GPU descriptor use/readback, checked-close failure, duplicate submission, and
allocator reuse after retirement. The owner replacement test is not a full
shader-reload integration test. Final acceptance remains blocked on that audit
and the unimplemented phases above.

## Latest validation evidence

### Retirement follow-up

Live debugging at `org::DeviceManager::Cleanup` confirmed that the descriptor
manager's shader-visible heap was already empty while 958 deferred-release
entries remained. The host had cleaned descriptor services before releasing
graph, manager and ECS owners; their destructors then populated the retirement
queue again. Artifacts are under `build/async-frame-validation/retirement`.

`StopFrameProduction` now joins compilation/recording and cancels unsubmitted
work while retaining submitted/recovery owners. It rejects further updates and
execution. GPU idle precedes successful-frame retirement and ownership release.
The host retains its descriptor service through graph, manager and ECS teardown,
then drains recursively retired owners before shutting down the allocator.
`Cleanup` drains to an empty queue rather than assuming only two release waves.
The external-resource suite covers explicit completion against unrelated fence
advancement, a five-level reentrant retirement chain, and idempotent shutdown.

The first follow-up scene run (`retirement/Async/radius1/run1`) drained the
frame slots and 949 deferred entries but still reported 83 live buffers and four
textures. A second ownership path was in BasicRenderer's `AsyncStateGraph`:
dependency selection and ready-gate latching used owning thread-local artifact
snapshots. TBB worker lifetime exceeds graph/device lifetime, so those snapshots
kept whole publication generations alive. Selection now uses caller-scoped
storage. A regression destroys a graph while keeping its worker pool alive and
checks payload release for LatestAtLeast and ReadyGate dependencies. It fails
before this fix and passes afterward; both logs are preserved.

Removing accidental TLS retention exposed a diagnostic bug: a completed
coalescible consumer could be reported blocked on an already-consumed ReadyGate
when its old source version retired. Diagnostics now recognize the consumed gate;
unfinished successors still report blockers. A regression verifies both cases and
actual source-payload reclamation. The intervening `Async/radius1/run2` timeout is
preserved as failed evidence, not accepted as a stable run.

Resource-group resolver snapshots also formed a cycle: the cache owned its
snapshot, whose dependency identity owned the cache. Identity is now an independent
shared token, stable across cloned resolvers and revisions without retaining the
cache. A regression verifies cache reuse, revision changes, queued snapshot lifetime
beyond resolver destruction, and final snapshot release; it fails before the fix.

The benchmark wrapper now rejects renderer error/critical log lines even when
its report says stable. Structured analysis includes slots before/after shutdown,
pending GPU frames, and drained ownership counts. Async and Off radius-1 pass on
the installed fix (`retirement/Async/radius1/run3`, `retirement/Off/radius1/run1`):
zero error/critical messages, zero monitored failures/dropped events, three slots
retired to zero, no pending GPU frames, and empty deferred-release queues.
All nine selected CTest suites pass, including the new resolver regression.

The first radius-100 runs also pass correctness and shutdown in both modes
(`retirement/Async/radius100/run1`, `retirement/Off/radius100/run1`). Both have zero
monitored failures, dropped events, and renderer error/critical lines. Active slots
peak at three and finish at zero with no pending GPU work. Shutdown drains 2,608
owners in Async and 2,633 in Off, then reports empty retirement queues.

These changes do not enable additional scene overlap: Async still records at most
one logical scene frame at a time and prepares on the main thread. The trace
continues to show zero main-thread planning or batch-recording events. Ahead-of-time
descriptor snapshots, immutable host input publication, and presentation separation
remain prerequisites for the next scheduling step.

Performance acceptance remains **failed/open**, not waived by clean shutdown.
The first Async radius-1 run shared the machine with focused test compilation and
is correctness-only (30.40/41.77 ms median/p95). Off radius-1 is 33.36/42.81 ms,
within 5% of its prior median and lower at p95. Radius-100 Async runs are
36.80/61.39 and 38.22/60.42 ms, versus 31.36/40.00 ms in the phase-3 reference;
Off's first run is 47.40/74.79 ms versus 39.51/50.95 ms. These repeatable Async
increases exceed the threshold. Stage increases are broad across preparation,
compilation, planning and recording. A five-second CPU activity sample during the
repeat records 30.27 CPU-seconds in an unrelated Java process and 4.20 in the C++
language service (`retirement/cpu-activity-during-async100-repeat.json`). This is a
confounder, not proof of the cause or grounds to accept the regression. A controlled
comparison against the reference and new Off mode is still required. Unrelated
processes were left running. No final performance, visual, D3D diagnostic, or
three-warm-run matrix is claimed for this follow-up.

The Off radius-100 repeat (`retirement/Off/radius100/run2`) also passes correctness
and clean shutdown at 40.23/54.67 ms: its median is within 5% of the reference,
but p95 remains 7.3% higher. All six completed runs have zero monitored failures,
dropped events and renderer error/critical lines. Summary evidence is in
`build/async-frame-validation/retirement/validation-summary.json`; failed earlier
runs remain archived separately. Root deployment succeeded and the benchmark
verified the launched executable identity in every accepted correctness run.

| Mode | Radius | Runs under `retirement` | Correctness / retirement | Median / p95 ms |
| --- | --- | --- | --- | --- |
| Async | 1 | `Async/radius1/run3` | Pass | 30.40 / 41.77 (concurrent test build; correctness only) |
| Off | 1 | `Off/radius1/run1` | Pass | 33.36 / 42.81 |
| Async | 100 | `Async/radius100/run1`, `run2` | Both pass | 36.80 / 61.39; 38.22 / 60.42 |
| Off | 100 | `Off/radius100/run1`, `run2` | Both pass | 47.40 / 74.79; 40.23 / 54.67 |

This closes the reproduced checkpoint ownership/shutdown failures in these gates.
It does not complete phase 3/4 or final acceptance: multi-frame scene overlap,
immutable off-thread preparation, descriptor integration, presentation separation,
and the controlled performance/diagnostic/visual matrix remain outstanding.

### Previous phase-3 evidence (before retirement fixes)

The following measurements and failures are preserved as the comparison reference.

Root `build.cmd` built and installed the current renderer. Eight selected CTest
suites passed, covering frame ownership, compiler, external-resource GPU
execution, renderer async state, and contributor ABI/header/registry/loader.

| Scheduling | Radius | Artifact directory below `build/async-frame-validation` | Result |
| --- | --- | --- | --- |
| Async | 1 | `phase3/Async/radius1/run2` | Stable; executable hash verified; zero monitored failures/dropped events |
| Off | 1 | `phase3/Off/radius1/run3` | Stable; executable hash verified; zero monitored failures/dropped events |
| Async | 100 | `phase3/Async/radius100/run1` | Stable; executable hash verified; zero monitored failures/dropped events |
| Off | 100 | `phase3/Off/radius100/run1` | Stable; executable hash verified; zero monitored failures/dropped events |

The Async radius-100 trace reports 123 annotated logical frames, at most one
simultaneously recording frame, and zero preparation/recording overlap between
different frames. Its main thread runs 120 preparation events and **zero planning
or batch-recording events**, versus 3,437 main-thread batch recordings in the
phase-2 reference. Thus the recording-thread boundary passes, but the full
overlap and main-thread-only-submission gates **do not pass yet**. Active slots
remain within the configured bound of three. Planner depth is one in the current
scene path; symbolic signals peak at 97. Compiler retained-input/selected bytes
peak at 8,106,876 (this is not total renderer memory).

The stable reports above do not establish clean shutdown. The repaired checkpoint,
phase-2 and phase-3 logs report live buffers/textures at allocator cleanup. For
example, Async radius-100 reports 202 buffers and three textures, versus 209 and
three in the reference. These pre-existing failures motivated the retirement follow-up above; the old
runs do not pass final acceptance. The analyzer now includes error/critical log lines and exits
unsuccessfully for these errors even when telemetry failure counters are zero.

Each capture contains 119 non-warmup whole-host-frame samples. Initial phase-3
median/p95 measurements (milliseconds) are 47.10/57.65 for Async radius-1,
43.26/51.93 for Off radius-1, 31.36/40.00 for Async radius-100, and 44.77/59.67
for Off radius-100. These are preliminary, not a performance acceptance matrix.
Against phase-2, Async radius-100 improves, while the other initial runs regress.
In Async radius-1 the largest increase is the host's GPU frame wait (median
15.76 ms versus 2.98 ms); planning and confirmation cost 4.38 and 0.84 ms.
Repeat measurements and GPU timing investigation remain necessary. No final
visual, three-warm-run performance, diagnostic scene, or full failure-stress
matrix is claimed.

Radius-1 repeats on the same binary (`phase3/Off/radius1/run4` and
`phase3/Async/radius1/run3`) report stable completion, zero monitored counters
and zero dropped events, with the same shutdown errors. Their median/p95 times
are 32.80/44.48 ms (Off) and 27.59/36.71 ms (Async). GPU frame-wait medians fall
to 0.04 ms in both. Thus the initial radius-1 regression did not reproduce;
the substantial run variation still prevents a final performance conclusion.
The Off radius-100 repeat (`phase3/Off/radius100/run2`) likewise completes stably
at 39.51/50.95 ms, versus 39.40/53.30 ms in phase-2. Its initial greater-than-5%
regression also does not reproduce. All repeat reports retain the shutdown
errors described above.

## Validation entry points

Build/deploy with root `build.cmd`. Run registered CTest suites using
`ctest --test-dir build/vs2026-renderer-host -C RelWithDebInfo --output-on-failure`.
Because BasicRenderer is excluded from the root default test build, build the
three focused ORG test executables explicitly through root `build.cmd` with
`SARP_SKIP_INSTALL=1` and arguments
`8 OpenRenderGraphAsyncFrameTests BasicRenderer/OpenRenderGraph`, then unset that
environment variable before the next deployment build.
Root `scripts/validate_async_frame.ps1 -Phase <phase> -Mode Off|Async -Radius 1|100 -Run <n>`
launches the MO2 scene benchmark for world `0000003C`, captures structured
telemetry and archives the reused benchmark reports into a unique directory.
It rejects an existing output directory, nonzero exit, missing/freshness-failed
report, non-stable status, renderer error/critical log lines and missing telemetry artifacts.
It also compares the installed and launched executable against the selected
build hash and saves configuration, MO2 profile files and tracked-tree patches.
`scripts/analyze_async_frame.py <run-directory> --baseline <reference-directory>`
reports counters, renderer log errors, median/p95 stage and whole-host-frame
timings, main-thread stage events and measured
overlap from logical frame annotations in the SQLite trace. Absence of overlap
is reported directly, not converted into an inferred performance benefit.

Final acceptance requires three warm runs per radius/mode, visual checks,
bounded frame ownership, no ordinary preparation/recording on the main thread,
deterministic overlap and cancellation tests, and no repeatable median/p95
frame-time regression exceeding 5%. Diagnostic runs are separate from timed
runs. No phase is complete until its validation gate passes.
