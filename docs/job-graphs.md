# Job graphs: ORG without a main graph

> **Status (2026-10-30): design, Phase 0.** This document describes the target architecture and the migration plan. It
> supersedes the service-ownership items of `rebuild-plan.md` Phase 7 (per-runtime services) and replaces
> `persistent-epochs.md` once the migration completes.

> **WIP: cross-thread synchronization and locking.** The threading model below (one admission lock in the sequencer,
> any-thread submit through the submit gate, completion callbacks on a runtime thread, thread-safe services) is a first
> cut. It is not yet designed in detail, and it deserves its own investigation before Phase 2 lands:
> - lock granularity and ordering;
> - which structures are lock-free;
> - contention under many concurrent jobs;
> - interaction with the backends' own locks (Vulkan's device-wide `timelinesMutex` held across a submit, DXVK's queue lock);
> - reentrancy of callbacks.
>
> Treat every "under one lock" statement here as provisional.

## Why

ORG grew out of a synchronous, one-graph-per-frame design, and that design still shows:

- **Per-graph state.** A `RenderGraph` owns its registry, its queue timelines (identity `slot + 1`) and its
  state/access/alias ledgers. Resource state is seeded once from the resource's `SymbolicTracker` and never written back,
  so two graphs touching one resource disagree about its state.
- **Compile mixed into execution.** `Update` interleaves compile work with per-execution work: resolver and backing
  polling, binding edits, structural builds and pass preparation all happen there. A pass that adds declarations forces
  an inline structural build on the calling (render) thread.
- **`GraphHost` defines the frame.** It fixes "a frame is one pass over the epoch order", owns the frame slots, and
  re-initializes and shuts down the process runtime.
- **Process singletons.** Upload, readback, statistics, deletion (rotated by frame count), descriptor deferred
  release, settings and the default task service are all singletons. Whichever graph or host set them up last owns
  them.

As a result, independent GPU work cannot run as its own graph. BasicRenderer runs mipmapping, BC7 and environment
filtering as passes in its frame graph, so new work changes declarations and triggers rebuilds. Community Shaders splits
one compiled graph into nine epochs.

**Goal.** Graphs are values that can be assembled, compiled (and cached), executed and reused arbitrarily, from any
thread. An external async system drives them, for example ORGModuleServices' `StateGraph`. ORG designates no graph as
main. Jobs that touch the same resource are ordered correctly, and resource state is handed off between them. That
handoff is computed off the render thread and synchronized with barriers and queue waits, without re-recording or
interrupting either job.

## Architecture

```
                    ┌──────────────────────── org::Runtime (one per device set) ────────────────────────┐
 any thread         │  Devices/queues    Descriptor heaps   Pipelines   Upload  Readback  Statistics  │
   │                │  Timeline pool (globally unique ids)  Retirement (timeline-based, not frames)    │
   │                │  ┌─────────────────────────────┐   ┌──────────────────────────────────────────┐ │
   │                │  │ ResourceSequencer           │   │ SubmitGate (per native queue)            │ │
   │                │  │  state + access + alias     │   │  releases a batch once every point it    │ │
   │                │  │  ledgers, keyed by backing  │   │  waits on is submitted; not FIFO         │ │
   │                │  └─────────────────────────────┘   └──────────────────────────────────────────┘ │
   │                └───────────────────────────────────────────────────────────────────────────────┘
   ▼
 GraphDesc ──compile (worker, cached by structural key)──► CompiledGraph (immutable, shareable)
   passes, order edges,                                    schedule, batches, body barrier plans,
   parameters (imports)                                    per-import first/last access, transient layout
                                                                │
                              Bind(params, resolvers, host data)│   (any thread)
                                                                ▼
                    Execution: Prepare ─► Admit ─► Record ─► Submit ─► Complete/Retire
                    ExecutionHandle milestones: Admitted · Submitted · Completed · Poisoned
```

### `org::Runtime`

The runtime replaces `GraphHost`, the manager singletons and `InitializeRuntimeDevice`. It owns every service that is
genuinely shared:

- **Device resources:** devices and queues, a timeline pool whose ids are globally unique, descriptor heaps, and the
  pipeline lifetime.
- **Upload service:** one service, collapsing `UploadManager`, `UploadInstance` and the copy-queue path.
- **Readback service:** completion callbacks run on a runtime thread; the render thread no longer polls.
- **Statistics:** query ranges are allocated per execution.
- **Settings and the task service.**
- **Retirement:** retirement is by timeline points throughout. The `DeletionManager` frame rotation,
  `ProcessDeferredReleases(frameIndex)` and frame-indexed upload pages are removed.

### `GraphDesc` and `CompiledGraph`

**A desc** contains:
- passes, as `TypedRenderGraphPass` objects with Declare / Update / Prepare / Record (unchanged);
- order edges;
- named **parameters**, which are the imported resources.

Extensions become functions that add passes to a desc. The anchor and priority merge from `CompileStructural` stays as
desc-assembly helpers.

**Compiling** (`GraphCompiler`, any thread) produces:
- the schedule, queue assignment and batches;
- the in-graph barrier plan (the *body*);
- for each import, the first-use and last-use step per cell (state, queue, batch);
- the transient layout.

Compilation reuses `GraphEditTransaction::Build` and `CompileWorkspace`.

**`GraphCache`** keys compiled graphs by structure: the passes and their lowering fingerprints, plus the parameter
descriptions. A structural change means building another desc. The old compiled graph keeps executing until the new
one is ready. Nothing compiles inline, and `IDynamicDeclaredResources`, inline structural builds and frame-interrupting
rebuilds are removed.

### Execution

| Step | Thread | What it does |
|---|---|---|
| `Bind` | any | Binds parameters to concrete resources, captures resolver states (resolvers become binding sources) and host data. A binding change produces a new binding set against the same compiled graph. |
| `Prepare` | worker | Runs the passes' `Prepare`. |
| `Admit` | worker | Sequences the execution in the runtime; see below. |
| `Record` | task workers | Records the bodies plus per-batch handoff lists. |
| `Submit` | any | Hands batches to the submit gate. |
| Completion | runtime | Completion callbacks; retirement. |

Other changes:
- **Execution index.** LatchBlock slots and the `PassPrepareContext` frame index become an execution index, with a slot
  ring per compiled graph. "Frame" leaves the core.
- **External submission.** An execution can be recorded into caller-provided command lists and submitted by the
  caller, who reports back with `NoteSubmitted(points)`. The handoffs go into a leading list. This generalizes
  `ExternalEpochExecution::leading`; Community Shaders uses it to submit inside DXVK's stream.

## Cross-graph ordering and state handoff

1. **One sequencer per device set.** It holds the three ledgers from `ExecutionState.h` (state, access as writer and
   readers, alias), lifted out of the graph.
   - Cells are keyed by backing handle plus subresource. Textures are split by mip × slice, plus planes for
     depth/stencil.
   - Buffers get byte-range cells, so a small upload does not serialize behind every reader of a large buffer.
   - Timeline ids come from the runtime pool and are globally unique.
2. **Admission happens on the preparing worker.** The compiled graph's import list and per-cell first/last tables are
   precomputed. For each import cell, the execution:
   - appends itself;
   - receives the predecessor's **promised exit state** and the predecessor's signal points (the writer's, plus every
     reader's for write-after-read);
   - commits its own promised exit.

   Partial submits are tracked per batch. *(Locking: see the WIP note at the top.)*
3. **Admission order must follow the async graph's order.** An execution may be admitted only after every predecessor
   that touches its resources has been admitted; otherwise it silently reads stale data. ORG asserts what it can
   (`Admit({.after = handles})`). The ORGModuleServices adapter lets dependents wait for **Admitted**, not for GPU
   completion, so consumers prepare and record in parallel with their producers and only the GPU waits.
4. **Readers.**
   - Concurrent readers that need the same layout on the same queue are not ordered against each other.
   - A reader that needs a different layout is a **transitioner**: it is recorded as a writer, waits on every prior
     reader and merges their scopes.
   - Exclusive-mode textures are serialized across queues, as the in-graph compiler already does. Concurrent sharing
     is opt-in per resource.
5. **Signal points are known before submit.**
   - Each execution signals a pooled timeline at `base + batchOrdinal`, and values stay monotonic when the timeline is
     reused.
   - Successors build their waits at admission, using wait-before-signal.
   - If the predecessor has already been submitted on the same native queue, the wait is dropped and only the barrier
     remains.
6. **Submit gate.**
   - A batch reaches the GPU only once every point it waits on has been submitted.
   - Batches are FIFO only within one execution's batches on a queue, so unrelated ready batches never block each
     other.
   - Each submit flushes the batches that have just become eligible.
   - DXVK-hosted submissions never receive waits on unsubmitted ORG points; they are gated, using a "submitted"
     callback from the submission hook.
7. **Handoff lists, one per batch.**
   - **Entry list:** transitions from the predecessor's exit state to this batch's first-use state, plus the Vulkan
     acquire. It is recorded separately and submitted ahead of the batch.
   - **Release list:** submitted on the predecessor's queue type, with its own signal point, whenever the predecessor's
     exit layout or sync is illegal on the consumer's queue. Two cases: a Vulkan queue-family transfer, and a D3D12
     compute or copy queue, which cannot leave render-target, depth or `Direct*` layouts.
   - Vulkan release and acquire use the same old and new layouts.
   - The ledger records the real exit layout. Copy-queue accesses exit as `COMMON` on D3D12.
   - Handoffs always set `discard` and the initial layout explicitly. The Vulkan backend reads live layouts at record
     time, and a handoff may be recorded before its predecessor submits.
   - Body barrier plans use only compiler-known states, so bodies are independent of handoffs and can be reused.
8. **Abandonment and failure.**
   - An admitted execution that will not submit is spliced out of the chain if no successor has recorded its handoff
     yet.
   - Otherwise it submits a pass-through: a transition to its promised exit, plus its signals.
   - An abandoned writer marks its handle `Poisoned`, and successors are told. An exception during submit takes the
     same path.
9. **Body reuse.**
   - A recorded body can be reused while the compiled graph, every binding's backing handle and the heaps are
     unchanged.
   - Two in-flight executions never share one recorded body.
10. **Clients outside ORG graphs** enroll as pseudo-executions with a declared entry and exit (`AdmitExternal`):
    - swapchain acquire, a producer, and present, a consumer;
    - resources written by DXVK, reported as incoming states;
    - upload copies;
    - BasicRenderer's `MaterialTextureTransferService`.
11. **Transient alias arenas.** Each arena belongs to a compiled graph and may be shared between graphs. Arenas are
    sequenced through the alias ledger as pseudo-resources.

**Example: upload → mip job → frame graph sampling the texture.**

```
copy queue   : [upload copies]─sig U
compute queue:          wait U ─[entry: COMMON→UAV]─[mip body]─sig M ─[release: UAV→SRV (D3D12)/QFOT (VK)]─sig R
graphics     :                                                          wait R ─[entry: acquire→SRV]─[frame body ...]
               admission order: upload ▸ mip ▸ frame. All waits/barriers were computed at admission on workers;
               none of the bodies were re-recorded; the render thread only calls Submit.
```

Frame-to-frame continuity, such as history buffers, uses the same mechanism. Execution N+1 of a frame graph is simply
the next admission on those cells.

## API sketch

```cpp
auto runtime = org::Runtime::Create({.devices = ..., .taskService = ...});

org::GraphDesc mip;
auto src = mip.Parameter<org::Texture>("source");
mip.Add<MipChainPass>(src);
std::shared_ptr<const org::CompiledGraph> g = runtime->Graphs().GetOrCompile(mip);

auto exec = g->Bind({{src, texture}})
              .Prepare(hostData)
              .Admit({.after = {uploadHandle}})
              .Record();
exec.Submit();
exec.Handle().OnCompleted([&] { /* runtime completion thread */ });
```

**ORGModuleServices adapter** (in ORGModuleServices; ORG does not depend on it):
- `MakeGpuSubmissionSet(ExecutionHandle)`.
- A `GraphJob` producer helper: it binds, prepares and admits on the producer's worker, posts Record and Submit, and
  returns `Ready(payload, submissions)`.
- A new **Admitted** readiness level between `CpuReady` and `UploadSubmitted`, so successors can start as soon as their
  producers are sequenced.

## Migration

| Phase | Content |
|---|---|
| 0 | Baselines (BasicRenderer benchmark, Community Shaders tour) and this document. |
| 1 | `org::Runtime`: services owned per runtime, with no singletons and no frame-indexed retirement. `GraphHost` becomes a thin client for now. |
| 2 | `ResourceSequencer` with open executions:<br>• promised-exit admission and per-batch handoff lists<br>• the submit gate, splice/pass-through and poison<br>• external clients<br>• removal of closed executions and home states<br>• fixes for the Vulkan QFOT layout pair and D3D12 copy-queue exit layouts<br>• the synchronization investigation (see WIP note) |
| 3 | Split `RenderGraph` into `GraphDesc` / `CompiledGraph` / `Execution`, with `GraphCache` and execution indices. Remove the Pre/Main/Tail segments, `IDynamicDeclaredResources` and inline builds. |
| 4 | ORGModuleServices adapter. |
| 5 | BasicRenderer's frame becomes a job; swapchain acquire and present become external clients; the rebuild-and-stall path is removed. |
| 6 | BasicRenderer jobs: mip chain, BC7, environment set, BRDF LUT. |
| 7 | Community Shaders: epochs become compiled graphs sharing an arena; tickets become prepared executions; external submission into DXVK. |
| 8 | Delete `GraphHost`, epochs, tickets and the frame vocabulary. Rewrite the README. |

## Top risks

1. **Admission order diverging from the async graph's order** produces silently wrong data. Mitigations: the `after`
   assertion, the Admitted readiness level, and a debug check in the ledger.
2. **Deadlocks in the submit gate or with DXVK.** Mitigations: the gate is not FIFO, unsubmitted waits are never handed
   to DXVK, and a randomized multi-graph stress test.
3. **D3D12 layout legality per queue type, and the Vulkan QFOT fix.** Both are covered by cross-queue tests on both
   backends with validation enabled.
4. **Cross-thread synchronization** (the WIP note above).
