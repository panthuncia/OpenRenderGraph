# Job graphs: ORG without a main graph

> **Status (2026-10-31).** This began as the design for ORG without one main graph. What exists now:
> - **`org::Runtime`** owns the device, the services, the task service and retirement by timeline points (no frame
>   rotation). Every graph and client on a runtime shares its `ResourceSequencer`, which orders executions across graphs
>   and queues and hands resources between them (sections "Admission at submission" and "Cross-graph ordering").
> - **`org::ExternalClient`** submits work that is not a graph (uploads, readbacks, a presentation copy) in the same order.
> - **`org::GraphInstance`**, **`org::Execution`**, **`org::JobGraph`** and **`org::GraphCache`** run small graphs built
>   from a `GraphDesc` (section "Graph descs and instances"). An app-built graph can be adopted by an instance
>   (BasicRenderer's frame graph).
> - **`org::GraphHost`** stays as the layer for embedding ORG in another renderer: epochs, tickets, external epochs and
>   recording reuse (`persistent-epochs.md`). Community Shaders uses it.
>
> Not built, from the design below: compiling on a worker and sharing compiled graphs between instances by structural
> key; completion callbacks on a runtime thread (handles complete when their graph retires); statistics per execution;
> executions recorded into caller lists outside `GraphHost`'s external epochs. The threading model is a first cut: one
> admission lock in the sequencer, submissions from any thread; it has not had a dedicated review.

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
 GraphDesc ──compile (target: worker, cached by key)───► CompiledGraph (immutable)             
   passes, order edges,                                    schedule, batches, body barrier plans,
   parameters (imports)                                    per-import first/last access, transient layout
                                                                │
                              Bind(params, resolvers, host data)│   (any thread)
                                                                ▼
                    Execution: Prepare ─► Admit ─► Record ─► Submit ─► Complete/Retire
                    ExecutionHandle milestones: Admitted · Submitted · Completed · Poisoned
```

### `org::Runtime`

The runtime replaces the manager singletons' ownership and `InitializeRuntimeDevice`. It owns every service that is
genuinely shared (built: the device, the services, the task service and timeline-based retirement; the per-execution
statistics, the runtime completion thread and the single upload path are design targets):

- **Device resources:** devices and queues, a timeline pool whose ids are globally unique, descriptor heaps, and the
  pipeline lifetime.
- **Upload service:** one service, collapsing `UploadManager`, `UploadInstance` and the copy-queue path.
- **Readback service:** completion callbacks run on a runtime thread; the render thread no longer polls.
- **Statistics:** query ranges are allocated per execution.
- **Settings and the task service.**
- **Retirement:** retirement is by timeline points throughout. The `DeletionManager` frame rotation,
  `ProcessDeferredReleases(frameIndex)` and frame-indexed upload pages are removed.

### `GraphDesc`, `GraphInstance` and `CompiledGraph`

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
one is ready.

*Update (2026-10-31).* `IDynamicDeclaredResources` stays, narrowed. A redeclaration builds on a worker by default
(`PreparesOnlyDeclaredWork`). One with the same content, or with the same structure over existing slots, builds
nothing (the README's "Declarations that change"). Inline builds remain a mechanism the app controls: bootstrap,
`DrainPersistentStructuralBuild`, and passes that opt out.

#### Graph descs and instances (implemented, 2026-10-31)

`OpenRenderGraph/GraphDesc.h` and `OpenRenderGraph/GraphInstance.h` have the public API. The internals are a first
version over the existing graph: an instance wraps a `RenderGraph`.

- **`GraphDesc`** holds passes, in the order added, and named parameters. `Parameter(name, initial)` returns a
  `DynamicResource` wrapper that the passes declare. A desc carries a key that the caller gives. Descs with the same key
  must be interchangeable: their passes take everything that changes per execution from parameters and host data.
- **`GraphInstance::Create(runtime, desc)`** builds the desc's graph, with no `GraphHost`. The owner thread drives it:
  - `Bind(bindings)` sets parameters;
  - `Prepare(hostData)` runs the passes' updates, polls their declarations and installs the edits (a structural one
    builds on a worker), and prepares the invocations and uploads. It returns an **`Execution`**;
  - `Execution::Submit()` submits the uploads, then admits, records and submits the main segment, and returns its
    `ExecutionHandle`. One execution is prepared at a time; one dropped without `Submit` is submitted by its
    destructor, because its preparation already recorded uploads and advanced the passes.
- **Slots, not a ring.** An instance has `framesInFlight` preparation slots. A slot is free once its last execution
  has completed. `Prepare` waits on the CPU only when none is free; `CanPrepare()` tells. Completion is noticed by
  `Retire()` (which never waits) or the next `Prepare`.
- **`CompiledGraph`** is what an instance's executions run: the selected publication's schedule. `Compiled()` returns
  it; binding edits keep it (`SameSchedule`), a structural edit replaces it. Sharing it between instances by key is
  later work (the async compile, plan Phase 5 step 7).
- **Adopted graphs.** `GraphInstance::Adopt(runtime, graph)` takes an app-built `RenderGraph` (BasicRenderer's frame
  graph). The app declares it through `Rebuild(declare)` and drives it with its own frame slots and contexts:
  `Prepare(UpdateExecutionContext)`, then `Execution::Submit(PassExecutionContext)`. It keeps the app's services and frame
  services; an execution dropped unsubmitted is discarded, not submitted. BasicRenderer no longer uses `GraphHost`: its
  frame graph is adopted, and its texture jobs (mips, BC7) and environment jobs run on instances of their own.
- **`JobGraph::Create(runtime, desc)`** is the convenience over an instance: `Execute(bindings, hostData)` binds,
  prepares and submits one execution under a lock. Any thread may call it; one job graph runs one execution at a
  time. `TryRetire()` never blocks, for pollers such as a StateGraph's completion pump.
- **Uploads:** an instance uploads through a service of its own. A shared service resolves its queued uploads against
  the registry of the graph built last, so a job graph built while the frame graph was being built broke the frame's
  uploads.
- **Frame services:** an instance polls no readbacks, begins no statistics frames and takes no captures. The frame
  graph does those.
- **Ordering across graphs** comes from the runtime's sequencer, as for any two graphs on one runtime. A consumer
  executed after a producer waits for it on the GPU, across queues too.
- **`GraphCache`** maps keys to job graphs, so a repeated desc reuses the whole executable, alias plan included.
  Measurements showed the schedule cache alone was not worth much (2 hits in 17).
- **Limit of this version:** creating an instance builds the graph, and the first `Prepare` lowers the passes and
  builds the first schedule inline.
- **Inside `RenderGraph`** the persistent state has three parts (`RenderGraphPersistentState.h`): the instance (passes,
  slot directory, groups, alias pools, edits in flight), the layout (tables built from the selected publication), and
  the executions (admission, the one prepared, the submitted ones). Preparation is `BindPersistentFrame` (polls and
  edits) then `PreparePersistentExecution`; execution is `SubmitPersistentUploads`, then `AdmitPersistentSegment`,
  `RecordPersistentSegment` and `SubmitPersistentSegment`. A ticket's flags are the execution's options.
- Tested in `RebindingGpuTests.cpp` on D3D12 and Vulkan:
  - `RunGraphDescScenario`: a producer job graph on the graphics queue and a consumer on the copy queue share a buffer,
    with no CPU wait between them. They rebind to a second buffer with no rebuild, and the cache returns the job graph
    for a known key.
  - `RunGraphInstanceScenario`: `Prepare` and `Submit` as separate calls, a second `Prepare` refused, the compiled
    graph kept across rebinding, two slots reused with no device wait, a dropped execution submitted.

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

## Admission at submission

This revises items 2, 3, 5, 6 and 8 of the next section, which describe the Phase 0 design.

**The Phase 0 design admits at prepare, against promised exits.** Successors build their waits on points that are not yet
submitted. That buys a little overlap (a successor could record its handoff before its predecessor submits), but it needs:
- a submit gate that never hands an unsubmitted wait to a queue, because on one native queue it deadlocks;
- splice or pass-through for executions that never submit, and poisoning for successors of abandoned writers;
- signal values reserved before submission, on pooled per-execution timelines.

**Revised: an execution is admitted when it submits.**
- **Before admission, nothing touches the ledgers.** Prepare and record run on any thread, for as long as they take and in
  any order: every pass's Prepare, the body recordings (whose barriers only use compiler-known states, item 7) and the
  compiled graph's import tables.
- **Admission.** Under the sequencer lock, Submit:
  1. reads each import cell's exit state and the points of its last writer and readers;
  2. records the per-batch handoff lists (entry and release, item 7), a few barriers each;
  3. submits the handoff lists and the bodies with waits on those points;
  4. commits the execution's exit states and the points it just signalled.
- **What this removes.**
  - No wait-before-signal: a successor's waits name only work already submitted.
  - No submit gate: submission order is admission order, by construction.
  - No promised exits, splice, pass-through or poison. An execution that never submits never touched the ledgers.
- **Ordering.** Jobs that touch the same resource are ordered by the order they submit in. The async graph supplies that
  order: a dependent submits after its producer. The ORGModuleServices adapter gates a dependent's Submit, not its Prepare
  or Record, on its producers' submission, which its GPU submission sets already express. No new Admitted level is needed.
- **Off the render thread.** The handoff calculation is a lookup over the import cells, and it runs on whichever thread
  submits. Neither job's body is re-recorded or interrupted. A job that submits from a worker never blocks the render
  thread for longer than the lock is held.

**Later improvement.** Admission at submission is the simplest correct model, and it is the one to revisit when profiling or
latency asks for it. Things to reconsider:
- A successor cannot record its handoff until its predecessor has submitted. Prepare and body recording still overlap; only
  the handoff list and the submission itself wait.
- One lock serializes every submission on the runtime.
- The Phase 0 promised-exit design (admission at prepare, with a submit gate) buys back that overlap at the cost of the
  machinery listed above. Per-resource locks or optimistic admission (record the handoff outside the lock, then validate
  the ledger version under it and retry on a change) are intermediate options.

**Lock.**
- **What it covers.** One mutex in `ResourceSequencer`, held across steps 1 to 4. That includes the queue submission
  itself, so that a predecessor's commit and its submission are seen together. The work under it is lookups over the
  import cells, recording a handful of barriers, and one `Queue::Submit` per batch.
- **Lock order.** Callers' locks (DXVK's queue lock, for external submission) are taken first, then the sequencer's, then
  BasicRHI's queue and timeline locks. Nothing is called back under the sequencer lock, and ORG never calls into a host's
  submission while holding it.
- **Contention.** Contention is per submission, not per pass. If profiling shows it matters, striped per-resource locks
  taken in sorted order can replace the single mutex, because admission only touches the cells of the execution's
  imports.
- **Closed executions.** Closed executions (every resource returned to its home state) are now an interop option, off by
  default (`persistent-epochs.md`, "Interop options"). Community Shaders runs its epochs open, a graph each, admitted at
  submission against the shared ledgers.

**Implementation order.**
1. **2a:** runtime-unique timeline identities. A queue slot's identity is no longer `slot + 1`, and a foreign timeline's is
   no longer the segment's next number.
2. **2b:** the ledgers move into a runtime `ResourceSequencer` shared by every graph's admission.
3. **2c:** boundary barriers move out of the bodies into per-batch entry lists, with release lists on the predecessor's
   queue. This includes the Vulkan QFOT layout fix and the D3D12 copy-queue `COMMON` exit.
4. **2d:** open admission moves to submission, under the lock.
5. **2e:** external clients (uploads, swapchain).
6. **Tests:** two graphs with write then read across queues on both backends, reader fan-out, and an abandoned execution.

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

## Asset processing jobs gate readiness, not frames

BasicRenderer's mip generation, BC7 compression and environment processing run as job graphs alongside the frame graph.
Their results reach frames through the asset dependency graph, never through a frame-side wait.

**The chain.** Each job gates its asset's readiness:
- A texture is ready once its mip and BC7 jobs have completed on the GPU.
- A material is ready once its textures are.
- Whatever uses the material is ready once the material is.

A frame only uses what has become ready. The ordering comes from the existing dependency system (ORGModuleServices'
`StateGraph`), and no frame waits on async work, on the CPU or the GPU.

**Why not a frame-side gate.** Textures are bindless: materials sample them through descriptors that no pass declares, so
the sequencer cannot order a frame's reads against a job's writes. A frame-side "wait on this signal before these
descriptors" primitive could still be useful for other cases. These jobs don't need it: until an asset is ready, nothing
the frame records can reach it.

**What the job graphs need from ORG.**
- **Concurrency.** Done:
  - `CreateGraphUploadService`: a graph's own upload queue, drained only by its own upload pass;
  - `SetDrivesFrameServices(false)`: no statistics, persistent captures or readback polling;
  - `BufferBase` materialization under a lock.
- **Exit state.** A job graph leaves its textures in the state bindless sampling expects. The ledger hands that state to
  later users.
- **Upload pages.** The job graph's driver retires its private upload pages on its own slot ring
  (`ProcessDeferredReleases(slot)` once the slot's work is complete).

### BasicRenderer: where the gates go

This plan is based on a survey of BasicRenderer, ORG checkout at `org-rebuild`, 2026-10-30.

**Today.**

*Material textures.*
- They are mipped on the CPU (`CreateMaterialResidentPixelBuffer`).
- `MaterialTextureTransferService` uploads them on the graphics queue.
- They reach materials through the `TextureBinding` artifact. `QueueBindingChanged` submits it, and its `AwaitExact(...,
  UploadSubmitted)` callback calls `PublishPreparedImage`.
- Material rows require `TextureBinding` at `UploadSubmitted`.

*Always-resident textures.*
- `CreateAlwaysResidentPixelBuffer` posts mip 0 and enqueues `MipmappingPass`.
- The texture is published at once (`m_publishedImage = m_image`).
- That is correct only because the mip pass runs inside the same frame graph, ahead of its consumers.

*BC7.*
- `SubmitBC7CompressionJob` makes a working texture and enqueues three passes.
- The pass guesses when the source upload has completed by counting four frame executions.
- Results come back through a readback, then `CompleteGpuProcessing` (cache write) and the `Ready` state, and reach
  materials through the binding path.

*Environment.*
- Work goes through `FrameWorkQueue`s, and the prefiltered cubemap is published immediately.
- It has no completion gate at all.

**Plan.**
1. **Texture job graph.** Add a `RenderGraph` of BasicRenderer's own with:
   - `CreateGraphUploadService()`;
   - `SetDrivesFrameServices(false)`;
   - an extension carrying its upload pass and the mip and BC7 passes, removed from `RenderGraphIOService`.

   It executes only when work is pending, at first on the render thread ahead of the frame graph and later on a worker.
   Its rebuilds never touch the frame graph. Each execution yields a `GpuSubmissionSet`, the job graph's queue points as
   in `MakeGpuSubmissionSet`, which is handed to the jobs it ran.
2. **Gate texture readiness on the job's set.** A texture whose mips or BC7 compression are pending is not published:
   - Remove the early `m_publishedImage = m_image` (Texture.cpp `uploadSourceDataThroughFactory`, `AdoptUploadedImage`).
   - Bring those textures through the `TextureBinding` boundary, merging the job's set into the binding's.
   - The binding awaits **`GpuReady`**, not `UploadSubmitted`. The job graph runs on its own queues and timelines, so a
     binding published at submission would still leave the frame to wait on the GPU. At `GpuReady`, no frame waits on a
     job at all.
3. **BC7.** Replace the four-execution source-upload guess with the upload's real completion. The readback completion
   stays the CPU-side gate, as today.
4. **Environment.** One job per environment set (conversion, prefilter, SH), published through an artifact at `GpuReady`
   in place of the `FrameWorkQueue` entries.
5. **Open item.** `MaterialManager::AppendTextureDisplayRequirements` has no caller: display gates may not yet gate
   renderables. Check this before relying on them.
