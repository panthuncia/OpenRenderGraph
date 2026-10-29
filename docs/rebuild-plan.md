# Rebuilding ORG around the persistent compiler

Status: plan, 2026-10-29. Survey baseline: `backend_interop` at 9f2cf0f (CS) and 1e20651 (BasicRenderer).

## Goal

Rebuild ORG so that it has one compiler and one execution model: the persistent compiler. Everything else goes,
apart from the API surface that Community Shaders (CS) and SARP/BasicRenderer (BR) actually use. Any old-compiler
capability that a consumer relies on is ported into the persistent compiler first, multi-backend included. The API
gets better while it shrinks.

## Where things stand

### There are three execution routes

The branch has three routes, all inside one `RenderGraph` class.

| Route | Entered when | Used by |
|---|---|---|
| **Persistent** (`PreparePersistentFrame` / `ExecutePersistentFrame`) | persistent enabled and a task service set | CS (only route, through `PersistentGraphHost`); BR when `SARP_PERSISTENT_GRAPH=1` (the benchmark script's default) |
| **Owned/fresh** (`SubmitOwnedCompileRequest` → `experimental::CompileGraph`, then `TryExecuteSelectedAsyncFrame`) | not persistent, task service set | BR default; `SARP_ASYNC_COMPILE_MODE=2` |
| **True legacy** (`AutoScheduleAndBuildBatches` + the legacy `Execute` body, RenderGraph.cpp:10116-11503) | no task service, compile diagnostics on, or more than one backend device | `ContributorGraphHost`; BR multi-RHI (unconfirmed whether its frames actually execute here) |

The persistent route still enters through the old dispatcher:

- `Update` leads to `CompileFrame`, then `PrepareAndCompileFrame`, which branches at RenderGraphCompile.cpp:2787.
- `Execute` branches at RenderGraph.cpp:10075.

It also leans on old state in several places:

- `m_masterPassList` is built by `CompileStructural`.
- The retained-declaration cache, `RefreshRetainedDeclarationsForFrame`, is read through `LowerLegacyPass`.
- It uses `MaterializeExternalPass`, `alias::Alias*ResourceDesc` and the old alias-settings `std::function`s.

The comment at RenderGraphPersistent.cpp:8 notes that every existing pass is hosted through the legacy adapter.

### What the consumers use

**CS** uses `PersistentGraphHost`, Vulkan over DXVK's adopted device, and nothing else.

- **Host:**
  - async epochs and tickets;
  - closed executions;
  - revision recordings;
  - external epochs and frame slots;
  - queue-scoped and copy-queue streaming uploads;
  - frame-wait timeline;
  - retired and completed callbacks.
- **Passes:** `TypedRenderGraphPass<D, Frame, Bindings>` with `Declare`, `InvocationRevision`, `Prepare` and `Record`, delivered as `ExternalPassDesc` from `IRenderGraphExtension::{PrepareForBuild, GatherStructuralPasses}`.
- **Resources:** `Buffer`, `PixelBuffer`, `ExternalTextureResource`, `VersionedBuffer`, `LatchBlock`, `PreparedTablePublisher`.
- **Services:** upload (staged batches, tracked streaming tickets), descriptors (owned views, slots, cleanup queue), statistics (timestamps, budgets, recording sub-ranges).
- **Debug UI:** `ScheduleInspector`, `MemoryIntrospectionWidget`, `ScheduleTrace` JSON/text, NvPerfCapture, `experimental::DiagnosticSkip*`.
- **Not used:** aliasing, the readback service, resolvers/providers, `Add*Pass`, legacy passes, multi-backend.

**BR / SARP** drive `RenderGraph` directly: `CompileStructural`, `Setup`, `Update`, `Execute` every frame.

- **Passes:** about 139 `TypedRenderGraphPass` uses, including Recipe, lifecycle hooks, `CaptureProgramBinding` and `Update` overrides. 3 `PersistentTypedPass`es.
- **Graph inputs:** providers (11), resolvers, `IDynamicDeclaredResources`, `ResourceGroup`, and extensions with `ExternalInsertPoint` and `SetPassTechnique`.
- **Services:** readback capture and capture passes, upload policy, `StreamingUploadPass`, `UploadInstance`, `FrameWorkQueue`.
- **Aliasing:** `AutoAliasMode` and packing settings, plus the debug snapshot.
- **Debug UI:** `RGInspector` (reads `GetBatches()`), `MemoryIntrospectionWidget` with the frame-graph timeline, VRAM dump.
- **Multi-backend:** multi-RHI peer device via `RegisterBackendDevice`, and `PassBuilder::LegacyInterop` (SSR, upscaling).
- **Legacy passes:** about 13 legacy `ComputePass`/`RenderPass` grass passes in SARP (`src/RendererHost/Grass/GrassRenderExtension.cpp`). Also `SARPInteropValidation`'s raw builders, and `ImmediateCommandList` in BR's `ReadbackManager`.
- **Contributor ABI:** SARP produces the `org_c_*` v3 ABI (`Render/External/ContributorAPI.h`). Nothing hosts it.
- **Backend:** in practice DX12. Vulkan is built and selectable.

**Used by neither:**

- `ContributorGraphHost`, `ContributorRegistry`, `ExtensionRegistry`, both `ContributorRuntime`s, and the `ORG_RG_*` ABI. The only user is `my-community-shaders`, which is out of scope; see Q3.
- `D3D11Interop`, `ExecutionBoundary`, `InteropAllocator`, `ResourceDescriptorIndexHelper`, `ORGUtilities`, `ExternalBufferResource`, `ExternalBackingResource`, `ResourceGroup` (included but dead in ORG).
- The `OpenRenderGraph.h` umbrella as CS sees it (BR includes it 26 times), and `DebugUI.h`.

## Capability gap: old path → persistent

**Port.** A consumer needs these, and the persistent compiler lacks them or has only part:

| # | Capability | Old location | Persistent today | Needed by |
|---|---|---|---|---|
| G1 | **Multi-backend graph**, with these parts: `DeviceRegistry` peer devices; peer queues and imported D3D12↔Vulkan timelines; per-backend representations (committed `InteropAllocator`, shared-heap `SharedAliasPoolManager` placed pool, D3D12 discard init); cross-API ownership transfers (`PlanMultiBackendOwnershipTransfers`); cross-backend batch isolation; `LegacyInterop` builder declarations | RenderGraph.cpp:3721, 6632-7253, 7386-7421; Algorithms.cpp:1424, 1998-2201 | none (the alias planner rejects it, RenderGraphPersistent.cpp:3200) | BR multi-RHI, SSR, upscaling |
| G2 | **Queue activation and async-compute offload**: level width, UAV pressure, `useAsyncCompute` | `PlanActiveQueueSlots` 1689-1934 | every queue active; passes default to graphics | BR (CS pins its queues itself) |
| G3 | **Multi-queue batch scheduler quality**: scored scheduler with graphics bias, overlap bonus and handoff penalty | `AutoScheduleAndBuildBatches`, `IsNewBatchNeeded`, `CommitPassToBatch` | conservative topological walk, first-fit, one wait per batch | BR performance |
| G4 | **Transition placement**: texture same-state elision, read-only uniform elision, early placement (hoist into the last-use batch), graphics-queue fallback | `AddTransition*` 3166-3569 | buffers elided only; barriers only before the pass | both (barrier count) |
| G5 | **Aliasing extras**: BranchAndBound packing, idle pool retirement, the debug snapshot / exclusion reasons; shared multi-backend pools under G1 | Algorithms.cpp | greedy only, never shrinks, logs only | BR |
| G6 | **Idle dematerialization** | `ApplyIdleDematerializationPolicy` 5183 | none | BR memory |
| G7 | **Diagnostics data**: compile profile, VRAM dump, a frame-graph timeline source for memory introspection, an `RGInspector` source | 3940-4112, 1286, Algorithms.cpp:153 | `ScheduleTrace` covers part; `GetBatches()` is empty | BR UI |
| G8 | **Legacy-pass hosting**: either native persistent support for `ComputePass`/`RenderPass`-style passes, or porting the consumers | `pr.pass->Execute` 8642/8982 | rejected ("persistent execution requires PrepareFrame") | SARP grass, `SARPInteropValidation` |

**Already at parity or better in persistent, so nothing to port:**

- cross-frame waits (the admission ledgers);
- fence coalescing;
- alias activation and alias queue sync;
- resolvers (redesigned);
- frame extensions (Tail);
- immediate replay in Pre;
- readback and streaming typed passes;
- Tracy, NvPerf and statistics;
- copy-queue uploads;
- GraphReplay.

**Drop without replacement:**

- the owned/fresh route and `GraphCompileCoordinator` (see Q2);
- `FramePlanning` / `FrameWorker` / `FrameContext` and the owned-path half of `FrameRecording`;
- `CommandRecordingManager`, `ExecutionSchedule.h` and the legacy `PassBatch`.

## Target architecture

### One entry point: `GraphHost`

- Generalize `PersistentGraphHost` into the only way to drive a graph. BR moves onto it.
- A graph with no epoch tags is a one-epoch frame, so `ExecuteFrame()` covers BR's loop.
- `RenderGraph::Update` / `Execute` / `CompileFrame` / `Setup` stop being public.
- The host owns what BR currently does by hand:
  - structural rebuild (providers, resolvers, extensions, `CompileStructural`);
  - the settings service;
  - the task service, which becomes mandatory. This removes the silent "no task service" fallback.

### One compiler

`GraphCompiler`, renamed from `experimental::CompileWorkspace`, is fed by a native lowering of `PassDeclaration`.

- The legacy adapter (`LowerLegacyPass`) and the retained-declaration cache are replaced by declarations captured once at structural build.
- Scheduling (G2, G3), transition planning (G4) and alias placement (G5) become pure stages over the compile input. This is the "pure compiler" target that `experimental-async-compile.md` already describes.

### Multi-backend inside the persistent model (G1)

- **Backend instances.** A program carries a set of backend instances. Each resource binding names the instances that touch it.
- **Representations at bind time.** Representations are materialized when the binding is made, through `InteropAllocator` or a shared placed pool. They are versioned like any backing, so backing polling and revision checks cover them.
- **Cross-API transfers.** Ownership transfers are planned at compile time as explicit release/acquire steps on the boundary between batches. The admission ledger tracks one state per backend instance and imported timeline points.
- **Batches stay on one backend.** That is the old batch isolation, and it falls out naturally because each persistent batch is already single-queue.
- **Aliasing.** Placement across backends uses shared-heap pools. Persistent aliasing of multi-backend resources stays off unless the pool supports it.

### Services owned per runtime, not singletons

- A `Runtime` (today's `RuntimeDevice`) owns one each of the device, descriptor heaps, uploads, readback, statistics, deletion and the ECS world.
- `Default*Service` facades and `GetInstance()` go.
- Thread-local "active service" access stays only as a scoped convenience: passes get services from their context.
- The upload stack collapses into one `UploadService` that owns `UploadInstance` and the copy queue. Its interface does not vary with `BUILD_TYPE`.

### Header split

- `include/OpenRenderGraph/` holds the public API: Host, Pass, Builder, Resources, Services, DebugUI.
- Compiler, admission, scheduling and aliasing internals move to `src/`.
- `RenderGraph.h` becomes a pimpl, because about 60 nested implementation structs currently leak from it.

## Phases

Each phase lands on its own branch off `backend_interop` and ends green on the gates listed under it. **Consumers move
in the same phase as the API they depend on.** CS and BR are bumped together, so neither is ever left pinned to a
dead ORG.

### Phase 0: freeze baselines and the safety net

- Fix the CMake defects:
  - duplicate sources (CMakeLists.txt:105-117);
  - the `GPUBufferBacking.cpp` case mismatch;
  - `MemoryIntrospectionWidget` compiled into two targets;
  - the `BUILD_DEBUG` typo;
  - the shared `OPEN_RENDER_GRAPH_CONTRIBUTOR_API_H` include guard.
- Add a headless persistent-vs-owned parity test over the graph-replay fixtures. It compares pass order, batch and queue assignment, barrier counts and the alias footprint.
- Record baselines:
  - CS: tour frame report and VRAM;
  - BR: `scripts/sarp_mo2_scene_benchmark.cmd` with persistent on and off, frame time, barrier and batch counts, VRAM.

**Gate:** both consumers build on the fixed ORG. The baselines are committed under `docs/baselines/`.

### Phase 1: delete what no consumer uses

- **Remove:**
  - `D3D11Interop`;
  - `ContributorGraphHost`, `ContributorRegistry`, `ExtensionRegistry`, both contributor runtimes and the `ORG_RG_*` header (pending Q3);
  - `ExecutionBoundary`, `StreamingUploadPass` if BR's two includes turn out to be dead, `ResourceDescriptorIndexHelper`, `ORGUtilities` from `include/`, `ResourceGroup` (BR's 104 uses are `org::ResourceGroup`; verify they're live, otherwise this item moves to Phase 3), the `OpenRenderGraph.h`/`DebugUI.h` umbrellas, and `ExternalBufferResource`/`ExternalBackingResource` (the latter only if it is not on the resize path).
- **Keep** the `org_c_*` v3 header as a standalone C header. SARP's producer and ABI audit use it.
- **Move** the `SARP_*` env vars read inside ORG to `ORG_*` settings fields.

**Gate:** CS and BR build and their baselines are unchanged.

### Phase 2: native persistent lowering

- Lower `TypedRenderGraphPass` / `PersistentTypedPass` declarations into `persistent::PassDeclaration` at structural build, replacing `LowerLegacyPass` and the retained-declaration cache.
- Move resolver requirement capture into persistent resolver groups. Most of this already exists, at RenderGraphPersistent.cpp:1348-1460.
- Give `PersistentGraph` its own pass storage instead of `m_masterPassList`.
- Make the task service mandatory.

**Gate:** CS tour and BR persistent benchmark at parity. The parity test passes.

### Phase 3: bring BR onto persistent-only

- Port SARP's grass passes (G8) and `SARPInteropValidation` to `TypedRenderGraphPass`.
- Replace `ImmediateCommandList` use in BR's `ReadbackManager` with a typed pass or a host-recorded Pre segment.
- Port G6 (idle dematerialization) and the parts of G5 BR uses (BranchAndBound behind the existing strategy setting, pool retirement).
- Port G7:
  - rebase `RGInspector` and the memory frame-graph timeline on `ScheduleTrace` / persistent schedule snapshots;
  - port the compile profile and VRAM dump.
- Port G4. Texture elision and early placement benefit CS too.
- Flip BR to persistent by default and remove `SARP_PERSISTENT_GRAPH`. The multi-RHI build stays on the old path until Phase 4.

**Gate:** the BR benchmark with persistent is within noise of the Phase 0 owned-path baseline for frame time, and its barrier count and VRAM are no worse.

### Phase 4: multi-backend in persistent (G1) and scheduler quality (G2, G3)

- Implement the multi-backend design above. Enable BR multi-RHI and `LegacyInterop` (renamed `CrossBackend`) on it.
- Port queue activation and the scored scheduler as a pure stage. Keep the conservative scheduler as the reference for the validator.

**Gate:** the BR multi-RHI scene renders correctly. Cross-API validation (`SARPInteropValidation`) passes under GPU-based validation on both APIs. Async-compute overlap is at least the old scheduler's on the benchmark.

### Phase 5: delete the old compiler

- **Remove** the true-legacy and owned routes:
  - RenderGraph.cpp: about 8.1k old-only lines plus the 525-line owned path;
  - RenderGraphCompile.cpp: about 3.5k old-only lines plus 1k owned;
  - the `PassBatch` and old state in RenderGraph.h;
  - the old aliasing subsystem, apart from the desc helpers, which move into persistent aliasing;
  - `CommandRecordingManager`, `FramePlanning`, `FrameWorker`, `FrameContext`, `GraphCompileCoordinator` (Q2), `RenderGraphCompileProfile` once G7 is ported, and `ExecutionSchedule.h`.
- `ComputePass`/`RenderPass`/`CopyPass` and the three builder classes go if Phase 3 removed every user. Otherwise they stay as thin wrappers that lower to typed passes.

**Gate:** everything builds, tests pass, both consumers are at their baselines, and ORG is about 40-45k lines (from 86k).

### Phase 6: API and code quality

- Rename the `Experimental*` files and `org::experimental` to their real roles: `GraphCompiler`, `ExecutionState`, `RhiExecution`. Keep `DiagnosticSkip*` as `org::diagnostics`.
- Make services per-runtime, collapse the upload stack, and remove the singletons.
- Split the code:
  - `RenderGraph.cpp` into ingestion, structural build and host glue;
  - `PassBuilders.h` into one header per builder over a CRTP base. This removes the 60 duplicated `&`/`&&` overloads.
- Make `RenderGraph.h` a pimpl and move the internals to `src/`.
- Replace the custom-`main` tests with one test framework and register `HeadlessGraphBenchmark`.
- Rewrite the README and docs around `GraphHost`. Fold `experimental-async-compile.md` and `async-frame-implementation.md` into one design doc, and drop the history.

**Gate:** both consumers build against only `include/OpenRenderGraph/**`. No public header includes a compiler internal.

## Verification across all phases

- **ORG tests:**
  - the persistent graph and Vulkan host tests;
  - the new parity test (owned vs persistent until Phase 5, then persistent vs committed golden schedules);
  - the async compile tests, retargeted to the compiler.
- **CS:** build plus a 60 s in-game run (Whiterun, Dragonsreach) with the DCLF frame report, VRAM ledger dump and screenshots. Compare against Phase 0.
- **BR / SARP:** `scripts/sarp_mo2_scene_benchmark.cmd` plus the renderer host's validation scripts; DX12 and Vulkan; multi-RHI from Phase 4.

## Decisions needed

- **Q1, multi-backend scope.** SARP runs DX12 in practice. Is BR's multi-RHI peer device (G1) a real requirement, or can it be dropped instead of ported? Phase 4 is the most expensive phase.
- **Q2, async structural compile.** BR's `SARP_ASYNC_COMPILE_MODE` compiles off-thread every frame. Persistent compiles only on rebuild. Is off-thread compile on rebuild worth keeping, as a `GraphCompiler` job, or does it go?
- **Q3, contributor ABIs.** CS doesn't use `ORG_RG_*` / `ContributorGraphHost`; only `my-community-shaders` does. Delete them, or move them to that consumer? And does `org_c_*` v3 stay in ORG as a header only, or move to SARP?
- **Q4, legacy pass bases.** Should Phase 3 port SARP's grass passes so the bases can be deleted, or should ORG keep `ComputePass`/`RenderPass` as typed-lowering wrappers?
