# Rebuild ORG around the persistent compiler (revised)

> **Resumed (2026-10-30).** Phase 5 is committed and pushed (ORG `org-rebuild`, BasicRenderer `dev`, SARP `org-rebuild`,
> CS `org_integration`); its CS tour is still owed. The build-speed work (header firewall, persistent split) landed on top.
> Phase 5's CS tour passed (Dev-Fast build: same stops, no new errors, equivalent screenshots, VRAM within 300 MiB).
> Phase 6 implemented (uncommitted while the CS host and in-game check finish):
> - `ContributorAPI.h` v2 (ORGTypes.h, prepare/record packets, epochs, exports, pipeline service, render-relative buffers,
>   namespace = the contributor id's first segment); `ContributorHost` (src/Contributor: host, BorrowedRecorder, DeviceTable);
>   v1 runtime, ExtensionRegistry and the ComputePass/CopyPass bases and builders deleted (ExternalPassDesc has one `pass`).
> - Tests: registry, C header, end to end (PersistentVulkanHostTests contributor mode), samples/ContributorSample.cpp;
>   SARP's ProducerCore, diagnostic contributor, loader test, C smoke and ABI audit ported.
> - CS: ContributorHost beside GraphHost (epochs cs.<segment>, anchors cs.frame.begin/end, no exports yet), the
>   ORG_GetRenderGraphAPI export, CS_ORG_TEST_CONTRIBUTOR loader for the sample.
> Known: OpenRenderGraphTrackingWorldTests fails ~40% (ECS tracking-world race), unrelated to the rebuild.
> Next: Phase 7, then 8.
>
> **Earlier deferral (2026-10-30)** for the build-speed plan (`sarp-build-speed.md`). State when paused:
> - Phase 5 is uncommitted in every repo: ORG at the CS copy and mirrored to BR, plus BR, SARP and CS.
> - In progress: the resolver-group redesign in `RenderGraphPersistent.cpp`, `PersistentLowering.cpp`, `PassBuilders.h`, `IResourceResolver.h` and `RenderGraph.h/.cpp`:
>   - `ResolverCaptureScope` makes one capture per declaration, and `ResolverSnapshot::state` holds it;
>   - `SyncGroupMembers`/`GrowGroup` are shared by install and poll;
>   - lineage (`memberReplacedIDs`, `MemberPosition`), forward sync at install (`DeclarationIsAhead`);
>   - `memberOrdinal` and `memberResourceIDs` are removed.
>   ORG ctest passes (except the TrackingWorld flake). Not yet validated in SARP: the last build failed in `HierarchicalCullingPass1` ("unbound persistent slot"), and the fix for that, forward sync instead of add-only admission, is built but unrun.
> - Temporary edit to revert: SARP `scripts/sarp_mo2_task.ps1` has a `SARP_CDB_SCRIPT` override.
> - Still owed for Phase 5: grass ×3, DX12 and multi-RHI benchmarks, the CS tour, then the commits.

## Context

OpenRenderGraph (ORG) runs three execution routes inside one `RenderGraph` class:

| Route | Entered through | Used by |
|---|---|---|
| **Persistent** | `PreparePersistentFrame` / `ExecutePersistentFrame` | CS, through `PersistentGraphHost` (its only route); BasicRenderer when `SARP_PERSISTENT_GRAPH=1` |
| **Owned/fresh** | `SubmitOwnedCompileRequest` → `experimental::CompileGraph` | BasicRenderer, by default |
| **True legacy** | `AutoScheduleAndBuildBatches` plus the legacy `Execute` | `ContributorGraphHost`; any graph with no task service, with compile diagnostics on, or with more than one backend device |

The branch (`backend_interop`, 86k lines) has also collected:
- two half-finished contributor ABIs;
- singleton-backed service facades;
- leaked internals in `include/`;
- build defects.

**Goal:** one compiler (persistent) and one host. Keep only the surface CS and SARP/BasicRenderer (BR) use. Port every old-path capability they need into the persistent compiler, multi-RHI included. Improve the API as it shrinks.

**Decisions (from the user):**
- Multi-RHI is required and gets ported.
- `SARP_ASYNC_COMPILE_MODE` and the owned/async compile route are deleted. SARP later moves to DCLF-style async epochs.
- SARP's grass passes are ported, and the legacy pass bases are deleted.
- The contributor API is kept in spirit and rebuilt persistent-native. It is the seed of a CS plugin API (findings below).
- SARP's `tesworld` branch is merged before any SARP change.
- SARP validation always runs the preprocess script first.

## Surface to keep

**CS uses `PersistentGraphHost` only (Vulkan over DXVK):**
- **Host features:**
  - async epochs and tickets;
  - closed executions;
  - revision recordings;
  - external epochs and frame slots;
  - queue-scoped and copy-queue streaming uploads;
  - frame-wait timeline;
  - retired and completed callbacks.
- **Passes:** `TypedRenderGraphPass<D, Frame, Bindings>` delivered as `ExternalPassDesc` from `IRenderGraphExtension::{PrepareForBuild, GatherStructuralPasses}`.
- **Resources:** `Buffer`, `PixelBuffer`, `ExternalTextureResource`, `VersionedBuffer`, `LatchBlock`, `PreparedTablePublisher`.
- **Services:** upload, descriptor and statistics.
- **Debug:** `ScheduleInspector`, `ScheduleTrace`, `MemoryIntrospectionWidget`, `NvPerfCapture`, `DiagnosticSkip*`.
- **Also needed for CS:** ORGModuleServices needs `Buffer`, `VersionedBuffer`, `IUploadService`, `StreamingUploadTypes`, `LatchBlock` and `ShaderCompilerService.h`.

**BR/SARP:**
- **Driving:** BR drives `RenderGraph` directly: `CompileStructural`, `Setup`, `Update`, `Execute`.
- **Passes:** about 139 typed passes, using lifecycle hooks and `Update` overrides. Recipes, `CaptureProgramBinding` and the 3 `PersistentTypedPass`es are gone: a pass's `Revision` drives kept-recording reuse only, and `Record` binds pipelines (`RecordingContext::BindComputePipeline`/`BindGraphicsPipeline`, retained by the recording) and resolves declared views and resources at record time.
- **Graph inputs:** providers, resolvers, `IDynamicDeclaredResources`, `ResourceGroup`, and extensions with `ExternalInsertPoint`/`SetPassTechnique`.
- **Readback:** the readback service and capture passes.
- **Uploads:** upload policy, `StreamingUploadPass`, `UploadInstance`, `FrameWorkQueue`.
- **Aliasing:** `AutoAliasMode`/packing and the alias debug snapshot.
- **Debug and diagnostics:** `RGInspector`, the memory frame-graph timeline, the VRAM dump.
- **Multi-RHI:** a peer device via `RegisterBackendDevice`, and `LegacyInterop` (used by SSR and upscaling).
- **Legacy code to port:** SARP's 13 legacy grass passes, `SARPInteropValidation` (which uses the raw builders), and the `ImmediateCommandList` in BR's `ReadbackManager`.

**Used by neither consumer (delete):**
- `D3D11Interop`, `ExecutionBoundary`, `InteropAllocator`, which is re-homed under multi-RHI only if the port needs it.
- `ResourceDescriptorIndexHelper` (deleted: descriptor slots are filled from declarations at bind), `ORGUtilities` in `include/`, the `OpenRenderGraph.h`/`DebugUI.h` umbrellas for CS, and `ExternalBufferResource`.

## Gaps to port into the persistent compiler

| # | Capability | Old location (ORG `src/Render/RenderGraph/`) | Needed by |
|---|---|---|---|
| G1 | **Multi-RHI:**<ul><li>`DeviceRegistry` peer devices</li><li>peer queues and imported D3D12↔Vulkan timelines (`Setup` RenderGraph.cpp:7386-7421, `QueueRegistry::EnableD3D12VulkanInterop`)</li><li>per-backend representations (`MaterializeMultiBackendRepresentations` 6632-6991: committed `InteropAllocator` path, `SharedAliasPoolManager` placed pool, D3D12 discard init)</li><li>cross-API ownership transfers (`PlanMultiBackendOwnershipTransfers` 6993-7229)</li><li>batch isolation</li><li>`LegacyInterop` declarations</li></ul> | as listed | BR |
| G2 | Queue activation and async-compute offload | `PlanActiveQueueSlots` 1689-1934 | BR |
| G3 | Scored multi-queue scheduler | `AutoScheduleAndBuildBatches`, `IsNewBatchNeeded`, `CommitPassToBatch` | BR performance |
| G4 | Texture same-state elision, read-only uniform elision, early placement, graphics-queue fallback | `AddTransition*` 3166-3569 | both |
| G5 | BranchAndBound packing, idle pool retirement, alias debug snapshot | Aliasing/RenderGraphAliasingAlgorithms.cpp | BR |
| G6 | Idle dematerialization | `ApplyIdleDematerializationPolicy` 5183 | BR |
| G7 | Compile profile, VRAM dump, memory frame-graph timeline source, `RGInspector` source | RenderGraph.cpp:1286, 3940-4112; Algorithms.cpp:153 | BR UI |

Persistent already matches or exceeds the old path on the rest: admission ledgers, fence coalescing, alias activation and sync, resolvers, Tail and Pre segments, Tracy/NvPerf/statistics, copy-queue uploads, GraphReplay.

## Contributor API findings

Neither ABI is live:
- **SARP's `ProducerCore` (v1 `org_c_`)** declares 21 resources and 2 compute marker passes, but nothing hosts it.
- **`my-community-shaders` (v2 `ORG_RG_`)** hosts v2 for D3D11 + D3D12 only through sample DLLs; no real plugin uses it.
- **The live CS** uses neither.

Each ABI got half the design right:
- **v1 has the right per-frame shape.** Prepare returns an owned, immutable packet; record gets a borrowed, token-validated recorder; it has submitted/completed/abandoned hooks. That maps 1:1 onto `TypedRenderGraphPass` Prepare/Record. Its host still builds legacy passes, though (`src/Render/External/ContributorRuntime.cpp:438-646`).
- **v2 has the right structure.** `ContributorRegistry` gives namespaced IDs, required/optional/diagnostic pruning, generation activation and retirement at completion, async unregister, rich access/view descriptions with exact views made by the host, `QueueBufferUpload`, `QueryService` and diagnostics. But a contributor can't record commands, and `ContributorGraphHost` runs the legacy Update/Execute.

**So the spirit is worth keeping:** build one new ABI from v1's packet and recorder plus v2's registry and descriptions, hosted as one persistent extension. Phase 6 below covers it.

## Phases

Each phase lands on its own ORG branch off `backend_interop`. CS and BR bump their ORG pin in the same phase as the API they depend on, so no consumer is left on a dead ORG. Every phase ends with the gates listed under Verification.

### Phase 0: SARP prep, ORG safety net, baselines

1. **SARP:**
   - Set aside the uncommitted work on `main` (`GrassRuntime.cpp`, `scripts/community_shaders_mo2_launch.ps1`, the BasicRenderer submodule pointer) as a WIP commit on a side branch. No bare stash.
   - `git fetch origin tesworld`, then merge `origin/tesworld` into SARP `main`. Resolve conflicts, update submodules (tesworld may move BasicRenderer), then re-apply the WIP if it's still wanted.
   - Build the `vs2026-renderer-host` preset, deploy, then run `scripts\sarp_mo2_preprocess.cmd` (long: run it in the background).
2. **ORG fixes:**
   - duplicate CMake sources (CMakeLists.txt:105-117);
   - the `GPUBufferBacking.cpp` case mismatch;
   - `MemoryIntrospectionWidget` compiled into both DebugUI and MemoryUI;
   - the `BUILD_DEBUG` typo in the GPUBacking headers;
   - the shared `OPEN_RENDER_GRAPH_CONTRIBUTOR_API_H` guard.
3. **Parity test:** a headless test (owned vs persistent) over `tests/fixtures/*.orggraph`. It compares pass order, queue and batch assignment, barrier counts and alias footprint.
4. **Baselines**, recorded under `docs/baselines/`:
   - **BR:** `scripts\sarp_mo2_scene_benchmark.cmd` with persistent on and off, and on DX12, Vulkan and multi-RHI. Capture the `GetBenchmarkSnapshot` counts (they must match exactly later), timings, barrier and batch counts, and VRAM.
   - **CS:** tour frame report, VRAM ledger and screenshots.

### Phase 1: delete unused code and async compile

- Delete the unused items listed above.
- Delete `SARP_ASYNC_COMPILE_MODE` (BR `RendererLifecycle.cpp:337-343`) and `experimentalAsyncCompileMode`/`experimentalCompileConcurrency`.
- Delete `GraphCompileCoordinator` (ExperimentalGraphCompiler.cpp:1088-1340) and `PrepareAsyncFrame`.
- Move the `SARP_*` env vars read inside ORG into `ORG_*` settings fields.
- Keep the contributor code until Phase 6 has replaced it.

### Phase 2: native persistent lowering

- Lower typed pass declarations straight into persistent graph declarations (`GraphEditTransaction::Declare`/`DeclareView`) at structural build. This replaces `LowerLegacyPass` (RenderGraphPersistent.cpp:657-776) and the retained-declaration cache it reads.
- Move resolver requirement capture fully into persistent resolver groups (RenderGraphPersistent.cpp:1348-1460).
- Give `PersistentGraph` its own pass storage instead of `m_masterPassList`.
- Make the task service mandatory, which removes the silent fallback at RenderGraphCompile.cpp:2787.

### Phase 3: BR/SARP onto persistent-only (single backend)

- **Port off the legacy pass bases** (`ComputePass`/`RenderPass` with `DeclareResourceUsages`/`Setup`/`Update`/`Execute`) to `TypedRenderGraphPass` (Declare/Prepare/Record), using BR's existing typed passes as the model:
  - SARP's grass passes (`src/RendererHost/Grass/GrassRenderExtension.cpp`);
  - `SARPInteropValidation` (moves to typed builders; its multi-RHI checks re-enable in Phase 4).
- Replace the `ImmediateCommandList` use in BR's `ReadbackManager` with a typed pass.
- Port G4, G5, G6 and G7. Rebase `RGInspector` and the memory frame-graph timeline on persistent schedule snapshots (`ScheduleTrace`).
- Make persistent the only BR mode and remove `SARP_PERSISTENT_GRAPH`. Multi-RHI builds keep running in this phase on the owned/legacy route, which is deleted in Phase 5.

### Phase 4: multi-RHI and scheduler in persistent (G1–G3)

- A program carries a set of backend instances, and each binding names the instances that touch it.
- Representations are made at bind time and versioned like any backing, so backing polling and revisions cover them. Reuse the representation code from `MaterializeMultiBackendRepresentations`, `InteropAllocator` and `SharedAliasPoolManager`.
- Cross-API transfers become compile-time release/acquire steps on batch boundaries. The admission ledgers keep one state per backend instance, plus imported timeline points.
- Batches stay single-backend; persistent batches are already single-queue.
- Alias placement across backends uses shared-heap pools only.
- Rename `LegacyInterop` to `CrossBackend`.
- Port G2 and G3 as pure compile stages. Keep the conservative scheduler as the validator's reference.

### Phase 5: delete the old compiler

Remove the true-legacy and owned routes:
- RenderGraph.cpp: about 8.6k lines;
- RenderGraphCompile.cpp: about 4.5k lines;
- `PassBatch` and the old state in RenderGraph.h;
- the old aliasing subsystem, keeping only the `alias::Alias*ResourceDesc` helpers;
- `CommandRecordingManager`, `FramePlanning`, `FrameWorker`, `FrameContext`, the owned half of `FrameRecording`, `RenderGraphCompileProfile` and `ExecutionSchedule.h`;
- the `ComputePass`/`RenderPass`/`CopyPass` bases and their three builders.

Make `GraphHost` (generalized from `PersistentGraphHost`) the only public way to drive a graph:
- An untagged graph is a single-epoch frame, so BR calls `ExecuteFrame()`.
- The host owns structural rebuild: providers, resolvers, extensions.
- `Update`/`Execute`/`Setup`/`CompileFrame` become private.

Target size: about 40–45k lines.

### Phase 6: persistent-native contributor ABI (seed of the CS plugin API)

1. **One C header:** `include/OpenRenderGraph/ContributorAPI.h`, with one prefix, one include guard and versioned `structSize`/`apiVersion`.
   - Common `ORGStatus`/`ORGDiagnostic`/call-convention go in a small shared header, so `ShaderCompilerService.h` and ORGModuleServices' `ShaderServiceAdapter` keep compiling.
   - **From v2:** registry, resource and access descriptions, `ORGFormat` ordinals, diagnostics, `QueueBufferUpload` and `QueryService`.
   - **From v1:** the prepare packet, the borrowed recorder and the lifecycle hooks.
   - **New:**
     - an `invocation_revision` callback, so async-epoch tickets can reuse invocations; it is exclusive with the lifecycle hooks, per `TypedRenderGraphPass.h:120`;
     - epoch and insert-point naming;
     - a host resource export table (symbolic ID, format, extent, allowed accesses, valid epochs, read/write);
     - a pipeline service on top of `ORGShaderCompilerServiceAPI`;
     - a per-frame host snapshot in the prepare context.
2. **Host side:** one `ContributorPass : TypedRenderGraphPass<…>` adapter.
   - Declare comes from the access list; Prepare wraps the packet and captures exact views; Record hands over the recorder.
   - It is emitted by one `IRenderGraphExtension` registered with `GraphHost::AddExtension`.
   - Generations retire on the host's completed and retired callbacks.
   - Reuse `ContributorRegistry`'s validation, pruning and generation logic (`src/Contributor/ContributorRegistry.cpp`) and `ExtensionRegistry`'s export/import resolution, unified into one namespace with the native exports.
   - The exact-view creation code moves over from `ContributorGraphHost.cpp:305-451`.
3. **Delete:** `ContributorGraphHost`, both `ContributorRuntime`s, `Render/External/*`, the v2 Update/Execute callbacks, and the anchor and boundary passes.
4. **Tests:**
   - port SARP's `tests/ExternalContributor/DiagnosticContributor.cpp` and `ContributorLoaderTests.cpp`;
   - port `CHeaderSmoke.c`, ORG's `ContributorHeaderCompile.c` and `ContributorRegistryTests.cpp`;
   - update SARP's `ContributorAbiAudit.cmake` to the new header and forbidden names;
   - add a cross-contributor resource test modelled on the `my-community-shaders` Producer/Consumer samples.
5. **Consumers:**
   - Port SARP's `ProducerCore` to the new ABI: exports, plus marker passes in a named epoch.
   - On the CS side, a minimal host: export `ORG_GetRenderGraphAPI` from CommunityShaders.dll (push model) and publish CS epoch IDs (`DCLF/Schedule/Frame.cpp:154-166`) and the `diag.*` anchors under `cs.*` names.
   - Defer a game-target export table beyond a smoke set (depth and colour imported through `ImportImage`, `DCLF/Core/Render/Passes/Common.cpp:22-53`) to a follow-up CS plugin-API project.
   - `my-community-shaders` is not migrated.

**Implementation design (2026-10-30):**
- `OpenRenderGraph/ORGTypes.h`: `ORG_RG_CALL`, `ORGStatus`, `ORGBackend`, `ORGFormat`, handles. `ShaderCompilerService.h`
  and ORGModuleServices include it instead of the contributor header.
- `OpenRenderGraph/ContributorAPI.h`, `ORG_RENDER_GRAPH_API_VERSION_2`, includes `rhi_c.h` for the recorder and the device
  table. Pass callbacks: `prepare` returns an owned `ORGPreparedPass` packet (record, submitted, completed, abandoned,
  destroy); `invocationRevision` optional and exclusive with the lifecycle hooks. Prepare and record contexts carry the
  frame, the bindings (exact views captured by the host: descriptor index, resource handle, view handle) and the host's
  per-frame snapshot (type id, pointer, size).
- Host tables: anchors and epochs by name (`GetAnchor`, `GetEpoch`), exports (`GetExport`: id, dimension, format, extent,
  allowed accesses, readable/writable). Pass accesses name a contributor resource or an export.
- `org.pipelines` service (`QueryService`): the host's `basicrhi_c_device_api_v1` (implemented over `rhi::Device`) and its
  bindless pipeline layout.
- Host side, `src/Contributor/`: `ContributorHost` owns the registry, compiles a candidate per build, and contributes
  `ContributorPass` (a `TypedRenderGraphPass` adapter) through one extension (`GraphHost::AddExtension`). Generations
  activate on install and retire on the host's retired callback.

### Phase 7: API and code quality

- **Naming:** rename `Experimental*` and `org::experimental` to `GraphCompiler`, `ExecutionState`, `RhiExecution`, and `org::diagnostics` for `DiagnosticSkip*`.
- **Services:**
  - Services become per-runtime and owned by `RuntimeDevice`, which removes the `GetInstance()` singletons and `Default*Service` facades.
  - Collapse the upload stack (`DefaultUploadService` → `UploadManager` → `UploadInstance`/`CopyQueueUploadService`) into one `UploadService`, with no `BUILD_TYPE`-dependent vtable.
- **Code layout:**
  - Split `RenderGraph.cpp`.
  - Split `PassBuilders.h` into one header per builder over a CRTP base.
  - Make `RenderGraph.h` a pimpl and move compiler, admission and aliasing internals to `src/`.
- **Tests:** one test framework; register `HeadlessGraphBenchmark`.
- **Docs:** rewrite the README and docs around `GraphHost`, and replace `extern/OpenRenderGraph/docs/rebuild-plan.md` with this plan once it's approved.

### Phase 8: per-pass resource readback capture

> **Status (2026-10-30): implemented.** A capture anchored after a main pass is appended to that pass's packet
> (`PreparedPass::WithTrailer`: barrier to copy source from the pass's final compiled state, copy, barrier back), so the
> executable is not rebuilt (`RenderGraphPersistentCaptures.cpp`); a pass that does not bind the resource, or not over the
> whole range, falls back to an interrupting capture pass next frame, and after-graph captures run at the end. The graph
> consumes the readback service's requests itself (BasicRenderer's `ReadbackCaptureExtension` is gone). The schedule
> inspector takes a capture callback (`ScheduleCaptureFn`); requests by scheduling ID
> (`IReadbackService::RequestScheduledReadbackCapture`) resolve on the render thread. Viewer: BR opens `MemoryViewWidget`;
> CS has a byte/u32/f32 view. Test: `PersistentVulkanHostTests` sync mode captures after `test.write` with the structural
> revision unchanged.

Phase 5 deleted the old `RGInspector` (`DebugUI/RenderGraphInspector`), which read the old compiler's `PassBatch` list. With it went per-pass resource readback capture: pick a pass in the inspector, pick a resource it touches, and see that resource's contents as they were right after the pass. Bring this forward onto the persistent schedule.

What still exists:
- `IReadbackService::RequestReadbackCapture(passName, resource, range, callback)`, `ReadbackCapturePass` and `ReadbackCaptureTests`.
- Persistent hosting of frame-interrupting passes (`UpdatePersistentFrameInterruptingPasses`): an inserted pass observes resources at its insert point.
- `ScheduleInspector` and its `PersistentScheduleSnapshot`, used by both CS and BR.

Work:
1. **Capture on the persistent route.** A capture requested for a pass inserts a `ReadbackCapturePass` right after that pass for one frame, through the frame-interrupting path. The insert must not force a structural rebuild of the main executable, and the capture must see the pass's output state. Cover textures (any mip or slice) and buffers, and resources a resolver group provides as well as direct ones.
2. **Inspector UI.** `ScheduleInspector` lists, for the selected pass, the resources it declares, with access, state and range from the snapshot. It offers a capture action through an optional host callback, so the inspector itself stays free of service dependencies.
3. **Viewer.** Port the old inspector's preview: textures shown through the host's ImGui descriptor allocator (the callbacks `RGInspectorOptions` took), with mip and slice selection and basic channel and range controls. Buffers shown as a hex and typed-value dump.
4. **Consumers.** BR wires the callback to its readback service in the RG Inspector window. CS wires it in its schedule window, if CS's readback path supports it; otherwise it states in the UI that capture is unavailable.
5. **Tests.** Extend `ReadbackCaptureTests` with a persistent graph: capture after a mid-frame pass that writes a known pattern, and check the bytes and that the main executable was not rebuilt.

## Verification (every phase)

- **ORG:**
  - `OpenRenderGraph` tests: PersistentGraph, the persistent Vulkan host, the parity test (against golden schedules after Phase 5), and the contributor tests (Phase 6);
  - shader and Vulkan validation layers on the Vulkan host tests.
- **CS:**
  - `cmake --build build/ALL --config Release --target CommunityShaders`, then deploy;
  - one 60 s MO2 run (Whiterun or Dragonsreach), with the DCLF frame report, VRAM ledger and screenshots compared to the Phase 0 baseline.
- **SARP/BR:**
  - Build the renderer host and run ctest, including `SARPSourceAudit` and `SARPContributorAbiAudit`.
  - **Then always run `scripts\sarp_mo2_preprocess.cmd` first** (long, in the background), followed by `scripts\sarp_mo2_scene_benchmark.cmd`. `GetBenchmarkSnapshot` counts must match the baseline exactly; frame time must be within noise; barrier count and VRAM must be no worse.
  - Run on DX12 and Vulkan from Phase 3, and on multi-RHI with `SARPInteropValidation` under GPU-based validation from Phase 4.
- **Phase 6:** a sample contributor DLL draws through CS's host in-game, and unregistering it mid-run retires it cleanly.
- **Phase 8:** in BR (and CS if supported), capturing a texture and a buffer after a mid-frame pass in-game shows the expected contents, with no structural rebuild logged.
