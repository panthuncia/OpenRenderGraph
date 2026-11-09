# Pipelines

A pass describes its pipelines and command signatures by value, in `Declare`. ORG builds each one for the device the pass
runs on, before the pass is published. The pass has no backend code, so placing it on another device
(`RequireBackend`, `PreferBackend`, `RequireDevice`) needs nothing more from it.

```cpp
LuminanceHistogramBindings Declare(org::PassBuilder& builder) {
    builder.ShaderResource(Builtin::Color::HDRColorTarget);
    builder.UnorderedAccess(Builtin::PostProcessing::LuminanceHistogram);
    return {builder.Pipeline(org::ComputePipelineDesc{
        .shader = {L"shaders/PostProcessing/LuminanceHistogram.hlsl", L"CSMain"},
        .name = "LuminanceHistogramPassCS"})};
}
static void Record(const LuminanceHistogramBindings& bindings, const Frame& frame, org::PassRecordContext& context) {
    context.BindComputePipeline(bindings.pipeline); // the build for the pass's device
    ...
}
```

## Descriptions

`Render/PipelineDesc.h`. They own all their data.

- `ComputePipelineDesc`: a shader, a name, defines and a layout.
- `GraphicsPipelineDesc`: vertex or mesh (with an optional amplification) and pixel shaders, and optional raster, blend,
  depth, render-target, depth-format and sample states.
  - An absent state is not given to the device, so the backend's default applies.
  - Topology is the exception. Backends disagree about an omitted one, so vertex pipelines always get one, `TriangleList`
    unless set.
  - `inputLayout` gives a vertex pipeline vertex buffers (`PipelineInputLayout`: bindings, and attributes that own their
    semantic names). Without one, the vertex shader pulls its own vertices.
    - D3D12 matches attributes to shader inputs by semantic, and Vulkan by location. An unset location counts up in
      declaration order, which is the order DXC gives the shader's inputs.
    - The layout must allow the input assembler on D3D12. The library refuses a custom layout without
      `rhi::PF_AllowInputAssembler` on every backend, so a description that builds on Vulkan also builds on D3D12.
    - It is part of the description's hash, and the compile request carries it (`PipelineCompileRequest::inputLayout`).
- `CommandSignatureDesc`: indirect arguments, the byte stride, the layout, and the Vulkan sequencing flags.
- `PipelineLayoutRef`:
  - `StandardCompute` and `StandardGraphics` are the host's layouts.
  - `Custom` holds a `PipelineLayoutSpec`. ORG creates it on each device that needs it, and pipelines with the same spec
    share it.

A description's content hash is its identity (`PipelineContentHash`). `RenderPassBuilder::Pipeline` returns the same
`PipelineState` for the same description while any declaration holds it. So a pass that is declared again gets the same
pipeline, and passes that describe the same pipeline share one build. A description that differs, such as a define taken
from a setting, is a new pipeline.

## Compiling

ORG has no shader compiler. The host supplies one as `Runtime::Desc::pipelineCompiler`, an `org::PipelineCompiler`:

- `Compile(request, completion)` compiles one pipeline's shaders for one device's backend. It also returns the
  descriptor slots the shaders name (`PipelineResources`), because those come from the host's shader conventions.
  - It runs on task service threads and completes exactly once, on any thread.
  - `request.bypassCache` marks a reload, which the host's cache must not answer.
- `StandardLayout(kind, device)` returns the host's layout of that kind on a device, with the object that keeps it alive.

Without a compiler, declaring a pipeline by description throws. Pipelines built by the host (an `org::PipelineState`
made from an `rhi::PipelinePtr`) still work, on the device they were made for.

BasicRenderer's compiler is `src/Pipeline/PipelineState/OrgPipelineCompiler.cpp`, over `PSOManager`'s compile path and
shader cache.

## When pipelines build

`PipelineLibrary` (one per `Runtime`, `Runtime::Pipelines()`) builds on the task service. It builds a pipeline once per
device, and only for devices a pass using it runs on. A pass is never published with one of its pipelines unbuilt:

| When | What happens |
| --- | --- |
| First build of a graph | Every pass's pipelines build in parallel, and the first publication waits for all of them. |
| A pass declares a new pipeline later | Frames keep preparing against the declaration they have until the new pipeline is ready, then the pass is lowered again. |
| A stale direct resolver | The new declaration cannot wait, because the old one may name resources that are gone, so the frame waits. |
| Structural edits and passes added by extensions | These wait. |

A failed build names the pass, the pipeline, the device and the compiler's message. On the first build that is an
exception. Later, the pass keeps its previous declaration and the failure is logged.

Each device's result lands in its slot of the `PipelineState` atomically. Recordings on other devices read their slots
meanwhile, and nothing is locked. A recording retains the payload it bound until the recording retires.

Command signatures are created on the device when the pass is lowered. Creating one compiles nothing, so this happens
inline. `RecordingContext::ExecuteIndirect(const CommandSignatureState&, ...)` picks the pass's device's signature.

## Waits across devices

A device waits only on its own timelines. A graph's queue timelines are shared between its devices (`QueueRegistry`'s bridge
fences). A submission can also wait on timelines the graph does not own: another graph's queues through the runtime's
ledgers, an `ExternalClient`'s, or the frame-wait timeline. For those, the graph and `ExternalClient` call
`Runtime::RelayWaits`:

1. The timeline's own device waits for the points, in an empty submission on its graphics queue.
2. It signals a D3D12 fence shared with the waiting device, one per pair of devices.
3. The submission waits on that fence instead.

`Runtime::RegisterDevice` must be given the device's API for this, which `RenderGraph::RegisterBackendDevice` does.

## Resources shared between devices

A resource that a pass on another device uses is shared: it gets a resource on each device over one allocation, and its
descriptors on each device name that device's resource. The primary device's resource replaces the resource's own backing.
D3D12 allocates the memory, and Vulkan imports it.

- **Alias candidates** are placed in a shared alias pool, one per resource class. A pool is a D3D12 heap created shared and
  imported on the Vulkan device (`SharedAliasHeaps`).
  - A placement is a resource placed at the same offset on both devices (`RenderGraph::PlaceSharedRepresentations`).
  - Its size and alignment are the larger of the two devices' requirements.
  - Bindings record it as aliased: it enters every execution undefined, and the alias ledger orders its range against other
    occupants on either device.
  - Placement orderings between users on different devices become cross-device waits.
- **Committed instead:** resources that do not qualify (`RenderGraph::IsSharedAliasEligible`) get one shared committed
  allocation each (`EnsurePersistentRepresentations`):
  - render-target and depth textures, which D3D12 requires to be initialized by D3D12 after each occupant switch;
  - formats the Vulkan device cannot place in an imported heap (`rhi::vulkan::query_d3d12_heap_texture_support`);
  - resources that are not alias candidates.

`BasicRHIWin32SharedHeapAliasSmoke` checks that a driver gives both devices the same bytes for placed textures, every mip
included, through an occupant switch.

## Reloading

`PipelineLibrary::Rebuild(entry, options)` builds the pipeline again on every device it was built for:

- Nothing changes until `PublishRebuilt`, and then it changes on all of those devices together.
- A failure on any device keeps the old payloads.
- A changed resource interface also fails, because declarations were lowered against the old descriptor slots. A host
  that rebuilds its graphs afterwards passes `allowInterfaceChange`.
- A rebuild has one generation number on every device.
- The last 8 generations are retained, and `Activate` makes one current again.

The graph publishes finished rebuilds when it prepares a frame, and a host may also do so between frames. BasicRenderer's
live-pipeline tools (`br::diagnostics::RequestRecompile`, `RequestActivation`, and the global shader reload) route
described pipelines through these calls.

## Not covered

- Pipelines picked at `Prepare` time from a permutation cache. BasicRenderer's cluster rasterization, material
  evaluation, deep visibility resolve and deferred shading still use `PSOManager`'s per-device-unaware caches, so those
  passes run on the primary device only.
- Ray tracing pipelines and work graphs.
- Shaders that differ in behaviour between APIs, and resources that cannot be shared between devices (see
  `EnsurePersistentRepresentations`). Known open problems with peer passes are in SARP's `docs/open-defects.md`.
