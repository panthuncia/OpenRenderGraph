# Pipelines

ORG builds no pipelines. The host builds them and hands them to passes, which bind them in `Record`:

- `RecordingContext::BindComputePipeline(const PipelineState&)` (and the graphics and ray tracing forms) binds a pipeline
  with descriptor slots. A `PipelineState` holds one payload per device, and binding picks the pass's device's payload.
- `RecordingContext::BindComputePipeline(std::shared_ptr<const rhi::PipelinePtr>, rhi::PipelineLayoutHandle)` binds a
  pipeline made outside `PipelineState`, with no descriptor slots.
- `RecordingContext::ExecuteIndirect(const CommandSignatureState&, ...)` picks the pass's device's command signature.

A recording retains the payload it bound until the recording retires.

## Descriptions

`Render/PipelineDesc.h` describes pipelines by value, for hosts that build them from a description. They own all their
data.

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
    - The layout must allow the input assembler on D3D12. A custom layout needs `rhi::PF_AllowInputAssembler`.
- `PipelineLayoutRef`:
  - `StandardCompute` and `StandardGraphics` are the host's layouts.
  - `Custom` holds a `PipelineLayoutSpec`.
  - `RootConstants(count, stages, name)` is a custom layout of one block of root constants (b0, root parameter 0). The
    shader reaches its resources through the descriptor heap, with indices from those constants.

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

## Not covered

- Ray tracing pipelines and work graphs.
- Shaders that differ in behaviour between APIs, and resources that cannot be shared between devices (see
  `EnsurePersistentRepresentations`). Known open problems with peer passes are in SARP's `docs/open-defects.md`.
