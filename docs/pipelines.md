# Pipelines

ORG builds no pipelines. The host builds them (OMS's `PipelineRegistry` and `BuildPipeline` in CS) and hands them to
passes, which bind them in `Record`:

- `RecordingContext::BindComputePipeline(std::shared_ptr<const PipelineStatePayload>)` (and the graphics and ray tracing
  forms) binds a payload's pipeline and layout, and fills its descriptor slots from the pass's declarations.
- `RecordingContext::BindComputePipeline(std::shared_ptr<const rhi::PipelinePtr>, rhi::PipelineLayoutHandle)` binds a
  bare pipeline, with no descriptor slots.

A recording retains what it bound until the recording retires.

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
