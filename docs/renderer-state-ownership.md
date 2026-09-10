# Renderer state ownership and manager migration

This inventory is the authority for the manager migration. Rendering consumes
only immutable publications selected by an accepted frame. Mutable objects may
remain behind ingestion, allocation, cache, IO, upload, readback, and vendor
service boundaries, but they are not alternative sources of frame state.

Implementation note (September 10, 2026): material usage and row producers are
pure publication steps. Their preallocated reservation tokens commit accepted
results to the versioned journals exactly once; the removed
`IMaterialStateStorage` callback is no longer part of producer execution.
Material raster flags and compile-slot mappings are consumed from
`PublishedMaterialState`. Geometry-residency publications retain exact slab and
page-pool versions, and CLOD streaming uses a scoped storage/residency service
instead of a broad `MeshManager` dependency. Object publications also own the
skinned-placement inputs consumed by Procedural Wind. These paths complete the
render-facing portion of migration step 1; remaining manager calls are within
scene ingestion or scoped residency services.

View-family, light-table, and pose state are async-graph artifacts and renderer
manifest fragments. The view family retains camera/culling buffers, visibility
attachments, current/history depth resources, their bindless indices, and
immutable per-view metadata. Ordinary render-graph extensions no longer
enumerate `ViewManager`; deep-visibility resources are allocated before the
family is published. The light fragment retains the coherent GPU table set and
has an exact state-graph dependency on the view-family revision containing its
shadow view IDs. Shadow view allocation is a narrow ingestion-time identity and
storage service; `LightManager` no longer owns or queries `ViewManager`.
View-creation events are composed at the renderer boundary to maintain legacy
indirect workload storage, so `ViewManager` no longer owns an indirect-buffer
manager.
The pose fragment retains palette resources, active instance membership,
offsets, and immutable base-skeleton owners. Procedural Wind consumes that pose
selection and calls only the narrow serialized wind-palette allocation service.
Depth history is represented in accepted frame data as an owned selection of
the exact resource, history epoch, and producer submission. Consumers no longer
consult a separate global validity flag. Selecting an unsubmitted producer and
removing the measured startup view fallback remain scheduling-integration work.

## Ownership model

| Layer | Owns | Must not own |
| --- | --- | --- |
| Scene and asset sources | Editable entities/assets, stable source identity, ordered change streams | GPU binding selection or frame lifetime |
| Async state graph | Version selection, dependency/readiness edges, build scheduling, cancellation, retained artifact results | Device caches, allocation arenas, mutable scene registries |
| Renderer publication | Compatible immutable artifact roots, resource catalog, bindless indices, allocation/program holds | Mutable producer containers |
| Persistent service | Stable-ID/range allocation, backing pools, caches, IO, upload/readback, exclusive SDK state | The authoritative renderable scene |
| Frame render graph | Exact publication lease, frame inputs, declarations, preparation, compilation, recording and submission | Latest-state queries into producers |

An artifact producer accepts owned values and exact retained dependencies. A
producer may submit an explicit service request, but it must not capture a broad
manager or mutate one from its acceptance callback. Physical resource and
descriptor versions may change; the publication updates their indices and GPU
tables together.

## Current responsibility inventory

| Current owner | Source state | Derived/published state | Persistent service work | Compatibility to remove | Destination |
| --- | --- | --- | --- | --- | --- |
| `SceneRenderBridge` / `ManagerInterface` | Scene deltas and alive sets | None directly | None | Replays changes into every manager through raw pointers | Renderer source-state store emits ordered artifact intents; retain a temporary ingestion adapter |
| `MaterialManager` | Material identity, usage counts, raster flags | Material rows/tables, compile-flag slots, texture-image dependencies | Slot/raster-bucket allocation and texture-streaming coordination | Resource-provider lookup and graph producers capturing `MaterialManager&` | Owned material inputs and table artifacts; scoped identity/allocation and texture-streaming services |
| `ObjectManager` | Object/group identity and transform changes | Draw records, transforms, visibility generations, active draw sets | Static-import reservations, range allocation, compaction and deferred retirement | Direct buffer getters and rendering-time resource-provider lookup | Object source records; draw-page/buffer artifacts; allocation/import/retirement services |
| `MeshManager` | Mesh/instance registration and imported geometry | Mesh tables and geometry/residency selections | Owns the current implementation of the scoped CLOD geometry-storage/residency service | View/skeleton links and bootstrap resource-provider registration | Owned geometry artifacts plus the extracted geometry-storage and residency service |
| `IndirectCommandBufferManager` | Requested workload keys and capacity policy | Immutable workload membership and argument-buffer versions | Per-frame argument/counter allocation and build scheduling | Direct object/material/view queries and submission caches exposed as manager state | Workload artifact producer; frame-owned scratch/argument service |
| `ViewManager` | Stable view identity and camera/light association | Camera/culling tables and view-family metadata | View-ID allocation; generation-owned attachment creation | Calls indirect manager, mutable view enumeration, global depth-history validity | View-family artifacts; view identity service; frame resources and history service |
| `LightManager` | Light source records and shadow configuration | Light tables and view requirements | None beyond storage allocation | Creates/mutates views imperatively and reads their live indices | Light artifacts feeding the view-family producer |
| `SkeletonManager` | Skeleton/instance identity and evaluated poses | Palette/instance tables and pose history | Palette range allocation | Mutable pointer iteration and fixed current/previous rotation | Pose artifacts plus palette allocation/history service |
| `TerrainManager` | Terrain source/configuration changes | Terrain tables, material/texture selections | Terrain texture production and streaming requests | Live material/texture-manager links and resource-provider lookup | Terrain artifacts plus texture-production service |
| `EnvironmentManager` | Environment identity and source image selection | Environment table and filtered-output selection | Conversion, filtering, SH and readback work queues | Resource-group mutation embedded in work payload commit callbacks | Environment artifacts with frame-owned service reservations |
| `TextureStreamingManager` / `TextureFactory` | Texture source identity and requested quality | Texture binding/image-table publications | Decode, descriptor allocation, upload, readback, residency and cache | Material-manager ownership and live provider fallback | Renderer-scoped texture storage/streaming services emitting binding artifacts |
| Pipeline/signature managers | Program request keys | Coherent immutable program versions | Device-scoped compilation/cache and signature/work-graph creation | Recording-time singleton lookup | Program-version service retained by declarations |
| Upscaling/FFX managers | Requested mode/quality | Per-generation SDK configuration and selected history | Exclusive SDK contexts and ordered evaluation | Settings/global singleton reads during preparation or recording | Generation-owned vendor service requests captured by accepted frames |
| Upload/readback/statistics services | Work requests | Submission receipts and completed feedback | Queue reservations, staging/query allocation and completion processing | Generic immediate-pass replay and slot-global timing reuse | Frame-owned reservations resolved exactly once on submission or cancellation |
| External contributor host | ABI-owned callback state | Contributor declarations and packets | Contractually required callback dispatch | Internal compatibility behavior leaking beyond the ABI edge | Preserve as the sole compatibility boundary and measure main-thread callbacks |

## Migration invariants

- Advancing any producer after frame acceptance cannot change that frame's
  resources, bindless indices, table contents, programs, camera/pose data, or
  history selection.
- Publication catalogs are the only rendering-time resource lookup path.
  Bootstrap providers may construct initial artifacts but cannot serve as a
  fallback for a missing declared publication.
- Exact version identity and ownership travel together. Numeric artifact IDs,
  raw pointers, cache handles, and resource wrappers alone are not ownership.
- CPU-written bytes, GPU-written counters, attachments, command allocators, and
  timing queries cannot be reused until their owning frame retires or joined
  cancellation completes.
- Mutable services expose typed requests and reservation tokens. Commit and
  cancellation are mutually exclusive and occur exactly once.
- The state graph builds persistent versions; the render graph schedules one
  accepted frame. Neither graph absorbs the other's lifetime domain.

## Migration order and removal gates

1. Complete material, draw, geometry, and workload publications; remove their
   rendering-time manager/provider reads.
2. Publish coherent light/view families and pose/history versions; remove
   cross-manager callbacks and mutable enumeration from preparation.
3. Route scene ingestion to a renderer source-state store and finish terrain,
   environment, streaming, upload/readback, and vendor service extraction.
4. Replace transitional `UpdateContext`/`RenderContext` manager pointers with an
   exact publication lease and owned frame values.
5. Move preparation to its serialized worker, add bounded recording-ahead
   queues and presentation-tail recording, then remove manager facades and
   transitional execution paths after parity.

Each removal gate requires an empty internal caller inventory, while external
ABI adaptation remains explicit at the contributor boundary.
