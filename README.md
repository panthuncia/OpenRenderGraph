# OpenRenderGraph

When BasicRHI is built with `BASICRHI_ENABLE_TRACY_GPU_PROFILING`, every
registered queue slot receives its own named Tracy GPU context. Render-graph
passes are emitted as GPU zones on the queue that executes them, so Tracy shows
graphics, compute, copy, and additional same-kind queue instances as separate
timelines aligned with the CPU capture. Parallel command-list recording workers
do not create additional GPU rows; pass metadata is ordered when the recorded
command lists are submitted, so concurrent recording cannot corrupt Tracy's GPU
zone nesting. No application-side pass callbacks are required.

OpenRenderGraph (ORG) is a powerful rendering/GPU-compute framework, built around a render graph.

ORG provides a pass-oriented API for building frame pipelines, tracking resource state/transitions, handling queue synchronization, planning automatic resource aliasing on eligible resources, and executing graphics/compute work with explicit resource usage contracts.

## What ORG does

- Compiles a persistent schedule from typed passes once, and executes it every frame; only structural changes rebuild it.
- Tracks resource access/layout transitions, batch ordering and queue synchronization, across one or more devices.
- Runs a frame as several executions (epochs) at points of a host's own frame, prepared ahead on a host thread.
- Hosts passes from other modules through a C ABI (`OpenRenderGraph/ContributorAPI.h`).
- Exposes runtime services for upload, readback, statistics, and graph settings.
- Includes memory introspection and aliasing planning/debug support.
- Includes optional ImGui-based debug UI widgets for graph and memory inspection.

## Include model

### Main API

```cpp
#include "OpenRenderGraph/OpenRenderGraph.h"
```

Use this for graph composition, pass construction, resource APIs, queue/runtime interfaces, and memory introspection APIs.

### Debug UI widgets (ImGui)

Each widget has its own header and CMake target:

- `DebugUI/ScheduleInspector.h` (`OpenRenderGraph::ScheduleUI`, `OPENRENDERGRAPH_ENABLE_SCHEDULE_UI`)
- `DebugUI/MemoryIntrospectionWidget.h` (`OpenRenderGraph::MemoryUI`, `OPENRENDERGRAPH_ENABLE_MEMORY_UI`)
- `DebugUI/MemoryViewWidget.h` (`OpenRenderGraph::DebugUI`, `OPENRENDERGRAPH_ENABLE_DEBUG_UI`)

## Basic usage

A host drives a graph through `GraphHost` (`OpenRenderGraph/GraphHost.h`). The graph's passes come from extensions, which
the host asks for them at each structural build; every frame then runs the compiled schedule.

### 1) Create a host

```cpp
#include <OpenRenderGraph/GraphHost.h>

org::GraphHost host({
    .device = device,                       // a BasicRHI device the host keeps alive
    .backend = rhi::Backend::D3D12,
    .tasks = std::make_shared<org::runtime::ThreadPoolTaskService>(4),
});
```

`Desc` also takes the host's epochs (`epochOrder`), frames in flight, closed executions, recording reuse and the external
queue boundary (for a host that shares its queue with another API, as Community Shaders shares DXVK's).

### 2) Write a pass

A pass declares what it uses once (`Declare`), prepares an owned packet of this frame's values (`Prepare`, optional),
and records it (`Record`, static: it sees only its bindings, the packet and the context). `Record` resolves declared views,
resources and pipelines through the context, against the frame the packet was prepared for.

```cpp
#include "RenderPasses/Base/TypedRenderGraphPass.h"
#include "Render/PassBuilders.h"

struct ClearBindings { org::DeclaredViewToken target; std::shared_ptr<const org::PipelineState> pipeline; };
struct ClearFrame { uint32_t count = 0; };

class ClearPass final : public org::TypedRenderGraphPass<ClearPass, ClearFrame, ClearBindings> {
public:
    ClearPass(std::shared_ptr<org::Buffer> target, std::shared_ptr<const org::PipelineState> pipeline)
        : m_target(std::move(target)), m_pipeline(std::move(pipeline)) {}
    ClearBindings Declare(org::PassBuilder& builder) {
        builder.PreferQueue(org::QueueKind::Compute).AutomaticQueueAssignment();
        return {builder.UnorderedAccess(m_target).View(), m_pipeline};
    }
    ClearFrame Prepare(const ClearBindings&, const org::PassPrepareContext&) const { return {1024}; }
    static void Record(const ClearBindings& bindings, const ClearFrame& frame, org::PassRecordContext& recording) {
        recording.BindComputePipeline(*bindings.pipeline);  // retained by the recording; descriptor slots filled by name
        const uint32_t constants[] = {recording.Index(bindings.target), frame.count};
        recording.PushConstants(1, constants);
        recording.Commands().Dispatch((frame.count + 63) / 64, 1, 1);
    }
    void Revision(std::vector<uint64_t>& out) const { out.push_back(m_pipeline->GetPayload()->generation); }
private:
    std::shared_ptr<org::Buffer> m_target;
    std::shared_ptr<const org::PipelineState> m_pipeline;
};
```

`Revision` (optional) appends what the recording depends on beyond its bindings and the frame's resources (settings,
pipeline generations). It only drives kept-recording reuse: while it stays equal, a kept recording of the pass may be
submitted again instead of recorded anew; a pass without it is never part of a kept recording. Lifecycle hooks
(`Submitted`, `Completed`, `Abandoned`) see a packet through to the GPU.

Declarations are the pass's access contract. A pipeline's `ResourceDescriptorIndex` slots are filled from the views the
pass declared under the same name (`builder.ShaderResource(Builtin::X)` and friends; a name declared for unordered access
wins). An undeclared slot gets `UINT32_MAX` only when it is optional, belongs to a feature domain the pass names nothing
of, or matches `BindOptions::absentPrefixes`; otherwise binding throws. The BRSL macro scan only names the slots.

### 3) Add passes through an extension

```cpp
class MyExtension final : public org::IRenderGraphExtension {
public:
    void PrepareForBuild(org::RenderGraph& graph) override {
        graph.RegisterResource(org::ResourceIdentifier("my.target"), m_target);  // named resources, providers, resolvers
    }
    void GatherStructuralPasses(org::RenderGraph&, std::vector<org::ExternalPassDesc>& out) override {
        out.push_back(org::ExternalPassDesc::Compute("my.clear", std::make_shared<ClearPass>(m_target, m_pipeline))
            .At(org::ExternalInsertPoint::Begin()));
    }
private:
    std::shared_ptr<org::Buffer> m_target = org::Buffer::CreateSharedUnmaterialized(rhi::HeapType::DeviceLocal, 4096, true);
    std::shared_ptr<const org::PipelineState> m_pipeline;  // the clear shader, compiled by the host
};

host.AddExtension("my.extension", [] { return std::make_unique<MyExtension>(); });
```

`ExternalPassDesc` places a pass (`At`: after or before other passes), names its queue, and tags it with an epoch.
`GatherFramePasses` adds per-frame passes (readback captures); `InterruptsFrame` hosts one at its insert point.

Resources are `Buffer`, `PixelBuffer` and the wrappers built on them (`VersionedBuffer`, `ExternalTextureResource`,
...). They are created unmaterialized and get memory when the graph first uses them; transient ones may share memory
with others whose lifetimes do not overlap (automatic aliasing). Providers and resolvers supply resources by name and
resource sets that change between frames.

### 4) Run frames

```cpp
host.ExecuteFrame();                      // one execution of every pass
// or, with epochs: host.SubmitEpoch(epoch) at each of the host's points
```

`RequestRebuild` (or a registry change an extension notices) rebuilds the schedule at the next frame. With async epochs
(`SetAsyncEpochs`) the host's thread prepares, admits and records each epoch ahead; the frame's point only submits.

### 5) Contributors from other modules

`OpenRenderGraph/ContributorHost.h` hosts the C ABI in `OpenRenderGraph/ContributorAPI.h`: another module (a plugin)
registers through the table the host exports (`ORG_GetRenderGraphAPI`), declares resources and passes at each build,
and prepares an owned packet per pass per frame, recorded through a borrowed BasicRHI recorder. The host names its
epochs, anchors and exported resources; `samples/ContributorSample.cpp` is a minimal contributor.

## Runtime services

ORG exposes service interfaces through `RenderGraph`:

- `GetUploadService()`
- `GetReadbackService()`
- `GetStatisticsService()`
- `GetRenderGraphSettingsService()`

Integrators typically initialize needed services during startup, then consume them during update/execute and teardown.

### Resource tracking (memory statistics)

Every resource ORG creates gets an entity in a flecs world (its size, name, usage), which the memory view reads.
Resources are created and destroyed on any thread: a render thread, a persistent host's thread, task workers, wherever a
retired backing is released. flecs is not thread-safe, so the world has one owner at a time (`ECSManager`, private):

- **Changes are posted, never applied in place.** Creating, attaching to and destroying an entity each post a command to
  a wait-free queue (`ECSManager::Post`). Tracking tokens are deferred: the entity is made when its creation command is
  applied, with whatever was attached before then; a token reset before then never gets one.
- **Whoever posts drains, if no one else is.** `Drain` applies the queue only when it can take ownership at once, so a
  poster never waits; the drainer looks again after releasing, so no command is left behind.
- **Readers wait for ownership.** `Access` takes it, applies everything posted before, and reads. The snapshot provider
  (`CreateECSMemorySnapshotProvider()`) reads this way, so it may run on any thread.

`InitializeRuntimeDevice` installs these hooks. A host that tracks into a world of its own installs its own
`TrackedEntityToken` hooks after it, and passes that world to `CreateECSMemorySnapshotProvider(world)`, synchronizing it
itself. `tests/TrackingWorldTests.cpp` creates, attaches and destroys from eight threads beside a reader.

## Debug UI usage (ImGui)

Widgets can be used directly if your app already has an ImGui frame:

```cpp
ui::MemoryIntrospectionWidget memWidget;
memWidget.PushFrameSample(timeSeconds, totalBytes);
memWidget.Draw(&open, &memorySnapshot, &frameGraphSnapshot);

graph->SetScheduleTraceEnabled(true);
org::debugui::ScheduleInspectorState state;
if (auto snapshot = graph->GetPersistentScheduleSnapshot()) org::debugui::ShowScheduleInspector(*snapshot, state);
```

## CMake notes

`OpenRenderGraph` is built as a static library in this workspace and links against:

- `BasicRHI`
- `flecs`
- `Boost::container_hash`
- `spdlog`
- `imgui`
- `implot`
- `slang`

If you consume ORG in another project, ensure equivalent dependencies are available.

## Packaging and standalone usage

`OpenRenderGraph` exports a package target:

- `OpenRenderGraph::OpenRenderGraph`

and expects `BasicRHI::BasicRHI` to be available (installed package preferred).

### Standalone build and install

`OpenRenderGraph` now uses CMake presets for standalone builds.

Prerequisites:

- CMake 3.23+
- Ninja
- `VCPKG_ROOT` set to your vcpkg checkout path

Configure once:

```powershell
cmake --preset ninja-x64-vcpkg
```

Build:

```powershell
cmake --build --preset debug
cmake --build --preset release
```

Install (example prefix):

```powershell
cmake --install out/build/ninja-x64-vcpkg --prefix out/install/org
```

If `VCPKG_ROOT` is not set in your shell:

```powershell
$env:VCPKG_ROOT = "C:/src/vcpkg"
cmake --preset ninja-x64-vcpkg
```

If you have an installed `BasicRHI` package, point CMake to it:

```powershell
cmake --preset ninja-x64-vcpkg -DCMAKE_PREFIX_PATH="out/install/rhi"
```

If you want to force submodule fallback mode explicitly:

```powershell
cmake --preset ninja-x64-vcpkg -DOPENRENDERGRAPH_ENABLE_SUBMODULE_FALLBACK=ON
```

### Consume ORG from another CMake project

```cmake
find_package(OpenRenderGraph CONFIG REQUIRED)
target_link_libraries(MyTarget PRIVATE OpenRenderGraph::OpenRenderGraph)
```

### Submodule fallback mode

If `BasicRHI` is not installed, ORG can fallback to an in-tree sibling `../BasicRHI`:

- `OPENRENDERGRAPH_ENABLE_SUBMODULE_FALLBACK=ON` (default)

When fallback is used, you can forward BasicRHI manual dependency options:

- `OPENRENDERGRAPH_FORWARD_BASICRHI_DEP_OPTIONS=ON` (default)
- `OPENRENDERGRAPH_BASICRHI_ENABLE_STREAMLINE=ON|OFF`
- `OPENRENDERGRAPH_BASICRHI_ENABLE_PIX=ON|OFF`
- `OPENRENDERGRAPH_BASICRHI_STREAMLINE_HEADERS_DIR=<path>`
- `OPENRENDERGRAPH_BASICRHI_PIX_HEADERS_DIR=<path>`


See [resource uses and declared views](RESOURCE_USES.md) for the unified declaration API and migration rules.
