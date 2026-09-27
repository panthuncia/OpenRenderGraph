# Resource uses and declared views

`PassBuilder` (the queue-agnostic renderer-facing builder) supports declaring access and descriptor requirements together:

```cpp
bindings.input = builder.ShaderResource(input);          // shader access and default SRV
bindings.output = builder.UnorderedAccess(output);       // unordered access and default UAV
bindings.arguments = builder.IndirectArguments(arguments); // no descriptor
auto clear = builder.UnorderedAccessClear(output);        // shader and CPU UAV views
```

Usage-specific methods include `ShaderResource`, `UnorderedAccess`, `ConstantBuffer`, `RenderTarget`, `DepthRead`, `DepthReadWrite`, `CopySource`, and `CopyDestination`. `ShaderResource` and `UnorderedAccess` also accept several resources in one call, or a span of typed views for one resource. A single declaration returns its resource bindings and declared view tokens; scalar consumers can assign the result to a `DeclaredViewToken`, which checks that there is exactly one view. Groups and multiple-view declarations expose indexed `View(index)` and their full token lists. `UnorderedAccessClear` declares both shader-visible and non-shader-visible UAVs. The common access normalization remains internal to the builder.

Use `Subresources(resource, Mip(...), Slice(...))` to specify access ranges. The default is whole-resource access. A descriptor's mip/slice selects a published descriptor, not an inferred access extent. `SrvView` and `UavView` specify variant, mip, and slice. Named identifiers preserve automatic shader descriptor registration, resolver snapshots, and feature-domain activation.

During preparation, use `context.Resolve(viewToken)` for a descriptor slot or `context.Capture(viewToken)` for an owned packet reference. Use `viewToken.Resource()` when an operation needs the resource binding (for example its clear value). Never fetch a live descriptor from a resource wrapper. Tokens belong to a particular pass declaration; another pass or a replacement declaration cannot resolve them.

Logical declarations do not allocate descriptors or require physical backing. The persistent adapter records requirements using `RequireView`; materializing or replacing the backing validates the owned descriptor snapshot. Direct persistent `DeclareView` remains strict and requires an already published snapshot. Persistent tokens are mapped by the compiled pass, while the legacy preparation path resolves the same requirements against frozen bindings. Old publications retain their descriptor owners. Resolver view requirements follow reserved group positions through membership changes and capacity growth; group bindings do not add direct scheduling accesses for unoccupied positions. Scalar view resolution uses the selected publication's native backing, so rotating resolver global IDs do not leave stale preparation lookups. View-layout changes publish synchronously before preparing the newly declared pass.

## GPU tables

`DeclaredTableLayout<Row>` holds only row/member destinations and declared view tokens:

```cpp
bindings.table = org::DeclaredTableLayout<Row>(rowCount);
builder.ShaderResource(texture, org::SrvView{}, bindings.table.Field(camera, &Row::textureIndex));
// During preparation; rows contain ordinary per-frame values:
auto index = bindings.table.Publish(context, publisher, std::span<const Row>(rows));
```

A token can populate additional destinations through `Field(...).Bind(token)` without declaring another resource use. Invalid rows, null member pointers, invalid tokens, and duplicate destinations are errors. Table publication copies the input rows, resolves frozen descriptors, and uses the existing `PreparedTablePublisher` ownership and byte-reuse policy.

Select resource identities and view requests before compilation. Resource/camera mapping and table destination changes require redeclaration. Ordinary row-value changes belong in the recipe revision; backing changes are tracked by descriptor resolution. Comparing layouts uses member-pointer equality rather than hashing member-pointer bytes.

## Compatibility and first migration

Existing `With*`, `Bind*`, and `ResolveView(resourceToken, request)` APIs remain available. They share the existing access-lowering path; strict view-token permissions apply to the new API. `ClusterRasterizationPass` is the first migrated consumer, including named builtins, output variants, GPU table fields, and diagnostic constants. Other CLod table consumers retain their legacy implementation. Descriptor allocation policy and shader layouts are unchanged.
