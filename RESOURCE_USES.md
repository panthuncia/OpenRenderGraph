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

`RtvView` and `DsvView` carry the same selection for render targets and depth attachments. `PassPrepareContext::Describe(viewToken)` reads the selected resource's frozen description. A pass that also needs the native resource for a barrier or render target uses `DeclaredReference(viewToken)`; the token continues to identify the same declared resource use.

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

## Declaration ownership

The former resource-access `With*` and `Bind*` methods, and token-plus-view-request resolution, have been removed. BasicRenderer and SARP passes use the usage-specific declarations. Named shader descriptor registrations, including explicit view variants, derive from the same declaration; passes no longer register those descriptors separately in `Initialize`. CLod per-view tables select their resource and camera mapping in `Update`, bind destinations in `Declare`, and resolve the tokens immediately before publication. No migrated table reads a live descriptor slot. Legacy `ComputePassBuilder` uses the same access names through its resolver-aware lowering; it has no frozen preparation-token lifecycle. Descriptor allocation policy and shader layouts are unchanged.

Store view tokens and table layouts in the binding returned by `Declare`, not in mutable pass members. The compiled pass owns that binding while older frames prepare or record. `IDynamicDeclaredResources` passes redeclare when their selected resource identity or descriptor slice changes. Legacy `ComputePass` implementations can use the simplified declaration names, but still obtain descriptors through their existing execution path until converted to frozen preparation packets.
