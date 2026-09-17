# Graph replay fixtures

`radius20_graph_snapshot.orggraph` is the last numeric compilation from a synchronous
SARP radius-20 camera-flight run on 2026-09-16. It contains 176 passes and 646 resources.
Native realizations, program objects, scene assets and GPU alias heaps are excluded.
Use it for compiler legality regression and synthetic admission workloads.

The source benchmark **timed out**, with pending texture reloads. Its desired scene
digest was `10986299020466424073` and it had 40,204 live objects, but its applied
placement count was 86,853, below the accepted 88,631. This fixture is representative
numeric graph input, not proof of final scene correctness or application acceptance.

It was captured after correcting the recorder to detach numeric metadata from the
compiler input's aliased control block. The earlier lease-retaining capture is not
used. Full publication/edit/completion replay streams are still to be implemented.

`binding_rotation_tail.orgsequence` is a compact hand-authored sequence: two
queues, a writer and reader, a binding replacement while a submitted frame is
held, an additional GPU tail, and a subsequent partial submission. The production
admission path consumes its recorded signals and completion observations.
`OpenRenderGraphHeadlessBenchmark --sequence FILE` runs it without assets or a GPU.
The adapter supplies synthetic backing identities and reports bootstrap,
publication installation, admission distributions and total replay time separately.
It is a correctness fixture, not a representative timing acceptance workload.
