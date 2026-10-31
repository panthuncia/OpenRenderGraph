# Decision trace

ORG records the decisions it makes while it compiles, admits, plans barriers for and submits executions, with the inputs that
decided each one. It is the tool for "why did this resource get that barrier", "why did this pass go to that queue", and
"what did this execution wait on". API: `include/OpenRenderGraph/Trace.h`.

## Turning it on

From the environment, read once when the `org::Runtime` is created:

| Variable | Meaning |
|---|---|
| `ORG_TRACE` | Comma-separated categories, or `all`: `schedule`, `barriers`, `handoff`, `admission`, `aliasing`, `submission`, `retirement` |
| `ORG_TRACE_FILE` | JSON-lines output (default `org_trace.jsonl` in the working directory) |
| `ORG_TRACE_RESOURCE` | Keep only events that name no resource, or a resource whose name contains this text |

Or from code: `org::trace::SetSink(org::trace::MakeJsonLinesSink(path))` (or a `MemorySink` in a test), then
`org::trace::SetCategories(...)`. Off (the default) a trace point costs one relaxed atomic load.

Names: a graph captures pass and resource names for the trace while it is on, so turn it on before the graphs of interest
build. Events from a graph built earlier name passes `pass#<index>` and resources `resource#<slot>`.

## Events

Every line has `t` (steady clock, ns), `thread`, `category` and `decision`, then the decision's fields.

| Category | Decision | Fields |
|---|---|---|
| schedule | `pass-scheduled` | pass, queue, preferred and compatible queues, batch, `batching` (first-batch, extends-previous, queue-change, isolated, after-isolated, new-wait), the batches it waits on per queue |
| barriers | `state-step` | one resource use: `path` (body-same-batch, body-same-queue, body-cross-queue, entry-seeded, entry-ledger, entry-invalidated, entry-aliased, entry-authoritative), pass, batch, queue, previous pass/batch/queue, the compiled before and after layouts, and for entry-ledger the ledger's layout and queue |
| barriers | `barrier` | every texture barrier the plan holds: `list` (release-list, entry, entry-acquired, before-pass, after-pass), batch, queue, pass, layouts, accesses, ownership, discard |
| handoff | `handoff-planned` / `handoff-skipped` | a cross-execution handoff: from and to queue, layouts; when skipped, `reason` (closed-execution, discarded, previous-queue-unknown, same-queue, previous-queue-not-in-graph, other-device) |
| admission | `admitted`, `incoming-waits`, `ledger-commit` | an admission (or readmission): whether its plan was cached, its waits per batch, and each state the ledger commits with its queue |
| submission | `batch-submitted`, `release-list-submitted` | a batch's queue, signal, waits, leading lists and extra timelines; a handoff release list's queue, signal and consumers |

A plan is made twice for an open execution (at preparation and again at readmission, under the sequencer); both show.

## Reading it

`barrier` events of one resource, in batch order, give the layout chain the GPU sees; `state-step` events tell which path
produced each one. Correlate with the D3D12 debug layer or a GPU capture by `handle` (generation << 32 | index).
