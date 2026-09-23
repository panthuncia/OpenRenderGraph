# Host epochs in persistent execution

A host that embeds the graph inside another engine often cannot submit it once per frame. It has to submit at
several fixed points of its own frame, and each point's work must land between the host's own GPU work before
and after it: the shadow maps, a depth prepass, a light-culling step, the main colour pass. Each of those
submission points is an **epoch**.

Before epochs, the only option was one graph executed once per point, in which every point prepared, admitted,
recorded and submitted every pass of every feature. Passes that did not belong to the current point returned
empty work. Every epoch therefore paid for every pass.

## The contract

-   **Tagging.** A pass declares its epoch with `ExternalPassDesc::Epoch(id)`. Untagged passes
    (`persistent::AllEpochs`, the default) belong to every epoch. A graph with no tags behaves exactly as it did
    before.
-   **Order.** The host names the order of its epochs within one frame once, with
    `RenderGraph::SetPersistentEpochOrder` (or `PersistentGraphHost::Desc::epochOrder`). Epochs not listed
    follow the listed ones, in ascending order. The order is structural.
-   **Execution.** `PersistentGraphHost::ExecuteFrame(hostData, beforePrepare, epoch)`, or
    `RenderGraph::SetPersistentEpoch` before `Update`/`Execute`, runs one epoch.
-   **What an epoch runs.** Only that epoch's passes and the untagged ones are updated, prepared, admitted,
    recorded and submitted. The dynamic Pre and Tail segments (the upload pass, per-frame passes) run in every
    epoch as before. An epoch with no main passes runs only those segments.

## One compile, per-epoch executables

The persistent program compiles once over the whole frame, and each epoch additionally gets an executable of
its own.

**The whole-frame executable** (`ExecutableGeneration::graph`) is built from every active pass:

-   Hazards are derived in **frame** order: every pass's authored order is reassigned by epoch rank first,
    with untagged passes ahead of every epoch, and by authored order within an epoch. Registration order across
    epochs means nothing to the host, and deriving hazards in it produces cycles against the epoch order. An
    example is a resource the colour pass writes and the next frame's depth prepass reads.
-   Explicit edges order every pass of an epoch before every pass of the next epoch that has passes.
-   Scheduling, alias placement (`PlanPersistentAliasPlacement`) and alias-order validation all run over this
    one timeline. Transients that live inside different epochs can therefore share memory: the transient
    footprint is the largest epoch's peak, not the sum of all epochs. A resource that crosses epochs gets one
    lifetime that spans them.

**The per-epoch executables** (`ExecutableGeneration::epochs`) are built from the same lowered compile input,
restricted to one epoch's passes plus the untagged ones:

-   They keep the same resource enumeration, bindings and alias placements, and the same explicit and
    placement edges between the passes they contain.
-   Each is an ordinary compiled graph, so its first use of each resource is an admission boundary. An epoch's
    entry state comes from the admission ledger, meaning whatever actually ran before it, including when an
    earlier epoch was skipped that frame.
-   Hazards on aliased memory across epochs are the alias ledger's, as they are across frames.

**Publications.** `SelectedPublication::ForEpoch(id)` returns a publication that shares the bindings and
logical graph and carries the epoch's executable. It is cached per publication. Admission, sealing and
recording take it like any other publication, so nothing downstream of the main segment knows about epochs.
Binding-only edits keep the executables, epoch splits included; structural edits rebuild all of them.

## Why not one compile per epoch

Compiling each epoch as a program of its own would give each its own alias plan. Transients could then only
share memory within an epoch, and every resource crossing epochs would have to become external to both
programs. It would also turn every binding edit into one edit per program. The single epoch-aware compile keeps
one program, one set of bindings and one alias plan, and needs only a compile-time split per epoch.

## Contract for hosts

-   A resource read in an epoch must have been written in that epoch or an earlier one of the same frame, as in
    any frame. With aliasing on, a skipped producer epoch leaves an aliased consumer reading undefined contents
    rather than last frame's. The alias ledger initialises the placement as a new occupant.
-   A pass instance belongs to one epoch. Work that runs in two epochs is registered twice, with its epoch fixed
    per instance.

## Closed executions

With `PersistentGraphHost::Desc::closedExecutions` (`RenderGraph::SetPersistentClosedExecutions`), every
execution of an epoch is **closed**: it starts from, and returns to, a fixed home state for everything it
touches.

-   **Home.** Each resource's home state is its seed state, or else the before-state of its first step. The
    admission ledger's `Prepare(..., closeToHome)` records each resource's first and last step, and adds an
    exit barrier back to home after the last pass that uses it.
-   **Entry.** Every closed execution begins with a full barrier on each queue's first batch. Together with the
    exit barriers, this makes an epoch's work independent of which epochs ran before it.
-   **Admission is cached.** Admission then depends only on (publication, epoch), so `SynchronousAdmission`
    caches one plan per executable. Any number of admissions may be pending at once, and they may be committed
    or abandoned in any order.
-   **Limit.** A resource used from more than one queue inside one closed execution is rejected. CS runs
    everything on DXVK's graphics queue.

A program that is not closed keeps the ordered, ledger-chained admission described above.

## Async epochs: tickets

`PersistentGraphHost::SetAsyncEpochs(epochs)` requires closed executions. It moves everything except submission
onto a host thread of ORG's own. The render thread then calls `SubmitEpoch(epoch, beforeSubmit)` at each point
instead of `ExecuteFrame`. `ExecuteFrame` throws while async epochs are on.

-   **Tickets.** For each epoch, the host thread prepares a ticket (`RenderGraph::PreparePersistentTicket`)
    about one frame ahead:
    -   it reserves a frame slot, and waits there for the slot's previous submission, never on the render thread;
    -   it runs update, invocations and cached admission;
    -   it records the epoch's command lists and publishes the ticket in that epoch's atomic cell.
-   **Submission.** `SubmitEpoch` on the render thread:
    1.  takes the ticket from the cell (waiting only if it is not ready);
    2.  runs `beforeSubmit`, the feature's commit;
    3.  checks the passes' revision hashes against the ticket (`InvocationRevisionHash`); a stale ticket is
        prepared again for the same slot;
    4.  records the pending uploads into the slot's upload list, which the host thread has already reset and
        begun;
    5.  submits the uploads and the ticket, with timeline values assigned at submit.
-   **Completion.** The host thread receives the submission through a single-producer mailbox. It commits
    admission, retires, reads statistics and releases deferred resources, in submission order.
-   **Constraints on a ticket.** Its recording may not depend on anything the render thread produces at the
    epoch. Dynamic Pre and Tail segments may contain only the upload pass. Per-frame values come from latches.

## Latches

`org::LatchBlock` is a host-visible buffer with one region per frame slot. A recording refers only to the slot's
offset (`RecordingContext::FrameSlot()`), and the render thread writes the values into the region before it
submits. A pass whose recording would otherwise bake a per-frame value, such as a count, a matrix or a dispatch
size, reads that value from its latch. Indirect dispatches go through `ExecuteIndirect` with a dispatch command
signature.

## Staged uploads

`org::runtime::StagedUploadBatch` lets a producer (a worker thread, or the render thread) write upload data
straight into mapped upload pages: `Stage(target, dstOffset, size[, data])`. `IUploadService::SubmitStagedUploads`
hands a batch over on the owner thread.

-   **Synchronous hosts.** The batch joins the upload pass's queue in order, with no copy of its bytes.
-   **Async epochs (direct mode).** The host records the waiting batches straight into the ticket's upload
    list (`RecordStagedUploads`), one copy per entry.
-   **Ordering.** Copies land in submission order. A copy that overlaps an earlier one in the same list,
    whether from an earlier batch or from the upload pass's queued copies, waits for it through a barrier.
    This happens when a batch's epoch was never submitted and its replacement follows it into the next list.
    Steady state has no overlaps.
-   **Lifetime.** A batch is held until the slot whose submission recorded it has retired. A producer reuses
    a batch once it holds the only reference.

## Tests

`tests/PersistentGraphTests.cpp` builds a program whose reader and writer are registered in the opposite order
to their epochs and checks:

-   the whole-frame order;
-   each epoch executable's placements: its own passes and the untagged one, and nothing else;
-   `ForEpoch` caching, and null for an absent epoch;
-   admission of both splits in sequence through one `SynchronousAdmission`;
-   that a program without tags has no epochs.

A second test covers closed executions:

-   closure (the exit barriers to home);
-   plan caching;
-   several admissions pending at once;
-   out-of-order commit and abandon;
-   the error for a program that is not closed.
