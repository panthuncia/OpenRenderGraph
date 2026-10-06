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
    3.  checks the passes' revision hashes against the ticket (`InvocationRevisionHash`), and that no backing
        changed since it was prepared (below); a stale ticket is prepared again for the same slot;
    4.  records the pending uploads into the slot's upload list, which the host thread has already reset and
        begun;
    5.  submits the uploads and the ticket, with timeline values assigned at submit. The uploads go to the
        graphics queue, where the ticket's entry barrier orders its batches after them. A batch on another queue
        (async compute) waits for the uploads' signal instead (`SubmitPersistentTicket`'s `uploadsDone`).
        Without that wait, a compute pass could read the epoch's uploaded inputs before they land: on the first
        frames, never-written memory.
-   **Completion.** The host thread receives the submission through a single-producer mailbox. It commits
    admission, retires, reads statistics and releases deferred resources, in submission order.
-   **Constraints on a ticket.** Its recording may not depend on anything the render thread produces at the
    epoch. Dynamic Pre and Tail segments may contain only the upload pass. Per-frame values come from latches.
-   **Backing changes.** Each preparation captures every slot's backing, while the render thread runs, so the
    two must not overlap (`BackedResource`: capture and backing mutation are serialized by the preparation
    owner). `Buffer::ResizeBytes` and `ResizeStructured` release the old backing before creating the new one; a
    capture between the two binds nothing, or materializes the buffer from the host thread. So the render
    thread changes a backing only inside `PersistentGraphHost::MutateBackings()`:
    -   the scope waits for a preparation in progress to end (never for the slot's GPU wait, which comes
        before it) and holds the next one off until it closes;
    -   it advances the host's backing version: a ticket prepared under an older one is stale at submission,
        since its passes would use the old backing while the commit wrote the new one;
    -   while async epochs run, a resize outside a mutation scope throws
        (`BufferBase::RequireBackingMutationAllowed`).

    Open it only where a backing does change, so a frame that changes none never waits. `AsyncStats` counts
    the scopes and the ones that waited.

## Recording reuse

With `PersistentGraphHost::Desc::reuseRecordings` (`RenderGraph::SetPersistentRecordingReuse`), a ticket's recording is kept
and submitted again while what it was recorded for holds, so an epoch is prepared and executed as it is and never prepared
again.

-   **What is kept.** A ticket's segment is recorded for resubmission when its execution is closed, nothing is late-bound
    (no rebound slot, no incoming effect), every pass reports its revision, and every pass is replayable
    (`PreparedPass::IsReplayable`: no submission, completion or abandonment callback, no external signal, no dependency
    lifecycle effect). Its packets are then replayable (`PreparedRhiExecutionBatch`, `IPreparedExecutionBatch::Replayable`):
    a submission commits nothing but the commands, completion releases nothing, and the lists return to their pool when the
    last owner lets the packet go.
-   **The key.** Per (epoch, frame slot), up to `RenderGraph::kKeptRecordingVariants` recordings, the most recently used
    first, each with what it was recorded for: the publication, the bindings, the descriptor heaps, the host tag (the
    host's backing version) and every pass's revision. A set keeps only recordings of the current publication, bindings and
    host tag.
-   **Reuse.** The owner thread, preparing a ticket for what a kept recording was recorded for, takes it instead of
    recording (the new invocations are abandoned unrecorded).
-   **Adoption.** At submission, a ticket that `beforeSubmit` made stale takes the kept recording of what its passes depend
    on now (`RenderGraph::AdoptKeptPersistentRecording`: a lock-free read of the set on the submitting thread), and is not
    prepared again. Only a variant never recorded for that slot is prepared again. A reserved submission adopts too.
-   **One submission at a time.** A kept recording belongs to one frame slot, and a slot's ticket is prepared after the slot's
    previous submission completed, so a list is never in two submissions at once. BasicRHI's Vulkan lists begin without
    `ONE_TIME_SUBMIT` for this.
-   **What a kept recording leaves out.** Every resource a list names must be owned by its recording. Kept recordings
    therefore record no profiler GPU zones or ranges (Tracy), and no GPU statistics queries: the statistics service recreates
    its query pools and readback buffers as passes register, and a resubmitted list resolved into freed ones (a GPU page
    fault). Turn reuse off to profile an epoch's passes on the GPU.
-   **Statistics.** `AsyncStats::recorded`, `kept`, `reused` and `adopted`.
-   **Release.** `StopAsync` lets every kept recording go once the slots' work is done (`ClearKeptPersistentRecordings`).

A kept recording also keeps whatever its passes' prepared data hold, for as long as it is kept. A host that versions
resources by reference count (writing only into a version nobody holds) must therefore not cap its versions: with a fixed
number of them, kept recordings can hold every one, and the next version waits forever (CS's DCLF pipeline sets did).

For the variants to come back, a host's pass revisions must come back too: a shape that alternates should republish its
earlier identity rather than a new one (CS's DCLF keeps its last few shapes for this).

## Revision-driven epochs

`SetAsyncEpochs(epochs, revisionEpochs)` makes the listed epochs revision-driven. Their recordings are made ahead, for a
revision the host's scene supplies, and nothing about them is recorded or prepared at submission. This is the epochs' side of
a scene revision recorded by a dependency graph (CS's DCLF: the scene revision, phase 6b R1–R7).

-   **Slot rings.** Async epoch i uses frame slots `i * framesInFlight` up to `+ framesInFlight - 1` (when the host's slots
    hold `framesInFlight` per async epoch, which `FrameSlots` does). Its tickets take the next slot of its ring, so a
    recording per ring slot covers every ticket. Every async epoch uses its ring, revision-driven or not.
-   **`RequestEpochRecording(epoch, hostData, done)`.** Lock-free, from any thread: a request pushed on a Treiber stack that
    the host's thread takes whole, in posting order.
    -   The host's thread records the epoch for `hostData` into every slot of its ring: what the passes prepare for
        (`FramePreparationContext::preparationData`), retained while it records.
    -   It records a slot once the slot's last work is done; it waits for that, never the caller. A slot the epoch's
        unsubmitted ticket holds is recorded too, since that ticket holds its admission alone.
    -   `done(EpochRecording, error)` runs on the host's thread once every slot is recorded. A recording that is not
        replayable is an error.
-   **Immutable inputs.** A recording for a revision resolves the graph's resolvers through it: `PreparePersistentTicket`
    passes its host data as the resolver capture context (`ResolverCaptureContext` of `IHostExecutionData`). The recording's
    publication and bindings are the revision's own, and its frozen bindings hold the backings and descriptors it was
    recorded with, so it stays valid whatever is published after it. A buffer that grows does so by versions
    (`VersionedBuffer`, below), never in place.
-   **Tickets admit candidates** (`RenderGraph::PrepareRevisionTicket`). A revision-driven epoch's ticket prepares nothing
    from live state. For its slot, the host's thread admits each revision recording still held by its caller as a candidate
    (`AddPersistentTicketCandidate`: `SynchronousAdmission::Prepare` for the recording's publication). A recording finished
    while a ticket waits is added to that ticket before its `done` runs, so a revision the caller publishes afterwards is
    always admitted. Candidates are an append-only list: the host's thread pushes, the submitting thread reads, with no lock.
-   **Submission.** `SubmitEpoch(epoch, beforeSubmit, owner, recording)` binds the revision's recording for the ticket's slot
    (`RenderGraph::BindPersistentTicketRecording`): the ticket takes the candidate admitted for exactly that recording.
    Completion commits it and abandons the other candidates (closed executions allow several admissions pending, abandoned
    in any order). A recording no candidate admits is an error: the ticket is discarded and the call throws. A
    revision-driven epoch is never prepared again, and its currency check and adoption do not apply.
-   **Not kept.** A recording made for host data is neither taken from nor added to the reuse sets. A pass's revision hash
    (`InvocationRevisionHash`) reads live state, not the host data, so the key could not tell two revisions apart.
-   **Statistics.** `AsyncStats::revisionSlotsRecorded`, `revisionRecordings` and `revisionSubmissions`.
-   **Test.** `PersistentVulkanHostTests revision`:
    -   a revision of the full count recorded for the 3 ring slots and submitted 6 times;
    -   the target grown as a new version (`VersionedBuffer::GrowBytes`) and a revision naming it recorded; the old revision,
        which names the first version, still submitted twice after it (each ticket admits both), then the new one;
    -   a revision of half the count, on the grown version, which writes its half and leaves the rest;
    -   an epoch submitted without its recording rejected, and the host going on;
    -   12 submissions bound, none prepared again.

### Versioned buffers

`org::VersionedBuffer` (`Resources/Buffers/VersionedBuffer.h`) is a buffer that grows by versions. `GrowStructured` and
`GrowBytes` make the next version (`Buffer::UnmaterializedLike`: the same heap, access, structure, descriptors and name, at
the new size), materialize it and publish it as current. The old version is untouched, and it is released when the last
thing holding it does: a revision, a recording's bindings, a frame in flight.

-   **Declared as a resolver.** Passes declare the `VersionedBuffer` itself. A preparation resolves the version its host data
    names (`IResourceVersions::Find`), else the current one. Each version has its own declaration state, made once.
-   **The live path.** An epoch that is not revision-driven resolves the current version, and a growth is followed by
    `PersistentGraphHost::NoteNewVersions`: a ticket prepared before resolved the old version, so it is no longer current
    (it is prepared again, or takes a kept recording). Nothing waits, unlike `MutateBackings`, because nothing changed in
    place.
-   **Thread contract.** `Publish` and `Grow*` on the owner's thread; `Current`, `Get` and the resolver from any thread (the
    current version is an atomic `shared_ptr`).
-   A buffer without a backing has nothing a capture could see, so sizing an unmaterialized version needs no backing
    mutation scope (`RequireBackingMutationAllowed`).

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
    list (`RecordStagedUploads`), one copy per entry, after one barrier against the work before them. The barrier
    is recorded only when there is a copy: an epoch whose feature staged nothing records nothing.
-   **Per-frame values need no upload.** A value written whole every frame (a block, a counter to zero) can go
    into a latch, and a pass of the epoch copy it from there: the latch region's place depends only on the frame
    slot, so the copy is recorded with the ticket, ahead, and the submission records nothing for it.
-   **Ordering.** Copies land in submission order. A copy that overlaps an earlier one in the same list,
    whether from an earlier batch or from the upload pass's queued copies, waits for it through a barrier.
    This happens when a batch's epoch was never submitted and its replacement follows it into the next list.
    Steady state has no overlaps.
-   **Lifetime.** A batch is held until the slot whose submission recorded it has retired. A producer reuses
    a batch once it holds the only reference.

### Recorded ahead by the producer

Recording is a copy command per entry, on the submitting thread. A producer can record its batch itself, on its own
thread, as soon as it has staged it: `StagedUploadBatch::Record(device)` writes the copies into a command list of the
batch's own, starting with a full barrier against the work before it. The owner then hands it over with
`IUploadService::SubmitRecordedUploads`, and the submission takes the list as it is (`RecordPendingUploads` returns it
in `recordedLists`, ahead of the host's own list).

-   **Its place.** The list goes to the queue ahead of every copy the owner records, so the service accepts a batch only
    when that is its place in submission order: nothing staged, queued, posted or copied (a resize) before it is still
    waiting. Otherwise `SubmitRecordedUploads` returns false and the caller submits the batch as staged.
-   **Its targets.** A recorded copy names each target's backing as it was. `BufferBase::BackingReleaseSerial` counts
    every backing any buffer releases; a batch recorded under another value is not accepted, and one outdated between
    its acceptance and the submission is recorded by the owner instead, first.
-   **Recording.** `StagedUploadBatch::RecordCopies` is the one recorder, for both: copies in order, with a barrier only
    before a copy that overlaps one since the last barrier (per backing, an interval lookup).
-   **Statistics.** `AsyncStats::recordedUploadLists` counts the lists submitted as recorded.

## Tests

`tests/PersistentGraphTests.cpp` builds a program whose reader and writer are registered in the opposite order
to their epochs and checks:

-   the whole-frame order;
-   each epoch executable's placements: its own passes and the untagged one, and nothing else;
-   `ForEpoch` caching, and null for an absent epoch;
-   admission of both splits in sequence through one `SynchronousAdmission`;
-   that a program without tags has no epochs.

`tests/PersistentVulkanHostTests.cpp`, run as `reuse` (`OpenRenderGraphPersistentVulkanHostTests.Reuse`), covers recording
reuse: steady frames take their kept recordings (none prepared again) and each submission of the same lists writes that
frame's uploaded value; a pass's revision alternating inside `beforeSubmit` every frame, once both variants are kept per
slot, makes every stale ticket adopt a kept recording, with none prepared again and every frame's output right.

A second test covers closed executions:

-   closure (the exit barriers to home);
-   plan caching;
-   several admissions pending at once;
-   out-of-order commit and abandon;
-   the error for a program that is not closed.
