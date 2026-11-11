# Embedding ORG: hosts, epochs and in-stream execution

A host that embeds ORG inside another engine (CS inside Skyrim's D3D11 renderer, through DXVK) often cannot submit the
graph once per frame. It submits at several fixed points of its own frame, and each point's work must land between the
host's own GPU work before and after it: the shadow maps, a depth prepass, light culling, the main colour pass. Each
submission point is an **epoch**.

`org::GraphHost` is ORG's embedding layer. It owns:

-   a graph per epoch,
-   the async machinery that prepares their executions ahead,
-   the in-stream lowering that hands them to the host's own command stream,
-   the interop options for what the host shares with the graphs.

## Epochs as graphs

-   **Tagging.** A pass declares its epoch with `ExternalPassDesc::Epoch(id)`. Untagged passes (`persistent::AllEpochs`,
    the default) belong to every epoch. A host with no tags runs one graph.
-   **Order.** `GraphHost::Desc::epochOrder` names the order of the epochs within one frame.
-   **One graph per epoch.** `GraphHost` builds a `RenderGraph` per epoch of the order, from the same extension
    factories. Each graph keeps its epoch's passes and the untagged ones (`RenderGraph::SetPassEpochFilter`). Its structure
    holds only those passes' resources, so admitting an epoch costs what the epoch touches, not the whole frame.
-   **Running an epoch.** `GraphHost::ExecuteFrame(hostData, beforePrepare, epoch)` runs one epoch synchronously.
    `SubmitEpoch(epoch, beforeSubmit)` submits it while async epochs are on (below).
-   **Rebuilding one epoch.** `RequestRebuild(epoch)` rebuilds that epoch's graph alone: the host thread detaches it, the
    owner thread builds the new graph, and the old one is released once its own submitted work completes. There is no
    device wait, and the other epochs keep their graphs and builds (`BuildGeneration(epoch)`). `RequestRebuild()`
    rebuilds all of them.
-   **Graph access.** `Graph(epoch)` gives an epoch's graph, `Graph()` the first one; `GraphCount`, `GraphAt`, `EpochAt`
    enumerate them. The runtime's services (uploads, descriptors, tasks) are shared by all of them.

### Failure isolation

A failure while one epoch's graph is built or prepared stays with that epoch.

-   **What counts.** An exception from the epoch's build (extension factories, `PrepareForBuild`, compile, `Setup`) or from
    its preparation (a pass's `Prepare`, a bindless view it resolves, its recording). This covers the host thread's ticket
    preparation and `ExecuteFrame`'s `Update` and `Execute`.
-   **What happens.** The epoch is marked failed. A failed build keeps the epoch's previous graph when it had a working one
    (`EpochFailure::keptGraph`); otherwise the epoch runs nothing. `SubmitEpoch` and `ExecuteFrame` then throw
    `EpochFailedError` for it, having submitted nothing. The host thread keeps running and the other epochs keep their
    graphs and tickets. Recording requests for the epoch complete with the failure.
-   **How the owner hears.** `PollEpochFailures()` hands each failure to the callback set with
    `SetEpochFailedCallback`, on the owner thread. Call it where nothing of the epoch is in progress, for example before a
    frame decides what its epochs run. A failure of a build the epoch has since replaced is dropped.
-   **Recovery.** A failed epoch is not built again every frame. The owner fixes the cause (for example drops the pass that
    failed) and calls `RequestRebuild(epoch)`. A build that succeeds clears the failure.
-   **Host-wide errors.** A failed GPU completion wait or a lost device still ends the host thread. `SubmitEpoch` then
    throws for every epoch, as before.

### What ties the graphs together

-   **Shared queue timelines.** Every graph signals the same timeline per queue (`RenderGraph::SetSharedQueueTimelines`).
    Two epochs on one queue are then ordered by the queue itself, with no wait between them. The ledger drops waits on the
    timeline of the queue a batch runs on.
-   **One sequencer.** Every graph on a runtime admits against the runtime's `ResourceSequencer`. Its state, access and
    alias ledgers say, for every resource, which execution touched it last, on which queue, and in which state it left
    it. An epoch's entry into a resource is admitted from that, whatever ran before it, including when an earlier epoch
    was skipped that frame.
-   **A shared alias arena.** With `Desc::sharedAliasArena` (default on), every graph places its transients into the
    same pools (below).

### Contract for hosts

-   A resource read in an epoch must have been written in that epoch or an earlier one, as in any frame. An aliased
    transient keeps no contents across executions: each execution enters it from Undefined.
-   A pass instance belongs to one epoch. Work that runs in two epochs is registered twice, with its epoch fixed per
    instance.

## Open executions

An execution is **open** by default: it leaves every resource as its last use left it, and the next execution takes it
from there through the ledgers.

-   **Admission at submission.** An execution's body is recorded ahead, from a plan of its publication. At submission,
    under the sequencer (`RenderGraph::HoldPersistentSequencer`), it is admitted against the ledgers as they stand then.
    That gives the entry barriers each batch needs, any releases on another queue, and the waits on earlier executions.
    It is then submitted and committed with nothing admitted in between.
-   **The body is ledger-independent.** An open plan keeps the entry barriers out of the body (`PreparedBatchBarriers`
    `textures`/`buffers`/`acquired`). What is left (`beforePass`, `afterPass`, an aliased backing's entry from
    Undefined) depends on the compiled graph alone, so a body can be recorded ahead and submitted again.
-   **Handoff lists.** Each batch with entry barriers gets a short list ahead of it: a primary submitted with it, or a
    secondary its segment runs first in the host's stream. A texture an earlier execution left on another queue is
    released there first, by a list submitted on that queue, and the consumers wait for it.
-   **Predicted admission.** The ledger's view of an executable's entry resources repeats from frame to frame. Each
    executable keeps the last entry-only plan it was admitted with and the ledger's fingerprint of each entry resource
    (`StatePrediction`). While the fingerprints hold, the admission takes that plan as it is (`FrameAdmission::
    sharedBarriers`). The access ledger predicts the waits the same way (`AccessPrediction`). The handoff lists a ticket
    recorded from a predicted plan are kept per frame slot and submitted again by the slot's next ticket that takes the
    same plan.
-   **Cost.** In steady state, admission plus commit is a few microseconds per epoch on the submitting thread.

## Async epochs: tickets

`GraphHost::SetAsyncEpochs(epochs)` moves everything except submission onto a host thread of ORG's own. The render
thread calls `SubmitEpoch(epoch, beforeSubmit)` at each point instead of `ExecuteFrame`.

-   **Tickets.** For each epoch, the host thread prepares a ticket (`RenderGraph::PreparePersistentTicket`) about one frame
    ahead:
    -   it reserves a frame slot, and waits there for the slot's previous submission, never on the render thread;
    -   it runs update and invocations;
    -   it records the epoch's command lists from a plan, and publishes the ticket in that epoch's atomic cell.
-   **Submission.** `SubmitEpoch` on the render thread:
    1.  takes the ticket from the cell (waiting only if it is not ready);
    2.  runs `beforeSubmit`, the feature's commit;
    3.  checks the passes' revision hashes against the ticket (`RenderGraphPass::RevisionHash`, a typed pass's `Revision`).
        A stale ticket takes a kept recording (below) or is prepared again for the same slot;
    4.  admits the ticket (`AdmitPersistentTicket`), records the pending uploads into the slot's upload list, and submits
        the uploads and the ticket with the handoff lists (`SubmitPersistentTicket`), then commits.
-   **Uploads.** The uploads go to the graphics queue, ahead of the ticket. A batch on another queue waits for them only
    when it uses what they write.
    -   **Queue-scoped uploads** (`Desc::queueScopedUploads`, default on). A staged upload whose destination the ticket first
        uses on another queue is recorded into that queue's list instead, just ahead of that queue's first batch. That list
        waits for what a write of the destination needs against earlier executions (`FrameAdmission::boundaryAccessWaits`).
    -   When a destination is not known, nothing is planned, and every batch off the graphics queue waits for the uploads.
-   **Completion.** The host thread receives the submission through a single-producer mailbox. It retires, reads
    statistics and releases deferred resources, in submission order.
-   **Constraints on a ticket.** Its recording may not depend on anything the render thread produces at the epoch.
    Per-frame values come from latches.

### Slot rings and epochs that run many times a frame

Each async epoch takes the frame slots of a ring of its own, in turn: `framesInFlight` slots, after the rings of the async
epochs before it. The host thread waits for a slot's last run on the GPU before it prepares the slot again. That wait is
the CPU's backpressure, and it is only safe on work the device has. A host that holds submissions in a stream until the
frame's end has not handed the current frame's runs to the device. CS, for example, puts its epochs into DXVK's command
list, and DXVK submits it at Present. An epoch that runs more than `framesInFlight` times in one such frame makes the host
thread wait for one of its own held runs, while the render thread waits for the host thread at the epoch's next run. That
is a deadlock.

`Desc::epochRuns` lists such epochs with the most runs a frame they are sized for:

-   **Capacity.** The epoch's ring has `framesInFlight * runsPerFrame` slots. `FrameSlots()` counts them
    (`GraphHost::FrameSlotsFor(desc)` before the host exists, to size the runtime). It is fixed when the host is made, so
    a `LatchBlock` sized from it, and every other per-slot table, stays right.
-   **Limit.** A host has at most `Runtime::kMaxSlots` (4096) frame slots over all its epochs. Slot indices are 32-bit
    everywhere (upload pages, kept recordings, tickets), so the bound is not a type limit. It keeps the per-slot tables
    in check: each slot adds a range to the statistics query pools (two timestamps per pass), an upload page list, and a
    region to every `LatchBlock` sized from `FrameSlots()`. Every run a frame of every epoch costs one slot per frame in
    flight. A run cannot share another run's slot: it is a submission of its own, with its own latch region, its own
    kept recording (a list cannot be pending twice) and its own slot wait.
-   **Growth.** The ring starts with `framesInFlight` slots. The next ticket takes the least recently submitted slot in
    use. It waits for that slot only when the slot's last run is done, or is at least `max(framesInFlight - 1, 1)` frame
    boundaries old. Otherwise the ring grows by one slot. Frame boundaries are `SealSubmitted` calls: the host declares
    there that everything submitted so far is on the device. So waits target only runs of earlier frames, with the same
    lead as an epoch that runs once a frame, and the ring settles at about its runs a frame times `framesInFlight`.
-   **Full.** At capacity, a slot whose last run is sealed is waited for: that is safe, only more backpressure. When every
    slot holds a run the host has not sealed, the epoch's next ticket waits for the next seal. Meanwhile `SubmitEpoch`
    throws `EpochRingFullError` (an `EpochFailedError`): nothing was submitted, the epoch has not failed, and the caller
    skips that run. One warning is logged per epoch.
-   **Contract.** A host that lists an epoch in `epochRuns` calls `SealSubmitted` at every frame boundary. Epochs not
    listed keep a ring of `framesInFlight` slots and always wait, as before.
-   **Statistics.** `AsyncStats::ringGrowths` and `ringFull`.
-   **Test.** `PersistentVulkanHostTests rings` (and `rings open`): an external epoch run six times a frame with two
    frames in flight, its executions held until the frame's end, runs without a deadlock; nine runs between two seals
    skip the ninth. `rings-wide` does the same with 260 runs a frame in a ring of 272 slots.

## Recording reuse

With `Desc::reuseRecordings` (`RenderGraph::SetPersistentRecordingReuse`), a ticket's recording is kept and submitted again
while what it was recorded for holds, so a steady epoch is never recorded again.

-   **What is kept.** A segment with nothing late-bound (no rebound slot, no incoming effect), every pass reporting its
    revision (`void Revision(std::vector<uint64_t>&) const`: what its recording depends on beyond its bindings), and every
    pass replayable (`PreparedPass::IsReplayable`: no submission, completion or abandonment callback, no external signal,
    no dependency lifecycle effect). Pipelines and other objects a recording binds are retained by the recording itself
    (`RecordingRetention`, on `OwnedRecordingList`).
-   **The key.** Per (epoch, frame slot), up to `RenderGraph::kKeptRecordingVariants` recordings, the most recently used
    first, each with what it was recorded for: the publication, the bindings, the descriptor heaps, the host tag and every
    pass's revision.
-   **Reuse and adoption.** The owner thread, preparing a ticket for what a kept recording was recorded for, takes it
    instead of recording. At submission, a ticket that `beforeSubmit` made stale takes the kept recording of what its
    passes depend on now (`RenderGraph::AdoptKeptPersistentRecording`), and is not prepared again.
-   **One submission at a time.** A kept recording belongs to one frame slot, and a slot's ticket is prepared after the
    slot's previous submission completed, so a list is never in two submissions at once. BasicRHI's Vulkan lists begin
    without `ONE_TIME_SUBMIT` for this.
-   **What a kept recording leaves out.** Profiler GPU zones and GPU statistics queries the statistics service owns. With
    `GraphHost::SetRecordingTimestamps` a kept list owns its timestamp queries instead
    (`IStatisticsService::CreateOwnedTimestamps`), read at its slot's completion.
-   **Statistics.** `AsyncStats::recorded`, `kept`, `reused` and `adopted`.

A kept recording keeps whatever its passes' prepared data hold. A host that versions resources by reference count must
therefore not cap its versions. For variants to come back, a host's pass revisions must come back too: a shape that
alternates should republish its earlier identity rather than a new one.

## Revision-driven epochs

`SetAsyncEpochs(epochs, revisionEpochs)` makes the listed epochs revision-driven. Their recordings are made ahead, for a
revision the host's scene supplies (CS's DCLF scene revision), and nothing about them is recorded at submission.

-   **Slot rings.** Each async epoch has a ring of frame slots of its own (above). A recording has one per slot of the
    ring's capacity, so it covers every ticket, also after the ring grows.
-   **`RequestEpochRecording(epoch, hostData, done)`.** Lock-free, from any thread, safe across a build. The host thread
    records the epoch for `hostData` into every slot of its ring, each once the slot's last work is done, and calls `done`
    once with the recording, the error that stopped it, or `EpochRecordingDropped` when a build stopped the host thread
    first.
-   **Candidates are recordings.** A revision-driven ticket prepares nothing from live state. The revision recordings its
    caller still holds are its candidates, and `SubmitEpoch(epoch, beforeSubmit, owner, recording)` binds one. The
    submission admits whichever recording it takes. A live epoch's `beforeSubmit` may choose one too
    (`UseEpochRecording`).
-   **Immutable inputs.** A recording for a revision resolves the graph's resolvers through it (`ResolverCaptureContext` of
    `IHostExecutionData`), and its frozen bindings hold the backings and descriptors it was recorded with. A buffer that
    grows does so by versions (`VersionedBuffer`), never in place.
-   **Statistics.** `AsyncStats::revisionSlotsRecorded`, `revisionRecordings` and `revisionSubmissions`.

### Versioned buffers

`org::VersionedBuffer` grows by versions. `GrowStructured` and `GrowBytes` make the next version, materialize it and
publish it as current. The old version is released when the last thing holding it does: a revision, a recording's
bindings, a frame in flight. Passes declare the `VersionedBuffer` itself; a preparation resolves the version its host
data names (`IResourceVersions::Find`), else the current one. After a growth, `GraphHost::NoteNewVersions` makes the
tickets prepared before it stale.

## In-stream execution

`GraphHost::SetExternalEpochs(epochs)` (`RenderGraph::SetExternalPersistentEpochs`) marks async epochs whose graphics
work the host executes itself, in its own command stream. CS runs them inside DXVK's command list through DXVK's
external-commands callbacks, so ORG's work does not end DXVK's list.

-   **The stream.** A batch is in the host's stream when its queue is the host's graphics queue or another slot on the same
    native queue (`RenderGraph::InHostStream`: a device that aliases an unset queue kind to the graphics queue). Its batches
    are recorded as secondaries (a pool per stream slot). Batches on other queues are recorded as primaries and submitted by
    ORG on their queues, at the ticket's submission.
-   **Segments.** `TakeExternalExecution` returns an `ExternalEpochExecution`: an ordered list of segments. Each holds the
    lists recorded ahead of it, its batches (secondaries, or late batches recorded at execution), and the waits and
    signals of the host submission that holds it. A segment's signals include the graph's values for its batches on the
    graphics timeline, so retirement, slot waits and later tickets work as for a submitted epoch.
-   **Where a segment starts.** A new segment starts where a stream batch waits for a batch off the stream.
-   **When it splits.** A segment is marked `split` when that off-stream batch waits, directly or through others, for a
    stream batch. The host must start a new submission for it, or the submission would wait for itself. DXVK attaches an
    external command's waits to the whole submission it lands in, so a split is a DXVK flush.
-   **Waits dropped in the stream.** Waits on a stream slot's timeline are dropped: queue order covers them.
-   **`exitSplit`.** Set when a batch off the stream waits for the stream. The host's wait for the epoch's off-stream work
    then starts a submission of its own.
-   **Uploads.** With queue-scoped uploads the off-stream users' uploads go to their own queue. Otherwise the stream's
    uploads get a segment of their own, with a value the off-stream users wait for.
-   **Execution.** `ExternalEpochExecution::Execute(segment, primary)`, on the host's thread, once per segment and in
    order. It binds the graph's heaps before each execution of secondaries, executes them, and records each late batch.
    Everything the primary had bound is undefined afterwards.
-   **Not executed.** A host that cannot execute a segment calls `SignalExternalExecutionUnexecuted(execution, first)`. That
    is a submission of no lists with the waits and signals of each segment from `first` on, and its late passes are
    abandoned.
-   **Passes recorded at execution.** A pass declared with `ExternalPassDesc::RecordAtExecution()` is alone in its batch,
    which must be in the stream. The ticket keeps its invocation and barrier plan, and the host's execution records it
    there: for work only known then, such as a CPU cull's draws.
-   **Retirement.** No ORG queue sees the host's stream. So the runtime's deletion and descriptor-release frontier also
    covers every value given out on the host's shared timelines (`Runtime::RegisterAssignedTimeline`). An object released
    while an in-stream list that names it is still waiting in the host's stream outlives that list.
    -   The statistics' shared query pool and its readback buffers are replaced when a build registers a new pass. Each
        list holds the set it recorded with (`QueryRecordingContext::sharedQueries`), and a replaced set is deleted only
        when the last such list is released. This also covers lists recorded ahead, which the frontier cannot cover.
    -   A list whose set was replaced while it recorded writes and resolves no more queries. Its resolve would otherwise
        wait on the GPU for queries the new pool never wrote.
    -   Test: `TestAssignedTimelineRetirement` and `TestStatisticsPoolRetirement` (`tests/RuntimeTests.cpp`).
-   **Test.** `PersistentVulkanHostTests`:
    -   `external`: a late pass in the stream.
    -   `stream` / `stream open`: a write on a compute queue and a copy on a copy queue aliased to the graphics one, with
        stream uploads (two segments, one split) and queue-scoped uploads (one segment, no split).
    -   `stream open home`: external home states, below.

## Interop options

Nothing in ORG requires the following options. They are for hosts whose own GPU work shares resources or queues with
the graphs.

-   **Queue boundary** (`Desc::queueBoundary`, `RenderGraph::SetExternalQueueBoundary`). A full barrier at the entry and/or
    exit of each execution on the queues it shares with the host. It orders the host's own work around the graph's on that
    queue, which the graph's ledgers do not see.
-   **External home states** (`Desc::externalHomeStates`, `RenderGraph::SetPersistentExternalHomeStates`). Every texture the
    host owns (`Resource::HostOwned`: `ExternalTextureResource`) leaves every execution in its home state on the graphics
    queue: its declared incoming state, or the state its first use expects. The host then finds its own resources where
    it left them.
    -   A last use on another queue releases the texture to the graphics queue. The plan's `homeAcquires` acquire it there,
        after its producers: in a list submitted after them, or a trailing segment of an in-stream execution. That segment
        is split only when no segment since the last split already waits for the same points.
    -   An execution takes a host-owned texture from the graphics queue at entry.
    -   On Vulkan, a pass that touches an exclusive resource stays on the graphics queue anyway. So the queue transfer
        applies to textures the host shares concurrently, and to D3D12.
    -   Everything else flows through the ledgers.
-   **Closed executions** (`Desc::closedExecutions`, `RenderGraph::SetPersistentClosedExecutions`). The fallback for a host
    that cannot tolerate any state carried between executions.
    -   Every resource an execution touches returns to its home state after its last use.
    -   Every execution starts with a full barrier on each queue's first batch.
    -   An execution's admission is then independent of what ran before, and cached per executable. Several admissions may
        be pending and committed or abandoned in any order.
    -   A texture used from more than one queue inside one closed execution is rejected.

## Shared alias arena

A host's graphs place their transients into one `PersistentAliasArena` (`RenderGraph::SetPersistentAliasArena`;
`GraphHost::AliasArena`).

-   **Packing.** Each graph packs its own transients from offset 0 of a pool (`PlanPersistentAliasPlacement`, unchanged).
    Transients of different epochs therefore overlap the same memory.
-   **Size.** A pool is as large as the largest requirement of any graph placed in it, with headroom. It shrinks when every
    graph needs less than half of it.
-   **Safety.** The runtime's alias ledger, which every graph shares through the sequencer. A placement's first use waits
    for the last use of any overlapping placement by another graph's earlier execution, and enters from Undefined.
-   **Growth.** A graph that needs more grows the pool into a new generation. The other graphs' placements keep the
    earlier generation alive and share nothing with the new one. `RenderGraph::PersistentAliasPlacementsCurrent` reports
    that, and `GraphHost` rebuilds those graphs, which places them in the current generation and frees the old one.
-   **Test.** `PersistentVulkanHostTests arena`: two epochs whose scratch buffers share one pool at offset 0 (the second's
    write on the compute queue). Each epoch's copy reads what it wrote, synchronous and async.

## Latches

`org::LatchBlock` is a host-visible buffer with one region per frame slot. A recording refers only to the slot's offset
(`RecordingContext::FrameSlot()`), and the render thread writes the values into the region before it submits. A pass whose
recording would otherwise bake a per-frame value (a count, a matrix, a dispatch size) reads it from its latch.

## Staged uploads

`org::runtime::StagedUploadBatch` lets a producer (a worker thread, or the render thread) write upload data straight into
mapped upload pages: `Stage(target, dstOffset, size[, data])`. `IUploadService::SubmitStagedUploads` hands a batch over on
the owner thread.

-   **Async epochs.** The host records the waiting batches straight into the ticket's upload list
    (`RecordStagedUploads`), one copy per entry. A copy that overlaps an earlier one in the same list waits for it through
    a barrier.
-   **Recorded ahead by the producer.** `StagedUploadBatch::Record(device)` writes the copies into a command list of the
    batch's own, on the producer's thread. `IUploadService::SubmitRecordedUploads` hands it over when that is its place in
    submission order. A batch whose targets' backings changed since (`BufferBase::BackingReleaseSerial`) is recorded by the
    owner instead.
-   **Lifetime.** A batch is held until the slot whose submission recorded it has retired.

## Tests

-   `tests/PersistentGraphTests.cpp`:
    -   epoch splits and their admission through one `SynchronousAdmission`;
    -   closed executions (closure, plan caching, out-of-order commit and abandon);
    -   graphs sharing a sequencer;
    -   external home states: a host-owned texture used on the compute queue is taken from and given back to the graphics
        queue each execution.
-   `tests/PersistentVulkanHostTests.cpp` modes:
    -   `async`, `reuse`, `revision` and `external`, each also `open`;
    -   `split` (two epochs, two graphs, one rebuilt alone);
    -   `stream` and `stream open home`;
    -   `arena`;
    -   `rings` and `rings open` (an epoch that runs many times a frame, `Desc::epochRuns`), and `rings-wide` (a ring of
        272 slots, so slot indices pass 255, with recording reuse).
