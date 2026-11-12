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
    rebuilds all of them. Every request takes a reason; the build's log line names the reasons it answers, with counts
    (`GraphHost: epoch N's graph built again ... why: ...`), and a build point that builds several graphs logs a total.
-   **Rebuilding only what changed.** Each graph builds every extension made into it, so a rebuild costs every one of their
    gathers. An extension that only some epochs use names them (`AddExtension(id, factory, epochs)`): the other graphs
    never make it, and adding it again rebuilds only its epochs. An extension made into a graph of one epoch gathers only
    that epoch's passes (`rg.PassEpochFilter()`). A name whose meaning changed (a resource registered under a new name)
    rebuilds only the graphs that looked it up (`RequestRebuildFor(id)`, `RenderGraph::LookedUp`).
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
-   **Shared body plans.** Since a body's barriers depend only on the compiled graph, the backings and the execution kind,
    `SynchronousAdmission::PreparePlan` plans them once per backing set and every later plan shares that plan
    (`CachedBody`). The ledgers' grids are read for it once per graph build, not once per ticket.
-   **Handoff lists.** Each batch with entry barriers gets a short list ahead of it: a primary submitted with it, or a
    secondary its segment runs first in the host's stream. A texture an earlier execution left on another queue is
    released there first, by a list submitted on that queue, and the consumers wait for it.
-   **Predicted admission.** The ledger's view of an executable's entry resources repeats from frame to frame. Each
    executable keeps the last entry-only plan it was admitted with and the ledger's fingerprint of each entry resource
    (`StatePrediction`). While the fingerprints hold, the admission takes that plan as it is (`FrameAdmission::
    sharedBarriers`). The access ledger predicts the waits the same way (`AccessPrediction`). The handoff lists a ticket
    recorded from a predicted plan are kept per run lease index and submitted again by the index's next ticket that takes
    the same plan.
-   **Cost.** In steady state, admission plus commit is a few microseconds per epoch on the submitting thread.

## Async epochs: tickets

`GraphHost::SetAsyncEpochs(epochs)` moves everything except submission onto a host thread of ORG's own. The render
thread calls `SubmitEpoch(epoch, beforeSubmit)` at each point instead of `ExecuteFrame`.

-   **Tickets.** For each epoch, the host thread prepares a ticket (`RenderGraph::PreparePersistentTicket`) about one frame
    ahead:
    -   it opens a run lease for it (below), which waits for nothing;
    -   it runs update and invocations;
    -   it records the epoch's command lists from a plan, and publishes the ticket in that epoch's queue.
-   **Tickets ahead.** By default an epoch has one ticket in hand: the next is prepared once the last one was submitted.
    `GraphHost::SetEpochTicketsAhead(epoch, most)` lets a live epoch that runs several times a frame keep up to `most`,
    as many as it ran in the last frame (counted between seals). Its back-to-back runs then find their tickets ready.
    -   The extra tickets are prepared while the epoch's earlier tickets are submitted. So the epoch's passes' `Prepare`
        and `Revision` must read nothing its commits write (CS's frame anchors follow that rule).
    -   The host thread first prepares a ticket for every epoch with none in hand, then one ticket ahead per loop pass,
        the epochs taking turns. Messages are read between them.
    -   Only the first ticket queued has its upload lists begun on the host thread. The render thread begins the others'
        when it submits them, so a queue of tickets does not keep a command pool open per list.
    -   A stale ticket's replacement goes to a slot of its own, which its submission takes. The tickets queued behind it
        that are stale too are dropped then and prepared again (`AsyncStats::staleAhead`).
-   **Submission.** `SubmitEpoch` on the render thread:
    1.  takes the next ticket from the epoch's queue (waiting only if none is ready);
    2.  runs `beforeSubmit`, the feature's commit;
    3.  checks the passes' revision hashes against the ticket (`RenderGraphPass::RevisionHash`, a typed pass's `Revision`).
        A stale ticket is prepared again for the same run lease, whose latch regions `beforeSubmit` just wrote;
    4.  admits the ticket (`AdmitPersistentTicket`), records the pending uploads into the lease's upload list, and submits
        the uploads and the ticket with the handoff lists (`SubmitPersistentTicket`), then commits.
-   **Uploads.** The uploads go to the graphics queue, ahead of the ticket. A batch on another queue waits for them only
    when it uses what they write.
    -   **Queue-scoped uploads** (`Desc::queueScopedUploads`, default on). A staged upload whose destination the ticket first
        uses on another queue is recorded into that queue's list instead, just ahead of that queue's first batch. That list
        waits for what a write of the destination needs against earlier executions (`FrameAdmission::boundaryAccessWaits`).
    -   When a destination is not known, nothing is planned, and every batch off the graphics queue waits for the uploads.
-   **Completion.** The host thread receives the submission through a single-producer mailbox. It retires each run's lease
    once the run's points are reached: it reads the run's statistics and frees what the lease held.
-   **Retirement per graph.** Each loop pass reads every shared queue timeline's completed value once for all the graphs
    (`RenderGraph::CompletedTimelineValues`). A graph returns at once while its timelines' submitted values are unchanged and
    every queue's completed value is below the lowest value anything it holds waits for (`RetireWatch`).
-   **Constraints on a ticket.** Its recording may not depend on anything the render thread produces at the epoch.
    Per-frame values come from latches.

### Run leases

A **run** is one submission of an epoch: a `SubmitEpoch`, an `ExecuteFrame`, or a client's external run. Each run holds a
**run lease** (`RunRef`: a lease index, a generation, and a copy index for a published copy). The lease owns what the run
uses beyond its recording:

-   its upload lists (a primary, an external epoch's secondary, and one per compute or copy queue), requested from the
    host-wide pools when the ticket is prepared;
-   its frame arena regions (`LatchBlock`, below);
-   its statistics range (the statistics service's ranges are indexed by lease index, and grow with them:
    `IStatisticsService::EnsureFrameCapacity`) and its upload pages (`IUploadService::ProcessDeferredReleases(index)`);
-   the uploads' staging, the submission's handoff lists and the caller's `resourceOwner`.

A lease retires once its run's points are reached. A retirement pass on the thread that sees completions (the host thread
with async epochs, the render thread without) reads the run's statistics, calls the completed-frame and retired callbacks,
recycles the lists (`CommandListPool::RecycleForNextRequest`), frees the arena regions and returns the index to a last in
first out free list. So lease indices stay dense, and a table indexed by them follows the runs in flight.

-   **Backpressure.** The only blocking wait for the GPU is frame pacing: `ExecuteFrame` waits for its run of
    `framesInFlight` frames back (a frame is each epoch of the order once). A ticket's lease waits for nothing: when no index
    is free, the table grows (up to `Runtime::kMaxSlots` runs in flight, where it waits for the oldest sealed run).
-   **Why.** A host that holds submissions in a stream until the frame's end has not handed the current frame's runs to the
    device. CS, for example, puts its epochs into DXVK's command list, and DXVK submits it at Present. A wait for a run of
    the same frame would never end, because the render thread that would flush it waits for the host thread. Frame
    boundaries are `SealSubmitted` calls: the host declares there that everything submitted so far is on the device. Only a
    run behind a seal (or done) is ever waited for.
-   **`CurrentRun()`.** The run the host may write per-run data for: the ticket's inside `beforeSubmit`, the execution's
    inside `ExecuteFrame`'s `beforePrepare`, otherwise the active external run. `RecordingContext::Run` names the same run,
    so a writer and a recorder agree on everything keyed by it.
-   **External runs.** `BeginExternalRun(timeline, value)` opens a lease for a client that records work reading its latch
    regions outside the host's executions (DCLF's CPU pipeline records into DXVK's own list). The lease retires once the
    client's `timeline` reaches `value`, which it signals after its last work. Its upload pages are taken at the call, and
    every deferred deletion and descriptor release from then on waits for the point too. `EndExternalRun` ends it as
    `CurrentRun`; opening another replaces it, and both stay leased until their points.
-   **Statistics.** `GraphHost::TakeRunMemoryStats`: leases in flight and their peak, lease indices made, external runs,
    published copies' leases, statistics ranges, and the frame arena's chunks and live bytes.
-   **Test.** `PersistentVulkanHostTests leases` (and `leases open`): an external epoch run six times a frame with two
    frames in flight, its executions held until the frame's end, runs without a deadlock, each run with a lease of its own,
    and four frames' worth of runs between two seals all go through. `leases-wide` does the same with 260 runs a frame. The
    test also checks the frame arena's regions.

### The frame arena

Per-run constants (latches) live in one `FrameArena` per runtime (`Runtime::Arena`): host-visible, persistently mapped,
in chunks of 256 KiB (more for a larger region). A chunk never moves and has one shader resource view, so a region's
`{view, offset}` stays valid.

-   **`LatchBlock`** is a view into the arena with no memory of its own. A leased run's region is allocated the first time
    anything asks for it (`At`, `Offset`, `Write`): the writer and the recorder name the same run, so whichever comes
    first allocates it. It is freed when the run's lease retires. A region is zeroed when it is allocated: a run that reads
    a block without writing it reads zeros.
-   **Stable runs.** A published copy's run (`RunRef::Copy`) and an index outside a lease (`RunRef::Unleased`, a graph
    executed on its own) get a pinned region per index, kept for the block's life, in pinned chunks of their own. A
    pinned region a dying block frees is reused only once the GPU reached the runtime's frontier at the free.
-   **Dedicated blocks** (`LatchBlock::CreateDedicated`): a buffer of the block's own with one region, for a client's own
    staging buffer that it keeps from reuse itself.
-   **Size.** The arena holds what the runs in flight wrote. `SealSubmitted` trims chunks that stayed empty for a while,
    keeping one spare.

## Live and published recordings

With `Desc::reuseRecordings` (`RenderGraph::SetPersistentRecordingReuse`), tickets record replayable recordings: a segment
with nothing late-bound (no rebound slot, no incoming effect), every pass reporting its revision (`void
Revision(std::vector<uint64_t>&) const`: what its recording depends on beyond its bindings), and every pass replayable
(`PreparedPass::IsReplayable`: no submission, completion or abandonment callback, no external signal, no dependency lifecycle
effect). Pipelines and other objects a recording binds are retained by the recording itself (`RecordingRetention`, on
`OwnedRecordingList`).

-   **Live epochs record every run.** A live ticket's recording is used once and nothing keeps it. A ticket that
    `beforeSubmit` made stale is prepared again, for the same run lease. A live epoch takes no revision recording.
-   **Published recordings** are a revision-driven epoch's (below): recorded once, submitted many times.
-   **What a replayable recording leaves out.** Profiler GPU zones and GPU statistics queries the statistics service owns.
    With `GraphHost::SetRecordingTimestamps` a replayable list owns its timestamp queries instead
    (`IStatisticsService::CreateOwnedTimestamps`), read when its run retires.
-   **Statistics.** `AsyncStats::recorded`.

## Revision-driven epochs

`SetAsyncEpochs(epochs, revisionEpochs)` makes the listed epochs revision-driven. Their recordings are made ahead, for a
revision the host's scene supplies (CS's DCLF scene revision), and nothing about them is recorded at submission.

-   **Published copies.** A list cannot be pending twice, so a recording is published as `framesInFlight` copies
    (`EpochRecording::copies`). Copy i is recorded for its copy's lease (`RunRef::Copy(lease, i)`), whose LatchBlock regions
    are pinned: every submission of the copy reads the same place. The epoch's copies' leases are made at its first use and
    kept for the host's life.
-   **Taking a copy.** A revision-driven ticket takes an idle copy, else the least recently submitted copy whose last run is
    done or behind a seal, which it waits for on the host thread. When every copy holds an unsealed run that is not done,
    the epoch's next ticket waits for the next seal or a completion. Meanwhile `SubmitEpoch` throws `EpochCopiesBusyError`
    (an `EpochFailedError`): nothing was submitted, the epoch has not failed, and the caller skips that run. One warning
    is logged per epoch.
-   **`RequestEpochRecording(epoch, hostData, done)`.** Lock-free, from any thread, safe across a build. The host thread
    records each copy for `hostData` once that copy's last run is done (never waiting for it), and calls `done` once with
    the recording, the error that stopped it, or `EpochRecordingDropped` when a build stopped the host thread first. A
    request for a live epoch completes with an error.
-   **Candidates are recordings.** A revision-driven ticket prepares nothing from live state. The revision recordings its
    caller still holds are its candidates (their copy for the ticket's run), and `SubmitEpoch(epoch, beforeSubmit, owner,
    recording)` binds one, or the one `beforeSubmit` chose (`UseEpochRecording`, checked with `CanUseEpochRecording`).
-   **Immutable inputs.** A recording for a revision resolves the graph's resolvers through it (`ResolverCaptureContext` of
    `IHostExecutionData`), and its frozen bindings hold the backings and descriptors it was recorded with. A buffer that
    grows does so by versions (`VersionedBuffer`), never in place.
-   **Statistics.** `AsyncStats::revisionCopiesRecorded`, `revisionRecordings`, `revisionSubmissions`, `copyWaits` and
    `copiesBusy`.

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
    graphics timeline, so retirement, run leases and later tickets work as for a submitted epoch.
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
-   **Passes recorded at execution.** A pass declared with `ExternalPassDesc::RecordAtExecution()` is in a batch of such
    passes alone, which must be in the stream. The ticket keeps their invocations and barrier plan, and the host's
    execution records them there: for work only known then, such as a CPU cull's draws.
    -   An extension can tell whether its graph's epoch is in the stream while it gathers its passes
        (`RenderGraph::PassEpochInHostStream`).
    -   **Skipped runs.** Just before such a pass is recorded, its recorder asks it whether it has nothing to do this run
        (`PreparedPass::SkipsAtExecution`, a typed pass's static `SkipsAtExecution`). It decides from what the run's
        commit wrote. A pass that skips records only the barriers around it, with no debug label, timestamps or
        `Record`. CS's image-space anchor registers some 70 gated passes, of which a run opens a few: the rest cost a
        barrier each instead of a recording on the host thread.
-   **Retirement.** No ORG queue sees the host's stream. So the runtime's deletion and descriptor-release frontier also
    covers every value given out on the host's shared timelines (`Runtime::RegisterAssignedTimeline`). An object released
    while an in-stream list that names it is still waiting in the host's stream outlives that list.
    -   The statistics' shared query pool and its readback buffers are replaced when a build registers a new pass. Each
        list holds the set it recorded with (`QueryRecordingContext::sharedQueries`), and a replaced set is deleted only
        when the last such list is released. This also covers lists recorded ahead, which the frontier cannot cover.
    -   A list whose set was replaced while it recorded writes and resolves no more queries. Its resolve would otherwise
        wait on the GPU for queries the new pool never wrote.
    -   Test: `TestAssignedTimelineRetirement` and `TestStatisticsPoolRetirement` (`tests/RuntimeTests.cpp`).
    -   A host's own work outside the graphs can use ORG objects too: draws in its stream that read a graph's buffers.
        `Runtime::NoteExternalPoint` names a value the stream signals after that work, and the frontier covers the latest one
        per timeline. `GraphHost::BeginExternalRun` notes its value there as well as with the descriptor releases. The
        stream must signal it, or deletions wait forever.
    -   A submission takes the queues' values, submits, then publishes them. A frontier taken in between misses them. So a
        deletion batch is not sealed while an `org::Runtime::AssignmentScope` is open (`SubmitTicket`, `ExecuteFrame`).
    -   No driver free on the host's thread. On Windows, freeing a GPU virtual address waits for the GPU. The GPU may wait for
        a signal of work the host's thread prepares, so a free there can wait forever (Community Shaders deadlocked this way
        on the RX 6600). `PrepareTicket` calls `Runtime::RetireDetached`, and a submission's retired objects go to
        `Runtime::DestroyDetached`. Both hand the objects to the runtime's deleter thread. A free there slower than 100 ms is
        logged, and one still running after a second is logged at the next hand-off.
    -   A retired object whose free takes over 100 ms is logged on its own (`DeletionManager::Retired::DestroyTimed`): its
        name and size, the `MarkForDelete`, seal and `Retire` callers (`std::source_location`), the frontier it was sealed with
        against each timeline's completed value, and every timeline now. A sealed point not reached is a frontier gap; all
        reached means the free waited for other GPU work or for the kernel.
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

`org::LatchBlock` is a view into the frame arena with one region per run (above). A recording refers only to its run's region
(`RecordingContext::Run()`), and the render thread writes the values into the region before it submits (`CurrentRun()`). A
pass whose recording would otherwise bake a per-frame value (a count, a matrix, a dispatch size) reads it from its latch.

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
-   **Lifetime.** A batch is held until the run whose submission recorded it has retired.

## Command lists and allocators

`CommandListPool` hands out command lists. Many lists share one command allocator, called a shard. On Vulkan each
allocator is a `VkCommandPool`, and each pool costs driver memory once one of its buffers is recorded: about 2 MB of VRAM
on the AMD driver. One allocator per list made thousands of pools. Shared allocators keep the count near the number of
lists that record at the same time.

Vulkan requires external synchronization of a pool for every reset, begin, recorded command, end and free of any buffer
in it. The pool follows these rules:

-   **One claim per shard.** A thread records into a list only while the list's pair holds its shard's claim.
-   **Begin claims.** `CommandListPool::Begin` resets the list and takes the claim. `Request` begins the list it returns.
    `AcquireBatch` returns lists that are not begun. The recorder begins each one on its own thread
    (`RecordPreparedRhiExecutionBatch`), unless the acquirer began it already to record the Tracy frame mark.
-   **The claim travels.** A list begun on one thread may be recorded on another, when the hand-over orders the two. The
    host's thread begins a run lease's upload lists in `PrepareTicket`, and the render thread records them in `SubmitTicket`.
-   **Closing releases.** `CommandListPair::EndRecording` (`FrameCommandAllocation::Close`) gives up the claim once the
    list is closed. Recycling, discarding or destroying a pair releases it too.
-   **Claims never wait.** Begin takes the list's own shard when it is free. Otherwise it trades the list for an idle one
    of a free shard, or makes a list in a free or new shard.
-   **Frees happen under the claim.** A list dropped without its claim goes to its shard's orphans. The shard's next
    claimant frees it.
-   **No whole-allocator reset.** Each list is reset on its own when it is begun. Published recordings and lists still on the
    GPU stay valid while other lists of their shard are reset. The background worker frees idle lists over the targets,
    then trims the shards nobody holds (`CommandAllocator::Trim`).
-   **Submission needs no claim.** Neither does executing a closed secondary from another list (an external epoch's
    secondaries in the host's command buffer, on DXVK's thread).

The pool keeps idle lists for one generation's high water plus two. Lists still in flight raise the capacity when an
acquisition runs short. D3D12 allocators cannot share lists (`CommandAllocator::SupportsSharedLists`): there each list
keeps a shard of its own, and Begin resets the whole allocator.

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
    -   `leases` and `leases open` (an epoch that runs many times a frame, each run with a lease of its own, and the frame
        arena's regions), and `leases-wide` (260 runs a frame, so lease indices pass 255, with replayable recordings).
