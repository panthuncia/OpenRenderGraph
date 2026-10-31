#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace org {

// A persistent executable's schedule, explained. It lists:
// - its batches and passes per queue,
// - each pass's accesses, and the state changes (barriers) before it,
// - every wait a batch makes, with its cause.
// It is plain data. The persistent execution makes it once per executable, and once per epoch's executable, while
// RenderGraph::SetScheduleTraceEnabled is on. Get it with RenderGraph::GetPersistentScheduleSnapshot. It is read by
// the schedule inspector (DebugUI/ScheduleInspector.h), its JSON form (ScheduleTraceToJson) and the persistent
// compile dump.
struct ScheduleTrace {
    struct Access {
        uint32_t resource = 0;  // index into resources
        bool write = false;
    };
    // A state change the compiler planned before a pass: a barrier (or a cross-queue release/acquire) at execution.
    struct StateChange {
        uint32_t resource = 0;
        uint32_t previousBatch = UINT32_MAX;  // batch of the previous use, or UINT32_MAX for the execution's entry state
        uint32_t previousPass = UINT32_MAX;   // index into passes, or UINT32_MAX for the execution's entry state
        uint64_t beforeAccess = 0, beforeLayout = 0, beforeSync = 0;
        uint64_t afterAccess = 0, afterLayout = 0, afterSync = 0;
        bool beforeWrite = false, afterWrite = false;
        uint32_t firstMip = 0, mipCount = 0, firstSlice = 0, sliceCount = 0;  // counts of 0 mean the whole resource
    };
    struct Pass {
        std::string name;
        uint32_t preparedPassIndex = UINT32_MAX;
        uint32_t batch = 0, position = 0;
        uint32_t queue = 0, preferredQueue = 0;
        bool forceBatchIsolation = false;
        std::vector<Access> accesses;
        std::vector<StateChange> stateChanges;  // before this pass, in plan order
    };
    // Why a consumer batch waits for a producer batch on another queue: each scheduling edge between them.
    struct WaitCause {
        enum class Kind : uint8_t {
            Resource,   // a hazard on a resource both passes use (see hazard)
            Explicit,   // an authored ordering edge
            Placement,  // an alias placement edge (two resources sharing memory)
            Unknown,    // an edge with no resource in common (none of the above)
        };
        enum class Hazard : uint8_t {
            // Not a Resource cause, or a Resource cause with no data hazard. The second case covers two reads that the
            // compiler still orders: reads of a texture, whose layout is state that the plan transitions, or reads from two
            // backends, ordered for API ownership.
            None,
            ReadAfterWrite,
            WriteAfterRead,
            WriteAfterWrite,
        };
        Kind kind = Kind::Unknown;
        uint32_t fromPass = 0, toPass = 0;  // indices into passes
        uint32_t resource = UINT32_MAX;     // for a Resource cause, the index into resources
        Hazard hazard = Hazard::None;
    };
    // A wait inside the execution. consumerBatch's queue waits for producerBatch's signal. For each consumer and producer
    // queue, only the latest producer batch is waited on. causes covers every edge from that queue's batches up to it.
    struct Wait {
        uint32_t consumerBatch = 0, producerBatch = 0;
        std::vector<WaitCause> causes;
    };
    struct Batch {
        uint32_t queue = 0;
        std::vector<uint32_t> passes;  // into passes, in execution order
    };
    struct Resource {
        std::string name;
        uint64_t id = 0;
        bool hasLayout = false;  // true for a texture (layout transitions), false for a buffer
    };

    std::string label;           // the executable's: "frame" or "epoch <n>"
    uint64_t structuralRevision = 0;
    uint32_t epoch = UINT32_MAX;  // UINT32_MAX for the whole-frame executable
    std::vector<std::string> queueNames;  // indexed by queue slot
    std::vector<Batch> batches;
    std::vector<Pass> passes;
    std::vector<Resource> resources;
    std::vector<Wait> waits;
    std::string stateFallbackReason;  // why the state plan is incomplete, if it is
};

// The waits that one execution of an executable made at submission, beyond its batches' own waits
// (ScheduleTrace::waits). They are absolute timeline points, as the last frame submitted them.
struct ScheduleSubmissionWaits {
    struct Point {
        enum class Source : uint8_t {
            Incoming,     // from admission: a cross-frame hazard on a resource (or an alias) that an earlier execution used
            Uploads,      // the graphics-queue uploads ahead of the ticket, for other queues' batches that use what they write
            FrameValues,  // the frame values' timeline (SetFrameWaitTimeline)
        };
        Source source = Source::Incoming;
        // The timeline identity from admission, or RenderGraph::kFrameWaitTimelineIdentity. A timeline identity
        // (Runtime::TimelineIdentity) is a graph queue's timeline or a foreign timeline that a pass waits on.
        // For Uploads, this is the uploads' queue slot + 1.
        uint64_t timeline = 0, value = 0;
    };
    struct Batch {
        std::vector<Point> points;
        bool externalEntryBarrier = false, externalExitBarrier = false;
        // Queue-scoped uploads (RenderGraph::RecordPendingUploads). queueUploadCopies counts the copies recorded on this
        // batch's queue and submitted just ahead of it. queueUploadWaits are that submission's own waits: Incoming points
        // for earlier executions' use of the copy destinations.
        uint64_t queueUploadCopies = 0;
        std::vector<Point> queueUploadWaits;
    };
    // The executable's trace label ("frame", "epoch <n>"), or "pre"/"tail" for the per-frame dynamic segments (no trace).
    std::string label;
    uint64_t frame = 0;  // the ticket's host frame, or the graph's frame number for a synchronous execution
    std::vector<Batch> batches;  // indexed by the executable's batch
};

// The schedules a persistent graph executes now:
// - the whole-frame executable,
// - each epoch's executable,
// - the submission waits of the latest execution of each label, by label.
// It is published once per round of executions, so all of a snapshot's entries are from one round. It is cleared
// when the selected executable changes, because batch indices refer to the traced executables.
struct PersistentScheduleSnapshot {
    std::vector<std::shared_ptr<const ScheduleTrace>> executables;
    std::vector<ScheduleSubmissionWaits> lastSubmission;
};

std::string ScheduleTraceToJson(const PersistentScheduleSnapshot& snapshot);
std::string ScheduleTraceToText(const ScheduleTrace& trace);

}  // namespace org
