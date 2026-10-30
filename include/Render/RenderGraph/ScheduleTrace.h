#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace org {

// A persistent executable's schedule, explained: its batches and passes per queue, each pass's accesses and the state changes
// (barriers) before it, and every wait a batch makes with its cause. Plain data, made once per executable (and per epoch's
// executable) by the persistent execution while RenderGraph::SetScheduleTraceEnabled is on (RenderGraph::GetPersistentScheduleSnapshot),
// read by the schedule inspector (DebugUI/ScheduleInspector.h), its JSON (ScheduleTraceToJson) and the persistent compile dump.
struct ScheduleTrace {
    struct Access {
        uint32_t resource = 0;  // into resources
        bool write = false;
    };
    // A state change the compiler planned before a pass: a barrier (or a cross-queue release/acquire) at execution.
    struct StateChange {
        uint32_t resource = 0;
        uint32_t previousBatch = UINT32_MAX;  // the batch of the previous use; UINT32_MAX: the execution's entry state
        uint32_t previousPass = UINT32_MAX;   // into passes; UINT32_MAX: the execution's entry state
        uint64_t beforeAccess = 0, beforeLayout = 0, beforeSync = 0;
        uint64_t afterAccess = 0, afterLayout = 0, afterSync = 0;
        bool beforeWrite = false, afterWrite = false;
        uint32_t firstMip = 0, mipCount = 0, firstSlice = 0, sliceCount = 0;  // 0 counts: the whole resource
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
            // Not a Resource cause; or one with no data hazard: two reads the compiler still orders (a texture's, whose layout
            // is state the plan transitions, or reads from two backends, ordered for API ownership).
            None,
            ReadAfterWrite,
            WriteAfterRead,
            WriteAfterWrite,
        };
        Kind kind = Kind::Unknown;
        uint32_t fromPass = 0, toPass = 0;  // into passes
        uint32_t resource = UINT32_MAX;     // Resource: into resources
        Hazard hazard = Hazard::None;
    };
    // A wait inside the execution: consumerBatch's queue waits for producerBatch's signal. Only the latest producer batch per
    // (consumer, producer queue) is waited on; causes covers every edge from that queue's batches up to it.
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
        bool hasLayout = false;  // a texture (layout transitions), else a buffer
    };

    std::string label;           // the executable's: "frame" or "epoch <n>"
    uint64_t structuralRevision = 0;
    uint32_t epoch = UINT32_MAX;  // UINT32_MAX: the whole-frame executable
    std::vector<std::string> queueNames;  // by queue slot
    std::vector<Batch> batches;
    std::vector<Pass> passes;
    std::vector<Resource> resources;
    std::vector<Wait> waits;
    std::string stateFallbackReason;  // the state plan's, when incomplete
};

// The waits one execution of an executable made at submission, beyond its batches' own (ScheduleTrace::waits): absolute
// timeline points, as the last frame submitted them.
struct ScheduleSubmissionWaits {
    struct Point {
        enum class Source : uint8_t {
            Incoming,     // admission: a cross-frame hazard on a resource an earlier execution used (or an alias)
            Uploads,      // the uploads ahead of the ticket on the graphics queue (another queue's batches that use what they write)
            FrameValues,  // the frame values' timeline (SetFrameWaitTimeline)
        };
        Source source = Source::Incoming;
        // The admission's timeline identity (Runtime::TimelineIdentity: a graph queue's or a foreign timeline a pass waits on) or
        // RenderGraph::kFrameWaitTimelineIdentity. Uploads: the uploads' queue slot + 1.
        uint64_t timeline = 0, value = 0;
    };
    struct Batch {
        std::vector<Point> points;
        bool externalEntryBarrier = false, externalExitBarrier = false;
        // Queue-scoped uploads (RenderGraph::RecordPendingUploads): copies recorded on this batch's queue and submitted just
        // ahead of it, and that submission's own waits (earlier executions' use of their destinations; Incoming points).
        uint64_t queueUploadCopies = 0;
        std::vector<Point> queueUploadWaits;
    };
    // The executable's trace label ("frame", "epoch <n>"), or "pre"/"tail" for the per-frame dynamic segments (no trace).
    std::string label;
    uint64_t frame = 0;  // the ticket's host frame, or the graph's frame number for a synchronous execution
    std::vector<Batch> batches;  // by batch of the executable
};

// The schedules a persistent graph executes now: the whole-frame executable, each epoch's, and the submission waits of the latest
// execution of each label, by label (published once per round of executions, so a snapshot's entries are of one round). Cleared
// when the selected executable changes: batch indices are the traced executables'.
struct PersistentScheduleSnapshot {
    std::vector<std::shared_ptr<const ScheduleTrace>> executables;
    std::vector<ScheduleSubmissionWaits> lastSubmission;
};

std::string ScheduleTraceToJson(const PersistentScheduleSnapshot& snapshot);
std::string ScheduleTraceToText(const ScheduleTrace& trace);

}  // namespace org
