#pragma once

#include <cstdint>
#include <string>
#include <vector>


namespace org {

namespace ui {
    struct MemoryCategorySlice {
        std::string label;
        uint64_t bytes = 0;
    };

    enum MemoryRowFlags : uint32_t {
        MemoryRowFlag_None = 0,
        // Placed in an aliasing pool, whose own row carries the bytes.
        MemoryRowFlag_Aliased = 1u << 0,
        // Shown, but its bytes are counted by another row (aliased placements, shared memory): left out of totals.
        MemoryRowFlag_NotCounted = 1u << 1,
        // Memory another allocator owns, wrapped by this one.
        MemoryRowFlag_Imported = 1u << 2,
        // No owner could be determined.
        MemoryRowFlag_Unattributed = 1u << 3,
        // The runtime's own memory rather than an application resource.
        MemoryRowFlag_Internal = 1u << 4,
    };

    struct MemoryResourceRow {
        std::string name;
        std::string type;
        uint64_t bytes = 0;
        uint64_t uid = 0;

        // Attribution; optional. '/' nests levels ("Engine/Textures/terrain").
        std::string owner;
        // The allocator or API the memory came from ("ORG", "D3D11", ...).
        std::string source;
        // The memory heap ("Device", "Host").
        std::string heap;
        // Where the resource was created, if known.
        std::string callsite;
        uint32_t flags = MemoryRowFlag_None;
    };

    // One memory heap as the driver reports it, against what the snapshot's rows account for.
    struct MemoryHeapRow {
        std::string label;
        uint64_t usageBytes = 0;
        uint64_t budgetBytes = 0;
        uint64_t attributedBytes = 0;
        bool deviceLocal = true;
    };

    enum class MemoryReconciliationKind : uint8_t {
        // A total that the rows below it (one depth deeper) break down.
        Total = 0,
        // Bytes attributed to resources.
        Attributed = 1,
        // Bytes known to exist but not attributed to a resource (allocator slack, caches, in-flight).
        Residual = 2,
        // Bytes no allocator in the process accounts for.
        Opaque = 3,
    };

    // A row of a breakdown from what the driver reports down to attributed resources.
    struct MemoryReconciliationRow {
        std::string label;
        uint64_t bytes = 0;
        MemoryReconciliationKind kind = MemoryReconciliationKind::Attributed;
        int depth = 0;
        std::string note;
    };

    struct MemorySnapshot {
        std::vector<MemoryCategorySlice> categories;
        std::vector<MemoryResourceRow> resources;
        uint64_t totalBytes = 0;

        // Optional: the heaps and the breakdown of their usage.
        std::vector<MemoryHeapRow> heaps;
        std::vector<MemoryReconciliationRow> reconciliation;
    };

    struct FrameGraphBatchRow {
        std::string label;
        uint64_t footprintBytes = 0;
        uint64_t peakLiveBytes = 0;
        uint64_t peakNaiveLiveBytes = 0;
        uint64_t aliasSavingsBytes = 0;
        bool hasEndTransitions = false;
        std::vector<std::string> passNames;
        std::vector<MemoryCategorySlice> categories;
    };

    struct FrameGraphSnapshot {
        std::vector<FrameGraphBatchRow> batches;
    };
}


} // namespace org
