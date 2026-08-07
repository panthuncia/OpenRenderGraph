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

    struct MemoryResourceRow {
        std::string name;
        std::string type;
        uint64_t bytes = 0;
        uint64_t uid = 0;
    };

    struct MemorySnapshot {
        std::vector<MemoryCategorySlice> categories;
        std::vector<MemoryResourceRow> resources;
        uint64_t totalBytes = 0;
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
