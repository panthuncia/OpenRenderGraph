#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Render/Runtime/UploadServiceAccess.h"

namespace rg::runtime {

struct BulkWriteHandle {
    uint8_t* data = nullptr;
    size_t capacity = 0;
};

enum class UploadPolicyTag : uint8_t {
    Immediate = 0,
    Coalesced = 1,
    CoalescedRetained = 2,
};

struct UploadPolicyConfig {
    UploadPolicyTag tag = UploadPolicyTag::Immediate;
};

struct BufferUploadPolicyStats {
    uint64_t stagedWrites = 0;
    uint64_t stagedBytes = 0;
    uint64_t flushedWrites = 0;
    uint64_t flushedBytes = 0;
    uint64_t mergedWrites = 0;
    uint64_t overlapEvents = 0;
    uint64_t overlapBytes = 0;
};

class BufferUploadPolicyState {
public:
    BufferUploadPolicyState() = default;

    void SetPolicy(const UploadPolicyConfig& config, size_t currentBufferSize) {
        m_config.tag = config.tag == UploadPolicyTag::Coalesced
            ? UploadPolicyTag::Immediate
            : config.tag;
        if (m_config.tag == UploadPolicyTag::CoalescedRetained && m_retainedBytes.size() != currentBufferSize) {
            m_retainedBytes.resize(currentBufferSize, 0u);
        } else if (m_config.tag != UploadPolicyTag::CoalescedRetained) {
            m_retainedBytes.clear();
            m_retainedDirtyRanges.clear();
            m_retainedDirtyRangesSorted = true;
            m_retainedStagedWrites = 0;
            m_retainedStagedBytes = 0;
            m_retainedOverlapEvents = 0;
            m_retainedOverlapBytes = 0;
        }
    }

    UploadPolicyConfig GetPolicy() const {
        return m_config;
    }

    bool IsImmediate() const {
        return m_config.tag == UploadPolicyTag::Immediate;
    }

    void OnBufferResized(size_t newSize) {
        if (m_config.tag == UploadPolicyTag::CoalescedRetained) {
            m_retainedBytes.resize(newSize, 0u);
        }
        // Preserve pending dirty ranges across grows so writes staged before a
        // resize are not dropped prior to FlushToUploadService().
    }

    void BeginFrame() {
        // Intentionally do not clear staged writes here.
        // Writes may be staged before the first frame BeginFrame() call
        // (for example during scene/resource initialization), and clearing
        // them would drop required uploads before they are ever flushed.
        // Staged data is consumed/cleared in FlushToUploadService().
    }

    // Pre-sizes the retained CPU mirror so callers can memcpy into non-overlapping
    // regions from multiple threads without synchronization.  Must be called
    // single-threaded before the parallel writes begin.
    BulkWriteHandle PrepareBulkWrite(size_t currentBufferSize) {
        if (m_config.tag != UploadPolicyTag::CoalescedRetained) {
            throw std::runtime_error("Bulk writes require a retained buffer upload policy");
        }
        if (m_retainedBytes.size() < currentBufferSize) {
            m_retainedBytes.resize(currentBufferSize, 0u);
        }
        return { m_retainedBytes.data(), m_retainedBytes.size() };
    }

    // Registers a dirty range after parallel writes are complete.
    // Must be called single-threaded.
    void CommitBulkRegion(size_t offset, size_t size) {
        if (m_config.tag != UploadPolicyTag::CoalescedRetained) {
            return;
        }
        if (size == 0) {
            return;
        }
        AddOrMergeDirtyRange(offset, offset + size, nullptr, 0);
    }

#if BUILD_TYPE == BUILD_TYPE_DEBUG
    bool StageWrite(const void* data, size_t size, size_t offset, size_t currentBufferSize, const char* file, int line) {
#else
    bool StageWrite(const void* data, size_t size, size_t offset, size_t currentBufferSize) {
        const char* file = nullptr;
        int line = 0;
#endif
        if (!data || size == 0) {
            return true;
        }

        if (m_config.tag == UploadPolicyTag::Immediate) {
            return false;
        }

        if (offset + size > currentBufferSize) {
            throw std::runtime_error("Upload policy write is out of bounds for target buffer");
        }

        if (m_retainedBytes.size() != currentBufferSize) {
            m_retainedBytes.resize(currentBufferSize, 0u);
        }

        std::memcpy(m_retainedBytes.data() + static_cast<std::ptrdiff_t>(offset), data, size);
        AddOrMergeDirtyRange(offset, offset + size, file, line);
        return true;
    }

    void FlushToUploadService(UploadTarget target) {
        if (m_config.tag != UploadPolicyTag::CoalescedRetained) {
            m_lastFlushStats = {};
            return;
        }

        BufferUploadPolicyStats stats{};
        auto* uploadService = GetActiveUploadService();
        if (!uploadService) {
            throw std::runtime_error("Upload service is not active while flushing upload policies");
        }

        stats.stagedWrites = m_retainedStagedWrites;
        stats.stagedBytes = m_retainedStagedBytes;
        stats.overlapEvents = m_retainedOverlapEvents;
        stats.overlapBytes = m_retainedOverlapBytes;

        auto mergedDirty = CoalesceDirtyRanges(std::move(m_retainedDirtyRanges), m_retainedDirtyRangesSorted);
        for (const auto& range : mergedDirty) {
            const size_t uploadSize = range.end - range.begin;
            if (uploadSize == 0) {
                continue;
            }

#if BUILD_TYPE == BUILD_TYPE_DEBUG
            uploadService->UploadData(
                m_retainedBytes.data() + static_cast<std::ptrdiff_t>(range.begin),
                uploadSize,
                target,
                range.begin,
                range.file,
                range.line);
#else
            uploadService->UploadData(
                m_retainedBytes.data() + static_cast<std::ptrdiff_t>(range.begin),
                uploadSize,
                target,
                range.begin);
#endif
            ++stats.flushedWrites;
            stats.flushedBytes += static_cast<uint64_t>(uploadSize);
        }

        stats.mergedWrites = stats.stagedWrites > stats.flushedWrites ? (stats.stagedWrites - stats.flushedWrites) : 0;
        m_retainedDirtyRanges.clear();
        m_retainedDirtyRangesSorted = true;
        m_retainedStagedWrites = 0;
        m_retainedStagedBytes = 0;
        m_retainedOverlapEvents = 0;
        m_retainedOverlapBytes = 0;
        m_lastFlushStats = stats;
    }

    BufferUploadPolicyStats GetLastFlushStats() const {
        return m_lastFlushStats;
    }

    bool HasPendingWork() const {
        if (m_config.tag != UploadPolicyTag::CoalescedRetained) {
            return false;
        }
        return !m_retainedDirtyRanges.empty();
    }

    bool RetainExternalWrite(const void* data, size_t size, size_t offset, size_t currentBufferSize) {
        if (m_config.tag != UploadPolicyTag::CoalescedRetained) {
            return false;
        }
        if (!data || size == 0) {
            return true;
        }

        if (offset + size > currentBufferSize) {
            throw std::runtime_error("External retained upload write is out of bounds for target buffer");
        }

        if (m_retainedBytes.size() != currentBufferSize) {
            m_retainedBytes.resize(currentBufferSize, 0u);
        }

        std::memcpy(m_retainedBytes.data() + static_cast<std::ptrdiff_t>(offset), data, size);
        return true;
    }

private:
    struct DirtyRange {
        size_t begin = 0;
        size_t end = 0;
        const char* file = nullptr;
        int line = 0;
    };

    static std::vector<DirtyRange> CoalesceDirtyRanges(std::vector<DirtyRange> ranges, bool alreadySorted) {
        if (ranges.empty()) {
            return ranges;
        }

        if (!alreadySorted) {
            std::sort(ranges.begin(), ranges.end(), [](const DirtyRange& lhs, const DirtyRange& rhs) {
                return lhs.begin < rhs.begin;
            });
        }

        std::vector<DirtyRange> merged;
        merged.reserve(ranges.size());
        merged.push_back(ranges.front());

        for (size_t i = 1; i < ranges.size(); ++i) {
            auto& tail = merged.back();
            const auto& current = ranges[i];
            if (current.begin <= tail.end) {
                tail.end = std::max(tail.end, current.end);
                tail.file = current.file;
                tail.line = current.line;
            }
            else {
                merged.push_back(current);
            }
        }
        return merged;
    }

    void AddOrMergeDirtyRange(size_t begin, size_t end, const char* file, int line) {
        DirtyRange incoming{ begin, end, file, line };
        ++m_retainedStagedWrites;
        m_retainedStagedBytes += static_cast<uint64_t>(end - begin);

        if (m_retainedDirtyRanges.empty()) {
            m_retainedDirtyRanges.push_back(incoming);
            return;
        }

        auto& tail = m_retainedDirtyRanges.back();
        if (incoming.begin <= tail.end && incoming.end >= tail.begin) {
            const size_t overlapBegin = std::max(incoming.begin, tail.begin);
            const size_t overlapEnd = std::min(incoming.end, tail.end);
            if (overlapBegin < overlapEnd) {
                ++m_retainedOverlapEvents;
                m_retainedOverlapBytes += static_cast<uint64_t>(overlapEnd - overlapBegin);
            }

            tail.begin = std::min(tail.begin, incoming.begin);
            tail.end = std::max(tail.end, incoming.end);
            tail.file = file;
            tail.line = line;
            return;
        }

        if (incoming.begin >= tail.end) {
            m_retainedDirtyRanges.push_back(incoming);
            return;
        }

        m_retainedDirtyRanges.push_back(incoming);
        m_retainedDirtyRangesSorted = false;
    }

    UploadPolicyConfig m_config{};
    std::vector<uint8_t> m_retainedBytes;
    std::vector<DirtyRange> m_retainedDirtyRanges;
    bool m_retainedDirtyRangesSorted = true;
    uint64_t m_retainedStagedWrites = 0;
    uint64_t m_retainedStagedBytes = 0;
    uint64_t m_retainedOverlapEvents = 0;
    uint64_t m_retainedOverlapBytes = 0;
    BufferUploadPolicyStats m_lastFlushStats{};
};

}
