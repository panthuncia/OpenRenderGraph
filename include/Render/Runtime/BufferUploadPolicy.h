#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Render/Runtime/UploadServiceAccess.h"

namespace rg::runtime {

struct BulkWriteHandle {
    uint8_t* data = nullptr;
    size_t capacity = 0;
    std::shared_ptr<void> lock;
};

enum class UploadPolicyTag : uint8_t {
    Immediate = 0,
    Coalesced = 1,
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
        (void)currentBufferSize;
        m_config = config;
        if (m_config.tag != UploadPolicyTag::Coalesced) {
            ClearPendingWork();
        }
    }

    UploadPolicyConfig GetPolicy() const {
        return m_config;
    }

    bool IsImmediate() const {
        return m_config.tag == UploadPolicyTag::Immediate;
    }

    void OnBufferResized(size_t newSize) {
        (void)newSize;
        // Preserve pending dirty ranges across grows so writes staged before a
        // resize are not dropped prior to FlushToUploadService(). Buffer-owned
        // CPU mirrors are authoritative; this policy only tracks byte ranges.
    }

    void BeginFrame() {
        // Intentionally do not clear staged writes here.
        // Writes may be staged before the first frame BeginFrame() call
        // (for example during scene/resource initialization), and clearing
        // them would drop required uploads before they are ever flushed.
        // Staged data is consumed/cleared in FlushToUploadService().
    }

    // Bulk writes are now owned by buffer-specific CPU mirrors. This helper
    // remains only to fail loudly if old retained-mirror code is reintroduced.
    BulkWriteHandle PrepareBulkWrite(size_t currentBufferSize) {
        (void)currentBufferSize;
        throw std::runtime_error("Upload policy bulk writes were removed; write to the buffer CPU mirror instead");
    }

    // Registers a dirty range after parallel writes to a buffer-owned CPU mirror.
    // Must be called single-threaded.
    void CommitBulkRegion(size_t offset, size_t size) {
        if (m_config.tag != UploadPolicyTag::Coalesced) {
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
        if (size == 0) {
            return true;
        }

        if (m_config.tag == UploadPolicyTag::Immediate) {
            return false;
        }

        if (offset + size > currentBufferSize) {
            throw std::runtime_error("Upload policy write is out of bounds for target buffer");
        }

        (void)data;
        AddOrMergeDirtyRange(offset, offset + size, file, line);
        return true;
    }

    template<class SourceBytesFn>
    void FlushToUploadService(UploadTarget target, SourceBytesFn&& sourceBytes) {
        if (m_config.tag != UploadPolicyTag::Coalesced) {
            m_lastFlushStats = {};
            return;
        }

        BufferUploadPolicyStats stats{};
        auto* uploadService = GetActiveUploadService();
        if (!uploadService) {
            throw std::runtime_error("Upload service is not active while flushing upload policies");
        }

        stats.stagedWrites = m_coalescedStagedWrites;
        stats.stagedBytes = m_coalescedStagedBytes;
        stats.overlapEvents = m_coalescedOverlapEvents;
        stats.overlapBytes = m_coalescedOverlapBytes;

        auto mergedDirty = CoalesceDirtyRanges(std::move(m_coalescedDirtyRanges), m_coalescedDirtyRangesSorted);
        for (const auto& range : mergedDirty) {
            const size_t uploadSize = range.end - range.begin;
            if (uploadSize == 0) {
                continue;
            }

            const void* source = sourceBytes(range.begin, uploadSize);
            if (!source) {
                throw std::runtime_error("Upload policy source mirror returned null for a dirty range");
            }

#if BUILD_TYPE == BUILD_TYPE_DEBUG
            uploadService->UploadData(
                source,
                uploadSize,
                target,
                range.begin,
                range.file,
                range.line);
#else
            uploadService->UploadData(
                source,
                uploadSize,
                target,
                range.begin);
#endif
            ++stats.flushedWrites;
            stats.flushedBytes += static_cast<uint64_t>(uploadSize);
        }

        stats.mergedWrites = stats.stagedWrites > stats.flushedWrites ? (stats.stagedWrites - stats.flushedWrites) : 0;
        ClearPendingWork();
        m_lastFlushStats = stats;
    }

    BufferUploadPolicyStats GetLastFlushStats() const {
        return m_lastFlushStats;
    }

    bool HasPendingWork() const {
        if (m_config.tag != UploadPolicyTag::Coalesced) {
            return false;
        }
        return !m_coalescedDirtyRanges.empty();
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
        ++m_coalescedStagedWrites;
        m_coalescedStagedBytes += static_cast<uint64_t>(end - begin);

        if (m_coalescedDirtyRanges.empty()) {
            m_coalescedDirtyRanges.push_back(incoming);
            return;
        }

        auto& tail = m_coalescedDirtyRanges.back();
        if (incoming.begin <= tail.end && incoming.end >= tail.begin) {
            const size_t overlapBegin = std::max(incoming.begin, tail.begin);
            const size_t overlapEnd = std::min(incoming.end, tail.end);
            if (overlapBegin < overlapEnd) {
                ++m_coalescedOverlapEvents;
                m_coalescedOverlapBytes += static_cast<uint64_t>(overlapEnd - overlapBegin);
            }

            tail.begin = std::min(tail.begin, incoming.begin);
            tail.end = std::max(tail.end, incoming.end);
            tail.file = file;
            tail.line = line;
            return;
        }

        if (incoming.begin >= tail.end) {
            m_coalescedDirtyRanges.push_back(incoming);
            return;
        }

        m_coalescedDirtyRanges.push_back(incoming);
        m_coalescedDirtyRangesSorted = false;
    }

    void ClearPendingWork() {
        m_coalescedDirtyRanges.clear();
        m_coalescedDirtyRangesSorted = true;
        m_coalescedStagedWrites = 0;
        m_coalescedStagedBytes = 0;
        m_coalescedOverlapEvents = 0;
        m_coalescedOverlapBytes = 0;
    }

    UploadPolicyConfig m_config{};
    std::vector<DirtyRange> m_coalescedDirtyRanges;
    bool m_coalescedDirtyRangesSorted = true;
    uint64_t m_coalescedStagedWrites = 0;
    uint64_t m_coalescedStagedBytes = 0;
    uint64_t m_coalescedOverlapEvents = 0;
    uint64_t m_coalescedOverlapBytes = 0;
    BufferUploadPolicyStats m_lastFlushStats{};
};

}
