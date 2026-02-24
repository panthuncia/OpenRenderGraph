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
        m_config = config;
        if (m_config.tag == UploadPolicyTag::Coalesced || m_config.tag == UploadPolicyTag::CoalescedRetained) {
            if (m_coalescedScratchBytes.size() != currentBufferSize) {
                m_coalescedScratchBytes.resize(currentBufferSize, 0u);
            }
        }
        else {
            m_coalescedScratchBytes.clear();
        }

        if (m_config.tag == UploadPolicyTag::CoalescedRetained) {
            if (m_retainedBytes.size() != currentBufferSize) {
                m_retainedBytes.resize(currentBufferSize, 0u);
            }
        }
        else {
            m_retainedBytes.clear();
            m_retainedDirtyRanges.clear();
        }
        m_coalescedDirtyRanges.clear();
        m_coalescedStagedWrites = 0;
        m_coalescedStagedBytes = 0;
    }

    UploadPolicyConfig GetPolicy() const {
        return m_config;
    }

    bool IsImmediate() const {
        return m_config.tag == UploadPolicyTag::Immediate;
    }

    void OnBufferResized(size_t newSize) {
        if (m_config.tag == UploadPolicyTag::Coalesced || m_config.tag == UploadPolicyTag::CoalescedRetained) {
            m_coalescedScratchBytes.resize(newSize, 0u);
        }

        if (m_config.tag == UploadPolicyTag::CoalescedRetained) {
            m_retainedBytes.resize(newSize, 0u);
            // Preserve pending retained dirty ranges across grows so writes staged
            // before a resize are not dropped prior to FlushToUploadService().
        }
    }

    void BeginFrame() {
        // Intentionally do not clear staged writes here.
        // Writes may be staged before the first frame BeginFrame() call
        // (for example during scene/resource initialization), and clearing
        // them would drop required uploads before they are ever flushed.
        // Staged data is consumed/cleared in FlushToUploadService().
    }

#if BUILD_TYPE == BUILD_TYPE_DEBUG
    bool StageWrite(const void* data, size_t size, size_t offset, size_t currentBufferSize, const char* file, int line) {
#else
    bool StageWrite(const void* data, size_t size, size_t offset, size_t currentBufferSize) {
        const char* file = nullptr;
        int line = 0;
#endif
        if (m_config.tag == UploadPolicyTag::Immediate) {
            return false;
        }

        if (!data || size == 0) {
            return true;
        }

        if (offset + size > currentBufferSize) {
            throw std::runtime_error("Upload policy write is out of bounds for target buffer");
        }

        if (m_config.tag == UploadPolicyTag::Coalesced) {
            if (m_coalescedScratchBytes.size() != currentBufferSize) {
                m_coalescedScratchBytes.resize(currentBufferSize, 0u);
            }

            std::memcpy(m_coalescedScratchBytes.data() + static_cast<std::ptrdiff_t>(offset), data, size);
            m_coalescedDirtyRanges.push_back(DirtyRange{ offset, offset + size, file, line });
            ++m_coalescedStagedWrites;
            m_coalescedStagedBytes += static_cast<uint64_t>(size);
            return true;
        }

        if (m_retainedBytes.size() != currentBufferSize) {
            m_retainedBytes.resize(currentBufferSize, 0u);
        }

        std::memcpy(m_retainedBytes.data() + static_cast<std::ptrdiff_t>(offset), data, size);
        AddOrMergeDirtyRange(offset, offset + size, file, line);
        return true;
    }

    void FlushToUploadService(UploadTarget target) {
        BufferUploadPolicyStats stats{};
        if (m_config.tag == UploadPolicyTag::Immediate) {
            m_lastFlushStats = stats;
            return;
        }

        auto* uploadService = GetActiveUploadService();
        if (!uploadService) {
            throw std::runtime_error("Upload service is not active while flushing upload policies");
        }

        if (m_config.tag == UploadPolicyTag::Coalesced) {
            stats.stagedWrites = m_coalescedStagedWrites;
            stats.stagedBytes = m_coalescedStagedBytes;

            auto coalesced = CoalesceDirtyRanges(std::move(m_coalescedDirtyRanges));
            for (const auto& range : coalesced) {
                const size_t uploadSize = range.end - range.begin;
                if (uploadSize == 0) {
                    continue;
                }

#if BUILD_TYPE == BUILD_TYPE_DEBUG
                uploadService->UploadData(
                    m_coalescedScratchBytes.data() + static_cast<std::ptrdiff_t>(range.begin),
                    uploadSize,
                    target,
                    range.begin,
                    range.file,
                    range.line);
#else
                uploadService->UploadData(
                    m_coalescedScratchBytes.data() + static_cast<std::ptrdiff_t>(range.begin),
                    uploadSize,
                    target,
                    range.begin);
#endif
                ++stats.flushedWrites;
                stats.flushedBytes += static_cast<uint64_t>(uploadSize);
            }

            stats.mergedWrites = stats.stagedWrites > stats.flushedWrites ? (stats.stagedWrites - stats.flushedWrites) : 0;
            m_coalescedDirtyRanges.clear();
            m_coalescedStagedWrites = 0;
            m_coalescedStagedBytes = 0;
            m_lastFlushStats = stats;
            return;
        }

        stats.stagedWrites = static_cast<uint64_t>(m_retainedDirtyRanges.size());
        auto mergedDirty = CoalesceDirtyRanges(m_retainedDirtyRanges);
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
            stats.stagedBytes += static_cast<uint64_t>(uploadSize);
            ++stats.flushedWrites;
            stats.flushedBytes += static_cast<uint64_t>(uploadSize);
        }

        stats.mergedWrites = stats.stagedWrites > stats.flushedWrites ? (stats.stagedWrites - stats.flushedWrites) : 0;
        m_retainedDirtyRanges.clear();
        m_lastFlushStats = stats;
    }

    BufferUploadPolicyStats GetLastFlushStats() const {
        return m_lastFlushStats;
    }

private:
    struct DirtyRange {
        size_t begin = 0;
        size_t end = 0;
        const char* file = nullptr;
        int line = 0;
    };

    static std::vector<DirtyRange> CoalesceDirtyRanges(std::vector<DirtyRange> ranges) {
        if (ranges.empty()) {
            return ranges;
        }

        std::sort(ranges.begin(), ranges.end(), [](const DirtyRange& lhs, const DirtyRange& rhs) {
            return lhs.begin < rhs.begin;
        });

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
        std::vector<DirtyRange> updated;
        updated.reserve(m_retainedDirtyRanges.size() + 1);

        for (const auto& range : m_retainedDirtyRanges) {
            if (range.end < incoming.begin || range.begin > incoming.end) {
                updated.push_back(range);
                continue;
            }

            incoming.begin = std::min(incoming.begin, range.begin);
            incoming.end = std::max(incoming.end, range.end);
            incoming.file = file;
            incoming.line = line;
        }

        updated.push_back(incoming);
        m_retainedDirtyRanges = std::move(updated);
    }

    UploadPolicyConfig m_config{};
    std::vector<uint8_t> m_coalescedScratchBytes;
    std::vector<DirtyRange> m_coalescedDirtyRanges;
    uint64_t m_coalescedStagedWrites = 0;
    uint64_t m_coalescedStagedBytes = 0;
    std::vector<uint8_t> m_retainedBytes;
    std::vector<DirtyRange> m_retainedDirtyRanges;
    BufferUploadPolicyStats m_lastFlushStats{};
};

}
