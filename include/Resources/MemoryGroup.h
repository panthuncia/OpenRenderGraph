#pragma once

#include <cstdint>

namespace org {

// Which blocks a buffer's backing is allocated from.
// - Default: the allocator's shared blocks, with textures and long-lived buffers. The defragmenter may move it.
// - PerFrame: a block pool of its own, for buffers that are made and dropped every few frames (per-frame pools). It never shares
//   a block with a Default allocation, so its churn leaves no holes among long-lived memory. It is never a defragmentation source
//   or target.
enum class MemoryGroup : uint8_t {
    Default,
    PerFrame,
};

// The group's name for logs and memory reports.
inline const char* MemoryGroupName(MemoryGroup group) noexcept {
    switch (group) {
    case MemoryGroup::Default: return "Default";
    case MemoryGroup::PerFrame: return "PerFrame";
    }
    return "Unknown";
}

} // namespace org
