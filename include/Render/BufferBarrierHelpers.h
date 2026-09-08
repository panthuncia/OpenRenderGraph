#pragma once
#include <rhi.h>

namespace org {

inline bool NeedsWholeBufferBarrier(rhi::ResourceAccessType beforeAccess,
    rhi::ResourceAccessType afterAccess, rhi::ResourceSyncState beforeSync,
    rhi::ResourceSyncState afterSync) noexcept {
    if (beforeAccess != afterAccess || beforeSync != afterSync) return true;
    // An otherwise identical UAV state still needs an ordering barrier between
    // accesses. Identical read/indirect/constant states are not transitions.
    return (static_cast<uint64_t>(afterAccess)
        & static_cast<uint64_t>(rhi::ResourceAccessType::UnorderedAccess)) != 0;
}

// Common encoding for a whole-buffer state transition. Admission supplies
// concrete backing and resolved states; this helper never consults trackers.
inline rhi::BufferBarrier MakeWholeBufferBarrier(rhi::ResourceHandle buffer,
    rhi::ResourceAccessType beforeAccess, rhi::ResourceAccessType afterAccess,
    rhi::ResourceSyncState beforeSync, rhi::ResourceSyncState afterSync,
    bool discard = false) noexcept {
    rhi::BufferBarrier barrier{};
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = UINT64_MAX;
    barrier.beforeAccess = beforeAccess;
    barrier.afterAccess = afterAccess;
    barrier.beforeSync = beforeSync;
    barrier.afterSync = afterSync;
    barrier.discard = discard;
    return barrier;
}

} // namespace org
