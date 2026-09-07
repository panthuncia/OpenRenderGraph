#pragma once
#include <rhi.h>

namespace org {

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
