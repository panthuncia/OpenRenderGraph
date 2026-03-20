#pragma once

#include <rhi.h>
#include <resource_states.h>

#include "Resources/Resource.h"
#include "Resources/ResourceStateTracker.h"

// Resource wrapper around an externally-owned texture (e.g. a
// swapchain image).  Does NOT allocate or free the underlying GPU resource;
// the caller retains ownership.  Provides the SymbolicTracker and barrier
// generation the render-graph needs for automatic state tracking.
class ExternalTextureResource : public Resource {
public:
    ExternalTextureResource(rhi::ResourceHandle handle,
                            unsigned int width,
                            unsigned int height)
        : m_handle(handle)
        , m_width(width)
        , m_height(height)
    {
        m_hasLayout = true;
        m_mipLevels = 1;
        m_arraySize = 1;
    }

    rhi::Resource GetAPIResource() override {
        // Returns a thin Resource wrapper.  impl/vt are null (not a managed
        // resource) but the handle is valid for barrier and copy operations.
        return rhi::Resource(m_handle, /*isTexture=*/true);
    }

    rhi::BarrierBatch GetEnhancedBarrierGroup(
        RangeSpec range,
        rhi::ResourceAccessType prevAccessType,
        rhi::ResourceAccessType newAccessType,
        rhi::ResourceLayout prevLayout,
        rhi::ResourceLayout newLayout,
        rhi::ResourceSyncState prevSyncState,
        rhi::ResourceSyncState newSyncState) override
    {
        auto resolvedRange = ResolveRangeSpec(range, m_mipLevels, m_arraySize);

        m_barrier.afterAccess  = newAccessType;
        m_barrier.beforeAccess = prevAccessType;
        m_barrier.afterLayout  = newLayout;
        m_barrier.beforeLayout = prevLayout;
        m_barrier.afterSync    = newSyncState;
        m_barrier.beforeSync   = prevSyncState;
        m_barrier.discard      = false;
        m_barrier.range        = { resolvedRange.firstMip, resolvedRange.mipCount,
                                   resolvedRange.firstSlice, resolvedRange.sliceCount };
        m_barrier.texture      = m_handle;

        rhi::BarrierBatch batch{};
        batch.textures = { &m_barrier };
        return batch;
    }

    SymbolicTracker* GetStateTracker() override {
        return &m_stateTracker;
    }

    unsigned int GetWidth()  const { return m_width; }
    unsigned int GetHeight() const { return m_height; }

    void SetHandle(rhi::ResourceHandle handle) { m_handle = handle; }

    // Reset the symbolic tracker to Common state.  Must be called after any
    // out-of-graph barrier (e.g. TransitionForPresent) so the render graph
    // knows the true GPU state on the next frame.
    void ResetToCommon() {
        m_stateTracker = SymbolicTracker{};
    }

private:
    rhi::ResourceHandle   m_handle;
    unsigned int          m_width;
    unsigned int          m_height;
    rhi::TextureBarrier   m_barrier{};
    SymbolicTracker       m_stateTracker;
};
