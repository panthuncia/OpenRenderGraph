#pragma once

#include <memory>

#include <rhi.h>

namespace org::runtime {

// Publication channels: how a pass declares a read of a per-frame publication.
//
// A publication is immutable data a host makes each frame into a fresh pooled backing. Passes read it through a bindless
// index the host writes into frame data (a latch block, say), so their recordings do not depend on which backing holds
// it. No pass can declare that backing: it changes every frame, after the recordings were made.
//
// A channel stands in for it. The channel is any graph resource that stays the same across frames, such as a small
// buffer. Every pass that reads the publication declares the channel. When the host makes a publication, it binds the
// publication's backing to the channel. The upload planner (RenderGraph::RecordPendingUploads' queue-scoped uploads) then
// takes each pass that declares the channel as a user of every staged copy into a bound backing:
// - the copy goes to that pass's queue, ahead of its batch;
// - or the batch waits for the graphics queue's copies.
// Without the binding, such a batch could run before the copy that fills what it reads.
//
// Thread contract: Bind and ChannelOf may run on any thread.
class PublicationChannels {
public:
    // Names `backing` as content of the channel whose backing is `channel`, until the returned binding is released.
    // Hold the binding as long as the publication it names may still be uploaded. Binding a backing again replaces its
    // channel. Releasing an older binding then leaves the newer one in place. Invalid handles bind nothing (null).
    [[nodiscard]] static std::shared_ptr<void> Bind(rhi::ResourceHandle backing, rhi::ResourceHandle channel);

    // The channel bound to `backing`, or an invalid handle for none.
    [[nodiscard]] static rhi::ResourceHandle ChannelOf(rhi::ResourceHandle backing) noexcept;

    // Whether any backing is bound: lets the planner skip the lookups when no host uses channels.
    [[nodiscard]] static bool Any() noexcept;
};

} // namespace org::runtime
