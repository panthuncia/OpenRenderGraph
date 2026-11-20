#pragma once

#include <cstdint>
#include <memory>

#include "Resources/Buffers/Buffer.h"
#include "Resources/VersionedResource.h"

namespace org {

// One version of a VersionedBuffer. It is immutable once made.
struct BufferVersion : ResourceVersion {
    std::shared_ptr<Buffer> buffer;  // the same object as `resource`
};

namespace detail {
// Late-bound versions' backings, by the stand-in their readers declare (VersionedBuffer::CreateLateBound). The upload planner
// (RenderGraph::RecordTakenUploads' queue-scoped uploads) reads it. Any thread.
// - BindLateBacking: names `backing` as content of `standIn` until the returned token is released. Binding a backing again
//   replaces its stand-in, and releasing the older token then keeps the newer entry. Invalid handles bind nothing (null).
// - LateStandInOf: the stand-in `backing` is bound to, or an invalid handle.
// - AnyLateBacking: whether anything is bound, so the planner can skip the lookups.
// - MarkFrameWaitStandIn, IsFrameWaitStandIn: the stand-ins of late-bound buffers whose versions are ready at the frame wait
//   (LateReady::FrameWait). With scoped frame waits, only the batches that declare one wait for it.
[[nodiscard]] std::shared_ptr<void> BindLateBacking(rhi::ResourceHandle backing, rhi::ResourceHandle standIn);
[[nodiscard]] rhi::ResourceHandle LateStandInOf(rhi::ResourceHandle backing) noexcept;
[[nodiscard]] bool AnyLateBacking() noexcept;
void MarkFrameWaitStandIn(rhi::ResourceHandle standIn);
[[nodiscard]] bool IsFrameWaitStandIn(rhi::ResourceHandle standIn) noexcept;
}  // namespace detail

// A buffer whose growth makes a new version, instead of changing a backing in place (VersionedResource). Passes declare it,
// since it is a resolver. A preparation binds the version its revision names (IResourceVersions), or else the current one.
//
// Growth as graph work (the SARP/BasicRenderer model):
// 1. A producer on any thread makes the next version with Make*, which materializes it.
// 2. The producer fills it through the dedicated uploader (IUploadService::QueueTrackedStreamingUploadSegments), as a
//    WorkerOwnedDestination of ownership PendingVersion. No queue uses a version before it is adopted.
// 3. The version is ready once those copies complete.
// 4. The owner adopts it when the revision that names it is selected.
//
// Late-bound versions (CreateLateBound) are for data a host publishes after the recordings that read it, such as a fresh
// pooled backing each frame. Shaders read it through the bindless index the host writes into frame data, so no recording
// binds it. Passes declare the buffer as usual, but it always resolves to a stand-in that never changes. So a new version
// costs no binding edit and no lowering. While a version lives, the upload planner takes each batch that declares the
// buffer as a user of the version's backing: an upload into it goes to that batch's queue, or the batch waits for it.
// Barriers come from the upload lists' own barriers, so the readers must only read.
// A host that fills its versions some other way and signals the frame-wait timeline after (RenderGraph::SetFrameWaitTimeline)
// makes the buffer with LateReady::FrameWait: then the batches that declare it wait for the frame wait.
//
// Thread contract:
// - Make* and PublishLate may run on any thread.
// - Publish, Adopt and Grow* run on the owner thread.
// - Current, Key and the resolver calls may run on any thread.
class VersionedBuffer final : public VersionedResource {
public:
    // When a late-bound version is ready for its readers: when its staged uploads land (the upload planner orders them), or at
    // the frame wait.
    enum class LateReady : uint8_t { Uploads, FrameWait };
    // `first` becomes the current version. `owner` is kept alive with it, such as the lease of a pooled backing.
    static std::shared_ptr<VersionedBuffer> Create(std::shared_ptr<Buffer> first, std::shared_ptr<const void> owner = {});
    // A late-bound buffer. `standIn` is what its readers' declarations resolve to: any small materialized buffer that lives as
    // long as this one. It has no version until the first PublishLate.
    static std::shared_ptr<VersionedBuffer> CreateLateBound(std::shared_ptr<Buffer> standIn, LateReady ready = LateReady::Uploads);
    // Late-bound only, any thread: `backing` becomes the current version. `owner` is kept alive with the version (the lease
    // of a pooled backing, say). Hold the version while an upload into `backing` may still be recorded, and while the GPU
    // may read it.
    std::shared_ptr<const BufferVersion> PublishLate(std::shared_ptr<Buffer> backing, std::shared_ptr<const void> owner = {});

    [[nodiscard]] std::shared_ptr<const BufferVersion> Current() const noexcept {
        return std::static_pointer_cast<const BufferVersion>(CurrentVersion());
    }
    // The current version's buffer.
    [[nodiscard]] std::shared_ptr<Buffer> Get() const noexcept {
        const auto version = Current();
        return version ? version->buffer : nullptr;
    }
    Buffer* operator->() const noexcept { return Current()->buffer.get(); }

    // Owner thread: installs `next` as the current version. Not for a late-bound buffer (PublishLate).
    std::shared_ptr<const BufferVersion> Publish(std::shared_ptr<Buffer> next, std::shared_ptr<const void> owner = {});
    // Any thread. Makes `next` the next version, kept alive with `owner`. It is numbered but not current (pending). A
    // revision can name it, and its owner makes it current (Adopt) when that revision is selected. Until then, live
    // preparations use the current version.
    std::shared_ptr<const BufferVersion> Make(std::shared_ptr<Buffer> next, std::shared_ptr<const void> owner = {});
    // Owner thread: a pending version (Make, of this buffer) becomes the current one.
    void Adopt(const std::shared_ptr<const BufferVersion>& version) { AdoptVersion(version); }
    // Makes the next version, materialized. It is like the current version at another size: elements for a structured
    // buffer, or bytes for a raw buffer.
    // - MakeStructured and MakeBytes make it pending, and may run on any thread.
    // - Grow* makes it current, and runs on the owner thread.
    // The new version holds no data. Its maker fills it, and the old version is untouched. It is made without an ECS
    // entity, as a pooled backing is (Resource::ScopedECSRegistrationSuppression), because a producer's thread may not
    // touch the host's world.
    std::shared_ptr<const BufferVersion> MakeStructured(uint32_t elements);
    std::shared_ptr<const BufferVersion> MakeBytes(uint64_t bytes);
    std::shared_ptr<const BufferVersion> GrowStructured(uint32_t elements);
    std::shared_ptr<const BufferVersion> GrowBytes(uint64_t bytes);

    // A copy shares the versions (Key), as a clone does.
    VersionedBuffer(const VersionedBuffer&) = default;
    std::unique_ptr<IResourceResolver> Clone() const override { return std::unique_ptr<IResourceResolver>(new VersionedBuffer(*this)); }

private:
    VersionedBuffer() = default;
    std::shared_ptr<const ResourceVersion> Named(const IResourceVersions& versions) const noexcept override { return versions.Find(*this); }
};

} // namespace org
