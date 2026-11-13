#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <typeinfo>

#include "Interfaces/IResourceResolver.h"
#include "Render/PassExecutionContext.h"
#include "Resources/Buffers/Buffer.h"

namespace org {

class VersionedBuffer;

// One version of a VersionedBuffer: immutable once published.
struct BufferVersion {
    uint64_t number = 0;
    std::shared_ptr<Buffer> buffer;
    std::shared_ptr<const ResolverDeclarationState> declaration;  // what a preparation resolving this version declares
    // A late-bound version only (VersionedBuffer::PublishLate): what keeps its backing (a pooled lease, say), and its entry in
    // the upload planner's map. Both are released with the version.
    std::shared_ptr<const void> owner;
    std::shared_ptr<void> publication;
};

namespace detail {
// Late-bound versions' backings, by the stand-in their readers declare (VersionedBuffer::CreateLateBound). The upload planner
// (RenderGraph::RecordPendingUploads' queue-scoped uploads) reads it. Any thread.
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

// What a revision names as its immutable inputs: the version of each versioned buffer its passes use. A preparation
// made for host data (the hostData of RenderGraph::PreparePersistentTicket) resolves every VersionedBuffer through
// it. So the preparation's publication and bindings are the revision's. The host data returns this interface from
// TryGet(typeid(IResourceVersions)), as a pointer to its IResourceVersions subobject.
struct IResourceVersions {
    virtual ~IResourceVersions() = default;
    // The version of `buffer` the revision names; null for the current one.
    virtual std::shared_ptr<const BufferVersion> Find(const VersionedBuffer& buffer) const noexcept = 0;
};

// A buffer whose growth makes a new version, instead of changing a backing in place. Passes declare it, since it is a
// resolver. A preparation resolves the version its revision names (IResourceVersions), or else the current one.
// A version stays alive while a revision, a recording or a frame holds it. A recording holds it through its bindings,
// which hold the backing. So the work recorded for an older revision runs unchanged while a newer one is recorded.
// There is no backing mutation and no wait.
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
// - Current, Key and the resolver calls may run on any thread. The current version is an atomic shared_ptr, read
//   whole.
class VersionedBuffer final : public ClonableResolver<VersionedBuffer> {
public:
    // When a late-bound version is ready for its readers: when its staged uploads land (the upload planner orders them), or at
    // the frame wait.
    enum class LateReady : uint8_t { Uploads, FrameWait };
    static std::shared_ptr<VersionedBuffer> Create(std::shared_ptr<Buffer> first) {
        auto versioned = std::shared_ptr<VersionedBuffer>(new VersionedBuffer());
        versioned->Publish(std::move(first));
        return versioned;
    }
    // A late-bound buffer. `standIn` is what its readers' declarations resolve to: any small materialized buffer that lives as
    // long as this one. It has no version until the first PublishLate.
    static std::shared_ptr<VersionedBuffer> CreateLateBound(std::shared_ptr<Buffer> standIn, LateReady ready = LateReady::Uploads) {
        if (!standIn) throw std::invalid_argument("VersionedBuffer: a late-bound buffer needs a stand-in");
        auto versioned = std::shared_ptr<VersionedBuffer>(new VersionedBuffer());
        auto declaration = std::make_shared<ResolverDeclarationState>();
        declaration->tracked = true;
        declaration->dependencyIdentity = versioned->m_state;
        declaration->resourceSetIdentity = {standIn->GetSchedulingResourceID(), 0};
        declaration->resources = std::make_shared<const ResolverResourceList>(ResolverResourceList{standIn});
        versioned->m_state->lateDeclaration = std::move(declaration);
        versioned->m_state->standIn = std::move(standIn);
        versioned->m_state->ready = ready;
        return versioned;
    }
    [[nodiscard]] bool LateBound() const noexcept { return m_state->standIn != nullptr; }
    // Late-bound only, any thread: `backing` becomes the current version. `owner` is kept alive with the version (the lease
    // of a pooled backing, say). Hold the version while an upload into `backing` may still be recorded, and while the GPU
    // may read it.
    std::shared_ptr<const BufferVersion> PublishLate(std::shared_ptr<Buffer> backing, std::shared_ptr<const void> owner = {}) {
        if (!LateBound()) throw std::logic_error("VersionedBuffer: PublishLate needs a late-bound buffer");
        if (!backing) throw std::invalid_argument("VersionedBuffer: a version needs a buffer");
        auto version = std::make_shared<BufferVersion>();
        version->number = m_state->published.fetch_add(1, std::memory_order_relaxed) + 1;
        version->declaration = m_state->lateDeclaration;
        const auto standIn = m_state->standIn->GetAPIResource().GetHandle();
        version->publication = detail::BindLateBacking(backing->GetAPIResource().GetHandle(), standIn);
        if (m_state->ready == LateReady::FrameWait) detail::MarkFrameWaitStandIn(standIn);
        if (!version->publication) throw std::runtime_error("VersionedBuffer: a late version's backing or stand-in is not materialized");
        version->owner = std::move(owner);
        version->buffer = std::move(backing);
        m_state->current.store(version, std::memory_order_release);
        return version;
    }

    [[nodiscard]] std::shared_ptr<const BufferVersion> Current() const noexcept {
        return m_state->current.load(std::memory_order_acquire);
    }
    // The current version's buffer.
    [[nodiscard]] std::shared_ptr<Buffer> Get() const noexcept {
        const auto version = Current();
        return version ? version->buffer : nullptr;
    }
    Buffer* operator->() const noexcept { return Current()->buffer.get(); }
    // Identifies the buffer across its versions (and the resolver's clones).
    [[nodiscard]] const void* Key() const noexcept { return m_state.get(); }

    // Owner thread: installs `next` as the current version. Not for a late-bound buffer (PublishLate).
    std::shared_ptr<const BufferVersion> Publish(std::shared_ptr<Buffer> next) {
        auto version = Make(std::move(next));
        Adopt(version);
        return version;
    }
    // Any thread. Makes `next` the next version. It is numbered but not current (pending). A revision can name it, and
    // its owner makes it current (Adopt) when that revision is selected. Until then, writes and live preparations use the
    // current version.
    std::shared_ptr<const BufferVersion> Make(std::shared_ptr<Buffer> next) {
        if (LateBound()) throw std::logic_error("VersionedBuffer: a late-bound buffer's versions come from PublishLate");
        if (!next) throw std::invalid_argument("VersionedBuffer: a version needs a buffer");
        auto version = std::make_shared<BufferVersion>();
        version->number = m_state->published.fetch_add(1, std::memory_order_relaxed) + 1;
        auto declaration = std::make_shared<ResolverDeclarationState>();
        declaration->tracked = true;
        declaration->dependencyIdentity = m_state;
        declaration->resourceSetIdentity = {next->GetSchedulingResourceID(), version->number};
        declaration->resources = std::make_shared<const ResolverResourceList>(ResolverResourceList{next});
        version->declaration = std::move(declaration);
        version->buffer = std::move(next);
        return version;
    }
    // Owner thread: a pending version (Make, of this buffer) becomes the current one.
    void Adopt(const std::shared_ptr<const BufferVersion>& version) {
        if (!version || version->declaration->dependencyIdentity != m_state)
            throw std::invalid_argument("VersionedBuffer: adopting another buffer's version");
        m_state->current.store(version, std::memory_order_release);
    }
    // Makes the next version, materialized. It is like the current version at another size: elements for a structured
    // buffer, or bytes for a raw buffer.
    // - MakeStructured and MakeBytes make it pending, and may run on any thread.
    // - Grow* makes it current, and runs on the owner thread.
    // The new version holds no data. Its maker fills it, and the old version is untouched. It is made without an ECS
    // entity, as a pooled backing is (Resource::ScopedECSRegistrationSuppression), because a producer's thread may not
    // touch the host's world.
    std::shared_ptr<const BufferVersion> MakeStructured(uint32_t elements) {
        Resource::ScopedECSRegistrationSuppression suppressECS;
        auto next = Get()->UnmaterializedLike();
        next->ResizeStructured(elements);
        next->Materialize();
        return Make(std::move(next));
    }
    std::shared_ptr<const BufferVersion> MakeBytes(uint64_t bytes) {
        Resource::ScopedECSRegistrationSuppression suppressECS;
        auto next = Get()->UnmaterializedLike();
        next->ResizeBytes(bytes);
        next->Materialize();
        return Make(std::move(next));
    }
    std::shared_ptr<const BufferVersion> GrowStructured(uint32_t elements) {
        auto version = MakeStructured(elements);
        Adopt(version);
        return version;
    }
    std::shared_ptr<const BufferVersion> GrowBytes(uint64_t bytes) {
        auto version = MakeBytes(bytes);
        Adopt(version);
        return version;
    }

    // IResourceResolver: the current version, or the one the preparation's revision names. A late-bound buffer: its stand-in.
    std::vector<std::shared_ptr<Resource>> Resolve() const override {
        if (LateBound()) return {m_state->standIn};
        return {Get()};
    }
    std::shared_ptr<const ResolverDeclarationState> CaptureDeclarationState() const override {
        if (LateBound()) return m_state->lateDeclaration;
        return Current()->declaration;
    }
    std::shared_ptr<const ResolverDeclarationState> CaptureDeclarationState(const ResolverCaptureContext& context) const override {
        if (LateBound()) return m_state->lateDeclaration;
        if (const auto data = context.Get<IHostExecutionData>())
            if (const auto* versions = data->Get<IResourceVersions>())
                if (const auto version = versions->Find(*this)) return version->declaration;
        return Current()->declaration;
    }
    // Always unknown. A revision may name a version other than the current one, so every preparation captures.
    // The declaration belongs to the version and is never rebuilt.
    uint64_t DeclarationVersionHint() const noexcept override { return 0; }

private:
    VersionedBuffer() : m_state(std::make_shared<State>()) {}
    struct State {
        std::atomic<std::shared_ptr<const BufferVersion>> current;
        std::atomic<uint64_t> published{ 0 };  // number of the last version made (by Make, on any thread)
        // A late-bound buffer only: what its declarations resolve to, set once at creation.
        std::shared_ptr<Buffer> standIn;
        std::shared_ptr<const ResolverDeclarationState> lateDeclaration;
        LateReady ready = LateReady::Uploads;
    };
    std::shared_ptr<State> m_state;
};

} // namespace org
