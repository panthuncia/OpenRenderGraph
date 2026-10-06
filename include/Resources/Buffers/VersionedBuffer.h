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
};

// What a revision names (immutable inputs): the version of each versioned buffer its passes use. A preparation made for host data
// (RenderGraph::PreparePersistentTicket's hostData) resolves every VersionedBuffer through it, so its publication and bindings are
// the revision's. The host data returns this interface from TryGet(typeid(IResourceVersions)), as a pointer to its
// IResourceVersions subobject.
struct IResourceVersions {
    virtual ~IResourceVersions() = default;
    // The version of `buffer` the revision names; null for the current one.
    virtual std::shared_ptr<const BufferVersion> Find(const VersionedBuffer& buffer) const noexcept = 0;
};

// A buffer whose growth is a new version, never a backing changed in place. Passes declare it (it is a resolver); a preparation
// resolves the version its revision names (IResourceVersions), else the current one. A version stays alive while a revision, a
// recording (its bindings hold the backing) or a frame holds it, so the work recorded for an older revision runs unchanged while
// a newer one is recorded: no backing mutation, no wait.
//
// Growth as graph work (the SARP/BasicRenderer model): a producer on any thread makes the next version (Make*, which materializes
// it), fills it through the dedicated uploader (IUploadService::QueueTrackedStreamingUploadSegments, a WorkerOwnedDestination of
// ownership PendingVersion: no queue uses a version before it is adopted), and the version is ready once those copies complete;
// the owner adopts it when the revision that names it is selected.
//
// Thread contract: Make* on any thread; Publish, Adopt and Grow* on the owner thread; Current, Key and the resolver calls on any
// thread (the current version is an atomic shared_ptr, read whole).
class VersionedBuffer final : public ClonableResolver<VersionedBuffer> {
public:
    static std::shared_ptr<VersionedBuffer> Create(std::shared_ptr<Buffer> first) {
        auto versioned = std::shared_ptr<VersionedBuffer>(new VersionedBuffer());
        versioned->Publish(std::move(first));
        return versioned;
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

    // Owner thread: installs `next` as the current version.
    std::shared_ptr<const BufferVersion> Publish(std::shared_ptr<Buffer> next) {
        auto version = Make(std::move(next));
        Adopt(version);
        return version;
    }
    // Any thread: the next version of `next`, numbered but not current (pending): a revision can name it, and its owner makes
    // it current (Adopt) when that revision is selected. Until then the current version is what writes and live preparations use.
    std::shared_ptr<const BufferVersion> Make(std::shared_ptr<Buffer> next) {
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
    // The next version, like the current one at another size (a structured buffer's elements, or a raw buffer's bytes),
    // materialized; pending (MakeStructured, MakeBytes: any thread) or current (Grow*: the owner's). It holds nothing: its maker
    // fills it (the old version is untouched). Made without an ECS entity, as a pooled backing is (Resource::
    // ScopedECSRegistrationSuppression): a producer's thread may not touch the host's world.
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

    // IResourceResolver: the current version, or the one the preparation's revision names.
    std::vector<std::shared_ptr<Resource>> Resolve() const override { return {Get()}; }
    std::shared_ptr<const ResolverDeclarationState> CaptureDeclarationState() const override { return Current()->declaration; }
    std::shared_ptr<const ResolverDeclarationState> CaptureDeclarationState(const ResolverCaptureContext& context) const override {
        if (const auto data = context.Get<IHostExecutionData>())
            if (const auto* versions = data->Get<IResourceVersions>())
                if (const auto version = versions->Find(*this)) return version->declaration;
        return Current()->declaration;
    }
    // Unknown: a revision may name another version than the current one, so every preparation captures (the declaration is the
    // version's own, never rebuilt).
    uint64_t DeclarationVersionHint() const noexcept override { return 0; }

private:
    VersionedBuffer() : m_state(std::make_shared<State>()) {}
    struct State {
        std::atomic<std::shared_ptr<const BufferVersion>> current;
        std::atomic<uint64_t> published{ 0 };  // the last version's number (Make, any thread)
    };
    std::shared_ptr<State> m_state;
};

} // namespace org
