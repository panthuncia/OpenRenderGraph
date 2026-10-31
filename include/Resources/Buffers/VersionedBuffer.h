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
// Thread contract:
// - Make* may run on any thread.
// - Publish, Adopt and Grow* run on the owner thread.
// - Current, Key and the resolver calls may run on any thread. The current version is an atomic shared_ptr, read
//   whole.
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
    // Any thread. Makes `next` the next version. It is numbered but not current (pending). A revision can name it, and
    // its owner makes it current (Adopt) when that revision is selected. Until then, writes and live preparations use the
    // current version.
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

    // IResourceResolver: the current version, or the one the preparation's revision names.
    std::vector<std::shared_ptr<Resource>> Resolve() const override { return {Get()}; }
    std::shared_ptr<const ResolverDeclarationState> CaptureDeclarationState() const override { return Current()->declaration; }
    std::shared_ptr<const ResolverDeclarationState> CaptureDeclarationState(const ResolverCaptureContext& context) const override {
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
    };
    std::shared_ptr<State> m_state;
};

} // namespace org
