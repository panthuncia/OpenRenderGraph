#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include <rhi.h>

#include "Interfaces/IResourceResolver.h"
#include "Render/PassExecutionContext.h"
#include "Resources/Resource.h"

namespace org {

class VersionedBuffer;
class VersionedTexture;
struct BufferVersion;
struct TextureVersion;

// One version of a versioned resource. It is immutable once made.
struct ResourceVersion {
    uint64_t number = 0;
    std::shared_ptr<Resource> resource;
    std::shared_ptr<const ResolverDeclarationState> declaration;  // what a preparation resolving this version declares
    // What keeps the version's backing, such as the lease of a pooled backing. It is released with the version.
    std::shared_ptr<const void> owner;
    // A late-bound version only (VersionedBuffer::PublishLate): its entry in the upload planner's map.
    std::shared_ptr<void> publication;
};

// What the graph needs from a resource whose backing changes by versions. The graph gives such a resource one slot. Each
// preparation binds the slot to the version it selects, as an overlay of that preparation's bindings. So a new version
// needs no binding edit and no lowering.
class IVersionedResource {
public:
    virtual ~IVersionedResource() = default;
    // Identifies the resource across its versions and its clones. It is the dependency identity its declarations name:
    // lowering finds the slot of a use by it.
    virtual const void* Key() const noexcept = 0;
    // The scheduling identity of the resource's slot. It stays the same across versions.
    virtual uint64_t SlotID() const noexcept = 0;
    // A late-bound resource's declarations resolve to a fixed stand-in. The graph binds the stand-in like any resource.
    virtual bool LateBound() const noexcept { return false; }
    // The version a preparation binds. That is the one the revision in `context` names (IResourceVersions), or else the
    // current one. Null when there is none.
    virtual std::shared_ptr<const ResourceVersion> SelectVersion(const ResolverCaptureContext* context) const = 0;
    // True when a recording made with one version may only run while that version is selected, such as a swapchain's
    // acquired image. The graph never keeps such recordings for reuse.
    virtual bool PerFrame() const noexcept { return false; }
};

// What a revision names as its immutable inputs: the version of each versioned resource its passes use. A preparation
// made for host data (the hostData of RenderGraph::PreparePersistentTicket) binds every versioned resource through it.
// So the preparation's bindings are the revision's. The host data returns this interface from
// TryGet(typeid(IResourceVersions)), as a pointer to its IResourceVersions subobject.
struct IResourceVersions {
    virtual ~IResourceVersions() = default;
    // The version of `buffer` the revision names. Null for the current one.
    virtual std::shared_ptr<const BufferVersion> Find(const VersionedBuffer& buffer) const noexcept = 0;
    // The version of `texture` the revision names. Null for the current one.
    virtual std::shared_ptr<const TextureVersion> FindTexture(const VersionedTexture&) const noexcept { return nullptr; }
};

// The shared implementation of VersionedBuffer and VersionedTexture. Its versions are immutable. A preparation selects one,
// and whoever holds a version keeps it alive: a revision, a recording, a frame in flight. There is no backing mutation and
// no wait.
//
// Passes declare the versioned resource itself, since it is a resolver. The persistent graph gives it one slot, which no
// publication binds. Each preparation binds the slot to the version it selects, as an overlay of that preparation's
// bindings. Admission, barriers and recording all see that version's backing.
//
// Every version must have the shape class of the first one. For a texture that is its format, mip count and slice count.
// Buffers have no constraint, since a buffer's size is not part of its graph slot. A resource of another shape class is a
// new resource, and its passes are declared again.
//
// Thread contract:
// - Making a version may run on any thread.
// - Adopting or publishing a version runs on the owner thread.
// - Reading the current version and the resolver calls may run on any thread. The current version is an atomic
//   shared_ptr, read whole.
class VersionedResource : public IResourceResolver, public IVersionedResource {
public:
    const void* Key() const noexcept final { return m_state->identity.get(); }
    uint64_t SlotID() const noexcept final { return m_state->slotID; }
    bool LateBound() const noexcept final { return m_state->standIn != nullptr; }
    std::shared_ptr<const ResourceVersion> SelectVersion(const ResolverCaptureContext* context) const final;

    // IResourceResolver: the current version, or the one the preparation's revision names. A late-bound resource resolves
    // to its stand-in.
    std::vector<std::shared_ptr<Resource>> Resolve() const override;
    std::shared_ptr<const ResolverDeclarationState> CaptureDeclarationState() const override;
    std::shared_ptr<const ResolverDeclarationState> CaptureDeclarationState(const ResolverCaptureContext& context) const override;
    // Always unknown. A revision may name a version other than the current one, so every preparation captures. The
    // declaration belongs to the version and is never rebuilt.
    uint64_t DeclarationVersionHint() const noexcept override { return 0; }

protected:
    // What every version of one resource must share. A buffer's is empty: its size is not part of its graph slot.
    struct ShapeClass {
        bool texture = false;
        rhi::Format format = rhi::Format::Unknown;
        uint32_t mips = 0, slices = 0;
        bool operator==(const ShapeClass&) const = default;
    };
    static ShapeClass ShapeClassOf(const Resource& resource);
    struct State {
        std::atomic<std::shared_ptr<const ResourceVersion>> current;
        std::atomic<uint64_t> published{0};  // number of the last version made, on any thread
        // The dependency identity of every declaration this resource makes. It is a token of its own, not the state.
        // The state holds the current version, and the version holds its declaration. If the declaration held the state,
        // the three would keep each other alive, and the last version would never be released.
        std::shared_ptr<const void> identity = std::make_shared<char>('\0');
        uint64_t slotID = Resource::ReserveGlobalResourceID();
        // The shape class every version must have. The first adopted version sets it, before anything else sees the resource.
        ShapeClass shape;
        bool shaped = false;
        // A late-bound resource only. These are set once at creation.
        std::shared_ptr<Resource> standIn;
        std::shared_ptr<const ResolverDeclarationState> lateDeclaration;
        bool lateFrameWait = false;
    };

    VersionedResource() : m_state(std::make_shared<State>()) {}
    VersionedResource(const VersionedResource&) = default;
    VersionedResource& operator=(const VersionedResource&) = default;

    [[nodiscard]] std::shared_ptr<const ResourceVersion> CurrentVersion() const noexcept {
        return m_state->current.load(std::memory_order_acquire);
    }
    // Any thread. Fills `version` as the next version, around `resource`, kept alive with `owner`. It is not current.
    // Throws when `resource` has another shape class than the first version.
    void MakeVersion(ResourceVersion& version, std::shared_ptr<Resource> resource, std::shared_ptr<const void> owner) const;
    // Owner thread: `version` becomes the current one. It must be this resource's.
    void AdoptVersion(const std::shared_ptr<const ResourceVersion>& version);
    // At creation: makes this a late-bound resource whose declarations resolve to `standIn`.
    void MakeLateBound(std::shared_ptr<Resource> standIn, bool frameWait);
    // The version `versions` names for this resource, or null.
    virtual std::shared_ptr<const ResourceVersion> Named(const IResourceVersions& versions) const noexcept = 0;

    std::shared_ptr<State> m_state;
};

} // namespace org
