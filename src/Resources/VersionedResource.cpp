// Versioned resources: the shared implementation (VersionedResource) and its buffer and texture forms.

#include "Resources/VersionedResource.h"

#include "Resources/Buffers/VersionedBuffer.h"
#include "Resources/VersionedTexture.h"

#include <algorithm>
#include <mutex>
#include <stdexcept>

namespace org {

namespace {
std::atomic<VersionedResource::AdoptionObserver> g_adoptionObserver{nullptr};
}

void VersionedResource::SetAdoptionObserver(AdoptionObserver observer) noexcept {
    g_adoptionObserver.store(observer, std::memory_order_release);
}

VersionedResource::ShapeClass VersionedResource::ShapeClassOf(const Resource& resource) {
    ShapeClass shape;
    if (!resource.HasLayout()) return shape;
    shape.texture = true;
    shape.mips = resource.GetMipLevels();
    rhi::ResourceDesc description{};
    if (resource.TryGetRHIResourceDesc(description) && description.type != rhi::ResourceType::Buffer)
        shape.format = description.texture.format;
    return shape;
}

uint32_t VersionedResource::SlicesOf(const Resource& resource) noexcept {
    return resource.HasLayout() ? (std::max)(1u, static_cast<uint32_t>(resource.GetArraySize())) : 0u;
}

std::shared_ptr<const ResourceVersion> VersionedResource::Reshaped() const {
    std::scoped_lock lock(m_state->reshapeMutex);
    return m_state->reshaped.lock();
}

void VersionedResource::NoteMade(const std::shared_ptr<const ResourceVersion>& version) const {
    if (!version || !version->resource) return;
    const auto current = CurrentVersion();
    const bool reshapes = current && current->resource && SlicesOf(*version->resource) != SlicesOf(*current->resource);
    std::scoped_lock lock(m_state->reshapeMutex);
    // Only a newer version replaces the noted one. A version with the current slice count clears it.
    if (const auto noted = m_state->reshaped.lock(); noted && noted->number > version->number) return;
    if (reshapes) m_state->reshaped = version;
    else m_state->reshaped.reset();
}

std::shared_ptr<const ResourceVersion> VersionedResource::SelectVersion(const ResolverCaptureContext* context) const {
    if (!LateBound() && context)
        if (const auto data = context->Get<IHostExecutionData>())
            if (const auto* versions = data->Get<IResourceVersions>())
                if (auto version = Named(*versions)) return version;
    return CurrentVersion();
}

std::vector<std::shared_ptr<Resource>> VersionedResource::Resolve() const {
    if (LateBound()) return {m_state->standIn};
    const auto version = CurrentVersion();
    return {version ? version->resource : nullptr};
}

std::shared_ptr<const ResolverDeclarationState> VersionedResource::CaptureDeclarationState() const {
    if (LateBound()) return m_state->lateDeclaration;
    const auto version = CurrentVersion();
    return version ? version->declaration : nullptr;
}

std::shared_ptr<const ResolverDeclarationState> VersionedResource::CaptureDeclarationState(const ResolverCaptureContext& context) const {
    if (LateBound()) return m_state->lateDeclaration;
    const auto version = SelectVersion(&context);
    return version ? version->declaration : nullptr;
}

void VersionedResource::MakeVersion(ResourceVersion& version, std::shared_ptr<Resource> resource, std::shared_ptr<const void> owner) const {
    if (LateBound()) throw std::logic_error("VersionedResource: a late-bound resource's versions come from PublishLate");
    if (!resource) throw std::invalid_argument("VersionedResource: a version needs a resource");
    // The first version sets the shape class. Versions made before it is set (none, in practice) are not checked.
    if (m_state->shaped && !(ShapeClassOf(*resource) == m_state->shape))
        throw std::invalid_argument("VersionedResource: version '" + resource->GetName() + "' has another format or mip count than the "
            "first version. A resource of another shape is a new resource: declare its passes again");
    version.number = m_state->published.fetch_add(1, std::memory_order_relaxed) + 1;
    auto declaration = std::make_shared<ResolverDeclarationState>();
    declaration->tracked = true;
    declaration->dependencyIdentity = m_state->identity;
    declaration->resourceSetIdentity = {resource->GetSchedulingResourceID(), version.number};
    declaration->resources = std::make_shared<const ResolverResourceList>(ResolverResourceList{resource});
    version.declaration = std::move(declaration);
    version.resource = std::move(resource);
    version.owner = std::move(owner);
}

void VersionedResource::AdoptVersion(const std::shared_ptr<const ResourceVersion>& version) {
    if (!version || !version->declaration || version->declaration->dependencyIdentity != m_state->identity)
        throw std::invalid_argument("VersionedResource: adopting another resource's version");
    if (!m_state->shaped && version->resource) {
        m_state->shape = ShapeClassOf(*version->resource);
        m_state->shaped = true;
    }
    m_state->current.store(version, std::memory_order_release);
    {
        // The reshaped version is pending no more once it, or a newer version, is current.
        std::scoped_lock lock(m_state->reshapeMutex);
        if (const auto noted = m_state->reshaped.lock(); !noted || noted->number <= version->number) m_state->reshaped.reset();
    }
    if (const auto observer = g_adoptionObserver.load(std::memory_order_relaxed)) observer(*this, *version);
}

void VersionedResource::MakeLateBound(std::shared_ptr<Resource> standIn, bool frameWait) {
    if (!standIn) throw std::invalid_argument("VersionedResource: a late-bound resource needs a stand-in");
    auto declaration = std::make_shared<ResolverDeclarationState>();
    declaration->tracked = true;
    declaration->dependencyIdentity = m_state->identity;
    declaration->resourceSetIdentity = {standIn->GetSchedulingResourceID(), 0};
    declaration->resources = std::make_shared<const ResolverResourceList>(ResolverResourceList{standIn});
    m_state->lateDeclaration = std::move(declaration);
    m_state->standIn = std::move(standIn);
    m_state->lateFrameWait = frameWait;
}

// ---------------------------------------------------------------------------
// VersionedBuffer

std::shared_ptr<VersionedBuffer> VersionedBuffer::Create(std::shared_ptr<Buffer> first, std::shared_ptr<const void> owner) {
    auto versioned = std::shared_ptr<VersionedBuffer>(new VersionedBuffer());
    versioned->Publish(std::move(first), std::move(owner));
    return versioned;
}

std::shared_ptr<VersionedBuffer> VersionedBuffer::CreateLateBound(std::shared_ptr<Buffer> standIn, LateReady ready) {
    auto versioned = std::shared_ptr<VersionedBuffer>(new VersionedBuffer());
    versioned->MakeLateBound(std::move(standIn), ready == LateReady::FrameWait);
    return versioned;
}

std::shared_ptr<const BufferVersion> VersionedBuffer::PublishLate(std::shared_ptr<Buffer> backing, std::shared_ptr<const void> owner) {
    if (!LateBound()) throw std::logic_error("VersionedBuffer: PublishLate needs a late-bound buffer");
    if (!backing) throw std::invalid_argument("VersionedBuffer: a version needs a buffer");
    auto version = std::make_shared<BufferVersion>();
    version->number = m_state->published.fetch_add(1, std::memory_order_relaxed) + 1;
    version->declaration = m_state->lateDeclaration;
    const auto standIn = m_state->standIn->GetAPIResource().GetHandle();
    version->publication = detail::BindLateBacking(backing->GetAPIResource().GetHandle(), standIn);
    if (m_state->lateFrameWait) detail::MarkFrameWaitStandIn(standIn);
    if (!version->publication) throw std::runtime_error("VersionedBuffer: a late version's backing or stand-in is not materialized");
    version->owner = std::move(owner);
    version->resource = backing;
    version->buffer = std::move(backing);
    m_state->current.store(version, std::memory_order_release);
    return version;
}

std::shared_ptr<const BufferVersion> VersionedBuffer::Publish(std::shared_ptr<Buffer> next, std::shared_ptr<const void> owner) {
    auto version = Make(std::move(next), std::move(owner));
    Adopt(version);
    return version;
}

std::shared_ptr<const BufferVersion> VersionedBuffer::Make(std::shared_ptr<Buffer> next, std::shared_ptr<const void> owner) {
    auto version = std::make_shared<BufferVersion>();
    version->buffer = next;
    MakeVersion(*version, std::move(next), std::move(owner));
    return version;
}

std::shared_ptr<const BufferVersion> VersionedBuffer::MakeStructured(uint32_t elements) {
    Resource::ScopedECSRegistrationSuppression suppressECS;
    auto next = Get()->UnmaterializedLike();
    next->ResizeStructured(elements);
    next->Materialize();
    return Make(std::move(next));
}

std::shared_ptr<const BufferVersion> VersionedBuffer::MakeBytes(uint64_t bytes) {
    Resource::ScopedECSRegistrationSuppression suppressECS;
    auto next = Get()->UnmaterializedLike();
    next->ResizeBytes(bytes);
    next->Materialize();
    return Make(std::move(next));
}

std::shared_ptr<const BufferVersion> VersionedBuffer::GrowStructured(uint32_t elements) {
    auto version = MakeStructured(elements);
    Adopt(version);
    return version;
}

std::shared_ptr<const BufferVersion> VersionedBuffer::GrowBytes(uint64_t bytes) {
    auto version = MakeBytes(bytes);
    Adopt(version);
    return version;
}

// ---------------------------------------------------------------------------
// VersionedTexture

std::shared_ptr<VersionedTexture> VersionedTexture::Create(std::shared_ptr<PixelBuffer> first, std::shared_ptr<const void> owner) {
    auto versioned = std::shared_ptr<VersionedTexture>(new VersionedTexture());
    versioned->Publish(std::move(first), std::move(owner));
    return versioned;
}

std::shared_ptr<const TextureVersion> VersionedTexture::Publish(std::shared_ptr<PixelBuffer> next, std::shared_ptr<const void> owner) {
    auto version = Make(std::move(next), std::move(owner));
    Adopt(version);
    return version;
}

std::shared_ptr<const TextureVersion> VersionedTexture::Make(std::shared_ptr<PixelBuffer> next, std::shared_ptr<const void> owner) {
    auto version = std::make_shared<TextureVersion>();
    version->texture = next;
    MakeVersion(*version, std::move(next), std::move(owner));
    NoteMade(version);
    return version;
}

std::shared_ptr<VersionedTexture> VersionedTexture::CreateImage(std::shared_ptr<GloballyIndexedResource> first, std::shared_ptr<const void> owner) {
    auto versioned = std::shared_ptr<VersionedTexture>(new VersionedTexture());
    versioned->PublishImage(std::move(first), std::move(owner));
    return versioned;
}

std::shared_ptr<const TextureVersion> VersionedTexture::PublishImage(std::shared_ptr<GloballyIndexedResource> next, std::shared_ptr<const void> owner) {
    auto version = MakeImage(std::move(next), std::move(owner));
    Adopt(version);
    return version;
}

std::shared_ptr<const TextureVersion> VersionedTexture::MakeImage(std::shared_ptr<GloballyIndexedResource> next, std::shared_ptr<const void> owner) {
    if (!next || !next->HasLayout()) throw std::invalid_argument("VersionedTexture: an image version needs a texture");
    auto version = std::make_shared<TextureVersion>();
    version->texture = std::dynamic_pointer_cast<PixelBuffer>(next);
    MakeVersion(*version, std::move(next), std::move(owner));
    NoteMade(version);
    return version;
}

} // namespace org
