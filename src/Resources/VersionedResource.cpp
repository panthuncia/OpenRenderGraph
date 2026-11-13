// Versioned resources: the shared implementation (VersionedResource) and its buffer and texture forms.

#include "Resources/VersionedResource.h"

#include "Resources/Buffers/VersionedBuffer.h"
#include "Resources/VersionedTexture.h"

#include <stdexcept>

namespace org {

VersionedResource::ShapeClass VersionedResource::ShapeClassOf(const Resource& resource) {
    ShapeClass shape;
    if (!resource.HasLayout()) return shape;
    shape.texture = true;
    shape.mips = resource.GetMipLevels();
    shape.slices = resource.GetArraySize();
    rhi::ResourceDesc description{};
    if (resource.TryGetRHIResourceDesc(description) && description.type != rhi::ResourceType::Buffer)
        shape.format = description.texture.format;
    return shape;
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
        throw std::invalid_argument("VersionedResource: version '" + resource->GetName() + "' has another format, mip count or slice "
            "count than the first version. A resource of another shape is a new resource: declare its passes again");
    version.number = m_state->published.fetch_add(1, std::memory_order_relaxed) + 1;
    auto declaration = std::make_shared<ResolverDeclarationState>();
    declaration->tracked = true;
    declaration->dependencyIdentity = m_state;
    declaration->resourceSetIdentity = {resource->GetSchedulingResourceID(), version.number};
    declaration->resources = std::make_shared<const ResolverResourceList>(ResolverResourceList{resource});
    version.declaration = std::move(declaration);
    version.resource = std::move(resource);
    version.owner = std::move(owner);
}

void VersionedResource::AdoptVersion(const std::shared_ptr<const ResourceVersion>& version) {
    if (!version || !version->declaration || version->declaration->dependencyIdentity != m_state)
        throw std::invalid_argument("VersionedResource: adopting another resource's version");
    if (!m_state->shaped && version->resource) {
        m_state->shape = ShapeClassOf(*version->resource);
        m_state->shaped = true;
    }
    m_state->current.store(version, std::memory_order_release);
}

void VersionedResource::MakeLateBound(std::shared_ptr<Resource> standIn, bool frameWait) {
    if (!standIn) throw std::invalid_argument("VersionedResource: a late-bound resource needs a stand-in");
    auto declaration = std::make_shared<ResolverDeclarationState>();
    declaration->tracked = true;
    declaration->dependencyIdentity = m_state;
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
    return version;
}

} // namespace org
