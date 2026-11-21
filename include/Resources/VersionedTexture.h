#pragma once

#include <memory>

#include "Resources/GloballyIndexedResource.h"
#include "Resources/PixelBuffer.h"
#include "Resources/VersionedResource.h"

namespace org {

// One version of a VersionedTexture. It is immutable once made.
struct TextureVersion : ResourceVersion {
    std::shared_ptr<PixelBuffer> texture;  // the same object as `resource`; null for an image version (MakeImage)
};

// A texture whose backing changes by versions (VersionedResource). Every version has the first one's format and mip count.
// Its extent and its slice count may differ; a new slice count switches in with graphs built for it (VersionedResource,
// "Array-size versions"). A texture of another format or mip count is a new resource.
// Passes declare it, since it is a resolver. A preparation binds the version its revision names
// (IResourceVersions::FindTexture), or else the current one.
//
// Thread contract:
// - Make may run on any thread.
// - Publish and Adopt run on the owner thread.
// - Current, Key and the resolver calls may run on any thread.
class VersionedTexture final : public VersionedResource {
public:
    // `first` becomes the current version. `owner` is kept alive with it, such as the lease of a pooled backing.
    static std::shared_ptr<VersionedTexture> Create(std::shared_ptr<PixelBuffer> first, std::shared_ptr<const void> owner = {});

    [[nodiscard]] std::shared_ptr<const TextureVersion> Current() const noexcept {
        return std::static_pointer_cast<const TextureVersion>(CurrentVersion());
    }
    // The current version's texture.
    [[nodiscard]] std::shared_ptr<PixelBuffer> Get() const noexcept {
        const auto version = Current();
        return version ? version->texture : nullptr;
    }
    PixelBuffer* operator->() const noexcept { return Current()->texture.get(); }

    // Owner thread: installs `next` as the current version.
    std::shared_ptr<const TextureVersion> Publish(std::shared_ptr<PixelBuffer> next, std::shared_ptr<const void> owner = {});
    // Any thread. Makes `next` the next version, kept alive with `owner`. It is not current until its owner adopts it.
    std::shared_ptr<const TextureVersion> Make(std::shared_ptr<PixelBuffer> next, std::shared_ptr<const void> owner = {});
    // Owner thread: a pending version (Make, of this texture) becomes the current one.
    void Adopt(const std::shared_ptr<const TextureVersion>& version) { AdoptVersion(version); }

    // Image versions: a texture the graph does not make, such as an imported or exported image (ExternalTextureResource).
    // The same rules hold: one shape class, immutable versions. Such a version's `texture` is null; `resource` holds it.
    static std::shared_ptr<VersionedTexture> CreateImage(std::shared_ptr<GloballyIndexedResource> first, std::shared_ptr<const void> owner = {});
    // Owner thread: installs `next` as the current version.
    std::shared_ptr<const TextureVersion> PublishImage(std::shared_ptr<GloballyIndexedResource> next, std::shared_ptr<const void> owner = {});
    // Any thread. Makes `next` the next version. It is not current until its owner adopts it.
    std::shared_ptr<const TextureVersion> MakeImage(std::shared_ptr<GloballyIndexedResource> next, std::shared_ptr<const void> owner = {});
    // The current version's resource, a PixelBuffer or an image.
    [[nodiscard]] std::shared_ptr<Resource> CurrentResource() const noexcept {
        const auto version = CurrentVersion();
        return version ? version->resource : nullptr;
    }

    // A copy shares the versions (Key), as a clone does.
    VersionedTexture(const VersionedTexture&) = default;
    std::unique_ptr<IResourceResolver> Clone() const override { return std::unique_ptr<IResourceResolver>(new VersionedTexture(*this)); }

private:
    VersionedTexture() = default;
    std::shared_ptr<const ResourceVersion> Named(const IResourceVersions& versions) const noexcept override {
        return versions.FindTexture(*this);
    }
};

} // namespace org
