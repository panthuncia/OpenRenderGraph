#pragma once

#include <memory>

#include "Resources/PixelBuffer.h"
#include "Resources/VersionedResource.h"

namespace org {

// One version of a VersionedTexture. It is immutable once made.
struct TextureVersion : ResourceVersion {
    std::shared_ptr<PixelBuffer> texture;  // the same object as `resource`
};

// A texture whose backing changes by versions (VersionedResource). Every version has the first one's format, mip count
// and slice count. Its extent may differ. A texture of another format, mip count or slice count is a new resource.
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
