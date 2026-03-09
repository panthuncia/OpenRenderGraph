#pragma once

#include <vector>
#include <memory>

#include "Resources/ResourceStateTracker.h"

class IResourceResolver {
	public:
	virtual ~IResourceResolver() = default;
	virtual std::vector<std::shared_ptr<Resource>> Resolve() const = 0;

    /// Returns a monotonically-increasing version that changes whenever the
    /// resolved set changes.  A return value of 0 means "no version tracking —
    /// the caller must not assume the set is stable between frames."
    /// Resolvers backed by a versioned container (e.g. ResourceGroup) should
    /// override this so that the render graph can detect changes automatically.
    virtual uint64_t GetContentVersion() const { return 0; }

    template<typename T>
    std::vector<std::shared_ptr<T>> ResolveAs(bool require_all_casts = true) const {
        static_assert(std::is_base_of_v<Resource, T>, "T must derive from Resource");

        auto base = Resolve();
        std::vector<std::shared_ptr<T>> out;
        out.reserve(base.size());

        for (auto& p : base) {
            if (auto d = std::dynamic_pointer_cast<T>(p)) {
                out.push_back(std::move(d));
            }
            else if (require_all_casts) {
                assert(false && "Resource could not be cast to requested type");
            }
        }
        return out;
    }

    virtual std::unique_ptr<IResourceResolver> Clone() const = 0;
};

// Helper to avoid rewriting Clone in every derived type
template<class Derived>
struct ClonableResolver : IResourceResolver {
    std::unique_ptr<IResourceResolver> Clone() const override {
        return std::make_unique<Derived>(static_cast<const Derived&>(*this));
    }
};

struct ResourceResolverAndRange {
    ResourceResolverAndRange(const IResourceResolver& resolver) {
        range = {}; // Full range
        // Copy resolver into unique_ptr
        pResolver = resolver.Clone();
    }
    ResourceResolverAndRange(const ResourceIdentifier& resource, const RangeSpec& range) : range(range) {}
    std::unique_ptr<IResourceResolver> pResolver;
    RangeSpec range;
};

/// Snapshot of a resolver taken during DeclareResourceUsages.
/// The render graph stores these alongside each pass so it can detect
/// when a versioned resolver's content changes between frames and
/// automatically trigger re-declaration.
struct ResolverSnapshot {
    std::unique_ptr<IResourceResolver> resolver;
    uint64_t version = 0;

    ResolverSnapshot() = default;
    ResolverSnapshot(std::unique_ptr<IResourceResolver> r, uint64_t v)
        : resolver(std::move(r)), version(v) {}
    ResolverSnapshot(ResolverSnapshot&&) = default;
    ResolverSnapshot& operator=(ResolverSnapshot&&) = default;

    // Deep-copy via Clone() so that PassAndResources structs remain copyable.
    ResolverSnapshot(const ResolverSnapshot& other)
        : resolver(other.resolver ? other.resolver->Clone() : nullptr)
        , version(other.version) {}
    ResolverSnapshot& operator=(const ResolverSnapshot& other) {
        if (this != &other) {
            resolver = other.resolver ? other.resolver->Clone() : nullptr;
            version = other.version;
        }
        return *this;
    }
};