#pragma once

#include <vector>
#include <memory>
#include <cstdint>

#include "Resources/ResourceStateTracker.h"
#include "Resources/ResourceIdentifier.h"
#include "RenderPasses/Base/PassReturn.h"


namespace org {

struct ResolverRequirementBlock;

using ResolverResourceList = std::vector<std::shared_ptr<Resource>>;

struct ResolverResourceSetIdentity {
    uint64_t low = 0;
    uint64_t high = 0;
    auto operator<=>(const ResolverResourceSetIdentity&) const = default;
};

struct ResolverDeclarationState {
    std::shared_ptr<const void> dependencyIdentity;
    ResolverResourceSetIdentity resourceSetIdentity{};
    uint64_t contentRevision = 0;
    uint64_t waitRevision = 0;
    std::shared_ptr<const ResolverResourceList> resources;
    std::shared_ptr<const std::vector<ExternalTimelinePoint>> waits;
    bool tracked = false;
};

class IResourceResolver {
	public:
	virtual ~IResourceResolver() = default;
	virtual std::vector<std::shared_ptr<Resource>> Resolve() const = 0;
    virtual std::shared_ptr<const ResolverDeclarationState> CaptureDeclarationState() const = 0;

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

/// Snapshot of a resolver taken during DeclareResourceUsages. Resource-set
/// identity drives retained re-declaration; content and wait revisions are
/// tracked independently so they cannot invalidate the structural declaration.
struct ResolverSnapshot {
    std::unique_ptr<IResourceResolver> resolver;
    std::shared_ptr<const void> dependencyIdentity;
    ResolverResourceSetIdentity resourceSetIdentity{};
    uint64_t contentRevision = 0;
    uint64_t waitRevision = 0;
    std::vector<ExternalTimelinePoint> waits;
    std::vector<uint64_t> resourceIDs;
    struct RequirementTemplate {
        RangeSpec range{};
        ResourceState state{};
    };
    std::vector<RequirementTemplate> requirementTemplates;
    std::shared_ptr<const ResolverRequirementBlock> requirementBlock;

    ResolverSnapshot() = default;
    ResolverSnapshot(std::unique_ptr<IResourceResolver> r, const ResolverDeclarationState& state)
        : resolver(std::move(r)), dependencyIdentity(state.dependencyIdentity)
        , resourceSetIdentity(state.resourceSetIdentity)
        , contentRevision(state.contentRevision), waitRevision(state.waitRevision)
        , waits(state.waits ? *state.waits : std::vector<ExternalTimelinePoint>{}) {}
    ResolverSnapshot(ResolverSnapshot&&) = default;
    ResolverSnapshot& operator=(ResolverSnapshot&&) = default;

    // Deep-copy via Clone() so that PassAndResources structs remain copyable.
    ResolverSnapshot(const ResolverSnapshot& other)
        : resolver(other.resolver ? other.resolver->Clone() : nullptr)
        , dependencyIdentity(other.dependencyIdentity)
        , resourceSetIdentity(other.resourceSetIdentity)
        , contentRevision(other.contentRevision), waitRevision(other.waitRevision)
        , waits(other.waits), resourceIDs(other.resourceIDs)
        , requirementTemplates(other.requirementTemplates), requirementBlock(other.requirementBlock) {}
    ResolverSnapshot& operator=(const ResolverSnapshot& other) {
        if (this != &other) {
            resolver = other.resolver ? other.resolver->Clone() : nullptr;
            dependencyIdentity = other.dependencyIdentity;
            resourceSetIdentity = other.resourceSetIdentity;
            contentRevision = other.contentRevision;
            waitRevision = other.waitRevision;
            waits = other.waits;
            resourceIDs = other.resourceIDs;
            requirementTemplates = other.requirementTemplates;
            requirementBlock = other.requirementBlock;
        }
        return *this;
    }
};


} // namespace org
