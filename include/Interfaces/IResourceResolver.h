#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <typeindex>

#include "Resources/ResourceStateTracker.h"
#include "Resources/ResourceIdentifier.h"
#include "RenderPasses/Base/PassReturn.h"
#include "Render/PublicationBindingBundle.h"


namespace org {

struct ResolverRequirementBlock;

using ResolverResourceList = std::vector<std::shared_ptr<Resource>>;

// An owned, typed publication token. The preparation owner supplies it.
// Workers read captured declaration states. They never read a live publication source of the process.
class ResolverCaptureContext {
public:
    template<class T>
    explicit ResolverCaptureContext(std::shared_ptr<const T> publication,
        std::shared_ptr<const PublicationBindingBundle> bindings = {})
        : m_type(typeid(T)), m_publication(std::move(publication)), m_bindings(std::move(bindings)) {}
    const std::shared_ptr<const PublicationBindingBundle>& BindingBundle() const noexcept { return m_bindings; }
    template<class T> std::shared_ptr<const T> Get() const noexcept {
        return m_type == std::type_index(typeid(T))
            ? std::static_pointer_cast<const T>(m_publication) : nullptr;
    }
    template<class T> bool Is() const noexcept { return m_type == std::type_index(typeid(T)); }
private:
    std::type_index m_type;
    std::shared_ptr<const void> m_publication;
    std::shared_ptr<const PublicationBindingBundle> m_bindings;
};

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
    std::shared_ptr<const void> publicationLease;
    bool tracked = false;
};

class IResourceResolver {
	public:
	virtual ~IResourceResolver() = default;
	virtual std::vector<std::shared_ptr<Resource>> Resolve() const = 0;
    virtual std::shared_ptr<const ResolverDeclarationState> CaptureDeclarationState() const = 0;
    virtual std::shared_ptr<const ResolverDeclarationState> CaptureDeclarationState(
        const ResolverCaptureContext&) const { return CaptureDeclarationState(); }
    // A cheap change hint for per-frame polling. While the returned value is
    // nonzero and unchanged, CaptureDeclarationState() would return an
    // equivalent state. Zero means the state is unknown, and the caller must capture it to find out.
    virtual uint64_t DeclarationVersionHint() const noexcept { return 0; }
    // How many members the graph should reserve slots for when it first declares this resolver. Going past the
    // reserved count needs a structural build, so a resolver whose set grows over time should return its expected
    // size. Zero leaves it to the graph. It must depend only on the resolver's configuration, so clones agree.
    virtual uint32_t CapacityHint() const noexcept { return 0; }

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
    std::unique_ptr<IResourceResolver> pResolver;
    RangeSpec range;
};

/// A snapshot of a resolver, taken during DeclareResourceUsages. A change in the
/// identity of the captured resource set triggers re-declaration. Content and wait
/// revisions are tracked separately, so they cannot invalidate the structural declaration.
struct ResolverSnapshot {
    std::unique_ptr<IResourceResolver> resolver;
    // The resolver's state as the declaration captured it (see ResolverCaptureScope). The declaration's uses,
    // its views and the group it lowers to all describe this one resource set.
    std::shared_ptr<const ResolverDeclarationState> state;
    struct RequirementTemplate {
        RangeSpec range{};
        ResourceState state{};
    };
    std::vector<RequirementTemplate> requirementTemplates;
    // Authored state exists even when the first captured resource set is empty.
    std::vector<RequirementTemplate> declaredRequirementTemplates;
    bool hasUnclassifiedDeclaration = false;
    std::shared_ptr<const ResolverRequirementBlock> requirementBlock;

    ResolverSnapshot() = default;
    ResolverSnapshot(std::unique_ptr<IResourceResolver> r, std::shared_ptr<const ResolverDeclarationState> captured)
        : resolver(std::move(r)), state(std::move(captured)) {}
    ResolverSnapshot(ResolverSnapshot&&) = default;
    ResolverSnapshot& operator=(ResolverSnapshot&&) = default;

    // Deep-copy via Clone() so that PassAndResources structs remain copyable.
    ResolverSnapshot(const ResolverSnapshot& other)
        : resolver(other.resolver ? other.resolver->Clone() : nullptr), state(other.state)
        , requirementTemplates(other.requirementTemplates)
        , declaredRequirementTemplates(other.declaredRequirementTemplates)
        , hasUnclassifiedDeclaration(other.hasUnclassifiedDeclaration), requirementBlock(other.requirementBlock) {}
    ResolverSnapshot& operator=(const ResolverSnapshot& other) {
        if (this != &other) *this = ResolverSnapshot(other);
        return *this;
    }

    const std::shared_ptr<const void>& DependencyIdentity() const noexcept {
        static const std::shared_ptr<const void> none;
        return state ? state->dependencyIdentity : none;
    }
};


// Held open for the length of one pass declaration, on the declaring thread. While it is open, every use, view and
// snapshot the declaration takes of a resolver describes the same resource set
// (RenderGraph::CaptureResolverDeclarationState). This holds even when the resolver publishes on another thread in the
// meantime.
class ResolverCaptureScope {
public:
    ResolverCaptureScope();
    ~ResolverCaptureScope();
    ResolverCaptureScope(const ResolverCaptureScope&) = delete;
    ResolverCaptureScope& operator=(const ResolverCaptureScope&) = delete;
private:
    friend class RenderGraph;
    std::vector<std::shared_ptr<const ResolverDeclarationState>> m_captures;
    ResolverCaptureScope* m_previous = nullptr;
};

} // namespace org
