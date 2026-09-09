#pragma once

#include <algorithm>
#include <type_traits>
#include <memory>
#include <stdexcept>
#include <rhi.h>

#include <spdlog/spdlog.h>

#include "Render/FeatureDomainRegistry.h"
#include "Render/RenderGraph/RenderGraph.h"
#include "ResourceRequirements.h"
#include "Resources/ResourceStateTracker.h"
#include "Resources/ResourceIdentifier.h"
#include "Interfaces/IResourceResolver.h"
#include "Interfaces/IPassBuilder.h"
#include "Interfaces/IResourceResolver.h"


// Tag for a contiguous mip-range [first..first+count)
namespace org {

struct Mip {
	Mip(uint32_t first, uint32_t count) : first(first), count(count) {}
    uint32_t first, count;
};

// Tag for a half-open "from" mip-range [first..inf)
struct FromMip {
    uint32_t first;
};

// Tag for a half-open "up to" mip-range [0..last]
struct UpToMip {
    uint32_t last;
};

// Tag for a contiguous slice-range [first..first+count)
struct Slice {
	Slice(uint32_t first, uint32_t count) : first(first), count(count) {}
    uint32_t first, count;
};

// Tag for a half-open "from" slice-range [first..inf)
struct FromSlice {
    uint32_t first;
};

// Tag for a half-open "up to" slice-range [0..last]
struct UpToSlice {
    uint32_t last;
};

inline ResourcePtrAndRange Subresources(const std::shared_ptr<Resource>& r) {
    return { r }; // full range
}

inline ResourcePtrAndRange Subresources(const std::shared_ptr<Resource>& r, Mip m) {
    RangeSpec spec;
    spec.mipLower = { BoundType::Exact, m.first };
    spec.mipUpper = { BoundType::Exact, m.first + m.count - 1 };
    return { r, spec };
}

inline ResourcePtrAndRange Subresources(const std::shared_ptr<Resource>& r, FromMip fm) {
    RangeSpec spec;
    spec.mipLower = { BoundType::From, fm.first };
    return { r, spec };
}

inline ResourcePtrAndRange Subresources(const std::shared_ptr<Resource>& r, UpToMip um) {
    RangeSpec spec;
    spec.mipUpper = { BoundType::UpTo, um.last };
    return { r, spec };
}

inline ResourcePtrAndRange Subresources(const std::shared_ptr<Resource>& r, Slice s) {
    RangeSpec spec;
    spec.sliceLower = { BoundType::Exact, s.first };
    spec.sliceUpper = { BoundType::Exact, s.first + s.count - 1 };
    return { r, spec };
}

inline ResourcePtrAndRange Subresources(const std::shared_ptr<Resource>& r, FromSlice fs) {
    RangeSpec spec;
    spec.sliceLower = { BoundType::From, fs.first };
    return { r, spec };
}

inline ResourcePtrAndRange Subresources(const std::shared_ptr<Resource>& r, UpToSlice us) {
    RangeSpec spec;
    spec.sliceUpper = { BoundType::UpTo, us.last };
    return { r, spec };
}

inline ResourcePtrAndRange Subresources(const std::shared_ptr<Resource>& r, Mip m, Slice s) {
    RangeSpec spec;
    spec.mipLower = { BoundType::Exact, m.first };
    spec.mipUpper = { BoundType::Exact, m.first + m.count - 1 };
    spec.sliceLower = { BoundType::Exact, s.first };
    spec.sliceUpper = { BoundType::Exact, s.first + s.count - 1 };
    return { r, spec };
}


// ResourceIdentifier
inline ResourceIdentifierAndRange Subresources(const ResourceIdentifier& r) {
    // everything
    return { r };
}

inline ResourceIdentifierAndRange Subresources(const ResourceIdentifier& r,
    Mip m)
{
    RangeSpec spec;
    spec.mipLower   = { BoundType::Exact, m.first      };
    spec.mipUpper   = { BoundType::Exact, m.first + m.count - 1 };
    return { r, spec };
}

inline ResourceIdentifierAndRange Subresources(const ResourceIdentifier& r,
    FromMip fm)
{
    RangeSpec spec;
    spec.mipLower   = { BoundType::From, fm.first };
    return { r, spec };
}

inline ResourceIdentifierAndRange Subresources(const ResourceIdentifier& r,
    UpToMip um)
{
    RangeSpec spec;
    spec.mipUpper   = { BoundType::UpTo, um.last };
    return { r, spec };
}

inline ResourceIdentifierAndRange Subresources(const ResourceIdentifier& r,
    Slice s)
{
    RangeSpec spec;
    spec.sliceLower = { BoundType::Exact, s.first       };
    spec.sliceUpper = { BoundType::Exact, s.first + s.count - 1 };
    return { r, spec };
}

inline ResourceIdentifierAndRange Subresources(const ResourceIdentifier& r,
    FromSlice fs)
{
    RangeSpec spec;
    spec.sliceLower = { BoundType::From, fs.first };
    return { r, spec };
}

inline ResourceIdentifierAndRange Subresources(const ResourceIdentifier& r,
    UpToSlice us)
{
    RangeSpec spec;
    spec.sliceUpper = { BoundType::UpTo, us.last };
    return { r, spec };
}

inline ResourceIdentifierAndRange Subresources(const ResourceIdentifier& r,
    Mip     m,
    Slice   s)
{
    RangeSpec spec;
    spec.mipLower   = { BoundType::Exact, m.first      };
    spec.mipUpper   = { BoundType::Exact, m.first + m.count - 1 };
    spec.sliceLower = { BoundType::Exact, s.first       };
    spec.sliceUpper = { BoundType::Exact, s.first + s.count - 1 };
    return { r, spec };
}

// BuiltinResource
inline ResourceIdentifierAndRange Subresources(const char* r) {
	return Subresources(ResourceIdentifier{ r });
}

inline ResourceIdentifierAndRange Subresources(const char* r,
    Mip m) {
	return Subresources(ResourceIdentifier{ r }, m);
}

inline ResourceIdentifierAndRange Subresources(const char* r,
    FromMip fm) {
	return Subresources(ResourceIdentifier{ r }, fm);
}

inline ResourceIdentifierAndRange Subresources(const char* r,
    UpToMip um) {
	return Subresources(ResourceIdentifier{ r }, um);
}

inline ResourceIdentifierAndRange Subresources(const char* r,
    Slice s) {
	return Subresources(ResourceIdentifier{ r }, s);
}

inline ResourceIdentifierAndRange Subresources(const char* r,
    FromSlice fs) {
	return Subresources(ResourceIdentifier{ r }, fs);
}

inline ResourceIdentifierAndRange Subresources(const char* r,
    UpToSlice us) {
	return Subresources(ResourceIdentifier{ r }, us);
}

inline ResourceIdentifierAndRange Subresources(const char* r,
    Mip     m,
    Slice   s) {
	return Subresources(ResourceIdentifier{ r }, m, s);
}

// If we have a ResourceIdentifierAndRange, ask the builder to resolve it into an actual ResourceAndRange:
std::vector<ResourceHandleAndRange> expandToRanges(ResourceIdentifierAndRange const& rir, RenderGraph* graph);

// If we have an initializer_list of ResourceIdentifierAndRange,
inline std::vector<ResourceHandleAndRange>
expandToRanges(std::initializer_list<ResourceIdentifierAndRange> list,
    RenderGraph* graph)
{
    std::vector<ResourceHandleAndRange> out;
    out.reserve(list.size());
    for (auto const & rir : list) {
        if (auto vec = expandToRanges(rir, graph); !vec.empty()) {
            // vec always has exactly one element, but we push it.
            out.push_back(std::move(vec.front()));
        }
    }
    return out;
}

template<typename> 
constexpr bool is_shared_ptr_v = false;

template<typename U> 
constexpr bool is_shared_ptr_v<std::shared_ptr<U>> = true;

// processResourceArguments(...) is a set of overloads that take one of several 
// ways to represent a resource and return a vector of ResourceHandleAndRange

// For a ResourceHandleAndRange, just return it in a vector
inline std::vector<ResourceHandleAndRange>
processResourceArguments(const ResourceHandleAndRange& rar,
    RenderGraph* graph)
{
    //if (!rar.resource) return {};
    return { rar };
}

// For a resource pointer + range spec, wrap it and expand it to actual resource handles + ranges
inline std::vector<ResourceHandleAndRange>
processResourceArguments(const ResourcePtrAndRange& rar,
    RenderGraph* graph)
{
	auto range = rar.range;
    auto handle = graph->RequestResourceHandle(rar.resource.get());
    return { ResourceHandleAndRange{ handle, range } };
}

// For a resource pointer, assume full range
inline std::vector<ResourceHandleAndRange>
processResourceArguments(const std::shared_ptr<Resource>& r,
    RenderGraph* graph)
{
    return processResourceArguments(
        ResourcePtrAndRange{ r },
        graph
    );
}

// For a resource resolver + range spec, resolve it and process the result
inline std::vector<ResourceHandleAndRange>
processResourceArguments(const ResourceResolverAndRange& rrr,
    RenderGraph* graph)
{
    const auto captured = graph->CaptureResolverDeclarationState(*rrr.pResolver);
    if (captured && captured->resources) {
        const auto handles = graph->RequestResolverResourceHandles(*captured);
        std::vector<ResourceHandleAndRange> out;
        out.reserve(handles->size());
        for (const auto& handle : *handles) out.emplace_back(handle.resource, rrr.range);
        return out;
    }
    auto resources = rrr.pResolver->Resolve();
    std::vector<ResourceHandleAndRange> out;
    out.reserve(resources.size());
    for (const auto& resource : resources) {
        out.push_back(ResourceHandleAndRange{
            graph->RequestResourceHandle(resource.get()), rrr.range });
    }
    return out;
}

// For a resource resolver, assume full range
inline std::vector<ResourceHandleAndRange>
processResourceArguments(const IResourceResolver& resolver,
    RenderGraph* graph)
{
    return processResourceArguments(
        ResourceResolverAndRange(resolver),
        graph
    );
}

// For a resource identifier + range spec, expand it to actual resource handles + ranges
inline std::vector<ResourceHandleAndRange>
processResourceArguments(const ResourceIdentifierAndRange& rir,
    RenderGraph* graph)
{
    // This could be a resolver- ask the graph.
    if (auto resolver = graph->RequestResolver(rir.identifier, true)) {
        const auto captured = graph->CaptureResolverDeclarationState(*resolver);
        auto resources = captured && captured->resources ? *captured->resources : resolver->Resolve();
        // Dynamic-wrapper identifiers are explicit aliases whose concrete resource
        // may change between retained declaration revisions.  Ordinary resolvers
        // must not replace their symbolic entry: some of them intentionally expose
        // a different declaration shape (such as a resource set).
        if (resources.size() == 1u && resources.front()) {
            graph->RegisterResolvedResourceAlias(rir.identifier, resources.front());
        }
        std::vector<ResourceHandleAndRange> resolvedRanges;
        resolvedRanges.reserve(resources.size());
        for (const auto& resource : resources) {
            resolvedRanges.push_back(ResourceHandleAndRange{
                graph->RequestResourceHandle(resource.get()), rir.range });
        }
        return resolvedRanges;
    }

    return expandToRanges(rir, graph);
}

// For a bare resource identifier, assume full range
inline std::vector<ResourceHandleAndRange>
processResourceArguments(const ResourceIdentifier& rid,
    RenderGraph* graph)
{
    return processResourceArguments(
        ResourceIdentifierAndRange{ rid },
        graph
    );
}

// For a builtin resource name, assume full range
inline std::vector<ResourceHandleAndRange>
processResourceArguments(const char* br,
    RenderGraph* graph)
{
    return processResourceArguments(
        ResourceIdentifierAndRange{ ResourceIdentifier{ br } },
        graph
    );
}

// For an initializer_list, process each element individually
template<typename T>
inline std::enable_if_t<
    std::is_same_v<std::decay_t<T>, std::initializer_list<typename std::decay_t<T>::value_type>>,
    std::vector<ResourceHandleAndRange>
>
processResourceArguments(T&& list, RenderGraph* graph)
{
    std::vector<ResourceHandleAndRange> out;
    out.reserve(list.size());

    for (auto const & elem : list) {
        auto vec = processResourceArguments(elem, graph);
        if (!vec.empty()) 
            out.push_back(std::move(vec.front()));
    }
    return out;
}

namespace detail {
    template<typename U>
    inline void extractId(std::unordered_set<ResourceIdentifier, ResourceIdentifier::Hasher>&, std::shared_ptr<U> const&) {
        // Concrete resources are authorized by their handles. identifierSet is
        // reserved for symbolic names and namespaces used through a pass view.
    }
    inline void extractId(auto& out, const ResourcePtrAndRange& rar) {
		extractId(out, rar.resource); // empty identifier
    }
    inline void extractId(auto& out, ResourceIdentifierAndRange const& rir) {
        out.insert(rir.identifier);
    }
    inline void extractId(auto& out, ResourceIdentifier const& rid) {
        out.insert(rid);
    }
    inline void extractId(auto& out, char* br) {
        out.insert(ResourceIdentifier{ br });
    }
    inline void extractId(auto&, const ResourceResolverAndRange&) {}
    inline void extractId(auto&, const ResourceHandleAndRange&) {
    }

    template<typename T>
    inline void extractId(auto& out, std::initializer_list<T> list) {
        for (auto const& e : list) extractId(out, e);
    }

    inline void TrackFeatureDomainActivation(
        std::unordered_set<FeatureDomainIdentifier, FeatureDomainIdentifier::Hasher>& activeDomains,
        const ResourceIdentifier& id)
    {
        auto domain = FeatureDomainRegistry::Get().FindResourceDomain(id);
        if (domain.has_value()) {
            activeDomains.insert(*domain);
        }
    }

    inline void TrackFeatureDomainActivation(
        std::unordered_set<FeatureDomainIdentifier, FeatureDomainIdentifier::Hasher>& activeDomains,
        const ResourceIdentifierAndRange& idAndRange)
    {
        TrackFeatureDomainActivation(activeDomains, idAndRange.identifier);
    }

    inline void TrackFeatureDomainActivation(
        std::unordered_set<FeatureDomainIdentifier, FeatureDomainIdentifier::Hasher>& activeDomains,
        const char* id)
    {
        if (!id) {
            return;
        }
        TrackFeatureDomainActivation(activeDomains, ResourceIdentifier{ id });
    }

    inline void TrackFeatureDomainActivation(
        std::unordered_set<FeatureDomainIdentifier, FeatureDomainIdentifier::Hasher>& activeDomains,
        std::string_view id)
    {
        TrackFeatureDomainActivation(activeDomains, ResourceIdentifier{ id });
    }

    template<class S>
        requires (std::convertible_to<S, std::string_view> && !std::is_same_v<std::remove_cvref_t<S>, const char*> && !std::is_same_v<std::remove_cvref_t<S>, char*>)
    inline void TrackFeatureDomainActivation(
        std::unordered_set<FeatureDomainIdentifier, FeatureDomainIdentifier::Hasher>& activeDomains,
        S&& id)
    {
        TrackFeatureDomainActivation(activeDomains, ResourceIdentifier{ std::string_view{ std::forward<S>(id) } });
    }

    template<typename T>
    inline void TrackFeatureDomainActivation(std::unordered_set<FeatureDomainIdentifier, FeatureDomainIdentifier::Hasher>&, const std::shared_ptr<T>&) {}

    inline void TrackFeatureDomainActivation(std::unordered_set<FeatureDomainIdentifier, FeatureDomainIdentifier::Hasher>&, const ResourcePtrAndRange&) {}
    inline void TrackFeatureDomainActivation(std::unordered_set<FeatureDomainIdentifier, FeatureDomainIdentifier::Hasher>&, const ResourceHandleAndRange&) {}
    inline void TrackFeatureDomainActivation(std::unordered_set<FeatureDomainIdentifier, FeatureDomainIdentifier::Hasher>&, const ResourceResolverAndRange&) {}
    inline void TrackFeatureDomainActivation(std::unordered_set<FeatureDomainIdentifier, FeatureDomainIdentifier::Hasher>&, const IResourceResolver&) {}
}


namespace detail
{
    template<class T>
    struct shared_ptr_pointee { using type = void; };

    template<class U>
    struct shared_ptr_pointee<std::shared_ptr<U>> { using type = U; };

    template<class T>
    using shared_ptr_pointee_t = typename shared_ptr_pointee<std::remove_cvref_t<T>>::type;

    template<class T>
    inline constexpr bool SharedPtrToResource =
        std::derived_from<shared_ptr_pointee_t<T>, Resource>; // false for void

    template<class S>
    concept StringLike =
        std::convertible_to<S, std::string_view>;

        inline bool IsWholeRange(const RangeSpec& range) {
		return range.mipLower == Bound{ BoundType::All, 0 }
			&& range.mipUpper == Bound{ BoundType::All, 0 }
			&& range.sliceLower == Bound{ BoundType::All, 0 }
			&& range.sliceUpper == Bound{ BoundType::All, 0 };
        }

        inline void AppendUniqueDescriptorRegistration(std::vector<AutoDescriptorRegistration>& registrations, const AutoDescriptorRegistration& registration) {
        if (std::find(registrations.begin(), registrations.end(), registration) == registrations.end()) {
            registrations.push_back(registration);
        }
        }

        inline bool TryResolveAutoDescriptorRange(const RangeSpec& range, unsigned int& mip, unsigned int& slice) {
            if (IsWholeRange(range)) {
                mip = 0;
                slice = 0;
                return true;
            }

            const bool exactMip = range.mipLower.type == BoundType::Exact && range.mipUpper.type == BoundType::Exact;
            const bool wholeMip = range.mipLower == Bound{ BoundType::All, 0 } && range.mipUpper == Bound{ BoundType::All, 0 };
            const bool exactSlice = range.sliceLower.type == BoundType::Exact && range.sliceUpper.type == BoundType::Exact;
            const bool wholeSlice = range.sliceLower == Bound{ BoundType::All, 0 } && range.sliceUpper == Bound{ BoundType::All, 0 };

            if (!(wholeMip || (exactMip && range.mipLower.value == range.mipUpper.value))) {
                return false;
            }
            if (!(wholeSlice || (exactSlice && range.sliceLower.value == range.sliceUpper.value))) {
                return false;
            }

            mip = exactMip ? range.mipLower.value : 0;
            slice = exactSlice ? range.sliceLower.value : 0;
            return true;
        }

        inline bool IsResolverBackedIdentifier(RenderGraph* graph, const ResourceIdentifier& id) {
		return graph != nullptr && graph->RequestResolver(id, true) != nullptr;
        }

        inline void TrackDefaultDescriptorIdentifier(RenderGraph* graph, std::vector<AutoDescriptorRegistration>& registrations, const ResourceIdentifier& id, DescriptorType type) {
            if (IsResolverBackedIdentifier(graph, id)) {
				auto resolver = graph->RequestResolver(id, true);
				if (resolver) {
					DescriptorAccessor accessor{};
					accessor.type = type;
					accessor.mip = 0;
					accessor.slice = 0;
					AppendUniqueDescriptorRegistration(registrations,
						AutoDescriptorRegistration{ id, accessor, {}, std::move(resolver) });
				}
                return;
            }

            DescriptorAccessor accessor{};
            accessor.type = type;
            accessor.mip = 0;
            accessor.slice = 0;
            AppendUniqueDescriptorRegistration(registrations, AutoDescriptorRegistration{ id, accessor });
        }

        inline void TrackDefaultDescriptorIdentifier(RenderGraph* graph, std::vector<AutoDescriptorRegistration>& registrations, const ResourceIdentifierAndRange& idAndRange, DescriptorType type) {
            if (IsResolverBackedIdentifier(graph, idAndRange.identifier)) {
                return;
            }

            unsigned int mip = 0;
            unsigned int slice = 0;
            if (!TryResolveAutoDescriptorRange(idAndRange.range, mip, slice)) {
                return;
            }

            DescriptorAccessor accessor{};
            accessor.type = type;
            accessor.mip = mip;
            accessor.slice = slice;
            AppendUniqueDescriptorRegistration(registrations, AutoDescriptorRegistration{ idAndRange.identifier, accessor });
        }

        inline void TrackDefaultDescriptorIdentifier(RenderGraph* graph, std::vector<AutoDescriptorRegistration>& registrations, const char* id, DescriptorType type) {
        TrackDefaultDescriptorIdentifier(graph, registrations, ResourceIdentifier{ id }, type);
        }

        inline void TrackDefaultDescriptorIdentifier(RenderGraph* graph, std::vector<AutoDescriptorRegistration>& registrations, std::string_view id, DescriptorType type) {
        TrackDefaultDescriptorIdentifier(graph, registrations, ResourceIdentifier{ id }, type);
        }

        template<class S>
		requires (StringLike<S> && !std::is_same_v<std::remove_cvref_t<S>, const char*> && !std::is_same_v<std::remove_cvref_t<S>, char*>)
        inline void TrackDefaultDescriptorIdentifier(RenderGraph* graph, std::vector<AutoDescriptorRegistration>& registrations, S&& id, DescriptorType type) {
        TrackDefaultDescriptorIdentifier(graph, registrations, ResourceIdentifier{ std::string_view{ std::forward<S>(id) } }, type);
	}

        template<typename T>
        inline void TrackDefaultDescriptorIdentifier(RenderGraph*, std::vector<AutoDescriptorRegistration>&, const std::shared_ptr<T>&, DescriptorType) {}

        inline void TrackDefaultDescriptorIdentifier(RenderGraph*, std::vector<AutoDescriptorRegistration>&, const ResourcePtrAndRange&, DescriptorType) {}
        inline void TrackDefaultDescriptorIdentifier(RenderGraph*, std::vector<AutoDescriptorRegistration>&, const ResourceHandleAndRange&, DescriptorType) {}
        inline void TrackDefaultDescriptorIdentifier(RenderGraph*, std::vector<AutoDescriptorRegistration>&, const ResourceResolverAndRange&, DescriptorType) {}
        inline void TrackDefaultDescriptorIdentifier(RenderGraph*, std::vector<AutoDescriptorRegistration>&, const IResourceResolver&, DescriptorType) {}
}

template<class S>
    requires (detail::StringLike<S> &&
!std::is_same_v<std::remove_cvref_t<S>, std::string_view>)
inline std::vector<ResourceHandleAndRange>
processResourceArguments(S&& s, RenderGraph* graph)
{
    return processResourceArguments(std::string_view{ std::forward<S>(s) }, graph);
}

inline std::vector<ResourceHandleAndRange>
processResourceArguments(std::string_view name, RenderGraph* graph)
{
    return processResourceArguments(ResourceIdentifier{ name }, graph);
}

template<class T>
inline constexpr bool ResourceLike =
detail::SharedPtrToResource<T> ||
detail::StringLike<T> ||
std::is_same_v<std::remove_cvref_t<T>, ResourceIdentifier> ||
std::is_same_v<std::remove_cvref_t<T>, ResourcePtrAndRange> ||
std::is_same_v<std::remove_cvref_t<T>, ResourceIdentifierAndRange> ||
std::is_same_v<std::remove_cvref_t<T>, ResourceHandleAndRange> ||
std::is_same_v<std::remove_cvref_t<T>, ResourceResolverAndRange> ||
std::is_same_v<std::remove_cvref_t<T>, IResourceResolver>;

template<typename T>
concept NotIResourceResolver = !std::derived_from<std::decay_t<T>, IResourceResolver>; // annoying

template<typename T>
concept DerivedRenderPass = std::derived_from<T, RenderPass>;


namespace detail
{
    template<class...>
    inline constexpr bool dependent_false_v = false;

    inline void TrackResolverSnapshot(std::vector<ResolverSnapshot>& snapshots,
        const IResourceResolver& resolver, const ResolverDeclarationState& state,
        std::optional<ResolverSnapshot::RequirementTemplate> binding = std::nullopt) {
        const auto identity = state.dependencyIdentity.get();
        if (!state.tracked || !identity) return;
        auto found = std::find_if(snapshots.begin(), snapshots.end(), [identity](const auto& snapshot) {
            return snapshot.dependencyIdentity.get() == identity;
        });
        if (found == snapshots.end()) {
            snapshots.emplace_back(resolver.Clone(), state);
            found = std::prev(snapshots.end());
            if (state.resources) {
                found->resourceIDs.reserve(state.resources->size());
                for (const auto& resource : *state.resources)
                    found->resourceIDs.push_back(resource ? resource->GetGlobalResourceID() : 0u);
            }
        }
        if (!binding) {
            found->hasUnclassifiedDeclaration = true;
        } else if (std::none_of(found->declaredRequirementTemplates.begin(), found->declaredRequirementTemplates.end(),
            [&](const auto& existing) { return existing.range == binding->range
                && existing.state == binding->state && existing.state.sync == binding->state.sync; })) {
            found->declaredRequirementTemplates.push_back(*binding);
        }
    }

    template<typename T>
    inline void TrackResolverDeclaration(RenderGraph* graph, std::vector<ResolverSnapshot>& snapshots,
        const T& value, ResourceState required) {
        auto capture = [&](const IResourceResolver& resolver, RangeSpec range) {
            const auto state = graph->CaptureResolverDeclarationState(resolver);
            if (state) TrackResolverSnapshot(snapshots, resolver, *state,
                ResolverSnapshot::RequirementTemplate{range, required});
        };
        using Value = std::remove_cvref_t<T>;
        if constexpr (std::is_same_v<Value, ResourceResolverAndRange>) {
            capture(*value.pResolver, value.range);
        } else if constexpr (std::is_same_v<Value, ResourceIdentifierAndRange>) {
            if (auto resolver = graph->RequestResolver(value.identifier, true)) capture(*resolver, value.range);
        } else if constexpr (std::is_same_v<Value, ResourceIdentifier>) {
            if (auto resolver = graph->RequestResolver(value, true)) capture(*resolver, {});
        } else if constexpr (StringLike<T>) {
            if (auto resolver = graph->RequestResolver(ResourceIdentifier{std::string_view{value}}, true)) capture(*resolver, {});
        }
    }

    template<typename T>
    inline void MaybeTrackResolverSnapshot(RenderGraph*, std::vector<ResolverSnapshot>&, const T&) {
    }

    inline void MaybeTrackResolverSnapshot(RenderGraph* graph, std::vector<ResolverSnapshot>& resolverSnapshots, const ResourceIdentifier& id) {
        if (!graph) {
            return;
        }
        if (auto resolver = graph->RequestResolver(id, true)) {
            const auto state = graph->CaptureResolverDeclarationState(*resolver);
            if (state) TrackResolverSnapshot(resolverSnapshots, *resolver, *state);
        }
    }

    inline void MaybeTrackResolverSnapshot(RenderGraph* graph, std::vector<ResolverSnapshot>& resolverSnapshots, const ResourceIdentifierAndRange& id) {
        MaybeTrackResolverSnapshot(graph, resolverSnapshots, id.identifier);
    }

    inline void MaybeTrackResolverSnapshot(RenderGraph* graph, std::vector<ResolverSnapshot>& resolverSnapshots, const char* id) {
        if (!id) {
            return;
        }
        MaybeTrackResolverSnapshot(graph, resolverSnapshots, ResourceIdentifier{ id });
    }

    inline void MaybeTrackResolverSnapshot(RenderGraph* graph, std::vector<ResolverSnapshot>& resolverSnapshots, std::string_view id) {
        MaybeTrackResolverSnapshot(graph, resolverSnapshots, ResourceIdentifier{ id });
    }

    template<class S>
        requires (StringLike<S> &&
            !std::is_same_v<std::remove_cvref_t<S>, const char*> &&
            !std::is_same_v<std::remove_cvref_t<S>, char*> &&
            !std::is_same_v<std::remove_cvref_t<S>, std::string_view>)
    inline void MaybeTrackResolverSnapshot(RenderGraph* graph, std::vector<ResolverSnapshot>& resolverSnapshots, S&& id) {
        MaybeTrackResolverSnapshot(graph, resolverSnapshots,
            ResourceIdentifier{ std::string_view{ std::forward<S>(id) } });
    }

    template<typename IdSet, typename DestVec, typename T>
    inline void AppendTrackedResource(RenderGraph* graph, IdSet& ids, DestVec& dest, T&& value) {
        extractId(ids, std::forward<T>(value));
        auto ranges = processResourceArguments(std::forward<T>(value), graph);
        dest.insert(dest.end(), std::make_move_iterator(ranges.begin()), std::make_move_iterator(ranges.end()));
    }

    template<typename IdSet, typename DestVec, typename Range>
    inline void AppendTrackedResourceRange(RenderGraph* graph, IdSet& ids, DestVec& dest, Range&& values) {
		if constexpr (std::is_same_v<std::remove_cv_t<std::ranges::range_value_t<Range>>, ResourceHandleAndRange>) {
			if constexpr (requires { values.size(); }) dest.reserve(dest.size() + values.size());
			for (const auto& value : values) dest.push_back(value);
		}
		else {
			for (auto&& value : values) {
				AppendTrackedResource(graph, ids, dest, std::forward<decltype(value)>(value));
			}
        }
    }

    template<typename IdSet, typename TransitionVec, typename T>
    inline void AppendInternalTransition(RenderGraph* graph, IdSet& ids, TransitionVec& transitions, T&& value, ResourceState exitState) {
        extractId(ids, std::forward<T>(value));
        auto ranges = processResourceArguments(std::forward<T>(value), graph);
        for (auto& range : ranges) {
            transitions.emplace_back(range, exitState);
        }
    }

    template<typename SyncFunction, typename... Sources>
    inline std::vector<ResourceRequirement> BuildRequirements(SyncFunction&& syncFunction, Sources&&... sources) {
        std::vector<std::pair<ResourceHandleAndRange, rhi::ResourceAccessType>> entries;
        const size_t sourceEntryCount = (std::get<0>(sources).get().size() + ... + 0ull);
        entries.reserve(sourceEntryCount);
		bool canUseUniqueResourceFastPath = true;
		constexpr size_t linearUniqueCheckLimit = 16;
		std::unordered_set<uint64_t> uniqueResourceIDs;

        auto append = [&](auto&& src) {
            auto const& list = std::get<0>(src).get();
            auto access = std::get<1>(src);
            for (auto const& rr : list) {
				if (access == rhi::ResourceAccessType::Common) {
					canUseUniqueResourceFastPath = false;
				}
				else if (canUseUniqueResourceFastPath) {
					const uint64_t resourceID = rr.resource.GetGlobalResourceID();
					if (entries.size() < linearUniqueCheckLimit) {
						canUseUniqueResourceFastPath = std::none_of(
							entries.begin(), entries.end(),
							[resourceID](const auto& entry) {
								return entry.first.resource.GetGlobalResourceID() == resourceID;
							});
					}
					else {
						if (uniqueResourceIDs.empty()) {
							uniqueResourceIDs.reserve(sourceEntryCount);
							for (const auto& entry : entries) {
								uniqueResourceIDs.insert(entry.first.resource.GetGlobalResourceID());
							}
						}
						canUseUniqueResourceFastPath = uniqueResourceIDs.insert(resourceID).second;
					}
				}
                entries.emplace_back(rr, access);
            }
        };

        (append(std::forward<Sources>(sources)), ...);

		// Dynamic upload/readback declarations overwhelmingly contain one range per
		// resource.  The symbolic tracker is only needed when multiple declarations
		// for the same resource must be merged.  Avoid two hash tables, one tracker
		// per resource, and temporary transition vectors for the unique case.
		if (canUseUniqueResourceFastPath) {
			std::vector<ResourceRequirement> out;
			out.reserve(entries.size());
			for (const auto& [resourceAndRange, access] : entries) {
				ResourceRequirement requirement(resourceAndRange);
				requirement.state = ResourceState{
					access,
					AccessToLayout(access, /*directQueue=*/true),
					syncFunction(access)
				};
				out.push_back(std::move(requirement));
			}
			return out;
		}

        constexpr ResourceState initialState{
            rhi::ResourceAccessType::Common,
            rhi::ResourceLayout::Common,
            rhi::ResourceSyncState::All
        };

        std::unordered_map<uint64_t, SymbolicTracker> trackers;
        std::unordered_map<uint64_t, ResourceRegistry::RegistryHandle> handleMap;

        for (auto& [rar, access] : entries) {
            const uint64_t id = rar.resource.GetGlobalResourceID();
            handleMap[id] = rar.resource;

            auto [it, _] = trackers.try_emplace(id, RangeSpec{}, initialState);
            auto& tracker = it->second;

            ResourceState want{
                access,
                AccessToLayout(access, /*directQueue=*/true),
                syncFunction(access)
            };

            std::vector<ResourceTransition> dummy;
            tracker.Apply(rar.range, nullptr, want, dummy);
        }

        std::vector<ResourceRequirement> out;
        out.reserve(entries.size());

        for (auto& [id, tracker] : trackers) {
            auto pRes = handleMap[id];
            for (auto const& seg : tracker.Flatten(initialState)) {
                ResourceHandleAndRange rr(pRes);
                rr.range = seg.rangeSpec;

                ResourceRequirement req(rr);
                req.state = seg.state;
                out.push_back(std::move(req));
            }
        }

        for (auto& [rar, access] : entries) {
            if (access != rhi::ResourceAccessType::Common) {
                continue;
            }

            ResourceRequirement req(rar);
            req.state = ResourceState{
                access,
                AccessToLayout(access, /*directQueue=*/true),
                syncFunction(access)
            };
            out.push_back(std::move(req));
        }

        return out;
    }

    // Prefer (Inputs, StableArgs...) if available, else (StableArgs...), else default ctor.
    template<class PassT, class InputsT, class... StableArgs>
    std::shared_ptr<PassT> MakePass(InputsT&& inputs, StableArgs&&... stableArgs)
    {
        using In = std::remove_cvref_t<InputsT>;

        // Perfect-forwarded inputs (supports PassT(Inputs&&) etc.)
        if constexpr (std::constructible_from<PassT, InputsT, StableArgs...>)
        {
            return std::make_shared<PassT>(
                std::forward<InputsT>(inputs),
                std::forward<StableArgs>(stableArgs)...);
        }
        // Common case: PassT(const Inputs&) (also binds rvalues)
        else if constexpr (std::constructible_from<PassT, const In&, StableArgs...>)
        {
            return std::make_shared<PassT>(
                static_cast<const In&>(inputs),
                std::forward<StableArgs>(stableArgs)...);
        }
        // No inputs-ctor: try stable args only
        else if constexpr (std::constructible_from<PassT, StableArgs...>)
        {
            return std::make_shared<PassT>(std::forward<StableArgs>(stableArgs)...);
        }
        // Finally, default ctor
        else if constexpr (std::default_initializable<PassT>)
        {
            return std::make_shared<PassT>();
        }
        else
        {
            static_assert(dependent_false_v<PassT>,
                "PassT is not constructible with (Inputs[, StableArgs...]), (StableArgs...), or default ctor.");
        }
    }
}

class RenderPassBuilder : public IPassBuilder {
public:
    PassBuilderKind Kind() const noexcept override { return PassBuilderKind::Render; }
    IResourceProvider* ResourceProvider() noexcept override { return pass.get(); }
    // Typed declaration entry points return a stable token for Prepare. This
    // keeps the familiar fluent With* API intact while avoiding registry
    // lookups and global-ID plumbing in typed passes.
    template<class ResourceT>
        requires std::derived_from<ResourceT, Resource>
    ResourceBindingToken BindShaderResource(const std::shared_ptr<ResourceT>& resource) {
        if (!resource) throw std::invalid_argument("Cannot bind an empty shader resource");
        addShaderResource(resource);
        return { resource->GetSchedulingResourceID(), graph->RequestResourceHandle(resource.get()).GetGlobalResourceID() };
    }

    ResourceBindingToken BindShaderResource(const ResourceIdentifierAndRange& resource) {
        addShaderResource(resource);
        const auto handle = graph->RequestResourceHandle(resource.identifier);
        return {handle.GetGlobalResourceID(), handle.GetGlobalResourceID()};
    }

    ResourceBindingToken BindUnorderedAccessClear(const ResourceIdentifier& identifier) {
        addUnorderedAccessClear(identifier);
        const auto handle = graph->RequestResourceHandle(identifier);
        return {handle.GetGlobalResourceID(), handle.GetGlobalResourceID()};
    }

    ResourceBindingToken BindDepthStencilClear(const ResourceIdentifier& identifier) {
        addDepthStencilClear(identifier);
        const auto handle = graph->RequestResourceHandle(identifier);
        return {handle.GetGlobalResourceID(), handle.GetGlobalResourceID()};
    }

    template<class ResourceT>
        requires std::derived_from<ResourceT, Resource>
    ResourceBindingToken BindUnorderedAccess(const std::shared_ptr<ResourceT>& resource) {
        if (!resource) throw std::invalid_argument("Cannot bind an empty unordered-access resource");
        addUnorderedAccess(resource);
        return { resource->GetSchedulingResourceID(), graph->RequestResourceHandle(resource.get()).GetGlobalResourceID() };
    }

    template<class ResourceT>
        requires std::derived_from<ResourceT, Resource>
    ResourceBindingToken BindUnorderedAccessClear(const std::shared_ptr<ResourceT>& resource) {
        if (!resource) throw std::invalid_argument("Cannot bind an empty unordered-access clear resource");
        addUnorderedAccessClear(resource);
        return { resource->GetSchedulingResourceID(), graph->RequestResourceHandle(resource.get()).GetGlobalResourceID() };
    }

    template<class ResourceT>
        requires std::derived_from<ResourceT, Resource>
    ResourceBindingToken BindIndirectArguments(const std::shared_ptr<ResourceT>& resource) {
        if (!resource) throw std::invalid_argument("Cannot bind an empty indirect-argument resource");
        addIndirectArguments(resource);
        return { resource->GetSchedulingResourceID(), graph->RequestResourceHandle(resource.get()).GetGlobalResourceID() };
    }

    template<class ResourceT>
        requires std::derived_from<ResourceT, Resource>
    ResourceBindingToken BindDepthReadWrite(const std::shared_ptr<ResourceT>& resource) {
        if (!resource) throw std::invalid_argument("Cannot bind an empty depth resource");
        addDepthReadWrite(resource);
        return { resource->GetSchedulingResourceID(), graph->RequestResourceHandle(resource.get()).GetGlobalResourceID() };
    }

    template<class ResourceT>
        requires std::derived_from<ResourceT, Resource>
    ResourceBindingToken BindRenderTarget(const std::shared_ptr<ResourceT>& resource) {
        if (!resource) throw std::invalid_argument("Cannot bind an empty render target");
        addRenderTarget(resource);
        return { resource->GetSchedulingResourceID(), graph->RequestResourceHandle(resource.get()).GetGlobalResourceID() };
    }

    template<class ResourceT>
        requires std::derived_from<ResourceT, Resource>
    ResourceBindingToken BindRenderTargetClear(const std::shared_ptr<ResourceT>& resource) {
        if (!resource) throw std::invalid_argument("Cannot bind an empty render-target clear resource");
        addRenderTargetClear(resource);
        return { resource->GetSchedulingResourceID(), graph->RequestResourceHandle(resource.get()).GetGlobalResourceID() };
    }

    ResourceBindingToken BindRenderTarget(const ResourceIdentifier& identifier) {
        addRenderTarget(identifier);
        const auto handle = graph->RequestResourceHandle(identifier);
        return {handle.GetGlobalResourceID(), handle.GetGlobalResourceID()};
    }

    ResourceBindingToken BindRenderTarget(const ResourceIdentifierAndRange& resource) {
        addRenderTarget(resource);
        const auto handle = graph->RequestResourceHandle(resource.identifier);
        return {handle.GetGlobalResourceID(), handle.GetGlobalResourceID()};
    }

    template<class ResourceT>
        requires std::derived_from<ResourceT, Resource>
    ResourceBindingToken BindCopySource(const std::shared_ptr<ResourceT>& resource) {
        if (!resource) throw std::invalid_argument("Cannot bind an empty copy-source resource");
        addCopySource(resource);
        return { resource->GetSchedulingResourceID(), graph->RequestResourceHandle(resource.get()).GetGlobalResourceID() };
    }

    template<class ResourceT>
        requires std::derived_from<ResourceT, Resource>
    ResourceBindingToken BindCopyDestination(const std::shared_ptr<ResourceT>& resource) {
        if (!resource) throw std::invalid_argument("Cannot bind an empty copy-destination resource");
        addCopyDest(resource);
        return { resource->GetSchedulingResourceID(), graph->RequestResourceHandle(resource.get()).GetGlobalResourceID() };
    }

    // Variadic entry points

    //First set, callable on Lvalues
    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder& WithShaderResource(Args&&... args) & {
        (addShaderResource(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder& WithRenderTarget(Args&&... args) & {
        (addRenderTarget(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder& WithRenderTargetClear(Args&&... args) & {
        (addRenderTargetClear(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder& WithPresent(Args&&... args) & {
        (addPresent(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder& WithDepthRead(Args&&... args) & {
        (addDepthRead(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder& WithDepthReadWrite(Args&&... args) & {
        (addDepthReadWrite(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder& WithDepthStencilClear(Args&&... args) & {
        (addDepthStencilClear(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder& WithConstantBuffer(Args&&... args) & {
        (addConstantBuffer(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder& WithUnorderedAccess(Args&&... args) & {
        (addUnorderedAccess(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder& WithUnorderedAccessClear(Args&&... args) & {
        (addUnorderedAccessClear(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder& WithCopyDest(Args&&... args) & {
        (addCopyDest(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder& WithCopySource(Args&&... args) & {
        (addCopySource(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder& WithIndirectArguments(Args&&... args) & {
        (addIndirectArguments(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder& WithIndexBuffer(Args&&... args) & {
        (addIndexBuffer(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder& WithLegacyInterop(Args&&... args)& {
        (addLegacyInterop(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename T>
        requires ResourceLike<T>
    RenderPassBuilder& WithInternalTransition(T&& resource, ResourceState exitState)& {
        addInternalTransition(std::forward<T>(resource), exitState);
        return *this;
    }

    template<typename... Args>
    RenderPassBuilder& WithActiveFeatureDomain(Args&&... args) & {
        (addActiveFeatureDomain(std::forward<Args>(args)), ...);
        return *this;
    }

    // Second set, callable on temporaries
    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder WithShaderResource(Args&&... args) && {
        (addShaderResource(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder WithRenderTarget(Args&&... args) && {
        (addRenderTarget(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder WithRenderTargetClear(Args&&... args) && {
        (addRenderTargetClear(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder WithPresent(Args&&... args) && {
        (addPresent(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder WithDepthReadWrite(Args&&... args) && {
        (addDepthReadWrite(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder WithDepthStencilClear(Args&&... args) && {
        (addDepthStencilClear(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder WithDepthRead(Args&&... args) && {
        (addDepthRead(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder WithConstantBuffer(Args&&... args) && {
        (addConstantBuffer(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder WithUnorderedAccess(Args&&... args) && {
        (addUnorderedAccess(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder WithUnorderedAccessClear(Args&&... args) && {
        (addUnorderedAccessClear(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
    RenderPassBuilder WithCopyDest(Args&&... args) && {
        (addCopyDest(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder WithCopySource(Args&&... args) && {
        (addCopySource(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder WithIndirectArguments(Args&&... args) && {
        (addIndirectArguments(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder WithIndexBuffer(Args&&... args) && {
        (addIndexBuffer(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    RenderPassBuilder WithLegacyInterop(Args&&... args)&& {
        (addLegacyInterop(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename T>
        requires ResourceLike<T>
    RenderPassBuilder WithInternalTransition(T&& resource, ResourceState exitState)&& {
        addInternalTransition(std::forward<T>(resource), exitState);
        return std::move(*this);
    }

    template<typename... Args>
    RenderPassBuilder WithActiveFeatureDomain(Args&&... args) && {
        (addActiveFeatureDomain(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename AddCallable>
    RenderPassBuilder& WithResolverDeclaration(const IResourceResolver& resolver, AddCallable&& addCallable) & {
        const auto state = graph->CaptureResolverDeclarationState(resolver);
        if (state && state->resources) {
            addCallable(ResourceResolverAndRange{resolver});
        } else {
            const auto resources = resolver.Resolve();
            for (const auto& resource : resources) graph->AddResource(resource);
            addCallable(resources);
        }
        return *this;
    }

    template<typename AddCallable>
    RenderPassBuilder WithResolverDeclaration(const IResourceResolver& resolver, AddCallable&& addCallable) && {
        const auto state = graph->CaptureResolverDeclarationState(resolver);
        if (state && state->resources) {
            addCallable(ResourceResolverAndRange{resolver});
        } else {
            const auto resources = resolver.Resolve();
            for (const auto& resource : resources) graph->AddResource(resource);
            addCallable(resources);
        }
        return std::move(*this);
    }

    std::vector<ResolverSnapshot> TakeResolverSnapshots() { return std::move(resolverSnapshots_); }
    void AdoptResolverSnapshots(std::vector<ResolverSnapshot> snapshots) {
        resolverSnapshots_ = std::move(snapshots);
    }

	// LVALUE overloads for IResourceResolver
    RenderPassBuilder& WithShaderResource(const IResourceResolver& r)& {
        return WithResolverDeclaration(r, [&](auto&& resolved) { addShaderResource(std::forward<decltype(resolved)>(resolved)); });
    }

    RenderPassBuilder& WithRenderTarget(const IResourceResolver& r)& {
        return WithResolverDeclaration(r, [&](auto&& resolved) { addRenderTarget(std::forward<decltype(resolved)>(resolved)); });
	}

    RenderPassBuilder& WithRenderTargetClear(const IResourceResolver& r)& {
        return WithResolverDeclaration(r, [&](auto&& resolved) { addRenderTargetClear(std::forward<decltype(resolved)>(resolved)); });
    }

    RenderPassBuilder& WithPresent(const IResourceResolver& r)& {
        return WithResolverDeclaration(r, [&](auto&& resolved) { addPresent(std::forward<decltype(resolved)>(resolved)); });
    }

    RenderPassBuilder& WithDepthReadWrite(const IResourceResolver& r)& {
		return WithResolverDeclaration(r, [&](auto&& resolved) { addDepthReadWrite(std::forward<decltype(resolved)>(resolved)); });
    }

        RenderPassBuilder& WithDepthStencilClear(const IResourceResolver& r)& {
		return WithResolverDeclaration(r, [&](auto&& resolved) { addDepthStencilClear(std::forward<decltype(resolved)>(resolved)); });
        }

    RenderPassBuilder& WithDepthRead(const IResourceResolver& r)& {
		return WithResolverDeclaration(r, [&](auto&& resolved) { addDepthRead(std::forward<decltype(resolved)>(resolved)); });
    }

	RenderPassBuilder& WithConstantBuffer(const IResourceResolver& r)& {
        return WithResolverDeclaration(r, [&](auto&& resolved) { addConstantBuffer(std::forward<decltype(resolved)>(resolved)); });
	}

    RenderPassBuilder& WithUnorderedAccess(const IResourceResolver& r)& {
        return WithResolverDeclaration(r, [&](auto&& resolved) { addUnorderedAccess(std::forward<decltype(resolved)>(resolved)); });
	}

    RenderPassBuilder& WithUnorderedAccessClear(const IResourceResolver& r)& {
        return WithResolverDeclaration(r, [&](auto&& resolved) { addUnorderedAccessClear(std::forward<decltype(resolved)>(resolved)); });
	}

    RenderPassBuilder& WithCopyDest(const IResourceResolver& r)& {
		return WithResolverDeclaration(r, [&](auto&& resolved) { addCopyDest(std::forward<decltype(resolved)>(resolved)); });
    }

    RenderPassBuilder& WithCopySource(const IResourceResolver& r)& {
		return WithResolverDeclaration(r, [&](auto&& resolved) { addCopySource(std::forward<decltype(resolved)>(resolved)); });
    }

    RenderPassBuilder& WithIndirectArguments(const IResourceResolver& r)& {
        return WithResolverDeclaration(r, [&](auto&& resolved) { addIndirectArguments(std::forward<decltype(resolved)>(resolved)); });
	}

    RenderPassBuilder& WithLegacyInterop(const IResourceResolver& r)& {
        return WithResolverDeclaration(r, [&](auto&& resolved) { addLegacyInterop(std::forward<decltype(resolved)>(resolved)); });
	}

	// RVALUE overloads for IResourceResolver

    RenderPassBuilder WithShaderResource(const IResourceResolver& r)&& {
        return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addShaderResource(std::forward<decltype(resolved)>(resolved)); });
	}
    RenderPassBuilder WithRenderTarget(const IResourceResolver& r)&& {
		return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addRenderTarget(std::forward<decltype(resolved)>(resolved)); });
    }
        RenderPassBuilder WithRenderTargetClear(const IResourceResolver& r)&& {
                return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addRenderTargetClear(std::forward<decltype(resolved)>(resolved)); });
        }
    RenderPassBuilder WithPresent(const IResourceResolver& r)&& {
        return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addPresent(std::forward<decltype(resolved)>(resolved)); });
    }
    RenderPassBuilder WithDepthReadWrite(const IResourceResolver& r)&& {
		return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addDepthReadWrite(std::forward<decltype(resolved)>(resolved)); });
    }

        RenderPassBuilder WithDepthStencilClear(const IResourceResolver& r)&& {
		return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addDepthStencilClear(std::forward<decltype(resolved)>(resolved)); });
        }
    RenderPassBuilder WithDepthRead(const IResourceResolver& r)&& {
		return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addDepthRead(std::forward<decltype(resolved)>(resolved)); });
    }
    RenderPassBuilder WithConstantBuffer(const IResourceResolver& r)&& {
		return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addConstantBuffer(std::forward<decltype(resolved)>(resolved)); });
    }
    RenderPassBuilder WithUnorderedAccess(const IResourceResolver& r)&& {
		return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addUnorderedAccess(std::forward<decltype(resolved)>(resolved)); });
    }
        RenderPassBuilder WithUnorderedAccessClear(const IResourceResolver& r)&& {
		return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addUnorderedAccessClear(std::forward<decltype(resolved)>(resolved)); });
        }
    RenderPassBuilder WithCopyDest(const IResourceResolver& r)&& {
		return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addCopyDest(std::forward<decltype(resolved)>(resolved)); });
    }
    RenderPassBuilder WithCopySource(const IResourceResolver& r)&& {
		return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addCopySource(std::forward<decltype(resolved)>(resolved)); });
    }
    RenderPassBuilder WithIndirectArguments(const IResourceResolver& r)&& {
		return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addIndirectArguments(std::forward<decltype(resolved)>(resolved)); });
    }
    RenderPassBuilder WithLegacyInterop(const IResourceResolver& r)&& {
        return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addLegacyInterop(std::forward<decltype(resolved)>(resolved)); });
	}

	RenderPassBuilder& IsGeometryPass()& {
		m_isGeometryPass = true;
		return *this;
	}

    RenderPassBuilder& PreferQueue(QueueKind kind)& {
        if (!IsQueueKindSupportedByRenderPass(kind)) {
            throw std::invalid_argument("Render passes only support the graphics queue");
        }
        m_preferredQueueKind = kind;
        m_queueAssignmentPolicy = QueueAssignmentPolicy::ForcePreferred;
        return *this;
    }

    RenderPassBuilder IsGeometryPass() && {
        m_isGeometryPass = true;
		return std::move(*this);
    }

    RenderPassBuilder PreferQueue(QueueKind kind) && {
        if (!IsQueueKindSupportedByRenderPass(kind)) {
            throw std::invalid_argument("Render passes only support the graphics queue");
        }
        m_preferredQueueKind = kind;
        m_queueAssignmentPolicy = QueueAssignmentPolicy::ForcePreferred;
		return std::move(*this);
	}

	RenderPassBuilder& AutomaticQueueAssignment() & {
		m_queueAssignmentPolicy = QueueAssignmentPolicy::Automatic;
		return *this;
	}

	RenderPassBuilder AutomaticQueueAssignment() && {
		m_queueAssignmentPolicy = QueueAssignmentPolicy::Automatic;
        return std::move(*this);
    }

    RenderPassBuilder& PinToQueue(QueueSlotIndex slot) & {
        m_pinnedQueueSlot = slot;
        return *this;
    }

    RenderPassBuilder PinToQueue(QueueSlotIndex slot) && {
        m_pinnedQueueSlot = slot;
        return std::move(*this);
    }

	RenderPassBuilder& RequireBackend(rhi::Backend backend) & { m_backendAffinity = { BackendAffinityStrength::Required, backend }; return *this; }
	RenderPassBuilder RequireBackend(rhi::Backend backend) && { m_backendAffinity = { BackendAffinityStrength::Required, backend }; return std::move(*this); }
	RenderPassBuilder& PreferBackend(rhi::Backend backend) & { m_backendAffinity = { BackendAffinityStrength::Preferred, backend }; return *this; }
	RenderPassBuilder PreferBackend(rhi::Backend backend) && { m_backendAffinity = { BackendAffinityStrength::Preferred, backend }; return std::move(*this); }
	RenderPassBuilder& RequireAPI(rhi::Backend backend) & { return RequireBackend(backend); }
	RenderPassBuilder RequireAPI(rhi::Backend backend) && { return std::move(*this).RequireBackend(backend); }
	RenderPassBuilder& PreferAPI(rhi::Backend backend) & { return PreferBackend(backend); }
	RenderPassBuilder PreferAPI(rhi::Backend backend) && { return std::move(*this).PreferBackend(backend); }
	RenderPassBuilder& RequireDevice(DeviceInstanceId device) & { m_backendAffinity = { BackendAffinityStrength::Required, rhi::Backend::Null, device }; return *this; }
	RenderPassBuilder RequireDevice(DeviceInstanceId device) && { m_backendAffinity = { BackendAffinityStrength::Required, rhi::Backend::Null, device }; return std::move(*this); }
	RenderPassBuilder& PreferDevice(DeviceInstanceId device) & { m_backendAffinity = { BackendAffinityStrength::Preferred, rhi::Backend::Null, device }; return *this; }
	RenderPassBuilder PreferDevice(DeviceInstanceId device) && { m_backendAffinity = { BackendAffinityStrength::Preferred, rhi::Backend::Null, device }; return std::move(*this); }

	RenderPassBuilder& WithExternalWaitBeforeTransitions(rhi::Timeline timeline, uint64_t value) & {
		params.externalWaitsBeforeTransitions.push_back({ timeline, value });
		return *this;
	}

	RenderPassBuilder WithExternalWaitBeforeTransitions(rhi::Timeline timeline, uint64_t value) && {
		params.externalWaitsBeforeTransitions.push_back({ timeline, value });
		return std::move(*this);
	}
	RenderPassBuilder& WithExternalWaitBindingBeforeTransitions(ExternalTimelineBinding binding) & { params.externalWaitBindingsBeforeTransitions.push_back(binding); return *this; }
	RenderPassBuilder WithExternalWaitBindingBeforeTransitions(ExternalTimelineBinding binding) && { params.externalWaitBindingsBeforeTransitions.push_back(binding); return std::move(*this); }

    auto const& DeclaredResourceIds() const { return _declaredIds; }

private:
    struct AuthorState {
        RenderPassParameters params;
        std::unordered_set<ResourceIdentifier, ResourceIdentifier::Hasher> declaredIds;
        std::vector<ResolverSnapshot> resolverSnapshots;
    };

    RenderPassBuilder(RenderGraph* g, std::string name)
        : graph(g), passName(std::move(name)) {}

	// Copy and move constructors are default, but private
	RenderPassBuilder(const RenderPassBuilder&) = default;
	RenderPassBuilder(RenderPassBuilder&&) = default;

    // Same for assignment
	RenderPassBuilder& operator=(const RenderPassBuilder&) = default;
	RenderPassBuilder& operator=(RenderPassBuilder&&) = default;

    AuthorState CaptureAuthorState() && {
        return {
            std::move(params),
            std::move(_declaredIds),
            std::move(resolverSnapshots_)
        };
    }

    void RestoreAuthorState(AuthorState&& state) {
        params = std::move(state.params);
        _declaredIds = std::move(state.declaredIds);
        resolverSnapshots_ = std::move(state.resolverSnapshots);
    }

    template<DerivedRenderPass PassT, org::PassInputs InputsT, typename... StableCtorArgs>
    void Instantiate(InputsT&& inputs, StableCtorArgs&&... ctorArgs)
    {
        if (!built_)
        {
            built_ = true;
            pass = detail::MakePass<PassT>(
                std::forward<InputsT>(inputs),
                std::forward<StableCtorArgs>(ctorArgs)...);
        }

        pass->SetInputs(std::forward<InputsT>(inputs));
    }

    template<DerivedRenderPass PassT, typename... StableCtorArgs>
    void Instantiate(StableCtorArgs&&... ctorArgs)
    {
        Instantiate<PassT>(org::NoInputs{}, std::forward<StableCtorArgs>(ctorArgs)...);
    }

    void Finalize() override {
        if (!built_) return;

        auto authorState = std::move(*this).CaptureAuthorState();

        params = {}; // Rebuild pass-declared state from scratch each finalize.
        _declaredIds.clear();
        resolverSnapshots_.clear();

        RestoreAuthorState(std::move(authorState));

        pass->DeclareResourceUsages(this);

        params.isGeometryPass = m_isGeometryPass;
        params.preferredQueueKind = m_preferredQueueKind;
        params.queueAssignmentPolicy = m_queueAssignmentPolicy;
        params.pinnedQueueSlot = m_pinnedQueueSlot;
		params.backendAffinity = m_backendAffinity;
        params.identifierSet = _declaredIds;
        params.staticResourceRequirements = GatherResourceRequirements();

		const bool hasGraphicsOnlyOperations = !params.renderTargets.empty()
			|| !params.renderTargetClearResources.empty()
			|| !params.depthReadResources.empty()
			|| !params.depthReadWriteResources.empty()
			|| !params.depthStencilClearResources.empty()
			|| !params.indexBuffers.empty()
			|| !params.presentResources.empty();
		const bool hasShaderOperations = !params.shaderResources.empty()
			|| !params.constantBuffers.empty()
			|| !params.unorderedAccessViews.empty()
			|| !params.unorderedAccessClearViews.empty()
			|| !params.indirectArgumentBuffers.empty();
		if (params.preferredQueueKind == QueueKind::Compute && hasGraphicsOnlyOperations)
			throw std::invalid_argument("Pass declares graphics-only operations but prefers the compute queue");
		if (params.preferredQueueKind == QueueKind::Copy && (hasGraphicsOnlyOperations || hasShaderOperations))
			throw std::invalid_argument("Pass declares non-copy operations but prefers the copy queue");

        graph->AddRenderPass(pass, params, passName, TakeResolverSnapshots());
    }

    void Reset() override {
        built_ = false;
        pass = nullptr;
        params = {};
        _declaredIds.clear();
        resolverSnapshots_.clear();
        m_isGeometryPass = false;
		m_preferredQueueKind = QueueKind::Graphics;
		m_queueAssignmentPolicy = QueueAssignmentPolicy::ForcePreferred;
        m_pinnedQueueSlot = std::nullopt;
		m_backendAffinity = {};
	}

    // Shader Resource
	template<typename T>
        requires ResourceLike<T>
	RenderPassBuilder& addShaderResource(T&& x) {
    detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::ShaderResource, AccessToLayout(rhi::ResourceAccessType::ShaderResource, true), RenderSyncFromAccess(rhi::ResourceAccessType::ShaderResource)});
    detail::TrackDefaultDescriptorIdentifier(graph, params.autoDescriptorShaderResources, x, DescriptorType::SRV);
        detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.shaderResources, std::forward<T>(x));
		return *this;
	}
    template<class Range>
        requires (std::ranges::range<Range>&&
    ResourceLike<std::ranges::range_value_t<Range>>)
        RenderPassBuilder& addShaderResource(Range&& xs) {
        for (auto&& e : xs) {
            addShaderResource(std::forward<decltype(e)>(e));
        }
        return *this;
    }

    // Render target
    template<typename T>
        requires ResourceLike<T>
    RenderPassBuilder& addRenderTarget(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::RenderTarget, AccessToLayout(rhi::ResourceAccessType::RenderTarget, true), RenderSyncFromAccess(rhi::ResourceAccessType::RenderTarget)});
        detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.renderTargets, std::forward<T>(x));
		return *this;
    }
    template<class Range>
		requires (std::ranges::range<Range>&&
	ResourceLike<std::ranges::range_value_t<Range>>)
		RenderPassBuilder& addRenderTarget(Range&& xs) {
		for (auto&& e : xs) {
			addRenderTarget(std::forward<decltype(e)>(e));
		}
		return *this;
	}

    template<typename T>
        requires ResourceLike<T>
    RenderPassBuilder& addRenderTargetClear(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::RenderTargetClear, AccessToLayout(rhi::ResourceAccessType::RenderTargetClear, true), RenderSyncFromAccess(rhi::ResourceAccessType::RenderTargetClear)});
        detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.renderTargetClearResources, std::forward<T>(x));
        return *this;
    }
    template<class Range>
        requires (std::ranges::range<Range>&&
    ResourceLike<std::ranges::range_value_t<Range>>)
        RenderPassBuilder& addRenderTargetClear(Range&& xs) {
        for (auto&& e : xs) {
            addRenderTargetClear(std::forward<decltype(e)>(e));
        }
        return *this;
    }

    // Depth target
	template<typename T>
        requires ResourceLike<T>
	RenderPassBuilder& addDepthReadWrite(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::DepthReadWrite, AccessToLayout(rhi::ResourceAccessType::DepthReadWrite, true), RenderSyncFromAccess(rhi::ResourceAccessType::DepthReadWrite)});
    detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.depthReadWriteResources, std::forward<T>(x));
		return *this;
	}
	template<class Range>
		requires (std::ranges::range<Range>&&
	ResourceLike<std::ranges::range_value_t<Range>>)
        RenderPassBuilder& addDepthReadWrite(Range&& xs) {
        for (auto&& e : xs) {
            addDepthReadWrite(std::forward<decltype(e)>(e));
        }
        return *this;
	}

	template<typename T>
    requires ResourceLike<T>
	RenderPassBuilder& addDepthStencilClear(T&& x) {
    detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::DepthStencilClear, AccessToLayout(rhi::ResourceAccessType::DepthStencilClear, true), RenderSyncFromAccess(rhi::ResourceAccessType::DepthStencilClear)});
    detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
    detail::AppendTrackedResource(graph, _declaredIds, params.depthStencilClearResources, std::forward<T>(x));
		return *this;
	}
	template<class Range>
		requires (std::ranges::range<Range>&&
	ResourceLike<std::ranges::range_value_t<Range>>)
    RenderPassBuilder& addDepthStencilClear(Range&& xs) {
    for (auto&& e : xs) {
        addDepthStencilClear(std::forward<decltype(e)>(e));
    }
    return *this;
	}

	template<typename T>
        requires ResourceLike<T>
	RenderPassBuilder& addDepthRead(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::DepthRead, AccessToLayout(rhi::ResourceAccessType::DepthRead, true), RenderSyncFromAccess(rhi::ResourceAccessType::DepthRead)});
    detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.depthReadResources, std::forward<T>(x));
		return *this;
	}
	template<class Range>
		requires (std::ranges::range<Range>&&
    ResourceLike<std::ranges::range_value_t<Range>>)
        RenderPassBuilder& addDepthRead(Range&& xs) {
        for (auto&& e : xs) {
            addDepthRead(std::forward<decltype(e)>(e));
        }
		return *this;
	}

    // Constant buffer
	template<typename T>
        requires ResourceLike<T>
	RenderPassBuilder& addConstantBuffer(T&& x) {
    detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::ConstantBuffer, AccessToLayout(rhi::ResourceAccessType::ConstantBuffer, true), RenderSyncFromAccess(rhi::ResourceAccessType::ConstantBuffer)});
    detail::TrackDefaultDescriptorIdentifier(graph, params.autoDescriptorConstantBuffers, x, DescriptorType::CBV);
        detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.constantBuffers, std::forward<T>(x));
		return *this;
	}
	template<class Range>
		requires (std::ranges::range<Range>&&
    ResourceLike<std::ranges::range_value_t<Range>>)
        RenderPassBuilder& addConstantBuffer(Range&& xs) {
        for (auto&& e : xs) {
            addConstantBuffer(std::forward<decltype(e)>(e));
		}
		return *this;
	}

    // Unordered access
	template<typename T>
        requires ResourceLike<T>
	RenderPassBuilder& addUnorderedAccess(T&& x) {
    detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::UnorderedAccess, AccessToLayout(rhi::ResourceAccessType::UnorderedAccess, true), RenderSyncFromAccess(rhi::ResourceAccessType::UnorderedAccess)});
    detail::TrackDefaultDescriptorIdentifier(graph, params.autoDescriptorUnorderedAccessViews, x, DescriptorType::UAV);
        detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.unorderedAccessViews, std::forward<T>(x));
		return *this;
	}
	template<class Range>
		requires (std::ranges::range<Range>&&
    ResourceLike<std::ranges::range_value_t<Range>>)
        RenderPassBuilder& addUnorderedAccess(Range&& xs) {
        for (auto&& e : xs) {
            addUnorderedAccess(std::forward<decltype(e)>(e));
		}
		return *this;
	}

    template<typename T>
        requires ResourceLike<T>
    RenderPassBuilder& addUnorderedAccessClear(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::UnorderedAccessClear, AccessToLayout(rhi::ResourceAccessType::UnorderedAccessClear, true), RenderSyncFromAccess(rhi::ResourceAccessType::UnorderedAccessClear)});
        detail::TrackDefaultDescriptorIdentifier(graph, params.autoDescriptorUnorderedAccessViews, x, DescriptorType::UAV);
        detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.unorderedAccessClearViews, std::forward<T>(x));
        return *this;
    }
    template<class Range>
        requires (std::ranges::range<Range>&&
    ResourceLike<std::ranges::range_value_t<Range>>)
    RenderPassBuilder& addUnorderedAccessClear(Range&& xs) {
        for (auto&& e : xs) {
            addUnorderedAccessClear(std::forward<decltype(e)>(e));
        }
        return *this;
    }

    // Copy destination
	template<typename T>
        requires ResourceLike<T>
	RenderPassBuilder& addCopyDest(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::CopyDest, AccessToLayout(rhi::ResourceAccessType::CopyDest, true), RenderSyncFromAccess(rhi::ResourceAccessType::CopyDest)});
    detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.copyTargets, std::forward<T>(x));
		return *this;
	}
	template<class Range>
		requires (std::ranges::range<Range>&&
    ResourceLike<std::ranges::range_value_t<Range>>)
        RenderPassBuilder& addCopyDest(Range&& xs) {
        for (auto&& e : xs) {
			addCopyDest(std::forward<decltype(e)>(e));
		}
		return *this;
	}

    // Copy source
	template<typename T>
        requires ResourceLike<T>
	RenderPassBuilder& addCopySource(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::CopySource, AccessToLayout(rhi::ResourceAccessType::CopySource, true), RenderSyncFromAccess(rhi::ResourceAccessType::CopySource)});
    detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.copySources, std::forward<T>(x));
		return *this;
	}
	template<class Range>
		requires (std::ranges::range<Range>&&
    ResourceLike<std::ranges::range_value_t<Range>>)
        RenderPassBuilder& addCopySource(Range&& xs) {
		for (auto&& e : xs) {
			addCopySource(std::forward<decltype(e)>(e));
		}
		return *this;
	}

    // Indirect arguments
	template<typename T>
        requires ResourceLike<T>
	RenderPassBuilder& addIndirectArguments(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::IndirectArgument, AccessToLayout(rhi::ResourceAccessType::IndirectArgument, true), RenderSyncFromAccess(rhi::ResourceAccessType::IndirectArgument)});
    detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.indirectArgumentBuffers, std::forward<T>(x));
		return *this;
	}

    template<typename T>
        requires ResourceLike<T>
    RenderPassBuilder& addIndexBuffer(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::IndexBuffer, AccessToLayout(rhi::ResourceAccessType::IndexBuffer, true), RenderSyncFromAccess(rhi::ResourceAccessType::IndexBuffer)});
        detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.indexBuffers, std::forward<T>(x));
        return *this;
    }
	template <class Range>
		requires (std::ranges::range<Range>&&
    ResourceLike<std::ranges::range_value_t<Range>>)
        RenderPassBuilder& addIndirectArguments(Range&& xs) {
        for (auto&& e : xs) {
			addIndirectArguments(std::forward<decltype(e)>(e));
		}
		return *this;
	}

    template<typename T>
        requires ResourceLike<T>
    RenderPassBuilder& addPresent(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::Present, AccessToLayout(rhi::ResourceAccessType::Present, true), RenderSyncFromAccess(rhi::ResourceAccessType::Present)});
        detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.presentResources, std::forward<T>(x));
        return *this;
    }

	template <class Range>
		requires (std::ranges::range<Range>&&
    ResourceLike<std::ranges::range_value_t<Range>>)
    RenderPassBuilder& addPresent(Range&& xs) {
        for (auto&& e : xs) {
            addPresent(std::forward<decltype(e)>(e));
        }
        return *this;
    }

    // Legacy interop resources
    template<typename T>
        requires ResourceLike<T>
    RenderPassBuilder& addLegacyInterop(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::Common, AccessToLayout(rhi::ResourceAccessType::Common, true), RenderSyncFromAccess(rhi::ResourceAccessType::Common)});
        detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.legacyInteropResources, std::forward<T>(x));
        return *this;
    }
    template<class Range>
		requires (std::ranges::range<Range>&&
    ResourceLike<std::ranges::range_value_t<Range>>)
        RenderPassBuilder& addLegacyInterop(Range&& xs) {
        for (auto&& e : xs) {
            addLegacyInterop(std::forward<decltype(e)>(e));
        }
        return *this;
	}

    template<typename T>
        requires ResourceLike<T>
    RenderPassBuilder& addInternalTransition(T&& x, ResourceState exitState)& {
        detail::MaybeTrackResolverSnapshot(graph, resolverSnapshots_, x);
        detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendInternalTransition(graph, _declaredIds, params.internalTransitions, std::forward<T>(x), exitState);
        return *this;
    }

    void addActiveFeatureDomain(const FeatureDomainIdentifier& domain) {
        params.activeFeatureDomains.insert(domain);
    }

    void addActiveFeatureDomain(FeatureDomainIdentifier&& domain) {
        params.activeFeatureDomains.insert(std::move(domain));
    }

    void addActiveFeatureDomain(const char* domain) {
        if (domain) {
            params.activeFeatureDomains.insert(FeatureDomainIdentifier{ domain });
        }
    }

    void addActiveFeatureDomain(std::string_view domain) {
        params.activeFeatureDomains.insert(FeatureDomainIdentifier{ domain });
    }

    template<class S>
        requires (detail::StringLike<S> && !std::is_same_v<std::remove_cvref_t<S>, const char*> && !std::is_same_v<std::remove_cvref_t<S>, char*>)
    void addActiveFeatureDomain(S&& domain) {
        params.activeFeatureDomains.insert(FeatureDomainIdentifier{ std::string_view{ std::forward<S>(domain) } });
    }

	template <class Range>
		requires (std::ranges::range<Range>&&
    std::is_same_v<std::ranges::range_value_t<Range>, ResourceIdentifierAndRange>)
        RenderPassBuilder& addInternalTransition(Range&& xs, ResourceState exitState) {
        for (auto&& e : xs) {
            addInternalTransition(std::forward<decltype(e)>(e), exitState);
        }
        return *this;
	}

    std::vector<ResourceRequirement> GatherResourceRequirements() const {
        return detail::BuildRequirements(
            [](rhi::ResourceAccessType access) { return RenderSyncFromAccess(access); },
            std::pair{ std::cref(params.shaderResources), rhi::ResourceAccessType::ShaderResource },
            std::pair{ std::cref(params.constantBuffers), rhi::ResourceAccessType::ConstantBuffer },
            std::pair{ std::cref(params.renderTargets), rhi::ResourceAccessType::RenderTarget },
            std::pair{ std::cref(params.renderTargetClearResources), rhi::ResourceAccessType::RenderTargetClear },
            std::pair{ std::cref(params.depthReadResources), rhi::ResourceAccessType::DepthRead },
            std::pair{ std::cref(params.depthReadWriteResources), rhi::ResourceAccessType::DepthReadWrite },
            std::pair{ std::cref(params.depthStencilClearResources), rhi::ResourceAccessType::DepthStencilClear },
            std::pair{ std::cref(params.unorderedAccessViews), rhi::ResourceAccessType::UnorderedAccess },
            std::pair{ std::cref(params.unorderedAccessClearViews), rhi::ResourceAccessType::UnorderedAccessClear },
            std::pair{ std::cref(params.copySources), rhi::ResourceAccessType::CopySource },
            std::pair{ std::cref(params.copyTargets), rhi::ResourceAccessType::CopyDest },
            std::pair{ std::cref(params.indirectArgumentBuffers), rhi::ResourceAccessType::IndirectArgument },
            std::pair{ std::cref(params.indexBuffers), rhi::ResourceAccessType::IndexBuffer },
                std::pair{ std::cref(params.presentResources), rhi::ResourceAccessType::Present },
            std::pair{ std::cref(params.legacyInteropResources), rhi::ResourceAccessType::Common });
    }

    // storage
    RenderGraph*             graph;
    std::string              passName;
    RenderPassParameters     params;
	std::shared_ptr<RenderPass> pass;
    bool built_ = false;
    bool m_isGeometryPass = false;
	QueueKind m_preferredQueueKind = QueueKind::Graphics;
    QueueAssignmentPolicy m_queueAssignmentPolicy = QueueAssignmentPolicy::ForcePreferred;
    std::optional<QueueSlotIndex> m_pinnedQueueSlot;
	BackendAffinity m_backendAffinity{};
    std::unordered_set<ResourceIdentifier, ResourceIdentifier::Hasher> _declaredIds;
    std::vector<ResolverSnapshot> resolverSnapshots_;

    friend class RenderGraph; // Allow RenderGraph to create instances of this builder
};

template<typename T>
concept DerivedComputePass = std::derived_from<T, ComputePass>;

class ComputePassBuilder : public IPassBuilder {
public:
    PassBuilderKind Kind() const noexcept override { return PassBuilderKind::Compute; }
    IResourceProvider* ResourceProvider() noexcept override { return pass.get(); }
    // Typed declaration entry points mirror RenderPassBuilder.  Prepared
    // packets retain these stable declaration slots and resolve the admitted
    // backing at record time; they must not capture a buffer's current native
    // handle during preparation.
    template<class ResourceT>
        requires std::derived_from<ResourceT, Resource>
    ResourceBindingToken BindShaderResource(const std::shared_ptr<ResourceT>& resource) {
        if (!resource) throw std::invalid_argument("Cannot bind an empty shader resource");
        addShaderResource(resource);
        return { resource->GetSchedulingResourceID(), graph->RequestResourceHandle(resource.get()).GetGlobalResourceID() };
    }

    template<class ResourceT>
        requires std::derived_from<ResourceT, Resource>
    ResourceBindingToken BindUnorderedAccess(const std::shared_ptr<ResourceT>& resource) {
        if (!resource) throw std::invalid_argument("Cannot bind an empty unordered-access resource");
        addUnorderedAccess(resource);
        return { resource->GetSchedulingResourceID(), graph->RequestResourceHandle(resource.get()).GetGlobalResourceID() };
    }

    template<class ResourceT>
        requires std::derived_from<ResourceT, Resource>
    ResourceBindingToken BindIndirectArguments(const std::shared_ptr<ResourceT>& resource) {
        if (!resource) throw std::invalid_argument("Cannot bind an empty indirect-argument resource");
        addIndirectArguments(resource);
        return { resource->GetSchedulingResourceID(), graph->RequestResourceHandle(resource.get()).GetGlobalResourceID() };
    }
    // Variadic entry points

    //First set, callable on Lvalues
    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    ComputePassBuilder& WithShaderResource(Args&&... args) & {
        (addShaderResource(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    ComputePassBuilder& WithConstantBuffer(Args&&... args) & {
        (addConstantBuffer(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    ComputePassBuilder& WithUnorderedAccess(Args&&... args) & {
        (addUnorderedAccess(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    ComputePassBuilder& WithUnorderedAccessClear(Args&&... args) & {
        (addUnorderedAccessClear(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    ComputePassBuilder& WithIndirectArguments(Args&&... args) & {
        (addIndirectArguments(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    ComputePassBuilder& WithLegacyInterop(Args&&... args)& {
        (addLegacyInterop(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename T>
        requires ResourceLike<T>
    ComputePassBuilder& WithInternalTransition(T&& resource, ResourceState exitState)& {
        addInternalTransition(std::forward<T>(resource), exitState);
        return *this;
    }

    template<typename... Args>
    ComputePassBuilder& WithActiveFeatureDomain(Args&&... args) & {
        (addActiveFeatureDomain(std::forward<Args>(args)), ...);
        return *this;
    }

    // Second set, callable on temporaries
    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    ComputePassBuilder WithShaderResource(Args&&... args) && {
        (addShaderResource(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    ComputePassBuilder WithConstantBuffer(Args&&... args) && {
        (addConstantBuffer(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    ComputePassBuilder WithUnorderedAccess(Args&&... args) && {
        (addUnorderedAccess(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    ComputePassBuilder WithUnorderedAccessClear(Args&&... args) && {
        (addUnorderedAccessClear(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    ComputePassBuilder WithIndirectArguments(Args&&... args) && {
        (addIndirectArguments(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    ComputePassBuilder WithLegacyInterop(Args&&... args)&& {
        (addLegacyInterop(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename T>
        requires ResourceLike<T>
    ComputePassBuilder WithInternalTransition(T&& resource, ResourceState exitState)&& {
        addInternalTransition(std::forward<T>(resource), exitState);
        return std::move(*this);
    }

    template<typename... Args>
    ComputePassBuilder WithActiveFeatureDomain(Args&&... args) && {
        (addActiveFeatureDomain(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename AddCallable>
    ComputePassBuilder& WithResolverDeclaration(const IResourceResolver& resolver, AddCallable&& addCallable) & {
        const auto state = graph->CaptureResolverDeclarationState(resolver);
        if (state && state->resources) {
            addCallable(ResourceResolverAndRange{resolver});
        } else {
            const auto resources = resolver.Resolve();
            for (const auto& resource : resources) graph->AddResource(resource);
            addCallable(resources);
        }
        return *this;
    }

    template<typename AddCallable>
    ComputePassBuilder WithResolverDeclaration(const IResourceResolver& resolver, AddCallable&& addCallable) && {
        const auto state = graph->CaptureResolverDeclarationState(resolver);
        if (state && state->resources) {
            addCallable(ResourceResolverAndRange{resolver});
        } else {
            const auto resources = resolver.Resolve();
            for (const auto& resource : resources) graph->AddResource(resource);
            addCallable(resources);
        }
        return std::move(*this);
    }

        std::vector<ResolverSnapshot> TakeResolverSnapshots() { return std::move(resolverSnapshots_); }

        ComputePassBuilder& PreferQueue(QueueKind kind) & {
                if (!IsQueueKindSupportedByComputePass(kind)) {
                        throw std::invalid_argument("Compute passes only support graphics or compute queues");
                }
                m_preferredQueueKind = kind;
            m_queueAssignmentPolicy = QueueAssignmentPolicy::ForcePreferred;
                return *this;
        }

        ComputePassBuilder PreferQueue(QueueKind kind) && {
                if (!IsQueueKindSupportedByComputePass(kind)) {
                        throw std::invalid_argument("Compute passes only support graphics or compute queues");
                }
                m_preferredQueueKind = kind;
            m_queueAssignmentPolicy = QueueAssignmentPolicy::ForcePreferred;
    				return std::move(*this);
    		}

    		ComputePassBuilder& AutomaticQueueAssignment() & {
    				m_queueAssignmentPolicy = QueueAssignmentPolicy::Automatic;
    				return *this;
    		}

    		ComputePassBuilder AutomaticQueueAssignment() && {
    				m_queueAssignmentPolicy = QueueAssignmentPolicy::Automatic;
                return std::move(*this);
        }

        ComputePassBuilder& PinToQueue(QueueSlotIndex slot) & {
                m_pinnedQueueSlot = slot;
                return *this;
        }

        ComputePassBuilder PinToQueue(QueueSlotIndex slot) && {
                m_pinnedQueueSlot = slot;
                return std::move(*this);
        }

		ComputePassBuilder& RequireBackend(rhi::Backend backend) & { m_backendAffinity = { BackendAffinityStrength::Required, backend }; return *this; }
		ComputePassBuilder RequireBackend(rhi::Backend backend) && { m_backendAffinity = { BackendAffinityStrength::Required, backend }; return std::move(*this); }
		ComputePassBuilder& PreferBackend(rhi::Backend backend) & { m_backendAffinity = { BackendAffinityStrength::Preferred, backend }; return *this; }
		ComputePassBuilder PreferBackend(rhi::Backend backend) && { m_backendAffinity = { BackendAffinityStrength::Preferred, backend }; return std::move(*this); }
		ComputePassBuilder& RequireAPI(rhi::Backend backend) & { return RequireBackend(backend); }
		ComputePassBuilder RequireAPI(rhi::Backend backend) && { return std::move(*this).RequireBackend(backend); }
		ComputePassBuilder& PreferAPI(rhi::Backend backend) & { return PreferBackend(backend); }
		ComputePassBuilder PreferAPI(rhi::Backend backend) && { return std::move(*this).PreferBackend(backend); }
		ComputePassBuilder& RequireDevice(DeviceInstanceId device) & { m_backendAffinity = { BackendAffinityStrength::Required, rhi::Backend::Null, device }; return *this; }
		ComputePassBuilder RequireDevice(DeviceInstanceId device) && { m_backendAffinity = { BackendAffinityStrength::Required, rhi::Backend::Null, device }; return std::move(*this); }
		ComputePassBuilder& PreferDevice(DeviceInstanceId device) & { m_backendAffinity = { BackendAffinityStrength::Preferred, rhi::Backend::Null, device }; return *this; }
		ComputePassBuilder PreferDevice(DeviceInstanceId device) && { m_backendAffinity = { BackendAffinityStrength::Preferred, rhi::Backend::Null, device }; return std::move(*this); }

		ComputePassBuilder& WithExternalWaitBeforeTransitions(rhi::Timeline timeline, uint64_t value) & {
			params.externalWaitsBeforeTransitions.push_back({ timeline, value });
			return *this;
		}

		ComputePassBuilder WithExternalWaitBeforeTransitions(rhi::Timeline timeline, uint64_t value) && {
			params.externalWaitsBeforeTransitions.push_back({ timeline, value });
			return std::move(*this);
		}
		ComputePassBuilder& WithExternalWaitBindingBeforeTransitions(ExternalTimelineBinding binding) & { params.externalWaitBindingsBeforeTransitions.push_back(binding); return *this; }
		ComputePassBuilder WithExternalWaitBindingBeforeTransitions(ExternalTimelineBinding binding) && { params.externalWaitBindingsBeforeTransitions.push_back(binding); return std::move(*this); }

        // LVALUE overloads for IResourceResolver
        ComputePassBuilder& WithShaderResource(const IResourceResolver& r)& {
		return WithResolverDeclaration(r, [&](auto&& resolved) { addShaderResource(std::forward<decltype(resolved)>(resolved)); });
        }

        ComputePassBuilder& WithConstantBuffer(const IResourceResolver& r)& {
		return WithResolverDeclaration(r, [&](auto&& resolved) { addConstantBuffer(std::forward<decltype(resolved)>(resolved)); });
        }

        ComputePassBuilder& WithUnorderedAccess(const IResourceResolver& r)& {
		return WithResolverDeclaration(r, [&](auto&& resolved) { addUnorderedAccess(std::forward<decltype(resolved)>(resolved)); });
        }

        ComputePassBuilder& WithUnorderedAccessClear(const IResourceResolver& r)& {
		return WithResolverDeclaration(r, [&](auto&& resolved) { addUnorderedAccessClear(std::forward<decltype(resolved)>(resolved)); });
        }

        ComputePassBuilder& WithIndirectArguments(const IResourceResolver& r)& {
		return WithResolverDeclaration(r, [&](auto&& resolved) { addIndirectArguments(std::forward<decltype(resolved)>(resolved)); });
        }

        ComputePassBuilder& WithLegacyInterop(const IResourceResolver& r)& {
		return WithResolverDeclaration(r, [&](auto&& resolved) { addLegacyInterop(std::forward<decltype(resolved)>(resolved)); });
        }

        // RVALUE overloads for IResourceResolver
        ComputePassBuilder WithShaderResource(const IResourceResolver& r)&& {
		return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addShaderResource(std::forward<decltype(resolved)>(resolved)); });
        }

        ComputePassBuilder WithConstantBuffer(const IResourceResolver& r)&& {
		return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addConstantBuffer(std::forward<decltype(resolved)>(resolved)); });
        }

        ComputePassBuilder WithUnorderedAccess(const IResourceResolver& r)&& {
		return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addUnorderedAccess(std::forward<decltype(resolved)>(resolved)); });
        }

        ComputePassBuilder WithUnorderedAccessClear(const IResourceResolver& r)&& {
		return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addUnorderedAccessClear(std::forward<decltype(resolved)>(resolved)); });
        }

        ComputePassBuilder WithIndirectArguments(const IResourceResolver& r)&& {
		return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addIndirectArguments(std::forward<decltype(resolved)>(resolved)); });
        }

        ComputePassBuilder WithLegacyInterop(const IResourceResolver& r)&& {
		return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addLegacyInterop(std::forward<decltype(resolved)>(resolved)); });
        }

    auto const& DeclaredResourceIds() const { return _declaredIds; }

private:
    struct AuthorState {
        ComputePassParameters params;
        std::unordered_set<ResourceIdentifier, ResourceIdentifier::Hasher> declaredIds;
        std::vector<ResolverSnapshot> resolverSnapshots;
    };

    ComputePassBuilder(RenderGraph* g, std::string name)
        : graph(g), passName(std::move(name)) {}

    // Copy and move constructors are default, but private
    ComputePassBuilder(const ComputePassBuilder&) = default;
    ComputePassBuilder(ComputePassBuilder&&) = default;

    // Same for assignment
    ComputePassBuilder& operator=(const ComputePassBuilder&) = default;
    ComputePassBuilder& operator=(ComputePassBuilder&&) = default;

    AuthorState CaptureAuthorState() && {
        return {
            std::move(params),
            std::move(_declaredIds),
            std::move(resolverSnapshots_)
        };
    }

    void RestoreAuthorState(AuthorState&& state) {
        params = std::move(state.params);
        _declaredIds = std::move(state.declaredIds);
        resolverSnapshots_ = std::move(state.resolverSnapshots);
    }

    template<DerivedComputePass PassT, org::PassInputs InputsT, typename... StableCtorArgs>
    void Instantiate(InputsT&& inputs, StableCtorArgs&&... ctorArgs)
    {
        if (!built_)
        {
            built_ = true;
            pass = detail::MakePass<PassT>(
                std::forward<InputsT>(inputs),
                std::forward<StableCtorArgs>(ctorArgs)...);
        }

        pass->SetInputs(std::forward<InputsT>(inputs));
    }

    template<DerivedComputePass PassT, typename... StableCtorArgs>
    void Instantiate(StableCtorArgs&&... ctorArgs)
    {
        Instantiate<PassT>(org::NoInputs{}, std::forward<StableCtorArgs>(ctorArgs)...);
    }

    void Finalize() override {
        if (!built_) return;

        auto authorState = std::move(*this).CaptureAuthorState();

        params = {};
        _declaredIds.clear();
        resolverSnapshots_.clear();

        RestoreAuthorState(std::move(authorState));

        pass->DeclareResourceUsages(this);

        params.identifierSet = _declaredIds;
        params.preferredQueueKind = m_preferredQueueKind;
        params.queueAssignmentPolicy = m_queueAssignmentPolicy;
        params.pinnedQueueSlot = m_pinnedQueueSlot;
		params.backendAffinity = m_backendAffinity;
        params.staticResourceRequirements = GatherResourceRequirements();

        graph->AddComputePass(pass, params, passName, TakeResolverSnapshots());
    }

    void Reset() override {
        built_ = false;
        pass = nullptr;
        params = {};
        _declaredIds.clear();
        resolverSnapshots_.clear();
        m_preferredQueueKind = QueueKind::Compute;
		m_queueAssignmentPolicy = QueueAssignmentPolicy::Automatic;
        m_pinnedQueueSlot = std::nullopt;
		m_backendAffinity = {};
    }

    // Shader resource
	template<typename T>
        requires ResourceLike<T>
	ComputePassBuilder& addShaderResource(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::ShaderResource, AccessToLayout(rhi::ResourceAccessType::ShaderResource, true), ComputeSyncFromAccess(rhi::ResourceAccessType::ShaderResource)});
    detail::TrackDefaultDescriptorIdentifier(graph, params.autoDescriptorShaderResources, x, DescriptorType::SRV);
        detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.shaderResources, std::forward<T>(x));
		return *this;
	}
    template<class Range>
        requires (std::ranges::range<Range>&&
    ResourceLike<std::ranges::range_value_t<Range>>)
        ComputePassBuilder& addShaderResource(Range&& xs) {
        for (auto&& e : xs) {
            addShaderResource(std::forward<decltype(e)>(e));
        }
        return *this;
    }

    // Constant buffer
	template<typename T>
        requires ResourceLike<T>
	ComputePassBuilder& addConstantBuffer(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::ConstantBuffer, AccessToLayout(rhi::ResourceAccessType::ConstantBuffer, true), ComputeSyncFromAccess(rhi::ResourceAccessType::ConstantBuffer)});
    detail::TrackDefaultDescriptorIdentifier(graph, params.autoDescriptorConstantBuffers, x, DescriptorType::CBV);
        detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.constantBuffers, std::forward<T>(x));
		return *this;
	}
    template<class Range>
		requires (std::ranges::range<Range>&&
	ResourceLike<std::ranges::range_value_t<Range>>)
        ComputePassBuilder& addConstantBuffer(Range&& xs) {
		for (auto&& e : xs) {
			addConstantBuffer(std::forward<decltype(e)>(e));
		}
		return *this;
	}

    // Unordered access
	template<typename T>
        requires ResourceLike<T>
	ComputePassBuilder& addUnorderedAccess(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::UnorderedAccess, AccessToLayout(rhi::ResourceAccessType::UnorderedAccess, true), ComputeSyncFromAccess(rhi::ResourceAccessType::UnorderedAccess)});
    detail::TrackDefaultDescriptorIdentifier(graph, params.autoDescriptorUnorderedAccessViews, x, DescriptorType::UAV);
        detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.unorderedAccessViews, std::forward<T>(x));
		return *this;
	}
    template<class Range>
		requires (std::ranges::range<Range>&&
	ResourceLike<std::ranges::range_value_t<Range>>)
        ComputePassBuilder& addUnorderedAccess(Range&& xs) {
        for (auto&& e : xs) {
            addUnorderedAccess(std::forward<decltype(e)>(e));
        }
        return *this;
	}

    template<typename T>
        requires ResourceLike<T>
    ComputePassBuilder& addUnorderedAccessClear(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::UnorderedAccessClear, AccessToLayout(rhi::ResourceAccessType::UnorderedAccessClear, true), ComputeSyncFromAccess(rhi::ResourceAccessType::UnorderedAccessClear)});
        detail::TrackDefaultDescriptorIdentifier(graph, params.autoDescriptorUnorderedAccessViews, x, DescriptorType::UAV);
        detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.unorderedAccessClearViews, std::forward<T>(x));
        return *this;
    }
    template<class Range>
        requires (std::ranges::range<Range>&&
    ResourceLike<std::ranges::range_value_t<Range>>)
    ComputePassBuilder& addUnorderedAccessClear(Range&& xs) {
        for (auto&& e : xs) {
            addUnorderedAccessClear(std::forward<decltype(e)>(e));
        }
        return *this;
    }

	// Indirect arguments
	template<typename T>
        requires ResourceLike<T>
	ComputePassBuilder& addIndirectArguments(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::IndirectArgument, AccessToLayout(rhi::ResourceAccessType::IndirectArgument, true), ComputeSyncFromAccess(rhi::ResourceAccessType::IndirectArgument)});
    detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.indirectArgumentBuffers, std::forward<T>(x));
		return *this;
	}
	template<class Range>
		requires (std::ranges::range<Range>&&
	ResourceLike<std::ranges::range_value_t<Range>>)
        ComputePassBuilder& addIndirectArguments(Range&& xs) {
		for (auto&& e : xs) {
			addIndirectArguments(std::forward<decltype(e)>(e));
		}
		return *this;
	}

    // Legacy interop resources
    template<typename T>
        requires ResourceLike<T>
    ComputePassBuilder& addLegacyInterop(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::Common, AccessToLayout(rhi::ResourceAccessType::Common, true), ComputeSyncFromAccess(rhi::ResourceAccessType::Common)});
        detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendTrackedResource(graph, _declaredIds, params.legacyInteropResources, std::forward<T>(x));
        return *this;
    }
    template<class Range>
		requires (std::ranges::range<Range>&&
    ResourceLike<std::ranges::range_value_t<Range>>)
        ComputePassBuilder& addLegacyInterop(Range&& xs) {
        for (auto&& e : xs) {
            addLegacyInterop(std::forward<decltype(e)>(e));
        }
		return *this;
	}

    ComputePassBuilder& addInternalTransition(
        ResourceIdentifierAndRange rar,
        ResourceState exitState) &
    {
		auto ranges = processResourceArguments(rar, graph);
        for (auto& r : ranges) {
            //if (!r.resource) continue;
            params.internalTransitions.emplace_back(r, exitState);
		}
        return *this;
	}
    template<typename T>
        requires ResourceLike<T>
    ComputePassBuilder& addInternalTransition(T&& x, ResourceState exitState)& {
        detail::MaybeTrackResolverSnapshot(graph, resolverSnapshots_, x);
        detail::TrackFeatureDomainActivation(params.activeFeatureDomains, x);
        detail::AppendInternalTransition(graph, _declaredIds, params.internalTransitions, std::forward<T>(x), exitState);
        return *this;
    }

    void addActiveFeatureDomain(const FeatureDomainIdentifier& domain) {
        params.activeFeatureDomains.insert(domain);
    }

    void addActiveFeatureDomain(FeatureDomainIdentifier&& domain) {
        params.activeFeatureDomains.insert(std::move(domain));
    }

    void addActiveFeatureDomain(const char* domain) {
        if (domain) {
            params.activeFeatureDomains.insert(FeatureDomainIdentifier{ domain });
        }
    }

    void addActiveFeatureDomain(std::string_view domain) {
        params.activeFeatureDomains.insert(FeatureDomainIdentifier{ domain });
    }

    template<class S>
        requires (detail::StringLike<S> && !std::is_same_v<std::remove_cvref_t<S>, const char*> && !std::is_same_v<std::remove_cvref_t<S>, char*>)
    void addActiveFeatureDomain(S&& domain) {
        params.activeFeatureDomains.insert(FeatureDomainIdentifier{ std::string_view{ std::forward<S>(domain) } });
    }

    std::vector<ResourceRequirement> GatherResourceRequirements() const {
        return detail::BuildRequirements(
            [](rhi::ResourceAccessType access) { return ComputeSyncFromAccess(access); },
            std::pair{ std::cref(params.shaderResources), rhi::ResourceAccessType::ShaderResource },
            std::pair{ std::cref(params.constantBuffers), rhi::ResourceAccessType::ConstantBuffer },
            std::pair{ std::cref(params.unorderedAccessViews), rhi::ResourceAccessType::UnorderedAccess },
            std::pair{ std::cref(params.unorderedAccessClearViews), rhi::ResourceAccessType::UnorderedAccessClear },
            std::pair{ std::cref(params.indirectArgumentBuffers), rhi::ResourceAccessType::IndirectArgument },
            std::pair{ std::cref(params.legacyInteropResources), rhi::ResourceAccessType::Common });
    }

    // storage
    RenderGraph*             graph;
    std::string              passName;
    ComputePassParameters     params;
    std::shared_ptr<ComputePass> pass;
    bool built_ = false;
	QueueKind m_preferredQueueKind = QueueKind::Compute;
    QueueAssignmentPolicy m_queueAssignmentPolicy = QueueAssignmentPolicy::Automatic;
    std::optional<QueueSlotIndex> m_pinnedQueueSlot;
	BackendAffinity m_backendAffinity{};
    std::unordered_set<ResourceIdentifier, ResourceIdentifier::Hasher> _declaredIds;
    std::vector<ResolverSnapshot> resolverSnapshots_;

	friend class RenderGraph; // Allow RenderGraph to create instances of this builder
};

template<typename T>
concept DerivedCopyPass = std::derived_from<T, CopyPass>;

class CopyPassBuilder : public IPassBuilder {
public:
    PassBuilderKind Kind() const noexcept override { return PassBuilderKind::Copy; }
    IResourceProvider* ResourceProvider() noexcept override { return pass.get(); }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    CopyPassBuilder& WithCopyDest(Args&&... args) & {
        (addCopyDest(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    CopyPassBuilder& WithCopySource(Args&&... args) & {
        (addCopySource(std::forward<Args>(args)), ...);
        return *this;
    }

    template<typename T>
        requires ResourceLike<T>
    CopyPassBuilder& WithInternalTransition(T&& resource, ResourceState exitState)& {
        addInternalTransition(std::forward<T>(resource), exitState);
        return *this;
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    CopyPassBuilder WithCopyDest(Args&&... args) && {
        (addCopyDest(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename... Args>
        requires ((NotIResourceResolver<Args>) && ...)
    CopyPassBuilder WithCopySource(Args&&... args) && {
        (addCopySource(std::forward<Args>(args)), ...);
        return std::move(*this);
    }

    template<typename T>
        requires ResourceLike<T>
    CopyPassBuilder WithInternalTransition(T&& resource, ResourceState exitState)&& {
        addInternalTransition(std::forward<T>(resource), exitState);
        return std::move(*this);
    }

    template<typename AddCallable>
    CopyPassBuilder& WithResolverDeclaration(const IResourceResolver& resolver, AddCallable&& addCallable) & {
        const auto state = graph->CaptureResolverDeclarationState(resolver);
        if (state && state->resources) {
            addCallable(ResourceResolverAndRange{resolver});
        } else {
            const auto resources = resolver.Resolve();
            for (const auto& resource : resources) graph->AddResource(resource);
            addCallable(resources);
        }
        return *this;
    }

    template<typename AddCallable>
    CopyPassBuilder WithResolverDeclaration(const IResourceResolver& resolver, AddCallable&& addCallable) && {
        const auto state = graph->CaptureResolverDeclarationState(resolver);
        if (state && state->resources) {
            addCallable(ResourceResolverAndRange{resolver});
        } else {
            const auto resources = resolver.Resolve();
            for (const auto& resource : resources) graph->AddResource(resource);
            addCallable(resources);
        }
        return std::move(*this);
    }

    std::vector<ResolverSnapshot> TakeResolverSnapshots() { return std::move(resolverSnapshots_); }

    CopyPassBuilder& PreferQueue(QueueKind kind) & {
        if (!IsQueueKindSupportedByCopyPass(kind)) {
            throw std::invalid_argument("Copy passes only support graphics or copy queues");
        }
        m_preferredQueueKind = kind;
        m_queueAssignmentPolicy = QueueAssignmentPolicy::ForcePreferred;
        return *this;
    }

    CopyPassBuilder PreferQueue(QueueKind kind) && {
        if (!IsQueueKindSupportedByCopyPass(kind)) {
            throw std::invalid_argument("Copy passes only support graphics or copy queues");
        }
        m_preferredQueueKind = kind;
        m_queueAssignmentPolicy = QueueAssignmentPolicy::ForcePreferred;
		return std::move(*this);
	}

	CopyPassBuilder& AutomaticQueueAssignment() & {
		m_queueAssignmentPolicy = QueueAssignmentPolicy::Automatic;
		return *this;
	}

	CopyPassBuilder AutomaticQueueAssignment() && {
		m_queueAssignmentPolicy = QueueAssignmentPolicy::Automatic;
        return std::move(*this);
    }

    CopyPassBuilder& PinToQueue(QueueSlotIndex slot) & {
        m_pinnedQueueSlot = slot;
        return *this;
    }

    CopyPassBuilder PinToQueue(QueueSlotIndex slot) && {
        m_pinnedQueueSlot = slot;
        return std::move(*this);
    }

	CopyPassBuilder& RequireBackend(rhi::Backend backend) & { m_backendAffinity = { BackendAffinityStrength::Required, backend }; return *this; }
	CopyPassBuilder RequireBackend(rhi::Backend backend) && { m_backendAffinity = { BackendAffinityStrength::Required, backend }; return std::move(*this); }
	CopyPassBuilder& PreferBackend(rhi::Backend backend) & { m_backendAffinity = { BackendAffinityStrength::Preferred, backend }; return *this; }
	CopyPassBuilder PreferBackend(rhi::Backend backend) && { m_backendAffinity = { BackendAffinityStrength::Preferred, backend }; return std::move(*this); }
	CopyPassBuilder& RequireAPI(rhi::Backend backend) & { return RequireBackend(backend); }
	CopyPassBuilder RequireAPI(rhi::Backend backend) && { return std::move(*this).RequireBackend(backend); }
	CopyPassBuilder& PreferAPI(rhi::Backend backend) & { return PreferBackend(backend); }
	CopyPassBuilder PreferAPI(rhi::Backend backend) && { return std::move(*this).PreferBackend(backend); }
	CopyPassBuilder& RequireDevice(DeviceInstanceId device) & { m_backendAffinity = { BackendAffinityStrength::Required, rhi::Backend::Null, device }; return *this; }
	CopyPassBuilder RequireDevice(DeviceInstanceId device) && { m_backendAffinity = { BackendAffinityStrength::Required, rhi::Backend::Null, device }; return std::move(*this); }
	CopyPassBuilder& PreferDevice(DeviceInstanceId device) & { m_backendAffinity = { BackendAffinityStrength::Preferred, rhi::Backend::Null, device }; return *this; }
	CopyPassBuilder PreferDevice(DeviceInstanceId device) && { m_backendAffinity = { BackendAffinityStrength::Preferred, rhi::Backend::Null, device }; return std::move(*this); }

	CopyPassBuilder& WithExternalWaitBeforeTransitions(rhi::Timeline timeline, uint64_t value) & {
		params.externalWaitsBeforeTransitions.push_back({ timeline, value });
		return *this;
	}

	CopyPassBuilder WithExternalWaitBeforeTransitions(rhi::Timeline timeline, uint64_t value) && {
		params.externalWaitsBeforeTransitions.push_back({ timeline, value });
		return std::move(*this);
	}
	CopyPassBuilder& WithExternalWaitBindingBeforeTransitions(ExternalTimelineBinding binding) & { params.externalWaitBindingsBeforeTransitions.push_back(binding); return *this; }
	CopyPassBuilder WithExternalWaitBindingBeforeTransitions(ExternalTimelineBinding binding) && { params.externalWaitBindingsBeforeTransitions.push_back(binding); return std::move(*this); }

    CopyPassBuilder& WithCopyDest(const IResourceResolver& r)& {
        return WithResolverDeclaration(r, [&](auto&& resolved) { addCopyDest(std::forward<decltype(resolved)>(resolved)); });
    }

    CopyPassBuilder& WithCopySource(const IResourceResolver& r)& {
        return WithResolverDeclaration(r, [&](auto&& resolved) { addCopySource(std::forward<decltype(resolved)>(resolved)); });
    }

    CopyPassBuilder WithCopyDest(const IResourceResolver& r)&& {
        return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addCopyDest(std::forward<decltype(resolved)>(resolved)); });
    }

    CopyPassBuilder WithCopySource(const IResourceResolver& r)&& {
        return std::move(*this).WithResolverDeclaration(r, [&](auto&& resolved) { addCopySource(std::forward<decltype(resolved)>(resolved)); });
    }

    auto const& DeclaredResourceIds() const { return _declaredIds; }

private:
    struct AuthorState {
        CopyPassParameters params;
        std::unordered_set<ResourceIdentifier, ResourceIdentifier::Hasher> declaredIds;
        std::vector<ResolverSnapshot> resolverSnapshots;
    };

    CopyPassBuilder(RenderGraph* g, std::string name)
        : graph(g), passName(std::move(name)) {}

    CopyPassBuilder(const CopyPassBuilder&) = default;
    CopyPassBuilder(CopyPassBuilder&&) = default;
    CopyPassBuilder& operator=(const CopyPassBuilder&) = default;
    CopyPassBuilder& operator=(CopyPassBuilder&&) = default;

    AuthorState CaptureAuthorState() && {
        return {
            std::move(params),
            std::move(_declaredIds),
            std::move(resolverSnapshots_)
        };
    }

    void RestoreAuthorState(AuthorState&& state) {
        params = std::move(state.params);
        _declaredIds = std::move(state.declaredIds);
        resolverSnapshots_ = std::move(state.resolverSnapshots);
    }

    template<DerivedCopyPass PassT, org::PassInputs InputsT, typename... StableCtorArgs>
    void Instantiate(InputsT&& inputs, StableCtorArgs&&... ctorArgs)
    {
        if (!built_)
        {
            built_ = true;
            pass = detail::MakePass<PassT>(
                std::forward<InputsT>(inputs),
                std::forward<StableCtorArgs>(ctorArgs)...);
        }

        pass->SetInputs(std::forward<InputsT>(inputs));
    }

    template<DerivedCopyPass PassT, typename... StableCtorArgs>
    void Instantiate(StableCtorArgs&&... ctorArgs)
    {
        Instantiate<PassT>(org::NoInputs{}, std::forward<StableCtorArgs>(ctorArgs)...);
    }

    void Finalize() override {
        if (!built_) return;

        auto authorState = std::move(*this).CaptureAuthorState();

        params = {};
        _declaredIds.clear();
        resolverSnapshots_.clear();

        RestoreAuthorState(std::move(authorState));

        pass->DeclareResourceUsages(this);

        params.identifierSet = _declaredIds;
        params.preferredQueueKind = m_preferredQueueKind;
        params.queueAssignmentPolicy = m_queueAssignmentPolicy;
        params.pinnedQueueSlot = m_pinnedQueueSlot;
		params.backendAffinity = m_backendAffinity;
        params.staticResourceRequirements = GatherResourceRequirements();

        graph->AddCopyPass(pass, params, passName, TakeResolverSnapshots());
    }

    void Reset() override {
        built_ = false;
        pass = nullptr;
        params = {};
        _declaredIds.clear();
        resolverSnapshots_.clear();
        m_preferredQueueKind = QueueKind::Copy;
		m_queueAssignmentPolicy = QueueAssignmentPolicy::ForcePreferred;
        m_pinnedQueueSlot = std::nullopt;
		m_backendAffinity = {};
    }

    template<typename T>
        requires ResourceLike<T>
    CopyPassBuilder& addCopyDest(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::CopyDest, AccessToLayout(rhi::ResourceAccessType::CopyDest, true), rhi::ResourceSyncState::Copy});
        detail::AppendTrackedResource(graph, _declaredIds, params.copyTargets, std::forward<T>(x));
        return *this;
    }

    template<class Range>
        requires (std::ranges::range<Range>&&
    ResourceLike<std::ranges::range_value_t<Range>>)
    CopyPassBuilder& addCopyDest(Range&& xs) {
        for (auto&& e : xs) {
            addCopyDest(std::forward<decltype(e)>(e));
        }
        return *this;
    }

    template<typename T>
        requires ResourceLike<T>
    CopyPassBuilder& addCopySource(T&& x) {
        detail::TrackResolverDeclaration(graph, resolverSnapshots_, x,
            ResourceState{rhi::ResourceAccessType::CopySource, AccessToLayout(rhi::ResourceAccessType::CopySource, true), rhi::ResourceSyncState::Copy});
        detail::AppendTrackedResource(graph, _declaredIds, params.copySources, std::forward<T>(x));
        return *this;
    }

    template<class Range>
        requires (std::ranges::range<Range>&&
    ResourceLike<std::ranges::range_value_t<Range>>)
    CopyPassBuilder& addCopySource(Range&& xs) {
        for (auto&& e : xs) {
            addCopySource(std::forward<decltype(e)>(e));
        }
        return *this;
    }

    template<typename T>
        requires ResourceLike<T>
    CopyPassBuilder& addInternalTransition(T&& x, ResourceState exitState)& {
        detail::MaybeTrackResolverSnapshot(graph, resolverSnapshots_, x);
        detail::AppendInternalTransition(graph, _declaredIds, params.internalTransitions, std::forward<T>(x), exitState);
        return *this;
    }

    std::vector<ResourceRequirement> GatherResourceRequirements() const {
        return detail::BuildRequirements(
            [](rhi::ResourceAccessType access) {
                if ((access & (rhi::ResourceAccessType::CopySource | rhi::ResourceAccessType::CopyDest)) != 0) {
                    return rhi::ResourceSyncState::Copy;
                }
                return rhi::ResourceSyncState::All;
            },
            std::pair{ std::cref(params.copySources), rhi::ResourceAccessType::CopySource },
            std::pair{ std::cref(params.copyTargets), rhi::ResourceAccessType::CopyDest });
    }

    RenderGraph* graph;
    std::string passName;
    CopyPassParameters params;
    std::shared_ptr<CopyPass> pass;
    bool built_ = false;
    QueueKind m_preferredQueueKind = QueueKind::Copy;
	QueueAssignmentPolicy m_queueAssignmentPolicy = QueueAssignmentPolicy::ForcePreferred;
    std::optional<QueueSlotIndex> m_pinnedQueueSlot;
	BackendAffinity m_backendAffinity{};
    std::unordered_set<ResourceIdentifier, ResourceIdentifier::Hasher> _declaredIds;
    std::vector<ResolverSnapshot> resolverSnapshots_;

    friend class RenderGraph;
};

template<typename PassT, org::PassInputs InputsT, typename... StableCtorArgs>
ComputePassBuilder& RenderGraph::BuildComputePass(std::string const& name, InputsT&& inputs, StableCtorArgs&&... ctorArgs) {
    static_assert(DerivedComputePass<PassT>);
    auto& builder = GetOrCreateComputePassBuilder(name);
    builder.template Instantiate<PassT>(std::forward<InputsT>(inputs), std::forward<StableCtorArgs>(ctorArgs)...);
    return builder;
}

template<typename PassT, typename... StableCtorArgs>
ComputePassBuilder& RenderGraph::BuildComputePass(std::string const& name, StableCtorArgs&&... ctorArgs) {
    static_assert(DerivedComputePass<PassT>);
    auto& builder = GetOrCreateComputePassBuilder(name);
    builder.template Instantiate<PassT>(std::forward<StableCtorArgs>(ctorArgs)...);
    return builder;
}

template<typename PassT, org::PassInputs InputsT, typename... StableCtorArgs>
RenderPassBuilder& RenderGraph::BuildRenderPass(std::string const& name, InputsT&& inputs, StableCtorArgs&&... ctorArgs) {
    static_assert(DerivedRenderPass<PassT>);
    auto& builder = GetOrCreateRenderPassBuilder(name);
    builder.template Instantiate<PassT>(std::forward<InputsT>(inputs), std::forward<StableCtorArgs>(ctorArgs)...);
    return builder;
}

template<typename PassT, typename... StableCtorArgs>
RenderPassBuilder& RenderGraph::BuildRenderPass(std::string const& name, StableCtorArgs&&... ctorArgs) {
    static_assert(DerivedRenderPass<PassT>);
    auto& builder = GetOrCreateRenderPassBuilder(name);
    builder.template Instantiate<PassT>(std::forward<StableCtorArgs>(ctorArgs)...);
    return builder;
}

template<typename PassT, org::PassInputs InputsT, typename... StableCtorArgs>
CopyPassBuilder& RenderGraph::BuildCopyPass(std::string const& name, InputsT&& inputs, StableCtorArgs&&... ctorArgs) {
    static_assert(DerivedCopyPass<PassT>);
    auto& builder = GetOrCreateCopyPassBuilder(name);
    builder.template Instantiate<PassT>(std::forward<InputsT>(inputs), std::forward<StableCtorArgs>(ctorArgs)...);
    return builder;
}

template<typename PassT, typename... StableCtorArgs>
CopyPassBuilder& RenderGraph::BuildCopyPass(std::string const& name, StableCtorArgs&&... ctorArgs) {
    static_assert(DerivedCopyPass<PassT>);
    auto& builder = GetOrCreateCopyPassBuilder(name);
    builder.template Instantiate<PassT>(std::forward<StableCtorArgs>(ctorArgs)...);
    return builder;
}


} // namespace org
