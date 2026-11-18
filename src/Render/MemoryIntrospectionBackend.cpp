#include "Render/MemoryIntrospectionBackend.h"

#include "Render/MemoryIntrospectionAPI.h"
#include "Managers/Singletons/ECSManager.h"
#include "Resources/MemoryStatisticsComponents.h"
#include "Resources/ResourceIdentifier.h"

#include <unordered_map>

namespace org::memory {

namespace {
class ECSMemorySnapshotProvider final : public IMemorySnapshotProvider {
    // Defined before the constructor that calls it. A deduced return type is only known after its definition.
    template <class F>
    static decltype(auto) WithWorld(flecs::world* a_world, F&& a_use) {
        if (a_world)
            return std::forward<F>(a_use)(*a_world);
        return ECSManager::GetInstance().Access(std::forward<F>(a_use));
    }

public:
    // a_world is a host's own world. The host keeps it from changing while a snapshot reads it.
    // a_world is null for ORG's tracking world. ECSManager gives that world one owner at a time.
    // The query is built and read only while holding that ownership.
    explicit ECSMemorySnapshotProvider(flecs::world* a_world)
        : m_hostWorld(a_world)
        , m_memoryQuery(WithWorld(a_world, [](flecs::world& world) {
            return world.query_builder<const MemoryStatisticsComponents::MemSizeBytes>().build();
        })) {
    }

    void BuildSnapshot(std::vector<ResourceMemoryRecord>& out) override {
        out.clear();
        out.reserve(2048);

        auto read = [&](flecs::world&) {
        m_memoryQuery.each([&](flecs::entity e, const MemoryStatisticsComponents::MemSizeBytes& sz) {
            ResourceMemoryRecord row;
            row.bytes = sz.size;

            if (auto rid = e.try_get<MemoryStatisticsComponents::ResourceID>()) {
                row.resourceID = rid->id;
            }
            if (auto rt = e.try_get<MemoryStatisticsComponents::ResourceType>()) {
                row.resourceType = rt->type;
            }
            if (auto rn = e.try_get<MemoryStatisticsComponents::ResourceName>()) {
                row.resourceName = rn->name;
            }
            if (auto heap = e.try_get<MemoryStatisticsComponents::HeapKind>()) {
                row.heapType = heap->type;
            }
            if (auto usage = e.try_get<MemoryStatisticsComponents::ResourceUsage>()) {
                row.usage = usage->usage;
            }
            if (auto owner = e.try_get<MemoryStatisticsComponents::ResourceOwner>()) {
                row.owner = owner->owner;
            }
            if (auto pool = e.try_get<MemoryStatisticsComponents::AliasingPool>(); pool && pool->poolID) {
                row.aliasPoolID = *pool->poolID;
            }
            if (auto placed = e.try_get<MemoryStatisticsComponents::PlacedInAliasPool>()) {
                row.placedInAliasPool = true;
                row.aliased = true;
                if (placed->poolID) row.aliasPoolID = placed->poolID;
            }
            if (auto ident = e.try_get<ResourceIdentifier>()) {
                row.identifier = ident->name;
            }
            if (auto shape = e.try_get<MemoryStatisticsComponents::TextureShape>()) {
                row.width = shape->width;
                row.height = shape->height;
                row.mipLevels = shape->mipLevels;
                row.arraySize = shape->arraySize;
                row.format = shape->format;
                row.aliased = row.aliased || shape->aliased;
            }

            out.push_back(std::move(row));
            });
        };
        if (m_hostWorld)
            read(*m_hostWorld);
        else
            ECSManager::GetInstance().Access(read);

		// Replacement resources and deferred-deletion backings keep the stable
		// owner ResourceID. Metadata applied after materialization may exist only
		// on the current backing. So every record for that owner inherits its
		// semantic ownership. This keeps retired generations from showing as
		// "Unspecified" while they wait for their GPU fence.
		struct OwnerMetadata {
			std::string usage;
			std::string name;
			std::string identifier;
			std::string owner;
		};
		std::unordered_map<std::uint64_t, OwnerMetadata> metadataByOwner;
		metadataByOwner.reserve(out.size());
		for (const auto& row : out) {
			if (row.resourceID == 0) continue;
			auto& metadata = metadataByOwner[row.resourceID];
			if (metadata.usage.empty() && !row.usage.empty()) metadata.usage = row.usage;
			if (metadata.name.empty() && !row.resourceName.empty()) metadata.name = row.resourceName;
			if (metadata.identifier.empty() && !row.identifier.empty()) metadata.identifier = row.identifier;
			if (metadata.owner.empty() && !row.owner.empty()) metadata.owner = row.owner;
		}
		for (auto& row : out) {
			if (row.resourceID == 0) continue;
			const auto found = metadataByOwner.find(row.resourceID);
			if (found == metadataByOwner.end()) continue;
			const auto& metadata = found->second;
			if (row.usage.empty()) row.usage = metadata.usage;
			if (row.resourceName.empty()) row.resourceName = metadata.name;
			if (row.identifier.empty()) row.identifier = metadata.identifier;
			if (row.owner.empty()) row.owner = metadata.owner;
		}
    }

private:
    flecs::world* m_hostWorld;  // null means ORG's tracking world
    flecs::query<const MemoryStatisticsComponents::MemSizeBytes> m_memoryQuery;
};
}

std::shared_ptr<IMemorySnapshotProvider> CreateECSMemorySnapshotProvider() {
    return std::make_shared<ECSMemorySnapshotProvider>(nullptr);
}

std::shared_ptr<IMemorySnapshotProvider> CreateECSMemorySnapshotProvider(flecs::world& world) {
    return std::make_shared<ECSMemorySnapshotProvider>(&world);
}

}
