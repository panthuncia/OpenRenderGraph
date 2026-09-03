#include "Render/MemoryIntrospectionBackend.h"

#include "Render/MemoryIntrospectionAPI.h"
#include "Managers/Singletons/ECSManager.h"
#include "Resources/MemoryStatisticsComponents.h"
#include "Resources/ResourceIdentifier.h"

#include <unordered_map>

namespace org::memory {

namespace {
class ECSMemorySnapshotProvider final : public IMemorySnapshotProvider {
public:
    explicit ECSMemorySnapshotProvider(flecs::world& world)
        : m_memoryQuery(world.query_builder<const MemoryStatisticsComponents::MemSizeBytes>().build()) {
    }

    void BuildSnapshot(std::vector<ResourceMemoryRecord>& out) override {
        out.clear();
        out.reserve(2048);

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
            if (auto usage = e.try_get<MemoryStatisticsComponents::ResourceUsage>()) {
                row.usage = usage->usage;
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
                row.aliased = shape->aliased;
            }

            out.push_back(std::move(row));
            });

		// Replacement resources and deferred-deletion backings retain the stable
		// owner ResourceID, but metadata applied after materialization can exist on
		// only the current backing. Inherit semantic ownership across all records
		// for that owner so retired generations do not become "Unspecified" while
		// they wait for their GPU fence.
		struct OwnerMetadata {
			std::string usage;
			std::string name;
			std::string identifier;
		};
		std::unordered_map<std::uint64_t, OwnerMetadata> metadataByOwner;
		metadataByOwner.reserve(out.size());
		for (const auto& row : out) {
			if (row.resourceID == 0) continue;
			auto& metadata = metadataByOwner[row.resourceID];
			if (metadata.usage.empty() && !row.usage.empty()) metadata.usage = row.usage;
			if (metadata.name.empty() && !row.resourceName.empty()) metadata.name = row.resourceName;
			if (metadata.identifier.empty() && !row.identifier.empty()) metadata.identifier = row.identifier;
		}
		for (auto& row : out) {
			if (row.resourceID == 0) continue;
			const auto found = metadataByOwner.find(row.resourceID);
			if (found == metadataByOwner.end()) continue;
			const auto& metadata = found->second;
			if (row.usage.empty()) row.usage = metadata.usage;
			if (row.resourceName.empty()) row.resourceName = metadata.name;
			if (row.identifier.empty()) row.identifier = metadata.identifier;
		}
    }

private:
    flecs::query<const MemoryStatisticsComponents::MemSizeBytes> m_memoryQuery;
};
}

std::shared_ptr<IMemorySnapshotProvider> CreateECSMemorySnapshotProvider() {
    auto& world = ECSManager::GetInstance().GetWorld();
    return CreateECSMemorySnapshotProvider(world);
}

std::shared_ptr<IMemorySnapshotProvider> CreateECSMemorySnapshotProvider(flecs::world& world) {
    return std::make_shared<ECSMemorySnapshotProvider>(world);
}

}
