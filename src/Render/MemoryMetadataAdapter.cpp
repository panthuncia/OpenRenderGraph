#include "Interfaces/IHasMemoryMetadata.h"

#include "Resources/MemoryStatisticsComponents.h"
#include "Resources/ResourceIdentifier.h"


namespace org {

void IHasMemoryMetadata::SetMemoryUsageHint(std::string usage) {
    EntityComponentBundle bundle;
    bundle.Set<MemoryStatisticsComponents::ResourceUsage>({ std::move(usage) });
    ApplyMetadataComponentBundle(bundle);
}

void IHasMemoryMetadata::SetMemoryIdentifier(std::string identifier) {
    EntityComponentBundle bundle;
    bundle.Set<ResourceIdentifier>({ std::move(identifier) });
    ApplyMetadataComponentBundle(bundle);
}


} // namespace org
