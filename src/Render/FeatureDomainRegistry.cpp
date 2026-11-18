#include "Render/FeatureDomainRegistry.h"

#include <algorithm>

namespace org {

FeatureDomainRegistry& FeatureDomainRegistry::Get() {
    static FeatureDomainRegistry instance;
    return instance;
}

FeatureDomainRegistry::FeatureDomainRegistry() {
    RegisterResourceDomain(ResourceIdentifier{ "Builtin::Shadows" }, FeatureDomainIdentifier{ "Builtin::FeatureDomains::Shadows" });
    RegisterResourceDomain(ResourceIdentifier{ "Builtin::GTAO" }, FeatureDomainIdentifier{ "Builtin::FeatureDomains::GTAO" });
}

void FeatureDomainRegistry::RegisterResourceDomain(const ResourceIdentifier& resourcePrefix, const FeatureDomainIdentifier& domain) {
    std::scoped_lock lock(m_mutex);

    auto existing = std::find_if(
        m_resourceDomainMappings.begin(),
        m_resourceDomainMappings.end(),
        [&](const ResourceDomainMapping& entry) {
            return entry.resourcePrefix == resourcePrefix;
        });

    if (existing != m_resourceDomainMappings.end()) {
        existing->domain = domain;
    }
    else {
        m_resourceDomainMappings.push_back(ResourceDomainMapping{ resourcePrefix, domain });
    }
}

std::optional<FeatureDomainIdentifier> FeatureDomainRegistry::FindResourceDomain(const ResourceIdentifier& resourceId) const {
    std::scoped_lock lock(m_mutex);

    const ResourceDomainMapping* bestMatch = nullptr;
    for (const auto& entry : m_resourceDomainMappings) {
        if (!(resourceId == entry.resourcePrefix || resourceId.hasPrefix(entry.resourcePrefix))) {
            continue;
        }

        if (!bestMatch || entry.resourcePrefix.segments.size() > bestMatch->resourcePrefix.segments.size()) {
            bestMatch = &entry;
        }
    }

    if (!bestMatch) {
        return std::nullopt;
    }

    return bestMatch->domain;
}

} // namespace org
