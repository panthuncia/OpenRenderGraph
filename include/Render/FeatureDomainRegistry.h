#pragma once

#include <algorithm>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "Resources/ResourceIdentifier.h"

struct FeatureDomainIdentifier {
    std::vector<std::string> segments;
    size_t hash = 0;
    std::string name;

    FeatureDomainIdentifier() = default;

    FeatureDomainIdentifier(std::string_view s) {
        size_t start = 0;
        while (start < s.size()) {
            auto pos = s.find("::", start);
            if (pos == std::string_view::npos) {
                pos = s.size();
            }
            segments.emplace_back(s.substr(start, pos - start));
            start = pos + 2;
        }
        hash = Hasher{}(*this);
        name = s;
    }

    FeatureDomainIdentifier(const std::string& s) : FeatureDomainIdentifier(std::string_view(s)) {}
    FeatureDomainIdentifier(const char* s) : FeatureDomainIdentifier(std::string_view{ s }) {}

    std::string ToString() const {
        std::string out;
        for (size_t i = 0; i < segments.size(); ++i) {
            if (i) {
                out += "::";
            }
            out += segments[i];
        }
        return out;
    }

    bool operator==(const FeatureDomainIdentifier& other) const noexcept {
        return segments == other.segments;
    }

    struct Hasher {
        size_t operator()(const FeatureDomainIdentifier& id) const noexcept {
            size_t value = 0;
            for (const auto& seg : id.segments) {
                value = value * 31 + std::hash<std::string>()(seg);
            }
            return value;
        }
    };
};

namespace std {
    template<>
    struct hash<FeatureDomainIdentifier> {
        size_t operator()(const FeatureDomainIdentifier& id) const noexcept {
            return FeatureDomainIdentifier::Hasher{}(id);
        }
    };
}

class FeatureDomainRegistry {
public:
    struct ResourceDomainMapping {
        ResourceIdentifier resourcePrefix;
        FeatureDomainIdentifier domain;
    };

    static FeatureDomainRegistry& Get() {
        static FeatureDomainRegistry instance;
        return instance;
    }

    void RegisterResourceDomain(const ResourceIdentifier& resourcePrefix, const FeatureDomainIdentifier& domain) {
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

    std::optional<FeatureDomainIdentifier> FindResourceDomain(const ResourceIdentifier& resourceId) const {
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

private:
    FeatureDomainRegistry() {
        RegisterResourceDomain(ResourceIdentifier{ "Builtin::Shadows" }, FeatureDomainIdentifier{ "Builtin::FeatureDomains::Shadows" });
        RegisterResourceDomain(ResourceIdentifier{ "Builtin::GTAO" }, FeatureDomainIdentifier{ "Builtin::FeatureDomains::GTAO" });
    }

    mutable std::mutex m_mutex;
    std::vector<ResourceDomainMapping> m_resourceDomainMappings;
};