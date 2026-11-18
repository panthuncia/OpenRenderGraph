#pragma once

#include <algorithm>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "Resources/ResourceIdentifier.h"

namespace org {

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

} // namespace org

namespace std {
    template<>
    struct hash<org::FeatureDomainIdentifier> {
        size_t operator()(const org::FeatureDomainIdentifier& id) const noexcept {
            return org::FeatureDomainIdentifier::Hasher{}(id);
        }
    };
}

namespace org {

class FeatureDomainRegistry {
public:
    struct ResourceDomainMapping {
        ResourceIdentifier resourcePrefix;
        FeatureDomainIdentifier domain;
    };

    static FeatureDomainRegistry& Get();

    void RegisterResourceDomain(const ResourceIdentifier& resourcePrefix, const FeatureDomainIdentifier& domain);
    std::optional<FeatureDomainIdentifier> FindResourceDomain(const ResourceIdentifier& resourceId) const;

private:
    FeatureDomainRegistry();

    mutable std::mutex m_mutex;
    std::vector<ResourceDomainMapping> m_resourceDomainMappings;
};

} // namespace org
