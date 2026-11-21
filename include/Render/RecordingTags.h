#pragma once

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace org {

// The key of one recording tag. An extension defines its key with static storage and tags graphs under the key's address,
// so two extensions never share a tag. The name is for logs.
struct RecordingTagKey {
    std::string_view name;
};

// The values every recording of a graph carries (RenderGraph::SetRecordingTag, RecordingContext::Tag), one per key. A set is
// immutable once shared: setting a tag makes a new set, so a recording keeps the values it was recorded with.
class RecordingTags {
public:
    // The value under key, null without one.
    const void* Find(const RecordingTagKey& key) const noexcept {
        for (const auto& entry : m_entries)
            if (entry.key == &key) return entry.value.get();
        return nullptr;
    }
    // A copy with key's value set to value. A null value removes the key.
    RecordingTags With(const RecordingTagKey& key, std::shared_ptr<const void> value) const {
        RecordingTags tags;
        tags.m_entries.reserve(m_entries.size() + 1);
        for (const auto& entry : m_entries)
            if (entry.key != &key) tags.m_entries.push_back(entry);
        if (value) tags.m_entries.push_back({&key, std::move(value)});
        return tags;
    }
    bool Empty() const noexcept { return m_entries.empty(); }
    size_t Size() const noexcept { return m_entries.size(); }

private:
    struct Entry {
        const RecordingTagKey* key = nullptr;
        std::shared_ptr<const void> value;
    };
    std::vector<Entry> m_entries;
};

}
