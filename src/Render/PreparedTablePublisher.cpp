#include "Render/PreparedTablePublisher.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

#include <BasicTelemetry/Tracy.h>
#include <atomic>
#include <spdlog/spdlog.h>

#include "Resources/Buffers/Buffer.h"

namespace org {

namespace {
// Published tables by name and contents. A table never changes once published, so a publisher whose bytes another one published
// (the same pass of a newer graph after a rebuild, with the same view tables) binds that table instead of creating one: an upload
// buffer and its view cost about 0.1 ms each, and a new graph's first preparation published dozens.
struct PublishedTable {
    std::string name;
    uint32_t stride = 0;
    std::vector<std::byte> bytes;
    std::weak_ptr<Buffer> table;
    uint32_t index = 0;
};
std::mutex g_publishedMutex;
std::unordered_multimap<uint64_t, PublishedTable> g_published;

uint64_t TableKey(const std::string& name, uint32_t stride, std::span<const std::byte> bytes) {
    uint64_t h = 0xcbf29ce484222325ull;
    const auto mix = [&](uint64_t value) {
        h ^= value;
        h *= 0x100000001b3ull;
    };
    mix(std::hash<std::string>{}(name));
    mix(stride);
    for (const auto byte : bytes) mix(static_cast<uint64_t>(byte));
    return h;
}
}  // namespace

uint32_t PreparedTablePublisher::Publish(const FramePreparationContext& preparation, std::span<const std::byte> bytes, uint32_t stride) const {
    if (stride == 0 || bytes.empty() || bytes.size() % stride != 0)
        throw std::invalid_argument("Prepared table '" + m_name + "' needs a non-empty whole number of rows");
    std::lock_guard lock(m_mutex);
    if (!m_table || m_stride != stride || !std::ranges::equal(bytes, m_bytes)) {
        BT_ZONE_SCOPE("ORG.PreparedTable.Publish");
        const auto key = TableKey(m_name, stride, bytes);
        {
            std::lock_guard published(g_publishedMutex);
            const auto [first, last] = g_published.equal_range(key);
            for (auto it = first; it != last; ++it) {
                const auto& entry = it->second;
                if (entry.name != m_name || entry.stride != stride || !std::ranges::equal(bytes, entry.bytes)) continue;
                auto table = entry.table.lock();
                if (!table) continue;
                m_table = std::move(table);
                m_index = entry.index;
                m_bytes.assign(bytes.begin(), bytes.end());
                m_stride = stride;
                basic_telemetry::AddCounter("ORG.PreparedTable.Reused");
                preparation.Retain(m_table);
                return m_index;
            }
        }
        const auto rows = static_cast<uint32_t>(bytes.size() / stride);
        auto table = Buffer::CreateUnmaterializedStructuredBuffer(rows, stride, false, false, false, rhi::HeapType::Upload);
        table->SetName(m_name);
        table->Materialize();
        void* mapped = nullptr;
        table->GetAPIResource().Map(&mapped, 0, 0);
        if (!mapped) throw std::runtime_error("Prepared table '" + m_name + "' could not be mapped");
        std::memcpy(mapped, bytes.data(), bytes.size());
        table->GetAPIResource().Unmap(0, bytes.size());
        m_index = table->GetSRVInfo(0).slot.index;
        m_table = std::move(table);
        m_bytes.assign(bytes.begin(), bytes.end());
        m_stride = stride;
        {
            std::lock_guard published(g_publishedMutex);
            // Entries whose tables are gone are dropped now and then.
            if (g_published.size() > 256) std::erase_if(g_published, [](const auto& entry) { return entry.second.table.expired(); });
            g_published.emplace(key, PublishedTable{m_name, stride, m_bytes, m_table, m_index});
        }
        basic_telemetry::AddCounter("ORG.PreparedTable.Publications");
        static std::atomic<uint32_t> reported{0};
        if (reported.fetch_add(1, std::memory_order_relaxed) < 512)
            spdlog::info("Prepared table '{}' published: rows={} srv={} frame={}", m_name, rows, m_index, preparation.frameNumber);
    }
    preparation.Retain(m_table);
    return m_index;
}

} // namespace org
