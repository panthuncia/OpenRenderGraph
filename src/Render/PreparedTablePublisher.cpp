#include "Render/PreparedTablePublisher.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <BasicTelemetry/Tracy.h>
#include <atomic>
#include <spdlog/spdlog.h>

#include "Resources/Buffers/Buffer.h"

namespace org {

uint32_t PreparedTablePublisher::Publish(const FramePreparationContext& preparation, std::span<const std::byte> bytes, uint32_t stride) const {
    if (stride == 0 || bytes.empty() || bytes.size() % stride != 0)
        throw std::invalid_argument("Prepared table '" + m_name + "' needs a non-empty whole number of rows");
    std::lock_guard lock(m_mutex);
    if (!m_table || m_stride != stride || !std::ranges::equal(bytes, m_bytes)) {
        BT_ZONE_SCOPE("ORG.PreparedTable.Publish");
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
        basic_telemetry::AddCounter("ORG.PreparedTable.Publications");
        static std::atomic<uint32_t> reported{0};
        if (reported.fetch_add(1, std::memory_order_relaxed) < 512)
            spdlog::info("Prepared table '{}' published: rows={} srv={} frame={}", m_name, rows, m_index, preparation.frameNumber);
    }
    preparation.Retain(m_table);
    return m_index;
}

} // namespace org
