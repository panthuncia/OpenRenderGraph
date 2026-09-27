#include "Render/Runtime/StagedUploadBatch.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "Resources/Buffers/Buffer.h"

namespace org::runtime {

std::shared_ptr<StagedUploadBatch> StagedUploadBatch::Create(size_t pageSize) {
    return std::shared_ptr<StagedUploadBatch>(new StagedUploadBatch(pageSize ? pageSize : size_t{4} << 20));
}

StagedUploadBatch::~StagedUploadBatch() {
    for (auto& page : m_pages)
        if (page.buffer && page.mapped) page.buffer->GetAPIResource().Unmap(0, page.capacity);
}

std::byte* StagedUploadBatch::Stage(UploadTarget target, size_t dstOffset, size_t size) {
    if (!size) return nullptr;
    // Copies want 16-byte aligned sources for every backend's fast path.
    const auto aligned = (size + 15) & ~size_t{15};
    while (m_current < m_pages.size() && m_pages[m_current].capacity - m_pages[m_current].used < aligned) ++m_current;
    if (m_current == m_pages.size()) {
        Page page;
        page.capacity = (std::max)(m_pageSize, aligned);
        page.buffer = Buffer::CreateShared(rhi::HeapType::Upload, page.capacity, false);
        page.buffer->SetName("org.staged-uploads");
        void* mapped = nullptr;
        page.buffer->GetAPIResource().Map(&mapped, 0, page.capacity);
        if (!mapped) throw std::runtime_error("Staged upload page could not be mapped");
        page.mapped = static_cast<std::byte*>(mapped);
        m_pages.push_back(std::move(page));
    }
    auto& page = m_pages[m_current];
    const size_t offset = page.used;
    page.used += aligned;
    // Consecutive array/journal ranges are commonly staged one element at a time. Keep them as one copy when
    // both source and destination are contiguous; this substantially reduces command-recording work without
    // changing ordering or merging across the alignment padding between non-aligned writes.
    if (!m_entries.empty()) {
        auto& previous = m_entries.back();
        if (previous.target == target && previous.page == page.buffer &&
            previous.dstOffset + previous.size == dstOffset && previous.pageOffset + previous.size == offset) {
            previous.size += size;
        } else {
            m_entries.push_back({std::move(target), dstOffset, size, page.buffer, offset});
        }
    } else {
        m_entries.push_back({std::move(target), dstOffset, size, page.buffer, offset});
    }
    m_bytes += size;
    return page.mapped + offset;
}

std::byte* StagedUploadBatch::Stage(UploadTarget target, size_t dstOffset, const void* data, size_t size) {
    auto* staging = Stage(std::move(target), dstOffset, size);
    if (staging) std::memcpy(staging, data, size);
    return staging;
}

void StagedUploadBatch::Reset() {
    for (auto& page : m_pages) page.used = 0;
    m_current = 0;
    m_entries.clear();
    m_bytes = 0;
}

} // namespace org::runtime
