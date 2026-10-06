#include "Render/Runtime/StagedUploadBatch.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <unordered_map>

#include <rhi_helpers.h>
#include <spdlog/spdlog.h>
#include <BasicTelemetry/Tracy.h>

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
    m_recorded = false;  // the recording no longer covers every entry
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
    m_recorded = false;
}

rhi::CommandList StagedUploadBatch::RecordedList() const noexcept {
    return m_recorded && m_list ? m_list.Get() : rhi::CommandList{};
}

bool StagedUploadBatch::Record(rhi::Device device) {
    BT_ZONE_SCOPE("ORG.Upload.RecordBatch");
    m_recorded = false;
    if (m_entries.empty() || !device) return false;
    // Before resolving any target: a release after this point moves the serial, and the owner then refuses the list.
    const uint64_t serial = BufferBase::BackingReleaseSerial();
    // A target without a backing (GetAPIResource throws) leaves the batch to the owner's recording.
    try {
        for (const auto& entry : m_entries)
            if (entry.target.kind != UploadTarget::Kind::PinnedShared || !entry.target.pinned || !entry.page ||
                !entry.target.pinned->GetAPIResource().GetHandle().valid())
                return false;
    } catch (const std::exception&) {
        return false;
    }
    if (!m_list) {
        if (!rhi::IsOk(device.CreateCommandAllocator(rhi::QueueKind::Graphics, m_allocator)) ||
            !rhi::IsOk(device.CreateCommandList(rhi::QueueKind::Graphics, m_allocator.Get(), m_list))) {
            m_allocator = {};
            m_list = {};
            return false;
        }
    } else {
        m_allocator->Recycle();
        m_list->Recycle(m_allocator.Get());
    }
    auto list = m_list.Get();
    // The copies overwrite buffers the work before them may still read.
    const auto full = rhi::FullMemoryBarrier();
    rhi::BarrierBatch barriers{};
    barriers.globals = {&full, 1u};
    list.Barriers(barriers);
    const std::shared_ptr<StagedUploadBatch> self(std::shared_ptr<StagedUploadBatch>{}, this);  // non-owning, for the span
    RecordCopies(list, {&self, 1}, "a recorded batch");
    list.End();
    m_recordedSerial = serial;
    m_recorded = true;
    return true;
}

size_t StagedUploadBatch::RecordCopies(rhi::CommandList& a_list, std::span<const std::shared_ptr<StagedUploadBatch>> a_batches, std::string_view a_owner) {
    BT_ZONE_SCOPE("ORG.Upload.RecordCopies");
    size_t copies = 0;
    // Copies in one list are unordered without a barrier, and batches can overlap (a producer's batch whose submission
    // never went out, then its replacement): the later write has to win. Per backing, the ranges written since the last
    // barrier, disjoint (begin -> end); an overlap takes a barrier and starts them again.
    std::unordered_map<uint64_t, std::map<uint64_t, uint64_t>> written;
    written.reserve(32);
    auto ordered = [&] {
        const auto full = rhi::FullMemoryBarrier();
        rhi::BarrierBatch barriers{};
        barriers.globals = {&full, 1u};
        a_list.Barriers(barriers);
        written.clear();
    };
    for (const auto& batch : a_batches) {
        if (!batch) continue;
        for (const auto& entry : batch->Entries()) {
            if (entry.target.kind != UploadTarget::Kind::PinnedShared || !entry.target.pinned || !entry.page)
                throw std::logic_error("Recorded staged uploads need pointer targets");
            const auto target = entry.target.pinned->GetAPIResource().GetHandle();
            const uint64_t begin = entry.dstOffset, end = entry.dstOffset + entry.size;
            const uint64_t key = (uint64_t{target.generation} << 32) | target.index;
            auto* ranges = &written[key];
            auto after = ranges->upper_bound(begin);
            const bool overlaps = (after != ranges->end() && after->first < end) || (after != ranges->begin() && std::prev(after)->second > begin);
            if (overlaps) {
                ordered();
                ranges = &written[key];
            }
            ranges->emplace(begin, end);
            // A copy past the target's current backing is the producer's defect (staged against another size): the RHI
            // rejects it and the submission fails, so name it here, where the target's name is known.
            if (const auto* buffer = dynamic_cast<const Buffer*>(entry.target.pinned.get()); buffer && end > buffer->GetSize())
                spdlog::error("{}: staged upload of {} bytes at {} into '{}' is past its {} bytes", a_owner, entry.size, entry.dstOffset,
                    buffer->GetName(), buffer->GetSize());
            a_list.CopyBufferRegion(target, entry.dstOffset, entry.page->GetAPIResource().GetHandle(), entry.pageOffset, entry.size);
            ++copies;
        }
    }
    return copies;
}

} // namespace org::runtime
