#include "Render/Runtime/StagedUploadBatch.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

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

std::byte* StagedUploadBatch::Allocate(size_t size, size_t alignment, std::shared_ptr<Buffer>& outPage, size_t& outOffset) {
    const auto alignUp = [alignment](size_t value) { return (value + alignment - 1) / alignment * alignment; };
    while (m_current < m_pages.size() && alignUp(m_pages[m_current].used) + size > m_pages[m_current].capacity) ++m_current;
    if (m_current == m_pages.size()) {
        Page page;
        page.capacity = (std::max)(m_pageSize, size);
        page.buffer = Buffer::CreateShared(rhi::HeapType::Upload, page.capacity, false);
        page.buffer->SetName("org.staged-uploads");
        void* mapped = nullptr;
        page.buffer->GetAPIResource().Map(&mapped, 0, page.capacity);
        if (!mapped) throw std::runtime_error("Staged upload page could not be mapped");
        page.mapped = static_cast<std::byte*>(mapped);
        m_pages.push_back(std::move(page));
    }
    auto& page = m_pages[m_current];
    outOffset = alignUp(page.used);
    page.used = outOffset + size;
    outPage = page.buffer;
    return page.mapped + outOffset;
}

std::byte* StagedUploadBatch::Stage(UploadTarget target, size_t dstOffset, size_t size) {
    if (!size) return nullptr;
    // Copies want 16-byte aligned sources for every backend's fast path.
    const auto aligned = (size + 15) & ~size_t{15};
    std::shared_ptr<Buffer> pageBuffer;
    size_t offset = 0;
    std::byte* const staging = Allocate(aligned, 16, pageBuffer, offset);
    auto& page = m_pages[m_current];
    // Consecutive array or journal ranges are often staged one element at a time. Keep them as one copy when
    // both the source and the destination are contiguous. This greatly reduces command-recording work. It does
    // not change ordering, and it never merges across the alignment padding between non-aligned writes.
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
    return staging;
}

bool StagedUploadBatch::StageTexture(UploadTarget target, rhi::Format format, uint32_t width, uint32_t height, uint32_t depthOrLayers,
    uint32_t mips, uint32_t arraySize, std::span<const rhi::helpers::SubresourceData> subresources) {
    if (subresources.empty()) return false;
    const rhi::Span<const rhi::helpers::SubresourceData> sources{subresources.data(), static_cast<uint32_t>(subresources.size())};
    const auto plan = rhi::helpers::PlanTextureUploadSubresources(format, width, height, depthOrLayers, mips, arraySize, sources);
    if (plan.totalSize == 0 || plan.footprints.empty()) return false;
    // Placed footprints start at 512-byte multiples (D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT).
    TextureEntry entry;
    size_t offset = 0;
    std::byte* const staging = Allocate(static_cast<size_t>(plan.totalSize), 512, entry.page, offset);
    rhi::helpers::WriteTextureUploadSubresources(plan, sources, staging, 0);
    entry.target = std::move(target);
    entry.mips = (std::max)(mips, 1u);
    entry.layers = (std::max)(arraySize, 1u);
    entry.copies.reserve(plan.footprints.size());
    for (const auto& footprint : plan.footprints)
        entry.copies.push_back({footprint.mip, footprint.arraySlice, footprint.zSlice,
            {offset + footprint.offset, footprint.rowPitch, footprint.width, footprint.height, footprint.depth}});
    m_textures.push_back(std::move(entry));
    m_bytes += static_cast<size_t>(plan.totalSize);
    return true;
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
    m_textures.clear();
    m_bytes = 0;
}

bool StagedUploadBatch::AppendDestinations(std::span<const std::shared_ptr<StagedUploadBatch>> a_batches, std::vector<rhi::ResourceHandle>& a_out) {
    try {
        for (const auto& batch : a_batches) {
            if (!batch) continue;
            if (!batch->Textures().empty()) return false;
            for (const auto& entry : batch->Entries()) {
                if (entry.target.kind != UploadTarget::Kind::PinnedShared || !entry.target.pinned || !entry.page) return false;
                const auto target = entry.target.pinned->GetAPIResource().GetHandle();
                if (!target.valid()) return false;
                a_out.push_back(target);
            }
        }
    } catch (const std::exception&) {
        return false;  // a target without a backing
    }
    return true;
}

size_t StagedUploadBatch::RecordCopies(rhi::CommandList& a_list, std::span<const std::shared_ptr<StagedUploadBatch>> a_batches, std::string_view a_owner,
    const std::function<bool(rhi::ResourceHandle)>& a_include) {
    BT_ZONE_SCOPE("ORG.Upload.RecordCopies");
    size_t copies = 0;
    // Copies in one list are unordered without a barrier. Batches can also overlap. For example, a producer's batch
    // that was never submitted can be followed by its replacement. The later write has to win. So for each backing,
    // track the ranges written since the last barrier, kept disjoint as begin -> end. An overlap records a barrier and
    // starts tracking again.
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
            if (a_include && !a_include(target)) continue;
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
            // A copy past the end of the target's current backing is a bug in the producer, which staged it against another
            // size. The RHI rejects it and the submission fails. So report it here, where the target's name is known.
            if (const auto* buffer = dynamic_cast<const Buffer*>(entry.target.pinned.get()); buffer && end > buffer->GetSize())
                spdlog::error("{}: staged upload of {} bytes at {} into '{}' is past its {} bytes", a_owner, entry.size, entry.dstOffset,
                    buffer->GetName(), buffer->GetSize());
            a_list.CopyBufferRegion(target, entry.dstOffset, entry.page->GetAPIResource().GetHandle(), entry.pageOffset, entry.size);
            ++copies;
        }
    }
    // Fresh textures: Undefined to CopyDest once each, every copy in order, then Common. A subresource written again is
    // ordered behind its earlier write.
    std::vector<rhi::TextureBarrier> toCopy, toCommon;
    std::unordered_map<uint64_t, size_t> seen;
    for (const auto& batch : a_batches) {
        if (!batch) continue;
        for (const auto& entry : batch->Textures()) {
            if (entry.target.kind != UploadTarget::Kind::PinnedShared || !entry.target.pinned || !entry.page)
                throw std::logic_error("Recorded staged uploads need pointer targets");
            if (a_include) throw std::logic_error("Staged texture uploads cannot be split between lists");
            const auto texture = entry.target.pinned->GetAPIResource().GetHandle();
            const uint64_t key = (uint64_t{texture.generation} << 32) | texture.index;
            if (seen.contains(key)) continue;
            seen.emplace(key, toCopy.size());
            rhi::TextureBarrier barrier{};
            barrier.texture = texture;
            barrier.range = {0, entry.mips, 0, entry.layers};
            barrier.beforeSync = rhi::ResourceSyncState::None;
            barrier.afterSync = rhi::ResourceSyncState::Copy;
            barrier.beforeAccess = rhi::ResourceAccessType::None;
            barrier.afterAccess = rhi::ResourceAccessType::CopyDest;
            barrier.beforeLayout = rhi::ResourceLayout::Undefined;
            barrier.afterLayout = rhi::ResourceLayout::CopyDest;
            barrier.discard = true;
            toCopy.push_back(barrier);
            barrier.beforeSync = rhi::ResourceSyncState::Copy;
            barrier.afterSync = rhi::ResourceSyncState::All;
            barrier.beforeAccess = rhi::ResourceAccessType::CopyDest;
            barrier.afterAccess = rhi::ResourceAccessType::Common;
            barrier.beforeLayout = rhi::ResourceLayout::CopyDest;
            barrier.afterLayout = rhi::ResourceLayout::Common;
            barrier.discard = false;
            toCommon.push_back(barrier);
        }
    }
    if (toCopy.empty()) return copies;
    a_list.Barriers({.textures = {toCopy.data(), static_cast<uint32_t>(toCopy.size())}});
    std::unordered_set<uint64_t> textureCells;
    for (const auto& batch : a_batches) {
        if (!batch) continue;
        for (const auto& entry : batch->Textures()) {
            const auto texture = entry.target.pinned->GetAPIResource().GetHandle();
            const auto page = entry.page->GetAPIResource().GetHandle();
            for (const auto& copy : entry.copies) {
                const uint64_t cell = ((uint64_t{texture.index} * 4099 + copy.mip) * 8191 + copy.slice) * 8209 + copy.z;
                if (!textureCells.insert(cell).second) {
                    const auto full = rhi::FullMemoryBarrier();
                    a_list.Barriers({.globals = {&full, 1u}});
                    textureCells.clear();
                    textureCells.insert(cell);
                }
                rhi::BufferTextureCopyFootprint region{};
                region.texture = texture;
                region.buffer = page;
                region.mip = copy.mip;
                region.arraySlice = copy.slice;
                region.z = copy.z;
                region.footprint = copy.footprint;
                a_list.CopyBufferToTexture(region);
                ++copies;
            }
        }
    }
    a_list.Barriers({.textures = {toCommon.data(), static_cast<uint32_t>(toCommon.size())}});
    return copies;
}

} // namespace org::runtime
