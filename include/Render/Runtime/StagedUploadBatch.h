#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <rhi.h>
#include <rhi_helpers.h>

#include "Render/Runtime/UploadTypes.h"

namespace org {
class Buffer;
}

namespace org::runtime {

// Buffer uploads staged away from the upload service's owner thread. A producer that builds an upload payload on a
// worker writes it straight into upload memory here. Later, the owner queues the whole batch with
// IUploadService::SubmitStagedUploads. The owner copies nothing. It only records the copies. A batch that is never
// submitted is simply dropped.
//
// A batch belongs to one producer at a time. The service holds a submitted batch until the GPU is done with its
// frame slot. A producer that keeps batches to reuse their memory resets a batch only when it holds the last
// reference (use_count() == 1).
class StagedUploadBatch {
public:
    struct Entry {
        UploadTarget target;
        size_t dstOffset = 0;
        size_t size = 0;
        std::shared_ptr<Buffer> page;
        size_t pageOffset = 0;
    };

    // A texture's contents, for a texture no queue has used yet. Its copies leave it in the Common layout. The host that
    // records them first seeds its state ledger with Common for the texture (GraphHost), so the texture's first execution
    // starts from the state the copies leave.
    struct TextureCopy {
        uint32_t mip = 0, slice = 0, z = 0;
        rhi::CopyableFootprint footprint{};  // offset is in the page
    };
    struct TextureEntry {
        UploadTarget target;
        std::shared_ptr<Buffer> page;
        uint32_t mips = 1, layers = 1;  // the subresources the copies' barriers cover
        std::vector<TextureCopy> copies;
    };

    // pageSize: the size of each upload page the batch allocates (a larger entry gets a page of its own).
    static std::shared_ptr<StagedUploadBatch> Create(size_t pageSize = size_t{4} << 20);
    ~StagedUploadBatch();
    StagedUploadBatch(const StagedUploadBatch&) = delete;
    StagedUploadBatch& operator=(const StagedUploadBatch&) = delete;

    // Returns staging for `size` bytes that the owner will copy to `target` at `dstOffset`. It is persistently mapped,
    // write-combined upload memory. Write it, but never read it back.
    std::byte* Stage(UploadTarget target, size_t dstOffset, size_t size);
    std::byte* Stage(UploadTarget target, size_t dstOffset, const void* data, size_t size);
    // Stages every given subresource of a fresh texture (see TextureEntry). The arguments are those of
    // rhi::helpers::PlanTextureUploadSubresources. False when the format has no copy layout or nothing was given.
    bool StageTexture(UploadTarget target, rhi::Format format, uint32_t width, uint32_t height, uint32_t depthOrLayers, uint32_t mips,
        uint32_t arraySize, std::span<const rhi::helpers::SubresourceData> subresources);
    // Forgets the entries and keeps the pages, for the next payload.
    void Reset();

    const std::vector<Entry>& Entries() const noexcept { return m_entries; }
    const std::vector<TextureEntry>& Textures() const noexcept { return m_textures; }
    bool Empty() const noexcept { return m_entries.empty() && m_textures.empty(); }
    size_t Bytes() const noexcept { return m_bytes; }

    // Records the copies of a_batches into a_list, in order. A copy that overlaps an earlier copy gets a barrier before
    // it. There are none in steady state. Every target must be a pointer target with a backing. A copy past the end of a
    // buffer's backing is logged under a_owner. Returns the number of copies recorded.
    // Texture entries follow the buffer copies: every texture goes from Undefined to CopyDest, takes its copies, and goes on to
    // Common.
    // a_include is optional. When given, only the entries whose destination backing it accepts are recorded, in their
    // original order. Use it to split the batches' copies between lists by destination. Batches with textures cannot be split.
    static size_t RecordCopies(rhi::CommandList& a_list, std::span<const std::shared_ptr<StagedUploadBatch>> a_batches, std::string_view a_owner,
        const std::function<bool(rhi::ResourceHandle)>& a_include = {});
    // Appends each entry's destination backing, one per entry in recording order. The backing is the target's buffer as
    // it is now. Returns false when a target is not a pointer target with a backing, or a batch holds a texture. Then nothing
    // is known about where the batches write, and the copies go on one list.
    static bool AppendDestinations(std::span<const std::shared_ptr<StagedUploadBatch>> a_batches, std::vector<rhi::ResourceHandle>& a_out);

private:
    explicit StagedUploadBatch(size_t pageSize) : m_pageSize(pageSize) {}
    struct Page {
        std::shared_ptr<Buffer> buffer;
        std::byte* mapped = nullptr;
        size_t capacity = 0;
        size_t used = 0;
    };
    // `size` bytes at an `alignment` multiple in a page, and the page.
    std::byte* Allocate(size_t size, size_t alignment, std::shared_ptr<Buffer>& outPage, size_t& outOffset);
    std::vector<Page> m_pages;
    size_t m_current = 0;
    std::vector<Entry> m_entries;
    std::vector<TextureEntry> m_textures;
    size_t m_bytes = 0;
    size_t m_pageSize;
};

} // namespace org::runtime
