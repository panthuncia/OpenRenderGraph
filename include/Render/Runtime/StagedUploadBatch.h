#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <rhi.h>

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

    // pageSize: the size of each upload page the batch allocates (a larger entry gets a page of its own).
    static std::shared_ptr<StagedUploadBatch> Create(size_t pageSize = size_t{4} << 20);
    ~StagedUploadBatch();
    StagedUploadBatch(const StagedUploadBatch&) = delete;
    StagedUploadBatch& operator=(const StagedUploadBatch&) = delete;

    // Returns staging for `size` bytes that the owner will copy to `target` at `dstOffset`. It is persistently mapped,
    // write-combined upload memory. Write it, but never read it back.
    std::byte* Stage(UploadTarget target, size_t dstOffset, size_t size);
    std::byte* Stage(UploadTarget target, size_t dstOffset, const void* data, size_t size);
    // Forgets the entries (and a recording of them) and keeps the pages and the command list, for the next payload.
    void Reset();

    const std::vector<Entry>& Entries() const noexcept { return m_entries; }
    size_t Bytes() const noexcept { return m_bytes; }

    // Call on the producer thread, once the batch is staged. Records the batch's copies into a command list that the
    // batch owns (graphics queue). The owner then submits them without recording anything
    // (IUploadService::SubmitRecordedUploads). The list starts with a full barrier against the work before it. It orders
    // overlapping copies with barriers, so the later write wins. It names each target's backing as it is now. So the
    // recording stays valid while BufferBase::BackingReleaseSerial() has not moved since (RecordedSerial).
    // Returns false, with nothing recorded, when the batch is empty, a target has no backing, or the list could not be
    // made. Staging after this call drops the recording.
    bool Record(rhi::Device device);
    bool Recorded() const noexcept { return m_recorded; }
    uint64_t RecordedSerial() const noexcept { return m_recordedSerial; }
    rhi::CommandList RecordedList() const noexcept;

    // Records the copies of a_batches into a_list, in order. A copy that overlaps an earlier copy gets a barrier before
    // it. There are none in steady state. Every target must be a pointer target with a backing. A copy past the end of a
    // buffer's backing is logged under a_owner. Returns the number of copies recorded.
    // a_include is optional. When given, only the entries whose destination backing it accepts are recorded, in their
    // original order. Use it to split the batches' copies between lists by destination.
    static size_t RecordCopies(rhi::CommandList& a_list, std::span<const std::shared_ptr<StagedUploadBatch>> a_batches, std::string_view a_owner,
        const std::function<bool(rhi::ResourceHandle)>& a_include = {});
    // Appends each entry's destination backing, one per entry in recording order. The backing is the target's buffer as
    // it is now. Returns false when a target is not a pointer target with a backing. Then nothing is known about where
    // the batches write.
    static bool AppendDestinations(std::span<const std::shared_ptr<StagedUploadBatch>> a_batches, std::vector<rhi::ResourceHandle>& a_out);

private:
    explicit StagedUploadBatch(size_t pageSize) : m_pageSize(pageSize) {}
    struct Page {
        std::shared_ptr<Buffer> buffer;
        std::byte* mapped = nullptr;
        size_t capacity = 0;
        size_t used = 0;
    };
    std::vector<Page> m_pages;
    size_t m_current = 0;
    std::vector<Entry> m_entries;
    size_t m_bytes = 0;
    size_t m_pageSize;
    // Used by Record: the batch's own command list. It is made once and recycled with the batch. A producer resets a
    // batch only when it holds the last reference, after the GPU is done with it.
    rhi::CommandAllocatorPtr m_allocator;
    rhi::CommandListPtr m_list;
    bool m_recorded = false;
    uint64_t m_recordedSerial = 0;
};

} // namespace org::runtime
