#pragma once

#include <cstddef>
#include <cstdint>
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

// Buffer uploads staged away from the upload service's owner thread. A producer that builds an upload payload
// on a worker writes it straight into upload memory here, and the owner later queues the whole batch with
// IUploadService::SubmitStagedUploads: the owner copies nothing, it only records the copies. A batch that is
// never submitted is simply dropped.
//
// A batch belongs to one producer at a time. The service holds a submitted batch until the GPU is done with
// its frame slot; a producer that keeps batches to reuse their memory resets one only when it holds the last
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

    // Staging for `size` bytes the owner will copy to `target` at `dstOffset`: persistently mapped upload
    // memory, write-combined - write it, never read it back.
    std::byte* Stage(UploadTarget target, size_t dstOffset, size_t size);
    std::byte* Stage(UploadTarget target, size_t dstOffset, const void* data, size_t size);
    // Forgets the entries (and a recording of them) and keeps the pages and the command list, for the next payload.
    void Reset();

    const std::vector<Entry>& Entries() const noexcept { return m_entries; }
    size_t Bytes() const noexcept { return m_bytes; }

    // Producer thread, once the batch is staged: records its copies into a command list of the batch's own (graphics
    // queue), so the owner submits them without recording anything (IUploadService::SubmitRecordedUploads). The list
    // starts with a full barrier against the work before it, and orders overlapping copies with barriers, the later
    // write winning. It names each target's backing as it is now: the recording holds while
    // BufferBase::BackingReleaseSerial() has not moved since (RecordedSerial). False, with nothing recorded, when the
    // batch is empty, a target has no backing, or the list could not be made. Staging after it drops the recording.
    bool Record(rhi::Device device);
    bool Recorded() const noexcept { return m_recorded; }
    uint64_t RecordedSerial() const noexcept { return m_recordedSerial; }
    rhi::CommandList RecordedList() const noexcept;

    // Records a_batches' copies into a_list, in order, with a barrier before a copy that overlaps an earlier one (none in
    // steady state). Every target must be a pointer target with a backing; a copy past a buffer's backing is logged under
    // a_owner. Returns the copies recorded.
    static size_t RecordCopies(rhi::CommandList& a_list, std::span<const std::shared_ptr<StagedUploadBatch>> a_batches, std::string_view a_owner);

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
    // Record: the batch's own command list, made once and recycled with the batch (a producer resets a batch only when it
    // holds the last reference, after the GPU is done with it).
    rhi::CommandAllocatorPtr m_allocator;
    rhi::CommandListPtr m_list;
    bool m_recorded = false;
    uint64_t m_recordedSerial = 0;
};

} // namespace org::runtime
