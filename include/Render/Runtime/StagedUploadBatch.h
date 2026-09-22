#pragma once

#include <cstddef>
#include <memory>
#include <vector>

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
    // Forgets the entries and keeps the pages, for the next payload.
    void Reset();

    const std::vector<Entry>& Entries() const noexcept { return m_entries; }
    size_t Bytes() const noexcept { return m_bytes; }

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
};

} // namespace org::runtime
