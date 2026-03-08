#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <cstddef>

#include "Resources/Buffers/Buffer.h"

/// A dedicated pool of Upload-heap buffers for async copy-queue streaming uploads.
/// Separate from UploadManager's pages to avoid thread-safety contention.
///
/// Uses a simple per-frame bump allocator: each frame gets a clean slate of
/// tail offsets (pages are reused across frames once retired).
class AsyncCopyPagePool {
public:
    struct Allocation {
        std::shared_ptr<Resource> buffer;
        size_t offset = 0;
    };

    explicit AsyncCopyPagePool(size_t pageSize = kDefaultPageSize)
        : m_pageSize(pageSize) {
    }

    /// Allocate `size` bytes with the given alignment from the current active page.
    /// Grows by adding new pages as needed.
    /// Thread-safe (internally locked).
    Allocation Allocate(size_t size, size_t alignment = 1) {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_pages.empty()) {
            AddPage(std::max(size, m_pageSize));
        }

        auto* page = &m_pages[m_activePage];
        size_t aligned = AlignUp(page->tailOffset, alignment);

        // If it won't fit, try next page or allocate a new one
        if (aligned + size > page->capacity) {
            ++m_activePage;
            if (m_activePage >= m_pages.size()) {
                AddPage(std::max(size, m_pageSize));
            }
            page = &m_pages[m_activePage];
            page->tailOffset = 0;
            aligned = 0;
        }

        page->tailOffset = aligned + size;
        return { page->buffer, aligned };
    }

    /// Reset all pages for reuse. Call once per frame after the GPU is done
    /// with the previous frame's uploads (tracked externally via fences).
    void ResetForFrame() {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (auto& page : m_pages) {
            page.tailOffset = 0;
        }
        m_activePage = 0;
    }

    /// Release all pages.
    void Cleanup() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_pages.clear();
        m_activePage = 0;
    }

    static constexpr size_t kDefaultPageSize = 64 * 1024 * 1024; // 64 MB

private:
    struct Page {
        std::shared_ptr<Resource> buffer;
        size_t capacity = 0;
        size_t tailOffset = 0;
    };

    void AddPage(size_t minSize) {
        size_t allocSize = std::max(minSize, m_pageSize);
        auto buffer = Buffer::CreateShared(rhi::HeapType::Upload, allocSize, /*uav=*/false);
        buffer->SetName("AsyncCopyPagePool::Page");
        m_pages.push_back({ std::move(buffer), allocSize, 0 });
        m_activePage = m_pages.size() - 1;
    }

    static size_t AlignUp(size_t value, size_t alignment) noexcept {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    size_t m_pageSize;
    std::vector<Page> m_pages;
    size_t m_activePage = 0;
    std::mutex m_mutex;
};
