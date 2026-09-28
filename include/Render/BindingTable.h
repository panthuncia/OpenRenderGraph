#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include "Render/OwnedDescriptorBinding.h"
#include "Render/Runtime/ResourceCleanupQueue.h"

namespace org {
class PublicationBindingBundle;
class BindingRecordBuilder;
class BindingTableBuilder;

class BindingRecord {
public:
    std::span<const std::byte> Bytes() const noexcept { return m_bytes; }
private:
    friend class BindingRecordBuilder;
    struct Field { size_t offset; OwnedDescriptorBinding binding; };
    std::vector<std::byte> m_bytes;
    std::vector<Field> m_fields;
};

// The only path for writing descriptor fields also records their exact owner.
// Non-descriptor bytes may be written before binding fields are assigned.
class BindingRecordBuilder {
public:
    BindingRecordBuilder(std::shared_ptr<runtime::ResourceCleanupQueue> cleanup, size_t bytes);
    BindingRecordBuilder(std::shared_ptr<runtime::ResourceCleanupQueue> cleanup, const BindingRecord& base);
    BindingRecordBuilder(const BindingRecordBuilder&) = delete;
    BindingRecordBuilder& operator=(const BindingRecordBuilder&) = delete;
    void WriteBytes(size_t offset, std::span<const std::byte> bytes);
    void WriteBinding(size_t offset, OwnedDescriptorBinding binding);
    std::shared_ptr<const BindingRecord> Seal();
private:
    std::shared_ptr<BindingRecord> m_record;
};

class BindingTableVersion {
public:
    static constexpr size_t PageSize = 64;
    using Record = std::shared_ptr<const BindingRecord>;
    struct Page { std::array<Record, PageSize> records; };
    uint64_t Version() const noexcept { return m_version; }
    uint64_t BaseVersion() const noexcept { return m_baseVersion; }
    Record Get(size_t index) const noexcept;
    size_t PageCount() const noexcept { return m_pages.size(); }
    const Page* PageAt(size_t index) const noexcept { return index < m_pages.size() ? m_pages[index].get() : nullptr; }
private:
    friend class BindingTableBuilder;
    uint64_t m_version = 0, m_baseVersion = 0;
    std::vector<std::shared_ptr<const Page>> m_pages;
};

class BindingTableBuilder {
public:
    explicit BindingTableBuilder(std::shared_ptr<runtime::ResourceCleanupQueue> cleanup,
        std::shared_ptr<const BindingTableVersion> base = {});
    BindingTableBuilder(const BindingTableBuilder&) = delete;
    BindingTableBuilder& operator=(const BindingTableBuilder&) = delete;
    void Set(size_t index, BindingTableVersion::Record record);
    void Clear(size_t index) { Set(index, {}); }
    std::shared_ptr<const BindingTableVersion> Seal();
private:
    std::shared_ptr<runtime::ResourceCleanupQueue> m_cleanup;
    std::shared_ptr<const BindingTableVersion> m_base;
    std::vector<std::shared_ptr<const BindingTableVersion::Page>> m_pages;
    std::vector<std::shared_ptr<BindingTableVersion::Page>> m_changed;
    bool m_dirty = false, m_sealed = false;
};

// Composition happens during preparation/patch construction. Copies and moves
// on the render thread retain a single cleanup-owned root.
class ExecutionResourceLease {
public:
    ExecutionResourceLease() = default;
    static ExecutionResourceLease Create(std::shared_ptr<runtime::ResourceCleanupQueue> cleanup,
        std::vector<std::shared_ptr<const void>> owners);
    static ExecutionResourceLease Combine(std::shared_ptr<runtime::ResourceCleanupQueue> cleanup,
        const ExecutionResourceLease& prepared, const ExecutionResourceLease& patches);
    static ExecutionResourceLease FromBundle(std::shared_ptr<const PublicationBindingBundle> bundle);
    const std::shared_ptr<const PublicationBindingBundle>& Bundle() const noexcept { return m_owner; }
    explicit operator bool() const noexcept { return static_cast<bool>(m_owner); }
    std::shared_ptr<const void> Owner() const noexcept;
private:
    std::shared_ptr<const PublicationBindingBundle> m_owner;
};
} // namespace org
