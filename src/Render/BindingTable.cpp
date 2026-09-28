#include "Render/BindingTable.h"
#include "Render/PublicationBindingBundle.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>

namespace org {
namespace { std::atomic<uint64_t> nextTableVersion{1}; }
BindingRecordBuilder::BindingRecordBuilder(std::shared_ptr<runtime::ResourceCleanupQueue> cleanup, size_t bytes) {
    if (!cleanup) throw std::invalid_argument("Binding record needs a cleanup queue");
    m_record = cleanup->Make<BindingRecord>();
    m_record->m_bytes.resize(bytes);
}
BindingRecordBuilder::BindingRecordBuilder(std::shared_ptr<runtime::ResourceCleanupQueue> cleanup, const BindingRecord& base) {
    if (!cleanup) throw std::invalid_argument("Binding record needs a cleanup queue");
    m_record = cleanup->Make<BindingRecord>(base);
}
void BindingRecordBuilder::WriteBytes(size_t offset, std::span<const std::byte> bytes) {
    if (!m_record) throw std::logic_error("Binding record already sealed");
    if (offset > m_record->m_bytes.size() || bytes.size() > m_record->m_bytes.size() - offset)
        throw std::out_of_range("Binding record write exceeds record");
    for (const auto& field : m_record->m_fields)
        if (offset < field.offset + sizeof(uint32_t) && field.offset < offset + bytes.size())
            throw std::logic_error("Raw write overlaps an owned descriptor field");
    if (!bytes.empty()) std::memcpy(m_record->m_bytes.data() + offset, bytes.data(), bytes.size());
}
void BindingRecordBuilder::WriteBinding(size_t offset, OwnedDescriptorBinding binding) {
    if (!m_record) throw std::logic_error("Binding record already sealed");
    if (offset > m_record->m_bytes.size() || sizeof(uint32_t) > m_record->m_bytes.size() - offset)
        throw std::out_of_range("Descriptor field exceeds record");
    auto found = m_record->m_fields.end();
    for (auto it = m_record->m_fields.begin(); it != m_record->m_fields.end(); ++it) {
        if (it->offset == offset) found = it;
        else if (offset < it->offset + sizeof(uint32_t) && it->offset < offset + sizeof(uint32_t))
            throw std::logic_error("Overlapping descriptor fields");
    }
    const auto index = binding.Index();
    if (found == m_record->m_fields.end()) m_record->m_fields.push_back({offset, std::move(binding)});
    else found->binding = std::move(binding);
    std::memcpy(m_record->m_bytes.data() + offset, &index, sizeof(index));
}
std::shared_ptr<const BindingRecord> BindingRecordBuilder::Seal() {
    if (!m_record) throw std::logic_error("Binding record already sealed");
    return std::exchange(m_record, {});
}
BindingTableVersion::Record BindingTableVersion::Get(size_t index) const noexcept {
    const auto page = index / PageSize;
    return page < m_pages.size() && m_pages[page] ? m_pages[page]->records[index % PageSize] : Record{};
}
BindingTableBuilder::BindingTableBuilder(std::shared_ptr<runtime::ResourceCleanupQueue> cleanup,
    std::shared_ptr<const BindingTableVersion> base) : m_cleanup(std::move(cleanup)), m_base(std::move(base)) {
    if (!m_cleanup) throw std::invalid_argument("Binding table needs a cleanup queue");
    if (m_base) m_pages = m_base->m_pages;
    m_changed.resize(m_pages.size());
}
void BindingTableBuilder::Set(size_t index, BindingTableVersion::Record record) {
    if (m_sealed) throw std::logic_error("Binding table already sealed");
    const auto page = index / BindingTableVersion::PageSize, entry = index % BindingTableVersion::PageSize;
    if (page >= m_pages.size()) {
        if (!record) return;
        m_pages.resize(page + 1);
        m_changed.resize(page + 1);
    }
    if (m_pages[page] && m_pages[page]->records[entry] == record) return;
    if (!m_pages[page] && !record) return;
    if (!m_changed[page]) {
        auto next = m_cleanup->Make<BindingTableVersion::Page>();
        if (m_pages[page]) *next = *m_pages[page];
        m_changed[page] = next;
        m_pages[page] = std::move(next);
    }
    m_changed[page]->records[entry] = std::move(record);
    m_dirty = true;
}
std::shared_ptr<const BindingTableVersion> BindingTableBuilder::Seal() {
    if (m_sealed) throw std::logic_error("Binding table already sealed");
    m_sealed = true;
    if (!m_dirty && m_base) {
        m_pages.clear();
        m_changed.clear();
        return std::exchange(m_base, {});
    }
    for (size_t i = 0; i < m_changed.size(); ++i)
        if (m_changed[i] && std::ranges::all_of(m_changed[i]->records, [](const auto& record) { return !record; })) m_pages[i].reset();
    while (!m_pages.empty() && !m_pages.back()) m_pages.pop_back();
    auto version = m_cleanup->Make<BindingTableVersion>();
    version->m_baseVersion = m_base ? m_base->Version() : 0;
    version->m_version = nextTableVersion.fetch_add(1, std::memory_order_relaxed);
    version->m_pages = std::move(m_pages);
    m_changed.clear();
    m_base.reset();
    return version;
}
ExecutionResourceLease ExecutionResourceLease::Create(std::shared_ptr<runtime::ResourceCleanupQueue> cleanup,
    std::vector<std::shared_ptr<const void>> owners) {
    if (!cleanup) throw std::invalid_argument("Execution ownership needs a cleanup queue");
    std::erase(owners, std::shared_ptr<const void>{});
    ExecutionResourceLease result;
    if (!owners.empty()) result.m_owner = cleanup->Make<PublicationBindingBundle>(
        std::vector<PublicationBindingBundle::Snapshot>{}, std::move(owners));
    return result;
}
ExecutionResourceLease ExecutionResourceLease::Combine(std::shared_ptr<runtime::ResourceCleanupQueue> cleanup,
    const ExecutionResourceLease& prepared, const ExecutionResourceLease& patches) {
    if (!prepared) return patches;
    if (!patches || prepared.m_owner == patches.m_owner) return prepared;
    if (!cleanup) throw std::invalid_argument("Execution ownership needs a cleanup queue");
    return FromBundle(cleanup->Make<PublicationBindingBundle>(
        std::vector<PublicationBindingBundle::Snapshot>{}, std::vector<std::shared_ptr<const void>>{},
        std::vector<std::shared_ptr<const PublicationBindingBundle>>{prepared.m_owner, patches.m_owner}));
}
ExecutionResourceLease ExecutionResourceLease::FromBundle(std::shared_ptr<const PublicationBindingBundle> bundle) {
    ExecutionResourceLease result;
    result.m_owner = std::move(bundle);
    return result;
}
std::shared_ptr<const void> ExecutionResourceLease::Owner() const noexcept { return m_owner; }
} // namespace org
