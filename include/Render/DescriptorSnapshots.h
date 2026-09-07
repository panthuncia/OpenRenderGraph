#pragma once

#include <array>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <rhi.h>
#include <BasicTelemetry/Tracy.h>

namespace org {

// Preparation-owner API. Recipes must own concrete backing versions, not live
// resource wrappers. The callback only encodes a view; it cannot borrow a pass.
class OwnedDescriptorWrite {
public:
    template<class Data>
    static std::shared_ptr<const OwnedDescriptorWrite> Make(Data data,
        rhi::Result (*write)(const Data&, rhi::Device, rhi::DescriptorSlot) noexcept) {
        if (!write) throw std::invalid_argument("Missing descriptor writer");
        return std::make_shared<Model<Data>>(std::move(data), write);
    }
    virtual ~OwnedDescriptorWrite() = default;
    virtual rhi::Result Apply(rhi::Device, rhi::DescriptorSlot) const noexcept = 0;
private:
    template<class Data> struct Model;
};
template<class Data> struct OwnedDescriptorWrite::Model final : OwnedDescriptorWrite {
    Model(Data value, rhi::Result (*fn)(const Data&, rhi::Device, rhi::DescriptorSlot) noexcept)
        : data(std::move(value)), write(fn) {}
    rhi::Result Apply(rhi::Device device, rhi::DescriptorSlot slot) const noexcept override {
        return write(data, device, slot);
    }
    const Data data;
    rhi::Result (*const write)(const Data&, rhi::Device, rhi::DescriptorSlot) noexcept;
};

// Persistent paged dirty-slot journal. Captures share unchanged pages; updates
// copy only a dirty page. No history grows with publication count. Recipe object
// identity is the content version (reuse it only for identical owned contents).
class DescriptorContentJournal {
public:
    static constexpr uint32_t PageSize = 64;
    using Recipe = std::shared_ptr<const OwnedDescriptorWrite>;
    using Page = std::array<Recipe, PageSize>;
    struct Layout {
        rhi::DescriptorHeapType type;
        uint32_t capacity;
        bool shaderVisible;
    };
    class Capture {
        friend class DescriptorContentJournal;
        friend class DescriptorSnapshotPool;
        std::shared_ptr<const Layout> layout;
        std::vector<std::shared_ptr<const Page>> pages;
    };
    explicit DescriptorContentJournal(Layout layout)
        : m_layout(std::make_shared<const Layout>(layout)),
          m_pages((static_cast<size_t>(layout.capacity) + PageSize - 1) / PageSize) {
        if (!layout.capacity) throw std::invalid_argument("Empty descriptor layout");
    }
    void Write(uint32_t index, Recipe recipe) {
        if (index >= m_layout->capacity) throw std::out_of_range("Descriptor logical index");
        auto& page = m_pages[index / PageSize];
        if ((!page && !recipe) || (page && (*page)[index % PageSize] == recipe)) return;
        auto next = page ? std::make_shared<Page>(*page) : std::make_shared<Page>();
        (*next)[index % PageSize] = std::move(recipe);
        page = std::move(next);
        m_capture.reset();
    }
    std::shared_ptr<const Capture> CaptureContents() const {
        if (m_capture) return m_capture;
        auto result = std::make_shared<Capture>();
        result->layout = m_layout;
        result->pages = m_pages;
        m_capture = result;
        return m_capture;
    }
private:
    std::shared_ptr<const Layout> m_layout;
    std::vector<std::shared_ptr<const Page>> m_pages;
    mutable std::shared_ptr<const Capture> m_capture;
};

// One pool per heap kind/visibility and generation, with one heap per execution
// slot, never per compile job. All mutation happens at the admission owner.
// Returned owners must survive recording AND actual GPU retirement.
class DescriptorSnapshotPool {
public:
    class Snapshot {
        friend class DescriptorSnapshotPool;
    public:
        rhi::DescriptorHeap Heap() const { return heap.Get(); }
        rhi::DescriptorSlot Resolve(uint32_t logicalIndex) const {
            if (logicalIndex >= contents->layout->capacity) throw std::out_of_range("Descriptor logical index");
            const auto& page = contents->pages[logicalIndex / DescriptorContentJournal::PageSize];
            if (!page || !(*page)[logicalIndex % DescriptorContentJournal::PageSize])
                throw std::out_of_range("Undeclared descriptor slot");
            return {heap.Get().GetHandle(), logicalIndex};
        }
    private:
        // Owners declared before heap so native heap dies before its device.
        std::shared_ptr<const void> deviceOwner;
        std::vector<DescriptorContentJournal::Recipe> applied;
        std::shared_ptr<const DescriptorContentJournal::Capture> contents;
        rhi::DescriptorHeapPtr heap;
    };
    DescriptorSnapshotPool(rhi::Device device, std::shared_ptr<const void> deviceOwner, uint32_t executionSlots)
        : m_device(device), m_deviceOwner(std::move(deviceOwner)), m_slots(executionSlots) {
        if (!device || !m_deviceOwner || !executionSlots || executionSlots > 64)
            throw std::invalid_argument("Invalid descriptor snapshot pool");
    }
    DescriptorSnapshotPool(const DescriptorSnapshotPool&) = delete;
    DescriptorSnapshotPool& operator=(const DescriptorSnapshotPool&) = delete;
    std::shared_ptr<const Snapshot> Assemble(uint32_t executionSlot,
        std::shared_ptr<const DescriptorContentJournal::Capture> contents) {
        BT_ZONE_SCOPE("ORG.AsyncRealization.Descriptors.Assemble");
        if (!contents) throw std::invalid_argument("Missing descriptor contents");
        auto& slot = m_slots.at(executionSlot);
        if (m_layout && m_layout != contents->layout) throw std::invalid_argument("Descriptor generation mismatch");
        if (slot && slot.use_count() != 1) {
            basic_telemetry::AddCounter("ORG.AsyncRealization.Descriptors.Busy");
            return {}; // Never overwrite recording or GPU-visible contents.
        }
        if (!slot) {
            BT_ZONE_SCOPE("ORG.AsyncRealization.Descriptors.Allocate");
            auto next = std::make_shared<Snapshot>();
            next->deviceOwner = m_deviceOwner;
            next->applied.resize(contents->layout->capacity);
            const auto& layout = *contents->layout;
            if (m_device.CreateDescriptorHeap({layout.type, layout.capacity, layout.shaderVisible,
                    "Owned execution descriptors"}, next->heap) != rhi::Result::Ok)
                throw std::runtime_error("Descriptor snapshot allocation failed");
            slot = std::move(next);
            m_layout = contents->layout;
            basic_telemetry::AddCounter("ORG.AsyncRealization.Descriptors.Allocations");
        }
        try {
            BT_ZONE_SCOPE("ORG.AsyncRealization.Descriptors.Populate");
            for (size_t p = 0; p < contents->pages.size(); ++p) {
                const auto& page = contents->pages[p];
                if (!page || (slot->contents && slot->contents->pages[p] == page)) continue;
                for (size_t j = 0; j < DescriptorContentJournal::PageSize; ++j) {
                    const auto index = p * DescriptorContentJournal::PageSize + j;
                    if (index >= slot->applied.size()) break;
                    const auto& recipe = (*page)[j];
                    if (!recipe || recipe == slot->applied[index]) continue;
                    if (recipe->Apply(m_device, {slot->heap.Get().GetHandle(), static_cast<uint32_t>(index)}) != rhi::Result::Ok)
                        throw std::runtime_error("Descriptor snapshot population failed");
                    slot->applied[index] = recipe;
                    basic_telemetry::AddCounter("ORG.AsyncRealization.Descriptors.Writes");
                }
            }
            // Removed slots are inaccessible through Resolve. Retain their old
            // recipes until overwritten: unused physical descriptors still hold
            // old addresses. This retention is bounded by heap capacity.
            slot->contents = std::move(contents);
        } catch (...) {
            slot.reset(); // Only this unleased slot was touched; selected heaps survive.
            throw;
        }
        return slot;
    }
private:
    rhi::Device m_device;
    std::shared_ptr<const void> m_deviceOwner;
    std::shared_ptr<const DescriptorContentJournal::Layout> m_layout;
    std::vector<std::shared_ptr<Snapshot>> m_slots;
};

} // namespace org
