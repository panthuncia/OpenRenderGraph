#pragma once

#include <span>
#include <vector>
#include "Render/PreparedTablePublisher.h"

namespace org {

// The immutable routing portion of a prepared GPU table. Row values are supplied
// at publication; only declaration changes rebuild this layout.
template<class Row>
class DeclaredTableLayout {
public:
    explicit DeclaredTableLayout(size_t rows = 0) : m_rows(rows) {}
    struct Destination {
        DeclaredTableLayout* table;
        size_t row;
        uint32_t Row::* field;
        void Bind(DeclaredViewToken token) const { table->Bind(row, field, std::move(token)); }
    };
    Destination Field(size_t row, uint32_t Row::* field) {
        if (row >= m_rows || !field) throw std::invalid_argument("Invalid table destination");
        return {this, row, field};
    }
    std::vector<Row> Resolve(const FramePreparationContext& context, std::span<const Row> rows) const {
        if (rows.size() != m_rows) throw std::invalid_argument("Table row count changed without redeclaration");
        std::vector<Row> result(rows.begin(), rows.end());
        for (const auto& binding : m_bindings) result[binding.row].*binding.field = context.Resolve(binding.view).index;
        return result;
    }
    uint32_t Publish(const FramePreparationContext& context, const PreparedTablePublisher& publisher,
        std::span<const Row> rows) const {
        if (rows.empty()) return UINT32_MAX;
        const auto resolved = Resolve(context, rows);
        return publisher.Publish(context, std::span<const Row>(resolved));
    }
    bool SameLayout(const DeclaredTableLayout& other) const {
        if (m_rows != other.m_rows || m_bindings.size() != other.m_bindings.size()) return false;
        for (size_t i = 0; i < m_bindings.size(); ++i) {
            const auto& a = m_bindings[i]; const auto& b = other.m_bindings[i];
            if (a.row != b.row || a.field != b.field || a.view.layout != b.view.layout
                || a.view.use != b.view.use || a.view.view != b.view.view) return false;
        }
        return true;
    }
private:
    struct Binding { size_t row; uint32_t Row::* field; DeclaredViewToken view; };
    void Bind(size_t row, uint32_t Row::* field, DeclaredViewToken view) {
        if (!view.layout || view.use >= view.layout->uses.size()
            || view.view >= view.layout->uses[view.use].requiredViews.size())
            throw std::invalid_argument("Table requires a scalar declared view");
        for (const auto& binding : m_bindings)
            if (binding.row == row && binding.field == field) throw std::invalid_argument("Duplicate table destination");
        m_bindings.push_back({row, field, std::move(view)});
    }
    size_t m_rows;
    std::vector<Binding> m_bindings;
};

} // namespace org
