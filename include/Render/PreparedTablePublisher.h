#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "Render/PreparedPass.h"

namespace org {

class Buffer;

// Small read-only GPU tables built during preparation, for data that embeds
// bindless descriptor indices (per-view tables and the like).
//
// A descriptor index is only valid for the backing a frame binds: the render
// graph gives a resource fresh slots whenever it realizes it on a new backing
// (for example when the persistent graph re-places an aliased resource). Tables
// uploaded ahead of preparation therefore go stale on exactly the frame a
// resource moves. Built in Prepare (or BuildRecipe), the indices come from the
// frame's frozen bindings, and the framework's invocation/recipe reuse rebuilds
// the table only when a binding it resolved changes.
//
// Publish() creates an immutable upload-heap structured buffer only when the
// bytes differ from the previous publication, retains it in the packet (or
// recipe) being prepared, and returns its SRV index. Older tables stay alive
// through the packets that captured them. Unchanged tables cost a comparison.
class PreparedTablePublisher {
public:
    explicit PreparedTablePublisher(std::string name = "PreparedTable") : m_name(std::move(name)) {}

    uint32_t Publish(const FramePreparationContext& preparation, std::span<const std::byte> bytes, uint32_t stride) const;

    template<class T>
    uint32_t Publish(const FramePreparationContext& preparation, std::span<const T> rows) const {
        return Publish(preparation, std::as_bytes(rows), static_cast<uint32_t>(sizeof(T)));
    }

private:
    std::string m_name;
    mutable std::mutex m_mutex;
    mutable std::vector<std::byte> m_bytes;
    mutable std::shared_ptr<Buffer> m_table;
    mutable uint32_t m_stride = 0;
    mutable uint32_t m_index = 0;
};

} // namespace org
