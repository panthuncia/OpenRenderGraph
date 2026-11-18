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
// A descriptor index is only valid for the backing that a frame binds. The
// render graph gives a resource fresh slots whenever it realizes the resource
// on a new backing, for example when the persistent graph re-places an aliased
// resource. So tables uploaded before preparation go stale on exactly the
// frame a resource moves. Tables built in Prepare take their indices from the
// frame's frozen bindings, which are checked every frame the pass prepares.
//
// Publish() creates an immutable upload-heap structured buffer only when the
// bytes differ from the previous publication. It retains the buffer in the
// packet (or recipe) being prepared, and returns its SRV index. Older tables
// stay alive through the packets that captured them. An unchanged table costs
// only a comparison.
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
