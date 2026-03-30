#pragma once

#include <cstdint>
#include <optional>

enum class QueueKind : uint8_t { Graphics = 0, Compute = 1, Copy = 2, Count };

/// Dense index into queue-parallel arrays. Forward-declared here for use in pass parameters.
enum class QueueSlotIndex : uint8_t {};

constexpr bool IsQueueKindSupportedByRenderPass(QueueKind kind) noexcept {
	return kind == QueueKind::Graphics;
}

constexpr bool IsQueueKindSupportedByComputePass(QueueKind kind) noexcept {
	return kind == QueueKind::Graphics || kind == QueueKind::Compute;
}

constexpr bool IsQueueKindSupportedByCopyPass(QueueKind kind) noexcept {
	return kind == QueueKind::Graphics || kind == QueueKind::Copy;
}