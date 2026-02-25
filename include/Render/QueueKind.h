#pragma once

#include <cstdint>

enum class QueueKind : uint8_t { Graphics = 0, Compute = 1, Copy = 2, Count };

enum class RenderQueueSelection : uint8_t {
	Graphics = 0,
};

enum class ComputeQueueSelection : uint8_t {
	Compute = 0,
	Graphics = 1,
};

enum class CopyQueueSelection : uint8_t {
	Copy = 0,
	Graphics = 1,
};

constexpr QueueKind ResolveQueueKind(RenderQueueSelection) noexcept {
	return QueueKind::Graphics;
}

constexpr QueueKind ResolveQueueKind(ComputeQueueSelection selection) noexcept {
	return selection == ComputeQueueSelection::Compute ? QueueKind::Compute : QueueKind::Graphics;
}

constexpr QueueKind ResolveQueueKind(CopyQueueSelection selection) noexcept {
	return selection == CopyQueueSelection::Copy ? QueueKind::Copy : QueueKind::Graphics;
}