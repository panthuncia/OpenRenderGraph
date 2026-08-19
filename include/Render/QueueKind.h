#pragma once

#include <cstdint>
#include <optional>
#include <rhi.h>


namespace org {

enum class QueueKind : uint8_t { Graphics = 0, Compute = 1, Copy = 2, Count };

enum class QueueAssignmentPolicy : uint8_t {
	ForcePreferred = 0,
	Automatic = 1
};

/// Dense index into queue-parallel arrays. Forward-declared here for use in pass parameters.
enum class QueueSlotIndex : uint8_t {};
enum class BackendInstanceId : uint8_t { Primary = 0, Peer = 1 };

enum class BackendAffinityStrength : uint8_t {
	Primary = 0,
	Preferred,
	Required,
};

struct BackendAffinity {
	BackendAffinityStrength strength = BackendAffinityStrength::Primary;
	rhi::Backend backend = rhi::Backend::Null;
};

constexpr bool IsQueueKindSupportedByRenderPass(QueueKind kind) noexcept {
	return kind == QueueKind::Graphics;
}

constexpr bool IsQueueKindSupportedByComputePass(QueueKind kind) noexcept {
	return kind == QueueKind::Graphics || kind == QueueKind::Compute;
}

constexpr bool IsQueueKindSupportedByCopyPass(QueueKind kind) noexcept {
	return kind == QueueKind::Graphics || kind == QueueKind::Copy;
}


} // namespace org
