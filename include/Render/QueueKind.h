#pragma once

#include <cstdint>
#include <compare>
#include <type_traits>
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

/// Opaque identity for one initialized RHI device. The numeric value is an
/// implementation detail of DeviceRegistry and is not an API/backend selector.
class DeviceInstanceId {
public:
	constexpr DeviceInstanceId() noexcept = default;
	explicit constexpr DeviceInstanceId(uint8_t value) noexcept : m_value(value) {}
	template<class T> requires std::is_integral_v<T>
	explicit constexpr operator T() const noexcept { return static_cast<T>(m_value); }
	constexpr bool IsValid() const noexcept { return m_value != 0xFF; }
	friend constexpr bool operator==(DeviceInstanceId, DeviceInstanceId) noexcept = default;
	friend constexpr auto operator<=>(DeviceInstanceId, DeviceInstanceId) noexcept = default;

	static const DeviceInstanceId Invalid;
	// Compatibility constants for callers not yet migrated to registry lookup.
	static const DeviceInstanceId Primary;
	static const DeviceInstanceId Peer;

private:
	uint8_t m_value = 0xFF;
};

inline constexpr DeviceInstanceId DeviceInstanceId::Invalid{ 0xFF };
inline constexpr DeviceInstanceId DeviceInstanceId::Primary{ 0 };
inline constexpr DeviceInstanceId DeviceInstanceId::Peer{ 1 };

using BackendInstanceId = DeviceInstanceId;

enum class BackendAffinityStrength : uint8_t {
	Primary = 0,
	Preferred,
	Required,
};

struct BackendAffinity {
	BackendAffinityStrength strength = BackendAffinityStrength::Primary;
	rhi::Backend backend = rhi::Backend::Null;
	std::optional<DeviceInstanceId> device;
};

constexpr bool IsQueueKindSupportedByRenderPass(QueueKind kind) noexcept {
	// The historical RenderPass storage is now the unified typed-pass path.
	// Operation-level validation in PassBuilder prevents graphics-only
	// declarations from selecting compute/copy queues.
	return kind == QueueKind::Graphics || kind == QueueKind::Compute || kind == QueueKind::Copy;
}

constexpr bool IsQueueKindSupportedByComputePass(QueueKind kind) noexcept {
	return kind == QueueKind::Graphics || kind == QueueKind::Compute;
}

constexpr bool IsQueueKindSupportedByCopyPass(QueueKind kind) noexcept {
	return kind == QueueKind::Graphics || kind == QueueKind::Copy;
}


} // namespace org
