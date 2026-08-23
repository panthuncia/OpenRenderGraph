#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <rhi.h>

#include "QueueKind.h"

namespace org {

struct DeviceRegistryEntry {
	DeviceInstanceId id = DeviceInstanceId::Invalid;
	rhi::Backend backend = rhi::Backend::Null;
	rhi::Device device{};
};

/// Registry of initialized RHI device instances. API identity is metadata;
/// two devices using the same API still receive distinct instance IDs.
class DeviceRegistry {
public:
	DeviceInstanceId RegisterPrimary(rhi::Backend backend, rhi::Device device) {
		if (!m_entries.empty()) throw std::logic_error("DeviceRegistry primary device is already registered");
		m_primary = RegisterNew(backend, device);
		return m_primary;
	}

	DeviceInstanceId Register(rhi::Backend backend, rhi::Device device) {
		if (!device || backend == rhi::Backend::Null) throw std::invalid_argument("DeviceRegistry requires a valid device and API");
		for (const auto& entry : m_entries) {
			if (entry.device.impl == device.impl && entry.device.vt == device.vt) return entry.id;
		}
		return RegisterNew(backend, device);
	}

	DeviceInstanceId PrimaryId() const noexcept { return m_primary; }
	const DeviceRegistryEntry* Find(DeviceInstanceId id) const noexcept {
		const size_t index = static_cast<uint8_t>(id);
		return index < m_entries.size() && m_entries[index].id == id ? &m_entries[index] : nullptr;
	}
	DeviceRegistryEntry* Find(DeviceInstanceId id) noexcept {
		return const_cast<DeviceRegistryEntry*>(std::as_const(*this).Find(id));
	}
	const DeviceRegistryEntry* FindFirst(rhi::Backend backend) const noexcept {
		for (const auto& entry : m_entries) if (entry.backend == backend) return &entry;
		return nullptr;
	}

	bool empty() const noexcept { return m_entries.empty(); }
	size_t size() const noexcept { return m_entries.size(); }
	DeviceRegistryEntry& front() noexcept { return m_entries.front(); }
	const DeviceRegistryEntry& front() const noexcept { return m_entries.front(); }
	DeviceRegistryEntry& operator[](size_t index) noexcept { return m_entries[index]; }
	const DeviceRegistryEntry& operator[](size_t index) const noexcept { return m_entries[index]; }
	auto begin() noexcept { return m_entries.begin(); }
	auto end() noexcept { return m_entries.end(); }
	auto begin() const noexcept { return m_entries.begin(); }
	auto end() const noexcept { return m_entries.end(); }

private:
	DeviceInstanceId RegisterNew(rhi::Backend backend, rhi::Device device) {
		if (!device || backend == rhi::Backend::Null) throw std::invalid_argument("DeviceRegistry requires a valid device and API");
		if (m_entries.size() >= 255) throw std::runtime_error("DeviceRegistry exhausted its instance ID space");
		const DeviceInstanceId id{ static_cast<uint8_t>(m_entries.size()) };
		m_entries.push_back({ id, backend, device });
		return id;
	}

	std::vector<DeviceRegistryEntry> m_entries;
	DeviceInstanceId m_primary = DeviceInstanceId::Invalid;
};

} // namespace org
