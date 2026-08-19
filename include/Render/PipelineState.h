#pragma once

#include <rhi.h>
#include <atomic>
#include <memory>
#include <stdint.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "Resources/ResourceIdentifier.h"
#include "Render/QueueKind.h"


namespace org {

struct PipelineResources {
	std::vector<ResourceIdentifier> mandatoryResourceDescriptorSlots;
	std::vector<ResourceIdentifier> optionalResourceDescriptorSlots;
};

struct PipelineStatePayload {
	PipelineStatePayload(
		rhi::PipelinePtr pipeline,
		uint64_t resourceHash,
		PipelineResources resources,
		uint64_t pipelineGeneration = 1,
		uint64_t sourceFingerprint = 0,
		uint64_t bytecodeFingerprint = 0,
		std::string pipelineLabel = {}) :
		resourceIDsHash(resourceHash),
		pso(std::move(pipeline)),
		pipelineResources(std::move(resources)),
		generation(pipelineGeneration),
		sourceHash(sourceFingerprint),
		bytecodeHash(bytecodeFingerprint),
		label(std::move(pipelineLabel)) {}

	uint64_t resourceIDsHash = 0;
	rhi::PipelinePtr pso;
	PipelineResources pipelineResources;
	uint64_t generation = 1;
	uint64_t sourceHash = 0;
	uint64_t bytecodeHash = 0;
	std::string label;
};

class PipelineStateSlot {
public:
	explicit PipelineStateSlot(std::shared_ptr<PipelineStatePayload> payload) :
		m_active(std::move(payload)) {}

	std::shared_ptr<PipelineStatePayload> Load() const {
		return m_active.load(std::memory_order_acquire);
	}

	std::shared_ptr<PipelineStatePayload> Exchange(std::shared_ptr<PipelineStatePayload> payload) {
		return m_active.exchange(std::move(payload), std::memory_order_acq_rel);
	}

private:
	std::atomic<std::shared_ptr<PipelineStatePayload>> m_active;
};

class PipelineState {
public:
	PipelineState(rhi::PipelinePtr pso,
		uint64_t resourceIDsHash, 
		PipelineResources resources) :
		m_slot(std::make_shared<PipelineStateSlot>(
			std::make_shared<PipelineStatePayload>(
				std::move(pso),
				resourceIDsHash,
				std::move(resources)))),
		m_backendSlots(std::make_shared<BackendSlotMap>()) {}
	PipelineState() :
		m_slot(std::make_shared<PipelineStateSlot>(
			std::make_shared<PipelineStatePayload>(
				rhi::PipelinePtr{},
				0,
				PipelineResources{}))),
		m_backendSlots(std::make_shared<BackendSlotMap>()) {}

	const rhi::Pipeline& GetAPIPipelineState() const {
		return RequirePayload()->pso.Get();
	}
	const rhi::Pipeline& GetAPIPipelineState(BackendInstanceId backendInstance) const {
		return RequirePayload(backendInstance)->pso.Get();
	}
	bool HasBackendPipeline(BackendInstanceId backendInstance) const {
		const auto payload = GetPayload(backendInstance);
		return payload && payload->pso;
	}
	void AttachBackendPipeline(BackendInstanceId backendInstance, rhi::PipelinePtr pipeline,
		uint64_t resourceIDsHash, PipelineResources resources) const {
		const auto key = static_cast<uint8_t>(backendInstance);
		if (key == static_cast<uint8_t>(BackendInstanceId::Primary)) {
			m_slot->Exchange(std::make_shared<PipelineStatePayload>(std::move(pipeline), resourceIDsHash, std::move(resources)));
			return;
		}
		(*m_backendSlots)[key] = std::make_shared<PipelineStateSlot>(
			std::make_shared<PipelineStatePayload>(std::move(pipeline), resourceIDsHash, std::move(resources)));
	}
	uint64_t GetResourceIDsHash() const {
		return RequirePayload()->resourceIDsHash;
	}
	const PipelineResources& GetResourceDescriptorSlots() const {
		return RequirePayload()->pipelineResources;
	}
	uint64_t GetGeneration() const {
		const auto payload = GetPayload();
		return payload ? payload->generation : 0;
	}
	uint64_t GetSourceHash() const {
		const auto payload = GetPayload();
		return payload ? payload->sourceHash : 0;
	}
	uint64_t GetBytecodeHash() const {
		const auto payload = GetPayload();
		return payload ? payload->bytecodeHash : 0;
	}
	std::string GetLabel() const {
		const auto payload = GetPayload();
		return payload ? payload->label : std::string{};
	}
	explicit operator bool() const {
		const auto payload = GetPayload();
		return payload && payload->pso;
	}

	std::shared_ptr<PipelineStatePayload> GetPayload() const {
		return m_slot ? m_slot->Load() : nullptr;
	}
	std::shared_ptr<PipelineStatePayload> GetPayload(BackendInstanceId backendInstance) const {
		const auto key = static_cast<uint8_t>(backendInstance);
		if (key == static_cast<uint8_t>(BackendInstanceId::Primary)) return GetPayload();
		const auto it = m_backendSlots->find(key);
		return it == m_backendSlots->end() || !it->second ? nullptr : it->second->Load();
	}
	std::shared_ptr<PipelineStateSlot> GetSlot() const {
		return m_slot;
	}
	std::shared_ptr<PipelineStatePayload> ReplacePayload(std::shared_ptr<PipelineStatePayload> payload) const {
		return m_slot ? m_slot->Exchange(std::move(payload)) : nullptr;
	}

private:
	std::shared_ptr<PipelineStatePayload> RequirePayload() const {
		auto payload = GetPayload();
		if (!payload) {
			throw std::runtime_error("PipelineState has no active payload");
		}
		return payload;
	}
	std::shared_ptr<PipelineStatePayload> RequirePayload(BackendInstanceId backendInstance) const {
		auto payload = GetPayload(backendInstance);
		if (!payload || !payload->pso) {
			throw std::runtime_error("PipelineState has no payload for backend instance " +
				std::to_string(static_cast<uint8_t>(backendInstance)));
		}
		return payload;
	}

	std::shared_ptr<PipelineStateSlot> m_slot;
	using BackendSlotMap = std::unordered_map<uint8_t, std::shared_ptr<PipelineStateSlot>>;
	std::shared_ptr<BackendSlotMap> m_backendSlots;
};


} // namespace org

using org::PipelineResources;
using org::PipelineState;
using org::PipelineStatePayload;
using org::PipelineStateSlot;
