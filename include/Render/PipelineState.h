#pragma once

#include <rhi.h>
#include <atomic>
#include <memory>
#include <stdint.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "Resources/ResourceIdentifier.h"


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
				std::move(resources)))) {}
	PipelineState() :
		m_slot(std::make_shared<PipelineStateSlot>(
			std::make_shared<PipelineStatePayload>(
				rhi::PipelinePtr{},
				0,
				PipelineResources{}))) {}

	const rhi::Pipeline& GetAPIPipelineState() const {
		return RequirePayload()->pso.Get();
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

	std::shared_ptr<PipelineStateSlot> m_slot;
};


} // namespace org

using org::PipelineResources;
using org::PipelineState;
using org::PipelineStatePayload;
using org::PipelineStateSlot;
