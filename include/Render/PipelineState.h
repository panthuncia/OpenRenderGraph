#pragma once

#include <rhi.h>
#include <cstddef>
#include <memory>
#include <stdint.h>
#include <string>
#include <vector>

#include "Resources/ResourceIdentifier.h"


namespace org {

struct PipelineResources {
	std::vector<ResourceIdentifier> mandatoryResourceDescriptorSlots;
	std::vector<ResourceIdentifier> optionalResourceDescriptorSlots;
};

// A built pipeline, its layout and its descriptor slots. RecordingContext::Bind*Pipeline binds one and retains it.
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

    // Layout ownership must survive until after the pipeline is destroyed.
    std::shared_ptr<const void> layoutOwner;
    rhi::PipelineLayoutHandle layout{};
	uint64_t resourceIDsHash = 0;
	rhi::PipelinePtr pso;
	PipelineResources pipelineResources;
	uint64_t generation = 1;
	uint64_t sourceHash = 0;
	uint64_t bytecodeHash = 0;
	std::string label;
};

} // namespace org
