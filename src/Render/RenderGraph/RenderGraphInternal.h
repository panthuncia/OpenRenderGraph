#pragma once

// What RenderGraph's units (RenderGraph.cpp, RenderGraphStructural.cpp, RenderGraphDiagnostics.cpp) share. Private.

#include "Render/RenderGraph/RenderGraph.h"
#include "Render/Runtime/OpenRenderGraphSettings.h"
#include "RenderGraphCompilerState.h"
#include "PersistentLowering.h"
#include "FrameRecording.h"
#include "Render/RenderGraph/InteropAllocator.h"
#include "Render/RenderGraph/RhiExecution.h"

#include <span>
#include <algorithm>
#include <type_traits>
#include <atomic>
#include <cmath>
#include <map>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <future>
#include <thread>
#include "Render/Runtime/IUploadService.h"
#include "Render/Runtime/IDescriptorService.h"
#include "Render/Runtime/DescriptorServiceAccess.h"
#include "Render/Runtime/ScopedActiveGraphServices.h"
#include <BasicTelemetry/Tracy.h>
#include <rhi_helpers.h>
#include <rhi_debug.h>
#include <rhi_interop_dx12.h>
#include <rhi_interop_vulkan.h>
#include <random>

#include "Render/PassExecutionContext.h"
#include "Utilities/ORGUtilities.h"
#include "Managers/Singletons/DeviceManager.h"
#include "Managers/Singletons/DeletionManager.h"
#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Managers/Singletons/UploadManager.h"
#include "Managers/Singletons/StatisticsManager.h"
#include "Render/PassBuilders.h"
#include "Resources/ResourceGroup.h"
#include "Interfaces/IHasMemoryMetadata.h"
#include "Interfaces/IDynamicDeclaredResources.h"
#include "Resources/DynamicResource.h"
#include "Resources/ExternalTextureResource.h"
#include "Resources/BackedResource.h"
#include "Render/BufferBarrierHelpers.h"
#include "Resources/PixelBuffer.h"
#include "Resources/Buffers/DynamicBufferBase.h"
#include "Resources/MemoryStatisticsComponents.h"

namespace org::graph_internal {

inline void MergeResolverWaits(std::vector<ExternalTimelinePoint>& waits,
    const std::vector<ResolverSnapshot>& snapshots)
{
    for (const auto& snapshot : snapshots)
        if (snapshot.state && snapshot.state->waits) waits.insert(waits.end(), snapshot.state->waits->begin(), snapshot.state->waits->end());
    std::sort(waits.begin(), waits.end(), [](const auto& lhs, const auto& rhs) {
        const auto lh = lhs.timeline.GetHandle(); const auto rh = rhs.timeline.GetHandle();
        return lh.index != rh.index ? lh.index < rh.index :
            (lh.generation != rh.generation ? lh.generation < rh.generation : lhs.value < rhs.value);
    });
    std::vector<ExternalTimelinePoint> normalized;
    normalized.reserve(waits.size());
    for (const auto& wait : waits) {
        const auto handle = wait.timeline.GetHandle();
        if (!normalized.empty()) {
            const auto previous = normalized.back().timeline.GetHandle();
            if (previous.index == handle.index && previous.generation == handle.generation) {
                normalized.back().value = (std::max)(normalized.back().value, wait.value);
                continue;
            }
        }
        normalized.push_back(wait);
    }
    waits = std::move(normalized);
}

template<class PassResourceData>
std::unique_ptr<ResourceRegistryView> MakePassResourceRegistryView(
	ResourceRegistry& registry, const PassResourceData& resources)
{
	auto view = std::make_unique<ResourceRegistryView>(registry, resources.identifierSet);
	for (const auto& requirement : resources.staticResourceRequirements) {
		view->AllowConcreteResourceID(requirement.resourceHandleAndRange.resource.GetGlobalResourceID());
	}
	for (const auto& requirement : resources.frameResourceRequirements)
		view->AllowConcreteResourceID(requirement.resourceHandleAndRange.resource.GetGlobalResourceID());
	for (const auto& block : resources.resolverRequirementBlocks) {
		view->AllowConcretePredicate([slot = &block](uint64_t id) {
			if (!slot || !*slot) return false;
			return std::ranges::any_of((*slot)->requirements, [id](const auto& requirement) {
				return requirement.resourceHandleAndRange.resource.GetGlobalResourceID() == id;
			});
		});
	}
	for (const auto& transition : resources.internalTransitions)
		view->AllowConcreteResourceID(transition.first.resource.GetGlobalResourceID());
	return view;
}

inline const char* QueueKindToString(QueueKind queue) noexcept {
	switch (queue) {
	case QueueKind::Graphics: return "Graphics";
	case QueueKind::Compute: return "Compute";
	case QueueKind::Copy: return "Copy";
	default: return "Unknown";
	}
}

inline const char* PassTypeToString(RenderGraph::PassType type) noexcept {
	switch (type) {
	case RenderGraph::PassType::Render: return "Render";
	default: return "Unknown";
	}
}

} // namespace org::graph_internal
