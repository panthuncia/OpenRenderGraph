#pragma once

#include "Render/RenderGraph/RenderGraph.h"

// Private compiler representation. Keep compiler-only data here rather than in
// RenderGraph's installed public header so compile algorithm changes have a
// narrow incremental-build footprint.
struct RenderGraph::Node {
	size_t passIndex = 0;
	size_t queueSlot = 0;
	QueueKind preferredQueueKind = QueueKind::Graphics;
	QueueAssignmentPolicy queueAssignmentPolicy = QueueAssignmentPolicy::ForcePreferred;
	std::vector<size_t> compatibleQueueSlots;
	uint8_t compatibleQueueKindMask = 0;
	std::optional<size_t> assignedQueueSlot;
	uint32_t originalOrder = 0;
	size_t topoRank = 0;

	const std::vector<uint64_t>* touchedIDs = nullptr;
	const std::vector<uint64_t>* uavIDs = nullptr;

	std::vector<size_t> out;
	std::vector<size_t> in;
	uint32_t indegree = 0;
	uint32_t criticality = 0;
};

struct RenderGraph::CompilerState {
	struct SeqState {
		std::optional<size_t> lastWriter;
		std::vector<size_t> readsSinceWrite;
	};

	std::vector<Node> nodes;
	std::vector<SeqState> dependencySeqStates;
	std::vector<uint64_t> dependencyEdgeKeys;
	std::vector<uint32_t> dependencyIndegrees;
	std::vector<size_t> dependencyTopoOrder;
	std::vector<size_t> dependencyReadyHeap;

	std::vector<uint64_t> resourceIDs;
	std::vector<uint8_t> resourcesWritten;
	std::vector<uint32_t> accessEpochs;
	std::vector<uint32_t> accessWriteEpochs;
	std::vector<uint32_t> accessUavEpochs;
	std::vector<uint32_t> accessDagEpochs;
	std::vector<uint32_t> accessOrder;
	std::vector<uint32_t> schedulingSummaryResourceEpochs;
	std::vector<uint32_t> schedulingSummaryUAVEpochs;
	uint32_t schedulingSummaryEpoch = 1;
	std::vector<size_t> schedulingResourceIndexByDagResourceIndex;
	struct SchedulingPlacedResource {
		uint64_t poolID = 0;
		uint64_t resourceID = 0;
		uint64_t startByte = 0;
		uint64_t endByte = 0;
	};
	std::vector<SchedulingPlacedResource> schedulingPlacedResources;
	std::vector<uint64_t> preferredDynamicStableIDByIndex;
	std::vector<uint64_t> compileTrackerBackingGenerationByIndex;
	std::vector<uint8_t> compileTrackerPublishableByIndex;
	struct CompiledResourceBatch {
		uint64_t resourceID = 0;
		unsigned int batchIndex = 0;
		bool anonymous = false;
	};
	std::vector<std::vector<CompiledResourceBatch>> compiledLastProducerBatchByResourceByQueue;
	std::vector<std::vector<CompiledResourceBatch>> compiledLastAccessBatchByResourceByQueue;
	std::vector<CompiledResourceBatch> denseCompiledProducerBatchByQueueResource;
	std::vector<CompiledResourceBatch> denseCompiledAccessBatchByQueueResource;
	uint32_t accessEpoch = 1;
	std::vector<size_t> refreshNeededMasterIndices;

	std::vector<const void*> immediateModePassPointers;
	std::vector<IHasImmediateModeCommands*> immediateModeInterfaces;

	std::vector<uint32_t> crossFrameDecisionCounts;
	std::vector<uint32_t> crossFrameDecisionOffsets;
	std::vector<uint32_t> crossFrameDecisionWriteOffsets;
	std::vector<uint32_t> crossFrameOrderedDecisionIndices;
	std::vector<uint32_t> crossFrameFirstWriteResourceEpochs;
	uint32_t crossFrameFirstWriteResourceEpoch = 1;

	bool readOnlyUniformTransitionElisionEnabled = false;
	std::vector<ResourceTransition> ignoredInternalTransitions;
};
