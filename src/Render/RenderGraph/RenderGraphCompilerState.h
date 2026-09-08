#pragma once

#include "Render/RenderGraph/RenderGraph.h"
#include "Render/RenderGraph/ExperimentalGraphCompiler.h"
#include "Render/RenderGraph/ExperimentalExecutionState.h"


// Private compiler representation. Keep compiler-only data here rather than in
// RenderGraph's installed public header so compile algorithm changes have a
// narrow incremental-build footprint.
namespace org {

namespace experimental { struct RenderFrameSnapshot; struct FramePreparationBasis; }

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
    std::unique_ptr<experimental::GraphCompileCoordinator> shadowCompiler;
    std::shared_ptr<const experimental::RenderFrameSnapshot> selectedAsyncFrame;
    std::shared_ptr<const experimental::GraphCompileInput> currentAsyncInput;
    experimental::BackingStateAdmissionLedger asyncBackingStateLedger;
    experimental::BackingAccessAdmissionLedger asyncBackingAccessLedger;
    experimental::AliasAccessAdmissionLedger asyncAliasAccessLedger;
    std::unique_ptr<experimental::ExecutionTimelineAdmission> asyncTimelineAdmission;
    uint64_t lastRequestedAsyncSequence = 0;
    uint64_t nextAsyncExecutionSequence = 1;
    uint64_t asyncPreparationFrameNumber = 0;
    std::optional<uint32_t> lastExecutedPreparationSlot;
    uint64_t reportedAsyncSelectionFailures = 0;
    uint64_t reportedAsyncUnownedResources = 0;
    std::unordered_set<std::string> reportedAsyncLegacyPasses;
    uint64_t reportedShadowFailures = 0;
    uint64_t shadowCaptureFailures = 0;

	std::vector<Node> nodes;
	std::vector<compiler::DependencySequence<size_t>> dependencySeqStates;
	std::vector<uint64_t> dependencyEdgeKeys;
	std::vector<uint32_t> dependencyIndegrees;
	std::vector<size_t> dependencyTopoOrder;
	std::vector<size_t> dependencyReadyHeap;

	std::vector<uint64_t> resourceIDs;
	std::vector<uint64_t> densePassAccessKeys;
	std::vector<uint8_t> resourcesWritten;
	std::vector<uint32_t> accessEpochs;
	std::vector<uint32_t> accessWriteEpochs;
	std::vector<uint32_t> accessUavEpochs;
	std::vector<uint32_t> accessDagEpochs;
	std::vector<uint32_t> accessOrder;
	std::vector<uint32_t> schedulingSummaryResourceEpochs;
	std::vector<uint32_t> schedulingSummaryWriteEpochs;
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

	struct PendingFrameInsert {
		AnyPassAndResources* pass = nullptr;
		size_t slotIndex = 0;
		size_t nextInsertIndex = std::numeric_limits<size_t>::max();
	};
	std::vector<ExternalPassDesc> frameExtensions;
	std::unordered_set<std::string> frameExtensionPassNames;
	std::vector<std::pair<std::string, std::string>> frameExplicitAfterByName;
	std::vector<std::pair<std::string_view, size_t>> explicitPassNameIndices;
	std::vector<std::pair<size_t, size_t>> explicitEdges;
	std::vector<PendingFrameInsert> pendingFrameInserts;
	std::vector<size_t> frameInsertSlotHeads;
	std::vector<size_t> frameInsertSlotTails;
	// Frame-extension names are assembled from temporary/materialized strings in
	// CompileFrame.  These tables outlive each loop iteration, so their keys must
	// own their storage; string_view keys here previously dangled as soon as the
	// local insertedPassName was destroyed and corrupted subsequent lookups.
	std::unordered_map<std::string, size_t> pendingInsertIndexByName;
	std::unordered_map<std::string, size_t> pendingInsertTailByAnchorName;

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


} // namespace org
