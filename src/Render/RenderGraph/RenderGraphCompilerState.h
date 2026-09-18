#pragma once
#include <future>
#include <deque>
#include <map>

#include "Render/RenderGraph/RenderGraph.h"
#include "Render/RenderGraph/ExperimentalGraphCompiler.h"
#include "Render/RenderGraph/ExperimentalExecutionState.h"
#include "FramePlanning.h"
#include "FrameRecording.h"


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
    // Resource-version state seed. This is realization ownership, not graph
    // caching: every frame still builds and compiles a distinct structural IR.
    // Stable concrete versions publish their immutable admission seed once.
    struct RealizationSeed {
        // The pointer used as the cache key is only an identity.  Keep a weak
        // reference so stale publication resources can be removed without the
        // seed cache extending their lifetime.
        std::weak_ptr<Resource> owner;
        uint64_t backingGeneration = 0;
        rhi::Resource capturedResource{};
        rhi::ResourceHandle resource{};
        experimental::CompileResourceShape shape{};
        rhi::HeapType heapType = rhi::HeapType::DeviceLocal;
        std::shared_ptr<const AliasHeapGeneration> aliasHeap;
        uint64_t aliasPoolID = 0;
        uint64_t aliasOffset = 0;
        uint64_t aliasSize = 0;
        std::shared_ptr<const BindlessResourceViews> bindlessViews;
        std::shared_ptr<const std::vector<experimental::PreparedStateRegion>> regions;
    };
    std::unordered_map<const Resource*, RealizationSeed> realizationSeeds;
    experimental::CompileWorkspace synchronousCompileWorkspace;
    std::shared_ptr<const experimental::DependencyEdges> frameDependencyAnalysis;
    std::map<uint32_t, std::shared_ptr<PreparedInvocationArena>> invocationArenas;
    std::shared_ptr<runtime::ITaskScope> ownershipRetirementScope;
    std::shared_ptr<runtime::ITaskScope> persistentBuildScope; // worker-side structural builds
    std::shared_ptr<const PublicationBindingBundle> graphLocalBindingBundle;

    struct RecordingFrameOwner {
        uint64_t sequence = 0;
        std::shared_ptr<const experimental::RenderFrameSnapshot> snapshot;
        std::shared_ptr<const experimental::PlannedFrameState> planning;
        std::shared_ptr<const experimental::SynchronousFramePlan> synchronousPlanning;
        std::vector<std::shared_ptr<experimental::OwnedRecordingStatistics>> statistics;
        std::optional<experimental::RecordedFrame> inlineResult;
        experimental::DispatchedFrameRecording workerRecording;

        bool Ready() const {
            if (inlineResult) return true;
            return workerRecording.Valid() && workerRecording.Ready();
        }

        experimental::RecordedFrame Join() {
            if (inlineResult) {
                auto result = std::move(*inlineResult);
                inlineResult.reset();
                return result;
            }
            if (!workerRecording.Valid()) throw std::logic_error("Recording owner has no completion");
            return workerRecording.Join();
        }
    };

    std::unique_ptr<FrameSlotPool> frameSlots;
    std::shared_ptr<FrameContext> preparingFrame;
    std::map<uint32_t, std::weak_ptr<FrameContext>> frameSlotOwners;
    void RetireCompletedFrames(QueueRegistry& queues, runtime::ITaskService* tasks = nullptr) {
        if (!asyncTimelineAdmission) return;
        std::vector<experimental::ExecutionTimelinePoint> completed;
        const auto submitted = asyncTimelineAdmission->Submitted();
        completed.reserve(submitted.size());
        for (size_t slot = 0; slot < submitted.size(); ++slot) {
            const auto value = queues.GetFence(static_cast<QueueSlotIndex>(static_cast<uint8_t>(slot))).GetCompletedValue();
            if (value == UINT64_MAX) throw std::runtime_error("Invalid GPU completion while retiring frame slots");
            completed.push_back({submitted[slot].timeline, (std::min)(submitted[slot].value, value)});
        }
        asyncTimelineAdmission->RetireCompleted(completed);
        DrainRetiredOwnership(tasks);
    }
    void DrainRetiredOwnership(runtime::ITaskService* tasks) {
        // Both execution policies retire here. Synchronous execution bypasses
        // TryAcceptFrame, so leaving garbage destruction there archives every
        // completed frame and permanently pins publication backing rings.
        auto retired = asyncTimelineAdmission->TakeRetiredGarbage();
        if (retired.empty()) return;
        auto garbage = std::make_shared<decltype(retired)>(std::move(retired));
        if (tasks && !frameWorkerScope) frameWorkerScope = tasks->CreateScope("ORG.Frame.Worker");
        if (!tasks || !frameWorkerScope || !tasks->Submit(frameWorkerScope,
            runtime::TaskPriority::Streaming, "ORG.Frame.RetiredOwnership", [garbage] {
                BT_ZONE_SCOPE("ORG.Frame.DestroyRetiredOwnership");
                basic_telemetry::Record("ORG.Frame.RetiredOwnershipDestroyed", garbage->size());
                garbage->clear();
            })) {
            basic_telemetry::AddCounter("ORG.Frame.InlineRetiredOwnershipDestruction");
            garbage->clear();
        }
    }
    std::unique_ptr<experimental::GraphCompileCoordinator> compileCoordinator;
    // Direct fresh result for BorrowedSynchronous compilation. It never enters
    // the async coordinator/reorder mailbox.
    std::shared_ptr<const experimental::CompiledGraphBundle> synchronousBundle;
    std::shared_ptr<const experimental::RenderFrameSnapshot> selectedAsyncFrame;
    std::shared_ptr<const experimental::GraphCompileInput> currentAsyncInput;
    std::unique_ptr<experimental::FramePlanningState> framePlanner;
    std::unique_ptr<experimental::SynchronousPlanningState> synchronousPlanner;
    std::shared_ptr<const experimental::PlannedFrameState> selectedPlanning;
    std::shared_ptr<const experimental::SynchronousFramePlan> selectedSynchronousPlanning;
    // BorrowedSynchronous submits directly; it never enters the asynchronous
    // recording FIFO or consumes a retained frame slot.
    std::optional<RecordingFrameOwner> synchronousRecording;
    std::deque<RecordingFrameOwner> recordingFrames;
    std::unique_ptr<experimental::PersistentRecordingLanes> recordingLanes;
    std::shared_ptr<FrameContext> pendingPresentationFrame;
    uint64_t pendingPresentationSequence = 0;
    std::shared_ptr<runtime::ITaskScope> frameWorkerScope;
    std::shared_ptr<runtime::ITaskScope> preparationWorkerScope;
    std::future<void> preparationWorker;
    bool frameProductionStopped = false;
    std::unique_ptr<experimental::ExecutionTimelineAdmission> asyncTimelineAdmission;
    uint64_t lastRequestedAsyncSequence = 0;
    uint64_t nextAsyncExecutionSequence = 1;
    uint64_t asyncPreparationFrameNumber = 0;
    std::optional<uint32_t> lastExecutedPreparationSlot;
    std::shared_ptr<const IHostExecutionData> lastSubmittedFrameData;
    uint64_t reportedAsyncSelectionFailures = 0;
    uint64_t reportedAsyncUnownedResources = 0;
    std::unordered_set<std::string> reportedAsyncLegacyPasses;
    uint64_t reportedCompileFailures = 0;
    uint64_t compileCaptureFailures = 0;
    // Measurement-only (ORG_PERSISTENT_PROBE): classifies frame-to-frame changes
    // of the lowered compile structure to size persistent executable reuse.
    struct StructureProbe {
        std::optional<experimental::GraphCompileStructure> previous;
        std::vector<std::string> previousNames;
        std::map<std::string, uint64_t> reasons;
        uint64_t frames = 0, unchanged = 0;
    };
    std::unique_ptr<StructureProbe> structureProbe;
    // Owned by RenderGraphPersistent.cpp; shared_ptr keeps the type opaque here.
    std::shared_ptr<PersistentExecutionState> persistent;

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
