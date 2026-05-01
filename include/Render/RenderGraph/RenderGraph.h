#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <string_view>
#include <memory>
#include <optional>
#include <variant>
#include <span>
#include <utility>
#include <array>
#include <spdlog/spdlog.h>
#include <rhi.h>
#include <tracy/Tracy.hpp>

#include "RenderPasses/Base/RenderPass.h"
#include "RenderPasses/Base/ComputePass.h"
#include "RenderPasses/Base/CopyPass.h"
#include "Resources/ResourceStateTracker.h"
#include "Interfaces/IResourceProvider.h"
#include "Render/ResourceRegistry.h"
#include "Render/CommandListPool.h"
#include "Interfaces/IPassBuilder.h"
#include "Render/MemoryIntrospectionAPI.h"
#include "Render/Runtime/IStatisticsService.h"
#include "Render/Runtime/IUploadService.h"
#include "Render/Runtime/IReadbackService.h"
#include "Render/Runtime/IDescriptorService.h"
#include "Render/Runtime/IRenderGraphSettingsService.h"
#include "Render/Runtime/ITaskService.h"
#include "Render/QueueKind.h"
#include "Render/QueueRegistry.h"
#include "Resources/PixelBuffer.h"
#include "Resources/Buffers/Buffer.h"
#include "Resources/TrackedAllocation.h"
#include "Render/RenderGraph/Aliasing/RenderGraphAliasingSubsystem.h"
#include "Render/RenderGraph/ExecutionSchedule.h"
#include "Interfaces/IResourceResolver.h"

class Resource;
class RenderPassBuilder;
class ComputePassBuilder;
class CopyPassBuilder;
class CommandRecordingManager;
struct IPassBuilder;

namespace ui {
	struct FrameGraphSnapshot;
}

template<typename T>
concept DerivedResource = std::derived_from<T, Resource>;

enum class PassRunMask : uint8_t;
[[nodiscard]] constexpr PassRunMask operator|(PassRunMask a, PassRunMask b) noexcept;

enum class PassRunMask : uint8_t {
	None = 0,
	Immediate = 1u << 0,
	Retained = 1u << 1,
	Both = Immediate | Retained
};

constexpr uint8_t to_u8(PassRunMask v) noexcept {
	return static_cast<uint8_t>(v);
}

[[nodiscard]] constexpr PassRunMask operator&(PassRunMask a, PassRunMask b) noexcept {
	return static_cast<PassRunMask>(to_u8(a) & to_u8(b));
}

[[nodiscard]] constexpr PassRunMask operator|(PassRunMask a, PassRunMask b) noexcept {
	return static_cast<PassRunMask>(to_u8(a) | to_u8(b));
}

[[nodiscard]] constexpr PassRunMask operator^(PassRunMask a, PassRunMask b) noexcept {
	return static_cast<PassRunMask>(to_u8(a) ^ to_u8(b));
}

[[nodiscard]] constexpr PassRunMask operator~(PassRunMask a) noexcept {
	return static_cast<PassRunMask>(~to_u8(a));
}

constexpr PassRunMask& operator&=(PassRunMask& a, PassRunMask b) noexcept { return a = (a & b); }
constexpr PassRunMask& operator|=(PassRunMask& a, PassRunMask b) noexcept { return a = (a | b); }
constexpr PassRunMask& operator^=(PassRunMask& a, PassRunMask b) noexcept { return a = (a ^ b); }

enum class AutoAliasMode : uint8_t {
	Off = 0,
	Conservative = 1,
	Balanced = 2,
	Aggressive = 3
};

enum class AutoAliasPackingStrategy : uint8_t {
	GreedySweepLine = 0,
	BranchAndBound = 1,
};

class RenderGraph {
public:

	enum class ExternalInsertKind : uint8_t { Begin, End, Before, After };

	struct ExternalInsertPoint {
		int priority = 0;

		// default: preserve extension-local emission order
		bool keepExtensionOrder = true;

		std::vector<std::string> after;   // anchor keys that must precede this pass
		std::vector<std::string> before;  // anchor keys that must follow this pass

		static ExternalInsertPoint Begin(int prio = 0) { ExternalInsertPoint p; p.priority = prio; p.before.push_back("__rg_begin__"); return p; }
		static ExternalInsertPoint End(int prio = 0) { ExternalInsertPoint p; p.priority = prio; p.after.push_back("__rg_end__");   return p; }

		static ExternalInsertPoint After(std::string a, int prio = 0) { ExternalInsertPoint p; p.priority = prio; p.after.push_back(std::move(a)); return p; }
		static ExternalInsertPoint Before(std::string a, int prio = 0) { ExternalInsertPoint p; p.priority = prio; p.before.push_back(std::move(a)); return p; }

		static ExternalInsertPoint Between(std::string a, std::string b, int prio = 0) {
			ExternalInsertPoint p; p.priority = prio;
			p.after.push_back(std::move(a));
			p.before.push_back(std::move(b));
			return p;
		}

		ExternalInsertPoint& AlsoAfter(std::string a) { after.push_back(std::move(a)); return *this; }
		ExternalInsertPoint& AlsoBefore(std::string a) { before.push_back(std::move(a)); return *this; }
	};

	enum class PassType {
		Unknown,
		Render,
		Compute,
		Copy
	};

	struct ExternalPassDesc {
		PassType type = PassType::Unknown;
		std::string name;
		std::string techniquePath;
		std::optional<ExternalInsertPoint> where;
		std::variant<std::monostate, std::shared_ptr<RenderPass>, std::shared_ptr<ComputePass>, std::shared_ptr<CopyPass>> pass;
		std::optional<QueueKind> preferredQueueKind;
		std::optional<QueueAssignmentPolicy> queueAssignmentPolicy;
		std::optional<QueueSlotIndex> pinnedQueueSlot; // Target a specific queue slot, bypassing queue preference

		// Optional: if true, the pass will be registered in Get*PassByName().
		bool registerName = true;
		bool isGeometryPass = false; // Optional: opts pass into statistics tracking for rasterization
		bool collectStatistics = true;

		static ExternalPassDesc Render(std::string name, std::shared_ptr<RenderPass> renderPass) {
			ExternalPassDesc desc{};
			desc.type = PassType::Render;
			desc.name = std::move(name);
			desc.pass = std::move(renderPass);
			return desc;
		}

		static ExternalPassDesc Compute(std::string name, std::shared_ptr<ComputePass> computePass) {
			ExternalPassDesc desc{};
			desc.type = PassType::Compute;
			desc.name = std::move(name);
			desc.pass = std::move(computePass);
			return desc;
		}

		static ExternalPassDesc Copy(std::string name, std::shared_ptr<CopyPass> copyPass) {
			ExternalPassDesc desc{};
			desc.type = PassType::Copy;
			desc.name = std::move(name);
			desc.pass = std::move(copyPass);
			return desc;
		}

		ExternalPassDesc& At(ExternalInsertPoint insertPoint) & {
			where = std::move(insertPoint);
			return *this;
		}

		ExternalPassDesc At(ExternalInsertPoint insertPoint) && {
			where = std::move(insertPoint);
			return std::move(*this);
		}

		ExternalPassDesc& PreferQueue(QueueKind queueKind) & {
			preferredQueueKind = queueKind;
			queueAssignmentPolicy = QueueAssignmentPolicy::ForcePreferred;
			return *this;
		}

		ExternalPassDesc PreferQueue(QueueKind queueKind) && {
			preferredQueueKind = queueKind;
			queueAssignmentPolicy = QueueAssignmentPolicy::ForcePreferred;
			return std::move(*this);
		}

		ExternalPassDesc& AutomaticQueueAssignment() & {
			queueAssignmentPolicy = QueueAssignmentPolicy::Automatic;
			return *this;
		}

		ExternalPassDesc AutomaticQueueAssignment() && {
			queueAssignmentPolicy = QueueAssignmentPolicy::Automatic;
			return std::move(*this);
		}

		ExternalPassDesc& PinToQueue(QueueSlotIndex queueSlot) & {
			pinnedQueueSlot = queueSlot;
			return *this;
		}

		ExternalPassDesc PinToQueue(QueueSlotIndex queueSlot) && {
			pinnedQueueSlot = queueSlot;
			return std::move(*this);
		}

		ExternalPassDesc& RegisterByName(bool enabled = true) & {
			registerName = enabled;
			return *this;
		}

		ExternalPassDesc RegisterByName(bool enabled = true) && {
			registerName = enabled;
			return std::move(*this);
		}

		ExternalPassDesc& GeometryPass(bool enabled = true) & {
			isGeometryPass = enabled;
			return *this;
		}

		ExternalPassDesc GeometryPass(bool enabled = true) && {
			isGeometryPass = enabled;
			return std::move(*this);
		}

		ExternalPassDesc& Technique(std::string path) & {
			techniquePath = std::move(path);
			return *this;
		}

		ExternalPassDesc Technique(std::string path) && {
			techniquePath = std::move(path);
			return std::move(*this);
		}

		ExternalPassDesc& CollectStatistics(bool enabled = true) & {
			collectStatistics = enabled;
			return *this;
		}

		ExternalPassDesc CollectStatistics(bool enabled = true) && {
			collectStatistics = enabled;
			return std::move(*this);
		}
	};

	struct IRenderGraphExtension {
		virtual ~IRenderGraphExtension() = default;
		virtual void PrepareForBuild(RenderGraph& rg) { (void)rg; }

		// Called once from RenderGraph::Setup() after queue registry is populated.
		// Extensions can create queues, allocate upload instances, etc.
		virtual void Initialize(RenderGraph& rg) { (void)rg; }

		// lets systems react to registry recreation without RenderGraph including them
		virtual void OnRegistryReset(ResourceRegistry* registry) {}

		// main hook: inject passes
		virtual void GatherStructuralPasses(RenderGraph& rg, std::vector<ExternalPassDesc>& out) = 0;

		// per-frame hook: inject ephemeral passes (e.g. readback captures)
		// Default: no per-frame passes.
		virtual void GatherFramePasses(RenderGraph& rg, std::vector<ExternalPassDesc>& out) { (void)rg; (void)out; }
	};

	inline bool Has(PassRunMask m, PassRunMask f) {
		return (uint8_t(m) & uint8_t(f)) != 0;
	}

	struct RenderPassAndResources { // TODO: I'm currently copying these a lot; maybe use pointers instead
		std::shared_ptr<RenderPass> pass;
		RenderPassParameters resources;
		std::string name;
		std::string techniquePath;
		int statisticsIndex = -1;
		bool collectStatistics = true;

		PassRunMask run = PassRunMask::Both; // default behavior
		std::vector<std::byte> immediateBytecode; // Stores the immediate execution bytecode
		std::shared_ptr<rg::imm::KeepAliveBag> immediateKeepAlive = nullptr; // Keeps alive resources used by immediate execution bytecode
		std::vector<std::shared_ptr<Resource>> retainedAnonymousKeepAlive; // Keeps retained anonymous handles alive across frames
		std::vector<ResolverSnapshot> resolverSnapshots; // Versioned resolver snapshots for auto-invalidation
	};

	struct ComputePassAndResources { // TODO: Same as above
		std::shared_ptr<ComputePass> pass;
		ComputePassParameters resources;
		std::string name;
		std::string techniquePath;
		int statisticsIndex = -1;
		bool collectStatistics = true;

		PassRunMask run = PassRunMask::Both;
		std::vector<std::byte> immediateBytecode; // Stores the immediate execution bytecode
		std::shared_ptr<rg::imm::KeepAliveBag> immediateKeepAlive = nullptr; // Keeps alive resources used by immediate execution bytecode
		std::vector<std::shared_ptr<Resource>> retainedAnonymousKeepAlive; // Keeps retained anonymous handles alive across frames
		std::vector<ResolverSnapshot> resolverSnapshots; // Versioned resolver snapshots for auto-invalidation
	};

	struct CopyPassAndResources {
		std::shared_ptr<CopyPass> pass;
		CopyPassParameters resources;
		std::string name;
		std::string techniquePath;
		int statisticsIndex = -1;
		bool collectStatistics = true;

		PassRunMask run = PassRunMask::Both;
		std::vector<std::byte> immediateBytecode;
		std::shared_ptr<rg::imm::KeepAliveBag> immediateKeepAlive = nullptr;
		std::vector<std::shared_ptr<Resource>> retainedAnonymousKeepAlive; // Keeps retained anonymous handles alive across frames
		std::vector<ResolverSnapshot> resolverSnapshots; // Versioned resolver snapshots for auto-invalidation
	};

	enum class BatchWaitPhase : uint8_t {
		BeforeTransitions = 0,
		BeforeExecution = 1,
		BeforeAfterPasses = 2,
		Count
	};

	enum class BatchSignalPhase : uint8_t {
		AfterTransitions = 0,
		AfterExecution = 1,
		AfterCompletion = 2,
		Count
	};

	enum class BatchTransitionPhase : uint8_t {
		BeforePasses = 0,
		AfterPasses = 1,
		Count
	};

	struct PassBatch {
		static constexpr size_t kQueueCount = static_cast<size_t>(QueueKind::Count); // Legacy; prefer QueueCount()
		static constexpr size_t kWaitPhaseCount = static_cast<size_t>(BatchWaitPhase::Count);
		static constexpr size_t kSignalPhaseCount = static_cast<size_t>(BatchSignalPhase::Count);
		static constexpr size_t kTransitionPhaseCount = static_cast<size_t>(BatchTransitionPhase::Count);
		using QueuedPass = std::variant<RenderPassAndResources*, ComputePassAndResources*, CopyPassAndResources*>;
		using TransitionPositionList = std::vector<uint32_t>;
		using TransitionIndexByResource = std::vector<TransitionPositionList>;

		PassBatch(size_t queueCount = kQueueCount, size_t resourceCount = 0)
			: queuePasses(queueCount) {
			for (auto& v : queueTransitions) v.resize(queueCount);
			for (auto& phaseBuckets : transitionPositionsByResource) {
				phaseBuckets.assign(queueCount, TransitionIndexByResource(resourceCount));
			}
			passBatchTrackersByResourceIndex.assign(resourceCount, nullptr);
			for (auto& p : queueWaitEnabled) p.assign(queueCount, std::vector<uint8_t>(queueCount, 0));
			for (auto& p : queueWaitFenceValue) p.assign(queueCount, std::vector<UINT64>(queueCount, 0));
			for (auto& p : queueSignalEnabled) p.assign(queueCount, 0);
			for (auto& p : queueSignalFenceValue) p.assign(queueCount, 0);
		}

		size_t QueueCount() const noexcept { return queuePasses.size(); }
		size_t TransitionIndexedResourceCount() const noexcept {
			if (transitionPositionsByResource.empty() || transitionPositionsByResource.front().empty()) {
				return 0;
			}
			return transitionPositionsByResource.front().front().size();
		}

		std::vector<std::vector<QueuedPass>> queuePasses;
		std::array<std::vector<std::vector<ResourceTransition>>, kTransitionPhaseCount> queueTransitions;
		std::array<std::vector<TransitionIndexByResource>, kTransitionPhaseCount> transitionPositionsByResource;

		// Resources that passes in this batch transition internally
		// Cannot be batched with other passes which use these resources
		// Ideally, would be tracked per-subresource, but that sounds hard to implement
		// Kept sorted for O(log N) lookup via binary_search.
		std::vector<uint64_t> internallyTransitionedResources;
		std::vector<uint64_t> allResources; // All resources used in this batch, including those that are not transitioned internally

		// Queue dependencies and signals are modeled as queue-to-queue edges per phase.
		// queueWaitEnabled[phase][dstQueue][srcQueue] + queueWaitFenceValue[phase][dstQueue][srcQueue]
		std::array<std::vector<std::vector<uint8_t>>, kWaitPhaseCount> queueWaitEnabled;
		std::array<std::vector<std::vector<UINT64>>, kWaitPhaseCount> queueWaitFenceValue;

		// queueSignalEnabled[phase][queue] + queueSignalFenceValue[phase][queue]
		std::array<std::vector<uint8_t>, kSignalPhaseCount> queueSignalEnabled;
		std::array<std::vector<UINT64>, kSignalPhaseCount> queueSignalFenceValue;

		static constexpr size_t QueueIndex(QueueKind queue) noexcept {
			return static_cast<size_t>(queue);
		}

		static constexpr size_t WaitPhaseIndex(BatchWaitPhase phase) noexcept {
			return static_cast<size_t>(phase);
		}

		static constexpr size_t SignalPhaseIndex(BatchSignalPhase phase) noexcept {
			return static_cast<size_t>(phase);
		}

		static constexpr size_t TransitionPhaseIndex(BatchTransitionPhase phase) noexcept {
			return static_cast<size_t>(phase);
		}

		std::vector<QueuedPass>& Passes(QueueKind queue) {
			return queuePasses[QueueIndex(queue)];
		}

		const std::vector<QueuedPass>& Passes(QueueKind queue) const {
			return queuePasses[QueueIndex(queue)];
		}

		bool HasPasses(QueueKind queue) const {
			return !Passes(queue).empty();
		}

		std::vector<ResourceTransition>& Transitions(QueueKind queue, BatchTransitionPhase phase) {
			return queueTransitions[TransitionPhaseIndex(phase)][QueueIndex(queue)];
		}

		const std::vector<ResourceTransition>& Transitions(QueueKind queue, BatchTransitionPhase phase) const {
			return queueTransitions[TransitionPhaseIndex(phase)][QueueIndex(queue)];
		}

		bool HasTransitions(QueueKind queue, BatchTransitionPhase phase) const {
			return !Transitions(queue, phase).empty();
		}

		void SetQueueSignalFenceValue(BatchSignalPhase phase, QueueKind queue, UINT64 fenceValue) {
			queueSignalFenceValue[SignalPhaseIndex(phase)][QueueIndex(queue)] = fenceValue;
		}

		UINT64 GetQueueSignalFenceValue(BatchSignalPhase phase, QueueKind queue) const {
			return queueSignalFenceValue[SignalPhaseIndex(phase)][QueueIndex(queue)];
		}

		void MarkQueueSignal(BatchSignalPhase phase, QueueKind queue) {
			queueSignalEnabled[SignalPhaseIndex(phase)][QueueIndex(queue)] = true;
		}

		void ClearQueueSignal(BatchSignalPhase phase, QueueKind queue) {
			queueSignalEnabled[SignalPhaseIndex(phase)][QueueIndex(queue)] = false;
		}

		bool HasQueueSignal(BatchSignalPhase phase, QueueKind queue) const {
			return queueSignalEnabled[SignalPhaseIndex(phase)][QueueIndex(queue)];
		}

		void AddQueueWait(BatchWaitPhase phase, QueueKind dstQueue, QueueKind srcQueue, UINT64 fenceValue) {
			if (dstQueue == srcQueue) {
				return;
			}

			auto& enabled = queueWaitEnabled[WaitPhaseIndex(phase)][QueueIndex(dstQueue)][QueueIndex(srcQueue)];
			auto& maxFence = queueWaitFenceValue[WaitPhaseIndex(phase)][QueueIndex(dstQueue)][QueueIndex(srcQueue)];
			enabled = true;
			if (fenceValue > maxFence) {
				maxFence = fenceValue;
			}
		}

		void ClearQueueWait(BatchWaitPhase phase, QueueKind dstQueue, QueueKind srcQueue) {
			queueWaitEnabled[WaitPhaseIndex(phase)][QueueIndex(dstQueue)][QueueIndex(srcQueue)] = false;
			queueWaitFenceValue[WaitPhaseIndex(phase)][QueueIndex(dstQueue)][QueueIndex(srcQueue)] = 0;
		}

		bool HasQueueWait(BatchWaitPhase phase, QueueKind dstQueue, QueueKind srcQueue) const {
			return queueWaitEnabled[WaitPhaseIndex(phase)][QueueIndex(dstQueue)][QueueIndex(srcQueue)];
		}

		UINT64 GetQueueWaitFenceValue(BatchWaitPhase phase, QueueKind dstQueue, QueueKind srcQueue) const {
			return queueWaitFenceValue[WaitPhaseIndex(phase)][QueueIndex(dstQueue)][QueueIndex(srcQueue)];
		}

		// ---- Index-based overloads for N-queue support ----

		std::vector<QueuedPass>& Passes(size_t qi) { return queuePasses[qi]; }
		const std::vector<QueuedPass>& Passes(size_t qi) const { return queuePasses[qi]; }
		bool HasPasses(size_t qi) const { return !queuePasses[qi].empty(); }

		std::vector<ResourceTransition>& Transitions(size_t qi, BatchTransitionPhase phase) {
			return queueTransitions[TransitionPhaseIndex(phase)][qi];
		}
		const std::vector<ResourceTransition>& Transitions(size_t qi, BatchTransitionPhase phase) const {
			return queueTransitions[TransitionPhaseIndex(phase)][qi];
		}
		bool HasTransitions(size_t qi, BatchTransitionPhase phase) const {
			return !queueTransitions[TransitionPhaseIndex(phase)][qi].empty();
		}

		void ResetTransitionIndexes(size_t resourceCount) {
			for (auto& phaseBuckets : transitionPositionsByResource) {
				phaseBuckets.assign(QueueCount(), TransitionIndexByResource(resourceCount));
			}
			passBatchTrackersByResourceIndex.assign(resourceCount, nullptr);
		}

		void RecordTransitionPosition(size_t qi, BatchTransitionPhase phase, size_t resourceIndex, uint32_t transitionIndex) {
			auto& byResource = transitionPositionsByResource[TransitionPhaseIndex(phase)][qi];
			if (resourceIndex >= byResource.size()) {
				return;
			}
			byResource[resourceIndex].push_back(transitionIndex);
		}

		TransitionPositionList& TransitionPositions(size_t qi, BatchTransitionPhase phase, size_t resourceIndex) {
			return transitionPositionsByResource[TransitionPhaseIndex(phase)][qi][resourceIndex];
		}

		const TransitionPositionList& TransitionPositions(size_t qi, BatchTransitionPhase phase, size_t resourceIndex) const {
			return transitionPositionsByResource[TransitionPhaseIndex(phase)][qi][resourceIndex];
		}

		void SetQueueSignalFenceValue(BatchSignalPhase phase, size_t qi, UINT64 fenceValue) {
			queueSignalFenceValue[SignalPhaseIndex(phase)][qi] = fenceValue;
		}
		UINT64 GetQueueSignalFenceValue(BatchSignalPhase phase, size_t qi) const {
			return queueSignalFenceValue[SignalPhaseIndex(phase)][qi];
		}
		void MarkQueueSignal(BatchSignalPhase phase, size_t qi) {
			queueSignalEnabled[SignalPhaseIndex(phase)][qi] = 1;
		}
		void ClearQueueSignal(BatchSignalPhase phase, size_t qi) {
			queueSignalEnabled[SignalPhaseIndex(phase)][qi] = 0;
		}
		bool HasQueueSignal(BatchSignalPhase phase, size_t qi) const {
			return queueSignalEnabled[SignalPhaseIndex(phase)][qi] != 0;
		}

		void AddQueueWait(BatchWaitPhase phase, size_t dst, size_t src, UINT64 fenceValue) {
			if (dst == src) return;
			auto& enabled = queueWaitEnabled[WaitPhaseIndex(phase)][dst][src];
			auto& maxFence = queueWaitFenceValue[WaitPhaseIndex(phase)][dst][src];
			enabled = 1;
			if (fenceValue > maxFence) maxFence = fenceValue;
		}
		void ClearQueueWait(BatchWaitPhase phase, size_t dst, size_t src) {
			queueWaitEnabled[WaitPhaseIndex(phase)][dst][src] = 0;
			queueWaitFenceValue[WaitPhaseIndex(phase)][dst][src] = 0;
		}
		bool HasQueueWait(BatchWaitPhase phase, size_t dst, size_t src) const {
			return queueWaitEnabled[WaitPhaseIndex(phase)][dst][src] != 0;
		}
		UINT64 GetQueueWaitFenceValue(BatchWaitPhase phase, size_t dst, size_t src) const {
			return queueWaitFenceValue[WaitPhaseIndex(phase)][dst][src];
		}

		std::vector<SymbolicTracker*> passBatchTrackersByResourceIndex; // Trackers for the resources in this batch, keyed by frame scheduling resource index
	};

	RenderGraph(rhi::Device device);
	~RenderGraph();
	using AutoAliasReasonCount = rg::alias::AutoAliasReasonCount;
	using AutoAliasPoolRangeDebug = rg::alias::AutoAliasPoolRangeDebug;
	using AutoAliasPoolDebug = rg::alias::AutoAliasPoolDebug;
	using AutoAliasDebugSnapshot = rg::alias::AutoAliasDebugSnapshot;
	AutoAliasDebugSnapshot GetAutoAliasDebugSnapshot() const;
	void BuildMemoryIntrospectionFrameGraphSnapshot(
		ui::FrameGraphSnapshot& out,
		const std::vector<rg::memory::ResourceMemoryRecord>& memoryRecords) const;
	void AddRenderPass(std::shared_ptr<RenderPass> pass, RenderPassParameters& resources, std::string name = "", std::vector<ResolverSnapshot> resolverSnapshots = {});
	void AddComputePass(std::shared_ptr<ComputePass> pass, ComputePassParameters& resources, std::string name = "", std::vector<ResolverSnapshot> resolverSnapshots = {});
	void AddCopyPass(std::shared_ptr<CopyPass> pass, CopyPassParameters& resources, std::string name = "", std::vector<ResolverSnapshot> resolverSnapshots = {});
	void Update(const UpdateExecutionContext& context, rhi::Device device);
	void Execute(PassExecutionContext& context);
	void CompileStructural();
	void ResetForFrame();
	void ResetForRebuild();
	void PrepareExtensionsForBuild();
	void Setup();
	void RegisterExtension(std::unique_ptr<IRenderGraphExtension> ext, std::optional<std::string_view> id = std::nullopt);
	const std::vector<PassBatch>& GetBatches() const { return batches; }
	rg::memory::SnapshotProvider& GetMemorySnapshotProvider() { return m_memorySnapshotProvider; }
	const rg::memory::SnapshotProvider& GetMemorySnapshotProvider() const { return m_memorySnapshotProvider; }
	void SetStatisticsService(std::shared_ptr<rg::runtime::IStatisticsService> service) { m_statisticsService = std::move(service); }
	rg::runtime::IStatisticsService* GetStatisticsService() { return m_statisticsService.get(); }
	const rg::runtime::IStatisticsService* GetStatisticsService() const { return m_statisticsService.get(); }
	void SetUploadService(std::shared_ptr<rg::runtime::IUploadService> service) { m_uploadService = std::move(service); }
	rg::runtime::IUploadService* GetUploadService() { return m_uploadService.get(); }
	const rg::runtime::IUploadService* GetUploadService() const { return m_uploadService.get(); }
	void SetReadbackService(std::shared_ptr<rg::runtime::IReadbackService> service) { m_readbackService = std::move(service); }
	rg::runtime::IReadbackService* GetReadbackService() { return m_readbackService.get(); }
	const rg::runtime::IReadbackService* GetReadbackService() const { return m_readbackService.get(); }
	void SetDescriptorService(std::shared_ptr<rg::runtime::IDescriptorService> service) { m_descriptorService = std::move(service); }
	rg::runtime::IDescriptorService* GetDescriptorService() { return m_descriptorService.get(); }
	const rg::runtime::IDescriptorService* GetDescriptorService() const { return m_descriptorService.get(); }
	void SetRenderGraphSettingsService(std::shared_ptr<rg::runtime::IRenderGraphSettingsService> service) { m_renderGraphSettingsService = std::move(service); }
	rg::runtime::IRenderGraphSettingsService* GetRenderGraphSettingsService() { return m_renderGraphSettingsService.get(); }
	const rg::runtime::IRenderGraphSettingsService* GetRenderGraphSettingsService() const { return m_renderGraphSettingsService.get(); }
	void SetTaskService(std::shared_ptr<rg::runtime::ITaskService> service) { m_taskService = std::move(service); }
	rg::runtime::ITaskService* GetTaskService() { return m_taskService.get(); }
	const rg::runtime::ITaskService* GetTaskService() const { return m_taskService.get(); }
	void SetStructuralMaterializeCheckpointCallback(std::function<void(std::string_view)> callback) {
		m_structuralMaterializeCheckpointCallback = std::move(callback);
	}
	void SetStructuralMaterializeResourceCheckpointCallback(std::function<void(std::string_view, std::string_view)> callback) {
		m_structuralMaterializeResourceCheckpointCallback = std::move(callback);
	}
	//void AllocateResources(PassExecutionContext& context);
	//void CreateResource(std::wstring name);
	std::shared_ptr<Resource> GetResourceByName(const std::string& name);
	std::shared_ptr<Resource> GetResourceByID(const uint64_t id);
	std::shared_ptr<RenderPass> GetRenderPassByName(const std::string& name);
	std::shared_ptr<ComputePass> GetComputePassByName(const std::string& name);

	void RegisterProvider(IResourceProvider* prov);
	void EnsureProviderRegistered(IResourceProvider* prov);
	void RegisterResource(ResourceIdentifier id, std::shared_ptr<Resource> resource, IResourceProvider* provider = nullptr);

	std::unordered_map<ResourceIdentifier, std::shared_ptr<IResourceResolver>, ResourceIdentifier::Hasher> _resolverMap;

	void RegisterResolver(ResourceIdentifier id, const std::shared_ptr<IResourceResolver>& resolver);
	std::shared_ptr<IResourceResolver> RequestResolver(ResourceIdentifier const& rid, bool allowFailure = false);
	void SetPassTechnique(std::string passName, std::string techniquePath);

	std::shared_ptr<Resource> RequestResourcePtr(ResourceIdentifier const& rid, bool allowFailure = false);
	ResourceRegistry::RegistryHandle RequestResourceHandle(ResourceIdentifier const& rid, bool allowFailure = false);
	ResourceRegistry::RegistryHandle RequestResourceHandle(Resource* const& pResource, bool allowFailure = false);

	//void RegisterECSRenderPhaseEntities(const std::unordered_map<RenderPhase, flecs::entity, RenderPhase::Hasher>& phaseEntities);

	template<DerivedResource T>
	std::shared_ptr<T> RequestResourcePtr(ResourceIdentifier const& rid, bool allowFailure = false) {
		auto basePtr = RequestResourcePtr(rid, allowFailure);

		if (!basePtr) {
			if (allowFailure) {
				return nullptr;
			}

			throw std::runtime_error(
				"RequestResource<" + std::string(typeid(T).name()) +
				">: underlying Resource* is null (rid = " + rid.ToString() + ")"
			);
		}

		auto derivedPtr = std::dynamic_pointer_cast<T>(basePtr);
		if (!derivedPtr) {
			throw std::runtime_error(
				"Requested resource is not a " + std::string(typeid(T).name()) +
				": " + rid.ToString()
			);
		}

		return derivedPtr;
	}

	template<typename PassT, rg::PassInputs InputsT, typename... StableCtorArgs>
	ComputePassBuilder& BuildComputePass(std::string const& name, InputsT&& inputs, StableCtorArgs&&... ctorArgs);

	template<typename PassT, typename... StableCtorArgs>
	ComputePassBuilder& BuildComputePass(std::string const& name, StableCtorArgs&&... ctorArgs);

	template<typename PassT, rg::PassInputs InputsT, typename... StableCtorArgs>
	RenderPassBuilder& BuildRenderPass(std::string const& name, InputsT&& inputs, StableCtorArgs&&... ctorArgs);

	template<typename PassT, typename... StableCtorArgs>
	RenderPassBuilder& BuildRenderPass(std::string const& name, StableCtorArgs&&... ctorArgs);

	template<typename PassT, rg::PassInputs InputsT, typename... StableCtorArgs>
	CopyPassBuilder& BuildCopyPass(std::string const& name, InputsT&& inputs, StableCtorArgs&&... ctorArgs);

	template<typename PassT, typename... StableCtorArgs>
	CopyPassBuilder& BuildCopyPass(std::string const& name, StableCtorArgs&&... ctorArgs);

	/// Create a new queue of the given kind and register it with the render graph.
	/// Must be called after Setup() and before the first Execute().
	QueueSlotIndex CreateQueue(
		QueueKind kind,
		const char* name,
		QueueAutoAssignmentPolicy autoAssignmentPolicy = QueueAutoAssignmentPolicy::AllowAutomaticScheduling);
	void SetMinimumAutomaticSchedulingQueues(QueueKind kind, uint8_t count);
	uint8_t GetMinimumAutomaticSchedulingQueues(QueueKind kind) const noexcept {
		return m_minAutomaticSchedulingQueuesByKind[static_cast<size_t>(kind)];
	}

	/// Access the queue registry (read-only).
	const QueueRegistry& GetQueueRegistry() const noexcept { return m_queueRegistry; }
	QueueRegistry& GetQueueRegistry() noexcept { return m_queueRegistry; }

private:
	std::string GetTechniquePathForPassName(std::string_view passName) const;

	struct AnyPassAndResources {
		PassType type = PassType::Unknown;
		std::variant<std::monostate, RenderPassAndResources, ComputePassAndResources, CopyPassAndResources> pass;
		std::string name;

		AnyPassAndResources() = default;

		explicit AnyPassAndResources(RenderPassAndResources const& rp)
			: type(PassType::Render), pass(rp) {}
		explicit AnyPassAndResources(RenderPassAndResources&& rp)
			: type(PassType::Render), pass(std::move(rp)) {}

		explicit AnyPassAndResources(ComputePassAndResources const& cp)
			: type(PassType::Compute), pass(cp) {}
		explicit AnyPassAndResources(ComputePassAndResources&& cp)
			: type(PassType::Compute), pass(std::move(cp)) {}

		explicit AnyPassAndResources(CopyPassAndResources const& cp)
			: type(PassType::Copy), pass(cp) {}
		explicit AnyPassAndResources(CopyPassAndResources&& cp)
			: type(PassType::Copy), pass(std::move(cp)) {}
	};

	struct CompileContext {
		std::unordered_map<uint64_t, unsigned int> usageHistCompute;
		std::unordered_map<uint64_t, unsigned int> usageHistRender;
	};

	struct LastProducerAcrossFrames {
		size_t queueSlot = 0;
		uint64_t fenceValue = 0;
		uint64_t publishSerial = 0;
		bool anonymous = false;
	};

	struct LastAliasPlacementProducerAcrossFrames {
		uint64_t resourceID = 0;
		uint64_t poolID = 0;
		uint64_t poolGeneration = 0;
		uint64_t startByte = 0;
		uint64_t endByte = 0;
		LastProducerAcrossFrames producer{};
	};
	
	enum class AccessKind : uint8_t { Read, Write };

	struct DenseEquivalentResourceSummary {
		uint64_t resourceID = 0;
		size_t resourceIndex = 0;
		std::vector<size_t> equivalentResourceIndices;
	};

	struct DenseRequirementSummary {
		ResourceRegistry::RegistryHandle resource;
		uint64_t resourceID = 0;
		size_t resourceIndex = 0;
		RangeSpec range{};
		ResourceState state{};
		bool isUAV = false;
		std::vector<size_t> equivalentResourceIndices;
	};

	struct FramePassSchedulingSummary {
		std::vector<DenseRequirementSummary> requirements;
		std::vector<DenseEquivalentResourceSummary> internalTransitions;
		std::vector<size_t> requiredResourceIndices;
		std::vector<size_t> touchedResourceIndices;
		std::vector<size_t> uavResourceIndices;
	};

	struct FrameResourceEventSummary {
		unsigned int latestBatch = 0;
		unsigned int previousBatch = 0;
		uint64_t latestQueueMask = 0;
		uint64_t previousQueueMask = 0;
	};

	struct BatchBuildState {
		size_t queueCount = 0;
		size_t resourceCount = 0;
		uint32_t nodeEpoch = 1;
		uint32_t resourceEpoch = 1;
		uint32_t internalTransitionEpoch = 1;
		uint32_t otherQueueUAVEpoch = 1;
		std::vector<uint32_t> nodeEpochByIndex;
		std::vector<uint32_t> resourceEpochByIndex;
		std::vector<uint32_t> internalTransitionEpochByIndex;
		std::vector<uint32_t> otherQueueUAVEpochByOffset;

		void Initialize(size_t nodeCount, size_t inQueueCount, size_t inResourceCount) {
			queueCount = inQueueCount;
			resourceCount = inResourceCount;
			nodeEpoch = 1;
			resourceEpoch = 1;
			internalTransitionEpoch = 1;
			otherQueueUAVEpoch = 1;
			nodeEpochByIndex.assign(nodeCount, 0);
			resourceEpochByIndex.assign(resourceCount, 0);
			internalTransitionEpochByIndex.assign(resourceCount, 0);
			otherQueueUAVEpochByOffset.assign(queueCount * resourceCount, 0);
		}

		void ResetForNewBatch() {
			AdvanceEpoch(nodeEpochByIndex, nodeEpoch);
			AdvanceEpoch(resourceEpochByIndex, resourceEpoch);
			AdvanceEpoch(internalTransitionEpochByIndex, internalTransitionEpoch);
			AdvanceEpoch(otherQueueUAVEpochByOffset, otherQueueUAVEpoch);
		}

		bool ContainsNode(size_t nodeIndex) const {
			return nodeIndex < nodeEpochByIndex.size() && nodeEpochByIndex[nodeIndex] == nodeEpoch;
		}

		void MarkNode(size_t nodeIndex) {
			if (nodeIndex < nodeEpochByIndex.size()) {
				nodeEpochByIndex[nodeIndex] = nodeEpoch;
			}
		}

		bool ContainsResource(size_t resourceIndex) const {
			return resourceIndex < resourceEpochByIndex.size() && resourceEpochByIndex[resourceIndex] == resourceEpoch;
		}

		void MarkResource(size_t resourceIndex) {
			if (resourceIndex < resourceEpochByIndex.size()) {
				resourceEpochByIndex[resourceIndex] = resourceEpoch;
			}
		}

		bool ContainsInternalTransition(size_t resourceIndex) const {
			return resourceIndex < internalTransitionEpochByIndex.size()
				&& internalTransitionEpochByIndex[resourceIndex] == internalTransitionEpoch;
		}

		void MarkInternalTransition(size_t resourceIndex) {
			if (resourceIndex < internalTransitionEpochByIndex.size()) {
				internalTransitionEpochByIndex[resourceIndex] = internalTransitionEpoch;
			}
		}

		bool ContainsOtherQueueUAV(size_t queueIndex, size_t resourceIndex) const {
			if (queueIndex >= queueCount || resourceIndex >= resourceCount) {
				return false;
			}
			return otherQueueUAVEpochByOffset[queueIndex * resourceCount + resourceIndex] == otherQueueUAVEpoch;
		}

		void MarkOtherQueueUAV(size_t queueIndex, size_t resourceIndex) {
			if (queueIndex >= queueCount || resourceIndex >= resourceCount) {
				return;
			}
			otherQueueUAVEpochByOffset[queueIndex * resourceCount + resourceIndex] = otherQueueUAVEpoch;
		}

	private:
		static void AdvanceEpoch(std::vector<uint32_t>& values, uint32_t& epoch) {
			++epoch;
			if (epoch == 0) {
				std::fill(values.begin(), values.end(), 0);
				epoch = 1;
			}
		}
	};

	struct Node {
		size_t   passIndex = 0;
		size_t   queueSlot = 0; // Default/preferred queue slot for compatibility-preserving fallback
		QueueKind preferredQueueKind = QueueKind::Graphics;
		QueueAssignmentPolicy queueAssignmentPolicy = QueueAssignmentPolicy::ForcePreferred;
		std::vector<size_t> compatibleQueueSlots; // All legal queue slots for this pass
		uint8_t compatibleQueueKindMask = 0;
		std::optional<size_t> assignedQueueSlot; // Final slot chosen during frame scheduling
		uint32_t originalOrder = 0;

		// Expanded IDs (aliases + group/child fixpoint)
		std::vector<uint64_t> touchedIDs;
		std::vector<uint64_t> uavIDs;

		// For dependency building: per expanded ID, strongest access in this pass.
		// Write dominates read. Sorted by ID after BuildNodes.
		std::vector<std::pair<uint64_t, AccessKind>> accessByID;

		// DAG
		std::vector<size_t> out;
		std::vector<size_t> in;
		uint32_t indegree = 0;

		// Longest-path-to-sink (for tie-breaking)
		uint32_t criticality = 0;
	};

	std::vector<IResourceProvider*> _providers;
	ResourceRegistry _registry;
	std::unordered_map<ResourceIdentifier, IResourceProvider*, ResourceIdentifier::Hasher> _providerMap;

	std::vector<IPassBuilder*> m_passBuilderOrder;
	std::unordered_map<std::string, std::unique_ptr<IPassBuilder>> m_passBuildersByName;
	std::unordered_set<std::string> m_passNamesSeenThisReset;

	std::vector<AnyPassAndResources> m_masterPassList;
	std::vector<AnyPassAndResources> m_framePasses;
	std::vector<size_t> m_assignedQueueSlotsByFramePass;
	std::vector<uint8_t> m_activeQueueSlotsThisFrame;
	std::array<uint8_t, static_cast<size_t>(QueueKind::Count)> m_minAutomaticSchedulingQueuesByKind = { 1, 1, 1 };
	std::unordered_map<std::string, std::shared_ptr<RenderPass>> renderPassesByName;
	std::unordered_map<std::string, std::shared_ptr<ComputePass>> computePassesByName;
	std::unordered_map<std::string, std::string> m_passTechniquePathsByName;
	std::unordered_map<std::string, std::shared_ptr<Resource>> resourcesByName;
	std::unordered_map<uint64_t, std::shared_ptr<Resource>> resourcesByID;
	std::unordered_map<uint64_t, std::shared_ptr<Resource>> m_transientFrameResourcesByID;
	std::unordered_map<std::string, std::shared_ptr<Resource>> m_transientFrameResourcesByName;
	std::unordered_map<uint64_t, uint64_t> resourceBackingGenerationByID;
	std::unordered_map<uint64_t, uint32_t> resourceIdleFrameCounts;
	std::unordered_map<uint64_t, uint64_t> compiledResourceGenerationByID;
	using ResourceMaterializeOptions = std::variant<PixelBuffer::MaterializeOptions, BufferBase::MaterializeOptions>;
	std::unordered_map<uint64_t, ResourceMaterializeOptions> aliasMaterializeOptionsByID;
	std::unordered_map<uint64_t, uint64_t> aliasPlacementSignatureByID;
	std::unordered_map<uint64_t, rg::alias::AliasPlacementRange> aliasPlacementRangesByID;
	std::unordered_map<uint64_t, rg::alias::AliasPlacementRange> schedulingPlacementRangesByID;
	std::unordered_map<uint64_t, std::vector<uint64_t>> m_schedulingEquivalentIDsCache;
	std::unordered_map<uint64_t, size_t> m_frameSchedulingResourceIndexByID;
	size_t m_frameSchedulingResourceCount = 0;
	std::vector<unsigned int> m_frameQueueLastUsageBatch;
	std::vector<unsigned int> m_frameQueueLastProducerBatch;
	std::vector<unsigned int> m_frameQueueLastTransitionBatch;
	std::vector<std::vector<unsigned int>> m_frameTransitionPlacementBatchesByResource;
	std::vector<FrameResourceEventSummary> m_frameResourceEventSummaries;
	std::vector<FramePassSchedulingSummary> m_framePassSchedulingSummaries;
	std::unordered_map<uint64_t, uint64_t> aliasPlacementPoolByID;
	std::unordered_set<uint64_t> aliasActivationPending;

	using PersistentAliasPoolState = rg::alias::PersistentAliasPoolState;
	std::unordered_map<uint64_t, PersistentAliasPoolState> persistentAliasPools;
	uint64_t aliasPoolPlanFrameIndex = 0;
	uint32_t aliasPoolRetireIdleFrames = 120;
	float aliasPoolGrowthHeadroom = 1.5f;

	std::unordered_map<uint64_t, ResourceTransition> initialTransitions; // Transitions needed to reach the initial state of the resources before executing the first batch. Executed on graph setup.
	std::vector<PassBatch> batches;
	rg::memory::SnapshotProvider m_memorySnapshotProvider;
	std::shared_ptr<rg::runtime::IStatisticsService> m_statisticsService;
	std::shared_ptr<rg::runtime::IUploadService> m_uploadService;
	std::shared_ptr<rg::runtime::IReadbackService> m_readbackService;
	std::shared_ptr<rg::runtime::IDescriptorService> m_descriptorService;
	std::shared_ptr<rg::runtime::IRenderGraphSettingsService> m_renderGraphSettingsService;
	std::shared_ptr<rg::runtime::ITaskService> m_taskService;
	std::unordered_map<uint64_t, SymbolicTracker*> trackers; // Tracks the state of resources in the graph.
	std::unordered_map<uint64_t, SymbolicTracker> compileTrackers; // Compile-only symbolic state, decoupled from backing lifetime.
	std::unordered_map<uint64_t, LastProducerAcrossFrames> m_lastProducerByResourceAcrossFrames;
	std::unordered_map<uint64_t, std::vector<LastAliasPlacementProducerAcrossFrames>> m_lastAliasPlacementProducersByPoolAcrossFrames;
	std::vector<std::unordered_map<uint64_t, unsigned int>> m_compiledLastProducerBatchByResourceByQueue;
	uint64_t m_crossFrameProducerPublishSerial = 0;
	std::vector<std::vector<uint8_t>> m_hasPendingFrameStartQueueWait;
	std::vector<std::vector<UINT64>> m_pendingFrameStartQueueWaitFenceValue;

	QueueRegistry m_queueRegistry;

	rhi::CommandAllocatorPtr initialTransitionCommandAllocator;
	rhi::TimelinePtr m_initialTransitionFence;
	UINT64 m_initialTransitionFenceValue = 0;

	rhi::TimelinePtr m_frameStartSyncFence; // TODO: Is there a better way of handling waiting for pre-frame things like copying resources?

	rhi::TimelinePtr m_readbackFence;

	std::unique_ptr<CommandRecordingManager> m_pCommandRecordingManager;
	ExecutionSchedule m_executionSchedule;

	void BuildExecutionSchedule();
	void ResizeQueueParallelVectors();
	void EnsureMinimumAutomaticSchedulingQueues();

	rg::imm::ImmediateDispatch m_immediateDispatch{};

	std::vector<std::unique_ptr<IRenderGraphExtension>> m_extensions;
    std::vector<std::string> m_extensionRegistrationIds;

	UINT64 GetNextQueueFenceValue(size_t queueSlotIndex) {
		return m_queueRegistry.GetNextFenceValue(static_cast<QueueSlotIndex>(static_cast<uint8_t>(queueSlotIndex)));
	}

	std::function<bool()> m_getUseAsyncCompute;
	std::function<bool()> m_getHeavyDebug;

	void AddResource(std::shared_ptr<Resource> resource, bool transition = false);
	void TrackTransientFrameResource(const std::shared_ptr<Resource>& resource);
	void TrackTransientFrameResource(Resource* resource);
	void ShutdownOwnedState();

	/// Dispatches to the injected ITaskService when available, otherwise runs a serial loop.
	void ParallelForOptional(std::string_view taskName, size_t itemCount, std::function<void(size_t)> func, bool override = false) {
		if (m_taskService && !override) {
			m_taskService->ParallelFor(taskName, itemCount, std::move(func));
		} else {
			for (size_t i = 0; i < itemCount; ++i) {
				func(i);
			}
		}
	}

	void MaterializeUnmaterializedResources(const std::unordered_set<uint64_t>* onlyResourceIDs = nullptr);
	SymbolicTracker& GetOrCreateCompileTracker(Resource* resource, uint64_t resourceID);
	void CaptureCompileTrackersForExecution(const std::unordered_set<uint64_t>& resourceIDs);
	void PublishCompiledTrackerStates();
	void MaterializeReferencedResources(
		const std::vector<ResourceRequirement>& resourceRequirements,
		const std::vector<std::pair<ResourceHandleAndRange, ResourceState>>& internalTransitions,
		std::string_view debugPassName = {});
	std::vector<std::shared_ptr<Resource>> CaptureRetainedAnonymousKeepAlive(
		const std::vector<ResourceRequirement>& resourceRequirements,
		const std::vector<std::pair<ResourceHandleAndRange, ResourceState>>& internalTransitions) const;
	void CollectFrameResourceIDs(std::unordered_set<uint64_t>& out) const;
	void ApplyIdleDematerializationPolicy(const std::unordered_set<uint64_t>& usedResourceIDs);
	void SnapshotCompiledResourceGenerations(const std::unordered_set<uint64_t>& usedResourceIDs);
	void ValidateCompiledResourceGenerations() const;
	void RebuildSchedulingEquivalentIDCache(const std::unordered_set<uint64_t>& resourceIDs);
	void RebuildFrameSchedulingResourceIndex(const std::unordered_set<uint64_t>& resourceIDs);
	void RebuildFramePassSchedulingSummaries();
	void ClearFrameSchedulingResourceIndex();
	void ClearFramePassSchedulingSummaries();
	void ResetFrameQueueBatchHistoryTables();
	std::optional<size_t> TryGetFrameSchedulingResourceIndex(uint64_t resourceID) const;
	size_t FrameQueueBatchHistoryOffset(size_t queueSlot, size_t resourceIndex) const;
	unsigned int GetFrameQueueHistoryValue(const std::vector<unsigned int>& history, size_t queueSlot, size_t resourceIndex) const;
	void SetFrameQueueHistoryValue(std::vector<unsigned int>& history, size_t queueSlot, size_t resourceIndex, unsigned int batchIndex);
	void RecordFrameResourceEvent(size_t queueSlot, size_t resourceIndex, unsigned int batchIndex);
	void RecordFrameTransitionPlacementBatch(size_t resourceIndex, unsigned int batchIndex);
	void RecordFrameQueueUsageBatch(size_t queueSlot, size_t resourceIndex, unsigned int batchIndex);
	void RecordFrameQueueTransitionBatch(size_t queueSlot, size_t resourceIndex, unsigned int batchIndex);
	std::pair<unsigned int, uint64_t> GetFrameResourceLastEventBeforeBatch(size_t resourceIndex, unsigned int batchIndex) const;
	const std::vector<uint64_t>& GetSchedulingEquivalentIDsCached(uint64_t resourceID);

	void RefreshRetainedDeclarationsForFrame(RenderPassAndResources& p, uint8_t frameIndex);
	void RefreshRetainedDeclarationsForFrame(ComputePassAndResources& p, uint8_t frameIndex);
	void RefreshRetainedDeclarationsForFrame(CopyPassAndResources& p, uint8_t frameIndex);
	void CompileFrame(rhi::Device device, uint8_t frameIndex, const IHostExecutionData* hostData);
	void WriteCompiledGraphDebugDump(uint8_t frameIndex, const std::vector<Node>& nodes) const;
	void WriteVramUsageDebugDump(uint8_t frameIndex) const;
	AnyPassAndResources MaterializeExternalPass(const ExternalPassDesc& desc, bool callSetup, bool materializeReferencedResources);
	void RegisterExternalPassName(const ExternalPassDesc& desc, AnyPassAndResources& any);

	//void ComputeResourceLoops();
	bool IsNewBatchNeeded(
		const FramePassSchedulingSummary& passSummary,
		const std::vector<SymbolicTracker*>& passBatchTrackersByResourceIndex,
		const BatchBuildState& batchBuildState,
		std::string_view candidatePassName,
		unsigned int currentBatchIndex,
		size_t candidateQueueSlot);
	

	std::tuple<int, int, int> GetBatchesToWaitOn(const ComputePassAndResources& pass,
		size_t sourceQueueSlot,
		std::unordered_set<uint64_t> const& resourcesTransitionedThisPass);
	std::tuple<int, int, int> GetBatchesToWaitOn(const RenderPassAndResources& pass,
		size_t sourceQueueSlot,
		std::unordered_set<uint64_t> const& resourcesTransitionedThisPass);
	std::tuple<int, int, int> GetBatchesToWaitOn(const CopyPassAndResources& pass,
		size_t sourceQueueSlot,
		std::unordered_set<uint64_t> const& resourcesTransitionedThisPass);

	void ProcessResourceRequirements(
		size_t passQueueSlot,
		const std::vector<DenseRequirementSummary>& resourceRequirements,
		std::string_view passName,
		unsigned int batchIndex,
		PassBatch& currentBatch,
		std::unordered_set<uint64_t>& outTransitionedResourceIDs,
		std::unordered_set<size_t>& outFallbackResourceIndices,
		std::vector<ResourceTransition>& scratchTransitions);

	template<typename PassRes>
	void applySynchronization(
		size_t                            passQueueSlot,
		size_t                            sourceQueueSlot,
		PassBatch&                        currentBatch,
		unsigned int                      currentBatchIndex,
		const PassRes&                    pass, // either ComputePassAndResources or RenderPassAndResources
		const std::unordered_set<uint64_t> resourcesTransitionedThisPass)
	{
		ZoneScopedN("RenderGraph::applySynchronization");
		if (!pass.name.empty()) {
			ZoneText(pass.name.data(), pass.name.size());
		}
		if (passQueueSlot == sourceQueueSlot) {
			return;
		}

		auto markSourceCompletionSignal = [&](int batchIndex) {
			if (batchIndex <= 0) {
				return;
			}

			if (static_cast<unsigned int>(batchIndex) == currentBatchIndex) {
				currentBatch.MarkQueueSignal(BatchSignalPhase::AfterCompletion, sourceQueueSlot);
				return;
			}

			batches[batchIndex].MarkQueueSignal(BatchSignalPhase::AfterCompletion, sourceQueueSlot);
		};

		auto sourceCompletionFence = [&](int batchIndex) -> UINT64 {
			if (batchIndex <= 0) {
				return 0;
			}

			if (static_cast<unsigned int>(batchIndex) == currentBatchIndex) {
				return currentBatch.GetQueueSignalFenceValue(BatchSignalPhase::AfterCompletion, sourceQueueSlot);
			}

			return batches[batchIndex].GetQueueSignalFenceValue(BatchSignalPhase::AfterCompletion, sourceQueueSlot);
		};

		// figure out which two numbers we wait on
		auto [lastTransBatch, lastProdBatch, lastUsageBatch] =
			GetBatchesToWaitOn(pass, sourceQueueSlot, resourcesTransitionedThisPass);

		// Handle the "transition" wait
		if (lastTransBatch != -1) {
			if (static_cast<unsigned int>(lastTransBatch) == currentBatchIndex) {
				// same batch, signal & immediate wait
				currentBatch.MarkQueueSignal(BatchSignalPhase::AfterTransitions, sourceQueueSlot);
				currentBatch.AddQueueWait(
					BatchWaitPhase::BeforeExecution,
					passQueueSlot,
					sourceQueueSlot,
					currentBatch.GetQueueSignalFenceValue(BatchSignalPhase::AfterTransitions, sourceQueueSlot));
			} else {
				// different batch, signal that batch's completion, then wait before *transition*
				const UINT64 completionFence = sourceCompletionFence(lastTransBatch);
				if (completionFence != 0) {
					markSourceCompletionSignal(lastTransBatch);
					currentBatch.AddQueueWait(
						BatchWaitPhase::BeforeTransitions,
						passQueueSlot,
						sourceQueueSlot,
						completionFence);
				}
			}
		}

		// Handle the "producer" wait
		// Writes are also recorded into the transition history for scheduling.
		// Only preserve a separate producer wait if it would extend beyond the
		// transition wait we already emitted.
		if (lastProdBatch != -1 && lastProdBatch > lastTransBatch) {
			const UINT64 completionFence = sourceCompletionFence(lastProdBatch);
			if (completionFence != 0) {
				markSourceCompletionSignal(lastProdBatch);
				currentBatch.AddQueueWait(
					BatchWaitPhase::BeforeTransitions,
					passQueueSlot,
					sourceQueueSlot,
					completionFence);
			}
		}

		// Handle the "usage" wait
		if (lastUsageBatch != -1) {
			const UINT64 completionFence = sourceCompletionFence(lastUsageBatch);
			if (completionFence != 0) {
				markSourceCompletionSignal(lastUsageBatch);
				currentBatch.AddQueueWait(
					BatchWaitPhase::BeforeTransitions,
					passQueueSlot,
					sourceQueueSlot,
					completionFence);
			}
		}
	}

	void AddTransition(
		unsigned int batchIndex,
		PassBatch& currentBatch,
		size_t passQueueSlot,
		std::string_view passName,
		const DenseRequirementSummary& requirement,
		std::unordered_set<uint64_t>& outTransitionedResourceIDs,
		std::unordered_set<size_t>& outFallbackResourceIndices,
		std::vector<ResourceTransition>& scratchTransitions);
	void AppendBatchTransition(
		PassBatch& batch,
		unsigned int batchIndex,
		size_t queueSlot,
		BatchTransitionPhase phase,
		size_t resourceIndex,
		const ResourceTransition& transition);

	static inline bool IsUAVState(const ResourceState& s) noexcept {
		return ((s.access & rhi::ResourceAccessType::UnorderedAccess) != 0) ||
			(s.layout == rhi::ResourceLayout::UnorderedAccess);
	}

	struct PassView {
		bool isCompute = false;
		std::vector<ResourceRequirement>* reqs = nullptr;
		std::vector<std::pair<ResourceHandleAndRange, ResourceState>>* internalTransitions = nullptr;
	};

	struct SeqState {
		std::optional<size_t> lastWriter;
		std::vector<size_t>   readsSinceWrite;
	};

	using AutoAliasPlannerStats = rg::alias::AutoAliasPlannerStats;

	static PassView GetPassView(AnyPassAndResources& pr);
	static bool BuildDependencyGraph(std::vector<Node>& nodes);
	static bool BuildDependencyGraph(std::vector<Node>& nodes, std::span<const std::pair<size_t, size_t>> explicitEdges);
	static bool FinalizeDependencyGraph(std::vector<Node>& nodes);
	static std::vector<Node> BuildNodes(RenderGraph& rg, std::vector<AnyPassAndResources>& passes);
	static std::vector<uint8_t> PlanActiveQueueSlots(RenderGraph& rg, const std::vector<AnyPassAndResources>& passes, const std::vector<Node>& nodes);
	static bool AddEdgeDedup(
		size_t from, size_t to,
		std::vector<Node>& nodes,
		std::unordered_set<uint64_t>& edgeSet);
	bool AddCurrentFrameAliasSchedulingEdges(std::vector<Node>& nodes);
	void CommitPassToBatch(
		RenderGraph& rg,
		AnyPassAndResources& pr,
		const Node& node,

		unsigned int currentBatchIndex,
		PassBatch& currentBatch,
		std::unordered_set<uint64_t>& scratchTransitioned,
		std::unordered_set<size_t>& scratchFallback,
		std::vector<ResourceTransition>& scratchTransitions);
	void AutoScheduleAndBuildBatches(
		RenderGraph& rg,
		std::vector<AnyPassAndResources>& passes,
		std::vector<Node>& nodes);

	std::unordered_map<uint64_t, uint64_t> autoAliasPoolByID;
	std::unordered_map<uint64_t, std::string> autoAliasExclusionReasonByID;
	std::vector<AutoAliasReasonCount> autoAliasExclusionReasonSummary;
	std::vector<AutoAliasPoolDebug> autoAliasPoolDebug;
	AutoAliasPlannerStats autoAliasPlannerStats;
	AutoAliasMode autoAliasPreviousMode = AutoAliasMode::Off;
	AutoAliasMode autoAliasModeLastFrame = AutoAliasMode::Off;
	AutoAliasPackingStrategy autoAliasPackingStrategyLastFrame = AutoAliasPackingStrategy::GreedySweepLine;
	std::function<AutoAliasMode()> m_getAutoAliasMode;
	std::function<AutoAliasPackingStrategy()> m_getAutoAliasPackingStrategy;
	std::function<bool()> m_getRenderGraphCompileDumpEnabled;
	std::function<bool()> m_getRenderGraphVramDumpEnabled;
	std::function<bool()> m_getRenderGraphBatchTraceEnabled;
	std::function<bool()> m_getAutoAliasEnableLogging;
	std::function<bool()> m_getAutoAliasLogExclusionReasons;
	std::function<bool()> m_getQueueSchedulingEnableLogging;
	std::function<float()> m_getQueueSchedulingWidthScale;
	std::function<float()> m_getQueueSchedulingPenaltyBias;
	std::function<float()> m_getQueueSchedulingMinPenalty;
	std::function<float()> m_getQueueSchedulingResourcePressureWeight;
	std::function<float()> m_getQueueSchedulingUavPressureWeight;
	std::function<float()> m_getQueueSchedulingAutoGraphicsBias;
	std::function<float()> m_getQueueSchedulingAsyncOverlapBonus;
	std::function<float()> m_getQueueSchedulingCrossQueueHandoffPenalty;
	std::function<uint32_t()> m_getAutoAliasPoolRetireIdleFrames;
	std::function<float()> m_getAutoAliasPoolGrowthHeadroom;
	std::function<void(std::string_view)> m_structuralMaterializeCheckpointCallback;
	std::function<void(std::string_view, std::string_view)> m_structuralMaterializeResourceCheckpointCallback;
	rg::alias::RenderGraphAliasingSubsystem m_aliasingSubsystem;

	ComputePassBuilder& GetOrCreateComputePassBuilder(std::string const& name);
	RenderPassBuilder& GetOrCreateRenderPassBuilder(std::string const& name);
	CopyPassBuilder& GetOrCreateCopyPassBuilder(std::string const& name);

	friend class RenderPassBuilder;
	friend class ComputePassBuilder;
	friend class CopyPassBuilder;
	friend class rg::alias::RenderGraphAliasingSubsystem;
};

#include "Render/PassBuilders.h"
