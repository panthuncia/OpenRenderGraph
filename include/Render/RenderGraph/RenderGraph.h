#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <memory>
#include <variant>
#include <span>
#include <utility>
#include <array>
#include <spdlog/spdlog.h>
#include <rhi.h>

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
#include "Render/QueueKind.h"
#include "Resources/PixelBuffer.h"
#include "Resources/Buffers/Buffer.h"
#include "Resources/TrackedAllocation.h"
#include "Render/RenderGraph/Aliasing/RenderGraphAliasingSubsystem.h"

class Resource;
class RenderPassBuilder;
class ComputePassBuilder;
class CopyPassBuilder;
class CommandRecordingManager;
struct IPassBuilder;

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
		std::optional<ExternalInsertPoint> where;
		std::variant<std::monostate, std::shared_ptr<RenderPass>, std::shared_ptr<ComputePass>, std::shared_ptr<CopyPass>> pass;
		std::optional<RenderQueueSelection> renderQueueSelection;
		std::optional<ComputeQueueSelection> computeQueueSelection;
		std::optional<CopyQueueSelection> copyQueueSelection;

		// Optional: if true, the pass will be registered in Get*PassByName().
		bool registerName = true;
		bool isGeometryPass = false; // Optional: opts pass into statistics tracking for rasterization
	};

	struct IRenderGraphExtension {
		virtual ~IRenderGraphExtension() = default;

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
		int statisticsIndex = -1;

		PassRunMask run = PassRunMask::Both; // default behavior
		std::vector<std::byte> immediateBytecode; // Stores the immediate execution bytecode
		std::shared_ptr<rg::imm::KeepAliveBag> immediateKeepAlive = nullptr; // Keeps alive resources used by immediate execution bytecode
	};

	struct ComputePassAndResources { // TODO: Same as above
		std::shared_ptr<ComputePass> pass;
		ComputePassParameters resources;
		std::string name;
		int statisticsIndex = -1;

		PassRunMask run = PassRunMask::Both;
		std::vector<std::byte> immediateBytecode; // Stores the immediate execution bytecode
		std::shared_ptr<rg::imm::KeepAliveBag> immediateKeepAlive = nullptr; // Keeps alive resources used by immediate execution bytecode
	};

	struct CopyPassAndResources {
		std::shared_ptr<CopyPass> pass;
		CopyPassParameters resources;
		std::string name;
		int statisticsIndex = -1;

		PassRunMask run = PassRunMask::Both;
		std::vector<std::byte> immediateBytecode;
		std::shared_ptr<rg::imm::KeepAliveBag> immediateKeepAlive = nullptr;
	};

	enum class BatchWaitPhase : uint8_t {
		BeforeTransitions = 0,
		BeforeExecution = 1,
		Count
	};

	enum class BatchSignalPhase : uint8_t {
		AfterTransitions = 0,
		AfterCompletion = 1,
		Count
	};

	enum class BatchTransitionPhase : uint8_t {
		BeforePasses = 0,
		AfterPasses = 1,
		Count
	};

	struct PassBatch {
		static constexpr size_t kQueueCount = static_cast<size_t>(QueueKind::Count);
		static constexpr size_t kWaitPhaseCount = static_cast<size_t>(BatchWaitPhase::Count);
		static constexpr size_t kSignalPhaseCount = static_cast<size_t>(BatchSignalPhase::Count);
		static constexpr size_t kTransitionPhaseCount = static_cast<size_t>(BatchTransitionPhase::Count);
		using QueuedPass = std::variant<RenderPassAndResources, ComputePassAndResources, CopyPassAndResources>;

		std::array<std::vector<QueuedPass>, kQueueCount> queuePasses;
		//std::unordered_map<uint64_t, ResourceAccessType> resourceAccessTypes; // Desired access types in this batch
		//std::unordered_map<uint64_t, ResourceLayout> resourceLayouts; // Desired layouts in this batch
		std::array<std::array<std::vector<ResourceTransition>, kQueueCount>, kTransitionPhaseCount> queueTransitions;

		// Resources that passes in this batch transition internally
		// Cannot be batched with other passes which use these resources
		// Ideally, would be tracked per-subresource, but that sounds hard to implement
		std::unordered_set<uint64_t> internallyTransitionedResources;
		std::unordered_set<uint64_t> allResources; // All resources used in this batch, including those that are not transitioned internally

		// Queue dependencies and signals are modeled as queue-to-queue edges per phase.
		// queueWaitEnabled[phase][dstQueue][srcQueue] + queueWaitFenceValue[phase][dstQueue][srcQueue]
		std::array<std::array<std::array<bool, kQueueCount>, kQueueCount>, kWaitPhaseCount> queueWaitEnabled{};
		std::array<std::array<std::array<UINT64, kQueueCount>, kQueueCount>, kWaitPhaseCount> queueWaitFenceValue{};

		// queueSignalEnabled[phase][queue] + queueSignalFenceValue[phase][queue]
		std::array<std::array<bool, kQueueCount>, kSignalPhaseCount> queueSignalEnabled{};
		std::array<std::array<UINT64, kQueueCount>, kSignalPhaseCount> queueSignalFenceValue{};

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

		std::unordered_map<uint64_t, SymbolicTracker*> passBatchTrackers; // Trackers for the resources in this batch
	};

	RenderGraph(rhi::Device device);
	~RenderGraph();
	using AutoAliasReasonCount = rg::alias::AutoAliasReasonCount;
	using AutoAliasPoolRangeDebug = rg::alias::AutoAliasPoolRangeDebug;
	using AutoAliasPoolDebug = rg::alias::AutoAliasPoolDebug;
	using AutoAliasDebugSnapshot = rg::alias::AutoAliasDebugSnapshot;
	AutoAliasDebugSnapshot GetAutoAliasDebugSnapshot() const;
	void AddRenderPass(std::shared_ptr<RenderPass> pass, RenderPassParameters& resources, std::string name = "");
	void AddComputePass(std::shared_ptr<ComputePass> pass, ComputePassParameters& resources, std::string name = "");
	void AddCopyPass(std::shared_ptr<CopyPass> pass, CopyPassParameters& resources, std::string name = "");
	void Update(const UpdateExecutionContext& context, rhi::Device device);
	void Execute(PassExecutionContext& context);
	void CompileStructural();
	void ResetForFrame();
	void ResetForRebuild();
	void Setup();
	void RegisterExtension(std::unique_ptr<IRenderGraphExtension> ext);
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
	//void AllocateResources(PassExecutionContext& context);
	//void CreateResource(std::wstring name);
	std::shared_ptr<Resource> GetResourceByName(const std::string& name);
	std::shared_ptr<Resource> GetResourceByID(const uint64_t id);
	std::shared_ptr<RenderPass> GetRenderPassByName(const std::string& name);
	std::shared_ptr<ComputePass> GetComputePassByName(const std::string& name);

	void RegisterProvider(IResourceProvider* prov);
	void RegisterResource(ResourceIdentifier id, std::shared_ptr<Resource> resource, IResourceProvider* provider = nullptr);

	std::unordered_map<ResourceIdentifier, std::shared_ptr<IResourceResolver>, ResourceIdentifier::Hasher> _resolverMap;

	void RegisterResolver(ResourceIdentifier id, const std::shared_ptr<IResourceResolver>& resolver);
	std::shared_ptr<IResourceResolver> RequestResolver(ResourceIdentifier const& rid, bool allowFailure = false);

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

	ComputePassBuilder& BuildComputePass(std::string const& name);
	RenderPassBuilder& BuildRenderPass(std::string const& name);
	CopyPassBuilder& BuildCopyPass(std::string const& name);

private:

	struct AnyPassAndResources {
		PassType type = PassType::Unknown;
		std::variant<std::monostate, RenderPassAndResources, ComputePassAndResources, CopyPassAndResources> pass;
		std::string name;

		AnyPassAndResources() = default;

		explicit AnyPassAndResources(RenderPassAndResources const& rp)
			: type(PassType::Render), pass(rp) {}

		explicit AnyPassAndResources(ComputePassAndResources const& cp)
			: type(PassType::Compute), pass(cp) {}

		explicit AnyPassAndResources(CopyPassAndResources const& cp)
			: type(PassType::Copy), pass(cp) {}
	};

	struct CompileContext {
		std::unordered_map<uint64_t, unsigned int> usageHistCompute;
		std::unordered_map<uint64_t, unsigned int> usageHistRender;
	};

	struct LastProducerAcrossFrames {
		QueueKind queue = QueueKind::Graphics;
		uint64_t fenceValue = 0;
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

	struct Node {
		size_t   passIndex = 0;
		QueueKind queueKind = QueueKind::Graphics;
		uint32_t originalOrder = 0;

		// Expanded IDs (aliases + group/child fixpoint)
		std::vector<uint64_t> touchedIDs;
		std::vector<uint64_t> uavIDs;

		// For dependency building: per expanded ID, strongest access in this pass.
		// Write dominates read.
		std::unordered_map<uint64_t, AccessKind> accessByID;

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
	std::unordered_map<std::string, std::shared_ptr<RenderPass>> renderPassesByName;
	std::unordered_map<std::string, std::shared_ptr<ComputePass>> computePassesByName;
	std::unordered_map<std::string, std::shared_ptr<Resource>> resourcesByName;
	std::unordered_map<uint64_t, std::shared_ptr<Resource>> resourcesByID;
	std::unordered_map<uint64_t, uint64_t> resourceBackingGenerationByID;
	std::unordered_map<uint64_t, uint32_t> resourceIdleFrameCounts;
	std::unordered_map<uint64_t, uint64_t> compiledResourceGenerationByID;
	using ResourceMaterializeOptions = std::variant<PixelBuffer::MaterializeOptions, BufferBase::MaterializeOptions>;
	std::unordered_map<uint64_t, ResourceMaterializeOptions> aliasMaterializeOptionsByID;
	std::unordered_map<uint64_t, uint64_t> aliasPlacementSignatureByID;
	std::unordered_map<uint64_t, rg::alias::AliasPlacementRange> aliasPlacementRangesByID;
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
	std::unordered_map<uint64_t, SymbolicTracker*> trackers; // Tracks the state of resources in the graph.
	std::unordered_map<uint64_t, SymbolicTracker> compileTrackers; // Compile-only symbolic state, decoupled from backing lifetime.
	std::unordered_map<uint64_t, LastProducerAcrossFrames> m_lastProducerByResourceAcrossFrames;
	std::unordered_map<uint64_t, std::vector<LastAliasPlacementProducerAcrossFrames>> m_lastAliasPlacementProducersByPoolAcrossFrames;
	std::array<std::unordered_map<uint64_t, unsigned int>, static_cast<size_t>(QueueKind::Count)> m_compiledLastProducerBatchByResourceByQueue;
	std::array<std::array<bool, static_cast<size_t>(QueueKind::Count)>, static_cast<size_t>(QueueKind::Count)> m_hasPendingFrameStartQueueWait{};
	std::array<std::array<UINT64, static_cast<size_t>(QueueKind::Count)>, static_cast<size_t>(QueueKind::Count)> m_pendingFrameStartQueueWaitFenceValue{};

	std::unique_ptr<CommandListPool> m_graphicsCommandListPool;
	std::unique_ptr<CommandListPool> m_computeCommandListPool;
	std::unique_ptr<CommandListPool> m_copyCommandListPool;

	rhi::CommandAllocatorPtr initialTransitionCommandAllocator;
	rhi::TimelinePtr m_initialTransitionFence;
	UINT64 m_initialTransitionFenceValue = 0;

	rhi::TimelinePtr m_frameStartSyncFence; // TODO: Is there a better way of handling waiting for pre-frame things like copying resources?

	rhi::TimelinePtr m_graphicsQueueFence;
	rhi::TimelinePtr m_computeQueueFence;
	rhi::TimelinePtr m_copyQueueFence;

	std::unique_ptr<CommandRecordingManager> m_pCommandRecordingManager;

	rg::imm::ImmediateDispatch m_immediateDispatch{};

	std::vector<std::unique_ptr<IRenderGraphExtension>> m_extensions;

	UINT64 m_graphicsQueueFenceValue = 0;
	UINT64 m_computeQueueFenceValue = 0;
	UINT64 m_copyQueueFenceValue = 0;
	UINT64 GetNextQueueFenceValue(QueueKind queue) {
		switch (queue) {
		case QueueKind::Graphics:
			return m_graphicsQueueFenceValue++;
		case QueueKind::Compute:
			return m_computeQueueFenceValue++;
		case QueueKind::Copy:
			return m_copyQueueFenceValue++;
		default:
			return 0;
		}
	}

	std::function<bool()> m_getUseAsyncCompute;

	void AddResource(std::shared_ptr<Resource> resource, bool transition = false);
	void MaterializeUnmaterializedResources(const std::unordered_set<uint64_t>* onlyResourceIDs = nullptr);
	SymbolicTracker& GetOrCreateCompileTracker(Resource* resource, uint64_t resourceID);
	void MaterializeReferencedResources(
		const std::vector<ResourceRequirement>& resourceRequirements,
		const std::vector<std::pair<ResourceHandleAndRange, ResourceState>>& internalTransitions);
	std::unordered_set<uint64_t> CollectFrameResourceIDs() const;
	void ApplyIdleDematerializationPolicy(const std::unordered_set<uint64_t>& usedResourceIDs);
	void SnapshotCompiledResourceGenerations(const std::unordered_set<uint64_t>& usedResourceIDs);
	void ValidateCompiledResourceGenerations() const;

	void RefreshRetainedDeclarationsForFrame(RenderPassAndResources& p, uint8_t frameIndex);
	void RefreshRetainedDeclarationsForFrame(ComputePassAndResources& p, uint8_t frameIndex);
	void RefreshRetainedDeclarationsForFrame(CopyPassAndResources& p, uint8_t frameIndex);
	void CompileFrame(rhi::Device device, uint8_t frameIndex, const IHostExecutionData* hostData);

	//void ComputeResourceLoops();
	bool IsNewBatchNeeded(
		const std::vector<ResourceRequirement>& reqs,
		const std::vector<std::pair<ResourceHandleAndRange, ResourceState>> passInternalTransitions,
		const std::unordered_map<uint64_t, SymbolicTracker*>& passBatchTrackers,
		const std::unordered_set<uint64_t>& currentBatchInternallyTransitionedResources,
		const std::unordered_set<uint64_t>& currentBatchAllResources,
		const std::unordered_set<uint64_t>& otherQueueUAVs);
	

	std::tuple<int, int, int> GetBatchesToWaitOn(const ComputePassAndResources& pass, 
		const std::unordered_map<uint64_t, unsigned int>& transitionHistory, 
		const std::unordered_map<uint64_t, unsigned int>& producerHistory,
		std::unordered_map<uint64_t, unsigned int> const& usageHistory,
		std::unordered_set<uint64_t> const& resourcesTransitionedThisPass);
    std::tuple<int, int, int> GetBatchesToWaitOn(const RenderPassAndResources& pass, 
		const std::unordered_map<uint64_t, unsigned int>& transitionHistory, 
		const std::unordered_map<uint64_t, unsigned int>& producerHistory,
		std::unordered_map<uint64_t, unsigned int> const& usageHistory,
		std::unordered_set<uint64_t> const& resourcesTransitionedThisPass);
	std::tuple<int, int, int> GetBatchesToWaitOn(const CopyPassAndResources& pass,
		const std::unordered_map<uint64_t, unsigned int>& transitionHistory,
		const std::unordered_map<uint64_t, unsigned int>& producerHistory,
		std::unordered_map<uint64_t, unsigned int> const& usageHistory,
		std::unordered_set<uint64_t> const& resourcesTransitionedThisPass);

	void ProcessResourceRequirements(
		QueueKind passQueue,
		std::vector<ResourceRequirement>& resourceRequirements,
		std::unordered_map<uint64_t, unsigned int>&  batchOfLastGraphicsQueueUsage,
		std::unordered_map<uint64_t, unsigned int>& producerHistory,
		unsigned int batchIndex,
		PassBatch& currentBatch,
		std::unordered_set<uint64_t>& outTransitionedResourceIDs);

	template<typename PassRes>
	void applySynchronization(
		QueueKind                         passQueue,
		QueueKind                         sourceQueue,
		PassBatch&                        currentBatch,
		unsigned int                      currentBatchIndex,
		const PassRes&                    pass, // either ComputePassAndResources or RenderPassAndResources
		const std::unordered_map<uint64_t, unsigned int>& oppTransHist,
		const std::unordered_map<uint64_t, unsigned int>& oppProdHist,
		const std::unordered_map<uint64_t, unsigned int>& oppUsageHist,
		const std::unordered_set<uint64_t> resourcesTransitionedThisPass)
	{
		if (passQueue == sourceQueue) {
			return;
		}

		auto markSourceCompletionSignal = [&](int batchIndex) {
			if (batchIndex < 0) {
				return;
			}
			batches[batchIndex].MarkQueueSignal(BatchSignalPhase::AfterCompletion, sourceQueue);
		};

		auto sourceCompletionFence = [&](int batchIndex) -> UINT64 {
			return batches[batchIndex].GetQueueSignalFenceValue(BatchSignalPhase::AfterCompletion, sourceQueue);
		};

		// figure out which two numbers we wait on
		auto [lastTransBatch, lastProdBatch, lastUsageBatch] =
			GetBatchesToWaitOn(pass, oppTransHist, oppProdHist, oppUsageHist, resourcesTransitionedThisPass);

		// Handle the "transition" wait
		if (lastTransBatch != -1) {
			if (static_cast<unsigned int>(lastTransBatch) == currentBatchIndex) {
				// same batch, signal & immediate wait
				currentBatch.MarkQueueSignal(BatchSignalPhase::AfterTransitions, sourceQueue);
				currentBatch.AddQueueWait(
					BatchWaitPhase::BeforeExecution,
					passQueue,
					sourceQueue,
					currentBatch.GetQueueSignalFenceValue(BatchSignalPhase::AfterTransitions, sourceQueue));
			} else {
				// different batch, signal that batch's completion, then wait before *transition*
				markSourceCompletionSignal(lastTransBatch);
				currentBatch.AddQueueWait(
					BatchWaitPhase::BeforeTransitions,
					passQueue,
					sourceQueue,
					sourceCompletionFence(lastTransBatch));
			}
		}

		// Handle the "producer" wait
#if defined(_DEBUG)
		if (lastProdBatch == currentBatchIndex) {
			spdlog::error("Producer batch is the same as current batch");
			__debugbreak();
		}
#endif
		if (lastProdBatch != -1) {
			markSourceCompletionSignal(lastProdBatch);
			currentBatch.AddQueueWait(
				BatchWaitPhase::BeforeTransitions,
				passQueue,
				sourceQueue,
				sourceCompletionFence(lastProdBatch));
		}

		// Handle the "usage" wait
		if (lastUsageBatch != -1) {
			markSourceCompletionSignal(lastUsageBatch);
			currentBatch.AddQueueWait(
				BatchWaitPhase::BeforeTransitions,
				passQueue,
				sourceQueue,
				sourceCompletionFence(lastUsageBatch));
		}
	}

	void AddTransition(
		std::unordered_map<uint64_t, unsigned int>&  batchOfLastGraphicsQueueUsage,
		unsigned int batchIndex,
		PassBatch& currentBatch,
		QueueKind passQueue,
		const ResourceRequirement& r,
		std::unordered_set<uint64_t>& outTransitionedResourceIDs);

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
	static std::vector<Node> BuildNodes(RenderGraph& rg, std::vector<AnyPassAndResources>& passes);
	static bool AddEdgeDedup(
		size_t from, size_t to,
		std::vector<Node>& nodes,
		std::unordered_set<uint64_t>& edgeSet);
	void CommitPassToBatch(
		RenderGraph& rg,
		AnyPassAndResources& pr,
		const Node& node,

		unsigned int currentBatchIndex,
		PassBatch& currentBatch,

		std::array<std::unordered_set<uint64_t>, static_cast<size_t>(QueueKind::Count)>& queueUAVs,

		std::array<std::unordered_map<uint64_t, unsigned int>, static_cast<size_t>(QueueKind::Count)>& batchOfLastQueueTransition,
		std::array<std::unordered_map<uint64_t, unsigned int>, static_cast<size_t>(QueueKind::Count)>& batchOfLastQueueProducer,
		std::array<std::unordered_map<uint64_t, unsigned int>, static_cast<size_t>(QueueKind::Count)>& batchOfLastQueueUsage);
	void AutoScheduleAndBuildBatches(
		RenderGraph& rg,
		std::vector<AnyPassAndResources>& passes,
		std::vector<Node>& nodes);

	std::unordered_map<uint64_t, uint64_t> autoAliasPoolByID;
	std::unordered_map<uint64_t, std::string> autoAliasExclusionReasonByID;
	std::vector<AutoAliasReasonCount> autoAliasExclusionReasonSummary;
	std::vector<AutoAliasPoolDebug> autoAliasPoolDebug;
	AutoAliasPlannerStats autoAliasPlannerStats;
	AutoAliasMode autoAliasModeLastFrame = AutoAliasMode::Off;
	AutoAliasPackingStrategy autoAliasPackingStrategyLastFrame = AutoAliasPackingStrategy::GreedySweepLine;
	std::function<AutoAliasMode()> m_getAutoAliasMode;
	std::function<AutoAliasPackingStrategy()> m_getAutoAliasPackingStrategy;
	std::function<bool()> m_getAutoAliasLogExclusionReasons;
	std::function<uint32_t()> m_getAutoAliasPoolRetireIdleFrames;
	std::function<float()> m_getAutoAliasPoolGrowthHeadroom;
	rg::alias::RenderGraphAliasingSubsystem m_aliasingSubsystem;

	friend class RenderPassBuilder;
	friend class ComputePassBuilder;
	friend class CopyPassBuilder;
	friend class rg::alias::RenderGraphAliasingSubsystem;
};