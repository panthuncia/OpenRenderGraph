#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <rhi.h>

#include "Render/DescriptorHeap.h"
#include "Render/QueueKind.h"
#include "Render/Runtime/DescriptorServiceTypes.h"
#include "Resources/GPUBacking/GpuBufferBacking.h"


namespace org {

class GloballyIndexedResource;
class Resource;

class DescriptorHeapManager {
public:
	using ViewRequirements = org::runtime::DescriptorViewRequirements;

	static DescriptorHeapManager& GetInstance() {
		static DescriptorHeapManager instance;
		return instance;
	}

	void Initialize();
	void RegisterBackend(BackendInstanceId backendInstance, rhi::Device device);
	void Cleanup();

	void AssignDescriptorSlots(
		GloballyIndexedResource& target,
		rhi::Resource& apiResource,
		const ViewRequirements& req);

	void ReserveDescriptorSlots(
		GloballyIndexedResource& target,
		const ViewRequirements& req);

	void UpdateDescriptorContents(
		GloballyIndexedResource& target,
		rhi::Resource& apiResource,
		const ViewRequirements& req,
		BackendInstanceId backendInstance = BackendInstanceId::Primary);

	void RetireDescriptorSlots(std::vector<std::pair<std::shared_ptr<DescriptorHeap>, UINT>> slots);
	void RetireResource(std::shared_ptr<Resource> resource);
	// Retain a replaced imported backing until every queue represented by the
	// latest graph fence snapshot has finished using it.
	void RetireNativeResource(rhi::ResourcePtr resource);
	void RetireBufferBacking(std::unique_ptr<GpuBufferBacking> backing);
    void RetireExecutionLease(std::shared_ptr<const void> lease);
	struct QueueFenceSnapshotPoint {
		rhi::Timeline timeline;
		uint64_t value = 0;
	};
	struct DeferredReleaseStats {
		uint64_t releaseCount = 0;
		uint64_t descriptorSlotCount = 0;
		uint64_t bufferBackingCount = 0;
		uint64_t resourceCount = 0;
		uint64_t blockedReleaseCount = 0;
		uint64_t invalidTimelineCount = 0;
		uint64_t deviceErrorTimelineCount = 0;
		uint64_t incompleteTimelineCount = 0;
		std::vector<uint64_t> resourceIDs;
	};
	void PublishQueueFenceSnapshot(std::vector<QueueFenceSnapshotPoint> fenceSnapshot);
	std::vector<QueueFenceSnapshotPoint> GetQueueFenceSnapshot() const;
	void ProcessDeferredReleases(uint8_t frameIndex);
	DeferredReleaseStats GetDeferredReleaseStats();
	// The caller must have waited for the device to become idle. Releases every
	// pending object and forgets queue-fence snapshots before their owning queue
	// registry destroys and recreates its timelines.
	void DrainDeferredReleasesAfterDeviceIdle();

	rhi::DescriptorHeap GetSRVDescriptorHeap(BackendInstanceId backendInstance = BackendInstanceId::Primary) const;
	rhi::DescriptorHeap GetSamplerDescriptorHeap(BackendInstanceId backendInstance = BackendInstanceId::Primary) const;
	rhi::DescriptorHeap GetRTVDescriptorHeap(BackendInstanceId backendInstance = BackendInstanceId::Primary) const;
	rhi::DescriptorHeap GetDSVDescriptorHeap(BackendInstanceId backendInstance = BackendInstanceId::Primary) const;
	rhi::DescriptorHeap GetNonShaderVisibleDescriptorHeap(BackendInstanceId backendInstance = BackendInstanceId::Primary) const;
	rhi::DescriptorSlot ResolveDescriptorSlot(BackendInstanceId backendInstance, rhi::DescriptorHeapType type,
		bool shaderVisible, uint32_t logicalIndex) const;
	UINT CreateIndexedSampler(const rhi::SamplerDesc& samplerDesc);
	rhi::DescriptorSlot AllocateDescriptorSlot(rhi::DescriptorHeapType type, bool shaderVisible);
	void RetireDescriptorSlot(rhi::DescriptorSlot slot);

	const std::shared_ptr<DescriptorHeap>& GetCBVSRVUAVHeap() const { return m_cbvSrvUavHeap; }
	const std::shared_ptr<DescriptorHeap>& GetSamplerHeap() const { return m_samplerHeap; }
	const std::shared_ptr<DescriptorHeap>& GetRTVHeap() const { return m_rtvHeap; }
	const std::shared_ptr<DescriptorHeap>& GetDSVHeap() const { return m_dsvHeap; }
	const std::shared_ptr<DescriptorHeap>& GetNonShaderVisibleHeap() const { return m_nonShaderVisibleHeap; }

private:
	DescriptorHeapManager() = default;

	void ReserveDescriptorSlotsUnlocked(
		GloballyIndexedResource& target,
		const ViewRequirements& req);

	void UpdateDescriptorContentsUnlocked(
		GloballyIndexedResource& target,
		rhi::Resource& apiResource,
		const ViewRequirements& req,
		BackendInstanceId backendInstance);

	struct BackendHeaps {
		rhi::Device device{};
		std::shared_ptr<DescriptorHeap> cbvSrvUav;
		std::shared_ptr<DescriptorHeap> sampler;
		std::shared_ptr<DescriptorHeap> rtv;
		std::shared_ptr<DescriptorHeap> dsv;
		std::shared_ptr<DescriptorHeap> nonShaderVisible;
	};
	BackendHeaps* FindBackendHeaps(BackendInstanceId backendInstance);
	const BackendHeaps* FindBackendHeaps(BackendInstanceId backendInstance) const;

	std::shared_ptr<DescriptorHeap> m_cbvSrvUavHeap;
	std::shared_ptr<DescriptorHeap> m_samplerHeap;
	std::shared_ptr<DescriptorHeap> m_rtvHeap;
	std::shared_ptr<DescriptorHeap> m_dsvHeap;
	std::shared_ptr<DescriptorHeap> m_nonShaderVisibleHeap;
	std::unordered_map<uint8_t, BackendHeaps> m_backendHeaps;
	// Logical sampler indices are part of shader ABI. Retain recipes so a
	// backend registered after renderer startup receives identical contents.
	std::unordered_map<UINT, rhi::SamplerDesc> m_indexedSamplerDescriptions;
	struct DeferredRelease {
		std::vector<std::pair<std::shared_ptr<DescriptorHeap>, UINT>> descriptorSlots;
		std::vector<std::unique_ptr<GpuBufferBacking>> bufferBackings;
		std::vector<std::shared_ptr<Resource>> resources;
		std::vector<rhi::ResourcePtr> nativeResources;
        std::vector<std::shared_ptr<const void>> executionLeases;
		std::vector<QueueFenceSnapshotPoint> requiredFences;
	};
	std::vector<DeferredRelease> m_deferredReleases;
	std::unordered_set<const Resource*> m_deferredResourcePointers;
	std::vector<QueueFenceSnapshotPoint> m_latestQueueFenceSnapshot;
	mutable std::mutex m_descriptorMutationMutex;
};


} // namespace org
