#pragma once

#include <memory>
#include <mutex>
#include <variant>
#include <vector>

#include <rhi.h>

#include "Render/DescriptorHeap.h"
#include "Render/Runtime/DescriptorServiceTypes.h"
#include "Resources/GPUBacking/GpuBufferBacking.h"

class GloballyIndexedResource;

class DescriptorHeapManager {
public:
	using ViewRequirements = rg::runtime::DescriptorViewRequirements;

	static DescriptorHeapManager& GetInstance() {
		static DescriptorHeapManager instance;
		return instance;
	}

	void Initialize();
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
		const ViewRequirements& req);

	void RetireDescriptorSlots(std::vector<std::pair<std::shared_ptr<DescriptorHeap>, UINT>> slots);
	void RetireBufferBacking(std::unique_ptr<GpuBufferBacking> backing);
	struct QueueFenceSnapshotPoint {
		rhi::Timeline timeline;
		uint64_t value = 0;
	};
	void PublishQueueFenceSnapshot(std::vector<QueueFenceSnapshotPoint> fenceSnapshot);
	void ProcessDeferredReleases(uint8_t frameIndex);

	rhi::DescriptorHeap GetSRVDescriptorHeap() const;
	rhi::DescriptorHeap GetSamplerDescriptorHeap() const;
	UINT CreateIndexedSampler(const rhi::SamplerDesc& samplerDesc);

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
		const ViewRequirements& req);

	std::shared_ptr<DescriptorHeap> m_cbvSrvUavHeap;
	std::shared_ptr<DescriptorHeap> m_samplerHeap;
	std::shared_ptr<DescriptorHeap> m_rtvHeap;
	std::shared_ptr<DescriptorHeap> m_dsvHeap;
	std::shared_ptr<DescriptorHeap> m_nonShaderVisibleHeap;
	struct DeferredRelease {
		std::vector<std::pair<std::shared_ptr<DescriptorHeap>, UINT>> descriptorSlots;
		std::vector<std::unique_ptr<GpuBufferBacking>> bufferBackings;
		std::vector<QueueFenceSnapshotPoint> requiredFences;
	};
	std::vector<DeferredRelease> m_deferredReleases;
	std::vector<QueueFenceSnapshotPoint> m_latestQueueFenceSnapshot;
	std::mutex m_descriptorMutationMutex;
};
