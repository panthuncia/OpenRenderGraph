#include "Managers/Singletons/UploadManager.h"

#include <cstring>
#include <sstream>
#include <unordered_set>

#include <spdlog/spdlog.h>

#include "Render/PassBuilders.h"
#include "Render/Runtime/OpenRenderGraphSettings.h"

namespace {
	RangeSpec SingleSubresourceRange(uint32_t mip, uint32_t slice) noexcept
	{
		RangeSpec range;
		range.mipLower = { BoundType::Exact, mip };
		range.mipUpper = { BoundType::Exact, mip };
		range.sliceLower = { BoundType::Exact, slice };
		range.sliceUpper = { BoundType::Exact, slice };
		return range;
	}
}

void UploadManager::Initialize() {
	m_numFramesInFlight = rg::runtime::GetOpenRenderGraphSettings().numFramesInFlight;

	UploadInstance::Config config;
	config.numFramesInFlight = m_numFramesInFlight;
	config.pageSizeBytes = UploadInstance::kDefaultPageSize;
	config.preallocateCapacityBytes = UploadInstance::kDefaultPreallocateCapacity;
	config.debugName = "UploadManager";
	config.pageNamePrefix = "UploadManagerPage";
	config.usageHint = "Upload buffer";
	m_uploadInstance = std::make_unique<UploadInstance>(std::move(config));
	m_uploadInstance->SetPendingWorkChangedCallback([this] {
		MarkUploadPassDirty();
	});
	m_uploadInstance->SetTargetTelemetryCallback([this](const UploadTarget& target, uint64_t& outId, std::string& outName) {
		CaptureUploadTargetTelemetry(target, outId, outName);
	});
	m_uploadInstance->SetInvalidRegistryHandleCallback(
		[this](const UploadTarget& target, const char* reason, const char* file, int line) {
			return IsUploadTargetValid(target, reason, file, line);
		});
	m_uploadInstance->SetResolveContext(m_ctx);
	MarkUploadPassDirty();
}

void UploadManager::SetUploadResolveContext(UploadResolveContext ctx) {
	{
		std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
		m_ctx = ctx;
		RefreshQueuedCopyTelemetryLocked();
	}
	if (m_uploadInstance) {
		m_uploadInstance->SetResolveContext(ctx);
	}
	MarkUploadPassDirty();
}

void UploadManager::MarkUploadPassDirty()
{
	if (m_uploadPass) {
		m_uploadPass->MarkDeclaredResourcesDirty();
	}
}

void UploadManager::CaptureResourceCopyTelemetry(ResourceCopy& copy)
{
	copy.sourceGlobalResourceId = copy.source ? copy.source->GetGlobalResourceID() : 0;
	copy.destinationGlobalResourceId = copy.destination ? copy.destination->GetGlobalResourceID() : 0;
	copy.sourceDebugName = copy.source ? copy.source->GetName() : std::string{};
	copy.destinationDebugName = copy.destination ? copy.destination->GetName() : std::string{};
}

void UploadManager::RefreshQueuedCopyTelemetryLocked()
{
	for (auto& copy : queuedResourceCopies) {
		CaptureResourceCopyTelemetry(copy);
	}
}

void UploadManager::CaptureUploadTargetTelemetry(const UploadTarget& target, uint64_t& outId, std::string& outName)
{
	outId = 0;
	outName.clear();
	switch (target.kind) {
	case UploadTarget::Kind::PinnedShared:
		if (target.pinned) {
			outId = target.pinned->GetGlobalResourceID();
			outName = target.pinned->GetName();
		}
		break;
	case UploadTarget::Kind::RegistryHandle:
		outId = target.h.GetGlobalResourceID();
		if (m_ctx.registry) {
			if (auto* resource = m_ctx.registry->Resolve(target.h)) {
				outName = resource->GetName();
			}
		}
		break;
	}
}

bool UploadManager::IsUploadTargetValid(const UploadTarget& target, const char* reason, const char* file, int line)
{
	if (target.kind != UploadTarget::Kind::RegistryHandle || !m_ctx.registry) {
		return true;
	}
	if (m_ctx.registry->IsValid(target.h)) {
		return true;
	}

	spdlog::error(
		"UploadManager: invalid queued registry-handle upload detected during {}: handle idx={} generation={} epoch={} queued at {}:{}",
		reason ? reason : "upload-processing",
		target.h.GetKey().idx,
		target.h.GetGeneration(),
		target.h.GetEpoch(),
		file ? file : "<unknown>",
		line);
#if BUILD_TYPE == BUILD_TYPE_DEBUG
	throw std::runtime_error("UploadManager: invalid registry handle uploads detected. This likely indicates a bug where uploads are being queued referencing registry handles that have since been released.");
#endif
	return false;
}

#if BUILD_TYPE == BUILD_TYPE_DEBUG
void UploadManager::UploadData(const void* data, size_t size, UploadTarget resourceToUpdate, size_t dataBufferOffset, const char* file, int line)
#else
void UploadManager::UploadData(const void* data, size_t size, UploadTarget resourceToUpdate, size_t dataBufferOffset)
#endif
{
	if (!m_uploadInstance) {
		Initialize();
	}
#if BUILD_TYPE == BUILD_TYPE_DEBUG
	m_uploadInstance->UploadData(data, size, std::move(resourceToUpdate), dataBufferOffset, file, line);
#else
	m_uploadInstance->UploadData(data, size, std::move(resourceToUpdate), dataBufferOffset);
#endif
}

#if BUILD_TYPE == BUILD_TYPE_DEBUG
void UploadManager::UploadTextureSubresources(
	UploadTarget target,
	rhi::Format fmt,
	uint32_t baseWidth,
	uint32_t baseHeight,
	uint32_t depthOrLayers,
	uint32_t mipLevels,
	uint32_t arraySize,
	const rhi::helpers::SubresourceData* srcSubresources,
	uint32_t srcCount,
	const char* file,
	int line)
#else
void UploadManager::UploadTextureSubresources(
	UploadTarget target,
	rhi::Format fmt,
	uint32_t baseWidth,
	uint32_t baseHeight,
	uint32_t depthOrLayers,
	uint32_t mipLevels,
	uint32_t arraySize,
	const rhi::helpers::SubresourceData* srcSubresources,
	uint32_t srcCount)
#endif
{
	if (!m_uploadInstance) {
		Initialize();
	}
#if BUILD_TYPE == BUILD_TYPE_DEBUG
	m_uploadInstance->UploadTextureSubresources(
		std::move(target),
		fmt,
		baseWidth,
		baseHeight,
		depthOrLayers,
		mipLevels,
		arraySize,
		srcSubresources,
		srcCount,
		file,
		line);
#else
	m_uploadInstance->UploadTextureSubresources(
		std::move(target),
		fmt,
		baseWidth,
		baseHeight,
		depthOrLayers,
		mipLevels,
		arraySize,
		srcSubresources,
		srcCount);
#endif
}

void UploadManager::ProcessDeferredReleases(uint8_t frameIndex)
{
	if (m_uploadInstance) {
		m_uploadInstance->ProcessDeferredReleases(frameIndex);
	}
}

void UploadManager::DeclareUploadPassResourceUsages(RenderPassBuilder* builder)
{
	if (!builder) {
		return;
	}

	{
		std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
		RefreshQueuedCopyTelemetryLocked();
		for (const auto& copy : queuedResourceCopies) {
			if (copy.source) {
				builder->WithCopySource(copy.source);
			}
			if (copy.destination) {
				builder->WithCopyDest(copy.destination);
			}
		}
	}

	if (!m_uploadInstance) {
		return;
	}

	m_uploadInstance->DeclarePendingUploadResourceUsages(
		[builder](const std::shared_ptr<Resource>& source) {
			if (source) {
				builder->WithCopySource(source);
			}
		},
		[builder](const UploadTarget& target, uint32_t mip, uint32_t slice) {
			const bool isBuffer = mip == UINT32_MAX && slice == UINT32_MAX;
			switch (target.kind) {
			case UploadTarget::Kind::PinnedShared:
				if (!target.pinned) {
					return;
				}
				if (isBuffer) {
					builder->WithCopyDest(target.pinned);
				} else {
					builder->WithCopyDest(ResourcePtrAndRange{ target.pinned, SingleSubresourceRange(mip, slice) });
				}
				break;
			case UploadTarget::Kind::RegistryHandle:
				if (isBuffer) {
					builder->WithCopyDest(ResourceHandleAndRange{ target.h });
				} else {
					builder->WithCopyDest(ResourceHandleAndRange{ target.h, SingleSubresourceRange(mip, slice) });
				}
				break;
			}
		});
}

std::string UploadManager::DescribeQueuedTargetByGlobalResourceId(uint64_t globalResourceId)
{
	if (globalResourceId == 0) {
		return {};
	}

	std::ostringstream result;
	size_t matchCount = 0;
	if (m_uploadInstance) {
		const auto uploadDescription = m_uploadInstance->DescribeQueuedTargetByGlobalResourceId(globalResourceId);
		if (!uploadDescription.empty()) {
			result << uploadDescription;
			++matchCount;
		}
	}

	std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
	RefreshQueuedCopyTelemetryLocked();

	for (const auto& copy : queuedResourceCopies) {
		if (copy.destinationGlobalResourceId == globalResourceId) {
			if (matchCount++ > 0) {
				result << " | ";
			}
			result
				<< "resource-copy-dest"
				<< " name='" << (copy.destinationDebugName.empty() ? std::string("<unknown>") : copy.destinationDebugName) << "'"
				<< " sourceName='" << (copy.sourceDebugName.empty() ? std::string("<unknown>") : copy.sourceDebugName) << "'"
				<< " bytes=" << copy.size;
		}

		if (copy.sourceGlobalResourceId == globalResourceId) {
			if (matchCount++ > 0) {
				result << " | ";
			}
			result
				<< "resource-copy-source"
				<< " name='" << (copy.sourceDebugName.empty() ? std::string("<unknown>") : copy.sourceDebugName) << "'"
				<< " destinationName='" << (copy.destinationDebugName.empty() ? std::string("<unknown>") : copy.destinationDebugName) << "'"
				<< " bytes=" << copy.size;
		}
	}

	return result.str();
}

void UploadManager::ProcessUploads(uint8_t frameIndex, rg::imm::ImmediateCommandList& commandList) {
	if (m_uploadInstance) {
		m_uploadInstance->ProcessUploads(frameIndex, commandList);
	}
}

void UploadManager::QueueResourceCopy(const std::shared_ptr<Resource>& destination, const std::shared_ptr<Resource>& source, size_t size) {
	std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
	ResourceCopy copy;
	copy.source = source;
	copy.destination = destination;
	copy.size = size;
	CaptureResourceCopyTelemetry(copy);
	queuedResourceCopies.push_back(std::move(copy));
	MarkUploadPassDirty();
}

void UploadManager::ExecuteResourceCopies(uint8_t frameIndex, rg::imm::ImmediateCommandList& commandList) {
	(void)frameIndex;
	std::vector<ResourceCopy> resourceCopies;
	{
		std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
		resourceCopies.swap(queuedResourceCopies);
		if (!resourceCopies.empty()) {
			MarkUploadPassDirty();
		}
	}

	std::unordered_set<Resource*> seenDestinations;
	for (auto& copy : resourceCopies) {
		auto* dstPtr = copy.destination.get();
		if (!seenDestinations.insert(dstPtr).second) {
			continue;
		}
		commandList.CopyBufferRegion(
			copy.destination,
			0,
			copy.source,
			0,
			copy.size);
	}
}

void UploadManager::Cleanup() {
	if (m_uploadInstance) {
		m_uploadInstance->Cleanup();
		m_uploadInstance.reset();
	}

	{
		std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
		queuedResourceCopies.clear();
	}

	m_streamingPagePool.Cleanup();
	{
		std::lock_guard<std::mutex> lock(m_streamingMutex);
		m_pendingStreamingUploads.clear();
	}
	MarkUploadPassDirty();
}

void UploadManager::QueueStreamingUpload(
    const void* data, size_t size,
    std::shared_ptr<Resource> destination, size_t dstOffset)
{
	if (!data || size == 0 || !destination) return;

	auto uploadBuffer = Buffer::CreateShared(rhi::HeapType::Upload, size, /*uav=*/false);
	uploadBuffer->SetName("StreamingUploadTemp");

	uint8_t* mapped = nullptr;
	uploadBuffer->GetAPIResource().Map(reinterpret_cast<void**>(&mapped), 0, size);
	if (mapped) {
		std::memcpy(mapped, data, size);
		uploadBuffer->GetAPIResource().Unmap(0, size);
	}

	StreamingUploadDescriptor desc;
	desc.srcUploadBuffer = std::move(uploadBuffer);
	desc.srcOffset       = 0;
	desc.dstResource     = std::move(destination);
	desc.dstOffset       = dstOffset;
	desc.size            = size;

	{
		std::lock_guard<std::mutex> lock(m_streamingMutex);
		m_pendingStreamingUploads.push_back(std::move(desc));
	}
}

std::vector<StreamingUploadDescriptor> UploadManager::ConsumeStreamingUploads()
{
	std::lock_guard<std::mutex> lock(m_streamingMutex);
	std::vector<StreamingUploadDescriptor> result;
	result.swap(m_pendingStreamingUploads);
	return result;
}
