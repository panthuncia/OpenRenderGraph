#include "Managers/Singletons/UploadManager.h"

#include <BasicTelemetry/Telemetry.h>

#include <cstring>
#include <algorithm>
#include <sstream>
#include <unordered_set>

#include <spdlog/spdlog.h>

#include "Render/PassBuilders.h"
#include "Render/Runtime/OpenRenderGraphSettings.h"
#include "Managers/Singletons/DeviceManager.h"


namespace org {

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
	m_numFramesInFlight = org::runtime::GetOpenRenderGraphSettings().numFramesInFlight;

	UploadInstance::Config config;
	config.numFramesInFlight = m_numFramesInFlight;
	config.pageSizeBytes = UploadInstance::kDefaultPageSize;
	config.preallocateCapacityBytes = UploadInstance::kDefaultPreallocateCapacity;
	config.debugName = "UploadManager";
	config.pageNamePrefix = "UploadManagerPage";
	config.usageHint = "Upload buffer";
	m_uploadInstance = std::make_unique<UploadInstance>(std::move(config));
	m_uploadInstance->SetStagedUploadsRecordedDirectly(m_stagedUploadsDirect);
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
	if (!m_copyQueueUploads) {
		m_copyQueueUploads = std::make_unique<org::runtime::CopyQueueUploadService>();
	}
	if (!m_copyQueueUploads->Initialized()) {
		auto& deviceManager = DeviceManager::GetInstance();
		m_copyQueueUploads->Initialize(deviceManager.GetDevice(), deviceManager.GetCopyQueue());
		if (!m_copyQueueUploads->Initialized()) {
			throw std::runtime_error("UploadManager failed to initialize the copy-queue upload service");
		}
	}
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
void UploadManager::UploadDataBatch(UploadTarget resourceToUpdate, std::span<const org::runtime::UploadRegion> regions, const char* file, int line)
#else
void UploadManager::UploadDataBatch(UploadTarget resourceToUpdate, std::span<const org::runtime::UploadRegion> regions)
#endif
{
	if (!m_uploadInstance) {
		Initialize();
	}
#if BUILD_TYPE == BUILD_TYPE_DEBUG
	m_uploadInstance->UploadDataBatch(std::move(resourceToUpdate), regions, file, line);
#else
	m_uploadInstance->UploadDataBatch(std::move(resourceToUpdate), regions);
#endif
}

#if BUILD_TYPE == BUILD_TYPE_DEBUG
void UploadManager::PostTextureSubresources(UploadTarget target, rhi::Format fmt, uint32_t baseWidth, uint32_t baseHeight,
	uint32_t depthOrLayers, uint32_t mipLevels, uint32_t arraySize,
	std::shared_ptr<const std::vector<rhi::helpers::SubresourceData>> subresources,
	std::shared_ptr<const void> keepAlive, const char* file, int line)
#else
void UploadManager::PostTextureSubresources(UploadTarget target, rhi::Format fmt, uint32_t baseWidth, uint32_t baseHeight,
	uint32_t depthOrLayers, uint32_t mipLevels, uint32_t arraySize,
	std::shared_ptr<const std::vector<rhi::helpers::SubresourceData>> subresources,
	std::shared_ptr<const void> keepAlive)
#endif
{
	if (!m_uploadInstance) {
		Initialize();
	}
#if BUILD_TYPE == BUILD_TYPE_DEBUG
	m_uploadInstance->PostTextureSubresources(std::move(target), fmt, baseWidth, baseHeight, depthOrLayers,
		mipLevels, arraySize, std::move(subresources), std::move(keepAlive), file, line);
#else
	m_uploadInstance->PostTextureSubresources(std::move(target), fmt, baseWidth, baseHeight, depthOrLayers,
		mipLevels, arraySize, std::move(subresources), std::move(keepAlive));
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
	if (target.kind == UploadTarget::Kind::PinnedShared && target.pinned &&
		!target.pinned->IsRenderGraphManaged()) {
		throw std::runtime_error(
			"UploadManager::UploadTextureSubresources rejected an externally managed immutable shader resource ('" +
			target.pinned->GetName() + "'). Use its external transfer service.");
	}
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

// The instance is created on first use, like UploadData's: a frame whose uploads are all staged makes no
// other call that would create it. The direct mode is the manager's, so an instance created later has it too.
void UploadManager::SubmitStagedUploads(std::shared_ptr<org::runtime::StagedUploadBatch> batch)
{
	if (!m_uploadInstance) {
		Initialize();
	}
	m_uploadInstance->SubmitStagedUploads(std::move(batch));
}

void UploadManager::SetStagedUploadsRecordedDirectly(bool direct)
{
	m_stagedUploadsDirect = direct;
	if (m_uploadInstance) {
		m_uploadInstance->SetStagedUploadsRecordedDirectly(direct);
	}
}

size_t UploadManager::RecordStagedUploads(rhi::CommandList& list, uint8_t frameIndex)
{
	return m_uploadInstance ? m_uploadInstance->RecordStagedUploads(list, frameIndex) : 0;
}

void UploadManager::ProcessDeferredReleases(uint8_t frameIndex)
{
	if (m_uploadInstance) {
		m_uploadInstance->ProcessDeferredReleases(frameIndex);
	}
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

void UploadManager::SetOwnerThread() {
	if (!m_uploadInstance) {
		Initialize();
	}
	m_uploadInstance->SetOwnerThread();
}

void UploadManager::ProcessUploads(uint8_t frameIndex, org::imm::ImmediateCommandList& commandList) {
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

void UploadManager::ExecuteResourceCopies(uint8_t frameIndex, org::imm::ImmediateCommandList& commandList) {
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
	if (m_copyQueueUploads) {
		m_copyQueueUploads->Cleanup();
		m_copyQueueUploads.reset();
	}

	if (m_uploadInstance) {
		m_uploadInstance->Cleanup();
		m_uploadInstance.reset();
	}

	{
		std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
		queuedResourceCopies.clear();
	}

	MarkUploadPassDirty();
}

std::shared_ptr<TrackedUploadTicket> UploadManager::QueueTrackedStreamingUploadSegments(
	std::span<const StreamingUploadSegment> segments, size_t totalSize,
	WorkerOwnedDestination destination, size_t dstOffset)
{
	if (!m_copyQueueUploads || !m_copyQueueUploads->Initialized()) Initialize();
	return m_copyQueueUploads->QueueBufferUpload(segments, totalSize, std::move(destination), dstOffset);
}


} // namespace org
