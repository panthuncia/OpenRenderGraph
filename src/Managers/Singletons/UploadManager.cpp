#include "Managers/Singletons/UploadManager.h"

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
	{
		std::lock_guard lock(m_streamingMutex);
		if (!m_streamingInitialized) {
			m_streamingTimeline = std::make_shared<rhi::TimelinePtr>();
			const auto result = DeviceManager::GetInstance().GetDevice().CreateTimeline(
				*m_streamingTimeline, 0, "TrackedStreamingUploadTimeline");
			if (rhi::Failed(result) || !*m_streamingTimeline) {
				m_streamingTimeline.reset();
				throw std::runtime_error("UploadManager failed to create the tracked streaming upload timeline");
			}
			m_nextStreamingTimelineValue = 0;
			m_streamingInitialized = true;
		}
	}
	if (!m_streamingCompletionWorker.joinable()) {
		m_streamingCompletionWorker = std::jthread(
			[this](std::stop_token stopToken) { RunStreamingCompletionWorker(stopToken); });
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
	if (m_streamingCompletionWorker.joinable()) {
		m_streamingCompletionWorker.request_stop();
		m_streamingCv.notify_all();
		m_streamingCompletionWorker.join();
	}

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
		for (auto& descriptor : m_pendingStreamingUploads) {
			if (descriptor.ticket) descriptor.ticket->Cancel();
		}
		m_pendingStreamingUploads.clear();
		m_streamingTimeline.reset();
		m_streamingInitialized = false;
		m_nextStreamingTimelineValue = 0;
	}
	MarkUploadPassDirty();
}

void UploadManager::QueueStreamingUpload(
    const void* data, size_t size,
    std::shared_ptr<Resource> destination, size_t dstOffset)
{
	(void)SubmitStreamingUpload(data, size, std::move(destination), dstOffset, false);
}

std::shared_ptr<TrackedUploadTicket> UploadManager::QueueTrackedStreamingUpload(
    const void* data, size_t size, std::shared_ptr<Resource> destination, size_t dstOffset)
{
	return SubmitStreamingUpload(data, size, std::move(destination), dstOffset, true);
}

std::shared_ptr<TrackedUploadTicket> UploadManager::SubmitStreamingUpload(
	const void* data, size_t size, std::shared_ptr<Resource> destination,
	size_t dstOffset, bool exposeTicket)
{
	if (!data || size == 0 || !destination) return {};
	if (!m_streamingInitialized) Initialize();

	auto ticket = std::make_shared<TrackedUploadTicket>();
	auto uploadBuffer = Buffer::CreateShared(rhi::HeapType::Upload, size, false);
	uploadBuffer->SetName(exposeTicket ? "TrackedStreamingUploadTemp" : "StreamingUploadTemp");
	uint8_t* mapped = nullptr;
	uploadBuffer->GetAPIResource().Map(reinterpret_cast<void**>(&mapped), 0, size);
	if (!mapped) {
		ticket->Cancel();
		return {};
	}
	std::memcpy(mapped, data, size);
	uploadBuffer->GetAPIResource().Unmap(0, size);

	{
		std::lock_guard lock(m_streamingMutex);
		StreamingUploadDescriptor descriptor;
		descriptor.srcUploadBuffer = std::move(uploadBuffer);
		descriptor.dstResource = std::move(destination);
		descriptor.dstOffset = dstOffset;
		descriptor.size = size;
		descriptor.ticket = ticket;
		m_pendingStreamingUploads.push_back(std::move(descriptor));
	}
	m_streamingCv.notify_one();
	return exposeTicket ? ticket : std::shared_ptr<TrackedUploadTicket>{};
}

void UploadManager::RunStreamingCompletionWorker(std::stop_token stopToken)
{
	try {
		for (;;) {
			SubmittedStreamingBatch batch;
			std::shared_ptr<rhi::TimelinePtr> timeline;
			{
				std::unique_lock lock(m_streamingMutex);
				m_streamingCv.wait(lock, stopToken, [this] { return !m_pendingStreamingUploads.empty(); });
				if (stopToken.stop_requested()) {
					for (auto& descriptor : m_pendingStreamingUploads) {
						if (descriptor.ticket) descriptor.ticket->Cancel();
					}
					m_pendingStreamingUploads.clear();
					return;
				}
				if (m_pendingStreamingUploads.empty()) {
					continue;
				}
				constexpr size_t maxDescriptorsPerBatch = 256;
				constexpr size_t maxBytesPerBatch = 16u * 1024u * 1024u;
				size_t batchBytes = 0;
				while (!m_pendingStreamingUploads.empty() &&
					batch.descriptors.size() < maxDescriptorsPerBatch) {
					auto& next = m_pendingStreamingUploads.front();
					if (!batch.descriptors.empty() && batchBytes + next.size > maxBytesPerBatch) break;
					batchBytes += next.size;
					batch.descriptors.push_back(std::move(next));
					m_pendingStreamingUploads.pop_front();
				}
				timeline = m_streamingTimeline;
			}

			for (auto& descriptor : batch.descriptors) {
				if (!descriptor.ticket) continue;
				auto expected = TrackedUploadTicketState::Queued;
				if (descriptor.ticket->state.compare_exchange_strong(expected,
						TrackedUploadTicketState::Claimed, std::memory_order_acq_rel,
						std::memory_order_acquire)) descriptor.ticket->NotifyChanged();
			}
			std::erase_if(batch.descriptors, [](const auto& descriptor) {
				return !descriptor.ticket || descriptor.ticket->state.load(std::memory_order_acquire) ==
					TrackedUploadTicketState::Cancelled;
			});
			if (batch.descriptors.empty()) continue;

			auto& deviceManager = DeviceManager::GetInstance();
			auto device = deviceManager.GetDevice();
			if (rhi::Failed(device.CreateCommandAllocator(rhi::QueueKind::Copy, batch.allocator)) ||
				rhi::Failed(device.CreateCommandList(rhi::QueueKind::Copy, batch.allocator.Get(), batch.commandList))) {
				for (auto& descriptor : batch.descriptors) descriptor.ticket->Cancel();
				continue;
			}
			std::vector<rhi::BufferBarrier> barriers(batch.descriptors.size());
			for (size_t index = 0; index < batch.descriptors.size(); ++index) {
				auto& barrier = barriers[index];
				barrier.buffer = batch.descriptors[index].dstResource->GetAPIResource().GetHandle();
				barrier.beforeSync = rhi::ResourceSyncState::Copy;
				barrier.afterSync = rhi::ResourceSyncState::Copy;
				barrier.beforeAccess = rhi::ResourceAccessType::Common;
				barrier.afterAccess = rhi::ResourceAccessType::CopyDest;
			}
			batch.commandList->Barriers({ .buffers = {
				barriers.data(), static_cast<uint32_t>(barriers.size()) } });
			for (const auto& descriptor : batch.descriptors) {
				batch.commandList->CopyBufferRegion(
					descriptor.dstResource->GetAPIResource().GetHandle(), descriptor.dstOffset,
					descriptor.srcUploadBuffer->GetAPIResource().GetHandle(), descriptor.srcOffset,
					descriptor.size);
			}
			for (auto& barrier : barriers) {
				barrier.beforeAccess = rhi::ResourceAccessType::CopyDest;
				barrier.afterAccess = rhi::ResourceAccessType::Common;
			}
			batch.commandList->Barriers({ .buffers = {
				barriers.data(), static_cast<uint32_t>(barriers.size()) } });
			batch.commandList->End();

			batch.timelineValue = ++m_nextStreamingTimelineValue;
			for (auto& descriptor : batch.descriptors) {
				std::lock_guard ticketLock(descriptor.ticket->timelineMutex);
				descriptor.ticket->timelineOwner = timeline;
				descriptor.ticket->timelineValue = batch.timelineValue;
				descriptor.ticket->isTimelineComplete = [timeline](uint64_t value) {
					return timeline && *timeline && (*timeline)->GetCompletedValue() >= value;
				};
			}
			const rhi::CommandList lists[] = { batch.commandList.Get() };
			const rhi::TimelinePoint signal{ (*timeline)->GetHandle(), batch.timelineValue };
			if (rhi::Failed(deviceManager.GetCopyQueue().Submit(lists, { .signals = { &signal, 1 } }))) {
				for (auto& descriptor : batch.descriptors) descriptor.ticket->Cancel();
				continue;
			}
			for (auto& descriptor : batch.descriptors) {
				auto expected = TrackedUploadTicketState::Claimed;
				if (descriptor.ticket->state.compare_exchange_strong(expected,
						TrackedUploadTicketState::Submitted, std::memory_order_release,
						std::memory_order_acquire)) descriptor.ticket->NotifyChanged();
			}
			if (timeline && *timeline) (void)(*timeline)->HostWait(batch.timelineValue);
			for (auto& descriptor : batch.descriptors) (void)descriptor.ticket->Complete();
		}
	} catch (const std::exception& error) {
		spdlog::error("Tracked streaming upload completion worker stopped after exception: {}", error.what());
	} catch (...) {
		spdlog::error("Tracked streaming upload completion worker stopped after unknown exception");
	}
}


} // namespace org
