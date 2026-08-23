#include "Managers/UploadInstance.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include <rhi_helpers.h>
#include <spdlog/spdlog.h>
#include <BasicTelemetry/Tracy.h>

#include "Resources/Buffers/Buffer.h"
#include "Resources/Resource.h"
#include "Render/MemoryIntrospectionAPI.h"
#include "Render/ImmediateExecution/ImmediateCommandList.h"
#include "Render/Runtime/TaskServiceAccess.h"


namespace org {

namespace {
	size_t AlignUpSizeT(const size_t v, const size_t a) noexcept {
		return (v + (a - 1)) & ~(a - 1);
	}

	bool UploadTelemetryLoggingEnabled() {
		static const bool enabled = [] {
			for (const char* name : {
				"SARP_UPLOAD_TELEMETRY_LOG",
				"SARP_TEXTURE_STREAMING_TRANSITION_LOG" }) {
				char* value = nullptr;
				size_t valueLength = 0;
				const bool isSet =
					_dupenv_s(&value, &valueLength, name) == 0 &&
					value != nullptr && value[0] != '\0' && value[0] != '0';
				std::free(value);
				if (isSet) return true;
			}
			return false;
		}();
		return enabled;
	}
}

UploadInstance::UploadInstance(uint8_t numFramesInFlight, size_t pageSize)
	: UploadInstance(Config{
		.numFramesInFlight = numFramesInFlight,
		.pageSizeBytes = pageSize,
	})
{
}

UploadInstance::UploadInstance(Config config)
	: m_pageSize((std::max)(size_t{ 1 }, config.pageSizeBytes))
	, m_preallocateCapacityBytes(config.preallocateCapacityBytes)
	, m_debugName(std::move(config.debugName))
	, m_pageNamePrefix(std::move(config.pageNamePrefix))
	, m_usageHint(std::move(config.usageHint))
	, m_numFramesInFlight((std::max)(uint8_t{ 1 }, config.numFramesInFlight))
{
	m_framePages.resize(m_numFramesInFlight);
	m_recentFrameBytes.assign(m_numFramesInFlight, 0);
	StartWorker();
}

UploadInstance::~UploadInstance() {
	Cleanup();
}

void UploadInstance::SetResolveContext(UploadResolveContext ctx) {
	std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
	m_ctx = ctx;
	RefreshQueuedTargetTelemetryLocked();
	PruneInvalidRegistryHandleUpdatesLocked("resolve-context-update");
	MarkPendingWorkChangedLocked();
}

void UploadInstance::SetPendingWorkChangedCallback(PendingWorkChangedCallback callback) {
	std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
	m_pendingWorkChanged = std::move(callback);
}

void UploadInstance::SetTargetTelemetryCallback(TargetTelemetryCallback callback) {
	std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
	m_targetTelemetry = std::move(callback);
	RefreshQueuedTargetTelemetryLocked();
}

void UploadInstance::SetInvalidRegistryHandleCallback(InvalidRegistryHandleCallback callback) {
	std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
	m_invalidRegistryHandle = std::move(callback);
}

void UploadInstance::StartWorker() {
	std::lock_guard<std::mutex> lock(m_workerMutex);
	if (m_taskScope) {
		return;
	}
	m_taskService = org::runtime::GetDefaultTaskService();
	if (!m_taskService) return;
	m_workerQuit = false;
	m_taskScope = m_taskService->CreateScope(m_debugName + "::PagePreparation");
}

void UploadInstance::StopWorker() {
	std::shared_ptr<org::runtime::ITaskScope> scope;
	{
		std::lock_guard<std::mutex> lock(m_workerMutex);
		m_workerQuit = true;
		scope = std::move(m_taskScope);
	}
	if (scope) scope->CancelAndWait();
	m_workerDrainScheduled.store(false, std::memory_order_release);
	std::lock_guard<std::mutex> lock(m_workerMutex);
	m_taskService.reset();
	m_workerRequestedPages = 0;
}

UploadInstance::UploadPagePtr UploadInstance::CreatePage(size_t size, bool dedicated) {
	auto page = std::make_shared<UploadPage>();
	page->capacity = (std::max)(size, m_pageSize);
	page->dedicated = dedicated;
	page->index = m_nextPageIndex++;
	page->buffer = Buffer::CreateShared(rhi::HeapType::Upload, page->capacity, false);
	return page;
}

void UploadInstance::TagPage(const UploadPagePtr& page) {
	if (page && page->buffer && !page->tagged) {
		page->buffer->SetName(m_pageNamePrefix + "_" + std::to_string(page->index));
		org::memory::SetResourceUsageHint(*page->buffer, m_usageHint);
		page->tagged = true;
	}
}

void UploadInstance::WorkerMain() {
	constexpr size_t kPagesPerDrain = 1;
	size_t pagesToCreate = 0;
	{
		std::lock_guard<std::mutex> lock(m_workerMutex);
		if (!m_workerQuit) {
			pagesToCreate = (std::min)(m_workerRequestedPages, kPagesPerDrain);
			m_workerRequestedPages -= pagesToCreate;
		}
	}

	for (size_t i = 0; i < pagesToCreate; ++i) {
			auto page = CreatePage(m_pageSize, false);
			std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
			const size_t maxWarmPages = m_preallocateCapacityBytes / m_pageSize;
			if (maxWarmPages == 0 || AvailableReusableNormalPagesLocked() >= maxWarmPages) {
				continue;
			}
			m_readyPages.push_back(std::move(page));
	}

	bool hasMore = false;
	{
		std::lock_guard<std::mutex> lock(m_workerMutex);
		m_workerDrainScheduled.store(false, std::memory_order_release);
		hasMore = !m_workerQuit && m_workerRequestedPages != 0;
	}
	if (hasMore) ScheduleWorkerDrain();
}

void UploadInstance::ScheduleWorkerDrain() {
	StartWorker();
	std::shared_ptr<org::runtime::ITaskService> service;
	std::shared_ptr<org::runtime::ITaskScope> scope;
	{
		std::lock_guard<std::mutex> lock(m_workerMutex);
		if (m_workerQuit || !m_taskService || !m_taskScope) return;
		service = m_taskService;
		scope = m_taskScope;
	}
	bool expected = false;
	if (!m_workerDrainScheduled.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return;
	if (!service->Submit(scope, org::runtime::TaskPriority::Background,
			m_debugName + "::PrepareUploadPages", [this] { WorkerMain(); })) {
		m_workerDrainScheduled.store(false, std::memory_order_release);
		spdlog::error("UploadInstance '{}' page preparation submission was rejected", m_debugName);
	}
}

size_t UploadInstance::NormalPageCountForBytes(size_t bytes) const noexcept {
	if (bytes == 0) {
		return 0;
	}
	return (bytes + m_pageSize - 1) / m_pageSize;
}

size_t UploadInstance::WarmTargetPageCountLocked() const noexcept {
	size_t highWaterBytes = 0;
	for (const size_t bytes : m_recentFrameBytes) {
		highWaterBytes = (std::max)(highWaterBytes, bytes);
	}
	const size_t capPages = m_preallocateCapacityBytes / m_pageSize;
	return (std::min)(NormalPageCountForBytes(highWaterBytes), capPages);
}

size_t UploadInstance::AvailableReusableNormalPagesLocked() const noexcept {
	size_t count = 0;
	for (const auto& page : m_freePages) {
		if (page && !page->dedicated && page->capacity == m_pageSize) {
			++count;
		}
	}
	for (const auto& page : m_readyPages) {
		if (page && !page->dedicated && page->capacity == m_pageSize) {
			++count;
		}
	}
	return count;
}

void UploadInstance::RequestWorkerPagesLocked() {
	const size_t targetPages = WarmTargetPageCountLocked();
	if (targetPages == 0) {
		return;
	}

	const size_t availablePages = AvailableReusableNormalPagesLocked();
	if (availablePages >= targetPages) {
		return;
	}

	{
		std::lock_guard<std::mutex> workerLock(m_workerMutex);
		m_workerRequestedPages += targetPages - availablePages;
	}
	ScheduleWorkerDrain();
}

void UploadInstance::TrackPageForCurrentFrameLocked(const UploadPagePtr& page) {
	if (!page) {
		return;
	}
	if (m_openPageSet.insert(page.get()).second) {
		m_openPages.push_back(page);
	}
}

UploadInstance::UploadPagePtr UploadInstance::AcquirePageLocked(size_t minSize, bool dedicated) {
	UploadPagePtr page;
	if (!dedicated && minSize <= m_pageSize) {
		if (!m_freePages.empty()) {
			page = std::move(m_freePages.front());
			m_freePages.pop_front();
		} else if (!m_readyPages.empty()) {
			page = std::move(m_readyPages.front());
			m_readyPages.pop_front();
		}
	}

	if (!page) {
		page = CreatePage((std::max)(minSize, m_pageSize), dedicated || minSize > m_pageSize);
		spdlog::info(
			"{} inline page create: page={}, capacity={} MiB, dedicated={}",
			m_debugName,
			page->index,
			page->capacity / (1024ull * 1024ull),
			page->dedicated ? 1 : 0);
	}

	TagPage(page);
	page->tailOffset = 0;
	TrackPageForCurrentFrameLocked(page);
	return page;
}

bool UploadInstance::AllocateUploadRegion(
	size_t size,
	size_t alignment,
	std::shared_ptr<Resource>& outUploadBuffer,
	size_t& outOffset)
{
	if (alignment == 0) {
		alignment = 1;
	}

	const bool dedicated = size > m_pageSize;
	if (!m_activePage || m_activePage->dedicated || dedicated) {
		m_activePage = AcquirePageLocked(size, dedicated);
	}

	size_t alignedTail = AlignUpSizeT(m_activePage->tailOffset, alignment);
	if (alignedTail + size > m_activePage->capacity) {
		m_activePage = AcquirePageLocked(size, dedicated);
		alignedTail = AlignUpSizeT(m_activePage->tailOffset, alignment);
	}

	if (alignedTail + size > m_activePage->capacity) {
		m_activePage = AcquirePageLocked(size, true);
		alignedTail = AlignUpSizeT(m_activePage->tailOffset, alignment);
	}

	outOffset = alignedTail;
	m_activePage->tailOffset = alignedTail + size;
	outUploadBuffer = m_activePage->buffer;
	m_currentFrameUploadBytes += size;
	return true;
}

bool UploadInstance::TryCoalesceAppend(ResourceUpdate& last, const ResourceUpdate& next) noexcept {
	if (!last.active || !next.active) return false;
	if (last.resourceToUpdate != next.resourceToUpdate) return false;
	if (last.uploadBuffer.get() != next.uploadBuffer.get()) return false;
	if (last.dataBufferOffset + last.size != next.dataBufferOffset) return false;
	if (last.uploadBufferOffset + last.size != next.uploadBufferOffset) return false;

	last.size += next.size;
	last.lastSequence = next.lastSequence;
#if BUILD_TYPE == BUILD_TYPE_DEBUG
	last.file = next.file;
	last.line = next.line;
	last.stackSize = next.stackSize;
	for (uint8_t i = 0; i < next.stackSize && i < ResourceUpdate::MaxStack; ++i) {
		last.stack[i] = next.stack[i];
	}
#endif
	return true;
}

void UploadInstance::MapUpload(const std::shared_ptr<Resource>& uploadBuffer, uint8_t** outMapped) noexcept {
	if (!outMapped) return;
	*outMapped = nullptr;
	if (!uploadBuffer) return;
	// Upload pages are write-only from the CPU. An empty read range avoids an
	// unnecessary GPU-to-CPU synchronization hint; UnmapUpload reports the
	// exact bytes written below.
	uploadBuffer->GetAPIResource().Map(reinterpret_cast<void**>(outMapped), 0, 0);
}

void UploadInstance::UnmapUpload(
	const std::shared_ptr<Resource>& uploadBuffer,
	size_t writeOffset,
	size_t writeSize) noexcept {
	if (!uploadBuffer) return;
	uploadBuffer->GetAPIResource().Unmap(writeOffset, writeSize);
}

void UploadInstance::MarkPendingWorkChangedLocked() {
	if (m_pendingWorkChanged) {
		m_pendingWorkChanged();
	}
}

void UploadInstance::CaptureTargetTelemetryLocked(const UploadTarget& target, uint64_t& outId, std::string& outName) {
	outId = 0;
	outName.clear();
	if (m_targetTelemetry) {
		m_targetTelemetry(target, outId, outName);
	}
}

void UploadInstance::RefreshQueuedTargetTelemetryLocked() {
	for (auto& update : m_resourceUpdates) {
		CaptureTargetTelemetryLocked(update.resourceToUpdate, update.targetGlobalResourceId, update.targetDebugName);
	}
	for (auto& update : m_textureUpdates) {
		CaptureTargetTelemetryLocked(update.texture, update.targetGlobalResourceId, update.targetDebugName);
	}
}

void UploadInstance::PruneInvalidRegistryHandleUpdatesLocked(const char* reason) {
	if (!m_invalidRegistryHandle) {
		return;
	}

	const auto prune = [&](auto& updates, auto&& targetAccessor) {
		size_t writeIndex = 0;
		bool changed = false;
		for (size_t readIndex = 0; readIndex < updates.size(); ++readIndex) {
			auto& update = updates[readIndex];
#if BUILD_TYPE == BUILD_TYPE_DEBUG
			const char* file = update.file;
			const int line = update.line;
#else
			const char* file = nullptr;
			const int line = 0;
#endif
			if (!m_invalidRegistryHandle(targetAccessor(update), reason, file, line)) {
				changed = true;
				continue;
			}
			if (writeIndex != readIndex) {
				updates[writeIndex] = std::move(update);
			}
			++writeIndex;
		}
		if (changed) {
			updates.resize(writeIndex);
			MarkPendingWorkChangedLocked();
		}
	};

	prune(m_resourceUpdates, [](const ResourceUpdate& update) -> const UploadTarget& {
		return update.resourceToUpdate;
	});
	prune(m_textureUpdates, [](const TextureUpdate& update) -> const UploadTarget& {
		return update.texture;
	});
}

#if BUILD_TYPE == BUILD_TYPE_DEBUG
void UploadInstance::UploadData(const void* data, size_t size, UploadTarget target, size_t dstOffset,
                                const char* file, int line)
#else
void UploadInstance::UploadData(const void* data, size_t size, UploadTarget target, size_t dstOffset)
#endif
{
	if (!data || size == 0) {
		return;
	}

	std::shared_ptr<Resource> uploadBuffer;
	size_t uploadOffset = 0;
	ResourceUpdate update;
	{
		std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
		AllocateUploadRegion(size, /*alignment*/16, uploadBuffer, uploadOffset);

		update.size = size;
		update.resourceToUpdate = target;
		update.uploadBuffer = uploadBuffer;
		update.uploadBufferOffset = uploadOffset;
		update.dataBufferOffset = dstOffset;
#if BUILD_TYPE == BUILD_TYPE_DEBUG
		update.file = file;
		update.line = line;
#ifdef _WIN32
		void* frames[ResourceUpdate::MaxStack];
		USHORT captured = RtlCaptureStackBackTrace(1, ResourceUpdate::MaxStack, frames, nullptr);
		update.stackSize = static_cast<uint8_t>(captured);
		for (USHORT i = 0; i < captured; i++) update.stack[i] = frames[i];
#endif
#endif
		CaptureTargetTelemetryLocked(update.resourceToUpdate, update.targetGlobalResourceId, update.targetDebugName);

		// Keep the reservation visible to ProcessUploads for the entire staging
		// write.  Previously the queue lock was dropped between allocation and
		// enqueue.  A concurrent ProcessUploads could then conclude that this
		// page had no remaining updates, retire it, and allow it to be recycled
		// while the CPU was still writing the reserved region.
		uint8_t* mapped = nullptr;
		MapUpload(uploadBuffer, &mapped);
#if BUILD_TYPE == BUILD_TYPE_DEBUG
		if (!mapped) {
			__debugbreak();
			return;
		}
#endif
		if (mapped) {
			std::memcpy(mapped + uploadOffset, data, size);
		}
		UnmapUpload(uploadBuffer, uploadOffset, size);

		update.firstSequence = ++m_lastUploadSequence;
		update.lastSequence = update.firstSequence;
		for (int i = static_cast<int>(m_resourceUpdates.size()) - 1; i >= 0; --i) {
			auto& last = m_resourceUpdates[static_cast<size_t>(i)];
			if (!last.active) {
				continue;
			}
			// A captured batch boundary is immutable: extending a sealed update
			// would move its earlier bytes past the cutoff without moving the fence
			// that was assigned to them.
			if (last.lastSequence > m_lastSealedUploadSequence && TryCoalesceAppend(last, update)) {
				MarkPendingWorkChangedLocked();
				return;
			}
			break;
		}
		m_resourceUpdates.push_back(std::move(update));
		MarkPendingWorkChangedLocked();
	}
}

#if BUILD_TYPE == BUILD_TYPE_DEBUG
void UploadInstance::UploadTextureSubresources(
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
void UploadInstance::UploadTextureSubresources(
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
	if (!srcSubresources || srcCount == 0) return;

	rhi::Span<const rhi::helpers::SubresourceData> srcSpan{ srcSubresources, srcCount };
	const auto plan = rhi::helpers::PlanTextureUploadSubresources(
		fmt, baseWidth, baseHeight, depthOrLayers, mipLevels, arraySize, srcSpan);
	if (plan.totalSize == 0 || plan.footprints.empty()) return;

	std::shared_ptr<Resource> uploadBuffer;
	size_t uploadBaseOffset = 0;
	{
		std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
		AllocateUploadRegion(static_cast<size_t>(plan.totalSize), /*alignment*/512, uploadBuffer, uploadBaseOffset);

		// Allocation, staging, and enqueue form one ownership transaction.  In
		// particular, ProcessUploads must not be able to retire this page while
		// its newly reserved region is not represented in m_textureUpdates yet.
		uint8_t* mapped = nullptr;
		MapUpload(uploadBuffer, &mapped);
		if (mapped) {
			rhi::helpers::WriteTextureUploadSubresources(plan, srcSpan, mapped, static_cast<uint64_t>(uploadBaseOffset));
		}
		UnmapUpload(uploadBuffer, uploadBaseOffset, static_cast<size_t>(plan.totalSize));

		for (const auto& fp : plan.footprints) {
			rhi::CopyableFootprint copyFootprint;
			copyFootprint.offset = static_cast<uint64_t>(uploadBaseOffset) + fp.offset;
			copyFootprint.rowPitch = fp.rowPitch;
			copyFootprint.width = fp.width;
			copyFootprint.height = fp.height;
			copyFootprint.depth = fp.depth;

			TextureUpdate update;
			update.texture = target;
			update.mip = fp.mip;
			update.slice = fp.arraySlice;
			update.footprint = copyFootprint;
			update.x = 0;
			update.y = 0;
			update.z = fp.zSlice;
			update.uploadBuffer = uploadBuffer;
#if BUILD_TYPE == BUILD_TYPE_DEBUG
			update.file = file;
			update.line = line;
#endif
			CaptureTargetTelemetryLocked(update.texture, update.targetGlobalResourceId, update.targetDebugName);
			update.sequence = ++m_lastUploadSequence;
			m_textureUpdates.push_back(std::move(update));
		}
		MarkPendingWorkChangedLocked();
	}
}

void UploadInstance::ProcessUploads(uint8_t frameIndex, org::imm::ImmediateCommandList& commandList) {
	ProcessUploadsThrough(frameIndex, commandList, UINT64_MAX);
}

void UploadInstance::ProcessUploadsThrough(
	uint8_t frameIndex,
	org::imm::ImmediateCommandList& commandList,
	uint64_t sequenceInclusive) {
	std::vector<ResourceUpdate> resourceUpdates;
	std::vector<TextureUpdate> textureUpdates;
	UploadResolveContext ctx;
	{
		std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
		PruneInvalidRegistryHandleUpdatesLocked("upload-pass-execute");
		resourceUpdates.reserve(m_resourceUpdates.size());
		textureUpdates.reserve(m_textureUpdates.size());
		std::vector<ResourceUpdate> remainingResourceUpdates;
		std::vector<TextureUpdate> remainingTextureUpdates;
		remainingResourceUpdates.reserve(m_resourceUpdates.size());
		remainingTextureUpdates.reserve(m_textureUpdates.size());

		for (auto& update : m_resourceUpdates) {
			if (update.lastSequence <= sequenceInclusive) {
				resourceUpdates.push_back(std::move(update));
			} else {
				remainingResourceUpdates.push_back(std::move(update));
			}
		}
		m_resourceUpdates = std::move(remainingResourceUpdates);

		for (auto& update : m_textureUpdates) {
			if (update.sequence <= sequenceInclusive) {
				textureUpdates.push_back(std::move(update));
			} else {
				remainingTextureUpdates.push_back(std::move(update));
			}
		}
		m_textureUpdates = std::move(remainingTextureUpdates);

		// Upload pages must retire relative to the frame that records their last
		// copy, not the frame in which CPU staging happened. A bounded batch can
		// remain queued for several frames while declarations are refreshed; the
		// old allocation-time retirement would recycle and overwrite its staging
		// memory before CopyBufferRegion was recorded.
		std::unordered_set<Resource*> remainingUploadBuffers;
		remainingUploadBuffers.reserve(m_resourceUpdates.size() + m_textureUpdates.size());
		for (const auto& update : m_resourceUpdates) {
			if (update.uploadBuffer) remainingUploadBuffers.insert(update.uploadBuffer.get());
		}
		for (const auto& update : m_textureUpdates) {
			if (update.uploadBuffer) remainingUploadBuffers.insert(update.uploadBuffer.get());
		}

		std::unordered_set<Resource*> completedUploadBuffers;
		completedUploadBuffers.reserve(resourceUpdates.size() + textureUpdates.size());
		for (const auto& update : resourceUpdates) {
			if (update.uploadBuffer && !remainingUploadBuffers.contains(update.uploadBuffer.get())) {
				completedUploadBuffers.insert(update.uploadBuffer.get());
			}
		}
		for (const auto& update : textureUpdates) {
			if (update.uploadBuffer && !remainingUploadBuffers.contains(update.uploadBuffer.get())) {
				completedUploadBuffers.insert(update.uploadBuffer.get());
			}
		}

		if (!completedUploadBuffers.empty() && m_numFramesInFlight != 0) {
			std::vector<UploadPagePtr> completedPages;
			auto extractCompletedPages = [&](std::vector<UploadPagePtr>& pages) {
				for (size_t i = 0; i < pages.size();) {
					auto& page = pages[i];
					if (page && page->buffer && completedUploadBuffers.contains(page->buffer.get())) {
						completedPages.push_back(std::move(page));
						pages[i] = std::move(pages.back());
						pages.pop_back();
						continue;
					}
					++i;
				}
			};
			extractCompletedPages(m_openPages);
			for (auto& pages : m_framePages) extractCompletedPages(pages);
			for (const auto& page : completedPages) {
				if (page) m_openPageSet.erase(page.get());
			}

			auto& submissionPages = m_framePages[frameIndex % m_numFramesInFlight];
			std::unordered_set<UploadPage*> alreadyScheduled;
			alreadyScheduled.reserve(submissionPages.size() + completedPages.size());
			for (const auto& page : submissionPages) {
				if (page) alreadyScheduled.insert(page.get());
			}
			for (auto& page : completedPages) {
				if (page && alreadyScheduled.insert(page.get()).second) {
					submissionPages.push_back(std::move(page));
				}
			}
		}
		ctx = m_ctx;
		if (!resourceUpdates.empty() || !textureUpdates.empty()) {
			MarkPendingWorkChangedLocked();
		}
	}

	RecordProcessedUploadTelemetry(resourceUpdates, textureUpdates);

	for (auto& update : resourceUpdates) {
		if (!update.active || !update.uploadBuffer || update.size == 0) continue;
		switch (update.resourceToUpdate.kind) {
		case UploadTarget::Kind::PinnedShared:
			commandList.CopyBufferRegion(
				update.resourceToUpdate.pinned,
				update.dataBufferOffset,
				update.uploadBuffer,
				update.uploadBufferOffset,
				update.size);
			break;
		case UploadTarget::Kind::RegistryHandle:
			commandList.CopyBufferRegion(
				ctx.registry->Resolve(update.resourceToUpdate.h),
				update.dataBufferOffset,
				update.uploadBuffer,
				update.uploadBufferOffset,
				update.size);
			break;
		}
	}

	for (auto& texUpdate : textureUpdates) {
		if (texUpdate.texture.kind == UploadTarget::Kind::PinnedShared) {
			commandList.CopyBufferToTexture(
				texUpdate.uploadBuffer,
				texUpdate.texture.pinned,
				texUpdate.mip,
				texUpdate.slice,
				texUpdate.footprint,
				texUpdate.x,
				texUpdate.y,
				texUpdate.z);
		} else {
			commandList.CopyBufferToTexture(
				texUpdate.uploadBuffer,
				ctx.registry->Resolve(texUpdate.texture.h),
				texUpdate.mip,
				texUpdate.slice,
				texUpdate.footprint,
				texUpdate.x,
				texUpdate.y,
				texUpdate.z);
		}
	}
}

void UploadInstance::RecordProcessedUploadTelemetry(
	const std::vector<ResourceUpdate>& resourceUpdates,
	const std::vector<TextureUpdate>& textureUpdates)
{
	if (!UploadTelemetryLoggingEnabled()) return;

	auto targetKey = [](uint64_t resourceID, const std::string& name) {
		return !name.empty()
			? name
			: std::string("resource:") + std::to_string(resourceID);
	};

	for (const auto& update : resourceUpdates) {
		if (!update.active || update.size == 0u) continue;
		auto& target = m_uploadTelemetryTargets[targetKey(update.targetGlobalResourceId, update.targetDebugName)];
		++target.bufferWrites;
		target.bytes += update.size;
		++m_uploadTelemetryBufferWrites;
		m_uploadTelemetryBytes += update.size;
	}

	struct TextureSubresourceKey {
		uint64_t resourceID = 0;
		uint32_t mip = 0;
		uint32_t slice = 0;
		uint32_t z = 0;
		bool operator==(const TextureSubresourceKey&) const = default;
	};
	struct TextureSubresourceHash {
		size_t operator()(const TextureSubresourceKey& key) const noexcept {
			size_t h = std::hash<uint64_t>{}(key.resourceID);
			h ^= static_cast<size_t>(key.mip) * 0x9e3779b1u;
			h ^= static_cast<size_t>(key.slice) * 0x85ebca6bu;
			h ^= static_cast<size_t>(key.z) * 0xc2b2ae35u;
			return h;
		}
	};
	std::unordered_set<TextureSubresourceKey, TextureSubresourceHash> textureSubresources;
	textureSubresources.reserve(textureUpdates.size());
	for (const auto& update : textureUpdates) {
		const uint64_t bytes =
			static_cast<uint64_t>(update.footprint.rowPitch) *
			static_cast<uint64_t>(update.footprint.height) *
			static_cast<uint64_t>((std::max)(update.footprint.depth, 1u));
		auto& target = m_uploadTelemetryTargets[targetKey(update.targetGlobalResourceId, update.targetDebugName)];
		++target.textureWrites;
		target.bytes += bytes;
		++m_uploadTelemetryTextureWrites;
		m_uploadTelemetryBytes += bytes;
		if (!textureSubresources.insert({update.targetGlobalResourceId, update.mip, update.slice, update.z}).second) {
			++m_uploadTelemetryDuplicateTextureSubresources;
		}
	}

	const auto now = std::chrono::steady_clock::now();
	if (m_uploadTelemetryLastLog.time_since_epoch().count() == 0) {
		m_uploadTelemetryLastLog = now;
		return;
	}
	if (now - m_uploadTelemetryLastLog < std::chrono::seconds(1)) return;

	std::vector<std::pair<std::string, UploadTelemetryTarget>> targets(
		m_uploadTelemetryTargets.begin(), m_uploadTelemetryTargets.end());
	std::sort(targets.begin(), targets.end(), [](const auto& lhs, const auto& rhs) {
		return lhs.second.bytes > rhs.second.bytes;
	});
	spdlog::info(
		"UploadTelemetry summary: instance='{}' bufferWrites={} textureWrites={} bytes={} duplicateTextureSubresources={} targets={}",
		m_debugName,
		m_uploadTelemetryBufferWrites,
		m_uploadTelemetryTextureWrites,
		m_uploadTelemetryBytes,
		m_uploadTelemetryDuplicateTextureSubresources,
		targets.size());
	const size_t topCount = (std::min)(targets.size(), size_t{12});
	for (size_t i = 0; i < topCount; ++i) {
		spdlog::info(
			"UploadTelemetry target: rank={} name='{}' bytes={} bufferWrites={} textureWrites={}",
			i + 1u,
			targets[i].first,
			targets[i].second.bytes,
			targets[i].second.bufferWrites,
			targets[i].second.textureWrites);
	}
	std::sort(targets.begin(), targets.end(), [](const auto& lhs, const auto& rhs) {
		const uint64_t lhsWrites = lhs.second.bufferWrites + lhs.second.textureWrites;
		const uint64_t rhsWrites = rhs.second.bufferWrites + rhs.second.textureWrites;
		return lhsWrites != rhsWrites ? lhsWrites > rhsWrites : lhs.second.bytes > rhs.second.bytes;
	});
	for (size_t i = 0; i < topCount; ++i) {
		spdlog::info(
			"UploadTelemetry commands: rank={} name='{}' writes={} bytes={} bufferWrites={} textureWrites={}",
			i + 1u,
			targets[i].first,
			targets[i].second.bufferWrites + targets[i].second.textureWrites,
			targets[i].second.bytes,
			targets[i].second.bufferWrites,
			targets[i].second.textureWrites);
	}
	m_uploadTelemetryTargets.clear();
	m_uploadTelemetryBufferWrites = 0;
	m_uploadTelemetryTextureWrites = 0;
	m_uploadTelemetryBytes = 0;
	m_uploadTelemetryDuplicateTextureSubresources = 0;
	m_uploadTelemetryLastLog = now;
}

void UploadInstance::ProcessDeferredReleases(uint8_t frameIndex) {
	BT_ZONE_SCOPE("UploadInstance::ProcessDeferredReleases");
	std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
	if (m_numFramesInFlight == 0) {
		return;
	}
	frameIndex %= m_numFramesInFlight;

	auto& retiringPages = m_framePages[frameIndex];
	std::unordered_set<Resource*> pendingUploadBuffers;
	pendingUploadBuffers.reserve(m_resourceUpdates.size() + m_textureUpdates.size());
	for (const auto& update : m_resourceUpdates) {
		if (update.uploadBuffer) pendingUploadBuffers.insert(update.uploadBuffer.get());
	}
	for (const auto& update : m_textureUpdates) {
		if (update.uploadBuffer) pendingUploadBuffers.insert(update.uploadBuffer.get());
	}
	for (auto& page : retiringPages) {
		if (!page) {
			continue;
		}
		if (page->buffer && pendingUploadBuffers.contains(page->buffer.get())) {
			// Keep staging memory alive while any bounded batch still references it.
			TrackPageForCurrentFrameLocked(page);
			continue;
		}
		page->tailOffset = 0;
		if (!page->dedicated && page->capacity == m_pageSize) {
			m_freePages.push_back(std::move(page));
		}
	}
	retiringPages.clear();

	retiringPages.swap(m_openPages);
	m_openPageSet.clear();

	const size_t capPages = m_preallocateCapacityBytes / m_pageSize;
	while (AvailableReusableNormalPagesLocked() > capPages && !m_freePages.empty()) {
		m_freePages.pop_back();
	}

	if (m_currentFrameUploadBytes > 0 && !m_recentFrameBytes.empty()) {
		m_recentFrameBytes[m_recentFrameCursor % m_recentFrameBytes.size()] = m_currentFrameUploadBytes;
		m_recentFrameCursor = (m_recentFrameCursor + 1) % m_recentFrameBytes.size();
	}

	m_currentFrameUploadBytes = 0;
	m_activePage.reset();
	RequestWorkerPagesLocked();
}

bool UploadInstance::HasPendingWork() const {
	std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
	return !m_resourceUpdates.empty() || !m_textureUpdates.empty();
}

uint64_t UploadInstance::CapturePendingUploadSequence() {
	std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
	m_lastSealedUploadSequence = m_lastUploadSequence;
	return m_lastSealedUploadSequence;
}

void UploadInstance::CollectPendingDestinations(std::vector<std::shared_ptr<Resource>>& out) const {
	CollectPendingDestinationsThrough(UINT64_MAX, out);
}

void UploadInstance::CollectPendingDestinationsThrough(
	uint64_t sequenceInclusive,
	std::vector<std::shared_ptr<Resource>>& out) const {
	std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
	std::unordered_set<Resource*> seen;
	for (const auto& u : m_resourceUpdates) {
		if (!u.active || u.lastSequence > sequenceInclusive) continue;
		if (u.resourceToUpdate.kind == UploadTarget::Kind::PinnedShared) {
			if (u.resourceToUpdate.pinned && seen.insert(u.resourceToUpdate.pinned.get()).second) {
				out.push_back(u.resourceToUpdate.pinned);
			}
		}
	}
	for (const auto& t : m_textureUpdates) {
		if (t.sequence > sequenceInclusive) continue;
		if (t.texture.kind == UploadTarget::Kind::PinnedShared) {
			if (t.texture.pinned && seen.insert(t.texture.pinned.get()).second) {
				out.push_back(t.texture.pinned);
			}
		}
	}
}

std::string UploadInstance::DescribeQueuedTargetByGlobalResourceId(uint64_t globalResourceId) {
	if (globalResourceId == 0) {
		return {};
	}

	std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
	RefreshQueuedTargetTelemetryLocked();

	std::ostringstream result;
	size_t matchCount = 0;
	auto appendMatch = [&](const auto& update, const UploadTarget& target, const char* updateKind) {
		if (update.targetGlobalResourceId != globalResourceId) {
			return;
		}
		if (matchCount++ > 0) {
			result << " | ";
		}
		result
			<< updateKind
			<< " name='" << (update.targetDebugName.empty() ? std::string("<unknown>") : update.targetDebugName) << "'";
#if BUILD_TYPE == BUILD_TYPE_DEBUG
		result << " queuedAt=" << (update.file ? update.file : "<unknown>") << ":" << update.line;
#endif
		if (target.kind == UploadTarget::Kind::RegistryHandle) {
			result
				<< " handle[idx=" << target.h.GetKey().idx
				<< " gen=" << target.h.GetGeneration()
				<< " epoch=" << target.h.GetEpoch()
				<< "]";
		}
	};

	for (const auto& update : m_resourceUpdates) {
		appendMatch(update, update.resourceToUpdate, "buffer-upload");
	}
	for (const auto& update : m_textureUpdates) {
		appendMatch(update, update.texture, "texture-upload");
	}
	return result.str();
}

void UploadInstance::Cleanup() {
	StopWorker();
	std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
	m_freePages.clear();
	m_readyPages.clear();
	m_openPages.clear();
	m_openPageSet.clear();
	for (auto& pages : m_framePages) {
		pages.clear();
	}
	m_activePage.reset();
	m_resourceUpdates.clear();
	m_textureUpdates.clear();
	m_currentFrameUploadBytes = 0;
}


} // namespace org
