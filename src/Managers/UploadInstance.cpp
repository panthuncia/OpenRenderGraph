#include "Managers/UploadInstance.h"

#include <algorithm>
#include <cstring>
#include <sstream>

#include <rhi_helpers.h>
#include <spdlog/spdlog.h>
#include <tracy/Tracy.hpp>

#include "Resources/Buffers/Buffer.h"
#include "Resources/Resource.h"
#include "Render/MemoryIntrospectionAPI.h"
#include "Render/ImmediateExecution/ImmediateCommandList.h"

namespace {
	size_t AlignUpSizeT(const size_t v, const size_t a) noexcept {
		return (v + (a - 1)) & ~(a - 1);
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
	if (m_workerThread.joinable()) {
		return;
	}
	m_workerQuit = false;
	m_workerThread = std::thread(&UploadInstance::WorkerMain, this);
}

void UploadInstance::StopWorker() {
	{
		std::lock_guard<std::mutex> lock(m_workerMutex);
		m_workerQuit = true;
	}
	m_workerCV.notify_all();
	if (m_workerThread.joinable()) {
		m_workerThread.join();
	}
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
		rg::memory::SetResourceUsageHint(*page->buffer, m_usageHint);
		page->tagged = true;
	}
}

void UploadInstance::WorkerMain() {
	for (;;) {
		size_t pagesToCreate = 0;
		{
			std::unique_lock<std::mutex> lock(m_workerMutex);
			m_workerCV.wait(lock, [this] {
				return m_workerQuit || m_workerRequestedPages > 0;
			});
			if (m_workerQuit) {
				return;
			}
			pagesToCreate = m_workerRequestedPages;
			m_workerRequestedPages = 0;
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
	m_workerCV.notify_one();
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

void UploadInstance::MapUpload(const std::shared_ptr<Resource>& uploadBuffer, size_t mapSize,
                               uint8_t** outMapped) noexcept {
	if (!outMapped) return;
	*outMapped = nullptr;
	if (!uploadBuffer) return;
	uploadBuffer->GetAPIResource().Map(reinterpret_cast<void**>(outMapped), 0, mapSize);
}

void UploadInstance::UnmapUpload(const std::shared_ptr<Resource>& uploadBuffer) noexcept {
	if (!uploadBuffer) return;
	uploadBuffer->GetAPIResource().Unmap(0, 0);
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
	}

	uint8_t* mapped = nullptr;
	MapUpload(uploadBuffer, uploadOffset + size, &mapped);
#if BUILD_TYPE == BUILD_TYPE_DEBUG
	if (!mapped) {
		__debugbreak();
		return;
	}
#endif
	if (mapped) {
		std::memcpy(mapped + uploadOffset, data, size);
	}
	UnmapUpload(uploadBuffer);

	std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
	for (int i = static_cast<int>(m_resourceUpdates.size()) - 1; i >= 0; --i) {
		auto& last = m_resourceUpdates[static_cast<size_t>(i)];
		if (!last.active) {
			continue;
		}
		if (TryCoalesceAppend(last, update)) {
			MarkPendingWorkChangedLocked();
			return;
		}
		break;
	}
	m_resourceUpdates.push_back(std::move(update));
	MarkPendingWorkChangedLocked();
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
	}

	uint8_t* mapped = nullptr;
	MapUpload(uploadBuffer, uploadBaseOffset + static_cast<size_t>(plan.totalSize), &mapped);
	if (mapped) {
		rhi::helpers::WriteTextureUploadSubresources(plan, srcSpan, mapped, static_cast<uint64_t>(uploadBaseOffset));
	}
	UnmapUpload(uploadBuffer);

	std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
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
		m_textureUpdates.push_back(std::move(update));
	}
	MarkPendingWorkChangedLocked();
}

void UploadInstance::ProcessUploads(uint8_t frameIndex, rg::imm::ImmediateCommandList& commandList) {
	(void)frameIndex;

	std::vector<ResourceUpdate> resourceUpdates;
	std::vector<TextureUpdate> textureUpdates;
	UploadResolveContext ctx;
	{
		std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
		PruneInvalidRegistryHandleUpdatesLocked("upload-pass-execute");
		resourceUpdates.swap(m_resourceUpdates);
		textureUpdates.swap(m_textureUpdates);
		ctx = m_ctx;
		if (!resourceUpdates.empty() || !textureUpdates.empty()) {
			MarkPendingWorkChangedLocked();
		}
	}

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

void UploadInstance::ProcessDeferredReleases(uint8_t frameIndex) {
	ZoneScopedN("UploadInstance::ProcessDeferredReleases");
	std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
	if (m_numFramesInFlight == 0) {
		return;
	}
	frameIndex %= m_numFramesInFlight;

	auto& retiringPages = m_framePages[frameIndex];
	for (auto& page : retiringPages) {
		if (!page) {
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

void UploadInstance::CollectPendingDestinations(std::vector<std::shared_ptr<Resource>>& out) const {
	std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
	std::unordered_set<Resource*> seen;
	for (auto& u : m_resourceUpdates) {
		if (!u.active) continue;
		if (u.resourceToUpdate.kind == UploadTarget::Kind::PinnedShared) {
			if (u.resourceToUpdate.pinned && seen.insert(u.resourceToUpdate.pinned.get()).second) {
				out.push_back(u.resourceToUpdate.pinned);
			}
		}
	}
	for (auto& t : m_textureUpdates) {
		if (t.texture.kind == UploadTarget::Kind::PinnedShared) {
			if (t.texture.pinned && seen.insert(t.texture.pinned.get()).second) {
				out.push_back(t.texture.pinned);
			}
		}
	}
}

void UploadInstance::DeclarePendingUploadResourceUsages(
	const std::function<void(const std::shared_ptr<Resource>&)>& copySource,
	const std::function<void(const UploadTarget&, uint32_t mip, uint32_t slice)>& copyDest)
{
	std::lock_guard<std::mutex> lock(m_uploadQueueMutex);
	RefreshQueuedTargetTelemetryLocked();
	PruneInvalidRegistryHandleUpdatesLocked("upload-pass-declare");

	for (const auto& update : m_resourceUpdates) {
		if (!update.active || !update.uploadBuffer) {
			continue;
		}
		copySource(update.uploadBuffer);
		copyDest(update.resourceToUpdate, UINT32_MAX, UINT32_MAX);
	}
	for (const auto& update : m_textureUpdates) {
		if (!update.uploadBuffer) {
			continue;
		}
		copySource(update.uploadBuffer);
		copyDest(update.texture, update.mip, update.slice);
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
