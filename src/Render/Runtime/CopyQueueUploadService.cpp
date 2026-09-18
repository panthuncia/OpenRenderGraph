#include "Render/Runtime/CopyQueueUploadService.h"

#include <algorithm>
#include <cstring>
#include <unordered_set>

#include <rhi_helpers.h>
#include <spdlog/spdlog.h>
#include <BasicTelemetry/Telemetry.h>
#include <BasicTelemetry/Tracy.h>

#include "Resources/Resource.h"

namespace org::runtime {

namespace {
	constexpr size_t kRegionAlignment = 16;

	size_t AlignUp(size_t value, size_t alignment) noexcept {
		return (value + alignment - 1) & ~(alignment - 1);
	}
}

CopyQueueUploadService::CopyQueueUploadService() = default;

CopyQueueUploadService::~CopyQueueUploadService() {
	Cleanup();
}

CopyQueueUploadService::ThreadSlot& CopyQueueUploadService::Slot() {
	// One open page per producer thread; the epoch invalidates slots that
	// outlive a Cleanup/Initialize cycle so no stale page from an old device
	// is ever written to.
	static thread_local ThreadSlot slot;
	const auto epoch = m_epoch.load(std::memory_order_acquire);
	if (slot.epoch != epoch) {
		slot.page.reset();
		slot.epoch = epoch;
	}
	return slot;
}

void CopyQueueUploadService::Initialize(rhi::Device device, rhi::Queue copyQueue, Config config) {
	if (m_initialized.load(std::memory_order_acquire)) return;
	if (!device || !copyQueue) {
		spdlog::error("CopyQueueUploadService: cannot initialize without a device and a copy queue");
		return;
	}
	m_device = device;
	m_copyQueue = copyQueue;
	m_config = std::move(config);
	m_config.pageBytes = (std::max)(m_config.pageBytes, size_t{ 64 } * 1024);
	m_config.maxInFlightBatches = (std::max)(m_config.maxInFlightBatches, size_t{ 1 });
	m_config.maxBatchCopies = (std::max)(m_config.maxBatchCopies, size_t{ 1 });

	m_timeline = std::make_shared<rhi::TimelinePtr>();
	if (rhi::Failed(m_device.CreateTimeline(*m_timeline, 0, m_config.debugName.c_str())) || !*m_timeline) {
		m_timeline.reset();
		spdlog::error("CopyQueueUploadService '{}': timeline creation failed", m_config.debugName);
		return;
	}
	m_nextTimelineValue.store(0, std::memory_order_release);
	m_lastSubmittedValue.store(0, std::memory_order_release);

	m_inFlightCount.store(0, std::memory_order_release);
	{
		std::lock_guard lock(m_statsMutex);
		m_stats = {};
	}

	m_epoch.fetch_add(1, std::memory_order_acq_rel);
	m_initialized.store(true, std::memory_order_release);
	m_submitter = std::jthread([this](std::stop_token stopToken) { SubmitterMain(stopToken); });
}

// ---------------------------------------------------------------------------
// Pages

std::shared_ptr<CopyQueueUploadService::StagingPage> CopyQueueUploadService::CreatePage(size_t capacity, bool dedicated) {
	auto page = std::make_shared<StagingPage>();
	page->capacity = capacity;
	page->dedicated = dedicated;
	page->index = m_nextPageIndex.fetch_add(1, std::memory_order_relaxed);
	auto desc = rhi::helpers::ResourceDesc::Buffer(capacity, rhi::HeapType::Upload, {}, m_config.debugName.c_str());
	desc.queueSharing = rhi::QueueSharing::Concurrent;
	if (rhi::Failed(m_device.CreateCommittedResource(desc, page->buffer)) || !page->buffer) {
		spdlog::error("CopyQueueUploadService '{}': upload page creation failed bytes={}", m_config.debugName, capacity);
		return {};
	}
	{
		std::lock_guard lock(m_pageMutex);
		m_allPages.push_back(page);
	}
	{
		std::lock_guard lock(m_statsMutex);
		++m_stats.pagesCreated;
	}
	basic_telemetry::AddCounter("ORG.Upload.Worker.PagesCreated");
	basic_telemetry::AddCounter("ORG.Upload.Worker.PageBytesCreated", static_cast<std::int64_t>(capacity));
	return page;
}

std::shared_ptr<CopyQueueUploadService::StagingPage> CopyQueueUploadService::AcquirePage() {
	{
		std::lock_guard lock(m_pageMutex);
		if (!m_freePages.empty()) {
			auto page = std::move(m_freePages.back());
			m_freePages.pop_back();
			return page;
		}
	}
	return CreatePage(m_config.pageBytes, false);
}

void CopyQueueUploadService::ClosePage(std::shared_ptr<StagingPage> page) {
	if (!page) return;
	page->closed.store(true, std::memory_order_release);
	std::lock_guard lock(m_pageMutex);
	m_closedPages.push_back(std::move(page));
}

void CopyQueueUploadService::RetirePages() {
	std::vector<std::shared_ptr<StagingPage>> closed;
	{
		std::lock_guard lock(m_pageMutex);
		closed.swap(m_closedPages);
		if (m_allPages.size() > 256) {
			std::erase_if(m_allPages, [](const auto& weak) { return weak.expired(); });
		}
	}
	std::vector<std::shared_ptr<StagingPage>> stillPending;
	uint64_t recycled = 0;
	uint64_t dropped = 0;
	for (auto& page : closed) {
		if (page->pendingRegions.load(std::memory_order_acquire) != 0) {
			stillPending.push_back(std::move(page));
			continue;
		}
		std::lock_guard lock(m_pageMutex);
		if (page->dedicated || page->capacity != m_config.pageBytes || m_freePages.size() >= m_config.maxFreePages) {
			++dropped;
			continue; // released with the last reference
		}
		page->tail = 0;
		page->closed.store(false, std::memory_order_release);
		m_freePages.push_back(std::move(page));
		++recycled;
	}
	if (!stillPending.empty()) {
		std::lock_guard lock(m_pageMutex);
		for (auto& page : stillPending) m_closedPages.push_back(std::move(page));
	}
	if (recycled != 0 || dropped != 0) {
		std::lock_guard lock(m_statsMutex);
		m_stats.pagesRecycled += recycled;
		m_stats.pagesDropped += dropped;
	}
	if (recycled != 0) basic_telemetry::AddCounter("ORG.Upload.Worker.PagesRecycled", static_cast<std::int64_t>(recycled));
}

// Stages up to chunkBytes from the segment cursor into one page region.
bool CopyQueueUploadService::StageChunk(std::span<const StreamingUploadSegment> segments,
	size_t& segmentIndex, size_t& segmentOffset, size_t chunkBytes,
	std::shared_ptr<StagingPage>& outPage, size_t& outOffset) {
	std::shared_ptr<StagingPage> page;
	size_t offset = 0;
	auto& slot = Slot();
	if (slot.page && slot.page->tail + chunkBytes > slot.page->capacity) {
		ClosePage(std::move(slot.page));
	}
	if (slot.page && !slot.page->buffer) slot.page.reset();
	if (!slot.page) {
		slot.page = AcquirePage();
		if (!slot.page) return false;
	}
	page = slot.page;
	offset = page->tail;
	page->tail = AlignUp(offset + chunkBytes, kRegionAlignment);
	page->pendingRegions.fetch_add(1, std::memory_order_acq_rel);

	void* mapped = nullptr;
	page->buffer->Map(&mapped, 0, page->capacity);
	if (!mapped) {
		page->pendingRegions.fetch_sub(1, std::memory_order_acq_rel);
		spdlog::error("CopyQueueUploadService '{}': upload page map failed", m_config.debugName);
		return false;
	}
	size_t written = 0;
	while (written < chunkBytes && segmentIndex < segments.size()) {
		const auto& segment = segments[segmentIndex];
		const size_t available = segment.size - segmentOffset;
		const size_t take = (std::min)(available, chunkBytes - written);
		if (take != 0 && segment.data) {
			std::memcpy(static_cast<uint8_t*>(mapped) + offset + written,
				static_cast<const uint8_t*>(segment.data) + segmentOffset, take);
		}
		written += take;
		segmentOffset += take;
		if (segmentOffset >= segment.size) {
			++segmentIndex;
			segmentOffset = 0;
		}
	}
	page->buffer->Unmap(offset, chunkBytes);
	outPage = std::move(page);
	outOffset = offset;
	return written == chunkBytes;
}

// ---------------------------------------------------------------------------
// Producer API

std::shared_ptr<TrackedUploadTicket> CopyQueueUploadService::QueueBufferUpload(
	std::span<const StreamingUploadSegment> segments, size_t totalSize,
	org::WorkerOwnedDestination destination, size_t dstOffset) {
	if (!Initialized() || !destination.resource || totalSize == 0 || segments.empty()) return {};
	size_t sum = 0;
	for (const auto& segment : segments) {
		if (segment.data == nullptr && segment.size != 0) return {};
		sum += segment.size;
	}
	if (sum != totalSize) return {};
	BT_ZONE_SCOPE("CopyQueueUploadService::QueueBufferUpload");
	BT_ZONE_VALUE(static_cast<int64_t>(totalSize));

	// An upload larger than a page is split into page-sized chunks that share
	// the producer thread's pages; the ticket rides on the last chunk, and
	// batches complete in order, so the ticket completes after every chunk.
	auto ticket = std::make_shared<TrackedUploadTicket>();
	const auto queued = std::chrono::steady_clock::now();
	size_t segmentIndex = 0;
	size_t segmentOffset = 0;
	size_t remaining = totalSize;
	size_t cursor = 0;
	while (remaining != 0) {
		const size_t chunk = (std::min)(remaining, m_config.pageBytes);
		PendingCopy copy;
		if (!StageChunk(segments, segmentIndex, segmentOffset, chunk, copy.page, copy.srcOffset)) {
			ticket->Cancel();
			return ticket;
		}
		copy.destination = destination.resource;
		copy.dstOffset = dstOffset + cursor;
		copy.size = chunk;
		copy.queued = queued;
		remaining -= chunk;
		cursor += chunk;
		if (remaining == 0) copy.ticket = ticket;
		m_intake.Push(std::move(copy));
	}
	m_wake.notify_one();
	return ticket;
}

// ---------------------------------------------------------------------------
// Submitter

CopyQueueUploadService::CommandPair CopyQueueUploadService::AcquireCommandPair() {
	if (!m_freePairs.empty()) {
		auto pair = std::move(m_freePairs.back());
		m_freePairs.pop_back();
		return pair;
	}
	CommandPair pair;
	if (rhi::Failed(m_device.CreateCommandAllocator(rhi::QueueKind::Copy, pair.allocator)) ||
		rhi::Failed(m_device.CreateCommandList(rhi::QueueKind::Copy, pair.allocator.Get(), pair.list))) {
		spdlog::error("CopyQueueUploadService '{}': copy command list creation failed", m_config.debugName);
		return {};
	}
	return pair;
}

void CopyQueueUploadService::RecycleCommandPair(CommandPair&& pair) {
	if (!pair.allocator || !pair.list) return;
	pair.allocator->Recycle();
	pair.list->Recycle(pair.allocator.Get());
	if (m_freePairs.size() < m_config.maxInFlightBatches + 1) {
		m_freePairs.push_back(std::move(pair));
	}
}

bool CopyQueueUploadService::CollectBatch(Batch& batch) {
	batch.copies.clear();
	batch.bytes = 0;
	PendingCopy copy;
	while (batch.copies.size() < m_config.maxBatchCopies) {
		if (!batch.copies.empty() && m_intake.Count() == 0) break;
		if (!m_intake.Pop(copy)) break;
		if (copy.ticket) {
			auto expected = TrackedUploadTicketState::Queued;
			if (!copy.ticket->state.compare_exchange_strong(expected, TrackedUploadTicketState::Claimed,
					std::memory_order_acq_rel, std::memory_order_acquire)) {
				// Cancelled before it was claimed: release the staging region.
				if (copy.page) copy.page->pendingRegions.fetch_sub(1, std::memory_order_acq_rel);
				std::lock_guard lock(m_statsMutex);
				++m_stats.copiesCancelled;
				continue;
			}
			copy.ticket->NotifyChanged();
		}
		batch.bytes += copy.size;
		batch.copies.push_back(std::move(copy));
		if (batch.bytes >= m_config.maxBatchBytes) break;
	}
	return !batch.copies.empty();
}

bool CopyQueueUploadService::RecordAndSubmit(Batch& batch) {
	BT_ZONE_SCOPE("CopyQueueUploadService::RecordAndSubmit");
	BT_ZONE_VALUE(static_cast<int64_t>(batch.bytes));
	batch.pair = AcquireCommandPair();
	if (!batch.pair.list) {
		FinishBatch(batch, true);
		return false;
	}
	auto& list = batch.pair.list.Get();

	// One barrier per distinct destination on each side of the copies. The
	// same barriers are legal on both backends: buffers have no layout, and the
	// destinations are QueueSharing::Concurrent so no family transfer is needed.
	std::vector<rhi::BufferBarrier> barriers;
	std::unordered_set<Resource*> seen;
	barriers.reserve(batch.copies.size());
	for (const auto& copy : batch.copies) {
		if (!copy.destination || !seen.insert(copy.destination.get()).second) continue;
		rhi::BufferBarrier barrier{};
		barrier.buffer = copy.destination->GetAPIResource().GetHandle();
		barrier.beforeSync = rhi::ResourceSyncState::All;
		barrier.afterSync = rhi::ResourceSyncState::Copy;
		barrier.beforeAccess = rhi::ResourceAccessType::Common;
		barrier.afterAccess = rhi::ResourceAccessType::CopyDest;
		barriers.push_back(barrier);
	}
	if (!barriers.empty()) {
		list.Barriers({ .buffers = { barriers.data(), static_cast<uint32_t>(barriers.size()) } });
	}
	for (const auto& copy : batch.copies) {
		if (!copy.destination || !copy.page) continue;
		list.CopyBufferRegion(copy.destination->GetAPIResource().GetHandle(), copy.dstOffset,
			copy.page->buffer->GetHandle(), copy.srcOffset, copy.size);
	}
	for (auto& barrier : barriers) {
		barrier.beforeSync = rhi::ResourceSyncState::Copy;
		barrier.afterSync = rhi::ResourceSyncState::All;
		barrier.beforeAccess = rhi::ResourceAccessType::CopyDest;
		barrier.afterAccess = rhi::ResourceAccessType::Common;
	}
	if (!barriers.empty()) {
		list.Barriers({ .buffers = { barriers.data(), static_cast<uint32_t>(barriers.size()) } });
	}
	list.End();

	batch.timelineValue = m_nextTimelineValue.fetch_add(1, std::memory_order_acq_rel) + 1;
	auto timeline = m_timeline;
	for (auto& copy : batch.copies) {
		if (!copy.ticket) continue;
		std::lock_guard ticketLock(copy.ticket->timelineMutex);
		copy.ticket->timelineOwner = timeline;
		copy.ticket->timelineValue = batch.timelineValue;
		copy.ticket->isTimelineComplete = [timeline](uint64_t value) {
			return timeline && *timeline && (*timeline)->GetCompletedValue() >= value;
		};
	}
	const rhi::CommandList lists[] = { batch.pair.list.Get() };
	const rhi::TimelinePoint signal{ (*timeline)->GetHandle(), batch.timelineValue };
	if (rhi::Failed(m_copyQueue.Submit(lists, { .signals = { &signal, 1 } }))) {
		spdlog::error("CopyQueueUploadService '{}': copy queue submission failed", m_config.debugName);
		FinishBatch(batch, true);
		return false;
	}
	m_lastSubmittedValue.store(batch.timelineValue, std::memory_order_release);
	const auto now = std::chrono::steady_clock::now();
	for (auto& copy : batch.copies) {
		if (!copy.ticket) continue;
		auto expected = TrackedUploadTicketState::Claimed;
		if (copy.ticket->state.compare_exchange_strong(expected, TrackedUploadTicketState::Submitted,
				std::memory_order_release, std::memory_order_acquire)) {
			copy.ticket->NotifyChanged();
		}
		basic_telemetry::Record("ORG.Upload.Worker.SubmitLatencyUs",
			std::chrono::duration_cast<std::chrono::microseconds>(now - copy.queued).count());
	}
	{
		std::lock_guard lock(m_statsMutex);
		++m_stats.batchesSubmitted;
		m_stats.copiesSubmitted += batch.copies.size();
		m_stats.bytesSubmitted += batch.bytes;
	}
	basic_telemetry::AddCounter("ORG.Upload.Worker.BatchesSubmitted");
	basic_telemetry::AddCounter("ORG.Upload.Worker.BytesSubmitted", static_cast<std::int64_t>(batch.bytes));
	return true;
}

void CopyQueueUploadService::FinishBatch(Batch& batch, bool cancelled) {
	for (auto& copy : batch.copies) {
		if (copy.page) copy.page->pendingRegions.fetch_sub(1, std::memory_order_acq_rel);
		if (!copy.ticket) continue;
		if (cancelled) {
			copy.ticket->Cancel();
		} else {
			(void)copy.ticket->Complete();
		}
	}
	if (cancelled) {
		std::lock_guard lock(m_statsMutex);
		m_stats.copiesCancelled += batch.copies.size();
	}
	batch.copies.clear();
	if (batch.pair.list) RecycleCommandPair(std::move(batch.pair));
}

void CopyQueueUploadService::CompleteFinishedBatches(uint64_t completedValue) {
	while (!m_inFlight.empty() && m_inFlight.front().timelineValue <= completedValue) {
		auto batch = std::move(m_inFlight.front());
		m_inFlight.pop_front();
		FinishBatch(batch, false);
		m_inFlightCount.store(m_inFlight.size(), std::memory_order_release);
	}
}

void CopyQueueUploadService::SubmitterMain(std::stop_token stopToken) {
	try {
		for (;;) {
			if (!m_inFlight.empty()) {
				CompleteFinishedBatches((*m_timeline)->GetCompletedValue());
			}
			bool submittedAny = false;
			while (m_inFlight.size() < m_config.maxInFlightBatches &&
				m_intake.Count() != 0 && !stopToken.stop_requested()) {
				Batch batch;
				if (!CollectBatch(batch)) break;
				if (RecordAndSubmit(batch)) {
					m_inFlight.push_back(std::move(batch));
					m_inFlightCount.store(m_inFlight.size(), std::memory_order_release);
					submittedAny = true;
				}
			}
			RetirePages();
			if (m_inFlight.empty() && m_intake.Count() == 0) {
				m_idle.notify_all();
			}
			if (stopToken.stop_requested() && m_inFlight.empty()) break;
			if (submittedAny) continue;
			std::unique_lock lock(m_wakeMutex);
			if (!m_inFlight.empty()) {
				// Poll completion at millisecond granularity; a new upload wakes us sooner.
				m_wake.wait_for(lock, stopToken, std::chrono::milliseconds(1), [this] {
					return m_intake.Count() != 0;
				});
			} else {
				m_wake.wait(lock, stopToken, [this] {
					return m_intake.Count() != 0;
				});
			}
		}
	} catch (const std::exception& error) {
		spdlog::error("CopyQueueUploadService '{}': submitter stopped after exception: {}", m_config.debugName, error.what());
	} catch (...) {
		spdlog::error("CopyQueueUploadService '{}': submitter stopped after unknown exception", m_config.debugName);
	}
}

// ---------------------------------------------------------------------------

bool CopyQueueUploadService::WaitIdle(uint32_t timeoutMs) {
	if (!Initialized()) return true;
	std::unique_lock lock(m_wakeMutex);
	const auto idle = [this] {
		return m_intake.Count() == 0 &&
			m_inFlightCount.load(std::memory_order_acquire) == 0;
	};
	if (idle()) return true;
	m_wake.notify_one();
	if (timeoutMs == UINT32_MAX) {
		m_idle.wait(lock, idle);
		return true;
	}
	return m_idle.wait_for(lock, std::chrono::milliseconds(timeoutMs), idle);
}

CopyQueueUploadService::Stats CopyQueueUploadService::GetStats() const {
	Stats stats;
	{
		std::lock_guard lock(m_statsMutex);
		stats = m_stats;
	}
	stats.queuedCopies = m_intake.Count();
	stats.inFlightBatches = m_inFlightCount.load(std::memory_order_acquire);
	{
		std::lock_guard lock(m_pageMutex);
		stats.freePages = m_freePages.size();
		stats.closedPages = m_closedPages.size();
	}
	return stats;
}

void CopyQueueUploadService::Cleanup() {
	if (!m_initialized.exchange(false, std::memory_order_acq_rel)) return;
	// Producers observing !Initialized() get null tickets from here on; a
	// producer that already staged bytes still pushes, and the drain below
	// cancels it.
	if (m_submitter.joinable()) {
		m_submitter.request_stop();
		m_wake.notify_all();
		m_submitter.join();
	}
	PendingCopy copy;
	while (m_intake.Pop(copy)) {
		if (copy.ticket) copy.ticket->Cancel();
		if (copy.page) copy.page->pendingRegions.fetch_sub(1, std::memory_order_acq_rel);
	}
	for (auto& batch : m_inFlight) FinishBatch(batch, true);
	m_inFlight.clear();
	m_inFlightCount.store(0, std::memory_order_release);
	if (m_timeline && *m_timeline) {
		const auto last = m_lastSubmittedValue.load(std::memory_order_acquire);
		if (last != 0) (void)(*m_timeline)->HostWait(last);
	}
	m_freePairs.clear();
	m_epoch.fetch_add(1, std::memory_order_acq_rel); // invalidate every thread's open page
	{
		std::lock_guard lock(m_pageMutex);
		m_freePages.clear();
		m_closedPages.clear();
		// A producer thread's open page may outlive this call in its thread
		// slot; drop the GPU buffer now so nothing survives the device.
		for (auto& weak : m_allPages) {
			if (auto page = weak.lock()) page->buffer.Reset();
		}
		m_allPages.clear();
	}
	m_timeline.reset();
}

} // namespace org::runtime
