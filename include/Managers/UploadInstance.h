#pragma once

#include <span>

#include <vector>
#include <memory>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_set>
#include <unordered_map>

#include <rhi.h>
#include <rhi_helpers.h>

#include "Render/Runtime/UploadTypes.h"
#include "Render/Runtime/ITaskService.h"
#include "Render/Runtime/MpscQueue.h"

namespace org::imm { class ImmediateCommandList; }
namespace org::runtime { class StagedUploadBatch; }

namespace org {

class Resource;
class Buffer;

// A standalone, non-singleton upload-heap primitive.
//
// Encapsulates the core upload-page ring buffer, CPU staging, coalescing,
// GPU copy emission, and frame-based page retirement that UploadManager
// provides, but as a reusable instance that can be scheduled on any queue.
//
// Usage:
//   1. Create an UploadInstance (one per subsystem / queue that needs uploads).
//   2. Call UploadData() / UploadTextureSubresources() during the Update phase.
//   3. Call ProcessUploads() from a pass's RecordImmediateCommands() to emit GPU copies.
//   4. Call ProcessDeferredReleases() once per frame (after GPU retire) to reclaim pages.
//
// Threading: the instance is owned by one thread (SetOwnerThread, the render
// thread). Every queue operation runs on that thread and there is no queue
// mutex. Other threads reach it only through lock-free mailboxes: posted
// uploads (PostTextureSubresources, off-owner UploadData) drained by the owner
// before it records, and pages prepared by the background task. Off-owner
// producer calls are counted (ORG.Upload.FrameInstance.OffThreadCalls).
class UploadInstance {
public:
	using UploadTarget        = org::runtime::UploadTarget;
	using UploadResolveContext = org::runtime::UploadResolveContext;

	static constexpr size_t kDefaultPageSize = 16 * 1024 * 1024; // 16 MB
	static constexpr size_t kDefaultPreallocateCapacity = 128 * 1024 * 1024; // 128 MB

	struct Config {
		uint8_t numFramesInFlight = 3;
		size_t pageSizeBytes = kDefaultPageSize;
		size_t preallocateCapacityBytes = kDefaultPreallocateCapacity;
		std::string debugName = "UploadInstance";
		std::string pageNamePrefix = "UploadInstancePage";
		std::string usageHint = "UploadInstance page";
	};

	struct ResourceUpdate {
		size_t size{};
		UploadTarget resourceToUpdate{};
		std::shared_ptr<Resource> uploadBuffer;
		size_t uploadBufferOffset{};
		size_t dataBufferOffset{};
		bool active = true;
		bool staging = false;
		uint64_t firstSequence = 0;
		uint64_t lastSequence = 0;
#if BUILD_TYPE == BUILD_TYPE_DEBUG
		const char* file{};
		int line{};
		static constexpr int MaxStack = 8;
		void* stack[MaxStack]{};
		uint8_t stackSize{};
#endif
		uint64_t targetGlobalResourceId = 0;
		std::string targetDebugName;
	};

	struct TextureUpdate {
		UploadTarget texture;
		uint32_t mip{};
		uint32_t slice{};
		rhi::CopyableFootprint footprint{};
		uint32_t x{};
		uint32_t y{};
		uint32_t z{};
		std::shared_ptr<Resource> uploadBuffer;
		bool active = true;
		bool staging = false;
		uint64_t sequence = 0;
#if BUILD_TYPE == BUILD_TYPE_DEBUG
		const char* file{};
		int line{};
#endif
		uint64_t targetGlobalResourceId = 0;
		std::string targetDebugName;
	};

	using PendingWorkChangedCallback = std::function<void()>;
	using TargetTelemetryCallback = std::function<void(const UploadTarget&, uint64_t&, std::string&)>;
	using UploadRegion = org::runtime::UploadRegion;
	using InvalidRegistryHandleCallback = std::function<bool(const UploadTarget&, const char* reason, const char* file, int line)>;

	// Construct an upload instance.
	// @param numFramesInFlight  Number of frames in flight for page retirement.
	// @param pageSize           Size of each upload-heap page in bytes (default 16 MB).
	explicit UploadInstance(uint8_t numFramesInFlight, size_t pageSize = kDefaultPageSize);
	explicit UploadInstance(Config config);
	~UploadInstance();

	UploadInstance(const UploadInstance&) = delete;
	UploadInstance& operator=(const UploadInstance&) = delete;
	UploadInstance(UploadInstance&&) = delete;
	UploadInstance& operator=(UploadInstance&&) = delete;

	// Buffer uploads
#if BUILD_TYPE == BUILD_TYPE_DEBUG
	void UploadData(const void* data, size_t size, UploadTarget target, size_t dstOffset,
	                const char* file = nullptr, int line = 0);
	void UploadDataBatch(UploadTarget target, std::span<const UploadRegion> regions,
	                     const char* file = nullptr, int line = 0);
#else
	void UploadData(const void* data, size_t size, UploadTarget target, size_t dstOffset);
	void UploadDataBatch(UploadTarget target, std::span<const UploadRegion> regions);
#endif

	// Texture uploads
#if BUILD_TYPE == BUILD_TYPE_DEBUG
	void UploadTextureSubresources(
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
		int line);
#else
	void UploadTextureSubresources(
		UploadTarget target,
		rhi::Format fmt,
		uint32_t baseWidth,
		uint32_t baseHeight,
		uint32_t depthOrLayers,
		uint32_t mipLevels,
		uint32_t arraySize,
		const rhi::helpers::SubresourceData* srcSubresources,
		uint32_t srcCount);
#endif

	// Execution

	// Emit GPU copy commands for all queued buffer and texture uploads.
	// Call from a pass's RecordImmediateCommands().
	// Owner-thread texture upload with caller-owned bytes; from any other
	// thread the request is posted and recorded by the owner. See IUploadService.
#if BUILD_TYPE == BUILD_TYPE_DEBUG
	void PostTextureSubresources(UploadTarget target, rhi::Format fmt, uint32_t baseWidth, uint32_t baseHeight,
		uint32_t depthOrLayers, uint32_t mipLevels, uint32_t arraySize,
		std::shared_ptr<const std::vector<rhi::helpers::SubresourceData>> subresources,
		std::shared_ptr<const void> keepAlive, const char* file = nullptr, int line = 0);
#else
	void PostTextureSubresources(UploadTarget target, rhi::Format fmt, uint32_t baseWidth, uint32_t baseHeight,
		uint32_t depthOrLayers, uint32_t mipLevels, uint32_t arraySize,
		std::shared_ptr<const std::vector<rhi::helpers::SubresourceData>> subresources,
		std::shared_ptr<const void> keepAlive);
#endif

	void ProcessUploads(uint8_t frameIndex, org::imm::ImmediateCommandList& commandList);
	void ProcessUploadsThrough(uint8_t frameIndex, org::imm::ImmediateCommandList& commandList, uint64_t sequenceInclusive);

	// Retire upload-heap pages that are no longer referenced by any in-flight frame.
	// Call once per frame after the GPU has finished with the retiring frame.
	void ProcessDeferredReleases(uint8_t frameIndex);

	// Owner thread: queues a producer's staged batch (IUploadService::SubmitStagedUploads).
	void SubmitStagedUploads(std::shared_ptr<org::runtime::StagedUploadBatch> batch);
	void SetStagedUploadsRecordedDirectly(bool direct);
	size_t RecordStagedUploads(rhi::CommandList& list, uint8_t frameIndex);

	// Configuration

	void SetResolveContext(UploadResolveContext ctx);
	// The frame upload instance is owned by the render thread. Producer calls
	// from any other thread are counted (ORG.Upload.FrameInstance.OffThreadCalls
	// and per-API) so they can be routed to the worker upload service; once the
	// count is zero the queue mutex can be removed.
	void SetOwnerThread();
	bool IsOwnerThread() const noexcept;
	void SetPendingWorkChangedCallback(PendingWorkChangedCallback callback);
	void SetTargetTelemetryCallback(TargetTelemetryCallback callback);
	void SetInvalidRegistryHandleCallback(InvalidRegistryHandleCallback callback);

	// Returns true if there are any pending buffer or texture uploads.
	bool HasPendingWork() const;
	uint64_t CapturePendingUploadSequence();

	// Collect the destination resources that pending uploads will write to.
	// Call during DeclareResourceUsages to declare copy targets.
	void CollectPendingDestinations(std::vector<std::shared_ptr<Resource>>& out) const;
	void CollectPendingDestinationsThrough(uint64_t sequenceInclusive, std::vector<std::shared_ptr<Resource>>& out) const;

	std::string DescribeQueuedTargetByGlobalResourceId(uint64_t globalResourceId);

	void Cleanup();

private:
	// Nested types

	struct UploadPage {
		std::shared_ptr<Buffer> buffer;
		size_t                  tailOffset = 0;
		size_t                  capacity = 0;
		size_t                  index = 0;
		bool                    dedicated = false;
		bool                    tagged = false;
	};
	using UploadPagePtr = std::shared_ptr<UploadPage>;

	// Internal helpers

	// A region larger than a page needs a dedicated page; callers create it with
	// PrepareDedicatedPage so GPU allocation
	// never runs under the lock the render thread drains through.
	UploadPagePtr PrepareDedicatedPage(size_t size);
	// Owner-thread texture upload with an optional pre-created dedicated page.
	void UploadTextureSubresourcesPrepared(UploadTarget target, rhi::Format fmt, uint32_t baseWidth, uint32_t baseHeight,
		uint32_t depthOrLayers, uint32_t mipLevels, uint32_t arraySize,
		const rhi::helpers::SubresourceData* srcSubresources, uint32_t srcCount,
		UploadPagePtr preparedPage, const char* file, int line);
	bool AllocateUploadRegion(size_t size, size_t alignment,
	                          std::shared_ptr<Resource>& outUploadBuffer, size_t& outOffset,
	                          UploadPagePtr preparedDedicated = nullptr);

	static bool TryCoalesceAppend(ResourceUpdate& last, const ResourceUpdate& next) noexcept;

	static void MapUpload(const std::shared_ptr<Resource>& uploadBuffer, uint8_t** outMapped) noexcept;
	static void UnmapUpload(
		const std::shared_ptr<Resource>& uploadBuffer,
		size_t writeOffset,
		size_t writeSize) noexcept;
	void MarkPendingWorkChangedLocked();
	void CaptureTargetTelemetry(const UploadTarget& target, uint64_t& outId, std::string& outName);
	void CaptureTargetTelemetryLocked(const UploadTarget& target, uint64_t& outId, std::string& outName);
	void RefreshQueuedTargetTelemetryLocked();
	void PruneInvalidRegistryHandleUpdatesLocked(const char* reason);
	UploadPagePtr CreatePage(size_t size, bool dedicated);
	void TagPage(const UploadPagePtr& page);
	UploadPagePtr AcquirePageLocked(size_t minSize, bool dedicated);
	void TrackPageForCurrentFrameLocked(const UploadPagePtr& page);
	void StartWorker();
	void StopWorker();
	void WorkerMain();
	void ScheduleWorkerDrain();
	void RequestWorkerPagesLocked();
	void RecordProcessedUploadTelemetry(
		const std::vector<ResourceUpdate>& resourceUpdates,
		const std::vector<TextureUpdate>& textureUpdates);
	size_t NormalPageCountForBytes(size_t bytes) const noexcept;
	size_t WarmTargetPageCountLocked() const noexcept;
	size_t AvailableReusableNormalPagesLocked() const noexcept;

	// State

	size_t                     m_pageSize;
	size_t                     m_preallocateCapacityBytes = kDefaultPreallocateCapacity;
	std::string                m_debugName = "UploadInstance";
	std::string                m_pageNamePrefix = "UploadInstancePage";
	std::string                m_usageHint = "UploadInstance page";

	std::deque<UploadPagePtr>  m_freePages;
	std::deque<UploadPagePtr>  m_readyPages;
	std::vector<UploadPagePtr> m_openPages;
	std::unordered_set<UploadPage*> m_openPageSet;
	std::vector<std::vector<UploadPagePtr>> m_framePages;
	// Staged batches submitted since the last upload pass, and those the pass of each frame slot recorded
	// (released with the slot).
	std::vector<std::shared_ptr<org::runtime::StagedUploadBatch>> m_pendingStaged;
	// Direct mode: submitted batches waiting for RecordStagedUploads.
	bool m_stagedDirect = false;
	std::vector<std::shared_ptr<org::runtime::StagedUploadBatch>> m_directStaged;
	std::vector<std::vector<std::shared_ptr<org::runtime::StagedUploadBatch>>> m_frameStaged;
	UploadPagePtr             m_activePage;
	std::atomic_size_t         m_nextPageIndex = 0;
	struct PageLifetimeTrace {
		std::weak_ptr<Resource> buffer;
		size_t capacity = 0;
		size_t index = 0;
		bool dedicated = false;
	};
	std::mutex m_pageLifetimeTraceMutex;
	std::vector<PageLifetimeTrace> m_pageLifetimeTraces;
	uint8_t                    m_numFramesInFlight;
	size_t                     m_currentFrameUploadBytes = 0;
	std::vector<size_t>        m_recentFrameBytes;
	size_t                     m_recentFrameCursor = 0;

	std::vector<ResourceUpdate>  m_resourceUpdates;
	std::vector<TextureUpdate>   m_textureUpdates;
	uint64_t                     m_lastUploadSequence = 0;
	uint64_t                     m_lastSealedUploadSequence = 0;
	struct UploadTelemetryTarget {
		uint64_t bufferWrites = 0;
		uint64_t textureWrites = 0;
		uint64_t bytes = 0;
	};
	std::unordered_map<std::string, UploadTelemetryTarget> m_uploadTelemetryTargets;
	std::chrono::steady_clock::time_point m_uploadTelemetryLastLog{};
	std::chrono::steady_clock::time_point m_lifetimeTelemetryLastLog{};
	uint64_t m_uploadTelemetryBufferWrites = 0;
	uint64_t m_uploadTelemetryTextureWrites = 0;
	uint64_t m_uploadTelemetryBytes = 0;
	uint64_t m_uploadTelemetryDuplicateTextureSubresources = 0;

	UploadResolveContext m_ctx{};
	PendingWorkChangedCallback m_pendingWorkChanged;
	TargetTelemetryCallback m_targetTelemetry;
	InvalidRegistryHandleCallback m_invalidRegistryHandle;

	void NoteProducerCall(const char* api, const UploadTarget& target);
	void NoteOffOwnerCall(const char* api);
	// Owner-thread: apply uploads other threads posted.
	void DrainPostedUploads();
	// Owner-thread: move pages the background task prepared into m_readyPages.
	void DrainReadyPageInbox();
	std::atomic<std::size_t> m_ownerThreadHash{ 0 };
	std::atomic<std::uint32_t> m_offThreadCallsLogged{ 0 };

	// A texture upload staged entirely on the posting thread: it created the
	// dedicated page and copied the subresources into it. The owner only
	// registers the copy records, so no allocation or memcpy runs on the
	// render thread for a posted upload.
	struct PostedTextureUpload {
		UploadTarget target;
		UploadPagePtr page;
		std::vector<TextureUpdate> updates; // footprint offsets are page-relative
		size_t bytes = 0;
		bool mapped = false;
	};
	struct PostedBufferUpload {
		UploadTarget target;
		std::vector<uint8_t> bytes;
		size_t dstOffset = 0;
		const char* file = nullptr;
		int line = 0;
	};
	org::runtime::MpscQueue<PostedTextureUpload> m_postedTextureUploads;
	org::runtime::MpscQueue<PostedBufferUpload> m_postedBufferUploads;
	org::runtime::MpscQueue<UploadPagePtr> m_readyPageInbox;

	std::mutex m_workerMutex; // page-preparation task lifetime only (start/stop)
	std::shared_ptr<org::runtime::ITaskService> m_taskService;
	std::shared_ptr<org::runtime::ITaskScope> m_taskScope;
	std::atomic<bool> m_workerDrainScheduled{false};
	std::atomic<bool> m_workerQuit{ false };
	std::atomic<size_t> m_workerRequestedPages{ 0 };
};

} // namespace org
