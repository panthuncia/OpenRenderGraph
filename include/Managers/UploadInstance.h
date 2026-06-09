#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>

#include <rhi.h>
#include <rhi_helpers.h>

#include "Render/Runtime/UploadTypes.h"

namespace rg::imm { class ImmediateCommandList; }
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
class UploadInstance {
public:
	using UploadTarget        = rg::runtime::UploadTarget;
	using UploadResolveContext = rg::runtime::UploadResolveContext;

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
#if BUILD_TYPE == BUILD_TYPE_DEBUG
		const char* file{};
		int line{};
#endif
		uint64_t targetGlobalResourceId = 0;
		std::string targetDebugName;
	};

	using PendingWorkChangedCallback = std::function<void()>;
	using TargetTelemetryCallback = std::function<void(const UploadTarget&, uint64_t&, std::string&)>;
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
#else
	void UploadData(const void* data, size_t size, UploadTarget target, size_t dstOffset);
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
	void ProcessUploads(uint8_t frameIndex, rg::imm::ImmediateCommandList& commandList);

	// Retire upload-heap pages that are no longer referenced by any in-flight frame.
	// Call once per frame after the GPU has finished with the retiring frame.
	void ProcessDeferredReleases(uint8_t frameIndex);

	// Configuration

	void SetResolveContext(UploadResolveContext ctx);
	void SetPendingWorkChangedCallback(PendingWorkChangedCallback callback);
	void SetTargetTelemetryCallback(TargetTelemetryCallback callback);
	void SetInvalidRegistryHandleCallback(InvalidRegistryHandleCallback callback);

	// Returns true if there are any pending buffer or texture uploads.
	bool HasPendingWork() const;

	// Collect the destination resources that pending uploads will write to.
	// Call during DeclareResourceUsages to declare copy targets.
	void CollectPendingDestinations(std::vector<std::shared_ptr<Resource>>& out) const;
	void DeclarePendingUploadResourceUsages(
		const std::function<void(const std::shared_ptr<Resource>&)>& copySource,
		const std::function<void(const UploadTarget&, uint32_t mip, uint32_t slice)>& copyDest);

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

	bool AllocateUploadRegion(size_t size, size_t alignment,
	                          std::shared_ptr<Resource>& outUploadBuffer, size_t& outOffset);

	static bool TryCoalesceAppend(ResourceUpdate& last, const ResourceUpdate& next) noexcept;

	static void MapUpload(const std::shared_ptr<Resource>& uploadBuffer, size_t mapSize,
	                       uint8_t** outMapped) noexcept;
	static void UnmapUpload(const std::shared_ptr<Resource>& uploadBuffer) noexcept;
	void MarkPendingWorkChangedLocked();
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
	void RequestWorkerPagesLocked();
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
	UploadPagePtr             m_activePage;
	std::atomic_size_t         m_nextPageIndex = 0;
	uint8_t                    m_numFramesInFlight;
	size_t                     m_currentFrameUploadBytes = 0;
	std::vector<size_t>        m_recentFrameBytes;
	size_t                     m_recentFrameCursor = 0;

	std::vector<ResourceUpdate>  m_resourceUpdates;
	std::vector<TextureUpdate>   m_textureUpdates;
	std::vector<uint64_t>        m_declarePendingUploadSourceBits;
	std::vector<size_t>          m_declarePendingUploadSourceMarkedWords;
	std::vector<uint64_t>        m_declarePendingUploadBufferDestBits;
	std::vector<size_t>          m_declarePendingUploadBufferDestMarkedWords;

	UploadResolveContext m_ctx{};
	PendingWorkChangedCallback m_pendingWorkChanged;
	TargetTelemetryCallback m_targetTelemetry;
	InvalidRegistryHandleCallback m_invalidRegistryHandle;

	mutable std::mutex m_uploadQueueMutex;
	std::mutex m_workerMutex;
	std::condition_variable m_workerCV;
	std::thread m_workerThread;
	bool m_workerQuit = false;
	size_t m_workerRequestedPages = 0;
};
