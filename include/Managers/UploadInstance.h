#pragma once

#include <vector>
#include <memory>
#include <cstddef>
#include <cstdint>

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

	// Construct an upload instance.
	// @param numFramesInFlight  Number of frames in flight for page retirement.
	// @param pageSize           Size of each upload-heap page in bytes (default 256 MB).
	explicit UploadInstance(uint8_t numFramesInFlight, size_t pageSize = kDefaultPageSize);

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

	void SetResolveContext(UploadResolveContext ctx) { m_ctx = ctx; }

	// Returns true if there are any pending buffer or texture uploads.
	bool HasPendingWork() const;

	// Collect the destination resources that pending uploads will write to.
	// Call during DeclareResourceUsages to declare copy targets.
	void CollectPendingDestinations(std::vector<std::shared_ptr<Resource>>& out) const;

	void Cleanup();

	// Constants

	static constexpr size_t kDefaultPageSize = 256 * 1024 * 1024; // 256 MB

private:
	// Nested types

	struct UploadPage {
		std::shared_ptr<Buffer> buffer;
		size_t                  tailOffset = 0;
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
	};

	// Internal helpers

	bool AllocateUploadRegion(size_t size, size_t alignment,
	                          std::shared_ptr<Resource>& outUploadBuffer, size_t& outOffset);

	static bool TryCoalesceAppend(ResourceUpdate& last, const ResourceUpdate& next) noexcept;

	static void MapUpload(const std::shared_ptr<Resource>& uploadBuffer, size_t mapSize,
	                       uint8_t** outMapped) noexcept;
	static void UnmapUpload(const std::shared_ptr<Resource>& uploadBuffer) noexcept;

	// State

	size_t                     m_pageSize;
	std::vector<UploadPage>    m_pages;
	size_t                     m_activePage = 0;
	std::vector<size_t>        m_frameStart;
	uint8_t                    m_numFramesInFlight;

	std::vector<ResourceUpdate>  m_resourceUpdates;
	std::vector<TextureUpdate>   m_textureUpdates;

	UploadResolveContext m_ctx{};
};
