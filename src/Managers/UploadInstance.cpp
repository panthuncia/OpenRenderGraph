#include "Managers/UploadInstance.h"

#include <algorithm>
#include <thread>
#include <unordered_set>

#include <rhi_helpers.h>
#include <spdlog/spdlog.h>

#include "Resources/Buffers/Buffer.h"
#include "Resources/Resource.h"
#include "Render/MemoryIntrospectionAPI.h"
#include "Render/ImmediateExecution/ImmediateCommandList.h"

namespace {
	size_t AlignUpSizeT(const size_t v, const size_t a) noexcept {
		return (v + (a - 1)) & ~(a - 1);
	}

	void TagUploadInstancePage(const std::shared_ptr<Buffer>& buffer, size_t pageIndex)
	{
		if (!buffer) {
			return;
		}
		buffer->SetName("UploadInstancePage_" + std::to_string(pageIndex));
		rg::memory::SetResourceUsageHint(*buffer, "UploadInstance page");
	}

	void LogUploadInstancePages(const char* reason, size_t pageCount, size_t activePage, size_t pageSize)
	{
		spdlog::info(
			"UploadInstance {}: pages={}, activePage={}, reservedBytes={} MiB",
			reason,
			pageCount,
			activePage,
			(pageCount * pageSize) / (1024ull * 1024ull));
	}
}

UploadInstance::UploadInstance(uint8_t numFramesInFlight, size_t pageSize)
	: m_pageSize(pageSize)
	, m_numFramesInFlight(numFramesInFlight)
{
	m_pages.push_back({ Buffer::CreateShared(rhi::HeapType::Upload, m_pageSize, false), 0 });
	TagUploadInstancePage(m_pages.back().buffer, 0);
	m_frameStart.assign(m_numFramesInFlight, 0);
	LogUploadInstancePages("initialize", m_pages.size(), m_activePage, m_pageSize);
}

// Page allocation

bool UploadInstance::AllocateUploadRegion(size_t size, size_t alignment,
                                          std::shared_ptr<Resource>& outUploadBuffer,
                                          size_t& outOffset)
{
	if (alignment == 0) alignment = 1;
	if (m_pages.empty()) {
		m_pages.push_back({ Buffer::CreateShared(rhi::HeapType::Upload, m_pageSize, false), 0 });
		TagUploadInstancePage(m_pages.back().buffer, 0);
		m_activePage = 0;
		LogUploadInstancePages("recreate-first-page", m_pages.size(), m_activePage, m_pageSize);
	}

	UploadPage* page = &m_pages[m_activePage];
	size_t alignedTail = AlignUpSizeT(page->tailOffset, alignment);

	if (alignedTail + size > page->buffer->GetSize()) {
		++m_activePage;
		if (m_activePage >= m_pages.size()) {
			size_t allocSize = (std::max)(m_pageSize, size);
			m_pages.push_back({ Buffer::CreateShared(rhi::HeapType::Upload, allocSize, false), 0 });
			TagUploadInstancePage(m_pages.back().buffer, m_pages.size() - 1);
			LogUploadInstancePages("grow", m_pages.size(), m_activePage, m_pageSize);
		}
		page = &m_pages[m_activePage];
		page->tailOffset = 0;
		alignedTail = AlignUpSizeT(page->tailOffset, alignment);

		if (alignedTail + size > page->buffer->GetSize()) {
			size_t allocSize = (std::max)(m_pageSize, size);
			m_pages.push_back({ Buffer::CreateShared(rhi::HeapType::Upload, allocSize, false), 0 });
			TagUploadInstancePage(m_pages.back().buffer, m_pages.size() - 1);
			LogUploadInstancePages("grow-dedicated", m_pages.size(), m_activePage, m_pageSize);
			m_activePage = m_pages.size() - 1;
			page = &m_pages[m_activePage];
			page->tailOffset = 0;
			alignedTail = AlignUpSizeT(page->tailOffset, alignment);
		}
	}

	outOffset = alignedTail;
	page->tailOffset = alignedTail + size;
	outUploadBuffer = page->buffer;
	return true;
}

// Coalescing

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
	for (uint8_t i = 0; i < next.stackSize && i < ResourceUpdate::MaxStack; ++i) last.stack[i] = next.stack[i];
#endif
	return true;
}

// Map/Unmap helpers

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

// Buffer upload

#if BUILD_TYPE == BUILD_TYPE_DEBUG
void UploadInstance::UploadData(const void* data, size_t size, UploadTarget target, size_t dstOffset,
                                const char* file, int line)
#else
void UploadInstance::UploadData(const void* data, size_t size, UploadTarget target, size_t dstOffset)
#endif
{
	if (size > m_pageSize) {
		size_t done = 0;
		size_t offset = dstOffset;
		while (done < size) {
			size_t chunk = (std::min)(size - done, m_pageSize);
#if BUILD_TYPE == BUILD_TYPE_DEBUG
			UploadData(reinterpret_cast<const uint8_t*>(data) + done, chunk, target, offset, file, line);
#else
			UploadData(reinterpret_cast<const uint8_t*>(data) + done, chunk, target, offset);
#endif
			done += chunk;
			offset += chunk;
		}
		return;
	}

	UploadPage* page = &m_pages[m_activePage];

	if (page->tailOffset + size > page->buffer->GetSize()) {
		++m_activePage;
		if (m_activePage >= m_pages.size()) {
			size_t allocSize = (std::max)(m_pageSize, size);
			m_pages.push_back({ Buffer::CreateShared(rhi::HeapType::Upload, allocSize, false), 0 });
			TagUploadInstancePage(m_pages.back().buffer, m_pages.size() - 1);
			LogUploadInstancePages("grow-buffer-upload", m_pages.size(), m_activePage, m_pageSize);
		}
		page = &m_pages[m_activePage];
		page->tailOffset = 0;
	}

	size_t uploadOffset = page->tailOffset;
	page->tailOffset += size;

	uint8_t* mapped = nullptr;
	page->buffer->GetAPIResource().Map(reinterpret_cast<void**>(&mapped), 0, size);
#if BUILD_TYPE == BUILD_TYPE_DEBUG
	if (!mapped) {
		__debugbreak();
		return;
	}
#endif
	memcpy(mapped + uploadOffset, data, size);
	page->buffer->GetAPIResource().Unmap(0, 0);

	ResourceUpdate update;
	update.size = size;
	update.resourceToUpdate = target;
	update.uploadBuffer = page->buffer;
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

	// Contiguous append coalescing against the most recent active update.
	for (int i = static_cast<int>(m_resourceUpdates.size()) - 1; i >= 0; --i) {
		auto& last = m_resourceUpdates[static_cast<size_t>(i)];
		if (!last.active) continue;
		if (TryCoalesceAppend(last, update)) return;
		break;
	}

	m_resourceUpdates.push_back(std::move(update));
}

// Texture upload

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
	AllocateUploadRegion(static_cast<size_t>(plan.totalSize), /*alignment*/512, uploadBuffer, uploadBaseOffset);

	uint8_t* mapped = nullptr;
	uploadBuffer->GetAPIResource().Map(reinterpret_cast<void**>(&mapped), 0,
	                                    uploadBaseOffset + static_cast<size_t>(plan.totalSize));
	rhi::helpers::WriteTextureUploadSubresources(plan, srcSpan, mapped, static_cast<uint64_t>(uploadBaseOffset));
	uploadBuffer->GetAPIResource().Unmap(0, 0);

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
		m_textureUpdates.push_back(std::move(update));
	}
}

// GPU copy emission

void UploadInstance::ProcessUploads(uint8_t frameIndex, rg::imm::ImmediateCommandList& commandList) {
	(void)frameIndex;

	for (auto& update : m_resourceUpdates) {
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
				m_ctx.registry->Resolve(update.resourceToUpdate.h),
				update.dataBufferOffset,
				update.uploadBuffer,
				update.uploadBufferOffset,
				update.size);
			break;
		}
	}

	for (auto& texUpdate : m_textureUpdates) {
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
				m_ctx.registry->Resolve(texUpdate.texture.h),
				texUpdate.mip,
				texUpdate.slice,
				texUpdate.footprint,
				texUpdate.x,
				texUpdate.y,
				texUpdate.z);
		}
	}

	m_resourceUpdates.clear();
	m_textureUpdates.clear();
}

// Page retirement

void UploadInstance::ProcessDeferredReleases(uint8_t frameIndex) {
	size_t retiringStart = m_frameStart[frameIndex];

	size_t minStart = retiringStart;
	for (uint8_t f = 0; f < m_numFramesInFlight; ++f) {
		if (f == frameIndex) continue;
		minStart = (std::min)(minStart, m_frameStart[f]);
	}

	if (minStart > 0) {
		size_t eraseCount = (std::min)(minStart, m_pages.size() - 1);
		if (eraseCount > 0) {
			m_pages.erase(m_pages.begin(), m_pages.begin() + static_cast<ptrdiff_t>(eraseCount));
			m_activePage -= eraseCount;
			for (auto& start : m_frameStart) {
				start = (start >= eraseCount ? start - eraseCount : 0);
			}
			for (size_t i = 0; i < m_pages.size(); ++i) {
				TagUploadInstancePage(m_pages[i].buffer, i);
			}
			LogUploadInstancePages("retire", m_pages.size(), m_activePage, m_pageSize);
		}
	}

	m_frameStart[frameIndex] = m_activePage;
}

// Query

bool UploadInstance::HasPendingWork() const {
	return !m_resourceUpdates.empty() || !m_textureUpdates.empty();
}

void UploadInstance::CollectPendingDestinations(std::vector<std::shared_ptr<Resource>>& out) const {
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

void UploadInstance::Cleanup() {
	m_pages.clear();
	m_resourceUpdates.clear();
	m_textureUpdates.clear();
	m_activePage = 0;
	for (auto& start : m_frameStart) start = 0;
}
