#include "Render/RenderGraph/FrameAliasArena.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include "AliasPacking.h"
#include "Managers/Singletons/DeletionManager.h"
#include "Managers/Singletons/DeviceManager.h"
#include "Render/RenderGraph/Aliasing/RenderGraphAliasingSubsystem.h"
#include "Resources/MemoryStatisticsComponents.h"
#include "Resources/PixelBuffer.h"

namespace org {

namespace {
// Pool ids of the arena's blocks: apart from the planner's pool ids.
constexpr uint64_t kFrameArenaPoolBase = 0xF4A3000000000000ull;
constexpr uint64_t kBlockGranularity = 64ull * 1024ull;
}

struct FrameAliasArena::Impl {
	struct Range {
		uint64_t id = 0;
		alias::SpannedRange range;
		std::string name;
	};
	struct Block {
		uint64_t poolID = 0, capacity = 0, alignment = 1;
		uint8_t resourceClass = 0;
		TrackedHandle allocation;
		std::shared_ptr<const AliasHeapGeneration> heap;
		std::vector<Range> ranges;
	};
	mutable std::mutex mutex;
	std::vector<Block> blocks;
	uint64_t nextBlock = 1, nextRange = 1;
};

std::shared_ptr<FrameAliasArena> FrameAliasArena::Create() {
	auto arena = std::make_shared<FrameAliasArena>();
	arena->m_impl = std::make_shared<Impl>();
	arena->m_self = arena;
	return arena;
}

FrameAliasArena::Reservation::~Reservation() {
	if (const auto arena = m_arena.lock()) arena->Release(m_id);
}

std::shared_ptr<FrameAliasArena::Reservation> FrameAliasArena::Reserve(uint8_t resourceClass, AliasSpan span, uint32_t lane, uint64_t size,
	uint64_t alignment, std::string name) {
	if (!size) throw std::invalid_argument("A frame alias reservation needs bytes");
	alignment = (std::max<uint64_t>)(alignment, 1);
	auto& impl = *m_impl;
	std::lock_guard lock(impl.mutex);
	Impl::Block* chosen = nullptr;
	uint64_t offset = UINT64_MAX;
	// The first block of the class where the range fits beside the ranges it may not share bytes with.
	for (auto& block : impl.blocks) {
		if (block.resourceClass != resourceClass || block.alignment < alignment) continue;
		std::vector<alias::SpannedRange> occupied;
		occupied.reserve(block.ranges.size());
		for (const auto& range : block.ranges) occupied.push_back(range.range);
		offset = alias::PlaceSpanned(occupied, block.capacity, size, alignment, span.first, span.last, lane);
		if (offset != UINT64_MAX) { chosen = &block; break; }
	}
	if (!chosen) {
		// A new block of the range's size. It never grows: reservations do not move.
		Impl::Block block;
		block.poolID = kFrameArenaPoolBase | impl.nextBlock++;
		block.resourceClass = resourceClass;
		block.alignment = (std::max)(alignment, kBlockGranularity);
		block.capacity = (size + kBlockGranularity - 1) / kBlockGranularity * kBlockGranularity;
		rhi::ma::AllocationDesc allocDesc{};
		allocDesc.heapType = rhi::HeapType::DeviceLocal;
		allocDesc.flags = rhi::ma::AllocationFlagCanAlias;
		rhi::ResourceAllocationInfo allocInfo{};
		allocInfo.alignment = block.alignment;
		allocInfo.sizeInBytes = block.capacity;
		AllocationTrackDesc trackDesc(0);
		trackDesc.attach
			.Set<MemoryStatisticsComponents::ResourceName>({"RenderGraph Frame Alias Arena"})
			.Set<MemoryStatisticsComponents::ResourceType>({rhi::ResourceType::Unknown})
			.Set<MemoryStatisticsComponents::ResourceUsage>({"RenderGraph alias pools"})
			.Set<MemoryStatisticsComponents::AliasingPool>({block.poolID});
		if (!rhi::IsOk(DeviceManager::GetInstance().AllocateMemoryTracked(allocDesc, allocInfo, block.allocation, trackDesc)))
			throw std::runtime_error("Failed to allocate frame alias arena memory for '" + name + "'");
		// Frame targets and per-frame working memory: over the OS budget, the costliest to leave in system memory.
		DeviceManager::GetInstance().SetResidencyPriority(block.allocation, rhi::ResidencyPriorityHigh);
		block.heap = AliasHeapGeneration::Capture(block.allocation, 1);
		spdlog::info("Frame alias arena: block {:#x} allocated: class={} capacity={} for '{}'", block.poolID, resourceClass, block.capacity, name);
		impl.blocks.push_back(std::move(block));
		chosen = &impl.blocks.back();
		offset = 0;
	}
	auto reservation = std::shared_ptr<Reservation>(new Reservation());
	reservation->m_arena = m_self;
	reservation->m_heap = chosen->heap;
	reservation->m_offset = offset;
	reservation->m_size = size;
	reservation->m_poolID = chosen->poolID;
	reservation->m_id = impl.nextRange++;
	chosen->ranges.push_back({reservation->m_id, {offset, offset + size, span.first, span.last, lane}, name});
	spdlog::info("Frame alias arena: '{}' reserved in block {:#x} at [{}, {}) for positions {}-{} lane {}", name, chosen->poolID, offset,
		offset + size, span.first, span.last, lane);
	return reservation;
}

void FrameAliasArena::Release(uint64_t id) noexcept {
	auto& impl = *m_impl;
	std::lock_guard lock(impl.mutex);
	for (auto block = impl.blocks.begin(); block != impl.blocks.end(); ++block) {
		const auto found = std::find_if(block->ranges.begin(), block->ranges.end(), [&](const auto& range) { return range.id == id; });
		if (found == block->ranges.end()) continue;
		block->ranges.erase(found);
		if (block->ranges.empty()) {
			// The resources placed in it keep its memory alive through their leases until they go.
			spdlog::info("Frame alias arena: block {:#x} freed (capacity {})", block->poolID, block->capacity);
			try {
				DeletionManager::GetInstance().MarkForDelete(std::move(block->allocation), "RenderGraph frame alias arena block");
			} catch (...) {
			}
			impl.blocks.erase(block);
		}
		return;
	}
}

std::shared_ptr<const void> FrameAliasArena::ReserveTextures(AliasSpan span, std::span<const std::shared_ptr<PixelBuffer>> textures, std::string name) {
	auto device = DeviceManager::GetInstance().GetDevice();
	struct Member {
		PixelBuffer* texture = nullptr;
		uint64_t size = 0, alignment = 1, offset = 0;
	};
	// Render targets and depth apart from other textures, as the planner's pools keep them.
	std::map<uint8_t, std::vector<Member>> byClass;
	for (const auto& texture : textures) {
		if (!texture) continue;
		if (texture->IsMaterialized()) throw std::logic_error("A texture placed in the frame alias arena must not be materialized: " + texture->GetName());
		const auto& desc = texture->GetDescription();
		const auto resourceDesc = alias::AliasTextureResourceDesc(desc);
		rhi::ResourceAllocationInfo info{};
		device.GetResourceAllocationInfo(&resourceDesc, 1, &info);
		byClass[(desc.hasRTV || desc.hasDSV) ? 2u : 3u].push_back({texture.get(), info.sizeInBytes, (std::max<uint64_t>)(info.alignment, 1)});
	}
	auto kept = std::make_shared<std::vector<std::shared_ptr<Reservation>>>();
	for (auto& [resourceClass, members] : byClass) {
		uint64_t end = 0, alignment = 1;
		for (auto& member : members) {
			member.offset = (end + member.alignment - 1) / member.alignment * member.alignment;
			end = member.offset + member.size;
			alignment = (std::max)(alignment, member.alignment);
		}
		auto reservation = Reserve(resourceClass, span, 0, end, alignment, name);
		for (const auto& member : members) {
			PixelBuffer::MaterializeOptions options{};
			options.aliasPlacement = TextureAliasPlacement{reservation->Heap(), reservation->Offset() + member.offset, reservation->PoolID()};
			member.texture->Materialize(&options);
		}
		kept->push_back(std::move(reservation));
	}
	return kept;
}

std::vector<FrameAliasArena::BlockInfo> FrameAliasArena::Blocks() const {
	auto& impl = *m_impl;
	std::lock_guard lock(impl.mutex);
	std::vector<BlockInfo> out;
	for (const auto& block : impl.blocks) {
		BlockInfo info{block.poolID, block.capacity, block.resourceClass, {}};
		for (const auto& range : block.ranges)
			info.ranges.push_back({range.name, range.range.begin, range.range.end, {range.range.first, range.range.last}, range.range.lane});
		out.push_back(std::move(info));
	}
	return out;
}

uint64_t FrameAliasArena::Bytes() const {
	auto& impl = *m_impl;
	std::lock_guard lock(impl.mutex);
	uint64_t bytes = 0;
	for (const auto& block : impl.blocks) bytes += block.capacity;
	return bytes;
}

}  // namespace org
