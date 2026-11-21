#pragma once

#include <cstdint>
#include <span>
#include <type_traits>

#include <rhi.h>

#include "Render/LatchBlock.h"
#include "Render/RunRef.h"

// The C++ side of shaders that read their resources through handles in a push block (the shader side is
// ORGModuleServices' OrgResources.hlsli).
//
// A shader lists its resources once. Under the bindless build each entry is a handle in the push block, and the shader
// fetches the resource from the descriptor heap by that handle. A pass mirrors the block in a struct, in the shader's
// list order, and pushes it at record time:
//
//     struct BlurBlock { org::shader_bindings::Handle input, output, linearSampler; };
//     constexpr auto words = org::shader_bindings::Words<BlurBlock>();
//     BlurBlock block{ recording.Index(input), recording.Index(output), sampler.Index() };
//     list.PushConstants(stages, 0, binding, 0, words, org::shader_bindings::Data(block).data());
//
// Per-execution values go in a LATCH entry, a region of an org::LatchBlock, so the recording does not depend on them and
// can be prepared ahead.
//
// A shader that reads more resources than the push block holds keeps their handles in a handle table: one Handle per
// resource, in a buffer the pass writes, read through one LATCH entry. The C++ table lists the same names in the same
// order as the shader's table struct.
namespace org::shader_bindings {

// The largest push block in 32-bit words. 128 bytes is Vulkan's guaranteed minimum, and D3D12 root constants hold 64.
inline constexpr std::uint32_t kMaxPushWords = 32;

// A descriptor-heap index, read by a shader's OrgHandle_<name>.
using Handle = std::uint32_t;

// The block's size in words. A compile error when the block is not whole words or is above kMaxPushWords.
template <class Block>
constexpr std::uint32_t Words() {
	static_assert(std::is_trivially_copyable_v<Block>, "a push block is plain data");
	static_assert(sizeof(Block) % sizeof(std::uint32_t) == 0, "a push block is whole 32-bit words");
	static_assert(sizeof(Block) <= kMaxPushWords * sizeof(std::uint32_t), "a push block is at most 128 bytes: move constants to a CBUFFER entry");
	return sizeof(Block) / sizeof(std::uint32_t);
}

// A LATCH entry: the latch block's raw view, then the byte offset of the struct in it.
struct Latch {
	Handle handle = 0;
	std::uint32_t offset = 0;
};

// A LATCH entry for a latch block's region of a run, `offset` bytes in. The offset must be a multiple of 4.
inline Latch LatchOf(const LatchBlock& block, const RunRef& run, std::uint32_t offset = 0) {
	const auto region = block.At(run);
	return { region.srvIndex, static_cast<std::uint32_t>(region.offset) + offset };
}

// The block as the words a push takes.
template <class Block>
std::span<const std::uint32_t> Data(const Block& block) {
	return { reinterpret_cast<const std::uint32_t*>(&block), Words<Block>() };
}

// Samplers most passes use. Linear ones filter mips linearly too; point ones are point in every filter.
enum class SamplerPreset : std::uint8_t {
	LinearClamp,       // rhi::SamplerDesc{}
	LinearWrap,
	LinearMirror,      // mirrored on all three axes
	LinearBorder,      // transparent black border
	PointClamp,
	PointBorderWhite,  // opaque white border
	kCount
};

// The description a preset stands for.
inline rhi::SamplerDesc SamplerDescOf(SamplerPreset preset) {
	rhi::SamplerDesc desc{};
	const auto address = [&desc](rhi::AddressMode mode) { desc.addressU = desc.addressV = desc.addressW = mode; };
	const auto point = [&desc] {
		desc.minFilter = desc.magFilter = rhi::Filter::Nearest;
		desc.mipFilter = rhi::MipFilter::Nearest;
	};
	switch (preset) {
	case SamplerPreset::LinearClamp:
		break;
	case SamplerPreset::LinearWrap:
		address(rhi::AddressMode::Wrap);
		break;
	case SamplerPreset::LinearMirror:
		address(rhi::AddressMode::Mirror);
		break;
	case SamplerPreset::LinearBorder:
		address(rhi::AddressMode::Border);
		desc.borderPreset = rhi::BorderPreset::TransparentBlack;
		break;
	case SamplerPreset::PointClamp:
		point();
		break;
	case SamplerPreset::PointBorderWhite:
		point();
		address(rhi::AddressMode::Border);
		desc.borderPreset = rhi::BorderPreset::OpaqueWhite;
		break;
	default:
		break;
	}
	return desc;
}

}  // namespace org::shader_bindings
