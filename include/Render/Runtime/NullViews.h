#pragma once

#include <cstdint>

#include <rhi.h>

namespace org::runtime {

// The shader resource views a null descriptor can stand in for, one per dimension a shader declares.
// Reads through a null descriptor return zero.
enum class NullViewKind : uint32_t {
	Texture2D,
	Texture2DArray,
	Texture3D,
	TextureCube,
	TextureCubeArray,
	TypedBuffer,       // a Buffer<uint>
	StructuredBuffer,  // a StructuredBuffer with a 4-byte stride
	RawBuffer,         // a ByteAddressBuffer
	Count,
};

// The view description of a null descriptor of `kind`, written with no resource (rhi::Device::CreateShaderResourceView
// with an invalid handle).
// - D3D12 makes a null view of the description's dimension. A texture needs a format, so every texture kind gets
//   R8G8B8A8_UNorm with one mip and one slice.
// - Vulkan needs VK_EXT_robustness2's nullDescriptor. Its null descriptor depends only on the descriptor type: a sampled
//   image for textures, a texel buffer for a typed buffer, and a storage buffer otherwise.
inline rhi::SrvDesc NullSrvDesc(NullViewKind kind) noexcept {
	rhi::SrvDesc srv{};
	switch (kind) {
	case NullViewKind::Texture2D:
		srv.dimension = rhi::SrvDim::Texture2D;
		srv.formatOverride = rhi::Format::R8G8B8A8_UNorm;
		srv.tex2D.mipLevels = 1;
		break;
	case NullViewKind::Texture2DArray:
		srv.dimension = rhi::SrvDim::Texture2DArray;
		srv.formatOverride = rhi::Format::R8G8B8A8_UNorm;
		srv.tex2DArray.mipLevels = 1;
		srv.tex2DArray.arraySize = 1;
		break;
	case NullViewKind::Texture3D:
		srv.dimension = rhi::SrvDim::Texture3D;
		srv.formatOverride = rhi::Format::R8G8B8A8_UNorm;
		srv.tex3D.mipLevels = 1;
		break;
	case NullViewKind::TextureCube:
		srv.dimension = rhi::SrvDim::TextureCube;
		srv.formatOverride = rhi::Format::R8G8B8A8_UNorm;
		srv.cube.mipLevels = 1;
		break;
	case NullViewKind::TextureCubeArray:
		srv.dimension = rhi::SrvDim::TextureCubeArray;
		srv.formatOverride = rhi::Format::R8G8B8A8_UNorm;
		srv.cubeArray.mipLevels = 1;
		srv.cubeArray.numCubes = 1;
		break;
	case NullViewKind::TypedBuffer:
		srv.dimension = rhi::SrvDim::Buffer;
		srv.buffer.kind = rhi::BufferViewKind::Typed;
		srv.formatOverride = rhi::Format::R32_UInt;
		break;
	case NullViewKind::StructuredBuffer:
		srv.dimension = rhi::SrvDim::Buffer;
		srv.buffer.kind = rhi::BufferViewKind::Structured;
		srv.buffer.structureByteStride = sizeof(uint32_t);
		break;
	case NullViewKind::RawBuffer:
		srv.dimension = rhi::SrvDim::Buffer;
		srv.buffer.kind = rhi::BufferViewKind::Raw;
		break;
	case NullViewKind::Count:
		break;  // Count names no view, so its dimension stays Undefined
	}
	return srv;
}

} // namespace org::runtime
