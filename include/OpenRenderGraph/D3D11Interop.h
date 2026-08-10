#pragma once

#include <d3d11_4.h>
#include <dxgi1_6.h>
#include <winrt/base.h>
#include <rhi.h>
#include <memory>

namespace org::interop {

class D3D11Interop
{
public:
	struct DeviceBundle
	{
		winrt::com_ptr<ID3D11Device5> device11;
		winrt::com_ptr<ID3D11DeviceContext4> context11;
		winrt::com_ptr<IDXGIAdapter> adapter;
		rhi::DevicePtr graphDevice;
	};

	static bool CreateDeviceBundle(ID3D11Device* device, ID3D11DeviceContext* context,
		uint32_t framesInFlight, bool enableDebug, DeviceBundle& output) noexcept;
	struct SharedTexture
	{
		winrt::com_ptr<ID3D11Texture2D> d3d11;
		std::shared_ptr<void> graphBacking;
		winrt::com_ptr<ID3D11ShaderResourceView> srv11;
		winrt::com_ptr<ID3D11UnorderedAccessView> uav11;
		D3D11_TEXTURE2D_DESC description{};

		explicit operator bool() const noexcept { return d3d11 && graphBacking; }
	};

	D3D11Interop(ID3D11Device5* device11, rhi::Device device);
	bool OpenTimeline(rhi::Timeline timeline, ID3D11Fence** output) const noexcept;
	bool ProbeGraphOwnedSharing() const noexcept;

	bool CreateSharedTexture(
		const D3D11_TEXTURE2D_DESC& description,
		bool createSRV,
		bool createUAV,
		SharedTexture& output) const noexcept;
	// Preferred graph -> D3D11 handoff. The RHI backend owns the allocation and
	// exports it through the platform bridge; D3D11 only opens views.
	// Callers must use a format supported by SharedResourceCompatibilityTier.
	// Do not re-share the opened D3D11 object or combine legacy KMT handles with
	// this path. Queue ownership is synchronized exclusively by the host fences.
	bool CreateGraphOwnedSharedTexture(
		uint32_t width,
		uint32_t height,
		rhi::Format format,
		rhi::ResourceFlags flags,
		bool createD3D11SRV,
		SharedTexture& output) const noexcept;
	// Full-description variant for host interop resources whose mip chain/array
	// shape must survive sharing. Material assets must not use the mirror path;
	// their future registry requires direct sharing or graph ownership.
	// Usage/CPU/bind/misc fields are normalized by the graph owner; dimensions,
	// format, mip levels, array size and sample description are authoritative.
	bool CreateGraphOwnedSharedTexture(
		const D3D11_TEXTURE2D_DESC& description,
		rhi::ResourceFlags flags,
		bool createD3D11SRV,
		SharedTexture& output) const noexcept;
	bool CreateGraphOwnedSharedTextureArray(
		uint32_t width,
		uint32_t height,
		uint16_t arraySize,
		rhi::Format format,
		rhi::ResourceFlags flags,
		bool createD3D11SRV,
		SharedTexture& output) const noexcept;

	// Preferred D3D11 -> graph read-only path when the producer allocation was
	// created with D3D11_RESOURCE_MISC_SHARED_NTHANDLE. No texture copy is
	// required: the host readiness fence orders all preceding D3D11 writes before
	// the first ORG transition/read of this imported allocation.
	bool ImportReadOnlySharedTexture(ID3D11Texture2D* source, bool createD3D11SRV, SharedTexture& imported) const noexcept;
	bool ImportGraphResource(const SharedTexture& source, rhi::ResourcePtr& output) const noexcept;
	bool EnsureReadOnlyMirror(ID3D11Texture2D* source, SharedTexture& mirror) const noexcept;
	bool CopyToMirror(ID3D11DeviceContext* context, ID3D11Texture2D* source, SharedTexture& mirror) const noexcept;

private:
	winrt::com_ptr<ID3D11Device5> device11;
	rhi::Device device{};
};

} // namespace org::interop
