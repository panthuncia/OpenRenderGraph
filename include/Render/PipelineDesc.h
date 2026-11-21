#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <rhi.h>

#include "Render/PipelineState.h"
#include "Render/QueueKind.h"

// Pipelines described by value, so one description can be built on any device (docs/pipelines.md). The host builds them.
namespace org {

struct PipelineShaderDefine {
	std::wstring name;
	std::wstring value;
};

// A shader of a pipeline. An empty target takes the stage's default (cs_6_6, vs_6_6, ...).
struct PipelineShaderSource {
	std::wstring path;
	std::wstring entry;
	std::wstring target;
};

enum class PipelineLayoutKind : uint8_t { StandardCompute, StandardGraphics, Custom };

// A layout the host creates on each device a pipeline needs it, shared by every pipeline with the same content.
struct PipelineLayoutSpec {
	std::vector<rhi::LayoutBindingRange> ranges;
	std::vector<rhi::PushConstantRangeDesc> pushConstants;
	std::vector<rhi::StaticSamplerDesc> staticSamplers;
	rhi::PipelineLayoutFlags flags = rhi::PF_None;
	std::string name;
};

struct PipelineLayoutRef {
	PipelineLayoutKind kind = PipelineLayoutKind::StandardCompute;
	std::optional<PipelineLayoutSpec> custom;

	static PipelineLayoutRef StandardCompute() { return {PipelineLayoutKind::StandardCompute, std::nullopt}; }
	static PipelineLayoutRef StandardGraphics() { return {PipelineLayoutKind::StandardGraphics, std::nullopt}; }
	static PipelineLayoutRef Custom(PipelineLayoutSpec spec) { return {PipelineLayoutKind::Custom, std::move(spec)}; }
	/// @brief A custom layout of one block of root constants, register b0 space0, at root parameter 0. Shaders reach their
	/// resources through the descriptor heap, with the indices in those constants.
	static PipelineLayoutRef RootConstants(uint32_t count, rhi::ShaderStage visibility, std::string name) {
		PipelineLayoutSpec spec;
		spec.pushConstants.push_back({.visibility = visibility, .num32BitValues = count});
		spec.name = std::move(name);
		return Custom(std::move(spec));
	}
};

struct ComputePipelineDesc {
	PipelineShaderSource shader;
	std::string name;
	std::vector<PipelineShaderDefine> defines;
	PipelineLayoutRef layout = PipelineLayoutRef::StandardCompute();
};

/// @brief One vertex attribute of a pipeline's input layout. It owns its semantic name, unlike rhi::InputAttributeDesc.
struct PipelineInputAttribute {
	uint32_t binding = 0;                  ///< The vertex buffer slot it reads from.
	uint32_t offset = rhi::APPEND_ALIGNED; ///< Byte offset in the vertex. APPEND_ALIGNED places it after the previous one.
	rhi::Format format = rhi::Format::Unknown;
	std::string semanticName;              ///< D3D12 matches the vertex shader's inputs by semantic, such as "TEXCOORD".
	uint32_t semanticIndex = 0;
	/// Vulkan matches inputs by location. Unset locations count up in declaration order. That is the order DXC gives the
	/// vertex shader's inputs.
	uint32_t location = 0xFFFFFFFFu;
};

/// @brief The vertex buffers a pipeline reads, and how their bytes map to vertex shader inputs.
///
/// A pipeline with an input layout draws from vertex buffers bound with CommandList::SetVertexBuffers. Without one, the
/// vertex shader pulls its own vertices. On D3D12 the pipeline's layout must allow the input assembler. A custom layout
/// needs rhi::PF_AllowInputAssembler. A standard layout is the host's choice.
struct PipelineInputLayout {
	std::vector<rhi::InputBindingDesc> bindings;
	std::vector<PipelineInputAttribute> attributes;
};

// An absent state is not given to the device at all, which leaves that state to the backend's defaults. Topology is the
// exception for vertex pipelines: backends disagree about an omitted one, so it is always given (TriangleList unless set).
struct GraphicsPipelineDesc {
	std::optional<PipelineShaderSource> vertex;
	std::optional<PipelineShaderSource> amplification;
	std::optional<PipelineShaderSource> mesh;
	std::optional<PipelineShaderSource> pixel;
	std::string name;
	std::vector<PipelineShaderDefine> defines;
	PipelineLayoutRef layout = PipelineLayoutRef::StandardGraphics();
	std::optional<rhi::RasterState> raster;
	std::optional<rhi::BlendState> blend;
	std::optional<rhi::DepthStencilState> depth;
	std::optional<rhi::RenderTargets> renderTargets;
	std::optional<rhi::Format> depthFormat;
	std::optional<rhi::SampleDesc> sample;
	std::optional<rhi::PrimitiveTopology> topology;
	/// Vertex pipelines only. When absent, the vertex shader pulls its own vertices.
	std::optional<PipelineInputLayout> inputLayout;
};

using PipelineDesc = std::variant<ComputePipelineDesc, GraphicsPipelineDesc>;

// A handle to a command signature's creations, one slot per device, as PipelineState is to a pipeline's builds.
class CommandSignatureState {
public:
	CommandSignatureState() : m_slots(std::make_shared<Slots>()) {}
	// A signature the host made, for the primary device only (tests, and signatures of paths that do not run elsewhere).
	explicit CommandSignatureState(std::shared_ptr<const rhi::CommandSignaturePtr> primary) : CommandSignatureState() {
		Attach(BackendInstanceId::Primary, std::move(primary));
	}
	std::shared_ptr<const rhi::CommandSignaturePtr> Get(BackendInstanceId device) const {
		const size_t index = static_cast<uint8_t>(device);
		return m_slots && index < kMaxPipelineDevices ? (*m_slots)[index].load(std::memory_order_acquire) : nullptr;
	}
	void Attach(BackendInstanceId device, std::shared_ptr<const rhi::CommandSignaturePtr> signature) const {
		const size_t index = static_cast<uint8_t>(device);
		if (index >= kMaxPipelineDevices) throw std::out_of_range("CommandSignatureState: device instance out of range");
		(*m_slots)[index].store(std::move(signature), std::memory_order_release);
	}

private:
	using Slots = std::array<std::atomic<std::shared_ptr<const rhi::CommandSignaturePtr>>, kMaxPipelineDevices>;
	std::shared_ptr<Slots> m_slots;
};

} // namespace org
