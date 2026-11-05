#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <rhi.h>

#include "Render/PipelineState.h"
#include "Render/QueueKind.h"

// Pipelines described by value, so one description can be built on any device (docs/pipelines.md). A pass passes one to
// RenderPassBuilder::Pipeline in Declare; PipelineLibrary builds it for the device the pass lands on.
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

// A layout the library creates on each device a pipeline needs it, shared by every pipeline with the same content.
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
};

struct ComputePipelineDesc {
	PipelineShaderSource shader;
	std::string name;
	std::vector<PipelineShaderDefine> defines;
	PipelineLayoutRef layout = PipelineLayoutRef::StandardCompute();
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
};

using PipelineDesc = std::variant<ComputePipelineDesc, GraphicsPipelineDesc>;

// A command signature by description: it belongs to a device and names a layout of that device.
struct CommandSignatureDesc {
	std::vector<rhi::IndirectArg> args;
	uint32_t byteStride = 0;
	PipelineLayoutRef layout = PipelineLayoutRef::StandardCompute();
	bool unorderedSequences = false;
	bool explicitPreprocess = false;
	std::string name;
};

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

// What the library asks the host to compile: the shaders of one pipeline for one device. Defines apply to every stage.
struct PipelineCompileRequest {
	struct Stage {
		rhi::ShaderStage stage{};
		PipelineShaderSource source;
	};
	std::vector<Stage> stages;
	std::vector<PipelineShaderDefine> defines;
	std::string name;
	DeviceInstanceId device = DeviceInstanceId::Primary;
	rhi::Backend backend = rhi::Backend::Null;
	// A reload: the caller knows the source changed, so a cached artifact must not answer it.
	bool bypassCache = false;
};

// The compiled stages, parallel to the request's. A non-empty error means the compile failed; the rest is then unused.
struct CompiledPipeline {
	std::vector<rhi::ShaderBinary> stages;
	std::shared_ptr<const void> codeOwner; // Keeps the stages' code alive until the pipeline is created.
	PipelineResources resources;
	uint64_t resourceIDsHash = 0;
	uint64_t sourceHash = 0;
	uint64_t bytecodeHash = 0;
	std::string error;
};

struct PipelineLayoutBinding {
	rhi::PipelineLayoutHandle layout{};
	std::shared_ptr<const void> owner; // Must outlive every pipeline created with the layout.
};

// The host's side of pipeline construction: ORG has no shader compiler, and a pipeline's descriptor slots come from the
// host's shader conventions. Called on task service threads.
class PipelineCompiler {
public:
	using Completion = std::function<void(CompiledPipeline)>;
	virtual ~PipelineCompiler() = default;
	// Compiles for request.backend. Completion runs exactly once, on any thread; it may run before Compile returns. An
	// exception thrown before it runs counts as a failed compile.
	virtual void Compile(const PipelineCompileRequest& request, Completion completion) = 0;
	// The host's layout of that kind on a device. Throws when there is none.
	virtual PipelineLayoutBinding StandardLayout(PipelineLayoutKind kind, DeviceInstanceId device) = 0;
};

} // namespace org
