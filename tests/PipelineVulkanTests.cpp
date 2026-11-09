// Pipelines on the Vulkan device of PersistentVulkanHostTests ("pipelines"):
//   - an input layout is part of a description's identity, reaches the compiler, and builds a vertex pipeline;
//   - PipelinePermutations selects pipelines at Prepare time: a pass skips its dispatch while the pipeline builds, and
//     runs it once the build is ready;
//   - GenerateMipsPass builds every mip of 2D textures, texture arrays and cube maps, odd sizes included.
// The compiler loads SPIR-V that the build compiled from tests/shaders and ORG's shaders folder.

#include "rhi.h"
#include "TestSupport.h"

#include <OpenRenderGraph/OpenRenderGraph.h>
#include "OpenRenderGraph/GraphHost.h"
#include "OpenRenderGraph/Runtime.h"
#include "Render/PipelineLibrary.h"
#include "Render/PipelinePermutations.h"
#include "Render/RenderGraph/RenderGraph.h"
#include "Render/Runtime/ThreadPoolTaskService.h"
#include "RenderPasses/Base/TypedRenderGraphPass.h"
#include "RenderPasses/GenerateMipsPass.h"
#include "Resources/Buffers/Buffer.h"
#include "Resources/PixelBuffer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifndef ORG_TEST_PIPELINE_SPV_DIR
#error "ORG_TEST_PIPELINE_SPV_DIR must name the folder of the test pipelines' SPIR-V"
#endif

namespace {

bool HasDefine(const org::PipelineCompileRequest& request, const wchar_t* name, const wchar_t* value = nullptr) {
	return std::ranges::any_of(request.defines, [&](const auto& d) { return d.name == name && (!value || d.value == value); });
}

// Serves precompiled SPIR-V by shader path, entry and defines. A pipeline whose name starts with "gated" completes only
// when the test calls Release, so a test can see a build that has not ended. A path of "fail" fails.
class SpirvCompiler final : public org::PipelineCompiler {
public:
	explicit SpirvCompiler(rhi::Device device) : m_device(device) {}

	void Compile(const org::PipelineCompileRequest& request, Completion completion) override {
		++compiles;
		if (request.inputLayout) lastInputAttributes = static_cast<int>(request.inputLayout->attributes.size());
		org::CompiledPipeline compiled;
		auto code = std::make_shared<std::vector<std::vector<char>>>();
		for (const auto& stage : request.stages) {
			const auto file = FileFor(request, stage);
			if (file.empty()) {
				compiled.error = "no SPIR-V for this shader";
				break;
			}
			std::ifstream in(std::string(ORG_TEST_PIPELINE_SPV_DIR) + "/" + file + ".spv", std::ios::binary);
			code->emplace_back(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
			if (code->back().empty()) {
				compiled.error = "missing " + file + ".spv";
				break;
			}
		}
		if (compiled.error.empty()) {
			for (const auto& stage : *code) compiled.stages.push_back({stage.data(), static_cast<uint32_t>(stage.size())});
			compiled.codeOwner = code;
		}
		if (request.name.starts_with("gated")) {
			std::scoped_lock lock(m_mutex);
			m_held.push_back([completion = std::move(completion), compiled = std::move(compiled)]() mutable {
				completion(std::move(compiled));
			});
			return;
		}
		completion(std::move(compiled));
	}

	org::PipelineLayoutBinding StandardLayout(org::PipelineLayoutKind, org::DeviceInstanceId) override {
		throw std::runtime_error("the tests use custom layouts");
	}

	// Completes the held compiles.
	void Release() {
		std::vector<std::function<void()>> held;
		{
			std::scoped_lock lock(m_mutex);
			held.swap(m_held);
		}
		for (auto& complete : held) complete();
	}

	std::atomic<int> compiles{0};
	std::atomic<int> lastInputAttributes{-1};

private:
	static std::string FileFor(const org::PipelineCompileRequest& request, const org::PipelineCompileRequest::Stage& stage) {
		const auto& path = stage.source.path;
		if (path == std::wstring(org::kBuiltinShaderPrefix) + L"GenerateMips.hlsl")
			return HasDefine(request, L"ORG_GENERATE_MIPS_ARRAY") ? "GenerateMipsArray" : "GenerateMips2D";
		if (path == std::wstring(org::kBuiltinShaderPrefix) + L"ResolveMultisample.hlsl") return "ResolveMultisample";
		if (path == L"test/InputLayout.hlsl") return stage.stage == rhi::ShaderStage::Vertex ? "InputLayoutVS" : "InputLayoutPS";
		if (path == L"test/PermutationWrite.hlsl")
			return HasDefine(request, L"WRITE_BIAS", L"100") ? "PermutationWrite100" : "PermutationWrite0";
		return {};
	}

	rhi::Device m_device;
	std::mutex m_mutex;
	std::vector<std::function<void()>> m_held;
};

org::PipelineInputLayout VertexLayout() {
	org::PipelineInputLayout layout;
	layout.bindings.push_back({.binding = 0});
	layout.attributes.push_back({.format = rhi::Format::R32G32B32_Float, .semanticName = "POSITION"});
	layout.attributes.push_back({.format = rhi::Format::R32G32B32A32_Float, .semanticName = "COLOR"});
	return layout;
}

org::GraphicsPipelineDesc VertexPipeline(bool allowInputAssembler) {
	org::GraphicsPipelineDesc desc;
	desc.vertex = org::PipelineShaderSource{L"test/InputLayout.hlsl", L"VSMain", {}};
	desc.pixel = org::PipelineShaderSource{L"test/InputLayout.hlsl", L"PSMain", {}};
	desc.name = allowInputAssembler ? "InputLayout" : "InputLayoutWithoutAssembler";
	org::PipelineLayoutSpec spec;
	spec.flags = allowInputAssembler ? rhi::PF_AllowInputAssembler : rhi::PF_None;
	spec.name = "InputLayoutTest";
	desc.layout = org::PipelineLayoutRef::Custom(std::move(spec));
	desc.renderTargets = rhi::RenderTargets{.count = 1, .formats = {rhi::Format::R8G8B8A8_UNorm}};
	desc.inputLayout = VertexLayout();
	return desc;
}

org::ComputePipelineDesc WritePipeline(uint32_t key) {
	org::ComputePipelineDesc desc;
	desc.shader = {L"test/PermutationWrite.hlsl", L"main", {}};
	desc.name = "gated.PermutationWrite" + std::to_string(key);
	desc.defines.push_back({L"WRITE_BIAS", key ? L"100" : L"0"});
	desc.layout = org::PipelineLayoutRef::RootConstants(3, rhi::ShaderStage::Compute, "PermutationWrite");
	return desc;
}

int TestInputLayoutIdentity() {
	auto a = VertexPipeline(true);
	auto b = a;
	CHECK(org::PipelineContentHash(a) == org::PipelineContentHash(b));
	b.inputLayout.reset();
	CHECK(org::PipelineContentHash(a) != org::PipelineContentHash(b));
	b = a;
	b.inputLayout->attributes[1].semanticIndex = 1;
	CHECK(org::PipelineContentHash(a) != org::PipelineContentHash(b));
	b = a;
	b.inputLayout->bindings[0].rate = rhi::InputRate::PerInstance;
	CHECK(org::PipelineContentHash(a) != org::PipelineContentHash(b));
	b = a;
	b.inputLayout->attributes[0].format = rhi::Format::R32G32_Float;
	CHECK(org::PipelineContentHash(a) != org::PipelineContentHash(b));
	return 0;
}

int TestInputLayoutBuild(org::PipelineLibrary& library, SpirvCompiler& compiler) {
	constexpr auto device = org::DeviceInstanceId::Primary;
	auto entry = library.Acquire(VertexPipeline(true));
	library.Request(entry, device);
	CHECK(library.Wait(entry, device) == org::PipelineBuildStatus::Ready);
	CHECK(entry->State().HasBackendPipeline(device));
	CHECK(compiler.lastInputAttributes == 2); // The compiler saw the layout.
	// D3D12 needs the input assembler flag on the layout, so the library asks for it on every backend.
	auto refused = library.Acquire(VertexPipeline(false));
	library.Request(refused, device);
	CHECK(library.Wait(refused, device) == org::PipelineBuildStatus::Failed);
	CHECK(refused->Error(device).find("PF_AllowInputAssembler") != std::string::npos);
	return 0;
}

constexpr uint32_t kCount = 64;
constexpr uint64_t kBytes = kCount * sizeof(uint32_t);
constexpr uint32_t kFill = 0xF111F111u;

struct CopyBindings { org::ResourceBindingToken source, destination; };
class CopyPass final : public org::TypedRenderGraphPass<CopyPass, org::EmptyPassFrameData, CopyBindings> {
public:
	CopyPass(std::shared_ptr<org::Resource> source, std::shared_ptr<org::Resource> destination)
		: m_source(std::move(source)), m_destination(std::move(destination)) {}
	CopyBindings Declare(org::PassBuilder& builder) { return {builder.CopySource(m_source), builder.CopyDestination(m_destination)}; }
	static void Record(const CopyBindings& bindings, org::PassRecordContext& recording) {
		recording.Commands().CopyBufferRegion(recording.Resolve(bindings.destination).GetHandle(), 0,
			recording.Resolve(bindings.source).GetHandle(), 0, kBytes);
	}
private:
	std::shared_ptr<org::Resource> m_source, m_destination;
};

// Selects its pipeline in Prepare and skips the dispatch while the pipeline builds.
struct WriteState {
	std::shared_ptr<org::PipelinePermutations> permutations;
	std::atomic<uint32_t> key{0};
	std::atomic<int> skipped{0}, selected{0};
	uint32_t value = 0x100;
};
struct WriteBindings { org::DeclaredViewToken output; };
struct WriteFrame { std::shared_ptr<const org::PipelineStatePayload> pipeline; uint32_t value = 0; };
class PermutationWritePass final : public org::TypedRenderGraphPass<PermutationWritePass, WriteFrame, WriteBindings> {
public:
	PermutationWritePass(std::shared_ptr<org::Resource> output, std::shared_ptr<WriteState> state)
		: m_output(std::move(output)), m_state(std::move(state)) {}
	WriteBindings Declare(org::PassBuilder& builder) {
		builder.PreferQueue(org::QueueKind::Graphics);
		if (!m_state->permutations)
			m_state->permutations = std::make_shared<org::PipelinePermutations>(builder.Pipelines(), "test.write");
		return {builder.UnorderedAccess(m_output).View()};
	}
	WriteFrame Prepare(const WriteBindings&, const org::PassPrepareContext& preparation) {
		const uint32_t key = m_state->key.load();
		const auto selection = m_state->permutations->Select(key, preparation.backendInstance, [&] { return WritePipeline(key); });
		++(selection ? m_state->selected : m_state->skipped);
		return {selection.payload, m_state->value};
	}
	static void Record(const WriteBindings& bindings, const WriteFrame& frame, org::PassRecordContext& recording) {
		if (!frame.pipeline) return; // Not built yet: this frame skips the work.
		recording.BindComputePipeline(frame.pipeline);
		const std::array<uint32_t, 3> constants{recording.Index(bindings.output), frame.value, kCount};
		recording.PushConstants(0, constants);
		recording.Commands().Dispatch(1, 1, 1);
	}
private:
	std::shared_ptr<org::Resource> m_output;
	std::shared_ptr<WriteState> m_state;
};

class WriteExtension final : public org::RenderGraph::IRenderGraphExtension {
public:
	WriteExtension(std::shared_ptr<org::Buffer> fill, std::shared_ptr<org::Buffer> output, std::shared_ptr<org::Buffer> readback,
		std::shared_ptr<WriteState> state)
		: m_fill(std::move(fill)), m_output(std::move(output)), m_readback(std::move(readback)), m_state(std::move(state)) {}
	void GatherStructuralPasses(org::RenderGraph&, std::vector<org::RenderGraph::ExternalPassDesc>& out) override {
		out.push_back(org::RenderGraph::ExternalPassDesc::Copy("write.fill", std::make_shared<CopyPass>(m_fill, m_output)));
		out.push_back(org::RenderGraph::ExternalPassDesc::Compute("write.select",
			std::make_shared<PermutationWritePass>(m_output, m_state)));
		out.push_back(org::RenderGraph::ExternalPassDesc::Copy("write.read", std::make_shared<CopyPass>(m_output, m_readback)));
	}
private:
	std::shared_ptr<org::Buffer> m_fill, m_output, m_readback;
	std::shared_ptr<WriteState> m_state;
};

// The value every word of the buffer holds, minus its index. 0 when they disagree.
uint32_t ReadWritten(org::Buffer& buffer) {
	void* mapped = nullptr;
	buffer.GetAPIResource().Map(&mapped, 0, kBytes);
	if (!mapped) return 0;
	const auto* words = static_cast<const uint32_t*>(mapped);
	uint32_t value = words[0];
	for (uint32_t i = 1; i < kCount; ++i) if (words[i] - i != value && words[i] != value) value = 0;
	buffer.GetAPIResource().Unmap(0, 0);
	return value;
}

int TestPermutations(rhi::Device device, const std::shared_ptr<org::Runtime>& runtime, SpirvCompiler& compiler) {
	auto fill = org::Buffer::CreateShared(rhi::HeapType::Upload, kBytes);
	auto readback = org::Buffer::CreateShared(rhi::HeapType::Readback, kBytes);
	auto output = org::Buffer::CreateSharedUnmaterialized(rhi::HeapType::DeviceLocal, kBytes, true);
	org::BufferBase::DescriptorRequirements views{};
	views.createUAV = true;
	views.uavDesc = {.dimension = rhi::UavDim::Buffer, .formatOverride = rhi::Format::R32_Typeless,
		.buffer = {.kind = rhi::BufferViewKind::Raw, .firstElement = 0, .numElements = kCount}};
	output->SetDescriptorRequirements(views);
	output->Materialize();
	fill->SetName("write.fill-source"); readback->SetName("write.readback"); output->SetName("write.output");
	{
		void* mapped = nullptr;
		fill->GetAPIResource().Map(&mapped, 0, kBytes);
		CHECK(mapped);
		std::fill_n(static_cast<uint32_t*>(mapped), kCount, kFill);
		fill->GetAPIResource().Unmap(0, kBytes);
	}
	auto state = std::make_shared<WriteState>();
	{
		org::GraphHost host({.runtime = runtime});
		host.AddExtension("test.permutations", [&] { return std::make_unique<WriteExtension>(fill, output, readback, state); });
		const auto frame = [&] {
			host.ExecuteFrame();
			ENSURE(device.WaitIdle() == rhi::Result::Ok);
			return ReadWritten(*readback);
		};
		// The build is held, so the pass skips its dispatch and the buffer keeps the fill.
		CHECK(frame() == kFill);
		CHECK(frame() == kFill);
		CHECK(state->skipped == 2 && state->selected == 0);
		CHECK(state->permutations->Size() == 1);
		const auto entry = state->permutations->Find(0);
		CHECK(entry && entry->Status(org::DeviceInstanceId::Primary) == org::PipelineBuildStatus::Building);
		const int compilesBefore = compiler.compiles.load();
		compiler.Release();
		CHECK(runtime->Pipelines()->Wait(entry, org::DeviceInstanceId::Primary) == org::PipelineBuildStatus::Ready);
		// Ready: the pass dispatches without the graph being built again.
		CHECK(frame() == state->value);
		CHECK(state->selected == 1);
		CHECK(compiler.compiles == compilesBefore); // A hit compiles nothing.
		// Another key is another permutation: skipped while it builds, then used.
		state->key = 1;
		CHECK(frame() == kFill);
		compiler.Release();
		CHECK(runtime->Pipelines()->Wait(state->permutations->Find(1), org::DeviceInstanceId::Primary) == org::PipelineBuildStatus::Ready);
		CHECK(frame() == state->value + 100);
		// Back to the first key, which is built.
		state->key = 0;
		CHECK(frame() == state->value);
		// A described pipeline is shared: acquiring the same description gives the permutation's entry.
		CHECK(runtime->Pipelines()->Acquire(WritePipeline(1)) == state->permutations->Find(1));
		// Clearing drops the set's hold. The pipeline goes once nothing else holds it, recordings included.
		const std::weak_ptr<const org::PipelineStatePayload> payload = state->permutations->Find(1)->State().GetPayload(
			org::DeviceInstanceId::Primary);
		state->permutations->Clear();
		CHECK(state->permutations->Size() == 0);
		for (int i = 0; i < 6 && !payload.expired(); ++i) {
			frame();
			compiler.Release();
		}
		CHECK(payload.expired());
	}
	// PipelineMiss::Wait blocks on the build instead.
	{
		org::PipelinePermutations permutations(runtime->Pipelines(), "test.wait");
		auto waiting = WritePipeline(0);
		waiting.name = "PermutationWrite.Wait";
		const auto selection = permutations.Select(7, org::DeviceInstanceId::Primary, [&] { return waiting; }, org::PipelineMiss::Wait);
		CHECK(selection && selection.status == org::PipelineBuildStatus::Ready);
		// A failed build is reported, and stays failed.
		auto failing = waiting;
		failing.name = "PermutationWrite.Fail";
		failing.shader.path = L"fail";
		const auto failed = permutations.Select(8, org::DeviceInstanceId::Primary, [&] { return failing; }, org::PipelineMiss::Wait);
		CHECK(!failed && failed.status == org::PipelineBuildStatus::Failed);
		CHECK(!permutations.Select(8, org::DeviceInstanceId::Primary, [&] { return failing; }));
	}
	return 0;
}

// Area-weighted downsample on the CPU, as shaders/GenerateMips.hlsl does it.
std::vector<float> Downsample(const std::vector<float>& source, uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh) {
	std::vector<float> out(size_t(dw) * dh * 4, 0.0f);
	const auto weights = [](uint32_t index, uint32_t sourceSize, uint32_t destinationSize, uint32_t& first, std::array<double, 3>& w) {
		const double ratio = double(sourceSize) / destinationSize, start = index * ratio, end = start + ratio;
		first = uint32_t(start);
		for (uint32_t i = 0; i < 3; ++i) {
			const double texel = double(first + i);
			w[i] = first + i < sourceSize ? (std::max)(0.0, (std::min)(end, texel + 1) - (std::max)(start, texel)) / ratio : 0.0;
		}
	};
	for (uint32_t y = 0; y < dh; ++y)
		for (uint32_t x = 0; x < dw; ++x) {
			uint32_t fx, fy;
			std::array<double, 3> wx, wy;
			weights(x, sw, dw, fx, wx);
			weights(y, sh, dh, fy, wy);
			for (uint32_t c = 0; c < 4; ++c) {
				double sum = 0;
				for (uint32_t j = 0; j < 3; ++j)
					for (uint32_t i = 0; i < 3; ++i)
						if (wx[i] * wy[j] > 0) sum += wx[i] * wy[j] * source[(size_t(fy + j) * sw + fx + i) * 4 + c];
				out[(size_t(y) * dw + x) * 4 + c] = float(sum);
			}
		}
	return out;
}

struct Subresource { uint32_t mip, slice; rhi::CopyableFootprint footprint; };

struct TextureCopyBindings { org::ResourceBindingToken buffer, texture; };

// Copies between a buffer and a texture's subresources, one footprint each. Record reads only its arguments, so the
// footprints travel in the frame data.
struct CopyFrame { std::shared_ptr<const std::vector<Subresource>> subresources; bool upload = false; };
class FootprintCopyPass final : public org::TypedRenderGraphPass<FootprintCopyPass, CopyFrame, TextureCopyBindings> {
public:
	FootprintCopyPass(std::shared_ptr<org::Resource> buffer, std::shared_ptr<org::Resource> texture,
		std::shared_ptr<const std::vector<Subresource>> subresources, bool upload)
		: m_buffer(std::move(buffer)), m_texture(std::move(texture)), m_subresources(std::move(subresources)), m_upload(upload) {}
	TextureCopyBindings Declare(org::PassBuilder& builder) {
		if (m_upload) return {builder.CopySource(m_buffer), builder.CopyDestination(org::Subresources(m_texture, org::Mip{0, 1}))};
		return {builder.CopyDestination(m_buffer), builder.CopySource(m_texture)};
	}
	CopyFrame Prepare(const TextureCopyBindings&, const org::PassPrepareContext&) { return {m_subresources, m_upload}; }
	static void Record(const TextureCopyBindings& bindings, const CopyFrame& frame, org::PassRecordContext& recording) {
		const auto buffer = recording.Resolve(bindings.buffer).GetHandle();
		const auto texture = recording.Resolve(bindings.texture).GetHandle();
		for (const auto& sub : *frame.subresources) {
			if (frame.upload && sub.mip != 0) continue;
			const rhi::BufferTextureCopyFootprint copy{.texture = texture, .buffer = buffer, .mip = sub.mip, .arraySlice = sub.slice,
				.footprint = sub.footprint};
			if (frame.upload) recording.Commands().CopyBufferToTexture(copy);
			else recording.Commands().CopyTextureToBuffer(copy);
		}
	}
private:
	std::shared_ptr<org::Resource> m_buffer, m_texture;
	std::shared_ptr<const std::vector<Subresource>> m_subresources;
	bool m_upload;
};

class MipsExtension final : public org::RenderGraph::IRenderGraphExtension {
public:
	MipsExtension(std::shared_ptr<org::Buffer> upload, std::shared_ptr<org::PixelBuffer> texture, std::shared_ptr<org::Buffer> readback,
		std::shared_ptr<const std::vector<Subresource>> subresources)
		: m_upload(std::move(upload)), m_texture(std::move(texture)), m_readback(std::move(readback)), m_subresources(std::move(subresources)) {}
	void GatherStructuralPasses(org::RenderGraph&, std::vector<org::RenderGraph::ExternalPassDesc>& out) override {
		out.push_back(org::RenderGraph::ExternalPassDesc::Copy("mips.upload",
			std::make_shared<FootprintCopyPass>(m_upload, m_texture, m_subresources, true)));
		out.push_back(org::RenderGraph::ExternalPassDesc::Compute("mips.generate", std::make_shared<org::GenerateMipsPass>(m_texture)));
		out.push_back(org::RenderGraph::ExternalPassDesc::Copy("mips.read",
			std::make_shared<FootprintCopyPass>(m_readback, m_texture, m_subresources, false)));
	}
private:
	std::shared_ptr<org::Buffer> m_upload;
	std::shared_ptr<org::PixelBuffer> m_texture;
	std::shared_ptr<org::Buffer> m_readback;
	std::shared_ptr<const std::vector<Subresource>> m_subresources;
};

int TestGenerateMips(rhi::Device device, const std::shared_ptr<org::Runtime>& runtime, uint32_t width, uint32_t height,
	uint32_t arraySize, bool cube) {
	org::TextureDescription desc;
	desc.imageDimensions = {{width, height, 0, 0}};
	desc.channels = 4;
	desc.format = rhi::Format::R32G32B32A32_Float;
	desc.hasSRV = true;
	desc.hasUAV = true;
	desc.generateMipMaps = true;
	desc.isCubemap = cube;
	desc.isArray = !cube && arraySize > 1;
	desc.arraySize = arraySize;
	auto texture = org::PixelBuffer::CreateShared(desc);
	texture->SetName("mips.texture");
	const uint32_t mips = texture->GetMipLevels();
	const uint32_t slices = cube ? 6 * arraySize : arraySize;
	CHECK(mips == uint32_t(std::floor(std::log2((std::max)(width, height)))) + 1);

	// One footprint per subresource, at 512-byte aligned offsets.
	auto subresources = std::make_shared<std::vector<Subresource>>();
	uint64_t bytes = 0;
	for (uint32_t slice = 0; slice < slices; ++slice)
		for (uint32_t mip = 0; mip < mips; ++mip) {
			Subresource sub{mip, slice, {}};
			const auto info = device.GetCopyableFootprints({.texture = texture->GetAPIResource().GetHandle(), .firstMip = mip,
				.mipCount = 1, .firstArraySlice = slice, .arraySize = 1, .baseOffset = bytes}, &sub.footprint, 1);
			CHECK(info.count == 1);
			bytes = (bytes + info.totalBytes + 511) & ~uint64_t(511);
			subresources->push_back(sub);
		}
	auto upload = org::Buffer::CreateShared(rhi::HeapType::Upload, bytes);
	auto readback = org::Buffer::CreateShared(rhi::HeapType::Readback, bytes);
	upload->SetName("mips.upload"); readback->SetName("mips.readback");

	const auto source = [&](uint32_t x, uint32_t y, uint32_t slice, uint32_t c) {
		return float(x) * 0.37f + float(y * y) * 0.11f + float(slice) * 10.0f + float(c) * 0.25f;
	};
	{
		void* mapped = nullptr;
		upload->GetAPIResource().Map(&mapped, 0, bytes);
		CHECK(mapped);
		for (const auto& sub : *subresources) {
			if (sub.mip) continue;
			for (uint32_t y = 0; y < height; ++y) {
				auto* row = reinterpret_cast<float*>(static_cast<char*>(mapped) + sub.footprint.offset + size_t(y) * sub.footprint.rowPitch);
				for (uint32_t x = 0; x < width; ++x)
					for (uint32_t c = 0; c < 4; ++c) row[x * 4 + c] = source(x, y, sub.slice, c);
			}
		}
		upload->GetAPIResource().Unmap(0, bytes);
	}
	{
		org::GraphHost host({.runtime = runtime});
		host.AddExtension("test.mips", [&] { return std::make_unique<MipsExtension>(upload, texture, readback, subresources); });
		host.ExecuteFrame();
		CHECK(device.WaitIdle() == rhi::Result::Ok);
	}
	void* mapped = nullptr;
	readback->GetAPIResource().Map(&mapped, 0, bytes);
	CHECK(mapped);
	int failure = 0;
	for (uint32_t slice = 0; slice < slices && !failure; ++slice) {
		std::vector<float> expected(size_t(width) * height * 4);
		for (uint32_t y = 0; y < height; ++y)
			for (uint32_t x = 0; x < width; ++x)
				for (uint32_t c = 0; c < 4; ++c) expected[(size_t(y) * width + x) * 4 + c] = source(x, y, slice, c);
		uint32_t w = width, h = height;
		for (uint32_t mip = 0; mip < mips && !failure; ++mip) {
			if (mip) {
				const uint32_t nw = (std::max)(1u, w / 2), nh = (std::max)(1u, h / 2);
				expected = Downsample(expected, w, h, nw, nh);
				w = nw; h = nh;
			}
			const auto& sub = *std::ranges::find_if(*subresources, [&](const auto& s) { return s.mip == mip && s.slice == slice; });
			for (uint32_t y = 0; y < h && !failure; ++y) {
				const auto* row = reinterpret_cast<const float*>(static_cast<const char*>(mapped) + sub.footprint.offset
					+ size_t(y) * sub.footprint.rowPitch);
				for (uint32_t x = 0; x < w * 4; ++x) {
					const float want = expected[size_t(y) * w * 4 + x];
					if (std::fabs(row[x] - want) > 1e-4f * (std::max)(1.0f, std::fabs(want))) {
						std::fprintf(stderr, "mip %u slice %u texel (%u,%u) channel %u: %f, expected %f\n", mip, slice, x / 4, y,
							x % 4, row[x], want);
						failure = __LINE__;
						break;
					}
				}
			}
		}
	}
	readback->GetAPIResource().Unmap(0, 0);
	return failure;
}

} // namespace

// Runs on the host's device (PersistentVulkanHostTests "pipelines"). Returns the failing line, or 0.
int TestPipelines(rhi::Device device) {
	if (const int line = TestInputLayoutIdentity()) return line;
	int result = 0;
	{
		auto compiler = std::make_shared<SpirvCompiler>(device);
		auto runtime = org::Runtime::Create({.device = device, .backend = rhi::Backend::Vulkan,
			.tasks = std::make_shared<org::runtime::ThreadPoolTaskService>(2), .pipelineCompiler = compiler});
		const auto run = [&](const char* name, auto&& test) {
			if (result) return;
			if (const int line = test()) {
				std::printf("FAIL: %s (line %d)\n", name, line);
				result = line;
			} else {
				std::printf("ok: %s\n", name);
			}
		};
		run("input layout build", [&] { return TestInputLayoutBuild(*runtime->Pipelines(), *compiler); });
		run("permutations", [&] { return TestPermutations(device, runtime, *compiler); });
		run("generate mips 2D, odd size", [&] { return TestGenerateMips(device, runtime, 7, 5, 1, false); });
		run("generate mips 2D array", [&] { return TestGenerateMips(device, runtime, 8, 8, 3, false); });
		run("generate mips cube", [&] { return TestGenerateMips(device, runtime, 4, 4, 1, true); });
		compiler->Release();
		device.WaitIdle();
	}
	return result;
}
