#include "RenderPasses/GenerateMipsPass.h"

#include <algorithm>
#include <array>
#include <stdexcept>

#include <rhi_helpers.h>

#include "Resources/GloballyIndexedResource.h"

namespace org {

namespace {

// The shader's GenerateMipsConstants, in order.
constexpr uint32_t kConstantCount = 7;
constexpr uint32_t kSrgbFlag = 1u;
constexpr uint32_t kGroupSize = 8;

} // namespace

std::vector<GenerateMipsStep> PlanGenerateMips(uint32_t width, uint32_t height, uint32_t mipLevels) {
	std::vector<GenerateMipsStep> steps;
	if (!width || !height || mipLevels < 2) return steps;
	steps.reserve(mipLevels - 1);
	for (uint32_t mip = 1; mip < mipLevels; ++mip) {
		const GenerateMipsStep step{
			(std::max)(1u, width >> (mip - 1)), (std::max)(1u, height >> (mip - 1)),
			(std::max)(1u, width >> mip), (std::max)(1u, height >> mip)};
		steps.push_back(step);
	}
	return steps;
}

GenerateMipsPass::GenerateMipsPass(std::shared_ptr<Resource> texture, GenerateMipsOptions options)
	: m_texture(std::move(texture)), m_options(std::move(options)) {
	if (!m_texture) throw std::invalid_argument("GenerateMipsPass needs a texture");
}

GenerateMipsBindings GenerateMipsPass::Declare(PassBuilder& builder) {
	rhi::ResourceDesc desc{};
	if (!m_texture->TryGetRHIResourceDesc(desc) || desc.type != rhi::ResourceType::Texture2D)
		throw std::invalid_argument("GenerateMipsPass: '" + m_texture->GetName() + "' is not a 2D texture");
	if (desc.texture.sampleCount > 1)
		throw std::invalid_argument("GenerateMipsPass: '" + m_texture->GetName() + "' is multisampled");

	GenerateMipsBindings bindings;
	bindings.slices = (std::max)(1u, static_cast<uint32_t>(desc.texture.depthOrLayers));
	const bool array = bindings.slices > 1;
	const uint32_t mips = desc.texture.mipLevels;
	if (mips < 2) return bindings;

	// One declaration for the whole chain. A dispatch reads one mip and writes the next, through unordered-access views.
	std::vector<UavView> views(mips);
	for (uint32_t mip = 0; mip < mips; ++mip)
		views[mip] = {array ? static_cast<uint32_t>(UAVViewType::Texture2DArrayFull) : UINT32_MAX, mip, 0};
	bindings.mips = builder.UnorderedAccess(m_texture, std::span<const UavView>(views)).views;

	ComputePipelineDesc pipeline;
	pipeline.shader = m_options.shader;
	pipeline.name = array ? "GenerateMips[Array]" : "GenerateMips[2D]";
	if (array) pipeline.defines.push_back({L"ORG_GENERATE_MIPS_ARRAY", L"1"});
	pipeline.layout = PipelineLayoutRef::RootConstants(kConstantCount, rhi::ShaderStage::Compute, "GenerateMips");
	bindings.pipeline = builder.Pipeline(std::move(pipeline));
	return bindings;
}

GenerateMipsFrameData GenerateMipsPass::Prepare(const GenerateMipsBindings& bindings, const PassPrepareContext& preparation) {
	GenerateMipsFrameData data;
	if (bindings.mips.size() < 2) return data;
	// The frame's backing, which may differ from the one the pass was declared with.
	const auto& texture = preparation.Describe(bindings.mips.front()).texture;
	data.steps = PlanGenerateMips(texture.width, texture.height, (std::min)(static_cast<uint32_t>(texture.mipLevels),
		static_cast<uint32_t>(bindings.mips.size())));
	if (rhi::helpers::IsSRGB(texture.format)) data.flags |= kSrgbFlag;
	return data;
}

void GenerateMipsPass::Record(const GenerateMipsBindings& bindings, const GenerateMipsFrameData& data, PassRecordContext& recording) {
	if (data.steps.empty()) return;
	recording.BindComputePipeline(bindings.pipeline);
	for (size_t i = 0; i < data.steps.size(); ++i) {
		const auto& step = data.steps[i];
		if (i) recording.UavBarrier(); // The previous dispatch wrote this one's source.
		const std::array<uint32_t, kConstantCount> constants{
			recording.Index(bindings.mips[i]), recording.Index(bindings.mips[i + 1]),
			step.sourceWidth, step.sourceHeight, step.destinationWidth, step.destinationHeight, data.flags};
		recording.PushConstants(0, constants);
		recording.Commands().Dispatch((step.destinationWidth + kGroupSize - 1) / kGroupSize,
			(step.destinationHeight + kGroupSize - 1) / kGroupSize, bindings.slices);
	}
}

} // namespace org
