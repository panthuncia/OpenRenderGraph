#include "RenderPasses/ResolveMultisamplePass.h"

#include <algorithm>
#include <array>
#include <stdexcept>

#include <rhi_helpers.h>

namespace org {

namespace {

// The shader's ResolveConstants, in order.
constexpr uint32_t kConstantCount = 6;
constexpr uint32_t kSrgbDestinationFlag = 1u;
constexpr uint32_t kGroupSize = 8;

} // namespace

ResolveMultisamplePass::ResolveMultisamplePass(std::shared_ptr<Resource> source, std::shared_ptr<Resource> destination,
	ResolveMultisampleOptions options)
	: m_source(std::move(source)), m_destination(std::move(destination)), m_options(std::move(options)) {
	if (!m_source || !m_destination) throw std::invalid_argument("ResolveMultisamplePass needs a source and a destination");
}

ResolveMultisampleBindings ResolveMultisamplePass::Declare(PassBuilder& builder) {
	ResolveMultisampleBindings bindings;
	bindings.source = builder.ShaderResource(m_source).View();
	bindings.destination = builder.UnorderedAccess(Subresources(m_destination, Mip{m_options.destinationMip, 1}),
		UavView{UINT32_MAX, m_options.destinationMip, 0}).View();
	ComputePipelineDesc pipeline;
	pipeline.shader = m_options.shader;
	pipeline.name = "ResolveMultisample";
	pipeline.layout = PipelineLayoutRef::RootConstants(kConstantCount, rhi::ShaderStage::Compute, "ResolveMultisample");
	bindings.pipeline = builder.Pipeline(std::move(pipeline));
	return bindings;
}

ResolveMultisampleFrameData ResolveMultisamplePass::Prepare(const ResolveMultisampleBindings& bindings,
	const PassPrepareContext& preparation) {
	const auto& source = preparation.Describe(bindings.source).texture;
	const auto& destination = preparation.Describe(bindings.destination).texture;
	if (source.sampleCount < 2) throw std::invalid_argument("ResolveMultisamplePass: the source is not multisampled");
	if (destination.sampleCount > 1) throw std::invalid_argument("ResolveMultisamplePass: the destination is multisampled");
	const uint32_t mip = m_options.destinationMip;
	ResolveMultisampleFrameData data;
	data.width = (std::min)(source.width, (std::max)(1u, destination.width >> mip));
	data.height = (std::min)(source.height, (std::max)(1u, destination.height >> mip));
	data.samples = source.sampleCount;
	if (rhi::helpers::IsSRGB(destination.format)) data.flags |= kSrgbDestinationFlag;
	return data;
}

void ResolveMultisamplePass::Record(const ResolveMultisampleBindings& bindings, const ResolveMultisampleFrameData& data,
	PassRecordContext& recording) {
	if (!data.width || !data.height) return;
	recording.BindComputePipeline(bindings.pipeline);
	const std::array<uint32_t, kConstantCount> constants{
		recording.Index(bindings.source), recording.Index(bindings.destination), data.width, data.height, data.samples, data.flags};
	recording.PushConstants(0, constants);
	recording.Commands().Dispatch((data.width + kGroupSize - 1) / kGroupSize, (data.height + kGroupSize - 1) / kGroupSize, 1);
}

} // namespace org
