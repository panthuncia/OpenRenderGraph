#include "DeviceTable.h"

#include <OpenRenderGraph/ContributorAPI.h>

#include "Managers/Singletons/DeletionManager.h"

#include <vector>

namespace org::contributor {

namespace {

uint64_t Key(basicrhi_c_handle handle) noexcept { return (uint64_t{handle.generation} << 32) | handle.index; }

basicrhi_c_handle ToC(uint32_t index, uint32_t generation) noexcept { return {index, generation}; }


basicrhi_c_result BASICRHI_C_CALL QueryCapability(void*, uint32_t, void*, size_t) { return BASICRHI_C_UNSUPPORTED; }
basicrhi_c_result BASICRHI_C_CALL CreateResource(void*, const basicrhi_c_resource_desc_v1*, basicrhi_c_resource*) { return BASICRHI_C_UNSUPPORTED; }
void BASICRHI_C_CALL DestroyResource(void*, basicrhi_c_resource, uint64_t) {}
basicrhi_c_result BASICRHI_C_CALL CreateView(void*, basicrhi_c_resource, const basicrhi_c_view_desc_v1*, basicrhi_c_view*) { return BASICRHI_C_UNSUPPORTED; }
void BASICRHI_C_CALL DestroyView(void*, basicrhi_c_view, uint64_t) {}
basicrhi_c_result BASICRHI_C_CALL CreateSampler(void*, const basicrhi_c_sampler_desc_v1*, basicrhi_c_sampler*) { return BASICRHI_C_UNSUPPORTED; }
void BASICRHI_C_CALL DestroySampler(void*, basicrhi_c_sampler, uint64_t) {}
basicrhi_c_result BASICRHI_C_CALL CreatePipelineLayout(void*, const basicrhi_c_pipeline_layout_desc_v1*, basicrhi_c_pipeline_layout*) { return BASICRHI_C_UNSUPPORTED; }
void BASICRHI_C_CALL DestroyPipelineLayout(void*, basicrhi_c_pipeline_layout, uint64_t) {}

bool HasBytes(basicrhi_c_byte_span span) noexcept { return span.data && span.size_bytes && span.size_bytes <= UINT32_MAX; }

rhi::ShaderBinary Binary(basicrhi_c_byte_span span) noexcept { return {span.data, static_cast<uint32_t>(span.size_bytes)}; }

} // namespace

DeviceTable::DeviceTable(rhi::Device device) : m_device(device) {
	m_api.structure_size = sizeof(m_api);
	m_api.abi_version = BASICRHI_C_ABI_VERSION_1;
	m_api.context = this;
	m_api.query_capability = QueryCapability;
	m_api.create_resource = CreateResource;
	m_api.destroy_resource = DestroyResource;
	m_api.create_view = CreateView;
	m_api.destroy_view = DestroyView;
	m_api.create_sampler = CreateSampler;
	m_api.destroy_sampler = DestroySampler;
	m_api.create_pipeline_layout = CreatePipelineLayout;
	m_api.destroy_pipeline_layout = DestroyPipelineLayout;
	m_api.create_pipeline = CreatePipeline;
	m_api.destroy_pipeline = DestroyPipeline;
	m_api.create_command_signature = CreateCommandSignature;
	m_api.destroy_command_signature = DestroyCommandSignature;
}

DeviceTable::~DeviceTable() {
	// Whatever the contributors did not destroy goes with the host, after the graph (no GPU work can still use it).
	std::scoped_lock lock(m_mutex);
	for (auto& [_, pipeline] : m_pipelines) DeletionManager::GetInstance().MarkForDelete(std::move(pipeline));
	for (auto& [_, signature] : m_signatures) DeletionManager::GetInstance().MarkForDelete(std::move(signature));
}

basicrhi_c_result BASICRHI_C_CALL DeviceTable::CreatePipeline(void* context, const basicrhi_c_pipeline_desc_v1* desc,
	basicrhi_c_pipeline* out) {
	auto* self = static_cast<DeviceTable*>(context);
	if (!self || !desc || !out || desc->structure_size < sizeof(*desc)) return BASICRHI_C_INVALID_ARGUMENT;
	const rhi::SubobjLayout layout{rhi::PipelineLayoutHandle{desc->layout.index, desc->layout.generation}};
	std::vector<rhi::SubobjShader> shaders;
	rhi::SubobjRaster raster{};
	rhi::SubobjBlend blend{};
	rhi::SubobjDepth depth{};
	rhi::SubobjRTVs targets{};
	rhi::SubobjDSV depthFormat{};
	rhi::SubobjSample sample{};
	rhi::SubobjPrimitiveTopology topology{};
	std::vector<rhi::PipelineStreamItem> items{rhi::Make(layout)};
	switch (desc->kind) {
	case BASICRHI_C_PIPELINE_COMPUTE:
		if (!HasBytes(desc->compute_shader)) return BASICRHI_C_INVALID_ARGUMENT;
		shaders.push_back({rhi::ShaderStage::Compute, Binary(desc->compute_shader), ""});
		break;
	case BASICRHI_C_PIPELINE_RASTER:
	case BASICRHI_C_PIPELINE_MESH: {
		if (desc->kind == BASICRHI_C_PIPELINE_RASTER) {
			if (!HasBytes(desc->vertex_shader)) return BASICRHI_C_INVALID_ARGUMENT;
			shaders.push_back({rhi::ShaderStage::Vertex, Binary(desc->vertex_shader), ""});
		} else {
			if (!HasBytes(desc->mesh_shader)) return BASICRHI_C_INVALID_ARGUMENT;
			if (HasBytes(desc->amplification_shader))
				shaders.push_back({rhi::ShaderStage::Task, Binary(desc->amplification_shader), ""});
			shaders.push_back({rhi::ShaderStage::Mesh, Binary(desc->mesh_shader), ""});
		}
		if (HasBytes(desc->pixel_shader)) shaders.push_back({rhi::ShaderStage::Pixel, Binary(desc->pixel_shader), ""});
		if (desc->render_target_count > 8) return BASICRHI_C_INVALID_ARGUMENT;
		const auto flags = desc->raster_flags;
		raster.rs.cull = (flags & ORG_RG_RASTER_CULL_BACK) ? rhi::CullMode::Back
			: (flags & ORG_RG_RASTER_CULL_FRONT) ? rhi::CullMode::Front : rhi::CullMode::None;
		raster.rs.frontCCW = (flags & ORG_RG_RASTER_FRONT_CCW) != 0;
		depth.ds.depthEnable = (flags & ORG_RG_RASTER_DEPTH_TEST) != 0;
		depth.ds.depthWrite = (flags & ORG_RG_RASTER_DEPTH_WRITE) != 0;
		depth.ds.depthFunc = (flags & ORG_RG_RASTER_DEPTH_GREATER) ? rhi::CompareOp::GreaterEqual : rhi::CompareOp::LessEqual;
		blend.bs.numAttachments = desc->render_target_count ? desc->render_target_count : 1;
		if (flags & ORG_RG_RASTER_ALPHA_BLEND)
			for (uint32_t i = 0; i < blend.bs.numAttachments; ++i) {
				auto& attachment = blend.bs.attachments[i];
				attachment.enable = true;
				attachment.srcColor = rhi::BlendFactor::SrcAlpha;
				attachment.dstColor = rhi::BlendFactor::InvSrcAlpha;
			}
		targets.rt.count = desc->render_target_count;
		for (uint32_t i = 0; i < desc->render_target_count; ++i) targets.rt.formats[i] = static_cast<rhi::Format>(desc->render_target_formats[i]);
		depthFormat.dsv = static_cast<rhi::Format>(desc->depth_format);
		sample.sd.count = desc->sample_count ? desc->sample_count : 1;
		topology.pt = desc->primitive_topology ? static_cast<rhi::PrimitiveTopology>(desc->primitive_topology)
			: rhi::PrimitiveTopology::TriangleList;
		items.push_back(rhi::Make(raster));
		items.push_back(rhi::Make(blend));
		items.push_back(rhi::Make(depth));
		items.push_back(rhi::Make(targets));
		items.push_back(rhi::Make(depthFormat));
		items.push_back(rhi::Make(sample));
		if (desc->kind == BASICRHI_C_PIPELINE_RASTER) items.push_back(rhi::Make(topology));
		break;
	}
	default: return BASICRHI_C_INVALID_ARGUMENT;
	}
	for (const auto& shader : shaders) items.push_back(rhi::Make(shader));
	rhi::PipelinePtr pipeline;
	if (self->m_device.CreatePipeline(items.data(), static_cast<uint32_t>(items.size()), pipeline) != rhi::Result::Ok || !pipeline)
		return BASICRHI_C_BACKEND_ERROR;
	const auto handle = pipeline->GetHandle();
	*out = ToC(handle.index, handle.generation);
	std::scoped_lock lock(self->m_mutex);
	self->m_pipelines[Key(*out)] = std::move(pipeline);
	return BASICRHI_C_OK;
}

void BASICRHI_C_CALL DeviceTable::DestroyPipeline(void* context, basicrhi_c_pipeline pipeline, uint64_t) {
	auto* self = static_cast<DeviceTable*>(context);
	if (!self) return;
	std::scoped_lock lock(self->m_mutex);
	const auto it = self->m_pipelines.find(Key(pipeline));
	if (it == self->m_pipelines.end()) return;
	DeletionManager::GetInstance().MarkForDelete(std::move(it->second));
	self->m_pipelines.erase(it);
}

basicrhi_c_result BASICRHI_C_CALL DeviceTable::CreateCommandSignature(void* context, const basicrhi_c_command_signature_desc_v1* desc,
	basicrhi_c_command_signature* out) {
	auto* self = static_cast<DeviceTable*>(context);
	if (!self || !desc || !out || desc->structure_size < sizeof(*desc) || (desc->argument_count && !desc->arguments))
		return BASICRHI_C_INVALID_ARGUMENT;
	std::vector<rhi::IndirectArg> arguments(desc->argument_count);
	for (size_t i = 0; i < desc->argument_count; ++i) {
		const auto& source = desc->arguments[i];
		auto& argument = arguments[i];
		argument.kind = static_cast<rhi::IndirectArgKind>(source.kind);
		if (argument.kind == rhi::IndirectArgKind::Constant) {
			argument.u.rootConstants.rootIndex = source.slot;
			argument.u.rootConstants.destOffset32 = source.byte_offset / 4;
			argument.u.rootConstants.num32 = source.byte_size / 4;
		} else if (argument.kind == rhi::IndirectArgKind::VertexBuffer) {
			argument.u.vertexBuffer.slot = source.slot;
		}
	}
	rhi::CommandSignatureDesc signatureDesc{};
	signatureDesc.args = {arguments.data(), static_cast<uint32_t>(arguments.size())};
	signatureDesc.byteStride = desc->byte_stride;
	rhi::CommandSignaturePtr signature;
	if (self->m_device.CreateCommandSignature(signatureDesc, rhi::PipelineLayoutHandle{desc->layout.index, desc->layout.generation},
		signature) != rhi::Result::Ok || !signature)
		return BASICRHI_C_BACKEND_ERROR;
	const auto handle = signature->GetHandle();
	*out = ToC(handle.index, handle.generation);
	std::scoped_lock lock(self->m_mutex);
	self->m_signatures[Key(*out)] = std::move(signature);
	return BASICRHI_C_OK;
}

void BASICRHI_C_CALL DeviceTable::DestroyCommandSignature(void* context, basicrhi_c_command_signature signature, uint64_t) {
	auto* self = static_cast<DeviceTable*>(context);
	if (!self) return;
	std::scoped_lock lock(self->m_mutex);
	const auto it = self->m_signatures.find(Key(signature));
	if (it == self->m_signatures.end()) return;
	DeletionManager::GetInstance().MarkForDelete(std::move(it->second));
	self->m_signatures.erase(it);
}

} // namespace org::contributor
