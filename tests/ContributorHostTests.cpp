// A contributor driven through the C table end to end (the "contributor" mode of PersistentVulkanHostTests): its compute
// pass writes value + index into a buffer it owns, through a pipeline made by the pipeline service and an input value it
// uploads while preparing; its copy pass copies the result into a host export the test reads back. Then it unregisters,
// and its generation retires.

#include <OpenRenderGraph/ContributorHost.h>
#include "TestSupport.h"
#include <OpenRenderGraph/GraphHost.h>

#include "Render/Runtime/ThreadPoolTaskService.h"
#include "Resources/Resource.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <vector>


namespace {

constexpr uint32_t kCount = 64;  // the host buffer PersistentVulkanHostTests imports
constexpr uint64_t kBytes = kCount * sizeof(uint32_t);

struct TestContributor {
	ORGRenderGraphAPI api{ sizeof(ORGRenderGraphAPI) };
	ORGPipelineService pipelines{ sizeof(ORGPipelineService) };
	basicrhi_c_pipeline pipeline{};
	uint32_t value = 0;
	std::atomic<int> activated{0}, retired{0}, packets{0}, destroyed{0}, completed{0};
};
TestContributor* g_contributor = nullptr;

struct WritePacket { uint32_t outputIndex = 0, inputIndex = 0; basicrhi_c_pipeline pipeline{}; basicrhi_c_pipeline_layout layout{}; };

const ORGBindingInfo* FindBinding(const ORGBindingInfo* bindings, uint32_t count, ORGBinding binding) {
	for (uint32_t i = 0; i < count; ++i) if (bindings[i].binding == binding) return &bindings[i];
	return nullptr;
}

void ORG_RG_CALL RecordWrite(const void* data, const ORGRecordContext* context) {
	const auto* packet = static_cast<const WritePacket*>(data);
	const auto& recorder = context->recorder;
	recorder.bind_layout(recorder.context, recorder.borrow_token, packet->layout);
	recorder.bind_pipeline(recorder.context, recorder.borrow_token, packet->pipeline);
	const uint32_t constants[3] = { packet->outputIndex, packet->inputIndex, kCount };
	recorder.push_constants(recorder.context, recorder.borrow_token, 4 /* compute */, 0, 0, 0, { constants, sizeof(constants) });
	recorder.dispatch(recorder.context, recorder.borrow_token, (kCount + 63) / 64, 1, 1);
}

void ORG_RG_CALL RecordCopy(const void*, const ORGRecordContext* context) {
	const auto* source = FindBinding(context->bindings, context->bindingCount, 1);
	const auto* destination = FindBinding(context->bindings, context->bindingCount, 2);
	if (!source || !destination) return;
	const auto& recorder = context->recorder;
	recorder.copy_buffer(recorder.context, recorder.borrow_token, destination->resource, 0, source->resource, 0, kBytes);
}

void ORG_RG_CALL Completed(const void*, uint64_t) { g_contributor->completed.fetch_add(1); }
void ORG_RG_CALL DestroyWrite(void* data) { delete static_cast<WritePacket*>(data); g_contributor->destroyed.fetch_add(1); }
void ORG_RG_CALL DestroyCopy(void*) { g_contributor->destroyed.fetch_add(1); }

ORGStatus ORG_RG_CALL PrepareWrite(void*, const ORGPrepareContext* context, ORGPreparedPass* out) {
	const auto* input = FindBinding(context->bindings, context->bindingCount, 1);
	const auto* output = FindBinding(context->bindings, context->bindingCount, 2);
	if (!input || !output || input->descriptorIndex == UINT32_MAX || output->descriptorIndex == UINT32_MAX) return ORG_RG_E_INVALID_ARGUMENT;
	ORGBufferUpload upload{ sizeof(upload), ORG_RENDER_GRAPH_API_CURRENT, 1, 0, &g_contributor->value, sizeof(uint32_t) };
	if (const auto status = g_contributor->api.QueueBufferUpload(context, &upload); status != ORG_RG_OK) return status;
	auto* packet = new WritePacket{ output->descriptorIndex, input->descriptorIndex, g_contributor->pipeline,
		g_contributor->pipelines.computeLayout };
	out->data = packet;
	out->record = RecordWrite;
	out->completed = Completed;
	out->destroy = DestroyWrite;
	g_contributor->packets.fetch_add(1);
	return ORG_RG_OK;
}

ORGStatus ORG_RG_CALL PrepareCopy(void*, const ORGPrepareContext*, ORGPreparedPass* out) {
	out->data = nullptr;
	out->record = RecordCopy;
	out->destroy = DestroyCopy;
	g_contributor->packets.fetch_add(1);
	return ORG_RG_OK;
}

ORGResourceAccessDesc Access(const char* resource, ORGBinding binding, ORGAccessKind kind, uint32_t viewFlags) {
	ORGResourceAccessDesc access{};
	access.structSize = sizeof(access); access.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
	access.resourceId = resource; access.binding = binding; access.access = kind;
	access.range = { 0, UINT32_MAX, 0, UINT32_MAX }; access.viewFlags = viewFlags; access.elementCount = UINT32_MAX;
	return access;
}

ORGStatus ORG_RG_CALL Build(void*, ORGBuildHandle build) {
	const auto& api = g_contributor->api;
	const auto buffer = [&](const char* id, uint64_t bytes, uint32_t usages) {
		ORGResourceDesc desc{};
		desc.structSize = sizeof(desc); desc.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
		desc.id = id; desc.lifetime = ORG_RG_RESOURCE_PERSISTENT; desc.dimension = ORG_RG_RESOURCE_BUFFER;
		desc.heapClass = ORG_RG_HEAP_DEVICE_LOCAL; desc.sizing = ORG_RG_SIZE_ABSOLUTE; desc.byteSize = bytes;
		desc.mipLevels = desc.sampleCount = 1; desc.allowedUsages = usages;
		return api.DeclareResource(build, &desc);
	};
	if (auto status = buffer("test.contributor.input", 16, ORG_RG_USAGE_SHADER_RESOURCE); status != ORG_RG_OK) return status;
	if (auto status = buffer("test.contributor.scratch", kBytes, ORG_RG_USAGE_UNORDERED_ACCESS | ORG_RG_USAGE_COPY_SOURCE); status != ORG_RG_OK)
		return status;

	const ORGResourceAccessDesc writeAccesses[] = {
		Access("test.contributor.input", 1, ORG_RG_ACCESS_SHADER_RESOURCE, ORG_RG_VIEW_FLAG_RAW_BUFFER),
		Access("test.contributor.scratch", 2, ORG_RG_ACCESS_UNORDERED_ACCESS, ORG_RG_VIEW_FLAG_RAW_BUFFER),
	};
	ORGPassDesc write{};
	write.structSize = sizeof(write); write.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
	write.id = "test.contributor.write"; write.kind = ORG_RG_PASS_COMPUTE; write.queue = ORG_RG_QUEUE_FORCE_GRAPHICS;
	write.accesses = writeAccesses; write.accessCount = 2; write.prepare = PrepareWrite;
	if (auto status = api.DeclarePass(build, &write); status != ORG_RG_OK) return status;

	const ORGResourceAccessDesc copyAccesses[] = {
		Access("test.contributor.scratch", 1, ORG_RG_ACCESS_COPY_SOURCE, 0),
		Access("test.output", 2, ORG_RG_ACCESS_COPY_DESTINATION, 0),
	};
	const char* after[] = { "test.contributor.write" };
	ORGPassDesc copy{};
	copy.structSize = sizeof(copy); copy.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
	copy.id = "test.contributor.copy"; copy.kind = ORG_RG_PASS_COPY; copy.queue = ORG_RG_QUEUE_FORCE_GRAPHICS;
	copy.after = after; copy.afterCount = 1; copy.accesses = copyAccesses; copy.accessCount = 2; copy.prepare = PrepareCopy;
	return api.DeclarePass(build, &copy);
}

void ORG_RG_CALL Activated(void*, ORGGenerationHandle) { g_contributor->activated.fetch_add(1); }
void ORG_RG_CALL Retired(void*, ORGGenerationHandle) { g_contributor->retired.fetch_add(1); }

} // namespace

int TestContributors(rhi::Device device, const std::function<std::shared_ptr<org::Resource>()>& importOutput, const uint32_t* mapped,
	const char* spirvPath,
	rhi::PipelineLayoutHandle layout) {
	std::ifstream file(spirvPath, std::ios::binary);
	const std::vector<char> spirv((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	REQUIRE(!spirv.empty(), "contributor shader");

	TestContributor contributor;
	g_contributor = &contributor;
	{
		auto runtime = org::Runtime::Create({.device = device, .backend = rhi::Backend::Vulkan,
			.tasks = std::make_shared<org::runtime::ThreadPoolTaskService>(2)});
		org::GraphHost host({.runtime = runtime,
			.queueBoundary = {.entry = true, .exit = true}, .epochOrder = {0}});
		// After the host: importing creates descriptors, which need the graph's runtime.
		const auto output = importOutput();
		REQUIRE(output, "import host buffer");
		ORGExportDescriptor exported{};
		exported.dimension = ORG_RG_RESOURCE_BUFFER; exported.byteSize = kBytes;
		exported.allowedUsages = ORG_RG_USAGE_COPY_DESTINATION; exported.flags = ORG_RG_EXPORT_WRITABLE;
		org::contributor::ContributorHost contributors({.id = "test.host", .displayName = "Test host", .version = "1",
			.device = device, .backend = rhi::Backend::Vulkan,
			.exports = {{"test.output", "Host output", exported, [&] { return output; }}},
			.computeLayout = layout});
		REQUIRE(org::contributor::ContributorHost::GetAPI(ORG_RENDER_GRAPH_API_CURRENT, &contributor.api) == ORG_RG_OK, "API table");
		REQUIRE(contributor.api.QueryService(ORG_PIPELINE_SERVICE_ID, ORG_PIPELINE_SERVICE_VERSION_1, &contributor.pipelines,
			sizeof(contributor.pipelines)) == ORG_RG_OK && contributor.pipelines.device, "pipeline service");
		basicrhi_c_pipeline_desc_v1 pipelineDesc{};
		pipelineDesc.structure_size = sizeof(pipelineDesc);
		pipelineDesc.kind = BASICRHI_C_PIPELINE_COMPUTE;
		pipelineDesc.layout = contributor.pipelines.computeLayout;
		pipelineDesc.compute_shader = { spirv.data(), spirv.size() };
		const auto* table = contributor.pipelines.device;
		REQUIRE(table->create_pipeline(table->context, &pipelineDesc, &contributor.pipeline) == BASICRHI_C_OK,
			"pipeline through the service");
		uint32_t exports = 0;
		ORGExportDescriptor described{ sizeof(described) };
		REQUIRE(contributor.api.GetExportCount(&exports) == ORG_RG_OK && exports == 1
			&& contributor.api.GetExport(0, &described) == ORG_RG_OK && std::strcmp(described.id, "test.output") == 0, "export table");

		ORGContributorDesc desc{};
		desc.structSize = sizeof(desc); desc.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
		desc.id = "test.contributor"; desc.kind = ORG_RG_CONTRIBUTOR_REQUIRED; desc.build = Build;
		desc.generationActivated = Activated; desc.generationRetired = Retired;
		ORGRegistrationHandle handle{};
		REQUIRE(contributor.api.RegisterContributor(&desc, &handle) == ORG_RG_OK, "register");
		host.AddExtension("org.contributors", [&] { return contributors.CreateExtension(); });

		for (uint32_t frame = 0; frame < 4; ++frame) {
			contributor.value = 7000u + 10u * frame;
			contributors.Poll(host);
			host.ExecuteFrame();
			if (frame == 0) REQUIRE(contributor.activated.load() == 1, "the generation activates with the build");
			REQUIRE(device.WaitIdle() == rhi::Result::Ok, "wait for the frame");
			bool matches = true;
			for (uint32_t i = 0; i < kCount; ++i) if (mapped[i] != contributor.value + i) { matches = false; break; }
			REQUIRE(matches, "the contributor's passes write and copy every element");
		}
		REQUIRE(contributor.completed.load() > 0, "packets complete");

		// Unregistering rebuilds without the contributor; its generation retires once nothing uses it.
		REQUIRE(contributor.api.BeginUnregister(handle) == ORG_RG_OK, "unregister");
		for (uint32_t frame = 0; frame < 4; ++frame) {
			contributors.Poll(host);
			host.ExecuteFrame();
		}
		host.DestroyGraph();
		ORGRegistrationState state{};
		REQUIRE(contributor.api.GetRegistrationState(handle, &state) == ORG_RG_OK && state == ORG_RG_REGISTRATION_RETIRED,
			"the contributor retired");
		REQUIRE(contributor.retired.load() == contributor.activated.load(), "every activated generation retired");
		REQUIRE(contributor.destroyed.load() == contributor.packets.load(), "every packet destroyed");
		table->destroy_pipeline(table->context, contributor.pipeline, 0);
	}
	g_contributor = nullptr;
	std::printf("ContributorHostTests: ok (packets=%d)\n", contributor.packets.load());
	return 0;
}
