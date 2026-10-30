#include <OpenRenderGraph/ContributorRegistry.h>

#include <string_view>

#define CHECK(value) do { if (!(value)) return __LINE__; } while (false)

namespace {
	org::contributor::ContributorRegistry* registry{};
	// What Build declares: its pass's access target and epoch.
	const char* accessTarget = "sample.contributor.buffer";
	ORGAccessKind accessKind = ORG_RG_ACCESS_SHADER_RESOURCE;
	const char* passEpoch = nullptr;
	ORGStatus declaredPass = ORG_RG_OK;

	ORGStatus ORG_RG_CALL Prepare(void*, const ORGPrepareContext*, ORGPreparedPass*) { return ORG_RG_OK; }
	ORGStatus ORG_RG_CALL Build(void*, ORGBuildHandle build)
	{
		ORGResourceDesc resource{};
		resource.structSize = sizeof(resource); resource.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
		resource.id = "sample.contributor.buffer"; resource.lifetime = ORG_RG_RESOURCE_TRANSIENT;
		resource.dimension = ORG_RG_RESOURCE_BUFFER; resource.heapClass = ORG_RG_HEAP_DEVICE_LOCAL;
		resource.sizing = ORG_RG_SIZE_ABSOLUTE; resource.byteSize = 1024;
		resource.mipLevels = resource.sampleCount = 1;
		resource.allowedUsages = ORG_RG_USAGE_SHADER_RESOURCE;
		auto status = registry->DeclareResource(build, &resource);
		if (status != ORG_RG_OK) return status;
		ORGResourceAccessDesc access{};
		access.structSize = sizeof(access); access.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
		access.resourceId = accessTarget; access.binding = 1; access.access = accessKind;
		access.range = { 0, UINT32_MAX, 0, UINT32_MAX };
		access.elementCount = UINT32_MAX;
		if (accessKind == ORG_RG_ACCESS_SHADER_RESOURCE && std::string_view(accessTarget) == "sample.contributor.buffer")
			access.viewFlags = ORG_RG_VIEW_FLAG_RAW_BUFFER;
		ORGPassDesc pass{};
		pass.structSize = sizeof(pass); pass.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
		pass.id = "sample.contributor.pass"; pass.kind = ORG_RG_PASS_COMPUTE; pass.queue = ORG_RG_QUEUE_AUTOMATIC;
		pass.accesses = &access; pass.accessCount = 1; pass.prepare = &Prepare; pass.epoch = passEpoch;
		declaredPass = registry->DeclarePass(build, &pass);
		return declaredPass;
	}
}

int main()
{
	org::contributor::ContributorRegistry instance;
	registry = &instance;
	ORGExportDescriptor color{};
	color.dimension = ORG_RG_RESOURCE_TEXTURE_2D; color.format = ORG_RG_FORMAT_R8G8B8A8_UNORM;
	color.width = 64; color.height = 64; color.depthOrArraySize = 1; color.mipLevels = 1;
	color.allowedUsages = ORG_RG_USAGE_SHADER_RESOURCE | ORG_RG_USAGE_UNORDERED_ACCESS; color.flags = ORG_RG_EXPORT_READABLE;
	instance.SetHostSymbols({ "host.begin", "host.end" }, { "host.main" }, { { "host.color", color } });
	ORGContributorDesc desc{};
	desc.structSize = sizeof(desc); desc.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
	desc.id = "sample.contributor"; desc.kind = ORG_RG_CONTRIBUTOR_REQUIRED; desc.build = &Build;
	ORGRegistrationHandle handle{};
	CHECK(instance.Register(&desc, &handle) == ORG_RG_OK && handle);
	org::contributor::ContributorRegistry::Candidate candidate;
	CHECK(instance.Compile(1, 1920, 1080, 2560, 1440, candidate) == ORG_RG_OK);
	CHECK(candidate.resources.size() == 1 && candidate.passes.size() == 1);
	CHECK(candidate.passes[0].prepare == &Prepare && candidate.passes[0].epoch.empty());
	instance.Activate(candidate);

	// An export read in a named epoch.
	accessTarget = "host.color"; passEpoch = "host.main";
	org::contributor::ContributorRegistry::Candidate exported;
	CHECK(instance.Compile(2, 1, 1, 1, 1, exported) == ORG_RG_OK);
	CHECK(exported.passes.size() == 1 && exported.passes[0].epoch == "host.main");
	CHECK(exported.passes[0].accesses[0].exportIndex == 0 && exported.passes[0].accesses[0].resource == 0);
	// Writing a read-only export is refused, and does not request another build.
	accessKind = ORG_RG_ACCESS_UNORDERED_ACCESS;
	org::contributor::ContributorRegistry::Candidate refused;
	CHECK(instance.Compile(3, 1, 1, 1, 1, refused) == ORG_RG_E_INCOMPATIBLE_RESOURCE);
	CHECK(!instance.HasPendingRebuild());
	// An epoch the host does not have.
	accessKind = ORG_RG_ACCESS_SHADER_RESOURCE; passEpoch = "host.missing";
	org::contributor::ContributorRegistry::Candidate unknownEpoch;
	CHECK(instance.Compile(4, 1, 1, 1, 1, unknownEpoch) != ORG_RG_OK && declaredPass == ORG_RG_E_MISSING_DEPENDENCY);
	accessTarget = "sample.contributor.buffer"; passEpoch = nullptr;

	CHECK(instance.BeginUnregister(handle) == ORG_RG_OK);
	CHECK(instance.HasPendingRebuild());
	org::contributor::ContributorRegistry::Candidate empty;
	CHECK(instance.Compile(5, 1, 1, 1, 1, empty) == ORG_RG_OK && empty.passes.empty());
	instance.Activate(empty); instance.Retire(1);
	ORGRegistrationState state{};
	CHECK(instance.GetRegistrationState(handle, &state) == ORG_RG_OK);
	CHECK(state == ORG_RG_REGISTRATION_RETIRED);
	return 0;
}
