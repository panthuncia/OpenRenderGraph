#include <OpenRenderGraph/ContributorRegistry.h>

#define CHECK(value) do { if (!(value)) return __LINE__; } while (false)

namespace {
	org::contributor::ContributorRegistry* registry{};
	ORGStatus ORG_RG_CALL Execute(void*, const ORGExecutionContext*) { return ORG_RG_OK; }
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
		access.resourceId = resource.id; access.binding = 1; access.access = ORG_RG_ACCESS_SHADER_RESOURCE;
		access.range = { 0, UINT32_MAX, 0, UINT32_MAX }; access.viewFlags = ORG_RG_VIEW_FLAG_RAW_BUFFER;
		access.elementCount = UINT32_MAX;
		ORGPassDesc pass{};
		pass.structSize = sizeof(pass); pass.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
		pass.id = "sample.contributor.pass"; pass.kind = ORG_RG_PASS_COMPUTE; pass.queue = ORG_RG_QUEUE_AUTOMATIC;
		pass.accesses = &access; pass.accessCount = 1; pass.execute = &Execute;
		return registry->DeclarePass(build, &pass);
	}
}

int main()
{
	org::contributor::ContributorRegistry instance;
	registry = &instance;
	instance.SetAnchors({ "host.begin", "host.end" });
	ORGContributorDesc desc{};
	desc.structSize = sizeof(desc); desc.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
	desc.id = "sample.contributor"; desc.kind = ORG_RG_CONTRIBUTOR_REQUIRED; desc.build = &Build;
	ORGRegistrationHandle handle{};
	CHECK(instance.Register(&desc, &handle) == ORG_RG_OK && handle);
	org::contributor::ContributorRegistry::Candidate candidate;
	CHECK(instance.Compile(1, 1920, 1080, 2560, 1440, candidate) == ORG_RG_OK);
	CHECK(candidate.resources.size() == 1 && candidate.passes.size() == 1);
	instance.Activate(candidate);
	CHECK(instance.BeginUnregister(handle) == ORG_RG_OK);
	org::contributor::ContributorRegistry::Candidate empty;
	CHECK(instance.Compile(2, 1, 1, 1, 1, empty) == ORG_RG_OK);
	instance.Activate(empty); instance.Retire(1);
	ORGRegistrationState state{};
	CHECK(instance.GetRegistrationState(handle, &state) == ORG_RG_OK);
	CHECK(state == ORG_RG_REGISTRATION_RETIRED);
	return 0;
}
