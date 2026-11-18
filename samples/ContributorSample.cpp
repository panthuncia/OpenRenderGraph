// A contributor module as a plugin would be one: it finds the host's table through the host module's export
// (ORG_GetRenderGraphAPI, the push model), registers, and adds two passes to one of the host's epochs: one clears a buffer
// of its own (the host's UNORDERED_ACCESS_CLEAR), the next copies it into a second. It counts its packets through their
// lifecycle, unregisters itself after a number of frames, and writes what happened to a log.
//
// ORGContributorSampleStart(hostModule, epoch, unregisterAfter, logPath): the host (or a test harness) calls it once.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#include <OpenRenderGraph/ContributorAPI.h>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>

namespace {

struct Sample {
	ORGRenderGraphAPI api{ sizeof(ORGRenderGraphAPI) };
	ORGRegistrationHandle registration = 0;
	std::string epoch;
	std::string logPath;
	uint32_t unregisterAfter = 0;
	std::atomic<uint32_t> prepared{0}, recorded{0}, submitted{0}, completed{0}, abandoned{0}, destroyed{0};
	std::atomic<uint32_t> activated{0}, retired{0};
	std::atomic<bool> unregistering{false};
	std::mutex logMutex;

	void Log(const char* format, ...) {
		std::scoped_lock lock(logMutex);
		FILE* file = nullptr;
		if (fopen_s(&file, logPath.c_str(), "a") != 0 || !file) return;
		va_list args;
		va_start(args, format);
		std::vfprintf(file, format, args);
		va_end(args);
		std::fputc('\n', file);
		std::fclose(file);
	}
	void Summary(const char* when) {
		Log("%s: prepared=%u recorded=%u submitted=%u completed=%u abandoned=%u destroyed=%u activated=%u retired=%u", when,
			prepared.load(), recorded.load(), submitted.load(), completed.load(), abandoned.load(), destroyed.load(), activated.load(),
			retired.load());
	}
};
Sample* g_sample = nullptr;

constexpr uint64_t kBytes = 4096;

const ORGBindingInfo* Binding(const ORGRecordContext* context, ORGBinding binding) {
	for (uint32_t i = 0; i < context->bindingCount; ++i) if (context->bindings[i].binding == binding) return &context->bindings[i];
	return nullptr;
}

void ORG_RG_CALL RecordClear(const void*, const ORGRecordContext*) { g_sample->recorded.fetch_add(1); }

void ORG_RG_CALL RecordCopy(const void*, const ORGRecordContext* context) {
	const auto* source = Binding(context, 1);
	const auto* destination = Binding(context, 2);
	if (!source || !destination) return;
	context->recorder.copy_buffer(context->recorder.context, context->recorder.borrow_token, destination->resource, 0, source->resource, 0, kBytes);
	g_sample->recorded.fetch_add(1);
}

void ORG_RG_CALL Submitted(const void*, uint64_t) { g_sample->submitted.fetch_add(1); }
void ORG_RG_CALL Completed(const void*, uint64_t) { g_sample->completed.fetch_add(1); }
void ORG_RG_CALL Abandoned(const void*, ORGAbandonReason) { g_sample->abandoned.fetch_add(1); }
void ORG_RG_CALL Destroy(void*) { g_sample->destroyed.fetch_add(1); }

ORGStatus Prepare(ORGPreparedPass* out, void(ORG_RG_CALL* record)(const void*, const ORGRecordContext*)) {
	out->data = nullptr;
	out->record = record;
	out->submitted = Submitted;
	out->completed = Completed;
	out->abandoned = Abandoned;
	out->destroy = Destroy;
	const auto count = g_sample->prepared.fetch_add(1) + 1;
	// Two packets a frame: unregister once the frames are done, from the preparation thread (any thread may).
	if (g_sample->unregisterAfter && count == 2 * g_sample->unregisterAfter && !g_sample->unregistering.exchange(true)) {
		g_sample->Summary("unregistering");
		g_sample->api.BeginUnregister(g_sample->registration);
	}
	return ORG_RG_OK;
}
ORGStatus ORG_RG_CALL PrepareClear(void*, const ORGPrepareContext*, ORGPreparedPass* out) { return Prepare(out, RecordClear); }
ORGStatus ORG_RG_CALL PrepareCopy(void*, const ORGPrepareContext*, ORGPreparedPass* out) { return Prepare(out, RecordCopy); }

ORGResourceAccessDesc Access(const char* id, ORGBinding binding, ORGAccessKind kind, uint32_t viewFlags = 0) {
	ORGResourceAccessDesc access{};
	access.structSize = sizeof(access); access.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
	access.resourceId = id; access.binding = binding; access.access = kind;
	access.range = { 0, UINT32_MAX, 0, UINT32_MAX }; access.viewFlags = viewFlags; access.elementCount = UINT32_MAX;
	return access;
}

ORGStatus ORG_RG_CALL Build(void*, ORGBuildHandle build) {
	const auto& api = g_sample->api;
	const auto buffer = [&](const char* id, uint32_t usages) {
		ORGResourceDesc desc{};
		desc.structSize = sizeof(desc); desc.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
		desc.id = id; desc.lifetime = ORG_RG_RESOURCE_PERSISTENT; desc.dimension = ORG_RG_RESOURCE_BUFFER;
		desc.heapClass = ORG_RG_HEAP_DEVICE_LOCAL; desc.sizing = ORG_RG_SIZE_ABSOLUTE; desc.byteSize = kBytes;
		desc.mipLevels = desc.sampleCount = 1; desc.allowedUsages = usages;
		return api.DeclareResource(build, &desc);
	};
	if (auto status = buffer("sample.source", ORG_RG_USAGE_UNORDERED_ACCESS | ORG_RG_USAGE_COPY_SOURCE); status != ORG_RG_OK) return status;
	if (auto status = buffer("sample.destination", ORG_RG_USAGE_COPY_DESTINATION); status != ORG_RG_OK) return status;
	const char* epoch = g_sample->epoch.empty() ? nullptr : g_sample->epoch.c_str();

	auto clear = Access("sample.source", 1, ORG_RG_ACCESS_UNORDERED_ACCESS_CLEAR, ORG_RG_VIEW_FLAG_RAW_BUFFER);
	clear.clearValue[0] = 7.0f;
	ORGPassDesc clearPass{};
	clearPass.structSize = sizeof(clearPass); clearPass.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
	clearPass.id = "sample.clear"; clearPass.kind = ORG_RG_PASS_COMPUTE; clearPass.queue = ORG_RG_QUEUE_FORCE_GRAPHICS;
	clearPass.accesses = &clear; clearPass.accessCount = 1; clearPass.epoch = epoch; clearPass.prepare = PrepareClear;
	if (auto status = api.DeclarePass(build, &clearPass); status != ORG_RG_OK) return status;

	const ORGResourceAccessDesc copy[] = {
		Access("sample.source", 1, ORG_RG_ACCESS_COPY_SOURCE),
		Access("sample.destination", 2, ORG_RG_ACCESS_COPY_DESTINATION),
	};
	const char* after[] = { "sample.clear" };
	ORGPassDesc copyPass{};
	copyPass.structSize = sizeof(copyPass); copyPass.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
	copyPass.id = "sample.copy"; copyPass.kind = ORG_RG_PASS_COPY; copyPass.queue = ORG_RG_QUEUE_FORCE_GRAPHICS;
	copyPass.after = after; copyPass.afterCount = 1; copyPass.accesses = copy; copyPass.accessCount = 2; copyPass.epoch = epoch;
	copyPass.prepare = PrepareCopy;
	return api.DeclarePass(build, &copyPass);
}

void ORG_RG_CALL Activated(void*, ORGGenerationHandle generation) {
	g_sample->activated.fetch_add(1);
	g_sample->Log("generation %llu activated", static_cast<unsigned long long>(generation));
}
void ORG_RG_CALL Retired(void*, ORGGenerationHandle generation) {
	g_sample->retired.fetch_add(1);
	g_sample->Log("generation %llu retired", static_cast<unsigned long long>(generation));
	g_sample->Summary("retired");
}
void ORG_RG_CALL Shutdown(void*) { g_sample->Summary("shutdown"); }

} // namespace

extern "C" __declspec(dllexport) int ORG_RG_CALL ORGContributorSampleStart(const wchar_t* hostModule, const char* epoch,
	uint32_t unregisterAfter, const char* logPath) {
	if (g_sample) return ORG_RG_E_DUPLICATE_ID;
	const HMODULE host = GetModuleHandleW(hostModule);
	using GetApi = ORGStatus(ORG_RG_CALL*)(uint32_t, ORGRenderGraphAPI*);
	const auto getApi = host ? reinterpret_cast<GetApi>(GetProcAddress(host, "ORG_GetRenderGraphAPI")) : nullptr;
	if (!getApi) return ORG_RG_E_RUNTIME_UNAVAILABLE;
	auto* sample = new Sample;  // lives for the process: the host may call back until it shuts down
	sample->epoch = epoch ? epoch : "";
	sample->unregisterAfter = unregisterAfter;
	sample->logPath = logPath ? logPath : "ORGContributorSample.log";
	if (const auto status = getApi(ORG_RENDER_GRAPH_API_CURRENT, &sample->api); status != ORG_RG_OK) {
		delete sample;
		return status;
	}
	g_sample = sample;
	ORGHostInfo info{ sizeof(info) };
	if (sample->api.GetHostInfo(&info) == ORG_RG_OK) sample->Log("host %s (%s) %s", info.id, info.displayName, info.version);
	ORGContributorDesc desc{};
	desc.structSize = sizeof(desc); desc.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
	desc.id = "sample.contributor"; desc.kind = ORG_RG_CONTRIBUTOR_OPTIONAL; desc.build = Build;
	desc.generationActivated = Activated; desc.generationRetired = Retired; desc.shutdown = Shutdown;
	const auto status = sample->api.RegisterContributor(&desc, &sample->registration);
	sample->Log("registered: %s (epoch %s, unregister after %u frames)", sample->api.StatusString(status),
		sample->epoch.empty() ? "every" : sample->epoch.c_str(), unregisterAfter);
	return status;
}
