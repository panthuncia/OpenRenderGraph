#include <OpenRenderGraph/ContributorRuntime.h>

#include "ContributorExecutionContext.h"

#include <cstring>
#include <stdexcept>
#include <unordered_set>

namespace org::contributor {
namespace {
	std::atomic<Runtime*> g_runtime{};

	template <class T> bool ValidOutput(T* value) noexcept
	{
		return value && value->structSize >= sizeof(T);
	}

	Runtime* Active() noexcept { return Runtime::GetCurrentForExport(); }

	ORGStatus ORG_RG_CALL RuntimeInfo(ORGRuntimeInfo* out)
	{
		auto* runtime = Active();
		if (!runtime || !runtime->IsAvailable()) return ORG_RG_E_RUNTIME_UNAVAILABLE;
		if (!ValidOutput(out)) return ORG_RG_E_INVALID_ARGUMENT;
		*out = { sizeof(*out), ORG_RENDER_GRAPH_API_CURRENT, 1u, runtime->GetBackend(),
			ORG_RG_CAP_RENDER_PASSES | ORG_RG_CAP_COMPUTE_PASSES | ORG_RG_CAP_COPY_PASSES |
			ORG_RG_CAP_ASYNC_COMPUTE | ORG_RG_CAP_COPY_QUEUE | ORG_RG_CAP_MANAGED_RESOURCES |
			ORG_RG_CAP_RESOURCE_ALIASING | ORG_RG_CAP_SCHEDULED_UPLOADS | ORG_RG_CAP_SERVICE_QUERY,
			runtime->GetActiveGeneration(), runtime->GetLastSubmittedCompletion(), runtime->GetCompletedValue(),
			runtime->GetFramesInFlight(), 0 };
		return ORG_RG_OK;
	}

	ORGStatus ORG_RG_CALL HostInfo(ORGHostInfo* out)
	{
		auto* runtime = Active();
		if (!runtime) return ORG_RG_E_RUNTIME_UNAVAILABLE;
		if (!ValidOutput(out)) return ORG_RG_E_INVALID_ARGUMENT;
		const auto& host = runtime->Host();
		*out = { sizeof(*out), ORG_RENDER_GRAPH_API_CURRENT,
			host.id.c_str(), host.displayName.c_str(), host.version.c_str() };
		return ORG_RG_OK;
	}

	ORGStatus ORG_RG_CALL AnchorCount(uint32_t* out)
	{
		auto* runtime = Active();
		if (!runtime) return ORG_RG_E_RUNTIME_UNAVAILABLE;
		if (!out) return ORG_RG_E_INVALID_ARGUMENT;
		*out = static_cast<uint32_t>(runtime->Host().anchors.size());
		return ORG_RG_OK;
	}

	ORGStatus ORG_RG_CALL Anchor(uint32_t index, ORGAnchorDescriptor* out)
	{
		auto* runtime = Active();
		if (!runtime) return ORG_RG_E_RUNTIME_UNAVAILABLE;
		if (!ValidOutput(out) || index >= runtime->Host().anchors.size()) return ORG_RG_E_INVALID_ARGUMENT;
		const auto& [id, name] = runtime->Host().anchors[index];
		*out = { sizeof(*out), ORG_RENDER_GRAPH_API_CURRENT, id.c_str(), name.c_str() };
		return ORG_RG_OK;
	}

	ORGStatus ORG_RG_CALL Register(const ORGContributorDesc* desc, ORGRegistrationHandle* out)
	{
		auto* runtime = Active(); if (!runtime) return ORG_RG_E_RUNTIME_UNAVAILABLE;
		const auto result = runtime->Registry().Register(desc, out);
		if (result == ORG_RG_OK) runtime->RequestRebuild();
		return result;
	}
	ORGStatus ORG_RG_CALL Unregister(ORGRegistrationHandle handle)
	{
		auto* runtime = Active(); if (!runtime) return ORG_RG_E_RUNTIME_UNAVAILABLE;
		const auto result = runtime->Registry().BeginUnregister(handle);
		if (result == ORG_RG_OK) runtime->RequestRebuild();
		return result;
	}
	ORGStatus ORG_RG_CALL RegistrationState(ORGRegistrationHandle handle, ORGRegistrationState* out)
	{ auto* runtime = Active(); return runtime ? runtime->Registry().GetRegistrationState(handle, out) : ORG_RG_E_RUNTIME_UNAVAILABLE; }
	ORGStatus ORG_RG_CALL Resource(ORGBuildHandle handle, const ORGResourceDesc* desc)
	{ auto* runtime = Active(); return runtime ? runtime->Registry().DeclareResource(handle, desc) : ORG_RG_E_RUNTIME_UNAVAILABLE; }
	ORGStatus ORG_RG_CALL Pass(ORGBuildHandle handle, const ORGPassDesc* desc)
	{ auto* runtime = Active(); return runtime ? runtime->Registry().DeclarePass(handle, desc) : ORG_RG_E_RUNTIME_UNAVAILABLE; }
	ORGStatus ORG_RG_CALL ApiRequestRebuild(ORGRegistrationHandle handle)
	{
		auto* runtime = Active(); if (!runtime) return ORG_RG_E_RUNTIME_UNAVAILABLE;
		const auto result = runtime->Registry().RequestRebuild(handle);
		if (result == ORG_RG_OK) runtime->RequestRebuild();
		return result;
	}
	ORGStatus ORG_RG_CALL Diagnostic(ORGRegistrationHandle handle, ORGDiagnostic* out)
	{ auto* runtime = Active(); return runtime ? runtime->Registry().GetDiagnostic(handle, out) : ORG_RG_E_RUNTIME_UNAVAILABLE; }

	ExecutionContextHost* ContextHost(const ORGExecutionContext* context) noexcept
	{
		if (!context || !context->hostContext) return nullptr;
		auto* host = const_cast<ExecutionContextHost*>(static_cast<const ExecutionContextHost*>(context->hostContext));
		return host->magic == ExecutionContextHost::kMagic ? host : nullptr;
	}
	ORGStatus ORG_RG_CALL BindingInfo(const ORGExecutionContext* context, ORGBinding binding, ORGBindingInfo* out)
	{
		auto* host = ContextHost(context);
		if (!host || !ValidOutput(out)) return ORG_RG_E_INVALID_ARGUMENT;
		const auto found = host->bindingInfo.find(binding);
		if (found == host->bindingInfo.end()) return ORG_RG_E_STALE_HANDLE;
		*out = found->second; return ORG_RG_OK;
	}
	ORGStatus ORG_RG_CALL BufferUpload(const ORGExecutionContext* context, const ORGBufferUpload* upload)
	{
		if (!upload || upload->structSize < sizeof(*upload) || upload->apiVersion != ORG_RENDER_GRAPH_API_CURRENT ||
			!upload->data || !upload->dataSize) return ORG_RG_E_INVALID_ARGUMENT;
		auto* host = ContextHost(context);
		if (!host || !host->queueBufferUpload) return ORG_RG_E_UNSUPPORTED_CAPABILITY;
		return host->queueBufferUpload(host->uploadUser, upload->destination, upload->destinationOffset,
			upload->data, upload->dataSize);
	}
	ORGStatus ORG_RG_CALL Service(const char* id, uint32_t version, void* out, uint32_t size)
	{
		auto* runtime = Active();
		return runtime && id ? runtime->QueryRegisteredService(id, version, out, size) : ORG_RG_E_INVALID_ARGUMENT;
	}
	const char* ORG_RG_CALL StatusString(ORGStatus status)
	{
		switch (status) {
		case ORG_RG_OK: return "success"; case ORG_RG_E_INVALID_ARGUMENT: return "invalid argument";
		case ORG_RG_E_UNSUPPORTED_VERSION: return "unsupported version"; case ORG_RG_E_RUNTIME_UNAVAILABLE: return "runtime unavailable";
		case ORG_RG_E_STALE_HANDLE: return "stale handle"; case ORG_RG_E_DUPLICATE_ID: return "duplicate identifier";
		case ORG_RG_E_RESERVED_ID: return "reserved identifier"; case ORG_RG_E_MISSING_DEPENDENCY: return "missing dependency";
		case ORG_RG_E_CYCLE: return "dependency cycle"; case ORG_RG_E_INCOMPATIBLE_RESOURCE: return "incompatible resource";
		case ORG_RG_E_WRONG_THREAD: return "wrong thread"; case ORG_RG_E_CALLBACK_FAILED: return "callback failed";
		case ORG_RG_E_CLOSED: return "closed"; case ORG_RG_E_NOT_READY: return "not ready";
		case ORG_RG_E_UNSUPPORTED_CAPABILITY: return "unsupported capability"; case ORG_RG_E_DUPLICATE_BINDING: return "duplicate binding";
		default: return "internal error";
		}
	}
}

Runtime::Runtime(RuntimeDesc desc) : desc_(std::move(desc))
{
	if (Current()) throw std::runtime_error("Only one exported contributor runtime may be active in a process");
	if (!desc_.device || !desc_.framesInFlight || desc_.host.structSize < sizeof(ORGHostDescriptor) ||
		desc_.host.apiVersion != ORG_RENDER_GRAPH_API_CURRENT || !desc_.host.id || !*desc_.host.id ||
		(desc_.host.anchorCount && !desc_.host.anchors))
		throw std::invalid_argument("Contributor runtime requires a device, frames-in-flight, and host identity");
	host_.id = desc_.host.id;
	host_.displayName = desc_.host.displayName ? desc_.host.displayName : host_.id;
	host_.version = desc_.host.version ? desc_.host.version : "unknown";
	std::vector<std::string> anchors;
	anchors.reserve(desc_.host.anchorCount);
	std::unordered_set<std::string> uniqueAnchors;
	for (uint32_t index = 0; index < desc_.host.anchorCount; ++index) {
		const auto& anchor = desc_.host.anchors[index];
		if (anchor.structSize < sizeof(ORGAnchorDescriptor) ||
			anchor.apiVersion != ORG_RENDER_GRAPH_API_CURRENT || !anchor.id || !*anchor.id ||
			!uniqueAnchors.emplace(anchor.id).second)
			throw std::invalid_argument("Host anchors must have valid, unique IDs");
		host_.anchors.emplace_back(anchor.id, anchor.displayName ? anchor.displayName : anchor.id);
		anchors.emplace_back(anchor.id);
	}
	registry_.SetAnchors(anchors);
	graph_ = ContributorGraphHost::Create(desc_.device, &extensions_, std::move(anchors));
	SetCurrent(this);
}

Runtime::~Runtime() { Shutdown(); }

Runtime* Runtime::Current() noexcept { return g_runtime.load(std::memory_order_acquire); }
void Runtime::SetCurrent(Runtime* value) noexcept
{
	g_runtime.store(value, std::memory_order_release);
}

ORGStatus Runtime::GetAPI(uint32_t version, ORGRenderGraphAPI* out) noexcept
{
	if (!out || out->structSize < sizeof(*out)) return ORG_RG_E_INVALID_ARGUMENT;
	if (version != ORG_RENDER_GRAPH_API_CURRENT) return ORG_RG_E_UNSUPPORTED_VERSION;
	*out = { sizeof(*out), version, &RuntimeInfo, &HostInfo, &AnchorCount, &Anchor, &Register, &Unregister,
		&RegistrationState, &Resource, &Pass, &ApiRequestRebuild, &Diagnostic, &BindingInfo, &BufferUpload, &Service, &StatusString };
	return ORG_RG_OK;
}

ExtensionRegistry::Handle Runtime::RegisterExtension(ExtensionRegistry::Descriptor descriptor)
{
	auto handle = extensions_.Register(std::move(descriptor)); RequestRebuild(); return handle;
}
void Runtime::BeginUnregisterExtension(ExtensionRegistry::Handle handle)
{ extensions_.BeginUnregister(handle); RequestRebuild(); }
void Runtime::RegisterService(std::string id, ServiceQuery query)
{ if (id.empty() || !query) throw std::invalid_argument("Service requires an ID and query callback"); std::scoped_lock lock(servicesMutex_); services_.insert_or_assign(std::move(id), std::move(query)); }
void Runtime::UnregisterService(std::string_view id)
{ std::scoped_lock lock(servicesMutex_); services_.erase(std::string(id)); }
ORGStatus Runtime::QueryRegisteredService(std::string_view id, uint32_t version, void* out, uint32_t size) noexcept
{
	ServiceQuery query;
	{
		std::scoped_lock lock(servicesMutex_);
		const auto found = services_.find(std::string(id));
		if (found == services_.end()) return ORG_RG_E_UNSUPPORTED_CAPABILITY;
		query = found->second;
	}
	try { return query(version, out, size); } catch (...) { return ORG_RG_E_INTERNAL; }
}

bool Runtime::Rebuild(const FrameDesc& frame) noexcept
{
	if (active_ && active_->lastCompletion > CompletedValue()) {
		rebuildRequested_.store(true, std::memory_order_release);
		return false;
	}
	auto candidate = std::make_unique<Generation>(); candidate->id = nextGeneration_.fetch_add(1);
	const auto status = registry_.Compile(candidate->id, frame.renderWidth, frame.renderHeight,
		frame.outputWidth, frame.outputHeight, candidate->candidate);
	if (status != ORG_RG_OK) return false;
	try { graph_->SetStructuralDefinition(candidate->candidate); }
	catch (...) { rebuildRequested_.store(true, std::memory_order_release); return false; }
	if (active_) retired_.emplace_back(active_->lastCompletion, std::move(active_));
	active_ = std::move(candidate);
	activeGeneration_.store(active_->id, std::memory_order_release);
	registry_.Activate(active_->candidate); return true;
}

bool Runtime::Execute(const FrameDesc& frame) noexcept
{
	if (!available_.load(std::memory_order_acquire) || !frame.renderWidth || !frame.renderHeight ||
		!frame.outputWidth || !frame.outputHeight || !frame.completeTimeline || !frame.completeValue) return false;
	if (frame.renderWidth != renderWidth_ || frame.renderHeight != renderHeight_ ||
		frame.outputWidth != outputWidth_ || frame.outputHeight != outputHeight_) {
		renderWidth_ = frame.renderWidth; renderHeight_ = frame.renderHeight;
		outputWidth_ = frame.outputWidth; outputHeight_ = frame.outputHeight;
		RequestRebuild();
	}
	if (rebuildRequested_.exchange(false, std::memory_order_acq_rel) || registry_.HasPendingRebuild())
		if (!Rebuild(frame)) rebuildRequested_.store(true, std::memory_order_release);
	if (!active_) return false;
	try {
		ContributorGraphHost::FrameInfo info{ frame.frameIndex, active_->id, frame.completeValue,
			static_cast<uint32_t>(frame.frameIndex % desc_.framesInFlight), desc_.framesInFlight,
			frame.renderWidth, frame.renderHeight, frame.outputWidth, frame.outputHeight };
		graph_->Execute(static_cast<uint32_t>(frame.frameIndex), frame.frameFenceValue,
			frame.readyTimeline, frame.readyValue, frame.completeTimeline, frame.completeValue, info);
	} catch (...) { NotifyDeviceLost(UINT32_MAX); return false; }
	active_->lastCompletion = frame.completeValue;
	lastSubmittedCompletion_.store(frame.completeValue, std::memory_order_release);
	Retire(CompletedValue());
	return true;
}

void Runtime::Retire(uint64_t completedValue) noexcept
{
	if (graph_) graph_->Retire(completedValue);
	while (!retired_.empty() && retired_.front().first <= completedValue) {
		registry_.Retire(retired_.front().second->id); retired_.pop_front();
	}
}
void Runtime::NotifyDeviceLost(uint32_t reason) noexcept
{ available_.store(false, std::memory_order_release); registry_.NotifyDeviceLost(reason); }
void Runtime::Shutdown() noexcept
{
	if (!available_.exchange(false) && !graph_) return;
	if (graph_) graph_->Retire(UINT64_MAX);
	registry_.Shutdown(); active_.reset(); activeGeneration_.store(0, std::memory_order_release);
	retired_.clear(); graph_.reset();
	{ std::scoped_lock lock(servicesMutex_); services_.clear(); }
	if (Current() == this) SetCurrent(nullptr);
}
uint64_t Runtime::CompletedValue() const noexcept { try { return desc_.completedValue ? desc_.completedValue() : 0; } catch (...) { return 0; } }
uint64_t Runtime::GetActiveGeneration() const noexcept { return activeGeneration_.load(std::memory_order_acquire); }

} // namespace org::contributor
