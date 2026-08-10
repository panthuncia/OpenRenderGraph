#include <OpenRenderGraph/ContributorRegistry.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <queue>
#include <unordered_set>
#include <string_view>

namespace org::contributor {

namespace
{
	constexpr uint32_t kKnownUsageMask = (ORG_RG_USAGE_LEGACY_INTEROP << 1) - 1;
	constexpr uint32_t kKnownViewFlags = ORG_RG_VIEW_FLAG_RAW_BUFFER | ORG_RG_VIEW_FLAG_READ_ONLY_DEPTH |
		ORG_RG_VIEW_FLAG_READ_ONLY_STENCIL;

	template <class T>
	bool ValidHeader(const T* value) noexcept
	{
		return value && value->structSize >= sizeof(T) && value->apiVersion == ORG_RENDER_GRAPH_API_CURRENT;
	}

	bool ValidCount(const void* pointer, uint32_t count) noexcept { return count == 0 || pointer != nullptr; }

	uint32_t UsageForAccess(ORGAccessKind access) noexcept
	{
		switch (access) {
		case ORG_RG_ACCESS_SHADER_RESOURCE: return ORG_RG_USAGE_SHADER_RESOURCE;
		case ORG_RG_ACCESS_CONSTANT_BUFFER: return ORG_RG_USAGE_CONSTANT_BUFFER;
		case ORG_RG_ACCESS_UNORDERED_ACCESS:
		case ORG_RG_ACCESS_UNORDERED_ACCESS_CLEAR: return ORG_RG_USAGE_UNORDERED_ACCESS;
		case ORG_RG_ACCESS_RENDER_TARGET:
		case ORG_RG_ACCESS_RENDER_TARGET_CLEAR: return ORG_RG_USAGE_RENDER_TARGET;
		case ORG_RG_ACCESS_DEPTH_READ: return ORG_RG_USAGE_DEPTH_READ;
		case ORG_RG_ACCESS_DEPTH_READ_WRITE:
		case ORG_RG_ACCESS_DEPTH_STENCIL_CLEAR: return ORG_RG_USAGE_DEPTH_WRITE;
		case ORG_RG_ACCESS_COPY_SOURCE: return ORG_RG_USAGE_COPY_SOURCE;
		case ORG_RG_ACCESS_COPY_DESTINATION: return ORG_RG_USAGE_COPY_DESTINATION;
		case ORG_RG_ACCESS_INDIRECT_ARGUMENT: return ORG_RG_USAGE_INDIRECT_ARGUMENT;
		case ORG_RG_ACCESS_INDEX_BUFFER: return ORG_RG_USAGE_INDEX_BUFFER;
		case ORG_RG_ACCESS_LEGACY_INTEROP: return ORG_RG_USAGE_LEGACY_INTEROP;
		default: return 0;
		}
	}

	bool ValidPassQueue(ORGPassKind kind, ORGQueueAssignment queue) noexcept
	{
		if (kind < ORG_RG_PASS_RENDER || kind > ORG_RG_PASS_COPY ||
			queue < ORG_RG_QUEUE_AUTOMATIC || queue > ORG_RG_QUEUE_FORCE_COPY) return false;
		if (kind == ORG_RG_PASS_RENDER) return queue == ORG_RG_QUEUE_AUTOMATIC || queue == ORG_RG_QUEUE_FORCE_GRAPHICS;
		if (kind == ORG_RG_PASS_COMPUTE) return queue != ORG_RG_QUEUE_FORCE_COPY;
		return true;
	}

	bool ValidAccessForPass(ORGPassKind kind, ORGAccessKind access) noexcept
	{
		if (!UsageForAccess(access)) return false;
		if (kind == ORG_RG_PASS_COPY)
			return access == ORG_RG_ACCESS_COPY_SOURCE || access == ORG_RG_ACCESS_COPY_DESTINATION;
		if (kind == ORG_RG_PASS_COMPUTE)
			return access == ORG_RG_ACCESS_SHADER_RESOURCE || access == ORG_RG_ACCESS_CONSTANT_BUFFER ||
				access == ORG_RG_ACCESS_UNORDERED_ACCESS || access == ORG_RG_ACCESS_UNORDERED_ACCESS_CLEAR ||
				access == ORG_RG_ACCESS_INDIRECT_ARGUMENT || access == ORG_RG_ACCESS_LEGACY_INTEROP;
		return true;
	}

	ORGViewKind DefaultViewKind(ORGAccessKind access) noexcept
	{
		switch (access) {
		case ORG_RG_ACCESS_SHADER_RESOURCE: return ORG_RG_VIEW_SHADER_RESOURCE;
		case ORG_RG_ACCESS_CONSTANT_BUFFER: return ORG_RG_VIEW_CONSTANT_BUFFER;
		case ORG_RG_ACCESS_UNORDERED_ACCESS:
		case ORG_RG_ACCESS_UNORDERED_ACCESS_CLEAR: return ORG_RG_VIEW_UNORDERED_ACCESS;
		case ORG_RG_ACCESS_RENDER_TARGET:
		case ORG_RG_ACCESS_RENDER_TARGET_CLEAR: return ORG_RG_VIEW_RENDER_TARGET;
		case ORG_RG_ACCESS_DEPTH_READ:
		case ORG_RG_ACCESS_DEPTH_READ_WRITE:
		case ORG_RG_ACCESS_DEPTH_STENCIL_CLEAR: return ORG_RG_VIEW_DEPTH_STENCIL;
		default: return ORG_RG_VIEW_NONE;
		}
	}

	uint32_t ResolveExtent(float scale, uint32_t reference) noexcept
	{
		if (scale <= 0.0f || !std::isfinite(scale)) return 0;
		const double result = static_cast<double>(scale) * reference;
		if (result < 1.0 || result > std::numeric_limits<uint32_t>::max()) return 0;
		return static_cast<uint32_t>(result + 0.5);
	}
}

struct ContributorRegistry::Contributor
{
	uint64_t handle{};
	std::string id;
	ORGContributorKind kind{};
	void* userData{};
	ORGBuildCallback build{};
	ORGGenerationCallback activated{};
	ORGGenerationCallback retired{};
	ORGDeviceLostCallback deviceLost{};
	ORGShutdownCallback shutdown{};
	std::unordered_set<uint64_t> activeGenerations;
};

struct ContributorRegistry::Build
{
	uint64_t handle{};
	Contributor contributor;
	std::vector<Resource> resources;
	std::vector<Pass> passes;
};

ContributorRegistry::ContributorRegistry() = default;
ContributorRegistry::~ContributorRegistry() = default;

void ContributorRegistry::SetAnchors(std::vector<std::string> anchors)
{
	std::unique_lock lock(mutex_);
	anchors_ = std::move(anchors);
}

bool ContributorRegistry::IsNamespaced(const char* value) noexcept
{
	if (!value) return false;
	const std::string_view id(value);
	const auto dot = id.find('.');
	return dot != 0 && dot != std::string_view::npos && dot + 1 < id.size();
}

void ContributorRegistry::SetDiagnostic(ORGRegistrationHandle contributor, ORGStatus status,
	uint32_t phase, uint64_t generation, std::string message) noexcept
{
	std::unique_lock lock(mutex_);
	auto& diagnostic = diagnostics_[contributor];
	diagnostic = {};
	diagnostic.structSize = sizeof(diagnostic);
	diagnostic.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
	diagnostic.status = status;
	diagnostic.phase = phase;
	diagnostic.sequence = ++diagnosticSequence_;
	diagnostic.generation = generation;
	diagnostic.contributor = contributor;
	std::strncpy(diagnostic.message, message.c_str(), sizeof(diagnostic.message) - 1);
}

ORGStatus ContributorRegistry::Register(const ORGContributorDesc* desc, ORGRegistrationHandle* out) noexcept
{
	if (!ValidHeader(desc) || !out || !IsNamespaced(desc->id) || !desc->build ||
		desc->kind < ORG_RG_CONTRIBUTOR_REQUIRED || desc->kind > ORG_RG_CONTRIBUTOR_DIAGNOSTIC)
		return ORG_RG_E_INVALID_ARGUMENT;
	if (std::string_view(desc->id).starts_with("org.")) return ORG_RG_E_RESERVED_ID;
	std::unique_lock lock(mutex_);
	if (!open_) return ORG_RG_E_CLOSED;
	const auto duplicate = [&](const auto& item) { return item.second.id == desc->id; };
	if (std::ranges::any_of(contributors_, duplicate) || std::ranges::any_of(unregistering_, duplicate))
		return ORG_RG_E_DUPLICATE_ID;
	Contributor contributor{};
	contributor.handle = nextHandle_++;
	contributor.id = desc->id;
	contributor.kind = desc->kind;
	contributor.userData = desc->userData;
	contributor.build = desc->build;
	contributor.activated = desc->generationActivated;
	contributor.retired = desc->generationRetired;
	contributor.deviceLost = desc->deviceLost;
	contributor.shutdown = desc->shutdown;
	*out = contributor.handle;
	contributors_.emplace(contributor.handle, std::move(contributor));
	rebuildRequested_ = true;
	return ORG_RG_OK;
}

ORGStatus ContributorRegistry::BeginUnregister(ORGRegistrationHandle handle) noexcept
{
	std::unique_lock lock(mutex_);
	const auto it = contributors_.find(handle);
	if (it == contributors_.end()) return unregistering_.contains(handle) ? ORG_RG_OK : ORG_RG_E_STALE_HANDLE;
	unregistering_.emplace(handle, std::move(it->second));
	contributors_.erase(it);
	rebuildRequested_ = true;
	if (unregistering_.at(handle).activeGenerations.empty()) unregistering_.erase(handle);
	return ORG_RG_OK;
}

ORGStatus ContributorRegistry::GetRegistrationState(ORGRegistrationHandle handle, ORGRegistrationState* out) const noexcept
{
	if (!out) return ORG_RG_E_INVALID_ARGUMENT;
	std::shared_lock lock(mutex_);
	if (contributors_.contains(handle)) { *out = ORG_RG_REGISTRATION_ACTIVE; return ORG_RG_OK; }
	if (unregistering_.contains(handle)) { *out = ORG_RG_REGISTRATION_UNREGISTERING; return ORG_RG_OK; }
	*out = ORG_RG_REGISTRATION_RETIRED;
	return ORG_RG_OK;
}

ORGStatus ContributorRegistry::DeclareResource(ORGBuildHandle buildHandle, const ORGResourceDesc* desc) noexcept
{
	std::scoped_lock lock(buildMutex_);
	if (!currentBuild_ || currentBuild_->handle != buildHandle) return ORG_RG_E_STALE_HANDLE;
	if (buildThread_ != std::this_thread::get_id()) return ORG_RG_E_WRONG_THREAD;
	if (!ValidHeader(desc) || !IsNamespaced(desc->id) || desc->lifetime < ORG_RG_RESOURCE_TRANSIENT ||
		desc->lifetime > ORG_RG_RESOURCE_PERSISTENT || desc->dimension < ORG_RG_RESOURCE_BUFFER ||
		desc->dimension > ORG_RG_RESOURCE_TEXTURE_3D || desc->heapClass < ORG_RG_HEAP_DEVICE_LOCAL ||
		desc->heapClass > ORG_RG_HEAP_READBACK || desc->sizing < ORG_RG_SIZE_ABSOLUTE ||
		desc->sizing > ORG_RG_SIZE_OUTPUT_RELATIVE || desc->format < ORG_RG_FORMAT_UNKNOWN ||
		desc->format >= ORG_RG_FORMAT_COUNT || (desc->allowedUsages & ~kKnownUsageMask) || desc->allowAlias > 1)
		return ORG_RG_E_INVALID_ARGUMENT;
	const std::string id(desc->id);
	if (!id.starts_with(currentBuild_->contributor.id + ".") || id.starts_with("org.") ||
		std::ranges::find(anchors_, id) != anchors_.end()) return ORG_RG_E_RESERVED_ID;
	if (std::ranges::any_of(currentBuild_->resources, [&](const Resource& r) { return r.id == id; }))
		return ORG_RG_E_DUPLICATE_ID;
	if (desc->dimension == ORG_RG_RESOURCE_BUFFER) {
		if (!desc->byteSize || desc->format != ORG_RG_FORMAT_UNKNOWN || desc->mipLevels > 1 || desc->sampleCount > 1)
			return ORG_RG_E_INVALID_ARGUMENT;
		if (desc->structureByteStride && (desc->structureByteStride > desc->byteSize ||
			desc->byteSize % desc->structureByteStride != 0)) return ORG_RG_E_INVALID_ARGUMENT;
		if (desc->allowedUsages & (ORG_RG_USAGE_RENDER_TARGET | ORG_RG_USAGE_DEPTH_READ | ORG_RG_USAGE_DEPTH_WRITE))
			return ORG_RG_E_INCOMPATIBLE_RESOURCE;
	} else if (desc->format == ORG_RG_FORMAT_UNKNOWN || !desc->depthOrArraySize || !desc->mipLevels || !desc->sampleCount) {
		return ORG_RG_E_INVALID_ARGUMENT;
	}
	if (desc->dimension != ORG_RG_RESOURCE_BUFFER) {
		if (desc->heapClass != ORG_RG_HEAP_DEVICE_LOCAL) return ORG_RG_E_INCOMPATIBLE_RESOURCE;
		if (desc->dimension == ORG_RG_RESOURCE_TEXTURE_1D && desc->height != 1)
			return ORG_RG_E_INCOMPATIBLE_RESOURCE;
		if (desc->dimension == ORG_RG_RESOURCE_TEXTURE_3D && desc->sampleCount != 1)
			return ORG_RG_E_INCOMPATIBLE_RESOURCE;
		if (desc->sampleCount > 1 && (desc->mipLevels != 1 ||
			(desc->allowedUsages & ORG_RG_USAGE_UNORDERED_ACCESS)))
			return ORG_RG_E_INCOMPATIBLE_RESOURCE;
	}
	if (desc->sizing == ORG_RG_SIZE_ABSOLUTE && desc->dimension != ORG_RG_RESOURCE_BUFFER && (!desc->width || !desc->height))
		return ORG_RG_E_INVALID_ARGUMENT;
	if (desc->sizing != ORG_RG_SIZE_ABSOLUTE && desc->dimension != ORG_RG_RESOURCE_BUFFER &&
		(desc->widthScale <= 0.0f || desc->heightScale <= 0.0f || !std::isfinite(desc->widthScale) || !std::isfinite(desc->heightScale)))
		return ORG_RG_E_INVALID_ARGUMENT;
	Resource resource{};
	resource.handle = nextHandle_++;
	resource.contributor = currentBuild_->contributor.handle;
	resource.id = id;
	resource.desc = *desc;
	resource.desc.id = nullptr;
	currentBuild_->resources.push_back(std::move(resource));
	return ORG_RG_OK;
}

ORGStatus ContributorRegistry::DeclarePass(ORGBuildHandle buildHandle, const ORGPassDesc* desc) noexcept
{
	std::scoped_lock lock(buildMutex_);
	if (!currentBuild_ || currentBuild_->handle != buildHandle) return ORG_RG_E_STALE_HANDLE;
	if (buildThread_ != std::this_thread::get_id()) return ORG_RG_E_WRONG_THREAD;
	constexpr uint32_t knownFlags = ORG_RG_PASS_PARALLEL_RECORDING_SAFE | ORG_RG_PASS_DISABLE_STATISTICS | ORG_RG_PASS_GEOMETRY;
	if (!ValidHeader(desc) || !IsNamespaced(desc->id) || !desc->execute || !ValidPassQueue(desc->kind, desc->queue) ||
		(desc->flags & ~knownFlags) || !ValidCount(desc->featureDomains, desc->featureDomainCount) ||
		!ValidCount(desc->after, desc->afterCount) || !ValidCount(desc->before, desc->beforeCount) ||
		!ValidCount(desc->accesses, desc->accessCount)) return ORG_RG_E_INVALID_ARGUMENT;
	const std::string id(desc->id);
	if (!id.starts_with(currentBuild_->contributor.id + ".") || id.starts_with("org.") ||
		std::ranges::find(anchors_, id) != anchors_.end()) return ORG_RG_E_RESERVED_ID;
	if (std::ranges::any_of(currentBuild_->passes, [&](const Pass& p) { return p.id == id; })) return ORG_RG_E_DUPLICATE_ID;
	Pass pass{};
	pass.contributor = currentBuild_->contributor.handle;
	pass.id = id; pass.kind = desc->kind; pass.queue = desc->queue; pass.flags = desc->flags;
	pass.priority = desc->priority; pass.technique = desc->techniquePath ? desc->techniquePath : "";
	pass.prepare = desc->prepare; pass.update = desc->update; pass.execute = desc->execute;
	pass.cleanup = desc->cleanup; pass.userData = currentBuild_->contributor.userData;
	std::unordered_set<uint32_t> bindings;
	for (uint32_t i = 0; i < desc->featureDomainCount; ++i) {
		if (!desc->featureDomains[i] || !*desc->featureDomains[i]) return ORG_RG_E_INVALID_ARGUMENT;
		pass.featureDomains.emplace_back(desc->featureDomains[i]);
	}
	for (uint32_t i = 0; i < desc->afterCount; ++i) {
		if (!IsNamespaced(desc->after[i])) return ORG_RG_E_INVALID_ARGUMENT;
		pass.after.emplace_back(desc->after[i]);
	}
	for (uint32_t i = 0; i < desc->beforeCount; ++i) {
		if (!IsNamespaced(desc->before[i])) return ORG_RG_E_INVALID_ARGUMENT;
		pass.before.emplace_back(desc->before[i]);
	}
	for (uint32_t i = 0; i < desc->accessCount; ++i) {
		const auto& source = desc->accesses[i];
		if (!ValidHeader(&source) || !IsNamespaced(source.resourceId) || !ValidAccessForPass(desc->kind, source.access) ||
			!source.range.mipCount || !source.range.arraySize || source.viewKind < ORG_RG_VIEW_NONE ||
			source.viewKind > ORG_RG_VIEW_DEPTH_STENCIL || source.viewDimension < ORG_RG_VIEW_DIMENSION_DEFAULT ||
			source.viewDimension > ORG_RG_VIEW_DIMENSION_TEXTURE_CUBE_ARRAY || source.viewFormat < ORG_RG_FORMAT_UNKNOWN ||
			source.viewFormat >= ORG_RG_FORMAT_COUNT || (source.viewFlags & ~kKnownViewFlags) || !source.elementCount)
			return ORG_RG_E_INVALID_ARGUMENT;
		if (!bindings.insert(source.binding).second) return ORG_RG_E_DUPLICATE_BINDING;
		if (source.counterBinding && !bindings.insert(source.counterBinding).second)
			return ORG_RG_E_DUPLICATE_BINDING;
		Access access{};
		access.resourceId = source.resourceId; access.binding = source.binding; access.kind = source.access; access.range = source.range;
		access.viewKind = source.viewKind == ORG_RG_VIEW_NONE ? DefaultViewKind(source.access) : source.viewKind;
		access.viewDimension = source.viewDimension; access.viewFormat = source.viewFormat;
		access.viewFlags = source.viewFlags; access.firstElement = source.firstElement;
		access.elementCount = source.elementCount; access.structureByteStride = source.structureByteStride;
		access.counterBinding = source.counterBinding;
		pass.accesses.push_back(std::move(access));
	}
	currentBuild_->passes.push_back(std::move(pass));
	return ORG_RG_OK;
}

ORGStatus ContributorRegistry::RequestRebuild(ORGRegistrationHandle handle) noexcept
{
	std::unique_lock lock(mutex_);
	if (!contributors_.contains(handle) && !unregistering_.contains(handle)) return ORG_RG_E_STALE_HANDLE;
	rebuildRequested_ = true;
	return ORG_RG_OK;
}

ORGStatus ContributorRegistry::GetDiagnostic(ORGRegistrationHandle handle, ORGDiagnostic* out) const noexcept
{
	if (!out || out->structSize < sizeof(*out)) return ORG_RG_E_INVALID_ARGUMENT;
	std::shared_lock lock(mutex_);
	const auto it = diagnostics_.find(handle);
	if (it == diagnostics_.end()) return ORG_RG_E_NOT_READY;
	*out = it->second;
	return ORG_RG_OK;
}

bool ContributorRegistry::HasPendingRebuild() const noexcept
{
	std::shared_lock lock(mutex_);
	return rebuildRequested_;
}

ORGStatus ContributorRegistry::Compile(uint64_t generation, uint32_t renderWidth, uint32_t renderHeight,
	uint32_t outputWidth, uint32_t outputHeight, Candidate& output) noexcept
{
	std::vector<Contributor> snapshot;
	{
		std::unique_lock lock(mutex_);
		rebuildRequested_ = false;
		for (const auto& [_, contributor] : contributors_) snapshot.push_back(contributor);
	}
	std::ranges::sort(snapshot, {}, &Contributor::id);
	Candidate candidate{};
	candidate.generation = generation;
	std::unordered_map<uint64_t, Contributor> included;
	for (const auto& contributor : snapshot) {
		Build staging{};
		staging.handle = nextHandle_++;
		staging.contributor = contributor;
		ORGStatus status = ORG_RG_E_CALLBACK_FAILED;
		{
			std::scoped_lock buildLock(buildMutex_);
			currentBuild_ = &staging;
			buildThread_ = std::this_thread::get_id();
		}
		try { status = contributor.build(contributor.userData, staging.handle); } catch (...) { status = ORG_RG_E_CALLBACK_FAILED; }
		{
			std::scoped_lock buildLock(buildMutex_);
			currentBuild_ = nullptr;
			buildThread_ = {};
		}
		if (status != ORG_RG_OK) {
			SetDiagnostic(contributor.handle, status, 1, generation, "Contributor declaration callback failed");
			if (contributor.kind == ORG_RG_CONTRIBUTOR_REQUIRED) {
				std::unique_lock lock(mutex_); rebuildRequested_ = true; return status;
			}
			continue;
		}
		candidate.resources.insert(candidate.resources.end(), std::make_move_iterator(staging.resources.begin()), std::make_move_iterator(staging.resources.end()));
		candidate.passes.insert(candidate.passes.end(), std::make_move_iterator(staging.passes.begin()), std::make_move_iterator(staging.passes.end()));
		candidate.contributors.push_back(contributor.handle);
		included.emplace(contributor.handle, contributor);
	}

	std::unordered_map<std::string, Resource*> resources;
	for (auto& resource : candidate.resources) {
		if (!resources.emplace(resource.id, &resource).second) {
			SetDiagnostic(resource.contributor, ORG_RG_E_DUPLICATE_ID, 2, generation, "Duplicate resource identifier: " + resource.id);
			std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_DUPLICATE_ID;
		}
		if (resource.desc.dimension != ORG_RG_RESOURCE_BUFFER && resource.desc.sizing != ORG_RG_SIZE_ABSOLUTE) {
			const auto referenceWidth = resource.desc.sizing == ORG_RG_SIZE_RENDER_RELATIVE ? renderWidth : outputWidth;
			const auto referenceHeight = resource.desc.sizing == ORG_RG_SIZE_RENDER_RELATIVE ? renderHeight : outputHeight;
			resource.desc.width = ResolveExtent(resource.desc.widthScale, referenceWidth);
			resource.desc.height = ResolveExtent(resource.desc.heightScale, referenceHeight);
			resource.desc.sizing = ORG_RG_SIZE_ABSOLUTE;
			if (!resource.desc.width || !resource.desc.height) {
				std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INVALID_ARGUMENT;
			}
		}
	}

	std::unordered_set<uint64_t> omitted;
	const std::unordered_set<std::string> anchors(anchors_.begin(), anchors_.end());
	bool changed = true;
	while (changed) {
		changed = false;
		std::unordered_set<std::string> availablePasses;
		for (const auto& pass : candidate.passes) if (!omitted.contains(pass.contributor)) availablePasses.insert(pass.id);
		for (const auto& pass : candidate.passes) {
			if (omitted.contains(pass.contributor)) continue;
			std::string missingOrder;
			for (const auto& dependency : pass.after)
				if (!anchors.contains(dependency) && !availablePasses.contains(dependency)) { missingOrder = dependency; break; }
			if (missingOrder.empty()) for (const auto& dependency : pass.before)
				if (!anchors.contains(dependency) && !availablePasses.contains(dependency)) { missingOrder = dependency; break; }
			if (!missingOrder.empty()) {
				const auto& contributor = included.at(pass.contributor);
				if (contributor.kind == ORG_RG_CONTRIBUTOR_REQUIRED) {
					SetDiagnostic(pass.contributor, ORG_RG_E_MISSING_DEPENDENCY, 2, generation,
						"Missing required ordering dependency: " + missingOrder);
					std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_MISSING_DEPENDENCY;
				}
				omitted.insert(pass.contributor); changed = true;
				SetDiagnostic(pass.contributor, ORG_RG_E_MISSING_DEPENDENCY, 2, generation,
					"Optional contributor omitted; ordering dependency unavailable: " + missingOrder);
				continue;
			}
			for (const auto& access : pass.accesses) {
				const auto resource = resources.find(access.resourceId);
				if (resource == resources.end() || omitted.contains(resource->second->contributor)) {
					const auto& contributor = included.at(pass.contributor);
					if (contributor.kind == ORG_RG_CONTRIBUTOR_REQUIRED) {
						SetDiagnostic(pass.contributor, ORG_RG_E_MISSING_DEPENDENCY, 2, generation, "Missing required resource: " + access.resourceId);
						std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_MISSING_DEPENDENCY;
					}
					omitted.insert(pass.contributor); changed = true;
					SetDiagnostic(pass.contributor, ORG_RG_E_MISSING_DEPENDENCY, 2, generation, "Optional contributor omitted; resource unavailable: " + access.resourceId);
					break;
				}
			}
		}
	}
	std::erase_if(candidate.resources, [&](const Resource& resource) { return omitted.contains(resource.contributor); });
	std::erase_if(candidate.passes, [&](const Pass& pass) { return omitted.contains(pass.contributor); });
	std::erase_if(candidate.contributors, [&](uint64_t handle) { return omitted.contains(handle); });
	resources.clear();
	for (auto& resource : candidate.resources) resources.emplace(resource.id, &resource);
	for (auto& pass : candidate.passes) for (auto& access : pass.accesses) {
		auto* resource = resources.at(access.resourceId);
		if ((resource->desc.allowedUsages & UsageForAccess(access.kind)) == 0) {
			SetDiagnostic(pass.contributor, ORG_RG_E_INCOMPATIBLE_RESOURCE, 2, generation, "Resource usage was not declared: " + access.resourceId);
			std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INCOMPATIBLE_RESOURCE;
		}
		const uint32_t mipCount = access.range.mipCount == UINT32_MAX ? resource->desc.mipLevels - access.range.firstMip : access.range.mipCount;
		const uint32_t slices = resource->desc.dimension == ORG_RG_RESOURCE_BUFFER ? 1u : resource->desc.depthOrArraySize;
		const uint32_t sliceCount = access.range.arraySize == UINT32_MAX ? slices - access.range.firstArraySlice : access.range.arraySize;
		if (access.range.firstMip >= resource->desc.mipLevels || mipCount > resource->desc.mipLevels - access.range.firstMip ||
			access.range.firstArraySlice >= slices || sliceCount > slices - access.range.firstArraySlice) {
			std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INVALID_ARGUMENT;
		}
		const auto expectedView = DefaultViewKind(access.kind);
		if (access.viewKind != expectedView) {
			SetDiagnostic(pass.contributor, ORG_RG_E_INCOMPATIBLE_RESOURCE, 2, generation,
				"View kind does not match access for resource: " + access.resourceId);
			std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INCOMPATIBLE_RESOURCE;
		}
		if (resource->desc.dimension == ORG_RG_RESOURCE_BUFFER) {
			if (access.range.firstMip != 0 || (access.range.mipCount != 1 && access.range.mipCount != UINT32_MAX) ||
				access.range.firstArraySlice != 0 || (access.range.arraySize != 1 && access.range.arraySize != UINT32_MAX) ||
				(access.viewDimension != ORG_RG_VIEW_DIMENSION_DEFAULT && access.viewDimension != ORG_RG_VIEW_DIMENSION_BUFFER) ||
				(access.viewFlags & (ORG_RG_VIEW_FLAG_READ_ONLY_DEPTH | ORG_RG_VIEW_FLAG_READ_ONLY_STENCIL))) {
				std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INCOMPATIBLE_RESOURCE;
			}
			access.viewDimension = ORG_RG_VIEW_DIMENSION_BUFFER;
			if (expectedView == ORG_RG_VIEW_NONE) {
				if (access.viewFormat != ORG_RG_FORMAT_UNKNOWN || access.viewFlags || access.firstElement ||
					access.elementCount != UINT32_MAX || access.structureByteStride) {
					std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INVALID_ARGUMENT;
				}
			} else if (expectedView == ORG_RG_VIEW_CONSTANT_BUFFER) {
				const uint64_t bytes = access.elementCount == UINT32_MAX ? resource->desc.byteSize - access.firstElement : access.elementCount;
				if (access.viewFormat != ORG_RG_FORMAT_UNKNOWN || access.viewFlags || access.structureByteStride ||
					access.firstElement > resource->desc.byteSize || bytes > resource->desc.byteSize - access.firstElement ||
					bytes > UINT32_MAX || (access.firstElement & 255u) || (bytes & 255u)) {
					std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INCOMPATIBLE_RESOURCE;
				}
			} else {
				const bool raw = (access.viewFlags & ORG_RG_VIEW_FLAG_RAW_BUFFER) != 0;
				if (access.counterBinding && (expectedView != ORG_RG_VIEW_UNORDERED_ACCESS || raw ||
					!resource->desc.structureByteStride)) {
					std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INCOMPATIBLE_RESOURCE;
				}
				if (raw && (access.structureByteStride || access.viewFormat != ORG_RG_FORMAT_UNKNOWN || (resource->desc.byteSize & 3u))) {
					std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INCOMPATIBLE_RESOURCE;
				}
				if (!raw) {
					if (!access.structureByteStride) access.structureByteStride = resource->desc.structureByteStride;
					if (resource->desc.structureByteStride && access.structureByteStride != resource->desc.structureByteStride) {
						std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INCOMPATIBLE_RESOURCE;
					}
					if (!access.structureByteStride && access.viewFormat == ORG_RG_FORMAT_UNKNOWN) {
						std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INCOMPATIBLE_RESOURCE;
					}
				}
				const uint64_t stride = raw ? 4u : access.structureByteStride ? access.structureByteStride : 1u;
				const uint64_t elements = resource->desc.byteSize / stride;
				const uint64_t count = access.elementCount == UINT32_MAX ? elements - access.firstElement : access.elementCount;
				if (access.firstElement > elements || count > elements - access.firstElement || count > UINT32_MAX) {
					std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INVALID_ARGUMENT;
				}
			}
		} else {
			if (access.firstElement || access.elementCount != UINT32_MAX || access.structureByteStride ||
				(access.viewFlags & ORG_RG_VIEW_FLAG_RAW_BUFFER) || access.counterBinding) {
				std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INCOMPATIBLE_RESOURCE;
			}
			if (expectedView == ORG_RG_VIEW_NONE && (access.viewFormat != ORG_RG_FORMAT_UNKNOWN || access.viewFlags)) {
				std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INVALID_ARGUMENT;
			}
			if (expectedView != ORG_RG_VIEW_DEPTH_STENCIL &&
				(access.viewFlags & (ORG_RG_VIEW_FLAG_READ_ONLY_DEPTH | ORG_RG_VIEW_FLAG_READ_ONLY_STENCIL))) {
				std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INCOMPATIBLE_RESOURCE;
			}
			if (access.viewDimension == ORG_RG_VIEW_DIMENSION_DEFAULT) {
				if (resource->desc.dimension == ORG_RG_RESOURCE_TEXTURE_1D)
					access.viewDimension = slices > 1 ? ORG_RG_VIEW_DIMENSION_TEXTURE_1D_ARRAY : ORG_RG_VIEW_DIMENSION_TEXTURE_1D;
				else if (resource->desc.dimension == ORG_RG_RESOURCE_TEXTURE_3D)
					access.viewDimension = ORG_RG_VIEW_DIMENSION_TEXTURE_3D;
				else if (resource->desc.sampleCount > 1)
					access.viewDimension = slices > 1 ? ORG_RG_VIEW_DIMENSION_TEXTURE_2D_MS_ARRAY : ORG_RG_VIEW_DIMENSION_TEXTURE_2D_MS;
				else access.viewDimension = slices > 1 ? ORG_RG_VIEW_DIMENSION_TEXTURE_2D_ARRAY : ORG_RG_VIEW_DIMENSION_TEXTURE_2D;
			}
			const bool validDimension =
				(resource->desc.dimension == ORG_RG_RESOURCE_TEXTURE_1D &&
					(access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_1D || access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_1D_ARRAY)) ||
				(resource->desc.dimension == ORG_RG_RESOURCE_TEXTURE_2D &&
					(access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_2D || access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_2D_ARRAY ||
					 access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_2D_MS || access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_2D_MS_ARRAY ||
					 access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_CUBE || access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_CUBE_ARRAY)) ||
				(resource->desc.dimension == ORG_RG_RESOURCE_TEXTURE_3D && access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_3D);
			if (!validDimension) {
				std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INCOMPATIBLE_RESOURCE;
			}
			if (resource->desc.sampleCount > 1 && access.viewDimension != ORG_RG_VIEW_DIMENSION_TEXTURE_2D_MS &&
				access.viewDimension != ORG_RG_VIEW_DIMENSION_TEXTURE_2D_MS_ARRAY) {
				std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INCOMPATIBLE_RESOURCE;
			}
			if (resource->desc.sampleCount > 1 && expectedView == ORG_RG_VIEW_UNORDERED_ACCESS) {
				std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_UNSUPPORTED_CAPABILITY;
			}
			if ((access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_CUBE ||
				access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_CUBE_ARRAY) &&
				expectedView != ORG_RG_VIEW_SHADER_RESOURCE) {
				std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INCOMPATIBLE_RESOURCE;
			}
			if (resource->desc.dimension == ORG_RG_RESOURCE_TEXTURE_3D && expectedView == ORG_RG_VIEW_DEPTH_STENCIL) {
				std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INCOMPATIBLE_RESOURCE;
			}
			if ((expectedView == ORG_RG_VIEW_RENDER_TARGET || expectedView == ORG_RG_VIEW_DEPTH_STENCIL ||
				expectedView == ORG_RG_VIEW_UNORDERED_ACCESS) && mipCount != 1) {
				std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INCOMPATIBLE_RESOURCE;
			}
			if (access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_CUBE &&
				(access.range.firstArraySlice != 0 || sliceCount != 6) ||
				access.viewDimension == ORG_RG_VIEW_DIMENSION_TEXTURE_CUBE_ARRAY &&
				((access.range.firstArraySlice % 6) || (sliceCount % 6))) {
				std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_INCOMPATIBLE_RESOURCE;
			}
		}
		access.resource = resource->handle;
	}

	std::unordered_set<std::string> passIds;
	for (const auto& pass : candidate.passes) if (!passIds.insert(pass.id).second) {
		std::unique_lock lock(mutex_); rebuildRequested_ = true; return ORG_RG_E_DUPLICATE_ID;
	}
	// Ordering and cycle detection intentionally remain authoritative in ORG's
	// dependency compiler. The contributor layer only validates symbolic inputs.
	output = std::move(candidate);
	return ORG_RG_OK;
}

void ContributorRegistry::Activate(const Candidate& candidate) noexcept
{
	std::vector<Contributor> callbacks;
	{
		std::unique_lock lock(mutex_);
		generationContributors_[candidate.generation] = candidate.contributors;
		for (const auto handle : candidate.contributors) {
			auto it = contributors_.find(handle);
			if (it == contributors_.end()) continue;
			if (it->second.activeGenerations.insert(candidate.generation).second) callbacks.push_back(it->second);
		}
	}
	for (const auto& contributor : callbacks) if (contributor.activated)
		try { contributor.activated(contributor.userData, candidate.generation); } catch (...) {
			SetDiagnostic(contributor.handle, ORG_RG_E_CALLBACK_FAILED, 3, candidate.generation, "Activation callback threw an exception");
		}
}

void ContributorRegistry::Retire(uint64_t generation) noexcept
{
	std::vector<Contributor> callbacks;
	{
		std::unique_lock lock(mutex_);
		const auto generationIt = generationContributors_.find(generation);
		if (generationIt == generationContributors_.end()) return;
		for (const auto handle : generationIt->second) {
			auto active = contributors_.find(handle);
			auto removing = unregistering_.find(handle);
			Contributor* contributor = active != contributors_.end() ? &active->second :
				removing != unregistering_.end() ? &removing->second : nullptr;
			if (contributor && contributor->activeGenerations.erase(generation)) callbacks.push_back(*contributor);
		}
		generationContributors_.erase(generationIt);
		std::erase_if(unregistering_, [](const auto& entry) { return entry.second.activeGenerations.empty(); });
	}
	for (const auto& contributor : callbacks) if (contributor.retired)
		try { contributor.retired(contributor.userData, generation); } catch (...) {
			SetDiagnostic(contributor.handle, ORG_RG_E_CALLBACK_FAILED, 4, generation, "Retirement callback threw an exception");
		}
}

void ContributorRegistry::NotifyDeviceLost(uint32_t reason) noexcept
{
	std::vector<Contributor> snapshot;
	{ std::shared_lock lock(mutex_); for (const auto& [_, value] : contributors_) snapshot.push_back(value); }
	for (const auto& contributor : snapshot) if (contributor.deviceLost)
		try { contributor.deviceLost(contributor.userData, reason); } catch (...) {
			SetDiagnostic(contributor.handle, ORG_RG_E_CALLBACK_FAILED, 5, 0, "Device-loss callback threw an exception");
		}
}

void ContributorRegistry::Shutdown() noexcept
{
	std::vector<Contributor> snapshot;
	{
		std::unique_lock lock(mutex_); open_ = false;
		for (const auto& [_, value] : contributors_) snapshot.push_back(value);
		for (const auto& [_, value] : unregistering_) snapshot.push_back(value);
		contributors_.clear(); unregistering_.clear(); generationContributors_.clear();
	}
	for (const auto& contributor : snapshot) {
		if (contributor.retired) for (const auto generation : contributor.activeGenerations)
			try { contributor.retired(contributor.userData, generation); } catch (...) {}
		if (contributor.shutdown) try { contributor.shutdown(contributor.userData); } catch (...) {}
	}
}

} // namespace org::contributor
