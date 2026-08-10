#pragma once

#include <OpenRenderGraph/ContributorAPI.h>
#include <OpenRenderGraph/ContributorGraphHost.h>
#include <OpenRenderGraph/ContributorRegistry.h>
#include <OpenRenderGraph/ExtensionRegistry.h>

#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace org::contributor {

class Runtime
{
public:
	using ServiceQuery = std::function<ORGStatus(uint32_t, void*, uint32_t)>;
	struct HostStorage { std::string id, displayName, version; std::vector<std::pair<std::string, std::string>> anchors; };

	struct RuntimeDesc
	{
		rhi::Device device{};
		ORGBackend backend{ ORG_RG_BACKEND_NONE };
		uint32_t framesInFlight{ 3 };
		ORGHostDescriptor host{};
		std::function<uint64_t()> completedValue;
	};

	struct FrameDesc
	{
		uint64_t frameIndex{};
		uint64_t frameFenceValue{};
		uint32_t renderWidth{};
		uint32_t renderHeight{};
		uint32_t outputWidth{};
		uint32_t outputHeight{};
		rhi::Timeline readyTimeline{};
		uint64_t readyValue{};
		rhi::Timeline completeTimeline{};
		uint64_t completeValue{};
	};

	explicit Runtime(RuntimeDesc desc);
	~Runtime();
	Runtime(const Runtime&) = delete;
	Runtime& operator=(const Runtime&) = delete;

	ORGStatus GetAPI(uint32_t version, ORGRenderGraphAPI* out) noexcept;
	ExtensionRegistry::Handle RegisterExtension(ExtensionRegistry::Descriptor descriptor);
	void BeginUnregisterExtension(ExtensionRegistry::Handle handle);
	void RegisterService(std::string id, ServiceQuery query);
	void UnregisterService(std::string_view id);
	void RequestRebuild() noexcept { rebuildRequested_.store(true, std::memory_order_release); }
	bool Execute(const FrameDesc& frame) noexcept;
	void Retire(uint64_t completedValue) noexcept;
	void NotifyDeviceLost(uint32_t reason) noexcept;
	void Shutdown() noexcept;

	rhi::Device GetDevice() const noexcept { return desc_.device; }
	uint64_t GetActiveGeneration() const noexcept;
	uint64_t GetLastSubmittedCompletion() const noexcept { return lastSubmittedCompletion_.load(std::memory_order_acquire); }
	uint32_t GetFramesInFlight() const noexcept { return desc_.framesInFlight; }
	bool IsAvailable() const noexcept { return available_.load(std::memory_order_acquire); }
	ContributorRegistry& Registry() noexcept { return registry_; }
	const HostStorage& Host() const noexcept { return host_; }
	uint64_t GetCompletedValue() const noexcept { return CompletedValue(); }
	ORGBackend GetBackend() const noexcept { return desc_.backend; }
	ORGStatus QueryRegisteredService(std::string_view id, uint32_t version, void* out, uint32_t size) noexcept;
	static Runtime* GetCurrentForExport() noexcept { return Current(); }

private:
	struct Generation { uint64_t id{}; ContributorRegistry::Candidate candidate; uint64_t lastCompletion{}; };

	bool Rebuild(const FrameDesc& frame) noexcept;
	uint64_t CompletedValue() const noexcept;
	static Runtime* Current() noexcept;
	static void SetCurrent(Runtime*) noexcept;

	RuntimeDesc desc_;
	HostStorage host_;
	ContributorRegistry registry_;
	ExtensionRegistry extensions_;
	std::unique_ptr<ContributorGraphHost> graph_;
	std::unique_ptr<Generation> active_;
	std::deque<std::pair<uint64_t, std::unique_ptr<Generation>>> retired_;
	std::unordered_map<std::string, ServiceQuery> services_;
	mutable std::mutex servicesMutex_;
	std::atomic_bool available_{ true };
	std::atomic_bool rebuildRequested_{ true };
	std::atomic_uint64_t nextGeneration_{ 1 };
	std::atomic_uint64_t activeGeneration_{};
	std::atomic_uint64_t lastSubmittedCompletion_{};
	uint32_t renderWidth_{}, renderHeight_{}, outputWidth_{}, outputHeight_{};
};

} // namespace org::contributor
