#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <typeindex>
#include <rhi.h>
#include <DirectXMath.h>

#include "Render/ImmediateExecution/ImmediateCommandList.h"
#include "Render/QueueKind.h"
#include "Render/ExternalBindings.h"
#include "RenderPasses/Base/PassReturn.h"


namespace org {

class Resource;
class SymbolicTracker;
class GloballyIndexedResource;
class PipelineState;
class ResolverCaptureContext;

struct IHostExecutionData {
	virtual ~IHostExecutionData() = default;
	virtual const void* TryGet(std::type_index t) const noexcept = 0;

	template<class T>
	const T* Get() const noexcept {
		return static_cast<const T*>(TryGet(std::type_index(typeid(T))));
	}
};

struct UpdateExecutionContext {
    // Owned publication context supplied by the preparation owner for this update.
    std::shared_ptr<const ResolverCaptureContext> resolverCaptureContext;
	UINT frameIndex = 0;
	// CPU-owned slot whose mutable frame data is reserved by this logical
	// frame. It is deliberately distinct from the swapchain image acquired at
	// ordered admission. frameIndex remains an alias during migration.
	UINT preparationSlot = 0;
	UINT64 frameFenceValue = 0;
	float deltaTime = 0.0f;
	const IHostExecutionData* hostData = nullptr;
	// Optional ownership for hostData when asynchronous admission may prepare
	// recording packets after Update returns.
	std::shared_ptr<const IHostExecutionData> ownedHostData;
	std::function<void()> beforeCompileFrame;
};

struct ImmediateExecutionContext {
	rhi::Device device;
	org::imm::ImmediateCommandList list;
	UINT frameIndex = 0;
	const IHostExecutionData* hostData = nullptr;
};

// An immediate pass may move its post-submit side effect into an owned packet.
// The signal is emitted by the packet's actual queue before commit is invoked;
// compilation never observes or consumes this state.
struct OwnedImmediateSubmissionEffect {
	std::vector<ExternalTimelinePoint> completionSignals;
	std::function<void()> commit;
};

struct IHasImmediateModeCommands {
	virtual ~IHasImmediateModeCommands() = default;
	virtual void RecordImmediateCommands(ImmediateExecutionContext& context) = 0;
	// True when Execute contributes no additional commands, waits, signals, or
	// submission effects. This lets owned preparation represent an empty
	// immediate stream as an explicit no-op without exposing compiler details.
	virtual bool ImmediateCommandsAreCompleteExecution() const noexcept { return false; }
	virtual std::optional<OwnedImmediateSubmissionEffect> TakeOwnedImmediateSubmissionEffect() { return std::nullopt; }
};

struct PassExecutionContext {
	rhi::Device device;
	BackendInstanceId backendInstance = BackendInstanceId::Primary;
	rhi::CommandList commandList;
	const org::imm::ImmediateDispatch* immediateDispatch = nullptr;
	std::function<void(rhi::CommandList, rhi::Queue, const char*, const char*)> beginGpuPassRange;
	std::function<void(rhi::CommandList, rhi::Queue)> endGpuPassRange;
	const char* currentPassName = nullptr;
	const char* currentTechniquePath = nullptr;
	UINT frameIndex = 0;
	// Execution-slot identity selected at admission. This is not necessarily
	// the preparation slot captured by the executable frame.
	UINT executionSlot = 0;
	UINT64 frameFenceValue = 0;
	float deltaTime = 0.0f;
	const IHostExecutionData* hostData = nullptr;
	// Retains hostData whenever execution or recording may outlive the caller's
	// stack frame. Code receiving hostData must treat the publication as
	// immutable and must not retain the raw pointer without this owner.
	std::shared_ptr<const IHostExecutionData> ownedHostData;
	// Values for structurally placed external-wait bindings. Bindings determine
	// the consuming batch/queue at compile time; timeline values remain per-frame.
	std::vector<ExternalTimelineBindingValue> externalTimelineBindings;
	std::vector<ExternalResourceBindingValue> externalResourceBindings;
	std::vector<ExternalDescriptorBindingValue> externalDescriptorBindings;
	rhi::Resource Resolve(Resource& resource) const;
	rhi::Resource Resolve(const std::shared_ptr<Resource>& resource) const;
	SymbolicTracker* ResolveState(Resource& resource) const;
	rhi::DescriptorHeap GetResourceDescriptorHeap() const;
	rhi::DescriptorHeap GetSamplerDescriptorHeap() const;
	rhi::DescriptorSlot ResolveSRV(const GloballyIndexedResource& resource, uint32_t mip = 0, uint32_t slice = 0) const;
	rhi::DescriptorSlot ResolveUAV(const GloballyIndexedResource& resource, uint32_t mip = 0, uint32_t slice = 0) const;
	rhi::DescriptorSlot ResolveCBV(const GloballyIndexedResource& resource) const;
	rhi::DescriptorSlot ResolveRTV(const GloballyIndexedResource& resource, uint32_t mip = 0, uint32_t slice = 0) const;
	rhi::DescriptorSlot ResolveDSV(const GloballyIndexedResource& resource, uint32_t mip = 0, uint32_t slice = 0) const;
	rhi::DescriptorSlot ResolveNonShaderVisibleUAV(const GloballyIndexedResource& resource, uint32_t mip = 0, uint32_t slice = 0) const;
	const rhi::Pipeline& ResolvePipeline(const PipelineState& pipeline) const;
};


} // namespace org
