#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <typeindex>
#include <rhi.h>
#include <DirectXMath.h>

#include "Render/ImmediateExecution/ImmediateCommandList.h"
#include "Render/QueueKind.h"
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
	UINT64 frameFenceValue = 0;
	float deltaTime = 0.0f;
	const IHostExecutionData* hostData = nullptr;
	std::function<void()> beforeCompileFrame;
};

struct ImmediateExecutionContext {
	rhi::Device device;
	org::imm::ImmediateCommandList list;
	UINT frameIndex = 0;
	const IHostExecutionData* hostData = nullptr;
};

struct IHasImmediateModeCommands {
	virtual ~IHasImmediateModeCommands() = default;
	virtual void RecordImmediateCommands(ImmediateExecutionContext& context) = 0;
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
	UINT64 frameFenceValue = 0;
	float deltaTime = 0.0f;
	const IHostExecutionData* hostData = nullptr;
	// Values for structurally placed external-wait bindings. Bindings determine
	// the consuming batch/queue at compile time; timeline values remain per-frame.
	std::vector<ExternalTimelineBindingValue> externalTimelineBindings;
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
