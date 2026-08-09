#pragma once

#include <cstdint>
#include <functional>
#include <typeindex>
#include <rhi.h>
#include <DirectXMath.h>

#include "Render/ImmediateExecution/ImmediateCommandList.h"
#include "RenderPasses/Base/PassReturn.h"


namespace org {

struct IHostExecutionData {
	virtual ~IHostExecutionData() = default;
	virtual const void* TryGet(std::type_index t) const noexcept = 0;

	template<class T>
	const T* Get() const noexcept {
		return static_cast<const T*>(TryGet(std::type_index(typeid(T))));
	}
};

struct UpdateExecutionContext {
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
};


} // namespace org
