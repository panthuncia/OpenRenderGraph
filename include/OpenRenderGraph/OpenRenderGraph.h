#pragma once

// Core user-facing API surface for OpenRenderGraph.
// Include this first for typical pass authoring and graph composition.

#include "../Render/RenderGraph/RenderGraph.h"
#include "../Render/PassBuilders.h"
#include "../Render/PassInputs.h"
#include "../Render/PassExecutionContext.h"
#include "../Render/QueueKind.h"
#include "../Render/DescriptorHeap.h"
#include "../Render/ImmediateExecution/ImmediateCommandList.h"
#include "../Render/CommandListPool.h"
#include "../Render/MemoryIntrospectionAPI.h"
#include "../Render/MemoryIntrospectionBackend.h"
#include "../Render/RenderGraph/Aliasing/RenderGraphAliasingSubsystem.h"

#include "../RenderPasses/Base/RenderPass.h"
#include "../RenderPasses/Base/ComputePass.h"
#include "../RenderPasses/Base/CopyPass.h"
#include "../RenderPasses/Base/PassReturn.h"

#include "../Interfaces/IResourceProvider.h"
#include "../Interfaces/IResourceResolver.h"
#include "../Interfaces/IDynamicDeclaredResources.h"

#include "../Resources/ResourceIdentifier.h"
#include "../Resources/Resource.h"
#include "../Resources/TextureDescription.h"
#include "../Resources/Buffers/Buffer.h"
#include "../Resources/PixelBuffer.h"

#include "../Render/Runtime/IStatisticsService.h"
#include "../Render/Runtime/IUploadService.h"
#include "../Render/Runtime/IUploadPolicyService.h"
#include "../Render/Runtime/IReadbackService.h"
#include "../Render/Runtime/IRenderGraphSettingsService.h"
#include "../Render/Runtime/BufferUploadPolicy.h"
#include "../Render/Runtime/UploadServiceAccess.h"
#include "../Render/Runtime/UploadPolicyServiceAccess.h"
#include "../Render/Runtime/UploadTypes.h"

// Transitional aliases for downstream code that has not yet qualified the
// OpenRenderGraph public API after its move into namespace org.
using namespace org;
using org::AsyncBufferBackingResizeState;
using org::Buffer;
using org::BufferBase;
using org::CopyPass;
using org::ComputePass;
using org::ComputePassBuilder;
using org::DescriptorHeap;
using org::DynamicGloballyIndexedResource;
using org::DynamicResource;
using org::EntityComponentBundle;
using org::GloballyIndexedResource;
using org::GpuBufferBacking;
using org::IDeferredBackingResizeClient;
using org::IHasMemoryMetadata;
using org::IDynamicDeclaredResources;
using org::PipelineResources;
using org::PipelineState;
using org::PipelineStatePayload;
using org::PixelBuffer;
using org::RenderGraph;
using org::RenderPass;
using org::Resource;
using org::TextureDescription;
using org::ViewedDynamicBufferBase;
