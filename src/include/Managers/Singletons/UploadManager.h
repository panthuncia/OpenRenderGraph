#pragma once
#include <wrl/client.h>
#include <atomic>
#include <vector>
#include <memory>
#include <functional>
#include <mutex>
#include <rhi.h>
#include <string>
#include <thread>
#include <stacktrace>

#include "rhi_helpers.h"
#include "Render/ResourceRegistry.h"
#include "Interfaces/IDynamicDeclaredResources.h"
#include "RenderPasses/Base/RenderPass.h"
#include "Resources/Buffers/Buffer.h"
#include "Render/ImmediateExecution/ImmediateCommandList.h"
#include "Render/Runtime/UploadTypes.h"
#include "Render/Runtime/StreamingUploadTypes.h"
#include "Managers/AsyncCopyPagePool.h"
#include "Managers/UploadInstance.h"

class Buffer;
class Resource;
class ExternalBackingResource;

struct ResourceCopy {
	std::shared_ptr<Resource> source;
	std::shared_ptr<Resource> destination;
	size_t size;
	uint64_t sourceGlobalResourceId = 0;
	uint64_t destinationGlobalResourceId = 0;
	std::string sourceDebugName;
	std::string destinationDebugName;
};

class UploadManager {
public:
	using UploadResolveContext = rg::runtime::UploadResolveContext;
	using UploadTarget = rg::runtime::UploadTarget;

	static UploadManager& GetInstance();
	void Initialize();
#if BUILD_TYPE == BUILD_TYPE_DEBUG
	void UploadData(const void* data, size_t size, UploadTarget resourceToUpdate, size_t dataBufferOffset, const char* file, int line);
	void UploadTextureSubresources(
		UploadTarget target,
		rhi::Format fmt,
		uint32_t baseWidth,
		uint32_t baseHeight,
		uint32_t depthOrLayers,
		uint32_t mipLevels,
		uint32_t arraySize,
		const rhi::helpers::SubresourceData* srcSubresources,
		uint32_t srcCount,
		const char* file,
		int line);
#else
	void UploadData(const void* data, size_t size, UploadTarget resourceToUpdate, size_t dataBufferOffset);
	void UploadTextureSubresources(
		UploadTarget target,
		rhi::Format fmt,
		uint32_t baseWidth,
		uint32_t baseHeight,
		uint32_t depthOrLayers,
		uint32_t mipLevels,
		uint32_t arraySize,
		const rhi::helpers::SubresourceData* srcSubresources,
		uint32_t srcCount);
#endif	
	void ProcessUploads(uint8_t frameIndex, rg::imm::ImmediateCommandList& commandList);
	void QueueResourceCopy(const std::shared_ptr<Resource>& destination, const std::shared_ptr<Resource>& source, size_t size);
	void ExecuteResourceCopies(uint8_t frameIndex, rg::imm::ImmediateCommandList& commandList);
	void ProcessDeferredReleases(uint8_t frameIndex);
	void SetUploadResolveContext(UploadResolveContext ctx);
	std::shared_ptr<RenderPass> GetUploadPass() const { return m_uploadPass; }
	std::string DescribeQueuedTargetByGlobalResourceId(uint64_t globalResourceId);

	// ── Streaming upload API (copy-queue path) ──────────────────────────
	/// Queue a streaming upload that will be executed on the copy queue
	/// via StreamingUploadPass. The data is copied into a dedicated
	/// upload-heap page pool (AsyncCopyPagePool) immediately.
	/// Thread-safe.
	void QueueStreamingUpload(const void* data, size_t size,
	                          std::shared_ptr<Resource> destination,
	                          size_t dstOffset = 0);

	/// Drain all pending streaming uploads. Returns the descriptors to be
	/// fed into a StreamingUploadPass. Called once per frame by the
	/// extension that creates the pass.
	std::vector<StreamingUploadDescriptor> ConsumeStreamingUploads();

	/// Reset the streaming page pool for the next frame. Should be called
	/// once the GPU is done with the previous frame's streaming uploads.
	void ResetStreamingPagePool() { m_streamingPagePool.ResetForFrame(); }

	void Cleanup();
private:

	class UploadPass : public RenderPass, public IDynamicDeclaredResources, public IHasImmediateModeCommands {
	public:
		UploadPass() {
		}

		void DeclareResourceUsages(RenderPassBuilder* builder) override {
			GetInstance().DeclareUploadPassResourceUsages(builder);
		}

		void Setup() override {

		}

		void RecordImmediateCommands(ImmediateExecutionContext& context) override {
			GetInstance().ExecuteResourceCopies(context.frameIndex, context.list);// copies come before uploads to avoid overwriting data
			GetInstance().ProcessUploads(context.frameIndex, context.list);
		}

		PassReturn Execute(PassExecutionContext& context) override {
			return {};
		}

		void Cleanup() override {
			// Cleanup if necessary
		}

		bool DeclaredResourcesChanged() const override {
			return m_declaredResourcesDirty.exchange(false);
		}

		void MarkDeclaredResourcesDirty() {
			m_declaredResourcesDirty.store(true);
		}

	private:
		mutable std::atomic_bool m_declaredResourcesDirty = true;

	};

	UploadManager() {
		m_uploadPass = std::make_shared<UploadPass>();
	}
	void MarkUploadPassDirty();
	void DeclareUploadPassResourceUsages(RenderPassBuilder* builder);
	void CaptureResourceCopyTelemetry(ResourceCopy& copy);
	void RefreshQueuedCopyTelemetryLocked();
	bool IsUploadTargetValid(const UploadTarget& target, const char* reason, const char* file, int line);
	void CaptureUploadTargetTelemetry(const UploadTarget& target, uint64_t& outId, std::string& outName);

	uint8_t m_numFramesInFlight = 0;

	std::vector<ResourceCopy> queuedResourceCopies;
	std::mutex m_uploadQueueMutex;

	UploadResolveContext m_ctx{};
	std::shared_ptr<UploadPass> m_uploadPass;
	std::unique_ptr<UploadInstance> m_uploadInstance;

	// ── Streaming upload (copy-queue) state ─────────────────────────────
	AsyncCopyPagePool                     m_streamingPagePool;
	std::mutex                            m_streamingMutex;
	std::vector<StreamingUploadDescriptor> m_pendingStreamingUploads;

};

inline UploadManager& UploadManager::GetInstance() {
	static UploadManager instance;
	return instance;
}

#include "Render/Runtime/UploadServiceAccess.h"
