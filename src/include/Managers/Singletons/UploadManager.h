#pragma once
#include <wrl/client.h>
#include <atomic>
#include <vector>
#include <memory>
#include <iterator>
#include <functional>
#include <mutex>
#include <rhi.h>
#include <string>
#include <thread>
#include <condition_variable>
#include <deque>
#include <stacktrace>

#include "rhi_helpers.h"
#include "Render/ResourceRegistry.h"
#include "Interfaces/IDynamicDeclaredResources.h"
#include "RenderPasses/Base/RenderPass.h"
#include "Resources/Buffers/Buffer.h"
#include "Render/ImmediateExecution/ImmediateCommandList.h"
#include "Render/Runtime/UploadTypes.h"
#include "Render/Runtime/StreamingUploadTypes.h"
#include "Render/PassBuilders.h"
#include "Managers/AsyncCopyPagePool.h"
#include "Managers/UploadInstance.h"

namespace org {

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
	using UploadResolveContext = org::runtime::UploadResolveContext;
	using UploadTarget = org::runtime::UploadTarget;

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
	void ProcessUploads(uint8_t frameIndex, org::imm::ImmediateCommandList& commandList);
	void QueueResourceCopy(const std::shared_ptr<Resource>& destination, const std::shared_ptr<Resource>& source, size_t size);
	void ExecuteResourceCopies(uint8_t frameIndex, org::imm::ImmediateCommandList& commandList);
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
	std::shared_ptr<TrackedUploadTicket> QueueTrackedStreamingUpload(
		const void* data, size_t size, std::shared_ptr<Resource> destination,
		size_t dstOffset = 0);

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
			(void)builder;
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

		bool RequiresPassRebindAfterDeclarationRefresh() const noexcept override { return false; }
		bool DeclarationsProvidedByImmediateCommands() const noexcept override { return false; }

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
	void CaptureResourceCopyTelemetry(ResourceCopy& copy);
	void RefreshQueuedCopyTelemetryLocked();
	bool IsUploadTargetValid(const UploadTarget& target, const char* reason, const char* file, int line);
	void CaptureUploadTargetTelemetry(const UploadTarget& target, uint64_t& outId, std::string& outName);
	std::shared_ptr<TrackedUploadTicket> SubmitStreamingUpload(
		const void* data, size_t size, std::shared_ptr<Resource> destination,
		size_t dstOffset, bool exposeTicket);
	void RunStreamingCompletionWorker(std::stop_token stopToken);

	struct SubmittedStreamingBatch {
		rhi::CommandAllocatorPtr allocator;
		rhi::CommandListPtr commandList;
		std::vector<StreamingUploadDescriptor> descriptors;
		uint64_t timelineValue = 0;
	};

	uint8_t m_numFramesInFlight = 0;

	std::vector<ResourceCopy> queuedResourceCopies;
	std::mutex m_uploadQueueMutex;

	UploadResolveContext m_ctx{};
	std::shared_ptr<UploadPass> m_uploadPass;
	std::unique_ptr<UploadInstance> m_uploadInstance;

	// ── Streaming upload (copy-queue) state ─────────────────────────────
	AsyncCopyPagePool                     m_streamingPagePool;
	std::mutex                            m_streamingMutex;
	std::condition_variable_any           m_streamingCv;
	std::shared_ptr<rhi::TimelinePtr>     m_streamingTimeline;
	std::deque<StreamingUploadDescriptor> m_pendingStreamingUploads;
	std::jthread                          m_streamingCompletionWorker;
	uint64_t                              m_nextStreamingTimelineValue = 0;
	bool                                  m_streamingInitialized = false;

};

inline UploadManager& UploadManager::GetInstance() {
	static UploadManager instance;
	return instance;
}

} // namespace org

#include "Render/Runtime/UploadServiceAccess.h"
