#pragma once

#include <span>
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
#include "Managers/UploadInstance.h"
#include "Render/Runtime/CopyQueueUploadService.h"

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
	void UploadDataBatch(UploadTarget resourceToUpdate, std::span<const org::runtime::UploadRegion> regions, const char* file, int line);
	void PostTextureSubresources(UploadTarget target, rhi::Format fmt, uint32_t baseWidth, uint32_t baseHeight,
		uint32_t depthOrLayers, uint32_t mipLevels, uint32_t arraySize,
		std::shared_ptr<const std::vector<rhi::helpers::SubresourceData>> subresources,
		std::shared_ptr<const void> keepAlive, const char* file, int line);
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
	void UploadDataBatch(UploadTarget resourceToUpdate, std::span<const org::runtime::UploadRegion> regions);
	void PostTextureSubresources(UploadTarget target, rhi::Format fmt, uint32_t baseWidth, uint32_t baseHeight,
		uint32_t depthOrLayers, uint32_t mipLevels, uint32_t arraySize,
		std::shared_ptr<const std::vector<rhi::helpers::SubresourceData>> subresources,
		std::shared_ptr<const void> keepAlive);
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
	void SetOwnerThread();
	void QueueResourceCopy(const std::shared_ptr<Resource>& destination, const std::shared_ptr<Resource>& source, size_t size);
	void ExecuteResourceCopies(uint8_t frameIndex, org::imm::ImmediateCommandList& commandList);
	void ProcessDeferredReleases(uint8_t frameIndex);
	void SubmitStagedUploads(std::shared_ptr<org::runtime::StagedUploadBatch> batch);
	void SetStagedUploadsRecordedDirectly(bool direct);
	size_t RecordStagedUploads(rhi::CommandList& list, uint8_t frameIndex, bool afterCopies);
	void SetUploadResolveContext(UploadResolveContext ctx);
	std::shared_ptr<RenderPass> GetUploadPass() const { return m_uploadPass; }
	std::string DescribeQueuedTargetByGlobalResourceId(uint64_t globalResourceId);

	// ── Worker upload path (copy queue) ─────────────────────────────────
	std::shared_ptr<TrackedUploadTicket> QueueTrackedStreamingUploadSegments(
		std::span<const StreamingUploadSegment> segments, size_t totalSize,
		WorkerOwnedDestination destination, size_t dstOffset = 0);
	org::runtime::CopyQueueUploadService* CopyQueueUploads() { return m_copyQueueUploads.get(); }

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
		bool ImmediateCommandsAreCompleteExecution() const noexcept override { return true; }

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

	uint8_t m_numFramesInFlight = 0;

	std::vector<ResourceCopy> queuedResourceCopies;
	std::mutex m_uploadQueueMutex;

	UploadResolveContext m_ctx{};
	std::shared_ptr<UploadPass> m_uploadPass;
	std::unique_ptr<UploadInstance> m_uploadInstance;
	bool m_stagedUploadsDirect = false;  // applied to every instance (SetStagedUploadsRecordedDirectly)

	// ── Worker upload path ──────────────────────────────────────────────
	std::unique_ptr<org::runtime::CopyQueueUploadService> m_copyQueueUploads;

};

inline UploadManager& UploadManager::GetInstance() {
	static UploadManager instance;
	return instance;
}

} // namespace org

#include "Render/Runtime/UploadServiceAccess.h"
