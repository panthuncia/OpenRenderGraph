#pragma once
#include <condition_variable>
#include <memory>
#include <vector>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <rhi.h>

#include "Render/QueueKind.h"
#include "Resources/ReadbackRequest.h"

class Resource;

struct ReadbackCaptureInfo {
	std::string passName;
	std::weak_ptr<Resource> resource;
	uint64_t resourceId = 0;
	RangeSpec range{};
	ReadbackCaptureCallback callback;
	QueueKind preferredQueueKind = QueueKind::Graphics;
};

struct ReadbackCaptureToken {
	uint64_t id = 0;
};

class ReadbackManager {
public:
	static ReadbackManager& GetInstance();
	~ReadbackManager();

	void Initialize(rhi::Timeline graphicsReadbackFence, rhi::Timeline copyReadbackFence) {
		EnsureReleaseWorker();
		m_graphicsReadbackFence = graphicsReadbackFence;
		m_copyReadbackFence = copyReadbackFence;
		m_initialized = m_graphicsReadbackFence.IsValid() || m_copyReadbackFence.IsValid();
		m_warnedUninitializedUse = false;
	}

	void RequestReadbackCapture(
		const std::string& passName,
		Resource* resource,
		const RangeSpec& range,
		ReadbackCaptureCallback callback,
		QueueKind preferredQueueKind = QueueKind::Graphics);

	std::vector<ReadbackCaptureInfo> ConsumeCaptureRequests();

	std::shared_ptr<Resource> AcquireReadbackBuffer(uint64_t byteSize, const char* debugName);
	ReadbackCaptureToken EnqueueCapture(ReadbackCaptureRequest&& request);
	void FinalizeCapture(ReadbackCaptureToken token, QueueKind queueKind, std::shared_ptr<rhi::TimelinePtr> signalFenceOwner, uint64_t fenceValue);

	uint64_t GetNextReadbackFenceValue(QueueKind queueKind);
	rhi::Timeline GetReadbackFence(QueueKind queueKind) const;

	void ProcessReadbackRequests();

	void Cleanup() {
		m_queuedCaptures.clear();
		m_readbackCaptureRequests.clear();
		m_graphicsReadbackFence.Reset();
		m_copyReadbackFence.Reset();
		m_initialized = false;
		m_warnedUninitializedUse = false;
		m_captureFenceValueGraphics.store(0, std::memory_order_relaxed);
		m_captureFenceValueCopy.store(0, std::memory_order_relaxed);
		StopReleaseWorker();
		{
			std::lock_guard lock(m_readbackBufferPoolMutex);
			m_readbackBufferPool.clear();
			m_readbackBufferPoolBytes = 0;
		}
	}

private:
	ReadbackManager() = default;

	void EnsureReleaseWorker();
	void StopReleaseWorker();
	void QueueDeferredRelease(std::vector<ReadbackCaptureRequest>&& requests);
	void ReleaseWorkerMain();
	void RecycleReadbackBuffer(std::shared_ptr<Resource>&& buffer);

	static QueueKind NormalizeQueueKind(QueueKind queueKind) {
		return queueKind == QueueKind::Copy ? QueueKind::Copy : QueueKind::Graphics;
	}

	rhi::Timeline ResolveReadbackFence(QueueKind queueKind) const {
		return NormalizeQueueKind(queueKind) == QueueKind::Copy ? m_copyReadbackFence : m_graphicsReadbackFence;
	}

	rhi::Timeline m_graphicsReadbackFence;
	rhi::Timeline m_copyReadbackFence;
	bool m_initialized = false;
	bool m_warnedUninitializedUse = false;
	std::mutex readbackRequestsMutex;
	std::vector<ReadbackCaptureRequest> m_readbackCaptureRequests;

	std::mutex m_captureQueueMutex;
	std::vector<ReadbackCaptureInfo> m_queuedCaptures;
	std::atomic<uint64_t> m_captureTokenCounter = 0;
	std::atomic<uint64_t> m_captureFenceValueGraphics = 0;
	std::atomic<uint64_t> m_captureFenceValueCopy = 0;

	std::mutex m_releaseQueueMutex;
	std::condition_variable m_releaseQueueCV;
	std::vector<ReadbackCaptureRequest> m_deferredReleaseRequests;
	std::thread m_releaseThread;
	bool m_releaseThreadQuit = false;

	std::mutex m_readbackBufferPoolMutex;
	std::vector<std::shared_ptr<Resource>> m_readbackBufferPool;
	uint64_t m_readbackBufferPoolBytes = 0;

	// Static pointer to hold the instance
	static std::unique_ptr<ReadbackManager> instance;
	// Static initialization flag
	static bool initialized;
};

inline ReadbackManager& ReadbackManager::GetInstance() {
	if (!initialized) {
		instance = std::unique_ptr<ReadbackManager>(new ReadbackManager());
		initialized = true;
	}
	return *instance;
}
