#include "Managers/Singletons/ReadbackManager.h"

#include <cstdio>
#include <cstring>
#include <iterator>
#include <spdlog/spdlog.h>
#include <tracy/Tracy.hpp>

#include "Resources/Buffers/Buffer.h"

std::unique_ptr<ReadbackManager> ReadbackManager::instance = nullptr;
bool ReadbackManager::initialized = false;

namespace {
	constexpr size_t kReadbackBufferPoolMaxBuffers = 64;
	constexpr uint64_t kReadbackBufferPoolMaxBytes = 256ull * 1024ull * 1024ull;
}

ReadbackManager::~ReadbackManager() {
    StopReleaseWorker();
}

void ReadbackManager::EnsureReleaseWorker() {
    std::lock_guard lock(m_releaseQueueMutex);
    if (m_releaseThread.joinable()) {
        return;
    }

    m_releaseThreadQuit = false;
    m_releaseThread = std::thread([this]() {
        ReleaseWorkerMain();
    });
}

void ReadbackManager::StopReleaseWorker() {
    {
        std::lock_guard lock(m_releaseQueueMutex);
        if (!m_releaseThread.joinable()) {
            m_deferredReleaseRequests.clear();
            m_releaseThreadQuit = false;
            return;
        }

        m_releaseThreadQuit = true;
    }
    m_releaseQueueCV.notify_one();
    m_releaseThread.join();

    std::lock_guard lock(m_releaseQueueMutex);
    m_deferredReleaseRequests.clear();
    m_releaseThreadQuit = false;
}

void ReadbackManager::QueueDeferredRelease(std::vector<ReadbackCaptureRequest>&& requests) {
    if (requests.empty()) {
        return;
    }

    EnsureReleaseWorker();
    {
        ZoneScopedN("ReadbackManager::ProcessReadbackRequests::QueueDeferredRelease");
        std::lock_guard lock(m_releaseQueueMutex);
        m_deferredReleaseRequests.insert(
            m_deferredReleaseRequests.end(),
            std::make_move_iterator(requests.begin()),
            std::make_move_iterator(requests.end()));
    }
    m_releaseQueueCV.notify_one();
}

void ReadbackManager::ReleaseWorkerMain() {
    std::vector<ReadbackCaptureRequest> releaseBatch;
    for (;;) {
        {
            std::unique_lock lock(m_releaseQueueMutex);
            m_releaseQueueCV.wait(lock, [this]() {
                return m_releaseThreadQuit || !m_deferredReleaseRequests.empty();
            });

            if (m_deferredReleaseRequests.empty() && m_releaseThreadQuit) {
                break;
            }

            releaseBatch.swap(m_deferredReleaseRequests);
        }

        {
            ZoneScopedN("ReadbackManager::DeferredReleaseWorker::ReleaseRequests");
            ZoneValue(releaseBatch.size());
            releaseBatch.clear();
        }
    }
}

std::shared_ptr<Resource> ReadbackManager::AcquireReadbackBuffer(uint64_t byteSize, const char* debugName) {
    ZoneScopedN("ReadbackManager::AcquireReadbackBuffer");
    ZoneValue(byteSize);
    TracyPlot("Readback.AcquireBufferBytes", static_cast<int64_t>(byteSize));

    if (byteSize == 0) {
        return nullptr;
    }

    {
        ZoneScopedN("ReadbackManager::AcquireReadbackBuffer::PoolLookup");
        std::lock_guard lock(m_readbackBufferPoolMutex);

        size_t bestIndex = m_readbackBufferPool.size();
        uint64_t bestSize = UINT64_MAX;
        for (size_t i = 0; i < m_readbackBufferPool.size(); ++i) {
            uint64_t candidateSize = 0;
            auto& candidate = m_readbackBufferPool[i];
            if (!candidate || !candidate->TryGetBufferByteSize(candidateSize) || candidateSize < byteSize) {
                continue;
            }

            if (candidateSize < bestSize) {
                bestSize = candidateSize;
                bestIndex = i;
            }
        }

        if (bestIndex != m_readbackBufferPool.size()) {
            auto buffer = std::move(m_readbackBufferPool[bestIndex]);
            const uint64_t bufferSize = bestSize;
            m_readbackBufferPoolBytes -= bufferSize;
            if (bestIndex + 1 != m_readbackBufferPool.size()) {
                m_readbackBufferPool[bestIndex] = std::move(m_readbackBufferPool.back());
            }
            m_readbackBufferPool.pop_back();
            if (debugName) {
                buffer->SetName(debugName);
            }
            TracyPlot("Readback.BufferPoolHits", int64_t{1});
            TracyPlot("Readback.BufferPoolBytes", static_cast<int64_t>(m_readbackBufferPoolBytes));
            return buffer;
        }
    }

    TracyPlot("Readback.BufferPoolHits", int64_t{0});
    {
        ZoneScopedN("ReadbackManager::AcquireReadbackBuffer::Create");
        auto buffer = Buffer::CreateShared(rhi::HeapType::Readback, byteSize);
        if (debugName) {
            buffer->SetName(debugName);
        }
        return buffer;
    }
}

void ReadbackManager::RecycleReadbackBuffer(std::shared_ptr<Resource>&& buffer) {
    ZoneScopedN("ReadbackManager::RecycleReadbackBuffer");
    if (!buffer) {
        return;
    }

    uint64_t byteSize = 0;
    if (!buffer->TryGetBufferByteSize(byteSize) || byteSize == 0) {
        return;
    }

    ZoneValue(byteSize);
    std::lock_guard lock(m_readbackBufferPoolMutex);
    if (m_readbackBufferPool.size() >= kReadbackBufferPoolMaxBuffers ||
        m_readbackBufferPoolBytes + byteSize > kReadbackBufferPoolMaxBytes) {
        TracyPlot("Readback.BufferPoolDrops", int64_t{1});
        return;
    }

    m_readbackBufferPoolBytes += byteSize;
    m_readbackBufferPool.push_back(std::move(buffer));
    TracyPlot("Readback.BufferPoolDrops", int64_t{0});
    TracyPlot("Readback.BufferPoolSize", static_cast<int64_t>(m_readbackBufferPool.size()));
    TracyPlot("Readback.BufferPoolBytes", static_cast<int64_t>(m_readbackBufferPoolBytes));
}

void ReadbackManager::RequestReadbackCapture(
    const std::string& passName,
    Resource* resource,
    const RangeSpec& range,
    ReadbackCaptureCallback callback,
    QueueKind preferredQueueKind)
{
    ZoneScopedN("ReadbackManager::RequestReadbackCapture");
    ZoneText(passName.c_str(), passName.size());

    std::weak_ptr<Resource> weakResource;
    uint64_t resourceId = 0;
    if (resource) {
        weakResource = resource->weak_from_this();
        resourceId = resource->GetGlobalResourceID();
    }

    std::scoped_lock lock(m_captureQueueMutex);
    m_queuedCaptures.push_back(ReadbackCaptureInfo{
        passName,
        weakResource,
        resourceId,
        range,
        std::move(callback),
        preferredQueueKind
        });
    TracyPlot("Readback.QueuedCaptures", static_cast<int64_t>(m_queuedCaptures.size()));
}

std::vector<ReadbackCaptureInfo> ReadbackManager::ConsumeCaptureRequests() {
    std::lock_guard<std::mutex> lock(m_captureQueueMutex);
    auto out = std::move(m_queuedCaptures);
    m_queuedCaptures.clear();
    return out;
}

ReadbackCaptureToken ReadbackManager::EnqueueCapture(ReadbackCaptureRequest&& request) {
    const uint64_t token = ++m_captureTokenCounter;
    request.token = token;
    std::lock_guard<std::mutex> lock(readbackRequestsMutex);
    m_readbackCaptureRequests.push_back(std::move(request));
    return { token };
}

void ReadbackManager::FinalizeCapture(ReadbackCaptureToken token, QueueKind queueKind, std::shared_ptr<rhi::TimelinePtr> signalFenceOwner, uint64_t fenceValue) {
    std::lock_guard<std::mutex> lock(readbackRequestsMutex);
    for (auto& request : m_readbackCaptureRequests) {
        if (request.token == token.id) {
            request.signalQueueKind = NormalizeQueueKind(queueKind);
            request.signalFenceOwner = std::move(signalFenceOwner);
            request.fenceValue = fenceValue;
            return;
        }
    }

    spdlog::warn(
        "ReadbackManager::FinalizeCapture could not find token {}. Pending captures: {}.",
        token.id,
        m_readbackCaptureRequests.size());
}

rhi::Timeline ReadbackManager::GetReadbackFence(QueueKind queueKind) const {
    return ResolveReadbackFence(queueKind);
}

uint64_t ReadbackManager::GetNextReadbackFenceValue(QueueKind queueKind) {
    return NormalizeQueueKind(queueKind) == QueueKind::Copy
        ? m_captureFenceValueCopy.fetch_add(1, std::memory_order_relaxed) + 1
        : m_captureFenceValueGraphics.fetch_add(1, std::memory_order_relaxed) + 1;
}

void ReadbackManager::ProcessReadbackRequests() {
    ZoneScopedN("ReadbackManager::ProcessReadbackRequests");
    std::vector<ReadbackCaptureRequest> pendingCaptures;
    {
        ZoneScopedN("ReadbackManager::ProcessReadbackRequests::AcquirePendingLock");
        std::lock_guard<std::mutex> lock(readbackRequestsMutex);
        pendingCaptures.swap(m_readbackCaptureRequests);
    }

    ZoneValue(pendingCaptures.size());
    TracyPlot("Readback.PendingCaptures", static_cast<int64_t>(pendingCaptures.size()));

    if (!m_initialized) {
        if (!m_warnedUninitializedUse) {
            spdlog::warn("ReadbackManager::ProcessReadbackRequests called before readback fence initialization; deferring {} pending readback captures.", pendingCaptures.size());
            m_warnedUninitializedUse = true;
        }
        if (!pendingCaptures.empty()) {
            ZoneScopedN("ReadbackManager::ProcessReadbackRequests::RestoreUninitializedPending");
            std::lock_guard<std::mutex> lock(readbackRequestsMutex);
            if (m_readbackCaptureRequests.empty()) {
                m_readbackCaptureRequests = std::move(pendingCaptures);
            }
            else {
                pendingCaptures.insert(
                    pendingCaptures.end(),
                    std::make_move_iterator(m_readbackCaptureRequests.begin()),
                    std::make_move_iterator(m_readbackCaptureRequests.end()));
                m_readbackCaptureRequests = std::move(pendingCaptures);
            }
        }
        return;
    }

    std::vector<ReadbackCaptureRequest> remainingCaptures;
    remainingCaptures.reserve(pendingCaptures.size());
    uint64_t completedCaptureCount = 0;
    uint64_t completedBytes = 0;
    uint64_t callbackCount = 0;
    bool cachedGraphicsCompletedValid = false;
    bool cachedCopyCompletedValid = false;
    uint64_t cachedGraphicsCompletedValue = 0;
    uint64_t cachedCopyCompletedValue = 0;
    for (auto& request : pendingCaptures) {
        ZoneScopedN("ReadbackManager::ProcessReadbackRequests::Request");
        ZoneValue(request.totalSize);
        char requestText[128]{};
        std::snprintf(
            requestText,
            sizeof(requestText),
            "token=%llu resource=%llu bytes=%llu",
            static_cast<unsigned long long>(request.token),
            static_cast<unsigned long long>(request.desc.resourceId),
            static_cast<unsigned long long>(request.totalSize));
        ZoneText(requestText, std::strlen(requestText));

        if (request.fenceValue == 0) {
            spdlog::warn(
                "ReadbackManager dropping capture token {} for resource {} because it has no fence value (FinalizeCapture was not applied).",
                request.token,
                request.desc.resourceId);
            continue;
        }

        rhi::Timeline requestFence;
        uint64_t completedValue = 0;
        {
            ZoneScopedN("ReadbackManager::ProcessReadbackRequests::GetCompletedValue");
            if (request.signalFenceOwner && *request.signalFenceOwner) {
                requestFence = request.signalFenceOwner->Get();
                completedValue = requestFence.GetCompletedValue();
            }
            else {
                const QueueKind normalizedQueueKind = NormalizeQueueKind(request.signalQueueKind);
                requestFence = ResolveReadbackFence(normalizedQueueKind);
                if (!requestFence.IsValid()) {
                    spdlog::warn(
                        "ReadbackManager dropping capture token {} for resource {} because queue {} has no initialized readback fence.",
                        request.token,
                        request.desc.resourceId,
                        static_cast<int>(request.signalQueueKind));
                    continue;
                }

                if (normalizedQueueKind == QueueKind::Copy) {
                    if (!cachedCopyCompletedValid) {
                        cachedCopyCompletedValue = requestFence.GetCompletedValue();
                        cachedCopyCompletedValid = true;
                    }
                    completedValue = cachedCopyCompletedValue;
                }
                else {
                    if (!cachedGraphicsCompletedValid) {
                        cachedGraphicsCompletedValue = requestFence.GetCompletedValue();
                        cachedGraphicsCompletedValid = true;
                    }
                    completedValue = cachedGraphicsCompletedValue;
                }
            }
        }

        if (!requestFence.IsValid()) {
                spdlog::warn(
                    "ReadbackManager dropping capture token {} for resource {} because queue {} has no initialized readback fence.",
                    request.token,
                    request.desc.resourceId,
                    static_cast<int>(request.signalQueueKind));
                continue;
        }

        if (completedValue >= request.fenceValue) {
            ++completedCaptureCount;
            completedBytes += request.totalSize;

            void* mappedData = nullptr;
            {
                ZoneScopedN("ReadbackManager::ProcessReadbackRequests::Map");
                request.readbackBuffer->GetAPIResource().Map(&mappedData);
            }

            ReadbackCaptureResult result{};
            {
                ZoneScopedN("ReadbackManager::ProcessReadbackRequests::PrepareResult");
                result.desc = request.desc;
                result.layouts = request.layouts;
                result.format = request.format;
                result.width = request.width;
                result.height = request.height;
                result.depth = request.depth;
                result.data.resize(request.totalSize);
            }

            {
                ZoneScopedN("ReadbackManager::ProcessReadbackRequests::CopyBytes");
                std::memcpy(result.data.data(), mappedData, request.totalSize);
            }
            {
                ZoneScopedN("ReadbackManager::ProcessReadbackRequests::Unmap");
                request.readbackBuffer->GetAPIResource().Unmap(0, 0);
            }

            if (request.callback) {
                ++callbackCount;
                ZoneScopedN("ReadbackManager::ProcessReadbackRequests::Callback");
                request.callback(std::move(result));
            }

            auto completedReadbackBuffer = std::move(request.readbackBuffer);
            request.signalFenceOwner.reset();
            request.callback = {};
            RecycleReadbackBuffer(std::move(completedReadbackBuffer));
        }
        else {
            remainingCaptures.push_back(std::move(request));
        }
    }

    {
        ZoneScopedN("ReadbackManager::ProcessReadbackRequests::MergeRemaining");
        std::lock_guard<std::mutex> lock(readbackRequestsMutex);
        if (m_readbackCaptureRequests.empty()) {
            m_readbackCaptureRequests = std::move(remainingCaptures);
        }
        else if (!remainingCaptures.empty()) {
            remainingCaptures.insert(
                remainingCaptures.end(),
                std::make_move_iterator(m_readbackCaptureRequests.begin()),
                std::make_move_iterator(m_readbackCaptureRequests.end()));
            m_readbackCaptureRequests = std::move(remainingCaptures);
        }
    }

    {
        ZoneScopedN("ReadbackManager::ProcessReadbackRequests::DeferCompletedRelease");
        QueueDeferredRelease(std::move(pendingCaptures));
    }

    TracyPlot("Readback.CompletedCaptures", static_cast<int64_t>(completedCaptureCount));
    TracyPlot("Readback.CompletedBytes", static_cast<int64_t>(completedBytes));
    TracyPlot("Readback.Callbacks", static_cast<int64_t>(callbackCount));
}
