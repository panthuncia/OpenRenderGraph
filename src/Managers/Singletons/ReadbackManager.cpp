#include "Managers/Singletons/ReadbackManager.h"

#include <cstring>
#include <spdlog/spdlog.h>

std::unique_ptr<ReadbackManager> ReadbackManager::instance = nullptr;
bool ReadbackManager::initialized = false;

void ReadbackManager::RequestReadbackCapture(
    const std::string& passName,
    Resource* resource,
    const RangeSpec& range,
    ReadbackCaptureCallback callback)
{
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
        std::move(callback)
        });
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

void ReadbackManager::FinalizeCapture(ReadbackCaptureToken token, uint64_t fenceValue) {
    std::lock_guard<std::mutex> lock(readbackRequestsMutex);
    for (auto& request : m_readbackCaptureRequests) {
        if (request.token == token.id) {
            request.fenceValue = fenceValue;
            return;
        }
    }

    spdlog::warn(
        "ReadbackManager::FinalizeCapture could not find token {}. Pending captures: {}.",
        token.id,
        m_readbackCaptureRequests.size());
}

uint64_t ReadbackManager::GetNextReadbackFenceValue() {
    return ++m_captureFenceValue;
}

void ReadbackManager::ProcessReadbackRequests() {
    std::lock_guard<std::mutex> lock(readbackRequestsMutex);

    if (!m_initialized || !m_readbackFence.IsValid()) {
        if (!m_warnedUninitializedUse) {
            spdlog::warn("ReadbackManager::ProcessReadbackRequests called before readback fence initialization; deferring {} pending readback captures.", m_readbackCaptureRequests.size());
            m_warnedUninitializedUse = true;
        }
        return;
    }

    const auto completedValue = m_readbackFence.GetCompletedValue();

    std::vector<ReadbackCaptureRequest> remainingCaptures;
    remainingCaptures.reserve(m_readbackCaptureRequests.size());
    for (auto& request : m_readbackCaptureRequests) {
        if (request.fenceValue == 0) {
            spdlog::warn(
                "ReadbackManager dropping capture token {} for resource {} because it has no fence value (FinalizeCapture was not applied).",
                request.token,
                request.desc.resourceId);
            continue;
        }

        if (completedValue >= request.fenceValue) {
            void* mappedData = nullptr;
            request.readbackBuffer->GetAPIResource().Map(&mappedData);

            ReadbackCaptureResult result{};
            result.desc = request.desc;
            result.layouts = request.layouts;
            result.format = request.format;
            result.width = request.width;
            result.height = request.height;
            result.depth = request.depth;
            result.data.resize(request.totalSize);

            std::memcpy(result.data.data(), mappedData, request.totalSize);
            request.readbackBuffer->GetAPIResource().Unmap(0, 0);

            if (request.callback) {
                request.callback(std::move(result));
            }
        }
        else {
            remainingCaptures.push_back(std::move(request));
        }
    }

    m_readbackCaptureRequests = std::move(remainingCaptures);
}