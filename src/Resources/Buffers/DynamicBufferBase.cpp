#include "Resources/Buffers/DynamicBufferBase.h"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <functional>
#include <stdexcept>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>
#include <tracy/Tracy.hpp>

#include "Managers/Singletons/DeviceManager.h"
#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Resources/GPUBacking/GpuBufferBacking.h"
#include "Resources/ExternalBackingResource.h"
#include "Render/Runtime/UploadServiceAccess.h"
#include "Render/Runtime/UploadPolicyServiceAccess.h"

namespace {
    thread_local uint32_t g_backingMutationScopeDepth = 0;
    std::mutex g_deferredBackingResizeClientsMutex;
    std::vector<IDeferredBackingResizeClient*> g_deferredBackingResizeClients;
    std::mutex g_asyncResizeSchedulerMutex;
    AsyncBufferBackingResizeScheduler g_asyncResizeScheduler;

    const char* HeapTypeToString(rhi::HeapType heapType) {
        switch (heapType) {
        case rhi::HeapType::DeviceLocal:
            return "DeviceLocal";
        case rhi::HeapType::Upload:
            return "Upload";
        case rhi::HeapType::Readback:
            return "Readback";
        case rhi::HeapType::Custom:
            return "Custom";
        default:
            return "Unknown";
        }
    }

    const char* BufferViewKindToString(rhi::BufferViewKind kind) {
        switch (kind) {
        case rhi::BufferViewKind::Raw:
            return "Raw";
        case rhi::BufferViewKind::Structured:
            return "Structured";
        case rhi::BufferViewKind::Typed:
            return "Typed";
        default:
            return "Unknown";
        }
    }

    bool ShouldProbeVirtualShadowBuffer(std::string_view bufferName) {
        return bufferName.starts_with("CLod[Shadow] Virtual Shadow ");
    }

    void ProbeGraphicsCommandListCreation(std::string_view phase) {
        (void)phase;
    }

    void ProbeVirtualShadowBufferStep(std::string_view bufferName, std::string_view step) {
        if (!ShouldProbeVirtualShadowBuffer(bufferName)) {
            return;
        }

        std::string phase = "BufferBase::Materialize ";
        phase += step;
        phase += " :: ";
        phase += bufferName;
        ProbeGraphicsCommandListCreation(phase);
    }
}

struct AsyncBufferBackingResizeState::SharedState {
    mutable std::mutex mutex;
    std::condition_variable cv;
    AsyncBufferBackingResizeRequest lastRequest;
    AsyncBufferBackingResizeResult readyResult;
    uint64_t desiredByteSize = 0;
    uint64_t inFlightByteSize = 0;
    uint64_t inFlightToken = 0;
    uint64_t nextToken = 0;
    bool inFlight = false;
    bool readyValid = false;
};

void SetAsyncBufferBackingResizeScheduler(AsyncBufferBackingResizeScheduler scheduler) {
    std::lock_guard<std::mutex> lock(g_asyncResizeSchedulerMutex);
    g_asyncResizeScheduler = std::move(scheduler);
}

AsyncBufferBackingResizeState::AsyncBufferBackingResizeState()
    : m_state(std::make_shared<SharedState>())
{
}

AsyncBufferBackingResizeState::~AsyncBufferBackingResizeState() = default;

void AsyncBufferBackingResizeState::Request(const AsyncBufferBackingResizeRequest& request) {
    ZoneScopedN("AsyncBufferBackingResizeState::Request");
    if (request.byteSize == 0) {
        return;
    }

    auto state = m_state;
    AsyncBufferBackingResizeRequest scheduleRequest;
    uint64_t scheduleToken = 0;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (request.byteSize > state->desiredByteSize) {
            state->desiredByteSize = request.byteSize;
        }

        state->lastRequest = request;
        state->lastRequest.byteSize = state->desiredByteSize;

        if (state->readyValid) {
            const bool readyFailed = static_cast<bool>(state->readyResult.exception);
            const bool readyTooSmall = !readyFailed && state->readyResult.byteSize < state->desiredByteSize;
            if (readyFailed || readyTooSmall) {
                TracyPlot("AsyncBufferBackingResize.StaleReadyBytes", static_cast<int64_t>(state->readyResult.byteSize));
                state->readyResult = {};
                state->readyValid = false;
            }
        }

        if (state->inFlight || state->readyValid) {
            TracyPlot("AsyncBufferBackingResize.CoalescedDesiredBytes", static_cast<int64_t>(state->desiredByteSize));
            return;
        }

        scheduleToken = ++state->nextToken;
        state->inFlight = true;
        state->inFlightToken = scheduleToken;
        state->inFlightByteSize = state->desiredByteSize;
        scheduleRequest = state->lastRequest;
        scheduleRequest.byteSize = state->desiredByteSize;
    }

    Schedule(std::move(state), std::move(scheduleRequest), scheduleToken);
}

std::optional<AsyncBufferBackingResizeResult> AsyncBufferBackingResizeState::ConsumeReady(bool wait) {
    ZoneScopedN("AsyncBufferBackingResizeState::ConsumeReady");
    auto state = m_state;

    for (;;) {
        AsyncBufferBackingResizeRequest scheduleRequest;
        uint64_t scheduleToken = 0;
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            if (wait) {
                ZoneScopedN("AsyncBufferBackingResizeState::ConsumeReady::Wait");
                state->cv.wait(lock, [&]() {
                    return state->readyValid || !state->inFlight;
                });
            } else if (!state->readyValid) {
                TracyPlot("AsyncBufferBackingResize.Ready", int64_t{ 0 });
                return std::nullopt;
            }

            if (!state->readyValid) {
                TracyPlot("AsyncBufferBackingResize.Ready", int64_t{ 0 });
                return std::nullopt;
            }

            auto result = std::move(state->readyResult);
            state->readyResult = {};
            state->readyValid = false;
            TracyPlot("AsyncBufferBackingResize.Ready", int64_t{ 1 });

            if (result.exception) {
                state->desiredByteSize = 0;
                state->inFlightByteSize = 0;
                return result;
            }

            if (result.byteSize >= state->desiredByteSize) {
                state->desiredByteSize = result.byteSize;
                state->inFlightByteSize = 0;
                return result;
            }

            TracyPlot("AsyncBufferBackingResize.StaleResultBytes", static_cast<int64_t>(result.byteSize));
            if (!state->inFlight) {
                scheduleToken = ++state->nextToken;
                state->inFlight = true;
                state->inFlightToken = scheduleToken;
                state->inFlightByteSize = state->desiredByteSize;
                scheduleRequest = state->lastRequest;
                scheduleRequest.byteSize = state->desiredByteSize;
            }
        }

        if (scheduleToken != 0) {
            Schedule(std::move(state), std::move(scheduleRequest), scheduleToken);
            state = m_state;
        }

        if (!wait) {
            return std::nullopt;
        }
    }
}

bool AsyncBufferBackingResizeState::HasPending() const {
    auto state = m_state;
    std::lock_guard<std::mutex> lock(state->mutex);
    return state->inFlight || state->readyValid;
}

uint64_t AsyncBufferBackingResizeState::DesiredByteSize() const {
    auto state = m_state;
    std::lock_guard<std::mutex> lock(state->mutex);
    return state->desiredByteSize;
}

void AsyncBufferBackingResizeState::Schedule(
    std::shared_ptr<SharedState> state,
    AsyncBufferBackingResizeRequest request,
    uint64_t token)
{
    ZoneScopedN("AsyncBufferBackingResizeState::Schedule");
    TracyPlot("AsyncBufferBackingResize.ScheduledBytes", static_cast<int64_t>(request.byteSize));

    auto task = [state = std::move(state), request = std::move(request), token]() mutable {
        ZoneScopedN("AsyncBufferBackingResizeState::WorkerCreateBacking");
        AsyncBufferBackingResizeResult result;
        result.byteSize = request.byteSize;
        result.requestToken = token;
        try {
            spdlog::debug(
                "Async buffer backing resize '{}' id={} create begin bytes={} token={}",
                request.debugName,
                request.resourceID,
                request.byteSize,
                token);
            result.backing = GpuBufferBacking::CreateUnique(
                request.heapType,
                request.byteSize,
                request.resourceID,
                request.unorderedAccess,
                request.debugName.empty() ? nullptr : request.debugName.c_str());
            spdlog::debug(
                "Async buffer backing resize '{}' id={} create complete bytes={} token={}",
                request.debugName,
                request.resourceID,
                request.byteSize,
                token);
        }
        catch (...) {
            result.exception = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (token == state->inFlightToken) {
                state->readyResult = std::move(result);
                state->readyValid = true;
                state->inFlight = false;
                state->inFlightByteSize = 0;
            }
        }
        state->cv.notify_all();
    };

    AsyncBufferBackingResizeScheduler scheduler;
    {
        std::lock_guard<std::mutex> lock(g_asyncResizeSchedulerMutex);
        scheduler = g_asyncResizeScheduler;
    }

    const auto taskName = request.debugName.empty()
        ? std::string("AsyncBufferBackingResize")
        : std::string("AsyncBufferBackingResize::") + request.debugName;
    if (scheduler) {
        scheduler(taskName, std::move(task));
        return;
    }

    std::thread(std::move(task)).detach();
}

void RegisterDeferredBackingResizeClient(IDeferredBackingResizeClient* client) {
    if (client == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_deferredBackingResizeClientsMutex);
    if (std::find(g_deferredBackingResizeClients.begin(), g_deferredBackingResizeClients.end(), client) == g_deferredBackingResizeClients.end()) {
        g_deferredBackingResizeClients.push_back(client);
    }
}

void UnregisterDeferredBackingResizeClient(IDeferredBackingResizeClient* client) {
    if (client == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_deferredBackingResizeClientsMutex);
    g_deferredBackingResizeClients.erase(
        std::remove(g_deferredBackingResizeClients.begin(), g_deferredBackingResizeClients.end(), client),
        g_deferredBackingResizeClients.end());
}

uint32_t PublishReadyDeferredBackingResizes(bool wait) {
    if (!BufferBase::IsBackingMutationAllowedOnThisThread()) {
        return 0u;
    }

    std::vector<IDeferredBackingResizeClient*> clients;
    {
        std::lock_guard<std::mutex> lock(g_deferredBackingResizeClientsMutex);
        clients = g_deferredBackingResizeClients;
    }

    uint32_t published = 0u;
    for (auto* client : clients) {
        if (client == nullptr || !client->HasPendingBackingResize()) {
            continue;
        }
        if (client->PublishPendingBackingResize(wait)) {
            ++published;
        }
    }
    return published;
}

BufferBase::BufferBase() = default;

BufferBase::ScopedBackingMutation::ScopedBackingMutation()
    : m_active(true)
{
    ++g_backingMutationScopeDepth;
}

BufferBase::ScopedBackingMutation::~ScopedBackingMutation() {
    if (m_active && g_backingMutationScopeDepth > 0u) {
        --g_backingMutationScopeDepth;
    }
}

bool BufferBase::IsBackingMutationAllowedOnThisThread() {
    return g_backingMutationScopeDepth > 0u;
}

BufferBase::BufferBase(
    rhi::HeapType accessType,
    uint64_t bufferSize,
    bool unorderedAccess,
    bool materialize)
    : m_accessType(accessType)
    , m_bufferSize(bufferSize)
    , m_unorderedAccess(unorderedAccess)
{
    if (materialize && bufferSize > 0) {
        Materialize();
    }
}

BufferBase::~BufferBase() {
    UnregisterUploadPolicyClient();
}

rhi::Resource BufferBase::GetAPIResource() {
    if (!m_dataBuffer) {
		std::ostringstream err = std::ostringstream() << "Buffer resource '" << GetName() << "' is not materialized";
        throw std::runtime_error(err.str());
    }
    return m_dataBuffer->GetAPIResource();
}

SymbolicTracker* BufferBase::GetStateTracker() {
    if (!m_dataBuffer) {
        std::ostringstream err = std::ostringstream() << "Buffer resource '" << GetName() << "' is not materialized";
        throw std::runtime_error(err.str());
    }
    return m_dataBuffer->GetStateTracker();
}

rhi::BarrierBatch BufferBase::GetEnhancedBarrierGroup(
    RangeSpec range,
    rhi::ResourceAccessType prevAccessType,
    rhi::ResourceAccessType newAccessType,
    rhi::ResourceLayout prevLayout,
    rhi::ResourceLayout newLayout,
    rhi::ResourceSyncState prevSyncState,
    rhi::ResourceSyncState newSyncState)
{
    if (!m_dataBuffer) {
        std::ostringstream err = std::ostringstream() << "Buffer resource '" << GetName() << "' is not materialized";
        throw std::runtime_error(err.str());
    }
    return m_dataBuffer->GetEnhancedBarrierGroup(
        range,
        prevAccessType,
        newAccessType,
        prevLayout,
        newLayout,
        prevSyncState,
        newSyncState);
}

bool BufferBase::TryGetBufferByteSize(uint64_t& outByteSize) const {
    if (!m_dataBuffer) return false;
    outByteSize = static_cast<uint64_t>(m_dataBuffer->GetSize());
    return true;
}

void BufferBase::ConfigureBacking(
    rhi::HeapType accessType,
    uint64_t bufferSize,
    bool unorderedAccess)
{
    m_accessType = accessType;
    m_bufferSize = bufferSize;
    m_unorderedAccess = unorderedAccess;
}

bool BufferBase::IsMaterialized() const {
    return m_dataBuffer != nullptr;
}

uint64_t BufferBase::GetBufferSize() const {
    return m_bufferSize;
}

rhi::HeapType BufferBase::GetAccessType() const {
    return m_accessType;
}

bool BufferBase::IsUnorderedAccessEnabled() const {
    return m_unorderedAccess;
}

uint64_t BufferBase::GetBackingGeneration() const {
    return m_backingGeneration;
}

void BufferBase::Materialize(const MaterializeOptions* options) {
    if (m_dataBuffer) {
        return;
    }
    if (m_bufferSize == 0) {
        throw std::runtime_error("Cannot materialize a zero-sized buffer");
    }

    if (options && options->aliasPlacement.has_value()) {
        m_dataBuffer = GpuBufferBacking::CreateUnique(
            m_accessType,
            m_bufferSize,
            GetGlobalResourceID(),
            options->aliasPlacement.value(),
            m_unorderedAccess);
    }
    else {
        m_dataBuffer = GpuBufferBacking::CreateUnique(
            m_accessType,
            m_bufferSize,
            GetGlobalResourceID(),
            m_unorderedAccess);
    }

    ProbeVirtualShadowBufferStep(GetName(), "after backing create");

    RefreshDescriptorContents();
    ProbeVirtualShadowBufferStep(GetName(), "after descriptor refresh");
    ++m_backingGeneration;
    OnBackingMaterialized();
}

void BufferBase::Dematerialize() {
    if (!m_dataBuffer) {
        return;
    }
    m_dataBuffer.reset();
    ++m_backingGeneration;
}

void BufferBase::SetDescriptorRequirements(const DescriptorRequirements& requirements) {
    m_descriptorRequirements = requirements;

    EnsureVirtualDescriptorSlotsAllocated();
    if (m_dataBuffer) {
        RefreshDescriptorContents();
    }
}

bool BufferBase::HasDescriptorRequirements() const {
    return m_descriptorRequirements.has_value();
}

void BufferBase::EnsureVirtualDescriptorSlotsAllocated() {
    if (!m_descriptorRequirements.has_value() || HasAnyDescriptorSlots()) {
        return;
    }

    DescriptorHeapManager::ViewRequirements req{};
    DescriptorHeapManager::ViewRequirements::BufferViews views{};

    views.createCBV = m_descriptorRequirements->createCBV;
    views.createSRV = m_descriptorRequirements->createSRV;
    views.createUAV = m_descriptorRequirements->createUAV;
    views.createNonShaderVisibleUAV = m_descriptorRequirements->createNonShaderVisibleUAV;
    views.cbvDesc = m_descriptorRequirements->cbvDesc;
    views.srvDesc = m_descriptorRequirements->srvDesc;
    views.uavDesc = m_descriptorRequirements->uavDesc;
    views.uavCounterOffset = m_descriptorRequirements->uavCounterOffset;

    req.views = views;
    DescriptorHeapManager::GetInstance().ReserveDescriptorSlots(*this, req);
}

void BufferBase::RefreshDescriptorContents() {
    if (!m_descriptorRequirements.has_value() || !m_dataBuffer) {
        return;
    }

    EnsureVirtualDescriptorSlotsAllocated();

    DescriptorHeapManager::ViewRequirements req{};
    DescriptorHeapManager::ViewRequirements::BufferViews views{};

    views.createCBV = m_descriptorRequirements->createCBV;
    views.createSRV = m_descriptorRequirements->createSRV;
    views.createUAV = m_descriptorRequirements->createUAV;
    views.createNonShaderVisibleUAV = m_descriptorRequirements->createNonShaderVisibleUAV;
    views.cbvDesc = m_descriptorRequirements->cbvDesc;
    views.srvDesc = m_descriptorRequirements->srvDesc;
    views.uavDesc = m_descriptorRequirements->uavDesc;
    views.uavCounterOffset = m_descriptorRequirements->uavCounterOffset;

    req.views = views;
    auto resource = m_dataBuffer->GetAPIResource();
    DescriptorHeapManager::GetInstance().UpdateDescriptorContents(*this, resource, req);
}

void BufferBase::SetAliasingPool(uint64_t poolID) {
    m_aliasingPoolID = poolID;
    m_allowAlias = true;
}

void BufferBase::ClearAliasingPoolHint() {
    m_aliasingPoolID.reset();
}

std::optional<uint64_t> BufferBase::GetAliasingPoolHint() const {
    return m_aliasingPoolID;
}

void BufferBase::SetAllowAlias(bool allowAlias) {
    m_allowAlias = allowAlias;
}

bool BufferBase::IsAliasingAllowed() const {
    return m_allowAlias;
}

void BufferBase::SetUploadPolicyTag(rg::runtime::UploadPolicyTag tag) {
    m_uploadPolicyTag = tag;
    RefreshUploadPolicyRegistration();
}

rg::runtime::UploadPolicyTag BufferBase::GetUploadPolicyTag() const {
    return m_uploadPolicyTag;
}

bool BufferBase::IsUploadPolicyImmediate() const {
    return m_uploadPolicyTag == rg::runtime::UploadPolicyTag::Immediate;
}

void BufferBase::EnsureUploadPolicyRegistration() {
    if (m_uploadPolicyRegistered) {
        return;
    }

    if (!rg::runtime::GetActiveUploadPolicyService()) {
        return;
    }

    rg::runtime::RegisterUploadPolicyClient(this);
    m_uploadPolicyRegistered = true;
}

void BufferBase::MarkUploadPolicyDirty() {
    EnsureUploadPolicyRegistration();
    rg::runtime::MarkUploadPolicyClientDirty(this);
}

void BufferBase::RefreshUploadPolicyRegistration() {
    EnsureUploadPolicyRegistration();
}

void BufferBase::UnregisterUploadPolicyClient() {
    if (!m_uploadPolicyRegistered) {
        return;
    }

    rg::runtime::UnregisterUploadPolicyClient(this);
    m_uploadPolicyRegistered = false;
}

void BufferBase::SetBacking(std::unique_ptr<GpuBufferBacking> backing, uint64_t bufferSize) {
    if (m_dataBuffer && !IsBackingMutationAllowedOnThisThread()) {
        std::ostringstream threadId;
        threadId << std::this_thread::get_id();
        std::ostringstream message;
        message
            << "GPU buffer backing replacement outside frame-boundary mutation scope: resource='"
            << GetName()
            << "' id=" << GetGlobalResourceID()
            << " oldSize=" << m_bufferSize
            << " newSize=" << bufferSize
            << " backingGeneration=" << m_backingGeneration
            << " thread=" << threadId.str();
        spdlog::critical("{}", message.str());
        spdlog::apply_all([](const std::shared_ptr<spdlog::logger>& logger) {
            logger->flush();
        });
        throw std::runtime_error(message.str());
    }

    if (m_dataBuffer) {
        if (HasAnyDescriptorSlots()) {
            DescriptorHeapManager::GetInstance().RetireDescriptorSlots(DetachDescriptorSlotsForDeferredRelease());
        }
        DescriptorHeapManager::GetInstance().RetireBufferBacking(std::move(m_dataBuffer));
    }
    m_dataBuffer = std::move(backing);
    m_bufferSize = bufferSize;
    RefreshDescriptorContents();
    ++m_backingGeneration;
    if (m_dataBuffer) {
        OnBackingMaterialized();
    }
}

void BufferBase::CreateAndSetBacking(rhi::HeapType accessType, uint64_t bufferSize, bool unorderedAccess) {
    auto newBacking = GpuBufferBacking::CreateUnique(
        accessType,
        bufferSize,
        GetGlobalResourceID(),
        unorderedAccess);
    SetBacking(std::move(newBacking), bufferSize);
}

void BufferBase::SetBackingName(const std::string& baseName, const std::string& suffix) {
    if (!m_dataBuffer) {
        return;
    }

    if (!suffix.empty()) {
        m_dataBuffer->SetName((baseName + ": " + suffix).c_str());
    }
    else {
        m_dataBuffer->SetName(baseName.c_str());
    }
}

void BufferBase::QueueResourceCopyFromOldBacking(uint64_t bytesToCopy) {
    if (!m_dataBuffer) {
        return;
    }

    auto oldBackingResource = ExternalBackingResource::CreateShared(std::move(m_dataBuffer));
    if (auto* uploadService = rg::runtime::GetActiveUploadService()) {
        uploadService->QueueResourceCopy(shared_from_this(), oldBackingResource, bytesToCopy);
    }
}

void BufferBase::ApplyMetadataToBacking(const EntityComponentBundle& bundle) {
    if (m_dataBuffer) {
        m_dataBuffer->ApplyMetadataComponentBundle(bundle);
    }
}
