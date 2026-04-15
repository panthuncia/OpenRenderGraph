#include "Resources/Buffers/DynamicBufferBase.h"

#include <stdexcept>
#include <string>
#include <string_view>

#include <spdlog/spdlog.h>

#include "Managers/Singletons/DeviceManager.h"
#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Resources/GPUBacking/GpuBufferBacking.h"
#include "Resources/ExternalBackingResource.h"
#include "Render/Runtime/UploadServiceAccess.h"
#include "Render/Runtime/UploadPolicyServiceAccess.h"

namespace {
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
        auto device = DeviceManager::GetInstance().GetDevice();
        if (!device) {
            spdlog::info("Buffer materialize command-list probe skipped phase={} deviceValid=false", phase);
            return;
        }

        spdlog::info("Buffer materialize command-list probe begin phase={}", phase);

        rhi::CommandAllocatorPtr allocator;
        const auto allocResult = device.CreateCommandAllocator(rhi::QueueKind::Graphics, allocator);
        spdlog::info(
            "Buffer materialize command-list probe phase={} allocator result={} valid={}",
            phase,
            static_cast<int>(allocResult),
            static_cast<bool>(allocator));
        if (allocResult != rhi::Result::Ok || !allocator) {
            return;
        }

        rhi::CommandListPtr list;
        const auto listResult = device.CreateCommandList(rhi::QueueKind::Graphics, allocator.Get(), list);
        spdlog::info(
            "Buffer materialize command-list probe phase={} list result={} valid={}",
            phase,
            static_cast<int>(listResult),
            static_cast<bool>(list));
        if (listResult != rhi::Result::Ok || !list) {
            return;
        }

        list->End();
        spdlog::info("Buffer materialize command-list probe end phase={}", phase);
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

BufferBase::BufferBase() = default;

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
        throw std::runtime_error("Buffer resource is not materialized");
    }
    return m_dataBuffer->GetAPIResource();
}

SymbolicTracker* BufferBase::GetStateTracker() {
    if (!m_dataBuffer) {
        throw std::runtime_error("Buffer resource is not materialized");
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
        throw std::runtime_error("Buffer resource is not materialized");
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
    spdlog::info(
        "BufferBase::SetDescriptorRequirements name='{}' id={} cbv={} srv={} uav={} cpuUav={} cbvBytes={} srvKind={} srvElements={} srvStride={} uavKind={} uavElements={} uavStride={} uavCounterOffset={}",
        GetName(),
        GetGlobalResourceID(),
        requirements.createCBV,
        requirements.createSRV,
        requirements.createUAV,
        requirements.createNonShaderVisibleUAV,
        requirements.cbvDesc.byteSize,
        BufferViewKindToString(requirements.srvDesc.buffer.kind),
        requirements.srvDesc.buffer.numElements,
        requirements.srvDesc.buffer.structureByteStride,
        BufferViewKindToString(requirements.uavDesc.buffer.kind),
        requirements.uavDesc.buffer.numElements,
        requirements.uavDesc.buffer.structureByteStride,
        requirements.uavCounterOffset);

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
    if (m_uploadPolicyTag == rg::runtime::UploadPolicyTag::Immediate) {
        return;
    }

    if (m_uploadPolicyRegistered) {
        return;
    }

    if (!rg::runtime::GetActiveUploadPolicyService()) {
        return;
    }

    rg::runtime::RegisterUploadPolicyClient(this);
    m_uploadPolicyRegistered = true;
}

void BufferBase::RefreshUploadPolicyRegistration() {
    if (m_uploadPolicyTag == rg::runtime::UploadPolicyTag::Immediate) {
        UnregisterUploadPolicyClient();
        return;
    }

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
    spdlog::info(
        "BufferBase::SetBacking begin name='{}' id={} oldSize={} newSize={} hadBacking={} hasDescriptors={}",
        GetName(),
        GetGlobalResourceID(),
        m_bufferSize,
        bufferSize,
        m_dataBuffer != nullptr,
        m_descriptorRequirements.has_value());
    m_dataBuffer = std::move(backing);
    m_bufferSize = bufferSize;
    spdlog::info(
        "BufferBase::SetBacking refreshing descriptors name='{}' id={}",
        GetName(),
        GetGlobalResourceID());
    RefreshDescriptorContents();
    ++m_backingGeneration;
    if (m_dataBuffer) {
        OnBackingMaterialized();
    }
    spdlog::info(
        "BufferBase::SetBacking complete name='{}' id={} backingGeneration={}",
        GetName(),
        GetGlobalResourceID(),
        m_backingGeneration);
}

void BufferBase::CreateAndSetBacking(rhi::HeapType accessType, uint64_t bufferSize, bool unorderedAccess) {
    spdlog::info(
        "BufferBase::CreateAndSetBacking name='{}' id={} heapType={} bytes={} unorderedAccess={}",
        GetName(),
        GetGlobalResourceID(),
        HeapTypeToString(accessType),
        bufferSize,
        unorderedAccess);
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
	spdlog::info(
		"BufferBase::QueueResourceCopyFromOldBacking name='{}' id={} bytes={} ",
		GetName(),
		GetGlobalResourceID(),
		bytesToCopy);

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
