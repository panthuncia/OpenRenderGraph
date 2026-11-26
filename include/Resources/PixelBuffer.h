#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <rhi_helpers.h>

#include "Resources/AliasingPlacement.h"
#include "Resources/BackedResource.h"
#include "Resources/GloballyIndexedResource.h"
#include "Resources/TextureDescription.h"
#include "Resources/MemoryStatisticsComponents.h"
#include "Interfaces/IHasMemoryMetadata.h"


namespace org {

class GpuTextureBacking;

class PixelBuffer : public GloballyIndexedResource, public BackedResource, public IHasMemoryMetadata {
public:
    struct MaterializeOptions {
        std::optional<TextureAliasPlacement> aliasPlacement;
    };

    static std::shared_ptr<PixelBuffer> CreateShared(const TextureDescription& desc)
    {
        auto pb = std::shared_ptr<PixelBuffer>(new PixelBuffer(desc, true));
        pb->NoteRelocationOwner();
        return pb;
    }

    static std::shared_ptr<PixelBuffer> CreateSharedUnmaterialized(const TextureDescription& desc)
    {
        auto pb = std::shared_ptr<PixelBuffer>(new PixelBuffer(desc, false));
        return pb;
    }

    rhi::Resource GetAPIResource() override;
    rhi::BarrierBatch GetEnhancedBarrierGroup(
        RangeSpec range, 
        rhi::ResourceAccessType prevAccessType, 
        rhi::ResourceAccessType newAccessType, 
        rhi::ResourceLayout prevLayout, 
        rhi::ResourceLayout newLayout, 
        rhi::ResourceSyncState prevSyncState, 
        rhi::ResourceSyncState newSyncState) override
	;

	rhi::Format GetFormat() const {
        return m_desc.format;
    }
	bool IsBlockCompressed() const { return rhi::helpers::IsBlockCompressed(GetFormat()); }
    const rhi::ClearValue& GetClearColor() const {
        return m_clearValue;
    }
    bool TryGetPublishedClearValue(rhi::ClearValue& value) const override {
        value = m_clearValue;
        return true;
    }
    unsigned int GetInternalWidth() const {
        return m_internalWidth;
    }
    unsigned int GetInternalHeight() const {
        return m_internalHeight;
    }
    unsigned int GetWidth() const {
        return m_desc.imageDimensions[0].width;
    }
    unsigned int GetHeight() const {
        return m_desc.imageDimensions[0].height;
    }
    unsigned int GetChannelCount()const {
	    return m_desc.channels;
    }
    bool IsCubemap() const {
        return m_desc.isCubemap;
	}
    void OnSetName() override;

    void ApplyMetadataComponentBundle(const EntityComponentBundle& bundle) const;

    SymbolicTracker* GetStateTracker() override;
	bool TryGetRHIResourceDesc(rhi::ResourceDesc& outDesc) const override;
	void RefreshAPIRepresentationDescriptors(BackendInstanceId backendInstance) override;

    bool IsMaterialized() const override {
        return m_backing != nullptr || GetAttachedAPIRepresentation(BackendInstanceId::Primary).IsValid();
    }

    bool HasValidBackingResource() const;
    BackingAllocationSnapshot CaptureBackingAllocation(bool retain = true) override;
    std::recursive_mutex* BackingMutex() noexcept override { return &m_materializationMutex; }

    uint64_t GetBackingGeneration() const override {
        return m_backingGeneration;
    }

    void SetAliasingPool(uint64_t poolID) {
        m_desc.aliasingPoolID = poolID;
        m_desc.allowAlias = true;
    }

    void ClearAliasingPoolHint() {
        m_desc.aliasingPoolID.reset();
    }

    std::optional<uint64_t> GetAliasingPoolHint() const {
        return m_desc.aliasingPoolID;
    }

	const TextureDescription& GetDescription() const {
		return m_desc;
	}

    void EnableIdleDematerialization(uint32_t idleFrameThreshold = 1) {
        m_allowIdleDematerialization = true;
        m_idleDematerializationThreshold = (std::max)(1u, idleFrameThreshold);
    }

    void DisableIdleDematerialization() {
        m_allowIdleDematerialization = false;
    }

    bool IsIdleDematerializationEnabled() const {
        return m_allowIdleDematerialization;
    }

    uint32_t GetIdleDematerializationThreshold() const {
        return m_idleDematerializationThreshold;
    }

    void Materialize(const MaterializeOptions* options = nullptr);

    void Dematerialize();

    // Relocation (relocation::Service). A standalone backing of the same description as the current one, for the current
    // backing to move into. Null without a backing of its own.
    std::unique_ptr<GpuTextureBacking> MakeRelocationBacking();
    // Relocation. Switches to a new backing while the resource keeps its identity, under the backing lock:
    // - next: the new backing. On success it holds the old one, for the caller to retire. Null when the current backing's
    //   allocation already names the new resource (a defragmentation move); the backing stays and only its views change.
    // - fromGeneration: the generation the move was made for. Nothing changes when the backing changed since.
    // - rewriteInPlace: the GPU is idle, so the same descriptor slots are written again and bindless indices stay. Otherwise
    //   the slots rotate: frames in flight keep the old ones, and new work gets new indices.
    // Bumps the backing generation and drops the cached publication snapshot. False when nothing changed.
    bool AdoptRelocatedBacking(std::unique_ptr<GpuTextureBacking>& next, uint64_t fromGeneration, bool rewriteInPlace);
    // Defragmentation. Creates the current backing's texture again at the start of destination, when the current backing's
    // allocation is source. Fails when the backing changed, or for an attached representation. The caller owns the result.
    rhi::Result MakeDefragmentationResource(const rhi::ma::Allocation& source, rhi::ma::Allocation& destination, rhi::ResourcePtr& out);

    void EnsureVirtualDescriptorSlotsAllocated() override;

    // A placed backing made ahead, and the switch to it at a frame boundary (BackedResource).
    std::shared_ptr<PreparedPlacedBacking> PreparePlacedBacking(std::shared_ptr<const AliasHeapGeneration> heap, uint64_t offset,
        uint64_t poolID, std::string* why = nullptr) override;
    bool AdoptPreparedBacking(PreparedPlacedBacking& prepared) override;

    ~PixelBuffer() override;

protected:
    rhi::ma::Allocation* RelocationAllocation() noexcept override;

private:
    PixelBuffer(const TextureDescription& desc, bool materialize);

    void EnsureMaterializedLocked(const char* operation) const;
    // The description the backing is made with: the logical one, padded to powers of two when it asks for that.
    TextureDescription BackingDescription() const;
    void EnsureVirtualDescriptorSlotsAllocatedLocked();

    void ApplyMetadataComponentBundle(const EntityComponentBundle& bundle) override;

    std::unique_ptr<GpuTextureBacking> m_backing;
	TextureDescription m_desc;
    mutable std::vector<EntityComponentBundle> m_metadataBundles;
    uint64_t m_backingGeneration = 0;
    bool m_allowIdleDematerialization = false;
    uint32_t m_idleDematerializationThreshold = 1;
    uint32_t m_internalWidth = 0; // Internal width, used for padding textures to power of two
    uint32_t m_internalHeight = 0; // Internal height, used for padding textures to power of two
    rhi::ClearValue m_clearValue;
    mutable std::recursive_mutex m_materializationMutex;
};


} // namespace org
