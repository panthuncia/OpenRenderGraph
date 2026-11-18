#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "Render/Runtime/IUploadService.h"

namespace org::runtime {

inline IUploadService*& UploadServiceSlot() {
    static thread_local IUploadService* service = nullptr;
    return service;
}

inline void SetActiveUploadService(IUploadService* service) {
    UploadServiceSlot() = service;
}

inline IUploadService* GetActiveUploadService() {
    return UploadServiceSlot();
}

inline void UploadBufferDataDispatch(
    const void* data,
    size_t size,
    UploadTarget resourceToUpdate,
    size_t dataBufferOffset,
    [[maybe_unused]] const char* file,
    [[maybe_unused]] int line) {
    if (auto* service = GetActiveUploadService()) {
        service->UploadData(data, size, std::move(resourceToUpdate), dataBufferOffset, file, line);
        return;
    }

    throw std::runtime_error("Upload service is not active for BUFFER_UPLOAD");
}

inline void UploadTextureSubresourcesDispatch(
    UploadTarget target,
    rhi::Format fmt,
    uint32_t baseWidth,
    uint32_t baseHeight,
    uint32_t depthOrLayers,
    uint32_t mipLevels,
    uint32_t arraySize,
    const rhi::helpers::SubresourceData* srcSubresources,
    uint32_t srcCount,
    [[maybe_unused]] const char* file,
    [[maybe_unused]] int line) {
    if (auto* service = GetActiveUploadService()) {
        service->UploadTextureSubresources(
            std::move(target),
            fmt,
            baseWidth,
            baseHeight,
            depthOrLayers,
            mipLevels,
            arraySize,
            srcSubresources,
            srcCount,
            file,
            line);
        return;
    }

    throw std::runtime_error("Upload service is not active for TEXTURE_UPLOAD_SUBRESOURCES");
}

}

#define BUFFER_UPLOAD(data,size,res,offset) \
    org::runtime::UploadBufferDataDispatch((data),(size),(res),(offset),__FILE__,__LINE__)
#define TEXTURE_UPLOAD_SUBRESOURCES(dstTexture,fmt,baseWidth,baseHeight,depthOrLayers,mipLevels,arraySize,srcSubresources,srcCount) \
	org::runtime::UploadTextureSubresourcesDispatch((dstTexture),(fmt),(baseWidth),(baseHeight),(depthOrLayers),(mipLevels),(arraySize),(srcSubresources),(srcCount),__FILE__,__LINE__)
