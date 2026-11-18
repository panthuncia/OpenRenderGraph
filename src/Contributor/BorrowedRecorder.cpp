#include "BorrowedRecorder.h"

#include <optional>

namespace org::contributor {

namespace {

template <class To, class From>
To ConvertHandle(const From& value) noexcept
{
    return To{ value.index, value.generation };
}

basicrhi_c_result Validate(BorrowState* state, uint64_t token) noexcept
{
    return state && state->active && state->commandList && state->token == token
        ? BASICRHI_C_OK : BASICRHI_C_STALE_BORROW;
}

basicrhi_c_result BASICRHI_C_CALL BindLayout(void* context, uint64_t token, basicrhi_c_pipeline_layout value)
{
    auto* state = static_cast<BorrowState*>(context);
    if (const auto result = Validate(state, token); result != BASICRHI_C_OK) return result;
    state->commandList->BindLayout(ConvertHandle<rhi::PipelineLayoutHandle>(value));
    return BASICRHI_C_OK;
}

basicrhi_c_result BASICRHI_C_CALL BindPipeline(void* context, uint64_t token, basicrhi_c_pipeline value)
{
    auto* state = static_cast<BorrowState*>(context);
    if (const auto result = Validate(state, token); result != BASICRHI_C_OK) return result;
    state->commandList->BindPipeline(ConvertHandle<rhi::PipelineHandle>(value));
    return BASICRHI_C_OK;
}

basicrhi_c_result BASICRHI_C_CALL SetDescriptorHeaps(void* context, uint64_t token,
    basicrhi_c_descriptor_heap resources, basicrhi_c_descriptor_heap samplers)
{
    auto* state = static_cast<BorrowState*>(context);
    if (const auto result = Validate(state, token); result != BASICRHI_C_OK) return result;
    const auto sampler = ConvertHandle<rhi::DescriptorHeapHandle>(samplers);
    state->commandList->SetDescriptorHeaps(ConvertHandle<rhi::DescriptorHeapHandle>(resources),
        sampler.valid() ? std::optional{ sampler } : std::nullopt);
    return BASICRHI_C_OK;
}

basicrhi_c_result BASICRHI_C_CALL PushConstants(void* context, uint64_t token, uint32_t stages,
    uint32_t set, uint32_t binding, uint32_t offset, basicrhi_c_byte_span bytes)
{
    auto* state = static_cast<BorrowState*>(context);
    if (const auto result = Validate(state, token); result != BASICRHI_C_OK) return result;
    if ((!bytes.data && bytes.size_bytes) || bytes.size_bytes % sizeof(uint32_t)) return BASICRHI_C_INVALID_ARGUMENT;
    state->commandList->PushConstants(static_cast<rhi::ShaderStage>(stages), set, binding, offset,
        static_cast<uint32_t>(bytes.size_bytes / sizeof(uint32_t)), static_cast<const uint32_t*>(bytes.data));
    return BASICRHI_C_OK;
}

basicrhi_c_result BASICRHI_C_CALL Draw(void* context, uint64_t token,
    uint32_t vertices, uint32_t instances, uint32_t firstVertex, uint32_t firstInstance)
{
    auto* state = static_cast<BorrowState*>(context);
    if (const auto result = Validate(state, token); result != BASICRHI_C_OK) return result;
    state->commandList->Draw(vertices, instances, firstVertex, firstInstance);
    return BASICRHI_C_OK;
}

basicrhi_c_result BASICRHI_C_CALL DrawIndexed(void* context, uint64_t token,
    uint32_t indices, uint32_t instances, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance)
{
    auto* state = static_cast<BorrowState*>(context);
    if (const auto result = Validate(state, token); result != BASICRHI_C_OK) return result;
    state->commandList->DrawIndexed(indices, instances, firstIndex, vertexOffset, firstInstance);
    return BASICRHI_C_OK;
}

basicrhi_c_result BASICRHI_C_CALL Dispatch(void* context, uint64_t token, uint32_t x, uint32_t y, uint32_t z)
{
    auto* state = static_cast<BorrowState*>(context);
    if (const auto result = Validate(state, token); result != BASICRHI_C_OK) return result;
    state->commandList->Dispatch(x, y, z);
    return BASICRHI_C_OK;
}

basicrhi_c_result BASICRHI_C_CALL DispatchMesh(void* context, uint64_t token, uint32_t x, uint32_t y, uint32_t z)
{
    auto* state = static_cast<BorrowState*>(context);
    if (const auto result = Validate(state, token); result != BASICRHI_C_OK) return result;
    state->commandList->DispatchMesh(x, y, z);
    return BASICRHI_C_OK;
}

basicrhi_c_result BASICRHI_C_CALL ExecuteIndirect(void* context, uint64_t token,
    basicrhi_c_command_signature signature, basicrhi_c_resource arguments, uint64_t argumentOffset,
    basicrhi_c_resource count, uint64_t countOffset, uint32_t maxCount)
{
    auto* state = static_cast<BorrowState*>(context);
    if (const auto result = Validate(state, token); result != BASICRHI_C_OK) return result;
    state->commandList->ExecuteIndirect(ConvertHandle<rhi::CommandSignatureHandle>(signature),
        ConvertHandle<rhi::ResourceHandle>(arguments), argumentOffset,
        ConvertHandle<rhi::ResourceHandle>(count), countOffset, maxCount);
    return BASICRHI_C_OK;
}

basicrhi_c_result BASICRHI_C_CALL CopyBuffer(void* context, uint64_t token,
    basicrhi_c_resource destination, uint64_t destinationOffset, basicrhi_c_resource source,
    uint64_t sourceOffset, uint64_t size)
{
    auto* state = static_cast<BorrowState*>(context);
    if (const auto result = Validate(state, token); result != BASICRHI_C_OK) return result;
    state->commandList->CopyBufferRegion(ConvertHandle<rhi::ResourceHandle>(destination), destinationOffset,
        ConvertHandle<rhi::ResourceHandle>(source), sourceOffset, size);
    return BASICRHI_C_OK;
}

basicrhi_c_result BASICRHI_C_CALL CopyTexture(void* context, uint64_t token,
    const basicrhi_c_texture_copy_region_v1* destination,
    const basicrhi_c_texture_copy_region_v1* source)
{
    auto* state = static_cast<BorrowState*>(context);
    if (const auto result = Validate(state, token); result != BASICRHI_C_OK) return result;
    if (!destination || !source) return BASICRHI_C_INVALID_ARGUMENT;
    const rhi::TextureCopyRegion dst{
        ConvertHandle<rhi::ResourceHandle>(destination->texture), destination->mip,
        destination->array_slice, destination->x, destination->y, destination->z,
        destination->width, destination->height, destination->depth };
    const rhi::TextureCopyRegion src{
        ConvertHandle<rhi::ResourceHandle>(source->texture), source->mip,
        source->array_slice, source->x, source->y, source->z,
        source->width, source->height, source->depth };
    state->commandList->CopyTextureRegion(dst, src);
    return BASICRHI_C_OK;
}

} // namespace

basicrhi_c_borrowed_recorder_v1 MakeRecorder(BorrowState& state)
{
    return { sizeof(basicrhi_c_borrowed_recorder_v1), BASICRHI_C_ABI_VERSION_1, &state, state.token,
        BindLayout, BindPipeline, SetDescriptorHeaps, PushConstants, Draw, DrawIndexed, Dispatch,
        DispatchMesh, ExecuteIndirect, CopyBuffer, CopyTexture };
}

} // namespace org::contributor
