#include "Render/External/ContributorRuntime.h"

#include "Render/PassBuilders.h"
#include "Render/PreparedPass.h"
#include "Render/RenderGraph/RenderGraph.h"
#include "RenderPasses/Base/ComputePass.h"
#include "RenderPasses/Base/CopyPass.h"
#include "RenderPasses/Base/RenderPass.h"
#include "Resources/GloballyIndexedResource.h"
#include "Resources/Buffers/Buffer.h"
#include "Resources/PixelBuffer.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace org::external
{
namespace
{
    template <class To, class From>
    To ConvertHandle(const From& value) noexcept
    {
        return To{ value.index, value.generation };
    }

    std::string CopyString(org_c_string_view value)
    {
        return value.data && value.size_bytes ? std::string(value.data, value.size_bytes) : std::string{};
    }

    struct BindingDescription {
        uint32_t kind = ORG_C_BINDING_SRV;
        std::string resource;
    };

    struct ContributorFrameState {
        mutable std::mutex mutex;
        org_c_frame_context frame{};

        org_c_frame_context Capture() const {
            std::scoped_lock lock(mutex);
            return frame;
        }
    };

    struct PassDescription {
        uint32_t kind = ORG_C_PASS_COMPUTE;
        std::string name;
        std::string technique;
        std::vector<BindingDescription> bindings;
        std::vector<std::string> after;
        std::vector<std::string> before;
        void* userData = nullptr;
        org_c_result (ORG_C_CALL *prepare)(void*, const org_c_pass_prepare_context*, org_c_prepared_pass*) = nullptr;
        std::shared_ptr<std::atomic_bool> active;
        std::shared_ptr<ContributorFrameState> frameState;
    };

    struct ResourceDescription {
        org_c_resource_desc value{};
        std::string id;
        std::string debugName;
    };

    struct ContributorState {
        const org_c_contributor_api* api = nullptr;
        uint64_t registration = 0;
        std::vector<PassDescription> passes;
        std::vector<ResourceDescription> resources;
        mutable std::unordered_map<std::string, std::shared_ptr<Resource>> persistentResources;
        std::shared_ptr<std::atomic_bool> active{ std::make_shared<std::atomic_bool>(true) };
        std::shared_ptr<ContributorFrameState> frameState{ std::make_shared<ContributorFrameState>() };
        bool initialized = false;
    };

    struct BorrowState {
        rhi::CommandList* commandList = nullptr;
        uint64_t token = 0;
        bool active = false;
    };

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

    basicrhi_c_borrowed_recorder_v1 MakeRecorder(BorrowState& state)
    {
        return { sizeof(basicrhi_c_borrowed_recorder_v1), BASICRHI_C_ABI_VERSION_1, &state, state.token,
            BindLayout, BindPipeline, SetDescriptorHeaps, PushConstants, Draw, DrawIndexed, Dispatch,
            DispatchMesh, ExecuteIndirect, CopyBuffer, CopyTexture };
    }

    rhi::Format ConvertFormat(uint32_t format)
    {
        switch (format) {
        case ORG_C_FORMAT_R8G8B8A8_UNORM: return rhi::Format::R8G8B8A8_UNorm;
        case ORG_C_FORMAT_R16G16_FLOAT: return rhi::Format::R16G16_Float;
        case ORG_C_FORMAT_R16G16B16A16_FLOAT: return rhi::Format::R16G16B16A16_Float;
        case ORG_C_FORMAT_R32_FLOAT: return rhi::Format::R32_Float;
        case ORG_C_FORMAT_R32_UINT: return rhi::Format::R32_UInt;
        case ORG_C_FORMAT_R32G32_UINT: return rhi::Format::R32G32_UInt;
        case ORG_C_FORMAT_D32_FLOAT: return rhi::Format::D32_Float;
        default: return rhi::Format::Unknown;
        }
    }

    uint32_t ScaledDimension(uint32_t absolute, uint32_t base, uint32_t numerator, uint32_t denominator)
    {
        if (absolute) return absolute;
        if (!numerator) numerator = 1;
        if (!denominator) denominator = 1;
        return std::max(1u, static_cast<uint32_t>((static_cast<uint64_t>(base) * numerator) / denominator));
    }

    std::shared_ptr<Resource> CreateResource(
        const ResourceDescription& source, uint32_t renderWidth, uint32_t renderHeight)
    {
        const auto& value = source.value;
        if (value.kind == ORG_C_RESOURCE_BUFFER) {
            const uint64_t elementCount = value.element_count_per_render_pixel
                ? value.element_count * renderWidth * renderHeight : value.element_count;
            if (!elementCount) throw std::invalid_argument("External buffer has zero elements");
            std::shared_ptr<Buffer> buffer;
            if (value.element_stride) {
                buffer = Buffer::CreateUnmaterializedStructuredBuffer(
                    static_cast<uint32_t>(elementCount), value.element_stride,
                    (value.usage_flags & ORG_C_RESOURCE_USAGE_UAV) != 0u, false, true);
            } else {
                buffer = Buffer::CreateSharedUnmaterialized(
                    rhi::HeapType::DeviceLocal, elementCount,
                    (value.usage_flags & ORG_C_RESOURCE_USAGE_UAV) != 0u);
            }
            buffer->SetName(source.debugName.empty() ? source.id : source.debugName);
            return buffer;
        }

        if (value.kind != ORG_C_RESOURCE_TEXTURE_2D) return nullptr;
        TextureDescription desc;
        desc.format = ConvertFormat(value.format);
        if (desc.format == rhi::Format::Unknown) throw std::invalid_argument("External texture has unsupported format");
        desc.channels = 4;
        desc.arraySize = std::max(1u, value.array_size);
        desc.isArray = desc.arraySize > 1;
        desc.hasSRV = (value.usage_flags & ORG_C_RESOURCE_USAGE_SRV) != 0u;
        desc.hasUAV = (value.usage_flags & ORG_C_RESOURCE_USAGE_UAV) != 0u;
        desc.hasNonShaderVisibleUAV = desc.hasUAV;
        desc.hasRTV = (value.usage_flags & ORG_C_RESOURCE_USAGE_RTV) != 0u;
        desc.hasDSV = (value.usage_flags & ORG_C_RESOURCE_USAGE_DSV) != 0u;
        desc.srvFormat = desc.uavFormat = desc.rtvFormat = desc.dsvFormat = desc.format;
        desc.allowAlias = value.lifetime == ORG_C_RESOURCE_TRANSIENT;
        desc.imageDimensions.push_back({
            ScaledDimension(value.width, renderWidth, value.width_scale_numerator, value.width_scale_denominator),
            ScaledDimension(value.height, renderHeight, value.height_scale_numerator, value.height_scale_denominator),
            0, 0 });
        for (size_t index = 0; index < 4; ++index)
            std::memcpy(&desc.clearColor[index], &value.clear_value[index], sizeof(float));
        auto texture = PixelBuffer::CreateSharedUnmaterialized(desc);
        texture->SetName(source.debugName.empty() ? source.id : source.debugName);
        return texture;
    }

    void AddBinding(RenderPassBuilder& builder, const BindingDescription& binding)
    {
        const ResourceIdentifier id{ binding.resource };
        switch (binding.kind) {
        case ORG_C_BINDING_SRV: builder.WithShaderResource(id); break;
        case ORG_C_BINDING_UAV: builder.WithUnorderedAccess(id); break;
        case ORG_C_BINDING_RTV: builder.WithRenderTarget(id); break;
        case ORG_C_BINDING_DSV: builder.WithDepthReadWrite(id); break;
        case ORG_C_BINDING_INDIRECT: builder.WithIndirectArguments(id); break;
        case ORG_C_BINDING_COPY_SOURCE: builder.WithCopySource(id); break;
        case ORG_C_BINDING_COPY_DESTINATION: builder.WithCopyDest(id); break;
        default: break;
        }
    }

    void AddBinding(ComputePassBuilder& builder, const BindingDescription& binding)
    {
        const ResourceIdentifier id{ binding.resource };
        switch (binding.kind) {
        case ORG_C_BINDING_SRV: builder.WithShaderResource(id); break;
        case ORG_C_BINDING_UAV: builder.WithUnorderedAccess(id); break;
        case ORG_C_BINDING_INDIRECT: builder.WithIndirectArguments(id); break;
        default: break;
        }
    }

    void AddBinding(CopyPassBuilder& builder, const BindingDescription& binding)
    {
        const ResourceIdentifier id{ binding.resource };
        if (binding.kind == ORG_C_BINDING_COPY_SOURCE) builder.WithCopySource(id);
        else if (binding.kind == ORG_C_BINDING_COPY_DESTINATION) builder.WithCopyDest(id);
    }

    class ExternalPassState
    {
    public:
        explicit ExternalPassState(std::shared_ptr<const PassDescription> description)
            : m_description(std::move(description)) {}

        void Setup(const std::shared_ptr<ResourceRegistryView>& registry)
        {
            m_resources.clear();
            for (const auto& binding : m_description->bindings)
                m_resources.push_back(registry->RequestShared(ResourceIdentifier{ binding.resource }));
        }

        struct PacketOwner {
            org_c_prepared_pass packet{};
            ~PacketOwner() { if (packet.destroy) packet.destroy(packet.data); }
        };

        struct PreparedData {
            std::shared_ptr<PacketOwner> owner;
            std::shared_ptr<const PassDescription> description;
            std::vector<std::shared_ptr<Resource>> resources;
            std::vector<org_c_binding> bindings;
            uint64_t frameIndex = 0;
            uint64_t completionValue = 0;
        };

        struct PacketCallbacks {
            static void Record(const PreparedData& data, RecordingContext& recording)
            {
                const auto& packet = data.owner->packet;
                if (!packet.record || !data.description->active->load(std::memory_order_acquire)) return;
                BorrowState borrow{ &recording.Commands(), ++s_nextBorrowToken, true };
                org_c_record_context context{ sizeof(context), ORG_CONTRIBUTOR_API_VERSION,
                    data.frameIndex, data.completionValue, MakeRecorder(borrow),
                    data.bindings.data(), data.bindings.size() };
                packet.record(packet.data, &context);
                borrow.active = false;
                borrow.commandList = nullptr;
            }
            static void Submitted(const PreparedData& data, const SubmissionContext& context)
            { if (data.owner->packet.submitted) data.owner->packet.submitted(data.owner->packet.data, context.submissionID); }
            static void Completed(const PreparedData& data, const CompletionContext& context)
            { if (data.owner->packet.completed) data.owner->packet.completed(data.owner->packet.data, context.submissionID); }
            static void Abandoned(const PreparedData& data, AbandonReason reason)
            {
                if (!data.owner->packet.abandoned) return;
                uint32_t externalReason = ORG_C_ABANDON_SHUTDOWN;
                switch (reason) {
                case AbandonReason::Shutdown: externalReason = ORG_C_ABANDON_SHUTDOWN; break;
                case AbandonReason::GenerationInvalidated: externalReason = ORG_C_ABANDON_GENERATION_INVALIDATED; break;
                case AbandonReason::PreparationFailed: externalReason = ORG_C_ABANDON_PREPARATION_FAILED; break;
                case AbandonReason::AdmissionFailed: externalReason = ORG_C_ABANDON_ADMISSION_FAILED; break;
                }
                data.owner->packet.abandoned(data.owner->packet.data, externalReason);
            }
            inline static std::atomic<uint64_t> s_nextBorrowToken{ 0 };
        };

        PreparedPass PrepareFrame()
        {
            if (!m_description->prepare || !m_description->active ||
                !m_description->active->load(std::memory_order_acquire)) return PreparedPass::NoOp();
            std::vector<org_c_binding> bindings;
            bindings.reserve(m_description->bindings.size());
            for (size_t index = 0; index < m_description->bindings.size(); ++index) {
                const auto& description = m_description->bindings[index];
                auto* resource = m_resources[index].get();
                org_c_binding binding{};
                binding.structure_size = sizeof(binding);
                binding.kind = description.kind;
                binding.symbolic_resource = { description.resource.data(), description.resource.size() };
                if (resource) {
                    binding.resource = ConvertHandle<basicrhi_c_resource>(resource->GetAPIResource().GetHandle());
                    if (const auto* indexed = dynamic_cast<const GloballyIndexedResource*>(resource)) {
                        rhi::DescriptorSlot slot{};
                        if (description.kind == ORG_C_BINDING_SRV && indexed->HasSRV()) slot = indexed->GetSRVInfo(0).slot;
                        else if (description.kind == ORG_C_BINDING_UAV && indexed->HasUAVShaderVisible()) slot = indexed->GetUAVShaderVisibleInfo(0).slot;
                        else if (description.kind == ORG_C_BINDING_RTV && indexed->HasRTV()) slot = indexed->GetRTVInfo(0).slot;
                        else if (description.kind == ORG_C_BINDING_DSV && indexed->HasDSV()) slot = indexed->GetDSVInfo(0).slot;
                        binding.descriptor_heap = ConvertHandle<basicrhi_c_descriptor_heap>(slot.heap);
                        binding.descriptor_slot = slot.index;
                    }
                }
                bindings.push_back(binding);
            }
            const auto frame = m_description->frameState->Capture();
            org_c_pass_prepare_context context{ sizeof(context), ORG_CONTRIBUTOR_API_VERSION,
                &frame, bindings.data(), bindings.size() };
            auto owner = std::make_shared<PacketOwner>();
            owner->packet.structure_size = sizeof(org_c_prepared_pass);
            owner->packet.api_version = ORG_CONTRIBUTOR_API_VERSION;
            if (m_description->prepare(m_description->userData, &context, &owner->packet) != ORG_C_OK ||
                owner->packet.structure_size < sizeof(org_c_prepared_pass) ||
                owner->packet.api_version != ORG_CONTRIBUTOR_API_VERSION || !owner->packet.record)
                throw std::runtime_error("External contributor failed to prepare an owned pass packet");
            return PreparedPass::FromTyped<PacketCallbacks>(PreparedData{
                std::move(owner), m_description, m_resources, std::move(bindings),
                frame.frame_index, frame.completion_value });
        }

        PassReturn ExecuteInline(PassExecutionContext& execution)
        {
            auto packet = PrepareFrame();
            auto bindings = std::make_shared<const FrozenExecutionBindings>(
                std::vector<FrozenExecutionBindings::ResourceBinding>{});
            auto external = std::make_shared<const std::vector<ExternalDescriptorBindingValue>>(
                execution.externalDescriptorBindings);
            RecordingContext recording(execution.commandList, std::move(bindings), std::move(external));
            packet.Record(recording);
            return {};
        }

        const PassDescription& Description() const noexcept { return *m_description; }

    private:
        std::shared_ptr<const PassDescription> m_description;
        std::vector<std::shared_ptr<Resource>> m_resources;
    };

    class ExternalRenderPass final : public RenderPass {
    public:
        explicit ExternalRenderPass(std::shared_ptr<const PassDescription> description) : m_state(std::move(description)) {}
        void Setup() override { m_state.Setup(m_resourceRegistryView); }
        PreparedPass PrepareFrame(FramePreparationContext&) override { return m_state.PrepareFrame(); }
        PassReturn Execute(PassExecutionContext& context) override { return m_state.ExecuteInline(context); }
        void Cleanup() override {}
    protected:
        void DeclareResourceUsages(RenderPassBuilder* builder) override
        { for (const auto& binding : m_state.Description().bindings) AddBinding(*builder, binding); }
    private:
        ExternalPassState m_state;
    };

    class ExternalComputePass final : public ComputePass {
    public:
        explicit ExternalComputePass(std::shared_ptr<const PassDescription> description) : m_state(std::move(description)) {}
        void Setup() override { m_state.Setup(m_resourceRegistryView); }
        PreparedPass PrepareFrame(FramePreparationContext&) override { return m_state.PrepareFrame(); }
        PassReturn Execute(PassExecutionContext& context) override { return m_state.ExecuteInline(context); }
        void Cleanup() override {}
    protected:
        void DeclareResourceUsages(ComputePassBuilder* builder) override
        { for (const auto& binding : m_state.Description().bindings) AddBinding(*builder, binding); }
    private:
        ExternalPassState m_state;
    };

    class ExternalCopyPass final : public CopyPass {
    public:
        explicit ExternalCopyPass(std::shared_ptr<const PassDescription> description) : m_state(std::move(description)) {}
        void Setup() override { m_state.Setup(m_resourceRegistryView); }
        PreparedPass PrepareFrame(FramePreparationContext&) override { return m_state.PrepareFrame(); }
        PassReturn Execute(PassExecutionContext& context) override { return m_state.ExecuteInline(context); }
        void Cleanup() override {}
    protected:
        void DeclareResourceUsages(CopyPassBuilder* builder) override
        { for (const auto& binding : m_state.Description().bindings) AddBinding(*builder, binding); }
    private:
        ExternalPassState m_state;
    };
}

struct ContributorRuntime::Impl {
    mutable std::mutex mutex;
    uint64_t nextRegistration = 1;
    std::vector<std::shared_ptr<ContributorState>> contributors;
    org_c_host_api host{};
    org_c_frame_context frame{};
};

ContributorRuntime::ContributorRuntime() : m_impl(std::make_unique<Impl>()) {}
ContributorRuntime::~ContributorRuntime()
{
    std::scoped_lock lock(m_impl->mutex);
    for (const auto& state : m_impl->contributors) {
        state->active->store(false, std::memory_order_release);
        if (state->api->on_graph_unregistered) state->api->on_graph_unregistered(state->api->contributor);
        if (state->initialized && state->api->shutdown) state->api->shutdown(state->api->contributor);
    }
}

void ContributorRuntime::SetHostAPI(const org_c_host_api& host)
{
    if (host.structure_size < sizeof(host) || host.api_version != ORG_CONTRIBUTOR_API_VERSION)
        throw std::invalid_argument("Invalid ORG contributor host API");
    std::scoped_lock lock(m_impl->mutex);
    m_impl->host = host;
}

bool ContributorRuntime::Register(const org_c_contributor_api* api, uint64_t& registration)
{
    registration = 0;
    if (!api) return false;
    if (api->api_version != ORG_CONTRIBUTOR_API_VERSION) {
        spdlog::error("Rejected OpenRenderGraph contributor ABI version {}; host requires owned prepare/record ABI version {}",
            api->api_version, ORG_CONTRIBUTOR_API_VERSION);
        return false;
    }
    if (api->structure_size < sizeof(*api) || !api->describe_passes) return false;
    const size_t count = api->describe_passes(api->contributor, nullptr, 0);
    std::vector<org_c_pass> descriptions(count);
    for (auto& pass : descriptions) pass.structure_size = sizeof(pass);
    if (api->describe_passes(api->contributor, descriptions.data(), descriptions.size()) != count) return false;

    auto state = std::make_shared<ContributorState>();
    state->api = api;
    if (api->describe_resources) {
        const size_t resourceCount = api->describe_resources(api->contributor, nullptr, 0);
        std::vector<org_c_resource_desc> resources(resourceCount);
        for (auto& resource : resources) resource.structure_size = sizeof(resource);
        if (api->describe_resources(api->contributor, resources.data(), resources.size()) != resourceCount) return false;
        state->resources.reserve(resourceCount);
        for (const auto& source : resources) {
            if (source.structure_size < sizeof(source) || source.kind > ORG_C_RESOURCE_BUFFER ||
                source.lifetime > ORG_C_RESOURCE_IMPORTED) return false;
            ResourceDescription destination;
            destination.value = source;
            destination.id = CopyString(source.symbolic_resource);
            destination.debugName = CopyString(source.debug_name);
            if (destination.id.empty()) return false;
            destination.value.symbolic_resource = {};
            destination.value.debug_name = {};
            state->resources.push_back(std::move(destination));
        }
    }
    state->passes.reserve(count);
    for (const auto& source : descriptions) {
        if (source.structure_size < sizeof(source) || source.kind > ORG_C_PASS_COPY || !source.prepare ||
            (!source.declared_bindings && source.declared_binding_count)) return false;
        PassDescription destination;
        destination.kind = source.kind;
        destination.name = CopyString(source.name);
        destination.technique = CopyString(source.technique_path);
        destination.userData = source.user_data;
        destination.prepare = source.prepare;
        destination.active = state->active;
        destination.frameState = state->frameState;
        if (destination.name.empty()) return false;
        for (size_t index = 0; index < source.declared_binding_count; ++index) {
            const auto& binding = source.declared_bindings[index];
            if (binding.structure_size < sizeof(binding) || binding.kind > ORG_C_BINDING_COPY_DESTINATION) return false;
            destination.bindings.push_back({ binding.kind, CopyString(binding.symbolic_resource) });
            if (destination.bindings.back().resource.empty()) return false;
        }
        if ((!source.after_passes && source.after_pass_count) || (!source.before_passes && source.before_pass_count))
            return false;
        for (size_t index = 0; index < source.after_pass_count; ++index) {
            destination.after.push_back(CopyString(source.after_passes[index]));
            if (destination.after.back().empty()) return false;
        }
        for (size_t index = 0; index < source.before_pass_count; ++index) {
            destination.before.push_back(CopyString(source.before_passes[index]));
            if (destination.before.back().empty()) return false;
        }
        state->passes.push_back(std::move(destination));
    }

    std::scoped_lock lock(m_impl->mutex);
    if (api->initialize) {
        if (m_impl->host.structure_size < sizeof(m_impl->host) ||
            api->initialize(api->contributor, &m_impl->host) != ORG_C_OK) return false;
        state->initialized = true;
    }
    state->registration = m_impl->nextRegistration++;
    m_impl->contributors.push_back(state);
    registration = state->registration;
    if (api->on_graph_registered) api->on_graph_registered(api->contributor, registration);
    return true;
}

bool ContributorRuntime::Unregister(uint64_t registration)
{
    std::scoped_lock lock(m_impl->mutex);
    const auto found = std::find_if(m_impl->contributors.begin(), m_impl->contributors.end(),
        [&](const auto& value) { return value->registration == registration; });
    if (found == m_impl->contributors.end()) return false;
    (*found)->active->store(false, std::memory_order_release);
    if ((*found)->api->on_graph_unregistered)
        (*found)->api->on_graph_unregistered((*found)->api->contributor);
    if ((*found)->initialized && (*found)->api->shutdown)
        (*found)->api->shutdown((*found)->api->contributor);
    m_impl->contributors.erase(found);
    return true;
}

bool ContributorRuntime::PrepareFrame(const org_c_frame_context& frame) const
{
    if (frame.structure_size < sizeof(frame) || frame.api_version != ORG_CONTRIBUTOR_API_VERSION ||
        frame.render_width == 0 || frame.render_height == 0) return false;
    std::scoped_lock lock(m_impl->mutex);
    m_impl->frame = frame;
    for (const auto& state : m_impl->contributors) {
        {
            std::scoped_lock frameLock(state->frameState->mutex);
            state->frameState->frame = frame;
        }
        if (state->api->prepare_frame && state->api->prepare_frame(state->api->contributor, &frame) != ORG_C_OK)
            return false;
    }
    return true;
}

void ContributorRuntime::BuildPasses(RenderGraph& graph) const
{
    std::scoped_lock lock(m_impl->mutex);
    for (const auto& contributor : m_impl->contributors) {
        for (const auto& resource : contributor->resources) {
            if (resource.value.lifetime == ORG_C_RESOURCE_IMPORTED) continue;
            std::shared_ptr<Resource> instance;
            if (resource.value.lifetime == ORG_C_RESOURCE_CONTRIBUTOR_PERSISTENT) {
                auto& persistent = contributor->persistentResources[resource.id];
                if (!persistent) persistent = CreateResource(resource, m_impl->frame.render_width, m_impl->frame.render_height);
                instance = persistent;
            } else {
                instance = CreateResource(resource, m_impl->frame.render_width, m_impl->frame.render_height);
            }
            graph.RegisterResource(ResourceIdentifier{ resource.id }, std::move(instance));
        }
        for (const auto& pass : contributor->passes) {
            auto description = std::make_shared<PassDescription>(pass);
            if (pass.kind == ORG_C_PASS_RASTER) graph.BuildRenderPass<ExternalRenderPass>(pass.name, description);
            else if (pass.kind == ORG_C_PASS_COMPUTE) graph.BuildComputePass<ExternalComputePass>(pass.name, description);
            else graph.BuildCopyPass<ExternalCopyPass>(pass.name, description);
            if (!pass.technique.empty()) graph.SetPassTechnique(pass.name, pass.technique);
            for (const auto& anchor : pass.after) graph.AddExplicitPassDependency(anchor, pass.name);
            for (const auto& anchor : pass.before) graph.AddExplicitPassDependency(pass.name, anchor);
        }
    }
}

void ContributorRuntime::NotifyDeviceLost() const
{
    std::scoped_lock lock(m_impl->mutex);
    for (const auto& state : m_impl->contributors) {
        if (state->api->on_device_lost) state->api->on_device_lost(state->api->contributor);
        state->persistentResources.clear();
    }
}

void ContributorRuntime::NotifyGraphRebuilt(uint64_t revision) const
{
    std::scoped_lock lock(m_impl->mutex);
    for (const auto& state : m_impl->contributors)
        if (state->api->on_graph_rebuilt) state->api->on_graph_rebuilt(state->api->contributor, revision);
}

bool ContributorRuntime::NotifyDeviceRestored(const org_c_host_api& host) const
{
    if (host.structure_size < sizeof(host) || host.api_version != ORG_CONTRIBUTOR_API_VERSION) return false;
    std::scoped_lock lock(m_impl->mutex);
    m_impl->host = host;
    for (const auto& state : m_impl->contributors)
        if (state->api->on_device_restored &&
            state->api->on_device_restored(state->api->contributor, &m_impl->host) != ORG_C_OK) return false;
    return true;
}

size_t ContributorRuntime::ContributorCount() const noexcept
{
    std::scoped_lock lock(m_impl->mutex);
    return m_impl->contributors.size();
}
}
