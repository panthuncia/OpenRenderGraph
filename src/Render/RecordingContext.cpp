#include "Render/PreparedPass.h"

#include <algorithm>
#include <atomic>
#include <string>

#include "Render/FeatureDomainRegistry.h"
#include "Render/ShaderAPI.h"

#include <BasicTelemetry/Telemetry.h>
#include <spdlog/spdlog.h>

namespace org {

rhi::Resource RecordingContext::Resolve(ResourceBindingToken token) const {
    if (!m_declaredResourceSlots) throw std::logic_error("Declaration tokens require a declared recording scope");
    auto found = m_declaredResourceSlots->Find(token.globalResourceID);
    if (found == m_declaredResourceSlots->end()) found = m_declaredResourceSlots->Find(token.registryResourceID);
    if (found == m_declaredResourceSlots->end())
        throw std::invalid_argument("Recording attempted to resolve an undeclared resource");
    return m_bindings->Resolve(PreparedResourceReference{found->second});
}

rhi::DescriptorSlot RecordingContext::View(const DeclaredViewToken& token) const {
    if (!m_declaredScope || !m_declaredScope->layout) throw std::logic_error("Declared views require a declared recording scope");
    if (token.layout != m_declaredScope->layout) throw std::invalid_argument("Foreign or stale declared view token");
    const auto flat = m_declaredScope->layout->FlatView(token.use, token.view);
    const auto slot = flat < m_declaredScope->views.size() ? m_declaredScope->views[flat] : rhi::DescriptorSlot{};
    if (!slot.heap.valid())
        throw std::out_of_range("Declared view has no descriptor this frame; resource="
            + std::to_string(m_declaredScope->layout->uses[token.use].binding.globalResourceID));
    // A shader-visible index is only meaningful in the heap the list binds.
    const auto kind = m_declaredScope->layout->uses[token.use].requiredViews[token.view].kind;
    if (m_resourceHeap.valid() && (kind == BindlessViewKind::ShaderResource || kind == BindlessViewKind::UnorderedAccess
        || kind == BindlessViewKind::ConstantBuffer)
        && (slot.heap.index != m_resourceHeap.index || slot.heap.generation != m_resourceHeap.generation)) {
        basic_telemetry::AddCounter("ORG.Execution.ViewHeapMismatches");
        static std::atomic<uint32_t> reported{0};
        if (reported.fetch_add(1) < 4)
            spdlog::warn("Declared view of resource {} lives in another descriptor heap than the recording binds",
                m_declaredScope->layout->uses[token.use].binding.globalResourceID);
    }
    return slot;
}

rhi::DescriptorSlot RecordingContext::View(ResourceBindingToken resource, BindlessViewRequest request) const {
    if (!m_declaredResourceSlots) throw std::logic_error("Declaration tokens require a declared recording scope");
    auto found = m_declaredResourceSlots->Find(resource.globalResourceID);
    if (found == m_declaredResourceSlots->end()) found = m_declaredResourceSlots->Find(resource.registryResourceID);
    if (found == m_declaredResourceSlots->end())
        throw std::invalid_argument("Recording attempted to resolve an undeclared resource");
    return m_bindings->Views(PreparedResourceReference{found->second}).Resolve(request);
}

void RecordingContext::Retain(std::shared_ptr<const void> owner) {
    if (!m_retention) m_retention = std::make_shared<RecordingRetention>();
    m_retention->Retain(std::move(owner));
}

std::vector<uint32_t> RecordingContext::DescriptorSlotIndices(const PipelineStatePayload& payload, BindOptions options) const {
    const auto& slots = payload.pipelineResources;
    std::vector<uint32_t> indices;
    indices.reserve(slots.mandatoryResourceDescriptorSlots.size() + slots.optionalResourceDescriptorSlots.size());
    const ResourceUseLayout* layout = m_declaredScope ? m_declaredScope->layout.get() : nullptr;
    const auto resolve = [&](const ResourceIdentifier& name, bool optional) -> uint32_t {
        if (layout)
            if (const auto* named = layout->FindNamedView(name.hash)) {
                if (named->resolverGroup == UINT32_MAX)
                    return View(DeclaredViewToken{m_declaredScope->layout, named->use, named->view}).index;
                const auto ordinal = static_cast<size_t>(named - layout->namedViews.data());
                const auto& groupViews = m_declaredScope->namedGroupViews;
                if (ordinal < groupViews.size() && groupViews[ordinal].heap.valid()) return groupViews[ordinal].index;
                if (optional) return UINT32_MAX;
                throw std::invalid_argument("Pipeline " + payload.label + " uses " + name.name
                    + ", which its pass declared through a resolver with no member this frame");
            }
        if (optional) return UINT32_MAX;
        for (const auto prefix : options.absentPrefixes)
            if (std::string_view{name.name}.starts_with(prefix)) return UINT32_MAX;
        if (const auto domain = FeatureDomainRegistry::Get().FindResourceDomain(name)) {
            const bool active = layout && std::find(layout->activeFeatureDomains.begin(),
                layout->activeFeatureDomains.end(), domain->hash) != layout->activeFeatureDomains.end();
            if (!active) return UINT32_MAX;
        }
        throw std::invalid_argument("Pipeline " + payload.label + " uses " + name.name
            + " but its pass did not declare it");
    };
    for (const auto& name : slots.mandatoryResourceDescriptorSlots) indices.push_back(resolve(name, false));
    for (const auto& name : slots.optionalResourceDescriptorSlots) indices.push_back(resolve(name, true));
    if (indices.size() > shaderapi::kNumResourceDescriptorIndicesRootConstants)
        throw std::out_of_range("Pipeline " + payload.label + " has more descriptor slots than the root constants hold");
    return indices;
}

void RecordingContext::BindPayload(std::shared_ptr<const PipelineStatePayload> payload, rhi::ShaderStage stages, BindOptions options) {
    if (!payload || !payload->pso) throw std::invalid_argument("Pipeline has no immutable payload");
    const auto indices = DescriptorSlotIndices(*payload, options);
    m_commands.BindLayout(payload->layout);
    m_commands.BindPipeline(payload->pso.Get().GetHandle());
    m_boundStages = stages;
    if (!indices.empty())
        m_commands.PushConstants(stages, 0, shaderapi::kResourceDescriptorIndicesRootParameter, 0,
            static_cast<uint32_t>(indices.size()), indices.data());
    Retain(std::move(payload));
}

void RecordingContext::BindComputePipeline(const PipelineState& pipeline, BindOptions options) {
    BindPayload(pipeline.GetPayload(m_declaredScope ? m_declaredScope->backendInstance : BackendInstanceId::Primary),
        rhi::ShaderStage::Compute, options);
}
void RecordingContext::BindGraphicsPipeline(const PipelineState& pipeline, BindOptions options) {
    BindPayload(pipeline.GetPayload(m_declaredScope ? m_declaredScope->backendInstance : BackendInstanceId::Primary),
        rhi::ShaderStage::AllGraphics, options);
}
void RecordingContext::BindComputePipeline(std::shared_ptr<const PipelineStatePayload> payload, BindOptions options) {
    BindPayload(std::move(payload), rhi::ShaderStage::Compute, options);
}
void RecordingContext::BindGraphicsPipeline(std::shared_ptr<const PipelineStatePayload> payload, BindOptions options) {
    BindPayload(std::move(payload), rhi::ShaderStage::AllGraphics, options);
}

namespace {
void BindRawPipeline(RecordingContext& context, rhi::CommandList& commands,
    std::shared_ptr<const rhi::PipelinePtr> pipeline, rhi::PipelineLayoutHandle layout) {
    if (!pipeline || !*pipeline) throw std::invalid_argument("Cannot bind an empty pipeline");
    if (layout.valid()) commands.BindLayout(layout);
    commands.BindPipeline(pipeline->Get().GetHandle());
    context.Retain(std::move(pipeline));
}
}

void RecordingContext::BindComputePipeline(std::shared_ptr<const rhi::PipelinePtr> pipeline, rhi::PipelineLayoutHandle layout) {
    BindRawPipeline(*this, m_commands, std::move(pipeline), layout);
    m_boundStages = rhi::ShaderStage::Compute;
}
void RecordingContext::BindGraphicsPipeline(std::shared_ptr<const rhi::PipelinePtr> pipeline, rhi::PipelineLayoutHandle layout) {
    BindRawPipeline(*this, m_commands, std::move(pipeline), layout);
    m_boundStages = rhi::ShaderStage::AllGraphics;
}

void RecordingContext::PushConstants(uint32_t rootParameter, std::span<const uint32_t> values, uint32_t destinationOffset) {
    if (values.empty()) return;
    m_commands.PushConstants(m_boundStages, 0, rootParameter, destinationOffset,
        static_cast<uint32_t>(values.size()), values.data());
}

void RecordingContext::BufferBarriers(std::initializer_list<ResourceBindingToken> buffers,
    rhi::ResourceAccessType beforeAccess, rhi::ResourceAccessType afterAccess,
    rhi::ResourceSyncState beforeSync, rhi::ResourceSyncState afterSync) {
    std::vector<rhi::BufferBarrier> barriers;
    barriers.reserve(buffers.size());
    for (const auto& token : buffers) {
        rhi::BufferBarrier barrier{};
        barrier.buffer = Resolve(token).GetHandle();
        barrier.beforeAccess = beforeAccess;
        barrier.afterAccess = afterAccess;
        barrier.beforeSync = beforeSync;
        barrier.afterSync = afterSync;
        barriers.push_back(barrier);
    }
    if (barriers.empty()) return;
    rhi::BarrierBatch batch{};
    batch.buffers = rhi::Span<rhi::BufferBarrier>(barriers.data(), static_cast<uint32_t>(barriers.size()));
    m_commands.Barriers(batch);
}

void RecordingContext::UavBarrier() {
    rhi::GlobalBarrier barrier{};
    barrier.beforeSync = barrier.afterSync = rhi::ResourceSyncState::ComputeShading;
    barrier.beforeAccess = barrier.afterAccess = rhi::ResourceAccessType::UnorderedAccess;
    rhi::BarrierBatch batch{};
    batch.globals = {&barrier, 1};
    m_commands.Barriers(batch);
}

void RecordingContext::ExecuteIndirect(const std::shared_ptr<const rhi::CommandSignaturePtr>& signature,
    ResourceBindingToken arguments, uint64_t argumentOffset, uint32_t maxCommandCount,
    std::optional<ResourceBindingToken> count, uint64_t countOffset) {
    if (!signature || !*signature) throw std::invalid_argument("Cannot execute an empty command signature");
    m_commands.ExecuteIndirect(signature->Get().GetHandle(), Resolve(arguments).GetHandle(), argumentOffset,
        count ? Resolve(*count).GetHandle() : rhi::ResourceHandle{}, countOffset, maxCommandCount);
    Retain(signature);
}

void RecordingContext::SetWorkGraph(const std::shared_ptr<const rhi::WorkGraphPtr>& workGraph,
    ResourceBindingToken backing, bool initializeBacking) {
    if (!workGraph || !*workGraph) throw std::invalid_argument("Cannot set an empty work graph");
    m_commands.SetWorkGraph(workGraph->Get().GetHandle(), Resolve(backing).GetHandle(), initializeBacking);
    Retain(workGraph);
}

} // namespace org
