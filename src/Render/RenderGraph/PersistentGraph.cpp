#include "Render/RenderGraph/PersistentGraph.h"
#include "Render/PreparedPass.h"
#include <stdexcept>
#include <exception>
#include <algorithm>
#include <iterator>
#include <BasicTelemetry/Tracy.h>
#include <spdlog/spdlog.h>
#include <BasicTelemetry/Telemetry.h>

namespace org::persistent {
namespace {
std::atomic<uint64_t> nextProgramDomain{1};
std::atomic<uint64_t> nextBindingDeclaration{1};
uint64_t TakeIdentity(std::atomic<uint64_t>& counter) {
    auto value = counter.load(std::memory_order_relaxed);
    while (value && value != UINT64_MAX) {
        if (counter.compare_exchange_weak(value,value+1,std::memory_order_relaxed)) return value;
    }
    throw std::overflow_error("Persistent graph identity exhausted");
}
struct MutationGuard {
    bool& failed;
    int exceptions = std::uncaught_exceptions();
    ~MutationGuard() { if (std::uncaught_exceptions() > exceptions) failed = true; }
};
void Require(bool value, const char* message) {
    if (!value) throw std::invalid_argument(message);
}
void ValidateSharedBacking(const BindingVersion& a, const BindingVersion& b) {
    Require(a.shape == b.shape && a.backingRevision == b.backingRevision && a.contentRevision == b.contentRevision,
        "Shared physical slots disagree on backing or content version");
    Require(bool(a.recording) == bool(b.recording) && bool(a.admission) == bool(b.admission),
        "Shared physical slots disagree on binding metadata");
    if (a.recording) {
        const auto& x = *a.recording; const auto& y = *b.recording;
        Require(x.allocationOwner.get() == y.allocationOwner.get()
            && !x.allocationOwner.owner_before(y.allocationOwner) && !y.allocationOwner.owner_before(x.allocationOwner),
            "Shared physical slots disagree on allocation ownership");
        if (x.description.type != rhi::ResourceType::Unknown || y.description.type != rhi::ResourceType::Unknown)
            Require(NativeBindingContract::Capture(x.description).Matches(y.description,a.shape)
                && NativeBindingContract::Capture(y.description).Matches(x.description,b.shape),
                "Shared physical slots disagree on native description");
    }
    if (a.admission) {
        const auto& x = *a.admission; const auto& y = *b.admission;
        Require(x.resource.index == y.resource.index && x.resource.generation == y.resource.generation
            && x.heapType == y.heapType && x.aliasHeapIdentity == y.aliasHeapIdentity
            && x.aliasHeap.get() == y.aliasHeap.get() && !x.aliasHeap.owner_before(y.aliasHeap)
            && !y.aliasHeap.owner_before(x.aliasHeap) && x.aliasPoolID == y.aliasPoolID
            && x.aliasOffset == y.aliasOffset && x.aliasSize == y.aliasSize,
            "Shared physical slots disagree on placement");
        Require(x.regions && y.regions && x.regions->size() == y.regions->size(),
            "Shared physical slots disagree on incoming state");
        for (size_t i = 0; i < x.regions->size(); ++i)
            Require(x.regions->at(i).range == y.regions->at(i).range && x.regions->at(i).state == y.regions->at(i).state,
                "Shared physical slots disagree on incoming state");
    }
}
void CanonicalizeUses(experimental::GraphCompileStructure& structure, const std::vector<uint32_t>& canonical) {
    for (auto& pass : structure.passes) {
        for (auto& access : pass.accesses) access.resourceIndex = canonical.at(access.resourceIndex);
        for (auto* uses : {&pass.entryStates,&pass.exitStates}) {
            for (auto& use : *uses) use.resource = canonical.at(use.resource);
            for (size_t a = 0; a < uses->size(); ++a) for (size_t b = a+1; b < uses->size(); ++b) {
                const auto& x = uses->at(a); const auto& y = uses->at(b);
                if (x.resource != y.resource || uint64_t{x.range.mip}+x.range.mips <= y.range.mip
                    || uint64_t{y.range.mip}+y.range.mips <= x.range.mip
                    || uint64_t{x.range.slice}+x.range.slices <= y.range.slice
                    || uint64_t{y.range.slice}+y.range.slices <= x.range.slice) continue;
                Require(x.state == y.state,"Overlapping physical declarations have incompatible states");
            }
        }
    }
}
void ValidateNativeView(const rhi::ResourceDesc& desc, BindlessViewKind kind) {
    if (desc.type == rhi::ResourceType::Unknown) return; // Numeric-only fixtures.
    const auto flags = static_cast<uint32_t>(desc.resourceFlags);
    const auto has = [&](rhi::ResourceFlags flag) { return (flags & static_cast<uint32_t>(flag)) != 0; };
    switch (kind) {
    case BindlessViewKind::ShaderResource:
        Require(!has(rhi::ResourceFlags::RF_DenyShaderResource), "Backing denies shader-resource views"); break;
    case BindlessViewKind::UnorderedAccess:
    case BindlessViewKind::NonShaderVisibleUnorderedAccess:
        Require(has(rhi::ResourceFlags::RF_AllowUnorderedAccess), "Backing does not allow unordered-access views");
        Require(desc.type == rhi::ResourceType::Buffer || (desc.type != rhi::ResourceType::AccelerationStructure
            && desc.texture.sampleCount == 1), "Backing type does not support unordered-access views"); break;
    case BindlessViewKind::RenderTarget:
        Require(has(rhi::ResourceFlags::RF_AllowRenderTarget), "Backing does not allow render-target views"); break;
    case BindlessViewKind::DepthStencil:
        Require(has(rhi::ResourceFlags::RF_AllowDepthStencil)
            && (desc.type == rhi::ResourceType::Texture1D || desc.type == rhi::ResourceType::Texture2D),
            "Backing does not support depth-stencil views"); break;
    case BindlessViewKind::ConstantBuffer:
        Require(desc.type == rhi::ResourceType::Buffer, "Constant-buffer view requires a buffer backing"); break;
    default: throw std::invalid_argument("Unsupported published view kind");
    }
}
std::shared_ptr<const experimental::GraphExecutionLayout> BuildPersistentExecutionLayout(
    uint64_t revision, std::shared_ptr<const experimental::CompiledGraph> graph,
    std::shared_ptr<const experimental::GraphCompileInput> input, size_t passCapacity) {
    BT_ZONE_SCOPE("ORG.Persistent.BuildExecutionLayout");
    Require(graph && input && graph->structure.get() == &input->structure
        && graph->scheduleValidated && graph->scheduleValidationError.empty(), "Invalid persistent execution layout");
    auto layout = std::make_shared<experimental::GraphExecutionLayout>();
    layout->bundle = std::make_shared<const experimental::CompiledGraphBundle>(
        experimental::CompiledGraphBundle{revision,graph,std::move(input)});
    layout->placements.resize(passCapacity,{UINT32_MAX,UINT32_MAX,UINT32_MAX});
    size_t scheduled = 0;
    for (uint32_t batch = 0; batch != graph->batches.size(); ++batch) {
        const auto& symbolic = graph->batches[batch];
        for (const auto compilerPass : symbolic.passes) {
            const auto prepared = graph->structure->passes.at(compilerPass).preparedPassIndex;
            Require(prepared < passCapacity && layout->placements[prepared].preparedPass == UINT32_MAX,
                "Invalid persistent pass placement");
            layout->placements[prepared] = {prepared,batch,symbolic.queue};
            ++scheduled;
        }
    }
    Require(scheduled == graph->structure->passes.size(), "Incomplete persistent pass placement");
    return layout;
}
void RetireOwnership(const std::function<void(std::shared_ptr<const void>)>& sink,
    std::shared_ptr<const void> owner) {
    if (!sink) return;
    try { sink(std::move(owner)); }
    catch (...) {
        // Selection/retirement has already committed. Enqueue failure must not
        // make the caller believe that a valid publication failed to install.
        basic_telemetry::AddCounter("ORG.Persistent.RetirementEnqueueFailures");
    }
}
void ValidateAliasOrder(const experimental::CompiledGraph& graph, const BindingTable& bindings) {
    const auto count = graph.structure->passes.size();
    std::vector<uint8_t> ordered(count * count);
    for (auto [from,to] : graph.schedulingEdges) ordered[from * count + to] = 1;
    std::vector<uint32_t> last(graph.structure->queues.size(),UINT32_MAX);
    for (const auto& batch : graph.batches) {
        auto& previous = last.at(batch.queue);
        for (auto pass : batch.passes) {
            if (previous != UINT32_MAX) ordered[previous * count + pass] = 1;
            previous = pass;
        }
    }
    for (size_t k = 0; k < count; ++k)
        for (size_t i = 0; i < count; ++i)
            if (ordered[i * count + k])
                for (size_t j = 0; j < count; ++j) ordered[i * count + j] |= ordered[k * count + j];
    std::vector<std::vector<uint32_t>> users(graph.structure->resourceIDs.size());
    for (uint32_t pass = 0; pass < count; ++pass) {
        const auto& declaration = graph.structure->passes[pass];
        for (const auto& use : declaration.accesses) users.at(use.resourceIndex).push_back(pass);
        for (const auto& use : declaration.entryStates) users.at(use.resource).push_back(pass);
        for (const auto& use : declaration.exitStates) users.at(use.resource).push_back(pass);
    }
    for (uint32_t a = 0; a < users.size(); ++a) {
        const auto& x = bindings.At(bindings.CurrentSlot(static_cast<uint32_t>(graph.structure->resourceIDs[a]-1))).admission;
        if (!x || !x->aliasSize) continue;
        for (uint32_t b = a + 1; b < users.size(); ++b) {
            const auto& y = bindings.At(bindings.CurrentSlot(static_cast<uint32_t>(graph.structure->resourceIDs[b]-1))).admission;
            if (!y || !y->aliasSize || x->aliasHeapIdentity != y->aliasHeapIdentity
                || x->aliasOffset >= y->aliasOffset + y->aliasSize
                || y->aliasOffset >= x->aliasOffset + x->aliasSize) continue;
            bool before = true, after = true;
            for (auto u : users[a]) for (auto v : users[b]) {
                before &= bool(ordered[u * count + v]);
                after &= bool(ordered[v * count + u]);
            }
            Require(before || after, "Alias placement has overlapping unordered lifetimes");
        }
    }
}
}
NativeBindingContract NativeBindingContract::Capture(const rhi::ResourceDesc& desc) {
    Require(desc.type != rhi::ResourceType::Unknown, "Missing native resource description");
    Require(!desc.castableFormats.size || desc.castableFormats.data, "Invalid castable format list");
    NativeBindingContract contract;
    contract.type = desc.type; contract.heapType = desc.heapType;
    contract.heapFlags = desc.heapFlags; contract.requiredFlags = desc.resourceFlags;
    if (desc.castableFormats.size)
        contract.castableFormats.assign(desc.castableFormats.data,desc.castableFormats.data + desc.castableFormats.size);
    std::sort(contract.castableFormats.begin(),contract.castableFormats.end());
    contract.castableFormats.erase(std::unique(contract.castableFormats.begin(),contract.castableFormats.end()),contract.castableFormats.end());
    if (desc.type == rhi::ResourceType::Buffer || desc.type == rhi::ResourceType::AccelerationStructure) {
        Require(desc.buffer.sizeBytes != 0, "Empty native buffer contract");
        contract.minimumBufferBytes = contract.maximumBufferBytes = desc.buffer.sizeBytes;
    } else {
        Require(desc.type == rhi::ResourceType::Texture1D || desc.type == rhi::ResourceType::Texture2D
            || desc.type == rhi::ResourceType::Texture3D, "Unsupported native resource contract");
        const auto& texture = desc.texture;
        Require(texture.width && texture.height && texture.depthOrLayers && texture.mipLevels && texture.sampleCount
            && texture.format != rhi::Format::Unknown, "Invalid native texture contract");
        contract.format = texture.format; contract.width = texture.width; contract.height = texture.height;
        contract.depthOrLayers = texture.depthOrLayers; contract.mips = texture.mipLevels; contract.samples = texture.sampleCount;
    }
    return contract;
}
bool NativeBindingContract::Matches(const rhi::ResourceDesc& desc, experimental::CompileResourceShape shape) const {
    if (type == rhi::ResourceType::Unknown || desc.type != type || desc.heapType != heapType || desc.heapFlags != heapFlags
        || (static_cast<uint32_t>(desc.resourceFlags) & static_cast<uint32_t>(requiredFlags)) != static_cast<uint32_t>(requiredFlags)
        || (desc.castableFormats.size && !desc.castableFormats.data)) return false;
    for (auto required : castableFormats) {
        bool found = false;
        for (uint32_t i = 0; i != desc.castableFormats.size; ++i) found |= desc.castableFormats.data[i] == required;
        if (!found) return false;
    }
    if (type == rhi::ResourceType::Buffer || type == rhi::ResourceType::AccelerationStructure)
        return minimumBufferBytes && minimumBufferBytes <= maximumBufferBytes
            && desc.buffer.sizeBytes >= minimumBufferBytes && desc.buffer.sizeBytes <= maximumBufferBytes
            && !shape.hasLayout && ((shape.mips == 1 && shape.slices == 1) || (!shape.mips && !shape.slices));
    if (type != rhi::ResourceType::Texture1D && type != rhi::ResourceType::Texture2D && type != rhi::ResourceType::Texture3D) return false;
    const auto& texture = desc.texture;
    return width && height && depthOrLayers && mips && samples && format != rhi::Format::Unknown
        && texture.format == format && texture.width == width && texture.height == height
        && texture.depthOrLayers == depthOrLayers && texture.mipLevels == mips && texture.sampleCount == samples
        && shape.hasLayout && shape.mips == mips && shape.slices == (type == rhi::ResourceType::Texture3D ? 1u : depthOrLayers);
}
const BindingVersion& BindingTable::At(ResourceSlotId slot) const {
    Require(slot.index < m_size, "Stale resource slot");
    const auto& binding = m_pages.at(slot.index / PageSize)->at(slot.index % PageSize);
    Require(binding.active && slot.generation == binding.slotGeneration, "Stale or retired resource slot");
    return binding;
}
ResourceSlotId BindingTable::CurrentSlot(uint32_t index) const {
    Require(index < m_size, "Invalid resource slot index");
    const auto& binding = m_pages[index / PageSize]->at(index % PageSize);
    Require(binding.active, "Retired resource slot");
    return {index,binding.slotGeneration};
}
const BindingVersion* BindingTable::TryAt(uint32_t index) const noexcept {
    if (index >= m_size) return nullptr;
    const auto& binding = m_pages[index / PageSize]->at(index % PageSize);
    return binding.active ? &binding : nullptr;
}
namespace {
// Reserved slots need a unique physical identity so canonicalization never
// merges them, and a placeholder admission entry so compiler resource order
// is preserved. The invalid handle is what admission and recording skip on.
constexpr uint64_t ReservedIdentityBit = uint64_t{1} << 63;
BindingVersion PlaceholderBinding(uint32_t index, experimental::CompileResourceShape shape) {
    BindingVersion binding;
    binding.identity = ReservedIdentityBit | index;
    binding.backingRevision = 1;
    binding.shape = shape;
    binding.owner = std::make_shared<const uint32_t>(index);
    binding.bound = false;
    auto state = std::make_shared<experimental::PreparedBackingState>();
    state->graphResourceID = uint64_t{index} + 1;
    state->shape = shape;
    state->regions = std::make_shared<const std::vector<experimental::PreparedStateRegion>>();
    binding.admission = std::move(state);
    return binding;
}
}
const BindingVersion& SelectedPublication::Resolve(BindingToken token) const {
    Require(token.domain == logical->domain && token.pass.index < logical->passSlots.size(), "Foreign binding token");
    const auto& pass = logical->passSlots[token.pass.index];
    Require(pass.active && pass.generation == token.pass.generation && pass.layoutRevision == token.layoutRevision
        && token.ordinal < pass.bindingSlots.size()
        && pass.bindingSlots[token.ordinal].declarationId == token.declarationId, "Stale or undeclared binding token");
    return bindings.At(pass.bindingSlots[token.ordinal].resource);
}
GraphEditTransaction::GraphEditTransaction(std::shared_ptr<const SelectedPublication> base)
    : m_base(std::move(base)), m_bindings(m_base ? m_base->bindings : BindingTable{}) {
    Require(m_base && m_base->executable && m_base->executable->graph, "Invalid transaction source");
}
uint64_t GraphEditTransaction::BaseRevision() const noexcept { return m_base->revision; }
experimental::GraphCompileStructure& GraphEditTransaction::EditStructure() {
    return EditLogical().declarations;
}
LogicalGraph& GraphEditTransaction::EditLogical(bool structural) {
    m_structuralChanged |= structural;
    if (!m_logical) m_logical = *m_base->logical;
    return *m_logical;
}
ResourceSlotId GraphEditTransaction::AddResource(experimental::CompileResourceShape shape, BindingVersion binding) {
    MutationGuard mutation{m_failed};
    Require(binding.shape == shape && ((shape.mips && shape.slices)
        || (!shape.mips && !shape.slices && !shape.hasLayout)), "Invalid resource shape");
    auto& structure = EditStructure();
    for (uint32_t i = 0; i < m_bindings.m_size; ++i) {
        const auto& old = m_bindings.m_pages[i / BindingTable::PageSize]->at(i % BindingTable::PageSize);
        if (old.active || old.slotGeneration == UINT32_MAX) continue;
        const ResourceSlotId reused{i,old.slotGeneration + 1};
        auto& page = m_changedPages[i / BindingTable::PageSize];
        if (!page) page = std::make_shared<BindingTable::Page>(*m_bindings.m_pages[i / BindingTable::PageSize]);
        auto empty = BindingVersion{}; empty.slotGeneration = reused.generation;
        (*page)[i % BindingTable::PageSize] = std::move(empty);
        m_bindings.m_pages[i / BindingTable::PageSize] = page;
        structure.resourceShapes[i] = shape;
        EditLogical().resourceActive[i] = 1;
        EditLogical().nativeContracts[i].reset();
        if (i < EditLogical().groupBySlot.size()) EditLogical().groupBySlot[i] = LogicalGraph::NoGroup;
        if (binding.recording && binding.recording->description.type != rhi::ResourceType::Unknown)
            EditLogical().nativeContracts[i] = NativeBindingContract::Capture(binding.recording->description);
        ReplaceBinding(reused,std::move(binding));
        return reused;
    }
    ResourceSlotId slot{m_bindings.m_size++, 1};
    structure.resourceIDs.push_back(uint64_t{slot.index} + 1);
    structure.resourceShapes.push_back(shape);
    EditLogical().resourceActive.push_back(1);
    EditLogical().nativeContracts.emplace_back();
    EditLogical().groupBySlot.resize(EditLogical().resourceActive.size(), LogicalGraph::NoGroup);
    if (binding.recording && binding.recording->description.type != rhi::ResourceType::Unknown)
        EditLogical().nativeContracts.back() = NativeBindingContract::Capture(binding.recording->description);
    EditLogical().bindingSubscribers.emplace_back();
    if (slot.index % BindingTable::PageSize == 0)
        m_bindings.m_pages.push_back(std::make_shared<const BindingTable::Page>());
    ReplaceBinding(slot, std::move(binding));
    return slot;
}
ResourceSlotId GraphEditTransaction::ReserveResource(experimental::CompileResourceShape shape,
    std::optional<NativeBindingContract> contract) {
    MutationGuard mutation{m_failed};
    Require(shape.mips && shape.slices, "Reserved slots require a stateful resource contract");
    // ReplaceBinding re-stamps the placeholder with the allocated slot index.
    auto slot = AddResource(shape,PlaceholderBinding(0,shape));
    if (contract) EditLogical().nativeContracts.at(slot.index) = std::move(*contract);
    return slot;
}
void GraphEditTransaction::BindReserved(ResourceSlotId slot, BindingVersion binding) {
    MutationGuard mutation{m_failed};
    Require(!m_bindings.At(slot).bound, "Slot is not reserved");
    Require(binding.bound && binding.recording, "Reserved slots bind exact recording snapshots");
    ReplaceBinding(slot,std::move(binding));
}
void GraphEditTransaction::Unbind(ResourceSlotId slot) {
    MutationGuard mutation{m_failed};
    const auto& current = m_bindings.At(slot);
    Require(current.bound, "Slot is already unbound");
    ReplaceBinding(slot,PlaceholderBinding(slot.index,current.shape));
}
std::vector<ResourceSlotId> GraphEditTransaction::ReserveGroupMembers(ResourceGroupId id, uint32_t count) {
    MutationGuard mutation{m_failed};
    const auto& logical = m_logical ? *m_logical : *m_base->logical;
    Require(id.index < logical.groups.size() && logical.groups[id.index].active
        && id.generation == logical.groups[id.index].generation, "Stale resource group");
    const auto shape = logical.groups[id.index].memberShape;
    auto members = logical.groups[id.index].members;
    std::vector<ResourceSlotId> reserved;
    reserved.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        reserved.push_back(ReserveResource(shape));
        members.push_back(reserved.back());
    }
    ReplaceGroupMembers(id,std::move(members));
    return reserved;
}
void GraphEditTransaction::RemoveResource(ResourceSlotId slot) {
    MutationGuard mutation{m_failed};
    const auto& binding = m_bindings.At(slot);
    const auto& logical = m_logical ? *m_logical : *m_base->logical;
    for (const auto& pass : logical.declarations.passes) {
        for (auto access : pass.accesses) Require(access.resourceIndex != slot.index, "Resource still has pass accesses");
        for (auto* uses : {&pass.entryStates,&pass.exitStates})
            for (const auto& use : *uses) Require(use.resource != slot.index, "Resource still has state declarations");
    }
    for (const auto& group : logical.groups)
        Require(std::find(group.members.begin(),group.members.end(),slot) == group.members.end(), "Resource still belongs to group");
    const auto bucketIndex = static_cast<uint32_t>(binding.identity % BindingTable::IdentityBuckets);
    auto& bucket = m_changedIdentities[bucketIndex];
    if (!bucket) bucket = std::make_shared<BindingTable::IdentityBucket>(*m_bindings.m_identities[bucketIndex]);
    auto& members = bucket->at(binding.identity);
    std::erase(members,slot.index);
    if (members.empty()) bucket->erase(binding.identity);
    m_bindings.m_identities[bucketIndex] = bucket;
    const auto pageIndex = slot.index / BindingTable::PageSize;
    auto& page = m_changedPages[pageIndex];
    if (!page) page = std::make_shared<BindingTable::Page>(*m_bindings.m_pages[pageIndex]);
    auto retired = BindingVersion{}; retired.active = false; retired.slotGeneration = slot.generation;
    (*page)[slot.index % BindingTable::PageSize] = std::move(retired);
    m_bindings.m_pages[pageIndex] = page;
    EditLogical().resourceActive.at(slot.index) = 0;
    EditLogical().nativeContracts.at(slot.index).reset();
}
void GraphEditTransaction::SetNativeBindingContract(ResourceSlotId slot, NativeBindingContract contract) {
    MutationGuard mutation{m_failed};
    const auto& binding = m_bindings.At(slot);
    Require(binding.recording && contract.Matches(binding.recording->description,binding.shape), "Invalid native binding contract");
    EditLogical().nativeContracts.at(slot.index) = std::move(contract);
}
void GraphEditTransaction::ValidatePass(PassId pass) const {
    const auto& logical = m_logical ? *m_logical : *m_base->logical;
    Require(pass.index < logical.passSlots.size() && logical.passSlots[pass.index].active
        && pass.generation == logical.passSlots[pass.index].generation, "Stale pass");
}
PassId GraphEditTransaction::AddPass(experimental::CompilePass pass, std::optional<uint32_t> authoredOrder) {
    MutationGuard mutation{m_failed};
    auto& logical = EditLogical();
    auto& structure = logical.declarations;
    PassId id{static_cast<uint32_t>(structure.passes.size()),1};
    for (uint32_t i = 0; i < logical.passSlots.size(); ++i) {
        auto& slot = logical.passSlots[i];
        if (slot.active || slot.generation == UINT32_MAX) continue;
        slot.active = true; ++slot.generation;
        slot.layoutRevision = 1; slot.bindingSlots.clear();
        slot.epoch = AllEpochs;
        id = {i,slot.generation};
        break;
    }
    Require(authoredOrder || id.index < UINT32_MAX / kAuthoredOrderStride, "Pass authored order exhausted");
    pass.originalOrder = authoredOrder.value_or(id.index * kAuthoredOrderStride);
    pass.preparedPassIndex = id.index;
    if (id.index == structure.passes.size()) {
        structure.passes.push_back(std::move(pass));
        logical.passSlots.push_back({});
    } else structure.passes[id.index] = std::move(pass);
    return id;
}
uint32_t GraphEditTransaction::AuthoredOrder(PassId pass) const {
    ValidatePass(pass);
    const auto& logical = m_logical ? *m_logical : *m_base->logical;
    return logical.declarations.passes[pass.index].originalOrder;
}
void GraphEditTransaction::SetPassEpoch(PassId pass, uint32_t epoch) {
    MutationGuard guard{m_failed};
    ValidatePass(pass);
    const auto& logical = m_logical ? *m_logical : *m_base->logical;
    if (logical.passSlots[pass.index].epoch == epoch) return;
    EditLogical().passSlots[pass.index].epoch = epoch;
}
void GraphEditTransaction::SetEpochOrder(std::vector<uint32_t> order) {
    MutationGuard guard{m_failed};
    const auto& logical = m_logical ? *m_logical : *m_base->logical;
    if (logical.epochOrder == order) return;
    EditLogical().epochOrder = std::move(order);
}
void GraphEditTransaction::SetPassRecordingInterface(PassId pass, std::shared_ptr<const void> recordingInterface) {
    MutationGuard guard{m_failed};
    ValidatePass(pass);
    Require(bool(recordingInterface), "Missing persistent pass recording interface");
    const auto& logical = m_logical ? *m_logical : *m_base->logical;
    if (logical.passSlots[pass.index].recordingInterface == recordingInterface) return;
    EditLogical(false).passSlots[pass.index].recordingInterface = std::move(recordingInterface);
}
void GraphEditTransaction::RemovePass(PassId pass) {
    MutationGuard mutation{m_failed};
    ValidatePass(pass);
    ClearPassBindings(pass);
    EditLogical().passSlots[pass.index].recordingInterface.reset();
    auto& logical = EditLogical();
    logical.passSlots[pass.index].active = false;
    logical.passSlots[pass.index].bindingSlots.clear();
    logical.declarations.passes[pass.index] = {};
    for (auto* edges : {&logical.declarations.explicitEdges,&logical.declarations.placementEdges})
        std::erase_if(*edges,[&](auto edge) { return edge.first == pass.index || edge.second == pass.index; });
    for (auto& group : logical.groups)
        std::erase_if(group.subscribers,[&](const auto& subscription) { return subscription.pass == pass; });
}
void GraphEditTransaction::ReplacePass(PassId pass, experimental::CompilePass declaration) {
    MutationGuard mutation{m_failed};
    auto& structure = EditStructure();
    ValidatePass(pass);
    auto& layout = EditLogical().passSlots[pass.index];
    Require(layout.layoutRevision != UINT64_MAX, "Pass layout revision exhausted");
    ClearPassBindings(pass);
    EditLogical().passSlots[pass.index].recordingInterface.reset();
    ++layout.layoutRevision; layout.bindingSlots.clear();
    declaration.originalOrder = structure.passes[pass.index].originalOrder;
    declaration.preparedPassIndex = pass.index;
    structure.passes[pass.index] = std::move(declaration);
    for (auto& group : EditLogical().groups)
        std::erase_if(group.subscribers,[&](const auto& subscription) { return subscription.pass == pass; });
}
rhi::Resource SelectedPublication::ResolveNative(BindingToken token) const {
    const auto& version = Resolve(token);
    Require(bool(version.recording), "Binding has no native recording snapshot");
    return version.recording->resource;
}
rhi::DescriptorSlot SelectedPublication::ResolveView(BindingToken token, BindlessViewRequest view) const {
    const auto& version = Resolve(token);
    const auto& requirements = logical->passSlots[token.pass.index].bindingSlots[token.ordinal].requiredViews;
    Require(std::any_of(requirements.begin(),requirements.end(),[&](const auto& required) {
        return required.request.kind == view.kind && required.request.variant == view.variant
            && required.request.mip == view.mip && required.request.slice == view.slice;
    }), "Recording requested an undeclared descriptor view");
    Require(version.recording && version.recording->views && version.recording->descriptorOwner,
        "Binding has no owned descriptor snapshot");
    return version.recording->views->Resolve(view);
}
BindingToken GraphEditTransaction::Declare(PassId pass, ResourceSlotId resource,
    experimental::CompileRange range, experimental::CompileResourceState state) {
    MutationGuard mutation{m_failed};
    ValidatePass(pass);
    const auto shape = m_bindings.At(resource).shape;
    Require(range.mips && range.slices && range.mip < shape.mips && range.slice < shape.slices
        && range.mips <= shape.mips-range.mip && range.slices <= shape.slices-range.slice, "Invalid declared binding range");
    auto token = DeclareDependency(pass,resource,state.write);
    EditLogical().declarations.passes[pass.index].entryStates.push_back({resource.index,range,state});
    return token;
}
rhi::DescriptorSlot SelectedPublication::ResolveView(ViewToken token) const {
    const auto& version = Resolve(token.binding);
    const auto& required = logical->passSlots[token.binding.pass.index].bindingSlots[token.binding.ordinal].requiredViews;
    Require(token.ordinal < required.size() && required[token.ordinal].declarationId == token.declarationId,
        "Stale or undeclared descriptor token");
    Require(bool(version.preparedViews), "Binding has no prepared descriptor layout");
    return version.preparedViews->at(token.binding.declarationId).at(token.ordinal);
}
BindingToken GraphEditTransaction::DeclareDependency(PassId pass, ResourceSlotId resource, bool write) {
    MutationGuard mutation{m_failed};
    ValidatePass(pass); m_bindings.At(resource);
    auto& logical = EditLogical();
    auto& layout = logical.passSlots[pass.index];
    const auto ordinal = static_cast<uint32_t>(layout.bindingSlots.size());
    const auto declarationId = TakeIdentity(nextBindingDeclaration);
    layout.bindingSlots.push_back({resource,declarationId});
    logical.bindingSubscribers.at(resource.index).push_back({pass.index,ordinal});
    auto& declaration = logical.declarations.passes[pass.index];
    declaration.accesses.push_back({resource.index,write});
    return BindingToken(logical.domain,layout.layoutRevision,pass,ordinal,declarationId);
}
void GraphEditTransaction::ClearPassBindings(PassId pass) {
    auto& logical = EditLogical();
    auto& entries = logical.passSlots[pass.index].bindingSlots;
    std::vector<ResourceSlotId> affected;
    for (uint32_t ordinal = 0; ordinal < entries.size(); ++ordinal) {
        if (!entries[ordinal].requiredViews.empty()
            && std::find(affected.begin(),affected.end(),entries[ordinal].resource) == affected.end())
            affected.push_back(entries[ordinal].resource);
        auto& subscribers = logical.bindingSubscribers.at(entries[ordinal].resource.index);
        std::erase(subscribers,std::pair{pass.index,ordinal});
    }
    entries.clear();
    // Drop obsolete descriptor layouts in this transaction's fragments. Old
    // selected publications/frames continue retaining their immutable tables.
    for (auto resource : affected) ReplaceBinding(resource,m_bindings.At(resource));
}
ViewToken GraphEditTransaction::DeclareView(BindingToken token, BindlessViewRequest view) {
    MutationGuard mutation{m_failed};
    const auto& logical = m_logical ? *m_logical : *m_base->logical;
    Require(token.domain == logical.domain, "Foreign view binding");
    ValidatePass(token.pass);
    const auto& layout = logical.passSlots[token.pass.index];
    Require(token.layoutRevision == layout.layoutRevision && token.ordinal < layout.bindingSlots.size()
        && token.declarationId == layout.bindingSlots[token.ordinal].declarationId, "Stale view binding");
    const auto& binding = m_bindings.At(layout.bindingSlots[token.ordinal].resource);
    Require(view.mip < binding.shape.mips && view.slice < binding.shape.slices, "Declared view outside resource contract");
    Require(binding.recording && binding.recording->views && binding.recording->descriptorOwner, "View has no owned snapshot");
    Require(binding.recording->views->Resolve(view).heap.valid(), "Declared view has invalid descriptor heap");
    const auto resource = layout.bindingSlots[token.ordinal].resource;
    auto replacement = binding;
    auto& required = EditLogical().passSlots[token.pass.index].bindingSlots[token.ordinal].requiredViews;
    const auto ordinal = static_cast<uint32_t>(required.size());
    const auto id = TakeIdentity(nextBindingDeclaration);
    required.push_back({view,id});
    ReplaceBinding(resource,std::move(replacement));
    return ViewToken(token,ordinal,id);
}
void GraphEditTransaction::DeclarePostcondition(BindingToken token, experimental::CompileRange range,
    experimental::CompileResourceState state) {
    MutationGuard mutation{m_failed};
    const auto& logical = m_logical ? *m_logical : *m_base->logical;
    Require(token.domain == logical.domain, "Foreign postcondition binding");
    ValidatePass(token.pass);
    const auto& layout = logical.passSlots[token.pass.index];
    Require(token.layoutRevision == layout.layoutRevision && token.ordinal < layout.bindingSlots.size()
        && token.declarationId == layout.bindingSlots[token.ordinal].declarationId, "Stale postcondition binding");
    const auto resource = layout.bindingSlots[token.ordinal].resource;
    const auto shape = m_bindings.At(resource).shape;
    Require(range.mips && range.slices && range.mip < shape.mips && range.slice < shape.slices
        && range.mips <= shape.mips-range.mip && range.slices <= shape.slices-range.slice, "Invalid postcondition range");
    auto& declaration = EditLogical().declarations.passes[token.pass.index];
    declaration.exitStates.push_back({resource.index,range,state});
    // A callback postcondition may describe a write even when its entry state
    // was read-only. The entire callback must participate in write hazards.
    declaration.accesses.push_back({resource.index,state.write});
}
void GraphEditTransaction::AddOrdering(PassId before, PassId after) {
    MutationGuard mutation{m_failed};
    auto& structure = EditStructure();
    ValidatePass(before); ValidatePass(after);
    Require(before != after, "Self ordering cycle");
    structure.explicitEdges.emplace_back(before.index, after.index);
}
void GraphEditTransaction::SetQueues(std::vector<experimental::CompileQueue> queues) {
    MutationGuard mutation{m_failed};
    Require(!queues.empty(), "Graph has no queues");
    EditStructure().queues = std::move(queues);
}
void GraphEditTransaction::AddPlacementOrdering(PassId before, PassId after) {
    MutationGuard mutation{m_failed};
    auto& structure = EditStructure();
    ValidatePass(before); ValidatePass(after);
    Require(before != after, "Self placement cycle");
    structure.placementEdges.emplace_back(before.index,after.index);
}
void GraphEditTransaction::ClearPlacementOrderings() {
    MutationGuard mutation{m_failed};
    EditStructure().placementEdges.clear();
}
void GraphEditTransaction::ReplaceResourceContract(ResourceSlotId slot,
    experimental::CompileResourceShape shape, BindingVersion binding) {
    MutationGuard mutation{m_failed};
    m_bindings.At(slot);
    Require(binding.shape == shape, "Replacement contract shape mismatch");
    const auto& previous = m_bindings.At(slot);
    Require(previous.shape == shape || previous.identity != binding.identity, "Shape changed without a new physical identity");
    EditStructure().resourceShapes.at(slot.index) = shape;
    EditLogical().nativeContracts.at(slot.index).reset();
    if (binding.recording && binding.recording->description.type != rhi::ResourceType::Unknown)
        EditLogical().nativeContracts.at(slot.index) = NativeBindingContract::Capture(binding.recording->description);
    ReplaceBinding(slot,std::move(binding));
}
ResourceSlotId GraphEditTransaction::AddSnapshot(std::shared_ptr<const ResourceBindingSnapshot> snapshot,
    const experimental::PreparedBackingState& initial, uint64_t descriptorRevision, uint64_t contentRevision) {
    MutationGuard guard{m_failed};
    return AddResource(initial.shape,BindingVersion::FromSnapshot(std::move(snapshot),initial,descriptorRevision,contentRevision));
}
void GraphEditTransaction::ReplaceSnapshot(ResourceSlotId slot, std::shared_ptr<const ResourceBindingSnapshot> snapshot,
    const experimental::PreparedBackingState& initial, uint64_t descriptorRevision, uint64_t contentRevision) {
    MutationGuard guard{m_failed};
    ReplaceBinding(slot,BindingVersion::FromSnapshot(std::move(snapshot),initial,descriptorRevision,contentRevision));
}
void GraphEditTransaction::ReplaceBinding(ResourceSlotId slot, BindingVersion binding) {
    MutationGuard mutation{m_failed};
    m_bindings.At(slot);
    binding.slotGeneration = slot.generation;
    const auto& structure = m_logical ? m_logical->declarations : m_base->logical->declarations;
    Require(binding.shape == structure.resourceShapes.at(slot.index), "Binding violates shape contract");
    Require(binding.active && binding.identity && binding.backingRevision && binding.owner, "Binding has no exact ownership/version");
    const bool placeholder = !binding.bound;
    if (placeholder) {
        Require(!binding.recording && !binding.preparedViews, "Reserved placeholder carries recording data");
        binding = PlaceholderBinding(slot.index,binding.shape);
        binding.slotGeneration = slot.generation;
    } else Require((binding.identity & ReservedIdentityBit) == 0, "Bound binding uses a reserved identity");
    if (binding.recording) {
        const auto handle = binding.recording->resource.GetHandle();
        Require(handle.valid() && binding.recording->allocationOwner
            && binding.identity == ((uint64_t{handle.generation} << 32) | handle.index)
            && binding.backingRevision == binding.recording->backingGeneration, "Recording snapshot violates binding version");
        Require(!binding.recording->views || binding.recording->views->views.empty() || binding.recording->descriptorOwner,
            "Recording snapshot has unowned descriptor views");
        if (binding.recording->views) {
            for (const auto& view : binding.recording->views->views)
                ValidateNativeView(binding.recording->description,view.kind);
            if (binding.recording->views->description.type != rhi::ResourceType::Unknown) {
                const auto views = NativeBindingContract::Capture(binding.recording->views->description);
                const auto backing = NativeBindingContract::Capture(binding.recording->description);
                Require(views.Matches(binding.recording->description,binding.shape)
                    && backing.Matches(binding.recording->views->description,binding.shape),
                    "Published views describe a different native backing");
            }
        }
        if (binding.admission) {
            Require(binding.recording->aliasHeap.get() == binding.admission->aliasHeapIdentity
                && binding.recording->aliasHeap.get() == binding.admission->aliasHeap.get()
                && !binding.recording->aliasHeap.owner_before(binding.admission->aliasHeap)
                && !binding.admission->aliasHeap.owner_before(binding.recording->aliasHeap)
                && binding.recording->aliasPoolID == binding.admission->aliasPoolID
                && binding.recording->aliasOffset == binding.admission->aliasOffset
                && binding.recording->aliasSize == binding.admission->aliasSize, "Recording snapshot violates alias contract");
        }
    }
    const auto& previousBinding = m_bindings.At(slot);
    if (previousBinding.identity == binding.identity && previousBinding.recording && binding.recording
        && previousBinding.recording->description.type != rhi::ResourceType::Unknown) {
        const auto before = NativeBindingContract::Capture(previousBinding.recording->description);
        const auto after = NativeBindingContract::Capture(binding.recording->description);
        Require(before.Matches(binding.recording->description,binding.shape)
            && after.Matches(previousBinding.recording->description,previousBinding.shape),
            "Native description changed without a new physical identity");
    }
    const auto& logical = m_logical ? *m_logical : *m_base->logical;
    if (!placeholder) {
        auto checkClass = [&](const ResourceGroup& group) {
            if (!group.active || group.memberType == rhi::ResourceType::Unknown) return;
            Require(binding.recording && binding.recording->description.type == group.memberType,
                "Replacement violates group resource class");
        };
        const auto groupIndex = slot.index < logical.groupBySlot.size() ? logical.groupBySlot[slot.index] : LogicalGraph::MultipleGroups;
        if (groupIndex == LogicalGraph::MultipleGroups) {
            for (const auto& group : logical.groups)
                if (std::ranges::find(group.members,slot) != group.members.end()) checkClass(group);
        } else if (groupIndex != LogicalGraph::NoGroup) checkClass(logical.groups.at(groupIndex));
    }
    if (const auto& contract = logical.nativeContracts.at(slot.index); contract && !placeholder) {
        Require(binding.recording && contract->Matches(binding.recording->description,binding.shape),
            "Replacement violates native binding contract");
        Require(!binding.admission || binding.admission->heapType == contract->heapType, "Admission heap type violates native binding contract");
    }
    std::shared_ptr<BindingVersion::DescriptorTable> descriptors;
    for (auto [pass,ordinal] : logical.bindingSubscribers.at(slot.index)) {
        const auto& entry = logical.passSlots.at(pass).bindingSlots.at(ordinal);
        if (entry.requiredViews.empty() || placeholder) continue;
        Require(binding.recording && binding.recording->views && binding.recording->descriptorOwner,
            "Replacement lost a required descriptor snapshot");
        if (!descriptors) descriptors = std::make_shared<BindingVersion::DescriptorTable>();
        auto& prepared = (*descriptors)[entry.declarationId];
        for (const auto& view : entry.requiredViews) {
            Require(view.request.mip < binding.shape.mips && view.request.slice < binding.shape.slices,
                "Required view outside replacement contract");
            const auto descriptor = binding.recording->views->Resolve(view.request);
            Require(descriptor.heap.valid(), "Required view has invalid descriptor heap");
            prepared.push_back(descriptor);
        }
    }
    binding.preparedViews = std::move(descriptors);
    if (binding.admission) {
        Require(!binding.admission->authoritativeIncoming, "Authoritative incoming states belong to frames, not publications");
        Require(binding.admission->shape == binding.shape, "Admission shape violates binding contract");
        Require(binding.admission->graphResourceID == structure.resourceIDs.at(slot.index), "Admission slot mismatch");
        const auto handle = binding.admission->resource;
        Require(placeholder || binding.identity == ((uint64_t{handle.generation} << 32) | handle.index), "Admission identity mismatch");
        if (binding.admission->aliasSize) {
            Require(binding.admission->aliasHeap && binding.admission->aliasHeapIdentity == binding.admission->aliasHeap.get(),
                "Alias placement has no exact heap owner");
            Require(binding.admission->aliasOffset <= UINT64_MAX - binding.admission->aliasSize, "Alias range overflow");
        }
    }
    const auto& previous = m_bindings.At(slot).admission;
    if (!m_structuralChanged && !previous && binding.admission) {
        // Introducing placement metadata is not a compatible backing update:
        // the selected executable has never checked these physical overlaps.
        Require(!binding.admission->aliasSize && !binding.admission->aliasHeapIdentity
            && !binding.admission->aliasPoolID && !binding.admission->aliasOffset,
            "Binding replacement requires a structural alias edit");
    }
    if (!m_structuralChanged && previous) {
        Require(bool(binding.admission), "Binding lost its admission contract");
        const auto& next = *binding.admission;
        Require(previous->aliasHeapIdentity == next.aliasHeapIdentity && previous->aliasPoolID == next.aliasPoolID
            && previous->aliasOffset == next.aliasOffset && previous->aliasSize == next.aliasSize,
            "Binding replacement requires a structural alias edit");
    }
    const auto newBucket = static_cast<uint32_t>(binding.identity % BindingTable::IdentityBuckets);
    auto editBucket = [&](uint32_t index) -> BindingTable::IdentityBucket& {
        auto& changed = m_changedIdentities[index];
        if (!changed) {
            const auto& original = m_bindings.m_identities[index];
            changed = original ? std::make_shared<BindingTable::IdentityBucket>(*original)
                : std::make_shared<BindingTable::IdentityBucket>();
            m_bindings.m_identities[index] = changed;
        }
        return *changed;
    };
    const auto oldIdentity = m_bindings.At(slot).identity;
    if (oldIdentity) {
        auto& bucket = editBucket(static_cast<uint32_t>(oldIdentity % BindingTable::IdentityBuckets));
        auto& members = bucket.at(oldIdentity);
        std::erase(members,slot.index);
        if (members.empty()) bucket.erase(oldIdentity);
    }
    auto& members = editBucket(newBucket)[binding.identity];
    members.push_back(slot.index);
    std::sort(members.begin(),members.end());
    const auto pageIndex = slot.index / BindingTable::PageSize;
    auto& page = m_changedPages[pageIndex];
    if (!page) page = std::make_shared<BindingTable::Page>(*m_bindings.m_pages.at(pageIndex));
    page->at(slot.index % BindingTable::PageSize) = std::move(binding);
    m_bindings.m_pages[pageIndex] = page;
}
ResourceGroupId GraphEditTransaction::AddGroup(experimental::CompileResourceShape shape,
    std::vector<ResourceSlotId> members) {
    MutationGuard mutation{m_failed};
    Require(shape.mips && shape.slices, "Group requires a stateful resource contract");
    auto& logical = EditLogical();
    ResourceGroupId id{static_cast<uint32_t>(logical.groups.size()),1};
    for (uint32_t i = 0; i < logical.groups.size(); ++i) {
        auto& retired = logical.groups[i];
        if (retired.active || retired.generation == UINT32_MAX) continue;
        id = {i,retired.generation + 1};
        retired = {shape,0,{},{},id.generation,true};
        ReplaceGroupMembers(id,std::move(members));
        return id;
    }
    logical.groups.push_back({shape,0,{},{}});
    ReplaceGroupMembers(id,std::move(members));
    return id;
}
void GraphEditTransaction::ReplaceGroupMembers(ResourceGroupId id, std::vector<ResourceSlotId> members) {
    MutationGuard mutation{m_failed};
    const auto& logical = m_logical ? *m_logical : *m_base->logical;
    Require(id.index < logical.groups.size() && logical.groups[id.index].active
        && id.generation == logical.groups[id.index].generation, "Stale resource group");
    const auto& group = logical.groups[id.index];
    std::sort(members.begin(),members.end(),[](auto a, auto b) { return a.index < b.index; });
    for (size_t i = 0; i < members.size(); ++i) {
        Require(m_bindings.At(members[i]).shape == group.memberShape, "Group member violates shape contract");
        const auto& binding = m_bindings.At(members[i]);
        Require(group.memberType == rhi::ResourceType::Unknown
            || (binding.recording && binding.recording->description.type == group.memberType),
            "Group member violates resource class");
        Require(!i || members[i-1].index != members[i].index, "Duplicate group member");
    }
    if (group.members != members) {
        Require(group.membershipRevision != UINT64_MAX, "Group membership revision exhausted");
        auto& edited = EditLogical();
        auto& changed = edited.groups[id.index];
        edited.groupBySlot.resize(edited.resourceActive.size(), LogicalGraph::NoGroup);
        const auto byIndex = [](ResourceSlotId a, ResourceSlotId b) { return a.index < b.index; };
        std::vector<ResourceSlotId> removed, added;
        std::set_difference(changed.members.begin(),changed.members.end(),members.begin(),members.end(),std::back_inserter(removed),byIndex);
        std::set_difference(members.begin(),members.end(),changed.members.begin(),changed.members.end(),std::back_inserter(added),byIndex);
        changed.members = std::move(members);
        ++changed.membershipRevision;
        for (const auto slot : removed) UnindexGroupMember(edited,slot,id.index);
        for (const auto slot : added) {
            auto& entry = edited.groupBySlot.at(slot.index);
            entry = entry == LogicalGraph::NoGroup ? id.index : entry == id.index ? entry : LogicalGraph::MultipleGroups;
        }
    }
}
void GraphEditTransaction::UnindexGroupMember(LogicalGraph& logical, ResourceSlotId slot, uint32_t groupIndex) {
    if (slot.index >= logical.groupBySlot.size()) return;
    auto& entry = logical.groupBySlot[slot.index];
    if (entry == groupIndex) { entry = LogicalGraph::NoGroup; return; }
    if (entry != LogicalGraph::MultipleGroups) return;
    // Rare: recompute from the remaining groups.
    entry = LogicalGraph::NoGroup;
    for (uint32_t i = 0; i < logical.groups.size(); ++i) {
        const auto& group = logical.groups[i];
        if (i == groupIndex || !group.active || std::ranges::find(group.members,slot) == group.members.end()) continue;
        entry = entry == LogicalGraph::NoGroup ? i : LogicalGraph::MultipleGroups;
    }
}
void GraphEditTransaction::SetGroupResourceClass(ResourceGroupId id, rhi::ResourceType type) {
    MutationGuard mutation{m_failed};
    const auto& logical = m_logical ? *m_logical : *m_base->logical;
    Require(id.index < logical.groups.size() && logical.groups[id.index].active
        && id.generation == logical.groups[id.index].generation, "Stale resource group");
    const bool texture = type == rhi::ResourceType::Texture1D || type == rhi::ResourceType::Texture2D
        || type == rhi::ResourceType::Texture3D;
    Require(type == rhi::ResourceType::Unknown || texture || type == rhi::ResourceType::Buffer
        || type == rhi::ResourceType::AccelerationStructure, "Unsupported group resource class");
    const auto& group = logical.groups[id.index];
    Require(type == rhi::ResourceType::Unknown || texture == group.memberShape.hasLayout,
        "Group resource class disagrees with shape");
    for (auto slot : group.members) {
        const auto& binding = m_bindings.At(slot);
        Require(type == rhi::ResourceType::Unknown || (binding.recording && binding.recording->description.type == type),
            "Existing group member violates resource class");
    }
    if (group.memberType != type) EditLogical().groups[id.index].memberType = type;
}
void GraphEditTransaction::DeclareGroupAccess(PassId pass, ResourceGroupId id,
    experimental::CompileResourceState state, uint32_t phase, std::optional<experimental::CompileRange> range) {
    MutationGuard mutation{m_failed};
    auto& logical = EditLogical();
    Require(id.index < logical.groups.size() && logical.groups[id.index].active
        && id.generation == logical.groups[id.index].generation, "Stale resource group");
    ValidatePass(pass);
    if (range) {
        const auto shape = logical.groups[id.index].memberShape;
        Require(range->mips && range->slices && range->mip < shape.mips && range->slice < shape.slices
            && range->mips <= shape.mips-range->mip && range->slices <= shape.slices-range->slice,
            "Invalid group access range");
    }
    auto& subscribers = logical.groups[id.index].subscribers;
    const auto shape = logical.groups[id.index].memberShape;
    const auto requested = range.value_or(experimental::CompileRange{0,shape.mips,0,shape.slices});
    for (const auto& value : subscribers) {
        if (value.pass != pass) continue;
        const auto previous = value.range.value_or(experimental::CompileRange{0,shape.mips,0,shape.slices});
        Require(uint64_t{requested.mip} + requested.mips <= previous.mip
            || uint64_t{previous.mip} + previous.mips <= requested.mip
            || uint64_t{requested.slice} + requested.slices <= previous.slice
            || uint64_t{previous.slice} + previous.slices <= requested.slice,
            "Overlapping group subscriptions from the same pass");
    }
    subscribers.push_back({pass,state,phase,range});
}
void GraphEditTransaction::RemoveGroupAccess(PassId pass, ResourceGroupId id) {
    MutationGuard mutation{m_failed};
    const auto& logical = m_logical ? *m_logical : *m_base->logical;
    Require(id.index < logical.groups.size() && logical.groups[id.index].active
        && id.generation == logical.groups[id.index].generation, "Stale resource group");
    ValidatePass(pass);
    const auto& subscribers = logical.groups[id.index].subscribers;
    if (std::none_of(subscribers.begin(),subscribers.end(),[&](const auto& value) { return value.pass == pass; })) return;
    std::erase_if(EditLogical().groups[id.index].subscribers,[&](const auto& value) { return value.pass == pass; });
}
void GraphEditTransaction::RemoveGroup(ResourceGroupId id) {
    MutationGuard mutation{m_failed};
    auto& logical = EditLogical();
    Require(id.index < logical.groups.size() && logical.groups[id.index].active
        && id.generation == logical.groups[id.index].generation, "Stale resource group");
    auto& group = logical.groups[id.index];
    group.active = false;
    for (const auto slot : group.members) UnindexGroupMember(logical,slot,id.index);
    group.members.clear();
    group.subscribers.clear();
}
std::shared_ptr<const SelectedPublication> GraphEditTransaction::Build(
    experimental::CompileWorkspace& workspace, const std::atomic_bool& cancelled, bool validateAliasOrder) {
    Require(!m_failed, "Publication transaction contains a failed edit");
    Require(m_base->revision != UINT64_MAX, "Publication revision exhausted");
    if (cancelled.load(std::memory_order_relaxed)) return {};
    // Inspect final equivalence classes, not transient intermediate replacements.
    // Rotating every slot of a shared backing atomically preserves the executable.
    if (!m_structuralChanged) for (const auto& [index,bucket] : m_changedIdentities) {
        for (const auto& [identity,members] : *bucket) for (const auto slot : members) {
            const auto& old = m_base->bindings.At(m_base->bindings.CurrentSlot(slot));
            const auto& previous = m_base->bindings.m_identities[old.identity % BindingTable::IdentityBuckets]->at(old.identity);
            if (previous != members) {
                basic_telemetry::AddCounter("ORG.Persistent.EquivalenceClassRebuilds");
                static std::atomic<uint32_t> reported{0};
                if (reported.fetch_add(1) < 8)
                    spdlog::info("Persistent binding edit changed an equivalence class: slot={} identity={:#x} oldIdentity={:#x} previousMembers={} members={}",
                        slot, identity, old.identity, previous.size(), members.size());
                EditLogical(); break;
            }
        }
    }
    // Validate only changed identity buckets on a binding-only publication.
    // Shared slots may have distinct descriptor owners and views, but name one
    // exact backing/content/placement version for state and hazard planning.
    for (const auto& [index,bucket] : m_changedIdentities) for (const auto& [identity,members] : *bucket) {
        for (size_t i = 1; i < members.size(); ++i)
            ValidateSharedBacking(m_bindings.At(m_bindings.CurrentSlot(members[0])),
                m_bindings.At(m_bindings.CurrentSlot(members[i])));
    }
    auto executable = m_base->executable;
    if (m_structuralChanged) {
        Require(executable->structuralRevision != UINT64_MAX, "Executable revision exhausted");
        BT_ZONE_SCOPE("ORG.Persistent.BuildExecutable");
        experimental::GraphCompileInput input;
        input.structure = m_logical->declarations;
        // Epochs execute in their declared order, so the whole-frame schedule (and the
        // alias placement planned over it) orders every pass of an epoch before every
        // pass of the next one that has passes. Untagged passes are left free.
        std::vector<uint32_t> epochSequence;
        {
            std::vector<uint32_t> present;
            for (const auto& slot : m_logical->passSlots)
                if (slot.active && slot.epoch != AllEpochs && std::ranges::find(present,slot.epoch) == present.end())
                    present.push_back(slot.epoch);
            for (const auto epoch : m_logical->epochOrder)
                if (std::ranges::find(present,epoch) != present.end() && std::ranges::find(epochSequence,epoch) == epochSequence.end())
                    epochSequence.push_back(epoch);
            std::ranges::sort(present);
            for (const auto epoch : present)
                if (std::ranges::find(epochSequence,epoch) == epochSequence.end()) epochSequence.push_back(epoch);
            // Hazards are derived in authored order, so with epochs that order has to be the
            // frame's: epoch rank first (untagged passes ahead of every epoch), authored order
            // within an epoch. Registration order across epochs means nothing - a resource the
            // colour pass writes and the next frame's depth pass reads would otherwise be
            // ordered colour-before-depth and contradict the epoch edges below.
            if (!epochSequence.empty()) {
                auto rank = [&](uint32_t epoch) -> uint32_t {
                    if (epoch == AllEpochs) return 0;
                    return static_cast<uint32_t>(std::ranges::find(epochSequence,epoch) - epochSequence.begin()) + 1;
                };
                std::vector<uint32_t> order;
                for (uint32_t i = 0; i < m_logical->passSlots.size(); ++i)
                    if (m_logical->passSlots[i].active) order.push_back(i);
                std::ranges::stable_sort(order,[&](uint32_t a, uint32_t b) {
                    const auto ra = rank(m_logical->passSlots[a].epoch), rb = rank(m_logical->passSlots[b].epoch);
                    return ra != rb ? ra < rb : input.structure.passes[a].originalOrder < input.structure.passes[b].originalOrder;
                });
                for (uint32_t position = 0; position < order.size(); ++position)
                    input.structure.passes[order[position]].originalOrder = position * kAuthoredOrderStride;
            }
            for (size_t k = 0; k + 1 < epochSequence.size(); ++k)
                for (uint32_t a = 0; a < m_logical->passSlots.size(); ++a) {
                    if (!m_logical->passSlots[a].active || m_logical->passSlots[a].epoch != epochSequence[k]) continue;
                    for (uint32_t b = 0; b < m_logical->passSlots.size(); ++b)
                        if (m_logical->passSlots[b].active && m_logical->passSlots[b].epoch == epochSequence[k + 1])
                            input.structure.explicitEdges.emplace_back(a,b);
                }
        }
        std::vector<uint32_t> canonical(input.structure.resourceIDs.size(),UINT32_MAX);
        for (uint32_t i = 0; i < canonical.size(); ++i) {
            if (!m_logical->resourceActive.at(i)) continue;
            const auto& binding = m_bindings.At(m_bindings.CurrentSlot(i));
            canonical[i] = m_bindings.m_identities[binding.identity % BindingTable::IdentityBuckets]->at(binding.identity).front();
        }
        {
            BT_ZONE_SCOPE("ORG.Persistent.PrepareGroups");
            for (const auto& group : m_logical->groups) {
                for (const auto member : group.members) {
                    Require(member.index < m_bindings.Size(), "Stale group member");
                    Require(m_bindings.At(member).shape == group.memberShape, "Group member violates shape contract");
                }
                for (const auto& subscription : group.subscribers) {
                    auto& pass = input.structure.passes.at(subscription.pass.index);
                    for (const auto member : group.members) {
                        pass.accesses.push_back({member.index,subscription.state.write});
                        pass.entryStates.push_back({member.index,
                            subscription.range.value_or(experimental::CompileRange{0,group.memberShape.mips,0,group.memberShape.slices}),subscription.state});
                    }
                }
            }
            // Phase contracts order overlapping groups as well as each group's
            // own producer/consumer phases. Equal-phase writes are ambiguous.
            auto explicitlyOrdered = [&](uint32_t from, uint32_t to) {
                std::vector<uint8_t> visited(input.structure.passes.size());
                std::vector<uint32_t> pending{from};
                while (!pending.empty()) {
                    const auto pass = pending.back(); pending.pop_back();
                    Require(pass < visited.size(), "Invalid authored group ordering");
                    if (pass == to) return true;
                    if (visited[pass]) continue;
                    visited[pass] = 1;
                    for (const auto* edges : {&m_logical->declarations.explicitEdges,&m_logical->declarations.placementEdges})
                        for (auto [a,b] : *edges) if (a == pass) pending.push_back(b);
                }
                return false;
            };
            for (size_t a = 0; a < m_logical->groups.size(); ++a)
                for (size_t b = a; b < m_logical->groups.size(); ++b) {
                    const auto& x = m_logical->groups[a];
                    const auto& y = m_logical->groups[b];
                    bool overlap = false;
                    for (auto member : x.members)
                        if (std::ranges::any_of(y.members,[&](auto other) { return canonical.at(member.index) == canonical.at(other.index); })) {
                            overlap = true; break;
                        }
                    if (!overlap) continue;
                    for (const auto& u : x.subscribers) for (const auto& v : y.subscribers) {
                        if (u.pass == v.pass || (!u.state.write && !v.state.write)) continue;
                        const auto ur = u.range.value_or(experimental::CompileRange{0,x.memberShape.mips,0,x.memberShape.slices});
                        const auto vr = v.range.value_or(experimental::CompileRange{0,y.memberShape.mips,0,y.memberShape.slices});
                        if (uint64_t{ur.mip} + ur.mips <= vr.mip || uint64_t{vr.mip} + vr.mips <= ur.mip
                            || uint64_t{ur.slice} + ur.slices <= vr.slice || uint64_t{vr.slice} + vr.slices <= ur.slice) continue;
                        if (u.phase == v.phase) {
                            Require(explicitlyOrdered(u.pass.index,v.pass.index) || explicitlyOrdered(v.pass.index,u.pass.index),
                                "Overlapping writable group phases require explicit ordering");
                            continue;
                        }
                        input.structure.explicitEdges.emplace_back(
                            u.phase < v.phase ? u.pass.index : v.pass.index,
                            u.phase < v.phase ? v.pass.index : u.pass.index);
                    }
                }
        }
        CanonicalizeUses(input.structure,canonical);
        if (!m_logical->groups.empty()) {
            // Group phases are authoritative even if subscribers registered in
            // another order. Preserve direct-access dependencies, then derive
            // lowered access hazards in a legal authored order without changing
            // stable pass indices or frame recording associations.
            auto directStructure = m_logical->declarations;
            CanonicalizeUses(directStructure,canonical);
            const auto direct = workspace.AnalyzeDependencies(directStructure,cancelled);
            if (!direct) return {};
            const auto count = input.structure.passes.size();
            std::vector<std::vector<uint32_t>> successors(count);
            std::vector<uint32_t> incoming(count);
            auto append = [&](uint32_t from, uint32_t to) {
                Require(from < count && to < count && from != to, "Invalid group phase ordering");
                successors[from].push_back(to); ++incoming[to];
            };
            for (auto [from,to] : *direct) append(from,to);
            for (auto [from,to] : input.structure.explicitEdges) append(from,to);
            for (auto [from,to] : input.structure.placementEdges) append(from,to);
            std::vector<uint32_t> order;
            order.reserve(count);
            std::vector<uint8_t> selected(count);
            while (order.size() < count) {
                uint32_t next = UINT32_MAX;
                for (uint32_t i = 0; i < count; ++i)
                    if (!selected[i] && !incoming[i] && (next == UINT32_MAX
                        || input.structure.passes[i].originalOrder < input.structure.passes[next].originalOrder)) next = i;
                Require(next != UINT32_MAX, "Cyclic group phase ordering");
                selected[next] = 1; order.push_back(next);
                for (auto to : successors[next]) --incoming[to];
            }
            input.analyzedDependencies = workspace.AnalyzeDependencies(input.structure,cancelled,order);
            if (!input.analyzedDependencies) return {};
        }
        std::vector<uint32_t> denseEpochs;  // per lowered pass
        {
            BT_ZONE_SCOPE("ORG.Persistent.LowerPassSlots");
            std::vector<uint32_t> dense(input.structure.passes.size(),UINT32_MAX);
            std::vector<experimental::CompilePass> active;
            for (uint32_t i = 0; i < dense.size(); ++i) {
                if (!m_logical->passSlots.at(i).active) continue;
                dense[i] = static_cast<uint32_t>(active.size());
                active.push_back(std::move(input.structure.passes[i]));
                denseEpochs.push_back(m_logical->passSlots[i].epoch);
            }
            auto lowerEdges = [&](experimental::DependencyEdges& edges) {
                for (auto& [from,to] : edges) {
                    Require(from < dense.size() && to < dense.size()
                        && dense[from] != UINT32_MAX && dense[to] != UINT32_MAX, "Ordering references retired pass");
                    from = dense[from]; to = dense[to];
                }
            };
            lowerEdges(input.structure.explicitEdges);
            lowerEdges(input.structure.placementEdges);
            if (input.analyzedDependencies) {
                auto edges = std::make_shared<experimental::DependencyEdges>(*input.analyzedDependencies);
                lowerEdges(*edges);
                input.analyzedDependencies = std::move(edges);
            }
            input.structure.passes = std::move(active);
            input.structure.diagnosticPassNames.clear();
        }
        {
            BT_ZONE_SCOPE("ORG.Persistent.LowerResourceSlots");
            std::vector<uint32_t> dense(input.structure.resourceIDs.size(),UINT32_MAX);
            std::vector<uint64_t> identities;
            std::vector<experimental::CompileResourceShape> shapes;
            for (uint32_t i = 0; i < dense.size(); ++i) {
                if (!m_logical->resourceActive.at(i) || canonical[i] != i) continue;
                dense[i] = static_cast<uint32_t>(identities.size());
                identities.push_back(input.structure.resourceIDs[i]);
                shapes.push_back(input.structure.resourceShapes[i]);
            }
            auto lower = [&](uint32_t& resource) {
                Require(resource < dense.size() && dense[resource] != UINT32_MAX, "Declaration references retired resource");
                resource = dense[resource];
            };
            for (auto& pass : input.structure.passes) {
                for (auto& access : pass.accesses) lower(access.resourceIndex);
                for (auto* uses : {&pass.entryStates,&pass.exitStates}) for (auto& use : *uses) lower(use.resource);
            }
            input.structure.resourceIDs = std::move(identities);
            input.structure.resourceShapes = std::move(shapes);
            input.structure.resourceKeys.clear();
            input.structure.diagnosticResourceNames.clear();
        }
        input.structure.generation = executable->structuralRevision + 1;
        experimental::NormalizeCompileInput(input);
        auto compileInput = std::make_shared<const experimental::GraphCompileInput>(std::move(input));
        auto graph = experimental::CompileGraph(compileInput, workspace, cancelled);
        if (!graph) return {};
        Require(graph->states.complete, "Incomplete persistent state plan");
        if (validateAliasOrder && std::ranges::any_of(m_bindings.m_pages, [](const auto& page) {
            return std::ranges::any_of(*page, [](const auto& binding) { return binding.admission && binding.admission->aliasSize; });
        })) ValidateAliasOrder(*graph, m_bindings);
        const auto revision = executable->structuralRevision + 1;
        auto generationOf = [&](std::shared_ptr<const experimental::CompiledGraph> compiled,
                                std::shared_ptr<const experimental::GraphCompileInput> compiledInput) {
            std::vector<ResourceSlotId> resourceSlots;
            resourceSlots.reserve(compiled->structure->resourceIDs.size());
            for (auto identity : compiled->structure->resourceIDs) {
                Require(identity && identity <= m_bindings.Size(), "Invalid compiler resource slot identity");
                resourceSlots.push_back(m_bindings.CurrentSlot(static_cast<uint32_t>(identity-1)));
            }
            auto layout = BuildPersistentExecutionLayout(revision,compiled,compiledInput,m_logical->passSlots.size());
            std::vector<uint32_t> resourceIndexBySlot(m_bindings.Size(),UINT32_MAX);
            for (uint32_t i = 0; i < resourceSlots.size(); ++i) resourceIndexBySlot[resourceSlots[i].index] = i;
            for (uint32_t slot = 0; slot < canonical.size(); ++slot)
                if (canonical[slot] != UINT32_MAX) resourceIndexBySlot[slot] = resourceIndexBySlot[canonical[slot]];
            std::vector<std::vector<uint32_t>> incomingBatchesByResource(resourceSlots.size());
            for (uint32_t batch = 0; batch < compiled->aliasFirstResourcesByBatch.size(); ++batch)
                for (const auto resource : compiled->aliasFirstResourcesByBatch[batch])
                    incomingBatchesByResource.at(resource).push_back(batch);
            std::vector<uint32_t> incomingStateBatchByResource(resourceSlots.size(),UINT32_MAX);
            for (const auto& step : compiled->states.steps)
                if (step.previousBatch == UINT32_MAX && incomingStateBatchByResource.at(step.resource) == UINT32_MAX)
                    incomingStateBatchByResource[step.resource] = step.batch;
            return ExecutableGeneration{revision, std::move(compiled),std::move(resourceSlots),
                std::move(compiledInput),std::move(layout),std::move(resourceIndexBySlot),std::move(incomingBatchesByResource),
                std::move(incomingStateBatchByResource)};
        };
        auto whole = generationOf(graph,compileInput);
        // Per epoch: the same compile input restricted to the epoch's passes (and the
        // untagged ones), over the same resources, bindings and alias placements. Its
        // first use of each resource is an admission boundary, so an epoch's entry
        // state comes from the ledger - whatever ran before it, or did not.
        for (const auto epoch : epochSequence) {
            BT_ZONE_SCOPE("ORG.Persistent.BuildEpochExecutable");
            experimental::GraphCompileInput split = *compileInput;
            std::vector<uint32_t> remap(split.structure.passes.size(),UINT32_MAX);
            std::vector<experimental::CompilePass> passes;
            for (uint32_t i = 0; i < remap.size(); ++i) {
                if (denseEpochs[i] != epoch && denseEpochs[i] != AllEpochs) continue;
                remap[i] = static_cast<uint32_t>(passes.size());
                passes.push_back(split.structure.passes[i]);
            }
            if (passes.empty()) continue;
            auto keep = [&](const experimental::DependencyEdges& edges) {
                experimental::DependencyEdges kept;
                for (auto [from,to] : edges)
                    if (remap[from] != UINT32_MAX && remap[to] != UINT32_MAX) kept.emplace_back(remap[from],remap[to]);
                return kept;
            };
            split.structure.passes = std::move(passes);
            split.structure.explicitEdges = keep(split.structure.explicitEdges);
            split.structure.placementEdges = keep(split.structure.placementEdges);
            if (split.analyzedDependencies)
                split.analyzedDependencies = std::make_shared<const experimental::DependencyEdges>(keep(*split.analyzedDependencies));
            split.expectedEdges.reset();
            split.expectedSchedulingEdges.reset();
            auto splitInput = std::make_shared<const experimental::GraphCompileInput>(std::move(split));
            auto splitGraph = experimental::CompileGraph(splitInput, workspace, cancelled);
            if (!splitGraph) return {};
            Require(splitGraph->states.complete, "Incomplete persistent epoch state plan");
            if (validateAliasOrder && std::ranges::any_of(m_bindings.m_pages, [](const auto& page) {
                return std::ranges::any_of(*page, [](const auto& binding) { return binding.admission && binding.admission->aliasSize; });
            })) ValidateAliasOrder(*splitGraph, m_bindings);
            whole.epochs.emplace_back(epoch, std::make_shared<const ExecutableGeneration>(generationOf(std::move(splitGraph),std::move(splitInput))));
        }
        executable = std::make_shared<const ExecutableGeneration>(std::move(whole));
    }
    // Freeze the pages this transaction edited by handing them to the
    // publication as-is. The transaction forgets them as mutable, so a later
    // edit to the same page clones it again from the (now frozen) table:
    // copy-on-write, one copy per page per edit, none at readiness.
    auto bindings = m_bindings;
    m_changedPages.clear();
    m_changedIdentities.clear();
    if (cancelled.load(std::memory_order_relaxed)) return {};
    return std::shared_ptr<const SelectedPublication>(new SelectedPublication(
        m_base->revision + 1, std::move(executable), std::move(bindings),
        m_logical ? std::make_shared<const LogicalGraph>(*m_logical) : m_base->logical, m_base));
}
GraphProgram::GraphProgram(std::function<void(std::shared_ptr<const void>)> retireOwnership)
    : m_retireOwnership(std::move(retireOwnership)) {
    experimental::GraphCompileInput input;
    experimental::CompileWorkspace workspace;
    std::atomic_bool cancelled{false};
    auto compileInput = std::make_shared<const experimental::GraphCompileInput>(std::move(input));
    auto graph = experimental::CompileGraph(compileInput, workspace, cancelled);
    auto logical = std::make_shared<LogicalGraph>();
    logical->domain = TakeIdentity(nextProgramDomain);
    auto layout = BuildPersistentExecutionLayout(0,graph,compileInput,0);
    m_selected = std::shared_ptr<const SelectedPublication>(new SelectedPublication(0,
        std::make_shared<const ExecutableGeneration>(ExecutableGeneration{0, std::move(graph),{},std::move(compileInput),std::move(layout)}), {},
        std::move(logical)));
}
std::shared_ptr<const SelectedPublication> SelectedPublication::ForEpoch(uint32_t epoch) const {
    if (!HasEpochs()) return {};
    std::lock_guard lock(m_epochMutex);
    for (const auto& [cached,view] : m_epochViews) if (cached == epoch) return view;
    std::shared_ptr<const SelectedPublication> view;
    for (const auto& [id,split] : executable->epochs)
        if (id == epoch) view = std::shared_ptr<const SelectedPublication>(new SelectedPublication(revision, split, bindings, logical, source));
    m_epochViews.emplace_back(epoch, view);
    return view;
}
std::shared_ptr<const SelectedPublication> GraphProgram::Select() const {
    std::lock_guard lock(m_mutex);
    return m_selected;
}
GraphEditTransaction GraphProgram::BeginEdit() const { return GraphEditTransaction(Select()); }
bool GraphProgram::Install(const GraphEditTransaction& transaction, std::shared_ptr<const SelectedPublication> ready) {
    if (!ready || ready->revision != transaction.BaseRevision() + 1) return false;
    return Install(std::move(ready));
}
bool GraphProgram::Install(std::shared_ptr<const SelectedPublication> ready) {
    BT_ZONE_SCOPE("ORG.Persistent.InstallPublication");
    if (!ready) return false;
    std::shared_ptr<const SelectedPublication> retired;
    {
        std::lock_guard lock(m_mutex);
        if (ready->revision != m_selected->revision + 1 || ready->source.lock() != m_selected) return false;
        retired = std::exchange(m_selected,std::move(ready));
    }
    // The renderer supplies its shared scheduler retirement queue. Never call
    // external retirement code while holding the selection mutex.
    RetireOwnership(m_retireOwnership,std::move(retired));
    return true;
}
bool GraphProgram::InstallRebased(std::shared_ptr<const SelectedPublication> ready) {
    BT_ZONE_SCOPE("ORG.Persistent.InstallPublication");
    if (!ready) return false;
    std::shared_ptr<const SelectedPublication> retired;
    {
        std::lock_guard lock(m_mutex);
        if (ready == m_selected) return true;
        if (m_selected->revision == UINT64_MAX) return false;
        std::shared_ptr<const SelectedPublication> renumbered(new SelectedPublication(
            m_selected->revision + 1, ready->executable, ready->bindings, ready->logical, m_selected));
        retired = std::exchange(m_selected,std::move(renumbered));
    }
    RetireOwnership(m_retireOwnership,std::move(retired));
    return true;
}
SynchronousAdmission::RetirementTicket SynchronousAdmission::CaptureRetirement(const BindingVersion& binding) {
    Require(binding.owner && binding.admission && binding.admission->resource.valid(), "Invalid retired binding");
    RetirementTicket ticket;
    ticket.resource = binding.admission->resource;
    ticket.owner = binding.owner;
    if (binding.recording) ticket.allocation = binding.recording->allocationOwner;
    ticket.heap = binding.admission->aliasHeap;
    ticket.heapIdentity = binding.admission->aliasHeapIdentity;
    Require(!ticket.heapIdentity || binding.admission->aliasHeap.get() == ticket.heapIdentity,
        "Retired alias heap lacks exact ownership");
    return ticket;
}
bool SynchronousAdmission::RetireBackingMetadata(const RetirementTicket& ticket) {
    Require(ticket.resource.valid(), "Invalid backing retirement ticket");
    Require(m_pending.empty(), "Metadata retirement during pending admission");
    if (!ticket.owner.expired() || !ticket.allocation.expired()) return false;
    if (!m_accesses.RetireCompleted(ticket.resource, m_completed)) return false;
    const std::array resources{ticket.resource};
    m_states.Invalidate(resources);
    m_incomingRevisions.erase((uint64_t{ticket.resource.generation} << 32) | ticket.resource.index);
    return true;
}
bool SynchronousAdmission::RetireAliasMetadata(const RetirementTicket& ticket) {
    Require(m_pending.empty(), "Metadata retirement during pending admission");
    if (!ticket.heapIdentity) return true;
    if (!ticket.heap.expired()) return false;
    return m_aliases.RetireCompleted(ticket.heapIdentity, ticket.heap, m_completed);
}
BindingVersion BindingVersion::FromSnapshot(std::shared_ptr<const ResourceBindingSnapshot> snapshot,
    const experimental::PreparedBackingState& initial, uint64_t descriptorRevision, uint64_t contentRevision) {
    Require(snapshot && snapshot->allocationOwner && snapshot->backingGeneration, "Incomplete publication snapshot");
    const auto handle = snapshot->resource.GetHandle();
    Require(handle.valid() && handle.index == initial.resource.index && handle.generation == initial.resource.generation,
        "Publication snapshot backing disagrees with admission");
    Require(snapshot->aliasHeap == initial.aliasHeap && snapshot->aliasHeap.get() == initial.aliasHeapIdentity
        && !snapshot->aliasHeap.owner_before(initial.aliasHeap) && !initial.aliasHeap.owner_before(snapshot->aliasHeap)
        && snapshot->aliasPoolID == initial.aliasPoolID && snapshot->aliasOffset == initial.aliasOffset
        && snapshot->aliasSize == initial.aliasSize, "Publication snapshot placement disagrees with admission");
    Require(bool(initial.regions), "Publication snapshot has no initial state regions");
    if (snapshot->description.type != rhi::ResourceType::Unknown) {
        Require(NativeBindingContract::Capture(snapshot->description).Matches(snapshot->description,initial.shape),
            "Publication snapshot shape disagrees with admission");
        Require(snapshot->description.heapType == initial.heapType, "Publication snapshot heap disagrees with admission");
    }
    BindingVersion result;
    result.identity = (uint64_t{handle.generation} << 32) | handle.index;
    result.backingRevision = snapshot->backingGeneration;
    result.descriptorRevision = descriptorRevision;
    result.contentRevision = contentRevision;
    result.shape = initial.shape;
    result.owner = snapshot;
    result.recording = std::move(snapshot);
    result.admission = std::make_shared<const experimental::PreparedBackingState>(initial);
    return result;
}
FrameAdmission SynchronousAdmission::Prepare(std::shared_ptr<const SelectedPublication> publication,
    std::span<const experimental::ExecutionTimelinePoint> queues, std::span<const FrameProducerWait> producerWaits,
    std::span<const FrameIncomingState> incomingStates, std::span<const FrameRebinding> rebindings) {
    BT_ZONE_SCOPE("ORG.Persistent.PrepareAdmission");
    Require(publication && publication->executable && publication->executable->graph, "No executable publication");
    Require(m_pending.empty() || (m_closed && m_pending.size() < m_capacity), "Previous synchronous admission has not terminated");
    Require(m_nextSequence != UINT64_MAX, "Admission sequence exhausted");
    Require(m_retained.size() < m_capacity, "Synchronous frame capacity exhausted");
    FrameAdmission frame;
    frame.publication = std::move(publication);
    const auto& graph = *frame.publication->executable->graph;
    Require(queues.size() == graph.structure->queues.size(), "Invalid admission queues");
    for (const auto point : queues) {
        Require(point.timeline && std::ranges::find(frame.queueTimelines,point.timeline) == frame.queueTimelines.end(),
            "Admission queues require distinct timeline identities");
        frame.queueTimelines.push_back(point.timeline);
    }
    Require(frame.publication->executable->resourceSlots.size() == graph.structure->resourceIDs.size(),
        "Invalid executable resource slot map");
    frame.backings.reserve(frame.publication->executable->resourceSlots.size());
    for (auto slot : frame.publication->executable->resourceSlots) {
        const auto& binding = frame.publication->bindings.At(slot);
        Require(bool(binding.admission), "Binding has no prepared admission state");
        frame.backings.push_back(*binding.admission);
    }
    if (!rebindings.empty()) {
        BT_ZONE_SCOPE("ORG.Persistent.ApplyFrameRebindings");
        for (const auto& rebinding : rebindings) {
            const auto& binding = frame.publication->bindings.At(rebinding.slot);
            Require(!binding.bound, "Frame rebinding targets a publication-owned slot");
            const auto index = frame.publication->executable->resourceIndexBySlot.at(rebinding.slot.index);
            Require(index < frame.backings.size(), "Frame rebinding has no compiled resource");
            Require(std::ranges::find(frame.rebound,index) == frame.rebound.end(), "Duplicate frame rebinding");
            Require(rebinding.backing.resource.valid() && rebinding.backing.regions
                && rebinding.backing.shape == binding.shape && !rebinding.backing.aliasSize
                && rebinding.backing.graphResourceID == uint64_t{rebinding.slot.index} + 1, "Invalid frame rebinding");
            Require(rebinding.recording && rebinding.recording->resource.GetHandle().index == rebinding.backing.resource.index
                && rebinding.recording->resource.GetHandle().generation == rebinding.backing.resource.generation,
                "Frame rebinding recording disagrees with admission");
            frame.backings[index] = rebinding.backing;
            frame.rebound.push_back(index);
        }
    }
    frame.incomingWaits.resize(graph.batches.size());
    const auto appendProducerWait = [&](uint32_t batch, experimental::ExecutionTimelinePoint completion) {
        Require(completion.timeline && completion.value, "Invalid frame producer completion");
        const auto consumerQueue = graph.batches.at(batch).queue;
        const auto producerQueue = std::ranges::find(queues,completion.timeline,&experimental::ExecutionTimelinePoint::timeline);
        if (producerQueue != queues.end())
            Require(completion.value <= producerQueue->value, "Frame producer wait depends on future graph-queue work");
        if (completion.timeline == queues[consumerQueue].timeline) return;
        auto& waits = frame.incomingWaits[batch];
        const auto found = std::ranges::find(waits,completion.timeline,&experimental::ExecutionTimelinePoint::timeline);
        if (found == waits.end()) waits.push_back(completion);
        else found->value = std::max(found->value,completion.value);
    };
    if (!producerWaits.empty()) {
        BT_ZONE_SCOPE("ORG.Persistent.ResolveProducerWaits");
        for (const auto& wait : producerWaits) {
            Require(wait.completion.timeline && wait.completion.value, "Invalid frame producer completion");
            const auto& logical = *frame.publication->logical;
            Require(wait.consumer.index < logical.passSlots.size(), "Frame producer wait has absent consumer");
            const auto& pass = logical.passSlots[wait.consumer.index];
            Require(pass.active && pass.generation == wait.consumer.generation, "Frame producer wait has stale consumer");
            const auto& layout = frame.publication->executable->executionLayout;
            Require(layout && wait.consumer.index < layout->placements.size(), "Frame producer wait has no execution placement");
            const auto placement = layout->placements[wait.consumer.index];
            Require(placement.preparedPass != UINT32_MAX && placement.batch < frame.incomingWaits.size()
                && placement.queue < queues.size(), "Frame producer wait has invalid execution placement");
            appendProducerWait(placement.batch,wait.completion);
        }
    }
    if (!incomingStates.empty()) {
        BT_ZONE_SCOPE("ORG.Persistent.ResolveIncomingStates");
        std::vector<uint32_t> reported; reported.reserve(incomingStates.size());
        for (const auto& report : incomingStates) {
            const auto& binding = frame.publication->bindings.At(report.resource);
            Require(binding.admission && report.backing.valid()
                && report.backing.index == binding.admission->resource.index
                && report.backing.generation == binding.admission->resource.generation, "Incoming state has wrong selected backing");
            const auto& executable = *frame.publication->executable;
            const auto index = executable.resourceIndexBySlot.at(report.resource.index);
            Require(index < frame.backings.size(), "Incoming state has no compiled resource");
            Require(std::ranges::find(reported,index) == reported.end(), "Incoming state repeats a shared physical resource");
            reported.push_back(index);
            Require(report.revision && report.producerCompletion.timeline && report.producerCompletion.value,
                "Incoming state has no producer revision/completion");
            const auto key = (uint64_t{report.backing.generation} << 32) | report.backing.index;
            if (const auto previous = m_incomingRevisions.find(key); previous != m_incomingRevisions.end()) {
                Require(report.revision >= previous->second.revision, "Incoming producer revision regressed");
                if (report.revision == previous->second.revision) {
                    Require(report.producerCompletion == previous->second.completion, "Incoming revision has inconsistent completion");
                    continue;
                }
                if (report.producerCompletion.timeline == previous->second.completion.timeline)
                    Require(report.producerCompletion.value > previous->second.completion.value,
                        "New incoming revision has no newer producer submission");
            }
            const auto& consumers = executable.incomingBatchesByResource.at(index);
            Require(!consumers.empty(), "Incoming state has no frame consumer");
            const auto stateBatch = executable.incomingStateBatchByResource.at(index);
            Require(stateBatch != UINT32_MAX, "Incoming state has no state consumer");
            frame.incomingEffects.push_back({report.backing,report.revision,report.producerCompletion,stateBatch});
            frame.backings[index].regions = report.regions;
            frame.backings[index].authoritativeIncoming = true;
            for (const auto batch : consumers) appendProducerWait(batch,report.producerCompletion);
        }
    }
    const auto invalidated = m_aliases.ApplyInitialStates(graph, frame.backings);
    frame.closed = m_closed;
    // A closed execution's plan is a function of its executable and backings alone, unless the frame brings
    // states of its own (rebindings, incoming states, alias invalidation).
    const bool cacheable = m_closed && rebindings.empty() && incomingStates.empty() && invalidated.empty();
    CachedPlan* cached = nullptr;
    if (cacheable) {
        std::erase_if(m_planCache, [](const CachedPlan& entry) { return entry.executable.expired(); });
        for (auto& entry : m_planCache) {
            if (entry.executable.lock() != frame.publication->executable || entry.resources.size() != frame.backings.size()) continue;
            bool same = true;
            for (size_t i = 0; i < frame.backings.size() && same; ++i)
                same = entry.resources[i].index == frame.backings[i].resource.index
                    && entry.resources[i].generation == frame.backings[i].resource.generation
                    && entry.regions[i] == frame.backings[i].regions.get();
            if (same) { cached = &entry; break; }
        }
    }
    if (cached) {
        frame.barriers = cached->barriers;
        frame.cachedPlan = true;
        ++m_planCacheHits;
    } else {
        frame.barriers = m_states.Prepare(graph, frame.backings, invalidated, m_closed);
        if (cacheable) {
            ++m_planCacheMisses;
            if (m_planCache.size() >= 64) m_planCache.erase(m_planCache.begin());
            CachedPlan entry;
            entry.executable = frame.publication->executable;
            for (const auto& backing : frame.backings) {
                entry.resources.push_back(backing.resource);
                entry.regions.push_back(backing.regions.get());
            }
            entry.barriers = frame.barriers;
            m_planCache.push_back(std::move(entry));
        }
    }
    m_accesses.AppendIncomingWaits(graph, frame.backings, queues, frame.incomingWaits);
    m_aliases.AppendIncomingWaits(graph, frame.backings, queues, frame.incomingWaits);
    frame.sequence = m_nextSequence++;
    frame.domain = m_domain;
    m_pending.push_back(frame.sequence);
    return frame;
}
void SynchronousAdmission::SetClosedExecutions(bool closed) {
    Require(m_pending.empty(), "Closed executions changed during a pending admission");
    if (m_closed != closed) m_planCache.clear();
    m_closed = closed;
}
SynchronousAdmission::SynchronousAdmission(size_t frameCapacity,
    std::function<void(std::shared_ptr<const void>)> retireOwnership)
    : m_capacity(frameCapacity), m_retireOwnership(std::move(retireOwnership)) {
    Require(frameCapacity && frameCapacity <= 64, "Invalid synchronous frame capacity");
    m_retained.reserve(frameCapacity);
    static std::atomic_uint64_t nextDomain{1};
    m_domain = nextDomain.fetch_add(1,std::memory_order_relaxed);
}
void SynchronousAdmission::Commit(const FrameAdmission& frame, const experimental::GraphExecutionTimeline& receipt,
    uint32_t signaledBatches) {
    // Open admissions commit in the order they were prepared. Closed ones depend on nothing prepared around
    // them, so they commit in the order they were submitted, whatever order they were prepared in (the
    // receipt's signals must still rise on every timeline, below).
    const auto pending = std::ranges::find(m_pending, frame.sequence);
    Require(frame.domain == m_domain && frame.sequence && pending != m_pending.end()
        && (m_closed || pending == m_pending.begin()), "Stale synchronous admission");
    const auto& graph = *frame.publication->executable->graph;
    Require(receipt.batches.size() == graph.batches.size(), "Invalid admission receipt");
    const auto count = signaledBatches == UINT32_MAX ? static_cast<uint32_t>(graph.batches.size()) : signaledBatches;
    Require(count <= graph.batches.size(), "Invalid partial admission receipt");
    auto submitted = m_submitted;
    for (uint32_t i = 0; i < count; ++i) {
        const auto& batch = receipt.batches[i];
        Require(batch.signal.timeline && batch.signal.value, "Admission receipt has no signal");
        Require(batch.signal.timeline == frame.queueTimelines.at(graph.batches[i].queue), "Receipt uses the wrong queue timeline");
        auto& previous = submitted[batch.signal.timeline];
        Require(batch.signal.value > previous, "Admission signal is not monotonic");
        previous = batch.signal.value;
    }
    RetainedFrame retained{frame.publication,{}};
    auto appendCompletion = [&](experimental::ExecutionTimelinePoint point) {
        Require(point.timeline && point.value, "Invalid frame completion");
        auto found = std::ranges::find(retained.completions,point.timeline,&experimental::ExecutionTimelinePoint::timeline);
        if (found == retained.completions.end()) retained.completions.push_back(point);
        else found->value = (std::max)(found->value,point.value);
    };
    for (uint32_t i = 0; i < count; ++i) appendCompletion(receipt.batches[i].signal);
    for (const auto point : receipt.tailCompletions) appendCompletion(point);
    for (const auto point : retained.completions)
        submitted[point.timeline] = (std::max)(submitted[point.timeline],point.value);
    // Allocate retention before mutating submission-backed ledgers.
    if (!retained.completions.empty()) m_retained.push_back(std::move(retained));
    for (uint32_t i = 0; i < count; ++i) m_states.CommitBatch(frame.barriers.batches[i]);
    for (const auto& effect : frame.incomingEffects) if (effect.firstBatch < count)
        m_incomingRevisions.insert_or_assign((uint64_t{effect.resource.generation} << 32) | effect.resource.index,
            IncomingRevision{effect.revision,effect.completion});
    m_accesses.Commit(graph, frame.backings, receipt, count);
    m_aliases.Commit(graph, frame.backings, receipt, count);
    m_submitted = std::move(submitted);
    m_pending.erase(pending);
}
void SynchronousAdmission::Abandon(const FrameAdmission& frame) {
    const auto found = std::ranges::find(m_pending, frame.sequence);
    Require(frame.domain == m_domain && frame.sequence && found != m_pending.end(), "Stale synchronous abandonment");
    // Only a closed admission can be abandoned out of order: nothing prepared after it depended on it.
    Require(m_closed || found == m_pending.begin(), "Out-of-order abandonment of an open admission");
    m_pending.erase(found);
}
void SynchronousAdmission::ExtendSubmitted(experimental::ExecutionTimelinePoint point) {
    Require(point.timeline && point.value, "Invalid submitted extension");
    auto& submitted = m_submitted[point.timeline];
    Require(point.value >= submitted, "Submitted extension is not monotonic");
    submitted = point.value;
}
uint64_t SynchronousAdmission::Submitted(uint64_t timeline) const noexcept {
    const auto found = m_submitted.find(timeline);
    return found == m_submitted.end() ? 0 : found->second;
}
size_t SynchronousAdmission::RetireCompleted(std::span<const experimental::ExecutionTimelinePoint> completed) {
    auto observations = m_completed;
    std::map<uint64_t,bool> seen;
    for (const auto point : completed) {
        Require(point.timeline && !seen[point.timeline], "Invalid duplicate completion timeline");
        seen[point.timeline] = true;
        const auto submitted = m_submitted.find(point.timeline);
        Require((submitted != m_submitted.end() && point.value <= submitted->second) || !point.value,
            "Completion exceeds submitted work");
        Require(point.value >= observations[point.timeline], "Completion moved backwards");
        observations[point.timeline] = point.value;
    }
    m_completed = std::move(observations);
    const auto before = m_retained.size();
    std::erase_if(m_retained,[&](const RetainedFrame& frame) {
        const bool done = std::ranges::all_of(frame.completions,[&](auto point) {
            const auto found = m_completed.find(point.timeline);
            return found != m_completed.end() && found->second >= point.value;
        });
        if (done) RetireOwnership(m_retireOwnership,frame.publication);
        return done;
    });
    return before - m_retained.size();
}
struct PublicationCoordinator::Job {
    explicit Job(GraphEditTransaction value) : transaction(std::move(value)) {}
    explicit Job(StructuralEdit value) : author(std::move(value)) {}
    std::optional<GraphEditTransaction> transaction;
    StructuralEdit author; // Replayable; rebased onto a newer selection when superseded.
    std::atomic_bool cancelled{false}, done{false};
    std::shared_ptr<const SelectedPublication> ready;
    std::string failure;
    PublicationState state = PublicationState::Pending;
    uint32_t rebases = 0;
};
PublicationCoordinator::PublicationCoordinator(GraphProgram& program, std::shared_ptr<runtime::ITaskService> tasks)
    : m_program(program), m_tasks(std::move(tasks)) {
    Require(bool(m_tasks), "Publication coordinator needs a task service");
    m_scope = m_tasks->CreateScope("ORG.Publication.Build");
    Require(bool(m_scope), "Publication coordinator has no task scope");
}
PublicationCoordinator::~PublicationCoordinator() {
    if (m_latest) m_latest->cancelled.store(true,std::memory_order_relaxed);
    m_scope->CancelAndWait();
}
void PublicationCoordinator::Start(const std::shared_ptr<Job>& job) {
    job->ready.reset();
    job->failure.clear();
    job->done.store(false,std::memory_order_release);
    job->state = PublicationState::Pending;
    if (!m_tasks->Submit(m_scope,runtime::TaskPriority::Streaming,"ORG.Publication.Build",[job,&program = m_program] {
        try {
            experimental::CompileWorkspace workspace;
            if (job->author) job->transaction.emplace(program.BeginEdit());
            if (job->author) job->author(*job->transaction);
            job->ready = job->transaction->Build(workspace,job->cancelled);
            if (!job->ready && !job->cancelled.load(std::memory_order_relaxed)) job->failure = "Publication build returned no result";
            if (job->cancelled.load(std::memory_order_relaxed)) job->ready.reset();
        } catch (const std::exception& error) { job->failure = error.what(); }
        job->transaction.reset();
        job->done.store(true,std::memory_order_release);
    })) {
        job->failure = "Publication worker submission rejected";
        job->done.store(true,std::memory_order_release);
    }
}
uint64_t PublicationCoordinator::Submit(GraphEditTransaction transaction) {
    if (m_latest) m_latest->cancelled.store(true,std::memory_order_relaxed);
    auto job = std::make_shared<Job>(std::move(transaction));
    m_latest = job;
    Require(m_nextSequence != UINT64_MAX, "Publication build sequence exhausted");
    const auto sequence = m_nextSequence++;
    Start(job);
    return sequence;
}
uint64_t PublicationCoordinator::Submit(StructuralEdit edit) {
    Require(bool(edit), "Missing structural edit");
    if (m_latest) m_latest->cancelled.store(true,std::memory_order_relaxed);
    auto job = std::make_shared<Job>(std::move(edit));
    m_latest = job;
    Require(m_nextSequence != UINT64_MAX, "Publication build sequence exhausted");
    const auto sequence = m_nextSequence++;
    Start(job);
    return sequence;
}
PublicationState PublicationCoordinator::Pump() {
    if (!m_latest) return PublicationState::Selected;
    auto& job = *m_latest;
    if (!job.done.load(std::memory_order_acquire)) return PublicationState::Pending;
    if (job.state != PublicationState::Pending && job.state != PublicationState::Ready) return job.state;
    if (!job.failure.empty()) return job.state = PublicationState::Failed;
    if (!job.ready || job.cancelled.load(std::memory_order_relaxed)) return job.state = PublicationState::Superseded;
    job.state = PublicationState::Ready;
    if (m_program.Install(job.ready)) return job.state = PublicationState::Selected;
    // A binding edit installed meanwhile. Replayable edits rebase instead of
    // being lost; bounded so a producer publishing every frame cannot starve it.
    constexpr uint32_t maximumRebases = 16;
    if (job.author && job.rebases < maximumRebases) {
        ++job.rebases; ++m_rebases;
        basic_telemetry::AddCounter("ORG.Persistent.StructuralRebases");
        Start(m_latest);
        return PublicationState::Pending;
    }
    return job.state = PublicationState::Superseded;
}
PublicationState PublicationCoordinator::State() const {
    if (!m_latest) return PublicationState::Selected;
    const auto& job = *m_latest;
    if (!job.done.load(std::memory_order_acquire)) return PublicationState::Pending;
    if (job.state != PublicationState::Pending) return job.state;
    if (!job.failure.empty()) return PublicationState::Failed;
    if (!job.ready || job.cancelled.load(std::memory_order_relaxed)) return PublicationState::Superseded;
    return PublicationState::Ready;
}
std::string PublicationCoordinator::Failure() const {
    if (!m_latest || !m_latest->done.load(std::memory_order_acquire)) return {};
    return m_latest->failure;
}
} // namespace org::persistent

namespace org {
RecordingContext::RecordingContext(rhi::CommandList commands,
    std::shared_ptr<const persistent::SelectedPublication> publication, PersistentTag)
    : m_persistentPublication(std::move(publication)), m_commands(commands) {
    if (!m_commands || !m_persistentPublication) throw std::invalid_argument("Incomplete persistent recording context");
}
RecordingContext RecordingContext::FromPersistentBindings(rhi::CommandList commands,
    std::shared_ptr<const persistent::SelectedPublication> publication) {
    return RecordingContext(commands,std::move(publication),PersistentTag{});
}
RecordingContext RecordingContext::WithPersistentBindings(
    std::shared_ptr<const persistent::SelectedPublication> publication) const {
    if (!publication) throw std::invalid_argument("Missing persistent recording publication");
    auto scoped = *this;
    scoped.m_persistentPublication = std::move(publication);
    return scoped;
}
rhi::Resource RecordingContext::Resolve(persistent::BindingToken token) const {
    if (!m_persistentPublication) throw std::logic_error("Persistent token requires selected recording bindings");
    return m_persistentPublication->ResolveNative(token);
}
rhi::DescriptorSlot RecordingContext::Resolve(persistent::ViewToken token) const {
    if (!m_persistentPublication) throw std::logic_error("Persistent view requires selected recording bindings");
    return m_persistentPublication->ResolveView(token);
}
} // namespace org
