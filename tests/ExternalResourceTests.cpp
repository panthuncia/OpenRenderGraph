#include <Resources/ExternalBufferResource.h>
#include <Resources/ExternalTextureResource.h>
#include <Render/Runtime/RuntimeDevice.h>
#include <Render/PassBuilders.h>
#include <RenderPasses/Base/TypedRenderGraphPass.h>
#include <Render/RenderGraph/ExperimentalRhiExecution.h>
#include "../src/Render/RenderGraph/FrameRecording.h"
#include <Render/RenderGraph/ExperimentalExecutionState.h>
#include <Render/BufferBarrierHelpers.h>
#include <Render/DescriptorSnapshots.h>
#include <Render/Runtime/FrameWorkQueue.h>
#include <Render/Runtime/ExternalSignalReservation.h>
#include <Resources/TrackedAllocation.h>
#include <rhi_interop_dx12.h>
#include <rhi_helpers.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <future>
#include <algorithm>
#include <cstring>
#include <type_traits>

#define CHECK(x) do { if (!(x)) return __LINE__; } while (false)
int TestDelayedFrameRecording(const rhi::DeviceCreateInfo& create);
int TestFrameRetirement(rhi::Device device);
int TestProgramVersions(rhi::Device device);
int TestReadbackCaptures(rhi::Device device);

namespace {
struct RecordingStatisticsProbe final : org::runtime::IStatisticsService {
    unsigned begins = 0, ends = 0, resolves = 0, merges = 0, cpuSamples = 0;
    void Initialize() override {}
    void BeginFrame() override {}
    void ClearAll() override {}
    unsigned RegisterPass(const std::string&, bool, std::string_view) override { return 0; }
    void RegisterQueue(rhi::QueueKind) override {}
    void SetupQueryHeap() override {}
    void BeginQuery(unsigned, unsigned, rhi::Queue&, rhi::CommandList&) override { throw std::logic_error("Shared query recording"); }
    void EndQuery(unsigned, unsigned, rhi::Queue&, rhi::CommandList&) override { throw std::logic_error("Shared query recording"); }
    void ResolveQueries(unsigned, rhi::Queue&, rhi::CommandList&) override { throw std::logic_error("Shared query recording"); }
    void OnFrameComplete(unsigned, rhi::Queue&) override {}
    void RecordCpuUpdateTime(unsigned, double) override {}
    void RecordCpuExecuteTime(unsigned, double milliseconds) override {
        if (milliseconds < 0) throw std::logic_error("Negative CPU timing");
        ++cpuSamples;
    }
    void BeginQuery(unsigned, unsigned, rhi::Queue&, rhi::CommandList&, org::runtime::QueryRecordingContext&) override { ++begins; }
    void EndQuery(unsigned, unsigned, rhi::Queue&, rhi::CommandList&, org::runtime::QueryRecordingContext&) override { ++ends; }
    void ResolveQueries(unsigned, rhi::Queue&, rhi::CommandList&, org::runtime::QueryRecordingContext&) override { ++resolves; }
    void MergePendingResolves(rhi::QueueKind, unsigned, org::runtime::QueryRecordingContext&) override { ++merges; }
    const std::vector<std::string>& GetPassNames() const override { return names; }
    const std::vector<std::string>& GetPassTechniquePaths() const override { return names; }
    const std::vector<org::runtime::PassStats>& GetPassStats() const override { return stats; }
    const std::vector<org::runtime::MeshPipelineStats>& GetMeshStats() const override { return mesh; }
    org::runtime::MemoryBudgetStats GetMemoryBudgetStats() const override { return {}; }
    const std::vector<bool>& GetIsGeometryPassVector() const override { return geometry; }
    const std::vector<unsigned>& GetVisiblePassIndices(uint64_t) const override { return visible; }
    std::vector<std::string> names;
    std::vector<org::runtime::PassStats> stats;
    std::vector<org::runtime::MeshPipelineStats> mesh;
    std::vector<bool> geometry;
    std::vector<unsigned> visible;
};
struct DirectRecordingPass : org::TypedRenderGraphPass<DirectRecordingPass> {
    inline static unsigned recordings = 0;
    void Declare(org::PassBuilder&) {}
    static void Record(org::PassRecordContext&) { ++recordings; }
};
struct DeclaredTestBindings { org::ResourceBindingToken resource; };
struct DeclaredRecordingPass : org::TypedRenderGraphPass<DeclaredRecordingPass,
    org::EmptyPassFrameData, DeclaredTestBindings> {
    inline static rhi::ResourceHandle observed{};
    uint64_t selected = 41;
    DeclaredTestBindings Declare(org::PassBuilder&) { return {{selected, 0}}; }
    static void Record(const DeclaredTestBindings& bindings, org::PassRecordContext& context) {
        observed = context.Resolve(bindings.resource).GetHandle();
    }
};
struct DeclaredPreparedPass : org::TypedRenderGraphPass<DeclaredPreparedPass,
    uint64_t, DeclaredTestBindings> {
    inline static uint64_t observedFrame = 0;
    DeclaredTestBindings Declare(org::PassBuilder&) { return {{41, 0}}; }
    uint64_t Prepare(const DeclaredTestBindings&, const org::PassPrepareContext& context) const {
        return context.frameNumber;
    }
    static void Record(const DeclaredTestBindings& bindings, const uint64_t& frame,
        org::PassRecordContext& context) {
        (void)context.Resolve(bindings.resource);
        observedFrame = frame;
    }
};

struct TypedLifecycleCounts { int recorded = 0, submitted = 0, completed = 0, abandoned = 0; };
struct SignalRecorder {
    static void Record(const org::EmptyPassFrameData&, org::RecordingContext&) {}
};

int TestSignalReservations(rhi::Device device) {
    for (bool submit : {false, true}) {
        int cancelled = 0;
        std::weak_ptr<rhi::TimelinePtr> lifetime;
        {
            auto timeline = std::make_shared<rhi::TimelinePtr>();
            CHECK(device.CreateTimeline(*timeline, 0, "Service reservation") == rhi::Result::Ok);
            lifetime = timeline;
            org::PreparedDependencyCollector collector;
            collector.Reserve(std::make_shared<org::runtime::ExternalSignalReservation>(
                timeline, 7, [&] { ++cancelled; }));
            auto dependencies = std::move(collector).Freeze();
            auto packet = org::PreparedPass::FromTyped<SignalRecorder>(org::EmptyPassFrameData{}, dependencies);
            timeline.reset();
            CHECK(!lifetime.expired());
            CHECK(packet.ExternalSignalsAfterCompletion().size() == 1);
            CHECK(packet.ExternalSignalsAfterCompletion()[0].value == 7);
            CHECK(packet.ExternalSignalsAfterCompletion()[0].timeline.IsValid());
            if (submit) dependencies->Submitted({1});
            else {
                CHECK(packet.Abandon(org::AbandonReason::GenerationInvalidated));
                CHECK(!packet.Abandon(org::AbandonReason::Shutdown));
            }
            dependencies.reset();
            CHECK(!lifetime.expired());
        }
        CHECK(lifetime.expired());
        CHECK(cancelled == (submit ? 0 : 1));
    }
    return 0;
}
struct TypedLifecycleData { std::shared_ptr<TypedLifecycleCounts> counts; };
struct TypedLifecyclePass {
    static void Record(const TypedLifecycleData& data, org::RecordingContext&) { ++data.counts->recorded; }
    static void Submitted(const TypedLifecycleData& data, const org::SubmissionContext&) { ++data.counts->submitted; }
    static void Completed(const TypedLifecycleData& data, const org::CompletionContext&) { ++data.counts->completed; }
    static void Abandoned(const TypedLifecycleData& data, org::AbandonReason) { ++data.counts->abandoned; }
};

int TestServiceWorkReservations() {
    org::runtime::FrameWorkQueue<int> queue;
    queue.Enqueue(1); queue.Enqueue(2);
    auto selected = queue.Pending();
    org::FramePreparationContext prepare;
    prepare.dependencyCollector = std::make_shared<org::PreparedDependencyCollector>();
    queue.Reserve(selected, prepare);
    auto first = std::move(*prepare.dependencyCollector).Freeze();
    CHECK(queue.Pending().empty());
    queue.Enqueue(3);
    first->Abandoned(org::AbandonReason::AdmissionFailed);
    first->Abandoned(org::AbandonReason::Shutdown);
    auto restored = queue.Pending();
    CHECK(restored.size() == 3 && restored[0]->work == 1 && restored[2]->work == 3);
    prepare.dependencyCollector = std::make_shared<org::PreparedDependencyCollector>();
    queue.Reserve(restored, prepare);
    auto submitted = std::move(*prepare.dependencyCollector).Freeze();
    submitted->Submitted({1});
    submitted->Abandoned(org::AbandonReason::Shutdown);
    submitted.reset();
    CHECK(queue.Pending().empty());
    // Publication failure restores the queue even without an explicit callback.
    queue.Enqueue(4);
    prepare.dependencyCollector.reset();
    bool rejected = false;
    try { queue.Reserve(queue.Pending(), prepare); } catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected && queue.Pending().size() == 1);
    prepare.dependencyCollector = std::make_shared<org::PreparedDependencyCollector>();
    queue.Reserve(queue.Pending(), prepare);
    auto discarded = std::move(*prepare.dependencyCollector).Freeze();
    queue.DiscardUnsubmitted([](int work) { return work == 4; });
    discarded->Abandoned(org::AbandonReason::GenerationInvalidated);
    CHECK(queue.Pending().empty());
    // Stale and duplicate selections must not partially consume valid work.
    queue.Enqueue(5);
    auto duplicates = queue.Pending(); duplicates.push_back(duplicates.front());
    rejected = false;
    try { queue.Reserve(duplicates, prepare); } catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected && queue.Pending().size() == 1);
    const auto counters = queue.ReadCounters();
    CHECK(counters.accepted == 5 && counters.submitted == 3 && counters.discarded == 1);
    CHECK(counters.pending == 1 && counters.reserved == 0 && counters.returned == 3);
    struct CommitWork { std::shared_ptr<unsigned> count; void Commit() const { ++*count; } };
    org::runtime::FrameWorkQueue<CommitWork> commits;
    auto count = std::make_shared<unsigned>(0);
    commits.Enqueue({count});
    prepare.dependencyCollector = std::make_shared<org::PreparedDependencyCollector>();
    commits.Reserve(commits.Pending(), prepare);
    auto effect = std::move(*prepare.dependencyCollector).Freeze();
    effect->Submitted({1}); effect->Submitted({1}); effect.reset();
    CHECK(*count == 1 && commits.Pending().empty());
    // CPU-written service allocations survive queue replacement and submission.
    // Cancellation returns the same owner; retirement releases it exactly once.
    unsigned releases = 0;
    org::runtime::FrameWorkQueue<std::shared_ptr<unsigned>> leases;
    auto lease = std::shared_ptr<unsigned>(new unsigned{42}, [&](unsigned* value) { ++releases; delete value; });
    leases.Enqueue(lease);
    lease.reset();
    prepare.dependencyCollector = std::make_shared<org::PreparedDependencyCollector>();
    leases.Reserve(leases.Pending(), prepare);
    auto retained = std::move(*prepare.dependencyCollector).Freeze();
    retained->Abandoned(org::AbandonReason::AdmissionFailed);
    retained.reset();
    CHECK(releases == 0 && leases.Pending().size() == 1);
    prepare.dependencyCollector = std::make_shared<org::PreparedDependencyCollector>();
    leases.Reserve(leases.Pending(), prepare);
    retained = std::move(*prepare.dependencyCollector).Freeze();
    retained->Submitted({2});
    leases = {};
    CHECK(releases == 0);
    retained.reset(); // The containing frame drops dependencies after retirement.
    CHECK(releases == 1);
    return 0;
}

int TestOwnedRenderFrameSnapshot() {
    org::experimental::GraphCompileInput input;
    input.structure.generation = input.structure.registryGeneration = 1;
    input.structure.resourceIDs = {1};
    input.structure.resourceShapes = {{1, 1, false}};
    input.structure.passes = {{0, 0, {{0, false}}}};
    input.structure.passes[0].preparedPassIndex = 0;
    std::atomic_bool cancelled{false};
    org::experimental::CompileWorkspace workspace;
    auto ownedInput = std::make_shared<const org::experimental::GraphCompileInput>(input);
    auto graph = org::experimental::CompileGraph(ownedInput, workspace, cancelled);
    auto bundle = std::make_shared<const org::experimental::CompiledGraphBundle>(
        org::experimental::CompiledGraphBundle{1, graph, ownedInput});
    auto layout = org::experimental::BuildExecutionLayout(bundle, input);
    auto bindings = std::make_shared<const org::FrozenExecutionBindings>(
        std::vector<org::FrozenExecutionBindings::ResourceBinding>{});
    auto record = +[](const uint32_t&, org::RecordingContext&) {};
    auto lease = std::make_shared<int>(3);
    auto barriers = std::make_shared<org::experimental::PreparedExecutionBarrierPlan>();
    barriers->batches.resize(1);
    auto frame = org::experimental::BuildRenderFrameSnapshot(7, layout,
        {org::PreparedPass::Make(uint32_t{1}, record)}, bindings,
        {org::experimental::PreparedBackingState{.graphResourceID = 1}}, barriers, {lease});
    CHECK(frame->frameNumber == 7 && frame->layout == layout && frame->passes.size() == 1);
    auto recordings = org::experimental::BuildPreparedBatchRecordings(*frame);
    CHECK(recordings.size() == 1 && recordings[0].queueSlot == 0
        && recordings[0].passes.size() == 1);
    auto realized = std::make_shared<org::experimental::RealizedResourceBundle>();
    realized->backingGenerations = {9};
    realized->bindings = bindings;
    realized->initialStates = {org::experimental::PreparedBackingState{
        .graphResourceID = 1, .resource = rhi::ResourceHandle{0, 1}}};
    realized->leases = {lease};
    std::vector<std::vector<org::ExternalTimelinePoint>> waits(1);
    waits[0].push_back({{}, 17});
    auto payload = org::experimental::BuildPreparedFramePayload(8,
        {org::PreparedPass::Make(uint32_t{2}, record)}, realized, waits);
    org::experimental::BackingStateAdmissionLedger ledger;
    auto realizedFrame = org::experimental::BuildRenderFrameSnapshot(layout, payload, ledger);
    CHECK(realizedFrame->resources == realized);
    CHECK(realizedFrame->resources->backingGenerations == std::vector<uint64_t>{9});
    CHECK(realizedFrame->externalWaitsByPreparedPass.size() == 1
        && realizedFrame->externalWaitsByPreparedPass[0][0].value == 17);
    bool rejected = false;
    try { org::experimental::BuildRenderFrameSnapshot(8, layout, {}, bindings, {}, barriers); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    return 0;
}

struct EmptyResolver final : org::ClonableResolver<EmptyResolver> {
	struct State {
		uint64_t setRevision = 1, contentRevision = 1;
		std::shared_ptr<const org::ResolverResourceList> resources = std::make_shared<const org::ResolverResourceList>();
	};
	std::shared_ptr<State> identity = std::make_shared<State>();
	std::vector<std::shared_ptr<org::Resource>> Resolve() const override { return *identity->resources; }
	std::shared_ptr<const org::ResolverDeclarationState> CaptureDeclarationState() const override {
		auto state = std::make_shared<org::ResolverDeclarationState>();
		state->tracked = true;
		state->dependencyIdentity = identity;
		state->resourceSetIdentity = {identity->setRevision, 0};
		state->contentRevision = identity->contentRevision;
		state->resources = identity->resources;
		return state;
	}
};
struct DeclarationTestPass final : org::ComputePass {
	void Setup() override {}
	void Cleanup() override {}
};
struct RefreshTestPass final : org::ComputePass {
	std::shared_ptr<EmptyResolver> resolver;
	std::shared_ptr<EmptyResolver> secondResolver;
	bool symbolic;
	int declarations = 0, setups = 0;
	RefreshTestPass(std::shared_ptr<EmptyResolver> r, bool s, std::shared_ptr<EmptyResolver> second = {})
		: resolver(std::move(r)), secondResolver(std::move(second)), symbolic(s) {}
	void Setup() override { ++setups; }
	void Cleanup() override {}
	org::ResourceRegistryView* View() const { return m_resourceRegistryView.get(); }
	void DeclareResourceUsages(org::ComputePassBuilder* builder) override {
		++declarations;
		if (symbolic) builder->WithShaderResource(org::ResourceIdentifier("test.empty-resolver"));
		else builder->WithShaderResource(*resolver);
		if (secondResolver) builder->WithShaderResource(*secondResolver);
	}
};

int TestEmptyResolverRefresh(rhi::Device device, const std::shared_ptr<org::Resource>& resource) {
	org::RenderGraph graph(device, rhi::Backend::D3D12);
	auto resolver = std::make_shared<EmptyResolver>();
	graph.RegisterResolver(org::ResourceIdentifier("test.empty-resolver"), resolver);
	graph.BuildComputePass<RefreshTestPass>("Direct", resolver, false);
	graph.BuildComputePass<RefreshTestPass>("Symbolic", resolver, true);
	graph.CompileStructural();
	graph.Setup();
	auto direct = std::dynamic_pointer_cast<RefreshTestPass>(graph.GetComputePassByName("Direct"));
	auto symbolic = std::dynamic_pointer_cast<RefreshTestPass>(graph.GetComputePassByName("Symbolic"));
	CHECK(direct && symbolic && direct->declarations == 1 && symbolic->declarations == 1);
	auto* directView = direct->View();
	auto* symbolicView = symbolic->View();
	const auto setupCount = direct->setups;
	for (unsigned step = 0; step != 3; ++step) {
		resolver->identity->resources = std::make_shared<const org::ResolverResourceList>(
			step == 1 ? org::ResolverResourceList{} : org::ResolverResourceList{resource});
		++resolver->identity->setRevision;
		org::UpdateExecutionContext context{};
		context.frameIndex = step % 2;
		context.frameFenceValue = step + 1;
		graph.Update(context, device);
		CHECK(direct->declarations == 1 && symbolic->declarations == 1);
		CHECK(direct->setups == setupCount + step + 1);
		CHECK(direct->View() == directView && symbolic->View() == symbolicView);
		const org::ResourceIdentifier numericID(std::to_string(resource->GetGlobalResourceID()));
		if (step != 1) {
			CHECK(directView->RequestHandle(numericID).GetGlobalResourceID() == resource->GetGlobalResourceID());
			CHECK(symbolicView->RequestHandle(numericID).GetGlobalResourceID() == resource->GetGlobalResourceID());
		} else {
			bool denied = false;
			try { directView->RequestHandle(numericID); } catch (const std::runtime_error&) { denied = true; }
			CHECK(denied);
		}
	}
	++resolver->identity->contentRevision;
	graph.Update(org::UpdateExecutionContext{}, device);
	CHECK(direct->declarations == 1 && direct->setups == setupCount + 3);
	return 0;
}

int TestMixedResolverCollision(rhi::Device device, const std::shared_ptr<org::Resource>& resource) {
	org::RenderGraph graph(device, rhi::Backend::D3D12);
	auto initiallyEmpty = std::make_shared<EmptyResolver>();
	auto populated = std::make_shared<EmptyResolver>();
	populated->identity->resources = std::make_shared<const org::ResolverResourceList>(org::ResolverResourceList{resource});
	graph.BuildComputePass<RefreshTestPass>("Mixed", initiallyEmpty, false, populated);
	graph.CompileStructural();
	graph.Setup();
	auto pass = std::dynamic_pointer_cast<RefreshTestPass>(graph.GetComputePassByName("Mixed"));
	CHECK(pass && pass->declarations == 1);
	initiallyEmpty->identity->resources = populated->identity->resources;
	++initiallyEmpty->identity->setRevision;
	graph.Update(org::UpdateExecutionContext{}, device);
	// A new set overlapping an unchanged resolver must use the full merge path,
	// not silently duplicate accesses after discarding unchanged membership IDs.
	CHECK(pass->declarations == 2);
	return 0;
}

int TestEmptyResolverDeclarations(rhi::Device device) {
	org::RenderGraph graph(device, rhi::Backend::D3D12);
	auto resolver = std::make_shared<EmptyResolver>();
	const org::ResourceIdentifier identifier("test.empty-resolver");
	graph.RegisterResolver(identifier, resolver);
	auto& direct = graph.BuildComputePass<DeclarationTestPass>("Direct");
	direct.WithShaderResource(*resolver);
	auto& symbolic = graph.BuildComputePass<DeclarationTestPass>("Symbolic");
	symbolic.WithShaderResource(identifier);
	auto directStates = direct.TakeResolverSnapshots();
	auto symbolicStates = symbolic.TakeResolverSnapshots();
	CHECK(directStates.size() == 1 && symbolicStates.size() == 1);
	const auto& a = directStates.front();
	const auto& b = symbolicStates.front();
	CHECK(a.resourceIDs.empty() && b.resourceIDs.empty());
	CHECK(a.dependencyIdentity == b.dependencyIdentity);
	CHECK(a.resolver.get() != resolver.get());
	CHECK(!a.hasUnclassifiedDeclaration && !b.hasUnclassifiedDeclaration);
	CHECK(a.declaredRequirementTemplates.size() == 1 && b.declaredRequirementTemplates.size() == 1);
	CHECK(a.declaredRequirementTemplates[0].range == b.declaredRequirementTemplates[0].range);
	CHECK(a.declaredRequirementTemplates[0].state == b.declaredRequirementTemplates[0].state);
	CHECK(a.declaredRequirementTemplates[0].state.sync == b.declaredRequirementTemplates[0].state.sync);
	// Multiple authored uses must survive an empty initial capture, too.
	auto& multiple = graph.BuildComputePass<DeclarationTestPass>("Multiple");
	multiple.WithShaderResource(*resolver).WithUnorderedAccess(*resolver).WithShaderResource(*resolver);
	auto multiStates = multiple.TakeResolverSnapshots();
	CHECK(multiStates.size() == 1);
	CHECK(multiStates[0].declaredRequirementTemplates.size() == 2);
	return 0;
}

Microsoft::WRL::ComPtr<ID3D12Resource> MakeResource(ID3D12Device* device, bool texture, uint64_t size) {
	D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
	D3D12_RESOURCE_DESC desc{}; desc.Dimension = texture ? D3D12_RESOURCE_DIMENSION_TEXTURE2D : D3D12_RESOURCE_DIMENSION_BUFFER;
	desc.Width = texture ? 64 : size; desc.Height = texture ? 64 : 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
	desc.Format = texture ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_UNKNOWN; desc.SampleDesc.Count = 1;
	desc.Layout = texture ? D3D12_TEXTURE_LAYOUT_UNKNOWN : D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	Microsoft::WRL::ComPtr<ID3D12Resource> result;
	device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&result));
	return result;
}
}

int TestPreparedGpuSubmission(rhi::Device device, ID3D12Device* nativeDevice) {
    using namespace org::experimental;
    GraphCompileInput input;
    input.structure.generation = input.structure.registryGeneration = 1;
    input.structure.resourceIDs = {1,2,3};
    input.structure.resourceShapes.resize(3);
    CompilePass pass;
    pass.accesses = {{0,false},{1,true}};
    const auto copyState = [](bool write) {
        return CompileResourceState{static_cast<uint64_t>(write ? rhi::ResourceAccessType::CopyDest : rhi::ResourceAccessType::CopySource),
            0, static_cast<uint64_t>(rhi::ResourceSyncState::Copy), write};
    };
    pass.entryStates = {{0,{},copyState(false)}, {1,{},copyState(true)}};
    // This test deliberately submits upload and readback as separate packets.
    // The compiler otherwise legally merges the two same-queue passes.
    pass.preparedPassIndex = 0;
    pass.forceBatchIsolation = true;
    CompilePass readbackPass;
    readbackPass.originalOrder = 1;
    readbackPass.accesses = {{1,false},{2,true}};
    readbackPass.entryStates = {{1,{},copyState(false)}, {2,{},copyState(true)}};
    readbackPass.preparedPassIndex = 1;
    readbackPass.forceBatchIsolation = true;
    input.structure.passes = {pass,readbackPass};
    auto owned = std::make_shared<const GraphCompileInput>(std::move(input));
    auto job = std::async(std::launch::async, [owned] {
        CompileWorkspace workspace; std::atomic_bool cancelled{false};
        return workspace.Compile(owned, cancelled);
    });
    auto compiled = job.get();
    CHECK(compiled && compiled->states.complete && compiled->stateValidationError.empty());
    auto bundle = std::make_shared<const CompiledGraphBundle>(CompiledGraphBundle{1,compiled,owned});
    rhi::TimelinePtr timeline;
    CHECK(device.CreateTimeline(timeline, 0, "Experimental submission test") == rhi::Result::Ok);
    ExecutionTimelineAdmission admission({{1,0}});
    BackingStateAdmissionLedger backingStates;
    // Fresh concrete backing and recording data on every execution; the compiled
    // structure is reused without replaying either upload or command list.
    for (uint32_t iteration = 1; iteration <= 2; ++iteration) {
        struct Lease {
            Microsoft::WRL::ComPtr<ID3D12Resource> native[3];
            rhi::Resource resources[3];
            std::shared_ptr<const org::TrackedHandle> allocations[3];
            rhi::CommandAllocatorPtr allocators[2];
            rhi::CommandListPtr lists[2];
        };
        auto lease = std::make_shared<Lease>();
        for (int i = 0; i < 3; ++i) {
            D3D12_HEAP_PROPERTIES heap{};
            heap.Type = i == 0 ? D3D12_HEAP_TYPE_UPLOAD : i == 1 ? D3D12_HEAP_TYPE_DEFAULT : D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Width = 4096; desc.Height = 1; desc.DepthOrArraySize = desc.MipLevels = 1;
            desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            CHECK(SUCCEEDED(nativeDevice->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                i == 0 ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr, IID_PPV_ARGS(&lease->native[i]))));
            rhi::ResourcePtr imported;
            CHECK(rhi::dx12::import_resource(device, lease->native[i].Get(), imported) == rhi::Result::Ok);
            auto tracked = org::TrackedHandle::FromResource(std::move(imported), {});
            lease->resources[i] = tracked.GetResource();
            lease->allocations[i] = tracked.CaptureAllocationLease();
            tracked.Reset(); // Simulate logical retirement before GPU submission.
        }
        const CompileResourceState commonState{
            static_cast<uint64_t>(rhi::ResourceAccessType::Common), 0,
            static_cast<uint64_t>(rhi::ResourceSyncState::All), false};
        std::vector<PreparedBackingState> initialStates;
        for (int i = 0; i < 3; ++i) {
            initialStates.push_back({static_cast<uint64_t>(i + 1), {}, lease->resources[i].GetHandle(),
                {1,1,false}, {{{},commonState}}});
            initialStates.back().heapType = i == 0 ? rhi::HeapType::Upload
                : i == 2 ? rhi::HeapType::Readback : rhi::HeapType::DeviceLocal;
        }
        auto barrierPlan = backingStates.Prepare(*compiled, initialStates);
        CHECK(barrierPlan.batches.size() == compiled->batches.size());
        CHECK(barrierPlan.batches.size() == 2);
        CHECK(!barrierPlan.batches[0].beforePass.empty() && !barrierPlan.batches[1].beforePass.empty());
        CHECK(barrierPlan.batches[0].beforePass[0].buffers.empty()
            && !barrierPlan.batches[1].beforePass[0].buffers.empty());
        void* mapped = nullptr;
        CHECK(SUCCEEDED(lease->native[0]->Map(0, nullptr, &mapped)));
        std::memset(mapped, static_cast<int>(iteration * 37), 4096);
        lease->native[0]->Unmap(0, nullptr);
        for (int i = 0; i < 2; ++i) {
            CHECK(device.CreateCommandAllocator(rhi::QueueKind::Graphics, lease->allocators[i]) == rhi::Result::Ok);
            CHECK(device.CreateCommandList(rhi::QueueKind::Graphics, lease->allocators[i].Get(), lease->lists[i]) == rhi::Result::Ok);
        }
        auto list = lease->lists[0].Get();
        org::imm::BytecodeWriter bytecode;
        bytecode.WriteOp(org::imm::Op::CopyBufferRegion);
        bytecode.WritePOD(org::imm::CopyBufferRegionCmd{{},0,{},0,2048});
        bytecode.WriteOp(org::imm::Op::CopyBufferRegion);
        bytecode.WritePOD(org::imm::CopyBufferRegionCmd{{},2048,{},2048,2048});
        int resolutions = 0;
        auto preparedCopies = org::imm::PreparedBufferCopies::Capture(bytecode.data,
            [&](org::ResourceRegistry::RegistryHandle) {
                const int index = resolutions++ % 2 == 0 ? 1 : 0;
                return org::BackingAllocationSnapshot{static_cast<uint64_t>(index + 1), iteration,
                    lease->resources[index], lease->allocations[index], 4096};
            });
        CHECK(preparedCopies && preparedCopies->Size() == 2 && resolutions == 4);
        bytecode.Reset(); // Recording cannot consult original bytecode or registry.
        auto bindings = std::make_shared<const org::FrozenExecutionBindings>(
            std::vector<org::FrozenExecutionBindings::ResourceBinding>{
                {lease->resources[0], lease->allocations[0]},
                {lease->resources[1], lease->allocations[1]}});
        org::RecordingContext recording(list, bindings);
        CHECK(rhi::HandleEqual<rhi::ResourceHandle>{}(recording.Resolve(org::PreparedResourceReference{0}).GetHandle(), lease->resources[0].GetHandle()));
        bool invalidSlotRejected = false;
        try { recording.Resolve(org::PreparedResourceReference{2}); }
        catch (const std::out_of_range&) { invalidSlotRejected = true; }
        CHECK(invalidSlotRejected);
        org::PreparedPass declaredPacket;
        {
            DeclaredRecordingPass pass;
            org::RenderGraph declarationGraph(device, rhi::Backend::D3D12);
            auto& declaration = declarationGraph.BuildRenderPass<DeclaredRecordingPass>("Declared binding lifetime");
            pass.DeclareUnified(declaration);
            auto permissions = std::make_shared<std::unordered_map<uint64_t, uint32_t>>();
            permissions->emplace(41, 0);
            org::FramePreparationContext preparation{};
            preparation.bindings = bindings;
            preparation.resourceSlots = permissions;
            preparation.frameNumber = iteration;
            declaredPacket = pass.PrepareFrame(preparation);
            DeclaredPreparedPass preparedAuthor;
            preparedAuthor.DeclareUnified(declaration);
            auto withData = preparedAuthor.PrepareFrame(preparation);
            preparation.frameNumber = iteration + 100;
            withData.Record(recording);
            CHECK(DeclaredPreparedPass::observedFrame == iteration);
            withData.CommitSubmitted();
            withData.CommitCompleted();
            // Both a graph-generation declaration refresh and mutation of the
            // original permission map must leave the queued packet untouched.
            pass.selected = 42;
            pass.DeclareUnified(declaration);
            permissions->at(41) = 99;
            bool invalidDeclarationRejected = false;
            try { (void)pass.PrepareFrame(preparation); }
            catch (const std::out_of_range&) { invalidDeclarationRejected = true; }
            CHECK(invalidDeclarationRejected);
        }
        auto replacementBindings = std::make_shared<const org::FrozenExecutionBindings>(
            std::vector<org::FrozenExecutionBindings::ResourceBinding>{
                {lease->resources[1], lease->allocations[1]},
                {lease->resources[0], lease->allocations[0]}});
        org::RecordingContext newerRecording(list, replacementBindings);
        declaredPacket.Record(newerRecording);
        CHECK(rhi::HandleEqual<rhi::ResourceHandle>{}(
            DeclaredRecordingPass::observed, lease->resources[0].GetHandle()));
        declaredPacket.CommitSubmitted();
        declaredPacket.CommitCompleted();
        bool unscopedTokenRejected = false;
        try { recording.Resolve(org::ResourceBindingToken{41, 0}); }
        catch (const std::logic_error&) { unscopedTokenRejected = true; }
        CHECK(unscopedTokenRejected);
        auto lifecycle = std::make_shared<TypedLifecycleCounts>();
        org::PreparedPass directPacket;
        {
            DirectRecordingPass pass;
            org::FramePreparationContext preparation{};
            directPacket = pass.PrepareFrame(preparation);
        } // A direct packet must not capture its authoring object.
        const auto directRecordingsBefore = DirectRecordingPass::recordings;
        directPacket.Record(recording);
        CHECK(DirectRecordingPass::recordings == directRecordingsBefore + 1);
        directPacket.CommitSubmitted();
        directPacket.CommitCompleted();
        auto typedPacket = org::PreparedPass::FromTyped<TypedLifecyclePass>(TypedLifecycleData{lifecycle});
        CHECK(typedPacket.IsWorkerSafe());
        typedPacket.Record(recording);
        typedPacket.CommitSubmitted();
        typedPacket.CommitCompleted({17});
        CHECK(lifecycle->recorded == 1 && lifecycle->submitted == 1
            && lifecycle->completed == 1 && lifecycle->abandoned == 0);
        auto abandonedPacket = org::PreparedPass::FromTyped<TypedLifecyclePass>(TypedLifecycleData{lifecycle});
        CHECK(abandonedPacket.Abandon(org::AbandonReason::GenerationInvalidated));
        CHECK(!abandonedPacket.Abandon(org::AbandonReason::Shutdown));
        CHECK(lifecycle->abandoned == 1);
        auto frameworkLifecycle = std::make_shared<TypedLifecycleCounts>();
        org::PreparedDependencyCollector dependencies;
        dependencies.Reserve(std::make_shared<const org::PreparedOwnedLifecycle<TypedLifecycleCounts>>(
            frameworkLifecycle,
            +[](TypedLifecycleCounts& value, org::SubmissionContext) { ++value.submitted; },
            +[](TypedLifecycleCounts& value, org::CompletionContext) { ++value.completed; },
            +[](TypedLifecycleCounts& value, org::AbandonReason) { ++value.abandoned; }));
        auto frameworkPacket = org::PreparedPass::FromTyped<TypedLifecyclePass>(
            TypedLifecycleData{lifecycle}, std::move(dependencies).Freeze());
        frameworkPacket.Record(recording);
        frameworkPacket.CommitSubmitted();
        frameworkPacket.CommitCompleted();
        CHECK(frameworkLifecycle->submitted == 1
            && frameworkLifecycle->completed == 1
            && frameworkLifecycle->abandoned == 0);
        org::PreparedDependencyCollector abandonedDependencies;
        abandonedDependencies.Reserve(std::make_shared<const org::PreparedOwnedLifecycle<TypedLifecycleCounts>>(
            frameworkLifecycle, nullptr, nullptr,
            +[](TypedLifecycleCounts& value, org::AbandonReason) { ++value.abandoned; }));
        auto frameworkAbandoned = org::PreparedPass::FromTyped<TypedLifecyclePass>(
            TypedLifecycleData{lifecycle}, std::move(abandonedDependencies).Freeze());
        CHECK(frameworkAbandoned.Abandon(org::AbandonReason::Shutdown));
        CHECK(frameworkLifecycle->abandoned == 1);
        struct CopyData {
            std::shared_ptr<const org::imm::PreparedBufferCopies> copies;
            std::shared_ptr<int> submitted;
        };
        auto submittedEffects = std::make_shared<int>(0);
        auto preparedPass = org::PreparedPass::Make(CopyData{preparedCopies, submittedEffects},
            +[](const CopyData& data, org::RecordingContext& context) { data.copies->Record(context.Commands()); },
            +[](const CopyData& data) { ++*data.submitted; });
        CHECK(!preparedPass.IsWorkerSafe());
        auto packetCopy = preparedPass;
        preparedPass.Record(recording);
        bool packetReplayRejected = false;
        try { packetCopy.Record(recording); }
        catch (const std::logic_error&) { packetReplayRejected = true; }
        CHECK(packetReplayRejected);
        CHECK(resolutions == 4);
        bool replayRejected = false;
        try { preparedCopies->Record(list); } catch (const std::logic_error&) { replayRejected = true; }
        CHECK(replayRejected);
        bytecode.WriteOp(org::imm::Op::ClearRTV);
        CHECK(!org::imm::PreparedBufferCopies::Capture(bytecode.data,
            [](org::ResourceRegistry::RegistryHandle) -> org::BackingAllocationSnapshot { return {}; }));
        bytecode.Reset();
        bytecode.WriteOp(org::imm::Op::CopyBufferRegion);
        bytecode.WritePOD(org::imm::CopyBufferRegionCmd{{},UINT64_MAX,{},0,16});
        CHECK(!org::imm::PreparedBufferCopies::Capture(bytecode.data,
            [&](org::ResourceRegistry::RegistryHandle) {
                return org::BackingAllocationSnapshot{1,iteration,lease->resources[0],lease->allocations[0],4096};
            }));
        list.End();
        list = lease->lists[1].Get();
        auto stateStep = std::find_if(compiled->states.steps.begin(), compiled->states.steps.end(),
            [](const auto& step) { return step.resource == 1 && step.batch == 1; });
        CHECK(stateStep != compiled->states.steps.end() && stateStep->previousBatch == 0);
        auto barrier = org::MakeWholeBufferBarrier(lease->resources[1].GetHandle(),
            static_cast<rhi::ResourceAccessType>(stateStep->before.access),
            static_cast<rhi::ResourceAccessType>(stateStep->after.access),
            static_cast<rhi::ResourceSyncState>(stateStep->before.sync),
            static_cast<rhi::ResourceSyncState>(stateStep->after.sync));
        rhi::BarrierBatch barriers{}; barriers.buffers = {&barrier,1};
        list.Barriers(barriers);
        list.CopyBufferRegion(lease->resources[2].GetHandle(), 0, lease->resources[1].GetHandle(), 0, 4096);
        list.End();
        std::vector<std::shared_ptr<const IPreparedExecutionBatch>> packets;
        for (int i = 0; i < 2; ++i)
            packets.push_back(std::make_shared<PreparedRhiExecutionBatch>(0, device.GetQueue(rhi::QueueKind::Graphics),
                std::vector<rhi::CommandList>{lease->lists[i].Get()}, std::vector<PreparedTimelineBinding>{{1,timeline.Get().GetHandle()}}, lease,
                i == 0 ? std::vector<org::PreparedPass>{packetCopy} : std::vector<org::PreparedPass>{}));
        auto execution = admission.SubmitPrepared(bundle, {{},{}}, packets);
        for (const auto& batchState : barrierPlan.batches) backingStates.CommitBatch(batchState);
        CHECK(*submittedEffects == 1);
        bool effectReplayRejected = false;
        try { packetCopy.CommitSubmitted(); } catch (const std::logic_error&) { effectReplayRejected = true; }
        CHECK(effectReplayRejected && *submittedEffects == 1);
        CHECK(timeline.Get().HostWait(iteration * 2, 10000) == rhi::Result::Ok);
        // Fault injection uses an inert queue: it must never execute these lists.
        struct QueueProbe { int fail = 0, waits = 0, submits = 0, signals = 0; };
        rhi::QueueVTable probeTable{};
        probeTable.wait = +[](rhi::Queue* queue, const rhi::TimelinePoint&) noexcept {
            auto& p = *static_cast<QueueProbe*>(queue->impl); ++p.waits;
            return p.fail == 1 ? rhi::Result::Failed : rhi::Result::Ok;
        };
        probeTable.submit = +[](rhi::Queue* queue, rhi::Span<rhi::CommandList>, const rhi::SubmitDesc&) noexcept {
            auto& p = *static_cast<QueueProbe*>(queue->impl); ++p.submits;
            return p.fail == 2 ? rhi::Result::Failed : rhi::Result::Ok;
        };
        probeTable.signal = +[](rhi::Queue* queue, const rhi::TimelinePoint&) noexcept {
            auto& p = *static_cast<QueueProbe*>(queue->impl); ++p.signals;
            return p.fail == 3 ? rhi::Result::Failed : rhi::Result::Ok;
        };
        for (int failure = 0; failure != 4; ++failure) {
            QueueProbe probe; probe.fail = failure;
            rhi::Queue queue; queue.impl = &probe; queue.vt = &probeTable;
            PreparedRhiExecutionBatch packet(0, queue, {lease->lists[0].Get()}, {{1,timeline.Get().GetHandle()}}, lease);
            auto batch = execution->batches[0]; batch.waits = {{1,0}};
            const auto receipt = packet.Submit(batch);
            CHECK(probe.waits == 1 && probe.submits == (failure != 1) && probe.signals == (failure != 1 && failure != 2));
            CHECK(receipt.state == (failure == 0 ? SubmissionState::Signaled : failure == 1 ? SubmissionState::NotSubmitted
                : failure == 2 ? SubmissionState::SubmissionUncertain : SubmissionState::SubmittedWithoutSignal));
            CHECK(receipt.failureStage == (failure == 0 ? SubmissionFailureStage::None : failure == 1 ? SubmissionFailureStage::Wait
                : failure == 2 ? SubmissionFailureStage::Submit : SubmissionFailureStage::Signal));
            CHECK(packet.Submit(batch).failureStage == SubmissionFailureStage::Replay);
            CHECK(probe.waits == 1);
        }
        CHECK(SUCCEEDED(lease->native[2]->Map(0, nullptr, &mapped)));
        for (size_t i = 0; i < 4096; ++i) CHECK(static_cast<uint8_t*>(mapped)[i] == iteration * 37);
        lease->native[2]->Unmap(0, nullptr);
        CHECK(!packets[0]->Submit(execution->batches[0])); // Single-consumption guard.
        CHECK(admission.RetireCompleted(std::vector<ExecutionTimelinePoint>{{1,iteration * 2}}) == 1);
    }
    return 0;
}

int TestDescriptorSnapshots(const rhi::DeviceCreateInfo& create) {
    auto deviceOwner = std::make_shared<rhi::DevicePtr>();
    CHECK(rhi::CreateD3D12Device(create, *deviceOwner) == rhi::Result::Ok);
    auto device = deviceOwner->Get();
    struct Backing {
        std::shared_ptr<rhi::DevicePtr> device;
        Microsoft::WRL::ComPtr<ID3D12Resource> native;
        rhi::ResourcePtr resource;
    };
    auto backing = std::make_shared<Backing>(); backing->device = deviceOwner;
    backing->native = MakeResource(rhi::dx12::get_device(device), false, 4096);
    CHECK(rhi::dx12::import_resource(device, backing->native.Get(), backing->resource) == rhi::Result::Ok);
    auto writes = std::make_shared<unsigned>(0);
    struct View { std::shared_ptr<Backing> backing; std::shared_ptr<unsigned> writes; bool fail; };
    const auto encode = +[](const View& view, rhi::Device target, rhi::DescriptorSlot slot) noexcept {
        ++*view.writes;
        if (view.fail) return rhi::Result::Failed;
        rhi::SrvDesc desc{}; desc.dimension = rhi::SrvDim::Buffer; desc.buffer.numElements = 1024;
        return target.CreateShaderResourceView(slot, view.backing->resource.Get().GetHandle(), desc);
    };
    auto original = org::OwnedDescriptorWrite::Make(View{backing,writes,false}, encode);
    org::DescriptorContentJournal journal({rhi::DescriptorHeapType::CbvSrvUav,130,true});
    journal.Write(3, original); journal.Write(129, original);
    auto old = journal.CaptureContents();
    CHECK(old == journal.CaptureContents());
    journal.Write(3, original);
    CHECK(old == journal.CaptureContents());
    org::DescriptorSnapshotPool pool(device, deviceOwner, 2);
    auto first = pool.Assemble(0, old);
    CHECK(first && *writes == 2 && first->Resolve(129).index == 129);
    const auto firstHeap = first->Heap().GetHandle();
    journal.Write(3, org::OwnedDescriptorWrite::Make(View{backing,writes,false}, encode));
    auto newer = journal.CaptureContents();
    CHECK(!pool.Assemble(0, newer)); // Recording/GPU lease blocks mutation.
    auto second = pool.Assemble(1, newer);
    CHECK(second && *writes == 4);
    CHECK(rhi::HandleEqual<rhi::DescriptorHeapHandle>{}(firstHeap, first->Heap().GetHandle()));
    second.reset();
    second = pool.Assemble(1, newer); CHECK(*writes == 4); // No static-frame writes.
    second.reset();
    second = pool.Assemble(1, old); CHECK(*writes == 5); // Historical contents, not latest publication.
    second.reset();
    second = pool.Assemble(1, newer); CHECK(*writes == 6); // One dirty slot, not 130.
    second.reset();
    journal.Write(3, {});
    auto removed = journal.CaptureContents();
    second = pool.Assemble(1, removed); CHECK(*writes == 6);
    bool denied = false;
    try { second->Resolve(3); } catch (const std::out_of_range&) { denied = true; }
    CHECK(denied);
    second.reset();
    journal.Write(3, org::OwnedDescriptorWrite::Make(View{backing,writes,true}, encode));
    bool failed = false;
    try { pool.Assemble(1, journal.CaptureContents()); } catch (const std::runtime_error&) { failed = true; }
    CHECK(failed && first->Resolve(3).index == 3); // Selected heap survives failed candidate population.
    second = pool.Assemble(1, old); CHECK(second && *writes == 9);
    org::DescriptorContentJournal other({rhi::DescriptorHeapType::CbvSrvUav,130,true});
    bool wrongGeneration = false;
    try { pool.Assemble(0, other.CaptureContents()); } catch (const std::invalid_argument&) { wrongGeneration = true; }
    CHECK(wrongGeneration);
    // Version-sensitive CPU views use the same journal/pool contract.
    org::DescriptorContentJournal cpuJournal({rhi::DescriptorHeapType::CbvSrvUav,130,false});
    cpuJournal.Write(3, original);
    org::DescriptorSnapshotPool cpuPool(device, deviceOwner, 1);
    auto cpu = cpuPool.Assemble(0, cpuJournal.CaptureContents());
    CHECK(cpu && cpu->Resolve(3).index == 3);
    CHECK(!cpuPool.Assemble(0, cpuJournal.CaptureContents()));

    // Heap creation failure cannot destroy a different selected execution slot.
    struct AllocationProbe { rhi::Device device; std::shared_ptr<rhi::DevicePtr> owner; bool fail = false; };
    auto probe = std::make_shared<AllocationProbe>(); probe->device = device; probe->owner = deviceOwner;
    auto faultTable = *device.vt;
    faultTable.createDescriptorHeap = +[](rhi::Device* wrapped, const rhi::DescriptorHeapDesc& desc,
        rhi::DescriptorHeapPtr& heap) noexcept {
        auto& p = *static_cast<AllocationProbe*>(wrapped->impl);
        return p.fail ? rhi::Result::Failed : p.device.CreateDescriptorHeap(desc, heap);
    };
    faultTable.createShaderResourceView = +[](rhi::Device* wrapped, rhi::DescriptorSlot slot,
        const rhi::ResourceHandle& resource, const rhi::SrvDesc& desc) noexcept {
        return static_cast<AllocationProbe*>(wrapped->impl)->device.CreateShaderResourceView(slot, resource, desc);
    };
    auto faultDevice = device; faultDevice.impl = probe.get(); faultDevice.vt = &faultTable;
    org::DescriptorSnapshotPool faultPool(faultDevice, probe, 2);
    auto selected = faultPool.Assemble(0, old);
    probe->fail = true;
    bool allocationFailed = false;
    try { faultPool.Assemble(1, newer); } catch (const std::runtime_error&) { allocationFailed = true; }
    CHECK(allocationFailed && selected->Resolve(3).index == 3);
    probe->fail = false;
    auto recovered = faultPool.Assemble(1, newer); CHECK(recovered);
    return 0;
}

int TestOwnedDescriptorGpuExecution(const rhi::DeviceCreateInfo& create) {
    using namespace org::experimental;
    struct Runtime {
        rhi::DevicePtr device;
        rhi::TimelinePtr timeline;
        rhi::ma::Allocator* allocator = nullptr;
        ~Runtime() { if (allocator) allocator->ReleaseThis(); }
    };
    auto runtime = std::make_shared<Runtime>();
    CHECK(rhi::CreateD3D12Device(create, runtime->device) == rhi::Result::Ok);
    auto device = runtime->device.Get();
    rhi::ma::AllocatorDesc allocatorDesc{}; allocatorDesc.device = device;
    CHECK(rhi::ma::CreateAllocator(&allocatorDesc,&runtime->allocator) == rhi::Result::Ok);
    CHECK(device.CreateTimeline(runtime->timeline, 0, "Owned descriptor execution") == rhi::Result::Ok);
    {
        OwnedRecordingList closed;
        closed.bindings = std::make_shared<const org::FrozenExecutionBindings>(
            std::vector<org::FrozenExecutionBindings::ResourceBinding>{});
        closed.passes.push_back(org::PreparedPass::Make(0,+[](const int&,org::RecordingContext&) {}));
        CHECK(device.CreateCommandAllocator(rhi::QueueKind::Graphics,closed.allocation->pair.allocator) == rhi::Result::Ok);
        CHECK(device.CreateCommandList(rhi::QueueKind::Graphics,closed.allocation->pair.allocator.Get(),closed.allocation->pair.list) == rhi::Result::Ok);
        auto legacy = closed.allocation->pair.list.Get(); auto legacyTable = *legacy.vt;
        legacyTable.abi_version = 5; legacy.vt = &legacyTable;
        CHECK(!legacy.SupportsCheckedEnd() && legacy.EndChecked() == rhi::Result::Unsupported);
        CHECK(closed.allocation->pair.list.Get().EndChecked() == rhi::Result::Ok);
        std::vector<OwnedRecordingList> lists; lists.push_back(std::move(closed));
        bool rejected = false;
        try {
            // Closing the already-closed native list fails. No submission packet
            // may escape, and this test never submits the invalid command list.
            RecordPreparedRhiExecutionBatch(0,device.GetQueue(rhi::QueueKind::Graphics),std::move(lists),
                {{1,runtime->timeline.Get().GetHandle()}},runtime);
        } catch (const std::runtime_error&) { rejected = true; }
        CHECK(rejected);
    }
    org::DescriptorContentJournal gpuJournal({rhi::DescriptorHeapType::CbvSrvUav,16,true});
    org::DescriptorContentJournal cpuJournal({rhi::DescriptorHeapType::CbvSrvUav,16,false});
    org::DescriptorSnapshotPool gpuPool(device, runtime, 1), cpuPool(device, runtime, 1);
    GraphCompileInput input;
    input.structure.generation = input.structure.registryGeneration = 1;
    input.structure.resourceIDs = {1,2}; input.structure.resourceShapes.resize(2);
    CompilePass clear, readback;
    clear.accesses = {{0,true}};
    clear.entryStates = {{0,{}, {static_cast<uint64_t>(rhi::ResourceAccessType::UnorderedAccessClear),0,
        static_cast<uint64_t>(rhi::ResourceSyncState::ClearUnorderedAccessView),true}}};
    readback.originalOrder = 1; readback.accesses = {{0,false},{1,true}};
    readback.entryStates = {
        {0,{}, {static_cast<uint64_t>(rhi::ResourceAccessType::CopySource),0,static_cast<uint64_t>(rhi::ResourceSyncState::Copy),false}},
        {1,{}, {static_cast<uint64_t>(rhi::ResourceAccessType::CopyDest),0,static_cast<uint64_t>(rhi::ResourceSyncState::Copy),true}}};
    input.structure.passes = {clear,readback};
    auto owned = std::make_shared<const GraphCompileInput>(std::move(input));
    CompileWorkspace workspace; std::atomic_bool cancelled{false};
    auto graph = workspace.Compile(owned, cancelled);
    CHECK(graph && graph->stateValidationError.empty());
    CHECK(graph->batches.size() == 1 && graph->batches[0].passes == (std::vector<uint32_t>{0,1}));
    auto bundle = std::make_shared<const CompiledGraphBundle>(CompiledGraphBundle{1,graph,owned});
    ExecutionTimelineAdmission admission({{1,0}},1);
    auto commandPool = std::make_shared<org::CommandListPool>(device, rhi::QueueKind::Graphics);
    for (uint32_t iteration = 1; iteration != 3; ++iteration) {
        struct Backing {
            std::shared_ptr<Runtime> runtime;
            std::shared_ptr<const org::AliasHeapGeneration> heapGeneration;
            Microsoft::WRL::ComPtr<ID3D12Resource> native[2];
            rhi::ResourcePtr resource[2];
        };
        auto backing = std::make_shared<Backing>(); backing->runtime = runtime;
        for (unsigned i = 0; i != 2; ++i) {
            if (i == 0) {
                auto desc = rhi::helpers::ResourceDesc::Buffer(4096);
                desc.resourceFlags |= rhi::ResourceFlags::RF_AllowUnorderedAccess;
                rhi::ResourceAllocationInfo info{}; device.GetResourceAllocationInfo(&desc,1,&info);
                rhi::ma::AllocationDesc allocationDesc{};
                allocationDesc.heapType = rhi::HeapType::DeviceLocal;
                allocationDesc.flags = rhi::ma::AllocationFlagCanAlias;
                rhi::ma::AllocationPtr allocation;
                CHECK(runtime->allocator->AllocateMemory(allocationDesc,info,allocation) == rhi::Result::Ok);
                auto pool = org::TrackedHandle::FromAllocation(std::move(allocation),{});
                pool.RetainLifetimeOwner(runtime);
                backing->heapGeneration = org::AliasHeapGeneration::Capture(pool,iteration);
                CHECK(runtime->allocator->CreateAliasingResource(&backing->heapGeneration->Allocation(),0,
                    &desc,0,nullptr,backing->resource[0]) == rhi::Result::Ok);
                pool.Reset(); // No mutable pool owner survives recording/submission.
                continue;
            }
            D3D12_HEAP_PROPERTIES heap{}; heap.Type = i ? D3D12_HEAP_TYPE_READBACK : D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Width = 4096; desc.Height = 1; desc.DepthOrArraySize = desc.MipLevels = 1;
            desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            desc.Flags = i ? D3D12_RESOURCE_FLAG_NONE : D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            CHECK(SUCCEEDED(rhi::dx12::get_device(device)->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                i ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                nullptr, IID_PPV_ARGS(&backing->native[i]))));
            CHECK(rhi::dx12::import_resource(device, backing->native[i].Get(), backing->resource[i]) == rhi::Result::Ok);
        }
        auto readbackResource = backing->native[1];
        auto recipe = org::OwnedDescriptorWrite::Make(std::shared_ptr<const Backing>(backing),
            +[](const std::shared_ptr<const Backing>& data, rhi::Device target, rhi::DescriptorSlot slot) noexcept {
                rhi::UavDesc desc{}; desc.buffer.numElements = 1024;
                return target.CreateUnorderedAccessView(slot, data->resource[0].Get().GetHandle(), desc);
            });
        gpuJournal.Write(7, recipe); cpuJournal.Write(7, recipe);
        recipe.reset();
        auto gpuContents = gpuJournal.CaptureContents(), cpuContents = cpuJournal.CaptureContents();
        auto gpu = gpuPool.Assemble(0,gpuContents), cpu = cpuPool.Assemble(0,cpuContents);
        CHECK(gpu && cpu);
        auto bindings = std::make_shared<const org::FrozenExecutionBindings>(
            std::vector<org::FrozenExecutionBindings::ResourceBinding>{
                {backing->resource[0].Get(),backing}, {backing->resource[1].Get(),backing}},
            std::vector<org::FrozenExecutionBindings::DescriptorBinding>{{gpu->Resolve(7),gpu},{cpu->Resolve(7),cpu}});
        std::vector<PreparedBackingState> initial(2);
        for (uint32_t resource = 0; resource < 2; ++resource) {
            initial[resource].graphResourceID = resource + 1;
            initial[resource].resource = backing->resource[resource].Get().GetHandle();
            initial[resource].shape = {1,1,false};
        }
        initial[0].regions.push_back({{}, {static_cast<uint64_t>(rhi::ResourceAccessType::Common),0,
            static_cast<uint64_t>(rhi::ResourceSyncState::All),false}});
        initial[1].regions.push_back({{}, {static_cast<uint64_t>(rhi::ResourceAccessType::CopyDest),0,
            static_cast<uint64_t>(rhi::ResourceSyncState::Copy),true}});
        BackingStateAdmissionLedger stateLedger;
        auto barrierPlan = stateLedger.Prepare(*graph, initial);
        CHECK(barrierPlan.batches.size() == 1);
        CHECK(barrierPlan.batches[0].beforePass.size() == 2);
        // A fresh COMMON buffer has no prior access to synchronize. The
        // clear-to-copy dependency must still have an intra-batch barrier.
        CHECK(barrierPlan.batches[0].beforePass[0].buffers.empty());
        CHECK(!barrierPlan.batches[0].beforePass[1].buffers.empty());
        std::vector<OwnedRecordingList> recordings(1);
        for (auto& recording : recordings) {
            recording.bindings = bindings;

        }
        recordings[0].passes.push_back(org::PreparedPass::Make(iteration,
            +[](const uint32_t& value, org::RecordingContext& context) {
                const auto shader = context.Resolve(org::PreparedDescriptorReference{0});
                context.Commands().SetDescriptorHeaps(shader.heap, {});
                context.Commands().ClearUavUint({shader,context.Resolve(org::PreparedDescriptorReference{1}),
                    context.Resolve(org::PreparedResourceReference{0})}, rhi::UavClearUint(std::array<uint32_t,4>{value,value,value,value}));
            }));
        const auto step = std::find_if(graph->states.steps.begin(),graph->states.steps.end(),
            [](const auto& s) { return s.resource == 0 && s.batch == 0 && s.pass == 1; });
        CHECK(step != graph->states.steps.end() && step->previousBatch == 0);
        recordings[0].passes.push_back(org::PreparedPass::Make(uint8_t{},
            +[](const uint8_t&, org::RecordingContext& context) {
                auto source = context.Resolve(org::PreparedResourceReference{0});
                context.Commands().CopyBufferRegion(context.Resolve(org::PreparedResourceReference{1}).GetHandle(),0,source.GetHandle(),0,4096);
            }));
        recordings[0].barriersBeforePass = barrierPlan.batches[0].beforePass;
        auto timingProbe = std::make_shared<RecordingStatisticsProbe>();
        auto timing = std::make_shared<OwnedRecordingStatistics>();
        timing->service = timingProbe;
        timing->passIndices = {0, 1};
        recordings[0].statistics = timing;
        auto recordingJob = std::async(std::launch::async,
            [recordings = std::move(recordings), runtime, bundle, device, commandPool]() mutable {
                auto layout = std::make_shared<GraphExecutionLayout>(); layout->bundle = bundle;
                auto snapshot = std::make_shared<RenderFrameSnapshot>(); snapshot->layout = layout;
                snapshot->leases.push_back(runtime);
                PlannedFrame plan; plan.snapshot = snapshot;
                plan.timelines = {{1,runtime->timeline.Get().GetHandle()}};
                plan.incomingWaits.resize(1);
                FrameRecordingJob job; job.device = device;
                job.queue = device.GetQueue(rhi::QueueKind::Graphics); job.pool = commandPool;
                job.recording = std::move(recordings[0]);
                plan.jobs.push_back(std::move(job));
                return RecordFrame(std::move(plan), {}, 1);
            });
        std::optional<RecordedFrame> recorded(recordingJob.get());
        CHECK(timingProbe->begins == 2 && timingProbe->ends == 2 && timingProbe->resolves == 1);
        CHECK(timingProbe->cpuSamples == 0 && timingProbe->merges == 0);
        std::weak_ptr<const org::FrozenExecutionBindings> bindingLease = bindings;
        gpu.reset(); cpu.reset(); backing.reset(); bindings.reset();
        auto execution = std::move(*recorded).Submit(admission);
        timing->Publish();
        CHECK(timingProbe->cpuSamples == 2 && timingProbe->merges == 1);
        bool duplicateRejected = false;
        try { std::move(*recorded).Submit(admission); }
        catch (const std::logic_error&) { duplicateRejected = true; }
        CHECK(duplicateRejected);
        recorded.reset(); execution.reset(); // Admission alone retains GPU recording ownership.
        CHECK(!bindingLease.expired());
        CHECK(commandPool->GetDiagnostics().checkedOutCount == 1);
        CHECK(!gpuPool.Assemble(0,gpuContents) && !cpuPool.Assemble(0,cpuContents));
        CHECK(runtime->timeline.Get().HostWait(iteration,10000) == rhi::Result::Ok);
        void* mapped = nullptr; CHECK(SUCCEEDED(readbackResource->Map(0,nullptr,&mapped)));
        for (size_t i = 0; i != 1024; ++i) CHECK(static_cast<const uint32_t*>(mapped)[i] == iteration);
        readbackResource->Unmap(0,nullptr);
        CHECK(admission.RetireCompleted(std::vector<ExecutionTimelinePoint>{{1,iteration}}) == 1);
        CHECK(bindingLease.expired());
        CHECK(commandPool->GetDiagnostics().checkedOutCount == 0);
        CHECK(commandPool->GetDiagnostics().totalOwnedCount == 1);
        CHECK(gpuPool.Assemble(0,gpuContents) && cpuPool.Assemble(0,cpuContents));
    }
    return 0;
}

int TestAliasHeapOwnership(const rhi::DeviceCreateInfo& create) {
    struct Runtime {
        rhi::DevicePtr device;
        rhi::ma::Allocator* allocator = nullptr;
        ~Runtime() { if (allocator) allocator->ReleaseThis(); }
    };
    auto runtime = std::make_shared<Runtime>();
    CHECK(rhi::CreateD3D12Device(create,runtime->device) == rhi::Result::Ok);
    rhi::ma::AllocatorDesc allocatorDesc{}; allocatorDesc.device = runtime->device.Get();
    CHECK(rhi::ma::CreateAllocator(&allocatorDesc,&runtime->allocator) == rhi::Result::Ok);
    auto desc = rhi::helpers::ResourceDesc::Buffer(4096);
    rhi::ResourceAllocationInfo info{};
    runtime->device.Get().GetResourceAllocationInfo(&desc,1,&info);
    rhi::ma::AllocationDesc allocationDesc{};
    allocationDesc.heapType = rhi::HeapType::DeviceLocal;
    allocationDesc.flags = rhi::ma::AllocationFlagCanAlias;
    rhi::ma::AllocationPtr allocation;
    CHECK(runtime->allocator->AllocateMemory(allocationDesc,info,allocation) == rhi::Result::Ok);
    auto pool = org::TrackedHandle::FromAllocation(std::move(allocation),{});
    pool.RetainLifetimeOwner(runtime);
    auto generation = org::AliasHeapGeneration::Capture(pool,1);
    auto* originalAllocation = &generation->Allocation();
    std::weak_ptr<const org::AliasHeapGeneration> oldGeneration = generation;
    org::TrackedHandle placed[2];
    // Two placed resources share exactly the same bytes; ownership acquisition
    // does not reserve a range or serialize CPU recording/compilation.
    for (auto& resource : placed) {
        rhi::ResourcePtr native;
        CHECK(runtime->allocator->CreateAliasingResource(&generation->Allocation(),0,&desc,0,nullptr,native) == rhi::Result::Ok);
        resource = org::TrackedHandle::FromResource(std::move(native),{});
        resource.RetainLifetimeOwner(generation);
    }
    bool disarmRejected = false;
    try { placed[0].ReleaseResourceDisarm(); } catch (const std::logic_error&) { disarmRejected = true; }
    CHECK(disarmRejected);
    // Replacing the pool's logical handle must not replace a captured generation.
    CHECK(runtime->allocator->AllocateMemory(allocationDesc,info,allocation) == rhi::Result::Ok);
    auto replacement = org::TrackedHandle::FromAllocation(std::move(allocation),{});
    replacement.RetainLifetimeOwner(runtime);
    pool = std::move(replacement);
    auto newer = org::AliasHeapGeneration::Capture(pool,2);
    CHECK(newer != generation && newer->Generation() == 2 && &generation->Allocation() == originalAllocation);
    auto recordingLease = placed[0].CaptureAllocationLease();
    placed[0].Reset();
    generation.reset();
    CHECK(!oldGeneration.expired());
    // Move assignment and deferred deletion keep resource/heap lifetimes paired.
    org::TrackedHandle deferred;
    deferred = std::move(placed[1]);
    deferred.Reset();
    CHECK(!oldGeneration.expired());
    recordingLease.reset();
    CHECK(oldGeneration.expired());
    std::weak_ptr<Runtime> runtimeLease = runtime;
    pool.Reset(); runtime.reset();
    CHECK(!runtimeLease.expired());
    newer.reset();
    CHECK(runtimeLease.expired());
    return 0;
}

int main() {
    if (const auto failure = TestServiceWorkReservations()) return failure;
    {
        // A work graph may be the only retained program in a pass.
        bool destroyed = false;
        rhi::WorkGraphVTable table{};
        rhi::WorkGraph graph(rhi::WorkGraphHandle{1, 1});
        graph.impl = &destroyed;
        graph.vt = &table;
        auto owner = std::make_shared<rhi::WorkGraphPtr>(rhi::Device{}, graph,
            +[](rhi::Device&, rhi::WorkGraph& value) noexcept { *static_cast<bool*>(value.impl) = true; });
        org::PreparedDependencyCollector collector;
        const auto reference = collector.CaptureWorkGraph(owner);
        auto dependencies = std::move(collector).Freeze();
        owner.reset();
        CHECK(dependencies && !dependencies->empty() && !destroyed);
        CHECK(dependencies->Resolve(reference).index == 1);
        dependencies.reset();
        CHECK(destroyed);
    }
	if (const auto failure = TestOwnedRenderFrameSnapshot()) return failure;
	Microsoft::WRL::ComPtr<IDXGIFactory6> factory; Microsoft::WRL::ComPtr<IDXGIAdapter> warp;
	CHECK(SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))));
	CHECK(SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))));
	rhi::DeviceCreateInfo create{}; create.backend = rhi::Backend::D3D12; create.nativeAdapter = warp.Get(); create.framesInFlight = 2;
    if (const auto failure = TestDescriptorSnapshots(create)) return failure;
    if (const auto failure = TestOwnedDescriptorGpuExecution(create)) return failure;
    if (const auto failure = TestDelayedFrameRecording(create)) return failure;
    if (const auto failure = TestAliasHeapOwnership(create)) return failure;
	rhi::DevicePtr device; CHECK(!rhi::Failed(rhi::CreateD3D12Device(create, device)) && device);
	auto* nativeDevice = rhi::dx12::get_device(device.Get()); CHECK(nativeDevice);
	org::runtime::InitializeRuntimeDevice(device.Get());

    {
        auto api = device.Get();
        auto heap = std::make_shared<org::DescriptorHeap>(api,
            rhi::DescriptorHeapType::CbvSrvUav, 2, true, "Queued descriptor ownership test");
        const auto first = heap->AllocateDescriptor();
        auto frame = heap->CaptureDescriptorLease(first);
        auto secondFrame = heap->CaptureDescriptorLease(first);
        CHECK(frame == secondFrame);
        // The old submitted fences can complete while an accepted CPU frame
        // still references the slot. It must not become available for reuse.
        heap->ReleaseDescriptor(first);
        const auto replacement = heap->AllocateDescriptor();
        CHECK(replacement != first);
        bool exhausted = false;
        try { (void)heap->AllocateDescriptor(); } catch (const std::runtime_error&) { exhausted = true; }
        CHECK(exhausted);
        frame.reset();
        CHECK(!std::weak_ptr<const void>(secondFrame).expired());
        secondFrame.reset();
        CHECK(heap->AllocateDescriptor() == first);
        heap->ReleaseDescriptor(first);
        heap->ReleaseDescriptor(replacement);
        for (unsigned i = 0; i < 128; ++i) {
            const auto slot = heap->AllocateDescriptor();
            auto queued = heap->CaptureDescriptorLease(slot);
            heap->ReleaseDescriptor(slot);
            queued.reset();
        }
    }
	if (const auto failure = TestPreparedGpuSubmission(device.Get(), nativeDevice)) return failure;
	if (const auto failure = TestEmptyResolverDeclarations(device.Get())) return failure;

	auto bufferA = MakeResource(nativeDevice, false, 4096), bufferB = MakeResource(nativeDevice, false, 4096);
    {
        rhi::ResourcePtr imported;
        CHECK(rhi::dx12::import_resource(device.Get(), bufferA.Get(), imported) == rhi::Result::Ok);
        auto tracked = org::TrackedHandle::FromResource(std::move(imported), {});
        auto handle = tracked.GetResource().GetHandle();
        auto lease = tracked.CaptureAllocationLease();
        auto clone = tracked.CaptureAllocationLease();
        CHECK(lease && lease == clone);
        CHECK(tracked.GetResource().GetHandle().index == handle.index
            && tracked.GetResource().GetHandle().generation == handle.generation);
        bool rejected = false;
        try { tracked.ReleaseResourceDisarm(); } catch (const std::logic_error&) { rejected = true; }
        CHECK(rejected);
        auto moved = std::move(tracked);
        CHECK(!tracked && moved);
        std::weak_ptr<const org::TrackedHandle> lifetime = lease;
        moved.Reset();
        CHECK(!moved && !lifetime.expired());
        clone.reset(); lease.reset();
        CHECK(lifetime.expired());
    }
	rhi::ResourcePtr importedA, importedB;
	CHECK(!rhi::Failed(rhi::dx12::import_resource(device.Get(), bufferA.Get(), importedA)));
	CHECK(!rhi::Failed(rhi::dx12::import_resource(device.Get(), bufferB.Get(), importedB)));
	org::ExternalBufferResource::ViewRequirements bufferViews{};
	auto buffer = org::ExternalBufferResource::CreateShared(std::move(importedA), 4096, bufferViews);
	CHECK(buffer && buffer->RefreshShared(std::move(importedB), 4096, bufferViews));
	rhi::ResourcePtr wrongSize; CHECK(!rhi::Failed(rhi::dx12::import_resource(device.Get(), bufferA.Get(), wrongSize)));
	CHECK(!buffer->RefreshShared(std::move(wrongSize), 8192, bufferViews));
	{
		org::ResourceRegistry registry;
		rhi::ResourcePtr temporaryImport;
		CHECK(!rhi::Failed(rhi::dx12::import_resource(device.Get(), bufferA.Get(), temporaryImport)));
		auto temporary = org::ExternalBufferResource::CreateShared(std::move(temporaryImport), 4096, bufferViews);
		const org::ResourceIdentifier expiredID(std::to_string(temporary->GetGlobalResourceID()));
		registry.RegisterAnonymousWeak(temporary);
		CHECK(registry.MakeHandle(expiredID).GetGeneration() != 0);
		temporary.reset();
		CHECK(registry.MakeHandle(expiredID).GetGeneration() == 0);
		registry.ReclaimExpiredAnonymous();
		registry.RegisterAnonymous(buffer); // reuses the expired slot
		CHECK(registry.MakeHandle(expiredID).GetGeneration() == 0);
		const org::ResourceIdentifier bufferID(std::to_string(buffer->GetGlobalResourceID()));
		CHECK(registry.MakeHandle(bufferID).GetGlobalResourceID() == buffer->GetGlobalResourceID());
		registry = {};
		CHECK(registry.MakeHandle(bufferID).GetGeneration() == 0);
	}
	if (const auto failure = TestEmptyResolverRefresh(device.Get(), buffer)) return failure;
	if (const auto failure = TestMixedResolverCollision(device.Get(), buffer)) return failure;

	org::TextureDescription description{}; description.imageDimensions = { { 64, 64 } };
	description.channels = 4; description.format = rhi::Format::R8G8B8A8_UNorm; description.hasSRV = true;
	description.initialLayout = rhi::ResourceLayout::Common;
	auto textureA = MakeResource(nativeDevice, true, 0), textureB = MakeResource(nativeDevice, true, 0);
	rhi::ResourcePtr textureImportA, textureImportB;
	CHECK(!rhi::Failed(rhi::dx12::import_resource(device.Get(), textureA.Get(), textureImportA)));
	CHECK(!rhi::Failed(rhi::dx12::import_resource(device.Get(), textureB.Get(), textureImportB)));
	auto texture = org::ExternalTextureResource::CreateShared(std::move(textureImportA), description, false);
	CHECK(texture);
	const auto oldTextureIndex = texture->GetSRVInfo(0).slot.index;
	auto oldTextureViews = texture->CaptureBindlessViews();
	auto oldTextureDescriptors = texture->CaptureDescriptorOwnership();
	CHECK(oldTextureDescriptors && texture->RefreshShared(std::move(textureImportB), description, false));
	CHECK(texture->GetSRVInfo(0).slot.index != oldTextureIndex);
	CHECK(oldTextureViews && oldTextureViews->Resolve({org::BindlessViewKind::ShaderResource}).index == oldTextureIndex);
	auto newTextureViews = texture->CaptureBindlessViews();
	CHECK(newTextureViews && newTextureViews->Resolve({org::BindlessViewKind::ShaderResource}).index
		== texture->GetSRVInfo(0).slot.index);
	CHECK(newTextureViews->description.texture.width == 64 && newTextureViews->description.texture.height == 64);
	auto changed = description; changed.imageDimensions[0].width = 32;
	rhi::ResourcePtr incompatible; CHECK(!rhi::Failed(rhi::dx12::import_resource(device.Get(), textureA.Get(), incompatible)));
	CHECK(!texture->RefreshShared(std::move(incompatible), changed, false));
	buffer.reset(); texture.reset();
    if (const auto failure = TestSignalReservations(device.Get())) return failure;
    if (const auto failure = TestProgramVersions(device.Get())) return failure;
    if (const auto failure = TestReadbackCaptures(device.Get())) return failure;
    if (const auto failure = TestFrameRetirement(device.Get())) return failure;
	org::runtime::ShutdownRuntimeDevice();
	return 0;
}
