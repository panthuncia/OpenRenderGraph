// Rebinding safety on the GPU, for any backend: copy passes only.
//
// Three passes: "write" copies this frame's value into the buffer `writeTarget` points at; "fill" copies a constant into
// buffer Y; "read" copies the buffer `readSource` points at into a readback buffer, on the copy queue when there is one.
// `writeTarget` and `readSource` are DynamicResources, so pointing them at buffers rebinds their slots:
// - both at X: the slots share a backing, so read waits for write and reads this frame's value;
// - both moved to X2 together: the same executable, still this frame's value;
// - read moved alone to Y: a split, a new schedule, read sees the constant;
// - read moved back onto X2: a merge, a new schedule, this frame's value again.
// The same swaps with SharingChangePolicy::Deferred: the frame of the swap still reads the old backing, and a worker build
// brings the new one in later. A frame never reads a stale value (last frame's), which a missing dependency would give.
// Also: a pass whose new declarations add resources builds inline, unless it says PreparesOnlyDeclaredWork, when a
// worker builds them and frames keep preparing against the old declaration meanwhile.

#include "OpenRenderGraph/GraphDesc.h"
#include "OpenRenderGraph/GraphHost.h"
#include "OpenRenderGraph/GraphInstance.h"
#include "OpenRenderGraph/Runtime.h"
#include "Render/RenderGraph/PersistentGraph.h"
#include "Render/RenderGraph/RenderGraph.h"
#include "Render/RenderGraph/ScheduleTrace.h"
#include "Render/PassBuilders.h"
#include "RenderPasses/Base/TypedRenderGraphPass.h"
#include "Resources/Buffers/Buffer.h"
#include "Resources/DynamicResource.h"
#include "Render/OwnedDescriptorBinding.h"
#include "Render/PublicationBindingBundle.h"
#include "Managers/Singletons/DescriptorHeapManager.h"
#include "Interfaces/IDynamicDeclaredResources.h"
#include "TestSupport.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kCount = 64;
constexpr uint64_t kBytes = kCount * sizeof(uint32_t);
constexpr uint32_t kFill = 0xB0B0B0B0u;

struct CopyBindings { org::ResourceBindingToken source, destination; };
class CopyPass final : public org::TypedRenderGraphPass<CopyPass, org::EmptyPassFrameData, CopyBindings> {
public:
    CopyPass(std::shared_ptr<org::Resource> source, std::shared_ptr<org::Resource> destination, org::QueueKind queue)
        : m_source(std::move(source)), m_destination(std::move(destination)), m_queue(queue) {}
    CopyBindings Declare(org::PassBuilder& builder) {
        builder.PreferQueue(m_queue);
        return {builder.CopySource(m_source), builder.CopyDestination(m_destination)};
    }
    static void Record(const CopyBindings& bindings, org::PassRecordContext& recording) {
        recording.Commands().CopyBufferRegion(recording.Resolve(bindings.destination).GetHandle(), 0,
            recording.Resolve(bindings.source).GetHandle(), 0, kBytes);
    }
private:
    std::shared_ptr<org::Resource> m_source, m_destination;
    org::QueueKind m_queue;
};

struct Buffers {
    std::shared_ptr<org::Buffer> value, fill, x, x2, y, readback;
    std::shared_ptr<org::DynamicResource> writeTarget, readSource;
};

class SharingExtension final : public org::RenderGraph::IRenderGraphExtension {
public:
    explicit SharingExtension(const Buffers& buffers) : m_buffers(buffers) {}
    void GatherStructuralPasses(org::RenderGraph&, std::vector<org::RenderGraph::ExternalPassDesc>& out) override {
        const auto copy = [&](const char* name, std::shared_ptr<org::Resource> source, std::shared_ptr<org::Resource> destination,
            org::QueueKind queue) {
            out.push_back(org::RenderGraph::ExternalPassDesc::Copy(name,
                std::make_shared<CopyPass>(std::move(source), std::move(destination), queue)).PreferQueue(queue));
        };
        copy("share.write", m_buffers.value, m_buffers.writeTarget, org::QueueKind::Graphics);
        copy("share.fill", m_buffers.fill, m_buffers.y, org::QueueKind::Graphics);
        copy("share.read", m_buffers.readSource, m_buffers.readback, org::QueueKind::Copy);
    }
private:
    Buffers m_buffers;
};

bool Write(org::Buffer& buffer, uint32_t value) {
    void* mapped = nullptr;
    buffer.GetAPIResource().Map(&mapped, 0, kBytes);
    if (!mapped) return false;
    auto* words = static_cast<uint32_t*>(mapped);
    for (uint32_t i = 0; i < kCount; ++i) words[i] = value;
    buffer.GetAPIResource().Unmap(0, kBytes);
    return true;
}
// The value every element of the readback holds, or 0 when they differ.
uint32_t Read(org::Buffer& buffer) {
    void* mapped = nullptr;
    buffer.GetAPIResource().Map(&mapped, 0, kBytes);
    if (!mapped) return 0;
    const auto* words = static_cast<const uint32_t*>(mapped);
    uint32_t value = words[0];
    for (uint32_t i = 1; i < kCount; ++i) if (words[i] != value) value = 0;
    buffer.GetAPIResource().Unmap(0, 0);
    return value;
}

struct Run {
    rhi::Device device;
    org::GraphHost& host;
    Buffers& buffers;
    org::RenderGraph::SharingChangePolicy policy;
    uint32_t frame = 0;
    // One frame: the write pass writes a new value. Returns what read copied, and whether the executable changed.
    struct Result { uint32_t value = 0, read = 0; bool rebuilt = false; };
    Result Frame() {
        Result result;
        result.value = 0x1000u + ++frame;
        ENSURE(Write(*buffers.value, result.value));
        const auto before = Revision();
        host.ExecuteFrame(nullptr, [&](org::RenderGraph& graph) {
            graph.SetSharingChangePolicy(policy);
            graph.SetScheduleTraceEnabled(true);
        });
        ENSURE(device.WaitIdle() == rhi::Result::Ok);
        result.read = Read(*buffers.readback);
        result.rebuilt = Revision() != before;
        return result;
    }
    uint64_t Revision() const {
        auto* graph = const_cast<org::GraphHost&>(host).Graph();
        const auto snapshot = graph ? graph->GetPersistentScheduleSnapshot() : nullptr;
        return snapshot && !snapshot->executables.empty() ? snapshot->executables.front()->structuralRevision : 0;
    }
};

// A descriptor that shaders read without the graph knowing (a bindless table's entry) may only name an externally
// immutable resource. Needs live descriptor heaps (a graph).
// withViews: capture a real view of an externally immutable buffer too (Vulkan run; the D3D12 run checks the refusals).
int TestUndeclaredReads(bool withViews) {
    auto buffer = withViews ? org::Buffer::CreateUnmaterializedStructuredBuffer(kCount, sizeof(uint32_t), false)
        : org::Buffer::CreateShared(rhi::HeapType::DeviceLocal, kBytes);
    buffer->SetName("share.bindless");
    if (withViews) buffer->Materialize();
    auto& descriptors = org::DescriptorHeapManager::GetInstance();
    const auto capture = [&] {
        return org::OwnedDescriptorBinding::CaptureUndeclaredView(org::PublicationBindingBundle::Capture(*buffer),
            org::SrvView{}, descriptors.DeviceGeneration(), descriptors.GetResourceCleanupQueue());
    };
    bool rejected = false;
    try { capture(); } catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected);
    rejected = false;
    try { org::RequireUndeclaredShaderRead(*buffer); } catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected);
    buffer->SetGraphOwnership(org::Resource::GraphOwnership::ExternalImmutableShaderResource);
    org::RequireUndeclaredShaderRead(*buffer);
    if (withViews) {
        const auto binding = capture();
        CHECK(binding && binding.Index() != UINT32_MAX);
    }
    return 0;
}

int RunScenario(rhi::Device device, const std::shared_ptr<org::Runtime>& runtime, org::RenderGraph::SharingChangePolicy policy) {
    using Policy = org::RenderGraph::SharingChangePolicy;
    Buffers buffers;
    buffers.value = org::Buffer::CreateShared(rhi::HeapType::Upload, kBytes);
    buffers.fill = org::Buffer::CreateShared(rhi::HeapType::Upload, kBytes);
    buffers.x = org::Buffer::CreateShared(rhi::HeapType::DeviceLocal, kBytes);
    buffers.x2 = org::Buffer::CreateShared(rhi::HeapType::DeviceLocal, kBytes);
    buffers.y = org::Buffer::CreateShared(rhi::HeapType::DeviceLocal, kBytes);
    buffers.readback = org::Buffer::CreateShared(rhi::HeapType::Readback, kBytes);
    buffers.value->SetName("share.value"); buffers.fill->SetName("share.fill-source");
    buffers.x->SetName("share.x"); buffers.x2->SetName("share.x2"); buffers.y->SetName("share.y");
    buffers.readback->SetName("share.readback");
    CHECK(Write(*buffers.fill, kFill));
    buffers.writeTarget = std::make_shared<org::DynamicResource>(buffers.x);
    buffers.readSource = std::make_shared<org::DynamicResource>(buffers.x);
    buffers.writeTarget->SetName("share.write-target");
    buffers.readSource->SetName("share.read-source");
    {
        org::GraphHost host({.runtime = runtime});
        host.AddExtension("test.sharing", [&] { return std::make_unique<SharingExtension>(buffers); });
        Run run{device, host, buffers, policy};
        // Shared from the start: read waits for write every frame.
        for (int i = 0; i < 4; ++i) {
            const auto frame = run.Frame();
            CHECK(frame.read == frame.value);
        }
        if (policy == Policy::Inline) {
            if (const int line = TestUndeclaredReads(runtime->Backend() == rhi::Backend::Vulkan)) return line;
        }
        // Both slots to X2 together: binding-only.
        buffers.writeTarget->SetResource(buffers.x2);
        buffers.readSource->SetResource(buffers.x2);
        {
            const auto frame = run.Frame();
            CHECK(frame.read == frame.value && !frame.rebuilt);
        }
        const auto settle = [&](uint32_t expectNew, bool newIsValue, uint32_t expectOld, bool oldIsValue) -> int {
            // Inline: the first frame already reads the new backing, from a new schedule. Deferred: the first frames read
            // the old one, with the same executable, until a worker build installs the new one. Never anything else.
            bool switched = false;
            for (int i = 0; i < 30; ++i) {
                const auto frame = run.Frame();
                const auto newValue = newIsValue ? frame.value : expectNew;
                const auto oldValue = oldIsValue ? frame.value : expectOld;
                if (i == 0 && policy == Policy::Inline) CHECK(frame.read == newValue && frame.rebuilt);
                if (i == 0 && policy == Policy::Deferred) CHECK(frame.read == oldValue && !frame.rebuilt);
                if (frame.read == newValue) switched = true;
                else CHECK(!switched && frame.read == oldValue);
                if (switched && i >= 3) break;
            }
            CHECK(switched);
            return 0;
        };
        // Read alone to Y: a split. It then reads the constant fill.
        buffers.readSource->SetResource(buffers.y);
        if (const int line = settle(kFill, false, 0, true)) return line;
        // Read back onto X2: a merge. It then waits for write again.
        buffers.readSource->SetResource(buffers.x2);
        if (const int line = settle(0, true, kFill, false)) return line;
        // Steady after the swaps: nothing rebuilds.
        for (int i = 0; i < 4; ++i) {
            const auto frame = run.Frame();
            CHECK(frame.read == frame.value && !frame.rebuilt);
        }
        // Two swaps in a row, back to where it started: away to Y, then onto X2 again the next frame, before a deferred
        // build of the first swap has installed. The second swap supersedes the first. Every frame reads what one of the
        // two bindings gives (this frame's value, or the fill), never last frame's value, and it ends on this frame's.
        buffers.readSource->SetResource(buffers.y);
        {
            const auto frame = run.Frame();
            CHECK(frame.read == (policy == Policy::Inline ? kFill : frame.value));
        }
        buffers.readSource->SetResource(buffers.x2);
        int current = 0;
        for (int i = 0; i < 30 && current < 4; ++i) {
            const auto frame = run.Frame();
            CHECK(frame.read == frame.value || frame.read == kFill);
            current = frame.read == frame.value ? current + 1 : 0;
        }
        CHECK(current >= 4);
    }
    CHECK(device.WaitIdle() == rhi::Result::Ok);
    return 0;
}
// A pass whose work comes and goes, each job with buffers of its own: a new job declares resources the live executable
// lacks. The pass snapshots its jobs in Declare and takes work only from the snapshot its bindings carry.
// With PreparesOnlyDeclaredWork the graph builds the new declaration on a worker; without it, inline.
struct Job {
    std::shared_ptr<org::Buffer> source, destination;
    uint32_t value = 0;
    int taken = 0; // Executions that copied it.
};
struct JobBindings {
    struct Copy { org::ResourceBindingToken source, destination; };
    std::vector<Copy> copies;
    std::shared_ptr<const std::vector<std::shared_ptr<Job>>> jobs; // The declaration's snapshot, in copy order.
};
struct JobFrame { std::vector<uint32_t> copies; };
class JobPass final : public org::TypedRenderGraphPass<JobPass, JobFrame, JobBindings>, public org::IDynamicDeclaredResources {
public:
    // No value: the interface's default.
    explicit JobPass(std::optional<bool> declaredOnly) : m_declaredOnly(declaredOnly) {}
    void Enqueue(std::shared_ptr<Job> job) { m_queued.push_back(std::move(job)); m_changed = true; }
    JobBindings Declare(org::PassBuilder& builder) {
        builder.PreferQueue(org::QueueKind::Graphics);
        JobBindings bindings;
        auto jobs = std::make_shared<std::vector<std::shared_ptr<Job>>>(m_queued);
        for (const auto& job : *jobs)
            bindings.copies.push_back({builder.CopySource(job->source), builder.CopyDestination(job->destination)});
        bindings.jobs = std::move(jobs);
        m_changed = false;
        return bindings;
    }
    JobFrame Prepare(const JobBindings& bindings, const org::PassPrepareContext&) {
        JobFrame frame;
        for (uint32_t index = 0; index < bindings.jobs->size(); ++index) {
            const auto& job = (*bindings.jobs)[index];
            const auto queued = std::find(m_queued.begin(), m_queued.end(), job);
            if (queued == m_queued.end()) continue; // An earlier execution took it.
            m_queued.erase(queued);
            ++job->taken;
            frame.copies.push_back(index);
            m_changed = true; // Redeclare without it.
        }
        return frame;
    }
    static void Record(const JobBindings& bindings, const JobFrame& frame, org::PassRecordContext& recording) {
        for (const auto index : frame.copies)
            recording.Commands().CopyBufferRegion(recording.Resolve(bindings.copies[index].destination).GetHandle(), 0,
                recording.Resolve(bindings.copies[index].source).GetHandle(), 0, kBytes);
    }
    bool DeclaredResourcesChanged() const override { return m_changed; }
    bool PreparesOnlyDeclaredWork() const noexcept override {
        return m_declaredOnly ? *m_declaredOnly : org::IDynamicDeclaredResources::PreparesOnlyDeclaredWork();
    }
private:
    std::optional<bool> m_declaredOnly;
    bool m_changed = true;
    std::vector<std::shared_ptr<Job>> m_queued;
};

class JobExtension final : public org::RenderGraph::IRenderGraphExtension {
public:
    explicit JobExtension(std::shared_ptr<JobPass> pass) : m_pass(std::move(pass)) {}
    void GatherStructuralPasses(org::RenderGraph&, std::vector<org::RenderGraph::ExternalPassDesc>& out) override {
        out.push_back(org::RenderGraph::ExternalPassDesc::Copy("relower.jobs", m_pass).PreferQueue(org::QueueKind::Graphics));
    }
private:
    std::shared_ptr<JobPass> m_pass;
};

// declaredOnly: the pass's PreparesOnlyDeclaredWork; no value tests the default, which is true.
int RunRelowerScenario(rhi::Device device, const std::shared_ptr<org::Runtime>& runtime, std::optional<bool> declaredOnly_) {
    auto pass = std::make_shared<JobPass>(declaredOnly_);
    const bool declaredOnly = declaredOnly_.value_or(true);
    uint32_t next = 0x5000u;
    const auto makeJob = [&] {
        auto job = std::make_shared<Job>();
        job->value = ++next;
        job->source = org::Buffer::CreateShared(rhi::HeapType::Upload, kBytes);
        job->destination = org::Buffer::CreateShared(rhi::HeapType::Readback, kBytes);
        job->source->SetName("relower.source"); job->destination->SetName("relower.destination");
        ENSURE(Write(*job->source, job->value));
        return job;
    };
    {
        org::GraphHost host({.runtime = runtime});
        host.AddExtension("test.relower", [&] { return std::make_unique<JobExtension>(pass); });
        Buffers unused;
        Run run{device, host, unused, org::RenderGraph::SharingChangePolicy::Inline};
        const auto frame = [&] {
            const auto before = run.Revision();
            host.ExecuteFrame(nullptr, [](org::RenderGraph& graph) { graph.SetScheduleTraceEnabled(true); });
            ENSURE(device.WaitIdle() == rhi::Result::Ok);
            return run.Revision() != before;
        };
        // Copied exactly once, with its own value: never twice, never from a declaration that lacked its buffers.
        const auto done = [](const Job& job) { return job.taken == 1 && Read(*job.destination) == job.value; };
        for (int i = 0; i < 3; ++i) frame();
        // One job. Inline: the frame it arrives in rebuilds and copies it. Worker: that frame keeps the old executable and
        // copies nothing; a later frame installs the new one and copies it.
        auto first = makeJob();
        pass->Enqueue(first);
        const bool rebuilt = frame();
        if (declaredOnly) CHECK(!rebuilt && first->taken == 0);
        else CHECK(rebuilt && done(*first));
        for (int i = 0; i < 30 && first->taken == 0; ++i) frame();
        CHECK(done(*first));
        // Jobs that arrive while a build is pending, one per frame: each is copied once, after a declaration with it installs.
        std::vector<std::shared_ptr<Job>> jobs;
        for (int i = 0; i < 4; ++i) {
            jobs.push_back(makeJob());
            pass->Enqueue(jobs.back());
            frame();
            for (const auto& job : jobs) CHECK(job->taken <= 1);
        }
        for (int i = 0; i < 40; ++i) {
            if (std::all_of(jobs.begin(), jobs.end(), [](const auto& job) { return job->taken == 1; })) break;
            frame();
        }
        for (const auto& job : jobs) CHECK(done(*job));
        // Idle again: the declaration without the jobs installs, then nothing rebuilds.
        for (int i = 0; i < 30; ++i) frame();
        for (int i = 0; i < 4; ++i) CHECK(!frame());
        CHECK(first->taken == 1);
        for (const auto& job : jobs) CHECK(job->taken == 1);
    }
    CHECK(device.WaitIdle() == rhi::Result::Ok);
    return 0;
}
// A pass that reports a changed declaration every frame and declares the same copy each time. Each Declare makes a new
// layout object with the same content, which must build nothing; the new bindings' tokens must still resolve.
class RedeclarePass final : public org::TypedRenderGraphPass<RedeclarePass, org::EmptyPassFrameData, CopyBindings>,
    public org::IDynamicDeclaredResources {
public:
    std::shared_ptr<org::Resource> source, destination;
    int declarations = 0;
    CopyBindings Declare(org::PassBuilder& builder) {
        builder.PreferQueue(org::QueueKind::Graphics);
        ++declarations;
        return {builder.CopySource(source), builder.CopyDestination(destination)};
    }
    static void Record(const CopyBindings& bindings, org::PassRecordContext& recording) {
        CopyPass::Record(bindings, recording);
    }
    bool DeclaredResourcesChanged() const override { return true; }
};

class RedeclareExtension final : public org::RenderGraph::IRenderGraphExtension {
public:
    explicit RedeclareExtension(std::shared_ptr<RedeclarePass> pass) : m_pass(std::move(pass)) {}
    void GatherStructuralPasses(org::RenderGraph&, std::vector<org::RenderGraph::ExternalPassDesc>& out) override {
        out.push_back(org::RenderGraph::ExternalPassDesc::Copy("redeclare.copy", m_pass).PreferQueue(org::QueueKind::Graphics));
    }
private:
    std::shared_ptr<RedeclarePass> m_pass;
};

int RunRedeclareScenario(rhi::Device device, const std::shared_ptr<org::Runtime>& runtime) {
    auto pass = std::make_shared<RedeclarePass>();
    auto source = org::Buffer::CreateShared(rhi::HeapType::Upload, kBytes);
    auto first = org::Buffer::CreateShared(rhi::HeapType::Readback, kBytes);
    auto second = org::Buffer::CreateShared(rhi::HeapType::Readback, kBytes);
    source->SetName("redeclare.source"); first->SetName("redeclare.first"); second->SetName("redeclare.second");
    pass->source = source;
    pass->destination = first;
    {
        org::GraphHost host({.runtime = runtime});
        host.AddExtension("test.redeclare", [&] { return std::make_unique<RedeclareExtension>(pass); });
        Buffers unused;
        Run run{device, host, unused, org::RenderGraph::SharingChangePolicy::Inline};
        uint32_t value = 0x7000u;
        const auto frame = [&](org::Buffer& destination) {
            ++value;
            ENSURE(Write(*source, value));
            const auto before = run.Revision();
            host.ExecuteFrame(nullptr, [](org::RenderGraph& graph) { graph.SetScheduleTraceEnabled(true); });
            ENSURE(device.WaitIdle() == rhi::Result::Ok);
            return std::pair{run.Revision() != before, Read(destination) == value};
        };
        for (int i = 0; i < 3; ++i) frame(*first);
        // The same declaration every frame: redeclared, never rebuilt, and the copy lands each time.
        const int declaredBefore = pass->declarations;
        for (int i = 0; i < 20; ++i) {
            const auto [rebuilt, copied] = frame(*first);
            CHECK(!rebuilt && copied);
        }
        CHECK(pass->declarations >= declaredBefore + 20);
        // Another destination of the same shape: a new declaration. It builds on a worker (PreparesOnlyDeclaredWork by
        // default), so the frames until it installs still copy into the first; then every frame copies into the second.
        pass->destination = second;
        {
            const auto [rebuilt, copied] = frame(*first);
            CHECK(!rebuilt && copied);
        }
        bool switched = false;
        for (int i = 0; i < 30 && !switched; ++i) {
            const auto [rebuilt, copied] = frame(*second);
            if (copied) switched = true;
            else CHECK(Read(*first) == value);
        }
        CHECK(switched);
        for (int i = 0; i < 5; ++i) {
            const auto [rebuilt, copied] = frame(*second);
            CHECK(!rebuilt && copied);
        }
    }
    CHECK(device.WaitIdle() == rhi::Result::Ok);
    return 0;
}
// A pass that RebindsConcreteResources: it copies this frame's value into `target`, a plain buffer it declares by pointer
// and the test swaps. Swapping between buffers nothing else uses is a rebind of the pass's slot: no build, and the
// value lands in the new buffer that frame. Swapping onto a buffer another pass reads changes which slots share a
// backing, so it is promoted to a structural build (or a worker build, Deferred) with the wait; swapping off it splits.
class RebindCopyPass final : public org::TypedRenderGraphPass<RebindCopyPass, org::EmptyPassFrameData, CopyBindings>,
    public org::IDynamicDeclaredResources {
public:
    RebindCopyPass(std::shared_ptr<org::Resource> source, std::shared_ptr<org::Resource> target, bool declaredOnly = false)
        : m_source(std::move(source)), m_target(std::move(target)), m_declaredOnly(declaredOnly) {}
    void SetTarget(std::shared_ptr<org::Resource> target) { m_target = std::move(target); m_changed = true; }
    CopyBindings Declare(org::PassBuilder& builder) {
        builder.PreferQueue(org::QueueKind::Graphics);
        m_changed = false;
        return {builder.CopySource(m_source), builder.CopyDestination(m_target)};
    }
    static void Record(const CopyBindings& bindings, org::PassRecordContext& recording) { CopyPass::Record(bindings, recording); }
    bool DeclaredResourcesChanged() const override { return m_changed; }
    bool RebindsConcreteResources() const noexcept override { return true; }
    bool PreparesOnlyDeclaredWork() const noexcept override { return m_declaredOnly; }
private:
    std::shared_ptr<org::Resource> m_source, m_target;
    bool m_declaredOnly;
    bool m_changed = true;
};

class RebindExtension final : public org::RenderGraph::IRenderGraphExtension {
public:
    RebindExtension(std::shared_ptr<RebindCopyPass> write, std::shared_ptr<org::Resource> shared,
        std::shared_ptr<org::Resource> readback) : m_write(std::move(write)), m_shared(std::move(shared)), m_readback(std::move(readback)) {}
    void GatherStructuralPasses(org::RenderGraph&, std::vector<org::RenderGraph::ExternalPassDesc>& out) override {
        out.push_back(org::RenderGraph::ExternalPassDesc::Copy("rebind.write", m_write).PreferQueue(org::QueueKind::Graphics));
        if (m_shared)
            out.push_back(org::RenderGraph::ExternalPassDesc::Copy("rebind.read",
                std::make_shared<CopyPass>(m_shared, m_readback, org::QueueKind::Copy)).PreferQueue(org::QueueKind::Copy));
    }
private:
    std::shared_ptr<RebindCopyPass> m_write;
    std::shared_ptr<org::Resource> m_shared, m_readback;
};

int RunRebindSlotScenario(rhi::Device device, const std::shared_ptr<org::Runtime>& runtime,
    org::RenderGraph::SharingChangePolicy policy) {
    using Policy = org::RenderGraph::SharingChangePolicy;
    auto value = org::Buffer::CreateShared(rhi::HeapType::Upload, kBytes);
    auto privateA = org::Buffer::CreateShared(rhi::HeapType::Readback, kBytes);
    auto privateB = org::Buffer::CreateShared(rhi::HeapType::Readback, kBytes);
    auto shared = org::Buffer::CreateShared(rhi::HeapType::DeviceLocal, kBytes);
    auto readback = org::Buffer::CreateShared(rhi::HeapType::Readback, kBytes);
    value->SetName("rebind.value"); privateA->SetName("rebind.private-a"); privateB->SetName("rebind.private-b");
    shared->SetName("rebind.shared"); readback->SetName("rebind.readback");
    auto write = std::make_shared<RebindCopyPass>(value, privateA);
    {
        org::GraphHost host({.runtime = runtime});
        host.AddExtension("test.rebind", [&] { return std::make_unique<RebindExtension>(write, shared, readback); });
        Buffers unused;
        Run run{device, host, unused, policy};
        uint32_t current = 0x9000u;
        struct Result { uint32_t value; bool rebuilt; };
        const auto frame = [&] {
            ++current;
            ENSURE(Write(*value, current));
            const auto before = run.Revision();
            host.ExecuteFrame(nullptr, [&](org::RenderGraph& graph) {
                graph.SetSharingChangePolicy(policy);
                graph.SetScheduleTraceEnabled(true);
            });
            ENSURE(device.WaitIdle() == rhi::Result::Ok);
            return Result{current, run.Revision() != before};
        };
        for (int i = 0; i < 4; ++i) {
            const auto result = frame();
            CHECK(Read(*privateA) == result.value);
        }
        // Onto another buffer nothing else uses: a rebind. No build, and the copy lands there that frame.
        write->SetTarget(privateB);
        for (int i = 0; i < 4; ++i) {
            const auto result = frame();
            CHECK(!result.rebuilt && Read(*privateB) == result.value);
        }
        // Onto the buffer the read pass copies from: the two slots now share a backing, so the read must wait for the
        // write. Inline: that frame rebuilds and reads this frame's value. Deferred: frames keep writing privateB until
        // a worker build installs the merge; the read never sees a value from a frame other than its own once merged.
        const auto readBefore = Read(*readback);
        write->SetTarget(shared);
        bool merged = false;
        for (int i = 0; i < 30; ++i) {
            const auto result = frame();
            const bool sees = Read(*readback) == result.value;
            if (i == 0 && policy == Policy::Inline) CHECK(result.rebuilt && sees);
            if (i == 0 && policy == Policy::Deferred) CHECK(!result.rebuilt && Read(*privateB) == result.value);
            if (merged) CHECK(sees);
            if (sees) merged = true;
            else CHECK(Read(*readback) == readBefore && Read(*privateB) == result.value);
            if (merged && i >= 3) break;
        }
        CHECK(merged);
        // Back onto a private buffer: a split. The read pass then copies whatever the shared buffer last held.
        write->SetTarget(privateA);
        bool split = false;
        uint32_t frozen = 0;
        for (int i = 0; i < 30; ++i) {
            const auto result = frame();
            const bool privateGets = Read(*privateA) == result.value;
            if (i == 0 && policy == Policy::Inline) CHECK(result.rebuilt && privateGets);
            if (!privateGets) { CHECK(!split && Read(*readback) == result.value); continue; }
            if (!split) { split = true; frozen = Read(*readback); }
            CHECK(Read(*readback) == frozen && frozen != result.value);
            if (i >= 3) break;
        }
        CHECK(split);
        for (int i = 0; i < 4; ++i) {
            const auto result = frame();
            CHECK(!result.rebuilt && Read(*privateA) == result.value);
        }
    }
    CHECK(device.WaitIdle() == rhi::Result::Ok);
    return 0;
}

// A resolver whose members the test sets. Reserves 4 member slots.
struct ListResolver final : org::ClonableResolver<ListResolver> {
    struct State { std::vector<std::shared_ptr<org::Resource>> members; uint64_t version = 1; };
    std::shared_ptr<State> state = std::make_shared<State>();
    std::vector<std::shared_ptr<org::Resource>> Resolve() const override { return state->members; }
    std::shared_ptr<const org::ResolverDeclarationState> CaptureDeclarationState() const override {
        auto captured = std::make_shared<org::ResolverDeclarationState>();
        captured->tracked = true;
        captured->dependencyIdentity = state;
        captured->resourceSetIdentity = {state->version, 0};
        captured->resources = std::make_shared<const org::ResolverResourceList>(state->members.begin(), state->members.end());
        return captured;
    }
    uint64_t DeclarationVersionHint() const noexcept override { return state->version; }
    uint32_t CapacityHint() const noexcept override { return 4; }
};

// Copies one source into every member of a resolver group, and redeclares whenever a member is added. The new
// declaration names one more member, so its layout has one more use, but the group already has a slot (and the pass a
// binding) for it: RemapDeclaration takes it with no build. A fifth member is past the reserved four: that grows the
// group, a structural build.
struct GroupCopyBindings {
    org::ResourceBindingToken source;
    std::vector<org::ResourceBindingToken> destinations;
};
class GroupCopyPass final : public org::TypedRenderGraphPass<GroupCopyPass, org::EmptyPassFrameData, GroupCopyBindings>,
    public org::IDynamicDeclaredResources {
public:
    GroupCopyPass(std::shared_ptr<org::Resource> source, ListResolver resolver) : m_source(std::move(source)), m_resolver(std::move(resolver)) {}
    void Changed() { m_changed = true; }
    GroupCopyBindings Declare(org::PassBuilder& builder) {
        builder.PreferQueue(org::QueueKind::Graphics);
        m_changed = false;
        return {builder.CopySource(m_source), builder.CopyDestination(m_resolver).resources};
    }
    static void Record(const GroupCopyBindings& bindings, org::PassRecordContext& recording) {
        for (const auto& destination : bindings.destinations)
            recording.Commands().CopyBufferRegion(recording.Resolve(destination).GetHandle(), 0,
                recording.Resolve(bindings.source).GetHandle(), 0, kBytes);
    }
    bool DeclaredResourcesChanged() const override { return m_changed; }
private:
    std::shared_ptr<org::Resource> m_source;
    ListResolver m_resolver;
    bool m_changed = true;
};

class GroupCopyExtension final : public org::RenderGraph::IRenderGraphExtension {
public:
    explicit GroupCopyExtension(std::shared_ptr<GroupCopyPass> pass) : m_pass(std::move(pass)) {}
    void GatherStructuralPasses(org::RenderGraph&, std::vector<org::RenderGraph::ExternalPassDesc>& out) override {
        out.push_back(org::RenderGraph::ExternalPassDesc::Copy("remap.copy", m_pass).PreferQueue(org::QueueKind::Graphics));
    }
private:
    std::shared_ptr<GroupCopyPass> m_pass;
};

int RunRemapScenario(rhi::Device device, const std::shared_ptr<org::Runtime>& runtime) {
    auto source = org::Buffer::CreateShared(rhi::HeapType::Upload, kBytes);
    source->SetName("remap.source");
    ListResolver resolver;
    std::vector<std::shared_ptr<org::Buffer>> members;
    const auto addMember = [&] {
        members.push_back(org::Buffer::CreateShared(rhi::HeapType::Readback, kBytes));
        members.back()->SetName("remap.member");
        resolver.state->members.push_back(members.back());
        ++resolver.state->version;
    };
    addMember();
    auto pass = std::make_shared<GroupCopyPass>(source, resolver);
    {
        org::GraphHost host({.runtime = runtime});
        host.AddExtension("test.remap", [&] { return std::make_unique<GroupCopyExtension>(pass); });
        Buffers unused;
        Run run{device, host, unused, org::RenderGraph::SharingChangePolicy::Inline};
        uint32_t value = 0xA000u;
        const auto frame = [&] {
            ++value;
            ENSURE(Write(*source, value));
            const auto before = run.Revision();
            host.ExecuteFrame(nullptr, [](org::RenderGraph& graph) { graph.SetScheduleTraceEnabled(true); });
            ENSURE(device.WaitIdle() == rhi::Result::Ok);
            return run.Revision() != before;
        };
        const auto allCopied = [&] {
            return std::all_of(members.begin(), members.end(), [&](const auto& member) { return Read(*member) == value; });
        };
        for (int i = 0; i < 3; ++i) { frame(); CHECK(allCopied()); }
        // Members two to four, each with a redeclaration: within the reserved slots, so no build, and the new member
        // gets the copy in the frame it arrives.
        for (int i = 0; i < 3; ++i) {
            addMember();
            pass->Changed();
            const bool rebuilt = frame();
            CHECK(!rebuilt && allCopied());
        }
        // A fifth: past the reserved four, the group grows, which builds. Every member gets the copy once it installs.
        addMember();
        pass->Changed();
        for (int i = 0; i < 30 && !allCopied(); ++i) frame();
        frame();
        CHECK(allCopied());
        for (int i = 0; i < 4; ++i) CHECK(!frame() && allCopied());
    }
    CHECK(device.WaitIdle() == rhi::Result::Ok);
    return 0;
}

// Both opt-ins at once is refused: rebinding applies new backings at once, while declared-work passes keep old work.
int TestRebindOptInsExclusive(const std::shared_ptr<org::Runtime>& runtime) {
    auto value = org::Buffer::CreateShared(rhi::HeapType::Upload, kBytes);
    auto target = org::Buffer::CreateShared(rhi::HeapType::Readback, kBytes);
    auto write = std::make_shared<RebindCopyPass>(value, target, true);
    bool refused = false;
    try {
        org::GraphHost host({.runtime = runtime});
        host.AddExtension("test.rebind-both", [&] { return std::make_unique<RebindExtension>(write, nullptr, nullptr); });
        host.ExecuteFrame(nullptr, [](org::RenderGraph&) {});
    } catch (const std::exception& error) {
        refused = std::string(error.what()).find("RebindsConcreteResources") != std::string::npos;
    }
    CHECK(refused);
    return 0;
}
// Execution handles: each frame's execution is submitted when Execute returns, and completes when a later execution
// retires it after the GPU finished. A callback added after that runs at once.
int RunExecutionHandleScenario(rhi::Device device, const std::shared_ptr<org::Runtime>& runtime) {
    auto pass = std::make_shared<RedeclarePass>();
    auto source = org::Buffer::CreateShared(rhi::HeapType::Upload, kBytes);
    auto destination = org::Buffer::CreateShared(rhi::HeapType::Readback, kBytes);
    source->SetName("handle.source"); destination->SetName("handle.destination");
    pass->source = source;
    pass->destination = destination;
    {
        org::GraphHost host({.runtime = runtime});
        host.AddExtension("test.handle", [&] { return std::make_unique<RedeclareExtension>(pass); });
        for (int i = 0; i < 3; ++i) host.ExecuteFrame(nullptr, [](org::RenderGraph&) {});
        auto* graph = host.Graph();
        CHECK(graph != nullptr);
        const auto first = graph->LastExecution();
        CHECK(first && first.Submitted() && !first.Poisoned());
        CHECK(!first.Signals().empty());
        int calls = 0;
        bool poisonedSeen = true;
        first.OnCompleted([&](bool poisoned) { ++calls; poisonedSeen = poisoned; });
        ENSURE(device.WaitIdle() == rhi::Result::Ok);
        host.ExecuteFrame(nullptr, [](org::RenderGraph&) {});
        const auto second = graph->LastExecution();
        CHECK(second.Index() == first.Index() + 1);
        CHECK(first.Completed() && calls == 1 && !poisonedSeen);
        int late = 0;
        first.OnCompleted([&](bool poisoned) { late += poisoned ? 100 : 1; });
        CHECK(late == 1);
        CHECK(second.Submitted());
    }
    CHECK(device.WaitIdle() == rhi::Result::Ok);
    return 0;
}
// Graph descs: a producer graph copies a value into its "out" parameter on the graphics queue, and a consumer graph
// copies its "in" parameter to a readback on the copy queue. Both are bound to the same buffer per execution and run
// with no CPU wait between them, so the runtime must order the consumer after the producer across graphs and queues.
// Binding other buffers of the same shape builds nothing. The cache returns the compiled graph for a known key.
int RunGraphDescScenario(rhi::Device device, const std::shared_ptr<org::Runtime>& runtime) {
    auto source = org::Buffer::CreateShared(rhi::HeapType::Upload, kBytes);
    auto readback = org::Buffer::CreateShared(rhi::HeapType::Readback, kBytes);
    auto first = org::Buffer::CreateShared(rhi::HeapType::DeviceLocal, kBytes);
    auto second = org::Buffer::CreateShared(rhi::HeapType::DeviceLocal, kBytes);
    source->SetName("desc.source"); readback->SetName("desc.readback"); first->SetName("desc.first"); second->SetName("desc.second");
    const auto revision = [](org::JobGraph& graph) -> uint64_t {
        const auto snapshot = graph.Graph() ? graph.Graph()->GetPersistentScheduleSnapshot() : nullptr;
        return snapshot && !snapshot->executables.empty() ? snapshot->executables.front()->structuralRevision : 0;
    };
    {
        org::GraphCache cache(runtime);
        const auto produceDesc = [&] {
            org::GraphDesc desc("test.produce", 0x5052u);
            auto out = desc.Parameter("out", first);
            desc.Add(org::RenderGraph::ExternalPassDesc::Copy("produce.copy",
                std::make_shared<CopyPass>(source, out, org::QueueKind::Graphics)).PreferQueue(org::QueueKind::Graphics));
            return desc;
        };
        org::GraphDesc consumeDesc("test.consume", 0x434Fu);
        auto in = consumeDesc.Parameter("in", first);
        consumeDesc.Add(org::RenderGraph::ExternalPassDesc::Copy("consume.copy",
            std::make_shared<CopyPass>(in, readback, org::QueueKind::Copy)).PreferQueue(org::QueueKind::Copy));
        auto producer = cache.GetOrCompile(produceDesc());
        auto consumer = cache.GetOrCompile(std::move(consumeDesc));
        CHECK(cache.GetOrCompile(produceDesc()) == producer);
        CHECK(cache.GetStats().hits == 1 && cache.GetStats().compiles == 2 && cache.GetStats().cached == 2);
        producer->Graph()->SetScheduleTraceEnabled(true);
        consumer->Graph()->SetScheduleTraceEnabled(true);

        uint32_t value = 0x9000u;
        uint64_t producerRevision = 0, consumerRevision = 0;
        for (int i = 0; i < 8; ++i) {
            const auto& target = (i % 2) ? second : first;
            ++value;
            ENSURE(Write(*source, value));
            const org::JobGraph::Binding out[] = {{"out", target}};
            const org::JobGraph::Binding inBinding[] = {{"in", target}};
            const auto produced = producer->Execute(out);
            const auto consumed = consumer->Execute(inBinding);
            CHECK(produced.Submitted() && consumed.Submitted());
            ENSURE(device.WaitIdle() == rhi::Result::Ok);
            CHECK(Read(*readback) == value);
            // After the first round, which lowers the passes: swapping buffers is a binding edit.
            if (i == 1) { producerRevision = revision(*producer); consumerRevision = revision(*consumer); }
            if (i > 1) CHECK(revision(*producer) == producerRevision && revision(*consumer) == consumerRevision);
            producer->Retire();
            CHECK(produced.Completed());
        }
        bool refused = false;
        try {
            const org::JobGraph::Binding unknown[] = {{"missing", first}};
            producer->Execute(unknown);
        } catch (const std::invalid_argument&) { refused = true; }
        CHECK(refused);
    }
    CHECK(device.WaitIdle() == rhi::Result::Ok);
    return 0;
}
// A graph instance driven directly: Prepare and Submit are separate calls, the execution's index is its handle's,
// rebinding keeps the compiled graph, a second Prepare before Submit is refused, and with two slots a third execution
// reuses the first slot once the GPU has finished it (no device wait between executions).
int RunGraphInstanceScenario(rhi::Device device, const std::shared_ptr<org::Runtime>& runtime) {
    auto source = org::Buffer::CreateShared(rhi::HeapType::Upload, kBytes);
    auto readback = org::Buffer::CreateShared(rhi::HeapType::Readback, kBytes);
    auto first = org::Buffer::CreateShared(rhi::HeapType::DeviceLocal, kBytes);
    auto second = org::Buffer::CreateShared(rhi::HeapType::DeviceLocal, kBytes);
    source->SetName("instance.source"); readback->SetName("instance.readback");
    first->SetName("instance.first"); second->SetName("instance.second");
    {
        org::GraphDesc desc("test.instance");
        desc.SetFramesInFlight(2);
        auto through = desc.Parameter("through", first);
        desc.Add(org::RenderGraph::ExternalPassDesc::Copy("instance.in",
            std::make_shared<CopyPass>(source, through, org::QueueKind::Graphics)).PreferQueue(org::QueueKind::Graphics));
        desc.Add(org::RenderGraph::ExternalPassDesc::Copy("instance.out",
            std::make_shared<CopyPass>(through, readback, org::QueueKind::Graphics)).PreferQueue(org::QueueKind::Graphics));
        auto instance = org::GraphInstance::Create(runtime, std::move(desc));
        CHECK(instance->FramesInFlight() == 2);
        CHECK(!instance->Compiled());
        uint32_t value = 0x7100u;
        std::shared_ptr<const org::CompiledGraph> compiled;
        org::ExecutionHandle previous;
        for (int i = 0; i < 6; ++i) {
            ++value;
            ENSURE(Write(*source, value));
            const org::GraphBinding bindings[] = {{"through", (i % 2) ? second : first}};
            instance->Bind(bindings);
            auto execution = instance->Prepare();
            CHECK(!execution.Submitted());
            bool refused = false;
            try { (void)instance->Prepare(); } catch (const std::logic_error&) { refused = true; }
            CHECK(refused);
            const auto handle = execution.Submit();
            CHECK(execution.Submitted() && handle.Submitted() && handle.Index() == execution.Index());
            CHECK(instance->LastExecution().Index() == handle.Index());
            // After the first execution, which lowers the passes: rebinding keeps the schedule.
            if (i == 1) compiled = instance->Compiled();
            if (i > 1) CHECK(compiled && instance->Compiled()->SameSchedule(*compiled));
            // The previous execution wrote the same readback: the GPU orders this one after it, so only the latest is read.
            previous = handle;
        }
        ENSURE(device.WaitIdle() == rhi::Result::Ok);
        CHECK(Read(*readback) == value);
        instance->Retire();
        CHECK(previous.Completed());
        CHECK(instance->CanPrepare());
        CHECK(compiled && compiled->BatchCount() > 0);
        // A dropped execution is submitted by its destructor.
        { auto dropped = instance->Prepare(); }
        CHECK(instance->LastExecution().Index() == previous.Index() + 1);
    }
    CHECK(device.WaitIdle() == rhi::Result::Ok);
    return 0;
}
} // namespace

int TestRebindingGpu(rhi::Device device, const std::shared_ptr<org::Runtime>& runtime) {
    const bool validation = org::persistent::SharingValidationEnabled();
    org::persistent::SetSharingValidation(true);
    if (const int line = RunScenario(device, runtime, org::RenderGraph::SharingChangePolicy::Inline)) return line;
    if (const int line = RunScenario(device, runtime, org::RenderGraph::SharingChangePolicy::Deferred)) return line;
    if (const int line = RunRelowerScenario(device, runtime, false)) return line;
    if (const int line = RunRelowerScenario(device, runtime, true)) return line;
    if (const int line = RunRelowerScenario(device, runtime, std::nullopt)) return line;
    if (const int line = RunRedeclareScenario(device, runtime)) return line;
    if (const int line = RunRebindSlotScenario(device, runtime, org::RenderGraph::SharingChangePolicy::Inline)) return line;
    if (const int line = RunRebindSlotScenario(device, runtime, org::RenderGraph::SharingChangePolicy::Deferred)) return line;
    if (const int line = TestRebindOptInsExclusive(runtime)) return line;
    if (const int line = RunRemapScenario(device, runtime)) return line;
    if (const int line = RunExecutionHandleScenario(device, runtime)) return line;
    if (const int line = RunGraphDescScenario(device, runtime)) return line;
    if (const int line = RunGraphInstanceScenario(device, runtime)) return line;
    org::persistent::SetSharingValidation(validation);
    return 0;
}
