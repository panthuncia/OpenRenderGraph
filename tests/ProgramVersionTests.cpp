#include <Render/PreparedPass.h>
#include "TestSupport.h"
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cstring>
#include <cstdio>


int TestProgramVersions(rhi::Device device) {
    Microsoft::WRL::ComPtr<ID3DBlob> code, errors;
    const char* source = "[numthreads(1,1,1)] void main() {}";
    CHECK(SUCCEEDED(D3DCompile(source, std::strlen(source), "ProgramLifetime", nullptr, nullptr,
        "main", "cs_5_0", 0, 0, &code, &errors)));
    auto create = [&](org::PipelineState& state) -> int {
        auto layout = std::make_shared<rhi::PipelineLayoutPtr>();
        CHECK(device.CreatePipelineLayout({}, *layout) == rhi::Result::Ok);
        rhi::SubobjLayout root{layout->Get().GetHandle()};
        rhi::SubobjShader shader{rhi::ShaderStage::Compute, rhi::DXIL(code.Get()), "main"};
        rhi::PipelineStreamItem items[] = {rhi::Make(root), rhi::Make(shader)};
        rhi::PipelinePtr pipeline;
        CHECK(device.CreatePipeline(items, 2, pipeline) == rhi::Result::Ok);
        state = org::PipelineState(std::move(pipeline), 0, {}, layout, root.layout);
        return 0;
    };
    org::PipelineState active, replacement;
    CHECK(create(active) == 0 && create(replacement) == 0);
    std::weak_ptr<const org::PipelineStatePayload> oldProgram = active.GetPayload();
    std::weak_ptr<const void> oldLayout = active.GetPayload()->layoutOwner;
    const auto oldHandle = active.GetPayload()->layout;
    auto signature = std::make_shared<rhi::CommandSignaturePtr>();
    rhi::IndirectArg dispatchArgs[] = {{.kind = rhi::IndirectArgKind::Dispatch}};
    CHECK(device.CreateCommandSignature(
        rhi::CommandSignatureDesc{rhi::Span<rhi::IndirectArg>(dispatchArgs, 1), 12},
        oldHandle, *signature) == rhi::Result::Ok);
    std::weak_ptr<const rhi::CommandSignaturePtr> oldSignature = signature;

    // A recording retains the pipeline payload it bound (and what else it names) until its retention is released:
    // replacing the pipeline's payload while the recording is in flight leaves the recorded program alive.
    rhi::CommandAllocatorPtr allocator;
    rhi::CommandListPtr list;
    CHECK(device.CreateCommandAllocator(rhi::QueueKind::Graphics, allocator) == rhi::Result::Ok);
    CHECK(device.CreateCommandList(rhi::QueueKind::Graphics, allocator.Get(), list) == rhi::Result::Ok);
    auto bindings = std::make_shared<const org::FrozenExecutionBindings>(
        std::vector<org::FrozenExecutionBindings::ResourceBinding>{});
    auto retention = std::make_shared<org::RecordingRetention>();
    {
        org::RecordingContext recording(list.Get(), bindings);
        recording.SetRetention(retention);
        recording.BindComputePipeline(active);
        recording.Retain(signature);
        signature.reset();
        CHECK(!oldSignature.expired());
        active.ReplacePayload(replacement.GetPayload());
        CHECK(!oldProgram.expired() && !oldLayout.expired());
        CHECK(!rhi::HandleEqual<rhi::PipelineLayoutHandle>{}(active.GetPayload()->layout, oldHandle));
        recording.Commands().Dispatch(1, 1, 1);
        CHECK(recording.Commands().EndChecked() == rhi::Result::Ok);
    }
    auto commands = list.Get();
    CHECK(device.GetQueue(rhi::QueueKind::Graphics).Submit({&commands, 1}) == rhi::Result::Ok);
    CHECK(device.WaitIdle() == rhi::Result::Ok);
    CHECK(!oldProgram.expired() && !oldLayout.expired() && !oldSignature.expired());
    retention.reset();
    CHECK(oldProgram.expired() && oldLayout.expired() && oldSignature.expired());
    return 0;
}
