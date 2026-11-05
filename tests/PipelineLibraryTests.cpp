#include <Render/PipelineLibrary.h>
#include <Render/Runtime/ThreadPoolTaskService.h>
#include "TestSupport.h"
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

// Compiles for any device with D3DCompile, and counts what it was asked. A path of "fail" fails the compile; a define named
// BROKEN fails a rebuild.
class FakeCompiler final : public org::PipelineCompiler {
public:
	explicit FakeCompiler(rhi::Device device) : m_device(device) {}

	void Compile(const org::PipelineCompileRequest& request, Completion completion) override {
		++compiles;
		devicesAsked.fetch_or(1u << static_cast<uint8_t>(request.device));
		bypassed += request.bypassCache ? 1 : 0;
		org::CompiledPipeline result;
		const bool broken = std::ranges::any_of(request.defines, [](const auto& define) { return define.name == L"BROKEN"; });
		if ((request.stages.at(0).source.path == L"fail" && !fixedSource) || broken) {
			result.error = "fake compile error";
			completion(std::move(result));
			return;
		}
		if (request.stages.at(0).source.target != L"cs_6_6") {
			result.error = "the library did not fill in the default target";
			completion(std::move(result));
			return;
		}
		Microsoft::WRL::ComPtr<ID3DBlob> code, errors;
		const char* source = "[numthreads(1,1,1)] void main() {}";
		if (FAILED(D3DCompile(source, std::strlen(source), "PipelineLibraryTest", nullptr, nullptr, "main", "cs_5_0", 0, 0, &code, &errors))) {
			result.error = "D3DCompile failed";
			completion(std::move(result));
			return;
		}
		result.stages.push_back(rhi::DXIL(code.Get()));
		result.codeOwner = std::shared_ptr<void>(code.Get(), [code](void*) {});
		completion(std::move(result));
	}

	org::PipelineLayoutBinding StandardLayout(org::PipelineLayoutKind, org::DeviceInstanceId) override {
		std::scoped_lock lock(m_mutex);
		if (!m_layout) {
			m_layout = std::make_shared<rhi::PipelineLayoutPtr>();
			if (rhi::Failed(m_device.CreatePipelineLayout({}, *m_layout))) throw std::runtime_error("no layout");
		}
		return {m_layout->Get().GetHandle(), m_layout};
	}

	std::atomic<bool> fixedSource{false}; // The "fail" path compiles from now on: its shader was edited.
	std::atomic<int> compiles{0};
	std::atomic<int> bypassed{0};
	std::atomic<uint32_t> devicesAsked{0};

private:
	rhi::Device m_device;
	std::mutex m_mutex;
	std::shared_ptr<rhi::PipelineLayoutPtr> m_layout;
};

org::ComputePipelineDesc Compute(std::wstring path = L"test.hlsl", std::string name = "Test") {
	org::ComputePipelineDesc desc;
	desc.shader = {std::move(path), L"main"};
	desc.name = std::move(name);
	return desc;
}

constexpr org::DeviceInstanceId kPrimary = org::DeviceInstanceId::Primary;
constexpr org::DeviceInstanceId kPeer = org::DeviceInstanceId::Peer;

} // namespace

int TestPipelineLibrary(rhi::Device device) {
	// Descriptions identify pipelines, and what affects the pipeline is what tells them apart.
	CHECK(org::PipelineContentHash(Compute()) == org::PipelineContentHash(Compute()));
	CHECK(org::PipelineContentHash(Compute()) != org::PipelineContentHash(Compute(L"other.hlsl")));
	{
		auto withDefine = Compute();
		withDefine.defines.push_back({L"A", L"1"});
		CHECK(org::PipelineContentHash(Compute()) != org::PipelineContentHash(withDefine));
		auto graphicsLayout = Compute();
		graphicsLayout.layout = org::PipelineLayoutRef::StandardGraphics();
		CHECK(org::PipelineContentHash(Compute()) != org::PipelineContentHash(graphicsLayout));
		org::GraphicsPipelineDesc a, b;
		b.topology = rhi::PrimitiveTopology::TriangleStrip;
		CHECK(org::PipelineContentHash(a) != org::PipelineContentHash(b));
		a.blend = rhi::BlendState{};
		b = a;
		b.blend->attachments[3].enable = true;
		CHECK(org::PipelineContentHash(a) != org::PipelineContentHash(b));
	}

	auto compiler = std::make_shared<FakeCompiler>(device);
	auto tasks = std::make_shared<org::runtime::ThreadPoolTaskService>();

	{
		org::PipelineLibrary noCompiler(tasks, nullptr);
		bool threw = false;
		try { noCompiler.Acquire(Compute()); } catch (const std::logic_error&) { threw = true; }
		CHECK(threw);
	}

	org::PipelineLibrary library(tasks, compiler);
	// One physical device under two ids stands in for a primary and a peer: what differs is the slot and the request.
	library.RegisterDevice(kPrimary, rhi::Backend::D3D12, device);
	library.RegisterDevice(kPeer, rhi::Backend::D3D12, device);

	// The same description is the same pipeline while it is alive, and acquiring compiles nothing.
	auto first = library.Acquire(Compute());
	auto again = library.Acquire(Compute());
	CHECK(first == again);
	CHECK(first->State().GetSlot() == again->State().GetSlot());
	CHECK(compiler->compiles == 0);
	CHECK(first->Status(kPrimary) == org::PipelineBuildStatus::Idle);

	// A build runs for the requested device alone, once, however many ask.
	library.Request(first, kPeer);
	library.Request(first, kPeer);
	CHECK(library.Wait(first, kPeer) == org::PipelineBuildStatus::Ready);
	library.Request(first, kPeer);
	CHECK(compiler->compiles == 1);
	CHECK(compiler->devicesAsked == (1u << 1));
	CHECK(first->State().HasBackendPipeline(kPeer));
	CHECK(!first->State().HasBackendPipeline(kPrimary));
	CHECK(first->Status(kPrimary) == org::PipelineBuildStatus::Idle);
	CHECK(library.Wait(first, kPrimary) == org::PipelineBuildStatus::Idle); // Never requested: does not block.
	const auto peerPayload = first->State().GetPayload(kPeer);
	CHECK(peerPayload->label == "initial");
	CHECK(peerPayload->layout.valid());
	CHECK(peerPayload->layoutOwner != nullptr);

	// A failed build reports why, and stays failed.
	auto failing = library.Acquire(Compute(L"fail", "Failing"));
	library.Request(failing, kPrimary);
	CHECK(library.Wait(failing, kPrimary) == org::PipelineBuildStatus::Failed);
	CHECK(failing->Error(kPrimary) == "fake compile error");
	CHECK(!failing->State().HasBackendPipeline(kPrimary));
	const auto compilesAfterFailure = compiler->compiles.load();
	library.Request(failing, kPrimary);
	CHECK(compiler->compiles == compilesAfterFailure);

	// A rebuild replaces the payload of every requested device at once, and only at the publish.
	library.Request(first, kPrimary);
	CHECK(library.Wait(first, kPrimary) == org::PipelineBuildStatus::Ready);
	const auto oldPrimary = first->State().GetPayload(kPrimary);
	const auto oldPeer = first->State().GetPayload(kPeer);
	const auto bypassedBefore = compiler->bypassed.load();
	{
		std::atomic<int> done{0};
		std::string reported = "unset";
		org::PipelineRebuildOptions options;
		options.completed = [&](const std::string& error) { reported = error; done.store(1); done.notify_all(); };
		library.Rebuild(first, std::move(options));
		done.wait(0);
		CHECK(reported.empty());
	}
	CHECK(compiler->bypassed == bypassedBefore + 2);
	CHECK(first->State().GetPayload(kPrimary) == oldPrimary);
	CHECK(first->State().GetPayload(kPeer) == oldPeer);
	CHECK(library.PublishRebuilt() == 1);
	CHECK(first->State().GetPayload(kPrimary) != oldPrimary);
	CHECK(first->State().GetPayload(kPeer) != oldPeer);
	// One generation number on every device, and the old one can come back.
	CHECK(first->State().GetPayload(kPeer)->generation == first->State().GetPayload(kPrimary)->generation);
	CHECK(first->State().GetPayload(kPeer)->generation == oldPeer->generation + 1);
	CHECK(library.PublishRebuilt() == 0);
	CHECK(library.RetainedGenerations(first) == std::vector<uint64_t>({oldPeer->generation, oldPeer->generation + 1}));
	CHECK(library.Activate(first, oldPeer->generation));
	CHECK(first->State().GetPayload(kPeer)->generation == oldPeer->generation + 1); // Not before the publish.
	CHECK(library.PublishRebuilt() == 1);
	CHECK(first->State().GetPayload(kPeer) == oldPeer);
	CHECK(first->State().GetPayload(kPrimary) == oldPrimary);
	CHECK(!library.Activate(first, 999));
	CHECK(library.Activate(first, oldPeer->generation + 1));
	CHECK(library.PublishRebuilt() == 1);

	// A failed rebuild keeps what is there.
	{
		const auto kept = first->State().GetPayload(kPeer);
		std::atomic<int> done{0};
		std::string reported;
		org::PipelineRebuildOptions options;
		options.defineOverrides.push_back({L"BROKEN", L"1"});
		options.completed = [&](const std::string& error) { reported = error; done.store(1); done.notify_all(); };
		library.Rebuild(first, std::move(options));
		done.wait(0);
		CHECK(reported == "fake compile error");
		CHECK(library.PublishRebuilt() == 0);
		CHECK(first->State().GetPayload(kPeer) == kept);
	}

	// A rebuild that succeeds clears an earlier failure: a fixed shader recovers without restarting.
	{
		compiler->fixedSource = true;
		std::atomic<int> done{0};
		std::string reported = "unset";
		org::PipelineRebuildOptions options;
		options.completed = [&](const std::string& error) { reported = error; done.store(1); done.notify_all(); };
		library.Rebuild(failing, std::move(options));
		done.wait(0);
		CHECK(reported.empty());
		CHECK(failing->Status(kPrimary) == org::PipelineBuildStatus::Failed); // Until it publishes.
		CHECK(library.PublishRebuilt() == 1);
		CHECK(failing->Status(kPrimary) == org::PipelineBuildStatus::Ready);
		CHECK(failing->Error(kPrimary).empty());
		CHECK(failing->State().HasBackendPipeline(kPrimary));
	}

	// Builds on one device land while recordings read another: readers see a whole payload or nothing, on every thread.
	{
		std::vector<org::PipelineLibraryEntryPtr> entries;
		for (int i = 0; i < 32; ++i) entries.push_back(library.Acquire(Compute(L"test.hlsl", "Concurrent" + std::to_string(i))));
		std::atomic<bool> stop{false};
		std::atomic<int> torn{0};
		std::vector<std::thread> readers;
		for (int thread = 0; thread < 4; ++thread) readers.emplace_back([&] {
			while (!stop.load(std::memory_order_relaxed))
				for (const auto& entry : entries) {
					const auto payload = entry->State().GetPayload(kPeer);
					if (payload && (!payload->pso || !payload->layout.valid())) ++torn;
				}
		});
		for (const auto& entry : entries) library.Request(entry, kPeer);
		for (const auto& entry : entries) CHECK(library.Wait(entry, kPeer) == org::PipelineBuildStatus::Ready);
		stop = true;
		for (auto& reader : readers) reader.join();
		CHECK(torn == 0);
		for (const auto& entry : entries) CHECK(entry->State().HasBackendPipeline(kPeer));
	}

	// An entry goes with its last declaration; a later Acquire starts a new one.
	{
		const auto liveBefore = library.LiveEntries().size();
		auto temporary = library.Acquire(Compute(L"test.hlsl", "Temporary"));
		CHECK(library.LiveEntries().size() == liveBefore + 1);
		const std::weak_ptr<org::PipelineLibraryEntry> weak = temporary;
		temporary.reset();
		CHECK(weak.expired());
		CHECK(library.LiveEntries().size() == liveBefore);
	}
	return 0;
}
