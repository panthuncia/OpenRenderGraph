// Persistent (static) graph execution on an adopted, host-owned Vulkan device: the
// embedding model used by DXVK-based hosts. The test plays the host. It owns the
// VkDevice, its queue and a destination buffer, and checks that:
//   - a typed compute pass and a typed copy pass from an extension run in the
//     persistent main segment and write a graph-owned and an imported buffer;
//   - Execute submits the current frame before returning (data is visible after a
//     wait on BasicRHI's own timelines);
//   - every submission is bracketed by the host's queue lock;
//   - host uploads (BUFFER_UPLOAD) reach the main passes through the persistent Pre
//     segment's upload pass;
//   - rotating the imported buffer with RefreshShared redirects writes on the next
//     frame (a binding-only edit, no rebuild), and a requested rebuild keeps working;
//   - shutdown leaves the host's device and buffers intact.

#include "rhi.h"
#include "rhi_helpers.h"
#include "rhi_interop_vulkan.h"

#include <OpenRenderGraph/OpenRenderGraph.h>
#include "Render/RenderGraph/RenderGraph.h"
#include "Render/Runtime/RuntimeDevice.h"
#include "Render/Runtime/ScopedActiveGraphServices.h"
#include "Render/Runtime/DescriptorServiceAccess.h"
#include "Render/Runtime/ThreadPoolTaskService.h"
#include "OpenRenderGraph/PersistentGraphHost.h"
#include "Managers/Singletons/DescriptorHeapManager.h"
#include "RenderPasses/Base/TypedRenderGraphPass.h"
#include "Resources/Buffers/Buffer.h"
#include "Resources/ExternalBufferResource.h"
#include "Render/Runtime/UploadServiceAccess.h"
#include "Render/Runtime/IUploadService.h"
#include "Render/Runtime/StagedUploadBatch.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <thread>
#include <vector>

#ifndef ORG_TEST_PERSISTENT_WRITE_SPV
#error "ORG_TEST_PERSISTENT_WRITE_SPV must name the compiled SPIR-V of tests/shaders/PersistentHostWrite.hlsl"
#endif

#define REQUIRE(cond, msg) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); return 1; } } while (0)

namespace {
constexpr uint32_t kCount = 64;
constexpr uint64_t kBytes = kCount * sizeof(uint32_t);

struct HostDevice {
	VkInstance instance = VK_NULL_HANDLE;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkQueue queue = VK_NULL_HANDLE;
	uint32_t family = 0;
	std::vector<const char*> extensions;
	VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	VkPhysicalDeviceVulkan12Features vulkan12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	VkPhysicalDeviceVulkan13Features vulkan13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
	VkPhysicalDeviceDescriptorHeapFeaturesEXT descriptorHeap{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT};
};

struct HostBuffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	uint32_t* mapped = nullptr;
	VkBufferUsageFlags usage = 0;
};

struct LockCounters {
	std::atomic_int locks{0}, unlocks{0}, depth{0};
	std::atomic_bool unbalanced{false};
};
void Lock(void* user, VkQueue) {
	auto* c = static_cast<LockCounters*>(user);
	++c->locks;
	if (++c->depth != 1) c->unbalanced = true;
}
void Unlock(void* user, VkQueue) {
	auto* c = static_cast<LockCounters*>(user);
	++c->unlocks;
	if (--c->depth != 0) c->unbalanced = true;
}

bool HasDeviceExtension(VkPhysicalDevice device, const char* name) {
	uint32_t count = 0;
	vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
	std::vector<VkExtensionProperties> props(count);
	vkEnumerateDeviceExtensionProperties(device, nullptr, &count, props.data());
	for (const auto& p : props) if (std::strcmp(p.extensionName, name) == 0) return true;
	return false;
}

// 0 ok, 77 skip, 1 failure.
int CreateHostDevice(HostDevice& host) {
	if (volkInitialize() != VK_SUCCESS) return 77;
	VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.pApplicationName = "PersistentVulkanHostTests";
	app.apiVersion = VK_API_VERSION_1_3;
	VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	ici.pApplicationInfo = &app;
	if (vkCreateInstance(&ici, nullptr, &host.instance) != VK_SUCCESS) return 77;
	volkLoadInstanceOnly(host.instance);
	uint32_t count = 0;
	vkEnumeratePhysicalDevices(host.instance, &count, nullptr);
	std::vector<VkPhysicalDevice> devices(count);
	vkEnumeratePhysicalDevices(host.instance, &count, devices.data());
	for (VkPhysicalDevice candidate : devices)
		if (HasDeviceExtension(candidate, VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME) &&
			HasDeviceExtension(candidate, VK_KHR_MAINTENANCE_5_EXTENSION_NAME)) { host.physicalDevice = candidate; break; }
	if (!host.physicalDevice) return 77;
	uint32_t familyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(host.physicalDevice, &familyCount, nullptr);
	std::vector<VkQueueFamilyProperties> families(familyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(host.physicalDevice, &familyCount, families.data());
	for (uint32_t i = 0; i < familyCount; ++i)
		if ((families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) { host.family = i; break; }

	host.extensions = {VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, VK_KHR_MAINTENANCE_5_EXTENSION_NAME};
	host.vulkan12.bufferDeviceAddress = VK_TRUE;
	host.vulkan12.timelineSemaphore = VK_TRUE;
	host.vulkan12.descriptorIndexing = VK_TRUE;
	host.vulkan12.runtimeDescriptorArray = VK_TRUE;
	host.vulkan12.scalarBlockLayout = VK_TRUE;
	host.vulkan12.descriptorBindingPartiallyBound = VK_TRUE;
	host.vulkan12.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
	host.vulkan12.hostQueryReset = VK_TRUE;
	host.vulkan13.dynamicRendering = VK_TRUE;
	host.vulkan13.synchronization2 = VK_TRUE;
	host.vulkan13.maintenance4 = VK_TRUE;
	host.descriptorHeap.descriptorHeap = VK_TRUE;
	host.features.pNext = &host.vulkan12;
	host.vulkan12.pNext = &host.vulkan13;
	host.vulkan13.pNext = &host.descriptorHeap;
	const float priority = 1.0f;
	VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	qci.queueFamilyIndex = host.family;
	qci.queueCount = 1;
	qci.pQueuePriorities = &priority;
	VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	dci.pNext = &host.features;
	dci.queueCreateInfoCount = 1;
	dci.pQueueCreateInfos = &qci;
	dci.enabledExtensionCount = static_cast<uint32_t>(host.extensions.size());
	dci.ppEnabledExtensionNames = host.extensions.data();
	if (vkCreateDevice(host.physicalDevice, &dci, nullptr, &host.device) != VK_SUCCESS) return 1;
	volkLoadDevice(host.device);
	vkGetDeviceQueue(host.device, host.family, 0, &host.queue);
	return 0;
}

bool CreateHostBuffer(const HostDevice& host, HostBuffer& out) {
	VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bci.size = kBytes;
	bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	if (vkCreateBuffer(host.device, &bci, nullptr, &out.buffer) != VK_SUCCESS) return false;
	out.usage = bci.usage;
	VkMemoryRequirements req{};
	vkGetBufferMemoryRequirements(host.device, out.buffer, &req);
	VkPhysicalDeviceMemoryProperties props{};
	vkGetPhysicalDeviceMemoryProperties(host.physicalDevice, &props);
	uint32_t type = UINT32_MAX;
	const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	for (uint32_t i = 0; i < props.memoryTypeCount && type == UINT32_MAX; ++i)
		if ((req.memoryTypeBits & (1u << i)) && (props.memoryTypes[i].propertyFlags & want) == want) type = i;
	if (type == UINT32_MAX) return false;
	VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
	flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
	VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	mai.pNext = &flags;
	mai.allocationSize = req.size;
	mai.memoryTypeIndex = type;
	if (vkAllocateMemory(host.device, &mai, nullptr, &out.memory) != VK_SUCCESS) return false;
	if (vkBindBufferMemory(host.device, out.buffer, out.memory, 0) != VK_SUCCESS) return false;
	void* mapped = nullptr;
	if (vkMapMemory(host.device, out.memory, 0, kBytes, 0, &mapped) != VK_SUCCESS) return false;
	out.mapped = static_cast<uint32_t*>(mapped);
	std::memset(out.mapped, 0, kBytes);
	return true;
}

void DestroyHostBuffer(const HostDevice& host, HostBuffer& buffer) {
	if (buffer.mapped) vkUnmapMemory(host.device, buffer.memory);
	vkDestroyBuffer(host.device, buffer.buffer, nullptr);
	vkFreeMemory(host.device, buffer.memory, nullptr);
	buffer = {};
}

std::shared_ptr<org::ExternalBufferResource> ImportHostBuffer(rhi::Device device, const HostBuffer& buffer, const char* name,
	std::shared_ptr<org::ExternalBufferResource> refresh = {}) {
	rhi::ResourcePtr imported;
	rhi::vulkan::ImportedBufferDesc desc{};
	desc.buffer = buffer.buffer;
	desc.size = kBytes;
	desc.usage = buffer.usage;
	desc.debugName = name;
	if (rhi::vulkan::import_buffer(device, desc, imported) != rhi::Result::Ok) {
		std::fprintf(stderr, "import_buffer failed for %s\n", name);
		return {};
	}
	if (refresh) {
		if (refresh->RefreshShared(std::move(imported), kBytes, {})) return refresh;
		std::fprintf(stderr, "RefreshShared rejected %s\n", name);
		return {};
	}
	return org::ExternalBufferResource::CreateShared(std::move(imported), kBytes);
}

// --- The graph ---------------------------------------------------------------

struct ComputeProgram {
	rhi::PipelineLayoutPtr layout;
	rhi::PipelinePtr pipeline;
};

struct WriteBindings { org::ResourceBindingToken input, target; };
struct WriteFrame {
	std::shared_ptr<const ComputeProgram> program;
	uint32_t inputIndex = 0;
	uint32_t targetIndex = 0;
};

// Reads the host-uploaded value (Pre-segment upload pass) and writes value + index.
class WritePass final : public org::TypedRenderGraphPass<WritePass, WriteFrame, WriteBindings> {
public:
	WritePass(std::shared_ptr<org::Buffer> input, std::shared_ptr<org::Buffer> target, std::shared_ptr<const ComputeProgram> program)
		: m_input(std::move(input)), m_target(std::move(target)), m_program(std::move(program)) {}
	WriteBindings Declare(org::PassBuilder& builder) {
		builder.PreferQueue(org::QueueKind::Graphics);
		return {builder.BindShaderResource(m_input), builder.BindUnorderedAccess(m_target)};
	}
	WriteFrame Prepare(const WriteBindings& bindings, const org::PassPrepareContext& preparation) const {
		return {m_program, preparation.ResolveView(bindings.input, {org::BindlessViewKind::ShaderResource}).index,
			preparation.ResolveView(bindings.target, {org::BindlessViewKind::UnorderedAccess}).index};
	}
	void InvocationRevision(const org::PassPrepareContext&, std::vector<uint64_t>& out) const {
		out.push_back(reinterpret_cast<uintptr_t>(m_program.get()));
	}
	static void Record(const WriteBindings&, const WriteFrame& frame, org::PassRecordContext& recording) {
		auto& commands = recording.Commands();
		commands.BindLayout(frame.program->layout->GetHandle());
		commands.BindPipeline(frame.program->pipeline->GetHandle());
		const uint32_t constants[3] = {frame.targetIndex, frame.inputIndex, kCount};
		commands.PushConstants(rhi::ShaderStage::Compute, 0, 0, 0, 3, constants);
		commands.Dispatch((kCount + 63) / 64, 1, 1);
	}
private:
	std::shared_ptr<org::Buffer> m_input;
	std::shared_ptr<org::Buffer> m_target;
	std::shared_ptr<const ComputeProgram> m_program;
};

struct CopyBindings { org::ResourceBindingToken source, destination; };
class CopyToHostPass final : public org::TypedRenderGraphPass<CopyToHostPass, org::EmptyPassFrameData, CopyBindings> {
public:
	CopyToHostPass(std::shared_ptr<org::Buffer> source, std::shared_ptr<org::ExternalBufferResource> destination,
		std::shared_ptr<std::atomic<uint64_t>> destinationRevision)
		: m_source(std::move(source)), m_destination(std::move(destination)), m_destinationRevision(std::move(destinationRevision)) {}
	CopyBindings Declare(org::PassBuilder& builder) {
		builder.PreferQueue(org::QueueKind::Graphics);
		return {builder.BindCopySource(m_source), builder.BindCopyDestination(m_destination)};
	}
	void InvocationRevision(const org::PassPrepareContext&, std::vector<uint64_t>& out) const {
		out.push_back(m_destinationRevision->load(std::memory_order_acquire));
	}
	static void Record(const CopyBindings& bindings, org::PassRecordContext& recording) {
		recording.Commands().CopyBufferRegion(recording.Resolve(bindings.destination).GetHandle(), 0,
			recording.Resolve(bindings.source).GetHandle(), 0, kBytes);
	}
private:
	std::shared_ptr<org::Buffer> m_source;
	std::shared_ptr<org::ExternalBufferResource> m_destination;
	std::shared_ptr<std::atomic<uint64_t>> m_destinationRevision;
};

class HostExtension final : public org::RenderGraph::IRenderGraphExtension {
public:
	HostExtension(std::shared_ptr<org::Buffer> input, std::shared_ptr<org::Buffer> scratch,
		std::shared_ptr<org::ExternalBufferResource> output, std::shared_ptr<const ComputeProgram> program,
		std::shared_ptr<std::atomic<uint64_t>> outputRevision)
		: m_input(std::move(input)), m_scratch(std::move(scratch)), m_output(std::move(output)), m_program(std::move(program)),
		m_outputRevision(std::move(outputRevision)) {}
	void PrepareForBuild(org::RenderGraph& graph) override {
		graph.RegisterResource(org::ResourceIdentifier("test.input"), m_input);
		graph.RegisterResource(org::ResourceIdentifier("test.scratch"), m_scratch);
		graph.RegisterResource(org::ResourceIdentifier("test.host-output"), m_output);
	}
	void GatherStructuralPasses(org::RenderGraph&, std::vector<org::RenderGraph::ExternalPassDesc>& out) override {
		out.push_back(org::RenderGraph::ExternalPassDesc::Compute("test.write",
			std::static_pointer_cast<org::RenderPass>(std::make_shared<WritePass>(m_input, m_scratch, m_program)))
			.PreferQueue(org::QueueKind::Graphics));
		out.push_back(org::RenderGraph::ExternalPassDesc::Copy("test.copy-to-host",
			std::static_pointer_cast<org::RenderPass>(std::make_shared<CopyToHostPass>(m_scratch, m_output, m_outputRevision)))
			.PreferQueue(org::QueueKind::Graphics));
	}
private:
	std::shared_ptr<org::Buffer> m_input;
	std::shared_ptr<org::Buffer> m_scratch;
	std::shared_ptr<org::ExternalBufferResource> m_output;
	std::shared_ptr<const ComputeProgram> m_program;
	std::shared_ptr<std::atomic<uint64_t>> m_outputRevision;
};

std::shared_ptr<ComputeProgram> CreateProgram(rhi::Device device) {
	std::ifstream file(ORG_TEST_PERSISTENT_WRITE_SPV, std::ios::binary);
	std::vector<char> spirv((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	if (spirv.empty()) return {};
	auto program = std::make_shared<ComputeProgram>();
	rhi::PushConstantRangeDesc constants{};
	constants.visibility = rhi::ShaderStage::Compute;
	constants.num32BitValues = 3;
	constants.set = 0;
	constants.binding = 0;
	if (device.CreatePipelineLayout(rhi::PipelineLayoutDesc{.pushConstants = {&constants, 1},
		.flags = rhi::PipelineLayoutFlags::PF_None}, program->layout) != rhi::Result::Ok) return {};
	rhi::SubobjLayout layout{program->layout->GetHandle()};
	rhi::SubobjShader shader{rhi::ShaderStage::Compute, {spirv.data(), static_cast<uint32_t>(spirv.size())}, "main"};
	const rhi::PipelineStreamItem items[] = {rhi::Make(layout), rhi::Make(shader)};
	if (device.CreatePipeline(items, 2, program->pipeline) != rhi::Result::Ok) return {};
	return program;
}

bool Matches(const HostBuffer& buffer, uint32_t value) {
	for (uint32_t i = 0; i < kCount; ++i) if (buffer.mapped[i] != value + i) return false;
	return true;
}
} // namespace

int TestFrameRetirement(rhi::Device device, rhi::Backend backend);

int main(int argc, char** argv) {
	const bool async = argc > 1 && std::strcmp(argv[1], "async") == 0;
	HostDevice host;
	if (const int created = CreateHostDevice(host); created != 0) {
		std::puts(created == 77 ? "SKIP: no Vulkan device with VK_EXT_descriptor_heap" : "FAIL: host device");
		return created == 77 ? 77 : 1;
	}
	HostBuffer first, second;
	REQUIRE(CreateHostBuffer(host, first) && CreateHostBuffer(host, second), "host buffers");

	LockCounters counters;
	{
		rhi::vulkan::AdoptedVulkanDeviceInfo info{};
		info.getInstanceProcAddr = vkGetInstanceProcAddr;
		info.instance = host.instance;
		info.instanceApiVersion = VK_API_VERSION_1_3;
		info.physicalDevice = host.physicalDevice;
		info.device = host.device;
		info.enabledDeviceExtensions = host.extensions.data();
		info.enabledDeviceExtensionCount = static_cast<uint32_t>(host.extensions.size());
		info.enabledFeatureChain = &host.features;
		info.queues[0] = {host.queue, host.family, 0};
		info.submissionHooks = {&counters, &Lock, &Unlock};
		rhi::DevicePtr device;
		REQUIRE(rhi::vulkan::AdoptVulkanDevice(info, device) == rhi::Result::Ok, "AdoptVulkanDevice");
		REQUIRE(TestFrameRetirement(device.Get(), rhi::Backend::Vulkan) == 0, "descriptor and frame retirement");
		// The host registers the runtime device, owns the task service and upload
		// pass, and drives persistent execution with the external queue boundary.
		org::PersistentGraphHost host({.device = device.Get(), .backend = rhi::Backend::Vulkan,
			.tasks = std::make_shared<org::runtime::ThreadPoolTaskService>(2),
			.queueBoundary = {.entry = true, .exit = true}, .epochOrder = {0}, .closedExecutions = async});
		{
			auto program = CreateProgram(device.Get());
			REQUIRE(program, "compute program");
			auto scratch = org::Buffer::CreateSharedUnmaterialized(rhi::HeapType::DeviceLocal, kBytes, true);
			org::BufferBase::DescriptorRequirements requirements{};
			requirements.createUAV = true;
			requirements.uavDesc = {.dimension = rhi::UavDim::Buffer, .formatOverride = rhi::Format::R32_Typeless,
				.buffer = {.kind = rhi::BufferViewKind::Raw, .firstElement = 0, .numElements = kCount}};
			scratch->SetDescriptorRequirements(requirements);
			scratch->SetName("test.scratch");
			auto output = ImportHostBuffer(device.Get(), first, "test.host-output.first");
			REQUIRE(output, "import host buffer");
			auto input = org::Buffer::CreateSharedUnmaterialized(rhi::HeapType::DeviceLocal, 16, false);
			org::BufferBase::DescriptorRequirements inputViews{};
			inputViews.createSRV = true;
			inputViews.srvDesc = {.dimension = rhi::SrvDim::Buffer, .formatOverride = rhi::Format::R32_Typeless,
				.buffer = {.kind = rhi::BufferViewKind::Raw, .firstElement = 0, .numElements = 4}};
			input->SetDescriptorRequirements(inputViews);
			input->SetName("test.input");

			uint32_t value = 0;
			// The value reaches the GPU through the graph's upload pass (persistent Pre segment).
			const auto upload = [&](org::RenderGraph&) {
				BUFFER_UPLOAD(&value, sizeof(value), org::runtime::UploadTarget::FromShared(input), 0);
				if (async) {
					// An abandoned payload followed by its replacement: the last overlapping write wins.
					auto first = org::runtime::StagedUploadBatch::Create(256);
					const auto wrong = value ^ 0x12345678u;
					first->Stage(org::runtime::UploadTarget::FromShared(input), 0, &wrong, sizeof(wrong));
					first->Stage(org::runtime::UploadTarget::FromShared(input), 4, &wrong, sizeof(wrong));
					org::runtime::GetActiveUploadService()->SubmitStagedUploads(std::move(first));
					auto replacement = org::runtime::StagedUploadBatch::Create(256);
					replacement->Stage(org::runtime::UploadTarget::FromShared(input), 0, &value, sizeof(value));
					org::runtime::GetActiveUploadService()->SubmitStagedUploads(std::move(replacement));
				}
			};
			auto outputRevision = std::make_shared<std::atomic<uint64_t>>(1);
			host.AddExtension("test.host", [&] { return std::make_unique<HostExtension>(input, scratch, output, program, outputRevision); });
			org::PersistentGraphHost::FrameCallback premature = [](org::RenderGraph&) {};
			REQUIRE(!host.TryPostOwnedPreparation(premature) && premature, "pre-build request stays with caller");
			org::PersistentGraphHost::FrameTicketReservation reservation;
			const uint32_t requiredEpoch = 0;
			REQUIRE(!host.TryReserveReadyEpochs({&requiredEpoch, 1}, reservation), "no reservation before async start");
			if (async) host.SetAsyncEpochs({0});
			std::atomic<uint32_t> ownedRequests{0};
			std::atomic<bool> ownedServices{false};
			std::weak_ptr<int> ownedWeak;
			if (async) {
				const uint32_t unavailableEpochs[] = {0, 1};
				REQUIRE(!host.TryReserveReadyEpochs(unavailableEpochs, reservation) && reservation.Empty(),
					"incomplete epoch set does not consume a control credit");
				const auto readyDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
				while (!host.TryReserveReadyEpochs({&requiredEpoch, 1}, reservation) && std::chrono::steady_clock::now() < readyDeadline)
					std::this_thread::yield();
				REQUIRE(!reservation.Empty(), "initial ticket and control-mailbox credit are reserved");
				auto ownedValue = std::make_shared<int>(42);
				ownedWeak = ownedValue;
				const auto renderThread = std::this_thread::get_id();
				org::PersistentGraphHost::FrameCallback request = [ownedValue, renderThread, &ownedRequests, &ownedServices](org::RenderGraph&) {
					ownedServices.store(*ownedValue == 42 && std::this_thread::get_id() != renderThread &&
						org::runtime::GetActiveDescriptorService() != nullptr && org::runtime::GetActiveUploadService() != nullptr);
					ownedRequests.fetch_add(1);
				};
				REQUIRE(host.TryPostOwnedPreparation(request), "owned preparation admitted without a wait");
				REQUIRE(!request, "accepted request transfers ownership");
				ownedValue.reset();
				// Hold the host consumer while filling its bounded mailbox. The
				// overflow call must return promptly with the request still owned.
				auto submittedOwner = std::make_shared<int>(73);
				std::weak_ptr<int> submittedWeak = submittedOwner;
				reservation.HoldResources(submittedOwner);
				submittedOwner.reset();
				std::promise<void> releaseConsumer;
				auto gate = releaseConsumer.get_future().share();
				std::atomic<bool> entered{false};
				org::PersistentGraphHost::FrameCallback blocker = [gate, &entered](org::RenderGraph&) {
					entered.store(true);
					gate.wait();
				};
				const bool blockerAccepted = host.TryPostOwnedPreparation(blocker);
				const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
				while (!entered.load() && std::chrono::steady_clock::now() < deadline)
					std::this_thread::yield();
				bool filled = entered.load();
				if (filled)
					for (uint32_t i = 0; i < 63; ++i) {
						org::PersistentGraphHost::FrameCallback filler = [](org::RenderGraph&) {};
						filled &= host.TryPostOwnedPreparation(filler);
					}
				org::PersistentGraphHost::FrameCallback overflow = [](org::RenderGraph&) {};
				const bool refused = filled && !host.TryPostOwnedPreparation(overflow) && static_cast<bool>(overflow);
				// The worker is still blocked and the mailbox has exactly one free
				// cell: the reserved frame must spend that cell without waiting.
				value = 1000u;
				const int firstLocksBefore = counters.locks.load();
				const bool submittedWhileFull = refused && host.TrySubmitReservedEpoch(reservation, 0, upload);
				releaseConsumer.set_value();
				REQUIRE(blockerAccepted && filled && refused && submittedWhileFull,
					"reserved submission uses its control credit while owned-request mailbox is saturated");
				REQUIRE(counters.locks.load() > firstLocksBefore, "reserved epoch reached host queue despite blocked worker");
				REQUIRE(!host.TrySubmitReservedEpoch(reservation, 0, upload), "reserved ticket is consumed exactly once");
				reservation.Clear();
				// No following frame is needed to release the exact submitted owner:
				// the host worker observes GPU completion independently of slot reuse.
				REQUIRE(device->WaitIdle() == rhi::Result::Ok, "wait for submitted owner completion");
				const auto ownerDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
				while (!submittedWeak.expired() && std::chrono::steady_clock::now() < ownerDeadline)
					std::this_thread::yield();
				REQUIRE(submittedWeak.expired(), "submitted owner retires without another frame");
				// DCLF still uses the legacy ticket path. It needs the same
				// completion lifetime as a nonblocking reservation.
				auto legacyOwner = std::make_shared<int>(74);
				std::weak_ptr<int> legacyWeak = legacyOwner;
				host.SubmitEpoch(0, upload, legacyOwner);
				legacyOwner.reset();
				REQUIRE(device->WaitIdle() == rhi::Result::Ok, "wait for legacy owned submission");
				const auto legacyDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
				while (!legacyWeak.expired() && std::chrono::steady_clock::now() < legacyDeadline)
					std::this_thread::yield();
				REQUIRE(legacyWeak.expired(), "legacy submission owner retires without another frame");
			}
			for (uint32_t frame = 0; frame < 6; ++frame) {
				auto syncOwner = !async && frame == 0 ? std::make_shared<int>(91) : std::shared_ptr<int>{};
				std::weak_ptr<int> syncWeak = syncOwner;
				if (frame == 3) {
					// Rotate the host buffer: the next frame must write the second one.
					REQUIRE(ImportHostBuffer(device.Get(), second, "test.host-output.second", output), "RefreshShared host buffer");
					outputRevision->fetch_add(1, std::memory_order_release);
				}
				value = 1000u * (frame + 1);
				const int locksBefore = counters.locks.load();
				if (async) {
					if (frame != 0) host.SubmitEpoch(0, upload);
				} else host.ExecuteFrame(nullptr, upload, UINT32_MAX, syncOwner);
				syncOwner.reset();
				if (!async || frame != 0)
					REQUIRE(counters.locks.load() > locksBefore, "Execute submitted the current frame before returning");
				REQUIRE(device->WaitIdle() == rhi::Result::Ok, "wait for BasicRHI timelines");
				if (!async && frame == 0) {
					REQUIRE(!syncWeak.expired(), "synchronous owner survives CPU caller release");
					org::DescriptorHeapManager::GetInstance().ProcessDeferredReleases(0);
					REQUIRE(syncWeak.expired(), "synchronous owner retires after accepted GPU work");
				}
				const HostBuffer& expected = frame < 3 ? first : second;
				if (!Matches(expected, 1000u * (frame + 1))) {
					std::fprintf(stderr, "FAIL: frame %u wrote %u..., expected %u...\n", frame, expected.mapped[0], 1000u * (frame + 1));
					return 1;
				}
				if (frame >= 3 && !Matches(first, 3000u)) {
					std::fprintf(stderr, "FAIL: frame %u touched the retired host buffer\n", frame);
					return 1;
				}
			}
			REQUIRE(!counters.unbalanced && counters.locks.load() == counters.unlocks.load(), "balanced host queue locks");
			if (async) {
				const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
				while (!host.TryReserveReadyEpochs({&requiredEpoch, 1}, reservation) && std::chrono::steady_clock::now() < deadline)
					std::this_thread::yield();
				REQUIRE(!reservation.Empty(), "ticket ready for revision-change test");
				bool rejectedStale = false;
				try {
					(void)host.TrySubmitReservedEpoch(reservation, 0, [&](org::RenderGraph&) {
						outputRevision->fetch_add(1, std::memory_order_release);
					});
				} catch (const std::logic_error&) {
					rejectedStale = true;
				}
				reservation.Clear();
				REQUIRE(rejectedStale, "post-reservation revision change rejects instead of submitting stale work");
				value = 6500u;
				host.SubmitEpoch(0, upload);
				REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(second, value), "replacement ticket recovers after stale reservation");
			}

			// A rebuild (e.g. on resize) retires the old graph and keeps rendering.
			host.RequestRebuild();
			if (async) REQUIRE(!host.TryReserveReadyEpochs({&requiredEpoch, 1}, reservation), "pending rebuild prevents reservation");
			value = 7000u;
			if (async) host.SubmitEpoch(0, upload);
			else host.ExecuteFrame(nullptr, upload);
			REQUIRE(device->WaitIdle() == rhi::Result::Ok, "wait after rebuild");
			REQUIRE(Matches(second, 7000u), "frame after rebuild");
			if (async) {
				host.SetAsyncEpochs({});
				REQUIRE(ownedRequests.load() == 1 && ownedServices.load(), "owned preparation ran on host thread with services");
				REQUIRE(ownedWeak.expired(), "owned request released after completion");
				org::PersistentGraphHost::FrameCallback afterStop = [](org::RenderGraph&) {};
				REQUIRE(!host.TryPostOwnedPreparation(afterStop) && afterStop, "post-stop request stays with caller");
			}
			host.DestroyGraph();
		}
	}

	REQUIRE(vkDeviceWaitIdle(host.device) == VK_SUCCESS, "host device survives ORG and BasicRHI shutdown");
	DestroyHostBuffer(host, first);
	DestroyHostBuffer(host, second);
	vkDestroyDevice(host.device, nullptr);
	vkDestroyInstance(host.instance, nullptr);
	std::printf("PersistentVulkanHostTests: ok (%s, locks=%d)\n", async ? "async" : "sync", counters.locks.load());
	return 0;
}
