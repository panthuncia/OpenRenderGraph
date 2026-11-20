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
#include "TestSupport.h"
#include "Render/RenderGraph/ScheduleTrace.h"
#include "rhi_helpers.h"
#include "rhi_interop_vulkan.h"

#include <OpenRenderGraph/OpenRenderGraph.h>
#include "Render/RenderGraph/RenderGraph.h"
#include "Render/Runtime/RuntimeDevice.h"
#include "Render/Runtime/ScopedActiveGraphServices.h"
#include "Render/Runtime/DescriptorServiceAccess.h"
#include "Render/Runtime/ThreadPoolTaskService.h"
#include "OpenRenderGraph/GraphHost.h"
#include "Render/LatchBlock.h"
#include "Managers/Singletons/DescriptorHeapManager.h"
#include "RenderPasses/Base/TypedRenderGraphPass.h"
#include "Resources/Buffers/Buffer.h"
#include "Resources/Buffers/VersionedBuffer.h"
#include "Resources/ExternalBufferResource.h"
#include "Render/Runtime/UploadServiceAccess.h"
#include "Render/Runtime/IUploadService.h"
#include "Render/Runtime/StagedUploadBatch.h"
#include "Render/Runtime/IStatisticsService.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <Resources/ExternalTextureResource.h>
#include <Render/RenderGraph/PersistentAliasArena.h>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifndef ORG_TEST_PERSISTENT_WRITE_SPV
#error "ORG_TEST_PERSISTENT_WRITE_SPV must name the compiled SPIR-V of tests/shaders/PersistentHostWrite.hlsl"
#endif


namespace {
constexpr uint32_t kCount = 64;
constexpr uint64_t kBytes = kCount * sizeof(uint32_t);
// Counts the recordings of CopyToHostPass. In "external" mode the pass is recorded at execution, so each execution
// records it once more.
std::atomic<uint32_t> g_copyRecords{0};

struct HostDevice {
	VkInstance instance = VK_NULL_HANDLE;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkQueue queue = VK_NULL_HANDLE;
	VkQueue spare = VK_NULL_HANDLE;  // A second queue of the family, if it has one. It is offered to BasicRHI as a spare queue.
	uint32_t family = 0;
	std::vector<const char*> extensions;
	VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	VkPhysicalDeviceVulkan12Features vulkan12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	VkPhysicalDeviceVulkan13Features vulkan13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
	VkPhysicalDeviceDescriptorHeapFeaturesEXT descriptorHeap{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT};
	// Without descriptor heaps, BasicRHI uses its descriptor-buffer model. That model uses descriptor buffers with mutable
	// descriptors.
	VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptorBuffer{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT};
	VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT mutableDescriptors{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT};
	bool bufferModel = false;
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

// Returns 0 on success, 77 to skip the test, and 1 on failure.
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
	// Otherwise take the first device with descriptor buffers and mutable descriptors. ORG_TEST_DEVICE=<name substring>
	// picks among them.
	if (!host.physicalDevice) {
		const char* wanted = std::getenv("ORG_TEST_DEVICE");
		for (VkPhysicalDevice candidate : devices) {
			VkPhysicalDeviceProperties properties{};
			vkGetPhysicalDeviceProperties(candidate, &properties);
			if ((!wanted || std::strstr(properties.deviceName, wanted)) && HasDeviceExtension(candidate, VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME) &&
				HasDeviceExtension(candidate, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME)) {
				host.physicalDevice = candidate;
				host.bufferModel = true;
				std::printf("descriptor-buffer model on %s\n", properties.deviceName);
				break;
			}
		}
	}
	if (!host.physicalDevice) return 77;
	uint32_t familyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(host.physicalDevice, &familyCount, nullptr);
	std::vector<VkQueueFamilyProperties> families(familyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(host.physicalDevice, &familyCount, families.data());
	for (uint32_t i = 0; i < familyCount; ++i)
		if ((families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) { host.family = i; break; }

	host.extensions = host.bufferModel ? std::vector<const char*>{VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME}
		: std::vector<const char*>{VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, VK_KHR_MAINTENANCE_5_EXTENSION_NAME};
	host.vulkan12.bufferDeviceAddress = VK_TRUE;
	host.vulkan12.timelineSemaphore = VK_TRUE;
	host.vulkan12.descriptorIndexing = VK_TRUE;
	host.vulkan12.runtimeDescriptorArray = VK_TRUE;
	host.vulkan12.scalarBlockLayout = VK_TRUE;
	host.vulkan12.descriptorBindingPartiallyBound = VK_TRUE;
	host.vulkan12.descriptorBindingVariableDescriptorCount = VK_TRUE;
	host.vulkan12.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
	host.vulkan12.hostQueryReset = VK_TRUE;
	host.vulkan13.dynamicRendering = VK_TRUE;
	host.vulkan13.synchronization2 = VK_TRUE;
	host.vulkan13.maintenance4 = VK_TRUE;
	host.descriptorHeap.descriptorHeap = VK_TRUE;
	host.descriptorBuffer.descriptorBuffer = VK_TRUE;
	host.descriptorBuffer.pNext = &host.mutableDescriptors;
	host.mutableDescriptors.mutableDescriptorType = VK_TRUE;
	host.features.pNext = &host.vulkan12;
	host.vulkan12.pNext = &host.vulkan13;
	host.vulkan13.pNext = host.bufferModel ? static_cast<void*>(&host.descriptorBuffer) : &host.descriptorHeap;
	const float priorities[2] = {1.0f, 1.0f};
	VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	qci.queueFamilyIndex = host.family;
	qci.queueCount = host.family < familyCount && families[host.family].queueCount > 1 ? 2u : 1u;
	qci.pQueuePriorities = priorities;
	VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	dci.pNext = &host.features;
	dci.queueCreateInfoCount = 1;
	dci.pQueueCreateInfos = &qci;
	dci.enabledExtensionCount = static_cast<uint32_t>(host.extensions.size());
	dci.ppEnabledExtensionNames = host.extensions.data();
	if (vkCreateDevice(host.physicalDevice, &dci, nullptr, &host.device) != VK_SUCCESS) return 1;
	volkLoadDevice(host.device);
	vkGetDeviceQueue(host.device, host.family, 0, &host.queue);
	if (qci.queueCount > 1)
		vkGetDeviceQueue(host.device, host.family, 1, &host.spare);
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

struct WriteBindings { org::DeclaredViewToken input, target; };
struct WriteFrame {
	std::shared_ptr<const ComputeProgram> program;
	uint32_t inputIndex = 0;
	uint32_t targetIndex = 0;
	uint32_t count = kCount;
};

// A scene revision, passed as host data. Used by the "revision" mode. It holds:
// - how many elements the write pass writes. This count is baked into the pass's recording.
// - the version of the target that the pass writes. Inputs are immutable, so a grown target is a new version. The
//   revisions after the growth name that new version.
struct TestRevision final : org::IHostExecutionData, org::IResourceVersions {
	uint32_t count = kCount;
	bool skipWrite = false;  // "segment": the write pass is turned off for this revision (RenderGraphPass::SetSkipWhen)
	const void* versioned = nullptr;  // VersionedBuffer::Key
	std::shared_ptr<const org::BufferVersion> target;
	const void* TryGet(std::type_index type) const noexcept override {
		if (type == std::type_index(typeid(TestRevision))) return this;
		if (type == std::type_index(typeid(org::IResourceVersions))) return static_cast<const org::IResourceVersions*>(this);
		return nullptr;
	}
	std::shared_ptr<const org::BufferVersion> Find(const org::VersionedBuffer& buffer) const noexcept override {
		return buffer.Key() == versioned ? target : nullptr;
	}
};
uint32_t RevisionCount(const org::PassPrepareContext& preparation) {
	const auto* revision = preparation.preparationData ? preparation.preparationData->Get<TestRevision>() : nullptr;
	return revision ? revision->count : kCount;
}

// Reads the host-uploaded value (Pre-segment upload pass) and writes value + index.
class WritePass final : public org::TypedRenderGraphPass<WritePass, WriteFrame, WriteBindings> {
public:
	WritePass(std::shared_ptr<org::Buffer> input, std::shared_ptr<org::Buffer> target, std::shared_ptr<const ComputeProgram> program,
		std::shared_ptr<const org::VersionedBuffer> versions = {}, org::QueueKind queue = org::QueueKind::Graphics,
		std::shared_ptr<org::ExternalTextureResource> hostTexture = {})
		: m_input(std::move(input)), m_target(std::move(target)), m_program(std::move(program)), m_versions(std::move(versions)), m_queue(queue),
		m_hostTexture(std::move(hostTexture)) {}
	WriteBindings Declare(org::PassBuilder& builder) {
		builder.PreferQueue(m_queue);
		// "stream home": a host-owned texture the write reads (it only needs the declaration), so it is taken onto the write's queue.
		if (m_hostTexture) builder.ShaderResource(m_hostTexture);
		if (m_versions) return {builder.ShaderResource(m_input).View(), builder.UnorderedAccess(*m_versions).View()};
		return {builder.ShaderResource(m_input).View(), builder.UnorderedAccess(m_target).View()};
	}
	WriteFrame Prepare(const WriteBindings& bindings, const org::PassPrepareContext& preparation) const {
		return {m_program, preparation.Resolve(bindings.input).index,
			preparation.Resolve(bindings.target).index, RevisionCount(preparation)};
	}
	// Revision sees no frame. It names only the pass's own state, so here the host-data count is the default.
	void Revision(std::vector<uint64_t>& out) const {
		out.push_back(reinterpret_cast<uintptr_t>(m_program.get()));
		out.push_back(kCount);
	}
	static void Record(const WriteBindings&, const WriteFrame& frame, org::PassRecordContext& recording) {
		auto& commands = recording.Commands();
		commands.BindLayout(frame.program->layout->GetHandle());
		commands.BindPipeline(frame.program->pipeline->GetHandle());
		const uint32_t constants[3] = {frame.targetIndex, frame.inputIndex, frame.count};
		commands.PushConstants(rhi::ShaderStage::Compute, 0, 0, 0, 3, constants);
		commands.Dispatch((frame.count + 63) / 64, 1, 1);
	}
private:
	std::shared_ptr<org::Buffer> m_input;
	std::shared_ptr<org::Buffer> m_target;
	std::shared_ptr<const ComputeProgram> m_program;
	std::shared_ptr<const org::VersionedBuffer> m_versions;
	org::QueueKind m_queue = org::QueueKind::Graphics;
	std::shared_ptr<org::ExternalTextureResource> m_hostTexture;
};

struct CopyBindings { org::ResourceBindingToken source, destination; };
class CopyToHostPass final : public org::TypedRenderGraphPass<CopyToHostPass, org::EmptyPassFrameData, CopyBindings> {
public:
	CopyToHostPass(std::shared_ptr<org::Buffer> source, std::shared_ptr<org::ExternalBufferResource> destination,
		std::shared_ptr<std::atomic<uint64_t>> destinationRevision, std::shared_ptr<const org::VersionedBuffer> versions = {},
		org::QueueKind queue = org::QueueKind::Graphics)
		: m_source(std::move(source)), m_destination(std::move(destination)), m_destinationRevision(std::move(destinationRevision)),
		m_versions(std::move(versions)), m_queue(queue) {}
	CopyBindings Declare(org::PassBuilder& builder) {
		builder.PreferQueue(m_queue);
		if (m_versions) {
			// Declare the buffer two ways, like a pass that reads a buffer both as indirect arguments and through its
			// address. The declarations merge into one requirement. The pass is still lowered as a group, which is
			// resolved again for each preparation.
			const std::span<const org::SrvView> noViews{};
			CopyBindings bindings{builder.CopySource(*m_versions), builder.CopyDestination(m_destination)};
			builder.ShaderResource(*m_versions, noViews);
			return bindings;
		}
		return {builder.CopySource(m_source), builder.CopyDestination(m_destination)};
	}
	void Revision(std::vector<uint64_t>& out) const {
		out.push_back(m_destinationRevision->load(std::memory_order_acquire));
	}
	static void Record(const CopyBindings& bindings, org::PassRecordContext& recording) {
		g_copyRecords.fetch_add(1, std::memory_order_relaxed);
		recording.Commands().CopyBufferRegion(recording.Resolve(bindings.destination).GetHandle(), 0,
			recording.Resolve(bindings.source).GetHandle(), 0, kBytes);
	}
private:
	std::shared_ptr<org::Buffer> m_source;
	std::shared_ptr<org::ExternalBufferResource> m_destination;
	std::shared_ptr<std::atomic<uint64_t>> m_destinationRevision;
	std::shared_ptr<const org::VersionedBuffer> m_versions;
	org::QueueKind m_queue = org::QueueKind::Graphics;
};

class HostExtension final : public org::RenderGraph::IRenderGraphExtension {
public:
	HostExtension(std::shared_ptr<org::Buffer> input, std::shared_ptr<org::Buffer> scratch,
		std::shared_ptr<org::ExternalBufferResource> output, std::shared_ptr<const ComputeProgram> program,
		std::shared_ptr<std::atomic<uint64_t>> outputRevision, std::shared_ptr<const org::VersionedBuffer> versions = {}, bool lateCopy = false,
		bool splitEpochs = false, bool computeWrite = false, std::shared_ptr<org::ExternalTextureResource> hostTexture = {})
		: m_input(std::move(input)), m_scratch(std::move(scratch)), m_output(std::move(output)), m_program(std::move(program)),
		m_outputRevision(std::move(outputRevision)), m_versions(std::move(versions)), m_lateCopy(lateCopy), m_splitEpochs(splitEpochs),
		m_computeWrite(computeWrite), m_hostTexture(std::move(hostTexture)) {}
	void PrepareForBuild(org::RenderGraph& graph) override {
		if (m_hostTexture) graph.RegisterResource(org::ResourceIdentifier("test.host-texture"), m_hostTexture);
		graph.RegisterResource(org::ResourceIdentifier("test.input"), m_input);
		graph.RegisterResource(org::ResourceIdentifier("test.scratch"), m_scratch);
		graph.RegisterResource(org::ResourceIdentifier("test.host-output"), m_output);
	}
	void GatherStructuralPasses(org::RenderGraph&, std::vector<org::RenderGraph::ExternalPassDesc>& out) override {
		// "stream": the write on the compute queue, so an external epoch's execution has a batch off the host's stream.
		const auto writeQueue = m_computeWrite ? org::QueueKind::Compute : org::QueueKind::Graphics;
		auto writePass = std::make_shared<WritePass>(m_input, m_scratch, m_program, m_versions, writeQueue, m_hostTexture);
		// A revision may turn the write off: it then records only its barriers.
		writePass->SetSkipWhen([](const org::IHostExecutionData* data) {
			const auto* revision = data ? data->Get<TestRevision>() : nullptr;
			return revision && revision->skipWrite;
		});
		auto write = org::RenderGraph::ExternalPassDesc::Compute("test.write", std::static_pointer_cast<org::RenderPass>(writePass)).PreferQueue(writeQueue);
		// "stream": the copy on the copy queue, which the device aliases to the graphics queue (the host's stream's): in the stream.
		const auto copyQueue = m_computeWrite ? org::QueueKind::Copy : org::QueueKind::Graphics;
		auto copy = org::RenderGraph::ExternalPassDesc::Copy("test.copy-to-host",
			std::static_pointer_cast<org::RenderPass>(std::make_shared<CopyToHostPass>(m_scratch, m_output, m_outputRevision, m_versions, copyQueue)))
			.PreferQueue(copyQueue);
		// "split": the write in epoch 0, the copy in epoch 1, so each is a graph of its own and the scratch buffer goes from one
		// to the other through the runtime's ledgers.
		if (m_splitEpochs) {
			write.Epoch(0);
			copy.Epoch(1);
		}
		out.push_back(std::move(write));
		// In "external" mode, the host's execution of the epoch records this pass, after the write pass's secondary command
		// buffer.
		if (m_lateCopy) copy.RecordAtExecution();
		out.push_back(std::move(copy));
	}
private:
	std::shared_ptr<org::Buffer> m_input;
	std::shared_ptr<org::Buffer> m_scratch;
	std::shared_ptr<org::ExternalBufferResource> m_output;
	std::shared_ptr<const ComputeProgram> m_program;
	std::shared_ptr<std::atomic<uint64_t>> m_outputRevision;
	std::shared_ptr<const org::VersionedBuffer> m_versions;
	bool m_lateCopy = false;
	bool m_splitEpochs = false;
	bool m_computeWrite = false;
	std::shared_ptr<org::ExternalTextureResource> m_hostTexture;
};

// "arena": an epoch's write into a scratch buffer of its own, in a manual alias pool, and its copy to a host buffer. Two
// epochs, each a graph of its own, place their scratch buffers at the same offset of the host's shared arena's pool.
class ArenaExtension final : public org::RenderGraph::IRenderGraphExtension {
public:
	struct Epoch {
		uint32_t epoch = 0;
		std::shared_ptr<org::Buffer> input, scratch;
		std::shared_ptr<org::ExternalBufferResource> output;
		org::QueueKind writeQueue = org::QueueKind::Graphics;
	};
	ArenaExtension(std::vector<Epoch> epochs, std::shared_ptr<const ComputeProgram> program, std::shared_ptr<std::atomic<uint64_t>> outputRevision)
		: m_epochs(std::move(epochs)), m_program(std::move(program)), m_outputRevision(std::move(outputRevision)) {}
	void PrepareForBuild(org::RenderGraph& graph) override {
		for (const auto& epoch : m_epochs) {
			const auto suffix = std::to_string(epoch.epoch);
			graph.RegisterResource(org::ResourceIdentifier("test.arena.input." + suffix), epoch.input);
			graph.RegisterResource(org::ResourceIdentifier("test.arena.scratch." + suffix), epoch.scratch);
			graph.RegisterResource(org::ResourceIdentifier("test.arena.output." + suffix), epoch.output);
		}
	}
	void GatherStructuralPasses(org::RenderGraph&, std::vector<org::RenderGraph::ExternalPassDesc>& out) override {
		for (const auto& epoch : m_epochs) {
			const auto suffix = std::to_string(epoch.epoch);
			out.push_back(org::RenderGraph::ExternalPassDesc::Compute("test.arena.write." + suffix,
				std::static_pointer_cast<org::RenderPass>(std::make_shared<WritePass>(epoch.input, epoch.scratch, m_program, nullptr, epoch.writeQueue)))
				.PreferQueue(epoch.writeQueue).Epoch(epoch.epoch));
			out.push_back(org::RenderGraph::ExternalPassDesc::Copy("test.arena.copy." + suffix,
				std::static_pointer_cast<org::RenderPass>(std::make_shared<CopyToHostPass>(epoch.scratch, epoch.output, m_outputRevision)))
				.Epoch(epoch.epoch));
		}
	}
private:
	std::vector<Epoch> m_epochs;
	std::shared_ptr<const ComputeProgram> m_program;
	std::shared_ptr<std::atomic<uint64_t>> m_outputRevision;
};

// "built", staged sets: a write into a versioned buffer, in one epoch only. GraphHost::EpochsDeclaring finds that epoch for it.
class VersionedWriteExtension final : public org::RenderGraph::IRenderGraphExtension {
public:
	VersionedWriteExtension(uint32_t epoch, std::shared_ptr<org::Buffer> input, std::shared_ptr<const org::VersionedBuffer> target,
		std::shared_ptr<const ComputeProgram> program)
		: m_epoch(epoch), m_input(std::move(input)), m_target(std::move(target)), m_program(std::move(program)) {}
	void PrepareForBuild(org::RenderGraph& graph) override { graph.RegisterResource(org::ResourceIdentifier("test.versioned.input"), m_input); }
	void GatherStructuralPasses(org::RenderGraph&, std::vector<org::RenderGraph::ExternalPassDesc>& out) override {
		out.push_back(org::RenderGraph::ExternalPassDesc::Compute("test.versioned.write",
			std::static_pointer_cast<org::RenderPass>(std::make_shared<WritePass>(m_input, nullptr, m_program, m_target)))
			.Epoch(m_epoch));
	}
private:
	uint32_t m_epoch;
	std::shared_ptr<org::Buffer> m_input;
	std::shared_ptr<const org::VersionedBuffer> m_target;
	std::shared_ptr<const ComputeProgram> m_program;
};

// Executes an external execution's segments, each in its own submission of a primary of its own (a submission per segment
// meets every segment's split). The caller waits for the device before executing the next.
struct SegmentExecutor {
	rhi::Device device;
	std::vector<std::pair<rhi::CommandAllocatorPtr, rhi::CommandListPtr>> primaries;
	bool Execute(org::ExternalEpochExecution& a_execution) {
		auto queue = device.GetQueue(rhi::QueueKind::Graphics);
		for (size_t index = 0; index < a_execution.segments.size(); ++index) {
			if (primaries.size() <= index) {
				auto& [allocator, primary] = primaries.emplace_back();
				if (device.CreateCommandAllocator(rhi::QueueKind::Graphics, allocator) != rhi::Result::Ok
					|| device.CreateCommandList(rhi::QueueKind::Graphics, allocator.Get(), primary) != rhi::Result::Ok)
					return false;
				primary->End();
			}
			auto& [allocator, primary] = primaries[index];
			allocator->Recycle();
			primary->Recycle(allocator.Get());
			a_execution.Execute(index, primary.Get());
			if (primary->EndChecked() != rhi::Result::Ok) return false;
			auto list = primary.Get();
			const auto& segment = a_execution.segments[index];
			auto waits = segment.waits;
			auto signals = segment.signals;
			if (queue.Submit({&list, 1}, {{waits.data(), static_cast<uint32_t>(waits.size())},
					{signals.data(), static_cast<uint32_t>(signals.size())}}) != rhi::Result::Ok)
				return false;
		}
		return true;
	}
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
int TestRebindingGpu(rhi::Device device, const std::shared_ptr<org::Runtime>& runtime);
int TestPipelines(rhi::Device device);
int TestContributors(rhi::Device device, const std::function<std::shared_ptr<org::Resource>()>& importOutput, const uint32_t* mapped,
	const char* spirvPath,
	rhi::PipelineLayoutHandle layout);

static int Run(int argc, char** argv) {
	// "reuse": async epochs that record replayable recordings (GraphHost::Desc::reuseRecordings). A live epoch's recording is
	// still used once: every run records.
	// "external-reuse": "external" with replayable recordings. The late pass is replayable.
	const bool externalReuse = argc > 1 && std::strcmp(argv[1], "external-reuse") == 0;
	const bool reuse = externalReuse || (argc > 1 && std::strcmp(argv[1], "reuse") == 0);
	// "revision": a revision-driven epoch, set through the revisionEpochs of SetAsyncEpochs. It is recorded ahead for
	// each revision (RequestEpochRecording) and submitted with that recording.
	// "segment": "revision" with the epoch closed by SetClosedEpochs (the host's other executions open), its copies taken by the
	// caller's frame slot (SetEpochCopy, EpochCopyRun), and a revision that turns the write pass off (SetSkipWhen).
	const bool segment = argc > 1 && std::strcmp(argv[1], "segment") == 0;
	const bool revision = segment || (argc > 1 && std::strcmp(argv[1], "revision") == 0);
	// "external": an async epoch that the test executes itself (GraphHost::SetExternalEpochs). Its copy pass is
	// recorded at execution, into a primary command buffer the test owns. It is submitted with the execution's waits
	// and signals.
	const bool external = externalReuse || (argc > 1 && std::strcmp(argv[1], "external") == 0);
	const bool async = reuse || revision || external || (argc > 1 && std::strcmp(argv[1], "async") == 0);
	// "contributor": a contributor registered through the C table (ContributorHostTests.cpp) instead of the graph scenario.
	const bool contributor = argc > 1 && std::strcmp(argv[1], "contributor") == 0;
	// "rebinding": slots that start or stop sharing a backing (RebindingGpuTests.cpp) instead of the graph scenario.
	const bool rebinding = argc > 1 && std::strcmp(argv[1], "rebinding") == 0;
	// "pipelines": input layouts, PipelinePermutations and GenerateMipsPass (PipelineVulkanTests.cpp) instead of the graph scenario.
	const bool pipelines = argc > 1 && std::strcmp(argv[1], "pipelines") == 0;
	// "arena": two epochs whose transient scratch buffers share the host's alias arena (the second's write on the compute queue
	// when the family has a second queue), against two of their own.
	const bool arena = argc > 1 && std::strcmp(argv[1], "arena") == 0;
	// "split": a host of two epochs, each its own graph: the write pass in the first and the copy to the host buffer in the
	// second (SplitEpochsScenario).
	const bool split = argc > 1 && std::strcmp(argv[1], "split") == 0;
	// "built": application-driven builds (GraphHost::BuildEpochGraph, AttachEpochGraph): a build on a worker thread while both
	// epochs submit, attached between frames; a failed build; stale builds rejected; and stale alias placements reported
	// (Desc::reportStaleAliasPlacements) instead of rebuilt.
	const bool built = argc > 1 && std::strcmp(argv[1], "built") == 0;
	// "stream": an external epoch with its write on the compute queue (the family's second queue): batches off the host's stream,
	// and the stream's segments.
	const bool stream = argc > 1 && std::strcmp(argv[1], "stream") == 0;
	// "segment-external": what a host's stage in another API's stream needs (CS's guest stages in DXVK's stream). One revision-driven
	// epoch that is also external: open executions (entry barriers planned from the ledgers at each submission), external home states
	// with a host-owned texture the write reads, the epoch's transients in alias pools of its own (SetPrivateAliasEpochs), each frame's
	// values staged in its slot's staging run and copied into the pinned copy's run at the submission, the copy pass recorded at
	// execution, each execution executed on a thread other than the submitting one (the host's worker), the host using the texture in
	// its home state between the blocks, a description that turns the write off, and a staged build switched in.
	const bool segmentExternal = argc > 1 && std::strcmp(argv[1], "segment-external") == 0;
	// "leases": an external async epoch that runs many times a frame, its executions held, as a host's stream holds them, and
	// executed only at the frame's end, before SealSubmitted. Each run takes a run lease of its own instead of waiting for a run of
	// the same frame, and the leases retire once their runs complete. Also checks the frame arena's regions (LatchBlock).
	const bool leases = argc > 1 && (std::strcmp(argv[1], "leases") == 0 || std::strcmp(argv[1], "leases-wide") == 0);
	// "leases-wide": the same, with 260 runs a frame, so lease indices pass 255, and with replayable recordings.
	const bool wide = argc > 1 && std::strcmp(argv[1], "leases-wide") == 0;
	// "open" after the mode: async epochs on open executions, admitted at submission (no closed executions).
	const bool open = argc > 2 && std::strcmp(argv[2], "open") == 0;
	// "stream open home": the write also reads a host-owned texture, under external home states.
	const bool home = open && argc > 3 && std::strcmp(argv[3], "home") == 0;
	// "inline" anywhere (alone: the synchronous frames): the copy pass is recorded at execution, inline (GraphHost::
	// SetInlineAtExecution). Each submission records it into a primary list of the graph's and submits it in its place. Nothing
	// is taken or executed by the test.
	bool inlineLate = false;
	for (int arg = 1; arg < argc; ++arg) inlineLate = inlineLate || std::strcmp(argv[arg], "inline") == 0;
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
		const bool spareQueue = host.spare != VK_NULL_HANDLE;
		const rhi::vulkan::AdoptedQueue spare{host.spare, host.family, 1};
		if (stream) {
			if (!host.spare) {
				std::puts("SKIP: stream needs a second queue in the family");
				return 77;
			}
			info.queues[1] = spare;  // the compute queue
		} else if (arena && host.spare) {
			info.queues[1] = spare;  // the compute queue
		} else if (host.spare) {
			info.spareQueues = &spare;
			info.spareQueueCount = 1;
		}
		info.submissionHooks = {&counters, &Lock, &Unlock};
		rhi::DevicePtr device;
		REQUIRE(rhi::vulkan::AdoptVulkanDevice(info, device) == rhi::Result::Ok, "AdoptVulkanDevice");
		REQUIRE(TestFrameRetirement(device.Get(), rhi::Backend::Vulkan) == 0, "descriptor and frame retirement");
		if (pipelines) {
			const int line = TestPipelines(device.Get());
			if (line) std::printf("FAIL: pipelines (line %d)\n", line);
			REQUIRE(line == 0, "pipelines");
		} else if (rebinding) {
			auto runtime = org::Runtime::Create({.device = device.Get(), .backend = rhi::Backend::Vulkan,
				.tasks = std::make_shared<org::runtime::ThreadPoolTaskService>(2)});
			const int line = TestRebindingGpu(device.Get(), runtime);
			runtime.reset();
			if (line) std::printf("FAIL: rebinding (line %d)\n", line);
			REQUIRE(line == 0, "rebinding");
		} else if (split) {
			auto runtime = org::Runtime::Create({.device = device.Get(), .backend = rhi::Backend::Vulkan, .slots = 4,
				.tasks = std::make_shared<org::runtime::ThreadPoolTaskService>(2)});
			{
				org::GraphHost host({.runtime = runtime, .queueBoundary = {.entry = true, .exit = true}, .framesInFlight = 2, .epochOrder = {0, 1},
					.closedExecutions = !open});
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
				const auto upload = [&](org::RenderGraph&) {
					BUFFER_UPLOAD(&value, sizeof(value), org::runtime::UploadTarget::FromShared(input), 0);
				};
				auto outputRevision = std::make_shared<std::atomic<uint64_t>>(1);
				host.AddExtension("test.host", [&] {
					return std::make_unique<HostExtension>(input, scratch, output, program, outputRevision, nullptr, false, true);
				});
				// Synchronous frames: epoch 0 writes, epoch 1 copies what it wrote.
				for (uint32_t frame = 0; frame < 4; ++frame) {
					value = 40000u + 10u * frame;
					host.ExecuteFrame(nullptr, upload, 0);
					host.ExecuteFrame(nullptr, {}, 1);
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "split epochs, synchronous: the second copies what the first wrote");
				}
				REQUIRE(host.GraphCount() == 2 && host.Graph(0) && host.Graph(1) && host.Graph(0) != host.Graph(1), "a graph per epoch");
				// Async epochs, then one epoch built again: the other keeps its graph and its build.
				host.SetAsyncEpochs({0, 1});
				for (uint32_t frame = 0; frame < 8; ++frame) {
					if (frame == 4) host.RequestRebuild(1);
					const auto builds0 = host.BuildGeneration(0), builds1 = host.BuildGeneration(1);
					auto* graph0 = host.Graph(0);
					value = 41000u + 10u * frame;
					host.SubmitEpoch(0, upload);
					host.SubmitEpoch(1);
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "split epochs, async: the second copies what the first wrote");
					REQUIRE(host.BuildGeneration(0) == builds0 && host.Graph(0) == graph0, "the first epoch keeps its graph");
					if (frame == 4) REQUIRE(host.BuildGeneration(1) == builds1 + 1, "only the second epoch was built again");
				}
				const auto stats = host.TakeAsyncStats();
				std::printf("split (%s): %llu submissions, %llu waited, %llu prepared again, %llu rebuilds\n", open ? "open" : "closed",
					static_cast<unsigned long long>(stats.submitted), static_cast<unsigned long long>(stats.waited),
					static_cast<unsigned long long>(stats.stale), static_cast<unsigned long long>(stats.rebuilds));
				REQUIRE(stats.rebuilds == 1, "one epoch rebuild");
				REQUIRE(!counters.unbalanced && counters.locks.load() == counters.unlocks.load(), "balanced host queue locks");
				host.SetAsyncEpochs({});
				host.DestroyGraph();
			}
			runtime.reset();
		} else if (built) {
			// "built": application-driven builds (GraphHost::BuildEpochGraph, AttachEpochGraph) on a host of two async epochs.
			auto runtime = org::Runtime::Create({.device = device.Get(), .backend = rhi::Backend::Vulkan, .slots = 4,
				.tasks = std::make_shared<org::runtime::ThreadPoolTaskService>(2)});
			{
				org::GraphHost host({.runtime = runtime, .queueBoundary = {.entry = true, .exit = true}, .framesInFlight = 2, .epochOrder = {0, 1},
					.closedExecutions = !open, .reportStaleAliasPlacements = true});
				// Only the test's attaches change graphs: a request alone builds nothing.
				host.SetExplicitBuilds(true);
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
				const auto upload = [&](org::RenderGraph&) {
					BUFFER_UPLOAD(&value, sizeof(value), org::runtime::UploadTarget::FromShared(input), 0);
				};
				auto outputRevision = std::make_shared<std::atomic<uint64_t>>(1);
				// The factory runs on whichever thread builds. What it reads is set up before and never changes.
				host.AddExtension("test.host", [=] {
					return std::make_unique<HostExtension>(input, scratch, output, program, outputRevision, nullptr, false, true);
				});
				REQUIRE(host.BuildIfRequested(), "the first builds, synchronous");
				host.SetAsyncEpochs({0, 1});
				uint32_t frame = 0;
				const auto run = [&]() -> int {
					value = 43000u + 10u * frame++;
					host.SubmitEpoch(0, upload);
					host.SubmitEpoch(1);
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "built: the second epoch copies what the first wrote");
					return 0;
				};
				for (int i = 0; i < 3; ++i)
					if (const int failed = run()) return failed;

				// 1. A build on a worker thread while both epochs keep submitting, attached at a build point between frames.
				{
					auto* graph0 = host.Graph(0);
					auto* graph1 = host.Graph(1);
					const auto builds0 = host.BuildGeneration(0), builds1 = host.BuildGeneration(1);
					host.RequestRebuild(1, "test: built on a worker");
					std::atomic<bool> done{false};
					std::unique_ptr<org::GraphHost::EpochGraph> next;
					std::thread worker([&] {
						next = host.BuildEpochGraph(1, "test: built on a worker");
						done.store(true, std::memory_order_release);
					});
					uint32_t framesDuring = 0;
					while (!done.load(std::memory_order_acquire)) {
						if (const int failed = run()) {
							worker.join();
							return failed;
						}
						++framesDuring;
					}
					worker.join();
					REQUIRE(next && next->Built(), "the worker's build succeeded");
					REQUIRE(host.Graph(1) == graph1 && host.BuildGeneration(1) == builds1, "the epoch runs its old graph until the attach");
					const auto attached = host.AttachEpochGraph(next);
					REQUIRE(attached.attached && !next, "the built graph is attached");
					REQUIRE(host.Graph(1) != graph1 && host.BuildGeneration(1) == builds1 + 1, "the epoch runs the new graph");
					REQUIRE(host.Graph(0) == graph0 && host.BuildGeneration(0) == builds0, "the other epoch keeps its graph");
					REQUIRE(!host.RebuildRequested(1), "the attach answers the request made before its build");
					std::printf("built: %u frames submitted during the worker's build\n", framesDuring);
					for (int i = 0; i < 3; ++i)
						if (const int failed = run()) return failed;
				}

				// 2. A failed build: nothing to attach, and the epoch keeps its graph.
				{
					auto* graph1 = host.Graph(1);
					host.AddExtension("test.fails", []() -> std::unique_ptr<org::RenderGraph::IRenderGraphExtension> {
						throw std::runtime_error("test: this extension cannot be made");
					}, {1});
					std::unique_ptr<org::GraphHost::EpochGraph> failed;
					std::thread worker([&] { failed = host.BuildEpochGraph(1, "test: a failing build"); });
					for (int i = 0; i < 2; ++i)
						if (const int error = run()) {
							worker.join();
							return error;
						}
					worker.join();
					REQUIRE(failed && !failed->Built() && !failed->Error().empty(), "the build reports its failure");
					const auto attached = host.AttachEpochGraph(failed);
					REQUIRE(!attached.attached && !attached.reason.empty() && failed, "a failed build is not attached");
					failed.reset();
					REQUIRE(host.Graph(1) == graph1 && !host.EpochFailed(1), "the epoch keeps its old graph");
					host.RemoveExtension("test.fails");
					for (int i = 0; i < 2; ++i)
						if (const int error = run()) return error;
				}

				// 3. Stale builds are rejected: one whose epoch's extensions changed since it started, and one older than the graph attached.
				{
					auto stale = host.BuildEpochGraph(1, "test: built before a change");
					host.AddExtension("test.nothing", [] { return std::unique_ptr<org::RenderGraph::IRenderGraphExtension>{}; }, {1});
					auto rejected = host.AttachEpochGraph(stale);
					REQUIRE(!rejected.attached && stale, "a graph built before its epoch's extensions changed is rejected");
					std::printf("built: rejected: %s\n", rejected.reason.c_str());
					stale.reset();
					auto older = host.BuildEpochGraph(1, "test: the older build");
					auto newer = host.BuildEpochGraph(1, "test: the newer build");
					REQUIRE(older->Built() && newer->Built(), "two builds of one epoch");
					REQUIRE(host.AttachEpochGraph(newer).attached, "the newer build is attached");
					rejected = host.AttachEpochGraph(older);
					REQUIRE(!rejected.attached && older, "the older build is rejected after the newer one");
					std::printf("built: rejected: %s\n", rejected.reason.c_str());
					older.reset();
					for (int i = 0; i < 2; ++i)
						if (const int error = run()) return error;
				}
				const auto stats = host.TakeAsyncStats();
				std::printf("built (%s): %llu submissions, %llu waited, %llu prepared again, %llu replaced graphs\n", open ? "open" : "closed",
					static_cast<unsigned long long>(stats.submitted), static_cast<unsigned long long>(stats.waited),
					static_cast<unsigned long long>(stats.stale), static_cast<unsigned long long>(stats.rebuilds));
				REQUIRE(stats.rebuilds == 2, "two graphs replaced by attaches");

				// 5. Staged sets (GraphHost::AttachStagedEpochGraphs) and the closure query (EpochsDeclaring).
				{
					const auto refusalsBefore = host.AttachStatsTotal();
					const auto stageBoth = [&](const char* why) -> bool {
						auto zero = host.BuildEpochGraph(0, why);
						auto one = host.BuildEpochGraph(1, why);
						return zero->Built() && one->Built() && host.StageEpochGraph(zero).staged && host.StageEpochGraph(one).staged;
					};
					// a. Two staged epochs attach together.
					REQUIRE(stageBoth("test: a staged set"), "both epochs staged");
					const auto staged0 = host.StagedGeneration(0), staged1 = host.StagedGeneration(1);
					const uint32_t both[] = {0, 1};
					const auto attached = host.AttachStagedEpochGraphs(both);
					REQUIRE(attached.attached, "a staged set attaches");
					REQUIRE(host.BuildGeneration(0) == staged0 && host.BuildGeneration(1) == staged1, "both epochs run their staged builds");
					REQUIRE(host.StagedGeneration(0) == 0 && host.StagedGeneration(1) == 0, "nothing is left staged");
					for (int i = 0; i < 2; ++i)
						if (const int failed = run()) return failed;
					// An empty set attaches nothing. A set that names an epoch twice, or one with nothing staged, is refused.
					REQUIRE(host.AttachStagedEpochGraphs({}).attached, "an empty set");
					const uint32_t twice[] = {0, 0};
					REQUIRE(host.AttachStagedEpochGraphs(twice).refusal == org::GraphHost::AttachRefusal::Duplicate, "an epoch named twice");
					REQUIRE(host.AttachStagedEpochGraphs(both).refusal == org::GraphHost::AttachRefusal::NotStaged, "nothing staged");

					// b. A configuration change of one epoch refuses the whole set and drops neither staged graph.
					REQUIRE(stageBoth("test: a set one change spoils"), "both epochs staged again");
					const auto kept0 = host.StagedGeneration(0), kept1 = host.StagedGeneration(1);
					const auto running0 = host.BuildGeneration(0), running1 = host.BuildGeneration(1);
					auto target = org::Buffer::CreateSharedUnmaterialized(rhi::HeapType::DeviceLocal, kBytes, true);
					org::BufferBase::DescriptorRequirements targetViews{};
					targetViews.createUAV = true;
					targetViews.uavDesc = {.dimension = rhi::UavDim::Buffer, .formatOverride = rhi::Format::R32_Typeless,
						.buffer = {.kind = rhi::BufferViewKind::Raw, .firstElement = 0, .numElements = kCount}};
					target->SetDescriptorRequirements(targetViews);
					target->SetName("test.versioned.target");
					target->Materialize();
					auto versioned = org::VersionedBuffer::Create(target);
					auto unrelated = org::VersionedBuffer::Create(org::Buffer::CreateShared(rhi::HeapType::DeviceLocal, kBytes));
					auto versionedInput = org::Buffer::CreateSharedUnmaterialized(rhi::HeapType::DeviceLocal, 16, false);
					org::BufferBase::DescriptorRequirements versionedInputViews{};
					versionedInputViews.createSRV = true;
					versionedInputViews.srvDesc = {.dimension = rhi::SrvDim::Buffer, .formatOverride = rhi::Format::R32_Typeless,
						.buffer = {.kind = rhi::BufferViewKind::Raw, .firstElement = 0, .numElements = 4}};
					versionedInput->SetDescriptorRequirements(versionedInputViews);
					versionedInput->SetName("test.versioned.input");
					host.AddExtension("test.versioned", [=] { return std::make_unique<VersionedWriteExtension>(1, versionedInput, versioned, program); }, {1});
					const auto refused = host.AttachStagedEpochGraphs(both);
					std::printf("built: a staged set refused: %s (epoch %u)\n", refused.reason.c_str(), refused.epoch);
					REQUIRE(!refused.attached && refused.refusal == org::GraphHost::AttachRefusal::EpochConfiguration && refused.epoch == 1,
						"a set with one epoch's configuration changed is refused, naming that epoch");
					REQUIRE(host.StagedGeneration(0) == kept0 && host.StagedGeneration(1) == kept1, "a refused set keeps both staged");
					REQUIRE(host.BuildGeneration(0) == running0 && host.BuildGeneration(1) == running1, "a refused set attaches nothing");

					// c. EpochsDeclaring: epoch 1's new build declares the versioned buffer while it is staged, and after its attach.
					const org::IVersionedResource* slots[] = {versioned.get()};
					const org::IVersionedResource* other[] = {unrelated.get(), nullptr};
					REQUIRE(host.EpochsDeclaring(slots).empty(), "no graph declares the buffer before a build holds the new pass");
					auto rebuilt = host.BuildEpochGraph(1, "test: a build with the versioned write");
					REQUIRE(rebuilt->Built() && host.StageEpochGraph(rebuilt).staged, "epoch 1 staged again with the new pass");
					REQUIRE(host.EpochsDeclaring(slots) == std::vector<uint32_t>{1}, "a staged graph's declaration is found");
					REQUIRE(host.EpochsDeclaring(other).empty(), "a buffer no graph declares is in no epoch");
					REQUIRE(host.AttachStagedEpochGraphs(both).attached, "the set attaches once both builds are current");
					REQUIRE(host.EpochsDeclaring(slots) == std::vector<uint32_t>{1}, "an attached graph's declaration is found");
					for (int i = 0; i < 3; ++i)
						if (const int failed = run()) return failed;
					REQUIRE(host.EpochsDeclaring(slots) == std::vector<uint32_t>{1}, "the declaration stays after preparations");
					const auto totals = host.AttachStatsTotal();
					const auto refusals = [&](org::GraphHost::AttachRefusal reason) {
						const auto index = static_cast<size_t>(reason);
						return totals.refusals[index] - refusalsBefore.refusals[index];
					};
					REQUIRE(totals.setsAttached - refusalsBefore.setsAttached == 2, "two sets attached");
					REQUIRE(refusals(org::GraphHost::AttachRefusal::EpochConfiguration) == 1 && refusals(org::GraphHost::AttachRefusal::Duplicate) == 1
							&& refusals(org::GraphHost::AttachRefusal::NotStaged) == 1,
						"refusals are counted by reason");
				}
				REQUIRE(!counters.unbalanced && counters.locks.load() == counters.unlocks.load(), "balanced host queue locks");
				host.SetAsyncEpochs({});
				host.DestroyGraph();
			}

			// 4. Desc::reportStaleAliasPlacements: a pool another epoch's build grew is reported, not rebuilt at the build point.
			{
				org::GraphHost host({.runtime = runtime, .queueBoundary = {.entry = true, .exit = true}, .framesInFlight = 2, .epochOrder = {0, 1},
					.reportStaleAliasPlacements = true});
				auto program = CreateProgram(device.Get());
				REQUIRE(program, "compute program");
				const auto makeScratch = [&](uint32_t epoch, uint64_t bytes) {
					auto scratch = org::Buffer::CreateSharedUnmaterialized(rhi::HeapType::DeviceLocal, bytes, true);
					org::BufferBase::DescriptorRequirements requirements{};
					requirements.createUAV = true;
					requirements.uavDesc = {.dimension = rhi::UavDim::Buffer, .formatOverride = rhi::Format::R32_Typeless,
						.buffer = {.kind = rhi::BufferViewKind::Raw, .firstElement = 0, .numElements = kCount}};
					scratch->SetDescriptorRequirements(requirements);
					scratch->SetName("test.arena.scratch." + std::to_string(epoch));
					scratch->SetAllowAlias(true);
					scratch->SetAliasingPool(0xA5);
					return scratch;
				};
				std::vector<ArenaExtension::Epoch> epochs;
				std::array<uint32_t, 2> values{};
				for (uint32_t epoch = 0; epoch < 2; ++epoch) {
					auto& entry = epochs.emplace_back();
					entry.epoch = epoch;
					entry.scratch = makeScratch(epoch, kBytes);
					entry.input = org::Buffer::CreateSharedUnmaterialized(rhi::HeapType::DeviceLocal, 16, false);
					org::BufferBase::DescriptorRequirements inputViews{};
					inputViews.createSRV = true;
					inputViews.srvDesc = {.dimension = rhi::SrvDim::Buffer, .formatOverride = rhi::Format::R32_Typeless,
						.buffer = {.kind = rhi::BufferViewKind::Raw, .firstElement = 0, .numElements = 4}};
					entry.input->SetDescriptorRequirements(inputViews);
					entry.input->SetName("test.arena.input." + std::to_string(epoch));
					entry.output = ImportHostBuffer(device.Get(), epoch == 0 ? first : second, epoch == 0 ? "test.arena.output.0" : "test.arena.output.1");
					REQUIRE(entry.output, "import host buffer");
				}
				const auto upload = [&](uint32_t epoch) {
					return [&, epoch](org::RenderGraph&) {
						BUFFER_UPLOAD(&values[epoch], sizeof(uint32_t), org::runtime::UploadTarget::FromShared(epochs[epoch].input), 0);
					};
				};
				auto outputRevision = std::make_shared<std::atomic<uint64_t>>(1);
				// The factory copies the epochs as they are when a build starts.
				host.AddExtension("test.arena", [&] { return std::make_unique<ArenaExtension>(epochs, program, outputRevision); });
				uint32_t frame = 0;
				const auto run = [&]() -> int {
					values = {90000u + 10u * frame, 95000u + 10u * frame};
					++frame;
					host.SubmitEpoch(0, upload(0));
					host.SubmitEpoch(1, upload(1));
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, values[0]) && Matches(second, values[1]),
						"each epoch copies what it wrote into its scratch buffer");
					return 0;
				};
				host.SetAsyncEpochs({0, 1});
				for (int i = 0; i < 3; ++i)
					if (const int failed = run()) return failed;
				REQUIRE(host.StaleAliasPlacementEpochs().empty(), "every placement current");
				// Epoch 1's scratch buffer grows: its new graph grows the shared pool, and epoch 0's placement is in the old generation.
				epochs[1].scratch = makeScratch(1, 4 * kBytes);
				auto grown = host.BuildEpochGraph(1, "test: a larger scratch buffer");
				REQUIRE(grown->Built() && host.AttachEpochGraph(grown).attached, "the larger graph is attached");
				// A detached build places its transients at its first preparation after the attach.
				if (const int failed = run()) return failed;
				const auto stale = host.StaleAliasPlacementEpochs();
				REQUIRE(stale.size() == 1 && stale.front() == 0, "the other epoch's placements are reported");
				const auto builds0 = host.BuildGeneration(0);
				(void)host.BuildIfRequested();
				REQUIRE(host.BuildGeneration(0) == builds0 && !host.RebuildRequested(0), "the build point does not rebuild the reported epoch");
				for (int i = 0; i < 2; ++i)
					if (const int failed = run()) return failed;
				auto placed = host.BuildEpochGraph(0, "test: placed again in the grown pool");
				REQUIRE(placed->Built() && host.AttachEpochGraph(placed).attached, "the reported epoch is built again by the application");
				if (const int failed = run()) return failed;
				REQUIRE(host.StaleAliasPlacementEpochs().empty(), "every placement current again");
				for (int i = 0; i < 2; ++i)
					if (const int failed = run()) return failed;
				std::puts("built: stale alias placements reported and placed again");
				REQUIRE(!counters.unbalanced && counters.locks.load() == counters.unlocks.load(), "balanced host queue locks");
				host.SetAsyncEpochs({});
				host.DestroyGraph();
			}
			runtime.reset();
		} else if (arena) {
			auto runtime = org::Runtime::Create({.device = device.Get(), .backend = rhi::Backend::Vulkan, .slots = 4,
				.tasks = std::make_shared<org::runtime::ThreadPoolTaskService>(2)});
			const auto writeQueue = spareQueue ? org::QueueKind::Compute : org::QueueKind::Graphics;
			for (const bool shared : {true, false}) {
				org::GraphHost host({.runtime = runtime, .queueBoundary = {.entry = true, .exit = true}, .framesInFlight = 2, .epochOrder = {0, 1},
					.sharedAliasArena = shared});
				auto program = CreateProgram(device.Get());
				REQUIRE(program, "compute program");
				std::vector<ArenaExtension::Epoch> epochs;
				std::array<uint32_t, 2> values{};
				for (uint32_t epoch = 0; epoch < 2; ++epoch) {
					auto& entry = epochs.emplace_back();
					entry.epoch = epoch;
					entry.writeQueue = epoch == 1 ? writeQueue : org::QueueKind::Graphics;
					entry.scratch = org::Buffer::CreateSharedUnmaterialized(rhi::HeapType::DeviceLocal, kBytes, true);
					org::BufferBase::DescriptorRequirements requirements{};
					requirements.createUAV = true;
					requirements.uavDesc = {.dimension = rhi::UavDim::Buffer, .formatOverride = rhi::Format::R32_Typeless,
						.buffer = {.kind = rhi::BufferViewKind::Raw, .firstElement = 0, .numElements = kCount}};
					entry.scratch->SetDescriptorRequirements(requirements);
					entry.scratch->SetName("test.arena.scratch." + std::to_string(epoch));
					// A manual pool: placed whatever the auto-aliasing mode. The first use is the write.
					entry.scratch->SetAllowAlias(true);
					entry.scratch->SetAliasingPool(0xA5);
					entry.input = org::Buffer::CreateSharedUnmaterialized(rhi::HeapType::DeviceLocal, 16, false);
					org::BufferBase::DescriptorRequirements inputViews{};
					inputViews.createSRV = true;
					inputViews.srvDesc = {.dimension = rhi::SrvDim::Buffer, .formatOverride = rhi::Format::R32_Typeless,
						.buffer = {.kind = rhi::BufferViewKind::Raw, .firstElement = 0, .numElements = 4}};
					entry.input->SetDescriptorRequirements(inputViews);
					entry.input->SetName("test.arena.input." + std::to_string(epoch));
					entry.output = ImportHostBuffer(device.Get(), epoch == 0 ? first : second, epoch == 0 ? "test.arena.output.0" : "test.arena.output.1");
					REQUIRE(entry.output, "import host buffer");
				}
				const auto upload = [&](uint32_t epoch) {
					return [&, epoch](org::RenderGraph&) {
						BUFFER_UPLOAD(&values[epoch], sizeof(uint32_t), org::runtime::UploadTarget::FromShared(epochs[epoch].input), 0);
					};
				};
				auto outputRevision = std::make_shared<std::atomic<uint64_t>>(1);
				host.AddExtension("test.arena", [&] { return std::make_unique<ArenaExtension>(epochs, program, outputRevision); });
				const auto run = [&](uint32_t frame, bool async) -> int {
					values = {70000u + 10u * frame, 80000u + 10u * frame};
					if (async) {
						host.SubmitEpoch(0, upload(0));
						host.SubmitEpoch(1, upload(1));
					} else {
						host.ExecuteFrame(nullptr, upload(0), 0);
						host.ExecuteFrame(nullptr, upload(1), 1);
					}
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, values[0]) && Matches(second, values[1]),
						"each epoch copies what it wrote into its scratch buffer");
					return 0;
				};
				for (uint32_t frame = 0; frame < 4; ++frame)
					if (const int failed = run(frame, false)) return failed;
				host.SetAsyncEpochs({0, 1});
				for (uint32_t frame = 4; frame < 12; ++frame)
					if (const int failed = run(frame, true)) return failed;
				host.SetAsyncEpochs({});
				// Shared: one pool both scratch buffers are placed in, as large as one of them. Not shared: none in the host's arena.
				const auto pools = host.AliasArena();
				size_t poolCount = 0, owners = 0;
				uint64_t capacity = 0, required = 0;
				if (pools) {
					std::lock_guard lock(pools->mutex);
					poolCount = pools->pools.size();
					for (const auto& [id, pool] : pools->pools) {
						owners += pool.required.size();
						capacity += pool.capacityBytes;
						required = (std::max)(required, org::PersistentAliasArena::Required(pool));
					}
				}
				std::printf("arena (%s): %zu pools, %zu graphs placed, required %llu, capacity %llu\n", shared ? "shared" : "per graph", poolCount,
					owners, static_cast<unsigned long long>(required), static_cast<unsigned long long>(capacity));
				if (shared) REQUIRE(poolCount == 1 && owners == 2 && required >= kBytes && required < 2 * kBytes, "one pool, both graphs at offset 0");
				else REQUIRE(!pools, "no shared arena");
				REQUIRE(!counters.unbalanced && counters.locks.load() == counters.unlocks.load(), "balanced host queue locks");
				host.DestroyGraph();
			}
			runtime.reset();
		} else if (stream) {
			auto runtime = org::Runtime::Create({.device = device.Get(), .backend = rhi::Backend::Vulkan,
				.tasks = std::make_shared<org::runtime::ThreadPoolTaskService>(2)});
			// Without queue-scoped uploads the write waits for the uploads, which run in the host's stream: a segment of their own,
			// and the copy's segment split from it. With them the upload goes to the compute queue, ahead of the write.
			for (const bool scopedUploads : {false, true}) {
				org::GraphHost host({.runtime = runtime, .queueBoundary = {.entry = true, .exit = true}, .epochOrder = {0},
					.closedExecutions = !open, .externalHomeStates = home, .queueScopedUploads = scopedUploads});
				auto program = CreateProgram(device.Get());
				REQUIRE(program, "compute program");
				// "home": a texture the host owns, home in Common on the graphics queue.
				std::shared_ptr<org::ExternalTextureResource> hostTexture;
				if (home) {
					rhi::ResourceDesc desc{};
					desc.type = rhi::ResourceType::Texture2D;
					desc.debugName = "test.host-texture";
					desc.texture.format = rhi::Format::R8G8B8A8_UNorm;
					desc.texture.width = desc.texture.height = 16;
					desc.texture.initialLayout = rhi::ResourceLayout::Common;
					// Concurrent: a pass that touches an exclusive resource stays on the graphics queue on Vulkan.
					desc.queueSharing = rhi::QueueSharing::Concurrent;
					rhi::ResourcePtr texture;
					REQUIRE(device->CreateCommittedResource(desc, texture) == rhi::Result::Ok, "host texture");
					org::TextureDescription description{};
					description.imageDimensions = {{16, 16}};
					description.channels = 4;
					description.format = rhi::Format::R8G8B8A8_UNorm;
					description.hasSRV = true;
					description.initialLayout = rhi::ResourceLayout::Common;
					description.queueSharing = rhi::QueueSharing::Concurrent;
					hostTexture = org::ExternalTextureResource::CreateShared(std::move(texture), description, false);
					REQUIRE(hostTexture, "host texture resource");
					hostTexture->SetName("test.host-texture");
				}
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
				// Queue-scoped: a staged upload, which can go to the queue that first uses it (an immediate one stays on the graphics
				// list).
				const auto upload = [&](org::RenderGraph&) {
					if (!scopedUploads) {
						BUFFER_UPLOAD(&value, sizeof(value), org::runtime::UploadTarget::FromShared(input), 0);
						return;
					}
					auto staged = org::runtime::StagedUploadBatch::Create(256);
					staged->Stage(org::runtime::UploadTarget::FromShared(input), 0, &value, sizeof(value));
					org::runtime::GetActiveUploadService()->SubmitStagedUploads(std::move(staged));
				};
				auto outputRevision = std::make_shared<std::atomic<uint64_t>>(1);
				host.AddExtension("test.host", [&] {
					return std::make_unique<HostExtension>(input, scratch, output, program, outputRevision, nullptr, false, false, true, hostTexture);
				});
				host.SetExternalEpochs({0});
				host.SetAsyncEpochs({0});
				SegmentExecutor executor{device.Get()};
				size_t splits = 0, segments = 0;
				for (uint32_t frame = 0; frame < 8; ++frame) {
					value = 50000u + 10u * frame + (scopedUploads ? 5000u : 0u);
					host.SubmitEpoch(0, upload);
					auto execution = host.TakeExternalExecution();
					REQUIRE(execution && !execution->segments.empty(), "a stream epoch's execution");
					segments += execution->segments.size();
					for (const auto& segment : execution->segments) splits += segment.split ? 1 : 0;
					REQUIRE(executor.Execute(*execution), "the stream's segments");
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "the copy in the stream reads what the compute write wrote");
				}
				std::printf("stream (%s%s, %s uploads): %zu segments, %zu split over 8 frames\n", open ? "open" : "closed",
					home ? ", external home" : "", scopedUploads ? "queue-scoped" : "stream", segments, splits);
				// External home: the texture is released from the graphics queue in a segment ahead (the write waits for it, so the
				// copy's segment is split), and acquired back in a segment after the copy's (its wait is the copy's: no split).
				if (home) REQUIRE(splits == 8 && segments == (scopedUploads ? 24u : 32u), "the host texture's release and acquire segments");
				else if (!scopedUploads) REQUIRE(splits == 8 && segments == 16, "the copy's segment is split from the uploads' when the write waits for them");
				else REQUIRE(splits == 0 && segments == 8, "an upload on the write's queue needs no segment of its own, and no split");
				REQUIRE(!counters.unbalanced && counters.locks.load() == counters.unlocks.load(), "balanced host queue locks");
				host.SetAsyncEpochs({});
				host.DestroyGraph();
			}
			runtime.reset();
		} else if (segmentExternal) {
			auto runtime = org::Runtime::Create({.device = device.Get(), .backend = rhi::Backend::Vulkan,
				.tasks = std::make_shared<org::runtime::ThreadPoolTaskService>(2)});
			{
				org::GraphHost host({.runtime = runtime, .queueBoundary = {.entry = true, .exit = true}, .epochOrder = {0},
					.closedExecutions = false, .externalHomeStates = true, .reuseRecordings = true});
				host.SetPrivateAliasEpochs({0});
				auto program = CreateProgram(device.Get());
				REQUIRE(program, "compute program");
				// A texture the host owns, exclusive (as DXVK's images are), home on the graphics queue in RenderTarget: a layout the write's read
				// is not in, as DXVK rests a render target. Each block takes it out of its home state and must give it back.
				rhi::ResourceDesc textureDesc{};
				textureDesc.type = rhi::ResourceType::Texture2D;
				textureDesc.debugName = "test.host-texture";
				textureDesc.texture.format = rhi::Format::R8G8B8A8_UNorm;
				textureDesc.texture.width = textureDesc.texture.height = 16;
				textureDesc.texture.initialLayout = rhi::ResourceLayout::Common;
				textureDesc.resourceFlags = rhi::ResourceFlags::RF_AllowRenderTarget;
				rhi::ResourcePtr texture;
				REQUIRE(device->CreateCommittedResource(textureDesc, texture) == rhi::Result::Ok, "host texture");
				const auto textureHandle = texture->GetHandle();
				org::TextureDescription description{};
				description.imageDimensions = {{16, 16}};
				description.channels = 4;
				description.format = rhi::Format::R8G8B8A8_UNorm;
				description.hasSRV = true;
				description.hasRTV = true;
				description.initialLayout = rhi::ResourceLayout::RenderTarget;
				auto hostTexture = org::ExternalTextureResource::CreateShared(std::move(texture), description, false);
				REQUIRE(hostTexture, "host texture resource");
				hostTexture->SetName("test.host-texture");
				// The transient the write fills and the copy reads: placed in an alias pool, which SetPrivateAliasEpochs keeps the graph's own.
				auto scratch = org::Buffer::CreateSharedUnmaterialized(rhi::HeapType::DeviceLocal, kBytes, true);
				org::BufferBase::DescriptorRequirements requirements{};
				requirements.createUAV = true;
				requirements.uavDesc = {.dimension = rhi::UavDim::Buffer, .formatOverride = rhi::Format::R32_Typeless,
					.buffer = {.kind = rhi::BufferViewKind::Raw, .firstElement = 0, .numElements = kCount}};
				scratch->SetDescriptorRequirements(requirements);
				scratch->SetName("test.scratch");
				scratch->SetAllowAlias(true);
				scratch->SetAliasingPool(0xB7);
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
				const auto upload = [&](org::RenderGraph&) {
					auto staged = org::runtime::StagedUploadBatch::Create(256);
					staged->Stage(org::runtime::UploadTarget::FromShared(input), 0, &value, sizeof(value));
					org::runtime::GetActiveUploadService()->SubmitStagedUploads(std::move(staged));
				};
				auto outputRevision = std::make_shared<std::atomic<uint64_t>>(1);
				host.AddExtension("test.host", [&] {
					return std::make_unique<HostExtension>(input, scratch, output, program, outputRevision, nullptr, true, false, false, hostTexture);
				});
				host.SetExternalEpochs({0});
				host.SetAsyncEpochs({0}, {0});
				// A description's recording, made on the host's thread (a_generation: of a staged build).
				const auto record = [&](bool a_skipWrite, uint64_t a_generation = 0) -> std::shared_ptr<const org::GraphHost::EpochRecording> {
					auto data = std::make_shared<TestRevision>();
					data->skipWrite = a_skipWrite;
					std::atomic<int> state{0};
					std::shared_ptr<const org::GraphHost::EpochRecording> result;
					if (!host.RequestEpochRecording(0, data, [&](std::shared_ptr<const org::GraphHost::EpochRecording> a_recording, std::exception_ptr a_error) {
							result = std::move(a_recording);
							state.store(a_error ? 2 : 1, std::memory_order_release);
						}, org::RunRef::kNoCopy, a_generation))
						return nullptr;
					const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
					while (state.load(std::memory_order_acquire) == 0 && std::chrono::steady_clock::now() < deadline)
						std::this_thread::yield();
					return state.load(std::memory_order_acquire) == 1 ? result : nullptr;
				};
				SegmentExecutor executor{device.Get()};
				// The host's own use of the texture between two blocks: a barrier that expects it in its home state (RenderTarget), out and back.
				// Under the validation layer a texture left in another layout is an error.
				rhi::CommandAllocatorPtr hostAllocator;
				rhi::CommandListPtr hostList;
				REQUIRE(device->CreateCommandAllocator(rhi::QueueKind::Graphics, hostAllocator) == rhi::Result::Ok &&
					device->CreateCommandList(rhi::QueueKind::Graphics, hostAllocator.Get(), hostList) == rhi::Result::Ok, "the host's list");
				hostList->End();
				const auto hostUse = [&](bool a_first = false) -> bool {
					hostAllocator->Recycle();
					hostList->Recycle(hostAllocator.Get());
					if (a_first) {
						// The image is made in UNDEFINED: the host puts it in its home state once, as DXVK initializes its images.
						rhi::TextureBarrier initial{};
						initial.texture = textureHandle;
						initial.afterSync = rhi::ResourceSyncState::All;
						initial.beforeLayout = rhi::ResourceLayout::Undefined;
						initial.afterLayout = rhi::ResourceLayout::RenderTarget;
						initial.discard = true;
						hostList->Barriers({.textures = {&initial, 1}});
					}
					rhi::TextureBarrier barriers[2]{};
					barriers[0].texture = textureHandle;
					barriers[0].range = {};
					barriers[0].beforeSync = rhi::ResourceSyncState::All;
					barriers[0].afterSync = rhi::ResourceSyncState::Copy;
					barriers[0].beforeAccess = rhi::ResourceAccessType::None;
					barriers[0].afterAccess = rhi::ResourceAccessType::CopyDest;
					barriers[0].beforeLayout = rhi::ResourceLayout::RenderTarget;
					barriers[0].afterLayout = rhi::ResourceLayout::CopyDest;
					barriers[1] = barriers[0];
					barriers[1].beforeSync = rhi::ResourceSyncState::Copy;
					barriers[1].afterSync = rhi::ResourceSyncState::All;
					barriers[1].beforeAccess = rhi::ResourceAccessType::CopyDest;
					barriers[1].afterAccess = rhi::ResourceAccessType::None;
					barriers[1].beforeLayout = rhi::ResourceLayout::CopyDest;
					barriers[1].afterLayout = rhi::ResourceLayout::RenderTarget;
					hostList->Barriers({.textures = {&barriers[0], 1}});
					hostList->Barriers({.textures = {&barriers[1], 1}});
					if (hostList->EndChecked() != rhi::Result::Ok) return false;
					auto list = hostList.Get();
					return device->GetQueue(rhi::QueueKind::Graphics).Submit({&list, 1}, {}) == rhi::Result::Ok;
				};
				REQUIRE(hostUse(true) && device->WaitIdle() == rhi::Result::Ok, "the texture in its home state before the first block");
				auto staged = org::LatchBlock::Create("stage values", 16);
				auto full = record(false);
				REQUIRE(full && full->copies.size() == 3, "a description's recording, one copy per frame in flight");
				size_t segments = 0, splits = 0;
				// One frame: the slot's values staged, the pinned copy taken, the staged values copied into its run, the execution executed
				// on another thread, then the host's own use of the texture.
				const auto frame = [&](uint32_t a_frame, const std::shared_ptr<const org::GraphHost::EpochRecording>& a_recording, uint32_t a_value, bool a_writes) -> int {
					const uint32_t slot = a_frame % 3;
					const auto staging = host.EpochStagingRun(0, slot);
					org::LatchBlock::ZeroStableRun(staging.lease);
					host.UseRun(staging);
					staged->WriteValue(host.CurrentRun(), 0, 0x5000u + a_frame);
					host.EndHeldRun();
					const uint32_t kept = first.mapped[0];
					value = a_value;
					host.SetEpochCopy(0, slot);
					org::RunRef seen{};
					uint32_t copied = 0;
					host.SubmitEpoch(0, [&](org::RenderGraph& graph) {
						upload(graph);
						seen = host.CurrentRun();
						org::LatchBlock::CopyStableRun(staging.lease, seen.lease);
						std::memcpy(&copied, staged->Bytes(seen).data(), sizeof(copied));
					}, {}, a_recording);
					REQUIRE(seen == host.EpochCopyRun(0, slot), "the run is its frame slot's copy");
					REQUIRE(copied == 0x5000u + a_frame, "the copy's run reads what the slot staged");
					auto execution = host.TakeExternalExecution();
					REQUIRE(execution && !execution->segments.empty(), "an external execution of the revision-driven epoch");
					REQUIRE(!host.TakeExternalExecution(), "taken once");
					segments += execution->segments.size();
					for (const auto& segment : execution->segments) splits += segment.split ? 1 : 0;
					const auto copies = g_copyRecords.load(std::memory_order_relaxed);
					bool executed = false;
					std::thread worker([&] { executed = executor.Execute(*execution); });
					worker.join();
					REQUIRE(executed, "the execution, on the host's worker thread");
					REQUIRE(g_copyRecords.load(std::memory_order_relaxed) == copies + 1, "the copy pass recorded at execution, once a run");
					REQUIRE(hostUse(), "the host's use of the texture in its home state");
					REQUIRE(device->WaitIdle() == rhi::Result::Ok, "the frame completes");
					REQUIRE(a_writes ? Matches(first, a_value) : first.mapped[0] == kept, a_writes ? "the block writes and copies" : "a write the description turns off writes nothing");
					return 0;
				};
				(void)host.TakeAsyncStats();
				uint32_t frameIndex = 0;
				for (; frameIndex < 9; ++frameIndex)
					if (const int failed = frame(frameIndex, full, 80000u + 10u * frameIndex, true)) return failed;
				// A description that turns the write off: the copy still runs and copies what the scratch holds, which is not this frame's.
				// The transient's bytes are the pool's, so only the write's absence is checked: the output is not this frame's value.
				{
					auto skipped = record(true);
					REQUIRE(skipped, "the skipped write's recording");
					value = 0;
					const uint32_t slot = frameIndex % 3;
					host.SetEpochCopy(0, slot);
					host.SubmitEpoch(0, upload, {}, skipped);
					auto execution = host.TakeExternalExecution();
					bool executed = false;
					std::thread worker([&] { executed = execution && executor.Execute(*execution); });
					worker.join();
					REQUIRE(executed && hostUse() && device->WaitIdle() == rhi::Result::Ok, "the skipped write's block");
					REQUIRE(!Matches(first, 0), "a write the description turns off writes nothing");
					++frameIndex;
				}
				// A staged build: recorded for while the attached build's recordings still submit, then switched in with its recordings.
				{
					auto next = host.BuildEpochGraph(0, "test: a staged build of an external epoch");
					REQUIRE(next && next->Built(), "the staged build");
					const auto staging = host.StageEpochGraph(next);
					REQUIRE(staging.staged && host.StagedGeneration(0) == staging.generation, "the build is staged");
					auto stagedRecording = record(false, staging.generation);
					REQUIRE(stagedRecording && !host.EpochRecordingCurrent(*stagedRecording), "a recording made on the staged build");
					if (const int failed = frame(frameIndex, full, 81000u + frameIndex, true)) return failed;
					++frameIndex;
					REQUIRE(host.AttachStagedEpochGraph(0).attached && host.EpochRecordingCurrent(*stagedRecording), "the staged build switched in");
					for (uint32_t i = 0; i < 6; ++i, ++frameIndex)
						if (const int failed = frame(frameIndex, stagedRecording, 82000u + 10u * frameIndex, true)) return failed;
				}
				const auto stats = host.TakeAsyncStats();
				// The epoch's transients are in pools of its own, not in the host's shared arena.
				size_t owners = 0;
				if (const auto pools = host.AliasArena()) {
					std::lock_guard lock(pools->mutex);
					for (const auto& [id, pool] : pools->pools) owners += pool.required.size();
				}
				std::printf("segment-external: %llu revision submissions, %llu prepared again, %llu copy waits, %llu copies busy; %zu segments, %zu split; "
					"%zu graphs placed in the shared arena\n",
					static_cast<unsigned long long>(stats.revisionSubmissions), static_cast<unsigned long long>(stats.stale),
					static_cast<unsigned long long>(stats.copyWaits), static_cast<unsigned long long>(stats.copiesBusy), segments, splits, owners);
				REQUIRE(stats.stale == 0 && stats.copiesBusy == 0, "every submission bound its description's recording; no copy was busy");
				REQUIRE(owners == 0, "the epoch's transients stay in its own pools");
				REQUIRE(!counters.unbalanced && counters.locks.load() == counters.unlocks.load(), "balanced host queue locks");
				host.SetAsyncEpochs({});
				host.SetExternalEpochs({});
				host.DestroyGraph();
			}
			runtime.reset();
		} else if (leases) {
			// Two frames in flight. The runtime starts with 4 statistics ranges and grows them with the leases in flight.
			auto runtime = org::Runtime::Create({.device = device.Get(), .backend = rhi::Backend::Vulkan, .slots = 4,
				.tasks = std::make_shared<org::runtime::ThreadPoolTaskService>(2)});
			{
				org::GraphHost host({.runtime = runtime, .queueBoundary = {.entry = true, .exit = true}, .framesInFlight = 2, .epochOrder = {0},
					.closedExecutions = !open, .reuseRecordings = wide});
				// The frame arena. A leased run's region is found again by every call for the run, zeroed when allocated, and freed
				// with its lease. Two runs in flight have two regions. A published copy's region is pinned to its index.
				{
					auto block = org::LatchBlock::Create("test.latch", 64);
					REQUIRE(block->Stride() == org::FrameArena::kAlignment, "a region is rounded up to the arena's alignment");
					const org::RunRef run{3, 1, org::RunRef::kNoCopy};
					const auto region = block->At(run);
					bool zeroed = static_cast<bool>(region);
					for (uint32_t i = 0; zeroed && i < region.size; ++i) zeroed = region.mapped[i] == std::byte{0};
					REQUIRE(zeroed, "a run's region is zeroed");
					block->WriteValue(run, 0, uint32_t{7});
					REQUIRE(block->At(run).offset == region.offset && block->At(run).buffer == region.buffer && block->At(run).srvIndex == region.srvIndex,
						"the writer and the recorder of a run find the same region");
					const org::RunRef other{4, 1, org::RunRef::kNoCopy};
					const auto otherRegion = block->At(other);
					REQUIRE(otherRegion.offset != region.offset || otherRegion.buffer != region.buffer, "two runs in flight have two regions");
					const auto copy = org::RunRef::Copy(9, 0);
					const auto pinned = block->At(copy);
					runtime->Arena()->ReleaseRun(3, 1);
					const org::RunRef next{3, 2, org::RunRef::kNoCopy};
					uint32_t word = 1;
					std::memcpy(&word, block->At(next).mapped, sizeof(word));
					REQUIRE(word == 0, "the index's next run reads zeros, not the last run's values");
					REQUIRE(block->At(copy).offset == pinned.offset && block->At(copy).buffer == pinned.buffer, "a published copy's region stays");
					runtime->Arena()->ReleaseRun(3, 2);
					runtime->Arena()->ReleaseRun(4, 1);
				}
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
				// The runs the commits wrote for this frame (GraphHost::CurrentRun inside beforeSubmit).
				std::vector<org::RunRef> frameRuns;
				const auto upload = [&](org::RenderGraph&) {
					BUFFER_UPLOAD(&value, sizeof(value), org::runtime::UploadTarget::FromShared(input), 0);
					frameRuns.push_back(host.CurrentRun());
				};
				auto outputRevision = std::make_shared<std::atomic<uint64_t>>(1);
				host.AddExtension("test.host", [&] { return std::make_unique<HostExtension>(input, scratch, output, program, outputRevision); });
				host.SetExternalEpochs({0});
				host.SetAsyncEpochs({0});
				// A deadlock (the host's thread waiting for a held run while this thread waits for it) ends the test instead of hanging it.
				std::atomic<bool> finished{false};
				std::thread watchdog([&] {
					const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
					while (!finished.load(std::memory_order_acquire)) {
						if (std::chrono::steady_clock::now() > deadline) {
							std::puts("FAIL: leases: the frame did not finish (a wait for a run of the same frame)");
							std::fflush(stdout);
							std::_Exit(1);
						}
						std::this_thread::sleep_for(std::chrono::milliseconds(20));
					}
				});
				struct JoinWatchdog {
					std::atomic<bool>& finished;
					std::thread& thread;
					~JoinWatchdog() {
						finished.store(true, std::memory_order_release);
						if (thread.joinable()) thread.join();
					}
				} joinWatchdog{finished, watchdog};
				// The frame's executions, executed in order at its end (one at a time: the executor reuses its command buffers), then sealed.
				SegmentExecutor executor{device.Get()};
				std::vector<std::shared_ptr<org::ExternalEpochExecution>> held;
				const auto endFrame = [&](uint32_t expected) -> int {
					for (auto& execution : held)
						REQUIRE(executor.Execute(*execution) && device->WaitIdle() == rhi::Result::Ok, "a held execution at the frame's end");
					held.clear();
					REQUIRE(Matches(first, expected), "the frame's last run wrote last");
					// Every run of the frame had a lease of its own.
					for (size_t a = 0; a < frameRuns.size(); ++a) {
						REQUIRE(frameRuns[a].Leased() && !frameRuns[a].Published(), "a live run is leased");
						for (size_t b = 0; b < a; ++b) REQUIRE(!(frameRuns[a] == frameRuns[b]), "runs in flight have leases of their own");
					}
					frameRuns.clear();
					host.SealSubmitted();
					return 0;
				};
				(void)host.TakeAsyncStats();
				(void)host.TakeRunMemoryStats();
				// Six runs a frame, three times the frames in flight: a ring of two slots would wait for a run still held. Wide: 260
				// runs a frame.
				const uint32_t kRuns = wide ? 260u : 6u;
				const uint32_t kFrames = wide ? 3u : 6u;
				for (uint32_t frame = 0; frame < kFrames; ++frame) {
					for (uint32_t run = 0; run < kRuns; ++run) {
						value = 120000u + 100u * frame + 10u * run;
						host.SubmitEpoch(0, upload);
						auto execution = host.TakeExternalExecution();
						REQUIRE(execution, "an external execution per run");
						held.push_back(std::move(execution));
					}
					if (const int failed = endFrame(value)) return failed;
				}
				const auto stats = host.TakeAsyncStats();
				// The host's thread retires the last frame's runs at its next wake-up.
				const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
				while (host.TakeRunMemoryStats().leasesInFlight > 1 && std::chrono::steady_clock::now() < deadline) {
					(void)host.NudgeHostThread();
					std::this_thread::sleep_for(std::chrono::milliseconds(5));
				}
				const auto memory = host.TakeRunMemoryStats();
				std::printf("leases: %llu submissions, %u leases made, %u in flight now, %u statistics ranges, arena %llu chunks (%llu bytes)\n",
					static_cast<unsigned long long>(stats.submitted), memory.leaseIndices, memory.leasesInFlight, memory.statisticsRanges,
					static_cast<unsigned long long>(memory.arena.chunks), static_cast<unsigned long long>(memory.arena.chunkBytes));
				REQUIRE(stats.submitted == kFrames * kRuns && stats.copiesBusy == 0, "every run submitted, none skipped");
				REQUIRE(memory.leaseIndices >= kRuns && memory.leaseIndices <= 2 * kRuns + 4, "leases follow the runs in flight, reused once retired");
				REQUIRE(memory.leasesInFlight <= 1, "the runs retire once they complete: only the next ticket's lease is open");
				REQUIRE(memory.statisticsRanges >= memory.leaseIndices, "a statistics range per lease index");
				// Four frames' worth of runs between two seals: all of them go through, nothing waits and nothing is skipped.
				uint32_t last = 0;
				for (uint32_t run = 0; run < 4 * kRuns; ++run) {
					value = 130000u + 10u * run;
					host.SubmitEpoch(0, upload);
					auto execution = host.TakeExternalExecution();
					REQUIRE(execution, "an external execution per run");
					held.push_back(std::move(execution));
					last = value;
				}
				if (const int failed = endFrame(last)) return failed;
				REQUIRE(host.PollEpochFailures() == 0, "nothing reported");
				REQUIRE(!counters.unbalanced && counters.locks.load() == counters.unlocks.load(), "balanced host queue locks");
				host.SetAsyncEpochs({});
				REQUIRE(host.TakeRunMemoryStats().leasesInFlight == 0, "stopping async epochs retires every lease");
				host.DestroyGraph();
			}
			runtime.reset();
		} else if (contributor) {
			auto program = CreateProgram(device.Get());
			REQUIRE(program, "compute program");
			const auto importOutput = [&]() -> std::shared_ptr<org::Resource> {
				return ImportHostBuffer(device.Get(), second, "test.contributor-output");
			};
			REQUIRE(TestContributors(device.Get(), importOutput, second.mapped, ORG_TEST_PERSISTENT_WRITE_SPV,
				program->layout->GetHandle()) == 0, "contributors");
		} else {
		// The host registers the runtime device, owns the task service and upload
		// pass, and drives persistent execution with the external queue boundary.
		auto runtime = org::Runtime::Create({.device = device.Get(), .backend = rhi::Backend::Vulkan,
			.tasks = std::make_shared<org::runtime::ThreadPoolTaskService>(2)});
		org::GraphHost host({.runtime = runtime,
			.queueBoundary = {.entry = true, .exit = true}, .epochOrder = {0}, .closedExecutions = async && !open && !segment, .reuseRecordings = reuse || revision});
		if (segment) host.SetClosedEpochs({0});
		if (inlineLate) host.SetInlineAtExecution(true);
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
			// "revision": the write pass's target is a versioned buffer, which the revision names a version of.
			std::shared_ptr<org::VersionedBuffer> versions;
			if (revision) {
				scratch->Materialize();
				versions = org::VersionedBuffer::Create(scratch);
			}
			host.AddExtension("test.host", [&] { return std::make_unique<HostExtension>(input, scratch, output, program, outputRevision, versions, external || inlineLate); });
			if (revision) {
				host.SetAsyncEpochs({0}, {0});
				// Records the epoch for a revision of `count` elements; the callback runs on the host's thread.
				auto record = [&](uint32_t count, std::shared_ptr<const org::BufferVersion> target, bool skipWrite = false,
					uint32_t onlyCopy = org::RunRef::kNoCopy, uint64_t generation = 0) -> std::shared_ptr<const org::GraphHost::EpochRecording> {
					auto data = std::make_shared<TestRevision>();
					data->count = count;
					data->skipWrite = skipWrite;
					data->versioned = versions->Key();
					data->target = std::move(target);
					std::atomic<int> state{0};
					std::shared_ptr<const org::GraphHost::EpochRecording> result;
					const bool posted = host.RequestEpochRecording(0, data, [&](std::shared_ptr<const org::GraphHost::EpochRecording> a_recording,
						std::exception_ptr a_error) {
						result = std::move(a_recording);
						state.store(a_error ? 2 : 1, std::memory_order_release);
						state.notify_all();
					}, onlyCopy, generation);
					if (!posted) return nullptr;
					const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
					while (state.load(std::memory_order_acquire) == 0 && std::chrono::steady_clock::now() < deadline)
						std::this_thread::yield();
					return state.load(std::memory_order_acquire) == 1 ? result : nullptr;
				};
				const auto firstVersion = versions->Current();
				auto full = record(kCount, firstVersion);
				REQUIRE(full && full->copies.size() == 3, "a revision's recording, one copy per frame in flight");
				(void)host.TakeAsyncStats();
				for (uint32_t frame = 0; frame < 6; ++frame) {
					value = 40000u + 10u * frame;
					const auto copies = g_copyRecords.load(std::memory_order_relaxed);
					host.SubmitEpoch(0, upload, {}, full);
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "a revision's recording writes every element");
					REQUIRE(!inlineLate || g_copyRecords.load(std::memory_order_relaxed) == copies + 1,
						"a kept recording's inline late pass is recorded again at each submission");
					REQUIRE(!host.TakeExternalExecution(), "an inline epoch hands no execution back");
				}
				// Growth as a new version. A revision names the grown target. It is recorded while the frames still submit the old
				// one. Each recording resolved its own version, with its own publication and bindings. Each ticket admits both
				// recordings. The old one still submits after the new one was recorded, and the new one submits after it. Nothing
				// is prepared again.
				// The growth: the next version is current from now on. The old revision still names the first version.
				const auto grown = versions->GrowBytes(kBytes * 2);
				REQUIRE(grown->buffer != firstVersion->buffer && versions->Get() == grown->buffer, "growth is a new version");
				auto grownRecording = record(kCount, grown);
				REQUIRE(grownRecording, "the grown revision's recording");
				for (uint32_t frame = 0; frame < 4; ++frame) {
					value = 45000u + 10u * frame;
					host.SubmitEpoch(0, upload, {}, frame < 2 ? full : grownRecording);
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value),
						frame < 2 ? "the old version's revision after the grown one was recorded" : "the grown version's revision");
				}
				// A new revision: half the elements. Until its recording is ready the frames keep submitting the old one.
				const uint32_t half = kCount / 2;
				auto halfRecording = record(half, grown);
				REQUIRE(halfRecording, "the next revision's recording");
				const uint32_t before = value;
				value = 50000u;
				host.SubmitEpoch(0, upload, {}, halfRecording);
				REQUIRE(device->WaitIdle() == rhi::Result::Ok, "wait for the half revision");
				bool halfOk = true;
				for (uint32_t i = 0; i < kCount; ++i)
					halfOk &= first.mapped[i] == (i < half ? value + i : before + i);
				if (!halfOk)
					std::fprintf(stderr, "half revision: [0] %u [half-1] %u [half] %u [last] %u; expected %u, %u, %u, %u\n", first.mapped[0], first.mapped[half - 1],
						first.mapped[half], first.mapped[kCount - 1], value, value + half - 1, before + half, before + kCount - 1);
				REQUIRE(halfOk, "the new revision's recording writes its elements, and leaves the rest");
				// A pending version, made but not adopted, and named by a revision. Its recording writes and copies that version.
				// Every pass resolves it through the revision, including both of the copy pass's declarations of it. Meanwhile the
				// current version stays the one that live work uses.
				{
					const auto pending = versions->MakeBytes(kBytes * 2);
					REQUIRE(pending->buffer != grown->buffer && versions->Get() == grown->buffer, "a pending version is not current");
					auto pendingRecording = record(kCount, pending);
					REQUIRE(pendingRecording, "the pending version's revision's recording");
					value = 47000u;
					host.SubmitEpoch(0, upload, {}, pendingRecording);
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "a revision's recording copies the version it names");
				}
				// A revision-driven epoch without its recording is an error, never a re-preparation.
				bool rejected = false;
				try {
					host.SubmitEpoch(0, upload);
				} catch (const std::logic_error&) {
					rejected = true;
				}
				REQUIRE(rejected, "a revision-driven epoch without a recording is rejected");
				value = 60000u;
				host.SubmitEpoch(0, upload, {}, full);
				REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "the host goes on after a rejected submission");
				const auto stats = host.TakeAsyncStats();
				std::printf("revision: %llu recordings (%llu copies), %llu submissions, %llu recorded at tickets, %llu prepared again\n",
					static_cast<unsigned long long>(stats.revisionRecordings), static_cast<unsigned long long>(stats.revisionCopiesRecorded),
					static_cast<unsigned long long>(stats.revisionSubmissions), static_cast<unsigned long long>(stats.recorded),
					static_cast<unsigned long long>(stats.stale));
				REQUIRE(stats.revisionSubmissions == 13 && stats.stale == 0 && stats.revisionRecordings == 3 && stats.revisionCopiesRecorded == 9,
					"every submission bound a revision's recording; nothing prepared again");
				if (segment) {
					REQUIRE(host.ClosedEpoch(0), "SetClosedEpochs closes the epoch while the host's executions are open");
					// Pinned copies: each submission names its frame slot, and its run is that slot's copy.
					const auto current = versions->Current();
					auto slotted = record(kCount, current);
					REQUIRE(slotted, "the slotted revision's recording");
					const uint32_t copies = static_cast<uint32_t>(slotted->copies.size());
					for (uint32_t frame = 0; frame < 3 * copies; ++frame) {
						const uint32_t slot = frame % copies;
						value = 70000u + 10u * frame;
						host.SetEpochCopy(0, slot);
						org::RunRef seen{};
						host.SubmitEpoch(0, [&](org::RenderGraph& graph) {
							upload(graph);
							seen = host.CurrentRun();
						}, {}, slotted);
						REQUIRE(seen == host.EpochCopyRun(0, slot), "a pinned run is its frame slot's copy");
						REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "a pinned copy's recording writes every element");
					}
					// A frame without the epoch: the next slot is not the copy predicted. A ticket already prepared for the predicted copy
					// goes back once. One the host's thread prepares after the call takes the named copy at once.
					const auto before = host.TakeAsyncStats();
					value = 71000u;
					host.SetEpochCopy(0, 1);
					org::RunRef seen{};
					host.SubmitEpoch(0, [&](org::RenderGraph& graph) {
						upload(graph);
						seen = host.CurrentRun();
					}, {}, slotted);
					REQUIRE(seen == host.EpochCopyRun(0, 1), "a re-pinned run is the named copy");
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "a re-pinned copy's recording writes every element");
					const auto repinned = host.TakeAsyncStats();
					std::printf("segment: %llu copy re-pins over %u slotted frames, %llu after a skipped slot\n",
						static_cast<unsigned long long>(before.copyRepins), 3 * copies, static_cast<unsigned long long>(repinned.copyRepins));
					REQUIRE(before.copyRepins <= 1 && repinned.copyRepins <= 1, "only the first pin and the skipped slot may give a ticket back");
					// A revision that turns the write pass off: the copy runs, the write records only its barriers, so the host buffer
					// keeps what the last write made.
					auto skipped = record(kCount, current, true);
					REQUIRE(skipped, "the skipped write's recording");
					const uint32_t kept = value;
					value = 72000u;
					host.SetEpochCopy(0, 2);
					host.SubmitEpoch(0, upload, {}, skipped);
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, kept), "a pass the host data turns off writes nothing");
					value = 73000u;
					host.SetEpochCopy(0, 0);
					host.SubmitEpoch(0, upload, {}, slotted);
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "the next revision writes again");
					// One copy only: recorded for the slot the next run takes, and submitted with it. Its other copies stay empty.
					auto single = record(kCount, current, false, 1);
					REQUIRE(single && single->copies[1] && !single->copies[0], "a recording of one copy has that copy alone");
					value = 74000u;
					host.SetEpochCopy(0, 1);
					host.SubmitEpoch(0, upload, {}, single);
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "a one-copy recording writes every element");
					// The next slot's run takes the full recording again: the one-copy recording is no candidate for it.
					value = 75000u;
					host.SetEpochCopy(0, 2);
					host.SubmitEpoch(0, upload, {}, slotted);
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "the other slots go on with their recordings");
					// A staging run per copy (a host that gathers its values at several points and takes the copy once): its own lease
					// index, CurrentRun while it is used, and copied whole into the copy's run inside the copy's beforeSubmit.
					{
						auto staged = org::LatchBlock::Create("staged values", 16);
						const auto staging = host.EpochStagingRun(0, 2);
						const auto copyRun = host.EpochCopyRun(0, 2);
						REQUIRE(staging.Stable() && !staging.Published() && staging.lease != copyRun.lease, "a staging run is a stable run of its own index");
						REQUIRE(!(host.EpochStagingRun(0, 1) == staging), "each copy has its own staging run");
						host.UseRun(staging);
						REQUIRE(host.CurrentRun() == staging, "the staging run is CurrentRun while it is used");
						staged->WriteValue(host.CurrentRun(), 0, 0x5EEDu);
						REQUIRE(!staged->At(staging).buffer, "a staging run's region is host memory, not the arena's");
						host.EndHeldRun();
						// A block written for the copy's run before, and not staged: zero after the copy.
						auto stale = org::LatchBlock::Create("stale values", 16);
						stale->WriteValue(copyRun, 0, 0xBADu);
						value = 76500u;
						host.SetEpochCopy(0, 2);
						uint32_t copied = 0, cleared = 1;
						host.SubmitEpoch(0, [&](org::RenderGraph& graph) {
							upload(graph);
							org::LatchBlock::CopyStableRun(staging.lease, host.CurrentRun().lease);
							std::memcpy(&copied, staged->Bytes(host.CurrentRun()).data(), sizeof(copied));
							std::memcpy(&cleared, stale->Bytes(host.CurrentRun()).data(), sizeof(cleared));
						}, {}, slotted);
						REQUIRE(copied == 0x5EEDu && cleared == 0u, "the copy's run reads what was staged, and zero where nothing was");
						REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "a staged copy's recording writes every element");
						// The staging run cleared and staged again further in only: the copy moves the written part, and clears what the last copy
						// left before it. Then nothing staged at all: the copy's run reads zeros again.
						org::LatchBlock::ZeroStableRun(staging.lease);
						staged->WriteValue(staging, 8, 0xF00Du);
						uint32_t words[4] = {1, 1, 1, 1};
						for (const uint32_t round : {0u, 1u}) {
							if (round == 1) org::LatchBlock::ZeroStableRun(staging.lease);
							value = 76600u + round;
							host.SetEpochCopy(0, 2);
							host.SubmitEpoch(0, [&](org::RenderGraph& graph) {
								upload(graph);
								org::LatchBlock::CopyStableRun(staging.lease, host.CurrentRun().lease);
								std::memcpy(words, staged->Bytes(host.CurrentRun()).data(), sizeof(words));
							}, {}, slotted);
							REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "a staged copy's recording writes every element");
							REQUIRE(words[0] == 0u && words[1] == 0u && words[2] == (round == 0 ? 0xF00Du : 0u) && words[3] == 0u,
								"the copy's run reads exactly what was staged since the staging run was cleared");
						}
					}
					// A staged graph (StageEpochGraph): recorded for before it runs, while the attached graph's recordings still submit, then
					// swapped in with its recordings at one point.
					{
						auto next = host.BuildEpochGraph(0, "test: a staged build");
						REQUIRE(next && next->Built(), "the staged build");
						const auto before = host.BuildGeneration(0);
						const auto staging = host.StageEpochGraph(next);
						REQUIRE(staging.staged && !next && staging.generation > before && host.StagedGeneration(0) == staging.generation,
							"the build is staged with a new generation");
						REQUIRE(host.BuildGeneration(0) == before, "the attached graph keeps running");
						auto stagedRecording = record(kCount, versions->Current(), false, org::RunRef::kNoCopy, staging.generation);
						REQUIRE(stagedRecording && stagedRecording->buildGeneration == staging.generation, "a recording made on the staged graph");
						REQUIRE(!host.EpochRecordingCurrent(*stagedRecording), "not current before its attach");
						REQUIRE(!record(kCount, versions->Current()), "the attached graph records nothing while a graph is staged");
						value = 78000u;
						host.SetEpochCopy(0, 2);
						host.SubmitEpoch(0, upload, {}, slotted);
						REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "the attached graph's recording submits while staged");
						REQUIRE(host.AttachStagedEpochGraph(0).attached, "the staged graph is attached");
						REQUIRE(host.BuildGeneration(0) == staging.generation && host.StagedGeneration(0) == 0, "the epoch runs the staged build");
						REQUIRE(host.EpochRecordingCurrent(*stagedRecording) && !host.EpochRecordingCurrent(*slotted), "its recordings are current, the old ones not");
						for (uint32_t frame = 0; frame < 3; ++frame) {
							value = 79000u + frame;
							host.SetEpochCopy(0, frame);
							host.SubmitEpoch(0, upload, {}, stagedRecording);
							REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "the staged graph's recording submits after the attach");
						}
						// A second staging replaces the first; the replaced one's recording request is dropped.
						auto one = host.BuildEpochGraph(0, "test: staged, then replaced");
						auto two = host.BuildEpochGraph(0, "test: staged last");
						const auto stagedOne = host.StageEpochGraph(one);
						const auto stagedTwo = host.StageEpochGraph(two);
						REQUIRE(stagedOne.staged && stagedTwo.staged && stagedTwo.generation > stagedOne.generation, "a later staging replaces an earlier one");
						REQUIRE(!record(kCount, versions->Current(), false, org::RunRef::kNoCopy, stagedOne.generation), "a replaced staged graph records nothing");
						auto last = record(kCount, versions->Current(), false, org::RunRef::kNoCopy, stagedTwo.generation);
						REQUIRE(last && host.AttachStagedEpochGraph(0).attached, "the last staged graph is attached");
						value = 79500u;
						host.SetEpochCopy(0, 0);
						host.SubmitEpoch(0, upload, {}, last);
						REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "the last staged graph's recording submits");
					}
					// Versions: the slotted recording bound the current version; the first one bound the version before the growth.
					REQUIRE(host.EpochRecordingVersionsCurrent(*slotted, 0), "a recording of the current versions is current");
					REQUIRE(!host.EpochRecordingVersionsCurrent(*full, 0), "a recording of a replaced version is not");
					REQUIRE(!host.EpochRecordingVersionsCurrent(*single, 0), "a recording has no versions for a copy it does not have");
				}
				REQUIRE(!counters.unbalanced && counters.locks.load() == counters.unlocks.load(), "balanced host queue locks");
				// Growth as graph work. A version is made on a producer's thread and filled through the dedicated uploader. That is
				// the worker upload path. It runs on the spare queue the device offered, when it has one. The version is ready once
				// its copy completes. No queue uses it before then, and its owner adopts it after.
				{
					auto readable = org::Buffer::CreateShared(rhi::HeapType::Readback, kBytes);
					auto growing = org::VersionedBuffer::Create(readable);
					std::vector<uint32_t> contents(kCount * 2);
					for (uint32_t i = 0; i < contents.size(); ++i)
						contents[i] = 0xC0DE0000u + i;
					auto uploads = host.RetainUploads();
					REQUIRE(uploads, "the host's upload service");
					std::shared_ptr<const org::BufferVersion> made;
					std::shared_ptr<org::TrackedUploadTicket> ticket;
					std::thread producer([&] {
						made = growing->MakeBytes(kBytes * 2);
						const org::StreamingUploadSegment segment{contents.data(), contents.size() * sizeof(uint32_t)};
						ticket = uploads->QueueTrackedStreamingUploadSegments({&segment, 1}, segment.size,
							org::WorkerOwnedDestination{made->buffer, org::WorkerOwnedDestination::Ownership::PendingVersion}, 0);
					});
					producer.join();
					REQUIRE(made && ticket && growing->Get() == readable, "a pending version, made on a producer's thread");
					const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
					while (!ticket->Complete() && std::chrono::steady_clock::now() < deadline)
						std::this_thread::yield();
					REQUIRE(ticket->Complete(), "its fill completes on the uploader's queue");
					void* mapped = nullptr;
					auto resource = made->buffer->GetAPIResource();
					resource.Map(&mapped, 0, contents.size() * sizeof(uint32_t));
					const bool filled = mapped && std::memcmp(mapped, contents.data(), contents.size() * sizeof(uint32_t)) == 0;
					resource.Unmap(0, 0);
					REQUIRE(filled, "the new version holds its contents");
					growing->Adopt(made);
					REQUIRE(growing->Get() == made->buffer, "adopted by its owner");
					std::printf("growth as graph work: %s uploader queue\n", uploads->HasDedicatedStreamingQueue() ? "a dedicated" : "the shared");
					REQUIRE(!spareQueue || uploads->HasDedicatedStreamingQueue(), "the spare queue is the uploader's");
				}
				// A live epoch publishes nothing: it records every run. A recording request for it completes with an error, and a
				// recording given to its submission is rejected without hanging the host.
				host.SetAsyncEpochs({0});
				{
					value = 69990u;
					host.SubmitEpoch(0, upload);
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "the live epoch after the switch");
					auto data = std::make_shared<TestRevision>();
					data->count = kCount;
					data->versioned = versions->Key();
					data->target = versions->Current();
					std::atomic<int> state{0};
					REQUIRE(host.RequestEpochRecording(0, data, [&](std::shared_ptr<const org::GraphHost::EpochRecording> a_recording, std::exception_ptr a_error) {
						state.store(a_error && !a_recording ? 2 : 1, std::memory_order_release);
					}), "the request is queued");
					const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
					for (uint32_t frame = 0; state.load(std::memory_order_acquire) == 0 && std::chrono::steady_clock::now() < deadline; ++frame) {
						value = 70000u + 10u * (frame % 1000u);
						host.SubmitEpoch(0, upload);
						REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "the live epoch draws meanwhile");
					}
					REQUIRE(state.load(std::memory_order_acquire) == 2, "a live epoch's recording request completes with an error");
					bool rejected = false;
					try {
						host.SubmitEpoch(0, upload, {}, full);
					} catch (const std::logic_error&) {
						rejected = true;
					}
					REQUIRE(rejected, "a recording given to a live epoch is rejected");
					value = 77000u;
					host.SubmitEpoch(0, upload);
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "the live epoch submits again after the rejection");
				}
				host.SetAsyncEpochs({});
			}
			if (external) {				host.SetAsyncEpochs({});
			}
			if (external) {
				host.SetExternalEpochs({0});
				host.SetAsyncEpochs({0});
				SegmentExecutor executor{device.Get()};
				const auto execute = [&](org::ExternalEpochExecution& a_execution) { return executor.Execute(a_execution); };
				(void)host.TakeAsyncStats();
				for (uint32_t frame = 0; frame < 6; ++frame) {
					value = 90000u + 10u * frame;
					host.SubmitEpoch(0, upload);
					auto execution = host.TakeExternalExecution();
					REQUIRE(execution && execution->segments.size() == 1 && !execution->segments[0].signals.empty()
						&& !execution->segments[0].items.empty(), "an external epoch's execution, one segment");
					REQUIRE(!host.TakeExternalExecution(), "taken once");
					const auto before = g_copyRecords.load(std::memory_order_relaxed);
					REQUIRE(execute(*execution), "the execution in the test's primary");
					REQUIRE(g_copyRecords.load(std::memory_order_relaxed) == before + 1, "the late copy pass recorded at execution");
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "an external execution writes and copies");
				}
				if (externalReuse) {
					const auto recorded = host.TakeAsyncStats();
					std::printf("external-reuse: %llu recorded\n", static_cast<unsigned long long>(recorded.recorded));
					REQUIRE(recorded.recorded >= 6, "a live epoch with a replayable late pass still records every run");
				}
				// An execution that is not executed. It is signalled without running its work. The output keeps its last value, and
				// the next execution goes on.
				const uint32_t kept = value;
				value = 95000u;
				host.SubmitEpoch(0, upload);
				auto skipped = host.TakeExternalExecution();
				REQUIRE(skipped, "the execution to skip");
				host.SignalExternalExecutionUnexecuted(*skipped);
				REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, kept), "an unexecuted execution writes nothing");
				value = 96000u;
				host.SubmitEpoch(0, upload);
				auto next = host.TakeExternalExecution();
				REQUIRE(next && execute(*next) && device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "the epoch after a skipped execution");
				// Collect the pass timestamps of an external epoch by name, including its late pass. They come from the completed
				// frames, on the host's thread.
				std::mutex timedMutex;
				std::set<std::string> timed;
				host.SetCompletedFrameCallback([&](uint64_t, const org::runtime::IStatisticsService& a_stats) {
					std::scoped_lock lock(timedMutex);
					for (const auto& timestamp : a_stats.GetCompletedTimestamps())
						if (timestamp.kind == org::runtime::TimestampRangeKind::Pass && timestamp.beginTick && timestamp.endTick >= timestamp.beginTick)
							timed.emplace(timestamp.name);
				});
				const auto timedPasses = [&](bool a_recordingTimestamps, std::set<std::string>& result) -> int {
					host.SetRecordingTimestamps(a_recordingTimestamps);
					{
						std::scoped_lock lock(timedMutex);
						timed.clear();
					}
					for (uint32_t frame = 0; frame < 6; ++frame) {
						value = 97000u + 10u * frame;
						host.SubmitEpoch(0, upload);
						auto execution = host.TakeExternalExecution();
						REQUIRE(execution && execute(*execution), "a timed external execution");
					}
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "the timed executions write and copy");
					// Completions are read on the host's thread. Wait until both passes are read, or until one second passes.
					const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
					do {
						std::this_thread::sleep_for(std::chrono::milliseconds(10));
						std::scoped_lock lock(timedMutex);
						result = timed;
					} while (result.size() < 2 && std::chrono::steady_clock::now() < deadline);
					std::string names;
					for (const auto& name : result) names += " " + name;
					std::printf("external, recording timestamps %s: timed passes%s\n", a_recordingTimestamps ? "on" : "off", names.c_str());
					return 0;
				};
				std::set<std::string> timedOff, timedOn;
				if (const int failed = timedPasses(false, timedOff)) return failed;
				if (const int failed = timedPasses(true, timedOn)) return failed;
				// Replayable recordings, as in external-reuse, are timed only with recording timestamps.
				for (const auto* timedSet : {&timedOff, &timedOn})
					if (timedSet == &timedOn || !externalReuse)
						REQUIRE(timedSet->contains("test.write") && timedSet->contains("test.copy-to-host"),
							"an external epoch's passes are timed, its late pass's too (a replayable one's with recording timestamps)");
				host.SetRecordingTimestamps(false);
				host.SetCompletedFrameCallback({});
				const auto stats = host.TakeAsyncStats();
				std::printf("external: %llu submissions, %llu prepared again\n", static_cast<unsigned long long>(stats.submitted),
					static_cast<unsigned long long>(stats.stale));
				REQUIRE(!counters.unbalanced && counters.locks.load() == counters.unlocks.load(), "balanced host queue locks");
				host.SetAsyncEpochs({});
				host.SetExternalEpochs({});
			}
			if (!revision && !external) {
			if (async) host.SetAsyncEpochs({0});
			if (async) {
				// A submission's owner retires once its work completes, without another frame. This is frame 0's.
				value = 1000u;
				auto owner = std::make_shared<int>(74);
				std::weak_ptr<int> ownerWeak = owner;
				host.SubmitEpoch(0, upload, owner);
				owner.reset();
				REQUIRE(device->WaitIdle() == rhi::Result::Ok, "wait for the owned submission");
				const auto ownerDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
				while (!ownerWeak.expired() && std::chrono::steady_clock::now() < ownerDeadline)
					std::this_thread::yield();
				REQUIRE(ownerWeak.expired(), "a submission's owner retires without another frame");
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
				const auto copies = g_copyRecords.load(std::memory_order_relaxed);
				if (async) {
					if (frame != 0) host.SubmitEpoch(0, upload);
				} else host.ExecuteFrame(nullptr, upload, UINT32_MAX, syncOwner);
				if (inlineLate && (!async || frame != 0)) {
					REQUIRE(g_copyRecords.load(std::memory_order_relaxed) == copies + 1, "the inline late copy is recorded once per submission");
					REQUIRE(!host.TakeExternalExecution(), "an inline epoch hands no execution back");
				}
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
			if (reuse) {
				// Steady frames. A live epoch records every run, replayable or not, and nothing keeps its recordings.
				(void)host.TakeAsyncStats();
				for (uint32_t frame = 0; frame < 12; ++frame) {
					value = 20000u + 10u * frame;
					host.SubmitEpoch(0, upload);
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(second, value), "a run's recording writes its frame's value");
				}
				const auto steady = host.TakeAsyncStats();
				std::printf("reuse: steady %llu recorded, %llu prepared again\n", static_cast<unsigned long long>(steady.recorded),
					static_cast<unsigned long long>(steady.stale));
				// The window's first ticket is recorded ahead, on the host's thread, after the last submission before the window: it may be
				// counted before the window opens.
				REQUIRE(steady.recorded >= 11 && steady.stale == 0, "steady frames record every run");
				// A pass's revision alternates inside the commit (beforeSubmit), as a view's state does. Each ticket was prepared for the
				// other variant, so it is stale at submission and prepared again, for the same run.
				const uint64_t base = outputRevision->load() + 100;
				uint64_t warmStale = 0;
				for (uint32_t frame = 0; frame < 24; ++frame) {
					value = 30000u + 10u * frame;
					host.SubmitEpoch(0, [&](org::RenderGraph& graph) {
						upload(graph);
						outputRevision->store(base + (frame & 1u), std::memory_order_release);
					});
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(second, value), "a prepared-again ticket writes its frame's value");
					if (frame == 11) warmStale = host.TakeAsyncStats().stale;
				}
				const auto alternating = host.TakeAsyncStats();
				std::printf("reuse: alternating (%llu prepared again in the first half) %llu recorded, %llu prepared again\n",
					static_cast<unsigned long long>(warmStale), static_cast<unsigned long long>(alternating.recorded),
					static_cast<unsigned long long>(alternating.stale));
				REQUIRE(alternating.stale >= 6, "a ticket the commit made stale is prepared again");
			}
			if (async) {
				// The commit changes what the write pass depends on: the ticket is stale and prepared again, and the frame still
				// writes its value.
				value = 6500u;
				host.SubmitEpoch(0, [&](org::RenderGraph& graph) {
					upload(graph);
					outputRevision->fetch_add(1, std::memory_order_release);
				});
				REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(second, value), "a ticket made stale by its commit is prepared again");
			}

			// A readback capture after a mid-frame pass. It captures the write pass's target as the pass left it. The copy is
			// recorded in the pass's own recording, so the executable is not built again.
			if (!async && !revision) {
				auto* graph = host.Graph();
				graph->SetScheduleTraceEnabled(true);
				value = 6600u;
				host.ExecuteFrame(nullptr, upload);
				REQUIRE(device->WaitIdle() == rhi::Result::Ok, "wait before the capture");
				const auto traced = graph->GetPersistentScheduleSnapshot();
				REQUIRE(traced && !traced->executables.empty(), "a traced executable");
				const auto revisionBefore = traced->executables.front()->structuralRevision;
				std::vector<uint32_t> captured;
				graph->GetReadbackService()->RequestReadbackCapture("test.write", scratch.get(), {},
					[&](org::ReadbackCaptureResult&& result) {
						captured.resize(result.data.size() / sizeof(uint32_t));
						std::memcpy(captured.data(), result.data.data(), captured.size() * sizeof(uint32_t));
					});
				value = 6700u;
				host.ExecuteFrame(nullptr, upload);
				REQUIRE(device->WaitIdle() == rhi::Result::Ok, "wait for the captured frame");
				graph->GetReadbackService()->ProcessReadbackRequests();
				REQUIRE(captured.size() >= kCount, "the capture completes");
				bool matches = true;
				for (uint32_t i = 0; i < kCount; ++i) matches = matches && captured[i] == value + i;
				REQUIRE(matches, "the capture holds what the pass wrote");
				const auto after = graph->GetPersistentScheduleSnapshot();
				REQUIRE(after && !after->executables.empty() && after->executables.front()->structuralRevision == revisionBefore,
					"the capture did not build the executable again");
				graph->SetScheduleTraceEnabled(false);
			}

			// A rebuild (e.g. on resize) retires the old graph and keeps rendering.
			host.RequestRebuild();
			value = 7000u;
			if (async) host.SubmitEpoch(0, upload);
			else host.ExecuteFrame(nullptr, upload);
			REQUIRE(device->WaitIdle() == rhi::Result::Ok, "wait after rebuild");
			REQUIRE(Matches(second, 7000u), "frame after rebuild");

			// The frame-wait timeline. A frame's batches wait on the GPU for the value that the producer of per-frame values
			// signals. The render thread never waits. The producer signals the value from the dedicated uploader, after its
			// copies.
			{
				auto frameValues = std::make_shared<rhi::TimelinePtr>();
				REQUIRE(device->CreateTimeline(*frameValues, 0, "test.frame-values") == rhi::Result::Ok && *frameValues, "frame-wait timeline");
				host.SetFrameWaitTimeline(frameValues);
				REQUIRE(host.RebuildRequested(), "packets prepared before cannot wait on it: the graph is built again");
				auto uploads = host.RetainUploads();
				const bool dedicated = uploads && uploads->HasDedicatedStreamingQueue();
				for (uint64_t frame = 1; frame <= 3; ++frame) {
					host.SetFrameWaitValue(frame);
					value = 8000u + 10u * static_cast<uint32_t>(frame);
					if (async) host.SubmitEpoch(0, upload);
					else host.ExecuteFrame(nullptr, upload);
					const auto submitted = host.SubmittedPoint();
					REQUIRE(!submitted.points.empty(), "a submitted point");
					if (dedicated) {
						// Not signalled yet: the frame's work has not run.
						std::this_thread::sleep_for(std::chrono::milliseconds(20));
						REQUIRE(!submitted.Reached() && !Matches(second, value), "the frame waits for its values");
						REQUIRE(uploads->QueueStreamingSignal(frameValues, frame), "the producer's signal");
					} else {
						// There is no dedicated queue to signal from, so QueueStreamingSignal refuses. The test signals the value from the
						// compute queue instead.
						REQUIRE(!uploads || !uploads->QueueStreamingSignal(frameValues, frame), "no signal without a dedicated queue");
						const rhi::TimelinePoint signal{(*frameValues)->GetHandle(), frame};
						REQUIRE(device->GetQueue(rhi::QueueKind::Compute).Submit({}, {{}, {&signal, 1}}) == rhi::Result::Ok, "signal from another queue");
					}
					REQUIRE(submitted.Wait(5000) && Matches(second, value), "the frame runs once its values are signalled");
				}
				// No wait: frames run as before.
				host.SetFrameWaitValue(0);
				value = 8100u;
				if (async) host.SubmitEpoch(0, upload);
				else host.ExecuteFrame(nullptr, upload);
				REQUIRE(host.SubmittedPoint().Wait(5000) && Matches(second, value), "a frame without a wait");
				std::printf("frame wait: %s\n", dedicated ? "signalled from the dedicated uploader" : "signalled from another queue");
			}

			// The sealed point: empty until the host seals, then what was submitted at the seal. Later submissions leave it as it was.
			{
				const auto values = [](const org::GraphHost::GpuPoint& a_point) {
					std::vector<std::pair<const void*, uint64_t>> out;
					for (const auto& [timeline, at] : a_point.points) out.emplace_back(timeline.get(), at);
					return out;
				};
				REQUIRE(host.SealedPoint().points.empty(), "nothing sealed before the first seal");
				host.SealSubmitted();
				const auto sealed = values(host.SealedPoint());
				REQUIRE(!sealed.empty() && sealed == values(host.SubmittedPoint()), "the seal takes what was submitted");
				value = 8200u;
				if (async) host.SubmitEpoch(0, upload);
				else host.ExecuteFrame(nullptr, upload);
				REQUIRE(values(host.SealedPoint()) == sealed && values(host.SubmittedPoint()) != sealed, "a later submission is not sealed");
				REQUIRE(host.SubmittedPoint().Wait(5000) && Matches(second, value), "the frame after the seal");
				host.SealSubmitted();
				REQUIRE(values(host.SealedPoint()) == values(host.SubmittedPoint()), "the next seal takes it");
			}
			if (async) host.SetAsyncEpochs({});
			}
			host.DestroyGraph();
		}
		}
	}

	REQUIRE(vkDeviceWaitIdle(host.device) == VK_SUCCESS, "host device survives ORG and BasicRHI shutdown");
	DestroyHostBuffer(host, first);
	DestroyHostBuffer(host, second);
	vkDestroyDevice(host.device, nullptr);
	vkDestroyInstance(host.instance, nullptr);
	std::printf("PersistentVulkanHostTests: ok (%s%s, locks=%d)\n", contributor ? "contributor" : external ? "external" : async ? "async" : "sync",
		inlineLate ? ", inline" : "", counters.locks.load());
	return 0;
}

int main(int argc, char** argv) {
	// An exception the graph throws names what failed; uncaught, it would end the run without saying.
	try {
		return Run(argc, argv);
	} catch (const std::exception& e) {
		std::fprintf(stderr, "FAIL: uncaught exception: %s\n", e.what());
		return 1;
	}
}
