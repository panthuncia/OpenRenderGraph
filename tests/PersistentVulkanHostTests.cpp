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
		auto write = org::RenderGraph::ExternalPassDesc::Compute("test.write",
			std::static_pointer_cast<org::RenderPass>(std::make_shared<WritePass>(m_input, m_scratch, m_program, m_versions, writeQueue, m_hostTexture)))
			.PreferQueue(writeQueue);
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
int TestContributors(rhi::Device device, const std::function<std::shared_ptr<org::Resource>()>& importOutput, const uint32_t* mapped,
	const char* spirvPath,
	rhi::PipelineLayoutHandle layout);

static int Run(int argc, char** argv) {
	// "reuse": async epochs that keep and resubmit their recordings (GraphHost::Desc::reuseRecordings).
	// "external-reuse": "external" with its recordings kept. The late pass is replayable. Each execution of a kept
	// recording records the late pass again.
	const bool externalReuse = argc > 1 && std::strcmp(argv[1], "external-reuse") == 0;
	const bool reuse = externalReuse || (argc > 1 && std::strcmp(argv[1], "reuse") == 0);
	// "revision": a revision-driven epoch, set through the revisionEpochs of SetAsyncEpochs. It is recorded ahead for
	// each revision (RequestEpochRecording) and submitted with that recording.
	const bool revision = argc > 1 && std::strcmp(argv[1], "revision") == 0;
	// "external": an async epoch that the test executes itself (GraphHost::SetExternalEpochs). Its copy pass is
	// recorded at execution, into a primary command buffer the test owns. It is submitted with the execution's waits
	// and signals.
	const bool external = externalReuse || (argc > 1 && std::strcmp(argv[1], "external") == 0);
	const bool async = reuse || revision || external || (argc > 1 && std::strcmp(argv[1], "async") == 0);
	// "contributor": a contributor registered through the C table (ContributorHostTests.cpp) instead of the graph scenario.
	const bool contributor = argc > 1 && std::strcmp(argv[1], "contributor") == 0;
	// "rebinding": slots that start or stop sharing a backing (RebindingGpuTests.cpp) instead of the graph scenario.
	const bool rebinding = argc > 1 && std::strcmp(argv[1], "rebinding") == 0;
	// "arena": two epochs whose transient scratch buffers share the host's alias arena (the second's write on the compute queue
	// when the family has a second queue), against two of their own.
	const bool arena = argc > 1 && std::strcmp(argv[1], "arena") == 0;
	// "split": a host of two epochs, each its own graph: the write pass in the first and the copy to the host buffer in the
	// second (SplitEpochsScenario).
	const bool split = argc > 1 && std::strcmp(argv[1], "split") == 0;
	// "stream": an external epoch with its write on the compute queue (the family's second queue): batches off the host's stream,
	// and the stream's segments.
	const bool stream = argc > 1 && std::strcmp(argv[1], "stream") == 0;
	// "open" after the mode: async epochs on open executions, admitted at submission (no closed executions).
	const bool open = argc > 2 && std::strcmp(argv[2], "open") == 0;
	// "stream open home": the write also reads a host-owned texture, under external home states.
	const bool home = open && argc > 3 && std::strcmp(argv[3], "home") == 0;
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
		if (rebinding) {
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
			.queueBoundary = {.entry = true, .exit = true}, .epochOrder = {0}, .closedExecutions = async && !open, .reuseRecordings = reuse || revision});
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
			host.AddExtension("test.host", [&] { return std::make_unique<HostExtension>(input, scratch, output, program, outputRevision, versions, external); });
			if (revision) {
				host.SetAsyncEpochs({0}, {0});
				// Records the epoch for a revision of `count` elements; the callback runs on the host's thread.
				auto record = [&](uint32_t count, std::shared_ptr<const org::BufferVersion> target) -> std::shared_ptr<const org::GraphHost::EpochRecording> {
					auto data = std::make_shared<TestRevision>();
					data->count = count;
					data->versioned = versions->Key();
					data->target = std::move(target);
					std::atomic<int> state{0};
					std::shared_ptr<const org::GraphHost::EpochRecording> result;
					const bool posted = host.RequestEpochRecording(0, data, [&](std::shared_ptr<const org::GraphHost::EpochRecording> a_recording,
						std::exception_ptr a_error) {
						result = std::move(a_recording);
						state.store(a_error ? 2 : 1, std::memory_order_release);
						state.notify_all();
					});
					if (!posted) return nullptr;
					const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
					while (state.load(std::memory_order_acquire) == 0 && std::chrono::steady_clock::now() < deadline)
						std::this_thread::yield();
					return state.load(std::memory_order_acquire) == 1 ? result : nullptr;
				};
				const auto firstVersion = versions->Current();
				auto full = record(kCount, firstVersion);
				REQUIRE(full && full->slots.size() == 3, "a revision's recording, one per slot of the epoch's ring");
				(void)host.TakeAsyncStats();
				for (uint32_t frame = 0; frame < 6; ++frame) {
					value = 40000u + 10u * frame;
					host.SubmitEpoch(0, upload, {}, full);
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "a revision's recording writes every element");
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
				std::printf("revision: %llu recordings (%llu slots), %llu submissions, %llu recorded at tickets, %llu prepared again\n",
					static_cast<unsigned long long>(stats.revisionRecordings), static_cast<unsigned long long>(stats.revisionSlotsRecorded),
					static_cast<unsigned long long>(stats.revisionSubmissions), static_cast<unsigned long long>(stats.recorded),
					static_cast<unsigned long long>(stats.stale));
				REQUIRE(stats.revisionSubmissions == 13 && stats.stale == 0 && stats.revisionRecordings == 3 && stats.revisionSlotsRecorded == 9,
					"every submission bound a revision's recording; nothing prepared again");
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
				// A live epoch recorded for a revision, before the switch to revision-driven submission. It is only recorded, never
				// admitted. The slot that its live ticket holds is recorded once that ticket is submitted. A slot with work in
				// flight is recorded once that work completes.
				host.SetAsyncEpochs({0});
				{
					// The restarted host's first ticket (the first submission waits for it).
					value = 69990u;
					host.SubmitEpoch(0, upload);
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "the live epoch after the switch");
					(void)host.TakeAsyncStats();
					auto data = std::make_shared<TestRevision>();
					data->count = kCount;
					data->versioned = versions->Key();
					data->target = versions->Current();
					std::atomic<int> state{0};
					std::shared_ptr<const org::GraphHost::EpochRecording> live;
					REQUIRE(host.RequestEpochRecording(0, data, [&](std::shared_ptr<const org::GraphHost::EpochRecording> a_recording, std::exception_ptr a_error) {
						live = std::move(a_recording);
						state.store(a_error ? 2 : 1, std::memory_order_release);
					}), "a live async epoch takes a recording request");
					(void)host.TakeAsyncStats();
					const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
					for (uint32_t frame = 0; state.load(std::memory_order_acquire) == 0 && std::chrono::steady_clock::now() < deadline; ++frame) {
						value = 70000u + 10u * (frame % 1000u);
						host.SubmitEpoch(0, upload);
						REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "the live epoch draws while it is recorded for a revision");
					}
					REQUIRE(state.load(std::memory_order_acquire) == 1 && live && live->slots.size() == 3, "a live epoch's revision recording, one per slot of its ring");
					const auto liveStats = host.TakeAsyncStats();
					std::printf("live revision recording: %llu recordings, %llu revision submissions, %llu submitted, %llu waited\n",
						static_cast<unsigned long long>(liveStats.revisionRecordings), static_cast<unsigned long long>(liveStats.revisionSubmissions),
						static_cast<unsigned long long>(liveStats.submitted), static_cast<unsigned long long>(liveStats.waited));
					// This loop submits right after the GPU is idle, before the host's next ticket. So its frames wait either way.
					REQUIRE(liveStats.revisionRecordings == 1 && liveStats.revisionSubmissions == 0, "admitted, not submitted until chosen");
					// The commit chooses the recording with UseEpochRecording. For every slot, the ticket then takes it instead of its
					// own preparation.
					for (uint32_t frame = 0; frame < 4; ++frame) {
						value = 71000u + 10u * frame;
						bool usable = false;
						host.SubmitEpoch(0, [&](org::RenderGraph& a_graph) {
							upload(a_graph);
							std::string why;
							usable = host.CanUseEpochRecording(*live, &why);
							host.UseEpochRecording(live);
						});
						REQUIRE(usable, "the ticket being submitted takes the revision's recording");
						REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "a live epoch submits the revision's recording it chose");
					}
					// And its own preparation again when it chooses none.
					value = 72000u;
					host.SubmitEpoch(0, upload);
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "a live epoch's own preparation after a chosen recording");
					const auto chosenStats = host.TakeAsyncStats();
					REQUIRE(chosenStats.revisionSubmissions == 4 && chosenStats.stale == 0, "every chosen recording submitted, nothing prepared again");
					// Build the graph again by replacing its extension. The recording now belongs to the previous build. The commit
					// asks before it chooses the recording, and the rebuilt graph's ticket would not take it. So the epoch uses its
					// own preparation.
					const auto builds = host.BuildGeneration();
					host.AddExtension("test.host", [&] { return std::make_unique<HostExtension>(input, scratch, output, program, outputRevision, versions, external); });
					value = 73000u;
					bool usable = true;
					std::string why;
					host.SubmitEpoch(0, [&](org::RenderGraph& a_graph) {
						upload(a_graph);
						usable = host.CanUseEpochRecording(*live, &why);
					});
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "the rebuilt graph's own preparation");
					std::printf("after a rebuild: %s\n", why.c_str());
					REQUIRE(host.BuildGeneration() != builds && !usable && !why.empty(), "a recording of the build before is not the rebuilt graph's ticket's");
					// Explicit builds. An added extension waits for the build point (BuildIfRequested). Until then the graph runs as
					// it was built, and its tickets can still take a recording of it. The build point builds the graph. The recording
					// then belongs to the previous build.
					host.SetExplicitBuilds(true);
					std::shared_ptr<const org::GraphHost::EpochRecording> held;
					state.store(0, std::memory_order_relaxed);
					REQUIRE(host.RequestEpochRecording(0, data, [&](std::shared_ptr<const org::GraphHost::EpochRecording> a_recording, std::exception_ptr a_error) {
						held = std::move(a_recording);
						state.store(a_error ? 2 : 1, std::memory_order_release);
					}), "a recording request of the current build");
					const auto heldDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
					for (uint32_t frame = 0; state.load(std::memory_order_acquire) == 0 && std::chrono::steady_clock::now() < heldDeadline; ++frame) {
						value = 74000u + 10u * (frame % 100u);
						host.SubmitEpoch(0, upload);
						REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "the live epoch draws while it is recorded");
					}
					REQUIRE(state.load(std::memory_order_acquire) == 1 && held && host.EpochRecordingCurrent(*held), "a recording of the current build");
					const auto explicitBuilds = host.BuildGeneration();
					host.AddExtension("test.host", [&] { return std::make_unique<HostExtension>(input, scratch, output, program, outputRevision, versions, external); });
					REQUIRE(host.RebuildRequested(), "the extension added is a rebuild requested");
					for (uint32_t frame = 0; frame < 4; ++frame) {
						value = 75000u + 10u * frame;
						usable = false;
						host.SubmitEpoch(0, [&](org::RenderGraph& a_graph) {
							upload(a_graph);
							usable = host.CanUseEpochRecording(*held, &why);
							if (usable)
								host.UseEpochRecording(held);
						});
						REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "the graph as built draws until the build point");
						REQUIRE(usable && host.BuildGeneration() == explicitBuilds && host.EpochRecordingCurrent(*held), "its recording is its tickets' until the build point");
					}
					REQUIRE(host.BuildIfRequested() && !host.RebuildRequested() && host.BuildGeneration() != explicitBuilds && !host.EpochRecordingCurrent(*held),
						"the build point builds; the recording is the build before's");
					REQUIRE(!host.BuildIfRequested(), "nothing to build without a request");
					value = 76000u;
					host.SubmitEpoch(0, [&](org::RenderGraph& a_graph) {
						upload(a_graph);
						usable = host.CanUseEpochRecording(*held, &why);
					});
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value) && !usable, "the rebuilt graph's own preparation");
					host.SetExplicitBuilds(false);
					// An epoch that the caller does not submit. Its ready ticket holds a slot of its ring. So a recording requested for
					// it waits for that slot until the caller gives the ticket back (ReleaseEpochTicket). Then the recording completes,
					// without being submitted.
					std::shared_ptr<const org::GraphHost::EpochRecording> idle;
					state.store(0, std::memory_order_relaxed);
					const auto readyDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
					while (!host.ReleaseEpochTicket(0) && std::chrono::steady_clock::now() < readyDeadline)
						std::this_thread::yield();
					REQUIRE(host.RequestEpochRecording(0, data, [&](std::shared_ptr<const org::GraphHost::EpochRecording> a_recording, std::exception_ptr a_error) {
						idle = std::move(a_recording);
						state.store(a_error ? 2 : 1, std::memory_order_release);
					}), "a recording request for an epoch not submitted");
					std::this_thread::sleep_for(std::chrono::milliseconds(200));
					REQUIRE(state.load(std::memory_order_acquire) == 0, "it waits for the slot the epoch's ready ticket holds");
					const auto idleDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
					while (state.load(std::memory_order_acquire) == 0 && std::chrono::steady_clock::now() < idleDeadline) {
						(void)host.ReleaseEpochTicket(0);
						std::this_thread::sleep_for(std::chrono::milliseconds(5));
					}
					REQUIRE(state.load(std::memory_order_acquire) == 1 && idle && idle->slots.size() == 3, "given the ticket back, the recording completes unsubmitted");
					value = 77000u;
					host.SubmitEpoch(0, upload);
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), "the epoch submits again after its ticket was given back");
				}
				host.SetAsyncEpochs({});
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
					const auto kept = host.TakeAsyncStats();
					std::printf("external-reuse: %llu recorded, %llu kept, %llu reused\n", static_cast<unsigned long long>(kept.recorded),
						static_cast<unsigned long long>(kept.kept), static_cast<unsigned long long>(kept.reused));
					REQUIRE(kept.kept > 0 && kept.reused > 0, "an external epoch with a replayable late pass is kept and taken again");
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
				// Kept recordings, as in external-reuse, are timed only with recording timestamps. This is true of every kept
				// recording.
				for (const auto* timedSet : {&timedOff, &timedOn})
					if (timedSet == &timedOn || !externalReuse)
						REQUIRE(timedSet->contains("test.write") && timedSet->contains("test.copy-to-host"),
							"an external epoch's passes are timed, its late pass's too (a kept one's with recording timestamps)");
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
			if (reuse) {
				// Steady frames. The slots' tickets take their kept recordings. Each submission of the same command lists reads the
				// upload of its own frame.
				(void)host.TakeAsyncStats();
				for (uint32_t frame = 0; frame < 12; ++frame) {
					value = 20000u + 10u * frame;
					host.SubmitEpoch(0, upload);
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(second, value), "a resubmitted recording writes its frame's value");
				}
				const auto steady = host.TakeAsyncStats();
				std::printf("reuse: steady %llu recorded, %llu kept, %llu reused, %llu adopted, %llu prepared again\n",
					static_cast<unsigned long long>(steady.recorded), static_cast<unsigned long long>(steady.kept),
					static_cast<unsigned long long>(steady.reused), static_cast<unsigned long long>(steady.adopted),
					static_cast<unsigned long long>(steady.stale));
				REQUIRE(steady.reused >= 9 && steady.stale == 0, "steady frames take kept recordings");
				// A pass's revision alternates inside the commit (beforeSubmit), as a view's state does. Each slot's ticket was
				// prepared for the other variant, so it is stale at submission. Once every slot has kept both variants, a stale
				// ticket adopts the kept recording, and nothing is prepared again.
				const uint64_t base = outputRevision->load() + 100;
				uint64_t warmStale = 0;
				for (uint32_t frame = 0; frame < 24; ++frame) {
					value = 30000u + 10u * frame;
					host.SubmitEpoch(0, [&](org::RenderGraph& graph) {
						upload(graph);
						outputRevision->store(base + (frame & 1u), std::memory_order_release);
					});
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(second, value), "an adopted recording writes its frame's value");
					if (frame == 11) warmStale = host.TakeAsyncStats().stale;
				}
				const auto alternating = host.TakeAsyncStats();
				std::printf("reuse: alternating (after %llu prepared again to warm up) %llu recorded, %llu reused, %llu adopted, %llu prepared again\n",
					static_cast<unsigned long long>(warmStale), static_cast<unsigned long long>(alternating.recorded),
					static_cast<unsigned long long>(alternating.reused), static_cast<unsigned long long>(alternating.adopted),
					static_cast<unsigned long long>(alternating.stale));
				REQUIRE(alternating.stale == 0 && alternating.adopted >= 6, "warm alternating revisions adopt kept recordings");
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
	std::printf("PersistentVulkanHostTests: ok (%s, locks=%d)\n", contributor ? "contributor" : external ? "external" : async ? "async" : "sync", counters.locks.load());
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
