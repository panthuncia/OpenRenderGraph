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

#include <array>
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
#include "TestTaskService.h"
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
#include "Managers/Singletons/StatisticsManager.h"
#include "OpenRenderGraph/ExternalClient.h"

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
// "sync-reuse": an input of the write pass's frame data that its revision omits, for the revision audit.
std::atomic<uint32_t> g_unrevisedInput{0};
// "sync-reuse", batch reuse: the write pass's preparations (a batch submitted from a kept recording prepares nothing).
std::atomic<uint32_t> g_writePrepares{0};
// "sync-submit": the copy pass's next Record throws (a failing pass on the submission thread).
std::atomic<uint32_t> g_failCopyRecord{0};
// "sync-submit" (alone, or after "placement" or "shared-rebuild"): synchronous executions on the submission thread
// (GraphHost::Desc::executeOnSubmissionThread), with queued seals.
bool g_syncSubmit = false;
org::GraphHost::Desc WithSyncSubmit(org::GraphHost::Desc desc) {
	if (!g_syncSubmit) return desc;
	desc.submissionThread = true;
	desc.queuedSeals = true;
	desc.executeOnSubmissionThread = true;
	return desc;
}
// A device wait after the host's submissions: what the submission thread still holds goes to the device first.
rhi::Result WaitHostIdle(org::GraphHost& host, rhi::Device device) {
	if (g_syncSubmit) host.DrainSubmissions("test: a device wait");
	return device.WaitIdle();
}
// Holds the submission thread until the returned gate opens (OpenGate), so that what is queued behind stays queued.
std::shared_ptr<std::atomic<bool>> HoldSubmissionThread(org::GraphHost& host) {
	auto gate = std::make_shared<std::atomic<bool>>(false);
	host.EnqueueOnSubmitThread([gate] {
		while (!gate->load(std::memory_order_acquire)) gate->wait(false, std::memory_order_acquire);
	});
	return gate;
}
void OpenGate(const std::shared_ptr<std::atomic<bool>>& gate) {
	gate->store(true, std::memory_order_release);
	gate->notify_all();
}
// Opens the gate from another thread after a delay, while this thread waits for what the gate holds.
std::thread OpenGateLater(std::shared_ptr<std::atomic<bool>> gate, int milliseconds) {
	return std::thread([gate = std::move(gate), milliseconds] {
		std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
		OpenGate(gate);
	});
}

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
	// Set by the audit test (g_unrevisedInput), named by no revision, read by no Record.
	uint32_t unrevised = 0;
	bool operator==(const WriteFrame&) const = default;
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
		g_writePrepares.fetch_add(1, std::memory_order_relaxed);
		return {m_program, preparation.Resolve(bindings.input).index,
			preparation.Resolve(bindings.target).index, RevisionCount(preparation), g_unrevisedInput.load(std::memory_order_relaxed)};
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
		if (g_failCopyRecord.exchange(0, std::memory_order_acq_rel)) throw std::runtime_error("test: the copy pass fails to record");
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

// CopyToHostPass without a revision ("sync-reuse", batch reuse): a recording with it is never current as a whole, but with batch
// reuse the other batches of its epoch still are.
class UncheckedCopyPass final : public org::TypedRenderGraphPass<UncheckedCopyPass, org::EmptyPassFrameData, CopyBindings> {
public:
	UncheckedCopyPass(std::shared_ptr<org::ExternalBufferResource> destination, std::shared_ptr<const org::VersionedBuffer> versions)
		: m_destination(std::move(destination)), m_versions(std::move(versions)) {}
	CopyBindings Declare(org::PassBuilder& builder) {
		return {builder.CopySource(*m_versions), builder.CopyDestination(m_destination)};
	}
	static void Record(const CopyBindings& bindings, org::PassRecordContext& recording) {
		g_copyRecords.fetch_add(1, std::memory_order_relaxed);
		recording.Commands().CopyBufferRegion(recording.Resolve(bindings.destination).GetHandle(), 0,
			recording.Resolve(bindings.source).GetHandle(), 0, kBytes);
	}
private:
	std::shared_ptr<org::ExternalBufferResource> m_destination;
	std::shared_ptr<const org::VersionedBuffer> m_versions;
};

// Where an extension instance was shut down and destroyed ("retired": GraphHost::Desc::destroyRetiredGraphsDetached).
struct ExtensionEnds {
	std::mutex mutex;
	std::vector<std::thread::id> shutdownOn, destroyedOn;
};

class EndProbeExtension final : public org::RenderGraph::IRenderGraphExtension {
public:
	explicit EndProbeExtension(std::shared_ptr<ExtensionEnds> ends) : m_ends(std::move(ends)) {}
	~EndProbeExtension() override {
		std::scoped_lock lock(m_ends->mutex);
		m_ends->destroyedOn.push_back(std::this_thread::get_id());
	}
	void Shutdown(org::RenderGraph&) override {
		std::scoped_lock lock(m_ends->mutex);
		m_ends->shutdownOn.push_back(std::this_thread::get_id());
	}
	void GatherStructuralPasses(org::RenderGraph&, std::vector<org::RenderGraph::ExternalPassDesc>&) override {}
private:
	std::shared_ptr<ExtensionEnds> m_ends;
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
	// "sync-reuse", batch reuse: the copy reports no revision (UncheckedCopyPass).
	void SetUncheckedCopy(bool unchecked) { m_uncheckedCopy = unchecked; }
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
			m_uncheckedCopy && m_versions ? std::static_pointer_cast<org::RenderPass>(std::make_shared<UncheckedCopyPass>(m_output, m_versions))
				: std::static_pointer_cast<org::RenderPass>(std::make_shared<CopyToHostPass>(m_scratch, m_output, m_outputRevision, m_versions, copyQueue)))
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
	bool m_uncheckedCopy = false;
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

// "placement": writes into scratch buffers that the frame placement places, and their copies to host buffers. Every scratch
// buffer has the same name and size, so only their ordinals tell their keys apart (PlacementUse::key).
class PlacementExtension final : public org::RenderGraph::IRenderGraphExtension {
public:
	struct Pair {
		std::shared_ptr<org::Buffer> scratch;
		std::shared_ptr<org::ExternalBufferResource> output;
	};
	PlacementExtension(std::shared_ptr<org::Buffer> input, std::vector<Pair> pairs, std::shared_ptr<const ComputeProgram> program,
		std::shared_ptr<std::atomic<uint64_t>> outputRevision)
		: m_input(std::move(input)), m_pairs(std::move(pairs)), m_program(std::move(program)), m_outputRevision(std::move(outputRevision)) {}
	void PrepareForBuild(org::RenderGraph& graph) override {
		graph.RegisterResource(org::ResourceIdentifier("test.placement.input"), m_input);
		for (size_t i = 0; i < m_pairs.size(); ++i) {
			graph.RegisterResource(org::ResourceIdentifier("test.placement.scratch." + std::to_string(i)), m_pairs[i].scratch);
			graph.RegisterResource(org::ResourceIdentifier("test.placement.output." + std::to_string(i)), m_pairs[i].output);
		}
	}
	void GatherStructuralPasses(org::RenderGraph&, std::vector<org::RenderGraph::ExternalPassDesc>& out) override {
		for (size_t i = 0; i < m_pairs.size(); ++i) {
			const auto suffix = std::to_string(i);
			out.push_back(org::RenderGraph::ExternalPassDesc::Compute("test.placement.write." + suffix,
				std::static_pointer_cast<org::RenderPass>(std::make_shared<WritePass>(m_input, m_pairs[i].scratch, m_program))));
			out.push_back(org::RenderGraph::ExternalPassDesc::Copy("test.placement.copy." + suffix,
				std::static_pointer_cast<org::RenderPass>(std::make_shared<CopyToHostPass>(m_pairs[i].scratch, m_pairs[i].output, m_outputRevision))));
		}
	}
private:
	std::shared_ptr<org::Buffer> m_input;
	std::vector<Pair> m_pairs;
	std::shared_ptr<const ComputeProgram> m_program;
	std::shared_ptr<std::atomic<uint64_t>> m_outputRevision;
};

// "staged-placement": two epochs with frame placement. Epoch 0 writes a scratch buffer that epoch 1 copies to a host buffer (one
// object both epochs' graphs use), and epoch 1 writes and copies a scratch buffer of its own.
class StagedPlacementExtension final : public org::RenderGraph::IRenderGraphExtension {
public:
	StagedPlacementExtension(std::shared_ptr<org::Buffer> input, std::shared_ptr<org::Buffer> shared, std::shared_ptr<org::ExternalBufferResource> sharedOutput,
		std::shared_ptr<org::Buffer> own, std::shared_ptr<org::ExternalBufferResource> ownOutput, std::shared_ptr<const ComputeProgram> program,
		std::shared_ptr<std::atomic<uint64_t>> outputRevision)
		: m_input(std::move(input)), m_shared(std::move(shared)), m_sharedOutput(std::move(sharedOutput)), m_own(std::move(own)),
		m_ownOutput(std::move(ownOutput)), m_program(std::move(program)), m_outputRevision(std::move(outputRevision)) {}
	// The input the second epoch's own write reads (the shared write's by default).
	void SetOwnInput(std::shared_ptr<org::Buffer> input) { m_ownInput = std::move(input); }
	// A transient the first epoch writes alone, kept across rebuilds (the same object in every graph).
	void SetKept(std::shared_ptr<org::Buffer> kept) { m_kept = std::move(kept); }
	void PrepareForBuild(org::RenderGraph& graph) override {
		graph.RegisterResource(org::ResourceIdentifier("test.staged.input"), m_input);
		if (m_ownInput) graph.RegisterResource(org::ResourceIdentifier("test.staged.own-input"), m_ownInput);
		if (m_kept) graph.RegisterResource(org::ResourceIdentifier("test.staged.kept"), m_kept);
		graph.RegisterResource(org::ResourceIdentifier("test.staged.shared"), m_shared);
		graph.RegisterResource(org::ResourceIdentifier("test.staged.shared-output"), m_sharedOutput);
		graph.RegisterResource(org::ResourceIdentifier("test.staged.own"), m_own);
		graph.RegisterResource(org::ResourceIdentifier("test.staged.own-output"), m_ownOutput);
	}
	void GatherStructuralPasses(org::RenderGraph&, std::vector<org::RenderGraph::ExternalPassDesc>& out) override {
		if (m_kept)
			out.push_back(org::RenderGraph::ExternalPassDesc::Compute("test.staged.write-kept",
				std::static_pointer_cast<org::RenderPass>(std::make_shared<WritePass>(m_input, m_kept, m_program))).Epoch(0));
		out.push_back(org::RenderGraph::ExternalPassDesc::Compute("test.staged.write-shared",
			std::static_pointer_cast<org::RenderPass>(std::make_shared<WritePass>(m_input, m_shared, m_program))).Epoch(0));
		out.push_back(org::RenderGraph::ExternalPassDesc::Copy("test.staged.copy-shared",
			std::static_pointer_cast<org::RenderPass>(std::make_shared<CopyToHostPass>(m_shared, m_sharedOutput, m_outputRevision))).Epoch(1));
		out.push_back(org::RenderGraph::ExternalPassDesc::Compute("test.staged.write-own",
			std::static_pointer_cast<org::RenderPass>(std::make_shared<WritePass>(m_ownInput ? m_ownInput : m_input, m_own, m_program))).Epoch(1));
		out.push_back(org::RenderGraph::ExternalPassDesc::Copy("test.staged.copy-own",
			std::static_pointer_cast<org::RenderPass>(std::make_shared<CopyToHostPass>(m_own, m_ownOutput, m_outputRevision))).Epoch(1));
	}
private:
	std::shared_ptr<org::Buffer> m_input, m_shared, m_ownInput, m_kept;
	std::shared_ptr<org::ExternalBufferResource> m_sharedOutput;
	std::shared_ptr<org::Buffer> m_own;
	std::shared_ptr<org::ExternalBufferResource> m_ownOutput;
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
// meets every segment's split). The caller may execute the next before the last completes: a primary is reused only once
// its last submission has completed (the executor's own timeline, signalled with every segment), else a new one is made.
struct SegmentExecutor {
	struct Primary {
		rhi::CommandAllocatorPtr allocator;
		rhi::CommandListPtr list;
		uint64_t submitted = 0; // the executor's timeline value its last submission signals
	};
	rhi::Device device;
	std::vector<Primary> primaries;
	rhi::TimelinePtr done;
	uint64_t doneValue = 0;
	bool Execute(org::ExternalEpochExecution& a_execution) {
		auto queue = device.GetQueue(rhi::QueueKind::Graphics);
		if (!done && device.CreateTimeline(done, 0, "SegmentExecutor") != rhi::Result::Ok) return false;
		for (size_t index = 0; index < a_execution.segments.size(); ++index) {
			const uint64_t completed = done->GetCompletedValue();
			auto free = std::ranges::find_if(primaries, [&](const Primary& a_primary) { return a_primary.submitted <= completed; });
			if (free == primaries.end()) {
				auto& made = primaries.emplace_back();
				if (device.CreateCommandAllocator(rhi::QueueKind::Graphics, made.allocator) != rhi::Result::Ok
					|| device.CreateCommandList(rhi::QueueKind::Graphics, made.allocator.Get(), made.list) != rhi::Result::Ok)
					return false;
				made.list->End();
				free = std::prev(primaries.end());
			}
			auto& primary = *free;
			primary.allocator->Recycle();
			primary.list->Recycle(primary.allocator.Get());
			a_execution.Execute(index, primary.list.Get());
			if (primary.list->EndChecked() != rhi::Result::Ok) return false;
			auto list = primary.list.Get();
			const auto& segment = a_execution.segments[index];
			auto waits = segment.waits;
			auto signals = segment.signals;
			primary.submitted = ++doneValue;
			signals.push_back({done->GetHandle(), primary.submitted});
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
int TestFramePlacementCompute();
int TestFramePlacementStaging();
int TestFramePlacementKeeping();
int TestContributors(rhi::Device device, const std::function<std::shared_ptr<org::Resource>()>& importOutput, const uint32_t* mapped,
	const char* spirvPath,
	rhi::PipelineLayoutHandle layout);

static int Run(int argc, char** argv) {
	// "reuse": async epochs that record replayable recordings (GraphHost::Desc::reuseRecordings). A live epoch's runs take kept
	// copies, and a copy's next run binds the recording it kept while that recording is current.
	// "external-reuse": "external" with replayable recordings. The late pass is replayable.
	const bool externalReuse = argc > 1 && std::strcmp(argv[1], "external-reuse") == 0;
	const bool reuse = externalReuse || (argc > 1 && std::strcmp(argv[1], "reuse") == 0);
	// "revision": a revision-driven epoch, set through the revisionEpochs of SetAsyncEpochs. It is recorded ahead for
	// each revision (RequestEpochRecording) and submitted with that recording.
	// "segment": "revision" with its copies taken by the caller's frame slot (SetEpochCopy, EpochCopyRun), and a revision that turns the write pass off (SetSkipWhen).
	const bool segment = argc > 1 && std::strcmp(argv[1], "segment") == 0;
	const bool revision = segment || (argc > 1 && std::strcmp(argv[1], "revision") == 0);
	// "external": an async epoch that the test executes itself (GraphHost::SetExternalEpochs). Its copy pass is
	// recorded at execution, into a primary command buffer the test owns. It is submitted with the execution's waits
	// and signals.
	const bool external = externalReuse || (argc > 1 && std::strcmp(argv[1], "external") == 0);
	const bool async = reuse || revision || external || (argc > 1 && std::strcmp(argv[1], "async") == 0);
	// "sync-reuse": synchronous frames with kept recordings (GraphHost::SetKeptEpochs). Steady frames submit their copy's kept
	// recording; a pass's revision, a version of a slot the epoch binds and a rebuild record again, each with its reason; a
	// version of a slot the epoch does not bind leaves the recordings current. With "inline", the late copy is recorded at every
	// submission, the kept ones included.
	// "sync-submit": "sync-reuse" with the executions on the submission thread (GraphHost::Desc::executeOnSubmissionThread) and
	// queued seals. Every check of "sync-reuse" holds, each device wait draining the thread first. Then: an execution still queued
	// when ExecuteFrame returns, the epoch's next ExecuteFrame waiting for it, frames back to back, a graph staged by run attached
	// while the last run's kept copy is still queued (that copy is not free, so the new graph's first run takes the next one),
	// and a pass that fails to record on the thread (the epoch's next ExecuteFrame throws EpochFailedError; a rebuild recovers).
	// Then external client submissions (O-M7-3): one made with nothing queued goes straight to the queue; one made while the
	// epoch's execution is still queued is queued behind it and copies what that execution wrote; one made on the thread itself
	// is made there. And kept recordings' timestamps from shared pool slices (StatisticsManager::CreateOwnedTimestamps).
	// After "placement" or "shared-rebuild": those scenarios with their executions on the thread; "shared-rebuild" also
	// attaches each set while both epochs' executions are queued.
	for (int arg = 1; arg < argc; ++arg) g_syncSubmit = g_syncSubmit || std::strcmp(argv[arg], "sync-submit") == 0;
	const bool syncSubmit = argc > 1 && std::strcmp(argv[1], "sync-submit") == 0;
	const bool syncReuse = syncSubmit || (argc > 1 && std::strcmp(argv[1], "sync-reuse") == 0);
	// "contributor": a contributor registered through the C table (ContributorHostTests.cpp) instead of the graph scenario.
	const bool contributor = argc > 1 && std::strcmp(argv[1], "contributor") == 0;
	// "rebinding": slots that start or stop sharing a backing (RebindingGpuTests.cpp) instead of the graph scenario.
	const bool rebinding = argc > 1 && std::strcmp(argv[1], "rebinding") == 0;
	// "arena": two epochs whose transient scratch buffers share the host's alias arena (the second's write on the compute queue
	// when the family has a second queue), against two of their own.
	const bool arena = argc > 1 && std::strcmp(argv[1], "arena") == 0;
	// "placement": frame placement with one epoch (GraphHost::SetFramePlacement): two transients of one name and description
	// are both placed from it, the first plan places from the first version, a rebuilt graph's new resources take their
	// predecessors' entries, new versions are taken in place without a rebuild, and heaps grown with headroom are trimmed.
	const bool placement = argc > 1 && std::strcmp(argv[1], "placement") == 0;
	// "staged-placement": frame placement with two epochs and staged sets (GraphHost::PlaceEpochGraphs): a set placed on the thread
	// that built it while the attached graphs keep rendering, a version that keeps the running graphs' resources (taken by them as
	// it is), a dropped set that publishes nothing, and one version a frame while the worker publishes. Also the deviceless staging
	// and freeze checks, and versions that keep resources on real heaps (FramePlacementTests.cpp).
	const bool stagedPlacement = argc > 1 && std::strcmp(argv[1], "staged-placement") == 0;
	// "shared-rebuild": two epochs of a synchronous host with a shared alias arena, a transient written in the first and read in
	// the second (one object both graphs bind), and a transient of the second's own. Then sets of both epochs built for new objects
	// of other sizes (a resize: the renderer makes its frame targets again), placed at their build, staged and attached together.
	// With the frame placement off and on. No graph may place the shared transient in its own pools (they see only its own uses),
	// move it under the other's executable, or leave a binding to a backing that moved since: each frame's copy reads what the
	// first epoch wrote.
	const bool sharedRebuild = argc > 1 && std::strcmp(argv[1], "shared-rebuild") == 0;
	// "split": a host of two epochs, each its own graph: the write pass in the first and the copy to the host buffer in the
	// second (SplitEpochsScenario).
	const bool split = argc > 1 && std::strcmp(argv[1], "split") == 0;
	// "retired": a synchronous host that destroys its retired graphs on the runtime's deleter thread
	// (Desc::destroyRetiredGraphsDetached): a graph replaced by a rebuild is shut down on the owner thread and destroyed on
	// another, while the new graph keeps running; DestroyGraph still destroys on the owner thread.
	const bool retired = argc > 1 && std::strcmp(argv[1], "retired") == 0;
	// "built": application-driven builds (GraphHost::BuildEpochGraph, AttachEpochGraph): a build on a worker thread while both
	// epochs submit, attached between frames; a failed build; stale builds rejected; staged sets; BuildsOf read on other threads
	// while the owner stages and attaches; and stale alias placements reported (Desc::reportStaleAliasPlacements) instead of rebuilt.
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
	// "leases-wide": the same, with 260 runs a frame, so lease indices pass 255, and with replayable recordings. Then a run takes
	// one of the epoch's kept copies, and once every copy holds an unsealed run, a lease of its own.
	const bool wide = argc > 1 && std::strcmp(argv[1], "leases-wide") == 0;
	// "open" after the mode: the name the mode's output gives. Every execution is open, admitted at submission.
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
	// probe: "sync-submit"'s external client copies (O-M7-3).
	HostBuffer first, second, probe;
	REQUIRE(CreateHostBuffer(host, first) && CreateHostBuffer(host, second) && CreateHostBuffer(host, probe), "host buffers");

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
				.tasks = std::make_shared<org::test::TestTaskService>(2)});
			const int line = TestRebindingGpu(device.Get(), runtime);
			runtime.reset();
			if (line) std::printf("FAIL: rebinding (line %d)\n", line);
			REQUIRE(line == 0, "rebinding");
		} else if (retired) {
			auto runtime = org::Runtime::Create({.device = device.Get(), .backend = rhi::Backend::Vulkan, .slots = 4,
				.tasks = std::make_shared<org::test::TestTaskService>(2)});
			{
				// "retired sync-submit": the executions on the submission thread, so a replaced graph's last execution may still be
				// queued when it is replaced. It goes to the deleter thread only once its retirement point has passed and is reached.
				org::GraphHost host(WithSyncSubmit({.runtime = runtime, .queueBoundary = {.entry = true, .exit = true}, .framesInFlight = 2,
					.epochOrder = {0}, .destroyRetiredGraphsDetached = true}));
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
				auto ends = std::make_shared<ExtensionEnds>();
				host.AddExtension("test.host", [&] {
					return std::make_unique<HostExtension>(input, scratch, output, program, outputRevision);
				});
				host.AddExtension("test.end-probe", [ends] { return std::make_unique<EndProbeExtension>(ends); });
				const auto owner = std::this_thread::get_id();
				const auto ended = [&](size_t count) {
					std::scoped_lock lock(ends->mutex);
					return ends->destroyedOn.size() >= count;
				};
				for (uint32_t frame = 0; frame < 12; ++frame) {
					if (frame == 3 || frame == 6) host.RequestRebuild(0);
					value = 42000u + 10u * frame;
					host.ExecuteFrame(nullptr, upload, 0);
					host.SealSubmitted();
					REQUIRE(WaitHostIdle(host, device.Get()) == rhi::Result::Ok && Matches(first, value), "retired: the new graph runs while the old one is destroyed");
				}
				// The deleter thread runs on its own: wait for it a while.
				for (int wait = 0; wait < 200 && !ended(2); ++wait) std::this_thread::sleep_for(std::chrono::milliseconds(10));
				{
					std::scoped_lock lock(ends->mutex);
					REQUIRE(ends->shutdownOn.size() == 2 && ends->destroyedOn.size() == 2, "retired: both replaced graphs were shut down and destroyed");
					for (const auto id : ends->shutdownOn) REQUIRE(id == owner, "retired: a replaced graph's extensions are shut down on the owner thread");
					for (const auto id : ends->destroyedOn) REQUIRE(id != owner, "retired: a replaced graph is destroyed on another thread");
				}
				host.DestroyGraph();
				{
					std::scoped_lock lock(ends->mutex);
					REQUIRE(ends->destroyedOn.size() == 3 && ends->destroyedOn.back() == owner, "retired: DestroyGraph destroys on the owner thread");
				}
				REQUIRE(!counters.unbalanced && counters.locks.load() == counters.unlocks.load(), "balanced host queue locks");
				std::puts("retired: replaced graphs destroyed on the deleter thread");
			}
			runtime.reset();
		} else if (split) {
			auto runtime = org::Runtime::Create({.device = device.Get(), .backend = rhi::Backend::Vulkan, .slots = 4,
				.tasks = std::make_shared<org::test::TestTaskService>(2)});
			{
				org::GraphHost host({.runtime = runtime, .queueBoundary = {.entry = true, .exit = true}, .framesInFlight = 2, .epochOrder = {0, 1}});
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
				// Staged sets without async epochs (GraphHost::StageEpochGraph, AttachStagedEpochGraphs): both epochs are built and
				// staged, the attached graphs keep running, a refused set keeps both staged, a dropped graph leaves the set incomplete,
				// and the set attaches in one step.
				{
					uint32_t frame = 0;
					const auto run = [&](const char* what) -> int {
						value = 40500u + 10u * frame++;
						host.ExecuteFrame(nullptr, upload, 0);
						host.ExecuteFrame(nullptr, {}, 1);
						host.SealSubmitted();
						REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value), what);
						return 0;
					};
					const auto stage = [&](uint32_t epoch, const char* why) -> bool {
						auto graph = host.BuildEpochGraph(epoch, why);
						const auto staged = graph && graph->Built() ? host.StageEpochGraph(graph) : org::GraphHost::StageResult{};
						return staged.staged && !staged.attached && !graph;
					};
					auto* graph0 = host.Graph(0);
					auto* graph1 = host.Graph(1);
					const auto builds0 = host.BuildGeneration(0), builds1 = host.BuildGeneration(1);
					const auto before = host.AttachStatsTotal();
					REQUIRE(stage(0, "test: a synchronous staged set") && stage(1, "test: a synchronous staged set"), "sync: both epochs staged");
					const auto staged0 = host.StagedGeneration(0), staged1 = host.StagedGeneration(1);
					REQUIRE(staged0 && staged1, "sync: both have a staged generation");
					for (int i = 0; i < 2; ++i)
						if (const int failed = run("sync staged: the attached graphs keep running")) return failed;
					REQUIRE(host.Graph(0) == graph0 && host.Graph(1) == graph1 && host.BuildGeneration(0) == builds0 && host.BuildGeneration(1) == builds1,
						"sync: staging attaches nothing");
					const uint32_t both[] = {0, 1};
					const uint32_t twice[] = {0, 1, 1};
					const auto refused = host.AttachStagedEpochGraphs(twice);
					REQUIRE(!refused.attached && refused.refusal == org::GraphHost::AttachRefusal::Duplicate, "sync: a set naming an epoch twice is refused");
					REQUIRE(host.StagedGeneration(0) == staged0 && host.StagedGeneration(1) == staged1 && host.Graph(0) == graph0 && host.Graph(1) == graph1,
						"sync: a refused set keeps both staged and attaches nothing");
					// A staged graph dropped: the set is incomplete and refused as a whole; epoch 0 keeps its staged graph.
					const uint32_t one[] = {1};
					REQUIRE(host.DropStagedEpochGraphs(one) == 1 && host.StagedGeneration(1) == 0 && host.StagedGeneration(0) == staged0,
						"sync: one staged graph dropped");
					REQUIRE(host.AttachStagedEpochGraphs(both).refusal == org::GraphHost::AttachRefusal::NotStaged && host.Graph(0) == graph0,
						"sync: a set missing an epoch's graph attaches nothing");
					if (const int failed = run("sync staged: frames after a refusal")) return failed;
					// Staged again: a second staging of epoch 0 replaces its first.
					REQUIRE(stage(0, "test: staged again") && stage(1, "test: staged again"), "sync: both staged again");
					REQUIRE(host.StagedGeneration(0) > staged0, "sync: the second staging replaced the first");
					const auto next0 = host.StagedGeneration(0), next1 = host.StagedGeneration(1);
					REQUIRE(host.AttachStagedEpochGraphs(both).attached, "sync: the set attaches");
					REQUIRE(host.Graph(0) != graph0 && host.Graph(1) != graph1 && host.BuildGeneration(0) == next0 && host.BuildGeneration(1) == next1,
						"sync: both epochs run their staged builds");
					REQUIRE(host.StagedGeneration(0) == 0 && host.StagedGeneration(1) == 0, "sync: nothing is left staged");
					for (int i = 0; i < 4; ++i)
						if (const int failed = run("sync staged: the attached set runs")) return failed;
					const auto after = host.AttachStatsTotal();
					REQUIRE(after.setsAttached == before.setsAttached + 1 && after.staged == before.staged + 4, "sync: one set attached of four stagings");
					std::puts("split: a synchronous staged set attached together");
					(void)host.TakeAsyncStats();
				}
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
				.tasks = std::make_shared<org::test::TestTaskService>(2)});
			{
				// "built sync-submit": with the host's submission thread and its synchronous executions there (O-M7-2: replaced graphs
				// retired by a pending point, without a drain).
				org::GraphHost host(WithSyncSubmit({.runtime = runtime, .queueBoundary = {.entry = true, .exit = true}, .framesInFlight = 2,
					.epochOrder = {0, 1}, .reportStaleAliasPlacements = true}));
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
					REQUIRE(WaitHostIdle(host, device.Get()) == rhi::Result::Ok && Matches(first, value), "built: the second epoch copies what the first wrote");
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

				// 6. BuildsOf from other threads: each reader sees one pair the owner published, while the owner stages and attaches
				// both epochs again and again, and frames run between.
				{
					// The owner-thread checks follow SetOwnerThread. IsOwnerThread is true only on the thread that owns the host.
					host.SetOwnerThread();
					bool otherOwns = true;
					std::thread([&] { otherOwns = host.IsOwnerThread(); }).join();
					REQUIRE(host.IsOwnerThread() && !otherOwns, "only the owner thread owns the host");
					std::thread([&] { host.SetOwnerThread(); }).join();
					REQUIRE(!host.IsOwnerThread(), "SetOwnerThread on another thread moves the host there");
					host.SetOwnerThread();
					REQUIRE(host.IsOwnerThread(), "and back");

					std::atomic<bool> stop{false};
					std::atomic<uint64_t> reads{0}, stagedSeen{0};
					std::mutex failureMutex;
					std::string failure;
					// Every pair a reader can see in this test: an attached build, and no staged build or a newer one. A build seen
					// staged stays staged until a read finds it attached (the test never drops one). A torn read breaks one of these:
					// half of an attach reads as the old attached build with no staged one, or as the new build both staged and attached.
					const auto reader = [&](uint32_t epoch) {
						org::GraphHost::Builds last{};
						uint64_t staged = 0;
						while (!stop.load(std::memory_order_acquire)) {
							const auto builds = host.BuildsOf(epoch);
							const char* why = nullptr;
							if (builds.attached == 0) why = "no attached build";
							else if (builds.staged != 0 && builds.staged <= builds.attached) why = "a staged build not newer than the attached one";
							else if (builds.attached < last.attached) why = "the attached build went back";
							else if (staged != 0 && builds.attached < staged && builds.staged != staged) why = "a staged build was lost before its attach";
							if (why) {
								std::scoped_lock lock(failureMutex);
								if (failure.empty())
									failure = std::string(why) + " (epoch " + std::to_string(epoch) + ": attached " + std::to_string(builds.attached)
										+ ", staged " + std::to_string(builds.staged) + "; before: attached " + std::to_string(last.attached) + ", staged "
										+ std::to_string(staged) + ")";
								return;
							}
							if (builds.staged != 0) {
								if (builds.staged != staged) stagedSeen.fetch_add(1, std::memory_order_relaxed);
								staged = builds.staged;
							} else if (builds.attached >= staged) {
								staged = 0;
							}
							last = builds;
							reads.fetch_add(1, std::memory_order_relaxed);
							std::this_thread::yield();
						}
					};
					std::vector<std::thread> readers;
					for (const uint32_t epoch : {0u, 0u, 1u, 1u}) readers.emplace_back(reader, epoch);
					const auto cycles = [&]() -> int {
						const uint32_t both[] = {0, 1};
						for (int cycle = 0; cycle < 8; ++cycle) {
							for (const uint32_t epoch : both) {
								auto next = host.BuildEpochGraph(epoch, "test: BuildsOf readers");
								REQUIRE(next && next->Built(), "a build for the readers");
								const auto attachedBefore = host.BuildGeneration(epoch);
								const auto staging = host.StageEpochGraph(next);
								REQUIRE(staging.staged, "staged for the readers");
								const org::GraphHost::Builds expected{attachedBefore, staging.generation};
								REQUIRE(host.BuildsOf(epoch) == expected, "the owner reads its staging at once");
							}
							if (const int failed = run()) return failed;
							REQUIRE(host.AttachStagedEpochGraphs(both).attached, "the staged set attaches");
							for (const uint32_t epoch : both) {
								const org::GraphHost::Builds expected{host.BuildGeneration(epoch), 0};
								REQUIRE(host.BuildsOf(epoch) == expected && host.StagedGeneration(epoch) == 0, "the owner reads the attach at once");
							}
							if (const int failed = run()) return failed;
						}
						return 0;
					};
					const int failed = cycles();
					// The readers see the last attach before they stop.
					std::this_thread::sleep_for(std::chrono::milliseconds(20));
					stop.store(true, std::memory_order_release);
					for (auto& thread : readers) thread.join();
					if (failed) return failed;
					std::printf("built: BuildsOf: %llu reads on 4 threads, %llu staged builds seen\n", static_cast<unsigned long long>(reads.load()),
						static_cast<unsigned long long>(stagedSeen.load()));
					REQUIRE(failure.empty(), failure.c_str());
					REQUIRE(reads.load() > 0, "the readers read");
					REQUIRE(org::GraphHost::Builds{} == host.BuildsOf(7), "an epoch the host does not have reads {0, 0}");
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
		} else if (placement) {
			REQUIRE(TestFramePlacementCompute() == 0, "frame placement without a device");
			REQUIRE(TestFramePlacementStaging() == 0, "frame placement staging without a device");
			// Aggressive auto aliasing: the scratch buffers are candidates whatever their score.
			auto aliasing = org::runtime::GetOpenRenderGraphSettings();
			aliasing.autoAliasMode = 3;
			org::runtime::SetOpenRenderGraphSettings(aliasing);
			auto runtime = org::Runtime::Create({.device = device.Get(), .backend = rhi::Backend::Vulkan, .slots = 4,
				.tasks = std::make_shared<org::test::TestTaskService>(2)});
			{
				org::GraphHost host(WithSyncSubmit({.runtime = runtime, .queueBoundary = {.entry = true, .exit = true}, .framesInFlight = 2,
					.epochOrder = {0}, .reportStaleAliasPlacements = true}));
				host.SetFramePositions({{0, 0, false}});
				org::GraphHost::FramePlacementSettings settings;
				settings.enabled = true;
				settings.settleMilliseconds = 20;
				settings.heapTrimMilliseconds = 100;
				host.SetFramePlacement(settings);
				auto program = CreateProgram(device.Get());
				REQUIRE(program, "compute program");
				auto input = org::Buffer::CreateSharedUnmaterialized(rhi::HeapType::DeviceLocal, 16, false);
				org::BufferBase::DescriptorRequirements inputViews{};
				inputViews.createSRV = true;
				inputViews.srvDesc = {.dimension = rhi::SrvDim::Buffer, .formatOverride = rhi::Format::R32_Typeless,
					.buffer = {.kind = rhi::BufferViewKind::Raw, .firstElement = 0, .numElements = 4}};
				input->SetDescriptorRequirements(inputViews);
				input->SetName("test.placement.input");
				const std::array<std::shared_ptr<org::ExternalBufferResource>, 2> outputs{
					ImportHostBuffer(device.Get(), first, "test.placement.output.0"), ImportHostBuffer(device.Get(), second, "test.placement.output.1")};
				REQUIRE(outputs[0] && outputs[1], "import host buffers");
				std::vector<PlacementExtension::Pair> pairs;
				// Two scratch buffers of one name and size: new objects at each call.
				const auto makePairs = [&](uint64_t bytes) {
					pairs.clear();
					for (const auto& output : outputs) {
						auto scratch = org::Buffer::CreateSharedUnmaterialized(rhi::HeapType::DeviceLocal, bytes, true);
						org::BufferBase::DescriptorRequirements requirements{};
						requirements.createUAV = true;
						requirements.uavDesc = {.dimension = rhi::UavDim::Buffer, .formatOverride = rhi::Format::R32_Typeless,
							.buffer = {.kind = rhi::BufferViewKind::Raw, .firstElement = 0, .numElements = kCount}};
						scratch->SetDescriptorRequirements(requirements);
						scratch->SetName("test.placement.scratch");
						scratch->SetAllowAlias(true);
						pairs.push_back({std::move(scratch), output});
					}
				};
				constexpr uint64_t kMB = 1024ull * 1024ull;
				makePairs(8 * kMB);
				auto outputRevision = std::make_shared<std::atomic<uint64_t>>(1);
				host.AddExtension("test.placement", [&] { return std::make_unique<PlacementExtension>(input, pairs, program, outputRevision); });
				uint32_t value = 0, frame = 0;
				const auto upload = [&](org::RenderGraph&) {
					BUFFER_UPLOAD(&value, sizeof(value), org::runtime::UploadTarget::FromShared(input), 0);
				};
				const auto run = [&]() -> int {
					value = 50000u + 10u * frame++;
					host.ExecuteFrame(nullptr, upload, 0);
					host.SealSubmitted();
					REQUIRE(WaitHostIdle(host, device.Get()) == rhi::Result::Ok && Matches(first, value) && Matches(second, value),
						"placement: each copy reads what its write wrote");
					return 0;
				};
				const auto use = [&] { return host.FramePlacementInUse(0); };
				// The current heap for buffers, and its capacity.
				const auto bufferHeap = [&]() -> uint64_t {
					for (const auto& heap : host.FramePlacementHeaps())
						if (heap.current && heap.memoryClass == 1) return heap.capacity;
					return 0;
				};

				// 1. The first plan places both from the first version, published at once, and its heap is the need.
				if (const int failed = run()) return failed;
				const auto firstVersion = host.LatestFramePlacement();
				REQUIRE(firstVersion && use().version == firstVersion->Id(), "placement: the first plan places from the first version");
				REQUIRE(use().placed == 2, "placement: both scratch buffers of one name and description are placed from it");
				size_t placedEntries = 0;
				for (const auto& entry : firstVersion->Entries()) placedEntries += entry.placed && entry.name == "test.placement.scratch";
				REQUIRE(placedEntries == 2, "placement: the version has an entry for each");
				REQUIRE(bufferHeap() == firstVersion->BytesOfClass(1), "placement: a first heap is the need");

				// 2. A rebuild with new resources of the same names and sizes: their keys find their predecessors' entries.
				const auto builds = host.BuildGeneration(0);
				makePairs(8 * kMB);
				host.RequestRebuild(0);
				if (const int failed = run()) return failed;
				REQUIRE(host.BuildGeneration(0) == builds + 1, "placement: rebuilt");
				REQUIRE(use().version == firstVersion->Id() && use().placed == 2, "placement: a rebuilt graph's new resources take their entries");

				// 2b. Smaller resources: new keys the version does not know, but they fit in the free bytes of its block, so the graph
				// places them there instead of in pools of its own.
				makePairs(4 * kMB);
				host.RequestRebuild(0);
				if (const int failed = run()) return failed;
				REQUIRE(use().version == firstVersion->Id() && use().placed == 2, "placement: new resources that fit go in the version's block");
				REQUIRE(bufferHeap() == firstVersion->BytesOfClass(1), "placement: the heap is unchanged");

				// 3. Larger resources that do not fit: new keys the version does not know. The plan's summary reports them uncovered,
				// so the host computes the next version at once and the plan places them from it. The heap grows to the need (more
				// than headroom times its old size).
				makePairs(24 * kMB);
				host.RequestRebuild(0);
				if (const int failed = run()) return failed;
				const auto grown = host.LatestFramePlacement();
				REQUIRE(grown && grown->Id() != firstVersion->Id() && use().version == grown->Id() && use().placed == 2,
					"placement: resources that do not fit are placed from a version made for them at once");
				REQUIRE(bufferHeap() == grown->BytesOfClass(1), "placement: a heap that grows past its headroom takes the need");

				// 4. A little larger again: the heap grows with headroom (the larger of the need and 1.25 times its size), and once
				// quiet the layout is published again on a heap of its need. Both are taken in place.
				makePairs(25 * kMB);
				host.RequestRebuild(0);
				if (const int failed = run()) return failed;
				const auto rebuilt = host.BuildGeneration(0);
				bool headroom = false, trimmed = false;
				for (int wait = 0; wait < 400 && !trimmed; ++wait) {
					if (const int failed = run()) return failed;
					const auto latest = host.LatestFramePlacement();
					const auto heap = bufferHeap();
					if (latest && heap > latest->BytesOfClass(1) + org::FramePlacementMemory::TrimSlack(latest->BytesOfClass(1))) headroom = true;
					trimmed = latest && latest->ExactHeaps() && use().version == latest->Id() && use().placed == 2 && heap == latest->BytesOfClass(1);
					std::this_thread::sleep_for(std::chrono::milliseconds(5));
				}
				std::printf("placement: first version %llu, now %llu, buffer heap %llu bytes\n", static_cast<unsigned long long>(firstVersion->Id()),
					static_cast<unsigned long long>(use().version), static_cast<unsigned long long>(bufferHeap()));
				REQUIRE(headroom, "placement: a heap that grows takes headroom");
				REQUIRE(trimmed, "placement: once quiet, the heap is trimmed to the need and the graph places from that version");
				REQUIRE(host.BuildGeneration(0) == rebuilt, "placement: new versions are taken in place, without a rebuild");
				REQUIRE(host.StaleAliasPlacementEpochs().empty(), "placement: an older frame placement is not reported stale");
				REQUIRE(!counters.unbalanced && counters.locks.load() == counters.unlocks.load(), "balanced host queue locks");
				host.DestroyGraph();
				std::puts("placement: one name, two resources, the first version at once, rebuilds, in-place versions and a trim");
			}
			runtime.reset();
		} else if (stagedPlacement) {
			REQUIRE(TestFramePlacementStaging() == 0, "frame placement staging without a device");
			auto aliasing = org::runtime::GetOpenRenderGraphSettings();
			aliasing.autoAliasMode = 3;
			org::runtime::SetOpenRenderGraphSettings(aliasing);
			auto runtime = org::Runtime::Create({.device = device.Get(), .backend = rhi::Backend::Vulkan, .slots = 4,
				.tasks = std::make_shared<org::test::TestTaskService>(2)});
			REQUIRE(TestFramePlacementKeeping() == 0, "frame placement versions that keep resources");
			{
				org::GraphHost host({.runtime = runtime, .queueBoundary = {.entry = true, .exit = true}, .framesInFlight = 2, .epochOrder = {0, 1}});
				host.SetFramePositions({{0, 0, false}, {1, 1, false}});
				org::GraphHost::FramePlacementSettings settings;
				settings.enabled = true;
				settings.settleMilliseconds = 20;
				settings.heapTrimMilliseconds = 60;
				host.SetFramePlacement(settings);
				// "staged-placement kept": both epochs keep their recordings (GraphHost::SetKeptEpochs). Every version taken in place and
				// every re-placement installed makes the kept recordings stale (O-M6-3): each frame's outputs must still be right.
				const bool keptPlacement = argc > 2 && std::strcmp(argv[2], "kept") == 0;
				if (keptPlacement) host.SetKeptEpochs({0, 1});
				std::map<std::string, uint64_t> keptReasons;
				uint64_t keptReused = 0, keptRecorded = 0, keptUnkeepable = 0;
				const auto keptPhase = [&](const char* phase) {
					if (!keptPlacement) return;
					for (const uint32_t epoch : {0u, 1u}) {
						const auto stats = host.TakeKeptStats(epoch);
						std::string reasons;
						for (const auto& [reason, count] : stats.reasons) {
							reasons += (reasons.empty() ? "" : "; ") + reason + " x" + std::to_string(count);
							keptReasons[reason] += count;
						}
						keptReused += stats.reused;
						keptRecorded += stats.kept;
						keptUnkeepable += stats.unkeepable;
						std::printf("staged placement kept, %s, epoch %u: %llu runs, %llu reused, %llu kept, %llu not keepable (%s)\n", phase, epoch,
							static_cast<unsigned long long>(stats.runs), static_cast<unsigned long long>(stats.reused),
							static_cast<unsigned long long>(stats.kept), static_cast<unsigned long long>(stats.unkeepable),
							reasons.empty() ? "none" : reasons.c_str());
					}
				};
				auto program = CreateProgram(device.Get());
				REQUIRE(program, "compute program");
				auto input = org::Buffer::CreateSharedUnmaterialized(rhi::HeapType::DeviceLocal, 16, false);
				org::BufferBase::DescriptorRequirements inputViews{};
				inputViews.createSRV = true;
				inputViews.srvDesc = {.dimension = rhi::SrvDim::Buffer, .formatOverride = rhi::Format::R32_Typeless,
					.buffer = {.kind = rhi::BufferViewKind::Raw, .firstElement = 0, .numElements = 4}};
				input->SetDescriptorRequirements(inputViews);
				input->SetName("test.staged.input");
				const auto scratch = [&](uint64_t bytes, const char* name) {
					auto buffer = org::Buffer::CreateSharedUnmaterialized(rhi::HeapType::DeviceLocal, bytes, true);
					org::BufferBase::DescriptorRequirements requirements{};
					requirements.createUAV = true;
					requirements.uavDesc = {.dimension = rhi::UavDim::Buffer, .formatOverride = rhi::Format::R32_Typeless,
						.buffer = {.kind = rhi::BufferViewKind::Raw, .firstElement = 0, .numElements = kCount}};
					buffer->SetDescriptorRequirements(requirements);
					buffer->SetName(name);
					buffer->SetAllowAlias(true);
					return buffer;
				};
				constexpr uint64_t kMB = 1024ull * 1024ull;
				auto shared = scratch(8 * kMB, "test.staged.shared");
				auto own = scratch(4 * kMB, "test.staged.own");
				auto sharedOutput = ImportHostBuffer(device.Get(), first, "test.staged.shared-output");
				auto ownOutput = ImportHostBuffer(device.Get(), second, "test.staged.own-output");
				REQUIRE(sharedOutput && ownOutput, "import host buffers");
				auto outputRevision = std::make_shared<std::atomic<uint64_t>>(1);
				host.AddExtension("test.staged", [&] {
					return std::make_unique<StagedPlacementExtension>(input, shared, sharedOutput, own, ownOutput, program, outputRevision);
				});
				uint32_t value = 0, frame = 0;
				const auto upload = [&](org::RenderGraph&) {
					BUFFER_UPLOAD(&value, sizeof(value), org::runtime::UploadTarget::FromShared(input), 0);
				};
				// One version a frame: every epoch's graph prepared from the version the first one recorded with.
				size_t mixedVersions = 0;
				const auto run = [&](const char* what) -> int {
					value = 60000u + 10u * frame++;
					host.ExecuteFrame(nullptr, upload, 0);
					host.ExecuteFrame(nullptr, {}, 1);
					if (host.FramePlacementInUse(0).version != host.FramePlacementInUse(1).version) ++mixedVersions;
					host.SealSubmitted();
					REQUIRE(device->WaitIdle() == rhi::Result::Ok && Matches(first, value) && Matches(second, value), what);
					return 0;
				};
				const auto generation = [](const std::shared_ptr<org::Buffer>& buffer) { return buffer->GetBackingGeneration(); };
				for (int i = 0; i < 60; ++i) {
					if (const int failed = run("staged placement: frames before any set")) return failed;
					std::this_thread::sleep_for(std::chrono::milliseconds(2));
				}
				const auto settled = host.FramePlacementInUse(0).version;
				REQUIRE(settled && host.FramePlacementInUse(1).version == settled, "staged placement: both epochs place from one version");
				keptPhase("before any set");
				const auto stage = [&](std::unique_ptr<org::GraphHost::EpochGraph>& graph) {
					const auto staged = host.StageEpochGraph(graph);
					return staged.staged && !staged.attached;
				};

				// 1. Epoch 1 rebuilt with a larger scratch buffer of its own: a new version is needed. Its transients are placed on this
				// thread, from a staging the attached graphs do not see: they keep running on their version and backings until the set
				// attaches. The version keeps the shared buffer where it is, so epoch 0 takes it without placing again.
				{
					own = scratch(24 * kMB, "test.staged.own");
					const auto sharedBacking = generation(shared);
					const auto graph0 = host.Graph(0);
					const auto builds0 = host.BuildGeneration(0);
					auto graph = host.BuildEpochGraph(1, "test: a staged set that needs a new version");
					REQUIRE(graph && graph->Built(), "staged placement: built");
					org::GraphHost::EpochGraph* const graphs[] = {graph.get()};
					REQUIRE(host.PlaceEpochGraphs(graphs) == 1 && graph->Built(), "staged placement: placed at the build");
					REQUIRE(own->IsMaterialized() && generation(shared) == sharedBacking, "staged placement: the new transient is placed, the shared one stays");
					for (int i = 0; i < 3; ++i)
						if (const int failed = run("staged placement: the attached graphs keep running while a set is placed")) return failed;
					REQUIRE(host.FramePlacementInUse(0).version == settled && host.FramePlacementInUse(1).version == settled,
						"staged placement: nothing is published before the attach");
					REQUIRE(stage(graph), "staged placement: staged");
					const uint32_t one[] = {1};
					REQUIRE(host.AttachStagedEpochGraphs(one).attached, "staged placement: the set attaches");
					for (int i = 0; i < 3; ++i)
						if (const int failed = run("staged placement: after the attach")) return failed;
					const auto next = host.FramePlacementInUse(1).version;
					REQUIRE(next != settled && host.FramePlacementInUse(0).version == next, "staged placement: the set's version is every epoch's");
					REQUIRE(host.Graph(0) == graph0 && host.BuildGeneration(0) == builds0 && generation(shared) == sharedBacking,
						"staged placement: epoch 0 keeps its graph, and the shared buffer its backing");
					keptPhase("1. a set of epoch 1");
					if (keptPlacement) REQUIRE(keptReasons["frame placement"] > 0, "kept: the version epoch 0 takes in place makes its kept recordings stale");
				}
				// 2a. A failed build dropped after its placement: epoch 1 built again with the same transients (the same objects the
				// attached graph binds), placed at its build, then dropped without a staging (a build whose pipelines failed). It found
				// its own buffer where the attached graph has it and claimed it; its end must not give that backing back, since the
				// attached graph keeps running on it.
				{
					const auto attachedOwnBacking = generation(own);
					auto graph = host.BuildEpochGraph(1, "test: a failed build dropped after its placement");
					org::GraphHost::EpochGraph* const graphs[] = {graph.get()};
					REQUIRE(graph && graph->Built() && host.PlaceEpochGraphs(graphs) == 1, "failed build: placed at the build");
					REQUIRE(own->IsMaterialized() && generation(own) == attachedOwnBacking, "failed build: it found its transient in place");
					graph.reset();
					REQUIRE(own->IsMaterialized() && generation(own) == attachedOwnBacking,
						"failed build: dropping it leaves the attached graph's backing in place");
					for (int i = 0; i < 3; ++i)
						if (const int failed = run("failed build: the attached graphs keep running after the drop")) return failed;
				}
				// 2. A dropped set: placed at its build, staged, then dropped. Its staging publishes nothing.
				{
					const auto before = host.FramePlacementInUse(0).version;
					own = scratch(48 * kMB, "test.staged.own");
					auto graph = host.BuildEpochGraph(1, "test: a dropped set");
					org::GraphHost::EpochGraph* const graphs[] = {graph.get()};
					REQUIRE(graph && graph->Built() && host.PlaceEpochGraphs(graphs) == 1 && stage(graph), "dropped set: placed and staged");
					const uint32_t one[] = {1};
					REQUIRE(host.DropStagedEpochGraphs(one) == 1, "dropped set: dropped");
					own = scratch(24 * kMB, "test.staged.own");
					for (int i = 0; i < 3; ++i)
						if (const int failed = run("dropped set: frames after")) return failed;
					REQUIRE(host.FramePlacementInUse(0).version == before && host.FramePlacementInUse(1).version == before,
						"dropped set: its version was never published");
				}
				// 3. Versions the worker publishes while frames run (the settle and the trim of the heaps): each frame's epochs place
				// from one version.
				// (Earlier frames may differ by design: a later epoch's plan that needs a version at once for its uncovered candidates
				// makes it the frame's version, the first frame and an attach of epoch 1 alone among them.)
				mixedVersions = 0;
				for (int i = 0; i < 80; ++i) {
					if (const int failed = run("staged placement: frames while versions are published")) return failed;
					std::this_thread::sleep_for(std::chrono::milliseconds(3));
				}
				REQUIRE(mixedVersions == 0, "staged placement: no frame's epochs placed from different versions");
				keptPhase("2-3. a dropped set, versions while frames run");
				// 4. A set of both epochs with every transient new (a resize): epoch 0's new shared buffer is read by epoch 1's new
				// graph, which also has a new buffer of its own. Placed at the build (PlaceEpochGraphs), first with each summary
				// arriving with its graph's plan, then with the summaries first. Both ways the set's graphs end on one version at
				// the build: the second pass makes epoch 0's graph take the newest in place, or the summaries first make one
				// version for both. The outputs are right after the attach, and the attached graphs placed from that version.
				for (const bool summariesFirst : {false, true}) {
					auto changed = settings;
					changed.summariesFirst = summariesFirst;
					host.SetFramePlacement(changed);
					shared = scratch(summariesFirst ? 10 * kMB : 12 * kMB, "test.staged.shared");
					own = scratch(summariesFirst ? 14 * kMB : 16 * kMB, "test.staged.own");
					auto graph0 = host.BuildEpochGraph(0, "test: a set of every epoch with new transients");
					auto graph1 = host.BuildEpochGraph(1, "test: a set of every epoch with new transients");
					REQUIRE(graph0 && graph0->Built() && graph1 && graph1->Built(), "set of every epoch: built");
					org::GraphHost::EpochGraph* const graphs[] = {graph0.get(), graph1.get()};
					REQUIRE(host.PlaceEpochGraphs(graphs) == 2, "set of every epoch: both placed at the build");
					const auto version0 = graph0->PlacedFromVersion();
					const auto version1 = graph1->PlacedFromVersion();
					REQUIRE(version0 && version1, "set of every epoch: both placed from a frame placement");
					std::printf("staged placement, a set of every epoch (summaries first %d): epoch 0 placed from version %llu, epoch 1 from %llu\n",
						summariesFirst ? 1 : 0, static_cast<unsigned long long>(version0), static_cast<unsigned long long>(version1));
					REQUIRE(version0 == version1, "set of every epoch: the set's graphs placed from one version at the build");
					REQUIRE(shared->IsMaterialized() && own->IsMaterialized(), "set of every epoch: the new transients are placed");
					REQUIRE(stage(graph0) && stage(graph1), "set of every epoch: staged");
					const uint32_t both[] = {0, 1};
					REQUIRE(host.AttachStagedEpochGraphs(both).attached, "set of every epoch: the set attaches");
					if (const int failed = run("set of every epoch: the first frame after the attach")) return failed;
					REQUIRE(host.FramePlacementInUse(0).version == version0 && host.FramePlacementInUse(1).version == version0,
						"set of every epoch: the first frame runs the version the set placed from");
					for (int i = 0; i < 3; ++i)
						if (const int failed = run("set of every epoch: after the attach")) return failed;
				}
				keptPhase("4. sets of every epoch");
				// 5. The trim prepared on a worker (FramePlacementSettings::prepareReplacements). A set of epoch 1 with a larger transient
				// of its own makes a version that keeps the running graphs' resources (a heap lane). Once quiet, the trim packs the frame on
				// heaps of its need: the host offers it, each graph prepares its re-placement on a worker (backings of what moves made ahead,
				// the validated publication built), and the version is published at a frame boundary, where each graph installs what it
				// prepared. Every frame's outputs stay right. With the preparations off, the graphs re-place at their preparation instead.
				const auto installedCount = [&] {
					uint64_t installed = 0;
					for (const uint32_t epoch : {0u, 1u})
						if (auto* graph = host.Graph(epoch)) installed += graph->PreparedReplacementCounts().first;
					return installed;
				};
				for (const bool prepared : {true, false}) {
					auto changed = settings;
					changed.prepareReplacements = prepared;
					host.SetFramePlacement(changed);
					for (int i = 0; i < 40; ++i) {
						if (const int failed = run("trim: frames before the set")) return failed;
						std::this_thread::sleep_for(std::chrono::milliseconds(3));
					}
					own = scratch(prepared ? 40 * kMB : 72 * kMB, "test.staged.own");
					auto graph = host.BuildEpochGraph(1, "test: a set whose version keeps the running graphs' resources");
					org::GraphHost::EpochGraph* const graphs[] = {graph.get()};
					REQUIRE(graph && graph->Built() && host.PlaceEpochGraphs(graphs) == 1 && stage(graph), "trim: the set is placed and staged");
					const uint32_t one[] = {1};
					REQUIRE(host.AttachStagedEpochGraphs(one).attached, "trim: the set attaches");
					for (int i = 0; i < 3; ++i)
						if (const int failed = run("trim: after the attach")) return failed;
					const auto attached = host.FramePlacementInUse(0).version;
					const auto installedBefore = installedCount();
					mixedVersions = 0;
					// Until both graphs installed one (prepared), or a while after the first newer version (not prepared).
					bool trimmed = false;
					int after = 0;
					for (int i = 0; i < 600; ++i) {
						if (const int failed = run("trim: frames until the trim")) return failed;
						std::this_thread::sleep_for(std::chrono::milliseconds(3));
						trimmed = trimmed || host.FramePlacementInUse(0).version != attached;
						if (prepared ? installedCount() - installedBefore >= 2 : trimmed && ++after > 150) break;
					}
					REQUIRE(trimmed, "trim: a newer version is published once quiet");
					for (int i = 0; i < 5; ++i)
						if (const int failed = run("trim: frames after the trim")) return failed;
					const auto installed = installedCount() - installedBefore;
					std::printf("staged placement, the trim (prepared %d): version %llu -> %llu, %llu re-placements installed from a worker\n", prepared ? 1 : 0,
						static_cast<unsigned long long>(attached), static_cast<unsigned long long>(host.FramePlacementInUse(0).version),
						static_cast<unsigned long long>(installed));
					if (prepared) REQUIRE(installed == 2, "trim: both graphs installed the re-placement prepared on a worker");
					else REQUIRE(installed == 0, "trim: with the preparations off, the graphs re-place at their preparation");
					REQUIRE(host.FramePlacementInUse(0).version == host.FramePlacementInUse(1).version, "trim: both epochs place from the trim");
					REQUIRE(mixedVersions == 0, "trim: no frame's epochs placed from different versions");
					keptPhase(prepared ? "5. the trim, prepared" : "5. the trim, at preparation");
				}
				if (keptPlacement) {
					std::printf("staged placement kept: %llu reused, %llu recorded and kept, %llu not keepable, %llu for a frame placement\n",
						static_cast<unsigned long long>(keptReused), static_cast<unsigned long long>(keptRecorded),
						static_cast<unsigned long long>(keptUnkeepable), static_cast<unsigned long long>(keptReasons["frame placement"]));
					REQUIRE(keptReused > 0 && keptUnkeepable == 0, "kept: steady frames reuse their kept recordings");
				}
				host.SetFramePlacement(settings);
				REQUIRE(!counters.unbalanced && counters.locks.load() == counters.unlocks.load(), "balanced host queue locks");
				host.DestroyGraph();
				std::puts("staged placement: a set placed at its build, a kept version, a dropped set, one version a frame");
			}
			runtime.reset();
		} else if (sharedRebuild) {
			auto aliasing = org::runtime::GetOpenRenderGraphSettings();
			aliasing.autoAliasMode = 3;
			org::runtime::SetOpenRenderGraphSettings(aliasing);
			auto runtime = org::Runtime::Create({.device = device.Get(), .backend = rhi::Backend::Vulkan, .slots = 4,
				.tasks = std::make_shared<org::test::TestTaskService>(2)});
			// "shared-rebuild on" or "off": with the frame placement on or off only.
			const bool onlyOn = argc > 2 && std::strcmp(argv[2], "on") == 0, onlyOff = argc > 2 && std::strcmp(argv[2], "off") == 0;
			for (const bool framePlacement : {false, true}) {
				if ((onlyOn && !framePlacement) || (onlyOff && framePlacement)) continue;
				const char* const mode = framePlacement ? "shared rebuild (frame placement)" : "shared rebuild (no frame placement)";
				org::GraphHost host(WithSyncSubmit({.runtime = runtime, .queueBoundary = {.entry = true, .exit = true}, .framesInFlight = 2,
					.epochOrder = {0, 1}, .sharedAliasArena = true}));
				host.SetFramePositions({{0, 0, false}, {1, 1, false}});
				org::GraphHost::FramePlacementSettings settings;
				settings.enabled = framePlacement;
				settings.settleMilliseconds = 20;
				settings.heapTrimMilliseconds = 60;
				host.SetFramePlacement(settings);
				auto program = CreateProgram(device.Get());
				REQUIRE(program, "compute program");
				org::BufferBase::DescriptorRequirements inputViews{};
				inputViews.createSRV = true;
				inputViews.srvDesc = {.dimension = rhi::SrvDim::Buffer, .formatOverride = rhi::Format::R32_Typeless,
					.buffer = {.kind = rhi::BufferViewKind::Raw, .firstElement = 0, .numElements = 4}};
				auto input = org::Buffer::CreateSharedUnmaterialized(rhi::HeapType::DeviceLocal, 16, false);
				input->SetDescriptorRequirements(inputViews);
				input->SetName("test.staged.input");
				// The second epoch's own transient is written with another value than the shared one, so a shared transient in bytes
				// the second epoch's own pool gives to its own reads back wrong.
				auto ownInput = org::Buffer::CreateSharedUnmaterialized(rhi::HeapType::DeviceLocal, 16, false);
				ownInput->SetDescriptorRequirements(inputViews);
				ownInput->SetName("test.staged.own-input");
				const auto scratch = [&](uint64_t bytes, const char* name) {
					auto buffer = org::Buffer::CreateSharedUnmaterialized(rhi::HeapType::DeviceLocal, bytes, true);
					org::BufferBase::DescriptorRequirements requirements{};
					requirements.createUAV = true;
					requirements.uavDesc = {.dimension = rhi::UavDim::Buffer, .formatOverride = rhi::Format::R32_Typeless,
						.buffer = {.kind = rhi::BufferViewKind::Raw, .firstElement = 0, .numElements = kCount}};
					buffer->SetDescriptorRequirements(requirements);
					buffer->SetName(name);
					buffer->SetAllowAlias(true);
					return buffer;
				};
				constexpr uint64_t kMB = 1024ull * 1024ull;
				auto shared = scratch(4 * kMB, "test.shared-rebuild.shared");
				auto own = scratch(4 * kMB, "test.shared-rebuild.own");
				// Kept across the sets, so a set's version that packs the frame anew would move it under the running graphs.
				auto kept = scratch(10 * kMB, "test.shared-rebuild.kept");
				auto sharedOutput = ImportHostBuffer(device.Get(), first, "test.shared-rebuild.shared-output");
				auto ownOutput = ImportHostBuffer(device.Get(), second, "test.shared-rebuild.own-output");
				REQUIRE(sharedOutput && ownOutput, "import host buffers");
				auto outputRevision = std::make_shared<std::atomic<uint64_t>>(1);
				// The factory reads the current objects: a build after they were made again binds the new ones.
				host.AddExtension("test.shared-rebuild", [&] {
					auto extension = std::make_unique<StagedPlacementExtension>(input, shared, sharedOutput, own, ownOutput, program, outputRevision);
					extension->SetOwnInput(ownInput);
					extension->SetKept(kept);
					return extension;
				});
				uint32_t value = 0, frame = 0;
				const auto upload = [&](org::RenderGraph&) {
					BUFFER_UPLOAD(&value, sizeof(value), org::runtime::UploadTarget::FromShared(input), 0);
					const uint32_t ownValue = value + 5000u;
					BUFFER_UPLOAD(&ownValue, sizeof(ownValue), org::runtime::UploadTarget::FromShared(ownInput), 0);
				};
				size_t mixedVersions = 0;
				// The first frame's graphs are built one at a time, as each epoch first runs: the first epoch's graph places before the
				// second's exists, so it cannot know they share a transient. Its first frame may copy the wrong contents (the second
				// graph gives the shared transient a backing of its own); from the next frame on both bind that one.
				const auto run = [&](const char* what) -> int {
					const bool firstFrame = frame == 0;
					value = 90000u + 10u * frame++;
					host.ExecuteFrame(nullptr, upload, 0);
					host.ExecuteFrame(nullptr, {}, 1);
					if (host.FramePlacementInUse(0).version != host.FramePlacementInUse(1).version) ++mixedVersions;
					host.SealSubmitted();
					REQUIRE(WaitHostIdle(host, device.Get()) == rhi::Result::Ok, what);
					if (!firstFrame && (!Matches(first, value) || !Matches(second, value + 5000u))) {
						std::printf("FAIL: %s: %s (frame %u: shared %u, own %u, expected %u and %u)\n", mode, what, frame - 1, first.mapped[0],
							second.mapped[0], value, value + 5000u);
						return 1;
					}
					return 0;
				};
				for (int i = 0; i < 30; ++i) {
					if (const int failed = run("frames before any set")) return failed;
					std::this_thread::sleep_for(std::chrono::milliseconds(2));
				}
				// Sets of both epochs for new objects of other sizes, as a resize makes them. The running graphs keep the old ones.
				const uint64_t sizes[][2] = {{12, 6}, {2, 3}, {8, 8}, {6, 16}};
				for (const auto& size : sizes) {
					shared = scratch(size[0] * kMB, "test.shared-rebuild.shared");
					own = scratch(size[1] * kMB, "test.shared-rebuild.own");
					auto graph0 = host.BuildEpochGraph(0, "test: a set for new objects");
					auto graph1 = host.BuildEpochGraph(1, "test: a set for new objects");
					REQUIRE(graph0 && graph0->Built() && graph1 && graph1->Built(), "shared rebuild: built");
					org::GraphHost::EpochGraph* const graphs[] = {graph0.get(), graph1.get()};
					// With the frame placement, a graph may still leave its placement to the attach (a version that would move what an
					// earlier graph of the set placed). Either way the frames after the attach must be right.
					const auto placedAtBuild = host.PlaceEpochGraphs(graphs);
					REQUIRE(!framePlacement || placedAtBuild >= 1, "shared rebuild: placed at the build");
					if (const int failed = run("the attached graphs keep running while a set is built")) return failed;
					// "sync-submit": a frame of both epochs is still queued on the submission thread while the set is staged and attached.
					// The old graphs' executions still record and submit after the attach, and write that frame's values.
					std::shared_ptr<std::atomic<bool>> gate;
					if (g_syncSubmit) {
						gate = HoldSubmissionThread(host);
						value = 90000u + 10u * frame++;
						host.ExecuteFrame(nullptr, upload, 0);
						host.ExecuteFrame(nullptr, {}, 1);
						host.SealSubmitted();
						REQUIRE(!host.SubmittedPoint().Passed(), "shared rebuild: a frame is queued while the set attaches");
					}
					REQUIRE(host.StageEpochGraph(graph0).staged && host.StageEpochGraph(graph1).staged, "shared rebuild: staged");
					const uint32_t both[] = {0, 1};
					REQUIRE(host.AttachStagedEpochGraphs(both).attached, "shared rebuild: the set attaches");
					if (gate) {
						OpenGate(gate);
						REQUIRE(WaitHostIdle(host, device.Get()) == rhi::Result::Ok && Matches(first, value) && Matches(second, value + 5000u),
							"shared rebuild: the frame queued across the attach writes its values");
					}
					mixedVersions = 0;
					for (int i = 0; i < 20; ++i) {
						if (const int failed = run("frames after a set")) return failed;
						std::this_thread::sleep_for(std::chrono::milliseconds(2));
					}
					REQUIRE(shared->IsMaterialized() && own->IsMaterialized(), "shared rebuild: the new objects have backings");
					REQUIRE(mixedVersions == 0, "shared rebuild: no frame's epochs placed from different versions");
				}
				if (framePlacement)
					REQUIRE(host.FramePlacementInUse(0).version && host.FramePlacementInUse(0).version == host.FramePlacementInUse(1).version,
						"shared rebuild: both epochs place from one version");
				REQUIRE(!counters.unbalanced && counters.locks.load() == counters.unlocks.load(), "balanced host queue locks");
				host.DestroyGraph();
				std::printf("%s: four sets for new objects, each frame's copy right\n", mode);
			}
			runtime.reset();
		} else if (arena) {
			auto runtime = org::Runtime::Create({.device = device.Get(), .backend = rhi::Backend::Vulkan, .slots = 4,
				.tasks = std::make_shared<org::test::TestTaskService>(2)});
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
				.tasks = std::make_shared<org::test::TestTaskService>(2)});
			// Without queue-scoped uploads the write waits for the uploads, which run in the host's stream: a segment of their own,
			// and the copy's segment split from it. With them the upload goes to the compute queue, ahead of the write.
			for (const bool scopedUploads : {false, true}) {
				org::GraphHost host({.runtime = runtime, .queueBoundary = {.entry = true, .exit = true}, .epochOrder = {0},
					.externalHomeStates = home, .queueScopedUploads = scopedUploads});
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
				.tasks = std::make_shared<org::test::TestTaskService>(2)});
			{
				org::GraphHost host({.runtime = runtime, .queueBoundary = {.entry = true, .exit = true}, .epochOrder = {0},
					.externalHomeStates = true, .reuseRecordings = true});
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
					REQUIRE(host.AttachStagedEpochGraphs(std::array<uint32_t, 1>{0}).attached && host.EpochRecordingCurrent(*stagedRecording), "the staged build switched in");
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
				.tasks = std::make_shared<org::test::TestTaskService>(2)});
			{
				org::GraphHost host({.runtime = runtime, .queueBoundary = {.entry = true, .exit = true}, .framesInFlight = 2, .epochOrder = {0},
					.reuseRecordings = wide});
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
				// An oversize region has a chunk of its own at offset 0, and its chunk comes back once its last holder lets it go.
				{
					const auto& arena = runtime->Arena();
					const auto before = arena->TakeStats().oversizeChunks;
					auto large = arena->AcquireOversize(org::FrameArena::kChunkBytes + 1);
					REQUIRE(large && large->offset == 0 && large->size >= org::FrameArena::kChunkBytes + 1, "an oversize region starts its own chunk");
					auto tiny = arena->AcquireOversize(16);
					REQUIRE(tiny && tiny->offset == 0 && tiny->buffer != large->buffer, "each oversize region has its own chunk");
					const auto* largeBuffer = large->buffer;
					large.reset();
					arena->Trim();
					auto again = arena->AcquireOversize(org::FrameArena::kChunkBytes + 1);
					REQUIRE(again && again->buffer == largeBuffer, "a freed oversize chunk is reused by the next region that fits");
					REQUIRE(arena->TakeStats().oversizeChunks == before + 2, "no chunk is made for a reused region");
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
						// With replayable recordings a live run may hold a kept copy, which names its copy like a published one.
						REQUIRE(frameRuns[a].Leased() && (wide || !frameRuns[a].Published()), "a live run is leased");
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
			.tasks = std::make_shared<org::test::TestTaskService>(2)});
		org::GraphHost host(WithSyncSubmit({.runtime = runtime,
			.queueBoundary = {.entry = true, .exit = true}, .epochOrder = {0}, .reuseRecordings = reuse || revision}));
		const auto waitIdle = [&] { return WaitHostIdle(host, device.Get()); };
		if (inlineLate) host.SetInlineAtExecution(true);
		if (syncReuse) host.SetKeptEpochs({0});
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
			if (revision || syncReuse) {
				scratch->Materialize();
				versions = org::VersionedBuffer::Create(scratch);
			}
			bool uncheckedCopy = false;
			host.AddExtension("test.host", [&] {
				auto extension = std::make_unique<HostExtension>(input, scratch, output, program, outputRevision, versions, external || inlineLate);
				extension->SetUncheckedCopy(uncheckedCopy);
				return extension;
			});
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
					REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value), "a revision's recording writes every element");
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
					REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value),
						frame < 2 ? "the old version's revision after the grown one was recorded" : "the grown version's revision");
				}
				// A new revision: half the elements. Until its recording is ready the frames keep submitting the old one.
				const uint32_t half = kCount / 2;
				auto halfRecording = record(half, grown);
				REQUIRE(halfRecording, "the next revision's recording");
				const uint32_t before = value;
				value = 50000u;
				host.SubmitEpoch(0, upload, {}, halfRecording);
				REQUIRE(waitIdle() == rhi::Result::Ok, "wait for the half revision");
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
					REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value), "a revision's recording copies the version it names");
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
				REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value), "the host goes on after a rejected submission");
				const auto stats = host.TakeAsyncStats();
				std::printf("revision: %llu recordings (%llu copies), %llu submissions, %llu recorded at tickets, %llu prepared again\n",
					static_cast<unsigned long long>(stats.revisionRecordings), static_cast<unsigned long long>(stats.revisionCopiesRecorded),
					static_cast<unsigned long long>(stats.revisionSubmissions), static_cast<unsigned long long>(stats.recorded),
					static_cast<unsigned long long>(stats.stale));
				REQUIRE(stats.revisionSubmissions == 13 && stats.stale == 0 && stats.revisionRecordings == 3 && stats.revisionCopiesRecorded == 9,
					"every submission bound a revision's recording; nothing prepared again");
				if (segment) {
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
						REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value), "a pinned copy's recording writes every element");
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
					REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value), "a re-pinned copy's recording writes every element");
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
					REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, kept), "a pass the host data turns off writes nothing");
					value = 73000u;
					host.SetEpochCopy(0, 0);
					host.SubmitEpoch(0, upload, {}, slotted);
					REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value), "the next revision writes again");
					// One copy only: recorded for the slot the next run takes, and submitted with it. Its other copies stay empty.
					auto single = record(kCount, current, false, 1);
					REQUIRE(single && single->copies[1] && !single->copies[0], "a recording of one copy has that copy alone");
					value = 74000u;
					host.SetEpochCopy(0, 1);
					host.SubmitEpoch(0, upload, {}, single);
					REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value), "a one-copy recording writes every element");
					// The next slot's run takes the full recording again: the one-copy recording is no candidate for it.
					value = 75000u;
					host.SetEpochCopy(0, 2);
					host.SubmitEpoch(0, upload, {}, slotted);
					REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value), "the other slots go on with their recordings");
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
						REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value), "a staged copy's recording writes every element");
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
							REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value), "a staged copy's recording writes every element");
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
						REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value), "the attached graph's recording submits while staged");
						REQUIRE(host.AttachStagedEpochGraphs(std::array<uint32_t, 1>{0}).attached, "the staged graph is attached");
						REQUIRE(host.BuildGeneration(0) == staging.generation && host.StagedGeneration(0) == 0, "the epoch runs the staged build");
						REQUIRE(host.EpochRecordingCurrent(*stagedRecording) && !host.EpochRecordingCurrent(*slotted), "its recordings are current, the old ones not");
						for (uint32_t frame = 0; frame < 3; ++frame) {
							value = 79000u + frame;
							host.SetEpochCopy(0, frame);
							host.SubmitEpoch(0, upload, {}, stagedRecording);
							REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value), "the staged graph's recording submits after the attach");
						}
						// A second staging replaces the first; the replaced one's recording request is dropped.
						auto one = host.BuildEpochGraph(0, "test: staged, then replaced");
						auto two = host.BuildEpochGraph(0, "test: staged last");
						const auto stagedOne = host.StageEpochGraph(one);
						const auto stagedTwo = host.StageEpochGraph(two);
						REQUIRE(stagedOne.staged && stagedTwo.staged && stagedTwo.generation > stagedOne.generation, "a later staging replaces an earlier one");
						REQUIRE(!record(kCount, versions->Current(), false, org::RunRef::kNoCopy, stagedOne.generation), "a replaced staged graph records nothing");
						auto last = record(kCount, versions->Current(), false, org::RunRef::kNoCopy, stagedTwo.generation);
						REQUIRE(last && host.AttachStagedEpochGraphs(std::array<uint32_t, 1>{0}).attached, "the last staged graph is attached");
						value = 79500u;
						host.SetEpochCopy(0, 0);
						host.SubmitEpoch(0, upload, {}, last);
						REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value), "the last staged graph's recording submits");
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
				// A live epoch publishes nothing. A recording request for it completes with an error, and a
				// recording given to its submission is rejected without hanging the host.
				host.SetAsyncEpochs({0});
				{
					value = 69990u;
					host.SubmitEpoch(0, upload);
					REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value), "the live epoch after the switch");
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
						REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value), "the live epoch draws meanwhile");
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
					REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value), "the live epoch submits again after the rejection");
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
					REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value), "an external execution writes and copies");
				}
				if (externalReuse) {
					const auto recorded = host.TakeAsyncStats();
					std::printf("external-reuse: %llu recorded, %llu bound a kept recording\n", static_cast<unsigned long long>(recorded.recorded),
						static_cast<unsigned long long>(recorded.liveReused));
					// The late pass is recorded at every execution (checked above); the rest of the run is recorded or bound from its copy.
					REQUIRE(recorded.recorded + recorded.liveReused >= 6, "each run records or binds its copy's kept recording");
				}
				// An execution that is not executed. It is signalled without running its work. The output keeps its last value, and
				// the next execution goes on.
				const uint32_t kept = value;
				value = 95000u;
				host.SubmitEpoch(0, upload);
				auto skipped = host.TakeExternalExecution();
				REQUIRE(skipped, "the execution to skip");
				host.SignalExternalExecutionUnexecuted(*skipped);
				REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, kept), "an unexecuted execution writes nothing");
				value = 96000u;
				host.SubmitEpoch(0, upload);
				auto next = host.TakeExternalExecution();
				REQUIRE(next && execute(*next) && waitIdle() == rhi::Result::Ok && Matches(first, value), "the epoch after a skipped execution");
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
					REQUIRE(waitIdle() == rhi::Result::Ok && Matches(first, value), "the timed executions write and copy");
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
				REQUIRE(waitIdle() == rhi::Result::Ok, "wait for the owned submission");
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
				// "sync-submit": the execution may still be queued; the checks below are of a submitted one.
				if (syncSubmit) host.DrainSubmissions("test: a submitted frame");
				if (inlineLate && (!async || frame != 0)) {
					REQUIRE(g_copyRecords.load(std::memory_order_relaxed) == copies + 1, "the inline late copy is recorded once per submission");
					REQUIRE(!host.TakeExternalExecution(), "an inline epoch hands no execution back");
				}
				syncOwner.reset();
				if (!async || frame != 0)
					REQUIRE(counters.locks.load() > locksBefore, "Execute submitted the current frame before returning");
				REQUIRE(waitIdle() == rhi::Result::Ok, "wait for BasicRHI timelines");
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
			if (syncReuse) {
				// Runs a window of frames and returns what the kept epoch's runs did. Each frame must write its own value.
				bool windowFailed = false;
				const auto window = [&](uint32_t frames, uint32_t base, const std::function<void(uint32_t)>& before = {}) {
					(void)host.TakeKeptStats(0);
					for (uint32_t frame = 0; frame < frames && !windowFailed; ++frame) {
						if (before) before(frame);
						value = base + 10u * frame;
						host.ExecuteFrame(nullptr, upload, 0);
						if (waitIdle() != rhi::Result::Ok || !Matches(second, value)) {
							std::fprintf(stderr, "FAIL: sync-reuse frame %u wrote %u..., expected %u...\n", frame, second.mapped[0], value);
							windowFailed = true;
						}
					}
					return host.TakeKeptStats(0);
				};
				const auto print = [](const char* what, const org::GraphHost::KeptStats& stats) {
					std::string reasons;
					for (const auto& [reason, count] : stats.reasons) reasons += (reasons.empty() ? "" : "; ") + reason + " x" + std::to_string(count);
					std::printf("sync-reuse: %s: %llu runs, %llu reused, %llu recorded and kept, %llu not keepable (%s)\n", what,
						static_cast<unsigned long long>(stats.runs), static_cast<unsigned long long>(stats.reused),
						static_cast<unsigned long long>(stats.kept), static_cast<unsigned long long>(stats.unkeepable),
						reasons.empty() ? "none" : reasons.c_str());
				};
				const uint64_t copies = 4;  // framesInFlight (3) + 1
				// Steady frames: a copy whose recording is older than the host buffer's rotation records once; every later run of a
				// copy submits its kept recording and records nothing.
				const auto recordsBefore = g_copyRecords.load(std::memory_order_relaxed);
				auto steady = window(12, 20000u);
				print("steady", steady);
				REQUIRE(!windowFailed, "sync-reuse: steady frames write their values");
				REQUIRE(steady.runs == 12 && steady.unkeepable == 0 && steady.reused + steady.kept == 12, "every steady run reuses or keeps");
				REQUIRE(steady.reused >= 12 - copies, "steady runs submit their copies' kept recordings");
				const auto steadyRecords = g_copyRecords.load(std::memory_order_relaxed) - recordsBefore;
				if (inlineLate) REQUIRE(steadyRecords == 12, "the inline late copy is recorded at every submission, a kept recording's too");
				else REQUIRE(steadyRecords == steady.kept, "a reused run records nothing");
				// A pass's revision changes: each copy records once again, for that reason.
				outputRevision->fetch_add(1, std::memory_order_release);
				auto revised = window(8, 21000u);
				print("revision", revised);
				REQUIRE(!windowFailed, "sync-reuse: frames after a revision write their values");
				REQUIRE(revised.kept == copies && revised.reused == 8 - copies && revised.reasons["revision: test.copy-to-host"] == copies,
					"a revision change records each copy once, named");
				// O-M6-2: a new version of a versioned buffer the epoch does not bind leaves every kept recording current, the host's
				// version count notwithstanding.
				auto unrelatedFirst = org::Buffer::CreateShared(rhi::HeapType::DeviceLocal, kBytes);
				auto unrelated = org::VersionedBuffer::Create(unrelatedFirst);
				auto unbound = window(4, 22000u, [&](uint32_t frame) {
					if (frame != 0) return;
					(void)unrelated->GrowBytes(kBytes * 2);
					host.NoteNewVersions();
				});
				print("version of an unbound slot", unbound);
				REQUIRE(!windowFailed && unbound.reused == 4, "a version of a slot the epoch does not bind leaves its recordings current");
				// A new version of the buffer the epoch writes and copies: each copy records once again, naming it.
				const auto grown = versions->GrowBytes(kBytes * 2);
				REQUIRE(grown && versions->Get() == grown->buffer, "the scratch buffer's new version");
				auto bound = window(8, 23000u);
				print("version of a bound slot", bound);
				uint64_t versionRecords = 0;
				for (const auto& [reason, count] : bound.reasons) if (reason.rfind("version: ", 0) == 0) versionRecords += count;
				REQUIRE(!windowFailed && bound.kept == copies && versionRecords == copies && bound.reused == 8 - copies,
					"a version of a bound slot records each copy once, named");
				// A rebuild: the new graph has no kept recordings.
				host.RequestRebuild();
				auto rebuilt = window(6, 24000u);
				print("rebuild", rebuilt);
				REQUIRE(!windowFailed && rebuilt.reasons["first run"] == copies && rebuilt.kept == copies && rebuilt.reused == 2,
					"a rebuilt graph's copies record once each");
				// Keeping off: the runs take leases of their own again and keep nothing.
				host.SetKeptEpochs({});
				auto off = window(3, 25000u);
				REQUIRE(!windowFailed && off.runs == 0, "an epoch that keeps nothing counts no kept runs");
				host.SetKeptEpochs({0});
				// O-M6-7, the revision audit, on every reuse. Recordings keep their frame data only while it is on, so the graph is
				// built again first. Steady reuses compare equal; an input the write pass's revision omits is reported by name while
				// the kept recordings stay current (and still write each frame's value).
				const auto printAudit = [](const char* what, const org::GraphHost::KeptStats& stats) {
					std::string names;
					for (const auto& [name, count] : stats.mismatches) names += (names.empty() ? "" : "; ") + name + " x" + std::to_string(count);
					std::printf("sync-reuse: audit %s: %llu reused, %llu audited, %llu passes compared, %llu not comparable, %llu mismatches (%s)\n",
						what, static_cast<unsigned long long>(stats.reused), static_cast<unsigned long long>(stats.audited),
						static_cast<unsigned long long>(stats.auditCompared), static_cast<unsigned long long>(stats.auditUncomparable),
						static_cast<unsigned long long>(stats.auditMismatches), names.empty() ? "none" : names.c_str());
				};
				host.SetKeptRecordingAudit(1);
				host.RequestRebuild();
				(void)window(copies + 2, 26000u);
				auto audited = window(8, 27000u);
				printAudit("steady", audited);
				REQUIRE(!windowFailed && audited.reused == 8 && audited.audited == 8 && audited.auditCompared == 16
					&& audited.auditUncomparable == 0 && audited.auditMismatches == 0, "the audit finds the kept recordings' frame data unchanged");
				g_unrevisedInput.store(1, std::memory_order_relaxed);
				auto omitted = window(8, 28000u);
				printAudit("unrevised input", omitted);
				REQUIRE(!windowFailed && omitted.reused == 8 && omitted.auditMismatches == 8 && omitted.mismatches["test.write"] == 8,
					"the audit names a pass whose revision omits an input");
				g_unrevisedInput.store(0, std::memory_order_relaxed);
				host.SetKeptRecordingAudit(0);
				auto unaudited = window(4, 29000u);
				REQUIRE(!windowFailed && unaudited.audited == 0, "the audit off audits nothing");
				// O-M6-4, staged recordings: a graph built on another thread records its kept copies there before it is staged
				// (GraphHost::RecordStagedCopies). After the attach every copy's first run submits its staged recording (confirmed by
				// its frame data) and records nothing. A staging whose frame data differs from the runs' (an input no revision names)
				// is recorded again at each copy's first run, named; so is one whose revision changed between the staging and the run.
				const auto stage = [&](const std::function<void()>& before, const std::function<void()>& after, size_t byRun = 0) -> int {
					org::GraphHost::StagedRecordings staged;
					std::unique_ptr<org::GraphHost::EpochGraph> graph;
					const auto recordsBefore = g_copyRecords.load(std::memory_order_relaxed);
					std::thread builder([&] {
						graph = host.BuildEpochGraph(0, "staged recordings");
						if (!graph || !graph->Built()) return;
						if (before) before();
						if (byRun) {
							const std::vector<org::UpdateExecutionContext> runs(byRun);
							staged = host.RecordStagedCopies(*graph, runs);
						} else {
							staged = host.RecordStagedCopies(*graph, org::UpdateExecutionContext{});
						}
						if (after) after();
					});
					builder.join();
					const auto stagedRecords = g_copyRecords.load(std::memory_order_relaxed) - recordsBefore;
					std::printf("sync-reuse: staged: %u of %u copies recorded on the building thread in %.2f ms (%s)\n", staged.recorded,
						staged.copies, staged.milliseconds, staged.why.empty() ? "all" : staged.why.c_str());
					REQUIRE(graph && graph->Built(), "the staged build");
					const auto expected = byRun ? (std::min<uint32_t>)(static_cast<uint32_t>(byRun), copies) : copies;
					REQUIRE(staged.copies == copies && staged.recorded == expected, "every kept copy asked for is recorded on the building thread");
					if (!inlineLate) REQUIRE(stagedRecords == expected, "each staged copy is recorded once");
					const auto result = host.StageEpochGraph(graph);
					REQUIRE(result.staged, "the staged build is staged");
					const uint32_t epochs[] = {0};
					REQUIRE(host.AttachStagedEpochGraphs(epochs).attached, "the staged set attaches");
					return 0;
				};
				if (const int failed = stage({}, {})) return failed;
				const auto stagedRecordsBefore = g_copyRecords.load(std::memory_order_relaxed);
				auto staged = window(8, 30000u);
				print("staged", staged);
				REQUIRE(!windowFailed && staged.reused == 8 && staged.kept == 0 && staged.stagedReused == copies && staged.stagedRecorded == 0,
					"every copy's first run after the attach submits its staged recording");
				if (!inlineLate)
					REQUIRE(g_copyRecords.load(std::memory_order_relaxed) == stagedRecordsBefore, "runs of staged recordings record nothing");
				else
					REQUIRE(g_copyRecords.load(std::memory_order_relaxed) == stagedRecordsBefore + 8, "the inline late copy is recorded at each submission");
				// The staging's frame data differ from the runs' (g_unrevisedInput, which no revision names): confirmed against the
				// first run's own preparation, each copy records again.
				if (const int failed = stage([] { g_unrevisedInput.store(1, std::memory_order_relaxed); }, [] { g_unrevisedInput.store(0, std::memory_order_relaxed); }))
					return failed;
				auto differs = window(8, 31000u);
				print("staged, other frame data", differs);
				REQUIRE(!windowFailed && differs.stagedRecorded == copies && differs.stagedReused == 0 && differs.kept == copies
					&& differs.reasons["staged: frame data of test.write"] == copies && differs.reused == 8 - copies,
					"a staged recording whose frame data differ from its first run's records again, named");
				// A revision that changes between the staging and the attach.
				if (const int failed = stage({}, [&] { outputRevision->fetch_add(1, std::memory_order_release); })) return failed;
				auto stale = window(8, 32000u);
				print("staged, revision changed", stale);
				REQUIRE(!windowFailed && stale.stagedRecorded == copies && stale.reasons["staged: revision: test.copy-to-host"] == copies
					&& stale.reused == 8 - copies, "a staged recording whose revision changed records again, named");
				// M7 step 2, staged recordings by run (RecordStagedCopies with one frame per run): copies 0 and 1 are recorded for the
				// new graph's first two runs. Its copies are free at its first run (every frame here waits for the device), so its runs
				// take copy 0, then 1, wherever the old graph's turn was: both submit their staged recordings, and the later copies
				// record at their first runs. Twice, from two positions of the turn.
				if (copies >= 3) {
					for (uint32_t offset = 1; offset <= 2; ++offset) {
						(void)window(offset, 32500u + 100u * offset);
						if (const int failed = stage({}, {}, 2)) return failed;
						auto byRun = window(2, 32600u + 100u * offset);
						print("staged by run, first two runs", byRun);
						REQUIRE(!windowFailed && byRun.stagedReused == 2 && byRun.kept == 0 && byRun.reused == 2,
							"a graph staged by run takes its staged copies in run order");
						auto later = window(copies, 32700u + 100u * offset);
						print("staged by run, later runs", later);
						REQUIRE(!windowFailed && later.reasons["first run"] == copies - 2 && later.reused == 2,
							"the copies after the staged runs record at their first runs");
					}
				}
				// O-M6-6, batch reuse (GraphHost::SetKeptBatchReuse): each batch of a kept recording is checked on its own. With "inline"
				// the late copy is a batch of its own, so the write's batch stays current when only the copy is stale: a partial run
				// submits the write's kept packet, prepares only the copy and records its batch. Without "inline" the epoch is one
				// batch, and a stale pass records it, as without batch reuse.
				const auto printBatches = [](const char* what, const org::GraphHost::KeptStats& stats) {
					std::string reasons;
					for (const auto& [reason, count] : stats.batchReasons) reasons += (reasons.empty() ? "" : "; ") + reason + " x" + std::to_string(count);
					std::printf("sync-reuse: batches, %s: %llu runs, %llu reused, %llu partial, %llu kept, %llu not keepable; batches %llu reused, "
						"%llu recorded (%s)\n", what, static_cast<unsigned long long>(stats.runs), static_cast<unsigned long long>(stats.reused),
						static_cast<unsigned long long>(stats.partial), static_cast<unsigned long long>(stats.kept),
						static_cast<unsigned long long>(stats.unkeepable), static_cast<unsigned long long>(stats.batchesReused),
						static_cast<unsigned long long>(stats.batchesRecorded), reasons.empty() ? "none" : reasons.c_str());
				};
				host.SetKeptBatchReuse(true);
				host.RequestRebuild();
				(void)window(copies + 2, 33000u);
				auto batchSteady = window(8, 34000u);
				printBatches("steady", batchSteady);
				REQUIRE(!windowFailed && batchSteady.reused == 8 && batchSteady.partial == 0, "batch reuse: steady runs submit their whole recordings");
				outputRevision->fetch_add(1, std::memory_order_release);
				auto prepares = g_writePrepares.load(std::memory_order_relaxed);
				auto batchRevised = window(8, 35000u);
				printBatches("revision of the copy", batchRevised);
				REQUIRE(!windowFailed && batchRevised.reasons["revision: test.copy-to-host"] == copies && batchRevised.reused == 8 - copies,
					"batch reuse: a revision change records each copy once, named");
				if (inlineLate) {
					REQUIRE(batchRevised.partial == copies && batchRevised.kept == 0 && batchRevised.batchesReused == 8 + 8 - copies
						&& batchRevised.batchesRecorded == copies && g_writePrepares.load(std::memory_order_relaxed) == prepares,
						"batch reuse: only the copy's batch records; the write's is submitted from the kept recording, not prepared");
				} else {
					REQUIRE(batchRevised.partial == 0 && batchRevised.kept == copies, "batch reuse: a one-batch epoch records its batch whole");
				}
				// A pass with no revision: its batch records at every run, and with "inline" the write's batch is submitted again.
				// The audit on before the rebuild: recordings keep their frame data only while it is on.
				uncheckedCopy = true;
				host.SetKeptRecordingAudit(1);
				host.RequestRebuild();
				(void)window(copies + 2, 36000u);
				prepares = g_writePrepares.load(std::memory_order_relaxed);
				auto batchUnchecked = window(8, 37000u);
				printBatches("a pass without a revision", batchUnchecked);
				printAudit("a pass without a revision", batchUnchecked);
				host.SetKeptRecordingAudit(0);
				if (inlineLate) {
					REQUIRE(!windowFailed && batchUnchecked.partial == 8 && batchUnchecked.batchesReused == 8 && batchUnchecked.batchesRecorded == 8
						&& batchUnchecked.batchReasons["not kept: uncheckable: test.copy-to-host (no revision hash)"] == 8,
						"batch reuse: the batch of a pass without a revision records at every run, the others are submitted again");
					REQUIRE(batchUnchecked.audited == 8 && batchUnchecked.auditCompared == 8 && batchUnchecked.auditMismatches == 0
						&& g_writePrepares.load(std::memory_order_relaxed) == prepares + 8,
						"batch reuse: the audit compares the reused batch's pass (prepared only for it)");
				} else {
					REQUIRE(!windowFailed && batchUnchecked.unkeepable == 8 && batchUnchecked.partial == 0,
						"batch reuse: a one-batch epoch with a pass without a revision keeps nothing");
				}
				uncheckedCopy = false;
				host.SetKeptBatchReuse(false);
				host.RequestRebuild();
				(void)window(copies + 2, 38000u);
				REQUIRE(!windowFailed, "sync-reuse: frames after batch reuse write their values");
				if (syncSubmit) {
					const auto printWaits = [](const char* what, const org::GraphHost::SubmitThreadStats& stats) {
						std::string reasons;
						for (const auto& [why, counts] : stats.epochWaitsByReason)
							reasons += (reasons.empty() ? "" : "; ") + why + " x" + std::to_string(counts.first);
						std::printf("sync-submit: %s: %llu executions on the thread (%llu failed), %llu waits for queued executions (%.1f ms: %s)\n",
							what, static_cast<unsigned long long>(stats.executions), static_cast<unsigned long long>(stats.failedExecutions),
							static_cast<unsigned long long>(stats.epochWaits), stats.epochWaitUs / 1000.0, reasons.empty() ? "none" : reasons.c_str());
					};
					// 1. ExecuteFrame returns with its execution still queued. The epoch's next ExecuteFrame waits for it before its
					// Update, and both frames go through.
					(void)host.TakeSubmitThreadStats();
					(void)host.TakeSubmittedExecutions();
					auto gate = HoldSubmissionThread(host);
					value = 39000u;
					host.ExecuteFrame(nullptr, upload, 0);
					std::this_thread::sleep_for(std::chrono::milliseconds(20));
					REQUIRE(!Matches(second, value) && !host.SubmittedPoint().Passed(), "sync-submit: ExecuteFrame returns with its execution queued");
					auto opener = OpenGateLater(gate, 100);
					const auto waitStart = std::chrono::steady_clock::now();
					value = 39010u;
					host.ExecuteFrame(nullptr, upload, 0);
					const double waitedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - waitStart).count();
					opener.join();
					REQUIRE(waitIdle() == rhi::Result::Ok && Matches(second, value), "sync-submit: the frame after a queued one writes its value");
					auto waits = host.TakeSubmitThreadStats();
					printWaits("a queued execution", waits);
					bool previousWait = false;
					for (const auto& [why, counts] : waits.epochWaitsByReason) previousWait = previousWait || why == "the epoch's previous execution";
					REQUIRE(waitedMs >= 50.0 && waits.epochWaits >= 1 && previousWait && waits.executions == 2,
						"sync-submit: the epoch's next ExecuteFrame waits for its queued execution before its Update");
					const auto executions = host.TakeSubmittedExecutions();
					REQUIRE(executions.size() == 2 && !executions[0].failed && executions[1].kept != org::RenderGraph::KeptRunOutcome::None,
						"sync-submit: the thread reports each execution with its kept run");
					// 2. Frames back to back, no device waits between them: the last one's value is written.
					(void)host.TakeKeptStats(0);
					for (uint32_t frame = 0; frame < 16; ++frame) {
						value = 39100u + 10u * frame;
						host.ExecuteFrame(nullptr, upload, 0);
					}
					REQUIRE(waitIdle() == rhi::Result::Ok && Matches(second, value), "sync-submit: frames back to back");
					const auto backToBack = host.TakeKeptStats(0);
					print("back to back", backToBack);
					REQUIRE(backToBack.runs == 16 && backToBack.reused + backToBack.kept == 16, "sync-submit: back-to-back runs reuse or keep");
					printWaits("back to back", host.TakeSubmitThreadStats());
					// 3. A graph staged by run, attached while the last run of the old graph is still queued on kept copy 0. The staging
					// recorded into the epoch's other set of copies, whose runs are done, so the new graph's first run takes copy 0 of
					// that set (another lease than the queued run's) and submits its staged recording. Before the second set (M7 step 8),
					// the queued copy 0 was not free, so the first run took copy 1 and recorded again.
					uint32_t lastCopy = org::RunRef::kNoCopy;
					uint32_t lastLease = UINT32_MAX;
					const auto noteCopy = [&](org::RenderGraph& graph) {
						upload(graph);
						lastCopy = host.CurrentRun().copy;
						lastLease = host.CurrentRun().lease;
					};
					for (uint32_t frame = 0; frame < copies && lastCopy != copies - 1; ++frame) {
						value = 39300u + 10u * frame;
						host.ExecuteFrame(nullptr, noteCopy, 0);
						REQUIRE(waitIdle() == rhi::Result::Ok && Matches(second, value), "sync-submit: frames before the staging");
					}
					REQUIRE(lastCopy == copies - 1, "sync-submit: the turn reaches the last copy");
					gate = HoldSubmissionThread(host);
					value = 39400u;
					host.ExecuteFrame(nullptr, noteCopy, 0);
					REQUIRE(lastCopy == 0, "sync-submit: the queued run takes copy 0");
					const uint32_t queuedLease = lastLease;
					if (const int failed = stage({}, {}, 2)) return failed;
					(void)host.TakeKeptStats(0);
					opener = OpenGateLater(gate, 100);
					value = 39410u;
					host.ExecuteFrame(nullptr, noteCopy, 0);
					opener.join();
					const uint32_t firstCopy = lastCopy;
					REQUIRE(waitIdle() == rhi::Result::Ok && Matches(second, value), "sync-submit: the new graph's first run after a queued one");
					const auto firstRun = host.TakeKeptStats(0);
					std::printf("sync-submit: a graph staged by run, attached while copy 0 is queued: its first run takes copy %u (lease %u, the "
						"queued run's %u), %llu staged reused, %llu recorded again\n", firstCopy, lastLease, queuedLease,
						static_cast<unsigned long long>(firstRun.stagedReused), static_cast<unsigned long long>(firstRun.stagedRecorded));
					REQUIRE(firstCopy == 0 && lastLease != queuedLease && firstRun.stagedReused == 1 && firstRun.stagedRecorded == 0,
						"sync-submit: a graph staged by run takes copy 0 of the other set while the current set's copy 0 is queued");
					(void)window(copies + 2, 39500u);
					REQUIRE(!windowFailed, "sync-submit: frames after the attach write their values");
					// 4. A pass that fails to record on the thread. The thread goes on; the epoch's next ExecuteFrame throws
					// EpochFailedError, the failure is reported, and a rebuild recovers.
					(void)host.TakeSubmitThreadStats();
					outputRevision->fetch_add(1, std::memory_order_release);  // so the copy records (not a kept recording's reuse)
					g_failCopyRecord.store(1, std::memory_order_release);
					value = 39600u;
					host.ExecuteFrame(nullptr, upload, 0);
					host.DrainSubmissions("test: the failing execution");
					bool threw = false;
					try {
						value = 39610u;
						host.ExecuteFrame(nullptr, upload, 0);
					} catch (const org::GraphHost::EpochFailedError& error) {
						threw = true;
						std::printf("sync-submit: the next ExecuteFrame threw: %s\n", error.what());
					}
					const auto failedStats = host.TakeSubmitThreadStats();
					printWaits("a failing pass", failedStats);
					REQUIRE(threw && failedStats.failedExecutions == 1, "sync-submit: a failure on the thread fails the epoch's next ExecuteFrame");
					REQUIRE(host.PollEpochFailures() >= 1, "sync-submit: the failure is reported");
					g_failCopyRecord.store(0, std::memory_order_release);
					host.RequestRebuild();
					(void)window(copies + 2, 39700u);
					REQUIRE(!windowFailed, "sync-submit: a rebuild recovers the failed epoch");
					// 5. External client submissions (O-M7-3): a copy of the output the epoch's copy pass writes, into the probe.
					{
						auto* graph = host.Graph();
						REQUIRE(graph, "sync-submit: the epoch's graph");
						auto& client = graph->GetExternalClient();
						const auto copySlot = static_cast<uint32_t>(org::ToUnderlying(graph->GetQueueRegistry().FindGraphicsSlot()));
						const org::ResourceState copySource{rhi::ResourceAccessType::CopySource, rhi::ResourceLayout::CopySource,
							rhi::ResourceSyncState::Copy};
						const org::ResourceState copyDest{rhi::ResourceAccessType::CopyDest, rhi::ResourceLayout::CopyDest,
							rhi::ResourceSyncState::Copy};
						auto probeBuffer = ImportHostBuffer(device.Get(), probe, "test.external-probe");
						REQUIRE(probeBuffer, "sync-submit: the probe buffer");
						const auto copyOut = [&] {
							const std::array<org::ExternalClient::Access, 2> accesses{{{output.get(), {}, copySource}, {probeBuffer.get(), {}, copyDest}}};
							return client.Submit(copySlot, accesses, [&](rhi::CommandList& list) {
								list.CopyBufferRegion(probeBuffer->GetAPIResource().GetHandle(), 0, output->GetAPIResource().GetHandle(), 0, kBytes);
							});
						};
						const auto reached = [](org::ExternalClient::Signal signal) {
							return signal.timeline && signal.timeline.HostWait(signal.value, 5000) == rhi::Result::Ok;
						};
						(void)host.TakeSubmitThreadStats();
						// a. Nothing queued: made on the calling thread, after what the thread already submitted.
						REQUIRE(waitIdle() == rhi::Result::Ok && Matches(second, value), "sync-submit: the frames before the external copies");
						std::memset(probe.mapped, 0, kBytes);
						const auto queuedBefore = client.QueuedSubmissions();
						const auto direct = copyOut();
						REQUIRE(client.QueuedSubmissions() == queuedBefore, "sync-submit: with nothing queued, an external submission is not queued");
						REQUIRE(reached(direct) && Matches(probe, value), "sync-submit: the direct external copy reads the last frame's output");
						// b. The epoch's execution still queued (the thread held): the copy is queued behind it, and reads what it writes.
						auto heldGate = HoldSubmissionThread(host);
						value = 39800u;
						host.ExecuteFrame(nullptr, upload, 0);
						auto queued = copyOut();
						std::this_thread::sleep_for(std::chrono::milliseconds(20));
						REQUIRE(client.QueuedSubmissions() == queuedBefore + 1 && client.PendingQueuedSubmissions() == 1,
							"sync-submit: an external submission made while an execution is queued is queued behind it");
						REQUIRE(queued.timeline && queued.timeline.GetCompletedValue() < queued.value && !Matches(probe, value),
							"sync-submit: the queued external copy has not run before the thread passes the execution");
						OpenGate(heldGate);
						REQUIRE(reached(queued) && Matches(probe, value), "sync-submit: the queued external copy reads what the queued execution wrote");
						REQUIRE(client.PendingQueuedSubmissions() == 0, "sync-submit: the queued external submission passed");
						// Two in a row behind a queued execution: their values rise, and both run after it.
						heldGate = HoldSubmissionThread(host);
						value = 39810u;
						host.ExecuteFrame(nullptr, upload, 0);
						const auto queuedFirst = copyOut();
						const auto queuedSecond = copyOut();
						REQUIRE(queuedSecond.timeline.GetHandle().index == queuedFirst.timeline.GetHandle().index && queuedSecond.value == queuedFirst.value + 1,
							"sync-submit: queued submissions take rising values on the slot's queued timeline");
						OpenGate(heldGate);
						REQUIRE(reached(queuedFirst) && reached(queuedSecond) && Matches(probe, value), "sync-submit: both queued copies after the execution");
						// c. Made on the thread itself (a closure there): made at once, not queued again.
						std::memset(probe.mapped, 0, kBytes);
						org::ExternalClient::Signal onThread{};
						host.EnqueueOnSubmitThread([&] { onThread = copyOut(); });
						host.DrainSubmissions("test: an external submission on the thread");
						REQUIRE(client.QueuedSubmissions() == queuedBefore + 3 && reached(onThread) && Matches(probe, value),
							"sync-submit: an external submission made on the submission thread is made there");
						const auto externalStats = host.TakeSubmitThreadStats();
						std::printf("sync-submit: %llu external submissions queued behind the thread\n",
							static_cast<unsigned long long>(externalStats.externalQueued));
						REQUIRE(externalStats.externalQueued == 3, "sync-submit: the thread counts the queued external submissions");
						(void)window(copies + 2, 39900u);
						REQUIRE(!windowFailed, "sync-submit: frames after the external submissions write their values");
					}
					// 6. Kept recordings' timestamps are slices of shared pools (StatisticsManager::CreateOwnedTimestamps). Slices do not
					// overlap; a released one is used again only after kRecycleFrames frames; no new pool is made while one has room. Then
					// the kept recordings' timestamps read back from their slices.
					{
						auto& statistics = org::StatisticsManager::GetInstance();
						auto a = statistics.CreateOwnedTimestamps(rhi::QueueKind::Graphics, 40);
						auto b = statistics.CreateOwnedTimestamps(rhi::QueueKind::Graphics, 40);
						const auto pools = statistics.GetOwnedTimestampPoolStats();
						REQUIRE(a && b && a->slice && b->slice && a->poolHandle.index == b->poolHandle.index
							&& (a->first + 80 <= b->first || b->first + 80 <= a->first), "timestamps: two slices of one pool, not overlapping");
						const auto releasedFirst = a->first;
						a.reset();
						auto c = statistics.CreateOwnedTimestamps(rhi::QueueKind::Graphics, 40);
						REQUIRE(c && c->first != releasedFirst, "timestamps: a released slice is not used again in its frame");
						for (uint32_t frame = 0; frame < 9; ++frame) statistics.BeginFrame();
						std::vector<std::shared_ptr<org::runtime::OwnedTimestampQueries>> more;
						bool reused = false;
						for (uint32_t i = 0; i < 64 && !reused; ++i) {
							more.push_back(statistics.CreateOwnedTimestamps(rhi::QueueKind::Graphics, 40));
							reused = more.back() && more.back()->first == releasedFirst;
						}
						const auto after = statistics.GetOwnedTimestampPoolStats();
						std::printf("timestamps: %u pools (%llu queries, %llu sliced), %llu made, %llu slices made\n", after.blocks,
							static_cast<unsigned long long>(after.queries), static_cast<unsigned long long>(after.sliced),
							static_cast<unsigned long long>(after.blocksMade), static_cast<unsigned long long>(after.slicesMade));
						REQUIRE(reused && after.blocksMade == pools.blocksMade && after.slicesMade >= pools.slicesMade + 2,
							"timestamps: a released slice is used again after kRecycleFrames frames, from the same pool");
						more.clear();
						b.reset();
						c.reset();
						// The kept recordings' own timestamps, recorded again with them on, read back from their slices.
						std::mutex timedMutex;
						uint32_t replayed = 0;
						host.SetCompletedFrameCallback([&](uint64_t, const org::runtime::IStatisticsService& a_stats) {
							std::scoped_lock lock(timedMutex);
							for (const auto& timestamp : a_stats.GetCompletedTimestamps())
								if (timestamp.replayed && timestamp.beginTick && timestamp.endTick >= timestamp.beginTick) ++replayed;
						});
						host.SetRecordingTimestamps(true);
						(void)window(copies + 6, 39950u);
						REQUIRE(!windowFailed, "timestamps: frames with recording timestamps write their values");
						host.SetRecordingTimestamps(false);
						(void)window(copies + 2, 39980u);
						host.SetCompletedFrameCallback({});
						std::printf("timestamps: %u kept recording timestamps read back\n", replayed);
						REQUIRE(replayed > 0, "timestamps: kept recordings' timestamps read back from their slices");
					}
				}
			}
			if (reuse) {
				// Steady frames. A live epoch's runs take kept copies. Each copy records once, and its later runs bind that recording.
				(void)host.TakeAsyncStats();
				for (uint32_t frame = 0; frame < 12; ++frame) {
					value = 20000u + 10u * frame;
					host.SubmitEpoch(0, upload);
					REQUIRE(waitIdle() == rhi::Result::Ok && Matches(second, value), "a run's recording writes its frame's value");
				}
				const auto steady = host.TakeAsyncStats();
				std::printf("reuse: steady %llu recorded, %llu bound a kept recording, %llu kept, %llu prepared again\n",
					static_cast<unsigned long long>(steady.recorded), static_cast<unsigned long long>(steady.liveReused),
					static_cast<unsigned long long>(steady.liveKept), static_cast<unsigned long long>(steady.stale));
				// The window's first ticket is recorded ahead, on the host's thread, after the last submission before the window: it may be
				// counted before the window opens.
				REQUIRE(steady.recorded + steady.liveReused >= 11 && steady.stale == 0, "each steady run records or binds a kept recording");
				// A few copies cycle (the next ticket is prepared while the last run may still be on the GPU), so most runs bind.
				REQUIRE(steady.liveReused >= 6, "steady frames bind their copies' kept recordings");
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
					REQUIRE(waitIdle() == rhi::Result::Ok && Matches(second, value), "a prepared-again ticket writes its frame's value");
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
				REQUIRE(waitIdle() == rhi::Result::Ok && Matches(second, value), "a ticket made stale by its commit is prepared again");
			}

			// A rebuild (e.g. on resize) retires the old graph and keeps rendering.
			host.RequestRebuild();
			value = 7000u;
			if (async) host.SubmitEpoch(0, upload);
			else host.ExecuteFrame(nullptr, upload);
			REQUIRE(waitIdle() == rhi::Result::Ok, "wait after rebuild");
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
					REQUIRE(!submitted.points.empty() || submitted.pending, "a submitted point");
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
					for (const auto& [timeline, at] : a_point.Resolve()) out.emplace_back(timeline.get(), at);
					return out;
				};
				REQUIRE(host.SealedPoint().points.empty() && !host.SealedPoint().pending, "nothing sealed before the first seal");
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
	DestroyHostBuffer(host, probe);
	vkDestroyDevice(host.device, nullptr);
	vkDestroyInstance(host.instance, nullptr);
	std::printf("PersistentVulkanHostTests: ok (%s%s, locks=%d)\n", contributor ? "contributor" : external ? "external" : async ? "async" : syncSubmit ? "sync-submit" : syncReuse ? "sync-reuse" : "sync",
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
