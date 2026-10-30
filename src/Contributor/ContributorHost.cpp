#include <OpenRenderGraph/ContributorHost.h>

#include <OpenRenderGraph/GraphHost.h>

#include "BorrowedRecorder.h"
#include "DeviceTable.h"

#include "Render/PassBuilders.h"
#include "Render/RenderGraph/RenderGraph.h"
#include "Render/Runtime/IDescriptorService.h"
#include "Render/Runtime/UploadTypes.h"
#include "RenderPasses/Base/TypedRenderGraphPass.h"
#include "Resources/Buffers/Buffer.h"
#include "Resources/PixelBuffer.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <unordered_map>

namespace org::contributor {

namespace {

template <class To, class From>
To ConvertHandle(const From& value) noexcept { return To{ value.index, value.generation }; }

ORGBackend ToORG(rhi::Backend backend) noexcept {
	switch (backend) {
	case rhi::Backend::D3D12: return ORG_RG_BACKEND_D3D12;
	case rhi::Backend::Vulkan: return ORG_RG_BACKEND_VULKAN;
	default: return ORG_RG_BACKEND_NONE;
	}
}

rhi::ResourceAccessType ToAccess(ORGAccessKind access) noexcept {
	using A = rhi::ResourceAccessType;
	switch (access) {
	case ORG_RG_ACCESS_SHADER_RESOURCE: return A::ShaderResource;
	case ORG_RG_ACCESS_CONSTANT_BUFFER: return A::ConstantBuffer;
	case ORG_RG_ACCESS_UNORDERED_ACCESS: return A::UnorderedAccess;
	case ORG_RG_ACCESS_UNORDERED_ACCESS_CLEAR: return A::UnorderedAccessClear;
	case ORG_RG_ACCESS_RENDER_TARGET: return A::RenderTarget;
	case ORG_RG_ACCESS_RENDER_TARGET_CLEAR: return A::RenderTargetClear;
	case ORG_RG_ACCESS_DEPTH_READ: return A::DepthRead;
	case ORG_RG_ACCESS_DEPTH_READ_WRITE: return A::DepthReadWrite;
	case ORG_RG_ACCESS_DEPTH_STENCIL_CLEAR: return A::DepthStencilClear;
	case ORG_RG_ACCESS_COPY_SOURCE: return A::CopySource;
	case ORG_RG_ACCESS_COPY_DESTINATION: return A::CopyDest;
	case ORG_RG_ACCESS_INDIRECT_ARGUMENT: return A::IndirectArgument;
	case ORG_RG_ACCESS_INDEX_BUFFER: return A::IndexBuffer;
	default: return A::Common;
	}
}

bool IntegerFormat(rhi::Format format) noexcept {
	using F = rhi::Format;
	switch (format) {
	case F::R32G32B32A32_UInt: case F::R32G32B32A32_SInt: case F::R32G32B32_UInt: case F::R32G32B32_SInt:
	case F::R16G16B16A16_UInt: case F::R16G16B16A16_SInt: case F::R32G32_UInt: case F::R32G32_SInt:
	case F::R10G10B10A2_UInt: case F::R8G8B8A8_UInt: case F::R8G8B8A8_SInt: case F::R16G16_UInt: case F::R16G16_SInt:
	case F::R32_UInt: case F::R32_SInt: case F::R8G8_UInt: case F::R8G8_SInt: case F::R16_UInt: case F::R16_SInt:
	case F::R8_UInt: case F::R8_SInt: case F::Unknown: return true;
	default: return false;
	}
}

RangeSpec ToRange(const ORGSubresourceRange& range) noexcept {
	RangeSpec spec;
	const auto bounds = [](uint32_t first, uint32_t count, Bound& lower, Bound& upper) {
		if (first) lower = {BoundType::From, first};
		if (count != UINT32_MAX) upper = {BoundType::UpTo, first + count - 1};
	};
	bounds(range.firstMip, range.mipCount, spec.mipLower, spec.mipUpper);
	bounds(range.firstArraySlice, range.arraySize, spec.sliceLower, spec.sliceUpper);
	return spec;
}

bool WholeResource(const ORGSubresourceRange& range) noexcept {
	return range.firstMip == 0 && range.mipCount == UINT32_MAX && range.firstArraySlice == 0 && range.arraySize == UINT32_MAX;
}

std::mutex g_hostMutex;
ContributorHost::Impl* g_host = nullptr; // the process's host: the C table has no context

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Host state

struct GenerationLease;

// One pass of one generation, shared by the pass object and every packet prepared from it.
struct PassSource {
	ContributorRegistry::Pass pass;
	std::vector<std::shared_ptr<Resource>> resources; // per access
	std::vector<ORGResourceDesc> shapes;              // per access: the declared shape (extents resolved), or the export's
	std::shared_ptr<GenerationLease> lease;
	ContributorHost::Impl* host = nullptr; // outlives the lease
	ORGBackend backend = ORG_RG_BACKEND_NONE;
};

struct ContributorHost::Impl : std::enable_shared_from_this<Impl> {
	Desc desc;
	ContributorRegistry registry;
	std::unique_ptr<DeviceTable> deviceTable;
	ORGPipelineService pipelineService{};
	std::vector<ORGAnchorDescriptor> anchors;
	std::vector<ORGEpochDescriptor> epochs;
	std::vector<ORGExportDescriptor> exports;

	// Owner thread (structural builds).
	std::shared_ptr<GenerationLease> current;
	std::vector<std::shared_ptr<const PassSource>> currentPasses;
	std::unordered_map<std::string, std::pair<ORGResourceDesc, std::shared_ptr<Resource>>> resources;
	Extents builtExtents{};
	bool hasRelativeResources = false;
	uint64_t nextGeneration = 1;
	std::atomic<uint64_t> activeGeneration{0};
};

// Retires its generation when the last pass object or packet of it goes away.
struct GenerationLease {
	std::weak_ptr<ContributorHost::Impl> host;
	uint64_t generation = 0;
	~GenerationLease() {
		if (auto locked = host.lock()) locked->registry.Retire(generation);
	}
};

// ---------------------------------------------------------------------------------------------------------------------
// The pass adapter

struct AccessTokens {
	DeclaredViewToken view, clearView;
	ResourceBindingToken resource{};
	bool hasView = false, hasClearView = false;
};
struct ContributorBindings { std::vector<AccessTokens> accesses; };

struct PacketOwner {
	ORGPreparedPass packet{};
	~PacketOwner() { if (packet.destroy) packet.destroy(packet.data); }
};

struct ContributorFrame {
	struct Attachment {
		rhi::DescriptorSlot slot{};
		PreparedResourceReference resource{};
		bool clear = false;
		bool readOnly = false;
		rhi::Format format = rhi::Format::Unknown;
		float value[4]{};
	};
	struct UavClear {
		rhi::DescriptorSlot gpu{}, cpu{};
		PreparedResourceReference resource{};
		bool integer = false;
		float value[4]{};
	};
	std::shared_ptr<PacketOwner> owner;
	std::shared_ptr<const PassSource> source;
	ORGFrameInfo frame{};
	std::vector<ORGBindingInfo> bindings;
	std::vector<std::optional<PreparedResourceReference>> resources; // per access, re-resolved when recording
	std::vector<Attachment> colors;
	std::optional<Attachment> depth;
	std::vector<UavClear> uavClears;
	uint32_t width = 0, height = 0;
};

// Forwards the packet's lifecycle callbacks (PreparedLifecycleEffect).
class PacketLifecycle final : public PreparedLifecycleEffect {
public:
	explicit PacketLifecycle(std::shared_ptr<PacketOwner> owner) : m_owner(std::move(owner)) {}
	void Submitted(SubmissionContext context) const override {
		if (m_owner->packet.submitted) m_owner->packet.submitted(m_owner->packet.data, context.submissionID);
	}
	void Completed(CompletionContext context) const override {
		if (m_owner->packet.completed) m_owner->packet.completed(m_owner->packet.data, context.submissionID);
	}
	void Abandoned(AbandonReason reason) const override {
		if (!m_owner->packet.abandoned) return;
		ORGAbandonReason value = ORG_RG_ABANDON_SHUTDOWN;
		switch (reason) {
		case AbandonReason::Shutdown: value = ORG_RG_ABANDON_SHUTDOWN; break;
		case AbandonReason::GenerationInvalidated: value = ORG_RG_ABANDON_GENERATION_INVALIDATED; break;
		case AbandonReason::PreparationFailed: value = ORG_RG_ABANDON_PREPARATION_FAILED; break;
		case AbandonReason::AdmissionFailed: value = ORG_RG_ABANDON_ADMISSION_FAILED; break;
		}
		m_owner->packet.abandoned(m_owner->packet.data, value);
	}
private:
	std::shared_ptr<PacketOwner> m_owner;
};

// The pass being prepared on this thread, for QueueBufferUpload (which receives only the prepare context).
struct PrepareScope {
	ORGPrepareContext context{};
	const PassSource* source = nullptr;
	struct Upload { runtime::UploadTarget target; uint64_t offset = 0; std::vector<std::byte> bytes; };
	std::vector<Upload>* uploads = nullptr;
};
thread_local PrepareScope* t_prepareScope = nullptr;

// Reusable: the contributor gave invocationRevision, so the framework may reuse the previous packet.
template<bool Reusable>
class ContributorPass final : public TypedRenderGraphPass<ContributorPass<Reusable>, ContributorFrame, ContributorBindings> {
public:
	explicit ContributorPass(std::shared_ptr<const PassSource> source) : m_source(std::move(source)) {}

	ContributorBindings Declare(PassBuilder& builder) {
		const auto& pass = m_source->pass;
		switch (pass.queue) {
		case ORG_RG_QUEUE_FORCE_GRAPHICS: builder.PreferQueue(QueueKind::Graphics); break;
		case ORG_RG_QUEUE_FORCE_COMPUTE: builder.PreferQueue(QueueKind::Compute); break;
		case ORG_RG_QUEUE_FORCE_COPY: builder.PreferQueue(QueueKind::Copy); break;
		default:
			if (pass.kind == ORG_RG_PASS_COMPUTE) builder.PreferQueue(QueueKind::Compute).AutomaticQueueAssignment();
			else if (pass.kind == ORG_RG_PASS_COPY) builder.PreferQueue(QueueKind::Copy).AutomaticQueueAssignment();
			break;
		}
		if (pass.flags & ORG_RG_PASS_GEOMETRY) builder.IsGeometryPass();
		ContributorBindings bindings;
		bindings.accesses.resize(pass.accesses.size());
		for (size_t i = 0; i < pass.accesses.size(); ++i) {
			const auto& access = pass.accesses[i];
			auto& tokens = bindings.accesses[i];
			const auto& resource = m_source->resources[i];
			const ResourcePtrAndRange target = WholeResource(access.range) ? ResourcePtrAndRange(resource)
				: ResourcePtrAndRange(resource, ToRange(access.range));
			const uint32_t mip = access.range.firstMip, slice = access.range.firstArraySlice;
			const auto take = [&](const DeclaredResourceUse& use) {
				tokens.resource = use.resources.front();
				if (!use.views.empty()) { tokens.view = use.views.front(); tokens.hasView = true; }
				if (use.views.size() > 1) { tokens.clearView = use.views[1]; tokens.hasClearView = true; }
			};
			switch (access.kind) {
			case ORG_RG_ACCESS_SHADER_RESOURCE: take(builder.ShaderResource(target, SrvView{UINT32_MAX, mip, slice})); break;
			case ORG_RG_ACCESS_CONSTANT_BUFFER: take(builder.ConstantBuffer(target)); break;
			case ORG_RG_ACCESS_UNORDERED_ACCESS: take(builder.UnorderedAccess(target, UavView{UINT32_MAX, mip, slice})); break;
			case ORG_RG_ACCESS_UNORDERED_ACCESS_CLEAR: take(builder.UnorderedAccessClear(target, UavView{UINT32_MAX, mip, slice})); break;
			case ORG_RG_ACCESS_RENDER_TARGET: take(builder.RenderTarget(target, RtvView{UINT32_MAX, mip, slice})); break;
			case ORG_RG_ACCESS_RENDER_TARGET_CLEAR: take(builder.RenderTargetClear(target, RtvView{UINT32_MAX, mip, slice})); break;
			case ORG_RG_ACCESS_DEPTH_READ: take(builder.DepthRead(target, DsvView{UINT32_MAX, mip, slice})); break;
			case ORG_RG_ACCESS_DEPTH_READ_WRITE: take(builder.DepthReadWrite(target, DsvView{UINT32_MAX, mip, slice})); break;
			case ORG_RG_ACCESS_DEPTH_STENCIL_CLEAR: take(builder.DepthStencilClear(target, DsvView{UINT32_MAX, mip, slice})); break;
			case ORG_RG_ACCESS_COPY_SOURCE: tokens.resource = builder.CopySource(target); break;
			case ORG_RG_ACCESS_COPY_DESTINATION: tokens.resource = builder.CopyDestination(target); break;
			case ORG_RG_ACCESS_INDIRECT_ARGUMENT: tokens.resource = builder.IndirectArguments(target); break;
			case ORG_RG_ACCESS_INDEX_BUFFER: tokens.resource = builder.IndexBuffer(target); break;
			default: throw std::invalid_argument("Contributor access kind is not supported: " + pass.id);
			}
		}
		return bindings;
	}

	ContributorFrame Prepare(const ContributorBindings& bindings, const PassPrepareContext& preparation) const {
		const auto& source = *m_source;
		const auto& pass = source.pass;
		ContributorFrame data;
		data.source = m_source;
		ORGHostSnapshot snapshot{};
		if (source.host->desc.frame) {
			const auto view = source.host->desc.frame(preparation);
			data.frame = view.frame;
			snapshot = view.snapshot;
		}
		data.frame.structSize = sizeof(ORGFrameInfo);
		data.frame.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
		data.frame.generation = source.lease->generation;
		data.bindings.resize(pass.accesses.size());
		data.resources.resize(pass.accesses.size());
		for (size_t i = 0; i < pass.accesses.size(); ++i) {
			const auto& access = pass.accesses[i];
			const auto& tokens = bindings.accesses[i];
			auto& info = data.bindings[i];
			info.structSize = sizeof(info);
			info.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
			info.binding = access.binding;
			info.access = access.kind;
			info.viewKind = access.viewKind;
			info.viewFormat = access.viewFormat;
			info.descriptorIndex = UINT32_MAX;
			const auto reference = preparation.CaptureResource(tokens.resource);
			data.resources[i] = reference;
			const auto resource = preparation.ResolveCapturedResource(reference);
			info.resource = ConvertHandle<basicrhi_c_resource>(resource.GetHandle());
			const auto& shape = source.shapes[i];
			info.dimension = shape.dimension;
			info.resourceFormat = shape.format;
			info.byteSize = shape.byteSize;
			if (shape.dimension != ORG_RG_RESOURCE_BUFFER) {
				info.width = (std::max)(1u, shape.width >> access.range.firstMip);
				info.height = (std::max)(1u, shape.height >> access.range.firstMip);
				info.depthOrArraySize = shape.depthOrArraySize;
				info.mipLevels = shape.mipLevels;
			}
			if (!tokens.hasView) continue;
			const auto slot = preparation.Resolve(tokens.view);
			switch (access.viewKind) {
			case ORG_RG_VIEW_SHADER_RESOURCE:
			case ORG_RG_VIEW_CONSTANT_BUFFER:
			case ORG_RG_VIEW_UNORDERED_ACCESS:
				info.descriptorIndex = slot.index;
				info.shaderVisible = 1;
				break;
			case ORG_RG_VIEW_RENDER_TARGET: {
				ContributorFrame::Attachment color;
				color.slot = slot;
				color.resource = reference;
				color.clear = access.kind == ORG_RG_ACCESS_RENDER_TARGET_CLEAR;
				color.format = static_cast<rhi::Format>(info.resourceFormat);
				std::copy(std::begin(access.clearValue), std::end(access.clearValue), color.value);
				data.colors.push_back(color);
				data.width = data.width ? (std::min)(data.width, info.width) : info.width;
				data.height = data.height ? (std::min)(data.height, info.height) : info.height;
				break;
			}
			case ORG_RG_VIEW_DEPTH_STENCIL: {
				ContributorFrame::Attachment depth;
				depth.slot = slot;
				depth.resource = reference;
				depth.clear = access.kind == ORG_RG_ACCESS_DEPTH_STENCIL_CLEAR;
				depth.readOnly = access.kind == ORG_RG_ACCESS_DEPTH_READ;
				depth.format = static_cast<rhi::Format>(info.resourceFormat);
				std::copy(std::begin(access.clearValue), std::end(access.clearValue), depth.value);
				data.depth = depth;
				data.width = data.width ? (std::min)(data.width, info.width) : info.width;
				data.height = data.height ? (std::min)(data.height, info.height) : info.height;
				break;
			}
			default: break;
			}
			if (access.kind == ORG_RG_ACCESS_UNORDERED_ACCESS_CLEAR && tokens.hasClearView) {
				ContributorFrame::UavClear clear;
				clear.gpu = slot;
				clear.cpu = preparation.Resolve(tokens.clearView);
				clear.resource = reference;
				clear.integer = info.dimension == ORG_RG_RESOURCE_BUFFER || IntegerFormat(static_cast<rhi::Format>(info.resourceFormat));
				std::copy(std::begin(access.clearValue), std::end(access.clearValue), clear.value);
				data.uavClears.push_back(clear);
			}
		}

		PrepareScope scope;
		auto& context = scope.context;
		context.structSize = sizeof(context);
		context.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
		context.backend = source.backend;
		context.bindingCount = static_cast<uint32_t>(data.bindings.size());
		context.frame = &data.frame;
		context.bindings = data.bindings.data();
		context.snapshot = snapshot;
		const auto& descriptors = this->DescriptorService();
		context.resourceHeap = ConvertHandle<basicrhi_c_descriptor_heap>(descriptors.GetSRVDescriptorHeap().GetHandle());
		context.samplerHeap = ConvertHandle<basicrhi_c_descriptor_heap>(descriptors.GetSamplerDescriptorHeap().GetHandle());
		std::vector<PrepareScope::Upload> uploads;
		scope.source = &source;
		scope.uploads = &uploads;

		auto owner = std::make_shared<PacketOwner>();
		owner->packet.structSize = sizeof(ORGPreparedPass);
		owner->packet.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
		ORGStatus status = ORG_RG_E_CALLBACK_FAILED;
		{
			auto* previous = t_prepareScope;
			t_prepareScope = &scope;
			try { status = pass.prepare(pass.userData, &context, &owner->packet); } catch (...) { status = ORG_RG_E_CALLBACK_FAILED; }
			t_prepareScope = previous;
		}
		if (status != ORG_RG_OK || owner->packet.structSize < sizeof(ORGPreparedPass)
			|| owner->packet.apiVersion != ORG_RENDER_GRAPH_API_CURRENT || !owner->packet.record)
			throw std::runtime_error("Contributor pass failed to prepare a packet: " + pass.id);
		const bool lifecycle = owner->packet.submitted || owner->packet.completed || owner->packet.abandoned;
		if (lifecycle && Reusable)
			throw std::runtime_error("Contributor pass with invocationRevision returned lifecycle callbacks: " + pass.id);
		if (lifecycle) preparation.Reserve(std::make_shared<PacketLifecycle>(owner));
		for (auto& upload : uploads)
			this->UploadBufferData(upload.bytes.data(), upload.bytes.size(), std::move(upload.target), upload.offset);
		data.owner = std::move(owner);
		return data;
	}

	static void Record(const ContributorBindings&, const ContributorFrame& data, PassRecordContext& recording) {
		auto& commands = recording.Commands();
		for (const auto& clear : data.uavClears) {
			rhi::UavClearInfo info{};
			info.shaderVisible = clear.gpu;
			info.cpuVisible = clear.cpu;
			info.resource = recording.Resolve(clear.resource);
			if (clear.integer) {
				rhi::UavClearUint value;
				for (int i = 0; i < 4; ++i) value.v[i] = static_cast<uint32_t>(clear.value[i]);
				commands.ClearUavUint(info, value);
			} else {
				rhi::UavClearFloat value;
				for (int i = 0; i < 4; ++i) value.v[i] = clear.value[i];
				commands.ClearUavFloat(info, value);
			}
		}
		// Bindings as they resolve now (a recording kept for resubmission may record against rebound backings).
		auto bindings = data.bindings;
		for (size_t i = 0; i < bindings.size(); ++i)
			if (data.resources[i]) bindings[i].resource = ConvertHandle<basicrhi_c_resource>(recording.Resolve(*data.resources[i]).GetHandle());
		const bool renderPass = !data.colors.empty() || data.depth;
		std::vector<rhi::ColorAttachment> colors;
		rhi::DepthAttachment depth{};
		if (renderPass) {
			for (const auto& source : data.colors) {
				rhi::ColorAttachment color{};
				color.rtv = source.slot;
				color.loadOp = source.clear ? rhi::LoadOp::Clear : rhi::LoadOp::Load;
				color.clear.type = rhi::ClearValueType::Color;
				color.clear.format = source.format;
				std::copy(std::begin(source.value), std::end(source.value), color.clear.rgba);
				colors.push_back(color);
			}
			rhi::PassBeginInfo begin{};
			begin.colors = {colors.data(), static_cast<uint32_t>(colors.size())};
			if (data.depth) {
				depth.dsv = data.depth->slot;
				depth.depthLoad = data.depth->clear ? rhi::LoadOp::Clear : rhi::LoadOp::Load;
				depth.readOnly = data.depth->readOnly;
				depth.depthStore = data.depth->readOnly ? rhi::StoreOp::DontCare : rhi::StoreOp::Store;
				depth.clear.type = rhi::ClearValueType::DepthStencil;
				depth.clear.format = data.depth->format;
				depth.clear.depthStencil.depth = data.depth->value[0];
				depth.clear.depthStencil.stencil = static_cast<uint8_t>(data.depth->value[1]);
				begin.depth = &depth;
			}
			begin.width = data.width;
			begin.height = data.height;
			commands.BeginPass(begin);
		}
		BorrowState borrow{ &commands, s_nextBorrowToken.fetch_add(1, std::memory_order_relaxed) + 1, true };
		ORGRecordContext context{};
		context.structSize = sizeof(context);
		context.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
		context.bindingCount = static_cast<uint32_t>(bindings.size());
		context.frame = &data.frame;
		context.bindings = bindings.data();
		context.recorder = MakeRecorder(borrow);
		try { data.owner->packet.record(data.owner->packet.data, &context); }
		catch (...) { spdlog::error("Contributor pass {} threw while recording", data.source->pass.id); }
		borrow.active = false;
		borrow.commandList = nullptr;
		if (renderPass) commands.EndPass();
	}

	void InvocationRevision(const PassPrepareContext&, std::vector<uint64_t>& out) const requires Reusable {
		const auto& pass = m_source->pass;
		uint64_t revision = 0;
		if (pass.invocationRevision(pass.userData, &revision) != ORG_RG_OK)
			revision = s_nextBorrowToken.fetch_add(1, std::memory_order_relaxed); // a fresh value: never reused
		out.push_back(revision);
	}

private:
	std::shared_ptr<const PassSource> m_source;
	inline static std::atomic<uint64_t> s_nextBorrowToken{0};
};

// ---------------------------------------------------------------------------------------------------------------------
// Resources and the extension

namespace {

std::shared_ptr<Resource> CreateContributorResource(const std::string& id, const ORGResourceDesc& desc) {
	const bool unordered = (desc.allowedUsages & ORG_RG_USAGE_UNORDERED_ACCESS) != 0;
	if (desc.dimension == ORG_RG_RESOURCE_BUFFER) {
		const auto heap = desc.heapClass == ORG_RG_HEAP_UPLOAD ? rhi::HeapType::Upload
			: desc.heapClass == ORG_RG_HEAP_READBACK ? rhi::HeapType::Readback : rhi::HeapType::DeviceLocal;
		auto buffer = Buffer::CreateSharedUnmaterialized(heap, desc.byteSize, unordered);
		// Whole-buffer views (the registry admits no others): raw without a structure stride, structured with it.
		const bool structured = desc.structureByteStride != 0;
		const uint32_t elements = static_cast<uint32_t>(desc.byteSize / (structured ? desc.structureByteStride : 4u));
		BufferBase::DescriptorRequirements views{};
		views.createSRV = (desc.allowedUsages & ORG_RG_USAGE_SHADER_RESOURCE) != 0;
		views.srvDesc = {.dimension = rhi::SrvDim::Buffer, .formatOverride = structured ? rhi::Format::Unknown : rhi::Format::R32_Typeless,
			.buffer = {.kind = structured ? rhi::BufferViewKind::Structured : rhi::BufferViewKind::Raw, .firstElement = 0,
				.numElements = elements, .structureByteStride = desc.structureByteStride}};
		views.createUAV = unordered;
		views.createNonShaderVisibleUAV = unordered;
		views.uavDesc = {.dimension = rhi::UavDim::Buffer, .formatOverride = structured ? rhi::Format::Unknown : rhi::Format::R32_Typeless,
			.buffer = {.kind = structured ? rhi::BufferViewKind::Structured : rhi::BufferViewKind::Raw, .firstElement = 0,
				.numElements = elements, .structureByteStride = desc.structureByteStride}};
		views.createCBV = (desc.allowedUsages & ORG_RG_USAGE_CONSTANT_BUFFER) != 0;
		views.cbvDesc = {0, static_cast<uint32_t>(desc.byteSize)};
		buffer->SetDescriptorRequirements(views);
		buffer->SetName(id);
		return buffer;
	}
	if (desc.dimension != ORG_RG_RESOURCE_TEXTURE_2D || desc.sampleCount > 1)
		throw std::invalid_argument("Contributor resource kind is not supported (2D textures and buffers): " + id);
	TextureDescription texture;
	texture.format = static_cast<rhi::Format>(desc.format);
	texture.channels = 4;
	texture.arraySize = (std::max)(1u, desc.depthOrArraySize);
	texture.isArray = texture.arraySize > 1;
	texture.hasSRV = (desc.allowedUsages & ORG_RG_USAGE_SHADER_RESOURCE) != 0;
	texture.hasUAV = unordered;
	texture.hasNonShaderVisibleUAV = unordered;
	texture.hasRTV = (desc.allowedUsages & ORG_RG_USAGE_RENDER_TARGET) != 0;
	texture.hasDSV = (desc.allowedUsages & (ORG_RG_USAGE_DEPTH_READ | ORG_RG_USAGE_DEPTH_WRITE)) != 0;
	texture.srvFormat = texture.uavFormat = texture.rtvFormat = texture.dsvFormat = texture.format;
	texture.allowAlias = desc.lifetime == ORG_RG_RESOURCE_TRANSIENT && desc.allowAlias;
	for (uint32_t mip = 0; mip < (std::max)(1u, desc.mipLevels); ++mip)
		texture.imageDimensions.push_back({(std::max)(1u, desc.width >> mip), (std::max)(1u, desc.height >> mip), 0, 0});
	auto pixels = PixelBuffer::CreateSharedUnmaterialized(texture);
	pixels->SetName(id);
	return pixels;
}

bool SameDesc(const ORGResourceDesc& a, const ORGResourceDesc& b) noexcept {
	return a.lifetime == b.lifetime && a.dimension == b.dimension && a.heapClass == b.heapClass && a.format == b.format
		&& a.byteSize == b.byteSize && a.structureByteStride == b.structureByteStride && a.width == b.width && a.height == b.height
		&& a.depthOrArraySize == b.depthOrArraySize && a.mipLevels == b.mipLevels && a.sampleCount == b.sampleCount
		&& a.allowedUsages == b.allowedUsages && a.allowAlias == b.allowAlias;
}

} // namespace

class ContributorExtension final : public IRenderGraphExtension {
public:
	explicit ContributorExtension(std::shared_ptr<ContributorHost::Impl> host) : m_host(std::move(host)) {}

	void GatherStructuralPasses(RenderGraph&, std::vector<ExternalPassDesc>& out) override {
		auto& host = *m_host;
		const auto extents = host.desc.extents ? host.desc.extents() : ContributorHost::Extents{};
		if (!host.current || host.registry.HasPendingRebuild() || (host.hasRelativeResources && !SameExtents(extents, host.builtExtents)))
			Compile(extents);
		for (const auto& source : host.currentPasses) Emit(source, out);
	}

private:
	static bool SameExtents(const ContributorHost::Extents& a, const ContributorHost::Extents& b) noexcept {
		return a.renderWidth == b.renderWidth && a.renderHeight == b.renderHeight
			&& a.outputWidth == b.outputWidth && a.outputHeight == b.outputHeight;
	}

	void Compile(const ContributorHost::Extents& extents) {
		auto& host = *m_host;
		const uint64_t generation = host.nextGeneration++;
		ContributorRegistry::Candidate candidate;
		const auto status = host.registry.Compile(generation, extents.renderWidth, extents.renderHeight,
			extents.outputWidth, extents.outputHeight, candidate);
		if (status != ORG_RG_OK) {
			spdlog::error("Contributors: build {} failed ({}); their passes are left out until one asks to rebuild", generation,
				static_cast<int>(status));
			host.current.reset();
			host.currentPasses.clear();
			return;
		}
		auto lease = std::make_shared<GenerationLease>();
		lease->host = m_host;
		lease->generation = generation;

		std::unordered_map<std::string, std::shared_ptr<Resource>> resources;
		std::unordered_map<std::string, ORGResourceDesc> shapes;
		for (const auto& resource : candidate.resources) shapes[resource.id] = resource.desc;
		for (const auto& source : host.registry.Exports()) {
			ORGResourceDesc shape{};
			shape.dimension = source.desc.dimension;
			shape.format = source.desc.format;
			shape.byteSize = source.desc.byteSize;
			shape.structureByteStride = source.desc.structureByteStride;
			shape.width = source.desc.width;
			shape.height = source.desc.height;
			shape.depthOrArraySize = (std::max)(1u, source.desc.depthOrArraySize);
			shape.mipLevels = (std::max)(1u, source.desc.mipLevels);
			shapes[source.id] = shape;
		}
		host.hasRelativeResources = false;
		std::unordered_map<std::string, std::pair<ORGResourceDesc, std::shared_ptr<Resource>>> kept;
		for (const auto& resource : candidate.resources) {
			try {
				auto& entry = kept[resource.id];
				const auto previous = host.resources.find(resource.id);
				if (previous != host.resources.end() && SameDesc(previous->second.first, resource.desc)) entry = previous->second;
				else entry = {resource.desc, CreateContributorResource(resource.id, resource.desc)};
				resources[resource.id] = entry.second;
			} catch (const std::exception& error) {
				spdlog::error("Contributors: {}", error.what());
			}
		}
		for (const auto& source : host.registry.Exports()) {
			const auto found = std::ranges::find(host.desc.exports, source.id, &ContributorHost::Export::id);
			if (found != host.desc.exports.end() && found->resource)
				if (auto resource = found->resource()) resources[source.id] = std::move(resource);
		}
		host.resources = std::move(kept);

		std::vector<std::shared_ptr<const PassSource>> passes;
		for (const auto& pass : candidate.passes) {
			auto source = std::make_shared<PassSource>();
			source->pass = pass;
			source->lease = lease;
			source->host = m_host.get();
			source->backend = ToORG(host.desc.backend);
			bool complete = true;
			for (const auto& access : pass.accesses) {
				const auto found = resources.find(access.resourceId);
				if (found == resources.end() || !found->second) { complete = false; break; }
				source->resources.push_back(found->second);
				source->shapes.push_back(shapes.at(access.resourceId));
			}
			if (!complete) {
				spdlog::error("Contributors: pass {} left out: a resource it accesses is unavailable", pass.id);
				continue;
			}
			passes.push_back(std::move(source));
		}
		host.hasRelativeResources = host.registry.HasRelativeResources();
		host.builtExtents = extents;
		host.current = lease;
		host.currentPasses = std::move(passes);
		host.activeGeneration.store(generation, std::memory_order_release);
		host.registry.Activate(candidate);
	}

	void Emit(const std::shared_ptr<const PassSource>& source, std::vector<ExternalPassDesc>& out) const {
		auto& host = *m_host;
		const auto& pass = source->pass;
		std::shared_ptr<RenderPass> object = pass.invocationRevision
			? std::static_pointer_cast<RenderPass>(std::make_shared<ContributorPass<true>>(source))
			: std::static_pointer_cast<RenderPass>(std::make_shared<ContributorPass<false>>(source));
		ExternalPassDesc desc = pass.kind == ORG_RG_PASS_COMPUTE ? ExternalPassDesc::Compute(pass.id, object)
			: pass.kind == ORG_RG_PASS_COPY ? ExternalPassDesc::Copy(pass.id, object)
			: ExternalPassDesc::Render(pass.id, object);
		ExternalInsertPoint where;
		where.priority = pass.priority;
		const auto anchorPass = [&](const std::string& id) {
			const auto anchor = std::ranges::find(host.desc.anchors, id, &ContributorHost::Anchor::id);
			return anchor != host.desc.anchors.end() ? anchor->pass : id;
		};
		for (const auto& after : pass.after) where.after.push_back(anchorPass(after));
		for (const auto& before : pass.before) where.before.push_back(anchorPass(before));
		if (!where.after.empty() || !where.before.empty()) desc.where = std::move(where);
		if (!pass.technique.empty()) desc.techniquePath = pass.technique;
		desc.collectStatistics = (pass.flags & ORG_RG_PASS_DISABLE_STATISTICS) == 0;
		desc.isGeometryPass = (pass.flags & ORG_RG_PASS_GEOMETRY) != 0;
		if (!pass.epoch.empty()) {
			const auto epoch = std::ranges::find(host.desc.epochs, pass.epoch, &ContributorHost::Epoch::id);
			if (epoch != host.desc.epochs.end()) desc.epoch = epoch->epoch;
		}
		out.push_back(std::move(desc));
	}

	std::shared_ptr<ContributorHost::Impl> m_host;
};

// ---------------------------------------------------------------------------------------------------------------------
// The C table

namespace {

// Holds the host while a table call runs: unregistration and destruction may race with a contributor's calls.
template<class F>
ORGStatus WithHost(F&& call) noexcept {
	std::shared_ptr<ContributorHost::Impl> host;
	{
		std::scoped_lock lock(g_hostMutex);
		if (g_host) host = g_host->shared_from_this();
	}
	if (!host) return ORG_RG_E_RUNTIME_UNAVAILABLE;
	try { return call(*host); } catch (...) { return ORG_RG_E_INTERNAL; }
}

template<class T>
bool ValidOut(const T* value) noexcept { return value && value->structSize >= sizeof(T); }

template<class T>
ORGStatus CopyIndexed(const std::vector<T>& items, uint32_t index, T* out) noexcept {
	if (!ValidOut(out)) return ORG_RG_E_INVALID_ARGUMENT;
	if (index >= items.size()) return ORG_RG_E_INVALID_ARGUMENT;
	*out = items[index];
	return ORG_RG_OK;
}

ORGStatus ORG_RG_CALL GetRuntimeInfo(ORGRuntimeInfo* out) {
	return WithHost([&](ContributorHost::Impl& host) {
		if (!ValidOut(out)) return ORG_RG_E_INVALID_ARGUMENT;
		*out = {};
		out->structSize = sizeof(*out);
		out->apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
		out->available = 1;
		out->backend = ToORG(host.desc.backend);
		out->capabilities = ORG_RG_CAP_RENDER_PASSES | ORG_RG_CAP_COMPUTE_PASSES | ORG_RG_CAP_COPY_PASSES | ORG_RG_CAP_MANAGED_RESOURCES
			| ORG_RG_CAP_RESOURCE_ALIASING | ORG_RG_CAP_SCHEDULED_UPLOADS | ORG_RG_CAP_SERVICE_QUERY
			| (host.deviceTable ? ORG_RG_CAP_PIPELINES : 0);
		out->activeGeneration = host.activeGeneration.load(std::memory_order_acquire);
		return ORG_RG_OK;
	});
}

ORGStatus ORG_RG_CALL GetHostInfo(ORGHostInfo* out) {
	return WithHost([&](ContributorHost::Impl& host) {
		if (!ValidOut(out)) return ORG_RG_E_INVALID_ARGUMENT;
		*out = {sizeof(*out), ORG_RENDER_GRAPH_API_CURRENT, host.desc.id.c_str(), host.desc.displayName.c_str(), host.desc.version.c_str()};
		return ORG_RG_OK;
	});
}

ORGStatus ORG_RG_CALL GetAnchorCount(uint32_t* out) {
	return WithHost([&](ContributorHost::Impl& host) {
		if (!out) return ORG_RG_E_INVALID_ARGUMENT;
		*out = static_cast<uint32_t>(host.anchors.size());
		return ORG_RG_OK;
	});
}
ORGStatus ORG_RG_CALL GetAnchor(uint32_t index, ORGAnchorDescriptor* out) {
	return WithHost([&](ContributorHost::Impl& host) { return CopyIndexed(host.anchors, index, out); });
}
ORGStatus ORG_RG_CALL GetEpochCount(uint32_t* out) {
	return WithHost([&](ContributorHost::Impl& host) {
		if (!out) return ORG_RG_E_INVALID_ARGUMENT;
		*out = static_cast<uint32_t>(host.epochs.size());
		return ORG_RG_OK;
	});
}
ORGStatus ORG_RG_CALL GetEpoch(uint32_t index, ORGEpochDescriptor* out) {
	return WithHost([&](ContributorHost::Impl& host) { return CopyIndexed(host.epochs, index, out); });
}
ORGStatus ORG_RG_CALL GetExportCount(uint32_t* out) {
	return WithHost([&](ContributorHost::Impl& host) {
		if (!out) return ORG_RG_E_INVALID_ARGUMENT;
		*out = static_cast<uint32_t>(host.exports.size());
		return ORG_RG_OK;
	});
}
ORGStatus ORG_RG_CALL GetExport(uint32_t index, ORGExportDescriptor* out) {
	return WithHost([&](ContributorHost::Impl& host) { return CopyIndexed(host.exports, index, out); });
}
ORGStatus ORG_RG_CALL RegisterContributor(const ORGContributorDesc* desc, ORGRegistrationHandle* out) {
	return WithHost([&](ContributorHost::Impl& host) { return host.registry.Register(desc, out); });
}
ORGStatus ORG_RG_CALL BeginUnregister(ORGRegistrationHandle handle) {
	return WithHost([&](ContributorHost::Impl& host) { return host.registry.BeginUnregister(handle); });
}
ORGStatus ORG_RG_CALL GetRegistrationState(ORGRegistrationHandle handle, ORGRegistrationState* out) {
	return WithHost([&](ContributorHost::Impl& host) { return host.registry.GetRegistrationState(handle, out); });
}
ORGStatus ORG_RG_CALL DeclareResource(ORGBuildHandle build, const ORGResourceDesc* desc) {
	return WithHost([&](ContributorHost::Impl& host) { return host.registry.DeclareResource(build, desc); });
}
ORGStatus ORG_RG_CALL DeclarePass(ORGBuildHandle build, const ORGPassDesc* desc) {
	return WithHost([&](ContributorHost::Impl& host) { return host.registry.DeclarePass(build, desc); });
}
ORGStatus ORG_RG_CALL RequestRebuild(ORGRegistrationHandle handle) {
	return WithHost([&](ContributorHost::Impl& host) { return host.registry.RequestRebuild(handle); });
}
ORGStatus ORG_RG_CALL GetDiagnostic(ORGRegistrationHandle handle, ORGDiagnostic* out) {
	return WithHost([&](ContributorHost::Impl& host) { return host.registry.GetDiagnostic(handle, out); });
}

ORGStatus ORG_RG_CALL QueueBufferUpload(const ORGPrepareContext* context, const ORGBufferUpload* upload) {
	auto* scope = t_prepareScope;
	if (!scope || !context || context != &scope->context) return ORG_RG_E_WRONG_THREAD;
	if (!ValidOut(upload) || upload->apiVersion != ORG_RENDER_GRAPH_API_CURRENT || (!upload->data && upload->dataSize))
		return ORG_RG_E_INVALID_ARGUMENT;
	const auto& accesses = scope->source->pass.accesses;
	const auto found = std::ranges::find(accesses, upload->destination, &ContributorRegistry::Access::binding);
	if (found == accesses.end()) return ORG_RG_E_STALE_HANDLE;
	const auto& resource = scope->source->resources[static_cast<size_t>(found - accesses.begin())];
	if (resource->GetMipLevels() != 1 || !std::dynamic_pointer_cast<Buffer>(resource)) return ORG_RG_E_INCOMPATIBLE_RESOURCE;
	const auto* bytes = static_cast<const std::byte*>(upload->data);
	scope->uploads->push_back({runtime::UploadTarget::FromShared(resource), upload->destinationOffset,
		std::vector<std::byte>(bytes, bytes + upload->dataSize)});
	return ORG_RG_OK;
}

ORGStatus ORG_RG_CALL QueryService(const char* id, uint32_t version, void* out, uint32_t size) {
	return WithHost([&](ContributorHost::Impl& host) {
		if (!id || !out) return ORG_RG_E_INVALID_ARGUMENT;
		if (std::strcmp(id, ORG_PIPELINE_SERVICE_ID) == 0) {
			if (version != ORG_PIPELINE_SERVICE_VERSION_1) return ORG_RG_E_UNSUPPORTED_VERSION;
			if (!host.deviceTable) return ORG_RG_E_UNSUPPORTED_CAPABILITY;
			if (size < sizeof(ORGPipelineService)) return ORG_RG_E_INVALID_ARGUMENT;
			std::memcpy(out, &host.pipelineService, sizeof(ORGPipelineService));
			return ORG_RG_OK;
		}
		return ORG_RG_E_MISSING_DEPENDENCY;
	});
}

const char* ORG_RG_CALL StatusString(ORGStatus status) {
	switch (status) {
	case ORG_RG_OK: return "ok";
	case ORG_RG_E_INVALID_ARGUMENT: return "invalid argument";
	case ORG_RG_E_UNSUPPORTED_VERSION: return "unsupported version";
	case ORG_RG_E_RUNTIME_UNAVAILABLE: return "runtime unavailable";
	case ORG_RG_E_STALE_HANDLE: return "stale handle";
	case ORG_RG_E_DUPLICATE_ID: return "duplicate identifier";
	case ORG_RG_E_RESERVED_ID: return "reserved identifier";
	case ORG_RG_E_MISSING_DEPENDENCY: return "missing dependency";
	case ORG_RG_E_CYCLE: return "dependency cycle";
	case ORG_RG_E_INCOMPATIBLE_RESOURCE: return "incompatible resource";
	case ORG_RG_E_WRONG_THREAD: return "wrong thread";
	case ORG_RG_E_CALLBACK_FAILED: return "callback failed";
	case ORG_RG_E_CLOSED: return "closed";
	case ORG_RG_E_NOT_READY: return "not ready";
	case ORG_RG_E_UNSUPPORTED_CAPABILITY: return "unsupported capability";
	case ORG_RG_E_DUPLICATE_BINDING: return "duplicate binding";
	default: return "internal error";
	}
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------

ContributorHost::ContributorHost(Desc desc) : m_impl(std::make_shared<Impl>()) {
	auto& impl = *m_impl;
	impl.desc = std::move(desc);
	for (const auto& anchor : impl.desc.anchors)
		impl.anchors.push_back({sizeof(ORGAnchorDescriptor), ORG_RENDER_GRAPH_API_CURRENT, anchor.id.c_str(), anchor.displayName.c_str()});
	for (const auto& epoch : impl.desc.epochs)
		impl.epochs.push_back({sizeof(ORGEpochDescriptor), ORG_RENDER_GRAPH_API_CURRENT, epoch.id.c_str(), epoch.displayName.c_str()});
	std::vector<ContributorRegistry::Export> exports;
	for (auto& source : impl.desc.exports) {
		source.desc.structSize = sizeof(ORGExportDescriptor);
		source.desc.apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
		source.desc.id = source.id.c_str();
		source.desc.displayName = source.displayName.c_str();
		impl.exports.push_back(source.desc);
		exports.push_back({source.id, source.desc});
	}
	std::vector<std::string> anchors, epochs;
	for (const auto& anchor : impl.desc.anchors) anchors.push_back(anchor.id);
	for (const auto& epoch : impl.desc.epochs) epochs.push_back(epoch.id);
	impl.registry.SetHostSymbols(std::move(anchors), std::move(epochs), std::move(exports));
	if (impl.desc.device) {
		impl.deviceTable = std::make_unique<DeviceTable>(impl.desc.device);
		impl.pipelineService.structSize = sizeof(ORGPipelineService);
		impl.pipelineService.apiVersion = ORG_PIPELINE_SERVICE_VERSION_1;
		impl.pipelineService.backend = ToORG(impl.desc.backend);
		impl.pipelineService.device = impl.deviceTable->Api();
		impl.pipelineService.graphicsLayout = ConvertHandle<basicrhi_c_pipeline_layout>(impl.desc.graphicsLayout);
		impl.pipelineService.computeLayout = ConvertHandle<basicrhi_c_pipeline_layout>(impl.desc.computeLayout);
	}
	std::scoped_lock lock(g_hostMutex);
	if (g_host) throw std::logic_error("Only one ContributorHost may exist at a time");
	g_host = m_impl.get();
}

ContributorHost::~ContributorHost() {
	{
		std::scoped_lock lock(g_hostMutex);
		if (g_host == m_impl.get()) g_host = nullptr;
	}
	m_impl->current.reset();
	m_impl->currentPasses.clear();
	m_impl->registry.Shutdown();
}

ORGStatus ContributorHost::GetAPI(uint32_t version, ORGRenderGraphAPI* out) noexcept {
	if (!out || out->structSize < sizeof(ORGRenderGraphAPI)) return ORG_RG_E_INVALID_ARGUMENT;
	if (version != ORG_RENDER_GRAPH_API_CURRENT) return ORG_RG_E_UNSUPPORTED_VERSION;
	{
		std::scoped_lock lock(g_hostMutex);
		if (!g_host) return ORG_RG_E_RUNTIME_UNAVAILABLE;
	}
	*out = {};
	out->structSize = sizeof(ORGRenderGraphAPI);
	out->apiVersion = ORG_RENDER_GRAPH_API_CURRENT;
	out->GetRuntimeInfo = GetRuntimeInfo;
	out->GetHostInfo = GetHostInfo;
	out->GetAnchorCount = GetAnchorCount;
	out->GetAnchor = GetAnchor;
	out->GetEpochCount = GetEpochCount;
	out->GetEpoch = GetEpoch;
	out->GetExportCount = GetExportCount;
	out->GetExport = GetExport;
	out->RegisterContributor = RegisterContributor;
	out->BeginUnregister = BeginUnregister;
	out->GetRegistrationState = GetRegistrationState;
	out->DeclareResource = DeclareResource;
	out->DeclarePass = DeclarePass;
	out->RequestRebuild = RequestRebuild;
	out->GetDiagnostic = GetDiagnostic;
	out->QueueBufferUpload = QueueBufferUpload;
	out->QueryService = QueryService;
	out->StatusString = StatusString;
	return ORG_RG_OK;
}

std::unique_ptr<IRenderGraphExtension> ContributorHost::CreateExtension() {
	return std::make_unique<ContributorExtension>(m_impl);
}

void ContributorHost::Poll(GraphHost& host) {
	auto& impl = *m_impl;
	bool rebuild = impl.registry.HasPendingRebuild();
	if (!rebuild && impl.hasRelativeResources && impl.desc.extents) {
		const auto extents = impl.desc.extents();
		rebuild = extents.renderWidth != impl.builtExtents.renderWidth || extents.renderHeight != impl.builtExtents.renderHeight
			|| extents.outputWidth != impl.builtExtents.outputWidth || extents.outputHeight != impl.builtExtents.outputHeight;
	}
	if (rebuild) host.RequestRebuild();
}

ContributorRegistry& ContributorHost::Registry() noexcept { return m_impl->registry; }

} // namespace org::contributor
