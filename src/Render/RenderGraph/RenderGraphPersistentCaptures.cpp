// Readback captures on the persistent route. A capture anchored after a main pass is recorded in that pass's own packet,
// right after its commands (PreparedPass::WithTrailer): a transition from the pass's final state to copy source, the copy into
// a readback buffer, and the transition back, so the executable and its barriers are unchanged. A capture the pass cannot
// take (a resource it does not bind, or a range its uses do not cover) falls back to a frame-interrupting capture pass, and an
// after-graph capture to a per-frame pass at the end.

#include "RenderGraphPersistentState.h"

#include "RenderPasses/ReadbackCopyCapturePass.h"
#include "Render/Runtime/ReadbackCapturePlan.h"

namespace org {

using namespace persistent_impl;

namespace {

// Anchored captures whose pass is not prepared for this many frames are dropped: the pass is gone or never runs.
constexpr uint64_t kCaptureAnchorFrames = 64;

std::shared_ptr<Resource> ResolveCaptureResource(RenderGraph& graph, const runtime::ReadbackCaptureInfo& capture) {
	if (capture.scheduledID) return graph.GetScheduledResource(capture.resourceId);
	auto resource = capture.resource.lock();
	if (!resource && capture.resourceId) resource = graph.GetResourceByID(capture.resourceId);
	return resource;
}

std::optional<ExternalPassDesc> MakeCapturePass(RenderGraph& graph, runtime::ReadbackCaptureInfo& capture,
	const std::shared_ptr<runtime::IReadbackService>& service, uint32_t slot) {
	auto queue = capture.preferredQueueKind;
	if (queue != QueueKind::Graphics && queue != QueueKind::Copy) queue = QueueKind::Graphics;
	auto resource = ResolveCaptureResource(graph, capture);
	if (!resource) {
		spdlog::warn("Readback capture after '{}': resource {} is gone", capture.passName, capture.resourceId);
		return std::nullopt;
	}
	// A resolver's backing may not be a named graph resource; the capture pass declares it like any other.
	const auto handle = graph.RequestResourceHandle(resource.get(), /*allowFailure=*/true);
	if (handle.GetGeneration() == 0) {
		spdlog::warn("Readback capture after '{}': resource {} has no graph handle", capture.passName, capture.resourceId);
		return std::nullopt;
	}
	const bool afterGraph = capture.passName == runtime::kReadbackAfterGraph;
	const auto name = "ReadbackCapture::" + capture.passName + (queue == QueueKind::Copy ? "::Copy::Slot" : "::Graphics::Slot")
		+ std::to_string(slot);
	ReadbackCaptureInputs inputs{};
	inputs.target = ResourceHandleAndRange(handle, capture.range);
	auto desc = queue == QueueKind::Copy
		? ExternalPassDesc::Copy(name, std::make_shared<ReadbackCopyCapturePass>(inputs, resource, std::move(capture.callback), service, name))
		: ExternalPassDesc::Render(name, std::make_shared<ReadbackCapturePass>(inputs, resource, std::move(capture.callback), service, name))
			.PinToQueue(static_cast<QueueSlotIndex>(0));
	return std::move(desc)
		.At(afterGraph ? ExternalInsertPoint::End() : ExternalInsertPoint::After(capture.passName))
		.InterruptsFrame(!afterGraph)
		.CollectStatistics(false)
		.RegisterByName(false);
}

bool Covers(const std::vector<const compiler::CompileStateUse*>& uses, const SubresourceRange& range) {
	for (uint32_t slice = range.firstSlice; slice < range.firstSlice + range.sliceCount; ++slice)
		for (uint32_t mip = range.firstMip; mip < range.firstMip + range.mipCount; ++mip) {
			bool covered = false;
			for (const auto* use : uses)
				covered = covered || (mip >= use->range.mip && mip < use->range.mip + use->range.mips
					&& slice >= use->range.slice && slice < use->range.slice + use->range.slices);
			if (!covered) return false;
		}
	return true;
}

} // namespace

std::shared_ptr<Resource> RenderGraph::GetScheduledResource(uint64_t schedulingID) const {
	if (!m_compilerState || !m_compilerState->persistent) return nullptr;
	for (const auto& entry : m_compilerState->persistent->entries) {
		if (entry.retired || entry.resourceID != schedulingID) continue;
		if (entry.member) return entry.member;
		if (entry.resource) return entry.resource->shared_from_this();
	}
	return nullptr;
}

namespace persistent_impl {

void PlacePersistentCaptures(RenderGraph& graph, State& state, bool ticket, std::vector<ExternalPassDesc>& frameExt,
	std::vector<ExternalPassDesc>& interrupting) {
	const auto service = graph.GetReadbackServiceOwner();
	if (!service) return;
	std::unordered_map<std::string, uint32_t> slots;
	const auto insert = [&](runtime::ReadbackCaptureInfo& capture, std::vector<ExternalPassDesc>& out) {
		if (ticket) {
			spdlog::warn("Readback capture after '{}' needs an inserted pass, which tickets do not run; dropped", capture.passName);
			return;
		}
		if (auto desc = MakeCapturePass(graph, capture, service, slots[capture.passName]++)) out.push_back(std::move(*desc));
	};
	for (auto& capture : std::exchange(state.capturesToInsert, {})) insert(capture, interrupting);
	for (auto& capture : service->ConsumeCaptureRequests()) {
		if (capture.passName == runtime::kReadbackAfterGraph) insert(capture, frameExt);
		else state.pendingCaptures[capture.passName].push_back({std::move(capture), state.frameNumber});
	}
	for (auto it = state.pendingCaptures.begin(); it != state.pendingCaptures.end();) {
		auto& captures = it->second;
		std::erase_if(captures, [&](const State::PendingCapture& pending) {
			if (state.frameNumber - pending.requestedFrame < kCaptureAnchorFrames) return false;
			spdlog::warn("Readback capture after '{}': no such pass prepared for {} frames; dropped", it->first, kCaptureAnchorFrames);
			return true;
		});
		it = captures.empty() ? state.pendingCaptures.erase(it) : std::next(it);
	}
}

PreparedPass AppendPersistentCaptures(RenderGraph& graph, State& state, const State::MainPass& pass, const std::string& name,
	PreparedPass packet, const FramePreparationContext& preparation, const persistent::SelectedPublication& publication) {
	const auto found = state.pendingCaptures.find(name);
	if (found == state.pendingCaptures.end()) return packet;
	auto captures = std::move(found->second);
	state.pendingCaptures.erase(found);
	const auto service = graph.GetReadbackServiceOwner();
	const auto& executable = *publication.executable;
	const auto& structure = *executable.graph->structure;
	const compiler::CompilePass* compiled = nullptr;
	for (const auto& candidate : structure.passes)
		if (candidate.preparedPassIndex == pass.id.index) compiled = &candidate;

	struct Capture {
		rhi::ResourceHandle source;
		runtime::ReadbackCapturePlan plan;
	};
	auto copies = std::make_shared<std::vector<Capture>>();
	auto textureBarriers = std::make_shared<std::pair<std::vector<rhi::TextureBarrier>, std::vector<rhi::TextureBarrier>>>();
	auto bufferBarriers = std::make_shared<std::pair<std::vector<rhi::BufferBarrier>, std::vector<rhi::BufferBarrier>>>();
	std::vector<std::shared_ptr<const PreparedLifecycleEffect>> effects;
	auto leases = std::make_shared<std::vector<std::shared_ptr<const void>>>();
	std::vector<uint32_t> transitioned; // resources whose barriers are planned (two captures of one resource)
	for (auto& pending : captures) {
		auto& capture = pending.info;
		auto resource = ResolveCaptureResource(graph, capture);
		if (!resource || !service || !compiled || !preparation.resourceSlots || !preparation.bindings) {
			state.capturesToInsert.push_back(std::move(capture));
			continue;
		}
		const auto slot = preparation.resourceSlots->Find(resource->GetGlobalResourceID());
		const auto resourceIndex = slot != preparation.resourceSlots->end() && slot->second < executable.resourceIndexBySlot.size()
			? executable.resourceIndexBySlot[slot->second] : UINT32_MAX;
		// The pass's final states of the resource: its declared exit states, else the states it runs in.
		std::vector<const compiler::CompileStateUse*> uses;
		for (const auto& use : compiled->exitStates) if (use.resource == resourceIndex) uses.push_back(&use);
		if (uses.empty())
			for (const auto& use : compiled->entryStates) if (use.resource == resourceIndex) uses.push_back(&use);
		const auto range = ResolveRangeSpec(capture.range, resource->GetMipLevels(), resource->GetArraySize());
		if (uses.empty() || (resource->HasLayout() && !Covers(uses, range))) {
			state.capturesToInsert.push_back(std::move(capture));
			continue;
		}
		const auto bound = preparation.bindings->Resolve(PreparedResourceReference{slot->second});
		auto plan = runtime::PlanReadbackCapture(preparation.device, service, *resource, bound, capture.range,
			std::move(capture.callback), QueueKind::Graphics, "ReadbackCapture::" + name);
		if (!plan) continue;
		const auto handle = bound.GetHandle();
		if (std::find(transitioned.begin(), transitioned.end(), resourceIndex) != transitioned.end()) uses.clear();
		else transitioned.push_back(resourceIndex);
		for (const auto* use : uses) {
			const auto access = static_cast<rhi::ResourceAccessType>(use->state.access);
			const auto layout = static_cast<rhi::ResourceLayout>(use->state.layout);
			const auto sync = static_cast<rhi::ResourceSyncState>(use->state.sync);
			if (resource->HasLayout()) {
				if (layout == rhi::ResourceLayout::CopySource) continue;
				rhi::TextureBarrier toCopy{};
				toCopy.texture = handle;
				toCopy.range = {use->range.mip, use->range.mips, use->range.slice, use->range.slices};
				toCopy.beforeSync = sync; toCopy.beforeAccess = access; toCopy.beforeLayout = layout;
				toCopy.afterSync = rhi::ResourceSyncState::Copy; toCopy.afterAccess = rhi::ResourceAccessType::CopySource;
				toCopy.afterLayout = rhi::ResourceLayout::CopySource;
				auto back = toCopy;
				std::swap(back.beforeSync, back.afterSync); std::swap(back.beforeAccess, back.afterAccess);
				std::swap(back.beforeLayout, back.afterLayout);
				textureBarriers->first.push_back(toCopy);
				textureBarriers->second.push_back(back);
			} else {
				if (access == rhi::ResourceAccessType::CopySource) continue;
				rhi::BufferBarrier toCopy{};
				toCopy.buffer = handle;
				toCopy.beforeSync = sync; toCopy.beforeAccess = access;
				toCopy.afterSync = rhi::ResourceSyncState::Copy; toCopy.afterAccess = rhi::ResourceAccessType::CopySource;
				auto back = toCopy;
				std::swap(back.beforeSync, back.afterSync); std::swap(back.beforeAccess, back.afterAccess);
				bufferBarriers->first.push_back(toCopy);
				bufferBarriers->second.push_back(back);
			}
		}
		effects.push_back(std::move(plan.reservation));
		leases->push_back(std::move(plan.destinationLease));
		copies->push_back({handle, std::move(plan)});
	}
	if (copies->empty()) return packet;
	auto trailer = [copies, textureBarriers, bufferBarriers](RecordingContext& context) {
		auto& commands = context.Commands();
		const auto barriers = [&](std::vector<rhi::TextureBarrier>& textures, std::vector<rhi::BufferBarrier>& buffers) {
			if (textures.empty() && buffers.empty()) return;
			rhi::BarrierBatch batch{};
			batch.textures = {textures.data(), static_cast<uint32_t>(textures.size())};
			batch.buffers = {buffers.data(), static_cast<uint32_t>(buffers.size())};
			commands.Barriers(batch);
		};
		barriers(textureBarriers->first, bufferBarriers->first);
		for (const auto& copy : *copies) runtime::RecordReadbackCapture(copy.plan, commands, copy.source);
		barriers(textureBarriers->second, bufferBarriers->second);
	};
	return PreparedPass::WithTrailer(std::move(packet), std::move(trailer), std::move(effects), std::move(leases));
}

} // namespace persistent_impl

} // namespace org
