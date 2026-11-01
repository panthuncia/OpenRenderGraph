#include "Render/RenderGraph/ExternalEpochExecution.h"

#include "Render/RenderGraph/RhiExecution.h"

#include <stdexcept>

namespace org {

void ExternalEpochExecution::Execute(size_t a_segment, rhi::CommandList a_primary) const {
	BT_ZONE_SCOPE("ORG.ExternalEpoch.Execute");
	if (a_segment >= segments.size()) throw std::out_of_range("An external epoch's execution has no such segment");
	size_t expected = a_segment;
	if (!m_executed.compare_exchange_strong(expected, a_segment + 1, std::memory_order_acq_rel))
		throw std::logic_error("An external epoch's segments are executed once each, in order");
	const auto execute = [&](std::span<rhi::CommandList> a_lists, rhi::DescriptorHeapHandle a_resources, rhi::DescriptorHeapHandle a_samplers) {
		if (a_lists.empty()) return;
		if (a_resources.valid())
			a_primary.SetDescriptorHeaps(a_resources, a_samplers.valid() ? std::optional<rhi::DescriptorHeapHandle>{a_samplers} : std::nullopt);
		std::vector<rhi::CommandList*> pointers;
		pointers.reserve(a_lists.size());
		for (auto& list : a_lists) pointers.push_back(&list);
		a_primary.ExecuteCommands({pointers.data(), static_cast<uint32_t>(pointers.size())});
	};
	const auto& segment = segments[a_segment];
	auto leadingLists = segment.leading;
	execute(leadingLists, segment.leadingResourceHeap, segment.leadingSamplerHeap);
	for (const auto& item : segment.items) {
		if (!item.packet) continue;
		if (item.packet->Deferred()) {
			BT_ZONE_SCOPE("ORG.ExternalEpoch.RecordDeferred");
			item.packet->RecordDeferred(a_primary);
			continue;
		}
		execute(item.packet->Lists(), item.resourceHeap, item.samplerHeap);
	}
}

}  // namespace org
