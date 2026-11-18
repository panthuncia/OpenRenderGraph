#include "Render/PipelinePermutations.h"

#include <mutex>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace org {

PipelinePermutations::PipelinePermutations(std::shared_ptr<PipelineLibrary> library, std::string name)
	: m_library(std::move(library)), m_name(std::move(name)) {
	if (!m_library) throw std::invalid_argument("PipelinePermutations needs a PipelineLibrary");
}

PipelineLibraryEntryPtr PipelinePermutations::Find(uint64_t key) const {
	std::shared_lock lock(m_mutex);
	const auto found = m_entries.find(key);
	return found != m_entries.end() ? found->second : nullptr;
}

PipelineLibraryEntryPtr PipelinePermutations::Insert(uint64_t key, PipelineDesc desc) {
	// Acquire outside the lock. Another thread may insert the key meanwhile, and then its entry wins.
	auto entry = m_library->Acquire(std::move(desc));
	std::unique_lock lock(m_mutex);
	return m_entries.try_emplace(key, std::move(entry)).first->second;
}

void PipelinePermutations::Prewarm(uint64_t key, DeviceInstanceId device, PipelineDesc desc) {
	auto entry = Find(key);
	if (!entry) entry = Insert(key, std::move(desc));
	m_library->Request(entry, device);
}

PipelineSelection PipelinePermutations::Select(uint64_t key, const PipelineLibraryEntryPtr& entry, DeviceInstanceId device,
	PipelineMiss miss) {
	m_library->Request(entry, device);
	PipelineSelection selection;
	selection.status = miss == PipelineMiss::Wait ? m_library->Wait(entry, device) : entry->Status(device);
	// A build stores its payload before it publishes Ready.
	if (selection.status == PipelineBuildStatus::Ready) selection.payload = entry->State().GetPayload(device);
	else if (selection.status == PipelineBuildStatus::Failed) {
		std::unique_lock lock(m_mutex);
		if (m_reportedFailures.insert(key).second)
			spdlog::error("Pipeline '{}' of '{}' (key {:#x}) failed to build on device {}: {}", entry->Name(), m_name, key,
				static_cast<uint8_t>(device), entry->Error(device));
	}
	return selection;
}

size_t PipelinePermutations::Size() const {
	std::shared_lock lock(m_mutex);
	return m_entries.size();
}

void PipelinePermutations::Clear() {
	std::unique_lock lock(m_mutex);
	m_entries.clear();
	m_reportedFailures.clear();
}

} // namespace org
