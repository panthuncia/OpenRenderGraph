#include "RenderPasses/Base/RenderPass.h"

#include "Render/Runtime/IDescriptorService.h"
#include "Render/Runtime/IUploadService.h"
#include "Render/Runtime/DescriptorServiceAccess.h"

namespace org {

RenderGraphPass::~RenderGraphPass() = default;

runtime::IUploadService& RenderGraphPass::UploadService() const {
    auto service = m_uploadService.lock();
    if (!service) throw std::runtime_error("Pass upload service generation is unavailable");
    return *service;
}

runtime::IDescriptorService& RenderGraphPass::DescriptorService() const {
    auto service = m_descriptorService.lock();
    if (service) return *service;
    // During startup, a pass can be constructed before the graph binds its services. The pass still holds only a weak
    // binding. In that startup window, resolve through the graph's active generation instead.
    if (auto* active = runtime::GetActiveDescriptorService()) return *active;
    throw std::runtime_error("Pass descriptor service generation is unavailable");
}

void RenderGraphPass::UploadBufferData(const void* data, size_t size,
    runtime::UploadTarget target, size_t offset, std::source_location source) const {
    UploadService().UploadData(data, size, std::move(target), offset,
        source.file_name(), static_cast<int>(source.line()));
}

RenderPass::~RenderPass() = default;

void RenderPass::ConfigureResourceRegistryView(
	std::shared_ptr<ResourceRegistryView> view,
	const PassParameters&) {
	SetResourceRegistryView(std::move(view));
}

} // namespace org
