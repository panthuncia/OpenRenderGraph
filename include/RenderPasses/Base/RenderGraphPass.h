#pragma once

#include <memory>
#include <source_location>

#include "Interfaces/IResourceProvider.h"
#include "Render/PassInputs.h"
#include "Render/PreparedPass.h"
#include "Render/PassExecutionContext.h"

namespace org {

namespace runtime {
class IUploadService;
class IDescriptorService;
struct UploadTarget;
}

class ResourceRegistryView;
class RenderPassBuilder;
struct PassParameters;

// Queue-agnostic pass lifecycle. Queue eligibility is stored in the pass
// declaration and is not encoded by the pass object's C++ type.
class RenderGraphPass : public IResourceProvider, public RenderGraphPassBase {
public:
    virtual ~RenderGraphPass() = default;
    // Nonvirtual to preserve contributor vtables. The graph installs its exact
    // service generation before Setup; accepted frames retain the same owners.
    void ConfigureRuntimeServices(
        std::shared_ptr<runtime::IUploadService> uploads,
        std::shared_ptr<runtime::IDescriptorService> descriptors) noexcept {
        m_uploadService = std::move(uploads);
        m_descriptorService = std::move(descriptors);
    }
    virtual void ConfigureResourceRegistryView(
        std::shared_ptr<ResourceRegistryView> view,
        const PassParameters& parameters) = 0;
    virtual void Setup() = 0;
    virtual PreparedPass PrepareFrame(FramePreparationContext&) { return {}; }
    virtual void Update(const UpdateExecutionContext&) {}
    virtual PassReturn Execute(PassExecutionContext&) { return {}; }
    virtual void Cleanup() = 0;
    virtual void Invalidate() = 0;
    virtual bool IsInvalidated() const = 0;
    // Unified declarations deliberately use the queue-agnostic builder. Legacy
    // compute/copy passes continue through their old ingestion paths until
    // migrated, while typed passes always support this entry point.
    virtual bool SupportsUnifiedDeclaration() const { return false; }
    virtual void DeclareUnified(RenderPassBuilder&) {}
    // Typed preparation is the complete per-frame execution contract. Such a
    // pass must never also be captured through the legacy immediate replay
    // adapter, even if an intermediate class still implements that interface.
    virtual bool UsesTypedPreparation() const noexcept { return false; }
protected:
    runtime::IUploadService& UploadService() const;
    runtime::IDescriptorService& DescriptorService() const;
    void UploadBufferData(const void* data, size_t size, runtime::UploadTarget target,
        size_t offset, std::source_location source = std::source_location::current()) const;
private:
    std::weak_ptr<runtime::IUploadService> m_uploadService;
    std::weak_ptr<runtime::IDescriptorService> m_descriptorService;
};

} // namespace org
