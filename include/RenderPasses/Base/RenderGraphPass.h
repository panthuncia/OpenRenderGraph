#pragma once

#include <functional>
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
std::shared_ptr<const ResourceUseLayout> GetResourceUses(const RenderPassBuilder& builder);
struct PassParameters;

// Queue-agnostic pass lifecycle. Queue eligibility is stored in the pass
// declaration and is not encoded by the pass object's C++ type.
class RenderGraphPass : public IResourceProvider, public RenderGraphPassBase {
public:
    ~RenderGraphPass() override;
    std::shared_ptr<const ResourceUseLayout> ResourceUses() const { return m_resourceUses; }
    void SetResourceUses(std::shared_ptr<const ResourceUseLayout> uses) { m_resourceUses = std::move(uses); }
    // A declared pass's typed bindings (TypedRenderGraphPass), from the same declaration as ResourceUses. They are
    // type-erased, so a graph can keep the declaration that a compiled executable was built from. Null for undeclared
    // passes.
    std::shared_ptr<const void> DeclaredBindings() const { return m_declaredBindingsSnapshot; }
    void SetDeclaredBindings(std::shared_ptr<const void> bindings) { m_declaredBindingsSnapshot = std::move(bindings); }

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
    // For async epochs. Computes a hash of what the pass's recording depends on beyond its bindings and the
    // frame's resources. For a typed pass, this is its Revision. With it, the submitting thread can check an
    // invocation prepared ahead of its execution against what the pass would prepare now, and reuse a kept
    // recording. Safe to call at the same time as the pass's own preparation. Returns false when the pass cannot
    // tell, because its recording may depend on anything. A prepared execution that holds it is then never current.
    virtual bool RevisionHash(uint64_t&) const { return false; }
    // The values the pass's Revision and Prepare read that it takes from the frame's host data (a render size, a setting the
    // frame carries). The graph calls it before Update in every run, and a staged recording (GraphHost::RecordStagedCopies) calls
    // it alone, on the thread that records, with the host data the application gives for the build. So it must have no side
    // effect beyond the pass's own members: no upload, no state another pass or a manager reads. Update keeps those.
    virtual void TakeFrameValues(const UpdateExecutionContext&) {}
    // A host that turns the pass off for some executions by its host data (a frame description passed as the preparation's
    // host data). At each preparation the graph asks the predicate with that host data. True means the pass does nothing in
    // the execution: its recorder records only the barriers around it, as for a pass recorded at execution that skips. The
    // predicate must give the same answer for the same host data, because a kept recording keeps the answer. Set it before
    // the pass's first preparation. Without one, the pass is never skipped this way.
    void SetSkipWhen(std::function<bool(const IHostExecutionData*)> predicate) { m_skipWhen = std::move(predicate); }
    // The predicate's answer for host data (false without a predicate).
    bool SkipsFor(const IHostExecutionData* hostData) const { return m_skipWhen && m_skipWhen(hostData); }
protected:
    runtime::IUploadService& UploadService() const;
    runtime::IDescriptorService& DescriptorService() const;
    void UploadBufferData(const void* data, size_t size, runtime::UploadTarget target,
        size_t offset, std::source_location source = std::source_location::current()) const;
private:
    std::shared_ptr<const ResourceUseLayout> m_resourceUses;
    std::shared_ptr<const void> m_declaredBindingsSnapshot;
    std::weak_ptr<runtime::IUploadService> m_uploadService;
    std::weak_ptr<runtime::IDescriptorService> m_descriptorService;
    std::function<bool(const IHostExecutionData*)> m_skipWhen;
};

} // namespace org
