#pragma once

#include <memory>

#include "Interfaces/IResourceProvider.h"
#include "Render/PassInputs.h"
#include "Render/PreparedPass.h"
#include "Render/PassExecutionContext.h"

namespace org {

class ResourceRegistryView;
class RenderPassBuilder;
struct PassParameters;

// Queue-agnostic pass lifecycle. Queue eligibility is stored in the pass
// declaration and is not encoded by the pass object's C++ type.
class RenderGraphPass : public IResourceProvider, public RenderGraphPassBase {
public:
    virtual ~RenderGraphPass() = default;
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
};

} // namespace org
