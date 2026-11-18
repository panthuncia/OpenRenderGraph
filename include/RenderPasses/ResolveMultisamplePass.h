#pragma once

#include <cstdint>
#include <memory>

#include "RenderPasses/Base/TypedRenderGraphPass.h"
#include "Render/PassBuilders.h"
#include "Render/PipelineDesc.h"
#include "Resources/Resource.h"

namespace org {

/// @brief Options of ResolveMultisamplePass.
struct ResolveMultisampleOptions {
	/// The shader, shaders/ResolveMultisample.hlsl. A host that keeps it elsewhere names its own path.
	PipelineShaderSource shader{std::wstring(kBuiltinShaderPrefix) + L"ResolveMultisample.hlsl", L"main", {}};
	uint32_t destinationMip = 0;
};

struct ResolveMultisampleBindings {
	PipelineState pipeline;
	DeclaredViewToken source;      ///< The multisampled texture's shader-resource view.
	DeclaredViewToken destination; ///< The destination mip's unordered-access view.
};

struct ResolveMultisampleFrameData {
	uint32_t width = 0, height = 0, samples = 0, flags = 0;
};

/// @brief Resolves a multisampled 2D texture into a single-sampled one, averaging each texel's samples.
///
/// BasicRHI has no resolve command, so this is a compute pass. It matches a hardware resolve for float, UNORM and SNORM
/// formats. The source needs a shader-resource view, and the destination needs unordered access. The resolved area is the
/// smaller of the two sizes.
///
/// For an sRGB source, the average is in linear space. An sRGB destination must have a UNORM unordered-access view, as
/// for GenerateMipsPass.
class ResolveMultisamplePass final
	: public TypedRenderGraphPass<ResolveMultisamplePass, ResolveMultisampleFrameData, ResolveMultisampleBindings> {
public:
	ResolveMultisamplePass(std::shared_ptr<Resource> source, std::shared_ptr<Resource> destination,
		ResolveMultisampleOptions options = {});

	ResolveMultisampleBindings Declare(PassBuilder& builder);
	ResolveMultisampleFrameData Prepare(const ResolveMultisampleBindings& bindings, const PassPrepareContext& preparation);
	static void Record(const ResolveMultisampleBindings& bindings, const ResolveMultisampleFrameData& data,
		PassRecordContext& recording);

private:
	std::shared_ptr<Resource> m_source;
	std::shared_ptr<Resource> m_destination;
	ResolveMultisampleOptions m_options;
};

} // namespace org
