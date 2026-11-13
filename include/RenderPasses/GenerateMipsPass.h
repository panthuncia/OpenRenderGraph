#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "RenderPasses/Base/TypedRenderGraphPass.h"
#include "Render/PassBuilders.h"
#include "Render/PipelineDesc.h"
#include "Resources/Resource.h"

namespace org {

/// @brief Options of GenerateMipsPass.
struct GenerateMipsOptions {
	/// The shader, shaders/GenerateMips.hlsl. A host that keeps it elsewhere names its own path.
	PipelineShaderSource shader{std::wstring(kBuiltinShaderPrefix) + L"GenerateMips.hlsl", L"main", {}};
};

/// @brief The size of one mip step: the source mip and the mip it builds.
struct GenerateMipsStep {
	uint32_t sourceWidth = 0, sourceHeight = 0;
	uint32_t destinationWidth = 0, destinationHeight = 0;
};

/// @brief The steps that build every mip of a texture from mip 0. Mip sizes halve and round down, and stop at 1.
std::vector<GenerateMipsStep> PlanGenerateMips(uint32_t width, uint32_t height, uint32_t mipLevels);

struct GenerateMipsBindings {
	PipelineState pipeline;
	std::vector<DeclaredViewToken> mips; ///< An unordered-access view of each mip, all slices.
	uint32_t slices = 1;
};

struct GenerateMipsFrameData {
	std::vector<GenerateMipsStep> steps;
	uint32_t flags = 0;
};

/// @brief Builds mips 1 to N-1 of a texture from mip 0 with a compute shader, one dispatch per mip.
///
/// The texture is a 2D texture, a texture array or a cube map, and needs unordered access on every mip. Each mip is the
/// area-weighted average of the mip above it.
///
/// Formats:
/// - Float, UNORM and SNORM formats work when their unordered-access views support typed loads.
/// - An sRGB texture is averaged in linear space. Its unordered-access views must be the UNORM format, so the texture must
///   be created with that format castable. On Vulkan that needs BasicRHI to honour ResourceDesc::castableFormats.
/// - Integer and block-compressed formats are not supported.
class GenerateMipsPass final : public TypedRenderGraphPass<GenerateMipsPass, GenerateMipsFrameData, GenerateMipsBindings> {
public:
	/// @param texture The texture whose mips to build. Its mip 0 must be written before this pass.
	explicit GenerateMipsPass(std::shared_ptr<Resource> texture, GenerateMipsOptions options = {});

	GenerateMipsBindings Declare(PassBuilder& builder);
	GenerateMipsFrameData Prepare(const GenerateMipsBindings& bindings, const PassPrepareContext& preparation);
	static void Record(const GenerateMipsBindings& bindings, const GenerateMipsFrameData& data, PassRecordContext& recording);

private:
	std::shared_ptr<Resource> m_texture;
	GenerateMipsOptions m_options;
};

} // namespace org
