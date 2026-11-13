// Resolves a multisampled texture into a single-sampled one, for ResolveMultisamplePass.
//
// Each destination texel is the average of its source texel's samples. That is what a hardware resolve does for float,
// UNORM and SNORM formats. An sRGB source view decodes to linear, so the average is in linear space. An sRGB
// destination is written through a UNORM view, so the shader encodes before it stores.

#include "ColorSpace.hlsli"

cbuffer ResolveConstants : register(b0)
{
	uint sourceIndex;      // The multisampled source's shader-resource view in the resource descriptor heap.
	uint destinationIndex; // The destination's unordered-access view.
	uint width;
	uint height;
	uint sampleCount;
	uint flags;            // Bit 0: the destination is sRGB.
};

static const uint kSrgbDestination = 1u;

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= width || id.y >= height) return;
	Texture2DMS<float4> source = ResourceDescriptorHeap[sourceIndex];
	RWTexture2D<float4> destination = ResourceDescriptorHeap[destinationIndex];
	float4 sum = 0;
	for (uint i = 0; i < sampleCount; ++i) sum += source.Load(int2(id.xy), i);
	sum /= float(sampleCount);
	if (flags & kSrgbDestination) sum.rgb = OrgLinearToSrgb(saturate(sum.rgb));
	destination[id.xy] = sum;
}
