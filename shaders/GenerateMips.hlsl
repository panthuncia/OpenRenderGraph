// Builds one mip of a texture from the mip above it, for GenerateMipsPass.
//
// Each destination texel is the area-weighted average of the source texels it covers. For an even source size that is
// the 2x2 box. For an odd size the footprint spans 3 texels, and the outer two count partly.
//
// Both mips are read and written through unordered-access views, so one declaration covers the whole chain. An sRGB
// texture is viewed as UNORM there: the shader decodes on load, averages in linear space, and encodes on store.
//
// ORG_GENERATE_MIPS_ARRAY: the views are 2D arrays (texture arrays and cube maps). Z is the slice.

#include "ColorSpace.hlsli"

cbuffer GenerateMipsConstants : register(b0)
{
	uint sourceIndex;      // The source mip's view in the resource descriptor heap.
	uint destinationIndex; // The destination mip's view.
	uint sourceWidth;
	uint sourceHeight;
	uint destinationWidth;
	uint destinationHeight;
	uint flags;            // Bit 0: the texture is sRGB.
};

static const uint kSrgb = 1u;

#ifdef ORG_GENERATE_MIPS_ARRAY
#define MIP_TEXTURE RWTexture2DArray<float4>
#define MIP_COORD(x, y, slice) uint3(x, y, slice)
#else
#define MIP_TEXTURE RWTexture2D<float4>
#define MIP_COORD(x, y, slice) uint2(x, y)
#endif

float4 LoadSource(MIP_TEXTURE source, uint x, uint y, uint slice)
{
	float4 value = source[MIP_COORD(x, y, slice)];
	if (flags & kSrgb) value.rgb = OrgSrgbToLinear(value.rgb);
	return value;
}

// The source texels that destination texel `index` covers along one axis, and how much of each it covers.
// The footprint is [index * ratio, (index + 1) * ratio), with ratio = source / destination, at most 3 texels wide.
void Footprint(uint index, uint sourceSize, uint destinationSize, out uint first, out float weights[3])
{
	const float ratio = float(sourceSize) / float(destinationSize);
	const float start = index * ratio;
	const float end = start + ratio;
	first = uint(start);
	[unroll] for (uint i = 0; i < 3; ++i)
	{
		const float texelStart = float(first + i);
		const float covered = min(end, texelStart + 1.0) - max(start, texelStart);
		weights[i] = first + i < sourceSize ? max(covered, 0.0) / ratio : 0.0;
	}
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
	if (id.x >= destinationWidth || id.y >= destinationHeight) return;
	MIP_TEXTURE source = ResourceDescriptorHeap[sourceIndex];
	MIP_TEXTURE destination = ResourceDescriptorHeap[destinationIndex];

	uint firstX, firstY;
	float weightsX[3], weightsY[3];
	Footprint(id.x, sourceWidth, destinationWidth, firstX, weightsX);
	Footprint(id.y, sourceHeight, destinationHeight, firstY, weightsY);

	float4 sum = 0;
	[unroll] for (uint y = 0; y < 3; ++y)
	{
		[unroll] for (uint x = 0; x < 3; ++x)
		{
			const float weight = weightsX[x] * weightsY[y];
			if (weight > 0.0) sum += weight * LoadSource(source, firstX + x, firstY + y, id.z);
		}
	}
	if (flags & kSrgb) sum.rgb = OrgLinearToSrgb(saturate(sum.rgb));
	destination[MIP_COORD(id.x, id.y, id.z)] = sum;
}
