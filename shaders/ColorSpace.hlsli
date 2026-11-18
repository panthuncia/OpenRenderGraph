// sRGB transfer functions, for passes that read or write sRGB textures through UNORM views.
#ifndef ORG_COLOR_SPACE_HLSLI
#define ORG_COLOR_SPACE_HLSLI

float3 OrgSrgbToLinear(float3 value)
{
	return select(value <= 0.04045, value / 12.92, pow((value + 0.055) / 1.055, 2.4));
}

float3 OrgLinearToSrgb(float3 value)
{
	return select(value <= 0.0031308, value * 12.92, 1.055 * pow(value, 1.0 / 2.4) - 0.055);
}

#endif
